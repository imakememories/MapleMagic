"""Policy/value network: a transformer over observation tokens; each choice
attends over them and is scored against a summary token, which the value head reads."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from . import ACTION_VOCAB, HIST_VOCAB, NUM_CARDS, TOKEN_VOCAB


class FieldEmbedding(nn.Module):
    def __init__(self, vocab: list[int], d: int):
        super().__init__()
        self.tables = nn.ModuleList(nn.Embedding(v, d) for v in vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., fields] int
        out = self.tables[0](x[..., 0])
        for i in range(1, len(self.tables)):
            out = out + self.tables[i](x[..., i])
        return out


class Net(nn.Module):
    def __init__(self, d: int = 192, layers: int = 4, heads: int = 4):
        super().__init__()
        self.config = dict(d=d, layers=layers, heads=heads)
        self.tok = FieldEmbedding(TOKEN_VOCAB, d)
        self.act = FieldEmbedding(ACTION_VOCAB, d)
        self.summary = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.act_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.act_norm = nn.LayerNorm(d)
        self.policy = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.value = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1), nn.Tanh())

    def forward(self, tok, tok_len, act, act_len):
        """tok [B,N,F] long, tok_len [B], act [B,A,G] long, act_len [B].
        Returns (logits [B,A] with padding at -inf, value [B])."""
        B, N, _ = tok.shape
        A = act.shape[1]
        ar_n = torch.arange(N + 1, device=tok.device)
        # Summary token at position 0 is never masked.
        tok_pad = ar_n[None, :] > tok_len[:, None]
        x = torch.cat([self.summary.expand(B, 1, -1), self.tok(tok)], dim=1)
        h = self.norm(self.encoder(x, src_key_padding_mask=tok_pad))
        s = h[:, 0]
        a = self.act(act)
        ctx, _ = self.act_attn(a, h, h, key_padding_mask=tok_pad, need_weights=False)
        a = self.act_norm(a + ctx)
        logits = self.policy(torch.cat([a, s[:, None, :].expand(B, A, -1)], dim=-1)).squeeze(-1)
        act_pad = torch.arange(A, device=act.device)[None, :] >= act_len[:, None]
        logits = logits.masked_fill(act_pad, float("-inf"))
        return logits, self.value(s).squeeze(-1)


class Siamese(nn.Module):
    """Embeds what a player knows (observation and history) and candidate opponent
    hands; a hand close to the knowledge is a likely one."""

    def __init__(self, d: int = 96, layers: int = 2, heads: int = 4, emb: int = 64):
        super().__init__()
        self.config = dict(d=d, layers=layers, heads=heads, emb=emb)
        self.tok = FieldEmbedding(TOKEN_VOCAB, d)
        self.hist = FieldEmbedding(HIST_VOCAB, d)
        self.segment = nn.Embedding(2, d)
        self.summary = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.anchor_head = nn.Linear(d, emb)
        self.card = nn.Embedding(NUM_CARDS + 1, d, padding_idx=0)
        self.hand_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, emb))

    def anchor(self, tok, tok_len, hist, hist_len):
        B, N, H = tok.shape[0], tok.shape[1], hist.shape[1]
        x = torch.cat([
            self.summary.expand(B, 1, -1),
            self.tok(tok) + self.segment.weight[0],
            self.hist(hist) + self.segment.weight[1],
        ], dim=1)
        pad = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=tok.device),
            torch.arange(N, device=tok.device)[None, :] >= tok_len[:, None],
            torch.arange(H, device=tok.device)[None, :] >= hist_len[:, None],
        ], dim=1)
        return self.anchor_head(self.norm(self.encoder(x, src_key_padding_mask=pad))[:, 0])

    def hands(self, hands):
        """hands [..., H]: card definition + 1, 0 for padding."""
        return self.hand_head(self.card(hands).sum(-2))

    @staticmethod
    def distance(anchor, hands):
        return (hands - anchor.unsqueeze(-2)).norm(dim=-1)


def siamese_scorer(snet: Siamese, device, rows: int = 1024):
    """Scores a world batch (tokens, token_len, hist, hist_len, hands, hand_len) for SelfPlay.feed."""
    @torch.no_grad()
    def score(worlds):
        tok, tl, hist, hl, hands, _ = worlds
        snet.eval()
        out = np.zeros(hands.shape[:2], np.float32)
        t = lambda x: torch.from_numpy(x.astype(np.int64)).to(device)
        for at in range(0, len(tl), rows):
            r = slice(at, at + rows)
            a = snet.anchor(t(tok[r]), t(tl[r]), t(hist[r]), t(hl[r]))
            out[r] = Siamese.distance(a, snet.hands(t(hands[r]))).float().cpu().numpy()
        return out
    return score


def save_siamese(snet: Siamese, path: str, extra: dict | None = None):
    torch.save({"config": snet.config, "state": snet.state_dict(), **(extra or {})}, path)


def grow_tables(state: dict, model: nn.Module) -> dict:
    """Card embeddings grow as cards are appended to the pool; a checkpoint keeps its rows."""
    target = model.state_dict()
    for k, v in state.items():
        t = target.get(k)
        if t is not None and v.dim() == 2 and t.dim() == 2 and v.shape[1] == t.shape[1] and v.shape[0] < t.shape[0]:
            grown = t.clone()
            grown[: v.shape[0]] = v
            state[k] = grown
    return state


def load_siamese(path: str, device) -> Siamese:
    ck = torch.load(path, map_location=device, weights_only=False)
    snet = Siamese(**ck["config"]).to(device)
    snet.load_state_dict(grow_tables(ck["state"], snet))
    return snet


def to_tensors(batch, device):
    tok, tok_len, act, act_len = batch[:4]
    return (
        torch.from_numpy(tok.astype(np.int64)).to(device, non_blocking=True),
        torch.from_numpy(tok_len.astype(np.int64)).to(device, non_blocking=True),
        torch.from_numpy(act.astype(np.int64)).to(device, non_blocking=True),
        torch.from_numpy(act_len.astype(np.int64)).to(device, non_blocking=True),
    )


@torch.no_grad()
def evaluate(net: Net, tok, tok_len, act, act_len, device, amp_dtype=torch.float16):
    """Priors (softmax over legal choices) and values as numpy float32."""
    net.eval()
    t = to_tensors((tok, tok_len, act, act_len), device)
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
        logits, value = net(*t)
    priors = torch.softmax(logits.float(), dim=-1)
    return priors.cpu().numpy(), value.float().cpu().numpy()


def available_commit_bytes() -> float:
    """Free system commit on Windows, which GPU allocations are charged to; infinite elsewhere."""
    if os.name != "nt":
        return math.inf
    import ctypes

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
            (n, ctypes.c_ulonglong) for n in ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
                                              "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]

    st = MemoryStatusEx()
    st.dwLength = ctypes.sizeof(MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
        return math.inf
    return float(st.ullAvailPageFile)


@dataclass
class EvalPlan:
    """At most `max_batch` rows per forward pass; empty the cache above `flush_at` bytes."""
    max_batch: int
    flush_at: float
    bytes_per_row: float = 0.0
    budget: float = 0.0

    def describe(self) -> str:
        gb = 1 << 30
        return (f"eval plan: {self.max_batch} rows per pass, cache flushed above {self.flush_at / gb:.1f} GB "
                f"({self.bytes_per_row / (1 << 20):.2f} MB per row, budget {self.budget / gb:.1f} GB)")


def plan_eval(net: Net, device, tok_width: int, act_width: int, max_batch: int = 0, share: float = 0.25) -> EvalPlan:
    """Size forward passes to half of `share` of the free GPU memory or commit,
    whichever is smaller, from one measured pass at the widest shapes."""
    if device.type != "cuda":
        return EvalPlan(max_batch or 2048, math.inf)
    rows = 256
    tok = np.zeros((rows, tok_width, len(TOKEN_VOCAB)), np.uint16)
    act = np.zeros((rows, act_width, len(ACTION_VOCAB)), np.uint16)
    lens = (np.full(rows, tok_width, np.int32), np.full(rows, act_width, np.int32))
    # Warm up first: the first pass also allocates one-time workspaces.
    evaluate(net, tok, lens[0], act, lens[1], device)
    torch.cuda.synchronize(device)
    base = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    evaluate(net, tok, lens[0], act, lens[1], device)
    per_row = max(torch.cuda.max_memory_allocated(device) - base, 1) / rows
    torch.cuda.empty_cache()
    free_gpu, _ = torch.cuda.mem_get_info(device)
    budget = share * min(free_gpu, available_commit_bytes())
    if not max_batch:
        max_batch = int(budget / (2 * per_row)) // 256 * 256
        max_batch = min(max(max_batch, 256), 16384)
    return EvalPlan(max_batch, max(budget, 2 * max_batch * per_row), per_row, budget)


def losses(net: Net, batch, device, value_weight: float = 1.0, split: int | None = None):
    """With `split`, the first `split` rows train only the policy and the rest only the value."""
    tok, tok_len, act, act_len, policy, z = batch
    t = to_tensors((tok, tok_len, act, act_len), device)
    pi = torch.from_numpy(policy).to(device)
    zt = torch.from_numpy(z).to(device)
    logits, value = net(*t)
    logp = F.log_softmax(logits.float(), dim=-1).masked_fill(pi == 0, 0.0)
    if split is not None:
        policy_loss = -(pi[:split] * logp[:split]).sum(-1).mean()
        value_loss = F.mse_loss(value.float()[split:], zt[split:])
        return policy_loss + value_weight * value_loss, policy_loss.detach(), value_loss.detach()
    policy_loss = -(pi * logp).sum(-1).mean()
    value_loss = F.mse_loss(value.float(), zt)
    return policy_loss + value_weight * value_loss, policy_loss.detach(), value_loss.detach()


def save(net: Net, path: str, extra: dict | None = None):
    torch.save({"config": net.config, "state": net.state_dict(), **(extra or {})}, path)


def load_meta(path: str) -> dict:
    """Everything a checkpoint stores besides the weights (gen, search...)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return {k: v for k, v in ck.items() if k not in ("state", "opt")}


def load(path: str, device) -> Net:
    ck = torch.load(path, map_location=device, weights_only=False)
    net = Net(**ck["config"]).to(device)
    net.load_state_dict(grow_tables(ck["state"], net))
    return net
