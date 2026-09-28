"""Policy/value network.

Observation tokens (one per visible object, plus global scalars) are embedded
field by field, summed, and run through a transformer encoder with a learned
summary token. Each legal choice is embedded the same way, attends over the
encoded tokens (so "target this creature" can look at that creature), and is
scored against the summary. The value head reads the summary.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from . import ACTION_VOCAB, TOKEN_VOCAB


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


def losses(net: Net, batch, device, value_weight: float = 1.0):
    tok, tok_len, act, act_len, policy, z = batch
    t = to_tensors((tok, tok_len, act, act_len), device)
    pi = torch.from_numpy(policy).to(device)
    zt = torch.from_numpy(z).to(device)
    logits, value = net(*t)
    logp = F.log_softmax(logits.float(), dim=-1).masked_fill(pi == 0, 0.0)
    policy_loss = -(pi * logp).sum(-1).mean()
    value_loss = F.mse_loss(value.float(), zt)
    return policy_loss + value_weight * value_loss, policy_loss.detach(), value_loss.detach()


def save(net: Net, path: str, extra: dict | None = None):
    torch.save({"config": net.config, "state": net.state_dict(), **(extra or {})}, path)


def load(path: str, device) -> Net:
    ck = torch.load(path, map_location=device, weights_only=False)
    net = Net(**ck["config"]).to(device)
    net.load_state_dict(ck["state"])
    return net
