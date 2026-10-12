"""Learned world sampling: positions from frozen-network self-play, a retention
baseline, the Siamese sampler, and offline metrics against uniform and oracle."""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch

from . import (CARD_KINDS, HIST_FIELDS, LEGACY_TOKEN_FIELDS, MAX_HIST, MAX_TOKENS, NUM_CARDS, RETENTION_BUCKETS,
               SelfPlay)
from .model import EvalPlan, Siamese, grow_tables, load, load_meta, plan_eval, save_siamese, siamese_scorer

KINDS = np.frombuffer(bytes(CARD_KINDS), np.uint8)
CANDIDATES = 50
KEEP = 5


def collect(checkpoint: str, cfg, games: int, sims: int, device, keep: float = 0.25, seed: int = 0) -> dict:
    """Self-play positions of a frozen network, as it plays in matches (no noise or
    temperature). Keeps a `keep` share of each game's decisions."""
    from .train import drive
    net = load(checkpoint, device)
    s = {**cfg.search_settings(), **(load_meta(checkpoint).get("search") or {})}
    sp = SelfPlay(
        min(games, cfg.parallel), cfg.pairing_list(), sims=sims, leaves_per_step=cfg.leaves_per_step, temp_decisions=0,
        root_noise=0.0, record=True, max_games=games, seed=seed, maple_worlds=(s["maple_worlds"],) * 2,
        maple_select=s["maple_select"], perfect_obs=(s["perfect_obs"],) * 2, pimc_worlds=(s.get("pimc_worlds", 0),) * 2,
    )
    rng = np.random.default_rng(seed)
    parts = []

    def add(smp):
        tok, tl = smp[0], smp[1]
        hist, hl, pool, pl, hand_n, game, turn = smp[7][:7]
        rows = np.nonzero(rng.random(len(tl)) < keep)[0]
        parts.append(dict(tok=tok[rows, :MAX_TOKENS], tok_len=tl[rows], hist=hist[rows, :MAX_HIST], hist_len=hl[rows],
                          pool=pool[rows], pool_len=pl[rows], hand_n=hand_n[rows], game=game[rows], turn=turn[rows]))

    plan = plan_eval(net, device, MAX_TOKENS, 96, cfg.max_eval_batch)
    t0 = time.time()
    drive(sp, {0: net}, device, on_samples=add, plan=plan)
    data = {}
    for k in parts[0]:
        if parts[0][k].ndim == 1:
            data[k] = np.concatenate([p[k] for p in parts])
        else:
            width = max(p[k].shape[1] for p in parts)
            data[k] = np.concatenate([np.pad(p[k], [(0, 0), (0, width - p[k].shape[1])] + [(0, 0)] * (p[k].ndim - 2))
                                      for p in parts])
    data["seconds"] = np.array(time.time() - t0)
    return data


def split(data: dict, seed: int = 0) -> dict:
    """80/10/10 by game, so no game's positions are in two splits."""
    games = np.unique(data["game"])
    r = np.random.default_rng(seed).random(len(games))
    which = dict(zip(games.tolist(), np.where(r < 0.8, 0, np.where(r < 0.9, 1, 2)).tolist()))
    tag = np.array([which[g] for g in data["game"].tolist()])
    return {name: np.nonzero(tag == i)[0] for i, name in enumerate(["train", "val", "test"])}


def draw_hands(pool_row: np.ndarray, n: int, k: int, rng) -> np.ndarray:
    """k uniform hands of size n from one pool row (card + 1 values), each sorted."""
    p = pool_row[pool_row > 0]
    idx = np.argsort(rng.random((k, len(p))), axis=1)[:, :n]
    return np.sort(p[idx], axis=1)


def candidates(data: dict, i: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """The true hand and distinct uniform candidate hands, as the search draws them."""
    n = int(data["hand_n"][i])
    pool = data["pool"][i]
    truth = np.sort(pool[:n])
    cands = np.unique(draw_hands(pool, n, CANDIDATES, rng), axis=0)
    return truth, cands


def used_rows(data: dict, rows: np.ndarray, seed: int = 0) -> np.ndarray:
    """Positions where a search would ask for scores: more than KEEP distinct candidate hands."""
    rng = np.random.default_rng(seed)
    keep = [i for i in rows if data["hand_n"][i] > 0 and len(candidates(data, i, rng)[1]) > KEEP]
    return np.array(keep, np.int64)


def fit_retention(data: dict, rows: np.ndarray, prior: float = 1.0) -> np.ndarray:
    """Log-odds that an unseen card of each kind is in hand, by turn bucket."""
    inside = np.zeros((RETENTION_BUCKETS, NUM_CARDS))
    total = np.zeros((RETENTION_BUCKETS, NUM_CARDS))
    for i in rows:
        b = min(int(data["turn"][i]), RETENTION_BUCKETS - 1)
        n, pool = int(data["hand_n"][i]), data["pool"][i]
        cards = pool[pool > 0] - 1
        np.add.at(total[b], cards, 1)
        np.add.at(inside[b], cards[:n], 1)
    rate = inside.sum() / max(total.sum(), 1)
    p = (inside + prior * rate) / (total + prior)
    return np.log(p / (1 - p)).astype(np.float32).reshape(-1)


def tensors(data: dict, rows: np.ndarray, device):
    t = lambda x: torch.from_numpy(np.ascontiguousarray(x).astype(np.int64)).to(device)
    tl, hl = data["tok_len"][rows], data["hist_len"][rows]
    tw, hw = max(int(tl.max()), 1), max(int(hl.max()), 1)
    return t(data["tok"][rows, :tw]), t(tl), t(data["hist"][rows, :hw]), t(hl)


def pad_hands(hands: list[np.ndarray]) -> np.ndarray:
    k = max(max((h.shape[-1] for h in hands), default=1), 1)
    return np.stack([np.pad(h, [(0, 0)] * (h.ndim - 1) + [(0, k - h.shape[-1])]) for h in hands])


def triplet_loss(snet: Siamese, data: dict, rows: np.ndarray, rng, device, positives: dict | None = None):
    """The paper's eq. 4 triplet term on `rows`: the true hand (or `positives[row]`) against one
    uniform hand, margin 1. None when every negative equals its positive."""
    positives = positives or {}
    pos, neg, ok = [], [], []
    for i in rows:
        n = int(data["hand_n"][i])
        p = positives.get(int(i), np.sort(data["pool"][i][:n]))
        q = draw_hands(data["pool"][i], n, 1, rng)[0]
        pos.append(p)
        neg.append(q)
        ok.append(not np.array_equal(p, q))
    ok = np.array(ok)
    if not ok.any():
        return None
    r = np.asarray(rows)[ok]
    a = snet.anchor(*tensors(data, r, device))
    hp = snet.hands(torch.from_numpy(pad_hands([h for h, k in zip(pos, ok) if k]).astype(np.int64)).to(device))
    hn = snet.hands(torch.from_numpy(pad_hands([h for h, k in zip(neg, ok) if k]).astype(np.int64)).to(device))
    return torch.relu(1 + (hp - a).norm(dim=-1) - (hn - a).norm(dim=-1)).mean()


def train(data: dict, splits: dict, device, seed: int = 0, shuffled: bool = False, batch: int = 256,
          lr: float = 1e-3, max_epochs: int = 20, patience: int = 2, log=print) -> tuple[Siamese, dict]:
    """Triplet training (the paper's eq. 4 term) with early stopping on validation rank.
    `shuffled` replaces each true hand with a random one: a control that must learn nothing."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    snet = Siamese().to(device)
    opt = torch.optim.AdamW(snet.parameters(), lr=lr, weight_decay=1e-4)
    rows = used_rows(data, splits["train"], seed)
    val = used_rows(data, splits["val"], seed + 1)
    positives = {}
    if shuffled:
        for i in rows:
            positives[int(i)] = draw_hands(data["pool"][i], int(data["hand_n"][i]), 1, rng)[0]
    best, best_state, stale, history = math.inf, None, 0, []
    for epoch in range(max_epochs):
        snet.train()
        order = rng.permutation(rows)
        losses = []
        for at in range(0, len(order), batch):
            loss = triplet_loss(snet, data, order[at:at + batch], rng, device, positives)
            if loss is None:
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(snet.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        m = metrics(data, val, device, snet=snet, seed=seed)
        history.append({"epoch": epoch, "loss": round(float(np.mean(losses)), 4), "val_rank": m["siamese"]["rank"]})
        log(json.dumps(history[-1]))
        if m["siamese"]["rank"] < best - 1e-4:
            best, stale = m["siamese"]["rank"], 0
            best_state = {k: v.detach().clone() for k, v in snet.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    snet.load_state_dict(best_state)
    return snet, {"epochs": history, "train_rows": len(rows), "val_rows": len(val)}


@torch.no_grad()
def metrics(data: dict, rows: np.ndarray, device, snet: Siamese | None = None, retention: np.ndarray | None = None,
            seed: int = 0) -> dict:
    """Per position, against the true hand and the search's own uniform candidates:
    rank (share of candidates scored better than the truth, 0.5 for uniform), top-5
    recall of the truth, and recall of the truth's cards in the 5 chosen hands."""
    rng = np.random.default_rng(seed)
    if snet is not None:
        snet.eval()
    out = {name: {"rank": [], "top5": [], "cards": [], "nonland": [], "instant": []}
           for name in ["uniform", "oracle"] + (["siamese"] if snet is not None else []) + (["retention"] if retention is not None else [])}
    ret = None if retention is None else retention.reshape(RETENTION_BUCKETS, NUM_CARDS)
    for at in range(0, len(rows), 256):
        r = rows[at:at + 256]
        items = [candidates(data, i, rng) for i in r]
        dist = None
        if snet is not None:
            a = snet.anchor(*tensors(data, r, device))
            hands = [np.vstack([t[None], c]) for t, c in items]
            m = max(len(h) for h in hands)
            k = max(h.shape[1] for h in hands)
            flat = np.zeros((len(r), m, max(k, 1)), np.int64)
            for j, h in enumerate(hands):
                flat[j, :len(h), :h.shape[1]] = h
            dist = Siamese.distance(a, snet.hands(torch.from_numpy(flat).to(device))).float().cpu().numpy()
        for j, (i, (truth, cands)) in enumerate(zip(r, items)):
            scores = {"uniform": rng.random(len(cands) + 1), "oracle": np.r_[-1.0, np.zeros(len(cands))]}
            if dist is not None:
                scores["siamese"] = dist[j, :len(cands) + 1]
            if ret is not None:
                b = min(int(data["turn"][i]), RETENTION_BUCKETS - 1)
                ll = lambda h: -ret[b, h - 1].sum()
                scores["retention"] = np.array([ll(truth)] + [ll(c) for c in cands])
            for name, sc in scores.items():
                o = out[name]
                o["rank"].append(float((sc[1:] < sc[0]).mean()) if len(cands) else 0.0)
                o["top5"].append(float((sc[1:] < sc[0]).sum() < KEEP))
                chosen = cands[np.argsort(sc[1:])[:KEEP]] if name != "oracle" else np.repeat(truth[None], KEEP, 0)
                for key, mask in (("cards", np.ones(len(truth), bool)), ("nonland", KINDS[truth - 1] != 0),
                                  ("instant", KINDS[truth - 1] == 1)):
                    if mask.any():
                        o[key].append(float(np.mean([(chosen == c).any(axis=1).mean() for c in truth[mask]])))
    return {name: {k: round(float(np.mean(v)), 4) if v else None for k, v in o.items()} | {"n": len(o["rank"])}
            for name, o in out.items()}


def save(snet: Siamese, retention: np.ndarray, path: str, extra: dict | None = None):
    save_siamese(snet, path, {"retention": retention, **(extra or {})})


POOL_WIDTH = 128


class SamplerReplay:
    """Ring buffer of sampler rows: anchor tokens (the Siamese's fields), history, the
    opponent's unseen cards (card + 1, hand first) and its hand size."""

    FIELDS = ("tok", "tok_len", "hist", "hist_len", "pool", "hand_n", "game", "turn")

    def __init__(self, cap: int, seed: int = 0):
        self.cap = cap
        self.tok = np.zeros((cap, MAX_TOKENS, LEGACY_TOKEN_FIELDS), np.uint16)
        self.tok_len = np.zeros(cap, np.int32)
        self.hist = np.zeros((cap, MAX_HIST, HIST_FIELDS), np.uint16)
        self.hist_len = np.zeros(cap, np.int32)
        self.pool = np.zeros((cap, POOL_WIDTH), np.uint16)
        self.hand_n = np.zeros(cap, np.uint8)
        self.game = np.zeros(cap, np.int64)
        self.turn = np.zeros(cap, np.int32)
        self.n = 0
        self.pos = 0
        self.rng = np.random.default_rng(seed)

    def add(self, samples, per_game: int, rng=None) -> np.ndarray:
        """Up to `per_game` random rows per game among those where the opponent holds cards.
        Returns the buffer rows written. Seen-hand samples carry hidden anchor tokens as
        inner element 10, since their own tokens show the hand."""
        rng = self.rng if rng is None else rng
        inner = samples[7]
        tok = inner[10] if len(inner) > 10 and inner[10] is not None else samples[0]
        tok_len = samples[1]
        hist, hist_len, pool, _, hand_n, game, turn = inner[:7]
        assert pool.shape[1] <= POOL_WIDTH, f"a pool of {pool.shape[1]} cards doesn't fit {POOL_WIDTH}"
        held = np.flatnonzero(hand_n > 0)
        rows = []
        for g in np.unique(game[held]):
            r = held[game[held] == g]
            rows.append(np.sort(rng.choice(r, min(per_game, len(r)), replace=False)))
        rows = np.concatenate(rows)[-self.cap:] if rows else np.zeros(0, np.int64)
        at = (self.pos + np.arange(len(rows))) % self.cap
        tw, hw = min(tok.shape[1], MAX_TOKENS), min(hist.shape[1], MAX_HIST)
        self.tok[at] = 0
        self.tok[at, :tw] = tok[rows, :tw, :LEGACY_TOKEN_FIELDS]
        self.tok_len[at] = np.minimum(tok_len[rows], MAX_TOKENS)
        self.hist[at] = 0
        self.hist[at, :hw] = hist[rows, :hw]
        self.hist_len[at] = np.minimum(hist_len[rows], MAX_HIST)
        self.pool[at] = 0
        self.pool[at, :pool.shape[1]] = pool[rows]
        self.hand_n[at], self.game[at], self.turn[at] = hand_n[rows], game[rows], turn[rows]
        self.pos = int((self.pos + len(rows)) % self.cap)
        self.n = min(self.n + len(rows), self.cap)
        return at

    def view(self, rows: np.ndarray | None = None) -> dict:
        """The arrays `metrics`, `used_rows` and `tensors` read: the whole buffer's, or only `rows`'."""
        if rows is None:
            return {f: getattr(self, f)[:self.n] for f in self.FIELDS}
        return {f: getattr(self, f)[rows] for f in self.FIELDS}

    def save(self, path: str):
        tmp = path + ".tmp.npz"
        np.savez(tmp, n=self.n, pos=self.pos, **{f: getattr(self, f) for f in self.FIELDS})
        os.replace(tmp, path)

    def load(self, path: str) -> bool:
        import zipfile

        from .train import npy_header, read_npy_into
        with np.load(path) as f, zipfile.ZipFile(path) as z:
            shapes = tuple(tuple(npy_header(z, k)[0]) for k in ("tok", "hist", "pool"))
            want = (self.tok.shape, self.hist.shape, self.pool.shape)
            if shapes != want:
                print(f"not loading the sampler buffer {path}: its shapes {shapes} differ from {want}")
                return False
            for k in self.FIELDS:
                read_npy_into(z, k, getattr(self, k))
            self.n, self.pos = int(f["n"]), int(f["pos"])
        return True


class SamplerTrainer:
    """The Siamese trained beside the network on its self-play (eq. 4's triplet term),
    with its own AdamW and buffer."""

    def __init__(self, cfg, device, seed: int = 0):
        self.cfg, self.device = cfg, device
        # Its own seed, so adding the sampler leaves the network's initialization alone.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.snet = Siamese().to(device)
        self.opt = torch.optim.AdamW(self.snet.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.buf = SamplerReplay(cfg.sampler_buffer, seed)
        self.rng = np.random.default_rng(seed)
        self.scorings, self.score_s = 0, 0.0

    def add(self, samples) -> np.ndarray:
        return self.buf.add(samples, self.cfg.sampler_rows_per_game, self.rng)

    def train(self, steps: int, lr_at) -> tuple[float | None, float]:
        """`steps` batches of random buffer rows, each with a fresh uniform negative, at
        learning rate `lr_at(i)`. Returns (mean loss, seconds)."""
        t0 = time.time()
        if self.buf.n == 0:
            return None, 0.0
        self.snet.train()
        data = self.buf.view()
        losses = []
        for i in range(steps):
            for g in self.opt.param_groups:
                g["lr"] = lr_at(i)
            rows = self.rng.integers(0, self.buf.n, self.cfg.sampler_batch)
            loss = triplet_loss(self.snet, data, rows, self.rng, self.device)
            if loss is None:
                continue
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.snet.parameters(), 1.0)
            self.opt.step()
            losses.append(loss.item())
        return (float(np.mean(losses)) if losses else None), time.time() - t0

    def heldout(self, rows: np.ndarray, fresh: np.ndarray | None = None) -> dict:
        """The sampler on buffer `rows` it hasn't trained on, against uniform worlds and a
        retention table fit on the buffer's rows outside `fresh` (default: `rows`)."""
        data = self.buf.view()
        used = used_rows(data, rows)
        if not len(used):
            return {}
        older = np.setdiff1d(np.arange(self.buf.n), rows if fresh is None else fresh)
        retention = fit_retention(data, older) if len(older) else None
        m = metrics(data, used, self.device, snet=self.snet, retention=retention)
        s = m["siamese"]
        out = {"heldout_sampler_rank": s["rank"], "heldout_sampler_top5": s["top5"],
               "heldout_sampler_cards": s["cards"], "heldout_sampler_nonland": s["nonland"]}
        if retention is not None:
            out["heldout_retention_rank"] = m["retention"]["rank"]
        out["heldout_uniform_rank"] = m["uniform"]["rank"]
        return out

    def scorer(self):
        """`siamese_scorer`, counting the world requests it scores and its seconds."""
        score = siamese_scorer(self.snet, self.device)

        def counted(worlds):
            t0 = time.time()
            out = score(worlds)
            self.scorings += len(out)
            self.score_s += time.time() - t0
            return out
        return counted

    def state(self) -> dict:
        return {"config": self.snet.config, "state": self.snet.state_dict()}

    def load(self, state: dict, opt_state: dict | None = None):
        self.snet = Siamese(**state["config"]).to(self.device)
        self.snet.load_state_dict(grow_tables(state["state"], self.snet))
        self.opt = torch.optim.AdamW(self.snet.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        if opt_state is not None:
            self.opt.load_state_dict(opt_state)
            for p in self.snet.parameters():
                for k, v in self.opt.state.get(p, {}).items():
                    if torch.is_tensor(v) and v.dim() == 2 and v.shape[0] < p.shape[0]:
                        self.opt.state[p][k] = torch.cat([v, v.new_zeros(p.shape[0] - v.shape[0], v.shape[1])])
