"""Learned world sampling: positions from frozen-network self-play, a retention
baseline, the Siamese sampler, and offline metrics against uniform and oracle."""
from __future__ import annotations

import json
import math
import time

import numpy as np
import torch

from . import CARD_KINDS, MAX_HIST, MAX_TOKENS, NUM_CARDS, RETENTION_BUCKETS, SelfPlay
from .model import EvalPlan, Siamese, load, load_meta, plan_eval, save_siamese

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
            r = order[at:at + batch]
            pos, neg, ok = [], [], []
            for i in r:
                n = int(data["hand_n"][i])
                p = positives.get(int(i), np.sort(data["pool"][i][:n]))
                q = draw_hands(data["pool"][i], n, 1, rng)[0]
                pos.append(p)
                neg.append(q)
                ok.append(not np.array_equal(p, q))
            ok = np.array(ok)
            if not ok.any():
                continue
            r = r[ok]
            a = snet.anchor(*tensors(data, r, device))
            hp = snet.hands(torch.from_numpy(pad_hands([h for h, k in zip(pos, ok) if k]).astype(np.int64)).to(device))
            hn = snet.hands(torch.from_numpy(pad_hands([h for h, k in zip(neg, ok) if k]).astype(np.int64)).to(device))
            loss = torch.relu(1 + (hp - a).norm(dim=-1) - (hn - a).norm(dim=-1)).mean()
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
