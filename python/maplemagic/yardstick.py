"""Value heads scored on fresh games their run never trained on, beside the
hand-written heuristic's predictions of the same positions."""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from . import MAX_TOKENS, SelfPlay
from .model import load, load_meta, load_siamese, plan_eval, siamese_scorer, to_tensors
from .train import Config, drive

TURNS = [("1-3", 1, 3), ("4-7", 4, 7), ("8+", 8, 10**6)]


def run_config(checkpoint: str) -> Config:
    with open(os.path.join(os.path.dirname(checkpoint), "config.yaml")) as f:
        return Config(**yaml.safe_load(f))


def collect(checkpoint: str, games: int, seed: int, device) -> dict:
    """Self-play as the run trained (its sims, sampled opening decisions and root
    noise), with deals from `seed`. Every decision of both seats is kept."""
    cfg = run_config(checkpoint)
    meta = load_meta(checkpoint).get("search") or {}
    s = {**cfg.search_settings(), **meta}
    net = load(checkpoint, device)
    # Diagnostic samplers (oracle, retention) aren't how a run plays; only learned worlds are kept.
    sampler = "scored" if s.get("world_sampler") == "scored" else "uniform"
    sp = SelfPlay(
        min(games, cfg.parallel), cfg.pairing_list(), sims=cfg.sims, leaves_per_step=cfg.leaves_per_step,
        temp_decisions=cfg.temp_decisions, c_puct=cfg.c_puct, root_noise=cfg.root_noise,
        dirichlet_alpha=cfg.dirichlet_alpha, record=True, max_games=games, seed=seed,
        maple_worlds=(s["maple_worlds"],) * 2, maple_select=s["maple_select"], maple_resample=s["maple_resample"],
        perfect_obs=(s["perfect_obs"],) * 2, pimc_worlds=(s.get("pimc_worlds", 0),) * 2,
        value_mix=s.get("value_mix", 1.0), world_sampler=sampler, world_candidates=cfg.world_candidates,
        world_pool=cfg.world_pool, world_temp=cfg.world_temp, collisions=meta.get("collisions", False),
    )
    scorers = {0: siamese_scorer(load_siamese(checkpoint, device), device)} if sampler == "scored" else None
    parts = []
    drive(sp, {0: net}, device, on_samples=parts.append, plan=plan_eval(net, device, MAX_TOKENS, 96, cfg.max_eval_batch),
          scorers=scorers)
    pad = lambda xs: np.concatenate([np.pad(x, [(0, 0), (0, max(y.shape[1] for y in xs) - x.shape[1])]
                                            + [(0, 0)] * (x.ndim - 2)) for x in xs])
    cat = lambda xs: np.concatenate(xs)
    return {
        "tok": pad([p[0] for p in parts]), "tok_len": cat([p[1] for p in parts]),
        "act": pad([p[2] for p in parts]), "act_len": cat([p[3] for p in parts]),
        "pi": pad([p[4] for p in parts]), "z": cat([p[5] for p in parts]),
        "game": cat([p[7][5] for p in parts]), "turn": cat([p[7][6] for p in parts]),
        "heuristic": cat([p[7][7] for p in parts]), "perfect_obs": np.array(s["perfect_obs"]),
    }


def auc(score: np.ndarray, won: np.ndarray) -> float | None:
    """Probability that a won position scores above a lost one (ties count half)."""
    n1, n0 = int(won.sum()), int((~won).sum())
    if n1 == 0 or n0 == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    _, inv, counts = np.unique(score[order], return_inverse=True, return_counts=True)
    end = np.cumsum(counts)
    ranks = np.empty(len(score))
    ranks[order] = ((end - counts + 1 + end) / 2)[inv]
    return float((ranks[won].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


@torch.no_grad()
def predict(net, data: dict, device, rows: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Values and policy cross-entropy against the visit targets, per position."""
    net.eval()
    values, ce = [], []
    for at in range(0, len(data["z"]), rows):
        r = slice(at, at + rows)
        tl, al = data["tok_len"][r], data["act_len"][r]
        tw, aw = max(int(tl.max()), 1), max(int(al.max()), 1)
        logits, v = net(*to_tensors((data["tok"][r, :tw], tl, data["act"][r, :aw], al), device))
        pi = torch.from_numpy(data["pi"][r, :aw]).to(device)
        logp = F.log_softmax(logits.float(), dim=-1).masked_fill(pi == 0, 0.0)
        values.append(v.float().cpu().numpy())
        ce.append((-(pi * logp).sum(-1)).cpu().numpy())
    return np.concatenate(values), np.concatenate(ce)


def summary(pred: np.ndarray, data: dict) -> dict:
    z, turn = data["z"], data["turn"]
    decided = z != 0
    out = {"positions": int(len(z)), "games": int(len(np.unique(data["game"]))), "mse": round(float(np.mean((pred - z) ** 2)), 4),
           "auc": auc(pred[decided], z[decided] > 0)}
    for name, lo, hi in TURNS:
        m = decided & (turn >= lo) & (turn <= hi)
        out[f"auc_turn_{name}"] = auc(pred[m], z[m] > 0)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}
