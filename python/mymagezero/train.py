"""Generational AlphaZero loop: self-play -> replay buffer -> train -> evaluate.

Self-play runs in Rust (`SelfPlay`); this module only evaluates leaf batches
on the GPU and trains. Evaluation games pit the new network against the
built-in heuristic searcher and against the previous generation.
"""
from __future__ import annotations

import copy
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import yaml

from . import DECKS, MAX_TOKENS, SelfPlay
from .model import Net, evaluate, load, losses, save

MAX_ACTIONS = 96


@dataclass
class Config:
    name: str = "run"
    out_dir: str = "runs"
    # Deck pairings; "all" = every ordered pair of the pool.
    pairings: list | str = "all"
    generations: int = 50
    games_per_gen: int = 512
    parallel: int = 512
    sims: int = 96
    leaves_per_step: int = 8
    temp_decisions: int = 40
    c_puct: float = 1.0
    root_noise: float = 0.25
    dirichlet_alpha: float = 0.3
    # Network
    d_model: int = 192
    layers: int = 4
    heads: int = 4
    # Training
    buffer_size: int = 400_000
    batch_size: int = 512
    reuse: float = 4.0  # expected times each new sample is trained on
    lr: float = 1e-3
    weight_decay: float = 1e-4
    value_weight: float = 1.0
    # Evaluation
    eval_every: int = 1
    eval_games: int = 128
    eval_sims: int = 96
    heuristic_sims: int = 200
    seed: int = 0
    extra: dict = field(default_factory=dict)

    @staticmethod
    def from_yaml(path: str) -> "Config":
        with open(path) as f:
            return Config(**(yaml.safe_load(f) or {}))

    def pairing_list(self) -> list[tuple[str, str]]:
        if self.pairings == "all":
            return [(a, b) for a in DECKS for b in DECKS]
        return [tuple(p) for p in self.pairings]


class Replay:
    """Fixed-width ring buffer of samples (padded to MAX_TOKENS / MAX_ACTIONS)."""

    def __init__(self, cap: int, tok_fields: int, act_fields: int):
        self.cap = cap
        self.tok = np.zeros((cap, MAX_TOKENS, tok_fields), np.uint16)
        self.tok_len = np.zeros(cap, np.int32)
        self.act = np.zeros((cap, MAX_ACTIONS, act_fields), np.uint16)
        self.act_len = np.zeros(cap, np.int32)
        self.pi = np.zeros((cap, MAX_ACTIONS), np.float32)
        self.z = np.zeros(cap, np.float32)
        self.n = 0
        self.pos = 0

    def add(self, samples):
        tok, tok_len, act, act_len, pi, z, _rv = samples
        k = len(tok_len)
        tw, aw = min(tok.shape[1], MAX_TOKENS), min(act.shape[1], MAX_ACTIONS)
        for i in range(k):
            j = self.pos
            self.tok[j] = 0
            self.tok[j, :tw] = tok[i, :tw]
            self.tok_len[j] = min(tok_len[i], MAX_TOKENS)
            self.act[j] = 0
            self.act[j, :aw] = act[i, :aw]
            al = min(act_len[i], MAX_ACTIONS)
            self.act_len[j] = al
            self.pi[j] = 0
            p = pi[i, :al]
            self.pi[j, :al] = p / max(p.sum(), 1e-6)
            self.z[j] = z[i]
            self.pos = (self.pos + 1) % self.cap
            self.n = min(self.n + 1, self.cap)

    def sample(self, bs: int, rng: np.random.Generator):
        idx = rng.integers(0, self.n, bs)
        tl = self.tok_len[idx]
        al = self.act_len[idx]
        tw, aw = max(int(tl.max()), 1), max(int(al.max()), 1)
        return (self.tok[idx, :tw], tl, self.act[idx, :aw], al, self.pi[idx, :aw], self.z[idx])


def drive(sp: SelfPlay, nets: dict, device, on_samples=None) -> dict:
    """Run a SelfPlay to completion, evaluating leaves with `nets[model_id]`."""
    evals = 0
    t0 = time.time()
    while True:
        req = sp.gather()
        if req is None:
            break
        tok, tl, act, al, model = req
        priors = np.zeros(act.shape[:2], np.float32)
        values = np.zeros(len(tl), np.float32)
        for m in np.unique(model):
            rows = np.nonzero(model == m)[0]
            # Trim padding for this subset.
            tw = max(int(tl[rows].max()), 1)
            aw = max(int(al[rows].max()), 1)
            p, v = evaluate(nets[int(m)], tok[rows, :tw], tl[rows], act[rows, :aw], al[rows], device)
            priors[rows, :aw] = p
            values[rows] = v
        sp.feed(priors, values)
        evals += len(tl)
        if on_samples is not None:
            s = sp.take_samples()
            if s is not None:
                on_samples(s)
    return {"evals": evals, "seconds": time.time() - t0}


def match(cfg: Config, net_a, opponent, games: int, device, seed: int) -> dict:
    """Score of net_a (model 0) against `opponent` (a Net as model 1, or
    "heuristic"/"random"), both seats and all pairings."""
    nets = {0: net_a}
    opp_id = 1
    if isinstance(opponent, str):
        opp_id = opponent
    else:
        nets[1] = opponent
    sp = SelfPlay(
        min(games, cfg.parallel), cfg.pairing_list(), sims=cfg.eval_sims, leaves_per_step=cfg.leaves_per_step,
        temp_decisions=0, c_puct=cfg.c_puct, root_noise=0.0, seat_models=(0, opp_id), record=False,
        max_games=games, heuristic_sims=cfg.heuristic_sims, seed=seed,
    )
    stats = drive(sp, nets, device)
    w = l = d = 0
    for r in sp.take_results():
        if r["winner"] is None:
            d += 1
        elif r["models"][r["winner"]] == 0:
            w += 1
        else:
            l += 1
    n = max(w + l + d, 1)
    score = (w + 0.5 * d) / n
    return {"w": w, "l": l, "d": d, "score": score, "ci95": 1.96 * math.sqrt(score * (1 - score) / n), **stats}


def run(cfg: Config, resume: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = os.path.join(cfg.out_dir, cfg.name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.yaml"), "w") as f:
        yaml.safe_dump(asdict(cfg), f)
    log_path = os.path.join(out, "log.jsonl")
    latest = os.path.join(out, "latest.pt")

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    start_gen = 0
    if resume and os.path.exists(latest):
        net = load(latest, device)
        start_gen = torch.load(latest, map_location="cpu", weights_only=False).get("gen", -1) + 1
        print(f"resumed from {latest} at generation {start_gen}")
    else:
        net = Net(cfg.d_model, cfg.layers, cfg.heads).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    from . import ACTION_FIELDS, TOKEN_FIELDS
    replay = Replay(cfg.buffer_size, TOKEN_FIELDS, ACTION_FIELDS)
    print(f"device {device}, params {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M, "
          f"{len(cfg.pairing_list())} pairings")

    for gen in range(start_gen, cfg.generations):
        # --- self-play
        prev = copy.deepcopy(net).eval()
        new = [0]

        def add(s):
            replay.add(s)
            new[0] += len(s[1])

        sp = SelfPlay(
            cfg.parallel, cfg.pairing_list(), sims=cfg.sims, leaves_per_step=cfg.leaves_per_step,
            temp_decisions=cfg.temp_decisions, c_puct=cfg.c_puct, root_noise=cfg.root_noise,
            dirichlet_alpha=cfg.dirichlet_alpha, record=True, max_games=cfg.games_per_gen,
            seed=cfg.seed * 100_003 + gen,
        )
        sp_stats = drive(sp, {0: net}, device, on_samples=add)
        results = sp.take_results()
        aborted = sum(r["aborted"] for r in results)
        mean_dec = float(np.mean([r["decisions"] for r in results])) if results else 0.0

        # --- train
        steps = max(1, int(new[0] * cfg.reuse / cfg.batch_size))
        net.train()
        t0 = time.time()
        pl_sum = vl_sum = 0.0
        for _ in range(steps):
            batch = replay.sample(cfg.batch_size, rng)
            loss, pl, vl = losses(net, batch, device, cfg.value_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            pl_sum += float(pl)
            vl_sum += float(vl)
        train_s = time.time() - t0

        rec = {
            "gen": gen,
            "games": len(results),
            "aborted": aborted,
            "decisions_per_game": round(mean_dec, 1),
            "samples": new[0],
            "buffer": replay.n,
            "selfplay_s": round(sp_stats["seconds"], 1),
            "evals_per_s": round(sp_stats["evals"] / max(sp_stats["seconds"], 1e-6)),
            "train_steps": steps,
            "train_s": round(train_s, 1),
            "policy_loss": round(pl_sum / steps, 4),
            "value_loss": round(vl_sum / steps, 4),
        }

        # --- evaluate
        if cfg.eval_every and (gen + 1) % cfg.eval_every == 0:
            h = match(cfg, net, "heuristic", cfg.eval_games, device, seed=10_000 + gen)
            p = match(cfg, net, prev, cfg.eval_games, device, seed=20_000 + gen)
            rec["vs_heuristic"] = round(h["score"], 3)
            rec["vs_prev"] = round(p["score"], 3)
            rec["eval_s"] = round(h["seconds"] + p["seconds"], 1)

        save(net, latest, {"gen": gen})
        save(net, os.path.join(out, f"gen{gen:04d}.pt"), {"gen": gen})
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
