"""Generational AlphaZero loop: self-play (in Rust), train, evaluate."""
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
from .model import EvalPlan, Net, evaluate, load, losses, plan_eval, save

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
    # Search: MAPLE worlds (0 = IS-MCTS), and whether the network sees the guessed hand.
    maple_worlds: int = 0
    maple_select: str = "ref"
    maple_resample: bool = False
    perfect_obs: bool = False
    # PIMC (AlphaZe**) worlds, with `sims` split across them.
    pimc_worlds: int = 0
    # Rows per forward pass; 0 sizes it for this machine (model.plan_eval).
    max_eval_batch: int = 0
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

    def search_settings(self) -> dict:
        """How this run's networks search; saved in every checkpoint."""
        return {
            "maple_worlds": self.maple_worlds,
            "maple_select": self.maple_select,
            "maple_resample": self.maple_resample,
            "perfect_obs": self.perfect_obs,
            "pimc_worlds": self.pimc_worlds,
        }


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

    def batch(self, idx: np.ndarray):
        tl = self.tok_len[idx]
        al = self.act_len[idx]
        tw, aw = max(int(tl.max()), 1), max(int(al.max()), 1)
        return (self.tok[idx, :tw], tl, self.act[idx, :aw], al, self.pi[idx, :aw], self.z[idx])

    def sample(self, bs: int, rng: np.random.Generator):
        return self.batch(rng.integers(0, self.n, bs))

    def save(self, path: str):
        tmp = path + ".tmp.npz"
        np.savez(tmp, tok=self.tok, tok_len=self.tok_len, act=self.act, act_len=self.act_len, pi=self.pi, z=self.z,
                 n=self.n, pos=self.pos)
        os.replace(tmp, path)

    def load(self, path: str) -> bool:
        with np.load(path) as f:
            if f["tok"].shape != self.tok.shape or f["act"].shape != self.act.shape:
                return False
            for k in ("tok", "tok_len", "act", "act_len", "pi", "z"):
                getattr(self, k)[:] = f[k]
            self.n, self.pos = int(f["n"]), int(f["pos"])
        return True


class DeviceFailure(RuntimeError):
    """The GPU returned impossible results without raising an error."""


def check_softmax(p: np.ndarray):
    """Fail if the GPU silently stopped computing (priors no longer sum to 1)."""
    sums = p.sum(1)
    if not np.all(np.abs(sums - 1.0) < 1e-2):
        raise DeviceFailure(
            f"network priors do not sum to 1 (row sums {sums.min():.3g}..{sums.max():.3g}); "
            "the GPU has likely stopped executing kernels. Restart the process (training resumes from latest.pt)."
        )


def drive(sp: SelfPlay, nets: dict, device, on_samples=None, plan: EvalPlan | None = None) -> dict:
    """Run a SelfPlay to completion, evaluating leaves with `nets[model_id]`."""
    plan = plan or plan_eval(nets[0], device, MAX_TOKENS, MAX_ACTIONS)
    max_batch = plan.max_batch
    evals = {0: 0, 1: 0}
    t0 = time.time()
    while True:
        req = sp.gather()
        if req is None:
            break
        tok, tl, act, al, model = req
        priors = np.zeros(act.shape[:2], np.float32)
        values = np.zeros(len(tl), np.float32)
        for m in np.unique(model):
            idx = np.nonzero(model == m)[0]
            for at in range(0, len(idx), max_batch):
                rows = idx[at:at + max_batch]
                # Trim padding for this subset.
                tw = max(int(tl[rows].max()), 1)
                aw = max(int(al[rows].max()), 1)
                p, v = evaluate(nets[int(m)], tok[rows, :tw], tl[rows], act[rows, :aw], al[rows], device)
                check_softmax(p)
                if device.type == "cuda" and torch.cuda.memory_reserved(device) > plan.flush_at:
                    torch.cuda.empty_cache()
                priors[rows, :aw] = p
                values[rows] = v
            evals[int(m)] += len(idx)
        sp.feed(priors, values)
        if on_samples is not None:
            s = sp.take_samples()
            if s is not None:
                on_samples(s)
    return {"evals": evals[0] + evals[1], "evals_a": evals[0], "evals_b": evals[1], "seconds": time.time() - t0}


@torch.no_grad()
def heldout_losses(net: Net, replay: Replay, idx: np.ndarray, device, cfg: Config) -> tuple[float, float]:
    net.eval()
    pl = vl = 0.0
    for at in range(0, len(idx), cfg.batch_size):
        rows = idx[at:at + cfg.batch_size]
        _, p, v = losses(net, replay.batch(rows), device, cfg.value_weight)
        pl += float(p) * len(rows)
        vl += float(v) * len(rows)
    return pl / len(idx), vl / len(idx)


def summarize_stats(s: dict) -> dict:
    """Per-simulation rates from one model's search counters."""
    n = max(s["sims"], 1)
    return {
        "leaf_worlds_per_sim": round(s["leaf_worlds"] / n, 3),
        "evals_per_sim": round(s["unique_leaves"] / n, 3),
        "dropped_illegal_per_sim": round(s["dropped_illegal"] / n, 3),
        "dropped_diverged_per_sim": round(s["dropped_diverged"] / n, 3),
        "no_world_per_sim": round(s["no_world_sims"] / n, 4),
        "stale_priors_per_sim": round(s["stale_priors"] / n, 4),
    }


def match(cfg: Config, net_a, opponent, games: int, device, seed: int,
          search_a: dict | None = None, search_b: dict | None = None, opp_sims: int | None = None) -> dict:
    """Score of net_a against `opponent` (a Net, "heuristic" or "random"), with
    search settings per side (default: cfg's)."""
    nets = {0: net_a}
    opp_id = 1
    if isinstance(opponent, str):
        opp_id = opponent
    else:
        nets[1] = opponent
    sa = search_a or cfg.search_settings()
    sb = search_b or cfg.search_settings()
    sp = SelfPlay(
        min(games, cfg.parallel), cfg.pairing_list(), sims=cfg.eval_sims, leaves_per_step=cfg.leaves_per_step,
        temp_decisions=0, c_puct=cfg.c_puct, root_noise=0.0, seat_models=(0, opp_id), record=False,
        max_games=games, heuristic_sims=cfg.heuristic_sims, seed=seed, opp_sims=opp_sims,
        maple_worlds=(sa["maple_worlds"], sb["maple_worlds"]),
        maple_select=(sa["maple_select"], sb["maple_select"]),
        maple_resample=(sa["maple_resample"], sb["maple_resample"]), perfect_obs=(sa["perfect_obs"], sb["perfect_obs"]),
        pimc_worlds=(sa.get("pimc_worlds", 0), sb.get("pimc_worlds", 0)),
    )
    plan = plan_eval(net_a, device, MAX_TOKENS, MAX_ACTIONS, cfg.max_eval_batch)
    stats = drive(sp, nets, device, plan=plan)
    stats["plan"] = plan
    stats["search_stats"] = sp.search_stats()
    games = sp.take_results()
    w = l = d = 0
    for r in games:
        if r["winner"] is None:
            d += 1
        elif r["models"][r["winner"]] == 0:
            w += 1
        else:
            l += 1
    n = max(w + l + d, 1)
    score = (w + 0.5 * d) / n
    return {"w": w, "l": l, "d": d, "score": score, "ci95": 1.96 * math.sqrt(score * (1 - score) / n),
            "games": games, **stats}


def run(cfg: Config, resume: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = os.path.join(cfg.out_dir, cfg.name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.yaml"), "w") as f:
        yaml.safe_dump(asdict(cfg), f)
    log_path = os.path.join(out, "log.jsonl")
    latest = os.path.join(out, "latest.pt")
    replay_path = os.path.join(out, "replay.npz")

    from . import ACTION_FIELDS, TOKEN_FIELDS
    replay = Replay(cfg.buffer_size, TOKEN_FIELDS, ACTION_FIELDS)
    start_gen = 0
    opt_state = None
    if resume and os.path.exists(latest):
        net = load(latest, device)
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        start_gen = ck.get("gen", -1) + 1
        opt_state = ck.get("opt")
        kept = os.path.exists(replay_path) and replay.load(replay_path)
        print(f"resumed from {latest} at generation {start_gen}, replay buffer {replay.n if kept else 'empty'}")
    else:
        net = Net(cfg.d_model, cfg.layers, cfg.heads).to(device)
        if os.path.exists(log_path):
            os.replace(log_path, os.path.join(out, f"log.{time.strftime('%Y%m%d-%H%M%S')}.jsonl"))
    torch.manual_seed(cfg.seed + start_gen)
    rng = np.random.default_rng(cfg.seed + start_gen)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if opt_state is not None:
        opt.load_state_dict(opt_state)
    print(f"device {device}, params {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M, "
          f"{len(cfg.pairing_list())} pairings")
    plan = plan_eval(net, device, MAX_TOKENS, MAX_ACTIONS, cfg.max_eval_batch)
    print(plan.describe(), flush=True)

    for gen in range(start_gen, cfg.generations):
        # --- self-play
        prev = copy.deepcopy(net).eval()
        first = replay.pos
        new = [0]

        def add(s):
            replay.add(s)
            new[0] += len(s[1])

        sp = SelfPlay(
            cfg.parallel, cfg.pairing_list(), sims=cfg.sims, leaves_per_step=cfg.leaves_per_step,
            temp_decisions=cfg.temp_decisions, c_puct=cfg.c_puct, root_noise=cfg.root_noise,
            dirichlet_alpha=cfg.dirichlet_alpha, record=True, max_games=cfg.games_per_gen,
            seed=cfg.seed * 100_003 + gen, maple_worlds=(cfg.maple_worlds, cfg.maple_worlds),
            maple_select=cfg.maple_select, maple_resample=cfg.maple_resample,
            perfect_obs=(cfg.perfect_obs, cfg.perfect_obs), pimc_worlds=(cfg.pimc_worlds, cfg.pimc_worlds),
        )
        sp_stats = drive(sp, {0: net}, device, on_samples=add, plan=plan)
        search = summarize_stats(sp.search_stats()[0])
        results = sp.take_results()
        aborted = sum(r["aborted"] for r in results)
        mean_dec = float(np.mean([r["decisions"] for r in results])) if results else 0.0

        held = (first + rng.permutation(min(new[0], replay.cap))[:4096]) % replay.cap
        held_pl, held_vl = heldout_losses(net, replay, held, device, cfg) if len(held) else (0.0, 0.0)

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
        if pl_sum == 0.0 or not math.isfinite(pl_sum + vl_sum):
            raise DeviceFailure(f"training losses are {pl_sum / steps}, {vl_sum / steps}; the GPU is not computing. "
                                "Restart the process (training resumes from latest.pt).")

        rec = {
            "gen": gen,
            "games": len(results),
            "aborted": aborted,
            "decisions_per_game": round(mean_dec, 1),
            "samples": new[0],
            "buffer": replay.n,
            "selfplay_s": round(sp_stats["seconds"], 1),
            "evals_per_s": round(sp_stats["evals"] / max(sp_stats["seconds"], 1e-6)),
            "evals_per_move": round(sp_stats["evals"] / max(new[0], 1), 1),
            **({"search": search} if cfg.maple_worlds or cfg.pimc_worlds else {}),
            "train_steps": steps,
            "train_s": round(train_s, 1),
            "policy_loss": round(pl_sum / steps, 4),
            "value_loss": round(vl_sum / steps, 4),
            "heldout_policy_loss": round(held_pl, 4),
            "heldout_value_loss": round(held_vl, 4),
        }

        # --- evaluate
        if cfg.eval_every and (gen + 1) % cfg.eval_every == 0:
            h = match(cfg, net, "heuristic", cfg.eval_games, device, seed=10_000 + gen)
            p = match(cfg, net, prev, cfg.eval_games, device, seed=20_000 + gen)
            rec["vs_heuristic"] = round(h["score"], 3)
            rec["vs_prev"] = round(p["score"], 3)
            rec["eval_s"] = round(h["seconds"] + p["seconds"], 1)

        meta = {"gen": gen, "search": cfg.search_settings()}
        save(net, os.path.join(out, f"gen{gen:04d}.pt"), meta)
        replay.save(replay_path)
        save(net, latest, {**meta, "opt": opt.state_dict()})
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
