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

from . import DECK_PAIRS, MAX_TOKENS, SelfPlay
from .model import EvalPlan, Net, evaluate, load, losses, plan_eval, save, to_tensors

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
    # Learning-rate warm-up steps, then (with lr_cosine) a cosine decay over the run's generations.
    lr_warmup: int = 500
    lr_cosine: bool = True
    # Every embedding table N(0, init_std) in a fresh net (0 = PyTorch's defaults).
    init_std: float = 0.02
    weight_decay: float = 1e-4
    value_weight: float = 1.0
    # Value targets: "z" (the result) or "td" (TD(lambda) over search root values).
    value_target: str = "td"
    value_lambda: float = 0.95
    # Positions per game that train the value (0 = every position, all in one batch with the policy).
    value_positions_per_game: int = 16
    # Times each value position is trained on while in the buffer, when they are sampled apart.
    value_reuse: float = 2.0
    # Leaf values: this weight on the network, the rest on the hand-written evaluation.
    value_mix: float = 1.0
    # Playout-cap randomization: the chance of a full search; other moves use cheap_sims
    # (0 = max(8, sims // 6)) and train only the value.
    full_search_prob: float = 0.25
    cheap_sims: int = 0
    # Policy targets: "visits" (visit shares) or "completed_q" (Gumbel MuZero's completed-Q softmax).
    policy_target: str = "visits"
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
    # Where each world's opponent hand comes from: uniform, oracle, retention or scored.
    world_sampler: str = "uniform"
    world_candidates: int = 50
    world_pool: int = 16
    world_temp: float = 0.0
    # Card attributes (types, colors, keywords, mana value, subtypes, ability lines) beside card ids.
    card_attributes: bool = False
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
            return [tuple(p) for p in DECK_PAIRS]
        return [tuple(p) for p in self.pairings]

    def search_settings(self) -> dict:
        """How this run's networks search; saved in every checkpoint."""
        return {
            "maple_worlds": self.maple_worlds,
            "maple_select": self.maple_select,
            "maple_resample": self.maple_resample,
            "perfect_obs": self.perfect_obs,
            "pimc_worlds": self.pimc_worlds,
            "world_sampler": self.world_sampler,
            "value_mix": self.value_mix,
        }

    def split_batches(self) -> bool:
        """Whether policy and value rows are sampled apart."""
        return self.value_positions_per_game > 0 or self.full_search_prob < 1.0


def value_targets(game, player, z, root_value, kind: str, lam: float) -> np.ndarray:
    """Per-row value targets; rows are whole games in decision order."""
    if kind == "z":
        return z.astype(np.float32)
    if kind != "td":
        raise ValueError(f"unknown value_target {kind}")
    y = np.empty(len(z), np.float32)
    after = {}
    for i in range(len(z) - 1, -1, -1):
        key = (int(game[i]), int(player[i]))
        y[i] = lam * after.get(key, z[i]) + (1 - lam) * root_value[i]
        after[key] = y[i]
    return y


class Replay:
    """Fixed-width ring buffer of samples (padded to MAX_TOKENS / MAX_ACTIONS)."""

    ROW_FIELDS = ("game", "player", "turn", "root_value", "heuristic", "value_target", "value_row", "policy_row")

    def __init__(self, cap: int, tok_fields: int, act_fields: int, value_target: str = "z", value_lambda: float = 0.95,
                 value_positions: int = 0, seed: int = 0):
        self.cap = cap
        self.tok = np.zeros((cap, MAX_TOKENS, tok_fields), np.uint16)
        self.tok_len = np.zeros(cap, np.int32)
        self.act = np.zeros((cap, MAX_ACTIONS, act_fields), np.uint16)
        self.act_len = np.zeros(cap, np.int32)
        self.pi = np.zeros((cap, MAX_ACTIONS), np.float32)
        self.z = np.zeros(cap, np.float32)
        self.game = np.zeros(cap, np.int64)
        self.player = np.zeros(cap, np.uint8)
        self.turn = np.zeros(cap, np.int32)
        self.root_value = np.zeros(cap, np.float32)
        self.heuristic = np.zeros(cap, np.float32)
        self.value_target = np.zeros(cap, np.float32)
        self.value_row = np.zeros(cap, bool)
        self.policy_row = np.zeros(cap, bool)
        self.n = 0
        self.pos = 0
        self.kind, self.lam, self.k = value_target, value_lambda, value_positions
        # Picks value rows; apart from the training rng so K=0 leaves that unchanged.
        self.rng = np.random.default_rng(seed + 7_919)

    def add(self, samples):
        tok, tok_len, act, act_len, pi, z, root_value = samples[:7]
        game, turn, heuristic, player, full = samples[7][5:10]
        k = len(tok_len)
        assert (act_len >= 2).all(), "a forced decision was recorded"
        starts = np.flatnonzero(np.r_[True, game[1:] != game[:-1]])
        assert len(np.unique(game[starts])) == len(starts), "a game's samples are not together"
        target = value_targets(game, player, z, root_value, self.kind, self.lam)
        value_row = np.ones(k, bool)
        if self.k > 0:
            value_row[:] = False
            for a, b in zip(starts, np.r_[starts[1:], k]):
                value_row[a + self.rng.choice(b - a, min(self.k, b - a), replace=False)] = True
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
            self.game[j], self.player[j], self.turn[j] = game[i], player[i], turn[i]
            self.root_value[j], self.heuristic[j], self.value_target[j] = root_value[i], heuristic[i], target[i]
            self.value_row[j], self.policy_row[j] = value_row[i], full[i]
            self.pos = (self.pos + 1) % self.cap
            self.n = min(self.n + 1, self.cap)

    def batch(self, idx: np.ndarray):
        tl = self.tok_len[idx]
        al = self.act_len[idx]
        tw, aw = max(int(tl.max()), 1), max(int(al.max()), 1)
        return (self.tok[idx, :tw], tl, self.act[idx, :aw], al, self.pi[idx, :aw], self.value_target[idx])

    def sample(self, bs: int, rng: np.random.Generator):
        return self.batch(rng.integers(0, self.n, bs))

    def save(self, path: str):
        tmp = path + ".tmp.npz"
        np.savez(tmp, tok=self.tok, tok_len=self.tok_len, act=self.act, act_len=self.act_len, pi=self.pi, z=self.z,
                 n=self.n, pos=self.pos, kind=self.kind, lam=self.lam, k=self.k,
                 **{f: getattr(self, f) for f in self.ROW_FIELDS})
        os.replace(tmp, path)

    def load(self, path: str) -> bool:
        import zipfile
        with np.load(path) as f, zipfile.ZipFile(path) as z:
            why = None
            shapes = (npy_header(z, "tok")[0], npy_header(z, "act")[0])
            if shapes != (self.tok.shape, self.act.shape):
                why = f"its shapes {shapes} differ from {self.tok.shape}, {self.act.shape}"
            elif any(x not in f for x in self.ROW_FIELDS):
                if (self.kind, self.k) != ("z", 0):
                    why = "it predates per-row game, player and value fields"
                else:
                    for k in ("tok", "tok_len", "act", "act_len", "pi", "z"):
                        read_npy_into(z, k, getattr(self, k))
                    self.value_target[:] = self.z
                    self.value_row[:] = self.policy_row[:] = True
                    self.n, self.pos = int(f["n"]), int(f["pos"])
                    print(f"loaded {path} from before per-row fields: every row trains both heads on the result")
                    return True
            elif (str(f["kind"]), float(f["lam"]), int(f["k"])) != (self.kind, self.lam, self.k):
                why = f"its value targets ({f['kind']}, {f['lam']}, K={f['k']}) differ from ({self.kind}, {self.lam}, K={self.k})"
            if why:
                print(f"not loading the replay buffer {path}: {why}")
                return False
            for k in ("tok", "tok_len", "act", "act_len", "pi", "z") + self.ROW_FIELDS:
                read_npy_into(z, k, getattr(self, k))
            self.n, self.pos = int(f["n"]), int(f["pos"])
        return True


def npy_header(z, name: str):
    """(shape, fortran order, dtype) of array `name` in an open .npz."""
    with z.open(name + ".npy") as fp:
        version = np.lib.format.read_magic(fp)
        read = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        return read(fp)


def read_npy_into(z, name: str, out: np.ndarray, chunk: int = 64 << 20):
    """Reads array `name` of an .npz straight into `out`, a chunk at a time
    (np.load would hold a second full copy, which Windows keeps charged)."""
    with z.open(name + ".npy") as fp:
        version = np.lib.format.read_magic(fp)
        read = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = read(fp)
        if tuple(shape) != out.shape or dtype != out.dtype or fortran:
            raise ValueError(f"{name}: {shape} {dtype} doesn't fit {out.shape} {out.dtype}")
        view = memoryview(out.reshape(-1).view(np.uint8))
        at = 0
        while at < len(view):
            n = fp.readinto(view[at:at + chunk])
            if not n:
                raise EOFError(f"{name} ends early")
            at += n


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


def drive(sp: SelfPlay, nets: dict, device, on_samples=None, plan: EvalPlan | None = None,
          scorers: dict | None = None) -> dict:
    """Run a SelfPlay to completion, evaluating leaves with `nets[model_id]` and
    candidate hands with `scorers[model_id]` (a function of a world batch)."""
    plan = plan or plan_eval(nets[0], device, MAX_TOKENS, MAX_ACTIONS)
    max_batch = plan.max_batch
    evals = {0: 0, 1: 0}
    t0 = time.time()
    while True:
        req = sp.gather()
        if on_samples is not None:
            s = sp.take_samples()
            if s is not None:
                on_samples(s)
        if req is None:
            break
        tok, tl, act, al, model, worlds = req
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
        scores = None
        if worlds is not None:
            wmodel = worlds[-1]
            scores = np.zeros(worlds[4].shape[:2], np.float32)
            for m in np.unique(wmodel):
                if not scorers or int(m) not in scorers:
                    raise ValueError(f"model {int(m)} asks for scored worlds but has no sampler")
                rows = np.nonzero(wmodel == m)[0]
                scores[rows] = scorers[int(m)](tuple(x[rows] for x in worlds[:-1]))
        sp.feed(priors, values, scores)
    return {"evals": evals[0] + evals[1], "evals_a": evals[0], "evals_b": evals[1], "seconds": time.time() - t0}


@torch.no_grad()
def heldout(net: Net, replay: Replay, idx: np.ndarray, device, cfg: Config) -> dict:
    """The network and the hand-written evaluation on new positions, before training on them:
    policy loss on full-search rows, value MSE and AUC against the result."""
    from .yardstick import auc

    net.eval()
    ce, values, pass_net, pass_target = [], [], [], []
    for at in range(0, len(idx), cfg.batch_size):
        rows = idx[at:at + cfg.batch_size]
        tok, tl, act, al, pi, _ = replay.batch(rows)
        logits, v = net(*to_tensors((tok, tl, act, al), device))
        pit = torch.from_numpy(pi).to(device)
        logp = torch.log_softmax(logits.float(), dim=-1).masked_fill(pit == 0, 0.0)
        ce.append((-(pit * logp).sum(-1)).cpu().numpy())
        values.append(v.float().cpu().numpy())
        # Kind 0 is Pass (game.rs Kind).
        is_pass = (act[:, :, 0] == 0) & (np.arange(act.shape[1])[None, :] < al[:, None])
        prob = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        pass_net.append((prob * is_pass).sum(1))
        pass_target.append((pi * is_pass).sum(1))
    ce, v = np.concatenate(ce), np.concatenate(values)
    pass_net, pass_target = np.concatenate(pass_net), np.concatenate(pass_target)
    z, h, full = replay.z[idx], replay.heuristic[idx], replay.policy_row[idx]
    decided = z != 0
    return {
        "heldout_policy_loss": round(float(ce[full].mean()), 4) if full.any() else None,
        "heldout_value_loss": round(float(((v - z) ** 2).mean()), 4),
        "heldout_value_auc": auc(v[decided], z[decided] > 0),
        "heldout_heuristic_mse": round(float(((h - z) ** 2).mean()), 4),
        "heldout_heuristic_auc": auc(h[decided], z[decided] > 0),
        "heldout_confident": round(float((np.abs(v) > 0.9).mean()), 3),
        "heldout_pass_target": round(float(pass_target[full].mean()), 3) if full.any() else None,
        "heldout_pass_net": round(float(pass_net[full].mean()), 3) if full.any() else None,
    }


def summarize_stats(s: dict) -> dict:
    """Per-simulation rates from one model's search counters."""
    n = max(s["sims"], 1)
    return {
        "leaf_worlds_per_sim": round(s["leaf_worlds"] / n, 3),
        "evals_per_sim": round(s["unique_leaves"] / n, 3),
        "leaf_decisions": round(s["leaf_decisions"] / max(s["leaves"], 1), 2),
        "leaf_turns": round(s["leaf_turns"] / max(s["leaves"], 1), 3),
        "dropped_illegal_per_sim": round(s["dropped_illegal"] / n, 3),
        "dropped_diverged_per_sim": round(s["dropped_diverged"] / n, 3),
        "no_world_per_sim": round(s["no_world_sims"] / n, 4),
        "stale_priors_per_sim": round(s["stale_priors"] / n, 4),
    }


def match(cfg: Config, net_a, opponent, games: int, device, seed: int,
          search_a: dict | None = None, search_b: dict | None = None, opp_sims: int | None = None,
          scorers: dict | None = None, retention=None) -> dict:
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
        world_sampler=(sa.get("world_sampler", "uniform"), sb.get("world_sampler", "uniform")),
        world_candidates=cfg.world_candidates, world_pool=cfg.world_pool, world_temp=cfg.world_temp,
        retention=retention, value_scale=(sa.get("value_scale", 1.0), sb.get("value_scale", 1.0)),
        heuristic_values=(sa.get("heuristic_values", False), sb.get("heuristic_values", False)),
        value_mix=(sa.get("value_mix", 1.0), sb.get("value_mix", 1.0)),
    )
    plan = plan_eval(net_a, device, MAX_TOKENS, MAX_ACTIONS, cfg.max_eval_batch)
    stats = drive(sp, nets, device, plan=plan, scorers=scorers)
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


def lr_scale(cfg: Config, step: int, progress: float) -> float:
    """The learning rate's multiplier at global `step`, `progress` (0..1) through the run."""
    warm = min(1.0, (step + 1) / cfg.lr_warmup) if cfg.lr_warmup else 1.0
    return warm * (0.5 * (1 + math.cos(math.pi * progress)) if cfg.lr_cosine else 1.0)


def init_embeddings(net: Net, std: float):
    for m in net.modules():
        if isinstance(m, torch.nn.Embedding):
            torch.nn.init.normal_(m.weight, std=std)


def train_steps(net, opt, replay: Replay, cfg: Config, rng, device, plan: EvalPlan, new_policy: int, new_value: int,
                reuse: float, value_reuse: float, step0: int = 0, progress: tuple[float, float] = (0.0, 0.0)):
    """Train on the buffer so each new policy row is used about `reuse` times and
    each new value row about `value_reuse` times, from global step `step0`, the run
    `progress` from its first to its last step. Returns (steps, policy loss sum,
    value loss sum, seconds)."""
    steps = max(1, int(new_policy * reuse / cfg.batch_size))
    if cfg.split_batches():
        policy_rows = np.flatnonzero(replay.policy_row[:replay.n])
        value_rows = np.flatnonzero(replay.value_row[:replay.n])
        value_batch = max(1, round(value_reuse * new_value / steps))
    net.train()
    t0 = time.time()
    pl_sum = vl_sum = 0.0
    for i in range(steps):
        if cfg.lr_warmup or cfg.lr_cosine:
            for g in opt.param_groups:
                g["lr"] = cfg.lr * lr_scale(cfg, step0 + i, progress[0] + (progress[1] - progress[0]) * i / steps)
        if cfg.split_batches():
            rows = np.concatenate([policy_rows[rng.integers(0, len(policy_rows), cfg.batch_size)],
                                   value_rows[rng.integers(0, len(value_rows), value_batch)]])
            loss, pl, vl = losses(net, replay.batch(rows), device, cfg.value_weight, split=cfg.batch_size)
        else:
            batch = replay.sample(cfg.batch_size, rng)
            loss, pl, vl = losses(net, batch, device, cfg.value_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if device.type == "cuda" and torch.cuda.memory_reserved(device) > plan.flush_at:
            torch.cuda.empty_cache()
        pl_sum += float(pl)
        vl_sum += float(vl)
    if pl_sum == 0.0 or not math.isfinite(pl_sum + vl_sum):
        raise DeviceFailure(f"training losses are {pl_sum / steps}, {vl_sum / steps}; the GPU is not computing. "
                            "Restart the process (training resumes from latest.pt).")
    return steps, pl_sum, vl_sum, time.time() - t0


def collect_heuristic(cfg: Config, games: int, sims: int, seed: int, cap: int) -> tuple[Replay, dict]:
    """Self-play between heuristic agents searching `sims` simulations, recorded
    like network self-play (CPU only), in chunks of `cfg.parallel` games so each
    chunk's samples leave Rust before the next starts."""
    from . import ACTION_FIELDS, TOKEN_FIELDS
    replay = Replay(cap, TOKEN_FIELDS, ACTION_FIELDS, cfg.value_target, cfg.value_lambda,
                    cfg.value_positions_per_game, seed)
    t0 = time.time()
    played = aborted = 0
    for chunk, at in enumerate(range(0, games, cfg.parallel)):
        sp = SelfPlay(
            cfg.parallel, cfg.pairing_list(), temp_decisions=cfg.temp_decisions,
            seat_models=("heuristic", "heuristic"), record=True, max_games=min(cfg.parallel, games - at),
            heuristic_sims=sims, seed=seed * 1000 + chunk, record_heuristic=True,
            completed_q={"visits": False, "completed_q": True}[cfg.policy_target],
        )
        if sp.gather() is not None:
            raise RuntimeError("heuristic self-play asked for the network")
        while (s := sp.take_samples()) is not None:
            replay.add(s)
        results = sp.take_results()
        played += len(results)
        aborted += sum(r["aborted"] for r in results)
        del sp, s
        print(f"  {played} games, {replay.n} samples, {time.time() - t0:.0f}s", flush=True)
    return replay, {"games": played, "aborted": aborted, "samples": replay.n, "seconds": round(time.time() - t0, 1)}


def shrink_perturb(net: Net, cfg: Config, shrink: float, perturb: float):
    """theta <- shrink * theta + perturb * theta_fresh, theta_fresh a new initialization."""
    torch.manual_seed(cfg.seed + 7)
    fresh = Net(cfg.d_model, cfg.layers, cfg.heads, cfg.card_attributes).to(next(net.parameters()).device)
    with torch.no_grad():
        for p, q in zip(net.parameters(), fresh.parameters()):
            p.mul_(shrink).add_(perturb * q)


def plasticity(cfg: Config, start: str, out_path: str, seed: int = 3):
    """Train a start checkpoint and a fresh net on the same two generations of
    self-play (played by the start), each scored on a generation before training
    on it, as the loop does. A start that learns worse than a fresh net has lost
    plasticity."""
    from . import ACTION_FIELDS, TOKEN_FIELDS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    player = load(start, device)
    plan = plan_eval(player, device, MAX_TOKENS, MAX_ACTIONS, cfg.max_eval_batch)
    gens = []
    for g in range(2):
        replay = Replay(cfg.buffer_size, TOKEN_FIELDS, ACTION_FIELDS, cfg.value_target, cfg.value_lambda,
                        cfg.value_positions_per_game, seed + g)
        sp = SelfPlay(
            cfg.parallel, cfg.pairing_list(), sims=cfg.sims, leaves_per_step=cfg.leaves_per_step,
            temp_decisions=cfg.temp_decisions, c_puct=cfg.c_puct, root_noise=cfg.root_noise,
            dirichlet_alpha=cfg.dirichlet_alpha, record=True, max_games=cfg.games_per_gen, seed=seed * 1000 + g,
            value_mix=cfg.value_mix, full_search_prob=cfg.full_search_prob,
            cheap_sims=cfg.cheap_sims or max(8, cfg.sims // 6),
            completed_q={"visits": False, "completed_q": True}[cfg.policy_target],
        )
        drive(sp, {0: player}, device, on_samples=replay.add, plan=plan)
        gens.append(replay)
    report = {}
    for name in ("start", "fresh"):
        torch.manual_seed(cfg.seed)
        net = load(start, device) if name == "start" else Net(cfg.d_model, cfg.layers, cfg.heads, cfg.card_attributes).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        rng = np.random.default_rng(cfg.seed)
        rows = []
        for replay in gens:
            held = rng.permutation(replay.n)[:4096]
            rows.append(heldout(net, replay, held, device, cfg))
            n_policy, n_value = int(replay.policy_row[:replay.n].sum()), int(replay.value_row[:replay.n].sum())
            train_steps(net, opt, replay, cfg, rng, device, plan, n_policy, n_value, cfg.reuse, cfg.value_reuse)
        report[name] = rows
    with open(out_path, "w") as f:
        json.dump(report, f, indent=1)
    for name, rows in report.items():
        print(name, json.dumps({k: rows[1][k] for k in ("heldout_policy_loss", "heldout_value_loss", "heldout_value_auc")}))
    return report


def pretrain(cfg: Config, games: int, sims: int, passes: float, shrink: float, perturb: float, seed: int = 1):
    """Generation 0 from heuristic search: train a fresh net on heuristic self-play,
    shrink and perturb it (Ash & Adams 2020), and save it as the run's latest.pt
    with gen -1 so `train` continues at gen 0."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = os.path.join(cfg.out_dir, cfg.name)
    os.makedirs(out, exist_ok=True)
    data = os.path.join(out, f"pretrain_data_{games}x{sims}.npz")
    info_path = data.replace(".npz", ".json")
    if os.path.exists(data) and os.path.exists(info_path):
        from . import ACTION_FIELDS, TOKEN_FIELDS
        replay = Replay(180 * games, TOKEN_FIELDS, ACTION_FIELDS, cfg.value_target, cfg.value_lambda,
                        cfg.value_positions_per_game, seed)
        if not replay.load(data):
            raise RuntimeError(f"can't reuse {data}")
        info = json.load(open(info_path))
        print(f"reusing {data}: {json.dumps(info)}", flush=True)
    else:
        replay, info = collect_heuristic(cfg, games, sims, seed, cap=180 * games)
        replay.save(data)
        with open(info_path, "w") as f:
            json.dump(info, f)
        print(f"collected {json.dumps(info)}", flush=True)
    torch.manual_seed(cfg.seed)
    net = Net(cfg.d_model, cfg.layers, cfg.heads, cfg.card_attributes).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    plan = plan_eval(net, device, MAX_TOKENS, MAX_ACTIONS, cfg.max_eval_batch)
    rng = np.random.default_rng(cfg.seed)
    held = rng.permutation(replay.n)[:4096]
    before = heldout(net, replay, held, device, cfg)
    n_policy, n_value = int(replay.policy_row[:replay.n].sum()), int(replay.value_row[:replay.n].sum())
    steps, pl, vl, secs = train_steps(net, opt, replay, cfg, rng, device, plan, n_policy, n_value, passes, passes)
    trained = heldout(net, replay, held, device, cfg)
    save(net, os.path.join(out, "gen0_trained.pt"), {"gen": -1, "search": cfg.search_settings()})
    shrink_perturb(net, cfg, shrink, perturb)
    rec = {"pretrain": info, "sims": sims, "passes": passes, "shrink": shrink, "perturb": perturb, "train_steps": steps,
           "train_s": round(secs, 1), "policy_loss": round(pl / steps, 4), "value_loss": round(vl / steps, 4),
           "untrained_on_its_data": before, "trained_on_its_data": trained,
           "shrunk_on_its_data": heldout(net, replay, held, device, cfg)}
    save(net, os.path.join(out, "latest.pt"), {"gen": -1, "search": cfg.search_settings()})
    save(net, os.path.join(out, "gen0_start.pt"), {"gen": -1, "search": cfg.search_settings()})
    with open(os.path.join(out, "pretrain.json"), "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps(rec), flush=True)


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
    replay = Replay(cfg.buffer_size, TOKEN_FIELDS, ACTION_FIELDS, cfg.value_target, cfg.value_lambda,
                    cfg.value_positions_per_game, cfg.seed)
    start_gen = 0
    opt_state = None
    step = 0
    if resume and os.path.exists(latest):
        net = load(latest, device)
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        start_gen = ck.get("gen", -1) + 1
        opt_state = ck.get("opt")
        step = ck.get("step", 0)
        kept = os.path.exists(replay_path) and replay.load(replay_path)
        print(f"resumed from {latest} at generation {start_gen}, replay buffer {replay.n if kept else 'empty'}")
    else:
        net = Net(cfg.d_model, cfg.layers, cfg.heads, cfg.card_attributes).to(device)
        if cfg.init_std:
            init_embeddings(net, cfg.init_std)
        if os.path.exists(log_path):
            os.replace(log_path, os.path.join(out, f"log.{time.strftime('%Y%m%d-%H%M%S')}.jsonl"))
    torch.manual_seed(cfg.seed + start_gen)
    rng = np.random.default_rng(cfg.seed + start_gen)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if opt_state is not None:
        opt.load_state_dict(opt_state)
        for p in net.parameters():
            for k, v in opt.state.get(p, {}).items():
                if torch.is_tensor(v) and v.dim() == 2 and v.shape[0] < p.shape[0]:
                    opt.state[p][k] = torch.cat([v, v.new_zeros(p.shape[0] - v.shape[0], v.shape[1])])
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
            world_sampler=cfg.world_sampler, world_candidates=cfg.world_candidates, world_pool=cfg.world_pool,
            world_temp=cfg.world_temp, value_mix=cfg.value_mix, full_search_prob=cfg.full_search_prob,
            cheap_sims=cfg.cheap_sims or max(8, cfg.sims // 6),
            completed_q={"visits": False, "completed_q": True}[cfg.policy_target],
        )
        sp_stats = drive(sp, {0: net}, device, on_samples=add, plan=plan)
        search = summarize_stats(sp.search_stats()[0])
        results = sp.take_results()
        aborted = sum(r["aborted"] for r in results)
        mean_dec = float(np.mean([r["decisions"] for r in results])) if results else 0.0

        held = (first + rng.permutation(min(new[0], replay.cap))[:4096]) % replay.cap
        held_stats = heldout(net, replay, held, device, cfg) if len(held) else {}
        fresh = (first + np.arange(min(new[0], replay.cap))) % replay.cap
        new_policy, new_value = int(replay.policy_row[fresh].sum()), int(replay.value_row[fresh].sum())

        # --- train
        if device.type == "cuda":
            torch.cuda.empty_cache()  # self-play's cache, before training allocates its own
        steps, pl_sum, vl_sum, train_s = train_steps(net, opt, replay, cfg, rng, device, plan, new_policy, new_value,
                                                     cfg.reuse, cfg.value_reuse, step,
                                                     (gen / cfg.generations, (gen + 1) / cfg.generations))
        step += steps

        rec = {
            "gen": gen,
            "games": len(results),
            "aborted": aborted,
            "decisions_per_game": round(mean_dec, 1),
            "samples": new[0],
            "policy_samples": new_policy,
            "value_samples": new_value,
            "buffer": replay.n,
            "selfplay_s": round(sp_stats["seconds"], 1),
            "evals": sp_stats["evals"],
            "evals_per_s": round(sp_stats["evals"] / max(sp_stats["seconds"], 1e-6)),
            "evals_per_move": round(sp_stats["evals"] / max(new[0], 1), 1),
            **({"search": search} if cfg.maple_worlds or cfg.pimc_worlds else {}),
            "train_steps": steps,
            "train_s": round(train_s, 1),
            "policy_loss": round(pl_sum / steps, 4),
            "value_loss": round(vl_sum / steps, 4),
            **held_stats,
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
        save(net, latest, {**meta, "opt": opt.state_dict(), "step": step})
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
