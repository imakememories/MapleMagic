"""The agents a person plays against, how they search, and what the search made of each decision."""
from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields

import numpy as np

from .. import Match
from ..model import evaluate, load


@dataclass
class Search:
    """How an agent searches: its budget, and for a network the settings self-play uses (see `Match.start_search`)."""
    sims: int = 400
    collisions: bool = False
    maple_worlds: int = 0
    maple_select: str = "ref"
    maple_resample: bool = False
    pimc_worlds: int = 0
    perfect_obs: bool = False
    world_sampler: str = "uniform"
    value_mix: float = 1.0

    @classmethod
    def parse(cls, d: dict) -> "Search":
        """From the page's form; bad values raise ValueError."""
        s = cls(**{f.name: d[f.name] for f in fields(cls) if d.get(f.name) not in (None, "")})
        s.sims, s.maple_worlds, s.pimc_worlds = int(s.sims), int(s.maple_worlds), int(s.pimc_worlds)
        s.value_mix = float(s.value_mix)
        for flag in ("collisions", "maple_resample", "perfect_obs"):
            setattr(s, flag, getattr(s, flag) in (True, "true", "on", "1", 1))
        if not 1 <= s.sims <= 20000:
            raise ValueError("sims must be from 1 to 20000")
        if not (0 <= s.maple_worlds <= 64 and 0 <= s.pimc_worlds <= 64):
            raise ValueError("worlds must be from 0 to 64")
        if s.maple_worlds and s.pimc_worlds:
            raise ValueError("a search uses MAPLE worlds or PIMC worlds, not both")
        if s.maple_select not in ("ref", "union") or s.world_sampler not in ("uniform", "oracle"):
            raise ValueError("MAPLE selection is ref or union, and worlds are uniform or oracle")
        if not 0.0 <= s.value_mix <= 1.0:
            raise ValueError("value mix must be from 0 to 1")
        return s

    def describe(self) -> str:
        method = (f"MAPLE {self.maple_worlds} worlds ({self.maple_select}{', resampled' if self.maple_resample else ''})"
                  if self.maple_worlds else f"PIMC {self.pimc_worlds} worlds" if self.pimc_worlds else "IS-MCTS")
        extras = [x for x, on in (("collisions", self.collisions), ("perfect observation", self.perfect_obs),
                                  ("oracle worlds", self.world_sampler == "oracle")) if on]
        if self.value_mix != 1.0:
            extras.append(f"value mix {self.value_mix:g}")
        return f"{self.sims} sims, {method}" + "".join(f", {x}" for x in extras)

    def options(self) -> dict:
        """`Match.start_search`'s keyword arguments."""
        d = asdict(self)
        del d["sims"]
        return d


@dataclass
class Thought:
    """One decision. Values are for the player who searched (+1: a win)."""
    seat: int
    chose: int
    sims: int
    seconds: float
    value: float | None  # the search's
    net_value: float | None  # the evaluator's own, before searching
    rows: list[dict] = field(default_factory=list)  # per choice: text, prior, share of visits, q
    note: str = ""  # why there was no search


def unsearched(m: Match, chose: int, note: str) -> Thought:
    texts = m.choices(m.to_act())
    return Thought(m.to_act(), chose, 0, 0.0, None, None,
                   [{"text": t, "prior": None, "share": None, "q": None} for t in texts], note)


# Called during a search with the match, so a page can show how far it got.
Report = Callable[[Match], None]
# The heuristic searches in chunks of this many simulations, a multiple of its batch of 8, so the batches match one run.
CHUNK = 64


class Agent:
    """Searches the decision at hand, with the choices as the player to act reads them."""

    def search(self, m: Match, search: Search, seed: int, report: Report):
        raise NotImplementedError

    def think(self, m: Match, search: Search, seed: int, report: Report = lambda m: None) -> Thought:
        if len(m.choices(m.to_act())) == 1:
            return unsearched(m, 0, "One choice, so no search.")
        start = time.perf_counter()
        self.search(m, search, seed, report)
        seconds = time.perf_counter() - start
        s = m.root_stats()
        total = sum(s["visits"]) or 1.0
        rows = [{"text": c, "prior": p, "share": v / total, "q": q}
                for c, p, v, q in zip(m.choices(m.to_act()), s["prior"], s["visits"], s["q"])]
        return Thought(m.to_act(), s["best"], s["simulations"], seconds, s["value"], s["net_value"], rows)


class NetworkAgent(Agent):
    def __init__(self, checkpoint: str, device):
        self.net = load(checkpoint, device).eval()
        self.device = device

    def search(self, m: Match, search: Search, seed: int, report: Report):
        m.start_search(seed, **search.options())
        while m.simulations() < search.sims:
            batch = m.gather(8)
            if batch is None:
                continue
            tok, tl, act, al = batch
            priors, values = evaluate(self.net, tok, tl, act, al, self.device)
            m.feed(priors.astype(np.float32), values.astype(np.float32))
            report(m)


class HeuristicAgent(Agent):
    """IS-MCTS with uniform priors and the hand-written board evaluation."""

    def search(self, m: Match, search: Search, seed: int, report: Report):
        m.start_heuristic(seed)
        for done in range(0, search.sims, CHUNK):
            m.run_heuristic(min(CHUNK, search.sims - done))
            report(m)


class RandomAgent(Agent):
    def think(self, m: Match, search: Search, seed: int, report: Report = lambda m: None) -> Thought:
        return unsearched(m, random.Random(seed).randrange(len(m.choices(m.to_act()))), "The random agent doesn't search.")
