"""The agent a person plays against, and what its search made of each decision."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .. import Match
from ..model import evaluate, load, load_meta


@dataclass
class Thought:
    """One searched decision. Values are for the player who searched (+1: a win)."""
    seat: int
    chose: int
    sims: int
    seconds: float
    value: float | None  # the search's
    net_value: float | None  # the evaluator's own, before searching
    rows: list[dict] = field(default_factory=list)  # per choice: text, prior, share of visits, q


class Agent:
    """Chooses with a network-guided search, or with the heuristic agent's search."""

    def __init__(self, checkpoint: str | None, device):
        self.net = load(checkpoint, device).eval() if checkpoint else None
        self.device = device
        # As the network trained; a checkpoint without the key searched without collisions.
        self.collisions = bool(checkpoint and (load_meta(checkpoint).get("search") or {}).get("collisions", False))

    def search(self, m: Match, sims: int, seed: int):
        if self.net is None:
            m.heuristic_choice(sims, seed)
            return
        m.start_search(seed, self.collisions)
        while m.simulations() < sims:
            batch = m.gather(8)
            if batch is None:
                continue
            tok, tl, act, al = batch
            priors, values = evaluate(self.net, tok, tl, act, al, self.device)
            m.feed(priors.astype(np.float32), values.astype(np.float32))

    def think(self, m: Match, sims: int, seed: int) -> Thought:
        """Searches the decision at hand, with the choices as the player to act reads them."""
        seat = m.to_act()
        texts = m.choices(seat)
        if len(texts) == 1:
            return Thought(seat, 0, 0, 0.0, None, None, [{"text": texts[0], "prior": None, "share": None, "q": None}])
        start = time.perf_counter()
        self.search(m, sims, seed)
        seconds = time.perf_counter() - start
        s = m.root_stats()
        total = sum(s["visits"]) or 1.0
        rows = [{"text": c, "prior": p, "share": v / total, "q": q}
                for c, p, v, q in zip(texts, s["prior"], s["visits"], s["q"])]
        return Thought(seat, s["best"], s["simulations"], seconds, s["value"], s["net_value"], rows)
