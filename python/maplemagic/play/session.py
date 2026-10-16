"""One game between a person and an agent, as the page shows it."""
from __future__ import annotations

import json
import random
from collections import OrderedDict
from dataclasses import asdict

import torch

from .. import DECK_PAIRS, Match
from .agent import Agent, HeuristicAgent, NetworkAgent, RandomAgent, Search, Thought
from .models import Catalog

KINDS = ("network", "heuristic", "random")
# Networks kept loaded, so going back to one is quick.
NETWORKS_KEPT = 3


class Session:
    """One game at a time against an agent chosen per game."""

    def __init__(self, catalog: Catalog, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.catalog = catalog
        self.networks: OrderedDict[str, NetworkAgent] = OrderedDict()
        self.kind, self.agent, self.search, self.opponent = KINDS[0], None, Search(), ""
        self.human, self.seed = 0, 0
        self.m: Match | None = None
        self.log: list[str] = []
        # The agent's decisions since the person last chose, and its search of the person's decision.
        self.thoughts: list[Thought] = []
        self.advice: Thought | None = None

    def network(self, ident: str) -> NetworkAgent:
        if ident not in self.networks:
            self.networks[ident] = NetworkAgent(str(self.catalog.path(ident)), self.device)
            while len(self.networks) > NETWORKS_KEPT:
                self.networks.popitem(last=False)
        self.networks.move_to_end(ident)
        return self.networks[ident]

    def new(self, opts: dict):
        """A new game. `opts`: agent (network, heuristic or random), checkpoint (a catalog id), the
        `Search` fields, decks (yours, the agent's), seat (0, 1 or random) and seed (blank: random)."""
        kind = opts.get("agent", "network")
        if kind not in KINDS:
            raise ValueError(f"agent {kind!r}: one of {', '.join(KINDS)}")
        search = Search.parse(opts)
        decks = opts.get("decks") or ["Ognis", "Ognis"]
        if not (isinstance(decks, list) and len(decks) == 2 and tuple(decks) in DECK_PAIRS):
            raise ValueError(f"decks {decks!r}: two known decks of one format")
        seat = str(opts.get("seat", "random"))
        human = random.randrange(2) if seat == "random" else int(seat)
        if human not in (0, 1):
            raise ValueError(f"seat {seat!r}: 0, 1 or random")
        seed = str(opts.get("seed") or "").strip()
        seed = int(seed) if seed else random.randrange(1 << 31)
        if kind == "network":
            ident = str(opts.get("checkpoint") or self.catalog.default_id())
            agent: Agent = self.network(ident)
            gen = self.catalog.details(ident)["gen"]
            who, how = f"{ident.split('/')[0]} gen {gen}", search.describe()
        elif kind == "heuristic":
            agent, who, how = HeuristicAgent(), "the heuristic agent", f"{search.sims} sims"
        else:
            agent, who, how = RandomAgent(), "the random agent", "random choices"
        seats = [decks[1], decks[1]]
        seats[human] = decks[0]
        self.m = Match(seats[0], seats[1], seed)
        self.kind, self.agent, self.search, self.human, self.seed = kind, agent, search, human, seed
        self.opponent = f"{who} ({how})"
        self.log = [f"Game {seed}: you play {decks[0]} in seat {human}, against {who} playing {decks[1]} ({how})."]
        self.thoughts, self.advice = [], None
        self.advance()

    def choose(self, i: int):
        if self.m is None or self.m.is_over() or self.m.to_act() != self.human:
            return
        choices = self.m.choices(self.human)
        if not 0 <= i < len(choices):
            raise ValueError(f"choice {i} of {len(choices)}")
        self.log.append(f"You: {choices[i]}")
        self.m.apply(i)
        self.thoughts, self.advice = [], None
        self.advance()

    def advance(self):
        """Lets the agent move until it is the person's turn to choose."""
        while not self.m.is_over() and self.m.to_act() != self.human:
            t = self.agent.think(self.m, self.search, random.randrange(1 << 31))
            self.thoughts.append(t)
            self.log.append(f"Opponent: {self.m.choices(self.human)[t.chose]}")
            self.m.apply(t.chose)

    def ask(self):
        """The agent's search of the person's decision, from what the person can see."""
        if self.m is not None and not self.m.is_over() and self.m.to_act() == self.human:
            self.advice = self.agent.think(self.m, self.search, random.randrange(1 << 31))

    def state(self, debug: bool) -> dict:
        """What the page shows; the agent's hidden information only with `debug`."""
        if self.m is None:
            return {"status": "Set up a game.", "choices": [], "log": [], "opponent": ""}
        over = self.m.is_over()
        if not over:
            status = "Your decision"
        elif self.m.outcome() == f"win {self.human}":
            status = "You won!"
        elif self.m.outcome().startswith("win"):
            status = "The agent won."
        else:
            status = f"Game over: {self.m.outcome()}"
        s = {"status": status, "board": json.loads(self.m.board(self.human)),
             "choices": [] if over else self.m.choices(self.human), "log": self.log[-200:],
             "agent": self.kind, "opponent": self.opponent}
        if debug:
            s["debug"] = {"agent_hand": json.loads(self.m.board(1 - self.human))["you"]["hand"],
                          "thoughts": [asdict(t) for t in self.thoughts], "advice": asdict(self.advice) if self.advice else None}
        return s
