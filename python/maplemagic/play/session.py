"""One game between a person and an agent, as the page shows it."""
from __future__ import annotations

import random
from dataclasses import asdict

import torch

from .. import Match
from .agent import Agent, Thought

KINDS = ("network", "heuristic")


class Session:
    """One game at a time; each kind of agent loads on its first game."""

    def __init__(self, checkpoint: str, deck: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint, self.deck = checkpoint, deck
        self.agents: dict[str, Agent] = {}
        self.kind, self.sims, self.human, self.seed = KINDS[0], 400, 0, 0
        self.m: Match | None = None
        self.log: list[str] = []
        # The agent's decisions since the person last chose, and its search of the person's decision.
        self.thoughts: list[Thought] = []
        self.advice: Thought | None = None

    def new(self, kind: str, sims: int, seat: str, seed: int | None):
        if kind not in KINDS:
            raise ValueError(f"agent {kind!r}: one of {', '.join(KINDS)}")
        if sims < 1:
            raise ValueError("sims must be at least 1")
        human = random.randrange(2) if seat == "random" else int(seat)
        if human not in (0, 1):
            raise ValueError(f"seat {seat!r}: 0, 1 or random")
        if kind not in self.agents:
            self.agents[kind] = Agent(self.checkpoint if kind == "network" else None, self.device)
        self.kind, self.sims, self.human = kind, sims, human
        self.seed = random.randrange(1 << 31) if seed is None else seed
        self.m = Match(self.deck, self.deck, self.seed)
        who = f"the network {self.checkpoint}" if kind == "network" else "the heuristic agent"
        self.log = [f"Game {self.seed}: you are seat {human}, against {who} at {sims} simulations a move."]
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
            t = self.agents[self.kind].think(self.m, self.sims, random.randrange(1 << 31))
            self.thoughts.append(t)
            self.log.append(f"Opponent: {self.m.choices(self.human)[t.chose]}")
            self.m.apply(t.chose)

    def ask(self):
        """The agent's search of the person's decision, from what the person can see."""
        if self.m is not None and not self.m.is_over() and self.m.to_act() == self.human:
            self.advice = self.agents[self.kind].think(self.m, self.sims, random.randrange(1 << 31))

    def state(self, debug: bool) -> dict:
        """What the page shows; the agent's hidden information only with `debug`."""
        if self.m is None:
            return {"status": "Choose an agent and start a game.", "board": "", "choices": [], "log": []}
        over = self.m.is_over()
        if not over:
            status = "Your decision"
        elif self.m.outcome() == f"win {self.human}":
            status = "You won!"
        elif self.m.outcome().startswith("win"):
            status = "The agent won."
        else:
            status = f"Game over: {self.m.outcome()}"
        s = {"status": status, "board": self.m.render(self.human), "choices": [] if over else self.m.choices(self.human),
             "log": self.log[-200:], "agent": self.kind}
        if debug:
            s["debug"] = {"agent_board": self.m.render(1 - self.human), "thoughts": [asdict(t) for t in self.thoughts],
                          "advice": asdict(self.advice) if self.advice else None}
        return s
