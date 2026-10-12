"""Play a game against an agent in the terminal: `maplemagic play [checkpoint]`."""
from __future__ import annotations

import random

import numpy as np
import torch

from . import Match
from .model import evaluate, load


class Agent:
    """Chooses with a network-guided search, or with the heuristic agent's search."""

    def __init__(self, checkpoint: str | None, sims: int, device):
        self.net = load(checkpoint, device).eval() if checkpoint else None
        self.sims, self.device = sims, device

    def choose(self, m: Match, seed: int) -> tuple[int, str]:
        if self.net is None:
            return m.heuristic_choice(self.sims, seed), ""
        m.start_search(seed)
        while m.simulations() < self.sims:
            batch = m.gather(8)
            if batch is None:
                continue
            tok, tl, act, al = batch
            priors, values = evaluate(self.net, tok, tl, act, al, self.device)
            m.feed(priors.astype(np.float32), values.astype(np.float32))
        best, shares, value = m.best()
        return best, f"(it searched {self.sims} simulations, chose with {shares[best]:.0%} of them, and rates its position {value:+.2f})"


def ask(choices: list[str]) -> int | None:
    for i, c in enumerate(choices):
        print(f"  {i:>2}. {c}")
    while True:
        raw = input("Your choice (number, or q to quit): ").strip().lower()
        if raw in ("q", "quit", "exit"):
            return None
        if raw.isdigit() and int(raw) < len(choices):
            return int(raw)
        print(f"Enter a number from 0 to {len(choices) - 1}.")


def play(checkpoint: str | None, sims: int, seat: str, seed: int | None, deck: str, show_thinking: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(checkpoint, sims, device)
    seed = random.randrange(1 << 31) if seed is None else seed
    human = random.randrange(2) if seat == "random" else int(seat)
    m = Match(deck, deck, seed)
    who = f"network {checkpoint}" if checkpoint else "the heuristic agent"
    print(f"Game {seed}: you are seat {human} against {who} at {sims} simulations a move.\n")
    while not m.is_over():
        if m.to_act() == human:
            print("=" * 72)
            print(m.render(human), end="")
            print("Your decision:")
            i = ask(m.choices(human))
            if i is None:
                print("Game abandoned.")
                return
            m.apply(i)
        else:
            i, thinking = agent.choose(m, random.randrange(1 << 31))
            text = m.choices(human)[i]
            print(f"  Opponent: {text}" + (f"  {thinking}" if show_thinking and thinking else ""))
            m.apply(i)
    print("=" * 72)
    print(m.render(human), end="")
    result = m.outcome()
    if result == f"win {human}":
        print("You won!")
    elif result.startswith("win"):
        print("The agent won.")
    else:
        print(f"Game over: {result}")
