"""One game between a person and an agent, as the page shows it."""
from __future__ import annotations

import json
import random
import time
from collections import OrderedDict
from dataclasses import asdict

import torch

from .. import DECK_PAIRS, Match
from .agent import Agent, HeuristicAgent, NetworkAgent, RandomAgent, Search, Thought
from .models import Catalog

KINDS = ("network", "heuristic", "random")
# Networks kept loaded, so going back to one is quick.
NETWORKS_KEPT = 3
# The steps a player gets priority in, in turn order.
STEPS = ("Upkeep", "Draw", "Main1", "BeginCombat", "DeclareAttackers", "DeclareBlockers", "CombatDamage", "EndCombat",
         "Main2", "End")
# Where the person stops to act when nothing is on the stack, by whose turn it is; elsewhere they pass.
STOPS = {"you": ("Main1", "Main2"), "opponent": ("DeclareAttackers", "End")}
# Combat steps whose stop only matters while a creature attacks.
COMBAT = ("DeclareAttackers", "DeclareBlockers", "CombatDamage")
# Choices left out of the log, by who made them: passes, and what the step's events already say.
UNLOGGED = {"you": {"Pass", "Attack", "NoAttack", "NoBlock", "Bottom", "Discard"},
            "opponent": {"Pass", "ManaAbility", "Attack", "NoAttack", "NoBlock", "FinishTargets", "Bottom", "Discard"}}
# Steps of the log kept for the page.
TIMELINE_SHOWN = 120


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
        self.intro = ""
        # The log, step by step: {turn, step, active, mulligans, entries}; entries are numbered across the game.
        self.timeline: list[dict] = []
        self.seq = 0
        # The first entry since the person's last request, what they chose in it, and anything to tell them.
        self.since = 0
        self.made: list[str] = []
        self.notice = ""
        # The agent's decisions since the person last chose, and its search of the person's decision.
        self.thoughts: list[Thought] = []
        self.advice: Thought | None = None
        # Auto-pass: stops by whose turn, passing while only your own spells are on the stack, and
        # (turn, whose turn) while passing to the next turn.
        self.stops = {k: set(v) for k, v in STOPS.items()}
        self.resolve_own = True
        self.passing: tuple[int, str] | None = None
        self.debug = False
        # Read by the page while a request runs (from another thread, so it's only ever replaced whole).
        self.progress: dict = {"busy": False, "version": 0}

    def network(self, ident: str) -> NetworkAgent:
        if ident not in self.networks:
            self.networks[ident] = NetworkAgent(str(self.catalog.path(ident)), self.device)
            while len(self.networks) > NETWORKS_KEPT:
                self.networks.popitem(last=False)
        self.networks.move_to_end(ident)
        return self.networks[ident]

    def prefs(self, body: dict):
        """The page's settings sent with each request: debug, stops and resolve_own."""
        self.debug = bool(body.get("debug"))
        stops = body.get("stops")
        if stops is not None:
            if not (isinstance(stops, dict) and all(isinstance(stops.get(k), list) for k in STOPS)):
                raise ValueError("stops: a list of steps for you and for the opponent")
            bad = {s for k in STOPS for s in stops[k]} - set(STEPS)
            if bad:
                raise ValueError(f"stops: unknown steps {sorted(bad)}")
            self.stops = {k: set(stops[k]) for k in STOPS}
        if "resolve_own" in body:
            self.resolve_own = bool(body["resolve_own"])

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
        self.intro = f"Game {seed}: you play {decks[0]} in seat {human}, against {who} playing {decks[1]} ({how})."
        self.timeline, self.seq, self.passing = [], 0, None
        self.begin()
        self.advance()

    def acting(self) -> bool:
        return self.m is not None and not self.m.is_over() and self.m.to_act() == self.human

    def begin(self):
        """A request of the person's: what follows is new to them."""
        self.since, self.made, self.notice = self.seq, [], ""
        self.thoughts, self.advice = [], None

    def board(self) -> dict:
        return json.loads(self.m.board(self.human))

    def choose(self, i: int):
        if not self.acting():
            return
        choices = self.m.choices(self.human)
        if not 0 <= i < len(choices):
            raise ValueError(f"choice {i} of {len(choices)}")
        self.begin()
        self.made.append(choices[i])
        self.apply(i, "you")
        self.advance()

    def attack(self, keys: list):
        """Declares the attack in one go: the creatures in `keys` attack, the others still asked don't."""
        if not self.acting():
            return
        plan = self.board().get("attack")
        if plan is None:
            raise ValueError("no attack to declare")
        bad = set(keys) - set(plan["asking"])
        if bad:
            raise ValueError(f"these can't attack now: {sorted(bad)}")
        self.begin()
        names = self.names()
        self.made.append(f"Attack with {', '.join(names[k] for k in keys)}" if keys else "No attack")
        while self.acting() and (plan := self.board().get("attack")) and plan["asking"]:
            self.apply(self.m.kinds().index("Attack" if plan["asking"][0] in keys else "NoAttack"), "you")
        if self.m.declaration_fell_back():
            self.notice = "Those attackers weren't legal together, so no attack was declared."
        self.advance()

    def block(self, pairs: list):
        """Declares the blocks in one go: `pairs` of [your blocker, the attacker it blocks]; the others
        still asked don't block. Stops where the rules need something else, and says why."""
        if not self.acting():
            return
        board = self.board()
        plan = board.get("block")
        if plan is None:
            raise ValueError("no blocks to declare")
        can = {b["key"]: set(b["attackers"]) for b in plan["asking"]}
        want = {}
        for pair in pairs:
            b, a = pair
            if a not in can.get(b, ()):
                raise ValueError(f"{b} can't block {a}")
            want[b] = a
        self.begin()
        names = self.names()
        self.made.append("; ".join(f"Block {names[a]} with {names[b]}" for b, a in want.items()) or "No blocks")
        while self.acting() and (board := self.board()).get("block") and board["block"]["asking"]:
            b = board["block"]["asking"][0]["key"]
            kinds, anchors = board["kinds"], board["anchors"]
            if b not in want:
                if "NoBlock" not in kinds:
                    self.notice = f"{names[b]} must block (an attacker must be blocked if able). Choose what it blocks."
                    return
                i = kinds.index("NoBlock")
            else:
                options = [i for i, k in enumerate(kinds) if k == "Block" and want[b] in anchors[i]]
                if not options:
                    self.notice = f"{names[b]} must block another attacker. Choose what it blocks."
                    return
                i = options[0]
            self.apply(i, "you")
        if self.m.declaration_fell_back():
            self.notice = ("Those blocks weren't legal together (menace or a blocking limit), "
                           "so only the blocks the rules require were declared.")
        self.advance()

    def pass_turn(self):
        """Passes until the next turn starts. Still stops for the opponent's spells, blocks, and choices
        that aren't passes; your attack is skipped."""
        if not self.acting():
            return
        b = self.board()
        self.begin()
        self.passing = (b["turn"], b["active"])
        self.made.append("Pass until the next turn")
        # This decision is passed whatever is on the stack; the next ones as the stops say.
        if "Pass" in b["kinds"]:
            self.apply(b["kinds"].index("Pass"), "you")
        self.advance()

    def names(self) -> dict:
        b = self.board()
        return {c["key"]: c["name"] for side in ("you", "opponent") for c in b[side]["battlefield"]}

    def auto_pass(self) -> int | None:
        """The person's choice when the stops say to pass, else None."""
        b = self.board()
        kinds = b["kinds"]
        if self.passing and (b["turn"], b["active"]) != self.passing:
            self.passing = None
        if "Pass" not in kinds:
            return kinds.index("NoAttack") if self.passing and "NoAttack" in kinds else None
        i = kinds.index("Pass")
        # Nothing to do but make mana.
        if all(k in ("Pass", "ManaAbility") for k in kinds):
            return i
        if any(item["controller"] == "opponent" for item in b["stack"]):
            return None
        if b["stack"]:
            return i if self.resolve_own else None
        if self.passing:
            return i
        if b["step"] not in self.stops[b["active"]]:
            return i
        attacking = any(c.get("attacking") for side in ("you", "opponent") for c in b[side]["battlefield"])
        return i if b["step"] in COMBAT and not attacking else None

    def advance(self):
        """Lets the agent move, and passes for the person where the stops say, until the person has a decision."""
        while not self.m.is_over():
            if self.m.to_act() == self.human:
                i = self.auto_pass()
                if i is None:
                    break
                self.apply(i, "you")
                continue
            self.publish(thinking={"decision": len(self.thoughts) + 1, "sims": 0, "total": self.search.sims}, snapshot=False)
            started, shown = time.perf_counter(), [0.0]

            def report(m: Match):
                now = time.perf_counter()
                if now - shown[0] < 0.1:
                    return
                shown[0] = now
                t = {"decision": len(self.thoughts) + 1, "sims": m.simulations(), "total": self.search.sims,
                     "seconds": now - started}
                if self.debug:
                    s = m.root_stats()
                    best = max(range(len(s["visits"])), key=s["visits"].__getitem__)
                    t["leaning"] = m.choices(m.to_act())[best]
                    t["share"] = s["visits"][best] / max(1.0, sum(s["visits"]))
                self.publish(thinking=t, snapshot=False)

            t = self.agent.think(self.m, self.search, random.randrange(1 << 31), report)
            self.thoughts.append(t)
            self.apply(t.chose, "opponent")

    def apply(self, i: int, who: str):
        """Makes choice i, logging it and what happened after."""
        turn, step, active, mulligans = self.m.phase()
        if self.m.kinds()[i] not in UNLOGGED[who]:
            self.log(self.group(turn, step, active, mulligans), {"who": who, "kind": "choice", "text": self.m.choices(self.human)[i]})
        self.m.apply(i)
        for s in json.loads(self.m.take_events(self.human)):
            key = self.group(s["turn"], s["step"], s["active"], s["turn"] == 1 and s["step"] == "Untap")
            for line in s["lines"]:
                self.log(key, line)
        self.publish()

    def group(self, turn: int, step: str, active: int | str, mulligans: bool) -> tuple:
        """A step's place in the log; `active` is a seat, or "you" / "opponent"."""
        if mulligans:
            return ("mulligans",)
        if isinstance(active, int):
            active = "you" if active == self.human else "opponent"
        return (turn, step, active)

    def log(self, key: tuple, entry: dict):
        if not self.timeline or self.timeline[-1]["key"] != key:
            g = {"key": key, "mulligans": True} if key == ("mulligans",) else {"key": key, "turn": key[0], "step": key[1], "active": key[2]}
            self.timeline.append({**g, "entries": []})
        self.timeline[-1]["entries"].append({**entry, "seq": self.seq})
        self.seq += 1

    def publish(self, thinking: dict | None = None, snapshot: bool = True):
        """What the page sees while the request runs: the board after each move, and how far the agent's search is."""
        p = {"busy": True, "version": self.progress["version"] + snapshot, "thinking": thinking,
             "snapshot": self.state(self.debug) if snapshot else self.progress.get("snapshot")}
        self.progress = p

    def ask(self):
        """The agent's search of the person's decision, from what the person can see."""
        if self.acting():
            self.advice = self.agent.think(self.m, self.search, random.randrange(1 << 31))

    def state(self, debug: bool) -> dict:
        """What the page shows; the agent's hidden information only with `debug`."""
        if self.m is None:
            return {"status": "Set up a game.", "choices": [], "timeline": [], "opponent": ""}
        over = self.m.is_over()
        acting = self.acting()
        if acting:
            status = "Your decision"
        elif not over:
            status = "The opponent is thinking..."
        elif self.m.outcome() == f"win {self.human}":
            status = "You won!"
        elif self.m.outcome().startswith("win"):
            status = "The agent won."
        else:
            status = f"Game over: {self.m.outcome()}"
        timeline = [{**{k: v for k, v in g.items() if k != "key"}, "entries": list(g["entries"])} for g in self.timeline[-TIMELINE_SHOWN:]]
        s = {"status": status, "board": self.board(), "choices": self.m.choices(self.human) if acting else [],
             "timeline": timeline, "intro": self.intro, "since": self.since, "made": self.made, "notice": self.notice,
             "passing": self.passing is not None, "agent": self.kind, "opponent": self.opponent}
        if debug:
            s["debug"] = {"agent_hand": json.loads(self.m.board(1 - self.human))["you"]["hand"],
                          "thoughts": [asdict(t) for t in self.thoughts], "advice": asdict(self.advice) if self.advice else None}
        return s
