"""The networks a person can play against: every run folder's checkpoints, with
what the run trained on and how it scored, and each checkpoint's own details."""
from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import torch
import yaml

from .. import COMMANDER_DECKS, DECK_PAIRS
from ..train import Config

CHECKPOINT = re.compile(r"^(latest|gen\d{4}|gen0_start|gen0_trained)\.pt$")
# Shown for each evaluated generation.
EVAL_KEYS = ("vs_heuristic", "vs_prev", "heldout_value_auc")
# Shown for the chosen generation.
GEN_KEYS = ("games", "decisions_per_game", "policy_loss", "value_loss", "heldout_policy_loss", "heldout_value_loss",
            "heldout_value_auc", "vs_heuristic", "vs_prev")


def gen_of(file: str) -> int | None:
    m = re.match(r"gen(\d{4})\.pt$", file)
    return int(m.group(1)) if m else None


def read_config(folder: Path) -> Config | None:
    """The run's config; keys this version no longer has are dropped."""
    try:
        raw = yaml.safe_load((folder / "config.yaml").read_text()) or {}
    except (OSError, yaml.YAMLError):
        return None
    known = {f.name for f in dataclasses.fields(Config)}
    return Config(**{k: v for k, v in raw.items() if k in known})


def read_log(folder: Path) -> list[dict]:
    try:
        lines = (folder / "log.jsonl").read_text().splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    return [r for r in rows if isinstance(r.get("gen"), int)]


def as_trained(meta_search: dict | None) -> dict:
    """How a checkpoint's run searched, as eval reads it: missing keys take today's defaults,
    except collisions, which a checkpoint without the key never used."""
    s = {**Config().search_settings(), **(meta_search or {})}
    s["collisions"] = (meta_search or {}).get("collisions", False)
    return s


class Catalog:
    """Run folders under `root`, plus the folder of the default checkpoint if it is elsewhere."""

    def __init__(self, root: Path, default: Path):
        self.root, self.default = root, default.resolve()
        self.folders: dict[str, Path] = {}
        self.cache: dict[str, dict] = {}

    def default_id(self) -> str:
        return f"{self.default.parent.name}/{self.default.name}"

    def runs(self) -> list[dict]:
        """Every run with a config and a checkpoint: Commander runs first, then Pauper, newest first in each."""
        folders = sorted(p for p in self.root.iterdir() if p.is_dir()) if self.root.is_dir() else []
        if self.default.is_file() and self.default.parent not in [f.resolve() for f in folders]:
            folders.append(self.default.parent)
        self.folders, out = {}, []
        for folder in folders:
            run = self.summary(folder)
            if run is not None:
                self.folders[run["name"]] = folder
                out.append(run)
        out.sort(key=lambda r: (r["format"] != "Commander", -r["modified"]))
        return out

    def summary(self, folder: Path) -> dict | None:
        files = {p.name for p in folder.glob("*.pt") if CHECKPOINT.match(p.name)}
        cfg = read_config(folder) if files else None
        if cfg is None:
            return None
        try:
            pairings = [list(p) for p in cfg.pairing_list() if tuple(p) in DECK_PAIRS]
        except (TypeError, ValueError):
            pairings = []
        if not pairings:
            return None
        log = read_log(folder)
        evals = {r["gen"]: r for r in log if "vs_heuristic" in r}
        last = max((r["gen"] for r in log), default=None)
        checkpoints = []
        if "latest.pt" in files:
            checkpoints.append({"file": "latest.pt", "gen": last, "label": "latest" + (f" (gen {last})" if last is not None else "")})
        for f in sorted((f for f in files if gen_of(f) is not None), key=gen_of, reverse=True):
            e = evals.get(gen_of(f))
            checkpoints.append({"file": f, "gen": gen_of(f),
                                "label": f"gen {gen_of(f)}" + (f", vs heuristic {e['vs_heuristic']:.2f}" if e else "")})
        for f in ("gen0_trained.pt", "gen0_start.pt"):
            if f in files:
                checkpoints.append({"file": f, "gen": None, "label": f.removesuffix(".pt").replace("_", " ")})
        best = max(evals.values(), key=lambda r: r["vs_heuristic"], default=None)
        commander = all(d in COMMANDER_DECKS for p in pairings for d in p)
        return {
            "name": folder.name,
            "format": "Commander" if commander else "Pauper",
            "decks": sorted({d for p in pairings for d in p}),
            "pairings": pairings,
            "generations": cfg.generations,
            "trained": None if last is None else last + 1,
            "best": best and {"gen": best["gen"], "vs_heuristic": best["vs_heuristic"]},
            "checkpoints": checkpoints,
            "modified": max((folder / f).stat().st_mtime for f in files),
        }

    def path(self, ident: str) -> Path:
        """A checkpoint's file from its id ("run/file"); only files the catalog lists."""
        run, _, file = ident.partition("/")
        if not self.folders:
            self.runs()
        folder = self.folders.get(run)
        if folder is None or not CHECKPOINT.match(file) or not (folder / file).is_file():
            raise ValueError(f"no checkpoint {ident!r}")
        return folder / file

    def details(self, ident: str) -> dict:
        """One checkpoint: its network, how its run trained and searched, and the run's evaluations."""
        if ident in self.cache:
            return self.cache[ident]
        path = self.path(ident)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        folder = path.parent
        cfg = read_config(folder)
        log = read_log(folder)
        gen = ck.get("gen")
        row = next((r for r in log if r["gen"] == gen), {})
        search = as_trained(ck.get("search"))
        notes = []
        if search.get("world_sampler") not in ("uniform", "oracle"):
            notes.append(f"Trained with {search['world_sampler']} worlds, which need its sampler network; it plays here with uniform worlds.")
            search["world_sampler"] = "uniform"
        if search.get("perfect_obs"):
            notes.append("Trained seeing hidden cards (perfect observation): in a search it sees the hidden cards of each sampled world.")
        out = {
            "id": ident,
            "run": folder.name,
            "file": path.name,
            "gen": gen,
            "network": {**ck.get("config", {}), "parameters": sum(t.numel() for t in ck["state"].values())},
            "training": {
                "pairings": [list(p) for p in cfg.pairing_list()] if cfg else [],
                "generations": cfg and cfg.generations,
                "games_per_gen": cfg and cfg.games_per_gen,
                "sims": cfg and cfg.sims,
                "value_target": cfg and cfg.value_target,
                "policy_target": cfg and cfg.policy_target,
            },
            "search": {k: search[k] for k in ("collisions", "maple_worlds", "maple_select", "maple_resample",
                                              "pimc_worlds", "perfect_obs", "world_sampler", "value_mix")},
            "this_gen": {k: row[k] for k in GEN_KEYS if k in row},
            "evals": [{"gen": r["gen"], **{k: r.get(k) for k in EVAL_KEYS}} for r in log if "vs_heuristic" in r],
            "notes": notes,
        }
        self.cache[ident] = out
        return out
