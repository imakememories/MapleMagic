"""mmz train <config.yaml> | mmz eval <checkpoint> [--vs heuristic|random|<checkpoint>]"""
from __future__ import annotations

import argparse

import torch


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mmz")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="run the self-play training loop")
    t.add_argument("config")
    t.add_argument("--fresh", action="store_true", help="ignore latest.pt and start over")
    e = sub.add_parser("eval", help="play a checkpoint against an opponent")
    e.add_argument("checkpoint")
    e.add_argument("--vs", default="heuristic", help="heuristic | random | path to checkpoint")
    e.add_argument("--games", type=int, default=256)
    e.add_argument("--sims", type=int, default=96)
    e.add_argument("--config", default=None, help="config for pairings/search settings")
    a = ap.parse_args(argv)

    from .train import Config, match, run
    if a.cmd == "train":
        run(Config.from_yaml(a.config), resume=not a.fresh)
    elif a.cmd == "eval":
        from .model import load
        cfg = Config.from_yaml(a.config) if a.config else Config()
        cfg.eval_sims = a.sims
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = load(a.checkpoint, device)
        opp = a.vs if a.vs in ("heuristic", "random") else load(a.vs, device)
        r = match(cfg, net, opp, a.games, device, seed=12345)
        print(f"score {r['score']:.3f} ± {r['ci95']:.3f}  (W{r['w']} L{r['l']} D{r['d']}, {r['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
