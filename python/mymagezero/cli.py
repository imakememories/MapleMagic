"""mmz train <config.yaml> | mmz eval <checkpoint> [--vs heuristic|random|self|<checkpoint>]"""
from __future__ import annotations

import argparse
import json

import torch


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mmz")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="run the self-play training loop")
    t.add_argument("config")
    t.add_argument("--fresh", action="store_true", help="ignore latest.pt and start over")
    e = sub.add_parser("eval", help="play a checkpoint against an opponent")
    e.add_argument("checkpoint")
    e.add_argument("--vs", default="heuristic", help="heuristic | random | self | path to checkpoint")
    e.add_argument("--games", type=int, default=256)
    e.add_argument("--sims", type=int, default=96)
    e.add_argument("--opp-sims", type=int, default=None, help="the opponent network's sims (default: --sims)")
    e.add_argument("--config", default=None, help="config for pairings/search settings")
    # Each side searches the way its checkpoint was trained unless overridden.
    e.add_argument("--maple-worlds", type=int, default=None, help="MAPLE worlds for the checkpoint (0 = IS-MCTS)")
    e.add_argument("--opp-maple-worlds", type=int, default=None, help="MAPLE worlds for the opponent network")
    e.add_argument("--maple-select", choices=["ref", "union"], default=None)
    e.add_argument("--maple-resample", action="store_true", default=None)
    e.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args(argv)

    from .train import Config, match, run, summarize_stats
    if a.cmd == "train":
        run(Config.from_yaml(a.config), resume=not a.fresh)
    elif a.cmd == "eval":
        from .model import load, load_meta
        cfg = Config.from_yaml(a.config) if a.config else Config()
        cfg.eval_sims = a.sims
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def settings(path: str, worlds: int | None) -> dict:
            s = {**cfg.search_settings(), **load_meta(path).get("search", {})}
            if worlds is not None:
                s["maple_worlds"] = worlds
            if a.maple_select is not None:
                s["maple_select"] = a.maple_select
            if a.maple_resample is not None:
                s["maple_resample"] = a.maple_resample
            return s

        net = load(a.checkpoint, device)
        sa = settings(a.checkpoint, a.maple_worlds)
        sb = None
        if a.vs in ("heuristic", "random"):
            opp = a.vs
        else:
            path = a.checkpoint if a.vs == "self" else a.vs
            opp = net if a.vs == "self" else load(path, device)
            sb = settings(path, a.opp_maple_worlds)
        r = match(cfg, net, opp, a.games, device, seed=a.seed, search_a=sa, search_b=sb, opp_sims=a.opp_sims)
        print(r["plan"].describe())
        print(f"score {r['score']:.3f} ± {r['ci95']:.3f}  (W{r['w']} L{r['l']} D{r['d']}, {r['seconds']:.0f}s)")
        print(f"  A {json.dumps(sa)}: {r['evals_a']} evals, {json.dumps(summarize_stats(r['search_stats'][0]))}")
        if sb is not None:
            print(f"  B {json.dumps(sb)}: {r['evals_b']} evals, {json.dumps(summarize_stats(r['search_stats'][1]))}")


if __name__ == "__main__":
    main()
