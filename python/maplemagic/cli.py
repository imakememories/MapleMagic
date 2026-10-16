"""maplemagic train <config.yaml> | maplemagic pretrain <config.yaml> | maplemagic play [checkpoint] | maplemagic eval <checkpoint> [--vs ...] | maplemagic sampler-pilot <checkpoint> | maplemagic yardstick <checkpoint>..."""
from __future__ import annotations

import argparse
import json

import torch


def main(argv=None):
    ap = argparse.ArgumentParser(prog="maplemagic")
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
    e.add_argument("--pimc-worlds", type=int, default=None, help="PIMC worlds for the checkpoint (0 = off)")
    e.add_argument("--opp-pimc-worlds", type=int, default=None, help="PIMC worlds for the opponent network")
    e.add_argument("--maple-select", choices=["ref", "union"], default=None, help="MAPLE selection for the checkpoint")
    e.add_argument("--opp-maple-select", choices=["ref", "union"], default=None, help="MAPLE selection for the opponent network")
    e.add_argument("--maple-resample", action="store_true", default=None, help="fresh MAPLE worlds every simulation, for the checkpoint")
    e.add_argument("--opp-maple-resample", action="store_true", default=None, help="the same for the opponent network")
    samplers = ["uniform", "oracle", "retention", "scored"]
    e.add_argument("--world-sampler", choices=samplers, default=None, help="opponent hands for the checkpoint's worlds")
    e.add_argument("--opp-world-sampler", choices=samplers, default=None, help="the same for the opponent network")
    e.add_argument("--world-candidates", type=int, default=None)
    e.add_argument("--world-pool", type=int, default=None)
    e.add_argument("--world-temp", type=float, default=None)
    e.add_argument("--sampler-from", default=None,
                   help="sampler file for the checkpoint's scored or retention worlds (scored default: its own)")
    e.add_argument("--opp-sampler-from", default=None, help="the same for the opponent network")
    e.add_argument("--leaves-per-step", type=int, default=None, help="the checkpoint's leaves per step (default: config's)")
    e.add_argument("--opp-leaves-per-step", type=int, default=None, help="the same for the opponent")
    e.add_argument("--collisions", action=argparse.BooleanOptionalAction, default=None,
                   help="stop simulations at nodes awaiting evaluation, for the checkpoint (default: as trained)")
    e.add_argument("--opp-collisions", action=argparse.BooleanOptionalAction, default=None,
                   help="the same for the opponent network")
    e.add_argument("--value-scale", type=float, default=None, help="multiply the checkpoint's leaf values")
    e.add_argument("--opp-value-scale", type=float, default=None, help="the same for the opponent network")
    e.add_argument("--heuristic-values", action="store_true", default=None,
                   help="leaf values from the heuristic, priors from the checkpoint")
    e.add_argument("--opp-heuristic-values", action="store_true", default=None, help="the same for the opponent network")
    e.add_argument("--value-mix", type=float, default=None,
                   help="weight of the checkpoint's leaf values against the heuristic's (1 = network only)")
    e.add_argument("--opp-value-mix", type=float, default=None, help="the same for the opponent network")
    e.add_argument("--seed", type=int, default=12345)
    e.add_argument("--out", default=None, help="write the settings and every game's result to this JSON file")
    s = sub.add_parser("sampler-pilot", help="train world samplers on a frozen network's self-play")
    s.add_argument("checkpoint")
    s.add_argument("--out", required=True)
    s.add_argument("--config", default="configs/bench.yaml")
    s.add_argument("--games", type=int, default=600)
    s.add_argument("--sims", type=int, default=64)
    s.add_argument("--keep", type=float, default=0.25, help="share of each game's decisions kept")
    s.add_argument("--seeds", type=int, default=2)
    pt = sub.add_parser("pretrain", help="generation 0 from heuristic self-play, shrunk and perturbed")
    pt.add_argument("config")
    pt.add_argument("--games", type=int, default=2000)
    pt.add_argument("--sims", type=int, default=800)
    pt.add_argument("--passes", type=float, default=2.0)
    pt.add_argument("--shrink", type=float, default=0.4)
    pt.add_argument("--perturb", type=float, default=0.1)
    pl = sub.add_parser("plasticity", help="a start checkpoint and a fresh net trained on the same self-play")
    pl.add_argument("config")
    pl.add_argument("start")
    pl.add_argument("--out", required=True)
    sp_ = sub.add_parser("shrink-perturb", help="shrink and perturb a checkpoint into a run's gen-0 start")
    sp_.add_argument("config")
    sp_.add_argument("trained")
    sp_.add_argument("--shrink", type=float, required=True)
    sp_.add_argument("--perturb", type=float, required=True)
    pg = sub.add_parser("play", help="play against an agent in the browser")
    pg.add_argument("checkpoint", nargs="?", default="runs/ognis_s6e/latest.pt", help="the network chosen at first")
    pg.add_argument("--runs", default="runs", help="the folder of runs whose checkpoints the page offers")
    pg.add_argument("--deck", default="Ognis", help="the deck chosen at first for games against the heuristic or random agent")
    pg.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to play from another device on your network")
    pg.add_argument("--port", type=int, default=8765)
    pg.add_argument("--no-browser", action="store_true", help="don't open the page")
    pg.add_argument("--art-dir", default=None, help="where card images from Scryfall are kept (default ~/.cache/maplemagic/art)")
    pg.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                    help="where the network runs; cpu leaves the GPU and its memory to a training run")
    y = sub.add_parser("yardstick", help="score value heads on fresh games their runs never trained on")
    y.add_argument("checkpoints", nargs="+")
    y.add_argument("--games", type=int, default=100)
    y.add_argument("--seed", type=int, default=777777)
    y.add_argument("--out", default="runs/yardstick")
    a = ap.parse_args(argv)

    from .train import Config, match, run, summarize_stats
    if a.cmd == "train":
        run(Config.from_yaml(a.config), resume=not a.fresh)
    elif a.cmd == "pretrain":
        from .train import pretrain
        pretrain(Config.from_yaml(a.config), a.games, a.sims, a.passes, a.shrink, a.perturb)
    elif a.cmd == "play":
        from pathlib import Path

        from .play import serve
        art = Path(a.art_dir) if a.art_dir else Path.home() / ".cache" / "maplemagic" / "art"
        serve(a.checkpoint, Path(a.runs), a.deck, a.host, a.port, not a.no_browser, art, a.device)
    elif a.cmd == "plasticity":
        from .train import plasticity
        plasticity(Config.from_yaml(a.config), a.start, a.out)
    elif a.cmd == "shrink-perturb":
        import os

        from .model import load, save
        from .train import shrink_perturb
        cfg = Config.from_yaml(a.config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = load(a.trained, device)
        shrink_perturb(net, cfg, a.shrink, a.perturb)
        out = os.path.join(cfg.out_dir, cfg.name)
        for name in ("latest.pt", "gen0_start.pt"):
            save(net, os.path.join(out, name), {"gen": -1, "search": cfg.search_settings()})
        print(f"wrote {out}/latest.pt with shrink {a.shrink}, perturb {a.perturb}")
    elif a.cmd == "eval":
        from .model import load, load_meta
        cfg = Config.from_yaml(a.config) if a.config else Config()
        cfg.eval_sims = a.sims
        for k in ("world_candidates", "world_pool", "world_temp"):
            if getattr(a, k) is not None:
                setattr(cfg, k, getattr(a, k))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def settings(path: str, overrides: dict) -> dict:
            meta = load_meta(path).get("search")
            # Keys a checkpoint lacks postdate it: they take the defaults, not --config's.
            s = {**Config().search_settings(), **meta} if meta is not None else cfg.search_settings()
            # A checkpoint without the key searched without collisions, whatever the default is now.
            s["collisions"] = (meta or {}).get("collisions", False)
            s.update({k: v for k, v in overrides.items() if v is not None})
            return s

        net = load(a.checkpoint, device)
        sa = settings(a.checkpoint, {"maple_worlds": a.maple_worlds, "pimc_worlds": a.pimc_worlds,
                                     "maple_select": a.maple_select, "maple_resample": a.maple_resample,
                                     "world_sampler": a.world_sampler, "value_scale": a.value_scale,
                                     "heuristic_values": a.heuristic_values, "value_mix": a.value_mix,
                                     "collisions": a.collisions})
        sb = path = None
        if a.vs in ("heuristic", "random"):
            opp = a.vs
        else:
            path = a.checkpoint if a.vs == "self" else a.vs
            opp = net if a.vs == "self" else load(path, device)
            sb = settings(path, {"maple_worlds": a.opp_maple_worlds, "pimc_worlds": a.opp_pimc_worlds,
                                 "maple_select": a.opp_maple_select, "maple_resample": a.opp_maple_resample,
                                 "world_sampler": a.opp_world_sampler, "value_scale": a.opp_value_scale,
                                 "heuristic_values": a.opp_heuristic_values, "value_mix": a.opp_value_mix,
                                 "collisions": a.opp_collisions})
        scorers, retention = {}, None
        for model_id, search, src, own in ((0, sa, a.sampler_from, a.checkpoint), (1, sb, a.opp_sampler_from, path)):
            kind = (search or {}).get("world_sampler", "uniform")
            if kind == "scored":
                # Without a sampler file, the side's own checkpoint's sampler.
                from .model import load_siamese, siamese_scorer
                try:
                    scorers[model_id] = siamese_scorer(load_siamese(src or own, device), device)
                except ValueError as err:
                    ap.error(f"scored worlds: {err}; pass a sampler file (--sampler-from / --opp-sampler-from)")
                print(f"model {model_id}: worlds scored by the sampler in {src or own}")
            elif kind == "retention":
                if src is None:
                    ap.error("retention worlds need a sampler file (--sampler-from / --opp-sampler-from)")
                retention = torch.load(src, map_location=device, weights_only=False)["retention"]
        lps = None
        if a.leaves_per_step is not None or a.opp_leaves_per_step is not None:
            lps = (a.leaves_per_step or cfg.leaves_per_step, a.opp_leaves_per_step or cfg.leaves_per_step)
        r = match(cfg, net, opp, a.games, device, seed=a.seed, search_a=sa, search_b=sb, opp_sims=a.opp_sims,
                  scorers=scorers, retention=retention, leaves_per_step=lps)
        print(r["plan"].describe())
        print(f"score {r['score']:.3f} ± {r['ci95']:.3f}  (W{r['w']} L{r['l']} D{r['d']}, {r['seconds']:.0f}s)")
        print(f"  A {json.dumps(sa)}: {r['evals_a']} evals, {json.dumps(summarize_stats(r['search_stats'][0]))}")
        if sb is not None:
            print(f"  B {json.dumps(sb)}: {r['evals_b']} evals, {json.dumps(summarize_stats(r['search_stats'][1]))}")
        if a.out:
            with open(a.out, "w") as f:
                json.dump({"a": a.checkpoint, "b": a.vs, "sims": a.sims, "opp_sims": a.opp_sims or a.sims,
                           "leaves_per_step": lps or (cfg.leaves_per_step,) * 2,
                           "search_a": sa, "search_b": sb, "evals_a": r["evals_a"], "evals_b": r["evals_b"],
                           "stats": [summarize_stats(x) for x in r["search_stats"]], "games": r["games"]}, f)


    elif a.cmd == "sampler-pilot":
        sampler_pilot(a)
    elif a.cmd == "yardstick":
        yardstick(a)


def yardstick(a):
    import os

    import numpy as np

    from . import yardstick as Y
    from .model import load
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out, exist_ok=True)
    names = [os.path.basename(os.path.dirname(c)) for c in a.checkpoints]
    data = {}
    for name, ck in zip(names, a.checkpoints):
        path = os.path.join(a.out, f"{name}.npz")
        if os.path.exists(path):
            data[name] = dict(np.load(path))
        else:
            data[name] = Y.collect(ck, a.games, a.seed, device)
            np.savez(path, **data[name])
        print(f"{name}: {len(data[name]['z'])} held-out positions", flush=True)
    report = {"own": {}, "cross": {}}
    for name, ck in zip(names, a.checkpoints):
        with open(os.path.join(os.path.dirname(ck), "log.jsonl")) as f:
            last = json.loads(f.readlines()[-1])
        net = load(ck, device)
        own = data[name]
        v, ce = Y.predict(net, own, device)
        report["own"][name] = {
            "train_value_loss": last["value_loss"], "heldout": Y.summary(v, own), "heuristic": Y.summary(own["heuristic"], own),
            "train_policy_loss": last["policy_loss"], "heldout_policy_loss": round(float(ce.mean()), 4),
        }
        report["cross"][name] = {}
        for other, d in data.items():
            if other != name and (bool(own["perfect_obs"]) or bool(d["perfect_obs"])):
                continue
            report["cross"][name][other] = Y.summary(v if other == name else Y.predict(net, d, device)[0], d)["auc"]
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print("| Run | Train value loss | Held-out value MSE | Held-out AUC | Heuristic AUC | Heuristic MSE "
          "| AUC turns 1-3 / 4-7 / 8+ | Train / held-out policy loss |")
    for name, r in report["own"].items():
        h, e = r["heldout"], r["heuristic"]
        print(f"| {name} | {r['train_value_loss']} | {h['mse']} | {h['auc']} | {e['auc']} | {e['mse']} "
              f"| {h['auc_turn_1-3']} / {h['auc_turn_4-7']} / {h['auc_turn_8+']} "
              f"| {r['train_policy_loss']} / {r['heldout_policy_loss']} |")
    print()
    print("Value AUC of each checkpoint (rows) on each run's held-out games (columns):")
    print("| | " + " | ".join(names) + " |")
    for name in names:
        print(f"| {name} | " + " | ".join(str(report["cross"][name].get(o, "-")) for o in names) + " |")


def sampler_pilot(a):
    import os

    import numpy as np

    from . import sampler as S
    from .train import Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "log.txt"), "a")

    def note(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    path = os.path.join(a.out, "data.npz")
    if os.path.exists(path):
        data = dict(np.load(path))
    else:
        data = S.collect(a.checkpoint, Config.from_yaml(a.config), a.games, a.sims, device, keep=a.keep)
        np.savez(path, **data)
    splits = S.split(data)
    note(json.dumps({"positions": len(data["tok_len"]), **{k: len(v) for k, v in splits.items()}}))
    retention = S.fit_retention(data, splits["train"])
    test = S.used_rows(data, splits["test"], seed=7)
    report = {"test_rows": len(test), "baselines": S.metrics(data, test, device, retention=retention)}
    note(json.dumps(report["baselines"]))
    for seed in range(a.seeds):
        snet, info = S.train(data, splits, device, seed=seed, log=note)
        m = S.metrics(data, test, device, snet=snet)["siamese"]
        report[f"siamese_seed{seed}"] = {"test": m, **info}
        note(json.dumps({"seed": seed, "test": m}))
        S.save(snet, retention, os.path.join(a.out, f"sampler_seed{seed}.pt"))
    snet, info = S.train(data, splits, device, seed=100, shuffled=True, log=note)
    report["shuffled_control"] = {"test": S.metrics(data, test, device, snet=snet)["siamese"], **info}
    note(json.dumps({"shuffled_control": report["shuffled_control"]["test"]}))
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
