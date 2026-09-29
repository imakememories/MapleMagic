# MyMageZero

AlphaZero-style MTG agent trained on a fast Rust rules engine. The engine is
trimmed from [mtg-kernel](https://github.com/jackmaiorino/mtg-kernel) (MIT; see
`LICENSE-mtg-kernel.txt`). The search is information-set MCTS, so hidden
information does not leak.

- `engine/`: the rules engine, reduced to its core. It covers the nine-deck
  Pauper pool: Wildfire, Rally, Affinity, Elves, Spy, Burn, Terror, CawGates
  and Faeries.
- `mmz/`: the layer the search and training use.
  - `game.rs`: turns each engine decision into a flat list of semantic
    choices. Spells are auto-paid from lands, and combat choices are made
    one creature at a time.
  - `ismcts.rs`: single-observer IS-MCTS with PUCT. Each simulation
    re-determinizes hidden cards. Child nodes are split by the player to
    act, because forced decisions are skipped and so the same choice path
    can reach either player's decision. It can also run MAPLE (see below).
  - `features.rs`: encodes observations for the network.
  - `selfplay.rs`: batched self-play over many games at once.
- `mmz-py/`: PyO3 bindings, installed as `mymagezero._core`.
- `python/mymagezero/`: the network, the training loop and the CLI.

## Build

Requirements:
- Rust with the GNU toolchain.
- MinGW-w64 GCC at `C:/Users/manni/tools/mingw64` (see `.cargo/config.toml`).
- Python with PyTorch. Here that is the `rocm_env` conda environment, which
  uses the RX 7900 XTX.

```sh
export PATH="$HOME/.cargo/bin:/c/Users/manni/tools/mingw64/bin:$PATH"
PY=/c/Users/manni/miniconda3/envs/rocm_env/python.exe
# The mmz-py test binary needs the Python DLL on PATH.
PATH="$(dirname $PY):$PATH" cargo test --release --workspace --no-fail-fast
$PY -m maturin build --release -i $PY -o target/wheels
$PY -m pip install --force-reinstall --no-deps target/wheels/mymagezero-*.whl
```

## Use

```sh
$PY -m mymagezero.cli train configs/smoke.yaml --fresh   # a few minutes
$PY -m mymagezero.cli train configs/pauper.yaml          # full pool
$PY -m mymagezero.cli eval runs/pauper/latest.pt --vs heuristic
```

GPU memory: network evaluations are split into passes that fit the machine,
and each run prints the plan it chose (`eval plan: ...`). At startup it
measures the network's memory per row, free GPU memory and, on Windows, free
system commit. Windows charges GPU allocations to commit, and an unbounded
PyTorch cache there can freeze the whole PC. The cache gets a quarter of the
smaller of the two free-memory figures. Set `max_eval_batch` in a config to
fix the pass size yourself.

### MAPLE search

MAPLE ([arXiv 2605.24139](https://arxiv.org/abs/2605.24139)) samples k
worlds once per search. It applies each simulation's path to all of them and
evaluates the leaf in every world that survives. Priors are averaged per
choice over the worlds where that choice is legal, and values are averaged.
It is off by default. Config keys:

- `maple_worlds: 5`: the world count (0 = plain IS-MCTS). The paper uses 5.
- `maple_select: ref | union`: how a node picks between worlds that
  disagree.
  - `ref` (the default) follows one random world's player and legal choices.
    It is unbiased when a choice depends on a lucky draw.
  - `union` is the paper's rule: the majority player, and every choice legal
    in any world.
- `maple_resample: true`: draw fresh worlds for every simulation.
- `perfect_obs: true`: the network sees the opponent's hand by identity (the
  guessed hand during search, the true one in training samples). This is the
  paper's setup. A checkpoint must be played with the setting it was
  trained with.

Checkpoints record these settings, and `eval` uses each side's own settings
unless flags override them:

```sh
$PY -m mymagezero.cli eval runs/b/latest.pt --vs self --maple-worlds 5 --opp-maple-worlds 0
$PY -m mymagezero.cli eval runs/b/latest.pt --vs runs/a/latest.pt --opp-sims 240   # equal network evaluations
```

`eval` prints the network evaluations per side, and for MAPLE how many
worlds reached each leaf and how many were dropped. Identical leaf
observations are evaluated once, so MAPLE costs fewer than k evaluations per
simulation.

Rust-only tools:
- `cargo run --release -p mmz --example bench`: random-game throughput.
- `... --example arena -- h200 random Burn Rally 100`: play agents against
  each other. Agent A always plays the first deck, so run both deck orders
  to compare agents. Add `m<k>` for MAPLE (`h200m5`) and a trailing `u` for
  union selection (`h200m5u`).
- `scripts/upstream_diff.sh <mtg-kernel clone>`: compare `engine/` with
  upstream mtg-kernel, starting from the commit in `scripts/UPSTREAM_REV`.
- `... --example diag_halts -- Faeries Faeries 300`: find engine aborts.

## Engine fixes over upstream mtg-kernel

- Ninjutsu no longer halts when a returned attacker re-enters combat. Stale
  combat entries are cleared.
- A trigger with no legal targets is now removed (rule 603.3d) instead of
  asking for a target that doesn't exist.
- Cast triggers (Writhing Chrysalis) now resolve even if the spell was
  countered (rule 113.7a).
- Ward no longer refuses to work when two stack items target the same
  permanent.
