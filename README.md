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
    re-determinizes hidden cards.
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
cargo test --release --workspace
PY=/c/Users/manni/miniconda3/envs/rocm_env/python.exe
$PY -m maturin build --release -i $PY -o target/wheels
$PY -m pip install --force-reinstall --no-deps target/wheels/mymagezero-*.whl
```

## Use

```sh
$PY -m mymagezero.cli train configs/smoke.yaml --fresh   # a few minutes
$PY -m mymagezero.cli train configs/pauper.yaml          # full pool
$PY -m mymagezero.cli eval runs/pauper/latest.pt --vs heuristic
```

Rust-only tools:
- `cargo run --release -p mmz --example bench`: random-game throughput.
- `... --example arena -- h200 random Burn Rally 100`: play agents against
  each other.
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
