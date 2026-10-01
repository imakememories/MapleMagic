# MapleMagic

An AlphaZero-style agent for Magic: The Gathering, trained by self-play on a
fast Rust rules engine. It compares three ways of searching a game with
hidden cards: information-set MCTS, MAPLE
([arXiv 2605.24139](https://arxiv.org/abs/2605.24139)) and AlphaZe\*\*, the
paper's baseline.

The engine is trimmed from [mtg-kernel](https://github.com/jackmaiorino/mtg-kernel)
(MIT; see `LICENSE-mtg-kernel.txt`).

- `engine/`: the rules engine, covering nine Pauper decks.
- `mmz/`: the game as a list of choices per decision, the searches, the
  network's input encoding, and batched self-play.
- `mmz-py/`: Python bindings, installed as `mymagezero._core`.
- `python/mymagezero/`: the network, the training loop and the CLI.
- `configs/`: training and benchmark settings.

## Build

Requirements: Rust with the GNU toolchain, MinGW-w64 GCC at
`C:/Users/manni/tools/mingw64` (see `.cargo/config.toml`), and Python with
PyTorch (here the `rocm_env` conda environment, on an RX 7900 XTX).

```sh
export PATH="$HOME/.cargo/bin:/c/Users/manni/tools/mingw64/bin:$PATH"
PY=/c/Users/manni/miniconda3/envs/rocm_env/python.exe
PATH="$(dirname $PY):$PATH" cargo test --release --workspace --no-fail-fast
$PY -m maturin build --release -i $PY -o target/wheels
$PY -m pip install --force-reinstall --no-deps target/wheels/mymagezero-*.whl
```

## Use

```sh
$PY -m mymagezero.cli train configs/smoke.yaml --fresh    # a few minutes
$PY -m mymagezero.cli train configs/mid.yaml              # resumes from runs/<name>/latest.pt
$PY -m mymagezero.cli eval runs/mid_a/latest.pt --vs heuristic --config configs/bench.yaml
$PY -m mymagezero.cli eval runs/mid_a/latest.pt --vs runs/mid_b/latest.pt --sims 160 --opp-sims 68
```

- Training resumes with its replay buffer and optimizer state, and logs each
  generation's held-out losses: the network's loss on new self-play games
  before it trains on them.
- `eval` searches each side the way its checkpoint was trained; flags such as
  `--maple-worlds` / `--opp-maple-worlds` override either side. `--out`
  saves every game's decks and result.
- Network evaluations are split into passes sized to the machine (printed as
  `eval plan:`). On Windows the GPU allocator's cache counts against system
  commit, and an unbounded cache froze the PC, so it gets a quarter of the
  smaller of free GPU memory and free commit. `max_eval_batch` fixes the pass
  size.
- `cargo run --release -p mmz --example arena -- h200 h200p5 Burn Rally 400`
  plays two agents without a network: `h<sims>` is IS-MCTS with a hand-written
  evaluation, with `m<k>` for MAPLE over k worlds (`u` for union selection) or
  `p<k>` for AlphaZe\*\* over k worlds.

## The decks

Every match and every training run plays all nine decks against each other.

| Deck | Colors | Plan | Key cards |
|---|---|---|---|
| Wildfire | Black-red-green | Ramp and removal off artifact lands | Cleansing Wildfire, Writhing Chrysalis, Krark-Clan Shaman, Cast Down |
| Rally | Red | Goblin swarm and a big pump | Rally at the Hornburg, Goblin Bushwhacker, Burning-Tree Emissary, Lightning Bolt |
| Affinity | Blue-black-red | Cheap artifacts | Myr Enforcer, Thoughtcast, Galvanic Blast, artifact lands |
| Elves | Green | Mana elves into a wide board | Priest of Titania, Timberwatch Elf, Lead the Stampede, Avenging Hunter |
| Spy | Green-black | Mill itself, then reanimate | Balustrade Spy, Dread Return, Lotleth Giant, Land Grant |
| Burn | Red | Damage spells to the face | Lightning Bolt, Fireblast, Guttersnipe, Lava Dart |
| Terror | Blue | Cheap cantrips and counters, then a big flyer | Tolarian Terror, Counterspell, Brainstorm, Ponder |
| CawGates | White-blue | Gates, flyers and counters | Squadron Hawk, Basilisk Gate, Counterspell, Journey to Nowhere |
| Faeries | Blue | Flash flyers, counters and ninjas | Spellstutter Sprite, Ninja of the Deep Hours, Counterspell |

Decklists are in `data/runtime_decks_v1.json`. Each player sees its own hand,
the board, the graveyards, the opponent's hand size and any card revealed to
it. Both decklists are
known, so a guess at the hidden cards draws from exactly the cards that
remain.

## How the searches work

All three share one idea: since the opponent's hand and both libraries are
hidden, the search plays out guessed **worlds**, in which every card the
searching player can't see is dealt at random from the cards it could be.

- **IS-MCTS** keeps one tree, reached by the sequence of choices from the
  root. Each simulation deals a fresh world and walks down the tree choosing
  only among the choices legal in that world. With $L$ the legal choices,
  $N$ visits, $Q$ the mean value, $P$ the network's prior and $A$ how often a
  choice was legal at that node:

  $$a^* = \arg\max_{a \in L}\; Q(s,a) + P(s,a)\,\frac{\sqrt{A(s,a)}}{1 + N(s,a)}$$

  The leaf is evaluated once, so a simulation costs one network evaluation.
- **MAPLE-r** deals 5 worlds once per move. Each simulation plays its path in
  all of them, dropping worlds where a choice is illegal, and evaluates the
  leaf in every surviving world. A choice's prior is its mean over the worlds
  where it is legal, and the value is the mean over worlds. Worlds the
  network can't tell apart are evaluated once, so a simulation costs about
  2.4 evaluations. "-r" means the worlds are random; the paper's full MAPLE
  picks them with a learned network, which isn't implemented. Where worlds
  disagree on who acts or what is legal, MAPLE-r follows one random world.
  The **union** variant offers every choice legal in any world, as the paper
  does, and lets the majority's player act.
- **MAPLE-r seen hand** also shows the network each world's guessed opponent
  hand, as the paper's networks see the guessed board. Every world then looks
  different, so a simulation costs about 3.8 evaluations.
- **AlphaZe\*\*** deals 5 worlds once per move and grows an independent tree in
  each, sharing the simulations. The move comes from the average of the five
  trees' visit distributions.

In self-play the move is sampled in proportion to visits for the first 40
decisions of a game, and the visit distribution is the policy target.

## The models

All five networks have the same size (1.4M parameters: 160 wide, 3 layers)
and were trained the same way, from the same seed: 20 generations of 256
self-play games over all 81 deck pairings, with a 200,000-position replay
buffer. Only the search differs.

| Model | Config | Simulations per move | Network evaluations per move | Training time |
|---|---|---|---|---|
| IS-MCTS | `configs/mid.yaml` | 64 | 64 | 1.9 h |
| MAPLE-r | `configs/mid_b.yaml` | 64 | 152 | 4.3 h |
| MAPLE-r union | `configs/mid_bu.yaml` | 64 | 139 | 3.6 h |
| MAPLE-r seen hand | `configs/mid_c.yaml` | 64 | 233 | 4.6 h |
| AlphaZe\*\* | `configs/mid_d.yaml` | 160 (5 × 32) | 162 | 5.8 h |

## Benchmark

**Question:** with the same computing power per move, which model plays best?

**How the comparison is kept fair:**
- **Equal compute.** Network evaluations are the expensive part of a search,
  so each model gets about 160 per move. That means 160 simulations for
  IS-MCTS and AlphaZe\*\* (5 trees of 32), 68 for MAPLE-r and MAPLE-r union,
  and 42 for MAPLE-r seen hand. The heuristic agent, a fixed reference,
  runs IS-MCTS with a hand-written evaluation at 200 simulations.
- **Duplicate deals.** Games come in pairs with the same shuffled decks and
  the same seat going first; the two models swap decks between them. A strong
  deck or a lucky draw helps both models equally.
- **Every deck pairing.** A match plays each of the 45 pairings of the nine
  decks, mirrors included, three times as a duplicate pair: 270 games. Every
  match uses the same deals.
- **Round robin.** Every model plays every other model and the heuristic
  agent: 15 matches, 4,050 games.

A score is wins plus half of draws, divided by games. ± is a 95% interval,
$1.96\sqrt{s(1-s)/n}$, which is conservative here: duplicate deals cancel
some of the luck it allows for.

### Results

| Model | Network evaluations per move | Score against the heuristic agent | Score against the other four models | Rating |
|---|---|---|---|---|
| IS-MCTS | 158 | 0.322 ± 0.056 | **0.531 ± 0.030** | −127 [−152, −103] |
| MAPLE-r | 162 | 0.322 ± 0.056 | 0.494 ± 0.030 | −144 [−172, −122] |
| MAPLE-r union | 153 | 0.300 ± 0.055 | 0.495 ± 0.030 | −146 [−172, −120] |
| MAPLE-r seen hand | 162 | 0.296 ± 0.054 | 0.493 ± 0.030 | −148 [−174, −124] |
| AlphaZe\*\* | 158 | 0.281 ± 0.054 | 0.486 ± 0.030 | −153 [−179, −128] |

The rating is a Bradley–Terry fit over all 4,050 games on the Elo scale,
with the heuristic agent at 0 and a 95% bootstrap interval.

Head to head, score of the row's model against the column's:

| | IS-MCTS | MAPLE-r | MAPLE-r union | MAPLE-r seen hand | AlphaZe\*\* | Heuristic |
|---|---|---|---|---|---|---|
| IS-MCTS | — | 0.526 | 0.507 | 0.533 | 0.559 | 0.322 |
| MAPLE-r | 0.474 | — | 0.496 | 0.496 | 0.511 | 0.322 |
| MAPLE-r union | 0.493 | 0.504 | — | 0.489 | 0.496 | 0.300 |
| MAPLE-r seen hand | 0.467 | 0.504 | 0.511 | — | 0.489 | 0.296 |
| AlphaZe\*\* | 0.441 | 0.489 | 0.504 | 0.511 | — | 0.281 |
| Heuristic | 0.678 | 0.678 | 0.700 | 0.704 | 0.719 | — |

Each cell is 270 games, ±0.060 at most.

**What the results say:**
- **IS-MCTS is the strongest model, by a small margin.** It is the only one
  that scores above even against the others (0.531 ± 0.030, just outside
  noise). Its biggest margin is over AlphaZe\*\*, 0.559 ± 0.059, at the edge
  of significance. No other match separates any two models.
- **MAPLE-r is no better than AlphaZe\*\*.** They tie at 0.511 ± 0.060. In
  the paper, MAPLE-r was ahead of AlphaZe\*\* on Phantom Go and behind on Dark
  Hex; only the learned world sampler, which isn't implemented here, won on
  both.
- **The MAPLE-r variants make no difference.** The union rule ties the default
  (0.496), and showing the network the guessed hand doesn't help either
  (0.496 against MAPLE-r).
- **The heuristic agent beats every model:** they score 0.28–0.32 against
  it. Twenty generations of 256 self-play games weren't enough for any
  network to beat a hand-written evaluation searching 200 simulations.
- A likely reason IS-MCTS leads: it deals a fresh guess at the hidden cards
  every simulation, 160 per move, while MAPLE-r and AlphaZe\*\* commit to five
  guesses for the whole move.

### Without a network

The same searches with the heuristic agent's hand-written evaluation, at
about 200 evaluations per move: 200 simulations for IS-MCTS and AlphaZe\*\*
(5 trees of 40), 75 for MAPLE-r and 85 for MAPLE-r union. Each match is 500
games, in duplicate pairs, 100 on each of Burn–Rally, Faeries–Elves,
Terror–CawGates, Spy–Affinity and Wildfire–Faeries.

| Match | Score |
|---|---|
| AlphaZe\*\* against IS-MCTS | 0.412 ± 0.043 |
| MAPLE-r against IS-MCTS | 0.438 ± 0.043 |
| MAPLE-r union against IS-MCTS | 0.418 ± 0.043 |
| MAPLE-r against AlphaZe\*\* | 0.512 ± 0.044 |

Without a network the ordering is clearer, and it matches the networks':
IS-MCTS beats all three alternatives, and MAPLE-r ties AlphaZe\*\*.

### By deck

Each model's score with each deck, over all its matches, and each deck's win
rate in all non-mirror games:

| Deck | All games | IS-MCTS | MAPLE-r | MAPLE-r union | MAPLE-r seen hand | AlphaZe\*\* | Heuristic |
|---|---|---|---|---|---|---|---|
| Rally | 0.714 | 0.65 | 0.59 | 0.76 | 0.45 | 0.62 | 0.95 |
| Terror | 0.704 | 0.60 | 0.69 | 0.53 | 0.62 | 0.69 | 0.85 |
| Faeries | 0.604 | 0.59 | 0.47 | 0.52 | 0.52 | 0.53 | 0.87 |
| Elves | 0.529 | 0.55 | 0.52 | 0.51 | 0.51 | 0.36 | 0.69 |
| Burn | 0.501 | 0.49 | 0.54 | 0.59 | 0.47 | 0.31 | 0.61 |
| Affinity | 0.463 | 0.51 | 0.46 | 0.42 | 0.47 | 0.39 | 0.57 |
| Wildfire | 0.403 | 0.40 | 0.43 | 0.23 | 0.39 | 0.45 | 0.64 |
| CawGates | 0.308 | 0.33 | 0.23 | 0.39 | 0.29 | 0.24 | 0.59 |
| Spy | 0.274 | 0.29 | 0.20 | 0.15 | 0.37 | 0.43 | 0.47 |

- The deck matters more than the model. Rally and Terror win about 70% of
  their games, Spy and CawGates under a third.
- Each cell for a model is 150 games, ±0.08, so differences between models on
  one deck are mostly noise. The heuristic agent is ahead with every deck.

### Caveats

- **Scale.** One training run per model, 20 generations, with a small
  network.
- **Training compute wasn't equal.** IS-MCTS trained with 64 network
  evaluations per move, the others with 139–233. The benchmark equalizes
  compute only at play time.
- **Bugs fixed after training.** IS-MCTS's training games used a search that
  left some choices with too high a prior. Every model also trained while two
  copies of a card that differed in damage or combat counted as one choice.
  The benchmark uses the fixed code.
- **AlphaZe\*\*'s training was paused twice** and lost its replay buffer each
  time, so two of its generations trained on about 70,000 positions instead
  of 200,000. Training now saves the buffer.
- **The value heads may have memorized their games.** Training value loss
  ended at 0.05–0.08 on win/loss targets, too low for honest prediction from
  early in a game. These runs didn't log the held-out losses that would show
  it; training now does.

## Reproduce

```sh
for c in mid mid_b mid_bu mid_c mid_d; do $PY -m mymagezero.cli train configs/$c.yaml --fresh; done
# One match: each side's simulations per move, as in the table above.
$PY -m mymagezero.cli eval runs/mid_a/latest.pt --vs runs/mid_b/latest.pt --sims 160 --opp-sims 68 \
    --games 270 --config configs/bench.yaml --out runs/bench/ismcts-maple.json
$PY -m mymagezero.cli eval runs/mid_b/latest.pt --vs heuristic --sims 68 \
    --games 270 --config configs/bench.yaml --out runs/bench/maple-heuristic.json
# Without a network, at about 200 evaluations per move:
cargo run --release -p mmz --example arena -- h75m5 h200 Burn Rally 100
```

The benchmark was run at commit `8488808`. One training or benchmark job at a
time: a MAPLE-r or AlphaZe\*\* training run takes 4–6 hours on an RX 7900 XTX
with a 32-thread CPU, and IS-MCTS about 2.

## Engine fixes over upstream mtg-kernel

- Ninjutsu no longer halts when a returned attacker re-enters combat: stale
  combat entries are cleared.
- A trigger with no legal targets is removed (rule 603.3d) instead of asking
  for a target that doesn't exist.
- Cast triggers (Writhing Chrysalis) resolve even if the spell was countered
  (rule 113.7a).
- Ward works when two stack items target the same permanent.

`scripts/upstream_diff.sh <mtg-kernel clone>` compares `engine/` with
upstream from the commit in `scripts/UPSTREAM_REV`.
