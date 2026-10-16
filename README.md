# MapleMagic

An AlphaZero-style agent for Magic: The Gathering, trained by self-play on a
fast Rust rules engine. The current goal is the strongest bot for one
Commander deck, Ognis, the Dragon's Lash, in two-player games. The best
network beats a hand-written heuristic agent 0.828 ± 0.030, searching 160
simulations a move to its 200, and 0.870 ± 0.027 at 400.

Two benchmarks on nine Pauper decks compare ways of searching a game with
hidden cards: information-set MCTS, MAPLE
([arXiv 2605.24139](https://arxiv.org/abs/2605.24139)), with and without its
learned world sampler, and AlphaZe\*\*, the paper's baseline. Nothing beats
IS-MCTS in either.

The engine is trimmed from [mtg-kernel](https://github.com/jackmaiorino/mtg-kernel)
(MIT; see `LICENSE-mtg-kernel.txt`).

- `engine/`: the rules engine, covering nine Pauper decks and the Ognis
  Commander deck.
- `maplemagic/`: the game as a list of choices per decision, the searches,
  the network's input encoding, batched self-play, and the board as text.
- `maplemagic-py/`: Python bindings, installed as `maplemagic._core`.
- `python/maplemagic/`: the network, the training loop, the evaluations and
  the CLI.
- `configs/`: training and benchmark settings, and four Commander decklists
  (only Ognis has every card implemented).
- `scripts/`: `play.cmd` and `web.cmd` start a game against the best network.

## How the searches work

All three share one idea: since the opponent's hand and both libraries are
hidden, the search plays out guessed **worlds**, in which every card the
searching player can't see is dealt at random from the cards it could be.
Each player sees its own hand, the board, the graveyards, the opponent's
hand size and any card revealed to it. Both decklists are known, so a guess
at the hidden cards draws from exactly the cards that remain.

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
  2.4 evaluations. "-r" means the worlds are random. The paper's **full MAPLE**
  picks them with a learned network: each move it draws 50 candidate opponent
  hands and keeps the 5 a Siamese network rates closest to what the player
  has seen, and that network trains alongside the policy and value. Both are
  tested on Ognis and on Pauper below. Where worlds disagree on who acts or
  what is legal, MAPLE-r follows one random world. The **union** variant
  offers every choice legal in any world, as the paper does, and lets the
  majority's player act.
- **MAPLE-r seen hand** also shows the network each world's guessed opponent
  hand, as the paper's networks see the guessed board. Every world then looks
  different, so a simulation costs about 3.8 evaluations.
- **AlphaZe\*\*** deals 5 worlds once per move and grows an independent tree in
  each, sharing the simulations. The move comes from the average of the five
  trees' visit distributions.

In self-play the move is sampled in proportion to visits for the first 40
decisions of a game, and the visit distribution is the policy target.

Every search sends its leaves to the network in batches. A simulation that
reaches a position whose evaluation is still on its way would have to choose
without priors, and would walk the choice list from the top, usually Pass.
With **collisions** on, it stops and waits for the next batch instead. That
matters for small searches: at 10 simulations in batches of 8, seven of ten
choose blind. It is off by default and on where a model below says so.

## Ognis

Every game is the mirror match, Ognis against Ognis: two players at 40 life,
with the London mulligan. The decklist is `configs/commander/ognis.txt`.

| Deck | Colors | Plan | Key cards |
|---|---|---|---|
| Ognis | Red | Hasty attackers that make Treasure, big dragons, and stealing the opponent's creatures | Ognis, the Dragon's Lash, Goldspan Dragon, Hellkite Charger, Insurrection, Mob Rule |

### The best network

- A one-layer transformer, 192 wide (1.16M parameters), searching with
  IS-MCTS.
- 20 generations of 690 self-play games, with a 200,000-position replay
  buffer.
- **Playout-cap randomization:** a quarter of the moves get a full
  64-simulation search and train the policy; the rest get 10 simulations.
- **The value** trains on 16 positions a game, toward TD(λ = 0.95) targets.
- **The stable recipe:** embeddings start at std 0.02, and the learning rate
  warms up over 500 steps, then decays on a cosine.
- 46.7M self-play network evaluations: about 32 minutes of self-play and
  training on an RX 7900 XTX with a 32-thread CPU.

### Benchmark

**Question:** which changes to training make a stronger Ognis bot?

**How the comparison is kept fair:**
- **Similar training compute.** Each change trains from scratch, at about the
  self-play budget of the network it plays, counted in network evaluations.
  Playout-cap randomization makes most moves cheaper, so its runs play 690
  games a generation instead of 256 to spend the same.
- **Equal search at play time.** Every network plays IS-MCTS at 160
  simulations a move. The heuristic agent, a fixed reference, runs IS-MCTS
  with a hand-written evaluation at 200.
- **Duplicate deals.** A match is 600 games: 300 deals, each played twice
  with the seats swapped.
- **A change must beat the best network so far.** Each change plays the
  nearest kept model above it in the table, or the baseline. It is kept if
  its head-to-head interval lies above 0.5; where two settings tie, the
  cheaper one stays.

A score is wins plus half of draws, divided by games. ± is a 95% interval,
$1.96\sqrt{s(1-s)/n}$.

### Results

Every change tried, in order:

| Model | Config | Self-play network evaluations | Score against the heuristic agent | Head to head | Kept |
|---|---|---|---|---|---|
| Baseline: 3 layers, 160 wide, every position trains the value toward the result | `configs/ognis.yaml` | 58.8M | 0.372 ± 0.039 | – | – |
| Value on 16 positions a game | `configs/ognis_s1a.yaml` | 70.0M | 0.462 ± 0.040 | 0.622 ± 0.039 | yes |
| TD(λ) value targets | `configs/ognis_s1b.yaml` | 52.7M | 0.547 ± 0.040 | 0.560 ± 0.040 | yes |
| Playout-cap randomization | `configs/ognis_s1c.yaml` | 49.1M | 0.748 ± 0.035 | 0.693 ± 0.037 | yes |
| Heuristic values mixed into self-play's leaves | `configs/ognis_s1d.yaml` | 62.9M | 0.747 ± 0.035 | 0.517 ± 0.040 | no |
| Completed-Q policy targets | `configs/ognis_s2.yaml` | 49.6M | 0.738 ± 0.035 | 0.483 ± 0.040 | no |
| Generation 0 trained on heuristic self-play | `configs/ognis_s3.yaml` | 47.6M | 0.807 ± 0.032 | 0.527 ± 0.040 | no |
| Card attributes in the input | `configs/ognis_s5.yaml` | 51.8M | 0.752 ± 0.035 | 0.507 ± 0.040 | no |
| One layer, 192 wide | `configs/ognis_s6a.yaml` | 50.9M | 0.742 ± 0.035 | 0.497 ± 0.040 | no |
| Two layers, 192 wide | `configs/ognis_s6b.yaml` | 53.1M | 0.778 ± 0.033 | 0.507 ± 0.040 | no |
| Four layers, 192 wide | `configs/ognis_s6c.yaml` | 53.9M | 0.692 ± 0.037 | 0.465 ± 0.040 | no |
| The stable recipe | `configs/ognis_s6d.yaml` | 48.4M | 0.775 ± 0.033 | 0.540 ± 0.040 | yes |
| **One layer, 192 wide, with the recipe: the best network** | `configs/ognis_s6e.yaml` | 46.7M | **0.828 ± 0.030** | 0.530 ± 0.040 | yes, a tie at less cost |
| Generation 0 from heuristic self-play, with the best settings | `configs/ognis_s7.yaml` | 49.1M | 0.768 ± 0.034 | 0.445 ± 0.028 | no |
| 40 generations instead of 20 | `configs/ognis_s8.yaml` | 94.9M | 0.820 ± 0.031 | 0.464 ± 0.028 | no |
| MAPLE-r searching in training, the network seeing each world's guessed hand | `configs/ognis_m1r.yaml` | 50.1M | 0.708 ± 0.036 | 0.400 ± 0.039 | no |
| Full MAPLE: the same with worlds from a learned sampler | `configs/ognis_m1.yaml` | 65.8M | 0.842 ± 0.029 | 0.508 ± 0.040 | no |
| The best network's settings with collisions | `configs/ognis_s9.yaml` | 41.1M | 0.812 ± 0.031 | 0.508 ± 0.040 | no |

Two head-to-heads are 1,200 games: generation 0 from heuristic self-play
with the best settings, and 40 generations. Generation 0 from
heuristic self-play also takes about 15 minutes on the CPU: 2,000 heuristic
games at 800 simulations.

The two MAPLE models play MAPLE at 152–155 network evaluations a move (42
and 45 simulations; the best network uses 156 at 160), and their head-to-head
is against the best network. Full
MAPLE's games ran longer, so it trained on more evaluations; see
"Full MAPLE on Ognis" below.

**What the results say:**
- **The value head was the bottleneck.** The baseline memorized its games:
  training value loss fell to 0.08 while held-out loss stayed near 1.0, the
  score of predicting a draw everywhere. With the hand-written evaluation at
  its leaves instead of its own values, the same network beat the heuristic
  agent 0.605 ± 0.039.
- **Three changes fixed it:** value training on 16 positions a game, TD(λ)
  targets and playout-cap randomization. On fresh games, held-out value AUC
  rose from 0.76 to 0.89, above the hand-written evaluation's 0.85 on the
  same positions, and the score against the heuristic agent from 0.372 to
  0.748.
- **Playout-cap randomization was the biggest single gain**, 0.693 ± 0.037
  head to head, on slightly less self-play.
- **Size barely matters.** One to four layers all tie with three. One layer
  with the stable recipe is the cheapest and scores best against the
  heuristic agent.
- **Nothing else helped.** Mixed leaf values, completed-Q targets, card
  attributes and a pretrained generation 0 all tie. On the best settings, a
  pretrained generation 0 loses (0.445 ± 0.028), and so does training twice
  as long (0.464 ± 0.028): more of the same self-play doesn't help yet.

### Searches on Ognis

The playout-cap network on both sides, at about 160 network evaluations a
move each: 160 simulations for IS-MCTS and AlphaZe\*\* over one world, 59 for
MAPLE over five. 600 games each.

| Search, against IS-MCTS | Score |
|---|---|
| AlphaZe\*\* over one world | 0.505 ± 0.040 |
| MAPLE-r | 0.478 ± 0.040 |
| MAPLE with worlds from an offline sampler | 0.420 ± 0.039 |

- As on Pauper, nothing beats IS-MCTS. At 160 simulations the leaves are only
  3 to 4 decisions past the root, under a quarter of a turn, so every search
  sees nearly the same few plies, and how the worlds are dealt matters
  little.
- The offline sampler comes from `sampler-pilot`, trained on a frozen
  network's games. It ranks the true hand no better than a table of how often
  each unseen card is in the hand, by turn. Scoring took part of each move,
  so this MAPLE got 13% fewer evaluations.

### Full MAPLE on Ognis

**Question:** does the paper's full method, worlds picked by a sampler that
trains alongside the network, make a stronger bot?

Two models train with MAPLE over 5 worlds, the network seeing each world's
guessed opponent hand as the paper's networks do. They differ only in where
the worlds come from: random (MAPLE-r) or the learned sampler (full MAPLE).
Each move, the sampler draws 50 candidate hands and keeps the 5 it rates
closest to the player's observation and the opponent's past moves. It trains
on the true hands from self-play, and generation 0 plays random worlds.
Matches are 600 games at equal network evaluations a move per side, within 7%.

| Match | Score |
|---|---|
| Full MAPLE against MAPLE-r | 0.578 ± 0.040 |
| The same, full MAPLE at MAPLE-r's training budget (49.2M evaluations) | 0.555 ± 0.040 |
| Full MAPLE against the best network (IS-MCTS) | 0.508 ± 0.040 |
| MAPLE-r against the best network | 0.400 ± 0.039 |
| Full MAPLE's network: learned worlds against random worlds | 0.493 ± 0.040 |
| Full MAPLE's network: the true hand in every world against random worlds | 0.518 ± 0.040 |
| The best network: the true hand in every world against random worlds | 0.508 ± 0.040 |

- **The learned sampler works,** as in the paper: full MAPLE beats MAPLE-r,
  also at equal training compute. On fresh positions it ranks the true hand at
  0.159, against 0.201 for the per-turn table and 0.239 for the offline sampler.
- **The gain comes from training, not from play.** At play time, full MAPLE's
  network plays as well with random worlds, and even dealing it the opponent's
  true hand doesn't help. Training on searches over likelier worlds made the
  better network.
- **Full MAPLE only ties IS-MCTS.** Knowing the hand doesn't help IS-MCTS at
  160 simulations either: the search sees about 4 decisions ahead, and the
  opponent rarely gets to use its hand in that span. Ognis, a red deck with few
  instants, is a hard case for hidden information.
- **MAPLE's small searches need batches of one leaf, or collisions.** With
  larger batches, the first simulations of a search choose before the
  network's priors arrive. At 2 to 3 simulations a move, as in MAPLE's cheap
  training searches, the agent then passed almost every move.

### Collisions on Ognis

**Question:** the best network's cheap training searches chose blind on 7 of
10 simulations. Does fixing that make a better network?

| Match, 600 games each | Score |
|---|---|
| The best network with collisions against without, 10 simulations a move | 0.622 ± 0.039 |
| The same at 160 simulations | 0.513 ± 0.040 |
| Retrained with collisions against the best network | 0.508 ± 0.040 |
| The same at equal training compute (the best network's generation 17) | 0.488 ± 0.040 |

- **Small searches play better with collisions,** but the retrained network
  ties. Under the playout cap only the 64-simulation searches, where 7 of 64
  choices were blind, train the policy.
- Self-play takes the same time a generation: collisions mean more, smaller
  batches, but the retrained network's games ran 20% shorter.
- Collisions stay off by default, and `play` and `web` search with the
  network's own setting.

### More search at play time

The best network, 600 games each:

| Match | Score |
|---|---|
| 400 simulations against 160 | 0.542 ± 0.040 |
| 800 simulations against 160 | 0.565 ± 0.040 |
| 400 simulations against the heuristic agent | 0.870 ± 0.027 |

`play` and `web` therefore default to 400, about 0.2 s a move.

### Caveats

- **One training run per setting**, and runs aren't bitwise reproducible:
  forward passes are sized from free memory, which changes the numerics.
- **Narrow margins.** The stable recipe won by the least that counts (its
  interval starts at 0.5001), and the best network was kept on a tie.
- **Unequal budgets.** Games vary in length, so 20 generations cost
  47–70M evaluations. The value-on-16-positions model used the most; its
  checkpoint at the baseline's budget also beat the baseline, 0.692 ± 0.037.
- **One matchup.** Every game is Ognis against Ognis.

## Pauper: comparing the searches

Two benchmarks on the same nine decks. The second trains every model with the
Ognis recipe and tests the paper's full method, MAPLE with a learned world
sampler. The first came earlier, at commit `8488808`, with smaller budgets and
an older recipe.

### The decks

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

Decklists are in `data/runtime_decks_v1.json`.

### With the Ognis recipe

**Questions:** does the batch fix (collisions) help here, and does full MAPLE
beat IS-MCTS on decks whose hidden hands matter: counterspells, instant-speed
burn, 60-card decks with four copies of a card?

Every model is the best Ognis network's architecture and recipe (one layer,
192 wide; playout cap; 16 value positions a game; TD(λ); the stable recipe),
trained for 20 generations on all 81 ordered deck pairings.

| Model | Config | Training search | Games a generation | Self-play network evaluations | Training time |
|---|---|---|---|---|---|
| IS-MCTS | `configs/pauper_p0.yaml` | 64 / 10 simulations, batches of 8 | 690 | 53.6M | 0.9 h |
| **IS-MCTS with collisions: the baseline** | `configs/pauper_p0f.yaml` | the same, with collisions | 690 | 44.1M | 0.9 h |
| MAPLE-r, seen hand | `configs/pauper_m1r.yaml` | MAPLE over 5 random worlds, 16 / 3 simulations, collisions | 480 | 37.7M | 0.9 h |
| Full MAPLE | `configs/pauper_m1.yaml` | the same over 5 worlds from the learned sampler | 480 | 36.6M | 1.0 h |
| Full MAPLE, hidden hand | `configs/pauper_m2.yaml` | full MAPLE, the network not shown the guessed hand; 31 / 6 simulations | 630 | 45.8M | 1.4 h |
| IS-MCTS with learned worlds | `configs/pauper_m3.yaml` | the baseline, each simulation's world drawn from the sampler's 16 best of 50 hands | 690 | 54.4M | 1.6 h |

Each training search costs about 23–25 network evaluations a move. The game
counts aim at the baseline's self-play budget; games grow or shrink as a model
learns, so where totals ended more than 10% apart, the larger run's checkpoint
at the smaller one's budget plays too.

**How the comparison is kept fair:**
- **Equal search at play time:** about 155 network evaluations a move per
  side. That is 160 simulations for IS-MCTS, 49 for full MAPLE, 52 for MAPLE-r
  and 80 for full MAPLE with a hidden hand, whose worlds merge more. The
  heuristic agent runs IS-MCTS with its hand-written evaluation at 200.
- **Duplicate deals:** a match is 540 games, the 45 deck pairings each dealt
  six times and played from both seats.

| Match | Score |
|---|---|
| IS-MCTS with collisions against IS-MCTS (1,080 games) | 0.529 ± 0.030 |
| The same, IS-MCTS at the same training budget | **0.556 ± 0.042** |
| IS-MCTS with collisions: the opponent's true hand in every world against random worlds | **0.546 ± 0.042** |
| Full MAPLE against MAPLE-r | 0.509 ± 0.042 |
| Full MAPLE against IS-MCTS with collisions | 0.407 ± 0.041 |
| The same at the same training budget | 0.417 ± 0.042 |
| MAPLE-r against IS-MCTS with collisions | 0.376 ± 0.041 |
| The same at the same training budget | 0.420 ± 0.042 |
| Full MAPLE, hidden hand, against IS-MCTS with collisions | 0.465 ± 0.042 |
| IS-MCTS with learned worlds against IS-MCTS with collisions | 0.487 ± 0.042 |
| The same at the same training budget | 0.494 ± 0.042 |
| Full MAPLE's network: learned against random worlds | 0.493 ± 0.042 |
| Full MAPLE, hidden hand: learned against random worlds | 0.481 ± 0.042 |
| IS-MCTS with learned worlds: learned against random worlds | 0.498 ± 0.042 |
| Full MAPLE, hidden hand: the true hand against random worlds | 0.506 ± 0.042 |
| IS-MCTS with learned worlds: the true hand against random worlds | 0.530 ± 0.042 |

Against the heuristic agent: IS-MCTS 0.704, IS-MCTS with collisions 0.681,
full MAPLE 0.628, MAPLE-r 0.602, full MAPLE with a hidden hand 0.681, and
IS-MCTS with learned worlds 0.700 (each ± 0.04).

**What the results say:**
- **Every model now beats the heuristic agent,** which beat every model of the
  first benchmark below.
- **Collisions help on Pauper.** At the same training budget, the baseline
  with collisions beats IS-MCTS without (0.556), and it edges the IS-MCTS run
  that trained on 22% more evaluations. Its network plays the same with or
  without collisions, so the gain comes from training. On Ognis the same
  change tied (see "Collisions on Ognis").
- **Here knowing the hand helps IS-MCTS.** Dealing the opponent's true hand
  into every world beats random worlds (0.546, and 0.530 with the
  learned-worlds network), unlike on Ognis (0.508). Hidden hands matter more
  with counterspells and instants. It doesn't help MAPLE (0.506).
- **But no learned sampler turns that into wins.** The sampler learns: it ranks
  the true hand at about 0.17 among 50 candidates, against 0.22 for a per-turn
  table of how often each card is still in hand, and it passes that table
  after one generation. Yet learned worlds tie random worlds in every pairing,
  in training (full MAPLE against MAPLE-r) and at play time. The sampler ranks
  whole hands well but recalls instants worse than the table (0.38 against
  0.43), and instants are what decide counterspell and burn turns.
- **Full MAPLE loses to IS-MCTS here** (0.41), where it tied on Ognis. With
  the network shown each world's guessed hand it trails with every deck.
  Hiding the hand, so that worlds merge and the search goes deeper, brings it
  to a tie (0.465), as does IS-MCTS with learned worlds (0.487).
- **IS-MCTS with collisions stays the baseline:** nothing beats it.

By deck, each model's score with each deck against IS-MCTS with collisions,
and each deck's win rate in those matches' non-mirror games:

| Deck | Non-mirror win rate | IS-MCTS | MAPLE-r, seen hand | Full MAPLE | Full MAPLE, hidden hand | IS-MCTS with learned worlds | IS-MCTS with collisions |
|---|---|---|---|---|---|---|---|
| Rally | 0.764 | 0.72 | 0.57 | 0.67 | 0.82 | 0.70 | 0.72 |
| Terror | 0.738 | 0.63 | 0.67 | 0.68 | 0.72 | 0.73 | 0.70 |
| Burn | 0.571 | 0.62 | 0.37 | 0.50 | 0.55 | 0.58 | 0.57 |
| Elves | 0.538 | 0.47 | 0.33 | 0.42 | 0.40 | 0.55 | 0.62 |
| Affinity | 0.467 | 0.49 | 0.38 | 0.35 | 0.45 | 0.42 | 0.52 |
| Faeries | 0.410 | 0.38 | 0.27 | 0.30 | 0.43 | 0.35 | 0.51 |
| CawGates | 0.396 | 0.42 | 0.37 | 0.30 | 0.38 | 0.37 | 0.46 |
| Wildfire | 0.380 | 0.38 | 0.28 | 0.30 | 0.32 | 0.38 | 0.47 |
| Spy | 0.236 | 0.12 | 0.15 | 0.15 | 0.12 | 0.30 | 0.42 |

- A cell is 60 games (±0.12–0.13), 120 for IS-MCTS, and 360 for IS-MCTS with
  collisions over all five matches, so single cells are informative only.
- IS-MCTS with collisions leads with most decks. Its clearest gain over
  IS-MCTS is with Spy, the combo deck that must chain many actions in one
  turn: in their 1,080 games each scored 0.50 and 0.12 with it, 120 games
  apiece.

### The first benchmark

At commit `8488808`, before the Ognis recipe.

#### The models

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
| AlphaZe\*\* | `configs/mid_d.yaml` | 160 (5 × 32) | 158 | 5.1 h |

#### Benchmark

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

± is a 95% interval as above, which is conservative here: duplicate deals
cancel some of the luck it allows for.

#### Results

| Model | Network evaluations per move | Score against the heuristic agent | Score against the other four models | Rating |
|---|---|---|---|---|
| IS-MCTS | 158 | 0.322 ± 0.056 | **0.537 ± 0.030** | −115 [−140, −93] |
| MAPLE-r | 162 | 0.322 ± 0.056 | 0.489 ± 0.030 | −138 [−167, −116] |
| MAPLE-r union | 153 | 0.300 ± 0.055 | 0.501 ± 0.030 | −135 [−163, −111] |
| MAPLE-r seen hand | 162 | 0.296 ± 0.054 | 0.508 ± 0.030 | −132 [−158, −110] |
| AlphaZe\*\* | 158 | 0.344 ± 0.057 | 0.465 ± 0.030 | −147 [−174, −122] |

The rating is a Bradley–Terry fit over all 4,050 games on the Elo scale,
with the heuristic agent at 0 and a 95% bootstrap interval.

Head to head, score of the row's model against the column's:

| | IS-MCTS | MAPLE-r | MAPLE-r union | MAPLE-r seen hand | AlphaZe\*\* | Heuristic |
|---|---|---|---|---|---|---|
| IS-MCTS | — | 0.526 | 0.507 | 0.533 | 0.581 | 0.322 |
| MAPLE-r | 0.474 | — | 0.496 | 0.496 | 0.489 | 0.322 |
| MAPLE-r union | 0.493 | 0.504 | — | 0.489 | 0.519 | 0.300 |
| MAPLE-r seen hand | 0.467 | 0.504 | 0.511 | — | 0.552 | 0.296 |
| AlphaZe\*\* | 0.419 | 0.511 | 0.481 | 0.448 | — | 0.344 |
| Heuristic | 0.678 | 0.678 | 0.700 | 0.704 | 0.656 | — |

Each cell is 270 games, ±0.060 at most.

**What the results say:**
- **IS-MCTS is the strongest model, by a small margin.** It is the only one
  that scores above even against the others (0.537 ± 0.030). Its biggest
  margin is over AlphaZe\*\*, 0.581 ± 0.059, the one significant head-to-head
  result. No other match separates any two models.
- **MAPLE-r is no better than AlphaZe\*\*.** They tie: MAPLE-r scores
  0.489 ± 0.060. In the paper, MAPLE-r was ahead of AlphaZe\*\* on Phantom Go
  and behind on Dark Hex; only the learned world sampler won on both. The
  benchmark above tests it on these decks.
- **The MAPLE-r variants make no difference.** The union rule ties the default
  (0.496), and showing the network the guessed hand doesn't help either
  (0.496 against MAPLE-r).
- **AlphaZe\*\* does best against the heuristic agent and worst against the
  networks:** 0.344 against the heuristic agent, but 0.465 ± 0.030 against
  the other models. Its heuristic score is within noise of the others'.
- **The heuristic agent beats every model:** they score 0.30–0.34 against
  it. Twenty generations of 256 self-play games weren't enough for any
  network to beat a hand-written evaluation searching 200 simulations.
- A likely reason IS-MCTS leads: it deals a fresh guess at the hidden cards
  every simulation, 160 per move, while MAPLE-r and AlphaZe\*\* commit to five
  guesses for the whole move.

#### Without a network

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

#### By deck

Each model's score with each deck, over all its matches, and each deck's win
rate in all non-mirror games:

| Deck | All games | IS-MCTS | MAPLE-r | MAPLE-r union | MAPLE-r seen hand | AlphaZe\*\* | Heuristic |
|---|---|---|---|---|---|---|---|
| Rally | 0.725 | 0.67 | 0.63 | 0.74 | 0.50 | 0.59 | 0.95 |
| Terror | 0.724 | 0.61 | 0.70 | 0.55 | 0.63 | 0.69 | 0.88 |
| Faeries | 0.606 | 0.60 | 0.51 | 0.52 | 0.53 | 0.47 | 0.88 |
| Elves | 0.562 | 0.56 | 0.52 | 0.50 | 0.52 | 0.51 | 0.69 |
| Burn | 0.500 | 0.49 | 0.49 | 0.58 | 0.46 | 0.41 | 0.57 |
| Affinity | 0.469 | 0.49 | 0.41 | 0.44 | 0.49 | 0.48 | 0.54 |
| Wildfire | 0.411 | 0.39 | 0.43 | 0.25 | 0.43 | 0.45 | 0.63 |
| CawGates | 0.294 | 0.35 | 0.19 | 0.39 | 0.28 | 0.25 | 0.56 |
| Spy | 0.208 | 0.28 | 0.21 | 0.17 | 0.35 | 0.13 | 0.45 |

- The deck matters more than the model. Rally and Terror win about 70% of
  their games, Spy and CawGates under a third.
- Each cell for a model is 150 games, ±0.08, so differences between models on
  one deck are mostly noise. The heuristic agent is ahead with every deck.

#### Caveats

- **Scale.** One training run per model, 20 generations, with a small
  network.
- **Training compute wasn't equal.** IS-MCTS trained with 64 network
  evaluations per move, the others with 139–233. The benchmark equalizes
  compute only at play time.
- **Bugs fixed after training.** IS-MCTS's training games used a search that
  left some choices with too high a prior. The four models other than
  AlphaZe\*\* also trained while two copies of a card that differed in damage
  or combat counted as one choice. AlphaZe\*\* was trained after both fixes,
  and the benchmark uses the fixed code.
- **The value heads memorized their games.** Training value loss ended at
  0.05–0.08 on win/loss targets, too low for honest prediction from early in
  a game. AlphaZe\*\*, the only run that logged held-out losses, ended at a
  held-out value loss of 0.85 against a training loss of 0.05; predicting a
  draw everywhere scores 1.0. The Ognis runs fixed this.

## Reproduce

Ognis, one job at a time:

```sh
for c in ognis ognis_s1a ognis_s1b ognis_s1c ognis_s6d ognis_s6e; do $PY -m maplemagic.cli train configs/$c.yaml --fresh; done
# A change against the model before it, and against the heuristic agent:
$PY -m maplemagic.cli eval runs/ognis_s6e/latest.pt --vs runs/ognis_s6d/latest.pt --games 600 --sims 160 \
    --config configs/ognis.yaml --out runs/evals/s6e_vs_best.json
$PY -m maplemagic.cli eval runs/ognis_s6e/latest.pt --vs heuristic --games 600 --sims 160 \
    --config configs/ognis.yaml --out runs/evals/s6e_vs_heur.json
# Held-out value accuracy of two runs on the same fresh games:
$PY -m maplemagic.cli yardstick runs/ognis_s6d/latest.pt runs/ognis_s6e/latest.pt --games 100 --seed 777777
# Full MAPLE and MAPLE-r, then full MAPLE against the best network at equal evaluations a move:
for c in ognis_m1r ognis_m1; do $PY -m maplemagic.cli train configs/$c.yaml --fresh; done
$PY -m maplemagic.cli eval runs/ognis_m1/latest.pt --vs runs/ognis_s6e/latest.pt --games 600 --sims 42 --opp-sims 160 \
    --leaves-per-step 2 --opp-leaves-per-step 8 --config configs/ognis.yaml --out runs/evals/m1_vs_s6e.json
```

Each config pins the settings it trained with. A 20-generation run takes
30–65 minutes of self-play and training on an RX 7900 XTX with a 32-thread
CPU; the MAPLE runs take 60–110 minutes.

Pauper, with the Ognis recipe:

```sh
for c in pauper_p0 pauper_p0f pauper_m1r pauper_m1 pauper_m2 pauper_m3; do $PY -m maplemagic.cli train configs/$c.yaml --fresh; done
# Full MAPLE against IS-MCTS with collisions at about 155 network evaluations a move each:
$PY -m maplemagic.cli eval runs/pauper_m1/latest.pt --vs runs/pauper_p0f/latest.pt --sims 49 --opp-sims 160     --leaves-per-step 2 --opp-leaves-per-step 8 --games 540 --config configs/bench.yaml --out runs/evals/p1_vs_p0f.json
# The true hand in every world against random worlds, one network on both sides:
$PY -m maplemagic.cli eval runs/pauper_p0f/latest.pt --vs self --world-sampler oracle --sims 160     --games 540 --config configs/bench.yaml --out runs/evals/p0f_oracle_vs_uniform.json
```

A run takes 0.9–1.6 hours. Training compute drifts with game length, so
compare totals in each run's `log.jsonl` and use a `genNNNN.pt` checkpoint at
equal compute where they differ by more than 10%.

The first Pauper benchmark:

```sh
for c in mid mid_b mid_bu mid_c mid_d; do $PY -m maplemagic.cli train configs/$c.yaml --fresh; done
# One match: each side's simulations per move, as in the table above.
$PY -m maplemagic.cli eval runs/mid_a/latest.pt --vs runs/mid_b/latest.pt --sims 160 --opp-sims 68 \
    --games 270 --config configs/bench.yaml --out runs/bench/ismcts-maple.json
$PY -m maplemagic.cli eval runs/mid_b/latest.pt --vs heuristic --sims 68 \
    --games 270 --config configs/bench.yaml --out runs/bench/maple-heuristic.json
# Without a network, at about 200 evaluations per move:
cargo run --release -p maplemagic --example arena -- h75m5 h200 Burn Rally 100
```

The Pauper benchmark was run at commit `8488808`; AlphaZe\*\*'s training and
its five matches ran later, on code with the same search. A MAPLE-r or
AlphaZe\*\* training run takes 4–6 hours, and IS-MCTS about 2.

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
