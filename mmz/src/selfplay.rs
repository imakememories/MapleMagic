//! Many games at once, each with its own IS-MCTS tree, evaluated in shared
//! batches by an outside evaluator (the network, on the GPU).
//!
//! Each call to [`SelfPlay::gather`] advances every game until its search
//! needs leaf evaluations and returns all of them as one batch;
//! [`SelfPlay::feed`] hands the results back. Moves are made once a search
//! reaches its simulation budget. Finished games yield training samples
//! (observation, choices, visit distribution, final result).
//!
//! Every seat has a model id: evaluations are tagged with it so the caller
//! can run different networks per seat (for evaluation matches).
//! [`HEURISTIC`] seats are searched in Rust with the board heuristic and
//! never reach the caller.

use crate::features::{choice_fields, observe, ACTION_FIELDS, TOKEN_FIELDS};
use crate::game::{Game, Outcome};
use crate::ismcts::{Config, Eval, HeuristicEval, Search};
use mtg_kernel::ids::PlayerId;
use mtg_kernel::state::SplitMix64;
use rayon::prelude::*;

pub const HEURISTIC: u8 = 255;
pub const RANDOM: u8 = 254;

#[derive(Debug, Clone)]
pub struct SelfPlayConfig {
    pub sims: u32,
    /// Leaves each search contributes per batch (virtual loss spreads them).
    pub leaves_per_step: usize,
    /// Decisions (per game) that sample moves in proportion to visits
    /// instead of taking the most visited one.
    pub temp_decisions: u32,
    pub search: Config,
    /// Deck pairings (seat 0 deck, seat 1 deck) to draw games from.
    pub pairings: Vec<(String, String)>,
    /// Model id per seat. Games alternate which physical seat each model
    /// takes, so both play both decks of a pairing equally.
    pub seat_models: [u8; 2],
    /// Record training samples for seats with model 0.
    pub record: bool,
    /// Stop starting games after this many (None = forever).
    pub max_games: Option<u64>,
    /// Sims for heuristic-model seats.
    pub heuristic_sims: u32,
}

#[derive(Debug, Clone)]
pub struct Sample {
    pub tokens: Vec<[u16; TOKEN_FIELDS]>,
    pub actions: Vec<[u16; ACTION_FIELDS]>,
    pub policy: Vec<f32>,
    /// Final result from the sampled player's point of view.
    pub z: f32,
    /// Search root value, for diagnostics / value targets mixing.
    pub root_value: f32,
}

#[derive(Debug, Clone)]
pub struct GameResult {
    pub decks: [String; 2],
    pub models: [u8; 2],
    pub outcome: Outcome,
    pub decisions: u32,
}

/// One leaf waiting for evaluation.
pub struct LeafRequest {
    pub tokens: Vec<[u16; TOKEN_FIELDS]>,
    pub actions: Vec<[u16; ACTION_FIELDS]>,
    pub model: u8,
}

struct Pending {
    tokens: Vec<[u16; TOKEN_FIELDS]>,
    actions: Vec<[u16; ACTION_FIELDS]>,
    policy: Vec<f32>,
    root_value: f32,
    player: PlayerId,
}

struct Slot {
    game: Option<Game>,
    decks: [String; 2],
    models: [u8; 2],
    search: Option<Search>,
    leaves: Vec<Game>,
    pending: Vec<Pending>,
    rng: SplitMix64,
}

pub struct SelfPlay {
    cfg: SelfPlayConfig,
    slots: Vec<Slot>,
    started: u64,
    seed: u64,
    samples: Vec<Sample>,
    results: Vec<GameResult>,
    /// Slot index and leaf count for each slot contributing to the last batch.
    batch_plan: Vec<(usize, usize)>,
}

impl SelfPlay {
    pub fn new(parallel: usize, cfg: SelfPlayConfig, seed: u64) -> SelfPlay {
        let slots = (0..parallel)
            .map(|i| Slot {
                game: None,
                decks: [String::new(), String::new()],
                models: [0, 0],
                search: None,
                leaves: Vec::new(),
                pending: Vec::new(),
                rng: SplitMix64::seed(seed ^ (i as u64).wrapping_mul(0x9E3779B97F4A7C15)),
            })
            .collect();
        SelfPlay { cfg, slots, started: 0, seed, samples: Vec::new(), results: Vec::new(), batch_plan: Vec::new() }
    }

    pub fn games_started(&self) -> u64 {
        self.started
    }

    /// Advance all games; return every leaf needing evaluation. Empty once
    /// `max_games` games have all finished.
    pub fn gather(&mut self) -> Vec<LeafRequest> {
        // Start new games single-threaded (game ids must be deterministic).
        for slot in self.slots.iter_mut() {
            if slot.game.is_none() && self.cfg.max_games.is_none_or(|m| self.started < m) {
                let id = self.started;
                self.started += 1;
                let pairing = &self.cfg.pairings[(id / 2) as usize % self.cfg.pairings.len()];
                // Alternate seats: in odd games the models swap seats.
                let (m0, m1) = if id % 2 == 0 {
                    (self.cfg.seat_models[0], self.cfg.seat_models[1])
                } else {
                    (self.cfg.seat_models[1], self.cfg.seat_models[0])
                };
                let (d0, d1) = if id % 2 == 0 { (&pairing.0, &pairing.1) } else { (&pairing.1, &pairing.0) };
                let game_seed = self.seed.wrapping_mul(1_000_003).wrapping_add(id / 2);
                slot.game = Some(Game::new(d0, d1, game_seed).expect("valid deck names"));
                slot.decks = [d0.clone(), d1.clone()];
                slot.models = [m0, m1];
                slot.search = None;
                slot.pending.clear();
            }
        }
        let cfg = &self.cfg;
        let finished: Vec<(Vec<Sample>, Option<GameResult>)> =
            self.slots.par_iter_mut().map(|slot| advance(slot, cfg)).collect();
        for (samples, result) in finished {
            self.samples.extend(samples);
            self.results.extend(result);
        }
        self.batch_plan.clear();
        let mut out = Vec::new();
        for (i, slot) in self.slots.iter().enumerate() {
            if slot.leaves.is_empty() {
                continue;
            }
            self.batch_plan.push((i, slot.leaves.len()));
            let game = slot.game.as_ref().unwrap();
            let model = slot.models[game.to_act().index()];
            for leaf in &slot.leaves {
                out.push(LeafRequest { tokens: observe(leaf, leaf.to_act()), actions: choice_fields(leaf), model });
            }
        }
        out
    }

    /// Results for the batch returned by the last [`Self::gather`], in order.
    pub fn feed(&mut self, evals: Vec<Eval>) {
        let mut it = evals.into_iter();
        let mut per_slot: Vec<(usize, Vec<Eval>)> = Vec::new();
        for &(i, n) in &self.batch_plan {
            per_slot.push((i, it.by_ref().take(n).collect()));
        }
        assert!(it.next().is_none(), "more evaluations than leaves");
        let mut by_index: Vec<Option<Vec<Eval>>> = (0..self.slots.len()).map(|_| None).collect();
        for (i, e) in per_slot {
            by_index[i] = Some(e);
        }
        self.slots.par_iter_mut().zip(by_index).for_each(|(slot, evals)| {
            if let Some(evals) = evals {
                let leaves = std::mem::take(&mut slot.leaves);
                assert_eq!(leaves.len(), evals.len());
                slot.search.as_mut().unwrap().feed(&leaves, evals);
            }
        });
        self.batch_plan.clear();
    }

    pub fn take_samples(&mut self) -> Vec<Sample> {
        std::mem::take(&mut self.samples)
    }

    pub fn take_results(&mut self) -> Vec<GameResult> {
        std::mem::take(&mut self.results)
    }
}

/// Drive one slot until its search needs evaluations (or it has no game).
fn advance(slot: &mut Slot, cfg: &SelfPlayConfig) -> (Vec<Sample>, Option<GameResult>) {
    let mut samples = Vec::new();
    let mut result = None;
    loop {
        let Some(game) = slot.game.as_mut() else { break };
        if let Some(outcome) = game.outcome() {
            for p in slot.pending.drain(..) {
                let z = match outcome {
                    Outcome::Win(w) if w == p.player => 1.0,
                    Outcome::Win(_) => -1.0,
                    _ => 0.0,
                };
                samples.push(Sample { tokens: p.tokens, actions: p.actions, policy: p.policy, z, root_value: p.root_value });
            }
            result = Some(GameResult { decks: slot.decks.clone(), models: slot.models, outcome, decisions: game.decisions });
            slot.game = None;
            slot.search = None;
            break;
        }
        let me = game.to_act();
        let model = slot.models[me.index()];
        if model == RANDOM {
            let n = game.choices().len();
            game.apply((slot.rng.next_u64() % n as u64) as usize);
            continue;
        }
        if model == HEURISTIC {
            let mut s = Search::new(game, cfg.search.clone(), slot.rng.next_u64());
            s.run(cfg.heuristic_sims, 8, &mut HeuristicEval);
            let c = s.best();
            game.apply(c);
            continue;
        }
        let search = slot.search.get_or_insert_with(|| {
            let mut c = cfg.search.clone();
            if !cfg.record {
                c.root_noise = 0.0;
            }
            Search::new(game, c, slot.rng.next_u64())
        });
        if search.simulations >= cfg.sims {
            let visits = search.root_visits();
            let total: f32 = visits.iter().sum::<f32>().max(1e-6);
            let pick = if game.decisions < cfg.temp_decisions {
                let mut r = (slot.rng.next_u64() >> 11) as f32 / (1u64 << 53) as f32 * total;
                let mut pick = visits.len() - 1;
                for (i, &v) in visits.iter().enumerate() {
                    if r < v {
                        pick = i;
                        break;
                    }
                    r -= v;
                }
                pick
            } else {
                search.best()
            };
            if cfg.record && model == 0 {
                slot.pending.push(Pending {
                    tokens: observe(game, me),
                    actions: choice_fields(game),
                    policy: visits.iter().map(|v| v / total).collect(),
                    root_value: search.root_value(),
                    player: me,
                });
            }
            slot.search = None;
            game.apply(pick);
            continue;
        }
        let want = cfg.leaves_per_step.min((cfg.sims - search.simulations) as usize).max(1);
        let leaves = search.gather(want);
        if leaves.is_empty() {
            continue; // all simulations hit terminal states; loop again
        }
        slot.leaves = leaves;
        break;
    }
    (samples, result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selfplay_runs_games_and_records_samples() {
        let cfg = SelfPlayConfig {
            sims: 16,
            leaves_per_step: 4,
            temp_decisions: 10,
            search: Config::default(),
            pairings: vec![("Burn".into(), "Rally".into())],
            seat_models: [0, 0],
            record: true,
            max_games: Some(4),
            heuristic_sims: 16,
        };
        let mut sp = SelfPlay::new(4, cfg, 1);
        let mut steps = 0;
        loop {
            let leaves = sp.gather();
            if leaves.is_empty() {
                break;
            }
            // Uniform priors, zero value: stands in for the network.
            let evals = leaves.iter().map(|l| Eval { priors: vec![1.0; l.actions.len()], value: 0.0 }).collect();
            sp.feed(evals);
            steps += 1;
            assert!(steps < 200_000);
        }
        let results = sp.take_results();
        assert_eq!(results.len(), 4);
        let samples = sp.take_samples();
        assert!(!samples.is_empty());
        for s in &samples {
            assert_eq!(s.actions.len(), s.policy.len());
            assert!((s.policy.iter().sum::<f32>() - 1.0).abs() < 1e-3);
        }
    }
}
