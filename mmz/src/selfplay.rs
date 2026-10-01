//! Many games at once, with leaf evaluations batched for the network.
//! [`HEURISTIC`] and [`RANDOM`] seats are played in Rust.

use crate::features::{choice_fields, observe_with, ACTION_FIELDS, TOKEN_FIELDS};
use crate::game::{Game, Outcome};
use crate::ismcts::{Config, Eval, HeuristicEval, MapleSelect, Search, SearchStats};
use mtg_kernel::ids::PlayerId;
use mtg_kernel::state::SplitMix64;
use rayon::prelude::*;

pub const HEURISTIC: u8 = 255;
pub const RANDOM: u8 = 254;

#[derive(Debug, Clone)]
pub struct SelfPlayConfig {
    /// Simulations per move for each network model id (0, 1).
    pub model_sims: [u32; 2],
    /// Leaves each search contributes per batch (virtual loss spreads them).
    pub leaves_per_step: usize,
    /// Decisions per game that sample moves in proportion to visits.
    pub temp_decisions: u32,
    pub search: Config,
    /// Deck pairings (seat 0 deck, seat 1 deck) to draw games from.
    pub pairings: Vec<(String, String)>,
    /// Model id per seat; the models swap seats every other game.
    pub seat_models: [u8; 2],
    /// Record training samples for seats with model 0.
    pub record: bool,
    /// Stop starting games after this many (None = forever).
    pub max_games: Option<u64>,
    /// Sims for heuristic-model seats.
    pub heuristic_sims: u32,
    /// The search settings below are per network model id.
    pub maple_worlds: [u32; 2],
    pub maple_select: [MapleSelect; 2],
    pub maple_resample: [bool; 2],
    pub perfect_obs: [bool; 2],
    pub pimc_worlds: [u32; 2],
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
    /// Search counters per network model id, over finished searches.
    stats: [SearchStats; 2],
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
        SelfPlay { cfg, slots, started: 0, seed, samples: Vec::new(), results: Vec::new(), batch_plan: Vec::new(), stats: Default::default() }
    }

    pub fn games_started(&self) -> u64 {
        self.started
    }

    /// Advance all games and return the leaves to evaluate (empty once all games are done).
    pub fn gather(&mut self) -> Vec<LeafRequest> {
        // Start new games single-threaded (game ids must be deterministic).
        for slot in self.slots.iter_mut() {
            if slot.game.is_none() && self.cfg.max_games.is_none_or(|m| self.started < m) {
                let id = self.started;
                self.started += 1;
                let (decks, models, game_seed) = game_setup(&self.cfg, self.seed, id);
                slot.game = Some(Game::new(&decks[0], &decks[1], game_seed).expect("valid deck names"));
                slot.decks = decks;
                slot.models = models;
                slot.search = None;
                slot.pending.clear();
            }
        }
        let cfg = &self.cfg;
        let finished: Vec<(Vec<Sample>, Option<GameResult>, Vec<(u8, SearchStats)>)> =
            self.slots.par_iter_mut().map(|slot| advance(slot, cfg)).collect();
        for (samples, result, stats) in finished {
            self.samples.extend(samples);
            self.results.extend(result);
            for (m, s) in stats {
                self.stats[m as usize] += s;
            }
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
            let perfect = self.cfg.perfect_obs[model as usize];
            for leaf in &slot.leaves {
                out.push(LeafRequest { tokens: observe_with(leaf, leaf.to_act(), perfect), actions: choice_fields(leaf), model });
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

    /// Search counters per network model id, summed over finished searches.
    pub fn stats(&self) -> [SearchStats; 2] {
        self.stats
    }
}

fn game_setup(cfg: &SelfPlayConfig, seed: u64, id: u64) -> ([String; 2], [u8; 2], u64) {
    let pairing = &cfg.pairings[(id / 2) as usize % cfg.pairings.len()];
    let [a, b] = cfg.seat_models;
    let models = if id % 2 == 0 { [a, b] } else { [b, a] };
    let deal = if a != b { id / 2 } else { id };
    ([pairing.0.clone(), pairing.1.clone()], models, seed.wrapping_mul(1_000_003).wrapping_add(deal))
}

fn search_config(cfg: &SelfPlayConfig, m: usize) -> Config {
    let mut c = cfg.search.clone();
    if !cfg.record {
        c.root_noise = 0.0;
    }
    c.maple_worlds = cfg.maple_worlds[m];
    c.maple_select = cfg.maple_select[m];
    c.maple_resample = cfg.maple_resample[m];
    c.pimc_worlds = cfg.pimc_worlds[m];
    c.perfect_obs = cfg.perfect_obs[m];
    c
}

/// Drive one slot until its search needs evaluations (or it has no game).
fn advance(slot: &mut Slot, cfg: &SelfPlayConfig) -> (Vec<Sample>, Option<GameResult>, Vec<(u8, SearchStats)>) {
    let mut samples = Vec::new();
    let mut result = None;
    let mut stats = Vec::new();
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
            let c = Config { root_noise: 0.0, maple_worlds: 0, pimc_worlds: 0, perfect_obs: false, ..cfg.search.clone() };
            let mut s = Search::new(game, c, slot.rng.next_u64());
            s.run(cfg.heuristic_sims, 8, &mut HeuristicEval);
            let c = s.best();
            game.apply(c);
            continue;
        }
        let m = model as usize;
        let sims = cfg.model_sims[m];
        let search = slot.search.get_or_insert_with(|| Search::new(game, search_config(cfg, m), slot.rng.next_u64()));
        if search.simulations >= sims {
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
                    tokens: observe_with(game, me, cfg.perfect_obs[m]),
                    actions: choice_fields(game),
                    policy: visits.iter().map(|v| v / total).collect(),
                    root_value: search.root_value(),
                    player: me,
                });
            }
            stats.push((model, search.stats));
            slot.search = None;
            game.apply(pick);
            continue;
        }
        let want = cfg.leaves_per_step.min((sims - search.simulations) as usize).max(1);
        let leaves = search.gather(want);
        if leaves.is_empty() {
            continue; // all simulations hit terminal states; loop again
        }
        slot.leaves = leaves;
        break;
    }
    (samples, result, stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selfplay_runs_games_and_records_samples() {
        run_games(0, 0, [false, false]);
    }

    #[test]
    fn selfplay_with_maple_runs_games_and_records_samples() {
        let sp = run_games(5, 0, [false, false]);
        let [s0, s1] = sp.stats();
        assert!(s0.leaf_worlds > s0.sims, "model 0 aggregates worlds: {s0:?}");
        assert!(s1.sims > 0 && s1.leaf_worlds <= s1.sims, "model 1 is plain IS-MCTS: {s1:?}");
        run_games(3, 0, [true, false]);
    }

    #[test]
    fn selfplay_with_pimc_runs_games_and_records_samples() {
        let sp = run_games(0, 3, [false, false]);
        let [s0, _] = sp.stats();
        assert!(s0.sims > 0 && s0.leaf_worlds <= s0.sims, "PIMC evaluates one world per simulation: {s0:?}");
    }

    #[test]
    fn each_model_plays_both_decks_of_a_pairing() {
        let cfg = SelfPlayConfig {
            model_sims: [8, 8],
            leaves_per_step: 4,
            temp_decisions: 0,
            search: Config::default(),
            pairings: vec![("Burn".into(), "Rally".into())],
            seat_models: [HEURISTIC, RANDOM],
            record: false,
            max_games: Some(4),
            heuristic_sims: 8,
            maple_worlds: [0, 0],
            maple_select: [MapleSelect::RefWorld; 2],
            maple_resample: [false; 2],
            perfect_obs: [false, false],
            pimc_worlds: [0, 0],
        };
        let mut sp = SelfPlay::new(4, cfg, 3);
        assert!(sp.gather().is_empty());
        let results = sp.take_results();
        assert_eq!(results.len(), 4);
        let mut plays = std::collections::HashMap::new();
        for r in &results {
            for seat in 0..2 {
                *plays.entry((r.models[seat], r.decks[seat].clone())).or_insert(0) += 1;
            }
        }
        for m in [HEURISTIC, RANDOM] {
            for d in ["Burn", "Rally"] {
                assert_eq!(plays.get(&(m, d.to_string())), Some(&2), "model {m} with {d}: {plays:?}");
            }
        }
    }

    fn two_model_config(seat_models: [u8; 2]) -> SelfPlayConfig {
        SelfPlayConfig {
            model_sims: [8, 8],
            leaves_per_step: 4,
            temp_decisions: 0,
            search: Config::default(),
            pairings: vec![("Burn".into(), "Rally".into()), ("Elves".into(), "Faeries".into())],
            seat_models,
            record: false,
            max_games: None,
            heuristic_sims: 8,
            maple_worlds: [5, 5],
            maple_select: [MapleSelect::Union, MapleSelect::RefWorld],
            maple_resample: [true, false],
            perfect_obs: [false, true],
            pimc_worlds: [0, 0],
        }
    }

    #[test]
    fn paired_games_swap_models_over_one_deal() {
        let cfg = two_model_config([0, 1]);
        for j in 0..4u64 {
            let (d0, m0, s0) = game_setup(&cfg, 7, 2 * j);
            let (d1, m1, s1) = game_setup(&cfg, 7, 2 * j + 1);
            assert_eq!(d0, d1);
            assert_eq!(s0, s1);
            assert_eq!(m0, [0, 1]);
            assert_eq!(m1, [1, 0]);
        }
        let selfplay = two_model_config([0, 0]);
        let seeds: std::collections::HashSet<u64> = (0..8).map(|id| game_setup(&selfplay, 7, id).2).collect();
        assert_eq!(seeds.len(), 8);
    }

    #[test]
    fn each_model_searches_with_its_own_settings() {
        let cfg = two_model_config([0, 1]);
        let (a, b) = (search_config(&cfg, 0), search_config(&cfg, 1));
        assert_eq!((a.maple_select, a.maple_resample, a.perfect_obs), (MapleSelect::Union, true, false));
        assert_eq!((b.maple_select, b.maple_resample, b.perfect_obs), (MapleSelect::RefWorld, false, true));
        assert_eq!(a.root_noise, 0.0, "no exploration noise outside training");
    }

    /// Model 0 with MAPLE or PIMC against plain model 1, with a stand-in network.
    fn run_games(maple: u32, pimc: u32, perfect_obs: [bool; 2]) -> SelfPlay {
        let cfg = SelfPlayConfig {
            model_sims: [16, 12],
            leaves_per_step: 4,
            temp_decisions: 10,
            search: Config::default(),
            pairings: vec![("Burn".into(), "Rally".into())],
            seat_models: [0, 1],
            record: true,
            max_games: Some(4),
            heuristic_sims: 16,
            maple_worlds: [maple, 0],
            maple_select: [MapleSelect::RefWorld; 2],
            maple_resample: [false; 2],
            perfect_obs,
            pimc_worlds: [pimc, 0],
        };
        let mut sp = SelfPlay::new(4, cfg, 1);
        let mut steps = 0;
        loop {
            let leaves = sp.gather();
            if leaves.is_empty() {
                break;
            }
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
        sp
    }
}
