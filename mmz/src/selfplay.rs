//! Many games at once, with leaf evaluations batched for the network.
//! [`HEURISTIC`] and [`RANDOM`] seats are played in Rust.

use crate::features::{choice_fields, observe_with, ACTION_FIELDS, TOKEN_FIELDS};
use crate::game::{Game, Outcome};
use crate::history::{History, HIST_FIELDS};
use crate::ismcts::{heuristic_value, Config, Eval, HeuristicEval, MapleSelect, Search, SearchStats, WorldSampler};
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
    pub world_sampler: [WorldSampler; 2],
    /// Multiplies the network's leaf values.
    pub value_scale: [f32; 2],
    /// Leaf values from [`heuristic_value`] instead of the network.
    pub heuristic_values: [bool; 2],
    /// Weight of the network's leaf value against [`heuristic_value`] (1 = network only).
    pub value_mix: [f32; 2],
    /// Playout-cap randomization while recording: the chance a move gets a
    /// full search (1 = always); the others use `cheap_sims` and train only the value.
    pub full_search_prob: f32,
    pub cheap_sims: u32,
    /// Record completed-Q policy targets instead of visit shares.
    pub completed_q: bool,
    /// Record training samples for heuristic seats too, which then sample
    /// their opening moves like network seats.
    pub record_heuristic: bool,
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
    pub hist: Vec<[u16; HIST_FIELDS]>,
    /// The opponent's unseen cards, its unknown hand first.
    pub pool: Vec<u16>,
    pub hand_n: u8,
    pub game: u64,
    pub turn: u32,
    /// The hand-written evaluation of the position, for the same player.
    pub heuristic: f32,
    /// The player to act, by index.
    pub player: u8,
    /// Whether the move had a full search (its policy is a training target).
    pub full: bool,
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

/// Candidate opponent hands to score before a search can start.
pub struct WorldRequest {
    pub tokens: Vec<[u16; TOKEN_FIELDS]>,
    pub hist: Vec<[u16; HIST_FIELDS]>,
    pub hands: Vec<Vec<u16>>,
    pub model: u8,
}

pub struct Batch {
    pub leaves: Vec<LeafRequest>,
    pub worlds: Vec<WorldRequest>,
}

impl Batch {
    pub fn is_empty(&self) -> bool {
        self.leaves.is_empty() && self.worlds.is_empty()
    }
}

struct Pending {
    tokens: Vec<[u16; TOKEN_FIELDS]>,
    actions: Vec<[u16; ACTION_FIELDS]>,
    policy: Vec<f32>,
    root_value: f32,
    player: PlayerId,
    hist: Vec<[u16; HIST_FIELDS]>,
    pool: Vec<u16>,
    hand_n: u8,
    turn: u32,
    heuristic: f32,
    full: bool,
}

struct Slot {
    id: u64,
    game: Option<Game>,
    decks: [String; 2],
    models: [u8; 2],
    search: Option<Search>,
    /// Whether the current search is a full one.
    full: bool,
    leaves: Vec<Game>,
    world: Option<WorldRequest>,
    history: History,
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
    /// Slots whose world requests are in the last batch, in order.
    world_plan: Vec<usize>,
    /// Search counters per network model id, over finished searches.
    stats: [SearchStats; 2],
}

impl SelfPlay {
    pub fn new(parallel: usize, cfg: SelfPlayConfig, seed: u64) -> SelfPlay {
        let slots = (0..parallel)
            .map(|i| Slot {
                id: 0,
                game: None,
                decks: [String::new(), String::new()],
                models: [0, 0],
                search: None,
                full: true,
                leaves: Vec::new(),
                world: None,
                history: History::default(),
                pending: Vec::new(),
                rng: SplitMix64::seed(seed ^ (i as u64).wrapping_mul(0x9E3779B97F4A7C15)),
            })
            .collect();
        SelfPlay {
            cfg,
            slots,
            started: 0,
            seed,
            samples: Vec::new(),
            results: Vec::new(),
            batch_plan: Vec::new(),
            world_plan: Vec::new(),
            stats: Default::default(),
        }
    }

    pub fn games_started(&self) -> u64 {
        self.started
    }

    /// Advance all games and return what needs the network (empty once all games are done).
    pub fn gather(&mut self) -> Batch {
        loop {
            let batch = self.gather_once();
            let more = self.slots.iter().any(|s| s.game.is_some()) || self.cfg.max_games.is_none_or(|m| self.started < m);
            if !batch.is_empty() || !more {
                return batch;
            }
        }
    }

    fn gather_once(&mut self) -> Batch {
        // Start new games single-threaded (game ids must be deterministic).
        for slot in self.slots.iter_mut() {
            if slot.game.is_none() && self.cfg.max_games.is_none_or(|m| self.started < m) {
                let id = self.started;
                self.started += 1;
                let (decks, models, game_seed) = game_setup(&self.cfg, self.seed, id);
                slot.game = Some(Game::new(&decks[0], &decks[1], game_seed).expect("valid deck names"));
                slot.id = id;
                slot.decks = decks;
                slot.models = models;
                slot.search = None;
                slot.world = None;
                slot.history = History::default();
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
        self.world_plan.clear();
        let mut out = Vec::new();
        let mut worlds = Vec::new();
        for (i, slot) in self.slots.iter_mut().enumerate() {
            if let Some(w) = slot.world.take() {
                self.world_plan.push(i);
                worlds.push(w);
            }
            if !slot.leaves.is_empty() {
                self.batch_plan.push((i, slot.leaves.len()));
            }
        }
        // Observations in parallel, kept in slot order.
        let perfect = self.cfg.perfect_obs;
        let per_slot: Vec<Vec<LeafRequest>> = self
            .slots
            .par_iter()
            .map(|slot| {
                let Some(game) = slot.game.as_ref().filter(|_| !slot.leaves.is_empty()) else { return Vec::new() };
                let model = slot.models[game.to_act().index()];
                slot.leaves
                    .iter()
                    .map(|leaf| LeafRequest {
                        tokens: observe_with(leaf, leaf.to_act(), perfect[model as usize]),
                        actions: choice_fields(leaf),
                        model,
                    })
                    .collect()
            })
            .collect();
        out.extend(per_slot.into_iter().flatten());
        Batch { leaves: out, worlds }
    }

    /// Results for the batch returned by the last [`Self::gather`], in order.
    pub fn feed(&mut self, evals: Vec<Eval>, world_scores: Vec<Vec<f32>>) {
        assert_eq!(world_scores.len(), self.world_plan.len(), "one score list per world request");
        for (&i, scores) in self.world_plan.iter().zip(&world_scores) {
            self.slots[i].search.as_mut().unwrap().set_world_scores(scores);
        }
        self.world_plan.clear();
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
        let (scale, heuristic, mix) = (self.cfg.value_scale, self.cfg.heuristic_values, self.cfg.value_mix);
        self.slots.par_iter_mut().zip(by_index).for_each(|(slot, evals)| {
            if let Some(mut evals) = evals {
                let leaves = std::mem::take(&mut slot.leaves);
                assert_eq!(leaves.len(), evals.len());
                let m = slot.models[slot.game.as_ref().unwrap().to_act().index()] as usize;
                for (e, g) in evals.iter_mut().zip(&leaves) {
                    e.value = if heuristic[m] {
                        heuristic_value(g, g.to_act())
                    } else if mix[m] == 1.0 {
                        e.value * scale[m]
                    } else {
                        mix[m] * e.value * scale[m] + (1.0 - mix[m]) * heuristic_value(g, g.to_act())
                    };
                }
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

/// The move to play: in proportion to visits for the opening decisions, then the best.
fn pick_move(search: &Search, game: &Game, cfg: &SelfPlayConfig, rng: &mut SplitMix64) -> usize {
    if game.decisions >= cfg.temp_decisions {
        return search.best();
    }
    let visits = search.root_visits();
    let total: f32 = visits.iter().sum::<f32>().max(1e-6);
    let mut r = unit(rng) * total;
    for (i, &v) in visits.iter().enumerate() {
        if r < v {
            return i;
        }
        r -= v;
    }
    visits.len() - 1
}

/// A training sample of `search`'s root, for the player to act.
fn pending_sample(search: &Search, game: &Game, history: &History, cfg: &SelfPlayConfig, perfect: bool, full: bool) -> Pending {
    let me = game.to_act();
    let (hand, rest) = game.hidden_pool(me);
    let policy = if cfg.completed_q {
        search.root_policy_target()
    } else {
        let visits = search.root_visits();
        let total: f32 = visits.iter().sum::<f32>().max(1e-6);
        visits.iter().map(|v| v / total).collect()
    };
    Pending {
        tokens: observe_with(game, me, perfect),
        actions: choice_fields(game),
        policy,
        root_value: search.root_value(),
        player: me,
        hist: history.tokens(me, game.state.turn),
        hand_n: hand.len() as u8,
        pool: [hand, rest].concat(),
        turn: game.state.turn,
        heuristic: heuristic_value(game, me),
        full,
    }
}

/// Uniform in [0, 1).
fn unit(rng: &mut SplitMix64) -> f32 {
    (rng.next_u64() >> 11) as f32 / (1u64 << 53) as f32
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
    c.world_sampler = cfg.world_sampler[m];
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
                samples.push(Sample {
                    tokens: p.tokens,
                    actions: p.actions,
                    policy: p.policy,
                    z,
                    root_value: p.root_value,
                    hist: p.hist,
                    pool: p.pool,
                    hand_n: p.hand_n,
                    game: slot.id,
                    turn: p.turn,
                    heuristic: p.heuristic,
                    player: p.player.index() as u8,
                    full: p.full,
                });
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
            let c = (slot.rng.next_u64() % n as u64) as usize;
            slot.history.record(game, c);
            game.apply(c);
            continue;
        }
        if model == HEURISTIC {
            let c = Config {
                root_noise: 0.0,
                maple_worlds: 0,
                pimc_worlds: 0,
                perfect_obs: false,
                world_sampler: WorldSampler::Uniform,
                ..cfg.search.clone()
            };
            let mut s = Search::new(game, c, slot.rng.next_u64());
            s.run(cfg.heuristic_sims, 8, &mut HeuristicEval);
            let c = if cfg.record && cfg.record_heuristic {
                let pick = pick_move(&s, game, cfg, &mut slot.rng);
                slot.pending.push(pending_sample(&s, game, &slot.history, cfg, false, true));
                pick
            } else {
                s.best()
            };
            slot.history.record(game, c);
            game.apply(c);
            continue;
        }
        let m = model as usize;
        if slot.search.is_none() {
            let mut c = search_config(cfg, m);
            slot.full = !cfg.record || cfg.full_search_prob >= 1.0 || unit(&mut slot.rng) < cfg.full_search_prob;
            if !slot.full {
                c.root_noise = 0.0;
            }
            slot.search = Some(Search::new(game, c, slot.rng.next_u64()));
        }
        let sims = if slot.full { cfg.model_sims[m] } else { cfg.cheap_sims };
        let search = slot.search.as_mut().expect("made above");
        if let Some(hands) = search.world_request() {
            let hist = slot.history.tokens(me, game.state.turn);
            slot.world = Some(WorldRequest { tokens: observe_with(game, me, false), hist, hands, model });
            break;
        }
        if search.simulations >= sims {
            let search = slot.search.take().expect("made above");
            let pick = pick_move(&search, game, cfg, &mut slot.rng);
            if cfg.record && model == 0 {
                let p = pending_sample(&search, game, &slot.history, cfg, cfg.perfect_obs[m], slot.full);
                slot.pending.push(p);
            }
            stats.push((model, search.stats));
            slot.history.record(game, pick);
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
        run_games(0, 0, [false, false], WorldSampler::Uniform);
    }

    #[test]
    fn selfplay_with_maple_runs_games_and_records_samples() {
        let sp = run_games(5, 0, [false, false], WorldSampler::Uniform);
        let [s0, s1] = sp.stats();
        assert!(s0.leaf_worlds > s0.sims, "model 0 aggregates worlds: {s0:?}");
        assert!(s1.sims > 0 && s1.leaf_worlds <= s1.sims, "model 1 is plain IS-MCTS: {s1:?}");
        run_games(3, 0, [true, false], WorldSampler::Uniform);
    }

    #[test]
    fn selfplay_with_pimc_runs_games_and_records_samples() {
        let sp = run_games(0, 3, [false, false], WorldSampler::Uniform);
        let [s0, _] = sp.stats();
        assert!(s0.sims > 0 && s0.leaf_worlds <= s0.sims, "PIMC evaluates one world per simulation: {s0:?}");
    }

    #[test]
    fn selfplay_with_scored_worlds_asks_for_scores() {
        let sp = run_games(5, 0, [false, false], WorldSampler::Scored);
        let [s0, s1] = sp.stats();
        assert!(s0.world_requests > 0, "{s0:?}");
        assert_eq!(s1.world_requests, 0);
        run_games(0, 0, [false, false], WorldSampler::Scored);
        run_games(0, 3, [false, false], WorldSampler::Oracle);
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
            world_sampler: [WorldSampler::Uniform; 2],
            value_scale: [1.0; 2],
            heuristic_values: [false; 2],
            value_mix: [1.0; 2],
            full_search_prob: 1.0,
            cheap_sims: 0,
            completed_q: false,
            record_heuristic: false,
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
            world_sampler: [WorldSampler::Scored, WorldSampler::Uniform],
            value_scale: [1.0; 2],
            heuristic_values: [false; 2],
            value_mix: [1.0; 2],
            full_search_prob: 1.0,
            cheap_sims: 0,
            completed_q: false,
            record_heuristic: false,
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

    fn test_config() -> SelfPlayConfig {
        SelfPlayConfig {
            model_sims: [16, 12],
            leaves_per_step: 4,
            temp_decisions: 10,
            search: Config::default(),
            pairings: vec![("Burn".into(), "Rally".into())],
            seat_models: [0, 1],
            record: true,
            max_games: Some(4),
            heuristic_sims: 16,
            maple_worlds: [0, 0],
            maple_select: [MapleSelect::RefWorld; 2],
            maple_resample: [false; 2],
            perfect_obs: [false, false],
            pimc_worlds: [0, 0],
            world_sampler: [WorldSampler::Uniform; 2],
            value_scale: [1.0; 2],
            heuristic_values: [false; 2],
            value_mix: [1.0; 2],
            full_search_prob: 1.0,
            cheap_sims: 0,
            completed_q: false,
            record_heuristic: false,
        }
    }

    /// Model 0 with MAPLE or PIMC against plain model 1, with a stand-in network.
    fn run_games(maple: u32, pimc: u32, perfect_obs: [bool; 2], sampler: WorldSampler) -> SelfPlay {
        let cfg = SelfPlayConfig {
            maple_worlds: [maple, 0],
            perfect_obs,
            pimc_worlds: [pimc, 0],
            world_sampler: [sampler, WorldSampler::Uniform],
            ..test_config()
        };
        let mut sp = SelfPlay::new(4, cfg, 1);
        let mut steps = 0;
        loop {
            let batch = sp.gather();
            if batch.is_empty() {
                break;
            }
            let evals = batch.leaves.iter().map(|l| Eval { priors: vec![1.0; l.actions.len()], value: 0.0 }).collect();
            let scores = batch.worlds.iter().map(|w| w.hands.iter().map(|h| h.len() as f32 + h[0] as f32 / 1e4).collect()).collect();
            sp.feed(evals, scores);
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
            assert!(s.hand_n as usize <= s.pool.len());
            assert!(s.hist.len() <= crate::history::MAX_HIST);
            assert!(s.heuristic.abs() <= 1.0);
        }
        assert!(samples.iter().any(|s| !s.hist.is_empty()));
        sp
    }

    /// Model 0's recorded root values, with a stand-in network that rates every leaf 7.
    fn root_values(value_scale: [f32; 2], heuristic_values: [bool; 2]) -> Vec<f32> {
        samples_with(SelfPlayConfig { value_scale, heuristic_values, ..test_config() }).iter().map(|s| s.root_value).collect()
    }

    fn samples_with(cfg: SelfPlayConfig) -> Vec<Sample> {
        let mut sp = SelfPlay::new(4, cfg, 1);
        loop {
            let batch = sp.gather();
            if batch.is_empty() {
                break;
            }
            let evals = batch.leaves.iter().map(|l| Eval { priors: vec![1.0; l.actions.len()], value: 7.0 }).collect();
            sp.feed(evals, Vec::new());
        }
        sp.take_samples()
    }

    #[test]
    fn heuristic_seats_record_samples_without_a_network() {
        let cfg = SelfPlayConfig {
            seat_models: [HEURISTIC, HEURISTIC],
            record_heuristic: true,
            heuristic_sims: 16,
            max_games: Some(6),
            ..test_config()
        };
        let mut sp = SelfPlay::new(2, cfg, 3);
        assert!(sp.gather().is_empty(), "no leaf ever needs the network");
        assert_eq!(sp.take_results().len(), 6, "every game is played in one gather");
        let samples = sp.take_samples();
        assert!(samples.len() > 100);
        assert!(samples.iter().all(|s| s.full && (s.policy.iter().sum::<f32>() - 1.0).abs() < 1e-3));
    }

    #[test]
    fn a_zero_value_mix_is_the_heuristic() {
        let mixed = samples_with(SelfPlayConfig { value_mix: [0.0; 2], ..test_config() });
        let mixed: Vec<f32> = mixed.iter().map(|s| s.root_value).collect();
        assert_eq!(mixed, root_values([1.0; 2], [true; 2]));
        assert_ne!(root_values([1.0; 2], [false; 2]), root_values([1.0; 2], [true; 2]));
    }

    #[test]
    fn samples_come_in_whole_games_and_cheap_moves_are_marked() {
        let samples = samples_with(SelfPlayConfig { full_search_prob: 0.25, cheap_sims: 4, ..test_config() });
        let full = samples.iter().filter(|s| s.full).count();
        assert!(full > 0 && full < samples.len() / 2, "{full} of {}", samples.len());
        let mut seen = Vec::new();
        for w in samples.windows(2) {
            if w[0].game != w[1].game {
                assert!(!seen.contains(&w[1].game), "game {} is split", w[1].game);
                seen.push(w[0].game);
            }
        }
        assert!(samples.iter().any(|s| s.player == 1) && samples.iter().any(|s| s.player == 0));
        // Off by default: every move a full search, as before.
        assert!(samples_with(test_config()).iter().all(|s| s.full));
    }

    #[test]
    fn leaf_values_can_be_scaled_or_taken_from_the_heuristic() {
        let in_range = |v: &[f32]| !v.is_empty() && v.iter().all(|x| x.abs() <= 1.0 + 1e-4);
        assert!(!in_range(&root_values([1.0; 2], [false; 2])));
        assert!(!in_range(&root_values([1.0, 1.0 / 7.0], [false, true])), "model 1's settings leaked into model 0");
        assert!(in_range(&root_values([1.0 / 7.0, 1.0], [false; 2])));
        assert!(in_range(&root_values([1.0; 2], [true, false])));
    }
}
