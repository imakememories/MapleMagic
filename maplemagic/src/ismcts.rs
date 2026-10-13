//! Information-set MCTS with PUCT, and the MAPLE and PIMC (AlphaZe\*\*)
//! variants. [`Search::gather`] and [`Search::feed`] let a batched network
//! evaluate leaves outside the search.

use crate::features::{choice_fields, observe_with};
use crate::game::{Game, Outcome};
use mtg_kernel::card_def::CARD_DEFS;
use mtg_kernel::ids::PlayerId;
use mtg_kernel::state::SplitMix64;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Turn buckets of a retention table: turns 0..=10, then 11 and later.
pub const RETENTION_BUCKETS: usize = 12;

/// How MAPLE picks the acting player and legal keys when worlds disagree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MapleSelect {
    /// A random alive world decides.
    #[default]
    RefWorld,
    /// The majority player acts, over the union of legal keys.
    Union,
}

/// Where the opponent's hidden hand in each world comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WorldSampler {
    #[default]
    Uniform,
    /// The true hand: a diagnostic that cheats.
    Oracle,
    /// Drawn from per-turn log-odds of a card still being in hand.
    Retention,
    /// Candidate hands scored by the caller, the lowest kept.
    Scored,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub c_puct: f32,
    /// Dirichlet noise weight at the root (0 disables).
    pub root_noise: f32,
    pub dirichlet_alpha: f32,
    /// Value assumed for unvisited edges, relative to the parent's mean.
    pub fpu_reduction: f32,
    pub virtual_loss: f32,
    /// MAPLE world count (0 = plain SO-ISMCTS).
    pub maple_worlds: u32,
    pub maple_select: MapleSelect,
    /// Draw fresh worlds for every simulation instead of once per search.
    pub maple_resample: bool,
    /// Evaluate identical leaf observations once (off for rollouts, which read hidden state).
    pub maple_dedupe: bool,
    /// Observation encoding the evaluator uses (only affects dedupe).
    pub perfect_obs: bool,
    /// PIMC world count (0 = off): independent trees in fixed worlds.
    pub pimc_worlds: u32,
    pub world_sampler: WorldSampler,
    /// Uniform worlds drawn per move as candidates for scoring.
    pub world_candidates: u32,
    /// Hands an IS-MCTS search draws its worlds from.
    pub world_pool: u32,
    /// 0 keeps the best-scored hands; above 0, samples them at this temperature.
    pub world_temp: f32,
    /// Log-odds per (turn bucket, card definition).
    pub retention: Option<Arc<Vec<f32>>>,
    /// A simulation that reaches a node still awaiting its evaluation stops
    /// there and ends the batch, instead of choosing without priors.
    pub collisions: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            c_puct: 1.0,
            root_noise: 0.0,
            dirichlet_alpha: 0.3,
            fpu_reduction: 0.2,
            virtual_loss: 1.0,
            maple_worlds: 0,
            maple_select: MapleSelect::RefWorld,
            maple_resample: false,
            maple_dedupe: true,
            perfect_obs: false,
            pimc_worlds: 0,
            world_sampler: WorldSampler::Uniform,
            world_candidates: 50,
            world_pool: 16,
            world_temp: 0.0,
            retention: None,
            collisions: false,
        }
    }
}

/// Counters for how the search spent its evaluations.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SearchStats {
    /// Simulations completed (leaf evaluations or terminal backups).
    pub sims: u64,
    /// Worlds that reached a leaf, before dedupe.
    pub leaf_worlds: u64,
    /// Leaves actually sent for evaluation.
    pub unique_leaves: u64,
    /// Worlds dropped because the selected key was illegal there.
    pub dropped_illegal: u64,
    /// Worlds dropped because a different player was to act.
    pub dropped_diverged: u64,
    /// Worlds that ended the game along a path.
    pub terminal_worlds: u64,
    pub no_world_sims: u64,
    pub stale_priors: u64,
    pub world_requests: u64,
    pub world_skips: u64,
    /// Expanded leaves, with their summed depth in decisions below the root
    /// and in turns past the root's.
    pub leaves: u64,
    pub leaf_decisions: u64,
    pub leaf_turns: u64,
    /// Simulations stopped at a node awaiting its evaluation (not counted in `sims`).
    pub collisions: u64,
    /// Selections at a node awaiting its evaluation, made without priors.
    pub blind_selects: u64,
}

impl std::ops::AddAssign for SearchStats {
    fn add_assign(&mut self, o: SearchStats) {
        self.sims += o.sims;
        self.leaf_worlds += o.leaf_worlds;
        self.unique_leaves += o.unique_leaves;
        self.dropped_illegal += o.dropped_illegal;
        self.dropped_diverged += o.dropped_diverged;
        self.terminal_worlds += o.terminal_worlds;
        self.no_world_sims += o.no_world_sims;
        self.stale_priors += o.stale_priors;
        self.world_requests += o.world_requests;
        self.world_skips += o.world_skips;
        self.leaves += o.leaves;
        self.leaf_decisions += o.leaf_decisions;
        self.leaf_turns += o.leaf_turns;
        self.collisions += o.collisions;
        self.blind_selects += o.blind_selects;
    }
}

/// Priors aligned with `leaf.choices()`; value in [-1, 1] for `leaf.to_act()`.
#[derive(Debug, Clone)]
pub struct Eval {
    pub priors: Vec<f32>,
    pub value: f32,
}

pub trait Evaluator {
    fn eval(&mut self, leaves: &[Game]) -> Vec<Eval>;
}

const NONE: u32 = u32::MAX;
/// Tries to find a root determinization where the observer is still to act.
const ROOT_TRIES: u32 = 8;

/// FxHash, as in rustc: fast, and enough to tell a simulation's worlds apart.
#[derive(Default)]
struct FxHasher(u64);

impl FxHasher {
    fn add(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}

impl Hasher for FxHasher {
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut word = [0u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.add(u64::from_le_bytes(word));
        }
    }

    fn write_u8(&mut self, n: u8) {
        self.add(u64::from(n));
    }

    fn write_u16(&mut self, n: u16) {
        self.add(u64::from(n));
    }

    fn write_u32(&mut self, n: u32) {
        self.add(u64::from(n));
    }

    fn write_u64(&mut self, n: u64) {
        self.add(n);
    }

    fn write_usize(&mut self, n: usize) {
        self.add(n as u64);
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone)]
struct Edge {
    key: u64,
    prior: f32,
    visits: f32,
    /// Sum of values from the point of view of the player choosing here.
    value_sum: f32,
    avail: f32,
    /// Child node per player to act next (indexed by `PlayerId::index`).
    child: [u32; 2],
}

#[derive(Debug, Clone)]
struct Node {
    edges: Vec<Edge>,
    expanded: bool,
    /// The player choosing at this node.
    actor: PlayerId,
}

impl Node {
    fn new(actor: PlayerId) -> Node {
        Node { edges: Vec::new(), expanded: false, actor }
    }
}

struct PendingSim {
    path: Vec<(u32, usize, PlayerId)>,
    leaf_node: u32,
    /// Weight per returned leaf (duplicate count after dedupe).
    weights: Vec<f32>,
    /// Terminal worlds met along the path, observer's point of view.
    term_sum: f32,
    term_n: f32,
}

pub struct Search {
    root: Game,
    /// The root's deal plans, without and with the observer's library kept.
    deal_plans: [Option<crate::game::DealPlan>; 2],
    observer: PlayerId,
    cfg: Config,
    nodes: Vec<Node>,
    rng: SplitMix64,
    pending: Vec<PendingSim>,
    /// Fixed worlds: MAPLE's, or PIMC's (world `i` searched from root node `i`).
    worlds: Vec<Game>,
    /// PIMC: the world the next simulation descends in.
    next_world: usize,
    /// Opponent hands the worlds are dealt from (None: dealt at random).
    hands: Option<Vec<Vec<u16>>>,
    /// Scored sampling: distinct candidate hands and how often each was drawn.
    candidates: Vec<(Vec<u16>, f32)>,
    /// Per root node: the evaluator's value of the root, for the observer.
    root_evals: Vec<Option<f32>>,
    /// Per root node: (key, prior) before root noise.
    clean_priors: Vec<Vec<(u64, f32)>>,
    /// With collisions: the (node, edge) pairs whose `avail` this simulation raised.
    avail_log: Vec<(u32, u32)>,
    pub simulations: u32,
    pub stats: SearchStats,
}

fn terminal_value(o: Outcome, pov: PlayerId) -> f32 {
    match o {
        Outcome::Win(p) if p == pov => 1.0,
        Outcome::Win(_) => -1.0,
        Outcome::Draw | Outcome::Aborted => 0.0,
    }
}

/// Each key's weighted mean prior over the worlds where it is legal, normalized.
pub fn aggregate_priors(per_world: &[(&[u64], &[f32], f32)]) -> Vec<(u64, f32)> {
    let mut acc: Vec<(u64, f32, f32)> = Vec::new();
    for &(keys, priors, w) in per_world {
        let total: f32 = priors.iter().take(keys.len()).sum::<f32>().max(1e-6);
        for (&k, &p) in keys.iter().zip(priors) {
            match acc.iter_mut().find(|a| a.0 == k) {
                Some(a) => {
                    a.1 += w * p / total;
                    a.2 += w;
                }
                None => acc.push((k, w * p / total, w)),
            }
        }
    }
    let means: Vec<(u64, f32)> = acc.into_iter().map(|(k, s, w)| (k, if w > 0.0 { s / w } else { 0.0 })).collect();
    let total: f32 = means.iter().map(|m| m.1).sum();
    if total <= 0.0 {
        let n = means.len().max(1) as f32;
        return means.into_iter().map(|(k, _)| (k, 1.0 / n)).collect();
    }
    means.into_iter().map(|(k, p)| (k, p / total)).collect()
}

fn keys_of(g: &Game) -> Vec<u64> {
    g.choices().iter().map(|c| c.key).collect()
}

impl Search {
    pub fn new(root: &Game, cfg: Config, seed: u64) -> Search {
        let observer = root.to_act();
        let mut s = Search {
            observer,
            root: root.clone(),
            deal_plans: [None, None],
            nodes: vec![Node::new(observer)],
            rng: SplitMix64::seed(seed),
            pending: Vec::new(),
            worlds: Vec::new(),
            next_world: 0,
            hands: None,
            candidates: Vec::new(),
            root_evals: Vec::new(),
            clean_priors: Vec::new(),
            avail_log: Vec::new(),
            simulations: 0,
            stats: SearchStats::default(),
            cfg,
        };
        assert!(s.cfg.maple_worlds == 0 || s.cfg.pimc_worlds == 0, "MAPLE and PIMC are exclusive");
        assert!(!(s.cfg.maple_resample && s.cfg.world_sampler == WorldSampler::Scored), "scored worlds are fixed per move");
        if !s.root.is_over() {
            s.choose_hands();
            if s.candidates.is_empty() {
                s.fix_worlds();
            }
        }
        s
    }

    fn hands_wanted(&self) -> usize {
        match (self.cfg.maple_worlds, self.cfg.pimc_worlds) {
            (0, 0) => self.cfg.world_pool.max(1) as usize,
            (k, 0) => k as usize,
            (_, k) => k as usize,
        }
    }

    fn choose_hands(&mut self) {
        match self.cfg.world_sampler {
            WorldSampler::Uniform | WorldSampler::Retention => {}
            WorldSampler::Oracle => self.hands = Some(vec![self.root.hidden_pool(self.observer).0]),
            WorldSampler::Scored => {
                let mut cands: Vec<(Vec<u16>, f32)> = Vec::new();
                for _ in 0..self.cfg.world_candidates {
                    let Some(g) = self.determinize_root(None) else { continue };
                    let mut h = g.hidden_pool(self.observer).0;
                    h.sort_unstable();
                    match cands.iter_mut().find(|c| c.0 == h) {
                        Some(c) => c.1 += 1.0,
                        None => cands.push((h, 1.0)),
                    }
                }
                if cands.len() <= self.hands_wanted() {
                    self.stats.world_skips += 1;
                    self.hands = (!cands.is_empty()).then(|| cands.into_iter().map(|c| c.0).collect());
                } else {
                    self.stats.world_requests += 1;
                    self.candidates = cands;
                }
            }
        }
    }

    /// The fixed worlds of MAPLE (unless resampling) and PIMC.
    fn fix_worlds(&mut self) {
        if self.cfg.maple_worlds > 0 && !self.cfg.maple_resample {
            self.worlds = self.sample_worlds(self.cfg.maple_worlds);
        }
        if self.cfg.pimc_worlds > 0 {
            self.worlds = self.sample_worlds(self.cfg.pimc_worlds);
            for _ in 1..self.worlds.len() {
                self.nodes.push(Node::new(self.observer));
            }
        }
    }

    /// Candidate opponent hands to score, lower being more likely, before the search can run.
    pub fn world_request(&self) -> Option<Vec<Vec<u16>>> {
        (!self.candidates.is_empty()).then(|| self.candidates.iter().map(|c| c.0.clone()).collect())
    }

    pub fn set_world_scores(&mut self, scores: &[f32]) {
        assert_eq!(scores.len(), self.candidates.len());
        let t = self.cfg.world_temp;
        let cands = std::mem::take(&mut self.candidates);
        let mut keyed: Vec<(f32, usize)> = Vec::with_capacity(cands.len());
        for (i, &d) in scores.iter().enumerate() {
            let k = if t > 0.0 { d - t * (cands[i].1.ln() + gumbel(&mut self.rng)) } else { d };
            keyed.push((k, i));
        }
        keyed.sort_by(|a, b| a.0.total_cmp(&b.0));
        self.hands = Some(keyed.iter().take(self.hands_wanted()).map(|&(_, i)| cands[i].0.clone()).collect());
        self.fix_worlds();
    }

    /// The opponent hand for world `i`, if not dealt at random.
    fn world_hand(&mut self, i: usize) -> Option<Vec<u16>> {
        if let Some(hs) = &self.hands {
            return Some(hs[i % hs.len()].clone());
        }
        (self.cfg.world_sampler == WorldSampler::Retention).then(|| self.retention_hand())
    }

    fn retention_hand(&mut self) -> Vec<u16> {
        let table = self.cfg.retention.clone().expect("a retention table");
        let (hand, rest) = self.root.hidden_pool(self.observer);
        let mut pool: Vec<u16> = hand.iter().chain(&rest).copied().collect();
        pool.sort_unstable();
        let row = (self.root.state.turn as usize).min(RETENTION_BUCKETS - 1) * CARD_DEFS.len();
        let mut keyed: Vec<(f32, u16)> = pool.iter().map(|&d| (table[row + d as usize] + gumbel(&mut self.rng), d)).collect();
        keyed.sort_by(|a, b| b.0.total_cmp(&a.0));
        keyed.iter().take(hand.len()).map(|x| x.1).collect()
    }

    /// Up to `k` determinizations where the observer is to act.
    fn sample_worlds(&mut self, k: u32) -> Vec<Game> {
        let k = k as usize;
        let mut out = Vec::with_capacity(k);
        let mut tries = 0;
        while out.len() < k && tries < k as u32 * ROOT_TRIES {
            tries += 1;
            let hand = self.world_hand(out.len());
            if let Some(g) = self.determinize_root(hand.as_deref()) {
                out.push(g);
            }
        }
        out
    }

    /// Leaves for up to `max_leaves` simulations; evaluate them and pass the results to [`Self::feed`] in order.
    pub fn gather(&mut self, max_leaves: usize) -> Vec<Game> {
        assert!(self.candidates.is_empty(), "set_world_scores first");
        if self.cfg.maple_worlds > 0 {
            return self.gather_maple(max_leaves);
        }
        let mut leaves = Vec::new();
        let mut attempts = 0;
        while leaves.len() < max_leaves && attempts < max_leaves * 4 {
            attempts += 1;
            let start = if self.cfg.pimc_worlds > 0 {
                // PIMC: the worlds take turns, each from its own root.
                (!self.worlds.is_empty()).then(|| {
                    let w = self.next_world % self.worlds.len();
                    self.next_world += 1;
                    (self.worlds[w].clone(), w as u32)
                })
            } else {
                self.root_world().map(|g| (g, 0))
            };
            let Some((g, root)) = start else {
                // No world keeps the observer to act.
                self.simulations += 1;
                self.stats.sims += 1;
                self.stats.no_world_sims += 1;
                continue;
            };
            if !self.descend(g, root, &mut leaves) {
                if self.cfg.pimc_worlds > 0 {
                    // This world goes first in the next batch.
                    self.next_world -= 1;
                }
                break;
            }
        }
        leaves
    }

    /// One simulation in world `g` from root node `root`; false if it collided.
    fn descend(&mut self, mut g: Game, root: u32, leaves: &mut Vec<Game>) -> bool {
        let mut node = root;
        let mut path: Vec<(u32, usize, PlayerId)> = Vec::new();
        self.avail_log.clear();
        loop {
            if let Some(o) = g.outcome() {
                let v = terminal_value(o, self.observer);
                self.backup_observer(&path, v);
                self.simulations += 1;
                self.stats.sims += 1;
                self.stats.terminal_worlds += 1;
                return true;
            }
            if !self.nodes[node as usize].expanded {
                // Marked now so other simulations in this batch don't queue it again.
                self.nodes[node as usize].expanded = true;
                self.add_virtual_loss(&path);
                self.note_leaf(path.len(), &g);
                self.pending.push(PendingSim { path, leaf_node: node, weights: vec![1.0], term_sum: 0.0, term_n: 0.0 });
                self.stats.leaf_worlds += 1;
                self.stats.unique_leaves += 1;
                leaves.push(g);
                return true;
            }
            if self.collides(node) {
                return false;
            }
            let chooser = g.to_act();
            let e = self.select(node, &keys_of(&g));
            let key = self.nodes[node as usize].edges[e].key;
            let idx = g.choices().iter().position(|c| c.key == key).expect("selected key is legal");
            path.push((node, e, chooser));
            g.apply(idx);
            if !g.is_over() {
                node = self.child(node, e, g.to_act());
            }
        }
    }

    /// With collisions, whether `node` (expanded) still awaits its evaluation. If
    /// so the simulation stops: no virtual loss was added, so only `avail` is undone.
    fn collides(&mut self, node: u32) -> bool {
        if !self.cfg.collisions || !self.nodes[node as usize].edges.is_empty() {
            return false;
        }
        for &(n, e) in &self.avail_log {
            self.nodes[n as usize].edges[e as usize].avail -= 1.0;
        }
        self.stats.collisions += 1;
        true
    }

    fn note_leaf(&mut self, decisions: usize, leaf: &Game) {
        self.stats.leaves += 1;
        self.stats.leaf_decisions += decisions as u64;
        self.stats.leaf_turns += u64::from(leaf.state.turn.saturating_sub(self.root.state.turn));
    }

    /// A determinization of the root where the observer is to act.
    fn root_world(&mut self) -> Option<Game> {
        (0..ROOT_TRIES).find_map(|_| {
            let i = self.rng.next_u64() as usize;
            let hand = self.world_hand(i);
            self.determinize_root(hand.as_deref())
        })
    }

    /// One determinization, retried keeping the observer's library if resampling it breaks the root.
    fn determinize_root(&mut self, hand: Option<&[u16]>) -> Option<Game> {
        let seed = self.rng.next_u64();
        let observer = self.observer;
        let ok = |g: &Game| !g.is_over() && g.to_act() == observer;
        for keep_own_library in [false, true] {
            let k = usize::from(keep_own_library);
            if self.deal_plans[k].is_none() {
                self.deal_plans[k] = Some(self.root.deal_plan(self.observer, keep_own_library));
            }
            let plan = self.deal_plans[k].as_ref().expect("made above");
            let mut g = self.root.clone();
            g.determinize_planned(plan, self.observer, seed, hand);
            if ok(&g) {
                return Some(g);
            }
        }
        None
    }

    fn gather_maple(&mut self, max_sims: usize) -> Vec<Game> {
        let mut leaves = Vec::new();
        let mut done = 0;
        let mut attempts = 0;
        let mut collided = false;
        while done < max_sims && attempts < max_sims * 4 && !collided {
            attempts += 1;
            self.avail_log.clear();
            let mut alive = if self.cfg.maple_resample { self.sample_worlds(self.cfg.maple_worlds) } else { self.worlds.clone() };
            if alive.is_empty() {
                self.simulations += 1;
                self.stats.sims += 1;
                self.stats.no_world_sims += 1;
                done += 1;
                continue;
            }
            let mut node = 0u32;
            let mut via: Option<(u32, usize)> = None;
            let mut path: Vec<(u32, usize, PlayerId)> = Vec::new();
            let (mut term_sum, mut term_n) = (0.0f32, 0.0f32);
            loop {
                // Worlds whose game ended contribute their result.
                let before = alive.len();
                alive.retain(|w| match w.outcome() {
                    Some(o) => {
                        term_sum += terminal_value(o, self.observer);
                        term_n += 1.0;
                        false
                    }
                    None => true,
                });
                self.stats.terminal_worlds += (before - alive.len()) as u64;
                if alive.is_empty() {
                    self.backup_observer(&path, term_sum / term_n);
                    self.simulations += 1;
                    self.stats.sims += 1;
                    done += 1;
                    break;
                }
                // Who acts here; the reference world moves to index 0.
                let actor = match self.cfg.maple_select {
                    MapleSelect::RefWorld => {
                        let r = (self.rng.next_u64() % alive.len() as u64) as usize;
                        alive.swap(0, r);
                        alive[0].to_act()
                    }
                    MapleSelect::Union => {
                        let n_obs = alive.iter().filter(|w| w.to_act() == self.observer).count();
                        let n_opp = alive.len() - n_obs;
                        if n_obs >= n_opp { self.observer } else { self.observer.opponent() }
                    }
                };
                let before = alive.len();
                alive.retain(|w| w.to_act() == actor);
                self.stats.dropped_diverged += (before - alive.len()) as u64;
                if let Some((pn, pe)) = via {
                    node = self.child(pn, pe, actor);
                }
                if !self.nodes[node as usize].expanded {
                    self.nodes[node as usize].expanded = true;
                    self.add_virtual_loss(&path);
                    self.note_leaf(path.len(), &alive[0]);
                    self.stats.leaf_worlds += alive.len() as u64;
                    let (games, weights) = if self.cfg.maple_dedupe { self.dedupe(alive) } else {
                        let n = alive.len();
                        (alive, vec![1.0; n])
                    };
                    self.stats.unique_leaves += games.len() as u64;
                    self.pending.push(PendingSim { path, leaf_node: node, weights, term_sum, term_n });
                    leaves.extend(games);
                    done += 1;
                    break;
                }
                if self.collides(node) {
                    collided = true;
                    break;
                }
                let keys = match self.cfg.maple_select {
                    MapleSelect::RefWorld => keys_of(&alive[0]),
                    MapleSelect::Union => {
                        let mut u: Vec<u64> = Vec::new();
                        for w in &alive {
                            for c in w.choices() {
                                if !u.contains(&c.key) {
                                    u.push(c.key);
                                }
                            }
                        }
                        u
                    }
                };
                let e = self.select(node, &keys);
                let key = self.nodes[node as usize].edges[e].key;
                path.push((node, e, actor));
                let before = alive.len();
                alive = alive
                    .into_iter()
                    .filter_map(|mut w| {
                        let idx = w.choices().iter().position(|c| c.key == key)?;
                        w.apply(idx);
                        Some(w)
                    })
                    .collect();
                self.stats.dropped_illegal += (before - alive.len()) as u64;
                via = Some((node, e));
            }
        }
        leaves
    }

    /// Merge worlds the evaluator can't tell apart.
    fn dedupe(&self, worlds: Vec<Game>) -> (Vec<Game>, Vec<f32>) {
        if worlds.len() <= 1 {
            let n = worlds.len();
            return (worlds, vec![1.0; n]);
        }
        let mut games: Vec<Game> = Vec::with_capacity(worlds.len());
        let mut hashes: Vec<u64> = Vec::with_capacity(worlds.len());
        let mut weights: Vec<f32> = Vec::with_capacity(worlds.len());
        for w in worlds {
            let mut h = FxHasher::default();
            observe_with(&w, w.to_act(), self.cfg.perfect_obs).hash(&mut h);
            choice_fields(&w).hash(&mut h);
            keys_of(&w).hash(&mut h);
            let h = h.finish();
            match hashes.iter().position(|&x| x == h) {
                Some(i) => weights[i] += 1.0,
                None => {
                    hashes.push(h);
                    weights.push(1.0);
                    games.push(w);
                }
            }
        }
        (games, weights)
    }

    /// Back up evaluations for the leaves returned by the last [`Self::gather`].
    pub fn feed(&mut self, leaves: &[Game], evals: Vec<Eval>) {
        let pending = std::mem::take(&mut self.pending);
        assert_eq!(leaves.len(), evals.len());
        assert_eq!(pending.iter().map(|p| p.weights.len()).sum::<usize>(), evals.len());
        let mut at = 0;
        for sim in pending {
            let n = sim.weights.len();
            let (ls, es) = (&leaves[at..at + n], &evals[at..at + n]);
            at += n;
            self.remove_virtual_loss(&sim.path);
            let keys: Vec<Vec<u64>> = ls.iter().map(keys_of).collect();
            let per_world: Vec<(&[u64], &[f32], f32)> =
                keys.iter().zip(es).zip(&sim.weights).map(|((k, e), &w)| (k.as_slice(), e.priors.as_slice(), w)).collect();
            let priors = aggregate_priors(&per_world);
            let node = &mut self.nodes[sim.leaf_node as usize];
            let mean = 1.0 / priors.len().max(1) as f32;
            let mut stale = 0;
            for e in node.edges.iter_mut().filter(|e| !priors.iter().any(|p| p.0 == e.key)) {
                e.prior = mean;
                stale += 1;
            }
            for (key, p) in priors {
                match node.edges.iter_mut().find(|e| e.key == key) {
                    Some(e) => e.prior = p,
                    None => node.edges.push(Edge { key, prior: p, visits: 0.0, value_sum: 0.0, avail: 0.0, child: [NONE; 2] }),
                }
            }
            self.stats.stale_priors += stale;
            let (mut num, mut den) = (sim.term_sum, sim.term_n);
            for ((l, e), &w) in ls.iter().zip(es).zip(&sim.weights) {
                let v = if l.to_act() == self.observer { e.value } else { -e.value };
                num += w * v;
                den += w;
            }
            if sim.path.is_empty() {
                let r = sim.leaf_node as usize;
                if self.root_evals.len() <= r {
                    self.root_evals.resize(r + 1, None);
                }
                self.root_evals[r] = Some(num / den.max(1e-6));
            }
            self.backup_observer(&sim.path, num / den.max(1e-6));
            self.simulations += 1;
            self.stats.sims += 1;
        }
        if self.cfg.root_noise > 0.0 && self.simulations > 0 && self.roots().all(|r| !self.nodes[r].edges.is_empty()) {
            self.apply_root_noise_once();
        }
    }

    /// Run `sims` simulations with `eval`, in batches of `batch`.
    pub fn run(&mut self, sims: u32, batch: usize, eval: &mut dyn Evaluator) {
        if self.root.is_over() || self.root.choices().len() <= 1 {
            return;
        }
        let target = self.simulations + sims;
        while self.simulations < target {
            let want = batch.min((target - self.simulations) as usize).max(1);
            let leaves = self.gather(want);
            if !leaves.is_empty() {
                let evals = eval.eval(&leaves);
                self.feed(&leaves, evals);
            }
        }
    }

    /// Root node ids: one per PIMC world, else just node 0.
    fn roots(&self) -> std::ops::Range<usize> {
        0..if self.cfg.pimc_worlds > 0 { self.worlds.len().max(1) } else { 1 }
    }

    /// (visits, value sum, prior) per root choice, from root node `root`.
    fn root_edges(&self, root: usize) -> Vec<(f32, f32, f32)> {
        let edges = &self.nodes[root].edges;
        self.root
            .choices()
            .iter()
            .map(|c| edges.iter().find(|e| e.key == c.key).map_or((0.0, 0.0, 0.0), |e| (e.visits, e.value_sum, e.prior)))
            .collect()
    }

    /// Visits per root choice; with PIMC, the mean per-world distribution scaled to the total.
    pub fn root_visits(&self) -> Vec<f32> {
        if self.cfg.pimc_worlds == 0 {
            return self.root_edges(0).iter().map(|e| e.0).collect();
        }
        let mut avg = vec![0.0f32; self.root.choices().len()];
        let (mut total, mut worlds) = (0.0f32, 0.0f32);
        for r in self.roots() {
            let visits: Vec<f32> = self.root_edges(r).iter().map(|e| e.0).collect();
            let sum: f32 = visits.iter().sum();
            if sum > 0.0 {
                for (a, v) in avg.iter_mut().zip(&visits) {
                    *a += v / sum;
                }
                total += sum;
                worlds += 1.0;
            }
        }
        if worlds > 0.0 {
            avg.iter_mut().for_each(|a| *a *= total / worlds);
        }
        avg
    }

    /// The completed-Q policy target per root choice (Danihelka et al. 2022,
    /// section 4 and appendix D): softmax of the prior logits plus
    /// sigma(q), unvisited choices completed with v_mix, Q min-max
    /// normalized over the root's choices. PIMC averages the worlds' targets.
    pub fn root_policy_target(&self) -> Vec<f32> {
        const C_VISIT: f32 = 50.0;
        const C_SCALE: f32 = 1.0;
        let n = self.root.choices().len();
        let mut avg = vec![0.0f32; n];
        let mut worlds = 0.0f32;
        for r in self.roots() {
            let edges = self.root_edges(r);
            let total: f32 = edges.iter().map(|e| e.0).sum();
            if total <= 0.0 {
                continue;
            }
            let clean = self.clean_priors.get(r);
            let prior = |i: usize| {
                let key = self.root.choices()[i].key;
                clean.and_then(|c| c.iter().find(|p| p.0 == key)).map_or(edges[i].2, |p| p.1)
            };
            let (mut seen_prior, mut seen_pq) = (0.0f32, 0.0f32);
            for (i, e) in edges.iter().enumerate().filter(|(_, e)| e.0 > 0.0) {
                seen_prior += prior(i);
                seen_pq += prior(i) * e.1 / e.0;
            }
            let v = self.root_evals.get(r).copied().flatten().unwrap_or(0.0);
            let v_mix = if seen_prior > 0.0 { (v + total / seen_prior * seen_pq) / (1.0 + total) } else { v };
            let q: Vec<f32> = edges.iter().map(|e| if e.0 > 0.0 { e.1 / e.0 } else { v_mix }).collect();
            let (lo, hi) = q.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &x| (lo.min(x), hi.max(x)));
            let max_n = edges.iter().map(|e| e.0).fold(0.0f32, f32::max);
            let logits: Vec<f32> = (0..n)
                .map(|i| {
                    let qn = if hi > lo { (q[i] - lo) / (hi - lo) } else { 0.0 };
                    prior(i).max(1e-12).ln() + (C_VISIT + max_n) * C_SCALE * qn
                })
                .collect();
            let top = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp: Vec<f32> = logits.iter().map(|l| (l - top).exp()).collect();
            let sum: f32 = exp.iter().sum();
            for (a, e) in avg.iter_mut().zip(exp) {
                *a += e / sum;
            }
            worlds += 1.0;
        }
        if worlds > 0.0 {
            avg.iter_mut().for_each(|a| *a /= worlds);
        }
        avg
    }

    /// Most visited root choice (ties: higher mean value, then prior).
    pub fn best(&self) -> usize {
        let visits = self.root_visits();
        let per_root: Vec<Vec<(f32, f32, f32)>> = self.roots().map(|r| self.root_edges(r)).collect();
        let score = |i: usize| {
            let (n, w, p) = per_root.iter().fold((0.0, 0.0, 0.0), |(n, w, p), r| (n + r[i].0, w + r[i].1, p + r[i].2));
            (visits[i], if n > 0.0 { w / n } else { -2.0 }, p)
        };
        (0..self.root.choices().len())
            .max_by(|&a, &b| score(a).partial_cmp(&score(b)).unwrap())
            .unwrap_or(0)
    }

    /// Mean root value for the searching player (PIMC: mean over worlds).
    pub fn root_value(&self) -> f32 {
        let (mut sum, mut worlds) = (0.0, 0.0);
        for r in self.roots() {
            let (n, w) = self.nodes[r].edges.iter().fold((0.0, 0.0), |(n, w), e| (n + e.visits, w + e.value_sum));
            if n > 0.0 {
                sum += w / n;
                worlds += 1.0;
            }
        }
        if worlds > 0.0 { sum / worlds } else { 0.0 }
    }

    /// The child through edge `e` where `actor` is to act, created on first use.
    fn child(&mut self, node: u32, e: usize, actor: PlayerId) -> u32 {
        let c = self.nodes[node as usize].edges[e].child[actor.index()];
        if c != NONE {
            debug_assert_eq!(self.nodes[c as usize].actor, actor);
            return c;
        }
        self.nodes.push(Node::new(actor));
        let c = (self.nodes.len() - 1) as u32;
        self.nodes[node as usize].edges[e].child[actor.index()] = c;
        c
    }

    /// PUCT over the edges for `keys` (the choices legal in this simulation).
    fn select(&mut self, node: u32, keys: &[u64]) -> usize {
        let cfg = self.cfg.clone();
        let n = &mut self.nodes[node as usize];
        // Keys new to this node get its mean prior.
        let mean_prior = if n.edges.is_empty() {
            // Only a node awaiting its evaluation has no edges here.
            debug_assert!(!cfg.collisions, "select at a node awaiting its evaluation");
            self.stats.blind_selects += 1;
            1.0
        } else {
            n.edges.iter().map(|e| e.prior).sum::<f32>() / n.edges.len() as f32
        };
        let mut legal: Vec<usize> = Vec::with_capacity(keys.len());
        for &key in keys {
            let i = match n.edges.iter().position(|e| e.key == key) {
                Some(i) => i,
                None => {
                    n.edges.push(Edge { key, prior: mean_prior, visits: 0.0, value_sum: 0.0, avail: 0.0, child: [NONE; 2] });
                    n.edges.len() - 1
                }
            };
            n.edges[i].avail += 1.0;
            if cfg.collisions {
                self.avail_log.push((node, i as u32));
            }
            legal.push(i);
        }
        let (vis, sum) = legal.iter().fold((0.0f32, 0.0f32), |(v, s), &i| (v + n.edges[i].visits, s + n.edges[i].value_sum));
        let parent_q = if vis > 0.0 { sum / vis } else { 0.0 };
        let fpu = parent_q - cfg.fpu_reduction;
        let mut best = legal[0];
        let mut best_u = f32::NEG_INFINITY;
        for &i in &legal {
            let e = &n.edges[i];
            let q = if e.visits > 0.0 { e.value_sum / e.visits } else { fpu };
            let u = q + cfg.c_puct * e.prior * e.avail.sqrt() / (1.0 + e.visits);
            if u > best_u {
                best_u = u;
                best = i;
            }
        }
        best
    }

    /// Back up a value given from the observer's point of view.
    fn backup_observer(&mut self, path: &[(u32, usize, PlayerId)], v: f32) {
        let observer = self.observer;
        for &(node, e, chooser) in path {
            let edge = &mut self.nodes[node as usize].edges[e];
            edge.visits += 1.0;
            edge.value_sum += if chooser == observer { v } else { -v };
        }
    }

    fn add_virtual_loss(&mut self, path: &[(u32, usize, PlayerId)]) {
        let vl = self.cfg.virtual_loss;
        for &(node, e, _) in path {
            let edge = &mut self.nodes[node as usize].edges[e];
            edge.visits += vl;
            edge.value_sum -= vl;
        }
    }

    fn remove_virtual_loss(&mut self, path: &[(u32, usize, PlayerId)]) {
        let vl = self.cfg.virtual_loss;
        for &(node, e, _) in path {
            let edge = &mut self.nodes[node as usize].edges[e];
            edge.visits -= vl;
            edge.value_sum += vl;
        }
    }

    fn apply_root_noise_once(&mut self) {
        let alpha = self.cfg.dirichlet_alpha;
        let eps = self.cfg.root_noise;
        self.cfg.root_noise = 0.0;
        self.clean_priors = self.roots().map(|r| self.nodes[r].edges.iter().map(|e| (e.key, e.prior)).collect()).collect();
        for r in self.roots() {
            let edges = &mut self.nodes[r].edges;
            let noise: Vec<f32> = (0..edges.len()).map(|_| gamma_sample(&mut self.rng, alpha)).collect();
            let total: f32 = noise.iter().sum::<f32>().max(1e-9);
            for (e, n) in edges.iter_mut().zip(noise) {
                e.prior = (1.0 - eps) * e.prior + eps * n / total;
            }
        }
    }
}

fn uniform01(rng: &mut SplitMix64) -> f32 {
    ((rng.next_u64() >> 40) as f32 + 0.5) / (1u64 << 24) as f32
}

fn gumbel(rng: &mut SplitMix64) -> f32 {
    -(-uniform01(rng).ln()).ln()
}

/// Marsaglia-Tsang gamma sampler (alpha < 1 via the boost trick).
fn gamma_sample(rng: &mut SplitMix64, alpha: f32) -> f32 {
    if alpha < 1.0 {
        return gamma_sample(rng, alpha + 1.0) * uniform01(rng).powf(1.0 / alpha);
    }
    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        // Box-Muller normal.
        let (u1, u2) = (uniform01(rng), uniform01(rng));
        let x = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        let v = (1.0 + c * x).powi(3);
        if v <= 0.0 {
            continue;
        }
        let u = uniform01(rng);
        if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
            return d * v;
        }
    }
}

/// Uniform priors and a hand-written board evaluation.
pub struct HeuristicEval;

pub fn heuristic_value(g: &Game, pov: PlayerId) -> f32 {
    use mtg_kernel::card_def::CardType;
    use mtg_kernel::engine::{effective_power, effective_toughness, object_has_type};
    if let Some(o) = g.outcome() {
        return terminal_value(o, pov);
    }
    let st = &g.state;
    let side = |p: PlayerId| -> f32 {
        let ps = &st.players[p.index()];
        let mut life = ps.life as f32;
        let mut s = ps.hand.len() as f32;
        if let Some(c) = &st.commander {
            // Commander damage is a second life total; a commander at home is like a card in hand.
            let taken = c.damage[p.index()].iter().copied().max().unwrap_or(0);
            life = life.min(2.0 * (mtg_kernel::state::COMMANDER_DAMAGE_TO_LOSE - taken) as f32);
            s += (st.objects.get(c.commanders[p.index()]).zone == mtg_kernel::state::Zone::Command) as i32 as f32;
        }
        s += life * 0.5;
        for &id in &ps.battlefield {
            let def = &mtg_kernel::card_def::CARD_DEFS[st.objects.get(id).card_def as usize];
            if object_has_type(st, id, CardType::Creature) {
                s += 1.0 + effective_power(st, id).max(0) as f32 + 0.5 * effective_toughness(st, id).max(0) as f32;
            } else if def.is_land {
                s += 0.7;
            } else {
                s += 1.0;
            }
        }
        s
    };
    ((side(pov) - side(pov.opponent())) / 6.0).tanh()
}

impl Evaluator for HeuristicEval {
    fn eval(&mut self, leaves: &[Game]) -> Vec<Eval> {
        leaves
            .iter()
            .map(|g| Eval { priors: vec![1.0; g.choices().len()], value: heuristic_value(g, g.to_act()) })
            .collect()
    }
}

/// Uniform priors; value from random playouts scored by [`heuristic_value`].
pub struct RolloutEval {
    pub rng: SplitMix64,
    pub rollouts: u32,
    pub max_decisions: u32,
}

impl Evaluator for RolloutEval {
    fn eval(&mut self, leaves: &[Game]) -> Vec<Eval> {
        leaves
            .iter()
            .map(|g| {
                let pov = g.to_act();
                let mut total = 0.0;
                for _ in 0..self.rollouts {
                    let mut r = g.clone();
                    let start = r.decisions;
                    while !r.is_over() && r.decisions - start < self.max_decisions {
                        let n = r.choices().len();
                        r.apply((self.rng.next_u64() % n as u64) as usize);
                    }
                    total += heuristic_value(&r, pov);
                }
                Eval { priors: vec![1.0; g.choices().len()], value: total / self.rollouts as f32 }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn maple(k: u32, sel: MapleSelect) -> Config {
        Config { maple_worlds: k, maple_select: sel, ..Config::default() }
    }

    fn pimc(k: u32) -> Config {
        Config { pimc_worlds: k, ..Config::default() }
    }

    fn collide(c: Config) -> Config {
        Config { collisions: true, ..c }
    }

    fn configs() -> Vec<Config> {
        let base = vec![
            Config::default(),
            maple(5, MapleSelect::RefWorld),
            maple(5, MapleSelect::Union),
            Config { maple_resample: true, ..maple(3, MapleSelect::RefWorld) },
            pimc(5),
        ];
        base.iter().cloned().chain(base.iter().cloned().map(collide)).collect()
    }

    /// Prior 0.9 on one root choice, the rest shared.
    struct Favour(u64);

    impl Evaluator for Favour {
        fn eval(&mut self, leaves: &[Game]) -> Vec<Eval> {
            leaves
                .iter()
                .map(|g| {
                    let n = g.choices().len() as f32;
                    Eval { priors: g.choices().iter().map(|c| if c.key == self.0 { 0.9 } else { 0.1 / n }).collect(), value: 0.0 }
                })
                .collect()
        }
    }

    /// Mid-game positions with at least `choices` choices.
    fn positions(n: usize, choices: usize) -> Vec<Game> {
        let mut rng = SplitMix64::seed(17);
        let mut out = Vec::new();
        for seed in 0..40 {
            let mut g = Game::new("Faeries", "Spy", seed).unwrap();
            while !g.is_over() && out.len() < n {
                if g.state.turn >= 3 && g.choices().len() >= choices && rng.next_u64() % 6 == 0 {
                    out.push(g.clone());
                }
                let k = g.choices().len();
                g.apply((rng.next_u64() % k as u64) as usize);
            }
        }
        assert_eq!(out.len(), n);
        out
    }

    #[test]
    fn collisions_wait_for_the_priors() {
        for g in positions(10, 3) {
            let key = g.choices()[2].key;
            for base in [Config::default(), maple(5, MapleSelect::Union)] {
                // A 2-simulation search in one batch: without collisions the second
                // simulation chooses before the root's priors arrive, so it takes choice 0.
                for (cfg, want) in [(base.clone(), 0), (collide(base.clone()), 2)] {
                    let mut s = Search::new(&g, cfg.clone(), 1);
                    s.run(2, 2, &mut Favour(key));
                    let v = s.root_visits();
                    assert_eq!(v.iter().sum::<f32>(), 1.0, "{cfg:?}: {v:?}");
                    assert_eq!(v[want], 1.0, "{cfg:?}: {v:?}");
                    assert_eq!(s.stats.collisions, u64::from(cfg.collisions));
                }
            }
        }
    }

    #[test]
    fn collisions_never_select_without_priors() {
        let mut rng = SplitMix64::seed(8);
        let (mut blind, mut collisions) = (0, 0);
        for g in positions(20, 2) {
            for base in configs().into_iter().take(5) {
                let mut on = Search::new(&g, collide(base.clone()), rng.next_u64());
                on.run(64, 8, &mut HeuristicEval);
                assert_eq!(on.stats.blind_selects, 0, "{base:?}");
                assert_eq!(on.stats.sims, on.simulations as u64);
                collisions += on.stats.collisions;
                let mut off = Search::new(&g, base, rng.next_u64());
                off.run(64, 8, &mut HeuristicEval);
                blind += off.stats.blind_selects;
            }
        }
        assert!(collisions > 0 && blind > 0, "{collisions} collisions, {blind} blind selections");
    }

    /// Every edge that has been offered or visited: (key, visits, value sum, avail, children), per node.
    fn tree(s: &Search) -> Vec<Vec<(u64, f32, f32, f32, [u32; 2])>> {
        s.nodes
            .iter()
            .map(|n| {
                n.edges.iter().filter(|e| e.avail != 0.0 || e.visits != 0.0).map(|e| (e.key, e.visits, e.value_sum, e.avail, e.child)).collect()
            })
            .collect()
    }

    #[test]
    fn a_collision_leaves_no_trace() {
        let (mut checked, mut deep) = (0, 0);
        for g in positions(12, 2) {
            for cfg in [collide(Config::default()), collide(maple(5, MapleSelect::Union)), collide(pimc(3))] {
                let mut a = Search::new(&g, cfg.clone(), 4);
                let mut b = Search::new(&g, cfg.clone(), 4);
                a.run(12, 4, &mut HeuristicEval);
                b.run(12, 4, &mut HeuristicEval);
                let before = b.simulations;
                let leaves = b.gather(32);
                if b.stats.collisions == a.stats.collisions {
                    continue;
                }
                // The same simulations without the colliding one.
                let sims = if cfg.maple_worlds > 0 { (b.simulations - before) as usize + b.pending.len() } else { leaves.len() };
                a.gather(sims);
                if a.simulations != b.simulations || a.pending.len() != b.pending.len() {
                    continue; // b also ended games after its last leaf
                }
                assert_eq!(tree(&a), tree(&b), "{cfg:?}");
                assert_eq!(a.next_world, b.next_world);
                checked += 1;
                deep += usize::from(!b.avail_log.is_empty());
            }
        }
        assert!(checked >= 24 && deep >= 12, "{checked} collisions compared, {deep} below the root");
    }

    #[test]
    fn gathering_again_before_feeding_collides_at_once() {
        for g in positions(4, 2) {
            for cfg in [collide(Config::default()), collide(maple(5, MapleSelect::Union)), collide(pimc(3))] {
                let mut s = Search::new(&g, cfg.clone(), 5);
                let first = s.gather(8);
                assert!(!first.is_empty());
                let pending = s.pending.len();
                for _ in 0..3 {
                    assert!(s.gather(8).is_empty(), "{cfg:?}");
                }
                assert_eq!(s.pending.len(), pending);
                let evals = HeuristicEval.eval(&first);
                s.feed(&first, evals);
                assert_eq!(s.simulations as usize, pending);
            }
        }
    }

    #[test]
    fn search_visits_sum_to_simulations() {
        let g = Game::new("Burn", "Rally", 1).unwrap();
        for cfg in configs() {
            let mut s = Search::new(&g, cfg.clone(), 7);
            s.run(200, 8, &mut HeuristicEval);
            let total: f32 = s.root_visits().iter().sum();
            assert!(s.simulations >= 200);
            // Each root's first simulation expands it without visiting an edge.
            let roots = s.roots().len() as f32;
            assert!((total - s.simulations as f32).abs() <= roots + 1e-3, "{cfg:?}: {total} vs {}", s.simulations);
            assert!(s.best() < g.choices().len());
            assert_eq!(s.stats.sims, s.simulations as u64);
        }
    }

    #[test]
    fn search_never_changes_root() {
        let g = Game::new("Faeries", "Elves", 2).unwrap();
        let h = g.state.state_hash();
        for cfg in configs() {
            let mut s = Search::new(&g, cfg, 3);
            s.run(100, 4, &mut HeuristicEval);
            assert_eq!(g.state.state_hash(), h);
        }
    }

    #[test]
    fn maple_aggregates_several_worlds_per_leaf() {
        let g = Game::new("Faeries", "Spy", 5).unwrap();
        let mut s = Search::new(&g, maple(5, MapleSelect::RefWorld), 11);
        s.run(300, 8, &mut HeuristicEval);
        assert!(s.stats.leaf_worlds > s.stats.sims, "{:?}", s.stats);
        assert!(s.stats.unique_leaves <= s.stats.leaf_worlds);
    }

    /// Nodes reachable from `root`, `root` included.
    fn subtree(s: &Search, root: u32) -> Vec<u32> {
        let mut out = vec![root];
        let mut i = 0;
        while i < out.len() {
            for e in &s.nodes[out[i] as usize].edges {
                out.extend(e.child.iter().filter(|&&c| c != NONE));
            }
            i += 1;
        }
        out
    }

    #[test]
    fn pimc_searches_each_world_in_its_own_tree() {
        let g = Game::new("Faeries", "Spy", 5).unwrap();
        for cfg in [pimc(5), collide(pimc(5))] {
            let mut s = Search::new(&g, cfg, 11);
            s.run(200, 8, &mut HeuristicEval);
            assert_eq!(s.worlds.len(), 5);
            assert_eq!(s.simulations, 200);
            let mut seen = std::collections::HashSet::new();
            for r in 0..5u32 {
                let visits: f32 = s.nodes[r as usize].edges.iter().map(|e| e.visits).sum();
                // The first simulation expands the root and visits no edge.
                assert_eq!(visits, 39.0, "world {r}");
                for n in subtree(&s, r) {
                    assert!(seen.insert(n), "node {n} is shared between worlds");
                }
            }
            // The move comes from the average of the per-world distributions.
            let mut avg = vec![0.0f32; g.choices().len()];
            for r in 0..5 {
                for (a, e) in avg.iter_mut().zip(s.root_edges(r)) {
                    *a += e.0 / 39.0 / 5.0;
                }
            }
            for (v, a) in s.root_visits().iter().zip(&avg) {
                assert!((v - a * 195.0).abs() < 1e-3, "{v} vs {}", a * 195.0);
            }
        }
    }

    /// A position where one choice leads to either player's decision, depending on hidden cards.
    fn diverging_position() -> Option<(Game, u64)> {
        let mut rng = SplitMix64::seed(9);
        for seed in 0..40 {
            let mut g = Game::new("Elves", "Faeries", seed).unwrap();
            while !g.is_over() && g.decisions < 400 {
                let me = g.to_act();
                for c in g.choices() {
                    let mut actors = [false; 2];
                    for _ in 0..12 {
                        let mut d = g.clone();
                        d.determinize(me, rng.next_u64());
                        if d.to_act() != me {
                            continue;
                        }
                        if let Some(i) = d.choices().iter().position(|x| x.key == c.key) {
                            d.apply(i);
                            if !d.is_over() {
                                actors[d.to_act().index()] = true;
                            }
                        }
                    }
                    if actors == [true, true] {
                        return Some((g.clone(), c.key));
                    }
                }
                let n = g.choices().len();
                g.apply((rng.next_u64() % n as u64) as usize);
            }
        }
        None
    }

    #[test]
    fn edges_made_while_a_node_awaits_evaluation_get_real_priors() {
        let mut rng = SplitMix64::seed(5);
        let mut g = Game::new("Terror", "Burn", 0).unwrap();
        let resample = Config { maple_resample: true, ..maple(3, MapleSelect::RefWorld) };
        let mut stale = 0;
        for _ in 0..40 {
            if g.is_over() {
                break;
            }
            for (cfg, bound) in [(Config::default(), 0.5), (pimc(5), 0.5), (resample.clone(), 0.99)] {
                let mut s = Search::new(&g, cfg.clone(), rng.next_u64());
                s.run(64, 8, &mut HeuristicEval);
                let worst = s.nodes.iter().flat_map(|n| n.edges.iter()).map(|e| e.prior).fold(0.0f32, f32::max);
                assert!(worst <= bound + 1e-6, "{cfg:?}: an edge kept prior {worst} at decision {}", g.decisions);
                stale += s.stats.stale_priors;
            }
            let n = g.choices().len();
            g.apply((rng.next_u64() % n as u64) as usize);
        }
        assert!(stale > 0, "no search passed through a node awaiting evaluation");
    }

    /// Positions paired with a copy that swaps a hidden hand card with an unseen library card.
    fn hidden_swaps(n: usize) -> Vec<(Game, Game)> {
        use crate::game::{set_card, unseen_slots};
        let mut rng = SplitMix64::seed(21);
        let mut out = Vec::new();
        for seed in 0..20 {
            let mut g = Game::new("Faeries", "Terror", seed).unwrap();
            while !g.is_over() && out.len() < n {
                let (me, st) = (g.to_act(), &g.state);
                let hand = unseen_slots(st, me, me.opponent(), true);
                let all = unseen_slots(st, me, me.opponent(), false);
                let def = |id| st.objects.get(id).card_def;
                let pair = hand.iter().find_map(|&h| all.iter().find(|&&l| !hand.contains(&l) && def(l) != def(h)).map(|&l| (h, l)));
                if let Some((h, l)) = pair {
                    let mut swapped = g.clone();
                    let (dh, dl) = (def(h), def(l));
                    set_card(&mut swapped.state, h, dl);
                    set_card(&mut swapped.state, l, dh);
                    out.push((g.clone(), swapped));
                }
                let k = g.choices().len();
                g.apply((rng.next_u64() % k as u64) as usize);
            }
        }
        out
    }

    fn fake_score(hand: &[u16]) -> f32 {
        hand.iter().map(|&d| (d as u32).wrapping_mul(2654435761) % 1000).sum::<u32>() as f32
    }

    fn score_worlds(s: &mut Search) {
        if let Some(cands) = s.world_request() {
            let d: Vec<f32> = cands.iter().map(|h| fake_score(h)).collect();
            s.set_world_scores(&d);
        }
    }

    fn test_retention() -> Arc<Vec<f32>> {
        let n = RETENTION_BUCKETS * CARD_DEFS.len();
        Arc::new((0..n).map(|i| ((i as u32).wrapping_mul(7919) % 13) as f32 * 0.3).collect())
    }

    #[test]
    fn search_ignores_where_hidden_cards_sit() {
        let scored = |c: Config| Config { world_sampler: WorldSampler::Scored, ..c };
        let retention = |c: Config| Config { world_sampler: WorldSampler::Retention, retention: Some(test_retention()), ..c };
        let pairs = hidden_swaps(12);
        assert_eq!(pairs.len(), 12, "positions with a hidden card to swap");
        for (g, swapped) in &pairs {
            for cfg in [
                Config::default(),
                maple(5, MapleSelect::RefWorld),
                pimc(5),
                scored(Config::default()),
                scored(maple(5, MapleSelect::RefWorld)),
                retention(Config::default()),
                retention(maple(5, MapleSelect::RefWorld)),
                collide(Config::default()),
                collide(maple(5, MapleSelect::Union)),
            ] {
                let mut a = Search::new(g, cfg.clone(), 9);
                let mut b = Search::new(swapped, cfg.clone(), 9);
                score_worlds(&mut a);
                score_worlds(&mut b);
                a.run(48, 8, &mut HeuristicEval);
                b.run(48, 8, &mut HeuristicEval);
                assert_eq!(a.root_visits(), b.root_visits(), "{cfg:?}");
            }
        }
    }

    /// Lets `holder` play the exiled `id` until its next turn, as the engine
    /// grants it from the card's true identity.
    fn permit(state: &mut mtg_kernel::state::GameState, id: mtg_kernel::ids::ObjectId, holder: PlayerId) {
        use mtg_kernel::engine::{PlayOrCast, PlayPermission, PlayPermissionExpiry};
        let land = CARD_DEFS[state.objects.get(id).card_def as usize].is_playable_land();
        state.engine.exile_play_permissions.retain(|p| p.object != id);
        state.engine.exile_play_permissions.push(PlayPermission {
            object: id,
            holder,
            zone_change_generation: state.objects.get(id).zone_change_count,
            play_or_cast: if land { PlayOrCast::Play } else { PlayOrCast::Cast },
            expiry: PlayPermissionExpiry::UntilHoldersNextTurn { holder_turn_started: false },
        });
    }

    /// Whether the player to act can see its own library (it is searching it).
    fn sees_own_library(g: &Game) -> bool {
        let me = g.to_act();
        (0..4).any(|s| {
            let mut w = g.clone();
            w.determinize(me, s);
            w.is_over() || w.to_act() != me
        })
    }

    /// Positions paired with a copy that differs only in what the player to act
    /// can't know: the order of its own unseen library, or every future shuffle.
    fn unseen_twins(n: usize) -> Vec<(Game, Game, &'static str)> {
        use crate::game::{set_card, unseen_slots};
        let decks = [("Wildfire", "Spy"), ("Spy", "CawGates"), ("Faeries", "Wildfire"), ("Elves", "Spy")];
        let mut rng = SplitMix64::seed(5);
        let mut out = Vec::new();
        for seed in 0..40u64 {
            let (d0, d1) = decks[seed as usize % decks.len()];
            let mut g = Game::new(d0, d1, seed).unwrap();
            while !g.is_over() && out.len() < n {
                let me = g.to_act();
                let own = unseen_slots(&g.state, me, me, false);
                let def = |id: mtg_kernel::ids::ObjectId| g.state.objects.get(id).card_def;
                let swap = own.iter().find_map(|&a| own.iter().find(|&&b| def(b) != def(a)).map(|&b| (a, b)));
                if let (Some((a, b)), false) = (swap, sees_own_library(&g)) {
                    let mut t = g.clone();
                    let (da, db) = (def(a), def(b));
                    set_card(&mut t.state, a, db);
                    set_card(&mut t.state, b, da);
                    out.push((g.clone(), t, "own library"));
                }
                if rng.next_u64() % 3 == 0 {
                    let mut t = g.clone();
                    t.state.reseed_shuffles(rng.next_u64());
                    out.push((g.clone(), t, "shuffles"));
                }
                if rng.next_u64() % 4 == 0 {
                    // A card face down in exile that the player to act can't see.
                    let mut base = g.clone();
                    let owner = if rng.next_u64() % 2 == 0 { me } else { me.opponent() };
                    let owner_knows = owner != me && rng.next_u64() % 2 == 0;
                    let keep = owner == me && sees_own_library(&g);
                    if let Some(fd) = mtg_kernel::face_down::exile_top(&mut base.state, owner, owner_knows, None) {
                        // A card its owner may look at can also be playable (Headliner Scarlett).
                        if owner_knows {
                            permit(&mut base.state, fd, owner);
                        }
                        let d = |id: mtg_kernel::ids::ObjectId| base.state.objects.get(id).card_def;
                        let land = |id| CARD_DEFS[d(id) as usize].is_playable_land();
                        let slots = unseen_slots(&base.state, me, owner, keep);
                        let other = slots
                            .iter()
                            .copied()
                            .find(|&b| d(b) != d(fd) && land(b) != land(fd))
                            .or_else(|| slots.iter().copied().find(|&b| d(b) != d(fd)));
                        if let Some(b) = other {
                            let mut t = base.clone();
                            let (da, db) = (d(fd), d(b));
                            set_card(&mut t.state, fd, db);
                            set_card(&mut t.state, b, da);
                            if owner_knows {
                                permit(&mut t.state, fd, owner);
                            }
                            out.push((base, t, "face down"));
                        }
                    }
                }
                if rng.next_u64() % 4 == 0 {
                    // Cards the player to act knows, on a library bottom in an unknown order.
                    let owner = if rng.next_u64() % 2 == 0 { me } else { me.opponent() };
                    let mut base = g.clone();
                    let d = |id: mtg_kernel::ids::ObjectId| g.state.objects.get(id).card_def;
                    let top: Vec<_> = g.state.players[owner.index()].library.iter().copied().take(3).collect();
                    let differ = top.len() == 3 && d(top[0]) != d(top[1]);
                    if differ && !(owner == me && sees_own_library(&g)) {
                        base.state.move_library_cards_to_bottom_in_random_order(owner, &top, [true, true]).unwrap();
                        let mut t = base.clone();
                        let l = &mut t.state.players[owner.index()].library;
                        let (i, j) = (l.iter().position(|&x| x == top[0]).unwrap(), l.iter().position(|&x| x == top[1]).unwrap());
                        l.swap(i, j);
                        out.push((base, t, "library group"));
                    }
                }
                let k = g.choices().len();
                g.apply((rng.next_u64() % k as u64) as usize);
            }
        }
        out
    }

    #[test]
    fn search_ignores_own_library_order_and_future_shuffles() {
        let scored = |c: Config| Config { world_sampler: WorldSampler::Scored, ..c };
        let twins = unseen_twins(60);
        assert_eq!(twins.len(), 60);
        for kind in ["own library", "shuffles", "face down", "library group"] {
            assert!(twins.iter().any(|t| t.2 == kind), "no {kind} twins");
        }
        for (g, twin, what) in &twins {
            for cfg in [
                Config::default(),
                maple(5, MapleSelect::RefWorld),
                maple(5, MapleSelect::Union),
                Config { maple_resample: true, ..maple(3, MapleSelect::RefWorld) },
                pimc(5),
                scored(maple(5, MapleSelect::RefWorld)),
            ] {
                let mut a = Search::new(g, cfg.clone(), 9);
                let mut b = Search::new(twin, cfg.clone(), 9);
                score_worlds(&mut a);
                score_worlds(&mut b);
                a.run(48, 8, &mut HeuristicEval);
                b.run(48, 8, &mut HeuristicEval);
                assert_eq!(a.root_visits(), b.root_visits(), "{what}: {cfg:?}");
                assert_eq!(a.root_value(), b.root_value(), "{what}: {cfg:?}");
            }
        }
    }

    /// Ognis Commander positions paired with a copy that differs only in what
    /// the player to act can't know: its own unseen cards, the opponent's,
    /// face-down cards, the order of a known library group, future shuffles.
    fn commander_twins(per_kind: usize) -> Vec<(Game, Game, &'static str)> {
        use crate::game::{set_card, unseen_slots};
        use mtg_kernel::ids::ObjectId;
        let deck = crate::commander::CommanderDeck::named("Ognis").expect("the Ognis deck is complete");
        let mut rng = SplitMix64::seed(11);
        let mut out: Vec<(Game, Game, &'static str)> = Vec::new();
        let full = |out: &[(Game, Game, &'static str)], kind| out.iter().filter(|t| t.2 == kind).count() >= per_kind;
        for seed in 0..400u64 {
            let mut g = Game::new_commander(&deck, &deck, seed);
            while !g.is_over() {
                let (me, st) = (g.to_act(), &g.state);
                let def = |id: ObjectId| st.objects.get(id).card_def;
                let swapped = |a: ObjectId, b: ObjectId| {
                    let mut t = g.clone();
                    let (da, db) = (def(a), def(b));
                    set_card(&mut t.state, a, db);
                    set_card(&mut t.state, b, da);
                    t
                };
                let differing = |from: &[ObjectId], to: &[ObjectId]| {
                    from.iter().find_map(|&a| to.iter().find(|&&b| b != a && def(b) != def(a)).map(|&b| (a, b)))
                };
                let searching = sees_own_library(&g);
                match rng.next_u64() % 6 {
                    0 if !searching && !full(&out, "own unseen cards") => {
                        let own = unseen_slots(st, me, me, false);
                        if let Some((a, b)) = differing(&own, &own) {
                            out.push((g.clone(), swapped(a, b), "own unseen cards"));
                        }
                    }
                    1 if !full(&out, "opponent's hand") => {
                        let opp = me.opponent();
                        let (hand, all) = (unseen_slots(st, me, opp, true), unseen_slots(st, me, opp, false));
                        let rest: Vec<ObjectId> = all.iter().copied().filter(|x| !hand.contains(x)).collect();
                        if let Some((a, b)) = differing(&hand, &rest) {
                            out.push((g.clone(), swapped(a, b), "opponent's hand"));
                        }
                    }
                    2 if !full(&out, "face down") => {
                        for owner in [me, me.opponent()] {
                            let down: Vec<ObjectId> =
                                st.exile.iter().copied().filter(|&id| st.objects.get(id).owner == owner && st.face_down_hidden_from(id, me)).collect();
                            let others = unseen_slots(st, me, owner, owner == me && searching);
                            if let Some((a, b)) = differing(&down, &others) {
                                out.push((g.clone(), swapped(a, b), "face down"));
                                break;
                            }
                        }
                    }
                    3 if !full(&out, "shuffles") => {
                        let mut t = g.clone();
                        t.state.reseed_shuffles(rng.next_u64());
                        out.push((g.clone(), t, "shuffles"));
                    }
                    4 if !full(&out, "library group") => {
                        for owner in [me, me.opponent()] {
                            if owner == me && searching {
                                continue;
                            }
                            let groups: Vec<Vec<ObjectId>> =
                                st.known_library_groups(me, owner).map(|gr| gr.cards.iter().map(|l| l.object).collect()).collect();
                            if let Some((a, b)) = groups.iter().find_map(|cards| differing(cards, cards)) {
                                let mut t = g.clone();
                                let l = &mut t.state.players[owner.index()].library;
                                let (i, j) = (l.iter().position(|&x| x == a).unwrap(), l.iter().position(|&x| x == b).unwrap());
                                l.swap(i, j);
                                out.push((g.clone(), t, "library group"));
                                break;
                            }
                        }
                    }
                    _ => {}
                }
                let k = g.choices().len();
                g.apply((rng.next_u64() % k as u64) as usize);
            }
            if ["own unseen cards", "opponent's hand", "face down", "shuffles", "library group"].iter().all(|k| full(&out, k)) {
                break;
            }
        }
        out
    }

    #[test]
    fn ognis_search_ignores_what_the_player_to_act_cannot_know() {
        let scored = |c: Config| Config { world_sampler: WorldSampler::Scored, ..c };
        let twins = commander_twins(12);
        for kind in ["own unseen cards", "opponent's hand", "face down", "shuffles", "library group"] {
            assert_eq!(twins.iter().filter(|t| t.2 == kind).count(), 12, "{kind} twins");
        }
        for (g, twin, what) in &twins {
            for cfg in [Config::default(), maple(5, MapleSelect::RefWorld), pimc(5), scored(maple(5, MapleSelect::RefWorld))] {
                let mut a = Search::new(g, cfg.clone(), 9);
                let mut b = Search::new(twin, cfg.clone(), 9);
                score_worlds(&mut a);
                score_worlds(&mut b);
                a.run(48, 8, &mut HeuristicEval);
                b.run(48, 8, &mut HeuristicEval);
                assert_eq!(a.root_visits(), b.root_visits(), "{what}: {cfg:?}");
                assert_eq!(a.root_value(), b.root_value(), "{what}: {cfg:?}");
            }
        }
    }

    /// Ognis games where both players took one mulligan, paired with a copy
    /// in which the opponent of the player to act bottomed a different card:
    /// at the next declaration, and once both kept.
    fn bottom_twins() -> Vec<(Game, Game)> {
        use crate::game::{set_card, Kind};
        let deck = crate::commander::CommanderDeck::named("Ognis").expect("the Ognis deck is complete");
        let declare = |g: &mut Game, going: bool| {
            let i = g.choices().iter().position(|c| c.feat.kind == Kind::Mulligan && c.feat.arg == going as u16).unwrap();
            g.apply(i);
        };
        let mut out = Vec::new();
        for seed in 0..12u64 {
            let mut g = Game::new_commander(&deck, &deck, seed);
            let me = g.state.starting_player;
            let opp = me.opponent();
            declare(&mut g, true);
            declare(&mut g, true);
            while g.to_act() != me || g.choices()[0].feat.kind != Kind::Mulligan {
                g.apply((seed % g.choices().len() as u64) as usize);
            }
            let st = &g.state;
            let bottomed = *st.players[opp.index()].library.last().unwrap();
            let def = |id| st.objects.get(id).card_def;
            let Some(&other) = st.players[opp.index()].hand.iter().find(|&&id| def(id) != def(bottomed)) else { continue };
            let mut twin = g.clone();
            let (db, dother) = (def(bottomed), def(other));
            set_card(&mut twin.state, bottomed, dother);
            set_card(&mut twin.state, other, db);
            out.push((g.clone(), twin.clone()));
            for k in [&mut g, &mut twin] {
                declare(k, false);
                declare(k, false);
            }
            out.push((g, twin));
        }
        out
    }

    #[test]
    fn search_ignores_which_cards_the_opponent_bottomed() {
        let twins = bottom_twins();
        assert!(twins.len() >= 20, "{} twins", twins.len());
        let cards = |g: &Game| -> Vec<u16> {
            let st = &g.state;
            st.players.iter().flat_map(|p| p.hand.iter().chain(&p.library)).map(|&id| st.objects.get(id).card_def).collect()
        };
        for (g, twin) in &twins {
            let me = g.to_act();
            for seed in 0..8 {
                let (mut a, mut b) = (g.clone(), twin.clone());
                a.determinize(me, seed);
                b.determinize(me, seed);
                assert_eq!(cards(&a), cards(&b), "the same worlds are dealt");
            }
            for cfg in [Config::default(), maple(5, MapleSelect::RefWorld), pimc(5)] {
                let mut a = Search::new(g, cfg.clone(), 9);
                let mut b = Search::new(twin, cfg.clone(), 9);
                a.run(48, 8, &mut HeuristicEval);
                b.run(48, 8, &mut HeuristicEval);
                assert_eq!(a.root_visits(), b.root_visits(), "{cfg:?}");
                assert_eq!(a.root_value(), b.root_value(), "{cfg:?}");
            }
        }
    }

    #[test]
    fn completed_q_targets_follow_q_and_ignore_root_noise() {
        let mut rng = SplitMix64::seed(21);
        let mut checked = 0;
        for seed in 0..30u64 {
            let mut g = Game::new("Faeries", "Spy", seed).unwrap();
            while !g.is_over() && checked < 40 {
                if g.choices().len() >= 4 && rng.next_u64() % 5 == 0 {
                    for noise in [0.0, 0.25] {
                        for cfg in [Config { root_noise: noise, ..Config::default() }, Config { root_noise: noise, ..pimc(3) }] {
                            let mut s = Search::new(&g, cfg.clone(), seed);
                            s.run(24, 8, &mut HeuristicEval);
                            let t = s.root_policy_target();
                            assert_eq!(t.len(), g.choices().len());
                            assert!((t.iter().sum::<f32>() - 1.0).abs() < 1e-4, "{t:?}");
                            if cfg.pimc_worlds > 0 {
                                continue;
                            }
                            // Uniform priors before noise: the target ranks choices by completed Q,
                            // and every unvisited choice gets the same share.
                            let e = s.root_edges(0);
                            let q = |i: usize| if e[i].0 > 0.0 { Some(e[i].1 / e[i].0) } else { None };
                            for i in 0..t.len() {
                                for j in 0..t.len() {
                                    match (q(i), q(j)) {
                                        (Some(a), Some(b)) if a > b + 1e-6 => assert!(t[i] >= t[j], "{cfg:?}: {t:?} {e:?}"),
                                        (None, None) => assert!((t[i] - t[j]).abs() < 1e-6, "{cfg:?}: {t:?}"),
                                        _ => {}
                                    }
                                }
                            }
                        }
                    }
                    checked += 1;
                }
                let k = g.choices().len();
                g.apply((rng.next_u64() % k as u64) as usize);
            }
        }
        assert_eq!(checked, 40);
    }

    /// Ognis positions paired with a copy in which one card differs: an
    /// opponent's hand card that is Chaos Warp in one and a Mountain in the
    /// other (`known`: the player to act has seen it), or the player to act's
    /// own library top card. The swapped card trades places with an unseen
    /// card, so both copies hold the same cards.
    fn certification_twins(n: usize, known: bool) -> Vec<(Game, Game, &'static str)> {
        use crate::game::{set_card, unseen_slots};
        use mtg_kernel::card_def::card_id_by_name;
        use mtg_kernel::ids::ObjectId;
        let deck = crate::commander::CommanderDeck::named("Ognis").expect("the Ognis deck is complete");
        let (warp, mountain) = (card_id_by_name("Chaos Warp").unwrap(), card_id_by_name("Mountain").unwrap());
        let mut rng = SplitMix64::seed(23);
        let mut out: Vec<(Game, Game, &'static str)> = Vec::new();
        for seed in 0..200u64 {
            let mut g = Game::new_commander(&deck, &deck, seed);
            while !g.is_over() && out.len() < n {
                let me = g.to_act();
                let opp = me.opponent();
                let pick = rng.next_u64() % 5;
                if pick == 0 && g.state.turn >= 3 {
                    let st = &g.state;
                    let unseen_hand = unseen_slots(st, me, opp, true);
                    let hand: Vec<ObjectId> =
                        st.players[opp.index()].hand.iter().copied().filter(|id| known || unseen_hand.contains(id)).collect();
                    let rest: Vec<ObjectId> = unseen_slots(st, me, opp, false)
                        .into_iter()
                        .filter(|&id| st.objects.get(id).zone == mtg_kernel::state::Zone::Library)
                        .collect();
                    if let (Some(&h), Some(&l)) = (hand.first(), rest.first()) {
                        let mut a = g.clone();
                        if known {
                            a.state.reveal_hand_card(me, opp, h).unwrap();
                        }
                        let mut b = a.clone();
                        set_card(&mut a.state, h, warp);
                        set_card(&mut a.state, l, mountain);
                        set_card(&mut b.state, h, mountain);
                        set_card(&mut b.state, l, warp);
                        out.push((a, b, "opponent's hand: Chaos Warp or a Mountain"));
                    }
                } else if pick == 1 && !known && !sees_own_library(&g) {
                    let lib = g.state.players[me.index()].library.clone();
                    let def = |id: ObjectId| g.state.objects.get(id).card_def;
                    let own = unseen_slots(&g.state, me, me, false);
                    if let Some(&other) = lib.iter().skip(1).find(|&&id| own.contains(&id) && def(id) != def(lib[0])) {
                        if own.contains(&lib[0]) {
                            let mut b = g.clone();
                            let (top, below) = (def(lib[0]), def(other));
                            set_card(&mut b.state, lib[0], below);
                            set_card(&mut b.state, other, top);
                            out.push((g.clone(), b, "own library top card"));
                        }
                    }
                }
                let k = g.choices().len();
                g.apply((rng.next_u64() % k as u64) as usize);
            }
            if out.len() >= n {
                break;
            }
        }
        out
    }

    fn searches() -> [Config; 3] {
        [Config::default(), pimc(5), maple(5, MapleSelect::RefWorld)]
    }

    #[test]
    fn search_cannot_tell_hidden_removal_from_a_land_or_its_own_top_card() {
        let twins = certification_twins(40, false);
        for kind in ["opponent's hand: Chaos Warp or a Mountain", "own library top card"] {
            assert!(twins.iter().filter(|t| t.2 == kind).count() >= 8, "too few {kind} twins");
        }
        for (a, b, what) in &twins {
            for cfg in searches() {
                let mut x = Search::new(a, cfg.clone(), 9);
                let mut y = Search::new(b, cfg.clone(), 9);
                x.run(48, 8, &mut HeuristicEval);
                y.run(48, 8, &mut HeuristicEval);
                assert_eq!(x.root_visits(), y.root_visits(), "{what}: {cfg:?}");
                assert_eq!(x.root_value(), y.root_value(), "{what}: {cfg:?}");
            }
        }
    }

    #[test]
    fn search_sees_a_revealed_opponent_card() {
        // The positive control: the same swap on a card the searcher has seen.
        let twins = certification_twins(20, true);
        assert_eq!(twins.len(), 20);
        for cfg in searches() {
            let differ = twins
                .iter()
                .filter(|(a, b, _)| {
                    let mut x = Search::new(a, cfg.clone(), 9);
                    let mut y = Search::new(b, cfg.clone(), 9);
                    x.run(48, 8, &mut HeuristicEval);
                    y.run(48, 8, &mut HeuristicEval);
                    x.root_visits() != y.root_visits() || x.root_value() != y.root_value()
                })
                .count();
            assert!(differ >= 5, "{cfg:?}: a known card changed only {differ} of 20 searches");
        }
    }

    fn sorted_hand(g: &Game, observer: PlayerId) -> Vec<u16> {
        let mut h = g.hidden_pool(observer).0;
        h.sort_unstable();
        h
    }

    #[test]
    fn oracle_worlds_hold_the_true_hand() {
        let oracle = |c: Config| Config { world_sampler: WorldSampler::Oracle, ..c };
        for (g, _) in hidden_swaps(6) {
            let me = g.to_act();
            let truth = sorted_hand(&g, me);
            for cfg in [oracle(maple(5, MapleSelect::RefWorld)), oracle(pimc(5))] {
                let s = Search::new(&g, cfg, 4);
                assert!(!s.worlds.is_empty());
                for w in &s.worlds {
                    assert_eq!(sorted_hand(w, me), truth);
                }
            }
            let mut s = Search::new(&g, oracle(Config::default()), 4);
            for _ in 0..8 {
                let w = s.root_world().expect("a world");
                assert_eq!(sorted_hand(&w, me), truth);
            }
        }
    }

    #[test]
    fn scored_worlds_keep_the_best_hands() {
        let cfg = Config { world_sampler: WorldSampler::Scored, ..maple(5, MapleSelect::RefWorld) };
        let (mut requests, mut skips) = (0, 0);
        for (g, _) in hidden_swaps(12) {
            let me = g.to_act();
            let mut s = Search::new(&g, cfg.clone(), 6);
            match s.world_request() {
                Some(cands) => {
                    requests += 1;
                    assert!(cands.len() > 5);
                    let mut scores: Vec<f32> = cands.iter().map(|h| fake_score(h)).collect();
                    s.set_world_scores(&scores);
                    scores.sort_by(f32::total_cmp);
                    for w in &s.worlds {
                        assert!(fake_score(&sorted_hand(w, me)) <= scores[4]);
                    }
                }
                None => {
                    skips += 1;
                    assert_eq!(s.stats.world_skips, 1);
                }
            }
            assert!(!s.worlds.is_empty());
        }
        assert!(requests > 0, "no position needed scores");
        let _ = skips;
    }

    #[test]
    fn retention_hands_favour_kept_cards() {
        for (g, _) in hidden_swaps(6) {
            let me = g.to_act();
            let (hand, rest) = g.hidden_pool(me);
            let favourite = *rest.iter().chain(&hand).max().unwrap();
            let mut table = vec![-20.0f32; RETENTION_BUCKETS * CARD_DEFS.len()];
            for b in 0..RETENTION_BUCKETS {
                table[b * CARD_DEFS.len() + favourite as usize] = 20.0;
            }
            let cfg = Config { world_sampler: WorldSampler::Retention, retention: Some(Arc::new(table)), ..Config::default() };
            let mut s = Search::new(&g, cfg, 2);
            for _ in 0..8 {
                let w = s.root_world().expect("a world");
                assert!(sorted_hand(&w, me).contains(&favourite));
            }
        }
    }

    #[test]
    fn children_are_split_by_player_to_act() {
        let (g, key) = diverging_position().expect("a position where the next actor depends on hidden cards");
        let mut s = Search::new(&g, Config::default(), 1);
        s.run(3000, 8, &mut HeuristicEval);
        let e = s.nodes[0].edges.iter().find(|e| e.key == key).expect("root edge");
        assert!(e.child.iter().all(|&c| c != NONE), "both children reached: {:?}", e.child);
        for n in &s.nodes {
            for e in &n.edges {
                for (a, &c) in e.child.iter().enumerate() {
                    if c != NONE {
                        assert_eq!(s.nodes[c as usize].actor.index(), a);
                    }
                }
            }
        }
    }

    /// Determinizing mid-way through a library search halts the engine.
    #[test]
    fn searching_own_library_still_gets_visits() {
        let mut rng = SplitMix64::seed(1);
        let mut checked = 0;
        for seed in 0..60 {
            let mut g = Game::new("Wildfire", "Wildfire", seed).unwrap();
            while !g.is_over() {
                let mut d = g.clone();
                d.determinize(g.to_act(), rng.next_u64());
                if d.is_over() && g.choices().len() > 1 {
                    for cfg in [Config::default(), maple(3, MapleSelect::RefWorld), pimc(3)] {
                        let mut s = Search::new(&g, cfg, rng.next_u64());
                        s.run(16, 8, &mut HeuristicEval);
                        assert!(s.root_visits().iter().sum::<f32>() > 0.0, "no root visits at seed {seed}");
                    }
                    checked += 1;
                }
                let n = g.choices().len();
                g.apply((rng.next_u64() % n as u64) as usize);
            }
            if checked >= 5 {
                return;
            }
        }
        panic!("found only {checked} positions where determinizing halts the engine");
    }

    #[test]
    fn aggregate_priors_averages_over_worlds_where_legal() {
        // Key 3 is legal only in the first world, so it keeps that world's prior.
        let a = aggregate_priors(&[(&[1, 3], &[1.0, 1.0], 1.0), (&[1, 2], &[3.0, 1.0, 9.0], 1.0)]);
        let get = |k: u64| a.iter().find(|x| x.0 == k).unwrap().1;
        let raw = [(0.5 + 0.75) / 2.0, 0.5, 0.25];
        let total: f32 = raw.iter().sum();
        assert!((get(1) - raw[0] / total).abs() < 1e-6);
        assert!((get(3) - raw[1] / total).abs() < 1e-6);
        assert!((get(2) - raw[2] / total).abs() < 1e-6);
        assert!((a.iter().map(|x| x.1).sum::<f32>() - 1.0).abs() < 1e-6);
        // Weights count like duplicate worlds.
        let w = aggregate_priors(&[(&[1, 2], &[1.0, 0.0], 3.0), (&[1, 2], &[0.0, 1.0], 1.0)]);
        assert!((w[0].1 - 0.75).abs() < 1e-6 && (w[1].1 - 0.25).abs() < 1e-6, "{w:?}");
    }
}
