//! Single-observer information-set MCTS (SO-ISMCTS) with PUCT, and MAPLE.
//!
//! Every simulation starts from a fresh determinization of the root: cards
//! the searching player cannot see are resampled. Tree nodes are reached by
//! the sequence of semantic choice keys from the root, so the same node
//! gathers statistics across all determinizations. Because the legal choices
//! differ between determinizations, each edge counts how often it was
//! available and PUCT uses that count in place of the parent visit count
//! (the "subset-armed bandit" form of ISMCTS).
//!
//! Forced decisions are applied automatically, so one key path can reach the
//! opponent's decision in one world and ours in another (they hold an
//! instant or they don't). Each edge therefore has one child per player to
//! act, and a node only ever holds one player's choices.
//!
//! MAPLE (Multi-State Aggregated Policy Evaluation, arXiv 2605.24139), on
//! when [`Config::maple_worlds`] > 0: k worlds are sampled once per search,
//! each simulation applies its path to all of them (dropping worlds where a
//! key is illegal), and the leaf is evaluated in every surviving world.
//! Priors are averaged per key over the worlds where it is legal and values
//! are averaged, so one node's statistics reflect k hypotheses at once.
//!
//! The search is step-driven so a batched evaluator (a GPU network) can sit
//! outside it: [`Search::gather`] runs selection until it has a batch of
//! leaves, the caller evaluates them, and [`Search::feed`] backs the values
//! up. Virtual loss keeps concurrent simulations of one tree apart.

use crate::features::{choice_fields, observe_with};
use crate::game::{Game, Outcome};
use mtg_kernel::ids::PlayerId;
use mtg_kernel::state::SplitMix64;
use std::hash::{Hash, Hasher};

/// How MAPLE picks the player and the candidate keys at a node where the
/// alive worlds disagree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MapleSelect {
    /// A uniformly random alive world decides: its player acts and PUCT
    /// runs over its legal keys. Unbiased; k=1 with resampling is SO-ISMCTS.
    #[default]
    RefWorld,
    /// The paper's rule: the majority player acts and PUCT runs over the
    /// union of the alive worlds' legal keys.
    Union,
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
    /// Evaluate identical leaf observations once, weighted by their count.
    /// Exact for evaluators that only read the observation (the network);
    /// turn it off for evaluators that read hidden state (rollouts).
    pub maple_dedupe: bool,
    /// Observation encoding the evaluator uses (only affects dedupe).
    pub perfect_obs: bool,
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
}

impl std::ops::AddAssign for SearchStats {
    fn add_assign(&mut self, o: SearchStats) {
        self.sims += o.sims;
        self.leaf_worlds += o.leaf_worlds;
        self.unique_leaves += o.unique_leaves;
        self.dropped_illegal += o.dropped_illegal;
        self.dropped_diverged += o.dropped_diverged;
        self.terminal_worlds += o.terminal_worlds;
    }
}

/// Evaluation of one leaf: priors aligned with `leaf.choices()`, and a
/// value in [-1, 1] from the point of view of `leaf.to_act()`.
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
    observer: PlayerId,
    cfg: Config,
    nodes: Vec<Node>,
    rng: SplitMix64,
    pending: Vec<PendingSim>,
    /// MAPLE worlds sampled once per search (empty unless MAPLE is on and
    /// not resampling).
    worlds: Vec<Game>,
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

/// Aggregate per-world priors into one prior per key (paper eq. 2). Each
/// entry is (legal keys, priors aligned with them, weight); priors may be
/// longer than the keys (padding) and need not be normalized. A key's prior
/// is the weighted mean over the worlds where it is legal; the result is
/// normalized to sum to 1, in order of first appearance.
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
            nodes: vec![Node::new(observer)],
            rng: SplitMix64::seed(seed),
            pending: Vec::new(),
            worlds: Vec::new(),
            simulations: 0,
            stats: SearchStats::default(),
            cfg,
        };
        if s.cfg.maple_worlds > 0 && !s.cfg.maple_resample && !s.root.is_over() {
            s.worlds = s.sample_worlds();
        }
        s
    }

    /// Up to `maple_worlds` determinizations where the observer is to act
    /// (resampling hidden cards can make a root decision forced).
    fn sample_worlds(&mut self) -> Vec<Game> {
        let k = self.cfg.maple_worlds as usize;
        let mut out = Vec::with_capacity(k);
        let mut tries = 0;
        while out.len() < k && tries < k as u32 * ROOT_TRIES {
            tries += 1;
            if let Some(g) = self.determinize_root() {
                out.push(g);
            }
        }
        out
    }

    /// Run selection for up to `max_leaves` simulations. Simulations that
    /// end at a terminal state are backed up immediately; the rest return
    /// their leaf position(s) for evaluation. The returned games must be
    /// evaluated and passed to [`Self::feed`] in the same order. With MAPLE
    /// on, one simulation can return up to `maple_worlds` leaves.
    pub fn gather(&mut self, max_leaves: usize) -> Vec<Game> {
        if self.cfg.maple_worlds > 0 {
            return self.gather_maple(max_leaves);
        }
        let mut leaves = Vec::new();
        let mut attempts = 0;
        while leaves.len() < max_leaves && attempts < max_leaves * 4 {
            attempts += 1;
            let Some(mut g) = self.root_world() else {
                // No determinization keeps the observer to act: count the
                // simulation without information so the search progresses.
                self.simulations += 1;
                self.stats.sims += 1;
                continue;
            };
            let mut node = 0u32;
            let mut path: Vec<(u32, usize, PlayerId)> = Vec::new();
            loop {
                if let Some(o) = g.outcome() {
                    let v = terminal_value(o, self.observer);
                    self.backup_observer(&path, v);
                    self.simulations += 1;
                    self.stats.sims += 1;
                    self.stats.terminal_worlds += 1;
                    break;
                }
                if !self.nodes[node as usize].expanded {
                    // Leaf. Mark it now so parallel simulations in this batch
                    // don't queue it twice; they will pass through it with
                    // uniform-ish priors until the evaluation arrives.
                    self.nodes[node as usize].expanded = true;
                    self.add_virtual_loss(&path);
                    self.pending.push(PendingSim { path, leaf_node: node, weights: vec![1.0], term_sum: 0.0, term_n: 0.0 });
                    self.stats.leaf_worlds += 1;
                    self.stats.unique_leaves += 1;
                    leaves.push(g);
                    break;
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
        leaves
    }

    /// A determinization of the root where the observer is to act.
    fn root_world(&mut self) -> Option<Game> {
        (0..ROOT_TRIES).find_map(|_| self.determinize_root())
    }

    /// One determinization attempt. If resampling breaks the root (the
    /// observer is mid-search of their own library, which the engine's
    /// pending effect refers to), retry keeping that library as it is.
    fn determinize_root(&mut self) -> Option<Game> {
        let seed = self.rng.next_u64();
        let ok = |g: &Game| !g.is_over() && g.to_act() == self.observer;
        for keep_own_library in [false, true] {
            let mut g = self.root.clone();
            g.determinize_with(self.observer, seed, keep_own_library);
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
        while done < max_sims && attempts < max_sims * 4 {
            attempts += 1;
            let mut alive = if self.cfg.maple_resample { self.sample_worlds() } else { self.worlds.clone() };
            if alive.is_empty() {
                self.simulations += 1;
                self.stats.sims += 1;
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

    /// Merge worlds the evaluator cannot tell apart (same observation for
    /// the player to act and same choices).
    fn dedupe(&self, worlds: Vec<Game>) -> (Vec<Game>, Vec<f32>) {
        let mut games: Vec<Game> = Vec::with_capacity(worlds.len());
        let mut hashes: Vec<u64> = Vec::with_capacity(worlds.len());
        let mut weights: Vec<f32> = Vec::with_capacity(worlds.len());
        for w in worlds {
            let mut h = std::collections::hash_map::DefaultHasher::new();
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
            let node = &mut self.nodes[sim.leaf_node as usize];
            for (key, p) in aggregate_priors(&per_world) {
                match node.edges.iter_mut().find(|e| e.key == key) {
                    Some(e) => e.prior = p,
                    None => node.edges.push(Edge { key, prior: p, visits: 0.0, value_sum: 0.0, avail: 0.0, child: [NONE; 2] }),
                }
            }
            let (mut num, mut den) = (sim.term_sum, sim.term_n);
            for ((l, e), &w) in ls.iter().zip(es).zip(&sim.weights) {
                let v = if l.to_act() == self.observer { e.value } else { -e.value };
                num += w * v;
                den += w;
            }
            self.backup_observer(&sim.path, num / den.max(1e-6));
            self.simulations += 1;
            self.stats.sims += 1;
        }
        if self.cfg.root_noise > 0.0 && self.simulations > 0 && !self.nodes[0].edges.is_empty() {
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

    /// Visit counts per root choice, aligned with `root.choices()`.
    pub fn root_visits(&self) -> Vec<f32> {
        let edges = &self.nodes[0].edges;
        self.root
            .choices()
            .iter()
            .map(|c| edges.iter().find(|e| e.key == c.key).map_or(0.0, |e| e.visits))
            .collect()
    }

    /// Most visited root choice (ties: higher mean value, then prior).
    pub fn best(&self) -> usize {
        let edges = &self.nodes[0].edges;
        let score = |i: usize| {
            let key = self.root.choices()[i].key;
            edges.iter().find(|e| e.key == key).map_or((0.0, -2.0, 0.0), |e| {
                (e.visits, if e.visits > 0.0 { e.value_sum / e.visits } else { -2.0 }, e.prior)
            })
        };
        (0..self.root.choices().len())
            .max_by(|&a, &b| score(a).partial_cmp(&score(b)).unwrap())
            .unwrap_or(0)
    }

    /// Mean value of the root from the searching player's point of view.
    pub fn root_value(&self) -> f32 {
        let edges = &self.nodes[0].edges;
        let (n, w) = edges.iter().fold((0.0, 0.0), |(n, w), e| (n + e.visits, w + e.value_sum));
        if n > 0.0 { w / n } else { 0.0 }
    }

    // ------------------------------------------------------------ internals

    /// The child of `node`'s edge `e` where `actor` is to act, created on
    /// first use.
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
        // Edges for choices seen for the first time in this determinization
        // get the mean prior of the node's known edges.
        let mean_prior = if n.edges.is_empty() { 1.0 } else { n.edges.iter().map(|e| e.prior).sum::<f32>() / n.edges.len() as f32 };
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
        let edges = &mut self.nodes[0].edges;
        let noise: Vec<f32> = (0..edges.len()).map(|_| gamma_sample(&mut self.rng, alpha)).collect();
        let total: f32 = noise.iter().sum::<f32>().max(1e-9);
        for (e, n) in edges.iter_mut().zip(noise) {
            e.prior = (1.0 - eps) * e.prior + eps * n / total;
        }
    }
}

fn uniform01(rng: &mut SplitMix64) -> f32 {
    ((rng.next_u64() >> 40) as f32 + 0.5) / (1u64 << 24) as f32
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

// ------------------------------------------------------ simple evaluators

/// Uniform priors; value from a hand-written board evaluation. Cheap, no
/// playouts: for testing the search and as a bootstrap before a network.
pub struct HeuristicEval;

pub fn heuristic_value(g: &Game, pov: PlayerId) -> f32 {
    use mtg_kernel::card_def::CardType;
    use mtg_kernel::engine::{effective_power, effective_toughness};
    if let Some(o) = g.outcome() {
        return terminal_value(o, pov);
    }
    let st = &g.state;
    let side = |p: PlayerId| -> f32 {
        let ps = &st.players[p.index()];
        let mut s = ps.life as f32 * 0.5 + ps.hand.len() as f32 * 1.0;
        for &id in &ps.battlefield {
            let def = &mtg_kernel::card_def::CARD_DEFS[st.objects.get(id).card_def as usize];
            if def.has_type(CardType::Creature) {
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

/// Uniform priors; value = mean result of random playouts, cut off after
/// `max_decisions` and scored by [`heuristic_value`].
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

    fn configs() -> Vec<Config> {
        vec![Config::default(), maple(5, MapleSelect::RefWorld), maple(5, MapleSelect::Union), Config { maple_resample: true, ..maple(3, MapleSelect::RefWorld) }]
    }

    #[test]
    fn search_visits_sum_to_simulations() {
        let g = Game::new("Burn", "Rally", 1).unwrap();
        for cfg in configs() {
            let mut s = Search::new(&g, cfg.clone(), 7);
            s.run(200, 8, &mut HeuristicEval);
            let total: f32 = s.root_visits().iter().sum();
            assert!(s.simulations >= 200);
            assert!((total - s.simulations as f32).abs() <= 1.0, "{cfg:?}: {total} vs {}", s.simulations);
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

    /// Find a position where one choice leads to the opponent's decision in
    /// some determinizations and to the observer's in others.
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

    /// Mid-way through searching their own library (Wildfire's land
    /// fetchers), a plain determinization makes the engine halt. The search
    /// must still put its visits on the real root's choices.
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
                    for cfg in [Config::default(), maple(3, MapleSelect::RefWorld)] {
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
        // Key 3 is legal only in the first world; it keeps that world's prior
        // instead of being diluted by the world where it is illegal.
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
