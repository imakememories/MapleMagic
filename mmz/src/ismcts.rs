//! Information-set MCTS with PUCT, and the MAPLE and PIMC (AlphaZe\*\*)
//! variants. [`Search::gather`] and [`Search::feed`] let a batched network
//! evaluate leaves outside the search.

use crate::features::{choice_fields, observe_with};
use crate::game::{Game, Outcome};
use mtg_kernel::ids::PlayerId;
use mtg_kernel::state::SplitMix64;
use std::hash::{Hash, Hasher};

/// How MAPLE picks the acting player and legal keys when worlds disagree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MapleSelect {
    /// A random alive world decides.
    #[default]
    RefWorld,
    /// The majority player acts, over the union of legal keys.
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
    /// Evaluate identical leaf observations once (off for rollouts, which read hidden state).
    pub maple_dedupe: bool,
    /// Observation encoding the evaluator uses (only affects dedupe).
    pub perfect_obs: bool,
    /// PIMC world count (0 = off): independent trees in fixed worlds.
    pub pimc_worlds: u32,
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
    /// Fixed worlds: MAPLE's, or PIMC's (world `i` searched from root node `i`).
    worlds: Vec<Game>,
    /// PIMC: the world the next simulation descends in.
    next_world: usize,
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
            nodes: vec![Node::new(observer)],
            rng: SplitMix64::seed(seed),
            pending: Vec::new(),
            worlds: Vec::new(),
            next_world: 0,
            simulations: 0,
            stats: SearchStats::default(),
            cfg,
        };
        assert!(s.cfg.maple_worlds == 0 || s.cfg.pimc_worlds == 0, "MAPLE and PIMC are exclusive");
        if s.cfg.maple_worlds > 0 && !s.cfg.maple_resample && !s.root.is_over() {
            s.worlds = s.sample_worlds(s.cfg.maple_worlds);
        }
        if s.cfg.pimc_worlds > 0 && !s.root.is_over() {
            s.worlds = s.sample_worlds(s.cfg.pimc_worlds);
            for _ in 1..s.worlds.len() {
                s.nodes.push(Node::new(observer));
            }
        }
        s
    }

    /// Up to `k` determinizations where the observer is to act.
    fn sample_worlds(&mut self, k: u32) -> Vec<Game> {
        let k = k as usize;
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

    /// Leaves for up to `max_leaves` simulations; evaluate them and pass the results to [`Self::feed`] in order.
    pub fn gather(&mut self, max_leaves: usize) -> Vec<Game> {
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
            self.descend(g, root, &mut leaves);
        }
        leaves
    }

    /// One simulation in world `g` from root node `root`.
    fn descend(&mut self, mut g: Game, root: u32, leaves: &mut Vec<Game>) {
        let mut node = root;
        let mut path: Vec<(u32, usize, PlayerId)> = Vec::new();
        loop {
            if let Some(o) = g.outcome() {
                let v = terminal_value(o, self.observer);
                self.backup_observer(&path, v);
                self.simulations += 1;
                self.stats.sims += 1;
                self.stats.terminal_worlds += 1;
                return;
            }
            if !self.nodes[node as usize].expanded {
                // Marked now so other simulations in this batch don't queue it again.
                self.nodes[node as usize].expanded = true;
                self.add_virtual_loss(&path);
                self.pending.push(PendingSim { path, leaf_node: node, weights: vec![1.0], term_sum: 0.0, term_n: 0.0 });
                self.stats.leaf_worlds += 1;
                self.stats.unique_leaves += 1;
                leaves.push(g);
                return;
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

    /// A determinization of the root where the observer is to act.
    fn root_world(&mut self) -> Option<Game> {
        (0..ROOT_TRIES).find_map(|_| self.determinize_root())
    }

    /// One determinization, retried keeping the observer's library if resampling it breaks the root.
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

    /// Merge worlds the evaluator can't tell apart.
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

    fn configs() -> Vec<Config> {
        vec![
            Config::default(),
            maple(5, MapleSelect::RefWorld),
            maple(5, MapleSelect::Union),
            Config { maple_resample: true, ..maple(3, MapleSelect::RefWorld) },
            pimc(5),
        ]
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
            assert!((total - s.simulations as f32).abs() <= roots, "{cfg:?}: {total} vs {}", s.simulations);
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
        let mut s = Search::new(&g, pimc(5), 11);
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

    #[test]
    fn search_ignores_where_hidden_cards_sit() {
        use crate::game::{set_card, unseen_slots};
        let mut rng = SplitMix64::seed(21);
        let mut checked = 0;
        for seed in 0..20 {
            let mut g = Game::new("Faeries", "Terror", seed).unwrap();
            while !g.is_over() && checked < 12 {
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
                    for cfg in [Config::default(), maple(5, MapleSelect::RefWorld), pimc(5)] {
                        let mut a = Search::new(&g, cfg.clone(), 9);
                        let mut b = Search::new(&swapped, cfg.clone(), 9);
                        a.run(48, 8, &mut HeuristicEval);
                        b.run(48, 8, &mut HeuristicEval);
                        assert_eq!(a.root_visits(), b.root_visits(), "{cfg:?}");
                    }
                    checked += 1;
                }
                let n = g.choices().len();
                g.apply((rng.next_u64() % n as u64) as usize);
            }
        }
        assert!(checked >= 12, "only {checked} positions with a hidden card to swap");
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
