//! Single-observer information-set MCTS (SO-ISMCTS) with PUCT.
//!
//! Every simulation starts from a fresh determinization of the root: cards
//! the searching player cannot see are resampled. Tree nodes are reached by
//! the sequence of semantic choice keys from the root, so the same node
//! gathers statistics across all determinizations. Because the legal choices
//! differ between determinizations, each edge counts how often it was
//! available and PUCT uses that count in place of the parent visit count
//! (the "subset-armed bandit" form of ISMCTS).
//!
//! The search is step-driven so a batched evaluator (a GPU network) can sit
//! outside it: [`Search::gather`] runs selection until it has a batch of
//! leaves, the caller evaluates them, and [`Search::feed`] backs the values
//! up. Virtual loss keeps concurrent simulations of one tree apart.

use crate::game::{Game, Outcome};
use mtg_kernel::ids::PlayerId;
use mtg_kernel::state::SplitMix64;

#[derive(Debug, Clone)]
pub struct Config {
    pub c_puct: f32,
    /// Dirichlet noise weight at the root (0 disables).
    pub root_noise: f32,
    pub dirichlet_alpha: f32,
    /// Value assumed for unvisited edges, relative to the parent's mean.
    pub fpu_reduction: f32,
    pub virtual_loss: f32,
}

impl Default for Config {
    fn default() -> Self {
        Config { c_puct: 1.0, root_noise: 0.0, dirichlet_alpha: 0.3, fpu_reduction: 0.2, virtual_loss: 1.0 }
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

#[derive(Debug, Clone)]
struct Edge {
    key: u64,
    prior: f32,
    visits: f32,
    /// Sum of values from the point of view of the player choosing here.
    value_sum: f32,
    avail: f32,
    child: u32,
}

#[derive(Debug, Clone, Default)]
struct Node {
    edges: Vec<Edge>,
    expanded: bool,
}

struct PendingSim {
    path: Vec<(u32, usize, PlayerId)>,
    leaf_node: u32,
}

pub struct Search {
    root: Game,
    observer: PlayerId,
    cfg: Config,
    nodes: Vec<Node>,
    rng: SplitMix64,
    pending: Vec<PendingSim>,
    pub simulations: u32,
}

fn terminal_value(o: Outcome, pov: PlayerId) -> f32 {
    match o {
        Outcome::Win(p) if p == pov => 1.0,
        Outcome::Win(_) => -1.0,
        Outcome::Draw | Outcome::Aborted => 0.0,
    }
}

impl Search {
    pub fn new(root: &Game, cfg: Config, seed: u64) -> Search {
        Search {
            observer: root.to_act(),
            root: root.clone(),
            cfg,
            nodes: vec![Node::default()],
            rng: SplitMix64::seed(seed),
            pending: Vec::new(),
            simulations: 0,
        }
    }

    /// Run selection for up to `max_leaves` simulations. Simulations that
    /// end at a terminal state are backed up immediately; the rest return
    /// their leaf position for evaluation. The returned games must be
    /// evaluated and passed to [`Self::feed`] in the same order.
    pub fn gather(&mut self, max_leaves: usize) -> Vec<Game> {
        let mut leaves = Vec::new();
        let mut attempts = 0;
        while leaves.len() < max_leaves && attempts < max_leaves * 4 {
            attempts += 1;
            let mut g = self.root.clone();
            let seed = self.rng.next_u64();
            g.determinize(self.observer, seed);
            let mut node = 0u32;
            let mut path: Vec<(u32, usize, PlayerId)> = Vec::new();
            loop {
                if let Some(o) = g.outcome() {
                    self.backup(&path, |p| terminal_value(o, p), false);
                    self.simulations += 1;
                    break;
                }
                if !self.nodes[node as usize].expanded {
                    // Leaf. Mark it now so parallel simulations in this batch
                    // don't queue it twice; they will pass through it with
                    // uniform-ish priors until the evaluation arrives.
                    self.nodes[node as usize].expanded = true;
                    self.add_virtual_loss(&path);
                    self.pending.push(PendingSim { path, leaf_node: node });
                    leaves.push(g);
                    break;
                }
                let chooser = g.to_act();
                let e = self.select(node, &g);
                let edge = &self.nodes[node as usize].edges[e];
                let key = edge.key;
                let child = if edge.child == NONE {
                    self.nodes.push(Node::default());
                    let c = (self.nodes.len() - 1) as u32;
                    self.nodes[node as usize].edges[e].child = c;
                    c
                } else {
                    edge.child
                };
                let idx = g.choices().iter().position(|c| c.key == key).expect("selected key is legal");
                path.push((node, e, chooser));
                g.apply(idx);
                node = child;
            }
        }
        leaves
    }

    /// Back up evaluations for the leaves returned by the last [`Self::gather`].
    pub fn feed(&mut self, leaves: &[Game], evals: Vec<Eval>) {
        let pending = std::mem::take(&mut self.pending);
        assert_eq!(pending.len(), evals.len());
        for ((sim, ev), leaf) in pending.into_iter().zip(evals).zip(leaves) {
            self.remove_virtual_loss(&sim.path);
            let node = &mut self.nodes[sim.leaf_node as usize];
            let total: f32 = ev.priors.iter().sum::<f32>().max(1e-6);
            for (c, &p) in leaf.choices().iter().zip(&ev.priors) {
                match node.edges.iter_mut().find(|e| e.key == c.key) {
                    Some(e) => e.prior = p / total,
                    None => node.edges.push(Edge { key: c.key, prior: p / total, visits: 0.0, value_sum: 0.0, avail: 0.0, child: NONE }),
                }
            }
            let pov = leaf.to_act();
            let v = ev.value;
            self.backup(&sim.path, |p| if p == pov { v } else { -v }, false);
            self.simulations += 1;
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

    fn select(&mut self, node: u32, g: &Game) -> usize {
        let cfg = self.cfg.clone();
        let n = &mut self.nodes[node as usize];
        // Edges for choices seen for the first time in this determinization
        // get the mean prior of the node's known edges.
        let mean_prior = if n.edges.is_empty() { 1.0 } else { n.edges.iter().map(|e| e.prior).sum::<f32>() / n.edges.len() as f32 };
        let mut legal: Vec<usize> = Vec::with_capacity(g.choices().len());
        for c in g.choices() {
            let i = match n.edges.iter().position(|e| e.key == c.key) {
                Some(i) => i,
                None => {
                    n.edges.push(Edge { key: c.key, prior: mean_prior, visits: 0.0, value_sum: 0.0, avail: 0.0, child: NONE });
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

    fn backup(&mut self, path: &[(u32, usize, PlayerId)], value_for: impl Fn(PlayerId) -> f32, _virtual: bool) {
        for &(node, e, chooser) in path {
            let edge = &mut self.nodes[node as usize].edges[e];
            edge.visits += 1.0;
            edge.value_sum += value_for(chooser);
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

    #[test]
    fn search_visits_sum_to_simulations() {
        let g = Game::new("Burn", "Rally", 1).unwrap();
        let mut s = Search::new(&g, Config::default(), 7);
        s.run(200, 8, &mut HeuristicEval);
        let total: f32 = s.root_visits().iter().sum();
        assert!(s.simulations >= 200);
        assert!((total - s.simulations as f32).abs() <= 1.0, "{total} vs {}", s.simulations);
        assert!(s.best() < g.choices().len());
    }

    #[test]
    fn search_never_changes_root() {
        let g = Game::new("Faeries", "Elves", 2).unwrap();
        let h = g.state.state_hash();
        let mut s = Search::new(&g, Config::default(), 3);
        s.run(100, 4, &mut HeuristicEval);
        assert_eq!(g.state.state_hash(), h);
    }
}
