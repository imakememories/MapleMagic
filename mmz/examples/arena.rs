//! Play two agents against each other on a deck pairing, in parallel.
//!
//! usage: arena <agentA> <agentB> <deckA> <deckB> <games> [threads]
//! agents: random | h<sims> (IS-MCTS, heuristic leaf value) | r<sims> (IS-MCTS, rollout value)
//! MAPLE suffix: m<k> = k worlds (e.g. h200m5), then u = paper-literal union
//! selection (h200m5u). Rollout agents turn dedupe off (rollouts read hidden
//! state). Seats alternate: in odd games A plays deckB. Results are from A's side.
use mmz::game::{Game, Outcome};
use mmz::ismcts::{Config, HeuristicEval, MapleSelect, RolloutEval, Search, SearchStats};
use mmz::mtg_kernel::ids::PlayerId;
use mmz::mtg_kernel::state::SplitMix64;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone)]
enum Agent {
    Random,
    Heuristic(u32, Config),
    Rollout(u32, Config),
}

fn parse(s: &str) -> Agent {
    if s == "random" {
        return Agent::Random;
    }
    let (kind, rest) = s.split_at(1);
    let digits = |t: &str| t.chars().take_while(|c| c.is_ascii_digit()).collect::<String>();
    let sims_s = digits(rest);
    let sims: u32 = sims_s.parse().unwrap_or_else(|_| panic!("unknown agent {s}"));
    let mut rest = &rest[sims_s.len()..];
    let mut cfg = Config::default();
    if let Some(r) = rest.strip_prefix('m') {
        let k = digits(r);
        cfg.maple_worlds = k.parse().unwrap_or_else(|_| panic!("bad MAPLE world count in {s}"));
        rest = &r[k.len()..];
    }
    if let Some(r) = rest.strip_prefix('u') {
        cfg.maple_select = MapleSelect::Union;
        rest = r;
    }
    assert!(rest.is_empty(), "unknown agent {s}");
    match kind {
        "h" => Agent::Heuristic(sims, cfg),
        "r" => Agent::Rollout(sims, Config { maple_dedupe: false, ..cfg }),
        _ => panic!("unknown agent {s}"),
    }
}

fn choose(agent: &Agent, g: &Game, rng: &mut SplitMix64, stats: &mut SearchStats) -> usize {
    let n = g.choices().len();
    match agent {
        Agent::Random => (rng.next_u64() % n as u64) as usize,
        Agent::Heuristic(sims, cfg) => {
            let mut s = Search::new(g, cfg.clone(), rng.next_u64());
            s.run(*sims, 8, &mut HeuristicEval);
            *stats += s.stats;
            s.best()
        }
        Agent::Rollout(sims, cfg) => {
            let mut s = Search::new(g, cfg.clone(), rng.next_u64());
            let mut ev = RolloutEval { rng: SplitMix64::seed(rng.next_u64()), rollouts: 1, max_decisions: 40 };
            s.run(*sims, 8, &mut ev);
            *stats += s.stats;
            s.best()
        }
    }
}

fn describe(name: &str, s: &SearchStats) -> String {
    if s.sims == 0 {
        return format!("{name}: no searches");
    }
    let sims = s.sims as f64;
    format!(
        "{name}: {:.2} leaf worlds/sim, {:.2} evals/sim, dropped {:.2} illegal + {:.2} diverged per sim, {:.3} terminal worlds/sim",
        s.leaf_worlds as f64 / sims,
        s.unique_leaves as f64 / sims,
        s.dropped_illegal as f64 / sims,
        s.dropped_diverged as f64 / sims,
        s.terminal_worlds as f64 / sims,
    )
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (agent_a, agent_b) = (parse(&a[1]), parse(&a[2]));
    let (deck_a, deck_b) = (a[3].clone(), a[4].clone());
    let games: u64 = a[5].parse().unwrap();
    let threads: u64 = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(8);
    let next = Arc::new(AtomicU64::new(0));
    let t0 = Instant::now();
    let handles: Vec<_> = (0..threads)
        .map(|_| {
            let (agent_a, agent_b, deck_a, deck_b, next) = (agent_a.clone(), agent_b.clone(), deck_a.clone(), deck_b.clone(), next.clone());
            std::thread::spawn(move || {
                let (mut w, mut l, mut d, mut dec) = (0u32, 0u32, 0u32, 0u64);
                let mut stats = [SearchStats::default(); 2];
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= games {
                        break;
                    }
                    // Game i: A sits in seat P0 with deckA on even i, seat P1 with deckB... keep
                    // decks fixed per seat and swap which agent sits where.
                    let a_seat = if i % 2 == 0 { PlayerId::P0 } else { PlayerId::P1 };
                    let (d0, d1) = if i % 2 == 0 { (&deck_a, &deck_b) } else { (&deck_b, &deck_a) };
                    let mut g = Game::new(d0, d1, 1000 + i / 2).unwrap();
                    let mut rng = SplitMix64::seed(i * 7919 + 13);
                    while !g.is_over() {
                        let (agent, side) = if g.to_act() == a_seat { (&agent_a, 0) } else { (&agent_b, 1) };
                        let c = choose(agent, &g, &mut rng, &mut stats[side]);
                        g.apply(c);
                    }
                    dec += g.decisions as u64;
                    match g.outcome().unwrap() {
                        Outcome::Win(p) if p == a_seat => w += 1,
                        Outcome::Win(_) => l += 1,
                        _ => d += 1,
                    }
                }
                (w, l, d, dec, stats)
            })
        })
        .collect();
    let (mut w, mut l, mut d, mut dec) = (0, 0, 0, 0);
    let mut stats = [SearchStats::default(); 2];
    for h in handles {
        let r = h.join().unwrap();
        (w, l, d, dec) = (w + r.0, l + r.1, d + r.2, dec + r.3);
        stats[0] += r.4[0];
        stats[1] += r.4[1];
    }
    let n = (w + l + d) as f64;
    let score = (w as f64 + 0.5 * d as f64) / n;
    let se = (score * (1.0 - score) / n).sqrt();
    println!(
        "{} ({}) vs {} ({}): W{w} L{l} D{d}  score {:.3} ± {:.3}  [{:.1}s, {:.0} decisions/game]",
        a[1], deck_a, a[2], deck_b, score, 1.96 * se, t0.elapsed().as_secs_f64(), dec as f64 / n
    );
    println!("  {}", describe(&a[1], &stats[0]));
    println!("  {}", describe(&a[2], &stats[1]));
}
