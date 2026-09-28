//! Play two agents against each other on a deck pairing, in parallel.
//!
//! usage: arena <agentA> <agentB> <deckA> <deckB> <games> [threads]
//! agents: random | h<sims> (IS-MCTS, heuristic leaf value) | r<sims> (IS-MCTS, rollout value)
//! Seats alternate: in odd games A plays deckB. Results are from A's side.
use mmz::game::{Game, Outcome};
use mmz::ismcts::{Config, HeuristicEval, RolloutEval, Search};
use mmz::mtg_kernel::ids::PlayerId;
use mmz::mtg_kernel::state::SplitMix64;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone)]
enum Agent {
    Random,
    Heuristic(u32),
    Rollout(u32),
}

fn parse(s: &str) -> Agent {
    match s {
        "random" => Agent::Random,
        _ if s.starts_with('h') => Agent::Heuristic(s[1..].parse().unwrap()),
        _ if s.starts_with('r') => Agent::Rollout(s[1..].parse().unwrap()),
        _ => panic!("unknown agent {s}"),
    }
}

fn choose(agent: &Agent, g: &Game, rng: &mut SplitMix64) -> usize {
    let n = g.choices().len();
    match agent {
        Agent::Random => (rng.next_u64() % n as u64) as usize,
        Agent::Heuristic(sims) => {
            let mut s = Search::new(g, Config::default(), rng.next_u64());
            s.run(*sims, 8, &mut HeuristicEval);
            s.best()
        }
        Agent::Rollout(sims) => {
            let mut s = Search::new(g, Config::default(), rng.next_u64());
            let mut ev = RolloutEval { rng: SplitMix64::seed(rng.next_u64()), rollouts: 1, max_decisions: 40 };
            s.run(*sims, 8, &mut ev);
            s.best()
        }
    }
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
                        let agent = if g.to_act() == a_seat { &agent_a } else { &agent_b };
                        let c = choose(agent, &g, &mut rng);
                        g.apply(c);
                    }
                    dec += g.decisions as u64;
                    match g.outcome().unwrap() {
                        Outcome::Win(p) if p == a_seat => w += 1,
                        Outcome::Win(_) => l += 1,
                        _ => d += 1,
                    }
                }
                (w, l, d, dec)
            })
        })
        .collect();
    let (w, l, d, dec) = handles.into_iter().map(|h| h.join().unwrap()).fold((0, 0, 0, 0), |x, y| (x.0 + y.0, x.1 + y.1, x.2 + y.2, x.3 + y.3));
    let n = (w + l + d) as f64;
    let score = (w as f64 + 0.5 * d as f64) / n;
    let se = (score * (1.0 - score) / n).sqrt();
    println!(
        "{} ({}) vs {} ({}): W{w} L{l} D{d}  score {:.3} ± {:.3}  [{:.1}s, {:.0} decisions/game]",
        a[1], deck_a, a[2], deck_b, score, 1.96 * se, t0.elapsed().as_secs_f64(), dec as f64 / n
    );
}
