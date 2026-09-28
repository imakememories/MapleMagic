//! Random-policy throughput through the `Game` wrapper (auto-pay, flattened
//! combat), single thread and N threads.
use mmz::game::{deck_names, Game};
use mmz::mtg_kernel::state::SplitMix64;
use std::time::{Duration, Instant};

fn run(secs: f64, tid: u64) -> (u64, u64) {
    let decks = deck_names();
    let end = Instant::now() + Duration::from_secs_f64(secs);
    let (mut games, mut decisions) = (0u64, 0u64);
    let mut rng = SplitMix64::seed(tid.wrapping_mul(0x9E37) + 1);
    while Instant::now() < end {
        let a = decks[(rng.next_u64() % decks.len() as u64) as usize];
        let b = decks[(rng.next_u64() % decks.len() as u64) as usize];
        let mut g = Game::new(a, b, rng.next_u64()).unwrap();
        g.playout(&mut rng);
        games += 1;
        decisions += g.decisions as u64;
    }
    (games, decisions)
}

fn main() {
    let secs = 5.0;
    let max = std::thread::available_parallelism().map_or(8, |n| n.get());
    let mut counts = vec![1, 4, 8, max];
    counts.dedup();
    println!("threads  games/s  decisions/s  decisions/game");
    for t in counts {
        let hs: Vec<_> = (0..t).map(|i| std::thread::spawn(move || run(secs, i as u64))).collect();
        let (g, d) = hs.into_iter().map(|h| h.join().unwrap()).fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
        println!("{t:>7} {:>8.0} {:>12.0} {:>15.1}", g as f64 / secs, d as f64 / secs, d as f64 / g as f64);
    }
}
