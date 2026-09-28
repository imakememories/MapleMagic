//! Play random games and print the choices leading up to each abort.
use mmz::game::{Game, Outcome};
use mmz::mtg_kernel::card_def::CARD_DEFS;
use mmz::mtg_kernel::state::SplitMix64;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let a = args.get(1).map(String::as_str).unwrap_or("Faeries");
    let b = args.get(2).map(String::as_str).unwrap_or("Faeries");
    let n: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(200);
    let mut shown = 0;
    let mut aborted = 0;
    for seed in 0..n {
        let mut g = Game::new(a, b, seed).unwrap();
        let mut rng = SplitMix64::seed(seed ^ 77);
        let mut log: Vec<String> = Vec::new();
        while !g.is_over() {
            let n = g.choices().len();
            let i = (rng.next_u64() % n as u64) as usize;
            let c = &g.choices()[i];
            let src = if c.feat.src == u16::MAX { "-" } else { CARD_DEFS[c.feat.src as usize].name };
            log.push(format!("t{} {:?} p{} {:?} {src} tgt={:x} arg={} | {:?}", g.state.turn, g.state.step, g.to_act().0, c.feat.kind, c.feat.tgt, c.feat.arg, c));
            g.apply(i);
        }
        if g.outcome() == Some(Outcome::Aborted) {
            aborted += 1;
            if shown < 4 {
                shown += 1;
                println!("=== seed {seed}: {}", g.abort_reason.as_deref().unwrap_or("?"));
                for l in log.iter().rev().take(6).rev() {
                    println!("  {}", &l[..l.len().min(260)]);
                }
            }
        }
    }
    println!("{aborted}/{n} aborted");
}
