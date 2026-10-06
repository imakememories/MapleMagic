//! Random games between two Commander lists, as an engine smoke test. Cards
//! the engine can't play yet are Mountains. Reports results, aborts and
//! panics (with seeds), and how often each implemented card was used. With
//! `--check`, every decision also checks that resampling hidden cards leaves
//! the acting player's observation unchanged.
//!
//! `--replay` plays only the first seed and prints its last decisions.
//!
//! `--threads N` caps the worker threads (default: every core).
//!
//! usage: random_games <deck a> <deck b> [games] [first seed] [--check] [--replay] [--threads N]
use mmz::commander::CommanderDeck;
use mmz::features::observe;
use mmz::game::{Game, Kind, Outcome};
use mmz::mtg_kernel::card_def::CARD_DEFS;
use mmz::mtg_kernel::state::SplitMix64;
use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};

#[derive(Default)]
struct Tally {
    games: u32,
    wins: [u32; 2],
    draws: u32,
    capped: u32,
    turns: u64,
    decisions: u64,
    problems: BTreeMap<String, Vec<u64>>,
    used: BTreeMap<u16, u32>,
}

fn play(a: &CommanderDeck, b: &CommanderDeck, seed: u64, check: bool, t: &mut Tally) {
    let mut g = Game::new_commander(a, b, seed);
    let mut rng = SplitMix64::seed(seed ^ 0x5eed);
    while !g.is_over() {
        let me = g.to_act();
        if check {
            let mut w = g.clone();
            w.determinize(me, rng.next_u64());
            if observe(&w, me) != observe(&g, me) {
                t.problems.entry("observation changed when hidden cards were resampled".into()).or_default().push(seed);
                break;
            }
        }
        let i = (rng.next_u64() % g.choices().len() as u64) as usize;
        let f = g.choices()[i].feat;
        if matches!(f.kind, Kind::Cast | Kind::PlayLand | Kind::Activate | Kind::Plot | Kind::Foretell) && f.src != u16::MAX {
            *t.used.entry(f.src).or_default() += 1;
        }
        g.apply(i);
    }
    t.games += 1;
    t.turns += u64::from(g.state.turn);
    t.decisions += u64::from(g.decisions);
    match g.outcome() {
        Some(Outcome::Win(p)) => t.wins[p.index()] += 1,
        Some(Outcome::Draw) => {
            t.draws += 1;
            if !g.state.players.iter().any(|p| p.has_lost) {
                t.capped += 1;
            }
        }
        Some(Outcome::Aborted) => {
            let why = g.abort_reason.clone().unwrap_or_else(|| "aborted".into());
            t.problems.entry(why).or_default().push(seed);
        }
        None => {}
    }
}

/// Replays one game exactly as `play` does and prints how it ended.
fn replay(a: &CommanderDeck, b: &CommanderDeck, seed: u64, check: bool) {
    let mut g = Game::new_commander(a, b, seed);
    let mut rng = SplitMix64::seed(seed ^ 0x5eed);
    let mut log = Vec::new();
    while !g.is_over() {
        let me = g.to_act();
        if check {
            let mut w = g.clone();
            w.determinize(me, rng.next_u64());
            if observe(&w, me) != observe(&g, me) {
                log.push("observation changed when hidden cards were resampled".to_string());
                break;
            }
        }
        let i = (rng.next_u64() % g.choices().len() as u64) as usize;
        let f = g.choices()[i].feat;
        let card = if f.src == u16::MAX { "-" } else { CARD_DEFS[f.src as usize].name };
        let st = &g.state;
        let stack: Vec<&str> = st.stack.iter().map(|it| st.objects.get(it.source).name.as_str()).collect();
        log.push(format!("turn {} {:?} {me:?}: {:?} {card} (arg {}) | stack {stack:?}", st.turn, st.step, f.kind, f.arg));
        g.apply(i);
    }
    for line in &log[log.len().saturating_sub(15)..] {
        println!("{line}");
    }
    println!("outcome {:?} {}", g.outcome(), g.abort_reason.unwrap_or_default());
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let check = args.iter().any(|a| a == "--check");
    let threads_flag = args.iter().position(|a| a == "--threads");
    let cap: Option<u64> = threads_flag.and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok());
    let pos: Vec<&String> = args
        .iter()
        .enumerate()
        .filter(|&(i, a)| !a.starts_with("--") && threads_flag.is_none_or(|t| i != t + 1))
        .map(|(_, a)| a)
        .collect();
    let (da, db) = (pos.first().map_or("Ognis", |s| s.as_str()), pos.get(1).map_or("Ognis", |s| s.as_str()));
    let games: u64 = pos.get(2).and_then(|s| s.parse().ok()).unwrap_or(200);
    let first: u64 = pos.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
    let deck = |n: &str| CommanderDeck::named_with_stand_in(n, Some("Mountain")).expect("a Commander list");
    let (a, b) = (deck(da), deck(db));
    if args.iter().any(|a| a == "--replay") {
        return replay(&a, &b, first, check);
    }
    let cores = std::thread::available_parallelism().map_or(4, |n| n.get()) as u64;
    let threads = cap.map_or(cores, |c| c.clamp(1, cores));
    let parts: Vec<Tally> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|k| {
                let (a, b) = (&a, &b);
                s.spawn(move || {
                    let mut t = Tally::default();
                    for seed in (first + k..first + games).step_by(threads as usize) {
                        if catch_unwind(AssertUnwindSafe(|| play(a, b, seed, check, &mut t))).is_err() {
                            t.problems.entry("panic".into()).or_default().push(seed);
                        }
                    }
                    t
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut t = Tally::default();
    for p in parts {
        t.games += p.games;
        t.draws += p.draws;
        t.capped += p.capped;
        t.turns += p.turns;
        t.decisions += p.decisions;
        for i in 0..2 {
            t.wins[i] += p.wins[i];
        }
        for (k, v) in p.problems {
            t.problems.entry(k).or_default().extend(v);
        }
        for (k, v) in p.used {
            *t.used.entry(k).or_default() += v;
        }
    }
    let n = t.games.max(1) as f64;
    println!("{da} vs {db}: {} games from seed {first}{}", t.games, if check { ", hidden-information check on" } else { "" });
    println!(
        "wins {} / {}, draws {} ({} at the turn or decision cap), {:.1} turns and {:.0} decisions a game",
        t.wins[0], t.wins[1], t.draws, t.capped, t.turns as f64 / n, t.decisions as f64 / n
    );
    if t.problems.is_empty() {
        println!("no aborts or panics");
    }
    for (why, mut seeds) in t.problems {
        seeds.sort_unstable();
        println!("PROBLEM x{}: {why} (seeds {:?})", seeds.len(), &seeds[..seeds.len().min(8)]);
    }
    let mut cards: Vec<u16> = a.cards.iter().chain(&b.cards).copied().chain([a.commander, b.commander]).collect();
    cards.sort_unstable();
    cards.dedup();
    println!("uses per implemented card:");
    for c in cards {
        let name = CARD_DEFS[c as usize].name;
        if name != "Mountain" {
            println!("  {:>6}  {name}", t.used.get(&c).copied().unwrap_or(0));
        }
    }
}
