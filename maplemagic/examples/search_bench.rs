//! Single-thread cost of the search's primitives on mid-game positions:
//! cloning a game, dealing a world, one engine step (pass priority and advance
//! to the next decision, without the search's choice list), one search step
//! (apply a choice and build the next decision's choices), and whole searches
//! per simulation.
//!
//! usage: search_bench [positions]
use maplemagic::commander::CommanderDeck;
use maplemagic::game::Game;
use maplemagic::ismcts::{Config, HeuristicEval, MapleSelect, Search};
use maplemagic::mtg_kernel::card_def::card_id_by_name;
use maplemagic::mtg_kernel::engine::{self, Action, Decision};
use maplemagic::mtg_kernel::runtime_decks::runtime_deck_by_id;
use maplemagic::mtg_kernel::state::SplitMix64;
use std::hint::black_box;
use std::time::Instant;

/// A runtime deck padded to 99 cards with Mountains, led by `commander`.
fn padded(commander: &str, base: &str) -> CommanderDeck {
    let mut cards = runtime_deck_by_id(base).unwrap().card_ids.to_vec();
    cards.resize(99, card_id_by_name("Mountain").unwrap());
    CommanderDeck { name: base.into(), commander: card_id_by_name(commander).unwrap(), cards }
}

/// Positions from random games, turns 6 to 14.
fn positions(n: usize, new_game: &dyn Fn(u64) -> Game) -> Vec<Game> {
    let mut rng = SplitMix64::seed(1);
    let mut out = Vec::new();
    let mut seed = 0;
    while out.len() < n {
        let mut g = new_game(seed);
        seed += 1;
        while !g.is_over() && out.len() < n {
            if (6..=14).contains(&g.state.turn) && g.choices().len() >= 2 && rng.next_u64() % 8 == 0 {
                out.push(g.clone());
            }
            let k = g.choices().len();
            g.apply((rng.next_u64() % k as u64) as usize);
        }
    }
    out
}

/// Mean microseconds per call of `f` over `reps` passes of `ps`.
fn time(ps: &[Game], reps: usize, mut f: impl FnMut(&Game, u64)) -> f64 {
    let t = Instant::now();
    let mut i = 0u64;
    for _ in 0..reps {
        for g in ps {
            f(g, i);
            i += 1;
        }
    }
    t.elapsed().as_secs_f64() * 1e6 / i as f64
}

fn report(label: &str, ps: &[Game]) {
    let objects: f64 = ps.iter().map(|g| g.state.objects.len() as f64).sum::<f64>() / ps.len() as f64;
    let choices: f64 = ps.iter().map(|g| g.choices().len() as f64).sum::<f64>() / ps.len() as f64;
    let clone = time(ps, 20, |g, _| {
        black_box(g.clone());
    });
    let deal = time(ps, 10, |g, s| {
        let mut w = g.clone();
        w.determinize(g.to_act(), s);
        black_box(w);
    });
    let at_priority: Vec<Game> = ps
        .iter()
        .filter(|g| matches!(engine::advance_until_decision(&mut g.state.clone()), Decision::CastSpellOrPass { .. }))
        .cloned()
        .collect();
    let engine_step = time(&at_priority, 10, |g, _| {
        let mut st = g.state.clone();
        engine::step(&mut st, Action::Pass).unwrap();
        black_box(engine::advance_until_decision(&mut st));
    });
    let step = time(ps, 10, |g, s| {
        let mut w = g.clone();
        w.apply((s % w.choices().len() as u64) as usize);
        black_box(w);
    });
    // The digest of every root visit count tells whether two builds search alike.
    let search = |cfg: Config, sims: u32| {
        let t = Instant::now();
        let (mut n, mut evals, mut digest) = (0u64, 0u64, 0xcbf2_9ce4_8422_2325u64);
        for (i, g) in ps.iter().enumerate() {
            let mut s = Search::new(g, cfg.clone(), i as u64);
            s.run(sims, 8, &mut HeuristicEval);
            n += s.simulations as u64;
            evals += s.stats.unique_leaves;
            for v in s.root_visits() {
                digest = (digest ^ u64::from(v.to_bits())).wrapping_mul(0x100_0000_01b3);
            }
        }
        (t.elapsed().as_secs_f64() * 1e6 / n as f64, evals as f64 / n as f64, digest)
    };
    let (ismcts, _, ismcts_digest) = search(Config::default(), 200);
    let (maple, maple_evals, maple_digest) =
        search(Config { maple_worlds: 5, maple_select: MapleSelect::RefWorld, ..Config::default() }, 64);
    println!(
        "{label:<22} {:>7.0} {:>7.1} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>9.1} {:>8.1} ({maple_evals:.1} evals/sim) {:04x}{:04x}",
        objects,
        choices,
        clone,
        deal,
        engine_step,
        step,
        ismcts,
        maple,
        (ismcts_digest >> 48) ^ (ismcts_digest >> 32) & 0xffff,
        (maple_digest >> 48) ^ (maple_digest >> 32) & 0xffff,
    );
}

fn main() {
    let n: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(60);
    println!("{:<22} {:>7} {:>7} {:>8} {:>8} {:>8} {:>8} {:>9} {:>8}", "positions", "objects", "choices", "clone", "deal", "engine", "step", "IS-MCTS", "MAPLE");
    println!("{:<22} {:>7} {:>7} {:>8} {:>8} {:>8} {:>8} {:>9} {:>8}", "", "", "", "us", "us", "us", "us", "us/sim", "us/sim");
    report("Pauper (Faeries/Spy)", &positions(n, &|s| Game::new("Faeries", "Spy", s).unwrap()));
    report("Pauper (Rally/Burn)", &positions(n, &|s| Game::new("Rally", "Burn", s).unwrap()));
    let (a, b) = (padded("Masked Meower", "Rally"), padded("Goblin Tomb Raider", "Burn"));
    report("Commander (stand-in)", &positions(n, &|s| Game::new_commander(&a, &b, s)));
    let ognis = CommanderDeck::named_with_stand_in("Ognis", Some("Mountain")).unwrap();
    report("Commander (Ognis)", &positions(n, &|s| Game::new_commander(&ognis, &ognis, s)));
}
