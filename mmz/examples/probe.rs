use mmz::game::Game;
use mmz::ismcts::{Config, HeuristicEval, Search};
use mmz::mtg_kernel::state::SplitMix64;
fn main() {
    let mut g = Game::new("Burn", "Rally", 5).unwrap();
    let mut rng = SplitMix64::seed(1);
    for step in 0..30 {
        if g.is_over() { break; }
        let mut s = Search::new(&g, Config::default(), 9);
        s.run(100, 8, &mut HeuristicEval);
        let v = s.root_visits();
        let desc: Vec<String> = g.choices().iter().zip(&v).map(|(c, n)| format!("{:?}/{}={n}", c.feat.kind, c.feat.src)).collect();
        println!("{step} t{} p{} sims={} best={} val={:.2} {}", g.state.turn, g.to_act().0, s.simulations, s.best(), s.root_value(), desc.join(" "));
        let n = g.choices().len();
        let pick = if step % 2 == 0 { s.best() } else { (rng.next_u64() % n as u64) as usize };
        g.apply(pick);
    }
}
