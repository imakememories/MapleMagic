//! What each player has seen the other do, as tokens for the world sampler.
//! Only actions that always have a pass or "no" alternative are logged: whether
//! a pass was a real choice depends on hidden cards.

use crate::game::{Game, Kind};
use mtg_kernel::card_def::CARD_DEFS;
use mtg_kernel::ids::PlayerId;
use mtg_kernel::state::GameState;

pub const HIST_FIELDS: usize = 8;
pub const MAX_HIST: usize = 64;
const TURN_END: u16 = 31;

/// Vocabulary sizes per history field.
pub fn hist_vocab() -> [usize; HIST_FIELDS] {
    [32, CARD_DEFS.len() + 2, 3, 16, 16, CARD_DEFS.len() + 2, 16, 16]
}

#[derive(Debug, Clone, Copy)]
struct Event {
    kind: u16,
    actor: PlayerId,
    turn: u32,
    step: u16,
    card: u16,
    stack: u16,
    untapped: [u16; 2],
    hand: [u16; 2],
}

#[derive(Debug, Clone, Default)]
pub struct History {
    events: Vec<Event>,
    turn: u32,
    active: Option<PlayerId>,
    lands: u16,
    spells: [u16; 2],
}

fn untapped_lands(st: &GameState, p: PlayerId) -> u16 {
    st.players[p.index()]
        .battlefield
        .iter()
        .filter(|&&id| {
            let o = st.objects.get(id);
            !o.tapped && CARD_DEFS[o.card_def as usize].is_land
        })
        .count() as u16
}

fn snapshot(st: &GameState) -> ([u16; 2], [u16; 2]) {
    let u = [untapped_lands(st, PlayerId::P0), untapped_lands(st, PlayerId::P1)];
    let h = [st.players[0].hand.len() as u16, st.players[1].hand.len() as u16];
    (u, h)
}

impl History {
    /// Record choice `i` of `g`, before it is applied.
    pub fn record(&mut self, g: &Game, i: usize) {
        let st = &g.state;
        if st.turn != self.turn {
            self.close_turns(st);
        }
        let feat = g.choices()[i].feat;
        if !matches!(feat.kind, Kind::Cast | Kind::PlayLand | Kind::Activate | Kind::Plot | Kind::Foretell | Kind::Attack | Kind::Block) {
            return;
        }
        let actor = g.to_act();
        match feat.kind {
            Kind::PlayLand => self.lands += 1,
            Kind::Cast => self.spells[actor.index()] += 1,
            _ => {}
        }
        // Activations from hand and foretold cards don't show the card.
        let hidden_source = (feat.kind == Kind::Activate && feat.tgt <= 1) || feat.kind == Kind::Foretell;
        let card = if feat.src == u16::MAX || hidden_source { 0 } else { feat.src + 1 };
        let stack = st.stack.last().and_then(|s| st.objects.try_get(s.source)).map_or(0, |o| o.card_def + 1);
        let (untapped, hand) = snapshot(st);
        self.events.push(Event { kind: feat.kind as u16 + 1, actor, turn: st.turn, step: st.step as u16, card, stack, untapped, hand });
    }

    fn close_turns(&mut self, st: &GameState) {
        let (untapped, hand) = snapshot(st);
        if let Some(active) = self.active {
            for t in self.turn..st.turn {
                let actor = if (t - self.turn) % 2 == 0 { active } else { active.opponent() };
                let (lands, mine, theirs) =
                    if t == self.turn { (self.lands, self.spells[actor.index()], self.spells[actor.opponent().index()]) } else { (0, 0, 0) };
                self.events.push(Event { kind: TURN_END, actor, turn: t, step: lands, card: mine, stack: theirs, untapped, hand });
            }
        }
        self.turn = st.turn;
        self.active = Some(st.active_player);
        self.lands = 0;
        self.spells = [0, 0];
    }

    /// The latest events as `observer` sees them, oldest first.
    pub fn tokens(&self, observer: PlayerId, now_turn: u32) -> Vec<[u16; HIST_FIELDS]> {
        let opp = observer.opponent().index();
        let start = self.events.len().saturating_sub(MAX_HIST);
        self.events[start..]
            .iter()
            .map(|e| {
                let ago = now_turn.saturating_sub(e.turn).min(15) as u16;
                let actor = if e.actor == observer { 1 } else { 2 };
                let (step, card, stack) =
                    if e.kind == TURN_END { (e.step.min(15), e.card.min(15), e.stack.min(15)) } else { (e.step.min(15), e.card, e.stack) };
                [e.kind, card, actor, ago, step, stack, e.untapped[opp].min(15), e.hand[opp].min(15)]
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::observe;
    use crate::game::deck_names;
    use mtg_kernel::state::SplitMix64;

    #[test]
    fn activated_cards_are_ones_the_other_player_has_seen() {
        let mut rng = SplitMix64::seed(3);
        let (mut logged, mut hidden) = (0, Vec::new());
        for a in deck_names() {
            for b in deck_names() {
                let mut g = Game::new(a, b, 7).unwrap();
                let mut h = History::default();
                while !g.is_over() {
                    let n = g.choices().len();
                    let i = (rng.next_u64() % n as u64) as usize;
                    let before = h.events.len();
                    h.record(&g, i);
                    let new_card = h.events[before..].iter().rev().find(|e| e.kind == Kind::Activate as u16 + 1).map(|e| (e.card, e.actor));
                    let shown = |g: &Game, card: u16, actor: PlayerId| observe(g, actor.opponent()).iter().any(|t| t[0] == card);
                    let seen_before = new_card.is_some_and(|(card, actor)| shown(&g, card, actor));
                    g.apply(i);
                    if let Some((card, actor)) = new_card {
                        if card == 0 {
                            continue;
                        }
                        logged += 1;
                        let seen = seen_before || shown(&g, card, actor);
                        if !seen {
                            hidden.push(format!("{a} vs {b}: {}", CARD_DEFS[card as usize - 1].name));
                        }
                    }
                }
                let vocab = hist_vocab();
                for p in [PlayerId::P0, PlayerId::P1] {
                    let toks = h.tokens(p, g.state.turn);
                    assert!(toks.len() <= MAX_HIST);
                    for t in toks {
                        for (f, &v) in t.iter().enumerate() {
                            assert!((v as usize) < vocab[f], "field {f} = {v}");
                        }
                    }
                }
            }
        }
        assert!(logged > 500, "only {logged} logged activations");
        assert!(hidden.is_empty(), "{} of {logged} logged cards were never shown: {:?}", hidden.len(), &hidden[..hidden.len().min(10)]);
    }
}
