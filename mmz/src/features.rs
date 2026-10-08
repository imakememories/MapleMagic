//! Observation and action encodings for the network, from one player's view:
//! hidden cards are encoded as unknown, so a determinized world encodes the
//! same as the real one.
//!
//! Token fields: card + 1 (0 = hidden) or a global, place, flags, power,
//! toughness, damage, +1/+1 counters, -1/-1 counters.

use crate::game::{ActFeat, Game};
use mtg_kernel::card_def::{CardType, CARD_DEFS};
use mtg_kernel::engine::{effective_power, effective_toughness};
use mtg_kernel::ids::{ObjectId, PlayerId};
use mtg_kernel::state::{GameState, Zone};

pub const TOKEN_FIELDS: usize = 8;
pub const ACTION_FIELDS: usize = 5;
pub const MAX_TOKENS: usize = 160;

/// Vocabulary sizes per token field (embedding table sizes).
pub fn token_vocab() -> [usize; TOKEN_FIELDS] {
    [GLOBAL_BASE as usize + NUM_GLOBALS, 16, 32, 64, 64, 32, 32, 32]
}

/// Vocabulary sizes per action field.
pub fn action_vocab() -> [usize; ACTION_FIELDS] {
    [32, CARD_DEFS.len() + 2, CARD_DEFS.len() + 2, 64, 64]
}

pub const GLOBAL_BASE: u16 = 1024;
const NUM_GLOBALS: usize = 16;

pub(crate) fn place_code(zone: Zone, mine: bool) -> u16 {
    let z = match zone {
        Zone::Hand => 0,
        Zone::Battlefield => 1,
        Zone::Graveyard => 2,
        Zone::Exile => 3,
        Zone::Stack => 4,
        Zone::Library => 5,
        Zone::Command => 6,
    };
    2 + z * 2 + mine as u16
}

fn clamp(v: i32, lo: i32, hi: i32) -> u16 {
    (v.clamp(lo, hi) - lo) as u16
}

fn object_token(st: &GameState, id: ObjectId, viewer: PlayerId, zone: Zone, mine: bool) -> [u16; TOKEN_FIELDS] {
    let o = st.objects.get(id);
    let def = &CARD_DEFS[o.card_def as usize];
    let mut t = [0u16; TOKEN_FIELDS];
    t[0] = o.card_def + 1;
    t[1] = place_code(zone, mine);
    if zone == Zone::Battlefield {
        let combat = &st.engine.combat;
        let attacking = combat.attackers.contains(&id);
        let blocking = combat.blocked_by.iter().any(|(_, bs)| bs.contains(&id));
        let blocked = combat.blocked_by.iter().any(|(a, _)| *a == id);
        t[2] = o.tapped as u16 | (o.summoning_sick as u16) << 1 | (attacking as u16) << 2 | (blocking as u16) << 3 | (blocked as u16) << 4;
        if def.has_type(CardType::Creature) {
            t[3] = 1 + clamp(effective_power(st, id), -1, 40);
            t[4] = 1 + clamp(effective_toughness(st, id), -1, 40);
        }
        t[5] = clamp(o.damage as i32, 0, 31);
        t[6] = clamp(o.counters.plus1_plus1 as i32, 0, 31);
        t[7] = clamp(o.counters.minus1_minus1 as i32, 0, 31);
    }
    let _ = viewer;
    t
}

fn global_token(g: u16, value: i32) -> [u16; TOKEN_FIELDS] {
    let mut t = [0u16; TOKEN_FIELDS];
    t[0] = GLOBAL_BASE + g;
    t[1] = 1;
    t[3] = clamp(value, 0, 63);
    t
}

/// What `viewer` can see, in at most [`MAX_TOKENS`] tokens.
pub fn observe(g: &Game, viewer: PlayerId) -> Vec<[u16; TOKEN_FIELDS]> {
    observe_with(g, viewer, false)
}

/// [`observe`], or with `perfect` the opponent's whole hand by identity.
pub fn observe_with(g: &Game, viewer: PlayerId, perfect: bool) -> Vec<[u16; TOKEN_FIELDS]> {
    let st = &g.state;
    let me = viewer;
    let opp = viewer.opponent();
    let (pm, po) = (&st.players[me.index()], &st.players[opp.index()]);
    let mut out = Vec::with_capacity(96);

    // Globals.
    let step = st.step as i32;
    let pool: i32 = pm.mana_pool.iter().map(|&x| x as i32).sum();
    for (i, v) in [
        pm.life,
        po.life,
        st.turn.min(63) as i32,
        step,
        (st.active_player == me) as i32,
        (g.to_act() == me) as i32,
        pm.lands_played_this_turn as i32,
        pm.hand.len() as i32,
        po.hand.len() as i32,
        pm.library.len() as i32,
        po.library.len() as i32,
        pool,
        st.stack.len() as i32,
        (st.initiative == Some(me)) as i32 + 2 * (st.initiative == Some(opp)) as i32,
        pm.spells_cast_this_turn as i32,
        po.spells_cast_this_turn as i32,
    ]
    .into_iter()
    .enumerate()
    {
        out.push(global_token(i as u16, v));
    }

    // Commanders: whether in the command zone, casts so far (the tax), damage dealt.
    if let Some(c) = &st.commander {
        for p in [me, opp] {
            let o = st.objects.get(c.commanders[p.index()]);
            let mut t = [0u16; TOKEN_FIELDS];
            t[0] = o.card_def + 1;
            t[1] = place_code(Zone::Command, p == me);
            t[2] = (o.zone == Zone::Command) as u16;
            t[5] = clamp(c.casts[p.index()] as i32, 0, 31);
            t[7] = clamp(c.damage[p.opponent().index()][p.index()], 0, 31);
            out.push(t);
        }
    }

    // Public zones and my hand.
    for &id in &pm.hand {
        out.push(object_token(st, id, me, Zone::Hand, true));
    }
    // Opponent hand: known cards by identity, the rest hidden.
    for &id in &po.hand {
        let zcc = st.objects.get(id).zone_change_count;
        let known = perfect || st.known_hand_cards(me, opp).iter().any(|e| e.object == id && e.zone_change_count == zcc);
        if known {
            out.push(object_token(st, id, me, Zone::Hand, false));
        } else {
            let mut t = [0u16; TOKEN_FIELDS];
            t[1] = place_code(Zone::Hand, false);
            out.push(t);
        }
    }
    for p in [me, opp] {
        for &id in &st.players[p.index()].battlefield {
            out.push(object_token(st, id, me, Zone::Battlefield, p == me));
        }
    }
    for item in &st.stack {
        if st.objects.try_get(item.source).is_some() {
            let mut t = object_token(st, item.source, me, Zone::Stack, item.controller == me);
            // An ability's source is public as it was, even if it has since
            // moved to a hidden zone.
            if let Some(source) = item.v4.ability_source_contract {
                t[0] = source.card_def + 1;
            }
            out.push(t);
        }
    }
    for &id in &st.exile {
        let o = st.objects.get(id);
        let mut t = if !perfect && st.face_down_hidden_from(id, me) {
            let mut t = [0u16; TOKEN_FIELDS];
            t[1] = place_code(Zone::Exile, o.owner == me);
            t
        } else {
            object_token(st, id, me, Zone::Exile, o.owner == me)
        };
        // Public facts about exiled cards: face down, foretold, castable from
        // foretell now, exiled with a permanent, playable by a permission.
        if let Some(f) = st.face_down(id) {
            let earlier = mtg_kernel::face_down::foretold_before_this_turn(st, id).is_some();
            t[2] = 1 | (f.foretold_turn.is_some() as u16) << 1 | (earlier as u16) << 2 | (f.exiled_with.is_some() as u16) << 3;
        }
        let generation = o.zone_change_count;
        if st.engine.exile_play_permissions.iter().any(|p| p.object == id && p.zone_change_generation == generation) {
            t[2] |= 1 << 4;
        }
        out.push(t);
    }
    // Library cards the viewer knows (a revealed top card, say), with their position.
    for p in [me, opp] {
        let lib = &st.players[p.index()].library;
        for e in st.known_library_cards(me, p) {
            let pos = e.position as usize;
            if lib.get(pos) == Some(&e.object) && st.objects.get(e.object).zone_change_count == e.zone_change_count {
                let mut t = object_token(st, e.object, me, Zone::Library, p == me);
                t[5] = clamp(e.position as i32, 0, 31);
                out.push(t);
            }
        }
        // Known cards on the bottom in an unknown order.
        for g in st.known_library_groups(me, p) {
            for l in &g.cards {
                let mut t = object_token(st, l.object, me, Zone::Library, p == me);
                t[2] = 1;
                t[5] = 31;
                out.push(t);
            }
        }
    }
    // Graveyards last, newest first, so truncation drops the oldest.
    let mut yards: Vec<[u16; TOKEN_FIELDS]> = Vec::new();
    for p in [me, opp] {
        for &id in st.players[p.index()].graveyard.iter().rev() {
            yards.push(object_token(st, id, me, Zone::Graveyard, p == me));
        }
    }
    out.extend(yards);
    out.truncate(MAX_TOKENS);
    out
}

/// Kind, source card, target card, target code (0 none, 1 me, 2 opponent, 3+ place), arg.
pub fn action_fields(f: &ActFeat) -> [u16; ACTION_FIELDS] {
    let src = if f.src == u16::MAX { 0 } else { f.src + 1 };
    let (tcard, tcode) = if f.tgt & (1 << 24) != 0 {
        (0, if f.tgt & 1 == 1 { 1 } else { 2 })
    } else if f.tgt & (1 << 23) != 0 {
        let card = (f.tgt & 0xFFFF) as u16 + 1;
        let mine = (f.tgt >> 16) & 1;
        let zone = (f.tgt >> 17) & 7;
        let tapped = (f.tgt >> 21) & 1;
        (card, 3 + (((zone * 2 + mine) * 2 + tapped) as u16).min(60))
    } else {
        // Small integers (zone codes for casts/activations, 0 for none).
        (0, if f.tgt == 0 { 0 } else { 3 + (f.tgt as u16).min(60) })
    };
    [f.kind as u16, src, tcard, tcode, f.arg.min(63)]
}

pub fn choice_fields(g: &Game) -> Vec<[u16; ACTION_FIELDS]> {
    g.choices().iter().map(|c| action_fields(&c.feat)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Game;
    use mtg_kernel::state::SplitMix64;

    /// A runtime deck padded to 99 cards with Mountains, led by `commander`.
    fn commander_deck(commander: &str, base: &str) -> crate::commander::CommanderDeck {
        use mtg_kernel::card_def::card_id_by_name;
        let mut cards = mtg_kernel::runtime_decks::runtime_deck_by_id(base).unwrap().card_ids.to_vec();
        cards.resize(99, card_id_by_name("Mountain").unwrap());
        crate::commander::CommanderDeck { name: base.into(), commander: card_id_by_name(commander).unwrap(), cards }
    }

    #[test]
    fn known_library_cards_are_observed_only_by_who_knows_them() {
        let mut g = Game::new("Faeries", "Spy", 1).unwrap();
        let (p0, p1) = (PlayerId::P0, PlayerId::P1);
        g.state.reveal_library_top(p0, p1, 1);
        let card = g.state.objects.get(g.state.players[1].library[0]).card_def + 1;
        let shown = |o: &[[u16; TOKEN_FIELDS]], mine| o.iter().any(|t| t[0] == card && t[1] == place_code(Zone::Library, mine) && t[5] == 0);
        assert!(shown(&observe(&g, p0), false));
        assert!(!shown(&observe(&g, p1), true));
    }

    #[test]
    fn fields_stay_in_vocab_and_hidden_cards_do_not_leak() {
        let tv = token_vocab();
        let av = action_vocab();
        let mut rng = SplitMix64::seed(4);
        let (ca, cb) = (commander_deck("Masked Meower", "Rally"), commander_deck("Goblin Tomb Raider", "Burn"));
        for seed in 0..40 {
            let mut g = if seed < 30 { Game::new("Faeries", "Spy", seed).unwrap() } else { Game::new_commander(&ca, &cb, seed) };
            while !g.is_over() {
                let me = g.to_act();
                // Now and then a library's top card goes face down into exile.
                if rng.next_u64() % 6 == 0 {
                    let owner = if rng.next_u64() % 2 == 0 { me } else { me.opponent() };
                    mtg_kernel::face_down::exile_top(&mut g.state, owner, rng.next_u64() % 2 == 0, None);
                }
                let obs = observe(&g, me);
                for t in &obs {
                    for (i, &v) in t.iter().enumerate() {
                        assert!((v as usize) < tv[i], "token field {i} = {v}");
                    }
                }
                for a in choice_fields(&g) {
                    for (i, &v) in a.iter().enumerate() {
                        assert!((v as usize) < av[i], "action field {i} = {v}");
                    }
                }
                // Resampling hidden cards must not change the observation.
                let mut d = g.clone();
                d.determinize(me, rng.next_u64());
                assert_eq!(observe(&d, me), obs);
                let p = observe_with(&d, me, true);
                assert_eq!(p.len(), obs.len());
                let opp_hand = |o: &[[u16; TOKEN_FIELDS]]| o.iter().filter(|t| t[1] == place_code(Zone::Hand, false) && t[0] == 0).count();
                assert_eq!(opp_hand(&p), 0);
                let hidden_exile = |o: &[[u16; TOKEN_FIELDS]]| {
                    o.iter().filter(|t| t[0] == 0 && (t[1] == place_code(Zone::Exile, true) || t[1] == place_code(Zone::Exile, false))).count()
                };
                assert_eq!(hidden_exile(&p), 0);
                assert_eq!(hidden_exile(&obs), g.state.exile.iter().filter(|&&id| g.state.face_down_hidden_from(id, me)).count());
                let n = g.choices().len();
                g.apply((rng.next_u64() % n as u64) as usize);
            }
        }
    }
}
