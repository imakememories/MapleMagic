//! The board as data for a person's page: what one player sees, zone by zone,
//! and which of the visible cards, players and abilities each choice names, so
//! the page can put the choices on them. Keys: `o<object id>` for a card,
//! `a<stack item id>` for an ability on the stack, `you` and `opponent`.

use crate::game::{obj_role, obj_sig, Game, Kind};
use mtg_kernel::card_def::{CardType, CARD_DEFS};
use mtg_kernel::engine::{effective_power, effective_toughness, object_has_type};
use mtg_kernel::ids::{ObjectId, PlayerId};
use mtg_kernel::state::{GameState, Step, StackItemKind, Target, Zone};
use serde_json::{json, Value};

fn who(p: PlayerId, viewer: PlayerId) -> &'static str {
    if p == viewer {
        "you"
    } else {
        "opponent"
    }
}

/// A card `viewer` can see, with its state on the battlefield.
fn card(st: &GameState, id: ObjectId) -> Value {
    let o = st.objects.get(id);
    let def = &CARD_DEFS[o.card_def as usize];
    let mut v = json!({ "key": format!("o{}", id.0), "name": def.name, "land": def.is_land, "token": def.is_token });
    if o.zone != Zone::Battlefield {
        return v;
    }
    v["tapped"] = o.tapped.into();
    if object_has_type(st, id, CardType::Creature) {
        v["power"] = effective_power(st, id).into();
        v["toughness"] = effective_toughness(st, id).into();
    }
    let c = &o.counters;
    let counters: Vec<Value> = [(c.plus1_plus1, "+1/+1"), (c.minus1_minus1, "-1/-1"), (c.minus0_minus1, "-0/-1"), (c.charge, "charge"), (c.lore, "lore"), (c.stun, "stun")]
        .into_iter()
        .filter(|&(n, _)| n > 0)
        .map(|(n, label)| json!([label, n]))
        .collect();
    if !counters.is_empty() {
        v["counters"] = counters.into();
    }
    if o.damage > 0 {
        v["damage"] = o.damage.into();
    }
    let in_combat = matches!(st.step, Step::DeclareAttackers | Step::DeclareBlockers | Step::CombatDamage);
    let combat = &st.engine.combat;
    if in_combat && combat.attackers.contains(&id) {
        v["attacking"] = true.into();
    }
    if in_combat {
        if let Some((a, _)) = combat.blocked_by.iter().find(|(_, bs)| bs.contains(&id)) {
            v["blocking"] = format!("o{}", a.0).into();
        }
    }
    v
}

/// One player's side as `viewer` sees it; the cards shown go into `seen`.
fn side(g: &Game, p: PlayerId, viewer: PlayerId, seen: &mut Vec<ObjectId>) -> Value {
    let st = &g.state;
    let ps = &st.players[p.index()];
    let mut show = |id: ObjectId| {
        seen.push(id);
        card(st, id)
    };
    let known = |id: ObjectId| {
        p == viewer || st.known_hand_cards(viewer, p).iter().any(|e| e.object == id && e.zone_change_count == st.objects.get(id).zone_change_count)
    };
    // The other player's known cards first, so the order of the hidden ones tells nothing.
    let mut hand: Vec<Value> = ps.hand.iter().copied().filter(|&id| known(id)).map(&mut show).collect();
    hand.extend((hand.len()..ps.hand.len()).map(|_| json!({ "hidden": true })));
    let battlefield: Vec<Value> = ps.battlefield.iter().copied().map(&mut show).collect();
    let graveyard: Vec<Value> = ps.graveyard.iter().rev().copied().map(&mut show).collect();
    let exile: Vec<Value> = st
        .exile
        .iter()
        .copied()
        .filter(|&id| st.objects.get(id).owner == p)
        .map(|id| if st.face_down_hidden_from(id, viewer) { json!({ "hidden": true }) } else { show(id) })
        .collect();
    let mut v = json!({
        "life": ps.life,
        "library": ps.library.len(),
        "hand": hand,
        "battlefield": battlefield,
        "graveyard": graveyard,
        "exile": exile,
        "mana_pool": ps.mana_pool,
        "mulligans": g.mulligans_taken(p),
        "active": st.active_player == p,
    });
    if let Some(c) = &st.commander {
        let cmd = c.commanders[p.index()];
        let o = st.objects.get(cmd);
        v["commander"] = json!({
            "name": CARD_DEFS[o.card_def as usize].name,
            "casts": c.casts[p.index()],
            "tax": mtg_kernel::commander::tax(st, cmd),
            "damage_taken": c.damage[p.index()][p.opponent().index()],
            "in_command_zone": o.zone == Zone::Command,
        });
        if o.zone == Zone::Command {
            v["command_zone"] = json!([show(cmd)]);
        }
    }
    v
}

/// The stack, top first.
fn stack(st: &GameState, viewer: PlayerId, seen: &mut Vec<ObjectId>) -> Vec<Value> {
    st.stack
        .iter()
        .rev()
        .map(|item| {
            let def = item.v4.ability_source_contract.map_or_else(|| st.objects.get(item.source).card_def, |c| c.card_def);
            let spell = item.kind == StackItemKind::Spell;
            let key = if spell {
                seen.push(item.source);
                format!("o{}", item.source.0)
            } else {
                format!("a{}", item.v4.stack_item_id.0)
            };
            json!({ "key": key, "name": CARD_DEFS[def as usize].name, "ability": !spell, "controller": who(item.controller, viewer) })
        })
        .collect()
}

/// The keys of what choice `i` names among the things shown.
fn anchors(g: &Game, i: usize, viewer: PlayerId, seen: &[ObjectId]) -> Vec<String> {
    let st = &g.state;
    let mut out = Vec::new();
    for (r, alike) in g.choice_refs(i) {
        match r {
            Target::Object(id) if seen.contains(&id) => {
                let look = |x: ObjectId| (obj_sig(st, x, viewer), obj_role(st, x));
                let same = |x: &&ObjectId| **x == id || (alike && look(**x) == look(id));
                out.extend(seen.iter().filter(same).map(|x| format!("o{}", x.0)));
            }
            Target::Object(_) => {}
            Target::Player(p) => out.push(who(p, viewer).to_string()),
            Target::Ability(a) => out.push(format!("a{a}")),
        }
    }
    out.sort();
    out.dedup();
    out
}

fn key(id: ObjectId) -> String {
    format!("o{}", id.0)
}

/// What kind of choice each is, for the page: `Kind`'s names, with "Decline"
/// for not casting from exile (a pass that isn't priority's).
pub fn kinds(g: &Game) -> Vec<String> {
    g.choices()
        .iter()
        .map(|c| if c.feat.kind == Kind::Pass && c.feat.src != u16::MAX { "Decline".into() } else { format!("{:?}", c.feat.kind) })
        .collect()
}

/// The board as `viewer` sees it, with each choice's keys and kind while
/// `viewer` is to act, and while declaring attackers or blockers, the
/// creatures still to be asked.
pub fn board(g: &Game, viewer: PlayerId) -> Value {
    let st = &g.state;
    let mut seen = Vec::new();
    let you = side(g, viewer, viewer, &mut seen);
    let opponent = side(g, viewer.opponent(), viewer, &mut seen);
    let stack = stack(st, viewer, &mut seen);
    let acting = !g.is_over() && g.to_act() == viewer;
    let anchors: Vec<Vec<String>> = if acting { (0..g.choices().len()).map(|i| anchors(g, i, viewer, &seen)).collect() } else { Vec::new() };
    let mut v = json!({
        "turn": st.turn,
        "step": format!("{:?}", st.step),
        "active": who(st.active_player, viewer),
        "mulligans": g.mulligan_declarations(viewer).is_some(),
        "you": you,
        "opponent": opponent,
        "stack": stack,
        "anchors": anchors,
        "kinds": if acting { kinds(g) } else { Vec::new() },
    });
    if !acting {
        return v;
    }
    if let Some((asking, chosen)) = g.attack_plan() {
        v["attack"] = json!({ "asking": asking.iter().map(|&id| key(id)).collect::<Vec<_>>(), "chosen": chosen.iter().map(|&id| key(id)).collect::<Vec<_>>() });
    }
    if let Some((asking, pairs)) = g.block_plan() {
        let asking: Vec<Value> = asking.iter().map(|(b, can)| json!({ "key": key(*b), "attackers": can.iter().map(|&a| key(a)).collect::<Vec<_>>() })).collect();
        let pairs: Vec<Value> = pairs.iter().map(|&(b, a)| json!([key(b), key(a)])).collect();
        v["block"] = json!({ "asking": asking, "pairs": pairs });
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commander::CommanderDeck;
    use crate::play::describe;
    use mtg_kernel::state::SplitMix64;

    fn keys(v: &Value, out: &mut Vec<String>) {
        match v {
            Value::Object(m) => {
                if let Some(Value::String(k)) = m.get("key") {
                    out.push(k.clone());
                }
                m.values().for_each(|x| keys(x, out));
            }
            Value::Array(a) => a.iter().for_each(|x| keys(x, out)),
            _ => {}
        }
    }

    #[test]
    fn boards_hide_the_other_players_cards_and_anchor_on_what_they_show() {
        let deck = CommanderDeck::named("Ognis").expect("the Ognis deck is complete");
        let mut rng = SplitMix64::seed(9);
        let (mut anchored, mut decisions) = (0, 0);
        for seed in 0..20 {
            let mut g = Game::new_commander(&deck, &deck, seed);
            while !g.is_over() {
                let (me, opp) = (g.to_act(), g.to_act().opponent());
                let theirs = board(&g, opp);
                let hidden = theirs["opponent"]["hand"].as_array().unwrap().iter().filter(|c| c["hidden"] == true).count();
                let zcc = |id: ObjectId| g.state.objects.get(id).zone_change_count;
                let known = g.state.players[me.index()]
                    .hand
                    .iter()
                    .filter(|&&id| g.state.known_hand_cards(opp, me).iter().any(|e| e.object == id && e.zone_change_count == zcc(id)))
                    .count();
                assert_eq!(hidden, g.state.players[me.index()].hand.len() - known);
                assert!(theirs["anchors"].as_array().unwrap().is_empty());
                let mine = board(&g, me);
                let mut shown = vec!["you".to_string(), "opponent".to_string()];
                keys(&mine, &mut shown);
                let anchors = mine["anchors"].as_array().unwrap();
                assert_eq!(anchors.len(), g.choices().len());
                for (i, a) in anchors.iter().enumerate() {
                    for k in a.as_array().unwrap() {
                        assert!(shown.contains(&k.as_str().unwrap().to_string()), "{k} for {:?} not on the board", describe(&g, i, me));
                    }
                    anchored += !a.as_array().unwrap().is_empty() as u32;
                }
                decisions += g.choices().len() as u32;
                let n = g.choices().len();
                g.apply((rng.next_u64() % n as u64) as usize);
            }
        }
        // Most choices name a card: land drops, casts, attacks, blocks, targets.
        assert!(anchored * 2 > decisions, "{anchored} of {decisions} choices anchored");
    }
}
