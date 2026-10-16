//! Text for a person playing a game: the board as one player sees it, and
//! each choice described without what that player can't see.

use crate::game::{Game, Kind};
use mtg_kernel::card_def::{CardType, CARD_DEFS};
use mtg_kernel::engine::{effective_power, effective_toughness, object_has_type};
use mtg_kernel::ids::{ObjectId, PlayerId};
use mtg_kernel::state::{GameState, StackItemKind, Step, Zone};
use std::fmt::Write;

fn card(def: u16) -> &'static str {
    CARD_DEFS.get(def as usize).map_or("?", |d| d.name)
}

fn seat(p: PlayerId, viewer: PlayerId) -> &'static str {
    if p == viewer {
        "You"
    } else {
        "Opponent"
    }
}

/// A permanent with its state: name, P/T, counters, damage, tapped.
fn permanent(st: &GameState, id: ObjectId) -> String {
    let o = st.objects.get(id);
    let mut s = card(o.card_def).to_string();
    if object_has_type(st, id, CardType::Creature) {
        let _ = write!(s, " {}/{}", effective_power(st, id), effective_toughness(st, id));
    }
    let c = &o.counters;
    for (n, label) in [(c.plus1_plus1, "+1/+1"), (c.minus1_minus1, "-1/-1"), (c.charge, "charge"), (c.lore, "lore"), (c.stun, "stun")] {
        if n > 0 {
            let _ = write!(s, " [{n} {label}]");
        }
    }
    if o.damage > 0 {
        let _ = write!(s, " ({} damage)", o.damage);
    }
    if o.tapped {
        s.push_str(" (tapped)");
    }
    let in_combat = matches!(st.step, Step::DeclareAttackers | Step::DeclareBlockers | Step::CombatDamage);
    if in_combat && st.engine.combat.attackers.contains(&id) {
        s.push_str(" (attacking)");
    }
    s
}

fn list(names: impl IntoIterator<Item = String>) -> String {
    let v: Vec<String> = names.into_iter().collect();
    if v.is_empty() {
        "-".into()
    } else {
        v.join(", ")
    }
}

/// The board as `viewer` sees it.
pub fn render(g: &Game, viewer: PlayerId) -> String {
    let st = &g.state;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "Round {}, {} turn, {:?}{}",
        st.turn,
        if st.active_player == viewer { "your" } else { "the opponent's" },
        st.step,
        if g.mulligan_declarations(viewer).is_some() { ", deciding mulligans" } else { "" }
    );
    for p in [viewer.opponent(), viewer] {
        let ps = &st.players[p.index()];
        let mut line = format!("{}: {} life, {} cards in library", seat(p, viewer), ps.life, ps.library.len());
        if let Some(c) = &st.commander {
            let cmd = c.commanders[p.index()];
            let o = st.objects.get(cmd);
            let at = if o.zone == Zone::Command { "in the command zone" } else { "out" };
            let _ = write!(line, ", commander {} {at} (cast {} times)", card(o.card_def), c.casts[p.index()]);
            let taken = c.damage[p.index()][p.opponent().index()];
            if taken > 0 {
                let _ = write!(line, ", {taken} commander damage taken");
            }
        }
        if g.mulligans_taken(p) > 0 {
            let _ = write!(line, ", {} mulligans", g.mulligans_taken(p));
        }
        let _ = writeln!(out, "{line}");
        let hand = if p == viewer {
            list(ps.hand.iter().map(|&id| card(st.objects.get(id).card_def).to_string()))
        } else {
            let zcc = |id: ObjectId| st.objects.get(id).zone_change_count;
            let known: Vec<String> = ps
                .hand
                .iter()
                .filter(|&&id| st.known_hand_cards(viewer, p).iter().any(|e| e.object == id && e.zone_change_count == zcc(id)))
                .map(|&id| card(st.objects.get(id).card_def).to_string())
                .collect();
            let hidden = ps.hand.len() - known.len();
            if known.is_empty() {
                format!("{hidden} cards")
            } else {
                format!("{hidden} unknown cards and {}", known.join(", "))
            }
        };
        let _ = writeln!(out, "  hand: {hand}");
        let _ = writeln!(out, "  battlefield: {}", list(ps.battlefield.iter().map(|&id| permanent(st, id))));
        let _ = writeln!(out, "  graveyard: {}", list(ps.graveyard.iter().rev().map(|&id| card(st.objects.get(id).card_def).to_string())));
        let exiled = st.exile.iter().filter(|&&id| st.objects.get(id).owner == p).map(|&id| {
            if st.face_down_hidden_from(id, viewer) {
                "a face-down card".to_string()
            } else {
                card(st.objects.get(id).card_def).to_string()
            }
        });
        let _ = writeln!(out, "  exile: {}", list(exiled));
    }
    if !st.stack.is_empty() {
        let items = st.stack.iter().rev().map(|item| {
            let def = item.v4.ability_source_contract.map_or_else(|| st.objects.get(item.source).card_def, |c| c.card_def);
            let what = if item.kind == StackItemKind::Spell { card(def).to_string() } else { format!("ability of {}", card(def)) };
            format!("{what} ({})", seat(item.controller, viewer).to_lowercase())
        });
        let _ = writeln!(out, "Stack (top first): {}", list(items));
    }
    let pool: u32 = st.players[viewer.index()].mana_pool.iter().map(|&m| u32::from(m)).sum();
    if pool > 0 {
        let _ = writeln!(out, "Your mana pool: {:?}", st.players[viewer.index()].mana_pool);
    }
    out
}

/// A target or object signature (see `game::obj_sig`, `player_sig`,
/// `ability_sig`) as text, for a choice made by `actor` and read by `viewer`.
fn target(tgt: u32, actor: PlayerId, viewer: PlayerId, hidden: bool) -> String {
    if tgt & (1 << 24) != 0 {
        let p = if tgt & 1 == 1 { actor } else { actor.opponent() };
        return if p == viewer { "you".into() } else { "the opponent".into() };
    }
    if tgt & (1 << 22) != 0 && tgt & (1 << 23) == 0 {
        return format!("an ability of {}", card((tgt & 0xFFFF) as u16));
    }
    if tgt & (1 << 23) == 0 {
        return String::new();
    }
    let mine = (tgt >> 16) & 1 == 1;
    let owner = if mine { actor } else { actor.opponent() };
    let zone = (tgt >> 17) & 7;
    // Library (0) and hand (1) cards are hidden from everyone but their owner.
    if hidden && zone <= 1 && owner != viewer {
        return "a hidden card".into();
    }
    let place = match zone {
        0 => " in a library",
        1 => " in hand",
        3 => " in a graveyard",
        4 => " on the stack",
        5 => " in exile",
        _ => "",
    };
    let whose = if owner == viewer { "your" } else { "the opponent's" };
    format!("{whose} {}{place}", card((tgt & 0xFFFF) as u16))
}

/// P/T from a choice's `role` (see `game::obj_role`), if it has one.
fn size(role: u32) -> String {
    let (p, t) = ((role >> 7) & 63, (role >> 13) & 63);
    if p == 0 && t == 0 {
        String::new()
    } else {
        format!(" ({}/{})", p as i32 - 1, t as i32 - 1)
    }
}

/// Choice `i` of the player to act, as `viewer` may read it.
pub fn describe(g: &Game, i: usize, viewer: PlayerId) -> String {
    let actor = g.to_act();
    let f = g.choices()[i].feat;
    let hidden = actor != viewer;
    let src = if f.src == u16::MAX { "" } else { card(f.src) };
    let secret = |name: &str| if hidden { "a card".to_string() } else { name.to_string() };
    let tgt = target(f.tgt, actor, viewer, hidden);
    let zone = |code: u32| match code {
        1 => "",
        3 => " from the graveyard",
        5 => " from exile",
        6 => " from the command zone",
        _ => "",
    };
    match f.kind {
        Kind::Pass if f.src == u16::MAX => "Pass".into(),
        Kind::Pass => format!("Don't cast anything with {src}"),
        Kind::PlayLand => format!("Play {src}"),
        Kind::Cast => format!("Cast {src}{}", zone(f.tgt)),
        Kind::Activate => format!("Activate {src}{} (ability {})", zone(f.tgt), f.arg + 1),
        Kind::ManaAbility => {
            let color = ["W", "U", "B", "R", "G", "C"].get(f.arg as usize).copied().unwrap_or("?");
            if tgt.is_empty() { format!("Tap {src} for {color}") } else { format!("Tap {src} for {color}, using {tgt}") }
        }
        Kind::Plot => format!("Plot {src}"),
        Kind::Target => format!("{src}: target {tgt}{}", size(f.role)),
        Kind::FinishTargets if f.arg == 3 => format!("{src}: don't copy anything"),
        Kind::FinishTargets => format!("{src}: done choosing"),
        Kind::CostTarget => format!("{src}: pay with {tgt}{}", size(f.role)),
        Kind::CastMode => {
            let how = match f.arg {
                0 => "normally",
                1 => "for its alternative cost",
                2 => "with flashback",
                3 => "with escape",
                4 => "with retrace",
                _ => "another way",
            };
            format!("Cast {src} {how}")
        }
        Kind::Kicker => format!("{src}: {}", if f.arg == 1 { "kick it" } else { "don't kick it" }),
        Kind::SpellMode => format!("{src}: mode {}", f.arg + 1),
        Kind::EffectOption => format!("{src}: option {}", f.arg + 1),
        Kind::EffectTarget => match f.arg {
            1 => format!("Keep {tgt} (legend rule)"),
            3 => format!("{src}: copy {tgt}"),
            _ => format!("{src}: choose {tgt}{}", size(f.role)),
        },
        Kind::OptionalCost => ["Don't pay the optional cost", "Pay by discarding a card", "Pay by sacrificing a land"]
            .get(f.arg as usize)
            .copied()
            .unwrap_or("Optional cost")
            .into(),
        Kind::YesNo => {
            let yes = f.arg == 1;
            let what = match f.tgt {
                1 => "pay for the copy",
                2 => "choose new targets for the copy",
                3 => "cast it with madness",
                12 => "give it a +1/+1 counter (riot; no means haste)",
                13 => "cast it with cascade",
                t if (5..12).contains(&t) => "move the commander to the command zone",
                _ => "use the effect",
            };
            format!("{src}: {} {what}", if yes { "yes," } else { "no, don't" })
        }
        Kind::Discard => format!("Discard {src}"),
        Kind::Attack => format!("Attack with {src}{}", size(f.role)),
        Kind::NoAttack => format!("Don't attack with {src}{}", size(f.role)),
        Kind::Block => format!("Block {tgt}{} with {src}", size(f.role)),
        Kind::NoBlock => format!("Don't block with {src}"),
        Kind::OrderTrigger => format!("Put {src}'s trigger on the stack next"),
        Kind::Foretell => format!("Foretell {}", secret(src)),
        Kind::Mulligan => if f.arg == 1 { "Mulligan".into() } else { "Keep this hand".into() },
        Kind::Bottom => format!("Put {} on the bottom of the library", secret(src)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commander::CommanderDeck;
    use mtg_kernel::state::SplitMix64;

    #[test]
    fn texts_hide_the_other_players_cards() {
        let deck = CommanderDeck::named("Ognis").expect("the Ognis deck is complete");
        let mut rng = SplitMix64::seed(5);
        for seed in 0..20 {
            let mut g = Game::new_commander(&deck, &deck, seed);
            while !g.is_over() {
                let (me, opp) = (g.to_act(), g.to_act().opponent());
                let board = render(&g, opp);
                if g.state.known_hand_cards(opp, me).is_empty() {
                    let lines: Vec<&str> = board.lines().collect();
                    let at = lines.iter().position(|l| l.starts_with("Opponent:")).unwrap();
                    assert_eq!(lines[at + 1], format!("  hand: {} cards", g.state.players[me.index()].hand.len()), "{board}");
                }
                for i in 0..g.choices().len() {
                    let text = describe(&g, i, opp);
                    if g.choices()[i].feat.kind == Kind::Bottom {
                        assert_eq!(text, "Put a card on the bottom of the library");
                    }
                    assert!(!describe(&g, i, me).is_empty());
                }
                let n = g.choices().len();
                g.apply((rng.next_u64() % n as u64) as usize);
            }
        }
    }
}
