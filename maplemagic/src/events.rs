//! What happened in each step (see `mtg_kernel::watch`), as lines of text for
//! one player: damage and its source, life changes, draws, cards moving
//! between zones, tokens and attacks. Cards the player can't see stay hidden.

use crate::game::Game;
use mtg_kernel::card_def::{CardType, CARD_DEFS};
use mtg_kernel::event::CommittedEvent;
use mtg_kernel::ids::{ObjectId, PlayerId};
use mtg_kernel::state::{GameState, Target, Zone};
use mtg_kernel::watch::StepEvents;
use serde_json::{json, Value};

fn who(p: PlayerId, viewer: PlayerId) -> &'static str {
    if p == viewer {
        "you"
    } else {
        "opponent"
    }
}

/// "You" / "The opponent".
fn subject(p: PlayerId, viewer: PlayerId) -> &'static str {
    if p == viewer {
        "You"
    } else {
        "The opponent"
    }
}

/// "your" / "the opponent's".
fn whose(p: PlayerId, viewer: PlayerId) -> &'static str {
    if p == viewer {
        "your"
    } else {
        "the opponent's"
    }
}

/// "your" / "their", after a subject.
fn their(p: PlayerId, viewer: PlayerId) -> &'static str {
    if p == viewer {
        "your"
    } else {
        "their"
    }
}

fn cap(s: &str) -> String {
    let mut c = s.chars();
    c.next().map_or(String::new(), |f| f.to_uppercase().chain(c).collect())
}

fn name(st: &GameState, id: ObjectId) -> &'static str {
    CARD_DEFS[st.objects.get(id).card_def as usize].name
}

fn zone(z: Zone) -> &'static str {
    match z {
        Zone::Library => "library",
        Zone::Hand => "hand",
        Zone::Battlefield => "battlefield",
        Zone::Graveyard => "graveyard",
        Zone::Stack => "stack",
        Zone::Exile => "exile",
        Zone::Command => "command zone",
    }
}

/// Where a card came from: "the library", "exile"...
fn from(z: Zone) -> &'static str {
    match z {
        Zone::Library => "the library",
        Zone::Hand => "hand",
        Zone::Battlefield => "the battlefield",
        Zone::Graveyard => "the graveyard",
        Zone::Stack => "the stack",
        Zone::Exile => "exile",
        Zone::Command => "the command zone",
    }
}

fn public(z: Zone) -> bool {
    !matches!(z, Zone::Library | Zone::Hand)
}

/// Whether `viewer` may know which card `id` is after it moved `from` -> `to`.
fn visible(st: &GameState, id: ObjectId, from: Zone, to: Zone, viewer: PlayerId) -> bool {
    let o = st.objects.get(id);
    if o.owner == viewer {
        return true;
    }
    let face_down = to == Zone::Exile && st.face_down_hidden_from(id, viewer);
    let known = to == Zone::Hand
        && st.known_hand_cards(viewer, o.owner).iter().any(|e| e.object == id && e.zone_change_count == o.zone_change_count);
    (public(from) || public(to)) && !face_down || known
}

/// Cards in one line, merged with the next line's when they say the same thing.
struct Line {
    who: Option<PlayerId>,
    kind: &'static str,
    /// For merged lines: the text around the cards, and who it's about.
    merge: Option<(&'static str, PlayerId)>,
    /// Card names; None for a card the player can't see.
    cards: Vec<Option<&'static str>>,
    text: String,
}

impl Line {
    fn plain(who: Option<PlayerId>, kind: &'static str, card: Option<&'static str>, text: String) -> Line {
        Line { who, kind, merge: None, cards: card.into_iter().map(Some).collect(), text }
    }

    fn cards(who: PlayerId, kind: &'static str, template: &'static str, card: Option<&'static str>) -> Line {
        Line { who: Some(who), kind, merge: Some((template, who)), cards: vec![card], text: String::new() }
    }
}

/// "Mountain, Forest ×2 and 3 cards".
fn list(cards: &[Option<&str>]) -> String {
    let mut named: Vec<(&str, usize)> = Vec::new();
    for c in cards.iter().flatten() {
        match named.iter_mut().find(|(n, _)| n == c) {
            Some((_, k)) => *k += 1,
            None => named.push((c, 1)),
        }
    }
    let mut parts: Vec<String> = named.into_iter().map(|(n, k)| if k > 1 { format!("{n} ×{k}") } else { n.to_string() }).collect();
    match cards.iter().filter(|c| c.is_none()).count() {
        0 => {}
        1 => parts.push("a card".into()),
        k => parts.push(format!("{k} cards")),
    }
    match parts.len() {
        0 => String::new(),
        1 => parts.pop().unwrap(),
        _ => {
            let last = parts.pop().unwrap();
            format!("{} and {last}", parts.join(", "))
        }
    }
}

/// A card moving between zones, or None when another line already says it.
fn moved(st: &GameState, id: ObjectId, source: Zone, to: Zone, controller: PlayerId, viewer: PlayerId) -> Option<Line> {
    let o = st.objects.get(id);
    let def = &CARD_DEFS[o.card_def as usize];
    let seen = visible(st, id, source, to, viewer);
    let card = seen.then_some(def.name);
    let owner = o.owner;
    // Permanents are their controller's; cards elsewhere their owner's.
    let p = if source == Zone::Battlefield { controller } else { owner };
    let the = |n: Option<&str>| match n {
        Some(n) => format!("{} {n}", whose(p, viewer)),
        None => format!("a card of {}", whose(p, viewer)),
    };
    let text = match (source, to) {
        // The land drop and the cast are the choices' own lines.
        (Zone::Hand, Zone::Battlefield) if def.is_land => return None,
        (_, Zone::Stack) => return None,
        (Zone::Library, Zone::Hand) => return Some(Line::cards(owner, "zone", "{p} put {cards} from {their} library into {their} hand", card)),
        (Zone::Hand, Zone::Library) => return Some(Line::cards(owner, "zone", "{p} put {cards} from {their} hand into {their} library", card)),
        (Zone::Hand, Zone::Graveyard) => return Some(Line::cards(owner, "zone", "{p} discarded {cards}", card)),
        (Zone::Library, Zone::Graveyard) => return Some(Line::cards(owner, "zone", "{p} milled {cards}", card)),
        (Zone::Battlefield, Zone::Graveyard) if def.has_type(CardType::Creature) => format!("{} died", the(card)),
        (Zone::Battlefield, Zone::Graveyard) => format!("{} was put into the graveyard", the(card)),
        (Zone::Battlefield, Zone::Hand) => format!("{} returned to {} hand", the(card), whose(owner, viewer)),
        (Zone::Battlefield, Zone::Exile) => format!("{} was exiled", the(card)),
        (Zone::Battlefield, Zone::Library) => format!("{} was put into {} library", the(card), whose(owner, viewer)),
        (Zone::Stack, Zone::Battlefield) => format!("{} entered the battlefield", the(card)),
        (Zone::Stack, Zone::Graveyard) => format!("{} went to the graveyard", the(card)),
        (_, Zone::Battlefield) => format!("{} entered the battlefield from {}", the(card), from(source)),
        (_, Zone::Command) => format!("{} went to the command zone", the(card)),
        (Zone::Library, Zone::Exile) if card.is_none() => format!("A card from {} library was exiled face down", whose(owner, viewer)),
        (Zone::Hand, Zone::Exile) if card.is_none() => format!("A card from {} hand was exiled face down", whose(owner, viewer)),
        (_, Zone::Exile) => format!("{} was exiled from {}", the(card), from(source)),
        _ => format!("{} moved from {} to the {}", the(card), from(source), zone(to)),
    };
    Some(Line::plain(Some(p), "zone", card, cap(&text)))
}

fn finish(line: Line, viewer: PlayerId, after: Option<[i32; 2]>, before: Option<[i32; 2]>) -> Value {
    let text = match line.merge {
        Some((template, p)) => template
            .replace("{p}", subject(p, viewer))
            .replace("{their}", their(p, viewer))
            .replace("{cards}", &list(&line.cards)),
        None => line.text,
    };
    let mut v = json!({ "text": text, "kind": line.kind, "who": line.who.map_or("", |p| who(p, viewer)) });
    let cards: Vec<&str> = line.cards.iter().flatten().copied().collect();
    if !cards.is_empty() {
        v["cards"] = cards.into();
    }
    if let (Some(b), Some(a)) = (before, after) {
        let p = line.who.expect("life lines name a player").index();
        v["life"] = json!([b[p], a[p]]);
    }
    v
}

/// One step's lines for `viewer`.
fn lines(st: &GameState, s: &StepEvents, viewer: PlayerId) -> Vec<Value> {
    // Each player's life after each event, from the step's end backwards.
    let mut after = vec![s.life; s.events.len()];
    let mut life = s.life;
    for (i, e) in s.events.iter().enumerate().rev() {
        after[i] = life;
        match *e {
            CommittedEvent::Damage { target: Target::Player(p), amount, .. } | CommittedEvent::LifeLoss { player: p, amount } => {
                life[p.index()] += amount
            }
            CommittedEvent::LifeGain { player, amount } => life[player.index()] -= amount,
            _ => {}
        }
    }
    let mut out: Vec<(Line, Option<[i32; 2]>, Option<[i32; 2]>)> = Vec::new();
    let mut sacrificed: Vec<ObjectId> = Vec::new();
    for (i, e) in s.events.iter().enumerate() {
        let before = if i == 0 { life } else { after[i - 1] };
        let line = match *e {
            CommittedEvent::Damage { amount: ..=0, .. } => continue,
            CommittedEvent::Damage { source, target, amount } => {
                let src = name(st, source);
                let by = st.objects.get(source).controller;
                let to = match target {
                    Target::Player(p) => (if p == viewer { "you" } else { "the opponent" }).to_string(),
                    Target::Object(id) => format!("{} {}", whose(st.objects.get(id).controller, viewer), name(st, id)),
                    Target::Ability(_) => continue,
                };
                let hit = match target {
                    Target::Player(p) => Some(p),
                    _ => None,
                };
                let line = Line::plain(Some(hit.unwrap_or(by)), "damage", Some(src), format!("{} {src} dealt {amount} damage to {to}", cap(whose(by, viewer))));
                if hit.is_some() {
                    out.push((line, Some(after[i]), Some(before)));
                    continue;
                }
                line
            }
            CommittedEvent::LifeLoss { player, amount } => {
                out.push((Line::plain(Some(player), "life", None, format!("{} lost {amount} life", subject(player, viewer))), Some(after[i]), Some(before)));
                continue;
            }
            CommittedEvent::LifeGain { player, amount } => {
                out.push((Line::plain(Some(player), "life", None, format!("{} gained {amount} life", subject(player, viewer))), Some(after[i]), Some(before)));
                continue;
            }
            CommittedEvent::Draw { player, object: Some(id) } => {
                let known = player == viewer
                    || st.known_hand_cards(viewer, player).iter().any(|e| e.object == id && e.zone_change_count == st.objects.get(id).zone_change_count);
                Line::cards(player, "draw", "{p} drew {cards}", known.then(|| name(st, id)))
            }
            CommittedEvent::Draw { player, object: None } => {
                Line::plain(Some(player), "draw", None, format!("{} had to draw from an empty library", subject(player, viewer)))
            }
            CommittedEvent::ZoneChange { object, from, to, controller_before } => {
                if from == Zone::Battlefield && sacrificed.contains(&object) {
                    continue;
                }
                match moved(st, object, from, to, controller_before, viewer) {
                    Some(l) => l,
                    None => continue,
                }
            }
            CommittedEvent::Sacrificed { object, controller_before, .. } => {
                sacrificed.push(object);
                Line::cards(controller_before, "zone", "{p} sacrificed {cards}", Some(name(st, object)))
            }
            CommittedEvent::CreateToken { token_def, controller, .. } => {
                Line::cards(controller, "token", "{p} created {cards}", Some(CARD_DEFS[token_def as usize].name))
            }
            CommittedEvent::DeclaredAttacker { source, controller, .. } => {
                Line::cards(controller, "attack", "{p} attacked with {cards}", Some(name(st, source)))
            }
            CommittedEvent::SagaChapter { source, controller, chapter, .. } => {
                Line::plain(Some(controller), "zone", Some(name(st, source)), format!("{} {}: chapter {chapter}", cap(whose(controller, viewer)), name(st, source)))
            }
            _ => continue,
        };
        match out.last_mut() {
            Some((last, None, None)) if last.merge.is_some() && last.merge == line.merge => last.cards.extend(line.cards),
            _ => out.push((line, None, None)),
        }
    }
    out.into_iter().map(|(l, a, b)| finish(l, viewer, a, b)).collect()
}

/// Each step's lines for `viewer`, with whose turn it was and the life totals when it ended.
pub fn describe(g: &Game, steps: &[StepEvents], viewer: PlayerId) -> Value {
    let st = &g.state;
    let out: Vec<Value> = steps
        .iter()
        .map(|s| {
            json!({
                "turn": s.turn,
                "step": format!("{:?}", s.step),
                "active": who(s.active, viewer),
                "life": { "you": s.life[viewer.index()], "opponent": s.life[viewer.opponent().index()] },
                "lines": lines(st, s, viewer),
            })
        })
        .filter(|v| !v["lines"].as_array().unwrap().is_empty())
        .collect();
    out.into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commander::CommanderDeck;
    use mtg_kernel::state::SplitMix64;
    use mtg_kernel::watch;

    #[test]
    fn steps_keep_every_event_and_life_adds_up() {
        let deck = CommanderDeck::named("Ognis").expect("the Ognis deck is complete");
        let mut rng = SplitMix64::seed(3);
        let mut lines_seen = 0;
        for seed in 0..20 {
            let mut g = Game::new_commander(&deck, &deck, seed);
            let mut cursor = g.state.next_history_index().unwrap();
            let mut life = [g.state.players[0].life, g.state.players[1].life];
            while !g.is_over() {
                let n = g.choices().len();
                watch::start(cursor);
                g.apply((rng.next_u64() % n as u64) as usize);
                let (steps, next) = watch::stop(&g.state);
                // Nothing is lost, though Untap clears the history.
                let events: usize = steps.iter().map(|s| s.events.len()).sum();
                assert_eq!(events as u32, g.state.next_history_index().unwrap() - cursor);
                assert_eq!(next, g.state.next_history_index().unwrap());
                cursor = next;
                for viewer in [PlayerId::P0, PlayerId::P1] {
                    let mut at = life;
                    for v in describe(&g, &steps, viewer).as_array().unwrap() {
                        for l in v["lines"].as_array().unwrap() {
                            assert!(!l["text"].as_str().unwrap().is_empty());
                            lines_seen += 1;
                            if let Some(change) = l["life"].as_array() {
                                let p = if l["who"] == "you" { viewer } else { viewer.opponent() };
                                assert_eq!(change[0].as_i64().unwrap() as i32, at[p.index()], "{l}");
                                at[p.index()] = change[1].as_i64().unwrap() as i32;
                            }
                            // The other player's draws stay hidden.
                            if l["kind"] == "draw" && l["who"] == "opponent" {
                                assert!(l["cards"].is_null() || !g.state.known_hand_cards(viewer, viewer.opponent()).is_empty(), "{l}");
                            }
                        }
                        let end = |p: PlayerId| v["life"][if p == viewer { "you" } else { "opponent" }].as_i64().unwrap() as i32;
                        at = [PlayerId::P0, PlayerId::P1].map(|p| {
                            // Life paid as a cost has no event; nothing else should differ.
                            assert!(at[p.index()] >= end(p), "{v}");
                            end(p)
                        });
                    }
                }
                life = [g.state.players[0].life, g.state.players[1].life];
            }
        }
        assert!(lines_seen > 1000, "{lines_seen} lines");
    }
}
