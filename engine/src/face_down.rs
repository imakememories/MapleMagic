//! Face-down exile (406.3): putting cards there and finding them again. Who
//! may see each card is recorded in `GameState::face_down`; observation and
//! determinization code read it through `GameState::face_down_hidden_from`.

use crate::event::{self, ProposedEvent};
use crate::ids::{ObjectId, PlayerId};
use crate::state::{FaceDownV1, GameState, ObjectLinkV4, Zone};

/// Which players know which card `id` is where it is now: everyone in a
/// public zone; in a hand its owner and anyone it was revealed to; in a
/// library whoever knows that position; face down, whoever already did.
pub fn knowers(state: &GameState, id: ObjectId) -> [bool; 2] {
    let o = state.objects.get(id);
    let (owner, generation) = (o.owner, o.zone_change_count);
    let position = state.players[owner.index()].library.iter().position(|&x| x == id);
    [PlayerId::P0, PlayerId::P1].map(|p| match o.zone {
        Zone::Hand => {
            p == owner
                || state.known_hand_cards(p, owner).iter().any(|e| e.object == id && e.zone_change_count == generation)
        }
        Zone::Library => state
            .known_library_cards(p, owner)
            .iter()
            .any(|e| Some(e.position as usize) == position && e.object == id && e.zone_change_count == generation),
        Zone::Exile => !state.face_down_hidden_from(id, p),
        _ => true,
    })
}

/// Exiles `id` face down; `owner_may_look` lets its owner look at it. Anyone
/// who knew the card before still does. Returns false if the card didn't end
/// up in exile.
pub fn exile(
    state: &mut GameState,
    id: ObjectId,
    owner_may_look: bool,
    exiled_with: Option<ObjectLinkV4>,
    foretold_turn: Option<u32>,
) -> bool {
    let mut known_to = knowers(state, id);
    known_to[state.objects.get(id).owner.index()] |= owner_may_look;
    // 903.9a then offers a commander's owner the command zone: everyone
    // learns which card it is.
    if crate::commander::is_commander(state, id) {
        known_to = [true, true];
    }
    event::propose_and_commit(state, ProposedEvent::zone_change(id, Zone::Exile));
    let o = state.objects.get(id);
    if o.zone != Zone::Exile {
        return false;
    }
    let card = ObjectLinkV4 { object: id, zone_change_count: o.zone_change_count };
    let objects = &state.objects;
    state.face_down.retain(|f| {
        let o = objects.get(f.card.object);
        o.zone == Zone::Exile && o.zone_change_count == f.card.zone_change_count
    });
    state.face_down.push(FaceDownV1 { card, known_to, exiled_with, foretold_turn });
    true
}

/// Exiles the top card of `player`'s library face down, if there is one.
pub fn exile_top(
    state: &mut GameState,
    player: PlayerId,
    owner_may_look: bool,
    exiled_with: Option<ObjectLinkV4>,
) -> Option<ObjectId> {
    let top = *state.players[player.index()].library.first()?;
    exile(state, top, owner_may_look, exiled_with, None).then_some(top)
}

/// Counts individual turns (`GameState::turn` counts rounds): foretold
/// cards may be cast on any later turn, including the opponent's.
pub fn turn_index(state: &GameState) -> u32 {
    state.turn * 2 + u32::from(state.active_player != state.starting_player)
}

/// Whether `id` is a card its owner foretold on an earlier turn.
pub fn foretold_before_this_turn(state: &GameState, id: ObjectId) -> Option<u32> {
    let t = state.face_down(id)?.foretold_turn?;
    (t < turn_index(state)).then_some(t)
}

/// The cards still face down in exile that `source` exiled.
pub fn exiled_with(state: &GameState, source: ObjectLinkV4) -> Vec<ObjectId> {
    state
        .face_down
        .iter()
        .filter(|f| f.exiled_with == Some(source))
        .map(|f| f.card.object)
        .filter(|&id| state.face_down(id).is_some())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card_def::card_id_by_name;

    fn game() -> GameState {
        let m = card_id_by_name("Mountain").unwrap();
        let lib = [m, card_id_by_name("Lightning Bolt").unwrap(), m, m];
        GameState::new_from_libraries_with_starting_player_v1(&lib, &lib, |c| crate::card_def::CARD_DEFS[c as usize].name.to_string(), 1, PlayerId::P0)
    }

    #[test]
    fn face_down_cards_are_hidden_until_they_leave_exile() {
        let mut state = game();
        let p0 = PlayerId::P0;
        let unknown = exile_top(&mut state, p0, false, None).unwrap();
        let known = exile_top(&mut state, p0, true, None).unwrap();
        assert!(state.face_down_hidden_from(unknown, p0) && state.face_down_hidden_from(unknown, p0.opponent()));
        assert!(!state.face_down_hidden_from(known, p0) && state.face_down_hidden_from(known, p0.opponent()));
        event::propose_and_commit(&mut state, ProposedEvent::zone_change(unknown, Zone::Hand));
        assert!(state.face_down(unknown).is_none());
        event::propose_and_commit(&mut state, ProposedEvent::zone_change(unknown, Zone::Exile));
        assert!(state.face_down(unknown).is_none(), "a face-up return to exile is a new object");
        exile_top(&mut state, p0, true, None);
        assert_eq!(state.face_down.len(), 2, "stale records are pruned");
    }

    #[test]
    fn cards_exiled_with_a_source_stay_linked_to_that_incarnation() {
        let mut state = game();
        let p1 = PlayerId::P1;
        let link = |o: ObjectId| ObjectLinkV4 { object: o, zone_change_count: 0 };
        let src = state.players[0].library[3];
        let a = exile_top(&mut state, p1, false, Some(link(src))).unwrap();
        let b = exile_top(&mut state, p1, false, Some(link(src))).unwrap();
        exile_top(&mut state, p1, false, None).unwrap();
        assert_eq!(exiled_with(&state, link(src)), vec![a, b]);
        event::propose_and_commit(&mut state, ProposedEvent::zone_change(a, Zone::Hand));
        assert_eq!(exiled_with(&state, link(src)), vec![b]);
        assert!(exiled_with(&state, ObjectLinkV4 { object: src, zone_change_count: 1 }).is_empty());
    }

    #[test]
    fn players_who_knew_a_card_still_know_it_face_down() {
        // 406.3 stops players looking, not remembering.
        let mut state = game();
        let (p0, p1) = (PlayerId::P0, PlayerId::P1);
        state.reveal_library_top(p1, p0, 1);
        let top = exile_top(&mut state, p0, false, None).unwrap();
        assert!(!state.face_down_hidden_from(top, p1) && state.face_down_hidden_from(top, p0));

        let card = state.players[0].library[0];
        event::propose_and_commit(&mut state, ProposedEvent::zone_change(card, Zone::Hand));
        state.reveal_hand_card(p1, p0, card).unwrap();
        assert!(exile(&mut state, card, true, None, Some(0)));
        assert!(!state.face_down_hidden_from(card, p1), "a revealed card, foretold");

        // Back to hand: known there only to whoever knew it face down.
        event::propose_and_commit(&mut state, ProposedEvent::zone_change(top, Zone::Hand));
        assert!(state.known_hand_cards(p1, p0).iter().any(|e| e.object == top));
        let unseen = exile_top(&mut state, p0, true, None).unwrap();
        event::propose_and_commit(&mut state, ProposedEvent::zone_change(unseen, Zone::Hand));
        assert!(!state.known_hand_cards(p1, p0).iter().any(|e| e.object == unseen));
    }

    #[test]
    fn states_without_face_down_cards_serialize_as_before() {
        let state = game();
        assert!(!serde_json::to_string(&state).unwrap().contains("face_down"));
    }
}
