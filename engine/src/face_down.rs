//! Face-down exile (406.3): putting cards there and finding them again. Who
//! may see each card is recorded in `GameState::face_down`; observation and
//! determinization code read it through `GameState::face_down_hidden_from`.

use crate::event::{self, ProposedEvent};
use crate::ids::{ObjectId, PlayerId};
use crate::state::{FaceDownV1, GameState, ObjectLinkV4, Zone};

/// Exiles `id` face down; `owner_knows` lets its owner look at it.
/// Returns false if the card didn't end up in exile.
pub fn exile(
    state: &mut GameState,
    id: ObjectId,
    owner_knows: bool,
    exiled_with: Option<ObjectLinkV4>,
    foretold_turn: Option<u32>,
) -> bool {
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
    state.face_down.push(FaceDownV1 { card, owner_knows, exiled_with, foretold_turn });
    true
}

/// Exiles the top card of `player`'s library face down, if there is one.
pub fn exile_top(
    state: &mut GameState,
    player: PlayerId,
    owner_knows: bool,
    exiled_with: Option<ObjectLinkV4>,
) -> Option<ObjectId> {
    let top = *state.players[player.index()].library.first()?;
    exile(state, top, owner_knows, exiled_with, None).then_some(top)
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
    fn states_without_face_down_cards_serialize_as_before() {
        let state = game();
        assert!(!serde_json::to_string(&state).unwrap().contains("face_down"));
    }
}
