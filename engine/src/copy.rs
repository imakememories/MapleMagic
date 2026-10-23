//! Copy effects (707). A permanent that becomes a copy of another takes its
//! copiable values: `GameObject::card_def` names the copied card while the
//! effect lasts, so every characteristic read sees the copy, and
//! `GameState::copy_effects` remembers what the object itself is.

use crate::card_def::CARD_DEFS;
use crate::ids::ObjectId;
use crate::state::{CopyEffectV1, GameState, ObjectLinkV4, ObjectStateV4, Zone};

/// An object's copiable values (707.2): its card, plus the exceptions a copy
/// effect on it made (its name and haste), which a copy of it copies too
/// (707.9b).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CopiableValues {
    pub card_def: u16,
    pub name: u16,
    pub haste: bool,
}

fn live_effect(state: &GameState, id: ObjectId) -> Option<&CopyEffectV1> {
    if state.copy_effects.is_empty() {
        return None;
    }
    let generation = state.objects.get(id).zone_change_count;
    state
        .copy_effects
        .iter()
        .rfind(|c| c.object.object == id && c.object.zone_change_count == generation)
}

pub fn copiable_values(state: &GameState, id: ObjectId) -> CopiableValues {
    let card_def = state.objects.get(id).card_def;
    match live_effect(state, id) {
        Some(c) => CopiableValues { card_def, name: c.name, haste: c.haste },
        None => CopiableValues { card_def, name: card_def, haste: false },
    }
}

/// The card whose name `id` has.
pub fn name_def(state: &GameState, id: ObjectId) -> u16 {
    live_effect(state, id).map_or(state.objects.get(id).card_def, |c| c.name)
}

/// Whether a copy effect gave `id` haste.
pub fn has_copy_haste(state: &GameState, id: ObjectId) -> bool {
    live_effect(state, id).is_some_and(|c| c.haste)
}

/// The card a copy was as incarnation `zone_change_count` of `id` left the
/// battlefield.
pub fn last_known_def(state: &GameState, id: ObjectId, zone_change_count: u32) -> Option<u16> {
    state
        .last_known_defs
        .iter()
        .find(|(l, _)| l.object == id && l.zone_change_count == zone_change_count)
        .map(|&(_, def)| def)
}

fn note_history(state: &mut GameState, link: ObjectLinkV4, card_def: u16) {
    if !state.copy_history.contains(&(link, card_def)) {
        state.copy_history.push((link, card_def));
    }
}

/// Sets the characteristics that come from `card_def` and are stored on the
/// object rather than read from the card.
fn set_card(state: &mut GameState, id: ObjectId, card_def: u16, name: u16) {
    let fresh = ObjectStateV4::from_card_def(card_def);
    let o = state.objects.get_mut(id);
    o.card_def = card_def;
    o.name = CARD_DEFS[name as usize].object_name.into();
    o.v4.effective_color_mask = fresh.effective_color_mask;
    o.v4.effective_subtype_ids = fresh.effective_subtype_ids;
    o.v4.ward_generic = fresh.ward_generic;
}

/// Permanent `id` becomes a copy of `values` (layer 1), until end of turn or
/// for as long as it stays on the battlefield. Earlier copy effects stay and
/// apply again once it ends (613.2a: newest last).
pub fn become_copy(state: &mut GameState, id: ObjectId, values: CopiableValues, until_end_of_turn: bool) {
    let o = state.objects.get(id);
    let link = ObjectLinkV4 { object: id, zone_change_count: o.zone_change_count };
    let printed = live_effect(state, id).map_or(o.card_def, |c| c.printed);
    state.copy_effects.push(CopyEffectV1 {
        object: link,
        printed,
        card_def: values.card_def,
        name: values.name,
        haste: values.haste,
        until_end_of_turn,
    });
    note_history(state, link, printed);
    note_history(state, link, values.card_def);
    set_card(state, id, values.card_def, values.name);
}

/// A token created as a copy of `values` (707.2): its own card is the
/// copied one; only exceptions need an effect.
pub fn token_created(state: &mut GameState, token: ObjectId, values: CopiableValues) {
    let link = ObjectLinkV4 { object: token, zone_change_count: state.objects.get(token).zone_change_count };
    if values.name != values.card_def || values.haste {
        state.copy_effects.push(CopyEffectV1 {
            object: link,
            printed: values.card_def,
            card_def: values.card_def,
            name: values.name,
            haste: values.haste,
            until_end_of_turn: false,
        });
        set_card(state, token, values.card_def, values.name);
    }
}

/// 514.2: copy effects lasting until end of turn end.
pub fn end_until_end_of_turn(state: &mut GameState) {
    if !state.copy_effects.iter().any(|c| c.until_end_of_turn) {
        return;
    }
    let ending: Vec<CopyEffectV1> = state.copy_effects.iter().copied().filter(|c| c.until_end_of_turn).collect();
    state.copy_effects.retain(|c| !c.until_end_of_turn);
    for c in ending {
        let id = c.object.object;
        let o = state.objects.get(id);
        if o.zone == Zone::Battlefield && o.zone_change_count == c.object.zone_change_count {
            let (card_def, name) = live_effect(state, id).map_or((c.printed, c.printed), |e| (e.card_def, e.name));
            set_card(state, id, card_def, name);
        }
    }
}

/// A copy leaving the battlefield is its own card again (400.7); what it was
/// stays as last known information.
pub fn leave_battlefield(state: &mut GameState, id: ObjectId) {
    if state.copy_effects.is_empty() {
        return;
    }
    let Some(c) = live_effect(state, id).copied() else {
        return;
    };
    state.copy_effects.retain(|e| e.object != c.object);
    let current = state.objects.get(id).card_def;
    if current != c.printed {
        state.last_known_defs.push((c.object, current));
    }
    set_card(state, id, c.printed, c.printed);
}

/// Drops records of incarnations that are gone, once a turn.
pub fn prune(state: &mut GameState) {
    if state.copy_history.is_empty() && state.last_known_defs.is_empty() {
        return;
    }
    let objects = &state.objects;
    state.copy_history.retain(|(l, _)| objects.get(l.object).zone_change_count == l.zone_change_count);
    state.last_known_defs.clear();
}
