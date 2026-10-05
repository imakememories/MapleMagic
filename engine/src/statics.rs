//! Static abilities of permanents that affect other objects (604.1), such as
//! "Creatures you control have menace." Looked up by card id in a table built
//! once, so a query costs one pass over a battlefield.

use crate::card_def::{CardDef, CardType, Keywords, Subtype, CARD_DEFS};
use crate::ids::{ObjectId, PlayerId};
use crate::mana::{Cost, ManaColor};
use crate::state::{GameState, Zone};
use std::sync::LazyLock;

/// Keywords a permanent gives the creatures its controller controls.
fn creature_keywords_for(name: &str) -> Keywords {
    match name {
        "Goblin War Drums" => Keywords::MENACE,
        _ => Keywords::NONE,
    }
}

static GRANTS: LazyLock<Vec<Keywords>> =
    LazyLock::new(|| CARD_DEFS.iter().map(|def| creature_keywords_for(def.name)).collect());

/// Every keyword some card grants: other queries skip the battlefield scan.
static ANY_GRANT: LazyLock<Keywords> = LazyLock::new(|| Keywords(GRANTS.iter().fold(0, |m, k| m | k.0)));

/// Whether another permanent's static ability gives `id` keyword `kw`.
pub fn grants_keyword(state: &GameState, id: ObjectId, kw: Keywords) -> bool {
    if ANY_GRANT.0 & kw.0 == 0 {
        return false;
    }
    let o = state.objects.get(id);
    o.zone == Zone::Battlefield
        && state.players[o.controller.index()]
            .battlefield
            .iter()
            .any(|&source| GRANTS[state.objects.get(source).card_def as usize].0 & kw.0 != 0)
        && crate::engine::object_has_type(state, id, CardType::Creature)
}

/// Spells of this color its controller casts cost {1} less.
fn spell_reduction_color_for(name: &str) -> Option<ManaColor> {
    match name {
        "Ruby Medallion" => Some(ManaColor::R),
        _ => None,
    }
}

static SPELL_REDUCERS: LazyLock<Vec<Option<ManaColor>>> =
    LazyLock::new(|| CARD_DEFS.iter().map(|def| spell_reduction_color_for(def.name)).collect());

/// Generic mana `player`'s spell `def` costs less from permanents it controls.
pub fn spell_cost_reduction(state: &GameState, player: PlayerId, def: &CardDef) -> u8 {
    state.players[player.index()]
        .battlefield
        .iter()
        .filter(|&&id| SPELL_REDUCERS[state.objects.get(id).card_def as usize].is_some_and(|c| def.colors.contains(&c)))
        .count()
        .min(u8::MAX as usize) as u8
}

/// 601.2f: a reduction comes off the total cost's generic mana; colored mana
/// is never reduced. What the printed generic can't absorb comes off X, see
/// [`x_discount`].
pub fn reduce_spell_cost(cost: Cost, reduction: u8) -> Cost {
    Cost { generic: cost.generic.saturating_sub(reduction), ..cost }
}

/// The part of `reduction` left for X (one {X}) after `cost`'s generic mana.
pub fn x_discount(cost: Cost, reduction: u8) -> u8 {
    if cost.x_count == 1 {
        reduction.saturating_sub(cost.generic)
    } else {
        0
    }
}

/// Mana a Treasure's sacrifice ability adds: two of one color while its
/// controller controls Goldspan Dragon (which grants that ability).
pub fn treasure_mana_amount(state: &GameState, treasure: ObjectId) -> Option<u8> {
    let o = state.objects.get(treasure);
    if !CARD_DEFS[o.card_def as usize].subtypes.contains(&Subtype::Treasure) {
        return None;
    }
    state.players[o.controller.index()]
        .battlefield
        .iter()
        .any(|&id| CARD_DEFS[state.objects.get(id).card_def as usize].name == "Goldspan Dragon")
        .then_some(2)
}
