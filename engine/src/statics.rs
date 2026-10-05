//! Static abilities of permanents that affect other objects (604.1), such as
//! "Creatures you control have menace." Looked up by card id in a table built
//! once, so a query costs one pass over a battlefield.

use crate::card_def::{CardType, Keywords, CARD_DEFS};
use crate::ids::ObjectId;
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
