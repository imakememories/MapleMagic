//! Hand-written programs for the Commander decks' cards. `build.rs` points
//! spells at them through `hand_written_spell`, and activated abilities
//! through `AbilityEffectRecipe::Custom`.

use crate::card_def::DynamicValueDef;
use crate::effect::{EffectOp, LibraryCardFilter, ObjectRef, PlayerRef, TargetRef, ThreatenFilter};
use crate::state::Zone;

/// Abstergo Entertainment: "Return up to one target historic card from your
/// graveyard to your hand, then exile all graveyards."
pub fn abstergo_entertainment_effect() -> EffectOp {
    EffectOp::Sequence(vec![EffectOp::MoveAllTargets { to_zone: Zone::Hand }, EffectOp::ExileAllGraveyards])
}

/// "You may search": either nothing or the search.
fn may_search_basic_land(player: PlayerRef) -> EffectOp {
    EffectOp::Choice {
        controller: player,
        options: vec![
            EffectOp::Sequence(vec![]),
            EffectOp::SearchLibraryToBattlefield { player, filter: LibraryCardFilter::BasicLand },
        ],
    }
}

/// Demolition Field: destroy the target, then its controller and then you
/// may each search for a basic land. In two-player games the land's
/// controller is the opponent: the target must be theirs at resolution.
pub fn demolition_field_effect() -> EffectOp {
    EffectOp::Sequence(vec![
        EffectOp::DestroyObject { object: ObjectRef::Target(0) },
        may_search_basic_land(PlayerRef::Opponent),
        may_search_basic_land(PlayerRef::Controller),
    ])
}

/// Secluded Starforge: "Target creature gets +X/+0 until end of turn."
pub fn secluded_starforge_pump_effect() -> EffectOp {
    EffectOp::PumpTargetUntilEndOfTurnDynamic {
        target: TargetRef::Target(0),
        power: DynamicValueDef::X,
        toughness: DynamicValueDef::Fixed(0),
    }
}

/// "You may sacrifice an artifact or discard a card. If you do, draw two cards."
pub fn vision_of_love() -> Option<EffectOp> {
    Some(EffectOp::MayPayCostThen {
        discard: 1,
        sacrifice_lands: 0,
        sacrifice_artifacts: 1,
        then: Box::new(EffectOp::DrawCards {
            player: crate::effect::PlayerRef::Controller,
            count: 2,
        }),
    })
}

pub fn call_for_aid() -> Option<EffectOp> {
    Some(EffectOp::CallForAid)
}

pub fn shivan_devastator() -> Option<EffectOp> {
    Some(EffectOp::PutSourceOntoBattlefieldWithXPlusOneCounters)
}

pub fn delayed_blast_fireball() -> Option<EffectOp> {
    Some(EffectOp::DamageOpponentsAndTheirCreatures {
        amount: 2,
        exile_amount: 5,
    })
}

pub fn threaten_target() -> Option<EffectOp> {
    Some(EffectOp::Threaten {
        object: ObjectRef::Target(0),
    })
}

pub fn insurrection() -> Option<EffectOp> {
    Some(EffectOp::ThreatenAll {
        filter: ThreatenFilter::AllCreatures,
    })
}

pub fn mob_rule_power_four_or_more() -> Option<EffectOp> {
    Some(EffectOp::ThreatenAll {
        filter: ThreatenFilter::PowerAtLeast(4),
    })
}

pub fn mob_rule_power_three_or_less() -> EffectOp {
    EffectOp::ThreatenAll {
        filter: ThreatenFilter::PowerAtMost(3),
    }
}
