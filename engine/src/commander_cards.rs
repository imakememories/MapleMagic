//! Hand-written spell programs for the Commander decks' cards; `build.rs`'s
//! `hand_written_spell` points each card at its function.

use crate::effect::{EffectOp, ObjectRef, ThreatenFilter};

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
