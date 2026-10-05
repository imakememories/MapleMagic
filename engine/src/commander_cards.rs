//! Hand-written spell programs for the Commander decks' cards; `build.rs`'s
//! `hand_written_spell` points each card at its function.

use crate::effect::{EffectOp, ObjectRef, ThreatenFilter};

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
