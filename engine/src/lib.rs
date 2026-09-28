//! Deterministic MTG rules engine for a fixed Pauper pool.
//!
//! Trimmed from jackmaiorino/mtg-kernel (MIT): only the rules core is kept.
//! The research/audit/training-store layers of the original are dropped.

pub mod card_def;
pub mod effect;
pub mod engine;
pub mod environment_randomization_v2;
pub mod event;
pub mod ids;
pub mod mana;
pub mod runtime_decks;
pub mod snapshot;
pub mod state;
pub mod trigger;
