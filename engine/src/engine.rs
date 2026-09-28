//! Priority / stack / state-based-action turn engine.
//!
//! Two entry points, both fully general -- no "no response expected"
//! shortcuts and no auto-pass:
//!
//! - [`advance_until_decision`] drives the turn structure (the 12-step
//!   sequence in `state::Step`), runs SBAs/trigger-collection after every
//!   resolution, and returns the next real [`Decision`] a player must make.
//! - [`step`] applies a chosen [`Action`] in response to the last
//!   `Decision`, mutating `GameState`.
//!
//! All transient engine state that needs to survive between an
//! `advance_until_decision` call and the matching `step` call (whose spell
//! is mid-cast, who has already passed this priority round, the event log,
//! active replacements, queued triggers, in-progress combat) lives in
//! `GameState::engine` ([`EngineState`]) so both functions can stay pure
//! `&mut GameState` signatures with no separate engine object.
//!
//! Card-effect mutation still only ever happens through
//! `event::propose_and_commit`/`propose_and_commit_batch` (see
//! `effect.rs`/`event.rs`); everything in this module that mutates
//! `GameState` directly is turn/stack/priority/cost-payment bookkeeping,
//! not card behavior.
//!
//! ## Multi-stage decisions (casting, activating, discarding)
//!
//! Casting a spell, activating a non-mana ability, and discarding cards
//! are each potentially multi-step processes with real decision points in
//! the middle (choose targets, choose a cast mode, choose which cards to
//! discard). Each has a `Pending*` struct parked in `EngineState` that
//! `advance_until_decision`'s loop re-examines every pass: if a stage
//! isn't resolved yet, it returns the `Decision` for that stage (or, if
//! there's only one legal answer, resolves it automatically and
//! `continue`s -- same "don't ask when there's no real choice" pattern
//! increment 2 established for `OrderTriggers` singleton groups). Once
//! every stage is resolved, `finalize_cast`/`finalize_activation` pays the
//! remaining (non-discard) costs and pushes the stack item -- except a
//! cast's stack item: 601.2a puts a spell on the stack at *announcement*
//! (`begin_cast`), before targets/modes/costs, so `finalize_cast` only
//! fills in the placeholder `begin_cast` already pushed.

use crate::card_def::{
    self, ActivatedAbilityDef, ActivationTargetFilter, CardType, CostComponent, DynamicValueDef,
    Keywords, ManaAbilityAmountDef, ManaAbilityCostDef, ManaAbilityDef, OptionalAdditionalCostDef,
    PermanentFilter, PermanentFilterDef, TargetSpec,
};
use crate::effect::{
    self, EffectObjectBinding, EffectOp, ExecCtx, ObjectRef, StormCopyBindingV1, TargetRef,
};
use crate::event::{self, ActiveReplacement, CommittedEvent, ProposedEvent};
use crate::ids::{ObjectId, PlayerId, StackItemId};
use crate::mana::{self, Cost, ManaColor, ManaColorSetV1};
use crate::state::{
    stack_target_contract_is_structurally_valid, AbilityKindV4, AbilitySourceContractV4,
    CastMethodV4, FinalizedCastBindingV1, GameState, InitiativeTriggerKindV1,
    MadnessOfferSourceContractV4, ObjectLinkV4, ObjectStateV4, PaidCostRefV4, SpellCastOriginV4,
    SpellCastRouteV4, SpellCopyOriginV4, StackItem, StackItemKind, StackSourceContractV4,
    StackStateV4, StackTargetContractV4, Step, Target, Zone,
};
use crate::trigger::{self, PendingTrigger};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EngineState {
    /// Last allocated private stack-incarnation id. Zero remains reserved for
    /// legacy/default-constructed state, so every engine-created item is
    /// distinguishable from an unstamped continuation.
    #[serde(default)]
    pub next_stack_item_id: u64,
    /// Whether [P0, P1] has passed priority since the last time priority
    /// was reset (new step, a cast/activation/land-drop, or a resolution).
    pub priority_passes: [bool; 2],
    /// Bumped by `reset_priority` alone -- i.e. only on the two rules-level
    /// "everyone's pass streak clears and the active player is asked
    /// first" boundaries (a new step via `advance_step`, or a stack
    /// resolution) and the declare-attackers/declare-blockers transition
    /// into that step's own priority phase (`apply_declare_attackers`/
    /// `apply_declare_blockers`, which route through the same helper).
    /// Deliberately *not* bumped by the other four `priority_passes =
    /// [false, false]` sites (`finalize_cast`, `finalize_activation`,
    /// `play_land`, `plot_spell`, `push_trigger_onto_stack`): those hand
    /// priority back to the *same* player who just acted (601.2i/117.3b),
    /// which is a real fresh priority window under the comprehensive rules
    /// but not a new "round" in the sense this counter tracks. Exists
    /// purely so `mtg_kernel::surface::HarnessSurfaceV1` can detect the
    /// DeclareAttackers/DeclareBlockers one-action-per-round throttle
    /// (`ComputerPlayerRL.priorityPlay`'s hard-coded `act(); pass();` for
    /// those two steps) without re-deriving round boundaries from stack
    /// length or step identity, both of which are ambiguous across turns.
    pub priority_round: u64,
    /// `state.stack.len()` at the exact instant `reset_priority` last ran.
    /// Kept as a diagnostic distinction between the rules-level round
    /// boundary and the surface's deliberately lazy first-observation
    /// baseline. `HarnessSurfaceV2` must not use this value for its ordinary
    /// own-cast suppression: Java resets every player's passed flag after a
    /// resolution, then places any resolution-created triggers before the
    /// next priority ask, so those triggers need real priority windows. By
    /// contrast, a cast and its cast-time triggers happen without a
    /// `reset_priority` boundary and remain covered by the cast's trailing
    /// `ComputerPlayerRL.act()` force-pass. Eagerly using this snapshot for
    /// both cases regresses legitimate resolution ETBs such as Voldaren
    /// Epicure.
    pub stack_len_at_round_open: usize,
    /// A spell that has been announced (601.2a: already moved to the stack
    /// by `begin_cast`) but not yet finished being targeted, mode-chosen,
    /// or cost-paid.
    pub pending_cast: Option<PendingCast>,
    /// A non-mana ability that has begun activating but isn't fully
    /// targeted/cost-paid yet (battlefield abilities, hand-zone
    /// Cycling/typecycling, or graveyard abilities such as Embalm).
    pub pending_activation: Option<PendingActivation>,
    /// A discard obligation -- from a cast/activation's cost, from a
    /// resolving effect (`EffectOp::DiscardCards`), or from cleanup --
    /// waiting on `Decision::Discard`/`Action::Discard`. See `card_def`'s
    /// `CostComponent::DiscardCards` and the `EffectOp::DiscardCards` doc
    /// for why this needs its own pending-state slot instead of being
    /// solved synchronously like every other cost/effect leaf.
    pub pending_discard: Option<PendingDiscard>,
    /// A resolution-time optional cost (Highway Robbery's "you may discard
    /// a card or sacrifice a land") staged by `EffectOp::MayPayCostThen`,
    /// waiting on `Decision::ChooseOptionalCost`/`Action::ChooseOptionalCost`.
    pub pending_optional_cost: Option<PendingOptionalCost>,
    /// Once `Action::ChooseOptionalCost(SacrificeLand)` is chosen, *which*
    /// land(s) to sacrifice -- a real decision when more than one is legal,
    /// same `Decision::ChooseCostTargets`/`ChooseCastMode`-mode-Alternative
    /// shape as Fireblast's alt cost and Lava Dart's flashback cost (see
    /// `sacrifice_lands_needed`'s doc and `drain_pending_optional_cost_
    /// sacrifice_or_decide`). Previously auto-solved silently by
    /// `sacrifice_lowest_id_lands` with no `Decision` at all -- same bug
    /// class as the other two, root-caused the same way (Sol #90,
    /// increment 11).
    pub pending_optional_cost_sacrifice: Option<PendingOptionalCostSacrifice>,
    /// A Chain Lightning resolution suspended between dealing damage and
    /// finishing its spell/copy. Payment, copy construction, and optional
    /// retargeting all occur inside the same resolution: while this is
    /// present, the resolving stack item stays live and SBAs/triggers/
    /// priority are deliberately deferred.
    pub pending_spell_copy: Option<PendingSpellCopy>,
    /// Generic, owned effect-frame continuation for policy-visible choices
    /// yielded during resolution. New choice-bearing mechanics use this one
    /// typed interpreter state instead of adding card-specific `pending_*`
    /// slots. The certified legacy discard/optional-cost/Chain continuations
    /// remain separate until they are migrated deliberately.
    pub pending_effect: Option<effect::EffectContinuation>,
    /// Transient buffer for the *current* resolution: `event::commit`
    /// appends here, `trigger::collect_and_process` drains it after every
    /// resolution to match triggers. Empty between resolutions.
    pub event_log: Vec<CommittedEvent>,
    /// Full permanent record of every committed event this game, in
    /// commit order. Never drained; this is what
    /// `event::commit`/`trigger::collect_and_process`'s draining of
    /// `event_log` would otherwise make unobservable after the fact (game
    /// replay / RL trace logging / the acceptance tests' event-log
    /// assertions all read this instead).
    pub event_history: Vec<CommittedEvent>,
    pub active_replacements: Vec<ActiveReplacement>,
    pub next_replacement_id: u32,
    /// Exact historical links created by effects such as Mesmeric Fiend.
    /// A source zone change cannot erase these records before its queued
    /// leave trigger resolves, and a later exiled-card incarnation cannot
    /// satisfy the frozen binding.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub linked_exile_records: Vec<crate::state::LinkedExileRecordV4>,
    /// Triggered abilities collected but not yet placed on the stack,
    /// grouped APNAP (active player's group first); see
    /// `drain_pending_triggers_or_decide`.
    pub pending_triggers: Vec<PendingTrigger>,
    /// This turn's combat, reset fresh at every `Step::BeginCombat`.
    pub combat: CombatState,
    /// "Until end of turn" continuous effects, cleared at every
    /// `Step::Cleanup` (514.2). Basilisk Gate's resolved pump uses this
    /// generic shape; see
    /// `tests::until_end_of_turn_effects_are_cleared_at_cleanup`.
    pub until_end_of_turn: Vec<UntilEndOfTurnEffect>,
    /// Bumped every time `Action::ActivateManaAbility` runs. 605.3b: a mana
    /// ability never touches `GameState::stack`, so `HarnessSurfaceV2`'s
    /// `DeclareAttackers`/`DeclareBlockers` one-action-per-round throttle
    /// (`combat_priority_stack_len_seen`, `surface_v2.rs`) -- which re-arms
    /// both players' throttle flags by watching `state.stack.len()` change
    /// -- can't see a mana ability activation at all, even though Java's
    /// `PlayerImpl.activateAbility` calls `game.getPlayers().resetPassed()`
    /// unconditionally for *every* successful action, `ACTIVATED_MANA`
    /// included (see `Action::ActivateManaAbility`'s own comment). This
    /// counter is that missing signal: `HarnessSurfaceV2` compares it
    /// against its own last-seen value the same way it already compares
    /// `state.stack.len()`, and re-arms on either changing. Root-caused
    /// (increment 13) against `game_20260713_002148_0003.txt` decision 34
    /// and `game_20260713_002202_0024.txt` decision 179: in both, one
    /// player activates a mana ability mid-`DeclareAttackers`/
    /// `DeclareBlockers` round after *both* players had already spent this
    /// round's one action, and the reference genuinely re-asks the other
    /// player right afterward (matching `resetPassed()`) -- but the surface,
    /// seeing no stack-length change, kept both throttle flags stale-true
    /// and silently force-passed everyone straight through the rest of
    /// combat, reaching `Main2` a full combat phase early.
    pub mana_ability_activations: u64,
    /// `mana_ability_activations` at the exact instant `reset_priority`
    /// last ran. This is the boundary-scoped baseline for RL observation
    /// fields that claim "since priority boundary" semantics; the lifetime
    /// counter and last activator remain separate internal H2 signals.
    pub mana_ability_count_at_round_open: u64,
    /// Single-shot, cleared-every-time signal from a just-finished
    /// resolution to the trigger-collection pass that always immediately
    /// follows it: `Some(source)` iff the stack item that just resolved had
    /// `StackItem::kicked == true`, naming *that item's own* source object.
    /// `resolve_top_of_stack` sets this (to `Some` or explicitly `None`,
    /// never leaving a stale value from 2+ resolutions ago) right before
    /// running the resolution's effect; `trigger::collect_and_process`
    /// `take()`s it immediately when matching an ETB trigger, stamping
    /// `kicked` onto the resulting `PendingTrigger` only when its own
    /// `source` matches. This is cast-time *metadata flowing through the
    /// current resolution's context*, not a durable lookup table keyed by
    /// stable object id (CR 400.7: zone changes create new objects, so a
    /// persistent id-keyed marker could falsely survive a later, unkicked
    /// cast of the same physical card) -- see `EffectCond::WasKicked`'s doc.
    pub pending_kicked_source: Option<ObjectId>,
    /// Exile-zone "you may play/cast this" grants from an impulse-draw
    /// effect (Clockwork Percussionist, Experimental Synthesizer, Reckless
    /// Impulse) -- see `PlayPermission`'s doc.
    pub exile_play_permissions: Vec<PlayPermission>,
    /// Monotonic counter for `UntilEndOfTurnEffect::ResolvedSetEffect::
    /// timestamp` (613.7's timestamp ordering) -- bumped by
    /// `next_timestamp`. Evaluation stays flat (unordered) this increment;
    /// this exists so a future layer-aware evaluator has real creation-order
    /// data to sort by without needing every effect re-created retroactively.
    pub next_effect_timestamp: u64,
    /// `Some((mechanic, source))` iff the game walk hit a runtime branch this
    /// kernel cannot simulate faithfully. No currently supported Pauper-pool
    /// card sets this field; the fail-closed terminal remains part of the
    /// public runner contract for future card increments. Checked at the top
    /// of `advance_until_decision`'s loop, same as `check_game_over`:
    /// once set, the walk is over -- `Decision::Halted` is the only decision
    /// this state will ever produce again.
    pub halted: Option<(UnsupportedMechanic, ObjectId)>,
    /// Which player performed the most recent `Action::ActivateManaAbility`
    /// (paired with `mana_ability_activations` as its "version" counter).
    /// `HarnessSurfaceV2` needs this alongside the counter, not just the
    /// counter alone: unlike a stack-growing cast/activation (where
    /// `state.stack.last().controller` durably answers "is the thing that
    /// reopened this round still mine" for as long as that item sits
    /// unresolved), a mana ability leaves nothing behind on `state.stack` to
    /// re-inspect later -- so re-arming *both* players' throttle flags
    /// blindly on every count change (mirroring the stack-length re-arm)
    /// would also un-suppress the *activator's own* immediate reprompt,
    /// which must stay suppressed (they already had their one action this
    /// round). This field lets the surface tell "was I the one who just
    /// reopened this" apart from "did the *other* player just reopen this
    /// for me", the same distinction `stack_top_is_fresh_own_item` draws
    /// structurally for the stack case.
    pub last_mana_ability_activator: Option<PlayerId>,
    /// A land play whose as-this-enters color choice must complete before
    /// the card changes zones. Omitted from legacy snapshots when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pending_land_play: Option<PendingLandPlay>,
    /// Historical card-backed source used to represent the global Initiative
    /// designation's triggered abilities on this object-id based stack.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initiative_source: Option<AbilitySourceContractV4>,
    /// Keyword grants that expire as the named player's next turn begins.
    /// Undercity's Throne room is the first consumer.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub until_next_turn_keywords: Vec<UntilNextTurnKeywordEffectV1>,
}

fn next_stack_item_id(state: &mut GameState) -> StackItemId {
    state.engine.next_stack_item_id = state
        .engine
        .next_stack_item_id
        .checked_add(1)
        .expect("stack-item identity space exhausted");
    StackItemId(state.engine.next_stack_item_id)
}

/// 613.1's continuous-effect layers this kernel's card pool actually
/// grants, as a bitset (same pattern as `card_def::Keywords`): layer 6
/// (ability adding/removing -- a granted Haste) and layer 7c (effects that
/// modify power/toughness without setting it). A single `ResolvedSetEffect`
/// can span both at once (Goblin Bushwhacker's kicked ETB grants +1/+0
/// *and* Haste from one resolution) -- evaluation stays flat/unordered this
/// increment (nothing in this pool has two effects at the *same* layer that
/// would disagree on order), but the bits are recorded now so a future
/// layer-aware evaluator can filter/sequence by them without every existing
/// effect needing to be re-tagged retroactively.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash, Serialize, Deserialize)]
pub struct Layers(pub u8);

impl Layers {
    pub const NONE: Layers = Layers(0);
    /// 613.1, layer 4.
    pub const TYPE_CHANGING: Layers = Layers(1 << 2);
    /// 613.1, layer 6.
    pub const ABILITY_ADDING: Layers = Layers(1 << 0);
    /// 613.1, layer 7c.
    pub const POWER_TOUGHNESS: Layers = Layers(1 << 1);

    pub const fn has(self, other: Layers) -> bool {
        self.0 & other.0 != 0
    }
}

impl std::ops::BitOr for Layers {
    type Output = Layers;
    fn bitor(self, rhs: Layers) -> Layers {
        Layers(self.0 | rhs.0)
    }
}

/// How long a continuous effect lasts. One variant today (every effect this
/// pool creates is "until end of turn"); kept as its own type rather than
/// inlined so a longer-lived duration (Goblin Tomb Raider's static boost is
/// modeled separately, via `static_self_boost_for`, precisely because it
/// *isn't* a resolved, timed effect) doesn't force a reshape later.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EffectDuration {
    EndOfTurn,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UntilNextTurnKeywordEffectV1 {
    pub object_id: ObjectId,
    pub object_zone_change_count: u32,
    pub holder: PlayerId,
    pub expires_at_turn: u32,
    pub keywords: Keywords,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UntilEndOfTurnEffect {
    /// Synthetic placeholder -- see the `until_end_of_turn` field doc.
    SyntheticMarker(ObjectId),
    /// A resolved, locked-in set of affected objects getting `power`/
    /// `toughness` and (if `grant_haste`) Haste until end of turn
    /// (`effect::EffectOp::PumpControlled` -- Goblin Bushwhacker's kicked
    /// ETB, Rally at the Hornburg's token haste). `object_ids` is captured
    /// *at resolution* (611.2c: a resolving "creatures you control get..."
    /// fixes its own affected set right then; a creature that enters
    /// *later* this same turn is never added, even though the effect is
    /// still active) -- `effect::EffectOp::PumpControlled`'s own doc has
    /// the sequencing argument for why Rally at the Hornburg's two freshly
    /// -created tokens are still included (they exist *before* this leaf
    /// runs) while Goblin Bushwhacker's pump still correctly reaches
    /// Burning-Tree Emissary (already on the battlefield at resolution,
    /// same reasoning, opposite direction). `layer`/`timestamp` are
    /// recorded now (see `Layers`'s doc) even though `effective_power`/
    /// `effective_toughness`/`has_effective_keyword` still apply every
    /// entry unconditionally (flat evaluation).
    ResolvedSetEffect {
        object_ids: Vec<ObjectId>,
        layer: Layers,
        timestamp: u64,
        duration: EffectDuration,
        power: i32,
        toughness: i32,
        grant_haste: bool,
    },
    /// One exact battlefield incarnation receives a resolved temporary
    /// modification. Unlike the historical set form, this binding cannot
    /// follow the same arena id through a zone change under CR 400.7.
    ResolvedObjectEffect {
        object_id: ObjectId,
        object_zone_change_count: u32,
        layer: Layers,
        timestamp: u64,
        duration: EffectDuration,
        power: i32,
        toughness: i32,
        grant_haste: bool,
    },
    /// One exact battlefield incarnation receives a temporary keyword.
    /// Kept separate from the historical Haste-only pump fields so existing
    /// serialized variants retain their shape.
    ResolvedObjectKeywordEffect {
        object_id: ObjectId,
        object_zone_change_count: u32,
        layer: Layers,
        timestamp: u64,
        duration: EffectDuration,
        keywords: Keywords,
    },
    /// Global rule-changing effect from Flaring Pain. Prevention shields
    /// remain installed and unconsumed, but do not apply while this marker
    /// lasts.
    DamageCannotBePrevented {
        timestamp: u64,
        duration: EffectDuration,
    },
}

/// Whether an `effect::EffectOp::ImpulseDraw`-exiled card's `PlayPermission`
/// authorizes casting it (a spell) or playing it (a land) -- see that
/// struct's doc. Computed once, at grant time, from the definition's
/// executable cast/play capability; kept as its own explicit field (rather than re-deriving it from
/// the card def every time a permission is checked) so a future permission
/// shape that grants "play" without the card being a land in the normal
/// sense (not needed by this pool) doesn't have to be inferred implicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PlayOrCast {
    Cast,
    Play,
}

/// When a `PlayPermission` expires. `EndOfTurn` (Experimental Synthesizer)
/// is cleared unconditionally at the very next `Step::Cleanup`, whoever's --
/// only one cleanup can ever happen before the *current* turn ends,
/// regardless of round numbering, so "the next one, unconditionally" is
/// exactly "this turn". `UntilHoldersNextTurn` (Clockwork Percussionist,
/// Reckless Impulse) can't be tracked as a plain turn-number comparison
/// against `GameState::turn`: that counter is a *round* number shared by
/// both players' own single turn within it (see `state::GameState`'s module
/// doc / `run_step_entry_action`'s `Step::Draw` comment), so it can't by
/// itself distinguish "the holder's own next turn" from "the opponent's
/// turn happening to share the same round number". Tracked instead via the
/// holder's own `Step::Untap` boundary, which is unambiguous regardless of
/// round numbering: `holder_turn_started` flips `true` the first time the
/// *holder* (not the opponent) untaps after this permission was granted, and
/// the permission is removed at the holder's *next* `Step::Cleanup` after
/// that -- i.e. it survives the rest of the current turn, the opponent's
/// whole turn, and the holder's entire next turn, expiring exactly at that
/// turn's cleanup (702.163-ish "until end of your next turn" wording).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PlayPermissionExpiry {
    EndOfTurn,
    UntilHoldersNextTurn { holder_turn_started: bool },
}

/// Grants `holder` permission to play/cast `object` straight out of
/// `Zone::Exile`, following *ordinary* timing/costs/land-quota (i.e. this is
/// consulted by `castable_spells`/`land_drop_candidates` alongside their
/// normal checks, not a parallel "impulse-castable" legality system) --
/// Clockwork Percussionist's dies trigger, Experimental Synthesizer's
/// enters-or-leaves trigger, Reckless Impulse. Replaces an earlier
/// pseudo-hand-zone design per external review: this is a *permission*, not
/// membership in a zone-like list.
///
/// - `holder`: who may act on this permission -- not necessarily `object`'s
///   owner (this pool's own 3 cards always grant it to the caster/
///   controller, who is also always the owner, but the shape doesn't bake
///   that coincidence in, so a future "you may play target player's exiled
///   card" doesn't need a redesign).
/// - `zone_change_generation`: a snapshot of `GameObject::zone_change_count`
///   taken the instant this permission is granted (*after* the exile move
///   that creates it). `active_permission_for` re-checks this against the
///   object's *current* count every time: CR 400.7 (zone changes create new
///   objects) means this permission is void the moment `object` changes
///   zones again for any reason, not just when it's played through this
///   permission -- a stale entry can never silently "come back to life"
///   after e.g. some other effect returns the same physical card to exile
///   again later.
/// - `play_or_cast`: which ordinary action (`Action::CastSpell` vs
///   `Action::PlayLand`) this authorizes.
/// - `expiry`: see `PlayPermissionExpiry`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PlayPermission {
    pub object: ObjectId,
    pub holder: PlayerId,
    pub zone_change_generation: u32,
    pub play_or_cast: PlayOrCast,
    pub expiry: PlayPermissionExpiry,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EnginePublicContextV2 {
    pub priority_passes: [bool; 2],
    pub priority_round: u64,
    pub stack_len_at_round_open: usize,
    pub mana_ability_activations: u64,
    pub last_mana_ability_activator: Option<PlayerId>,
    pub pending_cast: Option<PendingCastPublicV2>,
    pub pending_activation: Option<PendingActivationPublicV2>,
    pub pending_discard: Option<PendingDiscardPublicV2>,
    pub pending_optional_cost: Option<PendingOptionalCostPublicV2>,
    pub pending_optional_cost_sacrifice: Option<PendingOptionalCostSacrificePublicV2>,
    pub pending_spell_copy: Option<PendingSpellCopyPublicV2>,
    pub pending_triggers: Vec<PendingTriggerPublicV2>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PendingCastPublicV2 {
    pub spell: ObjectId,
    pub controller: PlayerId,
    pub target_spec: TargetSpec,
    pub targets_chosen: Vec<Target>,
    pub is_flashback: bool,
    pub cast_mode: Option<CastMode>,
    pub additional_cost_discarded: Option<Vec<ObjectId>>,
    pub mode_chosen: Option<u8>,
    pub origin_zone: Zone,
    pub sacrifice_chosen: Vec<ObjectId>,
    pub kicked: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PendingActivationPublicV2 {
    pub source: ObjectId,
    pub controller: PlayerId,
    pub ability_index: u8,
    pub target_spec: TargetSpec,
    pub targets_chosen: Vec<Target>,
    pub cost_discard_paid: Option<Vec<ObjectId>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub object_cost_chosen: Vec<ObjectId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiscardResumePublicKindV2 {
    None,
    FinishCast,
    FinishActivation,
    FinishSpellResolution,
    FinishOptionalCost,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PendingDiscardPublicV2 {
    pub player: PlayerId,
    pub count: u32,
    pub resume_kind: DiscardResumePublicKindV2,
    pub resume_source: Option<ObjectId>,
    pub resume_controller: Option<PlayerId>,
    pub resume_to_zone: Option<Zone>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PendingOptionalCostPublicV2 {
    pub player: PlayerId,
    pub source: ObjectId,
    pub discard: u8,
    pub sacrifice_lands: u8,
    pub discard_payable: bool,
    pub sacrifice_payable: bool,
    pub spell_resume: Option<(ObjectId, Zone)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PendingOptionalCostSacrificePublicV2 {
    pub player: PlayerId,
    pub source: ObjectId,
    pub remaining: u8,
    pub chosen: Vec<ObjectId>,
    pub spell_resume: Option<(ObjectId, Zone)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PendingSpellCopyPublicV2 {
    pub resolving_source: ObjectId,
    pub player: PlayerId,
    pub inherited_target: Target,
    pub stage: SpellCopyStage,
    pub copy_source: Option<ObjectId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PendingTriggerPublicV2 {
    pub source: ObjectId,
    pub controller: PlayerId,
    pub is_madness_offer: bool,
    pub kicked: bool,
    pub target_spec: TargetSpec,
    pub targets: Vec<Target>,
    pub placement_ordered: bool,
}

/// A runtime mechanic the kernel cannot simulate faithfully. Distinct from
/// source-level fail-closed card support, this terminal classification is
/// discovered only after a game has begun. `EngineState::halted` and
/// `Decision::Halted` are how a runner rejects such an episode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UnsupportedMechanic {
    /// Backward-compatible wire value retained for old artifacts written
    /// before Chain Lightning gained its full copy/retarget state machine.
    /// Current engine code never emits it.
    SpellCopy,
    /// A generated choice-bearing effect program violated the resumable
    /// interpreter's fail-closed structural contract. Supported card
    /// definitions must never reach this at runtime.
    InvalidEffectContinuation,
}

/// Issues the next 613.7 timestamp for a newly-created
/// `UntilEndOfTurnEffect::ResolvedSetEffect`. `pub(crate)` since only
/// `effect::execute` (`EffectOp::PumpControlled`) needs to call it.
pub(crate) fn next_timestamp(state: &mut GameState) -> u64 {
    let t = state.engine.next_effect_timestamp;
    state.engine.next_effect_timestamp += 1;
    t
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CastMode {
    /// Pay the card's printed mana cost.
    Normal,
    /// Pay its `CardDef::alt_cost` instead (Fireblast: sacrifice 2 Mountains).
    Alternative,
}

/// Which interactive object-cost family `Decision::ChooseCostTargets` is
/// currently paying. The stable enum reserves more families than the engine
/// executes today; land sacrifice and Escape graveyard exile are the two
/// live cast-payment variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CostKind {
    SacrificeLands,
    SacrificePermanents,
    SacrificeCreatures,
    SacrificeArtifacts,
    DiscardCards,
    ExileFromGraveyard,
    TapPermanents,
    ReturnPermanentsToHand,
    PayLife,
    RemoveCounters,
    PutCounters,
    /// Monstrous Emergence's public permanent choice or privately staged
    /// hand-card choice. The selected zone is bound separately before this
    /// object pick is offered.
    ChooseCreatureOrRevealCreature,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChosenCreatureCostZoneV1 {
    Battlefield,
    Hand,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PendingCast {
    pub spell: ObjectId,
    /// Exact physical source incarnation and stack provenance captured after
    /// 601.2a moved the announced card. This must exactly equal the untouched
    /// top placeholder's own source contract throughout staging.
    pub source_contract: StackSourceContractV4,
    pub controller: PlayerId,
    pub target_spec: TargetSpec,
    pub targets_chosen: Vec<Target>,
    /// Exact target identities captured at 601.2c, before any cost can move
    /// one of those objects. Parallel to `targets_chosen`.
    #[serde(default)]
    pub target_contracts: Vec<StackTargetContractV4>,
    /// True once the caster explicitly finishes a variable-cardinality
    /// target announcement before reaching its maximum. Fixed-cardinality
    /// spells and variable selections that reach their maximum keep this
    /// false and complete by derivation.
    #[serde(default)]
    pub target_selection_finished: bool,
    /// True iff `spell` is being cast from the graveyard via flashback
    /// (`CardDef::flashback`), which uses a wholly different cost (never
    /// the printed mana cost) and exiles instead of going to the
    /// graveyard on resolution.
    pub is_flashback: bool,
    /// `None` until resolved. Pre-seeded to `Some(CastMode::Normal)` at
    /// `begin_cast` for every card without an `alt_cost` (i.e. almost
    /// everything), so the "which mode" decision stage is skipped
    /// entirely unless the card is Fireblast.
    pub cast_mode: Option<CastMode>,
    /// `None` until this cast's mandatory additional cost (if any) has
    /// been paid; pre-seeded to `Some(vec![])` at `begin_cast` for every
    /// card without an `additional_cost`. Holds whichever cards were
    /// discarded to pay it (Grab the Prize), read by
    /// `EffectCond::DiscardedNonLandForCost` via `ExecCtx::discarded`.
    pub additional_cost_discarded: Option<Vec<ObjectId>>,
    /// `None` until resolved, only meaningful for a modal card
    /// (`CardDef::mode2.is_some()`) -- pre-seeded to `Some(0)` at
    /// `begin_cast` for every non-modal card, so the "which mode" decision
    /// stage is skipped entirely unless the card is one of the four Blast
    /// cards. `0` = the card's primary `target_spec`/`spell_effect`, `1` =
    /// `mode2`.
    pub mode_chosen: Option<u8>,
    /// Which zone this cast was announced from (Hand, Graveyard for
    /// Flashback/Escape, or Exile for Plot/Madness) -- `begin_cast` captures the
    /// spell's zone *before* `move_to_stack` changes it, purely so
    /// `abort_cast` knows where to return the card if its cost turns out to
    /// be unpayable or a staged payment continuation becomes impossible.
    pub origin_zone: Zone,
    /// Objects chosen so far for this cast's one interactive object-cost
    /// component. The privileged serialized compatibility name predates Escape:
    /// Fireblast/Lava Dart store lands selected for `SacrificeLands`, while
    /// an Escape cast stores cards selected for
    /// `ExileOtherCardsFromOwnGraveyard`. Supported cost shapes make those
    /// two families mutually exclusive. The current public semantic/flat
    /// projection retains its frozen sacrifice-only meaning and deliberately
    /// hides this vector for Escape. Escape therefore remains outside the
    /// current policy-training authority until a versioned successor exposes
    /// generic object-cost selections to the value-state encoder.
    pub sacrifice_chosen: Vec<ObjectId>,
    /// `None` until resolved, only meaningful for a card with `CardDef::
    /// kicker_cost` (Goblin Bushwhacker) -- pre-seeded to `Some(false)` at
    /// `begin_cast_ex` for every card without one. `Some(true)` means the
    /// kicker cost will be paid alongside the base cost in `finalize_cast`,
    /// which also stamps `state::StackItem::kicked` on this cast's own
    /// stack item -- see that field's doc for how it flows onward into the
    /// resolution/ETB context from there.
    pub kicked: Option<bool>,
    /// `None` until the caster decides whether to pay this definition's
    /// optional additional cost. Cards without one, plus declined costs,
    /// use the legacy-compatible `Some(false)` seed.
    #[serde(
        default = "default_optional_additional_cost_declined",
        skip_serializing_if = "optional_additional_cost_is_declined"
    )]
    pub optional_additional_cost_paid: Option<bool>,
    /// Exact-incarnation objects selected to pay Collect Evidence or
    /// Bargain. Definition data supplies the filter and cardinality; these
    /// bindings never self-authenticate a payment.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub optional_additional_cost_chosen: Vec<EffectObjectBinding>,
    /// Collect Evidence alone needs an explicit finish marker after its
    /// selected mana-value sum reaches the printed threshold.
    #[serde(default, skip_serializing_if = "bool_is_false")]
    pub optional_additional_cost_selection_finished: bool,
    /// Announced X. X spells leave this unresolved until all modes and
    /// targets are chosen, then expose the existing numeric effect-choice
    /// surface. Non-X spells are seeded to zero.
    #[serde(default)]
    pub x_value: Option<u8>,
    /// Which branch of the generic controlled-creature-or-reveal-hand-card
    /// additional cost was selected. Cards without that cost keep `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chosen_creature_cost_zone: Option<ChosenCreatureCostZoneV1>,
    /// Exact chosen creature incarnation for that additional cost.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chosen_creature_cost: Option<EffectObjectBinding>,
}

fn default_optional_additional_cost_declined() -> Option<bool> {
    Some(false)
}

fn optional_additional_cost_is_declined(value: &Option<bool>) -> bool {
    *value == Some(false)
}

fn bool_is_false(value: &bool) -> bool {
    !*value
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PendingActivation {
    pub source: ObjectId,
    /// Exact CR 400.7 incarnation bound when activation began (or refreshed
    /// by the trusted interactive-cost payment path after that path moves
    /// the source). Restored/tampered pending activations fail closed before
    /// paying mana, moving a card, or pushing a stack item.
    pub source_zone_change_count: u32,
    pub controller: PlayerId,
    pub ability_index: u8,
    pub target_spec: TargetSpec,
    pub targets_chosen: Vec<Target>,
    #[serde(default)]
    pub target_contracts: Vec<StackTargetContractV4>,
    /// `None` while an interactive `DiscardCards` component still needs a
    /// choice. Abilities without one are seeded with `Some([])`. Paid
    /// interactive activations finalize synchronously in `apply_discard`,
    /// so an untrusted serialized record can never authenticate payment by
    /// filling this vector itself.
    pub cost_discard_paid: Option<Vec<ObjectId>>,
    /// Incarnation-bound physical permanents selected for the ability's one
    /// supported interactive object-cost component. This is definition-
    /// driven and deliberately independent of the source card's identity.
    #[serde(default)]
    pub object_cost_chosen: Vec<EffectObjectBinding>,
}

/// A staged land drop waiting for an as-this-enters color choice.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PendingLandPlay {
    pub source: ObjectId,
    pub source_zone_change_count: u32,
    pub controller: PlayerId,
    pub origin_zone: Zone,
    pub excluded_color: ManaColor,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DiscardResume {
    /// Nothing further to do once the discard lands (cleanup's
    /// discard-to-7).
    None,
    /// Write the discarded cards back into `EngineState::pending_cast`'s
    /// `additional_cost_discarded` and let the cast staging continue.
    FinishCast {
        source_contract: StackSourceContractV4,
        controller: PlayerId,
    },
    /// Same, but for `EngineState::pending_activation`'s interactive
    /// discard cost. The exact activation identity is duplicated here so a
    /// restored cross-slot mismatch fails before any selected card moves.
    FinishActivation {
        source: ObjectId,
        source_zone_change_count: u32,
        controller: PlayerId,
        ability_index: u8,
    },
    /// A resolving instant/sorcery's own effect discarded as part of its
    /// resolution (Faithless Looting: "draw two, then discard two").
    /// `EffectOp::DiscardCards` stages `pending_discard` and returns
    /// *synchronously*, before the discard is actually answered (see that
    /// leaf's doc) -- so by the time `execute` returns to
    /// `resolve_top_of_stack`, the discard hasn't happened yet. 608.2m: the
    /// spell can only move to its post-resolution zone as the *last* part
    /// of its resolution, which isn't done until this discard lands -- so
    /// `resolve_top_of_stack` defers that move here instead of doing it
    /// immediately, and `apply_discard` performs it once the discard is
    /// actually resolved.
    FinishSpellResolution { source: ObjectId, to_zone: Zone },
    /// The discard sub-cost of a resolution-time optional cost (Highway
    /// Robbery's `EffectOp::MayPayCostThen` -> `Action::ChooseOptionalCost
    /// (OptionalCostChoice::Discard)`) just landed -- run `then` now.
    /// `spell_resume`, if `Some`, is `PendingOptionalCost::spell_resume`
    /// carried over -- the deferred move to apply once `then` runs (see
    /// that field's doc).
    FinishOptionalCost {
        source: ObjectId,
        controller: PlayerId,
        then: Box<EffectOp>,
        spell_resume: Option<(ObjectId, Zone)>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PendingDiscard {
    pub player: PlayerId,
    pub count: u32,
    pub resume: DiscardResume,
}

/// A resolution-time optional cost (`effect::EffectOp::MayPayCostThen`),
/// waiting on `Decision::ChooseOptionalCost`. `discard_payable`/
/// `sacrifice_payable` are snapshotted at stage time (by `execute`, which
/// already checked at least one is true before staging this at all) so
/// `Action::ChooseOptionalCost` can validate the chosen option without
/// recomputing legality against however state has (not) changed since.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PendingOptionalCost {
    pub player: PlayerId,
    pub source: ObjectId,
    pub discard: u8,
    pub sacrifice_lands: u8,
    pub discard_payable: bool,
    pub sacrifice_payable: bool,
    pub then: EffectOp,
    /// `Some((source, to_zone))` iff this optional cost is itself part of
    /// `source`'s own spell resolution (Highway Robbery's "you may... if
    /// you do, draw two cards" -- `EffectOp::MayPayCostThen` staged this
    /// synchronously, from inside `resolve_top_of_stack`'s own `execute`
    /// call) and that spell hasn't moved to `to_zone` (its post-resolution
    /// zone) yet. See `resolve_top_of_stack`'s doc for why this can't just
    /// move there immediately the way it does for every other instant/
    /// sorcery -- same 608.2m "moves to its zone only as the very last
    /// part of resolution" deferral `DiscardResume::FinishSpellResolution`
    /// already handles for `EffectOp::DiscardCards`, just at the optional-
    /// cost layer instead. `None` for the normal case: an optional cost
    /// staged by something *other* than a spell's own top-level resolution
    /// has nothing waiting on it (no card in this pool triggers
    /// `MayPayCostThen` from anywhere but a spell resolving, but this
    /// isn't assumed -- `resolve_top_of_stack` is the only place that ever
    /// sets it `Some`).
    pub spell_resume: Option<(ObjectId, Zone)>,
}

/// Staged once `Action::ChooseOptionalCost(SacrificeLand)` is chosen --
/// see `EngineState::pending_optional_cost_sacrifice`'s doc.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PendingOptionalCostSacrifice {
    pub player: PlayerId,
    pub source: ObjectId,
    pub remaining: u8,
    pub chosen: Vec<ObjectId>,
    pub then: EffectOp,
    /// Carried over from `PendingOptionalCost::spell_resume` -- see that
    /// field's doc.
    pub spell_resume: Option<(ObjectId, Zone)>,
}

/// Which decision inside Chain Lightning's resolution is currently live.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpellCopyStage {
    /// The affected player decides whether to pay {R}{R}.
    Payment,
    /// Payment succeeded and the copy exists on the stack; decide whether
    /// to choose a target again or keep the inherited target.
    Retarget,
    /// The player chose to retarget and must now select one legal target.
    Target,
}

/// Durable continuation for a spell-copy choice made during resolution.
/// `resolving_source` plus `resolving_source_zone_change_count` identifies
/// the exact still-live parent stack incarnation; `copy_source` is a fresh
/// virtual arena object once payment succeeds.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PendingSpellCopy {
    /// Exact parent stack incarnation whose resolution is suspended.
    #[serde(default)]
    pub resolving_stack_item: StackItemId,
    pub resolving_source: ObjectId,
    pub resolving_source_zone_change_count: u32,
    pub player: PlayerId,
    pub inherited_target: Target,
    /// Exact cast-time target provenance inherited from the resolving
    /// parent. Kept separate from the convenience `Target` so a restored
    /// continuation cannot redirect a copy while retaining another
    /// target's identity contract.
    #[serde(default)]
    pub inherited_target_contract: Option<StackTargetContractV4>,
    pub stage: SpellCopyStage,
    pub copy_source: Option<ObjectId>,
    /// Exact virtual copy incarnation allocated after payment.
    #[serde(default)]
    pub copy_stack_item: Option<StackItemId>,
}

/// The answer to a `Decision::ChooseOptionalCost`. Declining is always
/// legal (matches `DoIfCostPaid`'s optional "may" framing); the other two
/// are only legal when the matching `PendingOptionalCost` field is true.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OptionalCostChoice {
    Decline,
    Discard,
    SacrificeLand,
}

/// This turn's combat. Reset at every `Step::BeginCombat`. An attacker
/// with no entry in `blocked_by` is unblocked.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CombatState {
    pub attackers_declared: bool,
    pub blockers_declared: bool,
    pub attackers: Vec<ObjectId>,
    /// Attacker -> blockers, preserving the defender's declaration order.
    /// XMage's `ComputerPlayerRL.selectBlockers` calls `declareBlocker` in
    /// its sequential-pick order, `CombatGroup.blockerDamage` walks that
    /// insertion order, and its default `Outcome.Damage` allocation assigns
    /// lethal from first to last. The replay trace's ordered
    /// `chosen_indices` therefore already carries the reference's effective
    /// damage-assignment order; sorting these ids changes combat outcomes.
    /// This is reference-AI behavior, not a rules-level ordering guarantee;
    /// a future surface can carry an explicit damage allocation instead.
    pub blocked_by: Vec<(ObjectId, Vec<ObjectId>)>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Decision {
    CastSpellOrPass {
        player: PlayerId,
        /// Hand-castable spells, cards with one unambiguous graveyard cast
        /// method (`CardDef::flashback` or `CardDef::escape`), and exiled
        /// Plotted cards castable for free (`CardDef::plot_cost`, cast on a
        /// later turn); `step()` tells them apart by zone and definition.
        castable_spells: Vec<ObjectId>,
        mana_abilities: Vec<ObjectId>,
        land_drops: Vec<ObjectId>,
        /// (source, ability_index) pairs for every non-mana activated
        /// ability currently affordable (Masked Meower's, the Blood
        /// token's).
        activatable_abilities: Vec<(ObjectId, u8)>,
        /// Hand cards with `CardDef::plot_cost` currently affordable at
        /// sorcery speed (Highway Robbery's `PlotAbility`, a `SpecialAction`
        /// -- see `Action::PlotSpell`). Disjoint from `castable_spells`:
        /// Plotting is a separate action from casting, offered alongside it
        /// in the same priority window (the real trace's "Plot {1}{R}"
        /// candidate, distinct from "Cast Highway Robbery").
        plot_actions: Vec<ObjectId>,
    },
    ChooseTargets {
        player: PlayerId,
        /// The spell or (non-mana-ability) source this targeting belongs
        /// to.
        spell: ObjectId,
        remaining: u8,
        legal_targets: Vec<Target>,
        /// Whether the active cast may legally stop before selecting its
        /// maximum number of targets. Mandatory targeting and spell-copy
        /// retargeting always set this false.
        can_finish: bool,
    },
    /// A cost component whose payment requires choosing WHICH objects pay
    /// it, not merely how many -- permanent sacrifices, Escape's choice
    /// of other graveyard cards, and typed permanent taps such as Heap
    /// Gate's additional activation cost. Fireblast's and Lava Dart's sacrifice
    /// choices were previously auto-solved silently
    /// by `sacrifice_lowest_id_lands`'s tapped-status heuristic with no
    /// `Decision` at all; the reference logs a real `SELECT_TARGETS`
    /// record for this pick (increment 11 characterization), so it's a
    /// real decision here too. Asked one object at a time (`remaining`
    /// counts down, `candidates` excludes whatever was already picked for
    /// this same cost) -- same shape as `ChooseTargets`'s `remaining`/
    /// `legal_targets`. A single pick auto-resolves only when its current
    /// candidate pool has at most one object; an aggregate-forced multi-pick
    /// still exposes every step whose current pool contains two or more.
    ChooseCostTargets {
        player: PlayerId,
        /// The spell (normal, Flashback, or Escape cast) this cost belongs to.
        source: ObjectId,
        cost_kind: CostKind,
        remaining: u8,
        candidates: Vec<ObjectId>,
    },
    /// Fireblast only, this increment: whether to pay its printed mana
    /// cost or sacrifice 2 Mountains instead. Only asked when *both* are
    /// currently legal (601.2b) -- if just one is affordable, that one is
    /// silently used, no decision.
    ChooseCastMode {
        player: PlayerId,
        spell: ObjectId,
        options: Vec<CastMode>,
    },
    /// Goblin Bushwhacker only, this increment: whether to pay its Kicker
    /// {R} on top of its printed cost (`card_def::CardDef::kicker_cost`).
    /// Only asked when the *combined* cost is currently affordable
    /// (`mana::can_pay_combined`) -- same "no real choice" shortcut as
    /// `ChooseCastMode` when kicking isn't even payable, silently resolved
    /// to unkicked instead. Unlike `ChooseCastMode`, kicking is never
    /// mandatory just because it's affordable -- declining is always legal.
    ChooseKicker {
        player: PlayerId,
        spell: ObjectId,
    },
    /// The four Blast cards: which of the spell's 2 modes to
    /// resolve (`CardDef::mode2`). Asked before targeting, since the modes
    /// have different `TargetSpec`s, and only when both modes admit a
    /// complete mandatory target assignment. If exactly one mode is viable
    /// it is selected silently while retaining its original index.
    ChooseSpellMode {
        player: PlayerId,
        spell: ObjectId,
        /// Total number of printed modes, preserving each mode's stable
        /// printed index even when only a sparse subset is currently legal.
        mode_count: u8,
        /// Exact printed mode indices whose targeting and payment can be
        /// completed in the live state.
        legal_modes: Vec<u8>,
    },
    /// A generic resumable effect program yielded a branch choice during
    /// resolution. `option_count` preserves the printed/program order; one
    /// legal action is emitted per zero-based option index.
    ChooseEffectOption {
        player: PlayerId,
        source: ObjectId,
        option_count: u16,
    },
    /// A generic resumable effect is selecting public targets/cards one at a
    /// time. Winding Way uses the exact-count ordered form for one
    /// graveyard batch; `can_finish` is reserved for variable-count generic
    /// selections and is false throughout that ordering flow.
    ChooseEffectTargets {
        player: PlayerId,
        source: ObjectId,
        selected_count: u16,
        min_targets: u16,
        max_targets: u16,
        legal_targets: Vec<Target>,
        can_finish: bool,
    },
    /// Highway Robbery only, this increment: a resolution-time optional
    /// cost (`effect::EffectOp::MayPayCostThen`). Always a real choice with
    /// at least 2 options (`Decline` plus whichever of `Discard`/
    /// `SacrificeLand` `PendingOptionalCost` marked payable) -- declining
    /// is always legal, so this is never auto-resolved for "no real
    /// option" the way `CastSpellOrPass` is.
    ChooseOptionalCost {
        player: PlayerId,
        discard_payable: bool,
        sacrifice_payable: bool,
    },
    /// Chain Lightning's affected player may pay {R}{R} to create the next
    /// link. The offer only exists when a concrete payment plan is available.
    ChooseSpellCopyPayment {
        player: PlayerId,
        spell: ObjectId,
    },
    /// Payment succeeded and a fresh spell-copy object is already on the
    /// stack. The player may keep its inherited target or choose again.
    ChooseSpellCopyRetarget {
        player: PlayerId,
        copy: ObjectId,
    },
    /// Fiery Temper only, this increment: whether to cast a just-discarded
    /// Madness card for its madness cost (`CardDef::madness_cost`) instead
    /// of letting it go to the graveyard. Unconditionally asked, with no
    /// affordability pre-check -- see `apply_choose_madness_cast`'s doc.
    /// Asked at *resolution* time (both players have passed priority with
    /// this card's Madness offer on top of the stack -- `advance_until_
    /// decision`'s `top.madness_offer` check), not at discard time: the
    /// offer is a real triggered ability (`state::StackItem::
    /// madness_offer`) that sits through normal priority like anything
    /// else on the stack, so other decisions (an opponent's instant, this
    /// same player's own mana ability) can genuinely happen first.
    ChooseMadnessCast {
        player: PlayerId,
        card: ObjectId,
    },
    /// Choose exactly `count` cards from `choices` to discard. Backs
    /// cleanup's discard-to-7, Faithless Looting's "draw two, then
    /// discard two", and the discard costs of Grab the Prize / Masked
    /// Meower / the Blood token. Only asked when `choices.len() >
    /// count` (otherwise there's no real choice -- discard everything,
    /// silently).
    Discard {
        player: PlayerId,
        count: u32,
        choices: Vec<ObjectId>,
    },
    /// 508.1: choose a (possibly empty) subset of `eligible` to attack
    /// with. Always asked whenever `Step::DeclareAttackers` is reached --
    /// `eligible` itself can be empty (no creature able to attack), and
    /// this is still asked rather than auto-passed: 508.1 makes Declare
    /// Attackers a turn-based action that always happens, even to declare
    /// zero attackers (see `advance_step`'s doc). Callers that mirror a
    /// reference implementation which *does* skip logging an empty-eligible
    /// window (as the Java harness does) need their own silent-auto-resolve
    /// handling for that case -- this decision itself makes no such
    /// distinction.
    DeclareAttackers {
        player: PlayerId,
        eligible: Vec<ObjectId>,
    },
    /// 509.1: choose blocks. `legal_blockers` is given per attacker
    /// (flying/reach constrain which of the defending player's untapped
    /// creatures may block which attacker -- see
    /// `card_def::Keywords::FLYING`/`REACH`). Always asked whenever
    /// `attackers` is non-empty, even if `legal_blockers` is empty for
    /// every attacker.
    DeclareBlockers {
        player: PlayerId,
        attackers: Vec<ObjectId>,
        legal_blockers: Vec<(ObjectId, Vec<ObjectId>)>,
    },
    /// Stub per the design brief: fixed APNAP grouping always happens;
    /// this decision only fires when one player controls 2+ simultaneous
    /// triggers and must choose an order for them (603.3b). No card in
    /// this increment's pool triggers 2+ simultaneously, so it's
    /// unreachable from either acceptance test; see
    /// `tests::order_triggers_decision_exists` for a synthetic proof it
    /// works.
    OrderTriggers {
        player: PlayerId,
        pending: Vec<PendingTrigger>,
    },
    GameOver {
        winner: Option<PlayerId>,
    },
    /// Terminal, same as `GameOver`: the walk hit a resolution requiring a
    /// mechanic this kernel has no representation for (see
    /// `EngineState::halted`'s doc). There is no `Action` that answers
    /// this -- once returned, this is the only decision `advance_until_
    /// decision` will ever produce again for this `GameState`.
    Halted {
        mechanic: UnsupportedMechanic,
        source: ObjectId,
    },
    /// A generic resumable effect yielded a Boolean choice. Appended after
    /// every pre-existing decision variant for identity stability; schema-v4
    /// already reserves the corresponding Boolean/Shuffle projection.
    ChooseEffectBoolean {
        player: PlayerId,
        source: ObjectId,
        default: Option<bool>,
        purpose: effect::EffectBooleanChoicePurpose,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    PlayLand(ObjectId),
    CastSpell(ObjectId),
    ActivateManaAbility(ObjectId),
    ActivateAbility(ObjectId, u8),
    Pass,
    ChooseTarget(Target),
    /// Answers one `Decision::ChooseCostTargets` pick (one object at a time
    /// -- see that decision's doc).
    ChooseCostTarget(ObjectId),
    ChooseCastMode(CastMode),
    /// Answers a `Decision::ChooseKicker`: `true` pays Kicker on top of the
    /// base cost, `false` declines.
    ChooseKicker(bool),
    Discard(Vec<ObjectId>),
    DeclareAttackers(Vec<ObjectId>),
    /// (blocker, attacker) pairs. A blocker may appear at most once; an
    /// attacker may appear multiple times (gang-blocked).
    DeclareBlockers(Vec<(ObjectId, ObjectId)>),
    /// Indices into the current `OrderTriggers` decision's `pending`, in
    /// the order they should be placed on the stack (last index resolves
    /// first -- stack is LIFO).
    OrderTriggers(Vec<usize>),
    /// Plots a hand card (`PlotAbility`, a `SpecialAction`): pays
    /// `CardDef::plot_cost`, exiles it, and marks it castable for free on a
    /// later turn. Doesn't use the stack and doesn't pass priority (605.3b-
    /// like: same shape as `PlayLand`).
    PlotSpell(ObjectId),
    ChooseSpellMode(u8),
    /// Answers `Decision::ChooseEffectOption` with a zero-based program
    /// option index.
    ChooseEffectOption(u16),
    /// Answers one target/card pick in `Decision::ChooseEffectTargets`.
    ChooseEffectTarget(Target),
    /// Completes a variable-count generic effect selection once its minimum
    /// has been met. Exact graveyard ordering auto-completes its forced last
    /// card and never offers this action.
    FinishEffectSelection,
    ChooseOptionalCost(OptionalCostChoice),
    /// Answers `Decision::ChooseSpellCopyPayment`.
    ChooseSpellCopyPayment(bool),
    /// Answers `Decision::ChooseSpellCopyRetarget`.
    ChooseSpellCopyRetarget(bool),
    /// `true` = cast the pending Madness card for its madness cost; `false`
    /// = let it go to the graveyard.
    ChooseMadnessCast(bool),
    /// Answers one stage of `HarnessSurfaceV2`'s `ChooseOptionalCost`
    /// reshape (see that module's `OptionalCostReshape`): the H2 surface
    /// splits the engine's one-shot, 3-way `Decision::ChooseOptionalCost`
    /// into a binary "use the cost at all?" gate, then (only when *both*
    /// sub-costs are payable) a second binary "which one?" pick -- matching
    /// Java's real two-`chooseUse`-calls shape (`DoIfCostPaid.apply`'s own
    /// gate, then `OrCost.pay`'s `usable.size() == 2` gate). `true`/`false`
    /// means "yes"/"no" at the gate stage, or "the first/second payable
    /// option" at the which stage. Presentation-only: never reaches this
    /// module's own `step` dispatch under normal use --
    /// `HarnessSurfaceV2::apply` always intercepts and resolves it into the
    /// real, single `Action::ChooseOptionalCost` once the reshape completes
    /// (see `surface_v2.rs`'s module for the full contract). The stub arm in
    /// `step` below exists only to fail loudly if that interception is ever
    /// bypassed, not as a supported direct-to-engine call.
    ChooseOptionalCostStage(bool),
    /// Answers `Decision::ChooseEffectBoolean`. Appended to preserve the
    /// ordering of every pre-existing engine action variant.
    ChooseEffectBoolean(bool),
    /// Activates one explicitly selected printed mana ability on a source
    /// with multiple choices. Single-color sources retain the historical
    /// `ActivateManaAbility` variant and therefore retain its action identity.
    ActivateManaAbilityChoice(ObjectId, ManaColor),
    /// Activates a mana ability whose cost includes one explicitly selected
    /// battlefield object, currently Saruli Caretaker's other untapped
    /// controlled creature. Appended for action identity stability.
    ActivateManaAbilityWithCostTarget(ObjectId, ManaColor, ObjectId),
}

const CHAIN_COPY_COST: Cost = Cost {
    pips: &[
        mana::Pip::Colored(mana::ManaColor::R),
        mana::Pip::Colored(mana::ManaColor::R),
    ],
    generic: 0,
    x_count: 0,
};

/// Whether the player in the live Chain Lightning payment continuation can
/// currently pay {R}{R}. The engine still presents the payment decision even
/// when this is false, matching XMage's effect order; parity drivers use this
/// query only to recognize the reference logger's silent, outcome-equivalent
/// unpayable prompt.
pub fn pending_spell_copy_cost_is_payable(state: &GameState) -> bool {
    state
        .engine
        .pending_spell_copy
        .as_ref()
        .is_some_and(|pending| {
            pending.stage == SpellCopyStage::Payment
                && mana::can_pay(&CHAIN_COPY_COST, 0, pending.player, state).is_some()
        })
}

const STEP_ORDER: [Step; 12] = [
    Step::Untap,
    Step::Upkeep,
    Step::Draw,
    Step::Main1,
    Step::BeginCombat,
    Step::DeclareAttackers,
    Step::DeclareBlockers,
    Step::CombatDamage,
    Step::EndCombat,
    Step::Main2,
    Step::End,
    Step::Cleanup,
];

fn step_grants_priority(step: Step) -> bool {
    !matches!(step, Step::Untap | Step::Cleanup)
}

fn target_count(spec: TargetSpec) -> u8 {
    match spec {
        TargetSpec::None => 0,
        TargetSpec::AnyTarget
        | TargetSpec::AnyPlayer
        | TargetSpec::AnySpellOnStack
        | TargetSpec::InstantSpellOnStack
        | TargetSpec::BlueSpellOnStack
        | TargetSpec::RedSpellOnStack
        | TargetSpec::AnyPermanent
        | TargetSpec::BluePermanent
        | TargetSpec::RedPermanent
        | TargetSpec::NonlandPermanent
        | TargetSpec::Creature
        | TargetSpec::NonlegendaryCreature
        | TargetSpec::ArtifactOrEnchantmentSpellOnStack
        | TargetSpec::SorcerySpellOnStack
        | TargetSpec::NoncreatureSpellOnStack
        | TargetSpec::ArtifactSpellOnStack
        | TargetSpec::ArtifactPermanent
        | TargetSpec::CreatureOrLandCardInGraveyard
        | TargetSpec::ControlledCreature
        | TargetSpec::EnchantmentPermanent
        | TargetSpec::CreatureCardInOwnGraveyard
        | TargetSpec::TargetOpponent
        | TargetSpec::OpponentControlledCreature
        | TargetSpec::SpellManaValueAtMostControlledSubtypes { .. }
        | TargetSpec::UpToOneTappedCreature
        | TargetSpec::NoncreatureArtifactPermanent
        | TargetSpec::Land
        | TargetSpec::OpponentArtifactOrEnchantmentPermanent
        | TargetSpec::CreatureOtherThanSource => 1,
        TargetSpec::PlayerThenTheirCreature
        | TargetSpec::UpToTwoCreatureCardsInOwnGraveyard
        | TargetSpec::UpToTwoCreatures
        | TargetSpec::ExactlyTwoArtifactPermanents
        | TargetSpec::UpToTwoPlayers
        | TargetSpec::UpToTwoCardsInGraveyards => 2,
    }
}

fn target_min_count(spec: TargetSpec) -> u8 {
    match spec {
        TargetSpec::UpToTwoCreatureCardsInOwnGraveyard
        | TargetSpec::UpToTwoCreatures
        | TargetSpec::UpToTwoPlayers
        | TargetSpec::UpToTwoCardsInGraveyards
        | TargetSpec::UpToOneTappedCreature => 0,
        _ => target_count(spec),
    }
}

fn target_cardinality_is_complete(spec: TargetSpec, count: usize) -> bool {
    count >= usize::from(target_min_count(spec)) && count <= usize::from(target_count(spec))
}

fn pending_cast_targeting_is_complete(pending: &PendingCast, spec: TargetSpec) -> bool {
    let count = pending.targets_chosen.len();
    let min = usize::from(target_min_count(spec));
    let max = usize::from(target_count(spec));
    if min == max {
        count == max && !pending.target_selection_finished
    } else {
        count == max || (pending.target_selection_finished && count >= min && count < max)
    }
}

fn is_color(state: &GameState, id: ObjectId, color: mana::ManaColor) -> bool {
    let def_idx = state.objects.get(id).card_def;
    card_def::CARD_DEFS[def_idx as usize]
        .colors
        .contains(&color)
}

fn target_contract_matches_live(
    state: &GameState,
    target: Target,
    contract: StackTargetContractV4,
) -> bool {
    if contract.target() != target {
        return false;
    }
    match contract {
        StackTargetContractV4::Player(_) => true,
        StackTargetContractV4::Object {
            object,
            card_def,
            owner,
            zone,
            zone_change_count,
            spell_copy_origin,
            ..
        } => state.objects.try_get(object).is_some_and(|live| {
            live.card_def == card_def
                && live.owner == owner
                && live.zone == zone
                && live.zone_change_count == zone_change_count
                && live.spell_copy_origin == spell_copy_origin
        }),
    }
}

fn target_contracts_are_structurally_valid(
    state: &GameState,
    targets: &[Target],
    contracts: &[StackTargetContractV4],
) -> bool {
    targets.len() == contracts.len()
        && targets
            .iter()
            .copied()
            .zip(contracts.iter().copied())
            .all(|(target, contract)| {
                if contract.target() != target {
                    return false;
                }
                match contract {
                    StackTargetContractV4::Player(_) => true,
                    StackTargetContractV4::Object {
                        object,
                        card_def,
                        owner,
                        controller,
                        zone,
                        zone_change_count,
                        spell_copy_origin,
                    } => state.objects.try_get(object).is_some_and(|live| {
                        live.card_def == card_def
                            && live.owner == owner
                            && live.controller == controller
                            && live.zone == zone
                            && live.zone_change_count == zone_change_count
                            && live.spell_copy_origin == spell_copy_origin
                    }),
                }
            })
}

/// Validates the immutable source fields of a spell stack item. This helper
/// deliberately does not require current stack-vector membership: departure
/// planning also runs after a resolver has popped the item but before its
/// source leaves `Zone::Stack`. [`validate_spell_stack_source`] adds the live
/// unique-membership requirement at every external stack/RL boundary.
fn validate_spell_copy_origin_chain(
    state: &GameState,
    child: ObjectId,
    child_card_def: u16,
    origin: SpellCopyOriginV4,
) -> Result<(), String> {
    if origin.parent == child || origin.parent.0 >= child.0 {
        return Err("spell copy origin does not name an earlier distinct arena object".to_string());
    }
    let parent = state
        .objects
        .try_get(origin.parent)
        .ok_or("spell copy parent object no longer exists")?;
    if child_card_def != origin.parent_card_def
        || parent.card_def != origin.parent_card_def
        || parent.owner != origin.parent_owner
        || origin.parent_was_copy != parent.spell_copy_origin.is_some()
        || if origin.parent_was_copy {
            parent.zone_change_count != origin.parent_stack_zone_change_count
                || parent.zone != Zone::Stack
                || parent.controller != origin.parent_controller
        } else {
            parent.zone_change_count < origin.parent_stack_zone_change_count
                || (parent.zone_change_count == origin.parent_stack_zone_change_count
                    && (parent.zone != Zone::Stack
                        || parent.controller != origin.parent_controller))
        }
    {
        return Err("spell copy origin no longer identifies its exact parent ancestry".to_string());
    }
    if let Some(parent_origin) = parent.spell_copy_origin {
        validate_spell_copy_origin_chain(state, origin.parent, parent.card_def, parent_origin)?;
    }
    Ok(())
}

/// Binds a physical spell's mutable stack metadata to independently frozen
/// evidence on the source object's current Stack incarnation. The route is
/// captured before 601.2a moves the card, while `finalized_method` is only
/// stamped after the selected definition-owned cost has actually committed.
fn validate_physical_spell_cast_origin(
    state: &GameState,
    item: &StackItem,
    contract: StackSourceContractV4,
    cast_method: CastMethodV4,
) -> Result<(), String> {
    let source = state.objects.get(item.source);
    let origin = source
        .v4
        .spell_cast_origin
        .ok_or("physical spell lost its independently frozen cast origin")?;
    if contract.spell_cast_origin != Some(origin)
        || contract.finalized_cast_binding != source.v4.finalized_cast_binding
        || origin
            .origin_zone_change_count
            .checked_add(1)
            .is_none_or(|stack_generation| stack_generation != source.zone_change_count)
    {
        return Err(
            "physical spell cast origin no longer identifies its stack incarnation".to_string(),
        );
    }
    let is_pending_placeholder = state
        .engine
        .pending_cast
        .as_ref()
        .is_some_and(|pending| pending.spell == item.source);
    if if is_pending_placeholder {
        origin.finalized_method.is_some()
    } else {
        origin.finalized_method != Some(cast_method)
    } {
        return Err("spell cast method disagrees with its paid cast-origin evidence".to_string());
    }

    let def = card_def::CARD_DEFS
        .get(source.card_def as usize)
        .ok_or("spell cast-origin definition is missing")?;
    match (is_pending_placeholder, source.v4.finalized_cast_binding) {
        (true, None) => {}
        (true, Some(_)) => {
            return Err("pending spell already carries finalized dynamic choices".to_string())
        }
        (false, Some(binding)) => {
            if binding.x_value != item.v4.x_value {
                return Err("spell X disagrees with its finalized source binding".to_string());
            }
            if has_chosen_creature_additional_cost(def) {
                let Some(chosen) = binding.chosen_creature_cost else {
                    return Err(
                        "chosen-creature cost lost its finalized source binding".to_string()
                    );
                };
                if item.v4.paid_cost_refs.as_slice() != [chosen] {
                    return Err(
                        "chosen-creature payment disagrees with its finalized source binding"
                            .to_string(),
                    );
                }
            } else if binding.chosen_creature_cost.is_some() {
                return Err("spell carries a foreign chosen-creature source binding".to_string());
            }
        }
        (false, None)
            if item.v4.x_value != 0
                || cast_method == CastMethodV4::Bestow
                || has_chosen_creature_additional_cost(def) =>
        {
            return Err("dynamic spell lost its finalized source binding".to_string())
        }
        (false, None) => {}
    }
    let route_supports_method = match origin.route {
        SpellCastRouteV4::Hand => {
            origin.origin_zone == Zone::Hand
                && source.owner == item.controller
                && matches!(
                    cast_method,
                    CastMethodV4::Normal
                        | CastMethodV4::Alternative
                        | CastMethodV4::Bestow
                        | CastMethodV4::Omen
                )
        }
        SpellCastRouteV4::GraveyardFlashback => {
            origin.origin_zone == Zone::Graveyard
                && source.owner == item.controller
                && cast_method == CastMethodV4::Flashback
                && def.flashback.is_some()
                && def.escape.is_none()
        }
        SpellCastRouteV4::ExilePermission {
            holder,
            permission_zone_change_count,
        } => {
            origin.origin_zone == Zone::Exile
                && holder == item.controller
                && permission_zone_change_count == origin.origin_zone_change_count
                && matches!(
                    cast_method,
                    CastMethodV4::Normal
                        | CastMethodV4::Alternative
                        | CastMethodV4::Bestow
                        | CastMethodV4::Omen
                )
                && state
                    .engine
                    .exile_play_permissions
                    .iter()
                    .any(|permission| {
                        permission.object == item.source
                            && permission.holder == holder
                            && permission.zone_change_generation == permission_zone_change_count
                            && permission.play_or_cast == PlayOrCast::Cast
                    })
        }
        SpellCastRouteV4::Plotted { plotted_turn } => {
            origin.origin_zone == Zone::Exile
                && source.owner == item.controller
                && cast_method == CastMethodV4::Plotted
                && def.plot_cost.is_some()
                && source.plotted_turn == Some(plotted_turn)
                && plotted_turn < state.turn
        }
        SpellCastRouteV4::Madness => {
            origin.origin_zone == Zone::Exile
                && source.owner == item.controller
                && cast_method == CastMethodV4::Madness
                && def.madness_cost.is_some()
        }
        SpellCastRouteV4::GraveyardEscape => {
            origin.origin_zone == Zone::Graveyard
                && source.owner == item.controller
                && cast_method == CastMethodV4::Escape
                && def.escape.is_some()
                && def.flashback.is_none()
        }
    };
    if !route_supports_method {
        return Err("spell cast method is incompatible with its exact pre-stack route".to_string());
    }
    Ok(())
}

fn validate_spell_source_contract_fields(
    state: &GameState,
    item: &StackItem,
) -> Result<(), String> {
    if item.kind != StackItemKind::Spell {
        if let Some(contract) = item.v4.source_contract {
            if item.v4.ability_source_contract.is_some() {
                return Err(
                    "nonspell stack item carries parallel source-provenance contracts".to_string(),
                );
            }
            return validate_spell_sourced_trigger(state, item, contract);
        }
        if item.v4.cast_method.is_some() || item.is_copy || item.is_flashback {
            return Err("nonspell stack item carries spell source provenance".to_string());
        }
        match item.kind {
            StackItemKind::TriggeredAbility | StackItemKind::ActivatedAbility => {
                let contract = item
                    .v4
                    .ability_source_contract
                    .ok_or("ability stack item lost its source incarnation")?;
                validate_ability_source_contract(state, item, contract)?;
            }
            StackItemKind::MadnessOffer => {
                if item.v4.ability_source_contract.is_some() {
                    return Err("Madness offer carries ability source provenance".to_string());
                }
                let source = state
                    .objects
                    .try_get(item.source)
                    .ok_or("nonspell stack source object no longer exists")?;
                // A spell-sourced trigger must carry the producer's exact
                // contract. Without it, reject a physical spell retyped into
                // an ability so it cannot bypass spell departure.
                if source.zone == Zone::Stack
                    || source.spell_copy_origin.is_some()
                    || source.v4.spell_cast_origin.is_some()
                {
                    return Err(
                        "nonspell stack item is sourced by a live spell incarnation".to_string()
                    );
                }
            }
            StackItemKind::Spell => unreachable!("handled above"),
        }
        return Ok(());
    }
    if item.v4.madness_source_contract.is_some()
        || item.v4.ability_source_contract.is_some()
        || item.v4.granted_by.is_some()
    {
        return Err("spell stack item carries nonspell source provenance".to_string());
    }
    if item.v4.hidden_ability_source.is_some() {
        return Err("spell stack item carries activated-ability source provenance".to_string());
    }
    if item.v4.ability_source_contract.is_some() {
        return Err("spell stack item carries ability source provenance".to_string());
    }
    let contract = item
        .v4
        .source_contract
        .ok_or("spell stack item lost its source contract")?;
    let source = state
        .objects
        .try_get(item.source)
        .ok_or("spell stack source object no longer exists")?;
    if contract.source != item.source
        || contract.card_def != source.card_def
        || contract.owner != source.owner
        || contract.controller != source.controller
        || contract.controller != item.controller
        || contract.zone != Zone::Stack
        || contract.zone != source.zone
        || contract.zone_change_count != source.zone_change_count
        || contract.spell_copy_origin != source.spell_copy_origin
        || contract.spell_cast_origin != source.v4.spell_cast_origin
        || contract.cast_method != item.v4.cast_method.unwrap_or_default()
    {
        return Err(
            "spell stack source contract no longer matches its exact incarnation".to_string(),
        );
    }
    let is_virtual_copy = source.spell_copy_origin.is_some();
    if item.is_copy != is_virtual_copy {
        return Err("spell copy flag disagrees with immutable source origin".to_string());
    }
    let cast_method = item
        .v4
        .cast_method
        .ok_or("spell stack item lost its cast method")?;
    let def = card_def::CARD_DEFS
        .get(source.card_def as usize)
        .ok_or("spell stack source definition is missing")?;
    if !def.is_executable() || !def.is_castable() {
        return Err("spell stack source is not an executable spell".to_string());
    }
    if let Some(kind) = item.v4.optional_additional_cost_paid {
        if def.optional_additional_cost != Some(kind) || is_virtual_copy {
            return Err(
                "spell optional additional-cost provenance is not definition-owned".to_string(),
            );
        }
        validate_optional_additional_paid_refs(kind, item.controller, &item.v4.paid_cost_refs)?;
    }
    match cast_method {
        CastMethodV4::Normal => {}
        CastMethodV4::Alternative if def.alt_cost.is_some() => {}
        CastMethodV4::Flashback if def.flashback.is_some() => {}
        CastMethodV4::Madness if def.madness_cost.is_some() => {}
        CastMethodV4::Escape if def.escape.is_some() => {}
        CastMethodV4::Omen if supported_omen(def).is_some() => {}
        CastMethodV4::Bestow if supported_bestow(def).is_some() => {}
        CastMethodV4::Plotted
            if def.plot_cost.is_some()
                && source
                    .plotted_turn
                    .is_some_and(|plotted_turn| plotted_turn < state.turn) => {}
        CastMethodV4::Alternative
        | CastMethodV4::Flashback
        | CastMethodV4::Madness
        | CastMethodV4::Plotted
        | CastMethodV4::Escape
        | CastMethodV4::Omen => {
            return Err("spell cast method is unsupported by its source definition".to_string())
        }
        CastMethodV4::Bestow => {
            return Err("spell stack item uses a malformed Bestow method".to_string())
        }
    }
    if is_virtual_copy {
        if item.is_flashback
            || cast_method != CastMethodV4::Normal
            || source.v4.spell_cast_origin.is_some()
        {
            return Err(
                "a supported virtual spell copy must carry normal, non-flashback provenance"
                    .to_string(),
            );
        }
        let origin = source
            .spell_copy_origin
            .expect("checked virtual-copy origin above");
        validate_spell_copy_origin_chain(state, item.source, source.card_def, origin)?;
    } else {
        if item.is_flashback != (cast_method == CastMethodV4::Flashback) {
            return Err("physical spell flashback flag disagrees with its cast method".to_string());
        }
        validate_physical_spell_cast_origin(state, item, contract, cast_method)?;
    }
    Ok(())
}

fn validated_ability_source_contract(
    state: &GameState,
    item: &StackItem,
) -> Result<AbilitySourceContractV4, String> {
    let contract = item
        .v4
        .ability_source_contract
        .ok_or("ability stack item lost its exact source contract")?;
    validate_historical_ability_source_contract(state, contract)?;
    if contract.source != item.source || contract.controller != item.controller {
        return Err("ability source contract disagrees with its stack item".to_string());
    }
    Ok(contract)
}

fn validate_historical_ability_source_contract(
    state: &GameState,
    contract: AbilitySourceContractV4,
) -> Result<(), String> {
    let live = state
        .objects
        .try_get(contract.source)
        .ok_or("ability source object no longer exists")?;
    if contract.card_def != live.card_def
        || contract.owner != live.owner
        || live.zone_change_count < contract.zone_change_count
        || (live.zone_change_count == contract.zone_change_count && live.zone != contract.zone)
        || live.spell_copy_origin.is_some()
    {
        return Err("ability source contract is structurally inconsistent".to_string());
    }
    Ok(())
}

fn validate_equipment_granted_trigger_contract(
    state: &GameState,
    host: AbilitySourceContractV4,
    equipment: AbilitySourceContractV4,
    controller: PlayerId,
    effect: &EffectOp,
) -> Result<(), String> {
    validate_historical_ability_source_contract(state, host)?;
    validate_historical_ability_source_contract(state, equipment)?;
    if host.source == equipment.source
        || host.controller != controller
        || host.zone != Zone::Battlefield
        || equipment.zone != Zone::Battlefield
        || equipment.attached_to
            != Some(ObjectLinkV4 {
                object: host.source,
                zone_change_count: host.zone_change_count,
            })
    {
        return Err("attachment-granted trigger source contract is inconsistent".to_string());
    }
    let profile = card_def::CARD_DEFS
        .get(equipment.card_def as usize)
        .and_then(|definition| definition.equipment)
        .ok_or("attachment-granted trigger producer is not Equipment")?;
    let expected = EffectOp::DealDamage {
        target: TargetRef::Opponent,
        amount: i32::from(profile.noncreature_spell_damage_to_each_opponent),
    };
    if profile.noncreature_spell_damage_to_each_opponent == 0 || *effect != expected {
        return Err("attachment-granted trigger no longer matches its producer".to_string());
    }
    Ok(())
}

/// Authenticates Weather the Storm's historical producing cast contract.
/// The source may still be the unique live spell or may have changed zones
/// after its independent Storm trigger was put on the stack.
fn storm_source_contract_is_structurally_valid(
    state: &GameState,
    contract: StackSourceContractV4,
) -> bool {
    let Some(source) = state.objects.try_get(contract.source) else {
        return false;
    };
    let Some(definition) = card_def::CARD_DEFS.get(contract.card_def as usize) else {
        return false;
    };
    let Some(origin) = contract.spell_cast_origin else {
        return false;
    };
    let route_is_valid = match origin.route {
        SpellCastRouteV4::Hand => {
            origin.origin_zone == Zone::Hand && contract.owner == contract.controller
        }
        SpellCastRouteV4::ExilePermission {
            holder,
            permission_zone_change_count,
        } => {
            origin.origin_zone == Zone::Exile
                && holder == contract.controller
                && permission_zone_change_count == origin.origin_zone_change_count
        }
        SpellCastRouteV4::GraveyardFlashback
        | SpellCastRouteV4::Plotted { .. }
        | SpellCastRouteV4::Madness
        | SpellCastRouteV4::GraveyardEscape => false,
    };
    definition.name == "Weather the Storm"
        && definition.is_executable()
        && definition.is_castable()
        && contract.zone == Zone::Stack
        && contract.cast_method == CastMethodV4::Normal
        && contract.spell_copy_origin.is_none()
        && origin.finalized_method == Some(CastMethodV4::Normal)
        && origin
            .origin_zone_change_count
            .checked_add(1)
            .is_some_and(|generation| generation == contract.zone_change_count)
        && route_is_valid
        && source.card_def == contract.card_def
        && source.owner == contract.owner
        && source.zone_change_count >= contract.zone_change_count
        && (source.zone_change_count != contract.zone_change_count
            || (source.zone == Zone::Stack
                && source.controller == contract.controller
                && source.spell_copy_origin == contract.spell_copy_origin
                && source.v4.spell_cast_origin == contract.spell_cast_origin))
}

fn storm_copy_count_from_binding(
    state: &GameState,
    binding: StormCopyBindingV1,
) -> Result<u16, String> {
    if binding.producing_stack_item == StackItemId::default()
        || binding.turn != state.turn
        || binding.active_player != state.active_player
        || !storm_source_contract_is_structurally_valid(state, binding.source_contract)
    {
        return Err("Storm cast binding is structurally malformed".to_string());
    }
    let event_index = usize::try_from(binding.spell_cast_event_index)
        .map_err(|_| "Storm event index does not fit this platform".to_string())?;
    if state.engine.event_history.get(event_index)
        != Some(&CommittedEvent::SpellCast {
            spell: binding.source_contract.source,
            controller: binding.source_contract.controller,
        })
    {
        return Err("Storm binding lost its exact SpellCast marker".to_string());
    }
    if state
        .engine
        .event_history
        .iter()
        .enumerate()
        .skip(event_index + 1)
        .any(|(_, event)| {
            matches!(
                event,
                CommittedEvent::SpellCast { spell, .. }
                    if *spell == binding.source_contract.source
            )
        })
    {
        return Err("Storm source was cast again after the bound cast marker".to_string());
    }

    let mut memberships = state
        .stack
        .iter()
        .filter(|item| item.v4.stack_item_id == binding.producing_stack_item);
    if let Some(producer) = memberships.next() {
        if memberships.next().is_some()
            || producer.kind != StackItemKind::Spell
            || producer.source != binding.source_contract.source
            || producer.v4.source_contract != Some(binding.source_contract)
        {
            return Err("Storm binding lost its unique producing spell".to_string());
        }
        validate_spell_source_contract_fields(state, producer)?;
        let exact_source_memberships = state
            .stack
            .iter()
            .filter(|item| {
                item.kind == StackItemKind::Spell && item.source == binding.source_contract.source
            })
            .count();
        let appears_outside_stack = [PlayerId::P0, PlayerId::P1].into_iter().any(|player| {
            let zones = &state.players[player.index()];
            zones.library.contains(&binding.source_contract.source)
                || zones.hand.contains(&binding.source_contract.source)
                || zones.battlefield.contains(&binding.source_contract.source)
                || zones.graveyard.contains(&binding.source_contract.source)
        }) || state.exile.contains(&binding.source_contract.source)
            || state.command.contains(&binding.source_contract.source);
        if exact_source_memberships != 1 || appears_outside_stack {
            return Err("Storm producing spell has malformed stack membership".to_string());
        }
    } else if state
        .objects
        .get(binding.source_contract.source)
        .zone_change_count
        <= binding.source_contract.zone_change_count
    {
        return Err(
            "Storm binding has neither its producer nor a historical departure".to_string(),
        );
    }

    let mut later_casts = [0_u16; 2];
    for event in state.engine.event_history.iter().skip(event_index + 1) {
        let CommittedEvent::SpellCast { controller, .. } = event else {
            continue;
        };
        later_casts[controller.index()] = later_casts[controller.index()]
            .checked_add(1)
            .ok_or("Storm later-cast count overflow")?;
    }
    for player in [PlayerId::P0, PlayerId::P1] {
        let expected = binding.casts_after_source[player.index()]
            .checked_add(later_casts[player.index()])
            .ok_or("Storm turn-cast count overflow")?;
        if state.players[player.index()].spells_cast_this_turn != expected {
            return Err("Storm binding disagrees with the exact later cast history".to_string());
        }
    }
    if binding.casts_after_source[binding.source_contract.controller.index()] == 0 {
        return Err("Storm binding does not include its own cast".to_string());
    }
    binding.casts_after_source[0]
        .checked_add(binding.casts_after_source[1])
        .and_then(|casts| casts.checked_sub(1))
        .ok_or_else(|| "Storm frozen copy count overflowed or omitted its cast".to_string())
}

pub(crate) fn materialize_storm_copy_binding(
    state: &GameState,
    source: ObjectId,
) -> Option<StormCopyBindingV1> {
    let mut producers = state
        .stack
        .iter()
        .filter(|item| item.kind == StackItemKind::Spell && item.source == source);
    let producer = producers.next()?;
    if producers.next().is_some() || validate_spell_stack_source(state, producer).is_err() {
        return None;
    }
    let source_contract = producer.v4.source_contract?;
    let spell_cast_event_index = state.engine.event_history.iter().rposition(|event| {
        *event
            == (CommittedEvent::SpellCast {
                spell: source,
                controller: producer.controller,
            })
    })?;
    if spell_cast_event_index + 1 != state.engine.event_history.len() {
        return None;
    }
    let binding = StormCopyBindingV1 {
        source_contract,
        producing_stack_item: producer.v4.stack_item_id,
        turn: state.turn,
        active_player: state.active_player,
        spell_cast_event_index: spell_cast_event_index.try_into().ok()?,
        casts_after_source: [
            state.players[0].spells_cast_this_turn,
            state.players[1].spells_cast_this_turn,
        ],
    };
    storm_copy_count_from_binding(state, binding).ok()?;
    Some(binding)
}

/// Authenticates the one supported nonspell shape whose physical source is
/// still on the stack: a definition-owned cast trigger. The trigger carries
/// the exact contract of its unique producing spell, and its inline program
/// must be one of that card's abilities that functions from the stack.
fn validate_spell_sourced_trigger(
    state: &GameState,
    item: &StackItem,
    contract: StackSourceContractV4,
) -> Result<(), String> {
    if item.kind != StackItemKind::TriggeredAbility
        || item.madness_offer
        || item.is_copy
        || item.is_flashback
        || item.kicked
        || item.inline_effect.is_none()
        || item.v4.cast_method.is_some()
        || item.v4.madness_source_contract.is_some()
        || item.v4.ability_source_contract.is_some()
        || item.v4.granted_by.is_some()
        || item.v4.face_index != 0
        || item.v4.x_value != 0
        || !item.v4.paid_cost_refs.is_empty()
        || item.v4.optional_additional_cost_paid.is_some()
        || item.v4.activated_ability_index.is_some()
    {
        return Err("spell-sourced trigger carries incompatible producer metadata".to_string());
    }
    if let Some(EffectOp::CreateStormCopies { binding }) = item.inline_effect.as_ref() {
        if contract != binding.source_contract
            || item.source != binding.source_contract.source
            || item.controller != binding.source_contract.controller
            || item.v4.stack_item_id == binding.producing_stack_item
        {
            return Err("Storm trigger changed its exact producing cast binding".to_string());
        }
        storm_copy_count_from_binding(state, *binding)?;
        return Ok(());
    }
    let source = state
        .objects
        .try_get(item.source)
        .ok_or("spell-sourced trigger source object is missing")?;
    // 113.7a: once put on the stack the trigger exists independently of the
    // spell. If that spell has since left the stack (countered, resolved),
    // the trigger resolves on last-known information.
    if contract.source == item.source
        && contract.controller == item.controller
        && contract.zone == Zone::Stack
        && (source.zone != Zone::Stack || source.zone_change_count != contract.zone_change_count)
    {
        return Ok(());
    }
    if contract.source != item.source
        || contract.card_def != source.card_def
        || contract.owner != source.owner
        || contract.controller != source.controller
        || contract.controller != item.controller
        || contract.zone != Zone::Stack
        || source.zone != Zone::Stack
        || contract.zone_change_count != source.zone_change_count
        || contract.spell_copy_origin != source.spell_copy_origin
        || contract.spell_cast_origin != source.v4.spell_cast_origin
    {
        return Err("spell-sourced trigger contract no longer matches its producer".to_string());
    }
    let mut producers = state.stack.iter().filter(|candidate| {
        candidate.kind == StackItemKind::Spell && candidate.source == item.source
    });
    let producer = producers
        .next()
        .ok_or("spell-sourced trigger has no live producing spell")?;
    if producers.next().is_some() || producer.v4.source_contract != Some(contract) {
        return Err("spell-sourced trigger lacks one exact producing spell".to_string());
    }
    validate_spell_source_contract_fields(state, producer)?;
    let definition_matches = crate::trigger::triggers_for(source.card_def)
        .iter()
        .filter(|trigger| trigger.home_zone == Zone::Stack)
        .any(|trigger| {
            crate::trigger::materialize_trigger_effect(trigger, item.source, state)
                == *item.inline_effect.as_ref().expect("checked Some above")
        });
    if !definition_matches {
        return Err("spell-sourced trigger no longer matches its card definition".to_string());
    }
    Ok(())
}

fn validate_ability_source_contract(
    state: &GameState,
    item: &StackItem,
    contract: AbilitySourceContractV4,
) -> Result<(), String> {
    let source = state
        .objects
        .try_get(item.source)
        .ok_or("ability source object no longer exists")?;
    if contract.source != item.source
        || contract.card_def != source.card_def
        || contract.owner != source.owner
        || contract.controller != item.controller
        || contract.zone == Zone::Stack
        || source.zone_change_count < contract.zone_change_count
        || (source.zone_change_count == contract.zone_change_count && source.zone != contract.zone)
    {
        // `attached_to` is deliberately not compared: an equipment can
        // become unattached (its creature died) while its equip ability is
        // on the stack, and the ability still resolves (113.7a).
        return Err("ability source contract is structurally malformed".to_string());
    }
    Ok(())
}

/// Validates a spell source at a live stack/RL boundary, including exact and
/// unique stack membership. Passing a detached clone, duplicate source, or a
/// source whose redundant flags were altered is malformed state.
pub(crate) fn validate_spell_stack_source(
    state: &GameState,
    item: &StackItem,
) -> Result<(), String> {
    validate_spell_source_contract_fields(state, item)?;
    if item.kind != StackItemKind::Spell {
        return Ok(());
    }
    let appears_outside_stack = [PlayerId::P0, PlayerId::P1].into_iter().any(|player| {
        let zones = &state.players[player.index()];
        zones.library.contains(&item.source)
            || zones.hand.contains(&item.source)
            || zones.battlefield.contains(&item.source)
            || zones.graveyard.contains(&item.source)
    }) || state.exile.contains(&item.source)
        || state.command.contains(&item.source);
    if appears_outside_stack {
        return Err("spell source is also indexed in an ordinary zone".to_string());
    }
    if item.v4.stack_item_id == StackItemId::default() {
        return Err("spell source has no stack-incarnation identity".to_string());
    }
    let mut matches = state
        .stack
        .iter()
        .filter(|candidate| candidate.v4.stack_item_id == item.v4.stack_item_id);
    let live = matches
        .next()
        .ok_or("spell source has no live stack membership")?;
    if matches.next().is_some() {
        return Err("spell source has duplicate stack-incarnation membership".to_string());
    }
    if live != item {
        return Err("spell source contract belongs to a different stack item".to_string());
    }
    if state
        .stack
        .iter()
        .filter(|candidate| {
            candidate.kind == StackItemKind::Spell && candidate.source == item.source
        })
        .count()
        != 1
    {
        return Err("spell source has duplicate physical-source membership".to_string());
    }
    for produced in state.stack.iter().filter(|candidate| {
        candidate.source == item.source && candidate.kind != StackItemKind::Spell
    }) {
        validate_spell_source_contract_fields(state, produced)?;
    }
    let source_contract = item
        .v4
        .source_contract
        .ok_or("spell stack item lost its source contract")?;
    if state
        .stack
        .iter()
        .filter(|candidate| {
            candidate.kind == StackItemKind::Spell
                && candidate.v4.source_contract == Some(source_contract)
        })
        .count()
        != 1
    {
        return Err("spell source contract has duplicate live spell membership".to_string());
    }
    Ok(())
}

fn battlefield_objects(state: &GameState) -> impl Iterator<Item = ObjectId> + '_ {
    [PlayerId::P0, PlayerId::P1]
        .into_iter()
        .flat_map(|p| state.players[p.index()].battlefield.iter().copied())
}

fn spell_targets_of_color(state: &GameState, color: mana::ManaColor) -> Vec<Target> {
    state
        .stack
        .iter()
        .filter(|item| item.kind == StackItemKind::Spell)
        .map(|item| item.source)
        .filter(|id| state.engine.pending_cast.as_ref().map(|p| p.spell) != Some(*id))
        .filter(|&id| is_color(state, id, color))
        .map(Target::Object)
        .collect()
}

fn permanent_targets_of_color(state: &GameState, color: mana::ManaColor) -> Vec<Target> {
    battlefield_objects(state)
        .filter(|&id| is_color(state, id, color))
        .map(Target::Object)
        .collect()
}

/// Stack-only card types after applying alternative spell characteristics.
/// Bestow is an Aura enchantment spell rather than a creature spell, while
/// Omen uses its definition-owned alternative type line. Other cast methods
/// retain the printed card types.
fn stack_spell_has_type(state: &GameState, item: &StackItem, card_type: CardType) -> bool {
    let def = &card_def::CARD_DEFS[state.objects.get(item.source).card_def as usize];
    match item.v4.cast_method {
        Some(CastMethodV4::Bestow) => card_type == CardType::Enchantment,
        Some(CastMethodV4::Omen) => {
            supported_omen(def).is_some_and(|omen| omen.types.contains(&card_type))
        }
        _ => def.has_type(card_type),
    }
}

/// Mana value of a spell on the stack. Alternative costs, including Bestow,
/// do not change mana value; announced X is substituted into each printed X
/// symbol in the card's mana cost.
fn stack_spell_mana_value(state: &GameState, item: &StackItem) -> u16 {
    let def = &card_def::CARD_DEFS[state.objects.get(item.source).card_def as usize];
    def.mana_value
        .saturating_add(item.v4.x_value.saturating_mul(u16::from(def.cost.x_count)))
}

fn spell_targets_with_any_type(state: &GameState, types: &[CardType]) -> Vec<Target> {
    let announcing = state
        .engine
        .pending_cast
        .as_ref()
        .map(|pending| pending.spell);
    state
        .stack
        .iter()
        .filter(|item| item.kind == StackItemKind::Spell && Some(item.source) != announcing)
        .filter(|item| {
            types
                .iter()
                .any(|&card_type| stack_spell_has_type(state, item, card_type))
        })
        .map(|item| Target::Object(item.source))
        .collect()
}

fn spell_targets_without_type(state: &GameState, excluded: CardType) -> Vec<Target> {
    let announcing = state
        .engine
        .pending_cast
        .as_ref()
        .map(|pending| pending.spell);
    state
        .stack
        .iter()
        .filter(|item| item.kind == StackItemKind::Spell && Some(item.source) != announcing)
        .filter(|item| !stack_spell_has_type(state, item, excluded))
        .map(|item| Target::Object(item.source))
        .collect()
}

fn permanent_targets_with_type(state: &GameState, card_type: CardType) -> Vec<Target> {
    battlefield_objects(state)
        .filter(|&id| object_has_type(state, id, card_type))
        .map(Target::Object)
        .collect()
}

/// `targets_chosen` is the *already-picked* prefix for this same targeting
/// pass -- needed for `PlayerThenTheirCreature`'s second, dependent pick
/// (any other spec's legal pool doesn't vary with what's already chosen, so
/// they simply ignore it).
pub fn legal_targets_for(
    spec: TargetSpec,
    targets_chosen: &[Target],
    state: &GameState,
) -> Vec<Target> {
    legal_targets_for_controller(spec, targets_chosen, state.priority_player, state)
}

#[derive(Debug, Clone, Copy)]
struct TargetingSource {
    object: ObjectId,
    card_def: u16,
}

fn targeting_source_for_object(state: &GameState, object: ObjectId) -> Option<TargetingSource> {
    state.objects.try_get(object).map(|live| TargetingSource {
        object,
        card_def: live.card_def,
    })
}

fn targeting_source_is_monocolored(source: TargetingSource) -> bool {
    card_def::CARD_DEFS
        .get(source.card_def as usize)
        .is_some_and(|def| (card_def::mana_colors_mask(def.colors) & 0x1f).count_ones() == 1)
}

fn legal_targets_for_controller(
    spec: TargetSpec,
    targets_chosen: &[Target],
    controller: PlayerId,
    state: &GameState,
) -> Vec<Target> {
    legal_targets_for_controller_from_source(spec, targets_chosen, controller, None, state)
}

fn legal_targets_for_controller_from_source(
    spec: TargetSpec,
    targets_chosen: &[Target],
    controller: PlayerId,
    source: Option<TargetingSource>,
    state: &GameState,
) -> Vec<Target> {
    let mut targets = match spec {
        TargetSpec::None => Vec::new(),
        TargetSpec::AnyPlayer => {
            vec![Target::Player(PlayerId::P0), Target::Player(PlayerId::P1)]
        }
        TargetSpec::UpToTwoPlayers => [PlayerId::P0, PlayerId::P1]
            .into_iter()
            .map(Target::Player)
            .filter(|target| !targets_chosen.contains(target))
            .collect(),
        TargetSpec::TargetOpponent => vec![Target::Player(controller.opponent())],
        TargetSpec::AnyTarget => {
            let mut out = vec![Target::Player(PlayerId::P0), Target::Player(PlayerId::P1)];
            for p in [PlayerId::P0, PlayerId::P1] {
                for &id in &state.players[p.index()].battlefield {
                    if object_has_type(state, id, CardType::Creature) {
                        out.push(Target::Object(id));
                    }
                }
            }
            out
        }
        TargetSpec::PlayerThenTheirCreature => {
            if targets_chosen.is_empty() {
                vec![Target::Player(PlayerId::P0), Target::Player(PlayerId::P1)]
            } else if let Some(Target::Player(p)) = targets_chosen.first() {
                state.players[p.index()]
                    .battlefield
                    .iter()
                    .copied()
                    .filter(|&id| object_has_type(state, id, CardType::Creature))
                    .map(Target::Object)
                    .collect()
            } else {
                Vec::new()
            }
        }
        TargetSpec::AnySpellOnStack => {
            let announcing = state.engine.pending_cast.as_ref().map(|p| p.spell);
            state
                .stack
                .iter()
                .filter(|item| item.kind == StackItemKind::Spell && Some(item.source) != announcing)
                .map(|item| Target::Object(item.source))
                .collect()
        }
        TargetSpec::InstantSpellOnStack => {
            let announcing = state.engine.pending_cast.as_ref().map(|p| p.spell);
            state
                .stack
                .iter()
                .filter(|item| item.kind == StackItemKind::Spell && Some(item.source) != announcing)
                .filter(|item| {
                    let def_idx = state.objects.get(item.source).card_def;
                    card_def::CARD_DEFS[def_idx as usize].has_type(CardType::Instant)
                })
                .map(|item| Target::Object(item.source))
                .collect()
        }
        TargetSpec::BlueSpellOnStack => spell_targets_of_color(state, mana::ManaColor::U),
        TargetSpec::RedSpellOnStack => spell_targets_of_color(state, mana::ManaColor::R),
        TargetSpec::ArtifactOrEnchantmentSpellOnStack => {
            spell_targets_with_any_type(state, &[CardType::Artifact, CardType::Enchantment])
        }
        TargetSpec::SorcerySpellOnStack => spell_targets_with_any_type(state, &[CardType::Sorcery]),
        TargetSpec::NoncreatureSpellOnStack => {
            spell_targets_without_type(state, CardType::Creature)
        }
        TargetSpec::ArtifactSpellOnStack => {
            spell_targets_with_any_type(state, &[CardType::Artifact])
        }
        TargetSpec::AnyPermanent => battlefield_objects(state).map(Target::Object).collect(),
        TargetSpec::BluePermanent => permanent_targets_of_color(state, mana::ManaColor::U),
        TargetSpec::RedPermanent => permanent_targets_of_color(state, mana::ManaColor::R),
        TargetSpec::NonlandPermanent => battlefield_objects(state)
            .filter(|&id| !object_has_type(state, id, CardType::Land))
            .map(Target::Object)
            .collect(),
        TargetSpec::Creature
        | TargetSpec::CreatureOtherThanSource
        | TargetSpec::UpToTwoCreatures => battlefield_objects(state)
            .filter(|&id| {
                object_has_type(state, id, CardType::Creature)
                    && (spec != TargetSpec::CreatureOtherThanSource
                        || source.is_none_or(|source| id != source.object))
            })
            .filter(|id| !targets_chosen.contains(&Target::Object(*id)))
            .map(Target::Object)
            .collect(),
        TargetSpec::NonlegendaryCreature => battlefield_objects(state)
            .filter(|&id| {
                let def_idx = state.objects.get(id).card_def;
                let def = &card_def::CARD_DEFS[def_idx as usize];
                object_has_type(state, id, CardType::Creature)
                    && !def.supertypes.contains(&card_def::Supertype::Legendary)
            })
            .map(Target::Object)
            .collect(),
        TargetSpec::ArtifactPermanent => permanent_targets_with_type(state, CardType::Artifact),
        TargetSpec::ExactlyTwoArtifactPermanents => {
            permanent_targets_with_type(state, CardType::Artifact)
                .into_iter()
                .filter(|target| !targets_chosen.contains(target))
                .collect()
        }
        TargetSpec::EnchantmentPermanent => {
            permanent_targets_with_type(state, CardType::Enchantment)
        }
        TargetSpec::CreatureOrLandCardInGraveyard => [PlayerId::P0, PlayerId::P1]
            .into_iter()
            .flat_map(|player| state.players[player.index()].graveyard.iter().copied())
            .filter(|&id| {
                let def = &card_def::CARD_DEFS[state.objects.get(id).card_def as usize];
                def.has_type(CardType::Creature) || def.has_type(CardType::Land)
            })
            .map(Target::Object)
            .collect(),
        TargetSpec::ControlledCreature => state
            .objects
            .iter()
            .filter_map(|(id, object)| {
                (object.controller == controller
                    && object.zone == Zone::Battlefield
                    && card_def::CARD_DEFS[object.card_def as usize].has_type(CardType::Creature))
                .then_some(id)
            })
            .map(Target::Object)
            .collect(),
        TargetSpec::UpToOneTappedCreature => battlefield_objects(state)
            .filter(|&id| {
                let object = state.objects.get(id);
                object.tapped && object_has_type(state, id, CardType::Creature)
            })
            .map(Target::Object)
            .collect(),
        TargetSpec::OpponentControlledCreature => state.players[controller.opponent().index()]
            .battlefield
            .iter()
            .copied()
            .filter(|&id| {
                let object = state.objects.get(id);
                object.controller == controller.opponent()
                    && object.zone == Zone::Battlefield
                    && object_has_type(state, id, CardType::Creature)
            })
            .map(Target::Object)
            .collect(),
        TargetSpec::OpponentArtifactOrEnchantmentPermanent => state.players
            [controller.opponent().index()]
        .battlefield
        .iter()
        .copied()
        .filter(|&id| {
            let object = state.objects.get(id);
            let definition = &card_def::CARD_DEFS[object.card_def as usize];
            object.zone == Zone::Battlefield
                && object.controller == controller.opponent()
                && (definition.has_type(CardType::Artifact)
                    || definition.has_type(CardType::Enchantment))
        })
        .map(Target::Object)
        .collect(),
        TargetSpec::SpellManaValueAtMostControlledSubtypes { first, second } => {
            let first = first.stable_id();
            let second = second.map(card_def::Subtype::stable_id);
            let controlled_count = state
                .objects
                .iter()
                .filter(|(_, object)| {
                    object.zone == Zone::Battlefield
                        && object.controller == controller
                        && (object
                            .v4
                            .effective_subtype_ids
                            .binary_search(&first)
                            .is_ok()
                            || second.is_some_and(|second| {
                                object
                                    .v4
                                    .effective_subtype_ids
                                    .binary_search(&second)
                                    .is_ok()
                            }))
                })
                .count() as u16;
            let announcing = state
                .engine
                .pending_cast
                .as_ref()
                .map(|pending| pending.spell);
            state
                .stack
                .iter()
                .filter(|item| item.kind == StackItemKind::Spell && Some(item.source) != announcing)
                .filter(|item| stack_spell_mana_value(state, item) <= controlled_count)
                .map(|item| Target::Object(item.source))
                .collect()
        }
        TargetSpec::Land => battlefield_objects(state)
            .filter(|&id| object_has_type(state, id, CardType::Land))
            .map(Target::Object)
            .collect(),
        TargetSpec::NoncreatureArtifactPermanent => battlefield_objects(state)
            .filter(|&id| {
                let definition = &card_def::CARD_DEFS[state.objects.get(id).card_def as usize];
                definition.has_type(CardType::Artifact) && !definition.has_type(CardType::Creature)
            })
            .map(Target::Object)
            .collect(),
        TargetSpec::UpToTwoCreatureCardsInOwnGraveyard => state.players[controller.index()]
            .graveyard
            .iter()
            .copied()
            .filter(|id| !targets_chosen.contains(&Target::Object(*id)))
            .filter(|&id| {
                let object = state.objects.get(id);
                object.owner == controller
                    && object.zone == Zone::Graveyard
                    && !object.v4.is_token
                    && card_def::CARD_DEFS[object.card_def as usize].has_type(CardType::Creature)
            })
            .map(Target::Object)
            .collect(),
        TargetSpec::UpToTwoCardsInGraveyards => [PlayerId::P0, PlayerId::P1]
            .into_iter()
            .flat_map(|player| state.players[player.index()].graveyard.iter().copied())
            .filter(|id| !targets_chosen.contains(&Target::Object(*id)))
            .filter(|&id| {
                let object = state.objects.get(id);
                object.zone == Zone::Graveyard && !object.v4.is_token
            })
            .map(Target::Object)
            .collect(),
        TargetSpec::CreatureCardInOwnGraveyard => state.players[controller.index()]
            .graveyard
            .iter()
            .copied()
            .filter(|&id| {
                let object = state.objects.get(id);
                object.owner == controller
                    && object.zone == Zone::Graveyard
                    && !object.v4.is_token
                    && card_def::CARD_DEFS[object.card_def as usize].has_type(CardType::Creature)
            })
            .map(Target::Object)
            .collect(),
    };
    targets.retain(|target| {
        let Target::Object(object) = *target else {
            return true;
        };
        let live = state.objects.get(object);
        live.zone != Zone::Battlefield
            || live.controller == controller
            || !has_effective_keyword(state, object, Keywords::HEXPROOF)
    });
    if source.is_some_and(targeting_source_is_monocolored) {
        targets.retain(|target| {
            let Target::Object(object) = target else {
                return true;
            };
            !has_effective_keyword(state, *object, Keywords::PROTECTION_FROM_MONOCOLORED)
        });
    }
    targets
}

/// Whether the already-chosen target prefix can be extended to a complete
/// mandatory assignment. This deliberately explores dependent specs
/// recursively: `PlayerThenTheirCreature` is viable only when at least one
/// first-pick player has a legal second-pick creature, not merely because
/// its unconditional first pool contains both players.
#[cfg(test)]
fn target_prefix_can_complete(
    spec: TargetSpec,
    targets_chosen: &[Target],
    state: &GameState,
) -> bool {
    target_prefix_can_complete_for_controller(spec, targets_chosen, state.priority_player, state)
}

pub(crate) fn target_prefix_can_complete_for_controller(
    spec: TargetSpec,
    targets_chosen: &[Target],
    controller: PlayerId,
    state: &GameState,
) -> bool {
    target_prefix_can_complete_for_controller_and_source(
        spec,
        targets_chosen,
        controller,
        None,
        state,
    )
}

fn target_prefix_can_complete_for_controller_and_source(
    spec: TargetSpec,
    targets_chosen: &[Target],
    controller: PlayerId,
    source: Option<TargetingSource>,
    state: &GameState,
) -> bool {
    let max = target_count(spec) as usize;
    if targets_chosen.len() > max {
        return false;
    }
    let mut validated_prefix = Vec::with_capacity(targets_chosen.len());
    for &target in targets_chosen {
        if !legal_targets_for_controller_from_source(
            spec,
            &validated_prefix,
            controller,
            source,
            state,
        )
        .contains(&target)
        {
            return false;
        }
        validated_prefix.push(target);
    }
    if targets_chosen.len() >= usize::from(target_min_count(spec)) {
        return true;
    }
    legal_targets_for_controller_from_source(spec, targets_chosen, controller, source, state)
        .into_iter()
        .any(|candidate| {
            let mut next = targets_chosen.to_vec();
            next.push(candidate);
            target_prefix_can_complete_for_controller_and_source(
                spec, &next, controller, source, state,
            )
        })
}

/// The next legal target choices that still admit a complete mandatory
/// assignment. Filtering at every prefix prevents a legal first pick from
/// leading to an empty dependent second-pick decision.
fn completable_next_targets_for(
    spec: TargetSpec,
    targets_chosen: &[Target],
    state: &GameState,
) -> Vec<Target> {
    completable_next_targets_for_controller(spec, targets_chosen, state.priority_player, state)
}

fn completable_next_targets_for_controller(
    spec: TargetSpec,
    targets_chosen: &[Target],
    controller: PlayerId,
    state: &GameState,
) -> Vec<Target> {
    completable_next_targets_for_controller_and_source(
        spec,
        targets_chosen,
        controller,
        None,
        state,
    )
}

fn completable_next_targets_for_controller_and_source(
    spec: TargetSpec,
    targets_chosen: &[Target],
    controller: PlayerId,
    source: Option<TargetingSource>,
    state: &GameState,
) -> Vec<Target> {
    legal_targets_for_controller_from_source(spec, targets_chosen, controller, source, state)
        .into_iter()
        .filter(|candidate| {
            let mut next = targets_chosen.to_vec();
            next.push(*candidate);
            target_prefix_can_complete_for_controller_and_source(
                spec, &next, controller, source, state,
            )
        })
        .collect()
}

fn activation_legal_targets_for(
    source: ObjectId,
    ability: &ActivatedAbilityDef,
    targets_chosen: &[Target],
    state: &GameState,
) -> Vec<Target> {
    let controller = state.objects.get(source).controller;
    legal_targets_for_controller_from_source(
        ability.target_spec,
        targets_chosen,
        controller,
        Some(TargetingSource {
            object: source,
            card_def: state.objects.get(source).card_def,
        }),
        state,
    )
    .into_iter()
    .filter(|target| match ability.activation_target_filter {
        ActivationTargetFilter::TargetSpecOnly => true,
        ActivationTargetFilter::CreatureBlockedBySource => {
            let Target::Object(attacker) = target else {
                return false;
            };
            state
                .engine
                .combat
                .blocked_by
                .iter()
                .any(|(candidate_attacker, blockers)| {
                    candidate_attacker == attacker && blockers.contains(&source)
                })
        }
    })
    .collect()
}

fn payable_activation_legal_targets_for(
    source: ObjectId,
    ability: &ActivatedAbilityDef,
    targets_chosen: &[Target],
    state: &GameState,
) -> Vec<Target> {
    let controller = state.objects.get(source).controller;
    activation_legal_targets_for(source, ability, targets_chosen, state)
        .into_iter()
        .filter(|target| {
            let Some(cost) = activation_mana_cost(ability.cost) else {
                return true;
            };
            if cost.x_count == 0 {
                return true;
            }
            let Target::Object(object) = target else {
                return false;
            };
            let Ok(x_value) = u8::try_from(
                card_def::CARD_DEFS[state.objects.get(*object).card_def as usize].mana_value,
            ) else {
                return false;
            };
            can_pay_activation_components_with_x(ability.cost, controller, source, state, x_value)
        })
        .collect()
}

fn activation_x_value(
    ability: &ActivatedAbilityDef,
    targets: &[Target],
    state: &GameState,
) -> Result<u8, String> {
    let Some(cost) = activation_mana_cost(ability.cost) else {
        return Ok(0);
    };
    if cost.x_count == 0 {
        return Ok(0);
    }
    if ability.target_spec != TargetSpec::NoncreatureArtifactPermanent {
        return Err("variable activation cost is not bound to an artifact target".to_string());
    }
    let Some(Target::Object(target)) = targets.first().copied() else {
        return Err("variable activation cost lost its target".to_string());
    };
    let live = state
        .objects
        .try_get(target)
        .ok_or("variable activation target is missing")?;
    u8::try_from(card_def::CARD_DEFS[live.card_def as usize].mana_value)
        .map_err(|_| "variable activation target mana value exceeds the payment range".to_string())
}

fn activation_target_prefix_can_complete(
    source: ObjectId,
    ability: &ActivatedAbilityDef,
    targets_chosen: &[Target],
    state: &GameState,
) -> bool {
    let max = usize::from(target_count(ability.target_spec));
    if targets_chosen.len() > max {
        return false;
    }
    let mut validated_prefix = Vec::with_capacity(targets_chosen.len());
    for &target in targets_chosen {
        if !payable_activation_legal_targets_for(source, ability, &validated_prefix, state)
            .contains(&target)
        {
            return false;
        }
        validated_prefix.push(target);
    }
    if targets_chosen.len() >= usize::from(target_min_count(ability.target_spec)) {
        return true;
    }
    payable_activation_legal_targets_for(source, ability, targets_chosen, state)
        .into_iter()
        .any(|candidate| {
            let mut next = targets_chosen.to_vec();
            next.push(candidate);
            activation_target_prefix_can_complete(source, ability, &next, state)
        })
}

fn completable_next_activation_targets_for(
    source: ObjectId,
    ability: &ActivatedAbilityDef,
    targets_chosen: &[Target],
    state: &GameState,
) -> Vec<Target> {
    payable_activation_legal_targets_for(source, ability, targets_chosen, state)
        .into_iter()
        .filter(|candidate| {
            let mut next = targets_chosen.to_vec();
            next.push(*candidate);
            activation_target_prefix_can_complete(source, ability, &next, state)
        })
        .collect()
}

/// Printed modal indices whose target requirements can be completed in the
/// current state. Omen uses the same public two-form decision, but its forms
/// also have distinct timing and costs and are filtered after announcement.
fn viable_printed_spell_modes(
    def: &card_def::CardDef,
    source: ObjectId,
    controller: PlayerId,
    state: &GameState,
) -> Vec<u8> {
    let mut modes =
        Vec::with_capacity(1 + usize::from(def.mode2.is_some()) + usize::from(def.mode3.is_some()));
    let source = targeting_source_for_object(state, source);
    if target_prefix_can_complete_for_controller_and_source(
        def.target_spec,
        &[],
        controller,
        source,
        state,
    ) {
        modes.push(0);
    }
    if let Some(mode2) = &def.mode2 {
        if target_prefix_can_complete_for_controller_and_source(
            mode2.target_spec,
            &[],
            controller,
            source,
            state,
        ) {
            modes.push(1);
        }
    }
    if let Some(mode3) = &def.mode3 {
        if target_prefix_can_complete_for_controller_and_source(
            mode3.target_spec,
            &[],
            controller,
            source,
            state,
        ) {
            modes.push(2);
        }
    }
    modes
}

// ------------------------------------------------------------------ query

/// A component of a non-mana cost that only makes sense in an already-
/// affordable / already-decided context (a `DiscardCards` component whose
/// actual discard already happened via `Decision::Discard`). Both
/// `can_pay_components` and `pay_cost_components` treat it as trivially
/// satisfied/a no-op; the real legality/payment lives in the
/// `pending_discard` staging.
fn discard_count_in(components: &[CostComponent]) -> Option<u8> {
    components.iter().find_map(|c| match c {
        CostComponent::DiscardCards(n) => Some(*n),
        _ => None,
    })
}

/// `pub(crate)`: also read by `effect::execute`'s `MayPayCostThen` handler
/// to check whether the sacrifice-a-land sub-cost is currently payable.
pub(crate) fn count_controlled_lands(player: PlayerId, state: &GameState) -> u32 {
    state.players[player.index()]
        .battlefield
        .iter()
        .filter(|&&id| card_def::CARD_DEFS[state.objects.get(id).card_def as usize].is_land)
        .count() as u32
}

/// Samples one reusable board-dependent value from current state. Effects
/// and mana abilities call this at resolution/activation time so intervening
/// battlefield changes are reflected exactly once.
pub(crate) fn evaluate_dynamic_value(
    state: &GameState,
    value: DynamicValueDef,
    controller: PlayerId,
) -> i32 {
    let count = match value {
        DynamicValueDef::Fixed(value) => return value,
        DynamicValueDef::BattlefieldPermanentsWithSubtype(subtype) => state
            .objects
            .iter()
            .filter(|(id, object)| {
                object.zone == Zone::Battlefield && has_effective_subtype(state, *id, subtype)
            })
            .count(),
        DynamicValueDef::ControllerGraveyardCardsWithType(card_type) => state.players
            [controller.index()]
        .graveyard
        .iter()
        .filter(|&&object| {
            let live = state.objects.get(object);
            live.owner == controller
                && live.zone == Zone::Graveyard
                && !live.v4.is_token
                && card_def::CARD_DEFS[live.card_def as usize].has_type(card_type)
        })
        .count(),
    };
    i32::try_from(count).expect("the object arena count fits the engine's signed value range")
}

fn activation_cost_object_candidates(
    player: PlayerId,
    source: ObjectId,
    subtype: card_def::Subtype,
    state: &GameState,
    already_chosen: &[ObjectId],
) -> Vec<ObjectId> {
    state.players[player.index()]
        .battlefield
        .iter()
        .copied()
        .filter(|candidate| *candidate != source && !already_chosen.contains(candidate))
        .filter(|&candidate| {
            let object = state.objects.get(candidate);
            object.controller == player
                && object.zone == Zone::Battlefield
                && !object.tapped
                && has_effective_subtype(state, candidate, subtype)
        })
        .collect()
}

fn activation_tap_cost_subtype(components: &[CostComponent]) -> Option<card_def::Subtype> {
    components.iter().find_map(|component| match component {
        CostComponent::TapOtherUntappedControlledPermanentWithSubtype(subtype) => Some(*subtype),
        _ => None,
    })
}

fn has_unblocked_attacker_return_cost(components: &[CostComponent]) -> bool {
    components.iter().any(|component| {
        matches!(
            component,
            CostComponent::ReturnControlledUnblockedAttackerToOwnersHand
        )
    })
}

fn ninjutsu_timing_ok(player: PlayerId, state: &GameState) -> bool {
    player == state.priority_player && ninjutsu_combat_window_open(player, state)
}

/// The post-blockers combat window ninjutsu needs, independent of who holds
/// priority. Activation additionally needs priority (above); resolution does
/// not -- when the ability resolves, priority sits with whichever player
/// passed last, usually the defending player.
fn ninjutsu_combat_window_open(player: PlayerId, state: &GameState) -> bool {
    player == state.active_player
        && state.engine.combat.blockers_declared
        && matches!(
            state.step,
            Step::DeclareBlockers | Step::CombatDamage | Step::EndCombat
        )
}

fn unblocked_attacker_return_candidates(
    player: PlayerId,
    state: &GameState,
    already_chosen: &[EffectObjectBinding],
) -> Vec<EffectObjectBinding> {
    if !ninjutsu_timing_ok(player, state) {
        return Vec::new();
    }
    state
        .engine
        .combat
        .attackers
        .iter()
        .copied()
        .filter(|attacker| {
            !state
                .engine
                .combat
                .blocked_by
                .iter()
                .any(|(blocked, _)| blocked == attacker)
        })
        .filter_map(|attacker| {
            let object = state.objects.try_get(attacker)?;
            let definition = card_def::CARD_DEFS.get(object.card_def as usize)?;
            (object.zone == Zone::Battlefield
                && object.controller == player
                && definition.has_type(CardType::Creature)
                && !already_chosen
                    .iter()
                    .any(|binding| binding.object == attacker))
            .then_some(EffectObjectBinding {
                object: attacker,
                expected_zone: Zone::Battlefield,
                expected_zone_change_count: object.zone_change_count,
            })
        })
        .collect()
}

fn activation_mana_cost(components: &[CostComponent]) -> Option<&Cost> {
    components.iter().find_map(|component| match component {
        CostComponent::Mana(cost) => Some(cost),
        _ => None,
    })
}

fn payable_activation_cost_object_candidates(
    player: PlayerId,
    source: ObjectId,
    components: &[CostComponent],
    state: &GameState,
) -> Vec<ObjectId> {
    let Some(subtype) = activation_tap_cost_subtype(components) else {
        return Vec::new();
    };
    activation_cost_object_candidates(player, source, subtype, state, &[])
        .into_iter()
        .filter(|candidate| {
            activation_mana_cost(components).is_none_or(|cost| {
                mana::can_pay_excluding_sources(cost, 0, player, state, &[source, *candidate])
                    .is_some()
            })
        })
        .collect()
}

/// Deterministic mana-solver bridge for resolution-time optional payments.
/// The owned color list lives in serializable `EffectOp` state; this helper
/// materializes the borrowed `Cost` shape used by the ordinary cast payer.
pub(crate) fn can_pay_effect_mana(
    player: PlayerId,
    colored: &[mana::ManaColor],
    generic: u8,
    state: &GameState,
) -> bool {
    let pips = colored
        .iter()
        .copied()
        .map(mana::Pip::Colored)
        .collect::<Vec<_>>();
    mana::can_pay_owned(&pips, generic, player, state).is_some()
}

/// Commits the exact deterministic plan preflighted by
/// `can_pay_effect_mana`. No priority or trigger checkpoint occurs here.
pub(crate) fn pay_effect_mana(
    player: PlayerId,
    colored: &[mana::ManaColor],
    generic: u8,
    state: &mut GameState,
) -> bool {
    let pips = colored
        .iter()
        .copied()
        .map(mana::Pip::Colored)
        .collect::<Vec<_>>();
    let Some(plan) = mana::can_pay_owned(&pips, generic, player, state) else {
        return false;
    };
    pay_plan(state, player, &plan);
    true
}

/// Whether the generic component payer can solve this exact ordered shape.
/// The current solver deliberately supports one component from each
/// interactive/resource family and requires mana to precede anything that
/// could change its available sources. Together with the concrete resource
/// checks in the offer/payment paths, this covers every generated cost in the
/// frozen pool, including Deep Analysis's `[Mana, PayLife]`, while failing
/// closed for a future multi-mana, repeated-discard/sacrifice, or
/// source-before-mana cost.
///
/// Fixed life and Phyrexian mana are also rejected in the same slice. The
/// mana solver may choose life for a Phyrexian pip, so checking the two
/// components independently could otherwise offer a cast whose combined life
/// payment is impossible.
fn component_payment_shape_supported(components: &[CostComponent]) -> bool {
    let mut mana = None;
    let mut saw_source_changing_component = false;
    let mut discard_count = 0;
    let mut sacrifice_lands_count = 0;
    let mut sacrifice_controlled_count = 0;
    let mut graveyard_exile_count = 0;
    let mut return_permanent_count = 0;
    let mut pay_life_count = 0;
    let mut tap_count = 0;
    let mut source_departure_count = 0;
    let mut tap_other_permanent_count = 0;
    let mut tap_filtered_permanent_count = 0;
    let mut reveal_hand_condition_count = 0;
    let mut chosen_creature_count = 0;

    for component in components {
        match component {
            CostComponent::Mana(cost) => {
                if mana.is_some() || saw_source_changing_component {
                    return false;
                }
                mana = Some(cost);
            }
            CostComponent::Tap => {
                tap_count += 1;
                saw_source_changing_component = true;
            }
            CostComponent::SacrificeSelf
            | CostComponent::ExileSelf
            | CostComponent::DiscardSelf => {
                source_departure_count += 1;
                saw_source_changing_component = true;
            }
            CostComponent::DiscardCards(amount) => {
                if *amount == 0 {
                    return false;
                }
                discard_count += 1;
            }
            CostComponent::SacrificeLands(amount) => {
                if *amount == 0 {
                    return false;
                }
                sacrifice_lands_count += 1;
                saw_source_changing_component = true;
            }
            CostComponent::SacrificeControlled { count, .. } => {
                if *count == 0 {
                    return false;
                }
                sacrifice_controlled_count += 1;
                saw_source_changing_component = true;
            }
            CostComponent::ExileOtherCardsFromOwnGraveyard(amount) => {
                if *amount == 0 {
                    return false;
                }
                graveyard_exile_count += 1;
                saw_source_changing_component = true;
            }
            CostComponent::ReturnControlledPermanentToOwnersHand(_) => {
                return_permanent_count += 1;
                saw_source_changing_component = true;
            }
            CostComponent::ReturnControlledUnblockedAttackerToOwnersHand => {
                return_permanent_count += 1;
                saw_source_changing_component = true;
            }
            CostComponent::PayLife(amount) => {
                if *amount == 0 {
                    return false;
                }
                pay_life_count += 1;
            }
            CostComponent::TapOtherUntappedControlledPermanentWithSubtype(_) => {
                tap_other_permanent_count += 1;
                saw_source_changing_component = true;
            }
            CostComponent::TapUntappedControlledPermanent(_) => {
                tap_filtered_permanent_count += 1;
                saw_source_changing_component = true;
            }
            CostComponent::RevealHandIfNoCardsWithType(_) => {
                reveal_hand_condition_count += 1;
            }
            CostComponent::ChooseControlledCreatureOrRevealCreatureCardFromHand => {
                chosen_creature_count += 1;
            }
        }
    }

    if discard_count > 1
        || sacrifice_lands_count
            + sacrifice_controlled_count
            + graveyard_exile_count
            + return_permanent_count
            + tap_other_permanent_count
            + tap_filtered_permanent_count
            > 1
        || pay_life_count > 1
        || tap_count > 1
        || source_departure_count > 1
        || tap_other_permanent_count > 1
        || tap_filtered_permanent_count > 1
        || reveal_hand_condition_count > 1
        || chosen_creature_count > 1
    {
        return false;
    }

    !(pay_life_count == 1
        && mana.is_some_and(|cost| {
            cost.pips
                .iter()
                .any(|pip| matches!(pip, mana::Pip::Phyrexian(_)))
        }))
}

/// Activated abilities stage filtered returns, filtered sacrifices, and typed
/// tap-other-permanent costs through `ChooseCostTargets`. Cast-only land
/// sacrifice and graveyard exile remain fail closed in this context.
fn activation_payment_shape_supported(components: &[CostComponent]) -> bool {
    component_payment_shape_supported(components)
        && !components.iter().any(|component| {
            matches!(
                component,
                CostComponent::SacrificeLands(_)
                    | CostComponent::ExileOtherCardsFromOwnGraveyard(_)
                    | CostComponent::TapUntappedControlledPermanent(_)
                    | CostComponent::ChooseControlledCreatureOrRevealCreatureCardFromHand
            )
        })
}

fn can_pay_activation_components(
    components: &[CostComponent],
    player: PlayerId,
    source: ObjectId,
    state: &GameState,
) -> bool {
    can_pay_activation_components_with_x(components, player, source, state, 0)
}

fn can_pay_activation_components_with_x(
    components: &[CostComponent],
    player: PlayerId,
    source: ObjectId,
    state: &GameState,
    x_value: u8,
) -> bool {
    if !activation_payment_shape_supported(components)
        || !can_pay_components(components, player, source, state)
    {
        return false;
    }

    if activation_tap_cost_subtype(components).is_some() {
        return !payable_activation_cost_object_candidates(player, source, components, state)
            .is_empty();
    }
    if has_unblocked_attacker_return_cost(components) {
        return !unblocked_attacker_return_candidates(player, state, &[]).is_empty();
    }
    if let Some(cost) = activation_mana_cost(components) {
        let plan = if components
            .iter()
            .any(|component| matches!(component, CostComponent::Tap))
        {
            mana::can_pay_excluding_source(cost, x_value, player, state, source)
        } else {
            mana::can_pay(cost, x_value, player, state)
        };
        if plan.is_none() {
            return false;
        }
    }
    true
}

fn return_permanent_filter_in(components: &[CostComponent]) -> Option<PermanentFilterDef> {
    components.iter().find_map(|component| match component {
        CostComponent::ReturnControlledPermanentToOwnersHand(filter) => Some(*filter),
        _ => None,
    })
}

fn permanent_matches_return_filter(
    state: &GameState,
    object_id: ObjectId,
    filter: PermanentFilterDef,
) -> bool {
    let Some(object) = state.objects.try_get(object_id) else {
        return false;
    };
    let Some(definition) = card_def::CARD_DEFS.get(object.card_def as usize) else {
        return false;
    };
    match filter {
        PermanentFilterDef::LandWithSubtype(subtype) => {
            definition.has_type(CardType::Land) && has_effective_subtype(state, object_id, subtype)
        }
        PermanentFilterDef::CreatureWithColor(color) => {
            definition.has_type(CardType::Creature)
                && object_color_mask(state, object_id) & card_def::mana_color_mask(color) != 0
        }
    }
}

fn tap_permanent_filter_in(components: &[CostComponent]) -> Option<PermanentFilterDef> {
    components.iter().find_map(|component| match component {
        CostComponent::TapUntappedControlledPermanent(filter) => Some(*filter),
        _ => None,
    })
}

fn tap_permanent_cost_candidates(
    player: PlayerId,
    state: &GameState,
    filter: PermanentFilterDef,
    already_chosen: &[ObjectId],
) -> Vec<ObjectId> {
    state.players[player.index()]
        .battlefield
        .iter()
        .copied()
        .filter(|id| !already_chosen.contains(id))
        .filter(|&id| {
            let object = state.objects.get(id);
            object.zone == Zone::Battlefield
                && object.controller == player
                && !object.tapped
                && permanent_matches_return_filter(state, id, filter)
        })
        .collect()
}

fn return_permanent_cost_candidates(
    player: PlayerId,
    state: &GameState,
    filter: PermanentFilterDef,
    already_chosen: &[EffectObjectBinding],
) -> Vec<EffectObjectBinding> {
    state
        .objects
        .iter()
        .filter_map(|(id, object)| {
            (object.controller == player
                && object.zone == Zone::Battlefield
                && permanent_matches_return_filter(state, id, filter)
                && !already_chosen.iter().any(|binding| binding.object == id))
            .then_some(EffectObjectBinding {
                object: id,
                expected_zone: Zone::Battlefield,
                expected_zone_change_count: object.zone_change_count,
            })
        })
        .collect()
}

/// Whether `player` can currently pay every component of `components`,
/// where `source` is the spell being cast (still nominally in hand) or
/// the permanent whose ability is being activated (on the battlefield --
/// so the hand-exclusion below is a no-op for abilities).
fn can_pay_components(
    components: &[CostComponent],
    player: PlayerId,
    source: ObjectId,
    state: &GameState,
) -> bool {
    if !component_payment_shape_supported(components) {
        return false;
    }
    for c in components {
        let ok = match c {
            CostComponent::Tap => {
                let obj = state.objects.get(source);
                let def = &card_def::CARD_DEFS[obj.card_def as usize];
                // 302.6: a *creature's* tap-cost ability needs continuous
                // control since the turn began. Irrelevant to every
                // tap-cost ability in this pool (Blood is an artifact),
                // kept for correctness if a future card needs it.
                !(obj.tapped || (def.has_type(CardType::Creature) && obj.summoning_sick))
            }
            CostComponent::SacrificeSelf | CostComponent::ExileSelf => true,
            CostComponent::DiscardSelf => {
                let object = state.objects.get(source);
                object.owner == player
                    && object.zone == Zone::Hand
                    && state.players[player.index()].hand.contains(&source)
            }
            CostComponent::DiscardCards(n) => {
                let hand_other = state.players[player.index()]
                    .hand
                    .iter()
                    .filter(|&&id| id != source)
                    .count();
                hand_other >= *n as usize
            }
            CostComponent::SacrificeLands(n) => count_controlled_lands(player, state) >= *n as u32,
            CostComponent::SacrificeControlled { count, filter } => {
                sacrificeable_controlled_permanents(player, *filter, state, &[]).len()
                    >= usize::from(*count)
            }
            CostComponent::ExileOtherCardsFromOwnGraveyard(n) => {
                checked_graveyard_exile_candidates(player, source, state, &[])
                    .is_some_and(|candidates| candidates.len() >= usize::from(*n))
            }
            CostComponent::ReturnControlledPermanentToOwnersHand(filter) => {
                !return_permanent_cost_candidates(player, state, *filter, &[]).is_empty()
            }
            CostComponent::ReturnControlledUnblockedAttackerToOwnersHand => {
                !unblocked_attacker_return_candidates(player, state, &[]).is_empty()
            }
            CostComponent::Mana(cost) => mana::can_pay(cost, 0, player, state).is_some(),
            CostComponent::PayLife(amount) => {
                state.players[player.index()].life >= i32::from(*amount)
            }
            CostComponent::TapOtherUntappedControlledPermanentWithSubtype(subtype) => {
                !activation_cost_object_candidates(player, source, *subtype, state, &[]).is_empty()
            }
            CostComponent::TapUntappedControlledPermanent(filter) => {
                !tap_permanent_cost_candidates(player, state, *filter, &[]).is_empty()
            }
            CostComponent::RevealHandIfNoCardsWithType(card_type) => {
                state.players[player.index()].hand.iter().all(|&object| {
                    !card_def::CARD_DEFS[state.objects.get(object).card_def as usize]
                        .has_type(*card_type)
                })
            }
            CostComponent::ChooseControlledCreatureOrRevealCreatureCardFromHand => {
                !chosen_creature_cost_candidates(
                    player,
                    ChosenCreatureCostZoneV1::Battlefield,
                    state,
                )
                .is_empty()
                    || !chosen_creature_cost_candidates(
                        player,
                        ChosenCreatureCostZoneV1::Hand,
                        state,
                    )
                    .is_empty()
            }
        };
        if !ok {
            return false;
        }
    }
    true
}

/// Pays every component of `components` except `DiscardCards` (already
/// paid via the `pending_discard` staging by the time this runs -- see
/// `EngineState::pending_discard`'s doc). `object_cost_chosen` is the
/// already-decided answer to the one supported interactive object-cost
/// component (`SacrificeLands`, `SacrificeControlled`,
/// `ExileOtherCardsFromOwnGraveyard`, one filtered returned permanent, or a
/// typed tap-other-permanent cost) in
/// `components`. Casts stage this in `PendingCast::sacrifice_chosen`;
/// activations use their incarnation-bound object-cost field. The cast field
/// name is a frozen compatibility alias. Pass `&[]` for a component list
/// statically known not to contain any family.
fn pay_cost_components(
    state: &mut GameState,
    player: PlayerId,
    source: ObjectId,
    components: &[CostComponent],
    object_cost_chosen: &[ObjectId],
) -> bool {
    pay_cost_components_with_x(state, player, source, components, object_cost_chosen, 0)
}

fn pay_cost_components_with_x(
    state: &mut GameState,
    player: PlayerId,
    source: ObjectId,
    components: &[CostComponent],
    object_cost_chosen: &[ObjectId],
    x_value: u8,
) -> bool {
    if !component_payment_shape_supported(components) {
        return false;
    }
    if components.iter().any(|component| {
        matches!(component, CostComponent::PayLife(amount) if state.players[player.index()].life < i32::from(*amount))
    }) {
        return false;
    }
    if components.iter().any(|component| {
        matches!(component, CostComponent::RevealHandIfNoCardsWithType(card_type) if state.players[player.index()].hand.iter().any(|&object| card_def::CARD_DEFS[state.objects.get(object).card_def as usize].has_type(*card_type)))
    }) {
        return false;
    }
    let sacrifice_needed = components.iter().find_map(|component| match component {
        CostComponent::SacrificeLands(amount) => Some(*amount as usize),
        _ => None,
    });
    if let Some(needed) = sacrifice_needed {
        let duplicate = object_cost_chosen
            .iter()
            .enumerate()
            .any(|(index, id)| object_cost_chosen[..index].contains(id));
        let invalid_land = object_cost_chosen.iter().any(|id| {
            !state.players[player.index()].battlefield.contains(id)
                || state.objects.try_get(*id).is_none_or(|object| {
                    object.controller != player
                        || object.zone != Zone::Battlefield
                        || !card_def::CARD_DEFS[object.card_def as usize].is_land
                })
        });
        let aliases_source_departure = object_cost_chosen.contains(&source)
            && components.iter().any(|component| {
                matches!(
                    component,
                    CostComponent::SacrificeSelf
                        | CostComponent::ExileSelf
                        | CostComponent::DiscardSelf
                )
            });
        if object_cost_chosen.len() != needed
            || duplicate
            || invalid_land
            || aliases_source_departure
        {
            return false;
        }
    }
    let sacrifice_controlled = components.iter().find_map(|component| match component {
        CostComponent::SacrificeControlled { count, filter } => {
            Some((usize::from(*count), *filter))
        }
        _ => None,
    });
    if let Some((needed, filter)) = sacrifice_controlled {
        let duplicate = object_cost_chosen
            .iter()
            .enumerate()
            .any(|(index, id)| object_cost_chosen[..index].contains(id));
        let invalid = object_cost_chosen.iter().any(|id| {
            !state.players[player.index()].battlefield.contains(id)
                || state.objects.try_get(*id).is_none_or(|object| {
                    object.controller != player
                        || object.zone != Zone::Battlefield
                        || !permanent_matches_filter(
                            &card_def::CARD_DEFS[object.card_def as usize],
                            filter,
                        )
                })
        });
        let aliases_source_departure = object_cost_chosen.contains(&source)
            && components.iter().any(|component| {
                matches!(
                    component,
                    CostComponent::SacrificeSelf
                        | CostComponent::ExileSelf
                        | CostComponent::DiscardSelf
                )
            });
        if object_cost_chosen.len() != needed || duplicate || invalid || aliases_source_departure {
            return false;
        }
    }
    let graveyard_exile_needed = components.iter().find_map(|component| match component {
        CostComponent::ExileOtherCardsFromOwnGraveyard(amount) => Some(*amount as usize),
        _ => None,
    });
    if let Some(needed) = graveyard_exile_needed {
        if object_cost_chosen.len() != needed
            || checked_graveyard_exile_candidates(player, source, state, object_cost_chosen)
                .is_none()
        {
            return false;
        }
    }
    let return_filter = return_permanent_filter_in(components);
    if let Some(filter) = return_filter {
        let invalid = object_cost_chosen.len() != 1
            || object_cost_chosen.iter().any(|id| {
                state.objects.try_get(*id).is_none_or(|object| {
                    object.controller != player
                        || object.zone != Zone::Battlefield
                        || !permanent_matches_return_filter(state, *id, filter)
                })
            })
            || (object_cost_chosen.contains(&source)
                && components.iter().any(|component| {
                    matches!(
                        component,
                        CostComponent::SacrificeSelf
                            | CostComponent::ExileSelf
                            | CostComponent::DiscardSelf
                    )
                }));
        if invalid {
            return false;
        }
    }
    let returns_unblocked_attacker = has_unblocked_attacker_return_cost(components);
    if returns_unblocked_attacker {
        let candidates = unblocked_attacker_return_candidates(player, state, &[]);
        if object_cost_chosen.len() != 1
            || !candidates
                .iter()
                .any(|binding| binding.object == object_cost_chosen[0])
        {
            return false;
        }
    }
    let tap_other_subtype = components.iter().find_map(|component| match component {
        CostComponent::TapOtherUntappedControlledPermanentWithSubtype(subtype) => Some(*subtype),
        _ => None,
    });
    if let Some(subtype) = tap_other_subtype {
        if object_cost_chosen.len() != 1
            || !activation_cost_object_candidates(player, source, subtype, state, &[])
                .contains(&object_cost_chosen[0])
        {
            return false;
        }
    }
    let tap_filter = tap_permanent_filter_in(components);
    if let Some(filter) = tap_filter {
        if object_cost_chosen.len() != 1
            || !tap_permanent_cost_candidates(player, state, filter, &[])
                .contains(&object_cost_chosen[0])
        {
            return false;
        }
    }
    if sacrifice_needed.is_none()
        && sacrifice_controlled.is_none()
        && graveyard_exile_needed.is_none()
        && return_filter.is_none()
        && !returns_unblocked_attacker
        && tap_other_subtype.is_none()
        && tap_filter.is_none()
        && !object_cost_chosen.is_empty()
    {
        return false;
    }

    // Derive the sole mana plan before applying any state-changing component.
    // This keeps a restored or forward-generated unaffordable shape from
    // partially paying life/discard-adjacent components before failing.
    let mana_cost = activation_mana_cost(components);
    let mut reserved = Vec::new();
    if components
        .iter()
        .any(|component| matches!(component, CostComponent::Tap))
    {
        reserved.push(source);
    }
    if tap_other_subtype.is_some() {
        reserved.extend(object_cost_chosen.iter().copied());
    }
    if tap_filter.is_some() {
        reserved.extend(object_cost_chosen.iter().copied());
    }
    let mana_plan = mana_cost.map(|cost| {
        if reserved.is_empty() {
            mana::can_pay(cost, x_value, player, state)
        } else {
            mana::can_pay_excluding_sources(cost, x_value, player, state, &reserved)
        }
    });
    if mana_plan.as_ref().is_some_and(Option::is_none) {
        return false;
    }
    let mana_plan = mana_plan.flatten();
    for c in components {
        match c {
            CostComponent::Tap => event::propose_and_commit(state, ProposedEvent::tap(source)),
            CostComponent::SacrificeSelf => commit_sacrifice(state, &[source]),
            CostComponent::ExileSelf => {
                event::propose_and_commit(state, ProposedEvent::zone_change(source, Zone::Exile))
            }
            CostComponent::DiscardSelf => event::propose_and_commit(
                state,
                ProposedEvent::zone_change(source, Zone::Graveyard),
            ),
            CostComponent::SacrificeLands(n) => {
                debug_assert_eq!(
                    object_cost_chosen.len(),
                    *n as usize,
                    "sacrifice_chosen must be exactly this component's already-decided picks"
                );
                commit_sacrifice(state, object_cost_chosen);
            }
            CostComponent::SacrificeControlled { count, .. } => {
                debug_assert_eq!(object_cost_chosen.len(), usize::from(*count));
                commit_sacrifice(state, object_cost_chosen);
            }
            CostComponent::ExileOtherCardsFromOwnGraveyard(n) => {
                debug_assert_eq!(
                    object_cost_chosen.len(),
                    *n as usize,
                    "sacrifice_chosen compatibility field must contain the exact graveyard picks"
                );
                commit_graveyard_exile(state, object_cost_chosen);
            }
            CostComponent::ReturnControlledPermanentToOwnersHand(_) => {
                event::propose_and_commit(
                    state,
                    ProposedEvent::zone_change(object_cost_chosen[0], Zone::Hand),
                );
            }
            CostComponent::ReturnControlledUnblockedAttackerToOwnersHand => {
                event::propose_and_commit(
                    state,
                    ProposedEvent::zone_change(object_cost_chosen[0], Zone::Hand),
                );
            }
            CostComponent::Mana(_) => pay_plan(
                state,
                player,
                mana_plan
                    .as_ref()
                    .expect("a supported component slice has at most one preflighted mana cost"),
            ),
            CostComponent::PayLife(amount) => event::propose_and_commit(
                state,
                ProposedEvent::life_loss(player, i32::from(*amount)),
            ),
            CostComponent::DiscardCards(_) => {}
            CostComponent::TapOtherUntappedControlledPermanentWithSubtype(_) => {
                event::propose_and_commit(state, ProposedEvent::tap(object_cost_chosen[0]));
            }
            CostComponent::TapUntappedControlledPermanent(_) => {
                event::propose_and_commit(state, ProposedEvent::tap(object_cost_chosen[0]));
            }
            CostComponent::RevealHandIfNoCardsWithType(_) => {
                let hand = state.players[player.index()].hand.clone();
                for object in hand {
                    for observer in [PlayerId::P0, PlayerId::P1] {
                        state
                            .reveal_hand_card(observer, player, object)
                            .expect("preflighted live hand card remains revealable");
                    }
                }
            }
            CostComponent::ChooseControlledCreatureOrRevealCreatureCardFromHand => {}
        }
    }
    true
}

/// Zone-changes exactly `chosen` to the graveyard -- the actual payment
/// half of a `SacrificeLands` sub-cost, once `Decision::ChooseCostTargets`
/// has already decided *which* permanents (`PendingCast::sacrifice_chosen`).
fn commit_sacrifice(state: &mut GameState, chosen: &[ObjectId]) {
    for &id in chosen {
        event::log_sacrifice(state, id);
        event::propose_and_commit(state, ProposedEvent::zone_change(id, Zone::Graveyard));
    }
}

/// Exiles the exact graveyard cards already selected for an escape cost.
/// All candidates and the mana plan are validated before this helper runs,
/// so no failed cast can leave behind a partially paid graveyard cost.
fn commit_graveyard_exile(state: &mut GameState, chosen: &[ObjectId]) {
    let events = chosen
        .iter()
        .copied()
        .map(|id| ProposedEvent::zone_change(id, Zone::Exile))
        .collect();
    event::propose_and_commit_batch(state, events);
}

fn collect_evidence_candidates(
    player: PlayerId,
    state: &GameState,
) -> Result<Vec<EffectObjectBinding>, String> {
    let graveyard = &state.players[player.index()].graveyard;
    let mut seen = Vec::with_capacity(graveyard.len());
    let mut candidates = Vec::with_capacity(graveyard.len());
    for &object_id in graveyard {
        if seen.contains(&object_id) {
            return Err("Collect Evidence graveyard index contains a duplicate".to_string());
        }
        seen.push(object_id);
        let object = state
            .objects
            .try_get(object_id)
            .ok_or("Collect Evidence graveyard object is missing")?;
        if object.zone != Zone::Graveyard || object.owner != player {
            return Err("Collect Evidence graveyard binding changed owner or zone".to_string());
        }
        if object.v4.face_index != 0 {
            return Err(
                "Collect Evidence mana value is unavailable for a non-front face".to_string(),
            );
        }
        let def = card_def::CARD_DEFS
            .get(object.card_def as usize)
            .ok_or("Collect Evidence card definition is missing")?;
        if object.v4.is_token != def.is_token {
            return Err("Collect Evidence token identity is inconsistent".to_string());
        }
        if !def.is_token {
            candidates.push(EffectObjectBinding {
                object: object_id,
                expected_zone: Zone::Graveyard,
                expected_zone_change_count: object.zone_change_count,
            });
        }
    }
    Ok(candidates)
}

fn bargain_candidates(
    player: PlayerId,
    state: &GameState,
) -> Result<Vec<EffectObjectBinding>, String> {
    let battlefield = &state.players[player.index()].battlefield;
    let mut seen = Vec::with_capacity(battlefield.len());
    let mut candidates = Vec::new();
    for &object_id in battlefield {
        if seen.contains(&object_id) {
            return Err("Bargain battlefield index contains a duplicate".to_string());
        }
        seen.push(object_id);
        let object = state
            .objects
            .try_get(object_id)
            .ok_or("Bargain battlefield object is missing")?;
        if object.zone != Zone::Battlefield || object.controller != player {
            return Err("Bargain permanent binding changed controller or zone".to_string());
        }
        if object.v4.face_index != 0 {
            return Err(
                "Bargain effective card types are unavailable for a non-front face".to_string(),
            );
        }
        let def = card_def::CARD_DEFS
            .get(object.card_def as usize)
            .ok_or("Bargain permanent definition is missing")?;
        if object.v4.is_token != def.is_token {
            return Err("Bargain token identity is inconsistent".to_string());
        }
        if def.is_token || def.has_type(CardType::Artifact) || def.has_type(CardType::Enchantment) {
            candidates.push(EffectObjectBinding {
                object: object_id,
                expected_zone: Zone::Battlefield,
                expected_zone_change_count: object.zone_change_count,
            });
        }
    }
    Ok(candidates)
}

fn optional_cost_binding_is_live(state: &GameState, binding: EffectObjectBinding) -> bool {
    state.objects.try_get(binding.object).is_some_and(|object| {
        object.zone == binding.expected_zone
            && object.zone_change_count == binding.expected_zone_change_count
    })
}

fn selected_bindings_are_unique_subset(
    selected: &[EffectObjectBinding],
    candidates: &[EffectObjectBinding],
) -> bool {
    selected.iter().enumerate().all(|(index, binding)| {
        candidates
            .iter()
            .filter(|candidate| *candidate == binding)
            .count()
            == 1
            && !selected[..index].contains(binding)
    })
}

fn collect_evidence_mana_value(
    state: &GameState,
    selected: &[EffectObjectBinding],
) -> Result<u32, String> {
    selected.iter().try_fold(0_u32, |sum, binding| {
        if !optional_cost_binding_is_live(state, *binding) {
            return Err("Collect Evidence selection is stale".to_string());
        }
        let object = state.objects.get(binding.object);
        let mana_value = u32::from(
            card_def::CARD_DEFS
                .get(object.card_def as usize)
                .ok_or("Collect Evidence card definition is missing")?
                .mana_value,
        );
        sum.checked_add(mana_value)
            .ok_or_else(|| "Collect Evidence mana-value sum overflowed".to_string())
    })
}

fn optional_additional_cost_is_payable(
    kind: OptionalAdditionalCostDef,
    player: PlayerId,
    state: &GameState,
) -> Result<bool, String> {
    match kind {
        OptionalAdditionalCostDef::CollectEvidence { minimum_mana_value } => {
            let candidates = collect_evidence_candidates(player, state)?;
            Ok(collect_evidence_mana_value(state, &candidates)? >= u32::from(minimum_mana_value))
        }
        OptionalAdditionalCostDef::Bargain => Ok(!bargain_candidates(player, state)?.is_empty()),
    }
}

fn optional_additional_cost_selection_is_complete(
    pending: &PendingCast,
    def: &card_def::CardDef,
    state: &GameState,
) -> Result<bool, String> {
    match (
        def.optional_additional_cost,
        pending.optional_additional_cost_paid,
    ) {
        (None, Some(false)) | (Some(_), Some(false)) => Ok(true),
        (Some(OptionalAdditionalCostDef::CollectEvidence { minimum_mana_value }), Some(true)) => {
            Ok(pending.optional_additional_cost_selection_finished
                && collect_evidence_mana_value(state, &pending.optional_additional_cost_chosen)?
                    >= u32::from(minimum_mana_value))
        }
        (Some(OptionalAdditionalCostDef::Bargain), Some(true)) => {
            Ok(pending.optional_additional_cost_chosen.len() == 1)
        }
        (Some(_), None) => Ok(false),
        _ => Err("pending cast optional additional-cost state is noncanonical".to_string()),
    }
}

fn validate_optional_additional_paid_refs(
    kind: OptionalAdditionalCostDef,
    controller: PlayerId,
    refs: &[PaidCostRefV4],
) -> Result<(), String> {
    if refs.is_empty()
        || refs.iter().enumerate().any(|(index, paid)| {
            refs[..index]
                .iter()
                .any(|prior| prior.object == paid.object)
        })
    {
        return Err("optional additional-cost provenance is empty or duplicated".to_string());
    }
    match kind {
        OptionalAdditionalCostDef::CollectEvidence { minimum_mana_value } => {
            let mut total = 0_u32;
            for paid in refs {
                let def = card_def::CARD_DEFS
                    .get(paid.card_def as usize)
                    .ok_or("Collect Evidence paid-card definition is missing")?;
                if paid.owner != controller
                    || paid.zone != Zone::Exile
                    || paid.visible_to_mask != 0b11
                    || def.is_token
                {
                    return Err(
                        "Collect Evidence paid-card provenance changed owner, zone, or type"
                            .to_string(),
                    );
                }
                total = total
                    .checked_add(u32::from(def.mana_value))
                    .ok_or("Collect Evidence paid mana-value sum overflowed")?;
            }
            if total < u32::from(minimum_mana_value) {
                return Err("Collect Evidence paid provenance is below its minimum".to_string());
            }
        }
        OptionalAdditionalCostDef::Bargain => {
            let [paid] = refs else {
                return Err(
                    "Bargain paid provenance must contain exactly one permanent".to_string()
                );
            };
            let def = card_def::CARD_DEFS
                .get(paid.card_def as usize)
                .ok_or("Bargain paid permanent definition is missing")?;
            if paid.controller != controller
                || paid.zone != Zone::Graveyard
                || paid.visible_to_mask != 0b11
                || !(def.is_token
                    || def.has_type(CardType::Artifact)
                    || def.has_type(CardType::Enchantment))
            {
                return Err("Bargain paid provenance is not one eligible sacrifice".to_string());
            }
        }
    }
    Ok(())
}

/// How many lands (0 if none) the cast currently staged in `pending`
/// still needs sacrificed to pay its cost: Fireblast's alt cost, once
/// `cast_mode` has resolved to `Alternative`; Lava Dart's flashback cost,
/// unconditionally (a flashback cast has no alternative payment path to
/// resolve first). A bare `u8` rather than `Option<u8>` -- no card in
/// this pool has `SacrificeLands(0)`, so "not applicable" and "applicable
/// but zero" are not a real ambiguity here -- mirrors `target_count`'s own
/// shape.
fn sacrifice_lands_needed(pending: &PendingCast, def: &card_def::CardDef) -> u8 {
    if pending.is_flashback {
        return def
            .flashback
            .as_ref()
            .and_then(|flashback| {
                flashback.cost.iter().find_map(|component| match component {
                    CostComponent::SacrificeLands(amount) => Some(*amount),
                    _ => None,
                })
            })
            .unwrap_or(0);
    }
    if pending.cast_mode == Some(CastMode::Alternative) {
        if let Some(alt) = def.alt_cost {
            for c in alt {
                if let CostComponent::SacrificeLands(n) = c {
                    return *n;
                }
            }
        }
    }
    0
}

fn cast_tap_permanent_filter_needed(
    pending: &PendingCast,
    def: &card_def::CardDef,
) -> Option<PermanentFilterDef> {
    if pending.is_flashback {
        if let Some(filter) = def
            .flashback
            .as_ref()
            .and_then(|flashback| tap_permanent_filter_in(flashback.cost))
        {
            return Some(filter);
        }
    }
    if pending.cast_mode == Some(CastMode::Alternative) {
        if let Some(filter) = def.alt_cost.and_then(tap_permanent_filter_in) {
            return Some(filter);
        }
    }
    def.additional_cost.and_then(tap_permanent_filter_in)
}

/// `player`'s currently-controlled lands, minus whichever have already
/// been picked this same `Decision::ChooseCostTargets` sequence -- the
/// candidate pool for the next pick (see that decision's doc for why a
/// picked land disappears from the next ask's candidates).
fn sacrificeable_lands(
    player: PlayerId,
    state: &GameState,
    already_chosen: &[ObjectId],
) -> Vec<ObjectId> {
    state.players[player.index()]
        .battlefield
        .iter()
        .copied()
        .filter(|&id| {
            card_def::CARD_DEFS[state.objects.get(id).card_def as usize].is_land
                && !already_chosen.contains(&id)
        })
        .collect()
}

fn permanent_matches_filter(def: &card_def::CardDef, filter: PermanentFilter) -> bool {
    match filter {
        PermanentFilter::ArtifactOrCreature => {
            def.has_type(CardType::Artifact) || def.has_type(CardType::Creature)
        }
        PermanentFilter::Artifact => def.has_type(CardType::Artifact),
        PermanentFilter::Creature => def.has_type(CardType::Creature),
    }
}

/// Ordered currently-controlled battlefield candidates for a typed
/// permanent-sacrifice cost. A permanent matching multiple listed types is
/// still one candidate, while opponent-controlled objects never enter the
/// pool.
fn sacrificeable_controlled_permanents(
    player: PlayerId,
    filter: PermanentFilter,
    state: &GameState,
    already_chosen: &[ObjectId],
) -> Vec<ObjectId> {
    state.players[player.index()]
        .battlefield
        .iter()
        .copied()
        .filter(|id| !already_chosen.contains(id))
        .filter(|&id| {
            let object = state.objects.get(id);
            object.zone == Zone::Battlefield
                && object.controller == player
                && permanent_matches_filter(&card_def::CARD_DEFS[object.card_def as usize], filter)
        })
        .collect()
}

/// Incarnation-bound form of `sacrificeable_controlled_permanents` used by
/// staged activations. Cast staging retains its frozen `ObjectId` wire shape,
/// while an activation must reject a selected permanent that left and later
/// returned before payment.
fn activation_sacrifice_cost_candidates(
    player: PlayerId,
    filter: PermanentFilter,
    state: &GameState,
    already_chosen: &[EffectObjectBinding],
) -> Vec<EffectObjectBinding> {
    let already_chosen_ids = already_chosen
        .iter()
        .map(|binding| binding.object)
        .collect::<Vec<_>>();
    sacrificeable_controlled_permanents(player, filter, state, &already_chosen_ids)
        .into_iter()
        .map(|object| {
            let live = state.objects.get(object);
            EffectObjectBinding {
                object,
                expected_zone: Zone::Battlefield,
                expected_zone_change_count: live.zone_change_count,
            }
        })
        .collect()
}

fn graveyard_exile_cards_needed(pending: &PendingCast, def: &card_def::CardDef) -> u8 {
    if pending.source_contract.cast_method != CastMethodV4::Escape {
        return 0;
    }
    def.escape
        .as_ref()
        .and_then(|escape| {
            escape.cost.iter().find_map(|component| match component {
                CostComponent::ExileOtherCardsFromOwnGraveyard(amount) => Some(*amount),
                _ => None,
            })
        })
        .unwrap_or(0)
}

fn controlled_permanent_sacrifice_in(
    components: &[CostComponent],
) -> Option<(u8, PermanentFilter)> {
    components.iter().find_map(|component| match component {
        CostComponent::SacrificeControlled { count, filter } => Some((*count, *filter)),
        _ => None,
    })
}

fn activation_permanent_sacrifice_needed(
    components: &[CostComponent],
) -> Option<(u8, PermanentFilter)> {
    components.iter().find_map(|component| match component {
        CostComponent::SacrificeControlled { count, filter } => Some((*count, *filter)),
        _ => None,
    })
}

fn cost_kind_for_permanent_filter(filter: PermanentFilter) -> CostKind {
    match filter {
        PermanentFilter::Artifact => CostKind::SacrificeArtifacts,
        PermanentFilter::ArtifactOrCreature | PermanentFilter::Creature => {
            CostKind::SacrificePermanents
        }
    }
}

fn controlled_permanent_sacrifice_needed(
    pending: &PendingCast,
    def: &card_def::CardDef,
) -> Option<(u8, PermanentFilter)> {
    if pending.is_flashback {
        if let Some(sacrifice) = def
            .flashback
            .as_ref()
            .and_then(|flashback| controlled_permanent_sacrifice_in(flashback.cost))
        {
            return Some(sacrifice);
        }
    }
    def.additional_cost
        .and_then(controlled_permanent_sacrifice_in)
}

fn has_chosen_creature_additional_cost(def: &card_def::CardDef) -> bool {
    def.additional_cost.is_some_and(|components| {
        components.iter().any(|component| {
            matches!(
                component,
                CostComponent::ChooseControlledCreatureOrRevealCreatureCardFromHand
            )
        })
    })
}

fn chosen_creature_cost_candidates(
    player: PlayerId,
    zone: ChosenCreatureCostZoneV1,
    state: &GameState,
) -> Vec<EffectObjectBinding> {
    let ids: Vec<ObjectId> = match zone {
        ChosenCreatureCostZoneV1::Battlefield => state
            .objects
            .iter()
            .filter_map(|(id, object)| {
                (object.zone == Zone::Battlefield
                    && object.controller == player
                    && object_has_type(state, id, CardType::Creature))
                .then_some(id)
            })
            .collect(),
        ChosenCreatureCostZoneV1::Hand => state.players[player.index()]
            .hand
            .iter()
            .copied()
            .filter(|&id| {
                let object = state.objects.get(id);
                object.owner == player
                    && object.zone == Zone::Hand
                    && card_def::CARD_DEFS[object.card_def as usize].has_type(CardType::Creature)
            })
            .collect(),
    };
    ids.into_iter()
        .map(|object| {
            let live = state.objects.get(object);
            EffectObjectBinding {
                object,
                expected_zone: live.zone,
                expected_zone_change_count: live.zone_change_count,
            }
        })
        .collect()
}

/// Ordered live candidates for the next Escape graveyard-cost pick. The
/// spell itself is excluded explicitly even though 601.2a normally moved it
/// to the stack before this helper runs.
fn checked_graveyard_exile_candidates(
    player: PlayerId,
    source: ObjectId,
    state: &GameState,
    already_chosen: &[ObjectId],
) -> Option<Vec<ObjectId>> {
    let graveyard = &state.players[player.index()].graveyard;
    let mut indexed = Vec::with_capacity(graveyard.len());
    for &id in graveyard {
        let object = state.objects.try_get(id)?;
        if object.owner != player
            || object.zone != Zone::Graveyard
            || object.v4.is_token
            || indexed.contains(&id)
        {
            return None;
        }
        indexed.push(id);
    }
    // Ceased tokens remain in the arena with their last physical zone marker
    // but are removed from every live zone index. Require reverse membership
    // only for non-token cards, and reject an indexed token outright: normal
    // SBA-clean decision states cannot expose one as an Escape candidate.
    if state.objects.iter().any(|(id, object)| {
        object.owner == player
            && object.zone == Zone::Graveyard
            && !object.v4.is_token
            && !indexed.contains(&id)
    }) {
        return None;
    }
    if already_chosen.iter().enumerate().any(|(index, id)| {
        *id == source || !indexed.contains(id) || already_chosen[..index].contains(id)
    }) {
        return None;
    }
    Some(
        indexed
            .into_iter()
            .filter(|id| *id != source && !already_chosen.contains(id))
            .collect(),
    )
}

/// Derives the printed-cost branch used by both cast offers and final
/// payment. Current consumers (Affinity, Cryptic Serpent, and Deem Inferior,
/// with Tolarian Terror planned) have no generic additional/alternative cost,
/// so reducing the printed generic component is the complete CR 601.2f total-
/// cost calculation for this certified shape. A future reducer paired with
/// Kicker or another generic additional cost must deliberately extend this
/// helper across the combined total rather than silently reusing it.
fn effective_normal_cast_cost(
    def: &card_def::CardDef,
    player: PlayerId,
    state: &GameState,
) -> Cost {
    let Some(reducer) = def.generic_cost_reduction else {
        return def.cost;
    };
    let count = match reducer.count {
        card_def::DynamicCountDef::ControllerBattlefieldAnyType(types) => state.players
            [player.index()]
        .battlefield
        .iter()
        .filter(|&&object| {
            let object = state.objects.get(object);
            object.zone == Zone::Battlefield
                && object.controller == player
                && card_def::CARD_DEFS[object.card_def as usize]
                    .types
                    .iter()
                    .any(|card_type| types.contains(card_type))
        })
        .count() as u32,
        card_def::DynamicCountDef::ControllerGraveyardAnyType(types) => state.players
            [player.index()]
        .graveyard
        .iter()
        .filter(|&&object| {
            let graveyard_def = &card_def::CARD_DEFS[state.objects.get(object).card_def as usize];
            graveyard_def
                .types
                .iter()
                .any(|card_type| types.contains(card_type))
        })
        .count() as u32,
        card_def::DynamicCountDef::ControllerDrawsThisTurn => {
            state.players[player.index()].draws_this_turn
        }
        card_def::DynamicCountDef::ControllerHasCreatureWithAndWithoutSubtype(subtype) => {
            let mut has_subtype = false;
            let mut lacks_subtype = false;
            for &object_id in &state.players[player.index()].battlefield {
                let object = state.objects.get(object_id);
                let def = &card_def::CARD_DEFS[object.card_def as usize];
                if object.zone != Zone::Battlefield
                    || object.controller != player
                    || !def.has_type(CardType::Creature)
                {
                    continue;
                }
                if has_effective_subtype(state, object_id, subtype) {
                    has_subtype = true;
                } else {
                    lacks_subtype = true;
                }
            }
            u32::from(has_subtype && lacks_subtype)
        }
    };
    let reduction = count.saturating_mul(u32::from(reducer.generic_per_count));
    let mut cost = def.cost;
    cost.generic = cost
        .generic
        .saturating_sub(reduction.min(u32::from(u8::MAX)) as u8);
    cost
}

/// Returns the supported Omen definition for the current cast pipeline.
/// Cast-form selection reuses the ordinary two-form decision while retaining
/// each form's own target shape. Other cast-cost modifiers remain excluded
/// until their ordering with Omen is needed by a pool card.
fn supported_omen(def: &card_def::CardDef) -> Option<&card_def::OmenDef> {
    let omen = def.omen.as_ref()?;
    let has_spell_type =
        omen.types.contains(&CardType::Instant) || omen.types.contains(&CardType::Sorcery);
    (has_spell_type
        && def.alt_cost.is_none()
        && def.kicker_cost.is_none()
        && def.additional_cost.is_none()
        && def.flashback.is_none()
        && def.plot_cost.is_none()
        && def.madness_cost.is_none()
        && def.mode2.is_none()
        && def.mode3.is_none()
        && def.escape.is_none()
        && def.generic_cost_reduction.is_none())
    .then_some(omen)
}

fn supported_bestow(def: &card_def::CardDef) -> Option<&card_def::BestowDef> {
    let bestow = def.bestow.as_ref()?;
    (def.types.contains(&CardType::Creature)
        && def.types.contains(&CardType::Enchantment)
        && bestow.target_spec == TargetSpec::Creature
        && def.cost.x_count == 1
        && bestow.cost.x_count == 1
        && def.alt_cost.is_none()
        && def.kicker_cost.is_none()
        && def.additional_cost.is_none()
        && def.flashback.is_none()
        && def.plot_cost.is_none()
        && def.madness_cost.is_none()
        && def.mode2.is_none()
        && def.mode3.is_none()
        && def.escape.is_none()
        && def.omen.is_none()
        && def.generic_cost_reduction.is_none())
    .then_some(bestow)
}

fn has_spell_form_choice(def: &card_def::CardDef) -> bool {
    def.mode2.is_some()
        || def.mode3.is_some()
        || supported_omen(def).is_some()
        || supported_bestow(def).is_some()
}

fn printed_spell_form_count(def: &card_def::CardDef) -> u8 {
    if def.mode3.is_some() {
        3
    } else if def.mode2.is_some()
        || supported_omen(def).is_some()
        || supported_bestow(def).is_some()
    {
        2
    } else {
        1
    }
}

fn spell_form_target_spec(def: &card_def::CardDef, form: u8) -> Option<TargetSpec> {
    match form {
        0 => Some(def.target_spec),
        1 => def
            .mode2
            .as_ref()
            .map(|mode| mode.target_spec)
            .or_else(|| supported_omen(def).map(|omen| omen.target_spec))
            .or_else(|| supported_bestow(def).map(|bestow| bestow.target_spec)),
        2 => def.mode3.as_ref().map(|mode| mode.target_spec),
        _ => None,
    }
}

fn spell_types_have_instant(types: &[CardType]) -> bool {
    types.contains(&CardType::Instant)
}

fn cast_form_timing_ok(
    types: &[CardType],
    keywords: Keywords,
    player: PlayerId,
    state: &GameState,
) -> bool {
    spell_types_have_instant(types)
        || keywords.has(Keywords::FLASH)
        || sorcery_speed_timing_ok(player, state)
}

fn pending_cast_form_timing_ok(
    types: &[CardType],
    keywords: Keywords,
    pending: &PendingCast,
    state: &GameState,
) -> bool {
    spell_types_have_instant(types)
        || keywords.has(Keywords::FLASH)
        || (pending.controller == state.active_player
            && pending.controller == state.priority_player
            && matches!(state.step, Step::Main1 | Step::Main2)
            && state.stack.len() == 1
            && state
                .stack
                .last()
                .is_some_and(|item| item.source == pending.spell))
}

/// Spell forms that remain both target-completable and payable after the
/// source has moved to the stack. Printed modal spells share one cost, while
/// an Omen card's creature and Omen forms each use their own types and cost.
fn viable_pending_spell_forms(
    def: &card_def::CardDef,
    pending: &PendingCast,
    state: &GameState,
) -> Vec<u8> {
    if let Some(bestow) = supported_bestow(def) {
        let mut forms = Vec::with_capacity(2);
        if target_prefix_can_complete_for_controller_and_source(
            def.target_spec,
            &pending.targets_chosen,
            pending.controller,
            targeting_source_for_object(state, pending.spell),
            state,
        ) && pending_cast_form_timing_ok(def.types, def.keywords, pending, state)
            && mana::can_pay(&def.cost, 0, pending.controller, state).is_some()
        {
            forms.push(0);
        }
        if target_prefix_can_complete_for_controller_and_source(
            bestow.target_spec,
            &pending.targets_chosen,
            pending.controller,
            targeting_source_for_object(state, pending.spell),
            state,
        ) && pending_cast_form_timing_ok(
            &[CardType::Enchantment],
            Keywords::NONE,
            pending,
            state,
        ) && mana::can_pay(&bestow.cost, 0, pending.controller, state).is_some()
        {
            forms.push(1);
        }
        return forms;
    }
    let Some(omen) = supported_omen(def) else {
        return viable_printed_spell_modes(def, pending.spell, pending.controller, state);
    };
    let mut forms = Vec::with_capacity(2);
    let normal_cost = effective_normal_cast_cost(def, pending.controller, state);
    if target_prefix_can_complete_for_controller_and_source(
        def.target_spec,
        &pending.targets_chosen,
        pending.controller,
        targeting_source_for_object(state, pending.spell),
        state,
    ) && pending_cast_form_timing_ok(def.types, def.keywords, pending, state)
        && mana::can_pay(&normal_cost, 0, pending.controller, state).is_some()
    {
        forms.push(0);
    }
    if target_prefix_can_complete_for_controller_and_source(
        omen.target_spec,
        &pending.targets_chosen,
        pending.controller,
        targeting_source_for_object(state, pending.spell),
        state,
    ) && pending_cast_form_timing_ok(omen.types, Keywords::NONE, pending, state)
        && mana::can_pay(&omen.cost, 0, pending.controller, state).is_some()
    {
        forms.push(1);
    }
    forms
}

fn payable_cast_modes(
    def: &card_def::CardDef,
    pending: &PendingCast,
    state: &GameState,
) -> Vec<CastMode> {
    let mut modes = Vec::new();
    let normal = effective_normal_cast_cost(def, pending.controller, state);
    if pending_cast_form_timing_ok(def.types, def.keywords, pending, state)
        && mana::can_pay(&normal, 0, pending.controller, state).is_some()
    {
        modes.push(CastMode::Normal);
    }
    if pending_cast_form_timing_ok(def.types, def.keywords, pending, state)
        && def.alt_cost.is_some_and(|components| {
            can_pay_components(components, pending.controller, pending.spell, state)
        })
    {
        modes.push(CastMode::Alternative);
    }
    modes
}

fn pending_cast_selected_mana_cost(
    def: &card_def::CardDef,
    pending: &PendingCast,
    state: &GameState,
) -> Option<Cost> {
    match pending.mode_chosen {
        Some(1) if supported_bestow(def).is_some() => supported_bestow(def).map(|b| b.cost),
        Some(1) if supported_omen(def).is_some() => supported_omen(def).map(|o| o.cost),
        Some(_) => Some(effective_normal_cast_cost(def, pending.controller, state)),
        None => None,
    }
}

fn maximum_payable_x(cost: &Cost, player: PlayerId, state: &GameState) -> Option<u8> {
    (0..=u8::MAX)
        .rev()
        .find(|&x| mana::can_pay(cost, x, player, state).is_some())
}

/// Returns the one graveyard cast method this definition can expose through
/// the current object-id-only `Action::CastSpell` surface. A future card with
/// both Flashback and Escape is deliberately omitted until the action schema
/// can carry an explicit method choice.
fn unambiguous_graveyard_cast_method(def: &card_def::CardDef) -> Option<CastMethodV4> {
    match (def.flashback.is_some(), def.escape.is_some()) {
        (true, false) => Some(CastMethodV4::Flashback),
        (false, true) => Some(CastMethodV4::Escape),
        (false, false) | (true, true) => None,
    }
}

/// Whether `id` is castable right now through the requested definition-owned
/// route, including timing, targets, and the complete cost shape.
fn is_castable_now(
    player: PlayerId,
    id: ObjectId,
    cast_method: CastMethodV4,
    state: &GameState,
) -> bool {
    let def = &card_def::CARD_DEFS[state.objects.get(id).card_def as usize];
    if !def.is_castable() {
        return false;
    }
    if def.omen.is_some() && supported_omen(def).is_none() {
        return false;
    }
    if def.bestow.is_some() && supported_bestow(def).is_none() {
        return false;
    }
    if cast_method != CastMethodV4::Normal
        && viable_printed_spell_modes(def, id, player, state).is_empty()
    {
        return false;
    }
    // 601.3a/117.1a: sorcery-speed timing applies to every permanent-or-
    // sorcery spell (only Instant -- and Flash, not modeled by any card in
    // this pool -- may be cast anytime the caster has priority). Previously
    // enumerated positively as "Sorcery or Creature only", which silently
    // let every OTHER non-Instant type (Artifact, Enchantment) bypass
    // sorcery-speed timing entirely -- inert for Burn (its one Artifact,
    // Relic of Progenitus, is fail-closed and never reaches `is_castable()`)
    // but a real bug for Rally: Experimental Synthesizer (a plain Artifact,
    // no Flash) was offered as castable mid-combat. Root-caused against
    // rally_mirror_v1 game_20260714_144616_0005.txt decision 50 (Combat
    // step, right after DeclareAttackers) and the "phase-mismatch:
    // kernel_step=Main1" divergence class -- both were this same gap, not a
    // step-tracking bug (state.step really was Combat; the timing check
    // just didn't consult it for this card's type).
    let main_timing_ok = cast_form_timing_ok(def.types, def.keywords, player, state);
    if cast_method != CastMethodV4::Normal && !main_timing_ok {
        return false;
    }

    match cast_method {
        CastMethodV4::Flashback => def.flashback.as_ref().is_some_and(|flashback| {
            can_pay_components(flashback.cost, player, id, state)
                && def
                    .additional_cost
                    .is_none_or(|cost| can_pay_components(cost, player, id, state))
        }),
        CastMethodV4::Escape => def.escape.as_ref().is_some_and(|escape| {
            def.additional_cost.is_none()
                && def.kicker_cost.is_none()
                && def.generic_cost_reduction.is_none()
                && can_pay_components(escape.cost, player, id, state)
        }),
        CastMethodV4::Normal => {
            let normal_cost = effective_normal_cast_cost(def, player, state);
            let normal_ok =
                main_timing_ok && mana::can_pay(&normal_cost, 0, player, state).is_some();
            let alt_ok = def
                .alt_cost
                .map(|c| main_timing_ok && can_pay_components(c, player, id, state))
                .unwrap_or(false);
            let main_ok = !viable_printed_spell_modes(def, id, player, state).is_empty()
                && (normal_ok || alt_ok)
                && def
                    .additional_cost
                    .is_none_or(|add| can_pay_components(add, player, id, state));
            let omen_ok = supported_omen(def).is_some_and(|omen| {
                matches!(state.objects.get(id).zone, Zone::Hand | Zone::Exile)
                    && cast_form_timing_ok(omen.types, Keywords::NONE, player, state)
                    && target_prefix_can_complete_for_controller_and_source(
                        omen.target_spec,
                        &[],
                        player,
                        targeting_source_for_object(state, id),
                        state,
                    )
                    && mana::can_pay(&omen.cost, 0, player, state).is_some()
            });
            let bestow_ok = supported_bestow(def).is_some_and(|bestow| {
                matches!(state.objects.get(id).zone, Zone::Hand | Zone::Exile)
                    && cast_form_timing_ok(&[CardType::Enchantment], Keywords::NONE, player, state)
                    && target_prefix_can_complete_for_controller_and_source(
                        bestow.target_spec,
                        &[],
                        player,
                        targeting_source_for_object(state, id),
                        state,
                    )
                    && mana::can_pay(&bestow.cost, 0, player, state).is_some()
            });
            main_ok || omen_ok || bestow_ok
        }
        CastMethodV4::Alternative
        | CastMethodV4::Madness
        | CastMethodV4::Plotted
        | CastMethodV4::Bestow
        | CastMethodV4::Omen => false,
    }
}

pub fn castable_spells(player: PlayerId, state: &GameState) -> Vec<ObjectId> {
    let mut out = Vec::new();
    for &id in &state.players[player.index()].hand {
        if is_castable_now(player, id, CastMethodV4::Normal, state) {
            out.push(id);
        }
    }
    for &id in &state.players[player.index()].graveyard {
        let def = &card_def::CARD_DEFS[state.objects.get(id).card_def as usize];
        if unambiguous_graveyard_cast_method(def)
            .is_some_and(|method| is_castable_now(player, id, method, state))
        {
            out.push(id);
        }
    }
    for &id in &state.exile {
        if state.objects.get(id).owner == player && is_plotted_castable_now(player, id, state) {
            out.push(id);
            continue;
        }
        // Exile play permission (impulse draw): a permission only ever
        // authorizes *either* `Cast` or `Play` (never both -- decided once,
        // at grant time, from `CardDef::is_castable`/`is_playable_land`);
        // the `Cast` half is
        // offered here through the *ordinary* `is_castable_now` check
        // (same timing/cost logic any hand card gets, per external review
        // -- no separate "impulse-castable" legality system), the `Play`
        // half through `land_drop_candidates` instead.
        if let Some(perm) = active_permission_for(player, id, state) {
            if perm.play_or_cast == PlayOrCast::Cast
                && is_castable_now(player, id, CastMethodV4::Normal, state)
            {
                out.push(id);
            }
        }
    }
    out
}

/// The active `PlayPermission` (if any) letting `holder` play/cast `id`
/// straight out of exile right now. Requires *both* `object`/`holder` to
/// match *and* `zone_change_generation` to still equal the object's current
/// `GameObject::zone_change_count` -- the latter is what makes a stale
/// permission structurally unable to "come back to life" after `id` changes
/// zones again for any reason (CR 400.7) rather than merely relying on this
/// module remembering to remove it -- see `PlayPermission`'s doc.
pub(crate) fn active_permission_for(
    holder: PlayerId,
    id: ObjectId,
    state: &GameState,
) -> Option<&PlayPermission> {
    state.engine.exile_play_permissions.iter().find(|p| {
        p.object == id
            && p.holder == holder
            && p.zone_change_generation == state.objects.get(id).zone_change_count
    })
}

/// Sorcery-speed timing (508.1a's "any time you could cast a sorcery"),
/// shared by an ordinary sorcery-speed cast (`is_castable_now`), a Plotted
/// card cast from exile, and Plotting itself (`plot_action_candidates`) --
/// 702.163a/`PlotAbility.setTiming(TimingRule.SORCERY)` grant that timing
/// regardless of the card's own type. All four of 508.1a's real
/// requirements are checked explicitly (active player, main phase, empty
/// stack, and -- per external review -- actually holding priority right
/// now): every current call site already only ever asks this for
/// `state.priority_player`, so that fourth check is currently always true
/// by construction, but it's asserted here directly rather than left as an
/// unstated caller convention.
fn sorcery_speed_timing_ok(player: PlayerId, state: &GameState) -> bool {
    player == state.priority_player
        && player == state.active_player
        && state.stack.is_empty()
        && matches!(state.step, Step::Main1 | Step::Main2)
}

/// Whether `id` (in `state.exile`, owned by `player`) was Plotted on an
/// earlier turn and is therefore castable for free right now (702.163a: at
/// sorcery speed, never the turn it was Plotted). `def.is_castable()` keeps
/// this fail-closed if a future Plot-able card is added to the JSON pool
/// before its `spell_effect` is implemented.
fn is_plotted_castable_now(player: PlayerId, id: ObjectId, state: &GameState) -> bool {
    let obj = state.objects.get(id);
    let Some(plotted_turn) = obj.plotted_turn else {
        return false;
    };
    let def = &card_def::CARD_DEFS[obj.card_def as usize];
    obj.owner == player
        && def.plot_cost.is_some()
        && plotted_turn < state.turn
        && def.is_castable()
        && !viable_printed_spell_modes(def, id, player, state).is_empty()
        && sorcery_speed_timing_ok(player, state)
}

/// Hand cards `player` can currently afford to Plot (`CardDef::plot_cost`).
fn plot_action_candidates(player: PlayerId, state: &GameState) -> Vec<ObjectId> {
    if !sorcery_speed_timing_ok(player, state) {
        return Vec::new();
    }
    state.players[player.index()]
        .hand
        .iter()
        .copied()
        .filter(|&id| {
            let def = &card_def::CARD_DEFS[state.objects.get(id).card_def as usize];
            def.is_castable()
                && def
                    .plot_cost
                    .is_some_and(|cost| mana::can_pay(&cost, 0, player, state).is_some())
        })
        .collect()
}

fn available_mana_abilities(player: PlayerId, state: &GameState) -> Vec<ObjectId> {
    state.players[player.index()]
        .battlefield
        .iter()
        .copied()
        .filter(|&id| !available_mana_ability_choices(player, id, state).is_empty())
        .collect()
}

fn rich_mana_ability_is_payable(
    player: PlayerId,
    source: ObjectId,
    ability_index: u16,
    rich: ManaAbilityDef,
    mana_cost: Option<Cost>,
    state: &GameState,
) -> bool {
    let object = state.objects.get(source);
    let def = &card_def::CARD_DEFS[object.card_def as usize];
    if rich.max_activations_per_turn.is_some_and(|limit| {
        mana_ability_use_count(state, source, ability_index) >= u16::from(limit)
    }) {
        return false;
    }
    if mana_cost.is_some_and(|cost| {
        mana::can_pay_excluding_source(&cost, 0, player, state, source).is_none()
    }) {
        return false;
    }
    match rich.cost {
        ManaAbilityCostDef::TapSelf | ManaAbilityCostDef::TapAndSacrificeSelf => {
            !(object.tapped || def.has_type(CardType::Creature) && object.summoning_sick)
        }
        ManaAbilityCostDef::SacrificeSelf | ManaAbilityCostDef::PutMinus0Minus1CounterOnSelf => {
            true
        }
        ManaAbilityCostDef::TapSelfAndOtherUntappedControlledCreature => {
            // The allocating candidate enumeration must stay behind the two
            // cheap checks: the flat-encode zero-allocation contract counts
            // on the short-circuit (tests/flat_action_allocation.rs).
            !(object.tapped
                || def.has_type(CardType::Creature) && object.summoning_sick
                || mana_ability_cost_targets(player, source, state).is_empty())
        }
    }
}

/// Non-allocating core of [`available_mana_ability_choices`]. Writes into
/// `out` (which it resets first) instead of returning an owned `Vec`, for hot
/// paths (such as flat-action validation) that must not touch the heap. Same
/// selection logic and ordering as the `Vec`-returning wrapper.
pub(crate) fn available_mana_ability_choices_into(
    player: PlayerId,
    source: ObjectId,
    state: &GameState,
    out: &mut ManaColorSetV1,
) {
    *out = ManaColorSetV1::new();
    let Some(object) = state.objects.try_get(source) else {
        return;
    };
    if object.controller != player
        || object.zone != Zone::Battlefield
        || !state.players[player.index()].battlefield.contains(&source)
    {
        return;
    }
    let Some(def) = card_def::CARD_DEFS.get(object.card_def as usize) else {
        return;
    };
    if !def.has_mana_ability() {
        return;
    }

    let mut primary = ManaColorSetV1::new();
    def.primary_mana_ability_choices_into(object.v4.chosen_color, &mut primary);
    let primary_payable = if let Some(rich) = def.mana_ability_def {
        rich_mana_ability_is_payable(player, source, 0, rich, None, state)
    } else {
        !(object.tapped || def.has_type(CardType::Creature) && object.summoning_sick)
    };
    if primary_payable {
        for &color in primary.as_slice() {
            out.push(color);
        }
    }

    let primary_ability_count =
        if def.mana_ability_def.is_some() || def.mana_ability_includes_chosen_color {
            u16::from(!primary.is_empty())
        } else {
            match u16::try_from(def.mana_ability_choices.len()) {
                Ok(count) => count,
                Err(_) => {
                    *out = ManaColorSetV1::new();
                    return;
                }
            }
        };
    for (index, additional) in def.additional_mana_abilities.iter().enumerate() {
        let Ok(index) = u16::try_from(index) else {
            *out = ManaColorSetV1::new();
            return;
        };
        if rich_mana_ability_is_payable(
            player,
            source,
            primary_ability_count + index,
            additional.ability,
            Some(additional.mana_cost),
            state,
        ) {
            for &color in additional.colors {
                if out.contains(color) {
                    *out = ManaColorSetV1::new();
                    return;
                }
                out.push(color);
            }
        }
    }
}

/// Exact currently-payable color choices across every printed mana ability
/// on `source`. This is shared by engine validation and RL enumeration so an
/// unaffordable paid mana ability is never surfaced as a legal action.
pub fn available_mana_ability_choices(
    player: PlayerId,
    source: ObjectId,
    state: &GameState,
) -> Vec<ManaColor> {
    let mut choices = ManaColorSetV1::new();
    available_mana_ability_choices_into(player, source, state, &mut choices);
    choices.as_slice().to_vec()
}

fn mana_ability_use_count(state: &GameState, source: ObjectId, ability_index: u16) -> u16 {
    state
        .objects
        .get(source)
        .v4
        .ability_uses_this_turn
        .iter()
        .find(|entry| {
            entry.ability_kind == AbilityKindV4::Mana && entry.ability_index == ability_index
        })
        .map_or(0, |entry| entry.uses)
}

fn activated_ability_use_count(state: &GameState, source: ObjectId, ability_index: u16) -> u16 {
    state
        .objects
        .get(source)
        .v4
        .ability_uses_this_turn
        .iter()
        .find(|entry| {
            entry.ability_kind == AbilityKindV4::Activated && entry.ability_index == ability_index
        })
        .map_or(0, |entry| entry.uses)
}

/// Exact additional object-cost choices for a rich mana ability. The source
/// is never a candidate for Saruli Caretaker because it must separately pay
/// its own tap-symbol cost. Summoning sickness does not restrict the other
/// creature: the printed cost says "tap," not the tap symbol.
pub fn mana_ability_cost_targets(
    player: PlayerId,
    source: ObjectId,
    state: &GameState,
) -> Vec<ObjectId> {
    let Some(object) = state.objects.try_get(source) else {
        return Vec::new();
    };
    let Some(definition) = card_def::CARD_DEFS.get(object.card_def as usize) else {
        return Vec::new();
    };
    if !matches!(
        definition.mana_ability_def.map(|ability| ability.cost),
        Some(ManaAbilityCostDef::TapSelfAndOtherUntappedControlledCreature)
    ) {
        return Vec::new();
    }
    state.players[player.index()]
        .battlefield
        .iter()
        .copied()
        .filter(|&candidate| candidate != source)
        .filter(|&candidate| {
            let object = state.objects.get(candidate);
            object.controller == player
                && object.zone == Zone::Battlefield
                && !object.tapped
                && card_def::CARD_DEFS[object.card_def as usize].has_type(CardType::Creature)
        })
        .collect()
}

fn activate_mana_ability_for(
    state: &mut GameState,
    player: PlayerId,
    source: ObjectId,
    choice: ManaColor,
    cost_target: Option<ObjectId>,
) -> Result<(), String> {
    let available_choices = available_mana_ability_choices(player, source, state);
    if !available_choices.contains(&choice) {
        return Err(format!("{source} cannot currently produce {choice:?}"));
    }
    let card_def = state.objects.get(source).card_def;
    let chosen_color = state.objects.get(source).v4.chosen_color;
    let definition = &card_def::CARD_DEFS[card_def as usize];
    let ability_index = definition
        .mana_ability_index(choice, chosen_color)
        .ok_or_else(|| format!("{source} cannot produce {choice:?}"))?;
    let source_zone_change_count = state.objects.get(source).zone_change_count;
    let primary_choices = definition.primary_mana_ability_choices(chosen_color);
    let primary_rich = definition.mana_ability_def;
    let additional = if primary_choices.contains(&choice) {
        None
    } else {
        let mut matches = definition
            .additional_mana_abilities
            .iter()
            .copied()
            .filter(|ability| ability.colors.contains(&choice));
        let found = matches.next();
        if matches.next().is_some() {
            return Err(
                "mana choice ambiguously identifies multiple printed abilities".to_string(),
            );
        }
        found
    };
    let rich = if primary_choices.contains(&choice) {
        primary_rich
    } else {
        additional.map(|ability| ability.ability)
    };
    if let Some(rich) = rich {
        let legal_cost_targets = mana_ability_cost_targets(player, source, state);
        match rich.cost {
            ManaAbilityCostDef::TapSelfAndOtherUntappedControlledCreature => {
                let target = cost_target.ok_or_else(|| {
                    format!("{source} requires another untapped controlled creature as a cost")
                })?;
                if !legal_cost_targets.contains(&target) {
                    return Err(format!("{target} is not a legal mana-ability cost target"));
                }
            }
            _ if cost_target.is_some() => {
                return Err(format!(
                    "{source} has no additional mana-ability object cost"
                ));
            }
            _ => {}
        }

        let mana_plan = additional
            .map(|ability| {
                mana::can_pay_excluding_source(&ability.mana_cost, 0, player, state, source)
                    .ok_or_else(|| "additional mana-ability cost became unpayable".to_string())
            })
            .transpose()?;

        let amount = match rich.amount {
            ManaAbilityAmountDef::Fixed(amount) => amount,
            ManaAbilityAmountDef::ControlledCreaturesWithKeyword(keyword) => state.players
                [player.index()]
            .battlefield
            .iter()
            .filter(|&&candidate| {
                let object = state.objects.get(candidate);
                card_def::CARD_DEFS[object.card_def as usize].has_type(CardType::Creature)
                    && has_effective_keyword(state, candidate, keyword)
            })
            .count()
            .try_into()
            .map_err(|_| "mana ability amount exceeds u8".to_string())?,
            ManaAbilityAmountDef::Dynamic(value) => evaluate_dynamic_value(state, value, player)
                .try_into()
                .map_err(|_| "mana ability amount exceeds u8".to_string())?,
        };

        if let Some(plan) = &mana_plan {
            pay_plan(state, player, plan);
        }

        match rich.cost {
            ManaAbilityCostDef::TapSelf => {
                event::propose_and_commit(state, ProposedEvent::tap(source));
            }
            ManaAbilityCostDef::SacrificeSelf => {
                commit_sacrifice(state, &[source]);
            }
            ManaAbilityCostDef::TapSelfAndOtherUntappedControlledCreature => {
                event::propose_and_commit(state, ProposedEvent::tap(source));
                event::propose_and_commit(
                    state,
                    ProposedEvent::tap(cost_target.expect("validated cost target")),
                );
            }
            ManaAbilityCostDef::PutMinus0Minus1CounterOnSelf => {
                let counters = &mut state.objects.get_mut(source).counters.minus0_minus1;
                *counters = counters
                    .checked_add(1)
                    .ok_or_else(|| "-0/-1 counter count overflow".to_string())?;
            }
            ManaAbilityCostDef::TapAndSacrificeSelf => {
                event::propose_and_commit(state, ProposedEvent::tap(source));
                commit_sacrifice(state, &[source]);
            }
        }
        if amount > 0 {
            event::propose_and_commit(
                state,
                ProposedEvent::mana_add(player, vec![choice; usize::from(amount)]),
            );
        }
        if rich.controller_damage > 0 {
            event::propose_and_commit(
                state,
                ProposedEvent::damage(
                    source,
                    Target::Player(player),
                    i32::from(rich.controller_damage),
                ),
            );
        }
    } else {
        if cost_target.is_some() {
            return Err(format!(
                "{source} has no additional mana-ability object cost"
            ));
        }
        let program = definition
            .mana_ability_program_for(choice, chosen_color)
            .expect("validated mana choice has a generated program");
        let ctx = ExecCtx::no_targets(source, player);
        effect::execute(&program, &ctx, state);
    }
    if state.objects.get(source).zone_change_count == source_zone_change_count {
        state
            .objects
            .get_mut(source)
            .v4
            .note_ability_use(AbilityKindV4::Mana, ability_index);
    }
    collect_and_queue_triggers(state);

    // 605.3b: mana abilities don't use the stack, so nothing goes on
    // it and no new *stack item* appears -- but this is not the same
    // claim as "doesn't reset the priority-passing count": Java's
    // `PlayerImpl.activateAbility` ends with an unconditional
    // `game.getPlayers().resetPassed()` for *any* successful action,
    // `ACTIVATED_MANA` included (same method, same shared tail as
    // the `SPELL`/other-ability branches; see `play_land`'s own
    // identical reset, for the same reason, on the land-drop side of
    // that method). Without this, a stale `priority_passes[other]
    // == true` left over from *before* this activation can combine
    // with the *activating* player's own next real pass to
    // spuriously satisfy `[true, true]` and advance the step/phase,
    // skipping the other player's now-legitimately-fresh priority
    // window entirely. Root-caused (increment 13) against
    // `game_20260713_002203_0026.txt` decision 64: PlayerRL1 taps a
    // Mountain in Postcombat Main after SelfPlay had already passed
    // once this round; the reference still lets SelfPlay act again
    // (two more Lava Dart casts, both self-targeted, -2 life) before
    // the turn ends, but the kernel -- never re-arming SelfPlay's
    // stale pass -- skipped straight to the next turn once
    // PlayerRL1 next had nothing left to do, silently losing both
    // points of self-damage.
    state.engine.priority_passes = [false, false];
    // See `EngineState::mana_ability_activations`'s doc: the
    // `DeclareAttackers`/`DeclareBlockers` combat throttle
    // (`HarnessSurfaceV2::combat_priority_stack_len_seen`) needs its
    // own re-arm signal for this action, since it never touches the
    // stack the way a cast/non-mana-activation does.
    state.engine.mana_ability_activations += 1;
    state.engine.last_mana_ability_activator = Some(player);
    Ok(())
}

pub fn available_activatable_abilities(player: PlayerId, state: &GameState) -> Vec<(ObjectId, u8)> {
    let mut out = Vec::new();
    // Preserve the historical battlefield and hand ordering, then append
    // graveyard actions in graveyard order. A definition's activation zone
    // is checked for every ability, so mixed-zone cards remain data-driven.
    for (zone, objects) in [
        (
            Zone::Battlefield,
            &state.players[player.index()].battlefield,
        ),
        (Zone::Hand, &state.players[player.index()].hand),
        (Zone::Graveyard, &state.players[player.index()].graveyard),
    ] {
        for &id in objects {
            let def = &card_def::CARD_DEFS[state.objects.get(id).card_def as usize];
            if !def.is_executable() {
                continue;
            }
            for (i, ability) in def.activated_abilities.iter().enumerate() {
                if ability.activation_zone != zone
                    || (ability.sorcery_speed_only && !sorcery_speed_timing_ok(player, state))
                    || ability.max_activations_per_turn.is_some_and(|limit| {
                        activated_ability_use_count(state, id, i as u16) >= u16::from(limit)
                    })
                {
                    continue;
                }
                if can_pay_activation_components(ability.cost, player, id, state)
                    && activation_target_prefix_can_complete(id, ability, &[], state)
                {
                    out.push((id, i as u8));
                }
            }
        }
    }
    out
}

fn land_drop_candidates(player: PlayerId, state: &GameState) -> Vec<ObjectId> {
    if player != state.active_player
        || state.players[player.index()].lands_played_this_turn > 0
        || !matches!(state.step, Step::Main1 | Step::Main2)
        || !state.stack.is_empty()
    {
        return Vec::new();
    }
    let mut out: Vec<ObjectId> = state.players[player.index()]
        .hand
        .iter()
        .copied()
        .filter(|&id| {
            card_def::CARD_DEFS[state.objects.get(id).card_def as usize].is_playable_land()
        })
        .collect();
    // Impulse-drawn lands (Clockwork Percussionist/Experimental
    // Synthesizer/Reckless Impulse can exile a land off the top of the
    // library same as any other card) are still subject to the ordinary
    // one-land-per-turn limit -- already enforced by the guard above, since
    // none of these cards grant an additional land play.
    for &id in &state.exile {
        if let Some(perm) = active_permission_for(player, id, state) {
            let def = &card_def::CARD_DEFS[state.objects.get(id).card_def as usize];
            if perm.play_or_cast == PlayOrCast::Play && def.is_playable_land() {
                out.push(id);
            }
        }
    }
    out
}

fn can_attack(state: &GameState, id: ObjectId) -> bool {
    let obj = state.objects.get(id);
    let def = &card_def::CARD_DEFS[obj.card_def as usize];
    def.is_executable()
        && def.has_type(CardType::Creature)
        && !obj.tapped
        && !has_effective_keyword(state, id, Keywords::DEFENDER)
        && (!obj.summoning_sick || has_effective_keyword(state, id, Keywords::HASTE))
}

fn eligible_attackers(state: &GameState) -> Vec<ObjectId> {
    state.players[state.active_player.index()]
        .battlefield
        .iter()
        .copied()
        .filter(|&id| can_attack(state, id))
        .collect()
}

fn goad_is_active(state: &GameState, goad: &crate::state::GoadStateV4) -> bool {
    goad.expires_at_turn > state.turn
        || (goad.expires_at_turn == state.turn && state.active_player != goad.player)
}

pub fn required_goaded_attackers(state: &GameState, eligible: &[ObjectId]) -> Vec<ObjectId> {
    eligible
        .iter()
        .copied()
        .filter(|id| {
            state
                .objects
                .get(*id)
                .v4
                .goaded_by
                .iter()
                .any(|goad| goad_is_active(state, goad))
        })
        .collect()
}

/// Which of the defending player's untapped creatures may legally block
/// `attacker` (509.1b: flying attackers can only be blocked by
/// flying/reach).
pub(crate) fn minimum_blockers_required(state: &GameState, attacker: ObjectId) -> usize {
    let attacker_obj = state.objects.get(attacker);
    let definition_minimum = card_def::CARD_DEFS[attacker_obj.card_def as usize].minimum_blockers;
    let keyword_minimum = if has_effective_keyword(state, attacker, Keywords::MENACE) {
        2
    } else {
        1
    };
    usize::from(
        attacker_obj
            .v4
            .minimum_blockers_override
            .unwrap_or_else(|| definition_minimum.max(keyword_minimum))
            .max(1),
    )
}

fn legal_blockers_for(state: &GameState, attacker: ObjectId) -> Vec<ObjectId> {
    let attacker_obj = state.objects.get(attacker);
    let defender = attacker_obj.controller.opponent();
    let defender_controls_island = state.players[defender.index()]
        .battlefield
        .iter()
        .copied()
        .any(|id| {
            let object = state.objects.get(id);
            let definition = &card_def::CARD_DEFS[object.card_def as usize];
            definition.has_type(CardType::Land)
                && has_effective_subtype(state, id, card_def::Subtype::Island)
        });
    if defender_controls_island && has_effective_keyword(state, attacker, Keywords::ISLANDWALK) {
        return Vec::new();
    }
    let attacker_flying = has_effective_keyword(state, attacker, Keywords::FLYING);
    let minimum = minimum_blockers_required(state, attacker);
    let blockers = state.players[defender.index()]
        .battlefield
        .iter()
        .copied()
        .filter(|&id| {
            let obj = state.objects.get(id);
            if obj.tapped {
                return false;
            }
            let def = &card_def::CARD_DEFS[obj.card_def as usize];
            if !def.is_executable() || !object_has_type(state, id, CardType::Creature) {
                return false;
            }
            if has_effective_keyword(state, attacker, Keywords::PROTECTION_FROM_MONOCOLORED)
                && object_is_monocolored(state, id)
            {
                return false;
            }
            if attacker_flying
                && !has_effective_keyword(state, id, Keywords::FLYING)
                && !has_effective_keyword(state, id, Keywords::REACH)
            {
                return false;
            }
            true
        })
        .collect::<Vec<_>>();
    if blockers.len() < minimum {
        Vec::new()
    } else {
        blockers
    }
}

fn check_game_over(state: &GameState) -> Option<Decision> {
    match (state.players[0].has_lost, state.players[1].has_lost) {
        (false, false) => None,
        (true, false) => Some(Decision::GameOver {
            winner: Some(PlayerId::P1),
        }),
        (false, true) => Some(Decision::GameOver {
            winner: Some(PlayerId::P0),
        }),
        (true, true) => Some(Decision::GameOver { winner: None }),
    }
}

// -------------------------------------------------------------- the loop

/// Drives the state machine forward until a real decision point is
/// reached. Never auto-passes and never skips a priority window that the
/// comprehensive rules would actually grant.
pub fn advance_until_decision(state: &mut GameState) -> Decision {
    loop {
        if let Err(source) = state.validate_attachment_relations() {
            state.engine.halted = Some((UnsupportedMechanic::InvalidEffectContinuation, source));
        }
        if let Some(d) = check_game_over(state) {
            return d;
        }
        if let Some((mechanic, source)) = state.engine.halted {
            return Decision::Halted { mechanic, source };
        }

        if let Some(pending) = state.engine.pending_land_play.as_ref() {
            if validate_pending_land_play(state, pending).is_err() {
                state.engine.halted = Some((
                    UnsupportedMechanic::InvalidEffectContinuation,
                    pending.source,
                ));
                return Decision::Halted {
                    mechanic: UnsupportedMechanic::InvalidEffectContinuation,
                    source: pending.source,
                };
            }
            return Decision::ChooseEffectOption {
                player: pending.controller,
                source: pending.source,
                option_count: 4,
            };
        }

        if let Some(d) = drain_pending_spell_copy_or_decide(state) {
            return d;
        }

        let had_pending_effect = state.engine.pending_effect.is_some();
        if let Some(d) = drain_pending_effect_or_decide(state) {
            return d;
        }
        // If a continuation existed, restart regardless of whether this
        // stage suspended again or completed. Suspension can immediately
        // yield a second generic choice (Winding Way's type choice followed
        // by graveyard ordering), while completion just ran the normal
        // post-resolution SBA/trigger checkpoint. In particular, an empty-
        // library draw must become GameOver at the top of the loop before
        // any ordinary priority decision can escape.
        if had_pending_effect {
            continue;
        }

        if let Some(d) = drain_pending_discard_or_decide(state) {
            return d;
        }

        let had_pending_cast = state.engine.pending_cast.is_some();
        if let Some(d) = drain_pending_cast_or_decide(state) {
            return d;
        }
        // Finishing a cast can itself create the next terminal checkpoint:
        // Deep Analysis may pay its mandatory flashback life cost down to
        // exactly zero. `collect_and_queue_triggers` performs the SBA pass,
        // but `check_game_over` is at the top of this loop, so restart before
        // an ordinary priority decision can escape. The same restart also
        // lets a newly-staged additional-cost discard run in its normal
        // ordering position above.
        if had_pending_cast {
            continue;
        }
        let had_pending_activation = state.engine.pending_activation.is_some();
        if let Some(d) = drain_pending_activation_or_decide(state) {
            return d;
        }
        // Same terminal/checkpoint reasoning as the cast path above. This
        // also restarts into a freshly staged discard cost (Masked Meower,
        // the Blood token) before any ordinary priority decision can escape.
        if had_pending_activation {
            continue;
        }

        // Unlike `pending_discard`/`pending_cast`/`pending_activation`,
        // `pending_optional_cost` never auto-resolves (declining is always
        // legal, so there's always a real choice once staged) -- no
        // `continue`-and-recheck dance needed here: if `Action::
        // ChooseOptionalCost(Discard)` stages a fresh `pending_discard`,
        // that's picked up at the very top of the *next* call to this
        // function (the loop only reaches this far after already
        // confirming `pending_discard` is `None`).
        if let Some(d) = drain_pending_optional_cost_or_decide(state) {
            return d;
        }

        if let Some(d) = drain_pending_optional_cost_sacrifice_or_decide(state) {
            return d;
        }
        // Same reasoning as the two `pending_discard` re-checks above:
        // once every land is chosen, this runs `poc.then` (Highway
        // Robbery's own "draw two cards"), which could in principle stage
        // a fresh `pending_discard` of its own.
        if state.engine.pending_discard.is_some() {
            continue;
        }

        if let Some(d) = drain_pending_triggers_or_decide(state) {
            return d;
        }

        if state.step == Step::DeclareAttackers && !state.engine.combat.attackers_declared {
            return Decision::DeclareAttackers {
                player: state.active_player,
                eligible: eligible_attackers(state),
            };
        }
        if state.step == Step::DeclareBlockers && !state.engine.combat.blockers_declared {
            // `ComputerPlayerRL.selectBlockers` explicitly sorts attackers by
            // power descending ("so biggest threats are handled first")
            // before asking about blockers one attacker at a time --
            // *not* declaration order (`Combat.attackers` is itself a Java
            // `HashSet<UUID>`, so "declaration order" was never a real
            // signal on the reference side to begin with). Root-caused
            // against `game_20260713_002212_0038.txt` decisions 115-116:
            // two identically-named attackers (both Voldaren Epicure, same
            // power) get their blocker windows asked in the *opposite*
            // order from how they were declared (`DECLARE_ATTACKS`'s own
            // `chosen_indices`), which only a power-based (not
            // declaration-based) ordering explains once a third, lower/
            // higher-power attacker (Sneaky Snacker, evasive and therefore
            // silently skipped -- zero eligible blockers) is filtered out of
            // the picture. Ties between equal-power attackers aren't
            // reproduced bit-for-bit here (the reference's own tie-break is
            // Java `HashSet<UUID>` iteration order, an implementation
            // artifact of `UUID.hashCode()`/bucket layout with no rules
            // meaning -- not worth porting), but the dominant, rules-visible
            // signal (power) now matches.
            //
            // BLOCKED(b) -- proven oracle ambiguity, increment 14. Read
            // (not modified) `Mage/src/main/java/mage/game/combat/
            // Combat.java:131-137`: `getAttackers()` builds a *fresh*
            // `new HashSet<UUID>()` by flattening every `CombatGroup`'s
            // (order-preserving `ArrayList<UUID>`, `CombatGroup.java:31`)
            // attackers into it -- any real ordering signal is thrown away
            // at that exact call, before either AI ever sees the list.
            // `ComputerPlayerRL.getAttackers` (`ComputerPlayerRL.java:9025-
            // 9037`, active by default since `RL_COMBAT_TAKEOVER` defaults
            // false) and `ComputerPlayer6.getAttackers`
            // (`ComputerPlayer6.java:1478-1490`, the alternate path) both
            // do a plain `for (UUID id : attackersUUID)` over that
            // `HashSet`, so the pre-sort list order literally *is*
            // `UUID.hashCode()` bucket order. Both subsequent sorts compare
            // power only -- `ComputerPlayerRL.java`'s inline
            // `attackers.sort((a, b) -> Integer.compare(b.getPower()...,
            // a.getPower()...))` and `CombatUtil.sortByPower`
            // (`CombatUtil.java:89-94`) -- neither ever consults toughness,
            // name, controller, or any creation-order/timestamp field
            // (`MageObject`/`PermanentImpl` expose no such field at all;
            // the closest, `zoneChangeCounter`, is a per-object
            // zone-change count that defaults to the same value, `1`, for
            // two same-turn creatures that never changed zones since
            // entering, so it can't disambiguate this tie even if
            // consulted). Since `List.sort` is stable, the tie survives
            // pre-sort order intact for `ComputerPlayerRL`'s direct
            // descending-comparator sort (this corpus's active path) --
            // i.e. hash-bucket order, full stop. There is no rules-visible,
            // kernel-computable key left to try: name, power, toughness,
            // controller are confirmed identical between the two Voldaren
            // Epicures in this trace, and the observed order (attacker
            // `be9f571a...` asked before `b01635f5...`) matches neither
            // declaration order nor its reverse nor any other stable
            // creature-attribute sort -- only `UUID.hashCode()` bucket
            // placement explains it, and reproducing that would mean
            // teaching the kernel to replicate Java's String/UUID hash
            // function over UUIDs that don't exist outside this replay
            // harness (a real self-play game mints no Java UUIDs at all),
            // which is architecturally backwards for a kernel meant to run
            // standalone. Left unresolved by design; see the increment-14
            // report for the corpus-wide classification this backs.
            // An attacker that left the battlefield during the priority
            // window after declaration is no longer in combat (506.4).
            // Filter per attacker, not only at the step boundary: a mixed
            // batch can contain both a surviving attacker and one killed by
            // an instant, in which case Declare Blockers still happens but
            // must not expose a phantom prompt for the dead creature.
            let mut attackers: Vec<ObjectId> = state
                .engine
                .combat
                .attackers
                .iter()
                .copied()
                .filter(|&id| is_still_in_combat(state, id))
                .collect();
            attackers.sort_by_key(|&id| std::cmp::Reverse(effective_power(state, id)));
            let legal_blockers = attackers
                .iter()
                .map(|&a| (a, legal_blockers_for(state, a)))
                .collect();
            return Decision::DeclareBlockers {
                player: state.active_player.opponent(),
                attackers,
                legal_blockers,
            };
        }

        // 500.4/514.3: the game never advances past a step/phase boundary
        // while the stack is non-empty, even a step that otherwise never
        // grants priority on its own (`step_grants_priority`'s Untap/
        // Cleanup case) -- 514.3 in particular carves out exactly this
        // exception for cleanup: "if any state-based actions would be
        // performed... or if any triggered abilities are waiting to be put
        // onto the stack... those actions are taken, then... players
        // receive priority." Untap can never actually hit this (no card in
        // this pool has an untap-triggered ability), but Cleanup can --
        // discarding to hand size, or a cleanup-timed forced discard, can
        // discard a Madness card, which is a real triggered ability
        // (`state::StackItem::madness_offer`) needing to sit through
        // normal priority before it resolves, same as anywhere else on the
        // stack. Root-caused (this increment) against
        // `game_20260713_002200_0021.txt`: without this guard,
        // `advance_step` blindly flips `active_player`/`turn` and resets
        // Untap/Draw state for the *next* turn while the Madness offer
        // from *this* turn's cleanup discard is still sitting, unresolved,
        // on the stack underneath -- a genuine active-player/turn
        // desync, not merely a missed decision.
        if !step_grants_priority(state.step) && state.stack.is_empty() {
            advance_step(state);
            continue;
        }

        if state.engine.priority_passes == [true, true] {
            if let Some(top) = state.stack.last() {
                // A Madness offer (see `push_trigger_onto_stack`'s
                // `is_madness_offer` doc) is a real stack item like any
                // other triggered ability -- it sits through normal
                // priority (both players may respond, e.g. `Guttersnipe`
                // off an instant, before it resolves) -- but *resolving*
                // it is a real decision (`Decision::ChooseMadnessCast`),
                // not a fixed `EffectOp` program, so it's intercepted
                // here, exactly where `resolve_top_of_stack` would
                // otherwise run, instead of being popped by it. Root-
                // caused (this increment) against two golden-trace games
                // (`game_20260713_002154_0011.txt`,
                // `game_20260713_002156_0015.txt`): the prior model asked
                // `Decision::ChooseMadnessCast` immediately at discard
                // time (`EngineState::pending_madness`, now removed),
                // which is wrong two ways at once -- it can preempt real
                // intervening priority windows the reference actually
                // offers first (0011: a mana ability activated while the
                // Madness trigger is still unresolved on the stack), and
                // after a *decline*, it let whatever else was already on
                // the stack underneath (there, the same discard's own
                // Blood Token activation) resolve too eagerly, without
                // waiting for the normal priority round the reference
                // actually grants first (0015: the reference casts
                // Fireblast in response to its own still-pending Blood
                // Token draw, which this model's silent immediate-resolve
                // never gave a chance to happen).
                if top.madness_offer || top.kind == StackItemKind::MadnessOffer {
                    let source = top.source;
                    match validate_madness_decision_preflight(state) {
                        Ok((card, player)) => {
                            return Decision::ChooseMadnessCast { player, card };
                        }
                        Err(_) => {
                            state.engine.halted =
                                Some((UnsupportedMechanic::InvalidEffectContinuation, source));
                            return Decision::Halted {
                                mechanic: UnsupportedMechanic::InvalidEffectContinuation,
                                source,
                            };
                        }
                    }
                }
                if resolve_top_of_stack(state) == ResolutionProgress::Complete {
                    collect_and_queue_triggers(state);
                    reset_priority(state);
                }
            } else {
                advance_step(state);
            }
            continue;
        }

        return Decision::CastSpellOrPass {
            player: state.priority_player,
            castable_spells: castable_spells(state.priority_player, state),
            mana_abilities: available_mana_abilities(state.priority_player, state),
            land_drops: land_drop_candidates(state.priority_player, state),
            activatable_abilities: available_activatable_abilities(state.priority_player, state),
            plot_actions: plot_action_candidates(state.priority_player, state),
        };
    }
}

fn reset_priority(state: &mut GameState) {
    state.engine.priority_passes = [false, false];
    state.priority_player = state.active_player;
    state.engine.priority_round += 1;
    // Captured here before this `advance_until_decision` iteration can
    // continue into a resolution-created trigger push. See the field's doc:
    // this is a diagnostic boundary snapshot, not the surface suppression's
    // deliberately later first-observation baseline.
    state.engine.stack_len_at_round_open = state.stack.len();
    state.engine.mana_ability_count_at_round_open = state.engine.mana_ability_activations;
}

fn collect_and_queue_triggers(state: &mut GameState) {
    let triggers = trigger::collect_and_process(state);
    state.engine.pending_triggers.extend(triggers);
}

fn remaining_cast_payment_is_payable(
    state: &GameState,
    pending: &PendingCast,
    staged_method: CastMethodV4,
) -> bool {
    let def = &card_def::CARD_DEFS[state.objects.get(pending.spell).card_def as usize];
    let cast_method = finalized_cast_method(pending, staged_method, def);
    let x_value = pending.x_value.unwrap_or(0);
    let base_payable = match cast_method {
        CastMethodV4::Normal => {
            let normal = effective_normal_cast_cost(def, pending.controller, state);
            if pending.kicked == Some(true) {
                def.kicker_cost.is_some_and(|kicker| {
                    mana::can_pay_combined(&[&normal, &kicker], 0, pending.controller, state)
                        .is_some()
                })
            } else {
                mana::can_pay(&normal, x_value, pending.controller, state).is_some()
            }
        }
        CastMethodV4::Alternative => def.alt_cost.is_some_and(|components| {
            can_pay_components(components, pending.controller, pending.spell, state)
        }),
        CastMethodV4::Flashback => def.flashback.as_ref().is_some_and(|flashback| {
            can_pay_components(flashback.cost, pending.controller, pending.spell, state)
        }),
        CastMethodV4::Escape => def.escape.as_ref().is_some_and(|escape| {
            can_pay_components(escape.cost, pending.controller, pending.spell, state)
        }),
        CastMethodV4::Madness => def
            .madness_cost
            .is_some_and(|cost| mana::can_pay(&cost, 0, pending.controller, state).is_some()),
        CastMethodV4::Plotted => true,
        CastMethodV4::Omen => supported_omen(def).is_some_and(|omen| {
            mana::can_pay(&omen.cost, x_value, pending.controller, state).is_some()
        }),
        CastMethodV4::Bestow => supported_bestow(def).is_some_and(|bestow| {
            mana::can_pay(&bestow.cost, x_value, pending.controller, state).is_some()
        }),
    };
    base_payable
        && def.additional_cost.is_none_or(|components| {
            can_pay_components(components, pending.controller, pending.spell, state)
        })
}

fn finalized_cast_method(
    pending: &PendingCast,
    staged_method: CastMethodV4,
    def: &card_def::CardDef,
) -> CastMethodV4 {
    if staged_method == CastMethodV4::Normal {
        if supported_omen(def).is_some() && pending.mode_chosen == Some(1) {
            CastMethodV4::Omen
        } else if supported_bestow(def).is_some() && pending.mode_chosen == Some(1) {
            CastMethodV4::Bestow
        } else if pending.cast_mode == Some(CastMode::Alternative) {
            CastMethodV4::Alternative
        } else {
            CastMethodV4::Normal
        }
    } else {
        staged_method
    }
}

/// Cross-slot validation for an interactive activation discard. The resume
/// record duplicates the exact activation identity so a stale or tampered
/// snapshot cannot redirect a discard to another player/ability or mutate
/// any card before the activation's complete payment shape is revalidated.
pub(crate) fn validate_pending_discard_binding(
    state: &GameState,
    pending_discard: &PendingDiscard,
) -> Result<(), (ObjectId, String)> {
    match &pending_discard.resume {
        DiscardResume::FinishCast {
            source_contract,
            controller,
        } => {
            let source = source_contract.source;
            let fail = |message: &str| Err((source, message.to_string()));
            let Some(pending) = state.engine.pending_cast.as_ref() else {
                return fail("cast discard has no pending cast");
            };
            if pending.source_contract != *source_contract || pending.controller != *controller {
                return fail("cast discard binding changed");
            }
            validate_pending_cast(state, pending).map_err(|message| (source, message))?;
            let method = state
                .stack
                .last()
                .and_then(|item| item.v4.cast_method)
                .ok_or_else(|| {
                    (
                        source,
                        "cast discard lost its placeholder method".to_string(),
                    )
                })?;
            if !remaining_cast_payment_is_payable(state, pending, method) {
                return fail("cast discard's remaining payment is no longer payable");
            }
            let available = state.players[pending.controller.index()]
                .hand
                .iter()
                .filter(|&&object| object != pending.spell)
                .count();
            if available < pending_discard.count as usize {
                return fail("cast discard has fewer eligible cards than its exact cost");
            }
            Ok(())
        }
        DiscardResume::FinishActivation {
            source,
            source_zone_change_count,
            controller,
            ability_index,
        } => {
            let (source, source_zone_change_count, controller, ability_index) = (
                *source,
                *source_zone_change_count,
                *controller,
                *ability_index,
            );
            let fail = |message: &str| Err((source, message.to_string()));
            let Some(pending) = state.engine.pending_activation.as_ref() else {
                return fail("activation discard has no pending activation");
            };
            if pending.source != source
                || pending.source_zone_change_count != source_zone_change_count
                || pending.controller != controller
                || pending.ability_index != ability_index
            {
                return fail("activation discard binding changed");
            }
            validate_pending_activation(state, pending).map_err(|message| (source, message))?;
            let object = state.objects.get(source);
            let def = &card_def::CARD_DEFS[object.card_def as usize];
            let ability = &def.activated_abilities[ability_index as usize];
            let Some(expected) = discard_count_in(ability.cost) else {
                return fail(
                    "activation discard resumed an ability without an interactive discard cost",
                );
            };
            if pending_discard.player != controller || pending_discard.count != u32::from(expected)
            {
                return fail("activation discard player or count changed");
            }
            if pending.cost_discard_paid.is_some() {
                return fail("activation discard is already marked resolved");
            }
            if pending.targets_chosen.len() != usize::from(target_count(pending.target_spec)) {
                return fail("activation discard began before targeting completed");
            }
            Ok(())
        }
        DiscardResume::None
        | DiscardResume::FinishSpellResolution { .. }
        | DiscardResume::FinishOptionalCost { .. } => Ok(()),
    }
}

/// If a discard is pending, either auto-resolves it (no real choice: the
/// legal pool is already <= the required count) or returns
/// `Decision::Discard`.
fn drain_pending_discard_or_decide(state: &mut GameState) -> Option<Decision> {
    let pd = state.engine.pending_discard.clone()?;
    if let Err((source, _message)) = validate_pending_discard_binding(state, &pd) {
        state.engine.halted = Some((UnsupportedMechanic::InvalidEffectContinuation, source));
        return Some(Decision::Halted {
            mechanic: UnsupportedMechanic::InvalidEffectContinuation,
            source,
        });
    }
    let exclude = match &pd.resume {
        DiscardResume::FinishCast { .. } => state.engine.pending_cast.as_ref().map(|p| p.spell),
        DiscardResume::FinishActivation { source, .. } => Some(*source),
        _ => None,
    };
    let choices: Vec<ObjectId> = state.players[pd.player.index()]
        .hand
        .iter()
        .copied()
        .filter(|&id| Some(id) != exclude)
        .collect();

    if choices.len() <= pd.count as usize {
        apply_discard(state, choices, pd);
        return None;
    }
    Some(Decision::Discard {
        player: pd.player,
        count: pd.count,
        choices,
    })
}

fn apply_discard(state: &mut GameState, chosen: Vec<ObjectId>, pending_discard: PendingDiscard) {
    if let Err((source, _message)) = validate_pending_discard_binding(state, &pending_discard) {
        state.engine.halted = Some((UnsupportedMechanic::InvalidEffectContinuation, source));
        return;
    }
    let cost_source = match &pending_discard.resume {
        DiscardResume::FinishCast {
            source_contract, ..
        } => Some(source_contract.source),
        DiscardResume::FinishActivation { source, .. } => Some(*source),
        _ => None,
    };
    if let Some(source) = cost_source {
        let mut unique = chosen.clone();
        unique.sort_unstable();
        unique.dedup();
        let exclude = match &pending_discard.resume {
            DiscardResume::FinishCast { .. } => Some(source),
            DiscardResume::FinishActivation { source, .. } => Some(*source),
            _ => None,
        };
        let hand = &state.players[pending_discard.player.index()].hand;
        if chosen.len() as u32 != pending_discard.count
            || unique.len() != chosen.len()
            || !chosen
                .iter()
                .all(|object| hand.contains(object) && Some(*object) != exclude)
        {
            state.engine.halted = Some((UnsupportedMechanic::InvalidEffectContinuation, source));
            return;
        }
    }
    let owned_cast = if matches!(pending_discard.resume, DiscardResume::FinishCast { .. }) {
        let staged_method = state
            .stack
            .last()
            .and_then(|item| item.v4.cast_method)
            .expect("validated FinishCast retains its exact top placeholder method");
        let pending = state
            .engine
            .pending_cast
            .take()
            .expect("validated FinishCast retains its pending cast");
        let def = &card_def::CARD_DEFS[state.objects.get(pending.spell).card_def as usize];
        let cast_method = finalized_cast_method(&pending, staged_method, def);
        Some((pending, cast_method))
    } else {
        None
    };
    state.engine.pending_discard = None;
    for &id in &chosen {
        let def = &card_def::CARD_DEFS[state.objects.get(id).card_def as usize];
        if def.is_castable() && def.madness_cost.is_some() {
            // 702.83b: a discarded Madness card is exiled instead of
            // graveyarded, and "as that card is exiled this way" a real
            // triggered ability fires offering its owner the chance to
            // cast it for its madness cost -- queued into
            // `pending_triggers` exactly like any other triggered ability
            // (`trigger::collect_and_process`'s Guttersnipe/Voldaren
            // Epicure triggers), not a special-cased side channel: it
            // goes through the same APNAP grouping/`Decision::
            // OrderTriggers` machinery if it happens to coincide with
            // another simultaneous trigger, and (`push_trigger_onto_stack`)
            // becomes a real `StackItem` that sits through normal priority
            // like anything else on the stack -- see that function's
            // `is_madness_offer` doc, and `advance_until_decision`'s
            // `top.madness_offer` check for where the actual `Decision::
            // ChooseMadnessCast` gets asked (at *resolution* time, not
            // discard time).
            event::propose_and_commit(state, ProposedEvent::zone_change(id, Zone::Exile));
            let owner = state.objects.get(id).owner;
            state.engine.pending_triggers.push(PendingTrigger {
                controller: owner,
                source: id,
                source_contract: Some(AbilitySourceContractV4::capture(state, id)),
                granted_by: None,
                effect: EffectOp::Sequence(vec![]),
                is_madness_offer: true,
                kicked: false,
                target_spec: TargetSpec::None,
                targets: Vec::new(),
                target_contracts: Vec::new(),
                placement_ordered: false,
                optional_additional_cost_paid: None,
                paid_cost_refs: Vec::new(),
            });
        } else {
            event::propose_and_commit(state, ProposedEvent::zone_change(id, Zone::Graveyard));
        }
    }
    match pending_discard.resume {
        DiscardResume::None => collect_and_queue_triggers(state),
        DiscardResume::FinishCast { .. } => {
            let (pending, cast_method) =
                owned_cast.expect("validated FinishCast owns its removed pending cast");
            let source = pending.spell;
            if finalize_owned_cast(state, pending, cast_method, chosen).is_err() {
                state.engine.halted =
                    Some((UnsupportedMechanic::InvalidEffectContinuation, source));
            }
        }
        DiscardResume::FinishActivation { .. } => {
            // 602.2b-h: every component of an activation's cost is paid
            // together, as one atomic action, the instant the interactive
            // part (which card(s) to discard) is determined -- not
            // deferred to `finalize_activation` (pushing the ability's
            // `StackItem`), which guards against re-paying by only paying
            // costs itself when there was no discard component at all
            // (`discard_count_in(ability.cost).is_none()`) and otherwise
            // trusts this branch already did it. Root-caused against
            // `game_20260713_002156_0014.txt` decision 45: Masked Meower's
            // `[DiscardCards(1), SacrificeSelf]` ability discards a Fiery
            // Temper, and the reference's own graveyard snapshot already
            // has Masked Meower in it *before* Fiery Temper's madness-cast
            // target pick is even offered -- i.e. the sacrifice happens
            // right here, alongside the discard. The Blood Token's ability
            // has the same `[..., DiscardCards(1), SacrificeSelf]` shape
            // and would hit the identical ordering bug were it ever
            // exercised this way.
            let p = state
                .engine
                .pending_activation
                .clone()
                .expect("validated activation discard retains its activation");
            let def = &card_def::CARD_DEFS[state.objects.get(p.source).card_def as usize];
            let ability = &def.activated_abilities[p.ability_index as usize];
            // Keep the state-changing payment outside `debug_assert!`: the
            // macro (including its argument) is compiled out in release
            // builds, but costs must be paid in every profile. The complete
            // activation/discard preflight above ran before any chosen card
            // moved, so this cannot fail for a reachable supported shape.
            let object_cost_ids = p
                .object_cost_chosen
                .iter()
                .map(|binding| binding.object)
                .collect::<Vec<_>>();
            let paid = pay_cost_components(
                state,
                p.controller,
                p.source,
                ability.cost,
                &object_cost_ids,
            );
            if !paid {
                state.engine.pending_activation = None;
                state.engine.halted =
                    Some((UnsupportedMechanic::InvalidEffectContinuation, p.source));
                return;
            }
            // 111.8/704.5d: paying `SacrificeSelf` here (a token's own
            // ability, e.g. the Blood Token's) can put a token into the
            // graveyard before the stack item is installed.
            trigger::sba_fixed_point(state);
            let p = state
                .engine
                .pending_activation
                .take()
                .expect("validated activation discard retains its activation");
            push_paid_activation(state, p, chosen);
        }
        DiscardResume::FinishSpellResolution { source, to_zone } => {
            // See `DiscardResume::FinishSpellResolution`'s doc: this is the
            // "move to post-resolution zone" step `resolve_top_of_stack`
            // deferred until the resolution-effect discard it triggered
            // (Faithless Looting) actually landed.
            event::propose_and_commit(state, ProposedEvent::zone_change(source, to_zone));
            collect_and_queue_triggers(state);
        }
        DiscardResume::FinishOptionalCost {
            source,
            controller,
            then,
            spell_resume,
        } => {
            // See `EffectOp::MayPayCostThen`'s doc: the discard sub-cost
            // just landed, so run the effect it unlocked (Highway Robbery:
            // draw two cards).
            let ctx = ExecCtx::no_targets(source, controller);
            effect::execute(&then, &ctx, state);
            // See `PendingOptionalCost::spell_resume`'s doc: only now is
            // the spell this optional cost belongs to *actually* fully
            // resolved -- perform the move `resolve_top_of_stack` deferred.
            if let Some((spell, to_zone)) = spell_resume {
                event::propose_and_commit(state, ProposedEvent::zone_change(spell, to_zone));
            }
            collect_and_queue_triggers(state);
        }
    }
}

/// If a resolution-time optional cost is pending (`EngineState::
/// pending_optional_cost`, staged by `EffectOp::MayPayCostThen`), returns
/// `Decision::ChooseOptionalCost`. Always asked, never auto-resolved:
/// declining is always a legal answer, so this is never a "no real choice"
/// situation the way a forced discard or a single-affordable-cast-mode is.
fn drain_pending_optional_cost_or_decide(state: &mut GameState) -> Option<Decision> {
    let poc = state.engine.pending_optional_cost.as_ref()?;
    Some(Decision::ChooseOptionalCost {
        player: poc.player,
        discard_payable: poc.discard_payable,
        sacrifice_payable: poc.sacrifice_payable,
    })
}

/// Presents whichever decision is live inside Chain Lightning's suspended
/// resolution. The target stage deliberately reuses ordinary
/// `Decision::ChooseTargets`/`Action::ChooseTarget`; the two preceding
/// binary decisions retain their distinct rules meanings on the RL wire.
fn drain_pending_spell_copy_or_decide(state: &mut GameState) -> Option<Decision> {
    let pending = state.engine.pending_spell_copy.clone()?;
    if validate_pending_spell_copy(state, &pending).is_err() {
        state.engine.halted = Some((
            UnsupportedMechanic::InvalidEffectContinuation,
            pending.resolving_source,
        ));
        return Some(Decision::Halted {
            mechanic: UnsupportedMechanic::InvalidEffectContinuation,
            source: pending.resolving_source,
        });
    }
    match pending.stage {
        SpellCopyStage::Payment => Some(Decision::ChooseSpellCopyPayment {
            player: pending.player,
            spell: pending.resolving_source,
        }),
        SpellCopyStage::Retarget => Some(Decision::ChooseSpellCopyRetarget {
            player: pending.player,
            copy: pending
                .copy_source
                .expect("Retarget stage requires a created spell copy"),
        }),
        SpellCopyStage::Target => {
            let copy = pending
                .copy_source
                .expect("Target stage requires a created spell copy");
            Some(Decision::ChooseTargets {
                player: pending.player,
                spell: copy,
                remaining: 1,
                legal_targets: legal_targets_for_controller_from_source(
                    TargetSpec::AnyTarget,
                    &[],
                    pending.player,
                    targeting_source_for_object(state, copy),
                    state,
                ),
                can_finish: false,
            })
        }
    }
}

/// Drives the generic v4 effect continuation. A suspended resolving item is
/// represented on the public stack while the player chooses, but is removed
/// again before interpreter execution resumes, matching the ordinary
/// resolution path's pop-before-effects invariant.
fn drain_pending_effect_or_decide(state: &mut GameState) -> Option<Decision> {
    state.engine.pending_effect.as_ref()?;
    if effect::validate_pending_effect_choice(state).is_err() {
        let source = state
            .engine
            .pending_effect
            .as_ref()
            .unwrap()
            .resolving_item
            .source;
        state.engine.halted = Some((UnsupportedMechanic::InvalidEffectContinuation, source));
        return None;
    }
    if state
        .engine
        .pending_effect
        .as_ref()
        .is_some_and(|pending| pending.choice.is_some())
    {
        let pending = state.engine.pending_effect.as_ref().unwrap();
        let choice = pending.choice.as_ref().unwrap();
        return Some(match choice {
            effect::PendingEffectChoice::ChooseOption {
                player, options, ..
            } => Decision::ChooseEffectOption {
                player: *player,
                source: pending.resolving_item.source,
                option_count: options
                    .len()
                    .try_into()
                    .expect("effect option count fits the u16 public contract"),
            },
            effect::PendingEffectChoice::SelectTargets {
                player,
                selected,
                legal,
                min_targets,
                max_targets,
                ..
            } => Decision::ChooseEffectTargets {
                player: *player,
                source: pending.resolving_item.source,
                selected_count: selected
                    .len()
                    .try_into()
                    .expect("effect selected-target count fits the u16 public contract"),
                min_targets: *min_targets,
                max_targets: *max_targets,
                legal_targets: legal.iter().map(|candidate| candidate.target).collect(),
                can_finish: selected.len() >= usize::from(*min_targets),
            },
            effect::PendingEffectChoice::ChooseBoolean {
                player,
                default,
                purpose,
                ..
            } => Decision::ChooseEffectBoolean {
                player: *player,
                source: pending.resolving_item.source,
                default: *default,
                purpose: purpose.clone(),
            },
        });
    }

    let pending = state.engine.pending_effect.as_ref().unwrap();
    let item = pending.resolving_item.clone();
    match state.stack.pop() {
        Some(public_item) if public_item == item => {}
        Some(public_item) => {
            state.stack.push(public_item);
            state.engine.halted =
                Some((UnsupportedMechanic::InvalidEffectContinuation, item.source));
            return None;
        }
        None => {
            state.engine.halted =
                Some((UnsupportedMechanic::InvalidEffectContinuation, item.source));
            return None;
        }
    }

    match effect::resume_resumable_resolution(state) {
        Ok(effect::ResumableProgress::Complete(completed)) => {
            if *completed != item {
                state.engine.halted =
                    Some((UnsupportedMechanic::InvalidEffectContinuation, item.source));
                return None;
            }
            if finish_resolved_stack_item(state, completed.as_ref()).is_err() {
                state.engine.halted =
                    Some((UnsupportedMechanic::InvalidEffectContinuation, item.source));
                return None;
            }
            collect_and_queue_triggers(state);
            reset_priority(state);
        }
        Ok(effect::ResumableProgress::Suspended) => {
            state.stack.push(item);
        }
        Err(_) => {
            state.stack.push(item.clone());
            state.engine.halted =
                Some((UnsupportedMechanic::InvalidEffectContinuation, item.source));
        }
    }
    None
}

/// If `Action::ChooseOptionalCost(SacrificeLand)` was just chosen, stages
/// *which* land(s) through the same per-pick `Decision::ChooseCostTargets`
/// flow `drain_pending_cast_or_decide` uses for Fireblast/Lava Dart (see
/// `sacrifice_lands_needed`'s doc for the per-pick auto-resolve rule this
/// mirrors: ask unless this single pick's own candidate pool is `<= 1`).
/// Once fully chosen, commits the sacrifice and runs `then` (Highway
/// Robbery's "draw two cards").
fn drain_pending_optional_cost_sacrifice_or_decide(state: &mut GameState) -> Option<Decision> {
    let pending = state.engine.pending_optional_cost_sacrifice.clone()?;
    if (pending.chosen.len() as u8) < pending.remaining {
        let candidates = sacrificeable_lands(pending.player, state, &pending.chosen);
        let remaining = pending.remaining - pending.chosen.len() as u8;
        if candidates.len() < usize::from(remaining) {
            state.engine.halted = Some((
                UnsupportedMechanic::InvalidEffectContinuation,
                pending.source,
            ));
            return Some(Decision::Halted {
                mechanic: UnsupportedMechanic::InvalidEffectContinuation,
                source: pending.source,
            });
        }
        if candidates.len() <= 1 {
            state
                .engine
                .pending_optional_cost_sacrifice
                .as_mut()
                .unwrap()
                .chosen
                .extend(candidates);
            return drain_pending_optional_cost_sacrifice_or_decide(state);
        }
        return Some(Decision::ChooseCostTargets {
            player: pending.player,
            source: pending.source,
            cost_kind: CostKind::SacrificeLands,
            remaining,
            candidates,
        });
    }
    let pending = state
        .engine
        .pending_optional_cost_sacrifice
        .take()
        .expect("checked Some above");
    commit_sacrifice(state, &pending.chosen);
    let ctx = ExecCtx::no_targets(pending.source, pending.player);
    effect::execute(&pending.then, &ctx, state);
    // See `PendingOptionalCost::spell_resume`'s doc: only now is the spell
    // this optional cost belongs to actually fully resolved.
    if let Some((spell, to_zone)) = pending.spell_resume {
        event::propose_and_commit(state, ProposedEvent::zone_change(spell, to_zone));
    }
    // `DiscardResume::FinishOptionalCost` (the discard-branch sibling of
    // this function, same "run `then`, then finish the spell's move"
    // shape) calls this; this branch didn't, an asymmetry bug -- so any
    // trigger condition on `pending.then`'s own effects (Sneaky Snacker's
    // `DrawNth(3)`, home zone Graveyard) never got matched at all when the
    // *sacrifice* sub-cost was the one paid, only ever when the *discard*
    // sub-cost was. Root-caused against `game_20260713_002158_0017.txt`
    // decision 225: Highway Robbery's sacrifice-a-Mountain branch draws the
    // controller's 3rd card of the turn, which should return their own
    // Sneaky Snacker from the graveyard to the battlefield (tapped) and
    // leave it sitting on the stack, unresolved, blocking a land drop until
    // it resolves -- the reference's very next record still withholds
    // `Play Mountain`, but the kernel, having never even queued the
    // trigger, offers it immediately.
    collect_and_queue_triggers(state);
    None
}

/// Stages `PendingCast` through its targets -> cast-mode -> additional-
/// cost-discard -> finalize pipeline, one stage per call (each stage that
/// makes progress `continue`s the outer loop instead of looping here, so
/// `pending_discard`/triggers/etc. staged along the way always get
/// checked first).
pub(crate) fn validate_pending_cast(
    state: &GameState,
    pending: &PendingCast,
) -> Result<(), String> {
    let source = state
        .objects
        .try_get(pending.spell)
        .ok_or("pending cast source object is missing")?;
    let def = card_def::CARD_DEFS
        .get(source.card_def as usize)
        .ok_or("pending cast source definition is missing")?;
    if !def.is_executable() || !def.is_castable() {
        return Err("pending cast source is not an executable spell".to_string());
    }
    if def.omen.is_some() && supported_omen(def).is_none() {
        return Err("pending cast source has an unsupported Omen definition".to_string());
    }
    if def.bestow.is_some() && supported_bestow(def).is_none() {
        return Err("pending cast source has an unsupported Bestow definition".to_string());
    }
    if source.spell_copy_origin.is_some() {
        return Err("a virtual spell copy cannot be staged as a cast".to_string());
    }
    if source.zone != Zone::Stack
        || source.controller != pending.controller
        || pending.source_contract.spell_copy_origin.is_some()
    {
        return Err("pending cast source contract changed".to_string());
    }
    let mut stack_matches = state.stack.iter().enumerate().filter(|(_, item)| {
        item.kind == StackItemKind::Spell
            && item.v4.source_contract == Some(pending.source_contract)
    });
    let (placeholder_index, placeholder) = stack_matches
        .next()
        .ok_or("pending cast source has no stack placeholder")?;
    if stack_matches.next().is_some() || placeholder_index + 1 != state.stack.len() {
        return Err("pending cast placeholder is not the unique stack top".to_string());
    }
    if placeholder.v4.stack_item_id == StackItemId::default()
        || state
            .stack
            .iter()
            .filter(|item| item.v4.stack_item_id == placeholder.v4.stack_item_id)
            .count()
            != 1
    {
        return Err("pending cast placeholder has no unique stack incarnation".to_string());
    }
    if placeholder.kind != StackItemKind::Spell
        || placeholder.source != pending.spell
        || placeholder.controller != pending.controller
        || placeholder.is_copy
        || placeholder.inline_effect.is_some()
        || !placeholder.targets.is_empty()
        || !placeholder.discarded.is_empty()
        || placeholder.madness_offer
        || placeholder.mode_chosen != 0
        || placeholder.kicked
        || placeholder.v4.source_contract != Some(pending.source_contract)
        || placeholder.v4.face_index != 0
        || placeholder.v4.x_value != 0
        || !placeholder.v4.paid_cost_refs.is_empty()
        || placeholder.v4.optional_additional_cost_paid.is_some()
        || placeholder.v4.target_spec.is_some()
        || !placeholder.v4.target_contracts.is_empty()
        || placeholder.v4.activated_ability_index.is_some()
    {
        return Err(
            "pending cast placeholder carries finalized or incompatible metadata".to_string(),
        );
    }
    validate_spell_stack_source(state, placeholder)?;
    let method = placeholder
        .v4
        .cast_method
        .ok_or("pending cast placeholder lost its cast method")?;
    if pending.source_contract != StackSourceContractV4::capture(state, pending.spell, method) {
        return Err("pending cast source contract changed".to_string());
    }
    if pending.is_flashback != (method == CastMethodV4::Flashback)
        || placeholder.is_flashback != pending.is_flashback
    {
        return Err("pending cast flashback metadata disagrees with its cast method".to_string());
    }

    let appears_outside_stack = [PlayerId::P0, PlayerId::P1].into_iter().any(|player| {
        let zones = &state.players[player.index()];
        zones.library.contains(&pending.spell)
            || zones.hand.contains(&pending.spell)
            || zones.battlefield.contains(&pending.spell)
            || zones.graveyard.contains(&pending.spell)
    }) || state.exile.contains(&pending.spell)
        || state.command.contains(&pending.spell);
    if appears_outside_stack {
        return Err("pending cast source is still indexed in an ordinary zone".to_string());
    }

    let has_prior_exile_permission = || {
        pending
            .source_contract
            .zone_change_count
            .checked_sub(1)
            .is_some_and(|generation| {
                state
                    .engine
                    .exile_play_permissions
                    .iter()
                    .any(|permission| {
                        permission.object == pending.spell
                            && permission.holder == pending.controller
                            && permission.zone_change_generation == generation
                            && permission.play_or_cast == PlayOrCast::Cast
                    })
            })
    };
    match method {
        CastMethodV4::Normal => {
            if pending.origin_zone != Zone::Hand
                && !(pending.origin_zone == Zone::Exile && has_prior_exile_permission())
            {
                return Err(
                    "normal cast has an invalid origin zone or exile permission".to_string()
                );
            }
            if pending.origin_zone == Zone::Hand && source.owner != pending.controller {
                return Err(
                    "a normal hand cast must be controlled by its physical owner".to_string(),
                );
            }
            if def.alt_cost.is_none() {
                if pending.cast_mode != Some(CastMode::Normal) {
                    return Err("fixed normal cast has a noncanonical cast mode".to_string());
                }
            } else if !matches!(
                pending.cast_mode,
                None | Some(CastMode::Normal) | Some(CastMode::Alternative)
            ) {
                return Err("normal cast method disagrees with its staged cast mode".to_string());
            }
        }
        CastMethodV4::Alternative | CastMethodV4::Omen | CastMethodV4::Bestow => {
            return Err(
                "pending cast stamped its selected cast method before finalization".to_string(),
            );
        }
        CastMethodV4::Flashback => {
            if pending.origin_zone != Zone::Graveyard
                || source.owner != pending.controller
                || def.flashback.is_none()
                || def.escape.is_some()
                || pending.cast_mode != Some(CastMode::Normal)
            {
                return Err("flashback cast has a noncanonical origin or definition".to_string());
            }
        }
        CastMethodV4::Escape => {
            if pending.origin_zone != Zone::Graveyard
                || source.owner != pending.controller
                || def.escape.is_none()
                || def.flashback.is_some()
                || def.additional_cost.is_some()
                || def.kicker_cost.is_some()
                || def.generic_cost_reduction.is_some()
                || pending.cast_mode != Some(CastMode::Normal)
            {
                return Err("escape cast has a noncanonical origin or definition".to_string());
            }
        }
        CastMethodV4::Plotted => {
            if pending.origin_zone != Zone::Exile
                || source.owner != pending.controller
                || def.plot_cost.is_none()
                || source
                    .plotted_turn
                    .is_none_or(|plotted_turn| plotted_turn >= state.turn)
                || pending.cast_mode != Some(CastMode::Normal)
            {
                return Err("Plotted cast has a noncanonical origin or marker".to_string());
            }
        }
        CastMethodV4::Madness => {
            if pending.origin_zone != Zone::Exile
                || source.owner != pending.controller
                || def.madness_cost.is_none()
                || pending.cast_mode != Some(CastMode::Normal)
            {
                return Err("Madness cast has a noncanonical origin or definition".to_string());
            }
        }
    }

    if pending.target_spec != def.target_spec {
        return Err("pending cast target specification changed".to_string());
    }
    let mode_shape_valid = match pending.mode_chosen {
        None => has_spell_form_choice(def),
        Some(mode) => {
            spell_form_target_spec(def, mode).is_some() && (has_spell_form_choice(def) || mode == 0)
        }
    };
    if !mode_shape_valid {
        return Err("pending cast mode selection is noncanonical".to_string());
    }
    if pending.mode_chosen.is_none()
        && (!pending.targets_chosen.is_empty() || pending.target_selection_finished)
    {
        return Err(
            "pending cast chose or finished targets before choosing its spell mode".to_string(),
        );
    }
    let active_spec = pending
        .mode_chosen
        .map(|mode| {
            spell_form_target_spec(def, mode)
                .ok_or("pending cast selected a nonexistent spell form".to_string())
        })
        .transpose()?;
    let required_targets = active_spec.map_or(0, |spec| usize::from(target_count(spec)));
    if pending.targets_chosen.len() > required_targets
        || !target_contracts_are_structurally_valid(
            state,
            &pending.targets_chosen,
            &pending.target_contracts,
        )
    {
        return Err("pending cast target prefix or contracts are malformed".to_string());
    }
    if pending.target_selection_finished {
        let spec = active_spec.ok_or("pending cast finished targets without an active form")?;
        let min = usize::from(target_min_count(spec));
        if min == required_targets
            || pending.targets_chosen.len() < min
            || pending.targets_chosen.len() >= required_targets
        {
            return Err("pending cast target-finish marker is noncanonical".to_string());
        }
    }
    if let Some(spec) = active_spec {
        let mut prefix = Vec::new();
        for &target in &pending.targets_chosen {
            if !completable_next_targets_for_controller_and_source(
                spec,
                &prefix,
                pending.controller,
                targeting_source_for_object(state, pending.spell),
                state,
            )
            .contains(&target)
            {
                return Err("pending cast carries an illegal target prefix".to_string());
            }
            prefix.push(target);
        }
    }
    if def.kicker_cost.is_none() {
        if pending.kicked != Some(false) {
            return Err("non-kicker spell carries a noncanonical kicker stage".to_string());
        }
    } else if !matches!(pending.kicked, None | Some(false) | Some(true)) {
        return Err("pending cast kicker stage is malformed".to_string());
    }

    match (
        def.optional_additional_cost,
        pending.optional_additional_cost_paid,
    ) {
        (None, Some(false)) => {
            if !pending.optional_additional_cost_chosen.is_empty()
                || pending.optional_additional_cost_selection_finished
            {
                return Err(
                    "cast without an optional additional cost carries payment state".to_string(),
                );
            }
        }
        (Some(_), None) | (Some(_), Some(false)) => {
            if !pending.optional_additional_cost_chosen.is_empty()
                || pending.optional_additional_cost_selection_finished
            {
                return Err("unpaid optional additional cost carries payment objects".to_string());
            }
        }
        (Some(OptionalAdditionalCostDef::CollectEvidence { minimum_mana_value }), Some(true)) => {
            let candidates = collect_evidence_candidates(pending.controller, state)?;
            if !selected_bindings_are_unique_subset(
                &pending.optional_additional_cost_chosen,
                &candidates,
            ) || pending
                .optional_additional_cost_chosen
                .iter()
                .any(|binding| !optional_cost_binding_is_live(state, *binding))
            {
                return Err(
                    "Collect Evidence selection is not a unique live graveyard subset".to_string(),
                );
            }
            let selected_sum =
                collect_evidence_mana_value(state, &pending.optional_additional_cost_chosen)?;
            let remaining = candidates
                .iter()
                .copied()
                .filter(|binding| !pending.optional_additional_cost_chosen.contains(binding))
                .collect::<Vec<_>>();
            let remaining_sum = collect_evidence_mana_value(state, &remaining)?;
            if selected_sum
                .checked_add(remaining_sum)
                .is_none_or(|total| total < u32::from(minimum_mana_value))
            {
                return Err("Collect Evidence selection can no longer meet its minimum".to_string());
            }
            if pending.optional_additional_cost_selection_finished
                && selected_sum < u32::from(minimum_mana_value)
            {
                return Err("Collect Evidence finished below its minimum mana value".to_string());
            }
        }
        (Some(OptionalAdditionalCostDef::Bargain), Some(true)) => {
            let candidates = bargain_candidates(pending.controller, state)?;
            if pending.optional_additional_cost_selection_finished
                || pending.optional_additional_cost_chosen.len() > 1
                || !selected_bindings_are_unique_subset(
                    &pending.optional_additional_cost_chosen,
                    &candidates,
                )
                || pending
                    .optional_additional_cost_chosen
                    .iter()
                    .any(|binding| !optional_cost_binding_is_live(state, *binding))
            {
                return Err("Bargain selection is not one live eligible permanent".to_string());
            }
        }
        _ => return Err("pending cast optional additional-cost state is noncanonical".to_string()),
    }

    let selected_cost =
        active_spec.and_then(|_| pending_cast_selected_mana_cost(def, pending, state));
    let expected_seed_x = if def.cost.x_count == 0
        && supported_bestow(def).is_none_or(|bestow| bestow.cost.x_count == 0)
    {
        Some(0)
    } else {
        None
    };
    if let Some(cost) = selected_cost {
        if cost.x_count == 0 {
            if pending.x_value != Some(0) {
                return Err("non-X cast carries a noncanonical X stage".to_string());
            }
        } else if cost.x_count != 1 {
            return Err("pending cast has an unsupported number of X symbols".to_string());
        } else if let Some(x_value) = pending.x_value {
            if mana::can_pay(&cost, x_value, pending.controller, state).is_none() {
                return Err("pending cast chose an unaffordable X value".to_string());
            }
            if active_spec.is_none_or(|spec| !pending_cast_targeting_is_complete(pending, spec))
                || pending.cast_mode.is_none()
            {
                return Err("pending cast chose X before modes and targets completed".to_string());
            }
        }
    } else if pending.x_value != expected_seed_x {
        return Err("pending cast advanced X before choosing its spell form".to_string());
    }

    let sacrifice_needed = sacrifice_lands_needed(pending, def);
    let graveyard_exile_needed = graveyard_exile_cards_needed(pending, def);
    let controlled_sacrifice = controlled_permanent_sacrifice_needed(pending, def);
    let tap_permanent_filter = cast_tap_permanent_filter_needed(pending, def);
    let active_object_cost_families = u8::from(sacrifice_needed != 0)
        + u8::from(graveyard_exile_needed != 0)
        + u8::from(controlled_sacrifice.is_some())
        + u8::from(tap_permanent_filter.is_some());
    if active_object_cost_families > 1 {
        return Err("pending cast has multiple interactive object-cost families".to_string());
    }
    let interactive_object_cost_needed = sacrifice_needed
        .max(graveyard_exile_needed)
        .max(controlled_sacrifice.map_or(0, |(count, _)| count))
        .max(u8::from(tap_permanent_filter.is_some()));
    if pending.sacrifice_chosen.len() > usize::from(interactive_object_cost_needed) {
        return Err("pending cast chose too many interactive cost objects".to_string());
    }
    if sacrifice_needed != 0 {
        let mut chosen = pending.sacrifice_chosen.clone();
        chosen.sort_unstable();
        chosen.dedup();
        if chosen.len() != pending.sacrifice_chosen.len()
            || pending.sacrifice_chosen.iter().any(|&land| {
                land == pending.spell
                    || !state.players[pending.controller.index()]
                        .battlefield
                        .contains(&land)
                    || state
                        .objects
                        .try_get(land)
                        .is_none_or(|object| !card_def::CARD_DEFS[object.card_def as usize].is_land)
            })
        {
            return Err("pending cast sacrifice prefix is not unique controlled lands".to_string());
        }
    } else if graveyard_exile_needed != 0 {
        if checked_graveyard_exile_candidates(
            pending.controller,
            pending.spell,
            state,
            &pending.sacrifice_chosen,
        )
        .is_none()
        {
            return Err(
                "pending cast graveyard-exile prefix or graveyard index is malformed".to_string(),
            );
        }
    } else if let Some((_, filter)) = controlled_sacrifice {
        let candidates =
            sacrificeable_controlled_permanents(pending.controller, filter, state, &[]);
        let mut chosen = pending.sacrifice_chosen.clone();
        chosen.sort_unstable();
        chosen.dedup();
        if chosen.len() != pending.sacrifice_chosen.len()
            || pending
                .sacrifice_chosen
                .iter()
                .any(|object| !candidates.contains(object))
        {
            return Err(
                "pending cast sacrifice prefix is not unique matching controlled permanents"
                    .to_string(),
            );
        }
    } else if let Some(filter) = tap_permanent_filter {
        let candidates = tap_permanent_cost_candidates(pending.controller, state, filter, &[]);
        if pending.sacrifice_chosen.len() > 1
            || pending
                .sacrifice_chosen
                .iter()
                .any(|object| !candidates.contains(object))
        {
            return Err(
                "pending cast tap-cost prefix is not an untapped matching controlled permanent"
                    .to_string(),
            );
        }
    } else if !pending.sacrifice_chosen.is_empty() {
        return Err("pending cast without an object cost carries chosen objects".to_string());
    }
    if !pending.sacrifice_chosen.is_empty()
        && (pending.cast_mode.is_none()
            || active_spec.is_none_or(|spec| !pending_cast_targeting_is_complete(pending, spec))
            || pending.kicked.is_none())
    {
        return Err("pending cast chose object costs before prior stages completed".to_string());
    }
    if (!pending.optional_additional_cost_chosen.is_empty()
        || pending.optional_additional_cost_selection_finished)
        && (pending.kicked.is_none()
            || pending.mode_chosen.is_none()
            || active_spec.is_none_or(|spec| !pending_cast_targeting_is_complete(pending, spec))
            || pending.cast_mode.is_none()
            || pending.sacrifice_chosen.len() != usize::from(interactive_object_cost_needed))
    {
        return Err(
            "optional additional-cost objects were chosen before prior cast stages completed"
                .to_string(),
        );
    }

    let has_chosen_creature_cost = has_chosen_creature_additional_cost(def);
    if !has_chosen_creature_cost {
        if pending.chosen_creature_cost_zone.is_some() || pending.chosen_creature_cost.is_some() {
            return Err("cast without a chosen-creature cost carries its binding".to_string());
        }
    } else {
        if pending.chosen_creature_cost.is_some() && pending.chosen_creature_cost_zone.is_none() {
            return Err("chosen-creature cost lost its selected zone".to_string());
        }
        if let Some(zone) = pending.chosen_creature_cost_zone {
            let candidates = chosen_creature_cost_candidates(pending.controller, zone, state);
            if let Some(binding) = pending.chosen_creature_cost {
                if !candidates.contains(&binding) {
                    return Err("chosen-creature cost changed incarnation or legality".to_string());
                }
            }
        }
        if (pending.chosen_creature_cost_zone.is_some() || pending.chosen_creature_cost.is_some())
            && (pending.kicked.is_none()
                || active_spec
                    .is_none_or(|spec| !pending_cast_targeting_is_complete(pending, spec))
                || pending.cast_mode.is_none()
                || pending.sacrifice_chosen.len() != usize::from(interactive_object_cost_needed))
        {
            return Err("chosen-creature cost advanced before prior cast stages".to_string());
        }
    }

    let interactive_discard = def.additional_cost.and_then(discard_count_in);
    let seeded_mode = if has_spell_form_choice(def) {
        None
    } else {
        Some(0)
    };
    let has_cast_mode_choice = def.alt_cost.is_some();
    let seeded_cast_mode = if method == CastMethodV4::Normal && has_cast_mode_choice {
        None
    } else {
        Some(CastMode::Normal)
    };
    let additional_is_at_seed = if def.additional_cost.is_none() {
        pending.additional_cost_discarded.as_deref() == Some(&[])
    } else {
        pending.additional_cost_discarded.is_none()
    };
    let optional_is_at_seed = if def.optional_additional_cost.is_none() {
        pending.optional_additional_cost_paid == Some(false)
    } else {
        pending.optional_additional_cost_paid.is_none()
    } && pending.optional_additional_cost_chosen.is_empty()
        && !pending.optional_additional_cost_selection_finished;
    let later_fields_are_seeded = |require_seeded_cast_mode: bool| {
        (!require_seeded_cast_mode || pending.cast_mode == seeded_cast_mode)
            && pending.sacrifice_chosen.is_empty()
            && pending.optional_additional_cost_chosen.is_empty()
            && !pending.optional_additional_cost_selection_finished
            && pending.chosen_creature_cost_zone.is_none()
            && pending.chosen_creature_cost.is_none()
            && pending.x_value == expected_seed_x
            && additional_is_at_seed
            && state.engine.pending_discard.is_none()
    };
    if pending.kicked.is_none()
        && (pending.mode_chosen != seeded_mode
            || !pending.targets_chosen.is_empty()
            || pending.target_selection_finished
            || !optional_is_at_seed
            || !later_fields_are_seeded(true))
    {
        return Err("pending cast advanced past an unresolved kicker choice".to_string());
    }
    if pending.optional_additional_cost_paid.is_none()
        && (pending.mode_chosen != seeded_mode
            || !pending.targets_chosen.is_empty()
            || pending.target_selection_finished
            || !later_fields_are_seeded(true))
    {
        return Err(
            "pending cast advanced past an unresolved optional additional-cost choice".to_string(),
        );
    }
    if has_spell_form_choice(def)
        && pending.mode_chosen.is_none()
        && (!pending.targets_chosen.is_empty()
            || pending.target_selection_finished
            || !later_fields_are_seeded(true))
    {
        return Err("pending cast advanced past an unresolved spell-mode choice".to_string());
    }
    if pending.mode_chosen.is_some()
        && active_spec.is_some_and(|spec| !pending_cast_targeting_is_complete(pending, spec))
        && !later_fields_are_seeded(true)
    {
        return Err("pending cast advanced past incomplete targeting".to_string());
    }
    if method == CastMethodV4::Normal
        && has_cast_mode_choice
        && pending.cast_mode.is_some()
        && (pending.kicked.is_none()
            || pending.mode_chosen.is_none()
            || active_spec.is_none_or(|spec| !pending_cast_targeting_is_complete(pending, spec)))
    {
        return Err("pending cast chose a cast mode before prior stages completed".to_string());
    }
    if pending.cast_mode.is_none()
        && (!pending.sacrifice_chosen.is_empty()
            || !additional_is_at_seed
            || state.engine.pending_discard.is_some())
    {
        return Err("pending cast advanced past an unresolved cast-mode choice".to_string());
    }
    if pending.sacrifice_chosen.len() < usize::from(interactive_object_cost_needed)
        && (!additional_is_at_seed || state.engine.pending_discard.is_some())
    {
        return Err("pending cast advanced past an incomplete object cost".to_string());
    }
    if !optional_additional_cost_selection_is_complete(pending, def, state)?
        && (!additional_is_at_seed || state.engine.pending_discard.is_some())
    {
        return Err(
            "pending cast advanced past an incomplete optional additional cost".to_string(),
        );
    }
    if has_chosen_creature_cost
        && pending.chosen_creature_cost.is_none()
        && (!additional_is_at_seed || state.engine.pending_discard.is_some())
    {
        return Err("pending cast advanced past an incomplete chosen-creature cost".to_string());
    }
    if pending.x_value.is_none()
        && (!additional_is_at_seed || state.engine.pending_discard.is_some())
    {
        return Err("pending cast advanced past an unresolved X value".to_string());
    }
    match (
        def.additional_cost,
        interactive_discard,
        pending.additional_cost_discarded.as_deref(),
    ) {
        (None, None, Some([])) => {}
        (Some(_), Some(_), None) => {}
        (Some(_), None, None | Some([])) => {}
        (None, _, _) => {
            return Err(
                "cast without an additional cost has noncanonical discard state".to_string(),
            )
        }
        (Some(_), Some(_), Some(_)) => {
            return Err("interactive cast discard cannot self-authenticate payment".to_string())
        }
        (Some(_), None, Some(_)) => {
            return Err("noninteractive additional cost carries discard objects".to_string())
        }
    }

    if state.engine.pending_activation.is_some()
        || state.engine.pending_effect.is_some()
        || state.engine.pending_optional_cost.is_some()
        || state.engine.pending_optional_cost_sacrifice.is_some()
        || state.engine.pending_spell_copy.is_some()
        || !state.engine.pending_triggers.is_empty()
        || state.engine.pending_kicked_source.is_some()
    {
        return Err("pending cast coexists with another staged engine context".to_string());
    }
    if let Some(discard) = state.engine.pending_discard.as_ref() {
        let DiscardResume::FinishCast {
            source_contract,
            controller,
        } = discard.resume
        else {
            return Err("pending cast coexists with a non-cast discard".to_string());
        };
        let expected_count = interactive_discard
            .ok_or("pending cast discard has no definition-owned interactive cost")?;
        if source_contract != pending.source_contract
            || controller != pending.controller
            || discard.player != pending.controller
            || discard.count != u32::from(expected_count)
            || active_spec.is_none_or(|spec| !pending_cast_targeting_is_complete(pending, spec))
            || pending.cast_mode.is_none()
            || pending.kicked.is_none()
            || pending.sacrifice_chosen.len() != usize::from(interactive_object_cost_needed)
            || !optional_additional_cost_selection_is_complete(pending, def, state)?
        {
            return Err("pending cast discard binding or stage changed".to_string());
        }
    }
    Ok(())
}

fn drain_pending_cast_or_decide(state: &mut GameState) -> Option<Decision> {
    let pending = state.engine.pending_cast.clone()?;
    if validate_pending_cast(state, &pending).is_err() {
        state.engine.halted = Some((
            UnsupportedMechanic::InvalidEffectContinuation,
            pending.spell,
        ));
        return None;
    }
    let def = &card_def::CARD_DEFS[state.objects.get(pending.spell).card_def as usize];
    let staged_method = state
        .stack
        .last()
        .and_then(|item| item.v4.cast_method)
        .expect("validated pending cast retains its placeholder method");

    // Ordinary, Flashback, and Plotted casts are pre-filtered before they
    // reach `begin_cast`, but Madness intentionally bypasses that offer-time
    // gate and restored snapshots are untrusted. Re-check the complete
    // mandatory target assignment against the announcing spell's real
    // self-excluding stack state, and revert any impossible announcement
    // instead of exposing an empty targeting decision.
    let targeting_can_complete = match pending.mode_chosen {
        Some(mode) => spell_form_target_spec(def, mode).is_some_and(|spec| {
            target_prefix_can_complete_for_controller_and_source(
                spec,
                &pending.targets_chosen,
                pending.controller,
                targeting_source_for_object(state, pending.spell),
                state,
            )
        }),
        None => !viable_pending_spell_forms(def, &pending, state).is_empty(),
    };
    if !targeting_can_complete {
        let cast_method = finalized_cast_method(&pending, staged_method, def);
        let pending = state.engine.pending_cast.take().unwrap();
        abort_cast(state, pending, cast_method);
        return None;
    }

    // Kicker (Goblin Bushwhacker) is decided before targets/modes, same as
    // any other cast-time cost choice -- only asked when paying the base
    // cost *and* the kicker cost together is currently affordable; every
    // card in this pool with `kicker_cost` has no `alt_cost`, so checking
    // against the effective normal cost (never `def.alt_cost`) is exhaustive
    // here.
    if let Some(kicker_cost) = def.kicker_cost {
        if pending.kicked.is_none() {
            let normal_cost = effective_normal_cast_cost(def, pending.controller, state);
            let payable =
                mana::can_pay_combined(&[&normal_cost, &kicker_cost], 0, pending.controller, state)
                    .is_some();
            if payable {
                return Some(Decision::ChooseKicker {
                    player: pending.controller,
                    spell: pending.spell,
                });
            }
            state.engine.pending_cast.as_mut().unwrap().kicked = Some(false);
            return drain_pending_cast_or_decide(state);
        }
    }

    if let Some(kind) = def.optional_additional_cost {
        if pending.optional_additional_cost_paid.is_none() {
            match optional_additional_cost_is_payable(kind, pending.controller, state) {
                Ok(true) => {
                    return Some(Decision::ChooseEffectOption {
                        player: pending.controller,
                        source: pending.spell,
                        option_count: 2,
                    });
                }
                Ok(false) => {
                    state
                        .engine
                        .pending_cast
                        .as_mut()
                        .expect("validated cast remains pending")
                        .optional_additional_cost_paid = Some(false);
                    return drain_pending_cast_or_decide(state);
                }
                Err(_) => {
                    state.engine.halted = Some((
                        UnsupportedMechanic::InvalidEffectContinuation,
                        pending.spell,
                    ));
                    return None;
                }
            }
        }
    }

    // 601.2b/c: choose a mode before targets, but only a mode whose complete
    // mandatory target assignment exists. If exactly one printed mode is
    // viable there is no policy choice, so retain that mode's original
    // index (including index 1) silently; if both are viable, expose the
    // existing two-action schema-v4 decision unchanged.
    if has_spell_form_choice(def) && pending.mode_chosen.is_none() {
        let viable_modes = viable_pending_spell_forms(def, &pending, state);
        if viable_modes.is_empty() {
            // Defensive against a state mutation between the general gate
            // above and this mode stage (none exists in today's pipeline).
            let cast_method = finalized_cast_method(&pending, staged_method, def);
            let pending = state.engine.pending_cast.take().unwrap();
            abort_cast(state, pending, cast_method);
            return None;
        }
        if viable_modes.len() == 1 {
            state.engine.pending_cast.as_mut().unwrap().mode_chosen = Some(viable_modes[0]);
            return drain_pending_cast_or_decide(state);
        }
        return Some(Decision::ChooseSpellMode {
            player: pending.controller,
            spell: pending.spell,
            mode_count: printed_spell_form_count(def),
            legal_modes: viable_modes,
        });
    }
    let active_target_spec = spell_form_target_spec(
        def,
        pending
            .mode_chosen
            .expect("spell-form choice completed before targeting"),
    )
    .expect("mode shape validated at drain entry");

    let need = target_count(active_target_spec);
    if !pending_cast_targeting_is_complete(&pending, active_target_spec) {
        return Some(Decision::ChooseTargets {
            player: pending.controller,
            spell: pending.spell,
            remaining: need - pending.targets_chosen.len() as u8,
            legal_targets: completable_next_targets_for_controller_and_source(
                active_target_spec,
                &pending.targets_chosen,
                pending.controller,
                targeting_source_for_object(state, pending.spell),
                state,
            ),
            can_finish: pending.targets_chosen.len()
                >= usize::from(target_min_count(active_target_spec)),
        });
    }

    if pending.cast_mode.is_none() {
        let modes = payable_cast_modes(def, &pending, state);
        if modes.len() >= 2 {
            return Some(Decision::ChooseCastMode {
                player: pending.controller,
                spell: pending.spell,
                options: modes,
            });
        }
        let Some(mode) = modes.first().copied() else {
            let cast_method = finalized_cast_method(&pending, staged_method, def);
            let pending = state.engine.pending_cast.take().unwrap();
            abort_cast(state, pending, cast_method);
            return None;
        };
        state.engine.pending_cast.as_mut().unwrap().cast_mode = Some(mode);
        return drain_pending_cast_or_decide(state);
    }

    let sacrifice_needed = sacrifice_lands_needed(&pending, def);
    if (pending.sacrifice_chosen.len() as u8) < sacrifice_needed {
        let candidates = sacrificeable_lands(pending.controller, state, &pending.sacrifice_chosen);
        let remaining = sacrifice_needed - pending.sacrifice_chosen.len() as u8;
        if candidates.len() < usize::from(remaining) {
            let cast_method = finalized_cast_method(&pending, staged_method, def);
            let pending = state.engine.pending_cast.take().unwrap();
            abort_cast(state, pending, cast_method);
            return None;
        }
        if candidates.len() <= 1 {
            // No real choice for *this single pick* (0 or 1 legal
            // candidate) -- auto-resolve just this one and let the next
            // loop pass re-derive whether the *following* pick (if any)
            // is still real. Deliberately per-pick, not "auto-resolve the
            // whole remaining batch whenever candidates.len() <=
            // remaining": empirically, the reference always logs a real
            // decision for every pick whose own candidate pool has 2+
            // legal choices, even when the *aggregate* choice is forced
            // (e.g. exactly 2 Mountains for Fireblast's 2-land cost still
            // logs one real 2-candidate pick before the final,
            // now-1-candidate pick goes silent) -- root-caused against the
            // v4 corpus's own `(candidate_count...)` sequence per Fireblast/
            // Lava Dart episode (e.g. Fireblast's dominant shape is exactly
            // 2 post-target records, `(target=N, sac1=2)`, never a 3rd
            // `sac2=1` record; Lava Dart with exactly 1 controlled Mountain
            // logs *zero* post-target records at all -- both match "ask
            // until this pick's own pool is <= 1", not "ask until the
            // aggregate is forced"). A first version of this auto-resolve
            // used the aggregate `candidates.len() <= remaining` test and
            // silently over-suppressed exactly this shape (verified against
            // `game_20260713_002152_0008.txt` decision 42: kernel skipped
            // straight past a real 2-candidate Fireblast sacrifice pick the
            // trace logs).
            state
                .engine
                .pending_cast
                .as_mut()
                .unwrap()
                .sacrifice_chosen
                .extend(candidates);
            return drain_pending_cast_or_decide(state);
        }
        return Some(Decision::ChooseCostTargets {
            player: pending.controller,
            source: pending.spell,
            cost_kind: CostKind::SacrificeLands,
            remaining,
            candidates,
        });
    }

    let graveyard_exile_needed = graveyard_exile_cards_needed(&pending, def);
    if (pending.sacrifice_chosen.len() as u8) < graveyard_exile_needed {
        let Some(candidates) = checked_graveyard_exile_candidates(
            pending.controller,
            pending.spell,
            state,
            &pending.sacrifice_chosen,
        ) else {
            let cast_method = finalized_cast_method(&pending, staged_method, def);
            let pending = state.engine.pending_cast.take().unwrap();
            abort_cast(state, pending, cast_method);
            return None;
        };
        let remaining = graveyard_exile_needed - pending.sacrifice_chosen.len() as u8;
        if candidates.len() < usize::from(remaining) {
            let cast_method = finalized_cast_method(&pending, staged_method, def);
            let pending = state.engine.pending_cast.take().unwrap();
            abort_cast(state, pending, cast_method);
            return None;
        }
        if candidates.len() <= 1 {
            state
                .engine
                .pending_cast
                .as_mut()
                .unwrap()
                .sacrifice_chosen
                .extend(candidates);
            return drain_pending_cast_or_decide(state);
        }
        return Some(Decision::ChooseCostTargets {
            player: pending.controller,
            source: pending.spell,
            cost_kind: CostKind::ExileFromGraveyard,
            remaining,
            candidates,
        });
    }

    if let Some((needed, filter)) = controlled_permanent_sacrifice_needed(&pending, def) {
        if (pending.sacrifice_chosen.len() as u8) < needed {
            let candidates = sacrificeable_controlled_permanents(
                pending.controller,
                filter,
                state,
                &pending.sacrifice_chosen,
            );
            let remaining = needed - pending.sacrifice_chosen.len() as u8;
            if candidates.len() < usize::from(remaining) {
                let cast_method = finalized_cast_method(&pending, staged_method, def);
                let pending = state.engine.pending_cast.take().unwrap();
                abort_cast(state, pending, cast_method);
                return None;
            }
            if candidates.len() <= 1 {
                state
                    .engine
                    .pending_cast
                    .as_mut()
                    .unwrap()
                    .sacrifice_chosen
                    .extend(candidates);
                return drain_pending_cast_or_decide(state);
            }
            return Some(Decision::ChooseCostTargets {
                player: pending.controller,
                source: pending.spell,
                cost_kind: match filter {
                    PermanentFilter::Creature => CostKind::SacrificeCreatures,
                    PermanentFilter::Artifact => CostKind::SacrificeArtifacts,
                    PermanentFilter::ArtifactOrCreature => CostKind::SacrificePermanents,
                },
                remaining,
                candidates,
            });
        }
    }

    if let Some(filter) = cast_tap_permanent_filter_needed(&pending, def) {
        if pending.sacrifice_chosen.is_empty() {
            let candidates = tap_permanent_cost_candidates(pending.controller, state, filter, &[]);
            if candidates.is_empty() {
                let cast_method = finalized_cast_method(&pending, staged_method, def);
                let pending = state.engine.pending_cast.take().unwrap();
                abort_cast(state, pending, cast_method);
                return None;
            }
            if candidates.len() == 1 {
                state
                    .engine
                    .pending_cast
                    .as_mut()
                    .unwrap()
                    .sacrifice_chosen
                    .push(candidates[0]);
                return drain_pending_cast_or_decide(state);
            }
            return Some(Decision::ChooseCostTargets {
                player: pending.controller,
                source: pending.spell,
                cost_kind: CostKind::TapPermanents,
                remaining: 1,
                candidates,
            });
        }
    }

    if pending.optional_additional_cost_paid == Some(true) {
        match def
            .optional_additional_cost
            .expect("a paid optional additional cost remains definition-owned")
        {
            OptionalAdditionalCostDef::CollectEvidence { minimum_mana_value } => {
                if !pending.optional_additional_cost_selection_finished {
                    let candidates = match collect_evidence_candidates(pending.controller, state) {
                        Ok(candidates) => candidates,
                        Err(_) => {
                            state.engine.halted = Some((
                                UnsupportedMechanic::InvalidEffectContinuation,
                                pending.spell,
                            ));
                            return None;
                        }
                    };
                    let remaining = candidates
                        .iter()
                        .copied()
                        .filter(|binding| {
                            !pending.optional_additional_cost_chosen.contains(binding)
                        })
                        .collect::<Vec<_>>();
                    let selected_sum = match collect_evidence_mana_value(
                        state,
                        &pending.optional_additional_cost_chosen,
                    ) {
                        Ok(sum) => sum,
                        Err(_) => {
                            state.engine.halted = Some((
                                UnsupportedMechanic::InvalidEffectContinuation,
                                pending.spell,
                            ));
                            return None;
                        }
                    };
                    if selected_sum < u32::from(minimum_mana_value) && remaining.len() == 1 {
                        state
                            .engine
                            .pending_cast
                            .as_mut()
                            .expect("validated cast remains pending")
                            .optional_additional_cost_chosen
                            .push(remaining[0]);
                        return drain_pending_cast_or_decide(state);
                    }
                    if remaining.is_empty() && selected_sum < u32::from(minimum_mana_value) {
                        state.engine.halted = Some((
                            UnsupportedMechanic::InvalidEffectContinuation,
                            pending.spell,
                        ));
                        return None;
                    }
                    let max_targets = match u16::try_from(candidates.len()) {
                        Ok(count) => count,
                        Err(_) => {
                            state.engine.halted = Some((
                                UnsupportedMechanic::InvalidEffectContinuation,
                                pending.spell,
                            ));
                            return None;
                        }
                    };
                    return Some(Decision::ChooseEffectTargets {
                        player: pending.controller,
                        source: pending.spell,
                        selected_count: pending.optional_additional_cost_chosen.len() as u16,
                        min_targets: 1,
                        max_targets,
                        legal_targets: remaining
                            .into_iter()
                            .map(|binding| Target::Object(binding.object))
                            .collect(),
                        can_finish: selected_sum >= u32::from(minimum_mana_value),
                    });
                }
            }
            OptionalAdditionalCostDef::Bargain => {
                if pending.optional_additional_cost_chosen.is_empty() {
                    let candidates = match bargain_candidates(pending.controller, state) {
                        Ok(candidates) => candidates,
                        Err(_) => {
                            state.engine.halted = Some((
                                UnsupportedMechanic::InvalidEffectContinuation,
                                pending.spell,
                            ));
                            return None;
                        }
                    };
                    match candidates.as_slice() {
                        [] => {
                            state.engine.halted = Some((
                                UnsupportedMechanic::InvalidEffectContinuation,
                                pending.spell,
                            ));
                            return None;
                        }
                        [binding] => {
                            state
                                .engine
                                .pending_cast
                                .as_mut()
                                .expect("validated cast remains pending")
                                .optional_additional_cost_chosen
                                .push(*binding);
                            return drain_pending_cast_or_decide(state);
                        }
                        _ => {
                            return Some(Decision::ChooseCostTargets {
                                player: pending.controller,
                                source: pending.spell,
                                cost_kind: CostKind::SacrificePermanents,
                                remaining: 1,
                                candidates: candidates
                                    .into_iter()
                                    .map(|binding| binding.object)
                                    .collect(),
                            });
                        }
                    }
                }
            }
        }
    }

    if has_chosen_creature_additional_cost(def) {
        if pending.chosen_creature_cost_zone.is_none() {
            let battlefield = chosen_creature_cost_candidates(
                pending.controller,
                ChosenCreatureCostZoneV1::Battlefield,
                state,
            );
            let hand = chosen_creature_cost_candidates(
                pending.controller,
                ChosenCreatureCostZoneV1::Hand,
                state,
            );
            match (battlefield.is_empty(), hand.is_empty()) {
                (true, true) => {
                    let cast_method = finalized_cast_method(&pending, staged_method, def);
                    let pending = state.engine.pending_cast.take().unwrap();
                    abort_cast(state, pending, cast_method);
                    return None;
                }
                (false, true) => {
                    state
                        .engine
                        .pending_cast
                        .as_mut()
                        .unwrap()
                        .chosen_creature_cost_zone = Some(ChosenCreatureCostZoneV1::Battlefield);
                    return drain_pending_cast_or_decide(state);
                }
                (true, false) => {
                    state
                        .engine
                        .pending_cast
                        .as_mut()
                        .unwrap()
                        .chosen_creature_cost_zone = Some(ChosenCreatureCostZoneV1::Hand);
                    return drain_pending_cast_or_decide(state);
                }
                (false, false) => {
                    return Some(Decision::ChooseEffectOption {
                        player: pending.controller,
                        source: pending.spell,
                        option_count: 2,
                    });
                }
            }
        }
        if pending.chosen_creature_cost.is_none() {
            let zone = pending
                .chosen_creature_cost_zone
                .expect("validated chosen-creature cost retains its zone");
            let candidates = chosen_creature_cost_candidates(pending.controller, zone, state);
            if candidates.is_empty() {
                let cast_method = finalized_cast_method(&pending, staged_method, def);
                let pending = state.engine.pending_cast.take().unwrap();
                abort_cast(state, pending, cast_method);
                return None;
            }
            if candidates.len() == 1 {
                state
                    .engine
                    .pending_cast
                    .as_mut()
                    .unwrap()
                    .chosen_creature_cost = Some(candidates[0]);
                return drain_pending_cast_or_decide(state);
            }
            return Some(Decision::ChooseCostTargets {
                player: pending.controller,
                source: pending.spell,
                cost_kind: CostKind::ChooseCreatureOrRevealCreature,
                remaining: 1,
                candidates: candidates.iter().map(|binding| binding.object).collect(),
            });
        }
    }

    if pending.x_value.is_none() {
        let Some(cost) = pending_cast_selected_mana_cost(def, &pending, state) else {
            state.engine.halted = Some((
                UnsupportedMechanic::InvalidEffectContinuation,
                pending.spell,
            ));
            return None;
        };
        let Some(maximum) = maximum_payable_x(&cost, pending.controller, state) else {
            let cast_method = finalized_cast_method(&pending, staged_method, def);
            let pending = state.engine.pending_cast.take().unwrap();
            abort_cast(state, pending, cast_method);
            return None;
        };
        if maximum == 0 {
            state.engine.pending_cast.as_mut().unwrap().x_value = Some(0);
            return drain_pending_cast_or_decide(state);
        }
        return Some(Decision::ChooseEffectOption {
            player: pending.controller,
            source: pending.spell,
            option_count: u16::from(maximum) + 1,
        });
    }

    if pending.additional_cost_discarded.is_none() {
        let add = def.additional_cost.expect(
            "additional_cost_discarded is None only when begin_cast saw an additional_cost",
        );
        match discard_count_in(add) {
            Some(n) => {
                state.engine.pending_discard = Some(PendingDiscard {
                    player: pending.controller,
                    count: n as u32,
                    resume: DiscardResume::FinishCast {
                        source_contract: pending.source_contract,
                        controller: pending.controller,
                    },
                });
                return None; // let the caller's next loop pass see pending_discard
            }
            None => {
                state
                    .engine
                    .pending_cast
                    .as_mut()
                    .unwrap()
                    .additional_cost_discarded = Some(vec![]);
                return drain_pending_cast_or_decide(state);
            }
        }
    }

    if finalize_cast(state, Vec::new()).is_err() {
        state.engine.halted = Some((
            UnsupportedMechanic::InvalidEffectContinuation,
            pending.spell,
        ));
    }
    None
}

fn drain_pending_activation_or_decide(state: &mut GameState) -> Option<Decision> {
    let pending = state.engine.pending_activation.clone()?;
    if validate_pending_activation(state, &pending).is_err() {
        state.engine.halted = Some((
            UnsupportedMechanic::InvalidEffectContinuation,
            pending.source,
        ));
        return Some(Decision::Halted {
            mechanic: UnsupportedMechanic::InvalidEffectContinuation,
            source: pending.source,
        });
    }
    let def = &card_def::CARD_DEFS[state.objects.get(pending.source).card_def as usize];
    let ability = &def.activated_abilities[pending.ability_index as usize];

    let max_targets = target_count(pending.target_spec);
    if (pending.targets_chosen.len() as u8) < max_targets {
        let legal_targets = completable_next_activation_targets_for(
            pending.source,
            ability,
            &pending.targets_chosen,
            state,
        );
        if target_min_count(pending.target_spec) < max_targets {
            if !legal_targets.is_empty() {
                return Some(Decision::ChooseEffectTargets {
                    player: pending.controller,
                    source: pending.source,
                    selected_count: pending.targets_chosen.len() as u16,
                    min_targets: u16::from(target_min_count(pending.target_spec)),
                    max_targets: u16::from(max_targets),
                    legal_targets,
                    can_finish: true,
                });
            }
        } else {
            return Some(Decision::ChooseTargets {
                player: pending.controller,
                spell: pending.source,
                remaining: max_targets - pending.targets_chosen.len() as u8,
                legal_targets,
                can_finish: false,
            });
        }
    }

    if let Some(filter) = return_permanent_filter_in(ability.cost) {
        if pending.object_cost_chosen.is_empty() {
            let candidates = return_permanent_cost_candidates(
                pending.controller,
                state,
                filter,
                &pending.object_cost_chosen,
            );
            if candidates.is_empty() {
                state.engine.halted = Some((
                    UnsupportedMechanic::InvalidEffectContinuation,
                    pending.source,
                ));
                return Some(Decision::Halted {
                    mechanic: UnsupportedMechanic::InvalidEffectContinuation,
                    source: pending.source,
                });
            }
            if candidates.len() == 1 {
                state
                    .engine
                    .pending_activation
                    .as_mut()
                    .expect("validated activation remains staged")
                    .object_cost_chosen
                    .push(candidates[0]);
                return drain_pending_activation_or_decide(state);
            }
            return Some(Decision::ChooseCostTargets {
                player: pending.controller,
                source: pending.source,
                cost_kind: CostKind::ReturnPermanentsToHand,
                remaining: 1,
                candidates: candidates
                    .into_iter()
                    .map(|binding| binding.object)
                    .collect(),
            });
        }
    } else if let Some((needed, filter)) = activation_permanent_sacrifice_needed(ability.cost) {
        if pending.object_cost_chosen.len() < usize::from(needed) {
            let candidates = activation_sacrifice_cost_candidates(
                pending.controller,
                filter,
                state,
                &pending.object_cost_chosen,
            );
            let remaining = needed - pending.object_cost_chosen.len() as u8;
            if candidates.len() == usize::from(remaining) {
                state
                    .engine
                    .pending_activation
                    .as_mut()
                    .expect("validated activation remains staged")
                    .object_cost_chosen
                    .extend(candidates);
                return drain_pending_activation_or_decide(state);
            }
            return Some(Decision::ChooseCostTargets {
                player: pending.controller,
                source: pending.source,
                cost_kind: cost_kind_for_permanent_filter(filter),
                remaining,
                candidates: candidates
                    .into_iter()
                    .map(|binding| binding.object)
                    .collect(),
            });
        }
    }

    if has_unblocked_attacker_return_cost(ability.cost) && pending.object_cost_chosen.is_empty() {
        let candidates = unblocked_attacker_return_candidates(
            pending.controller,
            state,
            &pending.object_cost_chosen,
        );
        if candidates.is_empty() {
            state.engine.halted = Some((
                UnsupportedMechanic::InvalidEffectContinuation,
                pending.source,
            ));
            return Some(Decision::Halted {
                mechanic: UnsupportedMechanic::InvalidEffectContinuation,
                source: pending.source,
            });
        }
        if candidates.len() == 1 {
            state
                .engine
                .pending_activation
                .as_mut()
                .expect("validated activation remains staged")
                .object_cost_chosen
                .push(candidates[0]);
            return drain_pending_activation_or_decide(state);
        }
        return Some(Decision::ChooseCostTargets {
            player: pending.controller,
            source: pending.source,
            cost_kind: CostKind::ReturnPermanentsToHand,
            remaining: 1,
            candidates: candidates
                .into_iter()
                .map(|binding| binding.object)
                .collect(),
        });
    }

    if activation_tap_cost_subtype(ability.cost).is_some() && pending.object_cost_chosen.is_empty()
    {
        let candidates = payable_activation_cost_object_candidates(
            pending.controller,
            pending.source,
            ability.cost,
            state,
        );
        if candidates.is_empty() {
            state.engine.halted = Some((
                UnsupportedMechanic::InvalidEffectContinuation,
                pending.source,
            ));
            return Some(Decision::Halted {
                mechanic: UnsupportedMechanic::InvalidEffectContinuation,
                source: pending.source,
            });
        }
        if candidates.len() == 1 {
            let chosen = candidates[0];
            let object = state.objects.get(chosen);
            let binding = EffectObjectBinding {
                object: chosen,
                expected_zone: Zone::Battlefield,
                expected_zone_change_count: object.zone_change_count,
            };
            state
                .engine
                .pending_activation
                .as_mut()
                .expect("validated activation remains staged")
                .object_cost_chosen
                .push(binding);
            return drain_pending_activation_or_decide(state);
        }
        return Some(Decision::ChooseCostTargets {
            player: pending.controller,
            source: pending.source,
            cost_kind: CostKind::TapPermanents,
            remaining: 1,
            candidates,
        });
    }

    if pending.cost_discard_paid.is_none() {
        let n = discard_count_in(ability.cost).expect(
            "cost_discard_paid is None only when begin_activation saw a DiscardCards component",
        );
        state.engine.pending_discard = Some(PendingDiscard {
            player: pending.controller,
            count: n as u32,
            resume: DiscardResume::FinishActivation {
                source: pending.source,
                source_zone_change_count: pending.source_zone_change_count,
                controller: pending.controller,
                ability_index: pending.ability_index,
            },
        });
        return None;
    }

    finalize_activation(state);
    None
}

/// Fail-closed validation for a serialized/restored activation staging
/// record. No priority window normally exists between announcement,
/// targeting, and payment, but snapshots do; this check prevents a stale or
/// coherently malformed record from paying against a later object
/// incarnation or a different zone/ability definition.
pub(crate) fn validate_pending_activation(
    state: &GameState,
    pending: &PendingActivation,
) -> Result<(), String> {
    let object = state
        .objects
        .try_get(pending.source)
        .ok_or_else(|| format!("pending activation source {} is missing", pending.source.0))?;
    if object.zone_change_count != pending.source_zone_change_count {
        return Err("pending activation source incarnation changed".to_string());
    }
    let def = card_def::CARD_DEFS
        .get(object.card_def as usize)
        .ok_or_else(|| "pending activation source definition is missing".to_string())?;
    if !def.is_executable() {
        return Err("pending activation source is not executable".to_string());
    }
    let ability = def
        .activated_abilities
        .get(pending.ability_index as usize)
        .ok_or_else(|| "pending activation ability index changed".to_string())?;
    if ability.target_spec != pending.target_spec {
        return Err("pending activation target specification changed".to_string());
    }
    if ability.max_activations_per_turn.is_some_and(|limit| {
        activated_ability_use_count(state, pending.source, pending.ability_index as u16)
            >= u16::from(limit)
    }) {
        return Err("pending activation exceeded its per-turn limit".to_string());
    }

    match (
        pending.cost_discard_paid.as_deref(),
        discard_count_in(ability.cost),
    ) {
        (Some([]), None) | (None, Some(_)) => {}
        (None, None) => {
            return Err("noninteractive activation has an unresolved discard marker".to_string())
        }
        (Some(_), None) => {
            return Err("noninteractive activation carries discard-choice objects".to_string())
        }
        (Some(_), Some(_)) => {
            return Err("interactive activation carries a self-authenticating payment".to_string())
        }
    }

    let return_filter = return_permanent_filter_in(ability.cost);
    let tap_cost_subtype = activation_tap_cost_subtype(ability.cost);
    let sacrifice_cost = activation_permanent_sacrifice_needed(ability.cost);
    let returns_unblocked_attacker = has_unblocked_attacker_return_cost(ability.cost);
    let interactive_families = usize::from(return_filter.is_some())
        + usize::from(tap_cost_subtype.is_some())
        + usize::from(sacrifice_cost.is_some())
        + usize::from(returns_unblocked_attacker);
    if interactive_families > 1 {
        return Err("activation has multiple interactive object-cost families".to_string());
    }
    if interactive_families == 0 && !pending.object_cost_chosen.is_empty() {
        return Err("activation without an object cost carries chosen cost objects".to_string());
    }
    if return_filter.is_some() || tap_cost_subtype.is_some() || returns_unblocked_attacker {
        if pending.object_cost_chosen.len() > 1 {
            return Err("activation carries too many chosen object-cost permanents".to_string());
        }
        for &binding in &pending.object_cost_chosen {
            if binding.expected_zone != Zone::Battlefield {
                return Err(
                    "activation object-cost binding does not expect the battlefield".to_string(),
                );
            }
            let live = state
                .objects
                .try_get(binding.object)
                .ok_or_else(|| "activation object-cost permanent no longer exists".to_string())?;
            let matches_cost = return_filter.is_some_and(|filter| {
                permanent_matches_return_filter(state, binding.object, filter)
            }) || (returns_unblocked_attacker
                && unblocked_attacker_return_candidates(pending.controller, state, &[])
                    .iter()
                    .any(|candidate| candidate == &binding))
                || tap_cost_subtype.is_some_and(|subtype| {
                    binding.object != pending.source
                        && !live.tapped
                        && live
                            .v4
                            .effective_subtype_ids
                            .binary_search(&subtype.stable_id())
                            .is_ok()
                        && payable_activation_cost_object_candidates(
                            pending.controller,
                            pending.source,
                            ability.cost,
                            state,
                        )
                        .contains(&binding.object)
                });
            if live.zone_change_count != binding.expected_zone_change_count
                || live.zone != Zone::Battlefield
                || live.controller != pending.controller
                || !matches_cost
            {
                return Err(
                    "activation object-cost permanent changed incarnation or legality".to_string(),
                );
            }
        }
    } else if let Some((needed, filter)) = sacrifice_cost {
        if pending.object_cost_chosen.len() > usize::from(needed) {
            return Err("pending activation has too many object-cost selections".to_string());
        }
        let duplicate = pending
            .object_cost_chosen
            .iter()
            .enumerate()
            .any(|(index, binding)| {
                pending.object_cost_chosen[..index]
                    .iter()
                    .any(|chosen| chosen.object == binding.object)
            });
        if duplicate {
            return Err("pending activation repeats an object-cost selection".to_string());
        }
        if pending.object_cost_chosen.iter().any(|binding| {
            binding.expected_zone != Zone::Battlefield
                || state.objects.try_get(binding.object).is_none_or(|live| {
                    live.zone_change_count != binding.expected_zone_change_count
                        || live.controller != pending.controller
                        || live.zone != Zone::Battlefield
                        || !state.players[pending.controller.index()]
                            .battlefield
                            .contains(&binding.object)
                        || !permanent_matches_filter(
                            &card_def::CARD_DEFS[live.card_def as usize],
                            filter,
                        )
                })
        }) {
            return Err("pending activation carries an illegal object-cost selection".to_string());
        }
        let remaining = activation_sacrifice_cost_candidates(
            pending.controller,
            filter,
            state,
            &pending.object_cost_chosen,
        );
        if remaining.len() < usize::from(needed).saturating_sub(pending.object_cost_chosen.len()) {
            return Err("pending activation object cost can no longer complete".to_string());
        }
    }

    let in_activation_zone = match ability.activation_zone {
        Zone::Battlefield => {
            object.controller == pending.controller
                && object.zone == Zone::Battlefield
                && state.players[pending.controller.index()]
                    .battlefield
                    .contains(&pending.source)
        }
        Zone::Hand => {
            object.owner == pending.controller
                && object.zone == Zone::Hand
                && state.players[pending.controller.index()]
                    .hand
                    .contains(&pending.source)
        }
        Zone::Graveyard => {
            object.owner == pending.controller
                && object.zone == Zone::Graveyard
                && state.players[pending.controller.index()]
                    .graveyard
                    .contains(&pending.source)
        }
        _ => false,
    };
    if !in_activation_zone {
        return Err("pending activation source left its activation zone".to_string());
    }
    if !can_pay_activation_components(ability.cost, pending.controller, pending.source, state) {
        return Err("pending activation cost is no longer payable".to_string());
    }

    let max_targets = usize::from(target_count(pending.target_spec));
    if pending.targets_chosen.len() > max_targets {
        return Err("pending activation has too many targets".to_string());
    }
    if !pending.object_cost_chosen.is_empty()
        && !target_cardinality_is_complete(pending.target_spec, pending.targets_chosen.len())
    {
        return Err("activation object cost was selected before targeting completed".to_string());
    }
    if !target_contracts_are_structurally_valid(
        state,
        &pending.targets_chosen,
        &pending.target_contracts,
    ) {
        return Err("pending activation target contract changed".to_string());
    }
    let mut prefix = Vec::new();
    for &target in &pending.targets_chosen {
        if !completable_next_activation_targets_for(pending.source, ability, &prefix, state)
            .contains(&target)
        {
            return Err("pending activation carries an illegal target prefix".to_string());
        }
        prefix.push(target);
    }
    if !activation_target_prefix_can_complete(
        pending.source,
        ability,
        &pending.targets_chosen,
        state,
    ) {
        return Err("pending activation target prefix cannot complete".to_string());
    }
    if state.engine.pending_cast.is_some()
        || state.engine.pending_effect.is_some()
        || state.engine.pending_optional_cost.is_some()
        || state.engine.pending_optional_cost_sacrifice.is_some()
        || state.engine.pending_spell_copy.is_some()
        || !state.engine.pending_triggers.is_empty()
        || state.engine.pending_kicked_source.is_some()
    {
        return Err("pending activation coexists with another staged engine context".to_string());
    }
    if let Some(discard) = state.engine.pending_discard.as_ref() {
        let DiscardResume::FinishActivation {
            source,
            source_zone_change_count,
            controller,
            ability_index,
        } = discard.resume
        else {
            return Err("pending activation coexists with a non-activation discard".to_string());
        };
        if source != pending.source
            || source_zone_change_count != pending.source_zone_change_count
            || controller != pending.controller
            || ability_index != pending.ability_index
            || discard.player != pending.controller
            || discard_count_in(ability.cost).map(u32::from) != Some(discard.count)
            || pending.cost_discard_paid.is_some()
            || activation_permanent_sacrifice_needed(ability.cost)
                .is_some_and(|(needed, _)| pending.object_cost_chosen.len() != usize::from(needed))
            || !target_cardinality_is_complete(pending.target_spec, pending.targets_chosen.len())
            || ((return_filter.is_some()
                || tap_cost_subtype.is_some()
                || returns_unblocked_attacker)
                && pending.object_cost_chosen.len() != 1)
        {
            return Err("pending activation discard binding or stage changed".to_string());
        }
    }
    Ok(())
}

/// If there are pending triggers, either auto-places a singleton group (no
/// real choice to make) or returns `Decision::OrderTriggers` for a group of
/// 2+ sharing a controller.
fn drain_pending_triggers_or_decide(state: &mut GameState) -> Option<Decision> {
    loop {
        if state.engine.pending_triggers.is_empty() {
            // Targeting markers created while putting the just-drained batch
            // on the stack can create Ward triggers. They form a fresh
            // trigger-placement checkpoint only after the prior APNAP batch
            // is fully placed, so collect them here rather than interleaving
            // them with an older same-controller group.
            if state.engine.event_log.is_empty() {
                return None;
            }
            collect_and_queue_triggers(state);
            if state.engine.pending_triggers.is_empty() {
                return None;
            }
        }
        let first = state.engine.pending_triggers[0].clone();
        let controller = first.controller;
        let group_len = state
            .engine
            .pending_triggers
            .iter()
            .take_while(|trigger| trigger.controller == controller)
            .count();
        let group_is_ordered = state.engine.pending_triggers[..group_len]
            .iter()
            .all(|trigger| trigger.placement_ordered);

        if group_len >= 2 && !group_is_ordered {
            return Some(Decision::OrderTriggers {
                player: controller,
                pending: state.engine.pending_triggers[..group_len].to_vec(),
            });
        }
        if group_len == 1 && !first.placement_ordered {
            state.engine.pending_triggers[0].placement_ordered = true;
        }

        let pending = state.engine.pending_triggers[0].clone();
        if validate_pending_trigger(state, &pending).is_err() {
            state.engine.halted = Some((
                UnsupportedMechanic::InvalidEffectContinuation,
                pending.source,
            ));
            return Some(Decision::Halted {
                mechanic: UnsupportedMechanic::InvalidEffectContinuation,
                source: pending.source,
            });
        }
        let need = target_count(pending.target_spec);
        if pending.targets.len() < usize::from(need) {
            let legal_targets = completable_next_targets_for_controller_and_source(
                pending.target_spec,
                &pending.targets,
                pending.controller,
                pending.source_contract.map(|contract| TargetingSource {
                    object: pending.source,
                    card_def: contract.card_def,
                }),
                state,
            );
            let can_finish =
                pending.targets.len() >= usize::from(target_min_count(pending.target_spec));
            if legal_targets.is_empty() && !can_finish {
                // 603.3d: a trigger that cannot get its required targets is
                // removed instead of being put on the stack (e.g. Journey to
                // Nowhere when every creature has protection from it).
                state.engine.pending_triggers.remove(0);
                continue;
            }
            return Some(Decision::ChooseTargets {
                player: pending.controller,
                spell: pending.source,
                remaining: need - pending.targets.len() as u8,
                legal_targets,
                can_finish,
            });
        }

        let pending = state.engine.pending_triggers.remove(0);
        if push_trigger_onto_stack(state, pending.clone()).is_err() {
            state.engine.halted = Some((
                UnsupportedMechanic::InvalidEffectContinuation,
                pending.source,
            ));
            return Some(Decision::Halted {
                mechanic: UnsupportedMechanic::InvalidEffectContinuation,
                source: pending.source,
            });
        }
    }
}

fn validate_pending_trigger(state: &GameState, pending: &PendingTrigger) -> Result<(), String> {
    if !pending.placement_ordered {
        return Err("pending trigger has not completed placement ordering".to_string());
    }
    validate_pending_trigger_identity(state, pending)?;
    if pending.is_madness_offer {
        if pending.target_spec != TargetSpec::None
            || !pending.targets.is_empty()
            || !pending.target_contracts.is_empty()
        {
            return Err("Madness offer carries triggered-ability targets".to_string());
        }
        return Ok(());
    }
    if pending.target_spec == TargetSpec::None {
        if !pending.targets.is_empty() || !pending.target_contracts.is_empty() {
            return Err("untargeted pending trigger carries targets".to_string());
        }
        return Ok(());
    }
    let source = state
        .objects
        .try_get(pending.source)
        .ok_or("pending targeted trigger source is missing")?;
    let expected = trigger::target_spec_for_trigger(source.card_def, &pending.effect)
        .ok_or("pending targeted trigger has no matching definition-owned ability")?;
    if pending.target_spec != expected {
        return Err("pending trigger target specification changed".to_string());
    }
    if pending.targets.len() > usize::from(target_count(expected))
        || !target_contracts_are_structurally_valid(
            state,
            &pending.targets,
            &pending.target_contracts,
        )
    {
        return Err("pending trigger target prefix or contracts are malformed".to_string());
    }
    let mut prefix = Vec::new();
    for &target in &pending.targets {
        if !completable_next_targets_for_controller(expected, &prefix, pending.controller, state)
            .contains(&target)
        {
            return Err("pending trigger carries an illegal target prefix".to_string());
        }
        prefix.push(target);
    }
    Ok(())
}

/// Emits one committed targeting marker per distinct permanent incarnation in
/// a completed targeting action. The private stack id must resolve uniquely;
/// source ids and mutable stack positions are never accepted as substitutes.
fn log_final_targeting_events(
    state: &mut GameState,
    stack_item_id: StackItemId,
) -> Result<(), String> {
    if stack_item_id == StackItemId::default() {
        return Err("a finalized targeting action lacks a stack-incarnation id".to_string());
    }
    let matches = state
        .stack
        .iter()
        .filter(|item| item.v4.stack_item_id == stack_item_id)
        .cloned()
        .collect::<Vec<_>>();
    let [item] = matches.as_slice() else {
        return Err(format!(
            "expected one stack item for {stack_item_id}, found {}",
            matches.len()
        ));
    };
    validated_stack_item_target_spec(item, state)?;
    if item.targets.len() != item.v4.target_contracts.len() {
        return Err("final targeting vectors no longer have parallel contracts".to_string());
    }

    let mut seen = Vec::new();
    for contract in &item.v4.target_contracts {
        let StackTargetContractV4::Object {
            object,
            zone: Zone::Battlefield,
            zone_change_count,
            ..
        } = *contract
        else {
            continue;
        };
        if seen.contains(&(object, zone_change_count)) {
            continue;
        }
        seen.push((object, zone_change_count));
        event::log_targeted(
            state,
            object,
            zone_change_count,
            stack_item_id,
            item.controller,
        );
    }
    Ok(())
}

/// Returns the exact producing-spell contract for a trigger whose ability
/// functions from the stack. Ordinary ability and Madness triggers use the
/// single historical `PendingTrigger::source_contract` field instead.
fn pending_trigger_spell_source_contract(
    state: &GameState,
    pending: &PendingTrigger,
) -> Result<Option<StackSourceContractV4>, String> {
    if pending.is_madness_offer || pending.source_contract.is_some() {
        return Ok(None);
    }
    let source = state
        .objects
        .try_get(pending.source)
        .ok_or("pending spell-sourced trigger object is missing")?;
    if source.zone != Zone::Stack {
        return Err("pending trigger has no exact source provenance".to_string());
    }
    let mut producers = state.stack.iter().filter(|candidate| {
        candidate.kind == StackItemKind::Spell && candidate.source == pending.source
    });
    let producer = producers
        .next()
        .ok_or("pending spell-sourced trigger has no live producing spell")?;
    if producers.next().is_some() {
        return Err("pending spell-sourced trigger has duplicate producers".to_string());
    }
    validate_spell_stack_source(state, producer)?;
    producer
        .v4
        .source_contract
        .map(Some)
        .ok_or_else(|| "pending spell-sourced trigger producer lost its contract".to_string())
}

fn validate_pending_trigger_for_stack(
    state: &GameState,
    pending: &PendingTrigger,
) -> Result<(), String> {
    validate_pending_trigger_identity(state, pending)?;
    if !target_cardinality_is_complete(pending.target_spec, pending.targets.len()) {
        return Err("pending trigger target metadata is incomplete".to_string());
    }
    if pending.is_madness_offer {
        return Ok(());
    }
    let prospective_stack_item_id = StackItemId(
        state
            .engine
            .next_stack_item_id
            .checked_add(1)
            .ok_or("stack-item identity space exhausted")?,
    );
    let spell_source_contract = pending_trigger_spell_source_contract(state, pending)?;
    let candidate = StackItem {
        kind: StackItemKind::TriggeredAbility,
        source: pending.source,
        controller: pending.controller,
        targets: pending.targets.clone(),
        is_copy: false,
        inline_effect: Some(pending.effect.clone()),
        discarded: Vec::new(),
        is_flashback: false,
        mode_chosen: 0,
        madness_offer: false,
        kicked: pending.kicked,
        v4: StackStateV4 {
            stack_item_id: prospective_stack_item_id,
            source_contract: spell_source_contract,
            paid_cost_refs: pending.paid_cost_refs.clone(),
            optional_additional_cost_paid: pending.optional_additional_cost_paid,
            target_spec: (pending.target_spec != TargetSpec::None).then_some(pending.target_spec),
            target_contracts: pending.target_contracts.clone(),
            ability_source_contract: pending.source_contract,
            granted_by: pending.granted_by,
            ..StackStateV4::default()
        },
    };
    validated_stack_item_target_spec(&candidate, state)?;
    Ok(())
}

fn validate_pending_trigger_identity(
    state: &GameState,
    pending: &PendingTrigger,
) -> Result<(), String> {
    if pending.targets.len() != pending.target_contracts.len() {
        return Err("pending trigger targeting vectors are not parallel".to_string());
    }
    let source = state
        .objects
        .try_get(pending.source)
        .ok_or("pending trigger source object is missing")?;
    if pending.is_madness_offer {
        let source_contract = pending
            .source_contract
            .ok_or("pending Madness trigger lost its exact source incarnation")?;
        let definition = card_def::CARD_DEFS
            .get(source_contract.card_def as usize)
            .ok_or("pending Madness trigger source definition is missing")?;
        if pending.target_spec != TargetSpec::None
            || !pending.targets.is_empty()
            || pending.optional_additional_cost_paid.is_some()
            || !pending.paid_cost_refs.is_empty()
            || pending.effect != EffectOp::Sequence(Vec::new())
            || pending.granted_by.is_some()
            || source_contract.source != pending.source
            || source_contract.card_def != source.card_def
            || source_contract.owner != source.owner
            || source_contract.controller != pending.controller
            || source.zone != Zone::Exile
            || source.zone_change_count != source_contract.zone_change_count
            || source.owner != pending.controller
            || definition.madness_cost.is_none()
        {
            return Err("pending Madness trigger provenance is malformed".to_string());
        }
        return Ok(());
    }
    let spell_source_contract = pending_trigger_spell_source_contract(state, pending)?;
    if pending.source_contract.is_some() == spell_source_contract.is_some() {
        return Err("pending trigger source provenance is missing or parallel".to_string());
    }
    let candidate = StackItem {
        kind: StackItemKind::TriggeredAbility,
        source: pending.source,
        controller: pending.controller,
        targets: pending.targets.clone(),
        is_copy: false,
        inline_effect: Some(pending.effect.clone()),
        discarded: Vec::new(),
        is_flashback: false,
        mode_chosen: 0,
        madness_offer: false,
        kicked: pending.kicked,
        v4: StackStateV4 {
            source_contract: spell_source_contract,
            paid_cost_refs: pending.paid_cost_refs.clone(),
            optional_additional_cost_paid: pending.optional_additional_cost_paid,
            target_spec: (pending.target_spec != TargetSpec::None).then_some(pending.target_spec),
            target_contracts: pending.target_contracts.clone(),
            ability_source_contract: pending.source_contract,
            granted_by: pending.granted_by,
            ..StackStateV4::default()
        },
    };
    let expected = triggered_stack_item_expected_target_spec(&candidate, state)?;
    if expected != candidate.v4.target_spec {
        return Err("pending trigger target specification is not definition-owned".to_string());
    }
    Ok(())
}

fn push_trigger_onto_stack(state: &mut GameState, t: PendingTrigger) -> Result<(), String> {
    validate_pending_trigger_for_stack(state, &t)?;
    let madness_source_contract = t
        .is_madness_offer
        .then(|| MadnessOfferSourceContractV4::capture(state, t.source));
    let spell_source_contract = pending_trigger_spell_source_contract(state, &t)?;
    let stack_item_id = next_stack_item_id(state);
    state.stack.push(StackItem {
        kind: if t.is_madness_offer {
            StackItemKind::MadnessOffer
        } else {
            StackItemKind::TriggeredAbility
        },
        source: t.source,
        controller: t.controller,
        targets: t.targets,
        is_copy: false,
        // A Madness offer (`is_madness_offer`, set by `apply_discard`) has
        // no fixed `EffectOp` program to run at resolution -- resolving it
        // means asking `Decision::ChooseMadnessCast`
        // (`advance_until_decision`'s `top.madness_offer` check), not
        // executing an effect, so it carries no `inline_effect` at all.
        inline_effect: if t.is_madness_offer {
            None
        } else {
            Some(t.effect)
        },
        discarded: vec![],
        is_flashback: false,
        mode_chosen: 0,
        madness_offer: t.is_madness_offer,
        // Propagates a kicked cast's own flag onward into this trigger's
        // own resolution context -- see `StackItem::kicked`'s doc for the
        // full chain.
        kicked: t.kicked,
        v4: StackStateV4 {
            stack_item_id,
            source_contract: spell_source_contract,
            madness_source_contract,
            paid_cost_refs: t.paid_cost_refs,
            optional_additional_cost_paid: t.optional_additional_cost_paid,
            target_spec: (t.target_spec != TargetSpec::None).then_some(t.target_spec),
            target_contracts: t.target_contracts,
            ability_source_contract: (!t.is_madness_offer && spell_source_contract.is_none()).then(
                || {
                    t.source_contract
                        .expect("validated nonspell trigger source contract")
                },
            ),
            granted_by: (!t.is_madness_offer).then_some(t.granted_by).flatten(),
            ..StackStateV4::default()
        },
    });
    if t.target_spec != TargetSpec::None {
        if let Err(error) = log_final_targeting_events(state, stack_item_id) {
            state.stack.pop();
            state.engine.next_stack_item_id -= 1;
            return Err(error);
        }
    }
    // Same `priority_passes`/`priority_player` reset as `reset_priority`
    // (117.5: priority passes to the active player once a triggered
    // ability is put on the stack), but deliberately inlined instead of
    // calling that shared helper: a trigger firing mid-cascade off another
    // action (e.g. Guttersnipe off a cast) is not a rules-level "everyone's
    // pass streak clears" boundary in the same sense `advance_step`/a
    // resolution/declare-attackers-or-blockers are, so it must not bump
    // `EngineState::priority_round` -- see that field's doc.
    state.engine.priority_passes = [false, false];
    state.priority_player = state.active_player;
    Ok(())
}

/// 704.5g/h creature death, 704.5a life-loss, 704.5c empty-draw-loss all
/// happen inside `resolve_top_of_stack`'s `collect_and_queue_triggers`
/// call; this just pops and executes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResolutionProgress {
    Complete,
    Suspended,
}

/// Validates the one nonspell stack shape that intentionally has no inline
/// effect program. A Madness offer is bound to the exact exiled incarnation
/// that produced it; neither its redundant flag nor its kind is authority to
/// move an arbitrary permanent/card when the choice is answered.
fn validate_madness_offer_stack_item(state: &GameState, item: &StackItem) -> Result<(), String> {
    if item.kind != StackItemKind::MadnessOffer
        || !item.madness_offer
        || item.inline_effect.is_some()
        || item.is_copy
        || item.is_flashback
        || !item.targets.is_empty()
        || !item.discarded.is_empty()
        || item.mode_chosen != 0
        || item.kicked
        || item.v4.cast_method.is_some()
        || item.v4.source_contract.is_some()
        || item.v4.face_index != 0
        || item.v4.x_value != 0
        || !item.v4.paid_cost_refs.is_empty()
        || item.v4.optional_additional_cost_paid.is_some()
        || item.v4.target_spec.is_some()
        || !item.v4.target_contracts.is_empty()
        || item.v4.activated_ability_index.is_some()
        || item.v4.hidden_ability_source.is_some()
        || item.v4.ability_source_contract.is_some()
        || item.v4.granted_by.is_some()
    {
        return Err("Madness offer carries incompatible producer metadata".to_string());
    }
    let contract = item
        .v4
        .madness_source_contract
        .ok_or("Madness offer lost its exact exiled-source contract")?;
    let source = state
        .objects
        .try_get(item.source)
        .ok_or("Madness offer source object is missing")?;
    if contract.source != item.source
        || contract.card_def != source.card_def
        || contract.owner != source.owner
        || contract.controller != source.controller
        || contract.controller != item.controller
        || contract.owner != item.controller
        || contract.zone != Zone::Exile
        || source.zone != Zone::Exile
        || contract.zone_change_count != source.zone_change_count
        || source.spell_copy_origin.is_some()
    {
        return Err(
            "Madness offer source contract no longer matches its exact incarnation".to_string(),
        );
    }
    let exile_memberships = state
        .exile
        .iter()
        .filter(|&&candidate| candidate == item.source)
        .count();
    let appears_elsewhere = [PlayerId::P0, PlayerId::P1].into_iter().any(|player| {
        let zones = &state.players[player.index()];
        zones.library.contains(&item.source)
            || zones.hand.contains(&item.source)
            || zones.battlefield.contains(&item.source)
            || zones.graveyard.contains(&item.source)
    }) || state.command.contains(&item.source);
    if exile_memberships != 1 || appears_elsewhere {
        return Err("Madness offer source is not indexed exclusively in exile".to_string());
    }
    let def = card_def::CARD_DEFS
        .get(source.card_def as usize)
        .ok_or("Madness offer source definition is missing")?;
    if !def.is_executable() || !def.is_castable() || def.madness_cost.is_none() {
        return Err("Madness offer source definition has no Madness cast".to_string());
    }
    if item.v4.stack_item_id == StackItemId::default() {
        return Err("Madness offer has no stack-incarnation identity".to_string());
    }
    let mut memberships = state
        .stack
        .iter()
        .filter(|candidate| candidate.v4.stack_item_id == item.v4.stack_item_id);
    let live = memberships
        .next()
        .ok_or("Madness offer has no live stack membership")?;
    if memberships.next().is_some() || live != item {
        return Err("Madness offer does not own unique exact stack membership".to_string());
    }
    Ok(())
}

fn triggered_stack_item_expected_target_spec(
    item: &StackItem,
    state: &GameState,
) -> Result<Option<TargetSpec>, String> {
    if item.kind != StackItemKind::TriggeredAbility || item.v4.activated_ability_index.is_some() {
        return Err("triggered stack item carries incompatible producer metadata".to_string());
    }
    let inline_effect = item
        .inline_effect
        .as_ref()
        .ok_or("triggered stack item carries incompatible producer metadata")?;
    if item.v4.madness_source_contract.is_some() || item.v4.hidden_ability_source.is_some() {
        return Err("triggered stack item carries incompatible producer metadata".to_string());
    }
    let source = state
        .objects
        .try_get(item.source)
        .ok_or("triggered stack item source object is missing")?;
    let (source_card_def, ability_source_contract) =
        match (item.v4.source_contract, item.v4.ability_source_contract) {
            (Some(spell_source_contract), None) => {
                validate_spell_sourced_trigger(state, item, spell_source_contract)?;
                (spell_source_contract.card_def, None)
            }
            (None, Some(ability_source_contract)) => {
                validate_ability_source_contract(state, item, ability_source_contract)?;
                (
                    ability_source_contract.card_def,
                    Some(ability_source_contract),
                )
            }
            _ => {
                return Err(
                    "triggered stack item source provenance is missing or parallel".to_string(),
                )
            }
        };
    if matches!(
        inline_effect,
        EffectOp::TapAttachedCreatureAndDamageControllerByPower
    ) && ability_source_contract.is_none_or(|contract| contract.attached_to.is_none())
    {
        return Err("attached-source trigger lost its host LKI".to_string());
    }
    if let EffectOp::PutPlusOnePlusOneCounterOnBoundObject { object } = inline_effect {
        let Some(source_contract) = ability_source_contract else {
            return Err("bound-source trigger lost its historical source contract".to_string());
        };
        if object.object != item.source
            || object.expected_zone != source_contract.zone
            || object.expected_zone_change_count != source_contract.zone_change_count
        {
            return Err("bound-source trigger changed its exact source binding".to_string());
        }
    }
    if let EffectOp::ResolveInitiativeTrigger { binding } = inline_effect {
        let Some(source_contract) = ability_source_contract else {
            return Err("Initiative trigger lost its historical designation source".to_string());
        };
        if source_contract != binding.source
            || item.controller != binding.player
            || item.source != binding.source.source
        {
            return Err("Initiative trigger changed its source or controller".to_string());
        }
        effect::validate_initiative_trigger_binding(state, *binding)?;
    }
    let source_def = card_def::CARD_DEFS
        .get(source_card_def as usize)
        .ok_or("triggered stack item source definition is missing")?;
    if !source_def.is_executable() {
        return Err("triggered stack item source definition is not executable".to_string());
    }
    let equipment_granted_trigger = match (ability_source_contract, item.v4.granted_by) {
        (Some(host), Some(equipment)) => {
            validate_equipment_granted_trigger_contract(
                state,
                host,
                equipment,
                item.controller,
                inline_effect,
            )?;
            true
        }
        (None, Some(_)) => {
            return Err("attachment-granted trigger lost its host contract".to_string())
        }
        (_, None) => false,
    };
    let definition_target_spec = trigger::target_spec_for_trigger(source_card_def, inline_effect);
    let expected_optional_cost =
        trigger::required_optional_additional_cost_for_trigger(source_card_def, inline_effect);
    if item.v4.optional_additional_cost_paid != expected_optional_cost {
        return Err("trigger optional additional-cost provenance changed".to_string());
    }
    if let Some(kind) = expected_optional_cost {
        validate_optional_additional_paid_refs(kind, item.controller, &item.v4.paid_cost_refs)?;
    }
    let definition_trigger = item.v4.granted_by.is_none() && definition_target_spec.is_some();
    let ward_trigger = if item.v4.granted_by.is_none() {
        if let Some(source_contract) = ability_source_contract {
            match inline_effect {
                EffectOp::CounterUnlessPaysGeneric {
                    ward_target,
                    targeting_stack_item,
                    generic,
                } => {
                    let source_contract_is_consistent = matches!(
                        *ward_target,
                        StackTargetContractV4::Object {
                            object,
                            card_def,
                            owner,
                            controller,
                            zone: Zone::Battlefield,
                            zone_change_count,
                            spell_copy_origin: None,
                        } if object == item.source
                            && card_def == source_contract.card_def
                            && owner == source_contract.owner
                            && controller == item.controller
                            && source.zone_change_count >= zone_change_count
                            && (source.zone_change_count != zone_change_count
                                || source.zone == Zone::Battlefield)
                    );
                    source_def.ward_cost == Some(crate::card_def::WardCostDef::Generic(*generic))
                        && *targeting_stack_item != StackItemId::default()
                        && source_contract_is_consistent
                }
                _ => false,
            }
        } else {
            false
        }
    } else {
        false
    };
    if !definition_trigger && !ward_trigger && !equipment_granted_trigger {
        return Err("triggered stack item effect is not definition-owned".to_string());
    }
    let spec = definition_target_spec.unwrap_or(TargetSpec::None);
    Ok((spec != TargetSpec::None).then_some(spec))
}

/// Checks every target of a spell again at resolution time (608.2b). The
/// target specs are sequential, so dependent specs such as
/// `PlayerThenTheirCreature` validate each target against the already-valid
/// prefix exactly as casting did.
pub(crate) fn validated_stack_item_target_spec(
    item: &StackItem,
    state: &GameState,
) -> Result<Option<TargetSpec>, String> {
    validate_spell_stack_source(state, item)?;
    if item.madness_offer != (item.kind == StackItemKind::MadnessOffer) {
        return Err("Madness-offer kind and flag disagree".to_string());
    }
    if state.engine.pending_cast.as_ref().is_some_and(|pending| {
        item.kind == StackItemKind::Spell
            && item.v4.source_contract == Some(pending.source_contract)
    }) {
        if item.kind != StackItemKind::Spell
            || item.inline_effect.is_some()
            || item.madness_offer
            || item.v4.cast_method.is_none()
            || item.v4.target_spec.is_some()
            || !item.targets.is_empty()
            || !item.v4.target_contracts.is_empty()
            || item.v4.activated_ability_index.is_some()
        {
            return Err("pending cast placeholder carries finalized target metadata".to_string());
        }
        return Ok(None);
    }
    let expected_spec = match item.kind {
        StackItemKind::Spell => {
            if item.inline_effect.is_some()
                || item.madness_offer
                || item.v4.cast_method.is_none()
                || item.v4.activated_ability_index.is_some()
            {
                return Err("spell stack item carries incompatible producer metadata".to_string());
            }
            let def = &card_def::CARD_DEFS[state.objects.get(item.source).card_def as usize];
            if item.v4.cast_method == Some(CastMethodV4::Omen) {
                if item.mode_chosen != 0 {
                    return Err("Omen spell stack item carries a modal index".to_string());
                }
                Some(
                    supported_omen(def)
                        .ok_or("Omen spell stack item lost its definition")?
                        .target_spec,
                )
            } else if item.v4.cast_method == Some(CastMethodV4::Bestow) {
                if item.mode_chosen != 0 {
                    return Err("Bestow spell stack item carries a modal index".to_string());
                }
                Some(
                    supported_bestow(def)
                        .ok_or("Bestow spell stack item lost its definition")?
                        .target_spec,
                )
            } else {
                Some(match item.mode_chosen {
                    0 => def.target_spec,
                    1 => {
                        def.mode2
                            .as_ref()
                            .ok_or("spell stack item selected a nonexistent second mode")?
                            .target_spec
                    }
                    2 => {
                        def.mode3
                            .as_ref()
                            .ok_or("spell stack item selected a nonexistent third mode")?
                            .target_spec
                    }
                    _ => return Err("spell stack item carries an unknown mode index".to_string()),
                })
            }
        }
        StackItemKind::ActivatedAbility => {
            if item.inline_effect.is_none()
                || item.v4.madness_source_contract.is_some()
                || item.v4.granted_by.is_some()
                || item.v4.optional_additional_cost_paid.is_some()
            {
                return Err(
                    "activated stack item carries incompatible producer metadata".to_string(),
                );
            }
            let ability_index = item
                .v4
                .activated_ability_index
                .ok_or("activated stack item lost its definition-owned ability index")?;
            let source_contract = validated_ability_source_contract(state, item)?;
            let def = card_def::CARD_DEFS
                .get(source_contract.card_def as usize)
                .ok_or("activated stack item source definition is missing")?;
            let ability = def
                .activated_abilities
                .get(ability_index as usize)
                .ok_or("activated stack item carries an out-of-range ability index")?;
            if item.inline_effect.as_ref() != Some(&(ability.effect)()) {
                return Err(
                    "activated stack item effect no longer matches its ability index".to_string(),
                );
            }
            if has_unblocked_attacker_return_cost(ability.cost) {
                let hidden_source = item
                    .v4
                    .hidden_ability_source
                    .ok_or("ninjutsu stack item lost its exact hand source")?;
                if hidden_source.object != item.source
                    || hidden_source.zone_change_count != source_contract.zone_change_count
                    || source_contract.zone != Zone::Hand
                    || source_contract.owner != item.controller
                {
                    return Err("ninjutsu stack source provenance changed".to_string());
                }
            } else if item.v4.hidden_ability_source.is_some() {
                return Err("non-ninjutsu activation carries hidden source provenance".to_string());
            }
            let expected_x = activation_x_value(ability, &item.targets, state)?;
            if item.v4.x_value != u16::from(expected_x) {
                return Err("activated stack item X value changed".to_string());
            }
            if source_contract.zone != ability.activation_zone {
                return Err("activated stack item source contract has the wrong zone".to_string());
            }
            Some(ability.target_spec)
        }
        StackItemKind::TriggeredAbility => triggered_stack_item_expected_target_spec(item, state)?,
        StackItemKind::MadnessOffer => {
            if item.v4.hidden_ability_source.is_some() {
                return Err("Madness offer carries activated-ability source provenance".to_string());
            }
            validate_madness_offer_stack_item(state, item)?;
            None
        }
    };
    let Some(spec) = expected_spec else {
        if item.v4.target_spec.is_some()
            || !item.targets.is_empty()
            || !item.v4.target_contracts.is_empty()
        {
            return Err("untargeted stack item carries target metadata".to_string());
        }
        return Ok(None);
    };
    if item.v4.target_spec != Some(spec) {
        return Err("stack target specification does not match its definition".to_string());
    }
    if !target_cardinality_is_complete(spec, item.targets.len())
        || item.v4.target_contracts.len() != item.targets.len()
    {
        return Err("stack target and contract cardinalities are malformed".to_string());
    }
    for (target_index, (&target, &contract)) in item
        .targets
        .iter()
        .zip(&item.v4.target_contracts)
        .enumerate()
    {
        if !stack_target_contract_is_structurally_valid(state, spec, target_index, target, contract)
        {
            return Err("stack target contract is structurally malformed".to_string());
        }
        if let StackTargetContractV4::Object {
            object,
            zone: Zone::Stack,
            zone_change_count,
            spell_copy_origin,
            ..
        } = contract
        {
            let live = state.objects.get(object);
            // A physical card that has left and later returned is an ordinary
            // stale target under CR 400.7. Do not inspect whatever unrelated
            // spell/ability its newer incarnation may currently source.
            // Structural validation above has already rejected a future
            // generation and requires virtual copies to stay exact-generation.
            if live.zone_change_count != zone_change_count {
                continue;
            }
            let mut matches = state.stack.iter().filter(|candidate| {
                candidate.kind == StackItemKind::Spell
                    && candidate.source == object
                    && candidate.v4.source_contract.is_some_and(|source_contract| {
                        source_contract.source == object
                            && source_contract.card_def == live.card_def
                            && source_contract.owner == live.owner
                            && source_contract.zone_change_count == zone_change_count
                            && source_contract.spell_copy_origin == spell_copy_origin
                    })
            });
            if let Some(target_item) = matches.next() {
                if matches.next().is_some() {
                    return Err("stack target contract names duplicate stack items".to_string());
                }
                validate_spell_stack_source(state, target_item).map_err(|error| {
                    format!("targeted spell has invalid source provenance: {error}")
                })?;
            } else {
                // A later incarnation is an ordinary stale target. A ceased
                // virtual copy deliberately retains its final Stack marker
                // and generation while losing membership, and is stale too.
                // Only a same-generation physical spell disappearing from
                // the stack without a zone transition is malformed.
                if live.zone_change_count == zone_change_count {
                    let Some(origin) = spell_copy_origin else {
                        return Err(
                            "physical stack target lost same-incarnation membership".to_string()
                        );
                    };
                    if live.spell_copy_origin != Some(origin) {
                        return Err("ceased stack-copy target origin changed".to_string());
                    }
                    validate_spell_copy_origin_chain(state, object, live.card_def, origin)?;
                }
            }
        }
    }
    Ok(Some(spec))
}

fn stack_targets_still_legal(item: &StackItem, state: &GameState) -> Result<bool, String> {
    let Some(spec) = validated_stack_item_target_spec(item, state)? else {
        return Ok(true);
    };
    if item.targets.is_empty() {
        return Ok(true);
    }
    let mut chosen = Vec::with_capacity(item.targets.len());
    let mut any_legal = false;
    for (&target, &contract) in item.targets.iter().zip(&item.v4.target_contracts) {
        let same_incarnation = target_contract_matches_live(state, target, contract);
        let legal_now = match item.kind {
            StackItemKind::ActivatedAbility => {
                let ability_index = item
                    .v4
                    .activated_ability_index
                    .ok_or("activated stack item lost its definition-owned ability index")?;
                let source_contract = validated_ability_source_contract(state, item)?;
                let def = card_def::CARD_DEFS
                    .get(source_contract.card_def as usize)
                    .ok_or("activated stack item source definition is missing")?;
                let ability = def
                    .activated_abilities
                    .get(ability_index as usize)
                    .ok_or("activated stack item carries an out-of-range ability index")?;
                activation_legal_targets_for(item.source, ability, &chosen, state).contains(&target)
            }
            _ => legal_targets_for_controller_from_source(
                spec,
                &chosen,
                item.controller,
                Some(match item.kind {
                    StackItemKind::Spell => TargetingSource {
                        object: item.source,
                        card_def: state.objects.get(item.source).card_def,
                    },
                    StackItemKind::TriggeredAbility => {
                        let source_contract = item
                            .v4
                            .ability_source_contract
                            .ok_or("triggered stack item lost its source incarnation")?;
                        TargetingSource {
                            object: item.source,
                            card_def: source_contract.card_def,
                        }
                    }
                    StackItemKind::MadnessOffer | StackItemKind::ActivatedAbility => {
                        unreachable!("handled or untargeted")
                    }
                }),
                state,
            )
            .contains(&target),
        };
        any_legal |= same_incarnation && legal_now;
        chosen.push(target);
    }
    // 608.2b: a spell with multiple targets is countered by the rules only
    // when *all* targets are illegal. Its effect program remains responsible
    // for ignoring each individually-illegal target (Searing Blaze already
    // guards its creature leaf this way).
    Ok(any_legal)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpellDeparture {
    Move { source: ObjectId, to_zone: Zone },
    Cease { source: ObjectId },
}

/// Purely derives how a spell source leaves the stack. Flashback replaces
/// every physical departure with exile, while a virtual copy never enters a
/// card zone. All reachable resolution, fizzle, copy-continuation, and
/// counter/move paths share this one discriminator.
fn plan_spell_departure(
    state: &GameState,
    item: &StackItem,
    requested_zone: Zone,
) -> Result<SpellDeparture, String> {
    if requested_zone == Zone::Stack {
        return Err("a spell departure cannot request the stack zone".to_string());
    }
    validate_spell_source_contract_fields(state, item)?;
    if item.kind != StackItemKind::Spell {
        return Err("only a spell stack item has a spell departure".to_string());
    }
    Ok(if item.is_copy {
        SpellDeparture::Cease {
            source: item.source,
        }
    } else {
        SpellDeparture::Move {
            source: item.source,
            to_zone: if item.is_flashback {
                Zone::Exile
            } else {
                requested_zone
            },
        }
    })
}

fn apply_spell_departure(state: &mut GameState, departure: SpellDeparture) -> Result<(), String> {
    match departure {
        SpellDeparture::Move { source, to_zone } => {
            event::propose_and_commit(state, ProposedEvent::zone_change(source, to_zone));
        }
        SpellDeparture::Cease { source } => {
            // Resolution pops its item before applying the final
            // disposition; countering/copy-continuation paths apply while it
            // is still indexed. Both represent the same virtual cease.
            let live_generation = state.objects.get(source).zone_change_count;
            if state.stack.iter().any(|item| {
                item.kind == StackItemKind::Spell
                    && item.source == source
                    && item
                        .v4
                        .source_contract
                        .is_some_and(|contract| contract.zone_change_count == live_generation)
            }) && !event::cease_to_exist(state, source)
            {
                return Err("the virtual spell copy could not cease to exist".to_string());
            }
        }
    }
    Ok(())
}

/// Counters one exact stack incarnation. Physical spells use the shared
/// flashback/copy-aware departure contract; abilities leave the stack without
/// moving their source or any already-paid cost object. An absent id is a
/// valid no-op for a later Ward trigger whose targeter was already countered.
pub(crate) fn counter_stack_item_by_id(
    state: &mut GameState,
    stack_item_id: StackItemId,
) -> Result<Option<StackItem>, String> {
    if stack_item_id == StackItemId::default() {
        return Err("cannot counter an unstamped stack incarnation".to_string());
    }
    let positions = state
        .stack
        .iter()
        .enumerate()
        .filter_map(|(index, item)| (item.v4.stack_item_id == stack_item_id).then_some(index))
        .collect::<Vec<_>>();
    let position = match positions.as_slice() {
        [] => return Ok(None),
        [position] => *position,
        _ => return Err("duplicate private stack-incarnation ids".to_string()),
    };
    let item = state.stack[position].clone();
    validated_stack_item_target_spec(&item, state)?;
    let departure = if item.kind == StackItemKind::Spell {
        Some(plan_spell_departure(state, &item, Zone::Graveyard)?)
    } else {
        None
    };
    state.stack.remove(position);
    if let Some(departure) = departure {
        apply_spell_departure(state, departure)?;
    }
    Ok(Some(item))
}

/// Applies the last part of a completed spell resolution. Physical cards
/// move normally; virtual spell copies cease without entering a card zone.
fn finish_resolved_stack_item(state: &mut GameState, item: &StackItem) -> Result<(), String> {
    if item.inline_effect.is_some() || item.kind != StackItemKind::Spell {
        return Ok(());
    }
    let def = &card_def::CARD_DEFS[state.objects.get(item.source).card_def as usize];
    if item.v4.cast_method == Some(CastMethodV4::Omen) {
        supported_omen(def).ok_or("resolved Omen spell lost its definition")?;
        let departure = plan_spell_departure(state, item, Zone::Library)?;
        if item.is_copy {
            return apply_spell_departure(state, departure);
        }
        let owner = state.objects.get(item.source).owner;
        let shuffle = state
            .preflight_library_shuffle(owner)
            .map_err(|error| error.to_string())?;
        apply_spell_departure(state, departure)?;
        if state.objects.get(item.source).zone != Zone::Library
            || !state.players[owner.index()].library.contains(&item.source)
        {
            return Err("resolved Omen source did not enter its owner's library".to_string());
        }
        return state
            .commit_library_shuffle(owner, shuffle)
            .map_err(|error| error.to_string());
    }
    if !(def.has_type(CardType::Instant) || def.has_type(CardType::Sorcery)) {
        return Ok(());
    }
    let departure = plan_spell_departure(state, item, Zone::Graveyard)?;
    apply_spell_departure(state, departure)
}

/// A spell countered by the rules for having no legal targets performs none
/// of its resolution effects. In particular, an Omen card goes to the
/// graveyard here rather than applying its successful source shuffle.
fn finish_failed_stack_item(state: &mut GameState, item: &StackItem) -> Result<(), String> {
    if item.kind == StackItemKind::Spell && item.v4.cast_method == Some(CastMethodV4::Omen) {
        let departure = plan_spell_departure(state, item, Zone::Graveyard)?;
        return apply_spell_departure(state, departure);
    }
    if item.kind == StackItemKind::Spell {
        let definition = &card_def::CARD_DEFS[state.objects.get(item.source).card_def as usize];
        if !(definition.has_type(CardType::Instant) || definition.has_type(CardType::Sorcery)) {
            let departure = plan_spell_departure(state, item, Zone::Graveyard)?;
            return apply_spell_departure(state, departure);
        }
    }
    finish_resolved_stack_item(state, item)
}

/// Applies a source-moving `EffectOp` to a live spell stack item after exact
/// source validation. Returns `Ok(false)` when `object` is not a live spell
/// source, allowing ordinary object movement to use the event pipeline.
pub(crate) fn apply_live_stack_spell_departure(
    state: &mut GameState,
    object: ObjectId,
    requested_zone: Zone,
) -> Result<bool, String> {
    let Some(live_source) = state.objects.try_get(object) else {
        return Ok(false);
    };
    let live_generation = live_source.zone_change_count;
    let Some(item) = state
        .stack
        .iter()
        .find(|candidate| {
            candidate.source == object
                && candidate.kind == StackItemKind::Spell
                && candidate
                    .v4
                    .source_contract
                    .is_some_and(|contract| contract.zone_change_count == live_generation)
        })
        .cloned()
    else {
        return Ok(false);
    };
    validate_spell_stack_source(state, &item)?;
    let departure = plan_spell_departure(state, &item, requested_zone)?;
    apply_spell_departure(state, departure)?;
    Ok(true)
}

fn execute_resolving_program(
    state: &mut GameState,
    item: &StackItem,
    ctx: &ExecCtx,
    program: &EffectOp,
) -> ResolutionProgress {
    if !effect::contains_player_choice(program) {
        effect::execute(program, ctx, state);
        return ResolutionProgress::Complete;
    }

    match effect::begin_resumable_resolution(program, ctx, item.clone(), state) {
        Ok(effect::ResumableProgress::Complete(completed)) if *completed == *item => {
            ResolutionProgress::Complete
        }
        Ok(effect::ResumableProgress::Suspended) => {
            // Keep the resolving item visible throughout the decision, but
            // grant no priority/SBA/trigger window until the continuation
            // completes.
            state.stack.push(item.clone());
            ResolutionProgress::Suspended
        }
        Ok(effect::ResumableProgress::Complete(_)) | Err(_) => {
            state.stack.push(item.clone());
            state.engine.halted =
                Some((UnsupportedMechanic::InvalidEffectContinuation, item.source));
            ResolutionProgress::Suspended
        }
    }
}

fn resolve_top_of_stack(state: &mut GameState) -> ResolutionProgress {
    let item = state
        .stack
        .last()
        .expect("resolve_top_of_stack called with an empty stack");
    let resolving_source = item.source;
    let targets_legal = match stack_targets_still_legal(item, state) {
        Ok(legal) => legal,
        Err(_) => {
            state.engine.halted = Some((
                UnsupportedMechanic::InvalidEffectContinuation,
                resolving_source,
            ));
            return ResolutionProgress::Suspended;
        }
    };
    let item = state
        .stack
        .pop()
        .expect("resolve_top_of_stack called with an empty stack");
    // Single-shot signal to the trigger-collection pass that always
    // immediately follows this resolution (`collect_and_queue_triggers`,
    // called right after by every caller of `resolve_top_of_stack`) --
    // explicitly set every time (`Some` or `None`), never left stale from a
    // previous resolution. See `EngineState::pending_kicked_source`'s doc.
    state.engine.pending_kicked_source = if item.kicked { Some(item.source) } else { None };
    let ctx = ExecCtx {
        stack_item_id: Some(item.v4.stack_item_id),
        source: item.source,
        controller: item.controller,
        targets: item.targets.clone(),
        target_contracts: item.v4.target_contracts.clone(),
        discarded: item.discarded.clone(),
        paid_cost_refs: item.v4.paid_cost_refs.clone(),
        hidden_ability_source: item.v4.hidden_ability_source,
        ability_source_contract: item.v4.ability_source_contract,
        kicked: item.kicked,
        optional_additional_cost_paid: item.v4.optional_additional_cost_paid,
        x_value: item.v4.x_value,
    };

    if !targets_legal {
        if item.kind == StackItemKind::Spell && item.v4.cast_method == Some(CastMethodV4::Bestow) {
            effect::execute(
                &EffectOp::PutSourceOntoBattlefieldWithXPlusOneCounters,
                &ctx,
                state,
            );
            if state.engine.halted.is_some() {
                return ResolutionProgress::Suspended;
            }
            return ResolutionProgress::Complete;
        }
        if finish_failed_stack_item(state, &item).is_err() {
            state.engine.halted =
                Some((UnsupportedMechanic::InvalidEffectContinuation, item.source));
            return ResolutionProgress::Suspended;
        }
        return ResolutionProgress::Complete;
    }

    if item.kind == StackItemKind::Spell {
        if let Some(kind) = item.v4.optional_additional_cost_paid {
            state
                .engine
                .event_log
                .push(CommittedEvent::OptionalAdditionalCostPaid {
                    source: item.source,
                    kind,
                    paid_cost_refs: item.v4.paid_cost_refs.clone(),
                });
        }
    }

    if let Some(effect) = item.inline_effect.clone() {
        return execute_resolving_program(state, &item, &ctx, &effect);
    }

    let card_def_idx = state.objects.get(item.source).card_def;
    let def = &card_def::CARD_DEFS[card_def_idx as usize];
    // 601.2b: a modal spell resolves whichever mode was chosen at cast time
    // (`PendingCast::mode_chosen`, threaded onto the `StackItem` by
    // `finalize_cast`). Each printed mode retains its own definition-owned
    // program, while Omen continues to use its separate cast form.
    let program = if item.v4.cast_method == Some(CastMethodV4::Omen) {
        supported_omen(def).map(|omen| (omen.effect)())
    } else if item.v4.cast_method == Some(CastMethodV4::Bestow) {
        supported_bestow(def).map(|_| {
            EffectOp::PutSourceOntoBattlefieldAttachedToTargetWithXPlusOneCounters {
                target: ObjectRef::Target(0),
            }
        })
    } else {
        match item.mode_chosen {
            0 => (def.spell_effect)(),
            1 => def.mode2.as_ref().map(|mode| (mode.effect)()),
            2 => def.mode3.as_ref().map(|mode| (mode.effect)()),
            _ => None,
        }
    };
    if let Some(program) = program {
        if execute_resolving_program(state, &item, &ctx, &program) == ResolutionProgress::Suspended
        {
            return ResolutionProgress::Suspended;
        }
    }

    if state
        .engine
        .pending_spell_copy
        .as_ref()
        .is_some_and(|pending| pending.resolving_source == item.source)
    {
        // Resolution-time choices do not create a priority/SBA window. Put
        // the parent back exactly where it was so the public stack contains
        // a real resolving object throughout the suspended decision.
        state.stack.push(item);
        return ResolutionProgress::Suspended;
    }

    // 608.2m: instants/sorceries go to the graveyard as the last part of
    // resolution -- or to exile instead, if this was a flashback cast
    // (702.10e). Creatures/artifacts/enchantments handle entering the
    // battlefield themselves, via their own MoveObject effect.
    if item.v4.cast_method == Some(CastMethodV4::Omen) {
        if finish_resolved_stack_item(state, &item).is_err() {
            state.engine.halted =
                Some((UnsupportedMechanic::InvalidEffectContinuation, item.source));
            return ResolutionProgress::Suspended;
        }
    } else if def.has_type(CardType::Instant) || def.has_type(CardType::Sorcery) {
        let to_zone = match plan_spell_departure(state, &item, Zone::Graveyard) {
            Ok(SpellDeparture::Move { to_zone, .. }) => to_zone,
            Ok(SpellDeparture::Cease { .. })
                if state.engine.pending_discard.is_none()
                    && state.engine.pending_optional_cost.is_none() =>
            {
                if finish_resolved_stack_item(state, &item).is_err() {
                    state.engine.halted =
                        Some((UnsupportedMechanic::InvalidEffectContinuation, item.source));
                    return ResolutionProgress::Suspended;
                }
                return ResolutionProgress::Complete;
            }
            Ok(SpellDeparture::Cease { .. }) | Err(_) => {
                state.engine.halted =
                    Some((UnsupportedMechanic::InvalidEffectContinuation, item.source));
                return ResolutionProgress::Suspended;
            }
        };
        if let Some(pd) = state.engine.pending_discard.as_mut() {
            // The effect just resolved into `EffectOp::DiscardCards`
            // (Faithless Looting: "draw two, then discard two"), which
            // stages `pending_discard` and returns *before* the discard is
            // actually answered -- see that leaf's doc and
            // `DiscardResume::FinishSpellResolution`'s. The spell can't
            // reach its post-resolution zone until that discard (part of
            // its own resolution) is done, so defer the move instead of
            // doing it here.
            pd.resume = DiscardResume::FinishSpellResolution {
                source: item.source,
                to_zone,
            };
        } else if let Some(poc) = state.engine.pending_optional_cost.as_mut() {
            // Same 608.2m deferral, one layer further out: the effect
            // resolved into `EffectOp::MayPayCostThen` (Highway Robbery:
            // "you may... if you do, draw two cards"), which stages
            // `pending_optional_cost` and returns before the "may pay?"
            // question is even answered, let alone `then` run. Root-caused
            // (Sol #90, increment 11) against several golden-trace games
            // (e.g. `game_20260713_002147_0002.txt` decision 115): moving
            // Highway Robbery to the graveyard *here* put it there several
            // decisions before the reference's own graveyard snapshot ever
            // shows it -- the reference doesn't finish resolving Highway
            // Robbery until its own optional-cost choice (and whichever
            // sub-cost it leads to) is fully paid. See
            // `PendingOptionalCost::spell_resume`'s doc for where the
            // deferred move actually happens once that's all done.
            poc.spell_resume = Some((item.source, to_zone));
        } else {
            if finish_resolved_stack_item(state, &item).is_err() {
                state.engine.halted =
                    Some((UnsupportedMechanic::InvalidEffectContinuation, item.source));
                return ResolutionProgress::Suspended;
            }
        }
    }
    ResolutionProgress::Complete
}

/// Moves `state.step`/`state.active_player`/`state.turn` to the next step,
/// clearing both players' mana pools (500.4: unused mana empties at the end
/// of every step and phase -- a turn-based action, unconditional on whether
/// the step just ended ever granted priority, so this single choke point
/// covers Untap/Cleanup's transitions too, not just the priority-bearing
/// steps), running that step's turn-based entry action (untap, draw,
/// cleanup, combat damage), and resetting priority. Only skips
/// declare-blockers/combat-damage, and only when the active player declared
/// zero attackers (509.4/510.4-ish -- no card in this pool changes that
/// "zero attackers" trigger, so this increment doesn't need the exact rule
/// number to get it right). Declare Attackers itself is *never* skipped,
/// even when no creature could possibly attack: 508.1 makes it a
/// turn-based action that always happens (it's how "the active player
/// declares no attackers" gets decided in the first place), still followed
/// by its own priority round -- confirmed against the real corpus: e.g.
/// `game_20260712_194609_0010.txt` decision #4 is a real, 2-candidate
/// `ACTIVATE_ABILITY_OR_SPELL` record with `phase="Combat"` on turn 1,
/// before either player has a creature on the battlefield (so `eligible`
/// is empty both sides -- yet the reference still asks). An earlier version
/// of this function skipped the whole step whenever `eligible` was empty,
/// which silently ate that priority window entirely -- the resulting
/// "missing decision" left the *next* real trace record unconsumed until
/// some later, unrelated kernel decision, at which point the kernel's
/// `state.step` (already advanced past combat) no longer matched what that
/// stale record was captured against, manifesting downstream as spurious
/// extra/missing `ACTIVATE_ABILITY_OR_SPELL` candidates rather than as an
/// obviously-combat-shaped divergence. See `Decision::DeclareAttackers`'s
/// doc and the replay comparator's handling of an empty `eligible`.
fn advance_step(state: &mut GameState) {
    let cur_idx = STEP_ORDER
        .iter()
        .position(|&s| s == state.step)
        .expect("state.step is always a STEP_ORDER member");
    let mut next_idx = cur_idx + 1;

    if next_idx >= STEP_ORDER.len() {
        state.active_player = state.active_player.opponent();
        // Round counter advances when the active player comes back around to
        // whoever started the game, not specifically to physical seat P0.
        // `state.starting_player` defaults to `PlayerId::P0`, reproducing
        // this condition's original `== PlayerId::P0` literal exactly when
        // unset; see `P1-METAMORPHIC-AUDIT-DESIGN-V4.md` Section 1.2, coupled
        // site 3.
        if state.active_player == state.starting_player {
            state.turn += 1;
        }
        next_idx = 0;
    }

    let mut next = STEP_ORDER[next_idx];
    // 508.4/509.1: `DeclareBlockers` (and everything after it this combat)
    // is skipped not just when *no attackers were ever declared*, but
    // whenever *none of them are still around to be blocked* -- confirmed
    // against the real reference: `DeclareBlockersStep`/`CombatDamageStep`
    // both override `skipStep()` to check `Combat.noAttackers()`, which
    // recomputes live from the mutable attacker set on every call (so a
    // declared attacker that died in the interim drops out), and
    // `Phase.play()` re-checks `skipStep()` immediately before *each* step,
    // not just once at Declare Attackers. `is_still_in_combat` already
    // exists for the exact same "declared, but no longer on the
    // battlefield" check (`combat_damage_wave`'s doc) -- reused here rather
    // than re-deriving it. `.all(..)` over an empty `attackers` list is
    // vacuously true, so this subsumes the original "nothing was ever
    // declared" case too. Root-caused against
    // `game_20260713_002153_0010.txt`: SelfPlay declares a single attacker
    // (Voldaren Epicure) then kills it themselves with Lava Dart still
    // inside the Declare Attackers priority window; the reference logs
    // nothing further for either player until Postcombat Main, but the
    // kernel -- still holding the now-dead attacker's `ObjectId` in
    // `combat.attackers` -- entered `DeclareBlockers` for real and asked a
    // phantom post-block priority question.
    if next == Step::DeclareBlockers
        && state
            .engine
            .combat
            .attackers
            .iter()
            .all(|&id| !is_still_in_combat(state, id))
    {
        next = Step::EndCombat;
    }

    state.step = next;
    state.players[0].mana_pool = [0; 6];
    state.players[1].mana_pool = [0; 6];
    run_step_entry_action(state, next);
    reset_priority(state);
}

fn attachment_prevents_untap(state: &GameState, host: ObjectId) -> bool {
    let host_object = state.objects.get(host);
    host_object.attachments.iter().copied().any(|attachment| {
        let Some(aura) = state.objects.try_get(attachment) else {
            return false;
        };
        if aura.zone != Zone::Battlefield
            || aura.v4.attached_to
                != Some(crate::state::ObjectLinkV4 {
                    object: host,
                    zone_change_count: host_object.zone_change_count,
                })
        {
            return false;
        }
        matches!(
            card_def::CARD_DEFS[aura.card_def as usize].attachment,
            Some(card_def::AttachmentDef::AuraCreature {
                prevents_untap: true
            })
        )
    })
}

fn run_step_entry_action(state: &mut GameState, step: Step) {
    match step {
        Step::Untap => {
            let p = state.active_player;
            state
                .engine
                .until_next_turn_keywords
                .retain(|effect| !(effect.holder == p && effect.expires_at_turn <= state.turn));
            for (_, object) in state.objects.iter_mut() {
                object
                    .v4
                    .goaded_by
                    .retain(|goad| !(goad.player == p && goad.expires_at_turn <= state.turn));
            }
            // Battlefield zone vectors are ownership-oriented. Untap is
            // controller-oriented, so inspect live objects to cover control
            // changes without consuming another player's skip marker.
            let permanents = state
                .objects
                .iter()
                .filter_map(|(id, object)| {
                    (object.zone == Zone::Battlefield && object.controller == p).then_some(id)
                })
                .collect::<Vec<_>>();
            for id in permanents {
                let prevented_by_attachment = attachment_prevents_untap(state, id);
                let obj = state.objects.get_mut(id);
                if obj.v4.skip_next_untap {
                    obj.v4.skip_next_untap = false;
                } else if !prevented_by_attachment {
                    if obj.tapped && obj.counters.stun > 0 {
                        obj.counters.stun -= 1;
                    } else {
                        obj.tapped = false;
                    }
                }
                obj.summoning_sick = false;
            }
            state.players[0].draws_this_turn = 0;
            state.players[1].draws_this_turn = 0;
            state.players[0].spells_cast_this_turn = 0;
            state.players[1].spells_cast_this_turn = 0;
            for (_, object) in state.objects.iter_mut() {
                object.v4.ability_uses_this_turn.clear();
            }
            // See `PlayPermissionExpiry`'s doc: the *holder's* own Untap
            // marks the start of their "next turn" for an "until end of
            // your next turn" impulse-draw permission (Clockwork
            // Percussionist, Reckless Impulse) -- only flips once (the
            // first such Untap after the permission was granted), not
            // re-armed on a later one.
            for perm in state.engine.exile_play_permissions.iter_mut() {
                if perm.holder == p {
                    if let PlayPermissionExpiry::UntilHoldersNextTurn {
                        holder_turn_started,
                    } = &mut perm.expiry
                    {
                        *holder_turn_started = true;
                    }
                }
            }
        }
        Step::Upkeep => {
            let p = state.active_player;
            if state.initiative == Some(p) {
                let Some(source) = state.engine.initiative_source else {
                    state.engine.halted =
                        Some((UnsupportedMechanic::InvalidEffectContinuation, ObjectId(0)));
                    return;
                };
                if event::log_initiative_trigger(
                    state,
                    p,
                    source,
                    InitiativeTriggerKindV1::VentureAtUpkeep,
                )
                .is_err()
                {
                    state.engine.halted = Some((
                        UnsupportedMechanic::InvalidEffectContinuation,
                        source.source,
                    ));
                    return;
                }
                collect_and_queue_triggers(state);
            }
        }
        Step::Draw => {
            let p = state.active_player;
            // 103.8a: the starting player skips the draw step of their
            // very first turn. `turn == 1 && p == state.starting_player`
            // uniquely identifies that turn under this kernel's round-based
            // turn numbering (see module doc): `turn` only becomes 1 again...
            // it never does, it's monotonic, so this combination occurs
            // exactly once, at the start of the game. `state.starting_player`
            // is the explicit authority set once at construction (defaults
            // to `PlayerId::P0`, reproducing this condition's original
            // `p == PlayerId::P0` literal exactly when unset; see
            // `P1-METAMORPHIC-AUDIT-DESIGN-V4.md` Section 1.2, coupled site 2).
            let is_first_turn_of_the_game = state.turn == 1 && p == state.starting_player;
            if !is_first_turn_of_the_game {
                event::propose_and_commit(state, ProposedEvent::draw(p));
                collect_and_queue_triggers(state);
            }
        }
        Step::Main1 => {
            // 714.3b: as the precombat main phase begins, each front-face Saga
            // controlled by the active player receives one lore counter. All
            // chapter markers are logged before their triggers are collected.
            let sagas = state
                .objects
                .iter()
                .filter_map(|(id, object)| {
                    (object.zone == Zone::Battlefield
                        && object.controller == state.active_player
                        && object.v4.face_index == 0
                        && card_def::CARD_DEFS[object.card_def as usize].saga.is_some())
                    .then_some(id)
                })
                .collect::<Vec<_>>();
            for saga in sagas {
                let chapter_count = card_def::CARD_DEFS[state.objects.get(saga).card_def as usize]
                    .saga
                    .as_ref()
                    .map_or(0, |definition| definition.chapter_effects.len());
                let current = state.objects.get(saga).counters.lore;
                if current < 0 || usize::try_from(current).unwrap_or(usize::MAX) >= chapter_count {
                    continue;
                }
                let chapter = {
                    let lore = &mut state.objects.get_mut(saga).counters.lore;
                    *lore = lore
                        .checked_add(1)
                        .expect("supported Saga lore counter fits i16");
                    u8::try_from(*lore).expect("supported Saga chapter fits u8")
                };
                event::log_saga_chapter(state, saga, chapter);
            }
            collect_and_queue_triggers(state);
        }
        Step::BeginCombat => {
            state.engine.combat = CombatState::default();
        }
        Step::CombatDamage => {
            deal_combat_damage(state);
        }
        Step::Cleanup => {
            // 514.1/514.2: reset damage, "until end of turn" effects end,
            // then discard down to the maximum hand size, then reset the
            // land-drop counter for the player whose turn just ended.
            for (_, obj) in state.objects.iter_mut() {
                obj.damage = 0;
                obj.v4.deathtouch_damage = false;
            }
            state.engine.until_end_of_turn.clear();
            state.engine.active_replacements.retain(|replacement| {
                !matches!(
                    replacement.kind,
                    event::ReplacementEffectKind::PreventDamageFromColorUntilEndOfTurn { .. }
                )
            });
            let p = state.active_player;
            state.players[p.index()].lands_played_this_turn = 0;
            // `EndOfTurn` permissions expire at the very next Cleanup,
            // whoever's; `UntilHoldersNextTurn` only once it's the
            // *holder's* own Cleanup, and only after their own Untap has
            // already opened this "next turn" -- see
            // `PlayPermissionExpiry`'s doc.
            state
                .engine
                .exile_play_permissions
                .retain(|perm| match perm.expiry {
                    PlayPermissionExpiry::EndOfTurn => false,
                    PlayPermissionExpiry::UntilHoldersNextTurn {
                        holder_turn_started,
                    } => !(perm.holder == p && holder_turn_started),
                });
            let hand_size = state.players[p.index()].hand.len();
            if hand_size > 7 {
                state.engine.pending_discard = Some(PendingDiscard {
                    player: p,
                    count: (hand_size - 7) as u32,
                    resume: DiscardResume::None,
                });
            }
        }
        _ => {}
    }
}

// --------------------------------------------------------------- combat

/// A permanent's own-name-keyed conditional static self-boost -- "As long as
/// you control an artifact, {this} gets +1/+0 and has haste" (Goblin Tomb
/// Raider). Always-on continuous effects gated on live board state, not a
/// resolution-time `EffectOp` (there's no cast/trigger moment to run one
/// at): re-evaluated fresh every time `effective_power`/`effective_toughness`/
/// `has_effective_keyword` reads it, same "recompute, don't persist" shape
/// `EffectCond::LandfallThisTurn`/`ControlsArtifactCount` already use. A
/// per-name table (matching `trigger::triggers_for`'s own style) rather
/// than a generic layered-continuous-effects system: only one card needs
/// this shape in this pool.
pub struct StaticSelfBoostDef {
    pub condition: fn(PlayerId, &GameState) -> bool,
    pub power: i32,
    pub toughness: i32,
    pub grant_haste: bool,
}

fn controls_an_artifact(controller: PlayerId, state: &GameState) -> bool {
    state.players[controller.index()]
        .battlefield
        .iter()
        .any(|&id| {
            let def = &card_def::CARD_DEFS[state.objects.get(id).card_def as usize];
            def.is_executable() && def.has_type(CardType::Artifact)
        })
}

pub(crate) fn static_self_boost_for(name: &str) -> Option<StaticSelfBoostDef> {
    match name {
        "Goblin Tomb Raider" => Some(StaticSelfBoostDef {
            condition: controls_an_artifact,
            power: 1,
            toughness: 0,
            grant_haste: true,
        }),
        _ => None,
    }
}

fn valid_bestow_attachment_host(state: &GameState, aura: ObjectId) -> Option<ObjectId> {
    let object = state.objects.try_get(aura)?;
    let definition = card_def::CARD_DEFS.get(object.card_def as usize)?;
    if object.zone != Zone::Battlefield || definition.bestow.is_none() {
        return None;
    }
    let link = object.v4.attached_to?;
    let host = state.objects.try_get(link.object)?;
    let host_definition = card_def::CARD_DEFS.get(host.card_def as usize)?;
    let host_is_attached_bestow = host_definition.bestow.is_some() && host.v4.attached_to.is_some();
    (host.zone == Zone::Battlefield
        && host.zone_change_count == link.zone_change_count
        && host_definition
            .types_for_face(host.v4.face_index)
            .contains(&CardType::Creature)
        && !host_is_attached_bestow
        && host.attachments.contains(&aura))
    .then_some(link.object)
}

fn bestow_host_counter_bonus(state: &GameState, host: ObjectId) -> i32 {
    state
        .objects
        .try_get(host)
        .map(|object| {
            object
                .attachments
                .iter()
                .copied()
                .filter(|&aura| valid_bestow_attachment_host(state, aura) == Some(host))
                .map(|aura| i32::from(state.objects.get(aura).counters.plus1_plus1))
                .sum()
        })
        .unwrap_or(0)
}

/// Effective card-type query for one live object face. Nonbattlefield cards
/// are always front-face objects because every zone change resets the face.
pub fn object_has_type(state: &GameState, id: ObjectId, card_type: CardType) -> bool {
    state.objects.try_get(id).is_some_and(|object| {
        card_def::CARD_DEFS
            .get(object.card_def as usize)
            .is_some_and(|def| {
                if def.bestow.is_some() && valid_bestow_attachment_host(state, id).is_some() {
                    return card_type == CardType::Enchantment;
                }
                def.types_for_face(object.v4.face_index)
                    .contains(&card_type)
            })
    })
}

/// Effective W/U/B/R/G/C mask already materialized on the object's current
/// incarnation. Missing objects fail closed as colorless.
pub fn object_color_mask(state: &GameState, id: ObjectId) -> u8 {
    state
        .objects
        .try_get(id)
        .map_or(0, |object| object.v4.effective_color_mask)
}

pub fn object_is_monocolored(state: &GameState, id: ObjectId) -> bool {
    const WUBRG_MASK: u8 = (1 << 5) - 1;
    (object_color_mask(state, id) & WUBRG_MASK).count_ones() == 1
}

pub fn damage_is_prevented_by_protection(
    state: &GameState,
    source: ObjectId,
    target: Target,
) -> bool {
    let Target::Object(target) = target else {
        return false;
    };
    state.objects.try_get(target).is_some_and(|object| {
        object.zone == Zone::Battlefield
            && has_effective_keyword(state, target, Keywords::PROTECTION_FROM_MONOCOLORED)
            && object_is_monocolored(state, source)
    })
}

pub fn effective_power(state: &GameState, id: ObjectId) -> i32 {
    let obj = state.objects.get(id);
    let def = &card_def::CARD_DEFS[obj.card_def as usize];
    let mut power = def.power_for_face(obj.v4.face_index).unwrap_or(0) as i32
        + obj.counters.plus1_plus1 as i32
        - obj.counters.minus1_minus1 as i32;
    power += bestow_host_counter_bonus(state, id);
    if def.is_executable() {
        if let Some(boost) = static_self_boost_for(def.name) {
            if (boost.condition)(obj.controller, state) {
                power += boost.power;
            }
        }
    }
    for (_, equipment) in attached_equipment_profiles(state, id) {
        power += i32::from(equipment.power_delta);
    }
    for eff in &state.engine.until_end_of_turn {
        match eff {
            UntilEndOfTurnEffect::ResolvedSetEffect {
                object_ids,
                power: p,
                ..
            } if object_ids.contains(&id) => power += p,
            UntilEndOfTurnEffect::ResolvedObjectEffect {
                object_id,
                object_zone_change_count,
                power: p,
                ..
            } if *object_id == id
                && obj.zone == Zone::Battlefield
                && obj.zone_change_count == *object_zone_change_count =>
            {
                power += p
            }
            _ => {}
        }
    }
    power
}

/// `effective_power`'s toughness twin -- see that function's doc. No card in
/// this pool's static/team boosts actually carries a nonzero toughness
/// delta (Goblin Tomb Raider/Goblin Bushwhacker are both "+1/+0"), but this
/// stays a real, symmetric code path rather than a power-only shortcut so
/// the next card that pumps toughness doesn't have to rediscover this
/// shape.
pub fn effective_toughness(state: &GameState, id: ObjectId) -> i32 {
    let obj = state.objects.get(id);
    let def = &card_def::CARD_DEFS[obj.card_def as usize];
    let mut toughness = def.toughness_for_face(obj.v4.face_index).unwrap_or(0) as i32
        + obj.counters.plus1_plus1 as i32
        - obj.counters.minus1_minus1 as i32
        - obj.counters.minus0_minus1 as i32;
    toughness += bestow_host_counter_bonus(state, id);
    if def.is_executable() {
        if let Some(boost) = static_self_boost_for(def.name) {
            if (boost.condition)(obj.controller, state) {
                toughness += boost.toughness;
            }
        }
    }
    for (_, equipment) in attached_equipment_profiles(state, id) {
        toughness += i32::from(equipment.toughness_delta);
    }
    for eff in &state.engine.until_end_of_turn {
        match eff {
            UntilEndOfTurnEffect::ResolvedSetEffect {
                object_ids,
                toughness: t,
                ..
            } if object_ids.contains(&id) => toughness += t,
            UntilEndOfTurnEffect::ResolvedObjectEffect {
                object_id,
                object_zone_change_count,
                toughness: t,
                ..
            } if *object_id == id
                && obj.zone == Zone::Battlefield
                && obj.zone_change_count == *object_zone_change_count =>
            {
                toughness += t
            }
            _ => {}
        }
    }
    toughness
}

/// Whether `id` currently has `kw`, folding in every source this kernel
/// models: the card's own static `Keywords`, `static_self_boost_for`'s
/// conditional self-grant (Goblin Tomb Raider's haste), and any active
/// `UntilEndOfTurnEffect::ResolvedSetEffect` naming it (Goblin Bushwhacker's/
/// Rally at the Hornburg's granted haste). Only `Keywords::HASTE` has a
/// granted source to fold in this pool -- `Flying`/`Reach`/etc. still just
/// read the card's own static bit, same as before this function existed.
/// Both boost sources route through this same query path (per external
/// review: not a parallel mechanism) -- there is no combat- or SBA-specific
/// shortcut anywhere else that reads power/toughness/keywords directly.
pub fn has_effective_keyword(state: &GameState, id: ObjectId, kw: Keywords) -> bool {
    let obj = state.objects.get(id);
    let def = &card_def::CARD_DEFS[obj.card_def as usize];
    if !def.is_executable() {
        return false;
    }
    if (kw == Keywords::REACH || kw == Keywords::TRAMPLE)
        && obj.attachments.iter().copied().any(|aura| {
            valid_bestow_attachment_host(state, aura) == Some(id)
                && card_def::CARD_DEFS[state.objects.get(aura).card_def as usize]
                    .keywords
                    .has(kw)
        })
    {
        return true;
    }
    if def.keywords_for_face(obj.v4.face_index).has(kw) {
        return true;
    }
    if kw.has(Keywords::LIFELINK) && obj.v4.lifelink_keyword_counters > 0 {
        return true;
    }
    if attached_equipment_profiles(state, id)
        .into_iter()
        .any(|(equipment_id, equipment)| {
            let equipment_controller = state.objects.get(equipment_id).controller;
            let keywords = if state.active_player == equipment_controller {
                equipment.controller_turn_keywords
            } else {
                equipment.other_turn_keywords
            };
            keywords.has(kw)
        })
    {
        return true;
    }
    if state.engine.until_next_turn_keywords.iter().any(|effect| {
        effect.object_id == id
            && obj.zone == Zone::Battlefield
            && obj.zone_change_count == effect.object_zone_change_count
            && effect.keywords.has(kw)
    }) {
        return true;
    }
    if state.engine.until_end_of_turn.iter().any(|effect| {
        matches!(
            effect,
            UntilEndOfTurnEffect::ResolvedObjectKeywordEffect {
                object_id,
                object_zone_change_count,
                keywords,
                ..
            } if *object_id == id
                && obj.zone == Zone::Battlefield
                && obj.zone_change_count == *object_zone_change_count
                && keywords.has(kw)
        )
    }) {
        return true;
    }
    if kw.has(Keywords::HASTE) {
        if let Some(boost) = static_self_boost_for(def.name) {
            if boost.grant_haste && (boost.condition)(obj.controller, state) {
                return true;
            }
        }
        for eff in &state.engine.until_end_of_turn {
            let grants = match eff {
                UntilEndOfTurnEffect::ResolvedSetEffect {
                    object_ids,
                    grant_haste,
                    ..
                } => *grant_haste && object_ids.contains(&id),
                UntilEndOfTurnEffect::ResolvedObjectEffect {
                    object_id,
                    object_zone_change_count,
                    grant_haste,
                    ..
                } => {
                    *grant_haste
                        && *object_id == id
                        && obj.zone == Zone::Battlefield
                        && obj.zone_change_count == *object_zone_change_count
                }
                UntilEndOfTurnEffect::ResolvedObjectKeywordEffect { .. } => false,
                UntilEndOfTurnEffect::SyntheticMarker(_) => false,
                UntilEndOfTurnEffect::DamageCannotBePrevented { .. } => false,
            };
            if grants {
                return true;
            }
        }
    }
    false
}

pub(crate) fn attached_equipment_profiles(
    state: &GameState,
    host: ObjectId,
) -> impl Iterator<Item = (ObjectId, card_def::EquipmentDef)> + '_ {
    let host_link = state
        .objects
        .try_get(host)
        .filter(|host_object| host_object.zone == Zone::Battlefield)
        .map(|host_object| crate::state::ObjectLinkV4 {
            object: host,
            zone_change_count: host_object.zone_change_count,
        });
    state
        .objects
        .try_get(host)
        .into_iter()
        .filter(|host_object| host_object.zone == Zone::Battlefield)
        .flat_map(|host_object| host_object.attachments.iter().copied())
        .filter_map(move |equipment_id| {
            let equipment_object = state.objects.try_get(equipment_id)?;
            if equipment_object.zone != Zone::Battlefield
                || equipment_object.v4.attached_to != host_link
            {
                return None;
            }
            let definition = card_def::CARD_DEFS.get(equipment_object.card_def as usize)?;
            definition
                .equipment
                .map(|equipment| (equipment_id, equipment))
        })
}

pub fn effective_subtype_ids(state: &GameState, id: ObjectId) -> Vec<u16> {
    let Some(object) = state.objects.try_get(id) else {
        return Vec::new();
    };
    let mut subtype_ids = object.v4.effective_subtype_ids.clone();
    subtype_ids.extend(
        attached_equipment_profiles(state, id)
            .filter_map(|(_, equipment)| equipment.add_subtype.map(|subtype| subtype.stable_id())),
    );
    subtype_ids.sort_unstable();
    subtype_ids.dedup();
    subtype_ids
}

pub fn has_effective_subtype(state: &GameState, id: ObjectId, subtype: card_def::Subtype) -> bool {
    let Some(object) = state.objects.try_get(id) else {
        return false;
    };
    if object
        .v4
        .effective_subtype_ids
        .binary_search(&subtype.stable_id())
        .is_ok()
    {
        return true;
    }
    attached_equipment_profiles(state, id)
        .any(|(_, equipment)| equipment.add_subtype == Some(subtype))
}

fn participates_in_wave(state: &GameState, id: ObjectId, first_strike_wave: bool) -> bool {
    let has_fs = has_effective_keyword(state, id, Keywords::FIRST_STRIKE);
    let has_ds = has_effective_keyword(state, id, Keywords::DOUBLE_STRIKE);
    if first_strike_wave {
        has_fs || has_ds
    } else {
        !has_fs || has_ds
    }
}

fn combat_has_first_or_double_strike(state: &GameState) -> bool {
    let all_combatants = state.engine.combat.attackers.iter().copied().chain(
        state
            .engine
            .combat
            .blocked_by
            .iter()
            .flat_map(|(_, bs)| bs.iter().copied()),
    );
    all_combatants.into_iter().any(|id| {
        has_effective_keyword(state, id, Keywords::FIRST_STRIKE)
            || has_effective_keyword(state, id, Keywords::DOUBLE_STRIKE)
    })
}

/// 510: all combat damage is dealt in one simultaneous batch, via
/// `event::propose_and_commit_batch` -- unless first/double strike is in
/// play, in which case there are two such batches (510.5), with an SBA
/// check between them so a first-strike kill actually removes its victim
/// before the normal wave. No creature in this increment's pool has
/// either keyword, so `combat_has_first_or_double_strike` is always
/// false here and this always takes the single-wave path; the branch
/// exists so the next increment that adds a first-strike creature only
/// needs to flip a keyword bit, not restructure this function.
fn deal_combat_damage(state: &mut GameState) {
    if combat_has_first_or_double_strike(state) {
        combat_damage_wave(state, true);
        trigger::sba_fixed_point(state);
        combat_damage_wave(state, false);
    } else {
        combat_damage_wave(state, false);
    }
}

/// 510.1c: "an attacking creature that's been removed from combat... doesn't
/// assign combat damage." A creature is removed from combat the instant it
/// leaves the battlefield (undefined-but-implied by 506.4, made explicit by
/// 510.1c's own wording) -- most commonly here, a creature killed by a
/// burn spell (Lava Dart, Lightning Bolt, ...) cast during the priority
/// window `apply_declare_blockers`/`finalize_activation`/etc already
/// grants before the actual `Step::CombatDamage` entry action runs
/// (`deal_combat_damage`). Root-caused against
/// `game_20260713_002204_0027.txt` decision 304: the attacking player
/// Lava-Darted their own just-declared (unblocked) attacker dead in
/// response, well before combat damage -- the reference correctly deals
/// zero combat damage from it (it's simply not there anymore by the time
/// the damage step runs), but this function used to compute `effective_
/// power`/assign damage purely from `EngineState::combat`'s *snapshot* of
/// who was attacking/blocking when blocks were declared, never rechecking
/// whether each participant was still actually on the battlefield by
/// damage time.
fn is_still_in_combat(state: &GameState, id: ObjectId) -> bool {
    state.objects.get(id).zone == Zone::Battlefield
}

fn combat_damage_wave(state: &mut GameState, first_strike_wave: bool) {
    let attackers = state.engine.combat.attackers.clone();
    let blocked_by = state.engine.combat.blocked_by.clone();
    let mut events = Vec::new();

    for &attacker in &attackers {
        if !is_still_in_combat(state, attacker)
            || !participates_in_wave(state, attacker, first_strike_wave)
        {
            continue;
        }
        let power = effective_power(state, attacker);
        if power <= 0 {
            continue;
        }
        if let Some((_, blockers)) = blocked_by.iter().find(|(a, _)| *a == attacker) {
            assign_attacker_damage_to_blockers(state, attacker, power, blockers, &mut events);
        } else {
            let defender = state.objects.get(attacker).controller.opponent();
            events.push(ProposedEvent::damage(
                attacker,
                Target::Player(defender),
                power,
            ));
        }
    }
    for (attacker, blockers) in &blocked_by {
        for &blocker in blockers {
            if !is_still_in_combat(state, blocker)
                || !participates_in_wave(state, blocker, first_strike_wave)
            {
                continue;
            }
            let power = effective_power(state, blocker);
            if power > 0 {
                events.push(ProposedEvent::damage(
                    blocker,
                    Target::Object(*attacker),
                    power,
                ));
            }
        }
    }

    let event_start = state.engine.event_log.len();
    event::propose_and_commit_batch(state, events);
    let combat_player_damage = state.engine.event_log[event_start..]
        .iter()
        .filter_map(|event| match *event {
            CommittedEvent::Damage {
                source,
                target: Target::Player(player),
                amount,
            } if amount > 0 => Some((
                source,
                state.objects.get(source).zone_change_count,
                player,
                amount,
            )),
            _ => None,
        })
        .collect::<Vec<_>>();
    for (source, source_zone_change_count, player, amount) in combat_player_damage {
        event::log_combat_damage_to_player(state, source, source_zone_change_count, player, amount);
    }
    if let Some(holder) = state.initiative {
        let transfer_player = state.engine.event_log[event_start..]
            .iter()
            .find_map(|event| match event {
                CommittedEvent::CombatDamageToPlayer {
                    source,
                    player,
                    amount,
                    ..
                } if *player == holder
                    && *amount > 0
                    && state.objects.try_get(*source).is_some_and(|object| {
                        object.zone == Zone::Battlefield
                            && object_has_type(state, *source, CardType::Creature)
                    }) =>
                {
                    Some(state.objects.get(*source).controller)
                }
                _ => None,
            });
        if let Some(player) = transfer_player {
            let Some(source) = state.engine.initiative_source else {
                state.engine.halted =
                    Some((UnsupportedMechanic::InvalidEffectContinuation, ObjectId(0)));
                return;
            };
            if event::log_initiative_trigger(
                state,
                player,
                source,
                InitiativeTriggerKindV1::CombatTransfer,
            )
            .is_err()
            {
                state.engine.halted = Some((
                    UnsupportedMechanic::InvalidEffectContinuation,
                    source.source,
                ));
                return;
            }
        }
    }
    collect_and_queue_triggers(state);
}

/// Resolves one definition-owned ninjutsu effect. The returned attacker was
/// paid before this ability reached the stack. Resolution still requires the
/// same live post-blockers combat, and only the exact revealed hand
/// incarnation can become a tapped attacking permanent.
pub(crate) fn put_ninjutsu_source_onto_battlefield_attacking(
    state: &mut GameState,
    source: ObjectId,
    controller: PlayerId,
    expected_source: Option<ObjectLinkV4>,
) -> Result<(), String> {
    if !ninjutsu_combat_window_open(controller, state) {
        return Err("ninjutsu resolved outside its post-blockers combat window".to_string());
    }
    let expected_source = expected_source.ok_or("ninjutsu effect lost its hand source binding")?;
    if expected_source.object != source {
        return Err("ninjutsu effect source binding changed".to_string());
    }
    let object = state
        .objects
        .try_get(source)
        .ok_or("ninjutsu source object is missing")?;
    let definition = card_def::CARD_DEFS
        .get(object.card_def as usize)
        .ok_or("ninjutsu source definition is missing")?;
    if object.owner != controller || !definition.has_type(CardType::Creature) {
        return Err("ninjutsu source definition or owner changed".to_string());
    }
    // Mage's NinjutsuEffect returns without moving a source that left the
    // exact revealed hand incarnation. The ability remains valid and simply
    // resolves without effect.
    if object.zone_change_count != expected_source.zone_change_count
        || object.zone != Zone::Hand
        || !state.players[controller.index()].hand.contains(&source)
    {
        return Ok(());
    }
    // A card in hand cannot be in combat (506.4): any combat entry for this
    // id is a stale one from an earlier battlefield incarnation, e.g. a ninja
    // that attacked, was returned by another ninjutsu, and now re-enters.
    state.engine.combat.attackers.retain(|&a| a != source);
    state
        .engine
        .combat
        .blocked_by
        .retain(|(attacker, _)| *attacker != source);
    event::propose_and_commit(state, ProposedEvent::zone_change(source, Zone::Battlefield));
    state.objects.get_mut(source).tapped = true;
    state.engine.combat.attackers.push(source);
    Ok(())
}

/// 510.1c, no-trample simplification: lethal damage (toughness minus
/// damage already marked) goes to each blocker in `blockers`' order
/// except the last, which absorbs whatever power remains (there being no
/// trample in this pool, that's the only legal recipient once the
/// attacker itself has already been assigned to blockers rather than the
/// player). A single blocker just gets it all directly. The order itself
/// is `CombatState::blocked_by`'s fixed deterministic sort -- see that
/// field's doc for why this is a stubbed decision point, not a real one,
/// this increment.
fn assign_attacker_damage_to_blockers(
    state: &GameState,
    attacker: ObjectId,
    power: i32,
    blockers: &[ObjectId],
    events: &mut Vec<ProposedEvent>,
) {
    if blockers.len() == 1 {
        events.push(ProposedEvent::damage(
            attacker,
            Target::Object(blockers[0]),
            power,
        ));
        return;
    }
    let mut remaining = power;
    for (i, &blocker) in blockers.iter().enumerate() {
        if remaining <= 0 {
            break;
        }
        let is_last = i + 1 == blockers.len();
        let assign = if is_last {
            remaining
        } else {
            let lethal_needed = if has_effective_keyword(state, attacker, Keywords::DEATHTOUCH) {
                1
            } else {
                let toughness = effective_toughness(state, blocker);
                let already = state.objects.get(blocker).damage as i32;
                (toughness - already).max(0)
            };
            remaining.min(lethal_needed)
        };
        if assign > 0 {
            events.push(ProposedEvent::damage(
                attacker,
                Target::Object(blocker),
                assign,
            ));
        }
        remaining -= assign;
    }
}

// ---------------------------------------------------------------- actions

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PendingCastActionStage {
    ChooseKicker,
    ChooseOptionalAdditionalCost,
    ChooseSpellMode,
    ChooseTarget,
    ChooseOptionalTarget,
    ChooseCastMode,
    ChooseCostTarget,
    ChooseCollectEvidenceCard,
    ChooseBargainPermanent,
    ChooseAdditionalCostZone,
    ChooseXValue,
    Discard,
    AwaitEngineAdvance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PendingActivationActionStage {
    ChooseTarget,
    ChooseOptionalTarget,
    Discard,
    AwaitEngineAdvance,
    /// Appended generic interactive permanent-cost stage.
    ChooseCostTarget,
}

/// Purely derives the one action family, if any, exposed by a staged cast.
/// `step` consults this before matching the requested action, so priority,
/// combat, and unrelated choice actions cannot overwrite a pending cast or
/// mutate around its 601.2 ordering.
fn pending_cast_action_stage(
    state: &GameState,
    pending: &PendingCast,
) -> Result<PendingCastActionStage, String> {
    validate_pending_cast(state, pending)?;
    if state.engine.pending_discard.is_some() {
        return Ok(PendingCastActionStage::Discard);
    }
    let def = &card_def::CARD_DEFS[state.objects.get(pending.spell).card_def as usize];
    if pending.kicked.is_none() {
        return Ok(PendingCastActionStage::ChooseKicker);
    }
    if pending.optional_additional_cost_paid.is_none() {
        return Ok(PendingCastActionStage::ChooseOptionalAdditionalCost);
    }
    if pending.mode_chosen.is_none() {
        return Ok(PendingCastActionStage::ChooseSpellMode);
    }
    let active_target_spec = spell_form_target_spec(
        def,
        pending
            .mode_chosen
            .expect("handled unresolved spell-form choice above"),
    )
    .ok_or("pending cast selected a nonexistent spell form")?;
    if !pending_cast_targeting_is_complete(pending, active_target_spec) {
        return Ok(
            if target_min_count(active_target_spec) < target_count(active_target_spec) {
                PendingCastActionStage::ChooseOptionalTarget
            } else {
                PendingCastActionStage::ChooseTarget
            },
        );
    }
    if pending.cast_mode.is_none() {
        return Ok(PendingCastActionStage::ChooseCastMode);
    }
    if pending.sacrifice_chosen.len() < usize::from(sacrifice_lands_needed(pending, def)) {
        return Ok(PendingCastActionStage::ChooseCostTarget);
    }
    if pending.sacrifice_chosen.len() < usize::from(graveyard_exile_cards_needed(pending, def)) {
        return Ok(PendingCastActionStage::ChooseCostTarget);
    }
    if pending.sacrifice_chosen.len()
        < controlled_permanent_sacrifice_needed(pending, def)
            .map_or(0, |(count, _)| usize::from(count))
    {
        return Ok(PendingCastActionStage::ChooseCostTarget);
    }
    if cast_tap_permanent_filter_needed(pending, def).is_some()
        && pending.sacrifice_chosen.is_empty()
    {
        return Ok(PendingCastActionStage::ChooseCostTarget);
    }
    if pending.optional_additional_cost_paid == Some(true) {
        match def
            .optional_additional_cost
            .ok_or("paid optional additional cost lost its definition")?
        {
            OptionalAdditionalCostDef::CollectEvidence { .. }
                if !pending.optional_additional_cost_selection_finished =>
            {
                return Ok(PendingCastActionStage::ChooseCollectEvidenceCard);
            }
            OptionalAdditionalCostDef::Bargain
                if pending.optional_additional_cost_chosen.is_empty() =>
            {
                return Ok(PendingCastActionStage::ChooseBargainPermanent);
            }
            _ => {}
        }
    }
    if has_chosen_creature_additional_cost(def) {
        if pending.chosen_creature_cost_zone.is_none() {
            return Ok(PendingCastActionStage::ChooseAdditionalCostZone);
        }
        if pending.chosen_creature_cost.is_none() {
            return Ok(PendingCastActionStage::ChooseCostTarget);
        }
    }
    if pending.x_value.is_none() {
        return Ok(PendingCastActionStage::ChooseXValue);
    }
    Ok(PendingCastActionStage::AwaitEngineAdvance)
}

fn action_matches_pending_cast_stage(action: &Action, stage: PendingCastActionStage) -> bool {
    matches!(
        (stage, action),
        (
            PendingCastActionStage::ChooseKicker,
            Action::ChooseKicker(_)
        ) | (
            PendingCastActionStage::ChooseOptionalAdditionalCost,
            Action::ChooseEffectOption(_)
        ) | (
            PendingCastActionStage::ChooseSpellMode,
            Action::ChooseSpellMode(_)
        ) | (
            PendingCastActionStage::ChooseTarget,
            Action::ChooseTarget(_)
        ) | (
            PendingCastActionStage::ChooseOptionalTarget,
            Action::ChooseTarget(_) | Action::FinishEffectSelection
        ) | (
            PendingCastActionStage::ChooseCastMode,
            Action::ChooseCastMode(_)
        ) | (
            PendingCastActionStage::ChooseCostTarget,
            Action::ChooseCostTarget(_)
        ) | (
            PendingCastActionStage::ChooseCollectEvidenceCard,
            Action::ChooseEffectTarget(_) | Action::FinishEffectSelection
        ) | (
            PendingCastActionStage::ChooseBargainPermanent,
            Action::ChooseCostTarget(_)
        ) | (
            PendingCastActionStage::ChooseAdditionalCostZone,
            Action::ChooseEffectOption(_)
        ) | (
            PendingCastActionStage::ChooseXValue,
            Action::ChooseEffectOption(_)
        ) | (PendingCastActionStage::Discard, Action::Discard(_))
    )
}

fn pending_activation_action_stage(
    state: &GameState,
    pending: &PendingActivation,
) -> Result<PendingActivationActionStage, String> {
    validate_pending_activation(state, pending)?;
    if state.engine.pending_discard.is_some() {
        return Ok(PendingActivationActionStage::Discard);
    }
    if pending.targets_chosen.len() < usize::from(target_count(pending.target_spec)) {
        if target_min_count(pending.target_spec) < target_count(pending.target_spec) {
            return Ok(PendingActivationActionStage::ChooseOptionalTarget);
        }
        return Ok(PendingActivationActionStage::ChooseTarget);
    }
    let def = &card_def::CARD_DEFS[state.objects.get(pending.source).card_def as usize];
    let ability = &def.activated_abilities[pending.ability_index as usize];
    let return_cost_incomplete =
        return_permanent_filter_in(ability.cost).is_some() && pending.object_cost_chosen.is_empty();
    let sacrifice_cost_incomplete = activation_permanent_sacrifice_needed(ability.cost)
        .is_some_and(|(needed, _)| pending.object_cost_chosen.len() < usize::from(needed));
    let tap_cost_incomplete = activation_tap_cost_subtype(ability.cost).is_some()
        && pending.object_cost_chosen.is_empty();
    let unblocked_attacker_return_cost_incomplete =
        has_unblocked_attacker_return_cost(ability.cost) && pending.object_cost_chosen.is_empty();
    if return_cost_incomplete
        || sacrifice_cost_incomplete
        || tap_cost_incomplete
        || unblocked_attacker_return_cost_incomplete
    {
        return Ok(PendingActivationActionStage::ChooseCostTarget);
    }
    Ok(PendingActivationActionStage::AwaitEngineAdvance)
}

fn action_matches_pending_activation_stage(
    action: &Action,
    stage: PendingActivationActionStage,
) -> bool {
    matches!(
        (stage, action),
        (
            PendingActivationActionStage::ChooseTarget,
            Action::ChooseTarget(_)
        ) | (
            PendingActivationActionStage::ChooseOptionalTarget,
            Action::ChooseEffectTarget(_) | Action::FinishEffectSelection
        ) | (PendingActivationActionStage::Discard, Action::Discard(_))
            | (
                PendingActivationActionStage::ChooseCostTarget,
                Action::ChooseCostTarget(_)
            )
    )
}

/// Applies `action` in response to the last `Decision` returned by
/// `advance_until_decision`. Returns `Err` for an action that isn't
/// currently legal (caller bug); never silently no-ops.
pub fn step(state: &mut GameState, action: Action) -> Result<(), String> {
    if state.engine.pending_land_play.is_some() && !matches!(&action, Action::ChooseEffectOption(_))
    {
        return Err(
            "only the exact color-choice action may answer a pending land play".to_string(),
        );
    }
    if let Some(pending) = state.engine.pending_cast.as_ref() {
        let stage = pending_cast_action_stage(state, pending)?;
        if !action_matches_pending_cast_stage(&action, stage) {
            return Err(format!(
                "action does not match pending cast stage {stage:?}"
            ));
        }
    }
    if let Some(pending) = state.engine.pending_activation.as_ref() {
        let stage = pending_activation_action_stage(state, pending)?;
        if !action_matches_pending_activation_stage(&action, stage) {
            return Err(format!(
                "action does not match pending activation stage {stage:?}"
            ));
        }
    }
    if let Some(pending_discard) = state.engine.pending_discard.as_ref() {
        validate_pending_discard_binding(state, pending_discard).map_err(|(_, message)| message)?;
        if !matches!(&action, Action::Discard(_)) {
            return Err("only the exact Discard action may answer a pending discard".to_string());
        }
    }
    if let Some(pending_trigger) = state.engine.pending_triggers.first() {
        if pending_trigger.placement_ordered
            && pending_trigger.targets.len()
                < usize::from(target_count(pending_trigger.target_spec))
        {
            validate_pending_trigger(state, pending_trigger)?;
            if !matches!(&action, Action::ChooseTarget(_)) {
                return Err(
                    "only ChooseTarget may answer a pending triggered ability target".to_string(),
                );
            }
        }
    }
    match action {
        Action::Pass => {
            let p = state.priority_player;
            state.engine.priority_passes[p.index()] = true;
            state.priority_player = p.opponent();
            Ok(())
        }
        Action::PlayLand(id) => {
            let p = state.priority_player;
            if !land_drop_candidates(p, state).contains(&id) {
                return Err(format!("{id} is not a legal land drop for {p:?}"));
            }
            let def = &card_def::CARD_DEFS[state.objects.get(id).card_def as usize];
            if let Some(excluded_color) = def.as_enters_choose_color_other_than {
                state.engine.pending_land_play = Some(PendingLandPlay {
                    source: id,
                    source_zone_change_count: state.objects.get(id).zone_change_count,
                    controller: p,
                    origin_zone: state.objects.get(id).zone,
                    excluded_color,
                });
            } else {
                play_land(state, p, id, None);
            }
            Ok(())
        }
        Action::ActivateManaAbility(id) => {
            let p = state.priority_player;
            let choices = available_mana_ability_choices(p, id, state);
            let [choice] = choices.as_slice() else {
                return Err(format!(
                    "{id} requires an explicit mana choice from {choices:?}"
                ));
            };
            activate_mana_ability_for(state, p, id, *choice, None)
        }
        Action::ActivateManaAbilityChoice(id, choice) => {
            let p = state.priority_player;
            let choices = available_mana_ability_choices(p, id, state);
            if choices.len() < 2 {
                return Err(format!(
                    "{id} is a single-color source and uses ActivateManaAbility"
                ));
            }
            activate_mana_ability_for(state, p, id, choice, None)
        }
        Action::ActivateManaAbilityWithCostTarget(id, choice, cost_target) => {
            let p = state.priority_player;
            activate_mana_ability_for(state, p, id, choice, Some(cost_target))
        }
        Action::ActivateAbility(source, index) => {
            let p = state.priority_player;
            if !available_activatable_abilities(p, state).contains(&(source, index)) {
                return Err(format!("ability {index} on {source} is not activatable by {p:?} right now"));
            }
            begin_activation(state, p, source, index);
            Ok(())
        }
        Action::CastSpell(id) => {
            let p = state.priority_player;
            if !castable_spells(p, state).contains(&id) {
                return Err(format!("{id} is not castable by {p:?} right now"));
            }
            begin_cast(state, p, id);
            Ok(())
        }
        Action::ChooseTarget(t) => {
            apply_choose_target(state, t)
        }
        Action::ChooseCostTarget(id) => apply_choose_cost_target(state, id),
        Action::ChooseCastMode(mode) => {
            let pending = state
                .engine
                .pending_cast
                .clone()
                .ok_or("no spell is currently being cast")?;
            validate_pending_cast(state, &pending)?;
            if pending.cast_mode.is_some() {
                return Err("this cast's mode has already been chosen".to_string());
            }
            let def = &card_def::CARD_DEFS[state.objects.get(pending.spell).card_def as usize];
            let active_target_spec = spell_form_target_spec(
                def,
                pending
                    .mode_chosen
                    .ok_or("cast mode chosen before the spell form completed")?,
            )
            .ok_or("pending cast selected a nonexistent spell form")?;
            if pending.kicked.is_none()
                || pending.targets_chosen.len()
                    != usize::from(target_count(active_target_spec))
                || pending.target_contracts.len() != pending.targets_chosen.len()
            {
                return Err("cast mode chosen before prior cast stages completed".to_string());
            }
            let payable = payable_cast_modes(def, &pending, state);
            if payable.len() < 2 || !payable.contains(&mode) {
                return Err(
                    "a cast-mode action requires a currently offered payable mode".to_string(),
                );
            }
            state
                .engine
                .pending_cast
                .as_mut()
                .expect("validated cast remains pending")
                .cast_mode = Some(mode);
            Ok(())
        }
        Action::ChooseKicker(kicked) => {
            let pending = state
                .engine
                .pending_cast
                .clone()
                .ok_or("no spell is currently being cast")?;
            validate_pending_cast(state, &pending)?;
            if pending.kicked.is_some() {
                return Err("this cast's kicker has already been chosen".to_string());
            }
            let def = &card_def::CARD_DEFS[state.objects.get(pending.spell).card_def as usize];
            let kicker = def
                .kicker_cost
                .ok_or("this spell has no kicker choice")?;
            let normal = effective_normal_cast_cost(def, pending.controller, state);
            if mana::can_pay_combined(&[&normal, &kicker], 0, pending.controller, state).is_none() {
                return Err("the kicker choice is no longer payable".to_string());
            }
            state
                .engine
                .pending_cast
                .as_mut()
                .expect("validated cast remains pending")
                .kicked = Some(kicked);
            Ok(())
        }
        Action::ChooseSpellMode(mode) => {
            let p = state.priority_player;
            let pending = state
                .engine
                .pending_cast
                .as_ref()
                .ok_or("no spell is currently being cast")?;
            validate_pending_cast(state, pending)?;
            if pending.kicked.is_none() {
                return Err("spell mode chosen before the kicker stage completed".to_string());
            }
            if pending.mode_chosen.is_some() {
                return Err("this cast's mode has already been chosen".to_string());
            }
            let def = &card_def::CARD_DEFS[state.objects.get(pending.spell).card_def as usize];
            let viable_modes = viable_pending_spell_forms(def, pending, state);
            if viable_modes.len() < 2 || !viable_modes.contains(&mode) {
                return Err(format!("{mode} is not a legal spell mode for {p:?}'s cast"));
            }
            state.engine.pending_cast.as_mut().unwrap().mode_chosen = Some(mode);
            Ok(())
        }
        Action::ChooseEffectOption(option_index) => {
            if state.engine.pending_land_play.is_some() {
                apply_choose_land_color(state, option_index)
            } else if state.engine.pending_cast.is_some() {
                apply_pending_cast_effect_option(state, option_index)
            } else {
                effect::choose_resumable_option(state, option_index)
            }
        }
        Action::ChooseEffectTarget(target) => {
            if state.engine.pending_cast.is_some() {
                apply_choose_collect_evidence_target(state, target)
            } else if state.engine.pending_activation.is_some() {
                apply_choose_optional_activation_target(state, target)
            } else {
                effect::choose_resumable_target(state, target)
            }
        }
        Action::FinishEffectSelection => {
            if state.engine.pending_cast.is_some() {
                finish_optional_cast_or_collect_evidence(state)
            } else if state.engine.pending_activation.is_some() {
                finish_optional_activation_targets(state)
            } else {
                effect::finish_resumable_target_selection(state)
            }
        }
        Action::ChooseEffectBoolean(value) => effect::choose_resumable_boolean(state, value),
        Action::ChooseOptionalCost(choice) => apply_choose_optional_cost(state, choice),
        Action::ChooseSpellCopyPayment(pay) => apply_choose_spell_copy_payment(state, pay),
        Action::ChooseSpellCopyRetarget(change_target) => {
            apply_choose_spell_copy_retarget(state, change_target)
        }
        Action::ChooseMadnessCast(cast_it) => apply_choose_madness_cast(state, cast_it),
        Action::ChooseOptionalCostStage(_) => {
            Err("ChooseOptionalCostStage is presentation-only (HarnessSurfaceV2's reshape); it must never reach step() directly".to_string())
        }
        Action::PlotSpell(id) => {
            let p = state.priority_player;
            if !plot_action_candidates(p, state).contains(&id) {
                return Err(format!("{id} is not a legal Plot action for {p:?}"));
            }
            plot_spell(state, p, id);
            Ok(())
        }
        Action::Discard(chosen) => apply_discard_action(state, chosen),
        Action::DeclareAttackers(attackers) => apply_declare_attackers(state, attackers),
        Action::DeclareBlockers(blocks) => apply_declare_blockers(state, blocks),
        Action::OrderTriggers(perm) => apply_order_triggers(state, perm),
    }
}

#[derive(Debug, Clone)]
enum TargetingProducer {
    Cast {
        pending: PendingCast,
        spec: TargetSpec,
    },
    Activation(PendingActivation),
    SpellCopy(PendingSpellCopy),
    Trigger(PendingTrigger),
}

fn exact_targeting_producer(state: &GameState) -> Result<TargetingProducer, String> {
    let mut producers = Vec::new();
    if let Some(pending) = state.engine.pending_cast.as_ref() {
        validate_pending_cast(state, pending)?;
        if pending.kicked.is_some() {
            if let Some(mode) = pending.mode_chosen {
                let def = &card_def::CARD_DEFS[state.objects.get(pending.spell).card_def as usize];
                let spec = spell_form_target_spec(def, mode)
                    .ok_or("pending cast selected a nonexistent spell form")?;
                if !pending_cast_targeting_is_complete(pending, spec) {
                    producers.push(TargetingProducer::Cast {
                        pending: pending.clone(),
                        spec,
                    });
                }
            }
        }
    }
    if let Some(pending) = state.engine.pending_activation.as_ref() {
        validate_pending_activation(state, pending)?;
        if target_min_count(pending.target_spec) == target_count(pending.target_spec)
            && pending.targets_chosen.len() < usize::from(target_count(pending.target_spec))
        {
            producers.push(TargetingProducer::Activation(pending.clone()));
        }
    }
    if let Some(pending) = state.engine.pending_spell_copy.as_ref() {
        if pending.stage == SpellCopyStage::Target {
            validate_pending_spell_copy(state, pending)?;
            producers.push(TargetingProducer::SpellCopy(pending.clone()));
        }
    }
    if let Some(pending) = state.engine.pending_triggers.first() {
        if pending.placement_ordered
            && pending.targets.len() < usize::from(target_count(pending.target_spec))
        {
            validate_pending_trigger(state, pending)?;
            producers.push(TargetingProducer::Trigger(pending.clone()));
        }
    }
    match producers.len() {
        1 => Ok(producers.pop().expect("length checked above")),
        0 => Err("no targeting context is currently pending".to_string()),
        _ => Err("multiple targeting contexts are simultaneously pending".to_string()),
    }
}

fn apply_choose_target(state: &mut GameState, target: Target) -> Result<(), String> {
    match exact_targeting_producer(state)? {
        TargetingProducer::Cast { pending, spec } => {
            if !completable_next_targets_for_controller_and_source(
                spec,
                &pending.targets_chosen,
                pending.controller,
                targeting_source_for_object(state, pending.spell),
                state,
            )
            .contains(&target)
            {
                return Err(format!("{target:?} is not a legal cast target"));
            }
            let contract = StackTargetContractV4::capture(state, target);
            let live = state
                .engine
                .pending_cast
                .as_mut()
                .ok_or("validated cast targeting context disappeared")?;
            if live != &pending {
                return Err("cast targeting context changed before commit".to_string());
            }
            live.targets_chosen.push(target);
            live.target_contracts.push(contract);
            Ok(())
        }
        TargetingProducer::Activation(pending) => {
            let definition =
                &card_def::CARD_DEFS[state.objects.get(pending.source).card_def as usize];
            let ability = &definition.activated_abilities[pending.ability_index as usize];
            if !completable_next_activation_targets_for(
                pending.source,
                ability,
                &pending.targets_chosen,
                state,
            )
            .contains(&target)
            {
                return Err(format!("{target:?} is not a legal activation target"));
            }
            let contract = StackTargetContractV4::capture(state, target);
            let live = state
                .engine
                .pending_activation
                .as_mut()
                .ok_or("validated activation targeting context disappeared")?;
            if live != &pending {
                return Err("activation targeting context changed before commit".to_string());
            }
            live.targets_chosen.push(target);
            live.target_contracts.push(contract);
            Ok(())
        }
        TargetingProducer::SpellCopy(pending) => {
            if state.engine.pending_spell_copy.as_ref() != Some(&pending) {
                return Err("spell-copy targeting context changed before commit".to_string());
            }
            apply_choose_spell_copy_target(state, target)
        }
        TargetingProducer::Trigger(pending) => {
            if !completable_next_targets_for_controller_and_source(
                pending.target_spec,
                &pending.targets,
                pending.controller,
                pending.source_contract.map(|contract| TargetingSource {
                    object: pending.source,
                    card_def: contract.card_def,
                }),
                state,
            )
            .contains(&target)
            {
                return Err(format!(
                    "{target:?} is not a legal triggered-ability target"
                ));
            }
            let contract = StackTargetContractV4::capture(state, target);
            let live = state
                .engine
                .pending_triggers
                .first_mut()
                .ok_or("validated trigger targeting context disappeared")?;
            if live != &pending {
                return Err("trigger targeting context changed before commit".to_string());
            }
            live.targets.push(target);
            live.target_contracts.push(contract);
            Ok(())
        }
    }
}

fn apply_choose_optional_additional_cost(
    state: &mut GameState,
    option_index: u16,
) -> Result<(), String> {
    let pending = state
        .engine
        .pending_cast
        .clone()
        .ok_or("no cast is choosing an optional additional cost")?;
    validate_pending_cast(state, &pending)?;
    if pending.kicked.is_none() || pending.optional_additional_cost_paid.is_some() {
        return Err("optional additional cost is not the active cast stage".to_string());
    }
    let def = &card_def::CARD_DEFS[state.objects.get(pending.spell).card_def as usize];
    let kind = def
        .optional_additional_cost
        .ok_or("this spell has no optional additional cost")?;
    if !optional_additional_cost_is_payable(kind, pending.controller, state)? {
        return Err("optional additional cost is no longer payable".to_string());
    }
    let paid = match option_index {
        0 => false,
        1 => true,
        _ => return Err("optional additional-cost option index is out of range".to_string()),
    };
    state
        .engine
        .pending_cast
        .as_mut()
        .expect("validated cast remains pending")
        .optional_additional_cost_paid = Some(paid);
    Ok(())
}

fn apply_choose_collect_evidence_target(
    state: &mut GameState,
    target: Target,
) -> Result<(), String> {
    let pending = state
        .engine
        .pending_cast
        .clone()
        .ok_or("no cast is selecting Collect Evidence cards")?;
    validate_pending_cast(state, &pending)?;
    if pending_cast_action_stage(state, &pending)?
        != PendingCastActionStage::ChooseCollectEvidenceCard
    {
        return Err("Collect Evidence is not the active cast stage".to_string());
    }
    let Target::Object(object_id) = target else {
        return Err("Collect Evidence requires a graveyard card".to_string());
    };
    let candidates = collect_evidence_candidates(pending.controller, state)?;
    let binding = candidates
        .into_iter()
        .find(|binding| {
            binding.object == object_id
                && !pending.optional_additional_cost_chosen.contains(binding)
        })
        .ok_or_else(|| format!("{object_id} is not a legal Collect Evidence card"))?;
    state
        .engine
        .pending_cast
        .as_mut()
        .expect("validated cast remains pending")
        .optional_additional_cost_chosen
        .push(binding);
    Ok(())
}

fn finish_optional_cast_or_collect_evidence(state: &mut GameState) -> Result<(), String> {
    let pending = state
        .engine
        .pending_cast
        .clone()
        .ok_or("no cast has a finishable selection")?;
    validate_pending_cast(state, &pending)?;
    if pending_cast_action_stage(state, &pending)?
        == PendingCastActionStage::ChooseCollectEvidenceCard
    {
        let def = &card_def::CARD_DEFS[state.objects.get(pending.spell).card_def as usize];
        let Some(OptionalAdditionalCostDef::CollectEvidence { minimum_mana_value }) =
            def.optional_additional_cost
        else {
            return Err("Collect Evidence finish lost its definition".to_string());
        };
        let selected_sum =
            collect_evidence_mana_value(state, &pending.optional_additional_cost_chosen)?;
        if selected_sum < u32::from(minimum_mana_value) {
            return Err("Collect Evidence cannot finish below its mana-value minimum".to_string());
        }
        state
            .engine
            .pending_cast
            .as_mut()
            .expect("validated cast remains pending")
            .optional_additional_cost_selection_finished = true;
        return Ok(());
    }
    finish_optional_cast_targets(state)
}

fn finish_optional_cast_targets(state: &mut GameState) -> Result<(), String> {
    let pending = state
        .engine
        .pending_cast
        .clone()
        .ok_or("no cast is selecting optional targets")?;
    validate_pending_cast(state, &pending)?;
    let def = &card_def::CARD_DEFS[state.objects.get(pending.spell).card_def as usize];
    let spec = spell_form_target_spec(
        def,
        pending
            .mode_chosen
            .ok_or("cast target selection finished before choosing a spell form")?,
    )
    .ok_or("pending cast selected a nonexistent spell form")?;
    let min = usize::from(target_min_count(spec));
    let max = usize::from(target_count(spec));
    if min >= max
        || pending.target_selection_finished
        || pending.targets_chosen.len() < min
        || pending.targets_chosen.len() >= max
    {
        return Err("the pending cast target selection cannot finish".to_string());
    }
    state
        .engine
        .pending_cast
        .as_mut()
        .expect("validated optional cast targeting remains staged")
        .target_selection_finished = true;
    Ok(())
}

fn apply_choose_optional_activation_target(
    state: &mut GameState,
    target: Target,
) -> Result<(), String> {
    let pending = state
        .engine
        .pending_activation
        .clone()
        .ok_or("no activation is selecting optional targets")?;
    validate_pending_activation(state, &pending)?;
    let min = target_min_count(pending.target_spec);
    let max = target_count(pending.target_spec);
    if min >= max || pending.targets_chosen.len() >= usize::from(max) {
        return Err("the pending activation has no open optional target slot".to_string());
    }
    if pending.cost_discard_paid.as_deref() != Some(&[]) {
        return Err(
            "optional activation targets cannot precede an interactive discard".to_string(),
        );
    }
    let legal = completable_next_targets_for(pending.target_spec, &pending.targets_chosen, state);
    if !legal.contains(&target) {
        return Err(format!(
            "{target:?} is not a legal optional activation target"
        ));
    }
    let contract = StackTargetContractV4::capture(state, target);
    let live = state
        .engine
        .pending_activation
        .as_mut()
        .ok_or("optional activation targeting context disappeared")?;
    if live != &pending {
        return Err("optional activation targeting context changed before commit".to_string());
    }
    live.targets_chosen.push(target);
    live.target_contracts.push(contract);
    Ok(())
}

fn finish_optional_activation_targets(state: &mut GameState) -> Result<(), String> {
    let pending = state
        .engine
        .pending_activation
        .clone()
        .ok_or("no activation is selecting optional targets")?;
    validate_pending_activation(state, &pending)?;
    if target_min_count(pending.target_spec) >= target_count(pending.target_spec)
        || !target_cardinality_is_complete(pending.target_spec, pending.targets_chosen.len())
    {
        return Err("the pending activation target selection cannot finish".to_string());
    }
    if pending.cost_discard_paid.as_deref() != Some(&[]) {
        return Err(
            "optional activation targets cannot finish before an interactive discard".to_string(),
        );
    }
    finalize_activation(state);
    if state.engine.halted.is_some() {
        return Err("optional activation failed validation during final payment".to_string());
    }
    Ok(())
}

/// Answers one `Decision::ChooseCostTargets` pick -- see that decision's
/// doc. `PendingCast` stages permanent sacrifices and Escape graveyard
/// exiles; `PendingOptionalCostSacrifice` stages Highway Robbery's
/// `SacrificeLand` choice. Activated abilities use the same stable action
/// family for non-target object costs such as Heap Gate tapping another
/// untapped controlled Gate.
fn apply_choose_cost_target(state: &mut GameState, id: ObjectId) -> Result<(), String> {
    if let Some(pending) = state.engine.pending_cast.clone() {
        validate_pending_cast(state, &pending)?;
        let def = &card_def::CARD_DEFS[state.objects.get(pending.spell).card_def as usize];
        let needed = sacrifice_lands_needed(&pending, def);
        if (pending.sacrifice_chosen.len() as u8) < needed {
            let active_target_spec = spell_form_target_spec(
                def,
                pending
                    .mode_chosen
                    .ok_or("cast sacrifice target chosen before spell form completed")?,
            )
            .ok_or("pending cast selected a nonexistent spell form")?;
            if pending.kicked.is_none()
                || !pending_cast_targeting_is_complete(&pending, active_target_spec)
                || pending.cast_mode.is_none()
            {
                return Err(
                    "cast sacrifice target chosen before prior cast stages completed".to_string(),
                );
            }
            let candidates =
                sacrificeable_lands(pending.controller, state, &pending.sacrifice_chosen);
            let remaining = needed - pending.sacrifice_chosen.len() as u8;
            if candidates.len() < usize::from(remaining) {
                return Err("cast sacrifice cost can no longer be completed".to_string());
            }
            if !candidates.contains(&id) {
                return Err(format!("{id} is not a legal cost-sacrifice candidate"));
            }
            state
                .engine
                .pending_cast
                .as_mut()
                .unwrap()
                .sacrifice_chosen
                .push(id);
            return Ok(());
        }
        let graveyard_exile_needed = graveyard_exile_cards_needed(&pending, def);
        if (pending.sacrifice_chosen.len() as u8) < graveyard_exile_needed {
            let active_target_spec = spell_form_target_spec(
                def,
                pending
                    .mode_chosen
                    .ok_or("cast graveyard-exile target chosen before spell form completed")?,
            )
            .ok_or("pending cast selected a nonexistent spell form")?;
            if pending.kicked.is_none()
                || !pending_cast_targeting_is_complete(&pending, active_target_spec)
                || pending.cast_mode.is_none()
            {
                return Err(
                    "cast graveyard-exile target chosen before prior cast stages completed"
                        .to_string(),
                );
            }
            let candidates = checked_graveyard_exile_candidates(
                pending.controller,
                pending.spell,
                state,
                &pending.sacrifice_chosen,
            )
            .ok_or("escape graveyard index or chosen prefix is malformed")?;
            let remaining = graveyard_exile_needed - pending.sacrifice_chosen.len() as u8;
            if candidates.len() < usize::from(remaining) {
                return Err("cast graveyard-exile cost can no longer be completed".to_string());
            }
            if !candidates.contains(&id) {
                return Err(format!(
                    "{id} is not a legal graveyard-exile cost candidate"
                ));
            }
            state
                .engine
                .pending_cast
                .as_mut()
                .unwrap()
                .sacrifice_chosen
                .push(id);
            return Ok(());
        }
        if let Some((needed, filter)) = controlled_permanent_sacrifice_needed(&pending, def) {
            if (pending.sacrifice_chosen.len() as u8) < needed {
                let active_target_spec = spell_form_target_spec(
                    def,
                    pending.mode_chosen.ok_or(
                        "cast permanent-sacrifice target chosen before spell form completed",
                    )?,
                )
                .ok_or("pending cast selected a nonexistent spell form")?;
                if pending.kicked.is_none()
                    || !pending_cast_targeting_is_complete(&pending, active_target_spec)
                    || pending.cast_mode.is_none()
                {
                    return Err(
                        "cast permanent-sacrifice target chosen before prior cast stages completed"
                            .to_string(),
                    );
                }
                let candidates = sacrificeable_controlled_permanents(
                    pending.controller,
                    filter,
                    state,
                    &pending.sacrifice_chosen,
                );
                let remaining = needed - pending.sacrifice_chosen.len() as u8;
                if candidates.len() < usize::from(remaining) {
                    return Err(
                        "cast permanent-sacrifice cost can no longer be completed".to_string()
                    );
                }
                if !candidates.contains(&id) {
                    return Err(format!(
                        "{id} is not a legal permanent-sacrifice cost candidate"
                    ));
                }
                state
                    .engine
                    .pending_cast
                    .as_mut()
                    .unwrap()
                    .sacrifice_chosen
                    .push(id);
                return Ok(());
            }
        }
        if let Some(filter) = cast_tap_permanent_filter_needed(&pending, def) {
            if pending.sacrifice_chosen.is_empty() {
                let active_target_spec = spell_form_target_spec(
                    def,
                    pending
                        .mode_chosen
                        .ok_or("cast tap-cost target chosen before spell form completed")?,
                )
                .ok_or("pending cast selected a nonexistent spell form")?;
                if pending.kicked.is_none()
                    || !pending_cast_targeting_is_complete(&pending, active_target_spec)
                    || pending.cast_mode.is_none()
                {
                    return Err(
                        "cast tap-cost target chosen before prior cast stages completed"
                            .to_string(),
                    );
                }
                let candidates =
                    tap_permanent_cost_candidates(pending.controller, state, filter, &[]);
                if !candidates.contains(&id) {
                    return Err(format!("{id} is not a legal tap-cost candidate"));
                }
                state
                    .engine
                    .pending_cast
                    .as_mut()
                    .unwrap()
                    .sacrifice_chosen
                    .push(id);
                return Ok(());
            }
        }
        if pending.optional_additional_cost_paid == Some(true)
            && def.optional_additional_cost == Some(OptionalAdditionalCostDef::Bargain)
            && pending.optional_additional_cost_chosen.is_empty()
        {
            if pending_cast_action_stage(state, &pending)?
                != PendingCastActionStage::ChooseBargainPermanent
            {
                return Err("Bargain is not the active cast stage".to_string());
            }
            let binding = bargain_candidates(pending.controller, state)?
                .into_iter()
                .find(|binding| binding.object == id)
                .ok_or_else(|| format!("{id} is not a legal Bargain permanent"))?;
            state
                .engine
                .pending_cast
                .as_mut()
                .expect("validated cast remains pending")
                .optional_additional_cost_chosen
                .push(binding);
            return Ok(());
        }
        if has_chosen_creature_additional_cost(def) && pending.chosen_creature_cost.is_none() {
            let zone = pending
                .chosen_creature_cost_zone
                .ok_or("chosen-creature cost object selected before its branch")?;
            let candidates = chosen_creature_cost_candidates(pending.controller, zone, state);
            if candidates.len() < 2 {
                return Err(
                    "chosen-creature cost object choice is not currently offered".to_string(),
                );
            }
            let binding = candidates
                .into_iter()
                .find(|binding| binding.object == id)
                .ok_or_else(|| format!("{id} is not a legal chosen-creature cost candidate"))?;
            state
                .engine
                .pending_cast
                .as_mut()
                .expect("validated cast remains pending")
                .chosen_creature_cost = Some(binding);
            return Ok(());
        }
    }
    if let Some(pending) = state.engine.pending_activation.clone() {
        validate_pending_activation(state, &pending)?;
        let def = &card_def::CARD_DEFS[state.objects.get(pending.source).card_def as usize];
        let ability = &def.activated_abilities[pending.ability_index as usize];
        if !target_cardinality_is_complete(pending.target_spec, pending.targets_chosen.len()) {
            return Err(
                "activation cost target chosen before activation targeting completed".to_string(),
            );
        }

        if let Some(filter) = return_permanent_filter_in(ability.cost) {
            if !pending.object_cost_chosen.is_empty() {
                return Err("activation object cost has already been selected".to_string());
            }
            let candidates = return_permanent_cost_candidates(
                pending.controller,
                state,
                filter,
                &pending.object_cost_chosen,
            );
            let binding = candidates
                .into_iter()
                .find(|binding| binding.object == id)
                .ok_or_else(|| format!("{id} is not a legal activation object-cost candidate"))?;
            state
                .engine
                .pending_activation
                .as_mut()
                .expect("validated activation remains staged")
                .object_cost_chosen
                .push(binding);
            return Ok(());
        } else if let Some((needed, filter)) = activation_permanent_sacrifice_needed(ability.cost) {
            if pending.object_cost_chosen.len() < usize::from(needed) {
                let candidates = activation_sacrifice_cost_candidates(
                    pending.controller,
                    filter,
                    state,
                    &pending.object_cost_chosen,
                );
                let remaining = needed - pending.object_cost_chosen.len() as u8;
                if candidates.len() < usize::from(remaining) {
                    return Err(
                        "activation permanent-sacrifice cost can no longer complete".to_string()
                    );
                }
                let binding = candidates
                    .into_iter()
                    .find(|binding| binding.object == id)
                    .ok_or_else(|| format!("{id} is not a legal activation sacrifice candidate"))?;
                state
                    .engine
                    .pending_activation
                    .as_mut()
                    .expect("validated activation remains staged")
                    .object_cost_chosen
                    .push(binding);
                return Ok(());
            }
        }
        if activation_tap_cost_subtype(ability.cost).is_some() {
            if !payable_activation_cost_object_candidates(
                pending.controller,
                pending.source,
                ability.cost,
                state,
            )
            .contains(&id)
            {
                return Err(format!("{id} is not a legal activation-cost candidate"));
            }
            let object = state.objects.get(id);
            let binding = EffectObjectBinding {
                object: id,
                expected_zone: Zone::Battlefield,
                expected_zone_change_count: object.zone_change_count,
            };
            let live = state
                .engine
                .pending_activation
                .as_mut()
                .expect("validated activation remains staged");
            if live != &pending {
                return Err("pending activation changed before object-cost commit".to_string());
            }
            live.object_cost_chosen.push(binding);
            return Ok(());
        }
        if has_unblocked_attacker_return_cost(ability.cost) {
            let binding = unblocked_attacker_return_candidates(
                pending.controller,
                state,
                &pending.object_cost_chosen,
            )
            .into_iter()
            .find(|binding| binding.object == id)
            .ok_or_else(|| format!("{id} is not a legal unblocked-attacker return candidate"))?;
            let live = state
                .engine
                .pending_activation
                .as_mut()
                .expect("validated activation remains staged");
            if live != &pending {
                return Err("pending activation changed before object-cost commit".to_string());
            }
            live.object_cost_chosen.push(binding);
            return Ok(());
        }
        return Err("pending activation has no interactive object cost".to_string());
    }
    if let Some(pending) = state.engine.pending_optional_cost_sacrifice.as_ref() {
        if (pending.chosen.len() as u8) < pending.remaining {
            let candidates = sacrificeable_lands(pending.player, state, &pending.chosen);
            let remaining = pending.remaining - pending.chosen.len() as u8;
            if candidates.len() < usize::from(remaining) {
                return Err("optional sacrifice cost can no longer be completed".to_string());
            }
            if !candidates.contains(&id) {
                return Err(format!("{id} is not a legal cost-sacrifice candidate"));
            }
            state
                .engine
                .pending_optional_cost_sacrifice
                .as_mut()
                .unwrap()
                .chosen
                .push(id);
            return Ok(());
        }
    }
    Err("no cost-target decision is pending".to_string())
}

fn apply_discard_action(state: &mut GameState, chosen: Vec<ObjectId>) -> Result<(), String> {
    let pd = state
        .engine
        .pending_discard
        .clone()
        .ok_or("no discard is pending")?;
    validate_pending_discard_binding(state, &pd).map_err(|(_, message)| message)?;
    if chosen.len() as u32 != pd.count {
        return Err(format!(
            "must discard exactly {} card(s), got {}",
            pd.count,
            chosen.len()
        ));
    }
    let mut dedup = chosen.clone();
    dedup.sort_unstable();
    dedup.dedup();
    if dedup.len() != chosen.len() {
        return Err("duplicate card in discard selection".to_string());
    }
    let exclude = match &pd.resume {
        DiscardResume::FinishCast { .. } => state.engine.pending_cast.as_ref().map(|p| p.spell),
        DiscardResume::FinishActivation { source, .. } => Some(*source),
        _ => None,
    };
    let hand = &state.players[pd.player.index()].hand;
    if !chosen
        .iter()
        .all(|id| hand.contains(id) && Some(*id) != exclude)
    {
        return Err("illegal discard selection".to_string());
    }
    apply_discard(state, chosen, pd);
    Ok(())
}

fn apply_choose_optional_cost(
    state: &mut GameState,
    choice: OptionalCostChoice,
) -> Result<(), String> {
    let poc = state
        .engine
        .pending_optional_cost
        .take()
        .ok_or("no optional cost is pending")?;
    match choice {
        OptionalCostChoice::Decline => {
            // See `PendingOptionalCost::spell_resume`'s doc: declining
            // still means the spell this cost belongs to is now fully
            // resolved (there's no `then` to run either way), so its
            // deferred move happens right here.
            if let Some((spell, to_zone)) = poc.spell_resume {
                event::propose_and_commit(state, ProposedEvent::zone_change(spell, to_zone));
            }
            Ok(())
        }
        OptionalCostChoice::Discard => {
            if !poc.discard_payable {
                return Err(
                    "discard is not currently a payable option for this optional cost".to_string(),
                );
            }
            state.engine.pending_discard = Some(PendingDiscard {
                player: poc.player,
                count: poc.discard as u32,
                resume: DiscardResume::FinishOptionalCost {
                    source: poc.source,
                    controller: poc.player,
                    then: Box::new(poc.then),
                    spell_resume: poc.spell_resume,
                },
            });
            Ok(())
        }
        OptionalCostChoice::SacrificeLand => {
            if !poc.sacrifice_payable {
                return Err(
                    "sacrificing a land is not currently a payable option for this optional cost"
                        .to_string(),
                );
            }
            state.engine.pending_optional_cost_sacrifice = Some(PendingOptionalCostSacrifice {
                player: poc.player,
                source: poc.source,
                remaining: poc.sacrifice_lands,
                chosen: vec![],
                then: poc.then,
                spell_resume: poc.spell_resume,
            });
            Ok(())
        }
    }
}

/// Allocates a distinct virtual card identity and stack item for the next
/// Chain Lightning link. It clones copiable stack choices but is not a cast:
/// no mana cost, cast event, or cast trigger is produced.
fn expected_spell_copy_source_object(
    original: &crate::state::GameObject,
    parent: &StackItem,
    pending: &PendingSpellCopy,
) -> crate::state::GameObject {
    crate::state::GameObject {
        card_def: original.card_def,
        name: original.name.clone(),
        owner: pending.player,
        controller: pending.player,
        zone: Zone::Stack,
        tapped: false,
        summoning_sick: false,
        damage: 0,
        counters: Default::default(),
        attachments: Vec::new(),
        v4: ObjectStateV4::from_card_def(original.card_def),
        spell_copy_origin: Some(SpellCopyOriginV4 {
            parent: parent.source,
            parent_card_def: original.card_def,
            parent_owner: original.owner,
            parent_controller: parent.controller,
            parent_stack_zone_change_count: original.zone_change_count,
            parent_was_copy: original.spell_copy_origin.is_some(),
        }),
        plotted_turn: None,
        zone_change_count: 0,
    }
}

fn expected_spell_copy(
    parent: &StackItem,
    pending: &PendingSpellCopy,
    copy_source: ObjectId,
    copy_stack_item: StackItemId,
    source_contract: StackSourceContractV4,
) -> StackItem {
    let mut copy = parent.clone();
    copy.source = copy_source;
    copy.controller = pending.player;
    copy.targets = vec![pending.inherited_target];
    copy.v4.target_contracts = vec![pending
        .inherited_target_contract
        .expect("validated copy continuation retains the inherited target contract")];
    copy.v4.stack_item_id = copy_stack_item;
    copy.v4.source_contract = Some(source_contract);
    copy.is_copy = true;
    copy.is_flashback = false;
    copy.madness_offer = false;
    copy
}

fn create_spell_copy(state: &mut GameState, pending: &PendingSpellCopy) -> (ObjectId, StackItemId) {
    let (_, parent) = pending_spell_copy_parent(state, pending)
        .expect("validated immediately before paying; no action can interleave");
    let original = state.objects.get(parent.source);
    let copy_source = state.objects.push(expected_spell_copy_source_object(
        original, &parent, pending,
    ));
    let copy_stack_item = next_stack_item_id(state);
    let source_contract = StackSourceContractV4::capture(state, copy_source, CastMethodV4::Normal);
    let copy = expected_spell_copy(
        &parent,
        pending,
        copy_source,
        copy_stack_item,
        source_contract,
    );
    state.stack.push(copy);
    (copy_source, copy_stack_item)
}

/// Materializes an already-authenticated Storm trigger as independent
/// virtual copies of Weather the Storm. This is transactional: either every
/// copy receives valid source/stack ancestry, or the caller's state is left
/// byte-for-byte unchanged and resolution fails closed.
pub(crate) fn create_storm_spell_copies(
    state: &mut GameState,
    ctx: &ExecCtx,
    binding: StormCopyBindingV1,
) -> Result<(), String> {
    if ctx.source != binding.source_contract.source
        || ctx.controller != binding.source_contract.controller
        || ctx
            .stack_item_id
            .is_none_or(|id| id == StackItemId::default() || id == binding.producing_stack_item)
        || !ctx.targets.is_empty()
        || !ctx.target_contracts.is_empty()
        || !ctx.discarded.is_empty()
        || !ctx.paid_cost_refs.is_empty()
        || ctx.hidden_ability_source.is_some()
        || ctx.ability_source_contract.is_some()
        || ctx.kicked
    {
        return Err("Storm resolution context changed from its bound trigger".to_string());
    }
    let copy_count = storm_copy_count_from_binding(state, binding)?;
    let parent = state.objects.get(binding.source_contract.source);
    let parent_name = parent.name.clone();
    let mut staged = state.clone();
    for _ in 0..copy_count {
        let copy_source = staged.objects.push(crate::state::GameObject {
            card_def: binding.source_contract.card_def,
            name: parent_name.clone(),
            owner: binding.source_contract.owner,
            controller: binding.source_contract.controller,
            zone: Zone::Stack,
            tapped: false,
            summoning_sick: false,
            damage: 0,
            counters: Default::default(),
            attachments: Vec::new(),
            v4: ObjectStateV4::from_card_def(binding.source_contract.card_def),
            spell_copy_origin: Some(SpellCopyOriginV4 {
                parent: binding.source_contract.source,
                parent_card_def: binding.source_contract.card_def,
                parent_owner: binding.source_contract.owner,
                parent_controller: binding.source_contract.controller,
                parent_stack_zone_change_count: binding.source_contract.zone_change_count,
                parent_was_copy: false,
            }),
            plotted_turn: None,
            zone_change_count: 0,
        });
        let stack_item_id = next_stack_item_id(&mut staged);
        let source_contract =
            StackSourceContractV4::capture(&staged, copy_source, CastMethodV4::Normal);
        let copy = StackItem {
            kind: StackItemKind::Spell,
            source: copy_source,
            controller: binding.source_contract.controller,
            targets: Vec::new(),
            is_copy: true,
            inline_effect: None,
            discarded: Vec::new(),
            is_flashback: false,
            mode_chosen: 0,
            madness_offer: false,
            kicked: false,
            v4: StackStateV4 {
                stack_item_id,
                cast_method: Some(CastMethodV4::Normal),
                source_contract: Some(source_contract),
                target_spec: Some(TargetSpec::None),
                ..StackStateV4::default()
            },
        };
        staged.stack.push(copy.clone());
        validate_spell_stack_source(&staged, &copy)?;
        validated_stack_item_target_spec(&copy, &staged)?;
    }
    *state = staged;
    Ok(())
}

fn pending_spell_copy_parent(
    state: &GameState,
    pending: &PendingSpellCopy,
) -> Result<(usize, StackItem), String> {
    let mut matches = state.stack.iter().enumerate().filter(|(_, item)| {
        item.v4.stack_item_id == pending.resolving_stack_item
            && item.source == pending.resolving_source
    });
    let (parent_index, parent) = matches
        .next()
        .ok_or("the resolving spell is no longer on the stack")?;
    if pending.resolving_stack_item == StackItemId::default() {
        return Err("the resolving spell lacks a stack-incarnation id".to_string());
    }
    if matches.next().is_some() {
        return Err("the resolving spell has duplicate stack membership".to_string());
    }
    if parent.kind != StackItemKind::Spell {
        return Err("only a spell can be copied by this resolution".to_string());
    }
    validate_spell_stack_source(state, parent)?;
    let source = state
        .objects
        .try_get(parent.source)
        .ok_or("the resolving spell source object no longer exists")?;
    if source.zone != Zone::Stack {
        return Err("the resolving spell source object is not in the stack zone".to_string());
    }
    if source.zone_change_count != pending.resolving_source_zone_change_count {
        return Err("the resolving spell source incarnation changed".to_string());
    }
    Ok((parent_index, parent.clone()))
}

fn has_supported_spell_copy_offer_program(state: &GameState, item: &StackItem) -> bool {
    if item.inline_effect.is_some() {
        return false;
    }
    let Some(source) = state.objects.try_get(item.source) else {
        return false;
    };
    let def = &card_def::CARD_DEFS[source.card_def as usize];
    let effect = match item.mode_chosen {
        0 => (def.spell_effect)(),
        1 => def.mode2.as_ref().map(|mode| (mode.effect)()),
        2 => def.mode3.as_ref().map(|mode| (mode.effect)()),
        _ => None,
    };
    effect
        == Some(EffectOp::Sequence(vec![
            EffectOp::DealDamage {
                target: TargetRef::Target(0),
                amount: 3,
            },
            EffectOp::OfferAffectedPlayerSpellCopy {
                affected: TargetRef::Target(0),
            },
        ]))
}

pub(crate) fn validate_pending_spell_copy(
    state: &GameState,
    pending: &PendingSpellCopy,
) -> Result<StackItem, String> {
    if state.engine.pending_cast.is_some()
        || state.engine.pending_activation.is_some()
        || state.engine.pending_discard.is_some()
        || state.engine.pending_optional_cost.is_some()
        || state.engine.pending_optional_cost_sacrifice.is_some()
        || state.engine.pending_effect.is_some()
        || !state.engine.pending_triggers.is_empty()
        || state.engine.pending_kicked_source.is_some()
    {
        return Err(
            "spell-copy continuation coexists with another staged engine context".to_string(),
        );
    }
    let (parent_index, parent) = pending_spell_copy_parent(state, pending)?;
    if parent.inline_effect.is_some()
        || !parent.discarded.is_empty()
        || parent.is_flashback
        || parent.madness_offer
        || parent.kicked
        || parent.mode_chosen != 0
        || parent.v4.cast_method != Some(CastMethodV4::Normal)
        || parent.v4.face_index != 0
        || parent.v4.x_value != 0
        || !parent.v4.paid_cost_refs.is_empty()
        || parent.v4.activated_ability_index.is_some()
    {
        return Err("the spell-copy continuation parent has invalid cast provenance".to_string());
    }
    if validated_stack_item_target_spec(&parent, state)? != Some(TargetSpec::AnyTarget) {
        return Err(
            "the spell-copy continuation parent is not a valid any-target spell".to_string(),
        );
    }
    if !has_supported_spell_copy_offer_program(state, &parent) {
        return Err(
            "the spell-copy continuation parent has no supported copy-offer program".to_string(),
        );
    }
    let Some(contract) = pending.inherited_target_contract else {
        return Err("the spell-copy continuation lost its inherited target contract".to_string());
    };
    if parent.v4.target_spec != Some(TargetSpec::AnyTarget)
        || parent.targets.as_slice() != [pending.inherited_target]
        || parent.v4.target_contracts.as_slice() != [contract]
        || contract.target() != pending.inherited_target
    {
        return Err(
            "the spell-copy continuation no longer matches its parent's target contract"
                .to_string(),
        );
    }
    if matches!(pending.inherited_target, Target::Object(_))
        && !target_contract_matches_live(state, pending.inherited_target, contract)
    {
        return Err("the copied spell's affected object changed incarnation".to_string());
    }
    let expected_player = match pending.inherited_target {
        Target::Player(player) => player,
        Target::Object(object) => {
            state
                .objects
                .try_get(object)
                .ok_or("the copied spell's affected target object no longer exists")?
                .controller
        }
    };
    if pending.player != expected_player {
        return Err("the spell-copy continuation changed the affected player".to_string());
    }
    match pending.stage {
        SpellCopyStage::Payment => {
            if pending.copy_source.is_some() || pending.copy_stack_item.is_some() {
                return Err("the payment-stage spell copy already has a copy object".to_string());
            }
            if parent_index + 1 != state.stack.len() {
                return Err(
                    "the payment-stage spell-copy parent is not the top stack item".to_string(),
                );
            }
        }
        SpellCopyStage::Retarget | SpellCopyStage::Target => {
            let copy_source = pending
                .copy_source
                .ok_or("the post-payment spell-copy continuation lost its copy object")?;
            let copy_stack_item = pending
                .copy_stack_item
                .ok_or("the post-payment spell-copy continuation lost its stack incarnation")?;
            if copy_stack_item == StackItemId::default()
                || copy_stack_item == pending.resolving_stack_item
            {
                return Err("the spell-copy stack incarnation is invalid".to_string());
            }
            if copy_source == pending.resolving_source {
                return Err("the spell-copy parent and copy source are not distinct".to_string());
            }
            let mut matches = state.stack.iter().enumerate().filter(|(_, item)| {
                item.v4.stack_item_id == copy_stack_item && item.source == copy_source
            });
            let (copy_index, copy) = matches
                .next()
                .ok_or("the spell-copy continuation's copy is no longer on the stack")?;
            let copy_object = state
                .objects
                .try_get(copy_source)
                .ok_or("the spell-copy continuation's copy object no longer exists")?;
            let expected_copy = expected_spell_copy(
                &parent,
                pending,
                copy_source,
                copy_stack_item,
                StackSourceContractV4::capture(state, copy_source, CastMethodV4::Normal),
            );
            let expected_copy_object = expected_spell_copy_source_object(
                state.objects.get(parent.source),
                &parent,
                pending,
            );
            if matches.next().is_some()
                || copy_index + 1 != state.stack.len()
                || copy_index != parent_index + 1
                || copy_object != &expected_copy_object
                || copy != &expected_copy
            {
                return Err(
                    "the created spell copy no longer exactly inherits its parent's stack semantics"
                        .to_string(),
                );
            }
            if validated_stack_item_target_spec(copy, state)? != Some(TargetSpec::AnyTarget)
                || !has_supported_spell_copy_offer_program(state, copy)
            {
                return Err(
                    "the created spell copy no longer has a valid copy-offer program".to_string(),
                );
            }
        }
    }
    Ok(parent)
}

/// Removes the suspended parent and performs its final resolution
/// disposition, then opens the single post-resolution SBA/trigger/priority
/// boundary shared by every answer path.
fn finish_pending_spell_copy_resolution(
    state: &mut GameState,
    pending: &PendingSpellCopy,
) -> Result<(), String> {
    let (_, item) = pending_spell_copy_parent(state, pending)?;
    if !item.is_copy {
        let def = &card_def::CARD_DEFS[state.objects.get(item.source).card_def as usize];
        if item.inline_effect.is_some()
            || !(def.has_type(CardType::Instant) || def.has_type(CardType::Sorcery))
        {
            return Err("the resolving physical spell has no supported departure".to_string());
        }
    }
    let departure = plan_spell_departure(state, &item, Zone::Graveyard)?;
    state.engine.pending_spell_copy = None;
    apply_spell_departure(state, departure)?;
    if state
        .stack
        .iter()
        .any(|entry| entry.v4.stack_item_id == item.v4.stack_item_id)
    {
        return Err("the resolving spell did not leave the stack".to_string());
    }
    collect_and_queue_triggers(state);
    reset_priority(state);
    Ok(())
}

fn apply_choose_spell_copy_payment(state: &mut GameState, pay: bool) -> Result<(), String> {
    let pending = state
        .engine
        .pending_spell_copy
        .clone()
        .ok_or("no spell-copy payment decision is pending")?;
    if pending.stage != SpellCopyStage::Payment {
        return Err("the pending spell-copy decision is not at its payment stage".to_string());
    }
    validate_pending_spell_copy(state, &pending)?;
    if !pay {
        return finish_pending_spell_copy_resolution(state, &pending);
    }
    let Some(plan) = mana::can_pay(&CHAIN_COPY_COST, 0, pending.player, state) else {
        // XMage asks first, then lets the real payment attempt fail without
        // creating a copy. The resolution still finishes normally.
        return finish_pending_spell_copy_resolution(state, &pending);
    };
    pay_plan(state, pending.player, &plan);
    let (copy_source, copy_stack_item) = create_spell_copy(state, &pending);
    let live = state
        .engine
        .pending_spell_copy
        .as_mut()
        .expect("pending was validated above");
    live.stage = SpellCopyStage::Retarget;
    live.copy_source = Some(copy_source);
    live.copy_stack_item = Some(copy_stack_item);
    Ok(())
}

fn apply_choose_spell_copy_retarget(
    state: &mut GameState,
    change_target: bool,
) -> Result<(), String> {
    let pending = state
        .engine
        .pending_spell_copy
        .clone()
        .ok_or("no spell-copy retarget decision is pending")?;
    if pending.stage != SpellCopyStage::Retarget {
        return Err("the pending spell-copy decision is not at its retarget stage".to_string());
    }
    validate_pending_spell_copy(state, &pending)?;
    if change_target {
        state
            .engine
            .pending_spell_copy
            .as_mut()
            .expect("pending was validated above")
            .stage = SpellCopyStage::Target;
        return Ok(());
    }
    let copy_stack_item = pending
        .copy_stack_item
        .ok_or("the retarget-stage spell copy lacks a stack incarnation")?;
    log_final_targeting_events(state, copy_stack_item)?;
    finish_pending_spell_copy_resolution(state, &pending)
}

fn apply_choose_spell_copy_target(state: &mut GameState, target: Target) -> Result<(), String> {
    let pending = state
        .engine
        .pending_spell_copy
        .clone()
        .ok_or("no spell-copy target decision is pending")?;
    if pending.stage != SpellCopyStage::Target {
        return Err("the pending spell-copy decision is not at its target stage".to_string());
    }
    validate_pending_spell_copy(state, &pending)?;
    let copy_source = pending
        .copy_source
        .ok_or("the target-stage spell copy does not exist")?;
    if !legal_targets_for_controller_from_source(
        TargetSpec::AnyTarget,
        &[],
        pending.player,
        targeting_source_for_object(state, copy_source),
        state,
    )
    .contains(&target)
    {
        return Err(format!("{target:?} is not a legal spell-copy target"));
    }
    let copy_stack_item = pending
        .copy_stack_item
        .ok_or("the target-stage spell copy lacks a stack incarnation")?;
    let target_contract = StackTargetContractV4::capture(state, target);
    let copy = state
        .stack
        .iter_mut()
        .find(|item| {
            item.v4.stack_item_id == copy_stack_item && item.source == copy_source && item.is_copy
        })
        .ok_or("the target-stage spell copy is no longer on the stack")?;
    copy.targets = vec![target];
    copy.v4.target_contracts = vec![target_contract];
    log_final_targeting_events(state, copy_stack_item)?;
    finish_pending_spell_copy_resolution(state, &pending)
}

/// `cast_it == true` no longer pre-verifies affordability -- it
/// unconditionally stages a real cast via `begin_cast_ex`, exactly as if
/// the card were being cast normally.
///
/// An earlier version of this decision (`drain_pending_madness_or_decide`,
/// removed this increment -- see `advance_until_decision`'s `top.
/// madness_offer` check, which replaced it) pre-filtered on `mana::can_pay`,
/// silently sending an unaffordable card to the graveyard with no `Decision`
/// at all (the same "don't ask when there's only one legal answer" shortcut
/// `drain_pending_cast_or_decide` uses for Fireblast's cast mode). That was
/// root-caused as wrong (Sol #90, increment 11) against the real Java
/// reference: `MadnessTriggeredAbility.resolve()` (`MadnessAbility.java`)
/// is a plain *optional* triggered ability -- it always calls
/// `player.chooseUse(...)` first, with no discrete `canPay()` gate anywhere
/// in that path. "Affordable" in the reference means nothing more than "the
/// real `cast()` call happened to succeed"; an unaffordable attempt still
/// gets offered, still begins a real cast (its own target-selection decision
/// genuinely logged), and only fails -- reverting the card to the graveyard
/// -- at the cost-payment step. Confirmed against two golden-trace games
/// (`game_20260713_002149_0004.txt` decision 26, `game_20260713_002156_0014.txt`
/// decision 45): both show a real `SELECT_TARGETS` record for the Madness
/// card's own target at a moment when the discarding player has zero
/// available mana (every battlefield permanent already tapped this same
/// turn paying for earlier, unrelated spells) -- and in the second game, the
/// very next trace record for that player shows the card sitting in the
/// graveyard, never on the stack/battlefield again, proving the attempt was
/// offered, attempted, and reverted, not silently declined upfront. If the
/// madness cost genuinely can't be paid, that surfaces naturally at
/// `finalize_cast`'s definition-derived affordability check, which
/// (see `abort_cast`'s doc) reverts a failed Madness attempt to the
/// graveyard instead of erroring -- matching the reference's observed
/// "offer it, attempt it, let it fizzle to the graveyard" behavior rather
/// than a hard failure this function used to return.
fn validate_madness_decision_preflight(state: &GameState) -> Result<(ObjectId, PlayerId), String> {
    if state.engine.priority_passes != [true, true] {
        return Err(
            "Madness decision has not reached its post-priority resolution stage".to_string(),
        );
    }
    if state.engine.pending_cast.is_some()
        || state.engine.pending_activation.is_some()
        || state.engine.pending_discard.is_some()
        || state.engine.pending_optional_cost.is_some()
        || state.engine.pending_optional_cost_sacrifice.is_some()
        || state.engine.pending_spell_copy.is_some()
        || state.engine.pending_effect.is_some()
        || !state.engine.pending_triggers.is_empty()
        || (state.step == Step::DeclareAttackers && !state.engine.combat.attackers_declared)
        || (state.step == Step::DeclareBlockers && !state.engine.combat.blockers_declared)
    {
        return Err("Madness decision conflicts with another pending engine stage".to_string());
    }
    let top = state.stack.last().ok_or("no Madness decision is pending")?;
    validated_stack_item_target_spec(top, state)?;
    if top.kind != StackItemKind::MadnessOffer {
        return Err("no Madness decision is pending".to_string());
    }
    Ok((top.source, top.controller))
}

fn apply_choose_madness_cast(state: &mut GameState, cast_it: bool) -> Result<(), String> {
    let (card, owner) = validate_madness_decision_preflight(state)?;
    let item = state
        .stack
        .pop()
        .expect("just confirmed the top of stack is a madness offer");
    debug_assert_eq!(item.source, card);
    debug_assert_eq!(item.controller, owner);
    if !cast_it {
        event::propose_and_commit(state, ProposedEvent::zone_change(card, Zone::Graveyard));
        // Same "a stack item just resolved" bookkeeping `resolve_top_of_
        // stack`'s own caller does (`advance_until_decision`) -- this is
        // no longer a side-channel decision that skips the stack, so it
        // owes the same priority reset.
        collect_and_queue_triggers(state);
        reset_priority(state);
        return Ok(());
    }
    let def = &card_def::CARD_DEFS[state.objects.get(card).card_def as usize];
    def.madness_cost
        .expect("only a Madness card's own offer is ever pushed as a madness_offer StackItem");
    begin_cast_ex(state, owner, card, Some(CastMethodV4::Madness));
    Ok(())
}

pub(crate) fn validate_declare_attackers(
    state: &GameState,
    attackers: &[ObjectId],
) -> Result<(), String> {
    if state.step != Step::DeclareAttackers || state.engine.combat.attackers_declared {
        return Err("no declare-attackers decision is pending".to_string());
    }
    let eligible = eligible_attackers(state);
    if !attackers.iter().all(|a| eligible.contains(a)) {
        return Err("one or more declared attackers is not an eligible attacker".to_string());
    }
    let required = required_goaded_attackers(state, &eligible);
    if !required.iter().all(|attacker| attackers.contains(attacker)) {
        return Err("one or more goaded creatures able to attack was omitted".to_string());
    }
    let mut dedup = attackers.to_vec();
    dedup.sort_unstable();
    dedup.dedup();
    if dedup.len() != attackers.len() {
        return Err("duplicate attacker in declaration".to_string());
    }

    Ok(())
}

fn apply_declare_attackers(state: &mut GameState, attackers: Vec<ObjectId>) -> Result<(), String> {
    validate_declare_attackers(state, &attackers)?;

    for &id in &attackers {
        if !has_effective_keyword(state, id, Keywords::VIGILANCE) {
            event::propose_and_commit(state, ProposedEvent::tap(id));
        }
    }
    state.engine.combat.attackers = attackers;
    state.engine.combat.attackers_declared = true;
    collect_and_queue_triggers(state);
    reset_priority(state);
    Ok(())
}

pub(crate) fn validate_declare_blockers(
    state: &GameState,
    blocks: &[(ObjectId, ObjectId)],
) -> Result<(), String> {
    if state.step != Step::DeclareBlockers || state.engine.combat.blockers_declared {
        return Err("no declare-blockers decision is pending".to_string());
    }
    let attackers: Vec<ObjectId> = state
        .engine
        .combat
        .attackers
        .iter()
        .copied()
        .filter(|&id| is_still_in_combat(state, id))
        .collect();
    let mut used_blockers = Vec::new();
    for &(blocker, attacker) in blocks {
        if !attackers.contains(&attacker) {
            return Err(format!("{attacker} is not an attacker this combat"));
        }
        if !legal_blockers_for(state, attacker).contains(&blocker) {
            return Err(format!("{blocker} cannot legally block {attacker}"));
        }
        if used_blockers.contains(&blocker) {
            return Err(format!(
                "{blocker} is assigned to block more than one attacker"
            ));
        }
        used_blockers.push(blocker);
    }
    for &attacker in &attackers {
        let assigned = blocks
            .iter()
            .filter(|(_, blocked)| *blocked == attacker)
            .count();
        let minimum = minimum_blockers_required(state, attacker);
        if assigned != 0 && assigned < minimum {
            return Err(format!(
                "{attacker} requires at least {minimum} creatures to block it"
            ));
        }
    }

    Ok(())
}

fn apply_declare_blockers(
    state: &mut GameState,
    blocks: Vec<(ObjectId, ObjectId)>,
) -> Result<(), String> {
    validate_declare_blockers(state, &blocks)?;

    let mut blocked_by: Vec<(ObjectId, Vec<ObjectId>)> = Vec::new();
    for &(blocker, attacker) in &blocks {
        match blocked_by.iter_mut().find(|(a, _)| *a == attacker) {
            Some((_, bs)) => bs.push(blocker),
            None => blocked_by.push((attacker, vec![blocker])),
        }
    }
    state.engine.combat.blocked_by = blocked_by;
    state.engine.combat.blockers_declared = true;
    collect_and_queue_triggers(state);
    reset_priority(state);
    Ok(())
}

fn apply_order_triggers(state: &mut GameState, perm: Vec<usize>) -> Result<(), String> {
    if state.engine.pending_triggers.is_empty() {
        return Err("no pending triggers to order".to_string());
    }
    let controller = state.engine.pending_triggers[0].controller;
    let group_len = state
        .engine
        .pending_triggers
        .iter()
        .take_while(|t| t.controller == controller)
        .count();
    if group_len < 2 {
        return Err("fewer than two triggers do not require ordering".to_string());
    }

    let mut sorted = perm.clone();
    sorted.sort_unstable();
    if sorted != (0..group_len).collect::<Vec<_>>() {
        return Err(format!(
            "OrderTriggers action must be a permutation of 0..{group_len}"
        ));
    }

    let group: Vec<_> = state.engine.pending_triggers.drain(..group_len).collect();
    let mut ordered = perm
        .into_iter()
        .map(|index| {
            let mut trigger = group[index].clone();
            trigger.placement_ordered = true;
            trigger
        })
        .collect::<Vec<_>>();
    ordered.append(&mut state.engine.pending_triggers);
    state.engine.pending_triggers = ordered;
    // Preserve the historical immediate placement for groups whose triggers
    // are all untargeted. A targeted member remains queued at its ordinary
    // ChooseTargets decision before the remaining group continues.
    let _ = drain_pending_triggers_or_decide(state);
    if state.engine.halted.is_some() {
        return Err("ordered trigger group failed placement validation".to_string());
    }
    Ok(())
}

pub(crate) fn validate_pending_land_play(
    state: &GameState,
    pending: &PendingLandPlay,
) -> Result<(), String> {
    let object = state
        .objects
        .try_get(pending.source)
        .ok_or("pending land source is missing")?;
    if object.zone_change_count != pending.source_zone_change_count
        || object.zone != pending.origin_zone
        || object.owner != pending.controller
        || !matches!(pending.origin_zone, Zone::Hand | Zone::Exile)
    {
        return Err("pending land source incarnation or ownership changed".to_string());
    }
    let def = card_def::CARD_DEFS
        .get(object.card_def as usize)
        .ok_or("pending land definition is missing")?;
    if !def.is_playable_land()
        || def.as_enters_choose_color_other_than != Some(pending.excluded_color)
        || pending.excluded_color == ManaColor::C
    {
        return Err("pending land color-choice definition changed".to_string());
    }
    if !land_drop_candidates(pending.controller, state).contains(&pending.source) {
        return Err("pending land is no longer a legal land drop".to_string());
    }
    if state.engine.pending_cast.is_some()
        || state.engine.pending_activation.is_some()
        || state.engine.pending_discard.is_some()
        || state.engine.pending_optional_cost.is_some()
        || state.engine.pending_optional_cost_sacrifice.is_some()
        || state.engine.pending_spell_copy.is_some()
        || state.engine.pending_effect.is_some()
        || !state.engine.pending_triggers.is_empty()
        || state.engine.pending_kicked_source.is_some()
    {
        return Err("pending land choice coexists with another staged context".to_string());
    }
    Ok(())
}

fn apply_choose_land_color(state: &mut GameState, option_index: u16) -> Result<(), String> {
    let pending = state
        .engine
        .pending_land_play
        .clone()
        .ok_or("no land color choice is pending")?;
    validate_pending_land_play(state, &pending)?;
    let legal_colors = [
        ManaColor::W,
        ManaColor::U,
        ManaColor::B,
        ManaColor::R,
        ManaColor::G,
    ]
    .into_iter()
    .filter(|color| *color != pending.excluded_color)
    .collect::<Vec<_>>();
    let color = *legal_colors
        .get(usize::from(option_index))
        .ok_or("land color-choice option index is outside the printed color set")?;
    state.engine.pending_land_play = None;
    play_land(state, pending.controller, pending.source, Some(color));
    Ok(())
}

fn apply_pending_cast_effect_option(
    state: &mut GameState,
    option_index: u16,
) -> Result<(), String> {
    let pending = state
        .engine
        .pending_cast
        .clone()
        .ok_or("no cast effect choice is pending")?;
    let stage = pending_cast_action_stage(state, &pending)?;
    let def = &card_def::CARD_DEFS[state.objects.get(pending.spell).card_def as usize];
    match stage {
        PendingCastActionStage::ChooseOptionalAdditionalCost => {
            apply_choose_optional_additional_cost(state, option_index)
        }
        PendingCastActionStage::ChooseAdditionalCostZone => {
            let zone = match option_index {
                0 => ChosenCreatureCostZoneV1::Battlefield,
                1 => ChosenCreatureCostZoneV1::Hand,
                _ => {
                    return Err(
                        "chosen-creature cost option index is outside its two branches".to_string(),
                    )
                }
            };
            let other = match zone {
                ChosenCreatureCostZoneV1::Battlefield => ChosenCreatureCostZoneV1::Hand,
                ChosenCreatureCostZoneV1::Hand => ChosenCreatureCostZoneV1::Battlefield,
            };
            if chosen_creature_cost_candidates(pending.controller, zone, state).is_empty()
                || chosen_creature_cost_candidates(pending.controller, other, state).is_empty()
            {
                return Err(
                    "chosen-creature cost branch choice is not currently offered".to_string(),
                );
            }
            state
                .engine
                .pending_cast
                .as_mut()
                .expect("validated cast remains pending")
                .chosen_creature_cost_zone = Some(zone);
            Ok(())
        }
        PendingCastActionStage::ChooseXValue => {
            let cost = pending_cast_selected_mana_cost(def, &pending, state)
                .ok_or("pending X choice lost its selected spell form")?;
            let maximum = maximum_payable_x(&cost, pending.controller, state)
                .ok_or("pending X choice has no payable value")?;
            let x_value = u8::try_from(option_index)
                .map_err(|_| "chosen X exceeds the supported u8 range".to_string())?;
            if maximum == 0 || x_value > maximum {
                return Err("chosen X is outside the currently payable range".to_string());
            }
            state
                .engine
                .pending_cast
                .as_mut()
                .expect("validated cast remains pending")
                .x_value = Some(x_value);
            Ok(())
        }
        _ => Err("pending cast is not awaiting an effect-option action".to_string()),
    }
}

fn play_land(
    state: &mut GameState,
    player: PlayerId,
    id: ObjectId,
    chosen_color: Option<ManaColor>,
) {
    let ctx = ExecCtx::no_targets(id, player);
    effect::execute(
        &EffectOp::MoveObject {
            object: ObjectRef::ThisSource,
            to_zone: Zone::Battlefield,
        },
        &ctx,
        state,
    );
    state.objects.get_mut(id).v4.chosen_color = chosen_color;
    state.players[player.index()].lands_played_this_turn += 1;
    collect_and_queue_triggers(state);
    state.engine.priority_passes = [false, false];
    state.priority_player = player;
}

/// Plots a hand card (`PlotAbility`): pays `CardDef::plot_cost`, exiles it,
/// and stamps `GameObject::plotted_turn` so `is_plotted_castable_now` can
/// recognize it later. A `SpecialAction` (`usesStack = false` in the Java
/// source): doesn't touch the stack and doesn't pass priority, same shape
/// as `play_land`.
fn plot_spell(state: &mut GameState, player: PlayerId, id: ObjectId) {
    let def = &card_def::CARD_DEFS[state.objects.get(id).card_def as usize];
    let cost = def
        .plot_cost
        .expect("plot_action_candidates only offers cards with a plot_cost");
    let plan = mana::can_pay(&cost, 0, player, state)
        .expect("legality already checked by plot_action_candidates");
    pay_plan(state, player, &plan);
    event::propose_and_commit(state, ProposedEvent::zone_change(id, Zone::Exile));
    state.objects.get_mut(id).plotted_turn = Some(state.turn);
    collect_and_queue_triggers(state);
    state.engine.priority_passes = [false, false];
    state.priority_player = player;
}

/// 601.2a: announcing a cast moves the spell from hand (or graveyard, for a
/// Flashback/Escape cast) onto the stack immediately -- *before* modes/targets are
/// chosen (601.2b/601.2c) or costs are paid (601.2f-h), which is why
/// `PendingCast`'s later stages (see `drain_pending_cast_or_decide`) mutate
/// the `StackItem` this pushes in place rather than building one from
/// scratch at `finalize_cast`. Pre-resolves the cast-mode/additional-cost/
/// spell-mode stages when there's no real choice to make -- see
/// `PendingCast`'s field docs.
fn begin_cast(state: &mut GameState, player: PlayerId, spell_id: ObjectId) {
    begin_cast_ex(state, player, spell_id, None);
}

/// `forced_cast_method`, when `Some`, is a Madness cast (`Action::
/// ChooseMadnessCast(true)`): the card is already sitting in `state.exile`
/// (moved there by `apply_discard`'s Madness interception), and its exact
/// definition-owned Madness cost is derived later from this method. `None` covers every
/// ordinary `Action::CastSpell`, where the cost (if overridden at all) is
/// inferred from the spell's zone and definition instead (Flashback or
/// Escape from the graveyard; `Cost::zero()` for a Plotted card from exile).
fn begin_cast_ex(
    state: &mut GameState,
    player: PlayerId,
    spell_id: ObjectId,
    forced_cast_method: Option<CastMethodV4>,
) {
    let origin_zone = state.objects.get(spell_id).zone;
    let origin_zone_change_count = state.objects.get(spell_id).zone_change_count;
    let def = &card_def::CARD_DEFS[state.objects.get(spell_id).card_def as usize];
    let plotted_turn = state.objects.get(spell_id).plotted_turn;
    debug_assert!(matches!(
        forced_cast_method,
        None | Some(CastMethodV4::Madness)
    ));
    // A card sitting in Exile isn't necessarily Plotted any more: an
    // impulse-draw effect (`effect::EffectOp::ImpulseDraw`) also exiles
    // cards that must still be cast for their *real* mana cost, not for
    // free. A Plotted cast requires both the definition-owned Plot ability and
    // an earlier-turn marker belonging to this card's exact exile
    // incarnation. Root-caused while
    // adding Reckless Impulse/Clockwork Percussionist/Experimental
    // Synthesizer: before this check, any impulse-exiled spell would have
    // been cast for `Cost::zero()`, same as a genuinely Plotted card.
    let is_plotted = forced_cast_method.is_none()
        && origin_zone == Zone::Exile
        && state.objects.get(spell_id).owner == player
        && def.plot_cost.is_some()
        && plotted_turn.is_some_and(|turn| turn < state.turn);
    let target_spec = def.target_spec;
    let cast_method = forced_cast_method.unwrap_or_else(|| {
        if origin_zone == Zone::Graveyard {
            unambiguous_graveyard_cast_method(def).expect(
                "a graveyard CastSpell action was offered through exactly one supported method",
            )
        } else if is_plotted {
            CastMethodV4::Plotted
        } else {
            CastMethodV4::Normal
        }
    });
    let is_flashback = cast_method == CastMethodV4::Flashback;
    let cast_route = match cast_method {
        CastMethodV4::Flashback => SpellCastRouteV4::GraveyardFlashback,
        CastMethodV4::Escape => SpellCastRouteV4::GraveyardEscape,
        CastMethodV4::Plotted => SpellCastRouteV4::Plotted {
            plotted_turn: plotted_turn
                .expect("a Plotted cast was derived from an exact Plot marker"),
        },
        CastMethodV4::Madness => SpellCastRouteV4::Madness,
        CastMethodV4::Normal if origin_zone == Zone::Hand => SpellCastRouteV4::Hand,
        CastMethodV4::Normal if origin_zone == Zone::Exile => {
            let permission = active_permission_for(player, spell_id, state)
                .expect("an ordinary Exile cast was offered through an exact play permission");
            debug_assert_eq!(permission.play_or_cast, PlayOrCast::Cast);
            SpellCastRouteV4::ExilePermission {
                holder: permission.holder,
                permission_zone_change_count: permission.zone_change_generation,
            }
        }
        _ => unreachable!("begin_cast_ex only announces supported cast routes"),
    };
    let cast_origin = SpellCastOriginV4 {
        origin_zone,
        origin_zone_change_count,
        route: cast_route,
        finalized_method: None,
    };
    let cast_mode = if cast_method != CastMethodV4::Normal || def.alt_cost.is_none() {
        Some(CastMode::Normal)
    } else {
        None
    };
    let additional_cost_discarded = if def.additional_cost.is_none() {
        Some(vec![])
    } else {
        None
    };
    let mode_chosen = if has_spell_form_choice(def) {
        None
    } else {
        Some(0)
    };
    let kicked = if def.kicker_cost.is_none() {
        Some(false)
    } else {
        None
    };
    let optional_additional_cost_paid = if def.optional_additional_cost.is_none() {
        Some(false)
    } else {
        None
    };

    move_to_stack(state, spell_id, origin_zone);
    if is_plotted {
        // `move_to_stack` clears incarnation-local Plot provenance by
        // default. Keep it only across the active Exile -> Stack attempt so
        // source validation and an abort-to-Exile retry retain the marker.
        state.objects.get_mut(spell_id).plotted_turn = plotted_turn;
    }
    // `move_to_stack` resets incarnation-local object state. Install the
    // independently frozen pre-move evidence only after that reset, before
    // either the placeholder or its duplicate source contract is created.
    state.objects.get_mut(spell_id).v4.spell_cast_origin = Some(cast_origin);
    // A spell on the stack is controlled by the player who cast it, which
    // may differ from its owner when an exact exile-play permission names a
    // different holder. Freeze that controller into the source contract.
    state.objects.get_mut(spell_id).controller = player;
    let source_contract = StackSourceContractV4::capture(state, spell_id, cast_method);
    let stack_item_id = next_stack_item_id(state);
    state.stack.push(StackItem {
        kind: StackItemKind::Spell,
        source: spell_id,
        controller: player,
        targets: vec![],
        is_copy: false,
        inline_effect: None,
        discarded: vec![],
        is_flashback,
        mode_chosen: 0,
        madness_offer: false,
        // `finalize_cast` fills this in once `PendingCast::kicked` resolves
        // -- see `StackItem::kicked`'s doc.
        kicked: false,
        v4: StackStateV4 {
            stack_item_id,
            source_contract: Some(source_contract),
            ..StackStateV4::spell(cast_method)
        },
    });

    state.engine.pending_cast = Some(PendingCast {
        spell: spell_id,
        source_contract,
        controller: player,
        target_spec,
        targets_chosen: vec![],
        target_contracts: vec![],
        target_selection_finished: false,
        is_flashback,
        cast_mode,
        additional_cost_discarded,
        mode_chosen,
        origin_zone,
        sacrifice_chosen: vec![],
        kicked,
        optional_additional_cost_paid,
        optional_additional_cost_chosen: vec![],
        optional_additional_cost_selection_finished: false,
        x_value: if def.cost.x_count == 0
            && supported_bestow(def).is_none_or(|bestow| bestow.cost.x_count == 0)
        {
            Some(0)
        } else {
            None
        },
        chosen_creature_cost_zone: None,
        chosen_creature_cost: None,
    });
}

/// Stages a `PendingActivation`. The cost itself isn't paid here -- see
/// `apply_discard`'s `DiscardResume::FinishActivation` arm (the discard
/// case) and `finalize_activation` (the no-discard case) for exactly when
/// each component pays, and why that split exists.
fn begin_activation(state: &mut GameState, player: PlayerId, source: ObjectId, ability_index: u8) {
    let def = &card_def::CARD_DEFS[state.objects.get(source).card_def as usize];
    let ability = &def.activated_abilities[ability_index as usize];
    state.engine.pending_activation = Some(PendingActivation {
        source,
        source_zone_change_count: state.objects.get(source).zone_change_count,
        controller: player,
        ability_index,
        target_spec: ability.target_spec,
        targets_chosen: vec![],
        target_contracts: vec![],
        cost_discard_paid: if discard_count_in(ability.cost).is_none() {
            Some(vec![])
        } else {
            None
        },
        object_cost_chosen: vec![],
    });
}

/// Pays whichever cost this cast settled on (flashback cost; or the
/// printed mana cost / alt cost, plus any mandatory additional cost) and
/// fills in the targets/discards on the `StackItem` `begin_cast` already
/// pushed (601.2a put the spell there before this ran). 117.3c: the caster
/// retains priority afterward.
fn finalize_cast(state: &mut GameState, discarded: Vec<ObjectId>) -> Result<(), String> {
    let staged = state
        .engine
        .pending_cast
        .as_ref()
        .ok_or("finalize_cast requires a pending cast")?;
    validate_pending_cast(state, staged)?;
    let staged_method = state
        .stack
        .last()
        .and_then(|item| item.v4.cast_method)
        .ok_or("validated cast placeholder lost its cast method")?;
    let pending = state
        .engine
        .pending_cast
        .take()
        .expect("finalize_cast requires a pending cast");
    let def = &card_def::CARD_DEFS[state.objects.get(pending.spell).card_def as usize];
    let cast_method = finalized_cast_method(&pending, staged_method, def);
    finalize_owned_cast(state, pending, cast_method, discarded)
}

/// Commits a cast whose complete staged record was already validated and
/// removed from `EngineState`. Interactive FinishCast discard uses this form
/// so the chosen cards and any Madness triggers they create never coexist
/// with a serialized, self-authenticating `PendingCast` payment state.
fn finalize_owned_cast(
    state: &mut GameState,
    pending: PendingCast,
    cast_method: CastMethodV4,
    discarded: Vec<ObjectId>,
) -> Result<(), String> {
    let def = &card_def::CARD_DEFS[state.objects.get(pending.spell).card_def as usize];
    let x_value = pending.x_value.unwrap_or(0);
    let mut was_kicked = false;
    // Escape choices are intentionally absent from the frozen public
    // sacrifice feature. Canonicalize the selected set by the controller's
    // graveyard-vector order before it can affect payment events or immutable
    // stack provenance, so two public-equivalent pick orders have identical
    // future behavior. Other object costs preserve their established order.
    let object_cost_chosen = if cast_method == CastMethodV4::Escape {
        let ordered: Vec<_> = state.players[pending.controller.index()]
            .graveyard
            .iter()
            .copied()
            .filter(|object| pending.sacrifice_chosen.contains(object))
            .collect();
        debug_assert_eq!(ordered.len(), pending.sacrifice_chosen.len());
        ordered
    } else {
        pending.sacrifice_chosen.clone()
    };
    let additional_owns_object_cost = def.additional_cost.is_some_and(|components| {
        controlled_permanent_sacrifice_in(components).is_some()
            || tap_permanent_filter_in(components).is_some()
    });
    let base_object_cost_chosen: &[ObjectId] = if additional_owns_object_cost {
        &[]
    } else {
        &object_cost_chosen
    };
    let additional_object_cost_chosen: &[ObjectId] = if additional_owns_object_cost {
        &object_cost_chosen
    } else {
        &[]
    };

    match cast_method {
        CastMethodV4::Plotted => {}
        CastMethodV4::Madness => {
            let cost = def
                .madness_cost
                .expect("validated Madness cast has a definition-owned cost");
            let Some(plan) = mana::can_pay(&cost, 0, pending.controller, state) else {
                abort_cast(state, pending, cast_method);
                return Ok(());
            };
            pay_plan(state, pending.controller, &plan);
        }
        CastMethodV4::Flashback => {
            let fb = def
                .flashback
                .as_ref()
                .expect("validated Flashback cast has a definition-owned cost");
            if !pay_cost_components(
                state,
                pending.controller,
                pending.spell,
                fb.cost,
                base_object_cost_chosen,
            ) {
                abort_cast(state, pending, cast_method);
                return Ok(());
            }
        }
        CastMethodV4::Escape => {
            let escape = def
                .escape
                .as_ref()
                .expect("validated Escape cast has a definition-owned cost");
            if !pay_cost_components(
                state,
                pending.controller,
                pending.spell,
                escape.cost,
                base_object_cost_chosen,
            ) {
                abort_cast(state, pending, cast_method);
                return Ok(());
            }
        }
        CastMethodV4::Normal => {
            let kicked = pending.kicked == Some(true);
            let normal_cost = effective_normal_cast_cost(def, pending.controller, state);
            let plan = if kicked {
                let kicker_cost = def
                    .kicker_cost
                    .expect("validated kicked cast has a definition-owned kicker cost");
                mana::can_pay_combined(
                    &[&normal_cost, &kicker_cost],
                    x_value,
                    pending.controller,
                    state,
                )
            } else {
                mana::can_pay(&normal_cost, x_value, pending.controller, state)
            };
            let Some(plan) = plan else {
                abort_cast(state, pending, cast_method);
                return Ok(());
            };
            pay_plan(state, pending.controller, &plan);
            was_kicked = kicked;
        }
        CastMethodV4::Alternative => {
            let alt = def
                .alt_cost
                .expect("validated alternative cast has a definition-owned cost");
            if !pay_cost_components(
                state,
                pending.controller,
                pending.spell,
                alt,
                base_object_cost_chosen,
            ) {
                abort_cast(state, pending, cast_method);
                return Ok(());
            }
        }
        CastMethodV4::Omen => {
            let omen = supported_omen(def)
                .expect("validated Omen cast has definition-owned spell characteristics");
            let Some(plan) = mana::can_pay(&omen.cost, x_value, pending.controller, state) else {
                abort_cast(state, pending, cast_method);
                return Ok(());
            };
            pay_plan(state, pending.controller, &plan);
        }
        CastMethodV4::Bestow => {
            let bestow = supported_bestow(def)
                .expect("validated Bestow cast has definition-owned characteristics");
            let Some(plan) = mana::can_pay(&bestow.cost, x_value, pending.controller, state) else {
                abort_cast(state, pending, cast_method);
                return Ok(());
            };
            pay_plan(state, pending.controller, &plan);
        }
    }
    if let Some(binding) = pending.chosen_creature_cost {
        let zone = pending
            .chosen_creature_cost_zone
            .expect("validated chosen-creature cost retains its zone");
        if !chosen_creature_cost_candidates(pending.controller, zone, state).contains(&binding) {
            abort_cast(state, pending, cast_method);
            return Ok(());
        }
        if zone == ChosenCreatureCostZoneV1::Hand {
            for observer in [PlayerId::P0, PlayerId::P1] {
                if state
                    .reveal_hand_card(observer, pending.controller, binding.object)
                    .is_err()
                {
                    abort_cast(state, pending, cast_method);
                    return Ok(());
                }
            }
        }
    }
    if let Some(add) = def.additional_cost {
        if !pay_cost_components(
            state,
            pending.controller,
            pending.spell,
            add,
            additional_object_cost_chosen,
        ) {
            abort_cast(state, pending, cast_method);
            return Ok(());
        }
    }

    let paid_optional_additional_cost = if pending.optional_additional_cost_paid == Some(true) {
        let kind = def
            .optional_additional_cost
            .expect("validated paid optional additional cost remains definition-owned");
        let chosen = pending
            .optional_additional_cost_chosen
            .iter()
            .map(|binding| binding.object)
            .collect::<Vec<_>>();
        match kind {
            OptionalAdditionalCostDef::CollectEvidence { .. } => {
                commit_graveyard_exile(state, &chosen)
            }
            OptionalAdditionalCostDef::Bargain => commit_sacrifice(state, &chosen),
        }
        Some(kind)
    } else {
        None
    };

    let mut paid_cost_objects = object_cost_chosen;
    paid_cost_objects.extend(discarded.iter().copied());
    paid_cost_objects.extend(
        pending
            .optional_additional_cost_chosen
            .iter()
            .map(|binding| binding.object),
    );
    if let Some(binding) = pending.chosen_creature_cost {
        paid_cost_objects.push(binding.object);
    }
    // Capture after sacrifice/discard zone changes have committed. These refs
    // describe the payment-time incarnation and visibility forever; later
    // reanimation, bounce, shuffle, or exile cannot rewrite stack provenance.
    let mut paid_cost_refs: Vec<PaidCostRefV4> = paid_cost_objects
        .into_iter()
        .map(|object| PaidCostRefV4::capture(state, object))
        .collect();
    if let Some(binding) = pending.chosen_creature_cost {
        let power = match state.objects.get(binding.object).zone {
            Zone::Battlefield => effective_power(state, binding.object),
            Zone::Hand => card_def::CARD_DEFS[state.objects.get(binding.object).card_def as usize]
                .power
                .map(i32::from)
                .unwrap_or(0),
            _ => {
                abort_cast(state, pending, cast_method);
                return Ok(());
            }
        };
        let Some(reference) = paid_cost_refs
            .iter_mut()
            .find(|reference| reference.object == binding.object)
        else {
            abort_cast(state, pending, cast_method);
            return Ok(());
        };
        reference.power_lki = Some(power);
    }
    let selected_target_spec = if cast_method == CastMethodV4::Omen {
        supported_omen(def)
            .expect("validated Omen cast retains its definition")
            .target_spec
    } else if cast_method == CastMethodV4::Bestow {
        supported_bestow(def)
            .expect("validated Bestow cast retains its definition")
            .target_spec
    } else {
        spell_form_target_spec(def, pending.mode_chosen.unwrap_or(0))
            .expect("validated cast retains its selected spell form")
    };
    // Payment is now irrevocably complete. Stamp the independently owned
    // source-object record before recapturing the finalized stack contract;
    // later validators require both copies to agree with this exact method.
    let chosen_creature_cost = pending.chosen_creature_cost.and_then(|binding| {
        paid_cost_refs
            .iter()
            .find(|reference| reference.object == binding.object)
            .copied()
    });
    let source_v4 = &mut state.objects.get_mut(pending.spell).v4;
    let cast_origin = source_v4
        .spell_cast_origin
        .as_mut()
        .expect("validated pending cast retains its frozen origin evidence");
    debug_assert!(cast_origin.finalized_method.is_none());
    cast_origin.finalized_method = Some(cast_method);
    debug_assert!(source_v4.finalized_cast_binding.is_none());
    source_v4.finalized_cast_binding = Some(FinalizedCastBindingV1 {
        x_value: u16::from(x_value),
        chosen_creature_cost,
    });
    let finalized_source_contract =
        StackSourceContractV4::capture(state, pending.spell, cast_method);
    let item = state.stack.last_mut().expect("begin_cast pushed this spell's StackItem and nothing can push another item while a cast is pending");
    debug_assert_eq!(
        item.source, pending.spell,
        "the top of the stack must still be this cast's own placeholder"
    );
    item.targets = pending.targets_chosen;
    item.v4.target_spec = Some(selected_target_spec);
    item.v4.target_contracts = pending.target_contracts;
    item.discarded = discarded;
    // The Omen selection is canonicalized into `cast_method`; the spell's
    // own modal index remains zero because the front-face mode2 program is
    // unrelated to the alternative Omen characteristics.
    item.mode_chosen = if matches!(cast_method, CastMethodV4::Omen | CastMethodV4::Bestow) {
        0
    } else {
        pending.mode_chosen.unwrap_or(0)
    };
    item.kicked = was_kicked;
    item.v4.paid_cost_refs = paid_cost_refs;
    item.v4.optional_additional_cost_paid = paid_optional_additional_cost;
    item.v4.x_value = u16::from(x_value);
    item.v4.cast_method = Some(cast_method);
    item.v4.source_contract = Some(finalized_source_contract);
    let stack_item_id = item.v4.stack_item_id;
    state.players[pending.controller.index()].spells_cast_this_turn = state.players
        [pending.controller.index()]
    .spells_cast_this_turn
    .saturating_add(1);
    log_final_targeting_events(state, stack_item_id)?;
    event::log_spell_cast(state, pending.spell, pending.controller);

    // 601.2i/603.3: casting is complete the instant costs are paid --
    // triggered abilities that saw it happen (Guttersnipe) go on the stack
    // *before* anyone gets priority again, same as `play_land`'s land-drop
    // trigger check.
    collect_and_queue_triggers(state);
    state.engine.priority_passes = [false, false];
    state.priority_player = pending.controller;
    Ok(())
}

/// 601.2a's flip side: if an announced cast turns out to be unpayable, the
/// spell returns to whichever zone it was announced from -- *except* a
/// Madness-cost cast (`PendingCast` stamped with `CastMethodV4::Madness`,
/// `apply_choose_madness_cast`'s `cast_it == true`
/// branch), which goes to the graveyard instead of back to exile.
///
/// Root-caused against two golden-trace games (Sol #90, increment 11 --
/// see `apply_choose_madness_cast`'s doc for the full citation): the
/// reference lets a player attempt an unaffordable Madness cast (a real,
/// logged target-selection decision happens) and the card ends up in the
/// graveyard once the attempt fails, not back in exile still waiting to be
/// offered again. 702.33a's own wording ("cast it... If you don't, it goes
/// to your graveyard") already frames the *whole* Madness window as
/// resolving to "graveyard" on any non-cast outcome; the reference
/// evidently applies that same ultimate destination to a cast that was
/// *attempted* but couldn't complete, rather than rules-textually
/// re-offering the exiled card later. A Plotted cast's definition-derived
/// zero cost can never actually fail this check (paying nothing is
/// always affordable), so this branch is unreachable for Plot -- the
/// madness-cost check is the only real discriminator needed here.
///
/// Root-caused (increment 15) against `game_20260713_002201_0023.txt`
/// decision 127: the only reachable caller of this function today is
/// `finalize_cast`'s method-derived cost branch, itself only reachable from
/// `apply_choose_madness_cast`'s `cast_it == true` arm -- i.e. this always
/// runs as the tail end of *resolving* a Madness-offer triggered ability
/// (`advance_until_decision`'s `top.madness_offer` interception), exactly
/// the same "this stack item's resolution just concluded" moment
/// `apply_choose_madness_cast`'s own `cast_it == false` (decline) branch
/// already models with its `collect_and_queue_triggers`/`reset_priority`
/// pair -- see that branch's comment ("this is no longer a side-channel
/// decision that skips the stack, so it owes the same priority reset").
/// This function was missing that same pair: an *attempted-but-unpayable*
/// Madness cast popped the spell off the stack (same net stack-length
/// change as a genuine resolution) but left `priority_passes` at whatever
/// stale value they held from the "both players already passed, then
/// intercepted a madness offer" moment that led here -- typically
/// `[true, true]`, since that is the only way `Decision::ChooseMadnessCast`
/// is ever reached. With the stack now empty and `priority_passes` still
/// `[true, true]`, `advance_until_decision`'s very next loop iteration
/// treated that as "both players passed with an empty stack" and called
/// `advance_step` immediately -- skipping the fresh priority window 601.2a
/// grants the active player once a cast attempt (successful or reversed)
/// concludes. In the 0023 trace: PlayerRL1 discards Fiery Temper (Madness)
/// via Faithless Looting's Flashback-paid draw-2-discard-2, having just
/// spent every untapped Mountain on that Flashback cost; the Madness offer
/// is accepted (decision 126 chooses its target) but `{R}` can no longer be
/// paid, so the attempt fizzles to the graveyard right here -- and the
/// reference's own next record for PlayerRL1 (decision 127) is a genuine,
/// still-`Precombat Main` `Play Mountain`, not the kernel's skip straight
/// past combat into `Main2`. Fixed by giving this function the same
/// `collect_and_queue_triggers`/`reset_priority` pair `apply_choose_
/// madness_cast`'s decline branch already has, rather than leaving
/// `priority_passes` untouched.
fn abort_cast(state: &mut GameState, pending: PendingCast, cast_method: CastMethodV4) {
    let item = state.stack.pop();
    debug_assert!(
        item.is_some_and(|i| i.source == pending.spell),
        "abort_cast expects its spell's placeholder to be the top of the stack"
    );
    let owner = state.objects.get(pending.spell).owner;
    let plotted_turn = state.objects.get(pending.spell).plotted_turn;
    let to_zone = if cast_method == CastMethodV4::Madness {
        Zone::Graveyard
    } else {
        pending.origin_zone
    };
    match to_zone {
        Zone::Hand => state.players[owner.index()].hand.push(pending.spell),
        Zone::Graveyard => state.players[owner.index()].graveyard.push(pending.spell),
        Zone::Exile => state.exile.push(pending.spell),
        _ => unreachable!("origin_zone is always Hand, Graveyard, or Exile"),
    }
    state.forget_hand_object(pending.spell);
    state.clear_object_relations(pending.spell);
    let turn = state.turn;
    {
        let object = state.objects.get_mut(pending.spell);
        object.zone = to_zone;
        object.controller = owner;
        object.plotted_turn = if cast_method == CastMethodV4::Plotted && to_zone == Zone::Exile {
            plotted_turn
        } else {
            None
        };
        object.zone_change_count += 1;
        object
            .v4
            .reset_for_zone_change(object.card_def, to_zone, turn);
    }
    if to_zone == Zone::Hand {
        for observer in [PlayerId::P0, PlayerId::P1] {
            state
                .reveal_hand_card(observer, owner, pending.spell)
                .expect("an aborted public stack spell just returned to hand");
        }
    }
    collect_and_queue_triggers(state);
    reset_priority(state);
}

/// Pays and pushes an activated ability after every target and object-cost
/// choice is complete. Interactive `DiscardCards` activations instead pay and
/// call `push_paid_activation` synchronously from `apply_discard`, so no
/// serialized Boolean or vector can claim that payment already happened.
fn finalize_activation(state: &mut GameState) {
    let Some(staged) = state.engine.pending_activation.as_ref() else {
        return;
    };
    if validate_pending_activation(state, staged).is_err()
        || !target_cardinality_is_complete(staged.target_spec, staged.targets_chosen.len())
    {
        state.engine.halted = Some((
            UnsupportedMechanic::InvalidEffectContinuation,
            staged.source,
        ));
        return;
    }
    let pending = state
        .engine
        .pending_activation
        .take()
        .expect("finalize_activation requires a pending activation");
    let def = &card_def::CARD_DEFS[state.objects.get(pending.source).card_def as usize];
    let ability = &def.activated_abilities[pending.ability_index as usize];
    let mut discarded = Vec::new();
    if ability
        .cost
        .iter()
        .any(|component| matches!(component, CostComponent::DiscardSelf))
    {
        discarded.push(pending.source);
    }
    // Payment is state-changing and therefore must not live inside a
    // `debug_assert!`, whose argument disappears from release builds.
    let object_cost_ids = pending
        .object_cost_chosen
        .iter()
        .map(|binding| binding.object)
        .collect::<Vec<_>>();
    let x_value = match activation_x_value(ability, &pending.targets_chosen, state) {
        Ok(value) => value,
        Err(_) => {
            state.engine.halted = Some((
                UnsupportedMechanic::InvalidEffectContinuation,
                pending.source,
            ));
            return;
        }
    };
    let paid = pay_cost_components_with_x(
        state,
        pending.controller,
        pending.source,
        ability.cost,
        &object_cost_ids,
        x_value,
    );
    if !paid {
        // Fail closed for a malformed/restored pending activation rather
        // than granting its effect without collecting the full cost.
        state.engine.halted = Some((
            UnsupportedMechanic::InvalidEffectContinuation,
            pending.source,
        ));
        return;
    }
    push_paid_activation(state, pending, discarded);
}

/// Installs an ability whose full cost has just committed. `discarded`
/// carries only actual discard components for resolution context; the
/// schema-v4 provenance additionally freezes every zone-changing payment
/// object, including a sacrificed/exiled source.
fn push_paid_activation(
    state: &mut GameState,
    pending: PendingActivation,
    discarded: Vec<ObjectId>,
) {
    let def = &card_def::CARD_DEFS[state.objects.get(pending.source).card_def as usize];
    let ability = &def.activated_abilities[pending.ability_index as usize];
    let source = state.objects.get(pending.source);
    let ability_source_contract = AbilitySourceContractV4 {
        source: pending.source,
        card_def: source.card_def,
        owner: source.owner,
        controller: pending.controller,
        zone: ability.activation_zone,
        zone_change_count: pending.source_zone_change_count,
        attached_to: if source.zone_change_count == pending.source_zone_change_count {
            source.v4.attached_to
        } else {
            None
        },
    };
    let mut paid_cost_objects = discarded.clone();
    paid_cost_objects.extend(
        pending
            .object_cost_chosen
            .iter()
            .map(|binding| binding.object),
    );
    if ability.cost.iter().any(|component| {
        matches!(
            component,
            CostComponent::SacrificeSelf | CostComponent::ExileSelf | CostComponent::DiscardSelf
        )
    }) && !paid_cost_objects.contains(&pending.source)
    {
        paid_cost_objects.push(pending.source);
    }
    let paid_cost_refs = paid_cost_objects
        .into_iter()
        .map(|object| PaidCostRefV4::capture(state, object))
        .collect();
    let effect = (ability.effect)();
    if ability.activation_zone == Zone::Hand && state.objects.get(pending.source).zone == Zone::Hand
    {
        let owner = state.objects.get(pending.source).owner;
        for observer in [PlayerId::P0, PlayerId::P1] {
            if state
                .reveal_hand_card(observer, owner, pending.source)
                .is_err()
            {
                state.engine.halted = Some((
                    UnsupportedMechanic::InvalidEffectContinuation,
                    pending.source,
                ));
                return;
            }
        }
    }
    let x_value = activation_x_value(ability, &pending.targets_chosen, state)
        .expect("validated paid activation retains its target-bound X value");
    let stack_item_id = next_stack_item_id(state);
    state.stack.push(StackItem {
        kind: StackItemKind::ActivatedAbility,
        source: pending.source,
        controller: pending.controller,
        targets: pending.targets_chosen,
        is_copy: false,
        inline_effect: Some(effect),
        discarded,
        is_flashback: false,
        mode_chosen: 0,
        madness_offer: false,
        kicked: false, // no activated ability in this pool has Kicker
        v4: StackStateV4 {
            stack_item_id,
            x_value: u16::from(x_value),
            paid_cost_refs,
            ability_source_contract: Some(ability_source_contract),
            target_spec: Some(pending.target_spec),
            target_contracts: pending.target_contracts,
            activated_ability_index: Some(pending.ability_index),
            hidden_ability_source: has_unblocked_attacker_return_cost(ability.cost).then_some(
                ObjectLinkV4 {
                    object: pending.source,
                    zone_change_count: pending.source_zone_change_count,
                },
            ),
            ..StackStateV4::default()
        },
    });
    if log_final_targeting_events(state, stack_item_id).is_err() {
        state.engine.halted = Some((
            UnsupportedMechanic::InvalidEffectContinuation,
            pending.source,
        ));
        return;
    }
    // A zone change creates a new object under CR 400.7 and resets its v4
    // state. Never stamp the old activation onto Lorien's new graveyard
    // incarnation (or any sacrificed/exiled source); the stack provenance
    // above owns that historical fact instead.
    if state.objects.get(pending.source).zone_change_count == pending.source_zone_change_count {
        state
            .objects
            .get_mut(pending.source)
            .v4
            .note_ability_use(AbilityKindV4::Activated, pending.ability_index as u16);
    }

    // No ability in this increment's pool triggers off another ability
    // being activated, but see `finalize_cast`'s identical call for why
    // this has to happen before priority is handed back regardless.
    collect_and_queue_triggers(state);
    state.engine.priority_passes = [false, false];
    state.priority_player = pending.controller;
}

/// Hand-or-graveyard -> Stack zone bookkeeping. Not routed through
/// `event::commit` (which explicitly panics on a Stack destination):
/// casting is an engine action, not a `MoveObject` effect leaf any card
/// program emits.
fn move_to_stack(state: &mut GameState, id: ObjectId, from_zone: Zone) {
    let owner = state.objects.get(id).owner;
    match from_zone {
        Zone::Graveyard => state.players[owner.index()].graveyard.retain(|&x| x != id),
        Zone::Exile => state.exile.retain(|&x| x != id),
        _ => state.players[owner.index()].hand.retain(|&x| x != id),
    }
    state.forget_hand_object(id);
    state.clear_object_relations(id);
    let turn = state.turn;
    let object = state.objects.get_mut(id);
    object.zone = Zone::Stack;
    object.plotted_turn = None;
    object.zone_change_count += 1;
    object
        .v4
        .reset_for_zone_change(object.card_def, Zone::Stack, turn);
}

pub(crate) fn pay_plan(state: &mut GameState, player: PlayerId, plan: &mana::PaymentPlan) {
    for &(id, color) in &plan.taps {
        event::propose_and_commit(state, ProposedEvent::tap(id));
        event::propose_and_commit(state, ProposedEvent::mana_add(player, vec![color]));
    }
    // Spend: every newly-tapped mana is fully consumed by this cost by
    // construction (the solver only taps what it needs), plus whatever
    // floating pool the plan says to use.
    for &(_, color) in &plan.taps {
        state.players[player.index()].mana_pool[color.pool_index()] -= 1;
    }
    for (i, &amt) in plan.pool_used.iter().enumerate() {
        state.players[player.index()].mana_pool[i] -= amt;
    }
    state.players[player.index()].life -= plan.life_paid;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card_def::card_id_by_name;
    use crate::effect::PlayerRef;
    use crate::mana::ManaColor;

    fn empty_game() -> GameState {
        GameState::new_from_libraries(&[], &[], |c| format!("card-{c}"), 1)
    }

    // ---- starting-player authority (P1-METAMORPHIC-AUDIT-DESIGN-V4.md
    // ---- Section 1.2): construction, first-turn draw-skip, and round
    // ---- wraparound, exercised in both P0-first and P1-first orientations.

    #[test]
    fn starting_player_authority_flips_active_and_priority_player_at_construction_v1() {
        let lib0: Vec<u16> = vec![1, 2, 3];
        let lib1: Vec<u16> = vec![4, 5, 6];

        let p0_first = GameState::new_from_libraries_with_starting_player_v1(
            &lib0,
            &lib1,
            |c| format!("card-{c}"),
            1,
            PlayerId::P0,
        );
        assert_eq!(p0_first.starting_player, PlayerId::P0);
        assert_eq!(p0_first.active_player, PlayerId::P0);
        assert_eq!(p0_first.priority_player, PlayerId::P0);

        let p1_first = GameState::new_from_libraries_with_starting_player_v1(
            &lib0,
            &lib1,
            |c| format!("card-{c}"),
            1,
            PlayerId::P1,
        );
        assert_eq!(p1_first.starting_player, PlayerId::P1);
        assert_eq!(p1_first.active_player, PlayerId::P1);
        assert_eq!(p1_first.priority_player, PlayerId::P1);

        // The legacy zero-argument constructor is unaffected: it still
        // structurally produces the P0-first literal.
        assert_eq!(empty_game().starting_player, PlayerId::P0);
    }

    /// 103.8a: only the starting player skips their own very first draw
    /// step. The non-starting player draws normally on their own (real)
    /// first turn. Checked in both orientations so the site-2 fix
    /// (`p == state.starting_player` instead of `p == PlayerId::P0`) is
    /// shown to track whichever player was actually forced to start, not
    /// merely physical seat P0.
    #[test]
    fn starting_player_authority_governs_first_turn_draw_skip_in_both_orientations_v1() {
        for starting_player in [PlayerId::P0, PlayerId::P1] {
            let lib0: Vec<u16> = (1..=10).collect();
            let lib1: Vec<u16> = (101..=110).collect();
            let mut state = GameState::new_from_libraries_with_starting_player_v1(
                &lib0,
                &lib1,
                |c| format!("card-{c}"),
                1,
                starting_player,
            );
            let opponent = starting_player.opponent();

            let starting_library_before = state.players[starting_player.index()].library.len();
            advance_step(&mut state); // Untap -> Upkeep
            advance_step(&mut state); // Upkeep -> Draw
            assert_eq!(state.step, Step::Draw);
            assert_eq!(state.active_player, starting_player);
            assert_eq!(
                state.players[starting_player.index()].library.len(),
                starting_library_before,
                "starting_player={starting_player:?} must skip their own first-turn draw"
            );

            // Drive forward to the non-starting player's own first Draw
            // step. However many steps combat-skip logic collapses along
            // the way is irrelevant here; only the eventual Draw entry
            // action matters.
            while state.active_player != opponent {
                advance_step(&mut state);
            }
            let opponent_library_before = state.players[opponent.index()].library.len();
            while state.step != Step::Draw {
                advance_step(&mut state);
            }
            assert_eq!(state.active_player, opponent);
            assert_eq!(
                state.players[opponent.index()].library.len(),
                opponent_library_before - 1,
                "starting_player={starting_player:?}: the non-starting player draws \
                 normally on their own real first turn"
            );
            assert_eq!(
                state.turn, 1,
                "still round 1 for the non-starting player's first real turn"
            );
        }
    }

    /// The round counter advances exactly when `active_player` returns to
    /// whoever started the game, not specifically to physical seat P0.
    /// Checked in both orientations (site 3). Uses an every-step invariant
    /// check rather than a fixed step count, since the no-attackers
    /// combat-skip (`advance_step`'s `DeclareBlockers` -> `EndCombat`
    /// override) changes how many `advance_step` calls one round takes.
    #[test]
    fn starting_player_authority_governs_round_wraparound_in_both_orientations_v1() {
        for starting_player in [PlayerId::P0, PlayerId::P1] {
            let lib0: Vec<u16> = (1..=20).collect();
            let lib1: Vec<u16> = (101..=120).collect();
            let mut state = GameState::new_from_libraries_with_starting_player_v1(
                &lib0,
                &lib1,
                |c| format!("card-{c}"),
                1,
                starting_player,
            );
            let mut expected_turn = state.turn;
            for _ in 0..80 {
                let active_before = state.active_player;
                advance_step(&mut state);
                let just_returned_to_starting_player =
                    active_before != starting_player && state.active_player == starting_player;
                if just_returned_to_starting_player {
                    expected_turn += 1;
                }
                assert_eq!(
                    state.turn, expected_turn,
                    "starting_player={starting_player:?}: turn must advance exactly when \
                     active_player returns to the starting player"
                );
            }
            assert!(
                expected_turn >= 3,
                "test must exercise at least two full round wraparounds"
            );
        }
    }

    #[test]
    fn component_payer_accepts_current_composites_and_fails_closed_on_ambiguous_shapes() {
        let blue = Cost {
            pips: &[mana::Pip::Colored(ManaColor::U)],
            generic: 1,
            x_count: 0,
        };
        assert!(component_payment_shape_supported(&[
            CostComponent::Mana(blue),
            CostComponent::PayLife(3),
        ]));
        assert!(component_payment_shape_supported(&[
            CostComponent::Mana(blue),
            CostComponent::ExileOtherCardsFromOwnGraveyard(3),
        ]));
        assert!(component_payment_shape_supported(&[
            CostComponent::Mana(Cost {
                pips: &[],
                generic: 1,
                x_count: 0,
            }),
            CostComponent::Tap,
            CostComponent::DiscardCards(1),
            CostComponent::SacrificeSelf,
        ]));

        assert!(!component_payment_shape_supported(&[
            CostComponent::Mana(blue),
            CostComponent::Mana(blue),
        ]));
        assert!(!component_payment_shape_supported(&[
            CostComponent::SacrificeLands(1),
            CostComponent::Mana(blue),
        ]));
        assert!(!component_payment_shape_supported(&[
            CostComponent::Mana(Cost {
                pips: &[mana::Pip::Phyrexian(ManaColor::U)],
                generic: 0,
                x_count: 0,
            }),
            CostComponent::PayLife(3),
        ]));
        assert!(!component_payment_shape_supported(&[
            CostComponent::PayLife(2),
            CostComponent::PayLife(1),
        ]));
        assert!(!component_payment_shape_supported(&[
            CostComponent::ExileOtherCardsFromOwnGraveyard(0),
        ]));
        assert!(!component_payment_shape_supported(&[
            CostComponent::SacrificeLands(1),
            CostComponent::ExileOtherCardsFromOwnGraveyard(3),
        ]));
        assert!(!activation_payment_shape_supported(&[
            CostComponent::SacrificeLands(1),
        ]));
    }

    #[test]
    fn escape_component_payment_preflights_mana_before_batch_exile() {
        let mut state = ready_game_in_main1(0);
        let source = put_in_graveyard(&mut state, PlayerId::P0, "Sleep of the Dead");
        let chosen = [
            put_in_graveyard(&mut state, PlayerId::P0, "Lightning Bolt"),
            put_in_graveyard(&mut state, PlayerId::P0, "Fireblast"),
            put_in_graveyard(&mut state, PlayerId::P0, "Counterspell"),
        ];
        let components = [
            CostComponent::Mana(Cost {
                pips: &[mana::Pip::Colored(ManaColor::U)],
                generic: 2,
                x_count: 0,
            }),
            CostComponent::ExileOtherCardsFromOwnGraveyard(3),
        ];
        let before = state.clone();

        assert!(!pay_cost_components(
            &mut state,
            PlayerId::P0,
            source,
            &components,
            &chosen,
        ));
        assert_eq!(state, before, "failed payment must be exactly nonmutating");
    }

    #[test]
    fn activation_cost_rejects_reusing_its_tap_source_for_mana() {
        let mut state = ready_game_in_main1(0);
        let source = put_on_battlefield(&mut state, PlayerId::P0, "Mountain");
        let components = [
            CostComponent::Mana(Cost {
                pips: &[],
                generic: 1,
                x_count: 0,
            }),
            CostComponent::Tap,
        ];

        assert!(!can_pay_activation_components(
            &components,
            PlayerId::P0,
            source,
            &state,
        ));
        assert!(!pay_cost_components(
            &mut state,
            PlayerId::P0,
            source,
            &components,
            &[],
        ));
        assert!(!state.objects.get(source).tapped);
        assert_eq!(state.players[0].mana_pool, [0; 6]);
    }

    #[test]
    fn discard_cost_activation_pays_every_component_in_release_builds() {
        let mut state = ready_game_in_main1(0);
        let blood = put_on_battlefield(&mut state, PlayerId::P0, "Blood Token");
        let discarded = put_in_hand(&mut state, PlayerId::P0, "Lightning Bolt");
        state.players[0].mana_pool[ManaColor::R.pool_index()] = 1;

        step(&mut state, Action::ActivateAbility(blood, 0)).unwrap();
        let decision = advance_until_decision(&mut state);

        assert!(matches!(
            decision,
            Decision::CastSpellOrPass {
                player: PlayerId::P0,
                ..
            }
        ));
        assert_eq!(state.players[0].mana_pool[ManaColor::R.pool_index()], 0);
        assert!(!state.players[0].hand.contains(&discarded));
        assert!(!state.players[0].battlefield.contains(&blood));
        assert_eq!(state.stack.len(), 1, "the paid ability reaches the stack");
        assert_eq!(state.stack[0].source, blood);
    }

    fn put_on_battlefield(state: &mut GameState, player: PlayerId, card_name: &str) -> ObjectId {
        let card_id =
            card_id_by_name(card_name).unwrap_or_else(|| panic!("{card_name} not in CARD_DEFS"));
        let obj_id = state.objects.push(crate::state::GameObject {
            card_def: card_id,
            name: card_name.to_string(),
            owner: player,
            controller: player,
            zone: Zone::Battlefield,
            tapped: false,
            summoning_sick: false,
            damage: 0,
            counters: Default::default(),
            attachments: Vec::new(),
            v4: ObjectStateV4::from_card_def(card_id),
            spell_copy_origin: None,
            plotted_turn: None,
            zone_change_count: 0,
        });
        state.players[player.index()].battlefield.push(obj_id);
        obj_id
    }

    fn put_in_hand(state: &mut GameState, player: PlayerId, card_name: &str) -> ObjectId {
        let card_id =
            card_id_by_name(card_name).unwrap_or_else(|| panic!("{card_name} not in CARD_DEFS"));
        let obj_id = state.objects.push(crate::state::GameObject {
            card_def: card_id,
            name: card_name.to_string(),
            owner: player,
            controller: player,
            zone: Zone::Hand,
            tapped: false,
            summoning_sick: false,
            damage: 0,
            counters: Default::default(),
            attachments: Vec::new(),
            v4: ObjectStateV4::from_card_def(card_id),
            spell_copy_origin: None,
            plotted_turn: None,
            zone_change_count: 0,
        });
        state.players[player.index()].hand.push(obj_id);
        obj_id
    }

    fn put_in_graveyard(state: &mut GameState, player: PlayerId, card_name: &str) -> ObjectId {
        let card_id =
            card_id_by_name(card_name).unwrap_or_else(|| panic!("{card_name} not in CARD_DEFS"));
        let obj_id = state.objects.push(crate::state::GameObject {
            card_def: card_id,
            name: card_name.to_string(),
            owner: player,
            controller: player,
            zone: Zone::Graveyard,
            tapped: false,
            summoning_sick: false,
            damage: 0,
            counters: Default::default(),
            attachments: Vec::new(),
            v4: ObjectStateV4::from_card_def(card_id),
            spell_copy_origin: None,
            plotted_turn: None,
            zone_change_count: 0,
        });
        state.players[player.index()].graveyard.push(obj_id);
        obj_id
    }

    fn push_ninja_of_the_deep_hours_choice_trigger(state: &mut GameState) -> (ObjectId, StackItem) {
        let source = put_on_battlefield(state, PlayerId::P0, "Ninja of the Deep Hours");
        let source_contract = AbilitySourceContractV4::capture(state, source);
        let effect = (trigger::triggers_for(state.objects.get(source).card_def)[0].effect)();
        let stack_item_id = next_stack_item_id(state);
        let item = StackItem {
            kind: StackItemKind::TriggeredAbility,
            source,
            controller: PlayerId::P0,
            targets: vec![],
            is_copy: false,
            inline_effect: Some(effect),
            discarded: vec![],
            is_flashback: false,
            mode_chosen: 0,
            madness_offer: false,
            kicked: false,
            v4: StackStateV4 {
                stack_item_id,
                ability_source_contract: Some(source_contract),
                ..StackStateV4::default()
            },
        };
        state.stack.push(item.clone());
        state.engine.priority_passes = [true, true];
        (source, item)
    }

    #[test]
    fn resumable_effect_choice_preserves_order_stack_and_snapshot_state() {
        let (mut state, source) = stage_may_shuffle_choice(one_card_p0_library(1));
        let resolving = state.stack[0].clone();
        let drawn = state.players[0].library[0];

        let decision = advance_until_decision(&mut state);
        assert_eq!(state.players[0].library, vec![drawn]);
        assert!(state.players[0].hand.is_empty());
        assert_eq!(state.stack, vec![resolving]);
        assert!(matches!(
            decision,
            Decision::ChooseEffectBoolean {
                player: PlayerId::P0,
                source: decision_source,
                ..
            } if decision_source == source
        ));

        let snapshot = state.snapshot();
        let pending_hash = state.state_hash();
        step(&mut state, Action::ChooseEffectBoolean(false)).unwrap();
        let after_first = advance_until_decision(&mut state);
        let completed_hash = state.state_hash();
        assert!(state.players[0].library.is_empty());
        assert_eq!(state.players[0].hand, vec![drawn]);
        assert!(state.stack.is_empty());
        assert!(state.engine.pending_effect.is_none());
        assert!(matches!(after_first, Decision::CastSpellOrPass { .. }));

        state.restore(&snapshot);
        assert_eq!(state.state_hash(), pending_hash);
        assert!(matches!(
            advance_until_decision(&mut state),
            Decision::ChooseEffectBoolean { .. }
        ));
        step(&mut state, Action::ChooseEffectBoolean(false)).unwrap();
        let after_restore = advance_until_decision(&mut state);
        assert_eq!(state.state_hash(), completed_hash);
        assert_eq!(after_restore, after_first);
    }

    #[test]
    fn invalid_resumable_choice_answer_is_nonmutating() {
        let mut state = empty_game();
        push_ninja_of_the_deep_hours_choice_trigger(&mut state);
        assert!(matches!(
            advance_until_decision(&mut state),
            Decision::ChooseEffectOption { .. }
        ));
        let before = state.clone();
        assert!(step(&mut state, Action::ChooseEffectOption(2)).is_err());
        assert_eq!(state, before);
    }

    /// Fireblast's alternative cost (Sol #85: alt costs are payment
    /// *choices*) surfaces a real `Decision::ChooseCastMode` when both the
    /// printed mana cost and sacrificing 2 Mountains are legal, and (Sol
    /// #90, increment 11) a real `Decision::ChooseCostTargets` for *which*
    /// 2 of the 6 Mountains, asked one at a time with the already-picked
    /// one excluded from the second ask's candidates.
    #[test]
    fn fireblast_asks_to_choose_between_mana_cost_and_sacrificing_mountains() {
        let mut state = empty_game();
        for _ in 0..6 {
            put_on_battlefield(&mut state, PlayerId::P0, "Mountain");
        }
        let fireblast = put_in_hand(&mut state, PlayerId::P0, "Fireblast");
        state.priority_player = PlayerId::P0;
        state.step = Step::Main1;

        step(&mut state, Action::CastSpell(fireblast)).unwrap();
        let target = Target::Player(PlayerId::P1);
        step(&mut state, Action::ChooseTarget(target)).unwrap();

        match advance_until_decision(&mut state) {
            Decision::ChooseCastMode {
                player,
                spell,
                options,
            } => {
                assert_eq!(player, PlayerId::P0);
                assert_eq!(spell, fireblast);
                assert_eq!(options, vec![CastMode::Normal, CastMode::Alternative]);
            }
            other => panic!("expected ChooseCastMode, got {other:?}"),
        }

        step(&mut state, Action::ChooseCastMode(CastMode::Alternative)).unwrap();

        let first_pick = match advance_until_decision(&mut state) {
            Decision::ChooseCostTargets {
                player,
                source,
                cost_kind,
                remaining,
                candidates,
            } => {
                assert_eq!(player, PlayerId::P0);
                assert_eq!(source, fireblast);
                assert_eq!(cost_kind, CostKind::SacrificeLands);
                assert_eq!(remaining, 2);
                assert_eq!(
                    candidates.len(),
                    6,
                    "all 6 Mountains are candidates before any pick"
                );
                candidates[0]
            }
            other => panic!("expected ChooseCostTargets, got {other:?}"),
        };
        step(&mut state, Action::ChooseCostTarget(first_pick)).unwrap();

        let second_pick = match advance_until_decision(&mut state) {
            Decision::ChooseCostTargets {
                remaining,
                candidates,
                ..
            } => {
                assert_eq!(remaining, 1);
                assert_eq!(
                    candidates.len(),
                    5,
                    "the first pick is excluded from the second ask"
                );
                assert!(!candidates.contains(&first_pick));
                candidates[0]
            }
            other => panic!("expected second ChooseCostTargets, got {other:?}"),
        };
        step(&mut state, Action::ChooseCostTarget(second_pick)).unwrap();

        advance_until_decision(&mut state); // drives the remaining cast stages (cost payment, stack push)
                                            // Alternative mode: exactly the 2 chosen Mountains sacrificed, no
                                            // mana tapped, and (since none were tapped) all 6 Mountains minus
                                            // the 2 sacrificed remain untapped.
        assert_eq!(
            state.players[0].graveyard.len(),
            2,
            "should have sacrificed exactly 2 Mountains"
        );
        assert!(state.players[0].graveyard.contains(&first_pick));
        assert!(state.players[0].graveyard.contains(&second_pick));
        assert_eq!(state.players[0].battlefield.len(), 4);
        assert!(
            state.players[0]
                .battlefield
                .iter()
                .all(|&id| !state.objects.get(id).tapped),
            "alt cost shouldn't tap any Mountain"
        );
        assert_eq!(state.stack.len(), 1);
        let payment_refs = &state.stack[0].v4.paid_cost_refs;
        assert_eq!(payment_refs.len(), 2);
        assert_eq!(payment_refs[0].object, first_pick);
        assert_eq!(payment_refs[1].object, second_pick);
        for payment in payment_refs {
            assert_eq!(payment.zone, Zone::Graveyard);
            assert_eq!(payment.visible_to_mask, 0b11);
            assert_eq!(
                payment.zone_change_count,
                state.objects.get(payment.object).zone_change_count
            );
        }
    }

    /// When only one of Fireblast's two cost paths is actually payable,
    /// there's no real choice of *mode* -- same "don't ask when there's
    /// only one legal answer" treatment `OrderTriggers` gets for a
    /// singleton group. But *which* Mountain still asks once: the
    /// reference logs a real decision for the first sacrifice pick even
    /// when the aggregate choice is forced (2 candidates for 2 needed),
    /// only going silent on the final pick once exactly 1 candidate
    /// remains (`sacrifice_lands_needed`'s doc / `drain_pending_cast_or_
    /// decide`'s per-pick auto-resolve comment -- root-caused against the
    /// v4 corpus's own candidate-count sequences).
    #[test]
    fn fireblast_auto_resolves_cast_mode_but_still_asks_the_first_sacrifice_pick() {
        let mut state = empty_game();
        // Only 2 Mountains: nowhere near {4}{R}{R}, but exactly enough to
        // sacrifice for the alt cost.
        let m1 = put_on_battlefield(&mut state, PlayerId::P0, "Mountain");
        let m2 = put_on_battlefield(&mut state, PlayerId::P0, "Mountain");
        let fireblast = put_in_hand(&mut state, PlayerId::P0, "Fireblast");
        state.priority_player = PlayerId::P0;
        state.step = Step::Main1;

        step(&mut state, Action::CastSpell(fireblast)).unwrap();
        step(
            &mut state,
            Action::ChooseTarget(Target::Player(PlayerId::P1)),
        )
        .unwrap();

        // No ChooseCastMode decision (only the alt cost is affordable),
        // but a real ChooseCostTargets for the first of the 2 Mountains --
        // both are legal candidates even though both will end up
        // sacrificed either way.
        let first_pick = match advance_until_decision(&mut state) {
            Decision::ChooseCastMode { .. } => {
                panic!("only the alt cost is affordable, so there's nothing to choose")
            }
            Decision::ChooseCostTargets {
                remaining,
                candidates,
                ..
            } => {
                assert_eq!(remaining, 2);
                let mut sorted = candidates.clone();
                sorted.sort_unstable();
                let mut expected = vec![m1, m2];
                expected.sort_unstable();
                assert_eq!(sorted, expected);
                candidates[0]
            }
            other => panic!("expected ChooseCostTargets, got {other:?}"),
        };
        step(&mut state, Action::ChooseCostTarget(first_pick)).unwrap();

        // The second (and final) pick is now forced to the one remaining
        // Mountain -- silently auto-resolved, no second ChooseCostTargets.
        let decision = advance_until_decision(&mut state);
        assert!(
            !matches!(decision, Decision::ChooseCostTargets { .. }),
            "exactly 1 Mountain left for the last pick is no real choice"
        );
        assert_eq!(state.players[0].graveyard.len(), 2);
        assert_eq!(state.stack.len(), 1);
    }

    /// Lava Dart's flashback cost (`CostComponent::SacrificeLands(1)`) is
    /// unconditional (no mana alternative to choose between first, unlike
    /// Fireblast) but still asks a real `Decision::ChooseCostTargets` for
    /// *which* Mountain when more than 1 is controlled.
    #[test]
    fn lava_dart_flashback_asks_which_mountain_to_sacrifice() {
        let mut state = empty_game();
        for _ in 0..3 {
            put_on_battlefield(&mut state, PlayerId::P0, "Mountain");
        }
        let lava_dart = put_in_graveyard(&mut state, PlayerId::P0, "Lava Dart");
        state.priority_player = PlayerId::P0;
        state.step = Step::Main1;

        step(&mut state, Action::CastSpell(lava_dart)).unwrap();
        step(
            &mut state,
            Action::ChooseTarget(Target::Player(PlayerId::P1)),
        )
        .unwrap();

        let pick = match advance_until_decision(&mut state) {
            Decision::ChooseCostTargets {
                player,
                source,
                cost_kind,
                remaining,
                candidates,
            } => {
                assert_eq!(player, PlayerId::P0);
                assert_eq!(source, lava_dart);
                assert_eq!(cost_kind, CostKind::SacrificeLands);
                assert_eq!(remaining, 1);
                assert_eq!(candidates.len(), 3);
                candidates[1]
            }
            other => panic!("expected ChooseCostTargets, got {other:?}"),
        };
        step(&mut state, Action::ChooseCostTarget(pick)).unwrap();

        advance_until_decision(&mut state); // drives the remaining cast stages (cost payment, stack push, exile-on-resolve)
        assert_eq!(
            state.players[0].graveyard.len(),
            1,
            "the sacrificed Mountain, not Lava Dart itself (flashback exiles on resolution)"
        );
        assert!(state.players[0].graveyard.contains(&pick));
        assert_eq!(state.players[0].battlefield.len(), 2);
        assert_eq!(state.stack.len(), 1);
    }

    /// Exactly 1 controlled Mountain for a 1-Mountain flashback cost is no
    /// real choice -- same auto-resolve shortcut as
    /// `fireblast_auto_resolves_to_the_only_affordable_mode`.
    #[test]
    fn lava_dart_flashback_auto_resolves_with_exactly_one_mountain() {
        let mut state = empty_game();
        put_on_battlefield(&mut state, PlayerId::P0, "Mountain");
        let lava_dart = put_in_graveyard(&mut state, PlayerId::P0, "Lava Dart");
        state.priority_player = PlayerId::P0;
        state.step = Step::Main1;

        step(&mut state, Action::CastSpell(lava_dart)).unwrap();
        step(
            &mut state,
            Action::ChooseTarget(Target::Player(PlayerId::P1)),
        )
        .unwrap();

        let decision = advance_until_decision(&mut state);
        assert!(
            !matches!(decision, Decision::ChooseCostTargets { .. }),
            "exactly 1 Mountain for a 1-Mountain cost is no real choice"
        );
        assert_eq!(state.players[0].graveyard.len(), 1);
        assert_eq!(state.players[0].battlefield.len(), 0);
        assert_eq!(state.stack.len(), 1);
    }

    #[test]
    fn illegal_action_returns_err_not_a_silent_noop() {
        let mut state = empty_game();
        let bogus = ObjectId(999);
        let err = step(&mut state, Action::PlayLand(bogus)).unwrap_err();
        assert!(err.contains("not a legal land drop"));
    }

    #[test]
    fn order_triggers_decision_exists_and_is_reachable() {
        let mut state = empty_game();
        let first = put_in_hand(&mut state, PlayerId::P0, "Voldaren Epicure");
        let second = put_in_hand(&mut state, PlayerId::P0, "Voldaren Epicure");
        event::propose_and_commit_batch(
            &mut state,
            vec![
                ProposedEvent::zone_change(first, Zone::Battlefield),
                ProposedEvent::zone_change(second, Zone::Battlefield),
            ],
        );
        collect_and_queue_triggers(&mut state);

        let decision = advance_until_decision(&mut state);
        let pending = match decision {
            Decision::OrderTriggers { player, pending } => {
                assert_eq!(player, PlayerId::P0);
                pending
            }
            other => panic!("expected OrderTriggers, got {other:?}"),
        };
        assert_eq!(pending.len(), 2);

        // Choose to place them reversed: the second source is pushed first
        // (bottom), and the first source is pushed last (top).
        step(&mut state, Action::OrderTriggers(vec![1, 0])).unwrap();
        assert!(state.engine.pending_triggers.is_empty());
        assert_eq!(state.stack.len(), 2);
        assert_eq!(state.stack[0].source, second);
        assert_eq!(state.stack[1].source, first);
    }

    #[test]
    fn order_triggers_rejects_a_non_permutation() {
        let mut state = empty_game();
        state.engine.pending_triggers.push(PendingTrigger {
            controller: PlayerId::P0,
            source: ObjectId(0),
            granted_by: None,
            effect: EffectOp::Sequence(vec![]),
            is_madness_offer: false,
            kicked: false,
            target_spec: TargetSpec::None,
            targets: Vec::new(),
            target_contracts: Vec::new(),
            placement_ordered: false,
            source_contract: None,
            optional_additional_cost_paid: None,
            paid_cost_refs: Vec::new(),
        });
        state.engine.pending_triggers.push(PendingTrigger {
            controller: PlayerId::P0,
            source: ObjectId(1),
            granted_by: None,
            effect: EffectOp::Sequence(vec![]),
            is_madness_offer: false,
            kicked: false,
            target_spec: TargetSpec::None,
            targets: Vec::new(),
            target_contracts: Vec::new(),
            placement_ordered: false,
            source_contract: None,
            optional_additional_cost_paid: None,
            paid_cost_refs: Vec::new(),
        });
        let err = step(&mut state, Action::OrderTriggers(vec![0, 0])).unwrap_err();
        assert!(err.contains("permutation"));
    }

    #[test]
    fn lethal_damage_kills_creature_via_sba() {
        let mut state = empty_game();
        let guttersnipe = put_on_battlefield(&mut state, PlayerId::P0, "Guttersnipe");
        event::propose_and_commit(
            &mut state,
            ProposedEvent::damage(ObjectId(0), Target::Object(guttersnipe), 2),
        );
        trigger::sba_fixed_point(&mut state);
        assert_eq!(state.objects.get(guttersnipe).zone, Zone::Graveyard);
        assert!(!state.players[0].battlefield.contains(&guttersnipe));
        assert!(state.players[0].graveyard.contains(&guttersnipe));
    }

    #[test]
    fn stun_counter_replaces_one_real_untap_attempt_and_respects_skip_untap() {
        let mut state = empty_game();
        let creature = put_on_battlefield(&mut state, PlayerId::P0, "Guttersnipe");
        state.active_player = PlayerId::P0;
        let object = state.objects.get_mut(creature);
        object.tapped = true;
        object.counters.stun = 2;
        object.v4.skip_next_untap = true;

        run_step_entry_action(&mut state, Step::Untap);
        assert!(state.objects.get(creature).tapped);
        assert_eq!(state.objects.get(creature).counters.stun, 2);
        assert!(!state.objects.get(creature).v4.skip_next_untap);

        run_step_entry_action(&mut state, Step::Untap);
        assert!(state.objects.get(creature).tapped);
        assert_eq!(state.objects.get(creature).counters.stun, 1);

        state.objects.get_mut(creature).tapped = false;
        run_step_entry_action(&mut state, Step::Untap);
        assert_eq!(state.objects.get(creature).counters.stun, 1);
    }

    #[test]
    fn zero_toughness_after_counters_also_dies() {
        // Sanity check on the toughness-<=0 leg of 704.5g independent of
        // damage marking.
        let mut state = empty_game();
        let id = put_on_battlefield(&mut state, PlayerId::P0, "Masked Meower"); // 1/1
        state.objects.get_mut(id).counters.plus1_plus1 = -1;
        trigger::sba_fixed_point(&mut state);
        assert_eq!(state.objects.get(id).zone, Zone::Graveyard);
    }

    #[test]
    fn until_end_of_turn_effects_are_cleared_at_cleanup() {
        let mut state = empty_game();
        state
            .engine
            .until_end_of_turn
            .push(UntilEndOfTurnEffect::SyntheticMarker(ObjectId(0)));
        run_step_entry_action(&mut state, Step::Cleanup);
        assert!(state.engine.until_end_of_turn.is_empty());
    }

    /// Regression test for the increment-13 fix (root-caused against
    /// `game_20260713_002153_0010.txt`, see `advance_step`'s doc): a
    /// declared attacker that dies before Declare Blockers must still let
    /// the kernel skip straight to `EndCombat`, same as if it had never
    /// been declared at all -- not just when `combat.attackers` was empty
    /// from the start.
    #[test]
    fn declare_blockers_is_skipped_once_the_sole_attacker_has_died() {
        let mut state = empty_game();
        let attacker = put_on_battlefield(&mut state, PlayerId::P0, "Guttersnipe");
        state.active_player = PlayerId::P0;
        state.step = Step::DeclareAttackers;
        state.engine.combat.attackers = vec![attacker];
        state.engine.combat.attackers_declared = true;

        // The attacker dies mid-Declare-Attackers (e.g. to a burn spell),
        // same as `is_still_in_combat`'s own "declared, but not on the
        // battlefield by damage time" shape.
        state.objects.get_mut(attacker).zone = Zone::Graveyard;

        advance_step(&mut state);
        assert_eq!(
            state.step,
            Step::EndCombat,
            "a dead sole attacker must not keep DeclareBlockers alive"
        );
    }

    /// Same fix, opposite polarity: a *surviving* declared attacker must
    /// still route through `DeclareBlockers` normally.
    #[test]
    fn declare_blockers_is_not_skipped_while_the_attacker_is_still_alive() {
        let mut state = empty_game();
        let attacker = put_on_battlefield(&mut state, PlayerId::P0, "Guttersnipe");
        state.active_player = PlayerId::P0;
        state.step = Step::DeclareAttackers;
        state.engine.combat.attackers = vec![attacker];
        state.engine.combat.attackers_declared = true;

        advance_step(&mut state);
        assert_eq!(state.step, Step::DeclareBlockers);
    }

    #[test]
    fn declare_blockers_filters_one_dead_attacker_from_a_mixed_batch() {
        let mut state = empty_game();
        let dead = put_on_battlefield(&mut state, PlayerId::P0, "Guttersnipe");
        let live = put_on_battlefield(&mut state, PlayerId::P0, "Masked Meower");
        let blocker = put_on_battlefield(&mut state, PlayerId::P1, "Voldaren Epicure");
        state.active_player = PlayerId::P0;
        state.step = Step::DeclareBlockers;
        state.engine.combat.attackers = vec![dead, live];
        state.engine.combat.attackers_declared = true;
        state.objects.get_mut(dead).zone = Zone::Graveyard;

        match advance_until_decision(&mut state) {
            Decision::DeclareBlockers {
                attackers,
                legal_blockers,
                ..
            } => {
                assert_eq!(attackers, vec![live]);
                assert_eq!(legal_blockers, vec![(live, vec![blocker])]);
            }
            other => panic!("expected DeclareBlockers, got {other:?}"),
        }

        let err = step(&mut state, Action::DeclareBlockers(vec![(blocker, dead)])).unwrap_err();
        assert!(
            err.contains("is not an attacker this combat"),
            "dead attackers must be rejected even through the direct engine API: {err}"
        );
    }

    #[test]
    fn haste_creature_can_attack_the_turn_it_enters() {
        let mut state = empty_game();
        let id = put_on_battlefield(&mut state, PlayerId::P0, "Masked Meower");
        state.objects.get_mut(id).summoning_sick = true; // just entered
        state.active_player = PlayerId::P0;
        assert!(
            can_attack(&state, id),
            "haste creature should be able to attack despite summoning sickness"
        );
    }

    #[test]
    fn non_haste_summoning_sick_creature_cannot_attack() {
        let mut state = empty_game();
        let id = put_on_battlefield(&mut state, PlayerId::P0, "Guttersnipe");
        state.objects.get_mut(id).summoning_sick = true;
        state.active_player = PlayerId::P0;
        assert!(!can_attack(&state, id));
    }

    #[test]
    fn flying_attacker_can_only_be_blocked_by_flying_or_reach() {
        let mut state = empty_game();
        let attacker = put_on_battlefield(&mut state, PlayerId::P0, "Sneaky Snacker"); // flying
        let _grounded_blocker = put_on_battlefield(&mut state, PlayerId::P1, "Guttersnipe");
        state.objects.get_mut(attacker).controller = PlayerId::P0;
        let legal = legal_blockers_for(&state, attacker);
        assert!(
            legal.is_empty(),
            "a non-flying, non-reach creature should not be able to block a flyer"
        );
    }

    #[test]
    fn granted_islandwalk_prevents_blocks_only_while_defender_controls_an_island() {
        let mut state = empty_game();
        let attacker = put_on_battlefield(&mut state, PlayerId::P0, "Guttersnipe");
        let blocker = put_on_battlefield(&mut state, PlayerId::P1, "Voldaren Epicure");
        let island = put_on_battlefield(&mut state, PlayerId::P1, "Island");
        let generation = state.objects.get(attacker).zone_change_count;
        state
            .engine
            .until_end_of_turn
            .push(UntilEndOfTurnEffect::ResolvedObjectKeywordEffect {
                object_id: attacker,
                object_zone_change_count: generation,
                layer: Layers::ABILITY_ADDING,
                timestamp: 1,
                duration: EffectDuration::EndOfTurn,
                keywords: Keywords::ISLANDWALK,
            });

        assert!(legal_blockers_for(&state, attacker).is_empty());
        event::propose_and_commit(
            &mut state,
            ProposedEvent::zone_change(island, Zone::Graveyard),
        );
        assert_eq!(legal_blockers_for(&state, attacker), vec![blocker]);
    }

    #[test]
    fn skip_next_untap_follows_current_controller_and_clears_exactly_once() {
        let mut state = empty_game();
        let controlled = put_on_battlefield(&mut state, PlayerId::P1, "Guttersnipe");
        let controlled_later = put_on_battlefield(&mut state, PlayerId::P0, "Guttersnipe");
        let ordinary = put_on_battlefield(&mut state, PlayerId::P0, "Guttersnipe");
        let already_untapped = put_on_battlefield(&mut state, PlayerId::P0, "Guttersnipe");

        // Keep the ownership-oriented battlefield vectors unchanged while
        // changing current control in both directions.
        state.objects.get_mut(controlled).controller = PlayerId::P0;
        state.objects.get_mut(controlled_later).controller = PlayerId::P1;
        for object in [controlled, controlled_later, ordinary] {
            let object = state.objects.get_mut(object);
            object.tapped = true;
            object.summoning_sick = true;
        }
        state.objects.get_mut(controlled).v4.skip_next_untap = true;
        state.objects.get_mut(controlled_later).v4.skip_next_untap = true;
        state.objects.get_mut(already_untapped).v4.skip_next_untap = true;

        state.active_player = PlayerId::P0;
        run_step_entry_action(&mut state, Step::Untap);

        let controlled_object = state.objects.get(controlled);
        assert!(controlled_object.tapped);
        assert!(!controlled_object.v4.skip_next_untap);
        assert!(!controlled_object.summoning_sick);
        let ordinary_object = state.objects.get(ordinary);
        assert!(!ordinary_object.tapped);
        assert!(!ordinary_object.summoning_sick);
        let already_untapped_object = state.objects.get(already_untapped);
        assert!(!already_untapped_object.tapped);
        assert!(!already_untapped_object.v4.skip_next_untap);
        let waiting_object = state.objects.get(controlled_later);
        assert!(waiting_object.tapped);
        assert!(waiting_object.v4.skip_next_untap);
        assert!(waiting_object.summoning_sick);

        state.active_player = PlayerId::P1;
        run_step_entry_action(&mut state, Step::Untap);
        let waiting_object = state.objects.get(controlled_later);
        assert!(waiting_object.tapped);
        assert!(!waiting_object.v4.skip_next_untap);
        assert!(!waiting_object.summoning_sick);

        // The effect was consumed, so the following untap step behaves
        // normally if the permanent becomes tapped again.
        state.objects.get_mut(controlled_later).tapped = true;
        run_step_entry_action(&mut state, Step::Untap);
        assert!(!state.objects.get(controlled_later).tapped);
    }

    /// Regression test for the increment-13 fix (root-caused against
    /// `game_20260713_002212_0038.txt` decisions 115-116, see the
    /// `Decision::DeclareBlockers` construction's own comment):
    /// `ComputerPlayerRL.selectBlockers` sorts attackers by power
    /// descending before asking about blockers one at a time, not by
    /// declaration order. Declares the *lower*-power attacker first, so a
    /// declaration-order bug would put it first in the returned decision
    /// too -- the fix must reorder it after the higher-power one.
    #[test]
    fn declare_blockers_orders_attackers_by_power_descending() {
        let mut state = empty_game();
        let weak = put_on_battlefield(&mut state, PlayerId::P0, "Masked Meower"); // 1/1
        let strong = put_on_battlefield(&mut state, PlayerId::P0, "Guttersnipe"); // 2/2
        let _blocker = put_on_battlefield(&mut state, PlayerId::P1, "Voldaren Epicure");
        state.active_player = PlayerId::P0;
        state.step = Step::DeclareBlockers;
        state.engine.combat.attackers_declared = true;
        // Declared weak-then-strong -- the opposite of the expected order.
        state.engine.combat.attackers = vec![weak, strong];

        match advance_until_decision(&mut state) {
            Decision::DeclareBlockers { attackers, .. } => {
                assert_eq!(
                    attackers,
                    vec![strong, weak],
                    "higher-power attacker (Guttersnipe) must be ordered first"
                );
            }
            other => panic!("expected DeclareBlockers, got {other:?}"),
        }
    }

    #[test]
    fn declare_blockers_preserves_sequential_pick_order_for_damage_assignment() {
        let mut state = empty_game();
        let raider = put_on_battlefield(&mut state, PlayerId::P0, "Goblin Tomb Raider");
        put_on_battlefield(&mut state, PlayerId::P0, "Great Furnace");
        // Create Epicure first so ObjectId sorting would reverse the actual
        // Samurai-then-Epicure sequential pick below.
        let epicure = put_on_battlefield(&mut state, PlayerId::P1, "Voldaren Epicure");
        let samurai = put_on_battlefield(&mut state, PlayerId::P1, "Samurai Token");
        assert!(epicure.0 < samurai.0);
        assert_eq!(
            effective_power(&state, raider),
            2,
            "the controlled artifact turns Tomb Raider into a 2/2"
        );

        state.active_player = PlayerId::P0;
        state.step = Step::DeclareBlockers;
        state.engine.combat.attackers = vec![raider];
        state.engine.combat.attackers_declared = true;
        apply_declare_blockers(&mut state, vec![(samurai, raider), (epicure, raider)]).unwrap();

        assert_eq!(
            state.engine.combat.blocked_by,
            vec![(raider, vec![samurai, epicure])]
        );
        deal_combat_damage(&mut state);

        assert_eq!(
            state.objects.get(epicure).zone,
            Zone::Battlefield,
            "Tomb Raider's 2 damage is all assigned to the first 2-toughness blocker"
        );
        assert!(
            !state.players[PlayerId::P1.index()]
                .battlefield
                .contains(&samurai),
            "the first blocker takes lethal damage and the token ceases to exist"
        );
    }

    // ================================================================
    // Increment 7: Highway Robbery (+ Plot), Fiery Temper (Madness),
    // Searing Blaze and the four Blast cards.
    // ================================================================

    fn put_on_stack_with_method(
        state: &mut GameState,
        player: PlayerId,
        card_name: &str,
        cast_method: CastMethodV4,
    ) -> ObjectId {
        let card_id =
            card_id_by_name(card_name).unwrap_or_else(|| panic!("{card_name} not in CARD_DEFS"));
        let (origin_zone, route, plotted_turn) = match cast_method {
            CastMethodV4::Flashback => {
                (Zone::Graveyard, SpellCastRouteV4::GraveyardFlashback, None)
            }
            CastMethodV4::Plotted => (
                Zone::Exile,
                SpellCastRouteV4::Plotted {
                    plotted_turn: state.turn.saturating_sub(1),
                },
                Some(state.turn.saturating_sub(1)),
            ),
            CastMethodV4::Madness => (Zone::Exile, SpellCastRouteV4::Madness, None),
            CastMethodV4::Escape => (Zone::Graveyard, SpellCastRouteV4::GraveyardEscape, None),
            CastMethodV4::Normal | CastMethodV4::Alternative => {
                (Zone::Hand, SpellCastRouteV4::Hand, None)
            }
            CastMethodV4::Bestow | CastMethodV4::Omen => {
                panic!("test helper cannot fabricate an unsupported cast method")
            }
        };
        let mut object_v4 = ObjectStateV4::from_card_def(card_id);
        object_v4.spell_cast_origin = Some(SpellCastOriginV4 {
            origin_zone,
            origin_zone_change_count: 0,
            route,
            finalized_method: Some(cast_method),
        });
        let obj_id = state.objects.push(crate::state::GameObject {
            card_def: card_id,
            name: card_name.to_string(),
            owner: player,
            controller: player,
            zone: Zone::Stack,
            tapped: false,
            summoning_sick: false,
            damage: 0,
            counters: Default::default(),
            attachments: Vec::new(),
            v4: object_v4,
            spell_copy_origin: None,
            plotted_turn,
            zone_change_count: 1,
        });
        let source_contract = StackSourceContractV4::capture(state, obj_id, cast_method);
        let stack_item_id = next_stack_item_id(state);
        state.stack.push(StackItem {
            kind: StackItemKind::Spell,
            source: obj_id,
            controller: player,
            targets: vec![],
            is_copy: false,
            inline_effect: None,
            discarded: vec![],
            is_flashback: cast_method == CastMethodV4::Flashback,
            mode_chosen: 0,
            madness_offer: false,
            kicked: false,
            v4: StackStateV4 {
                stack_item_id,
                source_contract: Some(source_contract),
                target_spec: Some(card_def::CARD_DEFS[card_id as usize].target_spec),
                ..StackStateV4::spell(cast_method)
            },
        });
        obj_id
    }

    fn put_on_stack(state: &mut GameState, player: PlayerId, card_name: &str) -> ObjectId {
        put_on_stack_with_method(state, player, card_name, CastMethodV4::Normal)
    }

    #[allow(dead_code)]
    fn stage_madness_offer(state: &mut GameState, player: PlayerId) -> ObjectId {
        let temper = put_in_hand(state, player, "Fiery Temper");
        state.engine.pending_discard = Some(PendingDiscard {
            player,
            count: 1,
            resume: DiscardResume::None,
        });
        step(state, Action::Discard(vec![temper])).unwrap();
        assert!(matches!(
            advance_until_decision(state),
            Decision::CastSpellOrPass { .. }
        ));
        assert_eq!(state.stack.last().map(|item| item.source), Some(temper));
        assert_eq!(
            state.stack.last().unwrap().kind,
            StackItemKind::MadnessOffer
        );
        temper
    }

    fn put_spell_copy_on_stack_from_method(
        state: &mut GameState,
        player: PlayerId,
        card_name: &str,
        parent_method: CastMethodV4,
    ) -> ObjectId {
        let parent = put_on_stack_with_method(state, player, card_name, parent_method);
        let mut copy_item = state
            .stack
            .pop()
            .expect("the physical parent was just placed on the stack");
        let parent_object = state.objects.get(parent).clone();
        let copy_origin = SpellCopyOriginV4 {
            parent,
            parent_card_def: parent_object.card_def,
            parent_owner: parent_object.owner,
            parent_controller: parent_object.controller,
            parent_stack_zone_change_count: parent_object.zone_change_count,
            parent_was_copy: false,
        };
        let copy = state.objects.push(crate::state::GameObject {
            card_def: parent_object.card_def,
            name: parent_object.name,
            owner: parent_object.owner,
            controller: parent_object.controller,
            zone: Zone::Stack,
            tapped: false,
            summoning_sick: false,
            damage: 0,
            counters: Default::default(),
            attachments: Vec::new(),
            v4: ObjectStateV4::from_card_def(parent_object.card_def),
            spell_copy_origin: Some(copy_origin),
            plotted_turn: None,
            zone_change_count: 0,
        });
        copy_item.source = copy;
        copy_item.is_copy = true;
        copy_item.is_flashback = false;
        copy_item.v4.stack_item_id = next_stack_item_id(state);
        copy_item.v4.cast_method = Some(CastMethodV4::Normal);
        copy_item.v4.source_contract = Some(StackSourceContractV4::capture(
            state,
            copy,
            CastMethodV4::Normal,
        ));
        state.stack.push(copy_item);

        let parent_destination = if parent_method == CastMethodV4::Flashback {
            Zone::Exile
        } else {
            Zone::Graveyard
        };
        event::propose_and_commit(
            state,
            ProposedEvent::zone_change(parent, parent_destination),
        );
        copy
    }

    fn put_spell_copy_on_stack(
        state: &mut GameState,
        player: PlayerId,
        card_name: &str,
    ) -> ObjectId {
        put_spell_copy_on_stack_from_method(state, player, card_name, CastMethodV4::Normal)
    }

    /// Both players passing priority (117.3c: the caster keeps priority
    /// after finishing a cast, so this always starts with them) is what
    /// actually triggers `resolve_top_of_stack` -- finishing targeting
    /// alone only finalizes the cast and hands priority back. Repeatedly
    /// passes a `CastSpellOrPass` window as long as the stack is still
    /// non-empty, then returns whatever decision comes next -- which may
    /// itself be a resolution-triggered `Decision` (`ChooseOptionalCost`,
    /// `ChooseMadnessCast`) rather than another `CastSpellOrPass`, since a
    /// resolving spell's own effect can stage one synchronously.
    fn pass_until_stack_resolves(state: &mut GameState) -> Decision {
        loop {
            let decision = advance_until_decision(state);
            match &decision {
                Decision::CastSpellOrPass { .. } if !state.stack.is_empty() => {
                    step(state, Action::Pass).unwrap()
                }
                _ => return decision,
            }
        }
    }

    /// Same idea, but stops as soon as the stack has shrunk by at least one
    /// item, instead of draining it completely -- for tests that put a
    /// second, synthetic (targetless) item under the one actually being
    /// exercised, which would panic if it were ever resolved for real.
    fn pass_until_one_stack_item_resolves(state: &mut GameState) {
        let starting_len = state.stack.len();
        loop {
            match advance_until_decision(state) {
                Decision::CastSpellOrPass { .. } => step(state, Action::Pass).unwrap(),
                other => panic!("unexpected decision while resolving one stack item: {other:?}"),
            }
            if state.stack.len() < starting_len {
                return;
            }
        }
    }

    fn ready_game_in_main1(p0_mountains: u32) -> GameState {
        let mut state = empty_game();
        for _ in 0..p0_mountains {
            put_on_battlefield(&mut state, PlayerId::P0, "Mountain");
        }
        state.active_player = PlayerId::P0;
        state.priority_player = PlayerId::P0;
        state.step = Step::Main1;
        state
    }

    #[test]
    fn affinity_reducer_counts_only_controlled_artifact_permanents_and_floors_generic() {
        let mut state = ready_game_in_main1(0);
        let enforcer = &card_def::CARD_DEFS
            [card_id_by_name("Myr Enforcer").expect("Myr Enforcer in pool") as usize];
        let thoughtcast = &card_def::CARD_DEFS
            [card_id_by_name("Thoughtcast").expect("Thoughtcast in pool") as usize];

        let zero_enforcer = effective_normal_cast_cost(enforcer, PlayerId::P0, &state);
        let zero_thoughtcast = effective_normal_cast_cost(thoughtcast, PlayerId::P0, &state);
        assert_eq!(zero_enforcer.generic, 7);
        assert_eq!(zero_thoughtcast.generic, 4);
        assert_eq!(zero_thoughtcast.pips, &[mana::Pip::Colored(ManaColor::U)]);

        put_on_battlefield(&mut state, PlayerId::P0, "Myr Enforcer");
        put_on_battlefield(&mut state, PlayerId::P0, "Great Furnace");
        put_on_battlefield(&mut state, PlayerId::P0, "Mountain");
        put_on_battlefield(&mut state, PlayerId::P1, "Blood Token");
        let partial_enforcer = effective_normal_cast_cost(enforcer, PlayerId::P0, &state);
        let partial_thoughtcast = effective_normal_cast_cost(thoughtcast, PlayerId::P0, &state);
        assert_eq!(
            partial_enforcer.generic, 5,
            "artifact creature and land count"
        );
        assert_eq!(partial_thoughtcast.generic, 2);
        assert_eq!(
            partial_thoughtcast.pips,
            &[mana::Pip::Colored(ManaColor::U)],
            "Affinity never removes the colored pip"
        );

        for _ in 0..6 {
            put_on_battlefield(&mut state, PlayerId::P0, "Blood Token");
        }
        let excess_enforcer = effective_normal_cast_cost(enforcer, PlayerId::P0, &state);
        let excess_thoughtcast = effective_normal_cast_cost(thoughtcast, PlayerId::P0, &state);
        assert_eq!(excess_enforcer.generic, 0);
        assert_eq!(excess_thoughtcast.generic, 0);
        assert_eq!(excess_thoughtcast.pips, &[mana::Pip::Colored(ManaColor::U)]);
    }

    #[test]
    fn myr_enforcer_affinity_controls_offer_payment_and_permanent_resolution() {
        let mut short = ready_game_in_main1(0);
        for name in ["Myr Enforcer", "Ichor Wellspring", "Blood Token"] {
            put_on_battlefield(&mut short, PlayerId::P0, name);
        }
        short.players[0].mana_pool[ManaColor::R.pool_index()] = 3;
        let unaffordable = put_in_hand(&mut short, PlayerId::P0, "Myr Enforcer");
        match advance_until_decision(&mut short) {
            Decision::CastSpellOrPass {
                castable_spells, ..
            } => assert!(!castable_spells.contains(&unaffordable)),
            other => panic!("expected CastSpellOrPass, got {other:?}"),
        }
        let before = short.clone();
        assert!(step(&mut short, Action::CastSpell(unaffordable)).is_err());
        assert_eq!(short, before, "an unaffordable cast attempt is nonmutating");

        let mut payable = ready_game_in_main1(0);
        for name in ["Myr Enforcer", "Ichor Wellspring", "Blood Token"] {
            put_on_battlefield(&mut payable, PlayerId::P0, name);
        }
        payable.players[0].mana_pool[ManaColor::R.pool_index()] = 4;
        let enforcer = put_in_hand(&mut payable, PlayerId::P0, "Myr Enforcer");
        match advance_until_decision(&mut payable) {
            Decision::CastSpellOrPass {
                castable_spells, ..
            } => assert!(castable_spells.contains(&enforcer)),
            other => panic!("expected CastSpellOrPass, got {other:?}"),
        }
        step(&mut payable, Action::CastSpell(enforcer)).unwrap();
        let _ = advance_until_decision(&mut payable);
        assert_eq!(payable.players[0].mana_pool, [0; 6]);
        assert_eq!(payable.stack.last().map(|item| item.source), Some(enforcer));
        pass_until_stack_resolves(&mut payable);
        assert_eq!(payable.objects.get(enforcer).zone, Zone::Battlefield);
        assert!(payable.players[0].battlefield.contains(&enforcer));
        let def = &card_def::CARD_DEFS[payable.objects.get(enforcer).card_def as usize];
        assert_eq!((def.power, def.toughness), (Some(4), Some(4)));
    }

    #[test]
    fn thoughtcast_affinity_draws_exactly_two_and_replays_from_snapshot() {
        let mut state = ready_game_in_main1(0);
        for _ in 0..4 {
            put_on_battlefield(&mut state, PlayerId::P0, "Blood Token");
        }
        state.players[0].mana_pool[ManaColor::U.pool_index()] = 1;
        let thoughtcast = put_in_hand(&mut state, PlayerId::P0, "Thoughtcast");
        for name in ["Mountain", "Island", "Lightning Bolt"] {
            set_library_top(&mut state, PlayerId::P0, name);
        }
        let snapshot = state.snapshot();

        let run = |state: &mut GameState| {
            match advance_until_decision(state) {
                Decision::CastSpellOrPass {
                    castable_spells, ..
                } => assert!(castable_spells.contains(&thoughtcast)),
                other => panic!("expected CastSpellOrPass, got {other:?}"),
            }
            step(state, Action::CastSpell(thoughtcast)).unwrap();
            pass_until_stack_resolves(state);
            assert_eq!(state.players[0].mana_pool, [0; 6]);
            assert_eq!(state.objects.get(thoughtcast).zone, Zone::Graveyard);
            assert_eq!(state.players[0].hand.len(), 2);
            assert_eq!(state.players[0].library.len(), 1);
        };

        run(&mut state);
        let resolved_hash = state.state_hash();
        state.restore(&snapshot);
        run(&mut state);
        assert_eq!(state.state_hash(), resolved_hash);
    }

    #[test]
    fn highway_robbery_decline_draws_nothing() {
        let mut state = ready_game_in_main1(2);
        let robbery = put_in_hand(&mut state, PlayerId::P0, "Highway Robbery");
        step(&mut state, Action::CastSpell(robbery)).unwrap();
        assert_eq!(
            state.players[0].hand.len(),
            0,
            "Highway Robbery itself already left the hand once announced"
        );

        match pass_until_stack_resolves(&mut state) {
            Decision::ChooseOptionalCost {
                player,
                discard_payable,
                sacrifice_payable,
            } => {
                assert_eq!(player, PlayerId::P0);
                assert!(
                    !discard_payable,
                    "hand is empty (only Highway Robbery was in it)"
                );
                assert!(sacrifice_payable, "2 Mountains are in play to sacrifice");
            }
            other => panic!("expected ChooseOptionalCost, got {other:?}"),
        }
        step(
            &mut state,
            Action::ChooseOptionalCost(OptionalCostChoice::Decline),
        )
        .unwrap();
        assert_eq!(
            state.players[0].hand.len(),
            0,
            "declining pays nothing and draws nothing"
        );
        assert_eq!(
            state.players[0].battlefield.len(),
            2,
            "declining doesn't sacrifice a land either"
        );
    }

    #[test]
    fn highway_robbery_discard_draws_two() {
        let mut state = ready_game_in_main1(2);
        let robbery = put_in_hand(&mut state, PlayerId::P0, "Highway Robbery");
        // 2 discardable cards left in hand (not just 1) so the discard is a
        // real choice -- `Decision::Discard` -- rather than silently
        // auto-resolving the same way a genuinely-forced 1-candidate
        // discard would (see `drain_pending_discard_or_decide`'s doc).
        let lava_dart = put_in_hand(&mut state, PlayerId::P0, "Lava Dart");
        let _lightning_bolt = put_in_hand(&mut state, PlayerId::P0, "Lightning Bolt");
        step(&mut state, Action::CastSpell(robbery)).unwrap();
        let decision = pass_until_stack_resolves(&mut state);
        assert!(matches!(
            decision,
            Decision::ChooseOptionalCost {
                discard_payable: true,
                ..
            }
        ));
        step(
            &mut state,
            Action::ChooseOptionalCost(OptionalCostChoice::Discard),
        )
        .unwrap();

        match advance_until_decision(&mut state) {
            Decision::Discard {
                player,
                count,
                choices,
            } => {
                assert_eq!(player, PlayerId::P0);
                assert_eq!(count, 1);
                assert_eq!(choices.len(), 2);
                assert!(choices.contains(&lava_dart));
                step(&mut state, Action::Discard(vec![lava_dart])).unwrap();
            }
            other => panic!("expected Discard, got {other:?}"),
        }
        // 608.2m: Highway Robbery moves to the graveyard only as the very
        // *last* part of its own resolution -- after the discard its own
        // "you may... if you do" text triggered -- so the discarded card
        // lands in the graveyard first, Highway Robbery itself second (see
        // `PendingOptionalCost::spell_resume`'s doc).
        assert_eq!(
            state.players[0].graveyard,
            vec![lava_dart, robbery],
            "the discarded card, then Highway Robbery itself, in that order"
        );
        assert_eq!(state.players[0].hand.len(), 1, "the undiscarded Lightning Bolt is still in hand; the 2 drawn cards came from an empty library");
    }

    #[test]
    fn highway_robbery_sacrifice_land_draws_two() {
        let mut state = ready_game_in_main1(2);
        let robbery = put_in_hand(&mut state, PlayerId::P0, "Highway Robbery");
        for i in 0..2 {
            state.draw_card(PlayerId::P0); // nothing to draw (empty library) -- just proves the count via hand growth below
            let _ = i;
        }
        state.players[0].hand.clear(); // isolate: only Highway Robbery matters for this test
        state.players[0].hand.push(robbery);

        step(&mut state, Action::CastSpell(robbery)).unwrap();
        let decision = pass_until_stack_resolves(&mut state);
        assert!(matches!(
            decision,
            Decision::ChooseOptionalCost {
                sacrifice_payable: true,
                ..
            }
        ));
        step(
            &mut state,
            Action::ChooseOptionalCost(OptionalCostChoice::SacrificeLand),
        )
        .unwrap();

        // 2 Mountains for a 1-Mountain cost is a real choice (candidates >
        // 1) -- see `sacrifice_lands_needed`'s doc for the per-pick
        // auto-resolve rule this is *not* hitting.
        let picked = match advance_until_decision(&mut state) {
            Decision::ChooseCostTargets {
                player,
                source,
                cost_kind,
                remaining,
                candidates,
            } => {
                assert_eq!(player, PlayerId::P0);
                assert_eq!(source, robbery);
                assert_eq!(cost_kind, CostKind::SacrificeLands);
                assert_eq!(remaining, 1);
                assert_eq!(candidates.len(), 2);
                candidates[0]
            }
            other => panic!("expected ChooseCostTargets, got {other:?}"),
        };
        step(&mut state, Action::ChooseCostTarget(picked)).unwrap();
        advance_until_decision(&mut state); // drives the sacrifice + "draw two" resolution

        assert_eq!(
            state.players[0].battlefield.len(),
            1,
            "exactly 1 of the 2 Mountains should have been sacrificed"
        );
        assert!(state.players[0].graveyard.contains(&picked));
        assert_eq!(
            state.players[0].graveyard.len(),
            2,
            "Highway Robbery + the sacrificed Mountain"
        );
    }

    /// Regression test for the increment-14 fix (root-caused against
    /// `game_20260713_002158_0017.txt` decision 225): `drain_pending_
    /// optional_cost_sacrifice_or_decide` (this test's own sacrifice-land
    /// branch, exercised just above) is the *only* one of the two Highway
    /// Robbery optional-cost branches that didn't call `collect_and_queue_
    /// triggers` after running `then` -- its discard-branch sibling,
    /// `DiscardResume::FinishOptionalCost`, always has. That asymmetry
    /// silently dropped any trigger condition on the "draw two cards"
    /// effect itself whenever the *sacrifice* sub-cost (not discard) was
    /// the one paid -- Sneaky Snacker's `DrawNth(3)` (`home_zone: Graveyard`)
    /// chief among them, since Highway Robbery is this pool's only source
    /// of a same-turn multi-card draw.
    #[test]
    fn highway_robbery_sacrifice_land_still_fires_a_draw_triggered_ability() {
        let mut state = ready_game_in_main1(2);
        let robbery = put_in_hand(&mut state, PlayerId::P0, "Highway Robbery");
        let snacker = put_in_graveyard(&mut state, PlayerId::P0, "Sneaky Snacker");
        // Real (non-empty-library) draws, so Highway Robbery's "draw two
        // cards" genuinely fires `CommittedEvent::Draw` twice.
        let mountain = card_def::card_id_by_name("Mountain").unwrap();
        for _ in 0..2 {
            let id = state.objects.push(crate::state::GameObject {
                card_def: mountain,
                name: "Mountain".to_string(),
                owner: PlayerId::P0,
                controller: PlayerId::P0,
                zone: Zone::Library,
                tapped: false,
                summoning_sick: false,
                damage: 0,
                counters: Default::default(),
                attachments: Vec::new(),
                v4: ObjectStateV4::from_card_def(mountain),
                spell_copy_origin: None,
                plotted_turn: None,
                zone_change_count: 0,
            });
            state.players[0].library.push(id);
        }
        // Already drew 1 card of their own this turn -- Highway Robbery's 2
        // draws bring the running count 1 -> 2 -> 3, crossing Sneaky
        // Snacker's `DrawNth(3)` threshold on the *second* draw.
        state.players[0].draws_this_turn = 1;

        step(&mut state, Action::CastSpell(robbery)).unwrap();
        let decision = pass_until_stack_resolves(&mut state);
        assert!(matches!(
            decision,
            Decision::ChooseOptionalCost {
                sacrifice_payable: true,
                ..
            }
        ));
        step(
            &mut state,
            Action::ChooseOptionalCost(OptionalCostChoice::SacrificeLand),
        )
        .unwrap();

        let picked = match advance_until_decision(&mut state) {
            Decision::ChooseCostTargets { candidates, .. } => candidates[0],
            other => panic!("expected ChooseCostTargets, got {other:?}"),
        };
        step(&mut state, Action::ChooseCostTarget(picked)).unwrap();

        // Driving the sacrifice + "draw two" resolution must also queue
        // Sneaky Snacker's own trigger -- observable as a fresh, unresolved
        // stack item before anyone's had a chance to respond to it (the
        // pre-fix bug silently dropped it: nothing would be here at all).
        let after_resolution = advance_until_decision(&mut state);
        assert!(
            matches!(after_resolution, Decision::CastSpellOrPass { .. }),
            "got {after_resolution:?}"
        );
        assert_eq!(state.stack.len(), 1, "Sneaky Snacker's own return-from-graveyard trigger must be sitting on the stack, unresolved");
        assert_eq!(state.stack.last().unwrap().source, snacker);

        pass_until_stack_resolves(&mut state);
        assert!(
            state.players[0].battlefield.contains(&snacker),
            "Sneaky Snacker must have returned to the battlefield"
        );
        assert!(
            state.objects.get(snacker).tapped,
            "\"return... to the battlefield tapped\""
        );
    }

    #[test]
    fn plot_then_free_cast_on_a_later_turn() {
        let mut state = ready_game_in_main1(2);
        let robbery = put_in_hand(&mut state, PlayerId::P0, "Highway Robbery");

        assert_eq!(plot_action_candidates(PlayerId::P0, &state), vec![robbery]);
        step(&mut state, Action::PlotSpell(robbery)).unwrap();
        assert_eq!(state.objects.get(robbery).zone, Zone::Exile);
        assert_eq!(state.objects.get(robbery).plotted_turn, Some(state.turn));
        assert_eq!(
            state.players[0]
                .battlefield
                .iter()
                .filter(|&&id| !state.objects.get(id).tapped)
                .count(),
            0,
            "Plot pays real mana"
        );

        // Same turn: not castable yet (702.163a).
        assert!(!castable_spells(PlayerId::P0, &state).contains(&robbery));

        // A later turn, still Main1: castable for free.
        state.turn += 1;
        state.players[0].mana_pool = [0; 6];
        assert!(
            castable_spells(PlayerId::P0, &state).contains(&robbery),
            "Plotted card should be castable for free on a later turn"
        );
        step(&mut state, Action::CastSpell(robbery)).unwrap();
        assert_eq!(state.stack.len(), 1);
        assert_eq!(state.stack[0].source, robbery);
        // Free: no Mountain should have been tapped for this cast (both were
        // already tapped paying the Plot cost and stay that way).
        assert_eq!(
            state.players[0]
                .battlefield
                .iter()
                .filter(|&&id| state.objects.get(id).tapped)
                .count(),
            2
        );
    }

    #[test]
    fn fiery_temper_discard_offers_madness_cast_for_r() {
        let mut state = ready_game_in_main1(1);
        let temper = put_in_hand(&mut state, PlayerId::P0, "Fiery Temper");
        state.engine.pending_discard = Some(PendingDiscard {
            player: PlayerId::P0,
            count: 1,
            resume: DiscardResume::None,
        });
        step(&mut state, Action::Discard(vec![temper])).unwrap();
        assert_eq!(
            state.objects.get(temper).zone,
            Zone::Exile,
            "Madness exiles instead of graveyarding"
        );
        assert!(state.players[0].graveyard.is_empty());

        // The Madness offer is a real triggered ability, sitting on the
        // stack through normal priority (117.5/603.3b) like anything else
        // there -- `pass_until_stack_resolves` passes both players'
        // trivial `CastSpellOrPass` windows (neither has anything else to
        // do here) until it's actually resolved, same as any other stack
        // item -- see `state::StackItem::madness_offer`'s doc.
        match pass_until_stack_resolves(&mut state) {
            Decision::ChooseMadnessCast { player, card } => {
                assert_eq!(player, PlayerId::P0);
                assert_eq!(card, temper);
            }
            other => panic!("expected ChooseMadnessCast, got {other:?}"),
        }
        step(&mut state, Action::ChooseMadnessCast(true)).unwrap();
        assert_eq!(
            state.stack.len(),
            1,
            "the cast is announced (601.2a) before its cost is paid or its target chosen"
        );
        assert!(
            !state.players[0]
                .battlefield
                .iter()
                .any(|&id| state.objects.get(id).tapped),
            "cost isn't paid until targeting finishes"
        );

        match advance_until_decision(&mut state) {
            Decision::ChooseTargets { legal_targets, .. } => {
                let target = Target::Player(PlayerId::P1);
                assert!(legal_targets.contains(&target));
                step(&mut state, Action::ChooseTarget(target)).unwrap();
            }
            other => panic!("expected ChooseTargets, got {other:?}"),
        }
        // `finalize_cast` (and hence cost payment) only actually runs on
        // the *next* pass through `advance_until_decision`'s loop, not
        // synchronously inside `step(ChooseTarget(..))` itself.
        pass_until_stack_resolves(&mut state);
        assert!(
            state.players[0]
                .battlefield
                .iter()
                .all(|&id| state.objects.get(id).tapped),
            "madness cost {{R}} should have tapped the only Mountain"
        );
        assert_eq!(
            state.players[1].life, 17,
            "Fiery Temper still deals its normal 3 damage when cast via madness"
        );
        assert_eq!(
            state.objects.get(temper).zone,
            Zone::Graveyard,
            "resolves to the graveyard as normal (madness doesn't change that)"
        );
    }

    /// Root-caused against `game_20260713_002149_0004.txt` (decision 26) and
    /// `game_20260713_002156_0014.txt` (decision 45): the reference offers
    /// `ChooseMadnessCast` unconditionally -- no affordability pre-check --
    /// then lets the attempt begin a real cast (a genuine, trace-logged
    /// target pick) that only fails at cost payment, reverting the card to
    /// the graveyard. See `apply_choose_madness_cast`'s doc for the full
    /// citation; this replaces a prior (wrong) version of this test that
    /// asserted the opposite -- a silent, no-decision auto-resolve.
    #[test]
    fn fiery_temper_madness_attempt_fizzles_to_the_graveyard_when_unaffordable() {
        let mut state = empty_game(); // no Mountains: {R} is never payable
                                      // Pinned to Main1 (matching the real corpus scenario this test's
                                      // increment-15 addendum root-causes against) rather than whatever
                                      // step `empty_game()` defaults to -- see that addendum below.
        state.step = Step::Main1;
        state.active_player = PlayerId::P0;
        let temper = put_in_hand(&mut state, PlayerId::P0, "Fiery Temper");
        state.engine.pending_discard = Some(PendingDiscard {
            player: PlayerId::P0,
            count: 1,
            resume: DiscardResume::None,
        });
        step(&mut state, Action::Discard(vec![temper])).unwrap();
        assert_eq!(
            state.objects.get(temper).zone,
            Zone::Exile,
            "Madness exiles instead of graveyarding, same as the affordable case"
        );

        // See `fiery_temper_discard_offers_madness_cast_for_r`'s comment:
        // the Madness offer is a real stack item, resolved only after both
        // players' (here, entirely trivial) priority windows pass.
        match pass_until_stack_resolves(&mut state) {
            Decision::ChooseMadnessCast { player, card } => {
                assert_eq!(player, PlayerId::P0);
                assert_eq!(card, temper);
            }
            other => panic!("expected ChooseMadnessCast (always offered, even when unaffordable), got {other:?}"),
        }
        step(&mut state, Action::ChooseMadnessCast(true)).unwrap();

        // The attempt proceeds through a real target pick, exactly like any
        // other cast -- this is the decision the reference's own
        // SELECT_TARGETS record captures for an ultimately-unpayable
        // attempt.
        match advance_until_decision(&mut state) {
            Decision::ChooseTargets {
                player,
                spell,
                legal_targets,
                ..
            } => {
                assert_eq!(player, PlayerId::P0);
                assert_eq!(spell, temper);
                let target = Target::Player(PlayerId::P1);
                assert!(legal_targets.contains(&target));
                step(&mut state, Action::ChooseTarget(target)).unwrap();
            }
            other => panic!("expected ChooseTargets, got {other:?}"),
        }

        // Only now, at cost payment (no Mountain exists to tap), does the
        // attempt fail -- reverting to the graveyard (not back to exile,
        // and never landing on the stack/resolving).
        let after_abort = advance_until_decision(&mut state);
        assert_eq!(
            state.objects.get(temper).zone,
            Zone::Graveyard,
            "a failed madness attempt goes to the graveyard, not back to exile"
        );
        assert!(
            state.stack.is_empty(),
            "the failed cast must not leave a stack item behind"
        );
        assert_eq!(
            state.players[1].life, 20,
            "no damage: the cast never actually resolved"
        );

        // Regression test (increment 15) for `abort_cast`'s own missing
        // priority reset -- root-caused against `game_20260713_002201_
        // 0023.txt` decision 127. Before that fix, `abort_cast` popped the
        // spell off the stack but never touched `priority_passes`/bumped
        // `priority_round` the way every other "a stack item just stopped
        // being on the stack" transition does (`finalize_cast`'s success
        // path, `apply_choose_madness_cast`'s decline branch, an ordinary
        // `resolve_top_of_stack`). Since a Madness offer is only ever
        // reached via `advance_until_decision`'s `priority_passes ==
        // [true, true]` branch, `priority_passes` is still `[true, true]`
        // stale from that same branch here -- with the stack now empty and
        // untouched-since passes still both `true`, the *same*
        // `advance_until_decision` call fell straight through to
        // `advance_step`, silently skipping the active player's genuine
        // fresh priority window (117.5/601.2a: casting is reversed as
        // though it never began, priority is not thereby lost) instead of
        // ever offering it. The fix must make this same single call return
        // a real `CastSpellOrPass` for the active player, still in
        // `Step::Main1`, not whatever step `advance_step` would have
        // fast-forwarded to.
        assert_eq!(state.step, Step::Main1, "the failed cast must not advance the step -- the active player still owes a genuine priority window here");
        match after_abort {
            Decision::CastSpellOrPass { player, .. } => assert_eq!(player, PlayerId::P0, "the active player, not whoever advance_step would have picked next, must be re-offered priority"),
            other => panic!("expected a fresh CastSpellOrPass reprompt right after the aborted cast, got {other:?}"),
        }
    }

    #[test]
    fn searing_blaze_landfall_triples_damage_to_player_and_creature() {
        let mut state = ready_game_in_main1(2);
        let blaze = put_in_hand(&mut state, PlayerId::P0, "Searing Blaze");
        let victim = put_on_battlefield(&mut state, PlayerId::P1, "Myr Enforcer"); // 4/4
        state.players[0].lands_played_this_turn = 1; // landfall satisfied

        step(&mut state, Action::CastSpell(blaze)).unwrap();
        step(
            &mut state,
            Action::ChooseTarget(Target::Player(PlayerId::P1)),
        )
        .unwrap();
        match advance_until_decision(&mut state) {
            Decision::ChooseTargets { legal_targets, .. } => {
                assert_eq!(
                    legal_targets,
                    vec![Target::Object(victim)],
                    "only P1's own creature is legal for the second target"
                );
                step(&mut state, Action::ChooseTarget(Target::Object(victim))).unwrap();
            }
            other => panic!("expected ChooseTargets, got {other:?}"),
        }
        pass_until_stack_resolves(&mut state);
        assert_eq!(
            state.players[1].life, 17,
            "landfall: 3 damage to the player"
        );
        assert_eq!(
            state.objects.get(victim).damage,
            3,
            "landfall: 3 damage to the creature"
        );
    }

    #[test]
    fn searing_blaze_deals_only_1_without_landfall() {
        let mut state = ready_game_in_main1(2);
        let blaze = put_in_hand(&mut state, PlayerId::P0, "Searing Blaze");
        let victim = put_on_battlefield(&mut state, PlayerId::P1, "Myr Enforcer");

        step(&mut state, Action::CastSpell(blaze)).unwrap();
        step(
            &mut state,
            Action::ChooseTarget(Target::Player(PlayerId::P1)),
        )
        .unwrap();
        step(&mut state, Action::ChooseTarget(Target::Object(victim))).unwrap();
        pass_until_stack_resolves(&mut state);
        assert_eq!(state.players[1].life, 19);
        assert_eq!(state.objects.get(victim).damage, 1);
    }

    #[test]
    fn searing_blaze_still_hits_the_player_when_the_creature_target_dies_first() {
        // 608.2b partial fizzle: the player target is always legal, so
        // Searing Blaze never fully fizzles in this pool -- only the
        // creature-damage leaf should be skipped once its target is gone.
        let mut state = ready_game_in_main1(2);
        let blaze = put_in_hand(&mut state, PlayerId::P0, "Searing Blaze");
        let victim = put_on_battlefield(&mut state, PlayerId::P1, "Voldaren Epicure");

        step(&mut state, Action::CastSpell(blaze)).unwrap();
        step(
            &mut state,
            Action::ChooseTarget(Target::Player(PlayerId::P1)),
        )
        .unwrap();
        step(&mut state, Action::ChooseTarget(Target::Object(victim))).unwrap();
        let _ = advance_until_decision(&mut state);
        assert!(state.engine.pending_cast.is_none());
        // The creature leaves the battlefield before Searing Blaze resolves
        // (simulating e.g. an in-response removal spell).
        event::propose_and_commit(
            &mut state,
            ProposedEvent::zone_change(victim, Zone::Graveyard),
        );
        pass_until_stack_resolves(&mut state);
        assert_eq!(
            state.players[1].life, 19,
            "the player is still hit for the full amount"
        );
    }

    #[test]
    fn target_completability_filters_dependent_first_picks() {
        let mut state = empty_game();
        assert_eq!(
            legal_targets_for(TargetSpec::PlayerThenTheirCreature, &[], &state),
            vec![Target::Player(PlayerId::P0), Target::Player(PlayerId::P1)],
            "the raw legality pool remains suitable for resolution-time partial-target checks"
        );
        assert!(!target_prefix_can_complete(
            TargetSpec::PlayerThenTheirCreature,
            &[],
            &state
        ));
        assert!(
            completable_next_targets_for(TargetSpec::PlayerThenTheirCreature, &[], &state)
                .is_empty()
        );

        let creature = put_on_battlefield(&mut state, PlayerId::P1, "Voldaren Epicure");
        assert!(target_prefix_can_complete(
            TargetSpec::PlayerThenTheirCreature,
            &[],
            &state
        ));
        assert_eq!(
            completable_next_targets_for(TargetSpec::PlayerThenTheirCreature, &[], &state),
            vec![Target::Player(PlayerId::P1)],
            "P0 is a legal first target in isolation but cannot complete the dependent assignment"
        );
        assert_eq!(
            completable_next_targets_for(
                TargetSpec::PlayerThenTheirCreature,
                &[Target::Player(PlayerId::P1)],
                &state
            ),
            vec![Target::Object(creature)]
        );
    }

    #[test]
    fn creature_target_spec_counts_one_and_only_enumerates_battlefield_creatures() {
        let mut state = empty_game();
        let own_creature = put_on_battlefield(&mut state, PlayerId::P0, "Goblin Bushwhacker");
        let land = put_on_battlefield(&mut state, PlayerId::P0, "Island");
        let noncreature = put_on_battlefield(&mut state, PlayerId::P0, "Blood Token");
        let opposing_creature = put_on_battlefield(&mut state, PlayerId::P1, "Cryptic Serpent");

        assert_eq!(target_count(TargetSpec::Creature), 1);
        let legal = legal_targets_for(TargetSpec::Creature, &[], &state);
        assert_eq!(
            legal,
            vec![
                Target::Object(own_creature),
                Target::Object(opposing_creature),
            ]
        );
        assert!(!legal.contains(&Target::Player(PlayerId::P0)));
        assert!(!legal.contains(&Target::Player(PlayerId::P1)));
        assert!(!legal.contains(&Target::Object(land)));
        assert!(!legal.contains(&Target::Object(noncreature)));
    }

    #[test]
    fn creature_target_contract_preserves_staleness_but_rejects_wrong_zone_shape() {
        let mut state = empty_game();
        let creature = put_on_battlefield(&mut state, PlayerId::P1, "Cryptic Serpent");
        let target = Target::Object(creature);
        let contract = StackTargetContractV4::capture(&state, target);

        assert!(stack_target_contract_is_structurally_valid(
            &state,
            TargetSpec::Creature,
            0,
            target,
            contract,
        ));
        assert!(target_contract_matches_live(&state, target, contract));
        assert!(!stack_target_contract_is_structurally_valid(
            &state,
            TargetSpec::Creature,
            0,
            Target::Player(PlayerId::P1),
            StackTargetContractV4::Player(PlayerId::P1),
        ));

        let StackTargetContractV4::Object {
            object,
            card_def,
            owner,
            controller,
            zone_change_count,
            spell_copy_origin,
            ..
        } = contract
        else {
            panic!("creature target should capture an object contract")
        };
        let wrong_zone_contract = StackTargetContractV4::Object {
            object,
            card_def,
            owner,
            controller,
            zone: Zone::Graveyard,
            zone_change_count,
            spell_copy_origin,
        };
        assert!(!stack_target_contract_is_structurally_valid(
            &state,
            TargetSpec::Creature,
            0,
            target,
            wrong_zone_contract,
        ));

        event::propose_and_commit(
            &mut state,
            ProposedEvent::zone_change(creature, Zone::Graveyard),
        );
        assert!(stack_target_contract_is_structurally_valid(
            &state,
            TargetSpec::Creature,
            0,
            target,
            contract,
        ));
        assert!(!target_contract_matches_live(&state, target, contract));
    }

    #[test]
    fn mandatory_target_viability_gates_offers_and_dependent_target_choices() {
        let mut state = ready_game_in_main1(2);
        put_on_battlefield(&mut state, PlayerId::P0, "Island");
        put_on_battlefield(&mut state, PlayerId::P0, "Island");
        let counterspell = put_in_hand(&mut state, PlayerId::P0, "Counterspell");
        let dispel = put_in_hand(&mut state, PlayerId::P0, "Dispel");
        let blaze = put_in_hand(&mut state, PlayerId::P0, "Searing Blaze");

        let empty_targets = castable_spells(PlayerId::P0, &state);
        assert!(!empty_targets.contains(&counterspell));
        assert!(!empty_targets.contains(&dispel));
        assert!(!empty_targets.contains(&blaze));

        let mut bypass = state.clone();
        begin_cast(&mut bypass, PlayerId::P0, counterspell);
        assert!(matches!(
            advance_until_decision(&mut bypass),
            Decision::CastSpellOrPass { .. }
        ));
        assert!(bypass.engine.pending_cast.is_none());
        assert!(bypass.stack.is_empty());
        assert!(bypass.players[0].hand.contains(&counterspell));

        put_on_stack(&mut state, PlayerId::P1, "Guttersnipe");
        let creature_spell_only = castable_spells(PlayerId::P0, &state);
        assert!(creature_spell_only.contains(&counterspell));
        assert!(!creature_spell_only.contains(&dispel));

        put_on_stack(&mut state, PlayerId::P1, "Lightning Bolt");
        let with_instant = castable_spells(PlayerId::P0, &state);
        assert!(with_instant.contains(&counterspell));
        assert!(with_instant.contains(&dispel));
        assert!(!with_instant.contains(&blaze));

        let victim = put_on_battlefield(&mut state, PlayerId::P1, "Voldaren Epicure");
        assert!(castable_spells(PlayerId::P0, &state).contains(&blaze));
        step(&mut state, Action::CastSpell(blaze)).unwrap();
        match advance_until_decision(&mut state) {
            Decision::ChooseTargets { legal_targets, .. } => {
                assert_eq!(legal_targets, vec![Target::Player(PlayerId::P1)]);
            }
            other => panic!("expected filtered first target, got {other:?}"),
        }
        let before = state.state_hash();
        assert!(step(
            &mut state,
            Action::ChooseTarget(Target::Player(PlayerId::P0))
        )
        .is_err());
        assert_eq!(
            state.state_hash(),
            before,
            "an invalid dead-end pick is nonmutating"
        );
        step(
            &mut state,
            Action::ChooseTarget(Target::Player(PlayerId::P1)),
        )
        .unwrap();
        match advance_until_decision(&mut state) {
            Decision::ChooseTargets { legal_targets, .. } => {
                assert_eq!(legal_targets, vec![Target::Object(victim)]);
            }
            other => panic!("expected dependent creature target, got {other:?}"),
        }
    }


    #[test]
    fn counterspell_targeting_excludes_its_announcing_stack_placeholder() {
        let mut state = ready_game_in_main1(0);
        put_on_battlefield(&mut state, PlayerId::P0, "Island");
        put_on_battlefield(&mut state, PlayerId::P0, "Island");
        let underlying = put_on_stack(&mut state, PlayerId::P1, "Lightning Bolt");
        let counterspell = put_in_hand(&mut state, PlayerId::P0, "Counterspell");
        step(&mut state, Action::CastSpell(counterspell)).unwrap();
        match advance_until_decision(&mut state) {
            Decision::ChooseTargets { legal_targets, .. } => {
                assert_eq!(legal_targets, vec![Target::Object(underlying)]);
                assert!(!legal_targets.contains(&Target::Object(counterspell)));
            }
            other => panic!("expected Counterspell targets, got {other:?}"),
        }
    }


    #[test]
    fn counterspell_and_dispel_resolve_through_the_shared_counter_effect() {
        for (counter_name, target_name, islands) in [
            ("Counterspell", "Guttersnipe", 2),
            ("Dispel", "Lightning Bolt", 1),
        ] {
            let mut state = ready_game_in_main1(0);
            for _ in 0..islands {
                put_on_battlefield(&mut state, PlayerId::P0, "Island");
            }
            let target = put_on_stack(&mut state, PlayerId::P1, target_name);
            let counter = put_in_hand(&mut state, PlayerId::P0, counter_name);
            step(&mut state, Action::CastSpell(counter)).unwrap();
            step(&mut state, Action::ChooseTarget(Target::Object(target))).unwrap();
            let terminal = pass_until_stack_resolves(&mut state);
            assert!(matches!(terminal, Decision::CastSpellOrPass { .. }));
            assert_eq!(state.objects.get(target).zone, Zone::Graveyard);
            assert_eq!(state.objects.get(counter).zone, Zone::Graveyard);
            assert!(state.stack.is_empty());
        }
    }

    #[test]
    fn counter_war_resolves_lifo_and_leaves_the_original_spell() {
        let mut state = ready_game_in_main1(0);
        for player in [PlayerId::P0, PlayerId::P1] {
            put_on_battlefield(&mut state, player, "Island");
            put_on_battlefield(&mut state, player, "Island");
        }
        let original = put_on_stack(&mut state, PlayerId::P1, "Lightning Bolt");
        let first = put_in_hand(&mut state, PlayerId::P0, "Counterspell");
        let second = put_in_hand(&mut state, PlayerId::P1, "Counterspell");

        step(&mut state, Action::CastSpell(first)).unwrap();
        step(&mut state, Action::ChooseTarget(Target::Object(original))).unwrap();
        assert!(matches!(
            advance_until_decision(&mut state),
            Decision::CastSpellOrPass {
                player: PlayerId::P0,
                ..
            }
        ));
        step(&mut state, Action::Pass).unwrap();
        assert!(matches!(
            advance_until_decision(&mut state),
            Decision::CastSpellOrPass {
                player: PlayerId::P1,
                ..
            }
        ));
        step(&mut state, Action::CastSpell(second)).unwrap();
        step(&mut state, Action::ChooseTarget(Target::Object(first))).unwrap();
        let _ = advance_until_decision(&mut state);
        pass_until_one_stack_item_resolves(&mut state);

        assert_eq!(state.objects.get(first).zone, Zone::Graveyard);
        assert_eq!(state.objects.get(second).zone, Zone::Graveyard);
        assert_eq!(state.objects.get(original).zone, Zone::Stack);
        assert_eq!(state.stack.len(), 1);
        assert_eq!(state.stack[0].source, original);
    }

    #[test]
    fn shared_counter_effect_preserves_flashback_and_copy_departure_and_stale_fizzle() {
        for is_copy in [false, true] {
            let mut state = ready_game_in_main1(0);
            let (target, counter_name) = if is_copy {
                put_on_battlefield(&mut state, PlayerId::P0, "Island");
                (
                    put_spell_copy_on_stack(&mut state, PlayerId::P1, "Counterspell"),
                    "Dispel",
                )
            } else {
                put_on_battlefield(&mut state, PlayerId::P0, "Island");
                put_on_battlefield(&mut state, PlayerId::P0, "Island");
                (
                    put_on_stack_with_method(
                        &mut state,
                        PlayerId::P1,
                        "Deep Analysis",
                        CastMethodV4::Flashback,
                    ),
                    "Counterspell",
                )
            };
            let counter = put_in_hand(&mut state, PlayerId::P0, counter_name);
            step(&mut state, Action::CastSpell(counter)).unwrap();
            step(&mut state, Action::ChooseTarget(Target::Object(target))).unwrap();
            pass_until_stack_resolves(&mut state);
            if is_copy {
                assert_eq!(state.objects.get(target).zone, Zone::Stack);
                assert!(!state.exile.contains(&target));
                assert!(!state.players[1].graveyard.contains(&target));
            } else {
                assert_eq!(state.objects.get(target).zone, Zone::Exile);
                assert!(state.exile.contains(&target));
            }
        }

        let mut stale = ready_game_in_main1(0);
        put_on_battlefield(&mut stale, PlayerId::P0, "Island");
        put_on_battlefield(&mut stale, PlayerId::P0, "Island");
        let target = put_on_stack(&mut stale, PlayerId::P1, "Lightning Bolt");
        let counter = put_in_hand(&mut stale, PlayerId::P0, "Counterspell");
        step(&mut stale, Action::CastSpell(counter)).unwrap();
        step(&mut stale, Action::ChooseTarget(Target::Object(target))).unwrap();
        let _ = advance_until_decision(&mut stale);
        assert!(stale.engine.pending_cast.is_none());
        event::propose_and_commit(
            &mut stale,
            ProposedEvent::zone_change(target, Zone::Graveyard),
        );
        pass_until_stack_resolves(&mut stale);
        assert_eq!(stale.objects.get(target).zone, Zone::Graveyard);
        assert_eq!(stale.objects.get(counter).zone, Zone::Graveyard);
    }

    #[test]
    fn pyroblast_counters_a_blue_spell_on_the_stack() {
        let mut state = ready_game_in_main1(1);
        let pyroblast = put_in_hand(&mut state, PlayerId::P0, "Pyroblast");
        let counterspell = put_on_stack(&mut state, PlayerId::P1, "Counterspell"); // blue instant

        step(&mut state, Action::CastSpell(pyroblast)).unwrap();
        match advance_until_decision(&mut state) {
            Decision::ChooseSpellMode { mode_count, .. } => {
                assert_eq!(mode_count, 2);
                step(&mut state, Action::ChooseSpellMode(0)).unwrap(); // counter mode
            }
            other => panic!("expected ChooseSpellMode, got {other:?}"),
        }
        match advance_until_decision(&mut state) {
            Decision::ChooseTargets { legal_targets, .. } => {
                assert!(legal_targets.contains(&Target::Object(counterspell)));
                step(
                    &mut state,
                    Action::ChooseTarget(Target::Object(counterspell)),
                )
                .unwrap();
            }
            other => panic!("expected ChooseTargets, got {other:?}"),
        }
        pass_until_stack_resolves(&mut state);
        assert_eq!(
            state.objects.get(counterspell).zone,
            Zone::Graveyard,
            "a blue spell should be countered (moved to its owner's graveyard)"
        );
        assert_eq!(
            state.objects.get(pyroblast).zone,
            Zone::Graveyard,
            "Pyroblast itself resolves to the graveyard as normal"
        );
    }

    #[test]
    fn pyroblast_countering_a_copied_blue_spell_makes_only_the_copy_cease() {
        let mut state = ready_game_in_main1(1);
        let pyroblast = put_in_hand(&mut state, PlayerId::P0, "Pyroblast");
        let copied_counterspell = put_spell_copy_on_stack(&mut state, PlayerId::P1, "Counterspell");

        step(&mut state, Action::CastSpell(pyroblast)).unwrap();
        step(&mut state, Action::ChooseSpellMode(0)).unwrap();
        step(
            &mut state,
            Action::ChooseTarget(Target::Object(copied_counterspell)),
        )
        .unwrap();
        pass_until_stack_resolves(&mut state);

        assert!(state.stack.is_empty());
        assert_eq!(state.objects.get(copied_counterspell).zone, Zone::Stack);
        assert!(!state.players[1].graveyard.contains(&copied_counterspell));
        assert!(!state.exile.contains(&copied_counterspell));
        assert!(!state.command.contains(&copied_counterspell));
        assert!(state.players[0].graveyard.contains(&pyroblast));
        assert!(
            !state.engine.event_history.iter().any(|event| matches!(
                event,
                CommittedEvent::ZoneChange { object, .. } if *object == copied_counterspell
            )),
            "a virtual spell copy must not generate a physical-card zone change"
        );
    }

    #[test]
    fn countered_physical_flashback_spell_is_exiled_but_its_copy_still_ceases() {
        for copied in [false, true] {
            let mut state = ready_game_in_main1(1);
            let pyroblast = put_in_hand(&mut state, PlayerId::P0, "Pyroblast");
            let target = if copied {
                put_spell_copy_on_stack_from_method(
                    &mut state,
                    PlayerId::P1,
                    "Deep Analysis",
                    CastMethodV4::Flashback,
                )
            } else {
                put_on_stack_with_method(
                    &mut state,
                    PlayerId::P1,
                    "Deep Analysis",
                    CastMethodV4::Flashback,
                )
            };

            step(&mut state, Action::CastSpell(pyroblast)).unwrap();
            step(&mut state, Action::ChooseSpellMode(0)).unwrap();
            step(&mut state, Action::ChooseTarget(Target::Object(target))).unwrap();
            pass_until_stack_resolves(&mut state);

            if copied {
                assert!(!state.exile.contains(&target));
                assert!(!state.players[1].graveyard.contains(&target));
                assert_eq!(state.objects.get(target).zone, Zone::Stack);
            } else {
                assert_eq!(state.objects.get(target).zone, Zone::Exile);
                assert!(state.exile.contains(&target));
                assert!(!state.players[1].graveyard.contains(&target));
            }
        }
    }

    #[test]
    fn spell_target_pools_include_instant_copies_but_exclude_all_nonspell_items() {
        let mut state = empty_game();
        let physical = put_on_stack(&mut state, PlayerId::P0, "Counterspell");
        let copy = put_spell_copy_on_stack(&mut state, PlayerId::P1, "Counterspell");

        let activated = put_on_battlefield(&mut state, PlayerId::P0, "Blood Token");
        state.stack.push(StackItem {
            kind: StackItemKind::ActivatedAbility,
            source: activated,
            controller: PlayerId::P0,
            targets: vec![],
            is_copy: false,
            inline_effect: Some(EffectOp::Sequence(vec![])),
            discarded: vec![],
            is_flashback: false,
            mode_chosen: 0,
            madness_offer: false,
            kicked: false,
            v4: StackStateV4::spell(CastMethodV4::Normal),
        });
        let triggered = put_on_battlefield(&mut state, PlayerId::P0, "Guttersnipe");
        state.stack.push(StackItem {
            kind: StackItemKind::TriggeredAbility,
            source: triggered,
            controller: PlayerId::P0,
            targets: vec![],
            is_copy: false,
            inline_effect: Some(EffectOp::Sequence(vec![])),
            discarded: vec![],
            is_flashback: false,
            mode_chosen: 0,
            madness_offer: false,
            kicked: false,
            v4: StackStateV4::default(),
        });
        let madness = put_in_graveyard(&mut state, PlayerId::P1, "Fiery Temper");
        state.stack.push(StackItem {
            kind: StackItemKind::MadnessOffer,
            source: madness,
            controller: PlayerId::P1,
            targets: vec![],
            is_copy: false,
            inline_effect: None,
            discarded: vec![],
            is_flashback: false,
            mode_chosen: 0,
            madness_offer: true,
            kicked: false,
            v4: StackStateV4::default(),
        });

        for spec in [
            TargetSpec::AnySpellOnStack,
            TargetSpec::InstantSpellOnStack,
            TargetSpec::BlueSpellOnStack,
        ] {
            let targets = legal_targets_for(spec, &[], &state);
            assert!(targets.contains(&Target::Object(physical)));
            assert!(targets.contains(&Target::Object(copy)));
            assert!(!targets.contains(&Target::Object(activated)));
            assert!(!targets.contains(&Target::Object(triggered)));
            assert!(!targets.contains(&Target::Object(madness)));
        }
    }

    #[test]
    fn exact_stack_counter_removes_one_trigger_incarnation_without_moving_its_source() {
        let mut state = empty_game();
        let source = put_on_battlefield(&mut state, PlayerId::P0, "Guttersnipe");
        let source_contract = AbilitySourceContractV4::capture(&state, source);
        let effect = (trigger::triggers_for(state.objects.get(source).card_def)[0].effect)();
        for stack_item_id in [StackItemId(90), StackItemId(91)] {
            state.stack.push(StackItem {
                kind: StackItemKind::TriggeredAbility,
                source,
                controller: PlayerId::P0,
                targets: vec![],
                is_copy: false,
                inline_effect: Some(effect.clone()),
                discarded: vec![],
                is_flashback: false,
                mode_chosen: 0,
                madness_offer: false,
                kicked: false,
                v4: StackStateV4 {
                    stack_item_id,
                    ability_source_contract: Some(source_contract),
                    ..StackStateV4::default()
                },
            });
        }

        let removed = counter_stack_item_by_id(&mut state, StackItemId(90))
            .expect("exact counter validates")
            .expect("named trigger remains live");

        assert_eq!(removed.v4.stack_item_id, StackItemId(90));
        assert_eq!(state.stack.len(), 1);
        assert_eq!(state.stack[0].v4.stack_item_id, StackItemId(91));
        assert_eq!(state.stack[0].source, source);
        assert_eq!(state.objects.get(source).zone, Zone::Battlefield);
        assert!(state.players[0].battlefield.contains(&source));
    }

    #[test]
    fn instant_spell_filter_preserves_raw_stack_order_across_mixed_spell_types() {
        let mut state = empty_game();
        assert!(legal_targets_for(TargetSpec::InstantSpellOnStack, &[], &state).is_empty());

        let creature = put_on_stack(&mut state, PlayerId::P0, "Guttersnipe");
        let sorcery = put_on_stack(&mut state, PlayerId::P1, "Faithless Looting");
        let red_instant = put_on_stack(&mut state, PlayerId::P0, "Lightning Bolt");
        let blue_instant_copy = put_spell_copy_on_stack(&mut state, PlayerId::P1, "Counterspell");

        assert_eq!(
            legal_targets_for(TargetSpec::AnySpellOnStack, &[], &state),
            vec![
                Target::Object(creature),
                Target::Object(sorcery),
                Target::Object(red_instant),
                Target::Object(blue_instant_copy),
            ]
        );
        assert_eq!(
            legal_targets_for(TargetSpec::InstantSpellOnStack, &[], &state),
            vec![
                Target::Object(red_instant),
                Target::Object(blue_instant_copy),
            ],
            "instant filtering must not reorder qualifying physical spells or copies"
        );
    }

    #[test]
    fn pyroblast_does_not_counter_a_non_blue_spell() {
        let mut state = ready_game_in_main1(1);
        let pyroblast = put_in_hand(&mut state, PlayerId::P0, "Pyroblast");
        let bolt = put_on_stack(&mut state, PlayerId::P1, "Lightning Bolt"); // red, not blue

        step(&mut state, Action::CastSpell(pyroblast)).unwrap();
        step(&mut state, Action::ChooseSpellMode(0)).unwrap();
        step(&mut state, Action::ChooseTarget(Target::Object(bolt))).unwrap();
        // Only resolve Pyroblast itself (the top of the stack) -- `bolt` was
        // planted directly via `put_on_stack` with no real targets of its
        // own, so actually letting it resolve too would panic on an empty
        // `ctx.targets`, unrelated to what this test is checking.
        pass_until_one_stack_item_resolves(&mut state);
        assert_eq!(
            state.objects.get(bolt).zone,
            Zone::Stack,
            "Pyroblast's counter-mode targets any spell but only actually counters a blue one"
        );
        assert_eq!(
            state.objects.get(pyroblast).zone,
            Zone::Graveyard,
            "Pyroblast itself still resolves normally even when its effect no-ops"
        );
    }

    #[test]
    fn pyroblast_mode2_destroys_a_blue_permanent() {
        let mut state = ready_game_in_main1(1);
        let pyroblast = put_in_hand(&mut state, PlayerId::P0, "Pyroblast");
        let serpent = put_on_battlefield(&mut state, PlayerId::P1, "Cryptic Serpent"); // blue creature

        step(&mut state, Action::CastSpell(pyroblast)).unwrap();
        match advance_until_decision(&mut state) {
            Decision::ChooseTargets { legal_targets, .. } => {
                assert_eq!(
                    state.engine.pending_cast.as_ref().unwrap().mode_chosen,
                    Some(1),
                    "destroy is the sole viable mode and keeps printed index 1"
                );
                assert!(legal_targets.contains(&Target::Object(serpent)));
                step(&mut state, Action::ChooseTarget(Target::Object(serpent))).unwrap();
            }
            other => panic!("expected ChooseTargets, got {other:?}"),
        }
        pass_until_stack_resolves(&mut state);
        assert_eq!(state.objects.get(serpent).zone, Zone::Graveyard);
    }

    #[test]
    fn red_elemental_blast_targeting_is_prefiltered_to_blue_only() {
        let mut state = ready_game_in_main1(1);
        let reb = put_in_hand(&mut state, PlayerId::P0, "Red Elemental Blast");
        let bolt = put_on_stack(&mut state, PlayerId::P1, "Lightning Bolt"); // red: illegal target
        let counterspell = put_on_stack(&mut state, PlayerId::P1, "Counterspell"); // blue: legal target

        step(&mut state, Action::CastSpell(reb)).unwrap();
        match advance_until_decision(&mut state) {
            Decision::ChooseTargets { legal_targets, .. } => {
                assert_eq!(
                    state.engine.pending_cast.as_ref().unwrap().mode_chosen,
                    Some(0),
                    "counter is the sole viable mode"
                );
                assert!(
                    !legal_targets.contains(&Target::Object(bolt)),
                    "REB's counter mode should never even offer a non-blue spell as a target"
                );
                assert!(legal_targets.contains(&Target::Object(counterspell)));
            }
            other => panic!("expected ChooseTargets, got {other:?}"),
        }
    }

    #[test]
    fn fizzle_when_the_targeted_spell_already_left_the_stack() {
        let mut state = ready_game_in_main1(1);
        let pyroblast = put_in_hand(&mut state, PlayerId::P0, "Pyroblast");
        let counterspell = put_on_stack(&mut state, PlayerId::P1, "Counterspell");

        step(&mut state, Action::CastSpell(pyroblast)).unwrap();
        step(&mut state, Action::ChooseSpellMode(0)).unwrap();
        step(
            &mut state,
            Action::ChooseTarget(Target::Object(counterspell)),
        )
        .unwrap();
        let _ = advance_until_decision(&mut state);
        assert!(state.engine.pending_cast.is_none());
        // The target resolves and leaves the stack before Pyroblast does
        // (simulated directly here; the kernel's real priority/stack
        // ordering can't actually reach this shape with this pool's cards,
        // but the guard should hold regardless of how it's reached).
        event::propose_and_commit(
            &mut state,
            ProposedEvent::zone_change(counterspell, Zone::Graveyard),
        );
        pass_until_stack_resolves(&mut state);
        // No crash, no double-move: Pyroblast itself still resolves to the
        // graveyard normally, and the fizzle guard is exercised via
        // EffectCond::TargetInZone, not a panic on a stale ObjectId.
        assert_eq!(state.objects.get(pyroblast).zone, Zone::Graveyard);
    }

    /// `Action::ActivateManaAbility` must reset `priority_passes` (already
    /// covered by the increment-13 fix root-caused against
    /// `game_20260713_002203_0026.txt`) *and* bump `mana_ability_
    /// activations`/stamp `last_mana_ability_activator`, the two fields
    /// `HarnessSurfaceV2`'s `DeclareAttackers`/`DeclareBlockers` combat
    /// throttle needs to detect a mid-round mana ability at all, since it
    /// never touches `state.stack` the way a cast/non-mana-activation does
    /// -- see `EngineState::mana_ability_activations`'s doc for the full
    /// root-cause (`game_20260713_002148_0003.txt` decision 34,
    /// `game_20260713_002202_0024.txt` decision 179).
    #[test]
    fn activate_mana_ability_resets_priority_and_stamps_the_activator() {
        let mut state = empty_game();
        let mountain = put_on_battlefield(&mut state, PlayerId::P0, "Mountain");
        state.step = Step::Main1;
        state.priority_player = PlayerId::P0;
        state.engine.priority_passes = [true, false];

        assert_eq!(state.engine.mana_ability_activations, 0);
        assert_eq!(state.engine.last_mana_ability_activator, None);

        step(&mut state, Action::ActivateManaAbility(mountain)).unwrap();

        assert_eq!(state.engine.priority_passes, [false, false]);
        assert_eq!(state.engine.mana_ability_activations, 1);
        assert_eq!(state.engine.last_mana_ability_activator, Some(PlayerId::P0));
    }

    // ================== Rally at the Hornburg increment ==================

    #[test]
    fn goblin_bushwhacker_kicked_pumps_the_team_and_grants_haste() {
        let mut state = ready_game_in_main1(2);
        let voldaren = put_on_battlefield(&mut state, PlayerId::P0, "Voldaren Epicure");
        let bushwhacker = put_in_hand(&mut state, PlayerId::P0, "Goblin Bushwhacker");

        step(&mut state, Action::CastSpell(bushwhacker)).unwrap();
        match advance_until_decision(&mut state) {
            Decision::ChooseKicker { player, spell } => {
                assert_eq!(player, PlayerId::P0);
                assert_eq!(spell, bushwhacker);
                step(&mut state, Action::ChooseKicker(true)).unwrap();
            }
            other => panic!("expected ChooseKicker, got {other:?}"),
        }
        pass_until_stack_resolves(&mut state);

        assert_eq!(
            effective_power(&state, bushwhacker),
            2,
            "1/1 base +1/+0 from its own kicked ETB"
        );
        assert_eq!(
            effective_power(&state, voldaren),
            2,
            "Voldaren Epicure is also a creature P0 controls"
        );
        assert!(
            can_attack(&state, bushwhacker),
            "kicked haste should let it attack despite just entering"
        );
        assert!(
            can_attack(&state, voldaren),
            "Voldaren Epicure should keep its own pre-existing ability to attack"
        );
    }

    #[test]
    fn goblin_bushwhacker_unaffordable_kicker_auto_resolves_unkicked() {
        let mut state = ready_game_in_main1(1);
        let bushwhacker = put_in_hand(&mut state, PlayerId::P0, "Goblin Bushwhacker");
        step(&mut state, Action::CastSpell(bushwhacker)).unwrap();
        // Only 1 Mountain in play: the combined {R}{R} kicked cost isn't
        // payable, so `Decision::ChooseKicker` is never even offered --
        // same "no real choice" shortcut as `ChooseCastMode`.
        assert!(
            !matches!(advance_until_decision(&mut state), Decision::ChooseKicker { .. }),
            "the matched-uniform policy must consume no leaf or physical decision for this forced unkicked cast"
        );
        pass_until_stack_resolves(&mut state);
        assert_eq!(effective_power(&state, bushwhacker), 1);
        assert!(
            !can_attack(&state, bushwhacker),
            "unkicked: no haste, and it just entered summoning sick"
        );
    }

    #[test]
    fn goblin_bushwhacker_declined_kicker_does_not_create_an_etb_trigger() {
        let mut state = ready_game_in_main1(2);
        let bushwhacker = put_in_hand(&mut state, PlayerId::P0, "Goblin Bushwhacker");

        step(&mut state, Action::CastSpell(bushwhacker)).unwrap();
        match advance_until_decision(&mut state) {
            Decision::ChooseKicker { .. } => step(&mut state, Action::ChooseKicker(false)).unwrap(),
            other => panic!("expected ChooseKicker, got {other:?}"),
        }

        match advance_until_decision(&mut state) {
            Decision::CastSpellOrPass { player, .. } => assert_eq!(player, PlayerId::P0),
            other => panic!("expected P0 priority after the cast, got {other:?}"),
        }
        step(&mut state, Action::Pass).unwrap();
        match advance_until_decision(&mut state) {
            Decision::CastSpellOrPass { player, .. } => assert_eq!(player, PlayerId::P1),
            other => panic!("expected P1 priority after P0 passes, got {other:?}"),
        }
        step(&mut state, Action::Pass).unwrap();

        match advance_until_decision(&mut state) {
            Decision::CastSpellOrPass { player, .. } => assert_eq!(player, PlayerId::P0),
            other => panic!("expected fresh P0 priority after Bushwhacker resolves, got {other:?}"),
        }
        assert!(
            state.stack.is_empty(),
            "603.4: declining Kicker means Bushwhacker's intervening-if ability never triggers"
        );
    }

    #[test]
    fn goblin_bushwhacker_declined_kicker_gets_no_pump_even_though_affordable() {
        let mut state = ready_game_in_main1(2);
        let bushwhacker = put_in_hand(&mut state, PlayerId::P0, "Goblin Bushwhacker");
        step(&mut state, Action::CastSpell(bushwhacker)).unwrap();
        match advance_until_decision(&mut state) {
            Decision::ChooseKicker { .. } => step(&mut state, Action::ChooseKicker(false)).unwrap(),
            other => panic!("expected ChooseKicker, got {other:?}"),
        }
        pass_until_stack_resolves(&mut state);
        assert_eq!(effective_power(&state, bushwhacker), 1);
        let tapped_lands = state.players[0]
            .battlefield
            .iter()
            .filter(|&&id| state.objects.get(id).tapped)
            .count();
        assert_eq!(
            tapped_lands, 1,
            "only the base {{R}} cost should have been paid, not the declined kicker too"
        );
    }

    #[test]
    fn rally_at_the_hornburg_creates_two_hasty_human_tokens_and_pumps_existing_humans_only() {
        let mut state = ready_game_in_main1(2);
        let human = put_on_battlefield(&mut state, PlayerId::P0, "Burning-Tree Emissary"); // Human Shaman
        let non_human = put_on_battlefield(&mut state, PlayerId::P0, "Goblin Tomb Raider"); // Goblin Pirate, no artifact in play
        let rally = put_in_hand(&mut state, PlayerId::P0, "Rally at the Hornburg");

        step(&mut state, Action::CastSpell(rally)).unwrap();
        pass_until_stack_resolves(&mut state);

        let token_def = card_id_by_name("Human Soldier Token").unwrap();
        let tokens: Vec<ObjectId> = state.players[0]
            .battlefield
            .iter()
            .copied()
            .filter(|&id| state.objects.get(id).card_def == token_def)
            .collect();
        assert_eq!(
            tokens.len(),
            2,
            "should create exactly two Human Soldier tokens"
        );

        assert!(
            has_effective_keyword(&state, human, Keywords::HASTE),
            "Burning-Tree Emissary is Human, should gain haste"
        );
        for &t in &tokens {
            assert!(
                has_effective_keyword(&state, t, Keywords::HASTE),
                "the tokens are Human Soldiers themselves"
            );
            assert!(
                can_attack(&state, t),
                "a hasty token should be able to attack the turn it's created"
            );
        }
        assert!(
            !has_effective_keyword(&state, non_human, Keywords::HASTE),
            "Goblin Tomb Raider isn't Human and controls no artifact here"
        );
    }

    #[test]
    fn goblin_tomb_raider_gets_plus_one_and_haste_only_while_controlling_an_artifact() {
        let mut state = empty_game();
        let raider = put_on_battlefield(&mut state, PlayerId::P0, "Goblin Tomb Raider");
        assert_eq!(effective_power(&state, raider), 1);
        assert_eq!(effective_toughness(&state, raider), 2);
        assert!(!has_effective_keyword(&state, raider, Keywords::HASTE));

        put_on_battlefield(&mut state, PlayerId::P0, "Great Furnace");
        assert_eq!(
            effective_power(&state, raider),
            2,
            "+1/+0 while controlling an artifact"
        );
        assert_eq!(
            effective_toughness(&state, raider),
            2,
            "the boost is +1/+0, toughness unaffected"
        );
        assert!(has_effective_keyword(&state, raider, Keywords::HASTE));
    }

    #[test]
    fn galvanic_blast_deals_only_2_without_metalcraft() {
        let mut state = ready_game_in_main1(1);
        let blast = put_in_hand(&mut state, PlayerId::P0, "Galvanic Blast");
        step(&mut state, Action::CastSpell(blast)).unwrap();
        step(
            &mut state,
            Action::ChooseTarget(Target::Player(PlayerId::P1)),
        )
        .unwrap();
        pass_until_stack_resolves(&mut state);
        assert_eq!(state.players[1].life, 18);
    }

    #[test]
    fn galvanic_blast_deals_4_with_metalcraft() {
        let mut state = ready_game_in_main1(1);
        put_on_battlefield(&mut state, PlayerId::P0, "Great Furnace");
        put_on_battlefield(&mut state, PlayerId::P0, "Clockwork Percussionist");
        put_on_battlefield(&mut state, PlayerId::P0, "Experimental Synthesizer");
        let blast = put_in_hand(&mut state, PlayerId::P0, "Galvanic Blast");
        step(&mut state, Action::CastSpell(blast)).unwrap();
        step(
            &mut state,
            Action::ChooseTarget(Target::Player(PlayerId::P1)),
        )
        .unwrap();
        pass_until_stack_resolves(&mut state);
        assert_eq!(
            state.players[1].life, 16,
            "3+ artifacts controlled: Metalcraft deals 4 instead of 2"
        );
    }

    #[test]
    fn end_the_festivities_hits_the_opponent_and_every_creature_they_control_only() {
        let mut state = ready_game_in_main1(1);
        let etf = put_in_hand(&mut state, PlayerId::P0, "End the Festivities");
        let p1_a = put_on_battlefield(&mut state, PlayerId::P1, "Voldaren Epicure");
        let p1_b = put_on_battlefield(&mut state, PlayerId::P1, "Goblin Tomb Raider");
        let p0_creature = put_on_battlefield(&mut state, PlayerId::P0, "Voldaren Epicure");

        step(&mut state, Action::CastSpell(etf)).unwrap();
        pass_until_stack_resolves(&mut state);

        assert_eq!(state.players[1].life, 19);
        assert_eq!(
            state.objects.get(p1_a).zone,
            Zone::Graveyard,
            "the 1-toughness creature dies after taking 1 damage"
        );
        assert!(state.players[1].graveyard.contains(&p1_a));
        assert_eq!(state.objects.get(p1_b).damage, 1);
        assert_eq!(
            state.objects.get(p0_creature).damage,
            0,
            "only the opponent and their own creatures are hit"
        );
        assert_eq!(
            state.players[0].life, 20,
            "the caster themself isn't damaged"
        );
    }

    #[test]
    fn experimental_synthesizer_sac_ability_creates_a_samurai_token() {
        let mut state = ready_game_in_main1(3);
        let synth = put_on_battlefield(&mut state, PlayerId::P0, "Experimental Synthesizer");
        match advance_until_decision(&mut state) {
            Decision::CastSpellOrPass {
                activatable_abilities,
                ..
            } => {
                assert!(
                    activatable_abilities.contains(&(synth, 0)),
                    "should be activatable at sorcery speed with an empty stack"
                );
            }
            other => panic!("expected CastSpellOrPass, got {other:?}"),
        }
        step(&mut state, Action::ActivateAbility(synth, 0)).unwrap();
        pass_until_stack_resolves(&mut state);

        let samurai_def = card_id_by_name("Samurai Token").unwrap();
        assert!(
            state.players[0]
                .battlefield
                .iter()
                .any(|&id| state.objects.get(id).card_def == samurai_def),
            "should have created a Samurai Token"
        );
        assert!(
            !state.players[0].battlefield.contains(&synth),
            "Experimental Synthesizer sacrificed itself to pay the cost"
        );
    }

    #[test]
    fn synthesizer_generic_payment_preserves_red_for_bushwhacker() {
        let mut state = ready_game_in_main1(0);
        let synth = put_on_battlefield(&mut state, PlayerId::P0, "Experimental Synthesizer");
        let bushwhacker = put_in_hand(&mut state, PlayerId::P0, "Goblin Bushwhacker");
        state.players[0].mana_pool[mana::ManaColor::R.pool_index()] = 3;
        state.players[0].mana_pool[mana::ManaColor::G.pool_index()] = 1;

        step(&mut state, Action::ActivateAbility(synth, 0)).unwrap();
        let decision = pass_until_stack_resolves(&mut state);
        assert_eq!(
            state.players[0].mana_pool[mana::ManaColor::R.pool_index()],
            1
        );
        assert_eq!(
            state.players[0].mana_pool[mana::ManaColor::G.pool_index()],
            0
        );

        match decision {
            Decision::CastSpellOrPass {
                castable_spells, ..
            } => assert!(
                castable_spells.contains(&bushwhacker),
                "the automatic generic payment must leave the red mana needed for Bushwhacker"
            ),
            other => panic!("expected CastSpellOrPass, got {other:?}"),
        }
    }

    #[test]
    fn experimental_synthesizer_sac_ability_is_not_offered_with_a_nonempty_stack() {
        let mut state = ready_game_in_main1(3);
        let synth = put_on_battlefield(&mut state, PlayerId::P0, "Experimental Synthesizer");
        put_on_stack(&mut state, PlayerId::P0, "Lightning Bolt");
        match advance_until_decision(&mut state) {
            Decision::CastSpellOrPass {
                activatable_abilities,
                ..
            } => {
                assert!(
                    !activatable_abilities.contains(&(synth, 0)),
                    "sorcery-speed-only: illegal with something already on the stack"
                );
            }
            other => panic!("expected CastSpellOrPass, got {other:?}"),
        }
    }

    fn set_library_top(state: &mut GameState, player: PlayerId, card_name: &str) -> ObjectId {
        let card_id =
            card_id_by_name(card_name).unwrap_or_else(|| panic!("{card_name} not in CARD_DEFS"));
        let obj_id = state.objects.push(crate::state::GameObject {
            card_def: card_id,
            name: card_name.to_string(),
            owner: player,
            controller: player,
            zone: Zone::Library,
            tapped: false,
            summoning_sick: false,
            damage: 0,
            counters: Default::default(),
            attachments: Vec::new(),
            v4: ObjectStateV4::from_card_def(card_id),
            spell_copy_origin: None,
            plotted_turn: None,
            zone_change_count: 0,
        });
        state.players[player.index()].library.insert(0, obj_id);
        obj_id
    }

    #[test]
    fn experimental_synthesizer_etb_impulse_draw_expires_at_end_of_turn() {
        // 2 Mountains: one pays for Experimental Synthesizer itself, the
        // second proves the exiled Lightning Bolt is *actually* affordable
        // (not just legal-if-you-had-mana) when checking `castable_spells`.
        let mut state = ready_game_in_main1(2);
        let bolt_id = set_library_top(&mut state, PlayerId::P0, "Lightning Bolt");
        let synth = put_in_hand(&mut state, PlayerId::P0, "Experimental Synthesizer");

        step(&mut state, Action::CastSpell(synth)).unwrap();
        pass_until_stack_resolves(&mut state);

        assert_eq!(
            state.objects.get(bolt_id).zone,
            Zone::Exile,
            "the ETB trigger should have exiled the top card"
        );
        assert!(active_permission_for(PlayerId::P0, bolt_id, &state).is_some());
        match advance_until_decision(&mut state) {
            Decision::CastSpellOrPass {
                castable_spells, ..
            } => assert!(
                castable_spells.contains(&bolt_id),
                "should be playable this turn"
            ),
            other => panic!("expected CastSpellOrPass, got {other:?}"),
        }

        while state.step != Step::Cleanup {
            advance_step(&mut state);
        }
        assert!(
            active_permission_for(PlayerId::P0, bolt_id, &state).is_none(),
            "an 'until end of turn' window closes at this turn's own cleanup"
        );
    }

    #[test]
    fn clockwork_percussionist_dies_impulse_draw_survives_the_opponents_turn_and_expires_after_owners_next_turn(
    ) {
        let mut state = ready_game_in_main1(0);
        let bolt_id = set_library_top(&mut state, PlayerId::P0, "Lightning Bolt");
        let percussionist = put_on_battlefield(&mut state, PlayerId::P0, "Clockwork Percussionist");

        event::propose_and_commit(
            &mut state,
            ProposedEvent::zone_change(percussionist, Zone::Graveyard),
        );
        collect_and_queue_triggers(&mut state);
        loop {
            if state.engine.pending_triggers.is_empty() && state.stack.is_empty() {
                break;
            }
            match advance_until_decision(&mut state) {
                Decision::CastSpellOrPass { .. } => step(&mut state, Action::Pass).unwrap(),
                other => panic!("unexpected decision resolving the dies trigger: {other:?}"),
            }
        }

        assert_eq!(
            state.objects.get(bolt_id).zone,
            Zone::Exile,
            "the dies trigger should have exiled the top card"
        );
        assert!(active_permission_for(PlayerId::P0, bolt_id, &state).is_some());

        // Rest of this same turn (P0's): survives its own cleanup.
        while state.step != Step::Cleanup {
            advance_step(&mut state);
        }
        assert!(
            active_permission_for(PlayerId::P0, bolt_id, &state).is_some(),
            "survives this turn's own cleanup (not the owner's next turn yet)"
        );

        // -> P1's Untap: the opponent's whole turn.
        advance_step(&mut state);
        assert_eq!(state.active_player, PlayerId::P1);
        assert!(active_permission_for(PlayerId::P0, bolt_id, &state).is_some());
        while state.step != Step::Cleanup {
            advance_step(&mut state);
        }
        assert!(
            active_permission_for(PlayerId::P0, bolt_id, &state).is_some(),
            "survives the opponent's whole turn too"
        );

        // -> P0's Untap: the owner's own next turn begins.
        advance_step(&mut state);
        assert_eq!(state.active_player, PlayerId::P0);
        assert!(
            active_permission_for(PlayerId::P0, bolt_id, &state).is_some(),
            "the owner's next turn has begun; still open"
        );

        // Through to that turn's own cleanup, where it finally expires.
        while state.step != Step::Cleanup {
            advance_step(&mut state);
        }
        assert!(
            active_permission_for(PlayerId::P0, bolt_id, &state).is_none(),
            "expires at the owner's next turn's own cleanup"
        );
    }

    #[test]
    fn combat_lethal_damage_queues_clockwork_percussionist_dies_trigger_immediately() {
        let mut state = empty_game();
        let percussionist = put_on_battlefield(&mut state, PlayerId::P0, "Clockwork Percussionist");
        let blocker = put_on_battlefield(&mut state, PlayerId::P1, "Human Soldier Token");
        state.engine.combat.attackers = vec![percussionist];
        state.engine.combat.blocked_by = vec![(percussionist, vec![blocker])];

        deal_combat_damage(&mut state);

        assert_eq!(state.objects.get(percussionist).zone, Zone::Graveyard);
        assert_eq!(
            state.engine.pending_triggers.len(),
            1,
            "the SBA-created death event must be collected before the next priority window"
        );
        assert_eq!(state.engine.pending_triggers[0].source, percussionist);
    }

    #[test]
    fn trigger_collection_preserves_pre_sba_etb_and_collects_sba_created_death_together() {
        let mut state = empty_game();
        let epicure = put_in_hand(&mut state, PlayerId::P1, "Voldaren Epicure");
        event::propose_and_commit(
            &mut state,
            ProposedEvent::zone_change(epicure, Zone::Battlefield),
        );
        let percussionist = put_on_battlefield(&mut state, PlayerId::P0, "Clockwork Percussionist");

        // Model two creatures that are lethal at the next SBA check. The
        // Epicure's ETB event already happened, so that trigger must survive
        // even though the source leaves the battlefield during the check.
        // Percussionist's dies event, by contrast, is created by the check
        // itself and must still be collected before the next priority window.
        let epicure_toughness = effective_toughness(&state, epicure) as u16;
        let percussionist_toughness = effective_toughness(&state, percussionist) as u16;
        state.objects.get_mut(epicure).damage = epicure_toughness;
        state.objects.get_mut(percussionist).damage = percussionist_toughness;

        collect_and_queue_triggers(&mut state);

        assert_eq!(state.objects.get(epicure).zone, Zone::Graveyard);
        assert_eq!(state.objects.get(percussionist).zone, Zone::Graveyard);
        assert_eq!(state.engine.pending_triggers.len(), 2);
        assert_eq!(
            state.engine.pending_triggers[0].source, percussionist,
            "the active player's SBA-created dies trigger is ordered first"
        );
        assert_eq!(state.engine.pending_triggers[1].source, epicure, "the nonactive player's pre-SBA ETB trigger survives its source dying and is ordered second");
    }

    // ================== external-review corrections ==================

    #[test]
    fn burning_tree_emissary_produces_its_etb_mana_exactly_once_and_is_never_a_mana_source() {
        let mut state = ready_game_in_main1(2);
        let bte = put_in_hand(&mut state, PlayerId::P0, "Burning-Tree Emissary");
        step(&mut state, Action::CastSpell(bte)).unwrap();
        pass_until_stack_resolves(&mut state);

        // Exactly the ETB's {R}{G} floating (the 2 Mountains that paid the
        // cost are spent, not left over -- `mana::pay_plan` nets them to
        // zero before this trigger's own `AddMana` ever runs).
        assert_eq!(
            state.players[0].mana_pool[crate::mana::ManaColor::R.pool_index()],
            1
        );
        assert_eq!(
            state.players[0].mana_pool[crate::mana::ManaColor::G.pool_index()],
            1
        );

        assert!(
            !state.objects.get(bte).tapped,
            "the mana came from its ETB trigger, not from tapping it"
        );
        assert!(
            available_mana_abilities(PlayerId::P0, &state).is_empty(),
            "Burning-Tree Emissary has no repeatable tap ability"
        );
        let sources = crate::mana::gather_sources(PlayerId::P0, &state);
        assert!(
            !sources.iter().any(|s| s.id == bte),
            "the mana solver must never treat it as a tappable source (root-caused this increment)"
        );
    }

    #[test]
    fn chain_lightning_asks_before_an_unpayable_copy_attempt_then_finishes() {
        let mut state = ready_game_in_main1(1);
        let bolt = put_in_hand(&mut state, PlayerId::P0, "Chain Lightning");
        step(&mut state, Action::CastSpell(bolt)).unwrap();
        step(
            &mut state,
            Action::ChooseTarget(Target::Player(PlayerId::P1)),
        )
        .unwrap();
        match pass_until_stack_resolves(&mut state) {
            Decision::ChooseSpellCopyPayment { player, spell } => {
                assert_eq!(player, PlayerId::P1);
                assert_eq!(spell, bolt);
            }
            other => panic!("expected the XMage-parity payment attempt, got {other:?}"),
        }

        assert_eq!(
            state.players[1].life, 17,
            "the mandatory 3 damage always happens"
        );
        assert_eq!(state.stack.len(), 1, "the parent stays live mid-resolution");
        assert_eq!(state.objects.get(bolt).zone, Zone::Stack);

        step(&mut state, Action::ChooseSpellCopyPayment(true)).unwrap();
        assert!(state.engine.pending_spell_copy.is_none());
        assert!(state.stack.is_empty());
        assert_eq!(state.objects.get(bolt).zone, Zone::Graveyard);
        assert_eq!(state.players[0].graveyard, vec![bolt]);
    }

    #[test]
    fn chain_lightning_unpayable_decline_has_the_same_no_copy_terminal_shape() {
        let mut state = ready_game_in_main1(1);
        let chain = put_in_hand(&mut state, PlayerId::P0, "Chain Lightning");
        step(&mut state, Action::CastSpell(chain)).unwrap();
        step(
            &mut state,
            Action::ChooseTarget(Target::Player(PlayerId::P1)),
        )
        .unwrap();
        assert!(matches!(
            pass_until_stack_resolves(&mut state),
            Decision::ChooseSpellCopyPayment {
                player: PlayerId::P1,
                spell,
            } if spell == chain
        ));

        step(&mut state, Action::ChooseSpellCopyPayment(false)).unwrap();
        assert!(state.engine.pending_spell_copy.is_none());
        assert!(state.stack.is_empty());
        assert_eq!(state.objects.get(chain).zone, Zone::Graveyard);
        assert_eq!(state.players[0].graveyard, vec![chain]);
        assert_eq!(
            state.objects.len(),
            2,
            "the declined unpayable branch creates no virtual copy"
        );
    }

    #[test]
    fn blood_activation_with_one_discardable_card_is_forced_before_policy_dispatch() {
        let mut state = ready_game_in_main1(1);
        let blood = put_on_battlefield(&mut state, PlayerId::P0, "Blood Token");
        let discarded = put_in_hand(&mut state, PlayerId::P0, "Lightning Bolt");

        step(&mut state, Action::ActivateAbility(blood, 0)).unwrap();
        let next = advance_until_decision(&mut state);
        assert!(
            !matches!(next, Decision::Discard { .. }),
            "a one-card Blood discard must auto-resolve without a policy decision"
        );
        assert!(state.players[0].hand.is_empty());
        assert_eq!(state.players[0].graveyard, vec![discarded]);
        assert!(state.engine.pending_discard.is_none());
        assert!(state.engine.pending_activation.is_none());
        assert_eq!(
            state.stack.len(),
            1,
            "the paid Blood ability is now on stack"
        );
    }

    #[test]
    fn chain_lightning_affordable_decline_spends_nothing_and_creates_no_copy() {
        let mut state = ready_game_in_main1(1);
        let bolt = put_in_hand(&mut state, PlayerId::P0, "Chain Lightning");
        let red1 = put_on_battlefield(&mut state, PlayerId::P1, "Great Furnace");
        let red2 = put_on_battlefield(&mut state, PlayerId::P1, "Great Furnace");

        step(&mut state, Action::CastSpell(bolt)).unwrap();
        step(
            &mut state,
            Action::ChooseTarget(Target::Player(PlayerId::P1)),
        )
        .unwrap();
        assert!(matches!(
            pass_until_stack_resolves(&mut state),
            Decision::ChooseSpellCopyPayment {
                player: PlayerId::P1,
                ..
            }
        ));
        step(&mut state, Action::ChooseSpellCopyPayment(false)).unwrap();

        assert!(!state.objects.get(red1).tapped);
        assert!(!state.objects.get(red2).tapped);
        assert!(state.stack.is_empty());
        assert_eq!(state.players[0].graveyard, vec![bolt]);
        assert_eq!(
            state.objects.len(),
            4,
            "declining allocates no virtual object"
        );
    }

    #[test]
    fn chain_lightning_pay_keep_target_creates_one_recursive_copy_without_casting_it() {
        let mut state = ready_game_in_main1(1);
        let bolt = put_in_hand(&mut state, PlayerId::P0, "Chain Lightning");
        let red1 = put_on_battlefield(&mut state, PlayerId::P1, "Great Furnace");
        let red2 = put_on_battlefield(&mut state, PlayerId::P1, "Great Furnace");
        let arena_before_copy = state.objects.len();

        step(&mut state, Action::CastSpell(bolt)).unwrap();
        step(
            &mut state,
            Action::ChooseTarget(Target::Player(PlayerId::P1)),
        )
        .unwrap();
        assert!(matches!(
            pass_until_stack_resolves(&mut state),
            Decision::ChooseSpellCopyPayment { .. }
        ));
        step(&mut state, Action::ChooseSpellCopyPayment(true)).unwrap();

        assert!(state.objects.get(red1).tapped && state.objects.get(red2).tapped);
        assert_eq!(state.objects.len(), arena_before_copy + 1);
        assert_eq!(state.stack.len(), 2);
        let copy = state.stack.last().unwrap().source;
        assert!(state.stack.last().unwrap().is_copy);
        assert_eq!(state.stack.last().unwrap().controller, PlayerId::P1);
        assert_eq!(
            state.stack.last().unwrap().targets,
            vec![Target::Player(PlayerId::P1)]
        );
        assert_eq!(
            state.stack.last().unwrap().v4.target_contracts,
            state.stack[0].v4.target_contracts,
            "a copy inherits the parent's exact target contract before any retarget"
        );
        assert!(matches!(
            advance_until_decision(&mut state),
            Decision::ChooseSpellCopyRetarget { player: PlayerId::P1, copy: c } if c == copy
        ));
        step(&mut state, Action::ChooseSpellCopyRetarget(false)).unwrap();

        assert_eq!(state.objects.get(bolt).zone, Zone::Graveyard);
        assert_eq!(state.stack.len(), 1);
        assert_eq!(state.stack[0].source, copy);
        assert!(
            matches!(
                advance_until_decision(&mut state),
                Decision::CastSpellOrPass {
                    player: PlayerId::P0,
                    ..
                }
            ),
            "normal active-player priority opens before the copy resolves"
        );

        assert!(matches!(
            pass_until_stack_resolves(&mut state),
            Decision::ChooseSpellCopyPayment { player: PlayerId::P1, spell } if spell == copy
        ));
        assert_eq!(state.players[1].life, 14);
        step(&mut state, Action::ChooseSpellCopyPayment(false)).unwrap();
        assert!(state.stack.is_empty());
        assert_eq!(state.players[0].graveyard, vec![bolt]);
        assert!(!state.players[1].graveyard.contains(&copy));
        assert_eq!(
            state
                .engine
                .event_history
                .iter()
                .filter(|event| matches!(event, CommittedEvent::SpellCast { .. }))
                .count(),
            1,
            "the copy is created, not cast"
        );
    }

    #[test]
    fn chain_lightning_retarget_waits_for_choice_before_lethal_sba() {
        let mut state = ready_game_in_main1(1);
        let bolt = put_in_hand(&mut state, PlayerId::P0, "Chain Lightning");
        let creature = put_on_battlefield(&mut state, PlayerId::P1, "Clockwork Percussionist");
        put_on_battlefield(&mut state, PlayerId::P1, "Great Furnace");
        put_on_battlefield(&mut state, PlayerId::P1, "Great Furnace");

        step(&mut state, Action::CastSpell(bolt)).unwrap();
        step(&mut state, Action::ChooseTarget(Target::Object(creature))).unwrap();
        assert!(matches!(
            pass_until_stack_resolves(&mut state),
            Decision::ChooseSpellCopyPayment {
                player: PlayerId::P1,
                ..
            }
        ));
        assert!(state.players[1].battlefield.contains(&creature));
        assert_eq!(state.objects.get(creature).damage, 3);

        step(&mut state, Action::ChooseSpellCopyPayment(true)).unwrap();
        step(&mut state, Action::ChooseSpellCopyRetarget(true)).unwrap();
        let copy = state
            .engine
            .pending_spell_copy
            .as_ref()
            .unwrap()
            .copy_source
            .unwrap();
        match advance_until_decision(&mut state) {
            Decision::ChooseTargets {
                player,
                spell,
                legal_targets,
                ..
            } => {
                assert_eq!(player, PlayerId::P1);
                assert_eq!(spell, copy);
                assert!(legal_targets.contains(&Target::Player(PlayerId::P0)));
            }
            other => panic!("expected copy target selection, got {other:?}"),
        }
        let before_invalid = state.state_hash();
        assert!(step(&mut state, Action::ChooseTarget(Target::Object(bolt))).is_err());
        assert_eq!(
            state.state_hash(),
            before_invalid,
            "an invalid copy target is non-mutating"
        );
        step(
            &mut state,
            Action::ChooseTarget(Target::Player(PlayerId::P0)),
        )
        .unwrap();

        assert!(!state.players[1].battlefield.contains(&creature));
        assert!(state.players[1].graveyard.contains(&creature));
        assert_eq!(state.stack.len(), 1);
        assert_eq!(state.stack[0].source, copy);
        assert_eq!(state.stack[0].targets, vec![Target::Player(PlayerId::P0)]);
    }

    /// Regression for the dominant CP7 panel void class (2026-09-01): a
    /// player taken to negative life by Chain Lightning must still be able to
    /// pay the zero-life `{R}{R}` copy cost (rule 119.4: paying zero life is
    /// always allowed) and reach the retarget decision before lethal SBA.
    /// The kernel previously rejected the payment plan because
    /// `0 <= -1` is false, finished resolution, and reported a terminal
    /// while XMage was still waiting on the retarget.
    #[test]
    fn chain_lightning_copy_payment_is_allowed_at_negative_life() {
        let mut state = ready_game_in_main1(1);
        let bolt = put_in_hand(&mut state, PlayerId::P0, "Chain Lightning");
        put_on_battlefield(&mut state, PlayerId::P1, "Great Furnace");
        put_on_battlefield(&mut state, PlayerId::P1, "Great Furnace");
        state.players[1].life = 2;

        step(&mut state, Action::CastSpell(bolt)).unwrap();
        step(
            &mut state,
            Action::ChooseTarget(Target::Player(PlayerId::P1)),
        )
        .unwrap();
        assert!(matches!(
            pass_until_stack_resolves(&mut state),
            Decision::ChooseSpellCopyPayment {
                player: PlayerId::P1,
                ..
            }
        ));
        assert_eq!(
            state.players[1].life, -1,
            "damage lands before the copy offer"
        );
        assert!(
            !state.players[1].has_lost,
            "SBA must not fire mid-resolution"
        );

        step(&mut state, Action::ChooseSpellCopyPayment(true)).unwrap();
        match advance_until_decision(&mut state) {
            Decision::ChooseSpellCopyRetarget {
                player: PlayerId::P1,
                ..
            } => {}
            other => panic!("expected the copy retarget decision, got {other:?}"),
        }
        assert!(
            !state.players[1].has_lost,
            "the payment succeeded at negative life; no terminal yet"
        );
    }

    #[test]
    fn chain_lightning_copy_of_copy_uses_fresh_ids_and_alternating_controllers() {
        let mut state = ready_game_in_main1(0);
        let bolt = put_in_hand(&mut state, PlayerId::P0, "Chain Lightning");
        for _ in 0..3 {
            put_on_battlefield(&mut state, PlayerId::P0, "Great Furnace");
        }
        for _ in 0..2 {
            put_on_battlefield(&mut state, PlayerId::P1, "Great Furnace");
        }

        step(&mut state, Action::CastSpell(bolt)).unwrap();
        step(
            &mut state,
            Action::ChooseTarget(Target::Player(PlayerId::P1)),
        )
        .unwrap();
        let _ = pass_until_stack_resolves(&mut state);
        step(&mut state, Action::ChooseSpellCopyPayment(true)).unwrap();
        let copy1 = state
            .engine
            .pending_spell_copy
            .as_ref()
            .unwrap()
            .copy_source
            .unwrap();
        step(&mut state, Action::ChooseSpellCopyRetarget(true)).unwrap();
        step(
            &mut state,
            Action::ChooseTarget(Target::Player(PlayerId::P0)),
        )
        .unwrap();

        let _ = pass_until_stack_resolves(&mut state);
        step(&mut state, Action::ChooseSpellCopyPayment(true)).unwrap();
        let copy2 = state
            .engine
            .pending_spell_copy
            .as_ref()
            .unwrap()
            .copy_source
            .unwrap();
        assert_ne!(copy1, copy2);
        assert_eq!(state.stack.last().unwrap().controller, PlayerId::P0);
        step(&mut state, Action::ChooseSpellCopyRetarget(true)).unwrap();
        step(
            &mut state,
            Action::ChooseTarget(Target::Player(PlayerId::P1)),
        )
        .unwrap();

        let _ = pass_until_stack_resolves(&mut state);
        step(&mut state, Action::ChooseSpellCopyPayment(false)).unwrap();
        assert_eq!([state.players[0].life, state.players[1].life], [17, 14]);
        assert!(state.stack.is_empty());
        assert_eq!(state.players[0].graveyard, vec![bolt]);
        assert!(!state.players[0].graveyard.contains(&copy2));
        assert!(!state.players[1].graveyard.contains(&copy1));
    }

    #[test]
    fn chain_lightning_copy_with_a_now_illegal_inherited_target_fizzles_and_ceases() {
        let mut state = ready_game_in_main1(1);
        let bolt = put_in_hand(&mut state, PlayerId::P0, "Chain Lightning");
        let creature = put_on_battlefield(&mut state, PlayerId::P1, "Clockwork Percussionist");
        put_on_battlefield(&mut state, PlayerId::P1, "Great Furnace");
        put_on_battlefield(&mut state, PlayerId::P1, "Great Furnace");

        step(&mut state, Action::CastSpell(bolt)).unwrap();
        step(&mut state, Action::ChooseTarget(Target::Object(creature))).unwrap();
        assert!(matches!(
            pass_until_stack_resolves(&mut state),
            Decision::ChooseSpellCopyPayment { .. }
        ));
        step(&mut state, Action::ChooseSpellCopyPayment(true)).unwrap();
        let copy = state
            .engine
            .pending_spell_copy
            .as_ref()
            .unwrap()
            .copy_source
            .unwrap();
        step(&mut state, Action::ChooseSpellCopyRetarget(false)).unwrap();

        assert_eq!(state.objects.get(creature).zone, Zone::Graveyard);
        assert_eq!(state.stack.len(), 1);
        assert_eq!(state.stack[0].source, copy);
        assert_eq!(state.players[1].life, 20);
        let event_count = state.engine.event_history.len();

        let next = pass_until_stack_resolves(&mut state);
        assert!(matches!(next, Decision::CastSpellOrPass { .. }));
        assert!(state.stack.is_empty());
        assert!(state.engine.pending_spell_copy.is_none());
        assert_eq!(
            state.players[1].life, 20,
            "the illegal-target copy deals no damage"
        );
        assert_eq!(state.objects.get(copy).zone, Zone::Stack);
        assert!(!state.players[1].graveyard.contains(&copy));
        assert!(
            !state.engine.event_history[event_count..]
                .iter()
                .any(|event| matches!(event, CommittedEvent::ZoneChange { object, .. } if *object == copy)),
            "a fizzled copy ceases without a zone-change event"
        );
    }

    #[test]
    fn chain_lightning_pending_copy_choices_round_trip_through_snapshots() {
        let mut state = ready_game_in_main1(1);
        let bolt = put_in_hand(&mut state, PlayerId::P0, "Chain Lightning");
        put_on_battlefield(&mut state, PlayerId::P1, "Great Furnace");
        put_on_battlefield(&mut state, PlayerId::P1, "Great Furnace");

        step(&mut state, Action::CastSpell(bolt)).unwrap();
        step(
            &mut state,
            Action::ChooseTarget(Target::Player(PlayerId::P1)),
        )
        .unwrap();
        assert!(matches!(
            pass_until_stack_resolves(&mut state),
            Decision::ChooseSpellCopyPayment { .. }
        ));
        let payment_snapshot = state.snapshot();

        step(&mut state, Action::ChooseSpellCopyPayment(true)).unwrap();
        let copy = state
            .engine
            .pending_spell_copy
            .as_ref()
            .unwrap()
            .copy_source
            .unwrap();
        step(&mut state, Action::ChooseSpellCopyRetarget(true)).unwrap();
        let target_snapshot = state.snapshot();
        step(
            &mut state,
            Action::ChooseTarget(Target::Player(PlayerId::P0)),
        )
        .unwrap();
        assert_eq!(
            state.stack.last().unwrap().targets,
            vec![Target::Player(PlayerId::P0)]
        );
        assert_eq!(
            state.stack.last().unwrap().v4.target_contracts,
            vec![StackTargetContractV4::Player(PlayerId::P0)],
            "an explicit retarget captures a fresh matching contract"
        );
        let expected_hash = state.state_hash();

        state.restore(&target_snapshot);
        assert!(matches!(
            state.engine.pending_spell_copy.as_ref().map(|p| p.stage),
            Some(SpellCopyStage::Target)
        ));
        step(
            &mut state,
            Action::ChooseTarget(Target::Player(PlayerId::P0)),
        )
        .unwrap();
        assert_eq!(state.state_hash(), expected_hash);

        state.restore(&payment_snapshot);
        step(&mut state, Action::ChooseSpellCopyPayment(true)).unwrap();
        assert_eq!(
            state
                .engine
                .pending_spell_copy
                .as_ref()
                .unwrap()
                .copy_source,
            Some(copy),
            "restoring before allocation must reproduce the same stable arena id"
        );
        step(&mut state, Action::ChooseSpellCopyRetarget(true)).unwrap();
        step(
            &mut state,
            Action::ChooseTarget(Target::Player(PlayerId::P0)),
        )
        .unwrap();
        assert_eq!(state.state_hash(), expected_hash);
    }

    #[test]
    fn chain_lightning_copy_target_contract_tampering_fails_before_payment_or_retarget() {
        let mut state = ready_game_in_main1(1);
        let bolt = put_in_hand(&mut state, PlayerId::P0, "Chain Lightning");
        let red1 = put_on_battlefield(&mut state, PlayerId::P1, "Great Furnace");
        let red2 = put_on_battlefield(&mut state, PlayerId::P1, "Great Furnace");
        step(&mut state, Action::CastSpell(bolt)).unwrap();
        step(
            &mut state,
            Action::ChooseTarget(Target::Player(PlayerId::P1)),
        )
        .unwrap();
        assert!(matches!(
            pass_until_stack_resolves(&mut state),
            Decision::ChooseSpellCopyPayment { .. }
        ));

        state.engine.pending_spell_copy.as_mut().unwrap().player = PlayerId::P0;
        let wrong_player = state.clone();
        assert!(step(&mut state, Action::ChooseSpellCopyPayment(true)).is_err());
        assert_eq!(state, wrong_player);
        assert!(!state.objects.get(red1).tapped && !state.objects.get(red2).tapped);
        state.engine.pending_spell_copy.as_mut().unwrap().player = PlayerId::P1;

        state
            .engine
            .pending_spell_copy
            .as_mut()
            .unwrap()
            .inherited_target = Target::Player(PlayerId::P0);
        let corrupted = state.clone();
        assert!(step(&mut state, Action::ChooseSpellCopyPayment(true)).is_err());
        assert_eq!(state, corrupted);
        assert!(!state.objects.get(red1).tapped && !state.objects.get(red2).tapped);

        state
            .engine
            .pending_spell_copy
            .as_mut()
            .unwrap()
            .inherited_target = Target::Player(PlayerId::P1);
        step(&mut state, Action::ChooseSpellCopyPayment(true)).unwrap();
        let copy = state
            .engine
            .pending_spell_copy
            .as_ref()
            .unwrap()
            .copy_source
            .unwrap();
        state
            .stack
            .iter_mut()
            .find(|item| item.source == copy)
            .unwrap()
            .targets = vec![Target::Player(PlayerId::P0)];
        let resolving_source = state
            .engine
            .pending_spell_copy
            .as_ref()
            .unwrap()
            .resolving_source;
        let decision = advance_until_decision(&mut state);
        assert!(
            matches!(
                decision,
                Decision::Halted {
                    mechanic: UnsupportedMechanic::InvalidEffectContinuation,
                    source,
            } if source == resolving_source
            ),
            "got {decision:?}"
        );
    }

    #[test]
    fn chain_copy_rejects_structurally_coherent_contract_tamper_and_wrong_producer() {
        let mut state = ready_game_in_main1(1);
        let chain = put_in_hand(&mut state, PlayerId::P0, "Chain Lightning");
        let creature = put_on_battlefield(&mut state, PlayerId::P1, "Clockwork Percussionist");
        let red1 = put_on_battlefield(&mut state, PlayerId::P1, "Great Furnace");
        let red2 = put_on_battlefield(&mut state, PlayerId::P1, "Great Furnace");
        step(&mut state, Action::CastSpell(chain)).unwrap();
        step(&mut state, Action::ChooseTarget(Target::Object(creature))).unwrap();
        assert!(matches!(
            pass_until_stack_resolves(&mut state),
            Decision::ChooseSpellCopyPayment { .. }
        ));

        let canonical = state.clone();
        let mut forged_flashback = canonical.clone();
        forged_flashback.stack.last_mut().unwrap().is_flashback = true;
        let before = forged_flashback.clone();
        assert!(step(&mut forged_flashback, Action::ChooseSpellCopyPayment(false)).is_err());
        assert_eq!(
            forged_flashback, before,
            "an impossible flashback parent must not exile or otherwise mutate the physical spell"
        );
        assert_eq!(forged_flashback.objects.get(chain).zone, Zone::Stack);

        let contract = &mut state.stack.last_mut().unwrap().v4.target_contracts[0];
        let StackTargetContractV4::Object { zone, .. } = contract else {
            panic!("Chain Lightning's creature target must carry an object contract");
        };
        *zone = Zone::Graveyard;
        state
            .engine
            .pending_spell_copy
            .as_mut()
            .unwrap()
            .inherited_target_contract = Some(*contract);
        let before = state.clone();
        assert!(step(&mut state, Action::ChooseSpellCopyPayment(true)).is_err());
        assert_eq!(
            state, before,
            "coherent contract tamper must be non-mutating"
        );
        assert!(!state.objects.get(red1).tapped && !state.objects.get(red2).tapped);

        let mut bolt_state = ready_game_in_main1(1);
        let bolt = put_in_hand(&mut bolt_state, PlayerId::P0, "Lightning Bolt");
        let red1 = put_on_battlefield(&mut bolt_state, PlayerId::P1, "Great Furnace");
        let red2 = put_on_battlefield(&mut bolt_state, PlayerId::P1, "Great Furnace");
        step(&mut bolt_state, Action::CastSpell(bolt)).unwrap();
        step(
            &mut bolt_state,
            Action::ChooseTarget(Target::Player(PlayerId::P1)),
        )
        .unwrap();
        assert!(matches!(
            advance_until_decision(&mut bolt_state),
            Decision::CastSpellOrPass { .. }
        ));
        let contract = bolt_state.stack.last().unwrap().v4.target_contracts[0];
        let resolving_stack_item = bolt_state.stack.last().unwrap().v4.stack_item_id;
        bolt_state.engine.pending_spell_copy = Some(PendingSpellCopy {
            resolving_stack_item,
            resolving_source: bolt,
            resolving_source_zone_change_count: bolt_state.objects.get(bolt).zone_change_count,
            player: PlayerId::P1,
            inherited_target: Target::Player(PlayerId::P1),
            inherited_target_contract: Some(contract),
            stage: SpellCopyStage::Payment,
            copy_source: None,
            copy_stack_item: None,
        });
        let before = bolt_state.clone();
        assert!(step(&mut bolt_state, Action::ChooseSpellCopyPayment(true)).is_err());
        assert_eq!(
            bolt_state, before,
            "an injected pending slot on a non-producer spell must be non-mutating"
        );
        assert!(!bolt_state.objects.get(red1).tapped && !bolt_state.objects.get(red2).tapped);
    }

    #[test]
    fn chain_copy_rejects_non_top_payment_parent_and_parent_copy_alias() {
        let mut state = ready_game_in_main1(1);
        let chain = put_in_hand(&mut state, PlayerId::P0, "Chain Lightning");
        let red1 = put_on_battlefield(&mut state, PlayerId::P1, "Great Furnace");
        let red2 = put_on_battlefield(&mut state, PlayerId::P1, "Great Furnace");
        step(&mut state, Action::CastSpell(chain)).unwrap();
        step(
            &mut state,
            Action::ChooseTarget(Target::Player(PlayerId::P1)),
        )
        .unwrap();
        assert!(matches!(
            pass_until_stack_resolves(&mut state),
            Decision::ChooseSpellCopyPayment { .. }
        ));

        let mut lower_item = state.stack.last().unwrap().clone();
        let lower_source = state.objects.push(state.objects.get(chain).clone());
        lower_item.source = lower_source;
        state.stack.insert(0, lower_item);
        state
            .engine
            .pending_spell_copy
            .as_mut()
            .unwrap()
            .resolving_source = lower_source;
        let before = state.clone();
        assert!(step(&mut state, Action::ChooseSpellCopyPayment(true)).is_err());
        assert_eq!(
            state, before,
            "a lower compatible parent must not receive payment"
        );
        assert!(!state.objects.get(red1).tapped && !state.objects.get(red2).tapped);

        let mut state = ready_game_in_main1(1);
        let chain = put_in_hand(&mut state, PlayerId::P0, "Chain Lightning");
        put_on_battlefield(&mut state, PlayerId::P1, "Great Furnace");
        put_on_battlefield(&mut state, PlayerId::P1, "Great Furnace");
        step(&mut state, Action::CastSpell(chain)).unwrap();
        step(
            &mut state,
            Action::ChooseTarget(Target::Player(PlayerId::P1)),
        )
        .unwrap();
        let _ = pass_until_stack_resolves(&mut state);
        step(&mut state, Action::ChooseSpellCopyPayment(true)).unwrap();
        let copy = state
            .engine
            .pending_spell_copy
            .as_ref()
            .unwrap()
            .copy_source
            .unwrap();
        state
            .engine
            .pending_spell_copy
            .as_mut()
            .unwrap()
            .resolving_source = copy;
        let before = state.clone();
        assert!(step(&mut state, Action::ChooseSpellCopyRetarget(false)).is_err());
        assert_eq!(
            state, before,
            "aliasing the parent to its copy must not cease or resolve either item"
        );
    }

    #[test]
    fn chain_copy_rejects_noncanonical_copy_item_and_source_before_finish() {
        let mut state = ready_game_in_main1(1);
        let chain = put_in_hand(&mut state, PlayerId::P0, "Chain Lightning");
        put_on_battlefield(&mut state, PlayerId::P1, "Great Furnace");
        put_on_battlefield(&mut state, PlayerId::P1, "Great Furnace");
        step(&mut state, Action::CastSpell(chain)).unwrap();
        step(
            &mut state,
            Action::ChooseTarget(Target::Player(PlayerId::P1)),
        )
        .unwrap();
        let _ = pass_until_stack_resolves(&mut state);
        step(&mut state, Action::ChooseSpellCopyPayment(true)).unwrap();
        let copy = state
            .engine
            .pending_spell_copy
            .as_ref()
            .unwrap()
            .copy_source
            .unwrap();
        let canonical = state.clone();

        let mut wrong_zone = canonical.clone();
        wrong_zone.objects.get_mut(copy).zone = Zone::Graveyard;
        let before = wrong_zone.clone();
        assert!(step(&mut wrong_zone, Action::ChooseSpellCopyRetarget(false)).is_err());
        assert_eq!(
            wrong_zone, before,
            "a malformed copy source zone must fail before clearing or resolving"
        );

        let mut injected_effect = canonical.clone();
        injected_effect.stack.last_mut().unwrap().inline_effect = Some(EffectOp::GainLife {
            player: PlayerRef::Controller,
            amount: 99,
        });
        let before = injected_effect.clone();
        assert!(step(&mut injected_effect, Action::ChooseSpellCopyRetarget(false)).is_err());
        assert_eq!(
            injected_effect, before,
            "a restored copy with an injected inline effect must be non-mutating"
        );

        let mut dirty_object = canonical;
        dirty_object.objects.get_mut(copy).damage = 1;
        let before = dirty_object.clone();
        assert!(step(&mut dirty_object, Action::ChooseSpellCopyRetarget(false)).is_err());
        assert_eq!(
            dirty_object, before,
            "a noncanonical virtual copy object must be non-mutating"
        );
    }

    #[test]
    fn chain_copy_rejects_reused_parent_and_target_incarnations() {
        let mut parent_reused = ready_game_in_main1(1);
        let chain = put_in_hand(&mut parent_reused, PlayerId::P0, "Chain Lightning");
        step(&mut parent_reused, Action::CastSpell(chain)).unwrap();
        step(
            &mut parent_reused,
            Action::ChooseTarget(Target::Player(PlayerId::P1)),
        )
        .unwrap();
        let _ = pass_until_stack_resolves(&mut parent_reused);
        let original_item = parent_reused.stack.last().unwrap().clone();
        let bound_generation = parent_reused
            .engine
            .pending_spell_copy
            .as_ref()
            .unwrap()
            .resolving_source_zone_change_count;
        event::propose_and_commit(
            &mut parent_reused,
            ProposedEvent::zone_change(chain, Zone::Graveyard),
        );
        move_to_stack(&mut parent_reused, chain, Zone::Graveyard);
        parent_reused.stack.push(original_item);
        assert_ne!(
            parent_reused.objects.get(chain).zone_change_count,
            bound_generation
        );
        let before = parent_reused.clone();
        assert!(step(&mut parent_reused, Action::ChooseSpellCopyPayment(false)).is_err());
        assert_eq!(
            parent_reused, before,
            "a later Chain incarnation reusing the arena id must not finish the old continuation"
        );

        let mut target_reused = ready_game_in_main1(1);
        let chain = put_in_hand(&mut target_reused, PlayerId::P0, "Chain Lightning");
        let target =
            put_on_battlefield(&mut target_reused, PlayerId::P1, "Clockwork Percussionist");
        step(&mut target_reused, Action::CastSpell(chain)).unwrap();
        step(
            &mut target_reused,
            Action::ChooseTarget(Target::Object(target)),
        )
        .unwrap();
        let _ = pass_until_stack_resolves(&mut target_reused);
        event::propose_and_commit(
            &mut target_reused,
            ProposedEvent::zone_change(target, Zone::Graveyard),
        );
        event::propose_and_commit(
            &mut target_reused,
            ProposedEvent::zone_change(target, Zone::Battlefield),
        );
        target_reused.players[PlayerId::P1.index()]
            .battlefield
            .retain(|&object| object != target);
        target_reused.players[PlayerId::P0.index()]
            .battlefield
            .push(target);
        target_reused.objects.get_mut(target).controller = PlayerId::P0;
        target_reused
            .engine
            .pending_spell_copy
            .as_mut()
            .unwrap()
            .player = PlayerId::P0;
        let before = target_reused.clone();
        assert!(step(&mut target_reused, Action::ChooseSpellCopyPayment(false)).is_err());
        assert_eq!(
            target_reused, before,
            "a coherently redirected later target incarnation must be non-mutating"
        );
    }

    #[test]
    fn chain_lightning_illegal_target_fizzles_before_damage_or_copy_offer() {
        let mut state = ready_game_in_main1(1);
        let bolt = put_in_hand(&mut state, PlayerId::P0, "Chain Lightning");
        let creature = put_on_battlefield(&mut state, PlayerId::P1, "Clockwork Percussionist");
        step(&mut state, Action::CastSpell(bolt)).unwrap();
        step(&mut state, Action::ChooseTarget(Target::Object(creature))).unwrap();
        let _ = advance_until_decision(&mut state);
        assert!(state.engine.pending_cast.is_none());
        event::propose_and_commit(
            &mut state,
            ProposedEvent::zone_change(creature, Zone::Graveyard),
        );

        let _ = pass_until_stack_resolves(&mut state);
        assert_eq!(state.objects.get(creature).damage, 0);
        assert!(state.engine.pending_spell_copy.is_none());
        assert_eq!(state.objects.get(bolt).zone, Zone::Graveyard);
        assert!(state.stack.is_empty());
    }

    #[test]
    fn goblin_bushwhacker_kicked_pump_is_tagged_with_both_layers_and_a_timestamp() {
        let mut state = ready_game_in_main1(2);
        let bushwhacker = put_in_hand(&mut state, PlayerId::P0, "Goblin Bushwhacker");
        step(&mut state, Action::CastSpell(bushwhacker)).unwrap();
        match advance_until_decision(&mut state) {
            Decision::ChooseKicker { .. } => step(&mut state, Action::ChooseKicker(true)).unwrap(),
            other => panic!("expected ChooseKicker, got {other:?}"),
        }
        pass_until_stack_resolves(&mut state);

        assert_eq!(state.engine.until_end_of_turn.len(), 1);
        match &state.engine.until_end_of_turn[0] {
            UntilEndOfTurnEffect::ResolvedSetEffect {
                layer,
                duration,
                object_ids,
                ..
            } => {
                assert!(layer.has(Layers::POWER_TOUGHNESS), "grants +1/+0");
                assert!(layer.has(Layers::ABILITY_ADDING), "grants Haste");
                assert_eq!(*duration, EffectDuration::EndOfTurn);
                assert!(
                    object_ids.contains(&bushwhacker),
                    "611.2c: the resolving effect's own locked-in set includes itself"
                );
            }
            other => panic!("expected ResolvedSetEffect, got {other:?}"),
        }
    }

    #[test]
    fn mid_cast_source_contracts_round_trip_for_normal_flashback_escape_plot_and_madness() {
        let mut normal = ready_game_in_main1(1);
        let bolt = put_in_hand(&mut normal, PlayerId::P0, "Lightning Bolt");
        step(&mut normal, Action::CastSpell(bolt)).unwrap();

        let mut flashback = ready_game_in_main1(1);
        let dart = put_in_graveyard(&mut flashback, PlayerId::P0, "Lava Dart");
        step(&mut flashback, Action::CastSpell(dart)).unwrap();

        let mut escape = ready_game_in_main1(0);
        for _ in 0..3 {
            put_on_battlefield(&mut escape, PlayerId::P0, "Island");
        }
        put_on_battlefield(&mut escape, PlayerId::P1, "Cryptic Serpent");
        let sleep = put_in_graveyard(&mut escape, PlayerId::P0, "Sleep of the Dead");
        for name in ["Lightning Bolt", "Fireblast", "Counterspell"] {
            put_in_graveyard(&mut escape, PlayerId::P0, name);
        }
        step(&mut escape, Action::CastSpell(sleep)).unwrap();

        let mut plotted = ready_game_in_main1(0);
        plotted.turn = 2;
        let robbery = put_in_hand(&mut plotted, PlayerId::P0, "Highway Robbery");
        event::propose_and_commit(
            &mut plotted,
            ProposedEvent::zone_change(robbery, Zone::Exile),
        );
        plotted.objects.get_mut(robbery).plotted_turn = Some(1);
        step(&mut plotted, Action::CastSpell(robbery)).unwrap();

        let mut madness = ready_game_in_main1(1);
        let temper = put_in_hand(&mut madness, PlayerId::P0, "Fiery Temper");
        event::propose_and_commit(
            &mut madness,
            ProposedEvent::zone_change(temper, Zone::Exile),
        );
        begin_cast_ex(
            &mut madness,
            PlayerId::P0,
            temper,
            Some(CastMethodV4::Madness),
        );

        for (state, expected_method, expected_route) in [
            (normal, CastMethodV4::Normal, SpellCastRouteV4::Hand),
            (
                flashback,
                CastMethodV4::Flashback,
                SpellCastRouteV4::GraveyardFlashback,
            ),
            (
                escape,
                CastMethodV4::Escape,
                SpellCastRouteV4::GraveyardEscape,
            ),
            (
                plotted,
                CastMethodV4::Plotted,
                SpellCastRouteV4::Plotted { plotted_turn: 1 },
            ),
            (madness, CastMethodV4::Madness, SpellCastRouteV4::Madness),
        ] {
            let json = serde_json::to_vec(&state).unwrap();
            let restored: GameState = serde_json::from_slice(&json).unwrap();
            assert_eq!(restored, state);
            let pending = restored.engine.pending_cast.as_ref().unwrap();
            validate_pending_cast(&restored, pending).unwrap();
            assert_eq!(pending.source_contract.cast_method, expected_method);
            assert_eq!(
                pending
                    .source_contract
                    .spell_cast_origin
                    .expect("physical pending cast has frozen origin")
                    .route,
                expected_route
            );
            assert_eq!(
                restored.stack.last().unwrap().v4.source_contract,
                Some(pending.source_contract)
            );
        }
    }

    #[test]
    fn owner_only_cast_methods_reject_coherent_cross_owner_source_tamper() {
        let mut normal = ready_game_in_main1(1);
        let bolt = put_in_hand(&mut normal, PlayerId::P0, "Lightning Bolt");
        step(&mut normal, Action::CastSpell(bolt)).unwrap();

        let mut flashback = ready_game_in_main1(1);
        let dart = put_in_graveyard(&mut flashback, PlayerId::P0, "Lava Dart");
        step(&mut flashback, Action::CastSpell(dart)).unwrap();

        let mut escape = ready_game_in_main1(0);
        for _ in 0..3 {
            put_on_battlefield(&mut escape, PlayerId::P0, "Island");
        }
        let escape_target = put_on_battlefield(&mut escape, PlayerId::P1, "Cryptic Serpent");
        let sleep = put_in_graveyard(&mut escape, PlayerId::P0, "Sleep of the Dead");
        for name in ["Lightning Bolt", "Fireblast", "Counterspell"] {
            put_in_graveyard(&mut escape, PlayerId::P0, name);
        }
        step(&mut escape, Action::CastSpell(sleep)).unwrap();

        let mut plotted = ready_game_in_main1(0);
        plotted.turn = 2;
        let robbery = put_in_hand(&mut plotted, PlayerId::P0, "Highway Robbery");
        event::propose_and_commit(
            &mut plotted,
            ProposedEvent::zone_change(robbery, Zone::Exile),
        );
        plotted.objects.get_mut(robbery).plotted_turn = Some(1);
        step(&mut plotted, Action::CastSpell(robbery)).unwrap();

        let mut madness = ready_game_in_main1(1);
        let temper = put_in_hand(&mut madness, PlayerId::P0, "Fiery Temper");
        event::propose_and_commit(
            &mut madness,
            ProposedEvent::zone_change(temper, Zone::Exile),
        );
        begin_cast_ex(
            &mut madness,
            PlayerId::P0,
            temper,
            Some(CastMethodV4::Madness),
        );

        for (mut state, action) in [
            (normal, Action::ChooseTarget(Target::Player(PlayerId::P1))),
            (
                flashback,
                Action::ChooseTarget(Target::Player(PlayerId::P1)),
            ),
            (escape, Action::ChooseTarget(Target::Object(escape_target))),
            (plotted, Action::Pass),
            (madness, Action::ChooseTarget(Target::Player(PlayerId::P1))),
        ] {
            let source = state.engine.pending_cast.as_ref().unwrap().spell;
            state.objects.get_mut(source).owner = PlayerId::P1;
            state
                .engine
                .pending_cast
                .as_mut()
                .unwrap()
                .source_contract
                .owner = PlayerId::P1;
            state
                .stack
                .last_mut()
                .unwrap()
                .v4
                .source_contract
                .as_mut()
                .unwrap()
                .owner = PlayerId::P1;
            assert!(
                validate_pending_cast(&state, state.engine.pending_cast.as_ref().unwrap()).is_err()
            );
            let before = state.clone();
            assert!(step(&mut state, action).is_err());
            assert_eq!(state, before);
        }
    }

    #[test]
    fn forged_plot_markers_and_dirty_placeholder_fail_before_action_mutation() {
        for plotted_turn in [3, 4] {
            let mut forged_plot = ready_game_in_main1(0);
            forged_plot.turn = 3;
            let robbery = put_in_hand(&mut forged_plot, PlayerId::P0, "Highway Robbery");
            event::propose_and_commit(
                &mut forged_plot,
                ProposedEvent::zone_change(robbery, Zone::Exile),
            );
            forged_plot.objects.get_mut(robbery).plotted_turn = Some(plotted_turn);
            let before = forged_plot.clone();
            assert!(step(&mut forged_plot, Action::CastSpell(robbery)).is_err());
            assert_eq!(forged_plot, before);
        }

        let mut non_plot = ready_game_in_main1(1);
        non_plot.turn = 3;
        let bolt = put_in_hand(&mut non_plot, PlayerId::P0, "Lightning Bolt");
        event::propose_and_commit(&mut non_plot, ProposedEvent::zone_change(bolt, Zone::Exile));
        non_plot.objects.get_mut(bolt).plotted_turn = Some(1);
        let before = non_plot.clone();
        assert!(step(&mut non_plot, Action::CastSpell(bolt)).is_err());
        assert_eq!(non_plot, before);

        let mut state = ready_game_in_main1(1);
        let bolt = put_in_hand(&mut state, PlayerId::P0, "Lightning Bolt");
        step(&mut state, Action::CastSpell(bolt)).unwrap();
        state.stack.last_mut().unwrap().targets = vec![Target::Player(PlayerId::P1)];
        let before = state.clone();
        assert!(step(
            &mut state,
            Action::ChooseTarget(Target::Player(PlayerId::P1))
        )
        .is_err());
        assert_eq!(state, before, "placeholder tamper must be non-mutating");
    }

    #[test]
    fn lava_dart_cost_target_cannot_precede_its_spell_target() {
        let mut state = ready_game_in_main1(0);
        let mountain = put_on_battlefield(&mut state, PlayerId::P0, "Mountain");
        put_on_battlefield(&mut state, PlayerId::P0, "Mountain");
        let dart = put_in_graveyard(&mut state, PlayerId::P0, "Lava Dart");
        step(&mut state, Action::CastSpell(dart)).unwrap();

        let before = state.clone();
        assert!(step(&mut state, Action::ChooseCostTarget(mountain)).is_err());
        assert_eq!(state, before);
        assert!(state
            .engine
            .pending_cast
            .as_ref()
            .unwrap()
            .sacrifice_chosen
            .is_empty());
    }

    #[test]
    fn fireblast_losing_remaining_candidates_aborts_without_partial_payment() {
        let mut state = ready_game_in_main1(0);
        let first = put_on_battlefield(&mut state, PlayerId::P0, "Mountain");
        let second = put_on_battlefield(&mut state, PlayerId::P0, "Mountain");
        let fireblast = put_in_hand(&mut state, PlayerId::P0, "Fireblast");
        step(&mut state, Action::CastSpell(fireblast)).unwrap();
        step(
            &mut state,
            Action::ChooseTarget(Target::Player(PlayerId::P1)),
        )
        .unwrap();
        assert!(matches!(
            advance_until_decision(&mut state),
            Decision::ChooseCostTargets { .. }
        ));
        let mut stale_action = state.clone();
        event::propose_and_commit(
            &mut stale_action,
            ProposedEvent::zone_change(second, Zone::Graveyard),
        );
        let before = stale_action.clone();
        assert!(step(&mut stale_action, Action::ChooseCostTarget(first)).is_err());
        assert_eq!(stale_action, before);

        step(&mut state, Action::ChooseCostTarget(first)).unwrap();
        event::propose_and_commit(
            &mut state,
            ProposedEvent::zone_change(second, Zone::Graveyard),
        );

        let _ = advance_until_decision(&mut state);
        assert!(state.engine.pending_cast.is_none());
        assert_eq!(state.objects.get(fireblast).zone, Zone::Hand);
        assert!(state.players[0].battlefield.contains(&first));
        assert_eq!(state.objects.get(first).zone, Zone::Battlefield);
    }

    #[test]
    fn highway_optional_sacrifice_with_no_remaining_candidate_halts_without_recursing() {
        let mut state = ready_game_in_main1(0);
        let first = put_on_battlefield(&mut state, PlayerId::P0, "Mountain");
        let second = put_on_battlefield(&mut state, PlayerId::P0, "Mountain");
        let robbery = put_in_hand(&mut state, PlayerId::P0, "Highway Robbery");
        step(&mut state, Action::CastSpell(robbery)).unwrap();
        assert!(matches!(
            pass_until_stack_resolves(&mut state),
            Decision::ChooseOptionalCost { .. }
        ));
        step(
            &mut state,
            Action::ChooseOptionalCost(OptionalCostChoice::SacrificeLand),
        )
        .unwrap();
        event::propose_and_commit(
            &mut state,
            ProposedEvent::zone_change(first, Zone::Graveyard),
        );
        event::propose_and_commit(
            &mut state,
            ProposedEvent::zone_change(second, Zone::Graveyard),
        );

        assert!(matches!(
            advance_until_decision(&mut state),
            Decision::Halted {
                mechanic: UnsupportedMechanic::InvalidEffectContinuation,
                source,
            } if source == robbery
        ));
    }


    #[test]
    fn frozen_spell_target_cannot_be_retyped_as_an_ability_before_counter_resolution() {
        let mut state = ready_game_in_main1(0);
        put_on_battlefield(&mut state, PlayerId::P0, "Island");
        put_on_battlefield(&mut state, PlayerId::P0, "Island");
        let target = put_on_stack(&mut state, PlayerId::P1, "Lightning Bolt");
        let counter = put_in_hand(&mut state, PlayerId::P0, "Counterspell");
        step(&mut state, Action::CastSpell(counter)).unwrap();
        step(&mut state, Action::ChooseTarget(Target::Object(target))).unwrap();
        let _ = advance_until_decision(&mut state);
        let target_item = state
            .stack
            .iter_mut()
            .find(|item| item.source == target)
            .unwrap();
        target_item.kind = StackItemKind::TriggeredAbility;
        target_item.v4 = StackStateV4::default();
        target_item.inline_effect = None;
        state.engine.priority_passes = [true, true];

        assert!(matches!(
            advance_until_decision(&mut state),
            Decision::Halted {
                mechanic: UnsupportedMechanic::InvalidEffectContinuation,
                source,
            } if source == counter
        ));
        assert_eq!(state.objects.get(target).zone, Zone::Stack);
    }

    #[test]
    fn nonowner_exile_permission_binds_caster_control_and_abort_restores_owner() {
        let mut success = ready_game_in_main1(1);
        let bolt = put_in_hand(&mut success, PlayerId::P1, "Lightning Bolt");
        event::propose_and_commit(&mut success, ProposedEvent::zone_change(bolt, Zone::Exile));
        success.engine.exile_play_permissions.push(PlayPermission {
            object: bolt,
            holder: PlayerId::P0,
            zone_change_generation: success.objects.get(bolt).zone_change_count,
            play_or_cast: PlayOrCast::Cast,
            expiry: PlayPermissionExpiry::EndOfTurn,
        });
        step(&mut success, Action::CastSpell(bolt)).unwrap();
        assert_eq!(success.objects.get(bolt).controller, PlayerId::P0);
        assert_eq!(
            success
                .engine
                .pending_cast
                .as_ref()
                .unwrap()
                .source_contract
                .controller,
            PlayerId::P0
        );
        let permission_origin = success
            .engine
            .pending_cast
            .as_ref()
            .unwrap()
            .source_contract
            .spell_cast_origin
            .expect("an Exile cast freezes its exact authorization route");
        assert!(matches!(
            permission_origin.route,
            SpellCastRouteV4::ExilePermission {
                holder: PlayerId::P0,
                permission_zone_change_count,
            } if permission_zone_change_count == permission_origin.origin_zone_change_count
        ));
        assert_eq!(permission_origin.origin_zone, Zone::Exile);
        assert_eq!(permission_origin.finalized_method, None);
        step(
            &mut success,
            Action::ChooseTarget(Target::Player(PlayerId::P1)),
        )
        .unwrap();
        let _ = advance_until_decision(&mut success);
        assert!(success.engine.pending_cast.is_none());
        assert_eq!(success.stack.last().unwrap().controller, PlayerId::P0);
        assert_eq!(
            success
                .objects
                .get(bolt)
                .v4
                .spell_cast_origin
                .unwrap()
                .finalized_method,
            Some(CastMethodV4::Normal)
        );

        let mut aborted = ready_game_in_main1(1);
        let land = aborted.players[0].battlefield[0];
        let bolt = put_in_hand(&mut aborted, PlayerId::P1, "Lightning Bolt");
        event::propose_and_commit(&mut aborted, ProposedEvent::zone_change(bolt, Zone::Exile));
        aborted.engine.exile_play_permissions.push(PlayPermission {
            object: bolt,
            holder: PlayerId::P0,
            zone_change_generation: aborted.objects.get(bolt).zone_change_count,
            play_or_cast: PlayOrCast::Cast,
            expiry: PlayPermissionExpiry::EndOfTurn,
        });
        step(&mut aborted, Action::CastSpell(bolt)).unwrap();
        aborted.objects.get_mut(land).tapped = true;
        step(
            &mut aborted,
            Action::ChooseTarget(Target::Player(PlayerId::P1)),
        )
        .unwrap();
        let _ = advance_until_decision(&mut aborted);
        assert!(aborted.engine.pending_cast.is_none());
        assert_eq!(aborted.objects.get(bolt).zone, Zone::Exile);
        assert_eq!(aborted.objects.get(bolt).controller, PlayerId::P1);
        assert!(aborted.exile.contains(&bolt));
    }


    #[test]
    fn pending_cast_rejects_unrelated_actions_without_orphaning_its_placeholder() {
        let mut state = ready_game_in_main1(2);
        let first = put_in_hand(&mut state, PlayerId::P0, "Lightning Bolt");
        let second = put_in_hand(&mut state, PlayerId::P0, "Lightning Bolt");
        step(&mut state, Action::CastSpell(first)).unwrap();

        for action in [Action::CastSpell(second), Action::Pass] {
            let mut attempt = state.clone();
            let before = attempt.clone();
            assert!(step(&mut attempt, action).is_err());
            assert_eq!(attempt, before);
        }
        assert_eq!(state.stack.len(), 1);
        assert_eq!(state.stack[0].source, first);
        assert_eq!(state.engine.pending_cast.as_ref().unwrap().spell, first);
    }

    #[test]
    fn fireblast_cast_mode_cannot_precede_targeting() {
        let mut state = ready_game_in_main1(6);
        let fireblast = put_in_hand(&mut state, PlayerId::P0, "Fireblast");
        step(&mut state, Action::CastSpell(fireblast)).unwrap();
        let before = state.clone();
        assert!(step(&mut state, Action::ChooseCastMode(CastMode::Alternative)).is_err());
        assert_eq!(state, before);
    }

    #[test]
    fn pending_activation_and_bound_discard_reject_unrelated_actions_without_mutation() {
        let mut activation = ready_game_in_main1(2);
        let lorien = put_in_hand(&mut activation, PlayerId::P0, "Lorien Revealed");
        let bolt = put_in_hand(&mut activation, PlayerId::P0, "Lightning Bolt");
        step(&mut activation, Action::ActivateAbility(lorien, 0)).unwrap();
        for action in [
            Action::Pass,
            Action::CastSpell(bolt),
            Action::ActivateAbility(lorien, 0),
        ] {
            let mut attempt = activation.clone();
            let before = attempt.clone();
            assert!(step(&mut attempt, action).is_err());
            assert_eq!(attempt, before);
        }

        let mut discard = ready_game_in_main1(1);
        let blood = put_on_battlefield(&mut discard, PlayerId::P0, "Blood Token");
        let bolt = put_in_hand(&mut discard, PlayerId::P0, "Lightning Bolt");
        put_in_hand(&mut discard, PlayerId::P0, "Fiery Temper");
        put_in_hand(&mut discard, PlayerId::P0, "Lava Dart");
        step(&mut discard, Action::ActivateAbility(blood, 0)).unwrap();
        assert!(matches!(
            advance_until_decision(&mut discard),
            Decision::Discard { .. }
        ));
        for action in [
            Action::Pass,
            Action::CastSpell(bolt),
            Action::ActivateAbility(blood, 0),
        ] {
            let mut attempt = discard.clone();
            let before = attempt.clone();
            assert!(step(&mut attempt, action).is_err());
            assert_eq!(attempt, before);
        }
    }


    #[test]
    fn orphaned_finish_discard_rejects_every_action_without_mutation() {
        let mut cast = ready_game_in_main1(3);
        let grab = put_in_hand(&mut cast, PlayerId::P0, "Grab the Prize");
        let bolt = put_in_hand(&mut cast, PlayerId::P0, "Lightning Bolt");
        put_in_hand(&mut cast, PlayerId::P0, "Fiery Temper");
        step(&mut cast, Action::CastSpell(grab)).unwrap();
        assert!(matches!(
            advance_until_decision(&mut cast),
            Decision::Discard { .. }
        ));
        cast.engine.pending_cast = None;
        for action in [Action::Pass, Action::CastSpell(bolt)] {
            let mut attempt = cast.clone();
            let before = attempt.clone();
            assert!(step(&mut attempt, action).is_err());
            assert_eq!(attempt, before);
        }

        let mut activation = ready_game_in_main1(1);
        let blood = put_on_battlefield(&mut activation, PlayerId::P0, "Blood Token");
        let bolt = put_in_hand(&mut activation, PlayerId::P0, "Lightning Bolt");
        put_in_hand(&mut activation, PlayerId::P0, "Fiery Temper");
        step(&mut activation, Action::ActivateAbility(blood, 0)).unwrap();
        assert!(matches!(
            advance_until_decision(&mut activation),
            Decision::Discard { .. }
        ));
        activation.engine.pending_activation = None;
        for action in [Action::Pass, Action::CastSpell(bolt)] {
            let mut attempt = activation.clone();
            let before = attempt.clone();
            assert!(step(&mut attempt, action).is_err());
            assert_eq!(attempt, before);
        }
    }




    #[test]
    fn old_spell_target_ignores_a_later_incarnations_nonspell_stack_item() {
        let mut state = ready_game_in_main1(0);
        put_on_battlefield(&mut state, PlayerId::P0, "Island");
        put_on_battlefield(&mut state, PlayerId::P0, "Island");
        let target = put_on_stack(&mut state, PlayerId::P1, "Lightning Bolt");
        let counter = put_in_hand(&mut state, PlayerId::P0, "Counterspell");
        step(&mut state, Action::CastSpell(counter)).unwrap();
        step(&mut state, Action::ChooseTarget(Target::Object(target))).unwrap();
        assert!(matches!(
            advance_until_decision(&mut state),
            Decision::CastSpellOrPass { .. }
        ));

        let target_index = state
            .stack
            .iter()
            .position(|item| item.source == target)
            .unwrap();
        state.stack.remove(target_index);
        event::propose_and_commit(
            &mut state,
            ProposedEvent::zone_change(target, Zone::Battlefield),
        );
        state.stack.insert(
            0,
            StackItem {
                kind: StackItemKind::TriggeredAbility,
                source: target,
                controller: PlayerId::P1,
                targets: vec![],
                is_copy: false,
                inline_effect: Some(EffectOp::Sequence(vec![])),
                discarded: vec![],
                is_flashback: false,
                mode_chosen: 0,
                madness_offer: false,
                kicked: false,
                v4: StackStateV4::default(),
            },
        );
        state.engine.priority_passes = [true, true];
        assert!(matches!(
            advance_until_decision(&mut state),
            Decision::CastSpellOrPass { .. }
        ));
        assert!(state.engine.halted.is_none());
        assert_eq!(state.objects.get(counter).zone, Zone::Graveyard);
        assert_eq!(state.stack.last().map(|item| item.source), Some(target));
    }

    #[test]
    fn plot_marker_is_exact_incarnation_provenance_and_survives_only_an_aborted_retry() {
        let mut resolved = ready_game_in_main1(4);
        resolved.turn = 2;
        let robbery = put_in_hand(&mut resolved, PlayerId::P0, "Highway Robbery");
        event::propose_and_commit(
            &mut resolved,
            ProposedEvent::zone_change(robbery, Zone::Exile),
        );
        resolved.objects.get_mut(robbery).plotted_turn = Some(1);
        step(&mut resolved, Action::CastSpell(robbery)).unwrap();
        assert!(matches!(
            pass_until_stack_resolves(&mut resolved),
            Decision::ChooseOptionalCost { .. }
        ));
        step(
            &mut resolved,
            Action::ChooseOptionalCost(OptionalCostChoice::Decline),
        )
        .unwrap();
        assert_eq!(resolved.objects.get(robbery).zone, Zone::Graveyard);
        assert_eq!(resolved.objects.get(robbery).plotted_turn, None);

        event::propose_and_commit(
            &mut resolved,
            ProposedEvent::zone_change(robbery, Zone::Exile),
        );
        resolved.engine.exile_play_permissions.push(PlayPermission {
            object: robbery,
            holder: PlayerId::P0,
            zone_change_generation: resolved.objects.get(robbery).zone_change_count,
            play_or_cast: PlayOrCast::Cast,
            expiry: PlayPermissionExpiry::EndOfTurn,
        });
        resolved.engine.priority_passes = [false, false];
        resolved.priority_player = PlayerId::P0;
        step(&mut resolved, Action::CastSpell(robbery)).unwrap();
        assert_eq!(
            resolved
                .engine
                .pending_cast
                .as_ref()
                .unwrap()
                .source_contract
                .cast_method,
            CastMethodV4::Normal
        );

        let mut retry = ready_game_in_main1(0);
        retry.turn = 2;
        let robbery = put_in_hand(&mut retry, PlayerId::P0, "Highway Robbery");
        event::propose_and_commit(&mut retry, ProposedEvent::zone_change(robbery, Zone::Exile));
        retry.objects.get_mut(robbery).plotted_turn = Some(1);
        step(&mut retry, Action::CastSpell(robbery)).unwrap();
        let pending = retry.engine.pending_cast.take().unwrap();
        abort_cast(&mut retry, pending, CastMethodV4::Plotted);
        assert_eq!(retry.objects.get(robbery).zone, Zone::Exile);
        assert_eq!(retry.objects.get(robbery).plotted_turn, Some(1));
        step(&mut retry, Action::CastSpell(robbery)).unwrap();
        assert_eq!(
            retry
                .engine
                .pending_cast
                .as_ref()
                .unwrap()
                .source_contract
                .cast_method,
            CastMethodV4::Plotted
        );
    }

    #[test]
    fn play_permission_is_voided_by_any_further_zone_change_not_just_being_played() {
        let mut state = ready_game_in_main1(0);
        let bolt_id = set_library_top(&mut state, PlayerId::P0, "Lightning Bolt");
        let percussionist = put_on_battlefield(&mut state, PlayerId::P0, "Clockwork Percussionist");
        event::propose_and_commit(
            &mut state,
            ProposedEvent::zone_change(percussionist, Zone::Graveyard),
        );
        collect_and_queue_triggers(&mut state);
        loop {
            if state.engine.pending_triggers.is_empty() && state.stack.is_empty() {
                break;
            }
            match advance_until_decision(&mut state) {
                Decision::CastSpellOrPass { .. } => step(&mut state, Action::Pass).unwrap(),
                other => panic!("unexpected decision resolving the dies trigger: {other:?}"),
            }
        }
        assert!(active_permission_for(PlayerId::P0, bolt_id, &state).is_some());

        // Something else moves the card out of exile entirely independent
        // of the permission (e.g. a hypothetical graveyard-hate effect) --
        // CR 400.7: this alone must void the permission, not just actually
        // playing it.
        event::propose_and_commit(
            &mut state,
            ProposedEvent::zone_change(bolt_id, Zone::Graveyard),
        );
        assert!(
            active_permission_for(PlayerId::P0, bolt_id, &state).is_none(),
            "any further zone change voids the permission"
        );

        // Even if it later returns to exile by some other means, the
        // *stale* `exile_play_permissions` entry (never explicitly removed)
        // must not reactivate: the generation moved twice, never back to
        // the exact snapshot the permission was granted at.
        event::propose_and_commit(&mut state, ProposedEvent::zone_change(bolt_id, Zone::Exile));
        assert!(
            active_permission_for(PlayerId::P0, bolt_id, &state).is_none(),
            "a later re-arrival in exile must not resurrect a stale permission"
        );
    }

    #[test]
    fn exiled_nonland_stays_unplayable_even_with_a_forged_play_permission() {
        let mut state = ready_game_in_main1(0);
        let clue = put_in_hand(&mut state, PlayerId::P0, "Clue Token");
        event::propose_and_commit(&mut state, ProposedEvent::zone_change(clue, Zone::Exile));
        state.engine.exile_play_permissions.push(PlayPermission {
            object: clue,
            holder: PlayerId::P0,
            zone_change_generation: state.objects.get(clue).zone_change_count,
            play_or_cast: PlayOrCast::Play,
            expiry: PlayPermissionExpiry::EndOfTurn,
        });

        assert!(active_permission_for(PlayerId::P0, clue, &state).is_some());
        assert!(!land_drop_candidates(PlayerId::P0, &state).contains(&clue));
        assert!(step(&mut state, Action::PlayLand(clue)).is_err());
        assert_eq!(state.objects.get(clue).zone, Zone::Exile);
    }

    fn one_card_p0_library(seed: u64) -> GameState {
        let island = card_id_by_name("Island").expect("Island in CARD_DEFS");
        GameState::new_from_libraries(
            &[island],
            &[],
            |card_def| card_def::CARD_DEFS[card_def as usize].name.to_string(),
            seed,
        )
    }

    /// Stages Ponder's pending MayShuffleLibrary Boolean choice for P0 through
    /// the real spell-resolution path, returning the state and source id. The
    /// caller supplies a one-card library so Ponder's preceding reorder is
    /// forced and yields no separate policy decision.
    fn stage_may_shuffle_choice(mut state: GameState) -> (GameState, ObjectId) {
        assert_eq!(state.players[PlayerId::P0.index()].library.len(), 1);
        let source = put_on_stack(&mut state, PlayerId::P0, "Ponder");
        state.engine.priority_passes = [true, true];
        let decision = advance_until_decision(&mut state);
        assert!(
            matches!(
                decision,
                Decision::ChooseEffectBoolean {
                    player: PlayerId::P0,
                    ..
                }
            ),
            "expected Ponder's shuffle choice, got {decision:?}"
        );
        (state, source)
    }

    /// Swaps the staged legacy seed-1 randomness for the given v2 fragment
    /// through the public serde surface, exactly as a restore would.
    fn into_environment_v2(state: &GameState, fragment: &str) -> GameState {
        let json = serde_json::to_string(state).expect("staged state serializes");
        let legacy_fragment = "\"rng\":{\"state\":1}";
        assert!(
            json.contains(legacy_fragment),
            "staged state must carry the untouched seed-1 legacy fragment"
        );
        serde_json::from_str(&json.replacen(legacy_fragment, fragment, 1))
            .expect("converted v2 state deserializes")
    }

    const HEALTHY_V2_FRAGMENT: &str = "\"environment_randomization_v2\":{\"pair_environment_seed\":9,\"next_live_shuffle_ordinal\":[0,0]}";
    const EXHAUSTED_V2_FRAGMENT: &str = "\"environment_randomization_v2\":{\"pair_environment_seed\":9,\"next_live_shuffle_ordinal\":[18446744073709551615,0]}";

    #[test]
    fn may_shuffle_true_preflights_v2_ordinal_before_any_choice_mutation() {
        use crate::environment_randomization_v2::PhysicalOwnerV2;
        let (staged, _source) = stage_may_shuffle_choice(one_card_p0_library(1));
        let mut exhausted = into_environment_v2(&staged, EXHAUSTED_V2_FRAGMENT);
        let before_json = serde_json::to_string(&exhausted).expect("serializes");
        let before_hot = exhausted.state_hash();
        let before_diag = exhausted.diagnostic_state_hash();
        assert!(
            step(&mut exhausted, Action::ChooseEffectBoolean(true)).is_err(),
            "accepting a shuffle at an exhausted ordinal must fail"
        );
        assert_eq!(
            serde_json::to_string(&exhausted).expect("serializes"),
            before_json,
            "the failed acceptance must be byte-exact nonmutating"
        );
        assert_eq!(exhausted.state_hash(), before_hot);
        assert_eq!(exhausted.diagnostic_state_hash(), before_diag);

        // The same pending choice remains answerable: declining stays legal
        // at the exhausted ordinal and consumes nothing.
        step(&mut exhausted, Action::ChooseEffectBoolean(false))
            .expect("declining stays legal at an exhausted ordinal");
        assert!(matches!(
            advance_until_decision(&mut exhausted),
            Decision::CastSpellOrPass { .. }
        ));
        let v2 = exhausted
            .environment_randomization_v2()
            .expect("v2 state after decline");
        assert_eq!(v2.next_live_shuffle_ordinal(PhysicalOwnerV2::P0), u64::MAX);
        assert_eq!(v2.next_live_shuffle_ordinal(PhysicalOwnerV2::P1), 0);
        assert!(exhausted.engine.pending_effect.is_none());
    }

    #[test]
    fn may_shuffle_true_commits_exactly_one_ordinal_at_the_frame() {
        use crate::environment_randomization_v2::PhysicalOwnerV2;
        let (staged, _source) = stage_may_shuffle_choice(one_card_p0_library(1));
        let mut healthy = into_environment_v2(&staged, HEALTHY_V2_FRAGMENT);
        step(&mut healthy, Action::ChooseEffectBoolean(true)).expect("accepting is legal");
        assert_eq!(
            healthy
                .environment_randomization_v2()
                .expect("v2 state")
                .next_live_shuffle_ordinal(PhysicalOwnerV2::P0),
            0,
            "the action-time preflight token is discarded without consuming an ordinal"
        );
        assert!(matches!(
            advance_until_decision(&mut healthy),
            Decision::CastSpellOrPass { .. }
        ));
        let v2 = healthy.environment_randomization_v2().expect("v2 state");
        assert_eq!(
            v2.next_live_shuffle_ordinal(PhysicalOwnerV2::P0),
            1,
            "the ShuffleLibrary frame commits exactly one ordinal"
        );
        assert_eq!(v2.next_live_shuffle_ordinal(PhysicalOwnerV2::P1), 0);
    }

    #[test]
    fn may_shuffle_frame_restored_into_exhausted_counter_halts_without_gameplay_mutation() {
        use crate::environment_randomization_v2::PhysicalOwnerV2;
        // Nonempty sentinels: a real P0 library and both-observer knowledge
        // rows, so the post-halt equality assertions prove preservation
        // rather than comparing empty structures.
        let mut base = one_card_p0_library(1);
        base.reveal_library_top(PlayerId::P0, PlayerId::P0, 1);
        base.reveal_library_top(PlayerId::P1, PlayerId::P0, 1);
        let (staged, source) = stage_may_shuffle_choice(base);
        let mut accepted = into_environment_v2(&staged, HEALTHY_V2_FRAGMENT);
        step(&mut accepted, Action::ChooseEffectBoolean(true)).expect("accept");
        // Simulate a restore into an exhausted counter after the accepted
        // frame was staged but before it ran.
        let sabotaged_json = serde_json::to_string(&accepted)
            .expect("serializes")
            .replacen(HEALTHY_V2_FRAGMENT, EXHAUSTED_V2_FRAGMENT, 1);
        let mut sabotaged: GameState =
            serde_json::from_str(&sabotaged_json).expect("sabotaged state deserializes");
        let library_before = sabotaged.players[PlayerId::P0.index()].library.clone();
        let stack_before = sabotaged.stack.clone();
        let library_knowledge_before = sabotaged.library_knowledge.clone();
        assert_eq!(library_before.len(), 1, "library sentinel must be nonempty");
        for observer in [PlayerId::P0, PlayerId::P1] {
            assert!(
                !library_knowledge_before[observer.index()][PlayerId::P0.index()].is_empty(),
                "observer {observer:?} knowledge sentinel must be nonempty"
            );
        }
        assert!(!stack_before.is_empty(), "stack sentinel must be nonempty");
        assert!(matches!(
            advance_until_decision(&mut sabotaged),
            Decision::Halted {
                mechanic: UnsupportedMechanic::InvalidEffectContinuation,
                source: halted_source,
            } if halted_source == source
        ));
        assert_eq!(
            sabotaged.players[PlayerId::P0.index()].library,
            library_before
        );
        assert_eq!(
            sabotaged.stack, stack_before,
            "the halt transition must restore the public stack exactly"
        );
        assert_eq!(
            sabotaged.library_knowledge, library_knowledge_before,
            "the halted frame must not touch any observer's library knowledge"
        );
        let v2 = sabotaged.environment_randomization_v2().expect("v2 state");
        assert_eq!(v2.next_live_shuffle_ordinal(PhysicalOwnerV2::P0), u64::MAX);
        assert_eq!(v2.next_live_shuffle_ordinal(PhysicalOwnerV2::P1), 0);
    }

    #[test]
    fn may_shuffle_legacy_path_is_byte_exact() {
        let base = one_card_p0_library(5);
        let (mut legacy, _source) = stage_may_shuffle_choice(base);
        let rng_at_answer = *legacy.legacy_rng().expect("legacy state");
        let (expected_library, expected_drawn, expected_rng) = {
            let mut rng = rng_at_answer;
            let mut library = legacy.players[PlayerId::P0.index()].library.clone();
            for i in (1..library.len()).rev() {
                let j = (rng.next_u64() % (i as u64 + 1)) as usize;
                library.swap(i, j);
            }
            let drawn = library.remove(0);
            (library, drawn, rng)
        };
        step(&mut legacy, Action::ChooseEffectBoolean(true)).expect("accept");
        assert_eq!(
            *legacy.legacy_rng().expect("legacy state"),
            rng_at_answer,
            "the action-time preflight must not advance the legacy RNG"
        );
        assert!(matches!(
            advance_until_decision(&mut legacy),
            Decision::CastSpellOrPass { .. }
        ));
        assert_eq!(
            legacy.players[PlayerId::P0.index()].library,
            expected_library
        );
        assert_eq!(
            legacy.players[PlayerId::P0.index()].hand,
            vec![expected_drawn],
            "Ponder draws the sole card after the accepted shuffle"
        );
        assert_eq!(
            *legacy.legacy_rng().expect("legacy state"),
            expected_rng,
            "the frame commit must advance the RNG exactly as the historical shuffle"
        );
    }
}
