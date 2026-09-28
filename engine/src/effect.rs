//! Interpreted effect programs.
//!
//! `EffectOp` is the only representation of card behavior: composition
//! primitives (`Sequence`, `Conditional`, `Choice`) plus a fixed
//! leaf-op vocabulary (`DealDamage`, `DrawCards`, `MoveObject`, library
//! partitioning, token creation, and other reusable state transitions).
//! There is no card-shaped op --
//! "Lightning Bolt" is not a variant, `DealDamage { amount: 3, .. }` is
//! (see `card_def.rs` / the generated `CARD_DEFS` table for how card
//! behavior handlers are wired up).
//!
//! `execute` and the resumable interpreter are the only paths that run an
//! `EffectOp`, and every leaf mutation goes through `event::propose_and_commit`
//! or `event::propose_and_commit_batch`, so nothing but the commit pipeline
//! (`event::commit`) mutates `GameState` in response to card behavior (see the
//! crate-level invariants in `lib.rs`).

use crate::card_def::{CardType, DynamicValueDef, Keywords, OptionalAdditionalCostDef, Subtype};
use crate::event;
use crate::ids::{ObjectId, PlayerId, StackItemId};
use crate::mana::{Cost, ManaColor};
use crate::state::{
    AbilitySourceContractV4, GameState, InitiativeTriggerBindingV1, InitiativeTriggerKindV1,
    LinkedExileRecordV4, ObjectLinkV4, PaidCostRefV4, StackItem, StackSourceContractV4,
    StackTargetContractV4, Target, UndercityRoomV1, Zone,
};
use serde::{Deserialize, Serialize};

/// Immutable cast-time evidence for one Storm trigger. The printed copy
/// count is frozen immediately after the source cast, while the historical
/// contract lets the trigger survive that physical spell leaving the stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StormCopyBindingV1 {
    pub source_contract: StackSourceContractV4,
    pub producing_stack_item: StackItemId,
    pub turn: u32,
    pub active_player: PlayerId,
    pub spell_cast_event_index: u32,
    pub casts_after_source: [u16; 2],
}

/// Which of a controller's creatures a team-wide pump/keyword effect
/// affects (`EffectOp::PumpControlled`). A closed, tiny enum rather than a
/// general subtype query: only the two shapes Rally's cards need exist
/// today (Goblin Bushwhacker's unfiltered "creatures you control", Rally at
/// the Hornburg's "Humans you control") -- a future card needing a
/// different subtype filter adds a `CreatureFilter` variant, reusing
/// `card_def::Subtype` (typed, not a string-contains check).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CreatureFilter {
    AnyControlled,
    ControlledWithSubtype(Subtype),
    /// Any creature that does not currently have `keyword`. Operations may
    /// apply this predicate across both battlefields.
    WithoutKeyword(Keywords),
}

/// Battlefield creature restriction for a player-directed sacrifice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CreatureSacrificeFilter {
    Any,
    GreatestPower,
}

/// Typed predicate for a private library search. The first consumer is
/// Islandcycling, whose Oracle filter is a physical land card carrying the
/// Island subtype (basic or nonbasic). This remains definition data and can
/// grow with other typecycling/basic-search shapes without card-name logic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LibraryCardFilter {
    /// The card must currently have the requested effective subtype and be
    /// a land. Effective subtypes come from `ObjectStateV4`; card types are
    /// definition-derived only while the object remains on face zero because
    /// the current kernel has no type-changing operation or effective-type
    /// field. Search validation fails closed outside that invariant.
    LandWithSubtype(Subtype),
    /// A card with both the Basic supertype and Land card type. This is
    /// intentionally independent of basic land subtypes: Roost Seek may
    /// find every basic land, including any future nonstandard basic land
    /// represented by the registry.
    BasicLand,
    /// A card that is either a basic land or carries the Gate subtype.
    /// Gatecreeper Vine is the first consumer. Gate cards are lands in the
    /// current pool, but the printed filter is subtype-based and remains so
    /// here rather than silently adding a land-type restriction.
    BasicLandOrGate,
    /// A physical card with this exact generated card-definition id. This
    /// is the reusable same-name search contract used by Squadron Hawk.
    CardDefinition(u16),
    /// A physical card with the Basic supertype and Land type that carries
    /// at least one of the three requested effective land subtypes.
    BasicLandWithAnySubtype([Subtype; 3]),
}

/// How long an impulse-drawn card (`EffectOp::ImpulseDraw`) stays playable
/// from exile. `EndOfTurn` is cleared unconditionally at the very next
/// `Step::Cleanup`, whoever's; `UntilOwnersNextTurn` survives through the
/// rest of this turn, the opponent's turn, and the owner's own next turn,
/// expiring at *that* turn's cleanup -- tracked via `engine::
/// PlayPermissionExpiry`, since a plain turn-number comparison can't tell
/// the owner's turn apart from the opponent's turn sharing the same kernel
/// round number (see that type's doc). Both durations are carried by an
/// `engine::PlayPermission`, not a pseudo-hand-zone membership list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ImpulseDuration {
    EndOfTurn,
    UntilOwnersNextTurn,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ObjectRef {
    /// The permanent/spell this effect program belongs to.
    ThisSource,
    /// A target resolved at cast/activation time, by index.
    Target(u8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PlayerRef {
    Controller,
    Target(u8),
    /// The controller of a target object (e.g. "that creature's
    /// controller").
    ObjectController(ObjectRef),
    /// The controller's one opponent in this strictly two-player kernel.
    Opponent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TargetRef {
    ThisSource,
    Target(u8),
    /// The controller's one opponent. The kernel only ever simulates 1v1
    /// games (see `lib.rs`), so "deal N damage to each opponent"
    /// (Guttersnipe, Voldaren Epicure, Grab the Prize) never needs a
    /// chosen target -- it's always exactly `ctx.controller.opponent()`.
    Opponent,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EffectCond {
    Always,
    Never,
    /// True iff this cast's mandatory additional cost discarded a
    /// non-land card (Grab the Prize). Reads `ExecCtx::discarded`, which
    /// `engine::finalize_cast` populates from the additional-cost payment
    /// before pushing the spell onto the stack.
    DiscardedNonLandForCost,
    /// True iff `ctx.controller` had a land enter the battlefield under
    /// their control this turn (Searing Blaze's landfall clause). Reads
    /// `PlayerState::lands_played_this_turn`, which only tracks land
    /// *drops* -- an accurate proxy for this pool, since nothing in it puts
    /// a land onto the battlefield any other way.
    LandfallThisTurn,
    /// True iff `ctx.targets[idx]` is `Target::Object(id)` and that
    /// object is currently in `zone`. `false` for a `Target::Player` (no
    /// card in this pool needs that combination) or an out-of-range index.
    /// The general-purpose 608.2b "is this target still legal" fizzle
    /// check: a creature that died, a spell that already left the stack, or
    /// a permanent that's no longer on the battlefield all read `false`
    /// here, and the guarded effect they'd otherwise feed is skipped
    /// instead of misfiring against a stale `ObjectId`.
    TargetInZone(u8, Zone),
    /// True iff `ctx.targets[idx]` is a `Target::Object(id)` whose card
    /// definition's static `colors` includes `color` (105.1/202.2). `false`
    /// for a `Target::Player`. Pyroblast and Hydroblast use this condition
    /// for their resolution-time "if it's [color]" checks; the Elemental
    /// Blasts' filtered target specs independently read the same static
    /// `CardDef::colors` source while choosing and revalidating targets.
    /// Unlike XMage's dynamic `getColor(game)`, this does not yet observe
    /// continuous color-changing effects.
    TargetIsColor(u8, crate::mana::ManaColor),
    /// Both sub-conditions must hold.
    And(Box<EffectCond>, Box<EffectCond>),
    /// Metalcraft (Galvanic Blast): true iff `ctx.controller` currently
    /// controls at least `n` permanents with `CardType::Artifact` (Great
    /// Furnace, Clockwork Percussionist, and Experimental Synthesizer are
    /// this pool's artifacts). A plain board-state count, not persisted
    /// anywhere -- recomputed fresh every time it's checked, same as
    /// `LandfallThisTurn`.
    ControlsArtifactCount(u8),
    /// True iff `ctx.controller` currently controls at least `minimum_count`
    /// battlefield permanents with `subtype`, excluding this effect's exact
    /// source object. Gingerbread Cabin uses this for the resolution half of
    /// its intervening-if clause.
    ControlsOtherSubtypeCount {
        subtype: Subtype,
        minimum_count: u8,
    },
    /// True iff `ctx.controller` currently controls a battlefield object
    /// with the same card definition as this effect's source, excluding the
    /// exact source object. Faerie Miscreant uses this for the resolution
    /// half of its intervening-if clause. Excluding the source is important:
    /// the condition can remain true after the triggering Miscreant leaves
    /// the battlefield as long as its controller still controls another one.
    ControlsAnotherSourceCard,
    /// True iff the cast this resolution's ETB trigger followed from was
    /// kicked (`card_def::CardDef::kicker_cost`, Goblin Bushwhacker's "if it
    /// was kicked" intervening-if). Reads `ExecCtx::kicked` -- cast-time
    /// metadata carried on the spell's `StackItem`, propagated to the ETB
    /// trigger's own `PendingTrigger`/`StackItem`/`ExecCtx` when it's
    /// queued, and gone once that trigger resolves (CR 702.33/601.2f: Kicker
    /// is a property of *this casting*, not a durable fact stored anywhere
    /// keyed by stable object id -- CR 400.7 zone changes create new
    /// objects, so a persistent id-keyed marker could falsely survive a
    /// later, unkicked cast of the same physical card if it were ever
    /// cleared incorrectly. Reading cast-scoped context instead of a lookup
    /// table makes that failure mode structurally impossible rather than
    /// merely avoided by careful clearing).
    WasKicked,
    /// True iff this exact cast paid the named optional additional cost.
    /// The provenance is carried by the resolving spell or its ETB trigger,
    /// never looked up by stable object id.
    OptionalAdditionalCostPaid(OptionalAdditionalCostDef),
    /// Whether the effect controller's graveyard currently contains a card
    /// with the requested type. Masked Vandal uses this to avoid offering an
    /// impossible optional payment.
    ControllerGraveyardHasType(CardType),
    /// Refurbished Familiar's 1v1 form: its opponent can discard exactly
    /// one card iff that opponent's hand is nonempty at resolution.
    OpponentHasCardsInHand,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EffectOp {
    Sequence(Vec<EffectOp>),
    Conditional {
        cond: EffectCond,
        then: Box<EffectOp>,
        else_: Box<EffectOp>,
    },
    /// The selected player picks one of `options` during resolution. The
    /// generic resumable interpreter preserves printed option order and
    /// yields a policy-visible decision without opening a priority, SBA, or
    /// trigger window.
    Choice {
        controller: PlayerRef,
        options: Vec<EffectOp>,
    },
    DealDamage {
        target: TargetRef,
        amount: i32,
    },
    GainLife {
        player: PlayerRef,
        amount: i32,
    },
    LoseLife {
        player: PlayerRef,
        amount: i32,
    },
    DrawCards {
        player: PlayerRef,
        count: u32,
    },
    /// Publicly reveals a fixed snapshot of the top `count` cards of
    /// `player`'s library, then moves every card with `card_type` to
    /// `matching_to` and the remainder to `rest_to`. The matching group
    /// moves first and each group is one replacement-evaluated batch. A
    /// 2+ card graveyard group is explicitly ordered by its owner (the
    /// forced final card auto-completes); other groups retain snapshot order.
    /// This is not a draw: a short/empty library simply contributes fewer
    /// cards and never sets the draw-from-empty loss marker.
    RevealTopAndPartitionByType {
        player: PlayerRef,
        count: u8,
        card_type: CardType,
        matching_to: Zone,
        rest_to: Zone,
    },
    /// Discard `count` cards from `player`'s hand, chosen by that player.
    /// Unlike every other leaf, this one doesn't necessarily mutate state
    /// synchronously: `execute` stages `EngineState::pending_discard` and
    /// returns, and `engine::advance_until_decision` asks
    /// `Decision::Discard`. Because of that, **this must be the last leaf
    /// in any `Sequence` it appears in** (see `engine.rs`'s
    /// `pending_discard` doc for why: nothing after it in the same
    /// resolution would run before the decision is answered). The only
    /// user this increment, Faithless Looting ("draw two, then discard
    /// two"), satisfies this by construction.
    DiscardCards {
        player: PlayerRef,
        count: u32,
    },
    MoveObject {
        object: ObjectRef,
        to_zone: Zone,
    },
    TapObject {
        object: ObjectRef,
    },
    /// The permanent does not untap during its current controller's next
    /// untap step. The marker is incarnation-local and is cleared by any
    /// zone change through `ObjectStateV4::reset_for_zone_change`.
    SkipNextUntap {
        object: ObjectRef,
    },
    /// Attach the exact resolving ability source incarnation to a target
    /// permanent through the generic two-way attachment graph.
    AttachSourceToTarget {
        object: ObjectRef,
    },
    /// Add public permanent counters atomically to one selected creature.
    /// `optional` permits an absent target slot, as on Cryogen Relic.
    AddCountersToTarget {
        target_index: u8,
        optional: bool,
        plus1_plus1: i16,
        lifelink: i16,
        stun: i16,
    },
    /// Job Select creates the token even if the Equipment later left, then
    /// attaches only when the original source incarnation remains live.
    CreateTokenAndAttachSource {
        token_def: u16,
    },
    AddMana {
        player: PlayerRef,
        colors: Vec<ManaColor>,
    },
    /// Creates a fresh token permanent (e.g. Blood) directly on the
    /// battlefield under `controller`'s control. `token_def` indexes
    /// `card_def::CARD_DEFS` same as any other object -- tokens are real
    /// `GameObject`s, not a separate representation (see
    /// `event::ProposedEvent::create_token`).
    CreateToken {
        token_def: u16,
        controller: PlayerRef,
    },
    /// The controller may pay ONE of {discard `discard` cards, sacrifice
    /// `sacrifice_lands` lands} -- only whichever options are currently
    /// legal are offered, and declining is always legal too (Highway
    /// Robbery's `DoIfCostPaid(OrCost(DiscardCardCost, SacrificeTargetCost))`).
    /// If they do, `then` runs. Like `DiscardCards`, this is deferred:
    /// `execute` stages `EngineState::pending_optional_cost` and returns
    /// without knowing the outcome yet (`engine::Decision::ChooseOptionalCost`
    /// asks), so **this must be the last leaf in any `Sequence` it appears
    /// in**, same constraint and same reason as `DiscardCards`. A future
    /// card that needs both sub-costs simultaneously payable (not this
    /// pool) is out of scope: `discard`/`sacrifice_lands` are mutually
    /// exclusive choices, never both paid.
    MayPayCostThen {
        discard: u8,
        sacrifice_lands: u8,
        then: Box<EffectOp>,
    },
    /// "Deals `amount` damage to each opponent and each creature they
    /// control" (End the Festivities). The kernel only ever simulates 1v1
    /// games (see `lib.rs`), so "each opponent" is always exactly one
    /// player, same rationale as `TargetRef::Opponent`; no planeswalker
    /// card exists in the 132-card pool this increment adds to, so the
    /// "and each planeswalker they control" half of the real card's text
    /// is vacuously covered (there is never one to hit).
    DamageOpponentAndTheirCreatures {
        amount: i32,
    },
    /// A team-wide, until-end-of-turn pump/haste grant (Goblin Bushwhacker's
    /// kicked ETB, Rally at the Hornburg's token haste). Snapshots exactly
    /// which of `ctx.controller`'s current creatures match `filter` *at the
    /// moment this executes* (611.2c: the affected-objects set is locked in
    /// when the effect begins, not re-evaluated later) and stages an
    /// `engine::UntilEndOfTurnEffect::ResolvedSetEffect` naming those
    /// specific `ObjectId`s, cleared unconditionally at the next
    /// `Step::Cleanup` --
    /// see that variant's doc. Sequencing matters for Rally at the Hornburg:
    /// its own two `CreateToken` calls run *before* this in the same
    /// `Sequence`, so the freshly-created Human Soldier tokens are already
    /// on the battlefield (and therefore in the snapshot) by the time this
    /// leaf runs.
    PumpControlled {
        filter: CreatureFilter,
        power: i32,
        toughness: i32,
        grant_haste: bool,
    },
    /// Exiles the top `count` cards of `ctx.controller`'s library (silently
    /// stopping short if the library runs out first -- this is not a draw,
    /// so an empty library here is not a loss condition) and marks each one
    /// playable by its owner for `duration` -- Clockwork Percussionist's
    /// dies trigger, Experimental Synthesizer's enters-or-leaves trigger,
    /// and Reckless Impulse all reduce to this, differing only in `count`/
    /// `duration`. See `ImpulseDuration`'s doc for how each duration is
    /// tracked/expired (`engine::PlayPermission`), and `engine::
    /// castable_spells`/`engine::land_drop_candidates` for where the
    /// resulting exiled cards become legally castable/playable again,
    /// through the *ordinary* timing/cost/land-quota checks.
    ImpulseDraw {
        count: u8,
        duration: ImpulseDuration,
    },
    /// Chain Lightning's post-damage "that player or that permanent's
    /// controller may pay {R}{R}" branch. XMage asks this choice before it
    /// attempts payment, even when payment will fail, so this always
    /// suspends the current resolution in the engine's dedicated spell-copy
    /// state machine; no SBA, trigger placement, zone
    /// move, or priority window can occur until payment and optional
    /// retargeting finish. This leaf must remain last in Chain Lightning's
    /// generated sequence for the same reason `DiscardCards` must remain
    /// last in its sequence.
    OfferAffectedPlayerSpellCopy {
        affected: TargetRef,
    },
    /// Puts up to the top `count` cards of `player`'s library into their
    /// graveyard as one library-to-graveyard zone-change batch. This is not a draw: a
    /// short or empty library simply contributes fewer cards and never sets
    /// the draw-from-empty marker. If two or more cards would move together,
    /// their owner orders the batch through the resumable interpreter; the
    /// pending private-library identities are exposed only to that owner.
    /// This primitive does not yet emit a distinct pre-batch `MILL_CARDS`
    /// replacement event or a post-move mill-summary event; those hooks stay
    /// fail-closed until a supported pool card consumes them. Appended to
    /// preserve every existing variant's derived hash identity.
    MillCards {
        player: PlayerRef,
        count: u8,
    },
    /// Privately looks at up to the top `count` cards of `player`'s
    /// library and lets that same player put them back in any order. The
    /// interpreter binds the exact prefix/incarnations before yielding the
    /// ordered choice. AIRL/XMage presents each explicit pick as the next
    /// deepest card; the forced final card is therefore the new top card.
    /// Appended to preserve every existing variant's derived hash identity.
    LookAtLibraryTopAndReorder {
        player: PlayerRef,
        count: u8,
    },
    /// The selected player may shuffle their library. This is a real
    /// resolution-time Boolean choice even for a zero- or one-card library;
    /// accepting uses the state's deterministic shuffle stream and declining
    /// leaves both order and knowledge untouched. Appended for hash identity.
    MayShuffleLibrary {
        player: PlayerRef,
    },
    /// Repeatedly lets `player` choose one card from their current hand and
    /// puts that card on top of their library, stopping after `count` cards
    /// or when the hand is empty. Each card is a distinct private choice and
    /// zone change: the first chosen card is therefore deepest and the last
    /// chosen card is topmost. Appended to preserve existing hash identities.
    PutCardsFromHandOnLibraryTop {
        player: PlayerRef,
        count: u8,
    },
    /// Privately looks at the top `min(count, library.len())` cards of
    /// `player`'s library. The currently certified contract is only
    /// Preordain/Scry2 final-state semantics: `count > 2` fails before any
    /// library binding or reveal. The player chooses an unordered subset to
    /// put on the bottom, explicitly orders a 2-card bottom group
    /// shallow-to-deep, then explicitly orders a 2-card retained group
    /// deepest-to-topmost. The three private stages never open priority, SBA,
    /// or trigger windows; one atomic state transition applies the final
    /// top/tail/bottom order.
    ///
    /// No partial SCRY/SCRY_TO_BOTTOM/SCRIED event family is emitted yet.
    /// Arbitrary higher-count scry requires XMage-order bottom commitment plus
    /// typed hooks for those events, as does any supported replacement or
    /// trigger that observes them. Appended to preserve existing derived hash
    /// identities.
    Scry {
        player: PlayerRef,
        count: u8,
    },
    /// Search `player`'s library for zero or one physical card matching the
    /// typed filter, put the chosen card into hand, reveal it publicly, then
    /// shuffle. The optional zero-card result is rules-correct fail-to-find,
    /// even while matches exist. Zero matches still suspend on a uniform
    /// private Finish-only prompt, preventing continuation presence from
    /// leaking match existence. Candidate identity remains chooser-private
    /// until the selected card reaches hand and is revealed.
    ///
    /// This operation intentionally does not synthesize XMage's typed
    /// SEARCH_LIBRARY/LIBRARY_SEARCHED/SHUFFLE_LIBRARY event family yet;
    /// only the ordinary replaceable zone change is emitted. A supported
    /// consumer of those replacement/trigger hooks must add them first.
    /// Appended to preserve existing derived hash identities.
    SearchLibraryToHand {
        player: PlayerRef,
        filter: LibraryCardFilter,
    },
    /// Binds one currently-live object and lets that object's owner choose
    /// whether it enters their library second from the top or on the bottom.
    /// The binding is captured before the choice, so a restored/stale
    /// continuation cannot redirect the selected branch to a later
    /// incarnation. Deem Inferior is the first generated consumer.
    PutObjectInOwnersLibrarySecondOrBottom {
        object: ObjectRef,
    },
    /// Interpreter-owned selected branch for
    /// `PutObjectInOwnersLibrarySecondOrBottom`. Generated card programs may
    /// not contain this dynamic form directly.
    PutBoundObjectInOwnersLibrary {
        object: EffectObjectBinding,
        owner: PlayerId,
        placement: event::LibraryPlacement,
    },
    /// Destroys a battlefield permanent without conflating destruction with
    /// an ordinary zone move. Indestructible prevents this operation and the
    /// lethal-damage state-based action, but does not prevent sacrifice,
    /// exile, bounce, or the zero-toughness state-based action. Appended to
    /// preserve all existing effect variant identities.
    DestroyObject {
        object: ObjectRef,
    },
    /// Counters the exact opposing stack incarnation that targeted the bound
    /// Ward permanent unless that item's controller pays generic mana.
    /// Trigger construction, not generated card data, creates this leaf.
    CounterUnlessPaysGeneric {
        ward_target: StackTargetContractV4,
        targeting_stack_item: StackItemId,
        generic: u8,
    },
    /// Deals one simultaneous damage batch to every creature without the
    /// excluded subtype. Breath Weapon is the first consumer.
    DamageEachCreatureWithoutSubtype {
        amount: i32,
        excluded_subtype: Subtype,
    },
    /// Counters the exact spell incarnation selected by `target` unless
    /// that spell's controller pays generic mana. Unlike Ward, this binding
    /// originates from the resolving spell's own cast-time target contract.
    /// Appended so every existing effect discriminant remains stable.
    CounterTargetUnlessPaysGeneric {
        target: TargetRef,
        generic: u8,
    },
    /// Gain a board-dependent amount of life, sampled when this leaf
    /// resolves. Wellwisher is the first consumer.
    GainLifeDynamic {
        player: PlayerRef,
        amount: DynamicValueDef,
    },
    /// Untap one bound object. Target legality and incarnation are checked by
    /// the stack target contract before this leaf executes.
    UntapObject {
        object: ObjectRef,
    },
    /// Give one target creature a board-dependent power/toughness bonus until
    /// end of turn. Each value is sampled once at resolution and the affected
    /// object set is fixed to that one target.
    PumpTargetUntilEndOfTurnDynamic {
        target: TargetRef,
        power: DynamicValueDef,
        toughness: DynamicValueDef,
    },
    /// Privately look at the top `count` cards, choose any number matching
    /// `card_type`, publicly reveal and move those cards to hand, then put
    /// the rest on the bottom in their owner's chosen order. The exact prefix
    /// and every private stage are incarnation-bound by the resumable
    /// interpreter. Lead the Stampede is the first consumer.
    LookTopSelectByTypeToHandBottomRest {
        player: PlayerRef,
        count: u8,
        card_type: CardType,
    },
    /// Gains life equal to the printed mana values of the objects frozen in
    /// this stack item's cost-payment provenance. Cost reductions never
    /// change those values. Reckoner's Bargain is the first consumer.
    GainLifeEqualToPaidCostManaValue {
        player: PlayerRef,
    },
    /// Moves each still-valid announced object target independently. This
    /// implements effects such as Blood Fountain's up-to-two graveyard
    /// returns, where one stale target does not stop the other from moving.
    MoveAllTargets {
        to_zone: Zone,
    },
    /// The target creature explores: its controller reveals the top card of
    /// their library, puts a land into hand, or puts a +1/+1 counter on the
    /// creature and may put a nonland into the graveyard.
    ExploreTarget {
        object: ObjectRef,
    },
    /// Interpreter-owned exact-incarnation move used after Explore reveals a
    /// nonland. Generated card programs never contain a pre-bound object.
    MoveBoundObject {
        object: EffectObjectBinding,
        to_zone: Zone,
        preserve_known_identity: bool,
    },
    /// Grants one keyword to the exact target permanent incarnation until
    /// end of turn. Piracy Charm's islandwalk mode is the first consumer.
    GrantKeywordTargetUntilEndOfTurn {
        object: ObjectRef,
        keyword: crate::card_def::Keywords,
    },
    /// Target creature gets +X/+X until end of turn, where X is the number
    /// of permanents the effect controller currently controls with the named
    /// effective subtype. The target set and count are both snapshotted at
    /// resolution.
    PumpTargetByControlledSubtypeCount {
        target: ObjectRef,
        subtype: Subtype,
    },
    /// Search for zero through `max_targets` cards matching a typed filter,
    /// reveal every selected card, put them into hand, then shuffle. Kept as
    /// an appended variant so the earlier zero-or-one search serialization
    /// remains byte-for-byte unchanged.
    SearchLibraryToHandUpTo {
        player: PlayerRef,
        filter: LibraryCardFilter,
        max_targets: u16,
    },
    /// Deals one simultaneous damage wave to every battlefield creature
    /// matching `filter`, independent of controller.
    DamageAllCreatures {
        filter: CreatureFilter,
        amount: i32,
    },
    /// Exiles every card in one selected player's graveyard as a single
    /// resolution batch.
    ExilePlayersGraveyard {
        player: PlayerRef,
    },
    /// The selected player chooses one card from their own graveyard and
    /// exiles it. Empty and singleton graveyards need no policy choice.
    ExileOneFromPlayersGraveyard {
        player: PlayerRef,
    },
    /// The selected player chooses one matching card from their own
    /// graveyard and exiles it. The complete graveyard and filtered
    /// candidates are incarnation-bound while the choice is suspended.
    MayExileFromPlayersGraveyardMatchingThen {
        player: PlayerRef,
        card_type: CardType,
        then: Box<EffectOp>,
    },
    /// Exiles all cards from both graveyards in one deterministic batch.
    ExileAllGraveyards,
    /// Offers `player` a Boolean choice to pay the owned mana cost. An
    /// accepted payment commits before `then` runs without opening priority.
    MayPayManaThen {
        player: PlayerRef,
        colored: Vec<ManaColor>,
        generic: u8,
        then: Box<EffectOp>,
    },
    /// Deals one simultaneous damage batch to every still-valid announced
    /// object target. Each target is incarnation-checked independently.
    DamageAllTargets {
        amount: i32,
    },
    /// Exiles every still-valid announced artifact-permanent target in one
    /// batch. Each target is checked independently at resolution.
    ExileAllArtifactTargets,
    /// Deals damage to one target equal to `multiplier` times the number of
    /// creatures the effect controller controls at resolution.
    DealDamageByControlledCreatureCount {
        target: TargetRef,
        multiplier: i32,
    },
    /// Exiles every card in each distinctly targeted player's graveyard as
    /// one simultaneous batch. The two-player kernel bounds this at two.
    ExileTargetPlayersGraveyards,
    /// Publicly reveal cards from the top of `player`'s library through the
    /// first card with `card_type`, including that card, then put the entire
    /// revealed prefix into its owner's graveyard. If no matching card
    /// exists, the complete library is revealed and moved. Balustrade Spy is
    /// the first consumer.
    RevealUntilCardTypeAndMill {
        player: PlayerRef,
        card_type: CardType,
    },
    /// Deal a state-dependent amount of damage, sampled once when this leaf
    /// resolves. Lotleth Giant is the first consumer.
    DealDamageDynamic {
        target: TargetRef,
        amount: DynamicValueDef,
    },
    /// Trigger-definition marker materialized into the bound variant below
    /// when the trigger is created. It must never reach resolution directly.
    BindPlusOnePlusOneCounterToTriggerSource,
    /// Put one +1/+1 counter on an exact battlefield incarnation. A stale
    /// binding is a rules-correct no-op, as the referenced permanent is no
    /// longer the object named by the resolving ability.
    PutPlusOnePlusOneCounterOnBoundObject {
        object: EffectObjectBinding,
    },
    /// Move this resolving permanent spell onto the battlefield attached to
    /// its exact creature target. Aura attachment is installed before the
    /// next SBA checkpoint.
    PutSourceOntoBattlefieldAttachedToTarget {
        target: ObjectRef,
    },
    /// Tap the creature currently attached to this Aura, then that creature
    /// deals damage to the Aura ability's controller equal to its power.
    TapAttachedCreatureAndDamageControllerByPower,
    /// Put a +1/+1 counter on the target and grant the keyword until end of
    /// turn iff the target is not this ability's source.
    BackupTarget {
        target: ObjectRef,
        keyword: crate::card_def::Keywords,
    },
    /// Move a hand-zone activated-ability source onto the battlefield tapped
    /// and attacking. The engine rechecks the live combat window.
    PutSourceOntoBattlefieldTappedAndAttacking,
    /// Let `chooser` choose zero through `max_targets` tapped lands, then
    /// untap the exact selected incarnations. The lands may have any
    /// controller.
    UntapUpToLands {
        chooser: PlayerRef,
        max_targets: u16,
    },
    /// Destroy the exact target land, then let the controller it had at
    /// resolution search for a basic land to enter tapped. The controller
    /// binding is captured before destruction.
    DestroyTargetLandThenMaySearchBasicTapped {
        object: ObjectRef,
    },
    /// Search one player's library for zero or one matching card, put it on
    /// the battlefield tapped, then shuffle. Zero selection is legal.
    SearchLibraryToBattlefieldTapped {
        player: PlayerRef,
        filter: LibraryCardFilter,
    },
    /// Reveal the target player's complete hand, then let the effect
    /// controller choose one noncreature, nonland card for that player to
    /// discard when at least one such card exists.
    RevealTargetHandChooseNoncreatureNonlandDiscard {
        player: PlayerRef,
    },
    /// Move the exact leave-trigger source from its owner's graveyard into
    /// that library, if it is still there, then shuffle that library.
    ShuffleTriggerSourceIntoOwnersLibrary,
    /// Trigger-definition marker replaced with a bound Storm program at the
    /// exact SpellCast checkpoint. It must never reach resolution directly.
    MaterializeStormCopies,
    /// Create the number of virtual spell copies frozen by `binding`.
    /// Copies are not casts and therefore emit no SpellCast marker.
    CreateStormCopies {
        binding: StormCopyBindingV1,
    },
    /// Creates a global until-end-of-turn rule effect under which no damage
    /// can be prevented. Existing prevention shields remain unconsumed.
    DamageCannotBePreventedThisTurn,
    /// Put one -1/-1 counter on an exact target creature incarnation.
    AddMinusOneMinusOneCounter {
        object: ObjectRef,
    },
    /// Publicly reveal the selected player's complete hand, then let this
    /// effect's controller choose exactly one nonland card from it to exile
    /// under the resolving ability source incarnation. An all-land or empty
    /// hand reveals and completes without a choice.
    RevealHandChooseNonlandToLinkedExile {
        player: PlayerRef,
    },
    /// Return the card still exiled by this exact historical ability-source
    /// incarnation to its owner's hand, if that exact incarnation remains.
    ReturnLinkedExiledCardToOwnersHand,
    /// Exile one exact target and link it to this resolving ability's frozen
    /// source incarnation. If that source already left before this trigger
    /// resolves, the target is still exiled but no return link is created.
    ExileTargetLinkedToSource {
        object: ObjectRef,
    },
    /// Return every object linked to this exact historical source
    /// incarnation from exile to the battlefield under its owner's control.
    ReturnObjectsExiledBySource,
    /// The resolving spell's controller chooses W, U, B, R, or G and installs
    /// a prevention effect covering all damage from sources sharing that
    /// color through the current turn.
    PreventDamageFromChosenColorUntilEndOfTurn {
        player: PlayerRef,
    },
    /// Interpreter-owned branch produced by the preceding color choice.
    InstallDamagePreventionFromColor {
        player: PlayerId,
        color: ManaColor,
    },
    /// Exile this exact Saga incarnation and return the same physical card
    /// transformed under this ability's controller.
    TransformSagaSource,
    /// The selected player sacrifices one creature matching the printed
    /// restriction. Ties for greatest power remain that player's choice.
    SacrificeCreature {
        player: PlayerRef,
        filter: CreatureSacrificeFilter,
    },
    /// Put one +1/+1 counter on an announced creature target.
    PutPlusOnePlusOneCounter {
        object: ObjectRef,
    },
    /// Move the resolving permanent spell to the battlefield and put its
    /// announced X +1/+1 counters on that exact new incarnation.
    PutSourceOntoBattlefieldWithXPlusOneCounters,
    /// Bestow's successful resolution: move the source to the battlefield,
    /// install X counters, and attach it to the exact creature target.
    PutSourceOntoBattlefieldAttachedToTargetWithXPlusOneCounters {
        target: ObjectRef,
    },
    /// Deal damage to the target creature equal to the power of the exact
    /// creature chosen for the resolving spell's additional cost. A live
    /// unchanged battlefield permanent is sampled at resolution; otherwise
    /// the payment-time LKI is used.
    DealDamageToTargetEqualToChosenCostCreaturePower {
        target: TargetRef,
    },
    /// Change the global Initiative designation to the selected player and
    /// emit the designation's venture trigger marker. Avenging Hunter's ETB
    /// is the definition-owned producer.
    TakeInitiative {
        player: PlayerRef,
    },
    /// Runtime-bound global Initiative or Undercity trigger. The event-history
    /// index and historical designation source are validated before dispatch.
    ResolveInitiativeTrigger {
        binding: InitiativeTriggerBindingV1,
    },
    /// Interpreter-owned route marker for an exact next Undercity room.
    EnterUndercityRoom {
        binding: InitiativeTriggerBindingV1,
        from_room: Option<UndercityRoomV1>,
        room: UndercityRoomV1,
    },
    AddPlusOnePlusOneCounters {
        object: ObjectRef,
        count: u8,
    },
    GoadTargetUntilSourcesNextTurn {
        object: ObjectRef,
    },
    /// Publicly reveal the top ten, choose a creature if needed, move it with
    /// three counters and hexproof until the room controller's next turn,
    /// then shuffle.
    ResolveUndercityThrone {
        binding: InitiativeTriggerBindingV1,
    },
}

/// One owned interpreter frame. `path` is the structural route through the
/// original effect program (sequence/branch/choice/group ordinals), making a
/// suspended continuation deterministic, hashable, and auditable without
/// storing closures or card-definition function pointers. Dynamic batch
/// frames are interpreter-owned: generated card programs only contain
/// `EffectOp`s.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EffectFrame {
    Program {
        op: EffectOp,
        path: Vec<u16>,
    },
    MoveObjectsBatch {
        objects: Vec<EffectObjectBinding>,
        to_zone: Zone,
        preserve_known_identity: bool,
        order_resolved: bool,
        path: Vec<u16>,
    },
    /// A bound library prefix awaiting its owner-selected graveyard order.
    /// Kept distinct from public reveal/partition batches so hidden milled
    /// identities stay chooser-private while the resolution is suspended.
    /// Appended to preserve existing continuation hashes.
    MillLibraryBatch {
        objects: Vec<EffectObjectBinding>,
        order_resolved: bool,
        path: Vec<u16>,
    },
    /// Commits one privately chosen ordering of an exact bound library
    /// prefix. `expected_prefix` preserves its pre-choice order separately
    /// from `ordered`, so a stale continuation cannot accept a same-set
    /// shuffle/reorder that incarnation-only validation would miss.
    ReorderLibraryTop {
        player: PlayerId,
        expected_prefix: Vec<EffectObjectBinding>,
        ordered: Vec<EffectObjectBinding>,
        path: Vec<u16>,
    },
    /// Executes an accepted optional shuffle on the next engine advance,
    /// rather than mutating during the action-answering call itself.
    ShuffleLibrary {
        player: PlayerId,
        path: Vec<u16>,
    },
    /// Coordinates one card at a time for a repeated private hand-to-library
    /// instruction. `chosen == None` stages the next exact-current-hand
    /// prompt; `Some` validates that prompt's hand snapshot and commits its
    /// single zone change before another prompt can be staged.
    PutCardsFromHandOnLibraryTop {
        player: PlayerId,
        /// Redundant copy of the originating op's requested count. Together
        /// with `remaining` and `prompt_index`, this makes trusted snapshot
        /// progress self-checking instead of trusting either counter alone.
        total: u8,
        remaining: u8,
        prompt_index: u16,
        expected_hand: Vec<EffectObjectBinding>,
        chosen: Option<EffectObjectBinding>,
        path: Vec<u16>,
        /// Redundant copy of the originating program path. Coordinator and
        /// prompt paths must remain mutually consistent with this copy.
        canonical_path: Vec<u16>,
    },
    /// Authenticated post-answer frame for Deem Inferior's owner choice.
    /// The exact pre-answer remainder and redundant option/path metadata
    /// prevent a restored answered snapshot from redirecting the move or
    /// inserting another effect before it commits.
    OwnerLibraryPlacement {
        object: EffectObjectBinding,
        owner: PlayerId,
        placement: event::LibraryPlacement,
        option_index: u16,
        path: Vec<u16>,
        canonical_path: Vec<u16>,
        expected_remaining_frames: Vec<EffectFrame>,
    },
    /// Resumes one private scry after a completed policy stage. All original
    /// prefix bindings, requested-count metadata, and canonical structural
    /// path remain redundant in every progress state so a stale or malformed
    /// snapshot fails before the atomic library transition.
    ScryLibrary {
        player: PlayerId,
        requested_count: u8,
        original_library_len: u32,
        original_prefix: Vec<EffectObjectBinding>,
        progress: ScryProgress,
        /// Deterministic redundant commitment to `progress`. This is not an
        /// authentication boundary, but it makes any isolated progress-field
        /// corruption fail closed instead of silently selecting another valid
        /// subset/order.
        progress_fingerprint: u64,
        path: Vec<u16>,
        canonical_path: Vec<u16>,
    },
    /// Commits an optional, incarnation-bound private search result. The
    /// exact full library order is retained independently from the selected
    /// candidate so restored state cannot alter a noncandidate card, append
    /// a hidden card, or reorder the library before the deterministic
    /// shuffle.
    SearchLibraryToHand {
        player: PlayerId,
        filter: LibraryCardFilter,
        filter_fingerprint: u64,
        original_library: Vec<EffectObjectBinding>,
        selected: Option<EffectObjectBinding>,
        path: Vec<u16>,
        canonical_path: Vec<u16>,
    },
    /// Authenticated post-answer Ward payment/counter completion.
    ResolveCounterUnlessPaysGeneric {
        ward_target: StackTargetContractV4,
        targeting_stack_item: StackItemId,
        player: PlayerId,
        generic: u8,
        pay: bool,
        path: Vec<u16>,
    },
    /// Authenticated post-answer completion for a resolving counterspell's
    /// exact target incarnation. Kept distinct from the Ward frame because
    /// its source and target bindings have opposite roles.
    ResolveCounterTargetUnlessPaysGeneric {
        target_contract: StackTargetContractV4,
        target_stack_item: StackItemId,
        player: PlayerId,
        generic: u8,
        pay: bool,
        path: Vec<u16>,
    },
    /// Resumes one private typed top-library partition after a completed
    /// policy stage. The complete original prefix and library length are
    /// retained through selection and remainder ordering so a restored or
    /// stale continuation cannot redirect either result.
    LookTopSelectByTypeToHandBottomRest {
        player: PlayerId,
        requested_count: u8,
        original_library_len: u32,
        card_type: CardType,
        original_prefix: Vec<EffectObjectBinding>,
        progress: LibraryPartitionProgress,
        progress_fingerprint: u64,
        path: Vec<u16>,
        canonical_path: Vec<u16>,
    },
    /// Commits a variable-cardinality private search while retaining the
    /// original library and every selected incarnation independently.
    SearchLibraryToHandMany {
        player: PlayerId,
        filter: LibraryCardFilter,
        filter_fingerprint: u64,
        original_library: Vec<EffectObjectBinding>,
        selected: Vec<EffectObjectBinding>,
        max_targets: u16,
        path: Vec<u16>,
        canonical_path: Vec<u16>,
    },
    /// Authenticated post-answer frame for one target-player graveyard pick.
    /// The exact pre-answer graveyard and remaining frame stack prevent a
    /// restored answer from redirecting or reordering resolution.
    ExileChosenGraveyardCard {
        player: PlayerId,
        original_graveyard: Vec<EffectObjectBinding>,
        chosen: EffectObjectBinding,
        path: Vec<u16>,
        canonical_path: Vec<u16>,
        expected_remaining_frames: Vec<EffectFrame>,
    },
    /// Authenticated completion for a filtered exact-one graveyard exile.
    ExileChosenMatchingGraveyardCard {
        player: PlayerId,
        card_type: CardType,
        original_graveyard: Vec<EffectObjectBinding>,
        candidates: Vec<EffectObjectBinding>,
        chosen: EffectObjectBinding,
        then: Box<EffectOp>,
        path: Vec<u16>,
        canonical_path: Vec<u16>,
        expected_remaining_frames: Vec<EffectFrame>,
    },
    /// Authenticated completion for one player-directed creature sacrifice.
    SacrificeChosenCreature {
        player: PlayerId,
        filter: CreatureSacrificeFilter,
        original_candidates: Vec<EffectObjectBinding>,
        chosen: EffectObjectBinding,
        path: Vec<u16>,
        canonical_path: Vec<u16>,
        expected_remaining_frames: Vec<EffectFrame>,
    },
    /// Authenticated post-answer frame for an accepted optional mana cost.
    /// Payment and installation of `then` happen together on the next engine
    /// advance, after the exact answered frame stack is revalidated.
    PayManaThen {
        player: PlayerId,
        colored: Vec<ManaColor>,
        generic: u8,
        then: Box<EffectOp>,
        path: Vec<u16>,
        canonical_path: Vec<u16>,
        expected_remaining_frames: Vec<EffectFrame>,
    },
    /// A publicly revealed exact library prefix awaiting or carrying its
    /// owner's graveyard order. `original_prefix` preserves top-to-bottom
    /// identity independently from the mutable chosen order.
    RevealedLibraryToGraveyardBatch {
        player: PlayerId,
        original_prefix: Vec<EffectObjectBinding>,
        objects: Vec<EffectObjectBinding>,
        order_resolved: bool,
        path: Vec<u16>,
    },
    /// Commits a public, incarnation-bound variable-size untap selection.
    UntapObjectsBatch {
        player: PlayerId,
        objects: Vec<EffectObjectBinding>,
        max_targets: u16,
        path: Vec<u16>,
    },
    /// Commits a private optional single-card search to the battlefield
    /// tapped while retaining the exact pre-search library order.
    SearchLibraryToBattlefieldTapped {
        player: PlayerId,
        filter: LibraryCardFilter,
        filter_fingerprint: u64,
        original_library: Vec<EffectObjectBinding>,
        selected: Option<EffectObjectBinding>,
        path: Vec<u16>,
        canonical_path: Vec<u16>,
    },
    /// Commits Duress's exact selected public hand incarnation.
    DiscardRevealedHandCard {
        player: PlayerId,
        original_hand: Vec<EffectObjectBinding>,
        eligible: Vec<EffectObjectBinding>,
        selected: EffectObjectBinding,
        path: Vec<u16>,
        canonical_path: Vec<u16>,
    },
    /// Authenticated post-answer start of Cleansing Wildfire's optional
    /// basic-land search.
    BeginSearchLibraryToBattlefieldTapped {
        player: PlayerId,
        filter: LibraryCardFilter,
        path: Vec<u16>,
        canonical_path: Vec<u16>,
        expected_remaining_frames: Vec<EffectFrame>,
    },
    /// Authenticated post-answer frame for Mesmeric Fiend's publicly
    /// revealed hand choice and exact historical linked exile.
    LinkedExileChosenHandCard {
        player: PlayerId,
        original_hand: Vec<EffectObjectBinding>,
        chosen: EffectObjectBinding,
        source: AbilitySourceContractV4,
        path: Vec<u16>,
        canonical_path: Vec<u16>,
        expected_remaining_frames: Vec<EffectFrame>,
    },
    EnterUndercityRoom {
        binding: InitiativeTriggerBindingV1,
        from_room: Option<UndercityRoomV1>,
        room: UndercityRoomV1,
        path: Vec<u16>,
        expected_remaining_frames: Vec<EffectFrame>,
    },
    ResolveUndercityThrone {
        binding: InitiativeTriggerBindingV1,
        original_library: Vec<EffectObjectBinding>,
        revealed_prefix: Vec<EffectObjectBinding>,
        candidates: Vec<EffectObjectBinding>,
        chosen: EffectObjectBinding,
        path: Vec<u16>,
        canonical_path: Vec<u16>,
        expected_remaining_frames: Vec<EffectFrame>,
    },
}

/// Completed private scry stages. A subset is canonicalized into original
/// prefix order before it enters this trusted frame, so the order in which
/// stage-one targets were selected can never leak into bottom ordering.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScryProgress {
    BottomSubsetChosen {
        bottom_subset: Vec<EffectObjectBinding>,
    },
    BottomOrderChosen {
        bottom_subset: Vec<EffectObjectBinding>,
        ordered_bottom: Vec<EffectObjectBinding>,
    },
    TopOrderChosen {
        bottom_subset: Vec<EffectObjectBinding>,
        ordered_bottom: Vec<EffectObjectBinding>,
        ordered_top: Vec<EffectObjectBinding>,
    },
}

/// Completed stages of a typed private top-library partition. The chosen
/// matching subset is canonicalized into original-prefix order, while the
/// remainder order is stored shallow-to-deep for direct bottom placement.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LibraryPartitionProgress {
    MatchingSubsetChosen {
        selected: Vec<EffectObjectBinding>,
    },
    RestOrderChosen {
        selected: Vec<EffectObjectBinding>,
        ordered_rest: Vec<EffectObjectBinding>,
    },
}

/// Binds a physical arena id to the exact incarnation selected when an effect
/// snapshotted it. Visibility is governed separately: public reveals expose
/// these bindings to both players, while private library and hand choices
/// expose their otherwise-hidden bindings only to the chooser. A
/// restored/stale continuation must never move a later incarnation that
/// happens to reuse the same stable `ObjectId`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EffectObjectBinding {
    pub object: ObjectId,
    pub expected_zone: Zone,
    pub expected_zone_change_count: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EffectTargetCandidate {
    pub target: Target,
    pub expected_object: Option<EffectObjectBinding>,
}

/// Internal reason/completion for a generic target-selection continuation.
/// Public schema-v4 projects these through already-reserved card-selection or
/// library-order purposes; no card-specific state or action identity is
/// introduced.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EffectTargetSelectionPurpose {
    OrderIntoGraveyard {
        preserve_known_identity: bool,
    },
    /// Orders a bound, otherwise-hidden library prefix for a mill batch.
    /// Only the milled cards' owner may inspect candidate identities while
    /// the choice is pending; the completed graveyard remains public.
    OrderMilledIntoGraveyard,
    /// Orders a privately looked-at library prefix. The original ordered
    /// bindings are retained independently from the mutable selected/legal
    /// partition so restore/tamper validation can require the exact prefix,
    /// not merely the same set of cards.
    OrderLookedLibraryTop {
        player: PlayerId,
        original_prefix: Vec<EffectObjectBinding>,
    },
    /// One of a repeated series of private, exact-one hand choices. The
    /// complete hand snapshot prevents a restored continuation from silently
    /// accepting a changed candidate pool. The next prompt is independent:
    /// after this choice commits, it snapshots the then-current hand anew.
    PutHandCardOnLibraryTop {
        player: PlayerId,
        original_hand: Vec<EffectObjectBinding>,
        total: u8,
        remaining: u8,
        prompt_index: u16,
        continuation_path: Vec<u16>,
        canonical_path: Vec<u16>,
    },
    /// One of the three private scry prompts. Stage one is an unordered,
    /// variable-size card selection; stages two and three are exact library
    /// orderings. Schema-v4 projects these through its existing
    /// CardSelection/LibraryOrder purposes and redacts identities from the
    /// non-chooser.
    ScryLibrary {
        player: PlayerId,
        requested_count: u8,
        original_library_len: u32,
        original_prefix: Vec<EffectObjectBinding>,
        stage: ScrySelectionStage,
        /// Redundant deterministic commitment to `stage`; see the frame's
        /// progress fingerprint for the trusted-snapshot threat boundary.
        stage_fingerprint: u64,
        canonical_path: Vec<u16>,
    },
    /// Optional zero-or-one result of a private, typed whole-library search.
    /// The complete original library binds exact order, membership, and
    /// incarnations; candidates are a canonical physical-card partition of
    /// that snapshot rather than name-deduplicated AIRL display entries.
    SearchLibraryToHand {
        player: PlayerId,
        filter: LibraryCardFilter,
        filter_fingerprint: u64,
        original_library: Vec<EffectObjectBinding>,
        canonical_path: Vec<u16>,
    },
    /// One of the two private prompts for a typed top-library partition:
    /// choose an optional matching subset, then order the unselected rest
    /// for the bottom. Both project through existing generic policy purposes.
    LookTopSelectByTypeToHandBottomRest {
        player: PlayerId,
        requested_count: u8,
        original_library_len: u32,
        card_type: CardType,
        original_prefix: Vec<EffectObjectBinding>,
        stage: LibraryPartitionSelectionStage,
        stage_fingerprint: u64,
        canonical_path: Vec<u16>,
    },
    /// Optional zero-through-N result of a private typed whole-library
    /// search. Selection order is retained for deterministic reveal and move
    /// ordering while candidates remain exact physical cards.
    SearchLibraryToHandMany {
        player: PlayerId,
        filter: LibraryCardFilter,
        filter_fingerprint: u64,
        original_library: Vec<EffectObjectBinding>,
        max_targets: u16,
        canonical_path: Vec<u16>,
    },
    /// Mandatory exact-one public graveyard choice made by that
    /// graveyard's owner. The full snapshot binds candidates/incarnations.
    ExileOneFromGraveyard {
        player: PlayerId,
        original_graveyard: Vec<EffectObjectBinding>,
        canonical_path: Vec<u16>,
    },
    /// Mandatory exact-one public choice from the matching subset of one
    /// player's fully bound graveyard.
    ExileOneMatchingFromGraveyard {
        player: PlayerId,
        card_type: CardType,
        original_graveyard: Vec<EffectObjectBinding>,
        candidates: Vec<EffectObjectBinding>,
        then: Box<EffectOp>,
        canonical_path: Vec<u16>,
    },
    /// Mandatory exact-one public creature sacrifice choice.
    SacrificeCreature {
        player: PlayerId,
        filter: CreatureSacrificeFilter,
        original_candidates: Vec<EffectObjectBinding>,
        canonical_path: Vec<u16>,
    },
    /// Orders one publicly revealed, exact library prefix into its owner's
    /// graveyard while retaining the original top-to-bottom binding for
    /// stale and tamper validation.
    OrderRevealedIntoGraveyard {
        player: PlayerId,
        original_prefix: Vec<EffectObjectBinding>,
    },
    /// Public zero-through-N selection of tapped lands by `chooser`.
    UntapLands {
        chooser: PlayerId,
        max_targets: u16,
        original_candidates: Vec<EffectObjectBinding>,
        canonical_path: Vec<u16>,
    },
    /// Optional zero-or-one result of a private whole-library search whose
    /// selected card enters the battlefield tapped.
    SearchLibraryToBattlefieldTapped {
        player: PlayerId,
        filter: LibraryCardFilter,
        filter_fingerprint: u64,
        original_library: Vec<EffectObjectBinding>,
        canonical_path: Vec<u16>,
    },
    /// Public exact-one selection from a hand that has just been revealed.
    DuressDiscard {
        player: PlayerId,
        original_hand: Vec<EffectObjectBinding>,
        eligible: Vec<EffectObjectBinding>,
        canonical_path: Vec<u16>,
    },
    /// Mandatory exact-one choice from a publicly revealed target hand.
    /// Only nonland cards are candidates, while the complete original hand
    /// and historical ability source authenticate the linked exile.
    LinkedExileNonlandFromRevealedHand {
        player: PlayerId,
        original_hand: Vec<EffectObjectBinding>,
        source: AbilitySourceContractV4,
        canonical_path: Vec<u16>,
    },
    /// Public exact-one creature selection from Throne's revealed top ten.
    UndercityThroneCreature {
        binding: InitiativeTriggerBindingV1,
        original_library: Vec<EffectObjectBinding>,
        revealed_prefix: Vec<EffectObjectBinding>,
        candidates: Vec<EffectObjectBinding>,
        canonical_path: Vec<u16>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScrySelectionStage {
    ChooseBottomSubset,
    OrderBottom {
        bottom_subset: Vec<EffectObjectBinding>,
    },
    OrderRetainedTop {
        bottom_subset: Vec<EffectObjectBinding>,
        ordered_bottom: Vec<EffectObjectBinding>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LibraryPartitionSelectionStage {
    ChooseMatchingSubset,
    OrderRest { selected: Vec<EffectObjectBinding> },
}

/// Internal completion semantics for a generic Boolean effect choice.
/// Public schema-v4 projects the shuffle use through its already-reserved
/// `BooleanChoicePurposeV4::Shuffle` variant.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EffectBooleanChoicePurpose {
    ShuffleLibrary {
        player: PlayerId,
    },
    CounterUnlessPaysGeneric {
        ward_target: StackTargetContractV4,
        targeting_stack_item: StackItemId,
        player: PlayerId,
        generic: u8,
    },
    CounterTargetUnlessPaysGeneric {
        target_contract: StackTargetContractV4,
        target_stack_item: StackItemId,
        player: PlayerId,
        generic: u8,
    },
    PayManaThen {
        player: PlayerId,
        colored: Vec<ManaColor>,
        generic: u8,
        then: Box<EffectOp>,
    },
    /// Cleansing Wildfire's controller-bound optional basic-land search.
    SearchLibraryToBattlefieldTapped {
        player: PlayerId,
        filter: LibraryCardFilter,
        canonical_path: Vec<u16>,
        expected_remaining_frames: Vec<EffectFrame>,
    },
    PayExileFromGraveyardThen {
        player: PlayerId,
        card_type: CardType,
        original_graveyard: Vec<EffectObjectBinding>,
        candidates: Vec<EffectObjectBinding>,
        then: Box<EffectOp>,
        canonical_path: Vec<u16>,
    },
}

/// Internal completion contract for an option choice. Public schema-v4
/// intentionally continues to expose only the generic option count/order;
/// the typed purpose makes trusted-snapshot validation independent of the
/// mutable option payloads themselves.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EffectOptionChoicePurpose {
    Generic,
    OwnerLibrarySecondOrBottom {
        object: EffectObjectBinding,
        owner: PlayerId,
        canonical_path: Vec<u16>,
        expected_remaining_frames: Vec<EffectFrame>,
    },
    ExploreNonlandTop {
        player: PlayerId,
        top: EffectObjectBinding,
        canonical_path: Vec<u16>,
    },
    /// A W/U/B/R/G choice whose public projection uses the already-reserved
    /// color semantic and action identity rather than generic options.
    ChooseColor {
        player: PlayerId,
        legal_colors: Vec<ManaColor>,
        canonical_path: Vec<u16>,
        expected_remaining_frames: Vec<EffectFrame>,
    },
    UndercityRoute {
        binding: InitiativeTriggerBindingV1,
        from_room: Option<UndercityRoomV1>,
        legal_rooms: Vec<UndercityRoomV1>,
        canonical_path: Vec<u16>,
        expected_remaining_frames: Vec<EffectFrame>,
    },
}

/// A policy-visible choice yielded by the generic effect interpreter. This is
/// intentionally typed and extensible: later library ordering, subset, Ward,
/// and Escape choices add variants here instead of adding card-specific
/// `EngineState::pending_*` fields.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PendingEffectChoice {
    ChooseOption {
        player: PlayerId,
        path: Vec<u16>,
        options: Vec<EffectOp>,
        purpose: EffectOptionChoicePurpose,
    },
    SelectTargets {
        player: PlayerId,
        path: Vec<u16>,
        selected: Vec<EffectTargetCandidate>,
        legal: Vec<EffectTargetCandidate>,
        min_targets: u16,
        max_targets: u16,
        ordered: bool,
        purpose: EffectTargetSelectionPurpose,
    },
    ChooseBoolean {
        player: PlayerId,
        path: Vec<u16>,
        default: Option<bool>,
        purpose: EffectBooleanChoicePurpose,
    },
}

impl PendingEffectChoice {
    pub fn player(&self) -> PlayerId {
        match self {
            PendingEffectChoice::ChooseOption { player, .. } => *player,
            PendingEffectChoice::SelectTargets { player, .. } => *player,
            PendingEffectChoice::ChooseBoolean { player, .. } => *player,
        }
    }

    pub fn structural_path(&self) -> &[u16] {
        match self {
            PendingEffectChoice::ChooseOption { path, .. }
            | PendingEffectChoice::SelectTargets { path, .. }
            | PendingEffectChoice::ChooseBoolean { path, .. } => path,
        }
    }
}

/// Full in-state continuation for one resolving stack item. The complete
/// `StackItem`, execution context, remaining frames, and active typed choice
/// all participate in clone/equality/hash/serde, so snapshot/restore cannot
/// lose or alias a mid-resolution decision.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EffectContinuation {
    pub resolving_item: StackItem,
    pub ctx: ExecCtx,
    pub frames: Vec<EffectFrame>,
    pub choice: Option<PendingEffectChoice>,
    #[serde(default)]
    pub answered_choice_guard: Option<EffectAnsweredChoiceGuard>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EffectAnsweredChoiceGuard {
    OwnerLibrarySecondOrBottom { frame: Box<EffectFrame> },
    CounterUnlessPaysGeneric { frame: Box<EffectFrame> },
    CounterTargetUnlessPaysGeneric { frame: Box<EffectFrame> },
    ExileOneFromGraveyard { frame: Box<EffectFrame> },
    ExileOneMatchingFromGraveyard { frame: Box<EffectFrame> },
    SacrificeCreature { frame: Box<EffectFrame> },
    PayManaThen { frame: Box<EffectFrame> },
    LinkedExileFromRevealedHand { frame: Box<EffectFrame> },
    SearchLibraryToBattlefieldTapped { frame: Box<EffectFrame> },
    UndercityRoute { frame: Box<EffectFrame> },
    UndercityThrone { frame: Box<EffectFrame> },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResumableProgress {
    Complete(Box<StackItem>),
    Suspended,
}

/// Everything an effect program needs to resolve symbolic refs against a
/// concrete game: which object it's running for, who controls it, and the
/// targets chosen when it was cast/activated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecCtx {
    /// Exact resolving stack incarnation. Direct unit-test contexts that do
    /// not originate from a real stack item leave this absent.
    #[serde(default)]
    pub stack_item_id: Option<StackItemId>,
    pub source: ObjectId,
    pub controller: PlayerId,
    pub targets: Vec<Target>,
    /// Cast/activation-time incarnation bindings parallel to `targets`.
    /// Player slots use `StackTargetContractV4::Player`; object slots must match before any
    /// individually guarded multi-target leaf may act on that object.
    #[serde(default)]
    pub target_contracts: Vec<StackTargetContractV4>,
    /// Cards discarded to pay this cast's mandatory additional cost (Grab
    /// the Prize), if any. Empty for everything else. Read by
    /// `EffectCond::DiscardedNonLandForCost`.
    pub discarded: Vec<ObjectId>,
    /// Exact historical objects used to pay this stack item's cost. Dynamic
    /// values read these frozen definitions, never the objects' later zones.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub paid_cost_refs: Vec<PaidCostRefV4>,
    /// Exact hidden-zone incarnation revealed to activate an ability. This
    /// remains frozen even if that card later changes zones while the
    /// ability is on the stack. Ninjutsu uses it to avoid following a later
    /// incarnation of the same physical card.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hidden_ability_source: Option<ObjectLinkV4>,
    /// Frozen nonspell source facts used by effects that explicitly consult
    /// last known information. Ordinary spell and direct-test contexts leave
    /// this absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ability_source_contract: Option<AbilitySourceContractV4>,
    /// True iff the spell/ability this resolution belongs to was kicked
    /// (`card_def::CardDef::kicker_cost`) -- carried on `state::StackItem::
    /// kicked` and copied in here by `engine::resolve_top_of_stack`, and
    /// (for the ETB trigger this spell's own resolution queues) propagated
    /// onto that trigger's own `trigger::PendingTrigger`/`StackItem` in turn.
    /// Read by `EffectCond::WasKicked`. `false` for every card without
    /// Kicker (the overwhelming majority), and for any ability/trigger not
    /// downstream of a kicked cast.
    pub kicked: bool,
    /// Optional additional cost paid for this exact cast, propagated to an
    /// ETB trigger when that trigger has an intervening-if dependency.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub optional_additional_cost_paid: Option<OptionalAdditionalCostDef>,
    /// X announced for the resolving spell. Non-X spells and direct unit
    /// contexts retain zero.
    #[serde(default, skip_serializing_if = "is_zero_u16")]
    pub x_value: u16,
}

fn is_zero_u16(value: &u16) -> bool {
    *value == 0
}

impl std::hash::Hash for ExecCtx {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Preserve the historical hot-path state hash for every pre-v12
        // continuation. The appended provenance participates only when a
        // program actually carries a paid object.
        std::hash::Hash::hash(&self.source, state);
        std::hash::Hash::hash(&self.controller, state);
        std::hash::Hash::hash(&self.targets, state);
        std::hash::Hash::hash(&self.target_contracts, state);
        std::hash::Hash::hash(&self.discarded, state);
        std::hash::Hash::hash(&self.kicked, state);
        if self.x_value != 0 {
            std::hash::Hash::hash(&0x785f_7661_6c75_655f_u64, state);
            std::hash::Hash::hash(&self.x_value, state);
        }
        if !self.paid_cost_refs.is_empty() {
            std::hash::Hash::hash(&0x7061_6964_5f72_6566_u64, state);
            std::hash::Hash::hash(&self.paid_cost_refs, state);
        }
        if let Some(source) = self.hidden_ability_source {
            std::hash::Hash::hash(&0x6869_6464_656e_5f73_u64, state);
            std::hash::Hash::hash(&source, state);
        }
        if let Some(contract) = self.ability_source_contract {
            std::hash::Hash::hash(&0x6162_696c_6974_795f_u64, state);
            std::hash::Hash::hash(&contract, state);
        }
        if let Some(kind) = self.optional_additional_cost_paid {
            std::hash::Hash::hash(&0x6f70_7469_6f6e_616c_u64, state);
            std::hash::Hash::hash(&kind, state);
        }
    }
}

/// Whether this program can yield a policy-visible choice anywhere in its
/// tree. Existing Burn/Rally programs stay on their frozen synchronous/legacy
/// continuation paths; explicit `Choice`, public partition ordering,
/// private library/hand ordering, and multi-card mill ordering enter the v4
/// interpreter.
pub fn contains_player_choice(op: &EffectOp) -> bool {
    match op {
        EffectOp::Sequence(ops) => ops.iter().any(contains_player_choice),
        EffectOp::Conditional { then, else_, .. } => {
            contains_player_choice(then) || contains_player_choice(else_)
        }
        EffectOp::Choice { options, .. } => {
            options.len() > 1 || options.iter().any(contains_player_choice)
        }
        // Whether the decision is needed depends on the revealed cards, but
        // the program must enter the resumable interpreter so a 2+ card
        // graveyard batch can yield its owner's ordering choice.
        EffectOp::RevealTopAndPartitionByType { .. } => true,
        EffectOp::RevealUntilCardTypeAndMill { .. } => true,
        EffectOp::MillCards { count, .. } => *count > 1,
        EffectOp::LookAtLibraryTopAndReorder { .. }
        | EffectOp::MayShuffleLibrary { .. }
        | EffectOp::PutCardsFromHandOnLibraryTop { .. }
        | EffectOp::Scry { .. }
        | EffectOp::SearchLibraryToHand { .. }
        | EffectOp::SearchLibraryToHandUpTo { .. }
        | EffectOp::UntapUpToLands { .. }
        | EffectOp::PutObjectInOwnersLibrarySecondOrBottom { .. }
        | EffectOp::CounterUnlessPaysGeneric { .. }
        | EffectOp::CounterTargetUnlessPaysGeneric { .. }
        | EffectOp::LookTopSelectByTypeToHandBottomRest { .. }
        | EffectOp::ExploreTarget { .. }
        | EffectOp::ExileOneFromPlayersGraveyard { .. }
        | EffectOp::MayExileFromPlayersGraveyardMatchingThen { .. }
        | EffectOp::SacrificeCreature { .. }
        | EffectOp::MayPayManaThen { .. }
        | EffectOp::DestroyTargetLandThenMaySearchBasicTapped { .. }
        | EffectOp::SearchLibraryToBattlefieldTapped { .. }
        | EffectOp::RevealTargetHandChooseNoncreatureNonlandDiscard { .. }
        | EffectOp::RevealHandChooseNonlandToLinkedExile { .. }
        | EffectOp::ReturnLinkedExiledCardToOwnersHand
        | EffectOp::PreventDamageFromChosenColorUntilEndOfTurn { .. }
        | EffectOp::ResolveInitiativeTrigger { .. }
        | EffectOp::ResolveUndercityThrone { .. } => true,
        _ => false,
    }
}

/// Starts a choice-bearing stack resolution and runs synchronously until it
/// either completes or yields a real player choice. Legacy-suspending leaves
/// are rejected up front in this first v4 slice: mixing them with remaining
/// frames would change the already-certified Burn/Rally completion timing.
pub fn begin_resumable_resolution(
    op: &EffectOp,
    ctx: &ExecCtx,
    resolving_item: StackItem,
    state: &mut GameState,
) -> Result<ResumableProgress, String> {
    if state.engine.pending_effect.is_some() {
        return Err("cannot begin an effect while another continuation is pending".to_string());
    }
    validate_resumable_program(op)?;
    state.engine.pending_effect = Some(EffectContinuation {
        resolving_item,
        ctx: ctx.clone(),
        frames: vec![EffectFrame::Program {
            op: op.clone(),
            path: Vec::new(),
        }],
        choice: None,
        answered_choice_guard: None,
    });
    drive_resumable(state)
}

/// Resumes after `choose_resumable_option` installed the selected branch.
pub fn resume_resumable_resolution(state: &mut GameState) -> Result<ResumableProgress, String> {
    if state.engine.pending_effect.is_none() {
        return Err("no effect continuation is pending".to_string());
    }
    drive_resumable(state)
}

/// Records one selected option without executing it. The next engine advance
/// owns resumption, preserving the usual step/advance separation and making a
/// snapshot taken immediately after the action deterministic too.
pub fn choose_resumable_option(state: &mut GameState, option_index: u16) -> Result<(), String> {
    validate_pending_effect_choice(state)?;
    let continuation = state
        .engine
        .pending_effect
        .as_mut()
        .ok_or("no effect continuation is pending")?;
    let choice = continuation
        .choice
        .take()
        .ok_or("the pending effect is not waiting for a choice")?;
    match choice {
        PendingEffectChoice::ChooseOption {
            player,
            mut path,
            options,
            purpose,
        } => {
            let index = option_index as usize;
            let Some(selected) = options.get(index).cloned() else {
                continuation.choice = Some(PendingEffectChoice::ChooseOption {
                    player,
                    path,
                    options,
                    purpose,
                });
                return Err(format!(
                    "effect option {option_index} is outside the available range"
                ));
            };
            match purpose {
                EffectOptionChoicePurpose::Generic => {
                    path.push(option_index);
                    continuation
                        .frames
                        .push(EffectFrame::Program { op: selected, path });
                }
                EffectOptionChoicePurpose::OwnerLibrarySecondOrBottom {
                    object,
                    owner,
                    canonical_path,
                    expected_remaining_frames,
                } => {
                    let placement = match selected {
                        EffectOp::PutBoundObjectInOwnersLibrary { placement, .. } => placement,
                        _ => {
                            return Err(
                                "validated owner-library option lost its bound move".to_string()
                            );
                        }
                    };
                    path.push(option_index);
                    let frame = EffectFrame::OwnerLibraryPlacement {
                        object,
                        owner,
                        placement,
                        option_index,
                        path: path.clone(),
                        canonical_path,
                        expected_remaining_frames,
                    };
                    continuation.answered_choice_guard =
                        Some(EffectAnsweredChoiceGuard::OwnerLibrarySecondOrBottom {
                            frame: Box::new(frame.clone()),
                        });
                    continuation.frames.push(frame);
                }
                EffectOptionChoicePurpose::ExploreNonlandTop { .. } => {
                    path.push(option_index);
                    continuation
                        .frames
                        .push(EffectFrame::Program { op: selected, path });
                }
                EffectOptionChoicePurpose::ChooseColor {
                    player: color_player,
                    legal_colors,
                    canonical_path,
                    expected_remaining_frames,
                } => {
                    let Some(&color) = legal_colors.get(index) else {
                        return Err("validated color choice lost its printed index".to_string());
                    };
                    if player != color_player
                        || path != canonical_path
                        || continuation.frames != expected_remaining_frames
                        || selected
                            != (EffectOp::InstallDamagePreventionFromColor {
                                player: color_player,
                                color,
                            })
                    {
                        return Err("color choice metadata changed before answer".to_string());
                    }
                    path.push(option_index);
                    continuation
                        .frames
                        .push(EffectFrame::Program { op: selected, path });
                }
                EffectOptionChoicePurpose::UndercityRoute {
                    binding,
                    from_room,
                    legal_rooms,
                    canonical_path,
                    expected_remaining_frames,
                } => {
                    let Some(&room) = legal_rooms.get(index) else {
                        return Err("validated Undercity route lost its printed index".to_string());
                    };
                    if player != binding.player
                        || path != canonical_path
                        || continuation.frames != expected_remaining_frames
                        || selected
                            != (EffectOp::EnterUndercityRoom {
                                binding,
                                from_room,
                                room,
                            })
                    {
                        return Err("Undercity route metadata changed before answer".to_string());
                    }
                    path.push(option_index);
                    let frame = EffectFrame::EnterUndercityRoom {
                        binding,
                        from_room,
                        room,
                        path,
                        expected_remaining_frames,
                    };
                    continuation.answered_choice_guard =
                        Some(EffectAnsweredChoiceGuard::UndercityRoute {
                            frame: Box::new(frame.clone()),
                        });
                    continuation.frames.push(frame);
                }
            }
            Ok(())
        }
        PendingEffectChoice::SelectTargets { .. } | PendingEffectChoice::ChooseBoolean { .. } => {
            continuation.choice = Some(choice);
            Err("the pending effect is not waiting for an option".to_string())
        }
    }
}

/// Records one target in an effect-owned selection. Exact-count selections
/// auto-append every forced remaining target, so an N-card ordering exposes
/// only N-1 picks. Validation precedes mutation, making stale, duplicate, or
/// wrong-shape actions byte-for-byte nonmutating.
pub fn choose_resumable_target(state: &mut GameState, target: Target) -> Result<(), String> {
    validate_pending_effect_choice(state)?;
    let choice = state
        .engine
        .pending_effect
        .as_ref()
        .and_then(|pending| pending.choice.as_ref())
        .ok_or("no effect continuation choice is pending")?;
    let PendingEffectChoice::SelectTargets {
        legal,
        selected,
        max_targets,
        purpose,
        ..
    } = choice
    else {
        return Err("the pending effect is not waiting for a target selection".to_string());
    };
    if selected.len() >= usize::from(*max_targets) {
        return Err("the pending effect target selection is already full".to_string());
    }
    let Some(position) = legal
        .iter()
        .position(|candidate| candidate.target == target)
    else {
        return Err(format!("{target:?} is not a legal remaining effect target"));
    };
    validate_effect_target_candidate(state, &legal[position])?;
    // A library search always ends in a shuffle: the shuffle must be
    // authorizable before this selection mutates. The token is discarded;
    // the SearchLibraryToHand frame preflights again and commits.
    if let EffectTargetSelectionPurpose::SearchLibraryToHand {
        player: search_player,
        ..
    }
    | EffectTargetSelectionPurpose::SearchLibraryToHandMany {
        player: search_player,
        ..
    }
    | EffectTargetSelectionPurpose::SearchLibraryToBattlefieldTapped {
        player: search_player,
        ..
    }
    | EffectTargetSelectionPurpose::UndercityThroneCreature {
        binding:
            InitiativeTriggerBindingV1 {
                player: search_player,
                ..
            },
        ..
    } = purpose
    {
        let token = state
            .preflight_library_shuffle(*search_player)
            .map_err(|error| error.to_string())?;
        drop(token);
    }

    let continuation = state.engine.pending_effect.as_mut().unwrap();
    let PendingEffectChoice::SelectTargets {
        selected,
        legal,
        min_targets,
        max_targets,
        ordered,
        ..
    } = continuation.choice.as_mut().unwrap()
    else {
        unreachable!("validated target-selection choice above")
    };
    selected.push(legal.remove(position));

    let required = usize::from(*min_targets).saturating_sub(selected.len());
    if (*ordered && required == 1 && legal.len() == 1)
        || (!*ordered && required > 0 && required == legal.len())
    {
        selected.append(legal);
    }
    if selected.len() == usize::from(*max_targets) {
        complete_resumable_target_selection(continuation)?;
    }
    Ok(())
}

/// Finishes a generic variable-count selection once its minimum has been
/// met. Winding Way's graveyard ordering has `min == max`, so its forced
/// final card auto-completes and this action is never legal there.
pub fn finish_resumable_target_selection(state: &mut GameState) -> Result<(), String> {
    validate_pending_effect_choice(state)?;
    let choice = state
        .engine
        .pending_effect
        .as_ref()
        .and_then(|pending| pending.choice.as_ref())
        .ok_or("no effect continuation choice is pending")?;
    let PendingEffectChoice::SelectTargets {
        selected,
        min_targets,
        ..
    } = choice
    else {
        return Err("the pending effect is not waiting for a target selection".to_string());
    };
    if selected.len() < usize::from(*min_targets) {
        return Err("the pending effect target selection has not reached its minimum".to_string());
    }
    // Completing a library search (including a zero-selection completion)
    // still ends in a shuffle; authorize it before completion mutates the
    // continuation. The token is discarded; the frame commits later.
    if let Some(PendingEffectChoice::SelectTargets {
        purpose:
            EffectTargetSelectionPurpose::SearchLibraryToHand {
                player: search_player,
                ..
            }
            | EffectTargetSelectionPurpose::SearchLibraryToHandMany {
                player: search_player,
                ..
            }
            | EffectTargetSelectionPurpose::SearchLibraryToBattlefieldTapped {
                player: search_player,
                ..
            }
            | EffectTargetSelectionPurpose::UndercityThroneCreature {
                binding:
                    InitiativeTriggerBindingV1 {
                        player: search_player,
                        ..
                    },
                ..
            },
        ..
    }) = state
        .engine
        .pending_effect
        .as_ref()
        .and_then(|pending| pending.choice.as_ref())
    {
        let token = state
            .preflight_library_shuffle(*search_player)
            .map_err(|error| error.to_string())?;
        drop(token);
    }
    complete_resumable_target_selection(state.engine.pending_effect.as_mut().unwrap())
}

/// Records one generic Boolean answer without executing its consequence.
/// The next engine advance owns the accepted shuffle, keeping `step()` a
/// pure continuation transition and making post-action snapshots stable.
pub fn choose_resumable_boolean(state: &mut GameState, value: bool) -> Result<(), String> {
    validate_pending_effect_choice(state)?;
    if value {
        // An accepted shuffle must be authorizable before the pending choice
        // is consumed or any continuation state mutates. The token is
        // discarded: the later ShuffleLibrary frame preflights again and
        // commits. Declining (`false`) stays legal at an exhausted ordinal
        // and consumes nothing.
        if let Some(PendingEffectChoice::ChooseBoolean {
            purpose:
                EffectBooleanChoicePurpose::ShuffleLibrary {
                    player: library_player,
                },
            ..
        }) = state
            .engine
            .pending_effect
            .as_ref()
            .and_then(|pending| pending.choice.as_ref())
        {
            let token = state
                .preflight_library_shuffle(*library_player)
                .map_err(|error| error.to_string())?;
            drop(token);
        }
        if let Some(PendingEffectChoice::ChooseBoolean {
            purpose:
                EffectBooleanChoicePurpose::SearchLibraryToBattlefieldTapped {
                    player: library_player,
                    ..
                },
            ..
        }) = state
            .engine
            .pending_effect
            .as_ref()
            .and_then(|pending| pending.choice.as_ref())
        {
            let token = state
                .preflight_library_shuffle(*library_player)
                .map_err(|error| error.to_string())?;
            drop(token);
        }
        if let Some(PendingEffectChoice::ChooseBoolean {
            purpose:
                EffectBooleanChoicePurpose::PayManaThen {
                    player,
                    colored,
                    generic,
                    ..
                },
            ..
        }) = state
            .engine
            .pending_effect
            .as_ref()
            .and_then(|pending| pending.choice.as_ref())
        {
            if !crate::engine::can_pay_effect_mana(*player, colored, *generic, state) {
                return Err("optional effect mana cost is no longer payable".to_string());
            }
        }
    }
    let continuation = state
        .engine
        .pending_effect
        .as_mut()
        .ok_or("no effect continuation is pending")?;
    let choice = continuation
        .choice
        .take()
        .ok_or("the pending effect is not waiting for a choice")?;
    match choice {
        PendingEffectChoice::ChooseBoolean {
            player,
            mut path,
            default,
            purpose,
        } => {
            path.push(u16::from(value));
            match purpose {
                EffectBooleanChoicePurpose::ShuffleLibrary {
                    player: library_player,
                } => {
                    if player != library_player {
                        continuation.choice = Some(PendingEffectChoice::ChooseBoolean {
                            player,
                            path,
                            default,
                            purpose: EffectBooleanChoicePurpose::ShuffleLibrary {
                                player: library_player,
                            },
                        });
                        return Err(
                            "shuffle choice player does not own the selected library".to_string()
                        );
                    }
                    if value {
                        continuation.frames.push(EffectFrame::ShuffleLibrary {
                            player: library_player,
                            path,
                        });
                    }
                }
                EffectBooleanChoicePurpose::CounterUnlessPaysGeneric {
                    ward_target,
                    targeting_stack_item,
                    player: payer,
                    generic,
                } => {
                    let StackTargetContractV4::Object {
                        object: ward_source,
                        ..
                    } = ward_target
                    else {
                        return Err("Ward Boolean choice lost its permanent binding".to_string());
                    };
                    if player != payer
                        || continuation.resolving_item.source != ward_source
                        || path.as_slice() != [u16::from(value)]
                        || !continuation.frames.is_empty()
                    {
                        continuation.choice = Some(PendingEffectChoice::ChooseBoolean {
                            player,
                            path,
                            default,
                            purpose,
                        });
                        return Err("Ward Boolean choice metadata is inconsistent".to_string());
                    }
                    let frame = EffectFrame::ResolveCounterUnlessPaysGeneric {
                        ward_target,
                        targeting_stack_item,
                        player: payer,
                        generic,
                        pay: value,
                        path,
                    };
                    continuation.answered_choice_guard =
                        Some(EffectAnsweredChoiceGuard::CounterUnlessPaysGeneric {
                            frame: Box::new(frame.clone()),
                        });
                    continuation.frames.push(frame);
                }
                EffectBooleanChoicePurpose::CounterTargetUnlessPaysGeneric {
                    target_contract,
                    target_stack_item,
                    player: payer,
                    generic,
                } => {
                    if player != payer
                        || path.as_slice() != [u16::from(value)]
                        || !continuation.frames.is_empty()
                    {
                        continuation.choice = Some(PendingEffectChoice::ChooseBoolean {
                            player,
                            path,
                            default,
                            purpose,
                        });
                        return Err("counter-unless-pay Boolean choice metadata is inconsistent"
                            .to_string());
                    }
                    let frame = EffectFrame::ResolveCounterTargetUnlessPaysGeneric {
                        target_contract,
                        target_stack_item,
                        player: payer,
                        generic,
                        pay: value,
                        path,
                    };
                    continuation.answered_choice_guard =
                        Some(EffectAnsweredChoiceGuard::CounterTargetUnlessPaysGeneric {
                            frame: Box::new(frame.clone()),
                        });
                    continuation.frames.push(frame);
                }
                EffectBooleanChoicePurpose::PayManaThen {
                    player: payer,
                    colored,
                    generic,
                    then,
                } => {
                    if player != payer {
                        continuation.choice = Some(PendingEffectChoice::ChooseBoolean {
                            player,
                            path,
                            default,
                            purpose: EffectBooleanChoicePurpose::PayManaThen {
                                player: payer,
                                colored,
                                generic,
                                then,
                            },
                        });
                        return Err("mana-payment choice player mismatch".to_string());
                    }
                    if value {
                        let mut canonical_path = path.clone();
                        canonical_path.pop();
                        let expected_remaining_frames = continuation.frames.clone();
                        let frame = EffectFrame::PayManaThen {
                            player: payer,
                            colored,
                            generic,
                            then,
                            path,
                            canonical_path,
                            expected_remaining_frames,
                        };
                        continuation.answered_choice_guard =
                            Some(EffectAnsweredChoiceGuard::PayManaThen {
                                frame: Box::new(frame.clone()),
                            });
                        continuation.frames.push(frame);
                    }
                }
                EffectBooleanChoicePurpose::SearchLibraryToBattlefieldTapped {
                    player: library_player,
                    filter,
                    canonical_path,
                    expected_remaining_frames,
                } => {
                    if player != library_player || continuation.frames != expected_remaining_frames
                    {
                        continuation.choice = Some(PendingEffectChoice::ChooseBoolean {
                            player,
                            path,
                            default,
                            purpose: EffectBooleanChoicePurpose::SearchLibraryToBattlefieldTapped {
                                player: library_player,
                                filter,
                                canonical_path,
                                expected_remaining_frames,
                            },
                        });
                        return Err(
                            "battlefield-search choice player or remainder changed".to_string()
                        );
                    }
                    let mut expected_path = canonical_path.clone();
                    expected_path.push(u16::from(value));
                    if path != expected_path {
                        return Err("battlefield-search Boolean path changed".to_string());
                    }
                    if value {
                        let frame = EffectFrame::BeginSearchLibraryToBattlefieldTapped {
                            player: library_player,
                            filter,
                            path,
                            canonical_path,
                            expected_remaining_frames,
                        };
                        continuation.answered_choice_guard = Some(
                            EffectAnsweredChoiceGuard::SearchLibraryToBattlefieldTapped {
                                frame: Box::new(frame.clone()),
                            },
                        );
                        continuation.frames.push(frame);
                    }
                }
                EffectBooleanChoicePurpose::PayExileFromGraveyardThen {
                    player: graveyard_player,
                    card_type,
                    original_graveyard,
                    candidates,
                    then,
                    canonical_path,
                } => {
                    if player != graveyard_player {
                        continuation.choice = Some(PendingEffectChoice::ChooseBoolean {
                            player,
                            path,
                            default,
                            purpose: EffectBooleanChoicePurpose::PayExileFromGraveyardThen {
                                player: graveyard_player,
                                card_type,
                                original_graveyard,
                                candidates,
                                then,
                                canonical_path,
                            },
                        });
                        return Err("filtered graveyard-exile choice player mismatch".to_string());
                    }
                    if value {
                        match candidates.len() {
                            0 => {
                                return Err(
                                    "accepted filtered graveyard-exile cost has no candidate"
                                        .to_string(),
                                )
                            }
                            1 => {
                                let chosen = candidates[0];
                                let expected_remaining_frames = continuation.frames.clone();
                                let frame = EffectFrame::ExileChosenMatchingGraveyardCard {
                                    player: graveyard_player,
                                    card_type,
                                    original_graveyard,
                                    candidates,
                                    chosen,
                                    then,
                                    path,
                                    canonical_path,
                                    expected_remaining_frames,
                                };
                                continuation.answered_choice_guard = Some(
                                    EffectAnsweredChoiceGuard::ExileOneMatchingFromGraveyard {
                                        frame: Box::new(frame.clone()),
                                    },
                                );
                                continuation.frames.push(frame);
                            }
                            _ => {
                                stage_matching_graveyard_exile_choice(
                                    continuation,
                                    graveyard_player,
                                    card_type,
                                    original_graveyard,
                                    candidates,
                                    then,
                                    path,
                                    canonical_path,
                                );
                            }
                        }
                    }
                }
            }
            Ok(())
        }
        PendingEffectChoice::ChooseOption { .. } | PendingEffectChoice::SelectTargets { .. } => {
            continuation.choice = Some(choice);
            Err("the pending effect is not waiting for a Boolean choice".to_string())
        }
    }
}

fn complete_resumable_target_selection(
    continuation: &mut EffectContinuation,
) -> Result<(), String> {
    let choice = continuation
        .choice
        .take()
        .ok_or("no effect continuation choice is pending")?;
    let PendingEffectChoice::SelectTargets {
        path,
        selected,
        purpose,
        ..
    } = choice
    else {
        continuation.choice = Some(choice);
        return Err("the pending effect is not waiting for a target selection".to_string());
    };
    let mut objects = selected
        .into_iter()
        .map(|candidate| {
            let binding = candidate.expected_object.ok_or_else(|| {
                "zone-order selection target lacks an object-incarnation binding".to_string()
            })?;
            if candidate.target != Target::Object(binding.object) {
                return Err("zone-order selection target/binding mismatch".to_string());
            }
            Ok(binding)
        })
        .collect::<Result<Vec<_>, String>>()?;
    match purpose {
        EffectTargetSelectionPurpose::OrderIntoGraveyard {
            preserve_known_identity,
        } => continuation.frames.push(EffectFrame::MoveObjectsBatch {
            objects,
            to_zone: Zone::Graveyard,
            preserve_known_identity,
            order_resolved: true,
            path,
        }),
        EffectTargetSelectionPurpose::OrderMilledIntoGraveyard => {
            continuation.frames.push(EffectFrame::MillLibraryBatch {
                objects,
                order_resolved: true,
                path,
            });
        }
        EffectTargetSelectionPurpose::OrderRevealedIntoGraveyard {
            player,
            original_prefix,
        } => {
            continuation
                .frames
                .push(EffectFrame::RevealedLibraryToGraveyardBatch {
                    player,
                    original_prefix,
                    objects,
                    order_resolved: true,
                    path,
                });
        }
        EffectTargetSelectionPurpose::OrderLookedLibraryTop {
            player,
            original_prefix,
        } => {
            // AIRL's ordered-card chooser treats the first explicit pick as
            // deepest and the forced final card as topmost. `selected` is in
            // pick order, while the state library is top-to-bottom.
            objects.reverse();
            continuation.frames.push(EffectFrame::ReorderLibraryTop {
                player,
                expected_prefix: original_prefix,
                ordered: objects,
                path,
            });
        }
        EffectTargetSelectionPurpose::PutHandCardOnLibraryTop {
            player,
            original_hand,
            total,
            remaining,
            prompt_index,
            continuation_path,
            canonical_path,
        } => {
            if objects.len() != 1 {
                return Err("hand-to-library prompt did not select exactly one card".to_string());
            }
            validate_hand_to_library_progress(
                total,
                remaining,
                prompt_index,
                &continuation_path,
                &canonical_path,
            )?;
            if remaining == 0 {
                return Err("completed hand-to-library progress cannot own a prompt".to_string());
            }
            let mut expected_choice_path = canonical_path.clone();
            expected_choice_path.push(prompt_index);
            if path != expected_choice_path {
                return Err("hand-to-library prompt structural path changed".to_string());
            }
            continuation
                .frames
                .push(EffectFrame::PutCardsFromHandOnLibraryTop {
                    player,
                    total,
                    remaining,
                    prompt_index,
                    expected_hand: original_hand,
                    chosen: objects.pop(),
                    path: continuation_path,
                    canonical_path,
                });
        }
        EffectTargetSelectionPurpose::ScryLibrary {
            player,
            requested_count,
            original_library_len,
            original_prefix,
            stage,
            stage_fingerprint,
            canonical_path,
        } => {
            validate_scry_bound_metadata(requested_count, original_library_len, &original_prefix)?;
            if stage_fingerprint != scry_stage_fingerprint(&stage) {
                return Err("scry prompt stage fingerprint changed".to_string());
            }
            let mut expected_choice_path = canonical_path.clone();
            expected_choice_path.push(scry_stage_tag(&stage));
            if path != expected_choice_path {
                return Err("scry prompt structural path changed".to_string());
            }
            let progress = match stage {
                ScrySelectionStage::ChooseBottomSubset => {
                    let bottom_subset = canonicalize_scry_subset(&original_prefix, &objects)?;
                    ScryProgress::BottomSubsetChosen { bottom_subset }
                }
                ScrySelectionStage::OrderBottom { bottom_subset } => {
                    validate_exact_binding_permutation(
                        &bottom_subset,
                        &objects,
                        "scry bottom order",
                    )?;
                    ScryProgress::BottomOrderChosen {
                        bottom_subset,
                        ordered_bottom: objects,
                    }
                }
                ScrySelectionStage::OrderRetainedTop {
                    bottom_subset,
                    ordered_bottom,
                } => {
                    validate_exact_binding_permutation(
                        &bottom_subset,
                        &ordered_bottom,
                        "scry ordered bottom",
                    )?;
                    let retained = scry_retained_prefix(&original_prefix, &bottom_subset)?;
                    validate_exact_binding_permutation(
                        &retained,
                        &objects,
                        "scry retained-top order",
                    )?;
                    // Like Ponder, this prompt's first explicit selection is
                    // deepest and the forced final card is topmost.
                    objects.reverse();
                    ScryProgress::TopOrderChosen {
                        bottom_subset,
                        ordered_bottom,
                        ordered_top: objects,
                    }
                }
            };
            let progress_fingerprint = scry_progress_fingerprint(&progress);
            continuation.frames.push(EffectFrame::ScryLibrary {
                player,
                requested_count,
                original_library_len,
                original_prefix,
                progress,
                progress_fingerprint,
                path: canonical_path.clone(),
                canonical_path,
            });
        }
        EffectTargetSelectionPurpose::SearchLibraryToHand {
            player,
            filter,
            filter_fingerprint,
            original_library,
            canonical_path,
        } => {
            if path != canonical_path {
                return Err("library-search prompt structural path changed".to_string());
            }
            if objects.len() > 1 {
                return Err("library search selected more than one card".to_string());
            }
            continuation.frames.push(EffectFrame::SearchLibraryToHand {
                player,
                filter,
                filter_fingerprint,
                original_library,
                selected: objects.pop(),
                path: canonical_path.clone(),
                canonical_path,
            });
        }
        EffectTargetSelectionPurpose::SearchLibraryToBattlefieldTapped {
            player,
            filter,
            filter_fingerprint,
            original_library,
            canonical_path,
        } => {
            if path != canonical_path {
                return Err("battlefield library-search prompt structural path changed".to_string());
            }
            if objects.len() > 1 {
                return Err("battlefield library search selected more than one card".to_string());
            }
            continuation
                .frames
                .push(EffectFrame::SearchLibraryToBattlefieldTapped {
                    player,
                    filter,
                    filter_fingerprint,
                    original_library,
                    selected: objects.pop(),
                    path: canonical_path.clone(),
                    canonical_path,
                });
        }
        EffectTargetSelectionPurpose::DuressDiscard {
            player,
            original_hand,
            eligible,
            canonical_path,
        } => {
            if path != canonical_path || objects.len() != 1 {
                return Err("Duress discard prompt changed path or cardinality".to_string());
            }
            continuation
                .frames
                .push(EffectFrame::DiscardRevealedHandCard {
                    player,
                    original_hand,
                    eligible,
                    selected: objects[0],
                    path: canonical_path.clone(),
                    canonical_path,
                });
        }
        EffectTargetSelectionPurpose::LookTopSelectByTypeToHandBottomRest {
            player,
            requested_count,
            original_library_len,
            card_type,
            original_prefix,
            stage,
            stage_fingerprint,
            canonical_path,
        } => {
            validate_library_partition_bound_metadata(
                requested_count,
                original_library_len,
                &original_prefix,
            )?;
            if stage_fingerprint != library_partition_stage_fingerprint(&stage) {
                return Err("library-partition prompt stage fingerprint changed".to_string());
            }
            let mut expected_choice_path = canonical_path.clone();
            expected_choice_path.push(library_partition_stage_tag(&stage));
            if path != expected_choice_path {
                return Err("library-partition prompt structural path changed".to_string());
            }
            let progress = match stage {
                LibraryPartitionSelectionStage::ChooseMatchingSubset => {
                    let selected = canonicalize_binding_subset(&original_prefix, &objects)?;
                    LibraryPartitionProgress::MatchingSubsetChosen { selected }
                }
                LibraryPartitionSelectionStage::OrderRest { selected } => {
                    validate_canonical_binding_subset(&original_prefix, &selected)?;
                    let rest = binding_partition_rest(&original_prefix, &selected)?;
                    validate_exact_binding_permutation(
                        &rest,
                        &objects,
                        "library-partition ordered rest",
                    )?;
                    LibraryPartitionProgress::RestOrderChosen {
                        selected,
                        ordered_rest: objects,
                    }
                }
            };
            let progress_fingerprint = library_partition_progress_fingerprint(&progress);
            continuation
                .frames
                .push(EffectFrame::LookTopSelectByTypeToHandBottomRest {
                    player,
                    requested_count,
                    original_library_len,
                    card_type,
                    original_prefix,
                    progress,
                    progress_fingerprint,
                    path: canonical_path.clone(),
                    canonical_path,
                });
        }
        EffectTargetSelectionPurpose::SearchLibraryToHandMany {
            player,
            filter,
            filter_fingerprint,
            original_library,
            max_targets,
            canonical_path,
        } => {
            if path != canonical_path {
                return Err("multi-card library-search prompt structural path changed".to_string());
            }
            if objects.len() > usize::from(max_targets) {
                return Err("multi-card library search exceeded its maximum".to_string());
            }
            continuation
                .frames
                .push(EffectFrame::SearchLibraryToHandMany {
                    player,
                    filter,
                    filter_fingerprint,
                    original_library,
                    selected: objects,
                    max_targets,
                    path: canonical_path.clone(),
                    canonical_path,
                });
        }
        EffectTargetSelectionPurpose::ExileOneFromGraveyard {
            player,
            original_graveyard,
            canonical_path,
        } => {
            if path != canonical_path || objects.len() != 1 {
                return Err("graveyard exile choice changed path or cardinality".to_string());
            }
            let expected_remaining_frames = continuation.frames.clone();
            let frame = EffectFrame::ExileChosenGraveyardCard {
                player,
                original_graveyard,
                chosen: objects[0],
                path: canonical_path.clone(),
                canonical_path,
                expected_remaining_frames,
            };
            continuation.answered_choice_guard =
                Some(EffectAnsweredChoiceGuard::ExileOneFromGraveyard {
                    frame: Box::new(frame.clone()),
                });
            continuation.frames.push(frame);
        }
        EffectTargetSelectionPurpose::ExileOneMatchingFromGraveyard {
            player,
            card_type,
            original_graveyard,
            candidates,
            then,
            canonical_path,
        } => {
            let mut expected_path = canonical_path.clone();
            expected_path.push(1);
            if path != expected_path || objects.len() != 1 || !candidates.contains(&objects[0]) {
                return Err(
                    "filtered graveyard exile choice changed path, cardinality, or candidate"
                        .to_string(),
                );
            }
            let expected_remaining_frames = continuation.frames.clone();
            let frame = EffectFrame::ExileChosenMatchingGraveyardCard {
                player,
                card_type,
                original_graveyard,
                candidates,
                chosen: objects[0],
                then,
                path,
                canonical_path,
                expected_remaining_frames,
            };
            continuation.answered_choice_guard =
                Some(EffectAnsweredChoiceGuard::ExileOneMatchingFromGraveyard {
                    frame: Box::new(frame.clone()),
                });
            continuation.frames.push(frame);
        }
        EffectTargetSelectionPurpose::SacrificeCreature {
            player,
            filter,
            original_candidates,
            canonical_path,
        } => {
            if path != canonical_path
                || objects.len() != 1
                || !original_candidates.contains(&objects[0])
            {
                return Err(
                    "creature-sacrifice choice changed path, cardinality, or candidate".to_string(),
                );
            }
            let expected_remaining_frames = continuation.frames.clone();
            let frame = EffectFrame::SacrificeChosenCreature {
                player,
                filter,
                original_candidates,
                chosen: objects[0],
                path: canonical_path.clone(),
                canonical_path,
                expected_remaining_frames,
            };
            continuation.answered_choice_guard =
                Some(EffectAnsweredChoiceGuard::SacrificeCreature {
                    frame: Box::new(frame.clone()),
                });
            continuation.frames.push(frame);
        }
        EffectTargetSelectionPurpose::UntapLands {
            chooser,
            max_targets,
            original_candidates,
            canonical_path,
        } => {
            if path != canonical_path {
                return Err("land-untap prompt structural path changed".to_string());
            }
            if objects.len() > usize::from(max_targets) {
                return Err("land-untap selection exceeded its maximum".to_string());
            }
            if objects
                .iter()
                .any(|binding| !original_candidates.contains(binding))
            {
                return Err("land-untap selection escaped its original candidates".to_string());
            }
            continuation.frames.push(EffectFrame::UntapObjectsBatch {
                player: chooser,
                objects,
                max_targets,
                path: canonical_path,
            });
        }
        EffectTargetSelectionPurpose::LinkedExileNonlandFromRevealedHand {
            player,
            original_hand,
            source,
            canonical_path,
        } => {
            if path != canonical_path || objects.len() != 1 {
                return Err("linked-exile choice changed path or cardinality".to_string());
            }
            let expected_remaining_frames = continuation.frames.clone();
            let frame = EffectFrame::LinkedExileChosenHandCard {
                player,
                original_hand,
                chosen: objects[0],
                source,
                path: canonical_path.clone(),
                canonical_path,
                expected_remaining_frames,
            };
            continuation.answered_choice_guard =
                Some(EffectAnsweredChoiceGuard::LinkedExileFromRevealedHand {
                    frame: Box::new(frame.clone()),
                });
            continuation.frames.push(frame);
        }
        EffectTargetSelectionPurpose::UndercityThroneCreature {
            binding,
            original_library,
            revealed_prefix,
            candidates,
            canonical_path,
        } => {
            if path != canonical_path || objects.len() != 1 || !candidates.contains(&objects[0]) {
                return Err("Undercity Throne choice changed path or candidate".to_string());
            }
            let expected_remaining_frames = continuation.frames.clone();
            let frame = EffectFrame::ResolveUndercityThrone {
                binding,
                original_library,
                revealed_prefix,
                candidates,
                chosen: objects[0],
                path: canonical_path.clone(),
                canonical_path,
                expected_remaining_frames,
            };
            continuation.answered_choice_guard = Some(EffectAnsweredChoiceGuard::UndercityThrone {
                frame: Box::new(frame.clone()),
            });
            continuation.frames.push(frame);
        }
    }
    Ok(())
}

fn validate_effect_target_candidate(
    state: &GameState,
    candidate: &EffectTargetCandidate,
) -> Result<(), String> {
    let Some(binding) = candidate.expected_object else {
        return Ok(());
    };
    if candidate.target != Target::Object(binding.object) {
        return Err("effect target/binding mismatch".to_string());
    }
    let object = state
        .objects
        .try_get(binding.object)
        .ok_or_else(|| format!("effect target object {} no longer exists", binding.object.0))?;
    if object.zone != binding.expected_zone
        || object.zone_change_count != binding.expected_zone_change_count
    {
        return Err(format!(
            "effect target object {} changed incarnation: expected {:?}/{} but found {:?}/{}",
            binding.object.0,
            binding.expected_zone,
            binding.expected_zone_change_count,
            object.zone,
            object.zone_change_count
        ));
    }
    Ok(())
}

fn validate_owner_library_target_binding(
    state: &GameState,
    pending: &EffectContinuation,
    object: EffectObjectBinding,
    owner: PlayerId,
) -> Result<(), String> {
    if pending.resolving_item.v4.target_spec != Some(crate::card_def::TargetSpec::NonlandPermanent)
        || pending.ctx.targets.as_slice() != [Target::Object(object.object)]
    {
        return Err("owner-library choice no longer matches the resolving target".to_string());
    }
    let [StackTargetContractV4::Object {
        object: contract_object,
        card_def,
        owner: contract_owner,
        zone,
        zone_change_count,
        ..
    }] = pending.ctx.target_contracts.as_slice()
    else {
        return Err("owner-library choice lost its object target contract".to_string());
    };
    if *contract_object != object.object
        || *contract_owner != owner
        || *zone != Zone::Battlefield
        || *zone_change_count != object.expected_zone_change_count
        || object.expected_zone != Zone::Battlefield
    {
        return Err("owner-library choice changed its cast-time target binding".to_string());
    }
    validate_effect_object_binding(state, object)?;
    let live = state.objects.get(object.object);
    if live.owner != owner
        || live.card_def != *card_def
        || crate::card_def::CARD_DEFS[live.card_def as usize]
            .has_type(crate::card_def::CardType::Land)
    {
        return Err("owner-library choice changed its target owner or definition".to_string());
    }
    let indexed_count = [PlayerId::P0, PlayerId::P1]
        .into_iter()
        .flat_map(|player| state.players[player.index()].battlefield.iter())
        .filter(|&&candidate| candidate == object.object)
        .count();
    if indexed_count != 1
        || !state.players[live.controller.index()]
            .battlefield
            .contains(&object.object)
    {
        return Err(
            "owner-library target is not indexed once under its live controller".to_string(),
        );
    }
    Ok(())
}

fn validate_owner_library_placement_frame(
    state: &GameState,
    pending: &EffectContinuation,
    frame: &EffectFrame,
) -> Result<(), String> {
    let EffectFrame::OwnerLibraryPlacement {
        object,
        owner,
        placement,
        option_index,
        path,
        canonical_path,
        expected_remaining_frames,
    } = frame
    else {
        return Err("owner-library answer guard does not contain its typed frame".to_string());
    };
    let expected_placement = match option_index {
        0 => event::LibraryPlacement::SecondFromTop,
        1 => event::LibraryPlacement::Bottom,
        _ => return Err("owner-library answered option index is outside 0..2".to_string()),
    };
    if *placement != expected_placement {
        return Err("owner-library answered option index/placement changed".to_string());
    }
    let mut expected_path = canonical_path.clone();
    expected_path.push(*option_index);
    if !canonical_path.is_empty() || !expected_remaining_frames.is_empty() || path != &expected_path
    {
        return Err("owner-library answered option path changed".to_string());
    }
    validate_owner_library_target_binding(state, pending, *object, *owner)
}

fn generic_mana_cost(generic: u8) -> Cost {
    Cost {
        pips: &[],
        generic,
        x_count: 0,
    }
}

/// Revalidates every Ward binding from immutable target provenance through
/// the exact targeter stack id. `allow_absent` is used only before a later
/// Ward trigger begins resolving, where an earlier Ward may already have
/// countered the shared targeter.
#[allow(clippy::too_many_arguments)]
fn validate_counter_unless_pays_generic(
    state: &GameState,
    ward_target: StackTargetContractV4,
    ward_controller: PlayerId,
    targeting_stack_item: StackItemId,
    player: PlayerId,
    generic: u8,
    allow_absent: bool,
    require_public_unambiguous_and_payable: bool,
) -> Result<Option<StackItem>, String> {
    if generic == 0 {
        return Err("zero-mana Ward is outside the certified payment shape".to_string());
    }
    if targeting_stack_item == StackItemId::default() {
        return Err("Ward targets an unstamped stack incarnation".to_string());
    }
    let StackTargetContractV4::Object {
        object: ward_source,
        card_def,
        owner,
        controller: target_controller,
        zone: Zone::Battlefield,
        zone_change_count,
        spell_copy_origin: None,
    } = ward_target
    else {
        return Err("Ward lost its battlefield-permanent target contract".to_string());
    };
    if target_controller != ward_controller {
        return Err("Ward trigger controller changed from the targeting event".to_string());
    }
    let live = state
        .objects
        .try_get(ward_source)
        .ok_or("Ward source object no longer exists")?;
    if live.card_def != card_def
        || live.owner != owner
        || live.zone_change_count < zone_change_count
        || (live.zone_change_count == zone_change_count && live.zone != Zone::Battlefield)
    {
        return Err("Ward source incarnation metadata is inconsistent".to_string());
    }

    let matches = state
        .stack
        .iter()
        .filter(|item| item.v4.stack_item_id == targeting_stack_item)
        .cloned()
        .collect::<Vec<_>>();
    let item = match matches.as_slice() {
        [] if allow_absent => return Ok(None),
        [] => return Err("the Ward-bound stack incarnation is absent".to_string()),
        [item] => item.clone(),
        _ => return Err("the Ward-bound stack incarnation is duplicated".to_string()),
    };
    if item.controller != player || player == ward_controller {
        return Err("the Ward payer/controller binding changed".to_string());
    }
    crate::engine::validated_stack_item_target_spec(&item, state)
        .map_err(|error| format!("Ward targeter has invalid stack provenance: {error}"))?;
    if !item.v4.target_contracts.contains(&ward_target) {
        return Err(
            "the Ward-bound stack item no longer carries its triggering target".to_string(),
        );
    }
    if require_public_unambiguous_and_payable {
        // The Ward trigger names its stack item by id, so several items
        // targeting the same Ward permanent are fine; mtg-kernel's refusal
        // here only protected its old RL observation schema.
        if crate::mana::can_pay(&generic_mana_cost(generic), 0, player, state).is_none() {
            return Err("the staged Ward payment is no longer payable".to_string());
        }
    }
    Ok(Some(item))
}

fn validate_counter_unless_pays_frame(
    state: &GameState,
    pending: &EffectContinuation,
    frame: &EffectFrame,
) -> Result<(), String> {
    let EffectFrame::ResolveCounterUnlessPaysGeneric {
        ward_target,
        targeting_stack_item,
        player,
        generic,
        pay,
        path,
    } = frame
    else {
        return Err("Ward answer guard does not contain its typed frame".to_string());
    };
    if path.as_slice() != [u16::from(*pay)]
        || !pending.frames.ends_with(std::slice::from_ref(frame))
    {
        return Err("Ward answered Boolean path or frame position changed".to_string());
    }
    let StackTargetContractV4::Object {
        object: ward_source,
        ..
    } = *ward_target
    else {
        return Err("Ward answer lost its permanent binding".to_string());
    };
    if pending.resolving_item.source != ward_source
        || pending.ctx.source != ward_source
        || pending.ctx.controller != pending.resolving_item.controller
    {
        return Err("Ward answer no longer matches its resolving trigger".to_string());
    }
    validate_counter_unless_pays_generic(
        state,
        *ward_target,
        pending.resolving_item.controller,
        *targeting_stack_item,
        *player,
        *generic,
        false,
        true,
    )?;
    Ok(())
}

fn validate_counter_target_unless_pays_program(
    state: &GameState,
    pending: &EffectContinuation,
    target_contract: StackTargetContractV4,
    generic: u8,
) -> Result<(), String> {
    if generic == 0 {
        return Err("counter-unless-pay cannot request zero generic mana".to_string());
    }
    let StackTargetContractV4::Object {
        object,
        zone: Zone::Stack,
        ..
    } = target_contract
    else {
        return Err("counter-unless-pay lost its stack-spell target binding".to_string());
    };
    if pending.ctx.targets.as_slice() != [Target::Object(object)]
        || pending.ctx.target_contracts.as_slice() != [target_contract]
    {
        return Err("counter-unless-pay no longer matches its resolving target".to_string());
    }
    let Some(target_spec) = pending.resolving_item.v4.target_spec else {
        return Err("counter-unless-pay resolving spell lost its target specification".to_string());
    };
    if !matches!(
        target_spec,
        crate::card_def::TargetSpec::AnySpellOnStack
            | crate::card_def::TargetSpec::InstantSpellOnStack
            | crate::card_def::TargetSpec::BlueSpellOnStack
            | crate::card_def::TargetSpec::RedSpellOnStack
            | crate::card_def::TargetSpec::ArtifactOrEnchantmentSpellOnStack
            | crate::card_def::TargetSpec::SorcerySpellOnStack
            | crate::card_def::TargetSpec::NoncreatureSpellOnStack
            | crate::card_def::TargetSpec::ArtifactSpellOnStack
    ) {
        return Err("counter-unless-pay resolving spell has a nonspell target filter".to_string());
    }
    let def = crate::card_def::CARD_DEFS
        .get(state.objects.get(pending.resolving_item.source).card_def as usize)
        .ok_or("counter-unless-pay resolving definition is missing")?;
    let program = match pending.resolving_item.mode_chosen {
        0 => (def.spell_effect)(),
        1 => def.mode2.as_ref().map(|mode| (mode.effect)()),
        2 => def.mode3.as_ref().map(|mode| (mode.effect)()),
        _ => None,
    };
    if program
        != Some(EffectOp::CounterTargetUnlessPaysGeneric {
            target: TargetRef::Target(0),
            generic,
        })
    {
        return Err(
            "counter-unless-pay continuation no longer matches its card program".to_string(),
        );
    }
    Ok(())
}

fn validate_counter_target_unless_pays_binding(
    state: &GameState,
    pending: &EffectContinuation,
    target_contract: StackTargetContractV4,
    target_stack_item: StackItemId,
    player: PlayerId,
    generic: u8,
    require_payable: bool,
) -> Result<StackItem, String> {
    validate_counter_target_unless_pays_program(state, pending, target_contract, generic)?;
    if target_stack_item == StackItemId::default() {
        return Err("counter-unless-pay targets an unstamped stack incarnation".to_string());
    }
    let StackTargetContractV4::Object {
        object,
        card_def,
        owner,
        controller,
        zone: Zone::Stack,
        zone_change_count,
        spell_copy_origin,
    } = target_contract
    else {
        unreachable!("program validation already established an object stack target")
    };
    let live = state
        .objects
        .try_get(object)
        .ok_or("counter-unless-pay target object no longer exists")?;
    if live.card_def != card_def
        || live.owner != owner
        || live.controller != controller
        || live.zone != Zone::Stack
        || live.zone_change_count != zone_change_count
        || live.spell_copy_origin != spell_copy_origin
    {
        return Err("counter-unless-pay target incarnation metadata changed".to_string());
    }
    let matches = state
        .stack
        .iter()
        .filter(|item| item.v4.stack_item_id == target_stack_item)
        .cloned()
        .collect::<Vec<_>>();
    let [item] = matches.as_slice() else {
        return Err(format!(
            "counter-unless-pay expected one bound stack item, found {}",
            matches.len()
        ));
    };
    if item.v4.stack_item_id == pending.resolving_item.v4.stack_item_id
        || item.kind != crate::state::StackItemKind::Spell
        || item.source != object
        || item.controller != player
        || controller != player
    {
        return Err("counter-unless-pay target/controller binding changed".to_string());
    }
    crate::engine::validated_stack_item_target_spec(item, state)
        .map_err(|error| format!("counter-unless-pay target provenance is invalid: {error}"))?;
    let source = item
        .v4
        .source_contract
        .ok_or("counter-unless-pay target lost its spell-source contract")?;
    if source.source != object
        || source.card_def != card_def
        || source.owner != owner
        || source.controller != controller
        || source.zone != Zone::Stack
        || source.zone_change_count != zone_change_count
        || source.spell_copy_origin != spell_copy_origin
    {
        return Err("counter-unless-pay target source contract changed".to_string());
    }
    if require_payable
        && crate::mana::can_pay(&generic_mana_cost(generic), 0, player, state).is_none()
    {
        return Err("the staged counter-unless-pay payment is no longer payable".to_string());
    }
    Ok(item.clone())
}

fn validate_counter_target_unless_pays_frame(
    state: &GameState,
    pending: &EffectContinuation,
    frame: &EffectFrame,
) -> Result<(), String> {
    let EffectFrame::ResolveCounterTargetUnlessPaysGeneric {
        target_contract,
        target_stack_item,
        player,
        generic,
        pay,
        path,
    } = frame
    else {
        return Err("counter-unless-pay answer guard does not contain its typed frame".to_string());
    };
    if path.as_slice() != [u16::from(*pay)]
        || !pending.frames.ends_with(std::slice::from_ref(frame))
    {
        return Err("counter-unless-pay answered Boolean path changed".to_string());
    }
    validate_counter_target_unless_pays_binding(
        state,
        pending,
        *target_contract,
        *target_stack_item,
        *player,
        *generic,
        true,
    )?;
    Ok(())
}

fn validated_definition_owned_root_effect(
    state: &GameState,
    pending: &EffectContinuation,
) -> Result<Box<EffectOp>, String> {
    let root = if let Some(inline) = pending.resolving_item.inline_effect.as_ref() {
        Box::new(inline.clone())
    } else if pending.resolving_item.kind == crate::state::StackItemKind::Spell {
        let source = state
            .objects
            .try_get(pending.resolving_item.source)
            .ok_or("answered spell frame lost its source object")?;
        let definition = crate::card_def::CARD_DEFS
            .get(source.card_def as usize)
            .ok_or("answered spell frame lost its source definition")?;
        let effect =
            if pending.resolving_item.v4.cast_method == Some(crate::state::CastMethodV4::Omen) {
                definition.omen.as_ref().map(|omen| (omen.effect)())
            } else {
                match pending.resolving_item.mode_chosen {
                    0 => (definition.spell_effect)(),
                    1 => definition.mode2.as_ref().map(|mode| (mode.effect)()),
                    2 => definition.mode3.as_ref().map(|mode| (mode.effect)()),
                    _ => None,
                }
            }
            .ok_or("answered spell frame lost its definition-owned root program")?;
        Box::new(effect)
    } else {
        return Err("answered effect frame lost its definition-owned root program".to_string());
    };
    if pending.resolving_item.kind == crate::state::StackItemKind::TriggeredAbility {
        let card_def = state
            .objects
            .try_get(pending.resolving_item.source)
            .ok_or("answered trigger frame lost its source object")?
            .card_def;
        if !crate::trigger::triggers_for(card_def)
            .iter()
            .any(|trigger| {
                crate::trigger::materialize_trigger_effect(
                    trigger,
                    pending.resolving_item.source,
                    state,
                ) == *root
            })
        {
            return Err(
                "answered trigger effect no longer matches its card definition".to_string(),
            );
        }
    }
    Ok(root)
}
fn effect_op_at_structural_path<'a>(root: &'a EffectOp, path: &[u16]) -> Option<&'a EffectOp> {
    let Some((&head, tail)) = path.split_first() else {
        return Some(root);
    };
    match root {
        EffectOp::Sequence(ops) => ops
            .get(usize::from(head))
            .and_then(|op| effect_op_at_structural_path(op, tail)),
        EffectOp::Conditional { then, else_, .. } => match head {
            0 => effect_op_at_structural_path(then, tail),
            1 => effect_op_at_structural_path(else_, tail),
            _ => None,
        },
        EffectOp::Choice { options, .. } => options
            .get(usize::from(head))
            .and_then(|op| effect_op_at_structural_path(op, tail)),
        _ => None,
    }
}

fn validate_exile_chosen_graveyard_frame(
    state: &GameState,
    pending: &EffectContinuation,
    frame: &EffectFrame,
) -> Result<(), String> {
    let EffectFrame::ExileChosenGraveyardCard {
        player,
        original_graveyard,
        chosen,
        path,
        canonical_path,
        ..
    } = frame
    else {
        return Err("graveyard-exile answer guard changed frame kind".to_string());
    };
    if !canonical_path.is_empty() || path != canonical_path {
        return Err("graveyard-exile answered path changed".to_string());
    }
    let root = validated_definition_owned_root_effect(state, pending)?;
    let EffectOp::ExileOneFromPlayersGraveyard {
        player: original_player,
    } = root.as_ref()
    else {
        return Err("graveyard-exile answer lost its originating operation".to_string());
    };
    if pending.ctx.resolve_player(*original_player, state) != *player {
        return Err("graveyard-exile answered player changed".to_string());
    }
    validate_bound_graveyard_exact(state, *player, original_graveyard)?;
    if original_graveyard
        .iter()
        .filter(|binding| **binding == *chosen)
        .count()
        != 1
    {
        return Err("graveyard-exile answer is not one bound original card".to_string());
    }
    Ok(())
}

fn validate_exile_chosen_matching_graveyard_frame(
    state: &GameState,
    pending: &EffectContinuation,
    frame: &EffectFrame,
) -> Result<(), String> {
    let EffectFrame::ExileChosenMatchingGraveyardCard {
        player,
        card_type,
        original_graveyard,
        candidates,
        chosen,
        then,
        path,
        canonical_path,
        ..
    } = frame
    else {
        return Err("filtered graveyard-exile answer guard changed frame kind".to_string());
    };
    let mut expected_path = canonical_path.clone();
    expected_path.push(1);
    if path != &expected_path {
        return Err("filtered graveyard-exile answered path changed".to_string());
    }
    let root = validated_definition_owned_root_effect(state, pending)?;
    let Some(EffectOp::MayExileFromPlayersGraveyardMatchingThen {
        player: original_player,
        card_type: original_card_type,
        then: original_then,
    }) = effect_op_at_structural_path(&root, canonical_path)
    else {
        return Err("filtered graveyard-exile answer lost its originating operation".to_string());
    };
    if pending.ctx.resolve_player(*original_player, state) != *player
        || original_card_type != card_type
        || original_then != then
    {
        return Err("filtered graveyard-exile player, filter, or consequence changed".to_string());
    }
    validate_bound_graveyard_exact(state, *player, original_graveyard)?;
    let expected_candidates = matching_graveyard_bindings(state, original_graveyard, *card_type)?;
    if &expected_candidates != candidates || candidates.is_empty() {
        return Err("filtered graveyard-exile candidates changed".to_string());
    }
    if candidates
        .iter()
        .filter(|binding| **binding == *chosen)
        .count()
        != 1
    {
        return Err("filtered graveyard-exile answer is not one bound candidate".to_string());
    }
    validate_resumable_program(then)?;
    Ok(())
}

fn validate_sacrifice_chosen_creature_frame(
    state: &GameState,
    pending: &EffectContinuation,
    frame: &EffectFrame,
) -> Result<(), String> {
    let EffectFrame::SacrificeChosenCreature {
        player,
        filter,
        original_candidates,
        chosen,
        path,
        canonical_path,
        ..
    } = frame
    else {
        return Err("creature-sacrifice answer guard changed frame kind".to_string());
    };
    if path != canonical_path {
        return Err("creature-sacrifice answered path changed".to_string());
    }
    let root = validated_definition_owned_root_effect(state, pending)?;
    let Some(EffectOp::SacrificeCreature {
        player: original_player,
        filter: original_filter,
    }) = effect_op_at_structural_path(&root, canonical_path)
    else {
        return Err("creature-sacrifice answer lost its originating operation".to_string());
    };
    if pending.ctx.resolve_player(*original_player, state) != *player || original_filter != filter {
        return Err("creature-sacrifice player or restriction changed".to_string());
    }
    let expected = creature_sacrifice_bindings(state, *player, *filter)?;
    if &expected != original_candidates || expected.is_empty() {
        return Err("creature-sacrifice candidates changed".to_string());
    }
    if original_candidates
        .iter()
        .filter(|binding| **binding == *chosen)
        .count()
        != 1
    {
        return Err("creature-sacrifice answer is not one bound candidate".to_string());
    }
    Ok(())
}

fn validate_linked_exile_chosen_hand_frame(
    state: &GameState,
    pending: &EffectContinuation,
    frame: &EffectFrame,
) -> Result<(), String> {
    let EffectFrame::LinkedExileChosenHandCard {
        player,
        original_hand,
        chosen,
        source,
        path,
        canonical_path,
        ..
    } = frame
    else {
        return Err("linked-exile answer guard changed frame kind".to_string());
    };
    if !canonical_path.is_empty() || path != canonical_path {
        return Err("linked-exile answered path changed".to_string());
    }
    let root = validated_definition_owned_root_effect(state, pending)?;
    let EffectOp::RevealHandChooseNonlandToLinkedExile {
        player: original_player,
    } = root.as_ref()
    else {
        return Err("linked-exile answer lost its originating operation".to_string());
    };
    if pending.ctx.resolve_player(*original_player, state) != *player
        || pending.resolving_item.v4.ability_source_contract != Some(*source)
        || pending.ctx.ability_source_contract != Some(*source)
        || source.source != pending.ctx.source
        || source.controller != pending.ctx.controller
        || source.zone != Zone::Battlefield
        || source.attached_to.is_some()
        || crate::card_def::CARD_DEFS[source.card_def as usize].name != "Mesmeric Fiend"
    {
        return Err("linked-exile player or source contract changed".to_string());
    }
    validate_bound_hand_exact(state, *player, original_hand)?;
    if original_hand
        .iter()
        .filter(|binding| **binding == *chosen)
        .count()
        != 1
    {
        return Err("linked-exile answer is not one bound original card".to_string());
    }
    let chosen_def =
        &crate::card_def::CARD_DEFS[state.objects.get(chosen.object).card_def as usize];
    if chosen_def.has_type(CardType::Land) {
        return Err("linked-exile answer selected a land card".to_string());
    }
    for observer in [PlayerId::P0, PlayerId::P1] {
        let known = observer == *player
            || state.hand_knowledge[observer.index()][player.index()]
                .iter()
                .any(|entry| {
                    entry.object == chosen.object
                        && entry.zone_change_count == chosen.expected_zone_change_count
                });
        if !known {
            return Err("linked-exile answer lost the public hand reveal".to_string());
        }
    }
    Ok(())
}

fn validate_pay_mana_then_frame(
    state: &GameState,
    pending: &EffectContinuation,
    frame: &EffectFrame,
) -> Result<(), String> {
    let EffectFrame::PayManaThen {
        player,
        colored,
        generic,
        then,
        path,
        canonical_path,
        ..
    } = frame
    else {
        return Err("optional-mana answer guard changed frame kind".to_string());
    };
    let mut expected_path = canonical_path.clone();
    expected_path.push(1);
    if !canonical_path.is_empty() || path != &expected_path {
        return Err("optional-mana answered path changed".to_string());
    }
    let root = validated_definition_owned_root_effect(state, pending)?;
    let EffectOp::MayPayManaThen {
        player: original_player,
        colored: original_colored,
        generic: original_generic,
        then: original_then,
    } = root.as_ref()
    else {
        return Err("optional-mana answer lost its originating operation".to_string());
    };
    if pending.ctx.resolve_player(*original_player, state) != *player
        || original_colored != colored
        || original_generic != generic
        || original_then != then
    {
        return Err("optional-mana answered cost or consequence changed".to_string());
    }
    validate_resumable_program(then)?;
    if !crate::engine::can_pay_effect_mana(*player, colored, *generic, state) {
        return Err("accepted optional mana cost is no longer payable".to_string());
    }
    Ok(())
}

fn effect_op_at_path<'a>(mut op: &'a EffectOp, path: &[u16]) -> Option<&'a EffectOp> {
    for &component in path {
        op = match op {
            EffectOp::Sequence(ops) => ops.get(usize::from(component))?,
            EffectOp::Conditional { then, else_, .. } => match component {
                0 => then,
                1 => else_,
                _ => return None,
            },
            EffectOp::Choice { options, .. } => options.get(usize::from(component))?,
            _ => return None,
        };
    }
    Some(op)
}

fn validate_search_library_to_battlefield_origin(
    state: &GameState,
    pending: &EffectContinuation,
    player: PlayerId,
    filter: LibraryCardFilter,
    canonical_path: &[u16],
) -> Result<(), String> {
    let root = validated_definition_owned_root_effect(state, pending)?;
    match effect_op_at_path(root.as_ref(), canonical_path) {
        Some(EffectOp::SearchLibraryToBattlefieldTapped {
            player: original_player,
            filter: original_filter,
        }) if *original_filter == filter
            && pending.ctx.resolve_player(*original_player, state) == player =>
        {
            Ok(())
        }
        Some(EffectOp::DestroyTargetLandThenMaySearchBasicTapped {
            object: ObjectRef::Target(0),
        }) if filter == LibraryCardFilter::BasicLand
            && pending.resolving_item.v4.target_spec == Some(crate::card_def::TargetSpec::Land)
            && pending.ctx.targets.len() == 1
            && pending.ctx.target_contracts.len() == 1
            && matches!(
                pending.ctx.target_contracts[0],
                StackTargetContractV4::Object {
                    controller,
                    zone: Zone::Battlefield,
                    ..
                } if controller == player
            ) =>
        {
            Ok(())
        }
        _ => Err("battlefield-search lost its definition-owned origin".to_string()),
    }
}

fn validate_duress_origin(
    state: &GameState,
    pending: &EffectContinuation,
    player: PlayerId,
    canonical_path: &[u16],
) -> Result<(), String> {
    let root = validated_definition_owned_root_effect(state, pending)?;
    match effect_op_at_path(root.as_ref(), canonical_path) {
        Some(EffectOp::RevealTargetHandChooseNoncreatureNonlandDiscard {
            player: original_player,
        }) if pending.ctx.resolve_player(*original_player, state) == player => Ok(()),
        _ => Err("Duress continuation lost its definition-owned origin".to_string()),
    }
}

fn validate_begin_search_library_to_battlefield_frame(
    state: &GameState,
    pending: &EffectContinuation,
    frame: &EffectFrame,
) -> Result<(), String> {
    let EffectFrame::BeginSearchLibraryToBattlefieldTapped {
        player,
        filter,
        path,
        canonical_path,
        expected_remaining_frames: _,
    } = frame
    else {
        return Err("battlefield-search answer guard changed frame kind".to_string());
    };
    let mut expected_path = canonical_path.clone();
    expected_path.push(1);
    if path != &expected_path || *filter != LibraryCardFilter::BasicLand {
        return Err("battlefield-search answered path or filter changed".to_string());
    }
    validate_search_library_to_battlefield_origin(state, pending, *player, *filter, canonical_path)
}

pub(crate) fn validate_initiative_trigger_binding(
    state: &GameState,
    binding: InitiativeTriggerBindingV1,
) -> Result<(), String> {
    let history_index = usize::try_from(binding.history_index)
        .map_err(|_| "Initiative history index exceeds usize".to_string())?;
    if state.engine.event_history.get(history_index)
        != Some(&event::CommittedEvent::InitiativeTrigger { binding })
    {
        return Err("Initiative trigger lost its exact committed marker".to_string());
    }
    let source = state
        .objects
        .try_get(binding.source.source)
        .ok_or("Initiative trigger source object is missing")?;
    let definition = crate::card_def::CARD_DEFS
        .get(binding.source.card_def as usize)
        .ok_or("Initiative trigger source definition is missing")?;
    if binding.source.controller != binding.player
        || binding.source.zone != Zone::Battlefield
        || binding.source.attached_to.is_some()
        || definition.name != "Avenging Hunter"
        || !definition.is_executable()
        || source.card_def != binding.source.card_def
        || source.owner != binding.source.owner
        || source.zone_change_count < binding.source.zone_change_count
        || (source.zone_change_count == binding.source.zone_change_count
            && (source.zone != binding.source.zone
                || source.v4.attached_to != binding.source.attached_to))
    {
        return Err("Initiative trigger source provenance is malformed".to_string());
    }
    Ok(())
}

fn validate_initiative_continuation_root(
    state: &GameState,
    pending: &EffectContinuation,
    binding: InitiativeTriggerBindingV1,
) -> Result<(), String> {
    validate_initiative_trigger_binding(state, binding)?;
    if pending.resolving_item.inline_effect.as_ref()
        != Some(&EffectOp::ResolveInitiativeTrigger { binding })
        || pending.ctx.source != binding.source.source
        || pending.ctx.controller != binding.player
        || pending.ctx.ability_source_contract != Some(binding.source)
        || pending.resolving_item.v4.ability_source_contract != Some(binding.source)
    {
        return Err("Initiative continuation no longer matches its bound trigger".to_string());
    }
    Ok(())
}

fn undercity_next_rooms(
    state: &GameState,
    player: PlayerId,
) -> Result<(Option<UndercityRoomV1>, Vec<UndercityRoomV1>), String> {
    let dungeon = &state.players[player.index()].dungeon;
    match (dungeon.dungeon_id, dungeon.room_id) {
        (None, None) => Ok((None, vec![UndercityRoomV1::SecretEntrance])),
        (Some(crate::state::UNDERCITY_DUNGEON_ID_V1), Some(room_id)) => {
            let room = UndercityRoomV1::from_stable_id(room_id)
                .ok_or("Undercity state carries an unknown room id")?;
            let next = match room {
                UndercityRoomV1::SecretEntrance => {
                    vec![UndercityRoomV1::Forge, UndercityRoomV1::LostWell]
                }
                UndercityRoomV1::Forge => {
                    vec![UndercityRoomV1::Trap, UndercityRoomV1::Arena]
                }
                UndercityRoomV1::LostWell => {
                    vec![UndercityRoomV1::Arena, UndercityRoomV1::Stash]
                }
                UndercityRoomV1::Trap => vec![UndercityRoomV1::Archives],
                UndercityRoomV1::Arena => {
                    vec![UndercityRoomV1::Archives, UndercityRoomV1::Catacombs]
                }
                UndercityRoomV1::Stash => vec![UndercityRoomV1::Catacombs],
                UndercityRoomV1::Archives | UndercityRoomV1::Catacombs => {
                    vec![UndercityRoomV1::ThroneOfTheDeadThree]
                }
                UndercityRoomV1::ThroneOfTheDeadThree => {
                    return Err("completed Undercity remains active".to_string())
                }
            };
            Ok((Some(room), next))
        }
        _ => Err("dungeon id and room id are not a coherent Undercity state".to_string()),
    }
}

fn validate_undercity_route_frame(
    state: &GameState,
    pending: &EffectContinuation,
    frame: &EffectFrame,
) -> Result<(), String> {
    let EffectFrame::EnterUndercityRoom {
        binding,
        from_room,
        room,
        path,
        expected_remaining_frames,
    } = frame
    else {
        return Err("Undercity route guard changed frame kind".to_string());
    };
    validate_initiative_continuation_root(state, pending, *binding)?;
    if !matches!(
        binding.kind,
        InitiativeTriggerKindV1::VentureAfterTaking | InitiativeTriggerKindV1::VentureAtUpkeep
    ) {
        return Err("Undercity route is not owned by a venture trigger".to_string());
    }
    let (expected_from, legal_rooms) = undercity_next_rooms(state, binding.player)?;
    let Some(option_index) = legal_rooms.iter().position(|candidate| candidate == room) else {
        return Err("Undercity route selected a nonadjacent room".to_string());
    };
    let expected_path =
        vec![u16::try_from(option_index)
            .map_err(|_| "Undercity route index exceeds u16".to_string())?];
    if *from_room != expected_from
        || !expected_remaining_frames.is_empty()
        || path != &expected_path
    {
        return Err("Undercity route path or origin changed".to_string());
    }
    Ok(())
}

fn validate_undercity_throne_metadata(
    state: &GameState,
    pending: &EffectContinuation,
    binding: InitiativeTriggerBindingV1,
    original_library: &[EffectObjectBinding],
    revealed_prefix: &[EffectObjectBinding],
    candidates: &[EffectObjectBinding],
    chosen: Option<EffectObjectBinding>,
) -> Result<(), String> {
    validate_initiative_continuation_root(state, pending, binding)?;
    if binding.kind != InitiativeTriggerKindV1::UndercityRoom(UndercityRoomV1::ThroneOfTheDeadThree)
    {
        return Err("Throne continuation is not owned by the Throne room".to_string());
    }
    let current = bind_library_exact(state, binding.player);
    if current != original_library {
        return Err("Throne library order or incarnation changed".to_string());
    }
    let prefix_len = original_library.len().min(10);
    if revealed_prefix != &original_library[..prefix_len] {
        return Err("Throne revealed prefix changed".to_string());
    }
    for observer in [PlayerId::P0, PlayerId::P1] {
        for (position, revealed) in revealed_prefix.iter().enumerate() {
            if !state
                .known_library_cards(observer, binding.player)
                .iter()
                .any(|entry| {
                    entry.position as usize == position
                        && entry.object == revealed.object
                        && entry.zone_change_count == revealed.expected_zone_change_count
                })
            {
                return Err("Throne continuation lost its public reveal".to_string());
            }
        }
    }
    let expected_candidates = revealed_prefix
        .iter()
        .copied()
        .filter(|candidate| {
            crate::card_def::CARD_DEFS[state.objects.get(candidate.object).card_def as usize]
                .has_type(CardType::Creature)
        })
        .collect::<Vec<_>>();
    if candidates != expected_candidates {
        return Err("Throne creature candidates changed".to_string());
    }
    if chosen.is_some_and(|selected| !candidates.contains(&selected)) {
        return Err("Throne selected a noncandidate".to_string());
    }
    Ok(())
}

fn validate_undercity_throne_frame(
    state: &GameState,
    pending: &EffectContinuation,
    frame: &EffectFrame,
) -> Result<(), String> {
    let EffectFrame::ResolveUndercityThrone {
        binding,
        original_library,
        revealed_prefix,
        candidates,
        chosen,
        path,
        canonical_path,
        expected_remaining_frames,
    } = frame
    else {
        return Err("Throne answer guard changed frame kind".to_string());
    };
    if !canonical_path.is_empty()
        || path != canonical_path
        || !expected_remaining_frames.is_empty()
        || candidates.len() < 2
    {
        return Err("Throne answered frame has noncanonical progress".to_string());
    }
    validate_undercity_throne_metadata(
        state,
        pending,
        *binding,
        original_library,
        revealed_prefix,
        candidates,
        Some(*chosen),
    )
}

fn enter_undercity_room(
    state: &mut GameState,
    binding: InitiativeTriggerBindingV1,
    from_room: Option<UndercityRoomV1>,
    room: UndercityRoomV1,
) -> Result<(), String> {
    let (expected_from, legal_rooms) = undercity_next_rooms(state, binding.player)?;
    if from_room != expected_from || !legal_rooms.contains(&room) {
        return Err("Undercity entry no longer follows the exact room graph".to_string());
    }
    let dungeon = &mut state.players[binding.player.index()].dungeon;
    if room == UndercityRoomV1::ThroneOfTheDeadThree {
        match dungeon
            .completed_dungeons
            .binary_search(&crate::state::UNDERCITY_DUNGEON_ID_V1)
        {
            Ok(_) => {}
            Err(index) => dungeon
                .completed_dungeons
                .insert(index, crate::state::UNDERCITY_DUNGEON_ID_V1),
        }
        dungeon.dungeon_id = None;
        dungeon.room_id = None;
    } else {
        dungeon.dungeon_id = Some(crate::state::UNDERCITY_DUNGEON_ID_V1);
        dungeon.room_id = Some(room.stable_id());
    }
    event::log_initiative_trigger(
        state,
        binding.player,
        binding.source,
        InitiativeTriggerKindV1::UndercityRoom(room),
    )?;
    Ok(())
}

fn until_players_next_turn_expiry(state: &GameState, player: PlayerId) -> Result<u32, String> {
    if state.active_player == player {
        state
            .turn
            .checked_add(1)
            .ok_or("turn counter overflow while scheduling next-turn expiry".to_string())
    } else {
        Ok(state.turn)
    }
}

fn apply_undercity_throne_result(
    state: &mut GameState,
    binding: InitiativeTriggerBindingV1,
    chosen: Option<EffectObjectBinding>,
) -> Result<(), String> {
    let shuffle_token = state
        .preflight_library_shuffle(binding.player)
        .map_err(|error| error.to_string())?;
    if let Some(chosen) = chosen {
        let old_generation = chosen.expected_zone_change_count;
        event::propose_and_commit(
            state,
            event::ProposedEvent::zone_change(chosen.object, Zone::Battlefield),
        );
        let permanent = state.objects.get(chosen.object);
        if permanent.zone != Zone::Battlefield
            || permanent.owner != binding.player
            || permanent.zone_change_count != old_generation.saturating_add(1)
        {
            return Err("Throne chosen creature did not enter as its next incarnation".to_string());
        }
        let generation = permanent.zone_change_count;
        let counters = &mut state.objects.get_mut(chosen.object).counters.plus1_plus1;
        *counters = counters
            .checked_add(3)
            .ok_or("Throne +1/+1 counter overflow")?;
        let expires_at_turn = until_players_next_turn_expiry(state, binding.player)?;
        state
            .engine
            .until_next_turn_keywords
            .push(crate::engine::UntilNextTurnKeywordEffectV1 {
                object_id: chosen.object,
                object_zone_change_count: generation,
                holder: binding.player,
                expires_at_turn,
                keywords: Keywords::HEXPROOF,
            });
        state.engine.until_next_turn_keywords.sort_by_key(|effect| {
            (
                effect.holder,
                effect.expires_at_turn,
                effect.object_id,
                effect.object_zone_change_count,
                effect.keywords.0,
            )
        });
    }
    state
        .commit_library_shuffle(binding.player, shuffle_token)
        .map_err(|error| error.to_string())
}

fn take_initiative(
    state: &mut GameState,
    player: PlayerId,
    mut source: AbilitySourceContractV4,
) -> Result<(), String> {
    source.controller = player;
    let live = state
        .objects
        .try_get(source.source)
        .ok_or("Initiative source object is missing")?;
    if source.zone != Zone::Battlefield
        || source.attached_to.is_some()
        || live.card_def != source.card_def
        || live.owner != source.owner
        || live.zone_change_count < source.zone_change_count
        || (live.zone_change_count == source.zone_change_count && live.zone != source.zone)
        || crate::card_def::CARD_DEFS
            .get(source.card_def as usize)
            .is_none_or(|definition| definition.name != "Avenging Hunter")
    {
        return Err("Initiative source contract is malformed".to_string());
    }
    state.initiative = Some(player);
    state.engine.initiative_source = Some(source);
    event::log_initiative_trigger(
        state,
        player,
        source,
        InitiativeTriggerKindV1::VentureAfterTaking,
    )?;
    Ok(())
}

fn stage_undercity_venture(
    state: &GameState,
    continuation: &mut EffectContinuation,
    binding: InitiativeTriggerBindingV1,
    path: Vec<u16>,
) -> Result<bool, String> {
    if !path.is_empty() || !continuation.frames.is_empty() {
        return Err("venture trigger is not a root-only effect".to_string());
    }
    let (from_room, legal_rooms) = undercity_next_rooms(state, binding.player)?;
    match legal_rooms.as_slice() {
        [] => Err("Undercity route has no next room".to_string()),
        [room] => {
            continuation.frames.push(EffectFrame::EnterUndercityRoom {
                binding,
                from_room,
                room: *room,
                path: vec![0],
                expected_remaining_frames: Vec::new(),
            });
            Ok(false)
        }
        _ => {
            let options = legal_rooms
                .iter()
                .copied()
                .map(|room| EffectOp::EnterUndercityRoom {
                    binding,
                    from_room,
                    room,
                })
                .collect();
            continuation.choice = Some(PendingEffectChoice::ChooseOption {
                player: binding.player,
                path: path.clone(),
                options,
                purpose: EffectOptionChoicePurpose::UndercityRoute {
                    binding,
                    from_room,
                    legal_rooms,
                    canonical_path: path,
                    expected_remaining_frames: Vec::new(),
                },
            });
            Ok(true)
        }
    }
}

fn undercity_room_program(
    binding: InitiativeTriggerBindingV1,
    room: UndercityRoomV1,
) -> Result<EffectOp, String> {
    Ok(match room {
        UndercityRoomV1::SecretEntrance => EffectOp::SearchLibraryToHand {
            player: PlayerRef::Controller,
            filter: LibraryCardFilter::BasicLand,
        },
        UndercityRoomV1::Forge => EffectOp::AddPlusOnePlusOneCounters {
            object: ObjectRef::Target(0),
            count: 2,
        },
        UndercityRoomV1::LostWell => EffectOp::Scry {
            player: PlayerRef::Controller,
            count: 2,
        },
        UndercityRoomV1::Trap => EffectOp::LoseLife {
            player: PlayerRef::Target(0),
            amount: 5,
        },
        UndercityRoomV1::Arena => EffectOp::GoadTargetUntilSourcesNextTurn {
            object: ObjectRef::Target(0),
        },
        UndercityRoomV1::Stash => EffectOp::CreateToken {
            token_def: crate::card_def::card_id_by_name("Treasure Token")
                .ok_or("Treasure Token definition is missing")?,
            controller: PlayerRef::Controller,
        },
        UndercityRoomV1::Archives => EffectOp::DrawCards {
            player: PlayerRef::Controller,
            count: 1,
        },
        UndercityRoomV1::Catacombs => EffectOp::CreateToken {
            token_def: crate::card_def::card_id_by_name("Skeleton Token")
                .ok_or("Skeleton Token definition is missing")?,
            controller: PlayerRef::Controller,
        },
        UndercityRoomV1::ThroneOfTheDeadThree => EffectOp::ResolveUndercityThrone { binding },
    })
}

fn stage_or_apply_undercity_throne(
    state: &mut GameState,
    continuation: &mut EffectContinuation,
    binding: InitiativeTriggerBindingV1,
    path: Vec<u16>,
) -> Result<bool, String> {
    if !path.is_empty() || !continuation.frames.is_empty() {
        return Err("Throne room effect is not a root-only operation".to_string());
    }
    validate_initiative_continuation_root(state, continuation, binding)?;
    let original_library = bind_library_exact(state, binding.player);
    let revealed_prefix = original_library[..original_library.len().min(10)].to_vec();
    for observer in [PlayerId::P0, PlayerId::P1] {
        state.reveal_library_top(observer, binding.player, revealed_prefix.len());
    }
    let candidates = revealed_prefix
        .iter()
        .copied()
        .filter(|candidate| {
            crate::card_def::CARD_DEFS[state.objects.get(candidate.object).card_def as usize]
                .has_type(CardType::Creature)
        })
        .collect::<Vec<_>>();
    validate_undercity_throne_metadata(
        state,
        continuation,
        binding,
        &original_library,
        &revealed_prefix,
        &candidates,
        None,
    )?;
    match candidates.as_slice() {
        [] => {
            apply_undercity_throne_result(state, binding, None)?;
            Ok(false)
        }
        [chosen] => {
            apply_undercity_throne_result(state, binding, Some(*chosen))?;
            Ok(false)
        }
        _ => {
            let token = state
                .preflight_library_shuffle(binding.player)
                .map_err(|error| error.to_string())?;
            drop(token);
            let legal = candidates
                .iter()
                .copied()
                .map(|candidate| EffectTargetCandidate {
                    target: Target::Object(candidate.object),
                    expected_object: Some(candidate),
                })
                .collect();
            continuation.choice = Some(PendingEffectChoice::SelectTargets {
                player: binding.player,
                path: path.clone(),
                selected: Vec::new(),
                legal,
                min_targets: 1,
                max_targets: 1,
                ordered: true,
                purpose: EffectTargetSelectionPurpose::UndercityThroneCreature {
                    binding,
                    original_library,
                    revealed_prefix,
                    candidates,
                    canonical_path: path,
                },
            });
            Ok(true)
        }
    }
}

fn validate_answered_choice_guard(
    state: &GameState,
    pending: &EffectContinuation,
) -> Result<(), String> {
    for frame in &pending.frames {
        if let EffectFrame::Program { op, .. } = frame {
            validate_resumable_program(op)?;
        }
    }
    match &pending.answered_choice_guard {
        None => {
            if pending.frames.iter().any(|frame| {
                matches!(
                    frame,
                    EffectFrame::OwnerLibraryPlacement { .. }
                        | EffectFrame::ResolveCounterUnlessPaysGeneric { .. }
                        | EffectFrame::ResolveCounterTargetUnlessPaysGeneric { .. }
                        | EffectFrame::ExileChosenGraveyardCard { .. }
                        | EffectFrame::ExileChosenMatchingGraveyardCard { .. }
                        | EffectFrame::SacrificeChosenCreature { .. }
                        | EffectFrame::PayManaThen { .. }
                        | EffectFrame::LinkedExileChosenHandCard { .. }
                        | EffectFrame::BeginSearchLibraryToBattlefieldTapped { .. }
                        | EffectFrame::EnterUndercityRoom { .. }
                        | EffectFrame::ResolveUndercityThrone { .. }
                )
            }) {
                return Err("typed answered-choice frame has no matching guard".to_string());
            }
        }
        Some(EffectAnsweredChoiceGuard::OwnerLibrarySecondOrBottom { frame }) => {
            if pending.choice.is_some() {
                return Err("answered owner-library guard still carries a live choice".to_string());
            }
            let EffectFrame::OwnerLibraryPlacement {
                expected_remaining_frames,
                ..
            } = frame.as_ref()
            else {
                return Err("owner-library answer guard changed frame kind".to_string());
            };
            let mut expected = expected_remaining_frames.clone();
            expected.push((**frame).clone());
            if pending.frames != expected {
                return Err("owner-library answered continuation frame stack changed".to_string());
            }
            validate_owner_library_placement_frame(state, pending, frame)?;
        }
        Some(EffectAnsweredChoiceGuard::CounterUnlessPaysGeneric { frame }) => {
            if pending.choice.is_some() {
                return Err("answered Ward guard still carries a live choice".to_string());
            }
            if pending.frames.as_slice() != [frame.as_ref().clone()] {
                return Err("Ward answered continuation frame stack changed".to_string());
            }
            validate_counter_unless_pays_frame(state, pending, frame)?;
        }
        Some(EffectAnsweredChoiceGuard::CounterTargetUnlessPaysGeneric { frame }) => {
            if pending.choice.is_some() {
                return Err(
                    "answered counter-unless-pay guard still carries a live choice".to_string(),
                );
            }
            if pending.frames.as_slice() != [frame.as_ref().clone()] {
                return Err(
                    "counter-unless-pay answered continuation frame stack changed".to_string(),
                );
            }
            validate_counter_target_unless_pays_frame(state, pending, frame)?;
        }
        Some(EffectAnsweredChoiceGuard::ExileOneFromGraveyard { frame }) => {
            if pending.choice.is_some() {
                return Err(
                    "answered graveyard-exile guard still carries a live choice".to_string()
                );
            }
            let EffectFrame::ExileChosenGraveyardCard {
                expected_remaining_frames,
                ..
            } = frame.as_ref()
            else {
                return Err("graveyard-exile answer guard changed frame kind".to_string());
            };
            let mut expected = expected_remaining_frames.clone();
            expected.push((**frame).clone());
            if pending.frames != expected {
                return Err("graveyard-exile answered continuation frame stack changed".to_string());
            }
            validate_exile_chosen_graveyard_frame(state, pending, frame)?;
        }
        Some(EffectAnsweredChoiceGuard::ExileOneMatchingFromGraveyard { frame }) => {
            if pending.choice.is_some() {
                return Err(
                    "answered filtered graveyard-exile guard still carries a live choice"
                        .to_string(),
                );
            }
            let EffectFrame::ExileChosenMatchingGraveyardCard {
                expected_remaining_frames,
                ..
            } = frame.as_ref()
            else {
                return Err("filtered graveyard-exile answer guard changed frame kind".to_string());
            };
            let mut expected = expected_remaining_frames.clone();
            expected.push((**frame).clone());
            if pending.frames != expected {
                return Err(
                    "filtered graveyard-exile answered continuation frame stack changed"
                        .to_string(),
                );
            }
            validate_exile_chosen_matching_graveyard_frame(state, pending, frame)?;
        }
        Some(EffectAnsweredChoiceGuard::SacrificeCreature { frame }) => {
            if pending.choice.is_some() {
                return Err(
                    "answered creature-sacrifice guard still carries a live choice".to_string(),
                );
            }
            let EffectFrame::SacrificeChosenCreature {
                expected_remaining_frames,
                ..
            } = frame.as_ref()
            else {
                return Err("creature-sacrifice answer guard changed frame kind".to_string());
            };
            let mut expected = expected_remaining_frames.clone();
            expected.push((**frame).clone());
            if pending.frames != expected {
                return Err(
                    "creature-sacrifice answered continuation frame stack changed".to_string(),
                );
            }
            validate_sacrifice_chosen_creature_frame(state, pending, frame)?;
        }
        Some(EffectAnsweredChoiceGuard::PayManaThen { frame }) => {
            if pending.choice.is_some() {
                return Err("answered optional-mana guard still carries a live choice".to_string());
            }
            let EffectFrame::PayManaThen {
                expected_remaining_frames,
                ..
            } = frame.as_ref()
            else {
                return Err("optional-mana answer guard changed frame kind".to_string());
            };
            let mut expected = expected_remaining_frames.clone();
            expected.push((**frame).clone());
            if pending.frames != expected {
                return Err("optional-mana answered continuation frame stack changed".to_string());
            }
            validate_pay_mana_then_frame(state, pending, frame)?;
        }
        Some(EffectAnsweredChoiceGuard::LinkedExileFromRevealedHand { frame }) => {
            if pending.choice.is_some() {
                return Err("answered linked-exile guard still carries a live choice".to_string());
            }
            let EffectFrame::LinkedExileChosenHandCard {
                expected_remaining_frames,
                ..
            } = frame.as_ref()
            else {
                return Err("linked-exile answer guard changed frame kind".to_string());
            };
            let mut expected = expected_remaining_frames.clone();
            expected.push((**frame).clone());
            if pending.frames != expected {
                return Err("linked-exile answered continuation frame stack changed".to_string());
            }
            validate_linked_exile_chosen_hand_frame(state, pending, frame)?;
        }
        Some(EffectAnsweredChoiceGuard::SearchLibraryToBattlefieldTapped { frame }) => {
            if pending.choice.is_some() {
                return Err(
                    "answered battlefield-search guard still carries a live choice".to_string(),
                );
            }
            let EffectFrame::BeginSearchLibraryToBattlefieldTapped {
                expected_remaining_frames,
                ..
            } = frame.as_ref()
            else {
                return Err("battlefield-search answer guard changed frame kind".to_string());
            };
            let mut expected = expected_remaining_frames.clone();
            expected.push((**frame).clone());
            if pending.frames != expected {
                return Err(
                    "battlefield-search answered continuation frame stack changed".to_string(),
                );
            }
            validate_begin_search_library_to_battlefield_frame(state, pending, frame)?;
        }
        Some(EffectAnsweredChoiceGuard::UndercityRoute { frame }) => {
            if pending.choice.is_some() {
                return Err("answered Undercity route still carries a live choice".to_string());
            }
            let EffectFrame::EnterUndercityRoom {
                expected_remaining_frames,
                ..
            } = frame.as_ref()
            else {
                return Err("Undercity route guard changed frame kind".to_string());
            };
            let mut expected = expected_remaining_frames.clone();
            expected.push((**frame).clone());
            if pending.frames != expected {
                return Err("Undercity route continuation frame stack changed".to_string());
            }
            validate_undercity_route_frame(state, pending, frame)?;
        }
        Some(EffectAnsweredChoiceGuard::UndercityThrone { frame }) => {
            if pending.choice.is_some() {
                return Err("answered Throne choice still carries a live choice".to_string());
            }
            let EffectFrame::ResolveUndercityThrone {
                expected_remaining_frames,
                ..
            } = frame.as_ref()
            else {
                return Err("Throne answer guard changed frame kind".to_string());
            };
            let mut expected = expected_remaining_frames.clone();
            expected.push((**frame).clone());
            if pending.frames != expected {
                return Err("Throne answered continuation frame stack changed".to_string());
            }
            validate_undercity_throne_frame(state, pending, frame)?;
        }
    }
    Ok(())
}

pub fn validate_pending_effect_choice(state: &GameState) -> Result<(), String> {
    let Some(pending) = state.engine.pending_effect.as_ref() else {
        return Ok(());
    };
    if pending.ctx.stack_item_id != Some(pending.resolving_item.v4.stack_item_id)
        || pending.ctx.source != pending.resolving_item.source
        || pending.ctx.controller != pending.resolving_item.controller
        || pending.ctx.targets != pending.resolving_item.targets
        || pending.ctx.target_contracts != pending.resolving_item.v4.target_contracts
        || pending.ctx.discarded != pending.resolving_item.discarded
        || pending.ctx.paid_cost_refs != pending.resolving_item.v4.paid_cost_refs
        || pending.ctx.hidden_ability_source != pending.resolving_item.v4.hidden_ability_source
        || pending.ctx.kicked != pending.resolving_item.kicked
        || pending.ctx.ability_source_contract != pending.resolving_item.v4.ability_source_contract
        || pending.ctx.optional_additional_cost_paid
            != pending.resolving_item.v4.optional_additional_cost_paid
    {
        return Err(
            "effect continuation context no longer mirrors its resolving stack item".to_string(),
        );
    }
    if state.stack.last() != Some(&pending.resolving_item) {
        return Err(
            "effect continuation resolving item no longer exactly matches the public stack top"
                .to_string(),
        );
    }
    crate::engine::validated_stack_item_target_spec(&pending.resolving_item, state)
        .map_err(|error| format!("effect continuation has invalid stack provenance: {error}"))?;
    validate_answered_choice_guard(state, pending)?;
    let Some(choice) = pending.choice.as_ref() else {
        return Ok(());
    };
    match choice {
        PendingEffectChoice::SelectTargets {
            player: chooser,
            path,
            selected,
            legal,
            min_targets,
            max_targets,
            ordered,
            purpose,
        } => {
            for candidate in selected.iter().chain(legal) {
                validate_effect_target_candidate(state, candidate)?;
            }
            match purpose {
                EffectTargetSelectionPurpose::OrderMilledIntoGraveyard => {
                    let bindings = selected
                        .iter()
                        .chain(legal)
                        .map(|candidate| {
                            candidate.expected_object.ok_or_else(|| {
                                "milled-card ordering target lacks an object-incarnation binding"
                                    .to_string()
                            })
                        })
                        .collect::<Result<Vec<_>, String>>()?;
                    validate_bound_library_prefix(state, &bindings)?;
                    let library_owner = bindings
                        .first()
                        .map(|binding| state.objects.get(binding.object).owner)
                        .ok_or_else(|| {
                            "milled-card ordering choice has no bound library objects".to_string()
                        })?;
                    if *chooser != library_owner {
                        return Err(
                            "milled-card ordering player does not own the selected library"
                                .to_string(),
                        );
                    }
                }
                EffectTargetSelectionPurpose::OrderLookedLibraryTop {
                    player: library_player,
                    original_prefix,
                } => {
                    if chooser != library_player {
                        return Err(
                            "library-order choice player does not own the selected library"
                                .to_string(),
                        );
                    }
                    validate_bound_library_prefix_exact(state, *library_player, original_prefix)?;
                    let mut partition = selected
                        .iter()
                        .chain(legal)
                        .map(|candidate| {
                            candidate.expected_object.ok_or_else(|| {
                                "library-order target lacks an object-incarnation binding"
                                    .to_string()
                            })
                        })
                        .collect::<Result<Vec<_>, String>>()?;
                    let mut original = original_prefix.clone();
                    partition.sort_by_key(|binding| binding.object);
                    original.sort_by_key(|binding| binding.object);
                    if partition != original {
                        return Err("library-order candidates do not partition the bound prefix"
                            .to_string());
                    }
                }
                EffectTargetSelectionPurpose::PutHandCardOnLibraryTop {
                    player: hand_player,
                    original_hand,
                    total,
                    remaining,
                    prompt_index,
                    continuation_path,
                    canonical_path,
                } => {
                    if chooser != hand_player {
                        return Err(
                            "hand-to-library choice player does not own the selected hand"
                                .to_string(),
                        );
                    }
                    validate_hand_to_library_progress(
                        *total,
                        *remaining,
                        *prompt_index,
                        continuation_path,
                        canonical_path,
                    )?;
                    if *remaining == 0 || original_hand.len() < 2 {
                        return Err(
                            "hand-to-library policy prompt has no genuine choice".to_string()
                        );
                    }
                    if *min_targets != 1 || *max_targets != 1 || !*ordered || !selected.is_empty() {
                        return Err(
                            "hand-to-library prompt is not an independent exact-one ordering choice"
                                .to_string(),
                        );
                    }
                    let mut expected_path = canonical_path.clone();
                    expected_path.push(*prompt_index);
                    if path != &expected_path {
                        return Err("hand-to-library prompt structural path changed".to_string());
                    }
                    validate_bound_hand_exact(state, *hand_player, original_hand)?;
                    let mut partition = selected
                        .iter()
                        .chain(legal)
                        .map(|candidate| {
                            candidate.expected_object.ok_or_else(|| {
                                "hand-to-library target lacks an object-incarnation binding"
                                    .to_string()
                            })
                        })
                        .collect::<Result<Vec<_>, String>>()?;
                    let mut original = original_hand.clone();
                    partition.sort_by_key(|binding| binding.object);
                    original.sort_by_key(|binding| binding.object);
                    if partition != original {
                        return Err("hand-to-library candidates do not partition the bound hand"
                            .to_string());
                    }
                }
                EffectTargetSelectionPurpose::ScryLibrary {
                    player: library_player,
                    requested_count,
                    original_library_len,
                    original_prefix,
                    stage,
                    stage_fingerprint,
                    canonical_path,
                } => {
                    if chooser != library_player {
                        return Err(
                            "scry choice player does not own the selected library".to_string()
                        );
                    }
                    validate_scry_live_metadata(
                        state,
                        *library_player,
                        *requested_count,
                        *original_library_len,
                        original_prefix,
                    )?;
                    if *stage_fingerprint != scry_stage_fingerprint(stage) {
                        return Err("scry prompt stage fingerprint changed".to_string());
                    }
                    let mut expected_path = canonical_path.clone();
                    expected_path.push(scry_stage_tag(stage));
                    if path != &expected_path {
                        return Err("scry prompt structural path changed".to_string());
                    }
                    let candidates = selected
                        .iter()
                        .chain(legal)
                        .map(|candidate| {
                            candidate.expected_object.ok_or_else(|| {
                                "scry target lacks an object-incarnation binding".to_string()
                            })
                        })
                        .collect::<Result<Vec<_>, String>>()?;
                    match stage {
                        ScrySelectionStage::ChooseBottomSubset => {
                            let count = u16::try_from(original_prefix.len())
                                .map_err(|_| "scry prefix exceeds u16".to_string())?;
                            if *min_targets != 0 || *max_targets != count || *ordered {
                                return Err("scry bottom-subset prompt has a noncanonical shape"
                                    .to_string());
                            }
                            validate_exact_binding_permutation(
                                original_prefix,
                                &candidates,
                                "scry bottom-subset candidates",
                            )?;
                        }
                        ScrySelectionStage::OrderBottom { bottom_subset } => {
                            validate_canonical_scry_subset(original_prefix, bottom_subset)?;
                            if bottom_subset.len() < 2 {
                                return Err(
                                    "scry bottom-order prompt has no genuine ordering choice"
                                        .to_string(),
                                );
                            }
                            let count = u16::try_from(bottom_subset.len())
                                .map_err(|_| "scry bottom group exceeds u16".to_string())?;
                            if *min_targets != count || *max_targets != count || !*ordered {
                                return Err(
                                    "scry bottom-order prompt has a noncanonical shape".to_string()
                                );
                            }
                            validate_exact_binding_permutation(
                                bottom_subset,
                                &candidates,
                                "scry bottom-order candidates",
                            )?;
                        }
                        ScrySelectionStage::OrderRetainedTop {
                            bottom_subset,
                            ordered_bottom,
                        } => {
                            validate_canonical_scry_subset(original_prefix, bottom_subset)?;
                            validate_exact_binding_permutation(
                                bottom_subset,
                                ordered_bottom,
                                "scry ordered bottom",
                            )?;
                            let retained = scry_retained_prefix(original_prefix, bottom_subset)?;
                            if retained.len() < 2 {
                                return Err(
                                    "scry retained-top prompt has no genuine ordering choice"
                                        .to_string(),
                                );
                            }
                            let count = u16::try_from(retained.len())
                                .map_err(|_| "scry retained group exceeds u16".to_string())?;
                            if *min_targets != count || *max_targets != count || !*ordered {
                                return Err(
                                    "scry retained-top prompt has a noncanonical shape".to_string()
                                );
                            }
                            validate_exact_binding_permutation(
                                &retained,
                                &candidates,
                                "scry retained-top candidates",
                            )?;
                        }
                    }
                }
                EffectTargetSelectionPurpose::SearchLibraryToHand {
                    player: library_player,
                    filter,
                    filter_fingerprint,
                    original_library,
                    canonical_path,
                } => {
                    if chooser != library_player {
                        return Err(
                            "library-search choice player does not own the selected library"
                                .to_string(),
                        );
                    }
                    if path != canonical_path {
                        return Err("library-search prompt structural path changed".to_string());
                    }
                    if *filter_fingerprint != library_filter_fingerprint(*filter) {
                        return Err("library-search filter fingerprint changed".to_string());
                    }
                    if *min_targets != 0 || *max_targets != 1 || *ordered || !selected.is_empty() {
                        return Err("library-search prompt has a noncanonical shape".to_string());
                    }
                    validate_library_search_live_metadata(
                        state,
                        *library_player,
                        *filter,
                        original_library,
                    )?;
                    let expected = library_search_candidates(
                        state,
                        *library_player,
                        *filter,
                        original_library,
                    )?;
                    let actual = legal
                        .iter()
                        .map(|candidate| {
                            candidate.expected_object.ok_or_else(|| {
                                "library-search target lacks an object-incarnation binding"
                                    .to_string()
                            })
                        })
                        .collect::<Result<Vec<_>, String>>()?;
                    if actual != expected {
                        return Err(
                            "library-search candidates are not the canonical physical match set"
                                .to_string(),
                        );
                    }
                }
                EffectTargetSelectionPurpose::SearchLibraryToBattlefieldTapped {
                    player: library_player,
                    filter,
                    filter_fingerprint,
                    original_library,
                    canonical_path,
                } => {
                    if chooser != library_player || path != canonical_path {
                        return Err(
                            "battlefield library-search choice player or path changed".to_string()
                        );
                    }
                    if *filter_fingerprint != library_filter_fingerprint(*filter)
                        || *min_targets != 0
                        || *max_targets != 1
                        || *ordered
                        || !selected.is_empty()
                    {
                        return Err("battlefield library-search prompt has a noncanonical shape"
                            .to_string());
                    }
                    validate_search_library_to_battlefield_origin(
                        state,
                        pending,
                        *library_player,
                        *filter,
                        canonical_path,
                    )?;
                    validate_library_search_live_metadata(
                        state,
                        *library_player,
                        *filter,
                        original_library,
                    )?;
                    let expected = library_search_candidates(
                        state,
                        *library_player,
                        *filter,
                        original_library,
                    )?;
                    let actual = legal
                        .iter()
                        .map(|candidate| {
                            candidate.expected_object.ok_or_else(|| {
                                "battlefield library-search target lacks an incarnation binding"
                                    .to_string()
                            })
                        })
                        .collect::<Result<Vec<_>, String>>()?;
                    if actual != expected {
                        return Err("battlefield library-search candidates changed".to_string());
                    }
                }
                EffectTargetSelectionPurpose::DuressDiscard {
                    player: hand_player,
                    original_hand,
                    eligible,
                    canonical_path,
                } => {
                    if *chooser != pending.ctx.controller
                        || path != canonical_path
                        || *hand_player == pending.ctx.controller
                        || pending.ctx.targets.as_slice() != [Target::Player(*hand_player)]
                        || pending.resolving_item.v4.target_spec
                            != Some(crate::card_def::TargetSpec::TargetOpponent)
                        || *min_targets != 1
                        || *max_targets != 1
                        || *ordered
                        || !selected.is_empty()
                    {
                        return Err("Duress discard prompt has a noncanonical shape".to_string());
                    }
                    validate_duress_origin(state, pending, *hand_player, canonical_path)?;
                    validate_bound_hand_exact(state, *hand_player, original_hand)?;
                    let recomputed = duress_eligible_hand(state, original_hand)?;
                    if eligible != &recomputed || eligible.is_empty() {
                        return Err("Duress eligible hand partition changed".to_string());
                    }
                    let actual = legal
                        .iter()
                        .map(|candidate| {
                            candidate.expected_object.ok_or_else(|| {
                                "Duress target lacks an object-incarnation binding".to_string()
                            })
                        })
                        .collect::<Result<Vec<_>, String>>()?;
                    if actual != *eligible {
                        return Err(
                            "Duress legal targets changed from the revealed hand".to_string()
                        );
                    }
                }
                EffectTargetSelectionPurpose::LookTopSelectByTypeToHandBottomRest {
                    player: library_player,
                    requested_count,
                    original_library_len,
                    card_type,
                    original_prefix,
                    stage,
                    stage_fingerprint,
                    canonical_path,
                } => {
                    if chooser != library_player {
                        return Err(
                            "library-partition choice player does not own the selected library"
                                .to_string(),
                        );
                    }
                    validate_library_partition_live_metadata(
                        state,
                        *library_player,
                        *requested_count,
                        *original_library_len,
                        *card_type,
                        original_prefix,
                    )?;
                    if *stage_fingerprint != library_partition_stage_fingerprint(stage) {
                        return Err(
                            "library-partition prompt stage fingerprint changed".to_string()
                        );
                    }
                    let mut expected_path = canonical_path.clone();
                    expected_path.push(library_partition_stage_tag(stage));
                    if path != &expected_path {
                        return Err("library-partition prompt structural path changed".to_string());
                    }
                    let candidates = selected
                        .iter()
                        .chain(legal)
                        .map(|candidate| {
                            candidate.expected_object.ok_or_else(|| {
                                "library-partition target lacks an object-incarnation binding"
                                    .to_string()
                            })
                        })
                        .collect::<Result<Vec<_>, String>>()?;
                    match stage {
                        LibraryPartitionSelectionStage::ChooseMatchingSubset => {
                            let matching = library_partition_matching_prefix(
                                state,
                                *card_type,
                                original_prefix,
                            )?;
                            let count = u16::try_from(matching.len()).map_err(|_| {
                                "library-partition matching set exceeds u16".to_string()
                            })?;
                            if *min_targets != 0 || *max_targets != count || *ordered {
                                return Err(
                                    "library-partition subset prompt has a noncanonical shape"
                                        .to_string(),
                                );
                            }
                            validate_exact_binding_permutation(
                                &matching,
                                &candidates,
                                "library-partition matching candidates",
                            )?;
                        }
                        LibraryPartitionSelectionStage::OrderRest { selected: chosen } => {
                            validate_canonical_binding_subset(original_prefix, chosen)?;
                            let matching = library_partition_matching_prefix(
                                state,
                                *card_type,
                                original_prefix,
                            )?;
                            if chosen.iter().any(|binding| !matching.contains(binding)) {
                                return Err(
                                    "library-partition selected card does not match the typed filter"
                                        .to_string(),
                                );
                            }
                            let rest = binding_partition_rest(original_prefix, chosen)?;
                            if rest.len() < 2 {
                                return Err(
                                    "library-partition rest-order prompt has no genuine choice"
                                        .to_string(),
                                );
                            }
                            let count = u16::try_from(rest.len()).map_err(|_| {
                                "library-partition rest set exceeds u16".to_string()
                            })?;
                            if *min_targets != count || *max_targets != count || !*ordered {
                                return Err(
                                    "library-partition rest-order prompt has a noncanonical shape"
                                        .to_string(),
                                );
                            }
                            validate_exact_binding_permutation(
                                &rest,
                                &candidates,
                                "library-partition rest-order candidates",
                            )?;
                        }
                    }
                }
                EffectTargetSelectionPurpose::SearchLibraryToHandMany {
                    player: library_player,
                    filter,
                    filter_fingerprint,
                    original_library,
                    max_targets: purpose_max,
                    canonical_path,
                } => {
                    if chooser != library_player || path != canonical_path {
                        return Err("multi-card library-search player or path changed".to_string());
                    }
                    if *filter_fingerprint != library_filter_fingerprint(*filter)
                        || *purpose_max == 0
                        || *max_targets != *purpose_max
                        || *min_targets != 0
                        || *ordered
                        || selected.len() >= usize::from(*max_targets)
                    {
                        return Err(
                            "multi-card library-search prompt has a noncanonical shape".to_string()
                        );
                    }
                    validate_library_search_live_metadata(
                        state,
                        *library_player,
                        *filter,
                        original_library,
                    )?;
                    let mut expected = library_search_candidates(
                        state,
                        *library_player,
                        *filter,
                        original_library,
                    )?;
                    let mut seen = Vec::with_capacity(selected.len());
                    for candidate in selected {
                        let binding = candidate.expected_object.ok_or_else(|| {
                            "multi-card library-search selection lacks an incarnation binding"
                                .to_string()
                        })?;
                        if candidate.target != Target::Object(binding.object)
                            || !expected.contains(&binding)
                            || seen.contains(&binding.object)
                        {
                            return Err("multi-card library-search selection is not a unique canonical match"
                                .to_string());
                        }
                        seen.push(binding.object);
                        expected.retain(|other| other != &binding);
                    }
                    let actual = legal
                        .iter()
                        .map(|candidate| {
                            candidate.expected_object.ok_or_else(|| {
                                "multi-card library-search target lacks an incarnation binding"
                                    .to_string()
                            })
                        })
                        .collect::<Result<Vec<_>, String>>()?;
                    if actual != expected {
                        return Err(
                            "multi-card library-search remaining candidates changed".to_string()
                        );
                    }
                }
                EffectTargetSelectionPurpose::ExileOneFromGraveyard {
                    player: graveyard_player,
                    original_graveyard,
                    canonical_path,
                } => {
                    if chooser != graveyard_player || path != canonical_path {
                        return Err(
                            "graveyard exile choice player or structural path changed".to_string()
                        );
                    }
                    if *min_targets != 1
                        || *max_targets != 1
                        || !*ordered
                        || !selected.is_empty()
                        || original_graveyard.len() < 2
                    {
                        return Err("graveyard exile prompt has a noncanonical shape".to_string());
                    }
                    validate_bound_graveyard_exact(state, *graveyard_player, original_graveyard)?;
                    let actual = legal
                        .iter()
                        .map(|candidate| {
                            candidate.expected_object.ok_or_else(|| {
                                "graveyard exile target lacks an incarnation binding".to_string()
                            })
                        })
                        .collect::<Result<Vec<_>, String>>()?;
                    if &actual != original_graveyard {
                        return Err(
                            "graveyard exile candidates changed from the bound graveyard"
                                .to_string(),
                        );
                    }
                }
                EffectTargetSelectionPurpose::ExileOneMatchingFromGraveyard {
                    player: graveyard_player,
                    card_type,
                    original_graveyard,
                    candidates,
                    then,
                    canonical_path,
                } => {
                    let mut expected_path = canonical_path.clone();
                    expected_path.push(1);
                    if chooser != graveyard_player || path != &expected_path {
                        return Err(
                            "filtered graveyard exile choice player or structural path changed"
                                .to_string(),
                        );
                    }
                    if *min_targets != 1
                        || *max_targets != 1
                        || !*ordered
                        || !selected.is_empty()
                        || candidates.len() < 2
                    {
                        return Err(
                            "filtered graveyard exile prompt has a noncanonical shape".to_string()
                        );
                    }
                    validate_bound_graveyard_exact(state, *graveyard_player, original_graveyard)?;
                    let expected =
                        matching_graveyard_bindings(state, original_graveyard, *card_type)?;
                    if &expected != candidates {
                        return Err("filtered graveyard exile candidates changed".to_string());
                    }
                    let actual = legal
                        .iter()
                        .map(|candidate| {
                            candidate.expected_object.ok_or_else(|| {
                                "filtered graveyard exile target lacks an incarnation binding"
                                    .to_string()
                            })
                        })
                        .collect::<Result<Vec<_>, String>>()?;
                    if &actual != candidates {
                        return Err("filtered graveyard exile legal targets changed".to_string());
                    }
                    validate_resumable_program(then)?;
                }
                EffectTargetSelectionPurpose::SacrificeCreature {
                    player: sacrifice_player,
                    filter,
                    original_candidates,
                    canonical_path,
                } => {
                    if chooser != sacrifice_player || path != canonical_path {
                        return Err(
                            "creature-sacrifice choice player or structural path changed"
                                .to_string(),
                        );
                    }
                    if *min_targets != 1
                        || *max_targets != 1
                        || !*ordered
                        || !selected.is_empty()
                        || original_candidates.len() < 2
                    {
                        return Err(
                            "creature-sacrifice prompt has a noncanonical shape".to_string()
                        );
                    }
                    let expected = creature_sacrifice_bindings(state, *sacrifice_player, *filter)?;
                    if &expected != original_candidates {
                        return Err("creature-sacrifice candidates changed".to_string());
                    }
                    let actual = legal
                        .iter()
                        .map(|candidate| {
                            candidate.expected_object.ok_or_else(|| {
                                "creature-sacrifice target lacks an incarnation binding".to_string()
                            })
                        })
                        .collect::<Result<Vec<_>, String>>()?;
                    if &actual != original_candidates {
                        return Err("creature-sacrifice legal targets changed".to_string());
                    }
                }
                EffectTargetSelectionPurpose::OrderRevealedIntoGraveyard {
                    player: library_player,
                    original_prefix,
                } => {
                    if chooser != library_player {
                        return Err(
                            "revealed-library ordering player does not own the library".to_string()
                        );
                    }
                    validate_bound_library_prefix_exact(state, *library_player, original_prefix)?;
                    let candidates = selected
                        .iter()
                        .chain(legal)
                        .map(|candidate| {
                            candidate.expected_object.ok_or_else(|| {
                                "revealed-library ordering target lacks an object binding"
                                    .to_string()
                            })
                        })
                        .collect::<Result<Vec<_>, String>>()?;
                    validate_exact_binding_permutation(
                        original_prefix,
                        &candidates,
                        "revealed-library ordering candidates",
                    )?;
                    let count = u16::try_from(original_prefix.len()).map_err(|_| {
                        "revealed-library prefix exceeds u16 target cardinality".to_string()
                    })?;
                    if *min_targets != count || *max_targets != count || !*ordered {
                        return Err(
                            "revealed-library ordering prompt has a noncanonical shape".to_string()
                        );
                    }
                }
                EffectTargetSelectionPurpose::UntapLands {
                    chooser: land_chooser,
                    max_targets: purpose_max,
                    original_candidates,
                    canonical_path,
                } => {
                    if chooser != land_chooser
                        || path != canonical_path
                        || *land_chooser != pending.ctx.controller
                        || *purpose_max == 0
                        || *max_targets != *purpose_max
                        || *min_targets != 0
                        || *ordered
                        || selected.len() >= usize::from(*max_targets)
                    {
                        return Err("land-untap prompt has a noncanonical shape".to_string());
                    }
                    if &tapped_land_bindings(state) != original_candidates {
                        return Err("land-untap original candidate set changed".to_string());
                    }
                    let mut partition = selected
                        .iter()
                        .chain(legal)
                        .map(|candidate| {
                            candidate.expected_object.ok_or_else(|| {
                                "land-untap target lacks an incarnation binding".to_string()
                            })
                        })
                        .collect::<Result<Vec<_>, String>>()?;
                    partition.sort_by_key(|binding| binding.object);
                    let mut expected = original_candidates.clone();
                    expected.sort_by_key(|binding| binding.object);
                    if partition != expected
                        || partition
                            .windows(2)
                            .any(|pair| pair[0].object == pair[1].object)
                    {
                        return Err(
                            "land-untap candidates no longer form the exact partition".to_string()
                        );
                    }
                }
                EffectTargetSelectionPurpose::LinkedExileNonlandFromRevealedHand {
                    player: hand_player,
                    original_hand,
                    source,
                    canonical_path,
                } => {
                    if chooser != &pending.ctx.controller || path != canonical_path {
                        return Err("linked-exile chooser or structural path changed".to_string());
                    }
                    if *min_targets != 1
                        || *max_targets != 1
                        || !*ordered
                        || !selected.is_empty()
                        || pending.resolving_item.v4.ability_source_contract != Some(*source)
                        || pending.ctx.ability_source_contract != Some(*source)
                    {
                        return Err("linked-exile prompt has a noncanonical shape".to_string());
                    }
                    validate_bound_hand_exact(state, *hand_player, original_hand)?;
                    for observer in [PlayerId::P0, PlayerId::P1] {
                        for binding in original_hand {
                            if observer != *hand_player
                                && !state.hand_knowledge[observer.index()][hand_player.index()]
                                    .iter()
                                    .any(|entry| {
                                        entry.object == binding.object
                                            && entry.zone_change_count
                                                == binding.expected_zone_change_count
                                    })
                            {
                                return Err(
                                    "linked-exile prompt lost its complete public hand reveal"
                                        .to_string(),
                                );
                            }
                        }
                    }
                    let expected = original_hand
                        .iter()
                        .copied()
                        .filter(|binding| {
                            !crate::card_def::CARD_DEFS
                                [state.objects.get(binding.object).card_def as usize]
                                .has_type(CardType::Land)
                        })
                        .collect::<Vec<_>>();
                    if expected.len() < 2 {
                        return Err(
                            "linked-exile prompt has no genuine multi-card choice".to_string()
                        );
                    }
                    let actual = legal
                        .iter()
                        .map(|candidate| {
                            candidate.expected_object.ok_or_else(|| {
                                "linked-exile target lacks an incarnation binding".to_string()
                            })
                        })
                        .collect::<Result<Vec<_>, String>>()?;
                    if actual != expected {
                        return Err("linked-exile candidates changed from the revealed nonlands"
                            .to_string());
                    }
                }
                EffectTargetSelectionPurpose::UndercityThroneCreature {
                    binding,
                    original_library,
                    revealed_prefix,
                    candidates,
                    canonical_path,
                } => {
                    if chooser != &binding.player
                        || path != canonical_path
                        || *min_targets != 1
                        || *max_targets != 1
                        || !*ordered
                        || !selected.is_empty()
                        || candidates.len() < 2
                        || !pending.frames.is_empty()
                    {
                        return Err("Throne prompt has a noncanonical shape".to_string());
                    }
                    validate_undercity_throne_metadata(
                        state,
                        pending,
                        *binding,
                        original_library,
                        revealed_prefix,
                        candidates,
                        None,
                    )?;
                    let actual = legal
                        .iter()
                        .map(|candidate| {
                            candidate.expected_object.ok_or_else(|| {
                                "Throne target lacks a library incarnation binding".to_string()
                            })
                        })
                        .collect::<Result<Vec<_>, String>>()?;
                    if &actual != candidates {
                        return Err("Throne prompt candidates changed".to_string());
                    }
                }
                EffectTargetSelectionPurpose::OrderIntoGraveyard { .. } => {}
            }
        }
        PendingEffectChoice::ChooseBoolean {
            player,
            path,
            purpose,
            ..
        } => match purpose {
            EffectBooleanChoicePurpose::ShuffleLibrary {
                player: library_player,
            } => {
                if player != library_player {
                    return Err("shuffle choice player/library mismatch".to_string());
                }
            }
            EffectBooleanChoicePurpose::CounterUnlessPaysGeneric {
                ward_target,
                targeting_stack_item,
                player: payer,
                generic,
            } => {
                let StackTargetContractV4::Object {
                    object: ward_source,
                    ..
                } = *ward_target
                else {
                    return Err("Ward Boolean choice lost its permanent binding".to_string());
                };
                if player != payer
                    || pending.resolving_item.source != ward_source
                    || !path.is_empty()
                    || !pending.frames.is_empty()
                {
                    return Err("Ward Boolean choice metadata is inconsistent".to_string());
                }
                validate_counter_unless_pays_generic(
                    state,
                    *ward_target,
                    pending.resolving_item.controller,
                    *targeting_stack_item,
                    *payer,
                    *generic,
                    false,
                    true,
                )?;
            }
            EffectBooleanChoicePurpose::CounterTargetUnlessPaysGeneric {
                target_contract,
                target_stack_item,
                player: payer,
                generic,
            } => {
                if player != payer || !path.is_empty() || !pending.frames.is_empty() {
                    return Err(
                        "counter-unless-pay Boolean choice metadata is inconsistent".to_string()
                    );
                }
                validate_counter_target_unless_pays_binding(
                    state,
                    pending,
                    *target_contract,
                    *target_stack_item,
                    *payer,
                    *generic,
                    true,
                )?;
            }
            EffectBooleanChoicePurpose::PayManaThen {
                player: payer,
                colored,
                generic,
                then,
            } => {
                if player != payer {
                    return Err("mana-payment choice player mismatch".to_string());
                }
                validate_resumable_program(then)?;
                if !crate::engine::can_pay_effect_mana(*payer, colored, *generic, state) {
                    return Err("pending optional mana cost is not payable".to_string());
                }
            }
            EffectBooleanChoicePurpose::SearchLibraryToBattlefieldTapped {
                player: library_player,
                filter,
                canonical_path,
                expected_remaining_frames,
            } => {
                if player != library_player
                    || path != canonical_path
                    || *filter != LibraryCardFilter::BasicLand
                    || &pending.frames != expected_remaining_frames
                {
                    return Err("battlefield-search Boolean metadata is inconsistent".to_string());
                }
                let root = validated_definition_owned_root_effect(state, pending)?;
                if effect_op_at_path(root.as_ref(), canonical_path)
                    != Some(&EffectOp::DestroyTargetLandThenMaySearchBasicTapped {
                        object: ObjectRef::Target(0),
                    })
                {
                    return Err(
                        "battlefield-search Boolean lost its originating operation".to_string()
                    );
                }
                let [StackTargetContractV4::Object {
                    controller,
                    zone: Zone::Battlefield,
                    ..
                }] = pending.ctx.target_contracts.as_slice()
                else {
                    return Err("battlefield-search Boolean lost its target binding".to_string());
                };
                if controller != library_player {
                    return Err("battlefield-search Boolean target controller changed".to_string());
                }
            }
            EffectBooleanChoicePurpose::PayExileFromGraveyardThen {
                player: graveyard_player,
                card_type,
                original_graveyard,
                candidates,
                then,
                canonical_path,
            } => {
                if player != graveyard_player || path != canonical_path || candidates.is_empty() {
                    return Err(
                        "filtered graveyard-exile Boolean choice metadata changed".to_string()
                    );
                }
                validate_bound_graveyard_exact(state, *graveyard_player, original_graveyard)?;
                let expected = matching_graveyard_bindings(state, original_graveyard, *card_type)?;
                if &expected != candidates {
                    return Err("filtered graveyard-exile candidates changed".to_string());
                }
                let root = validated_definition_owned_root_effect(state, pending)?;
                let Some(EffectOp::MayExileFromPlayersGraveyardMatchingThen {
                    player: original_player,
                    card_type: original_card_type,
                    then: original_then,
                }) = effect_op_at_structural_path(root.as_ref(), canonical_path)
                else {
                    return Err(
                        "filtered graveyard-exile choice lost its originating operation"
                            .to_string(),
                    );
                };
                if pending.ctx.resolve_player(*original_player, state) != *graveyard_player
                    || original_card_type != card_type
                    || original_then != then
                {
                    return Err("filtered graveyard-exile choice payload changed".to_string());
                }
                validate_resumable_program(then)?;
            }
        },
        PendingEffectChoice::ChooseOption {
            player,
            options,
            purpose,
            path,
        } => match purpose {
            EffectOptionChoicePurpose::Generic => {
                for option in options {
                    validate_resumable_program(option)?;
                }
            }
            EffectOptionChoicePurpose::OwnerLibrarySecondOrBottom {
                object,
                owner,
                canonical_path,
                expected_remaining_frames,
            } => {
                if !path.is_empty() || !canonical_path.is_empty() {
                    return Err(
                        "owner-library placement choice is not the generated root path".to_string(),
                    );
                }
                if !pending.frames.is_empty() || !expected_remaining_frames.is_empty() {
                    return Err(
                        "owner-library placement choice has a nonempty generated remainder"
                            .to_string(),
                    );
                }
                let [EffectOp::PutBoundObjectInOwnersLibrary {
                    object: second_object,
                    owner: second_owner,
                    placement: event::LibraryPlacement::SecondFromTop,
                }, EffectOp::PutBoundObjectInOwnersLibrary {
                    object: bottom_object,
                    owner: bottom_owner,
                    placement: event::LibraryPlacement::Bottom,
                }] = options.as_slice()
                else {
                    return Err(
                        "owner-library placement choice changed its exact two-option payload"
                            .to_string(),
                    );
                };
                if second_object != object
                    || bottom_object != object
                    || second_owner != owner
                    || bottom_owner != owner
                    || player != owner
                {
                    return Err(
                        "owner-library placement choice changed its bound object, owner, or printed option order"
                        .to_string(),
                    );
                }
                validate_owner_library_target_binding(state, pending, *object, *owner)?;
            }
            EffectOptionChoicePurpose::ExploreNonlandTop {
                player: library_player,
                top,
                canonical_path,
            } => {
                if player != library_player || path != canonical_path {
                    return Err("explore choice player or structural path changed".to_string());
                }
                let [EffectOp::Sequence(keep), EffectOp::MoveBoundObject {
                    object,
                    to_zone: Zone::Graveyard,
                    preserve_known_identity: true,
                }] = options.as_slice()
                else {
                    return Err(
                        "explore choice changed its exact keep-or-graveyard options".to_string()
                    );
                };
                if !keep.is_empty() || object != top {
                    return Err("explore choice changed its bound top card".to_string());
                }
                validate_effect_object_binding(state, *top)?;
                if top.expected_zone != Zone::Library
                    || state.players[library_player.index()]
                        .library
                        .first()
                        .copied()
                        != Some(top.object)
                    || state.objects.get(top.object).owner != *library_player
                    || crate::card_def::CARD_DEFS[state.objects.get(top.object).card_def as usize]
                        .has_type(CardType::Land)
                {
                    return Err(
                        "explore choice no longer binds the revealed nonland on top".to_string()
                    );
                }
            }
            EffectOptionChoicePurpose::ChooseColor {
                player: color_player,
                legal_colors,
                canonical_path,
                expected_remaining_frames,
            } => {
                if player != color_player
                    || path != canonical_path
                    || &pending.frames != expected_remaining_frames
                    || legal_colors.as_slice()
                        != [
                            ManaColor::W,
                            ManaColor::U,
                            ManaColor::B,
                            ManaColor::R,
                            ManaColor::G,
                        ]
                {
                    return Err("color choice metadata is not canonical".to_string());
                }
                let expected = legal_colors
                    .iter()
                    .copied()
                    .map(|color| EffectOp::InstallDamagePreventionFromColor {
                        player: *color_player,
                        color,
                    })
                    .collect::<Vec<_>>();
                if options != &expected {
                    return Err("color choice options changed from WUBRG order".to_string());
                }
            }
            EffectOptionChoicePurpose::UndercityRoute {
                binding,
                from_room,
                legal_rooms,
                canonical_path,
                expected_remaining_frames,
            } => {
                validate_initiative_continuation_root(state, pending, *binding)?;
                if !matches!(
                    binding.kind,
                    InitiativeTriggerKindV1::VentureAfterTaking
                        | InitiativeTriggerKindV1::VentureAtUpkeep
                ) || player != &binding.player
                    || path != canonical_path
                    || !canonical_path.is_empty()
                    || &pending.frames != expected_remaining_frames
                    || !expected_remaining_frames.is_empty()
                {
                    return Err("Undercity route prompt has noncanonical metadata".to_string());
                }
                let (expected_from, expected_rooms) = undercity_next_rooms(state, binding.player)?;
                if *from_room != expected_from || legal_rooms != &expected_rooms {
                    return Err("Undercity route graph changed".to_string());
                }
                let expected_options = legal_rooms
                    .iter()
                    .copied()
                    .map(|room| EffectOp::EnterUndercityRoom {
                        binding: *binding,
                        from_room: *from_room,
                        room,
                    })
                    .collect::<Vec<_>>();
                if options != &expected_options {
                    return Err("Undercity route options changed".to_string());
                }
            }
        },
    }
    Ok(())
}

fn validate_resumable_program(op: &EffectOp) -> Result<(), String> {
    match op {
        EffectOp::Sequence(ops) => {
            for inner in ops {
                validate_resumable_program(inner)?;
            }
        }
        EffectOp::Conditional { then, else_, .. } => {
            validate_resumable_program(then)?;
            validate_resumable_program(else_)?;
        }
        EffectOp::Choice { options, .. } => {
            for option in options {
                validate_resumable_program(option)?;
            }
        }
        EffectOp::MayPayManaThen { then, .. }
        | EffectOp::MayExileFromPlayersGraveyardMatchingThen { then, .. } => {
            validate_resumable_program(then)?
        }
        EffectOp::MayPayCostThen { .. } | EffectOp::OfferAffectedPlayerSpellCopy { .. } => {
            return Err(
                "choice-bearing programs cannot yet mix legacy-suspending effect leaves"
                    .to_string(),
            );
        }
        EffectOp::PutBoundObjectInOwnersLibrary { .. } => {
            return Err(
                "generated programs cannot contain an already-bound owner-library move".to_string(),
            );
        }
        EffectOp::EnterUndercityRoom { .. } | EffectOp::ResolveUndercityThrone { .. } => {
            return Err("generated programs cannot contain bound Undercity operations".to_string());
        }
        _ => {}
    }
    Ok(())
}

fn drive_resumable(state: &mut GameState) -> Result<ResumableProgress, String> {
    let mut continuation = state
        .engine
        .pending_effect
        .take()
        .ok_or("no effect continuation is pending")?;
    if continuation.choice.is_some() {
        state.engine.pending_effect = Some(continuation);
        return Ok(ResumableProgress::Suspended);
    }

    while let Some(frame) = continuation.frames.pop() {
        let EffectFrame::Program { op, path } = frame else {
            match frame {
                EffectFrame::MoveObjectsBatch {
                    objects,
                    to_zone,
                    preserve_known_identity,
                    order_resolved,
                    path,
                } => {
                    if to_zone == Zone::Graveyard && objects.len() >= 2 && !order_resolved {
                        for binding in &objects {
                            validate_effect_object_binding(state, *binding)?;
                        }
                        let player = state.objects.get(objects[0].object).owner;
                        assert!(
                            objects
                                .iter()
                                .all(|binding| state.objects.get(binding.object).owner == player),
                            "one graveyard-order batch must contain cards from one owner"
                        );
                        stage_graveyard_order_choice(
                            &mut continuation,
                            player,
                            path,
                            objects,
                            EffectTargetSelectionPurpose::OrderIntoGraveyard {
                                preserve_known_identity,
                            },
                        );
                        state.engine.pending_effect = Some(continuation);
                        return Ok(ResumableProgress::Suspended);
                    }
                    commit_zone_change_batch(state, &objects, to_zone, preserve_known_identity)?;
                }
                EffectFrame::MillLibraryBatch {
                    objects,
                    order_resolved,
                    path,
                } => {
                    validate_bound_library_prefix(state, &objects)?;
                    if objects.len() >= 2 && !order_resolved {
                        let player = state.objects.get(objects[0].object).owner;
                        // A mill instruction does not publicly reveal its
                        // library snapshot before the move. The owner must
                        // nevertheless see the cards to order them, so grant
                        // only that perspective exact temporary knowledge.
                        state.reveal_library_top(player, player, objects.len());
                        stage_graveyard_order_choice(
                            &mut continuation,
                            player,
                            path,
                            objects,
                            EffectTargetSelectionPurpose::OrderMilledIntoGraveyard,
                        );
                        state.engine.pending_effect = Some(continuation);
                        return Ok(ResumableProgress::Suspended);
                    }
                    commit_zone_change_batch(state, &objects, Zone::Graveyard, false)?;
                }
                EffectFrame::RevealedLibraryToGraveyardBatch {
                    player,
                    original_prefix,
                    objects,
                    order_resolved,
                    path,
                } => {
                    validate_bound_library_prefix_exact(state, player, &original_prefix)?;
                    validate_exact_binding_permutation(
                        &original_prefix,
                        &objects,
                        "revealed-library graveyard order",
                    )?;
                    if objects.len() >= 2 && !order_resolved {
                        stage_graveyard_order_choice(
                            &mut continuation,
                            player,
                            path,
                            objects,
                            EffectTargetSelectionPurpose::OrderRevealedIntoGraveyard {
                                player,
                                original_prefix,
                            },
                        );
                        state.engine.pending_effect = Some(continuation);
                        return Ok(ResumableProgress::Suspended);
                    }
                    commit_zone_change_batch(state, &objects, Zone::Graveyard, true)?;
                }
                EffectFrame::ReorderLibraryTop {
                    player,
                    expected_prefix,
                    ordered,
                    path: _,
                } => {
                    validate_bound_library_prefix_exact(state, player, &expected_prefix)?;
                    for &binding in &ordered {
                        validate_effect_object_binding(state, binding)?;
                    }
                    let mut expected_set = expected_prefix
                        .iter()
                        .map(|binding| binding.object)
                        .collect::<Vec<_>>();
                    let mut ordered_set = ordered
                        .iter()
                        .map(|binding| binding.object)
                        .collect::<Vec<_>>();
                    expected_set.sort_unstable();
                    ordered_set.sort_unstable();
                    if ordered_set != expected_set {
                        return Err(
                            "chosen library order is not the bound prefix permutation".to_string()
                        );
                    }
                    let ordered_ids = ordered
                        .iter()
                        .map(|binding| binding.object)
                        .collect::<Vec<_>>();
                    state.reorder_library_top(player, &ordered_ids, &[player])?;
                }
                EffectFrame::ShuffleLibrary { player, path: _ } => {
                    state
                        .shuffle_library(player)
                        .map_err(|error| error.to_string())?;
                }
                EffectFrame::ResolveCounterUnlessPaysGeneric {
                    ward_target,
                    targeting_stack_item,
                    player,
                    generic,
                    pay,
                    path,
                } => {
                    let answered_frame = EffectFrame::ResolveCounterUnlessPaysGeneric {
                        ward_target,
                        targeting_stack_item,
                        player,
                        generic,
                        pay,
                        path: path.clone(),
                    };
                    if !continuation.frames.is_empty()
                        || continuation.answered_choice_guard.as_ref()
                            != Some(&EffectAnsweredChoiceGuard::CounterUnlessPaysGeneric {
                                frame: Box::new(answered_frame),
                            })
                        || path.as_slice() != [u16::from(pay)]
                    {
                        return Err("Ward answered frame/guard mismatch".to_string());
                    }
                    validate_counter_unless_pays_generic(
                        state,
                        ward_target,
                        continuation.resolving_item.controller,
                        targeting_stack_item,
                        player,
                        generic,
                        false,
                        true,
                    )?;
                    continuation.answered_choice_guard = None;
                    if pay {
                        let plan =
                            crate::mana::can_pay(&generic_mana_cost(generic), 0, player, state)
                                .ok_or("the accepted Ward payment became unpayable")?;
                        crate::engine::pay_plan(state, player, &plan);
                    } else if crate::engine::counter_stack_item_by_id(state, targeting_stack_item)?
                        .is_none()
                    {
                        return Err(
                            "the declined Ward payment lost its bound stack item".to_string()
                        );
                    }
                }
                EffectFrame::ResolveCounterTargetUnlessPaysGeneric {
                    target_contract,
                    target_stack_item,
                    player,
                    generic,
                    pay,
                    path,
                } => {
                    let answered_frame = EffectFrame::ResolveCounterTargetUnlessPaysGeneric {
                        target_contract,
                        target_stack_item,
                        player,
                        generic,
                        pay,
                        path: path.clone(),
                    };
                    if !continuation.frames.is_empty()
                        || continuation.answered_choice_guard.as_ref()
                            != Some(&EffectAnsweredChoiceGuard::CounterTargetUnlessPaysGeneric {
                                frame: Box::new(answered_frame),
                            })
                        || path.as_slice() != [u16::from(pay)]
                    {
                        return Err("counter-unless-pay answered frame/guard mismatch".to_string());
                    }
                    validate_counter_target_unless_pays_binding(
                        state,
                        &continuation,
                        target_contract,
                        target_stack_item,
                        player,
                        generic,
                        true,
                    )?;
                    continuation.answered_choice_guard = None;
                    if pay {
                        let plan =
                            crate::mana::can_pay(&generic_mana_cost(generic), 0, player, state)
                                .ok_or(
                                    "the accepted counter-unless-pay payment became unpayable",
                                )?;
                        crate::engine::pay_plan(state, player, &plan);
                    } else if crate::engine::counter_stack_item_by_id(state, target_stack_item)?
                        .is_none()
                    {
                        return Err(
                            "the declined counter-unless-pay lost its bound spell".to_string()
                        );
                    }
                }
                EffectFrame::ExileChosenGraveyardCard {
                    player,
                    original_graveyard,
                    chosen,
                    path,
                    canonical_path,
                    expected_remaining_frames,
                } => {
                    let answered_frame = EffectFrame::ExileChosenGraveyardCard {
                        player,
                        original_graveyard,
                        chosen,
                        path,
                        canonical_path,
                        expected_remaining_frames: expected_remaining_frames.clone(),
                    };
                    if continuation.frames != expected_remaining_frames {
                        return Err(
                            "graveyard-exile answered continuation remainder changed".to_string()
                        );
                    }
                    if continuation.answered_choice_guard.as_ref()
                        != Some(&EffectAnsweredChoiceGuard::ExileOneFromGraveyard {
                            frame: Box::new(answered_frame.clone()),
                        })
                    {
                        return Err("graveyard-exile answered frame/guard mismatch".to_string());
                    }
                    validate_exile_chosen_graveyard_frame(state, &continuation, &answered_frame)?;
                    continuation.answered_choice_guard = None;
                    event::propose_and_commit(
                        state,
                        event::ProposedEvent::zone_change(chosen.object, Zone::Exile),
                    );
                }
                EffectFrame::ExileChosenMatchingGraveyardCard {
                    player,
                    card_type,
                    original_graveyard,
                    candidates,
                    chosen,
                    then,
                    path,
                    canonical_path,
                    expected_remaining_frames,
                } => {
                    let answered_frame = EffectFrame::ExileChosenMatchingGraveyardCard {
                        player,
                        card_type,
                        original_graveyard,
                        candidates,
                        chosen,
                        then: then.clone(),
                        path: path.clone(),
                        canonical_path,
                        expected_remaining_frames: expected_remaining_frames.clone(),
                    };
                    if continuation.frames != expected_remaining_frames {
                        return Err(
                            "filtered graveyard-exile answered continuation remainder changed"
                                .to_string(),
                        );
                    }
                    if continuation.answered_choice_guard.as_ref()
                        != Some(&EffectAnsweredChoiceGuard::ExileOneMatchingFromGraveyard {
                            frame: Box::new(answered_frame.clone()),
                        })
                    {
                        return Err(
                            "filtered graveyard-exile answered frame/guard mismatch".to_string()
                        );
                    }
                    validate_exile_chosen_matching_graveyard_frame(
                        state,
                        &continuation,
                        &answered_frame,
                    )?;
                    continuation.answered_choice_guard = None;
                    event::propose_and_commit(
                        state,
                        event::ProposedEvent::zone_change(chosen.object, Zone::Exile),
                    );
                    continuation
                        .frames
                        .push(EffectFrame::Program { op: *then, path });
                }
                EffectFrame::SacrificeChosenCreature {
                    player,
                    filter,
                    original_candidates,
                    chosen,
                    path,
                    canonical_path,
                    expected_remaining_frames,
                } => {
                    let answered_frame = EffectFrame::SacrificeChosenCreature {
                        player,
                        filter,
                        original_candidates,
                        chosen,
                        path,
                        canonical_path,
                        expected_remaining_frames: expected_remaining_frames.clone(),
                    };
                    if continuation.frames != expected_remaining_frames {
                        return Err("creature-sacrifice answered continuation remainder changed"
                            .to_string());
                    }
                    if continuation.answered_choice_guard.as_ref()
                        != Some(&EffectAnsweredChoiceGuard::SacrificeCreature {
                            frame: Box::new(answered_frame.clone()),
                        })
                    {
                        return Err("creature-sacrifice answered frame/guard mismatch".to_string());
                    }
                    validate_sacrifice_chosen_creature_frame(
                        state,
                        &continuation,
                        &answered_frame,
                    )?;
                    continuation.answered_choice_guard = None;
                    event::log_sacrifice(state, chosen.object);
                    event::propose_and_commit(
                        state,
                        event::ProposedEvent::zone_change(chosen.object, Zone::Graveyard),
                    );
                }
                EffectFrame::PayManaThen {
                    player,
                    colored,
                    generic,
                    then,
                    path,
                    canonical_path,
                    expected_remaining_frames,
                } => {
                    let answered_frame = EffectFrame::PayManaThen {
                        player,
                        colored: colored.clone(),
                        generic,
                        then: then.clone(),
                        path: path.clone(),
                        canonical_path,
                        expected_remaining_frames: expected_remaining_frames.clone(),
                    };
                    if continuation.frames != expected_remaining_frames {
                        return Err(
                            "optional-mana answered continuation remainder changed".to_string()
                        );
                    }
                    if continuation.answered_choice_guard.as_ref()
                        != Some(&EffectAnsweredChoiceGuard::PayManaThen {
                            frame: Box::new(answered_frame.clone()),
                        })
                    {
                        return Err("optional-mana answered frame/guard mismatch".to_string());
                    }
                    validate_pay_mana_then_frame(state, &continuation, &answered_frame)?;
                    continuation.answered_choice_guard = None;
                    if !crate::engine::pay_effect_mana(player, &colored, generic, state) {
                        return Err("accepted optional mana payment became unpayable".to_string());
                    }
                    continuation
                        .frames
                        .push(EffectFrame::Program { op: *then, path });
                }
                EffectFrame::PutCardsFromHandOnLibraryTop {
                    player,
                    total,
                    remaining,
                    prompt_index,
                    expected_hand,
                    chosen,
                    path,
                    canonical_path,
                } => {
                    validate_hand_to_library_progress(
                        total,
                        remaining,
                        prompt_index,
                        &path,
                        &canonical_path,
                    )?;
                    if remaining == 0 {
                        if prompt_index != u16::from(total)
                            || chosen.is_some()
                            || !expected_hand.is_empty()
                        {
                            return Err(
                                "completed hand-to-library frame has noncanonical progress"
                                    .to_string(),
                            );
                        }
                        continue;
                    }

                    if let Some(chosen) = chosen {
                        if expected_hand.is_empty() {
                            return Err(
                                "chosen hand-to-library frame lacks its bound hand snapshot"
                                    .to_string(),
                            );
                        }
                        validate_bound_hand_exact(state, player, &expected_hand)?;
                        if !expected_hand.contains(&chosen) {
                            return Err(
                                "chosen hand-to-library card is outside the bound hand".to_string()
                            );
                        }
                        let next_remaining = remaining
                            .checked_sub(1)
                            .ok_or("active hand-to-library frame has no remaining card count")?;
                        let next_prompt_index = prompt_index
                            .checked_add(1)
                            .ok_or("hand-to-library prompt index overflowed")?;
                        validate_hand_to_library_progress(
                            total,
                            next_remaining,
                            next_prompt_index,
                            &path,
                            &canonical_path,
                        )?;
                        let next_frame = EffectFrame::PutCardsFromHandOnLibraryTop {
                            player,
                            total,
                            remaining: next_remaining,
                            prompt_index: next_prompt_index,
                            expected_hand: Vec::new(),
                            chosen: None,
                            path,
                            canonical_path,
                        };
                        // The private subset choice invalidates every exact
                        // nonowner hand fact, not only the card that happened
                        // to be selected. Otherwise a previously known card
                        // left behind would reveal the hidden choice by
                        // elimination.
                        state.clear_nonowner_hand_knowledge(player);
                        event::propose_and_commit(
                            state,
                            event::ProposedEvent::private_top_library_insert(chosen.object),
                        );
                        if state.objects.get(chosen.object).zone != Zone::Library
                            || state.players[player.index()].library.first() != Some(&chosen.object)
                        {
                            return Err("private hand-to-library insertion did not commit on top"
                                .to_string());
                        }
                        continuation.frames.push(next_frame);
                        continue;
                    }

                    if !expected_hand.is_empty() {
                        return Err(
                            "hand-to-library coordinator carries an unchosen hand snapshot"
                                .to_string(),
                        );
                    }
                    let current_hand = bind_hand(state, player);
                    if current_hand.is_empty() {
                        continue;
                    }
                    state.clear_nonowner_hand_knowledge(player);
                    if current_hand.len() == 1 {
                        continuation
                            .frames
                            .push(EffectFrame::PutCardsFromHandOnLibraryTop {
                                player,
                                total,
                                remaining,
                                prompt_index,
                                chosen: current_hand.first().copied(),
                                expected_hand: current_hand,
                                path,
                                canonical_path,
                            });
                        continue;
                    }
                    stage_hand_to_library_choice(
                        &mut continuation,
                        player,
                        total,
                        remaining,
                        prompt_index,
                        canonical_path,
                        current_hand,
                    );
                    state.engine.pending_effect = Some(continuation);
                    return Ok(ResumableProgress::Suspended);
                }
                EffectFrame::ScryLibrary {
                    player,
                    requested_count,
                    original_library_len,
                    original_prefix,
                    progress,
                    progress_fingerprint,
                    path,
                    canonical_path,
                } => {
                    if path != canonical_path {
                        return Err(
                            "scry coordinator path changed from its canonical path".to_string()
                        );
                    }
                    if progress_fingerprint != scry_progress_fingerprint(&progress) {
                        return Err("scry coordinator progress fingerprint changed".to_string());
                    }
                    validate_scry_live_metadata(
                        state,
                        player,
                        requested_count,
                        original_library_len,
                        &original_prefix,
                    )?;
                    match progress {
                        ScryProgress::BottomSubsetChosen { bottom_subset } => {
                            validate_canonical_scry_subset(&original_prefix, &bottom_subset)?;
                            if bottom_subset.len() >= 2 {
                                stage_scry_choice(
                                    &mut continuation,
                                    player,
                                    requested_count,
                                    original_library_len,
                                    original_prefix,
                                    ScrySelectionStage::OrderBottom { bottom_subset },
                                    canonical_path,
                                )?;
                                state.engine.pending_effect = Some(continuation);
                                return Ok(ResumableProgress::Suspended);
                            }
                            let ordered_bottom = bottom_subset.clone();
                            let progress = ScryProgress::BottomOrderChosen {
                                bottom_subset,
                                ordered_bottom,
                            };
                            let progress_fingerprint = scry_progress_fingerprint(&progress);
                            continuation.frames.push(EffectFrame::ScryLibrary {
                                player,
                                requested_count,
                                original_library_len,
                                original_prefix,
                                progress,
                                progress_fingerprint,
                                path,
                                canonical_path,
                            });
                        }
                        ScryProgress::BottomOrderChosen {
                            bottom_subset,
                            ordered_bottom,
                        } => {
                            validate_canonical_scry_subset(&original_prefix, &bottom_subset)?;
                            validate_exact_binding_permutation(
                                &bottom_subset,
                                &ordered_bottom,
                                "scry ordered bottom",
                            )?;
                            let retained = scry_retained_prefix(&original_prefix, &bottom_subset)?;
                            if retained.len() >= 2 {
                                stage_scry_choice(
                                    &mut continuation,
                                    player,
                                    requested_count,
                                    original_library_len,
                                    original_prefix,
                                    ScrySelectionStage::OrderRetainedTop {
                                        bottom_subset,
                                        ordered_bottom,
                                    },
                                    canonical_path,
                                )?;
                                state.engine.pending_effect = Some(continuation);
                                return Ok(ResumableProgress::Suspended);
                            }
                            let progress = ScryProgress::TopOrderChosen {
                                bottom_subset,
                                ordered_bottom,
                                ordered_top: retained,
                            };
                            let progress_fingerprint = scry_progress_fingerprint(&progress);
                            continuation.frames.push(EffectFrame::ScryLibrary {
                                player,
                                requested_count,
                                original_library_len,
                                original_prefix,
                                progress,
                                progress_fingerprint,
                                path,
                                canonical_path,
                            });
                        }
                        ScryProgress::TopOrderChosen {
                            bottom_subset,
                            ordered_bottom,
                            ordered_top,
                        } => {
                            validate_canonical_scry_subset(&original_prefix, &bottom_subset)?;
                            validate_exact_binding_permutation(
                                &bottom_subset,
                                &ordered_bottom,
                                "scry ordered bottom",
                            )?;
                            let retained = scry_retained_prefix(&original_prefix, &bottom_subset)?;
                            validate_exact_binding_permutation(
                                &retained,
                                &ordered_top,
                                "scry ordered retained top",
                            )?;
                            let expected_prefix = original_prefix
                                .iter()
                                .map(|binding| crate::state::ObjectLinkV4 {
                                    object: binding.object,
                                    zone_change_count: binding.expected_zone_change_count,
                                })
                                .collect::<Vec<_>>();
                            let retained_top = ordered_top
                                .iter()
                                .map(|binding| binding.object)
                                .collect::<Vec<_>>();
                            let bottom = ordered_bottom
                                .iter()
                                .map(|binding| binding.object)
                                .collect::<Vec<_>>();
                            state.apply_scry_result(
                                player,
                                &expected_prefix,
                                &retained_top,
                                &bottom,
                            )?;
                        }
                    }
                }
                EffectFrame::SearchLibraryToHand {
                    player,
                    filter,
                    filter_fingerprint,
                    original_library,
                    selected,
                    path,
                    canonical_path,
                } => {
                    if path != canonical_path {
                        return Err(
                            "library-search coordinator path changed from its canonical path"
                                .to_string(),
                        );
                    }
                    if filter_fingerprint != library_filter_fingerprint(filter) {
                        return Err("library-search coordinator filter changed".to_string());
                    }
                    validate_library_search_live_metadata(
                        state,
                        player,
                        filter,
                        &original_library,
                    )?;
                    let candidates =
                        library_search_candidates(state, player, filter, &original_library)?;
                    if let Some(binding) = &selected {
                        if !candidates.contains(binding) {
                            return Err("library-search result is outside the canonical match set"
                                .to_string());
                        }
                    }
                    // The trailing shuffle must be authorized after all of
                    // the read-only metadata/candidate validation above and
                    // before the searched card moves or is revealed. The
                    // token is a nonserialized stack local held across those
                    // mutations and committed at the shuffle point below.
                    let shuffle_token = state
                        .preflight_library_shuffle(player)
                        .map_err(|error| error.to_string())?;
                    if let Some(binding) = selected {
                        // The search move itself is an ordinary replaceable
                        // zone change. Public reveal follows the successful
                        // move, then the remaining library is shuffled.
                        event::propose_and_commit(
                            state,
                            event::ProposedEvent::zone_change(binding.object, Zone::Hand),
                        );
                        if state.objects.get(binding.object).zone == Zone::Hand {
                            for observer in [PlayerId::P0, PlayerId::P1] {
                                state
                                    .reveal_hand_card(observer, player, binding.object)
                                    .expect("a successful searched-card move is publicly revealed");
                            }
                        }
                    }
                    state
                        .commit_library_shuffle(player, shuffle_token)
                        .map_err(|error| error.to_string())?;
                }
                EffectFrame::SearchLibraryToBattlefieldTapped {
                    player,
                    filter,
                    filter_fingerprint,
                    original_library,
                    selected,
                    path,
                    canonical_path,
                } => {
                    if path != canonical_path
                        || filter_fingerprint != library_filter_fingerprint(filter)
                    {
                        return Err(
                            "battlefield library-search coordinator metadata changed".to_string()
                        );
                    }
                    validate_search_library_to_battlefield_origin(
                        state,
                        &continuation,
                        player,
                        filter,
                        &canonical_path,
                    )?;
                    validate_library_search_live_metadata(
                        state,
                        player,
                        filter,
                        &original_library,
                    )?;
                    let candidates =
                        library_search_candidates(state, player, filter, &original_library)?;
                    if selected.is_some_and(|binding| !candidates.contains(&binding)) {
                        return Err(
                            "battlefield library-search result is outside the canonical match set"
                                .to_string(),
                        );
                    }
                    let shuffle_token = state
                        .preflight_library_shuffle(player)
                        .map_err(|error| error.to_string())?;
                    if let Some(binding) = selected {
                        event::propose_and_commit(
                            state,
                            event::ProposedEvent::zone_change_to_battlefield_tapped(binding.object),
                        );
                    }
                    state
                        .commit_library_shuffle(player, shuffle_token)
                        .map_err(|error| error.to_string())?;
                }
                EffectFrame::DiscardRevealedHandCard {
                    player,
                    original_hand,
                    eligible,
                    selected,
                    path,
                    canonical_path,
                } => {
                    if path != canonical_path {
                        return Err("Duress discard frame path changed".to_string());
                    }
                    validate_duress_origin(state, &continuation, player, &canonical_path)?;
                    validate_bound_hand_exact(state, player, &original_hand)?;
                    let recomputed = duress_eligible_hand(state, &original_hand)?;
                    if eligible != recomputed
                        || eligible
                            .iter()
                            .filter(|binding| **binding == selected)
                            .count()
                            != 1
                    {
                        return Err("Duress discard binding changed".to_string());
                    }
                    event::propose_and_commit(
                        state,
                        event::ProposedEvent::zone_change(selected.object, Zone::Graveyard),
                    );
                }
                EffectFrame::BeginSearchLibraryToBattlefieldTapped {
                    player,
                    filter,
                    path,
                    canonical_path,
                    expected_remaining_frames,
                } => {
                    let answered_frame = EffectFrame::BeginSearchLibraryToBattlefieldTapped {
                        player,
                        filter,
                        path: path.clone(),
                        canonical_path: canonical_path.clone(),
                        expected_remaining_frames: expected_remaining_frames.clone(),
                    };
                    if continuation.frames != expected_remaining_frames
                        || continuation.answered_choice_guard.as_ref()
                            != Some(
                                &EffectAnsweredChoiceGuard::SearchLibraryToBattlefieldTapped {
                                    frame: Box::new(answered_frame.clone()),
                                },
                            )
                    {
                        return Err("battlefield-search answered frame/guard mismatch".to_string());
                    }
                    validate_begin_search_library_to_battlefield_frame(
                        state,
                        &continuation,
                        &answered_frame,
                    )?;
                    continuation.answered_choice_guard = None;
                    let original_library = bind_library_exact(state, player);
                    validate_library_search_live_metadata(
                        state,
                        player,
                        filter,
                        &original_library,
                    )?;
                    let candidates =
                        library_search_candidates(state, player, filter, &original_library)?;
                    stage_library_search_to_battlefield_choice(
                        &mut continuation,
                        player,
                        filter,
                        original_library,
                        candidates,
                        canonical_path,
                    );
                    state.engine.pending_effect = Some(continuation);
                    return Ok(ResumableProgress::Suspended);
                }
                EffectFrame::LookTopSelectByTypeToHandBottomRest {
                    player,
                    requested_count,
                    original_library_len,
                    card_type,
                    original_prefix,
                    progress,
                    progress_fingerprint,
                    path,
                    canonical_path,
                } => {
                    if path != canonical_path {
                        return Err(
                            "library-partition coordinator path changed from its canonical path"
                                .to_string(),
                        );
                    }
                    if progress_fingerprint != library_partition_progress_fingerprint(&progress) {
                        return Err("library-partition coordinator progress fingerprint changed"
                            .to_string());
                    }
                    validate_library_partition_live_metadata(
                        state,
                        player,
                        requested_count,
                        original_library_len,
                        card_type,
                        &original_prefix,
                    )?;
                    match progress {
                        LibraryPartitionProgress::MatchingSubsetChosen { selected } => {
                            validate_canonical_binding_subset(&original_prefix, &selected)?;
                            let matching = library_partition_matching_prefix(
                                state,
                                card_type,
                                &original_prefix,
                            )?;
                            if selected.iter().any(|binding| !matching.contains(binding)) {
                                return Err(
                                    "library-partition selected card does not match the typed filter"
                                        .to_string(),
                                );
                            }
                            let rest = binding_partition_rest(&original_prefix, &selected)?;
                            if rest.len() >= 2 {
                                stage_library_partition_choice(
                                    &mut continuation,
                                    state,
                                    player,
                                    requested_count,
                                    original_library_len,
                                    card_type,
                                    original_prefix,
                                    LibraryPartitionSelectionStage::OrderRest { selected },
                                    canonical_path,
                                )?;
                                state.engine.pending_effect = Some(continuation);
                                return Ok(ResumableProgress::Suspended);
                            }
                            let progress = LibraryPartitionProgress::RestOrderChosen {
                                selected,
                                ordered_rest: rest,
                            };
                            let progress_fingerprint =
                                library_partition_progress_fingerprint(&progress);
                            continuation.frames.push(
                                EffectFrame::LookTopSelectByTypeToHandBottomRest {
                                    player,
                                    requested_count,
                                    original_library_len,
                                    card_type,
                                    original_prefix,
                                    progress,
                                    progress_fingerprint,
                                    path,
                                    canonical_path,
                                },
                            );
                        }
                        LibraryPartitionProgress::RestOrderChosen {
                            selected,
                            ordered_rest,
                        } => {
                            validate_canonical_binding_subset(&original_prefix, &selected)?;
                            let matching = library_partition_matching_prefix(
                                state,
                                card_type,
                                &original_prefix,
                            )?;
                            if selected.iter().any(|binding| !matching.contains(binding)) {
                                return Err(
                                    "library-partition selected card does not match the typed filter"
                                        .to_string(),
                                );
                            }
                            let rest = binding_partition_rest(&original_prefix, &selected)?;
                            validate_exact_binding_permutation(
                                &rest,
                                &ordered_rest,
                                "library-partition ordered rest",
                            )?;
                            let expected_prefix = original_prefix
                                .iter()
                                .map(|binding| crate::state::ObjectLinkV4 {
                                    object: binding.object,
                                    zone_change_count: binding.expected_zone_change_count,
                                })
                                .collect::<Vec<_>>();
                            let bottom = selected
                                .iter()
                                .chain(&ordered_rest)
                                .map(|binding| binding.object)
                                .collect::<Vec<_>>();
                            state.apply_scry_result(player, &expected_prefix, &[], &bottom)?;
                            let events = selected
                                .iter()
                                .map(|binding| {
                                    event::ProposedEvent::zone_change(binding.object, Zone::Hand)
                                })
                                .collect();
                            event::propose_and_commit_batch(state, events);
                            for binding in selected {
                                if state.objects.get(binding.object).zone == Zone::Hand {
                                    for observer in [PlayerId::P0, PlayerId::P1] {
                                        state
                                            .reveal_hand_card(observer, player, binding.object)
                                            .expect(
                                            "a successful selected-card move is publicly revealed",
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
                EffectFrame::SearchLibraryToHandMany {
                    player,
                    filter,
                    filter_fingerprint,
                    original_library,
                    selected,
                    max_targets,
                    path,
                    canonical_path,
                } => {
                    if path != canonical_path || max_targets == 0 {
                        return Err(
                            "multi-card library-search coordinator metadata changed".to_string()
                        );
                    }
                    if filter_fingerprint != library_filter_fingerprint(filter)
                        || selected.len() > usize::from(max_targets)
                    {
                        return Err("multi-card library-search contract changed".to_string());
                    }
                    validate_library_search_live_metadata(
                        state,
                        player,
                        filter,
                        &original_library,
                    )?;
                    let candidates =
                        library_search_candidates(state, player, filter, &original_library)?;
                    let mut seen = Vec::with_capacity(selected.len());
                    for binding in &selected {
                        if !candidates.contains(binding) || seen.contains(&binding.object) {
                            return Err(
                                "multi-card library-search result is not a unique canonical match"
                                    .to_string(),
                            );
                        }
                        seen.push(binding.object);
                    }
                    let shuffle_token = state
                        .preflight_library_shuffle(player)
                        .map_err(|error| error.to_string())?;
                    commit_zone_change_batch(state, &selected, Zone::Hand, false)?;
                    for binding in selected {
                        if state.objects.get(binding.object).zone == Zone::Hand {
                            for observer in [PlayerId::P0, PlayerId::P1] {
                                state
                                    .reveal_hand_card(observer, player, binding.object)
                                    .expect("a successful searched-card move is publicly revealed");
                            }
                        }
                    }
                    state
                        .commit_library_shuffle(player, shuffle_token)
                        .map_err(|error| error.to_string())?;
                }
                EffectFrame::UntapObjectsBatch {
                    player,
                    objects,
                    max_targets,
                    path: _,
                } => {
                    if player != continuation.ctx.controller
                        || max_targets == 0
                        || objects.len() > usize::from(max_targets)
                    {
                        return Err("land-untap frame metadata changed".to_string());
                    }
                    let candidates = tapped_land_bindings(state);
                    let mut seen = Vec::new();
                    for binding in objects {
                        validate_effect_object_binding(state, binding)?;
                        if !candidates.contains(&binding) || seen.contains(&binding.object) {
                            return Err(
                                "land-untap frame contains a stale or duplicate land".to_string()
                            );
                        }
                        seen.push(binding.object);
                        state.objects.get_mut(binding.object).tapped = false;
                    }
                }
                EffectFrame::LinkedExileChosenHandCard {
                    player,
                    original_hand,
                    chosen,
                    source,
                    path,
                    canonical_path,
                    expected_remaining_frames,
                } => {
                    let answered_frame = EffectFrame::LinkedExileChosenHandCard {
                        player,
                        original_hand,
                        chosen,
                        source,
                        path,
                        canonical_path,
                        expected_remaining_frames: expected_remaining_frames.clone(),
                    };
                    if continuation.frames != expected_remaining_frames {
                        return Err(
                            "linked-exile answered continuation remainder changed".to_string()
                        );
                    }
                    if continuation.answered_choice_guard.as_ref()
                        != Some(&EffectAnsweredChoiceGuard::LinkedExileFromRevealedHand {
                            frame: Box::new(answered_frame.clone()),
                        })
                    {
                        return Err("linked-exile answered frame/guard mismatch".to_string());
                    }
                    validate_linked_exile_chosen_hand_frame(state, &continuation, &answered_frame)?;
                    if state.engine.linked_exile_records.iter().any(|record| {
                        record.source.source == source.source
                            && record.source.zone_change_count == source.zone_change_count
                    }) {
                        return Err(
                            "linked-exile source incarnation already owns a card".to_string()
                        );
                    }
                    continuation.answered_choice_guard = None;
                    event::propose_and_commit(
                        state,
                        event::ProposedEvent::zone_change_preserving_known_identity(
                            chosen.object,
                            Zone::Exile,
                        ),
                    );
                    let exiled = state.objects.get(chosen.object);
                    let record = LinkedExileRecordV4 {
                        source,
                        exiled: chosen.object,
                        exiled_card_def: exiled.card_def,
                        exiled_owner: exiled.owner,
                        exiled_zone_change_count: exiled.zone_change_count,
                    };
                    let source_incarnation_is_live =
                        state.objects.try_get(source.source).is_some_and(|live| {
                            live.zone == source.zone
                                && live.zone_change_count == source.zone_change_count
                        });
                    if source_incarnation_is_live {
                        state.objects.get_mut(chosen.object).v4.exiled_by = Some(ObjectLinkV4 {
                            object: source.source,
                            zone_change_count: source.zone_change_count,
                        });
                        state.engine.linked_exile_records.push(record);
                    }
                }
                EffectFrame::OwnerLibraryPlacement {
                    object,
                    owner,
                    placement,
                    option_index,
                    path,
                    canonical_path,
                    expected_remaining_frames,
                } => {
                    let answered_frame = EffectFrame::OwnerLibraryPlacement {
                        object,
                        owner,
                        placement,
                        option_index,
                        path,
                        canonical_path,
                        expected_remaining_frames: expected_remaining_frames.clone(),
                    };
                    if continuation.frames != expected_remaining_frames {
                        return Err(
                            "owner-library answered continuation remainder changed".to_string()
                        );
                    }
                    if continuation.answered_choice_guard.as_ref()
                        != Some(&EffectAnsweredChoiceGuard::OwnerLibrarySecondOrBottom {
                            frame: Box::new(answered_frame.clone()),
                        })
                    {
                        return Err("owner-library answered frame/guard mismatch".to_string());
                    }
                    validate_owner_library_placement_frame(state, &continuation, &answered_frame)?;
                    continuation.answered_choice_guard = None;
                    event::propose_and_commit(
                        state,
                        event::ProposedEvent::public_library_insert(object.object, placement),
                    );
                }
                EffectFrame::EnterUndercityRoom {
                    binding,
                    from_room,
                    room,
                    path,
                    expected_remaining_frames,
                } => {
                    let answered_frame = EffectFrame::EnterUndercityRoom {
                        binding,
                        from_room,
                        room,
                        path: path.clone(),
                        expected_remaining_frames: expected_remaining_frames.clone(),
                    };
                    if continuation.frames != expected_remaining_frames {
                        return Err("Undercity route continuation remainder changed".to_string());
                    }
                    if let Some(guard) = continuation.answered_choice_guard.as_ref() {
                        if guard
                            != &(EffectAnsweredChoiceGuard::UndercityRoute {
                                frame: Box::new(answered_frame.clone()),
                            })
                        {
                            return Err("Undercity route frame/guard mismatch".to_string());
                        }
                        validate_undercity_route_frame(state, &continuation, &answered_frame)?;
                        continuation.answered_choice_guard = None;
                    } else {
                        validate_initiative_continuation_root(state, &continuation, binding)?;
                        let (expected_from, legal_rooms) =
                            undercity_next_rooms(state, binding.player)?;
                        if from_room != expected_from
                            || legal_rooms.as_slice() != [room]
                            || path.as_slice() != [0]
                            || !expected_remaining_frames.is_empty()
                        {
                            return Err("automatic Undercity route changed".to_string());
                        }
                    }
                    enter_undercity_room(state, binding, from_room, room)?;
                }
                EffectFrame::ResolveUndercityThrone {
                    binding,
                    original_library,
                    revealed_prefix,
                    candidates,
                    chosen,
                    path,
                    canonical_path,
                    expected_remaining_frames,
                } => {
                    let answered_frame = EffectFrame::ResolveUndercityThrone {
                        binding,
                        original_library,
                        revealed_prefix,
                        candidates,
                        chosen,
                        path,
                        canonical_path,
                        expected_remaining_frames: expected_remaining_frames.clone(),
                    };
                    if continuation.frames != expected_remaining_frames
                        || continuation.answered_choice_guard.as_ref()
                            != Some(&EffectAnsweredChoiceGuard::UndercityThrone {
                                frame: Box::new(answered_frame.clone()),
                            })
                    {
                        return Err("Throne answered frame/guard mismatch".to_string());
                    }
                    validate_undercity_throne_frame(state, &continuation, &answered_frame)?;
                    continuation.answered_choice_guard = None;
                    apply_undercity_throne_result(state, binding, Some(chosen))?;
                }
                EffectFrame::Program { .. } => unreachable!(),
            }
            continue;
        };
        match op {
            EffectOp::Sequence(ops) => {
                for (index, inner) in ops.into_iter().enumerate().rev() {
                    let mut inner_path = path.clone();
                    inner_path.push(index as u16);
                    continuation.frames.push(EffectFrame::Program {
                        op: inner,
                        path: inner_path,
                    });
                }
            }
            EffectOp::Conditional { cond, then, else_ } => {
                let branch = if eval_cond(&cond, &continuation.ctx, state) {
                    0
                } else {
                    1
                };
                let mut branch_path = path;
                branch_path.push(branch);
                continuation.frames.push(EffectFrame::Program {
                    op: if branch == 0 { *then } else { *else_ },
                    path: branch_path,
                });
            }
            EffectOp::Choice {
                controller,
                mut options,
            } => match options.len() {
                0 => {}
                1 => {
                    let mut option_path = path;
                    option_path.push(0);
                    continuation.frames.push(EffectFrame::Program {
                        op: options.remove(0),
                        path: option_path,
                    });
                }
                _ => {
                    let player = continuation.ctx.resolve_player(controller, state);
                    continuation.choice = Some(PendingEffectChoice::ChooseOption {
                        player,
                        path,
                        options,
                        purpose: EffectOptionChoicePurpose::Generic,
                    });
                    state.engine.pending_effect = Some(continuation);
                    return Ok(ResumableProgress::Suspended);
                }
            },
            EffectOp::ResolveInitiativeTrigger { binding } => {
                validate_initiative_continuation_root(state, &continuation, binding)?;
                match binding.kind {
                    InitiativeTriggerKindV1::CombatTransfer => {
                        if !path.is_empty() || !continuation.frames.is_empty() {
                            return Err("Initiative transfer is not a root-only effect".to_string());
                        }
                        take_initiative(state, binding.player, binding.source)?;
                    }
                    InitiativeTriggerKindV1::VentureAfterTaking
                    | InitiativeTriggerKindV1::VentureAtUpkeep => {
                        if stage_undercity_venture(state, &mut continuation, binding, path)? {
                            state.engine.pending_effect = Some(continuation);
                            return Ok(ResumableProgress::Suspended);
                        }
                    }
                    InitiativeTriggerKindV1::UndercityRoom(room) => {
                        if !path.is_empty() || !continuation.frames.is_empty() {
                            return Err(
                                "Undercity room ability is not a root-only effect".to_string()
                            );
                        }
                        if room == UndercityRoomV1::ThroneOfTheDeadThree {
                            if stage_or_apply_undercity_throne(
                                state,
                                &mut continuation,
                                binding,
                                path,
                            )? {
                                state.engine.pending_effect = Some(continuation);
                                return Ok(ResumableProgress::Suspended);
                            }
                        } else {
                            continuation.frames.push(EffectFrame::Program {
                                op: undercity_room_program(binding, room)?,
                                path: vec![0],
                            });
                        }
                    }
                }
            }
            EffectOp::EnterUndercityRoom { .. } | EffectOp::ResolveUndercityThrone { .. } => {
                return Err(
                    "bound Undercity operation escaped its typed interpreter frame".to_string(),
                );
            }
            EffectOp::CounterUnlessPaysGeneric {
                ward_target,
                targeting_stack_item,
                generic,
            } => {
                let StackTargetContractV4::Object {
                    object: ward_source,
                    ..
                } = ward_target
                else {
                    return Err("Ward effect lost its permanent binding".to_string());
                };
                if continuation.ctx.source != ward_source || !path.is_empty() {
                    return Err("Ward effect no longer matches its root trigger".to_string());
                }
                let Some(bound) = validate_counter_unless_pays_generic(
                    state,
                    ward_target,
                    continuation.ctx.controller,
                    targeting_stack_item,
                    state
                        .stack
                        .iter()
                        .find(|item| item.v4.stack_item_id == targeting_stack_item)
                        .map(|item| item.controller)
                        .unwrap_or(continuation.ctx.controller.opponent()),
                    generic,
                    true,
                    false,
                )?
                else {
                    // Another Ward trigger already countered this targeter.
                    continue;
                };
                let player = bound.controller;
                if crate::mana::can_pay(&generic_mana_cost(generic), 0, player, state).is_none() {
                    crate::engine::counter_stack_item_by_id(state, targeting_stack_item)?
                        .ok_or("the unpayable Ward binding disappeared during resolution")?;
                    continue;
                }
                if !continuation.frames.is_empty() {
                    return Err("Ward payment is not a root-only generated effect".to_string());
                }
                validate_counter_unless_pays_generic(
                    state,
                    ward_target,
                    continuation.ctx.controller,
                    targeting_stack_item,
                    player,
                    generic,
                    false,
                    true,
                )?;
                continuation.choice = Some(PendingEffectChoice::ChooseBoolean {
                    player,
                    path,
                    default: Some(false),
                    purpose: EffectBooleanChoicePurpose::CounterUnlessPaysGeneric {
                        ward_target,
                        targeting_stack_item,
                        player,
                        generic,
                    },
                });
                state.engine.pending_effect = Some(continuation);
                return Ok(ResumableProgress::Suspended);
            }
            EffectOp::CounterTargetUnlessPaysGeneric { target, generic } => {
                if target != TargetRef::Target(0)
                    || !path.is_empty()
                    || !continuation.frames.is_empty()
                {
                    return Err(
                        "counter-unless-pay must be a root effect bound to target zero".to_string(),
                    );
                }
                let [target_contract] = continuation.ctx.target_contracts.as_slice() else {
                    return Err(
                        "counter-unless-pay lost its single cast-time target contract".to_string(),
                    );
                };
                let target_contract = *target_contract;
                let StackTargetContractV4::Object {
                    object,
                    zone: Zone::Stack,
                    ..
                } = target_contract
                else {
                    return Err("counter-unless-pay target is not a stack spell".to_string());
                };
                let target_items = state
                    .stack
                    .iter()
                    .filter(|item| {
                        item.source == object && item.kind == crate::state::StackItemKind::Spell
                    })
                    .cloned()
                    .collect::<Vec<_>>();
                let [target_item] = target_items.as_slice() else {
                    return Err(format!(
                        "counter-unless-pay expected one targeted spell, found {}",
                        target_items.len()
                    ));
                };
                let target_stack_item = target_item.v4.stack_item_id;
                let player = target_item.controller;
                validate_counter_target_unless_pays_binding(
                    state,
                    &continuation,
                    target_contract,
                    target_stack_item,
                    player,
                    generic,
                    false,
                )?;
                if crate::mana::can_pay(&generic_mana_cost(generic), 0, player, state).is_none() {
                    crate::engine::counter_stack_item_by_id(state, target_stack_item)?
                        .ok_or("the unpayable counter-unless-pay target disappeared")?;
                    continue;
                }
                validate_counter_target_unless_pays_binding(
                    state,
                    &continuation,
                    target_contract,
                    target_stack_item,
                    player,
                    generic,
                    true,
                )?;
                continuation.choice = Some(PendingEffectChoice::ChooseBoolean {
                    player,
                    path,
                    default: Some(false),
                    purpose: EffectBooleanChoicePurpose::CounterTargetUnlessPaysGeneric {
                        target_contract,
                        target_stack_item,
                        player,
                        generic,
                    },
                });
                state.engine.pending_effect = Some(continuation);
                return Ok(ResumableProgress::Suspended);
            }
            EffectOp::RevealTopAndPartitionByType {
                player,
                count,
                card_type,
                matching_to,
                rest_to,
            } => {
                let player = continuation.ctx.resolve_player(player, state);
                let (matching, rest) =
                    reveal_top_and_partition(state, player, count, card_type, matching_to, rest_to);
                // Frames are LIFO: push the rest group first so the matching
                // group is committed/ordered first, mirroring XMage's two
                // separate moveCards calls.
                for (group_index, objects, to_zone) in
                    [(1_u16, rest, rest_to), (0_u16, matching, matching_to)]
                {
                    let mut group_path = path.clone();
                    group_path.push(group_index);
                    continuation.frames.push(EffectFrame::MoveObjectsBatch {
                        objects,
                        to_zone,
                        preserve_known_identity: true,
                        order_resolved: false,
                        path: group_path,
                    });
                }
            }
            EffectOp::MillCards { player, count } => {
                let player = continuation.ctx.resolve_player(player, state);
                continuation.frames.push(EffectFrame::MillLibraryBatch {
                    objects: bind_library_top(state, player, count),
                    order_resolved: false,
                    path,
                });
            }
            EffectOp::RevealUntilCardTypeAndMill { player, card_type } => {
                let player = continuation.ctx.resolve_player(player, state);
                let original_prefix = bind_library_through_first_type(state, player, card_type);
                for observer in [PlayerId::P0, PlayerId::P1] {
                    state.reveal_library_top(observer, player, original_prefix.len());
                }
                continuation
                    .frames
                    .push(EffectFrame::RevealedLibraryToGraveyardBatch {
                        player,
                        objects: original_prefix.clone(),
                        original_prefix,
                        order_resolved: false,
                        path,
                    });
            }
            EffectOp::LookAtLibraryTopAndReorder { player, count } => {
                let player = continuation.ctx.resolve_player(player, state);
                let original_prefix = bind_library_top(state, player, count);
                // A private look changes only this observer's knowledge. A
                // 0/1-card prefix has no ordering choice and must not erase
                // another observer's still-valid prior fact.
                state.reveal_library_top(player, player, original_prefix.len());
                if original_prefix.len() >= 2 {
                    stage_library_order_choice(&mut continuation, player, path, original_prefix);
                    state.engine.pending_effect = Some(continuation);
                    return Ok(ResumableProgress::Suspended);
                }
            }
            EffectOp::MayShuffleLibrary { player } => {
                let player = continuation.ctx.resolve_player(player, state);
                continuation.choice = Some(PendingEffectChoice::ChooseBoolean {
                    player,
                    path,
                    default: Some(false),
                    purpose: EffectBooleanChoicePurpose::ShuffleLibrary { player },
                });
                state.engine.pending_effect = Some(continuation);
                return Ok(ResumableProgress::Suspended);
            }
            EffectOp::PutCardsFromHandOnLibraryTop { player, count } => {
                let player = continuation.ctx.resolve_player(player, state);
                let canonical_path = path.clone();
                continuation
                    .frames
                    .push(EffectFrame::PutCardsFromHandOnLibraryTop {
                        player,
                        total: count,
                        remaining: count,
                        prompt_index: 0,
                        expected_hand: Vec::new(),
                        chosen: None,
                        path,
                        canonical_path,
                    });
            }
            EffectOp::Scry { player, count } => {
                let player = continuation.ctx.resolve_player(player, state);
                let original_library_len = state.players[player.index()]
                    .library
                    .len()
                    .try_into()
                    .expect("a live library length fits the u32 state contract");
                // Reject outside the certified Scry2 envelope before even a
                // private prefix binding is materialized.
                validate_scry_static_metadata(count, original_library_len)?;
                let original_prefix = bind_library_top(state, player, count);
                validate_scry_bound_metadata(count, original_library_len, &original_prefix)?;
                // Looking is private. Candidate identities remain visible to
                // the owner alone through both state knowledge and RL
                // projection; another observer receives only the typed choice
                // envelope and its public cardinalities.
                state.reveal_library_top(player, player, original_prefix.len());
                if !original_prefix.is_empty() {
                    stage_scry_choice(
                        &mut continuation,
                        player,
                        count,
                        original_library_len,
                        original_prefix,
                        ScrySelectionStage::ChooseBottomSubset,
                        path,
                    )?;
                    state.engine.pending_effect = Some(continuation);
                    return Ok(ResumableProgress::Suspended);
                }
            }
            EffectOp::SearchLibraryToHand { player, filter } => {
                let player = continuation.ctx.resolve_player(player, state);
                let original_library = bind_library_exact(state, player);
                validate_library_search_live_metadata(state, player, filter, &original_library)?;
                let candidates =
                    library_search_candidates(state, player, filter, &original_library)?;
                // Always suspend on the same private, Finish-capable prompt,
                // even when there are zero matches. Auto-resolving the empty
                // case would let a nonacting observer distinguish zero from
                // nonzero matches by the mere presence of a continuation.
                stage_library_search_choice(
                    &mut continuation,
                    player,
                    filter,
                    original_library,
                    candidates,
                    path,
                );
                state.engine.pending_effect = Some(continuation);
                return Ok(ResumableProgress::Suspended);
            }
            EffectOp::LookTopSelectByTypeToHandBottomRest {
                player,
                count,
                card_type,
            } => {
                let player = continuation.ctx.resolve_player(player, state);
                let original_library_len = state.players[player.index()]
                    .library
                    .len()
                    .try_into()
                    .expect("a live library length fits the u32 state contract");
                let original_prefix = bind_library_top(state, player, count);
                validate_library_partition_live_metadata(
                    state,
                    player,
                    count,
                    original_library_len,
                    card_type,
                    &original_prefix,
                )?;
                state.reveal_library_top(player, player, original_prefix.len());
                if !original_prefix.is_empty() {
                    stage_library_partition_choice(
                        &mut continuation,
                        state,
                        player,
                        count,
                        original_library_len,
                        card_type,
                        original_prefix,
                        LibraryPartitionSelectionStage::ChooseMatchingSubset,
                        path,
                    )?;
                    state.engine.pending_effect = Some(continuation);
                    return Ok(ResumableProgress::Suspended);
                }
            }
            EffectOp::SearchLibraryToHandUpTo {
                player,
                filter,
                max_targets,
            } => {
                if max_targets == 0 {
                    return Err("multi-card library search requires a positive maximum".to_string());
                }
                let player = continuation.ctx.resolve_player(player, state);
                let original_library = bind_library_exact(state, player);
                validate_library_search_live_metadata(state, player, filter, &original_library)?;
                let candidates =
                    library_search_candidates(state, player, filter, &original_library)?;
                stage_library_search_many_choice(
                    &mut continuation,
                    player,
                    filter,
                    original_library,
                    candidates,
                    max_targets,
                    path,
                );
                state.engine.pending_effect = Some(continuation);
                return Ok(ResumableProgress::Suspended);
            }
            EffectOp::ExileOneFromPlayersGraveyard { player } => {
                let player = continuation.ctx.resolve_player(player, state);
                let original_graveyard = bind_graveyard_exact(state, player);
                validate_bound_graveyard_exact(state, player, &original_graveyard)?;
                match original_graveyard.len() {
                    0 => {}
                    1 => continuation.frames.push(EffectFrame::MoveObjectsBatch {
                        objects: original_graveyard,
                        to_zone: Zone::Exile,
                        preserve_known_identity: false,
                        order_resolved: true,
                        path,
                    }),
                    _ => {
                        stage_graveyard_exile_choice(
                            &mut continuation,
                            player,
                            original_graveyard,
                            path,
                        );
                        state.engine.pending_effect = Some(continuation);
                        return Ok(ResumableProgress::Suspended);
                    }
                }
            }
            EffectOp::MayExileFromPlayersGraveyardMatchingThen {
                player,
                card_type,
                then,
            } => {
                let player = continuation.ctx.resolve_player(player, state);
                let original_graveyard = bind_graveyard_exact(state, player);
                validate_bound_graveyard_exact(state, player, &original_graveyard)?;
                let candidates =
                    matching_graveyard_bindings(state, &original_graveyard, card_type)?;
                if !candidates.is_empty() {
                    continuation.choice = Some(PendingEffectChoice::ChooseBoolean {
                        player,
                        path: path.clone(),
                        default: Some(false),
                        purpose: EffectBooleanChoicePurpose::PayExileFromGraveyardThen {
                            player,
                            card_type,
                            original_graveyard,
                            candidates,
                            then,
                            canonical_path: path,
                        },
                    });
                    state.engine.pending_effect = Some(continuation);
                    return Ok(ResumableProgress::Suspended);
                }
            }
            EffectOp::SacrificeCreature { player, filter } => {
                let player = continuation.ctx.resolve_player(player, state);
                let candidates = creature_sacrifice_bindings(state, player, filter)?;
                match candidates.as_slice() {
                    [] => {}
                    [chosen] => {
                        let chosen = *chosen;
                        let expected_remaining_frames = continuation.frames.clone();
                        let frame = EffectFrame::SacrificeChosenCreature {
                            player,
                            filter,
                            original_candidates: candidates,
                            chosen,
                            path: path.clone(),
                            canonical_path: path,
                            expected_remaining_frames,
                        };
                        continuation.answered_choice_guard =
                            Some(EffectAnsweredChoiceGuard::SacrificeCreature {
                                frame: Box::new(frame.clone()),
                            });
                        continuation.frames.push(frame);
                    }
                    _ => {
                        stage_creature_sacrifice_choice(
                            &mut continuation,
                            player,
                            filter,
                            candidates,
                            path,
                        );
                        state.engine.pending_effect = Some(continuation);
                        return Ok(ResumableProgress::Suspended);
                    }
                }
            }
            EffectOp::RevealHandChooseNonlandToLinkedExile { player } => {
                if !path.is_empty() || !continuation.frames.is_empty() {
                    return Err("linked-exile hand choice must be a root effect".to_string());
                }
                let player = continuation.ctx.resolve_player(player, state);
                let source = continuation
                    .resolving_item
                    .v4
                    .ability_source_contract
                    .ok_or("linked-exile effect lost its historical source contract")?;
                if continuation.ctx.ability_source_contract != Some(source)
                    || source.source != continuation.ctx.source
                    || source.controller != continuation.ctx.controller
                    || source.zone != Zone::Battlefield
                    || source.attached_to.is_some()
                    || crate::card_def::CARD_DEFS[source.card_def as usize].name != "Mesmeric Fiend"
                {
                    return Err("linked-exile effect has the wrong source contract".to_string());
                }
                let original_hand = bind_hand(state, player);
                validate_bound_hand_exact(state, player, &original_hand)?;
                for binding in &original_hand {
                    for observer in [PlayerId::P0, PlayerId::P1] {
                        state
                            .reveal_hand_card(observer, player, binding.object)
                            .map_err(|error| {
                                format!("linked-exile public hand reveal failed: {error}")
                            })?;
                    }
                }
                let candidates = original_hand
                    .iter()
                    .copied()
                    .filter(|binding| {
                        !crate::card_def::CARD_DEFS
                            [state.objects.get(binding.object).card_def as usize]
                            .has_type(CardType::Land)
                    })
                    .collect::<Vec<_>>();
                match candidates.as_slice() {
                    [] => {}
                    [chosen] => {
                        let expected_remaining_frames = continuation.frames.clone();
                        let frame = EffectFrame::LinkedExileChosenHandCard {
                            player,
                            original_hand,
                            chosen: *chosen,
                            source,
                            path: path.clone(),
                            canonical_path: path,
                            expected_remaining_frames,
                        };
                        continuation.answered_choice_guard =
                            Some(EffectAnsweredChoiceGuard::LinkedExileFromRevealedHand {
                                frame: Box::new(frame.clone()),
                            });
                        continuation.frames.push(frame);
                    }
                    _ => {
                        let chooser = continuation.ctx.controller;
                        stage_linked_exile_hand_choice(
                            &mut continuation,
                            chooser,
                            player,
                            original_hand,
                            candidates,
                            source,
                            path,
                        );
                        state.engine.pending_effect = Some(continuation);
                        return Ok(ResumableProgress::Suspended);
                    }
                }
            }
            EffectOp::ReturnLinkedExiledCardToOwnersHand => {
                if !path.is_empty() || !continuation.frames.is_empty() {
                    return Err("linked-exile return must be a root effect".to_string());
                }
                let source = continuation
                    .resolving_item
                    .v4
                    .ability_source_contract
                    .ok_or("linked-exile return lost its historical source contract")?;
                if continuation.ctx.ability_source_contract != Some(source)
                    || source.source != continuation.ctx.source
                    || source.controller != continuation.ctx.controller
                    || source.zone != Zone::Battlefield
                    || source.attached_to.is_some()
                    || crate::card_def::CARD_DEFS[source.card_def as usize].name != "Mesmeric Fiend"
                {
                    return Err("linked-exile return has the wrong source contract".to_string());
                }
                let positions = state
                    .engine
                    .linked_exile_records
                    .iter()
                    .enumerate()
                    .filter_map(|(index, record)| (record.source == source).then_some(index))
                    .collect::<Vec<_>>();
                let Some(&position) = positions.first() else {
                    continue;
                };
                if positions.len() != 1 {
                    return Err("linked-exile source owns duplicate records".to_string());
                }
                let record = state.engine.linked_exile_records[position];
                let live = state
                    .objects
                    .try_get(record.exiled)
                    .ok_or("linked-exile record names a missing object")?;
                if live.card_def != record.exiled_card_def
                    || live.owner != record.exiled_owner
                    || live.zone_change_count < record.exiled_zone_change_count
                {
                    return Err("linked-exile record is structurally inconsistent".to_string());
                }
                if live.zone_change_count == record.exiled_zone_change_count {
                    let memberships = state
                        .exile
                        .iter()
                        .filter(|&&candidate| candidate == record.exiled)
                        .count();
                    if live.zone != Zone::Exile || memberships != 1 {
                        return Err(
                            "linked-exile exact incarnation lost its exile membership".to_string()
                        );
                    }
                    event::propose_and_commit(
                        state,
                        event::ProposedEvent::zone_change_preserving_known_identity(
                            record.exiled,
                            Zone::Hand,
                        ),
                    );
                }
                state.engine.linked_exile_records.remove(position);
            }
            EffectOp::PreventDamageFromChosenColorUntilEndOfTurn { player } => {
                let player = continuation.ctx.resolve_player(player, state);
                let legal_colors = vec![
                    ManaColor::W,
                    ManaColor::U,
                    ManaColor::B,
                    ManaColor::R,
                    ManaColor::G,
                ];
                let options = legal_colors
                    .iter()
                    .copied()
                    .map(|color| EffectOp::InstallDamagePreventionFromColor { player, color })
                    .collect();
                let canonical_path = path.clone();
                let expected_remaining_frames = continuation.frames.clone();
                continuation.choice = Some(PendingEffectChoice::ChooseOption {
                    player,
                    path,
                    options,
                    purpose: EffectOptionChoicePurpose::ChooseColor {
                        player,
                        legal_colors,
                        canonical_path,
                        expected_remaining_frames,
                    },
                });
                state.engine.pending_effect = Some(continuation);
                return Ok(ResumableProgress::Suspended);
            }
            EffectOp::MayPayManaThen {
                player,
                colored,
                generic,
                then,
            } => {
                let player = continuation.ctx.resolve_player(player, state);
                if crate::engine::can_pay_effect_mana(player, &colored, generic, state) {
                    continuation.choice = Some(PendingEffectChoice::ChooseBoolean {
                        player,
                        path,
                        default: Some(false),
                        purpose: EffectBooleanChoicePurpose::PayManaThen {
                            player,
                            colored,
                            generic,
                            then,
                        },
                    });
                    state.engine.pending_effect = Some(continuation);
                    return Ok(ResumableProgress::Suspended);
                }
            }
            EffectOp::UntapUpToLands {
                chooser,
                max_targets,
            } => {
                let chooser = continuation.ctx.resolve_player(chooser, state);
                if !tapped_land_bindings(state).is_empty() {
                    stage_land_untap_choice(&mut continuation, state, chooser, max_targets, path)?;
                    state.engine.pending_effect = Some(continuation);
                    return Ok(ResumableProgress::Suspended);
                }
            }
            EffectOp::DestroyTargetLandThenMaySearchBasicTapped { object } => {
                if object != ObjectRef::Target(0)
                    || !continuation.ctx.target_incarnation_matches(0, state)
                {
                    return Err(
                        "destroy-then-search lost its exact target-land incarnation".to_string()
                    );
                }
                let target = continuation.ctx.resolve_object(object);
                let live = state
                    .objects
                    .try_get(target)
                    .ok_or("destroy-then-search target no longer exists")?;
                if live.zone != Zone::Battlefield
                    || !crate::card_def::CARD_DEFS[live.card_def as usize].has_type(CardType::Land)
                {
                    return Err("destroy-then-search target is not a battlefield land".to_string());
                }
                let player = live.controller;
                execute(
                    &EffectOp::DestroyObject { object },
                    &continuation.ctx,
                    state,
                );
                let canonical_path = path.clone();
                let expected_remaining_frames = continuation.frames.clone();
                continuation.choice = Some(PendingEffectChoice::ChooseBoolean {
                    player,
                    path,
                    default: Some(false),
                    purpose: EffectBooleanChoicePurpose::SearchLibraryToBattlefieldTapped {
                        player,
                        filter: LibraryCardFilter::BasicLand,
                        canonical_path,
                        expected_remaining_frames,
                    },
                });
                state.engine.pending_effect = Some(continuation);
                return Ok(ResumableProgress::Suspended);
            }
            EffectOp::SearchLibraryToBattlefieldTapped { player, filter } => {
                let player = continuation.ctx.resolve_player(player, state);
                let original_library = bind_library_exact(state, player);
                validate_library_search_live_metadata(state, player, filter, &original_library)?;
                let candidates =
                    library_search_candidates(state, player, filter, &original_library)?;
                stage_library_search_to_battlefield_choice(
                    &mut continuation,
                    player,
                    filter,
                    original_library,
                    candidates,
                    path,
                );
                state.engine.pending_effect = Some(continuation);
                return Ok(ResumableProgress::Suspended);
            }
            EffectOp::RevealTargetHandChooseNoncreatureNonlandDiscard { player } => {
                let player = continuation.ctx.resolve_player(player, state);
                if player == continuation.ctx.controller {
                    return Err("Duress must target an opponent".to_string());
                }
                let original_hand = bind_hand(state, player);
                validate_bound_hand_exact(state, player, &original_hand)?;
                for binding in &original_hand {
                    for observer in [PlayerId::P0, PlayerId::P1] {
                        state.reveal_hand_card(observer, player, binding.object)?;
                    }
                }
                let eligible = duress_eligible_hand(state, &original_hand)?;
                if !eligible.is_empty() {
                    stage_duress_discard_choice(
                        &mut continuation,
                        player,
                        original_hand,
                        eligible,
                        path,
                    );
                    state.engine.pending_effect = Some(continuation);
                    return Ok(ResumableProgress::Suspended);
                }
            }
            EffectOp::PutObjectInOwnersLibrarySecondOrBottom { object } => {
                let object = continuation.ctx.resolve_object(object);
                let live = state
                    .objects
                    .try_get(object)
                    .ok_or_else(|| format!("owner-library object {} no longer exists", object.0))?;
                if live.zone != Zone::Battlefield {
                    return Err(
                        "owner-library placement object is no longer on the battlefield"
                            .to_string(),
                    );
                }
                let binding = EffectObjectBinding {
                    object,
                    expected_zone: Zone::Battlefield,
                    expected_zone_change_count: live.zone_change_count,
                };
                let owner = live.owner;
                let canonical_path = path.clone();
                let expected_remaining_frames = continuation.frames.clone();
                continuation.choice = Some(PendingEffectChoice::ChooseOption {
                    player: owner,
                    path,
                    options: vec![
                        EffectOp::PutBoundObjectInOwnersLibrary {
                            object: binding,
                            owner,
                            placement: event::LibraryPlacement::SecondFromTop,
                        },
                        EffectOp::PutBoundObjectInOwnersLibrary {
                            object: binding,
                            owner,
                            placement: event::LibraryPlacement::Bottom,
                        },
                    ],
                    purpose: EffectOptionChoicePurpose::OwnerLibrarySecondOrBottom {
                        object: binding,
                        owner,
                        canonical_path,
                        expected_remaining_frames,
                    },
                });
                state.engine.pending_effect = Some(continuation);
                return Ok(ResumableProgress::Suspended);
            }
            EffectOp::ExploreTarget { object } => {
                let target = continuation.ctx.resolve_object(object);
                let target_index = match object {
                    ObjectRef::Target(index) => usize::from(index),
                    ObjectRef::ThisSource => {
                        return Err("Explore requires an announced creature target".to_string())
                    }
                };
                if !continuation
                    .ctx
                    .target_incarnation_matches(target_index, state)
                    || state.objects.get(target).zone != Zone::Battlefield
                {
                    return Err("Explore target incarnation is no longer valid".to_string());
                }
                let player = continuation.ctx.controller;
                let Some(top) = bind_library_top(state, player, 1).into_iter().next() else {
                    continue;
                };
                state.reveal_library_top(PlayerId::P0, player, 1);
                state.reveal_library_top(PlayerId::P1, player, 1);
                let top_def =
                    &crate::card_def::CARD_DEFS[state.objects.get(top.object).card_def as usize];
                if top_def.has_type(CardType::Land) {
                    event::propose_and_commit(
                        state,
                        event::ProposedEvent::zone_change_preserving_known_identity(
                            top.object,
                            Zone::Hand,
                        ),
                    );
                    continue;
                }
                let counters = &mut state.objects.get_mut(target).counters.plus1_plus1;
                *counters = counters
                    .checked_add(1)
                    .ok_or("Explore +1/+1 counter overflow")?;
                let canonical_path = path.clone();
                continuation.choice = Some(PendingEffectChoice::ChooseOption {
                    player,
                    path,
                    options: vec![
                        EffectOp::Sequence(vec![]),
                        EffectOp::MoveBoundObject {
                            object: top,
                            to_zone: Zone::Graveyard,
                            preserve_known_identity: true,
                        },
                    ],
                    purpose: EffectOptionChoicePurpose::ExploreNonlandTop {
                        player,
                        top,
                        canonical_path,
                    },
                });
                state.engine.pending_effect = Some(continuation);
                return Ok(ResumableProgress::Suspended);
            }
            EffectOp::PutBoundObjectInOwnersLibrary {
                object,
                owner,
                placement,
            } => {
                validate_effect_object_binding(state, object)?;
                let live = state.objects.get(object.object);
                if object.expected_zone != Zone::Battlefield || live.owner != owner {
                    return Err(
                        "bound owner-library object changed its battlefield binding or owner"
                            .to_string(),
                    );
                }
                if !matches!(
                    placement,
                    event::LibraryPlacement::SecondFromTop | event::LibraryPlacement::Bottom
                ) {
                    return Err("bound owner-library placement is not a printed option".to_string());
                }
                event::propose_and_commit(
                    state,
                    event::ProposedEvent::public_library_insert(object.object, placement),
                );
            }
            EffectOp::MoveBoundObject {
                object,
                to_zone,
                preserve_known_identity,
            } => {
                validate_effect_object_binding(state, object)?;
                let proposed = if preserve_known_identity {
                    event::ProposedEvent::zone_change_preserving_known_identity(
                        object.object,
                        to_zone,
                    )
                } else {
                    event::ProposedEvent::zone_change(object.object, to_zone)
                };
                event::propose_and_commit(state, proposed);
            }
            leaf => {
                if matches!(leaf, EffectOp::DiscardCards { .. }) && !continuation.frames.is_empty()
                {
                    return Err("a resumable discard must be the terminal effect leaf".to_string());
                }
                execute(&leaf, &continuation.ctx, state)
            }
        }
    }

    if continuation.answered_choice_guard.is_some() {
        return Err("effect completed with an unconsumed answered-choice guard".to_string());
    }
    Ok(ResumableProgress::Complete(Box::new(
        continuation.resolving_item,
    )))
}

impl ExecCtx {
    pub fn no_targets(source: ObjectId, controller: PlayerId) -> ExecCtx {
        ExecCtx {
            stack_item_id: None,
            source,
            controller,
            targets: Vec::new(),
            target_contracts: Vec::new(),
            discarded: Vec::new(),
            paid_cost_refs: Vec::new(),
            hidden_ability_source: None,
            ability_source_contract: None,
            kicked: false,
            optional_additional_cost_paid: None,
            x_value: 0,
        }
    }

    fn resolve_object(&self, r: ObjectRef) -> ObjectId {
        match r {
            ObjectRef::ThisSource => self.source,
            ObjectRef::Target(i) => match self.targets[i as usize] {
                Target::Object(id) => id,
                Target::Player(_) => panic!("effect expected an object target at index {i}"),
            },
        }
    }

    fn resolve_target(&self, r: TargetRef) -> Target {
        match r {
            TargetRef::ThisSource => Target::Object(self.source),
            TargetRef::Target(i) => self.targets[i as usize],
            TargetRef::Opponent => Target::Player(self.controller.opponent()),
        }
    }

    fn target_incarnation_matches(&self, index: usize, state: &GameState) -> bool {
        match (self.targets.get(index), self.target_contracts.get(index)) {
            (Some(Target::Player(player)), Some(StackTargetContractV4::Player(bound))) => {
                player == bound
            }
            (
                Some(Target::Object(object)),
                Some(StackTargetContractV4::Object {
                    object: bound,
                    card_def,
                    owner,
                    zone_change_count,
                    ..
                }),
            ) => {
                bound == object
                    && state.objects.try_get(*object).is_some_and(|live| {
                        live.card_def == *card_def
                            && live.owner == *owner
                            && live.zone_change_count == *zone_change_count
                    })
            }
            _ => false,
        }
    }

    fn resolve_player(&self, r: PlayerRef, state: &GameState) -> PlayerId {
        match r {
            PlayerRef::Controller => self.controller,
            PlayerRef::Target(i) => match self.targets[i as usize] {
                Target::Player(p) => p,
                Target::Object(_) => panic!("effect expected a player target at index {i}"),
            },
            PlayerRef::ObjectController(oref) => {
                state.objects.get(self.resolve_object(oref)).controller
            }
            PlayerRef::Opponent => self.controller.opponent(),
        }
    }
}

fn stage_graveyard_order_choice(
    continuation: &mut EffectContinuation,
    player: PlayerId,
    path: Vec<u16>,
    objects: Vec<EffectObjectBinding>,
    purpose: EffectTargetSelectionPurpose,
) {
    let count = objects
        .len()
        .try_into()
        .expect("effect target count fits the u16 public contract");
    continuation.choice = Some(PendingEffectChoice::SelectTargets {
        player,
        path,
        selected: Vec::new(),
        legal: objects
            .into_iter()
            .map(|binding| EffectTargetCandidate {
                target: Target::Object(binding.object),
                expected_object: Some(binding),
            })
            .collect(),
        min_targets: count,
        max_targets: count,
        ordered: true,
        purpose,
    });
}

fn stage_library_order_choice(
    continuation: &mut EffectContinuation,
    player: PlayerId,
    path: Vec<u16>,
    original_prefix: Vec<EffectObjectBinding>,
) {
    let count = original_prefix
        .len()
        .try_into()
        .expect("library-order target count fits the u16 public contract");
    continuation.choice = Some(PendingEffectChoice::SelectTargets {
        player,
        path,
        selected: Vec::new(),
        legal: original_prefix
            .iter()
            .copied()
            .map(|binding| EffectTargetCandidate {
                target: Target::Object(binding.object),
                expected_object: Some(binding),
            })
            .collect(),
        min_targets: count,
        max_targets: count,
        ordered: true,
        purpose: EffectTargetSelectionPurpose::OrderLookedLibraryTop {
            player,
            original_prefix,
        },
    });
}

fn stage_hand_to_library_choice(
    continuation: &mut EffectContinuation,
    player: PlayerId,
    total: u8,
    remaining: u8,
    prompt_index: u16,
    canonical_path: Vec<u16>,
    original_hand: Vec<EffectObjectBinding>,
) {
    let continuation_path = canonical_path.clone();
    debug_assert!(validate_hand_to_library_progress(
        total,
        remaining,
        prompt_index,
        &continuation_path,
        &canonical_path,
    )
    .is_ok());
    debug_assert!(remaining > 0);
    debug_assert!(original_hand.len() >= 2);
    let mut choice_path = canonical_path.clone();
    choice_path.push(prompt_index);
    continuation.choice = Some(PendingEffectChoice::SelectTargets {
        player,
        path: choice_path,
        selected: Vec::new(),
        legal: original_hand
            .iter()
            .copied()
            .map(|binding| EffectTargetCandidate {
                target: Target::Object(binding.object),
                expected_object: Some(binding),
            })
            .collect(),
        min_targets: 1,
        max_targets: 1,
        ordered: true,
        purpose: EffectTargetSelectionPurpose::PutHandCardOnLibraryTop {
            player,
            original_hand,
            total,
            remaining,
            prompt_index,
            continuation_path,
            canonical_path,
        },
    });
}

fn stage_scry_choice(
    continuation: &mut EffectContinuation,
    player: PlayerId,
    requested_count: u8,
    original_library_len: u32,
    original_prefix: Vec<EffectObjectBinding>,
    stage: ScrySelectionStage,
    canonical_path: Vec<u16>,
) -> Result<(), String> {
    validate_scry_bound_metadata(requested_count, original_library_len, &original_prefix)?;
    let (candidates, min_targets, max_targets, ordered) = match &stage {
        ScrySelectionStage::ChooseBottomSubset => (
            original_prefix.clone(),
            0,
            u16::try_from(original_prefix.len())
                .map_err(|_| "scry prefix exceeds u16".to_string())?,
            false,
        ),
        ScrySelectionStage::OrderBottom { bottom_subset } => {
            validate_canonical_scry_subset(&original_prefix, bottom_subset)?;
            if bottom_subset.len() < 2 {
                return Err("scry bottom-order prompt has no genuine choice".to_string());
            }
            let count = u16::try_from(bottom_subset.len())
                .map_err(|_| "scry bottom group exceeds u16".to_string())?;
            (bottom_subset.clone(), count, count, true)
        }
        ScrySelectionStage::OrderRetainedTop {
            bottom_subset,
            ordered_bottom,
        } => {
            validate_canonical_scry_subset(&original_prefix, bottom_subset)?;
            validate_exact_binding_permutation(
                bottom_subset,
                ordered_bottom,
                "scry ordered bottom",
            )?;
            let retained = scry_retained_prefix(&original_prefix, bottom_subset)?;
            if retained.len() < 2 {
                return Err("scry retained-top prompt has no genuine choice".to_string());
            }
            let count = u16::try_from(retained.len())
                .map_err(|_| "scry retained group exceeds u16".to_string())?;
            (retained, count, count, true)
        }
    };
    let mut choice_path = canonical_path.clone();
    choice_path.push(scry_stage_tag(&stage));
    let stage_fingerprint = scry_stage_fingerprint(&stage);
    continuation.choice = Some(PendingEffectChoice::SelectTargets {
        player,
        path: choice_path,
        selected: Vec::new(),
        legal: candidates
            .into_iter()
            .map(|binding| EffectTargetCandidate {
                target: Target::Object(binding.object),
                expected_object: Some(binding),
            })
            .collect(),
        min_targets,
        max_targets,
        ordered,
        purpose: EffectTargetSelectionPurpose::ScryLibrary {
            player,
            requested_count,
            original_library_len,
            original_prefix,
            stage,
            stage_fingerprint,
            canonical_path,
        },
    });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn stage_library_partition_choice(
    continuation: &mut EffectContinuation,
    state: &GameState,
    player: PlayerId,
    requested_count: u8,
    original_library_len: u32,
    card_type: CardType,
    original_prefix: Vec<EffectObjectBinding>,
    stage: LibraryPartitionSelectionStage,
    canonical_path: Vec<u16>,
) -> Result<(), String> {
    validate_library_partition_live_metadata(
        state,
        player,
        requested_count,
        original_library_len,
        card_type,
        &original_prefix,
    )?;
    let (candidates, min_targets, max_targets, ordered) = match &stage {
        LibraryPartitionSelectionStage::ChooseMatchingSubset => {
            let matching = library_partition_matching_prefix(state, card_type, &original_prefix)?;
            let count = u16::try_from(matching.len())
                .map_err(|_| "library-partition matching set exceeds u16".to_string())?;
            (matching, 0, count, false)
        }
        LibraryPartitionSelectionStage::OrderRest { selected } => {
            validate_canonical_binding_subset(&original_prefix, selected)?;
            let matching = library_partition_matching_prefix(state, card_type, &original_prefix)?;
            if selected.iter().any(|binding| !matching.contains(binding)) {
                return Err(
                    "library-partition selected card does not match the typed filter".to_string(),
                );
            }
            let rest = binding_partition_rest(&original_prefix, selected)?;
            if rest.len() < 2 {
                return Err("library-partition rest-order prompt has no genuine choice".to_string());
            }
            let count = u16::try_from(rest.len())
                .map_err(|_| "library-partition rest set exceeds u16".to_string())?;
            (rest, count, count, true)
        }
    };
    let mut choice_path = canonical_path.clone();
    choice_path.push(library_partition_stage_tag(&stage));
    let stage_fingerprint = library_partition_stage_fingerprint(&stage);
    continuation.choice = Some(PendingEffectChoice::SelectTargets {
        player,
        path: choice_path,
        selected: Vec::new(),
        legal: candidates
            .into_iter()
            .map(|binding| EffectTargetCandidate {
                target: Target::Object(binding.object),
                expected_object: Some(binding),
            })
            .collect(),
        min_targets,
        max_targets,
        ordered,
        purpose: EffectTargetSelectionPurpose::LookTopSelectByTypeToHandBottomRest {
            player,
            requested_count,
            original_library_len,
            card_type,
            original_prefix,
            stage,
            stage_fingerprint,
            canonical_path,
        },
    });
    Ok(())
}

fn stage_library_search_choice(
    continuation: &mut EffectContinuation,
    player: PlayerId,
    filter: LibraryCardFilter,
    original_library: Vec<EffectObjectBinding>,
    candidates: Vec<EffectObjectBinding>,
    canonical_path: Vec<u16>,
) {
    continuation.choice = Some(PendingEffectChoice::SelectTargets {
        player,
        path: canonical_path.clone(),
        selected: Vec::new(),
        legal: candidates
            .into_iter()
            .map(|binding| EffectTargetCandidate {
                target: Target::Object(binding.object),
                expected_object: Some(binding),
            })
            .collect(),
        min_targets: 0,
        max_targets: 1,
        ordered: false,
        purpose: EffectTargetSelectionPurpose::SearchLibraryToHand {
            player,
            filter,
            filter_fingerprint: library_filter_fingerprint(filter),
            original_library,
            canonical_path,
        },
    });
}

fn stage_library_search_to_battlefield_choice(
    continuation: &mut EffectContinuation,
    player: PlayerId,
    filter: LibraryCardFilter,
    original_library: Vec<EffectObjectBinding>,
    candidates: Vec<EffectObjectBinding>,
    canonical_path: Vec<u16>,
) {
    continuation.choice = Some(PendingEffectChoice::SelectTargets {
        player,
        path: canonical_path.clone(),
        selected: Vec::new(),
        legal: candidates
            .into_iter()
            .map(|binding| EffectTargetCandidate {
                target: Target::Object(binding.object),
                expected_object: Some(binding),
            })
            .collect(),
        min_targets: 0,
        max_targets: 1,
        ordered: false,
        purpose: EffectTargetSelectionPurpose::SearchLibraryToBattlefieldTapped {
            player,
            filter,
            filter_fingerprint: library_filter_fingerprint(filter),
            original_library,
            canonical_path,
        },
    });
}

fn stage_duress_discard_choice(
    continuation: &mut EffectContinuation,
    player: PlayerId,
    original_hand: Vec<EffectObjectBinding>,
    eligible: Vec<EffectObjectBinding>,
    canonical_path: Vec<u16>,
) {
    continuation.choice = Some(PendingEffectChoice::SelectTargets {
        player: continuation.ctx.controller,
        path: canonical_path.clone(),
        selected: Vec::new(),
        legal: eligible
            .iter()
            .copied()
            .map(|binding| EffectTargetCandidate {
                target: Target::Object(binding.object),
                expected_object: Some(binding),
            })
            .collect(),
        min_targets: 1,
        max_targets: 1,
        ordered: false,
        purpose: EffectTargetSelectionPurpose::DuressDiscard {
            player,
            original_hand,
            eligible,
            canonical_path,
        },
    });
}

fn stage_library_search_many_choice(
    continuation: &mut EffectContinuation,
    player: PlayerId,
    filter: LibraryCardFilter,
    original_library: Vec<EffectObjectBinding>,
    candidates: Vec<EffectObjectBinding>,
    max_targets: u16,
    canonical_path: Vec<u16>,
) {
    continuation.choice = Some(PendingEffectChoice::SelectTargets {
        player,
        path: canonical_path.clone(),
        selected: Vec::new(),
        legal: candidates
            .into_iter()
            .map(|binding| EffectTargetCandidate {
                target: Target::Object(binding.object),
                expected_object: Some(binding),
            })
            .collect(),
        min_targets: 0,
        max_targets,
        ordered: false,
        purpose: EffectTargetSelectionPurpose::SearchLibraryToHandMany {
            player,
            filter,
            filter_fingerprint: library_filter_fingerprint(filter),
            original_library,
            max_targets,
            canonical_path,
        },
    });
}

fn stage_graveyard_exile_choice(
    continuation: &mut EffectContinuation,
    player: PlayerId,
    original_graveyard: Vec<EffectObjectBinding>,
    canonical_path: Vec<u16>,
) {
    debug_assert!(original_graveyard.len() >= 2);
    continuation.choice = Some(PendingEffectChoice::SelectTargets {
        player,
        path: canonical_path.clone(),
        selected: Vec::new(),
        legal: original_graveyard
            .iter()
            .copied()
            .map(|binding| EffectTargetCandidate {
                target: Target::Object(binding.object),
                expected_object: Some(binding),
            })
            .collect(),
        min_targets: 1,
        max_targets: 1,
        ordered: true,
        purpose: EffectTargetSelectionPurpose::ExileOneFromGraveyard {
            player,
            original_graveyard,
            canonical_path,
        },
    });
}

#[allow(clippy::too_many_arguments)]
fn stage_matching_graveyard_exile_choice(
    continuation: &mut EffectContinuation,
    player: PlayerId,
    card_type: CardType,
    original_graveyard: Vec<EffectObjectBinding>,
    candidates: Vec<EffectObjectBinding>,
    then: Box<EffectOp>,
    selection_path: Vec<u16>,
    canonical_path: Vec<u16>,
) {
    debug_assert!(candidates.len() >= 2);
    continuation.choice = Some(PendingEffectChoice::SelectTargets {
        player,
        path: selection_path,
        selected: Vec::new(),
        legal: candidates
            .iter()
            .copied()
            .map(|binding| EffectTargetCandidate {
                target: Target::Object(binding.object),
                expected_object: Some(binding),
            })
            .collect(),
        min_targets: 1,
        max_targets: 1,
        ordered: true,
        purpose: EffectTargetSelectionPurpose::ExileOneMatchingFromGraveyard {
            player,
            card_type,
            original_graveyard,
            candidates,
            then,
            canonical_path,
        },
    });
}

fn stage_creature_sacrifice_choice(
    continuation: &mut EffectContinuation,
    player: PlayerId,
    filter: CreatureSacrificeFilter,
    original_candidates: Vec<EffectObjectBinding>,
    canonical_path: Vec<u16>,
) {
    debug_assert!(original_candidates.len() >= 2);
    continuation.choice = Some(PendingEffectChoice::SelectTargets {
        player,
        path: canonical_path.clone(),
        selected: Vec::new(),
        legal: original_candidates
            .iter()
            .copied()
            .map(|binding| EffectTargetCandidate {
                target: Target::Object(binding.object),
                expected_object: Some(binding),
            })
            .collect(),
        min_targets: 1,
        max_targets: 1,
        ordered: true,
        purpose: EffectTargetSelectionPurpose::SacrificeCreature {
            player,
            filter,
            original_candidates,
            canonical_path,
        },
    });
}

fn stage_linked_exile_hand_choice(
    continuation: &mut EffectContinuation,
    chooser: PlayerId,
    player: PlayerId,
    original_hand: Vec<EffectObjectBinding>,
    candidates: Vec<EffectObjectBinding>,
    source: AbilitySourceContractV4,
    canonical_path: Vec<u16>,
) {
    debug_assert!(candidates.len() >= 2);
    continuation.choice = Some(PendingEffectChoice::SelectTargets {
        player: chooser,
        path: canonical_path.clone(),
        selected: Vec::new(),
        legal: candidates
            .into_iter()
            .map(|binding| EffectTargetCandidate {
                target: Target::Object(binding.object),
                expected_object: Some(binding),
            })
            .collect(),
        min_targets: 1,
        max_targets: 1,
        ordered: true,
        purpose: EffectTargetSelectionPurpose::LinkedExileNonlandFromRevealedHand {
            player,
            original_hand,
            source,
            canonical_path,
        },
    });
}

fn tapped_land_bindings(state: &GameState) -> Vec<EffectObjectBinding> {
    state
        .objects
        .iter()
        .filter_map(|(object, live)| {
            (live.zone == Zone::Battlefield
                && live.tapped
                && crate::card_def::CARD_DEFS[live.card_def as usize].has_type(CardType::Land))
            .then_some(EffectObjectBinding {
                object,
                expected_zone: Zone::Battlefield,
                expected_zone_change_count: live.zone_change_count,
            })
        })
        .collect()
}

fn stage_land_untap_choice(
    continuation: &mut EffectContinuation,
    state: &GameState,
    player: PlayerId,
    max_targets: u16,
    canonical_path: Vec<u16>,
) -> Result<(), String> {
    if max_targets == 0 {
        return Err("land-untap selection requires a positive maximum".to_string());
    }
    let original_candidates = tapped_land_bindings(state);
    let max_targets = max_targets.min(
        original_candidates
            .len()
            .try_into()
            .map_err(|_| "land-untap candidate count exceeds u16".to_string())?,
    );
    continuation.choice = Some(PendingEffectChoice::SelectTargets {
        player,
        path: canonical_path.clone(),
        selected: Vec::new(),
        legal: original_candidates
            .iter()
            .copied()
            .map(|binding| EffectTargetCandidate {
                target: Target::Object(binding.object),
                expected_object: Some(binding),
            })
            .collect(),
        min_targets: 0,
        max_targets,
        ordered: false,
        purpose: EffectTargetSelectionPurpose::UntapLands {
            chooser: player,
            max_targets,
            original_candidates,
            canonical_path,
        },
    });
    Ok(())
}

fn bind_library_exact(state: &GameState, player: PlayerId) -> Vec<EffectObjectBinding> {
    state.players[player.index()]
        .library
        .iter()
        .copied()
        .map(|object| EffectObjectBinding {
            object,
            expected_zone: Zone::Library,
            expected_zone_change_count: state.objects.get(object).zone_change_count,
        })
        .collect()
}

fn bind_graveyard_exact(state: &GameState, player: PlayerId) -> Vec<EffectObjectBinding> {
    state.players[player.index()]
        .graveyard
        .iter()
        .copied()
        .map(|object| EffectObjectBinding {
            object,
            expected_zone: Zone::Graveyard,
            expected_zone_change_count: state.objects.get(object).zone_change_count,
        })
        .collect()
}

fn matching_graveyard_bindings(
    state: &GameState,
    original_graveyard: &[EffectObjectBinding],
    card_type: CardType,
) -> Result<Vec<EffectObjectBinding>, String> {
    original_graveyard
        .iter()
        .copied()
        .filter_map(|binding| {
            if let Err(error) = validate_effect_object_binding(state, binding) {
                return Some(Err(error));
            }
            let object = state.objects.get(binding.object);
            if object.v4.face_index != 0 {
                return Some(Err(
                    "graveyard cost effective card types are unavailable for a non-front face"
                        .to_string(),
                ));
            }
            let def = crate::card_def::CARD_DEFS
                .get(object.card_def as usize)
                .ok_or_else(|| "graveyard cost card definition is missing".to_string());
            match def {
                Ok(def) if !def.is_token && def.has_type(card_type) => Some(Ok(binding)),
                Ok(_) => None,
                Err(error) => Some(Err(error)),
            }
        })
        .collect()
}

fn creature_sacrifice_bindings(
    state: &GameState,
    player: PlayerId,
    filter: CreatureSacrificeFilter,
) -> Result<Vec<EffectObjectBinding>, String> {
    let mut bindings = Vec::new();
    for &object_id in &state.players[player.index()].battlefield {
        let object = state.objects.get(object_id);
        if object.controller != player || object.v4.face_index != 0 {
            continue;
        }
        let def = crate::card_def::CARD_DEFS
            .get(object.card_def as usize)
            .ok_or_else(|| "creature-sacrifice card definition is missing".to_string())?;
        if def.has_type(CardType::Creature) {
            bindings.push(EffectObjectBinding {
                object: object_id,
                expected_zone: Zone::Battlefield,
                expected_zone_change_count: object.zone_change_count,
            });
        }
    }
    if filter == CreatureSacrificeFilter::GreatestPower {
        let Some(greatest) = bindings
            .iter()
            .map(|binding| crate::engine::effective_power(state, binding.object))
            .max()
        else {
            return Ok(bindings);
        };
        bindings
            .retain(|binding| crate::engine::effective_power(state, binding.object) == greatest);
    }
    Ok(bindings)
}

fn library_filter_matches(
    state: &GameState,
    binding: EffectObjectBinding,
    filter: LibraryCardFilter,
) -> Result<bool, String> {
    validate_effect_object_binding(state, binding)?;
    let object = state.objects.get(binding.object);
    let def = crate::card_def::CARD_DEFS
        .get(object.card_def as usize)
        .ok_or_else(|| "library-search card definition is missing".to_string())?;
    if object.v4.face_index != 0 {
        // ObjectStateV4 currently materializes effective subtypes but not an
        // effective card-type mask. No executable kernel operation changes a
        // card's types today, so base types are current only for face zero.
        // Fail closed on any alternate/dynamic face instead of silently
        // treating its front-face type line as effective. A future type- or
        // face-changing mechanic must add effective type state first.
        return Err(
            "library-search effective card types are unavailable for a non-front face".to_string(),
        );
    }
    let subtype_ids = &object.v4.effective_subtype_ids;
    if subtype_ids.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err("library-search effective subtypes are not sorted and unique".to_string());
    }
    Ok(match filter {
        LibraryCardFilter::LandWithSubtype(subtype) => {
            def.has_type(CardType::Land) && subtype_ids.binary_search(&subtype.stable_id()).is_ok()
        }
        LibraryCardFilter::BasicLand => {
            def.has_type(CardType::Land)
                && def.supertypes.contains(&crate::card_def::Supertype::Basic)
        }
        LibraryCardFilter::BasicLandOrGate => {
            (def.has_type(CardType::Land)
                && def.supertypes.contains(&crate::card_def::Supertype::Basic))
                || subtype_ids
                    .binary_search(&Subtype::Gate.stable_id())
                    .is_ok()
        }
        LibraryCardFilter::CardDefinition(card_def) => object.card_def == card_def,
        LibraryCardFilter::BasicLandWithAnySubtype(subtypes) => {
            def.has_type(CardType::Land)
                && def.supertypes.contains(&crate::card_def::Supertype::Basic)
                && subtypes
                    .iter()
                    .any(|subtype| subtype_ids.binary_search(&subtype.stable_id()).is_ok())
        }
    })
}

fn library_filter_fingerprint(filter: LibraryCardFilter) -> u64 {
    match filter {
        LibraryCardFilter::LandWithSubtype(subtype) => {
            // `Subtype::stable_id` is explicitly append-only for schema v4,
            // unlike a derived Rust hash implementation detail.
            fnv1a_u64(
                fnv1a_u64(0xcbf2_9ce4_8422_2325, 0),
                u64::from(subtype.stable_id()),
            )
        }
        LibraryCardFilter::BasicLand => fnv1a_u64(0xcbf2_9ce4_8422_2325, 1),
        LibraryCardFilter::BasicLandOrGate => fnv1a_u64(0xcbf2_9ce4_8422_2325, 2),
        LibraryCardFilter::CardDefinition(card_def) => {
            fnv1a_u64(fnv1a_u64(0xcbf2_9ce4_8422_2325, 3), u64::from(card_def))
        }
        LibraryCardFilter::BasicLandWithAnySubtype(subtypes) => subtypes
            .iter()
            .fold(fnv1a_u64(0xcbf2_9ce4_8422_2325, 4), |hash, subtype| {
                fnv1a_u64(hash, u64::from(subtype.stable_id()))
            }),
    }
}

fn library_search_candidates(
    state: &GameState,
    player: PlayerId,
    filter: LibraryCardFilter,
    original_library: &[EffectObjectBinding],
) -> Result<Vec<EffectObjectBinding>, String> {
    validate_library_search_live_metadata(state, player, filter, original_library)?;
    let mut matches = original_library
        .iter()
        .copied()
        .enumerate()
        .filter_map(
            |(position, binding)| match library_filter_matches(state, binding, filter) {
                Ok(true) => Some(Ok((position, binding))),
                Ok(false) => None,
                Err(error) => Some(Err(error)),
            },
        )
        .collect::<Result<Vec<_>, String>>()?;
    // Match AIRL's human-facing name order while retaining every physical
    // copy. Library position is the deterministic tie-break for same-name
    // cards, so stable action ids remain object-semantic rather than name-
    // deduplicated or candidate-index-based.
    matches.sort_by(|(left_position, left), (right_position, right)| {
        let left_name =
            crate::card_def::CARD_DEFS[state.objects.get(left.object).card_def as usize].name;
        let right_name =
            crate::card_def::CARD_DEFS[state.objects.get(right.object).card_def as usize].name;
        left_name
            .cmp(right_name)
            .then_with(|| left_position.cmp(right_position))
    });
    Ok(matches.into_iter().map(|(_, binding)| binding).collect())
}

fn validate_library_search_live_metadata(
    state: &GameState,
    player: PlayerId,
    _filter: LibraryCardFilter,
    original_library: &[EffectObjectBinding],
) -> Result<(), String> {
    let library = &state.players[player.index()].library;
    if library.len() != original_library.len() {
        return Err("library-search library length changed while pending".to_string());
    }
    let mut seen = Vec::with_capacity(original_library.len());
    for (position, &binding) in original_library.iter().enumerate() {
        if binding.expected_zone != Zone::Library {
            return Err("library-search binding does not expect the library zone".to_string());
        }
        validate_effect_object_binding(state, binding)?;
        let object = state.objects.get(binding.object);
        if object.owner != player {
            return Err("library-search binding has the wrong library owner".to_string());
        }
        if library[position] != binding.object {
            return Err("library-search library order or identity changed".to_string());
        }
        seen.push(binding.object);
    }
    seen.sort_unstable();
    seen.dedup();
    if seen.len() != original_library.len() {
        return Err("library-search snapshot contains a duplicate physical object".to_string());
    }
    Ok(())
}

fn library_partition_stage_tag(stage: &LibraryPartitionSelectionStage) -> u16 {
    match stage {
        LibraryPartitionSelectionStage::ChooseMatchingSubset => 0,
        LibraryPartitionSelectionStage::OrderRest { .. } => 1,
    }
}

fn library_partition_stage_fingerprint(stage: &LibraryPartitionSelectionStage) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325;
    hash = fnv1a_u64(hash, u64::from(library_partition_stage_tag(stage)));
    match stage {
        LibraryPartitionSelectionStage::ChooseMatchingSubset => hash,
        LibraryPartitionSelectionStage::OrderRest { selected } => fnv1a_bindings(hash, selected),
    }
}

fn library_partition_progress_fingerprint(progress: &LibraryPartitionProgress) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325;
    match progress {
        LibraryPartitionProgress::MatchingSubsetChosen { selected } => {
            hash = fnv1a_u64(hash, 0);
            fnv1a_bindings(hash, selected)
        }
        LibraryPartitionProgress::RestOrderChosen {
            selected,
            ordered_rest,
        } => {
            hash = fnv1a_u64(hash, 1);
            hash = fnv1a_bindings(hash, selected);
            fnv1a_bindings(hash, ordered_rest)
        }
    }
}

fn validate_library_partition_bound_metadata(
    requested_count: u8,
    original_library_len: u32,
    original_prefix: &[EffectObjectBinding],
) -> Result<(), String> {
    let library_len = usize::try_from(original_library_len)
        .map_err(|_| "library-partition original length does not fit usize".to_string())?;
    let expected_prefix_len = usize::from(requested_count).min(library_len);
    if original_prefix.len() != expected_prefix_len {
        return Err(
            "library-partition prefix length disagrees with requested count and original length"
                .to_string(),
        );
    }
    if original_prefix
        .iter()
        .any(|binding| binding.expected_zone != Zone::Library)
    {
        return Err("library-partition binding does not expect the library zone".to_string());
    }
    let mut ids = original_prefix
        .iter()
        .map(|binding| binding.object)
        .collect::<Vec<_>>();
    ids.sort_unstable();
    ids.dedup();
    if ids.len() != original_prefix.len() {
        return Err("library-partition prefix contains a duplicate physical object".to_string());
    }
    Ok(())
}

fn validate_library_partition_live_metadata(
    state: &GameState,
    player: PlayerId,
    requested_count: u8,
    original_library_len: u32,
    card_type: CardType,
    original_prefix: &[EffectObjectBinding],
) -> Result<(), String> {
    validate_library_partition_bound_metadata(
        requested_count,
        original_library_len,
        original_prefix,
    )?;
    if state.players[player.index()].library.len()
        != usize::try_from(original_library_len)
            .map_err(|_| "library-partition original length does not fit usize".to_string())?
    {
        return Err(
            "library-partition library length changed while its private choice was pending"
                .to_string(),
        );
    }
    validate_bound_library_prefix_exact(state, player, original_prefix)?;
    let _ = library_partition_matching_prefix(state, card_type, original_prefix)?;
    Ok(())
}

fn library_partition_matching_prefix(
    state: &GameState,
    card_type: CardType,
    original_prefix: &[EffectObjectBinding],
) -> Result<Vec<EffectObjectBinding>, String> {
    let mut matching = Vec::new();
    for &binding in original_prefix {
        let object = state.objects.try_get(binding.object).ok_or_else(|| {
            format!(
                "library-partition object {} no longer exists",
                binding.object.0
            )
        })?;
        let definition = crate::card_def::CARD_DEFS
            .get(object.card_def as usize)
            .ok_or_else(|| "library-partition card definition is missing".to_string())?;
        if definition.has_type(card_type) {
            matching.push(binding);
        }
    }
    Ok(matching)
}

fn canonicalize_binding_subset(
    original: &[EffectObjectBinding],
    selected: &[EffectObjectBinding],
) -> Result<Vec<EffectObjectBinding>, String> {
    let mut selected_sorted = selected.to_vec();
    selected_sorted.sort_by_key(|binding| binding.object);
    selected_sorted.dedup();
    if selected_sorted.len() != selected.len()
        || selected_sorted
            .iter()
            .any(|binding| !original.contains(binding))
    {
        return Err(
            "library-partition selection is not a unique subset of the bound prefix".to_string(),
        );
    }
    Ok(original
        .iter()
        .copied()
        .filter(|binding| selected.contains(binding))
        .collect())
}

fn validate_canonical_binding_subset(
    original: &[EffectObjectBinding],
    selected: &[EffectObjectBinding],
) -> Result<(), String> {
    if canonicalize_binding_subset(original, selected)? != selected {
        return Err("library-partition subset is not in canonical prefix order".to_string());
    }
    Ok(())
}

fn binding_partition_rest(
    original: &[EffectObjectBinding],
    selected: &[EffectObjectBinding],
) -> Result<Vec<EffectObjectBinding>, String> {
    validate_canonical_binding_subset(original, selected)?;
    Ok(original
        .iter()
        .copied()
        .filter(|binding| !selected.contains(binding))
        .collect())
}

fn scry_stage_tag(stage: &ScrySelectionStage) -> u16 {
    match stage {
        ScrySelectionStage::ChooseBottomSubset => 0,
        ScrySelectionStage::OrderBottom { .. } => 1,
        ScrySelectionStage::OrderRetainedTop { .. } => 2,
    }
}

fn scry_stage_fingerprint(stage: &ScrySelectionStage) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325;
    hash = fnv1a_u64(hash, u64::from(scry_stage_tag(stage)));
    match stage {
        ScrySelectionStage::ChooseBottomSubset => hash,
        ScrySelectionStage::OrderBottom { bottom_subset } => fnv1a_bindings(hash, bottom_subset),
        ScrySelectionStage::OrderRetainedTop {
            bottom_subset,
            ordered_bottom,
        } => {
            hash = fnv1a_bindings(hash, bottom_subset);
            fnv1a_bindings(hash, ordered_bottom)
        }
    }
}

fn scry_progress_fingerprint(progress: &ScryProgress) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325;
    match progress {
        ScryProgress::BottomSubsetChosen { bottom_subset } => {
            hash = fnv1a_u64(hash, 0);
            fnv1a_bindings(hash, bottom_subset)
        }
        ScryProgress::BottomOrderChosen {
            bottom_subset,
            ordered_bottom,
        } => {
            hash = fnv1a_u64(hash, 1);
            hash = fnv1a_bindings(hash, bottom_subset);
            fnv1a_bindings(hash, ordered_bottom)
        }
        ScryProgress::TopOrderChosen {
            bottom_subset,
            ordered_bottom,
            ordered_top,
        } => {
            hash = fnv1a_u64(hash, 2);
            hash = fnv1a_bindings(hash, bottom_subset);
            hash = fnv1a_bindings(hash, ordered_bottom);
            fnv1a_bindings(hash, ordered_top)
        }
    }
}

fn fnv1a_bindings(mut hash: u64, bindings: &[EffectObjectBinding]) -> u64 {
    hash = fnv1a_u64(hash, bindings.len() as u64);
    for binding in bindings {
        hash = fnv1a_u64(hash, u64::from(binding.object.0));
        hash = fnv1a_u64(hash, zone_fingerprint(binding.expected_zone));
        hash = fnv1a_u64(hash, u64::from(binding.expected_zone_change_count));
    }
    hash
}

fn fnv1a_u64(mut hash: u64, value: u64) -> u64 {
    for byte in value.to_le_bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn zone_fingerprint(zone: Zone) -> u64 {
    match zone {
        Zone::Library => 0,
        Zone::Hand => 1,
        Zone::Battlefield => 2,
        Zone::Graveyard => 3,
        Zone::Stack => 4,
        Zone::Exile => 5,
        Zone::Command => 6,
    }
}

fn validate_scry_static_metadata(
    requested_count: u8,
    original_library_len: u32,
) -> Result<usize, String> {
    if requested_count > 2 {
        return Err(
            "scry counts above two are outside the certified Preordain contract".to_string(),
        );
    }
    let library_len = usize::try_from(original_library_len)
        .map_err(|_| "scry original library length does not fit usize".to_string())?;
    Ok(usize::from(requested_count).min(library_len))
}

fn validate_scry_bound_metadata(
    requested_count: u8,
    original_library_len: u32,
    original_prefix: &[EffectObjectBinding],
) -> Result<(), String> {
    let expected_prefix_len = validate_scry_static_metadata(requested_count, original_library_len)?;
    if original_prefix.len() != expected_prefix_len {
        return Err(
            "scry-bound prefix length disagrees with requested count and original library length"
                .to_string(),
        );
    }
    if original_prefix
        .iter()
        .any(|binding| binding.expected_zone != Zone::Library)
    {
        return Err("scry binding does not expect the library zone".to_string());
    }
    let mut ids = original_prefix
        .iter()
        .map(|binding| binding.object)
        .collect::<Vec<_>>();
    ids.sort_unstable();
    ids.dedup();
    if ids.len() != original_prefix.len() {
        return Err("scry-bound prefix contains a duplicate physical object".to_string());
    }
    Ok(())
}

fn validate_scry_live_metadata(
    state: &GameState,
    player: PlayerId,
    requested_count: u8,
    original_library_len: u32,
    original_prefix: &[EffectObjectBinding],
) -> Result<(), String> {
    validate_scry_bound_metadata(requested_count, original_library_len, original_prefix)?;
    if state.players[player.index()].library.len()
        != usize::try_from(original_library_len)
            .map_err(|_| "scry original library length does not fit usize".to_string())?
    {
        return Err("scry library length changed while its private choice was pending".to_string());
    }
    validate_bound_library_prefix_exact(state, player, original_prefix)
}

fn validate_exact_binding_permutation(
    expected: &[EffectObjectBinding],
    actual: &[EffectObjectBinding],
    label: &str,
) -> Result<(), String> {
    let mut expected = expected.to_vec();
    let mut actual = actual.to_vec();
    expected.sort_by_key(|binding| binding.object);
    actual.sort_by_key(|binding| binding.object);
    if actual != expected {
        return Err(format!("{label} is not an exact bound-object permutation"));
    }
    Ok(())
}

fn canonicalize_scry_subset(
    original_prefix: &[EffectObjectBinding],
    selected: &[EffectObjectBinding],
) -> Result<Vec<EffectObjectBinding>, String> {
    let mut selected_sorted = selected.to_vec();
    selected_sorted.sort_by_key(|binding| binding.object);
    selected_sorted.dedup();
    if selected_sorted.len() != selected.len()
        || selected_sorted
            .iter()
            .any(|binding| !original_prefix.contains(binding))
    {
        return Err("scry bottom selection is not a unique subset of the bound prefix".to_string());
    }
    Ok(original_prefix
        .iter()
        .copied()
        .filter(|binding| selected.contains(binding))
        .collect())
}

fn validate_canonical_scry_subset(
    original_prefix: &[EffectObjectBinding],
    bottom_subset: &[EffectObjectBinding],
) -> Result<(), String> {
    if canonicalize_scry_subset(original_prefix, bottom_subset)? != bottom_subset {
        return Err("scry bottom subset is not in canonical prefix order".to_string());
    }
    Ok(())
}

fn scry_retained_prefix(
    original_prefix: &[EffectObjectBinding],
    bottom_subset: &[EffectObjectBinding],
) -> Result<Vec<EffectObjectBinding>, String> {
    validate_canonical_scry_subset(original_prefix, bottom_subset)?;
    Ok(original_prefix
        .iter()
        .copied()
        .filter(|binding| !bottom_subset.contains(binding))
        .collect())
}

/// Validates the redundant progress/path metadata carried by Brainstorm-style
/// repeated hand-to-library coordinators. The copied total and structural path
/// catch stale or internally inconsistent trusted snapshots; they do not
/// authenticate a `GameState` whose related private fields were coherently
/// rewritten outside the opaque in-process `Snapshot` API.
fn validate_hand_to_library_progress(
    total: u8,
    remaining: u8,
    prompt_index: u16,
    path: &[u16],
    canonical_path: &[u16],
) -> Result<(), String> {
    if path != canonical_path {
        return Err("hand-to-library coordinator path changed from its canonical path".to_string());
    }
    if remaining > total {
        return Err("hand-to-library remaining count exceeds its canonical total".to_string());
    }
    if remaining > 0 && total == 0 {
        return Err("active hand-to-library progress has a zero canonical total".to_string());
    }
    let expected_prompt_index = u16::from(total - remaining);
    if prompt_index != expected_prompt_index {
        return Err("hand-to-library prompt index disagrees with canonical progress".to_string());
    }
    Ok(())
}

fn bind_hand(state: &GameState, player: PlayerId) -> Vec<EffectObjectBinding> {
    state.players[player.index()]
        .hand
        .iter()
        .copied()
        .map(|object| EffectObjectBinding {
            object,
            expected_zone: Zone::Hand,
            expected_zone_change_count: state.objects.get(object).zone_change_count,
        })
        .collect()
}

fn bind_library_top(state: &GameState, player: PlayerId, count: u8) -> Vec<EffectObjectBinding> {
    state.players[player.index()].library
        [..usize::from(count).min(state.players[player.index()].library.len())]
        .iter()
        .copied()
        .map(|object| EffectObjectBinding {
            object,
            expected_zone: Zone::Library,
            expected_zone_change_count: state.objects.get(object).zone_change_count,
        })
        .collect()
}

fn bind_library_through_first_type(
    state: &GameState,
    player: PlayerId,
    card_type: CardType,
) -> Vec<EffectObjectBinding> {
    let mut prefix = Vec::new();
    for &object in &state.players[player.index()].library {
        let live = state.objects.get(object);
        prefix.push(EffectObjectBinding {
            object,
            expected_zone: Zone::Library,
            expected_zone_change_count: live.zone_change_count,
        });
        if crate::card_def::CARD_DEFS[live.card_def as usize].has_type(card_type) {
            break;
        }
    }
    prefix
}

fn validate_bound_hand_exact(
    state: &GameState,
    player: PlayerId,
    objects: &[EffectObjectBinding],
) -> Result<(), String> {
    if objects
        .iter()
        .any(|binding| binding.expected_zone != Zone::Hand)
    {
        return Err("hand-to-library binding does not expect the hand zone".to_string());
    }
    for &binding in objects {
        validate_effect_object_binding(state, binding)?;
        if state.objects.get(binding.object).owner != player {
            return Err("hand-to-library binding has the wrong hand owner".to_string());
        }
    }
    let mut expected = objects
        .iter()
        .map(|binding| binding.object)
        .collect::<Vec<_>>();
    let mut current = state.players[player.index()].hand.clone();
    expected.sort_unstable();
    current.sort_unstable();
    if expected != current {
        return Err("bound hand changed identity or membership".to_string());
    }
    Ok(())
}

fn duress_eligible_hand(
    state: &GameState,
    original_hand: &[EffectObjectBinding],
) -> Result<Vec<EffectObjectBinding>, String> {
    let mut eligible = Vec::new();
    for &binding in original_hand {
        validate_effect_object_binding(state, binding)?;
        let object = state
            .objects
            .try_get(binding.object)
            .ok_or("Duress hand object is missing")?;
        let definition = crate::card_def::CARD_DEFS
            .get(object.card_def as usize)
            .ok_or("Duress hand card definition is missing")?;
        if !definition.has_type(CardType::Creature) && !definition.has_type(CardType::Land) {
            eligible.push(binding);
        }
    }
    Ok(eligible)
}

fn validate_bound_graveyard_exact(
    state: &GameState,
    player: PlayerId,
    objects: &[EffectObjectBinding],
) -> Result<(), String> {
    if objects
        .iter()
        .any(|binding| binding.expected_zone != Zone::Graveyard)
    {
        return Err("graveyard binding does not expect the graveyard zone".to_string());
    }
    for &binding in objects {
        validate_effect_object_binding(state, binding)?;
        if state.objects.get(binding.object).owner != player {
            return Err("graveyard binding has the wrong owner".to_string());
        }
    }
    let current = &state.players[player.index()].graveyard;
    let expected = objects
        .iter()
        .map(|binding| binding.object)
        .collect::<Vec<_>>();
    if current != &expected {
        return Err("bound graveyard changed order, identity, or membership".to_string());
    }
    Ok(())
}

/// Validates both object incarnation and membership in the exact current
/// library prefix. Zone-change generations alone cannot detect a shuffle or
/// reorder, so a restored mill continuation must also prove that its bound
/// set is still precisely the top N cards it originally snapshotted.
fn validate_bound_library_prefix(
    state: &GameState,
    objects: &[EffectObjectBinding],
) -> Result<(), String> {
    let Some(first) = objects.first() else {
        return Ok(());
    };
    if objects
        .iter()
        .any(|binding| binding.expected_zone != Zone::Library)
    {
        return Err("milled-card binding does not expect the library zone".to_string());
    }
    for &binding in objects {
        validate_effect_object_binding(state, binding)?;
    }
    let owner = state.objects.get(first.object).owner;
    if objects
        .iter()
        .any(|binding| state.objects.get(binding.object).owner != owner)
    {
        return Err("milled-card bindings do not share one library owner".to_string());
    }
    let library = &state.players[owner.index()].library;
    if library.len() < objects.len() {
        return Err("milled-card binding is longer than the live library".to_string());
    }
    let mut expected = objects
        .iter()
        .map(|binding| binding.object)
        .collect::<Vec<_>>();
    let mut current = library[..objects.len()].to_vec();
    expected.sort_unstable();
    current.sort_unstable();
    if expected != current {
        return Err("milled-card bindings no longer match the live library prefix".to_string());
    }
    Ok(())
}

/// Exact-order counterpart to the mill batch's set validator. Private look
/// choices retain their originally observed prefix separately from the
/// mutable selected/legal partition, so even a same-card-set reorder in a
/// restored/tampered state fails closed before accepting another action.
fn validate_bound_library_prefix_exact(
    state: &GameState,
    player: PlayerId,
    objects: &[EffectObjectBinding],
) -> Result<(), String> {
    if objects
        .iter()
        .any(|binding| binding.expected_zone != Zone::Library)
    {
        return Err("library-order binding does not expect the library zone".to_string());
    }
    for &binding in objects {
        validate_effect_object_binding(state, binding)?;
        if state.objects.get(binding.object).owner != player {
            return Err("library-order binding has the wrong library owner".to_string());
        }
    }
    let library = &state.players[player.index()].library;
    if library.len() < objects.len() {
        return Err("library-order binding is longer than the live library".to_string());
    }
    let expected = objects
        .iter()
        .map(|binding| binding.object)
        .collect::<Vec<_>>();
    if library[..objects.len()] != expected {
        return Err("bound library prefix changed order or identity".to_string());
    }
    Ok(())
}

fn reveal_top_and_partition(
    state: &mut GameState,
    player: PlayerId,
    count: u8,
    card_type: CardType,
    matching_to: Zone,
    rest_to: Zone,
) -> (Vec<EffectObjectBinding>, Vec<EffectObjectBinding>) {
    assert!(
        !matches!(matching_to, Zone::Library | Zone::Stack)
            && !matches!(rest_to, Zone::Library | Zone::Stack),
        "library partition destinations must be ordinary nonlibrary card zones"
    );
    let revealed = state.players[player.index()].library
        [..usize::from(count).min(state.players[player.index()].library.len())]
        .to_vec();
    let mut matching = Vec::new();
    let mut rest = Vec::new();
    for object in revealed.iter().copied() {
        let live = state.objects.get(object);
        let binding = EffectObjectBinding {
            object,
            expected_zone: Zone::Library,
            expected_zone_change_count: live.zone_change_count,
        };
        let def = &crate::card_def::CARD_DEFS[live.card_def as usize];
        if def.has_type(card_type) {
            matching.push(binding);
        } else {
            rest.push(binding);
        }
    }

    // "Reveal" is public, unlike a private look. Record the exact prefix
    // for both perspectives before any member leaves and shifts the
    // remaining position facts.
    for observer in [PlayerId::P0, PlayerId::P1] {
        state.reveal_library_top(observer, player, revealed.len());
    }
    (matching, rest)
}

fn validate_effect_object_binding(
    state: &GameState,
    binding: EffectObjectBinding,
) -> Result<(), String> {
    let object = state
        .objects
        .try_get(binding.object)
        .ok_or_else(|| format!("effect object {} no longer exists", binding.object.0))?;
    if object.zone != binding.expected_zone
        || object.zone_change_count != binding.expected_zone_change_count
    {
        return Err(format!(
            "effect object {} changed incarnation: expected {:?}/{} but found {:?}/{}",
            binding.object.0,
            binding.expected_zone,
            binding.expected_zone_change_count,
            object.zone,
            object.zone_change_count
        ));
    }
    Ok(())
}

fn commit_zone_change_batch(
    state: &mut GameState,
    objects: &[EffectObjectBinding],
    to_zone: Zone,
    preserve_known_identity: bool,
) -> Result<(), String> {
    for &binding in objects {
        validate_effect_object_binding(state, binding)?;
    }
    let events = objects
        .iter()
        .map(|binding| {
            if preserve_known_identity {
                event::ProposedEvent::zone_change_preserving_known_identity(binding.object, to_zone)
            } else {
                event::ProposedEvent::zone_change(binding.object, to_zone)
            }
        })
        .collect();
    event::propose_and_commit_batch(state, events);
    Ok(())
}

fn creature_matches_filter(state: &GameState, object: ObjectId, filter: &CreatureFilter) -> bool {
    let live = state.objects.get(object);
    let def = &crate::card_def::CARD_DEFS[live.card_def as usize];
    if live.zone != Zone::Battlefield || !def.has_type(CardType::Creature) {
        return false;
    }
    match filter {
        CreatureFilter::AnyControlled => true,
        CreatureFilter::ControlledWithSubtype(subtype) => def.subtypes.contains(subtype),
        CreatureFilter::WithoutKeyword(keyword) => {
            !crate::engine::has_effective_keyword(state, object, *keyword)
        }
    }
}

pub fn execute(op: &EffectOp, ctx: &ExecCtx, state: &mut GameState) {
    match op {
        EffectOp::Sequence(ops) => {
            for inner in ops {
                execute(inner, ctx, state);
            }
        }
        EffectOp::Conditional { cond, then, else_ } => {
            let taken = eval_cond(cond, ctx, state);
            execute(if taken { then } else { else_ }, ctx, state);
        }
        EffectOp::Choice { options, .. } => {
            if let Some(first) = options.first() {
                execute(first, ctx, state);
            }
        }
        EffectOp::DealDamage { target, amount } => {
            let target = ctx.resolve_target(*target);
            event::propose_and_commit(
                state,
                event::ProposedEvent::damage(ctx.source, target, *amount),
            );
        }
        EffectOp::DealDamageDynamic { target, amount } => {
            let target = ctx.resolve_target(*target);
            let amount = crate::engine::evaluate_dynamic_value(state, *amount, ctx.controller);
            event::propose_and_commit(
                state,
                event::ProposedEvent::damage(ctx.source, target, amount),
            );
        }
        EffectOp::DamageCannotBePreventedThisTurn => {
            let timestamp = crate::engine::next_timestamp(state);
            state.engine.until_end_of_turn.push(
                crate::engine::UntilEndOfTurnEffect::DamageCannotBePrevented {
                    timestamp,
                    duration: crate::engine::EffectDuration::EndOfTurn,
                },
            );
        }
        EffectOp::AddMinusOneMinusOneCounter { object } => {
            let object_id = ctx.resolve_object(*object);
            let target_index = match object {
                ObjectRef::Target(index) => Some(usize::from(*index)),
                ObjectRef::ThisSource => None,
            };
            if target_index.is_some_and(|index| !ctx.target_incarnation_matches(index, state))
                || state.objects.get(object_id).zone != Zone::Battlefield
                || !crate::card_def::CARD_DEFS[state.objects.get(object_id).card_def as usize]
                    .has_type(CardType::Creature)
            {
                return;
            }
            let Some(next) = state
                .objects
                .get(object_id)
                .counters
                .minus1_minus1
                .checked_add(1)
            else {
                state.engine.halted = Some((
                    crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                    ctx.source,
                ));
                return;
            };
            state.objects.get_mut(object_id).counters.minus1_minus1 = next;
        }
        EffectOp::PutPlusOnePlusOneCounter { object } => {
            let object_id = ctx.resolve_object(*object);
            let target_index = match object {
                ObjectRef::Target(index) => Some(usize::from(*index)),
                ObjectRef::ThisSource => None,
            };
            if target_index.is_some_and(|index| !ctx.target_incarnation_matches(index, state))
                || state.objects.get(object_id).zone != Zone::Battlefield
                || !crate::card_def::CARD_DEFS[state.objects.get(object_id).card_def as usize]
                    .has_type(CardType::Creature)
            {
                return;
            }
            let Some(next) = state
                .objects
                .get(object_id)
                .counters
                .plus1_plus1
                .checked_add(1)
            else {
                state.engine.halted = Some((
                    crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                    ctx.source,
                ));
                return;
            };
            state.objects.get_mut(object_id).counters.plus1_plus1 = next;
        }
        EffectOp::BindPlusOnePlusOneCounterToTriggerSource => {
            state.engine.halted = Some((
                crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                ctx.source,
            ));
        }
        EffectOp::MaterializeStormCopies => {
            state.engine.halted = Some((
                crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                ctx.source,
            ));
        }
        EffectOp::CreateStormCopies { binding } => {
            if crate::engine::create_storm_spell_copies(state, ctx, *binding).is_err() {
                state.engine.halted = Some((
                    crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                    ctx.source,
                ));
            }
        }
        EffectOp::PutPlusOnePlusOneCounterOnBoundObject { object } => {
            if validate_effect_object_binding(state, *object).is_err()
                || object.expected_zone != Zone::Battlefield
            {
                return;
            }
            let counters = &mut state.objects.get_mut(object.object).counters.plus1_plus1;
            let Some(next) = counters.checked_add(1) else {
                state.engine.halted = Some((
                    crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                    ctx.source,
                ));
                return;
            };
            *counters = next;
        }
        EffectOp::GainLife { player, amount } => {
            let player = ctx.resolve_player(*player, state);
            event::propose_and_commit(state, event::ProposedEvent::life_gain(player, *amount));
        }
        EffectOp::GainLifeDynamic { player, amount } => {
            let player = ctx.resolve_player(*player, state);
            let amount = crate::engine::evaluate_dynamic_value(state, *amount, ctx.controller);
            event::propose_and_commit(state, event::ProposedEvent::life_gain(player, amount));
        }
        EffectOp::GainLifeEqualToPaidCostManaValue { player } => {
            let player = ctx.resolve_player(*player, state);
            let amount = ctx.paid_cost_refs.iter().try_fold(0_i32, |total, paid| {
                let mana_value = crate::card_def::CARD_DEFS
                    .get(paid.card_def as usize)
                    .map(|def| i32::from(def.mana_value))?;
                total.checked_add(mana_value)
            });
            let Some(amount) = amount else {
                state.engine.halted = Some((
                    crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                    ctx.source,
                ));
                return;
            };
            event::propose_and_commit(state, event::ProposedEvent::life_gain(player, amount));
        }
        EffectOp::LoseLife { player, amount } => {
            let player = ctx.resolve_player(*player, state);
            event::propose_and_commit(state, event::ProposedEvent::life_loss(player, *amount));
        }
        EffectOp::DrawCards { player, count } => {
            let player = ctx.resolve_player(*player, state);
            for _ in 0..*count {
                event::propose_and_commit(state, event::ProposedEvent::draw(player));
            }
        }
        EffectOp::RevealTopAndPartitionByType {
            player,
            count,
            card_type,
            matching_to,
            rest_to,
        } => {
            let player = ctx.resolve_player(*player, state);
            let (matching, rest) =
                reveal_top_and_partition(state, player, *count, *card_type, *matching_to, *rest_to);
            assert!(
                !(*matching_to == Zone::Graveyard && matching.len() >= 2
                    || *rest_to == Zone::Graveyard && rest.len() >= 2),
                "a multi-card graveyard partition must use the resumable interpreter"
            );
            for (objects, destination) in [(&matching, *matching_to), (&rest, *rest_to)] {
                commit_zone_change_batch(state, objects, destination, true)
                    .expect("freshly revealed batch bindings remain valid");
            }
        }
        EffectOp::MoveObject { object, to_zone } => {
            let object = ctx.resolve_object(*object);
            if *to_zone != Zone::Stack {
                match crate::engine::apply_live_stack_spell_departure(state, object, *to_zone) {
                    Ok(true) => return,
                    Ok(false) => {}
                    Err(_) => {
                        state.engine.halted = Some((
                            crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                            object,
                        ));
                        return;
                    }
                }
            }
            event::propose_and_commit(state, event::ProposedEvent::zone_change(object, *to_zone));
        }
        EffectOp::PutSourceOntoBattlefieldAttachedToTarget { target } => {
            let target_index = match target {
                ObjectRef::Target(index) => usize::from(*index),
                ObjectRef::ThisSource => {
                    state.engine.halted = Some((
                        crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                        ctx.source,
                    ));
                    return;
                }
            };
            let target = ctx.resolve_object(*target);
            let source_is_stack = state.objects.get(ctx.source).zone == Zone::Stack;
            let target_is_creature = ctx.target_incarnation_matches(target_index, state)
                && state.objects.get(target).zone == Zone::Battlefield
                && crate::card_def::CARD_DEFS[state.objects.get(target).card_def as usize]
                    .has_type(CardType::Creature);
            if !source_is_stack || !target_is_creature {
                state.engine.halted = Some((
                    crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                    ctx.source,
                ));
                return;
            }
            event::propose_and_commit(
                state,
                event::ProposedEvent::zone_change(ctx.source, Zone::Battlefield),
            );
            let link = ObjectLinkV4 {
                object: target,
                zone_change_count: state.objects.get(target).zone_change_count,
            };
            state.objects.get_mut(ctx.source).v4.attached_to = Some(link);
            if !state.objects.get(target).attachments.contains(&ctx.source) {
                state.objects.get_mut(target).attachments.push(ctx.source);
            }
        }
        EffectOp::TapAttachedCreatureAndDamageControllerByPower => {
            let Some(source_contract) = ctx.ability_source_contract else {
                return;
            };
            let live_source = state.objects.try_get(ctx.source);
            let source_is_same_battlefield_incarnation = live_source.is_some_and(|source| {
                source.zone == Zone::Battlefield
                    && source.zone_change_count == source_contract.zone_change_count
            });
            let link = if source_is_same_battlefield_incarnation {
                live_source.and_then(|source| source.v4.attached_to)
            } else {
                source_contract.attached_to
            };
            let Some(link) = link else {
                return;
            };
            let Some(attached) = state.objects.try_get(link.object) else {
                return;
            };
            if attached.zone != Zone::Battlefield
                || attached.zone_change_count != link.zone_change_count
                || !crate::card_def::CARD_DEFS[attached.card_def as usize]
                    .has_type(CardType::Creature)
                || (source_is_same_battlefield_incarnation
                    && !attached.attachments.contains(&ctx.source))
            {
                return;
            }
            let attached = link.object;
            event::propose_and_commit(state, event::ProposedEvent::tap(attached));
            let amount = crate::engine::effective_power(state, attached).max(0);
            if amount > 0 {
                event::propose_and_commit(
                    state,
                    event::ProposedEvent::damage(attached, Target::Player(ctx.controller), amount),
                );
            }
        }
        EffectOp::BackupTarget { target, keyword } => {
            let target_index = match target {
                ObjectRef::Target(index) => usize::from(*index),
                ObjectRef::ThisSource => usize::MAX,
            };
            let target = ctx.resolve_object(*target);
            if target_index != usize::MAX && !ctx.target_incarnation_matches(target_index, state) {
                return;
            }
            let Some(object) = state.objects.try_get(target) else {
                return;
            };
            if object.zone != Zone::Battlefield {
                return;
            }
            let counters = &mut state.objects.get_mut(target).counters.plus1_plus1;
            let Some(updated) = counters.checked_add(1) else {
                state.engine.halted = Some((
                    crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                    ctx.source,
                ));
                return;
            };
            *counters = updated;
            if target != ctx.source {
                let timestamp = crate::engine::next_timestamp(state);
                state.engine.until_end_of_turn.push(
                    crate::engine::UntilEndOfTurnEffect::ResolvedObjectKeywordEffect {
                        object_id: target,
                        object_zone_change_count: state.objects.get(target).zone_change_count,
                        layer: crate::engine::Layers::ABILITY_ADDING,
                        timestamp,
                        duration: crate::engine::EffectDuration::EndOfTurn,
                        keywords: *keyword,
                    },
                );
            }
        }
        EffectOp::PutSourceOntoBattlefieldTappedAndAttacking => {
            if crate::engine::put_ninjutsu_source_onto_battlefield_attacking(
                state,
                ctx.source,
                ctx.controller,
                ctx.hidden_ability_source,
            )
            .is_err()
            {
                state.engine.halted = Some((
                    crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                    ctx.source,
                ));
            }
        }
        EffectOp::MoveAllTargets { to_zone } => {
            let events = ctx
                .targets
                .iter()
                .enumerate()
                .filter_map(|(index, target)| {
                    let Target::Object(object) = target else {
                        return None;
                    };
                    ctx.target_incarnation_matches(index, state).then(|| {
                        event::ProposedEvent::zone_change_preserving_known_identity(
                            *object, *to_zone,
                        )
                    })
                })
                .collect();
            event::propose_and_commit_batch(state, events);
        }
        EffectOp::DestroyObject { object } => {
            let object = ctx.resolve_object(*object);
            if state.objects.get(object).zone == Zone::Battlefield
                && !crate::engine::has_effective_keyword(
                    state,
                    object,
                    crate::card_def::Keywords::INDESTRUCTIBLE,
                )
            {
                event::propose_and_commit(
                    state,
                    event::ProposedEvent::zone_change(object, Zone::Graveyard),
                );
            }
        }
        EffectOp::DamageEachCreatureWithoutSubtype {
            amount,
            excluded_subtype,
        } => {
            let events = [PlayerId::P0, PlayerId::P1]
                .into_iter()
                .flat_map(|player| state.players[player.index()].battlefield.iter().copied())
                .filter_map(|id| {
                    let object = state.objects.get(id);
                    let def = &crate::card_def::CARD_DEFS[object.card_def as usize];
                    (def.has_type(crate::card_def::CardType::Creature)
                        && crate::engine::effective_subtype_ids(state, id)
                            .binary_search(&excluded_subtype.stable_id())
                            .is_err())
                    .then(|| event::ProposedEvent::damage(ctx.source, Target::Object(id), *amount))
                })
                .collect();
            event::propose_and_commit_batch(state, events);
        }
        EffectOp::ExploreTarget { .. } => {
            panic!("ExploreTarget must run through the resumable interpreter")
        }
        EffectOp::MoveBoundObject {
            object,
            to_zone,
            preserve_known_identity,
        } => {
            if validate_effect_object_binding(state, *object).is_err() {
                state.engine.halted = Some((
                    crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                    ctx.source,
                ));
                return;
            }
            let proposed = if *preserve_known_identity {
                event::ProposedEvent::zone_change_preserving_known_identity(object.object, *to_zone)
            } else {
                event::ProposedEvent::zone_change(object.object, *to_zone)
            };
            event::propose_and_commit(state, proposed);
        }
        EffectOp::TapObject { object } => {
            let object = ctx.resolve_object(*object);
            event::propose_and_commit(state, event::ProposedEvent::tap(object));
        }
        EffectOp::UntapObject { object } => {
            let object = ctx.resolve_object(*object);
            if state.objects.get(object).zone == Zone::Battlefield {
                state.objects.get_mut(object).tapped = false;
            }
        }
        EffectOp::SkipNextUntap { object } => {
            let object = ctx.resolve_object(*object);
            state.objects.get_mut(object).v4.skip_next_untap = true;
        }
        EffectOp::AttachSourceToTarget { object } => {
            let Some(source) = ctx.ability_source_contract else {
                state.engine.halted = Some((
                    crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                    ctx.source,
                ));
                return;
            };
            let ObjectRef::Target(target_index) = object else {
                state.engine.halted = Some((
                    crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                    ctx.source,
                ));
                return;
            };
            let target_index = usize::from(*target_index);
            if source.source != ctx.source || !ctx.target_incarnation_matches(target_index, state) {
                return;
            }
            let target = ctx.resolve_object(*object);
            let target_live = state.objects.get(target);
            let target_zone_change_count = target_live.zone_change_count;
            let target_is_creature = target_live.zone == Zone::Battlefield
                && crate::card_def::CARD_DEFS[target_live.card_def as usize]
                    .has_type(CardType::Creature);
            let source_is_equipment = state.objects.try_get(ctx.source).is_some_and(|live| {
                live.card_def == source.card_def
                    && live.owner == source.owner
                    && live.zone == Zone::Battlefield
                    && live.zone_change_count == source.zone_change_count
                    && crate::card_def::CARD_DEFS[live.card_def as usize]
                        .equipment
                        .is_some()
            });
            if !target_is_creature || !source_is_equipment {
                return;
            }
            if state
                .attach_object_exact(
                    ctx.source,
                    source.zone_change_count,
                    target,
                    target_zone_change_count,
                )
                .is_err()
            {
                state.engine.halted = Some((
                    crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                    ctx.source,
                ));
            }
        }
        EffectOp::AddCountersToTarget {
            target_index,
            optional,
            plus1_plus1,
            lifelink,
            stun,
        } => {
            let index = usize::from(*target_index);
            let Some(Target::Object(object)) = ctx.targets.get(index).copied() else {
                if !*optional {
                    state.engine.halted = Some((
                        crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                        ctx.source,
                    ));
                }
                return;
            };
            if !ctx.target_incarnation_matches(index, state) {
                return;
            }
            let live = state.objects.get(object);
            if live.zone != Zone::Battlefield
                || !crate::card_def::CARD_DEFS[live.card_def as usize].has_type(CardType::Creature)
                || *plus1_plus1 < 0
                || *lifelink < 0
                || *stun < 0
            {
                state.engine.halted = Some((
                    crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                    ctx.source,
                ));
                return;
            }
            let next = (
                live.counters.plus1_plus1.checked_add(*plus1_plus1),
                live.v4.lifelink_keyword_counters.checked_add(*lifelink),
                live.counters.stun.checked_add(*stun),
            );
            let (Some(next_plus), Some(next_lifelink), Some(next_stun)) = next else {
                state.engine.halted = Some((
                    crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                    ctx.source,
                ));
                return;
            };
            let object = state.objects.get_mut(object);
            object.counters.plus1_plus1 = next_plus;
            object.v4.lifelink_keyword_counters = next_lifelink;
            object.counters.stun = next_stun;
        }
        EffectOp::CreateTokenAndAttachSource { token_def } => {
            let Some(token) = crate::card_def::CARD_DEFS.get(*token_def as usize) else {
                state.engine.halted = Some((
                    crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                    ctx.source,
                ));
                return;
            };
            if !token.is_token || !token.has_full_support() {
                state.engine.halted = Some((
                    crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                    ctx.source,
                ));
                return;
            }
            event::propose_and_commit(
                state,
                event::ProposedEvent::create_token(*token_def, ctx.controller),
            );
            let Some(crate::event::CommittedEvent::CreateToken { object, .. }) =
                state.engine.event_log.last().cloned()
            else {
                state.engine.halted = Some((
                    crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                    ctx.source,
                ));
                return;
            };
            let Some(source) = ctx.ability_source_contract else {
                state.engine.halted = Some((
                    crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                    ctx.source,
                ));
                return;
            };
            let source_is_live = state.objects.try_get(ctx.source).is_some_and(|live| {
                live.card_def == source.card_def
                    && live.owner == source.owner
                    && live.zone == Zone::Battlefield
                    && live.zone_change_count == source.zone_change_count
                    && crate::card_def::CARD_DEFS[live.card_def as usize]
                        .equipment
                        .is_some_and(|equipment| equipment.job_select)
            });
            if !source_is_live {
                return;
            }
            let token_generation = state.objects.get(object).zone_change_count;
            if state
                .attach_object_exact(
                    ctx.source,
                    source.zone_change_count,
                    object,
                    token_generation,
                )
                .is_err()
            {
                state.engine.halted = Some((
                    crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                    ctx.source,
                ));
            }
        }
        EffectOp::AddMana { player, colors } => {
            let player = ctx.resolve_player(*player, state);
            event::propose_and_commit(
                state,
                event::ProposedEvent::mana_add(player, colors.clone()),
            );
        }
        EffectOp::DiscardCards { player, count } => {
            let player = ctx.resolve_player(*player, state);
            state.engine.pending_discard = Some(crate::engine::PendingDiscard {
                player,
                count: *count,
                resume: crate::engine::DiscardResume::None,
            });
        }
        EffectOp::CreateToken {
            token_def,
            controller,
        } => {
            let token = crate::card_def::CARD_DEFS
                .get(*token_def as usize)
                .unwrap_or_else(|| panic!("CreateToken references unknown definition {token_def}"));
            assert!(
                token.is_token && token.is_executable() && token.has_full_support(),
                "CreateToken requires a fully supported executable token definition, got {} ({:?}, is_token={})",
                token.name,
                token.capability,
                token.is_token
            );
            let controller = ctx.resolve_player(*controller, state);
            event::propose_and_commit(
                state,
                event::ProposedEvent::create_token(*token_def, controller),
            );
        }
        EffectOp::MayPayCostThen {
            discard,
            sacrifice_lands,
            then,
        } => {
            let discard_payable = *discard > 0
                && state.players[ctx.controller.index()].hand.len() >= *discard as usize;
            let sacrifice_payable = *sacrifice_lands > 0
                && crate::engine::count_controlled_lands(ctx.controller, state)
                    >= *sacrifice_lands as u32;
            if !discard_payable && !sacrifice_payable {
                // Nothing payable: DoIfCostPaid's own `cost.canPay(...)`
                // gate is false too, so the reference never even offers the
                // "may pay?" prompt here -- matches, no-op.
                return;
            }
            state.engine.pending_optional_cost = Some(crate::engine::PendingOptionalCost {
                player: ctx.controller,
                source: ctx.source,
                discard: *discard,
                sacrifice_lands: *sacrifice_lands,
                discard_payable,
                sacrifice_payable,
                then: (**then).clone(),
                // `resolve_top_of_stack` fills this in right after this
                // call returns, if it's resolving this same spell -- see
                // `PendingOptionalCost::spell_resume`'s doc.
                spell_resume: None,
            });
        }
        EffectOp::DamageOpponentAndTheirCreatures { amount } => {
            // 611.2c-adjacent simultaneity: one instance of this effect
            // hits the opponent and every creature they control at once,
            // same pattern `engine::combat_damage_wave` already uses for
            // combat damage -- a single propose/replace/commit batch, not
            // sequential individual commits that could see each other's
            // side effects mid-resolution.
            let opponent = ctx.controller.opponent();
            let mut events = vec![event::ProposedEvent::damage(
                ctx.source,
                Target::Player(opponent),
                *amount,
            )];
            events.extend(
                state.players[opponent.index()]
                    .battlefield
                    .iter()
                    .copied()
                    .filter_map(|id| {
                        let def =
                            &crate::card_def::CARD_DEFS[state.objects.get(id).card_def as usize];
                        def.has_type(crate::card_def::CardType::Creature).then(|| {
                            event::ProposedEvent::damage(ctx.source, Target::Object(id), *amount)
                        })
                    }),
            );
            event::propose_and_commit_batch(state, events);
        }
        EffectOp::DamageAllCreatures { filter, amount } => {
            let events = [PlayerId::P0, PlayerId::P1]
                .into_iter()
                .flat_map(|player| state.players[player.index()].battlefield.iter().copied())
                .filter(|object| creature_matches_filter(state, *object, filter))
                .map(|object| {
                    event::ProposedEvent::damage(ctx.source, Target::Object(object), *amount)
                })
                .collect();
            event::propose_and_commit_batch(state, events);
        }
        EffectOp::ExilePlayersGraveyard { player } => {
            let player = ctx.resolve_player(*player, state);
            let events = state.players[player.index()]
                .graveyard
                .iter()
                .copied()
                .map(|object| event::ProposedEvent::zone_change(object, Zone::Exile))
                .collect();
            event::propose_and_commit_batch(state, events);
        }
        EffectOp::ExileAllGraveyards => {
            let events = [PlayerId::P0, PlayerId::P1]
                .into_iter()
                .flat_map(|player| state.players[player.index()].graveyard.iter().copied())
                .map(|object| event::ProposedEvent::zone_change(object, Zone::Exile))
                .collect();
            event::propose_and_commit_batch(state, events);
        }
        EffectOp::DamageAllTargets { amount } => {
            let events = ctx
                .targets
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, target)| {
                    let Target::Object(object) = target else {
                        return None;
                    };
                    let live = state.objects.try_get(object)?;
                    (ctx.target_incarnation_matches(index, state)
                        && live.zone == Zone::Battlefield
                        && crate::card_def::CARD_DEFS[live.card_def as usize]
                            .has_type(crate::card_def::CardType::Creature))
                    .then(|| event::ProposedEvent::damage(ctx.source, target, *amount))
                })
                .collect();
            event::propose_and_commit_batch(state, events);
        }
        EffectOp::ExileAllArtifactTargets => {
            let events = ctx
                .targets
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, target)| {
                    let Target::Object(object) = target else {
                        return None;
                    };
                    let live = state.objects.try_get(object)?;
                    (ctx.target_incarnation_matches(index, state)
                        && live.zone == Zone::Battlefield
                        && crate::card_def::CARD_DEFS[live.card_def as usize]
                            .has_type(crate::card_def::CardType::Artifact))
                    .then(|| {
                        event::ProposedEvent::zone_change_preserving_known_identity(
                            object,
                            Zone::Exile,
                        )
                    })
                })
                .collect();
            event::propose_and_commit_batch(state, events);
        }
        EffectOp::DealDamageByControlledCreatureCount { target, multiplier } => {
            let amount = state.players[ctx.controller.index()]
                .battlefield
                .iter()
                .filter(|&&object| {
                    crate::card_def::CARD_DEFS[state.objects.get(object).card_def as usize]
                        .has_type(crate::card_def::CardType::Creature)
                })
                .count()
                .try_into()
                .unwrap_or(i32::MAX);
            let amount = amount.saturating_mul(*multiplier);
            event::propose_and_commit(
                state,
                event::ProposedEvent::damage(ctx.source, ctx.resolve_target(*target), amount),
            );
        }
        EffectOp::ExileTargetPlayersGraveyards => {
            let mut players = ctx
                .targets
                .iter()
                .filter_map(|target| match target {
                    Target::Player(player) => Some(*player),
                    Target::Object(_) => None,
                })
                .collect::<Vec<_>>();
            players.sort_unstable();
            players.dedup();
            let events = players
                .into_iter()
                .flat_map(|player| state.players[player.index()].graveyard.iter().copied())
                .map(|object| event::ProposedEvent::zone_change(object, Zone::Exile))
                .collect();
            event::propose_and_commit_batch(state, events);
        }
        EffectOp::PumpControlled {
            filter,
            power,
            toughness,
            grant_haste,
        } => {
            let object_ids: Vec<ObjectId> = state.players[ctx.controller.index()]
                .battlefield
                .iter()
                .copied()
                .filter(|&id| {
                    let def = &crate::card_def::CARD_DEFS[state.objects.get(id).card_def as usize];
                    if !def.has_type(crate::card_def::CardType::Creature) {
                        return false;
                    }
                    creature_matches_filter(state, id, filter)
                })
                .collect();
            if !object_ids.is_empty() {
                let mut layer = crate::engine::Layers::NONE;
                if *power != 0 || *toughness != 0 {
                    layer = layer | crate::engine::Layers::POWER_TOUGHNESS;
                }
                if *grant_haste {
                    layer = layer | crate::engine::Layers::ABILITY_ADDING;
                }
                let timestamp = crate::engine::next_timestamp(state);
                state.engine.until_end_of_turn.push(
                    crate::engine::UntilEndOfTurnEffect::ResolvedSetEffect {
                        object_ids,
                        layer,
                        timestamp,
                        duration: crate::engine::EffectDuration::EndOfTurn,
                        power: *power,
                        toughness: *toughness,
                        grant_haste: *grant_haste,
                    },
                );
            }
        }
        EffectOp::PumpTargetUntilEndOfTurnDynamic {
            target,
            power,
            toughness,
        } => {
            let Target::Object(object) = ctx.resolve_target(*target) else {
                panic!("dynamic target pump requires an object target");
            };
            let power = crate::engine::evaluate_dynamic_value(state, *power, ctx.controller);
            let toughness =
                crate::engine::evaluate_dynamic_value(state, *toughness, ctx.controller);
            if power != 0 || toughness != 0 {
                let timestamp = crate::engine::next_timestamp(state);
                state.engine.until_end_of_turn.push(
                    crate::engine::UntilEndOfTurnEffect::ResolvedObjectEffect {
                        object_id: object,
                        object_zone_change_count: state.objects.get(object).zone_change_count,
                        layer: crate::engine::Layers::POWER_TOUGHNESS,
                        timestamp,
                        duration: crate::engine::EffectDuration::EndOfTurn,
                        power,
                        toughness,
                        grant_haste: false,
                    },
                );
            }
        }
        EffectOp::GrantKeywordTargetUntilEndOfTurn { object, keyword } => {
            let object = ctx.resolve_object(*object);
            if state.objects.get(object).zone == Zone::Battlefield {
                let timestamp = crate::engine::next_timestamp(state);
                state.engine.until_end_of_turn.push(
                    crate::engine::UntilEndOfTurnEffect::ResolvedObjectKeywordEffect {
                        object_id: object,
                        object_zone_change_count: state.objects.get(object).zone_change_count,
                        layer: crate::engine::Layers::ABILITY_ADDING,
                        timestamp,
                        duration: crate::engine::EffectDuration::EndOfTurn,
                        keywords: *keyword,
                    },
                );
            }
        }
        EffectOp::PumpTargetByControlledSubtypeCount { target, subtype } => {
            let target = ctx.resolve_object(*target);
            let target_is_creature = state.objects.try_get(target).is_some_and(|object| {
                object.zone == Zone::Battlefield
                    && crate::card_def::CARD_DEFS[object.card_def as usize]
                        .has_type(crate::card_def::CardType::Creature)
            });
            if target_is_creature {
                let amount = state.players[ctx.controller.index()]
                    .battlefield
                    .iter()
                    .filter(|&&id| {
                        crate::engine::effective_subtype_ids(state, id)
                            .binary_search(&subtype.stable_id())
                            .is_ok()
                    })
                    .count()
                    .try_into()
                    .unwrap_or(i32::MAX);
                if amount != 0 {
                    let timestamp = crate::engine::next_timestamp(state);
                    state.engine.until_end_of_turn.push(
                        crate::engine::UntilEndOfTurnEffect::ResolvedSetEffect {
                            object_ids: vec![target],
                            layer: crate::engine::Layers::POWER_TOUGHNESS,
                            timestamp,
                            duration: crate::engine::EffectDuration::EndOfTurn,
                            power: amount,
                            toughness: amount,
                            grant_haste: false,
                        },
                    );
                }
            }
        }
        EffectOp::ImpulseDraw { count, duration } => {
            for _ in 0..*count {
                let Some(&top) = state.players[ctx.controller.index()].library.first() else {
                    break; // library ran dry partway through -- not a draw, no loss condition
                };
                if std::env::var("REPLAY_DEBUG_IMPULSE").is_ok() {
                    eprintln!(
                        "IMPULSE_DRAW controller={:?} source={:?} exiling id={} name={:?} turn={} round_step={:?} lib_remaining_before={} hand_size={} priority_round={}",
                        ctx.controller,
                        state.objects.get(ctx.source).name,
                        top.0,
                        state.objects.get(top).name,
                        state.turn,
                        state.step,
                        state.players[ctx.controller.index()].library.len(),
                        state.players[ctx.controller.index()].hand.len(),
                        state.engine.priority_round,
                    );
                }
                event::propose_and_commit(
                    state,
                    event::ProposedEvent::zone_change(top, Zone::Exile),
                );
                let expiry = match duration {
                    ImpulseDuration::EndOfTurn => crate::engine::PlayPermissionExpiry::EndOfTurn,
                    ImpulseDuration::UntilOwnersNextTurn => {
                        crate::engine::PlayPermissionExpiry::UntilHoldersNextTurn {
                            holder_turn_started: false,
                        }
                    }
                };
                let def = &crate::card_def::CARD_DEFS[state.objects.get(top).card_def as usize];
                let play_or_cast = if def.is_playable_land() {
                    crate::engine::PlayOrCast::Play
                } else if def.is_castable() {
                    crate::engine::PlayOrCast::Cast
                } else {
                    // Exiling still happened, but an unsupported definition
                    // never receives an executable permission. This is the
                    // runtime half of the fail-closed deck preflight.
                    continue;
                };
                state
                    .engine
                    .exile_play_permissions
                    .push(crate::engine::PlayPermission {
                        object: top,
                        holder: ctx.controller,
                        // Snapshot *after* the exile move above, so this
                        // permission's own creating zone change isn't what
                        // immediately invalidates it -- see `PlayPermission::
                        // zone_change_generation`'s doc.
                        zone_change_generation: state.objects.get(top).zone_change_count,
                        play_or_cast,
                        expiry,
                    });
            }
        }
        EffectOp::OfferAffectedPlayerSpellCopy { affected } => {
            let target = ctx.resolve_target(*affected);
            let decider = match target {
                Target::Player(p) => p,
                Target::Object(id) => state.objects.get(id).controller,
            };
            state.engine.pending_spell_copy = Some(crate::engine::PendingSpellCopy {
                resolving_stack_item: ctx
                    .stack_item_id
                    .expect("spell-copy offers only resolve from a real stack item"),
                resolving_source: ctx.source,
                resolving_source_zone_change_count: state.objects.get(ctx.source).zone_change_count,
                player: decider,
                inherited_target: target,
                inherited_target_contract: ctx
                    .targets
                    .iter()
                    .position(|candidate| *candidate == target)
                    .and_then(|index| ctx.target_contracts.get(index).copied()),
                stage: crate::engine::SpellCopyStage::Payment,
                copy_source: None,
                copy_stack_item: None,
            });
        }
        EffectOp::MillCards { player, count } => {
            let player = ctx.resolve_player(*player, state);
            let objects = bind_library_top(state, player, *count);
            assert!(
                objects.len() < 2,
                "a multi-card mill must use the resumable interpreter"
            );
            commit_zone_change_batch(state, &objects, Zone::Graveyard, false)
                .expect("fresh mill bindings remain valid");
        }
        EffectOp::RevealUntilCardTypeAndMill { player, card_type } => {
            let player = ctx.resolve_player(*player, state);
            let objects = bind_library_through_first_type(state, player, *card_type);
            assert!(
                objects.len() < 2,
                "a multi-card revealed mill must use the resumable interpreter"
            );
            for observer in [PlayerId::P0, PlayerId::P1] {
                state.reveal_library_top(observer, player, objects.len());
            }
            commit_zone_change_batch(state, &objects, Zone::Graveyard, true)
                .expect("freshly revealed library prefix remains valid");
        }
        EffectOp::ShuffleTriggerSourceIntoOwnersLibrary => {
            let Some(source) = ctx.ability_source_contract else {
                state.engine.halted = Some((
                    crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                    ctx.source,
                ));
                return;
            };
            let Some(expected_generation) = source.zone_change_count.checked_add(1) else {
                state.engine.halted = Some((
                    crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                    ctx.source,
                ));
                return;
            };
            let Some(live) = state.objects.try_get(source.source) else {
                return;
            };
            if live.card_def != source.card_def || live.owner != source.owner {
                state.engine.halted = Some((
                    crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                    ctx.source,
                ));
                return;
            }
            if live.zone != Zone::Graveyard || live.zone_change_count != expected_generation {
                return;
            }
            let mut staged = state.clone();
            let Ok(shuffle_token) = staged.preflight_library_shuffle(source.owner) else {
                state.engine.halted = Some((
                    crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                    ctx.source,
                ));
                return;
            };
            event::propose_and_commit(
                &mut staged,
                event::ProposedEvent::public_library_insert(
                    source.source,
                    event::LibraryPlacement::Top,
                ),
            );
            if staged
                .commit_library_shuffle(source.owner, shuffle_token)
                .is_err()
            {
                state.engine.halted = Some((
                    crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                    ctx.source,
                ));
                return;
            }
            *state = staged;
        }
        EffectOp::ExileTargetLinkedToSource { object } => {
            let target_index = match *object {
                ObjectRef::Target(index) => Some(usize::from(index)),
                ObjectRef::ThisSource => None,
            };
            let object = ctx.resolve_object(*object);
            let Some(target_index) = target_index else {
                state.engine.halted = Some((
                    crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                    ctx.source,
                ));
                return;
            };
            if !ctx.target_incarnation_matches(target_index, state)
                || state.objects.get(object).zone != Zone::Battlefield
            {
                return;
            }
            let Some(source_contract) = ctx.ability_source_contract else {
                state.engine.halted = Some((
                    crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                    ctx.source,
                ));
                return;
            };
            let source_is_live = state.objects.try_get(ctx.source).is_some_and(|source| {
                source.card_def == source_contract.card_def
                    && source.owner == source_contract.owner
                    && source.zone == Zone::Battlefield
                    && source.zone_change_count == source_contract.zone_change_count
            });
            let target_card_def = state.objects.get(object).card_def;
            let target_owner = state.objects.get(object).owner;
            event::propose_and_commit(
                state,
                event::ProposedEvent::zone_change(object, Zone::Exile),
            );
            if source_is_live {
                let exiled_zone_change_count = state.objects.get(object).zone_change_count;
                let source = ObjectLinkV4 {
                    object: ctx.source,
                    zone_change_count: source_contract.zone_change_count,
                };
                if state.engine.linked_exile_records.iter().any(|record| {
                    record.source == source_contract
                        || (record.exiled == object
                            && record.exiled_zone_change_count == exiled_zone_change_count)
                }) {
                    state.engine.halted = Some((
                        crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                        ctx.source,
                    ));
                    return;
                }
                state.engine.linked_exile_records.push(LinkedExileRecordV4 {
                    source: source_contract,
                    exiled: object,
                    exiled_card_def: target_card_def,
                    exiled_owner: target_owner,
                    exiled_zone_change_count,
                });
                state.objects.get_mut(object).v4.exiled_by = Some(source);
            }
        }
        EffectOp::ReturnObjectsExiledBySource => {
            let Some(source_contract) = ctx.ability_source_contract else {
                state.engine.halted = Some((
                    crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                    ctx.source,
                ));
                return;
            };
            let same_source_incarnation = |record: &LinkedExileRecordV4| {
                record.source.source == ctx.source
                    && record.source.card_def == source_contract.card_def
                    && record.source.owner == source_contract.owner
                    && record.source.zone == Zone::Battlefield
                    && record.source.zone_change_count == source_contract.zone_change_count
            };
            let matching = state
                .engine
                .linked_exile_records
                .iter()
                .copied()
                .filter(same_source_incarnation)
                .collect::<Vec<_>>();
            let structurally_valid = matching.iter().all(|record| {
                record.source.zone == source_contract.zone
                    && state.objects.try_get(record.exiled).is_some_and(|exiled| {
                        exiled.card_def == record.exiled_card_def
                            && exiled.owner == record.exiled_owner
                            && exiled.zone_change_count >= record.exiled_zone_change_count
                            && (exiled.zone_change_count != record.exiled_zone_change_count
                                || exiled.zone == Zone::Exile)
                    })
            });
            if !structurally_valid {
                state.engine.halted = Some((
                    crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                    ctx.source,
                ));
                return;
            }
            state
                .engine
                .linked_exile_records
                .retain(|record| !same_source_incarnation(record));
            let events = matching
                .iter()
                .filter(|record| {
                    let live = state.objects.get(record.exiled);
                    live.zone == Zone::Exile
                        && live.zone_change_count == record.exiled_zone_change_count
                })
                .map(|record| event::ProposedEvent::zone_change(record.exiled, Zone::Battlefield))
                .collect::<Vec<_>>();
            event::propose_and_commit_batch(state, events);
        }
        EffectOp::InstallDamagePreventionFromColor { player, color } => {
            if *player != ctx.controller
                || event::install_color_damage_prevention(state, ctx.source, *color).is_err()
            {
                state.engine.halted = Some((
                    crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                    ctx.source,
                ));
            }
        }
        EffectOp::TransformSagaSource => {
            let Some(source_contract) = ctx.ability_source_contract else {
                state.engine.halted = Some((
                    crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                    ctx.source,
                ));
                return;
            };
            let Some(source) = state.objects.try_get(ctx.source) else {
                return;
            };
            if source.card_def != source_contract.card_def
                || source.owner != source_contract.owner
                || source.zone_change_count != source_contract.zone_change_count
                || source.zone != Zone::Battlefield
            {
                return;
            }
            let def = &crate::card_def::CARD_DEFS[source.card_def as usize];
            if source.v4.face_index != 0 || def.saga.is_none() || def.transform_face.is_none() {
                state.engine.halted = Some((
                    crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                    ctx.source,
                ));
                return;
            }
            event::propose_and_commit(
                state,
                event::ProposedEvent::zone_change(ctx.source, Zone::Exile),
            );
            event::propose_and_commit(
                state,
                event::ProposedEvent::transformed_battlefield_return(ctx.source, 1, ctx.controller),
            );
        }
        EffectOp::PutSourceOntoBattlefieldWithXPlusOneCounters => {
            if state.objects.get(ctx.source).zone != Zone::Stack {
                state.engine.halted = Some((
                    crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                    ctx.source,
                ));
                return;
            }
            event::propose_and_commit(
                state,
                event::ProposedEvent::zone_change(ctx.source, Zone::Battlefield),
            );
            let Ok(amount) = i16::try_from(ctx.x_value) else {
                state.engine.halted = Some((
                    crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                    ctx.source,
                ));
                return;
            };
            state.objects.get_mut(ctx.source).counters.plus1_plus1 = amount;
        }
        EffectOp::PutSourceOntoBattlefieldAttachedToTargetWithXPlusOneCounters { target } => {
            let ObjectRef::Target(target_index) = target else {
                state.engine.halted = Some((
                    crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                    ctx.source,
                ));
                return;
            };
            let target = ctx.resolve_object(*target);
            if state.objects.get(ctx.source).zone != Zone::Stack
                || !ctx.target_incarnation_matches(usize::from(*target_index), state)
                || state.objects.get(target).zone != Zone::Battlefield
                || !crate::engine::object_has_type(state, target, CardType::Creature)
            {
                state.engine.halted = Some((
                    crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                    ctx.source,
                ));
                return;
            }
            event::propose_and_commit(
                state,
                event::ProposedEvent::zone_change(ctx.source, Zone::Battlefield),
            );
            let Ok(amount) = i16::try_from(ctx.x_value) else {
                state.engine.halted = Some((
                    crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                    ctx.source,
                ));
                return;
            };
            state.objects.get_mut(ctx.source).counters.plus1_plus1 = amount;
            let link = ObjectLinkV4 {
                object: target,
                zone_change_count: state.objects.get(target).zone_change_count,
            };
            state.objects.get_mut(ctx.source).v4.attached_to = Some(link);
            state.objects.get_mut(ctx.source).v4.effective_subtype_ids =
                vec![crate::card_def::Subtype::Aura.stable_id()];
            if !state.objects.get(target).attachments.contains(&ctx.source) {
                state.objects.get_mut(target).attachments.push(ctx.source);
            }
        }
        EffectOp::DealDamageToTargetEqualToChosenCostCreaturePower { target } => {
            let [reference] = ctx.paid_cost_refs.as_slice() else {
                state.engine.halted = Some((
                    crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                    ctx.source,
                ));
                return;
            };
            let Some(last_known_power) = reference.power_lki else {
                state.engine.halted = Some((
                    crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                    ctx.source,
                ));
                return;
            };
            let Some(definition) = crate::card_def::CARD_DEFS.get(reference.card_def as usize)
            else {
                state.engine.halted = Some((
                    crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                    ctx.source,
                ));
                return;
            };
            if !definition.has_type(CardType::Creature) {
                state.engine.halted = Some((
                    crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                    ctx.source,
                ));
                return;
            }
            let live = state.objects.try_get(reference.object);
            let amount = match live {
                Some(live)
                    if live.card_def == reference.card_def
                        && live.owner == reference.owner
                        && live.zone_change_count == reference.zone_change_count
                        && live.zone == reference.zone =>
                {
                    match live.zone {
                        Zone::Battlefield => {
                            crate::engine::effective_power(state, reference.object)
                        }
                        Zone::Hand => definition.power.map(i32::from).unwrap_or(0),
                        _ => last_known_power,
                    }
                }
                Some(live)
                    if live.card_def == reference.card_def
                        && live.owner == reference.owner
                        && live.zone_change_count > reference.zone_change_count =>
                {
                    last_known_power
                }
                _ => {
                    state.engine.halted = Some((
                        crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                        ctx.source,
                    ));
                    return;
                }
            };
            if amount > 0 {
                event::propose_and_commit(
                    state,
                    event::ProposedEvent::damage(ctx.source, ctx.resolve_target(*target), amount),
                );
            }
        }
        EffectOp::TakeInitiative { player } => {
            let player = ctx.resolve_player(*player, state);
            let Some(source) = ctx.ability_source_contract else {
                state.engine.halted = Some((
                    crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                    ctx.source,
                ));
                return;
            };
            if take_initiative(state, player, source).is_err() {
                state.engine.halted = Some((
                    crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                    ctx.source,
                ));
            }
        }
        EffectOp::AddPlusOnePlusOneCounters { object, count } => {
            let object = ctx.resolve_object(*object);
            if *count == 0
                || state.objects.try_get(object).is_none_or(|live| {
                    live.zone != Zone::Battlefield
                        || !crate::engine::object_has_type(state, object, CardType::Creature)
                })
            {
                state.engine.halted = Some((
                    crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                    ctx.source,
                ));
                return;
            }
            let count = i16::from(*count);
            let counters = &mut state.objects.get_mut(object).counters.plus1_plus1;
            let Some(total) = counters.checked_add(count) else {
                state.engine.halted = Some((
                    crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                    ctx.source,
                ));
                return;
            };
            *counters = total;
        }
        EffectOp::GoadTargetUntilSourcesNextTurn { object } => {
            let object = ctx.resolve_object(*object);
            if state.objects.try_get(object).is_none_or(|live| {
                live.zone != Zone::Battlefield
                    || !crate::engine::object_has_type(state, object, CardType::Creature)
            }) {
                state.engine.halted = Some((
                    crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                    ctx.source,
                ));
                return;
            }
            let Ok(expires_at_turn) = until_players_next_turn_expiry(state, ctx.controller) else {
                state.engine.halted = Some((
                    crate::engine::UnsupportedMechanic::InvalidEffectContinuation,
                    ctx.source,
                ));
                return;
            };
            let goads = &mut state.objects.get_mut(object).v4.goaded_by;
            if let Some(existing) = goads
                .iter_mut()
                .find(|existing| existing.player == ctx.controller)
            {
                existing.expires_at_turn = existing.expires_at_turn.max(expires_at_turn);
            } else {
                goads.push(crate::state::GoadStateV4 {
                    player: ctx.controller,
                    expires_at_turn,
                });
            }
            goads.sort_by_key(|goad| (goad.player, goad.expires_at_turn));
        }
        EffectOp::LookAtLibraryTopAndReorder { .. }
        | EffectOp::MayShuffleLibrary { .. }
        | EffectOp::PutCardsFromHandOnLibraryTop { .. }
        | EffectOp::Scry { .. }
        | EffectOp::SearchLibraryToHand { .. }
        | EffectOp::SearchLibraryToHandUpTo { .. }
        | EffectOp::UntapUpToLands { .. }
        | EffectOp::PutObjectInOwnersLibrarySecondOrBottom { .. }
        | EffectOp::PutBoundObjectInOwnersLibrary { .. }
        | EffectOp::CounterUnlessPaysGeneric { .. }
        | EffectOp::CounterTargetUnlessPaysGeneric { .. }
        | EffectOp::LookTopSelectByTypeToHandBottomRest { .. }
        | EffectOp::ExileOneFromPlayersGraveyard { .. }
        | EffectOp::MayExileFromPlayersGraveyardMatchingThen { .. }
        | EffectOp::SacrificeCreature { .. }
        | EffectOp::MayPayManaThen { .. }
        | EffectOp::DestroyTargetLandThenMaySearchBasicTapped { .. }
        | EffectOp::SearchLibraryToBattlefieldTapped { .. }
        | EffectOp::RevealTargetHandChooseNoncreatureNonlandDiscard { .. }
        | EffectOp::RevealHandChooseNonlandToLinkedExile { .. }
        | EffectOp::ReturnLinkedExiledCardToOwnersHand
        | EffectOp::ResolveInitiativeTrigger { .. }
        | EffectOp::EnterUndercityRoom { .. }
        | EffectOp::ResolveUndercityThrone { .. } => {
            panic!("choice-bearing effects must use the resumable interpreter")
        }
        EffectOp::PreventDamageFromChosenColorUntilEndOfTurn { .. } => {
            panic!("color-choice effects must use the resumable interpreter")
        }
    }
}

fn eval_cond(cond: &EffectCond, ctx: &ExecCtx, state: &GameState) -> bool {
    match cond {
        EffectCond::Always => true,
        EffectCond::Never => false,
        EffectCond::DiscardedNonLandForCost => ctx.discarded.iter().any(|&id| {
            let def_idx = state.objects.get(id).card_def;
            !crate::card_def::CARD_DEFS[def_idx as usize].is_land
        }),
        EffectCond::LandfallThisTurn => {
            state.players[ctx.controller.index()].lands_played_this_turn > 0
        }
        EffectCond::TargetInZone(idx, zone)
            if !ctx.target_incarnation_matches(*idx as usize, state) =>
        {
            false
        }
        EffectCond::TargetInZone(idx, zone) => match ctx.targets.get(*idx as usize) {
            Some(Target::Object(id)) if *zone == Zone::Stack => {
                let live_generation = state.objects.get(*id).zone_change_count;
                state.stack.iter().any(|item| {
                    item.kind == crate::state::StackItemKind::Spell
                        && item.source == *id
                        && item
                            .v4
                            .source_contract
                            .is_some_and(|contract| contract.zone_change_count == live_generation)
                })
            }
            Some(Target::Object(id)) => state.objects.get(*id).zone == *zone,
            _ => false,
        },
        EffectCond::TargetIsColor(idx, _)
            if !ctx.target_incarnation_matches(*idx as usize, state) =>
        {
            false
        }
        EffectCond::TargetIsColor(idx, color) => match ctx.targets.get(*idx as usize) {
            Some(Target::Object(id)) => {
                let def_idx = state.objects.get(*id).card_def;
                crate::card_def::CARD_DEFS[def_idx as usize]
                    .colors
                    .contains(color)
            }
            _ => false,
        },
        EffectCond::And(a, b) => eval_cond(a, ctx, state) && eval_cond(b, ctx, state),
        EffectCond::ControlsArtifactCount(n) => {
            let count = state.players[ctx.controller.index()]
                .battlefield
                .iter()
                .filter(|&&id| {
                    let def = &crate::card_def::CARD_DEFS[state.objects.get(id).card_def as usize];
                    def.has_type(crate::card_def::CardType::Artifact)
                })
                .count();
            count >= *n as usize
        }
        EffectCond::ControlsOtherSubtypeCount {
            subtype,
            minimum_count,
        } => {
            let subtype_id = subtype.stable_id();
            let count = state.players[ctx.controller.index()]
                .battlefield
                .iter()
                .copied()
                .filter(|id| *id != ctx.source)
                .filter(|id| {
                    crate::engine::effective_subtype_ids(state, *id)
                        .binary_search(&subtype_id)
                        .is_ok()
                })
                .count();
            count >= usize::from(*minimum_count)
        }
        EffectCond::ControlsAnotherSourceCard => {
            let source_def = state.objects.get(ctx.source).card_def;
            state.players[ctx.controller.index()]
                .battlefield
                .iter()
                .copied()
                .any(|id| id != ctx.source && state.objects.get(id).card_def == source_def)
        }
        EffectCond::WasKicked => ctx.kicked,
        EffectCond::OptionalAdditionalCostPaid(kind) => {
            ctx.optional_additional_cost_paid == Some(*kind)
        }
        EffectCond::ControllerGraveyardHasType(card_type) => state.players[ctx.controller.index()]
            .graveyard
            .iter()
            .any(|&object| {
                let object = state.objects.get(object);
                object.v4.face_index == 0
                    && crate::card_def::CARD_DEFS[object.card_def as usize].has_type(*card_type)
            }),
        EffectCond::OpponentHasCardsInHand => !state.players[ctx.controller.opponent().index()]
            .hand
            .is_empty(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::CommittedEvent;
    use crate::ids::PlayerId;

    fn two_card_libraries() -> GameState {
        GameState::new_from_libraries(&[1, 2], &[3, 4], |c| format!("card-{c}"), 1)
    }

    #[test]
    fn sequence_runs_every_leaf_in_order() {
        let mut state = two_card_libraries();
        let ctx = ExecCtx::no_targets(ObjectId(0), PlayerId::P0);
        let op = EffectOp::Sequence(vec![
            EffectOp::LoseLife {
                player: PlayerRef::Controller,
                amount: 2,
            },
            EffectOp::GainLife {
                player: PlayerRef::Controller,
                amount: 5,
            },
        ]);
        execute(&op, &ctx, &mut state);
        assert_eq!(state.players[0].life, 20 - 2 + 5);
    }

    /// Proves the `Conditional` composition primitive works end-to-end even
    /// though no card in this increment's pool needs it.
    #[test]
    fn conditional_picks_then_or_else_branch() {
        let mut state = two_card_libraries();
        let ctx = ExecCtx::no_targets(ObjectId(0), PlayerId::P0);

        let taken = EffectOp::Conditional {
            cond: EffectCond::Always,
            then: Box::new(EffectOp::LoseLife {
                player: PlayerRef::Controller,
                amount: 3,
            }),
            else_: Box::new(EffectOp::Sequence(vec![])),
        };
        execute(&taken, &ctx, &mut state);
        assert_eq!(state.players[0].life, 17);

        let not_taken = EffectOp::Conditional {
            cond: EffectCond::Never,
            then: Box::new(EffectOp::LoseLife {
                player: PlayerRef::Controller,
                amount: 100,
            }),
            else_: Box::new(EffectOp::Sequence(vec![])),
        };
        execute(&not_taken, &ctx, &mut state);
        assert_eq!(state.players[0].life, 17, "else branch is a no-op here");
    }

    #[test]
    fn deal_damage_to_target_player_reduces_life() {
        let mut state = two_card_libraries();
        let ctx = ExecCtx {
            stack_item_id: None,
            source: ObjectId(0),
            controller: PlayerId::P0,
            targets: vec![Target::Player(PlayerId::P1)],
            target_contracts: vec![StackTargetContractV4::Player(PlayerId::P1)],
            discarded: Vec::new(),
            paid_cost_refs: Vec::new(),
            hidden_ability_source: None,
            ability_source_contract: None,
            kicked: false,
            optional_additional_cost_paid: None,
            x_value: 0,
        };
        execute(
            &EffectOp::DealDamage {
                target: TargetRef::Target(0),
                amount: 3,
            },
            &ctx,
            &mut state,
        );
        assert_eq!(state.players[1].life, 17);
    }

    #[test]
    fn deal_damage_to_target_object_marks_damage() {
        let mut state = two_card_libraries();
        let creature = state.draw_card(PlayerId::P1).unwrap();
        state.move_hand_to_battlefield(PlayerId::P1, creature);
        let ctx = ExecCtx {
            stack_item_id: None,
            source: ObjectId(0),
            controller: PlayerId::P0,
            targets: vec![Target::Object(creature)],
            target_contracts: vec![StackTargetContractV4::Object {
                object: creature,
                card_def: state.objects.get(creature).card_def,
                owner: state.objects.get(creature).owner,
                controller: state.objects.get(creature).controller,
                zone: state.objects.get(creature).zone,
                zone_change_count: state.objects.get(creature).zone_change_count,
                spell_copy_origin: state.objects.get(creature).spell_copy_origin,
            }],
            discarded: Vec::new(),
            paid_cost_refs: Vec::new(),
            hidden_ability_source: None,
            ability_source_contract: None,
            kicked: false,
            optional_additional_cost_paid: None,
            x_value: 0,
        };
        execute(
            &EffectOp::DealDamage {
                target: TargetRef::Target(0),
                amount: 4,
            },
            &ctx,
            &mut state,
        );
        assert_eq!(state.objects.get(creature).damage, 4);
    }

    #[test]
    fn draw_cards_leaf_draws_the_requested_count() {
        let mut state = two_card_libraries();
        let ctx = ExecCtx::no_targets(ObjectId(0), PlayerId::P0);
        execute(
            &EffectOp::DrawCards {
                player: PlayerRef::Controller,
                count: 2,
            },
            &ctx,
            &mut state,
        );
        assert_eq!(state.players[0].hand.len(), 2);
    }

    fn card_ids(names: &[&str]) -> Vec<u16> {
        names
            .iter()
            .map(|name| crate::card_def::card_id_by_name(name).unwrap())
            .collect()
    }

    #[test]
    fn reveal_top_partition_is_public_ordered_and_not_a_draw() {
        let definitions = card_ids(&[
            "Elvish Mystic",
            "Quirion Ranger",
            "Llanowar Elves",
            "Lightning Bolt",
            "Island",
        ]);
        let mut state = GameState::new_from_libraries(
            &definitions,
            &[],
            |card_def| {
                crate::card_def::CARD_DEFS[card_def as usize]
                    .name
                    .to_string()
            },
            9,
        );
        let original = state.players[0].library.clone();
        state.reveal_library_top(PlayerId::P1, PlayerId::P0, 5);
        let ctx = ExecCtx::no_targets(original[0], PlayerId::P0);

        execute(
            &EffectOp::RevealTopAndPartitionByType {
                player: PlayerRef::Controller,
                count: 4,
                card_type: CardType::Creature,
                matching_to: Zone::Hand,
                rest_to: Zone::Graveyard,
            },
            &ctx,
            &mut state,
        );

        assert_eq!(
            state.players[0].hand,
            vec![original[0], original[1], original[2]]
        );
        assert_eq!(state.players[0].graveyard, vec![original[3]]);
        assert_eq!(state.players[0].library, vec![original[4]]);
        assert_eq!(
            state
                .known_hand_cards(PlayerId::P1, PlayerId::P0)
                .iter()
                .map(|entry| entry.object)
                .collect::<Vec<_>>(),
            vec![original[0], original[1], original[2]]
        );
        assert!(state
            .known_hand_cards(PlayerId::P0, PlayerId::P0)
            .is_empty());
        assert_eq!(
            state
                .known_library_cards(PlayerId::P1, PlayerId::P0)
                .iter()
                .map(|entry| (entry.position, entry.object))
                .collect::<Vec<_>>(),
            vec![(0, original[4])]
        );
        assert_eq!(state.players[0].draws_this_turn, 0);
        assert!(!state.players[0].drew_from_empty);
        assert!(state
            .engine
            .event_history
            .iter()
            .all(|event| !matches!(event, CommittedEvent::Draw { .. })));
        assert_eq!(
            state
                .engine
                .event_history
                .iter()
                .filter_map(|event| match event {
                    CommittedEvent::ZoneChange { object, .. } => Some(*object),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            vec![original[0], original[1], original[2], original[3]]
        );
        for &object in &original[..4] {
            assert_eq!(state.objects.get(object).zone_change_count, 1);
        }
    }

    #[test]
    fn reveal_top_partition_handles_short_empty_and_zero_hit_libraries() {
        let definitions = card_ids(&["Lightning Bolt"]);
        let mut short = GameState::new_from_libraries(
            &definitions,
            &[],
            |card_def| {
                crate::card_def::CARD_DEFS[card_def as usize]
                    .name
                    .to_string()
            },
            1,
        );
        let original = short.players[0].library.clone();
        let ctx = ExecCtx::no_targets(original[0], PlayerId::P0);
        let op = EffectOp::RevealTopAndPartitionByType {
            player: PlayerRef::Controller,
            count: 4,
            card_type: CardType::Creature,
            matching_to: Zone::Hand,
            rest_to: Zone::Graveyard,
        };
        execute(&op, &ctx, &mut short);
        assert!(short.players[0].hand.is_empty());
        assert_eq!(short.players[0].graveyard, original);
        assert!(!short.players[0].drew_from_empty);

        let mut empty = GameState::new_from_libraries(&[], &[], |_| String::new(), 1);
        let before = empty.clone();
        execute(
            &op,
            &ExecCtx::no_targets(ObjectId(0), PlayerId::P0),
            &mut empty,
        );
        assert_eq!(empty, before);
    }

    #[test]
    fn tap_and_add_mana_leaves_compose_a_mana_ability() {
        let mut state = two_card_libraries();
        let land = state.draw_card(PlayerId::P0).unwrap();
        state.move_hand_to_battlefield(PlayerId::P0, land);
        let ctx = ExecCtx::no_targets(land, PlayerId::P0);
        let op = EffectOp::Sequence(vec![
            EffectOp::TapObject {
                object: ObjectRef::ThisSource,
            },
            EffectOp::AddMana {
                player: PlayerRef::Controller,
                colors: vec![ManaColor::R],
            },
        ]);
        execute(&op, &ctx, &mut state);
        assert!(state.objects.get(land).tapped);
        assert_eq!(state.players[0].mana_pool[ManaColor::R.pool_index()], 1);
    }

    #[test]
    fn skip_next_untap_is_incarnation_local_and_snapshot_deterministic() {
        let mut state = two_card_libraries();
        let permanent = state.draw_card(PlayerId::P0).unwrap();
        state.move_hand_to_battlefield(PlayerId::P0, permanent);
        let before_hash = state.diagnostic_state_hash();

        let skip = EffectOp::SkipNextUntap {
            object: ObjectRef::ThisSource,
        };
        let ctx = ExecCtx::no_targets(permanent, PlayerId::P0);
        execute(&skip, &ctx, &mut state);

        assert!(state.objects.get(permanent).v4.skip_next_untap);
        assert_ne!(state.diagnostic_state_hash(), before_hash);
        let once_hash = state.diagnostic_state_hash();

        // Multiple applications before the affected untap step deliberately
        // merge into the same one-shot marker rather than queueing skips.
        execute(&skip, &ctx, &mut state);
        assert_eq!(state.diagnostic_state_hash(), once_hash);

        let snapshot = serde_json::to_vec(&state).unwrap();
        let restored: GameState = serde_json::from_slice(&snapshot).unwrap();
        assert_eq!(restored, state);
        assert_eq!(
            restored.diagnostic_state_hash(),
            state.diagnostic_state_hash()
        );

        event::propose_and_commit(
            &mut state,
            event::ProposedEvent::zone_change(permanent, Zone::Graveyard),
        );
        assert!(!state.objects.get(permanent).v4.skip_next_untap);
    }

    #[test]
    fn create_token_requires_and_materializes_a_full_token_definition() {
        let mut state = two_card_libraries();
        let ctx = ExecCtx::no_targets(ObjectId(0), PlayerId::P0);
        let blood = crate::card_def::card_id_by_name("Blood Token").unwrap();
        execute(
            &EffectOp::CreateToken {
                token_def: blood,
                controller: PlayerRef::Controller,
            },
            &ctx,
            &mut state,
        );
        let created = *state.players[0].battlefield.last().unwrap();
        assert_eq!(state.objects.get(created).card_def, blood);
        assert!(crate::card_def::CARD_DEFS[blood as usize].has_full_support());
    }

    #[test]
    #[should_panic(expected = "CreateToken requires a fully supported executable token definition")]
    fn create_token_fails_loudly_for_a_nontoken_definition() {
        let mut state = two_card_libraries();
        let ctx = ExecCtx::no_targets(ObjectId(0), PlayerId::P0);
        execute(
            &EffectOp::CreateToken {
                token_def: crate::card_def::card_id_by_name("Island").unwrap(),
                controller: PlayerRef::Controller,
            },
            &ctx,
            &mut state,
        );
    }

    #[test]
    fn impulse_draw_exiles_but_does_not_authorize_a_nonplayable_token() {
        let clue = crate::card_def::card_id_by_name("Clue Token").unwrap();
        let mut state = GameState::new_from_libraries(
            &[clue],
            &[],
            |card_def| {
                crate::card_def::CARD_DEFS[card_def as usize]
                    .name
                    .to_string()
            },
            1,
        );
        let card = state.players[0].library[0];
        let ctx = ExecCtx::no_targets(card, PlayerId::P0);

        execute(
            &EffectOp::ImpulseDraw {
                count: 1,
                duration: ImpulseDuration::EndOfTurn,
            },
            &ctx,
            &mut state,
        );

        assert!(state.players[0].library.is_empty());
        assert_eq!(state.objects.get(card).zone, Zone::Exile);
        assert!(state.exile.contains(&card));
        assert!(state.engine.exile_play_permissions.is_empty());
    }
}
