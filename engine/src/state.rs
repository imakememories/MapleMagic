//! Core game state. Every collection here is a `Vec` (or a fixed array) with
//! caller-controlled order; nothing is ever iterated via a `HashMap`, so two
//! states built from the same inputs serialize and hash identically (see
//! `state_hash` and the determinism test below).

use crate::card_def::TargetSpec;
use crate::ids::{Arena, ObjectId, PlayerId, StackItemId};
use crate::mana::ManaColor;
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};

pub const STARTING_LIFE: i32 = 20;

fn i16_is_zero(value: &i16) -> bool {
    *value == 0
}

fn bool_is_false(value: &bool) -> bool {
    !*value
}

/// Exact diagnostic full-state hash contract written into privileged audit
/// artifacts. The algorithm is FNV-1a-64 over the compact UTF-8 JSON bytes of
/// `DiagnosticStateHashEnvelopeV8` below.
///
/// Changing the envelope, JSON representation, or digest algorithm requires a
/// new constant value and an audit-artifact schema bump. Policy artifacts do
/// not contain this privileged full-state diagnostic.
pub const DIAGNOSTIC_STATE_HASH_ALGORITHM: &str = "fnv1a64-serde-json-game-state-envelope-v8";
pub const DIAGNOSTIC_STATE_HASH_ENVELOPE_SCHEMA_VERSION: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Zone {
    Library,
    Hand,
    Battlefield,
    Graveyard,
    Stack,
    Exile,
    /// Emblems, suspended cards, etc. Unused by the 132-card pool today but
    /// cheap to carry so the enum doesn't need to change later.
    Command,
}

/// Counter families required by the Pauper pool. Signed storage is deliberate:
/// effect validation may reject an underflow without first converting between
/// unrelated integer shapes, while i16 leaves ample headroom for copied and
/// doubled counter effects.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Counters {
    pub plus1_plus1: i16,
    pub minus1_minus1: i16,
    pub minus0_minus1: i16,
    pub stun: i16,
    pub lore: i16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ObjectLinkV4 {
    pub object: ObjectId,
    pub zone_change_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AbilityKindV4 {
    Mana,
    Activated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AbilityUseV4 {
    pub ability_kind: AbilityKindV4,
    pub ability_index: u16,
    pub uses: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GoadStateV4 {
    pub player: PlayerId,
    /// First turn number at whose appropriate untap boundary this goad no
    /// longer applies. Duration stays explicit rather than being inferred
    /// from presence in the vector.
    pub expires_at_turn: u32,
}

/// Schema-v4 dynamic object substrate. Base colors, subtypes, and token
/// identity are materialized from the registry at object creation, rather
/// than assigning zero a meaning that would change when mechanics arrive.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ObjectStateV4 {
    pub is_token: bool,
    pub face_index: u8,
    pub effective_color_mask: u8,
    /// Sorted, unique `card_def::Subtype::stable_id()` values.
    pub effective_subtype_ids: Vec<u16>,
    pub chosen_color: Option<ManaColor>,
    pub entered_battlefield_turn: Option<u32>,
    /// Sorted by `(ability_kind, ability_index)`; absent abilities have zero uses.
    pub ability_uses_this_turn: Vec<AbilityUseV4>,
    /// Incarnation-local delayed effect. Control-changing effects must preserve
    /// this marker; only a zone change or the affected controller's next
    /// untap step consumes it.
    pub skip_next_untap: bool,
    /// Whether any currently marked damage was dealt by a source with
    /// deathtouch. This is cleared at cleanup and on every zone change.
    #[serde(default, skip_serializing_if = "bool_is_false")]
    pub deathtouch_damage: bool,
    /// Incarnation-local Lifelink keyword counters. This remains private v4
    /// state so the frozen public `Counters` shape does not change.
    #[serde(default, skip_serializing_if = "i16_is_zero")]
    pub lifelink_keyword_counters: i16,
    /// Sorted by `(player, expires_at_turn)`.
    pub goaded_by: Vec<GoadStateV4>,
    pub attached_to: Option<ObjectLinkV4>,
    pub exiled_by: Option<ObjectLinkV4>,
    pub ward_generic: u16,
    /// `None` means derive the ordinary one-blocker baseline (or Menace)
    /// from effective keywords; `Some` is a rules-effect override.
    pub minimum_blockers_override: Option<u8>,
    /// W/U/B/R/G/C bits naming the land types this object can landwalk.
    pub landwalk_mask: u8,
    /// Exact pre-stack incarnation and authorization route for a physical
    /// spell currently being cast or waiting on the stack. This is kept on
    /// the source object, independently of the mutable `StackItem`, so a
    /// restored stack record cannot coherently relabel Flashback as Normal
    /// (or vice versa) by changing only redundant stack fields. Every
    /// ordinary zone change clears it through `reset_for_zone_change`.
    pub spell_cast_origin: Option<SpellCastOriginV4>,
    /// Independently owned dynamic choices frozen only after a physical
    /// spell's costs commit. The stack item carries the same X and chosen
    /// creature reference; keeping this copy on the source incarnation lets
    /// restored state reject a unilateral rewrite of either value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finalized_cast_binding: Option<FinalizedCastBindingV1>,
}

impl ObjectStateV4 {
    pub fn from_card_def(card_def: u16) -> ObjectStateV4 {
        let def = &crate::card_def::CARD_DEFS[card_def as usize];
        let mut subtype_ids: Vec<u16> = def
            .subtypes
            .iter()
            .map(|subtype| subtype.stable_id())
            .collect();
        if def.changeling {
            subtype_ids.extend(
                crate::card_def::Subtype::CREATURE_TYPES
                    .iter()
                    .map(|subtype| subtype.stable_id()),
            );
        }
        subtype_ids.sort_unstable();
        subtype_ids.dedup();
        ObjectStateV4 {
            is_token: def.is_token,
            face_index: 0,
            effective_color_mask: crate::card_def::mana_colors_mask(def.colors),
            effective_subtype_ids: subtype_ids,
            chosen_color: None,
            entered_battlefield_turn: None,
            ability_uses_this_turn: Vec::new(),
            skip_next_untap: false,
            deathtouch_damage: false,
            lifelink_keyword_counters: 0,
            goaded_by: Vec::new(),
            attached_to: None,
            exiled_by: None,
            ward_generic: match def.ward_cost {
                Some(crate::card_def::WardCostDef::Generic(amount)) => u16::from(amount),
                None => 0,
            },
            minimum_blockers_override: None,
            landwalk_mask: 0,
            spell_cast_origin: None,
            finalized_cast_binding: None,
        }
    }

    pub fn reset_for_zone_change(&mut self, card_def: u16, to_zone: Zone, turn: u32) {
        let base = ObjectStateV4::from_card_def(card_def);
        *self = base;
        if to_zone == Zone::Battlefield {
            self.entered_battlefield_turn = Some(turn);
        }
    }

    pub fn note_ability_use(&mut self, ability_kind: AbilityKindV4, ability_index: u16) {
        let key = (ability_kind, ability_index);
        match self
            .ability_uses_this_turn
            .binary_search_by_key(&key, |entry| (entry.ability_kind, entry.ability_index))
        {
            Ok(index) => {
                self.ability_uses_this_turn[index].uses =
                    self.ability_uses_this_turn[index].uses.saturating_add(1);
            }
            Err(index) => self.ability_uses_this_turn.insert(
                index,
                AbilityUseV4 {
                    ability_kind,
                    ability_index,
                    uses: 1,
                },
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GameObject {
    /// Index into the (not-yet-built) card database.
    pub card_def: u16,
    /// Debug-only display name; not used for gameplay logic.
    pub name: String,
    pub owner: PlayerId,
    pub controller: PlayerId,
    pub zone: Zone,
    pub tapped: bool,
    pub summoning_sick: bool,
    pub damage: u16,
    pub counters: Counters,
    pub attachments: Vec<ObjectId>,
    pub v4: ObjectStateV4,
    /// Immutable provenance for a virtual spell-copy object. Physical cards
    /// and tokens always carry `None`; the spell-copy allocator is the only
    /// production path that creates `Some`. This lives on the arena object,
    /// rather than only on a mutable stack item flag, so a restored state
    /// cannot turn a physical card into a copy (or vice versa) by flipping a
    /// single redundant boolean.
    #[serde(default)]
    pub spell_copy_origin: Option<SpellCopyOriginV4>,
    /// `Some(turn)` iff this card was Plotted (`PlotAbility`) on kernel
    /// round `turn` -- set by `engine::plot_spell`, read by
    /// `engine::is_plotted_castable_now` (castable from exile for free at
    /// sorcery speed, but never the same turn it was plotted). `None` for
    /// every card that has never been Plotted. Only Highway Robbery in this
    /// pool has `CardDef::plot_cost`, so this is `None` for every other
    /// card for the whole game.
    pub plotted_turn: Option<u32>,
    /// How many times this object has ever changed zones (CR 400.7's own
    /// `zoneChangeCounter` concept, ported deliberately -- see `engine::
    /// legal_blockers_for`'s sibling doc mentioning the reference engine's
    /// version). Bumped once per `event::commit_zone_change` call for this
    /// id, regardless of which zones. Read by `engine::PlayPermission::
    /// zone_change_generation`: a permission snapshots this value the
    /// instant it's granted, and is only ever honored while the object's
    /// *current* count still matches -- any further zone change (playing
    /// the card through the permission, or anything else) silently voids
    /// it, structurally, without this module needing to remember to remove
    /// the stale entry.
    pub zone_change_count: u32,
}

impl GameObject {
    fn new_in_library(card_def: u16, name: String, owner: PlayerId) -> GameObject {
        GameObject {
            card_def,
            name,
            owner,
            controller: owner,
            zone: Zone::Library,
            tapped: false,
            summoning_sick: false,
            damage: 0,
            counters: Counters::default(),
            attachments: Vec::new(),
            v4: ObjectStateV4::from_card_def(card_def),
            spell_copy_origin: None,
            plotted_turn: None,
            zone_change_count: 0,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DungeonStateV4 {
    pub dungeon_id: Option<u16>,
    pub room_id: Option<u16>,
    /// Sorted, unique stable dungeon ids.
    pub completed_dungeons: Vec<u16>,
}

pub const UNDERCITY_DUNGEON_ID_V1: u16 = 1;

/// Stable room identity for the one exact dungeon currently supported by
/// the Pauper pool. The numeric ids are also stored in `DungeonStateV4`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UndercityRoomV1 {
    SecretEntrance,
    Forge,
    LostWell,
    Trap,
    Arena,
    Stash,
    Archives,
    Catacombs,
    ThroneOfTheDeadThree,
}

impl UndercityRoomV1 {
    pub const fn stable_id(self) -> u16 {
        match self {
            UndercityRoomV1::SecretEntrance => 1,
            UndercityRoomV1::Forge => 2,
            UndercityRoomV1::LostWell => 3,
            UndercityRoomV1::Trap => 4,
            UndercityRoomV1::Arena => 5,
            UndercityRoomV1::Stash => 6,
            UndercityRoomV1::Archives => 7,
            UndercityRoomV1::Catacombs => 8,
            UndercityRoomV1::ThroneOfTheDeadThree => 9,
        }
    }

    pub fn from_stable_id(id: u16) -> Option<Self> {
        Some(match id {
            1 => UndercityRoomV1::SecretEntrance,
            2 => UndercityRoomV1::Forge,
            3 => UndercityRoomV1::LostWell,
            4 => UndercityRoomV1::Trap,
            5 => UndercityRoomV1::Arena,
            6 => UndercityRoomV1::Stash,
            7 => UndercityRoomV1::Archives,
            8 => UndercityRoomV1::Catacombs,
            9 => UndercityRoomV1::ThroneOfTheDeadThree,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InitiativeTriggerKindV1 {
    CombatTransfer,
    VentureAfterTaking,
    VentureAtUpkeep,
    UndercityRoom(UndercityRoomV1),
}

/// Self-indexed provenance for one global Initiative or Undercity trigger.
/// The matching committed marker and historical designation source must both
/// survive snapshot/restore unchanged before this binding can reach a stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct InitiativeTriggerBindingV1 {
    pub history_index: u32,
    pub player: PlayerId,
    pub source: AbilitySourceContractV4,
    pub kind: InitiativeTriggerKindV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PlayerState {
    pub life: i32,
    /// Index 0 = TOP of library. Drawing removes index 0.
    pub library: Vec<ObjectId>,
    /// Insertion order, oldest first; order is player-visible info in traces.
    pub hand: Vec<ObjectId>,
    /// Insertion order (order a permanent entered), not board position.
    pub battlefield: Vec<ObjectId>,
    /// Order matters (top of graveyard, Karmic Guide-style effects, etc).
    /// Last element is the most-recently-added (top).
    pub graveyard: Vec<ObjectId>,
    /// [W, U, B, R, G, C].
    pub mana_pool: [u8; 6],
    pub has_lost: bool,
    pub lands_played_this_turn: u8,
    /// Set by `event::commit` when a `Draw` was attempted against an empty
    /// library. Checked (and turned into `has_lost`) by
    /// `trigger::sba_fixed_point` (rule 704.5c).
    pub drew_from_empty: bool,
    /// Cards successfully drawn since the current turn began (both
    /// players' counters reset together at every `Step::Untap`, matching
    /// the reference engine's `DrawNthCardWatcher`, which is a
    /// whole-game-scoped watcher whose backing map is cleared once per
    /// turn boundary). Used by `trigger::TriggerCondition::DrawNth`
    /// (Sneaky Snacker: "whenever you draw your third card in a turn").
    pub draws_this_turn: u32,
    pub spells_cast_this_turn: u16,
    pub dungeon: DungeonStateV4,
}

impl PlayerState {
    fn new(life: i32) -> PlayerState {
        PlayerState {
            life,
            library: Vec::new(),
            hand: Vec::new(),
            battlefield: Vec::new(),
            graveyard: Vec::new(),
            mana_pool: [0; 6],
            has_lost: false,
            lands_played_this_turn: 0,
            drew_from_empty: false,
            draws_this_turn: 0,
            spells_cast_this_turn: 0,
            dungeon: DungeonStateV4::default(),
        }
    }
}

/// Steps the RL decision stream actually visits (see golden-trace `phase`
/// field). Untap/Cleanup are included even though the reference engine
/// rarely stops for priority there, since the kernel still transitions
/// through them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Step {
    Untap,
    Upkeep,
    Draw,
    Main1,
    BeginCombat,
    DeclareAttackers,
    DeclareBlockers,
    CombatDamage,
    EndCombat,
    Main2,
    End,
    Cleanup,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Target {
    Object(ObjectId),
    Player(PlayerId),
}

/// Publicly distinguishable origin of a stack item. This is stamped by the
/// creation path, not inferred from card text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StackItemKind {
    Spell,
    ActivatedAbility,
    TriggeredAbility,
    MadnessOffer,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CastMethodV4 {
    #[default]
    Normal,
    Alternative,
    Flashback,
    Madness,
    Plotted,
    Escape,
    Bestow,
    Omen,
}

/// Independently frozen route by which a physical spell's immediately
/// preceding incarnation was authorized to enter the stack. Exile routes
/// are deliberately distinct: ordinary impulse permission, Plot, and
/// Madness share a zone but never share a cost or departure contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpellCastRouteV4 {
    Hand,
    GraveyardFlashback,
    ExilePermission {
        holder: PlayerId,
        permission_zone_change_count: u32,
    },
    Plotted {
        plotted_turn: u32,
    },
    Madness,
    GraveyardEscape,
}

/// Incarnation-local cast provenance stored on the physical source object
/// and copied into its stack source contract. `finalized_method` remains
/// `None` while 601.2's choices and payments are pending, then is stamped
/// exactly once after the definition-derived cost successfully commits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SpellCastOriginV4 {
    pub origin_zone: Zone,
    pub origin_zone_change_count: u32,
    pub route: SpellCastRouteV4,
    pub finalized_method: Option<CastMethodV4>,
}

/// Immutable provenance of a virtual spell-copy arena object. The parent
/// binding is historical: once the parent leaves the stack its live zone
/// generation may advance, while this record continues to identify exactly
/// which stack incarnation produced the copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SpellCopyOriginV4 {
    pub parent: ObjectId,
    pub parent_card_def: u16,
    pub parent_owner: PlayerId,
    pub parent_controller: PlayerId,
    pub parent_stack_zone_change_count: u32,
    /// Whether the immediate parent arena object was itself a virtual copy.
    /// This redundant immutable bit lets descendants validate ancestry even
    /// after that parent has ceased and no stack item remains to carry an
    /// `is_copy` flag.
    pub parent_was_copy: bool,
}

/// Exact source identity owned by a spell stack item. A triggered ability
/// created by that spell while it remains on the stack carries the same
/// contract as producer provenance; other abilities carry `None`. Every
/// spell, including a transient cast placeholder and a virtual copy, carries
/// `Some` captured after its source entered the stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StackSourceContractV4 {
    pub source: ObjectId,
    pub card_def: u16,
    pub owner: PlayerId,
    pub controller: PlayerId,
    pub zone: Zone,
    pub zone_change_count: u32,
    pub spell_copy_origin: Option<SpellCopyOriginV4>,
    pub spell_cast_origin: Option<SpellCastOriginV4>,
    pub cast_method: CastMethodV4,
    /// Duplicate of the source object's finalized dynamic cast choices.
    /// Pending placeholders and virtual copies retain `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finalized_cast_binding: Option<FinalizedCastBindingV1>,
}

impl StackSourceContractV4 {
    pub fn capture(
        state: &GameState,
        source: ObjectId,
        cast_method: CastMethodV4,
    ) -> StackSourceContractV4 {
        let object = state.objects.get(source);
        StackSourceContractV4 {
            source,
            card_def: object.card_def,
            owner: object.owner,
            controller: object.controller,
            zone: object.zone,
            zone_change_count: object.zone_change_count,
            spell_copy_origin: object.spell_copy_origin,
            spell_cast_origin: object.v4.spell_cast_origin,
            cast_method,
            finalized_cast_binding: object.v4.finalized_cast_binding,
        }
    }
}

/// Exact exiled-card incarnation that owns a Madness offer on the stack.
/// This is deliberately separate from [`StackSourceContractV4`]: the latter
/// describes a spell after 601.2a has moved it to the stack, while a Madness
/// offer is a triggered ability whose physical source must remain in exile
/// until the offer is accepted or declined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MadnessOfferSourceContractV4 {
    pub source: ObjectId,
    pub card_def: u16,
    pub owner: PlayerId,
    pub controller: PlayerId,
    pub zone: Zone,
    pub zone_change_count: u32,
}

impl MadnessOfferSourceContractV4 {
    pub fn capture(state: &GameState, source: ObjectId) -> MadnessOfferSourceContractV4 {
        let object = state.objects.get(source);
        MadnessOfferSourceContractV4 {
            source,
            card_def: object.card_def,
            owner: object.owner,
            controller: object.controller,
            zone: object.zone,
            zone_change_count: object.zone_change_count,
        }
    }
}

/// Frozen source identity for a non-Madness triggered or activated ability.
/// The physical card may legally change zones, or even be cast again, while
/// the ability remains on the stack. This contract authenticates the
/// definition, owner, controller, zone, generation, and attachment at
/// creation without requiring the historical incarnation to remain live.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AbilitySourceContractV4 {
    pub source: ObjectId,
    pub card_def: u16,
    pub owner: PlayerId,
    pub controller: PlayerId,
    pub zone: Zone,
    pub zone_change_count: u32,
    /// Exact host relation at trigger or activation creation. This is LKI,
    /// not a claim that the source remains attached while the ability waits.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attached_to: Option<ObjectLinkV4>,
}

impl AbilitySourceContractV4 {
    pub fn capture(state: &GameState, source: ObjectId) -> AbilitySourceContractV4 {
        let object = state.objects.get(source);
        AbilitySourceContractV4 {
            source,
            card_def: object.card_def,
            owner: object.owner,
            controller: object.controller,
            zone: object.zone,
            zone_change_count: object.zone_change_count,
            attached_to: object.v4.attached_to,
        }
    }
}

/// One card exiled by an exact historical source incarnation and still
/// eligible for that incarnation's linked leave-the-battlefield return.
/// The source may move while its leave trigger waits, and the exiled arena
/// id may later represent a different incarnation, so both generations are
/// frozen independently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LinkedExileRecordV4 {
    pub source: AbilitySourceContractV4,
    pub exiled: ObjectId,
    pub exiled_card_def: u16,
    pub exiled_owner: PlayerId,
    pub exiled_zone_change_count: u32,
}

/// Historical identity of one object used to pay a cost. It belongs to the
/// stack incarnation and must not follow the arena object through later zone
/// changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PaidCostRefV4 {
    pub object: ObjectId,
    pub card_def: u16,
    pub owner: PlayerId,
    pub controller: PlayerId,
    /// Zone and generation immediately after the cost finished moving the
    /// object. These are historical provenance, not a live object lookup.
    pub zone: Zone,
    pub zone_change_count: u32,
    /// P0/P1 bitmask frozen at payment time. Public payment destinations are
    /// visible to both seats; hidden destinations retain only observers who
    /// actually knew that exact incarnation.
    pub visible_to_mask: u8,
    /// Last-known effective power for a cost that explicitly binds a chosen
    /// creature. Battlefield bindings are refreshed immediately before that
    /// exact incarnation leaves; hand-card bindings retain printed power.
    /// Other costs leave this absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub power_lki: Option<i32>,
}

/// Dynamic cast choices whose exact values must remain independently bound
/// to a physical spell's source incarnation. The optional creature reference
/// is used by additional costs that choose a controlled creature or reveal a
/// creature card from hand; its power is refreshed as LKI immediately before
/// that exact battlefield incarnation leaves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FinalizedCastBindingV1 {
    pub x_value: u16,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chosen_creature_cost: Option<PaidCostRefV4>,
}

/// Historical contract for one announced stack target. Unlike the live
/// `Target` convenience vector, an object target freezes every field needed
/// to identify the exact CR 400.7 incarnation even if that arena id later
/// leaves and re-enters a zone before resolution. Stack observations project
/// this record, never the potentially newer live object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StackTargetContractV4 {
    Player(PlayerId),
    Object {
        object: ObjectId,
        card_def: u16,
        owner: PlayerId,
        controller: PlayerId,
        zone: Zone,
        zone_change_count: u32,
        #[serde(default)]
        spell_copy_origin: Option<SpellCopyOriginV4>,
    },
}

impl StackTargetContractV4 {
    pub fn capture(state: &GameState, target: Target) -> StackTargetContractV4 {
        match target {
            Target::Player(player) => StackTargetContractV4::Player(player),
            Target::Object(object) => {
                let live = state.objects.get(object);
                StackTargetContractV4::Object {
                    object,
                    card_def: live.card_def,
                    owner: live.owner,
                    controller: live.controller,
                    zone: live.zone,
                    zone_change_count: live.zone_change_count,
                    spell_copy_origin: live.spell_copy_origin,
                }
            }
        }
    }

    pub const fn target(self) -> Target {
        match self {
            StackTargetContractV4::Player(player) => Target::Player(player),
            StackTargetContractV4::Object { object, .. } => Target::Object(object),
        }
    }
}

/// Validates immutable target provenance without asking whether the target is
/// still legal now. A later incarnation is an ordinary stale target and may
/// fizzle; a contract from the future, an impossible announcement zone, or a
/// same-incarnation zone disagreement is malformed state and must fail closed.
/// Controller is deliberately historical only: control can change without a
/// zone change, so comparing it with the live object would reject valid games.
pub fn stack_target_contract_is_structurally_valid(
    state: &GameState,
    spec: TargetSpec,
    target_index: usize,
    target: Target,
    contract: StackTargetContractV4,
) -> bool {
    if contract.target() != target {
        return false;
    }
    let shape_is_valid = matches!(
        (spec, target_index, contract),
        (
            TargetSpec::AnyPlayer
                | TargetSpec::AnyTarget
                | TargetSpec::PlayerThenTheirCreature
                | TargetSpec::TargetOpponent,
            0,
            StackTargetContractV4::Player(_),
        ) | (
            TargetSpec::UpToTwoPlayers,
            0 | 1,
            StackTargetContractV4::Player(_),
        ) | (
            TargetSpec::PlayerThenTheirCreature,
            1,
            StackTargetContractV4::Object {
                zone: Zone::Battlefield,
                ..
            },
        ) | (
            TargetSpec::AnyTarget
                | TargetSpec::AnyPermanent
                | TargetSpec::BluePermanent
                | TargetSpec::RedPermanent
                | TargetSpec::NonlandPermanent
                | TargetSpec::Creature
                | TargetSpec::CreatureOtherThanSource
                | TargetSpec::NonlegendaryCreature
                | TargetSpec::ArtifactPermanent
                | TargetSpec::EnchantmentPermanent
                | TargetSpec::ControlledCreature
                | TargetSpec::OpponentControlledCreature
                | TargetSpec::UpToOneTappedCreature
                | TargetSpec::NoncreatureArtifactPermanent
                | TargetSpec::Land
                | TargetSpec::OpponentArtifactOrEnchantmentPermanent,
            0,
            StackTargetContractV4::Object {
                zone: Zone::Battlefield,
                ..
            },
        ) | (
            TargetSpec::UpToTwoCreatures | TargetSpec::ExactlyTwoArtifactPermanents,
            0 | 1,
            StackTargetContractV4::Object {
                zone: Zone::Battlefield,
                ..
            },
        ) | (
            TargetSpec::CreatureOrLandCardInGraveyard | TargetSpec::CreatureCardInOwnGraveyard,
            0,
            StackTargetContractV4::Object {
                zone: Zone::Graveyard,
                ..
            },
        ) | (
            TargetSpec::UpToTwoCreatureCardsInOwnGraveyard,
            0 | 1,
            StackTargetContractV4::Object {
                zone: Zone::Graveyard,
                ..
            },
        ) | (
            TargetSpec::UpToTwoCardsInGraveyards,
            0 | 1,
            StackTargetContractV4::Object {
                zone: Zone::Graveyard,
                ..
            },
        ) | (
            TargetSpec::AnySpellOnStack
                | TargetSpec::InstantSpellOnStack
                | TargetSpec::BlueSpellOnStack
                | TargetSpec::RedSpellOnStack
                | TargetSpec::ArtifactOrEnchantmentSpellOnStack
                | TargetSpec::SorcerySpellOnStack
                | TargetSpec::NoncreatureSpellOnStack
                | TargetSpec::ArtifactSpellOnStack
                | TargetSpec::SpellManaValueAtMostControlledSubtypes { .. },
            0,
            StackTargetContractV4::Object {
                zone: Zone::Stack,
                ..
            },
        )
    );
    if !shape_is_valid {
        return false;
    }
    let StackTargetContractV4::Object {
        object,
        card_def,
        owner,
        zone,
        zone_change_count,
        spell_copy_origin,
        ..
    } = contract
    else {
        return true;
    };
    state.objects.try_get(object).is_some_and(|live| {
        live.card_def == card_def
            && live.owner == owner
            && live.spell_copy_origin == spell_copy_origin
            && if spell_copy_origin.is_some() {
                zone == Zone::Stack
                    && live.zone == Zone::Stack
                    && zone_change_count == live.zone_change_count
            } else {
                zone_change_count <= live.zone_change_count
                    && (zone_change_count != live.zone_change_count || zone == live.zone)
            }
    })
}

/// Cast/payment provenance that belongs to the stack incarnation, not the
/// underlying card object. Abilities use `cast_method: None`; spells always
/// carry an explicit method (ordinary casts are `Some(Normal)`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StackStateV4 {
    /// Private deterministic identity of this exact stack incarnation.
    /// Public stack and policy schemas deliberately continue to use their
    /// frozen positional representation.
    #[serde(default)]
    pub stack_item_id: StackItemId,
    pub cast_method: Option<CastMethodV4>,
    pub face_index: u8,
    pub x_value: u16,
    pub paid_cost_refs: Vec<PaidCostRefV4>,
    /// Exact source incarnation for a spell stack item, also copied onto a
    /// triggered ability produced by that spell before it leaves the stack.
    /// This is internal full-state provenance and intentionally does not
    /// alter public stack, observation, action, or policy schemas.
    #[serde(default)]
    pub source_contract: Option<StackSourceContractV4>,
    /// Exact exiled source incarnation for a Madness-offer triggered ability.
    /// `None` for spells and every other ability kind. Internal full-state
    /// provenance only; public observation/action/policy shapes are unchanged.
    #[serde(default)]
    pub madness_source_contract: Option<MadnessOfferSourceContractV4>,
    /// Target specification selected for this stack item (mode-aware for a
    /// spell, definition-owned for an activation). `None` only for untargeted
    /// abilities/triggers and the transient cast placeholder before 601.2c.
    #[serde(default)]
    pub target_spec: Option<crate::card_def::TargetSpec>,
    /// Announced targets in printed order with full historical identities.
    /// Kept internal to full state/snapshots; public wire shapes are unchanged.
    #[serde(default)]
    pub target_contracts: Vec<StackTargetContractV4>,
    /// Definition-owned index for an activated ability stack item. This is
    /// internal provenance used to validate its target specification at
    /// resolution; public stack/action schemas remain unchanged.
    #[serde(default)]
    pub activated_ability_index: Option<u8>,
    /// Exact hidden-zone source incarnation revealed for an activated
    /// ability. Ninjutsu freezes this link while the ability is on the
    /// stack; if that incarnation leaves the hand, resolution has no effect.
    /// Other abilities retain `None`.
    #[serde(default)]
    pub hidden_ability_source: Option<ObjectLinkV4>,
    /// Historical source incarnation for a definition-owned triggered or
    /// activated ability. Unlike a spell source, it need not remain live in
    /// its captured zone while the ability waits or resolves.
    #[serde(default)]
    pub ability_source_contract: Option<AbilitySourceContractV4>,
    /// Exact Equipment incarnation that granted a triggered ability to this
    /// stack item's source creature. Only attachment-granted triggers use it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub granted_by: Option<AbilitySourceContractV4>,
    /// Optional additional cost actually paid for this exact cast or the
    /// triggered ability produced by it. Internal cast provenance only;
    /// existing public stack schemas continue to expose the associated
    /// physical payments through `paid_cost_refs`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub optional_additional_cost_paid: Option<crate::card_def::OptionalAdditionalCostDef>,
}

impl StackStateV4 {
    pub fn spell(cast_method: CastMethodV4) -> StackStateV4 {
        StackStateV4 {
            cast_method: Some(cast_method),
            ..StackStateV4::default()
        }
    }
}

/// Minimal stack entry: enough to represent "something is on the stack with
/// these targets." Resolution/effect semantics belong to the step layer.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StackItem {
    pub kind: StackItemKind,
    pub source: ObjectId,
    pub controller: PlayerId,
    pub targets: Vec<Target>,
    /// True iff this spell is a virtual copy rather than a physical card.
    /// Copies receive their own stable arena object so stack targeting never
    /// aliases the original spell, but they cease to exist instead of moving
    /// to a graveyard/exile when they resolve, fizzle, or are countered.
    /// Always false for abilities and ordinary cast spells.
    pub is_copy: bool,
    /// `Some` for a triggered ability or a non-mana activated ability
    /// (Masked Meower's, the Blood token's -- see `card_def::
    /// ActivatedAbilityDef`): an inline effect program that isn't looked
    /// up from a `CARD_DEFS` entry at resolution time. `None` for a spell,
    /// whose program is looked up from
    /// `card_def::CARD_DEFS[objects[source].card_def].spell_effect`
    /// instead.
    pub inline_effect: Option<crate::effect::EffectOp>,
    /// Cards discarded to pay this cast/activation's cost (Grab the Prize's
    /// additional cost or Cycling/typecycling's source), threaded through to
    /// `effect::ExecCtx::discarded` at resolution time.
    pub discarded: Vec<ObjectId>,
    /// True iff this spell was cast via flashback: on resolution, an
    /// instant/sorcery goes to exile instead of the graveyard (702.10e).
    pub is_flashback: bool,
    /// Which mode this cast chose, for a modal spell (`card_def::CardDef::
    /// mode2`): `0` = the card's primary `target_spec`/`spell_effect`, `1`
    /// = `mode2`. Always `0` for a non-modal card (`mode2 == None`), which
    /// is every card in this pool except the four Blast cards.
    pub mode_chosen: u8,
    /// True iff this item is a Madness triggered-ability offer (`card_def::
    /// CardDef::madness_cost`), not a normal spell/ability -- pushed by
    /// `engine::push_trigger_onto_stack` from a `trigger::PendingTrigger`
    /// whose own `is_madness_offer` is set (see that field's doc). Resolving
    /// this item (both players pass priority with it on top, same as any
    /// other stack object -- 117.5) is a real player decision
    /// (`engine::Decision::ChooseMadnessCast`: cast `source` for its madness
    /// cost, or let it go to the graveyard), not a fixed `EffectOp` program,
    /// so `inline_effect` is always `None` here and `engine::
    /// advance_until_decision`'s stack-resolution check special-cases this
    /// flag before ever calling `resolve_top_of_stack`. `false` for every
    /// other stack item (a spell, a normal triggered ability, or a non-mana
    /// activated ability).
    pub madness_offer: bool,
    /// True iff this stack item's own cast paid `card_def::CardDef::
    /// kicker_cost` (Goblin Bushwhacker). Cast-time metadata (CR 702.33/
    /// 601.2f), not a durable fact stored anywhere keyed by stable object
    /// id: `engine::finalize_cast` stamps it on the spell's own item;
    /// `engine::resolve_top_of_stack` copies it into that resolution's
    /// `effect::ExecCtx::kicked` and (via `EngineState::
    /// pending_kicked_source`) into the ETB trigger's own `trigger::
    /// PendingTrigger`, whose `engine::push_trigger_onto_stack` copies it
    /// again onto *that* trigger's stack item -- so by the time the
    /// trigger itself resolves, its own `ExecCtx::kicked` is correctly set,
    /// with nothing left over anywhere once both items have resolved.
    /// `false` for every other stack item (no other card in this pool has
    /// Kicker).
    pub kicked: bool,
    pub v4: StackStateV4,
}

/// One card identity an observer is entitled to know at a specific library
/// position. Knowledge is stored separately from the omniscient library order
/// so observations can project only the acting player's information without
/// leaking the opponent's private look/reorder choices.
///
/// `zone_change_count` binds the fact to this exact object incarnation. A card
/// that leaves and later returns to a library cannot accidentally resurrect an
/// older knowledge entry merely because the arena id is stable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LibraryKnowledgeEntry {
    pub position: u32,
    pub object: ObjectId,
    pub zone_change_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HandKnowledgeEntry {
    pub object: ObjectId,
    pub zone_change_count: u32,
}

/// Counter-based, seedable, serializable PRNG (SplitMix64). Deterministic:
/// same seed and same call sequence always produce the same stream. The wire
/// shape is strict: exactly the one `state` member, unknown members rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn seed(seed: u64) -> SplitMix64 {
        SplitMix64 { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

/// The one-of game randomness representation, flattened at the historical
/// `rng` field position. Externally tagged so legacy states serialize the
/// exact `"rng"` key and v2 states the exact `"environment_randomization_v2"`
/// key. `Hash` is manual: the legacy arm delegates directly to the inner
/// `SplitMix64` with no enum discriminant, preserving the frozen hot-path
/// `state_hash` sequence; the v2 arm hashes an explicit discriminator plus
/// root and ordinals.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
enum GameRandomnessState {
    #[serde(rename = "rng")]
    Legacy(SplitMix64),
    #[serde(rename = "environment_randomization_v2")]
    EnvironmentV2(crate::environment_randomization_v2::GameEnvironmentRandomizationV2),
}

impl std::hash::Hash for GameRandomnessState {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            GameRandomnessState::Legacy(rng) => rng.hash(state),
            GameRandomnessState::EnvironmentV2(v2) => {
                state.write_u8(2);
                v2.hash(state);
            }
        }
    }
}

/// The unified library-shuffle error. Minimal and separate from the sealed
/// KDF error enum; the only automatic conversion is a KDF error into the
/// `Derivation` variant. Effect trust boundaries map this to `String`
/// explicitly.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LibraryShuffleError {
    /// The owner is not exactly `PlayerId::P0` or `PlayerId::P1`.
    InvalidOwner,
    /// The owner's committed live-shuffle ordinal has no successor.
    ExhaustedOwnerOrdinal,
    /// The caller's owner does not match the token's owner.
    CallerOwnerMismatch,
    /// The token does not match the state (mode, root, ordinal, successor,
    /// expected RNG, or recomputed seed).
    TokenStateMismatch,
    /// The KDF rejected the derivation.
    Derivation(crate::environment_randomization_v2::EnvironmentRandomizationErrorV2),
}

impl From<crate::environment_randomization_v2::EnvironmentRandomizationErrorV2>
    for LibraryShuffleError
{
    fn from(error: crate::environment_randomization_v2::EnvironmentRandomizationErrorV2) -> Self {
        LibraryShuffleError::Derivation(error)
    }
}

impl std::fmt::Display for LibraryShuffleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LibraryShuffleError::InvalidOwner => write!(f, "library shuffle: invalid owner"),
            LibraryShuffleError::ExhaustedOwnerOrdinal => {
                write!(f, "library shuffle: owner live-shuffle ordinal exhausted")
            }
            LibraryShuffleError::CallerOwnerMismatch => {
                write!(
                    f,
                    "library shuffle: caller owner does not match token owner"
                )
            }
            LibraryShuffleError::TokenStateMismatch => {
                write!(f, "library shuffle: token does not match state")
            }
            LibraryShuffleError::Derivation(error) => {
                write!(f, "library shuffle: derivation rejected: {error:?}")
            }
        }
    }
}

fn library_shuffle_owner(owner: PlayerId) -> Result<PlayerId, LibraryShuffleError> {
    if owner == PlayerId::P0 || owner == PlayerId::P1 {
        Ok(owner)
    } else {
        Err(LibraryShuffleError::InvalidOwner)
    }
}

fn physical_owner_v2_exact(
    owner: PlayerId,
) -> Result<crate::environment_randomization_v2::PhysicalOwnerV2, LibraryShuffleError> {
    use crate::environment_randomization_v2 as env2;
    if owner == PlayerId::P0 {
        Ok(env2::PhysicalOwnerV2::P0)
    } else if owner == PlayerId::P1 {
        Ok(env2::PhysicalOwnerV2::P1)
    } else {
        Err(LibraryShuffleError::InvalidOwner)
    }
}

/// The private per-mode authorization carried by a shuffle token. The legacy
/// arm snapshots the complete expected RNG; commit rechecks it in full.
#[derive(Debug)]
enum LibraryShuffleAuthorization {
    Legacy {
        expected_rng: SplitMix64,
    },
    EnvironmentV2 {
        pair_root: u64,
        ordinal: u64,
        next_ordinal: u64,
        derived_seed: u64,
    },
}

/// A preflighted library-shuffle authorization for either randomness mode.
/// Crate-private, neither `Clone` nor `Copy` nor serializable, single-use by
/// move. Obtainable only from `preflight_library_shuffle`; consumed only by
/// `commit_library_shuffle`, which revalidates every binding against the
/// state before any mutation. Tokens never enter any `EffectFrame` or
/// serialized state.
#[must_use]
#[derive(Debug)]
pub(crate) struct LibraryShuffleToken {
    owner: PlayerId,
    authorization: LibraryShuffleAuthorization,
}

/// `Hash` is manual (see the `impl Hash for GameState` block below this
/// struct): it must reproduce the exact pre-existing field-hash sequence for
/// a legacy P0-first state, the same discipline `starting_player`'s serde
/// attributes above already apply to JSON. A plain `#[derive(Hash)]` would
/// unconditionally fold the new `starting_player` field into every hash,
/// changing `state_hash()`'s output for every existing P0-first state, which
/// `legacy_randomization_bytes_are_sealed` below (a sealed golden byte/hash
/// artifact predating this field) catches immediately.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GameState {
    pub objects: Arena<GameObject>,
    pub players: [PlayerState; 2],
    pub turn: u32,
    pub active_player: PlayerId,
    pub priority_player: PlayerId,
    /// The physical player who took the very first turn of the game. Read by
    /// the first-turn draw-skip (`engine.rs`'s `Step::Draw` arm) and the
    /// round-wraparound counter (`engine.rs`'s `advance_step`) instead of
    /// either site hardcoding `PlayerId::P0`. Set once at construction and
    /// never mutated afterward. Skipped on the wire whenever it is the
    /// legacy default (`PlayerId::P0`), via `skip_serializing_if` below, so
    /// every existing P0-first serialized `GameState`, diagnostic hash, and
    /// sealed golden byte artifact stays byte-identical; see
    /// `starting_player_is_p0_v1`/`default_starting_player_v1` and
    /// `P1-METAMORPHIC-AUDIT-DESIGN-V4.md` Section 1.2's bit-identity
    /// requirement.
    #[serde(
        default = "default_starting_player_v1",
        skip_serializing_if = "starting_player_is_p0_v1"
    )]
    pub starting_player: PlayerId,
    pub step: Step,
    pub stack: Vec<StackItem>,
    pub exile: Vec<ObjectId>,
    pub command: Vec<ObjectId>,
    /// The player who currently holds the initiative, if any.
    pub initiative: Option<PlayerId>,
    /// Observer x library-owner knowledge. Each inner vector is sorted by
    /// `position` and contains no duplicate positions or object incarnations.
    /// This is full engine state (and therefore snapshot/hash state), but RL
    /// observations expose only the row for the acting observer.
    pub library_knowledge: [[Vec<LibraryKnowledgeEntry>; 2]; 2],
    /// Observer x hand-owner identity knowledge. As with library knowledge,
    /// observation code may project only the acting observer's row.
    pub hand_knowledge: [[Vec<HandKnowledgeEntry>; 2]; 2],
    #[serde(flatten)]
    randomness: GameRandomnessState,
    /// Priority/stack/turn-structure bookkeeping and the propose-commit
    /// event log, all owned by the `engine`/`event`/`trigger` modules. See
    /// `engine::EngineState`.
    pub engine: crate::engine::EngineState,
}

/// Reproduces exactly the field-hash sequence `#[derive(Hash)]` produced
/// before `starting_player` existed, for every field except `starting_player`
/// itself, in the same declaration order. `starting_player` is folded in only
/// when it is not the legacy default (`PlayerId::P0`): a P0-first state
/// therefore hashes byte-for-byte identically to the pre-`starting_player`
/// code (see `legacy_randomization_bytes_are_sealed`'s sealed `state_hash_hex`
/// golden), while a P1-first state still hashes distinguishably from its
/// P0-first counterpart. Same discipline `GameRandomnessState`'s own manual
/// `Hash` impl already applies to its `Legacy` arm (no discriminant) versus
/// its `EnvironmentV2` arm (explicit discriminant).
impl Hash for GameState {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.objects.hash(state);
        self.players.hash(state);
        self.turn.hash(state);
        self.active_player.hash(state);
        self.priority_player.hash(state);
        self.step.hash(state);
        self.stack.hash(state);
        self.exile.hash(state);
        self.command.hash(state);
        self.initiative.hash(state);
        self.library_knowledge.hash(state);
        self.hand_knowledge.hash(state);
        self.randomness.hash(state);
        self.engine.hash(state);
        if self.starting_player != PlayerId::P0 {
            self.starting_player.hash(state);
        }
    }
}

impl PaidCostRefV4 {
    pub fn capture(state: &GameState, object_id: ObjectId) -> PaidCostRefV4 {
        let object = state.objects.get(object_id);
        let visible_to_mask = match object.zone {
            Zone::Battlefield | Zone::Graveyard | Zone::Stack | Zone::Exile | Zone::Command => 0b11,
            Zone::Hand => [PlayerId::P0, PlayerId::P1]
                .into_iter()
                .filter(|&observer| {
                    observer == object.owner
                        || state
                            .known_hand_cards(observer, object.owner)
                            .iter()
                            .any(|entry| {
                                entry.object == object_id
                                    && entry.zone_change_count == object.zone_change_count
                            })
                })
                .fold(0, |mask, observer| mask | (1 << observer.index())),
            Zone::Library => [PlayerId::P0, PlayerId::P1]
                .into_iter()
                .filter(|&observer| {
                    state
                        .known_library_cards(observer, object.owner)
                        .iter()
                        .any(|entry| {
                            entry.object == object_id
                                && entry.zone_change_count == object.zone_change_count
                        })
                })
                .fold(0, |mask, observer| mask | (1 << observer.index())),
        };
        PaidCostRefV4 {
            object: object_id,
            card_def: object.card_def,
            owner: object.owner,
            controller: object.controller,
            zone: object.zone,
            zone_change_count: object.zone_change_count,
            visible_to_mask,
            power_lki: None,
        }
    }

    pub fn visible_to(self, observer: PlayerId) -> bool {
        self.visible_to_mask & (1 << observer.index()) != 0
    }
}

impl GameState {
    /// Builds a fresh pre-game state from two post-shuffle library orders
    /// (index 0 = top, matching `GoldenTrace::opening_library`). Arena ids
    /// are assigned contiguously in library order, player 0 first, so the
    /// id assignment is fully determined by the two input vecs.
    pub fn new_from_libraries(
        lib0: &[u16],
        lib1: &[u16],
        names: impl Fn(u16) -> String,
        seed: u64,
    ) -> GameState {
        // Thin wrapper over the starting-player-aware core below, with the
        // exact pre-existing literal (`PlayerId::P0`) supplied structurally:
        // this function's own body and behavior are unchanged.
        Self::new_from_libraries_with_starting_player_v1(lib0, lib1, names, seed, PlayerId::P0)
    }

    /// Opt-in sibling of [`GameState::new_from_libraries`] taking an explicit
    /// starting player (`P1-METAMORPHIC-AUDIT-DESIGN-V4.md` Section 1.2, site
    /// 1 of the three coupled starting-player-authority sites). `starting_player`
    /// is the constructor's only extra input: it sets `active_player`,
    /// `priority_player`, and the new `starting_player` field to the same
    /// value, so all three stay consistent from the very first state. The
    /// other two coupled sites (`engine.rs`'s first-turn draw-skip and
    /// round-wraparound) read `state.starting_player` rather than a literal.
    /// `new_from_libraries` itself calls this with `PlayerId::P0`, so it is
    /// the exact pre-change literal reproduced structurally, not merely by
    /// convention.
    pub fn new_from_libraries_with_starting_player_v1(
        lib0: &[u16],
        lib1: &[u16],
        names: impl Fn(u16) -> String,
        seed: u64,
        starting_player: PlayerId,
    ) -> GameState {
        let mut objects = Arena::with_capacity(lib0.len() + lib1.len());
        let mut library0 = Vec::with_capacity(lib0.len());
        let mut library1 = Vec::with_capacity(lib1.len());

        for &card_def in lib0 {
            let id = objects.push(GameObject::new_in_library(
                card_def,
                names(card_def),
                PlayerId::P0,
            ));
            library0.push(id);
        }
        for &card_def in lib1 {
            let id = objects.push(GameObject::new_in_library(
                card_def,
                names(card_def),
                PlayerId::P1,
            ));
            library1.push(id);
        }

        let mut player0 = PlayerState::new(STARTING_LIFE);
        player0.library = library0;
        let mut player1 = PlayerState::new(STARTING_LIFE);
        player1.library = library1;

        GameState {
            objects,
            players: [player0, player1],
            turn: 1,
            active_player: starting_player,
            priority_player: starting_player,
            starting_player,
            step: Step::Untap,
            stack: Vec::new(),
            exile: Vec::new(),
            command: Vec::new(),
            initiative: None,
            library_knowledge: std::array::from_fn(|_| {
                std::array::from_fn(|_| Vec::<LibraryKnowledgeEntry>::new())
            }),
            hand_knowledge: std::array::from_fn(|_| {
                std::array::from_fn(|_| Vec::<HandKnowledgeEntry>::new())
            }),
            randomness: GameRandomnessState::Legacy(SplitMix64::seed(seed)),
            engine: crate::engine::EngineState::default(),
        }
    }

    /// Removes the top card of `player`'s library and puts it in hand.
    /// Returns `None` (no state change) if the library is empty.
    pub fn draw_card(&mut self, player: PlayerId) -> Option<ObjectId> {
        let id = {
            let ps = &mut self.players[player.index()];
            if ps.library.is_empty() {
                return None;
            }
            let id = ps.library.remove(0);
            ps.hand.push(id);
            id
        };
        self.transfer_library_knowledge_to_hand(player, 0, id);
        self.note_library_removal(player, 0);
        self.clear_object_relations(id);
        let turn = self.turn;
        let new_generation = {
            let object = self.objects.get_mut(id);
            object.zone = Zone::Hand;
            object.zone_change_count += 1;
            object
                .v4
                .reset_for_zone_change(object.card_def, Zone::Hand, turn);
            object.zone_change_count
        };
        for observer in [PlayerId::P0, PlayerId::P1] {
            for entry in &mut self.hand_knowledge[observer.index()][player.index()] {
                if entry.object == id {
                    entry.zone_change_count = new_generation;
                }
            }
        }
        Some(id)
    }

    pub fn known_hand_cards(&self, observer: PlayerId, owner: PlayerId) -> &[HandKnowledgeEntry] {
        &self.hand_knowledge[observer.index()][owner.index()]
    }

    /// Records a revealed hand identity for exactly one observer. The entry
    /// is incarnation-bound and stored in arena-id order for deterministic
    /// snapshots and observation hashes.
    pub fn reveal_hand_card(
        &mut self,
        observer: PlayerId,
        owner: PlayerId,
        object: ObjectId,
    ) -> Result<(), String> {
        let live = self
            .players
            .get(owner.index())
            .is_some_and(|player| player.hand.contains(&object));
        let Some(card) = self.objects.try_get(object) else {
            return Err(format!("cannot reveal missing hand object {object}"));
        };
        if !live || card.owner != owner || card.zone != Zone::Hand {
            return Err(format!("{object} is not a live card in {owner:?}'s hand"));
        }
        if observer == owner {
            // `own_hand` already carries the complete private hand. Keeping a
            // second copy would add no information and creates two facts that
            // future hand mutation would have to reconcile.
            return Ok(());
        }
        let entries = &mut self.hand_knowledge[observer.index()][owner.index()];
        if entries.iter().any(|entry| entry.object == object) {
            return Ok(());
        }
        entries.push(HandKnowledgeEntry {
            object,
            zone_change_count: card.zone_change_count,
        });
        entries.sort_by_key(|entry| entry.object);
        Ok(())
    }

    fn transfer_library_knowledge_to_hand(
        &mut self,
        owner: PlayerId,
        position: usize,
        object: ObjectId,
    ) {
        let generation = self.objects.get(object).zone_change_count;
        for observer in [PlayerId::P0, PlayerId::P1] {
            if observer == owner {
                continue;
            }
            if self.library_knowledge[observer.index()][owner.index()]
                .iter()
                .any(|entry| {
                    entry.position as usize == position
                        && entry.object == object
                        && entry.zone_change_count == generation
                })
            {
                let entries = &mut self.hand_knowledge[observer.index()][owner.index()];
                if !entries.iter().any(|entry| entry.object == object) {
                    entries.push(HandKnowledgeEntry {
                        object,
                        zone_change_count: generation,
                    });
                    entries.sort_by_key(|entry| entry.object);
                }
            }
        }
    }

    pub(crate) fn forget_hand_object(&mut self, object: ObjectId) {
        for observer in [PlayerId::P0, PlayerId::P1] {
            for owner in [PlayerId::P0, PlayerId::P1] {
                self.hand_knowledge[observer.index()][owner.index()]
                    .retain(|entry| entry.object != object);
            }
        }
    }

    /// Clears exact identities another observer knew in `owner`'s hand.
    /// A private subset choice from the whole hand invalidates even facts
    /// about cards that remain: retaining them would reveal which hidden
    /// candidates were moved by elimination.
    pub(crate) fn clear_nonowner_hand_knowledge(&mut self, owner: PlayerId) {
        for observer in [PlayerId::P0, PlayerId::P1] {
            if observer != owner {
                self.hand_knowledge[observer.index()][owner.index()].clear();
            }
        }
    }

    /// Clears zone-incarnation relations from the moving object and all
    /// reverse references to it. This is called before/while zone-change
    /// generation advances so an attachment or exile provenance link can
    /// never silently reconnect to a later incarnation.
    pub(crate) fn clear_object_relations(&mut self, object: ObjectId) {
        for (_, candidate) in self.objects.iter_mut() {
            candidate.attachments.retain(|&attached| attached != object);
            if candidate
                .v4
                .attached_to
                .is_some_and(|link| link.object == object)
            {
                candidate.v4.attached_to = None;
                let definition = &crate::card_def::CARD_DEFS[candidate.card_def as usize];
                if definition.bestow.is_some() {
                    let mut subtype_ids = definition
                        .subtypes
                        .iter()
                        .map(|subtype| subtype.stable_id())
                        .collect::<Vec<_>>();
                    subtype_ids.sort_unstable();
                    subtype_ids.dedup();
                    candidate.v4.effective_subtype_ids = subtype_ids;
                }
            }
            if candidate
                .v4
                .exiled_by
                .is_some_and(|link| link.object == object)
            {
                candidate.v4.exiled_by = None;
            }
        }
        let moving = self.objects.get_mut(object);
        moving.attachments.clear();
        moving.v4.attached_to = None;
        moving.v4.exiled_by = None;
    }

    /// Checks the exact two-way attachment graph. Restored state must never
    /// accept a dangling, stale, duplicate, or asymmetric relation.
    pub fn validate_attachment_relations(&self) -> Result<(), ObjectId> {
        for (id, object) in self.objects.iter() {
            if object.zone != Zone::Battlefield
                && (object.v4.attached_to.is_some() || !object.attachments.is_empty())
            {
                return Err(id);
            }
            if let Some(link) = object.v4.attached_to {
                let Some(host) = self.objects.try_get(link.object) else {
                    return Err(id);
                };
                if object.zone != Zone::Battlefield
                    || host.zone != Zone::Battlefield
                    || host.zone_change_count != link.zone_change_count
                    || host
                        .attachments
                        .iter()
                        .filter(|&&source| source == id)
                        .count()
                        != 1
                {
                    return Err(id);
                }
            }
            for (index, &source) in object.attachments.iter().enumerate() {
                if object.attachments[..index].contains(&source) {
                    return Err(id);
                }
                let Some(attachment) = self.objects.try_get(source) else {
                    return Err(id);
                };
                if object.zone != Zone::Battlefield
                    || attachment.zone != Zone::Battlefield
                    || attachment.v4.attached_to
                        != Some(ObjectLinkV4 {
                            object: id,
                            zone_change_count: object.zone_change_count,
                        })
                {
                    return Err(source);
                }
            }
        }
        Ok(())
    }

    /// Reattaches one exact source incarnation to one exact host incarnation.
    /// The caller owns target legality; this helper owns only the generic
    /// two-way attachment contract.
    pub(crate) fn attach_object_exact(
        &mut self,
        source: ObjectId,
        source_zone_change_count: u32,
        target: ObjectId,
        target_zone_change_count: u32,
    ) -> Result<(), ObjectId> {
        self.validate_attachment_relations()?;
        if source == target {
            return Err(source);
        }
        let Some(source_live) = self.objects.try_get(source) else {
            return Err(source);
        };
        let Some(target_live) = self.objects.try_get(target) else {
            return Err(target);
        };
        if source_live.zone != Zone::Battlefield
            || source_live.zone_change_count != source_zone_change_count
            || target_live.zone != Zone::Battlefield
            || target_live.zone_change_count != target_zone_change_count
        {
            return Err(source);
        }
        for (_, candidate) in self.objects.iter_mut() {
            candidate.attachments.retain(|&attached| attached != source);
        }
        self.objects.get_mut(source).v4.attached_to = None;
        self.objects.get_mut(source).v4.attached_to = Some(ObjectLinkV4 {
            object: target,
            zone_change_count: target_zone_change_count,
        });
        let attachments = &mut self.objects.get_mut(target).attachments;
        attachments.push(source);
        attachments.sort_unstable();
        attachments.dedup();
        self.validate_attachment_relations()
    }

    /// Returns the acting observer's currently valid, position-sorted facts
    /// about `owner`'s library. Callers should not inspect a different
    /// observer's row while producing a perspective-limited observation.
    pub fn known_library_cards(
        &self,
        observer: PlayerId,
        owner: PlayerId,
    ) -> &[LibraryKnowledgeEntry] {
        &self.library_knowledge[observer.index()][owner.index()]
    }

    /// Records that `observer` looked at the first `count` cards of `owner`'s
    /// library in their current order. Existing knowledge below that prefix is
    /// retained because revealing a prefix does not randomize the rest.
    pub fn reveal_library_top(&mut self, observer: PlayerId, owner: PlayerId, count: usize) {
        let count = count.min(self.players[owner.index()].library.len());
        let mut entries = self.library_knowledge[observer.index()][owner.index()].clone();
        entries.retain(|entry| entry.position as usize >= count);
        for position in 0..count {
            let object = self.players[owner.index()].library[position];
            entries.push(LibraryKnowledgeEntry {
                position: position as u32,
                object,
                zone_change_count: self.objects.get(object).zone_change_count,
            });
        }
        entries.sort_by_key(|entry| entry.position);
        self.library_knowledge[observer.index()][owner.index()] = entries;
    }

    /// Records one exact, publicly determined library position without
    /// revealing any identities above or below it. The caller must already
    /// have applied any insertion/removal position shifts.
    pub(crate) fn reveal_library_position(
        &mut self,
        observer: PlayerId,
        owner: PlayerId,
        position: usize,
    ) {
        let object = *self.players[owner.index()]
            .library
            .get(position)
            .expect("revealed library position must exist");
        let zone_change_count = self.objects.get(object).zone_change_count;
        let entries = &mut self.library_knowledge[observer.index()][owner.index()];
        entries.retain(|entry| {
            entry.position as usize != position
                && !(entry.object == object && entry.zone_change_count == zone_change_count)
        });
        entries.push(LibraryKnowledgeEntry {
            position: position as u32,
            object,
            zone_change_count,
        });
        entries.sort_by_key(|entry| entry.position);
    }

    /// Reorders exactly the top `ordered.len()` cards. The supplied ids must
    /// be a permutation of the current prefix. Observers in `revealed_to`
    /// learn the resulting order; everyone else loses facts inside the
    /// changed prefix while retaining facts below it.
    pub fn reorder_library_top(
        &mut self,
        owner: PlayerId,
        ordered: &[ObjectId],
        revealed_to: &[PlayerId],
    ) -> Result<(), String> {
        let count = ordered.len();
        let library = &self.players[owner.index()].library;
        if count > library.len() {
            return Err(format!(
                "cannot reorder {count} cards in a library of {}",
                library.len()
            ));
        }
        let mut expected = library[..count].to_vec();
        let mut actual = ordered.to_vec();
        expected.sort_unstable();
        actual.sort_unstable();
        if actual != expected {
            return Err("reordered library prefix is not an exact permutation".to_string());
        }

        self.players[owner.index()].library[..count].copy_from_slice(ordered);
        for observer in [PlayerId::P0, PlayerId::P1] {
            let knows_result = revealed_to.contains(&observer);
            let entries = &mut self.library_knowledge[observer.index()][owner.index()];
            entries.retain(|entry| entry.position as usize >= count);
            if knows_result {
                for (position, &object) in ordered.iter().enumerate() {
                    entries.push(LibraryKnowledgeEntry {
                        position: position as u32,
                        object,
                        zone_change_count: self.objects.get(object).zone_change_count,
                    });
                }
                entries.sort_by_key(|entry| entry.position);
            }
        }
        Ok(())
    }

    /// Atomically applies one private scry result to an exact, incarnation-
    /// bound library prefix. `retained_top` is top-to-bottom and
    /// `ordered_bottom` is shallow-to-deep (the last element becomes the
    /// physical bottom card). The untouched tail retains its order.
    ///
    /// Every validation runs before either the library or perspective-scoped
    /// knowledge changes. The owner learns the exact resulting retained top
    /// and ordered bottom. Other observers lose facts about an ambiguous
    /// multi-card private prefix while facts in the untouched tail shift up
    /// by the number of cards moved to the bottom. A one-card scry is
    /// deterministic once its public keep/bottom branch is known, so an
    /// observer's pre-existing exact fact for that card is preserved at its
    /// new position.
    ///
    /// This primitive intentionally emits no SCRY, SCRY_TO_BOTTOM, or SCRIED
    /// event. Full replacement/trigger event hooks remain deferred until a
    /// supported replacement or trigger observes them; callers must not
    /// synthesize a partial public event sequence around this private atomic
    /// transition. Higher-count scry also requires XMage-order bottom
    /// commitment before any retained-top ordering.
    pub(crate) fn apply_scry_result(
        &mut self,
        owner: PlayerId,
        expected_prefix: &[ObjectLinkV4],
        retained_top: &[ObjectId],
        ordered_bottom: &[ObjectId],
    ) -> Result<(), String> {
        let library = &self.players[owner.index()].library;
        let prefix_len = expected_prefix.len();
        if prefix_len > library.len() {
            return Err(format!(
                "scry prefix of {prefix_len} exceeds live library length {}",
                library.len()
            ));
        }
        for (position, expected) in expected_prefix.iter().enumerate() {
            if library[position] != expected.object {
                return Err("scry-bound library prefix changed order or identity".to_string());
            }
            let object = self.objects.try_get(expected.object).ok_or_else(|| {
                format!("scry-bound object {} no longer exists", expected.object.0)
            })?;
            if object.owner != owner
                || object.zone != Zone::Library
                || object.zone_change_count != expected.zone_change_count
            {
                return Err(format!(
                    "scry-bound object {} changed owner, zone, or incarnation",
                    expected.object.0
                ));
            }
        }

        let mut expected_objects = expected_prefix
            .iter()
            .map(|binding| binding.object)
            .collect::<Vec<_>>();
        let mut result_objects = retained_top
            .iter()
            .chain(ordered_bottom)
            .copied()
            .collect::<Vec<_>>();
        expected_objects.sort_unstable();
        result_objects.sort_unstable();
        if result_objects != expected_objects {
            return Err(
                "scry retained-top and ordered-bottom groups do not partition the bound prefix"
                    .to_string(),
            );
        }

        let library_len = library.len();
        let bottom_count = ordered_bottom.len();
        let untouched_tail = library[prefix_len..].to_vec();
        let mut result_library = Vec::with_capacity(library_len);
        result_library.extend_from_slice(retained_top);
        result_library.extend_from_slice(&untouched_tail);
        result_library.extend_from_slice(ordered_bottom);
        debug_assert_eq!(result_library.len(), library_len);

        let mut result_knowledge: [Vec<LibraryKnowledgeEntry>; 2] =
            std::array::from_fn(|_| Vec::new());
        for observer in [PlayerId::P0, PlayerId::P1] {
            let old = &self.library_knowledge[observer.index()][owner.index()];
            let updated = &mut result_knowledge[observer.index()];

            // Untouched-tail facts remain exact and simply move shallower by
            // one slot for every bound-prefix card moved to the bottom.
            updated.extend(old.iter().filter_map(|entry| {
                if entry.position as usize >= prefix_len {
                    Some(LibraryKnowledgeEntry {
                        position: entry.position - bottom_count as u32,
                        object: entry.object,
                        zone_change_count: entry.zone_change_count,
                    })
                } else {
                    None
                }
            }));

            if observer == owner {
                for (position, &object) in retained_top.iter().enumerate() {
                    updated.push(LibraryKnowledgeEntry {
                        position: position as u32,
                        object,
                        zone_change_count: self.objects.get(object).zone_change_count,
                    });
                }
                let bottom_start = library_len - bottom_count;
                for (offset, &object) in ordered_bottom.iter().enumerate() {
                    updated.push(LibraryKnowledgeEntry {
                        position: (bottom_start + offset) as u32,
                        object,
                        zone_change_count: self.objects.get(object).zone_change_count,
                    });
                }
            } else if prefix_len == 1 {
                // With one bound card, the public keep-vs-bottom branch fixes
                // its resulting position. Preserve the fact only if this
                // observer actually knew that exact original incarnation.
                let expected = expected_prefix[0];
                if old.iter().any(|entry| {
                    entry.position == 0
                        && entry.object == expected.object
                        && entry.zone_change_count == expected.zone_change_count
                }) {
                    updated.push(LibraryKnowledgeEntry {
                        position: if bottom_count == 1 {
                            (library_len - 1) as u32
                        } else {
                            0
                        },
                        object: expected.object,
                        zone_change_count: expected.zone_change_count,
                    });
                }
            }
            updated.sort_by_key(|entry| entry.position);
        }

        self.players[owner.index()].library = result_library;
        for observer in [PlayerId::P0, PlayerId::P1] {
            self.library_knowledge[observer.index()][owner.index()] =
                std::mem::take(&mut result_knowledge[observer.index()]);
        }
        Ok(())
    }

    /// Read-only preflight of one library shuffle in either randomness mode.
    /// Validates the owner exactly, snapshots the complete legacy RNG or
    /// derives the v2 substream for the owner's committed ordinal (checking
    /// the successor before derivation), and mutates nothing.
    pub(crate) fn preflight_library_shuffle(
        &self,
        owner: PlayerId,
    ) -> Result<LibraryShuffleToken, LibraryShuffleError> {
        use crate::environment_randomization_v2 as env2;
        let owner = library_shuffle_owner(owner)?;
        let authorization = match &self.randomness {
            GameRandomnessState::Legacy(rng) => {
                LibraryShuffleAuthorization::Legacy { expected_rng: *rng }
            }
            GameRandomnessState::EnvironmentV2(v2) => {
                let physical_owner = physical_owner_v2_exact(owner)?;
                let ordinal = v2.next_live_shuffle_ordinal(physical_owner);
                let next_ordinal = ordinal
                    .checked_add(1)
                    .ok_or(LibraryShuffleError::ExhaustedOwnerOrdinal)?;
                let derived_seed = env2::derive_environment_randomization_seed_v2(
                    v2.pair_environment_seed(),
                    physical_owner,
                    env2::ShufflePurposeV2::InGameLibraryShuffle,
                    ordinal,
                )?;
                LibraryShuffleAuthorization::EnvironmentV2 {
                    pair_root: v2.pair_environment_seed(),
                    ordinal,
                    next_ordinal,
                    derived_seed,
                }
            }
        };
        Ok(LibraryShuffleToken {
            owner,
            authorization,
        })
    }

    /// Token-validated commit of one library shuffle. Rechecks the caller
    /// owner, mode, and complete authorization against the state (legacy: the
    /// entire expected RNG; v2: root, the selected owner's committed ordinal,
    /// the checked successor, and a freshly derived seed) before any
    /// mutation, then computes the shuffled library and successor randomness
    /// locally and assigns library, randomness, and knowledge atomically.
    /// Only the selected owner's counter is checked, so P0 and P1 tokens
    /// preflighted from the same state commit in either order.
    pub(crate) fn commit_library_shuffle(
        &mut self,
        owner: PlayerId,
        token: LibraryShuffleToken,
    ) -> Result<(), LibraryShuffleError> {
        use crate::environment_randomization_v2 as env2;
        let owner = library_shuffle_owner(owner)?;
        if token.owner != owner {
            return Err(LibraryShuffleError::CallerOwnerMismatch);
        }
        match (&self.randomness, &token.authorization) {
            (
                GameRandomnessState::Legacy(rng),
                LibraryShuffleAuthorization::Legacy { expected_rng },
            ) => {
                if rng != expected_rng {
                    return Err(LibraryShuffleError::TokenStateMismatch);
                }
                let mut next_rng = *expected_rng;
                let mut library = self.players[owner.index()].library.clone();
                for i in (1..library.len()).rev() {
                    let j = (next_rng.next_u64() % (i as u64 + 1)) as usize;
                    library.swap(i, j);
                }
                self.players[owner.index()].library = library;
                self.randomness = GameRandomnessState::Legacy(next_rng);
                self.clear_library_knowledge(owner);
                Ok(())
            }
            (
                GameRandomnessState::EnvironmentV2(v2),
                LibraryShuffleAuthorization::EnvironmentV2 {
                    pair_root,
                    ordinal,
                    next_ordinal,
                    derived_seed,
                },
            ) => {
                let physical_owner = physical_owner_v2_exact(owner)?;
                if v2.pair_environment_seed() != *pair_root {
                    return Err(LibraryShuffleError::TokenStateMismatch);
                }
                let current = v2.next_live_shuffle_ordinal(physical_owner);
                if current != *ordinal {
                    return Err(LibraryShuffleError::TokenStateMismatch);
                }
                let expected_successor = current
                    .checked_add(1)
                    .ok_or(LibraryShuffleError::ExhaustedOwnerOrdinal)?;
                if expected_successor != *next_ordinal {
                    return Err(LibraryShuffleError::TokenStateMismatch);
                }
                let recomputed = env2::derive_environment_randomization_seed_v2(
                    *pair_root,
                    physical_owner,
                    env2::ShufflePurposeV2::InGameLibraryShuffle,
                    current,
                )?;
                if recomputed != *derived_seed {
                    return Err(LibraryShuffleError::TokenStateMismatch);
                }
                let mut library = self.players[owner.index()].library.clone();
                env2::shuffle_slice_in_place_v2(recomputed, &mut library);
                // Frozen local-computation rule: build the successor
                // randomness on a local clone of the validated current v2
                // state, then perform only infallible assignments.
                let mut successor_randomness = v2.clone();
                successor_randomness.set_live_shuffle_ordinal(physical_owner, expected_successor);
                self.players[owner.index()].library = library;
                self.randomness = GameRandomnessState::EnvironmentV2(successor_randomness);
                self.clear_library_knowledge(owner);
                Ok(())
            }
            _ => Err(LibraryShuffleError::TokenStateMismatch),
        }
    }

    /// The checked one-shot: preflight and commit one library shuffle in the
    /// state's randomness mode. Legacy states produce exactly the historical
    /// shuffle and RNG succession; v2 states consume exactly one owner
    /// ordinal. Fallible: callers must propagate the error. Crate-private
    /// because no external caller requires it; effect frames are the only
    /// consumers.
    pub(crate) fn shuffle_library(&mut self, owner: PlayerId) -> Result<(), LibraryShuffleError> {
        let token = self.preflight_library_shuffle(owner)?;
        self.commit_library_shuffle(owner, token)
    }

    /// Read-only shared view of the legacy RNG; `None` on an environment-v2
    /// state. There is deliberately no mutable RNG accessor: randomness
    /// advances only through the library-shuffle transaction. Same-module
    /// tests that genuinely need to perturb the RNG mutate the private enum
    /// directly.
    pub fn legacy_rng(&self) -> Option<&SplitMix64> {
        match &self.randomness {
            GameRandomnessState::Legacy(rng) => Some(rng),
            GameRandomnessState::EnvironmentV2(_) => None,
        }
    }

    /// Read-only shared view of the environment-v2 randomness state; `None`
    /// on a legacy state.
    pub fn environment_randomization_v2(
        &self,
    ) -> Option<&crate::environment_randomization_v2::GameEnvironmentRandomizationV2> {
        match &self.randomness {
            GameRandomnessState::Legacy(_) => None,
            GameRandomnessState::EnvironmentV2(v2) => Some(v2),
        }
    }

    /// Explicit v2 constructor: identical zone construction to the legacy
    /// `new_from_libraries`, but the randomness state is the environment-v2
    /// pair root with both live-shuffle ordinals at zero and no generic RNG.
    /// The mode is never inferred: callers choose this constructor
    /// explicitly with libraries already shuffled by the v2 initial-shuffle
    /// substreams.
    pub fn new_from_libraries_environment_v2(
        library0: &[u16],
        library1: &[u16],
        card_name: impl Fn(u16) -> String,
        pair_environment_seed: u64,
    ) -> GameState {
        // Thin wrapper, same discipline as `new_from_libraries` above.
        Self::new_from_libraries_environment_v2_with_starting_player_v1(
            library0,
            library1,
            card_name,
            pair_environment_seed,
            PlayerId::P0,
        )
    }

    /// Opt-in sibling of [`GameState::new_from_libraries_environment_v2`]
    /// taking an explicit starting player. Composes the starting-player
    /// authority with the environment-randomization-v2 construction path, so
    /// a P1-first request against an envrand-v2 game does not silently fall
    /// back to P0-first (`P1-METAMORPHIC-AUDIT-DESIGN-V4.md` Section 1.2's
    /// downstream-assumption audit, first bullet).
    pub fn new_from_libraries_environment_v2_with_starting_player_v1(
        library0: &[u16],
        library1: &[u16],
        card_name: impl Fn(u16) -> String,
        pair_environment_seed: u64,
        starting_player: PlayerId,
    ) -> GameState {
        let mut state = GameState::new_from_libraries_with_starting_player_v1(
            library0,
            library1,
            card_name,
            0,
            starting_player,
        );
        state.randomness = GameRandomnessState::EnvironmentV2(
            crate::environment_randomization_v2::GameEnvironmentRandomizationV2::new(
                pair_environment_seed,
            ),
        );
        state
    }

    /// Clears all observers' facts about one library. Used for shuffles and
    /// other whole-library randomization; known-position insertions/removals
    /// instead shift every still-valid fact precisely.
    pub(crate) fn clear_library_knowledge(&mut self, owner: PlayerId) {
        for observer in [PlayerId::P0, PlayerId::P1] {
            self.library_knowledge[observer.index()][owner.index()].clear();
        }
    }

    /// Updates position facts after a card at a publicly determined library
    /// position leaves. Top-card draws, mills, and impulse exile all use this
    /// exact shift operation.
    pub(crate) fn note_library_removal(&mut self, owner: PlayerId, position: usize) {
        for observer in [PlayerId::P0, PlayerId::P1] {
            let entries = &mut self.library_knowledge[observer.index()][owner.index()];
            entries.retain(|entry| entry.position as usize != position);
            for entry in entries {
                if entry.position as usize > position {
                    entry.position -= 1;
                }
            }
        }
    }

    /// Updates exact position facts after an identity-hidden insertion at a
    /// known library position. The inserted card is not learned here, but
    /// every previously known card at or below the insertion remains known
    /// and shifts one slot deeper. A caller with explicit visibility may
    /// reveal the inserted prefix after the zone change commits.
    pub(crate) fn note_library_insertion(&mut self, owner: PlayerId, position: usize) {
        for observer in [PlayerId::P0, PlayerId::P1] {
            for entry in &mut self.library_knowledge[observer.index()][owner.index()] {
                if entry.position as usize >= position {
                    entry.position = entry
                        .position
                        .checked_add(1)
                        .expect("a live library position fits u32");
                }
            }
        }
    }

    /// Placeholder zone transition (hand -> battlefield) used by tests and
    /// future step logic; no mana cost / land-drop accounting here.
    pub fn move_hand_to_battlefield(&mut self, player: PlayerId, id: ObjectId) -> bool {
        let ps = &mut self.players[player.index()];
        let Some(pos) = ps.hand.iter().position(|&h| h == id) else {
            return false;
        };
        ps.hand.remove(pos);
        ps.battlefield.push(id);
        self.forget_hand_object(id);
        self.clear_object_relations(id);
        let turn = self.turn;
        let enters_battlefield_tapped = crate::card_def::CARD_DEFS
            [self.objects.get(id).card_def as usize]
            .enters_battlefield_tapped;
        let obj = self.objects.get_mut(id);
        obj.zone = Zone::Battlefield;
        obj.tapped = enters_battlefield_tapped;
        obj.summoning_sick = true;
        obj.damage = 0;
        obj.counters = Counters::default();
        obj.attachments.clear();
        obj.zone_change_count += 1;
        obj.v4
            .reset_for_zone_change(obj.card_def, Zone::Battlefield, turn);
        true
    }

    /// Fast in-process FNV-1a over Rust's derived `Hash`. This is used only for
    /// hot-path mutation/rollback checks and deliberately is not an artifact
    /// interchange contract; it may depend on target width and Rust hashing
    /// details.
    pub fn state_hash(&self) -> u64 {
        let mut hasher = Fnv1a64::new();
        self.hash(&mut hasher);
        hasher.finish()
    }

    /// Cross-platform privileged audit hash: FNV-1a-64 over deterministic
    /// compact JSON bytes for a versioned, full-state envelope. Struct field
    /// order, enum spellings, and stored `Vec`/array order are part of this
    /// versioned artifact contract.
    pub fn diagnostic_state_hash(&self) -> u64 {
        fnv1a64(&diagnostic_state_hash_bytes(self))
    }

    /// The exact frozen algorithm identity of `diagnostic_state_hash` for
    /// this state's randomness representation: the legacy v8 constant on a
    /// legacy state, the environment-v2 v9 constant on a v2 state.
    pub fn diagnostic_state_hash_algorithm(&self) -> &'static str {
        match &self.randomness {
            GameRandomnessState::Legacy(_) => DIAGNOSTIC_STATE_HASH_ALGORITHM,
            GameRandomnessState::EnvironmentV2(_) => DIAGNOSTIC_STATE_HASH_ALGORITHM_ENVIRONMENT_V2,
        }
    }
}

struct Fnv1a64 {
    state: u64,
}

impl Fnv1a64 {
    fn new() -> Fnv1a64 {
        Fnv1a64 {
            state: 0xcbf29ce484222325,
        }
    }
}

impl Hasher for Fnv1a64 {
    fn finish(&self) -> u64 {
        self.state
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.state ^= u64::from(byte);
            self.state = self.state.wrapping_mul(0x100000001b3);
        }
    }
}

#[derive(Serialize)]
struct DiagnosticStateHashEnvelopeV8<'a> {
    schema_version: u32,
    state: &'a GameState,
}

/// Separately typed v9 identity for environment-v2 states. Final Pauper-pool
/// Initiative, Undercity, Bestow, X, and chosen-cost bindings change the full
/// serialized state for both randomness modes, so legacy advances from v7 to
/// v8 and environment-v2 advances from v8 to v9.
pub const DIAGNOSTIC_STATE_HASH_ALGORITHM_ENVIRONMENT_V2: &str =
    "fnv1a64-serde-json-game-state-envelope-v9";
pub const DIAGNOSTIC_STATE_HASH_ENVELOPE_SCHEMA_VERSION_ENVIRONMENT_V2: u32 = 9;

#[derive(Serialize)]
struct DiagnosticStateHashEnvelopeV9<'a> {
    schema_version: u32,
    state: &'a GameState,
}

fn diagnostic_state_hash_bytes(state: &GameState) -> Vec<u8> {
    match &state.randomness {
        GameRandomnessState::Legacy(_) => serde_json::to_vec(&DiagnosticStateHashEnvelopeV8 {
            schema_version: DIAGNOSTIC_STATE_HASH_ENVELOPE_SCHEMA_VERSION,
            state,
        })
        .expect("GameState diagnostic hash envelope must serialize"),
        GameRandomnessState::EnvironmentV2(_) => {
            serde_json::to_vec(&DiagnosticStateHashEnvelopeV9 {
                schema_version: DIAGNOSTIC_STATE_HASH_ENVELOPE_SCHEMA_VERSION_ENVIRONMENT_V2,
                state,
            })
            .expect("GameState v9 diagnostic hash envelope must serialize")
        }
    }
}

/// `serde(default = ...)` for `GameState::starting_player`: reproduces the
/// legacy hardcoded-`P0` literal so a missing field on deserialize (every
/// pre-existing serialized state) resolves to the exact old behavior.
fn default_starting_player_v1() -> PlayerId {
    PlayerId::P0
}

/// `serde(skip_serializing_if = ...)` for `GameState::starting_player`: omits
/// the field from the wire entirely when it is the legacy default, so a
/// P0-first game serializes with exactly the pre-change byte sequence (no
/// new JSON key appears).
fn starting_player_is_p0_v1(starting_player: &PlayerId) -> bool {
    *starting_player == PlayerId::P0
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut state = 0xcbf29ce484222325;
    for &byte in bytes {
        state ^= u64::from(byte);
        state = state.wrapping_mul(0x100000001b3);
    }
    state
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_card_libraries() -> (Vec<u16>, Vec<u16>) {
        (vec![1, 2, 3], vec![4, 5, 6, 7])
    }

    fn debug_names(card_def: u16) -> String {
        format!("card-{card_def}")
    }

    #[test]
    fn new_from_libraries_assigns_ids_p0_first() {
        let (lib0, lib1) = two_card_libraries();
        let state = GameState::new_from_libraries(&lib0, &lib1, debug_names, 42);

        assert_eq!(state.objects.len(), 7);
        assert_eq!(
            state.players[0].library,
            vec![ObjectId(0), ObjectId(1), ObjectId(2)]
        );
        assert_eq!(
            state.players[1].library,
            vec![ObjectId(3), ObjectId(4), ObjectId(5), ObjectId(6)]
        );
        assert_eq!(state.objects.get(ObjectId(0)).card_def, 1);
        assert_eq!(state.objects.get(ObjectId(3)).card_def, 4);
    }

    #[test]
    fn draw_removes_top_of_library_and_appends_to_hand() {
        let (lib0, lib1) = two_card_libraries();
        let mut state = GameState::new_from_libraries(&lib0, &lib1, debug_names, 1);

        let drawn = state.draw_card(PlayerId::P0).unwrap();
        assert_eq!(drawn, ObjectId(0)); // was index 0 = top
        assert_eq!(state.players[0].library, vec![ObjectId(1), ObjectId(2)]);
        assert_eq!(state.players[0].hand, vec![ObjectId(0)]);
        assert_eq!(state.objects.get(ObjectId(0)).zone, Zone::Hand);

        let drawn2 = state.draw_card(PlayerId::P0).unwrap();
        assert_eq!(drawn2, ObjectId(1));
        assert_eq!(state.players[0].hand, vec![ObjectId(0), ObjectId(1)]);
    }

    #[test]
    fn library_knowledge_is_perspective_scoped_and_draw_shifts_positions() {
        let (lib0, lib1) = two_card_libraries();
        let mut state = GameState::new_from_libraries(&lib0, &lib1, debug_names, 1);

        state.reveal_library_top(PlayerId::P0, PlayerId::P0, 3);
        assert_eq!(
            state
                .known_library_cards(PlayerId::P0, PlayerId::P0)
                .iter()
                .map(|entry| (entry.position, entry.object))
                .collect::<Vec<_>>(),
            vec![(0, ObjectId(0)), (1, ObjectId(1)), (2, ObjectId(2))]
        );
        assert!(state
            .known_library_cards(PlayerId::P1, PlayerId::P0)
            .is_empty());

        assert_eq!(state.draw_card(PlayerId::P0), Some(ObjectId(0)));
        assert_eq!(
            state
                .known_library_cards(PlayerId::P0, PlayerId::P0)
                .iter()
                .map(|entry| (entry.position, entry.object))
                .collect::<Vec<_>>(),
            vec![(0, ObjectId(1)), (1, ObjectId(2))]
        );
    }

    #[test]
    fn reorder_reveals_only_to_named_observers_and_shuffle_clears_everyone() {
        let (lib0, lib1) = two_card_libraries();
        let mut state = GameState::new_from_libraries(&lib0, &lib1, debug_names, 7);
        state.reveal_library_top(PlayerId::P0, PlayerId::P0, 3);
        state.reveal_library_top(PlayerId::P1, PlayerId::P0, 3);

        state
            .reorder_library_top(
                PlayerId::P0,
                &[ObjectId(2), ObjectId(0), ObjectId(1)],
                &[PlayerId::P0],
            )
            .unwrap();
        assert_eq!(
            state.players[0].library,
            vec![ObjectId(2), ObjectId(0), ObjectId(1)]
        );
        assert_eq!(
            state
                .known_library_cards(PlayerId::P0, PlayerId::P0)
                .iter()
                .map(|entry| entry.object)
                .collect::<Vec<_>>(),
            vec![ObjectId(2), ObjectId(0), ObjectId(1)]
        );
        assert!(state
            .known_library_cards(PlayerId::P1, PlayerId::P0)
            .is_empty());

        state
            .shuffle_library(PlayerId::P0)
            .expect("legacy shuffle succeeds");
        assert!(state
            .known_library_cards(PlayerId::P0, PlayerId::P0)
            .is_empty());
        assert!(state
            .known_library_cards(PlayerId::P1, PlayerId::P0)
            .is_empty());
    }

    #[test]
    fn invalid_library_reorder_is_rejected_without_mutation() {
        let (lib0, lib1) = two_card_libraries();
        let mut state = GameState::new_from_libraries(&lib0, &lib1, debug_names, 1);
        let before = state.clone();
        assert!(state
            .reorder_library_top(PlayerId::P0, &[ObjectId(0), ObjectId(0)], &[PlayerId::P0])
            .is_err());
        assert_eq!(state, before);
    }

    #[test]
    fn draw_from_empty_library_is_none_and_noop() {
        let mut state = GameState::new_from_libraries(&[], &[1], debug_names, 1);
        assert_eq!(state.draw_card(PlayerId::P0), None);
        assert!(state.players[0].hand.is_empty());
    }

    #[test]
    fn object_id_stable_across_multiple_zone_moves() {
        let (lib0, lib1) = two_card_libraries();
        let mut state = GameState::new_from_libraries(&lib0, &lib1, debug_names, 7);

        let id = state.draw_card(PlayerId::P0).unwrap();
        assert_eq!(id, ObjectId(0));
        assert_eq!(state.objects.get(id).zone, Zone::Hand);

        let moved = state.move_hand_to_battlefield(PlayerId::P0, id);
        assert!(moved);
        assert_eq!(id, ObjectId(0)); // same id throughout: library -> hand -> battlefield
        assert_eq!(state.objects.get(id).zone, Zone::Battlefield);
        assert!(state.players[0].battlefield.contains(&id));
        assert!(!state.players[0].hand.contains(&id));
        assert!(!state.players[0].library.contains(&id));
    }

    #[test]
    fn physical_target_contract_rejects_a_forged_copy_origin_in_a_later_generation() {
        let (lib0, lib1) = two_card_libraries();
        let mut state = GameState::new_from_libraries(&lib0, &lib1, debug_names, 7);
        let parent = ObjectId(0);
        let target = ObjectId(1);
        state.players[PlayerId::P0.index()]
            .library
            .retain(|&object| object != target);
        state.players[PlayerId::P0.index()].battlefield.push(target);
        state.objects.get_mut(target).zone = Zone::Battlefield;
        let contract = StackTargetContractV4::capture(&state, Target::Object(target));
        assert!(stack_target_contract_is_structurally_valid(
            &state,
            TargetSpec::AnyTarget,
            0,
            Target::Object(target),
            contract,
        ));

        state.objects.get_mut(target).zone_change_count += 1;
        assert!(stack_target_contract_is_structurally_valid(
            &state,
            TargetSpec::AnyTarget,
            0,
            Target::Object(target),
            contract,
        ));

        let parent_object = state.objects.get(parent).clone();
        state.objects.get_mut(target).spell_copy_origin = Some(SpellCopyOriginV4 {
            parent,
            parent_card_def: state.objects.get(target).card_def,
            parent_owner: parent_object.owner,
            parent_controller: parent_object.controller,
            parent_stack_zone_change_count: parent_object.zone_change_count,
            parent_was_copy: false,
        });
        assert!(
            !stack_target_contract_is_structurally_valid(
                &state,
                TargetSpec::AnyTarget,
                0,
                Target::Object(target),
                contract,
            ),
            "copy provenance is immutable for the arena object, even across later generations"
        );
    }

    #[test]
    fn state_hash_is_deterministic_for_identical_sequences() {
        let (lib0, lib1) = two_card_libraries();
        let mut a = GameState::new_from_libraries(&lib0, &lib1, debug_names, 99);
        let mut b = GameState::new_from_libraries(&lib0, &lib1, debug_names, 99);

        a.draw_card(PlayerId::P0);
        a.draw_card(PlayerId::P1);
        b.draw_card(PlayerId::P0);
        b.draw_card(PlayerId::P1);

        assert_eq!(a, b);
        assert_eq!(a.state_hash(), b.state_hash());
        assert_eq!(a.diagnostic_state_hash(), b.diagnostic_state_hash());
    }

    /// Builds the canonical legacy capture state: the exact state the frozen
    /// diagnostic golden uses (two-card libraries, seed 99, one draw each).
    fn legacy_capture_state() -> GameState {
        let (lib0, lib1) = two_card_libraries();
        let mut state = GameState::new_from_libraries(&lib0, &lib1, debug_names, 99);
        state.draw_card(PlayerId::P0);
        state.draw_card(PlayerId::P1);
        state
    }

    /// Gate 1 of environment-v2 step 2: the sealed historical bytes. The
    /// captured v5 GameState JSON and diagnostic envelope remain immutable,
    /// while additive state fields may make the current serializer emit a
    /// successor representation. The historical JSON must still deserialize
    /// to the same semantic legacy state under defaults.
    #[test]
    fn legacy_randomization_bytes_are_sealed() {
        let artifact_text = crate::environment_randomization_v2::LEGACY_STATE_BYTES_V1;
        {
            use sha2::Digest as _;
            let mut hasher = sha2::Sha256::new();
            hasher.update(artifact_text.as_bytes());
            let observed = hasher
                .finalize()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>();
            assert_eq!(
                observed,
                crate::environment_randomization_v2::LEGACY_STATE_BYTES_SHA256_V1,
                "sealed legacy-byte artifact does not match its pinned SHA-256"
            );
        }
        let artifact: serde_json::Value =
            serde_json::from_str(artifact_text).expect("decode legacy byte artifact");
        let object = artifact.as_object().expect("artifact is an object");
        let mut keys: Vec<&str> = object.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            [
                "diagnostic_envelope_json",
                "diagnostic_hash_hex",
                "schema",
                "state_hash_hex",
                "state_json",
            ],
            "sealed artifact must carry exactly the five frozen keys"
        );
        for key in object.keys() {
            assert!(
                object[key].is_string(),
                "sealed artifact value {key} must be a JSON string"
            );
        }
        assert_eq!(
            artifact["schema"].as_str().expect("schema"),
            "mtg-kernel-legacy-randomization-bytes/v1"
        );
        let historical_state_json = artifact["state_json"].as_str().expect("state_json");
        let restored: GameState =
            serde_json::from_str(historical_state_json).expect("historical state remains readable");
        assert_eq!(
            restored,
            legacy_capture_state(),
            "historical defaults no longer recover the canonical legacy state"
        );
        let historical_envelope =
            format!(r#"{{"schema_version":5,"state":{historical_state_json}}}"#);
        assert_eq!(
            historical_envelope,
            artifact["diagnostic_envelope_json"]
                .as_str()
                .expect("envelope json"),
            "sealed v5 diagnostic envelope bytes changed"
        );
        assert_eq!(
            format!("{:016x}", fnv1a64(historical_envelope.as_bytes())),
            artifact["diagnostic_hash_hex"].as_str().expect("hash hex"),
            "sealed v5 diagnostic hash changed"
        );
        assert_eq!(
            artifact["state_hash_hex"].as_str().expect("state hash hex"),
            "edb76411c96ff4ee",
            "sealed historical hot-path hash changed"
        );
        assert_eq!(
            artifact["diagnostic_hash_hex"].as_str().expect("hash hex"),
            "865030b60d413489"
        );
    }

    fn v2_capture_state(root: u64) -> GameState {
        let (lib0, lib1) = two_card_libraries();
        let mut state =
            GameState::new_from_libraries_environment_v2(&lib0, &lib1, debug_names, root);
        state.draw_card(PlayerId::P0);
        state.draw_card(PlayerId::P1);
        state
    }

    const V2_FRAGMENT_7: &str = "\"environment_randomization_v2\":{\"pair_environment_seed\":7,\"next_live_shuffle_ordinal\":[0,0]}";

    fn v2_json_with_randomness(fragment: &str) -> String {
        let v2_json = serde_json::to_string(&v2_capture_state(7)).expect("v2 serializes");
        assert!(
            v2_json.contains(V2_FRAGMENT_7),
            "compact [p0,p1] wire bytes must appear verbatim"
        );
        v2_json.replacen(V2_FRAGMENT_7, fragment, 1)
    }

    fn assert_rejected(json: &str, label: &str) {
        assert!(
            serde_json::from_str::<GameState>(json).is_err(),
            "{label} must be rejected"
        );
    }

    #[test]
    fn randomness_representation_rejects_hybrid_and_unknown_forms() {
        let legacy_json =
            serde_json::to_string(&legacy_capture_state()).expect("legacy state serializes");
        let rng_key = "\"rng\":";
        let rng_start = legacy_json.find(rng_key).expect("legacy rng key present");
        let rng_end = legacy_json[rng_start..]
            .find("},")
            .map(|offset| rng_start + offset + 1)
            .expect("legacy rng object terminates");
        let rng_fragment = &legacy_json[rng_start..rng_end];

        assert_rejected(
            &legacy_json.replacen(rng_fragment, &format!("{rng_fragment},{V2_FRAGMENT_7}"), 1),
            "both tags, legacy first",
        );
        assert_rejected(
            &legacy_json.replacen(rng_fragment, &format!("{V2_FRAGMENT_7},{rng_fragment}"), 1),
            "both tags, v2 first",
        );
        assert_rejected(
            &legacy_json.replacen(rng_fragment, &format!("{rng_fragment},{rng_fragment}"), 1),
            "duplicate legacy tag",
        );
        assert_rejected(
            &v2_json_with_randomness(&format!("{V2_FRAGMENT_7},{V2_FRAGMENT_7}")),
            "duplicate valid v2 tag",
        );
        // Duplicate v2 tags with two distinct valid payloads, both orders.
        const V2_FRAGMENT_7_ADVANCED: &str = "\"environment_randomization_v2\":{\"pair_environment_seed\":7,\"next_live_shuffle_ordinal\":[1,0]}";
        assert_rejected(
            &v2_json_with_randomness(&format!("{V2_FRAGMENT_7},{V2_FRAGMENT_7_ADVANCED}")),
            "distinct duplicate v2 tags, base first",
        );
        assert_rejected(
            &v2_json_with_randomness(&format!("{V2_FRAGMENT_7_ADVANCED},{V2_FRAGMENT_7}")),
            "distinct duplicate v2 tags, advanced first",
        );
        let v2_json = serde_json::to_string(&v2_capture_state(7)).expect("v2 serializes");
        assert_rejected(
            &v2_json.replacen(V2_FRAGMENT_7, &format!("{V2_FRAGMENT_7},{rng_fragment}"), 1),
            "v2 then legacy tag",
        );
        assert_rejected(
            &legacy_json.replacen(&format!("{rng_fragment},"), "", 1),
            "neither tag",
        );
        assert_rejected(
            &legacy_json.replacen(rng_fragment, &format!("{rng_fragment},\"garbage\":1"), 1),
            "unknown outer key after the randomness tag",
        );
        assert_rejected(
            &legacy_json.replacen(rng_fragment, &format!("\"garbage\":1,{rng_fragment}"), 1),
            "unknown outer key before the randomness tag",
        );
        assert_rejected(
            &v2_json_with_randomness(&format!("{V2_FRAGMENT_7},\"garbage\":1")),
            "unknown outer key after the v2 tag",
        );
        assert_rejected(
            &v2_json_with_randomness(&format!("\"garbage\":1,{V2_FRAGMENT_7}")),
            "unknown outer key before the v2 tag",
        );
        // The nested legacy RNG object is strict: exactly one `state` member.
        let padded_rng = rng_fragment.replacen('}', ",\"junk\":1}", 1);
        assert_rejected(
            &legacy_json.replacen(rng_fragment, &padded_rng, 1),
            "extra nested legacy RNG member",
        );
        assert_rejected(
            &v2_json_with_randomness("\"environment_randomization_v2\":null"),
            "null v2 payload",
        );
        assert_rejected(
            &legacy_json.replacen(rng_fragment, "\"rng\":null", 1),
            "null legacy payload",
        );
        assert!(serde_json::from_str::<GameState>(&legacy_json).is_ok());
        assert!(serde_json::from_str::<GameState>(&v2_json).is_ok());
    }

    #[test]
    fn v2_nested_serde_matrix() {
        for (fragment, label) in [
            (
                "\"environment_randomization_v2\":{\"next_live_shuffle_ordinal\":[0,0]}",
                "missing root",
            ),
            (
                "\"environment_randomization_v2\":{\"pair_environment_seed\":7}",
                "missing ordinal array",
            ),
            (
                "\"environment_randomization_v2\":{\"pair_environment_seed\":7,\"next_live_shuffle_ordinal\":[0,0],\"extra\":1}",
                "extra nested key",
            ),
            (
                "\"environment_randomization_v2\":{\"pair_environment_seed\":7,\"next_live_shuffle_ordinal\":[]}",
                "ordinal array length 0",
            ),
            (
                "\"environment_randomization_v2\":{\"pair_environment_seed\":7,\"next_live_shuffle_ordinal\":[0]}",
                "ordinal array length 1",
            ),
            (
                "\"environment_randomization_v2\":{\"pair_environment_seed\":7,\"next_live_shuffle_ordinal\":[0,0,0]}",
                "ordinal array length 3",
            ),
            (
                "\"environment_randomization_v2\":{\"pair_environment_seed\":7,\"next_live_shuffle_ordinal\":0}",
                "scalar in place of array",
            ),
            (
                "\"environment_randomization_v2\":{\"pair_environment_seed\":7,\"next_live_shuffle_ordinal\":{}}",
                "object in place of array",
            ),
            (
                "\"environment_randomization_v2\":{\"pair_environment_seed\":7,\"next_live_shuffle_ordinal\":null}",
                "null in place of array",
            ),
            (
                "\"environment_randomization_v2\":{\"pair_environment_seed\":7,\"next_live_shuffle_ordinal\":[-1,0]}",
                "negative ordinal",
            ),
            (
                "\"environment_randomization_v2\":{\"pair_environment_seed\":18446744073709551616,\"next_live_shuffle_ordinal\":[0,0]}",
                "root greater than u64",
            ),
            (
                "\"environment_randomization_v2\":{\"pair_environment_seed\":-1,\"next_live_shuffle_ordinal\":[0,0]}",
                "negative root",
            ),
            (
                "\"environment_randomization_v2\":{\"pair_environment_seed\":7,\"next_live_shuffle_ordinal\":[18446744073709551616,0]}",
                "ordinal greater than u64",
            ),
            (
                "\"environment_randomization_v2\":{\"pair_environment_seed\":null,\"next_live_shuffle_ordinal\":[0,0]}",
                "null root",
            ),
            (
                "\"environment_randomization_v2\":{\"pair_environment_seed\":7,\"pair_environment_seed\":7,\"next_live_shuffle_ordinal\":[0,0]}",
                "duplicate nested key",
            ),
        ] {
            assert_rejected(&v2_json_with_randomness(fragment), label);
        }

        // Serialized extremes: root u64::MAX round-trips; stored ordinal
        // u64::MAX deserializes and only preflight rejects it.
        let max_root = v2_json_with_randomness(
            "\"environment_randomization_v2\":{\"pair_environment_seed\":18446744073709551615,\"next_live_shuffle_ordinal\":[0,0]}",
        );
        let state: GameState = serde_json::from_str(&max_root).expect("root u64::MAX is legal");
        assert_eq!(
            state
                .environment_randomization_v2()
                .expect("v2")
                .pair_environment_seed(),
            u64::MAX
        );
    }

    #[test]
    fn v2_exhausted_and_near_exhausted_ordinals() {
        use crate::environment_randomization_v2 as env2;
        let exhausted_json = v2_json_with_randomness(
            "\"environment_randomization_v2\":{\"pair_environment_seed\":7,\"next_live_shuffle_ordinal\":[18446744073709551615,0]}",
        );
        let exhausted: GameState =
            serde_json::from_str(&exhausted_json).expect("stored ordinal u64::MAX deserializes");
        let before_json = serde_json::to_string(&exhausted).expect("serializes");
        let before_state_hash = exhausted.state_hash();
        let before_diag = exhausted.diagnostic_state_hash();
        assert_eq!(
            exhausted
                .preflight_library_shuffle(PlayerId::P0)
                .map(|_| ()),
            Err(LibraryShuffleError::ExhaustedOwnerOrdinal),
            "exhausted ordinal must fail at preflight"
        );
        assert_eq!(
            serde_json::to_string(&exhausted).expect("serializes"),
            before_json,
            "failed preflight must be byte-exact nonmutating"
        );
        assert_eq!(exhausted.state_hash(), before_state_hash);
        assert_eq!(exhausted.diagnostic_state_hash(), before_diag);
        // P1 remains fully usable while P0 is exhausted.
        let mut usable = exhausted.clone();
        let token = usable
            .preflight_library_shuffle(PlayerId::P1)
            .expect("P1 preflight");
        usable
            .commit_library_shuffle(PlayerId::P1, token)
            .expect("P1 commit");

        // max-minus-one advances to max exactly once, then exhausts.
        let near_json = v2_json_with_randomness(
            "\"environment_randomization_v2\":{\"pair_environment_seed\":7,\"next_live_shuffle_ordinal\":[18446744073709551614,0]}",
        );
        let mut near: GameState = serde_json::from_str(&near_json).expect("deserializes");
        let token = near
            .preflight_library_shuffle(PlayerId::P0)
            .expect("max-minus-one preflight");
        near.commit_library_shuffle(PlayerId::P0, token)
            .expect("max-minus-one commit");
        assert_eq!(
            near.environment_randomization_v2()
                .expect("v2")
                .next_live_shuffle_ordinal(env2::PhysicalOwnerV2::P0),
            u64::MAX
        );
        assert_eq!(
            near.preflight_library_shuffle(PlayerId::P0).map(|_| ()),
            Err(LibraryShuffleError::ExhaustedOwnerOrdinal)
        );
    }

    #[test]
    fn v2_variant_and_counters_survive_clone_and_serde() {
        use crate::environment_randomization_v2::PhysicalOwnerV2;
        let mut state = v2_capture_state(940_001);
        let token = state
            .preflight_library_shuffle(PlayerId::P1)
            .expect("preflight succeeds on v2");
        state
            .commit_library_shuffle(PlayerId::P1, token)
            .expect("commit succeeds");
        let cloned = state.clone();
        assert_eq!(state, cloned);
        let json = serde_json::to_string(&state).expect("v2 serializes");
        let restored: GameState = serde_json::from_str(&json).expect("v2 deserializes");
        assert_eq!(state, restored);
        let v2 = restored
            .environment_randomization_v2()
            .expect("restored state is v2");
        assert_eq!(v2.next_live_shuffle_ordinal(PhysicalOwnerV2::P0), 0);
        assert_eq!(v2.next_live_shuffle_ordinal(PhysicalOwnerV2::P1), 1);
        assert_eq!(v2.pair_environment_seed(), 940_001);
    }

    /// Serialized bytes, hot hash, and diagnostic hash of one state, taken
    /// before an expected-failure transaction attempt.
    fn state_fingerprint(state: &GameState) -> (String, u64, u64) {
        (
            serde_json::to_string(state).expect("state serializes"),
            state.state_hash(),
            state.diagnostic_state_hash(),
        )
    }

    fn assert_fingerprint_unchanged(
        state: &GameState,
        fingerprint: &(String, u64, u64),
        label: &str,
    ) {
        assert_eq!(
            serde_json::to_string(state).expect("state serializes"),
            fingerprint.0,
            "{label} must leave serialized bytes identical"
        );
        assert_eq!(
            state.state_hash(),
            fingerprint.1,
            "{label} must leave the hot state hash identical"
        );
        assert_eq!(
            state.diagnostic_state_hash(),
            fingerprint.2,
            "{label} must leave the diagnostic hash identical"
        );
    }

    #[test]
    fn library_shuffle_transaction_adversarial() {
        use crate::environment_randomization_v2 as env2;
        let mut state = v2_capture_state(31);
        let stale = state
            .preflight_library_shuffle(PlayerId::P0)
            .expect("first preflight");
        let fresh = state
            .preflight_library_shuffle(PlayerId::P0)
            .expect("second preflight at same ordinal");
        match &fresh.authorization {
            LibraryShuffleAuthorization::EnvironmentV2 { derived_seed, .. } => {
                assert_eq!(
                    *derived_seed,
                    env2::derive_environment_randomization_seed_v2(
                        31,
                        env2::PhysicalOwnerV2::P0,
                        env2::ShufflePurposeV2::InGameLibraryShuffle,
                        0,
                    )
                    .expect("module derivation"),
                    "preflight seed must equal the module KDF"
                );
            }
            LibraryShuffleAuthorization::Legacy { .. } => panic!("v2 token expected"),
        }
        state
            .commit_library_shuffle(PlayerId::P0, fresh)
            .expect("fresh commit");
        let fingerprint = state_fingerprint(&state);
        assert_eq!(
            state.commit_library_shuffle(PlayerId::P0, stale),
            Err(LibraryShuffleError::TokenStateMismatch),
            "stale token must fail after the ordinal advanced"
        );
        assert_fingerprint_unchanged(&state, &fingerprint, "replayed stale token");

        // Cross-root transplant.
        let donor = v2_capture_state(31);
        let transplant = donor
            .preflight_library_shuffle(PlayerId::P1)
            .expect("donor preflight");
        let mut other_root = v2_capture_state(32);
        let fingerprint = state_fingerprint(&other_root);
        assert_eq!(
            other_root.commit_library_shuffle(PlayerId::P1, transplant),
            Err(LibraryShuffleError::TokenStateMismatch),
            "cross-root transplant must fail"
        );
        assert_fingerprint_unchanged(&other_root, &fingerprint, "cross-root transplant");

        // Wrong caller owner.
        let token = donor
            .preflight_library_shuffle(PlayerId::P0)
            .expect("owner preflight");
        let mut donor_mut = donor.clone();
        let fingerprint = state_fingerprint(&donor_mut);
        assert_eq!(
            donor_mut.commit_library_shuffle(PlayerId::P1, token),
            Err(LibraryShuffleError::CallerOwnerMismatch)
        );
        assert_fingerprint_unchanged(&donor_mut, &fingerprint, "wrong caller owner");

        // Cross-mode in both directions.
        let legacy = legacy_capture_state();
        let legacy_token = legacy
            .preflight_library_shuffle(PlayerId::P0)
            .expect("legacy preflight");
        let mut v2_state = v2_capture_state(31);
        let fingerprint = state_fingerprint(&v2_state);
        assert_eq!(
            v2_state.commit_library_shuffle(PlayerId::P0, legacy_token),
            Err(LibraryShuffleError::TokenStateMismatch)
        );
        assert_fingerprint_unchanged(&v2_state, &fingerprint, "legacy token on v2 state");
        let v2_token = v2_state
            .preflight_library_shuffle(PlayerId::P0)
            .expect("v2 preflight");
        let mut legacy_mut = legacy.clone();
        let fingerprint = state_fingerprint(&legacy_mut);
        assert_eq!(
            legacy_mut.commit_library_shuffle(PlayerId::P0, v2_token),
            Err(LibraryShuffleError::TokenStateMismatch)
        );
        assert_fingerprint_unchanged(&legacy_mut, &fingerprint, "v2 token on legacy state");
    }

    #[test]
    fn library_shuffle_rejects_invalid_owner_nonmutating() {
        for mut state in [legacy_capture_state(), v2_capture_state(45)] {
            let fingerprint = state_fingerprint(&state);
            for forged in [PlayerId(2), PlayerId(255)] {
                assert_eq!(
                    state.preflight_library_shuffle(forged).map(|_| ()),
                    Err(LibraryShuffleError::InvalidOwner),
                    "preflight must reject PlayerId({forged:?}) exactly"
                );
                assert_eq!(
                    state.shuffle_library(forged),
                    Err(LibraryShuffleError::InvalidOwner)
                );
                let token = state
                    .preflight_library_shuffle(PlayerId::P0)
                    .expect("honest preflight");
                assert_eq!(
                    state.commit_library_shuffle(forged, token),
                    Err(LibraryShuffleError::InvalidOwner),
                    "commit must reject a forged owner before any indexing"
                );
                assert_fingerprint_unchanged(&state, &fingerprint, "invalid owner");
            }
        }
    }

    #[test]
    fn library_shuffle_rejects_tampered_and_stale_tokens_nonmutating() {
        let mut state = v2_capture_state(46);
        let fingerprint = state_fingerprint(&state);
        let honest = state
            .preflight_library_shuffle(PlayerId::P0)
            .expect("honest preflight");
        let (pair_root, ordinal, next_ordinal, derived_seed) = match honest.authorization {
            LibraryShuffleAuthorization::EnvironmentV2 {
                pair_root,
                ordinal,
                next_ordinal,
                derived_seed,
            } => (pair_root, ordinal, next_ordinal, derived_seed),
            LibraryShuffleAuthorization::Legacy { .. } => panic!("v2 token expected"),
        };
        for (tampered, label) in [
            (
                LibraryShuffleAuthorization::EnvironmentV2 {
                    pair_root,
                    ordinal,
                    next_ordinal: next_ordinal + 1,
                    derived_seed,
                },
                "successor tamper",
            ),
            (
                LibraryShuffleAuthorization::EnvironmentV2 {
                    pair_root,
                    ordinal,
                    next_ordinal,
                    derived_seed: derived_seed ^ 1,
                },
                "derived-seed tamper",
            ),
            (
                LibraryShuffleAuthorization::EnvironmentV2 {
                    pair_root,
                    ordinal: ordinal + 1,
                    next_ordinal: next_ordinal + 1,
                    derived_seed,
                },
                "future-ordinal tamper",
            ),
            (
                LibraryShuffleAuthorization::EnvironmentV2 {
                    pair_root: pair_root ^ 1,
                    ordinal,
                    next_ordinal,
                    derived_seed,
                },
                "root tamper",
            ),
        ] {
            let forged_token = LibraryShuffleToken {
                owner: PlayerId::P0,
                authorization: tampered,
            };
            assert_eq!(
                state.commit_library_shuffle(PlayerId::P0, forged_token),
                Err(LibraryShuffleError::TokenStateMismatch),
                "{label} must fail"
            );
            assert_fingerprint_unchanged(&state, &fingerprint, label);
        }

        // Stale legacy RNG: the RNG advances between preflight and commit.
        let mut legacy = legacy_capture_state();
        let token = legacy
            .preflight_library_shuffle(PlayerId::P0)
            .expect("legacy preflight");
        match &mut legacy.randomness {
            GameRandomnessState::Legacy(rng) => {
                rng.next_u64();
            }
            GameRandomnessState::EnvironmentV2(_) => panic!("capture state is legacy"),
        }
        let fingerprint = state_fingerprint(&legacy);
        assert_eq!(
            legacy.commit_library_shuffle(PlayerId::P0, token),
            Err(LibraryShuffleError::TokenStateMismatch),
            "a stale legacy RNG snapshot must fail the complete-RNG recheck"
        );
        assert_fingerprint_unchanged(&legacy, &fingerprint, "stale legacy RNG");
    }

    #[test]
    fn v2_commit_applies_module_permutation_and_scopes_knowledge() {
        use crate::environment_randomization_v2 as env2;
        let mut state = v2_capture_state(59);
        state.reveal_library_top(PlayerId::P0, PlayerId::P0, 1);
        state.reveal_library_top(PlayerId::P1, PlayerId::P0, 1);
        state.reveal_library_top(PlayerId::P1, PlayerId::P1, 1);
        let p1_own_knowledge =
            state.library_knowledge[PlayerId::P1.index()][PlayerId::P1.index()].clone();
        let original = state.players[PlayerId::P0.index()].library.clone();
        let derived = env2::derive_environment_randomization_seed_v2(
            59,
            env2::PhysicalOwnerV2::P0,
            env2::ShufflePurposeV2::InGameLibraryShuffle,
            0,
        )
        .expect("module derivation");
        let mut expected = original.clone();
        env2::shuffle_slice_in_place_v2(derived, &mut expected);
        state
            .shuffle_library(PlayerId::P0)
            .expect("v2 one-shot shuffle");
        assert_eq!(
            state.players[PlayerId::P0.index()].library,
            expected,
            "commit must apply exactly the module permutation for the committed ordinal"
        );
        for observer in [PlayerId::P0, PlayerId::P1] {
            assert!(
                state.library_knowledge[observer.index()][PlayerId::P0.index()].is_empty(),
                "a shuffle must clear every observer's facts about the shuffled library"
            );
        }
        assert_eq!(
            state.library_knowledge[PlayerId::P1.index()][PlayerId::P1.index()],
            p1_own_knowledge,
            "the other owner's library knowledge must survive"
        );
        let v2 = state.environment_randomization_v2().expect("v2 state");
        assert_eq!(
            v2.next_live_shuffle_ordinal(env2::PhysicalOwnerV2::P0),
            1,
            "exactly one P0 ordinal consumed"
        );
        assert_eq!(
            v2.next_live_shuffle_ordinal(env2::PhysicalOwnerV2::P1),
            0,
            "a P0 commit must not move P1's counter"
        );
    }

    #[test]
    fn legacy_one_shot_matches_historical_shuffle_exactly() {
        let mut transactional = legacy_capture_state();
        let mut manual = legacy_capture_state();
        assert_eq!(transactional, manual);
        transactional
            .shuffle_library(PlayerId::P0)
            .expect("legacy one-shot succeeds");
        // Historical algorithm replayed by hand on the twin state.
        {
            let mut rng = *manual.legacy_rng().expect("legacy");
            let library = &mut manual.players[PlayerId::P0.index()].library;
            for i in (1..library.len()).rev() {
                let j = (rng.next_u64() % (i as u64 + 1)) as usize;
                library.swap(i, j);
            }
            manual.randomness = GameRandomnessState::Legacy(rng);
            manual.clear_library_knowledge(PlayerId::P0);
        }
        assert_eq!(transactional, manual, "legacy behavior must be exact");
        assert_eq!(transactional.state_hash(), manual.state_hash());
    }

    #[test]
    fn v2_owner_tokens_commit_in_either_order() {
        let base = v2_capture_state(77);
        let mut forward = base.clone();
        let p0 = forward.preflight_library_shuffle(PlayerId::P0).expect("p0");
        let p1 = forward.preflight_library_shuffle(PlayerId::P1).expect("p1");
        forward
            .commit_library_shuffle(PlayerId::P0, p0)
            .expect("p0 first");
        forward
            .commit_library_shuffle(PlayerId::P1, p1)
            .expect("p1 second");
        let mut reverse = base.clone();
        let p0 = reverse.preflight_library_shuffle(PlayerId::P0).expect("p0");
        let p1 = reverse.preflight_library_shuffle(PlayerId::P1).expect("p1");
        reverse
            .commit_library_shuffle(PlayerId::P1, p1)
            .expect("p1 first");
        reverse
            .commit_library_shuffle(PlayerId::P0, p0)
            .expect("p0 second");
        assert_eq!(forward, reverse, "owner commits must commute");
        assert_eq!(forward.state_hash(), reverse.state_hash());
    }

    #[test]
    fn v2_short_libraries_advance_and_clear_knowledge() {
        use crate::environment_randomization_v2::PhysicalOwnerV2;
        for target_len in [0_usize, 1] {
            let mut state = v2_capture_state(88);
            while state.players[PlayerId::P0.index()].library.len() > target_len {
                state.players[PlayerId::P0.index()].library.pop();
            }
            if let Some(&object) = state.players[PlayerId::P0.index()].library.first() {
                for observer in [PlayerId::P0, PlayerId::P1] {
                    state.library_knowledge[observer.index()][PlayerId::P0.index()].push(
                        LibraryKnowledgeEntry {
                            position: 0,
                            object,
                            zone_change_count: state.objects.get(object).zone_change_count,
                        },
                    );
                }
            }
            let token = state
                .preflight_library_shuffle(PlayerId::P0)
                .expect("short-library preflight");
            state
                .commit_library_shuffle(PlayerId::P0, token)
                .expect("short-library commit");
            assert_eq!(
                state
                    .environment_randomization_v2()
                    .expect("v2")
                    .next_live_shuffle_ordinal(PhysicalOwnerV2::P0),
                1,
                "length-{target_len} shuffle must consume exactly one ordinal"
            );
            for observer in [PlayerId::P0, PlayerId::P1] {
                assert!(
                    state.library_knowledge[observer.index()][PlayerId::P0.index()].is_empty(),
                    "length-{target_len} shuffle must clear knowledge"
                );
            }
        }
    }

    #[test]
    fn v2_diagnostic_envelope_and_hash_sensitivity() {
        let base = v2_capture_state(11);
        let bytes = diagnostic_state_hash_bytes(&base);
        assert!(bytes.starts_with(b"{\"schema_version\":9,\"state\":{"));
        assert_eq!(
            DIAGNOSTIC_STATE_HASH_ALGORITHM_ENVIRONMENT_V2,
            "fnv1a64-serde-json-game-state-envelope-v9"
        );
        assert_eq!(
            DIAGNOSTIC_STATE_HASH_ENVELOPE_SCHEMA_VERSION_ENVIRONMENT_V2,
            9
        );
        let text = String::from_utf8(bytes).expect("ascii envelope");
        assert!(text.contains("\"environment_randomization_v2\":"));
        assert!(!text.contains("\"rng\":"));
        assert_ne!(
            legacy_capture_state().state_hash(),
            base.state_hash(),
            "legacy and v2 randomness must hash distinctly in the hot hash"
        );

        // Root, P0 ordinal, and P1 ordinal each independently change BOTH
        // the hot state hash and the diagnostic hash.
        let base_json = serde_json::to_string(&v2_capture_state(11)).expect("serializes");
        let fragment_11 = "\"environment_randomization_v2\":{\"pair_environment_seed\":11,\"next_live_shuffle_ordinal\":[0,0]}";
        assert!(base_json.contains(fragment_11));
        for (replacement, label) in [
            (
                "\"environment_randomization_v2\":{\"pair_environment_seed\":12,\"next_live_shuffle_ordinal\":[0,0]}",
                "root",
            ),
            (
                "\"environment_randomization_v2\":{\"pair_environment_seed\":11,\"next_live_shuffle_ordinal\":[1,0]}",
                "p0 ordinal",
            ),
            (
                "\"environment_randomization_v2\":{\"pair_environment_seed\":11,\"next_live_shuffle_ordinal\":[0,1]}",
                "p1 ordinal",
            ),
        ] {
            let variant: GameState =
                serde_json::from_str(&base_json.replacen(fragment_11, replacement, 1))
                    .expect("variant deserializes");
            assert_ne!(
                base.state_hash(),
                variant.state_hash(),
                "{label} must change the hot state hash"
            );
            assert_ne!(
                base.diagnostic_state_hash(),
                variant.diagnostic_state_hash(),
                "{label} must change the diagnostic hash"
            );
        }
    }

    #[test]
    fn diagnostic_state_hash_algorithm_dispatches_per_representation() {
        assert_eq!(
            legacy_capture_state().diagnostic_state_hash_algorithm(),
            "fnv1a64-serde-json-game-state-envelope-v8",
            "legacy states report the v8 algorithm"
        );
        assert_eq!(
            v2_capture_state(3).diagnostic_state_hash_algorithm(),
            "fnv1a64-serde-json-game-state-envelope-v9",
            "environment-v2 states report the v9 algorithm"
        );
    }

    #[test]
    fn diagnostic_state_hash_contract_and_golden_value_are_frozen() {
        let (lib0, lib1) = two_card_libraries();
        let mut state = GameState::new_from_libraries(&lib0, &lib1, debug_names, 99);
        state.draw_card(PlayerId::P0);
        state.draw_card(PlayerId::P1);

        assert_eq!(
            DIAGNOSTIC_STATE_HASH_ALGORITHM,
            "fnv1a64-serde-json-game-state-envelope-v8"
        );
        assert_eq!(DIAGNOSTIC_STATE_HASH_ENVELOPE_SCHEMA_VERSION, 8);
        assert!(
            diagnostic_state_hash_bytes(&state).starts_with(b"{\"schema_version\":8,\"state\":{")
        );
        assert_eq!(state.diagnostic_state_hash(), 0xa921_902d_e1a8_d8ce);
    }

    /// Draws to different players don't interact, so interleaving order
    /// across players is unobservable in the resulting state. This is the
    /// flip side of the "no unordered-map iteration" invariant: the only
    /// state each draw touches is `players[p].{library,hand}` and the
    /// touched object, so two draws to distinct players commute.
    #[test]
    fn state_hash_is_order_independent_across_distinct_players() {
        let (lib0, lib1) = two_card_libraries();
        let mut a = GameState::new_from_libraries(&lib0, &lib1, debug_names, 99);
        let mut b = GameState::new_from_libraries(&lib0, &lib1, debug_names, 99);

        a.draw_card(PlayerId::P0);
        a.draw_card(PlayerId::P1);
        b.draw_card(PlayerId::P1);
        b.draw_card(PlayerId::P0);

        assert_eq!(a, b);
        assert_eq!(a.state_hash(), b.state_hash());
    }

    #[test]
    fn state_hash_detects_a_genuine_state_difference() {
        let (lib0, lib1) = two_card_libraries();
        let mut a = GameState::new_from_libraries(&lib0, &lib1, debug_names, 99);
        let mut b = GameState::new_from_libraries(&lib0, &lib1, debug_names, 99);

        a.draw_card(PlayerId::P0);
        b.draw_card(PlayerId::P0);
        b.draw_card(PlayerId::P0); // b has one extra card drawn

        assert_ne!(a, b);
        assert_ne!(a.state_hash(), b.state_hash());
        assert_ne!(a.diagnostic_state_hash(), b.diagnostic_state_hash());
    }

    #[test]
    fn diagnostic_state_hash_includes_rng_and_pending_source_contract() {
        let (lib0, lib1) = two_card_libraries();
        let mut state = GameState::new_from_libraries(&lib0, &lib1, debug_names, 99);
        let initial = state.diagnostic_state_hash();
        match &mut state.randomness {
            GameRandomnessState::Legacy(rng) => {
                rng.next_u64();
            }
            GameRandomnessState::EnvironmentV2(_) => panic!("capture state is legacy"),
        }
        assert_ne!(
            state.diagnostic_state_hash(),
            initial,
            "RNG state is privileged full state"
        );

        let spell = state.players[0].library[0];
        let source_contract = StackSourceContractV4::capture(&state, spell, CastMethodV4::Normal);
        state.engine.pending_cast = Some(crate::engine::PendingCast {
            spell,
            source_contract,
            controller: PlayerId::P0,
            target_spec: crate::card_def::TargetSpec::None,
            targets_chosen: Vec::new(),
            target_contracts: Vec::new(),
            target_selection_finished: false,
            is_flashback: false,
            cast_mode: Some(crate::engine::CastMode::Normal),
            additional_cost_discarded: Some(Vec::new()),
            mode_chosen: Some(0),
            origin_zone: Zone::Hand,
            sacrifice_chosen: Vec::new(),
            kicked: Some(false),
            optional_additional_cost_paid: Some(false),
            optional_additional_cost_chosen: Vec::new(),
            optional_additional_cost_selection_finished: false,
            x_value: Some(0),
            chosen_creature_cost_zone: None,
            chosen_creature_cost: None,
        });
        let ordinary_contract = state.diagnostic_state_hash();
        state
            .engine
            .pending_cast
            .as_mut()
            .unwrap()
            .source_contract
            .zone_change_count += 1;
        assert_ne!(
            state.diagnostic_state_hash(),
            ordinary_contract,
            "the diagnostic full-state envelope must not skip the pending source contract"
        );
    }

    #[test]
    fn rng_stream_is_deterministic_per_seed() {
        let mut r1 = SplitMix64::seed(12345);
        let mut r2 = SplitMix64::seed(12345);
        for _ in 0..10 {
            assert_eq!(r1.next_u64(), r2.next_u64());
        }

        let mut r3 = SplitMix64::seed(6789);
        assert_ne!(r1.next_u64(), r3.next_u64());
    }
}
