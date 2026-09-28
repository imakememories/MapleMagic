//! Static card database. `CARD_DEFS` (and `card_id_by_name`,
//! `KERNEL_CARDDB_HASH`) are generated at build time by `build.rs` from
//! `data/cards_v1.json` -- see that file for the codegen and its
//! validation (duplicate names / empty deck coverage / schema-version
//! mismatch all fail the build).
//!
//! Every card in the pinned nine-deck Pauper pool carries a complete engine
//! program. Burn includes its alternate costs, Madness, Plot, landfall,
//! modal Blasts, creature abilities, and graveyard interaction. Relic of
//! Progenitus uses effect-level graveyard selection for its targeted
//! single-card exile and implements both activated abilities.
//! Definitions without an explicit registry `engine_capability` remain
//! `NoEffect`. Supported ordinary permanents and intrinsic basic-land mana
//! are generated from metadata only after that capability gate, so registry
//! metadata alone can never make a card playable.
//!
//! Mono Red Rally's 18 cards (6 shared with Burn: Lightning Bolt, Mountain,
//! Red Elemental Blast, Relic of Progenitus, Searing Blaze, Voldaren
//! Epicure) are implemented: Burning-Tree
//! Emissary (ETB mana), Chain Lightning (mandatory damage plus the complete
//! recursive pay/copy/retarget loop -- see `effect::EffectOp::
//! OfferAffectedPlayerSpellCopy` and `engine::PendingSpellCopy`), Clockwork
//! Percussionist
//! (haste + dies-trigger impulse draw), End the
//! Festivities (mass damage to the opponent + their creatures), Experimental
//! Synthesizer (ETB/leaves impulse draw + sac-for-a-token ability), Galvanic
//! Blast (Metalcraft), Goblin Bushwhacker (Kicker-gated team pump/haste),
//! Goblin Tomb Raider (static self-boost), Great Furnace (a second Mountain),
//! Rally at the Hornburg (tokens + Human haste), and Reckless Impulse
//! (impulse draw). Cast into the Fire implements both modes, including its
//! up-to-two artifact exile selection.

use crate::effect::{
    CreatureFilter, CreatureSacrificeFilter, EffectCond, EffectOp, ImpulseDuration,
    LibraryCardFilter, ObjectRef, PlayerRef, TargetRef,
};
use crate::mana::{Cost, ManaColor, ManaColorSetV1, Pip};
use crate::state::Zone;
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CardType {
    Land,
    Creature,
    Instant,
    Sorcery,
    Artifact,
    Enchantment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Supertype {
    Basic,
    Snow,
    /// Appended for exact nonlegendary target filtering. No current pool
    /// permanent is legendary, but the filter must remain correct as the
    /// registry grows.
    Legendary,
}

/// Fail-closed engine readiness for one registry definition. This is
/// generated from `cards_v1.json` and is deliberately independent of card
/// type: a fully supported land is executable even though it is played, not
/// cast; a token can be fully supported even though it can only be created
/// by another effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CardCapability {
    /// No executable program may be exposed by the engine.
    NoEffect,
    /// Some behavior is executable, but a full-deck preflight must reject
    /// the definition because a reachable branch is still unsupported.
    Partial,
    /// Every reachable in-scope branch is implemented.
    Full,
}

impl CardCapability {
    pub const fn is_executable(self) -> bool {
        !matches!(self, CardCapability::NoEffect)
    }

    pub const fn is_fully_supported(self) -> bool {
        matches!(self, CardCapability::Full)
    }
}

/// 105.1's subtype line, typed (per external review: "subtype queries
/// structured, typed access, not string contains"). A fully closed set --
/// one named variant per distinct subtype string across the whole
/// 136-definition pool -- rather than a named-variants-plus-string-fallback
/// design: `Subtype` is embedded in `effect::CreatureFilter` /
/// `effect::EffectOp`, which need to derive `Deserialize`, and a
/// `&'static str` payload (needed for a fallback variant to round-trip
/// arbitrary text) can't implement that (same reason `mana::Cost` doesn't
/// derive `Serialize`/`Deserialize` either -- see its own doc). A query
/// against a named variant (`Subtype::Human`) is a typed enum comparison
/// that can never silently match the wrong thing via a typo or a
/// differently-cased duplicate -- this pool's own JSON data has real
/// examples of the latter (`"Human"` and `"HUMAN"` on different cards,
/// likely an ingestion artifact upstream of this codegen, not a meaningful
/// distinction -- preserved as two distinct variants here rather than
/// silently merged, so this table stays a faithful mirror of the source
/// data). `build.rs::subtype_variant` panics on an unrecognized string,
/// same as `card_type_variant`/`supertype_variant`/`color_variant`, since
/// this is now a closed set the same way those are.
#[repr(u16)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Subtype {
    Ape,
    Aura,
    /// "BIRD" verbatim -- see the module doc's note on case-duplicated
    /// subtype strings.
    BirdAllCaps,
    Bird,
    Blood,
    Cat,
    Detective,
    Dragon,
    Drone,
    Druid,
    Eldrazi,
    Elf,
    Equipment,
    /// "FAERIE" verbatim -- see the module doc's note.
    FaerieAllCaps,
    Faerie,
    Food,
    Forest,
    Gate,
    Goblin,
    /// "HUMAN" verbatim -- see the module doc's note.
    HumanAllCaps,
    Hero,
    Human,
    Hydra,
    Island,
    Knight,
    /// "MONK" verbatim.
    Monk,
    /// "MOONFOLK" verbatim.
    Moonfolk,
    Monkey,
    Mountain,
    Myr,
    /// "NINJA" verbatim -- see the module doc's note.
    NinjaAllCaps,
    Ninja,
    Ouphe,
    Pirate,
    Plains,
    /// "ROGUE" verbatim -- see the module doc's note.
    RogueAllCaps,
    Ranger,
    Rat,
    Rogue,
    /// "SERPENT" verbatim.
    Serpent,
    Saga,
    Samurai,
    Shaman,
    Shapeshifter,
    Soldier,
    Spider,
    Spirit,
    Swamp,
    Toy,
    Treefolk,
    Vampire,
    /// "WIZARD" verbatim -- see the module doc's note.
    WizardAllCaps,
    Warrior,
    Wizard,
    Zombie,
    /// Appended for Bird Illusion Token. Existing stable ids are never
    /// renumbered when the closed pool gains another supported subtype.
    Illusion,
    /// Appended for Saruli Caretaker. Existing stable ids remain unchanged.
    Dryad,
    /// Appended for Tinder Wall and Wall of Roots.
    Plant,
    /// Appended for Overgrown Battlement, Tinder Wall, and Wall of Roots.
    Wall,
    /// Appended for Troll of Khazad-dum. Existing stable ids remain fixed.
    Troll,
    /// Appended for Healer of the Glade. Existing stable ids remain fixed.
    Elemental,
    /// Appended for Map Token. Existing stable ids remain fixed.
    Map,
    /// Appended for the reusable Treasure token created by Heap Gate.
    Treasure,
    /// Appended for Lotleth Giant. Existing stable ids remain fixed.
    Giant,
    /// Appended for the reusable Eldrazi Spawn token created by Writhing
    /// Chrysalis. Existing subtype ids remain unchanged.
    Spawn,
    /// Appended for Fume Spitter. Existing stable ids remain fixed.
    Phyrexian,
    /// Appended for Fume Spitter and Mesmeric Fiend.
    Horror,
    /// Appended for Mesmeric Fiend. Existing stable ids remain fixed.
    Nightmare,
    /// Appended for the reusable Clue token created by Investigate.
    /// Existing stable subtype ids remain unchanged.
    Clue,
    /// Appended for the 4/1 black Skeleton token created by Undercity's
    /// Catacombs room. Existing stable ids remain fixed.
    Skeleton,
}

impl Subtype {
    /// Every creature type represented by the checked-in pool, in stable-id
    /// order. Case-distinct registry spellings remain separate because their
    /// existing ids and subtype queries are intentionally preserved.
    pub const CREATURE_TYPES: &'static [Subtype] = &[
        Subtype::Ape,
        Subtype::BirdAllCaps,
        Subtype::Bird,
        Subtype::Cat,
        Subtype::Detective,
        Subtype::Dragon,
        Subtype::Drone,
        Subtype::Druid,
        Subtype::Eldrazi,
        Subtype::Elf,
        Subtype::FaerieAllCaps,
        Subtype::Faerie,
        Subtype::Goblin,
        Subtype::HumanAllCaps,
        Subtype::Hero,
        Subtype::Human,
        Subtype::Hydra,
        Subtype::Knight,
        Subtype::Monk,
        Subtype::Moonfolk,
        Subtype::Monkey,
        Subtype::Myr,
        Subtype::NinjaAllCaps,
        Subtype::Ninja,
        Subtype::Ouphe,
        Subtype::Pirate,
        Subtype::RogueAllCaps,
        Subtype::Ranger,
        Subtype::Rat,
        Subtype::Rogue,
        Subtype::Serpent,
        Subtype::Samurai,
        Subtype::Shaman,
        Subtype::Shapeshifter,
        Subtype::Soldier,
        Subtype::Spider,
        Subtype::Spirit,
        Subtype::Toy,
        Subtype::Treefolk,
        Subtype::Vampire,
        Subtype::WizardAllCaps,
        Subtype::Warrior,
        Subtype::Wizard,
        Subtype::Zombie,
        Subtype::Illusion,
        Subtype::Dryad,
        Subtype::Plant,
        Subtype::Wall,
        Subtype::Troll,
        Subtype::Elemental,
        Subtype::Giant,
        Subtype::Spawn,
        Subtype::Phyrexian,
        Subtype::Horror,
        Subtype::Nightmare,
    ];

    /// Schema-v4 observation id. Existing discriminants are append-only:
    /// feature encoders may sort and embed these ids without depending on
    /// source spelling or locale-sensitive string ordering.
    pub const fn stable_id(self) -> u16 {
        self as u16
    }

    /// Whether this closed-pool subtype is a creature type. Changeling
    /// materializes every true entry into an object's effective subtype set;
    /// card, artifact, enchantment, and land subtypes remain excluded.
    pub const fn is_creature_type(self) -> bool {
        matches!(
            self,
            Subtype::Ape
                | Subtype::BirdAllCaps
                | Subtype::Bird
                | Subtype::Cat
                | Subtype::Detective
                | Subtype::Dragon
                | Subtype::Drone
                | Subtype::Druid
                | Subtype::Eldrazi
                | Subtype::Elf
                | Subtype::FaerieAllCaps
                | Subtype::Faerie
                | Subtype::Goblin
                | Subtype::HumanAllCaps
                | Subtype::Hero
                | Subtype::Human
                | Subtype::Hydra
                | Subtype::Knight
                | Subtype::Monk
                | Subtype::Moonfolk
                | Subtype::Monkey
                | Subtype::Myr
                | Subtype::NinjaAllCaps
                | Subtype::Ninja
                | Subtype::Ouphe
                | Subtype::Pirate
                | Subtype::RogueAllCaps
                | Subtype::Ranger
                | Subtype::Rat
                | Subtype::Rogue
                | Subtype::Serpent
                | Subtype::Samurai
                | Subtype::Shaman
                | Subtype::Shapeshifter
                | Subtype::Soldier
                | Subtype::Spider
                | Subtype::Spirit
                | Subtype::Toy
                | Subtype::Treefolk
                | Subtype::Vampire
                | Subtype::WizardAllCaps
                | Subtype::Warrior
                | Subtype::Wizard
                | Subtype::Zombie
                | Subtype::Illusion
                | Subtype::Dryad
                | Subtype::Plant
                | Subtype::Wall
                | Subtype::Troll
                | Subtype::Elemental
                | Subtype::Giant
                | Subtype::Spawn
                | Subtype::Phyrexian
                | Subtype::Horror
                | Subtype::Nightmare
        )
    }
}

/// What a spell/ability needs targeted at cast/activation time.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TargetSpec {
    #[default]
    None,
    /// Exactly 1 target: any creature on either battlefield, or either
    /// player.
    AnyTarget,
    /// Exactly 2 targets, in order, the second dependent on the first
    /// (Searing Blaze): target any player, then target a creature *that
    /// player controls*. `engine::legal_targets_for`'s second-pick pool is
    /// computed from the first pick, not independently.
    PlayerThenTheirCreature,
    /// Exactly 1 target: any spell currently on the stack, regardless of
    /// card type or color (Counterspell; also Pyroblast's counter mode,
    /// whose color is checked at resolution rather than targeting -- see
    /// `EffectCond::TargetIsColor`).
    AnySpellOnStack,
    /// Exactly 1 target: an instant spell currently on the stack (Dispel).
    /// This filters by the targeted stack object's card definition, so both
    /// physical instant spells and instant spell copies qualify while
    /// activated/triggered abilities do not.
    InstantSpellOnStack,
    /// Exactly 1 target: a *blue* spell currently on the stack (Red
    /// Elemental Blast's counter mode -- color is filtered at targeting).
    BlueSpellOnStack,
    /// Exactly 1 target: any permanent on either battlefield (Pyroblast's
    /// destroy mode).
    AnyPermanent,
    /// Exactly 1 target: any *blue* permanent on either battlefield (Red
    /// Elemental Blast's destroy mode).
    BluePermanent,
    /// Exactly 1 target: either player. Unlike `AnyTarget`, this never
    /// offers creatures (Mental Note/Thought Scour style mill spells).
    /// Appended to preserve every existing variant's derived hash identity.
    AnyPlayer,
    /// Exactly 1 target: a *red* spell currently on the stack (Blue
    /// Elemental Blast's counter mode -- color is filtered at targeting).
    /// Appended so every pre-existing target-spec discriminant remains
    /// stable in snapshots and diagnostic hashes.
    RedSpellOnStack,
    /// Exactly 1 target: any *red* permanent on either battlefield (Blue
    /// Elemental Blast's destroy mode). Appended for the same identity
    /// reason as `RedSpellOnStack`.
    RedPermanent,
    /// Exactly 1 target: any nonland permanent on either battlefield
    /// (Deem Inferior). Appended so every pre-existing target-spec
    /// discriminant and serialized snapshot identity remains stable.
    NonlandPermanent,
    /// Exactly 1 target: any creature on either battlefield. Appended so
    /// every pre-existing target-spec discriminant and serialized snapshot
    /// identity remains stable.
    Creature,
    /// Exactly 1 target: a creature without the Legendary supertype.
    /// Appended for Cast Down without changing any earlier target identity.
    NonlegendaryCreature,
    /// Exactly 1 target: an artifact or enchantment spell on the stack
    /// (Annul). Appended so every pre-existing target-spec discriminant and
    /// serialized snapshot identity remains stable.
    ArtifactOrEnchantmentSpellOnStack,
    /// Exactly 1 target: a sorcery spell on the stack (Envelop). Appended
    /// for the same identity-preservation reason.
    SorcerySpellOnStack,
    /// Exactly 1 target: a noncreature spell on the stack (Spell Pierce).
    /// Type filtering happens at targeting time and is rechecked at
    /// resolution through the shared stack-target contract.
    NoncreatureSpellOnStack,
    /// Exactly 1 target: an artifact spell on the stack (Steel Sabotage's
    /// first mode).
    ArtifactSpellOnStack,
    /// Exactly 1 target: an artifact permanent on either battlefield (Steel
    /// Sabotage's second mode).
    ArtifactPermanent,
    /// Exactly 1 target: a creature or land card in either graveyard.
    /// Appended for Pulse of Murasa without changing any earlier target
    /// identity.
    CreatureOrLandCardInGraveyard,
    /// Exactly one creature controlled by the activating player. Map
    /// Token's Explore ability is the first consumer.
    ControlledCreature,
    /// Zero, one, or two creature cards in the activating player's own
    /// graveyard. Blood Fountain may legally announce none.
    UpToTwoCreatureCardsInOwnGraveyard,
    /// Zero, one, or two distinct battlefield creatures. Cast into the
    /// Fire's damage mode is the first consumer.
    UpToTwoCreatures,
    /// Exactly two distinct artifact permanents. Dust to Dust requires both
    /// targets to be announced even though either may later become illegal.
    ExactlyTwoArtifactPermanents,
    /// Exactly one enchantment permanent. Thraben Charm's second mode is the
    /// first consumer.
    EnchantmentPermanent,
    /// Zero, one, or two distinct players. The kernel is strictly two-player,
    /// so this is the complete bounded form of "any number of target players."
    UpToTwoPlayers,
    /// Exactly one creature card in the targeting player's own graveyard.
    /// Appended for Dread Return without changing any earlier target identity.
    CreatureCardInOwnGraveyard,
    /// Exactly the targeting player's opponent. Appended for targeted ETB
    /// abilities such as Lotleth Giant in this strictly two-player kernel.
    TargetOpponent,
    /// Exactly one creature controlled by an opponent of the announcing
    /// player. Humbling Elder is the first consumer.
    OpponentControlledCreature,
    /// Exactly one spell whose printed mana value is no greater than the
    /// number of permanents the announcing player controls with either of
    /// the named effective subtypes. The two-subtype form preserves the
    /// registry's case-distinct Faerie spellings without making the rules
    /// query card-name-specific.
    SpellManaValueAtMostControlledSubtypes {
        first: Subtype,
        second: Option<Subtype>,
    },
    /// Zero, one, or two distinct cards in either player's graveyard.
    /// Faerie Macabre may legally announce no target, and its two targets
    /// may come from different graveyards.
    UpToTwoCardsInGraveyards,
    /// Exactly one battlefield creature other than the targeting spell or
    /// ability's own source incarnation. Journey to Nowhere is the first
    /// consumer. The source-relative exclusion is applied by the caller so
    /// the stable target vocabulary remains card-name-neutral.
    CreatureOtherThanSource,
    /// Zero or one tapped creature on either battlefield. Cryogen Relic may
    /// activate without a target, but any selected target must be tapped.
    UpToOneTappedCreature,
    /// Exactly one noncreature artifact permanent. Gorilla Shaman derives X
    /// from this target's printed mana value.
    NoncreatureArtifactPermanent,
    /// Exactly one land permanent on either battlefield. Appended for
    /// Cleansing Wildfire without changing any earlier target identity.
    Land,
    /// Exactly one artifact or enchantment controlled by an opponent of the
    /// announcing player. Appended for Masked Vandal and Troublemaker Ouphe
    /// without changing any existing target identity.
    OpponentArtifactOrEnchantmentPermanent,
}

impl TargetSpec {
    /// Stable append-only identity used by tests and semantic tooling. This
    /// remains explicit now that the grammar includes a parameterized target
    /// filter and Rust no longer permits a direct enum-to-integer cast.
    pub const fn stable_id(self) -> u8 {
        match self {
            TargetSpec::None => 0,
            TargetSpec::AnyTarget => 1,
            TargetSpec::PlayerThenTheirCreature => 2,
            TargetSpec::AnySpellOnStack => 3,
            TargetSpec::InstantSpellOnStack => 4,
            TargetSpec::BlueSpellOnStack => 5,
            TargetSpec::AnyPermanent => 6,
            TargetSpec::BluePermanent => 7,
            TargetSpec::AnyPlayer => 8,
            TargetSpec::RedSpellOnStack => 9,
            TargetSpec::RedPermanent => 10,
            TargetSpec::NonlandPermanent => 11,
            TargetSpec::Creature => 12,
            TargetSpec::NonlegendaryCreature => 13,
            TargetSpec::ArtifactOrEnchantmentSpellOnStack => 14,
            TargetSpec::SorcerySpellOnStack => 15,
            TargetSpec::NoncreatureSpellOnStack => 16,
            TargetSpec::ArtifactSpellOnStack => 17,
            TargetSpec::ArtifactPermanent => 18,
            TargetSpec::CreatureOrLandCardInGraveyard => 19,
            TargetSpec::ControlledCreature => 20,
            TargetSpec::UpToTwoCreatureCardsInOwnGraveyard => 21,
            TargetSpec::UpToTwoCreatures => 22,
            TargetSpec::ExactlyTwoArtifactPermanents => 23,
            TargetSpec::EnchantmentPermanent => 24,
            TargetSpec::UpToTwoPlayers => 25,
            TargetSpec::CreatureCardInOwnGraveyard => 26,
            TargetSpec::TargetOpponent => 27,
            TargetSpec::OpponentControlledCreature => 28,
            TargetSpec::SpellManaValueAtMostControlledSubtypes { .. } => 29,
            TargetSpec::UpToTwoCardsInGraveyards => 30,
            TargetSpec::CreatureOtherThanSource => 31,
            TargetSpec::UpToOneTappedCreature => 32,
            TargetSpec::NoncreatureArtifactPermanent => 33,
            TargetSpec::Land => 34,
            TargetSpec::OpponentArtifactOrEnchantmentPermanent => 35,
        }
    }
}

/// Combat-relevant keyword abilities, as a bitset. Only `Flying`/`Reach`
/// (blocker legality) and `Haste` (summoning-sickness exemption) are
/// actually set by any card in this increment's pool (Sneaky Snacker,
/// Masked Meower); the rest exist so the shape is right the next time a
/// keyword-bearing card needs one -- in particular `FIRST_STRIKE`/
/// `DOUBLE_STRIKE` back the two-wave combat-damage hook in `engine.rs`
/// even though nothing in Mono-Red Burn has first strike.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash, Serialize, Deserialize)]
pub struct Keywords(pub u32);

impl Keywords {
    pub const NONE: Keywords = Keywords(0);
    pub const FLYING: Keywords = Keywords(1 << 0);
    pub const REACH: Keywords = Keywords(1 << 1);
    pub const HASTE: Keywords = Keywords(1 << 2);
    pub const VIGILANCE: Keywords = Keywords(1 << 3);
    pub const TRAMPLE: Keywords = Keywords(1 << 4);
    pub const FIRST_STRIKE: Keywords = Keywords(1 << 5);
    pub const DOUBLE_STRIKE: Keywords = Keywords(1 << 6);
    pub const DEATHTOUCH: Keywords = Keywords(1 << 7);
    pub const MENACE: Keywords = Keywords(1 << 8);
    pub const DEFENDER: Keywords = Keywords(1 << 9);
    pub const LIFELINK: Keywords = Keywords(1 << 10);
    pub const HEXPROOF: Keywords = Keywords(1 << 11);
    pub const INDESTRUCTIBLE: Keywords = Keywords(1 << 12);
    pub const PROTECTION_FROM_MONOCOLORED: Keywords = Keywords(1 << 13);
    pub const ISLANDWALK: Keywords = Keywords(1 << 14);
    /// The permanent spell may be cast whenever its controller has
    /// priority, using the same timing permission as an instant.
    pub const FLASH: Keywords = Keywords(1 << 15);

    pub const fn has(self, other: Keywords) -> bool {
        self.0 & other.0 != 0
    }
}

/// Stable W/U/B/R/G/C bit positions used by schema-v4 object colors,
/// landwalk, and color-selection contracts.
pub const fn mana_color_mask(color: ManaColor) -> u8 {
    match color {
        ManaColor::W => 1 << 0,
        ManaColor::U => 1 << 1,
        ManaColor::B => 1 << 2,
        ManaColor::R => 1 << 3,
        ManaColor::G => 1 << 4,
        ManaColor::C => 1 << 5,
    }
}

pub fn mana_colors_mask(colors: &[ManaColor]) -> u8 {
    colors
        .iter()
        .fold(0, |mask, &color| mask | mana_color_mask(color))
}

impl std::ops::BitOr for Keywords {
    type Output = Keywords;
    fn bitor(self, rhs: Keywords) -> Keywords {
        Keywords(self.0 | rhs.0)
    }
}

/// Typed filter for a chosen permanent paid as an activation cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermanentFilterDef {
    /// A permanent with both the Land card type and the named effective
    /// subtype. Control is checked independently from the ownership-oriented
    /// battlefield vectors.
    LandWithSubtype(Subtype),
    /// A permanent with the Creature card type and the named effective
    /// color. Control and untapped status are checked independently by the
    /// cost component that consumes this filter.
    CreatureWithColor(ManaColor),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermanentFilter {
    /// A controlled permanent whose definition has either the Artifact or
    /// Creature card type. Artifact creatures and artifact lands match once.
    ArtifactOrCreature,
    /// A controlled artifact permanent. Artifact creatures and artifact
    /// lands match, while permanents without the Artifact type do not.
    Artifact,
    /// A controlled permanent with the Creature card type. Appended for
    /// Dread Return's flashback cost.
    Creature,
}

/// One component of a composite cost. Composable (a real cost is `&'static
/// [CostComponent]`) rather than card-shaped, matching the `EffectOp`
/// philosophy in `effect.rs`: "sacrifice 2 Mountains" is
/// `SacrificeLands(2)`, not a `FireblastCost` variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostComponent {
    /// Tap the source permanent (activated abilities only -- casting a
    /// spell has no source permanent to tap).
    Tap,
    /// Sacrifice the source permanent itself.
    SacrificeSelf,
    /// Exile the source permanent/card itself.
    ExileSelf,
    /// Discard the source card itself from its owner's hand. Unlike
    /// `DiscardCards`, this is not an interactive choice: typecycling and
    /// ordinary Cycling bind the activating card as the cost's exact object.
    /// Kept distinct from `SacrificeSelf` because discard replacements and
    /// future cycle/discard event hooks apply to a hand-zone move, not a
    /// battlefield sacrifice.
    DiscardSelf,
    /// Discard `n` cards from hand, chosen by the payer (`engine::Decision::Discard`).
    DiscardCards(u8),
    /// Sacrifice `n` controlled lands. Exact lands are staged one at a time
    /// through `engine::Decision::ChooseCostTargets`; the public
    /// `PendingCast::sacrifice_chosen` compatibility field also carries the
    /// mutually-exclusive Escape graveyard selection family.
    SacrificeLands(u8),
    /// Sacrifice `count` controlled permanents matching `filter`, announced
    /// one at a time through `Decision::ChooseCostTargets`.
    SacrificeControlled { count: u8, filter: PermanentFilter },
    /// An ordinary mana payment, solved by `mana::solve` same as a spell's
    /// printed cost.
    Mana(Cost),
    /// Pay a fixed amount of life. Unlike Phyrexian mana this is a mandatory
    /// non-mana component, so the payer must have at least this much life and
    /// may legally pay down to exactly zero.
    PayLife(u8),
    /// Exile `n` other cards from the payer's own graveyard. The source is
    /// excluded even before 601.2a moves it to the stack, and the exact cards
    /// are staged through `engine::Decision::ChooseCostTargets` before any
    /// mana or zone-change payment commits. Escape is the first consumer.
    ExileOtherCardsFromOwnGraveyard(u8),
    /// Return exactly one controlled permanent matching `filter` to its
    /// owner's hand. The physical object is selected through the generic
    /// cost-target staging before any payment commits. Appended for Quirion
    /// Ranger without changing any existing cost recipe identity.
    ReturnControlledPermanentToOwnersHand(PermanentFilterDef),
    /// Tap one other untapped permanent the payer controls with the named
    /// effective subtype. The physical object is chosen during activation
    /// staging and paid atomically with the remaining components.
    TapOtherUntappedControlledPermanentWithSubtype(Subtype),
    /// Tap one untapped permanent the payer controls matching the typed
    /// filter. Unlike the preceding activation-only form, this component
    /// may appear in an alternative casting cost such as flashback.
    TapUntappedControlledPermanent(PermanentFilterDef),
    /// Reveal the payer's complete hand, but only if it contains no card of
    /// the named type. The reveal itself is the cost, so it happens during
    /// payment after the spell has left hand for the stack. Land Grant is
    /// the first consumer.
    RevealHandIfNoCardsWithType(CardType),
    /// Return one controlled unblocked attacking creature to its owner's
    /// hand. This component also owns ninjutsu's exact post-blockers combat
    /// timing gate, so hand-zone abilities using it cannot be offered in an
    /// ordinary priority window.
    ReturnControlledUnblockedAttackerToOwnersHand,
    /// As an additional casting cost, choose either one creature the caster
    /// controls or one creature card in their hand. A hand choice is
    /// publicly revealed and the exact incarnation is frozen on the spell.
    ChooseControlledCreatureOrRevealCreatureCardFromHand,
}

/// Optional additional costs chosen while announcing a spell. The selected
/// physical cards or permanents are staged separately in `engine::PendingCast`
/// and paid atomically with the spell's other costs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OptionalAdditionalCostDef {
    /// Exile any number of cards from the caster's own graveyard whose
    /// printed mana values total at least this amount.
    CollectEvidence { minimum_mana_value: u16 },
    /// Sacrifice one controlled artifact, enchantment, or token.
    Bargain,
}

/// Static rules carried by a permanent while it is attached. The host link
/// itself is incarnation-bound in `ObjectStateV4::attached_to`; this
/// definition describes what a valid attachment requires and grants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttachmentDef {
    AuraCreature { prevents_untap: bool },
}

/// The ordered cost of casting a card from the graveyard via flashback
/// (702.10), exiling it instead of returning it to the graveyard whenever it
/// would leave the stack. An ordered component slice supports composite costs
/// such as Deep Analysis's `{1}{U}`, pay 3 life without making flashback a
/// parallel card-shaped cost system.
pub struct FlashbackDef {
    pub cost: &'static [CostComponent],
}

/// The ordered alternative cost for casting a card from its owner's
/// graveyard via escape (702.138). Unlike flashback, escape does not replace
/// where the spell goes when it later leaves the stack; its graveyard exile
/// component is paid while casting and is represented explicitly here.
pub struct EscapeDef {
    pub cost: &'static [CostComponent],
}

/// A non-mana activated ability (605/602 use the stack, unlike a mana
/// ability). Permanent abilities, hand-zone Cycling/typecycling, and
/// graveyard abilities such as Embalm share the same no-target,
/// inline-`EffectOp` stack representation (see `state::StackItem::inline_effect`).
pub struct ActivatedAbilityDef {
    pub cost: &'static [CostComponent],
    pub target_spec: TargetSpec,
    pub effect: fn() -> EffectOp,
    /// Zone in which the printed ability may be activated. Existing
    /// permanent abilities use `Battlefield`; Cycling/typecycling use
    /// `Hand`. This is definition data rather than a card-name branch, so a
    /// future card with abilities in multiple zones composes normally.
    pub activation_zone: Zone,
    /// True iff this ability may only be activated at sorcery speed
    /// (`ActivateAsSorceryActivatedAbility` in Java -- Experimental
    /// Synthesizer's "Activate only as a sorcery."). Checked by
    /// `engine::available_activatable_abilities` via the same
    /// `sorcery_speed_timing_ok` helper a sorcery-speed cast/Plot action
    /// uses. `false` for Masked Meower's and the Blood token's abilities,
    /// which have no such restriction in their Java source.
    pub sorcery_speed_only: bool,
    /// Additional activation-time target restriction layered on top of
    /// `target_spec`. This is definition data so combat-relative targets such
    /// as "a creature this source is blocking" do not become card-name
    /// branches in the engine. Resolution rechecks this filter together with
    /// the ordinary target specification and incarnation contract.
    pub activation_target_filter: ActivationTargetFilter,
    /// Printed per-turn activation limit, counted on this source incarnation.
    /// `None` means unrestricted. Appended for Quirion Ranger without
    /// changing any existing ability selector.
    pub max_activations_per_turn: Option<u8>,
}

/// A source-relative restriction that applies while announcing a non-mana
/// activated ability. Appended as a separate vocabulary so serialized
/// `TargetSpec` discriminants remain untouched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationTargetFilter {
    TargetSpecOnly,
    CreatureBlockedBySource,
}

/// Cost paid by one printed mana ability. Special mana costs live here rather
/// than in `CostComponent` because they resolve without the stack and may need
/// an object choice in the same atomic action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManaAbilityCostDef {
    TapSelf,
    SacrificeSelf,
    TapSelfAndOtherUntappedControlledCreature,
    PutMinus0Minus1CounterOnSelf,
    /// Treasure's printed `{T}, Sacrifice this artifact` cost. Keeping the
    /// combined cost atomic prevents either half from being approximated.
    TapAndSacrificeSelf,
}

/// Amount of the chosen color added by a mana ability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManaAbilityAmountDef {
    Fixed(u8),
    ControlledCreaturesWithKeyword(Keywords),
    /// Evaluate one shared board-dependent value at activation resolution.
    /// Appended for Priest of Titania without renumbering existing variants.
    Dynamic(DynamicValueDef),
}

/// A reusable integer derived from current game state. All consumers sample
/// this at effect or mana-ability resolution, never at announcement time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DynamicValueDef {
    /// Count every battlefield permanent, regardless of controller, whose
    /// effective subtype set contains the named subtype.
    BattlefieldPermanentsWithSubtype(Subtype),
    /// A literal signed value routed through the same exact-incarnation
    /// duration machinery as board-dependent pumps.
    Fixed(i32),
    /// Count cards of the named type in the evaluating controller's
    /// graveyard. The controller is supplied by the effect or mana-ability
    /// context at the moment the value is sampled.
    ControllerGraveyardCardsWithType(CardType),
}

/// Reusable definition for a single printed mana ability whose cost, amount,
/// or side effect is richer than the legacy tap-and-add-one substrate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ManaAbilityDef {
    pub cost: ManaAbilityCostDef,
    pub amount: ManaAbilityAmountDef,
    pub controller_damage: u8,
    pub max_activations_per_turn: Option<u8>,
}

/// An additional printed mana ability beyond a card's legacy primary
/// tap-for-one choice set. Its color choices must be disjoint from every
/// other printed mana ability on the same card so the stable
/// source-plus-color action identifies exactly one ability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdditionalManaAbilityDef {
    pub colors: &'static [ManaColor],
    pub mana_cost: Cost,
    pub ability: ManaAbilityDef,
}

/// A data-owned conditional entry rule layered on top of the ordinary
/// `enters_battlefield_tapped` flag. The entering permanent itself is
/// excluded from the count, matching "other" in the printed condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntersBattlefieldTappedUnlessDef {
    pub controller_controls_other_subtype: Subtype,
    pub minimum_count: u8,
}

/// One alternative mode of a spell, with its own target shape and resolution
/// program. `engine::Decision::ChooseSpellMode` selects the printed index
/// before targeting begins.
pub struct ModeDef {
    pub target_spec: TargetSpec,
    pub effect: fn() -> EffectOp,
}

/// The alternative spell characteristics of an Omen card. The physical
/// card remains one front-face object in every non-stack zone; choosing this
/// definition while casting changes only the spell's cost, card types,
/// targeting, resolution program, and successful stack departure.
pub struct OmenDef {
    pub cost: Cost,
    pub types: &'static [CardType],
    pub target_spec: TargetSpec,
    pub effect: fn() -> EffectOp,
}

/// Bestow's alternative spell characteristics. Form zero remains the
/// ordinary creature spell; this form is an Aura spell with its own X cost
/// and creature target.
pub struct BestowDef {
    pub cost: Cost,
    pub target_spec: TargetSpec,
}

/// A deterministic value sampled while deriving a spell's total generic
/// mana cost. Kept data-driven and card-name-neutral so the same cast-cost
/// path can serve battlefield reducers (Affinity), graveyard reducers
/// (Cryptic Serpent/Tolarian Terror), and turn counters (Deem Inferior)
/// without teaching the engine individual card names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DynamicCountDef {
    /// Count permanents the caster controls that have any listed card type.
    ControllerBattlefieldAnyType(&'static [CardType]),
    /// Count cards in the caster's graveyard that have any listed type.
    ControllerGraveyardAnyType(&'static [CardType]),
    /// Count cards the caster has drawn during the current turn.
    ControllerDrawsThisTurn,
    /// One iff the controller has both a creature with the named subtype
    /// and a creature without it. Of One Mind uses Human.
    ControllerHasCreatureWithAndWithoutSubtype(Subtype),
}

/// Reduces only the generic portion of a spell's mana cost, flooring at
/// zero. Colored, hybrid, Phyrexian, and X pips are never removed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenericCostReductionDef {
    pub generic_per_count: u8,
    pub count: DynamicCountDef,
}

/// Printed Ward payments supported by the reusable opponent-target trigger.
/// New colored or nonmana costs require their own explicit variant rather
/// than an approximation through generic mana.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WardCostDef {
    Generic(u8),
}

/// Alternate battlefield characteristics for a transforming permanent's
/// back face. The physical card definition and stable card id remain those
/// of the front face; live characteristic queries select this immutable
/// record from `ObjectStateV4::face_index`.
pub struct TransformFaceDef {
    pub name: &'static str,
    pub types: &'static [CardType],
    pub subtypes: &'static [Subtype],
    pub colors: &'static [ManaColor],
    pub power: Option<i16>,
    pub toughness: Option<i16>,
    pub keywords: Keywords,
}

/// Ordered chapter programs for a Saga. Index zero is chapter I. A Saga's
/// final chapter is therefore `chapter_effects.len()`; the engine uses the
/// same definition both to create chapter triggers and to apply the final-
/// chapter state-based action.
pub struct SagaDef {
    pub chapter_effects: &'static [fn() -> EffectOp],
}

/// Reusable static and triggered grants produced by an attached Equipment.
/// The attachment relation itself is incarnation-bound in `state.rs`; this
/// definition contains only printed characteristics and granted abilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EquipmentDef {
    pub power_delta: i16,
    pub toughness_delta: i16,
    pub add_subtype: Option<Subtype>,
    pub controller_turn_keywords: Keywords,
    pub other_turn_keywords: Keywords,
    pub noncreature_spell_damage_to_each_opponent: u8,
    pub job_select: bool,
}

pub struct CardDef {
    pub name: &'static str,
    pub capability: CardCapability,
    pub cost: Cost,
    /// A spell-local generic cost reducer evaluated by the shared cast
    /// legality/payment pipeline. `None` for cards without this text.
    pub generic_cost_reduction: Option<GenericCostReductionDef>,
    /// Static Ward cost materialized when an opposing spell or ability
    /// finishes targeting this permanent. `None` means no implemented Ward.
    pub ward_cost: Option<WardCostDef>,
    /// Printed Equipment behavior shared by attachments, effective
    /// characteristics, cast triggers, and RL continuous-effect projection.
    pub equipment: Option<EquipmentDef>,
    pub types: &'static [CardType],
    /// This card's creature/land/artifact subtypes (105.1's subtype line),
    /// e.g. `[Subtype::Human, Subtype::Shaman]` for Burning-Tree Emissary --
    /// see `Subtype`'s own doc for why this is a fully-enumerated closed set.
    /// Read where a card's *own* effect needs it (Rally at the Hornburg's
    /// `CreatureFilter::ControlledWithSubtype(Subtype::Human)` -- see
    /// `effect.rs`).
    pub subtypes: &'static [Subtype],
    pub supertypes: &'static [Supertype],
    pub power: Option<i16>,
    pub toughness: Option<i16>,
    pub is_land: bool,
    pub produces_mana: &'static [ManaColor],
    /// This card's color identity per 105.1/202.2 (the color of mana
    /// symbols in its mana cost) -- empty for a colorless card. The four
    /// Blast cards inspect this through `EffectCond::TargetIsColor` or the
    /// color-filtered target specs. This is deliberately bounded to the
    /// current pool's static colors: XMage's `getColor(game)` also observes
    /// continuous color-changing effects, which this kernel does not yet
    /// model.
    pub colors: &'static [ManaColor],
    pub target_spec: TargetSpec,
    pub keywords: Keywords,
    /// Program run when the spell resolves off the stack. `None` = not
    /// implemented this increment (present in the table, not castable).
    pub spell_effect: fn() -> Option<EffectOp>,
    /// Program run when the card's mana ability is activated. `None` = no
    /// mana ability (or not implemented). This legacy function-pointer form
    /// represents single-color abilities. `mana_ability_choices` below is
    /// authoritative for multi-color permanents.
    pub mana_ability: fn() -> Option<EffectOp>,
    /// `Some` iff this card has an alternative cost you may pay instead of
    /// its mana cost (Fireblast). Choosing between them is a real decision
    /// (`engine::Decision::ChooseCastMode`) when both are legal.
    pub alt_cost: Option<&'static [CostComponent]>,
    /// `Some` iff this card has Kicker (`KickerAbility`): an optional
    /// additional cost you may pay as you cast it, stamped onto the spell's
    /// own `state::StackItem::kicked` once paid (`engine::finalize_cast`)
    /// and carried from there into its resolution/ETB context so a later
    /// triggered ability can check `EffectCond::WasKicked`. Only Goblin
    /// Bushwhacker's `Kicker {R}` this increment. Unlike
    /// `additional_cost` (mandatory) or `alt_cost` (replaces the printed
    /// cost), this is paid *in addition to* whichever of those two costs
    /// this cast otherwise settles on -- see `engine::Decision::
    /// ChooseKicker`/`mana::can_pay_combined`.
    pub kicker_cost: Option<Cost>,
    /// `Some` iff this card has a mandatory additional cost paid on top of
    /// its mana cost (Grab the Prize's discard).
    pub additional_cost: Option<&'static [CostComponent]>,
    /// `Some` iff this card can be cast from the graveyard for its
    /// flashback cost (Faithless Looting, Lava Dart, Deep Analysis).
    pub flashback: Option<FlashbackDef>,
    pub activated_abilities: &'static [ActivatedAbilityDef],
    /// `Some` iff this card can be Plotted (`PlotAbility`): exiled from
    /// hand for this cost at sorcery speed, then castable for free (any
    /// later turn, still sorcery speed) -- `engine::plot_action_candidates`/
    /// `engine::is_plotted_castable_now`. Only Highway Robbery in this pool.
    pub plot_cost: Option<Cost>,
    /// `Some` iff this card has Madness (`MadnessAbility`): whenever it
    /// would be discarded, it's exiled instead, and its owner may cast it
    /// for this cost rather than putting it into the graveyard --
    /// `engine::PendingMadness`/`Decision::ChooseMadnessCast`. Only Fiery
    /// Temper in this pool.
    pub madness_cost: Option<Cost>,
    /// `Some` iff this spell is modal with a second mode (the Blast cards'
    /// destroy mode) -- see `ModeDef`'s doc.
    pub mode2: Option<ModeDef>,
    /// Optional third printed mode. Piracy Charm is the first consumer.
    pub mode3: Option<ModeDef>,
    /// A permanent token (`cards_v1.json`'s own `is_token`, e.g. Blood),
    /// never itself a deck card -- read by `trigger::sba_fixed_point` for
    /// 111.8/704.5d ("if a token is in a zone other than the battlefield,
    /// it ceases to exist -- this is a state-based action"). Only `Blood
    /// Token` this increment.
    pub is_token: bool,
    /// `Some` iff this card can be cast from its owner's graveyard for an
    /// escape cost. Appended independently from `flashback` because the two
    /// mechanics have different stack-departure contracts.
    pub escape: Option<EscapeDef>,
    /// Exact colors available from this permanent's repeatable tap-for-mana
    /// abilities. This is deliberately distinct from `produces_mana`, which
    /// also includes one-shot production such as Burning-Tree Emissary's ETB
    /// trigger. Appended to preserve prior generated field identities.
    pub mana_ability_choices: &'static [ManaColor],
    /// Whether this permanent enters the battlefield tapped. The shared
    /// zone-change commit path enforces this for every battlefield entry.
    pub enters_battlefield_tapped: bool,
    /// A single printed mana ability whose rules are not exactly "tap this:
    /// add one of the selected color." `None` preserves the legacy automatic
    /// payment path for basics, artifact lands, bridges, and ordinary mana
    /// creatures.
    pub mana_ability_def: Option<ManaAbilityDef>,
    /// Minimum number of creatures required to block this attacker once it
    /// is blocked. Zero and one both mean the ordinary one-or-more rule;
    /// Troll of Khazad-dum is the first value above one (three).
    pub minimum_blockers: u8,
    /// Alternative Omen spell characteristics, if any. Appended so all
    /// pre-existing generated `CardDef` field identities stay fixed.
    pub omen: Option<OmenDef>,
    /// Printed mana value from the registry. This remains constant even when
    /// Affinity or another reducer changes the amount actually paid.
    pub mana_value: u16,
    /// True iff the primary printed mana ability can add the color stored in
    /// `ObjectStateV4::chosen_color` in addition to its fixed choices.
    pub mana_ability_includes_chosen_color: bool,
    /// An as-this-enters choice excluding the named color. The engine stages
    /// the choice before the permanent moves to the battlefield.
    pub as_enters_choose_color_other_than: Option<ManaColor>,
    /// Additional printed mana abilities with costs richer than the primary
    /// legacy tap-for-one shape. Appended so all earlier CardDef fields and
    /// generated bindings retain their identities.
    pub additional_mana_abilities: &'static [AdditionalManaAbilityDef],
    /// Public copiable object name when it differs from the registry's
    /// unique lookup label. Embalm tokens copy the source card's name while
    /// retaining a distinct generated definition id.
    pub object_name: &'static str,
    /// A conditional replacement for unconditional tapped entry. `None`
    /// preserves the existing `enters_battlefield_tapped` behavior.
    pub enters_battlefield_tapped_unless: Option<EntersBattlefieldTappedUnlessDef>,
    /// A permanent spell that enters attached uses this definition both for
    /// attachment state-based actions and for continuous host restrictions.
    /// Appended so every existing generated field identity remains fixed.
    pub attachment: Option<AttachmentDef>,
    /// Alternate battlefield face, if this card can transform. Appended so
    /// all existing generated field identities remain fixed.
    pub transform_face: Option<TransformFaceDef>,
    /// Ordered Saga chapter programs. Appended independently from subtype
    /// metadata because not every card with a printed Saga subtype is
    /// necessarily executable.
    pub saga: Option<SagaDef>,
    /// Optional cast-time additional cost, if any. Appended so all existing
    /// generated field identities remain fixed.
    pub optional_additional_cost: Option<OptionalAdditionalCostDef>,
    /// True iff the object has Changeling and therefore every creature type
    /// in the closed subtype registry. Appended independently from ordinary
    /// printed subtypes so Shapeshifter remains visible as printed metadata.
    pub changeling: bool,
    /// Alternative Bestow spell characteristics. Appended so every earlier
    /// generated field identity remains stable.
    pub bestow: Option<BestowDef>,
}

impl CardDef {
    pub fn has_type(&self, t: CardType) -> bool {
        self.types.contains(&t)
    }

    pub fn types_for_face(&self, face_index: u8) -> &'static [CardType] {
        if face_index == 1 {
            if let Some(face) = &self.transform_face {
                return face.types;
            }
        }
        self.types
    }

    pub fn subtypes_for_face(&self, face_index: u8) -> &'static [Subtype] {
        if face_index == 1 {
            if let Some(face) = &self.transform_face {
                return face.subtypes;
            }
        }
        self.subtypes
    }

    pub fn colors_for_face(&self, face_index: u8) -> &'static [ManaColor] {
        if face_index == 1 {
            if let Some(face) = &self.transform_face {
                return face.colors;
            }
        }
        self.colors
    }

    pub fn power_for_face(&self, face_index: u8) -> Option<i16> {
        if face_index == 1 {
            if let Some(face) = &self.transform_face {
                return face.power;
            }
        }
        self.power
    }

    pub fn toughness_for_face(&self, face_index: u8) -> Option<i16> {
        if face_index == 1 {
            if let Some(face) = &self.transform_face {
                return face.toughness;
            }
        }
        self.toughness
    }

    pub fn keywords_for_face(&self, face_index: u8) -> Keywords {
        if face_index == 1 {
            if let Some(face) = &self.transform_face {
                return face.keywords;
            }
        }
        self.keywords
    }

    pub fn is_castable(&self) -> bool {
        self.is_executable() && !self.is_land && !self.is_token
    }

    pub const fn is_executable(&self) -> bool {
        self.capability.is_executable()
    }

    pub const fn has_full_support(&self) -> bool {
        self.capability.is_fully_supported()
    }

    pub fn mana_ability_program(&self) -> Option<EffectOp> {
        (self.is_executable() && self.mana_ability_choices.len() == 1)
            .then(|| (self.mana_ability)())
            .flatten()
    }

    pub fn has_mana_ability(&self) -> bool {
        self.is_executable()
            && (!self.mana_ability_choices.is_empty()
                || self.mana_ability_includes_chosen_color
                || !self.additional_mana_abilities.is_empty())
    }

    /// Automatic spell-cost payment may only tap sources whose entire printed
    /// mana ability is the legacy tap-and-add-one program. Richer abilities
    /// remain fully playable through explicit mana actions, preventing the
    /// solver from silently skipping sacrifice, damage, counter, or object
    /// costs.
    pub fn is_automatic_payment_mana_source(&self) -> bool {
        self.is_executable()
            && !self.mana_ability_choices.is_empty()
            && self.mana_ability_def.is_none()
    }

    /// Non-allocating core of [`Self::primary_mana_ability_choices`]. Writes
    /// into `out` instead of returning an owned `Vec`, for hot paths (such as
    /// flat-action validation) that must not touch the heap. `out` is
    /// appended to, not cleared, so a fresh `ManaColorSetV1` is expected.
    pub(crate) fn primary_mana_ability_choices_into(
        &self,
        chosen_color: Option<ManaColor>,
        out: &mut ManaColorSetV1,
    ) {
        for &color in self.mana_ability_choices {
            out.push(color);
        }
        if self.mana_ability_includes_chosen_color {
            if let Some(color) = chosen_color {
                if !out.contains(color) {
                    out.push(color);
                }
            }
        }
    }

    /// Exact colors the primary printed mana ability can currently produce.
    /// Chosen-color lands remain fail closed until their entry choice has
    /// populated the incarnation-local object state.
    pub fn primary_mana_ability_choices(&self, chosen_color: Option<ManaColor>) -> Vec<ManaColor> {
        let mut choices = ManaColorSetV1::new();
        self.primary_mana_ability_choices_into(chosen_color, &mut choices);
        choices.as_slice().to_vec()
    }

    /// Every currently legal color across all printed mana abilities.
    pub fn all_mana_ability_choices(&self, chosen_color: Option<ManaColor>) -> Vec<ManaColor> {
        let mut choices = self.primary_mana_ability_choices(chosen_color);
        for additional in self.additional_mana_abilities {
            for &color in additional.colors {
                if !choices.contains(&color) {
                    choices.push(color);
                }
            }
        }
        choices
    }

    /// Stable printed-ability index used by per-turn limits and public
    /// provenance. A rich definition represents one printed ability even when
    /// that ability offers several colors; legacy dual lands retain one index
    /// per separately printed color ability.
    pub fn mana_ability_index(
        &self,
        choice: ManaColor,
        chosen_color: Option<ManaColor>,
    ) -> Option<u16> {
        let primary = self.primary_mana_ability_choices(chosen_color);
        if let Some(choice_index) = primary.iter().position(|candidate| *candidate == choice) {
            return Some(
                if self.mana_ability_def.is_some() || self.mana_ability_includes_chosen_color {
                    0
                } else {
                    u16::try_from(choice_index).ok()?
                },
            );
        }
        let primary_ability_count =
            if self.mana_ability_def.is_some() || self.mana_ability_includes_chosen_color {
                u16::from(!primary.is_empty())
            } else {
                u16::try_from(primary.len()).ok()?
            };
        let mut found = None;
        for (index, additional) in self.additional_mana_abilities.iter().enumerate() {
            if additional.colors.contains(&choice) {
                if found.is_some() {
                    return None;
                }
                found = Some(primary_ability_count + u16::try_from(index).ok()?);
            }
        }
        found
    }

    /// Builds the exact tap-and-add program for one printed mana ability.
    /// A dual land has two legal programs, each adding exactly one mana.
    pub fn mana_ability_program_for(
        &self,
        choice: ManaColor,
        chosen_color: Option<ManaColor>,
    ) -> Option<EffectOp> {
        (self.is_executable()
            && self.mana_ability_def.is_none()
            && self
                .primary_mana_ability_choices(chosen_color)
                .contains(&choice))
        .then(|| {
            EffectOp::Sequence(vec![
                EffectOp::TapObject {
                    object: ObjectRef::ThisSource,
                },
                EffectOp::AddMana {
                    player: PlayerRef::Controller,
                    colors: vec![choice],
                },
            ])
        })
    }

    pub const fn is_playable_land(&self) -> bool {
        self.is_land && self.is_executable()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeckPreflightError {
    UnknownCardDefinition {
        index: usize,
        card_def: u16,
    },
    TokenInDeck {
        index: usize,
        card_def: u16,
        name: &'static str,
    },
    NotFullySupported {
        index: usize,
        card_def: u16,
        name: &'static str,
        capability: CardCapability,
    },
}

impl fmt::Display for DeckPreflightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeckPreflightError::UnknownCardDefinition { index, card_def } => {
                write!(f, "deck card {index} references unknown definition {card_def}")
            }
            DeckPreflightError::TokenInDeck {
                index,
                card_def,
                name,
            } => write!(
                f,
                "deck card {index} ({name}, definition {card_def}) is a token and cannot be a deck entry"
            ),
            DeckPreflightError::NotFullySupported {
                index,
                card_def,
                name,
                capability,
            } => write!(
                f,
                "deck card {index} ({name}, definition {card_def}) is not fully supported: {capability:?}"
            ),
        }
    }
}

impl std::error::Error for DeckPreflightError {}

/// Rejects a deck at environment construction time unless every definition
/// is explicitly `Full`. Missing records and newly added records whose
/// capability was omitted both fail closed.
pub fn preflight_fully_supported_deck(card_defs: &[u16]) -> Result<(), DeckPreflightError> {
    for (index, &card_def) in card_defs.iter().enumerate() {
        let Some(def) = CARD_DEFS.get(card_def as usize) else {
            return Err(DeckPreflightError::UnknownCardDefinition { index, card_def });
        };
        if def.is_token {
            return Err(DeckPreflightError::TokenInDeck {
                index,
                card_def,
                name: def.name,
            });
        }
        if !def.has_full_support() {
            return Err(DeckPreflightError::NotFullySupported {
                index,
                card_def,
                name: def.name,
                capability: def.capability,
            });
        }
    }
    Ok(())
}

pub fn no_effect() -> Option<EffectOp> {
    None
}

include!(concat!(env!("OUT_DIR"), "/card_defs.rs"));

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effect::{EffectCond, ObjectRef, PlayerRef, TargetRef};
    use crate::state::Zone;

    #[test]
    fn card_defs_len_matches_pool() {
        // Hero Token remains id 159 and Clue Token remains id 160. Skeleton
        // Token is appended as id 161 without renumbering earlier ids.
        assert_eq!(CARD_DEFS.len(), 162);
    }

    #[test]
    fn target_spec_variants_are_append_only() {
        let stable_ordinals = [
            (TargetSpec::None, 0),
            (TargetSpec::AnyTarget, 1),
            (TargetSpec::PlayerThenTheirCreature, 2),
            (TargetSpec::AnySpellOnStack, 3),
            (TargetSpec::InstantSpellOnStack, 4),
            (TargetSpec::BlueSpellOnStack, 5),
            (TargetSpec::AnyPermanent, 6),
            (TargetSpec::BluePermanent, 7),
            (TargetSpec::AnyPlayer, 8),
            (TargetSpec::RedSpellOnStack, 9),
            (TargetSpec::RedPermanent, 10),
            (TargetSpec::NonlandPermanent, 11),
            (TargetSpec::Creature, 12),
            (TargetSpec::NonlegendaryCreature, 13),
            (TargetSpec::ArtifactOrEnchantmentSpellOnStack, 14),
            (TargetSpec::SorcerySpellOnStack, 15),
            (TargetSpec::NoncreatureSpellOnStack, 16),
            (TargetSpec::ArtifactSpellOnStack, 17),
            (TargetSpec::ArtifactPermanent, 18),
            (TargetSpec::CreatureOrLandCardInGraveyard, 19),
            (TargetSpec::ControlledCreature, 20),
            (TargetSpec::UpToTwoCreatureCardsInOwnGraveyard, 21),
            (TargetSpec::UpToTwoCreatures, 22),
            (TargetSpec::ExactlyTwoArtifactPermanents, 23),
            (TargetSpec::EnchantmentPermanent, 24),
            (TargetSpec::UpToTwoPlayers, 25),
            (TargetSpec::CreatureCardInOwnGraveyard, 26),
            (TargetSpec::TargetOpponent, 27),
            (TargetSpec::OpponentControlledCreature, 28),
            (
                TargetSpec::SpellManaValueAtMostControlledSubtypes {
                    first: Subtype::Faerie,
                    second: Some(Subtype::FaerieAllCaps),
                },
                29,
            ),
            (TargetSpec::UpToTwoCardsInGraveyards, 30),
            (TargetSpec::CreatureOtherThanSource, 31),
            (TargetSpec::UpToOneTappedCreature, 32),
            (TargetSpec::NoncreatureArtifactPermanent, 33),
            (TargetSpec::Land, 34),
            (TargetSpec::OpponentArtifactOrEnchantmentPermanent, 35),
        ];
        for (target_spec, ordinal) in stable_ordinals {
            assert_eq!(target_spec.stable_id(), ordinal);
        }
        assert_eq!(
            serde_json::to_string(&TargetSpec::CreatureOrLandCardInGraveyard).unwrap(),
            "\"CreatureOrLandCardInGraveyard\""
        );
        assert_eq!(
            serde_json::from_str::<TargetSpec>("\"CreatureOrLandCardInGraveyard\"").unwrap(),
            TargetSpec::CreatureOrLandCardInGraveyard
        );
    }

    #[test]
    fn card_db_hash_v32_is_frozen() {
        // Version 32 appends the final pool trio and Skeleton token after the
        // combined optional-cost root without renumbering prior definitions.
        assert_eq!(KERNEL_CARDDB_HASH, 0x64c8_2a26_1e07_8f1a);
    }

    #[test]
    fn card_names_are_unique() {
        let mut names: Vec<&str> = CARD_DEFS.iter().map(|c| c.name).collect();
        let before = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), before, "duplicate card names in CARD_DEFS");
    }

    #[test]
    fn lookup_by_name_round_trips_id() {
        for (i, def) in CARD_DEFS.iter().enumerate() {
            assert_eq!(
                card_id_by_name(def.name),
                Some(i as u16),
                "name={}",
                def.name
            );
        }
        assert_eq!(card_id_by_name("Not A Real Card"), None);
    }

    #[test]
    fn cryptic_serpent_has_the_generic_graveyard_spell_reducer() {
        let def = &CARD_DEFS[card_id_by_name("Cryptic Serpent").unwrap() as usize];
        assert_eq!(def.capability, CardCapability::Full);
        assert_eq!(def.cost.generic, 5);
        assert_eq!(
            def.cost.pips,
            &[Pip::Colored(ManaColor::U), Pip::Colored(ManaColor::U)]
        );
        assert_eq!(
            def.generic_cost_reduction,
            Some(GenericCostReductionDef {
                generic_per_count: 1,
                count: DynamicCountDef::ControllerGraveyardAnyType(&[
                    CardType::Instant,
                    CardType::Sorcery,
                ]),
            })
        );
        assert!(def.is_castable());
    }

    #[test]
    fn affinity_cards_share_the_artifact_reducer_and_keep_their_own_programs() {
        let reducer = Some(GenericCostReductionDef {
            generic_per_count: 1,
            count: DynamicCountDef::ControllerBattlefieldAnyType(&[CardType::Artifact]),
        });

        let enforcer = &CARD_DEFS[card_id_by_name("Myr Enforcer").unwrap() as usize];
        assert_eq!(enforcer.capability, CardCapability::Full);
        assert_eq!(enforcer.cost.generic, 7);
        assert!(enforcer.cost.pips.is_empty());
        assert_eq!(enforcer.generic_cost_reduction, reducer);
        assert_eq!(enforcer.types, &[CardType::Artifact, CardType::Creature]);
        assert_eq!((enforcer.power, enforcer.toughness), (Some(4), Some(4)));
        assert!(matches!(
            (enforcer.spell_effect)(),
            Some(EffectOp::MoveObject {
                object: ObjectRef::ThisSource,
                to_zone: Zone::Battlefield,
            })
        ));

        let thoughtcast = &CARD_DEFS[card_id_by_name("Thoughtcast").unwrap() as usize];
        assert_eq!(thoughtcast.capability, CardCapability::Full);
        assert_eq!(thoughtcast.cost.generic, 4);
        assert_eq!(thoughtcast.cost.pips, &[Pip::Colored(ManaColor::U)]);
        assert_eq!(thoughtcast.generic_cost_reduction, reducer);
        assert_eq!(thoughtcast.types, &[CardType::Sorcery]);
        assert_eq!(
            (thoughtcast.spell_effect)(),
            Some(EffectOp::DrawCards {
                player: PlayerRef::Controller,
                count: 2,
            })
        );
    }

    #[test]
    fn tolarian_terror_composes_the_shared_reducer_and_generic_ward() {
        let def = &CARD_DEFS[card_id_by_name("Tolarian Terror").unwrap() as usize];
        assert_eq!(def.capability, CardCapability::Full);
        assert_eq!(def.cost.generic, 6);
        assert_eq!(def.cost.pips, &[Pip::Colored(ManaColor::U)]);
        assert_eq!((def.power, def.toughness), (Some(5), Some(5)));
        assert_eq!(
            def.generic_cost_reduction,
            Some(GenericCostReductionDef {
                generic_per_count: 1,
                count: DynamicCountDef::ControllerGraveyardAnyType(&[
                    CardType::Instant,
                    CardType::Sorcery,
                ]),
            })
        );
        assert_eq!(def.ward_cost, Some(WardCostDef::Generic(2)));
        assert!(def.is_castable());
    }

    #[test]
    fn ward_costs_fail_closed_outside_the_static_generic_creature_shape() {
        for def in CARD_DEFS.iter().filter(|def| def.ward_cost.is_some()) {
            assert!(def.is_castable(), "{} is not executable", def.name);
            assert!(
                def.has_type(CardType::Creature),
                "{} is not a creature",
                def.name
            );
            match def.ward_cost.unwrap() {
                WardCostDef::Generic(amount) => assert_ne!(amount, 0, "{} has Ward 0", def.name),
            }
        }
    }

    #[test]
    fn generic_reducers_fail_closed_outside_the_certified_printed_cost_shape() {
        for def in CARD_DEFS
            .iter()
            .filter(|def| def.generic_cost_reduction.is_some())
        {
            assert!(def.is_castable(), "{} is not executable", def.name);
            assert_ne!(
                def.generic_cost_reduction.unwrap().generic_per_count,
                0,
                "{} has a zero reducer",
                def.name
            );
            assert_eq!(
                def.cost.x_count, 0,
                "{} has an X cost outside the certified reducer shape",
                def.name
            );
            assert!(
                def.alt_cost.is_none(),
                "{} has an alternative cost",
                def.name
            );
            assert!(def.kicker_cost.is_none(), "{} has kicker", def.name);
            assert!(
                def.additional_cost.is_none(),
                "{} has an additional cost",
                def.name
            );
            assert!(def.flashback.is_none(), "{} has flashback", def.name);
            assert!(def.escape.is_none(), "{} has escape", def.name);
            assert!(def.plot_cost.is_none(), "{} has plot", def.name);
            assert!(def.madness_cost.is_none(), "{} has madness", def.name);
        }
    }

    #[test]
    fn intrinsic_basic_lands_derive_the_exact_subtype_mana_ability() {
        for (name, color) in [
            ("Mountain", ManaColor::R),
            ("Island", ManaColor::U),
            ("Forest", ManaColor::G),
            ("Swamp", ManaColor::B),
            ("Snow-Covered Forest", ManaColor::G),
        ] {
            let id = card_id_by_name(name).unwrap_or_else(|| panic!("{name} in pool"));
            let def = &CARD_DEFS[id as usize];
            assert!(def.is_land, "{name}");
            assert!(def.supertypes.contains(&Supertype::Basic), "{name}");
            assert_eq!(def.produces_mana, &[color], "{name}");
            assert!(def.is_executable(), "{name}");
            assert!(def.has_full_support(), "{name}");
            assert!(!def.is_castable(), "lands aren't cast: {name}");
            match def.mana_ability_program() {
                Some(EffectOp::Sequence(ops)) => {
                    assert_eq!(
                        ops,
                        vec![
                            EffectOp::TapObject {
                                object: ObjectRef::ThisSource
                            },
                            EffectOp::AddMana {
                                player: PlayerRef::Controller,
                                colors: vec![color]
                            },
                        ],
                        "{name}"
                    );
                }
                other => panic!("{name}: expected tap+add-mana sequence, got {other:?}"),
            }
        }
    }

    #[test]
    fn capability_is_the_fail_closed_source_for_programs_and_deck_preflight() {
        let full = CARD_DEFS
            .iter()
            .filter(|def| def.capability == CardCapability::Full)
            .count();
        assert_eq!(full, 162, "150 pool cards plus twelve required tokens");
        assert_eq!(
            CARD_DEFS
                .iter()
                .filter(|def| def.capability == CardCapability::Partial)
                .count(),
            0
        );

        let supported = ["Island", "Counterspell", "Mountain"]
            .map(|name| card_id_by_name(name).expect("card in registry"));
        assert!(preflight_fully_supported_deck(&supported).is_ok());
        assert!(preflight_fully_supported_deck(&[card_id_by_name("Winding Way").unwrap()]).is_ok());
        assert!(preflight_fully_supported_deck(&[
            card_id_by_name("Island").unwrap(),
            card_id_by_name("Tolarian Terror").unwrap(),
        ])
        .is_ok());
        if let Some(unsupported_id) = CARD_DEFS
            .iter()
            .position(|def| !def.is_token && def.capability == CardCapability::NoEffect)
        {
            let unsupported = [card_id_by_name("Island").unwrap(), unsupported_id as u16];
            let err = preflight_fully_supported_deck(&unsupported)
                .expect_err("a no-effect card must remain fail closed");
            assert!(matches!(
                err,
                DeckPreflightError::NotFullySupported {
                    index: 1,
                    capability: CardCapability::NoEffect,
                    ..
                }
            ));
        }
        assert!(preflight_fully_supported_deck(&[
            card_id_by_name("Island").unwrap(),
            card_id_by_name("Mountain").unwrap(),
        ])
        .is_ok());
        assert!(matches!(
            preflight_fully_supported_deck(&[u16::MAX]),
            Err(DeckPreflightError::UnknownCardDefinition { .. })
        ));
        assert!(matches!(
            preflight_fully_supported_deck(&[card_id_by_name("Blood Token").unwrap()]),
            Err(DeckPreflightError::TokenInDeck {
                name: "Blood Token",
                ..
            })
        ));
    }

    #[test]
    fn winding_way_program_freezes_resolution_choice_and_printed_option_order() {
        let winding = &CARD_DEFS[card_id_by_name("Winding Way").unwrap() as usize];
        assert_eq!(winding.capability, CardCapability::Full);
        assert_eq!(winding.target_spec, TargetSpec::None);
        assert!(winding.is_castable());
        assert_eq!(
            (winding.spell_effect)(),
            Some(EffectOp::Choice {
                controller: PlayerRef::Controller,
                options: vec![
                    EffectOp::RevealTopAndPartitionByType {
                        player: PlayerRef::Controller,
                        count: 4,
                        card_type: CardType::Creature,
                        matching_to: Zone::Hand,
                        rest_to: Zone::Graveyard,
                    },
                    EffectOp::RevealTopAndPartitionByType {
                        player: PlayerRef::Controller,
                        count: 4,
                        card_type: CardType::Land,
                        matching_to: Zone::Hand,
                        rest_to: Zone::Graveyard,
                    },
                ],
            })
        );
    }

    #[test]
    fn preordain_program_is_generated_as_scry_two_then_draw_one() {
        let preordain = &CARD_DEFS[card_id_by_name("Preordain").unwrap() as usize];
        assert_eq!(preordain.capability, CardCapability::Full);
        assert_eq!(preordain.target_spec, TargetSpec::None);
        assert!(preordain.is_castable());
        assert_eq!(
            (preordain.spell_effect)(),
            Some(EffectOp::Sequence(vec![
                EffectOp::Scry {
                    player: PlayerRef::Controller,
                    count: 2,
                },
                EffectOp::DrawCards {
                    player: PlayerRef::Controller,
                    count: 1,
                },
            ]))
        );
    }

    #[test]
    fn supported_hunter_and_nonbasic_mana_metadata_are_exact() {
        let hunter = &CARD_DEFS[card_id_by_name("Avenging Hunter").unwrap() as usize];
        assert!(hunter.has_type(CardType::Creature));
        assert!(hunter.is_executable());
        assert!(hunter.is_castable());
        assert!(hunter.keywords.has(Keywords::TRAMPLE));
        assert_eq!(
            (hunter.spell_effect)(),
            Some(EffectOp::MoveObject {
                object: ObjectRef::ThisSource,
                to_zone: Zone::Battlefield,
            })
        );

        for name in ["Burning-Tree Emissary", "Azorius Guildgate"] {
            let def = &CARD_DEFS[card_id_by_name(name).unwrap() as usize];
            assert!(
                !def.produces_mana.is_empty(),
                "test requires mana metadata: {name}"
            );
            assert!(
                def.mana_ability_program().is_none(),
                "metadata alone must not grant {name} a tappable mana ability"
            );
        }

        let landscape = &CARD_DEFS[card_id_by_name("Twisted Landscape").unwrap() as usize];
        assert_eq!(landscape.produces_mana, &[ManaColor::C]);
        assert_eq!(
            landscape.mana_ability_program(),
            Some(EffectOp::Sequence(vec![
                EffectOp::TapObject {
                    object: ObjectRef::ThisSource,
                },
                EffectOp::AddMana {
                    player: PlayerRef::Controller,
                    colors: vec![ManaColor::C],
                },
            ]))
        );
    }

    #[test]
    fn the_four_burn_spells_deal_exactly_their_printed_damage_to_any_target() {
        let expected = [
            ("Lightning Bolt", 3),
            ("Fiery Temper", 3),
            ("Fireblast", 4),
            ("Lava Dart", 1),
        ];
        for (name, amount) in expected {
            let id = card_id_by_name(name).unwrap_or_else(|| panic!("{name} in pool"));
            let def = &CARD_DEFS[id as usize];
            assert_eq!(def.target_spec, TargetSpec::AnyTarget, "{name}");
            match (def.spell_effect)() {
                Some(EffectOp::DealDamage {
                    target: TargetRef::Target(0),
                    amount: a,
                }) => {
                    assert_eq!(a, amount, "{name}");
                }
                other => panic!("{name}: expected DealDamage to Target(0), got {other:?}"),
            }
        }
    }

    #[test]
    fn vanilla_creatures_resolve_straight_to_battlefield() {
        for name in [
            "Guttersnipe",
            "Masked Meower",
            "Voldaren Epicure",
            "Sneaky Snacker",
        ] {
            let id = card_id_by_name(name).unwrap_or_else(|| panic!("{name} in pool"));
            let def = &CARD_DEFS[id as usize];
            assert!(def.has_type(CardType::Creature), "{name}");
            assert_eq!(def.target_spec, TargetSpec::None, "{name}");
            match (def.spell_effect)() {
                Some(EffectOp::MoveObject {
                    object: ObjectRef::ThisSource,
                    to_zone: Zone::Battlefield,
                }) => {}
                other => panic!("{name}: expected MoveObject to Battlefield, got {other:?}"),
            }
        }
    }

    #[test]
    fn pauper_counterspell_wave_is_fully_supported() {
        for name in [
            "Annul",
            "Envelop",
            "Force Spike",
            "Spell Pierce",
            "Steel Sabotage",
        ] {
            let id = card_id_by_name(name).unwrap_or_else(|| panic!("{name} in pool"));
            assert!(CARD_DEFS[id as usize].has_full_support(), "{name}");
            assert!(CARD_DEFS[id as usize].is_castable(), "{name}");
        }
    }

    #[test]
    fn relic_of_progenitus_is_enabled() {
        let name = "Relic of Progenitus";
        let id = card_id_by_name(name).unwrap_or_else(|| panic!("{name} in pool"));
        let def = &CARD_DEFS[id as usize];
        assert!(def.is_castable(), "{name}");
        assert_eq!(def.activated_abilities.len(), 2, "{name}");
    }

    #[test]
    fn highway_robbery_is_castable_and_plottable() {
        let id = card_id_by_name("Highway Robbery").expect("Highway Robbery in pool");
        let def = &CARD_DEFS[id as usize];
        assert!(def.is_castable());
        assert_eq!(def.target_spec, TargetSpec::None);
        assert!(matches!(
            (def.spell_effect)(),
            Some(EffectOp::MayPayCostThen {
                discard: 1,
                sacrifice_lands: 1,
                ..
            })
        ));
        let plot_cost = def
            .plot_cost
            .expect("Highway Robbery should have Plot {1}{R}");
        assert_eq!(plot_cost.generic, 1);
        assert_eq!(plot_cost.pips, &[Pip::Colored(ManaColor::R)]);
    }

    #[test]
    fn fiery_temper_has_madness_r() {
        let id = card_id_by_name("Fiery Temper").expect("Fiery Temper in pool");
        let def = &CARD_DEFS[id as usize];
        let madness_cost = def
            .madness_cost
            .expect("Fiery Temper should have Madness {R}");
        assert_eq!(madness_cost.generic, 0);
        assert_eq!(madness_cost.pips, &[Pip::Colored(ManaColor::R)]);
    }

    #[test]
    fn searing_blaze_targets_player_then_their_creature() {
        let id = card_id_by_name("Searing Blaze").expect("Searing Blaze in pool");
        let def = &CARD_DEFS[id as usize];
        assert!(def.is_castable());
        assert_eq!(def.target_spec, TargetSpec::PlayerThenTheirCreature);
    }

    #[test]
    fn counterspell_and_dispel_share_the_counter_program_with_distinct_filters() {
        let counterspell = &CARD_DEFS[card_id_by_name("Counterspell").unwrap() as usize];
        let dispel = &CARD_DEFS[card_id_by_name("Dispel").unwrap() as usize];

        assert!(counterspell.has_full_support());
        assert!(dispel.has_full_support());
        assert_eq!(counterspell.target_spec, TargetSpec::AnySpellOnStack);
        assert_eq!(dispel.target_spec, TargetSpec::InstantSpellOnStack);
        assert_eq!((counterspell.spell_effect)(), (dispel.spell_effect)());
        assert!(matches!(
            (counterspell.spell_effect)(),
            Some(EffectOp::Conditional {
                cond: EffectCond::TargetInZone(0, Zone::Stack),
                ..
            })
        ));
    }

    #[test]
    fn all_four_blasts_are_modal_with_symmetric_target_timing() {
        for (name, spell_spec, permanent_spec) in [
            (
                "Blue Elemental Blast",
                TargetSpec::RedSpellOnStack,
                TargetSpec::RedPermanent,
            ),
            (
                "Hydroblast",
                TargetSpec::AnySpellOnStack,
                TargetSpec::AnyPermanent,
            ),
            (
                "Pyroblast",
                TargetSpec::AnySpellOnStack,
                TargetSpec::AnyPermanent,
            ),
            (
                "Red Elemental Blast",
                TargetSpec::BlueSpellOnStack,
                TargetSpec::BluePermanent,
            ),
        ] {
            let def = &CARD_DEFS[card_id_by_name(name).unwrap() as usize];
            assert!(def.is_castable(), "{name}");
            assert_eq!(def.target_spec, spell_spec, "{name}");
            assert_eq!(
                def.mode2.as_ref().map(|mode| mode.target_spec),
                Some(permanent_spec),
                "{name}"
            );
        }

        let beb = &CARD_DEFS[card_id_by_name("Blue Elemental Blast").unwrap() as usize];
        let reb = &CARD_DEFS[card_id_by_name("Red Elemental Blast").unwrap() as usize];
        assert_eq!((beb.spell_effect)(), (reb.spell_effect)());
        assert_eq!(
            (beb.mode2.as_ref().unwrap().effect)(),
            (reb.mode2.as_ref().unwrap().effect)()
        );

        let hydro = &CARD_DEFS[card_id_by_name("Hydroblast").unwrap() as usize];
        assert!(matches!(
            (hydro.spell_effect)(),
            Some(EffectOp::Conditional {
                cond: EffectCond::TargetIsColor(0, ManaColor::R),
                ..
            })
        ));
    }

    #[test]
    fn card_colors_are_populated_from_the_json_pool() {
        let bolt = &CARD_DEFS[card_id_by_name("Lightning Bolt").unwrap() as usize];
        assert_eq!(bolt.colors, &[ManaColor::R]);
        let relic = &CARD_DEFS[card_id_by_name("Relic of Progenitus").unwrap() as usize];
        assert!(relic.colors.is_empty(), "Relic of Progenitus is colorless");
    }

    #[test]
    fn grab_the_prize_is_castable_with_a_mandatory_discard_additional_cost() {
        let id = card_id_by_name("Grab the Prize").expect("Grab the Prize in pool");
        let def = &CARD_DEFS[id as usize];
        assert!(def.is_castable());
        assert_eq!(
            def.additional_cost,
            Some([CostComponent::DiscardCards(1)].as_slice())
        );
    }

    #[test]
    fn fireblast_has_a_sacrifice_two_mountains_alt_cost() {
        let id = card_id_by_name("Fireblast").expect("Fireblast in pool");
        let def = &CARD_DEFS[id as usize];
        assert_eq!(
            def.alt_cost,
            Some([CostComponent::SacrificeLands(2)].as_slice())
        );
    }

    #[test]
    fn faithless_looting_and_lava_dart_have_flashback() {
        let looting = &CARD_DEFS[card_id_by_name("Faithless Looting").unwrap() as usize];
        let looting_cost = looting.flashback.as_ref().unwrap().cost;
        assert!(matches!(looting_cost, [CostComponent::Mana(_)]));
        let lava_dart = &CARD_DEFS[card_id_by_name("Lava Dart").unwrap() as usize];
        let lava_dart_cost = lava_dart.flashback.as_ref().unwrap().cost;
        assert!(matches!(lava_dart_cost, [CostComponent::SacrificeLands(1)]));
    }

    #[test]
    fn deep_analysis_composes_targeted_draw_and_ordered_flashback_cost() {
        let deep = &CARD_DEFS[card_id_by_name("Deep Analysis").unwrap() as usize];
        assert_eq!(deep.capability, CardCapability::Full);
        assert_eq!(deep.target_spec, TargetSpec::AnyPlayer);
        let flashback = deep.flashback.as_ref().expect("Deep Analysis flashback");
        assert!(matches!(
            flashback.cost,
            [
                CostComponent::Mana(Cost {
                    pips: [Pip::Colored(ManaColor::U)],
                    generic: 1,
                    x_count: 0,
                }),
                CostComponent::PayLife(3)
            ]
        ));
        assert!(matches!(
            (deep.spell_effect)(),
            Some(EffectOp::DrawCards {
                player: PlayerRef::Target(0),
                count: 2,
            })
        ));
    }

    #[test]
    fn masked_meower_has_haste_and_a_draw_activated_ability() {
        let id = card_id_by_name("Masked Meower").expect("Masked Meower in pool");
        let def = &CARD_DEFS[id as usize];
        assert!(def.keywords.has(Keywords::HASTE));
        assert_eq!(def.activated_abilities.len(), 1);
        assert_eq!(
            def.activated_abilities[0].cost,
            [CostComponent::DiscardCards(1), CostComponent::SacrificeSelf].as_slice()
        );
    }

    #[test]
    fn sneaky_snacker_has_flying() {
        let id = card_id_by_name("Sneaky Snacker").expect("Sneaky Snacker in pool");
        assert!(CARD_DEFS[id as usize].keywords.has(Keywords::FLYING));
    }

    #[test]
    fn bird_illusion_token_is_append_only_blue_flying_one_one() {
        assert_eq!(Subtype::Zombie.stable_id(), 54);
        assert_eq!(Subtype::Illusion.stable_id(), 55);

        let id = card_id_by_name("Bird Illusion Token")
            .expect("Bird Illusion Token should be codegen'd as a token");
        assert_eq!(
            id, 135,
            "new token must append without renumbering pool ids"
        );
        let def = &CARD_DEFS[id as usize];
        assert_eq!(def.capability, CardCapability::Full);
        assert!(def.is_token);
        assert!(!def.is_castable());
        assert_eq!(def.colors, &[ManaColor::U]);
        assert_eq!(def.types, &[CardType::Creature]);
        assert_eq!(def.subtypes, &[Subtype::Bird, Subtype::Illusion]);
        assert_eq!(def.power, Some(1));
        assert_eq!(def.toughness, Some(1));
        assert!(def.keywords.has(Keywords::FLYING));
    }

    #[test]
    fn blood_token_exists_with_its_draw_a_card_ability() {
        let id =
            card_id_by_name("Blood Token").expect("Blood Token should be codegen'd as a token");
        let def = &CARD_DEFS[id as usize];
        assert!(!def.is_castable(), "tokens are never cast");
        assert_eq!(def.activated_abilities.len(), 1);
    }

    // ---- Rally at the Hornburg increment -----------------------------

    #[test]
    fn great_furnace_is_a_second_mountain_that_is_also_an_artifact() {
        let id = card_id_by_name("Great Furnace").expect("Great Furnace in pool");
        let def = &CARD_DEFS[id as usize];
        assert!(def.is_land);
        assert!(def.has_type(CardType::Artifact));
        assert_eq!(def.produces_mana, &[ManaColor::R]);
        assert!(!def.is_castable());
        assert!((def.mana_ability)().is_some());
    }

    #[test]
    fn burning_tree_emissary_has_hybrid_cost_and_no_spell_effect_of_its_own() {
        let id = card_id_by_name("Burning-Tree Emissary").expect("Burning-Tree Emissary in pool");
        let def = &CARD_DEFS[id as usize];
        assert_eq!(
            def.cost.pips,
            &[
                Pip::Hybrid(ManaColor::R, ManaColor::G),
                Pip::Hybrid(ManaColor::R, ManaColor::G)
            ]
        );
        assert_eq!(def.subtypes, &[Subtype::Human, Subtype::Shaman]);
        match (def.spell_effect)() {
            Some(EffectOp::MoveObject {
                object: ObjectRef::ThisSource,
                to_zone: Zone::Battlefield,
            }) => {}
            other => panic!("expected MoveObject to Battlefield, got {other:?}"),
        }
    }

    #[test]
    fn chain_lightning_deals_3_damage_then_offers_the_copy_cost() {
        let id = card_id_by_name("Chain Lightning").expect("Chain Lightning in pool");
        let def = &CARD_DEFS[id as usize];
        assert_eq!(def.target_spec, TargetSpec::AnyTarget);
        match (def.spell_effect)() {
            Some(EffectOp::Sequence(ops)) => {
                assert_eq!(ops.len(), 2);
                assert_eq!(
                    ops[0],
                    EffectOp::DealDamage {
                        target: TargetRef::Target(0),
                        amount: 3
                    }
                );
                assert_eq!(
                    ops[1],
                    EffectOp::OfferAffectedPlayerSpellCopy {
                        affected: TargetRef::Target(0)
                    }
                );
            }
            other => {
                panic!("expected a 2-op Sequence (damage, then the copy offer), got {other:?}")
            }
        }
    }

    #[test]
    fn cast_into_the_fire_uses_the_variable_target_modal_program() {
        let id = card_id_by_name("Cast into the Fire").expect("Cast into the Fire in pool");
        let def = &CARD_DEFS[id as usize];
        assert_eq!(def.capability, CardCapability::Full);
        assert!(def.is_castable());
        assert_eq!(def.target_spec, TargetSpec::UpToTwoCreatures);
        assert_eq!(
            (def.spell_effect)(),
            Some(EffectOp::DamageAllTargets { amount: 1 })
        );
    }

    #[test]
    fn goblin_bushwhacker_has_kicker_r_and_no_static_haste() {
        let id = card_id_by_name("Goblin Bushwhacker").expect("Goblin Bushwhacker in pool");
        let def = &CARD_DEFS[id as usize];
        let kicker = def
            .kicker_cost
            .expect("Goblin Bushwhacker should have Kicker {R}");
        assert_eq!(kicker.generic, 0);
        assert_eq!(kicker.pips, &[Pip::Colored(ManaColor::R)]);
        assert!(
            !def.keywords.has(Keywords::HASTE),
            "haste is conditional on Kicker, not a static keyword"
        );
    }

    #[test]
    fn goblin_tomb_raider_has_no_static_haste_either() {
        // "As long as you control an artifact, gets +1/+0 and has haste" is
        // a conditional static ability (`engine::static_self_boost_for`),
        // not an unconditional `Keywords` bit.
        let id = card_id_by_name("Goblin Tomb Raider").expect("Goblin Tomb Raider in pool");
        let def = &CARD_DEFS[id as usize];
        assert!(!def.keywords.has(Keywords::HASTE));
        assert_eq!(def.power, Some(1));
        assert_eq!(def.toughness, Some(2));
    }

    #[test]
    fn clockwork_percussionist_has_haste() {
        let id =
            card_id_by_name("Clockwork Percussionist").expect("Clockwork Percussionist in pool");
        assert!(CARD_DEFS[id as usize].keywords.has(Keywords::HASTE));
    }

    #[test]
    fn experimental_synthesizer_has_a_sorcery_speed_only_sacrifice_ability() {
        let id =
            card_id_by_name("Experimental Synthesizer").expect("Experimental Synthesizer in pool");
        let def = &CARD_DEFS[id as usize];
        assert_eq!(def.activated_abilities.len(), 1);
        let ability = &def.activated_abilities[0];
        assert!(ability.sorcery_speed_only);
        assert_eq!(
            ability.cost,
            [
                CostComponent::Mana(Cost {
                    pips: &[Pip::Colored(ManaColor::R)],
                    generic: 2,
                    x_count: 0
                }),
                CostComponent::SacrificeSelf
            ]
            .as_slice()
        );
    }

    #[test]
    fn galvanic_blast_is_conditional_on_metalcraft() {
        let id = card_id_by_name("Galvanic Blast").expect("Galvanic Blast in pool");
        let def = &CARD_DEFS[id as usize];
        assert_eq!(def.target_spec, TargetSpec::AnyTarget);
        assert!(matches!(
            (def.spell_effect)(),
            Some(EffectOp::Conditional {
                cond: EffectCond::ControlsArtifactCount(3),
                ..
            })
        ));
    }

    #[test]
    fn end_the_festivities_hits_the_opponent_and_their_creatures() {
        let id = card_id_by_name("End the Festivities").expect("End the Festivities in pool");
        let def = &CARD_DEFS[id as usize];
        assert_eq!(def.target_spec, TargetSpec::None);
        assert_eq!(
            (def.spell_effect)(),
            Some(EffectOp::DamageOpponentAndTheirCreatures { amount: 1 })
        );
    }

    #[test]
    fn reckless_impulse_exiles_two_cards_until_owners_next_turn() {
        let id = card_id_by_name("Reckless Impulse").expect("Reckless Impulse in pool");
        let def = &CARD_DEFS[id as usize];
        assert_eq!(
            (def.spell_effect)(),
            Some(EffectOp::ImpulseDraw {
                count: 2,
                duration: crate::effect::ImpulseDuration::UntilOwnersNextTurn
            })
        );
    }

    #[test]
    fn rally_at_the_hornburg_creates_two_tokens_and_pumps_humans() {
        let id = card_id_by_name("Rally at the Hornburg").expect("Rally at the Hornburg in pool");
        let def = &CARD_DEFS[id as usize];
        match (def.spell_effect)() {
            Some(EffectOp::Sequence(ops)) => {
                assert_eq!(ops.len(), 3);
                assert!(matches!(ops[0], EffectOp::CreateToken { .. }));
                assert!(matches!(ops[1], EffectOp::CreateToken { .. }));
                assert!(matches!(
                    ops[2],
                    EffectOp::PumpControlled {
                        filter: crate::effect::CreatureFilter::ControlledWithSubtype(
                            Subtype::Human
                        ),
                        grant_haste: true,
                        ..
                    }
                ));
            }
            other => panic!("expected a 3-op Sequence, got {other:?}"),
        }
    }

    #[test]
    fn human_soldier_and_samurai_tokens_exist() {
        let hst = card_id_by_name("Human Soldier Token")
            .expect("Human Soldier Token should be codegen'd as a token");
        let def = &CARD_DEFS[hst as usize];
        assert!(!def.is_castable());
        assert_eq!(def.power, Some(1));
        assert_eq!(def.toughness, Some(1));
        assert_eq!(def.subtypes, &[Subtype::Human, Subtype::Soldier]);

        let samurai =
            card_id_by_name("Samurai Token").expect("Samurai Token should be codegen'd as a token");
        let sdef = &CARD_DEFS[samurai as usize];
        assert_eq!(sdef.power, Some(2));
        assert_eq!(sdef.toughness, Some(2));
        assert!(sdef.keywords.has(Keywords::VIGILANCE));
    }
}
