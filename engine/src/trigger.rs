//! State-based actions and trigger collection.
//!
//! After every resolution (a spell/ability resolving, a land entering, a
//! turn-based draw -- anything that produced `CommittedEvent`s), the engine
//! calls `collect_and_process`: it matches the already-committed events,
//! runs SBAs to a fixed point, then matches any events those SBAs created.
//! The combined newly-pending triggers are returned in APNAP order for
//! `engine.rs` to place on the stack (or, if 2+ share a controller, to ask
//! that controller to order via `engine::Decision::OrderTriggers`).

use crate::card_def::{
    CardType, DynamicValueDef, Keywords, OptionalAdditionalCostDef, Subtype, TargetSpec,
};
use crate::effect::{EffectCond, EffectObjectBinding, EffectOp, ObjectRef, PlayerRef, TargetRef};
use crate::event::CommittedEvent;
use crate::ids::{ObjectId, PlayerId};
use crate::state::{
    AbilitySourceContractV4, GameState, InitiativeTriggerKindV1, PaidCostRefV4,
    StackTargetContractV4, Target, UndercityRoomV1, Zone,
};
use serde::{Deserialize, Serialize};

/// Trigger conditions this increment's kernel can match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriggerCondition {
    /// The permanent itself enters the battlefield (Voldaren Epicure).
    Etb,
    /// The permanent itself enters while its controller controls the named
    /// number of other permanents with the effective subtype. This is the
    /// trigger-time half of an intervening-if condition.
    EtbControlsOtherSubtypeCount { subtype: Subtype, minimum_count: u8 },
    /// The permanent itself deals damage. Unused by any Burn 16 card this
    /// increment (kept from increment 2's shape -- see `trigger_matches`).
    DealsDamage,
    /// The controller casts an instant or sorcery spell -- any such
    /// spell, not just ones this permanent's controller controls the
    /// *source* of (Guttersnipe and Murmuring Mystic: "whenever you cast an
    /// instant or sorcery spell"). Matched against
    /// `CommittedEvent::SpellCast`, logged by `engine::finalize_cast`.
    CastInstantOrSorcery,
    /// This exact source spell is cast. Creature cast triggers function
    /// while their source is on the stack.
    CastSelf,
    /// The controller draws their `n`th card since the current turn began
    /// (Sneaky Snacker: "whenever you draw your third card in a turn").
    /// This ability functions from the graveyard (`home_zone: Graveyard`
    /// on its `TriggeredAbilityDef`), so unlike every other condition
    /// here it's checked with the source in the graveyard, not the
    /// battlefield.
    DrawNth(u32),
    /// This permanent moves from the battlefield to its owner's graveyard
    /// (700.4 "dies", generalized to any battlefield -> graveyard
    /// transition regardless of card type -- Clockwork Percussionist's
    /// death and Experimental Synthesizer's own sacrifice-cost ability both
    /// have this exact event shape, and neither card's Java source
    /// distinguishes "dies" from "put into a graveyard from the
    /// battlefield" in a way this pool needs to model separately). Checked
    /// with the source *currently in the graveyard* (`home_zone: Graveyard`
    /// -- see `Etb`'s doc for why the outer gate uses the post-event zone,
    /// not a "functions from" zone: by the time `collect_and_process` looks,
    /// the object has already moved there).
    LeftBattlefieldToGraveyard,
    /// This permanent leaves the battlefield for any destination. Unlike a
    /// dies trigger, this includes exile, hand, and library moves.
    LeftBattlefield,
    /// This permanent's controller sacrifices another permanent that had
    /// the named effective subtype immediately before leaving.
    SacrificeAnotherWithSubtype(Subtype),
    /// This creature deals combat damage to a player. The committed marker
    /// carries the source's exact zone-change generation.
    DealsCombatDamageToPlayer,
}

pub struct TriggeredAbilityDef {
    pub condition: TriggerCondition,
    /// Which zone the source must be in for this ability to function.
    /// Every keyworded/triggered ability in MTG works from the
    /// battlefield unless its reminder text says otherwise (like Sneaky
    /// Snacker's "from your graveyard").
    pub home_zone: Zone,
    /// 603.4 intervening-if gate checked when the event happens. An unkicked
    /// Goblin Bushwhacker must not create a stack item at all (as opposed to
    /// creating a trigger whose effect later resolves to a no-op).
    pub intervening_if_kicked: bool,
    /// Faerie Miscreant's trigger-time 603.4 gate. This is deliberately a
    /// same-definition board predicate rather than a card-name branch. The
    /// matching resolution-time gate lives in `EffectCond`.
    pub intervening_if_controls_another_source_card: bool,
    pub effect: fn() -> EffectOp,
}

pub(crate) fn materialize_trigger_effect(
    trigger: &TriggeredAbilityDef,
    source: ObjectId,
    state: &GameState,
) -> EffectOp {
    match (trigger.effect)() {
        EffectOp::BindPlusOnePlusOneCounterToTriggerSource => {
            let live = state.objects.get(source);
            EffectOp::PutPlusOnePlusOneCounterOnBoundObject {
                object: EffectObjectBinding {
                    object: source,
                    expected_zone: live.zone,
                    expected_zone_change_count: live.zone_change_count,
                },
            }
        }
        EffectOp::MaterializeStormCopies => {
            crate::engine::materialize_storm_copy_binding(state, source)
                .map(|binding| EffectOp::CreateStormCopies { binding })
                .unwrap_or(EffectOp::MaterializeStormCopies)
        }
        effect => effect,
    }
}

fn guttersnipe_effect() -> EffectOp {
    // Guttersnipe deals 2 damage to each opponent.
    EffectOp::DealDamage {
        target: TargetRef::Opponent,
        amount: 2,
    }
}

fn murmuring_mystic_effect() -> EffectOp {
    // Create a 1/1 blue Bird Illusion creature token with flying. Token
    // characteristics live in the generated CardDef; the trigger uses the
    // same generic CreateToken leaf as Voldaren Epicure and Rally cards.
    let bird_illusion = crate::card_def::card_id_by_name("Bird Illusion Token")
        .expect("Bird Illusion Token in CARD_DEFS");
    EffectOp::CreateToken {
        token_def: bird_illusion,
        controller: PlayerRef::Controller,
    }
}

fn voldaren_epicure_effect() -> EffectOp {
    // It deals 1 damage to each opponent. Create a Blood token.
    let blood_token =
        crate::card_def::card_id_by_name("Blood Token").expect("Blood Token in CARD_DEFS");
    EffectOp::Sequence(vec![
        EffectOp::DealDamage {
            target: TargetRef::Opponent,
            amount: 1,
        },
        EffectOp::CreateToken {
            token_def: blood_token,
            controller: PlayerRef::Controller,
        },
    ])
}

fn generous_ent_effect() -> EffectOp {
    let food = crate::card_def::card_id_by_name("Food Token").expect("Food Token in CARD_DEFS");
    EffectOp::CreateToken {
        token_def: food,
        controller: PlayerRef::Controller,
    }
}

fn gingerbread_cabin_effect() -> EffectOp {
    let food = crate::card_def::card_id_by_name("Food Token").expect("Food Token in CARD_DEFS");
    EffectOp::Conditional {
        cond: EffectCond::ControlsOtherSubtypeCount {
            subtype: Subtype::Forest,
            minimum_count: 3,
        },
        then: Box::new(EffectOp::CreateToken {
            token_def: food,
            controller: PlayerRef::Controller,
        }),
        else_: Box::new(EffectOp::Sequence(Vec::new())),
    }
}

fn writhing_chrysalis_cast_effect() -> EffectOp {
    let spawn = crate::card_def::card_id_by_name("Eldrazi Spawn Token")
        .expect("Eldrazi Spawn Token in CARD_DEFS");
    EffectOp::Sequence(vec![
        EffectOp::CreateToken {
            token_def: spawn,
            controller: PlayerRef::Controller,
        },
        EffectOp::CreateToken {
            token_def: spawn,
            controller: PlayerRef::Controller,
        },
    ])
}

fn writhing_chrysalis_counter_marker_effect() -> EffectOp {
    EffectOp::BindPlusOnePlusOneCounterToTriggerSource
}

fn blood_fountain_effect() -> EffectOp {
    let blood = crate::card_def::card_id_by_name("Blood Token").expect("Blood Token in CARD_DEFS");
    EffectOp::CreateToken {
        token_def: blood,
        controller: PlayerRef::Controller,
    }
}

fn gain_three_life_effect() -> EffectOp {
    EffectOp::GainLife {
        player: PlayerRef::Controller,
        amount: 3,
    }
}

fn gatecreeper_vine_effect() -> EffectOp {
    EffectOp::SearchLibraryToHand {
        player: PlayerRef::Controller,
        filter: crate::effect::LibraryCardFilter::BasicLandOrGate,
    }
}

fn balustrade_spy_effect() -> EffectOp {
    EffectOp::RevealUntilCardTypeAndMill {
        player: PlayerRef::Target(0),
        card_type: CardType::Land,
    }
}

fn lotleth_giant_effect() -> EffectOp {
    EffectOp::DealDamageDynamic {
        target: TargetRef::Target(0),
        amount: DynamicValueDef::ControllerGraveyardCardsWithType(CardType::Creature),
    }
}

fn mesmeric_fiend_exile_effect() -> EffectOp {
    EffectOp::RevealHandChooseNonlandToLinkedExile {
        player: PlayerRef::Target(0),
    }
}

fn mesmeric_fiend_return_effect() -> EffectOp {
    EffectOp::ReturnLinkedExiledCardToOwnersHand
}

fn sneaky_snacker_effect() -> EffectOp {
    // Return Sneaky Snacker from your graveyard to the battlefield tapped.
    EffectOp::Sequence(vec![
        EffectOp::MoveObject {
            object: crate::effect::ObjectRef::ThisSource,
            to_zone: Zone::Battlefield,
        },
        EffectOp::TapObject {
            object: crate::effect::ObjectRef::ThisSource,
        },
    ])
}

const GUTTERSNIPE_TRIGGERS: [TriggeredAbilityDef; 1] = [TriggeredAbilityDef {
    condition: TriggerCondition::CastInstantOrSorcery,
    home_zone: Zone::Battlefield,
    intervening_if_kicked: false,
    intervening_if_controls_another_source_card: false,
    effect: guttersnipe_effect,
}];
const MURMURING_MYSTIC_TRIGGERS: [TriggeredAbilityDef; 1] = [TriggeredAbilityDef {
    condition: TriggerCondition::CastInstantOrSorcery,
    home_zone: Zone::Battlefield,
    intervening_if_kicked: false,
    intervening_if_controls_another_source_card: false,
    effect: murmuring_mystic_effect,
}];
const VOLDAREN_EPICURE_TRIGGERS: [TriggeredAbilityDef; 1] = [TriggeredAbilityDef {
    condition: TriggerCondition::Etb,
    home_zone: Zone::Battlefield,
    intervening_if_kicked: false,
    intervening_if_controls_another_source_card: false,
    effect: voldaren_epicure_effect,
}];
const GENEROUS_ENT_TRIGGERS: [TriggeredAbilityDef; 1] = [TriggeredAbilityDef {
    condition: TriggerCondition::Etb,
    home_zone: Zone::Battlefield,
    intervening_if_kicked: false,
    intervening_if_controls_another_source_card: false,
    effect: generous_ent_effect,
}];
const GINGERBREAD_CABIN_TRIGGERS: [TriggeredAbilityDef; 1] = [TriggeredAbilityDef {
    condition: TriggerCondition::EtbControlsOtherSubtypeCount {
        subtype: Subtype::Forest,
        minimum_count: 3,
    },
    home_zone: Zone::Battlefield,
    intervening_if_kicked: false,
    intervening_if_controls_another_source_card: false,
    effect: gingerbread_cabin_effect,
}];
const WRITHING_CHRYSALIS_TRIGGERS: [TriggeredAbilityDef; 2] = [
    TriggeredAbilityDef {
        condition: TriggerCondition::CastSelf,
        home_zone: Zone::Stack,
        intervening_if_kicked: false,
        intervening_if_controls_another_source_card: false,
        effect: writhing_chrysalis_cast_effect,
    },
    TriggeredAbilityDef {
        condition: TriggerCondition::SacrificeAnotherWithSubtype(Subtype::Eldrazi),
        home_zone: Zone::Battlefield,
        intervening_if_kicked: false,
        intervening_if_controls_another_source_card: false,
        effect: writhing_chrysalis_counter_marker_effect,
    },
];
const BLOOD_FOUNTAIN_TRIGGERS: [TriggeredAbilityDef; 1] = [TriggeredAbilityDef {
    condition: TriggerCondition::Etb,
    home_zone: Zone::Battlefield,
    intervening_if_kicked: false,
    intervening_if_controls_another_source_card: false,
    effect: blood_fountain_effect,
}];
const SAGU_WILDLING_TRIGGERS: [TriggeredAbilityDef; 1] = [TriggeredAbilityDef {
    condition: TriggerCondition::Etb,
    home_zone: Zone::Battlefield,
    intervening_if_kicked: false,
    intervening_if_controls_another_source_card: false,
    effect: gain_three_life_effect,
}];
const GATECREEPER_VINE_TRIGGERS: [TriggeredAbilityDef; 1] = [TriggeredAbilityDef {
    condition: TriggerCondition::Etb,
    home_zone: Zone::Battlefield,
    intervening_if_kicked: false,
    intervening_if_controls_another_source_card: false,
    effect: gatecreeper_vine_effect,
}];
const BALUSTRADE_SPY_TRIGGERS: [TriggeredAbilityDef; 1] = [TriggeredAbilityDef {
    condition: TriggerCondition::Etb,
    home_zone: Zone::Battlefield,
    intervening_if_kicked: false,
    intervening_if_controls_another_source_card: false,
    effect: balustrade_spy_effect,
}];
const LOTLETH_GIANT_TRIGGERS: [TriggeredAbilityDef; 1] = [TriggeredAbilityDef {
    condition: TriggerCondition::Etb,
    home_zone: Zone::Battlefield,
    intervening_if_kicked: false,
    intervening_if_controls_another_source_card: false,
    effect: lotleth_giant_effect,
}];
const MESMERIC_FIEND_TRIGGERS: [TriggeredAbilityDef; 2] = [
    TriggeredAbilityDef {
        condition: TriggerCondition::Etb,
        home_zone: Zone::Battlefield,
        intervening_if_kicked: false,
        intervening_if_controls_another_source_card: false,
        effect: mesmeric_fiend_exile_effect,
    },
    TriggeredAbilityDef {
        condition: TriggerCondition::LeftBattlefield,
        // Leave triggers are matched from battlefield last-known
        // information and therefore ignore the source's post-event zone.
        home_zone: Zone::Graveyard,
        intervening_if_kicked: false,
        intervening_if_controls_another_source_card: false,
        effect: mesmeric_fiend_return_effect,
    },
];
const GAIN_THREE_LIFE_ETB_TRIGGERS: [TriggeredAbilityDef; 1] = [TriggeredAbilityDef {
    condition: TriggerCondition::Etb,
    home_zone: Zone::Battlefield,
    intervening_if_kicked: false,
    intervening_if_controls_another_source_card: false,
    effect: gain_three_life_effect,
}];
const SNEAKY_SNACKER_TRIGGERS: [TriggeredAbilityDef; 1] = [TriggeredAbilityDef {
    condition: TriggerCondition::DrawNth(3),
    home_zone: Zone::Graveyard,
    intervening_if_kicked: false,
    intervening_if_controls_another_source_card: false,
    effect: sneaky_snacker_effect,
}];

fn burning_tree_emissary_effect() -> EffectOp {
    // When Burning-Tree Emissary enters the battlefield, add {R}{G}.
    EffectOp::AddMana {
        player: PlayerRef::Controller,
        colors: vec![crate::mana::ManaColor::R, crate::mana::ManaColor::G],
    }
}

fn clockwork_percussionist_dies_effect() -> EffectOp {
    // When Clockwork Percussionist dies, exile the top card of your
    // library. You may play it until the end of your next turn.
    EffectOp::ImpulseDraw {
        count: 1,
        duration: crate::effect::ImpulseDuration::UntilOwnersNextTurn,
    }
}

fn ichor_wellspring_draw_effect() -> EffectOp {
    EffectOp::DrawCards {
        player: PlayerRef::Controller,
        count: 1,
    }
}

fn job_select_effect() -> EffectOp {
    let hero = crate::card_def::card_id_by_name("Hero Token").expect("Hero Token in CARD_DEFS");
    EffectOp::CreateTokenAndAttachSource { token_def: hero }
}

fn nihil_spellbomb_graveyard_effect() -> EffectOp {
    EffectOp::MayPayManaThen {
        player: PlayerRef::Controller,
        colored: vec![crate::mana::ManaColor::B],
        generic: 0,
        then: Box::new(EffectOp::DrawCards {
            player: PlayerRef::Controller,
            count: 1,
        }),
    }
}

fn faerie_miscreant_effect() -> EffectOp {
    EffectOp::Conditional {
        cond: EffectCond::ControlsAnotherSourceCard,
        then: Box::new(EffectOp::DrawCards {
            player: PlayerRef::Controller,
            count: 1,
        }),
        else_: Box::new(EffectOp::Sequence(vec![])),
    }
}

fn faerie_seer_effect() -> EffectOp {
    EffectOp::Scry {
        player: PlayerRef::Controller,
        count: 2,
    }
}

fn outlaw_medic_dies_effect() -> EffectOp {
    EffectOp::DrawCards {
        player: PlayerRef::Controller,
        count: 1,
    }
}

fn refurbished_familiar_etb_effect() -> EffectOp {
    // The kernel is strictly 1v1. The opponent chooses and discards one
    // card when possible; otherwise the Familiar's controller draws one.
    EffectOp::Conditional {
        cond: EffectCond::OpponentHasCardsInHand,
        then: Box::new(EffectOp::DiscardCards {
            player: PlayerRef::Opponent,
            count: 1,
        }),
        else_: Box::new(EffectOp::DrawCards {
            player: PlayerRef::Controller,
            count: 1,
        }),
    }
}

fn squadron_hawk_etb_effect() -> EffectOp {
    let hawk =
        crate::card_def::card_id_by_name("Squadron Hawk").expect("Squadron Hawk in CARD_DEFS");
    EffectOp::SearchLibraryToHandUpTo {
        player: PlayerRef::Controller,
        filter: crate::effect::LibraryCardFilter::CardDefinition(hawk),
        max_targets: 3,
    }
}

fn bind_the_monster_etb_effect() -> EffectOp {
    EffectOp::TapAttachedCreatureAndDamageControllerByPower
}

fn harrier_strix_etb_effect() -> EffectOp {
    EffectOp::TapObject {
        object: ObjectRef::Target(0),
    }
}

fn humbling_elder_etb_effect() -> EffectOp {
    EffectOp::PumpTargetUntilEndOfTurnDynamic {
        target: TargetRef::Target(0),
        power: DynamicValueDef::Fixed(-2),
        toughness: DynamicValueDef::Fixed(0),
    }
}

fn moon_circuit_hacker_combat_effect_for_entered_this_turn(entered_this_turn: bool) -> EffectOp {
    let after_draw = if entered_this_turn {
        EffectOp::Sequence(vec![])
    } else {
        EffectOp::DiscardCards {
            player: PlayerRef::Controller,
            count: 1,
        }
    };
    EffectOp::Choice {
        controller: PlayerRef::Controller,
        options: vec![
            EffectOp::Sequence(vec![]),
            EffectOp::Sequence(vec![
                EffectOp::DrawCards {
                    player: PlayerRef::Controller,
                    count: 1,
                },
                after_draw,
            ]),
        ],
    }
}

fn moon_circuit_hacker_combat_effect() -> EffectOp {
    // Definition inventory placeholder. Event matching freezes the actual
    // source-incarnation fact into one of the two canonical programs below.
    moon_circuit_hacker_combat_effect_for_entered_this_turn(false)
}

fn ninja_of_the_deep_hours_combat_effect() -> EffectOp {
    EffectOp::Choice {
        controller: PlayerRef::Controller,
        options: vec![
            EffectOp::Sequence(vec![]),
            EffectOp::DrawCards {
                player: PlayerRef::Controller,
                count: 1,
            },
        ],
    }
}

fn saiba_cryptomancer_etb_effect() -> EffectOp {
    EffectOp::BackupTarget {
        target: ObjectRef::Target(0),
        keyword: Keywords::HEXPROOF,
    }
}

fn spellstutter_sprite_etb_effect() -> EffectOp {
    EffectOp::MoveObject {
        object: ObjectRef::Target(0),
        to_zone: Zone::Graveyard,
    }
}

fn lembas_etb_effect() -> EffectOp {
    EffectOp::Sequence(vec![
        EffectOp::Scry {
            player: PlayerRef::Controller,
            count: 1,
        },
        EffectOp::DrawCards {
            player: PlayerRef::Controller,
            count: 1,
        },
    ])
}

fn lembas_graveyard_effect() -> EffectOp {
    EffectOp::ShuffleTriggerSourceIntoOwnersLibrary
}

fn weather_the_storm_cast_effect() -> EffectOp {
    EffectOp::MaterializeStormCopies
}

fn journey_to_nowhere_etb_effect() -> EffectOp {
    EffectOp::ExileTargetLinkedToSource {
        object: crate::effect::ObjectRef::Target(0),
    }
}

fn journey_to_nowhere_ltb_effect() -> EffectOp {
    EffectOp::ReturnObjectsExiledBySource
}

fn masked_vandal_etb_effect() -> EffectOp {
    EffectOp::MayExileFromPlayersGraveyardMatchingThen {
        player: PlayerRef::Controller,
        card_type: CardType::Creature,
        then: Box::new(EffectOp::MoveObject {
            object: ObjectRef::Target(0),
            to_zone: Zone::Exile,
        }),
    }
}

fn troublemaker_ouphe_etb_effect() -> EffectOp {
    EffectOp::Conditional {
        cond: EffectCond::OptionalAdditionalCostPaid(OptionalAdditionalCostDef::Bargain),
        then: Box::new(EffectOp::MoveObject {
            object: ObjectRef::Target(0),
            to_zone: Zone::Exile,
        }),
        else_: Box::new(EffectOp::Sequence(Vec::new())),
    }
}

fn vitu_ghazi_inspector_etb_effect() -> EffectOp {
    EffectOp::Conditional {
        cond: EffectCond::OptionalAdditionalCostPaid(OptionalAdditionalCostDef::CollectEvidence {
            minimum_mana_value: 6,
        }),
        then: Box::new(EffectOp::Sequence(vec![
            EffectOp::PutPlusOnePlusOneCounter {
                object: ObjectRef::Target(0),
            },
            EffectOp::GainLife {
                player: PlayerRef::Controller,
                amount: 2,
            },
        ])),
        else_: Box::new(EffectOp::Sequence(Vec::new())),
    }
}

fn avenging_hunter_etb_effect() -> EffectOp {
    EffectOp::TakeInitiative {
        player: PlayerRef::Controller,
    }
}

fn experimental_synthesizer_impulse_effect() -> EffectOp {
    // When Experimental Synthesizer enters or leaves the battlefield, exile
    // the top card of your library. Until end of turn, you may play that
    // card. Both triggers share this one effect -- the card's text is
    // identical either way.
    EffectOp::ImpulseDraw {
        count: 1,
        duration: crate::effect::ImpulseDuration::EndOfTurn,
    }
}

fn goblin_bushwhacker_effect() -> EffectOp {
    // When this creature enters, if it was kicked, creatures you control
    // get +1/+0 and gain haste until end of turn.
    EffectOp::Conditional {
        cond: EffectCond::WasKicked,
        then: Box::new(EffectOp::PumpControlled {
            filter: crate::effect::CreatureFilter::AnyControlled,
            power: 1,
            toughness: 0,
            grant_haste: true,
        }),
        else_: Box::new(EffectOp::Sequence(vec![])),
    }
}

const BURNING_TREE_EMISSARY_TRIGGERS: [TriggeredAbilityDef; 1] = [TriggeredAbilityDef {
    condition: TriggerCondition::Etb,
    home_zone: Zone::Battlefield,
    intervening_if_kicked: false,
    intervening_if_controls_another_source_card: false,
    effect: burning_tree_emissary_effect,
}];
const CLOCKWORK_PERCUSSIONIST_TRIGGERS: [TriggeredAbilityDef; 1] = [TriggeredAbilityDef {
    condition: TriggerCondition::LeftBattlefieldToGraveyard,
    home_zone: Zone::Graveyard,
    intervening_if_kicked: false,
    intervening_if_controls_another_source_card: false,
    effect: clockwork_percussionist_dies_effect,
}];
const ICHOR_WELLSPRING_TRIGGERS: [TriggeredAbilityDef; 2] = [
    TriggeredAbilityDef {
        condition: TriggerCondition::Etb,
        home_zone: Zone::Battlefield,
        intervening_if_kicked: false,
        intervening_if_controls_another_source_card: false,
        effect: ichor_wellspring_draw_effect,
    },
    TriggeredAbilityDef {
        condition: TriggerCondition::LeftBattlefieldToGraveyard,
        home_zone: Zone::Graveyard,
        intervening_if_kicked: false,
        intervening_if_controls_another_source_card: false,
        effect: ichor_wellspring_draw_effect,
    },
];
const CRYOGEN_RELIC_TRIGGERS: [TriggeredAbilityDef; 2] = [
    TriggeredAbilityDef {
        condition: TriggerCondition::Etb,
        home_zone: Zone::Battlefield,
        intervening_if_kicked: false,
        intervening_if_controls_another_source_card: false,
        effect: ichor_wellspring_draw_effect,
    },
    TriggeredAbilityDef {
        condition: TriggerCondition::LeftBattlefield,
        home_zone: Zone::Graveyard,
        intervening_if_kicked: false,
        intervening_if_controls_another_source_card: false,
        effect: ichor_wellspring_draw_effect,
    },
];
const JOB_SELECT_TRIGGERS: [TriggeredAbilityDef; 1] = [TriggeredAbilityDef {
    condition: TriggerCondition::Etb,
    home_zone: Zone::Battlefield,
    intervening_if_kicked: false,
    intervening_if_controls_another_source_card: false,
    effect: job_select_effect,
}];
const NIHIL_SPELLBOMB_TRIGGERS: [TriggeredAbilityDef; 1] = [TriggeredAbilityDef {
    condition: TriggerCondition::LeftBattlefieldToGraveyard,
    home_zone: Zone::Graveyard,
    intervening_if_kicked: false,
    intervening_if_controls_another_source_card: false,
    effect: nihil_spellbomb_graveyard_effect,
}];
const EXPERIMENTAL_SYNTHESIZER_TRIGGERS: [TriggeredAbilityDef; 2] = [
    TriggeredAbilityDef {
        condition: TriggerCondition::Etb,
        home_zone: Zone::Battlefield,
        intervening_if_kicked: false,
        intervening_if_controls_another_source_card: false,
        effect: experimental_synthesizer_impulse_effect,
    },
    TriggeredAbilityDef {
        condition: TriggerCondition::LeftBattlefield,
        // Leave-the-battlefield conditions use last-known information and
        // therefore deliberately ignore this post-event zone gate.
        home_zone: Zone::Graveyard,
        intervening_if_kicked: false,
        intervening_if_controls_another_source_card: false,
        effect: experimental_synthesizer_impulse_effect,
    },
];
const GOBLIN_BUSHWHACKER_TRIGGERS: [TriggeredAbilityDef; 1] = [TriggeredAbilityDef {
    condition: TriggerCondition::Etb,
    home_zone: Zone::Battlefield,
    intervening_if_kicked: true,
    intervening_if_controls_another_source_card: false,
    effect: goblin_bushwhacker_effect,
}];

const FAERIE_MISCREANT_TRIGGERS: [TriggeredAbilityDef; 1] = [TriggeredAbilityDef {
    condition: TriggerCondition::Etb,
    home_zone: Zone::Battlefield,
    intervening_if_kicked: false,
    intervening_if_controls_another_source_card: true,
    effect: faerie_miscreant_effect,
}];

const FAERIE_SEER_TRIGGERS: [TriggeredAbilityDef; 1] = [TriggeredAbilityDef {
    condition: TriggerCondition::Etb,
    home_zone: Zone::Battlefield,
    intervening_if_kicked: false,
    intervening_if_controls_another_source_card: false,
    effect: faerie_seer_effect,
}];

const OUTLAW_MEDIC_TRIGGERS: [TriggeredAbilityDef; 1] = [TriggeredAbilityDef {
    condition: TriggerCondition::LeftBattlefieldToGraveyard,
    home_zone: Zone::Graveyard,
    intervening_if_kicked: false,
    intervening_if_controls_another_source_card: false,
    effect: outlaw_medic_dies_effect,
}];

const REFURBISHED_FAMILIAR_TRIGGERS: [TriggeredAbilityDef; 1] = [TriggeredAbilityDef {
    condition: TriggerCondition::Etb,
    home_zone: Zone::Battlefield,
    intervening_if_kicked: false,
    intervening_if_controls_another_source_card: false,
    effect: refurbished_familiar_etb_effect,
}];

const SQUADRON_HAWK_TRIGGERS: [TriggeredAbilityDef; 1] = [TriggeredAbilityDef {
    condition: TriggerCondition::Etb,
    home_zone: Zone::Battlefield,
    intervening_if_kicked: false,
    intervening_if_controls_another_source_card: false,
    effect: squadron_hawk_etb_effect,
}];

const BIND_THE_MONSTER_TRIGGERS: [TriggeredAbilityDef; 1] = [TriggeredAbilityDef {
    condition: TriggerCondition::Etb,
    home_zone: Zone::Battlefield,
    intervening_if_kicked: false,
    intervening_if_controls_another_source_card: false,
    effect: bind_the_monster_etb_effect,
}];

const HARRIER_STRIX_TRIGGERS: [TriggeredAbilityDef; 1] = [TriggeredAbilityDef {
    condition: TriggerCondition::Etb,
    home_zone: Zone::Battlefield,
    intervening_if_kicked: false,
    intervening_if_controls_another_source_card: false,
    effect: harrier_strix_etb_effect,
}];

const HUMBLING_ELDER_TRIGGERS: [TriggeredAbilityDef; 1] = [TriggeredAbilityDef {
    condition: TriggerCondition::Etb,
    home_zone: Zone::Battlefield,
    intervening_if_kicked: false,
    intervening_if_controls_another_source_card: false,
    effect: humbling_elder_etb_effect,
}];

const MOON_CIRCUIT_HACKER_TRIGGERS: [TriggeredAbilityDef; 1] = [TriggeredAbilityDef {
    condition: TriggerCondition::DealsCombatDamageToPlayer,
    home_zone: Zone::Battlefield,
    intervening_if_kicked: false,
    intervening_if_controls_another_source_card: false,
    effect: moon_circuit_hacker_combat_effect,
}];

const NINJA_OF_THE_DEEP_HOURS_TRIGGERS: [TriggeredAbilityDef; 1] = [TriggeredAbilityDef {
    condition: TriggerCondition::DealsCombatDamageToPlayer,
    home_zone: Zone::Battlefield,
    intervening_if_kicked: false,
    intervening_if_controls_another_source_card: false,
    effect: ninja_of_the_deep_hours_combat_effect,
}];

const SAIBA_CRYPTOMANCER_TRIGGERS: [TriggeredAbilityDef; 1] = [TriggeredAbilityDef {
    condition: TriggerCondition::Etb,
    home_zone: Zone::Battlefield,
    intervening_if_kicked: false,
    intervening_if_controls_another_source_card: false,
    effect: saiba_cryptomancer_etb_effect,
}];

const SPELLSTUTTER_SPRITE_TRIGGERS: [TriggeredAbilityDef; 1] = [TriggeredAbilityDef {
    condition: TriggerCondition::Etb,
    home_zone: Zone::Battlefield,
    intervening_if_kicked: false,
    intervening_if_controls_another_source_card: false,
    effect: spellstutter_sprite_etb_effect,
}];

const LEMBAS_TRIGGERS: [TriggeredAbilityDef; 2] = [
    TriggeredAbilityDef {
        condition: TriggerCondition::Etb,
        home_zone: Zone::Battlefield,
        intervening_if_kicked: false,
        intervening_if_controls_another_source_card: false,
        effect: lembas_etb_effect,
    },
    TriggeredAbilityDef {
        condition: TriggerCondition::LeftBattlefieldToGraveyard,
        home_zone: Zone::Graveyard,
        intervening_if_kicked: false,
        intervening_if_controls_another_source_card: false,
        effect: lembas_graveyard_effect,
    },
];

const WEATHER_THE_STORM_TRIGGERS: [TriggeredAbilityDef; 1] = [TriggeredAbilityDef {
    condition: TriggerCondition::CastSelf,
    home_zone: Zone::Stack,
    intervening_if_kicked: false,
    intervening_if_controls_another_source_card: false,
    effect: weather_the_storm_cast_effect,
}];

const JOURNEY_TO_NOWHERE_TRIGGERS: [TriggeredAbilityDef; 2] = [
    TriggeredAbilityDef {
        condition: TriggerCondition::Etb,
        home_zone: Zone::Battlefield,
        intervening_if_kicked: false,
        intervening_if_controls_another_source_card: false,
        effect: journey_to_nowhere_etb_effect,
    },
    TriggeredAbilityDef {
        condition: TriggerCondition::LeftBattlefield,
        // Leave-the-battlefield triggers use the historical battlefield
        // incarnation and deliberately ignore this post-event zone gate.
        home_zone: Zone::Graveyard,
        intervening_if_kicked: false,
        intervening_if_controls_another_source_card: false,
        effect: journey_to_nowhere_ltb_effect,
    },
];
const MASKED_VANDAL_TRIGGERS: [TriggeredAbilityDef; 1] = [TriggeredAbilityDef {
    condition: TriggerCondition::Etb,
    home_zone: Zone::Battlefield,
    intervening_if_kicked: false,
    intervening_if_controls_another_source_card: false,
    effect: masked_vandal_etb_effect,
}];

const TROUBLEMAKER_OUPHE_TRIGGERS: [TriggeredAbilityDef; 1] = [TriggeredAbilityDef {
    condition: TriggerCondition::Etb,
    home_zone: Zone::Battlefield,
    intervening_if_kicked: false,
    intervening_if_controls_another_source_card: false,
    effect: troublemaker_ouphe_etb_effect,
}];

const VITU_GHAZI_INSPECTOR_TRIGGERS: [TriggeredAbilityDef; 1] = [TriggeredAbilityDef {
    condition: TriggerCondition::Etb,
    home_zone: Zone::Battlefield,
    intervening_if_kicked: false,
    intervening_if_controls_another_source_card: false,
    effect: vitu_ghazi_inspector_etb_effect,
}];

const AVENGING_HUNTER_TRIGGERS: [TriggeredAbilityDef; 1] = [TriggeredAbilityDef {
    condition: TriggerCondition::Etb,
    home_zone: Zone::Battlefield,
    intervening_if_kicked: false,
    intervening_if_controls_another_source_card: false,
    effect: avenging_hunter_etb_effect,
}];

/// The pool's implemented triggered abilities, matched by card name (ids are
/// codegen-assigned from `cards_v1.json`'s array order and not worth
/// duplicating as constants here -- see `build.rs`'s module doc on id
/// stability). Every other card in the pool has no triggered ability
/// implemented and falls through to `&[]`.
pub fn triggers_for(card_def: u16) -> &'static [TriggeredAbilityDef] {
    let Some(card) = crate::card_def::CARD_DEFS.get(card_def as usize) else {
        return &[];
    };
    if !card.is_executable() {
        return &[];
    }
    if card.equipment.is_some_and(|equipment| equipment.job_select) {
        return &JOB_SELECT_TRIGGERS;
    }
    match card.name {
        "Guttersnipe" => &GUTTERSNIPE_TRIGGERS,
        "Murmuring Mystic" => &MURMURING_MYSTIC_TRIGGERS,
        "Voldaren Epicure" => &VOLDAREN_EPICURE_TRIGGERS,
        "Generous Ent" => &GENEROUS_ENT_TRIGGERS,
        "Gingerbread Cabin" => &GINGERBREAD_CABIN_TRIGGERS,
        "Writhing Chrysalis" => &WRITHING_CHRYSALIS_TRIGGERS,
        "Blood Fountain" => &BLOOD_FOUNTAIN_TRIGGERS,
        "Sagu Wildling" => &SAGU_WILDLING_TRIGGERS,
        "Gatecreeper Vine" => &GATECREEPER_VINE_TRIGGERS,
        "Balustrade Spy" => &BALUSTRADE_SPY_TRIGGERS,
        "Lotleth Giant" => &LOTLETH_GIANT_TRIGGERS,
        "Mesmeric Fiend" => &MESMERIC_FIEND_TRIGGERS,
        "Healer of the Glade" | "Spinewoods Paladin" => &GAIN_THREE_LIFE_ETB_TRIGGERS,
        "Sneaky Snacker" => &SNEAKY_SNACKER_TRIGGERS,
        "Burning-Tree Emissary" => &BURNING_TREE_EMISSARY_TRIGGERS,
        "Clockwork Percussionist" => &CLOCKWORK_PERCUSSIONIST_TRIGGERS,
        "Ichor Wellspring" => &ICHOR_WELLSPRING_TRIGGERS,
        "Cryogen Relic" => &CRYOGEN_RELIC_TRIGGERS,
        "Nihil Spellbomb" => &NIHIL_SPELLBOMB_TRIGGERS,
        "Experimental Synthesizer" => &EXPERIMENTAL_SYNTHESIZER_TRIGGERS,
        "Goblin Bushwhacker" => &GOBLIN_BUSHWHACKER_TRIGGERS,
        "Faerie Miscreant" => &FAERIE_MISCREANT_TRIGGERS,
        "Faerie Seer" => &FAERIE_SEER_TRIGGERS,
        "Outlaw Medic" => &OUTLAW_MEDIC_TRIGGERS,
        "Refurbished Familiar" => &REFURBISHED_FAMILIAR_TRIGGERS,
        "Squadron Hawk" => &SQUADRON_HAWK_TRIGGERS,
        "Bind the Monster" => &BIND_THE_MONSTER_TRIGGERS,
        "Harrier Strix" => &HARRIER_STRIX_TRIGGERS,
        "Humbling Elder" => &HUMBLING_ELDER_TRIGGERS,
        "Moon-Circuit Hacker" => &MOON_CIRCUIT_HACKER_TRIGGERS,
        "Ninja of the Deep Hours" => &NINJA_OF_THE_DEEP_HOURS_TRIGGERS,
        "Saiba Cryptomancer" => &SAIBA_CRYPTOMANCER_TRIGGERS,
        "Spellstutter Sprite" => &SPELLSTUTTER_SPRITE_TRIGGERS,
        "Journey to Nowhere" => &JOURNEY_TO_NOWHERE_TRIGGERS,
        "Lembas" => &LEMBAS_TRIGGERS,
        "Weather the Storm" => &WEATHER_THE_STORM_TRIGGERS,
        "Masked Vandal" => &MASKED_VANDAL_TRIGGERS,
        "Troublemaker Ouphe" => &TROUBLEMAKER_OUPHE_TRIGGERS,
        "Vitu-Ghazi Inspector" => &VITU_GHAZI_INSPECTOR_TRIGGERS,
        "Avenging Hunter" => &AVENGING_HUNTER_TRIGGERS,
        _ => &[],
    }
}

/// Definition-owned target specification for the pool's triggered abilities.
pub fn trigger_target_spec(card_def: u16) -> TargetSpec {
    let Some(card) = crate::card_def::CARD_DEFS.get(card_def as usize) else {
        return TargetSpec::None;
    };
    match card.name {
        "Balustrade Spy" => TargetSpec::AnyPlayer,
        "Lotleth Giant" => TargetSpec::TargetOpponent,
        "Harrier Strix" => TargetSpec::AnyPermanent,
        "Humbling Elder" => TargetSpec::OpponentControlledCreature,
        "Saiba Cryptomancer" => TargetSpec::Creature,
        "Spellstutter Sprite" => TargetSpec::SpellManaValueAtMostControlledSubtypes {
            first: Subtype::Faerie,
            second: Some(Subtype::FaerieAllCaps),
        },
        "Masked Vandal" | "Troublemaker Ouphe" => {
            TargetSpec::OpponentArtifactOrEnchantmentPermanent
        }
        "Vitu-Ghazi Inspector" => TargetSpec::Creature,
        _ => TargetSpec::None,
    }
}

pub(crate) fn required_optional_additional_cost_for_trigger(
    card_def: u16,
    effect: &EffectOp,
) -> Option<OptionalAdditionalCostDef> {
    let card = crate::card_def::CARD_DEFS.get(card_def as usize)?;
    match card.name {
        "Troublemaker Ouphe" if *effect == troublemaker_ouphe_etb_effect() => {
            Some(OptionalAdditionalCostDef::Bargain)
        }
        "Vitu-Ghazi Inspector" if *effect == vitu_ghazi_inspector_etb_effect() => {
            Some(OptionalAdditionalCostDef::CollectEvidence {
                minimum_mana_value: 6,
            })
        }
        _ => None,
    }
}

/// Authenticates the finite set of effects a definition-owned trigger can
/// place on the stack, including Moon-Circuit Hacker's event-frozen branch.
pub fn trigger_effect_matches(card_def: u16, effect: &EffectOp) -> bool {
    let Some(card) = crate::card_def::CARD_DEFS.get(card_def as usize) else {
        return false;
    };
    if card.name == "Moon-Circuit Hacker"
        && [false, true].into_iter().any(|entered| {
            moon_circuit_hacker_combat_effect_for_entered_this_turn(entered) == *effect
        })
    {
        return true;
    }
    if card.name == "Writhing Chrysalis"
        && matches!(
            effect,
            EffectOp::PutPlusOnePlusOneCounterOnBoundObject { .. }
        )
    {
        return true;
    }
    if card.name == "Weather the Storm" && matches!(effect, EffectOp::CreateStormCopies { .. }) {
        return true;
    }
    if card.name == "Avenging Hunter" && matches!(effect, EffectOp::ResolveInitiativeTrigger { .. })
    {
        return true;
    }
    triggers_for(card_def)
        .iter()
        .any(|definition| (definition.effect)() == *effect)
        || card.saga.as_ref().is_some_and(|saga| {
            saga.chapter_effects
                .iter()
                .any(|chapter_effect| chapter_effect() == *effect)
        })
}

pub fn target_spec_for_trigger(card_def: u16, effect: &EffectOp) -> Option<TargetSpec> {
    if !trigger_effect_matches(card_def, effect) {
        return None;
    }
    let card = crate::card_def::CARD_DEFS.get(card_def as usize)?;
    Some(
        if card.name == "Mesmeric Fiend" && *effect == mesmeric_fiend_exile_effect() {
            TargetSpec::TargetOpponent
        } else if card.name == "Journey to Nowhere" && *effect == journey_to_nowhere_etb_effect() {
            TargetSpec::CreatureOtherThanSource
        } else if card.name == "Avenging Hunter" {
            match effect {
                EffectOp::ResolveInitiativeTrigger { binding } => match binding.kind {
                    InitiativeTriggerKindV1::UndercityRoom(UndercityRoomV1::Forge)
                    | InitiativeTriggerKindV1::UndercityRoom(UndercityRoomV1::Arena) => {
                        TargetSpec::Creature
                    }
                    InitiativeTriggerKindV1::UndercityRoom(UndercityRoomV1::Trap) => {
                        TargetSpec::AnyPlayer
                    }
                    _ => TargetSpec::None,
                },
                _ => trigger_target_spec(card_def),
            }
        } else {
            trigger_target_spec(card_def)
        },
    )
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PendingTrigger {
    pub controller: PlayerId,
    pub source: ObjectId,
    pub effect: EffectOp,
    /// True iff this is a Madness triggered-ability offer (`engine::
    /// apply_discard`'s Madness branch), not one of this module's
    /// card-def-matched triggers (`triggers_for`). Threaded through the
    /// same APNAP grouping/`Decision::OrderTriggers` machinery as any other
    /// trigger (603.3b makes no distinction), but `effect` is a meaningless
    /// placeholder (`EffectOp::Sequence(vec![])`) for one of these --
    /// `engine::push_trigger_onto_stack` reads this flag to leave the
    /// resulting `StackItem`'s `inline_effect` as `None` and set its own
    /// `madness_offer` instead (see that field's doc). Always `false` for a
    /// real `triggers_for`-matched trigger.
    pub is_madness_offer: bool,
    /// True iff this trigger's own `source` is the object that just
    /// resolved from a kicked cast (`engine::EngineState::
    /// pending_kicked_source`, consumed by `collect_and_process`) -- Goblin
    /// Bushwhacker's ETB trigger reads this via `engine::
    /// push_trigger_onto_stack` copying it onto the trigger's own
    /// `state::StackItem::kicked`, then `effect::ExecCtx::kicked` at
    /// resolution. `false` for every other trigger.
    pub kicked: bool,
    /// Definition-owned target shape selected while this trigger is put on
    /// the stack. Existing untargeted triggers retain `None`.
    #[serde(default)]
    pub target_spec: TargetSpec,
    /// Target prefix chosen while placing this trigger on the stack.
    #[serde(default)]
    pub targets: Vec<Target>,
    /// Exact target-incarnation contracts parallel to `targets`.
    #[serde(default)]
    pub target_contracts: Vec<StackTargetContractV4>,
    /// True after the trigger's simultaneous-controller group has either
    /// been ordered by its controller or proven singleton.
    #[serde(default)]
    pub placement_ordered: bool,
    /// Exact historical source incarnation captured when the trigger was
    /// created. This lets independent and linked abilities remain valid
    /// across later zone changes of the same physical card.
    #[serde(default)]
    pub source_contract: Option<AbilitySourceContractV4>,
    /// Exact Equipment incarnation that granted this trigger to `source`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub granted_by: Option<AbilitySourceContractV4>,
    /// Exact cast-scoped optional additional cost inherited by this ETB.
    /// This is absent for every trigger without an intervening-if cost gate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub optional_additional_cost_paid: Option<OptionalAdditionalCostDef>,
    /// Historical physical-card payment provenance copied from the producing
    /// spell's stack incarnation.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub paid_cost_refs: Vec<PaidCostRefV4>,
}

fn creature_dies_to_state_based_actions(
    toughness: i32,
    marked_damage: i32,
    deathtouch_damage: bool,
    indestructible: bool,
) -> bool {
    toughness <= 0
        || ((marked_damage >= toughness || (marked_damage > 0 && deathtouch_damage))
            && !indestructible)
}

/// Runs state-based actions to a fixed point: repeat the full SBA sweep
/// until one pass makes no change (704.3).
pub fn sba_fixed_point(state: &mut GameState) {
    sba_fixed_point_with_protected_triggers(state, &[]);
}

fn saga_final_chapter_is_pending(
    state: &GameState,
    source: ObjectId,
    source_zone_change_count: u32,
    final_effect: &EffectOp,
    protected_triggers: &[PendingTrigger],
) -> bool {
    let matches_pending = |pending: &PendingTrigger| {
        pending.source == source
            && pending.effect == *final_effect
            && pending
                .source_contract
                .is_some_and(|contract| contract.zone_change_count == source_zone_change_count)
    };
    protected_triggers.iter().any(matches_pending)
        || state.engine.pending_triggers.iter().any(matches_pending)
        || state.stack.iter().any(|item| {
            item.source == source
                && item.kind == crate::state::StackItemKind::TriggeredAbility
                && item.inline_effect.as_ref() == Some(final_effect)
                && item
                    .v4
                    .ability_source_contract
                    .is_some_and(|contract| contract.zone_change_count == source_zone_change_count)
        })
        || state.engine.event_log.iter().any(|event| {
            matches!(
                event,
                CommittedEvent::SagaChapter {
                    source: event_source,
                    source_zone_change_count: event_generation,
                    chapter,
                    ..
                } if *event_source == source
                    && *event_generation == source_zone_change_count
                    && usize::from(*chapter)
                        == crate::card_def::CARD_DEFS[state.objects.get(source).card_def as usize]
                            .saga
                            .as_ref()
                            .map_or(0, |saga| saga.chapter_effects.len())
            )
        })
}

fn sba_fixed_point_with_protected_triggers(
    state: &mut GameState,
    protected_triggers: &[PendingTrigger],
) {
    loop {
        let mut changed = false;

        // 704.5g: a creature with toughness 0 or less is put into its
        // owner's graveyard. 704.5h: a creature with lethal damage marked
        // is destroyed.
        let mut dying = Vec::new();
        for (id, obj) in state.objects.iter() {
            if obj.zone != Zone::Battlefield {
                continue;
            }
            if !crate::engine::object_has_type(state, id, crate::card_def::CardType::Creature) {
                continue;
            }
            let toughness = crate::engine::effective_toughness(state, id);
            let indestructible = crate::engine::has_effective_keyword(
                state,
                id,
                crate::card_def::Keywords::INDESTRUCTIBLE,
            );
            // 704.5g is a put-into-graveyard action and ignores
            // indestructible. 704.5h destroys for lethal damage, so
            // indestructible prevents only that branch.
            if creature_dies_to_state_based_actions(
                toughness,
                obj.damage as i32,
                obj.v4.deathtouch_damage,
                indestructible,
            ) {
                dying.push(id);
            }
        }
        for id in dying {
            crate::event::commit(
                state,
                crate::event::ProposedEvent::zone_change(id, Zone::Graveyard),
            );
            changed = true;
        }

        // 704.5m: an Aura attached to an illegal object or player, or not
        // attached at all, is put into its owner's graveyard. The current
        // attachment grammar supports creature Auras only and requires both
        // exact-incarnation directions of the relation to agree.
        let invalid_auras = state
            .objects
            .iter()
            .filter_map(|(id, aura)| {
                if aura.zone != Zone::Battlefield {
                    return None;
                }
                let definition = &crate::card_def::CARD_DEFS[aura.card_def as usize];
                let Some(crate::card_def::AttachmentDef::AuraCreature { .. }) =
                    definition.attachment
                else {
                    return None;
                };
                let valid = aura.v4.attached_to.is_some_and(|link| {
                    state.objects.try_get(link.object).is_some_and(|host| {
                        host.zone == Zone::Battlefield
                            && host.zone_change_count == link.zone_change_count
                            && crate::card_def::CARD_DEFS[host.card_def as usize]
                                .has_type(crate::card_def::CardType::Creature)
                            && host.attachments.contains(&id)
                    })
                });
                (!valid).then_some(id)
            })
            .collect::<Vec<_>>();
        for id in invalid_auras {
            crate::event::commit(
                state,
                crate::event::ProposedEvent::zone_change(id, Zone::Graveyard),
            );
            changed = true;
        }

        // 704.5s: after a Saga's final chapter ability leaves the stack, its
        // controller sacrifices it. The chapter trigger is protected across
        // the pre-placement SBA checkpoint by its exact source incarnation.
        let completed_sagas = state
            .objects
            .iter()
            .filter_map(|(id, object)| {
                if object.zone != Zone::Battlefield || object.v4.face_index != 0 {
                    return None;
                }
                let saga = crate::card_def::CARD_DEFS[object.card_def as usize]
                    .saga
                    .as_ref()?;
                if object.counters.lore < saga.chapter_effects.len() as i16 {
                    return None;
                }
                let final_effect = (saga.chapter_effects.last().copied()?)();
                (!saga_final_chapter_is_pending(
                    state,
                    id,
                    object.zone_change_count,
                    &final_effect,
                    protected_triggers,
                ))
                .then_some(id)
            })
            .collect::<Vec<_>>();
        for id in completed_sagas {
            crate::event::commit(
                state,
                crate::event::ProposedEvent::zone_change(id, Zone::Graveyard),
            );
            changed = true;
        }

        // 704.5a: a player with 0 or less life loses. 704.5c: a player who
        // attempted to draw from an empty library loses.
        for p in [PlayerId::P0, PlayerId::P1] {
            let ps = &mut state.players[p.index()];
            if !ps.has_lost && (ps.life <= 0 || ps.drew_from_empty) {
                ps.has_lost = true;
                changed = true;
            }
        }

        // 111.8/704.5d: a token in any zone other than the battlefield
        // ceases to exist -- most commonly a sacrificed/died Blood Token
        // ending up back in the graveyard's card-count for the rest of the
        // game, which it never does in a real game (root-caused against
        // the real v3 corpus: `kernel_gy` carrying a stray "Blood Token"
        // entry the trace's own graveyard snapshot never has, many turns
        // after the token was created and then activated/sacrificed).
        let leaving: Vec<ObjectId> = state
            .objects
            .iter()
            .filter(|(_, obj)| {
                obj.zone != Zone::Battlefield
                    && crate::card_def::CARD_DEFS[obj.card_def as usize].is_token
            })
            .map(|(id, _)| id)
            .collect();
        for id in leaving {
            if crate::event::cease_to_exist(state, id) {
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }
}

/// Matches the events already in `state.engine.event_log`, runs SBAs to a
/// fixed point, then matches any events created by those SBAs. Returns the
/// combined newly-triggered abilities in APNAP order (active player's
/// triggers first).
pub fn collect_and_process(state: &mut GameState) -> Vec<PendingTrigger> {
    let events: Vec<CommittedEvent> = state.engine.event_log.drain(..).collect();
    // Single-shot: `engine::resolve_top_of_stack` set this immediately
    // before the resolution whose events we're about to match, explicitly
    // (`Some`/`None`) every single time -- taking it here means it can never
    // carry over into a later, unrelated `collect_and_process` call (see
    // `EngineState::pending_kicked_source`'s doc).
    let kicked_source = state.engine.pending_kicked_source.take();

    // Trigger conditions are evaluated at the moment their event happens,
    // before the following SBA check (603.2/704.3). In particular, an ETB
    // trigger must not disappear merely because its source dies during that
    // check, so match the pre-SBA batch while its sources still occupy the
    // zones from which their abilities function.
    let mut new_triggers = triggers_from_events(state, &events, kicked_source);

    // Conversely, SBAs can create new trigger events themselves. Lethal
    // combat damage, for example, moves Clockwork Percussionist to the
    // graveyard here; its dies trigger belongs to this same trigger-placement
    // checkpoint, not some later action. Match that second batch only after
    // the fixed point, then order both batches together under 603.3b.
    sba_fixed_point_with_protected_triggers(state, &new_triggers);
    let sba_events: Vec<CommittedEvent> = state.engine.event_log.drain(..).collect();
    new_triggers.extend(triggers_from_events(state, &sba_events, None));

    // 603.3d: a triggered ability requiring targets is removed from the
    // stack-placement queue when no complete legal target assignment exists.
    // This check belongs after the SBA fixed point, at the actual placement
    // checkpoint, rather than at the earlier event-matching snapshot.
    new_triggers.retain(|pending| {
        pending.target_spec == TargetSpec::None
            || crate::engine::target_prefix_can_complete_for_controller(
                pending.target_spec,
                &pending.targets,
                pending.controller,
                state,
            )
    });

    order_apnap(new_triggers, state.active_player)
}

fn triggers_from_events(
    state: &GameState,
    events: &[CommittedEvent],
    kicked_source: Option<ObjectId>,
) -> Vec<PendingTrigger> {
    let draws_this_turn_at = draws_this_turn_snapshot(events, state);
    let mut new_triggers = Vec::new();
    for (id, obj) in state.objects.iter() {
        let card = &crate::card_def::CARD_DEFS[obj.card_def as usize];
        if !card.is_executable() {
            continue;
        }
        if let Some(saga) = card.saga.as_ref() {
            for ev in events {
                let CommittedEvent::SagaChapter {
                    source,
                    source_zone_change_count,
                    controller,
                    chapter,
                } = ev
                else {
                    continue;
                };
                if *source != id
                    || *chapter == 0
                    || usize::from(*chapter) > saga.chapter_effects.len()
                    || obj.zone_change_count < *source_zone_change_count
                    || (obj.zone_change_count == *source_zone_change_count
                        && (obj.zone != Zone::Battlefield || obj.v4.face_index != 0))
                {
                    continue;
                }
                let effect = (saga.chapter_effects[usize::from(*chapter - 1)])();
                new_triggers.push(PendingTrigger {
                    controller: *controller,
                    source: id,
                    target_spec: target_spec_for_trigger(obj.card_def, &effect)
                        .expect("Saga chapter effect is definition-owned"),
                    effect,
                    is_madness_offer: false,
                    kicked: false,
                    targets: Vec::new(),
                    target_contracts: Vec::new(),
                    placement_ordered: false,
                    source_contract: Some(AbilitySourceContractV4 {
                        source: id,
                        card_def: obj.card_def,
                        owner: obj.owner,
                        controller: *controller,
                        zone: Zone::Battlefield,
                        zone_change_count: *source_zone_change_count,
                        attached_to: None,
                    }),
                    granted_by: None,
                    optional_additional_cost_paid: None,
                    paid_cost_refs: Vec::new(),
                });
            }
        }
        for def in triggers_for(obj.card_def) {
            let uses_leave_lki = matches!(
                def.condition,
                TriggerCondition::LeftBattlefieldToGraveyard | TriggerCondition::LeftBattlefield
            );
            if !uses_leave_lki && obj.zone != def.home_zone {
                continue;
            }
            for (i, ev) in events.iter().enumerate() {
                let event_controller = match ev {
                    CommittedEvent::ZoneChange {
                        object,
                        from: Zone::Battlefield,
                        controller_before,
                        ..
                    } if *object == id && uses_leave_lki => *controller_before,
                    _ => obj.controller,
                };
                if trigger_matches(
                    def.condition,
                    ev,
                    id,
                    event_controller,
                    state,
                    draws_this_turn_at[i],
                ) {
                    let kicked = Some(id) == kicked_source;
                    if def.intervening_if_kicked && !kicked {
                        continue;
                    }
                    if def.intervening_if_controls_another_source_card {
                        let source_def = obj.card_def;
                        let controls_another = state.players[event_controller.index()]
                            .battlefield
                            .iter()
                            .copied()
                            .any(|other| {
                                other != id && state.objects.get(other).card_def == source_def
                            });
                        if !controls_another {
                            continue;
                        }
                    }
                    let effect = if card.name == "Moon-Circuit Hacker"
                        && matches!(def.condition, TriggerCondition::DealsCombatDamageToPlayer)
                    {
                        moon_circuit_hacker_combat_effect_for_entered_this_turn(
                            obj.v4.entered_battlefield_turn == Some(state.turn),
                        )
                    } else {
                        materialize_trigger_effect(def, id, state)
                    };
                    let required_optional_cost =
                        required_optional_additional_cost_for_trigger(obj.card_def, &effect);
                    let (paid_optional_cost, paid_cost_refs) =
                        if let Some(required) = required_optional_cost {
                            let matching = events
                                .iter()
                                .filter_map(|event| match event {
                                    CommittedEvent::OptionalAdditionalCostPaid {
                                        source,
                                        kind,
                                        paid_cost_refs,
                                    } if *source == id => Some((*kind, paid_cost_refs.clone())),
                                    _ => None,
                                })
                                .collect::<Vec<_>>();
                            let [(kind, refs)] = matching.as_slice() else {
                                continue;
                            };
                            if *kind != required {
                                continue;
                            }
                            (Some(required), refs.clone())
                        } else {
                            (None, Vec::new())
                        };
                    let target_spec =
                        target_spec_for_trigger(obj.card_def, &effect).unwrap_or(TargetSpec::None);
                    let source_contract = match ev {
                        CommittedEvent::ZoneChange {
                            object,
                            from,
                            controller_before,
                            ..
                        } if *object == id && uses_leave_lki => {
                            let Some(zone_change_count) = obj.zone_change_count.checked_sub(1)
                            else {
                                continue;
                            };
                            Some(AbilitySourceContractV4 {
                                source: id,
                                card_def: obj.card_def,
                                owner: obj.owner,
                                controller: *controller_before,
                                zone: *from,
                                zone_change_count,
                                attached_to: None,
                            })
                        }
                        _ if obj.zone == Zone::Stack => None,
                        _ => {
                            let mut contract = AbilitySourceContractV4::capture(state, id);
                            contract.controller = event_controller;
                            Some(contract)
                        }
                    };
                    new_triggers.push(PendingTrigger {
                        controller: event_controller,
                        source: id,
                        granted_by: None,
                        effect,
                        is_madness_offer: false,
                        kicked,
                        target_spec,
                        targets: Vec::new(),
                        target_contracts: Vec::new(),
                        placement_ordered: false,
                        source_contract,
                        optional_additional_cost_paid: paid_optional_cost,
                        paid_cost_refs,
                    });
                }
            }
        }
        if obj.zone == Zone::Battlefield {
            if let Some(crate::card_def::WardCostDef::Generic(generic)) = card.ward_cost {
                for event in events {
                    let CommittedEvent::Targeted {
                        target,
                        target_zone_change_count,
                        targeting_stack_item,
                        targeting_controller,
                    } = event
                    else {
                        continue;
                    };
                    if *target != id
                        || *target_zone_change_count != obj.zone_change_count
                        || *targeting_controller == obj.controller
                    {
                        continue;
                    }
                    let ward_target = crate::state::StackTargetContractV4::capture(
                        state,
                        crate::state::Target::Object(id),
                    );
                    new_triggers.push(PendingTrigger {
                        controller: obj.controller,
                        source: id,
                        granted_by: None,
                        effect: EffectOp::CounterUnlessPaysGeneric {
                            ward_target,
                            targeting_stack_item: *targeting_stack_item,
                            generic,
                        },
                        is_madness_offer: false,
                        kicked: false,
                        target_spec: TargetSpec::None,
                        targets: Vec::new(),
                        target_contracts: Vec::new(),
                        placement_ordered: false,
                        source_contract: Some(AbilitySourceContractV4::capture(state, id)),
                        optional_additional_cost_paid: None,
                        paid_cost_refs: Vec::new(),
                    });
                }
            }
        }
    }

    // Attachment-granted abilities belong to the equipped creature. Each
    // Equipment grants a separate trigger, and its exact incarnation is
    // carried independently as provenance.
    for event in events {
        let CommittedEvent::SpellCast {
            spell,
            controller: caster,
        } = event
        else {
            continue;
        };
        if selected_spell_types(state, *spell).contains(&CardType::Creature) {
            continue;
        }
        for (host, host_live) in state.objects.iter() {
            if host_live.controller != *caster
                || host_live.zone != Zone::Battlefield
                || !crate::card_def::CARD_DEFS[host_live.card_def as usize]
                    .has_type(CardType::Creature)
            {
                continue;
            }
            for &equipment in &host_live.attachments {
                let equipment_live = state.objects.get(equipment);
                let exact_relation = equipment_live.v4.attached_to.is_some_and(|link| {
                    link.object == host
                        && link.zone_change_count == host_live.zone_change_count
                        && equipment_live.zone == Zone::Battlefield
                });
                let Some(profile) =
                    crate::card_def::CARD_DEFS[equipment_live.card_def as usize].equipment
                else {
                    continue;
                };
                if !exact_relation || profile.noncreature_spell_damage_to_each_opponent == 0 {
                    continue;
                }
                new_triggers.push(PendingTrigger {
                    controller: *caster,
                    source: host,
                    source_contract: Some(AbilitySourceContractV4::capture(state, host)),
                    granted_by: Some(AbilitySourceContractV4::capture(state, equipment)),
                    effect: EffectOp::DealDamage {
                        target: TargetRef::Opponent,
                        amount: i32::from(profile.noncreature_spell_damage_to_each_opponent),
                    },
                    is_madness_offer: false,
                    kicked: false,
                    target_spec: TargetSpec::None,
                    targets: Vec::new(),
                    target_contracts: Vec::new(),
                    placement_ordered: false,
                    optional_additional_cost_paid: None,
                    paid_cost_refs: Vec::new(),
                });
            }
        }
    }

    for event in events {
        let CommittedEvent::InitiativeTrigger { binding } = event else {
            continue;
        };
        let binding = *binding;
        let effect = EffectOp::ResolveInitiativeTrigger { binding };
        let target_spec =
            target_spec_for_trigger(binding.source.card_def, &effect).unwrap_or(TargetSpec::None);
        new_triggers.push(PendingTrigger {
            controller: binding.player,
            source: binding.source.source,
            effect,
            is_madness_offer: false,
            kicked: false,
            target_spec,
            targets: Vec::new(),
            target_contracts: Vec::new(),
            placement_ordered: false,
            source_contract: Some(binding.source),
            granted_by: None,
            optional_additional_cost_paid: None,
            paid_cost_refs: Vec::new(),
        });
    }

    new_triggers
}

fn selected_spell_types(state: &GameState, spell: ObjectId) -> &'static [CardType] {
    let object = state.objects.get(spell);
    let definition = &crate::card_def::CARD_DEFS[object.card_def as usize];
    if object
        .v4
        .spell_cast_origin
        .and_then(|origin| origin.finalized_method)
        == Some(crate::state::CastMethodV4::Omen)
    {
        definition
            .omen
            .as_ref()
            .map(|omen| omen.types)
            .unwrap_or(&[])
    } else {
        definition.types
    }
}

/// For each event in `events` (already committed, in commit order), the
/// value `draws_this_turn` genuinely held *at the moment that specific
/// event was committed* -- not `state`'s current (post-batch) value.
///
/// Root-caused against `game_20260713_002147_0002.txt`: `EffectOp::
/// DrawCards`'s loop (Faithless Looting's "draw two cards") commits both
/// draws into `event_log` before `collect_and_process` ever runs (both
/// resolve as one atomic ability, 608.2h -- correctly so; a trigger check
/// belongs *after* the whole ability resolves, not spliced mid-resolution,
/// 603.3/704.3), so by the time this function inspects `state`, `draws_
/// this_turn` already reflects *both* draws. Checking every event in the
/// batch against that single final value made `TriggerCondition::
/// DrawNth(3)` (Sneaky Snacker) miss entirely whenever a 2-draw batch
/// jumped straight over 3 (2 -> 4): neither event's *own* moment (3, then
/// 4) was ever actually tested, only the batch's final value (4) tested
/// twice. 608.2h itself settles that each drawn card is still a distinct,
/// sequential event ("the player draws that many cards, in that order"),
/// so a "your Nth draw this turn" condition must see each one's own true
/// historical count -- reconstructed here (not threaded through
/// `CommittedEvent::Draw` itself, which stays a plain, serializable
/// snapshot-free record) by walking `events` forward per player: the
/// count immediately *before* this batch is `state`'s current value minus
/// however many of this player's own `Draw` events are in this same
/// batch, then each of that player's `Draw` events in commit order adds
/// exactly 1.
fn draws_this_turn_snapshot(events: &[CommittedEvent], state: &GameState) -> Vec<u32> {
    let batch_draws = |p: PlayerId| {
        events
            .iter()
            .filter(
                |e| matches!(e, CommittedEvent::Draw { player, object: Some(_) } if *player == p),
            )
            .count() as u32
    };
    let mut running = [
        state.players[0].draws_this_turn - batch_draws(PlayerId::P0),
        state.players[1].draws_this_turn - batch_draws(PlayerId::P1),
    ];
    events
        .iter()
        .map(|ev| match ev {
            CommittedEvent::Draw {
                player,
                object: Some(_),
            } => {
                running[player.index()] += 1;
                running[player.index()]
            }
            _ => 0, // unused by any non-DrawNth trigger_matches arm
        })
        .collect()
}

/// `draws_this_turn_at_event`: only meaningful for a `CommittedEvent::Draw`
/// (see `draws_this_turn_snapshot`'s doc for why this can't just read
/// `state` live) -- unused, and irrelevant, for every other event/condition
/// pairing.
fn trigger_matches(
    cond: TriggerCondition,
    ev: &CommittedEvent,
    source: ObjectId,
    controller: PlayerId,
    state: &GameState,
    draws_this_turn_at_event: u32,
) -> bool {
    match (cond, ev) {
        (
            TriggerCondition::Etb,
            CommittedEvent::ZoneChange {
                object,
                to: Zone::Battlefield,
                ..
            },
        ) => *object == source,
        (
            TriggerCondition::EtbControlsOtherSubtypeCount {
                subtype,
                minimum_count,
            },
            CommittedEvent::ZoneChange {
                object,
                to: Zone::Battlefield,
                ..
            },
        ) => {
            if *object != source {
                return false;
            }
            let subtype_id = subtype.stable_id();
            let count = state.players[controller.index()]
                .battlefield
                .iter()
                .copied()
                .filter(|candidate| *candidate != source)
                .filter(|candidate| {
                    crate::engine::effective_subtype_ids(state, *candidate)
                        .binary_search(&subtype_id)
                        .is_ok()
                })
                .count();
            count >= usize::from(minimum_count)
        }
        (TriggerCondition::DealsDamage, CommittedEvent::Damage { source: s, .. }) => *s == source,
        (
            TriggerCondition::DealsCombatDamageToPlayer,
            CommittedEvent::CombatDamageToPlayer {
                source: event_source,
                source_zone_change_count,
                ..
            },
        ) => {
            *event_source == source
                && state.objects.get(source).zone_change_count == *source_zone_change_count
        }
        (
            TriggerCondition::CastInstantOrSorcery,
            CommittedEvent::SpellCast {
                spell,
                controller: caster,
            },
        ) => {
            *caster == controller && {
                let types = selected_spell_types(state, *spell);
                types.contains(&crate::card_def::CardType::Instant)
                    || types.contains(&crate::card_def::CardType::Sorcery)
            }
        }
        (
            TriggerCondition::CastSelf,
            CommittedEvent::SpellCast {
                spell,
                controller: caster,
            },
        ) => *spell == source && *caster == controller,
        (
            TriggerCondition::DrawNth(n),
            CommittedEvent::Draw {
                player,
                object: Some(_),
            },
        ) => *player == controller && draws_this_turn_at_event == n,
        (
            TriggerCondition::LeftBattlefieldToGraveyard,
            CommittedEvent::ZoneChange {
                object,
                from: Zone::Battlefield,
                to: Zone::Graveyard,
                ..
            },
        ) => *object == source,
        (
            TriggerCondition::LeftBattlefield,
            CommittedEvent::ZoneChange {
                object,
                from: Zone::Battlefield,
                ..
            },
        ) => *object == source,
        (
            TriggerCondition::SacrificeAnotherWithSubtype(subtype),
            CommittedEvent::Sacrificed {
                object,
                controller_before,
                effective_subtype_ids_before,
            },
        ) => {
            *object != source
                && *controller_before == controller
                && effective_subtype_ids_before
                    .binary_search(&subtype.stable_id())
                    .is_ok()
        }
        _ => false,
    }
}

/// 603.3b: each player puts triggered abilities they control on the stack
/// in an order of their choice, in APNAP order (active player's group
/// first). This only groups; a real per-player choice within a group of 2+
/// is `engine::Decision::OrderTriggers`.
pub fn order_apnap(triggers: Vec<PendingTrigger>, active_player: PlayerId) -> Vec<PendingTrigger> {
    let (mut active, mut other): (Vec<_>, Vec<_>) = triggers
        .into_iter()
        .partition(|t| t.controller == active_player);
    active.append(&mut other);
    active
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::PlayerId;
    use crate::state::GameState;

    // Lethal-damage creature death is exercised end-to-end in
    // `engine::tests::lethal_damage_kills_creature_via_sba`, using a real
    // `CARD_DEFS` creature (card-def ids here are synthetic and don't map
    // to real cards).

    #[test]
    fn sba_declares_loss_at_zero_life() {
        let mut state = GameState::new_from_libraries(&[1], &[2], |c| format!("card-{c}"), 1);
        state.players[0].life = 0;
        sba_fixed_point(&mut state);
        assert!(state.players[0].has_lost);
    }

    #[test]
    fn sba_declares_loss_on_drew_from_empty() {
        let mut state = GameState::new_from_libraries(&[1], &[2], |c| format!("card-{c}"), 1);
        state.players[1].drew_from_empty = true;
        sba_fixed_point(&mut state);
        assert!(state.players[1].has_lost);
    }

    #[test]
    fn indestructible_prevents_lethal_destruction_but_not_zero_toughness() {
        assert!(!creature_dies_to_state_based_actions(3, 3, false, true));
        assert!(creature_dies_to_state_based_actions(3, 3, false, false));
        assert!(creature_dies_to_state_based_actions(0, 0, false, true));
        assert!(creature_dies_to_state_based_actions(3, 1, true, false));
        assert!(!creature_dies_to_state_based_actions(3, 1, true, true));
    }

    #[test]
    fn apnap_orders_active_player_triggers_first() {
        let a = PendingTrigger {
            controller: PlayerId::P1,
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
        };
        let b = PendingTrigger {
            controller: PlayerId::P0,
            source: ObjectId(2),
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
        };
        let ordered = order_apnap(vec![a.clone(), b.clone()], PlayerId::P0);
        assert_eq!(ordered, vec![b, a]);
    }
}
