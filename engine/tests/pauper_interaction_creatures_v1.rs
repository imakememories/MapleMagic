//! Focused rules coverage for Cast Down, Breath Weapon, Outlaw Medic, and
//! Refurbished Familiar.
//!
//! The checked-in Mage source baseline is `0723fc0c2be922af47b0ef0539f28114cc23b998`.

use mtg_kernel::card_def::{
    card_id_by_name, CardCapability, CardType, DynamicCountDef, GenericCostReductionDef, Keywords,
    Subtype, TargetSpec, CARD_DEFS,
};
use mtg_kernel::effect::{self, EffectCond, EffectOp, ExecCtx, PlayerRef};
use mtg_kernel::engine::{self, Action, Decision};
use mtg_kernel::event::{
    self, ActiveReplacement, CommittedEvent, ProposedEvent, ReplacementEffectKind,
};
use mtg_kernel::ids::{ObjectId, PlayerId};
use mtg_kernel::mana::{ManaColor, Pip};
use mtg_kernel::state::{
    Counters, GameObject, GameState, StackTargetContractV4, Step, Target, Zone,
};
use mtg_kernel::trigger;

fn card_id(name: &str) -> u16 {
    card_id_by_name(name).unwrap_or_else(|| panic!("{name} in CARD_DEFS"))
}

fn card_name(card_def: u16) -> String {
    CARD_DEFS[card_def as usize].name.to_string()
}

fn ready_state() -> GameState {
    let library = vec![card_id("Mountain"); 8];
    let mut state =
        GameState::new_from_libraries(&library, &library, card_name, 0x494E_5445_5241_4354);
    state.step = Step::Main1;
    state.active_player = PlayerId::P0;
    state.priority_player = PlayerId::P0;
    state
}

fn put_object(state: &mut GameState, player: PlayerId, name: &str, zone: Zone) -> ObjectId {
    let card_def = card_id(name);
    let id = state.objects.push(GameObject {
        card_def,
        name: name.to_string(),
        owner: player,
        controller: player,
        zone,
        tapped: false,
        summoning_sick: false,
        damage: 0,
        counters: Counters::default(),
        attachments: Vec::new(),
        v4: mtg_kernel::state::ObjectStateV4::from_card_def(card_def),
        spell_copy_origin: None,
        plotted_turn: None,
        zone_change_count: 0,
    });
    match zone {
        Zone::Hand => state.players[player.index()].hand.push(id),
        Zone::Battlefield => state.players[player.index()].battlefield.push(id),
        Zone::Library => state.players[player.index()].library.push(id),
        Zone::Graveyard => state.players[player.index()].graveyard.push(id),
        Zone::Exile => state.exile.push(id),
        Zone::Command => state.command.push(id),
        Zone::Stack => panic!("test helper does not construct stack items"),
    }
    id
}

fn enter(state: &mut GameState, object: ObjectId) -> Vec<trigger::PendingTrigger> {
    event::propose_and_commit(state, ProposedEvent::zone_change(object, Zone::Battlefield));
    trigger::collect_and_process(state)
}

fn execute_trigger(state: &mut GameState, pending: &trigger::PendingTrigger) {
    effect::execute(
        &pending.effect,
        &ExecCtx {
            source: pending.source,
            controller: pending.controller,
            stack_item_id: None,
            targets: Vec::new(),
            target_contracts: Vec::new(),
            discarded: Vec::new(),
            paid_cost_refs: Vec::new(),
            hidden_ability_source: None,
            ability_source_contract: pending.source_contract,
            kicked: false,
            optional_additional_cost_paid: None,
            x_value: 0,
        },
        state,
    );
}

#[test]
fn definitions_match_the_four_printed_cards() {
    for name in [
        "Cast Down",
        "Breath Weapon",
        "Outlaw Medic",
        "Refurbished Familiar",
    ] {
        assert_eq!(
            CARD_DEFS[card_id(name) as usize].capability,
            CardCapability::Full,
            "{name} must be admitted as fully supported"
        );
    }

    let cast_down = &CARD_DEFS[card_id("Cast Down") as usize];
    assert_eq!(cast_down.target_spec, TargetSpec::NonlegendaryCreature);
    assert_eq!(cast_down.cost.generic, 1);
    assert_eq!(cast_down.cost.pips, &[Pip::Colored(ManaColor::B)]);
    assert!(matches!(
        (cast_down.spell_effect)(),
        Some(EffectOp::Conditional {
            cond: EffectCond::TargetInZone(0, Zone::Battlefield),
            ..
        })
    ));

    let breath = &CARD_DEFS[card_id("Breath Weapon") as usize];
    assert_eq!(breath.target_spec, TargetSpec::None);
    assert_eq!(breath.cost.generic, 2);
    assert_eq!(breath.cost.pips, &[Pip::Colored(ManaColor::R)]);
    assert!(matches!(
        (breath.spell_effect)(),
        Some(EffectOp::DamageEachCreatureWithoutSubtype {
            amount: 2,
            excluded_subtype: Subtype::Dragon,
        })
    ));

    let medic = &CARD_DEFS[card_id("Outlaw Medic") as usize];
    assert_eq!((medic.power, medic.toughness), (Some(1), Some(3)));
    assert!(medic.keywords.has(Keywords::LIFELINK));

    let familiar = &CARD_DEFS[card_id("Refurbished Familiar") as usize];
    assert_eq!(familiar.types, &[CardType::Artifact, CardType::Creature]);
    assert_eq!((familiar.power, familiar.toughness), (Some(2), Some(1)));
    assert!(familiar.keywords.has(Keywords::FLYING));
    assert_eq!(
        familiar.generic_cost_reduction,
        Some(GenericCostReductionDef {
            generic_per_count: 1,
            count: DynamicCountDef::ControllerBattlefieldAnyType(&[CardType::Artifact]),
        })
    );
}

#[test]
fn cast_down_uses_the_nonlegendary_filter_and_shared_destroy_path() {
    let mut state = ready_state();
    let cast_down = put_object(&mut state, PlayerId::P0, "Cast Down", Zone::Graveyard);
    let victim = put_object(&mut state, PlayerId::P1, "Guttersnipe", Zone::Battlefield);
    let land = put_object(&mut state, PlayerId::P1, "Mountain", Zone::Battlefield);

    let legal = engine::legal_targets_for(TargetSpec::NonlegendaryCreature, &[], &state);
    assert!(legal.contains(&Target::Object(victim)));
    assert!(!legal.contains(&Target::Object(land)));

    let target = Target::Object(victim);
    let target_contract = StackTargetContractV4::capture(&state, target);
    effect::execute(
        &(CARD_DEFS[card_id("Cast Down") as usize].spell_effect)().unwrap(),
        &ExecCtx {
            source: cast_down,
            controller: PlayerId::P0,
            stack_item_id: None,
            targets: vec![target],
            target_contracts: vec![target_contract],
            discarded: Vec::new(),
            paid_cost_refs: Vec::new(),
            hidden_ability_source: None,
            ability_source_contract: None,
            kicked: false,
            optional_additional_cost_paid: None,
            x_value: 0,
        },
        &mut state,
    );
    assert_eq!(state.objects.get(victim).zone, Zone::Graveyard);
}

#[test]
fn breath_weapon_damages_non_dragons_as_one_batch() {
    let mut state = ready_state();
    let breath = put_object(&mut state, PlayerId::P0, "Breath Weapon", Zone::Graveyard);
    let medic = put_object(&mut state, PlayerId::P0, "Outlaw Medic", Zone::Battlefield);
    let guttersnipe = put_object(&mut state, PlayerId::P1, "Guttersnipe", Zone::Battlefield);
    let dragon = put_object(
        &mut state,
        PlayerId::P1,
        "Avenging Hunter",
        Zone::Battlefield,
    );
    let land = put_object(&mut state, PlayerId::P0, "Mountain", Zone::Battlefield);
    state.engine.event_history.clear();

    effect::execute(
        &(CARD_DEFS[card_id("Breath Weapon") as usize].spell_effect)().unwrap(),
        &ExecCtx::no_targets(breath, PlayerId::P0),
        &mut state,
    );

    assert_eq!(state.objects.get(medic).damage, 2);
    assert_eq!(state.objects.get(guttersnipe).damage, 2);
    assert_eq!(state.objects.get(dragon).damage, 0);
    assert_eq!(state.objects.get(land).damage, 0);
    assert_eq!(
        state
            .engine
            .event_history
            .iter()
            .filter(|event| matches!(event, CommittedEvent::Damage { amount: 2, .. }))
            .count(),
        2
    );
}

#[test]
fn outlaw_medic_lifelink_uses_final_damage_and_its_dies_trigger_draws() {
    let mut state = ready_state();
    let medic = put_object(&mut state, PlayerId::P0, "Outlaw Medic", Zone::Battlefield);
    state.players[PlayerId::P0.index()].life = 10;
    state.players[PlayerId::P1.index()].life = 10;
    state.engine.event_history.clear();
    state.engine.active_replacements.push(ActiveReplacement {
        id: 1,
        source: medic,
        kind: ReplacementEffectKind::PreventNextDamage {
            target: Target::Player(PlayerId::P1),
            remaining: 2,
        },
    });

    event::propose_and_commit(
        &mut state,
        ProposedEvent::damage(medic, Target::Player(PlayerId::P1), 5),
    );
    assert_eq!(state.players[PlayerId::P0.index()].life, 13);
    assert_eq!(state.players[PlayerId::P1.index()].life, 7);
    assert!(matches!(
        state.engine.event_history.as_slice(),
        [
            CommittedEvent::Damage { amount: 3, .. },
            CommittedEvent::LifeGain {
                player: PlayerId::P0,
                amount: 3,
            }
        ]
    ));

    let mut dies_state = ready_state();
    let medic = put_object(
        &mut dies_state,
        PlayerId::P0,
        "Outlaw Medic",
        Zone::Battlefield,
    );
    let hand_before = dies_state.players[PlayerId::P0.index()].hand.len();
    event::propose_and_commit(
        &mut dies_state,
        ProposedEvent::zone_change(medic, Zone::Graveyard),
    );
    let pending = trigger::collect_and_process(&mut dies_state);
    assert_eq!(pending.len(), 1);
    assert!(matches!(
        pending[0].effect,
        EffectOp::DrawCards {
            player: PlayerRef::Controller,
            count: 1,
        }
    ));
    execute_trigger(&mut dies_state, &pending[0]);
    assert_eq!(
        dies_state.players[PlayerId::P0.index()].hand.len(),
        hand_before + 1
    );
}

#[test]
fn familiar_makes_the_opponent_choose_a_discard_when_possible() {
    let mut state = ready_state();
    let choice_a = put_object(&mut state, PlayerId::P1, "Mountain", Zone::Hand);
    let choice_b = put_object(&mut state, PlayerId::P1, "Mountain", Zone::Hand);
    let familiar = put_object(&mut state, PlayerId::P0, "Refurbished Familiar", Zone::Hand);

    let pending = enter(&mut state, familiar);
    assert_eq!(pending.len(), 1);
    assert!(matches!(
        pending[0].effect,
        EffectOp::Conditional {
            cond: EffectCond::OpponentHasCardsInHand,
            ..
        }
    ));
    execute_trigger(&mut state, &pending[0]);
    assert!(matches!(
        engine::advance_until_decision(&mut state),
        Decision::Discard {
            player: PlayerId::P1,
            count: 1,
            ref choices,
        } if choices == &vec![choice_a, choice_b]
    ));
    engine::step(&mut state, Action::Discard(vec![choice_b])).unwrap();
    assert_eq!(state.objects.get(choice_b).zone, Zone::Graveyard);
    assert_eq!(state.objects.get(choice_a).zone, Zone::Hand);
    assert!(state.engine.pending_discard.is_none());
}

#[test]
fn familiar_draws_for_its_controller_when_the_opponent_cannot_discard() {
    let mut state = ready_state();
    let familiar = put_object(&mut state, PlayerId::P0, "Refurbished Familiar", Zone::Hand);
    assert!(state.players[PlayerId::P1.index()].hand.is_empty());
    let pending = enter(&mut state, familiar);
    assert_eq!(pending.len(), 1);
    let hand_before = state.players[PlayerId::P0.index()].hand.len();

    execute_trigger(&mut state, &pending[0]);
    assert_eq!(
        state.players[PlayerId::P0.index()].hand.len(),
        hand_before + 1
    );
    assert!(state.engine.pending_discard.is_none());
}
