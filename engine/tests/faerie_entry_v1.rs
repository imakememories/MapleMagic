//! Focused rules coverage for Faerie Miscreant and Faerie Seer.
//!
//! Oracle baseline: checked-in Mage commit
//! `0723fc0c2be922af47b0ef0539f28114cc23b998`. `FaerieMiscreant.java`
//! has SHA-256
//! `ee7ca9fa5d82da41dfef6a439dac16c518482e88a0682d41d8adabd8eb8df`;
//! `FaerieSeer.java` has SHA-256
//! `1e7014901018987c775a22b20b1e9edf11f9f0711dc7bf123eac87df59bd774c`.

use mtg_kernel::card_def::{card_id_by_name, CardCapability, CardType, Keywords, CARD_DEFS};
use mtg_kernel::effect::{self, EffectCond, EffectOp, ExecCtx, PlayerRef};
use mtg_kernel::event::{self, ProposedEvent};
use mtg_kernel::ids::{ObjectId, PlayerId};
use mtg_kernel::mana::{ManaColor, Pip};
use mtg_kernel::state::{Counters, GameObject, GameState, Step, Zone};
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
        GameState::new_from_libraries(&library, &library, card_name, 0x4641_4552_4945_0001);
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

fn execute_trigger(state: &mut GameState, trigger: &trigger::PendingTrigger) {
    effect::execute(
        &trigger.effect,
        &ExecCtx {
            source: trigger.source,
            controller: trigger.controller,
            stack_item_id: None,
            targets: Vec::new(),
            target_contracts: Vec::new(),
            discarded: Vec::new(),
            paid_cost_refs: Vec::new(),
            hidden_ability_source: None,
            ability_source_contract: trigger.source_contract,
            kicked: false,
            optional_additional_cost_paid: None,
            x_value: 0,
        },
        state,
    );
}

#[test]
fn definitions_are_full_flying_one_drops() {
    for name in ["Faerie Miscreant", "Faerie Seer"] {
        let def = &CARD_DEFS[card_id(name) as usize];
        assert_eq!(def.capability, CardCapability::Full);
        assert_eq!(def.types, &[CardType::Creature]);
        assert_eq!((def.power, def.toughness), (Some(1), Some(1)));
        assert_eq!(def.cost.generic, 0);
        assert_eq!(def.cost.pips, &[Pip::Colored(ManaColor::U)]);
        assert!(def.keywords.has(Keywords::FLYING));
        assert!(matches!(
            (def.spell_effect)(),
            Some(EffectOp::MoveObject {
                object: effect::ObjectRef::ThisSource,
                to_zone: Zone::Battlefield,
            })
        ));
    }
}

#[test]
fn miscreant_requires_another_copy_when_entry_event_happens() {
    let mut state = ready_state();
    let first = put_object(&mut state, PlayerId::P0, "Faerie Miscreant", Zone::Hand);
    assert!(enter(&mut state, first).is_empty());

    let second = put_object(&mut state, PlayerId::P0, "Faerie Miscreant", Zone::Hand);
    let pending = enter(&mut state, second);
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].source, second);
    assert!(matches!(
        pending[0].effect,
        EffectOp::Conditional {
            cond: EffectCond::ControlsAnotherSourceCard,
            ..
        }
    ));
}

#[test]
fn miscreant_checks_the_same_condition_again_on_resolution() {
    let mut state = ready_state();
    let other = put_object(
        &mut state,
        PlayerId::P0,
        "Faerie Miscreant",
        Zone::Battlefield,
    );
    let source = put_object(&mut state, PlayerId::P0, "Faerie Miscreant", Zone::Hand);
    let pending = enter(&mut state, source);
    assert_eq!(pending.len(), 1);
    let hand_before = state.players[PlayerId::P0.index()].hand.len();

    event::propose_and_commit(
        &mut state,
        ProposedEvent::zone_change(other, Zone::Graveyard),
    );
    state.engine.event_log.clear();
    execute_trigger(&mut state, &pending[0]);
    assert_eq!(state.players[PlayerId::P0.index()].hand.len(), hand_before);
}

#[test]
fn miscreant_condition_excludes_the_source_but_survives_source_leaving() {
    let mut state = ready_state();
    let other = put_object(
        &mut state,
        PlayerId::P0,
        "Faerie Miscreant",
        Zone::Battlefield,
    );
    let source = put_object(&mut state, PlayerId::P0, "Faerie Miscreant", Zone::Hand);
    let pending = enter(&mut state, source);
    assert_eq!(pending.len(), 1);
    let hand_before = state.players[PlayerId::P0.index()].hand.len();

    event::propose_and_commit(
        &mut state,
        ProposedEvent::zone_change(source, Zone::Graveyard),
    );
    state.engine.event_log.clear();
    execute_trigger(&mut state, &pending[0]);
    assert_eq!(
        state.players[PlayerId::P0.index()].hand.len(),
        hand_before + 1
    );
    assert_eq!(state.objects.get(other).zone, Zone::Battlefield);
}

#[test]
fn faerie_seer_entry_queues_the_shared_private_scry_two_program() {
    let mut state = ready_state();
    let seer = put_object(&mut state, PlayerId::P0, "Faerie Seer", Zone::Hand);
    let pending = enter(&mut state, seer);
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].source, seer);
    assert_eq!(
        pending[0].effect,
        EffectOp::Scry {
            player: PlayerRef::Controller,
            count: 2,
        }
    );
    assert!(effect::contains_player_choice(&pending[0].effect));
}
