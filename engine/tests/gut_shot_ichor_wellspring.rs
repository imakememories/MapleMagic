//! Focused rules coverage for Gut Shot and Ichor Wellspring.
//!
//! The checked-in Mage source baseline is `0723fc0c2be922af47b0ef0539f28114cc23b998`.
//! `GutShot.java` has SHA-256
//! `ded03eb589cb1a247b56b758226bea6ecae0c3a1ddd6102b5697196a67656b24`;
//! `IchorWellspring.java` has SHA-256
//! `c24963ec940f1bf1b1ac7a8b6d31145752bf562998d64c63763398d359135771`.

use mtg_kernel::card_def::{card_id_by_name, CardCapability, CARD_DEFS};
use mtg_kernel::effect::{self, EffectOp, ExecCtx};
use mtg_kernel::engine::{self, Action, Decision};
use mtg_kernel::event::{self, ProposedEvent};
use mtg_kernel::ids::{ObjectId, PlayerId};
use mtg_kernel::mana::{self, ManaColor};
use mtg_kernel::state::{Counters, GameObject, GameState, Step, Target, Zone};
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
        GameState::new_from_libraries(&library, &library, card_name, 0x0047_5554_5348_4F54);
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

fn resolve_stack(state: &mut GameState) {
    for _ in 0..32 {
        if state.stack.is_empty() {
            return;
        }
        match engine::advance_until_decision(state) {
            Decision::CastSpellOrPass { .. } => engine::step(state, Action::Pass).unwrap(),
            other => panic!("unexpected decision while resolving stack: {other:?}"),
        }
    }
    panic!("stack did not resolve");
}

fn execute_trigger(state: &mut GameState, source: ObjectId, controller: PlayerId, op: &EffectOp) {
    effect::execute(
        op,
        &ExecCtx {
            source,
            controller,
            stack_item_id: None,
            targets: Vec::new(),
            target_contracts: Vec::new(),
            discarded: Vec::new(),
            paid_cost_refs: Vec::new(),
            hidden_ability_source: None,
            ability_source_contract: None,
            kicked: false,
            optional_additional_cost_paid: None,
            x_value: 0,
        },
        state,
    );
}

#[test]
fn gut_shot_uses_the_shared_phyrexian_any_target_contract() {
    let def = &CARD_DEFS[card_id("Gut Shot") as usize];
    assert_eq!(def.capability, CardCapability::Full);
    assert_eq!(def.cost.pips.len(), 1);
    assert!(matches!(
        def.cost.pips[0],
        mana::Pip::Phyrexian(ManaColor::R)
    ));
    assert_eq!(def.target_spec, mtg_kernel::card_def::TargetSpec::AnyTarget);
    assert!(matches!(
        (def.spell_effect)(),
        Some(EffectOp::DealDamage {
            target: effect::TargetRef::Target(0),
            amount: 1
        })
    ));
}

#[test]
fn gut_shot_pays_two_life_without_red_and_deals_one_damage() {
    let mut state = ready_state();
    let gut_shot = put_object(&mut state, PlayerId::P0, "Gut Shot", Zone::Hand);

    assert!(matches!(
        engine::advance_until_decision(&mut state),
        Decision::CastSpellOrPass {
            ref castable_spells,
            ..
        } if castable_spells.contains(&gut_shot)
    ));
    engine::step(&mut state, Action::CastSpell(gut_shot)).unwrap();
    assert!(matches!(
        engine::advance_until_decision(&mut state),
        Decision::ChooseTargets { spell, .. } if spell == gut_shot
    ));
    engine::step(
        &mut state,
        Action::ChooseTarget(Target::Player(PlayerId::P1)),
    )
    .unwrap();
    assert!(matches!(
        engine::advance_until_decision(&mut state),
        Decision::CastSpellOrPass { .. }
    ));

    assert_eq!(state.players[PlayerId::P0.index()].life, 18);
    resolve_stack(&mut state);
    assert_eq!(state.players[PlayerId::P1.index()].life, 19);
    assert_eq!(state.objects.get(gut_shot).zone, Zone::Graveyard);
}

#[test]
fn gut_shot_prefers_red_mana_and_cannot_pay_life_it_does_not_have() {
    let mut red = ready_state();
    red.players[PlayerId::P0.index()].mana_pool[ManaColor::R.pool_index()] = 1;
    let gut_shot = put_object(&mut red, PlayerId::P0, "Gut Shot", Zone::Hand);
    engine::step(&mut red, Action::CastSpell(gut_shot)).unwrap();
    engine::step(&mut red, Action::ChooseTarget(Target::Player(PlayerId::P1))).unwrap();
    assert!(matches!(
        engine::advance_until_decision(&mut red),
        Decision::CastSpellOrPass { .. }
    ));
    assert_eq!(red.players[PlayerId::P0.index()].life, 20);
    assert_eq!(
        red.players[PlayerId::P0.index()].mana_pool[ManaColor::R.pool_index()],
        0
    );

    let mut low_life = ready_state();
    low_life.players[PlayerId::P0.index()].life = 1;
    assert!(mana::can_pay(
        &CARD_DEFS[card_id("Gut Shot") as usize].cost,
        0,
        PlayerId::P0,
        &low_life
    )
    .is_none());
    low_life.players[PlayerId::P0.index()].life = 2;
    assert_eq!(
        mana::can_pay(
            &CARD_DEFS[card_id("Gut Shot") as usize].cost,
            0,
            PlayerId::P0,
            &low_life
        )
        .unwrap()
        .life_paid,
        2
    );
}

#[test]
fn ichor_wellspring_draws_once_on_entry_and_once_on_death() {
    let def = &CARD_DEFS[card_id("Ichor Wellspring") as usize];
    assert_eq!(def.capability, CardCapability::Full);
    assert!(matches!(
        (def.spell_effect)(),
        Some(EffectOp::MoveObject {
            object: effect::ObjectRef::ThisSource,
            to_zone: Zone::Battlefield
        })
    ));

    let mut state = ready_state();
    let wellspring = put_object(&mut state, PlayerId::P0, "Ichor Wellspring", Zone::Hand);
    let starting_hand = state.players[PlayerId::P0.index()].hand.len();

    event::propose_and_commit(
        &mut state,
        ProposedEvent::zone_change(wellspring, Zone::Battlefield),
    );
    let entry = trigger::collect_and_process(&mut state);
    assert_eq!(entry.len(), 1);
    execute_trigger(
        &mut state,
        entry[0].source,
        entry[0].controller,
        &entry[0].effect,
    );
    assert_eq!(
        state.players[PlayerId::P0.index()].hand.len(),
        starting_hand,
        "the Wellspring left hand and its ETB draw replaced it"
    );
    state.engine.event_log.clear();

    event::propose_and_commit(
        &mut state,
        ProposedEvent::zone_change(wellspring, Zone::Graveyard),
    );
    let dies = trigger::collect_and_process(&mut state);
    assert_eq!(dies.len(), 1);
    execute_trigger(
        &mut state,
        dies[0].source,
        dies[0].controller,
        &dies[0].effect,
    );
    assert_eq!(
        state.players[PlayerId::P0.index()].hand.len(),
        starting_hand + 1
    );
}

#[test]
fn ichor_wellspring_does_not_draw_when_exiled_from_battlefield() {
    let mut state = ready_state();
    let wellspring = put_object(
        &mut state,
        PlayerId::P0,
        "Ichor Wellspring",
        Zone::Battlefield,
    );
    event::propose_and_commit(
        &mut state,
        ProposedEvent::zone_change(wellspring, Zone::Exile),
    );
    assert!(trigger::collect_and_process(&mut state).is_empty());
}
