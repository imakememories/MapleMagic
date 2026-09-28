//! Bounded Murmuring Mystic checkpoint.
//!
//! XMage rules baseline `0723fc0c2be922af47b0ef0539f28114cc23b998`
//! (`MurmuringMystic.java` blob `219e0325fd5e611f8a7d7a6551b5696f0418d014`,
//! `BirdIllusionToken.java` blob `4ee17731ce3e15d2a56bdb3ade47584b857c8833`)
//! defines the ordinary battlefield trigger "Whenever you cast an instant or
//! sorcery spell, create a 1/1 blue Bird Illusion creature token with flying."
//! These tests keep the rule on the existing generic
//! `SpellCast -> CastInstantOrSorcery -> CreateToken` path and exercise the
//! cast/copy, countering, trigger-order, snapshot, combat, and token-SBA
//! boundaries that can otherwise make a superficially correct trigger drift.

use mtg_kernel::card_def::{card_id_by_name, Keywords, Subtype, CARD_DEFS};
use mtg_kernel::engine::{self, Action, Decision};
use mtg_kernel::event::{self, CommittedEvent, ProposedEvent};
use mtg_kernel::ids::{ObjectId, PlayerId};
use mtg_kernel::mana::ManaColor;
use mtg_kernel::state::{Counters, GameObject, GameState, Step, Target, Zone};
use mtg_kernel::trigger;

fn card_id(name: &str) -> u16 {
    card_id_by_name(name).unwrap_or_else(|| panic!("{name} in CARD_DEFS"))
}

fn card_name(card_def: u16) -> String {
    CARD_DEFS[card_def as usize].name.to_string()
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

fn ready_state(mystics: usize) -> (GameState, Vec<ObjectId>) {
    let library = vec![card_id("Mountain"); 8];
    let mut state = GameState::new_from_libraries(&library, &library, card_name, 0x4D59_5354_4943);
    state.step = Step::Main1;
    state.active_player = PlayerId::P0;
    state.priority_player = PlayerId::P0;
    let mystics = (0..mystics)
        .map(|_| {
            put_object(
                &mut state,
                PlayerId::P0,
                "Murmuring Mystic",
                Zone::Battlefield,
            )
        })
        .collect();
    (state, mystics)
}

fn add_mana(state: &mut GameState, player: PlayerId, color: ManaColor, amount: u8) {
    state.players[player.index()].mana_pool[color.pool_index()] += amount;
}

fn bird_tokens(state: &GameState, player: PlayerId) -> Vec<ObjectId> {
    state.players[player.index()]
        .battlefield
        .iter()
        .copied()
        .filter(|&id| state.objects.get(id).card_def == card_id("Bird Illusion Token"))
        .collect()
}

fn cast_no_target(state: &mut GameState, player: PlayerId, spell: ObjectId) -> Decision {
    let offer = engine::advance_until_decision(state);
    assert!(matches!(
        offer,
        Decision::CastSpellOrPass {
            player: offered,
            ref castable_spells,
            ..
        } if offered == player && castable_spells.contains(&spell)
    ));
    engine::step(state, Action::CastSpell(spell)).unwrap();
    engine::advance_until_decision(state)
}

fn pass_until_stack_len(state: &mut GameState, wanted: usize) -> Decision {
    for _ in 0..64 {
        let decision = engine::advance_until_decision(state);
        if state.stack.len() <= wanted {
            return decision;
        }
        match decision {
            Decision::CastSpellOrPass { .. } => engine::step(state, Action::Pass).unwrap(),
            other => panic!("unexpected decision while resolving stack: {other:?}"),
        }
    }
    panic!("stack did not reach length {wanted}");
}

fn pass_until_copy_payment(state: &mut GameState) -> Decision {
    for _ in 0..64 {
        let decision = engine::advance_until_decision(state);
        match decision {
            Decision::ChooseSpellCopyPayment { .. } => return decision,
            Decision::CastSpellOrPass { .. } => engine::step(state, Action::Pass).unwrap(),
            other => panic!("unexpected decision before copy payment: {other:?}"),
        }
    }
    panic!("copy payment was never offered");
}

#[test]
fn mystic_casts_and_resolves_as_a_canonical_human_wizard_one_five() {
    let (mut state, _) = ready_state(0);
    add_mana(&mut state, PlayerId::P0, ManaColor::U, 4);
    let mystic = put_object(&mut state, PlayerId::P0, "Murmuring Mystic", Zone::Hand);

    assert!(matches!(
        cast_no_target(&mut state, PlayerId::P0, mystic),
        Decision::CastSpellOrPass { .. }
    ));
    assert_eq!(state.stack.len(), 1, "Mystic cannot trigger from the stack");
    pass_until_stack_len(&mut state, 0);

    let def = &CARD_DEFS[state.objects.get(mystic).card_def as usize];
    assert_eq!(def.subtypes, &[Subtype::Human, Subtype::Wizard]);
    assert_eq!(def.power, Some(1));
    assert_eq!(def.toughness, Some(5));
    assert_eq!(
        state.objects.get(mystic).v4.effective_subtype_ids,
        vec![Subtype::Human.stable_id(), Subtype::Wizard.stable_id()]
    );

    add_mana(&mut state, PlayerId::P0, ManaColor::R, 2);
    let rally = put_object(
        &mut state,
        PlayerId::P0,
        "Rally at the Hornburg",
        Zone::Hand,
    );
    assert!(matches!(
        cast_no_target(&mut state, PlayerId::P0, rally),
        Decision::CastSpellOrPass { .. }
    ));
    pass_until_stack_len(&mut state, 0);
    assert_eq!(engine::effective_power(&state, mystic), 1);
    assert_eq!(engine::effective_toughness(&state, mystic), 5);
    assert!(engine::has_effective_keyword(
        &state,
        mystic,
        Keywords::HASTE
    ));
}

#[test]
fn own_instant_and_sorcery_each_create_exactly_one_bird() {
    for spell_name in ["Mental Note", "Ponder"] {
        let (mut state, mystics) = ready_state(1);
        add_mana(&mut state, PlayerId::P0, ManaColor::U, 1);
        let spell = put_object(&mut state, PlayerId::P0, spell_name, Zone::Hand);

        let decision = cast_no_target(&mut state, PlayerId::P0, spell);
        assert!(matches!(decision, Decision::CastSpellOrPass { .. }));
        assert_eq!(state.stack.len(), 2, "{spell_name}");
        assert_eq!(
            state.stack[0].source, spell,
            "physical spell stays below trigger"
        );
        assert_eq!(state.stack[1].source, mystics[0], "Mystic owns the trigger");
        assert!(state.stack[1].inline_effect.is_some());

        pass_until_stack_len(&mut state, 1);
        let birds = bird_tokens(&state, PlayerId::P0);
        assert_eq!(birds.len(), 1, "{spell_name}");
        assert!(
            state.objects.get(birds[0]).summoning_sick,
            "a freshly created Bird must have summoning sickness"
        );
        assert!(state.engine.event_history.iter().any(|event| matches!(
            event,
            CommittedEvent::CreateToken {
                token_def,
                controller: PlayerId::P0,
                ..
            } if *token_def == card_id("Bird Illusion Token")
        )));
    }
}

#[test]
fn flashback_cast_triggers_mystic_and_retains_the_flashback_stack_identity() {
    let (mut state, mystics) = ready_state(1);
    let dart = put_object(&mut state, PlayerId::P0, "Lava Dart", Zone::Graveyard);
    let mountain = put_object(&mut state, PlayerId::P0, "Mountain", Zone::Battlefield);

    let offer = engine::advance_until_decision(&mut state);
    assert!(matches!(
        offer,
        Decision::CastSpellOrPass { ref castable_spells, .. } if castable_spells.contains(&dart)
    ));
    engine::step(&mut state, Action::CastSpell(dart)).unwrap();
    assert!(matches!(
        engine::advance_until_decision(&mut state),
        Decision::ChooseTargets { spell, .. } if spell == dart
    ));
    engine::step(
        &mut state,
        Action::ChooseTarget(Target::Player(PlayerId::P1)),
    )
    .unwrap();
    let decision = engine::advance_until_decision(&mut state);
    assert!(matches!(decision, Decision::CastSpellOrPass { .. }));
    assert_eq!(state.stack.len(), 2);
    assert_eq!(state.stack[0].source, dart);
    assert!(state.stack[0].is_flashback);
    assert_eq!(state.stack[1].source, mystics[0]);
    assert_eq!(state.objects.get(mountain).zone, Zone::Graveyard);

    pass_until_stack_len(&mut state, 1);
    assert_eq!(bird_tokens(&state, PlayerId::P0).len(), 1);
}

#[test]
fn creature_and_opponent_casts_do_not_trigger_mystic() {
    let (mut creature_state, _) = ready_state(1);
    add_mana(&mut creature_state, PlayerId::P0, ManaColor::R, 1);
    let creature = put_object(
        &mut creature_state,
        PlayerId::P0,
        "Masked Meower",
        Zone::Hand,
    );
    let decision = cast_no_target(&mut creature_state, PlayerId::P0, creature);
    assert!(matches!(decision, Decision::CastSpellOrPass { .. }));
    assert_eq!(creature_state.stack.len(), 1);
    assert!(creature_state.engine.pending_triggers.is_empty());
    assert!(bird_tokens(&creature_state, PlayerId::P0).is_empty());

    let (mut opponent_state, _) = ready_state(1);
    opponent_state.priority_player = PlayerId::P1;
    add_mana(&mut opponent_state, PlayerId::P1, ManaColor::U, 1);
    let opponent_spell = put_object(&mut opponent_state, PlayerId::P1, "Mental Note", Zone::Hand);
    let decision = cast_no_target(&mut opponent_state, PlayerId::P1, opponent_spell);
    assert!(matches!(decision, Decision::CastSpellOrPass { .. }));
    assert_eq!(opponent_state.stack.len(), 1);
    assert!(opponent_state.engine.pending_triggers.is_empty());
    assert!(bird_tokens(&opponent_state, PlayerId::P0).is_empty());
}

#[test]
fn chain_lightning_copy_is_not_cast_and_does_not_create_a_second_bird() {
    let (mut state, _) = ready_state(1);
    put_object(
        &mut state,
        PlayerId::P1,
        "Murmuring Mystic",
        Zone::Battlefield,
    );
    add_mana(&mut state, PlayerId::P0, ManaColor::R, 1);
    add_mana(&mut state, PlayerId::P1, ManaColor::R, 2);
    let chain = put_object(&mut state, PlayerId::P0, "Chain Lightning", Zone::Hand);

    let offer = engine::advance_until_decision(&mut state);
    assert!(matches!(
        offer,
        Decision::CastSpellOrPass { ref castable_spells, .. } if castable_spells.contains(&chain)
    ));
    engine::step(&mut state, Action::CastSpell(chain)).unwrap();
    engine::step(
        &mut state,
        Action::ChooseTarget(Target::Player(PlayerId::P1)),
    )
    .unwrap();
    assert!(matches!(
        engine::advance_until_decision(&mut state),
        Decision::CastSpellOrPass { .. }
    ));
    assert_eq!(state.stack.len(), 2, "physical cast plus Mystic trigger");
    pass_until_stack_len(&mut state, 1);
    assert_eq!(bird_tokens(&state, PlayerId::P0).len(), 1);

    assert!(matches!(
        pass_until_copy_payment(&mut state),
        Decision::ChooseSpellCopyPayment {
            player: PlayerId::P1,
            spell,
        } if spell == chain
    ));
    engine::step(&mut state, Action::ChooseSpellCopyPayment(true)).unwrap();
    let retarget = engine::advance_until_decision(&mut state);
    let copy = match retarget {
        Decision::ChooseSpellCopyRetarget {
            player: PlayerId::P1,
            copy,
        } => copy,
        other => panic!("expected copy retarget decision, got {other:?}"),
    };
    engine::step(&mut state, Action::ChooseSpellCopyRetarget(false)).unwrap();
    assert!(state
        .stack
        .iter()
        .any(|item| item.source == copy && item.is_copy));

    assert!(matches!(
        pass_until_copy_payment(&mut state),
        Decision::ChooseSpellCopyPayment {
            player: PlayerId::P1,
            spell,
        } if spell == copy
    ));
    engine::step(&mut state, Action::ChooseSpellCopyPayment(false)).unwrap();
    assert_eq!(bird_tokens(&state, PlayerId::P0).len(), 1);
    assert!(
        bird_tokens(&state, PlayerId::P1).is_empty(),
        "P1's Mystic must not see P1's virtual copy as a cast"
    );
    assert_eq!(
        state
            .engine
            .event_history
            .iter()
            .filter(|event| matches!(event, CommittedEvent::SpellCast { .. }))
            .count(),
        1,
        "the virtual copy must not emit a cast event"
    );
}

#[test]
fn countering_the_spell_does_not_erase_the_already_created_trigger() {
    let (mut state, _) = ready_state(1);
    add_mana(&mut state, PlayerId::P0, ManaColor::U, 1);
    add_mana(&mut state, PlayerId::P1, ManaColor::U, 2);
    let note = put_object(&mut state, PlayerId::P0, "Mental Note", Zone::Hand);
    let counter = put_object(&mut state, PlayerId::P1, "Counterspell", Zone::Hand);

    assert!(matches!(
        cast_no_target(&mut state, PlayerId::P0, note),
        Decision::CastSpellOrPass {
            player: PlayerId::P0,
            ..
        }
    ));
    assert_eq!(state.stack.len(), 2);
    engine::step(&mut state, Action::Pass).unwrap();
    assert!(matches!(
        engine::advance_until_decision(&mut state),
        Decision::CastSpellOrPass {
            player: PlayerId::P1,
            ref castable_spells,
            ..
        } if castable_spells.contains(&counter)
    ));
    engine::step(&mut state, Action::CastSpell(counter)).unwrap();
    assert!(matches!(
        engine::advance_until_decision(&mut state),
        Decision::ChooseTargets { spell, .. } if spell == counter
    ));
    engine::step(&mut state, Action::ChooseTarget(Target::Object(note))).unwrap();
    assert!(matches!(
        engine::advance_until_decision(&mut state),
        Decision::CastSpellOrPass { .. }
    ));
    assert_eq!(state.stack.len(), 3);

    pass_until_stack_len(&mut state, 1);
    assert_eq!(state.objects.get(note).zone, Zone::Graveyard);
    assert!(bird_tokens(&state, PlayerId::P0).is_empty());
    pass_until_stack_len(&mut state, 0);
    assert_eq!(bird_tokens(&state, PlayerId::P0).len(), 1);
}

#[test]
fn trigger_resolves_after_its_mystic_source_leaves_the_battlefield() {
    let (mut state, mystics) = ready_state(1);
    add_mana(&mut state, PlayerId::P0, ManaColor::U, 1);
    let note = put_object(&mut state, PlayerId::P0, "Mental Note", Zone::Hand);
    assert!(matches!(
        cast_no_target(&mut state, PlayerId::P0, note),
        Decision::CastSpellOrPass { .. }
    ));
    assert_eq!(state.stack.len(), 2);

    event::propose_and_commit(
        &mut state,
        ProposedEvent::zone_change(mystics[0], Zone::Graveyard),
    );
    assert_eq!(state.objects.get(mystics[0]).zone, Zone::Graveyard);
    pass_until_stack_len(&mut state, 1);
    assert_eq!(bird_tokens(&state, PlayerId::P0).len(), 1);
}

#[test]
fn two_mystics_use_explicit_trigger_order_and_snapshot_restore_exactly() {
    let (mut state, mystics) = ready_state(2);
    add_mana(&mut state, PlayerId::P0, ManaColor::U, 1);
    let note = put_object(&mut state, PlayerId::P0, "Mental Note", Zone::Hand);

    let decision = cast_no_target(&mut state, PlayerId::P0, note);
    let pending = match decision {
        Decision::OrderTriggers {
            player: PlayerId::P0,
            pending,
        } => pending,
        other => panic!("expected two-Mystic trigger ordering, got {other:?}"),
    };
    assert_eq!(
        pending
            .iter()
            .map(|trigger| trigger.source)
            .collect::<Vec<_>>(),
        mystics
    );
    let snapshot = state.snapshot();

    engine::step(&mut state, Action::OrderTriggers(vec![1, 0])).unwrap();
    assert_eq!(
        state
            .stack
            .iter()
            .map(|item| item.source)
            .collect::<Vec<_>>(),
        vec![note, mystics[1], mystics[0]]
    );
    pass_until_stack_len(&mut state, 1);
    assert_eq!(bird_tokens(&state, PlayerId::P0).len(), 2);
    let first_result = state.clone();

    state.restore(&snapshot);
    assert!(matches!(
        engine::advance_until_decision(&mut state),
        Decision::OrderTriggers {
            player: PlayerId::P0,
            ref pending,
        } if pending.iter().map(|trigger| trigger.source).collect::<Vec<_>>() == mystics
    ));
    engine::step(&mut state, Action::OrderTriggers(vec![1, 0])).unwrap();
    pass_until_stack_len(&mut state, 1);
    assert_eq!(state, first_result);
}

#[test]
fn bird_token_has_flying_in_combat_and_ceases_after_lethal_sba() {
    let (mut state, _) = ready_state(1);
    add_mana(&mut state, PlayerId::P0, ManaColor::R, 1);
    let festivities = put_object(&mut state, PlayerId::P0, "End the Festivities", Zone::Hand);
    cast_no_target(&mut state, PlayerId::P0, festivities);
    pass_until_stack_len(&mut state, 0);
    let bird = bird_tokens(&state, PlayerId::P0)[0];
    assert!(CARD_DEFS[state.objects.get(bird).card_def as usize]
        .keywords
        .has(Keywords::FLYING));

    let grounded = put_object(&mut state, PlayerId::P1, "Guttersnipe", Zone::Battlefield);
    let flyer = put_object(
        &mut state,
        PlayerId::P1,
        "Sneaky Snacker",
        Zone::Battlefield,
    );
    state.step = Step::DeclareBlockers;
    state.active_player = PlayerId::P0;
    state.priority_player = PlayerId::P0;
    state.engine.priority_passes = [false, false];
    state.engine.combat.attackers = vec![bird];
    state.engine.combat.attackers_declared = true;
    match engine::advance_until_decision(&mut state) {
        Decision::DeclareBlockers {
            attackers,
            legal_blockers,
            ..
        } => {
            assert_eq!(attackers, vec![bird]);
            assert_eq!(legal_blockers, vec![(bird, vec![flyer])]);
            assert!(!legal_blockers[0].1.contains(&grounded));
        }
        other => panic!("expected blocker declaration, got {other:?}"),
    }

    event::propose_and_commit(
        &mut state,
        ProposedEvent::damage(grounded, Target::Object(bird), 1),
    );
    trigger::sba_fixed_point(&mut state);
    assert_eq!(state.objects.get(bird).zone, Zone::Graveyard);
    assert!(!state.players[0].battlefield.contains(&bird));
    assert!(!state.players[0].graveyard.contains(&bird));
    assert!(!state.exile.contains(&bird));
    assert!(!state.stack.iter().any(|item| item.source == bird));
}
