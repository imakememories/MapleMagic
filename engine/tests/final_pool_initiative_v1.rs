//! Initiative and Undercity authority checks against Mage
//! `a5c90fe180021e70e2a644ade00eeab07f857a40`.

use mtg_kernel::card_def::{
    card_id_by_name, CardCapability, CardType, Keywords, Subtype, CARD_DEFS,
};
use mtg_kernel::effect::{EffectOp, EffectTargetSelectionPurpose, PendingEffectChoice};
use mtg_kernel::engine::{self, Action, Decision, UnsupportedMechanic};
use mtg_kernel::event::{self, CommittedEvent, ProposedEvent};
use mtg_kernel::ids::{ObjectId, PlayerId, StackItemId};
use mtg_kernel::state::{
    AbilitySourceContractV4, Counters, GameObject, GameState, InitiativeTriggerBindingV1,
    InitiativeTriggerKindV1, ObjectStateV4, StackItemKind, Step, Target, UndercityRoomV1, Zone,
    UNDERCITY_DUNGEON_ID_V1,
};
use mtg_kernel::trigger;

fn card_id(name: &str) -> u16 {
    card_id_by_name(name).unwrap_or_else(|| panic!("{name} must exist in CARD_DEFS"))
}

fn card_name(card_def: u16) -> String {
    CARD_DEFS[card_def as usize].name.to_string()
}

fn ready_state(seed: u64) -> GameState {
    let mut state = GameState::new_from_libraries(&[], &[], card_name, seed);
    state.active_player = PlayerId::P0;
    state.priority_player = PlayerId::P0;
    state.step = Step::Main1;
    state
}

fn put_object(state: &mut GameState, player: PlayerId, name: &str, zone: Zone) -> ObjectId {
    let card_def = card_id(name);
    let id = state.objects.push(GameObject {
        card_def,
        name: CARD_DEFS[card_def as usize].object_name.to_string(),
        owner: player,
        controller: player,
        zone,
        tapped: false,
        summoning_sick: false,
        damage: 0,
        counters: Counters::default(),
        attachments: Vec::new(),
        v4: ObjectStateV4::from_card_def(card_def),
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
        Zone::Stack => panic!("test helper does not fabricate stack objects"),
    }
    id
}

fn collect_triggers(state: &mut GameState) {
    let pending = trigger::collect_and_process(state);
    state.engine.pending_triggers.extend(pending);
}

fn initiative_binding(item_effect: &Option<EffectOp>) -> Option<InitiativeTriggerBindingV1> {
    match item_effect {
        Some(EffectOp::ResolveInitiativeTrigger { binding }) => Some(*binding),
        _ => None,
    }
}

fn stack_binding(state: &GameState) -> Option<InitiativeTriggerBindingV1> {
    state
        .stack
        .last()
        .and_then(|item| initiative_binding(&item.inline_effect))
}

fn root_is_live(state: &GameState, id: StackItemId) -> bool {
    state.stack.iter().any(|item| item.v4.stack_item_id == id)
        || state
            .engine
            .pending_effect
            .as_ref()
            .is_some_and(|pending| pending.resolving_item.v4.stack_item_id == id)
}

fn advance_to_initiative_stack(
    state: &mut GameState,
    expected: InitiativeTriggerKindV1,
) -> InitiativeTriggerBindingV1 {
    for _ in 0..48 {
        let decision = engine::advance_until_decision(state);
        if let Some(binding) = stack_binding(state) {
            if binding.kind == expected {
                return binding;
            }
        }
        match decision {
            Decision::CastSpellOrPass { .. } => engine::step(state, Action::Pass).unwrap(),
            Decision::OrderTriggers { ref pending, .. } if pending.len() == 1 => {
                engine::step(state, Action::OrderTriggers(vec![0])).unwrap()
            }
            Decision::Halted { mechanic, source } => {
                panic!("unexpected halt {mechanic:?} from {source}")
            }
            other => panic!("unexpected decision before Initiative stack item: {other:?}"),
        }
    }
    panic!("Initiative stack item was not reached")
}

fn resolve_top_until_choice(state: &mut GameState) -> Option<Decision> {
    let root = state
        .stack
        .last()
        .expect("stack item to resolve")
        .v4
        .stack_item_id;
    for _ in 0..48 {
        let decision = engine::advance_until_decision(state);
        if !root_is_live(state, root) {
            return Some(decision);
        }
        match decision {
            Decision::CastSpellOrPass { .. } => engine::step(state, Action::Pass).unwrap(),
            Decision::Halted { mechanic, source } => {
                panic!("unexpected halt {mechanic:?} from {source}")
            }
            other => return Some(other),
        }
    }
    panic!("stack item did not resolve")
}

fn assert_eventually_invalid_continuation(state: &mut GameState) {
    for _ in 0..24 {
        match engine::advance_until_decision(state) {
            Decision::Halted {
                mechanic: UnsupportedMechanic::InvalidEffectContinuation,
                ..
            } => return,
            Decision::CastSpellOrPass { .. } => engine::step(state, Action::Pass).unwrap(),
            other => panic!("tampered state remained actionable: {other:?}"),
        }
    }
    panic!("tampered state did not fail closed")
}

fn install_initiative_source(
    state: &mut GameState,
    holder: PlayerId,
) -> (ObjectId, AbilitySourceContractV4) {
    let hunter = put_object(state, holder, "Avenging Hunter", Zone::Battlefield);
    let mut source = AbilitySourceContractV4::capture(state, hunter);
    source.controller = holder;
    state.initiative = Some(holder);
    state.engine.initiative_source = Some(source);
    (hunter, source)
}

fn enqueue_initiative(
    state: &mut GameState,
    player: PlayerId,
    source: AbilitySourceContractV4,
    kind: InitiativeTriggerKindV1,
) -> InitiativeTriggerBindingV1 {
    let binding = event::log_initiative_trigger(state, player, source, kind).unwrap();
    collect_triggers(state);
    binding
}

fn set_room(state: &mut GameState, player: PlayerId, room: Option<UndercityRoomV1>) {
    let dungeon = &mut state.players[player.index()].dungeon;
    dungeon.dungeon_id = room.map(|_| UNDERCITY_DUNGEON_ID_V1);
    dungeon.room_id = room.map(UndercityRoomV1::stable_id);
}

fn latest_room_event(state: &GameState) -> Option<UndercityRoomV1> {
    state
        .engine
        .event_history
        .iter()
        .rev()
        .find_map(|event| match event {
            CommittedEvent::InitiativeTrigger { binding } => match binding.kind {
                InitiativeTriggerKindV1::UndercityRoom(room) => Some(room),
                _ => None,
            },
            _ => None,
        })
}

#[test]
fn avenging_hunter_definition_and_etb_make_separate_venture_and_room_stack_items() {
    let hunter_def = &CARD_DEFS[card_id("Avenging Hunter") as usize];
    assert_eq!(hunter_def.capability, CardCapability::Full);
    assert_eq!(hunter_def.types, &[CardType::Creature]);
    assert_eq!(hunter_def.subtypes, &[Subtype::Dragon, Subtype::Ranger]);
    assert_eq!((hunter_def.power, hunter_def.toughness), (Some(5), Some(4)));
    assert!(hunter_def.keywords.has(Keywords::TRAMPLE));

    let mut state = ready_state(0x4156_454e_4749_4e47);
    let hunter = put_object(&mut state, PlayerId::P0, "Avenging Hunter", Zone::Hand);
    event::propose_and_commit(
        &mut state,
        ProposedEvent::zone_change(hunter, Zone::Battlefield),
    );
    collect_triggers(&mut state);

    for _ in 0..16 {
        let decision = engine::advance_until_decision(&mut state);
        if state.stack.last().is_some_and(|item| {
            item.source == hunter
                && item.inline_effect
                    == Some(EffectOp::TakeInitiative {
                        player: mtg_kernel::effect::PlayerRef::Controller,
                    })
        }) {
            break;
        }
        match decision {
            Decision::CastSpellOrPass { .. } => engine::step(&mut state, Action::Pass).unwrap(),
            other => panic!("unexpected ETB placement decision: {other:?}"),
        }
    }
    assert_eq!(state.initiative, None);
    assert_eq!(state.players[0].dungeon.room_id, None);
    assert_eq!(
        state.stack.last().unwrap().kind,
        StackItemKind::TriggeredAbility
    );

    resolve_top_until_choice(&mut state);
    let venture = stack_binding(&state).expect("taking Initiative queues venture separately");
    assert_eq!(venture.kind, InitiativeTriggerKindV1::VentureAfterTaking);
    assert_eq!(state.initiative, Some(PlayerId::P0));
    assert_eq!(state.players[0].dungeon.room_id, None);

    resolve_top_until_choice(&mut state);
    let room = stack_binding(&state).expect("venture queues room ability separately");
    assert_eq!(
        room.kind,
        InitiativeTriggerKindV1::UndercityRoom(UndercityRoomV1::SecretEntrance)
    );
    assert_eq!(
        state.players[0].dungeon.room_id,
        Some(UndercityRoomV1::SecretEntrance.stable_id())
    );

    let decision = resolve_top_until_choice(&mut state).expect("Secret Entrance search prompt");
    assert!(matches!(
        decision,
        Decision::ChooseEffectTargets {
            player: PlayerId::P0,
            min_targets: 0,
            max_targets: 1,
            can_finish: true,
            ..
        }
    ));
}

#[test]
fn every_undercity_route_edge_is_exact_and_throne_completes_the_dungeon() {
    let base = ready_state(0x554e_4445_5243_4954);
    let cases: &[(Option<UndercityRoomV1>, &[UndercityRoomV1])] = &[
        (None, &[UndercityRoomV1::SecretEntrance]),
        (
            Some(UndercityRoomV1::SecretEntrance),
            &[UndercityRoomV1::Forge, UndercityRoomV1::LostWell],
        ),
        (
            Some(UndercityRoomV1::Forge),
            &[UndercityRoomV1::Trap, UndercityRoomV1::Arena],
        ),
        (
            Some(UndercityRoomV1::LostWell),
            &[UndercityRoomV1::Arena, UndercityRoomV1::Stash],
        ),
        (Some(UndercityRoomV1::Trap), &[UndercityRoomV1::Archives]),
        (
            Some(UndercityRoomV1::Arena),
            &[UndercityRoomV1::Archives, UndercityRoomV1::Catacombs],
        ),
        (Some(UndercityRoomV1::Stash), &[UndercityRoomV1::Catacombs]),
        (
            Some(UndercityRoomV1::Archives),
            &[UndercityRoomV1::ThroneOfTheDeadThree],
        ),
        (
            Some(UndercityRoomV1::Catacombs),
            &[UndercityRoomV1::ThroneOfTheDeadThree],
        ),
    ];

    for &(from, legal) in cases {
        for (option, &expected) in legal.iter().enumerate() {
            let mut state = base.clone();
            let (_, source) = install_initiative_source(&mut state, PlayerId::P0);
            set_room(&mut state, PlayerId::P0, from);
            enqueue_initiative(
                &mut state,
                PlayerId::P0,
                source,
                InitiativeTriggerKindV1::VentureAtUpkeep,
            );
            advance_to_initiative_stack(&mut state, InitiativeTriggerKindV1::VentureAtUpkeep);
            let decision = resolve_top_until_choice(&mut state);
            if legal.len() > 1 {
                assert!(matches!(
                    decision,
                    Some(Decision::ChooseEffectOption {
                        player: PlayerId::P0,
                        option_count: 2,
                        ..
                    })
                ));
                engine::step(&mut state, Action::ChooseEffectOption(option as u16)).unwrap();
                let _ = engine::advance_until_decision(&mut state);
            }
            assert_eq!(latest_room_event(&state), Some(expected), "edge {from:?}");
            if expected == UndercityRoomV1::ThroneOfTheDeadThree {
                assert_eq!(state.players[0].dungeon.dungeon_id, None);
                assert_eq!(state.players[0].dungeon.room_id, None);
                assert_eq!(
                    state.players[0].dungeon.completed_dungeons,
                    vec![UNDERCITY_DUNGEON_ID_V1]
                );
            } else {
                assert_eq!(state.players[0].dungeon.room_id, Some(expected.stable_id()));
            }
        }
    }
}

#[test]
fn two_unblocked_creatures_batch_one_transfer_then_one_venture_and_upkeep_ventures() {
    let mut combat = ready_state(0x4241_5443_4844_4d47);
    let _ = install_initiative_source(&mut combat, PlayerId::P1);
    let first = put_object(
        &mut combat,
        PlayerId::P0,
        "Voldaren Epicure",
        Zone::Battlefield,
    );
    let second = put_object(
        &mut combat,
        PlayerId::P0,
        "Voldaren Epicure",
        Zone::Battlefield,
    );
    combat.step = Step::DeclareAttackers;
    combat.active_player = PlayerId::P0;
    combat.priority_player = PlayerId::P0;

    assert!(matches!(
        engine::advance_until_decision(&mut combat),
        Decision::DeclareAttackers {
            player: PlayerId::P0,
            ref eligible,
        } if eligible.contains(&first) && eligible.contains(&second)
    ));
    engine::step(&mut combat, Action::DeclareAttackers(vec![first, second])).unwrap();
    for _ in 0..24 {
        match engine::advance_until_decision(&mut combat) {
            Decision::CastSpellOrPass { .. } => engine::step(&mut combat, Action::Pass).unwrap(),
            Decision::DeclareBlockers {
                player: PlayerId::P1,
                ..
            } => {
                engine::step(&mut combat, Action::DeclareBlockers(Vec::new())).unwrap();
                break;
            }
            other => panic!("unexpected path to blockers: {other:?}"),
        }
    }
    advance_to_initiative_stack(&mut combat, InitiativeTriggerKindV1::CombatTransfer);
    assert_eq!(combat.players[1].life, 18);
    assert_eq!(combat.initiative, Some(PlayerId::P1));
    assert_eq!(
        combat
            .engine
            .event_history
            .iter()
            .filter(|event| matches!(
                event,
                CommittedEvent::InitiativeTrigger { binding }
                    if binding.kind == InitiativeTriggerKindV1::CombatTransfer
            ))
            .count(),
        1,
        "one or more creatures is one batch trigger"
    );
    assert_eq!(
        combat
            .engine
            .event_history
            .iter()
            .filter(|event| matches!(
                event,
                CommittedEvent::CombatDamageToPlayer {
                    player: PlayerId::P1,
                    ..
                }
            ))
            .count(),
        2
    );

    resolve_top_until_choice(&mut combat);
    assert_eq!(combat.initiative, Some(PlayerId::P0));
    assert_eq!(
        stack_binding(&combat).unwrap().kind,
        InitiativeTriggerKindV1::VentureAfterTaking
    );
    assert_eq!(
        combat
            .engine
            .event_history
            .iter()
            .filter(|event| matches!(
                event,
                CommittedEvent::InitiativeTrigger { binding }
                    if binding.kind == InitiativeTriggerKindV1::VentureAfterTaking
            ))
            .count(),
        1
    );

    let mut upkeep = ready_state(0x5550_4b45_4550_0001);
    let (_, upkeep_source) = install_initiative_source(&mut upkeep, PlayerId::P0);
    upkeep.step = Step::Untap;
    upkeep.active_player = PlayerId::P0;
    upkeep.priority_player = PlayerId::P0;
    advance_to_initiative_stack(&mut upkeep, InitiativeTriggerKindV1::VentureAtUpkeep);
    assert_eq!(upkeep.initiative, Some(PlayerId::P0));
    assert_eq!(upkeep.players[0].dungeon.room_id, None);
    assert_eq!(upkeep.engine.initiative_source, Some(upkeep_source));
}

#[test]
fn taking_initiative_again_while_already_holder_still_ventures() {
    let mut state = ready_state(0x5341_4d45_484f_4c44);
    let (_, original_source) = install_initiative_source(&mut state, PlayerId::P0);
    let second_hunter = put_object(&mut state, PlayerId::P0, "Avenging Hunter", Zone::Hand);
    event::propose_and_commit(
        &mut state,
        ProposedEvent::zone_change(second_hunter, Zone::Battlefield),
    );
    collect_triggers(&mut state);
    for _ in 0..16 {
        let decision = engine::advance_until_decision(&mut state);
        if state.stack.last().is_some_and(|item| {
            item.source == second_hunter
                && matches!(item.inline_effect, Some(EffectOp::TakeInitiative { .. }))
        }) {
            break;
        }
        match decision {
            Decision::CastSpellOrPass { .. } => engine::step(&mut state, Action::Pass).unwrap(),
            other => panic!("unexpected same-holder ETB placement: {other:?}"),
        }
    }
    resolve_top_until_choice(&mut state);
    assert_eq!(state.initiative, Some(PlayerId::P0));
    assert_ne!(state.engine.initiative_source, Some(original_source));
    let venture = stack_binding(&state).expect("same-holder take fires TOOK_INITIATIVE");
    assert_eq!(venture.kind, InitiativeTriggerKindV1::VentureAfterTaking);
    assert_eq!(venture.source.source, second_hunter);
}

#[test]
fn secret_entrance_is_optional_reveals_only_the_basic_and_always_shuffles() {
    let mut base = ready_state(0x5345_4352_4554_0001);
    let (_, source) = install_initiative_source(&mut base, PlayerId::P0);
    let basic = put_object(&mut base, PlayerId::P0, "Island", Zone::Library);
    let nonbasic = put_object(&mut base, PlayerId::P0, "Azorius Guildgate", Zone::Library);
    let other = put_object(&mut base, PlayerId::P0, "Mountain", Zone::Library);
    enqueue_initiative(
        &mut base,
        PlayerId::P0,
        source,
        InitiativeTriggerKindV1::UndercityRoom(UndercityRoomV1::SecretEntrance),
    );
    advance_to_initiative_stack(
        &mut base,
        InitiativeTriggerKindV1::UndercityRoom(UndercityRoomV1::SecretEntrance),
    );
    let decision = resolve_top_until_choice(&mut base).unwrap();
    let legal = match decision {
        Decision::ChooseEffectTargets {
            legal_targets,
            can_finish: true,
            min_targets: 0,
            max_targets: 1,
            ..
        } => legal_targets,
        other => panic!("expected optional Secret Entrance search, got {other:?}"),
    };
    assert!(legal.contains(&Target::Object(basic)));
    assert!(legal.contains(&Target::Object(other)));
    assert!(!legal.contains(&Target::Object(nonbasic)));

    let mut failed_to_find = base.clone();
    let before_members = failed_to_find.players[0].library.clone();
    engine::step(&mut failed_to_find, Action::FinishEffectSelection).unwrap();
    let _ = engine::advance_until_decision(&mut failed_to_find);
    let mut after_members = failed_to_find.players[0].library.clone();
    let mut expected_members = before_members;
    after_members.sort_unstable();
    expected_members.sort_unstable();
    assert_eq!(after_members, expected_members);
    assert!(failed_to_find.players[0].hand.is_empty());

    engine::step(&mut base, Action::ChooseEffectTarget(Target::Object(basic))).unwrap();
    let _ = engine::advance_until_decision(&mut base);
    assert_eq!(base.objects.get(basic).zone, Zone::Hand);
    assert!(base.players[0].hand.contains(&basic));
    assert!(
        base.hand_knowledge[PlayerId::P1.index()][PlayerId::P0.index()]
            .iter()
            .any(|entry| entry.object == basic)
    );
    assert_eq!(base.players[0].library.len(), 2);
}

#[test]
fn forge_trap_and_arena_have_exact_targets_effects_and_goad_lifetime() {
    let mut forge = ready_state(0x464f_5247_4500_0001);
    let (_, forge_source) = install_initiative_source(&mut forge, PlayerId::P0);
    let own_creature = put_object(
        &mut forge,
        PlayerId::P0,
        "Voldaren Epicure",
        Zone::Battlefield,
    );
    let opposing_creature = put_object(
        &mut forge,
        PlayerId::P1,
        "Voldaren Epicure",
        Zone::Battlefield,
    );
    let land = put_object(&mut forge, PlayerId::P1, "Island", Zone::Battlefield);
    enqueue_initiative(
        &mut forge,
        PlayerId::P0,
        forge_source,
        InitiativeTriggerKindV1::UndercityRoom(UndercityRoomV1::Forge),
    );
    let decision = engine::advance_until_decision(&mut forge);
    assert!(matches!(
        decision,
        Decision::ChooseTargets {
            player: PlayerId::P0,
            ref legal_targets,
            can_finish: false,
            ..
        } if legal_targets.contains(&Target::Object(own_creature))
            && legal_targets.contains(&Target::Object(opposing_creature))
            && !legal_targets.contains(&Target::Object(land))
            && !legal_targets.iter().any(|target| matches!(target, Target::Player(_)))
    ));
    engine::step(
        &mut forge,
        Action::ChooseTarget(Target::Object(opposing_creature)),
    )
    .unwrap();
    advance_to_initiative_stack(
        &mut forge,
        InitiativeTriggerKindV1::UndercityRoom(UndercityRoomV1::Forge),
    );
    resolve_top_until_choice(&mut forge);
    assert_eq!(forge.objects.get(opposing_creature).counters.plus1_plus1, 2);

    let mut trap = ready_state(0x5452_4150_0000_0001);
    let (_, trap_source) = install_initiative_source(&mut trap, PlayerId::P0);
    enqueue_initiative(
        &mut trap,
        PlayerId::P0,
        trap_source,
        InitiativeTriggerKindV1::UndercityRoom(UndercityRoomV1::Trap),
    );
    let decision = engine::advance_until_decision(&mut trap);
    assert!(matches!(
        decision,
        Decision::ChooseTargets {
            ref legal_targets,
            can_finish: false,
            ..
        } if legal_targets == &vec![
            Target::Player(PlayerId::P0),
            Target::Player(PlayerId::P1),
        ]
    ));
    engine::step(
        &mut trap,
        Action::ChooseTarget(Target::Player(PlayerId::P1)),
    )
    .unwrap();
    advance_to_initiative_stack(
        &mut trap,
        InitiativeTriggerKindV1::UndercityRoom(UndercityRoomV1::Trap),
    );
    resolve_top_until_choice(&mut trap);
    assert_eq!(trap.players[1].life, 15);

    let mut arena = ready_state(0x4152_454e_4100_0001);
    let (_, arena_source) = install_initiative_source(&mut arena, PlayerId::P0);
    let goaded = put_object(
        &mut arena,
        PlayerId::P1,
        "Voldaren Epicure",
        Zone::Battlefield,
    );
    enqueue_initiative(
        &mut arena,
        PlayerId::P0,
        arena_source,
        InitiativeTriggerKindV1::UndercityRoom(UndercityRoomV1::Arena),
    );
    assert!(matches!(
        engine::advance_until_decision(&mut arena),
        Decision::ChooseTargets { ref legal_targets, .. }
            if legal_targets.contains(&Target::Object(goaded))
    ));
    engine::step(&mut arena, Action::ChooseTarget(Target::Object(goaded))).unwrap();
    advance_to_initiative_stack(
        &mut arena,
        InitiativeTriggerKindV1::UndercityRoom(UndercityRoomV1::Arena),
    );
    resolve_top_until_choice(&mut arena);
    let [goad] = arena.objects.get(goaded).v4.goaded_by.as_slice() else {
        panic!("Arena installs one goad record")
    };
    assert_eq!(goad.player, PlayerId::P0);
    let goad_expiry = goad.expires_at_turn;

    arena.active_player = PlayerId::P1;
    arena.priority_player = PlayerId::P1;
    arena.step = Step::DeclareAttackers;
    arena.engine.combat = Default::default();
    assert!(matches!(
        engine::advance_until_decision(&mut arena),
        Decision::DeclareAttackers { ref eligible, .. } if eligible.contains(&goaded)
    ));
    assert!(engine::step(&mut arena, Action::DeclareAttackers(Vec::new())).is_err());
    engine::step(&mut arena, Action::DeclareAttackers(vec![goaded])).unwrap();

    let mut expired = arena.clone();
    expired.turn = goad_expiry.saturating_add(1);
    expired.active_player = PlayerId::P1;
    expired.priority_player = PlayerId::P1;
    expired.step = Step::DeclareAttackers;
    expired.engine.combat = Default::default();
    assert!(matches!(
        engine::advance_until_decision(&mut expired),
        Decision::DeclareAttackers { .. }
    ));
    engine::step(&mut expired, Action::DeclareAttackers(Vec::new())).unwrap();
}

#[test]
fn lost_well_stash_archives_and_catacombs_have_exact_room_effects() {
    let mut lost = ready_state(0x4c4f_5354_5745_4c4c);
    let (_, lost_source) = install_initiative_source(&mut lost, PlayerId::P0);
    let top = put_object(&mut lost, PlayerId::P0, "Island", Zone::Library);
    let second = put_object(&mut lost, PlayerId::P0, "Mountain", Zone::Library);
    enqueue_initiative(
        &mut lost,
        PlayerId::P0,
        lost_source,
        InitiativeTriggerKindV1::UndercityRoom(UndercityRoomV1::LostWell),
    );
    advance_to_initiative_stack(
        &mut lost,
        InitiativeTriggerKindV1::UndercityRoom(UndercityRoomV1::LostWell),
    );
    assert!(matches!(
        resolve_top_until_choice(&mut lost),
        Some(Decision::ChooseEffectTargets {
            player: PlayerId::P0,
            ref legal_targets,
            min_targets: 0,
            max_targets: 2,
            can_finish: true,
            ..
        }) if legal_targets == &vec![Target::Object(top), Target::Object(second)]
    ));

    let mut stash = ready_state(0x5354_4153_4800_0001);
    let (_, stash_source) = install_initiative_source(&mut stash, PlayerId::P0);
    enqueue_initiative(
        &mut stash,
        PlayerId::P0,
        stash_source,
        InitiativeTriggerKindV1::UndercityRoom(UndercityRoomV1::Stash),
    );
    advance_to_initiative_stack(
        &mut stash,
        InitiativeTriggerKindV1::UndercityRoom(UndercityRoomV1::Stash),
    );
    resolve_top_until_choice(&mut stash);
    let treasures = stash.players[0]
        .battlefield
        .iter()
        .filter(|&&object| {
            CARD_DEFS[stash.objects.get(object).card_def as usize].name == "Treasure Token"
        })
        .count();
    assert_eq!(treasures, 1);

    let mut archives = ready_state(0x4152_4348_4956_4553);
    let (_, archives_source) = install_initiative_source(&mut archives, PlayerId::P0);
    let draw = put_object(&mut archives, PlayerId::P0, "Island", Zone::Library);
    enqueue_initiative(
        &mut archives,
        PlayerId::P0,
        archives_source,
        InitiativeTriggerKindV1::UndercityRoom(UndercityRoomV1::Archives),
    );
    advance_to_initiative_stack(
        &mut archives,
        InitiativeTriggerKindV1::UndercityRoom(UndercityRoomV1::Archives),
    );
    resolve_top_until_choice(&mut archives);
    assert_eq!(archives.objects.get(draw).zone, Zone::Hand);

    let mut catacombs = ready_state(0x4341_5441_434f_4d42);
    let (_, catacombs_source) = install_initiative_source(&mut catacombs, PlayerId::P0);
    enqueue_initiative(
        &mut catacombs,
        PlayerId::P0,
        catacombs_source,
        InitiativeTriggerKindV1::UndercityRoom(UndercityRoomV1::Catacombs),
    );
    advance_to_initiative_stack(
        &mut catacombs,
        InitiativeTriggerKindV1::UndercityRoom(UndercityRoomV1::Catacombs),
    );
    resolve_top_until_choice(&mut catacombs);
    let skeleton = catacombs.players[0]
        .battlefield
        .iter()
        .copied()
        .find(|&object| {
            CARD_DEFS[catacombs.objects.get(object).card_def as usize].name == "Skeleton Token"
        })
        .expect("Catacombs creates Skeleton Token");
    let definition = &CARD_DEFS[catacombs.objects.get(skeleton).card_def as usize];
    assert!(definition.is_token);
    assert_eq!(definition.types, &[CardType::Creature]);
    assert_eq!(definition.subtypes, &[Subtype::Skeleton]);
    assert_eq!((definition.power, definition.toughness), (Some(4), Some(1)));
    assert!(definition.keywords.has(Keywords::MENACE));
    assert_eq!(definition.colors, &[mtg_kernel::mana::ManaColor::B]);
}

#[test]
fn throne_publicly_reveals_top_ten_chooses_creature_shuffles_and_expires_hexproof() {
    let mut state = ready_state(0x5448_524f_4e45_0001);
    let (_, source) = install_initiative_source(&mut state, PlayerId::P0);
    let mut library = Vec::new();
    for name in [
        "Island",
        "Tolarian Terror",
        "Mountain",
        "Island",
        "Mountain",
        "Island",
        "Mountain",
        "Cryptic Serpent",
        "Island",
        "Mountain",
        "Tolarian Terror",
        "Island",
    ] {
        library.push(put_object(&mut state, PlayerId::P0, name, Zone::Library));
    }
    let first_choice = library[1];
    let second_choice = library[7];
    let outside_top_ten = library[10];
    enqueue_initiative(
        &mut state,
        PlayerId::P0,
        source,
        InitiativeTriggerKindV1::UndercityRoom(UndercityRoomV1::ThroneOfTheDeadThree),
    );
    advance_to_initiative_stack(
        &mut state,
        InitiativeTriggerKindV1::UndercityRoom(UndercityRoomV1::ThroneOfTheDeadThree),
    );
    let decision = resolve_top_until_choice(&mut state).expect("Throne public creature choice");
    let legal = match &decision {
        Decision::ChooseEffectTargets {
            player: PlayerId::P0,
            legal_targets,
            min_targets: 1,
            max_targets: 1,
            can_finish: false,
            ..
        } => legal_targets,
        other => panic!("expected Throne exact-one selection, got {other:?}"),
    };
    assert_eq!(
        legal,
        &vec![Target::Object(first_choice), Target::Object(second_choice)]
    );
    assert!(!legal.contains(&Target::Object(outside_top_ten)));
    for observer in [PlayerId::P0, PlayerId::P1] {
        let known = state.known_library_cards(observer, PlayerId::P0);
        assert_eq!(known.len(), 10);
        assert_eq!(
            known.iter().map(|entry| entry.object).collect::<Vec<_>>(),
            library[..10]
        );
    }

    let bytes = serde_json::to_vec(&state).unwrap();
    let mut restored_choice: GameState = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(state, restored_choice);
    assert_eq!(
        engine::advance_until_decision(&mut state),
        engine::advance_until_decision(&mut restored_choice)
    );

    let mut bad_choice = restored_choice.clone();
    let Some(PendingEffectChoice::SelectTargets {
        purpose:
            EffectTargetSelectionPurpose::UndercityThroneCreature {
                original_library, ..
            },
        ..
    }) = bad_choice
        .engine
        .pending_effect
        .as_mut()
        .and_then(|pending| pending.choice.as_mut())
    else {
        panic!("typed Throne selection continuation")
    };
    original_library.swap(0, 1);
    assert_eventually_invalid_continuation(&mut bad_choice);

    engine::step(
        &mut restored_choice,
        Action::ChooseEffectTarget(Target::Object(second_choice)),
    )
    .unwrap();
    let answered_bytes = serde_json::to_vec(&restored_choice).unwrap();
    let mut answered_round_trip: GameState = serde_json::from_slice(&answered_bytes).unwrap();
    let mut bad_answered = answered_round_trip.clone();
    bad_answered.players[0].library.swap(0, 1);
    assert_eventually_invalid_continuation(&mut bad_answered);

    let next_a = engine::advance_until_decision(&mut restored_choice);
    let next_b = engine::advance_until_decision(&mut answered_round_trip);
    assert_eq!(next_a, next_b);
    assert_eq!(restored_choice, answered_round_trip);
    assert_eq!(
        restored_choice.objects.get(second_choice).zone,
        Zone::Battlefield
    );
    assert_eq!(
        restored_choice
            .objects
            .get(second_choice)
            .counters
            .plus1_plus1,
        3
    );
    assert!(engine::has_effective_keyword(
        &restored_choice,
        second_choice,
        Keywords::HEXPROOF
    ));
    assert_eq!(restored_choice.players[0].library.len(), 11);
    assert!(restored_choice.players[0]
        .library
        .contains(&outside_top_ten));
    for observer in [PlayerId::P0, PlayerId::P1] {
        assert!(restored_choice
            .known_library_cards(observer, PlayerId::P0)
            .is_empty());
    }

    restored_choice.active_player = PlayerId::P1;
    restored_choice.priority_player = PlayerId::P1;
    restored_choice.step = Step::Cleanup;
    restored_choice.engine.priority_passes = [true, true];
    for _ in 0..8 {
        let _ = engine::advance_until_decision(&mut restored_choice);
        if restored_choice.active_player == PlayerId::P0 {
            break;
        }
    }
    assert_eq!(restored_choice.active_player, PlayerId::P0);
    assert!(!engine::has_effective_keyword(
        &restored_choice,
        second_choice,
        Keywords::HEXPROOF
    ));
}

#[test]
fn initiative_history_and_route_continuations_round_trip_and_fail_closed_on_tamper() {
    let mut state = ready_state(0x4249_4e44_494e_4701);
    let (_, source) = install_initiative_source(&mut state, PlayerId::P0);
    set_room(
        &mut state,
        PlayerId::P0,
        Some(UndercityRoomV1::SecretEntrance),
    );
    let binding = enqueue_initiative(
        &mut state,
        PlayerId::P0,
        source,
        InitiativeTriggerKindV1::VentureAfterTaking,
    );
    advance_to_initiative_stack(&mut state, InitiativeTriggerKindV1::VentureAfterTaking);

    let bytes = serde_json::to_vec(&state).unwrap();
    let mut restored: GameState = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(state, restored);
    assert_eq!(
        engine::advance_until_decision(&mut state),
        engine::advance_until_decision(&mut restored)
    );

    let mut bad_history = restored.clone();
    let CommittedEvent::InitiativeTrigger { binding: marker } =
        &mut bad_history.engine.event_history[binding.history_index as usize]
    else {
        panic!("bound Initiative marker")
    };
    marker.player = PlayerId::P1;
    assert_eventually_invalid_continuation(&mut bad_history);

    resolve_top_until_choice(&mut restored);
    assert!(matches!(
        engine::advance_until_decision(&mut restored),
        Decision::ChooseEffectOption {
            option_count: 2,
            ..
        }
    ));
    let bytes = serde_json::to_vec(&restored).unwrap();
    let mut route_round_trip: GameState = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        engine::advance_until_decision(&mut restored),
        engine::advance_until_decision(&mut route_round_trip)
    );
    set_room(
        &mut route_round_trip,
        PlayerId::P0,
        Some(UndercityRoomV1::Forge),
    );
    assert_eventually_invalid_continuation(&mut route_round_trip);
}

#[test]
fn apnap_orders_simultaneous_global_designation_triggers_by_active_player() {
    let mut state = ready_state(0x4150_4e41_5000_0001);
    let (_, source) = install_initiative_source(&mut state, PlayerId::P0);
    event::log_initiative_trigger(
        &mut state,
        PlayerId::P1,
        source,
        InitiativeTriggerKindV1::VentureAfterTaking,
    )
    .unwrap();
    event::log_initiative_trigger(
        &mut state,
        PlayerId::P0,
        source,
        InitiativeTriggerKindV1::VentureAtUpkeep,
    )
    .unwrap();
    collect_triggers(&mut state);
    assert_eq!(
        state
            .engine
            .pending_triggers
            .iter()
            .map(|trigger| trigger.controller)
            .collect::<Vec<_>>(),
        vec![PlayerId::P0, PlayerId::P1]
    );
}
