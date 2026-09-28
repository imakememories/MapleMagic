//! Focused coverage for Piracy Charm's three printed modes. Card text is
//! bounded to the checked-in Mage `PiracyCharm.java` authority.

use mtg_kernel::card_def::{card_id_by_name, Keywords, CARD_DEFS};
use mtg_kernel::engine::{self, Action, Decision};
use mtg_kernel::event::{self, ProposedEvent};
use mtg_kernel::ids::{ObjectId, PlayerId};
use mtg_kernel::mana::ManaColor;
use mtg_kernel::state::{Counters, GameObject, GameState, Step, Target, Zone};

fn card_id(name: &str) -> u16 {
    card_id_by_name(name).unwrap_or_else(|| panic!("{name} in CARD_DEFS"))
}

fn card_name(card_def: u16) -> String {
    CARD_DEFS[card_def as usize].name.to_owned()
}

fn ready_game() -> GameState {
    let mut state = GameState::new_from_libraries(&[], &[], card_name, 0x5049_5241_4359);
    state.active_player = PlayerId::P0;
    state.priority_player = PlayerId::P0;
    state.step = Step::Main1;
    state.players[PlayerId::P0.index()].mana_pool[ManaColor::U.pool_index()] = 1;
    state
}

fn put_object(state: &mut GameState, player: PlayerId, name: &str, zone: Zone) -> ObjectId {
    let card_def = card_id(name);
    let object = state.objects.push(GameObject {
        card_def,
        name: name.to_owned(),
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
        Zone::Hand => state.players[player.index()].hand.push(object),
        Zone::Battlefield => state.players[player.index()].battlefield.push(object),
        Zone::Library => state.players[player.index()].library.push(object),
        Zone::Graveyard => state.players[player.index()].graveyard.push(object),
        Zone::Exile => state.exile.push(object),
        Zone::Command => state.command.push(object),
        Zone::Stack => panic!("casts own stack insertion"),
    }
    object
}

fn begin_mode(state: &mut GameState, charm: ObjectId, mode: u8) -> Decision {
    engine::step(state, Action::CastSpell(charm)).unwrap();
    let decision = engine::advance_until_decision(state);
    assert!(matches!(
        decision,
        Decision::ChooseSpellMode {
            player: PlayerId::P0,
            spell,
            mode_count: 3,
            ..
        } if spell == charm
    ));
    engine::step(state, Action::ChooseSpellMode(mode)).unwrap();
    engine::advance_until_decision(state)
}

fn choose_target(state: &mut GameState, decision: &Decision, target: Target) {
    assert!(matches!(
        decision,
        Decision::ChooseTargets { legal_targets, .. } if legal_targets.contains(&target)
    ));
    engine::step(state, Action::ChooseTarget(target)).unwrap();
}

fn advance_resolution(state: &mut GameState, spell: ObjectId) -> Decision {
    loop {
        let decision = engine::advance_until_decision(state);
        if state.objects.get(spell).zone != Zone::Stack
            || matches!(decision, Decision::Discard { .. })
        {
            return decision;
        }
        match decision {
            Decision::CastSpellOrPass { .. } => engine::step(state, Action::Pass).unwrap(),
            other => panic!("unexpected decision while resolving Piracy Charm: {other:?}"),
        }
    }
}

#[test]
fn mode_zero_grants_islandwalk_to_the_exact_target_incarnation() {
    let mut state = ready_game();
    let creature = put_object(&mut state, PlayerId::P0, "Guttersnipe", Zone::Battlefield);
    let charm = put_object(&mut state, PlayerId::P0, "Piracy Charm", Zone::Hand);

    let targets = begin_mode(&mut state, charm, 0);
    choose_target(&mut state, &targets, Target::Object(creature));
    advance_resolution(&mut state, charm);
    assert!(engine::has_effective_keyword(
        &state,
        creature,
        Keywords::ISLANDWALK
    ));

    event::propose_and_commit(
        &mut state,
        ProposedEvent::zone_change(creature, Zone::Graveyard),
    );
    event::propose_and_commit(
        &mut state,
        ProposedEvent::zone_change(creature, Zone::Battlefield),
    );
    assert!(!engine::has_effective_keyword(
        &state,
        creature,
        Keywords::ISLANDWALK
    ));
}

#[test]
fn mode_one_gives_plus_two_minus_one_until_end_of_turn() {
    let mut state = ready_game();
    let creature = put_object(&mut state, PlayerId::P1, "Guttersnipe", Zone::Battlefield);
    let charm = put_object(&mut state, PlayerId::P0, "Piracy Charm", Zone::Hand);

    let targets = begin_mode(&mut state, charm, 1);
    choose_target(&mut state, &targets, Target::Object(creature));
    advance_resolution(&mut state, charm);
    assert_eq!(engine::effective_power(&state, creature), 4);
    assert_eq!(engine::effective_toughness(&state, creature), 1);
}

#[test]
fn mode_two_has_the_target_player_choose_and_discard_one_card() {
    let mut state = ready_game();
    put_object(&mut state, PlayerId::P1, "Guttersnipe", Zone::Battlefield);
    let discarded = put_object(&mut state, PlayerId::P1, "Ponder", Zone::Hand);
    put_object(&mut state, PlayerId::P1, "Preordain", Zone::Hand);
    let charm = put_object(&mut state, PlayerId::P0, "Piracy Charm", Zone::Hand);

    let targets = begin_mode(&mut state, charm, 2);
    choose_target(&mut state, &targets, Target::Player(PlayerId::P1));
    let discard = advance_resolution(&mut state, charm);
    assert!(matches!(
        discard,
        Decision::Discard {
            player: PlayerId::P1,
            count: 1,
            ..
        }
    ));
    engine::step(&mut state, Action::Discard(vec![discarded])).unwrap();
    engine::advance_until_decision(&mut state);
    assert_eq!(state.objects.get(discarded).zone, Zone::Graveyard);
    assert_eq!(state.objects.get(charm).zone, Zone::Graveyard);
}
