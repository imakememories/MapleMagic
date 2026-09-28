//! Focused coverage for the green utility tranche.
//!
//! Card text is mapped from the checked-in Mage sources: Pulse of Murasa
//! returns a creature or land card from either graveyard and gains 6 life;
//! Gatecreeper Vine may find a basic land or Gate; Healer of the Glade and
//! Spinewoods Paladin gain 3 life on entry; the Paladin also has trample and
//! Plot {3}{G}.

use mtg_kernel::card_def::{
    card_id_by_name, CardCapability, CardType, Keywords, Subtype, TargetSpec, CARD_DEFS,
};
use mtg_kernel::effect::{EffectOp, LibraryCardFilter, ObjectRef, PlayerRef};
use mtg_kernel::engine::{self, Action, Decision};
use mtg_kernel::event::{self, ProposedEvent};
use mtg_kernel::ids::{ObjectId, PlayerId};
use mtg_kernel::mana::{Cost, ManaColor, Pip};
use mtg_kernel::state::{Counters, GameObject, GameState, ObjectStateV4, Step, Target, Zone};
use mtg_kernel::trigger;

fn card_id(name: &str) -> u16 {
    card_id_by_name(name).unwrap_or_else(|| panic!("{name} in CARD_DEFS"))
}

fn put_object(state: &mut GameState, player: PlayerId, name: &str, zone: Zone) -> ObjectId {
    let card_def = card_id(name);
    let object = state.objects.push(GameObject {
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
        v4: ObjectStateV4::from_card_def(card_def),
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
        Zone::Stack => panic!("test helper does not construct stack objects"),
    }
    object
}

fn ready_main(seed: u64) -> GameState {
    let mut state = GameState::new_from_libraries(&[], &[], |_| String::new(), seed);
    state.step = Step::Main1;
    state.active_player = PlayerId::P0;
    state.priority_player = PlayerId::P0;
    state
}

fn add_green(state: &mut GameState, amount: u8) {
    state.players[PlayerId::P0.index()].mana_pool[ManaColor::G.pool_index()] += amount;
}

fn pass_until<F>(state: &mut GameState, mut done: F) -> Decision
where
    F: FnMut(&GameState, &Decision) -> bool,
{
    for _ in 0..64 {
        let decision = engine::advance_until_decision(state);
        if done(state, &decision) {
            return decision;
        }
        match decision {
            Decision::CastSpellOrPass { .. } => engine::step(state, Action::Pass).unwrap(),
            other => panic!("unexpected decision during bounded resolution: {other:?}"),
        }
    }
    panic!("bounded resolution did not reach the expected state");
}

#[test]
fn generated_definitions_match_the_four_mage_cards() {
    assert_eq!(card_id("Gatecreeper Vine"), 145);
    assert_eq!(card_id("Healer of the Glade"), 146);
    assert_eq!(CARD_DEFS.len(), 162);
    assert_eq!(Subtype::Elemental.stable_id(), 60);

    let pulse = &CARD_DEFS[card_id("Pulse of Murasa") as usize];
    assert_eq!(pulse.capability, CardCapability::Full);
    assert_eq!(pulse.target_spec, TargetSpec::CreatureOrLandCardInGraveyard);
    assert_eq!(
        (pulse.spell_effect)(),
        Some(EffectOp::Sequence(vec![
            EffectOp::MoveObject {
                object: ObjectRef::Target(0),
                to_zone: Zone::Hand,
            },
            EffectOp::GainLife {
                player: PlayerRef::Controller,
                amount: 6,
            },
        ]))
    );

    let gatecreeper = &CARD_DEFS[card_id("Gatecreeper Vine") as usize];
    assert_eq!(gatecreeper.types, &[CardType::Creature]);
    assert_eq!(gatecreeper.subtypes, &[Subtype::Plant]);
    assert_eq!(
        (gatecreeper.power, gatecreeper.toughness),
        (Some(0), Some(2))
    );
    assert!(gatecreeper.keywords.has(Keywords::DEFENDER));
    let gate_trigger = &trigger::triggers_for(card_id("Gatecreeper Vine"))[0];
    assert_eq!(
        (gate_trigger.effect)(),
        EffectOp::SearchLibraryToHand {
            player: PlayerRef::Controller,
            filter: LibraryCardFilter::BasicLandOrGate,
        }
    );

    let healer = &CARD_DEFS[card_id("Healer of the Glade") as usize];
    assert_eq!(healer.subtypes, &[Subtype::Elemental]);
    assert_eq!((healer.power, healer.toughness), (Some(1), Some(2)));

    let paladin = &CARD_DEFS[card_id("Spinewoods Paladin") as usize];
    assert_eq!((paladin.power, paladin.toughness), (Some(5), Some(4)));
    assert!(paladin.keywords.has(Keywords::TRAMPLE));
    assert_eq!(
        paladin.plot_cost,
        Some(Cost {
            pips: &[Pip::Colored(ManaColor::G)],
            generic: 3,
            x_count: 0,
        })
    );
    assert_eq!(
        paladin.cost,
        Cost {
            pips: &[Pip::Colored(ManaColor::G)],
            generic: 4,
            x_count: 0,
        }
    );
}

#[test]
fn pulse_targets_only_creature_or_land_cards_in_either_graveyard() {
    let mut state = ready_main(0x5055_4c53_4500_0001);
    add_green(&mut state, 3);
    let pulse = put_object(&mut state, PlayerId::P0, "Pulse of Murasa", Zone::Hand);
    let own_creature = put_object(&mut state, PlayerId::P0, "Elvish Mystic", Zone::Graveyard);
    let opposing_land = put_object(&mut state, PlayerId::P1, "Mountain", Zone::Graveyard);
    let excluded_instant = put_object(&mut state, PlayerId::P1, "Lightning Bolt", Zone::Graveyard);

    engine::step(&mut state, Action::CastSpell(pulse)).unwrap();
    let Decision::ChooseTargets { legal_targets, .. } = engine::advance_until_decision(&mut state)
    else {
        panic!("Pulse must choose a graveyard target");
    };
    assert!(legal_targets.contains(&Target::Object(own_creature)));
    assert!(legal_targets.contains(&Target::Object(opposing_land)));
    assert!(!legal_targets.contains(&Target::Object(excluded_instant)));

    engine::step(
        &mut state,
        Action::ChooseTarget(Target::Object(opposing_land)),
    )
    .unwrap();
    pass_until(&mut state, |state, _| {
        state.objects.get(opposing_land).zone == Zone::Hand && state.players[0].life == 26
    });
    assert!(state.players[1].hand.contains(&opposing_land));
    assert_eq!(state.objects.get(opposing_land).owner, PlayerId::P1);
    assert_eq!(state.objects.get(pulse).zone, Zone::Graveyard);
}

#[test]
fn pulse_fizzles_entirely_when_its_only_target_changes_incarnation() {
    let mut state = ready_main(0x5055_4c53_4500_0002);
    add_green(&mut state, 3);
    let pulse = put_object(&mut state, PlayerId::P0, "Pulse of Murasa", Zone::Hand);
    let target = put_object(&mut state, PlayerId::P0, "Elvish Mystic", Zone::Graveyard);
    engine::step(&mut state, Action::CastSpell(pulse)).unwrap();
    assert!(matches!(
        engine::advance_until_decision(&mut state),
        Decision::ChooseTargets { .. }
    ));
    engine::step(&mut state, Action::ChooseTarget(Target::Object(target))).unwrap();
    assert!(matches!(
        engine::advance_until_decision(&mut state),
        Decision::CastSpellOrPass { .. }
    ));

    event::propose_and_commit(&mut state, ProposedEvent::zone_change(target, Zone::Exile));
    pass_until(&mut state, |state, _| {
        state.objects.get(pulse).zone == Zone::Graveyard && state.stack.is_empty()
    });
    assert_eq!(state.players[0].life, 20);
    assert_eq!(state.objects.get(target).zone, Zone::Exile);
}

#[test]
fn gatecreeper_etb_searches_basic_lands_or_gates_and_reveals_the_choice() {
    let mut state = ready_main(0x4741_5445_0000_0001);
    let basic = put_object(&mut state, PlayerId::P0, "Forest", Zone::Library);
    let gate = put_object(&mut state, PlayerId::P0, "Azorius Guildgate", Zone::Library);
    let nonbasic_forest = put_object(&mut state, PlayerId::P0, "Gingerbread Cabin", Zone::Library);
    let nonland = put_object(&mut state, PlayerId::P0, "Lightning Bolt", Zone::Library);
    let vine = put_object(&mut state, PlayerId::P0, "Gatecreeper Vine", Zone::Hand);
    add_green(&mut state, 2);

    engine::step(&mut state, Action::CastSpell(vine)).unwrap();
    let decision = pass_until(&mut state, |_, decision| {
        matches!(decision, Decision::ChooseEffectTargets { .. })
    });
    let Decision::ChooseEffectTargets { legal_targets, .. } = decision else {
        unreachable!()
    };
    assert!(legal_targets.contains(&Target::Object(basic)));
    assert!(legal_targets.contains(&Target::Object(gate)));
    assert!(!legal_targets.contains(&Target::Object(nonbasic_forest)));
    assert!(!legal_targets.contains(&Target::Object(nonland)));

    engine::step(&mut state, Action::ChooseEffectTarget(Target::Object(gate))).unwrap();
    assert!(matches!(
        engine::advance_until_decision(&mut state),
        Decision::CastSpellOrPass { .. }
    ));
    assert_eq!(state.objects.get(gate).zone, Zone::Hand);
    assert!(state.players[0].hand.contains(&gate));
    assert!(state
        .known_hand_cards(PlayerId::P1, PlayerId::P0)
        .iter()
        .any(|known| known.object == gate));
    assert_eq!(state.objects.get(vine).zone, Zone::Battlefield);
}

#[test]
fn healer_and_paladin_gain_three_life_when_their_entry_triggers_resolve() {
    for (case, name, mana) in [
        (0_u64, "Healer of the Glade", 1_u8),
        (1_u64, "Spinewoods Paladin", 5_u8),
    ] {
        let mut state = ready_main(0x4c49_4645_0000_0000 + case);
        add_green(&mut state, mana);
        let creature = put_object(&mut state, PlayerId::P0, name, Zone::Hand);
        engine::step(&mut state, Action::CastSpell(creature)).unwrap();
        pass_until(&mut state, |state, _| {
            state.objects.get(creature).zone == Zone::Battlefield && state.players[0].life == 23
        });
        assert_eq!(
            state.objects.get(creature).zone,
            Zone::Battlefield,
            "{name}"
        );
        assert_eq!(state.players[0].life, 23, "{name}");
    }
}
