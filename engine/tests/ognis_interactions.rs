//! Interactions between the Ognis deck's cards (audit: docs/audit/ognis/patterns.md).

mod common;
use common::*;
use mtg_kernel::engine::{self, Action, Decision};
use mtg_kernel::mana::ManaColor;
use mtg_kernel::ids::ObjectId;
use mtg_kernel::state::{GameState, Step, Target, Zone};

#[test]
fn shivan_devastators_x_counters_add_to_coin_of_masterys() {
    // Both are "enters with" counters (614.1c): {X}{R} with X = 2, paid by Sol Ring and a Mountain.
    let mut state = game();
    put(&mut state, P0, "Coin of Mastery");
    put(&mut state, P0, "Sol Ring");
    lands(&mut state, P0, 1);
    let dragon = cast(&mut state, "Shivan Devastator", None, 2);
    assert_eq!(state.objects.get(dragon).counters.plus1_plus1, 2 + 2);

    // X = 0 paid with a Treasure's red: Coin's one counter keeps it alive.
    let mut state = game();
    put(&mut state, P0, "Coin of Mastery");
    let treasure = put(&mut state, P0, "Treasure Token");
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    engine::step(&mut state, Action::ActivateManaAbilityChoice(treasure, ManaColor::R)).unwrap();
    let dragon = cast(&mut state, "Shivan Devastator", None, 0);
    assert_eq!(zone(&state, dragon), Zone::Battlefield);
    assert_eq!(state.objects.get(dragon).counters.plus1_plus1, 1);
}

#[test]
fn whirlwind_counts_only_creatures_that_entered_during_this_turn_not_this_round() {
    use mtg_kernel::event::{propose_and_commit, ProposedEvent};
    // Player 1's creature enters during player 0's turn of the same round,
    // then attacks on player 1's turn: it didn't enter this turn.
    let mut state = game();
    put(&mut state, P1, "Whirlwind, Killer Cyclone");
    let veteran = put_in(&mut state, P1, "Captain Lannery Storm", Zone::Hand);
    propose_and_commit(&mut state, ProposedEvent::zone_change(veteran, Zone::Battlefield));
    let wall = put(&mut state, P0, "Zealous Conscripts");
    end_turn(&mut state);
    assert_eq!(state.active_player, P1);
    advance(&mut state, |_, d| matches!(d, Decision::DeclareAttackers { .. }));
    engine::step(&mut state, Action::DeclareAttackers(vec![veteran])).unwrap();
    let blocks = advance(&mut state, |_, d| matches!(d, Decision::DeclareBlockers { .. }));
    let Decision::DeclareBlockers { legal_blockers, .. } = blocks else { unreachable!() };
    let of_veteran = legal_blockers.iter().find(|(a, _)| *a == veteran).map(|(_, b)| b.clone()).unwrap_or_default();
    assert!(of_veteran.contains(&wall), "no Whirlwind trigger: {of_veteran:?}");
}

/// Player 0 attacks with `attackers` and player 1 blocks with `blocks`; each
/// damage assignment takes the next of `amounts`, and any trigger targets
/// player 0. Returns each assignment's recipient and options.
fn fight(
    state: &mut GameState,
    attackers: Vec<ObjectId>,
    blocks: Vec<(ObjectId, ObjectId)>,
    amounts: &[u16],
) -> Vec<(Target, Vec<u16>)> {
    let mut asked = Vec::new();
    let mut amounts = amounts.iter();
    let (mut attackers, mut blocks) = (Some(attackers), Some(blocks));
    loop {
        let a = match engine::advance_until_decision(state) {
            Decision::DeclareAttackers { .. } => Action::DeclareAttackers(attackers.take().expect("one combat")),
            Decision::DeclareBlockers { .. } => Action::DeclareBlockers(blocks.take().expect("one combat")),
            Decision::AssignCombatDamage { recipient, options, .. } => {
                asked.push((recipient, options));
                Action::AssignCombatDamage(*amounts.next().expect("an amount for each assignment"))
            }
            Decision::ChooseTargets { legal_targets, .. } => {
                let me = Target::Player(P0);
                Action::ChooseTarget(if legal_targets.contains(&me) { me } else { legal_targets[0] })
            }
            Decision::OrderTriggers { pending, .. } => Action::OrderTriggers((0..pending.len()).collect()),
            Decision::CastSpellOrPass { .. } if attackers.is_none() && matches!(state.step, Step::EndCombat | Step::Main2) => {
                return asked
            }
            Decision::CastSpellOrPass { .. } => Action::Pass,
            other => panic!("unhandled decision {other:?}"),
        };
        engine::step(state, a).unwrap();
    }
}

#[test]
fn trample_assigns_lethal_to_the_blocker_and_the_rest_as_its_controller_chooses() {
    // 702.19b: Cavern-Hoard Dragon (6/6) blocked by Hellkite Charger (5/5).
    for to_player in [1u16, 0] {
        let mut state = game();
        let dragon = put(&mut state, P0, "Cavern-Hoard Dragon");
        let hellkite = put(&mut state, P1, "Hellkite Charger");
        let asked = fight(&mut state, vec![dragon], vec![(hellkite, dragon)], &[to_player]);
        assert_eq!(asked, vec![(Target::Player(P1), vec![0, 1])], "the blocker's lethal 5 isn't a choice");
        assert_eq!(state.players[1].life, 40 - i32::from(to_player));
        assert_eq!(zone(&state, hellkite), Zone::Graveyard);
        assert_eq!(zone(&state, dragon), Zone::Battlefield);
    }
}

#[test]
fn a_double_blocked_attacker_divides_its_damage_as_its_controller_chooses() {
    // 510.1c: Molten Primordial (6/4) blocked by two Zealous Conscripts (3/3).
    let mut state = game();
    let primordial = put(&mut state, P0, "Molten Primordial");
    let (first, second) = (put(&mut state, P1, "Zealous Conscripts"), put(&mut state, P1, "Zealous Conscripts"));
    let asked = fight(&mut state, vec![primordial], vec![(first, primordial), (second, primordial)], &[1]);
    // Past lethal is the same outcome, so the first blocker's options stop at 3; the second gets the rest.
    assert_eq!(asked, vec![(Target::Object(first), vec![0, 1, 2, 3])]);
    assert_eq!((zone(&state, first), state.objects.get(first).damage), (Zone::Battlefield, 1));
    assert_eq!(zone(&state, second), Zone::Graveyard);
}

#[test]
fn damage_past_lethal_still_matters_to_a_blocking_screaming_nemesis() {
    // Every amount dealt to Nemesis is dealt again by its trigger.
    let mut state = game();
    let primordial = put(&mut state, P0, "Molten Primordial");
    let nemesis = put(&mut state, P1, "Screaming Nemesis");
    let conscripts = put(&mut state, P1, "Zealous Conscripts");
    let asked = fight(&mut state, vec![primordial], vec![(nemesis, primordial), (conscripts, primordial)], &[5]);
    assert_eq!(asked, vec![(Target::Object(nemesis), (0..=6).collect())], "the Conscripts' share is forced: 1");
    assert_eq!(state.objects.get(conscripts).damage, 1);
    assert_eq!(state.players[0].life, 40 - 5, "Nemesis dealt the 5 it was dealt");
}

#[test]
fn a_double_strikers_first_damage_and_its_trigger_come_before_the_second_damage_step() {
    // 510.4: Knuckles (2/4, double strike) unblocked: a Treasure after the
    // first-strike step, with priority, before the regular damage.
    let mut state = game();
    let knuckles = put(&mut state, P0, "Knuckles the Echidna");
    advance(&mut state, |_, d| matches!(d, Decision::DeclareAttackers { .. }));
    engine::step(&mut state, Action::DeclareAttackers(vec![knuckles])).unwrap();
    let mut between = None;
    while !matches!(state.step, Step::EndCombat | Step::Main2) {
        match engine::advance_until_decision(&mut state) {
            Decision::CastSpellOrPass { .. } => {
                if state.step == Step::CombatDamage && state.stack.is_empty() && between.is_none() {
                    between = Some((state.players[1].life, named(&state, P0, "Treasure Token").len()));
                }
                engine::step(&mut state, Action::Pass).unwrap();
            }
            Decision::DeclareBlockers { .. } => engine::step(&mut state, Action::DeclareBlockers(vec![])).unwrap(),
            other => panic!("unhandled decision {other:?}"),
        }
    }
    assert_eq!(between, Some((40 - 2, 1)));
    assert_eq!((state.players[1].life, named(&state, P0, "Treasure Token").len()), (40 - 4, 2));
}

#[test]
fn a_blocker_that_left_combat_takes_no_share_of_the_damage() {
    // 506.4, 510.1c: one of the two blockers dies before damage; Hellkite
    // Charger (5/5) is now the only blocker and takes all 6, not a 3 share.
    use mtg_kernel::event::{propose_and_commit, ProposedEvent};
    let mut state = game();
    let primordial = put(&mut state, P0, "Molten Primordial");
    let (first, second) = (put(&mut state, P1, "Zealous Conscripts"), put(&mut state, P1, "Hellkite Charger"));
    advance(&mut state, |_, d| matches!(d, Decision::DeclareAttackers { .. }));
    engine::step(&mut state, Action::DeclareAttackers(vec![primordial])).unwrap();
    advance(&mut state, |_, d| matches!(d, Decision::DeclareBlockers { .. }));
    engine::step(&mut state, Action::DeclareBlockers(vec![(first, primordial), (second, primordial)])).unwrap();
    propose_and_commit(&mut state, ProposedEvent::zone_change(first, Zone::Graveyard));
    // `advance` panics on any damage assignment decision.
    advance(&mut state, |s, _| matches!(s.step, Step::EndCombat | Step::Main2));
    assert_eq!(zone(&state, second), Zone::Graveyard);
}

#[test]
fn lethal_damage_from_a_chosen_split_ends_the_game_at_once() {
    let mut state = game();
    state.players[1].life = 1;
    let dragon = put(&mut state, P0, "Cavern-Hoard Dragon");
    let hellkite = put(&mut state, P1, "Hellkite Charger");
    advance(&mut state, |_, d| matches!(d, Decision::DeclareAttackers { .. }));
    engine::step(&mut state, Action::DeclareAttackers(vec![dragon])).unwrap();
    advance(&mut state, |_, d| matches!(d, Decision::DeclareBlockers { .. }));
    engine::step(&mut state, Action::DeclareBlockers(vec![(hellkite, dragon)])).unwrap();
    advance(&mut state, |_, d| matches!(d, Decision::AssignCombatDamage { .. }));
    engine::step(&mut state, Action::AssignCombatDamage(1)).unwrap();
    assert!(matches!(engine::advance_until_decision(&mut state), Decision::GameOver { winner: Some(P0) }));
}
#[test]
fn vision_of_love_doesnt_offer_sacrificing_an_artifact_call_for_aid_forbids() {
    // Depthshaker Titan is an artifact creature: taken by Call for Aid, it
    // can't be sacrificed, so only the discard is offered.
    let mut state = game();
    lands(&mut state, P0, 7);
    put(&mut state, P1, "Depthshaker Titan");
    cast(&mut state, "Call for Aid", Some(Target::Player(P1)), 0);
    put_in(&mut state, P0, "Mountain", Zone::Hand);
    let vision = put_in(&mut state, P0, "Vision of Love", Zone::Hand);
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    engine::step(&mut state, Action::CastSpell(vision)).unwrap();
    let offer = advance(&mut state, |_, d| matches!(d, Decision::ChooseOptionalCost { .. }));
    assert!(matches!(offer, Decision::ChooseOptionalCost { discard_payable: true, sacrifice_payable: false, .. }), "{offer:?}");
}
#[test]
fn call_for_aids_sacrifice_ban_binds_only_its_caster() {
    // "You can't sacrifice those creatures": once its owner has it back (here
    // set directly; in play, Pyreswipe Hawk's expend), the owner may
    // sacrifice it, here as Demand Answers's additional cost.
    let mut state = game();
    lands(&mut state, P0, 5);
    lands(&mut state, P1, 2);
    let titan = put(&mut state, P1, "Depthshaker Titan");
    let answers = put_in(&mut state, P1, "Demand Answers", Zone::Hand);
    cast(&mut state, "Call for Aid", Some(Target::Player(P1)), 0);
    assert_eq!(state.objects.get(titan).controller, P0);
    state.players[0].battlefield.retain(|&o| o != titan);
    state.players[1].battlefield.push(titan);
    state.objects.get_mut(titan).controller = P1;
    match advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { player: P1, .. })) {
        Decision::CastSpellOrPass { castable_spells, .. } => assert!(castable_spells.contains(&answers)),
        _ => unreachable!(),
    }
}
