//! Rules points of the Ognis deck's cards that the card tests left open (audit: docs/audit/ognis).

mod common;
use common::*;
use mtg_kernel::card_def::Keywords;
use mtg_kernel::engine::{self, Action, Decision};
use mtg_kernel::ids::ObjectId;
use mtg_kernel::state::{GameState, Target};

/// Every creature card in the deck with its printed flying, haste, trample and
/// double strike, from the oracle text. Goddric's flying needs celebration;
/// Depthshaker Titan is an artifact creature, so its own static gives it haste and trample.
const PRINTED: &[(&str, &[Keywords])] = &[
    ("Ognis, the Dragon's Lash", &[Keywords::HASTE]),
    ("Captain Lannery Storm", &[Keywords::HASTE]),
    ("Cavern-Hoard Dragon", &[Keywords::FLYING, Keywords::HASTE, Keywords::TRAMPLE]),
    ("Depthshaker Titan", &[Keywords::HASTE, Keywords::TRAMPLE]),
    ("Goddric, Cloaked Reveler", &[Keywords::HASTE]),
    ("Gogo, Mysterious Mime", &[]),
    ("Goldspan Dragon", &[Keywords::FLYING, Keywords::HASTE]),
    ("Gornog, the Red Reaper", &[Keywords::HASTE]),
    ("Headliner Scarlett", &[Keywords::HASTE]),
    ("Hellkite Charger", &[Keywords::FLYING, Keywords::HASTE]),
    ("Hideous Taskmaster", &[Keywords::HASTE, Keywords::TRAMPLE]),
    ("Hobgoblin, Mantled Marauder", &[Keywords::FLYING, Keywords::HASTE]),
    ("Jurin, Leading the Charge", &[Keywords::HASTE]),
    ("Knuckles the Echidna", &[Keywords::HASTE, Keywords::TRAMPLE, Keywords::DOUBLE_STRIKE]),
    ("Laelia, the Blade Reforged", &[Keywords::HASTE]),
    ("Molten Primordial", &[Keywords::HASTE]),
    ("Pyreswipe Hawk", &[Keywords::FLYING, Keywords::HASTE]),
    ("Screaming Nemesis", &[Keywords::HASTE]),
    ("Shivan Devastator", &[Keywords::FLYING, Keywords::HASTE]),
    ("Slickshot Show-Off", &[Keywords::FLYING, Keywords::HASTE]),
    ("Smaug the Magnificent", &[Keywords::FLYING, Keywords::HASTE]),
    ("Spider-Man, Web-Spinner", &[Keywords::HASTE, Keywords::DOUBLE_STRIKE]),
    ("Spider-Punk", &[]),
    ("Summon: Esper Valigarmanda", &[Keywords::FLYING, Keywords::HASTE]),
    ("Urabrask, Heretic Praetor", &[Keywords::HASTE]),
    ("Whirlwind, Killer Cyclone", &[Keywords::HASTE]),
    ("Zealous Conscripts", &[Keywords::HASTE]),
];

const CHECKED: [Keywords; 4] = [Keywords::FLYING, Keywords::HASTE, Keywords::TRAMPLE, Keywords::DOUBLE_STRIKE];

fn has(keywords: &[Keywords], k: Keywords) -> bool {
    keywords.contains(&k)
}

/// Puts `card` onto player 0's battlefield; Shivan Devastator gets one +1/+1
/// counter so it isn't a 0/0.
fn creature(state: &mut GameState, card: &str) -> ObjectId {
    let o = put(state, P0, card);
    if card == "Shivan Devastator" {
        state.objects.get_mut(o).counters.plus1_plus1 = 1;
    }
    o
}

/// Answers every decision with the least it can (no payments, the opponent as
/// target) until `stop` accepts one.
fn drive_until(state: &mut GameState, mut stop: impl FnMut(&Decision) -> bool) -> Decision {
    for _ in 0..500 {
        let d = engine::advance_until_decision(state);
        if stop(&d) {
            return d;
        }
        let a = match &d {
            Decision::CastSpellOrPass { .. } => Action::Pass,
            Decision::DeclareAttackers { .. } => Action::DeclareAttackers(vec![]),
            Decision::OrderTriggers { pending, .. } => Action::OrderTriggers((0..pending.len()).collect()),
            Decision::ChooseTargets { legal_targets, .. } => {
                let opponent = Target::Player(P1);
                Action::ChooseTarget(if legal_targets.contains(&opponent) { opponent } else { legal_targets[0] })
            }
            Decision::ChooseEffectTargets { legal_targets, can_finish, .. } => {
                if *can_finish {
                    Action::FinishEffectSelection
                } else {
                    Action::ChooseEffectTarget(legal_targets[0])
                }
            }
            Decision::ChooseEffectBoolean { .. } => Action::ChooseEffectBoolean(false),
            Decision::ChooseEffectOption { .. } => Action::ChooseEffectOption(0),
            other => panic!("unhandled decision {other:?}"),
        };
        engine::step(state, a).unwrap();
    }
    panic!("no stop after 500 decisions");
}

#[test]
fn each_creature_card_has_its_printed_keywords() {
    for &(card, printed) in PRINTED {
        let mut state = game();
        let o = creature(&mut state, card);
        for k in CHECKED {
            assert_eq!(engine::has_effective_keyword(&state, o, k), has(printed, k), "{card}: {k:?}");
        }
    }
}

#[test]
fn each_hasty_creature_can_attack_the_turn_it_arrives() {
    for &(card, printed) in PRINTED {
        let mut state = game();
        let o = creature(&mut state, card);
        state.objects.get_mut(o).summoning_sick = true;
        match drive_until(&mut state, |d| matches!(d, Decision::DeclareAttackers { .. })) {
            Decision::DeclareAttackers { eligible, .. } => {
                assert_eq!(eligible.contains(&o), has(printed, Keywords::HASTE), "{card}")
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn ognis_makes_one_tapped_treasure_for_each_creature_with_haste_that_attacks_the_turn_it_arrives() {
    // Printed haste, or haste from The Fire Crystal for Gogo and Spider-Punk.
    for &(card, printed) in PRINTED {
        let mut state = game();
        let ognis = put(&mut state, P0, "Ognis, the Dragon's Lash");
        let o = if card == "Ognis, the Dragon's Lash" { ognis } else { creature(&mut state, card) };
        if !has(printed, Keywords::HASTE) {
            put(&mut state, P0, "The Fire Crystal");
        }
        state.objects.get_mut(o).summoning_sick = true;
        drive_until(&mut state, |d| matches!(d, Decision::DeclareAttackers { .. }));
        engine::step(&mut state, Action::DeclareAttackers(vec![o])).unwrap();
        drive_until(&mut state, |d| matches!(d, Decision::DeclareBlockers { .. }));
        let tapped: Vec<_> = named(&state, P0, "Treasure Token").into_iter().filter(|&t| state.objects.get(t).tapped).collect();
        assert_eq!(tapped.len(), 1, "{card}");
    }
}

#[test]
fn each_flyer_can_be_blocked_only_by_creatures_with_flying_or_reach() {
    for &(card, printed) in PRINTED {
        let mut state = game();
        let o = creature(&mut state, card);
        // Gornog, Whirlwind and Spider-Man's attack triggers take the first legal target.
        put(&mut state, P1, "Molten Primordial");
        let ground: ObjectId = put(&mut state, P1, "Zealous Conscripts");
        let flyer: ObjectId = put(&mut state, P1, "Hellkite Charger");
        drive_until(&mut state, |d| matches!(d, Decision::DeclareAttackers { .. }));
        engine::step(&mut state, Action::DeclareAttackers(vec![o])).unwrap();
        let blockers = match drive_until(&mut state, |d| matches!(d, Decision::DeclareBlockers { .. })) {
            Decision::DeclareBlockers { legal_blockers, .. } => {
                legal_blockers.into_iter().find(|(a, _)| *a == o).map(|(_, b)| b).unwrap_or_default()
            }
            _ => unreachable!(),
        };
        assert!(blockers.contains(&flyer), "{card}: {blockers:?}");
        assert_eq!(blockers.contains(&ground), !has(printed, Keywords::FLYING), "{card}: {blockers:?}");
    }
}

#[test]
fn a_trigger_from_the_cleanup_discard_gets_priority_and_another_cleanup_ends_its_effect() {
    use mtg_kernel::engine::effective_power;
    use mtg_kernel::state::{Step, Zone};
    // 514.1 then 514.3a: Hobgoblin's +2/+0 for each card discarded to hand size
    // lasts while players have priority in cleanup, and ends in the next cleanup step.
    let mut state = game();
    let hob = put(&mut state, P0, "Hobgoblin, Mantled Marauder");
    let base = effective_power(&state, hob);
    while state.players[0].hand.len() < 9 {
        put_in(&mut state, P0, "Mountain", Zone::Hand);
    }
    let mut discarded = 0;
    let mut cleanup_power = Vec::new();
    while state.active_player == P0 {
        let a = match engine::advance_until_decision(&mut state) {
            Decision::Discard { count, choices, .. } => {
                discarded += count;
                Action::Discard(choices[..count as usize].to_vec())
            }
            Decision::CastSpellOrPass { .. } => {
                if state.step == Step::Cleanup && state.stack.is_empty() {
                    cleanup_power.push(effective_power(&state, hob));
                }
                Action::Pass
            }
            Decision::DeclareAttackers { .. } => Action::DeclareAttackers(vec![]),
            Decision::OrderTriggers { pending, .. } => Action::OrderTriggers((0..pending.len()).collect()),
            other => panic!("unhandled decision {other:?}"),
        };
        engine::step(&mut state, a).unwrap();
    }
    assert_eq!(discarded, 2);
    assert_eq!(state.players[0].hand.len(), 7);
    assert!(!cleanup_power.is_empty() && cleanup_power.iter().all(|&p| p == base + 4), "{cleanup_power:?}");
    assert_eq!(effective_power(&state, hob), base, "the second cleanup step ended it");
}
