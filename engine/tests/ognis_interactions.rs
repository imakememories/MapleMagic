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

#[test]
fn creatures_stop_attacking_when_combat_ends() {
    // 511.3: Gornog's bonus for attacking Warriors is gone in main phase 2.
    let mut state = game();
    let gornog = put(&mut state, P0, "Gornog, the Red Reaper");
    put(&mut state, P1, "Zealous Conscripts");
    let base = mtg_kernel::engine::effective_power(&state, gornog);
    advance(&mut state, |_, d| matches!(d, Decision::DeclareAttackers { .. }));
    engine::step(&mut state, Action::DeclareAttackers(vec![gornog])).unwrap();
    advance(&mut state, |s, _| s.step == Step::CombatDamage);
    assert_eq!(mtg_kernel::engine::effective_power(&state, gornog), base + 1, "attacking, one Coward");
    advance(&mut state, |s, _| s.step == Step::Main2);
    assert_eq!(mtg_kernel::engine::effective_power(&state, gornog), base);
}

#[test]
fn depthshaker_titan_skips_a_target_that_is_no_longer_a_noncreature_artifact_you_control() {
    // 608.2b: Sol Ring changes control while the enters trigger waits; only
    // the Treasure, still legal, is animated.
    let mut state = game();
    lands(&mut state, P0, 7);
    let ring = put(&mut state, P0, "Sol Ring");
    let treasure = put(&mut state, P0, "Treasure Token");
    let titan = put_in(&mut state, P0, "Depthshaker Titan", Zone::Hand);
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    engine::step(&mut state, Action::CastSpell(titan)).unwrap();
    let mut picks = [ring, treasure].into_iter();
    let mut moved = false;
    loop {
        let a = match engine::advance_until_decision(&mut state) {
            Decision::ChooseTargets { .. } => picks.next().map_or(Action::FinishEffectSelection, |o| Action::ChooseTarget(Target::Object(o))),
            Decision::CastSpellOrPass { .. } if !state.stack.is_empty() => {
                if zone(&state, titan) == Zone::Battlefield && !moved {
                    moved = true;
                    state.players[0].battlefield.retain(|&o| o != ring);
                    state.players[1].battlefield.push(ring);
                    state.objects.get_mut(ring).controller = P1;
                }
                Action::Pass
            }
            Decision::CastSpellOrPass { .. } => break,
            other => panic!("unhandled decision {other:?}"),
        };
        engine::step(&mut state, a).unwrap();
    }
    use mtg_kernel::card_def::CardType;
    assert!(moved);
    assert!(engine::object_has_type(&state, treasure, CardType::Creature));
    assert!(!engine::object_has_type(&state, ring, CardType::Creature), "an illegal target is left alone");
}

#[test]
fn a_copy_of_spider_punk_next_to_spider_punk_has_riot_twice() {
    // 702.136b: its own riot and the one Spider-Punk gives other Spiders.
    let mut state = game();
    lands(&mut state, P0, 3);
    let punk = put(&mut state, P0, "Spider-Punk");
    let mirror = put_in(&mut state, P0, "Cursed Mirror", Zone::Hand);
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    engine::step(&mut state, Action::CastSpell(mirror)).unwrap();
    let mut riots = 0;
    loop {
        let a = match engine::advance_until_decision(&mut state) {
            Decision::ChooseCopy { .. } => Action::ChooseCopy(Some(punk)),
            Decision::ChooseRiot { creature, .. } => {
                assert_eq!(creature, mirror);
                riots += 1;
                Action::ChooseRiot(false)
            }
            Decision::ChooseLegendToKeep { .. } => Action::ChooseLegendToKeep(mirror),
            Decision::OrderTriggers { pending, .. } => Action::OrderTriggers((0..pending.len()).collect()),
            Decision::CastSpellOrPass { .. } if !state.stack.is_empty() => Action::Pass,
            Decision::CastSpellOrPass { .. } => break,
            other => panic!("unhandled decision {other:?}"),
        };
        engine::step(&mut state, a).unwrap();
    }
    assert_eq!(riots, 2);
    assert_eq!(state.objects.get(mirror).counters.plus1_plus1, 2);
}

#[test]
fn the_opposing_hideous_taskmaster_taken_by_yours_has_annihilator_twice() {
    // 702.86b: its printed annihilator 1 and the one your Taskmaster gives it.
    let mut state = game();
    lands(&mut state, P0, 7);
    let theirs = put(&mut state, P1, "Hideous Taskmaster");
    let lands_of_1: Vec<ObjectId> = (0..3).map(|_| put(&mut state, P1, "Mountain")).collect();
    cast(&mut state, "Hideous Taskmaster", Some(Target::Object(theirs)), 0);
    assert_eq!(state.objects.get(theirs).controller, P0);
    advance(&mut state, |_, d| matches!(d, Decision::DeclareAttackers { .. }));
    engine::step(&mut state, Action::DeclareAttackers(vec![theirs])).unwrap();
    let mut sacrifices = 0;
    loop {
        let a = match engine::advance_until_decision(&mut state) {
            Decision::ChooseEffectTargets { legal_targets, .. } => {
                sacrifices += 1;
                Action::ChooseEffectTarget(legal_targets[0])
            }
            Decision::OrderTriggers { pending, .. } => Action::OrderTriggers((0..pending.len()).collect()),
            Decision::CastSpellOrPass { .. } if !state.stack.is_empty() => Action::Pass,
            Decision::CastSpellOrPass { .. } => break,
            other => panic!("unhandled decision {other:?}"),
        };
        engine::step(&mut state, a).unwrap();
    }
    assert_eq!(sacrifices, 2);
    assert_eq!(lands_of_1.iter().filter(|&&m| zone(&state, m) == Zone::Graveyard).count(), 2);
}

#[test]
fn treasure_mana_floated_while_hellkite_chargers_trigger_waits_pays_for_the_extra_combat() {
    // Five lands and two Treasures pay {5}{R}{R}: the Treasures are sacrificed
    // for mana with the trigger on the stack, and it stays in the pool.
    let mut state = game();
    lands(&mut state, P0, 5);
    let hellkite = put(&mut state, P0, "Hellkite Charger");
    let treasures = [put(&mut state, P0, "Treasure Token"), put(&mut state, P0, "Treasure Token")];
    let (mut combats, mut floated, mut paid) = (0, false, None);
    loop {
        let a = match engine::advance_until_decision(&mut state) {
            Decision::DeclareAttackers { .. } if combats < 2 => {
                combats += 1;
                Action::DeclareAttackers(if combats == 1 { vec![hellkite] } else { vec![] })
            }
            Decision::DeclareAttackers { .. } => Action::DeclareAttackers(vec![]),
            Decision::CastSpellOrPass { .. } if !state.stack.is_empty() && !floated => {
                floated = true;
                for t in treasures {
                    engine::step(&mut state, Action::ActivateManaAbilityChoice(t, ManaColor::R)).unwrap();
                }
                Action::Pass
            }
            Decision::ChooseEffectBoolean { .. } => {
                paid = Some(true);
                Action::ChooseEffectBoolean(true)
            }
            Decision::CastSpellOrPass { .. } if state.step == Step::Main2 => break,
            Decision::CastSpellOrPass { .. } => Action::Pass,
            Decision::DeclareBlockers { .. } => Action::DeclareBlockers(vec![]),
            other => panic!("unhandled decision {other:?}"),
        };
        engine::step(&mut state, a).unwrap();
    }
    assert_eq!(paid, Some(true), "the payment was offered");
    assert_eq!(combats, 2, "an additional combat");
    assert!(named(&state, P0, "Treasure Token").is_empty());
}

#[test]
fn expend_counts_the_mana_a_flashback_cast_actually_spent() {
    // 702.170: Ruby Medallion makes Reckless Impulse's flashback {1}{R} cost
    // {R}, so it adds 1 to the mana spent on spells, not 2.
    let mut state = game();
    put(&mut state, P0, "Ruby Medallion");
    lands(&mut state, P0, 2);
    let impulse = put_in(&mut state, P0, "Reckless Impulse", Zone::Graveyard);
    cast(&mut state, "Flashback", Some(Target::Object(impulse)), 0);
    assert_eq!(state.spell_mana_this_turn[0], 1, "Flashback costs one red");
    cast_from_graveyard(&mut state, impulse, None);
    assert_eq!(state.spell_mana_this_turn[0], 2);
}

/// Puts `id` into its owner's graveyard and back: a new object (400.7).
fn bounce_through_graveyard(state: &mut GameState, id: ObjectId) {
    use mtg_kernel::event::{propose_and_commit, ProposedEvent};
    propose_and_commit(state, ProposedEvent::zone_change(id, Zone::Graveyard));
    propose_and_commit(state, ProposedEvent::zone_change(id, Zone::Battlefield));
}

/// Casts `card` from player 0's hand with `target`; once a trigger waits
/// above it, `respond` runs; then everything resolves.
fn cast_and_respond(state: &mut GameState, card: &str, target: Option<Target>, mut respond: impl FnMut(&mut GameState)) {
    let spell = put_in(state, P0, card, Zone::Hand);
    advance(state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    engine::step(state, Action::CastSpell(spell)).unwrap();
    let mut responded = false;
    loop {
        let a = match engine::advance_until_decision(state) {
            Decision::ChooseTargets { .. } => Action::ChooseTarget(target.expect("a target")),
            Decision::CastSpellOrPass { .. } if state.stack.len() >= 2 && !responded => {
                responded = true;
                respond(state);
                Action::Pass
            }
            Decision::CastSpellOrPass { .. } if !state.stack.is_empty() => Action::Pass,
            Decision::CastSpellOrPass { .. } => break,
            other => panic!("unhandled decision {other:?}"),
        };
        engine::step(state, a).unwrap();
    }
    assert!(responded, "a trigger waited above {card}");
}

#[test]
fn slickshots_pump_doesnt_follow_it_to_a_new_object() {
    // 400.7: Slickshot Show-Off leaves and returns while its trigger waits.
    let mut state = game();
    lands(&mut state, P0, 1);
    let show_off = put(&mut state, P0, "Slickshot Show-Off");
    let base = mtg_kernel::engine::effective_power(&state, show_off);
    cast_and_respond(&mut state, "Lightning Bolt", Some(Target::Player(P1)), |s| bounce_through_graveyard(s, show_off));
    assert_eq!(mtg_kernel::engine::effective_power(&state, show_off), base);
}

#[test]
fn laelias_counter_doesnt_follow_her_to_a_new_object() {
    // 400.7: Laelia leaves and returns while her exile trigger waits.
    let mut state = game();
    lands(&mut state, P0, 2);
    let laelia = put(&mut state, P0, "Laelia, the Blade Reforged");
    // Wrenn's Resolve exiles on resolution: respond to Laelia's trigger then.
    let spell = put_in(&mut state, P0, "Wrenn's Resolve", Zone::Hand);
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    engine::step(&mut state, Action::CastSpell(spell)).unwrap();
    let mut responded = false;
    loop {
        let a = match engine::advance_until_decision(&mut state) {
            Decision::CastSpellOrPass { .. } if zone(&state, spell) == Zone::Graveyard && !state.stack.is_empty() && !responded => {
                responded = true;
                bounce_through_graveyard(&mut state, laelia);
                Action::Pass
            }
            Decision::CastSpellOrPass { .. } if !state.stack.is_empty() => Action::Pass,
            Decision::CastSpellOrPass { .. } => break,
            other => panic!("unhandled decision {other:?}"),
        };
        engine::step(&mut state, a).unwrap();
    }
    assert!(responded);
    assert_eq!(state.objects.get(laelia).counters.plus1_plus1, 0);
}

#[test]
fn goddrics_dragon_pump_doesnt_follow_a_dragon_to_a_new_object() {
    // 400.7, 611.2c: Goldspan Dragon pumped by Goddric leaves and returns.
    use mtg_kernel::event::{propose_and_commit, ProposedEvent};
    let mut state = game();
    lands(&mut state, P0, 1);
    let goddric = put(&mut state, P0, "Goddric, Cloaked Reveler");
    let dragon = put(&mut state, P0, "Goldspan Dragon");
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    for _ in 0..2 {
        propose_and_commit(&mut state, ProposedEvent::create_token(id("Treasure Token"), P0));
    }
    activate(&mut state, goddric, 0, None);
    assert_eq!(mtg_kernel::engine::effective_power(&state, dragon), 5);
    bounce_through_graveyard(&mut state, dragon);
    assert_eq!(mtg_kernel::engine::effective_power(&state, dragon), 4);
}

#[test]
fn chaos_warp_on_a_token_reveals_the_top_card_not_the_token() {
    // 111.1, 111.8: the token is shuffled in and ceases to exist; it isn't a
    // card, so the revealed top card is a real one. Over seeds, the token is
    // sometimes shuffled to the top.
    for seed in 0..16u64 {
        let mut state = game();
        state.reseed_shuffles(seed);
        lands(&mut state, P0, 3);
        let treasure = put(&mut state, P1, "Treasure Token");
        for i in 0..10 {
            library_card(&mut state, P1, i, "Lightning Bolt");
        }
        cast(&mut state, "Chaos Warp", Some(Target::Object(treasure)), 0);
        let lib = &state.players[1].library;
        assert_eq!(lib.len(), 10, "seed {seed}: the token is gone");
        for p in [P0, P1] {
            assert!(
                state.known_library_cards(p, P1).iter().any(|e| e.position == 0 && e.object == lib[0]),
                "seed {seed}: the revealed card is known on top"
            );
        }
    }
}

#[test]
fn the_opposing_gornog_made_a_coward_loses_its_minotaur_type() {
    use mtg_kernel::card_def::Subtype;
    assert!(Subtype::CREATURE_TYPES.iter().all(|t| t.is_creature_type()));
    let mut state = game();
    let gornog = put(&mut state, P0, "Gornog, the Red Reaper");
    let theirs = put(&mut state, P1, "Gornog, the Red Reaper");
    attack_and_see_blocks(&mut state, vec![gornog], Target::Object(theirs));
    assert!(engine::has_effective_subtype(&state, theirs, Subtype::Coward));
    for t in [Subtype::Minotaur, Subtype::Warrior] {
        assert!(!engine::has_effective_subtype(&state, theirs, t), "{t:?}");
    }
}

#[test]
fn creature_type_overwrites_and_additions_apply_in_timestamp_order() {
    // 613.7: Gornog's Coward overwrites the creature types; Loki's Scepter's
    // Villain is "in addition", so whichever is newer decides.
    use mtg_kernel::card_def::Subtype;
    use mtg_kernel::engine::{effective_subtype_ids, has_effective_subtype};
    let is = |state: &GameState, o: ObjectId, t: Subtype| {
        let has = has_effective_subtype(state, o, t);
        assert_eq!(has, effective_subtype_ids(state, o).contains(&t.stable_id()), "{t:?}");
        has
    };

    // Coward, then Villain: both.
    let mut state = game();
    lands(&mut state, P0, 3);
    let gornog = put(&mut state, P0, "Gornog, the Red Reaper");
    let theirs = put(&mut state, P1, "Guttersnipe");
    attack_and_see_blocks(&mut state, vec![gornog], Target::Object(theirs));
    advance(&mut state, |s, _| s.step == Step::Main2);
    cast(&mut state, "Loki's Scepter", Some(Target::Object(theirs)), 0);
    assert!(is(&state, theirs, Subtype::Coward) && is(&state, theirs, Subtype::Villain));
    assert!(!is(&state, theirs, Subtype::Goblin));

    // Villain, then Coward: Coward only. Player 1 Chaos Warps its Treasure
    // at the beginning of combat and reveals Loki's Scepter, which makes its
    // own creature a Villain.
    let mut state = game();
    let gornog = put(&mut state, P0, "Gornog, the Red Reaper");
    let theirs = put(&mut state, P1, "Guttersnipe");
    lands(&mut state, P1, 3);
    let treasure = put(&mut state, P1, "Treasure Token");
    for i in 0..10 {
        library_card(&mut state, P1, i, "Loki's Scepter");
    }
    let warp = put_in(&mut state, P1, "Chaos Warp", Zone::Hand);
    advance(&mut state, |s, d| s.step == Step::BeginCombat && matches!(d, Decision::CastSpellOrPass { player: P1, .. }));
    engine::step(&mut state, Action::CastSpell(warp)).unwrap();
    let mut picks = [Target::Object(treasure), Target::Object(theirs)].into_iter();
    loop {
        let a = match engine::advance_until_decision(&mut state) {
            Decision::ChooseTargets { .. } => Action::ChooseTarget(picks.next().unwrap()),
            Decision::CastSpellOrPass { .. } if !state.stack.is_empty() => Action::Pass,
            Decision::CastSpellOrPass { .. } => break,
            other => panic!("unhandled decision {other:?}"),
        };
        engine::step(&mut state, a).unwrap();
    }
    assert_eq!(picks.next(), None, "Loki's Scepter's trigger targeted");
    assert!(is(&state, theirs, Subtype::Villain));
    attack_and_see_blocks(&mut state, vec![gornog], Target::Object(theirs));
    assert!(is(&state, theirs, Subtype::Coward));
    assert!(!is(&state, theirs, Subtype::Villain) && !is(&state, theirs, Subtype::Goblin));

    // A celebrating Goddric's static is older than Loki's Scepter's effect.
    use mtg_kernel::event::{propose_and_commit, ProposedEvent};
    let mut state = game();
    lands(&mut state, P0, 3);
    let goddric = put(&mut state, P0, "Goddric, Cloaked Reveler");
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    propose_and_commit(&mut state, ProposedEvent::create_token(id("Treasure Token"), P0));
    cast(&mut state, "Loki's Scepter", Some(Target::Object(goddric)), 0);
    assert!(is(&state, goddric, Subtype::Dragon) && is(&state, goddric, Subtype::Villain));
    assert!(!is(&state, goddric, Subtype::Human));
}

#[test]
fn an_attacker_that_leaves_and_returns_is_a_new_object_not_in_combat() {
    // 400.7, 506.4: as when Chaos Warp reveals the attacker it shuffled away.
    let mut state = game();
    let dragon = put(&mut state, P0, "Goldspan Dragon");
    advance(&mut state, |_, d| matches!(d, Decision::DeclareAttackers { .. }));
    engine::step(&mut state, Action::DeclareAttackers(vec![dragon])).unwrap();
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    bounce_through_graveyard(&mut state, dragon);
    assert!(!state.engine.combat.attackers.contains(&dragon));
    let life = state.players[1].life;
    advance(&mut state, |s, _| s.step == Step::Main2);
    assert_eq!(state.players[1].life, life);
}

#[test]
fn the_blockers_and_damage_steps_happen_after_the_only_attacker_died() {
    // 508.8 skips them only if no creature was declared as an attacker.
    use mtg_kernel::event::{propose_and_commit, ProposedEvent};
    let mut state = game();
    let dragon = put(&mut state, P0, "Goldspan Dragon");
    advance(&mut state, |_, d| matches!(d, Decision::DeclareAttackers { .. }));
    engine::step(&mut state, Action::DeclareAttackers(vec![dragon])).unwrap();
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    propose_and_commit(&mut state, ProposedEvent::zone_change(dragon, Zone::Graveyard));
    let mut priority = Vec::new();
    advance(&mut state, |s, d| {
        if let Decision::CastSpellOrPass { player, .. } = d {
            priority.push((s.step, *player));
        }
        s.step == Step::Main2
    });
    for step in [Step::DeclareBlockers, Step::CombatDamage] {
        for p in [P0, P1] {
            assert!(priority.contains(&(step, p)), "{step:?} {p:?}: {priority:?}");
        }
    }
}

#[test]
fn a_copy_effect_ending_uncovers_the_earlier_copy_effect() {
    // 707.9b, 613.2a: a Fire Crystal token of a Cursed Mirror copying Gogo is
    // a Gogo with haste; Gogo's trigger makes it a Goldspan Dragon until end
    // of turn, and then it's a hasty Gogo again. The token is normally
    // sacrificed at the end step first, so the copy is ended directly.
    use mtg_kernel::card_def::Keywords;
    let mut state = game();
    lands(&mut state, P0, 9);
    let gogo = put(&mut state, P1, "Gogo, Mysterious Mime");
    let dragon = put(&mut state, P0, "Goldspan Dragon");
    let crystal = put(&mut state, P0, "The Fire Crystal");
    let mirror = put_in(&mut state, P0, "Cursed Mirror", Zone::Hand);
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    engine::step(&mut state, Action::CastSpell(mirror)).unwrap();
    drive(&mut state, |s, d| match d {
        Decision::ChooseCopy { .. } => Some(Action::ChooseCopy(Some(gogo))),
        Decision::CastSpellOrPass { .. } if !s.stack.is_empty() => Some(Action::Pass),
        _ => None,
    });
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { activatable_abilities, .. } if activatable_abilities.contains(&(crystal, 0))));
    engine::step(&mut state, Action::ActivateAbility(crystal, 0)).unwrap();
    let mut token = None;
    drive(&mut state, |s, d| match d {
        Decision::ChooseTargets { .. } => Some(Action::ChooseTarget(Target::Object(mirror))),
        Decision::ChooseLegendToKeep { legends, .. } => {
            token = legends.iter().copied().find(|&l| l != mirror);
            Some(Action::ChooseLegendToKeep(token.unwrap()))
        }
        Decision::CastSpellOrPass { .. } if !s.stack.is_empty() => Some(Action::Pass),
        _ => None,
    });
    let token = token.expect("the token and the Mirror are both Gogo");
    let gogo_values = mtg_kernel::copy::copiable_values(&state, token);
    assert_eq!((gogo_values.card_def, gogo_values.haste), (id("Gogo, Mysterious Mime"), true));

    advance(&mut state, |s, d| s.step == Step::BeginCombat && matches!(d, Decision::ChooseTargets { .. }));
    engine::step(&mut state, Action::ChooseTarget(Target::Object(dragon))).unwrap();
    drive(&mut state, |s, d| match d {
        Decision::ChooseEffectOption { option_count: 2, .. } => Some(Action::ChooseEffectOption(1)),
        Decision::CastSpellOrPass { .. } if !s.stack.is_empty() => Some(Action::Pass),
        _ => None,
    });
    assert_eq!(state.objects.get(token).card_def, id("Goldspan Dragon"));

    mtg_kernel::copy::end_until_end_of_turn(&mut state);
    assert_eq!(mtg_kernel::copy::copiable_values(&state, token), gogo_values);
    assert_eq!(state.objects.get(token).name, "Gogo, Mysterious Mime");
    use mtg_kernel::event::{propose_and_commit, ProposedEvent};
    propose_and_commit(&mut state, ProposedEvent::zone_change(crystal, Zone::Graveyard));
    assert!(engine::has_effective_keyword(&state, token, Keywords::HASTE), "its own copiable haste");
}

#[test]
fn the_caster_gets_priority_first_with_its_trigger_on_the_stack() {
    // 117.3c, 117.5: player 1 casts on player 0's turn with its Slickshot
    // Show-Off out; the trigger goes on the stack and player 1 acts first.
    let mut state = game();
    put(&mut state, P1, "Slickshot Show-Off");
    lands(&mut state, P1, 1);
    let bolt = put_in(&mut state, P1, "Lightning Bolt", Zone::Hand);
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { player: P1, .. }));
    engine::step(&mut state, Action::CastSpell(bolt)).unwrap();
    let next = advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    assert_eq!(state.stack.len(), 2, "the Bolt and the trigger");
    assert!(matches!(next, Decision::CastSpellOrPass { player: P1, .. }), "{next:?}");
}

#[test]
fn the_legend_rule_choice_includes_a_legend_with_lethal_damage() {
    // 704.3: lethal damage and the legend rule are checked together, so the
    // player may keep the dying Laelia (losing both) or the other one.
    use mtg_kernel::event::{propose_and_commit, ProposedEvent};
    for keep_damaged in [false, true] {
        let mut state = game();
        let damaged = put(&mut state, P0, "Laelia, the Blade Reforged");
        let other = put_in(&mut state, P0, "Laelia, the Blade Reforged", Zone::Hand);
        let mountain = put_in(&mut state, P0, "Mountain", Zone::Hand);
        advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
        state.objects.get_mut(damaged).damage = 2;
        propose_and_commit(&mut state, ProposedEvent::zone_change(other, Zone::Battlefield));
        // A land drop runs the state-based action check.
        engine::step(&mut state, Action::PlayLand(mountain)).unwrap();
        let Decision::ChooseLegendToKeep { legends, .. } = engine::advance_until_decision(&mut state) else {
            panic!("the legend rule asks first");
        };
        assert!(legends.contains(&damaged) && legends.contains(&other), "{legends:?}");
        let keep = if keep_damaged { damaged } else { other };
        engine::step(&mut state, Action::ChooseLegendToKeep(keep)).unwrap();
        advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
        assert_eq!(zone(&state, damaged), Zone::Graveyard);
        assert_eq!(zone(&state, other), if keep_damaged { Zone::Graveyard } else { Zone::Battlefield });
    }
}

/// Player 1 casts `card` from its hand in its first main phase, choosing
/// `target` for every target, and passes until the stack is empty.
fn cast_as_player_1(state: &mut GameState, card: &str, target: Target) {
    let spell = put_in(state, P1, card, Zone::Hand);
    advance(state, |s, d| s.step == Step::Main1 && matches!(d, Decision::CastSpellOrPass { player: P1, .. }));
    engine::step(state, Action::CastSpell(spell)).unwrap();
    drive(state, |s, d| match d {
        Decision::ChooseTargets { .. } => Some(Action::ChooseTarget(target)),
        Decision::CastSpellOrPass { .. } if !s.stack.is_empty() => Some(Action::Pass),
        _ => None,
    });
}

#[test]
fn control_comes_from_the_control_effects_still_in_force_newest_first() {
    // 613.1b, 613.7, 611.2b: Zealous Conscripts (until end of turn) and
    // Pyreswipe Hawk (as long as you control it) on one Sol Ring.
    use mtg_kernel::event::{propose_and_commit, ProposedEvent};
    let controller = |state: &GameState, o: ObjectId| state.objects.get(o).controller;

    // Conscripts, then the Hawk: still player 0's after cleanup; its owner's
    // once the Hawk dies.
    let mut state = game();
    let hawk = put(&mut state, P0, "Pyreswipe Hawk");
    let ring = put(&mut state, P1, "Sol Ring");
    lands(&mut state, P0, 8);
    cast(&mut state, "Zealous Conscripts", Some(Target::Object(ring)), 0);
    cast(&mut state, "Guttersnipe", Some(Target::Object(ring)), 0);
    assert_eq!(controller(&state, ring), P0);
    end_turn(&mut state);
    assert_eq!(controller(&state, ring), P0, "the Hawk's effect outlasts the Conscripts'");
    propose_and_commit(&mut state, ProposedEvent::zone_change(hawk, Zone::Graveyard));
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    assert_eq!(controller(&state, ring), P1, "no effect left: its owner's");

    // The Hawk takes it; on player 1's turn, Conscripts takes it back and the
    // Hawk dies: player 1's after cleanup.
    let mut state = game();
    let hawk = put(&mut state, P0, "Pyreswipe Hawk");
    let ring = put(&mut state, P1, "Sol Ring");
    lands(&mut state, P0, 6);
    cast(&mut state, "Guttersnipe", None, 0);
    cast(&mut state, "Guttersnipe", Some(Target::Object(ring)), 0);
    assert_eq!(controller(&state, ring), P0);
    end_turn(&mut state);
    lands(&mut state, P1, 5);
    cast_as_player_1(&mut state, "Zealous Conscripts", Target::Object(ring));
    assert_eq!(controller(&state, ring), P1);
    propose_and_commit(&mut state, ProposedEvent::zone_change(hawk, Zone::Graveyard));
    end_turn(&mut state);
    assert_eq!(controller(&state, ring), P1);
}

#[test]
fn an_artifact_taken_by_a_stolen_pyreswipe_hawk_goes_home_at_cleanup_and_untaps() {
    // 611.2b, 514.2, 502.3: the Hawk returns at cleanup, so control of what it
    // took ends then, before its owner's untap step.
    let mut state = game();
    let hawk = put(&mut state, P1, "Pyreswipe Hawk");
    let ring = put(&mut state, P1, "Sol Ring");
    lands(&mut state, P0, 8);
    cast(&mut state, "Zealous Conscripts", Some(Target::Object(hawk)), 0);
    cast(&mut state, "Guttersnipe", Some(Target::Object(ring)), 0);
    assert_eq!(state.objects.get(ring).controller, P0);
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    engine::step(&mut state, Action::ActivateManaAbility(ring)).unwrap();
    assert!(state.objects.get(ring).tapped);
    end_turn(&mut state);
    advance(&mut state, |s, d| s.step == Step::Upkeep && matches!(d, Decision::CastSpellOrPass { .. }));
    assert_eq!(state.objects.get(ring).controller, P1);
    assert!(!state.objects.get(ring).tapped, "untapped in its owner's untap step");
}

/// A Throne of Eldraine of `color` for player 0, tapped for four.
fn tapped_throne(state: &mut GameState, color: ManaColor) -> ObjectId {
    use mtg_kernel::event::{propose_and_commit, ProposedEvent};
    let throne = put_in(state, P0, "Throne of Eldraine", Zone::Hand);
    propose_and_commit(state, ProposedEvent::zone_change(throne, Zone::Battlefield));
    assert!(matches!(engine::advance_until_decision(state), Decision::ChooseColor { .. }));
    engine::step(state, Action::ChooseColor(color)).unwrap();
    advance(state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    engine::step(state, Action::ActivateManaAbility(throne)).unwrap();
    throne
}

/// Casts `spell` from player 0's hand, answering a floating mana choice with
/// `amount` and a Cursed Mirror's copy choice with `copy`; returns the
/// options offered.
fn cast_paying_floating(state: &mut GameState, spell: ObjectId, target: Option<Target>, amount: u8, copy: Option<ObjectId>) -> Vec<u8> {
    advance(state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    engine::step(state, Action::CastSpell(spell)).unwrap();
    let mut offered = Vec::new();
    drive(state, |s, d| match d {
        Decision::ChooseFloatingMana { options, .. } => {
            offered = options.clone();
            Some(Action::ChooseFloatingMana(amount))
        }
        Decision::ChooseTargets { .. } => Some(Action::ChooseTarget(target.unwrap())),
        Decision::ChooseCopy { .. } => Some(Action::ChooseCopy(copy)),
        Decision::CastSpellOrPass { .. } if !s.stack.is_empty() => Some(Action::Pass),
        _ => None,
    });
    offered
}

#[test]
fn coin_of_mastery_doesnt_count_throne_mana_a_spell_cant_spend() {
    // Throne's red pays only monocolored red spells, so Ognis spends the
    // Mountain's floating red and no artifact mana.
    let mut state = game();
    put(&mut state, P0, "Coin of Mastery");
    tapped_throne(&mut state, ManaColor::R);
    let mountain = put(&mut state, P0, "Mountain");
    engine::step(&mut state, Action::ActivateManaAbility(mountain)).unwrap();
    lands(&mut state, P0, 3);
    let ognis = state.command.iter().copied().find(|&o| state.objects.get(o).owner == P0).unwrap();
    let offered = cast_paying_floating(&mut state, ognis, None, 0, None);
    assert!(offered.is_empty(), "no choice: {offered:?}");
    assert_eq!(zone(&state, ognis), Zone::Battlefield);
    assert_eq!(state.objects.get(ognis).counters.plus1_plus1, 0);
    let r = ManaColor::R.pool_index();
    assert_eq!((state.players[0].mana_pool[r], state.artifact_mana[0][r], state.restricted_mana[0][r]), (4, 4, 4));
}

#[test]
fn a_cursed_mirror_paid_from_floating_treasure_and_land_mana_spends_the_artifact_mana_its_caster_chooses() {
    // 601.2h: a Cursed Mirror may enter as a creature, so how much of the
    // floating Treasure mana pays for it decides Coin of Mastery's counters.
    let r = ManaColor::R.pool_index();
    for amount in [1u8, 2] {
        let mut state = game();
        put(&mut state, P0, "Coin of Mastery");
        let theirs = put(&mut state, P1, "Guttersnipe");
        let treasures = [put(&mut state, P0, "Treasure Token"), put(&mut state, P0, "Treasure Token")];
        let mountains = [put(&mut state, P0, "Mountain"), put(&mut state, P0, "Mountain")];
        advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
        for t in treasures {
            engine::step(&mut state, Action::ActivateManaAbilityChoice(t, ManaColor::R)).unwrap();
        }
        for m in mountains {
            engine::step(&mut state, Action::ActivateManaAbility(m)).unwrap();
        }
        let mirror = put_in(&mut state, P0, "Cursed Mirror", Zone::Hand);
        let offered = cast_paying_floating(&mut state, mirror, None, amount, Some(theirs));
        assert_eq!(offered, vec![1, 2], "one or both Treasure mana");
        assert_eq!(state.objects.get(mirror).counters.plus1_plus1, i16::from(amount));
        assert_eq!((state.players[0].mana_pool[r], state.artifact_mana[0][r]), (1, 2 - amount));
    }
}

#[test]
fn a_monocolored_red_spell_spends_the_throne_mana_its_caster_chooses() {
    // 601.2h: Throne's red pays only monocolored red spells, the Mountain's
    // pays anything: which one a Lightning Bolt takes is the caster's choice.
    let r = ManaColor::R.pool_index();
    for amount in [0u8, 1] {
        let mut state = game();
        tapped_throne(&mut state, ManaColor::R);
        let mountain = put(&mut state, P0, "Mountain");
        engine::step(&mut state, Action::ActivateManaAbility(mountain)).unwrap();
        let bolt = put_in(&mut state, P0, "Lightning Bolt", Zone::Hand);
        let offered = cast_paying_floating(&mut state, bolt, Some(Target::Player(P1)), amount, None);
        assert_eq!(offered, vec![0, 1]);
        assert_eq!(state.players[1].life, 37);
        assert_eq!((state.players[0].mana_pool[r], state.restricted_mana[0][r]), (4, 4 - amount));
    }
}

/// Every nonland card in the Ognis deck, from its list.
fn ognis_deck_spells() -> Vec<&'static str> {
    include_str!("../../configs/commander/ognis.txt")
        .lines()
        .map(|l| l.trim().trim_start_matches("commander:").trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| l.split_once(' ').filter(|(n, _)| n.parse::<u32>().is_ok()).map_or(l, |(_, name)| name))
        .filter(|&name| !mtg_kernel::card_def::CARD_DEFS[id(name) as usize].types.contains(&mtg_kernel::card_def::CardType::Land))
        .collect()
}

#[test]
fn ruby_medallion_and_the_fire_crystal_take_one_generic_each_off_red_spells_only() {
    // 601.2f, 118.7d: each reducer its caster controls takes {1} off a red
    // spell's generic cost, never off the colored part; colorless spells
    // and the opponent's reducers don't count. Cavern-Hoard Dragon's own
    // reduction counts the opponent's artifacts first.
    use mtg_kernel::card_def::CARD_DEFS;
    // Needs a spell or ability on the stack to target.
    let skipped = ["Deflecting Swat", "Redirect Lightning"];
    let reducers = ["Ruby Medallion", "The Fire Crystal"];
    let arrangements: [(&[&str], &[&str]); 5] =
        [(&[], &[]), (&reducers[..1], &[]), (&reducers[1..], &[]), (&reducers, &[]), (&[], &reducers)];
    let spells = ognis_deck_spells();
    assert_eq!(spells.len(), 65, "the commander and 64 nonland cards");
    for name in spells.into_iter().filter(|n| !skipped.contains(n)) {
        let def = &CARD_DEFS[id(name) as usize];
        let red = def.colors.contains(&mtg_kernel::mana::ManaColor::R);
        for (mine, theirs) in arrangements {
            let mut generic = def.cost.generic;
            if name == "Cavern-Hoard Dragon" {
                generic -= 1 + theirs.len() as u8;
            }
            let reduction = if red { generic.min(mine.len() as u8) } else { 0 };
            let need = usize::from(generic - reduction) + def.cost.pips.len();
            let castable_with = |lands_count: usize| {
                let mut state = game();
                put(&mut state, P1, "Guttersnipe");
                put(&mut state, P1, "Sol Ring");
                put_in(&mut state, P0, "Lightning Bolt", Zone::Graveyard);
                put_in(&mut state, P0, "Mountain", Zone::Hand);
                for r in mine {
                    put(&mut state, P0, r);
                }
                for r in theirs {
                    put(&mut state, P1, r);
                }
                lands(&mut state, P0, lands_count);
                let card = if name == "Ognis, the Dragon's Lash" {
                    state.command.iter().copied().find(|&o| state.objects.get(o).owner == P0).unwrap()
                } else {
                    put_in(&mut state, P0, name, Zone::Hand)
                };
                castable(&mut state, card)
            };
            assert!(castable_with(need), "{name} with {mine:?} / {theirs:?}: {need} mana");
            if need > 0 {
                assert!(!castable_with(need - 1), "{name} with {mine:?} / {theirs:?}: not {} mana", need - 1);
            }
        }
    }
}

/// Starts casting `spell` from player 0's hand (mode `mode` if modal, its
/// normal cost) and returns the targets its first target choice offers.
fn targets_offered(state: &mut GameState, spell: &str, mode: u8) -> Vec<Target> {
    let card = put_in(state, P0, spell, Zone::Hand);
    advance(state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    engine::step(state, Action::CastSpell(card)).unwrap();
    loop {
        let a = match engine::advance_until_decision(state) {
            Decision::ChooseSpellMode { .. } => Action::ChooseSpellMode(mode),
            Decision::ChooseCastMode { .. } => Action::ChooseCastMode(engine::CastMode::Normal),
            Decision::ChooseTargets { legal_targets, .. } => return legal_targets,
            other => panic!("{spell}: unhandled decision {other:?}"),
        };
        engine::step(state, a).unwrap();
    }
}

#[test]
fn each_kind_of_artifact_is_a_target_for_the_artifact_spells_and_lowers_cavern_hoards_cost() {
    // An artifact card, a Treasure token and a creature Liquimetal Torque
    // made an artifact (layer 4) are artifacts; a plain creature isn't.
    // Player 1 controls each one.
    #[derive(Debug, Clone, Copy, PartialEq)]
    enum Kind {
        Card,
        Token,
        Torqued,
        Plain,
    }
    let setup = |state: &mut GameState, kind: Kind| match kind {
        Kind::Card => put(state, P1, "Sol Ring"),
        Kind::Token => put(state, P1, "Treasure Token"),
        Kind::Torqued | Kind::Plain => {
            let g = put(state, P1, "Guttersnipe");
            if kind == Kind::Torqued {
                let torque = put(state, P0, "Liquimetal Torque");
                activate(state, torque, 0, Some(Target::Object(g)));
            }
            g
        }
    };
    for kind in [Kind::Card, Kind::Token, Kind::Torqued, Kind::Plain] {
        let artifact = kind != Kind::Plain;
        for (spell, mode) in [("Shattering Spree", 0), ("Vandalblast", 0), ("Untimely Malfunction", 0)] {
            let mut state = game();
            let o = setup(&mut state, kind);
            // So each spell has some legal target.
            let ring = put(&mut state, P1, "Sol Ring");
            lands(&mut state, P0, 2);
            let offered = targets_offered(&mut state, spell, mode);
            assert!(offered.contains(&Target::Object(ring)));
            assert_eq!(offered.contains(&Target::Object(o)), artifact, "{spell} at {kind:?}: {offered:?}");
        }
        // {7}{R}{R}, {X} less for each artifact the opponent controls.
        let need = 9 - usize::from(artifact);
        for (lands_count, ok) in [(need, true), (need - 1, false)] {
            let mut state = game();
            setup(&mut state, kind);
            lands(&mut state, P0, lands_count);
            let dragon = put_in(&mut state, P0, "Cavern-Hoard Dragon", Zone::Hand);
            assert_eq!(castable(&mut state, dragon), ok, "{kind:?} with {lands_count} lands");
        }
    }
}

#[test]
fn a_stolen_permanents_you_text_serves_the_thief() {
    // 109.5, 613.1b: Zealous Conscripts stands for the threaten cards.
    use mtg_kernel::card_def::Keywords;
    // The opposing Ognis makes the thief's tapped Treasure.
    let mut state = game();
    let ognis = put(&mut state, P1, "Ognis, the Dragon's Lash");
    lands(&mut state, P0, 5);
    cast(&mut state, "Zealous Conscripts", Some(Target::Object(ognis)), 0);
    let conscripts = named(&state, P0, "Zealous Conscripts")[0];
    attack_with(&mut state, vec![conscripts]);
    assert_eq!(named(&state, P0, "Treasure Token").len(), 1, "the stolen Ognis's trigger");
    assert!(named(&state, P1, "Treasure Token").is_empty());

    // The opposing Ruby Medallion reduces the thief's red spells.
    let mut state = game();
    let ruby = put(&mut state, P1, "Ruby Medallion");
    lands(&mut state, P0, 5);
    cast(&mut state, "Zealous Conscripts", Some(Target::Object(ruby)), 0);
    lands(&mut state, P0, 1);
    let impulse = put_in(&mut state, P0, "Reckless Impulse", Zone::Hand);
    assert!(castable(&mut state, impulse), "{{1}}{{R}} for one mana");

    // The opposing Fire Crystal gives the thief's creatures haste, not its owner's.
    let mut state = game();
    let crystal = put(&mut state, P1, "The Fire Crystal");
    let theirs = put(&mut state, P1, "Spider-Punk");
    state.objects.get_mut(theirs).summoning_sick = true;
    lands(&mut state, P0, 5);
    cast(&mut state, "Zealous Conscripts", Some(Target::Object(crystal)), 0);
    let mine = put(&mut state, P0, "Spider-Punk");
    state.objects.get_mut(mine).summoning_sick = true;
    assert!(engine::has_effective_keyword(&state, mine, Keywords::HASTE));
    assert!(!engine::has_effective_keyword(&state, theirs, Keywords::HASTE));
}

#[test]
fn each_trampler_assigns_lethal_to_its_blocker_and_the_rest_to_the_player() {
    // 702.19b-c: lethal counts damage already marked, so Knuckles's second
    // strike has 1 left over; Depthshaker Titan gives its artifact creatures
    // trample and melee (+1/+1 for the one opponent attacked). Each attacker
    // is blocked by a 3/3 (the flying Dragon by a 4/4 flyer) and gives the
    // player all it may.
    let rows: [(&str, &str, i32); 5] = [
        ("Knuckles the Echidna", "Zealous Conscripts", 1),
        ("Cavern-Hoard Dragon", "Pyreswipe Hawk", 2),
        ("Hideous Taskmaster", "Zealous Conscripts", 4),
        ("Depthshaker Titan", "Zealous Conscripts", 3),
        ("Sol Ring", "Zealous Conscripts", 1),
    ];
    for (card, blocker, to_player) in rows {
        let mut state = game();
        let attacker = put(&mut state, P0, card);
        if card == "Sol Ring" {
            // Animated by a Depthshaker Titan's enters trigger, paid with
            // floating land mana so the Sol Ring stays untapped.
            let mountains: Vec<ObjectId> = (0..7).map(|_| put(&mut state, P0, "Mountain")).collect();
            let titan = put_in(&mut state, P0, "Depthshaker Titan", Zone::Hand);
            advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
            for m in mountains {
                engine::step(&mut state, Action::ActivateManaAbility(m)).unwrap();
            }
            engine::step(&mut state, Action::CastSpell(titan)).unwrap();
            let mut picked = false;
            drive(&mut state, |s, d| match d {
                Decision::ChooseTargets { .. } if !picked => {
                    picked = true;
                    Some(Action::ChooseTarget(Target::Object(attacker)))
                }
                Decision::ChooseTargets { .. } => Some(Action::FinishEffectSelection),
                Decision::CastSpellOrPass { .. } if !s.stack.is_empty() => Some(Action::Pass),
                _ => None,
            });
        }
        let blocker = put(&mut state, P1, blocker);
        lands(&mut state, P1, 2);
        let life = state.players[1].life;
        advance(&mut state, |_, d| matches!(d, Decision::DeclareAttackers { .. }));
        engine::step(&mut state, Action::DeclareAttackers(vec![attacker])).unwrap();
        let mut blocked = false;
        drive(&mut state, |s, d| match d {
            Decision::DeclareBlockers { .. } => {
                blocked = true;
                Some(Action::DeclareBlockers(vec![(blocker, attacker)]))
            }
            Decision::AssignCombatDamage { options, .. } => Some(Action::AssignCombatDamage(*options.iter().max().unwrap())),
            // Annihilator: player 1 sacrifices a land, not its blocker.
            Decision::ChooseEffectTargets { legal_targets, .. } => {
                Some(Action::ChooseEffectTarget(*legal_targets.iter().find(|&&t| t != Target::Object(blocker)).unwrap()))
            }
            Decision::ChooseTargets { legal_targets, .. } => Some(Action::ChooseTarget(legal_targets[0])),
            Decision::OrderTriggers { pending, .. } => Some(Action::OrderTriggers((0..pending.len()).collect())),
            Decision::CastSpellOrPass { .. } if matches!(s.step, Step::EndCombat | Step::Main2) => None,
            Decision::CastSpellOrPass { .. } => Some(Action::Pass),
            _ => None,
        });
        assert!(blocked, "{card}");
        assert_eq!(life - state.players[1].life, to_player, "{card}");
        assert_eq!(zone(&state, blocker), Zone::Graveyard, "{card}");
    }
}

#[test]
fn a_blocked_trampler_whose_blockers_all_left_deals_its_damage_to_the_player() {
    // 702.19d; without trample a blocked creature stays blocked and deals
    // none (509.1h, 510.1c). The blocker dies before damage.
    use mtg_kernel::event::{propose_and_commit, ProposedEvent};
    for (card, to_player) in [("Knuckles the Echidna", 4), ("Depthshaker Titan", 6), ("Molten Primordial", 0)] {
        let mut state = game();
        let attacker = put(&mut state, P0, card);
        let blocker = put(&mut state, P1, "Zealous Conscripts");
        advance(&mut state, |_, d| matches!(d, Decision::DeclareAttackers { .. }));
        engine::step(&mut state, Action::DeclareAttackers(vec![attacker])).unwrap();
        advance(&mut state, |_, d| matches!(d, Decision::DeclareBlockers { .. }));
        engine::step(&mut state, Action::DeclareBlockers(vec![(blocker, attacker)])).unwrap();
        propose_and_commit(&mut state, ProposedEvent::zone_change(blocker, Zone::Graveyard));
        let life = state.players[1].life;
        advance(&mut state, |s, _| matches!(s.step, Step::EndCombat | Step::Main2));
        assert_eq!(life - state.players[1].life, to_player, "{card}");
    }
}

#[test]
fn two_until_end_of_turn_steals_of_one_creature_both_end_at_cleanup() {
    // 613.7, 514.2: Zealous Conscripts, then Loki's Scepter, take player 1's
    // Guttersnipe; both effects end together and it goes back to its owner.
    let mut state = game();
    let theirs = put(&mut state, P1, "Guttersnipe");
    lands(&mut state, P0, 8);
    cast(&mut state, "Zealous Conscripts", Some(Target::Object(theirs)), 0);
    cast(&mut state, "Loki's Scepter", Some(Target::Object(theirs)), 0);
    assert_eq!(state.objects.get(theirs).controller, P0);
    end_turn(&mut state);
    assert_eq!(state.objects.get(theirs).controller, P1);
    assert!(state.players[1].battlefield.contains(&theirs));
}

#[test]
fn granted_haste_lets_a_creature_attack_the_turn_it_arrives_and_ognis_makes_its_treasure() {
    // Each source of haste besides printed haste and The Fire Crystal's.
    let sources = ["threaten", "riot", "Gogo", "Cursed Mirror", "Depthshaker Titan"];
    for source in sources {
        let mut state = game();
        put(&mut state, P0, "Ognis, the Dragon's Lash");
        let attackers: Vec<ObjectId> = match source {
            "threaten" => {
                let theirs = put(&mut state, P1, "Spider-Punk");
                lands(&mut state, P0, 5);
                cast(&mut state, "Zealous Conscripts", Some(Target::Object(theirs)), 0);
                vec![theirs]
            }
            "riot" => {
                lands(&mut state, P0, 2);
                vec![cast_with_riot(&mut state, "Spider-Punk", true)]
            }
            "Gogo" => {
                let gogo = put(&mut state, P0, "Gogo, Mysterious Mime");
                let punk = put(&mut state, P0, "Spider-Punk");
                vec![gogo, punk]
            }
            "Cursed Mirror" => {
                let theirs = put(&mut state, P1, "Spider-Punk");
                lands(&mut state, P0, 3);
                let mirror = put_in(&mut state, P0, "Cursed Mirror", Zone::Hand);
                advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
                engine::step(&mut state, Action::CastSpell(mirror)).unwrap();
                drive(&mut state, |s, d| match d {
                    Decision::ChooseCopy { .. } => Some(Action::ChooseCopy(Some(theirs))),
                    Decision::ChooseRiot { .. } => Some(Action::ChooseRiot(false)),
                    Decision::CastSpellOrPass { .. } if !s.stack.is_empty() => Some(Action::Pass),
                    _ => None,
                });
                vec![mirror]
            }
            _ => {
                put(&mut state, P0, "Depthshaker Titan");
                let ring = put(&mut state, P0, "Sol Ring");
                animate(&mut state, ring);
                vec![ring]
            }
        };
        for &a in &attackers {
            state.objects.get_mut(a).summoning_sick = true;
        }
        let mut declared = false;
        drive(&mut state, |_, d| match d {
            Decision::DeclareAttackers { .. } if !declared => {
                declared = true;
                Some(Action::DeclareAttackers(attackers.clone()))
            }
            Decision::ChooseTargets { legal_targets, .. } => {
                // Gogo copies Spider-Punk; attack triggers aim at player 1.
                let punk = attackers.last().map(|&p| Target::Object(p)).filter(|t| legal_targets.contains(t));
                Some(Action::ChooseTarget(punk.unwrap_or(Target::Player(P1))))
            }
            Decision::ChooseEffectOption { .. } => Some(Action::ChooseEffectOption(1)),
            Decision::OrderTriggers { pending, .. } => Some(Action::OrderTriggers((0..pending.len()).collect())),
            Decision::DeclareBlockers { .. } => None,
            Decision::CastSpellOrPass { .. } => Some(Action::Pass),
            _ => None,
        });
        assert!(declared, "{source}");
        let tapped = named(&state, P0, "Treasure Token").into_iter().filter(|&t| state.objects.get(t).tapped).count();
        assert_eq!(tapped, attackers.len(), "{source}");
    }
}

#[test]
fn more_stolen_permanents_read_the_thief_as_you() {
    // 109.5, 613.1b: each is taken from player 1 with Zealous Conscripts.
    use mtg_kernel::card_def::{Keywords, Subtype};
    use mtg_kernel::event::{propose_and_commit, ProposedEvent};
    let steal = |state: &mut GameState, card: &str| {
        let theirs = put(state, P1, card);
        lands(state, P0, 5);
        cast(state, "Zealous Conscripts", Some(Target::Object(theirs)), 0);
        assert_eq!(state.objects.get(theirs).controller, P0, "{card}");
        theirs
    };
    let r = ManaColor::R.pool_index();

    // Goldspan Dragon: the thief's Treasures make two mana.
    let mut state = game();
    steal(&mut state, "Goldspan Dragon");
    let treasure = put(&mut state, P0, "Treasure Token");
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    engine::step(&mut state, Action::ActivateManaAbilityChoice(treasure, ManaColor::R)).unwrap();
    assert_eq!(state.players[0].mana_pool[r], 2);

    // Goblin War Drums: the thief's creatures have menace, the owner's don't.
    let mut state = game();
    let theirs = put(&mut state, P1, "Spider-Punk");
    steal(&mut state, "Goblin War Drums");
    let conscripts = named(&state, P0, "Zealous Conscripts")[0];
    assert!(engine::has_effective_keyword(&state, conscripts, Keywords::MENACE));
    assert!(!engine::has_effective_keyword(&state, theirs, Keywords::MENACE));

    // Slickshot Show-Off: the thief's noncreature spells pump it.
    let mut state = game();
    let show_off = steal(&mut state, "Slickshot Show-Off");
    let base = engine::effective_power(&state, show_off);
    lands(&mut state, P0, 1);
    cast(&mut state, "Lightning Bolt", Some(Target::Player(P1)), 0);
    assert_eq!(engine::effective_power(&state, show_off), base + 2);

    // Idol of Oblivion: the thief's token this turn lets it draw.
    let mut state = game();
    let idol = steal(&mut state, "Idol of Oblivion");
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    assert!(!offered_ability(&mut state, idol, 0), "no token yet");
    propose_and_commit(&mut state, ProposedEvent::create_token(id("Treasure Token"), P0));
    assert!(offered_ability(&mut state, idol, 0));

    // Dragonspark Reactor: artifacts entering under the thief add counters.
    let mut state = game();
    let reactor = steal(&mut state, "Dragonspark Reactor");
    let before = state.objects.get(reactor).counters.charge;
    for p in [P1, P0] {
        propose_and_commit(&mut state, ProposedEvent::create_token(id("Treasure Token"), p));
        advance(&mut state, |s, d| s.stack.is_empty() && matches!(d, Decision::CastSpellOrPass { .. }));
    }
    assert_eq!(state.objects.get(reactor).counters.charge, before + 1, "only the thief's Treasure");

    // Goddric: celebration counts what entered under the thief.
    let mut state = game();
    let goddric = steal(&mut state, "Goddric, Cloaked Reveler");
    assert!(!engine::has_effective_subtype(&state, goddric, Subtype::Dragon), "only Conscripts entered");
    propose_and_commit(&mut state, ProposedEvent::create_token(id("Treasure Token"), P1));
    assert!(!engine::has_effective_subtype(&state, goddric, Subtype::Dragon), "not the owner's Treasure");
    propose_and_commit(&mut state, ProposedEvent::create_token(id("Treasure Token"), P0));
    assert!(engine::has_effective_subtype(&state, goddric, Subtype::Dragon));
}

#[test]
fn the_legend_rule_applies_to_each_legend_in_the_deck_per_controller() {
    // 704.5j: two with the same name under one player ask which to keep; one
    // each is fine; stealing the opponent's makes it apply.
    use mtg_kernel::card_def::{Supertype, CARD_DEFS};
    use mtg_kernel::event::{propose_and_commit, ProposedEvent};
    let legends: Vec<&str> = ognis_deck_spells()
        .into_iter()
        .filter(|&n| CARD_DEFS[id(n) as usize].supertypes.contains(&Supertype::Legendary))
        .collect();
    assert!(legends.len() >= 15, "{legends:?}");
    for name in legends {
        for (first_owner, keep_second) in [(P0, false), (P0, true), (P1, false)] {
            let mut state = game();
            let first = put(&mut state, first_owner, name);
            let second = put_in(&mut state, P0, name, Zone::Hand);
            propose_and_commit(&mut state, ProposedEvent::zone_change(second, Zone::Battlefield));
            match engine::advance_until_decision(&mut state) {
                Decision::ChooseLegendToKeep { player, legends } if first_owner == P0 => {
                    assert_eq!((player, legends.len()), (P0, 2), "{name}");
                    let keep = if keep_second { second } else { first };
                    engine::step(&mut state, Action::ChooseLegendToKeep(keep)).unwrap();
                    let gone = if keep_second { first } else { second };
                    assert_eq!(zone(&state, keep), Zone::Battlefield, "{name}");
                    assert_ne!(zone(&state, gone), Zone::Battlefield, "{name}");
                }
                Decision::ChooseLegendToKeep { .. } => panic!("{name}: one each is fine"),
                _ if first_owner == P0 => panic!("{name}: no legend rule choice"),
                _ => {}
            }
        }
    }
    // Stolen: Zealous Conscripts takes the opposing Laelia while player 0 has one.
    let mut state = game();
    put(&mut state, P0, "Laelia, the Blade Reforged");
    let theirs = put(&mut state, P1, "Laelia, the Blade Reforged");
    lands(&mut state, P0, 5);
    let conscripts = put_in(&mut state, P0, "Zealous Conscripts", Zone::Hand);
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    engine::step(&mut state, Action::CastSpell(conscripts)).unwrap();
    let mut asked = false;
    drive(&mut state, |s, d| match d {
        Decision::ChooseTargets { .. } => Some(Action::ChooseTarget(Target::Object(theirs))),
        Decision::ChooseLegendToKeep { .. } => {
            asked = true;
            Some(Action::ChooseLegendToKeep(theirs))
        }
        Decision::CastSpellOrPass { .. } if !s.stack.is_empty() => Some(Action::Pass),
        _ => None,
    });
    assert!(asked, "the stolen Laelia makes two under player 0");
    assert_eq!(named(&state, P0, "Laelia, the Blade Reforged"), vec![theirs]);
}

#[test]
fn throne_of_eldraine_red_mana_casts_exactly_the_monocolored_red_spells() {
    // 106.6: four floating red from a red Throne, plus Mountains for anything
    // past four; only a monocolored red spell can use the Throne's mana.
    use mtg_kernel::card_def::CARD_DEFS;
    let skipped = ["Deflecting Swat", "Redirect Lightning"];
    for name in ognis_deck_spells().into_iter().filter(|n| !skipped.contains(n)) {
        let def = &CARD_DEFS[id(name) as usize];
        let mono_red = def.colors == [ManaColor::R];
        let cost = usize::from(def.cost.generic) + def.cost.pips.len();
        let mut state = game();
        put(&mut state, P1, "Guttersnipe");
        put(&mut state, P1, "Sol Ring");
        put_in(&mut state, P0, "Lightning Bolt", Zone::Graveyard);
        put_in(&mut state, P0, "Mountain", Zone::Hand);
        tapped_throne(&mut state, ManaColor::R);
        lands(&mut state, P0, cost.saturating_sub(4));
        let card = if name == "Ognis, the Dragon's Lash" {
            state.command.iter().copied().find(|&o| state.objects.get(o).owner == P0).unwrap()
        } else {
            put_in(&mut state, P0, name, Zone::Hand)
        };
        assert_eq!(castable(&mut state, card), mono_red, "{name}");
    }
}

#[test]
fn redirect_lightning_answers_exactly_the_single_target_spells_and_deflecting_swat_any_spell() {
    // 115.9a: "target spell ... with a single target"; Deflecting Swat targets
    // any spell. Player 0 casts each instant and sorcery in the deck, aiming
    // at player 1's things; player 1 holds both answers.
    use mtg_kernel::card_def::{CardType, CARD_DEFS};
    let single_target = [
        "Chaos Warp",
        "Shattering Spree",
        "Untimely Malfunction",
        "Vandalblast",
        "Wild Magic Surge",
        "Call for Aid",
        "Mass Mutiny",
        "Flashback",
    ];
    // These need a spell or ability on the stack to target.
    let skipped = ["Deflecting Swat", "Redirect Lightning"];
    let spells: Vec<&str> = ognis_deck_spells()
        .into_iter()
        .filter(|&n| {
            let types = CARD_DEFS[id(n) as usize].types;
            (types.contains(&CardType::Instant) || types.contains(&CardType::Sorcery)) && !skipped.contains(&n)
        })
        .collect();
    assert_eq!(spells.len(), 19, "{spells:?}");
    for name in spells {
        let mut state = game();
        let ring = put(&mut state, P1, "Sol Ring");
        put(&mut state, P1, "Guttersnipe");
        put_in(&mut state, P0, "Lightning Bolt", Zone::Graveyard);
        put_in(&mut state, P0, "Mountain", Zone::Hand);
        lands(&mut state, P0, 8);
        lands(&mut state, P1, 5);
        let redirect = put_in(&mut state, P1, "Redirect Lightning", Zone::Hand);
        let swat = put_in(&mut state, P1, "Deflecting Swat", Zone::Hand);
        let spell = put_in(&mut state, P0, name, Zone::Hand);
        let mut offered = None;
        drive(&mut state, |s, d| match d {
            Decision::CastSpellOrPass { player: P0, .. } if s.stack.is_empty() && zone(s, spell) == Zone::Hand => {
                Some(Action::CastSpell(spell))
            }
            Decision::CastSpellOrPass { player: P1, castable_spells, .. } if zone(s, spell) == Zone::Stack => {
                offered = Some((castable_spells.contains(&redirect), castable_spells.contains(&swat)));
                None
            }
            Decision::CastSpellOrPass { .. } if offered.is_none() && zone(s, spell) == Zone::Stack => Some(Action::Pass),
            Decision::ChooseTargets { legal_targets, .. } | Decision::ChooseEffectTargets { legal_targets, .. } => {
                let theirs = |t: &&Target| match **t {
                    Target::Player(p) => p == P1,
                    Target::Object(o) => o == ring || (s.objects.get(o).controller == P1 && s.objects.get(o).zone == Zone::Battlefield),
                    _ => false,
                };
                let t = *legal_targets.iter().find(theirs).unwrap_or(&legal_targets[0]);
                Some(if matches!(d, Decision::ChooseTargets { .. }) { Action::ChooseTarget(t) } else { Action::ChooseEffectTarget(t) })
            }
            Decision::ChooseSpellMode { .. } => Some(Action::ChooseSpellMode(0)),
            Decision::ChooseEffectOption { .. } => Some(Action::ChooseEffectOption(0)),
            Decision::ChooseCastMode { .. } => Some(Action::ChooseCastMode(engine::CastMode::Normal)),
            Decision::ChooseCostTargets { candidates, .. } => Some(Action::ChooseCostTarget(candidates[0])),
            Decision::Discard { choices, .. } => Some(Action::Discard(vec![choices[0]])),
            Decision::OrderTriggers { pending, .. } => Some(Action::OrderTriggers((0..pending.len()).collect())),
            Decision::CastSpellOrPass { .. } => None,
            other => panic!("{name}: unhandled decision {other:?}"),
        });
        let (redirect_offered, swat_offered) = offered.unwrap_or_else(|| panic!("{name}: player 1 got priority with it on the stack"));
        assert_eq!(redirect_offered, single_target.contains(&name), "{name}: Redirect Lightning");
        assert!(swat_offered, "{name}: Deflecting Swat");
    }
}

#[test]
fn goldspan_dragon_makes_a_treasure_when_any_spell_targets_it_but_not_an_ability() {
    // "Becomes the target of a spell": player 0's spells at player 1's
    // Goldspan Dragon give player 1 a Treasure; Hylda's Crown's ability doesn't.
    for (spell, mode) in [("Chaos Warp", 0u8), ("Wild Magic Surge", 0), ("Mass Mutiny", 0), ("Untimely Malfunction", 2)] {
        let mut state = game();
        let dragon = put(&mut state, P1, "Goldspan Dragon");
        lands(&mut state, P0, 5);
        let card = put_in(&mut state, P0, spell, Zone::Hand);
        advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
        engine::step(&mut state, Action::CastSpell(card)).unwrap();
        let mut aimed = false;
        drive(&mut state, |s, d| match d {
            Decision::ChooseSpellMode { .. } => Some(Action::ChooseSpellMode(mode)),
            Decision::ChooseTargets { .. } if !aimed => {
                aimed = true;
                Some(Action::ChooseTarget(Target::Object(dragon)))
            }
            Decision::ChooseTargets { .. } | Decision::ChooseEffectTargets { .. } => Some(Action::FinishEffectSelection),
            Decision::OrderTriggers { pending, .. } => Some(Action::OrderTriggers((0..pending.len()).collect())),
            Decision::CastSpellOrPass { .. } if !s.stack.is_empty() => Some(Action::Pass),
            _ => None,
        });
        assert!(aimed, "{spell}");
        assert_eq!(named(&state, P1, "Treasure Token").len(), 1, "{spell}");
    }
    let mut state = game();
    let dragon = put(&mut state, P1, "Goldspan Dragon");
    let crown = put(&mut state, P0, "Hylda's Crown of Winter");
    lands(&mut state, P0, 1);
    activate(&mut state, crown, 0, Some(Target::Object(dragon)));
    assert!(state.objects.get(dragon).tapped);
    assert!(named(&state, P1, "Treasure Token").is_empty(), "an ability isn't a spell");
}

#[test]
fn a_permanent_put_onto_the_battlefield_by_chaos_warp_isnt_cast_but_enters() {
    // Player 0 Chaos Warps its own Treasure; the top card of its library is
    // put onto the battlefield under its control. X is 0, "as enters" choices
    // and enters triggers happen, Sagas get their first lore counter, and
    // Coin of Mastery adds nothing (no mana was spent to cast it).
    for card in ["Shivan Devastator", "Spider-Punk", "Cursed Mirror", "Throne of Eldraine", "Summon: Esper Valigarmanda", "Zealous Conscripts", "Laelia, the Blade Reforged"] {
        let mut state = game();
        put(&mut state, P0, "Coin of Mastery");
        put(&mut state, P1, "Guttersnipe");
        let treasure = put(&mut state, P0, "Treasure Token");
        for i in 0..10 {
            library_card(&mut state, P0, i, card);
        }
        lands(&mut state, P0, 3);
        let warp = put_in(&mut state, P0, "Chaos Warp", Zone::Hand);
        advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
        engine::step(&mut state, Action::CastSpell(warp)).unwrap();
        let mut asked = Vec::new();
        drive(&mut state, |s, d| match d {
            Decision::ChooseTargets { legal_targets, .. } if zone(s, warp) == Zone::Stack && s.stack.len() == 1 && !legal_targets.is_empty() && asked.is_empty() && legal_targets.contains(&Target::Object(treasure)) => {
                Some(Action::ChooseTarget(Target::Object(treasure)))
            }
            Decision::ChooseTargets { legal_targets, .. } => {
                asked.push("enters trigger target");
                Some(Action::ChooseTarget(legal_targets[0]))
            }
            Decision::ChooseRiot { .. } => {
                asked.push("riot");
                Some(Action::ChooseRiot(true))
            }
            Decision::ChooseCopy { .. } => {
                asked.push("copy");
                Some(Action::ChooseCopy(None))
            }
            Decision::ChooseColor { .. } => {
                asked.push("color");
                Some(Action::ChooseColor(ManaColor::R))
            }
            Decision::ChooseEffectTargets { can_finish: true, .. } => Some(Action::FinishEffectSelection),
            Decision::OrderTriggers { pending, .. } => Some(Action::OrderTriggers((0..pending.len()).collect())),
            Decision::CastSpellOrPass { .. } if !s.stack.is_empty() => Some(Action::Pass),
            _ => None,
        });
        let entered = state.players[0].battlefield.iter().copied().find(|&o| state.objects.get(o).name == card);
        let graveyard = state.players[0].graveyard.iter().any(|&o| state.objects.get(o).name == card);
        match card {
            "Shivan Devastator" => assert!(entered.is_none() && graveyard, "X is 0: a 0/0"),
            "Spider-Punk" => assert_eq!(asked, ["riot"]),
            "Cursed Mirror" => assert_eq!(asked, ["copy"]),
            "Throne of Eldraine" => assert_eq!(asked, ["color"]),
            "Summon: Esper Valigarmanda" => assert_eq!(state.objects.get(entered.unwrap()).counters.lore, 1),
            "Zealous Conscripts" => assert_eq!(asked, ["enters trigger target"]),
            _ => {
                let laelia = entered.unwrap();
                assert_eq!(state.objects.get(laelia).counters.plus1_plus1, 0, "Coin counts only mana spent to cast it");
                assert_eq!(state.objects.get(laelia).controller, P0);
            }
        }
    }
}

#[test]
fn slickshot_show_off_grows_for_each_noncreature_spell_in_the_deck_and_no_creature_spell() {
    // Artifacts and enchantments count; Summon: Esper Valigarmanda is a
    // creature spell, so it doesn't.
    use mtg_kernel::card_def::{CardType, CARD_DEFS};
    let skipped = ["Deflecting Swat", "Redirect Lightning", "Slickshot Show-Off"];
    for name in ognis_deck_spells().into_iter().filter(|n| !skipped.contains(n)) {
        let noncreature = !CARD_DEFS[id(name) as usize].types.contains(&CardType::Creature);
        let mut state = game();
        let show_off = put(&mut state, P0, "Slickshot Show-Off");
        let base = engine::effective_power(&state, show_off);
        let ring = put(&mut state, P1, "Sol Ring");
        put(&mut state, P1, "Guttersnipe");
        put_in(&mut state, P0, "Lightning Bolt", Zone::Graveyard);
        put_in(&mut state, P0, "Mountain", Zone::Hand);
        lands(&mut state, P0, 9);
        let spell = if name == "Ognis, the Dragon's Lash" {
            state.command.iter().copied().find(|&o| state.objects.get(o).owner == P0).unwrap()
        } else {
            put_in(&mut state, P0, name, Zone::Hand)
        };
        advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
        engine::step(&mut state, Action::CastSpell(spell)).unwrap();
        let mut power_after_cast = None;
        drive(&mut state, |s, d| match d {
            Decision::CastSpellOrPass { .. } if !s.stack.is_empty() => {
                // The cast trigger resolves first; read the power before the
                // spell itself resolves.
                if power_after_cast.is_none() && s.stack.len() == 1 && zone(s, spell) == Zone::Stack {
                    power_after_cast = Some(engine::effective_power(s, show_off));
                }
                Some(Action::Pass)
            }
            Decision::ChooseTargets { legal_targets, .. } | Decision::ChooseEffectTargets { legal_targets, .. } => {
                let theirs = |t: &&Target| match **t {
                    Target::Player(p) => p == P1,
                    Target::Object(o) => o == ring || (s.objects.get(o).controller == P1 && s.objects.get(o).zone == Zone::Battlefield),
                    _ => false,
                };
                match legal_targets.iter().find(theirs).or(legal_targets.first()) {
                    Some(&t) if matches!(d, Decision::ChooseTargets { .. }) => Some(Action::ChooseTarget(t)),
                    Some(&t) => Some(Action::ChooseEffectTarget(t)),
                    None => Some(Action::FinishEffectSelection),
                }
            }
            Decision::ChooseSpellMode { .. } => Some(Action::ChooseSpellMode(0)),
            Decision::ChooseEffectOption { .. } => Some(Action::ChooseEffectOption(0)),
            Decision::ChooseCastMode { .. } => Some(Action::ChooseCastMode(engine::CastMode::Normal)),
            Decision::ChooseCostTargets { candidates, .. } => Some(Action::ChooseCostTarget(candidates[0])),
            Decision::Discard { choices, .. } => Some(Action::Discard(vec![choices[0]])),
            Decision::ChooseRiot { .. } => Some(Action::ChooseRiot(true)),
            Decision::ChooseCopy { .. } => Some(Action::ChooseCopy(None)),
            Decision::ChooseColor { .. } => Some(Action::ChooseColor(ManaColor::R)),
            Decision::ChooseEffectBoolean { .. } => Some(Action::ChooseEffectBoolean(false)),
            Decision::ChooseOptionalCost { .. } => Some(Action::ChooseOptionalCost(engine::OptionalCostChoice::Decline)),
            Decision::OrderTriggers { pending, .. } => Some(Action::OrderTriggers((0..pending.len()).collect())),
            Decision::CastSpellOrPass { .. } => None,
            other => panic!("{name}: unhandled decision {other:?}"),
        });
        let power = power_after_cast.unwrap_or_else(|| engine::effective_power(&state, show_off));
        assert_eq!(power, base + if noncreature { 2 } else { 0 }, "{name}");
    }
}

/// Answers a cast's and its resolution's choices with the first option,
/// aiming at player 1's things where it can; `None` at priority.
fn first_choice(s: &GameState, d: &Decision) -> Option<Action> {
    Some(match d {
        Decision::ChooseTargets { legal_targets, .. } | Decision::ChooseEffectTargets { legal_targets, .. } => {
            let theirs = |t: &&Target| match **t {
                Target::Player(p) => p == P1,
                Target::Object(o) => s.objects.get(o).controller == P1 && s.objects.get(o).zone == Zone::Battlefield,
                _ => false,
            };
            match legal_targets.iter().find(theirs).or(legal_targets.first()) {
                Some(&t) if matches!(d, Decision::ChooseTargets { .. }) => Action::ChooseTarget(t),
                Some(&t) => Action::ChooseEffectTarget(t),
                None => Action::FinishEffectSelection,
            }
        }
        Decision::ChooseSpellMode { .. } => Action::ChooseSpellMode(0),
        Decision::ChooseEffectOption { .. } => Action::ChooseEffectOption(0),
        Decision::ChooseCastMode { .. } => Action::ChooseCastMode(engine::CastMode::Normal),
        Decision::ChooseCostTargets { candidates, .. } => Action::ChooseCostTarget(candidates[0]),
        Decision::Discard { choices, .. } => Action::Discard(vec![choices[0]]),
        Decision::ChooseRiot { .. } => Action::ChooseRiot(true),
        Decision::ChooseCopy { .. } => Action::ChooseCopy(None),
        Decision::ChooseColor { .. } => Action::ChooseColor(ManaColor::R),
        Decision::ChooseEffectBoolean { .. } => Action::ChooseEffectBoolean(false),
        Decision::ChooseOptionalCost { .. } => Action::ChooseOptionalCost(engine::OptionalCostChoice::Decline),
        Decision::OrderTriggers { pending, .. } => Action::OrderTriggers((0..pending.len()).collect()),
        Decision::CastSpellOrPass { .. } => return None,
        other => panic!("unhandled decision {other:?}"),
    })
}

#[test]
fn flashback_lets_each_instant_and_sorcery_be_cast_from_the_graveyard_for_its_mana_cost_then_exiled() {
    // Flashback's granted cost equals the card's mana cost (X is 0); the card
    // is exiled as it leaves the stack.
    use mtg_kernel::card_def::{CardType, CARD_DEFS};
    use mtg_kernel::state::CastMethodV4;
    let skipped = ["Deflecting Swat", "Redirect Lightning"];
    let spells: Vec<&str> = ognis_deck_spells()
        .into_iter()
        .filter(|&n| {
            let types = CARD_DEFS[id(n) as usize].types;
            (types.contains(&CardType::Instant) || types.contains(&CardType::Sorcery)) && !skipped.contains(&n)
        })
        .collect();
    for name in spells {
        let def = &CARD_DEFS[id(name) as usize];
        let cost = usize::from(def.cost.generic) + def.cost.pips.len();
        for lands_count in [cost, cost - 1] {
            let mut state = game();
            put(&mut state, P1, "Sol Ring");
            put(&mut state, P1, "Guttersnipe");
            put_in(&mut state, P0, "Lightning Bolt", Zone::Graveyard);
            put_in(&mut state, P0, "Mountain", Zone::Hand);
            let card = put_in(&mut state, P0, name, Zone::Graveyard);
            lands(&mut state, P0, 1);
            cast(&mut state, "Flashback", Some(Target::Object(card)), 0);
            lands(&mut state, P0, lands_count);
            let ok = castable(&mut state, card);
            assert_eq!(ok, lands_count == cost, "{name} with {lands_count} lands");
            if !ok {
                continue;
            }
            engine::step(&mut state, Action::CastSpell(card)).unwrap();
            drive(&mut state, |s, d| match d {
                Decision::ChooseCastMethod { .. } => Some(Action::ChooseCastMethod(CastMethodV4::Flashback)),
                Decision::CastSpellOrPass { .. } if !s.stack.is_empty() => Some(Action::Pass),
                _ => first_choice(s, d),
            });
            assert_eq!(zone(&state, card), Zone::Exile, "{name}");
        }
    }
}

#[test]
fn gornog_pumps_each_attacking_warrior_in_the_deck_and_no_other_attacker() {
    // "Attacking Warriors you control get +X/+0, where X is the number of
    // Cowards your opponents control": one Coward, made by Gornog's trigger.
    use mtg_kernel::card_def::{Subtype, CARD_DEFS};
    let creatures: Vec<&str> = ognis_deck_spells()
        .into_iter()
        .filter(|&n| CARD_DEFS[id(n) as usize].types.contains(&mtg_kernel::card_def::CardType::Creature))
        .filter(|&n| !["Gornog, the Red Reaper", "Shivan Devastator", "Gogo, Mysterious Mime"].contains(&n))
        .collect();
    let mut warriors = 0;
    for name in creatures {
        let mut state = game();
        let gornog = put(&mut state, P0, "Gornog, the Red Reaper");
        let attacker = put(&mut state, P0, name);
        let theirs = put(&mut state, P1, "Guttersnipe");
        put(&mut state, P1, "Mountain");
        let warrior = engine::has_effective_subtype(&state, attacker, Subtype::Warrior);
        warriors += usize::from(warrior);
        let base = engine::effective_power(&state, attacker);
        let mut declared = false;
        drive(&mut state, |s, d| match d {
            Decision::DeclareAttackers { .. } if !declared => {
                declared = true;
                Some(Action::DeclareAttackers(vec![gornog, attacker]))
            }
            Decision::ChooseTargets { legal_targets, .. } if legal_targets.contains(&Target::Object(theirs)) => {
                Some(Action::ChooseTarget(Target::Object(theirs)))
            }
            // Annihilator: player 1 sacrifices the Mountain.
            Decision::ChooseEffectTargets { legal_targets, .. } if legal_targets.contains(&Target::Object(theirs)) => {
                Some(Action::ChooseEffectTarget(*legal_targets.iter().find(|&&t| t != Target::Object(theirs)).unwrap()))
            }
            Decision::DeclareBlockers { .. } => None,
            Decision::CastSpellOrPass { .. } => Some(Action::Pass),
            _ => first_choice(s, d),
        });
        assert!(engine::has_effective_subtype(&state, theirs, Subtype::Coward), "{name}");
        // Other attack pumps: the Titan's melee, Jurin's +1 for player 1's one
        // creature, Laelia's counter for the card she exiles.
        let pumped = engine::effective_power(&state, attacker) - base;
        let others = i32::from(["Depthshaker Titan", "Jurin, Leading the Charge", "Laelia, the Blade Reforged"].contains(&name));
        assert_eq!(pumped - others, i32::from(warrior), "{name}");
    }
    assert_eq!(warriors, 4, "Ognis, Knuckles, Laelia, Zealous Conscripts");
}

#[test]
fn a_celebrating_goddric_pumps_each_dragon_in_the_deck_and_no_other_creature() {
    // Two Treasures entered, so Goddric is a Dragon with "{R}: Dragons you
    // control get +1/+0 until end of turn".
    use mtg_kernel::card_def::{CardType, Subtype, CARD_DEFS};
    use mtg_kernel::event::{propose_and_commit, ProposedEvent};
    let creatures: Vec<&str> = ognis_deck_spells()
        .into_iter()
        .filter(|&n| CARD_DEFS[id(n) as usize].types.contains(&CardType::Creature) && n != "Goddric, Cloaked Reveler")
        .collect();
    let mut dragons = Vec::new();
    for name in creatures {
        let mut state = game();
        let goddric = put(&mut state, P0, "Goddric, Cloaked Reveler");
        let other = put(&mut state, P0, name);
        if name == "Shivan Devastator" {
            state.objects.get_mut(other).counters.plus1_plus1 = 1;
        }
        lands(&mut state, P0, 1);
        advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
        for _ in 0..2 {
            propose_and_commit(&mut state, ProposedEvent::create_token(id("Treasure Token"), P0));
        }
        let (goddric_base, base) = (engine::effective_power(&state, goddric), engine::effective_power(&state, other));
        let dragon = engine::has_effective_subtype(&state, other, Subtype::Dragon);
        activate(&mut state, goddric, 0, None);
        assert_eq!(engine::effective_power(&state, goddric), goddric_base + 1, "{name}: Goddric is a Dragon");
        assert_eq!(engine::effective_power(&state, other), base + i32::from(dragon), "{name}");
        if dragon {
            dragons.push(name);
        }
    }
    assert_eq!(
        dragons,
        ["Cavern-Hoard Dragon", "Goldspan Dragon", "Hellkite Charger", "Shivan Devastator", "Smaug the Magnificent"]
    );
}

#[test]
fn attack_triggers_fire_again_in_hellkite_chargers_extra_combat() {
    // 500.8, 508.3a: Hellkite Charger, Captain Lannery Storm and Goldspan
    // Dragon attack in both combats; Ognis makes a tapped Treasure for each
    // hasty attacker each time, Lannery and Goldspan an untapped one each.
    // The extra combat is paid with floating Mountain mana.
    let mut state = game();
    put(&mut state, P0, "Ognis, the Dragon's Lash");
    let attackers = vec![
        put(&mut state, P0, "Hellkite Charger"),
        put(&mut state, P0, "Captain Lannery Storm"),
        put(&mut state, P0, "Goldspan Dragon"),
    ];
    let mountains: Vec<ObjectId> = (0..7).map(|_| put(&mut state, P0, "Mountain")).collect();
    let (mut combats, mut paid) = (0, false);
    drive(&mut state, |s, d| match d {
        Decision::DeclareAttackers { .. } if combats < 2 => {
            combats += 1;
            Some(Action::DeclareAttackers(attackers.clone()))
        }
        // Float every Mountain while the first combat's triggers wait.
        Decision::CastSpellOrPass { player: P0, .. } if s.step == Step::DeclareAttackers && !s.stack.is_empty() && !paid => {
            match mountains.iter().copied().find(|&m| !s.objects.get(m).tapped) {
                Some(m) => Some(Action::ActivateManaAbility(m)),
                None => Some(Action::Pass),
            }
        }
        Decision::ChooseEffectBoolean { .. } => {
            let pay = !paid;
            paid = true;
            Some(Action::ChooseEffectBoolean(pay))
        }
        Decision::ChooseTargets { legal_targets, .. } => Some(Action::ChooseTarget(legal_targets[0])),
        Decision::DeclareBlockers { .. } => Some(Action::DeclareBlockers(vec![])),
        Decision::OrderTriggers { pending, .. } => Some(Action::OrderTriggers((0..pending.len()).collect())),
        Decision::CastSpellOrPass { .. } if s.step == Step::Main2 => None,
        Decision::CastSpellOrPass { .. } => Some(Action::Pass),
        _ => None,
    });
    assert_eq!(combats, 2);
    let treasures = named(&state, P0, "Treasure Token");
    let tapped = treasures.iter().filter(|&&t| state.objects.get(t).tapped).count();
    assert_eq!((tapped, treasures.len() - tapped), (6, 4), "Ognis's, then Lannery's and Goldspan's, twice");
}

#[test]
fn light_up_the_stages_spectacle_follows_the_opponents_life_loss_from_each_source_not_your_own() {
    // 702.137a: Delayed Blast Fireball's and Dragonspark Reactor's damage to
    // player 1 enable it; paying War Room's life yourself doesn't. Spectacle
    // {R} is castable with the one Mountain left.
    for source in ["Delayed Blast Fireball", "Dragonspark Reactor", "War Room"] {
        let mut state = game();
        match source {
            "Delayed Blast Fireball" => {
                lands(&mut state, P0, 3);
                cast(&mut state, source, None, 0);
            }
            "Dragonspark Reactor" => {
                let reactor = put(&mut state, P0, source);
                state.objects.get_mut(reactor).counters.charge = 1;
                lands(&mut state, P0, 4);
                activate(&mut state, reactor, 0, Some(Target::Player(P1)));
            }
            _ => {
                let room = put(&mut state, P0, source);
                lands(&mut state, P0, 3);
                activate(&mut state, room, 0, None);
                assert_eq!(state.players[0].life, 37);
            }
        }
        let lost = state.players[1].life < 40;
        assert_eq!(lost, source != "War Room", "{source}");
        lands(&mut state, P0, 1);
        let stage = put_in(&mut state, P0, "Light Up the Stage", Zone::Hand);
        assert_eq!(castable(&mut state, stage), lost, "{source}");
    }
}

#[test]
fn mob_rule_reads_each_creatures_power_with_counters_and_base_changes() {
    // Mode 1 takes power 4 or greater, mode 2 power 3 or less, read as it
    // resolves: Laelia 2/2 with two +1/+1 counters is 4; a celebrating
    // Goddric is a 4/4, otherwise 3/3.
    use mtg_kernel::event::{propose_and_commit, ProposedEvent};
    for (card, celebrate, counters, power) in [
        ("Laelia, the Blade Reforged", false, 2, 4),
        ("Laelia, the Blade Reforged", false, 0, 2),
        ("Goddric, Cloaked Reveler", true, 0, 4),
        ("Goddric, Cloaked Reveler", false, 0, 3),
    ] {
        for mode in [0u8, 1] {
            let mut state = game();
            let theirs = put(&mut state, P1, card);
            state.objects.get_mut(theirs).counters.plus1_plus1 = counters;
            if celebrate {
                for _ in 0..2 {
                    propose_and_commit(&mut state, ProposedEvent::create_token(id("Treasure Token"), P1));
                }
            }
            assert_eq!(engine::effective_power(&state, theirs), power, "{card}");
            lands(&mut state, P0, 6);
            cast(&mut state, "Mob Rule", None, mode);
            let taken = state.objects.get(theirs).controller == P0;
            assert_eq!(taken, (mode == 0) == (power >= 4), "{card} at power {power}, mode {mode}");
        }
    }
}

#[test]
fn pyreswipe_hawk_gets_the_greatest_mana_value_among_your_artifacts() {
    // A Treasure is 0, Sol Ring 1, Coin of Mastery 4, Throne of Eldraine 5;
    // Zealous Conscripts made an artifact by Liquimetal Torque is 5 (the
    // Torque itself is 2); the opponent's artifacts don't count.
    for (artifact, x) in [("Treasure Token", 0), ("Sol Ring", 1), ("Coin of Mastery", 4), ("Throne of Eldraine", 5), ("Zealous Conscripts", 5)] {
        let mut state = game();
        let hawk = put(&mut state, P0, "Pyreswipe Hawk");
        let o = put(&mut state, P0, artifact);
        put(&mut state, P1, "Throne of Eldraine");
        if artifact == "Zealous Conscripts" {
            let torque = put(&mut state, P0, "Liquimetal Torque");
            activate(&mut state, torque, 0, Some(Target::Object(o)));
        }
        let base = engine::effective_power(&state, hawk);
        attack_with(&mut state, vec![hawk]);
        assert_eq!(engine::effective_power(&state, hawk), base + x, "{artifact}");
    }
}

#[test]
fn screaming_nemesis_returns_damage_from_each_kind_of_source() {
    // Player 0's Delayed Blast Fireball (2), Dragonspark Reactor (1) and
    // Smaug's attack trigger (2 Treasures) damage player 1's Nemesis, which
    // deals as much to player 0.
    for (source, damage) in [("Delayed Blast Fireball", 2), ("Dragonspark Reactor", 1), ("Smaug the Magnificent", 2)] {
        let mut state = game();
        let nemesis = put(&mut state, P1, "Screaming Nemesis");
        let at_nemesis = |s: &GameState, d: &Decision| match d {
            Decision::ChooseTargets { player: P1, .. } => Some(Action::ChooseTarget(Target::Player(P0))),
            Decision::ChooseTargets { legal_targets, .. } if legal_targets.contains(&Target::Object(nemesis)) => {
                Some(Action::ChooseTarget(Target::Object(nemesis)))
            }
            Decision::ChooseTargets { .. } => Some(Action::ChooseTarget(Target::Player(P1))),
            Decision::DeclareBlockers { .. } => Some(Action::DeclareBlockers(vec![])),
            Decision::CastSpellOrPass { .. } if s.stack.is_empty() && matches!(s.step, Step::Main1 | Step::Main2) => None,
            Decision::CastSpellOrPass { .. } => Some(Action::Pass),
            _ => first_choice(s, d),
        };
        match source {
            "Delayed Blast Fireball" => {
                lands(&mut state, P0, 3);
                let fireball = put_in(&mut state, P0, source, Zone::Hand);
                advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
                engine::step(&mut state, Action::CastSpell(fireball)).unwrap();
            }
            "Dragonspark Reactor" => {
                let reactor = put(&mut state, P0, source);
                state.objects.get_mut(reactor).counters.charge = 1;
                lands(&mut state, P0, 4);
                advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
                engine::step(&mut state, Action::ActivateAbility(reactor, 0)).unwrap();
            }
            _ => {
                let smaug = put(&mut state, P0, source);
                put(&mut state, P0, "Treasure Token");
                put(&mut state, P0, "Treasure Token");
                advance(&mut state, |_, d| matches!(d, Decision::DeclareAttackers { .. }));
                engine::step(&mut state, Action::DeclareAttackers(vec![smaug])).unwrap();
            }
        }
        drive(&mut state, at_nemesis);
        assert_eq!(state.objects.get(nemesis).damage, damage as u16, "{source}");
        assert_eq!(40 - state.players[0].life, damage, "{source}");
    }
}

#[test]
fn hobgoblin_grows_for_the_card_each_discard_source_discards() {
    // Demand Answers' additional cost, Vision of Love's optional discard and
    // Embrace the Unknown's retrace each discard one card: +2/+0.
    use mtg_kernel::state::CastMethodV4;
    for source in ["Demand Answers", "Vision of Love", "Embrace the Unknown"] {
        let mut state = game();
        let hobgoblin = put(&mut state, P0, "Hobgoblin, Mantled Marauder");
        let base = engine::effective_power(&state, hobgoblin);
        put_in(&mut state, P0, "Mountain", Zone::Hand);
        lands(&mut state, P0, 3);
        let card = put_in(&mut state, P0, source, if source == "Embrace the Unknown" { Zone::Graveyard } else { Zone::Hand });
        advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
        engine::step(&mut state, Action::CastSpell(card)).unwrap();
        drive(&mut state, |s, d| match d {
            Decision::ChooseCastMethod { .. } => Some(Action::ChooseCastMethod(CastMethodV4::Retrace)),
            Decision::ChooseOptionalCost { .. } => Some(Action::ChooseOptionalCost(engine::OptionalCostChoice::Discard)),
            Decision::CastSpellOrPass { .. } if !s.stack.is_empty() => Some(Action::Pass),
            _ => first_choice(s, d),
        });
        assert_eq!(state.players[0].graveyard.iter().filter(|&&o| state.objects.get(o).name == "Mountain").count(), 1, "{source}: discarded");
        assert_eq!(engine::effective_power(&state, hobgoblin), base + 2, "{source}");
    }
}

#[test]
fn jurin_must_be_blocked_only_by_a_creature_able_to_block_it() {
    // 509.1c: player 1's one creature must block Jurin unless a restriction
    // makes it unable: Untimely Malfunction's third mode, Headliner
    // Scarlett, Spider-Man's attack trigger, or Goblin War Drums' menace.
    for restriction in ["none", "Untimely Malfunction", "Headliner Scarlett", "Spider-Man, Web-Spinner", "Goblin War Drums"] {
        let mut state = game();
        let jurin = put(&mut state, P0, "Jurin, Leading the Charge");
        let theirs = put(&mut state, P1, "Zealous Conscripts");
        let mut attackers = vec![jurin];
        match restriction {
            "Untimely Malfunction" => {
                lands(&mut state, P0, 2);
                let spell = put_in(&mut state, P0, restriction, Zone::Hand);
                advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
                engine::step(&mut state, Action::CastSpell(spell)).unwrap();
                let mut picked = false;
                drive(&mut state, |s, d| match d {
                    Decision::ChooseSpellMode { .. } => Some(Action::ChooseSpellMode(2)),
                    Decision::ChooseTargets { .. } if !picked => {
                        picked = true;
                        Some(Action::ChooseTarget(Target::Object(theirs)))
                    }
                    Decision::ChooseTargets { .. } => Some(Action::FinishEffectSelection),
                    Decision::CastSpellOrPass { .. } if !s.stack.is_empty() => Some(Action::Pass),
                    _ => None,
                });
            }
            "Headliner Scarlett" => {
                lands(&mut state, P0, 4);
                cast(&mut state, restriction, Some(Target::Player(P1)), 0);
            }
            "Spider-Man, Web-Spinner" => attackers.push(put(&mut state, P0, restriction)),
            "Goblin War Drums" => {
                put(&mut state, P0, restriction);
            }
            _ => {}
        }
        advance(&mut state, |_, d| matches!(d, Decision::DeclareAttackers { .. }));
        engine::step(&mut state, Action::DeclareAttackers(attackers)).unwrap();
        drive(&mut state, |s, d| match d {
            Decision::ChooseTargets { legal_targets, .. } if legal_targets.contains(&Target::Object(theirs)) => {
                Some(Action::ChooseTarget(Target::Object(theirs)))
            }
            Decision::DeclareBlockers { .. } => None,
            Decision::CastSpellOrPass { .. } => Some(Action::Pass),
            _ => first_choice(s, d),
        });
        let unblocked = engine::step(&mut state, Action::DeclareBlockers(vec![]));
        assert_eq!(unblocked.is_ok(), restriction != "none", "{restriction}: {unblocked:?}");
    }
}

#[test]
fn laelia_gets_a_counter_per_batch_exiled_from_the_library_or_graveyard_not_from_the_hand_or_stack() {
    // Reckless Impulse, Light Up the Stage and Embrace the Unknown exile two
    // cards from the library at once: one counter. A foretold card leaves the
    // hand, and a flashback spell the stack: none.
    for source in ["Reckless Impulse", "Light Up the Stage", "Embrace the Unknown", "Delayed Blast Fireball", "Flashback"] {
        let mut state = game();
        let laelia = put(&mut state, P0, "Laelia, the Blade Reforged");
        lands(&mut state, P0, 3);
        match source {
            "Delayed Blast Fireball" => {
                let fireball = put_in(&mut state, P0, source, Zone::Hand);
                advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
                engine::step(&mut state, Action::Foretell(fireball)).unwrap();
                assert_eq!(zone(&state, fireball), Zone::Exile);
            }
            "Flashback" => {
                let bolt = put_in(&mut state, P0, "Lightning Bolt", Zone::Graveyard);
                cast(&mut state, source, Some(Target::Object(bolt)), 0);
                cast_from_graveyard(&mut state, bolt, None);
                assert_eq!(zone(&state, bolt), Zone::Exile);
            }
            _ => {
                cast(&mut state, source, None, 0);
            }
        }
        let expected = i16::from(!["Delayed Blast Fireball", "Flashback"].contains(&source));
        assert_eq!(state.objects.get(laelia).counters.plus1_plus1, expected, "{source}");
    }
}

#[test]
fn a_treasure_feeds_dragonspark_idol_lannery_and_coin_of_mastery() {
    // Captain Lannery Storm's attack makes a Treasure: an artifact entering
    // (Dragonspark Reactor), a token created this turn (Idol of Oblivion),
    // sacrificed (Lannery's +1/+0), and artifact mana spent on a creature
    // spell (Coin of Mastery).
    let mut state = game();
    let lannery = put(&mut state, P0, "Captain Lannery Storm");
    let reactor = put(&mut state, P0, "Dragonspark Reactor");
    let idol = put(&mut state, P0, "Idol of Oblivion");
    put(&mut state, P0, "Coin of Mastery");
    lands(&mut state, P0, 1);
    attack_with(&mut state, vec![lannery]);
    advance(&mut state, |s, d| s.step == Step::Main2 && matches!(d, Decision::CastSpellOrPass { .. }));
    let treasure = named(&state, P0, "Treasure Token")[0];
    assert_eq!(state.objects.get(reactor).counters.charge, 1, "an artifact entered");
    assert!(offered_ability(&mut state, idol, 0), "a token was created this turn");
    let base = engine::effective_power(&state, lannery);
    engine::step(&mut state, Action::ActivateManaAbilityChoice(treasure, ManaColor::R)).unwrap();
    // Lannery's trigger resolves; the red stays floating in this step.
    drive(&mut state, |s, d| match d {
        Decision::CastSpellOrPass { .. } if !s.stack.is_empty() => Some(Action::Pass),
        _ => None,
    });
    assert_eq!(engine::effective_power(&state, lannery), base + 1, "a Treasure was sacrificed");
    assert_eq!(state.artifact_mana[0][ManaColor::R.pool_index()], 1);
    assert!(offered_ability(&mut state, idol, 0), "602.5b: the token was created this turn, though gone");
    let show_off = put_in(&mut state, P0, "Slickshot Show-Off", Zone::Hand);
    engine::step(&mut state, Action::CastSpell(show_off)).unwrap();
    drive(&mut state, |s, d| match d {
        Decision::CastSpellOrPass { .. } if !s.stack.is_empty() => Some(Action::Pass),
        _ => first_choice(s, d),
    });
    assert_eq!(state.objects.get(show_off).counters.plus1_plus1, 1, "one Treasure mana spent");
}

#[test]
fn urabrask_replaces_only_the_opponents_next_draw_even_outside_the_draw_step() {
    // 614.5: player 1 casts Demand Answers in its upkeep after Urabrask's
    // trigger: the first of its two draws becomes an exile, the second and
    // the draw step's draw are normal.
    let mut state = game();
    put(&mut state, P0, "Urabrask, Heretic Praetor");
    lands(&mut state, P1, 2);
    let answers = put_in(&mut state, P1, "Demand Answers", Zone::Hand);
    put_in(&mut state, P1, "Mountain", Zone::Hand);
    advance(&mut state, |s, d| {
        s.active_player == P1 && s.step == Step::Upkeep && s.stack.is_empty() && matches!(d, Decision::CastSpellOrPass { player: P1, .. })
    });
    let exiled_before = state.exile.len();
    engine::step(&mut state, Action::CastSpell(answers)).unwrap();
    drive(&mut state, |s, d| match d {
        Decision::CastSpellOrPass { .. } if !s.stack.is_empty() => Some(Action::Pass),
        _ => first_choice(s, d),
    });
    assert_eq!(state.exile.len(), exiled_before + 1, "the first draw was exiled");
    assert_eq!(state.players[1].hand.len(), 1, "the second draw is normal");
    advance(&mut state, |s, _| s.step == Step::Main1);
    assert_eq!(state.players[1].hand.len(), 2, "the draw step's draw is normal");
    assert_eq!(state.exile.len(), exiled_before + 1);
}

#[test]
fn the_land_destroyers_target_each_land_in_the_deck_demolition_field_only_nonbasic_ones() {
    // Strip Mine ("target land"), Wild Magic Surge ("target permanent an
    // opponent controls") and Demolition Field ("target nonbasic land an
    // opponent controls") against each of player 1's lands.
    use mtg_kernel::card_def::{CardType, Supertype, CARD_DEFS};
    let lands_in_deck: Vec<&str> = include_str!("../../configs/commander/ognis.txt")
        .lines()
        .filter_map(|l| l.split_once(' ').map(|(_, n)| n.trim()))
        .filter(|&n| mtg_kernel::card_def::card_id_by_name(n).is_some_and(|c| CARD_DEFS[c as usize].types.contains(&CardType::Land)))
        .collect();
    assert_eq!(lands_in_deck.len(), 9, "{lands_in_deck:?}");
    // Starts activating `source`'s ability 0 and returns its first target
    // choice; none if it isn't offered (no legal target).
    let ability_targets = |state: &mut GameState, source: ObjectId| {
        if !offered_ability(state, source, 0) {
            return Vec::new();
        }
        engine::step(state, Action::ActivateAbility(source, 0)).unwrap();
        match engine::advance_until_decision(state) {
            Decision::ChooseTargets { legal_targets, .. } => legal_targets,
            other => panic!("{other:?}"),
        }
    };
    for land in lands_in_deck {
        let basic = CARD_DEFS[id(land) as usize].supertypes.contains(&Supertype::Basic);
        for destroyer in ["Strip Mine", "Demolition Field", "Wild Magic Surge"] {
            let mut state = game();
            let theirs = put(&mut state, P1, land);
            lands(&mut state, P0, 2);
            let offered = if destroyer == "Wild Magic Surge" {
                targets_offered(&mut state, destroyer, 0)
            } else {
                let source = put(&mut state, P0, destroyer);
                ability_targets(&mut state, source)
            };
            let expected = destroyer != "Demolition Field" || !basic;
            assert_eq!(offered.contains(&Target::Object(theirs)), expected, "{destroyer} at {land}");
        }
    }
}

#[test]
fn abstergo_returns_exactly_the_historic_cards_in_the_deck() {
    // 700.6: artifacts, legendaries and Sagas are historic.
    use mtg_kernel::card_def::{CardType, Subtype, Supertype, CARD_DEFS};
    let mut historic_count = 0;
    for name in ognis_deck_spells().into_iter().filter(|&n| n != "Ognis, the Dragon's Lash") {
        let def = &CARD_DEFS[id(name) as usize];
        let historic = def.types.contains(&CardType::Artifact)
            || def.supertypes.contains(&Supertype::Legendary)
            || def.subtypes.contains(&Subtype::Saga);
        historic_count += usize::from(historic);
        let mut state = game();
        let abstergo = put(&mut state, P0, "Abstergo Entertainment");
        lands(&mut state, P0, 3);
        let card = put_in(&mut state, P0, name, Zone::Graveyard);
        assert!(offered_ability(&mut state, abstergo, 0), "{name}");
        engine::step(&mut state, Action::ActivateAbility(abstergo, 0)).unwrap();
        let offered = match engine::advance_until_decision(&mut state) {
            Decision::ChooseTargets { legal_targets, .. } | Decision::ChooseEffectTargets { legal_targets, .. } => legal_targets,
            _ => Vec::new(),
        };
        assert_eq!(offered.contains(&Target::Object(card)), historic, "{name}");
    }
    assert!(historic_count >= 25, "{historic_count}");
}

#[test]
fn exiling_the_graveyards_ends_flashback_and_retrace() {
    // Abstergo Entertainment or Scavenger Grounds exiles all graveyards: a
    // Reckless Impulse given flashback and an Embrace the Unknown with
    // retrace are gone and can no longer be cast.
    for exiler in ["Abstergo Entertainment", "Scavenger Grounds"] {
        let mut state = game();
        let impulse = put_in(&mut state, P0, "Reckless Impulse", Zone::Graveyard);
        let embrace = put_in(&mut state, P0, "Embrace the Unknown", Zone::Graveyard);
        put_in(&mut state, P0, "Mountain", Zone::Hand);
        lands(&mut state, P0, 1);
        cast(&mut state, "Flashback", Some(Target::Object(impulse)), 0);
        lands(&mut state, P0, 6);
        assert!(castable(&mut state, impulse) && castable(&mut state, embrace), "{exiler}");
        let source = put(&mut state, P0, exiler);
        assert!(offered_ability(&mut state, source, 0), "{exiler}");
        engine::step(&mut state, Action::ActivateAbility(source, 0)).unwrap();
        drive(&mut state, |s, d| match d {
            Decision::ChooseTargets { .. } | Decision::ChooseEffectTargets { .. } => Some(Action::FinishEffectSelection),
            Decision::CastSpellOrPass { .. } if !s.stack.is_empty() => Some(Action::Pass),
            _ => first_choice(s, d),
        });
        assert_eq!((zone(&state, impulse), zone(&state, embrace)), (Zone::Exile, Zone::Exile), "{exiler}");
        assert!(!castable(&mut state, impulse) && !castable(&mut state, embrace), "{exiler}");
    }
}

#[test]
fn call_for_aids_creatures_cant_pay_demand_answers_sacrifice() {
    // A stolen Depthshaker Titan is an artifact player 0 controls: taken with
    // Zealous Conscripts it may be sacrificed for Demand Answers; taken with
    // Call for Aid it can't, so Sol Ring is the only choice and is taken
    // without asking.
    for (steal, target) in [("Zealous Conscripts", None), ("Call for Aid", Some(Target::Player(P1)))] {
        let mut state = game();
        let titan = put(&mut state, P1, "Depthshaker Titan");
        let ring = put(&mut state, P0, "Sol Ring");
        lands(&mut state, P0, 7);
        cast(&mut state, steal, Some(target.unwrap_or(Target::Object(titan))), 0);
        assert_eq!(state.objects.get(titan).controller, P0, "{steal}");
        let answers = put_in(&mut state, P0, "Demand Answers", Zone::Hand);
        put_in(&mut state, P0, "Mountain", Zone::Hand);
        advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
        engine::step(&mut state, Action::CastSpell(answers)).unwrap();
        let mut offered = None;
        drive(&mut state, |s, d| match d {
            Decision::ChooseEffectOption { .. } => Some(Action::ChooseEffectOption(0)),
            Decision::ChooseCostTargets { candidates, .. } => {
                offered = Some(candidates.clone());
                Some(Action::ChooseCostTarget(ring))
            }
            Decision::CastSpellOrPass { .. } if !s.stack.is_empty() => Some(Action::Pass),
            _ => first_choice(s, d),
        });
        match steal {
            "Zealous Conscripts" => assert_eq!(offered.map(|mut c| { c.sort(); c }), Some(vec![titan, ring])),
            _ => assert_eq!(offered, None, "only Sol Ring"),
        }
        assert_eq!((zone(&state, ring), zone(&state, titan)), (Zone::Graveyard, Zone::Battlefield), "{steal}");
    }
}

#[test]
fn delayed_blast_fireball_deals_five_when_cast_from_exile_by_any_route() {
    // From hand: 2 to player 1. Foretold, or exiled by Reckless Impulse and
    // cast from there: 5.
    for route in ["hand", "foretell", "Reckless Impulse"] {
        let mut state = game();
        let fireball = match route {
            "hand" => put_in(&mut state, P0, "Delayed Blast Fireball", Zone::Hand),
            "foretell" => {
                let f = put_in(&mut state, P0, "Delayed Blast Fireball", Zone::Hand);
                lands(&mut state, P0, 2);
                advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
                engine::step(&mut state, Action::Foretell(f)).unwrap();
                end_turn(&mut state);
                end_turn(&mut state);
                f
            }
            _ => {
                let f = library_card(&mut state, P0, 0, "Delayed Blast Fireball");
                lands(&mut state, P0, 2);
                cast(&mut state, "Reckless Impulse", None, 0);
                assert_eq!(zone(&state, f), Zone::Exile);
                f
            }
        };
        lands(&mut state, P0, 6);
        advance(&mut state, |s, d| s.active_player == P0 && matches!(d, Decision::CastSpellOrPass { castable_spells, .. } if castable_spells.contains(&fireball)));
        let life = state.players[1].life;
        engine::step(&mut state, Action::CastSpell(fireball)).unwrap();
        drive(&mut state, |s, d| match d {
            Decision::ChooseCastMethod { methods, .. } => Some(Action::ChooseCastMethod(methods[0])),
            Decision::CastSpellOrPass { .. } if !s.stack.is_empty() => Some(Action::Pass),
            _ => first_choice(s, d),
        });
        assert_eq!(life - state.players[1].life, if route == "hand" { 2 } else { 5 }, "{route}");
    }
}

#[test]
fn a_celebrating_goddric_keeps_its_4_4_and_flying_through_other_type_changes() {
    // 613.1d, 613.4b, 613.7: Gornog's Coward (newer) replaces Goddric's
    // Dragon type but not his base 4/4 or flying; Liquimetal Torque adds the
    // artifact type and he stays a 4/4 Dragon creature.
    use mtg_kernel::card_def::{CardType, Keywords, Subtype};
    use mtg_kernel::event::{propose_and_commit, ProposedEvent};
    let celebrate = |state: &mut GameState, p| {
        for _ in 0..2 {
            propose_and_commit(state, ProposedEvent::create_token(id("Treasure Token"), p));
        }
    };
    let mut state = game();
    let gornog = put(&mut state, P0, "Gornog, the Red Reaper");
    let goddric = put(&mut state, P1, "Goddric, Cloaked Reveler");
    celebrate(&mut state, P1);
    attack_and_see_blocks(&mut state, vec![gornog], Target::Object(goddric));
    assert!(engine::has_effective_subtype(&state, goddric, Subtype::Coward));
    assert!(!engine::has_effective_subtype(&state, goddric, Subtype::Dragon));
    assert_eq!((engine::effective_power(&state, goddric), engine::effective_toughness(&state, goddric)), (4, 4));
    assert!(engine::has_effective_keyword(&state, goddric, Keywords::FLYING));

    let mut state = game();
    let goddric = put(&mut state, P0, "Goddric, Cloaked Reveler");
    let torque = put(&mut state, P0, "Liquimetal Torque");
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    celebrate(&mut state, P0);
    activate(&mut state, torque, 0, Some(Target::Object(goddric)));
    for t in [CardType::Artifact, CardType::Creature] {
        assert!(engine::object_has_type(&state, goddric, t), "{t:?}");
    }
    assert!(engine::has_effective_subtype(&state, goddric, Subtype::Dragon));
    assert_eq!((engine::effective_power(&state, goddric), engine::effective_toughness(&state, goddric)), (4, 4));
}

#[test]
fn chain_gogo_becoming_a_copy_with_coin_of_mastery_out_gets_no_counters() {
    // K13: becoming a copy isn't entering the battlefield (614.12).
    let mut state = game();
    put(&mut state, P0, "Coin of Mastery");
    let gogo = put(&mut state, P0, "Gogo, Mysterious Mime");
    let dragon = put(&mut state, P0, "Goldspan Dragon");
    advance(&mut state, |s, d| s.step == Step::BeginCombat && matches!(d, Decision::ChooseTargets { .. }));
    engine::step(&mut state, Action::ChooseTarget(Target::Object(dragon))).unwrap();
    drive(&mut state, |s, d| match d {
        Decision::ChooseEffectOption { option_count: 2, .. } => Some(Action::ChooseEffectOption(1)),
        Decision::CastSpellOrPass { .. } if !s.stack.is_empty() => Some(Action::Pass),
        _ => None,
    });
    assert_eq!(state.objects.get(gogo).card_def, id("Goldspan Dragon"));
    assert_eq!(state.objects.get(gogo).counters.plus1_plus1, 0);
}

#[test]
fn chain_a_cursed_mirror_copying_an_animated_sol_ring_is_a_plain_sol_ring() {
    // K8, 707.2: Depthshaker Titan's animation isn't copiable, so the Mirror
    // copies Sol Ring's printed values: a noncreature artifact.
    use mtg_kernel::card_def::CardType;
    let mut state = game();
    put(&mut state, P0, "Depthshaker Titan");
    let ring = put(&mut state, P0, "Sol Ring");
    animate(&mut state, ring);
    lands(&mut state, P0, 3);
    let mirror = put_in(&mut state, P0, "Cursed Mirror", Zone::Hand);
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    engine::step(&mut state, Action::CastSpell(mirror)).unwrap();
    let mut offered = Vec::new();
    drive(&mut state, |s, d| match d {
        Decision::ChooseCopy { creatures, .. } => {
            offered = creatures.clone();
            Some(Action::ChooseCopy(Some(ring)))
        }
        Decision::CastSpellOrPass { .. } if !s.stack.is_empty() => Some(Action::Pass),
        _ => first_choice(s, d),
    });
    assert!(offered.contains(&ring), "an animated Sol Ring is a creature to copy");
    assert_eq!(state.objects.get(mirror).card_def, id("Sol Ring"));
    assert!(!engine::object_has_type(&state, mirror, CardType::Creature));
}

#[test]
fn chain_chaos_warp_on_a_stolen_permanent_returns_it_to_its_owner() {
    // K2: Zealous Conscripts takes player 1's Laelia; Chaos Warp shuffles her
    // into player 1's library and reveals its top card: a Zealous Conscripts
    // enters under player 1 with player 1's enters trigger, or Laelia herself
    // comes back under player 1. Over seeds, both happen.
    let mut outcomes = [false; 2];
    for seed in 0..64u64 {
        let mut state = game();
        state.reseed_shuffles(seed);
        let laelia = put(&mut state, P1, "Laelia, the Blade Reforged");
        for i in 0..10 {
            library_card(&mut state, P1, i, "Zealous Conscripts");
        }
        lands(&mut state, P0, 8);
        cast(&mut state, "Zealous Conscripts", Some(Target::Object(laelia)), 0);
        let warp = put_in(&mut state, P0, "Chaos Warp", Zone::Hand);
        advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
        engine::step(&mut state, Action::CastSpell(warp)).unwrap();
        let mut trigger_controller = None;
        drive(&mut state, |s, d| match d {
            Decision::ChooseTargets { player: P0, .. } if zone(s, warp) == Zone::Stack && s.stack.len() == 1 => {
                Some(Action::ChooseTarget(Target::Object(laelia)))
            }
            Decision::ChooseTargets { player, legal_targets, .. } => {
                trigger_controller = Some(*player);
                Some(Action::ChooseTarget(legal_targets[0]))
            }
            Decision::CastSpellOrPass { .. } if !s.stack.is_empty() => Some(Action::Pass),
            _ => first_choice(s, d),
        });
        if zone(&state, laelia) == Zone::Battlefield {
            outcomes[0] = true;
            assert_eq!(state.objects.get(laelia).controller, P1, "seed {seed}");
        } else {
            outcomes[1] = true;
            assert!(state.players[1].library.contains(&laelia), "seed {seed}: into its owner's library");
            assert_eq!(named(&state, P1, "Zealous Conscripts").len(), 1, "seed {seed}");
            assert_eq!(trigger_controller, Some(P1), "seed {seed}");
        }
    }
    assert_eq!(outcomes, [true, true]);
}

#[test]
fn mirror_two_screaming_nemeses_in_one_combat_both_trigger_and_hit_the_other_player() {
    // M3, 603.3b: player 0's Nemesis is blocked by player 1's; both are
    // dealt 3 at once and each trigger aims at the other player.
    let mut state = game();
    let mine = put(&mut state, P0, "Screaming Nemesis");
    let theirs = put(&mut state, P1, "Screaming Nemesis");
    let mut triggers = Vec::new();
    let mut declared = (false, false);
    drive(&mut state, |s, d| match d {
        Decision::DeclareAttackers { .. } if !declared.0 => {
            declared.0 = true;
            Some(Action::DeclareAttackers(vec![mine]))
        }
        Decision::DeclareBlockers { .. } if !declared.1 => {
            declared.1 = true;
            Some(Action::DeclareBlockers(vec![(theirs, mine)]))
        }
        Decision::ChooseTargets { player, .. } => {
            triggers.push(*player);
            Some(Action::ChooseTarget(Target::Player(player.opponent())))
        }
        Decision::CastSpellOrPass { .. } if matches!(s.step, Step::EndCombat | Step::Main2) && s.stack.is_empty() => None,
        Decision::CastSpellOrPass { .. } => Some(Action::Pass),
        _ => first_choice(s, d),
    });
    triggers.sort_by_key(|p| p.index());
    assert_eq!(triggers, [P0, P1]);
    assert_eq!((state.players[0].life, state.players[1].life), (37, 37));
    assert!(state.cant_gain_life[0] && state.cant_gain_life[1]);
}

#[test]
fn mirror_each_spider_punk_gives_riot_only_to_its_controllers_spiders() {
    // M1, 613.1f: one Spider-Punk each; player 0's Spider-Man gets one riot
    // choice, from player 0's Spider-Punk only.
    let mut state = game();
    put(&mut state, P0, "Spider-Punk");
    put(&mut state, P1, "Spider-Punk");
    lands(&mut state, P0, 4);
    let spider_man = put_in(&mut state, P0, "Spider-Man, Web-Spinner", Zone::Hand);
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    engine::step(&mut state, Action::CastSpell(spider_man)).unwrap();
    let mut riots = 0;
    drive(&mut state, |s, d| match d {
        Decision::ChooseRiot { player, .. } => {
            assert_eq!(*player, P0);
            riots += 1;
            Some(Action::ChooseRiot(false))
        }
        Decision::CastSpellOrPass { .. } if !s.stack.is_empty() => Some(Action::Pass),
        _ => first_choice(s, d),
    });
    assert_eq!(riots, 1);
    assert_eq!(state.objects.get(spider_man).counters.plus1_plus1, 1);
}

/// Player 0 takes player 1's `card` with Zealous Conscripts (five Mountains,
/// all tapped after).
fn steal(state: &mut GameState, card: &str) -> ObjectId {
    let theirs = put(state, P1, card);
    lands(state, P0, 5);
    cast(state, "Zealous Conscripts", Some(Target::Object(theirs)), 0);
    assert_eq!(state.objects.get(theirs).controller, P0, "{card}");
    theirs
}

#[test]
fn stolen_creatures_read_the_thief_as_you() {
    // 109.5, 613.1b: Zealous Conscripts gives what it takes haste.
    use mtg_kernel::card_def::Keywords;
    // Laelia exiles the thief's top card for the thief, and grows.
    let mut state = game();
    let laelia = steal(&mut state, "Laelia, the Blade Reforged");
    let libraries = (state.players[0].library.len(), state.players[1].library.len());
    attack_with(&mut state, vec![laelia]);
    assert_eq!((state.players[0].library.len(), state.players[1].library.len()), (libraries.0 - 1, libraries.1));
    let exiled = *state.exile.last().unwrap();
    assert_eq!(state.objects.get(exiled).owner, P0);
    assert!(state.engine.exile_play_permissions.iter().any(|p| p.object == exiled && p.holder == P0));
    assert_eq!(state.objects.get(laelia).counters.plus1_plus1, 1);

    // Hobgoblin grows when the thief discards.
    let mut state = game();
    let hobgoblin = steal(&mut state, "Hobgoblin, Mantled Marauder");
    let base = engine::effective_power(&state, hobgoblin);
    put_in(&mut state, P0, "Mountain", Zone::Hand);
    lands(&mut state, P0, 2);
    let answers = put_in(&mut state, P0, "Demand Answers", Zone::Hand);
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    engine::step(&mut state, Action::CastSpell(answers)).unwrap();
    drive(&mut state, |s, d| match d {
        Decision::CastSpellOrPass { .. } if !s.stack.is_empty() => Some(Action::Pass),
        _ => first_choice(s, d),
    });
    assert_eq!(engine::effective_power(&state, hobgoblin), base + 2);

    // Gogo's trigger is the thief's, aimed at the thief's creatures.
    let mut state = game();
    let gogo = steal(&mut state, "Gogo, Mysterious Mime");
    let conscripts = named(&state, P0, "Zealous Conscripts")[0];
    let theirs = put(&mut state, P1, "Guttersnipe");
    let d = advance(&mut state, |s, d| s.step == Step::BeginCombat && matches!(d, Decision::ChooseTargets { .. }));
    let Decision::ChooseTargets { player, legal_targets, .. } = d else { unreachable!() };
    assert_eq!(player, P0);
    assert_eq!(legal_targets, vec![Target::Object(conscripts)], "not {gogo:?} or {theirs:?}");

    // Depthshaker Titan gives the thief's artifact creatures trample and haste.
    let mut state = game();
    steal(&mut state, "Depthshaker Titan");
    let mine = put(&mut state, P0, "Sol Ring");
    let theirs = put(&mut state, P1, "Sol Ring");
    animate(&mut state, mine);
    animate(&mut state, theirs);
    for k in [Keywords::TRAMPLE, Keywords::HASTE] {
        assert!(engine::has_effective_keyword(&state, mine, k));
        assert!(!engine::has_effective_keyword(&state, theirs, k));
    }
}

#[test]
fn stolen_noncreature_permanents_read_the_thief_as_you() {
    // 109.5, 613.1b: each is taken from player 1 with Zealous Conscripts,
    // which then attacks or pays where needed.
    use mtg_kernel::event::{propose_and_commit, ProposedEvent};
    let conscripts = |state: &GameState| named(state, P0, "Zealous Conscripts")[0];

    // Dolmen Gate protects the thief's attacker.
    let mut state = game();
    steal(&mut state, "Dolmen Gate");
    let blocker = put(&mut state, P1, "Zealous Conscripts");
    let attacker = conscripts(&state);
    assert_eq!(attack_into_block(&mut state, attacker, blocker), (0, true));

    // Raging River triggers on the thief's attack; player 1 divides.
    let mut state = game();
    steal(&mut state, "Raging River");
    put(&mut state, P1, "Guttersnipe");
    let attacker = conscripts(&state);
    advance(&mut state, |_, d| matches!(d, Decision::DeclareAttackers { .. }));
    engine::step(&mut state, Action::DeclareAttackers(vec![attacker])).unwrap();
    let d = advance(&mut state, |_, d| matches!(d, Decision::ChooseEffectTargets { .. } | Decision::DeclareBlockers { .. }));
    assert!(matches!(d, Decision::ChooseEffectTargets { player: P1, .. }), "{d:?}");

    // Connecting the Dots exiles from the thief's library.
    let mut state = game();
    steal(&mut state, "Connecting the Dots");
    let libraries = (state.players[0].library.len(), state.players[1].library.len());
    let attacker = conscripts(&state);
    attack_with(&mut state, vec![attacker]);
    assert_eq!((state.players[0].library.len(), state.players[1].library.len()), (libraries.0 - 1, libraries.1));

    // Arcane Signet makes the thief's commander's colors.
    let mut state = game();
    let signet = steal(&mut state, "Arcane Signet");
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    assert_eq!(engine::available_mana_ability_choices(P0, signet, &state), vec![ManaColor::B, ManaColor::R, ManaColor::G]);
    engine::step(&mut state, Action::ActivateManaAbilityChoice(signet, ManaColor::G)).unwrap();
    assert_eq!(state.players[0].mana_pool[ManaColor::G.pool_index()], 1);

    // Throne of Eldraine keeps its owner's color and draws for the thief.
    let mut state = game();
    let throne = put_in(&mut state, P1, "Throne of Eldraine", Zone::Hand);
    propose_and_commit(&mut state, ProposedEvent::zone_change(throne, Zone::Battlefield));
    assert!(matches!(engine::advance_until_decision(&mut state), Decision::ChooseColor { player: P1, .. }));
    engine::step(&mut state, Action::ChooseColor(ManaColor::R)).unwrap();
    lands(&mut state, P0, 5);
    cast(&mut state, "Zealous Conscripts", Some(Target::Object(throne)), 0);
    lands(&mut state, P0, 3);
    let hands = (state.players[0].hand.len(), state.players[1].hand.len());
    activate_with(&mut state, throne, 0, &[], &[], &[]);
    assert_eq!((state.players[0].hand.len(), state.players[1].hand.len()), (hands.0 + 2, hands.1));

    // Coin of Mastery makes the thief's Treasure and counts its creatures.
    let mut state = game();
    let coin = steal(&mut state, "Coin of Mastery");
    activate(&mut state, coin, 0, None);
    let treasure = named(&state, P0, "Treasure Token")[0];
    engine::step(&mut state, Action::ActivateManaAbilityChoice(treasure, ManaColor::R)).unwrap();
    lands(&mut state, P0, 2);
    let snipe = cast(&mut state, "Guttersnipe", None, 0);
    assert_eq!(state.objects.get(snipe).counters.plus1_plus1, 1);

    // Amulet of Vigor untaps the thief's Ognis Treasure.
    let mut state = game();
    steal(&mut state, "Amulet of Vigor");
    put(&mut state, P0, "Ognis, the Dragon's Lash");
    let attacker = conscripts(&state);
    attack_with(&mut state, vec![attacker]);
    let treasures = named(&state, P0, "Treasure Token");
    assert_eq!(treasures.len(), 1);
    assert!(!state.objects.get(treasures[0]).tapped);

    // Hylda's Crown taps for free on the thief's turn and counts the owner's
    // tapped creatures.
    let mut state = game();
    let crown = steal(&mut state, "Hylda's Crown of Winter");
    let theirs = put(&mut state, P1, "Guttersnipe");
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    assert!(offered_ability(&mut state, crown, 0), "no mana on the thief's turn");
    activate(&mut state, crown, 0, Some(Target::Object(theirs)));
    lands(&mut state, P0, 3);
    let hand = state.players[0].hand.len();
    activate(&mut state, crown, 1, None);
    assert_eq!(state.players[0].hand.len(), hand + 1);

    // Fomori Vault counts the thief's artifacts and looks at the thief's library.
    let mut state = game();
    let vault = steal(&mut state, "Fomori Vault");
    for p in [P0, P0, P1, P1, P1] {
        put(&mut state, p, "Sol Ring");
    }
    lands(&mut state, P0, 3);
    put_in(&mut state, P0, "Mountain", Zone::Hand);
    let top: Vec<Target> = state.players[0].library[..2].iter().map(|&o| Target::Object(o)).collect();
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    engine::step(&mut state, Action::ActivateAbility(vault, 0)).unwrap();
    let mut offered = None;
    drive(&mut state, |s, d| match d {
        Decision::ChooseEffectTargets { legal_targets, .. } => {
            offered = Some(legal_targets.clone());
            Some(Action::ChooseEffectTarget(legal_targets[0]))
        }
        Decision::CastSpellOrPass { .. } if !s.stack.is_empty() => Some(Action::Pass),
        _ => first_choice(s, d),
    });
    assert_eq!(offered, Some(top), "two artifacts: the thief's top two");

    // Demolition Field aims at the owner's lands and searches the thief's library.
    let mut state = game();
    let field = steal(&mut state, "Demolition Field");
    let mine = put(&mut state, P0, "Strip Mine");
    let theirs = put(&mut state, P1, "Strip Mine");
    lands(&mut state, P0, 2);
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    engine::step(&mut state, Action::ActivateAbility(field, 0)).unwrap();
    let libraries = (state.players[0].library.len(), state.players[1].library.len());
    let mut asked = Vec::new();
    drive(&mut state, |s, d| match d {
        Decision::ChooseTargets { legal_targets, .. } => {
            assert_eq!(legal_targets, &vec![Target::Object(theirs)], "not {mine:?}");
            Some(Action::ChooseTarget(Target::Object(theirs)))
        }
        Decision::ChooseEffectOption { player, .. } => {
            asked.push(*player);
            Some(Action::ChooseEffectOption(1))
        }
        Decision::CastSpellOrPass { .. } if !s.stack.is_empty() => Some(Action::Pass),
        _ => first_choice(s, d),
    });
    assert_eq!(asked, vec![P1, P0], "the land's controller, then the thief");
    assert_eq!((state.players[0].library.len(), state.players[1].library.len()), (libraries.0 - 1, libraries.1 - 1));
}

#[test]
fn spells_read_a_stolen_permanent_as_its_controllers() {
    // 109.5: "you control", "an opponent controls" and "your opponents
    // control" read the stolen permanent's controller, the thief.
    use mtg_kernel::event::{propose_and_commit, ProposedEvent};
    // Vandalblast: the thief's own Vandalblast spares it, aimed or overloaded.
    for overload in [false, true] {
        let mut state = game();
        let stolen = steal(&mut state, "Sol Ring");
        let other = put(&mut state, P1, "Sol Ring");
        if overload {
            lands(&mut state, P0, 5);
            cast(&mut state, "Vandalblast", None, 1);
            assert_eq!((zone(&state, stolen), zone(&state, other)), (Zone::Battlefield, Zone::Graveyard));
        } else {
            lands(&mut state, P0, 1);
            let offered = targets_offered(&mut state, "Vandalblast", 0);
            assert_eq!(offered, vec![Target::Object(other)]);
        }
    }

    // Wild Magic Surge: not the thief's target; its owner may aim at it, and
    // the thief reveals from the thief's library.
    let mut state = game();
    let ring = steal(&mut state, "Sol Ring");
    let snipe = put(&mut state, P1, "Guttersnipe");
    lands(&mut state, P0, 2);
    assert_eq!(targets_offered(&mut state, "Wild Magic Surge", 0), vec![Target::Object(snipe)], "not {ring:?}");
    let mut state = game();
    let ring = steal(&mut state, "Sol Ring");
    let found = library_card(&mut state, P0, 2, "Sol Ring");
    lands(&mut state, P1, 2);
    let surge = put_in(&mut state, P1, "Wild Magic Surge", Zone::Hand);
    let libraries = (state.players[0].library.len(), state.players[1].library.len());
    drive(&mut state, |s, d| match d {
        Decision::CastSpellOrPass { player: P1, castable_spells, .. } if castable_spells.contains(&surge) => Some(Action::CastSpell(surge)),
        Decision::ChooseTargets { player: P1, .. } => Some(Action::ChooseTarget(Target::Object(ring))),
        Decision::CastSpellOrPass { .. } if zone(s, surge) == Zone::Hand || !s.stack.is_empty() => Some(Action::Pass),
        Decision::CastSpellOrPass { .. } => None,
        _ => first_choice(s, d),
    });
    assert_eq!(zone(&state, ring), Zone::Graveyard);
    assert_eq!(named(&state, P0, "Sol Ring"), vec![found]);
    assert_eq!((state.players[0].library.len(), state.players[1].library.len()), (libraries.0 - 1, libraries.1));

    // Call Forth the Tempest: 5 (the Conscripts) to the owner's other
    // creature, none to the stolen one.
    let mut state = game();
    let stolen = steal(&mut state, "Spider-Punk");
    let other = put(&mut state, P1, "Guttersnipe");
    lands(&mut state, P0, 8);
    let tempest = put_in(&mut state, P0, "Call Forth the Tempest", Zone::Hand);
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    engine::step(&mut state, Action::CastSpell(tempest)).unwrap();
    drive(&mut state, |s, d| match d {
        Decision::CastSpellOrPass { .. } if !s.stack.is_empty() => Some(Action::Pass),
        _ => first_choice(s, d),
    });
    assert_eq!(zone(&state, other), Zone::Graveyard);
    assert_eq!((zone(&state, stolen), state.objects.get(stolen).damage), (Zone::Battlefield, 0));

    // Deflecting Swat: a stolen commander is a commander the thief controls.
    for take_ognis in [true, false] {
        let mut state = game();
        let ognis = state.commander.as_ref().unwrap().commanders[1];
        propose_and_commit(&mut state, ProposedEvent::zone_change(ognis, Zone::Battlefield));
        let snipe = put(&mut state, P1, "Guttersnipe");
        lands(&mut state, P0, 5);
        cast(&mut state, "Zealous Conscripts", Some(Target::Object(if take_ognis { ognis } else { snipe })), 0);
        lands(&mut state, P1, 1);
        let bolt = put_in(&mut state, P1, "Lightning Bolt", Zone::Hand);
        let swat = put_in(&mut state, P0, "Deflecting Swat", Zone::Hand);
        let mut offered = None;
        drive(&mut state, |s, d| match d {
            Decision::CastSpellOrPass { player: P1, castable_spells, .. } if castable_spells.contains(&bolt) => Some(Action::CastSpell(bolt)),
            Decision::ChooseTargets { player: P1, .. } => Some(Action::ChooseTarget(Target::Player(P0))),
            Decision::CastSpellOrPass { player: P0, castable_spells, .. } if zone(s, bolt) == Zone::Stack => {
                offered = Some(castable_spells.contains(&swat));
                None
            }
            Decision::CastSpellOrPass { .. } => Some(Action::Pass),
            other => panic!("unhandled decision {other:?}"),
        });
        assert_eq!(offered, Some(take_ognis), "with no mana left");
    }
}

#[test]
fn pyreswipe_hawks_steal_lasts_past_the_turn_and_serves_the_thief() {
    // 611.2b: "for as long as you control this creature". Liquimetal Torque
    // makes a creature an artifact long enough to be taken; control stays
    // after the Torque's effect ends.
    let take = |state: &mut GameState, card: &str| {
        put(state, P0, "Pyreswipe Hawk");
        let theirs = put(state, P1, card);
        if card != "Ruby Medallion" {
            let torque = put(state, P0, "Liquimetal Torque");
            activate(state, torque, 0, Some(Target::Object(theirs)));
        }
        lands(state, P0, 6);
        cast(state, "Guttersnipe", None, 0);
        cast(state, "Guttersnipe", Some(Target::Object(theirs)), 0);
        assert_eq!(state.objects.get(theirs).controller, P0, "{card}");
        theirs
    };

    // Ruby Medallion reduces the thief's spells, still the thief's next turn.
    let mut state = game();
    let ruby = take(&mut state, "Ruby Medallion");
    lands(&mut state, P0, 1);
    let impulse = put_in(&mut state, P0, "Reckless Impulse", Zone::Hand);
    assert!(castable(&mut state, impulse), "{{1}}{{R}} for one mana");
    to_my_next_upkeep(&mut state);
    assert_eq!(state.objects.get(ruby).controller, P0);

    // Urabrask: the owner's draw is exiled in the owner's upkeep, and the
    // thief impulses in the thief's.
    let mut state = game();
    let urabrask = take(&mut state, "Urabrask, Heretic Praetor");
    let turn = state.turn;
    advance(&mut state, |s, _| s.active_player == P1 && s.step == Step::Main1);
    assert_eq!(state.objects.get(urabrask).controller, P0);
    assert_eq!((state.players[1].hand.len(), state.players[1].library.len()), (0, 9), "no draw, only one card gone");
    let theirs: Vec<_> = state.exile.iter().copied().filter(|&o| state.objects.get(o).owner == P1).collect();
    assert_eq!(theirs.len(), 1);
    assert!(state.engine.exile_play_permissions.iter().any(|p| p.object == theirs[0] && p.holder == P1));
    advance(&mut state, |s, _| s.active_player == P0 && s.turn != turn && s.step == Step::Main1);
    let mine: Vec<_> = state.exile.iter().copied().filter(|&o| state.objects.get(o).owner == P0).collect();
    assert_eq!(mine.len(), 1);
    assert!(state.engine.exile_play_permissions.iter().any(|p| p.object == mine[0] && p.holder == P0));

    // Esper Valigarmanda: lore and chapter II come after the thief's draw
    // step, and the thief casts the card its owner exiled with it.
    let mut state = game();
    let saga = put(&mut state, P1, "Summon: Esper Valigarmanda");
    state.objects.get_mut(saga).counters.lore = 1;
    let bolt = put_in(&mut state, P1, "Lightning Bolt", Zone::Graveyard);
    {
        use mtg_kernel::event::{propose_and_commit, ProposedEvent};
        use mtg_kernel::state::{ExiledWithV1, ObjectLinkV4};
        propose_and_commit(&mut state, ProposedEvent::zone_change(bolt, Zone::Exile));
        let link = |s: &GameState, o: ObjectId| ObjectLinkV4 { object: o, zone_change_count: s.objects.get(o).zone_change_count };
        let entry = ExiledWithV1 { source: link(&state, saga), card: link(&state, bolt) };
        state.exiled_with.push(entry);
    }
    put(&mut state, P0, "Pyreswipe Hawk");
    let torque = put(&mut state, P0, "Liquimetal Torque");
    activate(&mut state, torque, 0, Some(Target::Object(saga)));
    lands(&mut state, P0, 6);
    cast(&mut state, "Guttersnipe", None, 0);
    cast(&mut state, "Guttersnipe", Some(Target::Object(saga)), 0);
    assert_eq!(state.objects.get(saga).controller, P0);
    end_turn(&mut state);
    let mut caster = None;
    pass_until(&mut state, |s, d| match d {
        _ if zone(s, bolt) == Zone::Graveyard => Some(None),
        Decision::ChooseCastFromExile { player, .. } => {
            caster = Some(*player);
            Some(Some(Action::ChooseCastFromExile(Some(bolt))))
        }
        Decision::ChooseTargets { .. } => Some(Some(Action::ChooseTarget(Target::Player(P1)))),
        _ => None,
    });
    assert_eq!(caster, Some(P0));
    assert_eq!(state.objects.get(saga).counters.lore, 2, "none after the owner's draw step");
    assert_eq!(state.players[1].life, 33, "the Bolt, and 2 from each Guttersnipe");
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ArtifactKind {
    Card,
    Token,
    TappedToken,
    TorquedCreature,
    TorquedEnchantment,
    Animated,
    Theirs,
    Plain,
}

impl ArtifactKind {
    const ALL: [ArtifactKind; 8] = [
        ArtifactKind::Card,
        ArtifactKind::Token,
        ArtifactKind::TappedToken,
        ArtifactKind::TorquedCreature,
        ArtifactKind::TorquedEnchantment,
        ArtifactKind::Animated,
        ArtifactKind::Theirs,
        ArtifactKind::Plain,
    ];

    /// An artifact player 0 controls.
    fn mine(self) -> bool {
        !matches!(self, ArtifactKind::Theirs | ArtifactKind::Plain)
    }

    fn creature(self) -> bool {
        matches!(self, ArtifactKind::TorquedCreature | ArtifactKind::Animated | ArtifactKind::Plain)
    }
}

/// One permanent of `kind`. The Torque is put into the graveyard after its
/// ability resolves (the effect stays), so it isn't counted itself.
fn artifact_of_kind(state: &mut GameState, kind: ArtifactKind) -> ObjectId {
    use mtg_kernel::event::{propose_and_commit, ProposedEvent};
    match kind {
        ArtifactKind::Card => put(state, P0, "Sol Ring"),
        ArtifactKind::Token => put(state, P0, "Treasure Token"),
        // As Ognis makes them.
        ArtifactKind::TappedToken => {
            let t = put(state, P0, "Treasure Token");
            state.objects.get_mut(t).tapped = true;
            t
        }
        ArtifactKind::Theirs => put(state, P1, "Sol Ring"),
        ArtifactKind::Plain => put(state, P0, "Guttersnipe"),
        ArtifactKind::Animated => {
            let ring = put(state, P0, "Sol Ring");
            animate(state, ring);
            ring
        }
        ArtifactKind::TorquedCreature | ArtifactKind::TorquedEnchantment => {
            let o = put(state, P0, if kind == ArtifactKind::TorquedCreature { "Guttersnipe" } else { "Connecting the Dots" });
            let torque = put(state, P0, "Liquimetal Torque");
            activate(state, torque, 0, Some(Target::Object(o)));
            propose_and_commit(state, ProposedEvent::zone_change(torque, Zone::Graveyard));
            assert!(engine::object_has_type(state, o, mtg_kernel::card_def::CardType::Artifact), "{kind:?}");
            o
        }
    }
}

#[test]
fn each_kind_of_artifact_counts_for_the_vault_starforge_titan_and_sacrifices_and_only_treasures_for_smaug() {
    // Fomori Vault's X and Secluded Starforge's taps count artifacts you
    // control; Depthshaker Titan targets noncreature ones; Demand Answers and
    // Vision of Love may sacrifice any; Smaug counts only Treasures you control.
    use mtg_kernel::engine::OptionalCostChoice;
    let sorted = |mut v: Vec<ObjectId>| {
        v.sort();
        v
    };
    for kind in ArtifactKind::ALL {
        let mine = kind.mine();

        // Fomori Vault: two Sol Rings, plus this one if it counts.
        let mut state = game();
        artifact_of_kind(&mut state, kind);
        let vault = put(&mut state, P0, "Fomori Vault");
        put(&mut state, P0, "Sol Ring");
        put(&mut state, P0, "Sol Ring");
        lands(&mut state, P0, 3);
        put_in(&mut state, P0, "Mountain", Zone::Hand);
        advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
        engine::step(&mut state, Action::ActivateAbility(vault, 0)).unwrap();
        let mut x = 0;
        drive(&mut state, |s, d| match d {
            Decision::ChooseEffectTargets { legal_targets, .. } => {
                x = legal_targets.len();
                Some(Action::ChooseEffectTarget(legal_targets[0]))
            }
            Decision::CastSpellOrPass { .. } if !s.stack.is_empty() => Some(Action::Pass),
            _ => first_choice(s, d),
        });
        assert_eq!(x, 2 + usize::from(mine), "{kind:?}: Fomori Vault");

        // Secluded Starforge: tappable for X if untapped and yours.
        let mut state = game();
        let o = artifact_of_kind(&mut state, kind);
        let forge = put(&mut state, P0, "Secluded Starforge");
        let creature = put(&mut state, P0, "Zealous Conscripts");
        lands(&mut state, P0, 2);
        advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
        engine::step(&mut state, Action::ActivateAbility(forge, 0)).unwrap();
        let mut candidates = Vec::new();
        drive(&mut state, |s, d| match d {
            Decision::ChooseTargets { .. } => Some(Action::ChooseTarget(Target::Object(creature))),
            Decision::ChooseCostTargets { candidates: c, .. } => {
                candidates = c.clone();
                Some(Action::FinishEffectSelection)
            }
            Decision::CastSpellOrPass { .. } if !s.stack.is_empty() => Some(Action::Pass),
            _ => first_choice(s, d),
        });
        assert_eq!(candidates.contains(&o), mine && kind != ArtifactKind::TappedToken, "{kind:?}: Starforge");

        // Depthshaker Titan: noncreature artifacts you control.
        let mut state = game();
        let o = artifact_of_kind(&mut state, kind);
        lands(&mut state, P0, 7);
        let titan = put_in(&mut state, P0, "Depthshaker Titan", Zone::Hand);
        advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
        engine::step(&mut state, Action::CastSpell(titan)).unwrap();
        let mut offered = Vec::new();
        drive(&mut state, |s, d| match d {
            Decision::ChooseTargets { legal_targets, .. } => {
                offered = legal_targets.clone();
                Some(Action::FinishEffectSelection)
            }
            Decision::CastSpellOrPass { .. } if !s.stack.is_empty() => Some(Action::Pass),
            _ => first_choice(s, d),
        });
        assert_eq!(offered.contains(&Target::Object(o)), mine && !kind.creature(), "{kind:?}: Depthshaker Titan");

        // Demand Answers and Vision of Love, next to a Sol Ring: a choice of
        // two only when this one may be sacrificed.
        for spell in ["Demand Answers", "Vision of Love"] {
            let mut state = game();
            let o = artifact_of_kind(&mut state, kind);
            let ring = put(&mut state, P0, "Sol Ring");
            lands(&mut state, P0, 2);
            put_in(&mut state, P0, "Mountain", Zone::Hand);
            let card = put_in(&mut state, P0, spell, Zone::Hand);
            advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
            engine::step(&mut state, Action::CastSpell(card)).unwrap();
            let mut offered = None;
            drive(&mut state, |s, d| match d {
                Decision::ChooseEffectOption { .. } => Some(Action::ChooseEffectOption(0)),
                Decision::ChooseOptionalCost { .. } => Some(Action::ChooseOptionalCost(OptionalCostChoice::SacrificeLand)),
                Decision::ChooseCostTargets { candidates, .. } => {
                    offered = Some(sorted(candidates.clone()));
                    Some(Action::ChooseCostTarget(ring))
                }
                Decision::CastSpellOrPass { .. } if !s.stack.is_empty() => Some(Action::Pass),
                _ => first_choice(s, d),
            });
            assert_eq!(offered, mine.then(|| sorted(vec![ring, o])), "{kind:?}: {spell}");
            assert_eq!(zone(&state, ring), Zone::Graveyard, "{kind:?}: {spell}");
        }

        // Smaug: only Treasures you control; player 1's Treasure doesn't count.
        let mut state = game();
        artifact_of_kind(&mut state, kind);
        put(&mut state, P1, "Treasure Token");
        let smaug = put(&mut state, P0, "Smaug the Magnificent");
        attack_with(&mut state, vec![smaug]);
        let treasure = matches!(kind, ArtifactKind::Token | ArtifactKind::TappedToken);
        assert_eq!(40 - state.players[1].life, 4 + i32::from(treasure), "{kind:?}: Smaug");

        // Player 0's own Shattering Spree and Untimely Malfunction may aim at
        // any artifact, Vandalblast only at one player 0 doesn't control.
        for spell in ["Shattering Spree", "Untimely Malfunction", "Vandalblast"] {
            let mut state = game();
            let o = artifact_of_kind(&mut state, kind);
            put(&mut state, P1, "Arcane Signet");
            lands(&mut state, P0, 2);
            let offered = targets_offered(&mut state, spell, 0);
            let expected = if spell == "Vandalblast" { kind == ArtifactKind::Theirs } else { kind != ArtifactKind::Plain };
            assert_eq!(offered.contains(&Target::Object(o)), expected, "{kind:?}: {spell}");
        }
    }
}

#[test]
fn knuckles_counts_thirty_artifacts_you_control_at_your_upkeep() {
    // 29 Treasures and one more: an artifact card or a Treasure makes thirty;
    // Liquimetal Torque's effect has ended by the next upkeep, and player 1's
    // artifact isn't yours.
    for kind in [ArtifactKind::Card, ArtifactKind::Token, ArtifactKind::TorquedCreature, ArtifactKind::Theirs] {
        let mut state = game();
        artifact_of_kind(&mut state, kind);
        put(&mut state, P0, "Knuckles the Echidna");
        for _ in 0..29 {
            put(&mut state, P0, "Treasure Token");
        }
        let d = advance(&mut state, |s, d| {
            matches!(d, Decision::GameOver { .. }) || (s.active_player == P0 && s.turn > 3 && s.step == Step::Main1)
        });
        let won = matches!(d, Decision::GameOver { winner: Some(P0) });
        assert_eq!(won, matches!(kind, ArtifactKind::Card | ArtifactKind::Token), "{kind:?}: {d:?}");
    }
}

/// Ruby Medallion, The Fire Crystal, either, both.
const REDUCER_SETS: [&[&str]; 4] = [&[], &["Ruby Medallion"], &["The Fire Crystal"], &["Ruby Medallion", "The Fire Crystal"]];

#[test]
fn red_spell_reducers_take_generic_off_flashback_retrace_foretold_and_spectacle_casts() {
    // 601.2f, 118.7d: each reducer takes {1} off the total cost however the
    // spell is cast, never off the colored part. Each card here has only one
    // way to be cast from where it is, so it is castable with exactly that
    // cost and not one mana less.
    use mtg_kernel::event::{propose_and_commit, ProposedEvent};
    // (card, generic, colored) of the cost paid.
    let rows = [
        ("Reckless Impulse", 1u8, 1u8),     // flashback {1}{R}, from Flashback
        ("Embrace the Unknown", 2, 1),      // retrace {2}{R} and a land card
        ("Delayed Blast Fireball", 4, 2),   // foretold {4}{R}{R}
        ("Light Up the Stage", 0, 1),       // spectacle {R}
    ];
    for reducers in REDUCER_SETS {
        let r = reducers.len() as u8;
        for (card, generic, colored) in rows {
            let need = usize::from(generic - generic.min(r) + colored);
            for lands_count in [need, need - 1] {
                let mut state = game();
                for x in reducers {
                    put(&mut state, P0, x);
                }
                let o = match card {
                    "Reckless Impulse" => {
                        let o = put_in(&mut state, P0, card, Zone::Graveyard);
                        lands(&mut state, P0, 1);
                        cast(&mut state, "Flashback", Some(Target::Object(o)), 0);
                        o
                    }
                    "Embrace the Unknown" => {
                        put_in(&mut state, P0, "Mountain", Zone::Hand);
                        put_in(&mut state, P0, card, Zone::Graveyard)
                    }
                    "Delayed Blast Fireball" => {
                        // Foretold on turn 2; cast in player 1's turn, when
                        // the two lands that paid are still tapped.
                        let o = put_in(&mut state, P0, card, Zone::Hand);
                        lands(&mut state, P0, 2);
                        advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
                        engine::step(&mut state, Action::Foretell(o)).unwrap();
                        end_turn(&mut state);
                        o
                    }
                    _ => {
                        advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
                        propose_and_commit(&mut state, ProposedEvent::life_loss(P1, 1));
                        put_in(&mut state, P0, card, Zone::Hand)
                    }
                };
                lands(&mut state, P0, lands_count);
                let ok = matches!(
                    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { player: P0, .. })),
                    Decision::CastSpellOrPass { castable_spells, .. } if castable_spells.contains(&o)
                );
                assert_eq!(ok, lands_count == need, "{card} with {reducers:?} and {lands_count} lands");
            }
        }
    }
}

#[test]
fn red_spell_reducers_lower_overload_x_and_redirect_lightnings_extra_cost_but_not_replicate_or_special_actions() {
    // 601.2f: overload {4}{R}, X and Redirect Lightning's additional {2} are
    // part of the total cost; replicate's {R} is colored. Foretelling ({2})
    // and plotting ({1}{R}) are special actions, not spells (116.2h, 116.2k).
    for reducers in REDUCER_SETS {
        let r = reducers.len();
        let setup = |lands_count: usize| {
            let mut state = game();
            for x in reducers {
                put(&mut state, P0, x);
            }
            lands(&mut state, P0, lands_count);
            state
        };
        // Starts casting `card` and returns the first spell-mode choice's
        // legal modes, or each effect option count asked, until a target.
        let start = |state: &mut GameState, card: &str| {
            let spell = put_in(state, P0, card, Zone::Hand);
            advance(state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
            engine::step(state, Action::CastSpell(spell)).unwrap();
            engine::advance_until_decision(state)
        };

        // Overload: offered with 5 - r lands, not one fewer.
        let need = 5 - r;
        for lands_count in [need, need - 1] {
            let mut state = setup(lands_count);
            put(&mut state, P1, "Sol Ring");
            // With overload unaffordable there is no mode choice.
            let overload = matches!(start(&mut state, "Vandalblast"), Decision::ChooseSpellMode { legal_modes, .. } if legal_modes.contains(&1));
            assert_eq!(overload, lands_count == need, "overload with {reducers:?} and {lands_count} lands");
        }

        // Shivan Devastator with three lands: X up to 2 + r.
        let mut state = setup(3);
        match start(&mut state, "Shivan Devastator") {
            Decision::ChooseEffectOption { option_count, .. } => {
                assert_eq!(usize::from(option_count) - 1, 2 + r, "largest X with {reducers:?}")
            }
            other => panic!("{other:?}"),
        }

        // Shattering Spree with three lands: up to two replicate payments.
        let mut state = setup(3);
        put(&mut state, P1, "Sol Ring");
        let spree = put_in(&mut state, P0, "Shattering Spree", Zone::Hand);
        advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
        engine::step(&mut state, Action::CastSpell(spree)).unwrap();
        let mut counts = Vec::new();
        drive(&mut state, |s, d| match d {
            Decision::ChooseEffectOption { option_count, .. } => {
                counts.push(*option_count);
                Some(Action::ChooseEffectOption(0))
            }
            Decision::CastSpellOrPass { .. } if !s.stack.is_empty() => Some(Action::Pass),
            _ => first_choice(s, d),
        });
        assert_eq!(counts, vec![3], "zero, one or two payments with {reducers:?}");

        // Redirect Lightning, player 1's: the {2} option alongside 5 life
        // needs 3 - r lands.
        for lands_count in 1..=3usize {
            let mut state = game();
            for x in reducers {
                put(&mut state, P1, x);
            }
            lands(&mut state, P1, lands_count);
            lands(&mut state, P0, 1);
            let bolt = put_in(&mut state, P0, "Lightning Bolt", Zone::Hand);
            let redirect = put_in(&mut state, P1, "Redirect Lightning", Zone::Hand);
            let mut choice = false;
            drive(&mut state, |s, d| match d {
                _ if zone(s, bolt) != Zone::Hand && zone(s, bolt) != Zone::Stack => None,
                Decision::CastSpellOrPass { player, castable_spells, .. } => Some(match *player {
                    P0 if castable_spells.contains(&bolt) => Action::CastSpell(bolt),
                    P1 if castable_spells.contains(&redirect) => Action::CastSpell(redirect),
                    _ => Action::Pass,
                }),
                Decision::ChooseTargets { player: P0, .. } => Some(Action::ChooseTarget(Target::Player(P1))),
                Decision::ChooseTargets { .. } => Some(Action::ChooseTarget(Target::Object(bolt))),
                Decision::ChooseEffectOption { .. } => {
                    choice = true;
                    Some(Action::ChooseEffectOption(1))
                }
                Decision::ChooseEffectTargets { .. } => Some(Action::ChooseEffectTarget(Target::Player(P0))),
                other => panic!("unhandled decision {other:?}"),
            });
            let affordable = lands_count + r >= 3;
            assert_eq!(choice, affordable, "{reducers:?} with {lands_count} lands");
            assert_eq!(state.players[1].life, if affordable { 40 } else { 35 }, "{reducers:?} with {lands_count} lands");
            assert_eq!(state.players[0].life, 37);
        }

        // Player 1's Deflecting Swat, with no commander out: {2}{R} less r.
        let need = 3 - r;
        for lands_count in [need, need - 1] {
            let mut state = game();
            for x in reducers {
                put(&mut state, P1, x);
            }
            lands(&mut state, P1, lands_count);
            let swat = put_in(&mut state, P1, "Deflecting Swat", Zone::Hand);
            assert_eq!(offered_against_bolt(&mut state, swat), lands_count == need, "Swat with {reducers:?} and {lands_count} lands");
        }

        // Player 0's reducers don't lower player 1's Redirect Lightning: with
        // two lands only the life payment is left, so there is no choice.
        let mut state = game();
        for x in reducers {
            put(&mut state, P0, x);
        }
        lands(&mut state, P0, 1);
        lands(&mut state, P1, 2);
        let redirect = bolt_and_answer(&mut state, Target::Player(P1), "Redirect Lightning", None, Target::Player(P0));
        assert_eq!((zone(&state, redirect), state.players[1].life), (Zone::Graveyard, 35), "{reducers:?}");

        // Secluded Starforge's ability isn't a spell: still {2}.
        for lands_count in [2, 1] {
            let mut state = setup(lands_count);
            let forge = put(&mut state, P0, "Secluded Starforge");
            put(&mut state, P0, "Guttersnipe");
            advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
            assert_eq!(offered_ability(&mut state, forge, 0), lands_count == 2, "Starforge with {reducers:?} and {lands_count} lands");
        }

        // Foretelling still takes {2}, plotting {1}{R}.
        for (card, cost) in [("Delayed Blast Fireball", 2usize), ("Slickshot Show-Off", 2)] {
            for lands_count in [cost, cost - 1] {
                let mut state = setup(lands_count);
                let o = put_in(&mut state, P0, card, Zone::Hand);
                let d = advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
                let Decision::CastSpellOrPass { foretell_actions, plot_actions, .. } = d else { unreachable!() };
                let offered = foretell_actions.contains(&o) || plot_actions.contains(&o);
                assert_eq!(offered, lands_count == cost, "{card} with {reducers:?} and {lands_count} lands");
            }
        }
    }
}

/// Answers what a copy asks as it enters and records it: riot (haste), the
/// legend rule (keeps the copy, not `original`), and its enters trigger's
/// targets (the first legal one). Targeting by `source`, the copier's own
/// ability, is left to the caller.
fn answer_copy_entering(s: &GameState, d: &Decision, original: ObjectId, source: ObjectId, asked: &mut Vec<&'static str>) -> Option<Action> {
    match d {
        Decision::ChooseRiot { .. } => {
            asked.push("riot");
            Some(Action::ChooseRiot(true))
        }
        Decision::ChooseLegendToKeep { legends, .. } => {
            asked.push("legend");
            Some(Action::ChooseLegendToKeep(*legends.iter().find(|&&l| l != original).unwrap()))
        }
        Decision::ChooseTargets { spell: by, legal_targets, .. } | Decision::ChooseEffectTargets { source: by, legal_targets, .. } if *by != source => {
            let first = !asked.contains(&"trigger");
            if first {
                asked.push("trigger");
            }
            Some(match legal_targets.first() {
                Some(&t) if first && matches!(d, Decision::ChooseTargets { .. }) => Action::ChooseTarget(t),
                Some(&t) if first => Action::ChooseEffectTarget(t),
                _ => Action::FinishEffectSelection,
            })
        }
        Decision::OrderTriggers { pending, .. } => Some(Action::OrderTriggers((0..pending.len()).collect())),
        Decision::CastSpellOrPass { .. } if !s.stack.is_empty() => Some(Action::Pass),
        _ => None,
    }
}

#[test]
fn a_permanent_copy_enters_with_the_copied_cards_enters_abilities_but_gogo_doesnt_enter() {
    // 707.2, 614.12, 603.6a; Cursed Mirror and The Fire Crystal rulings: the
    // copy has the copied card's enters triggers and "enters with" abilities,
    // with X 0, and a Saga's first lore counter. Gogo becomes a copy on the
    // battlefield, so none of these apply. Each original is player 0's.
    let originals = [
        "Zealous Conscripts",
        "Molten Primordial",
        "Headliner Scarlett",
        "Depthshaker Titan",
        "Spider-Punk",
        "Shivan Devastator",
        "Summon: Esper Valigarmanda",
    ];
    let crystal_of = |s: &GameState| named(s, P0, "The Fire Crystal")[0];
    for copier in ["Cursed Mirror", "The Fire Crystal", "Gogo, Mysterious Mime"] {
        for original in originals {
            let mut state = game();
            let o = put(&mut state, P0, original);
            if original == "Shivan Devastator" {
                state.objects.get_mut(o).counters.plus1_plus1 = 2;
            }
            put(&mut state, P1, "Guttersnipe");
            put(&mut state, P0, "Sol Ring");
            let mut asked = Vec::new();
            let copy = match copier {
                "Cursed Mirror" => {
                    lands(&mut state, P0, 3);
                    let mirror = put_in(&mut state, P0, copier, Zone::Hand);
                    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
                    engine::step(&mut state, Action::CastSpell(mirror)).unwrap();
                    drive(&mut state, |s, d| match d {
                        Decision::ChooseCopy { .. } => Some(Action::ChooseCopy(Some(o))),
                        _ => answer_copy_entering(s, d, o, o, &mut asked),
                    });
                    mirror
                }
                "The Fire Crystal" => {
                    let crystal = put(&mut state, P0, copier);
                    lands(&mut state, P0, 6);
                    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
                    engine::step(&mut state, Action::ActivateAbility(crystal, 0)).unwrap();
                    drive(&mut state, |s, d| match d {
                        Decision::ChooseTargets { spell, .. } if *spell == crystal => Some(Action::ChooseTarget(Target::Object(o))),
                        _ => answer_copy_entering(s, d, o, crystal, &mut asked),
                    });
                    // The token, unless it died.
                    match state.players[0].battlefield.iter().copied().find(|&t| t != o && state.objects.get(t).name == original) {
                        Some(token) => token,
                        None => crystal,
                    }
                }
                _ => {
                    let gogo = put(&mut state, P0, copier);
                    advance(&mut state, |s, d| s.step == Step::BeginCombat && matches!(d, Decision::ChooseTargets { .. }));
                    engine::step(&mut state, Action::ChooseTarget(Target::Object(o))).unwrap();
                    drive(&mut state, |s, d| match d {
                        Decision::ChooseEffectOption { option_count: 2, .. } => Some(Action::ChooseEffectOption(1)),
                        _ => answer_copy_entering(s, d, o, gogo, &mut asked),
                    });
                    gogo
                }
            };
            let gogo = copier == "Gogo, Mysterious Mime";
            let expected: &[&str] = match original {
                _ if gogo => &[],
                "Spider-Punk" => &["riot", "riot", "legend"],
                "Headliner Scarlett" => &["legend", "trigger"],
                "Shivan Devastator" | "Summon: Esper Valigarmanda" => &[],
                _ => &["trigger"],
            };
            // The engine asks riot after the legend rule, not as the copy
            // enters (614.12); one player makes both choices with nothing
            // revealed between, so the order can't change the outcome.
            asked.sort();
            let mut expected = expected.to_vec();
            expected.sort();
            assert_eq!(asked, expected, "{copier} copying {original}");
            match original {
                // X is 0: a 0/0 (the token is gone), or Gogo's 2/0.
                "Shivan Devastator" if copier == "The Fire Crystal" => assert_eq!(copy, crystal_of(&state), "the token died"),
                "Shivan Devastator" => assert_ne!(zone(&state, copy), Zone::Battlefield, "{copier}"),
                "Summon: Esper Valigarmanda" => assert_eq!(state.objects.get(copy).counters.lore, if gogo { 0 } else { 1 }, "{copier}"),
                _ => assert_eq!(state.objects.get(copy).name, if gogo { copier } else { original }, "{copier}"),
            }
        }
    }
}

#[test]
fn dragonspark_reactor_counts_a_copy_entering_only_as_an_artifact() {
    // Cursed Mirror entering as a copy of a nonartifact creature isn't an
    // artifact (707.2); as a copy of Depthshaker Titan, or as itself, it is.
    // The Fire Crystal's token likewise.
    for (copier, copied, counts) in [
        ("Cursed Mirror", Some("Guttersnipe"), false),
        ("Cursed Mirror", Some("Depthshaker Titan"), true),
        ("Cursed Mirror", None, true),
        ("The Fire Crystal", Some("Guttersnipe"), false),
        ("The Fire Crystal", Some("Depthshaker Titan"), true),
    ] {
        let mut state = game();
        let reactor = put(&mut state, P0, "Dragonspark Reactor");
        let o = copied.map(|c| put(&mut state, P0, c));
        let mut asked = Vec::new();
        if copier == "Cursed Mirror" {
            lands(&mut state, P0, 3);
            let mirror = put_in(&mut state, P0, copier, Zone::Hand);
            advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
            engine::step(&mut state, Action::CastSpell(mirror)).unwrap();
            drive(&mut state, |s, d| match d {
                Decision::ChooseCopy { .. } => Some(Action::ChooseCopy(o)),
                _ => answer_copy_entering(s, d, o.unwrap_or(reactor), reactor, &mut asked),
            });
        } else {
            let crystal = put(&mut state, P0, copier);
            lands(&mut state, P0, 6);
            advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
            engine::step(&mut state, Action::ActivateAbility(crystal, 0)).unwrap();
            let o = o.unwrap();
            drive(&mut state, |s, d| match d {
                Decision::ChooseTargets { spell, .. } if *spell == crystal => Some(Action::ChooseTarget(Target::Object(o))),
                _ => answer_copy_entering(s, d, o, crystal, &mut asked),
            });
        }
        assert_eq!(state.objects.get(reactor).counters.charge, if counts { 1 } else { 0 }, "{copier} copying {copied:?}");
    }
}

#[test]
fn a_land_exiled_to_be_played_is_playable_only_as_the_turns_land_drop() {
    // 305.2, 305.3; Laelia and Urabrask rulings: playing an exiled land uses
    // the turn's land play. Each source exiles an Abstergo Entertainment from
    // the top of player 0's library: offered while no land was played this
    // turn, not after a Mountain from hand.
    let sources = [
        "Reckless Impulse",
        "Wrenn's Resolve",
        "Light Up the Stage",
        "Embrace the Unknown",
        "Laelia, the Blade Reforged",
        "Urabrask, Heretic Praetor",
        "Headliner Scarlett",
    ];
    for source in sources {
        for land_played_first in [false, true] {
            let mut state = game();
            library_card(&mut state, P0, 0, "Abstergo Entertainment");
            match source {
                "Laelia, the Blade Reforged" => {
                    let laelia = put(&mut state, P0, source);
                    attack_with(&mut state, vec![laelia]);
                }
                "Urabrask, Heretic Praetor" | "Headliner Scarlett" => {
                    put(&mut state, P0, source);
                    to_my_next_upkeep(&mut state);
                }
                _ => {
                    lands(&mut state, P0, 3);
                    cast(&mut state, source, None, 0);
                }
            }
            advance(&mut state, |s, d| {
                matches!(s.step, Step::Main1 | Step::Main2) && s.stack.is_empty() && matches!(d, Decision::CastSpellOrPass { player: P0, .. })
            });
            let exiled = state.exile.iter().copied().find(|&o| state.objects.get(o).name == "Abstergo Entertainment");
            let exiled = exiled.unwrap_or_else(|| panic!("{source}: Abstergo exiled"));
            if land_played_first {
                let mountain = put_in(&mut state, P0, "Mountain", Zone::Hand);
                engine::step(&mut state, Action::PlayLand(mountain)).unwrap();
            }
            let Decision::CastSpellOrPass { land_drops, .. } = engine::advance_until_decision(&mut state) else { panic!("{source}: priority") };
            assert_eq!(land_drops.contains(&exiled), !land_played_first, "{source}, a land played first: {land_played_first}");
        }
    }
}

#[test]
fn raging_river_divides_only_creatures_without_flying_and_flyers_block_either_attacker() {
    // Raging River's piles hold the defending player's creatures without
    // flying; a flyer may block an attacker sent to either pile, and is the
    // only kind that may block player 0's flying Goldspan Dragon. Each
    // creature in the deck defends alone, put in the left pile if divided;
    // both attackers are sent right. A celebrating Goddric flies.
    use mtg_kernel::card_def::{CardType, Keywords, CARD_DEFS};
    use mtg_kernel::event::{propose_and_commit, ProposedEvent};
    let mut creatures: Vec<(&str, bool)> = ognis_deck_spells()
        .into_iter()
        .filter(|&n| CARD_DEFS[id(n) as usize].types.contains(&CardType::Creature))
        .map(|n| (n, false))
        .collect();
    creatures.push(("Goddric, Cloaked Reveler", true));
    let mut flyers = 0;
    for (name, celebrate) in creatures {
        let mut state = game();
        put(&mut state, P0, "Raging River");
        let attackers = vec![put(&mut state, P0, "Goldspan Dragon"), put(&mut state, P0, "Zealous Conscripts")];
        let theirs = put(&mut state, P1, name);
        if name == "Shivan Devastator" {
            state.objects.get_mut(theirs).counters.plus1_plus1 = 1;
        }
        if celebrate {
            for _ in 0..2 {
                propose_and_commit(&mut state, ProposedEvent::create_token(id("Treasure Token"), P1));
            }
        }
        let flying = engine::has_effective_keyword(&state, theirs, Keywords::FLYING);
        flyers += usize::from(flying);
        advance(&mut state, |_, d| matches!(d, Decision::DeclareAttackers { .. }));
        engine::step(&mut state, Action::DeclareAttackers(attackers.clone())).unwrap();
        let (mut piles, mut blocks) = (Vec::new(), None);
        drive(&mut state, |s, d| match d {
            Decision::DeclareBlockers { legal_blockers, .. } => {
                blocks = Some(legal_blockers.clone());
                None
            }
            Decision::ChooseEffectTargets { player: P1, selected_count: 0, legal_targets, .. } => {
                piles = legal_targets.clone();
                Some(Action::ChooseEffectTarget(Target::Object(theirs)))
            }
            Decision::ChooseEffectTargets { .. } => Some(Action::FinishEffectSelection),
            Decision::CastSpellOrPass { .. } => Some(Action::Pass),
            _ => first_choice(s, d),
        });
        assert_eq!(piles.contains(&Target::Object(theirs)), !flying, "{name}: divided");
        let blocks = blocks.unwrap_or_else(|| panic!("{name}: blockers"));
        for a in &attackers {
            let can_block = blocks.iter().any(|(o, v)| o == a && v.contains(&theirs));
            assert_eq!(can_block, flying, "{name} blocking {}", state.objects.get(*a).name);
        }
    }
    assert_eq!(flyers, 10, "nine printed flyers and a celebrating Goddric");
}

#[test]
fn each_players_commander_cards_read_their_own_commander() {
    // 903.4: player 1's Arcane Signet makes player 1's commander's colors and
    // its War Room costs player 1 three life; player 0's Chaos Warp on its
    // own Ognis lets player 0, the owner, choose the command zone.
    use mtg_kernel::event::{propose_and_commit, ProposedEvent};
    let mut state = game();
    let signet = put(&mut state, P1, "Arcane Signet");
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    assert_eq!(engine::available_mana_ability_choices(P1, signet, &state), vec![ManaColor::B, ManaColor::R, ManaColor::G]);

    let mut state = game();
    let room = put(&mut state, P1, "War Room");
    lands(&mut state, P1, 3);
    end_turn(&mut state);
    advance(&mut state, |s, d| s.step == Step::Main1 && matches!(d, Decision::CastSpellOrPass { player: P1, .. }));
    engine::step(&mut state, Action::ActivateAbility(room, 0)).unwrap();
    drive(&mut state, |s, d| match d {
        Decision::CastSpellOrPass { .. } if !s.stack.is_empty() => Some(Action::Pass),
        _ => first_choice(s, d),
    });
    assert_eq!((state.players[0].life, state.players[1].life), (40, 37));

    let mut state = game();
    let ognis = state.commander.as_ref().unwrap().commanders[0];
    propose_and_commit(&mut state, ProposedEvent::zone_change(ognis, Zone::Battlefield));
    lands(&mut state, P0, 3);
    cast(&mut state, "Chaos Warp", Some(Target::Object(ognis)), 1);
    assert_eq!(zone(&state, ognis), Zone::Command);
}

#[test]
fn light_up_the_stage_cast_with_flashback_cant_also_use_spectacle() {
    // 118.9a: one alternative cost per spell. From the graveyard with
    // flashback it costs {2}{R}, though an opponent lost life this turn.
    use mtg_kernel::event::{propose_and_commit, ProposedEvent};
    let mut state = game();
    let stage = put_in(&mut state, P0, "Light Up the Stage", Zone::Graveyard);
    lands(&mut state, P0, 1);
    cast(&mut state, "Flashback", Some(Target::Object(stage)), 0);
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    propose_and_commit(&mut state, ProposedEvent::life_loss(P1, 1));
    lands(&mut state, P0, 2);
    assert!(!castable(&mut state, stage), "not for {{R}}");
    lands(&mut state, P0, 1);
    assert!(castable(&mut state, stage));
}

#[test]
fn shivan_devastator_without_its_mana_cost_paid_has_x_0_and_dies() {
    // 107.3b: cast by cascade without paying its mana cost, X is 0; 107.3m:
    // put onto the battlefield (player 1's, by player 0's Chaos Warp or Wild
    // Magic Surge), X is 0 too. A 0/0 with no counters dies.
    for source in ["Call Forth the Tempest", "Chaos Warp", "Wild Magic Surge"] {
        let mut state = game();
        let player = if source == "Call Forth the Tempest" { P0 } else { P1 };
        for i in 0..10 {
            library_card(&mut state, player, i, "Shivan Devastator");
        }
        let target = match source {
            "Chaos Warp" => Some(put(&mut state, P1, "Treasure Token")),
            "Wild Magic Surge" => Some(put(&mut state, P1, "Guttersnipe")),
            _ => None,
        };
        lands(&mut state, P0, 8);
        let spell = put_in(&mut state, P0, source, Zone::Hand);
        advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
        engine::step(&mut state, Action::CastSpell(spell)).unwrap();
        let mut cast_by_cascade = 0;
        drive(&mut state, |s, d| match d {
            Decision::ChooseTargets { .. } if target.is_some() => Some(Action::ChooseTarget(Target::Object(target.unwrap()))),
            Decision::ChooseCascadeCast { .. } => {
                cast_by_cascade += 1;
                Some(Action::ChooseCascadeCast(true))
            }
            Decision::CastSpellOrPass { .. } if !s.stack.is_empty() => Some(Action::Pass),
            _ => first_choice(s, d),
        });
        if source == "Call Forth the Tempest" {
            assert_eq!(cast_by_cascade, 2, "both cascades hit a Shivan");
        }
        let graveyard = &state.players[player.index()].graveyard;
        let dead = graveyard.iter().filter(|&&o| state.objects.get(o).name == "Shivan Devastator").count();
        assert_eq!(dead, if source == "Call Forth the Tempest" { 2 } else { 1 }, "{source}");
        assert!(named(&state, player, "Shivan Devastator").is_empty(), "{source}");
    }
}
