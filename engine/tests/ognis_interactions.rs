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

