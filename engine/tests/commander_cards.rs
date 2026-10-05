//! Commander deck cards, driven through the public decision API.

use mtg_kernel::card_def::{card_id_by_name, CARD_DEFS};
use mtg_kernel::engine::{self, effective_power, Action, Decision};
use mtg_kernel::ids::{ObjectId, PlayerId};
use mtg_kernel::mana::ManaColor;
use mtg_kernel::state::{GameObject, GameState, ObjectStateV4, Step, Target, Zone};

const P0: PlayerId = PlayerId::P0;
const P1: PlayerId = PlayerId::P1;

fn id(name: &str) -> u16 {
    card_id_by_name(name).unwrap_or_else(|| panic!("{name} not in CARD_DEFS"))
}

/// A Commander game in player 0's first main phase, both libraries ten Mountains.
fn game() -> GameState {
    let lib = vec![id("Mountain"); 10];
    let mut state = GameState::new_commander_game_v1(
        &lib,
        &lib,
        [id("Ognis, the Dragon's Lash"), id("Ognis, the Dragon's Lash")],
        |c| CARD_DEFS[c as usize].name.to_string(),
        1,
        P0,
    );
    state.turn = 2;
    state.step = Step::Main1;
    state
}

fn put(state: &mut GameState, player: PlayerId, name: &str) -> ObjectId {
    put_in(state, player, name, Zone::Battlefield)
}

fn put_in(state: &mut GameState, player: PlayerId, name: &str, zone: Zone) -> ObjectId {
    let card_def = id(name);
    let obj = state.objects.push(GameObject {
        card_def,
        name: name.to_string(),
        owner: player,
        controller: player,
        zone,
        tapped: false,
        summoning_sick: false,
        damage: 0,
        counters: Default::default(),
        attachments: Vec::new(),
        v4: ObjectStateV4::from_card_def(card_def),
        spell_copy_origin: None,
        plotted_turn: None,
        zone_change_count: 0,
    });
    match zone {
        Zone::Battlefield => state.players[player.index()].battlefield.push(obj),
        Zone::Hand => state.players[player.index()].hand.push(obj),
        _ => panic!("put_in {zone:?}"),
    }
    obj
}

fn lands(state: &mut GameState, player: PlayerId, n: usize) {
    for _ in 0..n {
        put(state, player, "Mountain");
    }
}

/// Casts `card` from player 0's hand with `target` for every target choice and
/// `mode` for a modal spell, then resolves the stack.
fn cast(state: &mut GameState, card: &str, target: Option<Target>, mode: u8) -> ObjectId {
    let spell = put_in(state, P0, card, Zone::Hand);
    advance(state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    engine::step(state, Action::CastSpell(spell)).unwrap();
    loop {
        let d = engine::advance_until_decision(state);
        let a = match &d {
            Decision::ChooseTargets { legal_targets, .. } | Decision::ChooseEffectTargets { legal_targets, .. } => {
                let t = target.expect("a target");
                assert!(legal_targets.contains(&t), "{t:?} not in {legal_targets:?}");
                Action::ChooseTarget(t)
            }
            Decision::ChooseSpellMode { .. } => Action::ChooseSpellMode(mode),
            Decision::CastSpellOrPass { .. } if !state.stack.is_empty() => Action::Pass,
            Decision::CastSpellOrPass { .. } => return spell,
            other => panic!("unhandled decision {other:?}"),
        };
        engine::step(state, a).unwrap();
    }
}

/// Passes to the next player's turn.
fn end_turn(state: &mut GameState) {
    let me = state.active_player;
    advance(state, |s, _| s.active_player != me);
}

fn named(state: &GameState, player: PlayerId, name: &str) -> Vec<ObjectId> {
    state.players[player.index()].battlefield.iter().copied().filter(|&o| CARD_DEFS[state.objects.get(o).card_def as usize].name == name).collect()
}

/// Passes priority (and declines blocks) until `stop` accepts a decision.
fn advance(state: &mut GameState, mut stop: impl FnMut(&GameState, &Decision) -> bool) -> Decision {
    for _ in 0..500 {
        let d = engine::advance_until_decision(state);
        if stop(state, &d) {
            return d;
        }
        let a = match &d {
            Decision::CastSpellOrPass { .. } => Action::Pass,
            Decision::DeclareAttackers { .. } => Action::DeclareAttackers(vec![]),
            Decision::DeclareBlockers { .. } => Action::DeclareBlockers(vec![]),
            Decision::OrderTriggers { pending, .. } => Action::OrderTriggers((0..pending.len()).collect()),
            Decision::ChooseTargets { legal_targets, .. } => {
                let opponent = Target::Player(P1);
                Action::ChooseTarget(if legal_targets.contains(&opponent) { opponent } else { legal_targets[0] })
            }
            other => panic!("unhandled decision {other:?}"),
        };
        engine::step(state, a).unwrap();
    }
    panic!("no stop after 500 decisions");
}

/// Attacks with `attackers`, choosing the opponent for any targeted trigger,
/// and passes until combat damage is done.
fn attack_with(state: &mut GameState, attackers: Vec<ObjectId>) {
    advance(state, |_, d| matches!(d, Decision::DeclareAttackers { .. }));
    engine::step(state, Action::DeclareAttackers(attackers)).unwrap();
    advance(state, |s, _| s.step == Step::EndCombat || s.step == Step::Main2);
}

#[test]
fn ognis_makes_a_tapped_treasure_for_each_hasty_attacker() {
    let mut state = game();
    let ognis = put(&mut state, P0, "Ognis, the Dragon's Lash");
    let storm = put(&mut state, P0, "Captain Lannery Storm");
    let slow = put(&mut state, P0, "Guttersnipe");
    attack_with(&mut state, vec![ognis, storm, slow]);
    let treasures = named(&state, P0, "Treasure Token");
    let tapped = treasures.iter().filter(|&&t| state.objects.get(t).tapped).count();
    // Ognis and Lannery have haste (two tapped); Lannery's own trigger adds one untapped.
    assert_eq!((treasures.len(), tapped), (3, 2));
}

#[test]
fn smaug_makes_a_treasure_each_upkeep_and_burns_for_the_treasures() {
    let mut state = game();
    let smaug = put(&mut state, P0, "Smaug the Magnificent");
    put(&mut state, P0, "Treasure Token");
    attack_with(&mut state, vec![smaug]);
    // One Treasure: 1 damage from the trigger, then 4 combat damage.
    assert_eq!(state.players[1].life, 40 - 1 - 4);

    // Through player 1's turn to player 0's next upkeep.
    advance(&mut state, |s, _| s.active_player == P0 && s.step == Step::Upkeep && !s.stack.is_empty());
    advance(&mut state, |s, _| s.step == Step::Draw);
    assert_eq!(named(&state, P0, "Treasure Token").len(), 2);
}

#[test]
fn lannery_gets_power_when_a_treasure_is_sacrificed() {
    let mut state = game();
    let storm = put(&mut state, P0, "Captain Lannery Storm");
    let treasure = put(&mut state, P0, "Treasure Token");
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    engine::step(&mut state, Action::ActivateManaAbilityChoice(treasure, ManaColor::R)).unwrap();
    advance(&mut state, |s, _| s.stack.is_empty());
    assert!(named(&state, P0, "Treasure Token").is_empty());
    assert_eq!(effective_power(&state, storm), 3);
}

#[test]
fn zealous_conscripts_steals_untaps_and_hastes_until_end_of_turn() {
    let mut state = game();
    lands(&mut state, P0, 5);
    let theirs = put(&mut state, P1, "Guttersnipe");
    state.objects.get_mut(theirs).tapped = true;
    cast(&mut state, "Zealous Conscripts", Some(Target::Object(theirs)), 0);
    let o = state.objects.get(theirs);
    assert_eq!(o.controller, P0);
    assert!(state.players[0].battlefield.contains(&theirs) && !state.players[1].battlefield.contains(&theirs));
    assert!(!o.tapped);
    assert!(engine::has_effective_keyword(&state, theirs, mtg_kernel::card_def::Keywords::HASTE));
    match advance(&mut state, |_, d| matches!(d, Decision::DeclareAttackers { .. })) {
        Decision::DeclareAttackers { eligible, .. } => assert!(eligible.contains(&theirs)),
        _ => unreachable!(),
    }
    end_turn(&mut state);
    assert_eq!(state.objects.get(theirs).controller, P1);
    assert!(state.players[1].battlefield.contains(&theirs) && !state.players[0].battlefield.contains(&theirs));
}

#[test]
fn mob_rule_takes_big_or_small_creatures() {
    for (mode, big) in [(0u8, true), (1u8, false)] {
        let mut state = game();
        lands(&mut state, P0, 6);
        let enforcer = put(&mut state, P1, "Myr Enforcer");
        let snipe = put(&mut state, P1, "Guttersnipe");
        cast(&mut state, "Mob Rule", None, mode);
        assert_eq!(state.objects.get(enforcer).controller == P0, big, "mode {mode}");
        assert_eq!(state.objects.get(snipe).controller == P0, !big, "mode {mode}");
    }
}

#[test]
fn insurrection_takes_every_creature() {
    let mut state = game();
    lands(&mut state, P0, 8);
    let mine = put(&mut state, P0, "Guttersnipe");
    state.objects.get_mut(mine).tapped = true;
    let theirs = [put(&mut state, P1, "Myr Enforcer"), put(&mut state, P1, "Guttersnipe")];
    cast(&mut state, "Insurrection", None, 0);
    for o in theirs.into_iter().chain([mine]) {
        assert_eq!(state.objects.get(o).controller, P0);
        assert!(!state.objects.get(o).tapped);
    }
    end_turn(&mut state);
    assert!(theirs.iter().all(|&o| state.objects.get(o).controller == P1));
    assert_eq!(state.objects.get(mine).controller, P0);
}

#[test]
fn mass_mutiny_molten_primordial_and_lokis_scepter_steal_a_creature() {
    for (card, mana) in [("Mass Mutiny", 5), ("Molten Primordial", 7), ("Loki's Scepter", 3)] {
        let mut state = game();
        lands(&mut state, P0, mana);
        let theirs = put(&mut state, P1, "Guttersnipe");
        cast(&mut state, card, Some(Target::Object(theirs)), 0);
        assert_eq!(state.objects.get(theirs).controller, P0, "{card}");
        end_turn(&mut state);
        assert_eq!(state.objects.get(theirs).controller, P1, "{card}");
    }
}

#[test]
fn sol_ring_pays_two_and_floats_what_the_cost_leaves() {
    let mut state = game();
    put(&mut state, P0, "Sol Ring");
    put(&mut state, P0, "Mountain");
    let storm = cast(&mut state, "Captain Lannery Storm", None, 0);
    assert_eq!(state.objects.get(storm).zone, Zone::Battlefield, "{{2}}{{R}} paid by Sol Ring and a Mountain");

    let mut state = game();
    put(&mut state, P0, "Sol Ring");
    let second = put_in(&mut state, P0, "Sol Ring", Zone::Hand);
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    engine::step(&mut state, Action::CastSpell(second)).unwrap();
    engine::advance_until_decision(&mut state);
    assert_eq!(state.players[0].mana_pool[ManaColor::C.pool_index()], 1, "one of Sol Ring's two floats");
}

#[test]
fn identity_mana_makes_only_the_commanders_colors() {
    for source in ["Arcane Signet", "Command Tower"] {
        let mut state = game();
        let id = put(&mut state, P0, source);
        advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
        let colors = engine::available_mana_ability_choices(P0, id, &state);
        assert_eq!(colors, vec![ManaColor::B, ManaColor::R, ManaColor::G], "{source} under Ognis");
    }
}

/// Declares `attackers`, answers trigger targets with `target`, and returns the
/// blockers player 1 may use against each attacker.
fn attack_and_see_blocks(state: &mut GameState, attackers: Vec<ObjectId>, target: Target) -> Vec<(ObjectId, Vec<ObjectId>)> {
    advance(state, |_, d| matches!(d, Decision::DeclareAttackers { .. }));
    engine::step(state, Action::DeclareAttackers(attackers)).unwrap();
    loop {
        match engine::advance_until_decision(state) {
            Decision::ChooseTargets { .. } => engine::step(state, Action::ChooseTarget(target)).unwrap(),
            Decision::CastSpellOrPass { .. } => engine::step(state, Action::Pass).unwrap(),
            Decision::DeclareBlockers { legal_blockers, .. } => return legal_blockers,
            Decision::OrderTriggers { pending, .. } => engine::step(state, Action::OrderTriggers((0..pending.len()).collect())).unwrap(),
            other => panic!("unhandled decision {other:?}"),
        }
    }
}

#[test]
fn spider_man_stops_a_blocker() {
    let mut state = game();
    let spidey = put(&mut state, P0, "Spider-Man, Web-Spinner");
    let wall = put(&mut state, P1, "Guttersnipe");
    let other = put(&mut state, P1, "Myr Enforcer");
    let blocks = attack_and_see_blocks(&mut state, vec![spidey], Target::Object(wall));
    let can_block = &blocks.iter().find(|b| b.0 == spidey).unwrap().1;
    assert!(!can_block.contains(&wall) && can_block.contains(&other), "{can_block:?}");
}

#[test]
fn whirlwind_triggers_once_and_only_for_creatures_that_entered_this_turn() {
    for (new_attacker, expect_trigger) in [(true, true), (false, false)] {
        let mut state = game();
        let whirl = put(&mut state, P0, "Whirlwind, Killer Cyclone");
        let veteran = put(&mut state, P0, "Captain Lannery Storm");
        if new_attacker {
            state.objects.get_mut(whirl).v4.entered_battlefield_turn = Some(state.turn);
        }
        let wall = put(&mut state, P1, "Guttersnipe");
        let attackers = if new_attacker { vec![whirl, veteran] } else { vec![veteran] };
        let blocks = attack_and_see_blocks(&mut state, attackers, Target::Object(wall));
        let blocked_by_wall = blocks.iter().any(|b| b.1.contains(&wall));
        assert_eq!(blocked_by_wall, !expect_trigger, "entered this turn: {new_attacker}");
    }
}

#[test]
fn knuckles_makes_a_treasure_per_combat_damage_step_and_wins_at_thirty_artifacts() {
    let mut state = game();
    let knuckles = put(&mut state, P0, "Knuckles the Echidna");
    let storm = put(&mut state, P0, "Captain Lannery Storm");
    attack_with(&mut state, vec![knuckles, storm]);
    // Lannery's own trigger makes one; Knuckles makes one in each of the two damage steps.
    assert_eq!(named(&state, P0, "Treasure Token").len(), 3);

    let mut state = game();
    put(&mut state, P0, "Knuckles the Echidna");
    for _ in 0..30 {
        put(&mut state, P0, "Treasure Token");
    }
    let d = advance(&mut state, |s, d| matches!(d, Decision::GameOver { .. }) || (s.active_player == P0 && s.turn > 3));
    assert!(matches!(d, Decision::GameOver { winner: Some(P0) }), "{d:?}");
}

#[test]
fn wrenns_resolve_exiles_two_cards_to_play() {
    let mut state = game();
    lands(&mut state, P0, 2);
    cast(&mut state, "Wrenn's Resolve", None, 0);
    let exiled: Vec<_> = state.exile.iter().filter(|&&o| state.objects.get(o).owner == P0).collect();
    assert_eq!(exiled.len(), 2);
    assert_eq!(state.players[0].library.len(), 8);
}
