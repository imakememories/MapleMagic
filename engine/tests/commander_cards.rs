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
        Zone::Graveyard => state.players[player.index()].graveyard.push(obj),
        _ => panic!("put_in {zone:?}"),
    }
    obj
}

fn lands(state: &mut GameState, player: PlayerId, n: usize) {
    for _ in 0..n {
        put(state, player, "Mountain");
    }
}

/// Casts `card` from player 0's hand with `target` for every target choice (no
/// target where "up to" allows none) and `mode` for a modal spell or as X,
/// then resolves the stack.
fn cast(state: &mut GameState, card: &str, target: Option<Target>, mode: u8) -> ObjectId {
    let spell = put_in(state, P0, card, Zone::Hand);
    advance(state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    engine::step(state, Action::CastSpell(spell)).unwrap();
    loop {
        let d = engine::advance_until_decision(state);
        let a = match &d {
            Decision::ChooseTargets { legal_targets, can_finish, .. }
            | Decision::ChooseEffectTargets { legal_targets, can_finish, .. } => match target {
                Some(t) => {
                    assert!(legal_targets.contains(&t), "{t:?} not in {legal_targets:?}");
                    if matches!(d, Decision::ChooseEffectTargets { .. }) {
                        Action::ChooseEffectTarget(t)
                    } else {
                        Action::ChooseTarget(t)
                    }
                }
                // "Up to" targets: choose none.
                None if *can_finish => Action::FinishEffectSelection,
                None => panic!("a target is required"),
            },
            Decision::ChooseSpellMode { .. } => Action::ChooseSpellMode(mode),
            Decision::ChooseEffectOption { .. } => Action::ChooseEffectOption(u16::from(mode)),
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

#[test]
fn headliner_scarlett_stops_blockers_then_exiles_a_card_only_its_owner_sees() {
    let mut state = game();
    lands(&mut state, P0, 4);
    let wall = put(&mut state, P1, "Guttersnipe");
    let scarlett = cast(&mut state, "Headliner Scarlett", Some(Target::Player(P1)), 0);
    let blocks = attack_and_see_blocks(&mut state, vec![scarlett], Target::Player(P1));
    assert!(blocks.iter().all(|b| !b.1.contains(&wall)), "{blocks:?}");

    // Next upkeep: the top card goes face down, and its owner may play it this turn.
    let turn = state.turn;
    advance(&mut state, |s, d| {
        s.active_player == P0 && s.turn != turn && s.step == Step::Main1 && matches!(d, Decision::CastSpellOrPass { .. })
    });
    let card = *state.exile.last().expect("an exiled card");
    assert!(!state.face_down_hidden_from(card, P0) && state.face_down_hidden_from(card, P1));
    match engine::advance_until_decision(&mut state) {
        Decision::CastSpellOrPass { land_drops, .. } => assert!(land_drops.contains(&card), "{land_drops:?}"),
        other => panic!("{other:?}"),
    }
    end_turn(&mut state);
    assert!(state.engine.exile_play_permissions.iter().all(|p| p.object != card));
    assert!(state.face_down(card).is_some(), "still face down after the turn");
}

#[test]
fn connecting_the_dots_exiles_unseen_cards_and_cashes_them_in() {
    let mut state = game();
    lands(&mut state, P0, 2);
    let dots = put(&mut state, P0, "Connecting the Dots");
    let a = put(&mut state, P0, "Captain Lannery Storm");
    let b = put(&mut state, P0, "Guttersnipe");
    let bolt = put_in(&mut state, P0, "Lightning Bolt", Zone::Hand);
    attack_with(&mut state, vec![a, b]);
    let exiled = state.exile.clone();
    assert_eq!(exiled.len(), 2);
    assert!(exiled.iter().all(|&c| state.face_down_hidden_from(c, P0) && state.face_down_hidden_from(c, P1)));

    // {1}{R}, discard the hand, sacrifice: the exiled cards come back to hand.
    advance(&mut state, |_, d| {
        matches!(d, Decision::CastSpellOrPass { activatable_abilities, .. } if activatable_abilities.contains(&(dots, 0)))
    });
    engine::step(&mut state, Action::ActivateAbility(dots, 0)).unwrap();
    engine::advance_until_decision(&mut state); // pays the cost
    assert_eq!(state.objects.get(bolt).zone, Zone::Graveyard);
    assert_eq!(state.objects.get(dots).zone, Zone::Graveyard);
    advance(&mut state, |s, d| s.stack.is_empty() && matches!(d, Decision::CastSpellOrPass { .. }));
    for c in exiled {
        assert_eq!(state.objects.get(c).zone, Zone::Hand);
        assert!(state.face_down(c).is_none());
        assert!(!state.known_hand_cards(P1, P0).iter().any(|e| e.object == c), "never shown to the opponent");
    }
}

/// Passes priority until the stack is empty.
fn resolve(state: &mut GameState) {
    while !state.stack.is_empty() {
        match engine::advance_until_decision(state) {
            Decision::CastSpellOrPass { .. } => engine::step(state, Action::Pass).unwrap(),
            other => panic!("unhandled decision {other:?}"),
        }
    }
}

/// Damage on player 1's Myr Enforcer (4/4; `None` once dead) and on player 0's creature, and player 1's life.
fn fireball_marks(state: &GameState, theirs: ObjectId, mine: ObjectId) -> (Option<u16>, u16, i32) {
    let o = state.objects.get(theirs);
    ((o.zone == Zone::Battlefield).then_some(o.damage), state.objects.get(mine).damage, state.players[1].life)
}

#[test]
fn delayed_blast_fireball_deals_two_from_hand() {
    let mut state = game();
    lands(&mut state, P0, 3);
    let theirs = put(&mut state, P1, "Myr Enforcer");
    let mine = put(&mut state, P0, "Captain Lannery Storm");
    cast(&mut state, "Delayed Blast Fireball", None, 0);
    assert_eq!(fireball_marks(&state, theirs, mine), (Some(2), 0, 38));
}

#[test]
fn foretold_delayed_blast_fireball_waits_a_turn_then_deals_five() {
    let mut state = game();
    lands(&mut state, P0, 2);
    let theirs = put(&mut state, P1, "Myr Enforcer");
    let mine = put(&mut state, P0, "Captain Lannery Storm");
    let fireball = put_in(&mut state, P0, "Delayed Blast Fireball", Zone::Hand);
    match advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. })) {
        Decision::CastSpellOrPass { foretell_actions, .. } => assert_eq!(foretell_actions, vec![fireball]),
        other => panic!("{other:?}"),
    }
    engine::step(&mut state, Action::Foretell(fireball)).unwrap();
    assert!(state.face_down(fireball).is_some_and(|f| f.foretold_turn.is_some()));
    assert!(!state.face_down_hidden_from(fireball, P0) && state.face_down_hidden_from(fireball, P1));
    lands(&mut state, P0, 6);
    match engine::advance_until_decision(&mut state) {
        Decision::CastSpellOrPass { castable_spells, .. } => assert!(!castable_spells.contains(&fireball), "not the turn it was foretold"),
        other => panic!("{other:?}"),
    }
    // On the opponent's next turn it can be cast for {4}{R}{R}.
    advance(&mut state, |s, d| {
        s.active_player == P1 && matches!(d, Decision::CastSpellOrPass { player: P0, castable_spells, .. } if castable_spells.contains(&fireball))
    });
    engine::step(&mut state, Action::CastSpell(fireball)).unwrap();
    resolve(&mut state);
    assert_eq!(fireball_marks(&state, theirs, mine), (None, 0, 35));
    assert_eq!(state.objects.get(fireball).zone, Zone::Graveyard);
}

#[test]
fn delayed_blast_fireball_cast_from_any_exile_deals_five() {
    let mut state = game();
    lands(&mut state, P0, 5);
    let theirs = put(&mut state, P1, "Myr Enforcer");
    let mine = put(&mut state, P0, "Captain Lannery Storm");
    let top = state.players[0].library[0];
    let def = id("Delayed Blast Fireball");
    let o = state.objects.get_mut(top);
    o.card_def = def;
    o.name = "Delayed Blast Fireball".into();
    o.v4 = ObjectStateV4::from_card_def(def);
    cast(&mut state, "Wrenn's Resolve", None, 0);
    match engine::advance_until_decision(&mut state) {
        Decision::CastSpellOrPass { castable_spells, .. } => assert!(castable_spells.contains(&top), "{castable_spells:?}"),
        other => panic!("{other:?}"),
    }
    engine::step(&mut state, Action::CastSpell(top)).unwrap();
    resolve(&mut state);
    assert_eq!(fireball_marks(&state, theirs, mine), (None, 0, 35));
}

#[test]
fn mass_mutiny_and_molten_primordial_may_take_nothing() {
    // "Up to one target" allows zero targets (115.6), even when a creature could be taken.
    for (card, mana) in [("Mass Mutiny", 5), ("Molten Primordial", 7)] {
        let mut state = game();
        lands(&mut state, P0, mana);
        let theirs = put(&mut state, P1, "Guttersnipe");
        cast(&mut state, card, None, 0);
        assert_eq!(state.objects.get(theirs).controller, P1, "{card}");
    }
    // Mass Mutiny can be cast when the opponent controls no creatures (2012-06-01 ruling).
    let mut state = game();
    lands(&mut state, P0, 5);
    let mutiny = cast(&mut state, "Mass Mutiny", None, 0);
    assert_eq!(state.objects.get(mutiny).zone, Zone::Graveyard);
}

/// Advances to player 0's next upkeep, where it holds priority.
fn to_my_next_upkeep(state: &mut GameState) {
    let turn = state.turn;
    advance(state, |s, d| {
        s.turn != turn && s.active_player == P0 && s.step == Step::Upkeep && matches!(d, Decision::CastSpellOrPass { player: P0, .. })
    });
}

#[test]
fn knuckles_checks_thirty_artifacts_as_upkeep_begins_and_again_on_resolution() {
    // 603.4: with 29 artifacts as the upkeep begins, the ability doesn't trigger at all.
    let mut state = game();
    put(&mut state, P0, "Knuckles the Echidna");
    for _ in 0..29 {
        put(&mut state, P0, "Treasure Token");
    }
    to_my_next_upkeep(&mut state);
    assert!(state.stack.is_empty());

    // With 30 it triggers; dropping to 29 before it resolves makes it do nothing.
    let mut state = game();
    put(&mut state, P0, "Knuckles the Echidna");
    let treasures: Vec<_> = (0..30).map(|_| put(&mut state, P0, "Treasure Token")).collect();
    to_my_next_upkeep(&mut state);
    assert_eq!(state.stack.len(), 1);
    engine::step(&mut state, Action::ActivateManaAbilityChoice(treasures[0], ManaColor::R)).unwrap();
    let d = advance(&mut state, |s, d| matches!(d, Decision::GameOver { .. }) || s.step == Step::Draw);
    assert!(!matches!(d, Decision::GameOver { .. }), "{d:?}");
}

#[test]
fn lokis_scepter_makes_the_stolen_creature_a_villain_until_end_of_turn() {
    use mtg_kernel::card_def::Subtype;
    let mut state = game();
    lands(&mut state, P0, 3);
    let theirs = put(&mut state, P1, "Guttersnipe");
    cast(&mut state, "Loki's Scepter", Some(Target::Object(theirs)), 0);
    assert!(engine::has_effective_subtype(&state, theirs, Subtype::Villain));
    assert!(engine::effective_subtype_ids(&state, theirs).contains(&Subtype::Villain.stable_id()));
    assert!(engine::has_effective_subtype(&state, theirs, Subtype::Goblin), "in addition to its other types");
    end_turn(&mut state);
    assert!(!engine::has_effective_subtype(&state, theirs, Subtype::Villain));
    assert!(engine::has_effective_subtype(&state, theirs, Subtype::Goblin));
}

/// Activates `source`'s ability `index` and resolves it: `pick` for a target
/// when legal, else the first option offered at each choice.
fn activate(state: &mut GameState, source: ObjectId, index: u8, pick: Option<Target>) {
    advance(state, |_, d| {
        matches!(d, Decision::CastSpellOrPass { activatable_abilities, .. } if activatable_abilities.contains(&(source, index)))
    });
    engine::step(state, Action::ActivateAbility(source, index)).unwrap();
    loop {
        let a = match engine::advance_until_decision(state) {
            Decision::ChooseTargets { legal_targets, .. } => {
                Action::ChooseTarget(pick.filter(|t| legal_targets.contains(t)).unwrap_or(legal_targets[0]))
            }
            Decision::ChooseEffectTargets { legal_targets, .. } => {
                Action::ChooseEffectTarget(pick.filter(|t| legal_targets.contains(t)).unwrap_or(legal_targets[0]))
            }
            Decision::ChooseCostTargets { candidates, .. } => Action::ChooseCostTarget(candidates[0]),
            Decision::OrderTriggers { pending, .. } => Action::OrderTriggers((0..pending.len()).collect()),
            Decision::CastSpellOrPass { .. } if !state.stack.is_empty() => Action::Pass,
            Decision::CastSpellOrPass { .. } => return,
            other => panic!("unhandled decision {other:?}"),
        };
        engine::step(state, a).unwrap();
    }
}

#[test]
fn strip_mine_taps_for_colorless_and_destroys_a_land() {
    let mut state = game();
    let mine = put(&mut state, P0, "Strip Mine");
    let theirs = put(&mut state, P1, "Mountain");
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    assert_eq!(engine::available_mana_ability_choices(P0, mine, &state), vec![ManaColor::C]);
    activate(&mut state, mine, 0, Some(Target::Object(theirs)));
    assert_eq!(state.objects.get(theirs).zone, Zone::Graveyard);
    assert_eq!(state.objects.get(mine).zone, Zone::Graveyard);
}

#[test]
fn urzas_cave_fetches_a_land_tapped_which_amulet_of_vigor_untaps() {
    for amulet in [false, true] {
        let mut state = game();
        lands(&mut state, P0, 3);
        let cave = put(&mut state, P0, "Urza's Cave");
        if amulet {
            put(&mut state, P0, "Amulet of Vigor");
        }
        activate(&mut state, cave, 0, None);
        let mountains = named(&state, P0, "Mountain");
        assert_eq!(mountains.len(), 4);
        assert_eq!(state.objects.get(*mountains.last().unwrap()).tapped, !amulet, "amulet {amulet}");
        assert_eq!(state.players[0].library.len(), 9);
        assert_eq!(state.objects.get(cave).zone, Zone::Graveyard);
    }
}

#[test]
fn amulet_of_vigor_untaps_ognis_treasures_which_enter_tapped() {
    let mut state = game();
    put(&mut state, P0, "Amulet of Vigor");
    let ognis = put(&mut state, P0, "Ognis, the Dragon's Lash");
    attack_with(&mut state, vec![ognis]);
    let treasure = named(&state, P0, "Treasure Token")[0];
    assert!(!state.objects.get(treasure).tapped);
}

#[test]
fn scavenger_grounds_can_sacrifice_itself_and_exiles_every_graveyard() {
    let mut state = game();
    lands(&mut state, P0, 2);
    let grounds = put(&mut state, P0, "Scavenger Grounds");
    put_in(&mut state, P0, "Lightning Bolt", Zone::Graveyard);
    put_in(&mut state, P1, "Guttersnipe", Zone::Graveyard);
    activate(&mut state, grounds, 0, None);
    assert!(state.players[0].graveyard.is_empty() && state.players[1].graveyard.is_empty());
    assert_eq!(state.objects.get(grounds).zone, Zone::Exile, "the sacrificed Desert is exiled too");
}

#[test]
fn war_room_costs_one_life_per_color_in_the_commanders_identity() {
    // Ognis, the Dragon's Lash is black, red and green.
    let mut state = game();
    lands(&mut state, P0, 3);
    let room = put(&mut state, P0, "War Room");
    let hand = state.players[0].hand.len();
    activate(&mut state, room, 0, None);
    assert_eq!(state.players[0].life, 37, "Ognis's identity is black, red and green");
    assert_eq!(state.players[0].hand.len(), hand + 1);
}

#[test]
fn light_up_the_stage_has_spectacle_once_an_opponent_lost_life() {
    use mtg_kernel::engine::CastMode;
    use mtg_kernel::event::{propose_and_commit, ProposedEvent};
    let mut state = game();
    put(&mut state, P0, "Mountain");
    let stage = put_in(&mut state, P0, "Light Up the Stage", Zone::Hand);
    let castable = |s: &mut GameState| {
        matches!(engine::advance_until_decision(s), Decision::CastSpellOrPass { castable_spells, .. } if castable_spells.contains(&stage))
    };
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    assert!(!castable(&mut state), "{{2}}{{R}} with one land");
    // Losing life and then gaining more still counts (2024-01-12 ruling).
    propose_and_commit(&mut state, ProposedEvent::life_loss(P1, 1));
    propose_and_commit(&mut state, ProposedEvent::life_gain(P1, 2));
    assert!(castable(&mut state));
    engine::step(&mut state, Action::CastSpell(stage)).unwrap();
    loop {
        let a = match engine::advance_until_decision(&mut state) {
            Decision::ChooseCastMode { .. } => Action::ChooseCastMode(CastMode::Alternative),
            Decision::CastSpellOrPass { .. } if !state.stack.is_empty() => Action::Pass,
            Decision::CastSpellOrPass { .. } => break,
            other => panic!("unhandled decision {other:?}"),
        };
        engine::step(&mut state, a).unwrap();
    }
    let exiled = state.exile.iter().filter(|&&o| state.objects.get(o).owner == P0).count();
    assert_eq!(exiled, 2);
    assert_eq!(state.objects.get(stage).zone, Zone::Graveyard);
}

#[test]
fn slickshot_show_off_grows_on_noncreature_spells_and_can_be_plotted() {
    let mut state = game();
    lands(&mut state, P0, 2);
    let show_off = put(&mut state, P0, "Slickshot Show-Off");
    let base = effective_power(&state, show_off);
    cast(&mut state, "Lightning Bolt", Some(Target::Player(P1)), 0);
    assert_eq!(effective_power(&state, show_off), base + 2);
    cast(&mut state, "Masked Meower", None, 0);
    assert_eq!(effective_power(&state, show_off), base + 2, "a creature spell doesn't count");

    let mut state = game();
    lands(&mut state, P0, 2);
    let card = put_in(&mut state, P0, "Slickshot Show-Off", Zone::Hand);
    match advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. })) {
        Decision::CastSpellOrPass { plot_actions, .. } => assert_eq!(plot_actions, vec![card]),
        other => panic!("{other:?}"),
    }
}

#[test]
fn laelia_exiles_a_card_to_play_and_grows_once_per_batch_of_exiled_cards() {
    let mut state = game();
    let laelia = put(&mut state, P0, "Laelia, the Blade Reforged");
    attack_with(&mut state, vec![laelia]);
    let exiled: Vec<_> = state.exile.iter().copied().filter(|&o| state.objects.get(o).owner == P0).collect();
    assert_eq!(exiled.len(), 1);
    assert!(state.engine.exile_play_permissions.iter().any(|p| p.object == exiled[0]));
    assert_eq!(state.objects.get(laelia).counters.plus1_plus1, 1);
    lands(&mut state, P0, 2);
    advance(&mut state, |s, _| s.step == Step::Main2);
    cast(&mut state, "Wrenn's Resolve", None, 0);
    assert_eq!(state.objects.get(laelia).counters.plus1_plus1, 2, "two cards at once: one counter");
}

#[test]
fn hobgoblin_gets_two_power_for_each_card_discarded() {
    let mut state = game();
    lands(&mut state, P0, 2);
    let hob = put(&mut state, P0, "Hobgoblin, Mantled Marauder");
    let dots = put(&mut state, P0, "Connecting the Dots");
    put_in(&mut state, P0, "Lightning Bolt", Zone::Hand);
    put_in(&mut state, P0, "Mountain", Zone::Hand);
    let base = effective_power(&state, hob);
    activate(&mut state, dots, 0, None);
    assert_eq!(effective_power(&state, hob), base + 4);
}

#[test]
fn goblin_war_drums_gives_only_your_creatures_menace() {
    use mtg_kernel::card_def::Keywords;
    let mut state = game();
    put(&mut state, P0, "Goblin War Drums");
    let mine = put(&mut state, P0, "Captain Lannery Storm");
    let theirs = put(&mut state, P1, "Guttersnipe");
    put(&mut state, P1, "Myr Enforcer");
    assert!(engine::has_effective_keyword(&state, mine, Keywords::MENACE));
    assert!(!engine::has_effective_keyword(&state, theirs, Keywords::MENACE));
    advance(&mut state, |_, d| matches!(d, Decision::DeclareAttackers { .. }));
    engine::step(&mut state, Action::DeclareAttackers(vec![mine])).unwrap();
    advance(&mut state, |_, d| matches!(d, Decision::DeclareBlockers { .. }));
    assert!(engine::step(&mut state.clone(), Action::DeclareBlockers(vec![(theirs, mine)])).is_err(), "one blocker can't block it");
}

fn offered_ability(state: &mut GameState, source: ObjectId, index: u8) -> bool {
    matches!(engine::advance_until_decision(state), Decision::CastSpellOrPass { activatable_abilities, .. } if activatable_abilities.contains(&(source, index)))
}

#[test]
fn shivan_devastator_enters_with_x_counters() {
    let mut state = game();
    lands(&mut state, P0, 4);
    let dragon = cast(&mut state, "Shivan Devastator", None, 3);
    assert_eq!(state.objects.get(dragon).zone, Zone::Battlefield);
    assert_eq!(state.objects.get(dragon).counters.plus1_plus1, 3);
}

#[test]
fn ruby_medallion_takes_one_generic_off_red_spells_x_included() {
    let mut state = game();
    put(&mut state, P0, "Ruby Medallion");
    lands(&mut state, P0, 2);
    let guttersnipe = cast(&mut state, "Guttersnipe", None, 0);
    assert_eq!(state.objects.get(guttersnipe).zone, Zone::Battlefield, "{{2}}{{R}} with two lands");

    let mut state = game();
    put(&mut state, P0, "Ruby Medallion");
    let ring = put_in(&mut state, P0, "Sol Ring", Zone::Hand);
    match advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. })) {
        Decision::CastSpellOrPass { castable_spells, .. } => assert!(!castable_spells.contains(&ring), "colorless isn't reduced"),
        other => panic!("{other:?}"),
    }

    // {X}{R} with X = 3 and three lands: the {1} left over comes off X.
    let mut state = game();
    put(&mut state, P0, "Ruby Medallion");
    lands(&mut state, P0, 3);
    let dragon = cast(&mut state, "Shivan Devastator", None, 3);
    assert_eq!(state.objects.get(dragon).counters.plus1_plus1, 3);
}

#[test]
fn goldspan_dragon_makes_treasures_that_each_make_two_mana() {
    let mut state = game();
    let dragon = put(&mut state, P0, "Goldspan Dragon");
    attack_with(&mut state, vec![dragon]);
    assert_eq!(named(&state, P0, "Treasure Token").len(), 1);

    let mut state = game();
    let dragon = put(&mut state, P0, "Goldspan Dragon");
    lands(&mut state, P0, 6);
    cast(&mut state, "Lightning Bolt", Some(Target::Object(dragon)), 0);
    assert_eq!(named(&state, P0, "Treasure Token").len(), 1, "a spell targeted it");
    cast(&mut state, "Zealous Conscripts", Some(Target::Object(dragon)), 0);
    assert_eq!(named(&state, P0, "Treasure Token").len(), 1, "an ability isn't a spell");
    let treasure = named(&state, P0, "Treasure Token")[0];
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    engine::step(&mut state, Action::ActivateManaAbilityChoice(treasure, ManaColor::R)).unwrap();
    assert_eq!(state.players[0].mana_pool[ManaColor::R.pool_index()], 2);
}

#[test]
fn cavern_hoard_dragon_costs_less_per_opposing_artifact_and_cashes_them_in() {
    let mut state = game();
    for _ in 0..3 {
        put(&mut state, P1, "Treasure Token");
    }
    lands(&mut state, P0, 5);
    let dragon = put_in(&mut state, P0, "Cavern-Hoard Dragon", Zone::Hand);
    match advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. })) {
        Decision::CastSpellOrPass { castable_spells, .. } => assert!(!castable_spells.contains(&dragon), "five lands"),
        other => panic!("{other:?}"),
    }
    put(&mut state, P0, "Mountain");
    state.players[0].hand.retain(|&o| o != dragon);
    let dragon = cast(&mut state, "Cavern-Hoard Dragon", None, 0);
    assert_eq!(state.objects.get(dragon).zone, Zone::Battlefield, "{{7}}{{R}}{{R}} less three");
    attack_with(&mut state, vec![dragon]);
    assert_eq!(named(&state, P0, "Treasure Token").len(), 3);
}

#[test]
fn screaming_nemesis_redirects_damage_dealt_at_once_and_stops_life_gain() {
    use mtg_kernel::event::{propose_and_commit, propose_and_commit_batch, ProposedEvent};
    let mut state = game();
    let nemesis = put(&mut state, P0, "Screaming Nemesis");
    let source = put(&mut state, P1, "Guttersnipe");
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    propose_and_commit_batch(
        &mut state,
        vec![
            ProposedEvent::damage(source, Target::Object(nemesis), 2),
            ProposedEvent::damage(source, Target::Object(nemesis), 1),
        ],
    );
    let mut triggers = 0;
    loop {
        let a = match engine::advance_until_decision(&mut state) {
            Decision::ChooseTargets { legal_targets, .. } => {
                triggers += 1;
                assert!(!legal_targets.contains(&Target::Object(nemesis)), "any other target");
                Action::ChooseTarget(Target::Player(P1))
            }
            Decision::CastSpellOrPass { .. } if !state.stack.is_empty() => Action::Pass,
            Decision::CastSpellOrPass { .. } => break,
            other => panic!("unhandled decision {other:?}"),
        };
        engine::step(&mut state, a).unwrap();
    }
    assert_eq!(triggers, 1);
    assert_eq!(state.players[1].life, 37);
    propose_and_commit(&mut state, ProposedEvent::life_gain(P1, 5));
    assert_eq!(state.players[1].life, 37, "can't gain life");
    propose_and_commit(&mut state, ProposedEvent::life_gain(P0, 5));
    assert_eq!(state.players[0].life, 45);
}

#[test]
fn idol_of_oblivion_draws_after_you_create_a_token_and_makes_an_eldrazi() {
    use mtg_kernel::event::{propose_and_commit, ProposedEvent};
    let mut state = game();
    let idol = put(&mut state, P0, "Idol of Oblivion");
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    assert!(!offered_ability(&mut state, idol, 0));
    propose_and_commit(&mut state, ProposedEvent::create_token(id("Treasure Token"), P1));
    assert!(!offered_ability(&mut state, idol, 0), "the opponent's token doesn't count");
    propose_and_commit(&mut state, ProposedEvent::create_token(id("Treasure Token"), P0));
    assert!(offered_ability(&mut state, idol, 0));
    let hand = state.players[0].hand.len();
    activate(&mut state, idol, 0, None);
    assert_eq!(state.players[0].hand.len(), hand + 1);

    let mut state = game();
    let idol = put(&mut state, P0, "Idol of Oblivion");
    lands(&mut state, P0, 8);
    activate(&mut state, idol, 1, None);
    let eldrazi = named(&state, P0, "Eldrazi Token");
    assert_eq!(eldrazi.len(), 1);
    assert_eq!(effective_power(&state, eldrazi[0]), 10);
}

#[test]
fn hyldas_crown_taps_free_on_your_turn_and_draws_per_tapped_opposing_creature() {
    let mut state = game();
    let crown = put(&mut state, P0, "Hylda's Crown of Winter");
    let theirs = put(&mut state, P1, "Guttersnipe");
    put(&mut state, P1, "Myr Enforcer");
    activate(&mut state, crown, 0, Some(Target::Object(theirs)));
    assert!(state.objects.get(theirs).tapped, "no mana needed on your turn");
    lands(&mut state, P0, 3);
    let hand = state.players[0].hand.len();
    activate(&mut state, crown, 1, None);
    assert_eq!(state.players[0].hand.len(), hand + 1, "one tapped opposing creature");

    let mut state = game();
    end_turn(&mut state);
    let crown = put(&mut state, P0, "Hylda's Crown of Winter");
    put(&mut state, P1, "Guttersnipe");
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { player: P0, .. }));
    assert!(!offered_ability(&mut state, crown, 0), "{{1}} on the opponent's turn");
    put(&mut state, P0, "Mountain");
    assert!(offered_ability(&mut state, crown, 0));
}

#[test]
fn dragonspark_reactor_counts_artifacts_and_burns_a_player_and_up_to_one_creature() {
    use mtg_kernel::event::{propose_and_commit, ProposedEvent};
    let mut state = game();
    lands(&mut state, P0, 2);
    let reactor = cast(&mut state, "Dragonspark Reactor", None, 0);
    assert_eq!(state.objects.get(reactor).counters.charge, 1, "its own entering counts");
    propose_and_commit(&mut state, ProposedEvent::create_token(id("Treasure Token"), P0));
    propose_and_commit(&mut state, ProposedEvent::create_token(id("Treasure Token"), P1));
    advance(&mut state, |s, _| s.stack.is_empty());
    assert_eq!(state.objects.get(reactor).counters.charge, 2, "only artifacts you control");
    let theirs = put(&mut state, P1, "Myr Enforcer");
    lands(&mut state, P0, 4);
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    engine::step(&mut state, Action::ActivateAbility(reactor, 0)).unwrap();
    loop {
        let a = match engine::advance_until_decision(&mut state) {
            Decision::ChooseEffectTargets { selected_count: 0, can_finish, .. } => {
                assert!(!can_finish, "a player must be targeted");
                Action::ChooseEffectTarget(Target::Player(P1))
            }
            Decision::ChooseEffectTargets { .. } => Action::ChooseEffectTarget(Target::Object(theirs)),
            Decision::CastSpellOrPass { .. } if !state.stack.is_empty() => Action::Pass,
            Decision::CastSpellOrPass { .. } => break,
            other => panic!("unhandled decision {other:?}"),
        };
        engine::step(&mut state, a).unwrap();
    }
    assert_eq!(state.players[1].life, 38);
    assert_eq!(state.objects.get(theirs).damage, 2);
}

#[test]
fn urabrask_impulses_for_you_and_turns_the_opponents_draw_into_an_exile() {
    let mut state = game();
    put(&mut state, P0, "Urabrask, Heretic Praetor");
    let turn = state.turn;
    // Player 1's upkeep: their draw this turn becomes an exile they may play.
    advance(&mut state, |s, _| s.active_player == P1 && s.step == Step::Main1);
    assert_eq!(state.players[1].hand.len(), 0, "no card drawn");
    assert_eq!(state.players[1].library.len(), 9);
    let exiled: Vec<_> = state.exile.iter().copied().filter(|&o| state.objects.get(o).owner == P1).collect();
    assert_eq!(exiled.len(), 1);
    assert!(state.engine.exile_play_permissions.iter().any(|p| p.object == exiled[0] && p.holder == P1));
    // Player 0's upkeep: exile its top card to play this turn.
    advance(&mut state, |s, _| s.active_player == P0 && s.turn != turn && s.step == Step::Main1);
    let mine: Vec<_> = state.exile.iter().copied().filter(|&o| state.objects.get(o).owner == P0).collect();
    assert_eq!(mine.len(), 1);
    assert!(state.engine.exile_play_permissions.iter().any(|p| p.object == mine[0] && p.holder == P0));
}

/// Player 0 attacks with `attacker` and player 1 blocks with `blocker`;
/// returns the attacker's damage and whether the blocker died.
fn attack_into_block(state: &mut GameState, attacker: ObjectId, blocker: ObjectId) -> (u16, bool) {
    advance(state, |_, d| matches!(d, Decision::DeclareAttackers { .. }));
    engine::step(state, Action::DeclareAttackers(vec![attacker])).unwrap();
    advance(state, |_, d| matches!(d, Decision::DeclareBlockers { .. }));
    engine::step(state, Action::DeclareBlockers(vec![(blocker, attacker)])).unwrap();
    advance(state, |s, _| s.step == Step::EndCombat);
    (state.objects.get(attacker).damage, state.objects.get(blocker).zone == Zone::Graveyard)
}

#[test]
fn dolmen_gate_prevents_combat_damage_to_your_attackers_unless_spider_punk_says_no() {
    let mut state = game();
    put(&mut state, P0, "Dolmen Gate");
    let attacker = put(&mut state, P0, "Myr Enforcer");
    let blocker = put(&mut state, P1, "Guttersnipe");
    assert_eq!(attack_into_block(&mut state, attacker, blocker), (0, true), "only the attacker is protected");

    let mut state = game();
    put(&mut state, P0, "Dolmen Gate");
    put(&mut state, P1, "Spider-Punk");
    let attacker = put(&mut state, P0, "Myr Enforcer");
    let blocker = put(&mut state, P1, "Guttersnipe");
    assert_eq!(attack_into_block(&mut state, attacker, blocker), (2, true), "damage can't be prevented");
}

/// Casts `card` for player 0, answering riot with `haste`.
fn cast_with_riot(state: &mut GameState, card: &str, haste: bool) -> ObjectId {
    let spell = put_in(state, P0, card, Zone::Hand);
    advance(state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    engine::step(state, Action::CastSpell(spell)).unwrap();
    loop {
        let a = match engine::advance_until_decision(state) {
            Decision::ChooseRiot { creature, .. } => {
                assert_eq!(creature, spell);
                Action::ChooseRiot(haste)
            }
            Decision::CastSpellOrPass { .. } if !state.stack.is_empty() => Action::Pass,
            Decision::CastSpellOrPass { .. } => return spell,
            other => panic!("unhandled decision {other:?}"),
        };
        engine::step(state, a).unwrap();
    }
}

#[test]
fn spider_punk_has_riot_and_gives_it_to_other_spiders() {
    use mtg_kernel::card_def::Keywords;
    let mut state = game();
    lands(&mut state, P0, 6);
    let punk = cast_with_riot(&mut state, "Spider-Punk", true);
    assert!(engine::has_effective_keyword(&state, punk, Keywords::HASTE));
    assert_eq!(state.objects.get(punk).counters.plus1_plus1, 0);
    let spidey = cast_with_riot(&mut state, "Spider-Man, Web-Spinner", false);
    assert_eq!(state.objects.get(spidey).counters.plus1_plus1, 1);
    end_turn(&mut state);
    assert!(engine::has_effective_keyword(&state, punk, Keywords::HASTE), "riot haste lasts");
}

#[test]
fn with_spider_punk_out_counterspell_counters_nothing() {
    let mut state = game();
    put(&mut state, P0, "Spider-Punk");
    lands(&mut state, P0, 1);
    put(&mut state, P1, "Island");
    put(&mut state, P1, "Island");
    let counter = put_in(&mut state, P1, "Counterspell", Zone::Hand);
    let bolt = put_in(&mut state, P0, "Lightning Bolt", Zone::Hand);
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    engine::step(&mut state, Action::CastSpell(bolt)).unwrap();
    let mut countered = false;
    loop {
        let a = match engine::advance_until_decision(&mut state) {
            Decision::ChooseTargets { legal_targets, .. } => {
                if legal_targets.contains(&Target::Player(P1)) {
                    Action::ChooseTarget(Target::Player(P1))
                } else {
                    Action::ChooseTarget(legal_targets[0])
                }
            }
            Decision::CastSpellOrPass { player: P1, castable_spells, .. } if !countered && castable_spells.contains(&counter) => {
                countered = true;
                Action::CastSpell(counter)
            }
            Decision::CastSpellOrPass { .. } if !state.stack.is_empty() => Action::Pass,
            Decision::CastSpellOrPass { .. } => break,
            other => panic!("unhandled decision {other:?}"),
        };
        engine::step(&mut state, a).unwrap();
    }
    assert!(countered);
    assert_eq!(state.players[1].life, 37, "the Bolt resolved");
}

#[test]
fn vision_of_love_draws_two_after_sacrificing_an_artifact_or_discarding() {
    use mtg_kernel::engine::OptionalCostChoice;
    for (choice, hand_after, treasure_left) in [
        (OptionalCostChoice::SacrificeLand, 3, false),
        (OptionalCostChoice::Discard, 2, true),
        (OptionalCostChoice::Decline, 1, true),
    ] {
        let mut state = game();
        lands(&mut state, P0, 2);
        put(&mut state, P0, "Treasure Token");
        put(&mut state, P0, "Sol Ring");
        put_in(&mut state, P0, "Mountain", Zone::Hand);
        let vision = put_in(&mut state, P0, "Vision of Love", Zone::Hand);
        advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
        engine::step(&mut state, Action::CastSpell(vision)).unwrap();
        loop {
            let a = match engine::advance_until_decision(&mut state) {
                Decision::ChooseOptionalCost { .. } => Action::ChooseOptionalCost(choice),
                Decision::ChooseCostTargets { candidates, .. } => {
                    assert_eq!(candidates.len(), 2, "two artifacts to choose from");
                    let treasure = *candidates.iter().find(|&&c| state.objects.get(c).name == "Treasure Token").unwrap();
                    Action::ChooseCostTarget(treasure)
                }
                Decision::Discard { choices, .. } => Action::Discard(vec![choices[0]]),
                Decision::CastSpellOrPass { .. } if !state.stack.is_empty() => Action::Pass,
                Decision::CastSpellOrPass { .. } => break,
                other => panic!("unhandled decision {other:?}"),
            };
            engine::step(&mut state, a).unwrap();
        }
        assert_eq!(state.players[0].hand.len(), hand_after, "{choice:?}");
        assert_eq!(!named(&state, P0, "Treasure Token").is_empty(), treasure_left, "{choice:?}");
    }
}

#[test]
fn gornog_makes_a_coward_that_cant_block_warriors_and_pumps_attacking_warriors() {
    use mtg_kernel::card_def::Subtype;
    let mut state = game();
    let gornog = put(&mut state, P0, "Gornog, the Red Reaper");
    let theirs = put(&mut state, P1, "Guttersnipe");
    let base = effective_power(&state, gornog);
    let blocks = attack_and_see_blocks(&mut state, vec![gornog], Target::Object(theirs));
    assert!(engine::has_effective_subtype(&state, theirs, Subtype::Coward));
    assert!(!engine::has_effective_subtype(&state, theirs, Subtype::Goblin), "its other creature types are gone");
    assert!(blocks.iter().all(|b| !b.1.contains(&theirs)), "Cowards can't block Warriors");
    assert_eq!(effective_power(&state, gornog), base + 1, "one Coward");
    end_turn(&mut state);
    assert!(engine::has_effective_subtype(&state, theirs, Subtype::Coward), "no duration");
}

#[test]
fn goddric_is_a_four_four_flying_dragon_while_celebrating() {
    use mtg_kernel::card_def::{Keywords, Subtype};
    use mtg_kernel::event::{propose_and_commit, ProposedEvent};
    let mut state = game();
    lands(&mut state, P0, 1);
    let goddric = put(&mut state, P0, "Goddric, Cloaked Reveler");
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    assert!(engine::has_effective_subtype(&state, goddric, Subtype::Human));
    assert!(!engine::has_effective_keyword(&state, goddric, Keywords::FLYING));
    assert!(!offered_ability(&mut state, goddric, 0), "the ability exists only while celebrating");
    propose_and_commit(&mut state, ProposedEvent::create_token(id("Treasure Token"), P0));
    propose_and_commit(&mut state, ProposedEvent::create_token(id("Treasure Token"), P0));
    assert_eq!((effective_power(&state, goddric), engine::effective_toughness(&state, goddric)), (4, 4));
    assert!(engine::has_effective_keyword(&state, goddric, Keywords::FLYING));
    assert!(engine::has_effective_subtype(&state, goddric, Subtype::Dragon));
    assert!(!engine::has_effective_subtype(&state, goddric, Subtype::Human), "he loses his other creature types");
    activate(&mut state, goddric, 0, None);
    assert_eq!(effective_power(&state, goddric), 5, "{{R}}: Dragons you control get +1/+0");
}

#[test]
fn liquimetal_torque_makes_a_permanent_an_artifact_until_end_of_turn() {
    use mtg_kernel::card_def::CardType;
    let mut state = game();
    let torque = put(&mut state, P0, "Liquimetal Torque");
    let theirs = put(&mut state, P1, "Guttersnipe");
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    assert_eq!(engine::available_mana_ability_choices(P0, torque, &state), vec![ManaColor::C]);
    activate(&mut state, torque, 0, Some(Target::Object(theirs)));
    assert!(engine::object_has_type(&state, theirs, CardType::Artifact));
    assert!(engine::object_has_type(&state, theirs, CardType::Creature), "in addition to its other types");
    end_turn(&mut state);
    assert!(!engine::object_has_type(&state, theirs, CardType::Artifact));
}

#[test]
fn depthshaker_titan_animates_artifacts_that_attack_with_melee_then_are_sacrificed() {
    use mtg_kernel::card_def::{CardType, Keywords};
    let mut state = game();
    lands(&mut state, P0, 7);
    let ring = put(&mut state, P0, "Sol Ring");
    let treasure = put(&mut state, P0, "Treasure Token");
    let titan = put_in(&mut state, P0, "Depthshaker Titan", Zone::Hand);
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    engine::step(&mut state, Action::CastSpell(titan)).unwrap();
    let mut chosen = false;
    loop {
        let a = match engine::advance_until_decision(&mut state) {
            Decision::ChooseTargets { legal_targets, can_finish, .. } => {
                assert!(can_finish, "any number, zero included");
                if chosen {
                    Action::FinishEffectSelection
                } else {
                    assert!(legal_targets.contains(&Target::Object(ring)) && legal_targets.contains(&Target::Object(treasure)));
                    chosen = true;
                    Action::ChooseTarget(Target::Object(ring))
                }
            }
            Decision::CastSpellOrPass { .. } if !state.stack.is_empty() => Action::Pass,
            Decision::CastSpellOrPass { .. } => break,
            other => panic!("unhandled decision {other:?}"),
        };
        engine::step(&mut state, a).unwrap();
    }
    assert!(engine::object_has_type(&state, ring, CardType::Creature) && engine::object_has_type(&state, ring, CardType::Artifact));
    assert!(!engine::object_has_type(&state, treasure, CardType::Creature), "only the targets");
    assert_eq!((effective_power(&state, ring), engine::effective_toughness(&state, ring)), (3, 3));
    for kw in [Keywords::HASTE, Keywords::TRAMPLE] {
        assert!(engine::has_effective_keyword(&state, ring, kw));
        assert!(engine::has_effective_keyword(&state, titan, kw), "the Titan is an artifact creature too");
    }
    attack_with(&mut state, vec![ring]);
    assert_eq!(effective_power(&state, ring), 4, "melee");
    end_turn(&mut state);
    assert_eq!(state.objects.get(ring).zone, Zone::Graveyard, "sacrificed at the end step");
    assert_eq!(state.objects.get(treasure).zone, Zone::Battlefield);
    assert_eq!(state.objects.get(titan).zone, Zone::Battlefield);
}

#[test]
fn call_for_aid_takes_their_creatures_but_you_cant_attack_them_or_sacrifice_those() {
    let mut state = game();
    lands(&mut state, P0, 5);
    let guttersnipe = put(&mut state, P1, "Guttersnipe");
    let spitter = put(&mut state, P1, "Fume Spitter");
    cast(&mut state, "Call for Aid", Some(Target::Player(P1)), 0);
    assert_eq!((state.objects.get(guttersnipe).controller, state.objects.get(spitter).controller), (P0, P0));
    assert!(!offered_ability(&mut state, spitter, 0), "you can't sacrifice those creatures this turn");
    let d = advance(&mut state, |s, d| matches!(d, Decision::DeclareAttackers { .. }) || s.step == Step::Main2);
    if let Decision::DeclareAttackers { eligible, .. } = d {
        assert!(eligible.is_empty(), "you can't attack that player this turn");
    }
}

#[test]
fn hellkite_charger_pays_for_another_combat_with_its_attackers_untapped() {
    let mut state = game();
    lands(&mut state, P0, 7);
    let hellkite = put(&mut state, P0, "Hellkite Charger");
    let mut combats = 0;
    let mut paid = false;
    for _ in 0..200 {
        let a = match engine::advance_until_decision(&mut state) {
            Decision::DeclareAttackers { eligible, .. } if state.active_player == P0 => {
                combats += 1;
                assert!(eligible.contains(&hellkite), "combat {combats}");
                Action::DeclareAttackers(vec![hellkite])
            }
            Decision::ChooseEffectBoolean { .. } => {
                let pay = !paid;
                paid = true;
                Action::ChooseEffectBoolean(pay)
            }
            Decision::CastSpellOrPass { .. } if state.step == Step::Main2 => break,
            Decision::CastSpellOrPass { .. } => Action::Pass,
            Decision::DeclareBlockers { .. } => Action::DeclareBlockers(vec![]),
            Decision::OrderTriggers { pending, .. } => Action::OrderTriggers((0..pending.len()).collect()),
            other => panic!("unhandled decision {other:?}"),
        };
        engine::step(&mut state, a).unwrap();
    }
    assert_eq!(combats, 2);
    assert_eq!(state.players[1].life, 30, "5 in each combat");
}

#[test]
fn pyreswipe_hawk_grows_by_artifact_mana_value_and_takes_an_artifact_on_expend_six() {
    use mtg_kernel::event::{propose_and_commit, ProposedEvent};
    let mut state = game();
    let hawk = put(&mut state, P0, "Pyreswipe Hawk");
    put(&mut state, P0, "Sol Ring");
    put(&mut state, P0, "Liquimetal Torque");
    let base = effective_power(&state, hawk);
    attack_with(&mut state, vec![hawk]);
    assert_eq!(effective_power(&state, hawk), base + 2, "Liquimetal Torque has mana value 2");

    let mut state = game();
    let hawk = put(&mut state, P0, "Pyreswipe Hawk");
    let theirs = put(&mut state, P1, "Sol Ring");
    lands(&mut state, P0, 6);
    cast(&mut state, "Guttersnipe", None, 0);
    assert_eq!(state.objects.get(theirs).controller, P1, "three mana spent");
    cast(&mut state, "Guttersnipe", Some(Target::Object(theirs)), 0);
    assert_eq!(state.objects.get(theirs).controller, P0, "the sixth mana");
    propose_and_commit(&mut state, ProposedEvent::zone_change(hawk, Zone::Graveyard));
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    assert_eq!(state.objects.get(theirs).controller, P1, "only as long as you control the Hawk");
}
