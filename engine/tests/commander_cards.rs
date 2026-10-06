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

/// Activates `source`'s ability `index` and answers its decisions: targets
/// from `targets` in order (none left finishes an "up to"), artifacts to tap
/// from `tap` (then done), and effect options from `options` in order.
fn activate_with(state: &mut GameState, source: ObjectId, index: u8, targets: &[Target], tap: &[ObjectId], options: &[u16]) {
    advance(state, |_, d| {
        matches!(d, Decision::CastSpellOrPass { activatable_abilities, .. } if activatable_abilities.contains(&(source, index)))
    });
    engine::step(state, Action::ActivateAbility(source, index)).unwrap();
    let (mut targets, mut tap, mut options) = (targets.iter(), tap.iter(), options.iter());
    loop {
        let a = match engine::advance_until_decision(state) {
            Decision::ChooseTargets { legal_targets, .. } => {
                let t = *targets.next().expect("a target is required");
                assert!(legal_targets.contains(&t), "{t:?} not in {legal_targets:?}");
                Action::ChooseTarget(t)
            }
            Decision::ChooseEffectTargets { legal_targets, .. } => match targets.next() {
                Some(&t) => {
                    assert!(legal_targets.contains(&t), "{t:?} not in {legal_targets:?}");
                    Action::ChooseEffectTarget(t)
                }
                None => Action::FinishEffectSelection,
            },
            Decision::ChooseCostTargets { candidates, can_finish, .. } => match tap.next() {
                Some(&o) => {
                    assert!(candidates.contains(&o), "{o:?} not in {candidates:?}");
                    Action::ChooseCostTarget(o)
                }
                None => {
                    assert!(can_finish);
                    Action::FinishEffectSelection
                }
            },
            Decision::ChooseEffectOption { .. } => Action::ChooseEffectOption(*options.next().expect("an option")),
            Decision::CastSpellOrPass { .. } if !state.stack.is_empty() => Action::Pass,
            Decision::CastSpellOrPass { .. } => return,
            other => panic!("unhandled decision {other:?}"),
        };
        engine::step(state, a).unwrap();
    }
}

fn zone(state: &GameState, o: ObjectId) -> Zone {
    state.objects.get(o).zone
}

#[test]
fn abstergo_makes_colorless_or_any_color_for_one() {
    let mut state = game();
    let abstergo = put(&mut state, P0, "Abstergo Entertainment");
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    assert_eq!(engine::available_mana_ability_choices(P0, abstergo, &state), vec![ManaColor::C]);
    lands(&mut state, P0, 1);
    let choices = engine::available_mana_ability_choices(P0, abstergo, &state);
    assert!(choices.contains(&ManaColor::C) && choices.contains(&ManaColor::G), "{choices:?}");
}

#[test]
fn abstergo_returns_a_historic_card_then_exiles_all_graveyards() {
    let mut state = game();
    let abstergo = put(&mut state, P0, "Abstergo Entertainment");
    lands(&mut state, P0, 3);
    let signet = put_in(&mut state, P0, "Arcane Signet", Zone::Graveyard);
    let bolt = put_in(&mut state, P0, "Lightning Bolt", Zone::Graveyard);
    let theirs = put_in(&mut state, P1, "Lightning Bolt", Zone::Graveyard);
    activate_with(&mut state, abstergo, 0, &[Target::Object(signet)], &[], &[]);
    assert_eq!(zone(&state, signet), Zone::Hand);
    assert_eq!((zone(&state, bolt), zone(&state, theirs), zone(&state, abstergo)), (Zone::Exile, Zone::Exile, Zone::Exile));
}

#[test]
fn abstergo_targets_only_historic_cards_and_may_target_none() {
    let mut state = game();
    let abstergo = put(&mut state, P0, "Abstergo Entertainment");
    lands(&mut state, P0, 3);
    let bolt = put_in(&mut state, P0, "Lightning Bolt", Zone::Graveyard);
    // No historic card: the ability still exiles every graveyard.
    activate_with(&mut state, abstergo, 0, &[], &[], &[]);
    assert_eq!(zone(&state, bolt), Zone::Exile);
}

#[test]
fn abstergo_does_nothing_once_its_target_is_gone() {
    use mtg_kernel::event::{propose_and_commit, ProposedEvent};
    let mut state = game();
    let abstergo = put(&mut state, P0, "Abstergo Entertainment");
    lands(&mut state, P0, 3);
    let signet = put_in(&mut state, P0, "Arcane Signet", Zone::Graveyard);
    let bolt = put_in(&mut state, P0, "Lightning Bolt", Zone::Graveyard);
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    engine::step(&mut state, Action::ActivateAbility(abstergo, 0)).unwrap();
    engine::advance_until_decision(&mut state);
    engine::step(&mut state, Action::ChooseEffectTarget(Target::Object(signet))).unwrap();
    advance(&mut state, |s, _| !s.stack.is_empty());
    propose_and_commit(&mut state, ProposedEvent::zone_change(signet, Zone::Exile));
    resolve(&mut state);
    assert_eq!(zone(&state, bolt), Zone::Graveyard, "no graveyard is exiled");
    assert_eq!(zone(&state, abstergo), Zone::Exile, "exiling it was a cost");
}

#[test]
fn demolition_field_destroys_a_nonbasic_land_and_each_player_may_search() {
    let mut state = game();
    let field = put(&mut state, P0, "Demolition Field");
    lands(&mut state, P0, 2);
    let mine = put(&mut state, P0, "Strip Mine");
    let basic = put(&mut state, P1, "Mountain");
    let theirs = put(&mut state, P1, "Strip Mine");
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    engine::step(&mut state, Action::ActivateAbility(field, 0)).unwrap();
    match engine::advance_until_decision(&mut state) {
        Decision::ChooseTargets { legal_targets, .. } => {
            assert_eq!(legal_targets, vec![Target::Object(theirs)], "not {basic:?} or {mine:?}")
        }
        other => panic!("{other:?}"),
    }
    engine::step(&mut state, Action::ChooseTarget(Target::Object(theirs))).unwrap();
    let libraries = [state.players[0].library.len(), state.players[1].library.len()];
    let mut asked = Vec::new();
    loop {
        let a = match engine::advance_until_decision(&mut state) {
            Decision::ChooseEffectOption { player, option_count, .. } => {
                assert_eq!(option_count, 2);
                asked.push(player);
                Action::ChooseEffectOption(1)
            }
            Decision::ChooseEffectTargets { legal_targets, .. } => Action::ChooseEffectTarget(legal_targets[0]),
            Decision::CastSpellOrPass { .. } if !state.stack.is_empty() => Action::Pass,
            Decision::CastSpellOrPass { .. } => break,
            other => panic!("unhandled decision {other:?}"),
        };
        engine::step(&mut state, a).unwrap();
    }
    assert_eq!(asked, vec![P1, P0], "the land's controller first, then you");
    assert_eq!((zone(&state, theirs), zone(&state, field)), (Zone::Graveyard, Zone::Graveyard));
    for p in [P0, P1] {
        assert_eq!(state.players[p.index()].library.len(), libraries[p.index()] - 1);
        let searched = state.players[p.index()].battlefield.last().copied().unwrap();
        assert_eq!(CARD_DEFS[state.objects.get(searched).card_def as usize].name, "Mountain");
        assert!(!state.objects.get(searched).tapped, "it enters untapped");
    }
}

#[test]
fn demolition_field_searches_are_optional_and_need_a_legal_target() {
    use mtg_kernel::event::{propose_and_commit, ProposedEvent};
    let mut state = game();
    let field = put(&mut state, P0, "Demolition Field");
    lands(&mut state, P0, 2);
    let theirs = put(&mut state, P1, "Strip Mine");
    activate_with(&mut state, field, 0, &[Target::Object(theirs)], &[], &[0, 0]);
    assert_eq!(zone(&state, theirs), Zone::Graveyard);
    assert_eq!((state.players[0].library.len(), state.players[1].library.len()), (10, 10), "both declined");

    let mut state = game();
    let field = put(&mut state, P0, "Demolition Field");
    lands(&mut state, P0, 2);
    let theirs = put(&mut state, P1, "Strip Mine");
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    engine::step(&mut state, Action::ActivateAbility(field, 0)).unwrap();
    engine::advance_until_decision(&mut state);
    engine::step(&mut state, Action::ChooseTarget(Target::Object(theirs))).unwrap();
    advance(&mut state, |s, _| !s.stack.is_empty());
    propose_and_commit(&mut state, ProposedEvent::zone_change(theirs, Zone::Hand));
    resolve(&mut state);
    assert_eq!(zone(&state, field), Zone::Graveyard);
    assert_eq!((state.players[0].library.len(), state.players[1].library.len()), (10, 10), "no one searches");
}

#[test]
fn secluded_starforge_taps_x_artifacts_for_plus_x() {
    let mut state = game();
    let forge = put(&mut state, P0, "Secluded Starforge");
    lands(&mut state, P0, 2);
    let creature = put(&mut state, P0, "Guttersnipe");
    let artifacts = [put(&mut state, P0, "Treasure Token"), put(&mut state, P0, "Arcane Signet"), put(&mut state, P0, "Sol Ring")];
    let base = effective_power(&state, creature);
    activate_with(&mut state, forge, 0, &[Target::Object(creature)], &artifacts[..2], &[]);
    assert_eq!(effective_power(&state, creature), base + 2);
    let tapped: Vec<bool> = artifacts.iter().map(|&a| state.objects.get(a).tapped).collect();
    assert_eq!(tapped, vec![true, true, false]);
    assert_eq!(state.players[0].battlefield.len(), 7, "nothing was sacrificed");
}

#[test]
fn secluded_starforge_keeps_the_artifacts_its_mana_needs() {
    let mut state = game();
    let forge = put(&mut state, P0, "Secluded Starforge");
    let creature = put(&mut state, P0, "Guttersnipe");
    let ring = put(&mut state, P0, "Sol Ring");
    let treasure = put(&mut state, P0, "Treasure Token");
    let base = effective_power(&state, creature);
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    engine::step(&mut state, Action::ActivateAbility(forge, 0)).unwrap();
    engine::advance_until_decision(&mut state);
    engine::step(&mut state, Action::ChooseTarget(Target::Object(creature))).unwrap();
    // Sol Ring has to pay for {2}, so only the Treasure can be tapped for X.
    match engine::advance_until_decision(&mut state) {
        Decision::ChooseCostTargets { candidates, can_finish, .. } => assert_eq!((candidates, can_finish), (vec![treasure], true)),
        other => panic!("{other:?}"),
    }
    engine::step(&mut state, Action::ChooseCostTarget(treasure)).unwrap();
    advance(&mut state, |s, _| !s.stack.is_empty());
    resolve(&mut state);
    assert_eq!(effective_power(&state, creature), base + 1);
    assert!(state.objects.get(ring).tapped && state.objects.get(treasure).tapped);
    assert_eq!(zone(&state, treasure), Zone::Battlefield);
}

#[test]
fn secluded_starforge_pumps_only_as_a_sorcery_and_makes_robots() {
    let mut state = game();
    let forge = put(&mut state, P0, "Secluded Starforge");
    lands(&mut state, P0, 7);
    put(&mut state, P0, "Guttersnipe");
    advance(&mut state, |s, _| s.step == Step::BeginCombat);
    assert!(!offered_ability(&mut state, forge, 0));
    assert!(offered_ability(&mut state, forge, 1));
    activate_with(&mut state, forge, 1, &[], &[], &[]);
    let robot = named(&state, P0, "Robot Token")[0];
    let o = state.objects.get(robot);
    assert_eq!((effective_power(&state, robot), CARD_DEFS[o.card_def as usize].types.len()), (2, 2));
    assert!(o.v4.is_token);
}

/// Makes `player`'s library card at `position` (0 = top) a `name`.
fn library_card(state: &mut GameState, player: PlayerId, position: usize, name: &str) -> ObjectId {
    let o = state.players[player.index()].library[position];
    let obj = state.objects.get_mut(o);
    obj.card_def = id(name);
    obj.name = name.to_string();
    obj.v4 = ObjectStateV4::from_card_def(id(name));
    o
}

/// Activates Fomori Vault, discarding the first card offered, and picks `pick` if asked.
fn fomori(state: &mut GameState, vault: ObjectId, pick: Option<ObjectId>) -> Option<Vec<Target>> {
    advance(state, |_, d| matches!(d, Decision::CastSpellOrPass { activatable_abilities, .. } if activatable_abilities.contains(&(vault, 0))));
    engine::step(state, Action::ActivateAbility(vault, 0)).unwrap();
    let mut offered = None;
    loop {
        let a = match engine::advance_until_decision(state) {
            Decision::Discard { choices, .. } => Action::Discard(vec![choices[0]]),
            Decision::ChooseEffectTargets { legal_targets, .. } => {
                offered = Some(legal_targets);
                Action::ChooseEffectTarget(Target::Object(pick.expect("a pick")))
            }
            Decision::CastSpellOrPass { .. } if !state.stack.is_empty() => Action::Pass,
            Decision::CastSpellOrPass { .. } => return offered,
            other => panic!("unhandled decision {other:?}"),
        };
        engine::step(state, a).unwrap();
    }
}

#[test]
fn fomori_vault_takes_one_of_the_top_x_and_bottoms_the_rest_in_a_random_order() {
    let mut state = game();
    let vault = put(&mut state, P0, "Fomori Vault");
    lands(&mut state, P0, 3);
    put(&mut state, P0, "Myr Enforcer");
    put(&mut state, P0, "Arcane Signet");
    put_in(&mut state, P0, "Lightning Bolt", Zone::Hand);
    let snipe = library_card(&mut state, P0, 0, "Guttersnipe");
    let bolt = library_card(&mut state, P0, 1, "Lightning Bolt");
    let offered = fomori(&mut state, vault, Some(snipe));
    assert_eq!(offered, Some(vec![Target::Object(snipe), Target::Object(bolt)]), "two artifacts: the top two");
    assert_eq!(zone(&state, snipe), Zone::Hand);
    let lib = &state.players[0].library;
    assert_eq!((lib.len(), *lib.last().unwrap()), (9, bolt));
    // Its owner knows the card is down there, but not where; the opponent knows nothing.
    assert!(state.known_library_cards(P0, P0).iter().all(|e| e.object != bolt));
    let groups: Vec<_> = state.known_library_groups(P0, P0).map(|g| g.cards.iter().map(|l| l.object).collect::<Vec<_>>()).collect();
    assert_eq!(groups, vec![vec![bolt]]);
    assert_eq!(state.known_library_groups(P1, P0).count(), 0);
    assert!(!state.known_hand_cards(P1, P0).iter().any(|e| e.object == snipe));
}

#[test]
fn fomori_vault_without_artifacts_looks_at_nothing() {
    let mut state = game();
    let vault = put(&mut state, P0, "Fomori Vault");
    lands(&mut state, P0, 3);
    let discard = put_in(&mut state, P0, "Lightning Bolt", Zone::Hand);
    assert_eq!(fomori(&mut state, vault, None), None);
    assert_eq!(zone(&state, discard), Zone::Graveyard);
    assert_eq!((state.players[0].library.len(), state.players[0].hand.len()), (10, 0));
}

#[test]
fn chaos_warp_shuffles_the_permanent_in_and_reveals_the_top_card() {
    let mut state = game();
    lands(&mut state, P0, 3);
    let snipe = put(&mut state, P1, "Guttersnipe");
    for i in 0..10 {
        library_card(&mut state, P1, i, "Lightning Bolt");
    }
    cast(&mut state, "Chaos Warp", Some(Target::Object(snipe)), 0);
    let lib = state.players[1].library.clone();
    if zone(&state, snipe) == Zone::Battlefield {
        // The Guttersnipe itself was revealed on top: a permanent card.
        assert_eq!(lib.len(), 10);
    } else {
        assert_eq!((lib.len(), zone(&state, snipe)), (11, Zone::Library));
        assert_ne!(lib[0], snipe);
        // A revealed Bolt stays on top, and both players know it.
        for p in [P0, P1] {
            assert!(state.known_library_cards(p, P1).iter().any(|e| e.position == 0 && e.object == lib[0]));
        }
    }
}

#[test]
fn chaos_warp_puts_a_revealed_permanent_card_onto_the_battlefield() {
    let mut state = game();
    lands(&mut state, P0, 3);
    let snipe = put(&mut state, P1, "Guttersnipe");
    cast(&mut state, "Chaos Warp", Some(Target::Object(snipe)), 0);
    // Every other card is a Mountain, so something always enters.
    assert_eq!(state.players[1].battlefield.len(), 1);
    assert_eq!(state.players[1].library.len(), 10);
}

#[test]
fn chaos_warp_lets_a_commanders_owner_choose_the_command_zone() {
    use mtg_kernel::event::{propose_and_commit, ProposedEvent};
    for (choice, expected) in [(1u8, Zone::Command), (0u8, Zone::Library)] {
        let mut state = game();
        lands(&mut state, P0, 3);
        let ognis = state.commander.as_ref().unwrap().commanders[1];
        propose_and_commit(&mut state, ProposedEvent::zone_change(ognis, Zone::Battlefield));
        for i in 0..10 {
            library_card(&mut state, P1, i, "Lightning Bolt");
        }
        // `cast` answers the owner's choice with `choice` and would fail on a second commander prompt.
        cast(&mut state, "Chaos Warp", Some(Target::Object(ognis)), choice);
        match (expected, zone(&state, ognis)) {
            (Zone::Command, Zone::Command) => assert_eq!(state.players[1].library.len(), 10, "shuffled anyway"),
            (Zone::Library, Zone::Library) => assert_eq!(state.players[1].library.len(), 11),
            // Shuffled in, then revealed on top: a permanent card.
            (Zone::Library, Zone::Battlefield) => assert_eq!(state.players[1].library.len(), 10),
            other => panic!("{other:?}"),
        }
    }
}

#[test]
fn wild_magic_surge_finds_a_permanent_sharing_a_type_and_bottoms_the_rest() {
    let mut state = game();
    lands(&mut state, P0, 2);
    let enforcer = put(&mut state, P1, "Myr Enforcer");
    let bolts = [library_card(&mut state, P1, 0, "Lightning Bolt"), library_card(&mut state, P1, 1, "Lightning Bolt")];
    let snipe = library_card(&mut state, P1, 2, "Guttersnipe");
    cast(&mut state, "Wild Magic Surge", Some(Target::Object(enforcer)), 0);
    assert_eq!(zone(&state, enforcer), Zone::Graveyard);
    assert_eq!((zone(&state, snipe), state.objects.get(snipe).controller), (Zone::Battlefield, P1), "a creature, like the Enforcer");
    let lib = &state.players[1].library;
    assert_eq!(lib.len(), 9);
    let mut bottom = lib[7..].to_vec();
    bottom.sort();
    assert_eq!(bottom, bolts.to_vec());
    for p in [P0, P1] {
        assert_eq!(state.known_library_groups(p, P1).count(), 1, "revealed, so both know");
    }
}

#[test]
fn wild_magic_surge_on_a_land_finds_a_land() {
    let mut state = game();
    lands(&mut state, P0, 2);
    let mine = put(&mut state, P1, "Strip Mine");
    cast(&mut state, "Wild Magic Surge", Some(Target::Object(mine)), 0);
    assert_eq!(zone(&state, mine), Zone::Graveyard);
    assert_eq!(named(&state, P1, "Mountain").len(), 1);
    assert_eq!(state.known_library_groups(P0, P1).count(), 0, "nothing else was revealed");
}

/// Casts Call Forth the Tempest, answering each cascade offer from `cast_hits`
/// (targets: player 1), and resolves everything.
fn tempest(state: &mut GameState, cast_hits: &[bool]) -> Vec<ObjectId> {
    let spell = put_in(state, P0, "Call Forth the Tempest", Zone::Hand);
    advance(state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    engine::step(state, Action::CastSpell(spell)).unwrap();
    let (mut answers, mut offered) = (cast_hits.iter(), Vec::new());
    loop {
        let a = match engine::advance_until_decision(state) {
            Decision::ChooseCascadeCast { player, card } => {
                assert_eq!(player, P0);
                offered.push(card);
                Action::ChooseCascadeCast(*answers.next().expect("an answer"))
            }
            Decision::ChooseTargets { legal_targets, .. } => {
                Action::ChooseTarget(if legal_targets.contains(&Target::Player(P1)) { Target::Player(P1) } else { legal_targets[0] })
            }
            Decision::OrderTriggers { pending, .. } => Action::OrderTriggers((0..pending.len()).collect()),
            Decision::CastSpellOrPass { .. } if !state.stack.is_empty() => Action::Pass,
            Decision::CastSpellOrPass { .. } => return offered,
            other => panic!("unhandled decision {other:?}"),
        };
        engine::step(state, a).unwrap();
    }
}

#[test]
fn call_forth_the_tempest_cascades_twice_and_deals_the_other_spells_mana_value() {
    let mut state = game();
    lands(&mut state, P0, 8);
    let enforcer = put(&mut state, P1, "Myr Enforcer");
    let bolt = library_card(&mut state, P0, 1, "Lightning Bolt");
    let snipe = library_card(&mut state, P0, 3, "Guttersnipe");
    let offered = tempest(&mut state, &[true, true]);
    assert_eq!(offered, vec![bolt, snipe]);
    assert_eq!(state.players[1].life, 37, "the cascaded Bolt");
    assert_eq!(zone(&state, snipe), Zone::Battlefield);
    // Bolt (1) and Guttersnipe (3): 4 damage kills the 4/4.
    assert_eq!(zone(&state, enforcer), Zone::Graveyard);
    // The two Mountains exiled on the way went to the bottom, known to both players.
    assert_eq!(state.players[0].library.len(), 8);
    for p in [P0, P1] {
        assert_eq!(state.known_library_groups(p, P0).count(), 2);
    }
}

#[test]
fn a_declined_cascade_hit_goes_to_the_bottom_too() {
    let mut state = game();
    lands(&mut state, P0, 8);
    let bolt = library_card(&mut state, P0, 0, "Lightning Bolt");
    // Declined, the Bolt goes to the bottom, where the second cascade finds it again.
    assert_eq!(tempest(&mut state, &[false, false]), vec![bolt, bolt]);
    assert_eq!(state.players[1].life, 40);
    assert_eq!((zone(&state, bolt), state.players[0].library.len()), (Zone::Library, 10));
    let groups: Vec<usize> = state.known_library_groups(P1, P0).map(|g| g.cards.len()).collect();
    assert_eq!(groups, vec![10], "the whole library, known to both, order unknown");
}

#[test]
fn a_cascaded_x_spell_has_x_zero() {
    let mut state = game();
    lands(&mut state, P0, 8);
    let devastator = library_card(&mut state, P0, 0, "Shivan Devastator");
    tempest(&mut state, &[true, false]);
    // X = 0: a 0/0 that dies.
    assert_eq!(zone(&state, devastator), Zone::Graveyard);
}

#[test]
fn triggers_from_a_cascade_wait_while_its_hit_is_cast() {
    let mut state = game();
    lands(&mut state, P0, 8);
    let laelia = put(&mut state, P0, "Laelia, the Blade Reforged");
    let snipe = library_card(&mut state, P0, 0, "Guttersnipe");
    let aid = library_card(&mut state, P0, 2, "Call for Aid");
    assert_eq!(tempest(&mut state, &[true, true]), vec![snipe, aid]);
    assert_eq!(zone(&state, snipe), Zone::Battlefield);
    assert_eq!(zone(&state, aid), Zone::Graveyard, "cast, with a target");
    // Each cascade exiled cards from the library: two Laelia triggers.
    assert_eq!(state.objects.get(laelia).counters.plus1_plus1, 2);
}

#[test]
fn hideous_taskmaster_steals_a_creature_that_then_annihilates() {
    use mtg_kernel::card_def::Keywords;
    let mut state = game();
    lands(&mut state, P0, 7);
    let snipe = put(&mut state, P1, "Guttersnipe");
    state.objects.get_mut(snipe).tapped = true;
    let theirs: Vec<ObjectId> = (0..3).map(|_| put(&mut state, P1, "Mountain")).collect();
    let taskmaster = cast(&mut state, "Hideous Taskmaster", Some(Target::Object(snipe)), 0);
    let o = state.objects.get(snipe);
    assert_eq!((o.controller, o.tapped), (P0, false));
    for k in [Keywords::HASTE, Keywords::TRAMPLE, Keywords::ANNIHILATOR] {
        assert!(engine::has_effective_keyword(&state, snipe, k));
        assert!(engine::has_effective_keyword(&state, taskmaster, k), "printed");
    }
    // Both attack: two annihilator triggers, so player 1 sacrifices two permanents.
    advance(&mut state, |_, d| matches!(d, Decision::DeclareAttackers { .. }));
    engine::step(&mut state, Action::DeclareAttackers(vec![taskmaster, snipe])).unwrap();
    let mut sacrifices = 0;
    loop {
        let a = match engine::advance_until_decision(&mut state) {
            Decision::ChooseEffectTargets { player, legal_targets, .. } => {
                assert_eq!(player, P1);
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
    assert_eq!(theirs.iter().filter(|&&m| zone(&state, m) == Zone::Graveyard).count(), 2);
    end_turn(&mut state);
    assert_eq!(state.objects.get(snipe).controller, P1);
    assert!(!engine::has_effective_keyword(&state, snipe, Keywords::ANNIHILATOR));
}

#[test]
fn hideous_taskmaster_needs_no_target() {
    let mut state = game();
    lands(&mut state, P0, 7);
    let taskmaster = cast(&mut state, "Hideous Taskmaster", None, 0);
    assert_eq!(zone(&state, taskmaster), Zone::Battlefield);
    let def = &CARD_DEFS[state.objects.get(taskmaster).card_def as usize];
    assert!(def.colors.is_empty(), "devoid");
}

/// Casts Demand Answers, answering the additional-cost choice with `option`
/// if asked; returns whether it was asked.
fn demand_answers(state: &mut GameState, option: u16) -> bool {
    let spell = put_in(state, P0, "Demand Answers", Zone::Hand);
    advance(state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    engine::step(state, Action::CastSpell(spell)).unwrap();
    let mut asked = false;
    loop {
        let a = match engine::advance_until_decision(state) {
            Decision::ChooseEffectOption { option_count, .. } => {
                assert_eq!(option_count, 2);
                asked = true;
                Action::ChooseEffectOption(option)
            }
            Decision::ChooseCostTargets { candidates, .. } => Action::ChooseCostTarget(candidates[0]),
            Decision::Discard { choices, .. } => Action::Discard(vec![choices[0]]),
            Decision::CastSpellOrPass { .. } if !state.stack.is_empty() => Action::Pass,
            Decision::CastSpellOrPass { .. } => return asked,
            other => panic!("unhandled decision {other:?}"),
        };
        engine::step(state, a).unwrap();
    }
}

#[test]
fn demand_answers_sacrifices_an_artifact_or_discards_then_draws_two() {
    for (option, artifact_left, card_left) in [(0u16, false, true), (1u16, true, false)] {
        let mut state = game();
        lands(&mut state, P0, 2);
        let signet = put(&mut state, P0, "Arcane Signet");
        let card = put_in(&mut state, P0, "Lightning Bolt", Zone::Hand);
        assert!(demand_answers(&mut state, option));
        assert_eq!(zone(&state, signet) == Zone::Battlefield, artifact_left, "option {option}");
        assert_eq!(zone(&state, card) == Zone::Hand, card_left, "option {option}");
        assert_eq!(state.players[0].hand.len(), 2 + usize::from(card_left));
    }
}

#[test]
fn demand_answers_pays_the_only_cost_it_can() {
    let mut state = game();
    lands(&mut state, P0, 2);
    let card = put_in(&mut state, P0, "Lightning Bolt", Zone::Hand);
    assert!(!demand_answers(&mut state, 0), "no artifact: discard without asking");
    assert_eq!((zone(&state, card), state.players[0].hand.len()), (Zone::Graveyard, 2));

    let mut state = game();
    lands(&mut state, P0, 2);
    let spell = put_in(&mut state, P0, "Demand Answers", Zone::Hand);
    match advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. })) {
        Decision::CastSpellOrPass { castable_spells, .. } => assert!(!castable_spells.contains(&spell), "nothing to pay with"),
        _ => unreachable!(),
    }
}

#[test]
fn seize_the_spotlight_fame_takes_a_creature_the_caster_chooses() {
    let mut state = game();
    lands(&mut state, P0, 3);
    let _first = put(&mut state, P1, "Guttersnipe");
    let second = put(&mut state, P1, "Myr Enforcer");
    state.objects.get_mut(second).tapped = true;
    // Option 0 is fame; the caster then picks the Enforcer (no target).
    cast(&mut state, "Seize the Spotlight", Some(Target::Object(second)), 0);
    let o = state.objects.get(second);
    assert_eq!((o.controller, o.tapped), (P0, false));
    assert!(engine::has_effective_keyword(&state, second, mtg_kernel::card_def::Keywords::HASTE));
    end_turn(&mut state);
    assert_eq!(state.objects.get(second).controller, P1);
}

#[test]
fn seize_the_spotlight_fortune_draws_and_makes_a_treasure() {
    let mut state = game();
    lands(&mut state, P0, 3);
    put(&mut state, P1, "Guttersnipe");
    cast(&mut state, "Seize the Spotlight", None, 1);
    assert_eq!(state.players[0].hand.len(), 1);
    assert_eq!(named(&state, P0, "Treasure Token").len(), 1);

    // Fame with nothing to take does nothing.
    let mut state = game();
    lands(&mut state, P0, 3);
    cast(&mut state, "Seize the Spotlight", None, 0);
    assert_eq!((state.players[0].hand.len(), named(&state, P0, "Treasure Token").len()), (0, 0));
}

#[test]
fn depthshaker_titans_delayed_sacrifice_survives_a_change_of_control() {
    let mut state = game();
    lands(&mut state, P0, 7);
    let ring = put(&mut state, P0, "Sol Ring");
    let titan = put_in(&mut state, P0, "Depthshaker Titan", Zone::Hand);
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    engine::step(&mut state, Action::CastSpell(titan)).unwrap();
    let mut chosen = false;
    loop {
        let a = match engine::advance_until_decision(&mut state) {
            Decision::ChooseTargets { .. } if chosen => Action::FinishEffectSelection,
            Decision::ChooseTargets { .. } => {
                chosen = true;
                Action::ChooseTarget(Target::Object(ring))
            }
            Decision::CastSpellOrPass { .. } if !state.stack.is_empty() => Action::Pass,
            Decision::CastSpellOrPass { .. } => break,
            other => panic!("unhandled decision {other:?}"),
        };
        engine::step(&mut state, a).unwrap();
    }
    // Another player takes the Titan; the delayed trigger stays player 0's (603.7d).
    state.objects.get_mut(titan).controller = P1;
    advance(&mut state, |s, _| s.step == Step::Cleanup || s.active_player == P1);
    assert_eq!(zone(&state, ring), Zone::Graveyard);
}

#[test]
fn vandalblast_destroys_one_artifact_or_overloaded_each_you_dont_control() {
    let mut state = game();
    lands(&mut state, P0, 1);
    let mine = put(&mut state, P0, "Arcane Signet");
    let theirs = [put(&mut state, P1, "Sol Ring"), put(&mut state, P1, "Arcane Signet")];
    let spell = cast(&mut state, "Vandalblast", Some(Target::Object(theirs[0])), 0);
    assert_eq!((zone(&state, theirs[0]), zone(&state, theirs[1]), zone(&state, mine)), (Zone::Graveyard, Zone::Battlefield, Zone::Battlefield));
    assert_eq!(zone(&state, spell), Zone::Graveyard);

    let mut state = game();
    lands(&mut state, P0, 5);
    let mine = put(&mut state, P0, "Arcane Signet");
    let theirs = [put(&mut state, P1, "Sol Ring"), put(&mut state, P1, "Arcane Signet")];
    let spell = cast(&mut state, "Vandalblast", None, 1);
    assert!(theirs.iter().all(|&a| zone(&state, a) == Zone::Graveyard), "overloaded: each");
    assert_eq!(zone(&state, mine), Zone::Battlefield);
    assert_eq!(zone(&state, spell), Zone::Graveyard, "not shuffled away like an Omen");
}

/// Casts graveyard card `card`, answering a method choice with `method`, a
/// target with player 1, and a discard with the first offer (whose choices are
/// returned).
fn cast_from_graveyard(state: &mut GameState, card: ObjectId, method: Option<mtg_kernel::state::CastMethodV4>) -> Vec<ObjectId> {
    advance(state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    engine::step(state, Action::CastSpell(card)).unwrap();
    let mut discard_choices = Vec::new();
    loop {
        let a = match engine::advance_until_decision(state) {
            Decision::ChooseCastMethod { methods, .. } => {
                let m = method.expect("a method choice");
                assert!(methods.contains(&m), "{methods:?}");
                Action::ChooseCastMethod(m)
            }
            Decision::ChooseTargets { .. } => Action::ChooseTarget(Target::Player(P1)),
            Decision::Discard { choices, .. } => {
                discard_choices = choices.clone();
                Action::Discard(vec![choices[0]])
            }
            Decision::CastSpellOrPass { .. } if !state.stack.is_empty() => Action::Pass,
            Decision::CastSpellOrPass { .. } => return discard_choices,
            other => panic!("unhandled decision {other:?}"),
        };
        engine::step(state, a).unwrap();
    }
}

fn castable(state: &mut GameState, card: ObjectId) -> bool {
    matches!(advance(state, |_, d| matches!(d, Decision::CastSpellOrPass { .. })), Decision::CastSpellOrPass { castable_spells, .. } if castable_spells.contains(&card))
}

#[test]
fn flashback_lets_a_graveyard_spell_be_cast_once_for_its_mana_cost_this_turn() {
    let mut state = game();
    lands(&mut state, P0, 2);
    let bolt = put_in(&mut state, P0, "Lightning Bolt", Zone::Graveyard);
    assert!(!castable(&mut state, bolt));
    cast(&mut state, "Flashback", Some(Target::Object(bolt)), 0);
    assert!(castable(&mut state, bolt));
    cast_from_graveyard(&mut state, bolt, None);
    assert_eq!(state.players[1].life, 37);
    assert_eq!(zone(&state, bolt), Zone::Exile, "a flashback spell is exiled");

    // The grant lasts only this turn.
    let mut state = game();
    lands(&mut state, P0, 1);
    let bolt = put_in(&mut state, P0, "Lightning Bolt", Zone::Graveyard);
    cast(&mut state, "Flashback", Some(Target::Object(bolt)), 0);
    end_turn(&mut state);
    end_turn(&mut state);
    lands(&mut state, P0, 1);
    assert!(!castable(&mut state, bolt));
}

#[test]
fn embrace_the_unknown_retraces_by_discarding_a_land() {
    let mut state = game();
    lands(&mut state, P0, 3);
    let embrace = put_in(&mut state, P0, "Embrace the Unknown", Zone::Graveyard);
    let bolt = put_in(&mut state, P0, "Lightning Bolt", Zone::Hand);
    assert!(!castable(&mut state, embrace), "no land card to discard");
    let lands_in_hand = [put_in(&mut state, P0, "Mountain", Zone::Hand), put_in(&mut state, P0, "Mountain", Zone::Hand)];
    let offered = cast_from_graveyard(&mut state, embrace, None);
    assert_eq!(offered, lands_in_hand.to_vec(), "only land cards");
    assert_eq!(zone(&state, bolt), Zone::Hand);
    assert_eq!(zone(&state, embrace), Zone::Graveyard, "retrace returns it to the graveyard");
    assert_eq!(state.engine.exile_play_permissions.len(), 2, "two cards to play");
}

#[test]
fn a_retrace_card_with_flashback_asks_which_way_to_cast_it() {
    use mtg_kernel::state::CastMethodV4;
    let mut state = game();
    lands(&mut state, P0, 4);
    let embrace = put_in(&mut state, P0, "Embrace the Unknown", Zone::Graveyard);
    put_in(&mut state, P0, "Mountain", Zone::Hand);
    cast(&mut state, "Flashback", Some(Target::Object(embrace)), 0);
    cast_from_graveyard(&mut state, embrace, Some(CastMethodV4::Flashback));
    assert_eq!(zone(&state, embrace), Zone::Exile);
    assert_eq!(state.players[0].hand.len(), 1, "no land was discarded");
}

#[test]
fn a_flashback_vandalblast_cannot_be_overloaded() {
    let mut state = game();
    lands(&mut state, P0, 7);
    let theirs = [put(&mut state, P1, "Sol Ring"), put(&mut state, P1, "Arcane Signet")];
    let blast = put_in(&mut state, P0, "Vandalblast", Zone::Graveyard);
    cast(&mut state, "Flashback", Some(Target::Object(blast)), 0);
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    engine::step(&mut state, Action::CastSpell(blast)).unwrap();
    loop {
        let a = match engine::advance_until_decision(&mut state) {
            Decision::ChooseSpellMode { legal_modes, .. } => panic!("overload offered: {legal_modes:?}"),
            Decision::ChooseTargets { .. } => Action::ChooseTarget(Target::Object(theirs[0])),
            Decision::CastSpellOrPass { .. } if !state.stack.is_empty() => Action::Pass,
            Decision::CastSpellOrPass { .. } => break,
            other => panic!("unhandled decision {other:?}"),
        };
        engine::step(&mut state, a).unwrap();
    }
    assert_eq!((zone(&state, theirs[0]), zone(&state, theirs[1])), (Zone::Graveyard, Zone::Battlefield));
    assert_eq!(zone(&state, blast), Zone::Exile);
}

#[test]
fn shattering_spree_copies_itself_for_each_replicate_payment() {
    let mut state = game();
    lands(&mut state, P0, 3);
    let theirs = [put(&mut state, P1, "Sol Ring"), put(&mut state, P1, "Arcane Signet"), put(&mut state, P1, "Treasure Token")];
    put(&mut state, P0, "Guttersnipe");
    let spree = put_in(&mut state, P0, "Shattering Spree", Zone::Hand);
    advance(&mut state, |_, d| matches!(d, Decision::CastSpellOrPass { .. }));
    engine::step(&mut state, Action::CastSpell(spree)).unwrap();
    let mut copy_targets = theirs[1..].iter();
    loop {
        let a = match engine::advance_until_decision(&mut state) {
            Decision::ChooseTargets { .. } => Action::ChooseTarget(Target::Object(theirs[0])),
            Decision::ChooseEffectOption { option_count, .. } => {
                assert_eq!(option_count, 3, "zero, one or two payments of {{R}}");
                Action::ChooseEffectOption(2)
            }
            Decision::ChooseEffectTargets { legal_targets, .. } => {
                assert!(legal_targets.contains(&Target::Object(theirs[0])), "may keep the target");
                Action::ChooseEffectTarget(Target::Object(*copy_targets.next().unwrap()))
            }
            Decision::OrderTriggers { pending, .. } => Action::OrderTriggers((0..pending.len()).collect()),
            Decision::CastSpellOrPass { .. } if !state.stack.is_empty() => Action::Pass,
            Decision::CastSpellOrPass { .. } => break,
            other => panic!("unhandled decision {other:?}"),
        };
        engine::step(&mut state, a).unwrap();
    }
    assert!(theirs.iter().all(|&a| zone(&state, a) == Zone::Graveyard));
    assert_eq!(state.players[1].life, 38, "copies aren't cast: one Guttersnipe trigger");
    assert_eq!(zone(&state, spree), Zone::Graveyard);
}

#[test]
fn shattering_spree_without_replicate_is_one_spell() {
    let mut state = game();
    lands(&mut state, P0, 1);
    let theirs = [put(&mut state, P1, "Sol Ring"), put(&mut state, P1, "Arcane Signet")];
    cast(&mut state, "Shattering Spree", Some(Target::Object(theirs[0])), 0);
    assert_eq!((zone(&state, theirs[0]), zone(&state, theirs[1])), (Zone::Graveyard, Zone::Battlefield));
}

#[test]
fn jurin_must_be_blocked_if_able_and_pumps_attackers_per_defending_creature() {
    let mut state = game();
    let jurin = put(&mut state, P0, "Jurin, Leading the Charge");
    let other = put(&mut state, P0, "Guttersnipe");
    let blocker = put(&mut state, P1, "Myr Enforcer");
    put(&mut state, P1, "Guttersnipe");
    state.objects.get_mut(state.players[1].battlefield[1]).tapped = true;
    let (base_jurin, base_other) = (effective_power(&state, jurin), effective_power(&state, other));
    advance(&mut state, |_, d| matches!(d, Decision::DeclareAttackers { .. }));
    engine::step(&mut state, Action::DeclareAttackers(vec![jurin, other])).unwrap();
    let d = advance(&mut state, |_, d| matches!(d, Decision::DeclareBlockers { .. }));
    let Decision::DeclareBlockers { must_be_blocked, .. } = d else { unreachable!() };
    assert_eq!(must_be_blocked, vec![(jurin, 1)]);
    // Two creatures on player 1's side: +2/+0 for each attacker.
    assert_eq!((effective_power(&state, jurin), effective_power(&state, other)), (base_jurin + 2, base_other + 2));
    // The one untapped creature able to block must block Jurin.
    assert!(engine::step(&mut state.clone(), Action::DeclareBlockers(vec![])).is_err());
    assert!(engine::step(&mut state.clone(), Action::DeclareBlockers(vec![(blocker, other)])).is_err());
    engine::step(&mut state, Action::DeclareBlockers(vec![(blocker, jurin)])).unwrap();
}
