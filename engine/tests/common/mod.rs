//! Helpers shared by the Commander card tests: a two-player Ognis game driven through the public decision API.
#![allow(dead_code)]

use mtg_kernel::card_def::{card_id_by_name, CARD_DEFS};
use mtg_kernel::engine::{self, Action, Decision};
use mtg_kernel::ids::{ObjectId, PlayerId};
use mtg_kernel::state::{GameObject, GameState, ObjectStateV4, Step, Target, Zone};

pub const P0: PlayerId = PlayerId::P0;
pub const P1: PlayerId = PlayerId::P1;

pub fn id(name: &str) -> u16 {
    card_id_by_name(name).unwrap_or_else(|| panic!("{name} not in CARD_DEFS"))
}

/// A Commander game in player 0's first main phase, both libraries ten Mountains.
pub fn game() -> GameState {
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

pub fn put(state: &mut GameState, player: PlayerId, name: &str) -> ObjectId {
    put_in(state, player, name, Zone::Battlefield)
}

pub fn put_in(state: &mut GameState, player: PlayerId, name: &str, zone: Zone) -> ObjectId {
    let card_def = id(name);
    let obj = state.objects.push(GameObject {
        card_def,
        name: name.to_string().into(),
        owner: player,
        controller: player,
        zone,
        tapped: false,
        summoning_sick: false,
        damage: 0,
        counters: Default::default(),
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

pub fn lands(state: &mut GameState, player: PlayerId, n: usize) {
    for _ in 0..n {
        put(state, player, "Mountain");
    }
}

/// Casts `card` from player 0's hand with `target` for every target choice (no
/// target where "up to" allows none) and `mode` for a modal spell or as X,
/// then resolves the stack.
pub fn cast(state: &mut GameState, card: &str, target: Option<Target>, mode: u8) -> ObjectId {
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
pub fn end_turn(state: &mut GameState) {
    let me = state.active_player;
    advance(state, |s, _| s.active_player != me);
}

pub fn named(state: &GameState, player: PlayerId, name: &str) -> Vec<ObjectId> {
    state.players[player.index()].battlefield.iter().copied().filter(|&o| CARD_DEFS[state.objects.get(o).card_def as usize].name == name).collect()
}

/// Passes priority (and declines blocks) until `stop` accepts a decision.
pub fn advance(state: &mut GameState, mut stop: impl FnMut(&GameState, &Decision) -> bool) -> Decision {
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
            other => panic!("unhandled decision {other:?} (halt site {:?})", engine::last_halt_site()),
        };
        engine::step(state, a).unwrap();
    }
    panic!("no stop after 500 decisions");
}

/// Attacks with `attackers`, choosing the opponent for any targeted trigger,
/// and passes until combat damage is done.
pub fn attack_with(state: &mut GameState, attackers: Vec<ObjectId>) {
    advance(state, |_, d| matches!(d, Decision::DeclareAttackers { .. }));
    engine::step(state, Action::DeclareAttackers(attackers)).unwrap();
    advance(state, |s, _| s.step == Step::EndCombat || s.step == Step::Main2);
}

/// Declares `attackers`, answers trigger targets with `target`, and returns the
/// blockers player 1 may use against each attacker.
pub fn attack_and_see_blocks(state: &mut GameState, attackers: Vec<ObjectId>, target: Target) -> Vec<(ObjectId, Vec<ObjectId>)> {
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

/// Passes priority until the stack is empty.
pub fn resolve(state: &mut GameState) {
    while !state.stack.is_empty() {
        match engine::advance_until_decision(state) {
            Decision::CastSpellOrPass { .. } => engine::step(state, Action::Pass).unwrap(),
            other => panic!("unhandled decision {other:?}"),
        }
    }
}

/// Advances to player 0's next upkeep, where it holds priority.
pub fn to_my_next_upkeep(state: &mut GameState) {
    let turn = state.turn;
    advance(state, |s, d| {
        s.turn != turn && s.active_player == P0 && s.step == Step::Upkeep && matches!(d, Decision::CastSpellOrPass { player: P0, .. })
    });
}

/// Activates `source`'s ability `index` and resolves it: `pick` for a target
/// when legal, else the first option offered at each choice.
pub fn activate(state: &mut GameState, source: ObjectId, index: u8, pick: Option<Target>) {
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

pub fn offered_ability(state: &mut GameState, source: ObjectId, index: u8) -> bool {
    matches!(engine::advance_until_decision(state), Decision::CastSpellOrPass { activatable_abilities, .. } if activatable_abilities.contains(&(source, index)))
}

/// Player 0 attacks with `attacker` and player 1 blocks with `blocker`;
/// returns the attacker's damage and whether the blocker died.
pub fn attack_into_block(state: &mut GameState, attacker: ObjectId, blocker: ObjectId) -> (u16, bool) {
    advance(state, |_, d| matches!(d, Decision::DeclareAttackers { .. }));
    engine::step(state, Action::DeclareAttackers(vec![attacker])).unwrap();
    advance(state, |_, d| matches!(d, Decision::DeclareBlockers { .. }));
    engine::step(state, Action::DeclareBlockers(vec![(blocker, attacker)])).unwrap();
    advance(state, |s, _| s.step == Step::EndCombat);
    (state.objects.get(attacker).damage, state.objects.get(blocker).zone == Zone::Graveyard)
}

/// Casts `card` for player 0, answering riot with `haste`.
pub fn cast_with_riot(state: &mut GameState, card: &str, haste: bool) -> ObjectId {
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

/// Makes `o` a 3/3 artifact creature, as Depthshaker Titan does.
pub fn animate(state: &mut GameState, o: ObjectId) {
    use mtg_kernel::state::{ObjectLinkV4, TypeAdditionV1};
    let object = ObjectLinkV4 { object: o, zone_change_count: state.objects.get(o).zone_change_count };
    state.type_additions.push(TypeAdditionV1 { object, artifact: true, creature: true, base_power_toughness: Some((3, 3)), until_end_of_turn: false });
}

/// Activates `source`'s ability `index` and answers its decisions: targets
/// from `targets` in order (none left finishes an "up to"), artifacts to tap
/// from `tap` (then done), and effect options from `options` in order.
pub fn activate_with(state: &mut GameState, source: ObjectId, index: u8, targets: &[Target], tap: &[ObjectId], options: &[u16]) {
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

pub fn zone(state: &GameState, o: ObjectId) -> Zone {
    state.objects.get(o).zone
}

/// Makes `player`'s library card at `position` (0 = top) a `name`.
pub fn library_card(state: &mut GameState, player: PlayerId, position: usize, name: &str) -> ObjectId {
    let o = state.players[player.index()].library[position];
    let obj = state.objects.get_mut(o);
    obj.card_def = id(name);
    obj.name = name.to_string().into();
    obj.v4 = ObjectStateV4::from_card_def(id(name));
    o
}

/// Casts graveyard card `card`, answering a method choice with `method`, a
/// target with player 1, and a discard with the first offer (whose choices are
/// returned).
pub fn cast_from_graveyard(state: &mut GameState, card: ObjectId, method: Option<mtg_kernel::state::CastMethodV4>) -> Vec<ObjectId> {
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

pub fn castable(state: &mut GameState, card: ObjectId) -> bool {
    matches!(advance(state, |_, d| matches!(d, Decision::CastSpellOrPass { .. })), Decision::CastSpellOrPass { castable_spells, .. } if castable_spells.contains(&card))
}

/// Answers decisions with `policy` until it returns `None`.
pub fn drive(state: &mut GameState, mut policy: impl FnMut(&GameState, &Decision) -> Option<Action>) {
    for _ in 0..500 {
        let d = engine::advance_until_decision(state);
        let Some(a) = policy(state, &d) else {
            return;
        };
        engine::step(state, a.clone()).unwrap_or_else(|e| panic!("{a:?} at {d:?}: {e}"));
    }
    panic!("no stop after 500 decisions");
}

/// Player 0 bolts `bolt_target`; player 1 answers with `card` aimed at the
/// Bolt, choosing additional-cost option `option`; each new target choice
/// takes `pick`. Stops once the Bolt has left the stack.
pub fn bolt_and_answer(state: &mut GameState, bolt_target: Target, card: &str, option: Option<u16>, pick: Target) -> ObjectId {
    let bolt = put_in(state, P0, "Lightning Bolt", Zone::Hand);
    let answer = put_in(state, P1, card, Zone::Hand);
    drive(state, |s, d| match d {
        _ if zone(s, bolt) != Zone::Hand && zone(s, bolt) != Zone::Stack => None,
        Decision::CastSpellOrPass { player, castable_spells, .. } => Some(match *player {
            P0 if castable_spells.contains(&bolt) => Action::CastSpell(bolt),
            P1 if castable_spells.contains(&answer) => Action::CastSpell(answer),
            _ => Action::Pass,
        }),
        Decision::ChooseTargets { player: P0, .. } => Some(Action::ChooseTarget(bolt_target)),
        Decision::ChooseTargets { .. } => Some(Action::ChooseTarget(Target::Object(bolt))),
        Decision::ChooseEffectOption { .. } => Some(Action::ChooseEffectOption(option.expect("no additional-cost choice"))),
        Decision::ChooseCastMode { .. } => Some(Action::ChooseCastMode(mtg_kernel::engine::CastMode::Alternative)),
        Decision::ChooseEffectBoolean { .. } => Some(Action::ChooseEffectBoolean(true)),
        Decision::ChooseEffectTargets { legal_targets, .. } => {
            assert!(legal_targets.contains(&pick), "{pick:?} not in {legal_targets:?}");
            Some(Action::ChooseEffectTarget(pick))
        }
        Decision::OrderTriggers { pending, .. } => Some(Action::OrderTriggers((0..pending.len()).collect())),
        other => panic!("unhandled decision {other:?}"),
    });
    answer
}

/// Whether player 1 is offered `card` in response to a Bolt at them.
pub fn offered_against_bolt(state: &mut GameState, card: ObjectId) -> bool {
    put_in(state, P0, "Lightning Bolt", Zone::Hand);
    lands(state, P0, 1);
    let mut offered = false;
    drive(state, |s, d| match d {
        Decision::CastSpellOrPass { player: P0, castable_spells, .. } if s.stack.is_empty() => {
            castable_spells.first().map(|&b| Action::CastSpell(b))
        }
        Decision::CastSpellOrPass { player: P1, castable_spells, .. } => {
            offered = castable_spells.contains(&card);
            None
        }
        Decision::CastSpellOrPass { .. } => Some(Action::Pass),
        Decision::ChooseTargets { .. } => Some(Action::ChooseTarget(Target::Player(P1))),
        other => panic!("unhandled decision {other:?}"),
    });
    offered
}

/// Passes priority and combat; `answer` gives `Some(None)` to stop,
/// `Some(Some(action))` to act, `None` for the default.
pub fn pass_until(state: &mut GameState, mut answer: impl FnMut(&GameState, &Decision) -> Option<Option<Action>>) {
    drive(state, |s, d| {
        if let Some(a) = answer(s, d) {
            return a;
        }
        Some(match d {
            Decision::CastSpellOrPass { .. } => Action::Pass,
            Decision::DeclareAttackers { .. } => Action::DeclareAttackers(vec![]),
            Decision::DeclareBlockers { .. } => Action::DeclareBlockers(vec![]),
            Decision::OrderTriggers { pending, .. } => Action::OrderTriggers((0..pending.len()).collect()),
            other => panic!("unhandled decision {other:?}"),
        })
    });
}
