//! Interactions between the Ognis deck's cards (audit: docs/audit/ognis/patterns.md).

mod common;
use common::*;
use mtg_kernel::engine::{self, Action, Decision};
use mtg_kernel::mana::ManaColor;
use mtg_kernel::state::Zone;

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
