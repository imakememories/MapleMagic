//! Commander rules (903): casting the commander from the command zone with
//! its tax, its return to the command zone, and commander damage. Every
//! function is a no-op outside Commander games (`GameState::commander` is
//! `None`).

use crate::card_def::{CardDef, CARD_DEFS};
use crate::ids::{ObjectId, PlayerId};
use crate::mana::{Cost, ManaColor, Pip};
use crate::state::{GameState, Zone, COMMANDER_DAMAGE_TO_LOSE};

/// Sources of "one mana of any color in your commander's color identity".
pub fn makes_identity_mana(def: &CardDef) -> bool {
    matches!(def.name, "Arcane Signet" | "Command Tower")
}

/// `player`'s commander's color identity (903.4), from its colors and mana
/// cost; none of the commanders here have other mana symbols in their text.
/// Empty outside Commander games.
pub fn color_identity(state: &GameState, player: PlayerId) -> Vec<ManaColor> {
    let Some(c) = state.commander.as_ref() else {
        return Vec::new();
    };
    let def = &CARD_DEFS[state.objects.get(c.commanders[player.index()]).card_def as usize];
    let mut out: Vec<ManaColor> = def.colors.to_vec();
    for pip in def.cost.pips {
        match *pip {
            Pip::Colored(a) | Pip::Phyrexian(a) => out.push(a),
            Pip::Hybrid(a, b) => out.extend([a, b]),
        }
    }
    out.sort_by_key(|c| c.pool_index());
    out.dedup();
    out
}

/// The owner index of commander `id`, if it is one.
fn owner_slot(state: &GameState, id: ObjectId) -> Option<usize> {
    state.commander.as_ref()?.commanders.iter().position(|&c| c == id)
}

pub fn is_commander(state: &GameState, id: ObjectId) -> bool {
    owner_slot(state, id).is_some()
}

/// 903.8: {2} more for each earlier cast from the command zone.
pub fn tax(state: &GameState, id: ObjectId) -> u8 {
    match (state.commander.as_ref(), owner_slot(state, id)) {
        (Some(c), Some(owner)) => u8::try_from(2 * u32::from(c.casts[owner])).unwrap_or(u8::MAX),
        _ => 0,
    }
}

/// `cost` plus the commander tax when `id` is cast from the command zone.
pub fn with_tax(cost: Cost, state: &GameState, id: ObjectId, from: Zone) -> Cost {
    if from != Zone::Command {
        return cost;
    }
    Cost {
        generic: cost.generic.saturating_add(tax(state, id)),
        ..cost
    }
}

/// Counts a completed cast of `spell` from the command zone.
pub(crate) fn record_cast(state: &mut GameState, spell: ObjectId) {
    if let Some(owner) = owner_slot(state, spell) {
        let casts = &mut state.commander.as_mut().unwrap().casts[owner];
        *casts = casts.saturating_add(1);
    }
}

/// 903.10a: combat damage a commander deals to a player is tracked per
/// commander.
pub(crate) fn record_combat_damage(state: &mut GameState, source: ObjectId, player: PlayerId, amount: i32) {
    if let Some(owner) = owner_slot(state, source) {
        state.commander.as_mut().unwrap().damage[player.index()][owner] += amount;
    }
}

/// 903.9b: a commander that would go to a hand or library goes to the
/// command zone instead. The owner may decline; this engine never does.
/// 903.9: a commander whose owner may now put it into the command zone: one
/// put into a graveyard or exile (a state-based action, 903.9a) or into its
/// owner's hand or library (a replacement, 903.9b), unless the owner already
/// declined for this incarnation. The active player chooses first.
///
/// The engine commits zone changes inside effects without pausing, so the
/// hand/library choice is asked as soon as the effect finishes, before
/// triggers or priority. That matches 903.9b for every card here; it would
/// differ only for one effect that moves a commander to a hand or library and
/// then uses that same hand or library.
pub fn pending_zone_choice(state: &GameState) -> Option<(PlayerId, ObjectId)> {
    let c = state.commander.as_ref()?;
    [state.active_player, state.active_player.opponent()].into_iter().find_map(|p| {
        let id = c.commanders[p.index()];
        let o = state.objects.get(id);
        (matches!(o.zone, Zone::Graveyard | Zone::Exile | Zone::Hand | Zone::Library)
            && c.declined[p.index()] != Some(o.zone_change_count))
        .then_some((p, id))
    })
}

/// Answers [`pending_zone_choice`]: the command zone, or leave it where it is.
pub(crate) fn choose_zone(state: &mut GameState, to_command_zone: bool) -> Result<(), String> {
    let (p, id) = pending_zone_choice(state).ok_or("no commander zone choice is pending")?;
    if to_command_zone {
        crate::event::commit(state, crate::event::ProposedEvent::zone_change(id, Zone::Command));
    } else {
        let generation = state.objects.get(id).zone_change_count;
        state.commander.as_mut().expect("a pending choice implies a Commander game").declined[p.index()] =
            Some(generation);
    }
    Ok(())
}

/// Commander state-based actions; returns whether anything changed.
/// 704.6c: a player dealt 21 or more combat damage by one commander loses.
/// (704.6d, returning a commander, is the owner's choice: see
/// [`pending_zone_choice`].)
pub(crate) fn state_based_actions(state: &mut GameState) -> bool {
    let Some(c) = state.commander.as_ref() else {
        return false;
    };
    let damage = c.damage;
    let mut changed = false;
    for p in [PlayerId::P0, PlayerId::P1] {
        let ps = &mut state.players[p.index()];
        if !ps.has_lost && damage[p.index()].iter().any(|&d| d >= COMMANDER_DAMAGE_TO_LOSE) {
            ps.has_lost = true;
            changed = true;
        }
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card_def::card_id_by_name;
    use crate::engine::{advance_until_decision, step, Action, Decision};
    use crate::event::{commit, ProposedEvent};
    use crate::mana::ManaColor;
    use crate::state::{Step, COMMANDER_STARTING_LIFE};

    /// A Commander game with Goblin Tomb Raider ({R}) as both commanders and
    /// empty libraries, in player 0's first main phase.
    fn game() -> GameState {
        game_with("Goblin Tomb Raider")
    }

    fn game_with(commander_name: &str) -> GameState {
        let card = card_id_by_name(commander_name).unwrap();
        let mut state = GameState::new_commander_game_v1(
            &[],
            &[],
            [card, card],
            |c| crate::card_def::CARD_DEFS[c as usize].name.to_string(),
            1,
            PlayerId::P0,
        );
        state.step = Step::Main1;
        state
    }

    fn commander(state: &GameState, owner: PlayerId) -> ObjectId {
        state.commander.as_ref().unwrap().commanders[owner.index()]
    }

    fn castable(state: &mut GameState, id: ObjectId) -> bool {
        match advance_until_decision(state) {
            Decision::CastSpellOrPass { castable_spells, .. } => castable_spells.contains(&id),
            other => panic!("expected CastSpellOrPass, got {other:?}"),
        }
    }

    /// Answers the 903.9 choice `advance_until_decision` must now offer for `id`.
    fn choose(state: &mut GameState, id: ObjectId, to_command_zone: bool) {
        match advance_until_decision(state) {
            Decision::ChooseCommanderZone { player, commander } => {
                assert_eq!((player, commander), (state.objects.get(id).owner, id));
            }
            other => panic!("expected ChooseCommanderZone, got {other:?}"),
        }
        step(state, Action::ChooseCommanderZone(to_command_zone)).unwrap();
    }

    /// Casts `id` and passes priority until the stack is empty.
    fn cast_and_resolve(state: &mut GameState, id: ObjectId) {
        step(state, Action::CastSpell(id)).unwrap();
        while let Decision::CastSpellOrPass { .. } = advance_until_decision(state) {
            if state.stack.is_empty() {
                break;
            }
            step(state, Action::Pass).unwrap();
        }
    }

    #[test]
    fn commanders_start_in_the_command_zone_at_40_life() {
        let state = game();
        for p in [PlayerId::P0, PlayerId::P1] {
            let id = commander(&state, p);
            assert_eq!(state.objects.get(id).zone, Zone::Command);
            assert_eq!(state.objects.get(id).owner, p);
            assert!(state.command.contains(&id));
            assert_eq!(state.players[p.index()].life, COMMANDER_STARTING_LIFE);
        }
    }

    #[test]
    fn ordinary_games_have_no_commander_state() {
        let state = GameState::new_from_libraries(&[], &[], |c| format!("card-{c}"), 1);
        assert!(state.commander.is_none());
        assert_eq!(state.players[0].life, crate::state::STARTING_LIFE);
    }

    #[test]
    fn casting_from_the_command_zone_pays_a_growing_tax() {
        let mut state = game();
        let raider = commander(&state, PlayerId::P0);
        let red = ManaColor::R.pool_index();

        state.players[0].mana_pool[red] = 1;
        assert!(castable(&mut state, raider), "first cast costs only {{R}}");
        cast_and_resolve(&mut state, raider);
        assert_eq!(state.objects.get(raider).zone, Zone::Battlefield);
        assert_eq!(state.commander.as_ref().unwrap().casts, [1, 0]);
        assert_eq!(tax(&state, raider), 2);

        // Dies; its owner puts it into the command zone (903.9a).
        commit(&mut state, ProposedEvent::zone_change(raider, Zone::Graveyard));
        choose(&mut state, raider, true);
        assert_eq!(state.objects.get(raider).zone, Zone::Command);
        assert!(state.command.contains(&raider));
        assert!(!state.players[0].graveyard.contains(&raider));

        state.players[0].mana_pool[red] = 2;
        assert!(!castable(&mut state, raider), "{{2}}{{R}} is not payable with two mana");
        state.players[0].mana_pool[red] = 3;
        assert!(castable(&mut state, raider));
        cast_and_resolve(&mut state, raider);
        assert_eq!(state.objects.get(raider).zone, Zone::Battlefield);
        assert_eq!(state.players[0].mana_pool, [0; 6], "the tax was paid");
        assert_eq!(state.commander.as_ref().unwrap().casts, [2, 0]);
        assert_eq!(tax(&state, raider), 4);
    }

    #[test]
    fn opponent_cannot_cast_your_commander() {
        let mut state = game();
        state.players[0].mana_pool[ManaColor::R.pool_index()] = 1;
        let theirs = commander(&state, PlayerId::P1);
        assert!(!castable(&mut state, theirs));
    }

    #[test]
    fn owner_may_move_a_commander_to_the_command_zone_or_leave_it() {
        // 903.9a (graveyard, exile) and 903.9b (hand, library) are both "may".
        for to in [Zone::Graveyard, Zone::Exile, Zone::Hand, Zone::Library] {
            for accept in [true, false] {
                let mut state = game();
                let raider = commander(&state, PlayerId::P0);
                state.players[0].mana_pool[ManaColor::R.pool_index()] = 1;
                castable(&mut state, raider);
                cast_and_resolve(&mut state, raider);
                commit(&mut state, ProposedEvent::zone_change(raider, to));
                choose(&mut state, raider, accept);
                let expected = if accept { Zone::Command } else { to };
                assert_eq!(state.objects.get(raider).zone, expected, "{to:?}, accept {accept}");
                // Asked once per zone change.
                assert!(
                    matches!(advance_until_decision(&mut state), Decision::CastSpellOrPass { .. }),
                    "{to:?}, accept {accept}"
                );
            }
        }
    }

    #[test]
    fn a_commander_kept_in_hand_is_cast_without_tax() {
        // 903.8: only casts from the command zone pay or add to the tax.
        let mut state = game();
        let raider = commander(&state, PlayerId::P0);
        let red = ManaColor::R.pool_index();
        state.players[0].mana_pool[red] = 1;
        castable(&mut state, raider);
        cast_and_resolve(&mut state, raider);
        commit(&mut state, ProposedEvent::zone_change(raider, Zone::Hand));
        choose(&mut state, raider, false);
        assert!(state.players[0].hand.contains(&raider));
        state.players[0].mana_pool[red] = 1;
        assert!(castable(&mut state, raider), "{{R}} from hand");
        cast_and_resolve(&mut state, raider);
        assert_eq!(state.objects.get(raider).zone, Zone::Battlefield);
        assert_eq!(state.commander.as_ref().unwrap().casts, [1, 0]);
        // Declined in the graveyard, then exiled: a new zone change, asked again.
        commit(&mut state, ProposedEvent::zone_change(raider, Zone::Graveyard));
        choose(&mut state, raider, false);
        commit(&mut state, ProposedEvent::zone_change(raider, Zone::Exile));
        choose(&mut state, raider, true);
        assert_eq!(state.objects.get(raider).zone, Zone::Command);
    }

    #[test]
    fn the_active_player_chooses_first() {
        let mut state = game();
        let (mine, theirs) = (commander(&state, PlayerId::P0), commander(&state, PlayerId::P1));
        commit(&mut state, ProposedEvent::zone_change(theirs, Zone::Exile));
        commit(&mut state, ProposedEvent::zone_change(mine, Zone::Exile));
        choose(&mut state, mine, false);
        choose(&mut state, theirs, true);
        assert_eq!(state.objects.get(mine).zone, Zone::Exile);
        assert_eq!(state.objects.get(theirs).zone, Zone::Command);
    }

    #[test]
    fn twenty_one_commander_damage_loses_even_at_high_life() {
        // Masked Meower: {R}, 1/1 haste.
        let mut state = game_with("Masked Meower");
        let raider = commander(&state, PlayerId::P0);
        state.players[0].mana_pool[ManaColor::R.pool_index()] = 1;
        castable(&mut state, raider);
        cast_and_resolve(&mut state, raider);
        state.commander.as_mut().unwrap().damage[1][0] = 20;

        // Haste: attack this turn; player 1 doesn't block.
        let mut attacked = false;
        loop {
            match advance_until_decision(&mut state) {
                Decision::CastSpellOrPass { .. } => step(&mut state, Action::Pass).unwrap(),
                Decision::DeclareAttackers { eligible, .. } => {
                    assert!(eligible.contains(&raider));
                    step(&mut state, Action::DeclareAttackers(vec![raider])).unwrap();
                    attacked = true;
                }
                Decision::DeclareBlockers { .. } => {
                    step(&mut state, Action::DeclareBlockers(vec![])).unwrap()
                }
                Decision::GameOver { winner } => {
                    assert_eq!(winner, Some(PlayerId::P0));
                    break;
                }
                other => panic!("unexpected decision {other:?}"),
            }
            assert!(state.turn == 1, "the game should end in the first combat");
        }
        assert!(attacked);
        assert_eq!(state.commander.as_ref().unwrap().damage[1][0], 21);
        assert!(state.players[1].life > 0, "lost to commander damage, not life");
        assert!(state.players[1].has_lost);
    }
}
