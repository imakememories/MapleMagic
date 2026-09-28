//! EXACT backtracking mana payment solver.
//!
//! `Cost` describes a spell/ability's mana requirement as a set of pips
//! (colored / hybrid / phyrexian) plus a generic amount. `solve` finds a
//! `PaymentPlan` -- which floating pool mana and which untapped mana
//! sources pay for which part of the cost -- by exact backtracking over
//! pip/source assignment, not a greedy heuristic. See
//! `backtracking_is_required_for_modal_sources` below for a case greedy
//! gets wrong: a dual source assigned to the wrong pip first can strand a
//! later pip only that dual could pay.

use crate::ids::{ObjectId, PlayerId};
use crate::state::GameState;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ManaColor {
    W,
    U,
    B,
    R,
    G,
    C,
}

impl ManaColor {
    pub const ALL: [ManaColor; 6] = [
        ManaColor::W,
        ManaColor::U,
        ManaColor::B,
        ManaColor::R,
        ManaColor::G,
        ManaColor::C,
    ];

    /// Index into `PlayerState::mana_pool` ([W, U, B, R, G, C]).
    pub fn pool_index(self) -> usize {
        match self {
            ManaColor::W => 0,
            ManaColor::U => 1,
            ManaColor::B => 2,
            ManaColor::R => 3,
            ManaColor::G => 4,
            ManaColor::C => 5,
        }
    }

    pub fn from_code(c: &str) -> Option<ManaColor> {
        match c {
            "W" => Some(ManaColor::W),
            "U" => Some(ManaColor::U),
            "B" => Some(ManaColor::B),
            "R" => Some(ManaColor::R),
            "G" => Some(ManaColor::G),
            "C" => Some(ManaColor::C),
            _ => None,
        }
    }
}

/// Fixed-capacity, non-allocating set of mana colors. Bounded by
/// `ManaColor::ALL.len()`, which every printed mana ability stays within, so
/// capacity is never exceeded in practice. Exists so hot paths that must not
/// touch the heap (such as flat-action validation) can compute the same
/// small color sets that `CardDef::primary_mana_ability_choices` and
/// `engine::available_mana_ability_choices` return as an owned `Vec`
/// elsewhere.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ManaColorSetV1 {
    colors: [ManaColor; ManaColorSetV1::CAPACITY],
    len: u8,
}

impl ManaColorSetV1 {
    const CAPACITY: usize = ManaColor::ALL.len();

    pub(crate) fn new() -> Self {
        Self {
            colors: [ManaColor::W; ManaColorSetV1::CAPACITY],
            len: 0,
        }
    }

    /// Appends `color`. Every real caller stays within `CAPACITY` colors
    /// (see the type doc); the bounds guard only prevents a panic if that
    /// invariant is ever violated instead of silently corrupting state.
    pub(crate) fn push(&mut self, color: ManaColor) {
        if usize::from(self.len) < ManaColorSetV1::CAPACITY {
            self.colors[usize::from(self.len)] = color;
            self.len += 1;
        }
    }

    pub(crate) fn as_slice(&self) -> &[ManaColor] {
        &self.colors[..usize::from(self.len)]
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[allow(dead_code)]
    pub(crate) fn len(&self) -> usize {
        usize::from(self.len)
    }

    pub(crate) fn contains(&self, color: ManaColor) -> bool {
        self.as_slice().contains(&color)
    }
}

/// Deterministic policy for spending already-floating mana on generic
/// requirements. Magic permits any color, so the choice is strategically
/// observable whenever multiple colors remain. XMage's `ManaCostImpl`
/// checks colorless, black, blue, white, green, then red; matching that
/// order keeps automatic kernel payments aligned with the reference runner
/// and, importantly for Rally, preserves red before green when possible.
const GENERIC_POOL_PAYMENT_ORDER: [ManaColor; 6] = [
    ManaColor::C,
    ManaColor::B,
    ManaColor::U,
    ManaColor::W,
    ManaColor::G,
    ManaColor::R,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum Pip {
    Colored(ManaColor),
    Hybrid(ManaColor, ManaColor),
    Phyrexian(ManaColor),
}

/// A spell/ability's mana requirement. `pips` is `'static` because every
/// `Cost` this increment constructs comes from the generated `CARD_DEFS`
/// table (see `card_def.rs` / `build.rs`); nothing here prevents a future
/// increment adding an owned-slice variant for runtime-built costs (e.g. an
/// alternative cost).
/// Derives `Serialize` so the diagnostic full-state hash includes a pending
/// cast's exact override. It deliberately does not derive `Deserialize`:
/// rebuilding a `&'static [Pip]` requires registry ownership that the current
/// snapshot API does not need.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct Cost {
    pub pips: &'static [Pip],
    pub generic: u8,
    pub x_count: u8,
}

impl Cost {
    pub const fn zero() -> Cost {
        Cost {
            pips: &[],
            generic: 0,
            x_count: 0,
        }
    }
}

/// An untapped mana-producing permanent available to pay a cost. `choices`
/// lists every color it could produce (basics have exactly one; this shape
/// is what makes multi-choice producers -- and therefore backtracking --
/// meaningful).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManaSource {
    pub id: ObjectId,
    pub choices: Vec<ManaColor>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PaymentPlan {
    /// Newly tapped sources and the color each was tapped for, in the order
    /// they were committed (provenance).
    pub taps: Vec<(ObjectId, ManaColor)>,
    /// How much of the already-floating pool was spent, by color ([W, U,
    /// B, R, G, C], matching `PlayerState::mana_pool`).
    pub pool_used: [u8; 6],
    /// Life paid for phyrexian pips.
    pub life_paid: i32,
}

/// Rule 119.4: a player may pay life only if their life total is at least
/// the amount paid, and paying zero life is always allowed. The naive
/// `life_paid <= life` comparison wrongly rejected every zero-life payment
/// plan once a player's life went negative mid-resolution (Chain Lightning's
/// copy payment after lethal damage, before state-based actions), which made
/// the kernel finish resolution and reach lethal SBA while XMage was still
/// waiting on the copy retarget: the dominant CP7 panel void class.
pub(crate) fn life_payment_affordable(life_paid: i32, life: i32) -> bool {
    life_paid == 0 || life_paid <= life
}

/// Gathers `player`'s floating pool and untapped mana sources from `state`
/// and calls `solve`.
pub fn can_pay(
    cost: &Cost,
    x_value: u8,
    player: PlayerId,
    state: &GameState,
) -> Option<PaymentPlan> {
    let sources = gather_sources(player, state);
    let pool = state.players[player.index()].mana_pool;
    solve(cost, x_value, pool, &sources)
        .filter(|plan| life_payment_affordable(plan.life_paid, state.players[player.index()].life))
}

/// Owned-pip counterpart to `can_pay` for serialized resolution-time costs.
/// Card definitions keep `Cost` pips static, while a suspended effect owns
/// its color vector and therefore borrows it only for this solver call.
pub fn can_pay_owned(
    pips: &[Pip],
    generic: u8,
    player: PlayerId,
    state: &GameState,
) -> Option<PaymentPlan> {
    let sources = gather_sources(player, state);
    let pool = state.players[player.index()].mana_pool;
    let mut plan = PaymentPlan::default();
    let mut pool_remaining = pool;
    let mut used = vec![false; sources.len()];
    if !solve_pips(pips, 0, &sources, &mut used, &mut pool_remaining, &mut plan)
        || !pay_generic(
            u32::from(generic),
            &sources,
            &mut used,
            &mut pool_remaining,
            &mut plan,
        )
    {
        return None;
    }
    life_payment_affordable(plan.life_paid, state.players[player.index()].life).then_some(plan)
}

/// Like `can_pay`, but checks/solves 2+ costs *together* against the same
/// pool of mana sources -- Goblin Bushwhacker's base cost + its optional
/// Kicker cost, paid as one combined announcement (601.2b/f), never as two
/// independent affordability checks that could double-count a source. All
/// `pips` are concatenated (order doesn't matter to the solver) and
/// `generic`/`x_count` are summed.
pub fn can_pay_combined(
    costs: &[&Cost],
    x_value: u8,
    player: PlayerId,
    state: &GameState,
) -> Option<PaymentPlan> {
    let sources = gather_sources(player, state);
    let pool = state.players[player.index()].mana_pool;
    let combined_pips: Vec<Pip> = costs.iter().flat_map(|c| c.pips.iter().copied()).collect();
    let generic: u32 = costs.iter().map(|c| c.generic as u32).sum();
    let x_count: u32 = costs.iter().map(|c| c.x_count as u32).sum();

    let mut plan = PaymentPlan::default();
    let mut pool_remaining = pool;
    let mut used = vec![false; sources.len()];
    if !solve_pips(
        &combined_pips,
        0,
        &sources,
        &mut used,
        &mut pool_remaining,
        &mut plan,
    ) {
        return None;
    }
    if !pay_generic(
        generic + x_count * u32::from(x_value),
        &sources,
        &mut used,
        &mut pool_remaining,
        &mut plan,
    ) {
        return None;
    }
    life_payment_affordable(plan.life_paid, state.players[player.index()].life).then_some(plan)
}

pub fn gather_sources(player: PlayerId, state: &GameState) -> Vec<ManaSource> {
    let mut sources = Vec::new();
    for &id in &state.players[player.index()].battlefield {
        let obj = state.objects.get(id);
        if obj.tapped {
            continue;
        }
        let def = &crate::card_def::CARD_DEFS[obj.card_def as usize];
        // `produces_mana` also covers one-shot production such as
        // Burning-Tree Emissary's ETB trigger. The generated
        // primary mana-choice contract is the narrower repeatable tap-source
        // contract, including an incarnation-local chosen color. A creature
        // source also obeys summoning sickness for the tap symbol in its
        // activation cost.
        if def.is_automatic_payment_mana_source()
            && !(def.has_type(crate::card_def::CardType::Creature) && obj.summoning_sick)
        {
            sources.push(ManaSource {
                id,
                choices: def.primary_mana_ability_choices(obj.v4.chosen_color),
            });
        }
    }
    sources
}

/// Solves a mana cost while reserving permanents for other components of the
/// same cost. Heap Gate uses this to reserve both itself and the other Gate
/// that must remain untapped until the complete cost is paid.
pub fn can_pay_excluding_sources(
    cost: &Cost,
    x_value: u8,
    player: PlayerId,
    state: &GameState,
    excluded: &[ObjectId],
) -> Option<PaymentPlan> {
    let sources = gather_sources(player, state)
        .into_iter()
        .filter(|source| !excluded.contains(&source.id))
        .collect::<Vec<_>>();
    solve(
        cost,
        x_value,
        state.players[player.index()].mana_pool,
        &sources,
    )
    .filter(|plan| life_payment_affordable(plan.life_paid, state.players[player.index()].life))
}

/// One-source convenience wrapper for paid mana abilities whose own tap cost
/// must not also pay their mana component.
pub fn can_pay_excluding_source(
    cost: &Cost,
    x_value: u8,
    player: PlayerId,
    state: &GameState,
    excluded: ObjectId,
) -> Option<PaymentPlan> {
    can_pay_excluding_sources(cost, x_value, player, state, &[excluded])
}

/// Exact backtracking solve. Colored/hybrid/phyrexian pips are satisfied
/// first (pool mana preferred over tapping a new source, since spending
/// pool never removes a source), then any leftover pool + untapped sources
/// pay the generic amount. Paying pips before generic is what makes the
/// generic pass safe to do greedily: by the time it runs, every colored
/// requirement is already locked in, so which specific leftover
/// source/color pays generic can never strand a pip.
pub fn solve(
    cost: &Cost,
    x_value: u8,
    pool: [u8; 6],
    sources: &[ManaSource],
) -> Option<PaymentPlan> {
    let mut plan = PaymentPlan::default();
    let mut pool_remaining = pool;
    let mut used = vec![false; sources.len()];

    if !solve_pips(
        cost.pips,
        0,
        sources,
        &mut used,
        &mut pool_remaining,
        &mut plan,
    ) {
        return None;
    }

    let generic_needed = u32::from(cost.generic) + u32::from(cost.x_count) * u32::from(x_value);
    if !pay_generic(
        generic_needed,
        sources,
        &mut used,
        &mut pool_remaining,
        &mut plan,
    ) {
        return None;
    }

    Some(plan)
}

fn solve_pips(
    pips: &[Pip],
    idx: usize,
    sources: &[ManaSource],
    used: &mut [bool],
    pool_remaining: &mut [u8; 6],
    plan: &mut PaymentPlan,
) -> bool {
    let Some(pip) = pips.get(idx) else {
        return true;
    };

    let candidate_colors: Vec<ManaColor> = match *pip {
        Pip::Colored(c) => vec![c],
        Pip::Hybrid(a, b) => vec![a, b],
        Pip::Phyrexian(c) => vec![c],
    };

    // Prefer paying from the floating pool: it never needs backtracking,
    // since spending it doesn't remove a source with other uses.
    for &c in &candidate_colors {
        let pi = c.pool_index();
        if pool_remaining[pi] > 0 {
            pool_remaining[pi] -= 1;
            plan.pool_used[pi] += 1;
            if solve_pips(pips, idx + 1, sources, used, pool_remaining, plan) {
                return true;
            }
            pool_remaining[pi] += 1;
            plan.pool_used[pi] -= 1;
        }
    }

    // Try tapping each untapped source capable of one of the candidate
    // colors. This is the branch that requires real backtracking: a wrong
    // choice here can strand a later pip.
    for i in 0..sources.len() {
        if used[i] {
            continue;
        }
        for &c in &candidate_colors {
            if !sources[i].choices.contains(&c) {
                continue;
            }
            used[i] = true;
            plan.taps.push((sources[i].id, c));
            if solve_pips(pips, idx + 1, sources, used, pool_remaining, plan) {
                return true;
            }
            plan.taps.pop();
            used[i] = false;
        }
    }

    // Phyrexian pips may also be paid with 2 life instead of mana.
    if let Pip::Phyrexian(_) = pip {
        plan.life_paid += 2;
        if solve_pips(pips, idx + 1, sources, used, pool_remaining, plan) {
            return true;
        }
        plan.life_paid -= 2;
    }

    false
}

fn pay_generic(
    mut needed: u32,
    sources: &[ManaSource],
    used: &mut [bool],
    pool_remaining: &mut [u8; 6],
    plan: &mut PaymentPlan,
) -> bool {
    for color in GENERIC_POOL_PAYMENT_ORDER {
        let pi = color.pool_index();
        let amt = &mut pool_remaining[pi];
        while needed > 0 && *amt > 0 {
            *amt -= 1;
            plan.pool_used[pi] += 1;
            needed -= 1;
        }
    }
    for i in 0..sources.len() {
        if needed == 0 {
            break;
        }
        if used[i] {
            continue;
        }
        if let Some(&c) = sources[i].choices.first() {
            used[i] = true;
            plan.taps.push((sources[i].id, c));
            needed -= 1;
        }
    }
    needed == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn src(id: u32, choices: &[ManaColor]) -> ManaSource {
        ManaSource {
            id: ObjectId(id),
            choices: choices.to_vec(),
        }
    }

    #[test]
    fn simple_same_color_cost_is_satisfied() {
        let cost = Cost {
            pips: &[Pip::Colored(ManaColor::R)],
            generic: 0,
            x_count: 0,
        };
        let sources = vec![src(0, &[ManaColor::R])];
        let plan = solve(&cost, 0, [0; 6], &sources).expect("should pay");
        assert_eq!(plan.taps, vec![(ObjectId(0), ManaColor::R)]);
    }

    #[test]
    fn insufficient_mana_returns_none() {
        let cost = Cost {
            pips: &[Pip::Colored(ManaColor::R), Pip::Colored(ManaColor::R)],
            generic: 0,
            x_count: 0,
        };
        let sources = vec![src(0, &[ManaColor::R])];
        assert_eq!(solve(&cost, 0, [0; 6], &sources), None);
    }

    #[test]
    fn generic_paid_by_leftover_any_color_source() {
        let cost = Cost {
            pips: &[Pip::Colored(ManaColor::R)],
            generic: 1,
            x_count: 0,
        };
        let sources = vec![src(0, &[ManaColor::R]), src(1, &[ManaColor::G])];
        let plan = solve(&cost, 0, [0; 6], &sources).expect("should pay");
        assert_eq!(plan.taps.len(), 2);
    }

    #[test]
    fn generic_pool_payment_matches_xmage_and_preserves_red_before_green() {
        let cost = Cost {
            pips: &[Pip::Colored(ManaColor::R)],
            generic: 2,
            x_count: 0,
        };
        let mut pool = [0u8; 6];
        pool[ManaColor::R.pool_index()] = 3;
        pool[ManaColor::G.pool_index()] = 1;

        let plan = solve(&cost, 0, pool, &[]).expect("R3 G1 pays {2}{R}");
        assert_eq!(plan.pool_used[ManaColor::R.pool_index()], 2);
        assert_eq!(plan.pool_used[ManaColor::G.pool_index()], 1);
    }

    #[test]
    fn pool_mana_is_used_before_tapping_new_sources() {
        let cost = Cost {
            pips: &[Pip::Colored(ManaColor::R)],
            generic: 0,
            x_count: 0,
        };
        let mut pool = [0u8; 6];
        pool[ManaColor::R.pool_index()] = 1;
        let sources = vec![src(0, &[ManaColor::R])];
        let plan = solve(&cost, 0, pool, &sources).expect("should pay");
        assert!(
            plan.taps.is_empty(),
            "should have used pool mana, not tapped a source"
        );
        assert_eq!(plan.pool_used[ManaColor::R.pool_index()], 1);
    }

    #[test]
    fn hybrid_pip_can_be_paid_by_either_color() {
        let cost = Cost {
            pips: &[Pip::Hybrid(ManaColor::R, ManaColor::G)],
            generic: 0,
            x_count: 0,
        };
        let sources = vec![src(0, &[ManaColor::G])];
        let plan = solve(&cost, 0, [0; 6], &sources).expect("should pay via G");
        assert_eq!(plan.taps, vec![(ObjectId(0), ManaColor::G)]);
    }

    #[test]
    fn phyrexian_pip_can_be_paid_with_life() {
        let cost = Cost {
            pips: &[Pip::Phyrexian(ManaColor::B)],
            generic: 0,
            x_count: 0,
        };
        let plan = solve(&cost, 0, [0; 6], &[]).expect("should pay via life");
        assert_eq!(plan.life_paid, 2);
        assert!(plan.taps.is_empty());
    }

    /// The scenario greedy gets wrong: pip 0 (W) is satisfiable by either
    /// source; pip 1 (U) is satisfiable ONLY by the dual. A first-fit
    /// greedy that considers sources in declaration order [dual, w_only]
    /// assigns the dual to the W pip (since it's tried first and can
    /// produce W), stranding the U pip. Exact backtracking must retry with
    /// w_only for the W pip, freeing the dual for U.
    #[test]
    fn backtracking_is_required_for_modal_sources() {
        let cost = Cost {
            pips: &[Pip::Colored(ManaColor::W), Pip::Colored(ManaColor::U)],
            generic: 0,
            x_count: 0,
        };
        let dual = src(0, &[ManaColor::W, ManaColor::U]);
        let w_only = src(1, &[ManaColor::W]);
        let sources = vec![dual, w_only];

        let plan = solve(&cost, 0, [0; 6], &sources).expect("a valid assignment exists");
        let paid_colors: Vec<ManaColor> = plan.taps.iter().map(|(_, c)| *c).collect();
        assert!(paid_colors.contains(&ManaColor::W));
        assert!(paid_colors.contains(&ManaColor::U));
        // The dual (id 0) must have been the one to pay U, since w_only
        // (id 1) cannot.
        assert!(plan.taps.contains(&(ObjectId(0), ManaColor::U)));
    }

    #[test]
    fn x_cost_adds_to_generic_requirement() {
        let cost = Cost {
            pips: &[],
            generic: 0,
            x_count: 1,
        };
        let sources = vec![src(0, &[ManaColor::R]), src(1, &[ManaColor::R])];
        assert!(solve(&cost, 2, [0; 6], &sources).is_some());
        assert!(solve(&cost, 3, [0; 6], &sources).is_none());
    }

    #[test]
    fn repeated_x_symbols_each_add_the_chosen_value() {
        let cost = Cost {
            pips: &[],
            generic: 1,
            x_count: 2,
        };
        let sources = (0..5)
            .map(|id| src(id, &[ManaColor::R]))
            .collect::<Vec<_>>();
        assert!(solve(&cost, 2, [0; 6], &sources).is_some());
        assert!(solve(&cost, 3, [0; 6], &sources).is_none());
    }

    #[test]
    fn can_pay_combined_needs_both_costs_paid_from_the_same_pool() {
        // Goblin Bushwhacker's shape: base {R}, Kicker {R} -- exactly 2
        // untapped Mountains covers both combined; 1 Mountain covers
        // neither the combined check nor a double-count of the same source.
        use crate::state::GameState;
        let mountain = crate::card_def::card_id_by_name("Mountain").expect("Mountain in CARD_DEFS");
        let base = Cost {
            pips: &[Pip::Colored(ManaColor::R)],
            generic: 0,
            x_count: 0,
        };
        let kicker = Cost {
            pips: &[Pip::Colored(ManaColor::R)],
            generic: 0,
            x_count: 0,
        };

        let mut one_mountain =
            GameState::new_from_libraries(&[mountain], &[mountain], |_| "Mountain".to_string(), 1);
        let land = one_mountain.draw_card(PlayerId::P0).unwrap();
        one_mountain.move_hand_to_battlefield(PlayerId::P0, land);
        assert!(
            can_pay_combined(&[&base, &kicker], 0, PlayerId::P0, &one_mountain).is_none(),
            "1 Mountain can't pay 2 {{R}} pips at once"
        );

        let mut two_mountains = GameState::new_from_libraries(
            &[mountain, mountain],
            &[mountain],
            |_| "Mountain".to_string(),
            1,
        );
        let l0 = two_mountains.draw_card(PlayerId::P0).unwrap();
        let l1 = two_mountains.draw_card(PlayerId::P0).unwrap();
        two_mountains.move_hand_to_battlefield(PlayerId::P0, l0);
        two_mountains.move_hand_to_battlefield(PlayerId::P0, l1);
        let plan = can_pay_combined(&[&base, &kicker], 0, PlayerId::P0, &two_mountains)
            .expect("2 Mountains should pay both {R} pips");
        assert_eq!(plan.taps.len(), 2);
    }
}
