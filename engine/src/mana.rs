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
#[derive(Clone, Copy)]
pub struct ManaColorSetV1 {
    colors: [ManaColor; ManaColorSetV1::CAPACITY],
    len: u8,
}

impl std::fmt::Debug for ManaColorSetV1 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_slice().fmt(f)
    }
}

impl PartialEq for ManaColorSetV1 {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Eq for ManaColorSetV1 {}

impl ManaColorSetV1 {
    const CAPACITY: usize = ManaColor::ALL.len();

    pub fn from_slice(colors: &[ManaColor]) -> Self {
        let mut set = Self::new();
        for &color in colors {
            set.push(color);
        }
        set
    }

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

    pub fn as_slice(&self) -> &[ManaColor] {
        &self.colors[..usize::from(self.len)]
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[allow(dead_code)]
    pub(crate) fn len(&self) -> usize {
        usize::from(self.len)
    }

    pub fn contains(&self, color: ManaColor) -> bool {
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
    pub choices: ManaColorSetV1,
    /// Mana one tap makes; what the cost doesn't use floats.
    pub amount: u8,
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
    let pool = usable_pool(player, state);
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
    let pool = usable_pool(player, state);
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

thread_local! {
    /// The colors of the spell whose cost is being paid, if any: restricted
    /// floating mana counts only for a monocolored spell of its color.
    static SPELL_COLORS: std::cell::Cell<Option<u8>> = const { std::cell::Cell::new(None) };
}

/// While alive, mana payments are for a spell with these colors (a
/// `mana_colors_mask`).
pub struct SpellPaymentScope(Option<u8>);

impl SpellPaymentScope {
    /// Only when some restricted mana is floating; otherwise nothing to do.
    pub fn enter(state: &GameState, colors: u8) -> Option<SpellPaymentScope> {
        (state.restricted_mana != [[0; 6]; 2]).then(|| SpellPaymentScope(SPELL_COLORS.with(|c| c.replace(Some(colors)))))
    }
}

impl Drop for SpellPaymentScope {
    fn drop(&mut self) {
        SPELL_COLORS.with(|c| c.set(self.0));
    }
}

thread_local! {
    /// While set, mana of any type can be spent on the cost being paid.
    static ANY_MANA_TYPE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// While alive, "mana of any type can be spent" applies to payments.
pub struct AnyManaTypeScope(bool);

impl AnyManaTypeScope {
    pub fn enter(on: bool) -> Option<AnyManaTypeScope> {
        on.then(|| AnyManaTypeScope(ANY_MANA_TYPE.with(|c| c.replace(true))))
    }
}

impl Drop for AnyManaTypeScope {
    fn drop(&mut self) {
        ANY_MANA_TYPE.with(|c| c.set(self.0));
    }
}

fn any_mana_type() -> bool {
    ANY_MANA_TYPE.with(|c| c.get())
}

/// The pips the solver must match by color, plus the generic mana the rest
/// count as: with mana of any type, colored and hybrid pips are generic and
/// a Phyrexian pip takes any mana or 2 life.
fn color_bound_pips(pips: &[Pip]) -> (std::borrow::Cow<'_, [Pip]>, u32) {
    if !any_mana_type() {
        return (std::borrow::Cow::Borrowed(pips), 0);
    }
    let phyrexian: Vec<Pip> = pips.iter().copied().filter(|p| matches!(p, Pip::Phyrexian(_))).collect();
    let generic = (pips.len() - phyrexian.len()) as u32;
    (std::borrow::Cow::Owned(phyrexian), generic)
}

/// The single color restricted mana may pay for now, if any.
pub fn restricted_mana_color_allowed() -> Option<usize> {
    let mask = SPELL_COLORS.with(|c| c.get())?;
    (mask.count_ones() == 1 && mask & 0b1_1111 != 0).then(|| mask.trailing_zeros() as usize)
}

/// Floating mana the current payment may use.
fn usable_pool(player: PlayerId, state: &GameState) -> [u8; 6] {
    let mut pool = state.players[player.index()].mana_pool;
    if state.restricted_mana[player.index()] == [0; 6] {
        return pool;
    }
    let allowed = restricted_mana_color_allowed();
    for (i, amount) in pool.iter_mut().enumerate() {
        if Some(i) != allowed {
            *amount -= state.restricted_mana[player.index()][i];
        }
    }
    pool
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
    let pool = usable_pool(player, state);
    let all_pips: Vec<Pip> = costs.iter().flat_map(|c| c.pips.iter().copied()).collect();
    let (combined_pips, pip_generic) = color_bound_pips(&all_pips);
    let generic: u32 = costs.iter().map(|c| c.generic as u32).sum::<u32>() + pip_generic;
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
            let mut choices = ManaColorSetV1::new();
            def.primary_mana_ability_choices_into(obj.v4.chosen_color, &mut choices);
            if crate::commander::makes_identity_mana(def) {
                let identity = crate::commander::color_identity_mask(state, player);
                let mut kept = ManaColorSetV1::new();
                for &color in choices.as_slice() {
                    if identity & (1 << color.pool_index()) != 0 {
                        kept.push(color);
                    }
                }
                choices = kept;
            }
            if choices.is_empty() {
                continue;
            }
            sources.push(ManaSource {
                id,
                choices,
                amount: def.automatic_mana_amount().unwrap_or(1),
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
        usable_pool(player, state),
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
    let mut used_buf = [false; 64];
    let mut used_heap = Vec::new();
    let used: &mut [bool] = if sources.len() <= used_buf.len() {
        &mut used_buf[..sources.len()]
    } else {
        used_heap.resize(sources.len(), false);
        &mut used_heap
    };
    let (pips, pip_generic) = color_bound_pips(cost.pips);

    if !solve_pips(
        &pips,
        0,
        sources,
        used,
        &mut pool_remaining,
        &mut plan,
    ) {
        return None;
    }

    let generic_needed =
        u32::from(cost.generic) + u32::from(cost.x_count) * u32::from(x_value) + pip_generic;
    if !pay_generic(
        generic_needed,
        sources,
        used,
        &mut pool_remaining,
        &mut plan,
    ) {
        return None;
    }

    Some(plan)
}

/// Whether some color has more single-color pips than the pool and the
/// sources able to make it could supply.
fn pips_exceed_sources(pips: &[Pip], sources: &[ManaSource], pool: &[u8; 6]) -> bool {
    [ManaColor::W, ManaColor::U, ManaColor::B, ManaColor::R, ManaColor::G, ManaColor::C].into_iter().any(|color| {
        let need = pips.iter().filter(|&&pip| pip == Pip::Colored(color)).count();
        need > 0
            && need
                > usize::from(pool[color.pool_index()])
                    + sources
                        .iter()
                        .filter(|source| source.choices.contains(color))
                        .map(|source| usize::from(source.amount))
                        .sum::<usize>()
    })
}

fn solve_pips(
    pips: &[Pip],
    idx: usize,
    sources: &[ManaSource],
    used: &mut [bool],
    pool_remaining: &mut [u8; 6],
    plan: &mut PaymentPlan,
) -> bool {
    // A color needing more pips than its sources can ever make fails here,
    // before the search tries every ordering of sources (exponential in pips).
    if idx == 0 && pips_exceed_sources(pips, sources, pool_remaining) {
        return false;
    }
    let Some(pip) = pips.get(idx) else {
        return true;
    };

    let pair;
    let candidate_colors: &[ManaColor] = match *pip {
        Pip::Phyrexian(_) if any_mana_type() => &ManaColor::ALL,
        Pip::Colored(c) | Pip::Phyrexian(c) => {
            pair = [c, c];
            &pair[..1]
        }
        Pip::Hybrid(a, b) => {
            pair = [a, b];
            &pair
        }
    };

    // Prefer paying from the floating pool: it never needs backtracking,
    // since spending it doesn't remove a source with other uses.
    for &c in candidate_colors {
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
        for &c in candidate_colors {
            if !sources[i].choices.contains(c) {
                continue;
            }
            used[i] = true;
            plan.taps.push((sources[i].id, c));
            let extra = sources[i].amount.saturating_sub(1);
            pool_remaining[c.pool_index()] += extra;
            if solve_pips(pips, idx + 1, sources, used, pool_remaining, plan) {
                return true;
            }
            pool_remaining[c.pool_index()] -= extra;
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
        if let Some(&c) = sources[i].choices.as_slice().first() {
            used[i] = true;
            plan.taps.push((sources[i].id, c));
            // The tap's first mana is spent as the tap; the rest come out of the pool it fills.
            let take = u32::from(sources[i].amount.max(1)).min(needed);
            plan.pool_used[c.pool_index()] += (take - 1) as u8;
            needed -= take;
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
            choices: ManaColorSetV1::from_slice(choices),
            amount: 1,
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
