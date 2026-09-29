//! Lean game wrapper over the raw engine: a flat list of semantic choices per
//! decision, auto-paid casts, one-creature-at-a-time combat, and hidden-zone
//! redeterminization for IS-MCTS.
//!
//! Every engine `Decision` is turned into a `Vec<Choice>`. A choice carries a
//! stable semantic `key` (card definitions and relations, never raw object
//! ids) so tree statistics line up across determinizations, plus the engine
//! work needed to execute it. Decisions with exactly one choice are applied
//! automatically and never surface to the caller.

use mtg_kernel::card_def::{ManaAbilityCostDef, CARD_DEFS};
use mtg_kernel::engine::{
    self, available_mana_ability_choices, mana_ability_cost_targets, Action, CastMode, Decision,
    OptionalCostChoice,
};
use mtg_kernel::event::{self, ProposedEvent};
use mtg_kernel::ids::{ObjectId, PlayerId};
use mtg_kernel::mana::ManaColor;
use mtg_kernel::runtime_decks::{runtime_deck_by_id, RUNTIME_DECKS};
use mtg_kernel::state::{GameState, ObjectStateV4, SplitMix64, Target, Zone};

/// Hard stop for runaway games (loops the engine cannot detect).
pub const MAX_TURNS: u32 = 60;
const MAX_AUTO_STEPS: u32 = 100_000;
/// Surfaced decisions per game before calling it a draw. Players may hold
/// priority and repeat free activations, so a game could otherwise loop.
pub const MAX_DECISIONS: u32 = 5_000;
const AUTOPAY_NODE_CAP: usize = 256;

/// What kind of choice this is. Stable numbering: used in action keys and
/// as a model feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Kind {
    Pass = 0,
    PlayLand = 1,
    Cast = 2,
    Activate = 3,
    ManaAbility = 4,
    Plot = 5,
    Target = 6,
    FinishTargets = 7,
    CostTarget = 8,
    CastMode = 9,
    Kicker = 10,
    SpellMode = 11,
    EffectOption = 12,
    EffectTarget = 13,
    OptionalCost = 14,
    YesNo = 15,
    Discard = 16,
    Attack = 17,
    NoAttack = 18,
    Block = 19,
    NoBlock = 20,
    OrderTrigger = 21,
}

/// Semantic description of a choice, used both as the tree key and as model
/// input. `src` is the card definition acting/being chosen (u16::MAX = none).
/// `tgt` is the other object's semantic signature or a player (see
/// [`obj_sig`] / [`player_sig`]). `arg` is a small integer payload (mode
/// index, color, bool, ability index).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ActFeat {
    pub kind: Kind,
    pub src: u16,
    pub tgt: u32,
    pub arg: u16,
}

impl ActFeat {
    fn new(kind: Kind, src: u16, tgt: u32, arg: u16) -> Self {
        ActFeat { kind, src, tgt, arg }
    }

    pub fn key(&self) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for v in [self.kind as u64, self.src as u64, self.tgt as u64, self.arg as u64] {
            h ^= v;
            h = h.wrapping_mul(0x100000001b3);
            h ^= h >> 29;
        }
        h
    }
}

#[derive(Debug, Clone)]
enum Exec {
    Engine(Action),
    /// Tap these free mana sources, then take `then`.
    Autopay { taps: Vec<(ObjectId, ManaColor)>, then: Action },
    Attack(ObjectId, bool),
    Block(ObjectId, Option<ObjectId>),
    Discard(ObjectId),
    Order(usize),
}

#[derive(Debug, Clone)]
pub struct Choice {
    pub feat: ActFeat,
    pub key: u64,
    exec: Exec,
}

/// Multi-part decisions answered one piece at a time.
#[derive(Debug, Clone)]
enum Pending {
    None,
    Attack { eligible: Vec<ObjectId>, next: usize, chosen: Vec<ObjectId> },
    Block { blockers: Vec<(ObjectId, Vec<ObjectId>)>, next: usize, pairs: Vec<(ObjectId, ObjectId)> },
    Discard { count: u32, pool: Vec<ObjectId>, picked: Vec<ObjectId> },
    Order { sources: Vec<ObjectId>, remaining: Vec<usize>, placed: Vec<usize> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Win(PlayerId),
    Draw,
    /// Engine hit an unsupported mechanic or an internal error.
    Aborted,
}

#[derive(Debug, Clone)]
pub struct Game {
    pub state: GameState,
    pending: Pending,
    choices: Vec<Choice>,
    actor: PlayerId,
    outcome: Option<Outcome>,
    /// Number of surfaced (non-forced) decisions taken so far.
    pub decisions: u32,
    /// Why the game ended as [`Outcome::Aborted`], for diagnostics.
    pub abort_reason: Option<String>,
}

pub fn deck_names() -> Vec<&'static str> {
    RUNTIME_DECKS.iter().map(|d| d.id).collect()
}

/// Semantic signature of an object relative to `viewer`: card definition,
/// controller (self/opp), zone, tapped. Two objects with equal signatures are
/// interchangeable choices.
pub fn obj_sig(state: &GameState, id: ObjectId, viewer: PlayerId) -> u32 {
    let o = state.objects.get(id);
    let mine = (o.controller == viewer) as u32;
    let zone = zone_code(o.zone) as u32;
    (o.card_def as u32) | (mine << 16) | (zone << 17) | ((o.tapped as u32) << 21) | (1 << 23)
}

pub fn player_sig(p: PlayerId, viewer: PlayerId) -> u32 {
    (1 << 24) | ((p == viewer) as u32)
}

pub fn zone_code(z: Zone) -> u8 {
    match z {
        Zone::Library => 0,
        Zone::Hand => 1,
        Zone::Battlefield => 2,
        Zone::Graveyard => 3,
        Zone::Stack => 4,
        Zone::Exile => 5,
        _ => 6,
    }
}

fn target_sig(state: &GameState, t: Target, viewer: PlayerId) -> u32 {
    match t {
        Target::Object(id) => obj_sig(state, id, viewer),
        Target::Player(p) => player_sig(p, viewer),
    }
}

fn def_of(state: &GameState, id: ObjectId) -> u16 {
    state.objects.get(id).card_def
}

fn color_code(c: ManaColor) -> u16 {
    c.pool_index() as u16
}

fn shuffled(ids: &[u16], rng: &mut SplitMix64) -> Vec<u16> {
    let mut v = ids.to_vec();
    for i in (1..v.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
    v
}

impl Game {
    /// New game between two runtime decks (see [`deck_names`]). The starting
    /// player is chosen by `seed`.
    pub fn new(deck0: &str, deck1: &str, seed: u64) -> Result<Game, String> {
        let d0 = runtime_deck_by_id(deck0).ok_or_else(|| format!("unknown deck {deck0}"))?;
        let d1 = runtime_deck_by_id(deck1).ok_or_else(|| format!("unknown deck {deck1}"))?;
        let mut rng = SplitMix64::seed(seed);
        let lib0 = shuffled(d0.card_ids, &mut rng);
        let lib1 = shuffled(d1.card_ids, &mut rng);
        let first = if rng.next_u64() & 1 == 0 { PlayerId::P0 } else { PlayerId::P1 };
        let mut state = GameState::new_from_libraries_with_starting_player_v1(
            &lib0,
            &lib1,
            |id| CARD_DEFS[id as usize].name.to_string(),
            rng.next_u64(),
            first,
        );
        for _ in 0..7 {
            event::propose_and_commit(&mut state, ProposedEvent::draw(PlayerId::P0));
            event::propose_and_commit(&mut state, ProposedEvent::draw(PlayerId::P1));
        }
        let mut g = Game {
            state,
            pending: Pending::None,
            choices: Vec::new(),
            actor: PlayerId::P0,
            outcome: None,
            decisions: 0,
            abort_reason: None,
        };
        g.settle();
        Ok(g)
    }

    pub fn outcome(&self) -> Option<Outcome> {
        self.outcome
    }

    pub fn is_over(&self) -> bool {
        self.outcome.is_some()
    }

    /// Player who must choose among [`Self::choices`].
    pub fn to_act(&self) -> PlayerId {
        self.actor
    }

    pub fn choices(&self) -> &[Choice] {
        &self.choices
    }

    /// Apply choice `i` of [`Self::choices`], then advance through every
    /// forced decision to the next real one (or the end of the game).
    pub fn apply(&mut self, i: usize) {
        let c = self.choices[i].exec.clone();
        self.decisions += 1;
        self.exec(c);
        self.settle();
    }

    /// Resample every card `observer` cannot see (opponent's unknown hand
    /// cards, unknown library cards of both players), keeping card counts.
    /// Ported from mtg-kernel's `redeterminize_hidden_zones_v1`.
    pub fn determinize(&mut self, observer: PlayerId, seed: u64) {
        self.determinize_with(observer, seed, false);
    }

    /// [`Self::determinize`], optionally leaving the observer's own library
    /// alone. Needed while the observer is searching their library: the
    /// engine's pending effect refers to those exact cards (and they are
    /// revealed to the observer until the search shuffles them), so
    /// resampling them makes the engine halt.
    pub fn determinize_with(&mut self, observer: PlayerId, seed: u64, keep_own_library: bool) {
        let state = &mut self.state;
        let mut rng = SplitMix64::seed(seed);
        for owner in [PlayerId::P0, PlayerId::P1] {
            let mut slots = Vec::new();
            if owner != observer {
                for &id in &state.players[owner.index()].hand {
                    let zcc = state.objects.get(id).zone_change_count;
                    let known = state
                        .known_hand_cards(observer, owner)
                        .iter()
                        .any(|e| e.object == id && e.zone_change_count == zcc);
                    if !known {
                        slots.push(id);
                    }
                }
            }
            let library = if keep_own_library && owner == observer { &[][..] } else { &state.players[owner.index()].library[..] };
            for (pos, &id) in library.iter().enumerate() {
                let zcc = state.objects.get(id).zone_change_count;
                let known = state.known_library_cards(observer, owner).iter().any(|e| {
                    e.position as usize == pos && e.object == id && e.zone_change_count == zcc
                });
                if !known {
                    slots.push(id);
                }
            }
            let mut defs: Vec<u16> = slots.iter().map(|&id| state.objects.get(id).card_def).collect();
            for i in (1..defs.len()).rev() {
                let j = (rng.next_u64() % (i as u64 + 1)) as usize;
                defs.swap(i, j);
            }
            for (id, def) in slots.into_iter().zip(defs) {
                let o = state.objects.get_mut(id);
                if o.card_def != def {
                    o.card_def = def;
                    o.name = CARD_DEFS[def as usize].name.to_string();
                    o.v4 = ObjectStateV4::from_card_def(def);
                }
            }
        }
        // Choices can depend on hidden cards (e.g. the actor's own library is
        // never visible). Rebuild them for the new world.
        if self.outcome.is_none() && matches!(self.pending, Pending::None) {
            self.settle();
        }
    }

    // ------------------------------------------------------------ internals

    fn abort(&mut self, why: String) {
        if self.abort_reason.is_none() {
            self.abort_reason = Some(why);
        }
        self.finish(Outcome::Aborted);
    }

    fn finish(&mut self, o: Outcome) {
        self.outcome = Some(o);
        self.choices.clear();
        self.pending = Pending::None;
    }

    fn exec(&mut self, e: Exec) {
        match e {
            Exec::Engine(a) => {
                if let Err(e) = engine::step(&mut self.state, a.clone()) {
                    self.abort(format!("step {a:?}: {e}"));
                }
            }
            Exec::Autopay { taps, then } => {
                for (src, color) in taps {
                    if let Err(e) = tap(&mut self.state, src, color) {
                        self.abort(format!("autopay tap {src}: {e}"));
                        return;
                    }
                }
                if let Err(e) = engine::step(&mut self.state, then.clone()) {
                    self.abort(format!("autopay {then:?}: {e}"));
                }
            }
            Exec::Attack(id, yes) => {
                if let Pending::Attack { next, chosen, .. } = &mut self.pending {
                    if yes {
                        chosen.push(id);
                    }
                    *next += 1;
                }
            }
            Exec::Block(blocker, attacker) => {
                if let Pending::Block { next, pairs, .. } = &mut self.pending {
                    if let Some(a) = attacker {
                        pairs.push((blocker, a));
                    }
                    *next += 1;
                }
            }
            Exec::Discard(id) => {
                if let Pending::Discard { pool, picked, .. } = &mut self.pending {
                    pool.retain(|&x| x != id);
                    picked.push(id);
                }
            }
            Exec::Order(idx) => {
                if let Pending::Order { remaining, placed, .. } = &mut self.pending {
                    remaining.retain(|&x| x != idx);
                    placed.push(idx);
                }
            }
        }
    }

    /// Advance until a decision with two or more choices, or game end.
    fn settle(&mut self) {
        let mut guard = 0;
        while self.outcome.is_none() {
            guard += 1;
            if guard > MAX_AUTO_STEPS || self.state.turn > MAX_TURNS || self.decisions > MAX_DECISIONS {
                self.finish(Outcome::Draw);
                return;
            }
            if !matches!(self.pending, Pending::None) {
                if self.flush_pending() {
                    continue;
                }
                if self.choices.len() >= 2 {
                    return;
                }
                let only = self.choices[0].exec.clone();
                self.exec(only);
                continue;
            }
            let decision = engine::advance_until_decision(&mut self.state);
            match decision {
                Decision::GameOver { winner } => {
                    self.finish(winner.map_or(Outcome::Draw, Outcome::Win));
                    return;
                }
                Decision::Halted { mechanic, source } => {
                    self.abort(format!("halted {mechanic:?} source {}", CARD_DEFS[def_of(&self.state, source) as usize].name));
                    return;
                }
                _ => {}
            }
            let shown = format!("{decision:?}");
            self.build_choices(decision);
            if self.outcome.is_some() {
                return;
            }
            if !matches!(self.pending, Pending::None) {
                continue;
            }
            match self.choices.len() {
                0 => {
                    self.abort(format!("decision with no choices: {}", &shown[..shown.len().min(300)]));
                    return;
                }
                1 => {
                    let only = self.choices[0].exec.clone();
                    self.exec(only);
                }
                _ => return,
            }
        }
    }

    /// For a multi-part decision: either submit it to the engine (returns
    /// true) or build the choices for its next part (returns false).
    fn flush_pending(&mut self) -> bool {
        let viewer = self.actor;
        let st = &self.state;
        let mut out = Vec::new();
        let submit: Option<Action> = match &mut self.pending {
            Pending::None => return false,
            Pending::Attack { eligible, next, chosen } => {
                if *next >= eligible.len() {
                    Some(Action::DeclareAttackers(std::mem::take(chosen)))
                } else {
                    let id = eligible[*next];
                    let d = def_of(st, id);
                    let sig = obj_sig(st, id, viewer);
                    out.push(mk(ActFeat::new(Kind::Attack, d, sig, 0), Exec::Attack(id, true)));
                    out.push(mk(ActFeat::new(Kind::NoAttack, d, sig, 0), Exec::Attack(id, false)));
                    None
                }
            }
            Pending::Block { blockers, next, pairs } => {
                if *next >= blockers.len() {
                    Some(Action::DeclareBlockers(std::mem::take(pairs)))
                } else {
                    let (b, attackers) = &blockers[*next];
                    let d = def_of(st, *b);
                    out.push(mk(
                        ActFeat::new(Kind::NoBlock, d, obj_sig(st, *b, viewer), 0),
                        Exec::Block(*b, None),
                    ));
                    for &a in attackers {
                        push_unique(
                            &mut out,
                            mk(ActFeat::new(Kind::Block, d, obj_sig(st, a, viewer), 0), Exec::Block(*b, Some(a))),
                        );
                    }
                    None
                }
            }
            Pending::Discard { count, pool, picked } => {
                if picked.len() as u32 >= *count || pool.is_empty() {
                    Some(Action::Discard(std::mem::take(picked)))
                } else {
                    for &id in pool.iter() {
                        push_unique(
                            &mut out,
                            mk(ActFeat::new(Kind::Discard, def_of(st, id), 0, 0), Exec::Discard(id)),
                        );
                    }
                    None
                }
            }
            Pending::Order { sources, remaining, placed } => {
                if remaining.is_empty() {
                    Some(Action::OrderTriggers(std::mem::take(placed)))
                } else {
                    for &i in remaining.iter() {
                        let d = def_of(st, sources[i]);
                        push_unique(&mut out, mk(ActFeat::new(Kind::OrderTrigger, d, 0, 0), Exec::Order(i)));
                    }
                    None
                }
            }
        };
        match submit {
            Some(action) => {
                let fallback = match (&self.pending, &action) {
                    (Pending::Block { .. }, _) => Some(Action::DeclareBlockers(Vec::new())),
                    (Pending::Attack { eligible, .. }, Action::DeclareAttackers(a)) if !a.is_empty() => {
                        let _ = eligible;
                        Some(Action::DeclareAttackers(Vec::new()))
                    }
                    _ => None,
                };
                self.pending = Pending::None;
                if let Err(e) = engine::step(&mut self.state, action.clone()) {
                    // Combined declaration broke a rule the per-creature
                    // split can't see (menace, blocking limits): fall back to
                    // declaring nothing.
                    let ok = fallback.is_some_and(|f| engine::step(&mut self.state, f).is_ok());
                    if !ok {
                        self.abort(format!("submit {action:?}: {e}"));
                    }
                }
                true
            }
            None => {
                self.choices = out;
                false
            }
        }
    }

    fn build_choices(&mut self, decision: Decision) {
        let st = &self.state;
        let mut out: Vec<Choice> = Vec::new();
        match decision {
            Decision::CastSpellOrPass {
                player,
                castable_spells,
                mana_abilities,
                land_drops,
                activatable_abilities,
                plot_actions,
            } => {
                self.actor = player;
                out.push(mk(ActFeat::new(Kind::Pass, u16::MAX, 0, 0), Exec::Engine(Action::Pass)));
                for id in land_drops {
                    push_unique(&mut out, mk(ActFeat::new(Kind::PlayLand, def_of(st, id), 0, 0), Exec::Engine(Action::PlayLand(id))));
                }
                for &id in &castable_spells {
                    push_unique(&mut out, mk(cast_feat(st, id, player), Exec::Engine(Action::CastSpell(id))));
                }
                for &(id, idx) in &activatable_abilities {
                    push_unique(
                        &mut out,
                        mk(ActFeat::new(Kind::Activate, def_of(st, id), zone_code(st.objects.get(id).zone) as u32, idx as u16), Exec::Engine(Action::ActivateAbility(id, idx))),
                    );
                }
                for id in plot_actions {
                    push_unique(&mut out, mk(ActFeat::new(Kind::Plot, def_of(st, id), 0, 0), Exec::Engine(Action::PlotSpell(id))));
                }
                // Mana abilities with a real cost beyond tapping stay explicit.
                for &id in &mana_abilities {
                    if is_free_source(st, id) {
                        continue;
                    }
                    let d = def_of(st, id);
                    let colors = available_mana_ability_choices(player, id, st);
                    let cost_targets = mana_ability_cost_targets(player, id, st);
                    for &c in &colors {
                        if !cost_targets.is_empty() {
                            for &t in &cost_targets {
                                push_unique(
                                    &mut out,
                                    mk(ActFeat::new(Kind::ManaAbility, d, obj_sig(st, t, player), color_code(c)), Exec::Engine(Action::ActivateManaAbilityWithCostTarget(id, c, t))),
                                );
                            }
                        } else {
                            let a = if colors.len() == 1 { Action::ActivateManaAbility(id) } else { Action::ActivateManaAbilityChoice(id, c) };
                            push_unique(&mut out, mk(ActFeat::new(Kind::ManaAbility, d, 0, color_code(c)), Exec::Engine(a)));
                        }
                    }
                }
                // Spells and abilities reachable by tapping free sources.
                for (goal, taps) in autopay_goals(st, player, &mana_abilities, &castable_spells, &activatable_abilities) {
                    let feat = match goal {
                        Action::CastSpell(id) => cast_feat(st, id, player),
                        Action::ActivateAbility(id, idx) => ActFeat::new(Kind::Activate, def_of(st, id), zone_code(st.objects.get(id).zone) as u32, idx as u16),
                        _ => continue,
                    };
                    push_unique(&mut out, mk(feat, Exec::Autopay { taps, then: goal }));
                }
            }
            Decision::ChooseTargets { player, spell, legal_targets, can_finish, .. } => {
                self.actor = player;
                let d = def_of(st, spell);
                for t in legal_targets {
                    push_unique(&mut out, mk(ActFeat::new(Kind::Target, d, target_sig(st, t, player), 0), Exec::Engine(Action::ChooseTarget(t))));
                }
                if can_finish {
                    out.push(mk(ActFeat::new(Kind::FinishTargets, d, 0, 0), Exec::Engine(Action::FinishEffectSelection)));
                }
            }
            Decision::ChooseCostTargets { player, source, candidates, .. } => {
                self.actor = player;
                let d = def_of(st, source);
                for id in candidates {
                    push_unique(&mut out, mk(ActFeat::new(Kind::CostTarget, d, obj_sig(st, id, player), 0), Exec::Engine(Action::ChooseCostTarget(id))));
                }
            }
            Decision::ChooseCastMode { player, spell, options } => {
                self.actor = player;
                let d = def_of(st, spell);
                for m in options {
                    let arg = match m {
                        CastMode::Normal => 0,
                        CastMode::Alternative => 1,
                    };
                    out.push(mk(ActFeat::new(Kind::CastMode, d, 0, arg), Exec::Engine(Action::ChooseCastMode(m))));
                }
            }
            Decision::ChooseKicker { player, spell } => {
                self.actor = player;
                let d = def_of(st, spell);
                for b in [false, true] {
                    out.push(mk(ActFeat::new(Kind::Kicker, d, 0, b as u16), Exec::Engine(Action::ChooseKicker(b))));
                }
            }
            Decision::ChooseSpellMode { player, spell, legal_modes, .. } => {
                self.actor = player;
                let d = def_of(st, spell);
                for m in legal_modes {
                    out.push(mk(ActFeat::new(Kind::SpellMode, d, 0, m as u16), Exec::Engine(Action::ChooseSpellMode(m))));
                }
            }
            Decision::ChooseEffectOption { player, source, option_count } => {
                self.actor = player;
                let d = def_of(st, source);
                for i in 0..option_count {
                    out.push(mk(ActFeat::new(Kind::EffectOption, d, 0, i), Exec::Engine(Action::ChooseEffectOption(i))));
                }
            }
            Decision::ChooseEffectTargets { player, source, legal_targets, can_finish, .. } => {
                self.actor = player;
                let d = def_of(st, source);
                for t in legal_targets {
                    push_unique(&mut out, mk(ActFeat::new(Kind::EffectTarget, d, target_sig(st, t, player), 0), Exec::Engine(Action::ChooseEffectTarget(t))));
                }
                if can_finish {
                    out.push(mk(ActFeat::new(Kind::FinishTargets, d, 0, 0), Exec::Engine(Action::FinishEffectSelection)));
                }
            }
            Decision::ChooseOptionalCost { player, discard_payable, sacrifice_payable } => {
                self.actor = player;
                out.push(mk(ActFeat::new(Kind::OptionalCost, u16::MAX, 0, 0), Exec::Engine(Action::ChooseOptionalCost(OptionalCostChoice::Decline))));
                if discard_payable {
                    out.push(mk(ActFeat::new(Kind::OptionalCost, u16::MAX, 0, 1), Exec::Engine(Action::ChooseOptionalCost(OptionalCostChoice::Discard))));
                }
                if sacrifice_payable {
                    out.push(mk(ActFeat::new(Kind::OptionalCost, u16::MAX, 0, 2), Exec::Engine(Action::ChooseOptionalCost(OptionalCostChoice::SacrificeLand))));
                }
            }
            Decision::ChooseSpellCopyPayment { player, spell } => {
                self.actor = player;
                yes_no(&mut out, def_of(st, spell), 1, Action::ChooseSpellCopyPayment);
            }
            Decision::ChooseSpellCopyRetarget { player, copy } => {
                self.actor = player;
                yes_no(&mut out, def_of(st, copy), 2, Action::ChooseSpellCopyRetarget);
            }
            Decision::ChooseMadnessCast { player, card } => {
                self.actor = player;
                yes_no(&mut out, def_of(st, card), 3, Action::ChooseMadnessCast);
            }
            Decision::ChooseEffectBoolean { player, source, .. } => {
                self.actor = player;
                yes_no(&mut out, def_of(st, source), 4, Action::ChooseEffectBoolean);
            }
            Decision::Discard { player, count, choices } => {
                self.actor = player;
                self.pending = Pending::Discard { count, pool: choices, picked: Vec::new() };
            }
            Decision::DeclareAttackers { player, eligible } => {
                self.actor = player;
                // Goaded creatures must attack: pre-commit them.
                let chosen = engine::required_goaded_attackers(st, &eligible);
                let eligible = eligible.into_iter().filter(|id| !chosen.contains(id)).collect();
                self.pending = Pending::Attack { eligible, next: 0, chosen };
            }
            Decision::DeclareBlockers { player, legal_blockers, .. } => {
                self.actor = player;
                // Invert (attacker -> blockers) into (blocker -> attackers).
                let mut inv: Vec<(ObjectId, Vec<ObjectId>)> = Vec::new();
                for (a, bs) in legal_blockers {
                    for b in bs {
                        match inv.iter_mut().find(|(x, _)| *x == b) {
                            Some((_, v)) => v.push(a),
                            None => inv.push((b, vec![a])),
                        }
                    }
                }
                self.pending = Pending::Block { blockers: inv, next: 0, pairs: Vec::new() };
            }
            Decision::OrderTriggers { player, pending } => {
                self.actor = player;
                let sources: Vec<ObjectId> = pending.iter().map(|t| t.source).collect();
                let remaining = (0..sources.len()).collect();
                self.pending = Pending::Order { sources, remaining, placed: Vec::new() };
            }
            Decision::GameOver { .. } | Decision::Halted { .. } => unreachable!(),
        }
        self.choices = out;
    }
}

fn mk(feat: ActFeat, exec: Exec) -> Choice {
    Choice { key: feat.key(), feat, exec }
}

/// Keep only the first of several semantically identical choices (two
/// copies of the same card in hand, two identical untapped tokens...).
fn push_unique(out: &mut Vec<Choice>, c: Choice) {
    if !out.iter().any(|x| x.key == c.key) {
        out.push(c);
    }
}

fn yes_no(out: &mut Vec<Choice>, d: u16, which: u16, f: fn(bool) -> Action) {
    for b in [false, true] {
        out.push(mk(ActFeat::new(Kind::YesNo, d, which as u32, b as u16), Exec::Engine(f(b))));
    }
}

fn cast_feat(st: &GameState, id: ObjectId, _player: PlayerId) -> ActFeat {
    ActFeat::new(Kind::Cast, def_of(st, id), zone_code(st.objects.get(id).zone) as u32, 0)
}

/// A mana source whose every mana ability costs only `{T}`.
fn is_free_source(st: &GameState, id: ObjectId) -> bool {
    let def = &CARD_DEFS[def_of(st, id) as usize];
    let primary_free = def.mana_ability_def.is_none_or(|m| m.cost == ManaAbilityCostDef::TapSelf);
    primary_free
        && def
            .additional_mana_abilities
            .iter()
            .all(|a| a.ability.cost == ManaAbilityCostDef::TapSelf && a.mana_cost.pips.is_empty() && a.mana_cost.generic == 0)
}

fn tap(st: &mut GameState, id: ObjectId, color: ManaColor) -> Result<(), String> {
    let p = st.priority_player;
    let n = available_mana_ability_choices(p, id, st).len();
    let a = if n <= 1 { Action::ActivateManaAbility(id) } else { Action::ActivateManaAbilityChoice(id, color) };
    engine::step(st, a)
}

/// Find, for every spell/ability not castable from the current pool, a
/// minimal set of free-source taps that makes it castable.
///
/// Works on one scratch copy of the state: taps are simulated by writing the
/// mana straight into the pool (and marking the sources tapped), and the
/// engine's read-only castability checks are asked at each step. Sources are
/// grouped into interchangeable types (same card, same colors); the search is
/// breadth-first over multisets of (type, color), so the first payment found
/// uses the fewest sources, and single-color sources are preferred.
fn autopay_goals(
    st: &GameState,
    player: PlayerId,
    mana_abilities: &[ObjectId],
    castable_now: &[ObjectId],
    activatable_now: &[(ObjectId, u8)],
) -> Vec<(Action, Vec<(ObjectId, ManaColor)>)> {
    let ps = &st.players[player.index()];
    // Anything that could possibly become castable/activatable with more mana?
    let maybe_more = ps.hand.iter().any(|&id| !castable_now.contains(&id) && !CARD_DEFS[def_of(st, id) as usize].is_land)
        || !ps.graveyard.is_empty()
        || st.exile.iter().any(|&id| st.objects.get(id).owner == player)
        || ps.battlefield.iter().any(|&id| {
            let n = CARD_DEFS[def_of(st, id) as usize].activated_abilities.len();
            (0..n).any(|i| !activatable_now.contains(&(id, i as u8)))
        });
    if !maybe_more {
        return Vec::new();
    }

    // Interchangeable source types, single-color ones first.
    struct SourceType {
        ids: Vec<ObjectId>,
        colors: Vec<ManaColor>,
    }
    let mut types: Vec<(u16, SourceType)> = Vec::new();
    for &id in mana_abilities {
        if !is_free_source(st, id) {
            continue;
        }
        let colors = available_mana_ability_choices(player, id, st);
        if colors.is_empty() {
            continue;
        }
        let d = def_of(st, id);
        match types.iter_mut().find(|(td, t)| *td == d && t.colors == colors) {
            Some((_, t)) => t.ids.push(id),
            None => types.push((d, SourceType { ids: vec![id], colors })),
        }
    }
    if types.is_empty() {
        return Vec::new();
    }
    types.sort_by_key(|(d, t)| (t.colors.len(), *d));
    let types: Vec<SourceType> = types.into_iter().map(|(_, t)| t).collect();
    // Flattened (type, color) options.
    let opts: Vec<(usize, ManaColor)> = types
        .iter()
        .enumerate()
        .flat_map(|(ti, t)| t.colors.iter().map(move |&c| (ti, c)))
        .collect();

    let mut scratch = st.clone();
    let base_pool = ps.mana_pool;
    let pi = player.index();

    // Over-approximation: every source adds every color it can make.
    for t in &types {
        for &c in &t.colors {
            scratch.players[pi].mana_pool[c.pool_index()] += t.ids.len() as u8;
        }
    }
    let mut goals: Vec<Action> = Vec::new();
    let mut seen = Vec::new();
    for id in engine::castable_spells(player, &scratch) {
        let k = (0u8, def_of(st, id), zone_code(st.objects.get(id).zone), 0u8);
        if !castable_now.contains(&id) && !seen.contains(&k) {
            seen.push(k);
            goals.push(Action::CastSpell(id));
        }
    }
    for (id, idx) in engine::available_activatable_abilities(player, &scratch) {
        let k = (1u8, def_of(st, id), zone_code(st.objects.get(id).zone), idx);
        if !activatable_now.contains(&(id, idx)) && !seen.contains(&k) {
            seen.push(k);
            goals.push(Action::ActivateAbility(id, idx));
        }
    }
    scratch.players[pi].mana_pool = base_pool;
    if goals.is_empty() {
        return Vec::new();
    }

    // Breadth-first over non-decreasing sequences of option indices.
    let mut found: Vec<(Action, Vec<(ObjectId, ManaColor)>)> = Vec::new();
    let mut frontier: Vec<Vec<usize>> = vec![Vec::new()];
    let mut nodes = 0usize;
    let mut used = vec![0usize; types.len()];
    'search: while !frontier.is_empty() {
        let mut next = Vec::new();
        for node in &frontier {
            let start = node.last().copied().unwrap_or(0);
            for oi in start..opts.len() {
                let (ti, _) = opts[oi];
                let in_use = node.iter().filter(|&&o| opts[o].0 == ti).count();
                if in_use >= types[ti].ids.len() {
                    continue;
                }
                nodes += 1;
                if nodes > AUTOPAY_NODE_CAP {
                    break 'search;
                }
                let mut seq = node.clone();
                seq.push(oi);
                // Apply: pool = base + chosen colors; chosen sources tapped.
                let mut pool = base_pool;
                used.iter_mut().for_each(|u| *u = 0);
                let mut taps = Vec::with_capacity(seq.len());
                for &o in &seq {
                    let (t, c) = opts[o];
                    pool[c.pool_index()] += 1;
                    taps.push((types[t].ids[used[t]], c));
                    used[t] += 1;
                }
                scratch.players[pi].mana_pool = pool;
                for &(id, _) in &taps {
                    scratch.objects.get_mut(id).tapped = true;
                }
                let cast = engine::castable_spells(player, &scratch);
                let act = engine::available_activatable_abilities(player, &scratch);
                for &(id, _) in &taps {
                    scratch.objects.get_mut(id).tapped = false;
                }
                for g in &goals {
                    if found.iter().any(|(f, _)| f == g) {
                        continue;
                    }
                    let ok = match g {
                        Action::CastSpell(id) => cast.contains(id),
                        Action::ActivateAbility(id, idx) => act.contains(&(*id, *idx)),
                        _ => false,
                    };
                    if ok {
                        found.push((g.clone(), taps.clone()));
                    }
                }
                if found.len() == goals.len() {
                    break 'search;
                }
                next.push(seq);
            }
        }
        frontier = next;
    }
    found
}

impl Game {
    /// Random playout to the end; returns the outcome.
    pub fn playout(&mut self, rng: &mut SplitMix64) -> Outcome {
        while !self.is_over() {
            let n = self.choices.len();
            self.apply((rng.next_u64() % n as u64) as usize);
        }
        self.outcome.unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_games_finish_for_every_deck_pair() {
        let decks = deck_names();
        let mut rng = SplitMix64::seed(7);
        let mut aborted = 0;
        let mut total = 0;
        for a in &decks {
            for b in &decks {
                for s in 0..3 {
                    let mut g = Game::new(a, b, total * 31 + s).unwrap();
                    if g.playout(&mut rng) == Outcome::Aborted {
                        eprintln!("{a} vs {b}: {}", g.abort_reason.as_deref().unwrap_or("?"));
                        aborted += 1;
                    }
                    total += 1;
                }
            }
        }
        assert!(aborted * 50 <= total, "{aborted}/{total} games aborted");
    }

    #[test]
    fn same_seed_same_game() {
        let run = |seed| {
            let mut g = Game::new("Burn", "Rally", seed).unwrap();
            let mut rng = SplitMix64::seed(99);
            g.playout(&mut rng);
            (g.state.state_hash(), g.decisions)
        };
        assert_eq!(run(5), run(5));
    }

    #[test]
    fn determinize_keeps_public_state_and_counts() {
        let mut g = Game::new("Faeries", "Elves", 3).unwrap();
        let mut rng = SplitMix64::seed(1);
        for _ in 0..40 {
            if g.is_over() {
                break;
            }
            let n = g.choices().len();
            g.apply((rng.next_u64() % n as u64) as usize);
        }
        let me = g.to_act();
        let before = g.clone();
        g.determinize(me, 12345);
        let st0 = &before.state;
        let st1 = &g.state;
        for p in [PlayerId::P0, PlayerId::P1] {
            let (a, b) = (&st0.players[p.index()], &st1.players[p.index()]);
            assert_eq!(a.hand.len(), b.hand.len());
            assert_eq!(a.library.len(), b.library.len());
            for &id in a.battlefield.iter().chain(&a.graveyard) {
                assert_eq!(st0.objects.get(id).card_def, st1.objects.get(id).card_def);
            }
        }
        // Observer's own hand never changes.
        for &id in &st0.players[me.index()].hand {
            assert_eq!(st0.objects.get(id).card_def, st1.objects.get(id).card_def);
        }
        // Multiset of hidden cards per owner is preserved.
        for p in [PlayerId::P0, PlayerId::P1] {
            let ms = |s: &GameState| {
                let ps = &s.players[p.index()];
                let mut v: Vec<u16> = ps.hand.iter().chain(&ps.library).map(|&id| s.objects.get(id).card_def).collect();
                v.sort();
                v
            };
            assert_eq!(ms(st0), ms(st1));
        }
    }

    #[test]
    fn autopay_lets_burn_cast_from_lands() {
        // Burn should cast spells in random play; if auto-pay were broken,
        // no Cast choices would ever appear because the pool starts empty.
        let mut rng = SplitMix64::seed(3);
        let mut casts = 0;
        for seed in 0..20 {
            let mut g = Game::new("Burn", "Burn", seed).unwrap();
            while !g.is_over() {
                casts += g.choices().iter().filter(|c| c.feat.kind == Kind::Cast).count();
                let n = g.choices().len();
                g.apply((rng.next_u64() % n as u64) as usize);
            }
        }
        assert!(casts > 100, "only {casts} cast choices seen");
    }
}
