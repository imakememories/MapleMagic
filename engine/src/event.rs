//! propose -> replace/prevent -> commit.
//!
//! `effect::execute` never mutates `GameState` directly: every leaf op
//! builds a `ProposedEvent` and calls `propose_and_commit`, which runs the
//! replacement/prevention pass (`apply_replacements`) and then, if
//! anything survived, `commit`. `commit` is the *only* function that
//! mutates `GameState` in response to a game event, and it appends the
//! resulting `CommittedEvent` to `state.engine.event_log` for
//! `trigger::collect_and_process` to drain after the resolution finishes.

use crate::ids::{ObjectId, PlayerId, StackItemId};
use crate::mana::ManaColor;
use crate::state::{GameState, PaidCostRefV4, StackItemKind, Target, Zone};
use serde::{Deserialize, Serialize};

pub type ReplacementId = u32;

/// Deterministic placement within the destination library for a zone-change
/// proposal. Ordinary library moves retain `Top`; effects such as Deem
/// Inferior select one of the appended positional variants without bypassing
/// the replace/commit/event-log pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LibraryPlacement {
    Top,
    SecondFromTop,
    Bottom,
}

/// Which observers learn the identity of a card inserted into a library at
/// a publicly determined position. Position shifts are always public and
/// exact; this flag controls only whether the inserted identity is learned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LibraryInsertVisibility {
    Hidden,
    Owner,
    Public,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReplacementEffectKind {
    /// Prevent the next `remaining` damage that would be dealt to `target`.
    /// Synthetic -- no pool card grants this yet -- but proves the
    /// replacement pipeline shape end-to-end; see
    /// `tests::prevention_shield_absorbs_then_expires`.
    PreventNextDamage { target: Target, remaining: i32 },
    /// Prevent all damage that sources sharing `color` would deal during
    /// the turn in which this replacement was installed.
    PreventDamageFromColorUntilEndOfTurn {
        color: ManaColor,
        turn: u32,
        active_player: PlayerId,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ActiveReplacement {
    pub id: ReplacementId,
    pub source: ObjectId,
    pub kind: ReplacementEffectKind,
}

// ---------------------------------------------------------------- proposed

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DamageProposed {
    pub source: ObjectId,
    pub target: Target,
    pub amount: i32,
    pub touched_by: Vec<ReplacementId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZoneChangeProposed {
    pub object: ObjectId,
    pub to_zone: Zone,
    /// Meaningful only when `to_zone == Library`; other destinations retain
    /// the constructor-default `Top` marker.
    pub library_placement: LibraryPlacement,
    /// Preserve an already-known library identity if this move puts the
    /// card into a hidden zone. Ordinary library-to-hand moves remain
    /// secret; public reveal effects opt in after populating the existing
    /// perspective-scoped library-knowledge table.
    pub preserve_known_identity: bool,
    /// This is a private top-of-library insertion whose owner knows the
    /// inserted identity. Other observers learn no new identity, while their
    /// still-valid prior library position facts shift through the insertion.
    pub library_insert_visibility: LibraryInsertVisibility,
    /// Forces a permanent moved to the battlefield by this effect to enter
    /// tapped, independently of its own static entry rules.
    pub force_battlefield_tapped: bool,
    /// Optional transformed battlefield face. Ordinary zone changes retain
    /// `None`, which resets the object to its front face.
    pub battlefield_face_index: Option<u8>,
    /// Optional controller for a battlefield return whose effect says
    /// "under your control" rather than the ordinary owner-controlled rule.
    pub battlefield_controller: Option<PlayerId>,
    pub touched_by: Vec<ReplacementId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LifeLossProposed {
    pub player: PlayerId,
    pub amount: i32,
    pub touched_by: Vec<ReplacementId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LifeGainProposed {
    pub player: PlayerId,
    pub amount: i32,
    pub touched_by: Vec<ReplacementId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DrawProposed {
    pub player: PlayerId,
    pub touched_by: Vec<ReplacementId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TapProposed {
    pub object: ObjectId,
    pub touched_by: Vec<ReplacementId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManaAddProposed {
    pub player: PlayerId,
    pub colors: Vec<ManaColor>,
    pub touched_by: Vec<ReplacementId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreateTokenProposed {
    pub token_def: u16,
    pub controller: PlayerId,
    pub touched_by: Vec<ReplacementId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProposedEvent {
    Damage(DamageProposed),
    ZoneChange(ZoneChangeProposed),
    LifeLoss(LifeLossProposed),
    LifeGain(LifeGainProposed),
    Draw(DrawProposed),
    Tap(TapProposed),
    ManaAdd(ManaAddProposed),
    CreateToken(CreateTokenProposed),
}

impl ProposedEvent {
    pub fn damage(source: ObjectId, target: Target, amount: i32) -> ProposedEvent {
        ProposedEvent::Damage(DamageProposed {
            source,
            target,
            amount,
            touched_by: Vec::new(),
        })
    }
    pub fn zone_change(object: ObjectId, to_zone: Zone) -> ProposedEvent {
        ProposedEvent::ZoneChange(ZoneChangeProposed {
            object,
            to_zone,
            library_placement: LibraryPlacement::Top,
            preserve_known_identity: false,
            library_insert_visibility: LibraryInsertVisibility::Hidden,
            force_battlefield_tapped: false,
            battlefield_face_index: None,
            battlefield_controller: None,
            touched_by: Vec::new(),
        })
    }
    pub fn zone_change_to_battlefield_tapped(object: ObjectId) -> ProposedEvent {
        ProposedEvent::ZoneChange(ZoneChangeProposed {
            object,
            to_zone: Zone::Battlefield,
            library_placement: LibraryPlacement::Top,
            preserve_known_identity: false,
            library_insert_visibility: LibraryInsertVisibility::Hidden,
            force_battlefield_tapped: true,
            battlefield_face_index: None,
            battlefield_controller: None,
            touched_by: Vec::new(),
        })
    }
    pub fn zone_change_preserving_known_identity(object: ObjectId, to_zone: Zone) -> ProposedEvent {
        ProposedEvent::ZoneChange(ZoneChangeProposed {
            object,
            to_zone,
            library_placement: LibraryPlacement::Top,
            preserve_known_identity: true,
            library_insert_visibility: LibraryInsertVisibility::Hidden,
            force_battlefield_tapped: false,
            battlefield_face_index: None,
            battlefield_controller: None,
            touched_by: Vec::new(),
        })
    }
    /// Moves one privately selected card to the top of its owner's library.
    /// The move remains a normal replaceable/logged zone change; this flag
    /// only carries the visibility needed for exact library knowledge.
    pub fn private_top_library_insert(object: ObjectId) -> ProposedEvent {
        ProposedEvent::ZoneChange(ZoneChangeProposed {
            object,
            to_zone: Zone::Library,
            library_placement: LibraryPlacement::Top,
            preserve_known_identity: false,
            library_insert_visibility: LibraryInsertVisibility::Owner,
            force_battlefield_tapped: false,
            battlefield_face_index: None,
            battlefield_controller: None,
            touched_by: Vec::new(),
        })
    }
    /// Moves one publicly identified object to an exact position in its
    /// owner's library. Both observers retain that public identity at its
    /// new incarnation while every pre-existing known position shifts
    /// exactly around the insertion.
    pub fn public_library_insert(object: ObjectId, placement: LibraryPlacement) -> ProposedEvent {
        ProposedEvent::ZoneChange(ZoneChangeProposed {
            object,
            to_zone: Zone::Library,
            library_placement: placement,
            preserve_known_identity: false,
            library_insert_visibility: LibraryInsertVisibility::Public,
            force_battlefield_tapped: false,
            battlefield_face_index: None,
            battlefield_controller: None,
            touched_by: Vec::new(),
        })
    }
    pub fn transformed_battlefield_return(
        object: ObjectId,
        face_index: u8,
        controller: PlayerId,
    ) -> ProposedEvent {
        ProposedEvent::ZoneChange(ZoneChangeProposed {
            object,
            to_zone: Zone::Battlefield,
            library_placement: LibraryPlacement::Top,
            preserve_known_identity: false,
            library_insert_visibility: LibraryInsertVisibility::Public,
            force_battlefield_tapped: false,
            battlefield_face_index: Some(face_index),
            battlefield_controller: Some(controller),
            touched_by: Vec::new(),
        })
    }
    pub fn life_loss(player: PlayerId, amount: i32) -> ProposedEvent {
        ProposedEvent::LifeLoss(LifeLossProposed {
            player,
            amount,
            touched_by: Vec::new(),
        })
    }
    pub fn life_gain(player: PlayerId, amount: i32) -> ProposedEvent {
        ProposedEvent::LifeGain(LifeGainProposed {
            player,
            amount,
            touched_by: Vec::new(),
        })
    }
    pub fn draw(player: PlayerId) -> ProposedEvent {
        ProposedEvent::Draw(DrawProposed {
            player,
            touched_by: Vec::new(),
        })
    }
    pub fn tap(object: ObjectId) -> ProposedEvent {
        ProposedEvent::Tap(TapProposed {
            object,
            touched_by: Vec::new(),
        })
    }
    pub fn mana_add(player: PlayerId, colors: Vec<ManaColor>) -> ProposedEvent {
        ProposedEvent::ManaAdd(ManaAddProposed {
            player,
            colors,
            touched_by: Vec::new(),
        })
    }
    pub fn create_token(token_def: u16, controller: PlayerId) -> ProposedEvent {
        ProposedEvent::CreateToken(CreateTokenProposed {
            token_def,
            controller,
            touched_by: Vec::new(),
        })
    }

    fn touched_by(&self) -> &[ReplacementId] {
        match self {
            ProposedEvent::Damage(e) => &e.touched_by,
            ProposedEvent::ZoneChange(e) => &e.touched_by,
            ProposedEvent::LifeLoss(e) => &e.touched_by,
            ProposedEvent::LifeGain(e) => &e.touched_by,
            ProposedEvent::Draw(e) => &e.touched_by,
            ProposedEvent::Tap(e) => &e.touched_by,
            ProposedEvent::ManaAdd(e) => &e.touched_by,
            ProposedEvent::CreateToken(e) => &e.touched_by,
        }
    }

    fn mark_touched(&mut self, id: ReplacementId) {
        let v = match self {
            ProposedEvent::Damage(e) => &mut e.touched_by,
            ProposedEvent::ZoneChange(e) => &mut e.touched_by,
            ProposedEvent::LifeLoss(e) => &mut e.touched_by,
            ProposedEvent::LifeGain(e) => &mut e.touched_by,
            ProposedEvent::Draw(e) => &mut e.touched_by,
            ProposedEvent::Tap(e) => &mut e.touched_by,
            ProposedEvent::ManaAdd(e) => &mut e.touched_by,
            ProposedEvent::CreateToken(e) => &mut e.touched_by,
        };
        v.push(id);
    }
}

// --------------------------------------------------------------- committed

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CommittedEvent {
    Damage {
        source: ObjectId,
        target: Target,
        amount: i32,
    },
    ZoneChange {
        object: ObjectId,
        from: Zone,
        to: Zone,
        /// Last-known controller immediately before the move. Zone-change
        /// bookkeeping resets control to the owner, but leave/dies triggers
        /// remain controlled by the player who controlled the permanent.
        controller_before: PlayerId,
    },
    LifeLoss {
        player: PlayerId,
        amount: i32,
    },
    LifeGain {
        player: PlayerId,
        amount: i32,
    },
    /// `object` is `None` when the draw was attempted against an empty
    /// library; SBA picks that up as a loss condition (704.5c).
    Draw {
        player: PlayerId,
        object: Option<ObjectId>,
    },
    Tap {
        object: ObjectId,
    },
    ManaAdded {
        player: PlayerId,
        colors: Vec<ManaColor>,
    },
    CreateToken {
        object: ObjectId,
        token_def: u16,
        controller: PlayerId,
    },
    /// Logged by `engine::finalize_cast` the moment a spell is placed on
    /// the stack (not routed through `propose_and_commit`: casting a spell
    /// is an engine action with no replaceable "cast" event in this
    /// increment's scope, same rationale as the Hand->Stack zone move
    /// itself -- see `commit_zone_change`'s doc). Exists purely so
    /// `trigger::TriggerCondition::CastInstantOrSorcery` (Guttersnipe) has
    /// something to match against; it is still appended to both
    /// `event_log` (drained by `trigger::collect_and_process`) and
    /// `event_history` for consistency with every other committed event.
    SpellCast {
        spell: ObjectId,
        controller: PlayerId,
    },
    /// A completed cast, activation, or copy-target choice made this exact
    /// permanent incarnation a target. This is nonreplaceable bookkeeping
    /// consumed by reusable Ward trigger matching.
    Targeted {
        target: ObjectId,
        target_zone_change_count: u32,
        targeting_stack_item: StackItemId,
        targeting_controller: PlayerId,
    },
    /// A permanent was sacrificed by its current controller. The effective
    /// subtype set is captured as last-known information before the ensuing
    /// zone change resets the object incarnation.
    Sacrificed {
        object: ObjectId,
        controller_before: PlayerId,
        effective_subtype_ids_before: Vec<u16>,
    },
    /// Nonreplaceable marker emitted after a combat-damage batch only when
    /// positive damage to a player survived prevention. The exact source
    /// incarnation prevents a later zone-change generation from inheriting
    /// the combat trigger.
    CombatDamageToPlayer {
        source: ObjectId,
        source_zone_change_count: u32,
        player: PlayerId,
        amount: i32,
    },
    /// A Saga received this exact lore counter and its corresponding chapter
    /// ability triggered. Captures the source incarnation at counter time so
    /// later zone changes cannot relabel the chapter ability.
    SagaChapter {
        source: ObjectId,
        source_zone_change_count: u32,
        controller: PlayerId,
        chapter: u8,
    },
    /// Transient cast-provenance marker consumed from `event_log` by the
    /// trigger collection immediately following this resolution. It is not
    /// appended to permanent `event_history`; the spell stack item and its
    /// paid-cost references remain the durable provenance.
    OptionalAdditionalCostPaid {
        source: ObjectId,
        kind: crate::card_def::OptionalAdditionalCostDef,
        paid_cost_refs: Vec<PaidCostRefV4>,
    },
    /// Nonreplaceable marker for one global Initiative or Undercity trigger.
    /// `history_index` is self-authenticating against the permanent event
    /// history before the resulting ability may be placed on the stack.
    InitiativeTrigger {
        binding: crate::state::InitiativeTriggerBindingV1,
    },
}

/// Runs the replace/prevent pass to a fixed point: repeatedly finds an
/// active replacement that applies to `event` and hasn't already touched it
/// (loop-prevention via `touched_by`), applies it, and marks it touched.
/// Returns `None` if the event ends up fully prevented.
pub fn apply_replacements(
    state: &mut GameState,
    mut proposed: ProposedEvent,
) -> Option<ProposedEvent> {
    let damage_cannot_be_prevented = matches!(&proposed, ProposedEvent::Damage(_))
        && state.engine.until_end_of_turn.iter().any(|effect| {
            matches!(
                effect,
                crate::engine::UntilEndOfTurnEffect::DamageCannotBePrevented { .. }
            )
        });
    if let ProposedEvent::Damage(damage) = &proposed {
        if !damage_cannot_be_prevented
            && crate::engine::damage_is_prevented_by_protection(state, damage.source, damage.target)
        {
            return None;
        }
    }
    loop {
        let damage_cannot_be_prevented = matches!(&proposed, ProposedEvent::Damage(_))
            && state.engine.until_end_of_turn.iter().any(|effect| {
                matches!(
                    effect,
                    crate::engine::UntilEndOfTurnEffect::DamageCannotBePrevented { .. }
                )
            });
        let hit = state
            .engine
            .active_replacements
            .iter()
            .find(|r| {
                !(proposed.touched_by().contains(&r.id)
                    || damage_cannot_be_prevented
                        && matches!(
                            &r.kind,
                            ReplacementEffectKind::PreventNextDamage { .. }
                                | ReplacementEffectKind::PreventDamageFromColorUntilEndOfTurn { .. }
                        ))
                    && replacement_applies(r, &proposed, state)
            })
            .cloned();

        let Some(repl) = hit else {
            return Some(proposed);
        };

        proposed.mark_touched(repl.id);
        match replacement_apply(&repl, proposed, state) {
            Some(rewritten) => proposed = rewritten,
            None => return None,
        }
    }
}

fn replacement_applies(
    repl: &ActiveReplacement,
    proposed: &ProposedEvent,
    state: &GameState,
) -> bool {
    match (&repl.kind, proposed) {
        (
            ReplacementEffectKind::PreventNextDamage { target, remaining },
            ProposedEvent::Damage(d),
        ) => *remaining > 0 && d.target == *target,
        (
            ReplacementEffectKind::PreventDamageFromColorUntilEndOfTurn {
                color,
                turn,
                active_player,
            },
            ProposedEvent::Damage(d),
        ) => {
            *turn == state.turn
                && *active_player == state.active_player
                && crate::engine::object_color_mask(state, d.source)
                    & crate::card_def::mana_color_mask(*color)
                    != 0
        }
        _ => false,
    }
}

/// Applies `repl` to `proposed`, mutating the replacement's own bookkeeping
/// in `state.engine.active_replacements` (e.g. decrementing/expiring a
/// prevention shield's remaining count) as a side effect.
fn replacement_apply(
    repl: &ActiveReplacement,
    proposed: ProposedEvent,
    state: &mut GameState,
) -> Option<ProposedEvent> {
    match (&repl.kind, proposed) {
        (
            ReplacementEffectKind::PreventNextDamage { remaining, .. },
            ProposedEvent::Damage(mut d),
        ) => {
            let prevented = (*remaining).min(d.amount);
            d.amount -= prevented;

            if let Some(slot) = state
                .engine
                .active_replacements
                .iter_mut()
                .find(|r| r.id == repl.id)
            {
                // Only variant today, but matched explicitly (rather than
                // destructured directly) so this stays a real pattern match
                // -- and a compile error, not a silent no-op -- the moment
                // a second `ReplacementEffectKind` is added.
                #[allow(irrefutable_let_patterns)]
                if let ReplacementEffectKind::PreventNextDamage { remaining, .. } = &mut slot.kind {
                    *remaining -= prevented;
                }
            }
            state.engine.active_replacements.retain(|r| {
                !matches!(&r.kind, ReplacementEffectKind::PreventNextDamage { remaining, .. } if *remaining <= 0)
            });

            if d.amount <= 0 {
                None
            } else {
                Some(ProposedEvent::Damage(d))
            }
        }
        (
            ReplacementEffectKind::PreventDamageFromColorUntilEndOfTurn { .. },
            ProposedEvent::Damage(_),
        ) => None,
        (_, other) => Some(other),
    }
}

/// Installs a W/U/B/R/G prevention replacement through the same monotonic
/// identity allocator used by all replacement effects.
pub fn install_color_damage_prevention(
    state: &mut GameState,
    source: ObjectId,
    color: ManaColor,
) -> Result<ReplacementId, String> {
    if color == ManaColor::C {
        return Err("colorless is not a legal Prismatic Strands choice".to_string());
    }
    let id = state
        .engine
        .next_replacement_id
        .checked_add(1)
        .ok_or("replacement identity space exhausted")?;
    state.engine.next_replacement_id = id;
    state.engine.active_replacements.push(ActiveReplacement {
        id,
        source,
        kind: ReplacementEffectKind::PreventDamageFromColorUntilEndOfTurn {
            color,
            turn: state.turn,
            active_player: state.active_player,
        },
    });
    Ok(id)
}

fn lifelink_gain_for(state: &GameState, event: &ProposedEvent) -> Option<(PlayerId, i32)> {
    let ProposedEvent::Damage(damage) = event else {
        return None;
    };
    if damage.amount <= 0 {
        return None;
    }
    let source = state.objects.try_get(damage.source)?;
    crate::engine::has_effective_keyword(state, damage.source, crate::card_def::Keywords::LIFELINK)
        .then_some((source.controller, damage.amount))
}

/// Convenience: replace/prevent then commit if anything survived. Lifelink
/// is evaluated from the final damage amount and source immediately before
/// that damage is committed.
pub fn propose_and_commit(state: &mut GameState, event: ProposedEvent) {
    if let Some(final_event) = apply_replacements(state, event) {
        let lifelink_gain = lifelink_gain_for(state, &final_event);
        commit(state, final_event);
        if let Some((player, amount)) = lifelink_gain {
            commit(state, ProposedEvent::life_gain(player, amount));
        }
    }
}

/// Applies the (possibly rewritten) proposal to `GameState` and appends the
/// resulting `CommittedEvent` to the event log for this resolution.
pub fn commit(state: &mut GameState, event: ProposedEvent) {
    let committed = match event {
        ProposedEvent::Damage(d) => {
            let source_has_deathtouch = d.amount > 0
                && state.objects.try_get(d.source).is_some()
                && crate::engine::has_effective_keyword(
                    state,
                    d.source,
                    crate::card_def::Keywords::DEATHTOUCH,
                );
            match d.target {
                Target::Object(id) => {
                    let obj = state.objects.get_mut(id);
                    obj.damage = obj.damage.saturating_add(d.amount.max(0) as u16);
                    obj.v4.deathtouch_damage |= source_has_deathtouch;
                }
                Target::Player(p) => {
                    state.players[p.index()].life -= d.amount;
                }
            }
            CommittedEvent::Damage {
                source: d.source,
                target: d.target,
                amount: d.amount,
            }
        }
        ProposedEvent::ZoneChange(z) => {
            let from = state.objects.get(z.object).zone;
            let controller_before = state.objects.get(z.object).controller;
            commit_zone_change(
                state,
                z.object,
                z.to_zone,
                z.library_placement,
                z.preserve_known_identity,
                z.library_insert_visibility,
                z.force_battlefield_tapped,
                z.battlefield_face_index,
                z.battlefield_controller,
            );
            CommittedEvent::ZoneChange {
                object: z.object,
                from,
                to: z.to_zone,
                controller_before,
            }
        }
        ProposedEvent::LifeLoss(l) => {
            state.players[l.player.index()].life -= l.amount;
            CommittedEvent::LifeLoss {
                player: l.player,
                amount: l.amount,
            }
        }
        ProposedEvent::LifeGain(g) => {
            state.players[g.player.index()].life += g.amount;
            CommittedEvent::LifeGain {
                player: g.player,
                amount: g.amount,
            }
        }
        ProposedEvent::Draw(d) => {
            let empty_before = state.players[d.player.index()].library.is_empty();
            let drawn = state.draw_card(d.player);
            if empty_before {
                state.players[d.player.index()].drew_from_empty = true;
            }
            if drawn.is_some() {
                state.players[d.player.index()].draws_this_turn += 1;
            }
            CommittedEvent::Draw {
                player: d.player,
                object: drawn,
            }
        }
        ProposedEvent::Tap(t) => {
            state.objects.get_mut(t.object).tapped = true;
            CommittedEvent::Tap { object: t.object }
        }
        ProposedEvent::ManaAdd(m) => {
            for &c in &m.colors {
                state.players[m.player.index()].mana_pool[c.pool_index()] += 1;
            }
            CommittedEvent::ManaAdded {
                player: m.player,
                colors: m.colors,
            }
        }
        ProposedEvent::CreateToken(t) => {
            let token_def = &crate::card_def::CARD_DEFS[t.token_def as usize];
            let name = token_def.object_name.to_string();
            let object = state.objects.push(crate::state::GameObject {
                card_def: t.token_def,
                name,
                owner: t.controller,
                controller: t.controller,
                zone: Zone::Battlefield,
                tapped: false,
                // A token entering the battlefield is exactly as summoning-
                // sick as any other permanent that just entered (see
                // `commit_zone_change`'s identical `= true` a few lines
                // down for the ordinary cast/move path) -- this was
                // hardcoded `false` and never flipped, the one "enters
                // battlefield" path that skipped setting it. Found via the
                // branch-differential pilot (Sol #89/#91): a Blood Token's
                // controlled-since-turn-start flag disagreed with the
                // reference engine's `wasControlledFromStartOfControllerTurn()`
                // immediately after Voldaren Epicure's ETB created it.
                summoning_sick: true,
                damage: 0,
                counters: Default::default(),
                attachments: Vec::new(),
                v4: {
                    let mut v4 = crate::state::ObjectStateV4::from_card_def(t.token_def);
                    v4.entered_battlefield_turn = Some(state.turn);
                    v4
                },
                spell_copy_origin: None,
                plotted_turn: None,
                zone_change_count: 0,
            });
            state.players[t.controller.index()].battlefield.push(object);
            let enters_tapped = permanent_enters_battlefield_tapped(state, object, t.controller);
            state.objects.get_mut(object).tapped = enters_tapped;
            CommittedEvent::CreateToken {
                object,
                token_def: t.token_def,
                controller: t.controller,
            }
        }
    };
    let saga_entered = matches!(
        committed,
        CommittedEvent::ZoneChange {
            object,
            to: Zone::Battlefield,
            ..
        } if state.objects.get(object).v4.face_index == 0
            && crate::card_def::CARD_DEFS[state.objects.get(object).card_def as usize]
                .saga
                .is_some()
    );
    let saga_source = match &committed {
        CommittedEvent::ZoneChange { object, .. } if saga_entered => Some(*object),
        _ => None,
    };
    state.engine.event_log.push(committed.clone());
    state.engine.event_history.push(committed);
    if let Some(source) = saga_source {
        let chapter = {
            let lore = &mut state.objects.get_mut(source).counters.lore;
            *lore = lore
                .checked_add(1)
                .expect("a newly entered Saga's first lore counter fits i16");
            u8::try_from(*lore).expect("a supported Saga chapter fits u8")
        };
        log_saga_chapter(state, source, chapter);
    }
}

/// Runs the replace/prevent pass independently on every event in `events`
/// (each is evaluated against the currently-active replacements as if it
/// were the only proposal in flight -- true simultaneity: none of them can
/// see or react to one another), then commits every survivor back-to-back
/// with no SBA/trigger check interleaved. Used for combat damage (510.2:
/// all of it happens at once); the caller is responsible for running SBAs
/// / trigger collection exactly once after the whole batch (see
/// `engine::deal_combat_damage`), not per event.
pub fn propose_and_commit_batch(state: &mut GameState, events: Vec<ProposedEvent>) {
    let survivors: Vec<ProposedEvent> = events
        .into_iter()
        .filter_map(|e| apply_replacements(state, e))
        .collect();
    // Lifelink changes life at the same time as the damage. Capture every
    // source/controller before any member of the simultaneous batch can die.
    let lifelink_gains = survivors
        .iter()
        .filter_map(|event| lifelink_gain_for(state, event))
        .collect::<Vec<_>>();
    for e in survivors {
        commit(state, e);
    }
    for (player, amount) in lifelink_gains {
        commit(state, ProposedEvent::life_gain(player, amount));
    }
}

/// Logs a `SpellCast` marker with no accompanying state mutation (casting
/// itself -- moving hand to stack -- is handled by
/// `engine::move_hand_to_stack`; this is purely a trigger-matching hook).
/// Not named `commit_*` and not routed through `propose_and_commit`
/// because there is no proposed/replaceable form of "a spell was cast" in
/// this increment's scope (countering a spell removes it from the stack
/// later, it doesn't replace the cast event itself).
pub fn log_spell_cast(state: &mut GameState, spell: ObjectId, controller: PlayerId) {
    let committed = CommittedEvent::SpellCast { spell, controller };
    state.engine.event_log.push(committed.clone());
    state.engine.event_history.push(committed);
}

/// Logs one final targeting marker after the enclosing spell or ability has
/// completed announcement. Failed casts never reach this point.
pub fn log_targeted(
    state: &mut GameState,
    target: ObjectId,
    target_zone_change_count: u32,
    targeting_stack_item: StackItemId,
    targeting_controller: PlayerId,
) {
    let committed = CommittedEvent::Targeted {
        target,
        target_zone_change_count,
        targeting_stack_item,
        targeting_controller,
    };
    state.engine.event_log.push(committed.clone());
    state.engine.event_history.push(committed);
}

/// Records the rules action separately from its ensuing replaceable zone
/// change. Sacrifice triggers consume this marker instead of guessing from
/// an ordinary battlefield-to-graveyard move.
pub fn log_sacrifice(state: &mut GameState, object: ObjectId) {
    let live = state.objects.get(object);
    let committed = CommittedEvent::Sacrificed {
        object,
        controller_before: live.controller,
        effective_subtype_ids_before: crate::engine::effective_subtype_ids(state, object),
    };
    state.engine.event_log.push(committed.clone());
    state.engine.event_history.push(committed);
}

/// Records that one exact permanent incarnation dealt combat damage to a
/// player. The underlying damage event remains independently committed;
/// this marker exists only to distinguish combat from noncombat damage for
/// triggered-ability matching.
pub fn log_combat_damage_to_player(
    state: &mut GameState,
    source: ObjectId,
    source_zone_change_count: u32,
    player: PlayerId,
    amount: i32,
) {
    let committed = CommittedEvent::CombatDamageToPlayer {
        source,
        source_zone_change_count,
        player,
        amount,
    };
    state.engine.event_log.push(committed.clone());
    state.engine.event_history.push(committed);
}

/// Logs the nonreplaceable chapter marker immediately after a lore counter
/// is placed by the Saga rules action.
pub fn log_saga_chapter(state: &mut GameState, source: ObjectId, chapter: u8) {
    let object = state.objects.get(source);
    let committed = CommittedEvent::SagaChapter {
        source,
        source_zone_change_count: object.zone_change_count,
        controller: object.controller,
        chapter,
    };
    state.engine.event_log.push(committed.clone());
    state.engine.event_history.push(committed);
}

pub fn log_initiative_trigger(
    state: &mut GameState,
    player: PlayerId,
    mut source: crate::state::AbilitySourceContractV4,
    kind: crate::state::InitiativeTriggerKindV1,
) -> Result<crate::state::InitiativeTriggerBindingV1, String> {
    let live = state
        .objects
        .try_get(source.source)
        .ok_or("Initiative designation source no longer exists")?;
    if live.card_def != source.card_def
        || live.owner != source.owner
        || live.zone_change_count < source.zone_change_count
        || (live.zone_change_count == source.zone_change_count && live.zone != source.zone)
        || crate::card_def::CARD_DEFS
            .get(source.card_def as usize)
            .is_none_or(|definition| definition.name != "Avenging Hunter")
    {
        return Err("Initiative designation source contract is malformed".to_string());
    }
    source.controller = player;
    let history_index = u32::try_from(state.engine.event_history.len())
        .map_err(|_| "Initiative event history exceeds u32".to_string())?;
    let binding = crate::state::InitiativeTriggerBindingV1 {
        history_index,
        player,
        source,
        kind,
    };
    let committed = CommittedEvent::InitiativeTrigger { binding };
    state.engine.event_log.push(committed.clone());
    state.engine.event_history.push(committed);
    Ok(binding)
}

fn permanent_enters_battlefield_tapped(
    state: &GameState,
    object: ObjectId,
    controller: PlayerId,
) -> bool {
    let def = &crate::card_def::CARD_DEFS[state.objects.get(object).card_def as usize];
    if def.enters_battlefield_tapped {
        return true;
    }
    let Some(rule) = def.enters_battlefield_tapped_unless else {
        return false;
    };
    let controlled_other_count = state.players[controller.index()]
        .battlefield
        .iter()
        .copied()
        .filter(|candidate| *candidate != object)
        .filter(|candidate| {
            let live = state.objects.get(*candidate);
            live.zone == Zone::Battlefield
                && live.controller == controller
                && crate::engine::has_effective_subtype(
                    state,
                    *candidate,
                    rule.controller_controls_other_subtype,
                )
        })
        .count();
    controlled_other_count < usize::from(rule.minimum_count)
}

/// Zone bookkeeping shared by every `MoveObject` effect leaf. "Hand ->
/// Stack" (casting) is deliberately not reachable here: putting a spell on
/// the stack is an engine action (see `engine::begin_cast`), never
/// something a card's own effect program does.
#[allow(clippy::too_many_arguments)]
fn commit_zone_change(
    state: &mut GameState,
    id: ObjectId,
    to_zone: Zone,
    library_placement: LibraryPlacement,
    preserve_known_identity: bool,
    library_insert_visibility: LibraryInsertVisibility,
    force_battlefield_tapped: bool,
    battlefield_face_index: Option<u8>,
    battlefield_controller: Option<PlayerId>,
) {
    let owner = state.objects.get(id).owner;
    let from_zone = state.objects.get(id).zone;
    refresh_paid_creature_power_lki(state, id, from_zone);
    let informed_observer_mask =
        if preserve_known_identity && from_zone == Zone::Library && to_zone == Zone::Hand {
            let position = state.players[owner.index()]
                .library
                .iter()
                .position(|&candidate| candidate == id);
            let generation = state.objects.get(id).zone_change_count;
            position.map_or(0, |position| {
                [PlayerId::P0, PlayerId::P1]
                    .into_iter()
                    .filter(|&observer| {
                        state
                            .known_library_cards(observer, owner)
                            .iter()
                            .any(|entry| {
                                entry.position as usize == position
                                    && entry.object == id
                                    && entry.zone_change_count == generation
                            })
                    })
                    .fold(0_u8, |mask, observer| mask | (1 << observer.index()))
            })
        } else {
            0
        };

    remove_from_zone(state, owner, id, from_zone);
    state.forget_hand_object(id);
    state.clear_object_relations(id);

    match to_zone {
        Zone::Library => {
            let library_len = state.players[owner.index()].library.len();
            let position = match library_placement {
                LibraryPlacement::Top => 0,
                LibraryPlacement::SecondFromTop => 1.min(library_len),
                LibraryPlacement::Bottom => library_len,
            };
            // The insertion position is public even when the inserted
            // identity is not. Preserve every still-valid older fact by
            // shifting it deeper; visibility is installed only after the
            // inserted object's new incarnation exists below.
            state.note_library_insertion(owner, position);
            state.players[owner.index()].library.insert(position, id);
        }
        Zone::Hand => state.players[owner.index()].hand.push(id),
        Zone::Battlefield => state.players[battlefield_controller.unwrap_or(owner).index()]
            .battlefield
            .push(id),
        Zone::Graveyard => state.players[owner.index()].graveyard.push(id),
        Zone::Exile => state.exile.push(id),
        Zone::Command => state.command.push(id),
        Zone::Stack => panic!("MoveObject to Stack is an engine action, not an effect leaf"),
    }

    let turn = state.turn;
    let enters_battlefield_tapped = to_zone == Zone::Battlefield
        && (force_battlefield_tapped || permanent_enters_battlefield_tapped(state, id, owner));
    {
        let obj = state.objects.get_mut(id);
        obj.zone = to_zone;
        // A zone change creates a new object with no carried-over control
        // effect. Moves to Stack are engine actions and never enter this
        // helper, so every destination handled here begins owner-controlled.
        obj.controller = if to_zone == Zone::Battlefield {
            battlefield_controller.unwrap_or(owner)
        } else {
            owner
        };
        // Plot is provenance of one exact exile incarnation, not a durable
        // property of the physical card. `engine::plot_spell` re-stamps the
        // newly-created Exile incarnation immediately after this move; every
        // other ordinary zone change must clear the old marker.
        obj.plotted_turn = None;
        // CR 400.7's zone-change identity: bumped on *every* zone change,
        // regardless of which zones, so `engine::PlayPermission::
        // zone_change_generation` can tell "still sitting where it was granted"
        // apart from "moved since, for any reason" without needing a
        // zone-specific special case.
        obj.zone_change_count += 1;
        obj.v4.reset_for_zone_change(obj.card_def, to_zone, turn);
        obj.name = crate::card_def::CARD_DEFS[obj.card_def as usize]
            .object_name
            .to_string();
        obj.damage = 0;
        obj.counters = Default::default();
        obj.attachments.clear();
        if to_zone == Zone::Battlefield {
            if let Some(face_index) = battlefield_face_index {
                let def = &crate::card_def::CARD_DEFS[obj.card_def as usize];
                let Some(face) = (face_index == 1)
                    .then_some(def.transform_face.as_ref())
                    .flatten()
                else {
                    panic!("transformed battlefield return requested an undefined face");
                };
                obj.v4.face_index = face_index;
                obj.v4.effective_color_mask = crate::card_def::mana_colors_mask(face.colors);
                obj.v4.effective_subtype_ids = face
                    .subtypes
                    .iter()
                    .map(|subtype| subtype.stable_id())
                    .collect();
                obj.v4.effective_subtype_ids.sort_unstable();
                obj.v4.effective_subtype_ids.dedup();
                obj.name = face.name.to_string();
            }
            obj.tapped = enters_battlefield_tapped;
            obj.summoning_sick = true;
        } else {
            obj.tapped = false;
            obj.summoning_sick = false;
        }
    }
    if to_zone == Zone::Library {
        let position = state.players[owner.index()]
            .library
            .iter()
            .position(|&candidate| candidate == id)
            .expect("just-inserted library object remains indexed");
        match library_insert_visibility {
            LibraryInsertVisibility::Hidden => {}
            LibraryInsertVisibility::Owner => {
                state.reveal_library_position(owner, owner, position);
            }
            LibraryInsertVisibility::Public => {
                for observer in [PlayerId::P0, PlayerId::P1] {
                    state.reveal_library_position(observer, owner, position);
                }
            }
        }
    }
    if to_zone == Zone::Hand && from_zone != Zone::Library {
        // Returning a publicly identified card to hand does not make that
        // identity secret again. The owner sees it through `own_hand`; the
        // other observer receives an incarnation-bound known-hand fact.
        for observer in [PlayerId::P0, PlayerId::P1] {
            state
                .reveal_hand_card(observer, owner, id)
                .expect("just moved this live public object into its owner's hand");
        }
    } else if to_zone == Zone::Hand && informed_observer_mask != 0 {
        // A public library reveal followed by a move to hand keeps the
        // revealed identity public. Install the fact only for observers who
        // actually knew this exact position/object/incarnation before the
        // move; the ordinary constructor above remains deliberately hidden.
        for observer in [PlayerId::P0, PlayerId::P1] {
            if informed_observer_mask & (1 << observer.index()) != 0 {
                state
                    .reveal_hand_card(observer, owner, id)
                    .expect("known library object just moved into its owner's hand");
            }
        }
    }
}

/// Monstrous Emergence keeps the chosen battlefield creature's last-known
/// power, not merely its power when the casting cost was announced. Capture
/// that value immediately before the exact paid-cost incarnation leaves.
fn refresh_paid_creature_power_lki(state: &mut GameState, id: ObjectId, from_zone: Zone) {
    if from_zone != Zone::Battlefield {
        return;
    }
    let Some(object) = state.objects.try_get(id) else {
        return;
    };
    let generation = object.zone_change_count;
    let power = crate::engine::effective_power(state, id);
    let refresh = |references: &mut Vec<crate::state::PaidCostRefV4>| {
        for reference in references {
            if reference.object == id
                && reference.zone == Zone::Battlefield
                && reference.zone_change_count == generation
                && reference.power_lki.is_some()
            {
                reference.power_lki = Some(power);
            }
        }
    };
    let refresh_binding = |binding: &mut Option<crate::state::FinalizedCastBindingV1>| {
        let Some(reference) = binding
            .as_mut()
            .and_then(|binding| binding.chosen_creature_cost.as_mut())
        else {
            return;
        };
        if reference.object == id
            && reference.zone == Zone::Battlefield
            && reference.zone_change_count == generation
            && reference.power_lki.is_some()
        {
            reference.power_lki = Some(power);
        }
    };
    for item in &mut state.stack {
        refresh(&mut item.v4.paid_cost_refs);
        if let Some(contract) = item.v4.source_contract.as_mut() {
            refresh_binding(&mut contract.finalized_cast_binding);
        }
    }
    if let Some(pending) = state.engine.pending_effect.as_mut() {
        refresh(&mut pending.resolving_item.v4.paid_cost_refs);
        if let Some(contract) = pending.resolving_item.v4.source_contract.as_mut() {
            refresh_binding(&mut contract.finalized_cast_binding);
        }
        refresh(&mut pending.ctx.paid_cost_refs);
    }
    for (_, object) in state.objects.iter_mut() {
        refresh_binding(&mut object.v4.finalized_cast_binding);
    }
}

/// Removes a virtual game object from whichever live zone indexes it.
/// This covers 111.8/704.5d token cleanup and 707.10a spell copies leaving
/// the stack. Removes `id` from
/// whichever zone list it's currently tracked in (its owner's hand/library/
/// graveyard, `state.exile`/`command`, or the stack) without adding it
/// anywhere -- unlike every other zone transition, a token leaving the
/// battlefield doesn't go *to* another real zone, it just stops being
/// tracked. Token callers run from `trigger::sba_fixed_point`; copy callers
/// run synchronously from spell resolution/countering.
///
/// Returns whether `id` was actually still present (an already-ceased token
/// is a legal, idempotent no-op call) -- `sba_fixed_point`'s fixed-point
/// loop needs this to know whether the sweep made progress; unconditionally
/// reporting "changed" here would loop forever re-"removing" the same
/// already-gone object every pass.
///
/// Deliberately does *not* touch `GameObject::zone` (left as the object's
/// last physical marker). Live membership is authoritative from the zone
/// indexes and `state.stack`, which this function removes it from. Arena ids
/// are never freed, so snapshots and provenance may still refer to the inert
/// historical identity without making it a live target.
///
/// Also drops every `state.engine.linked_exile_records` row whose exiled
/// object is `id`. Because the zone marker and zone-change count stay as
/// recorded, such a row would otherwise still look like a live, uniquely
/// exiled incarnation: observation fails its uniqueness proof and
/// `ReturnObjectsExiledBySource` would put the ceased token back onto the
/// battlefield, which 111.8 forbids. Rows naming `id` only as their source
/// are kept: a token source ceases only after leaving the battlefield, and
/// its leaves-the-battlefield return still resolves from the row by source
/// contract.
pub fn cease_to_exist(state: &mut GameState, id: ObjectId) -> bool {
    let owner = state.objects.get(id).owner;
    let zone = state.objects.get(id).zone;
    let removed = remove_from_zone(state, owner, id, zone);
    if removed {
        state.forget_hand_object(id);
        state.clear_object_relations(id);
        state
            .engine
            .linked_exile_records
            .retain(|record| record.exiled != id);
    }
    removed
}

/// Returns whether `id` was actually present in `zone`'s list before being
/// removed -- see `cease_to_exist`'s doc for why that matters to callers.
fn remove_from_zone(state: &mut GameState, owner: PlayerId, id: ObjectId, zone: Zone) -> bool {
    fn drop_from(v: &mut Vec<ObjectId>, id: ObjectId) -> bool {
        let before = v.len();
        v.retain(|&x| x != id);
        before != v.len()
    }
    match zone {
        Zone::Library => {
            let position = state.players[owner.index()]
                .library
                .iter()
                .position(|&candidate| candidate == id);
            let removed = drop_from(&mut state.players[owner.index()].library, id);
            if let Some(position) = position {
                // At present every generic library departure is from a
                // publicly determined position (draw/top-card exile). A
                // future hidden search must use its own knowledge-aware
                // library operation instead of this generic zone move.
                state.note_library_removal(owner, position);
            }
            removed
        }
        Zone::Hand => drop_from(&mut state.players[owner.index()].hand, id),
        Zone::Battlefield => {
            // Battlefield membership is controller-keyed, unlike every
            // owner-keyed hidden/public card zone. A stolen permanent must
            // leave its current controller's battlefield before the same
            // physical card enters its owner's destination zone.
            let controller = state.objects.get(id).controller;
            drop_from(&mut state.players[controller.index()].battlefield, id)
        }
        Zone::Graveyard => drop_from(&mut state.players[owner.index()].graveyard, id),
        Zone::Exile => drop_from(&mut state.exile, id),
        Zone::Command => drop_from(&mut state.command, id),
        Zone::Stack => {
            let before = state.stack.len();
            let live_generation = state.objects.get(id).zone_change_count;
            state.stack.retain(|item| {
                item.kind != StackItemKind::Spell
                    || item.source != id
                    || item
                        .v4
                        .source_contract
                        .is_none_or(|contract| contract.zone_change_count != live_generation)
            });
            before != state.stack.len()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::PlayerId;

    fn fresh_state() -> GameState {
        GameState::new_from_libraries(&[1, 2, 3], &[4, 5, 6], |c| format!("card-{c}"), 1)
    }

    #[test]
    fn commit_damage_to_player_reduces_life() {
        let mut state = fresh_state();
        propose_and_commit(
            &mut state,
            ProposedEvent::damage(ObjectId(0), Target::Player(PlayerId::P1), 3),
        );
        assert_eq!(state.players[1].life, 17);
        assert_eq!(
            state.engine.event_log,
            vec![CommittedEvent::Damage {
                source: ObjectId(0),
                target: Target::Player(PlayerId::P1),
                amount: 3
            }]
        );
    }

    #[test]
    fn zone_change_moves_between_owner_zones_and_updates_object() {
        let mut state = fresh_state();
        let card = state.draw_card(PlayerId::P0).unwrap();
        propose_and_commit(
            &mut state,
            ProposedEvent::zone_change(card, Zone::Battlefield),
        );
        assert_eq!(state.objects.get(card).zone, Zone::Battlefield);
        assert!(state.players[0].battlefield.contains(&card));
        assert!(!state.players[0].hand.contains(&card));
    }

    #[test]
    fn library_to_hand_preserves_identity_only_when_explicit_and_informed() {
        let mut hidden_state = fresh_state();
        let hidden = hidden_state.players[0].library[0];
        propose_and_commit(
            &mut hidden_state,
            ProposedEvent::zone_change(hidden, Zone::Hand),
        );
        assert!(hidden_state
            .known_hand_cards(PlayerId::P1, PlayerId::P0)
            .is_empty());

        let mut revealed_state = fresh_state();
        let revealed = revealed_state.players[0].library[0];
        let old_generation = revealed_state.objects.get(revealed).zone_change_count;
        revealed_state.reveal_library_top(PlayerId::P1, PlayerId::P0, 1);
        propose_and_commit(
            &mut revealed_state,
            ProposedEvent::zone_change_preserving_known_identity(revealed, Zone::Hand),
        );
        assert_eq!(
            revealed_state
                .known_hand_cards(PlayerId::P1, PlayerId::P0)
                .iter()
                .map(|entry| (entry.object, entry.zone_change_count))
                .collect::<Vec<_>>(),
            vec![(revealed, old_generation + 1)]
        );
        assert!(revealed_state
            .known_hand_cards(PlayerId::P0, PlayerId::P0)
            .is_empty());

        let mut stale_state = fresh_state();
        let stale = stale_state.players[0].library[0];
        stale_state.reveal_library_top(PlayerId::P1, PlayerId::P0, 1);
        stale_state.objects.get_mut(stale).zone_change_count += 1;
        propose_and_commit(
            &mut stale_state,
            ProposedEvent::zone_change_preserving_known_identity(stale, Zone::Hand),
        );
        assert!(stale_state
            .known_hand_cards(PlayerId::P1, PlayerId::P0)
            .is_empty());
    }

    #[test]
    fn draw_from_empty_library_sets_drew_from_empty() {
        let mut state = GameState::new_from_libraries(&[], &[1], |c| format!("card-{c}"), 1);
        propose_and_commit(&mut state, ProposedEvent::draw(PlayerId::P0));
        assert!(state.players[0].drew_from_empty);
    }

    /// End-to-end proof of the replacement pipeline shape required by the
    /// design: a prevention shield partially absorbs one hit, then expires
    /// and lets a subsequent hit through in full.
    #[test]
    fn prevention_shield_absorbs_then_expires() {
        let mut state = fresh_state();
        state.engine.active_replacements.push(ActiveReplacement {
            id: 1,
            source: ObjectId(0),
            kind: ReplacementEffectKind::PreventNextDamage {
                target: Target::Player(PlayerId::P1),
                remaining: 2,
            },
        });

        // First hit: 5 damage, shield absorbs 2 -> 3 gets through.
        propose_and_commit(
            &mut state,
            ProposedEvent::damage(ObjectId(0), Target::Player(PlayerId::P1), 5),
        );
        assert_eq!(state.players[1].life, 17);
        assert!(
            state.engine.active_replacements.is_empty(),
            "shield should be fully consumed"
        );

        // Second hit: shield is gone, full damage applies.
        propose_and_commit(
            &mut state,
            ProposedEvent::damage(ObjectId(0), Target::Player(PlayerId::P1), 4),
        );
        assert_eq!(state.players[1].life, 13);
    }

    #[test]
    fn prevention_shield_can_fully_prevent_small_hits() {
        let mut state = fresh_state();
        state.engine.active_replacements.push(ActiveReplacement {
            id: 7,
            source: ObjectId(0),
            kind: ReplacementEffectKind::PreventNextDamage {
                target: Target::Player(PlayerId::P1),
                remaining: 10,
            },
        });
        propose_and_commit(
            &mut state,
            ProposedEvent::damage(ObjectId(0), Target::Player(PlayerId::P1), 3),
        );
        assert_eq!(
            state.players[1].life, 20,
            "fully prevented, no event should mutate life"
        );
        assert_eq!(
            state.engine.active_replacements[0].kind,
            ReplacementEffectKind::PreventNextDamage {
                target: Target::Player(PlayerId::P1),
                remaining: 7,
            }
        );
    }

    #[test]
    fn a_replacement_never_touches_the_same_proposal_twice() {
        // A replacement that always "applies" but rewrites to something it
        // would also match would loop forever without touched_by tracking.
        // PreventNextDamage never re-matches its own rewritten (smaller)
        // event because it fully consumes itself in one hit; this test
        // just pins that a single shield only ever fires once per event.
        let mut state = fresh_state();
        state.engine.active_replacements.push(ActiveReplacement {
            id: 3,
            source: ObjectId(0),
            kind: ReplacementEffectKind::PreventNextDamage {
                target: Target::Player(PlayerId::P1),
                remaining: 100,
            },
        });
        propose_and_commit(
            &mut state,
            ProposedEvent::damage(ObjectId(0), Target::Player(PlayerId::P1), 5),
        );
        assert_eq!(state.players[1].life, 20);
        assert_eq!(
            state.engine.active_replacements[0].kind,
            ReplacementEffectKind::PreventNextDamage {
                target: Target::Player(PlayerId::P1),
                remaining: 95,
            }
        );
    }

    /// Spellbench launch-benchmark reproduction (CawGates): Journey to Nowhere
    /// exiled a token, the 111.8/704.5d sweep ceased it, and the linked-exile
    /// record outlived it, so every later observation failed and Journey
    /// leaving would have returned the ceased token.
    #[test]
    fn cease_to_exist_drops_the_linked_exile_record_of_the_ceased_exiled_object() {
        use crate::state::{AbilitySourceContractV4, LinkedExileRecordV4, ObjectLinkV4};

        let mut state = fresh_state();
        let source = push_object_into(&mut state, PlayerId::P0, Zone::Battlefield);
        let exiled = push_object_into(&mut state, PlayerId::P1, Zone::Exile);
        state.players[0].battlefield.push(source);
        state.exile.push(exiled);
        state.objects.get_mut(exiled).v4.exiled_by = Some(ObjectLinkV4 {
            object: source,
            zone_change_count: 0,
        });
        state.engine.linked_exile_records.push(LinkedExileRecordV4 {
            source: AbilitySourceContractV4 {
                source,
                card_def: 0,
                owner: PlayerId::P0,
                controller: PlayerId::P0,
                zone: Zone::Battlefield,
                zone_change_count: 0,
                attached_to: None,
            },
            exiled,
            exiled_card_def: 0,
            exiled_owner: PlayerId::P1,
            exiled_zone_change_count: 0,
        });

        assert!(cease_to_exist(&mut state, exiled));
        assert!(!state.exile.contains(&exiled));
        assert_eq!(state.objects.get(exiled).v4.exiled_by, None);
        assert!(
            state.engine.linked_exile_records.is_empty(),
            "a ceased exiled object must not leave a linked-exile record: {:?}",
            state.engine.linked_exile_records
        );
    }

    /// A token source that left the battlefield has already triggered its
    /// return, which resolves from the row by source contract, so the row
    /// must survive the source ceasing to exist.
    #[test]
    fn cease_to_exist_keeps_the_linked_exile_record_of_a_ceased_token_source() {
        use crate::state::{AbilitySourceContractV4, LinkedExileRecordV4};

        let mut state = fresh_state();
        let token_source = push_object_into(&mut state, PlayerId::P0, Zone::Graveyard);
        state.players[0].graveyard.push(token_source);
        let still_exiled = push_object_into(&mut state, PlayerId::P1, Zone::Exile);
        state.exile.push(still_exiled);
        let record = LinkedExileRecordV4 {
            source: AbilitySourceContractV4 {
                source: token_source,
                card_def: 0,
                owner: PlayerId::P0,
                controller: PlayerId::P0,
                zone: Zone::Battlefield,
                zone_change_count: 0,
                attached_to: None,
            },
            exiled: still_exiled,
            exiled_card_def: 0,
            exiled_owner: PlayerId::P1,
            exiled_zone_change_count: 0,
        };
        state.objects.get_mut(token_source).zone_change_count = 1;
        state.engine.linked_exile_records.push(record);
        assert!(cease_to_exist(&mut state, token_source));
        assert_eq!(state.engine.linked_exile_records, vec![record]);
    }

    fn push_object_into(state: &mut GameState, owner: PlayerId, zone: Zone) -> ObjectId {
        state.objects.push(crate::state::GameObject {
            card_def: 0,
            name: "fixture".to_string(),
            owner,
            controller: owner,
            zone,
            tapped: false,
            summoning_sick: false,
            damage: 0,
            counters: Default::default(),
            attachments: Vec::new(),
            v4: crate::state::ObjectStateV4::from_card_def(0),
            spell_copy_origin: None,
            plotted_turn: None,
            zone_change_count: 0,
        })
    }
}
