//! What happened in each step, for a person watching a game. Off unless
//! started on this thread, so searches and self-play never record; not game
//! state.

use crate::event::CommittedEvent;
use crate::ids::PlayerId;
use crate::state::{GameState, Step};
use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicUsize, Ordering};

/// The events of one step, or of the part of it since the last look.
#[derive(Debug, Clone)]
pub struct StepEvents {
    pub turn: u32,
    pub step: Step,
    pub active: PlayerId,
    /// Each player's life when the step ended, or at the look.
    pub life: [i32; 2],
    pub events: Vec<CommittedEvent>,
}

/// Threads recording now: a plain load in every step's way, where a
/// thread-local lookup cost a fifth of an engine step on this toolchain.
static RECORDING: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static ON: Cell<bool> = const { Cell::new(false) };
    static CURSOR: Cell<u32> = const { Cell::new(0) };
    static STEPS: RefCell<Vec<StepEvents>> = const { RefCell::new(Vec::new()) };
}

/// Records on this thread from absolute history index `cursor` (see
/// [`GameState::next_history_index`]) until [`stop`].
pub fn start(cursor: u32) {
    if !ON.replace(true) {
        RECORDING.fetch_add(1, Ordering::Relaxed);
    }
    CURSOR.set(cursor);
    STEPS.with_borrow_mut(Vec::clear);
}

/// Stops recording: the steps with events since [`start`], and the cursor
/// to start from next time.
pub fn stop(state: &GameState) -> (Vec<StepEvents>, u32) {
    harvest(state);
    if ON.replace(false) {
        RECORDING.fetch_sub(1, Ordering::Relaxed);
    }
    (STEPS.take(), CURSOR.get())
}

/// A step is ending: keep its events before the next step's turn-based
/// actions (Untap clears the history).
pub(crate) fn step_ending(state: &GameState) {
    if RECORDING.load(Ordering::Relaxed) != 0 && ON.get() {
        harvest(state);
    }
}

fn harvest(state: &GameState) {
    let history = &state.engine.event_history;
    let from = CURSOR.get().saturating_sub(state.event_history_base) as usize;
    let events = history.get(from..).map_or_else(Vec::new, <[CommittedEvent]>::to_vec);
    CURSOR.set(state.event_history_base.saturating_add(history.len() as u32));
    if !events.is_empty() {
        let life = [state.players[0].life, state.players[1].life];
        STEPS.with_borrow_mut(|s| s.push(StepEvents { turn: state.turn, step: state.step, active: state.active_player, life, events }));
    }
}
