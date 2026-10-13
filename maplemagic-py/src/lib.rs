//! Python bindings, `maplemagic._core`. Observations and choices cross as
//! zero-padded uint16 numpy arrays with a length per row.

use maplemagic::features::{
    action_vocab, card_static_table, static_bits, static_vocab, token_bits, token_vocab, ACTION_FIELDS,
    LEGACY_TOKEN_FIELDS, MAX_TOKENS, STATIC_FIELDS, TOKEN_FIELDS,
};
use maplemagic::features::{choice_fields, observe_with};
use maplemagic::game::{deck_names, deck_pairs, Game, Outcome};
use maplemagic::ismcts::{HeuristicEval, Search};
use maplemagic::play;
use maplemagic::history::{hist_vocab, HIST_FIELDS, MAX_HIST};
use maplemagic::ismcts::{Config, Eval, MapleSelect, WorldSampler, RETENTION_BUCKETS};
use maplemagic::mtg_kernel::card_def::{CardType, CARD_DEFS};
use maplemagic::selfplay::{SelfPlay as RsSelfPlay, SelfPlayConfig, HEURISTIC, RANDOM};
use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;

type Tok = [u16; TOKEN_FIELDS];
type Act = [u16; ACTION_FIELDS];
type Hist = [u16; HIST_FIELDS];

fn per_model<'py, T, F>(value: Option<&Bound<'py, PyAny>>, default: T, parse: F) -> PyResult<[T; 2]>
where
    T: Copy,
    F: Fn(&Bound<'py, PyAny>) -> PyResult<T>,
{
    let Some(v) = value else { return Ok([default; 2]) };
    if let Ok((a, b)) = v.extract::<(Bound<'py, PyAny>, Bound<'py, PyAny>)>() {
        return Ok([parse(&a)?, parse(&b)?]);
    }
    let one = parse(v)?;
    Ok([one; 2])
}

fn parse_select(v: &Bound<'_, PyAny>) -> PyResult<MapleSelect> {
    match v.extract::<String>()?.as_str() {
        "ref" => Ok(MapleSelect::RefWorld),
        "union" => Ok(MapleSelect::Union),
        s => Err(PyValueError::new_err(format!("maple_select must be \"ref\" or \"union\", not {s:?}"))),
    }
}

fn parse_sampler(v: &Bound<'_, PyAny>) -> PyResult<WorldSampler> {
    match v.extract::<String>()?.as_str() {
        "uniform" => Ok(WorldSampler::Uniform),
        "oracle" => Ok(WorldSampler::Oracle),
        "retention" => Ok(WorldSampler::Retention),
        "scored" => Ok(WorldSampler::Scored),
        s => Err(PyValueError::new_err(format!("world_sampler must be uniform, oracle, retention or scored, not {s:?}"))),
    }
}

/// Card lists as a zero-padded [rows, width] array of card definition + 1.
fn pad_cards(rows: &[&[u16]]) -> (Vec<u16>, Vec<i32>, usize) {
    let width = rows.iter().map(|r| r.len()).max().unwrap_or(0).max(1);
    let mut flat = vec![0u16; rows.len() * width];
    for (i, r) in rows.iter().enumerate() {
        for (j, &d) in r.iter().enumerate() {
            flat[i * width + j] = d + 1;
        }
    }
    (flat, rows.iter().map(|r| r.len() as i32).collect(), width)
}

/// Pad variable-length rows into a flat buffer; returns (flat, lengths, width).
fn pad<const F: usize>(rows: &[&Vec<[u16; F]>], min_width: usize) -> (Vec<u16>, Vec<i32>, usize) {
    let width = rows.iter().map(|r| r.len()).max().unwrap_or(0).max(min_width);
    let mut flat = vec![0u16; rows.len() * width * F];
    let mut lens = Vec::with_capacity(rows.len());
    for (i, r) in rows.iter().enumerate() {
        lens.push(r.len() as i32);
        for (j, t) in r.iter().enumerate() {
            let at = (i * width + j) * F;
            flat[at..at + F].copy_from_slice(t);
        }
    }
    (flat, lens, width)
}

fn arr3<'py>(py: Python<'py>, flat: Vec<u16>, a: usize, b: usize, c: usize) -> PyResult<Bound<'py, PyAny>> {
    Ok(flat.into_pyarray(py).reshape([a, b, c])?.into_any())
}

#[pyclass]
struct SelfPlay {
    inner: RsSelfPlay,
    last_batch: usize,
    /// Candidate count of each world request in the last batch.
    last_worlds: Vec<usize>,
}

#[pymethods]
impl SelfPlay {
    /// seat_models: 0/1 for networks, or "heuristic" / "random". sims is model 0's
    /// budget, opp_sims model 1's. Search settings are pairs, one per model id;
    /// leaves_per_step (default 4), maple_select, maple_resample, world_sampler, collisions, value_scale, heuristic_values and value_mix may also be one value for both.
    #[new]
    #[pyo3(signature = (parallel, pairings, sims=200, leaves_per_step=None, temp_decisions=30, c_puct=1.0, root_noise=0.25, dirichlet_alpha=0.3, fpu_reduction=0.2, seat_models=None, record=true, max_games=None, heuristic_sims=200, seed=0, opp_sims=None, maple_worlds=(0, 0), maple_select=None, maple_resample=None, maple_dedupe=true, perfect_obs=(false, false), pimc_worlds=(0, 0), world_sampler=None, world_candidates=50, world_pool=16, world_temp=0.0, retention=None, value_scale=None, heuristic_values=None, value_mix=None, full_search_prob=1.0, cheap_sims=0, completed_q=false, record_heuristic=false, collisions=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        parallel: usize,
        pairings: Vec<(String, String)>,
        sims: u32,
        leaves_per_step: Option<Bound<'_, PyAny>>,
        temp_decisions: u32,
        c_puct: f32,
        root_noise: f32,
        dirichlet_alpha: f32,
        fpu_reduction: f32,
        seat_models: Option<(Bound<'_, PyAny>, Bound<'_, PyAny>)>,
        record: bool,
        max_games: Option<u64>,
        heuristic_sims: u32,
        seed: u64,
        opp_sims: Option<u32>,
        maple_worlds: (u32, u32),
        maple_select: Option<Bound<'_, PyAny>>,
        maple_resample: Option<Bound<'_, PyAny>>,
        maple_dedupe: bool,
        perfect_obs: (bool, bool),
        pimc_worlds: (u32, u32),
        world_sampler: Option<Bound<'_, PyAny>>,
        world_candidates: u32,
        world_pool: u32,
        world_temp: f32,
        retention: Option<PyReadonlyArray1<'_, f32>>,
        value_scale: Option<Bound<'_, PyAny>>,
        heuristic_values: Option<Bound<'_, PyAny>>,
        value_mix: Option<Bound<'_, PyAny>>,
        full_search_prob: f32,
        cheap_sims: u32,
        completed_q: bool,
        record_heuristic: bool,
        collisions: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if (maple_worlds.0 > 0 && pimc_worlds.0 > 0) || (maple_worlds.1 > 0 && pimc_worlds.1 > 0) {
            return Err(PyValueError::new_err("a model can't use both maple_worlds and pimc_worlds"));
        }
        let leaves_per_step = per_model(leaves_per_step.as_ref(), 4usize, |v| v.extract::<usize>().map_err(Into::into))?;
        let maple_select = per_model(maple_select.as_ref(), MapleSelect::RefWorld, parse_select)?;
        let maple_resample = per_model(maple_resample.as_ref(), false, |v| v.extract::<bool>().map_err(Into::into))?;
        let world_sampler = per_model(world_sampler.as_ref(), WorldSampler::Uniform, parse_sampler)?;
        let value_scale = per_model(value_scale.as_ref(), 1.0f32, |v| v.extract::<f32>().map_err(Into::into))?;
        let heuristic_values = per_model(heuristic_values.as_ref(), false, |v| v.extract::<bool>().map_err(Into::into))?;
        let value_mix = per_model(value_mix.as_ref(), 1.0f32, |v| v.extract::<f32>().map_err(Into::into))?;
        let collisions = per_model(collisions.as_ref(), false, |v| v.extract::<bool>().map_err(Into::into))?;
        let retention = retention.map(|r| r.as_array().to_vec());
        let want = RETENTION_BUCKETS * CARD_DEFS.len();
        if world_sampler.contains(&WorldSampler::Retention) && retention.as_ref().map(|r| r.len()) != Some(want) {
            return Err(PyValueError::new_err(format!("the retention sampler needs a table of {want} log-odds")));
        }
        let names = deck_names();
        for (a, b) in &pairings {
            for d in [a, b] {
                if !names.contains(&d.as_str()) {
                    return Err(PyValueError::new_err(format!("unknown deck {d}; known: {names:?}")));
                }
            }
        }
        if pairings.is_empty() {
            return Err(PyValueError::new_err("need at least one pairing"));
        }
        let model_id = |m: &Bound<'_, PyAny>| -> PyResult<u8> {
            if let Ok(s) = m.extract::<String>() {
                return match s.as_str() {
                    "heuristic" => Ok(HEURISTIC),
                    "random" => Ok(RANDOM),
                    _ => Err(PyValueError::new_err(format!("unknown agent {s}"))),
                };
            }
            m.extract::<u8>()
        };
        let cfg = SelfPlayConfig {
            model_sims: [sims, opp_sims.unwrap_or(sims)],
            leaves_per_step,
            temp_decisions,
            search: Config {
                c_puct,
                root_noise,
                dirichlet_alpha,
                fpu_reduction,
                virtual_loss: 1.0,
                maple_dedupe,
                world_candidates,
                world_pool,
                world_temp,
                retention: retention.map(Arc::new),
                ..Config::default()
            },
            maple_worlds: [maple_worlds.0, maple_worlds.1],
            maple_select,
            maple_resample,
            perfect_obs: [perfect_obs.0, perfect_obs.1],
            pimc_worlds: [pimc_worlds.0, pimc_worlds.1],
            world_sampler,
            collisions,
            value_scale,
            heuristic_values,
            value_mix,
            full_search_prob,
            cheap_sims,
            completed_q,
            record_heuristic,
            pairings,
            seat_models: match &seat_models {
                Some((a, b)) => [model_id(a)?, model_id(b)?],
                None => [0, 0],
            },
            record,
            max_games,
            heuristic_sims,
        };
        Ok(SelfPlay { inner: RsSelfPlay::new(parallel, cfg, seed), last_batch: 0, last_worlds: Vec::new() })
    }

    /// Advance all games until they need the network. Returns None when all games
    /// are done, else (tokens, token_len, actions, action_len, model, worlds), where
    /// worlds is None or (tokens, token_len, hist, hist_len, hands, hand_len, model):
    /// hands are [W, M, H] card definitions + 1, hand_len [W, M] (0 past the candidates).
    fn gather<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let inner = &mut self.inner;
        let batch = py.detach(|| inner.gather());
        if batch.is_empty() {
            self.last_batch = 0;
            self.last_worlds.clear();
            return Ok(None);
        }
        let reqs = &batch.leaves;
        self.last_batch = reqs.len();
        let toks: Vec<&Vec<Tok>> = reqs.iter().map(|r| &r.tokens).collect();
        let acts: Vec<&Vec<Act>> = reqs.iter().map(|r| &r.actions).collect();
        let (tflat, tlen, tw) = pad(&toks, 1);
        let (aflat, alen, aw) = pad(&acts, 1);
        let models: Vec<u8> = reqs.iter().map(|r| r.model).collect();
        let n = reqs.len();
        let worlds = &batch.worlds;
        self.last_worlds = worlds.iter().map(|w| w.hands.len()).collect();
        let worlds_out = if worlds.is_empty() {
            None
        } else {
            let w = worlds.len();
            let wt: Vec<&Vec<Tok>> = worlds.iter().map(|x| &x.tokens).collect();
            let wh: Vec<&Vec<Hist>> = worlds.iter().map(|x| &x.hist).collect();
            let (wtflat, wtlen, wtw) = pad(&wt, 1);
            let (hflat, hlen, hw) = pad(&wh, 1);
            let m = worlds.iter().map(|x| x.hands.len()).max().unwrap_or(1).max(1);
            let hmax = worlds.iter().flat_map(|x| x.hands.iter().map(|h| h.len())).max().unwrap_or(1).max(1);
            let mut hands = vec![0u16; w * m * hmax];
            let mut hand_len = vec![0i32; w * m];
            for (i, x) in worlds.iter().enumerate() {
                for (j, h) in x.hands.iter().enumerate() {
                    hand_len[i * m + j] = h.len() as i32;
                    for (k, &d) in h.iter().enumerate() {
                        hands[(i * m + j) * hmax + k] = d + 1;
                    }
                }
            }
            let wmodels: Vec<u8> = worlds.iter().map(|x| x.model).collect();
            Some((
                arr3(py, wtflat, w, wtw, TOKEN_FIELDS)?,
                wtlen.into_pyarray(py),
                arr3(py, hflat, w, hw, HIST_FIELDS)?,
                hlen.into_pyarray(py),
                arr3(py, hands, w, m, hmax)?,
                hand_len.into_pyarray(py).reshape([w, m])?,
                wmodels.into_pyarray(py),
            ))
        };
        let out = (
            arr3(py, tflat, n, tw, TOKEN_FIELDS)?,
            tlen.into_pyarray(py),
            arr3(py, aflat, n, aw, ACTION_FIELDS)?,
            alen.into_pyarray(py),
            models.into_pyarray(py),
            worlds_out,
        );
        Ok(Some(out.into_pyobject(py)?.into_any()))
    }

    /// Evaluations for the last gather: priors [B, A] (any scale, padded
    /// columns ignored), values [B] from the acting player's view, and for
    /// world requests a [W, M] score per candidate hand, lower being more likely.
    #[pyo3(signature = (priors, values, world_scores=None))]
    fn feed(
        &mut self,
        py: Python<'_>,
        priors: PyReadonlyArray2<'_, f32>,
        values: PyReadonlyArray1<'_, f32>,
        world_scores: Option<PyReadonlyArray2<'_, f32>>,
    ) -> PyResult<()> {
        let p = priors.as_array();
        let v = values.as_array();
        if p.shape()[0] != self.last_batch || v.len() != self.last_batch {
            return Err(PyValueError::new_err(format!("expected {} rows", self.last_batch)));
        }
        let evals: Vec<Eval> = (0..self.last_batch)
            .map(|i| Eval { priors: p.row(i).to_vec(), value: v[i] })
            .collect();
        let scores: Vec<Vec<f32>> = match (&world_scores, self.last_worlds.is_empty()) {
            (_, true) => Vec::new(),
            (None, false) => return Err(PyValueError::new_err("world requests need world_scores")),
            (Some(ws), false) => {
                let ws = ws.as_array();
                if ws.shape()[0] != self.last_worlds.len() {
                    return Err(PyValueError::new_err(format!("expected {} world rows", self.last_worlds.len())));
                }
                self.last_worlds.iter().enumerate().map(|(i, &m)| ws.row(i).iter().take(m).copied().collect()).collect()
            }
        };
        let inner = &mut self.inner;
        py.detach(|| inner.feed(evals, scores));
        self.last_batch = 0;
        self.last_worlds.clear();
        Ok(())
    }

    /// Training samples from finished games, or None: (tokens, token_len, actions,
    /// action_len, policy, z, root_value, (hist, hist_len, pool, pool_len, hand_n,
    /// game, turn, heuristic, player, full, anchor)). The pool is the opponent's unseen cards + 1, its hand first.
    /// anchor is None unless seen-hand rows were recorded; then it is the hidden
    /// observation, shaped like tokens with the same token_len (rows recorded
    /// without a seen hand repeat their tokens, which are already hidden).
    fn take_samples<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let s = self.inner.take_samples();
        if s.is_empty() {
            return Ok(None);
        }
        let toks: Vec<&Vec<Tok>> = s.iter().map(|x| &x.tokens).collect();
        let acts: Vec<&Vec<Act>> = s.iter().map(|x| &x.actions).collect();
        let (tflat, tlen, tw) = pad(&toks, 1);
        let (aflat, alen, aw) = pad(&acts, 1);
        let n = s.len();
        let mut policy = vec![0f32; n * aw];
        for (i, x) in s.iter().enumerate() {
            policy[i * aw..i * aw + x.policy.len()].copy_from_slice(&x.policy);
        }
        let z: Vec<f32> = s.iter().map(|x| x.z).collect();
        let rv: Vec<f32> = s.iter().map(|x| x.root_value).collect();
        let hists: Vec<&Vec<Hist>> = s.iter().map(|x| &x.hist).collect();
        let (hflat, hlen, hw) = pad(&hists, 1);
        let pools: Vec<&[u16]> = s.iter().map(|x| x.pool.as_slice()).collect();
        let (pflat, plen, pw) = pad_cards(&pools);
        let hand_n: Vec<u8> = s.iter().map(|x| x.hand_n).collect();
        let game: Vec<u64> = s.iter().map(|x| x.game).collect();
        let turn: Vec<u32> = s.iter().map(|x| x.turn).collect();
        let heuristic: Vec<f32> = s.iter().map(|x| x.heuristic).collect();
        let player: Vec<u8> = s.iter().map(|x| x.player).collect();
        let full: Vec<bool> = s.iter().map(|x| x.full).collect();
        let anchor = if s.iter().all(|x| x.anchor.is_empty()) {
            None
        } else {
            let rows: Vec<&Vec<Tok>> = s
                .iter()
                .map(|x| {
                    if x.anchor.is_empty() {
                        return &x.tokens;
                    }
                    assert_eq!(x.anchor.len(), x.tokens.len(), "the anchor matches the tokens token for token");
                    &x.anchor
                })
                .collect();
            let (aflat, _, w) = pad(&rows, tw);
            Some(arr3(py, aflat, n, w, TOKEN_FIELDS)?)
        };
        let out = (
            arr3(py, tflat, n, tw, TOKEN_FIELDS)?,
            tlen.into_pyarray(py),
            arr3(py, aflat, n, aw, ACTION_FIELDS)?,
            alen.into_pyarray(py),
            policy.into_pyarray(py).reshape([n, aw])?,
            z.into_pyarray(py),
            rv.into_pyarray(py),
            (
                arr3(py, hflat, n, hw, HIST_FIELDS)?,
                hlen.into_pyarray(py),
                pflat.into_pyarray(py).reshape([n, pw])?,
                plen.into_pyarray(py),
                hand_n.into_pyarray(py),
                game.into_pyarray(py),
                turn.into_pyarray(py),
                heuristic.into_pyarray(py),
                player.into_pyarray(py),
                full.into_pyarray(py),
                anchor,
            ),
        );
        Ok(Some(out.into_pyobject(py)?.into_any()))
    }

    /// Finished games: list of dicts (decks, models, winner seat or None, decisions).
    fn take_results<'py>(&mut self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let mut out = Vec::new();
        for r in self.inner.take_results() {
            let d = PyDict::new(py);
            d.set_item("decks", (r.decks[0].clone(), r.decks[1].clone()))?;
            let name = |m: u8| -> Py<PyAny> {
                match m {
                    HEURISTIC => "heuristic".into_pyobject(py).unwrap().into_any().unbind(),
                    RANDOM => "random".into_pyobject(py).unwrap().into_any().unbind(),
                    m => m.into_pyobject(py).unwrap().into_any().unbind(),
                }
            };
            d.set_item("models", (name(r.models[0]), name(r.models[1])))?;
            let winner: Option<u8> = match r.outcome {
                Outcome::Win(p) => Some(p.0),
                _ => None,
            };
            d.set_item("winner", winner)?;
            d.set_item("aborted", r.outcome == Outcome::Aborted)?;
            d.set_item("decisions", r.decisions)?;
            out.push(d);
        }
        Ok(out)
    }

    #[getter]
    fn games_started(&self) -> u64 {
        self.inner.games_started()
    }

    /// Search counters for model ids 0 and 1, summed over finished searches.
    fn search_stats<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let mut out = Vec::new();
        for s in self.inner.stats() {
            let d = PyDict::new(py);
            d.set_item("sims", s.sims)?;
            d.set_item("leaf_worlds", s.leaf_worlds)?;
            d.set_item("unique_leaves", s.unique_leaves)?;
            d.set_item("dropped_illegal", s.dropped_illegal)?;
            d.set_item("dropped_diverged", s.dropped_diverged)?;
            d.set_item("terminal_worlds", s.terminal_worlds)?;
            d.set_item("no_world_sims", s.no_world_sims)?;
            d.set_item("stale_priors", s.stale_priors)?;
            d.set_item("world_requests", s.world_requests)?;
            d.set_item("world_skips", s.world_skips)?;
            d.set_item("leaves", s.leaves)?;
            d.set_item("leaf_decisions", s.leaf_decisions)?;
            d.set_item("leaf_turns", s.leaf_turns)?;
            d.set_item("collisions", s.collisions)?;
            d.set_item("blind_selects", s.blind_selects)?;
            out.push(d);
        }
        Ok(out)
    }
}

/// One game for a person to play: the board and choices as text, and an agent
/// whose search the caller drives with a network (or the heuristic).
#[pyclass(unsendable)]
struct Match {
    game: Game,
    search: Option<Search>,
    leaves: Vec<Game>,
}

#[pymethods]
impl Match {
    #[new]
    #[pyo3(signature = (deck0="Ognis", deck1="Ognis", seed=0))]
    fn new(deck0: &str, deck1: &str, seed: u64) -> PyResult<Self> {
        let game = Game::new(deck0, deck1, seed).map_err(PyValueError::new_err)?;
        Ok(Match { game, search: None, leaves: Vec::new() })
    }

    /// The seat (0 or 1) to act.
    fn to_act(&self) -> u8 {
        self.game.to_act().index() as u8
    }

    fn is_over(&self) -> bool {
        self.game.is_over()
    }

    /// "win 0", "win 1", "draw" or "aborted: why"; None while playing.
    fn outcome(&self) -> Option<String> {
        self.game.outcome().map(|o| match o {
            Outcome::Win(p) => format!("win {}", p.index()),
            Outcome::Draw => "draw".into(),
            Outcome::Aborted => format!("aborted: {}", self.game.abort_reason.clone().unwrap_or_default()),
        })
    }

    /// The board as seat `viewer` sees it.
    fn render(&self, viewer: u8) -> String {
        play::render(&self.game, seat(viewer))
    }

    /// Each choice of the player to act, as seat `viewer` may read it.
    fn choices(&self, viewer: u8) -> Vec<String> {
        (0..self.game.choices().len()).map(|i| play::describe(&self.game, i, seat(viewer))).collect()
    }

    fn apply(&mut self, i: usize) -> PyResult<()> {
        if i >= self.game.choices().len() {
            return Err(PyValueError::new_err(format!("choice {i} of {}", self.game.choices().len())));
        }
        self.search = None;
        self.leaves.clear();
        self.game.apply(i);
        Ok(())
    }

    /// The heuristic agent's choice: IS-MCTS over `sims` simulations.
    fn heuristic_choice(&self, sims: u32, seed: u64) -> usize {
        let mut s = Search::new(&self.game, Config::default(), seed);
        s.run(sims, 8, &mut HeuristicEval);
        s.best()
    }

    /// Starts the agent's search at the current decision.
    fn start_search(&mut self, seed: u64) {
        self.search = Some(Search::new(&self.game, Config::default(), seed));
        self.leaves.clear();
    }

    fn simulations(&self) -> u32 {
        self.search.as_ref().map_or(0, |s| s.simulations)
    }

    /// Up to `n` leaves to evaluate as (tokens, token_len, actions, action_len),
    /// or None when every simulation in this batch ended the game.
    fn gather<'py>(&mut self, py: Python<'py>, n: usize) -> PyResult<Option<Bound<'py, PyAny>>> {
        let search = self.search.as_mut().ok_or_else(|| PyValueError::new_err("start_search first"))?;
        self.leaves = search.gather(n);
        if self.leaves.is_empty() {
            return Ok(None);
        }
        let toks: Vec<Vec<Tok>> = self.leaves.iter().map(|l| observe_with(l, l.to_act(), false)).collect();
        let acts: Vec<Vec<Act>> = self.leaves.iter().map(choice_fields).collect();
        let (tflat, tlen, tw) = pad(&toks.iter().collect::<Vec<_>>(), 1);
        let (aflat, alen, aw) = pad(&acts.iter().collect::<Vec<_>>(), 1);
        let n = self.leaves.len();
        let out = (arr3(py, tflat, n, tw, TOKEN_FIELDS)?, tlen.into_pyarray(py), arr3(py, aflat, n, aw, ACTION_FIELDS)?, alen.into_pyarray(py));
        Ok(Some(out.into_pyobject(py)?.into_any()))
    }

    /// The network's priors [B, A] and values [B] for the last gather.
    fn feed(&mut self, priors: PyReadonlyArray2<'_, f32>, values: PyReadonlyArray1<'_, f32>) -> PyResult<()> {
        let search = self.search.as_mut().ok_or_else(|| PyValueError::new_err("start_search first"))?;
        let (p, v) = (priors.as_array(), values.as_array());
        if p.shape()[0] != self.leaves.len() || v.len() != self.leaves.len() {
            return Err(PyValueError::new_err(format!("expected {} rows", self.leaves.len())));
        }
        let evals = (0..self.leaves.len()).map(|i| Eval { priors: p.row(i).to_vec(), value: v[i] }).collect();
        let leaves = std::mem::take(&mut self.leaves);
        search.feed(&leaves, evals);
        Ok(())
    }

    /// The searched choice, with the root's visit shares and value.
    fn best(&self) -> PyResult<(usize, Vec<f32>, f32)> {
        let s = self.search.as_ref().ok_or_else(|| PyValueError::new_err("start_search first"))?;
        let visits = s.root_visits();
        let total: f32 = visits.iter().sum::<f32>().max(1.0);
        Ok((s.best(), visits.iter().map(|v| v / total).collect(), s.root_value()))
    }
}

fn seat(i: u8) -> maplemagic::mtg_kernel::ids::PlayerId {
    if i == 0 {
        maplemagic::mtg_kernel::ids::PlayerId::P0
    } else {
        maplemagic::mtg_kernel::ids::PlayerId::P1
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SelfPlay>()?;
    m.add_class::<Match>()?;
    m.add("TOKEN_FIELDS", TOKEN_FIELDS)?;
    m.add("ACTION_FIELDS", ACTION_FIELDS)?;
    m.add("MAX_TOKENS", MAX_TOKENS)?;
    m.add("TOKEN_VOCAB", token_vocab().to_vec())?;
    m.add("TOKEN_BITS", token_bits().to_vec())?;
    m.add("LEGACY_TOKEN_FIELDS", LEGACY_TOKEN_FIELDS)?;
    m.add("STATIC_FIELDS", STATIC_FIELDS)?;
    m.add("STATIC_BITS", static_bits().to_vec())?;
    m.add("STATIC_VOCAB", static_vocab().to_vec())?;
    // Each token card's printed attributes, by token field 0 (row 0: hidden).
    m.add("CARD_STATIC", card_static_table().into_iter().flatten().collect::<Vec<u16>>())?;
    m.add("ACTION_VOCAB", action_vocab().to_vec())?;
    m.add("DECKS", deck_names())?;
    m.add("DECK_PAIRS", deck_pairs())?;
    let commander = maplemagic::commander::list_names();
    m.add("COMMANDER_DECKS", deck_names().into_iter().filter(|n| commander.contains(n)).collect::<Vec<_>>())?;
    m.add("HIST_FIELDS", HIST_FIELDS)?;
    m.add("MAX_HIST", MAX_HIST)?;
    m.add("HIST_VOCAB", hist_vocab().to_vec())?;
    m.add("NUM_CARDS", CARD_DEFS.len())?;
    m.add("CARD_NAMES", CARD_DEFS.iter().map(|d| d.name).collect::<Vec<_>>())?;
    let kind = |d: &maplemagic::mtg_kernel::card_def::CardDef| {
        if d.is_land {
            0u8
        } else if d.has_type(CardType::Instant) {
            1
        } else if d.has_type(CardType::Creature) {
            2
        } else {
            3
        }
    };
    m.add("CARD_KINDS", CARD_DEFS.iter().map(kind).collect::<Vec<u8>>())?;
    m.add("RETENTION_BUCKETS", RETENTION_BUCKETS)?;
    Ok(())
}
