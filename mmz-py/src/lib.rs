//! Python bindings, `mymagezero._core`. Observations and choices cross as
//! zero-padded uint16 numpy arrays with a length per row.

use mmz::features::{action_vocab, token_vocab, ACTION_FIELDS, MAX_TOKENS, TOKEN_FIELDS};
use mmz::game::{deck_names, Outcome};
use mmz::ismcts::{Config, Eval, MapleSelect};
use mmz::selfplay::{SelfPlay as RsSelfPlay, SelfPlayConfig, HEURISTIC, RANDOM};
use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

type Tok = [u16; TOKEN_FIELDS];
type Act = [u16; ACTION_FIELDS];

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
}

#[pymethods]
impl SelfPlay {
    /// seat_models: 0/1 for networks, or "heuristic" / "random". sims is model 0's
    /// budget, opp_sims model 1's. Search settings are pairs, one per model id;
    /// maple_select and maple_resample may also be one value for both.
    #[new]
    #[pyo3(signature = (parallel, pairings, sims=200, leaves_per_step=4, temp_decisions=30, c_puct=1.0, root_noise=0.25, dirichlet_alpha=0.3, fpu_reduction=0.2, seat_models=None, record=true, max_games=None, heuristic_sims=200, seed=0, opp_sims=None, maple_worlds=(0, 0), maple_select=None, maple_resample=None, maple_dedupe=true, perfect_obs=(false, false), pimc_worlds=(0, 0)))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        parallel: usize,
        pairings: Vec<(String, String)>,
        sims: u32,
        leaves_per_step: usize,
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
    ) -> PyResult<Self> {
        if (maple_worlds.0 > 0 && pimc_worlds.0 > 0) || (maple_worlds.1 > 0 && pimc_worlds.1 > 0) {
            return Err(PyValueError::new_err("a model can't use both maple_worlds and pimc_worlds"));
        }
        let maple_select = per_model(maple_select.as_ref(), MapleSelect::RefWorld, parse_select)?;
        let maple_resample = per_model(maple_resample.as_ref(), false, |v| v.extract::<bool>().map_err(Into::into))?;
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
                ..Config::default()
            },
            maple_worlds: [maple_worlds.0, maple_worlds.1],
            maple_select,
            maple_resample,
            perfect_obs: [perfect_obs.0, perfect_obs.1],
            pimc_worlds: [pimc_worlds.0, pimc_worlds.1],
            pairings,
            seat_models: match &seat_models {
                Some((a, b)) => [model_id(a)?, model_id(b)?],
                None => [0, 0],
            },
            record,
            max_games,
            heuristic_sims,
        };
        Ok(SelfPlay { inner: RsSelfPlay::new(parallel, cfg, seed), last_batch: 0 })
    }

    /// Advance all games until they need evaluations. Returns None when all
    /// games are done, else (tokens, token_len, actions, action_len, model).
    fn gather<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let inner = &mut self.inner;
        let reqs = py.detach(|| inner.gather());
        self.last_batch = reqs.len();
        if reqs.is_empty() {
            return Ok(None);
        }
        let toks: Vec<&Vec<Tok>> = reqs.iter().map(|r| &r.tokens).collect();
        let acts: Vec<&Vec<Act>> = reqs.iter().map(|r| &r.actions).collect();
        let (tflat, tlen, tw) = pad(&toks, 1);
        let (aflat, alen, aw) = pad(&acts, 1);
        let models: Vec<u8> = reqs.iter().map(|r| r.model).collect();
        let n = reqs.len();
        let out = (
            arr3(py, tflat, n, tw, TOKEN_FIELDS)?,
            tlen.into_pyarray(py),
            arr3(py, aflat, n, aw, ACTION_FIELDS)?,
            alen.into_pyarray(py),
            models.into_pyarray(py),
        );
        Ok(Some(out.into_pyobject(py)?.into_any()))
    }

    /// Evaluations for the last gather: priors [B, A] (any scale, padded
    /// columns ignored) and values [B] from the acting player's view.
    fn feed(&mut self, py: Python<'_>, priors: PyReadonlyArray2<'_, f32>, values: PyReadonlyArray1<'_, f32>) -> PyResult<()> {
        let p = priors.as_array();
        let v = values.as_array();
        if p.shape()[0] != self.last_batch || v.len() != self.last_batch {
            return Err(PyValueError::new_err(format!("expected {} rows", self.last_batch)));
        }
        let evals: Vec<Eval> = (0..self.last_batch)
            .map(|i| Eval { priors: p.row(i).to_vec(), value: v[i] })
            .collect();
        let inner = &mut self.inner;
        py.detach(|| inner.feed(evals));
        self.last_batch = 0;
        Ok(())
    }

    /// Training samples from finished games:
    /// (tokens, token_len, actions, action_len, policy, z, root_value) or None.
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
        let out = (
            arr3(py, tflat, n, tw, TOKEN_FIELDS)?,
            tlen.into_pyarray(py),
            arr3(py, aflat, n, aw, ACTION_FIELDS)?,
            alen.into_pyarray(py),
            policy.into_pyarray(py).reshape([n, aw])?,
            z.into_pyarray(py),
            rv.into_pyarray(py),
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
            out.push(d);
        }
        Ok(out)
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SelfPlay>()?;
    m.add("TOKEN_FIELDS", TOKEN_FIELDS)?;
    m.add("ACTION_FIELDS", ACTION_FIELDS)?;
    m.add("MAX_TOKENS", MAX_TOKENS)?;
    m.add("TOKEN_VOCAB", token_vocab().to_vec())?;
    m.add("ACTION_VOCAB", action_vocab().to_vec())?;
    m.add("DECKS", deck_names())?;
    Ok(())
}
