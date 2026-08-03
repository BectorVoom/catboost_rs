//! Shared estimator base logic: the verbatim kwargs store (D-06) + the fitted
//! model handle.
//!
//! `__init__` stores constructor kwargs verbatim (D-06: NO work / validation /
//! coercion). The param-vocabulary registry, alias handling, the
//! kwargs -> [`CatBoostBuilder`] map, and unknown/unsupported-param rejection
//! (D-05 / D-07) live in [`crate::params`] and run at `fit()` time.

use std::collections::BTreeMap;
use std::path::Path;

use catboost_rs::{CatBoostBuilder, CatBoostError, IngestSource, Model, Pool};
use numpy::{PyArray1, ToPyArray};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::errors::{CatBoostValueError, PyCbError};
use crate::ingest_py::ingest_to_owned;

/// Shared estimator state: kwargs stored verbatim (D-06) + the fitted model
/// (`None` until `fit` runs — the not-fitted sentinel; the typed `NotFittedError`
/// lands in 08-05).
pub(crate) struct EstimatorBase {
    /// Constructor kwargs, stored exactly as received (D-06). Keyed by name so
    /// `get_params`/`set_params` round-trip in later plans.
    pub(crate) params: BTreeMap<String, Py<PyAny>>,
    /// The fitted model; `None` means not-yet-fitted.
    pub(crate) model: Option<Model>,
    /// The categorical column indices this estimator was FITTED with (F17).
    ///
    /// Remembered because predict MUST declare the same categorical width: after
    /// F10 the model checks the pool's declared width against the width recorded
    /// at fit time, so a predict pool ingested without `cat_features` would be
    /// rejected. Empty for a float-only fit.
    pub(crate) cat_features: Vec<usize>,
}

impl EstimatorBase {
    /// Build an empty (unfitted) base from optional `**kwargs`, storing every key
    /// verbatim. No validation or coercion happens here (D-06).
    ///
    /// # Errors
    /// Propagates any failure extracting a kwargs key as a string.
    pub(crate) fn from_kwargs(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut params = BTreeMap::new();
        if let Some(dict) = kwargs {
            for (key, value) in dict.iter() {
                let name: String = key.extract()?;
                params.insert(name, value.unbind());
            }
        }
        Ok(Self {
            params,
            model: None,
            cat_features: Vec::new(),
        })
    }

    /// Build a fitted base directly from a loaded [`Model`] (no kwargs), used by
    /// the `load_model` constructors. `params` is empty (the loaded model already
    /// embeds its trained configuration); `model` is `Some(model)`.
    #[must_use]
    pub(crate) fn from_model(model: Model) -> Self {
        Self {
            params: BTreeMap::new(),
            model: Some(model),
            // A loaded model carries no fit-time categorical record: neither the
            // .cbm nor the JSON codec stores `cat_feature_count` (it is
            // runtime-only, F08), so `load_model` + `predict` works for
            // float-only models and reports a typed width mismatch otherwise.
            cat_features: Vec::new(),
        }
    }

    /// Return the verbatim constructor kwargs as a fresh `dict` (the sklearn
    /// `get_params` contract). The store is keyed by the EXACT name the user passed
    /// (D-06), so `set_params(**get_params())` is an identity round-trip and
    /// `sklearn.base.clone` (which does `__init__(**get_params())`) reconstructs an
    /// equal-params unfitted estimator (T-08-15). `deep` is accepted for sklearn
    /// signature parity; there are no nested sub-estimators, so it is a no-op.
    ///
    /// # Errors
    /// Propagates any failure cloning a stored value into the new dict.
    pub(crate) fn get_params<'py>(
        &self,
        py: Python<'py>,
        _deep: Option<bool>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (name, value) in &self.params {
            dict.set_item(name, value.bind(py))?;
        }
        Ok(dict)
    }

    /// Merge `**params` into the verbatim store (the sklearn `set_params`
    /// contract). Each key overwrites verbatim; no validation or coercion happens
    /// here (validation stays at `fit`, D-06). Keys NOT already present are still
    /// accepted (sklearn's `set_params` allows setting any valid `__init__` param).
    ///
    /// # Errors
    /// Propagates any failure extracting a key as a string.
    pub(crate) fn set_params(&mut self, params: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        if let Some(dict) = params {
            for (key, value) in dict.iter() {
                let name: String = key.extract()?;
                self.params.insert(name, value.unbind());
            }
        }
        Ok(())
    }

    /// `true` once `fit` (or `load_model`) has populated the model handle.
    #[must_use]
    pub(crate) fn is_fitted(&self) -> bool {
        self.model.is_some()
    }
}

/// Build the sklearn ≥1.6 `Tags` dataclass for an estimator of `estimator_type`
/// (`"classifier"` | `"regressor"`). sklearn 1.6 replaced the old `_get_tags()`
/// dict with the `__sklearn_tags__()` dataclass (RESEARCH Pitfall 5); modern
/// `check_estimator` reads `estimator_type` and the per-kind sub-tags off this
/// object. We construct it by calling into Python (`sklearn.utils.Tags` +
/// `TargetTags`/`ClassifierTags`/`RegressorTags`/`InputTags`) so we always match
/// the installed sklearn's exact dataclass shape rather than hard-coding fields.
///
/// `required=True` on the target tags marks the estimators as supervised (both
/// the classifier and regressor require `y` at fit). The Ranker presents with the
/// regressor-like `"regressor"` tag set (continuous score output) per RESEARCH
/// Open Q2; it is EXCLUDED from the `check_estimator` gate (Task 2).
///
/// # Errors
/// Propagates any failure importing `sklearn.utils` or constructing the dataclass.
pub(crate) fn build_sklearn_tags<'py>(
    py: Python<'py>,
    estimator_type: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let utils = py.import(intern!(py, "sklearn.utils"))?;
    let tags_cls = utils.getattr(intern!(py, "Tags"))?;
    let target_tags = utils
        .getattr(intern!(py, "TargetTags"))?
        .call1((true,))?; // TargetTags(required=True)
    let input_tags = utils.getattr(intern!(py, "InputTags"))?.call0()?;

    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "estimator_type"), estimator_type)?;
    kwargs.set_item(intern!(py, "target_tags"), target_tags)?;
    kwargs.set_item(intern!(py, "input_tags"), input_tags)?;
    if estimator_type == "classifier" {
        let clf = utils.getattr(intern!(py, "ClassifierTags"))?.call0()?;
        kwargs.set_item(intern!(py, "classifier_tags"), clf)?;
    } else {
        let reg = utils.getattr(intern!(py, "RegressorTags"))?.call0()?;
        kwargs.set_item(intern!(py, "regressor_tags"), reg)?;
    }
    tags_cls.call((), Some(&kwargs))
}

/// Coefficient of determination R² of `pred` vs the true `y` (the sklearn
/// `RegressorMixin.score` default). `R² = 1 - SS_res / SS_tot`; when `SS_tot == 0`
/// (constant `y`) sklearn returns `0.0` for a non-perfect fit, `1.0` for a perfect
/// one — mirror that.
#[must_use]
pub(crate) fn r2_score(y: &[f64], pred: &[f64]) -> f64 {
    let n = y.len();
    if n == 0 {
        return 0.0;
    }
    let mean = y.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = y.iter().map(|v| (v - mean).powi(2)).sum();
    let ss_res: f64 = y
        .iter()
        .zip(pred.iter())
        .map(|(t, p)| (t - p).powi(2))
        .sum();
    if ss_tot == 0.0 {
        return if ss_res == 0.0 { 1.0 } else { 0.0 };
    }
    1.0 - ss_res / ss_tot
}

/// Mean accuracy of class predictions vs `y` (the sklearn
/// `ClassifierMixin.score` default). Labels are compared after rounding the
/// f64 predictions to the nearest integer (the classifier emits `0.0`/`1.0`).
#[must_use]
pub(crate) fn accuracy_score(y: &[f64], pred: &[f64]) -> f64 {
    let n = y.len();
    if n == 0 {
        return 0.0;
    }
    // Compare the rounded labels as integers (the intent is integer equality).
    // The previous `< f64::EPSILON` form was correct only for 0.0/1.0 binary
    // labels — EPSILON (~2.2e-16) is the representable gap near 1.0, so equal
    // integer-valued f64s with magnitude > ~2 could compare unequal, and it is a
    // latent bug for multiclass labels (WR-06). A non-finite (NaN) rounded value
    // never matches (the `i64` guard short-circuits via the finite check).
    let correct = y
        .iter()
        .zip(pred.iter())
        .filter(|(t, p)| {
            let (tr, pr) = (t.round(), p.round());
            tr.is_finite() && pr.is_finite() && (tr as i64) == (pr as i64)
        })
        .count();
    correct as f64 / n as f64
}

/// Build a [`CatBoostBuilder`] from the params and fit it on an OWNED learn pool
/// with OWNED eval pools (PARAM-02). The caller is expected to invoke this under
/// `py.detach` (every pool is owned, so no Python buffer borrow is alive — D-11).
/// Returns the typed facade [`CatBoostError`]; the caller maps it via
/// `errors::to_pyerr` (PYAPI-05).
///
/// An EMPTY `eval_pools` is a plain learn-only fit: the facade's
/// `fit_with_eval_sets` and `fit` share one inner, so passing no eval set is
/// byte-identical to calling `fit` directly.
///
/// # Errors
/// Returns the facade [`CatBoostError`] training error (including
/// `FeatureMismatch` when an eval pool's width disagrees with the learn pool's).
pub(crate) fn fit_pool_with_eval(
    builder: CatBoostBuilder,
    pool: &Pool,
    eval_pools: &[Pool],
) -> Result<Model, CatBoostError> {
    let refs: Vec<&Pool> = eval_pools.iter().collect();
    builder.fit_with_eval_sets(pool, &refs).map(|out| out.model)
}

/// Ingest the `eval_set` fit kwarg into owned [`Pool`]s (PARAM-02).
///
/// Upstream accepts any of:
///  - a single `Pool`;
///  - a single `(X, y)` tuple;
///  - a LIST of either.
///
/// All four shapes are accepted here and normalized to a vector of owned pools,
/// ingested through the SAME [`data_to_pool`] path as the learn set so an eval
/// set gets identical dtype / layout / nullability validation (D-12).
///
/// `cat_features` is threaded through so an eval set declares the same
/// categorical columns the learn set does — the facade rejects a width mismatch,
/// and without this an eval set built from a DataFrame would silently arrive
/// float-only.
///
/// # Errors
/// [`CatBoostValueError`] on any shape the four forms above do not cover, or any
/// ingestion failure of a member.
pub(crate) fn eval_set_to_pools(
    py: Python<'_>,
    eval_set: &Bound<'_, PyAny>,
    cat_features: &[usize],
) -> PyResult<Vec<Pool>> {
    if eval_set.is_none() {
        return Ok(Vec::new());
    }
    // A `Pool` is itself iterable-looking to nothing, but a TUPLE is: check the
    // single-Pool and single-tuple forms BEFORE the list form, otherwise
    // `(X, y)` would be read as "a list of two eval sets".
    if eval_set.cast::<crate::pool::Pool>().is_ok() {
        return Ok(vec![one_eval_pool(py, eval_set, cat_features)?]);
    }
    if let Ok(tuple) = eval_set.cast::<pyo3::types::PyTuple>() {
        if tuple.len() == 2 {
            return Ok(vec![one_eval_pool(py, eval_set, cat_features)?]);
        }
        return Err(CatBoostValueError::new_err(format!(
            "eval_set tuple must be (X, y), got {} elements",
            tuple.len()
        )));
    }
    if let Ok(list) = eval_set.cast::<pyo3::types::PyList>() {
        let mut pools = Vec::with_capacity(list.len());
        for item in list.iter() {
            pools.push(one_eval_pool(py, &item, cat_features)?);
        }
        return Ok(pools);
    }
    Err(CatBoostValueError::new_err(
        "eval_set must be a Pool, an (X, y) tuple, or a list of either",
    ))
}

/// Ingest ONE eval-set member (a `Pool` or an `(X, y)` tuple) into an owned pool.
fn one_eval_pool(
    py: Python<'_>,
    item: &Bound<'_, PyAny>,
    cat_features: &[usize],
) -> PyResult<Pool> {
    if item.cast::<crate::pool::Pool>().is_ok() {
        // A Pool already declares its own categorical columns; passing
        // `cat_features` alongside one is the same ambiguity `data_to_pool`
        // rejects for the learn set, so route through it with no declaration.
        return data_to_pool(py, item, None, None);
    }
    let tuple = item.cast::<pyo3::types::PyTuple>().map_err(|_| {
        CatBoostValueError::new_err(
            "each eval_set entry must be a Pool or an (X, y) tuple",
        )
    })?;
    if tuple.len() != 2 {
        return Err(CatBoostValueError::new_err(format!(
            "eval_set tuple must be (X, y), got {} elements",
            tuple.len()
        )));
    }
    let x = tuple.get_item(0)?;
    let y = tuple.get_item(1)?;
    data_to_pool(py, &x, Some(&y), Some(cat_features))
}

/// Ingest SCORING data (`predict` / `predict_proba` / `score` /
/// `partial_dependence`) into an owned pool, declaring the estimator's
/// REMEMBERED fit-time `cat_features`.
///
/// The distinction from [`data_to_pool`] is whose `cat_features` it is.
/// `data_to_pool` raises when a non-empty `cat_features` accompanies a `Pool`,
/// because a `Pool` already declares its own categorical columns and upstream
/// treats the combination as ambiguous — but that rule is about an argument the
/// CALLER passed. On the scoring paths the value is not an argument at all: it is
/// the estimator's own record of what it was fit on, filled in automatically.
/// Routing it through `data_to_pool` therefore made
/// `clf.fit(df, y, cat_features=[0]); clf.predict(Pool(df_test, cat_features=[0]))`
/// raise "cat_features cannot be given when the data is a Pool" for a call that
/// passed no `cat_features` whatsoever — a sequence upstream accepts.
///
/// A `Pool` argument is therefore ingested with NO declaration (it is the single
/// source of truth for its own categorical columns, exactly as
/// [`one_eval_pool`] already does for an eval-set member); anything else keeps
/// the remembered declaration, which is what makes a raw frame ingest with the
/// same categorical layout it was fit with.
///
/// # Errors
/// [`CatBoostValueError`] on any dtype / layout / length / nullability failure,
/// via [`data_to_pool`].
pub(crate) fn scoring_data_to_pool(
    py: Python<'_>,
    x: &Bound<'_, PyAny>,
    cat_features: &[usize],
) -> PyResult<Pool> {
    if x.cast::<crate::pool::Pool>().is_ok() {
        return data_to_pool(py, x, None, None);
    }
    data_to_pool(py, x, None, Some(cat_features))
}

/// Resolve the categorical column indices for one `fit()` call (F17).
///
/// Upstream accepts `cat_features` BOTH as a constructor kwarg and as a `fit()`
/// kwarg; the `fit()` argument wins when both are given. Returns an empty vector
/// for a float-only fit.
///
/// # Errors
/// [`CatBoostValueError`] if the constructor's `cat_features` is not a list of
/// non-negative integers.
pub(crate) fn resolve_cat_features(
    params: &BTreeMap<String, Py<PyAny>>,
    py: Python<'_>,
    fit_kwarg: Option<Vec<usize>>,
) -> PyResult<Vec<usize>> {
    if let Some(from_fit) = fit_kwarg {
        return Ok(from_fit);
    }
    match params.get("cat_features") {
        None => Ok(Vec::new()),
        Some(obj) => obj.bind(py).extract::<Vec<usize>>().map_err(|_| {
            CatBoostValueError::new_err(
                "cat_features must be a list of non-negative column indices, e.g. [0, 3]",
            )
        }),
    }
}

/// Build a facade [`Pool`] from `x` (+ optional `y`), accepting EITHER a native
/// [`crate::pool::Pool`] OR a framework object (NumPy / Pandas / Arrow / Polars).
///
/// Shared by `CatBoostRegressor`, `CatBoostClassifier`, and `CatBoostRanker` so
/// the three estimators ingest identically (prep for the 08-05 sklearn contract).
///
/// When `x` is a `Pool`, its inherited `into_pool()` validation runs (and any `y`
/// is ignored — the Pool already carries its label). Otherwise `x`/`y` route
/// through the shared ingest adapter. In both cases the result is fully owned, so
/// the caller may `py.detach()` immediately (D-11 / PYAPI-06).
///
/// # Error-surface asymmetry (WR-04)
///
/// The two input kinds validate at DIFFERENT points, by design:
/// - A NumPy / Pandas / Arrow / Polars `x` runs the strict D-12 input checks
///   (float32 / contiguity / nullability) eagerly during ingestion here.
/// - A native `Pool` already had those checks run at its OWN construction, so the
///   fast-path runs only `OwnedColumns::into_pool()`'s length check. A
///   feature-width mismatch against the fitted model is therefore NOT caught here;
///   it surfaces later as the facade's `FeatureMismatch` inside `predict_with`
///   (still a typed error, just raised deeper in the call stack).
///
/// Additionally, on the `Pool` fast-path the `y` argument is IGNORED — the `Pool`
/// already carries its own label, so a `y` passed alongside a `Pool` is silently
/// dropped (the Pool is the single source of truth).
///
/// # Errors
/// [`CatBoostValueError`] on any dtype / layout / length / nullability failure.
pub(crate) fn data_to_pool(
    py: Python<'_>,
    x: &Bound<'_, PyAny>,
    y: Option<&Bound<'_, PyAny>>,
    cat_features: Option<&[usize]>,
) -> PyResult<Pool> {
    if let Ok(pool_ref) = x.cast::<crate::pool::Pool>() {
        // Pool fast-path (WR-04): `y` is intentionally ignored (the Pool carries its
        // own label) and only the inherited length check runs here — a feature-width
        // mismatch defers to the facade's `FeatureMismatch` inside `predict_with`.
        //
        // F17 / OQ-3 (upstream-exact, core.py:1522-1533): a `Pool` already
        // declares its own categorical columns, so combining it with an explicit
        // `cat_features` is ambiguous and upstream RAISES. Silently preferring
        // one over the other would be exactly the kind of ignored argument the
        // honesty policy forbids.
        if cat_features.is_some_and(|c| !c.is_empty()) {
            return Err(CatBoostValueError::new_err(
                "cat_features cannot be given when the data is a Pool: the Pool \
                 already declares its categorical columns (construct it with \
                 `Pool(..., cat_features=[...])` instead)",
            ));
        }
        return pool_ref.borrow().to_pool();
    }

    // MINOR-11: de-duplicate and range-check BEFORE ingestion. Without this,
    // `cat_features=[2, 2]` declares "2 categorical columns" and then produces
    // one, so the width guard below mis-reports "declared 2 ... carries 1"
    // instead of naming the real problem (the duplicate).
    let declared: Vec<usize> = match cat_features {
        None => Vec::new(),
        Some(list) => {
            let mut seen = std::collections::BTreeSet::new();
            for &idx in list {
                if !seen.insert(idx) {
                    return Err(CatBoostValueError::new_err(format!(
                        "cat_features contains duplicate column index {idx}"
                    )));
                }
            }
            seen.into_iter().collect()
        }
    };

    let owned = ingest_to_owned(py, x, y, Some(&declared))?;
    let pool = owned
        .into_pool()
        .map_err(|e| CatBoostValueError::new_err(e.to_string()))?;

    // F17 / Finding F2: `ingest_to_owned`'s NumPy branch calls
    // `numpy_to_owned(x, y)` and DROPS `cat_features` entirely, so a user who
    // passes `cat_features=[3]` with a NumPy matrix would otherwise train a
    // float-only model and never learn that the argument did nothing. The guard
    // lives HERE rather than in `ingest_py.rs`, which SPEC §7 lists as
    // verification-only.
    if pool.n_cat_features() != declared.len() {
        return Err(CatBoostValueError::new_err(format!(
            "cat_features declared {} categorical column(s) but the ingested data \
             carries {}; the NumPy ingestion path cannot carry categorical columns \
             (its dtype is float32) — pass a Pandas DataFrame, an Arrow/Polars \
             table, or a `Pool` constructed with `cat_features=[...]`",
            declared.len(),
            pool.n_cat_features()
        )));
    }
    Ok(pool)
}

/// Shared FSTR-03 partial-dependence adapter for the estimators. Ingests `x` into
/// an owned [`Pool`] under the GIL (D-11), releases the GIL for the compute
/// (`py.detach`), and returns a dict `{features: list[int], grids:
/// list[np.ndarray[f64]], values: np.ndarray[f64]}` — `values` row-major over the
/// Cartesian product of `grids` (first feature outer). Mirrors the `predict`
/// adapter shape.
///
/// # Errors
/// [`CatBoostValueError`] on a bad `x` (dtype/layout) or an invalid partial-
/// dependence request (bad arity / out-of-range / duplicate feature / empty
/// dataset), via [`crate::errors::to_pyerr`].
pub(crate) fn partial_dependence_py<'py>(
    model: &Model,
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    features: Vec<usize>,
    cat_features: &[usize],
) -> PyResult<Bound<'py, PyDict>> {
    // --- GIL HELD: own the input before any detach (D-11) ---
    let pool = scoring_data_to_pool(py, x, cat_features)?;
    // --- owned data only: safe to release the GIL for the compute ---
    let pd = py
        .detach(|| model.partial_dependence(&pool, &features))
        .map_err(PyCbError)?;
    let dict = PyDict::new(py);
    dict.set_item(intern!(py, "features"), pd.features)?;
    let grids: Vec<Bound<'py, PyArray1<f64>>> =
        pd.grids.iter().map(|g| g.to_pyarray(py)).collect();
    dict.set_item(intern!(py, "grids"), grids)?;
    dict.set_item(intern!(py, "values"), pd.values.to_pyarray(py))?;
    Ok(dict)
}

/// Load a reference model from `path`, dispatching on the file extension: a
/// `.json` path loads via [`Model::load_json`], anything else (notably `.cbm`)
/// loads via [`Model::load_cbm`]. Shared by the `load_model` constructors on the
/// regressor and classifier (the single deterministic oracle path, Path (a)).
///
/// A malformed model surfaces as the facade `CatBoostError::Deserialize` /
/// `SchemaVersion`, mapped by [`crate::errors::to_pyerr`] to `CatBoostValueError`
/// (threat T-08-12) — never a panic.
///
/// # Errors
/// `CatBoostValueError` (via [`crate::errors::PyCbError`]) on a malformed /
/// unreadable model file.
pub(crate) fn load_model_path(path: &str) -> PyResult<Model> {
    let p = Path::new(path);
    let model = if p.extension().and_then(|e| e.to_str()) == Some("json") {
        Model::load_json(p)
    } else {
        Model::load_cbm(p)
    };
    model.map_err(|e| crate::errors::PyCbError(e).into())
}
