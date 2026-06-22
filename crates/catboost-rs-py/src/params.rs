//! The D-07 param-vocabulary registry + the kwargs -> [`CatBoostBuilder`] map.
//!
//! # Honesty policy (D-05 / D-07, threat T-08-05)
//!
//! A user kwarg is one of three things:
//!
//! 1. **IMPLEMENTED** — it has a matching [`CatBoostBuilder`] setter; we apply it.
//! 2. **KNOWN_NOT_YET** — it is a real upstream `CatBoostClassifier.__init__`
//!    parameter (the AUTHORITATIVE list transcribed from
//!    `catboost-master/catboost/python-package/catboost/core.py:5333`) for which
//!    catboost-rs has no setter yet. We REJECT it at `fit()` with a
//!    `CatBoostParameterError` flagging it as a parity gap — never silently ignore
//!    it (which would train a silently-wrong model).
//! 3. **UNKNOWN** — not in the upstream vocabulary at all (likely a typo). We
//!    REJECT it and suggest the closest vocabulary entry (Levenshtein).
//!
//! Validation runs at `fit()` time (D-06), NOT in `__init__`, so the sklearn
//! "no work in `__init__`" contract (08-05) holds.

use std::collections::BTreeMap;

use catboost_rs::{CatBoostBuilder, EBootstrapType, EScoreFunction, LeafMethod, Loss};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::errors::CatBoostParameterError;

/// Whether a registry param is wired to a [`CatBoostBuilder`] setter or is a
/// known-but-unimplemented upstream parameter (a parity gap).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ParamStatus {
    /// Wired to a builder setter; applied at fit().
    Implemented,
    /// A real upstream parameter with no setter yet (rejected at fit() as a
    /// parity gap).
    KnownNotYet,
}

/// The IMPLEMENTED canonical params — each has a matching [`CatBoostBuilder`]
/// setter applied in [`build_and_fit`]. Used both for the registry tag and as the
/// alias-resolution target set.
const IMPLEMENTED: &[&str] = &[
    "iterations",
    "learning_rate",
    "depth",
    "l2_leaf_reg",
    "loss_function",
    "border_count",
    "random_seed",
    "random_strength",
    "bagging_temperature",
    "bootstrap_type",
    "subsample",
    "score_function",
    "boost_from_average",
    "leaf_estimation_method",
];

/// The full upstream `CatBoostClassifier.__init__` vocabulary (the AUTHORITATIVE
/// 119-name list transcribed verbatim from
/// `catboost-master/catboost/python-package/catboost/core.py:5333`). The registry
/// tags each Implemented (iff it appears in [`IMPLEMENTED`]) else KnownNotYet.
///
/// Note: the sklearn-alias names (`max_depth`, `n_estimators`, `num_boost_round`,
/// `num_trees`, `colsample_bylevel`, `random_state`, `reg_lambda`, `objective`,
/// `eta`, `max_bin`) ARE in this list (they are real upstream kwargs); they are
/// additionally resolved through [`ALIASES`] before tag lookup.
const VOCABULARY: &[&str] = &[
    "iterations",
    "learning_rate",
    "depth",
    "l2_leaf_reg",
    "model_size_reg",
    "rsm",
    "loss_function",
    "border_count",
    "feature_border_type",
    "per_float_feature_quantization",
    "input_borders",
    "output_borders",
    "fold_permutation_block",
    "od_pval",
    "od_wait",
    "od_type",
    "nan_mode",
    "counter_calc_method",
    "leaf_estimation_iterations",
    "leaf_estimation_method",
    "thread_count",
    "random_seed",
    "use_best_model",
    "best_model_min_trees",
    "verbose",
    "silent",
    "logging_level",
    "metric_period",
    "ctr_leaf_count_limit",
    "store_all_simple_ctr",
    "max_ctr_complexity",
    "has_time",
    "allow_const_label",
    "target_border",
    "classes_count",
    "class_weights",
    "auto_class_weights",
    "class_names",
    "one_hot_max_size",
    "random_strength",
    "random_score_type",
    "name",
    "ignored_features",
    "train_dir",
    "custom_loss",
    "custom_metric",
    "eval_metric",
    "bagging_temperature",
    "save_snapshot",
    "snapshot_file",
    "snapshot_interval",
    "fold_len_multiplier",
    "used_ram_limit",
    "gpu_ram_part",
    "pinned_memory_size",
    "allow_writing_files",
    "final_ctr_computation_mode",
    "approx_on_full_history",
    "boosting_type",
    "simple_ctr",
    "combinations_ctr",
    "per_feature_ctr",
    "ctr_description",
    "ctr_target_border_count",
    "task_type",
    "device_config",
    "devices",
    "bootstrap_type",
    "subsample",
    "mvs_reg",
    "sampling_unit",
    "sampling_frequency",
    "dev_score_calc_obj_block_size",
    "dev_efb_max_buckets",
    "sparse_features_conflict_fraction",
    "max_depth",
    "n_estimators",
    "num_boost_round",
    "num_trees",
    "colsample_bylevel",
    "random_state",
    "reg_lambda",
    "objective",
    "eta",
    "max_bin",
    "scale_pos_weight",
    "gpu_cat_features_storage",
    "data_partition",
    "metadata",
    "early_stopping_rounds",
    "cat_features",
    "grow_policy",
    "min_data_in_leaf",
    "min_child_samples",
    "max_leaves",
    "num_leaves",
    "score_function",
    "leaf_estimation_backtracking",
    "ctr_history_unit",
    "monotone_constraints",
    "feature_weights",
    "penalties_coefficient",
    "first_feature_use_penalties",
    "per_object_feature_penalties",
    "model_shrink_rate",
    "model_shrink_mode",
    "langevin",
    "diffusion_temperature",
    "posterior_sampling",
    "boost_from_average",
    "text_features",
    "tokenizers",
    "dictionaries",
    "feature_calcers",
    "text_processing",
    "embedding_features",
    "callback",
    "eval_fraction",
    "fixed_binary_splits",
];

/// sklearn / xgboost-style aliases -> the canonical upstream name. Resolving an
/// alias lets a migrating user keep their kwargs (`n_estimators`, `max_depth`,
/// ...). The canonical target then goes through the SAME Implemented/KnownNotYet
/// tag lookup (so `colsample_bylevel` -> `rsm` is correctly a parity gap).
const ALIASES: &[(&str, &str)] = &[
    ("max_depth", "depth"),
    ("n_estimators", "iterations"),
    ("num_trees", "iterations"),
    ("num_boost_round", "iterations"),
    ("random_state", "random_seed"),
    ("reg_lambda", "l2_leaf_reg"),
    ("objective", "loss_function"),
    ("eta", "learning_rate"),
    ("max_bin", "border_count"),
    ("colsample_bylevel", "rsm"),
];

/// Resolve an alias to its canonical name (identity if not an alias).
fn resolve_alias(name: &str) -> &str {
    ALIASES
        .iter()
        .find(|(alias, _)| *alias == name)
        .map_or(name, |(_, canonical)| *canonical)
}

/// The registry status of a CANONICAL (post-alias) param name, or `None` if it is
/// not in the upstream vocabulary at all.
pub(crate) fn status_of(canonical: &str) -> Option<ParamStatus> {
    if !VOCABULARY.contains(&canonical) {
        return None;
    }
    if IMPLEMENTED.contains(&canonical) {
        Some(ParamStatus::Implemented)
    } else {
        Some(ParamStatus::KnownNotYet)
    }
}

/// The registry status of any user-supplied name (alias-resolved first). Returns
/// `None` for an unknown (out-of-vocabulary) name.
pub(crate) fn status_of_user(name: &str) -> Option<ParamStatus> {
    status_of(resolve_alias(name))
}

/// Introspection helper (registered as `catboost_rs._param_status`): the registry
/// status of a user-supplied param name (alias-resolved) as a string
/// (`"IMPLEMENTED"` / `"KNOWN_NOT_YET"`), or `None` if it is not in the upstream
/// vocabulary. Lets the param-coverage test assert every upstream kwarg is known.
#[pyfunction]
pub(crate) fn _param_status(name: &str) -> Option<&'static str> {
    status_of_user(name).map(|s| match s {
        ParamStatus::Implemented => "IMPLEMENTED",
        ParamStatus::KnownNotYet => "KNOWN_NOT_YET",
    })
}

/// Classic Levenshtein edit distance (no panics; no indexing — checked
/// iteration), used to suggest the closest vocabulary entry for a typo (threat
/// T-08-07).
fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr: Vec<usize> = vec![0; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        curr[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let cost = usize::from(ca != cb);
            let del = prev[j + 1] + 1;
            let ins = curr[j] + 1;
            let sub = prev[j] + cost;
            curr[j + 1] = del.min(ins).min(sub);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()]
}

/// The vocabulary (incl. aliases) entry closest to `name` by edit distance, used
/// only for the "did you mean ...?" hint.
fn closest_match(name: &str) -> &'static str {
    let alias_names = ALIASES.iter().map(|(a, _)| *a);
    VOCABULARY
        .iter()
        .copied()
        .chain(alias_names)
        .min_by_key(|cand| levenshtein(name, cand))
        .unwrap_or("iterations")
}

/// Validate every user kwarg against the registry BEFORE ingest (D-06). Resolves
/// aliases, then rejects KnownNotYet (parity gap) and UNKNOWN (typo, with a
/// closest-match suggestion) params with a `CatBoostParameterError`.
///
/// # Errors
/// `CatBoostParameterError` on the first KnownNotYet or unknown kwarg.
pub(crate) fn validate_params(params: &BTreeMap<String, Py<PyAny>>) -> PyResult<()> {
    for name in params.keys() {
        let canonical = resolve_alias(name);
        match status_of(canonical) {
            Some(ParamStatus::Implemented) => {}
            Some(ParamStatus::KnownNotYet) => {
                let detail = if canonical == name.as_str() {
                    format!(
                        "parameter `{name}` is a known CatBoost parameter not yet implemented in \
                         catboost-rs (parity gap)"
                    )
                } else {
                    format!(
                        "parameter `{name}` (alias of `{canonical}`) is a known CatBoost parameter \
                         not yet implemented in catboost-rs (parity gap)"
                    )
                };
                return Err(CatBoostParameterError::new_err(detail));
            }
            None => {
                let suggestion = closest_match(name);
                return Err(CatBoostParameterError::new_err(format!(
                    "unknown parameter `{name}`; did you mean `{suggestion}`?"
                )));
            }
        }
    }
    Ok(())
}

/// Read a stored param (alias-resolved at the call site) as `T` via `extract`.
fn get<'py, T>(params: &'py BTreeMap<String, Py<PyAny>>, py: Python<'py>, name: &str) -> PyResult<Option<T>>
where
    T: FromPyObject<'py, 'py>,
    PyErr: From<<T as FromPyObject<'py, 'py>>::Error>,
{
    match params.get(name) {
        Some(v) => Ok(Some(v.bind(py).extract::<T>()?)),
        None => Ok(None),
    }
}

/// Look up a param under its canonical name OR any of its aliases (first hit
/// wins), extracting as `T`.
fn get_with_aliases<'py, T>(
    params: &'py BTreeMap<String, Py<PyAny>>,
    py: Python<'py>,
    canonical: &str,
) -> PyResult<Option<T>>
where
    T: FromPyObject<'py, 'py>,
    PyErr: From<<T as FromPyObject<'py, 'py>>::Error>,
{
    if let Some(v) = get::<T>(params, py, canonical)? {
        return Ok(Some(v));
    }
    for (alias, target) in ALIASES {
        if *target == canonical {
            if let Some(v) = get::<T>(params, py, alias)? {
                return Ok(Some(v));
            }
        }
    }
    Ok(None)
}

/// Map a `loss_function` string onto a [`Loss`]. Only the numeric-regression
/// losses with a built-in default-parameter form are accepted here; anything else
/// is rejected as a parity gap (the parametric losses need their args, 08-04+).
fn parse_loss(name: &str) -> PyResult<Loss> {
    match name {
        "RMSE" => Ok(Loss::Rmse),
        "Logloss" => Ok(Loss::Logloss),
        "CrossEntropy" => Ok(Loss::CrossEntropy),
        "MAE" => Ok(Loss::Mae),
        "LogCosh" => Ok(Loss::LogCosh),
        other => Err(CatBoostParameterError::new_err(format!(
            "loss_function `{other}` is not yet supported through the catboost-rs binding \
             (supported: RMSE, Logloss, CrossEntropy, MAE, LogCosh)"
        ))),
    }
}

/// Map a `score_function` string onto an [`EScoreFunction`].
fn parse_score_function(name: &str) -> PyResult<EScoreFunction> {
    match name {
        "Cosine" => Ok(EScoreFunction::Cosine),
        "L2" => Ok(EScoreFunction::L2),
        "SolarL2" => Ok(EScoreFunction::SolarL2),
        "NewtonL2" => Ok(EScoreFunction::NewtonL2),
        "NewtonCosine" => Ok(EScoreFunction::NewtonCosine),
        "LOOL2" => Ok(EScoreFunction::LOOL2),
        "SatL2" => Ok(EScoreFunction::SatL2),
        other => Err(CatBoostParameterError::new_err(format!(
            "unknown score_function `{other}`"
        ))),
    }
}

/// Map a `bootstrap_type` string onto an [`EBootstrapType`].
fn parse_bootstrap_type(name: &str) -> PyResult<EBootstrapType> {
    match name {
        "No" => Ok(EBootstrapType::No),
        "Bayesian" => Ok(EBootstrapType::Bayesian),
        "Bernoulli" => Ok(EBootstrapType::Bernoulli),
        "MVS" => Ok(EBootstrapType::Mvs),
        "Poisson" => Ok(EBootstrapType::Poisson),
        other => Err(CatBoostParameterError::new_err(format!(
            "unknown bootstrap_type `{other}`"
        ))),
    }
}

/// Map a `leaf_estimation_method` string onto a [`LeafMethod`].
fn parse_leaf_method(name: &str) -> PyResult<LeafMethod> {
    match name {
        "Gradient" => Ok(LeafMethod::Gradient),
        "Newton" => Ok(LeafMethod::Newton),
        "Simple" => Ok(LeafMethod::Simple),
        "Exact" => Ok(LeafMethod::Exact),
        other => Err(CatBoostParameterError::new_err(format!(
            "unknown leaf_estimation_method `{other}`"
        ))),
    }
}

/// Build a [`CatBoostBuilder`] from the validated params, applying every
/// IMPLEMENTED param (alias-resolved) with the correct type extraction. The
/// caller MUST have run [`validate_params`] first (so only IMPLEMENTED params are
/// present among the recognized set; KnownNotYet/UNKNOWN already rejected).
///
/// # Errors
/// A `PyTypeError`/`PyValueError` (via `extract`) if a param value has the wrong
/// type, or a `CatBoostParameterError` for an unsupported enum string.
pub(crate) fn make_builder(
    params: &BTreeMap<String, Py<PyAny>>,
    py: Python<'_>,
) -> PyResult<CatBoostBuilder> {
    let mut builder = CatBoostBuilder::new();
    if let Some(v) = get_with_aliases::<usize>(params, py, "iterations")? {
        builder = builder.iterations(v);
    }
    if let Some(v) = get_with_aliases::<usize>(params, py, "depth")? {
        builder = builder.depth(v);
    }
    if let Some(v) = get_with_aliases::<f64>(params, py, "learning_rate")? {
        builder = builder.learning_rate(v);
    }
    if let Some(v) = get_with_aliases::<f64>(params, py, "l2_leaf_reg")? {
        builder = builder.l2_leaf_reg(v);
    }
    if let Some(v) = get_with_aliases::<f64>(params, py, "random_strength")? {
        builder = builder.random_strength(v);
    }
    if let Some(v) = get_with_aliases::<u64>(params, py, "random_seed")? {
        builder = builder.random_seed(v);
    }
    if let Some(v) = get_with_aliases::<usize>(params, py, "border_count")? {
        builder = builder.border_count(v);
    }
    if let Some(v) = get_with_aliases::<f64>(params, py, "subsample")? {
        builder = builder.subsample(v);
    }
    if let Some(v) = get_with_aliases::<f32>(params, py, "bagging_temperature")? {
        builder = builder.bagging_temperature(v);
    }
    if let Some(v) = get_with_aliases::<bool>(params, py, "boost_from_average")? {
        builder = builder.boost_from_average(v);
    }
    if let Some(v) = get_with_aliases::<String>(params, py, "loss_function")? {
        builder = builder.loss(parse_loss(&v)?);
    }
    if let Some(v) = get_with_aliases::<String>(params, py, "score_function")? {
        builder = builder.score_function(parse_score_function(&v)?);
    }
    if let Some(v) = get_with_aliases::<String>(params, py, "bootstrap_type")? {
        builder = builder.bootstrap_type(parse_bootstrap_type(&v)?);
    }
    if let Some(v) = get_with_aliases::<String>(params, py, "leaf_estimation_method")? {
        builder = builder.leaf_method(parse_leaf_method(&v)?);
    }
    Ok(builder)
}
