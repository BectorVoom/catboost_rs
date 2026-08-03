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
//!
//! # KNOWN PARITY GAP — the CTR default is a single description, not a list
//!
//! Upstream's CPU default is a LIST of two CTR descriptions
//! (`[Borders(priors 0/1, 0.5/1, 1/1), Counter(prior 0/1)]`, set by
//! `SetCtrDefaults` at `catboost_options.cpp:439-453`; measured through
//! `get_all_params()` as
//! `['Borders:…:Prior=0/1:Prior=0.5/1:Prior=1/1', 'Counter:…:Prior=0/1']`).
//!
//! This crate models ONE CTR description with a prior LIST: the engine field is
//! a scalar `ECtrType` plus a `Vec<f64>` of priors. The CTR **type** and the
//! **full prior list** ARE honored (SPEC-CTRT-09/10/11) — but a simultaneous
//! `[Borders, Counter]` configuration is NOT representable, so `simple_ctr` /
//! `combinations_ctr` accept exactly one CTR description each (SPEC-CTRT-19).
//!
//! This is deliberate, not an oversight: `simple_ctr: ECtrType` is pinned at 62
//! construction sites across the workspace and retyping it to a list has zero
//! behavioral benefit to any of them. A user who passes more than one
//! description is REJECTED with a `CatBoostParameterError` naming this gap —
//! never silently narrowed to the first entry.

use std::collections::BTreeMap;

use catboost_rs::{
    CatBoostBuilder, CounterCalcMethod, EBootstrapType, ECtrType, EScoreFunction, LeafMethod, Loss,
};
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
    // The six categorical / CTR params promoted by F15-F17. Each is now
    // GENUINELY consumed: F01-F05 gave the builder its setter, F09 routes a
    // categorical pool through `train_cat`, and E10/E11/E22 made the engine read
    // the type, the full prior list and `counter_calc_method`. `cat_features` is
    // a `fit()` kwarg rather than a builder setter (F17) but is IMPLEMENTED all
    // the same, so it must not be rejected as a parity gap.
    "cat_features",
    "one_hot_max_size",
    "max_ctr_complexity",
    "simple_ctr",
    "combinations_ctr",
    "counter_calc_method",
];

/// The four CTR types CatBoost implements ON CPU. `FloatTargetMeanValue` and
/// `FeatureFreq` are GPU-only (`catboost_options.cpp:504-509`); upstream rejects
/// them on CPU and so does this crate's engine (the E02 guard), so accepting
/// them here would manufacture a kwarg that fails deeper with a worse message.
const CPU_LEGAL_CTR_TYPES: &[&str] = &[
    "Borders",
    "Buckets",
    "BinarizedTargetMeanValue",
    "Counter",
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
        // VOCABULARY is a non-empty `const`, so `min_by_key` always yields `Some`.
        // `expect` documents that invariant honestly (IN-02) rather than the
        // previous `.unwrap_or("iterations")`, whose dead "iterations" default would
        // have been a misleading typo suggestion for an unrelated name if it fired.
        .min_by_key(|cand| levenshtein(name, cand))
        .expect("VOCABULARY is a non-empty const, so the closest match always exists")
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

/// Map a `counter_calc_method` string onto a [`CounterCalcMethod`] (F15).
///
/// **Observable only with an eval set.** Training on a learn set alone produces
/// bit-identical models under either setting (measured max|diff| `0.000e+00`
/// learn-only versus `4.010e-01` with an eval set — the E23 gate), so this is
/// NOT a blanket parity control: it bites only through the eval-set training
/// entrypoint.
fn parse_counter_calc_method(name: &str) -> PyResult<CounterCalcMethod> {
    match name {
        "SkipTest" => Ok(CounterCalcMethod::SkipTest),
        "Full" => Ok(CounterCalcMethod::Full),
        other => Err(CatBoostParameterError::new_err(format!(
            "unknown counter_calc_method `{other}`; expected SkipTest or Full"
        ))),
    }
}

/// Map a CTR-type name onto an [`ECtrType`], rejecting the two GPU-only types
/// by name with upstream's own reasoning (`catboost_options.cpp:504-509`).
fn parse_ctr_type(name: &str, param: &str) -> PyResult<ECtrType> {
    match name {
        "Borders" => Ok(ECtrType::Borders),
        "Buckets" => Ok(ECtrType::Buckets),
        "BinarizedTargetMeanValue" => Ok(ECtrType::BinarizedTargetMeanValue),
        "Counter" => Ok(ECtrType::Counter),
        "FloatTargetMeanValue" | "FeatureFreq" => Err(CatBoostParameterError::new_err(format!(
            "`{param}` CTR type `{name}` is not implemented on CPU (it is GPU-only, \
             catboost_options.cpp:504-509); supported CPU types are {}",
            CPU_LEGAL_CTR_TYPES.join(", ")
        ))),
        other => Err(CatBoostParameterError::new_err(format!(
            "unknown `{param}` CTR type `{other}`; supported CPU types are {}",
            CPU_LEGAL_CTR_TYPES.join(", ")
        ))),
    }
}

/// Parse ONE CatBoost CTR description — `"<Type>[:Key=Value]..."` — into its
/// `(type, prior numerators)` pair (F16 / SPEC-CATF-13).
///
/// Recognised keys: `Prior` (repeatable — one CTR column per prior). A prior is
/// `<num>` or `<num>/<denom>`; upstream rejects `denom != 1` on CPU
/// (`ctr_helper.cpp:50`), which is exactly what vindicates the engine's
/// `prior_denom: 1.0` pin, so a non-unit denominator is rejected here rather
/// than silently renormalized.
///
/// Any other key is REJECTED, never ignored: silently dropping a
/// `CtrBorderCount=...` the engine does not read would train a
/// differently-parameterized model than the user asked for.
fn parse_ctr_description(desc: &str, param: &str) -> PyResult<(ECtrType, Vec<f64>)> {
    let mut tokens = desc.split(':');
    let type_name = tokens.next().unwrap_or("").trim();
    let ctr_type = parse_ctr_type(type_name, param)?;

    let mut priors: Vec<f64> = Vec::new();
    for token in tokens {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        let (key, value) = token.split_once('=').ok_or_else(|| {
            CatBoostParameterError::new_err(format!(
                "`{param}` description `{desc}`: malformed token `{token}` (expected Key=Value)"
            ))
        })?;
        if !key.trim().eq_ignore_ascii_case("Prior") {
            return Err(CatBoostParameterError::new_err(format!(
                "`{param}` description `{desc}`: key `{}` is not supported; this crate \
                 reads only the CTR type and its Prior list",
                key.trim()
            )));
        }
        let value = value.trim();
        let (num_str, denom) = match value.split_once('/') {
            Some((n, d)) => {
                let denom: f64 = d.trim().parse().map_err(|_| {
                    CatBoostParameterError::new_err(format!(
                        "`{param}` description `{desc}`: prior denominator `{d}` is not a number"
                    ))
                })?;
                (n.trim(), denom)
            }
            None => (value, 1.0),
        };
        if (denom - 1.0).abs() > f64::EPSILON {
            return Err(CatBoostParameterError::new_err(format!(
                "`{param}` description `{desc}`: a prior denominator other than 1 is \
                 illegal on CPU (ctr_helper.cpp:50), got `{denom}`; write the prior as \
                 `Prior=<numerator>` or `Prior=<numerator>/1`"
            )));
        }
        let num: f64 = num_str.trim().parse().map_err(|_| {
            CatBoostParameterError::new_err(format!(
                "`{param}` description `{desc}`: prior numerator `{num_str}` is not a number"
            ))
        })?;
        priors.push(num);
    }
    Ok((ctr_type, priors))
}

/// Parse a `simple_ctr` / `combinations_ctr` kwarg — a LIST of CTR descriptions
/// (upstream's shape) or a single bare string — into one `(type, priors)` pair.
///
/// `None` means "the list was empty", which only `combinations_ctr` gives a
/// meaning to (see [`make_builder`]).
///
/// KNOWN PARITY GAP (SPEC-CTRT-19): upstream's CPU default is a LIST of two
/// descriptions (`catboost_options.cpp:439-453`); this crate models ONE with a
/// prior LIST. A multi-description list is therefore REJECTED with that gap
/// named — never silently narrowed to its first entry, which would train a
/// model the user did not ask for.
fn parse_ctr_param(descs: &[String], param: &str) -> PyResult<Option<(ECtrType, Vec<f64>)>> {
    match descs {
        [] => Ok(None),
        [one] => parse_ctr_description(one, param).map(Some),
        many => Err(CatBoostParameterError::new_err(format!(
            "`{param}` accepts exactly one CTR description, got {} ({many:?}). KNOWN \
             PARITY GAP: upstream's CPU default is a LIST of two descriptions \
             (`[Borders(priors 0/1, 0.5/1, 1/1), Counter(prior 0/1)]`, \
             catboost_options.cpp:439-453), but this crate models ONE description \
             with a prior LIST — the type and the full prior list ARE honored, a \
             simultaneous [Borders, Counter] configuration is not representable. \
             Passing several descriptions is rejected rather than narrowed to the \
             first, which would silently train a different model.",
            many.len()
        ))),
    }
}

/// Extract a `simple_ctr` / `combinations_ctr` kwarg as a list of descriptions,
/// accepting either upstream's `list[str]` or a convenience bare `str`.
fn extract_ctr_descriptions(
    params: &BTreeMap<String, Py<PyAny>>,
    py: Python<'_>,
    name: &str,
) -> PyResult<Option<Vec<String>>> {
    let Some(obj) = params.get(name) else {
        return Ok(None);
    };
    let bound = obj.bind(py);
    if let Ok(one) = bound.extract::<String>() {
        return Ok(Some(vec![one]));
    }
    bound.extract::<Vec<String>>().map(Some).map_err(|_| {
        CatBoostParameterError::new_err(format!(
            "`{name}` must be a list of CTR description strings (or a single string), \
             e.g. [\"Borders:Prior=0:Prior=0.5\"]"
        ))
    })
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

/// Reject a numeric param whose VALUE is outside its valid domain (WR-05) with a
/// `CatBoostParameterError` naming the param and value, BEFORE it reaches the
/// builder/train loop. Param-name validation ([`validate_params`]) only checks the
/// kwarg is in the vocabulary; this guards the VALUE so nonsensical numerics
/// (`iterations=0`, `depth=0`, `learning_rate=-1.0`, `subsample=5.0`,
/// `l2_leaf_reg=-3.0`) surface a clear error at the offending param rather than a
/// degenerate model or a low-level error far downstream (mirrors upstream
/// CatBoost's at-construction range checks). A non-finite value (NaN/inf) is also
/// rejected.
fn check_range(name: &str, value: f64, lo: f64, hi: f64, lo_inclusive: bool) -> PyResult<()> {
    let lo_ok = if lo_inclusive { value >= lo } else { value > lo };
    if value.is_finite() && lo_ok && value <= hi {
        Ok(())
    } else {
        let lo_bracket = if lo_inclusive { "[" } else { "(" };
        Err(CatBoostParameterError::new_err(format!(
            "parameter `{name}` = {value} is out of range; expected {lo_bracket}{lo}, {hi}]"
        )))
    }
}

/// Build a [`CatBoostBuilder`] from the validated params, applying every
/// IMPLEMENTED param (alias-resolved) with the correct type extraction. The
/// caller MUST have run [`validate_params`] first (so only IMPLEMENTED params are
/// present among the recognized set; KnownNotYet/UNKNOWN already rejected).
///
/// Bounded numeric params are additionally RANGE-validated here (WR-05) so an
/// out-of-domain value (e.g. `iterations=0`, `learning_rate=-1.0`,
/// `subsample=5.0`) is rejected with a clear `CatBoostParameterError` before it
/// reaches the train loop.
///
/// # Errors
/// A `PyTypeError`/`PyValueError` (via `extract`) if a param value has the wrong
/// type, a `CatBoostParameterError` for an unsupported enum string, or a
/// `CatBoostParameterError` for a numeric value outside its valid domain.
pub(crate) fn make_builder(
    params: &BTreeMap<String, Py<PyAny>>,
    py: Python<'_>,
) -> PyResult<CatBoostBuilder> {
    let mut builder = CatBoostBuilder::new();
    if let Some(v) = get_with_aliases::<usize>(params, py, "iterations")? {
        // iterations >= 1 (an empty ensemble trains nothing).
        check_range("iterations", v as f64, 1.0, f64::INFINITY, true)?;
        builder = builder.iterations(v);
    }
    if let Some(v) = get_with_aliases::<usize>(params, py, "depth")? {
        // depth in [1, 16] (upstream caps oblivious-tree depth at 16).
        check_range("depth", v as f64, 1.0, 16.0, true)?;
        builder = builder.depth(v);
    }
    if let Some(v) = get_with_aliases::<f64>(params, py, "learning_rate")? {
        // 0 < learning_rate <= 1.
        check_range("learning_rate", v, 0.0, 1.0, false)?;
        builder = builder.learning_rate(v);
    }
    if let Some(v) = get_with_aliases::<f64>(params, py, "l2_leaf_reg")? {
        // l2_leaf_reg >= 0.
        check_range("l2_leaf_reg", v, 0.0, f64::INFINITY, true)?;
        builder = builder.l2_leaf_reg(v);
    }
    if let Some(v) = get_with_aliases::<f64>(params, py, "random_strength")? {
        // random_strength >= 0.
        check_range("random_strength", v, 0.0, f64::INFINITY, true)?;
        builder = builder.random_strength(v);
    }
    if let Some(v) = get_with_aliases::<u64>(params, py, "random_seed")? {
        builder = builder.random_seed(v);
    }
    if let Some(v) = get_with_aliases::<usize>(params, py, "border_count")? {
        // border_count in [1, 65535] (upstream limit).
        check_range("border_count", v as f64, 1.0, 65535.0, true)?;
        builder = builder.border_count(v);
    }
    if let Some(v) = get_with_aliases::<f64>(params, py, "subsample")? {
        // 0 < subsample <= 1.
        check_range("subsample", v, 0.0, 1.0, false)?;
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

    // ---- F15/F16: the categorical / CTR surface -----------------------------
    if let Some(v) = get_with_aliases::<u32>(params, py, "one_hot_max_size")? {
        // Upstream default 2; the CPU upper bound is `OneHotMaxSizeLimit =
        // GetMaxBinCount()` (cat_feature_options.cpp:232-233, 267-268).
        check_range("one_hot_max_size", f64::from(v), 0.0, 65535.0, true)?;
        builder = builder.one_hot_max_size(v);
    }
    if let Some(v) = get_with_aliases::<usize>(params, py, "max_ctr_complexity")? {
        // Upstream default 4; `CB_ENSURE(MaxTensorComplexity < GetMaxTreeDepth())`
        // (cat_feature_options.cpp:231, 269-271) with GetMaxTreeDepth() == 16.
        check_range("max_ctr_complexity", v as f64, 0.0, 15.0, true)?;
        builder = builder.max_ctr_complexity(v);
    }
    if let Some(v) = get_with_aliases::<String>(params, py, "counter_calc_method")? {
        builder = builder.counter_calc_method(parse_counter_calc_method(&v)?);
    }
    if let Some(descs) = extract_ctr_descriptions(params, py, "simple_ctr")? {
        match parse_ctr_param(&descs, "simple_ctr")? {
            Some((ctr_type, priors)) => {
                builder = builder.simple_ctr(ctr_type);
                if !priors.is_empty() {
                    builder = builder.simple_ctr_priors(priors);
                }
            }
            // `simple_ctr=[]` has NO in-engine meaning: simple CTRs cannot be
            // switched off (unlike combination CTRs, which `max_ctr_complexity=1`
            // suppresses). Reject rather than silently ignore the kwarg.
            None => {
                return Err(CatBoostParameterError::new_err(
                    "`simple_ctr=[]` has no meaning: simple CTRs cannot be disabled. \
                     Every high-cardinality categorical column is CTR-encoded; pass a \
                     single description such as [\"Borders:Prior=0.5\"] instead.",
                ))
            }
        }
    }
    if let Some(descs) = extract_ctr_descriptions(params, py, "combinations_ctr")? {
        match parse_ctr_param(&descs, "combinations_ctr")? {
            Some((ctr_type, priors)) => {
                builder = builder.combinations_ctr(ctr_type);
                if !priors.is_empty() {
                    builder = builder.combinations_ctr_priors(priors);
                }
            }
            // RECORDED MAPPING (F16, option (a)): the engine field is a scalar
            // `ECtrType` with no "disabled" representation, and
            // `max_ctr_complexity = 1` is the ONLY in-engine way to suppress
            // combination CTRs. `combinations_ctr=[]` therefore maps to
            // `max_ctr_complexity = 1`. Chosen over rejecting `[]` because the
            // committed crates/cb-oracle/fixtures/plain_ctr/config.json uses
            // exactly that form, so F19 requires it be expressible here.
            // Asserted by `empty_combinations_ctr_maps_to_max_ctr_complexity_one`,
            // which checks the value ACTUALLY moves rather than that fit() is quiet.
            None => builder = builder.max_ctr_complexity(1),
        }
    }
    Ok(builder)
}
