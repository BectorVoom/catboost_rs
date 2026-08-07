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
//! 4. **VALIDATED-INFORMATIONAL** (FPP-16) — a real upstream parameter this project
//!    honours by VALIDATING CONSISTENCY rather than by changing behaviour. It is tagged
//!    IMPLEMENTED (so it is accepted, not reported as a parity gap) but has no
//!    [`CatBoostBuilder`] setter, because there is nothing for it to set.
//!
//!    `task_type` is the only member today. Backend selection in catboost-rs is a
//!    COMPILE-TIME Cargo feature (`cpu` / `cuda` / `rocm` / `wgpu`) — see CLAUDE.md's
//!    Constraints — so no runtime value could switch it. Accepting the parameter silently
//!    would be dishonest in the same way category 2 warns about: a caller who wrote
//!    `task_type="GPU"` and got a CPU model would have a silently-wrong result. So
//!    `"CPU"` is always accepted and inert, `"GPU"` is accepted only when the wheel was
//!    actually built with a device backend, and anything else is an error listing the
//!    legal values.
//!
//!    This IS observable behaviour, which is what makes it honest rather than a no-op: an
//!    inconsistent request fails loudly. What it deliberately does NOT do is change the
//!    trained model — pinned by a pytest asserting bit-identical predictions with and
//!    without `task_type="CPU"`.
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
    parse_metric, AutoClassWeights, CatBoostBuilder, CounterCalcMethod, EBoostingType,
    EBootstrapType, ECtrType, EGrowPolicy, EOverfittingDetectorType, EScoreFunction, EvalMetric,
    LeafMethod, Loss,
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
    // FPP-16: VALIDATED-INFORMATIONAL (see the module doc's category 4). `task_type` has
    // no `CatBoostBuilder` setter and never will — backend selection here is a Cargo
    // feature, decided at compile time — so it is "implemented" in the sense that we
    // honour it by VALIDATING it against the wheel, not by acting on it.
    "task_type",
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
    // PARAM-02: promoted once `CatBoostBuilder` grew a setter for each (PARAM-01).
    // Before that the engine implemented every one of these but `boost_params()`
    // pinned it to a literal, so the honest status was KnownNotYet.
    //
    // The four EVAL-SET-ONLY params (`od_*`, `use_best_model`, `eval_metric`,
    // plus the already-implemented `counter_calc_method`) are inert unless
    // `fit(..., eval_set=...)` supplies a validation set. That is NOT a reason to
    // reject them — upstream accepts them the same way — but it IS why
    // `use_best_model` / `early_stopping_rounds` without an `eval_set` raise
    // rather than silently doing nothing (see `validate_eval_set_only_params`).
    "od_type",
    "od_pval",
    "od_wait",
    "early_stopping_rounds",
    "use_best_model",
    "eval_metric",
    "boosting_type",
    "has_time",
    "fold_len_multiplier",
    "monotone_constraints",
    "feature_weights",
    "penalties_coefficient",
    "first_feature_use_penalties",
    "per_object_feature_penalties",
    "grow_policy",
    "max_leaves",
    "min_data_in_leaf",
    // PARAM-03: the class-weight family + ignored_features. The class-weight
    // COMPUTATION already existed in `cb_data::weights` (oracle-gated against the
    // frozen upstream `class_weights/` fixture); what PARAM-03 added is the path
    // that APPLIES the resolved per-object weights during a fit.
    "class_weights",
    "auto_class_weights",
    "scale_pos_weight",
    "ignored_features",
];

/// The params that only DO something when `fit` is given an `eval_set`. Passing
/// one without a validation set is REJECTED rather than silently ignored (threat
/// T-08-05): a user who writes `use_best_model=True` and gets an untruncated
/// model back has no way to notice the parameter did nothing.
///
/// `eval_metric` and `counter_calc_method` are deliberately NOT in this list.
/// Both are also read on the learn-only path in upstream workflows (an
/// `eval_metric` is meaningful metadata on the model, and `counter_calc_method`
/// is a CTR-tally setting whose eval-set-only OBSERVABILITY is a property of the
/// data, not a misuse), so rejecting them would break valid calls.
const EVAL_SET_ONLY: &[&str] = &[
    "od_type",
    "od_pval",
    "od_wait",
    "early_stopping_rounds",
    "use_best_model",
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
    // PARAM-02: the two LightGBM-style leaf aliases. Both are real upstream
    // kwargs (they are in VOCABULARY) and both now resolve onto an IMPLEMENTED
    // canonical name.
    ("num_leaves", "max_leaves"),
    ("min_child_samples", "min_data_in_leaf"),
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
/// Whether this wheel was built with a device backend feature. Backend selection in
/// catboost-rs is COMPILE-TIME only (a Cargo feature), which is why `task_type` can be
/// validated but never acted on.
const DEVICE_FEATURE_COMPILED: bool =
    cfg!(any(feature = "wgpu", feature = "cuda", feature = "rocm"));

/// The legal `task_type` values, in the order the error message lists them.
const TASK_TYPE_VALUES: &[&str] = &["CPU", "GPU"];

/// The exact user-facing message for `task_type="GPU"` on a CPU-only wheel. Pinned as a
/// constant because it IS the contract — the tests assert it verbatim.
pub(crate) const TASK_TYPE_GPU_ON_CPU_WHEEL: &str =
    "task_type=\"GPU\" requires a wheel built with a device backend feature (--features \
     cuda, rocm, or wgpu); this wheel was built with --features cpu. Backend selection in \
     catboost-rs is compile-time only (see CLAUDE.md), so task_type cannot switch backends \
     at runtime.";

/// FPP-16: validate `task_type` against the wheel this code was compiled into.
///
/// `None` is inert — upstream's universal "not set" (the [`get`] convention). `"CPU"` is
/// always accepted and changes nothing. `"GPU"` is accepted only on a wheel built with a
/// device backend feature; on a `cpu`-only wheel it is an ERROR, because silently training
/// on the CPU after an explicit `task_type="GPU"` is exactly the silently-wrong-model
/// failure this module's honesty policy exists to prevent.
///
/// A wrong VALUE is not a wrong NAME: the error lists the legal values rather than
/// offering a Levenshtein suggestion, which would be nonsense here.
fn validate_task_type(py: Python<'_>, value: &Py<PyAny>) -> PyResult<()> {
    let bound = value.bind(py);
    if bound.is_none() {
        return Ok(());
    }
    let Ok(requested) = bound.extract::<String>() else {
        return Err(CatBoostParameterError::new_err(format!(
            "parameter `task_type` must be a string, one of {TASK_TYPE_VALUES:?}"
        )));
    };
    if requested.eq_ignore_ascii_case("CPU") {
        return Ok(());
    }
    if requested.eq_ignore_ascii_case("GPU") {
        if DEVICE_FEATURE_COMPILED {
            return Ok(());
        }
        return Err(CatBoostParameterError::new_err(TASK_TYPE_GPU_ON_CPU_WHEEL));
    }
    Err(CatBoostParameterError::new_err(format!(
        "parameter `task_type` has an unknown value `{requested}`; the legal values are \
         {TASK_TYPE_VALUES:?}"
    )))
}

pub(crate) fn validate_params(params: &BTreeMap<String, Py<PyAny>>) -> PyResult<()> {
    for (name, value) in params {
        let canonical = resolve_alias(name);
        match status_of(canonical) {
            Some(ParamStatus::Implemented) => {
                // FPP-16: the one VALIDATED-INFORMATIONAL param needs its VALUE checked
                // here; every other Implemented param is checked by its builder setter.
                if canonical == "task_type" {
                    Python::attach(|py| validate_task_type(py, value))?;
                }
            }
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

/// Whether an eval-set-only param is set to a value that would actually DO
/// something — i.e. whether ignoring it silently would lose anything.
///
/// The guard below keys on this, not on the name alone. Every one of these
/// params has a spelling that means "off", and passing the off-value is not a
/// request for behaviour the learn-only path cannot deliver — it is a statement
/// that the behaviour is not wanted. Rejecting `od_type="None"` or
/// `use_best_model=False` told the caller their explicit *disabling* of early
/// stopping was incompatible with not having an eval set, which reads as a bug
/// and blocks any wrapper or config layer that materializes a full default
/// parameter dict (a common sklearn / MLflow pattern).
///
/// The Python literal `None` is inert for ALL of them — it is upstream's
/// universal "not set" (see [`get`]) — and is handled once, up front.
fn eval_set_only_param_is_active(py: Python<'_>, canonical: &str, value: &Py<PyAny>) -> bool {
    let bound = value.bind(py);
    if bound.is_none() {
        return false;
    }
    match canonical {
        // The detector's own "no detector" spelling.
        "od_type" => !bound
            .extract::<String>()
            .is_ok_and(|s| s.eq_ignore_ascii_case("None")),
        // 0 is the default and leaves IncToDec / Wilcoxon inactive.
        "od_pval" => !bound.extract::<f64>().is_ok_and(|v| v == 0.0),
        // `False` is the default and truncates nothing.
        "use_best_model" => bound.extract::<bool>().unwrap_or(true),
        // `od_wait` and `early_stopping_rounds` have no inert numeric value —
        // any count is a real request to stop early — so only `None` (handled
        // above) disarms them.
        _ => true,
    }
}

/// The remedy sentence [`validate_eval_set_only_params`] appends for a `fit`
/// call that was given no `eval_set`.
pub(crate) const EVAL_SET_REMEDY_FIT: &str =
    "Pass `eval_set=(X_valid, y_valid)` (or a Pool) to `fit`, or drop it.";

/// The remedy sentence for `cv()`, which fits every fold through the learn-only
/// training path and never builds a validation curve for the detector to read.
pub(crate) const EVAL_SET_REMEDY_CV: &str =
    "`cv()` fits each fold through the learn-only path and never builds a per-iteration \
     validation curve, so this parameter cannot take effect here — drop it. (The fold's \
     held-out split is scored ONCE, after training, to produce the reported metrics.)";

/// The remedy sentence for `grid_search` / `randomized_search`, which evaluate
/// each candidate through the same learn-only path.
pub(crate) const EVAL_SET_REMEDY_SEARCH: &str =
    "`grid_search` / `randomized_search` fit every candidate through the learn-only path \
     and never build a per-iteration validation curve, so this parameter cannot take \
     effect here — drop it.";

/// Reject the EVAL-SET-ONLY params when training will run WITHOUT a validation
/// curve (PARAM-02, threat T-08-05) — but only when they are set to a value that
/// would actually take effect ([`eval_set_only_param_is_active`]).
///
/// `od_type` / `od_wait` / `od_pval` / `early_stopping_rounds` / `use_best_model`
/// all act on the per-iteration validation curve. Without one the trainer runs to
/// completion and returns the full model — a caller who asked for early stopping
/// gets none, with nothing to notice. This is exactly the "silently ignored
/// parameter" the honesty policy forbids, so it raises instead. A caller who
/// passed the param's OFF value loses nothing, so there is nothing to warn them
/// about.
///
/// `remedy` names what the caller can do about it, because the answer differs by
/// entry point — see the `EVAL_SET_REMEDY_*` constants. EVERY entry point that
/// reaches the learn-only training path must call this: `fit` alone did, so
/// promoting these params to IMPLEMENTED made `cv` / `grid_search` /
/// `randomized_search` start ACCEPTING them and silently dropping them, while the
/// same param on `fit` raised — the worst of both worlds, since the raise on one
/// path is what convinces the user the param works on the others.
///
/// # Errors
/// `CatBoostParameterError` naming the first ACTIVE eval-set-only param present.
pub(crate) fn validate_eval_set_only_params(
    py: Python<'_>,
    params: &BTreeMap<String, Py<PyAny>>,
    remedy: &str,
) -> PyResult<()> {
    for (name, value) in params {
        let canonical = resolve_alias(name);
        if EVAL_SET_ONLY.contains(&canonical) && eval_set_only_param_is_active(py, canonical, value)
        {
            return Err(CatBoostParameterError::new_err(format!(
                "parameter `{name}` only takes effect when training is given a validation \
                 set: it acts on the per-iteration validation metric, which this fit never \
                 computes. {remedy}"
            )));
        }
    }
    Ok(())
}

/// Whether `name` is present in `params` with a value that is not the Python
/// literal `None`. See [`get`] for why `None` counts as absent.
fn is_set(params: &BTreeMap<String, Py<PyAny>>, py: Python<'_>, name: &str) -> bool {
    params.get(name).is_some_and(|v| !v.bind(py).is_none())
}

/// Read a stored param (alias-resolved at the call site) as `T` via `extract`.
///
/// The Python literal `None` is treated as ABSENT rather than extracted.
/// Upstream CatBoost spells "this parameter is not set" as `None` for every
/// optional parameter — it is the declared default of `auto_class_weights`,
/// `od_type`, `eval_metric`, `monotone_constraints`, `early_stopping_rounds` and
/// many others — so `None` reaching an `extract::<String>()` is not a user error,
/// it is the ordinary value. Extracting it unconditionally raised
/// `TypeError: 'NoneType' object cannot be converted to 'PyString'` from PyO3
/// before any of this crate's own parsing (and its messages) could run, which
/// broke the two commonest sklearn idioms: an explicit `param=None` default in
/// wrapper code, and the `sklearn.clone(est)` / `set_params(**est.get_params())`
/// round-trip that materializes every declared default. Grids such as
/// `{"auto_class_weights": [None, "Balanced"]}` hit it too.
///
/// Skipping the param leaves the builder on its own default, which is exactly
/// what "not set" means.
fn get<'py, T>(params: &'py BTreeMap<String, Py<PyAny>>, py: Python<'py>, name: &str) -> PyResult<Option<T>>
where
    T: FromPyObject<'py, 'py>,
    PyErr: From<<T as FromPyObject<'py, 'py>>::Error>,
{
    match params.get(name) {
        Some(v) => {
            let bound = v.bind(py);
            if bound.is_none() {
                return Ok(None);
            }
            Ok(Some(bound.extract::<T>()?))
        }
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

/// Map an `od_type` string onto an [`EOverfittingDetectorType`] (PARAM-02).
fn parse_od_type(name: &str) -> PyResult<EOverfittingDetectorType> {
    match name {
        "IncToDec" => Ok(EOverfittingDetectorType::IncToDec),
        "Iter" => Ok(EOverfittingDetectorType::Iter),
        // Not an upstream `od_type` value, but the engine implements the
        // detector, so it is reachable here rather than silently absent.
        "Wilcoxon" => Ok(EOverfittingDetectorType::Wilcoxon),
        "None" => Ok(EOverfittingDetectorType::None),
        other => Err(CatBoostParameterError::new_err(format!(
            "unknown od_type `{other}`; expected IncToDec, Iter, Wilcoxon or None"
        ))),
    }
}

/// Map a `boosting_type` string onto an [`EBoostingType`] (PARAM-02).
fn parse_boosting_type(name: &str) -> PyResult<EBoostingType> {
    match name {
        "Plain" => Ok(EBoostingType::Plain),
        "Ordered" => Ok(EBoostingType::Ordered),
        other => Err(CatBoostParameterError::new_err(format!(
            "unknown boosting_type `{other}`; expected Plain or Ordered"
        ))),
    }
}

/// Map a `grow_policy` string onto an [`EGrowPolicy`] (PARAM-02).
///
/// `Region` is REJECTED here by name with its own reason rather than passed
/// through to the engine's `validate_grow_policy`: the engine's rejection fires
/// deep inside `fit`, after ingestion, with an internal-sounding message.
fn parse_grow_policy(name: &str) -> PyResult<EGrowPolicy> {
    match name {
        "SymmetricTree" => Ok(EGrowPolicy::SymmetricTree),
        "Lossguide" => Ok(EGrowPolicy::Lossguide),
        "Depthwise" => Ok(EGrowPolicy::Depthwise),
        "Region" => Err(CatBoostParameterError::new_err(
            "grow_policy `Region` is not implemented in catboost-rs (parity gap); \
             supported policies are SymmetricTree, Lossguide, Depthwise",
        )),
        other => Err(CatBoostParameterError::new_err(format!(
            "unknown grow_policy `{other}`; expected SymmetricTree, Lossguide or Depthwise"
        ))),
    }
}

/// Map an `eval_metric` descriptor string onto an [`EvalMetric`] (PARAM-02),
/// reusing the engine's own parser so the accepted vocabulary (including the
/// parametric forms like `"Quantile:alpha=0.9"` and `"NDCG:top=5"`) cannot drift
/// from what the trainer actually computes.
fn parse_eval_metric(descr: &str) -> PyResult<EvalMetric> {
    parse_metric(descr).map_err(|e| {
        CatBoostParameterError::new_err(format!("eval_metric `{descr}` is not supported: {e}"))
    })
}

/// Extract a `monotone_constraints` kwarg into the engine's per-float-feature
/// `Vec<i8>` (PARAM-02).
///
/// Upstream accepts three shapes, all supported here:
///  - a LIST / tuple of ints, one per feature: `[1, 0, -1]`;
///  - a STRING of comma-separated ints, optionally parenthesised: `"(1,0,-1)"`;
///  - a DICT keyed by feature INDEX: `{0: 1, 2: -1}` — expanded to a dense
///    vector of `max(index) + 1` entries, unlisted features free (`0`).
///
/// A dict keyed by feature NAME is REJECTED: the engine indexes constraints
/// positionally and a `Pool` built from NumPy carries no feature names, so
/// resolving a name would require guessing. Silently dropping named entries
/// would train an unconstrained model the caller believes is constrained.
fn extract_monotone_constraints(
    params: &BTreeMap<String, Py<PyAny>>,
    py: Python<'_>,
) -> PyResult<Option<Vec<i8>>> {
    let Some(obj) = params.get("monotone_constraints") else {
        return Ok(None);
    };
    let bound = obj.bind(py);
    // `None` is upstream's declared default and means "unconstrained" — the same
    // absent-not-invalid rule `get` applies. This reads the store directly rather
    // than through `get` (the three accepted shapes have no single `T`), so the
    // check has to be repeated here. Without it, `monotone_constraints=None` fell
    // through every shape arm to "must be a list of -1/0/1", telling the caller to
    // pass a list when they had explicitly asked for no constraints.
    if bound.is_none() {
        return Ok(None);
    }

    if let Ok(list) = bound.extract::<Vec<i64>>() {
        return list.into_iter().map(to_constraint).collect::<PyResult<_>>().map(Some);
    }
    if let Ok(text) = bound.extract::<String>() {
        let trimmed = text.trim().trim_start_matches(['(', '[']).trim_end_matches([')', ']']);
        let mut out = Vec::new();
        for token in trimmed.split(',') {
            let token = token.trim();
            if token.is_empty() {
                continue;
            }
            let v: i64 = token.parse().map_err(|_| {
                CatBoostParameterError::new_err(format!(
                    "monotone_constraints `{text}`: `{token}` is not an integer"
                ))
            })?;
            out.push(to_constraint(v)?);
        }
        return Ok(Some(out));
    }
    if let Ok(map) = bound.extract::<BTreeMap<usize, i64>>() {
        let width = map.keys().copied().max().map_or(0, |m| m + 1);
        let mut out = vec![0i8; width];
        for (idx, v) in map {
            // `idx < width` by construction (width is max(idx) + 1), so this
            // never silently drops an entry.
            if let Some(slot) = out.get_mut(idx) {
                *slot = to_constraint(v)?;
            }
        }
        return Ok(Some(out));
    }
    Err(CatBoostParameterError::new_err(
        "monotone_constraints must be a list of -1/0/1 (one per float feature), a \
         comma-separated string such as \"1,0,-1\", or a dict keyed by feature INDEX \
         such as {0: 1, 2: -1}. A dict keyed by feature NAME is not supported: the \
         engine applies constraints positionally and a Pool carries no feature names.",
    ))
}

/// Narrow one monotone-constraint value to the `{-1, 0, 1}` the engine expects.
fn to_constraint(v: i64) -> PyResult<i8> {
    match v {
        -1 => Ok(-1),
        0 => Ok(0),
        1 => Ok(1),
        other => Err(CatBoostParameterError::new_err(format!(
            "monotone_constraints entry {other} is invalid; expected -1 (non-increasing), \
             0 (free) or 1 (non-decreasing)"
        ))),
    }
}

/// Map an `auto_class_weights` string onto an [`AutoClassWeights`] (PARAM-03).
///
/// Upstream spells "off" as `None` (the Python literal `None`, i.e. the kwarg
/// absent) — the string `"None"` is accepted too so an explicit round-trip
/// through `get_params()` works.
fn parse_auto_class_weights(name: &str) -> PyResult<AutoClassWeights> {
    match name {
        "None" => Ok(AutoClassWeights::None),
        "Balanced" => Ok(AutoClassWeights::Balanced),
        "SqrtBalanced" => Ok(AutoClassWeights::SqrtBalanced),
        other => Err(CatBoostParameterError::new_err(format!(
            "unknown auto_class_weights `{other}`; expected Balanced, SqrtBalanced or None"
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

    // ---- PARAM-02: the overfitting-detector / best-model surface ------------
    //
    // ORDER MATTERS. `early_stopping_rounds` is upstream shorthand for the PAIR
    // (od_type=Iter, od_wait=rounds), so it is applied FIRST and an explicit
    // `od_type` / `od_wait` alongside it is rejected outright (below) rather than
    // resolved by precedence — either resolution silently discards one of two
    // parameters the user explicitly set.
    // Presence is `is_set`, not `contains_key`: a stored `None` is upstream's
    // "not set" spelling (see `get`), so `od_type=None` alongside
    // `early_stopping_rounds=10` is NOT the ambiguous both-forms combination — it
    // is the shorthand alone, which is exactly what an sklearn `clone` round-trip
    // reconstructs.
    let has_early_stop = is_set(params, py, "early_stopping_rounds");
    let has_explicit_od = is_set(params, py, "od_type") || is_set(params, py, "od_wait");
    if has_early_stop && has_explicit_od {
        return Err(CatBoostParameterError::new_err(
            "`early_stopping_rounds` is shorthand for `od_type=\"Iter\"` + `od_wait=<rounds>`; \
             passing it together with an explicit `od_type` / `od_wait` is ambiguous. Pass \
             either the shorthand or the pair, not both.",
        ));
    }
    if let Some(v) = get_with_aliases::<usize>(params, py, "early_stopping_rounds")? {
        // >= 1: waiting zero non-improving rounds would stop at the first
        // iteration that fails to improve, which upstream does not permit.
        check_range("early_stopping_rounds", v as f64, 1.0, f64::INFINITY, true)?;
        builder = builder.early_stopping_rounds(v);
    }
    if let Some(v) = get_with_aliases::<String>(params, py, "od_type")? {
        builder = builder.od_type(parse_od_type(&v)?);
    }
    if let Some(v) = get_with_aliases::<f64>(params, py, "od_pval")? {
        // A p-value threshold; 0 (the default) makes IncToDec/Wilcoxon inactive.
        check_range("od_pval", v, 0.0, 1.0, true)?;
        builder = builder.od_pval(v);
    }
    if let Some(v) = get_with_aliases::<usize>(params, py, "od_wait")? {
        builder = builder.od_wait(v);
    }
    if let Some(v) = get_with_aliases::<bool>(params, py, "use_best_model")? {
        builder = builder.use_best_model(v);
    }
    if let Some(v) = get_with_aliases::<String>(params, py, "eval_metric")? {
        builder = builder.eval_metric(parse_eval_metric(&v)?);
    }

    // ---- PARAM-02: the boosting-scheme controls ----------------------------
    if let Some(v) = get_with_aliases::<String>(params, py, "boosting_type")? {
        builder = builder.boosting_type(parse_boosting_type(&v)?);
    }
    if let Some(v) = get_with_aliases::<bool>(params, py, "has_time")? {
        builder = builder.has_time(v);
    }
    if let Some(v) = get_with_aliases::<f64>(params, py, "fold_len_multiplier")? {
        // Upstream requires fold_len_multiplier > 1 (a multiplier of 1 would never
        // grow the tail, so the dynamic fold would not advance).
        check_range("fold_len_multiplier", v, 1.0, f64::INFINITY, false)?;
        builder = builder.fold_len_multiplier(v);
    }

    // ---- PARAM-02: the grow policy -----------------------------------------
    if let Some(v) = get_with_aliases::<String>(params, py, "grow_policy")? {
        builder = builder.grow_policy(parse_grow_policy(&v)?);
    }
    if let Some(v) = get_with_aliases::<usize>(params, py, "max_leaves")? {
        // >= 2: a one-leaf "tree" is a constant, which the grower cannot produce.
        check_range("max_leaves", v as f64, 2.0, 65536.0, true)?;
        builder = builder.max_leaves(v);
    }
    if let Some(v) = get_with_aliases::<usize>(params, py, "min_data_in_leaf")? {
        builder = builder.min_data_in_leaf(v);
    }

    // ---- PARAM-02: the per-feature weighting / penalty surface -------------
    if let Some(v) = get_with_aliases::<Vec<f64>>(params, py, "feature_weights")? {
        builder = builder.feature_weights(v);
    }
    if let Some(v) = get_with_aliases::<Vec<f64>>(params, py, "first_feature_use_penalties")? {
        builder = builder.first_feature_use_penalties(v);
    }
    if let Some(v) = get_with_aliases::<Vec<f64>>(params, py, "per_object_feature_penalties")? {
        builder = builder.per_object_feature_penalties(v);
    }
    if let Some(v) = get_with_aliases::<f64>(params, py, "penalties_coefficient")? {
        check_range("penalties_coefficient", v, 0.0, f64::INFINITY, true)?;
        builder = builder.penalties_coefficient(v);
    }
    if let Some(v) = extract_monotone_constraints(params, py)? {
        builder = builder.monotone_constraints(v);
    }

    // ---- PARAM-03: the class-weight family + ignored_features --------------
    //
    // The three class-weight controls are NOT cross-validated here: the facade's
    // `resolve_weights` already rejects the combination (and rejects a
    // classification-only control on a regression loss) with a message naming the
    // conflict, and duplicating that rule in two places is how the two copies
    // drift apart. Only per-value validation lives here.
    if let Some(v) = get_with_aliases::<Vec<f64>>(params, py, "class_weights")? {
        for (i, w) in v.iter().enumerate() {
            check_range(&format!("class_weights[{i}]"), *w, 0.0, f64::INFINITY, true)?;
        }
        builder = builder.class_weights(v);
    }
    if let Some(v) = get_with_aliases::<String>(params, py, "auto_class_weights")? {
        builder = builder.auto_class_weights(parse_auto_class_weights(&v)?);
    }
    if let Some(v) = get_with_aliases::<f64>(params, py, "scale_pos_weight")? {
        // > 0: a zero or negative positive-class weight would erase or invert the
        // class's contribution rather than reweight it.
        check_range("scale_pos_weight", v, 0.0, f64::INFINITY, false)?;
        builder = builder.scale_pos_weight(v);
    }
    if let Some(v) = get_with_aliases::<Vec<usize>>(params, py, "ignored_features")? {
        builder = builder.ignored_features(v);
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
