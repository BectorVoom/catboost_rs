//! Unit tests for [`CatBoostBuilder`]'s categorical/CTR setters (F01–F06).
//!
//! Mounted as a **CHILD** module of `builder.rs` (`#[path]`), not a sibling in
//! `lib.rs`: `boost_params()` is a PRIVATE `fn` (`builder.rs`), so only a child
//! module can see it. Precedent: `crates/catboost-rs/src/cv.rs`'s `cv_test`
//! mount.

use super::CatBoostBuilder;
use cb_train::{
    boosting_type_default, combinations_ctr_default, combinations_ctr_priors_default,
    counter_calc_method_default, feature_weights_default, first_feature_use_penalties_default,
    fold_len_multiplier_default, grow_policy_default, has_time_default, max_ctr_complexity_default,
    max_leaves_default, min_data_in_leaf_default, monotone_constraints_default,
    one_hot_max_size_default, penalties_coefficient_default, per_object_feature_penalties_default,
    permutation_count_default, simple_ctr_default, simple_ctr_priors_default, CounterCalcMethod,
    EBoostingType, ECtrType, EGrowPolicy, EOverfittingDetectorType, EvalMetric,
};

/// F01 — `one_hot_max_size` is settable and reaches `BoostParams`.
#[test]
fn one_hot_max_size_setter_reaches_boost_params() {
    let default_params = CatBoostBuilder::new().boost_params();
    assert_eq!(
        default_params.one_hot_max_size,
        one_hot_max_size_default(),
        "the unset builder must still emit the upstream default"
    );

    let params = CatBoostBuilder::new().one_hot_max_size(7).boost_params();
    assert_eq!(params.one_hot_max_size, 7);
}

/// F02 — `max_ctr_complexity` is settable and reaches `BoostParams`.
#[test]
fn max_ctr_complexity_setter_reaches_boost_params() {
    let default_params = CatBoostBuilder::new().boost_params();
    assert_eq!(
        default_params.max_ctr_complexity,
        max_ctr_complexity_default()
    );

    let params = CatBoostBuilder::new().max_ctr_complexity(1).boost_params();
    assert_eq!(params.max_ctr_complexity, 1);
}

/// F03 — `simple_ctr` and `simple_ctr_priors` are settable in lockstep.
#[test]
fn simple_ctr_and_priors_setters_reach_boost_params() {
    let default_params = CatBoostBuilder::new().boost_params();
    assert_eq!(default_params.simple_ctr, simple_ctr_default());
    assert_eq!(default_params.simple_ctr_priors, simple_ctr_priors_default());

    let params = CatBoostBuilder::new()
        .simple_ctr(ECtrType::Counter)
        .simple_ctr_priors(vec![0.0, 0.5, 1.0])
        .boost_params();
    assert_eq!(params.simple_ctr, ECtrType::Counter);
    assert_eq!(params.simple_ctr_priors, vec![0.0, 0.5, 1.0]);
}

/// F04 — `combinations_ctr` and `combinations_ctr_priors` are settable in
/// lockstep, and are INDEPENDENT of the simple-CTR pair (a cross-wire between
/// the two families is exactly what this asserts against).
#[test]
fn combinations_ctr_and_priors_setters_reach_boost_params() {
    let default_params = CatBoostBuilder::new().boost_params();
    assert_eq!(default_params.combinations_ctr, combinations_ctr_default());
    assert_eq!(
        default_params.combinations_ctr_priors,
        combinations_ctr_priors_default()
    );

    let params = CatBoostBuilder::new()
        .simple_ctr(ECtrType::Buckets)
        .simple_ctr_priors(vec![0.25])
        .combinations_ctr(ECtrType::BinarizedTargetMeanValue)
        .combinations_ctr_priors(vec![0.75, 1.5])
        .boost_params();
    assert_eq!(params.combinations_ctr, ECtrType::BinarizedTargetMeanValue);
    assert_eq!(params.combinations_ctr_priors, vec![0.75, 1.5]);
    // The simple pair must NOT have been overwritten by the combination pair.
    assert_eq!(params.simple_ctr, ECtrType::Buckets);
    assert_eq!(params.simple_ctr_priors, vec![0.25]);
}

/// F05 — `counter_calc_method` is settable and reaches `BoostParams`.
#[test]
fn counter_calc_method_setter_reaches_boost_params() {
    let default_params = CatBoostBuilder::new().boost_params();
    assert_eq!(
        default_params.counter_calc_method,
        counter_calc_method_default()
    );

    let params = CatBoostBuilder::new()
        .counter_calc_method(CounterCalcMethod::Full)
        .boost_params();
    assert_eq!(params.counter_calc_method, CounterCalcMethod::Full);
}

/// F06 — the default-equivalence guard: an untouched builder's `boost_params()`
/// must equal the canonical upstream defaults FIELD BY FIELD for every one of
/// the seven newly-promoted fields.
///
/// This is the guard the two mandated mutations (§3.1) must break:
///  1. a WRITE cross-wire — `new()` seeded with a non-default value;
///  2. a READ cross-wire — `boost_params()` reading the wrong `self` field
///     (e.g. `one_hot_max_size: self.max_ctr_complexity as u32`). Mutation 1
///     alone cannot detect this, because `simple_ctr_default()` and
///     `combinations_ctr_default()` both return `Borders`.
#[test]
fn untouched_builder_emits_the_canonical_ctr_defaults() {
    let p = CatBoostBuilder::new().boost_params();
    assert_eq!(p.one_hot_max_size, one_hot_max_size_default());
    assert_eq!(p.max_ctr_complexity, max_ctr_complexity_default());
    assert_eq!(p.simple_ctr, simple_ctr_default());
    assert_eq!(p.simple_ctr_priors, simple_ctr_priors_default());
    assert_eq!(p.combinations_ctr, combinations_ctr_default());
    assert_eq!(p.combinations_ctr_priors, combinations_ctr_priors_default());
    assert_eq!(p.counter_calc_method, counter_calc_method_default());
}

/// F06 (read-cross-wire detector). `one_hot_max_size` and `max_ctr_complexity`
/// are DISTINCT fields with DISTINCT defaults (2 vs 4), so a `boost_params()`
/// that reads one where it means the other is observable. Setting only one and
/// asserting the other is unmoved is what makes mutation 2 fail.
#[test]
fn one_hot_max_size_and_max_ctr_complexity_do_not_cross_wire() {
    let p = CatBoostBuilder::new().one_hot_max_size(9).boost_params();
    assert_eq!(p.one_hot_max_size, 9);
    assert_eq!(
        p.max_ctr_complexity,
        max_ctr_complexity_default(),
        "setting one_hot_max_size must not move max_ctr_complexity"
    );

    let p = CatBoostBuilder::new().max_ctr_complexity(3).boost_params();
    assert_eq!(p.max_ctr_complexity, 3);
    assert_eq!(
        p.one_hot_max_size,
        one_hot_max_size_default(),
        "setting max_ctr_complexity must not move one_hot_max_size"
    );
}

// ─── PARAM-01: the previously-unreachable BoostParams surface ────────────────
//
// Before PARAM-01 these fifteen `BoostParams` fields were PINNED to literals
// inside `boost_params()`. The engine implemented every one of them, but no
// caller — Rust or Python — could set them, so they were dead configuration.
// The tests below are split deliberately:
//
//  * `untouched_builder_emits_the_pinned_engine_defaults` is the NO-REGRESSION
//    gate: it asserts `new()` still emits exactly what the literals pinned, so
//    every existing fit is byte-unchanged.
//  * the per-setter tests assert each value ACTUALLY REACHES `BoostParams` —
//    a setter that stores into the wrong field is what they detect.
//  * `..._do_not_cross_wire` covers the same-typed neighbours (the three `Vec<f64>`
//    penalty vectors, the two `usize` tree bounds), where a copy-paste
//    read-cross-wire is invisible to a single-field assertion.

/// PARAM-01 NO-REGRESSION GATE. An untouched builder must emit precisely the
/// values `boost_params()` used to hardcode, so no existing fit changes.
#[test]
fn untouched_builder_emits_the_pinned_engine_defaults() {
    let p = CatBoostBuilder::new().boost_params();
    // The four eval-set-only controls were pinned to the inert literals.
    assert_eq!(p.od_type, EOverfittingDetectorType::None);
    assert!((p.od_pval - 0.0).abs() < f64::EPSILON);
    assert_eq!(p.od_wait, 0);
    assert!(!p.use_best_model);
    assert_eq!(p.eval_metric, None);
    // The boosting-scheme controls were pinned to their upstream defaults.
    assert_eq!(p.boosting_type, boosting_type_default());
    assert_eq!(p.has_time, has_time_default());
    assert_eq!(p.permutation_count, permutation_count_default());
    assert!((p.fold_len_multiplier - fold_len_multiplier_default()).abs() < f64::EPSILON);
    // The penalty surface was pinned to the (empty) upstream defaults.
    assert_eq!(p.feature_weights, feature_weights_default());
    assert_eq!(
        p.first_feature_use_penalties,
        first_feature_use_penalties_default()
    );
    assert_eq!(
        p.per_object_feature_penalties,
        per_object_feature_penalties_default()
    );
    assert!((p.penalties_coefficient - penalties_coefficient_default()).abs() < f64::EPSILON);
    assert_eq!(p.monotone_constraints, monotone_constraints_default());
    // The grow policy was pinned to symmetric.
    assert_eq!(p.grow_policy, grow_policy_default());
    assert_eq!(p.max_leaves, max_leaves_default());
    assert_eq!(p.min_data_in_leaf, min_data_in_leaf_default());
}

/// PARAM-01 — the overfitting-detector triple reaches `BoostParams`.
#[test]
fn od_setters_reach_boost_params() {
    let p = CatBoostBuilder::new()
        .od_type(EOverfittingDetectorType::Wilcoxon)
        .od_pval(0.01)
        .od_wait(7)
        .boost_params();
    assert_eq!(p.od_type, EOverfittingDetectorType::Wilcoxon);
    assert!((p.od_pval - 0.01).abs() < f64::EPSILON);
    assert_eq!(p.od_wait, 7);
}

/// PARAM-01 — `early_stopping_rounds` sets BOTH halves of the upstream pair.
///
/// The point of the setter is that `od_wait` alone is inert: with `od_type`
/// left at `None` the detector never fires, so a caller who set only the wait
/// would get no early stopping and no error. This asserts the type moves too.
#[test]
fn early_stopping_rounds_sets_both_the_detector_and_the_wait() {
    let p = CatBoostBuilder::new().early_stopping_rounds(25).boost_params();
    assert_eq!(
        p.od_type,
        EOverfittingDetectorType::Iter,
        "early_stopping_rounds must turn the detector ON, not just set the wait"
    );
    assert_eq!(p.od_wait, 25);
}

/// PARAM-01 — `use_best_model` and `eval_metric` reach `BoostParams`.
#[test]
fn use_best_model_and_eval_metric_reach_boost_params() {
    let p = CatBoostBuilder::new()
        .use_best_model(true)
        .eval_metric(EvalMetric::Mae)
        .boost_params();
    assert!(p.use_best_model);
    assert_eq!(p.eval_metric, Some(EvalMetric::Mae));
}

/// PARAM-01 — the boosting-scheme controls reach `BoostParams`.
#[test]
fn boosting_scheme_setters_reach_boost_params() {
    let p = CatBoostBuilder::new()
        .boosting_type(EBoostingType::Ordered)
        .has_time(true)
        .permutation_count(2)
        .fold_len_multiplier(1.5)
        .boost_params();
    assert_eq!(p.boosting_type, EBoostingType::Ordered);
    assert!(p.has_time);
    assert_eq!(p.permutation_count, 2);
    assert!((p.fold_len_multiplier - 1.5).abs() < f64::EPSILON);
}

/// PARAM-01 — the grow-policy trio reaches `BoostParams`.
#[test]
fn grow_policy_setters_reach_boost_params() {
    let p = CatBoostBuilder::new()
        .grow_policy(EGrowPolicy::Lossguide)
        .max_leaves(12)
        .min_data_in_leaf(5)
        .boost_params();
    assert_eq!(p.grow_policy, EGrowPolicy::Lossguide);
    assert_eq!(p.max_leaves, 12);
    assert_eq!(p.min_data_in_leaf, 5);
}

/// PARAM-01 (read-cross-wire detector). `max_leaves` and `min_data_in_leaf` are
/// both `usize` with DISTINCT defaults (31 vs 1), so a `boost_params()` that
/// reads one where it means the other is observable only by setting one and
/// asserting the other is unmoved.
#[test]
fn max_leaves_and_min_data_in_leaf_do_not_cross_wire() {
    let p = CatBoostBuilder::new().max_leaves(12).boost_params();
    assert_eq!(p.max_leaves, 12);
    assert_eq!(
        p.min_data_in_leaf,
        min_data_in_leaf_default(),
        "setting max_leaves must not move min_data_in_leaf"
    );

    let p = CatBoostBuilder::new().min_data_in_leaf(5).boost_params();
    assert_eq!(p.min_data_in_leaf, 5);
    assert_eq!(
        p.max_leaves,
        max_leaves_default(),
        "setting min_data_in_leaf must not move max_leaves"
    );
}

/// PARAM-01 — the penalty surface reaches `BoostParams`.
#[test]
fn penalty_setters_reach_boost_params() {
    let p = CatBoostBuilder::new()
        .feature_weights(vec![1.0, 2.0, 3.0])
        .first_feature_use_penalties(vec![0.1, 0.2])
        .per_object_feature_penalties(vec![0.01])
        .penalties_coefficient(2.5)
        .monotone_constraints(vec![1, 0, -1])
        .boost_params();
    assert_eq!(p.feature_weights, vec![1.0, 2.0, 3.0]);
    assert_eq!(p.first_feature_use_penalties, vec![0.1, 0.2]);
    assert_eq!(p.per_object_feature_penalties, vec![0.01]);
    assert!((p.penalties_coefficient - 2.5).abs() < f64::EPSILON);
    assert_eq!(p.monotone_constraints, vec![1, 0, -1]);
}

/// PARAM-01 (read-cross-wire detector). The three penalty/weight vectors are ALL
/// `Vec<f64>` and ALL default to empty, so a setter or a `boost_params()` read
/// that confuses two of them is invisible to `penalty_setters_reach_boost_params`
/// (which sets all three at once). Setting exactly ONE and asserting the other
/// two are still empty is what detects it.
#[test]
fn the_three_penalty_vectors_do_not_cross_wire() {
    let p = CatBoostBuilder::new()
        .feature_weights(vec![9.0])
        .boost_params();
    assert_eq!(p.feature_weights, vec![9.0]);
    assert!(
        p.first_feature_use_penalties.is_empty() && p.per_object_feature_penalties.is_empty(),
        "feature_weights must not leak into either penalty vector"
    );

    let p = CatBoostBuilder::new()
        .first_feature_use_penalties(vec![9.0])
        .boost_params();
    assert_eq!(p.first_feature_use_penalties, vec![9.0]);
    assert!(
        p.feature_weights.is_empty() && p.per_object_feature_penalties.is_empty(),
        "first_feature_use_penalties must not leak into the other two vectors"
    );

    let p = CatBoostBuilder::new()
        .per_object_feature_penalties(vec![9.0])
        .boost_params();
    assert_eq!(p.per_object_feature_penalties, vec![9.0]);
    assert!(
        p.feature_weights.is_empty() && p.first_feature_use_penalties.is_empty(),
        "per_object_feature_penalties must not leak into the other two vectors"
    );
}
