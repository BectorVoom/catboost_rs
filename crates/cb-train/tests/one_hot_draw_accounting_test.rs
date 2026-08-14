//! T01a / SPEC-OH-27 — the one-hot RSM draw-order ground-truth artifact.
//!
//! This is the SCAFFOLDING half: it asserts only that the ground-truth artifact
//! exists and states a machine-readable verdict. The behavioral draw-count
//! assertion lives in T01b, where a production consumer exists.
//!
//! Why this matters: a one-hot-routed categorical column changes the number of
//! candidate sub-lists at each tree level, and upstream charges one unconditional
//! RNG draw per sub-list. Getting the count wrong desynchronises every subsequent
//! tree's bootstrap sample — the same defect class as the two fabricated MVS draws
//! fixed in `d7676b5`, which passed every non-bootstrap test.

use std::path::PathBuf;

fn ground_truth_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../.planning/plans/one-hot-categorical-training")
        .join("instrumented-ground-truth/ONE_HOT_GROUND_TRUTH.md")
}

/// The artifact exists and states exactly one of the three permitted verdicts.
/// A missing artifact, or one that hedges without committing, is a failure —
/// T01b consumes this verdict to decide between enforcing the rule and
/// typed-rejecting one-hot × bootstrap.
#[test]
fn one_hot_ground_truth_artifact_is_present_and_states_a_verdict() {
    let path = ground_truth_path();
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("ground-truth artifact missing at {}: {e}", path.display()));

    let verdicts = [
        "RSM_RULE: n_float + n_one_hot",
        "RSM_RULE: n_float",
        "STATUS: NOT-ESTABLISHED",
    ];
    let found: Vec<&str> = verdicts
        .iter()
        .copied()
        .filter(|v| text.contains(v))
        .collect();

    assert!(
        !found.is_empty(),
        "ONE_HOT_GROUND_TRUTH.md must state one of {verdicts:?}"
    );
    // `RSM_RULE: n_float` is a prefix of `RSM_RULE: n_float + n_one_hot`, so the
    // stricter reading wins; assert the artifact is not ambiguous BETWEEN a rule
    // and a non-establishment.
    assert!(
        !(text.contains("STATUS: NOT-ESTABLISHED") && text.contains("RSM_RULE:")),
        "the artifact must not claim BOTH a derived rule and NOT-ESTABLISHED"
    );
}

/// If a rule IS claimed, the artifact must also carry its evidence grade and the
/// compression caveat — a bare verdict line with no provenance is exactly what
/// SPEC-OH-27 forbids ("do NOT guess a draw count").
#[test]
fn a_claimed_rule_carries_its_evidence_grade_and_caveats() {
    let text = std::fs::read_to_string(ground_truth_path()).expect("ground-truth artifact");
    if !text.contains("RSM_RULE:") {
        return; // NOT-ESTABLISHED: nothing further to require.
    }

    assert!(
        text.contains("SOURCE-DERIVED") || text.contains("instrumented"),
        "a claimed rule must state how it was established"
    );
    assert!(
        text.contains("CompressCandidates"),
        "a claimed rule must address the CompressCandidates re-bundling path, \
         which has different draw arithmetic"
    );
    assert!(
        text.contains("greedy_tensor_search.cpp"),
        "a claimed rule must cite the upstream source it was derived from"
    );
}

// ── T01b — the executable half of SPEC-OH-27 ────────────────────────────────
//
// # Which branch this implements, and why
//
// The artifact above records `RSM_RULE: n_float + n_one_hot` with TWO evidence
// grades:
//
//   * the rule for the un-bundled `OneFeature` candidate path — **HIGH**
//     (source-derived from three agreeing sites in upstream 1.2.10's
//     `greedy_tensor_search.cpp`);
//   * the behaviour under **`CompressCandidates`** — **NOT ESTABLISHED**. That
//     pass runs BETWEEN `AddOneHotFeatures` and the draw site and can re-bundle
//     `OneFeature` candidates into `BinarySplits` / `ExclusiveBundle` /
//     `FeaturesGroup` ensembles, each with DIFFERENT draw arithmetic. The
//     artifact is explicit that a cardinality-2 categorical column — precisely
//     the default one-hot shape — "is exactly the shape most likely to be
//     packed", so the caveat is not hypothetical.
//
// The artifact's own "Consequence for T01b" therefore selects branch (b): keep
// one-hot x (`bootstrap_type != No` OR `random_strength != 0`) TYPED-REJECTED
// until an instrumented upstream run settles the compressed case. Consuming the
// un-bundled rule anyway would silently desynchronise every subsequent tree's
// bootstrap sample — the exact defect class fixed in `d7676b5`.
//
// **This does not affect the default path**: the facade defaults are
// `bootstrap_type = No` and `random_strength = 0.0`
// (`crates/catboost-rs/src/builder.rs:107,110`), both draw-inert, so one-hot
// training works by default. Only an EXPLICIT opt-in to draws is refused, and
// `one_hot_with_inert_draws_still_trains` below keeps the gate narrow.

use cb_backend::CpuBackend;
use cb_core::CbError;
use cb_compute::{LeafMethod, Loss};
use cb_train::{
    train_cat, BoostParams, EBoostingType, EBootstrapType, EOverfittingDetectorType,
};

/// A draw-inert one-hot config: `one_hot_max_size = 2` so a binary cat column
/// routes to the one-hot path, `max_ctr_complexity = 0` so nothing routes to
/// CTR.
fn one_hot_params() -> BoostParams {
    BoostParams {
        loss: Loss::Logloss,
        iterations: 3,
        depth: 2,
        learning_rate: 0.1,
        l2_leaf_reg: 3.0,
        random_strength: 0.0,
        boost_from_average: false,
        leaf_method: LeafMethod::Gradient,
        bootstrap_type: EBootstrapType::No,
        subsample: 1.0,
        bagging_temperature: 0.0,
        random_seed: 0,
        od_type: EOverfittingDetectorType::None,
        od_pval: 0.0,
        od_wait: 0,
        use_best_model: false,
        eval_metric: None,
        auto_learning_rate: false,
        one_hot_max_size: 2,
        permutation_count: 1,
        fold_len_multiplier: 2.0,
        simple_ctr: cb_train::simple_ctr_default(),
        simple_ctr_priors: cb_train::simple_ctr_priors_default(),
        counter_calc_method: cb_train::counter_calc_method_default(),
        boosting_type: EBoostingType::Plain,
        max_ctr_complexity: 0,
        combinations_ctr: cb_train::combinations_ctr_default(),
        combinations_ctr_priors: cb_train::combinations_ctr_priors_default(),
        score_function: cb_train::score_function_default(),
        has_time: false,
        feature_weights: cb_train::feature_weights_default(),
        first_feature_use_penalties: cb_train::first_feature_use_penalties_default(),
        per_object_feature_penalties: cb_train::per_object_feature_penalties_default(),
        penalties_coefficient: cb_train::penalties_coefficient_default(),
        monotone_constraints: cb_train::monotone_constraints_default(),
        grow_policy: cb_train::grow_policy_default(),
        max_leaves: cb_train::max_leaves_default(),
        min_data_in_leaf: cb_train::min_data_in_leaf_default(),
        extra: Default::default(),
    }
}

/// A tiny 1-float + 1-binary-cat learn set.
fn one_hot_pool() -> (Vec<Vec<f32>>, Vec<Vec<f64>>, Vec<Vec<String>>, Vec<f64>) {
    let n = 16;
    let floats: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
    let cats: Vec<String> = (0..n)
        .map(|i| if i % 2 == 0 { "a".to_owned() } else { "b".to_owned() })
        .collect();
    let target: Vec<f64> = (0..n).map(|i| f64::from(i % 3 == 0)).collect();
    (vec![floats], vec![vec![0.25, 0.5, 0.75]], vec![cats], target)
}

/// SPEC-OH-27 (branch b) — an EXPLICIT bootstrap opt-in on a one-hot pool is
/// refused with a typed error naming BOTH the feature and the reason, rather
/// than silently fitting against a desynchronised RNG stream.
#[test]
fn one_hot_with_active_draws_is_typed_rejected_until_ground_truth_exists() {
    let (floats, borders, cats, target) = one_hot_pool();
    let mut params = one_hot_params();
    params.bootstrap_type = EBootstrapType::Bayesian;
    params.bagging_temperature = 1.0;

    match train_cat(&CpuBackend, &floats, &borders, &cats, &target, &[], &params, None) {
        Err(CbError::Unsupported(msg)) => {
            assert!(msg.contains("one-hot"), "the error must name one-hot: {msg}");
            assert!(msg.contains("bootstrap"), "the error must name bootstrap: {msg}");
        }
        Err(other) => panic!("expected CbError::Unsupported, got {other:?}"),
        Ok(_) => panic!("one-hot x active bootstrap must be typed-rejected, got Ok(model)"),
    }
}

/// SPEC-OH-27 (branch b) — the same gate fires on `random_strength != 0`, which
/// activates the perturbed search's own `SelectBestCandidate` normal draws (one
/// per LISTED feature) and so has the identical un-established accounting.
#[test]
fn one_hot_with_random_strength_is_typed_rejected() {
    let (floats, borders, cats, target) = one_hot_pool();
    let mut params = one_hot_params();
    params.random_strength = 1.0;

    match train_cat(&CpuBackend, &floats, &borders, &cats, &target, &[], &params, None) {
        Err(CbError::Unsupported(msg)) => {
            assert!(msg.contains("one-hot"), "the error must name one-hot: {msg}");
            assert!(
                msg.contains("random_strength"),
                "the error must name random_strength: {msg}"
            );
        }
        Err(other) => panic!("expected CbError::Unsupported, got {other:?}"),
        Ok(_) => panic!("one-hot x random_strength must be typed-rejected, got Ok(model)"),
    }
}

/// The gate must be NARROW: the draw-inert default config (`bootstrap_type =
/// No`, `random_strength = 0`) still trains a one-hot pool. Without this, the
/// two gates above could be satisfied by refusing one-hot training outright.
#[test]
fn one_hot_with_inert_draws_still_trains() {
    let (floats, borders, cats, target) = one_hot_pool();
    let params = one_hot_params();

    let got = train_cat(&CpuBackend, &floats, &borders, &cats, &target, &[], &params, None);
    assert!(
        got.is_ok(),
        "the draw-inert default one-hot path must still train, got {:?}",
        got.err()
    );
}

/// The gate must not fire on a pool with NO one-hot-routed column: a
/// high-cardinality (CTR-routed) cat column with an active bootstrap is
/// unaffected, so the pre-existing bootstrap x CTR behaviour is preserved.
#[test]
fn active_draws_without_a_one_hot_column_are_unaffected() {
    let (floats, borders, _cats, target) = one_hot_pool();
    // Every value distinct -> cardinality 16 > one_hot_max_size 2 -> CTR route.
    let cats: Vec<Vec<String>> = vec![(0..16).map(|i| format!("v{i}")).collect()];
    let mut params = one_hot_params();
    params.bootstrap_type = EBootstrapType::Bayesian;
    params.bagging_temperature = 1.0;

    let got = train_cat(&CpuBackend, &floats, &borders, &cats, &target, &[], &params, None);
    assert!(
        got.is_ok(),
        "a CTR-routed cat column with an active bootstrap must be unaffected, got {:?}",
        got.err()
    );
}
