//! FEAT-03 monotone-constraint train→predict oracle (Phase 06.6 plan 02): train a
//! plain boosted oblivious-tree (SymmetricTree) RMSE model on the frozen
//! `numeric_tiny` inputs with `monotone_constraints=[-1,0,1,0]` (feature 0
//! NON-INCREASING, feature 2 NON-DECREASING) and gate per-tree splits, per-tree
//! leaf values, per-iteration staged approximants, and final raw predictions
//! against the committed upstream catboost 1.2.10 fixture at <= 1e-5 (D-08,
//! D-6.6-06, D-6.6-08).
//!
//! The fixture pins `model_shrink_rate=0` (CatBoost auto-enables model shrinkage
//! under monotone constraints, which would otherwise confound the projection with
//! an unrelated per-tree decay) so the isotonic (PAVA) leaf-value post-pass is the
//! ONLY difference vs an unconstrained model. The constraints GENUINELY BIND (the
//! unconstrained leaves violate them; max leaf diff ~3.9e-1), so this is a
//! non-vacuous oracle.
//!
//! Monotone constraints are enforced as an isotonic (PAVA) projection over the
//! per-leaf DELTAS during leaf estimation (`CalcMonotonicLeafDeltasSimple`,
//! `approx_calcer.cpp:551`), AFTER the structure is built — so the SPLITS are
//! UNAFFECTED (we assert them too as a sanity lock) and only the LEAF VALUES
//! change versus an unconstrained model. The fixture generator
//! (`gen_monotone_fixtures.py`) verifies the constrained predictions DIFFER from
//! the unconstrained baseline, so this is a non-vacuous oracle.
//!
//! The fixture is generated OFFLINE from the `.venv` catboost 1.2.10
//! (`crates/cb-oracle/generator/gen_monotone_fixtures.py`, pinned `random_seed=0`,
//! `thread_count=1`); CI only READS the committed `.npy` / `model.json`.
//!
//! Integration test (under `tests/`) so it can depend on `cb-oracle` / `cb-model`;
//! the top-line `#![allow(...)]` mirrors `penalty_oracle_test.rs:18`.
//!
//! NOTE on test LOCATION: the plan listed this file under `crates/cb-compute/tests/`,
//! but the train→predict oracle harness (`train`, `compare_stage`, `predict_raw`,
//! the `fixture()` resolver) lives in `cb-train` + `cb-oracle` — `cb-compute` has
//! no training entry point. It therefore lives in `cb-train/tests/` alongside the
//! analogous FEAT-04 `penalty_oracle_test.rs` (the prior wave made the identical
//! placement). The leaf-level isotonic primitives ARE unit-tested in
//! `cb-compute/src/leaf_test.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_backend::CpuBackend;
use cb_compute::{LeafMethod, Loss};
use cb_model::{predict_raw, Model as CbModel};
use cb_oracle::{compare_stage, load_f64_vec, load_model_json, Stage};
use cb_train::{
    train, BoostParams, EBootstrapType, EGrowPolicy, EOverfittingDetectorType, Model,
};
use ndarray::Array2;
use ndarray_npy::read_npy;

/// Resolve a path under `cb-oracle/fixtures/` from cb-train's manifest dir.
fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join(rel)
}

/// Load the `numeric_tiny` input matrix as per-feature `f32` SoA columns.
fn load_feature_columns() -> Vec<Vec<f32>> {
    let x: Array2<f64> = read_npy(fixture("inputs/numeric_tiny/X.npy"))
        .unwrap_or_else(|e| panic!("numeric_tiny/X.npy must load: {e:?}"));
    (0..x.ncols())
        .map(|fi| x.column(fi).iter().map(|&v| v as f32).collect())
        .collect()
}

/// Load the raw `numeric_tiny` regression target.
fn load_target() -> Vec<f64> {
    load_f64_vec(&fixture("inputs/numeric_tiny/y.npy")).unwrap()
}

/// Build the first-slice simplified isolating [`BoostParams`] (mirrors the
/// generator's `ISOLATING_PARAMS`), overriding only `monotone_constraints`.
fn isolating_params(monotone_constraints: Vec<i8>) -> BoostParams {
    BoostParams {
        loss: Loss::Rmse,
        iterations: 5,
        depth: 2,
        learning_rate: 0.1,
        l2_leaf_reg: 3.0,
        random_strength: 0.0,
        boost_from_average: true,
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
        one_hot_max_size: cb_train::one_hot_max_size_default(),
        permutation_count: cb_train::permutation_count_default(),
        fold_len_multiplier: cb_train::fold_len_multiplier_default(),
        simple_ctr: cb_train::simple_ctr_default(),
        simple_ctr_priors: cb_train::simple_ctr_priors_default(),
        counter_calc_method: cb_train::counter_calc_method_default(),
        boosting_type: cb_train::boosting_type_default(),
        max_ctr_complexity: cb_train::max_ctr_complexity_default(),
        combinations_ctr: cb_train::combinations_ctr_default(),
        combinations_ctr_priors: cb_train::combinations_ctr_priors_default(),
        // The generator pins score_function='L2' (the first-slice simplest math).
        score_function: cb_compute::EScoreFunction::L2,
        has_time: false,
        feature_weights: cb_train::feature_weights_default(),
        first_feature_use_penalties: cb_train::first_feature_use_penalties_default(),
        per_object_feature_penalties: cb_train::per_object_feature_penalties_default(),
        penalties_coefficient: cb_train::penalties_coefficient_default(),
        monotone_constraints,
        grow_policy: cb_train::grow_policy_default(),
        max_leaves: cb_train::max_leaves_default(),
        min_data_in_leaf: cb_train::min_data_in_leaf_default(),
    }
}

/// Train the monotone `scenario` and return the trained model, the float-feature
/// borders, the feature columns, and the recorded staged approximants.
fn train_scenario(
    scenario: &str,
    params: &BoostParams,
) -> (Model, Vec<Vec<f64>>, Vec<Vec<f32>>, Vec<f64>) {
    let columns = load_feature_columns();
    let target = load_target();
    let model_json = load_model_json(&fixture(&format!("monotone/{scenario}/model.json")))
        .unwrap_or_else(|e| panic!("monotone/{scenario}/model.json must load: {e:?}"));
    let borders = model_json.float_feature_borders();

    let mut staged = Vec::new();
    let model = train(
        &CpuBackend,
        &columns,
        &borders,
        &target,
        &[],
        params,
        Some(&mut staged),
    )
    .unwrap_or_else(|e| panic!("monotone/{scenario}: training failed: {e:?}"));

    (model, borders, columns, staged)
}

/// FEAT-03 SC-1 (monotone portion): oblivious monotone leaf values oracle-locked
/// <= 1e-5. Monotone is a leaf-value post-pass (the isotonic PAVA projection), so
/// for a FIXED structure the splits are unchanged — but because the projected
/// approx feeds back into later trees' gradients, the constrained model's chosen
/// splits can differ from an UNCONSTRAINED model. We therefore lock our trainer's
/// splits against THIS (monotone) fixture's own splits (self-consistent), then
/// gate LEAF VALUES / StagedApprox / Predictions against the catboost 1.2.10
/// fixture.
#[test]
fn monotone_oracle_increasing_decreasing() {
    let scenario = "increasing_decreasing";
    // feature 0 NON-INCREASING (-1), feature 2 NON-DECREASING (+1), others free.
    let params = isolating_params(vec![-1, 0, 1, 0]);
    let (model, borders, columns, staged) = train_scenario(scenario, &params);
    let model_json =
        load_model_json(&fixture(&format!("monotone/{scenario}/model.json"))).unwrap();

    // Stage::Splits — locked against the monotone fixture's own splits (the
    // structure our monotone-constrained trainer must reproduce).
    compare_stage(Stage::Splits, &model_json.split_borders(), &model.split_borders())
        .unwrap_or_else(|e| panic!("monotone/{scenario}: splits diverged: {e:?}"));

    // Stage::LeafValues — the isotonic (PAVA) projected, lr-scaled leaf values.
    compare_stage(Stage::LeafValues, &model_json.leaf_values(), &model.leaf_values())
        .unwrap_or_else(|e| panic!("monotone/{scenario}: leaf values diverged: {e:?}"));

    // Stage::StagedApprox — per-iteration raw approximants.
    let expected_staged =
        load_f64_vec(&fixture(&format!("monotone/{scenario}/staged.npy"))).unwrap();
    compare_stage(Stage::StagedApprox, &expected_staged, &staged)
        .unwrap_or_else(|e| panic!("monotone/{scenario}: staged approx diverged: {e:?}"));

    // Stage::Predictions — final raw approximants through the production apply path.
    let cb_model = CbModel::from_trained(&model, borders);
    let predictions = predict_raw(&cb_model, &columns);
    let expected_predictions =
        load_f64_vec(&fixture(&format!("monotone/{scenario}/predictions.npy"))).unwrap();
    compare_stage(Stage::Predictions, &expected_predictions, &predictions)
        .unwrap_or_else(|e| panic!("monotone/{scenario}: predictions diverged: {e:?}"));

    // The projected leaf values must be DEMONSTRABLY monotone along the constrained
    // feature 0 (-1 → NON-INCREASING): the model output must not increase as
    // feature 0 increases, holding the other features at their dataset medians (a
    // monotone-cone consequence of the PAVA projection, end-to-end).
    let med: Vec<f32> = (0..columns.len())
        .map(|f| {
            let mut col = columns[f].clone();
            col.sort_by(f32::total_cmp);
            col[col.len() / 2]
        })
        .collect();
    let lo = columns[0].iter().cloned().fold(f32::INFINITY, f32::min);
    let hi = columns[0].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut prev = f64::INFINITY;
    let steps = 20usize;
    for s in 0..=steps {
        let v0 = lo + (hi - lo) * (s as f32) / (steps as f32);
        let cols: Vec<Vec<f32>> = (0..columns.len())
            .map(|f| vec![if f == 0 { v0 } else { med[f] }])
            .collect();
        let out = predict_raw(&cb_model, &cols)[0];
        assert!(
            out <= prev + 1e-9,
            "feature-0 sweep not non-increasing at v0={v0}: {out} > {prev}"
        );
        prev = out;
    }
}

/// FEAT-03 escalated-gap guard (D-6.6-07): a MALFORMED `monotone_constraints`
/// entry (not in {-1, 0, +1}) is rejected with a typed [`cb_train::CbError`] —
/// no fabricated output. This is the SELF-CONTAINED guard reachable today.
///
/// ENABLED by Plan 06.6-04 (Task 3) — the two guard assertions DEFERRED by Plan
/// 06.6-02 (because `grow_policy` did not yet exist) are now reachable since
/// `grow_policy` lands in 06.6-04 Task 2. The monotone × non-symmetric and Region
/// typed-error rejections are asserted in
/// [`monotone_non_symmetric_and_region_are_typed_errors`]. No fabricated
/// non-symmetric-monotone fixture (no upstream ground truth — D-6.6-07); only the
/// typed-error rejection is asserted.
#[test]
fn monotone_invalid_constraint_is_typed_error() {
    let columns = load_feature_columns();
    let target = load_target();
    // A valid model.json border set (any) — borders are irrelevant; the guard
    // fires before any training work.
    let borders: Vec<Vec<f64>> = vec![vec![0.0]; columns.len()];
    // `2` is not a valid direction → typed CbError, never a trained model.
    let params = isolating_params(vec![1, 2, 0, 0]);
    let result = train(&CpuBackend, &columns, &borders, &target, &[], &params, None);
    assert!(
        result.is_err(),
        "an invalid monotone_constraints entry must be rejected with a typed error"
    );
}

/// FEAT-03 / FEAT-06 escalated-gap guards (D-6.6-07), ENABLED here once `grow_policy`
/// lands (06.6-04 Task 2). Two unsupported combinations must be rejected up front
/// with a typed [`cb_train::CbError`] — never a trained model, never fabricated
/// output:
///
///   1. `monotone_constraints` × a NON-SYMMETRIC `grow_policy` ({Lossguide,
///      Depthwise}). Upstream EXPLICITLY rejects monotone constraints under every
///      non-symmetric grow policy (`monotonic_constraint_utils.h:42`); the monotone
///      PAVA post-pass is oblivious-only (D-6.6-06), so routing a non-empty
///      `monotone_constraints` through the leaf-wise grower would silently DROP the
///      constraint. The guard rejects it instead.
///   2. `grow_policy == Region` — UNIMPLEMENTED on the CPU path (escalated gap,
///      D-6.6-04 "Region OUT"); there is no Region grower arm.
///
/// These were a commented `// TODO(06.6-04)` stub in this file under Plan 06.6-02
/// (the `grow_policy` enum did not exist yet); 06.6-04 OWNS enabling them. The
/// self-contained Region guard + the malformed-direction guard
/// ([`monotone_invalid_constraint_is_typed_error`]) stay intact.
#[test]
fn monotone_non_symmetric_and_region_are_typed_errors() {
    let columns = load_feature_columns();
    let target = load_target();
    let borders: Vec<Vec<f64>> = vec![vec![0.0]; columns.len()];

    // (1) monotone_constraints × non-symmetric grow_policy → typed error. Test BOTH
    //     non-symmetric policies (Lossguide AND Depthwise).
    for policy in [EGrowPolicy::Lossguide, EGrowPolicy::Depthwise] {
        let mut params = isolating_params(vec![-1, 0, 1, 0]);
        params.grow_policy = policy;
        let result = train(&CpuBackend, &columns, &borders, &target, &[], &params, None);
        assert!(
            result.is_err(),
            "monotone_constraints under grow_policy={policy:?} must be rejected with a \
             typed error (monotonic_constraint_utils.h:42, D-6.6-07)"
        );
    }

    // (2) grow_policy=Region → typed error (CPU-unimplemented escalated gap). Empty
    //     monotone_constraints so ONLY the Region rejection is exercised.
    let mut params = isolating_params(vec![]);
    params.grow_policy = EGrowPolicy::Region;
    let result = train(&CpuBackend, &columns, &borders, &target, &[], &params, None);
    assert!(
        result.is_err(),
        "grow_policy=Region must be rejected with a typed error (D-6.6-04 \"Region OUT\")"
    );
}
