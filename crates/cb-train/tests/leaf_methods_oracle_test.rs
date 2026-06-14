//! Leaf-methods train→predict oracle (TRAIN-03 / D-09): train a plain boosted
//! oblivious-tree model with each of the four leaf-estimation methods (Gradient,
//! Newton, Exact, Simple) and gate per-tree splits, per-tree leaf values, and
//! per-iteration staged approximants against the committed upstream catboost
//! 1.2.10 `leaf_methods/{gradient,newton,exact,simple}` fixtures at <= 1e-5.
//!
//! Each method gets its own scenario so a divergence is attributable to a single
//! method's leaf math (the D-07 simplified-isolating discipline carried into the
//! per-method oracle):
//!   - gradient: RMSE, boost_from_average=true (== the first-slice path).
//!   - newton:   Logloss, boost_from_average=false (der2 = -p(1-p) makes Newton
//!               distinct from Gradient; RMSE der2==-1 would collapse them).
//!   - exact:    MAE,  boost_from_average=false (Exact is rejected upstream for
//!               RMSE/Logloss; its leaf delta is the weighted median of leaf
//!               residuals).
//!   - simple:   RMSE, boost_from_average=true (== Gradient leaf delta, A6).
//!
//! Integration test (under `tests/`) so it can depend on `cb-oracle`; the
//! top-line `#![allow(...)]` mirrors `slice_first_oracle_test.rs:9`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_backend::CpuBackend;
use cb_compute::{LeafMethod, Loss};
use cb_oracle::{compare_stage, load_f64_vec, load_model_json, Stage};
use cb_train::{train, BoostParams, EBootstrapType, EOverfittingDetectorType, Model};
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

/// Load the raw `numeric_tiny` target (continuous `y`, used by RMSE and MAE).
fn load_regression_target() -> Vec<f64> {
    load_f64_vec(&fixture("inputs/numeric_tiny/y.npy")).unwrap()
}

/// Derive the binary Logloss label `y_binary = (y > median(y))`, matching the
/// generator's `y_bin` definition for the newton scenario.
fn load_binclf_target() -> Vec<f64> {
    let y = load_regression_target();
    let mut sorted = y.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let median = if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    };
    y.iter().map(|&v| if v > median { 1.0 } else { 0.0 }).collect()
}

/// Train one leaf-method scenario and return the model plus staged approximants.
fn train_scenario(
    scenario: &str,
    loss: Loss,
    leaf_method: LeafMethod,
    boost_from_average: bool,
) -> (Model, Vec<f64>) {
    let columns = load_feature_columns();
    let model_json = load_model_json(&fixture(&format!("{scenario}/model.json")))
        .unwrap_or_else(|e| panic!("{scenario}/model.json must load: {e:?}"));
    let borders = model_json.float_feature_borders();

    let target = match loss {
        Loss::Rmse | Loss::Mae => load_regression_target(),
        Loss::Logloss | Loss::CrossEntropy | Loss::Focal { .. } => load_binclf_target(),
    };

    let params = BoostParams {
        loss,
        iterations: 5,
        depth: 2,
        learning_rate: 0.1,
        l2_leaf_reg: 3.0,
        random_strength: 0.0,
        boost_from_average,
        leaf_method,
        // Leaf-method scenarios pin sampling off (D-07 isolating discipline).
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
    };

    let mut staged = Vec::new();
    let model = train(
        &CpuBackend,
        &columns,
        &borders,
        &target,
        &[],
        &params,
        Some(&mut staged),
    )
    .unwrap_or_else(|e| panic!("{scenario}: training failed: {e:?}"));

    (model, staged)
}

/// Gate splits, leaf values, and staged approximants for one method scenario.
fn check_scenario(
    scenario: &str,
    loss: Loss,
    leaf_method: LeafMethod,
    boost_from_average: bool,
) {
    let (model, staged) = train_scenario(scenario, loss, leaf_method, boost_from_average);
    let model_json = load_model_json(&fixture(&format!("{scenario}/model.json"))).unwrap();

    // Stage::Splits — per-tree split borders (float feature + border).
    let expected_splits = model_json.split_borders();
    let actual_splits = model.split_borders();
    compare_stage(Stage::Splits, &expected_splits, &actual_splits)
        .unwrap_or_else(|e| panic!("{scenario}: splits diverged: {e:?}"));

    // Stage::LeafValues — per-tree leaf values (already lr-scaled).
    let expected_leaves = model_json.leaf_values();
    let actual_leaves = model.leaf_values();
    compare_stage(Stage::LeafValues, &expected_leaves, &actual_leaves)
        .unwrap_or_else(|e| panic!("{scenario}: leaf values diverged: {e:?}"));

    // Stage::StagedApprox — per-iteration raw approximants / logits.
    let expected_staged = load_f64_vec(&fixture(&format!("{scenario}/staged.npy"))).unwrap();
    compare_stage(Stage::StagedApprox, &expected_staged, &staged)
        .unwrap_or_else(|e| panic!("{scenario}: staged approx diverged: {e:?}"));
}

#[test]
fn leaf_methods_oracle_gradient() {
    check_scenario("leaf_methods/gradient", Loss::Rmse, LeafMethod::Gradient, true);
}

#[test]
fn leaf_methods_oracle_newton() {
    // Logloss makes the Newton hessian (-p(1-p)) genuinely distinct from Gradient.
    check_scenario("leaf_methods/newton", Loss::Logloss, LeafMethod::Newton, false);
}

#[test]
fn leaf_methods_oracle_exact() {
    // MAE: Exact leaf delta == weighted median of leaf residuals.
    check_scenario("leaf_methods/exact", Loss::Mae, LeafMethod::Exact, false);
}

#[test]
fn leaf_methods_oracle_simple() {
    // Simple == Gradient leaf delta (A6); RMSE scenario mirrors gradient.
    check_scenario("leaf_methods/simple", Loss::Rmse, LeafMethod::Simple, true);
}
