//! First-slice train→predict oracle (TRAIN-01/02/03 Gradient): train a plain
//! boosted oblivious-tree model on the frozen `regression_skeleton` (RMSE) and
//! `binclf_skeleton` (Logloss) inputs and gate per-tree splits, per-tree leaf
//! values, and per-iteration staged approximants against the committed upstream
//! catboost 1.2.10 fixtures at <= 1e-5 for BOTH losses (D-08).
//!
//! Integration test (under `tests/`) so it can depend on `cb-oracle`; the
//! top-line `#![allow(...)]` mirrors `borders_oracle_test.rs:14`.
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

/// Load the raw `numeric_tiny` target (regression `y`).
fn load_regression_target() -> Vec<f64> {
    load_f64_vec(&fixture("inputs/numeric_tiny/y.npy")).unwrap()
}

/// Derive the binary Logloss label `y_binary = (y > median(y))`, matching the
/// binclf fixture's `label_definition`.
fn load_binclf_target() -> Vec<f64> {
    let y = load_regression_target();
    let mut sorted = y.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // numpy median for an even count is the mean of the two middle values.
    let n = sorted.len();
    let median = if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    };
    y.iter().map(|&v| if v > median { 1.0 } else { 0.0 }).collect()
}

/// Train a model on the given scenario and return it plus the recorded staged
/// approximants (flat, `iterations * n_rows`).
fn train_scenario(scenario: &str, loss: Loss, boost_from_average: bool) -> (Model, Vec<f64>) {
    let columns = load_feature_columns();
    let model_json = load_model_json(&fixture(&format!("{scenario}/model.json")))
        .unwrap_or_else(|e| panic!("{scenario}/model.json must load: {e:?}"));
    let borders = model_json.float_feature_borders();

    let target = match loss {
        // Regression family (RMSE / MAE / the Wave-1 smooth losses) trains on the
        // raw regression target.
        Loss::Rmse
        | Loss::Mae
        | Loss::LogCosh
        | Loss::Lq { .. }
        | Loss::Huber { .. }
        | Loss::Expectile { .. }
        | Loss::Poisson
        | Loss::Tweedie { .. }
        | Loss::Mape => load_regression_target(),
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
        leaf_method: LeafMethod::Gradient,
        // First-slice isolating params: sampling disabled (D-07).
        bootstrap_type: EBootstrapType::No,
        subsample: 1.0,
        bagging_temperature: 0.0,
        random_seed: 0,
        // No overfitting detection / early stopping in this slice (TRAIN-06 off).
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
        score_function: cb_compute::EScoreFunction::L2,
        has_time: false,
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

/// Gate splits, leaf values, and staged approximants for one scenario.
fn check_scenario(scenario: &str, loss: Loss, boost_from_average: bool) {
    let (model, staged) = train_scenario(scenario, loss, boost_from_average);
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
fn slice_first_oracle_regression_skeleton_rmse() {
    // RMSE with boost_from_average=true (bias == target mean).
    check_scenario("regression_skeleton", Loss::Rmse, true);
}

#[test]
fn slice_first_oracle_binclf_skeleton_logloss() {
    // Logloss with boost_from_average=false (bias == 0; raw-logit staged).
    check_scenario("binclf_skeleton", Loss::Logloss, false);
}
