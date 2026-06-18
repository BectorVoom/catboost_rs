//! Wave-3 quantile-family training oracle (LOSS-03 / Plan 06.1-03): train a plain
//! boosted regression model under `Quantile{alpha, delta}` with the Exact
//! weighted-alpha-quantile leaf, and gate per-tree splits, per-tree leaf values,
//! and per-iteration staged approximants against the committed upstream catboost
//! 1.2.10 `quantile_alpha07` / `quantile_alpha05_mae` fixtures at <= 1e-5.
//!
//! Two scenarios isolate the two acceptance claims:
//!   - quantile_alpha07       : `Quantile{alpha:0.7, delta:1e-6}`, Exact leaf.
//!     Exercises the alpha!=0.5 path — the Exact leaf must use the weighted
//!     0.7-quantile, NOT the hardcoded 0.5 median. A regression to the hardcoded
//!     `QUANTILE_ALPHA` (0.5) diverges here.
//!   - quantile_alpha05_mae   : `Quantile{alpha:0.5, delta:1e-6}`, Exact leaf.
//!     The MAE-equivalence ANCHOR. Additionally, the trained model's leaf values
//!     and staged approx are asserted to match the existing `leaf_methods/exact`
//!     (MAE) fixture <= 1e-5 — proving MAE == Quantile{alpha=0.5} (the re-
//!     expression keeps the existing MAE Exact oracle byte-stable).
//!
//! All share the D-07 isolating discipline (depth 2, 5 iterations, learning_rate
//! 0.1, l2_leaf_reg 3.0, no sampling / no random strength, boost_from_average=
//! false, score_function=L2, leaf_estimation_iterations=1) and train on the RAW
//! `numeric_tiny` target (Quantile admits the full real line — no positive shift,
//! unlike Wave 2). Quantile predictions are RAW (no link transform), so the
//! StagedApprox stage gates the raw approx; there is no Predictions stage.
//!
//! Integration test (under `tests/`) so it can depend on `cb-oracle`.
//!
//! RED until Tasks 2-3 land `Loss::Quantile{alpha,delta}` + the Exact-alpha
//! threading wiring — the `Loss::Quantile` variant does not exist yet, so this
//! file will not compile against the current `cb_compute::Loss`. That RED state is
//! the Task-1 acceptance signal.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_backend::CpuBackend;
use cb_compute::{LeafMethod, Loss};
use cb_oracle::{compare_stage, load_f64_vec, load_model_json, Stage};
use cb_train::{train, BoostParams, EBootstrapType, EOverfittingDetectorType, Model};
use ndarray::Array2;
use ndarray_npy::read_npy;

/// The MAE / Quantile deadzone half-width (`error_functions.h:468-469` default).
const QUANTILE_DELTA: f64 = 1e-6;

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join(rel)
}

fn load_feature_columns() -> Vec<Vec<f32>> {
    let x: Array2<f64> = read_npy(fixture("inputs/numeric_tiny/X.npy"))
        .unwrap_or_else(|e| panic!("numeric_tiny/X.npy must load: {e:?}"));
    (0..x.ncols())
        .map(|fi| x.column(fi).iter().map(|&v| v as f32).collect())
        .collect()
}

/// The RAW `numeric_tiny` regression target (Quantile admits the full real line —
/// no positive shift, unlike the Wave-2 positive-domain losses).
fn load_regression_target() -> Vec<f64> {
    load_f64_vec(&fixture("inputs/numeric_tiny/y.npy")).unwrap()
}

fn base_params(loss: Loss) -> BoostParams {
    BoostParams {
        loss,
        iterations: 5,
        depth: 2,
        learning_rate: 0.1,
        l2_leaf_reg: 3.0,
        random_strength: 0.0,
        boost_from_average: false,
        // The Quantile Exact leaf (weighted alpha-quantile of leaf residuals).
        leaf_method: LeafMethod::Exact,
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
        score_function: cb_compute::EScoreFunction::L2,
        has_time: false,
        feature_weights: cb_train::feature_weights_default(),
        first_feature_use_penalties: cb_train::first_feature_use_penalties_default(),
        per_object_feature_penalties: cb_train::per_object_feature_penalties_default(),
        penalties_coefficient: cb_train::penalties_coefficient_default(),
    }
}

fn train_scenario(loss: Loss, scenario: &str) -> (Model, Vec<f64>) {
    let columns = load_feature_columns();
    let target = load_regression_target();
    let model_json = load_model_json(&fixture(&format!("{scenario}/model.json")))
        .unwrap_or_else(|e| panic!("{scenario}/model.json must load: {e:?}"));
    let borders = model_json.float_feature_borders();

    let mut staged = Vec::new();
    let model = train(
        &CpuBackend,
        &columns,
        &borders,
        &target,
        &[],
        &base_params(loss),
        Some(&mut staged),
    )
    .unwrap_or_else(|e| panic!("{scenario}: training failed: {e:?}"));
    (model, staged)
}

/// Gate Splits / LeafValues / StagedApprox for a scenario against its fixture.
fn check_train_stages(loss: Loss, scenario: &str) -> (Model, Vec<f64>) {
    let (model, staged) = train_scenario(loss, scenario);
    let model_json = load_model_json(&fixture(&format!("{scenario}/model.json"))).unwrap();

    compare_stage(Stage::Splits, &model_json.split_borders(), &model.split_borders())
        .unwrap_or_else(|e| panic!("{scenario}: splits diverged: {e:?}"));
    compare_stage(Stage::LeafValues, &model_json.leaf_values(), &model.leaf_values())
        .unwrap_or_else(|e| panic!("{scenario}: leaf values diverged: {e:?}"));

    let expected_staged = load_f64_vec(&fixture(&format!("{scenario}/staged.npy"))).unwrap();
    compare_stage(Stage::StagedApprox, &expected_staged, &staged)
        .unwrap_or_else(|e| panic!("{scenario}: staged approx diverged: {e:?}"));
    (model, staged)
}

#[test]
fn wave3_oracle_quantile_alpha07() {
    // alpha=0.7: the Exact leaf MUST use the weighted 0.7-quantile (NOT the
    // hardcoded 0.5 median). A regression to the hardcoded QUANTILE_ALPHA diverges
    // at LeafValues / StagedApprox.
    check_train_stages(
        Loss::Quantile {
            alpha: 0.7,
            delta: QUANTILE_DELTA,
        },
        "quantile_alpha07",
    );
}

#[test]
fn wave3_oracle_quantile_alpha05_equals_mae() {
    // alpha=0.5: the MAE-equivalence anchor. First gate against the dedicated
    // quantile_alpha05_mae fixture.
    let (model, staged) = check_train_stages(
        Loss::Quantile {
            alpha: 0.5,
            delta: QUANTILE_DELTA,
        },
        "quantile_alpha05_mae",
    );

    // Then prove MAE == Quantile{alpha=0.5}: the Quantile{0.5} model must match the
    // existing leaf_methods/exact (MAE) fixture <= 1e-5 (leaf values + staged
    // approx). This is the byte-stability guarantee — re-expressing MAE through
    // Quantile must not move the existing MAE Exact oracle.
    let mae_model = load_model_json(&fixture("leaf_methods/exact/model.json")).unwrap();
    compare_stage(Stage::LeafValues, &mae_model.leaf_values(), &model.leaf_values())
        .unwrap_or_else(|e| panic!("Quantile{{0.5}} leaf values must equal MAE: {e:?}"));

    let mae_staged = load_f64_vec(&fixture("leaf_methods/exact/staged.npy")).unwrap();
    compare_stage(Stage::StagedApprox, &mae_staged, &staged)
        .unwrap_or_else(|e| panic!("Quantile{{0.5}} staged approx must equal MAE: {e:?}"));
}
