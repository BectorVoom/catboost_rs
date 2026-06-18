//! FEAT-04 feature-penalty train→predict oracle (Phase 06.6 plan 01): train a
//! plain boosted oblivious-tree (SymmetricTree) RMSE model on the frozen
//! `numeric_tiny` inputs with each of the three penalty kinds
//! (`feature_weights`, `first_feature_use_penalties`,
//! `per_object_feature_penalties`) set, and gate per-tree splits, per-tree leaf
//! values, per-iteration staged approximants, and final raw predictions against
//! the committed upstream catboost 1.2.10 fixtures at <= 1e-5 (D-08, D-6.6-08).
//!
//! The fixtures are generated OFFLINE from the `.venv` catboost 1.2.10
//! (`crates/cb-oracle/generator/gen_penalty_fixtures.py`, pinned `random_seed=0`,
//! `thread_count=1`); CI only READS the committed `.npy` / `model.json`. Each
//! penalty vector targets float feature index 1 (the unpenalized split favourite),
//! so the fixtures genuinely exercise the penalty path rather than trivially
//! matching the default oblivious model.
//!
//! Integration test (under `tests/`) so it can depend on `cb-oracle` / `cb-model`;
//! the top-line `#![allow(...)]` mirrors `slice_first_oracle_test.rs:9`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_backend::CpuBackend;
use cb_compute::{LeafMethod, Loss};
use cb_model::{predict_raw, Model as CbModel};
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

/// Load the raw `numeric_tiny` regression target.
fn load_target() -> Vec<f64> {
    load_f64_vec(&fixture("inputs/numeric_tiny/y.npy")).unwrap()
}

/// Build the first-slice simplified isolating [`BoostParams`] (mirrors the
/// generator's `ISOLATING_PARAMS`), overriding only the penalty fields supplied
/// by the caller via `with_penalties`.
fn isolating_params(with_penalties: impl FnOnce(&mut BoostParams)) -> BoostParams {
    let mut params = BoostParams {
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
        monotone_constraints: cb_train::monotone_constraints_default(),
    };
    with_penalties(&mut params);
    params
}

/// Train the penalty `scenario` and return the trained model, the float-feature
/// borders, the feature columns, and the recorded staged approximants.
fn train_scenario(
    scenario: &str,
    params: &BoostParams,
) -> (Model, Vec<Vec<f64>>, Vec<Vec<f32>>, Vec<f64>) {
    let columns = load_feature_columns();
    let target = load_target();
    let model_json = load_model_json(&fixture(&format!("penalty/{scenario}/model.json")))
        .unwrap_or_else(|e| panic!("penalty/{scenario}/model.json must load: {e:?}"));
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
    .unwrap_or_else(|e| panic!("penalty/{scenario}: training failed: {e:?}"));

    (model, borders, columns, staged)
}

/// Gate Splits | LeafValues | StagedApprox | Predictions for one penalty
/// scenario against the committed catboost 1.2.10 fixture at <= 1e-5.
fn check_scenario(scenario: &str, params: &BoostParams) {
    let (model, borders, columns, staged) = train_scenario(scenario, params);
    let model_json =
        load_model_json(&fixture(&format!("penalty/{scenario}/model.json"))).unwrap();

    // Stage::Splits — per-tree split borders (float feature + border).
    compare_stage(Stage::Splits, &model_json.split_borders(), &model.split_borders())
        .unwrap_or_else(|e| panic!("penalty/{scenario}: splits diverged: {e:?}"));

    // Stage::LeafValues — per-tree leaf values (already lr-scaled).
    compare_stage(Stage::LeafValues, &model_json.leaf_values(), &model.leaf_values())
        .unwrap_or_else(|e| panic!("penalty/{scenario}: leaf values diverged: {e:?}"));

    // Stage::StagedApprox — per-iteration raw approximants.
    let expected_staged =
        load_f64_vec(&fixture(&format!("penalty/{scenario}/staged.npy"))).unwrap();
    compare_stage(Stage::StagedApprox, &expected_staged, &staged)
        .unwrap_or_else(|e| panic!("penalty/{scenario}: staged approx diverged: {e:?}"));

    // Stage::Predictions — final raw approximants through the production apply path.
    let cb_model = CbModel::from_trained(&model, borders);
    let predictions = predict_raw(&cb_model, &columns);
    let expected_predictions =
        load_f64_vec(&fixture(&format!("penalty/{scenario}/predictions.npy"))).unwrap();
    compare_stage(Stage::Predictions, &expected_predictions, &predictions)
        .unwrap_or_else(|e| panic!("penalty/{scenario}: predictions diverged: {e:?}"));
}

#[test]
fn penalty_oracle_feature_weights() {
    // MULTIPLICATIVE candidate-gain weight (GetSplitFeatureWeight): heavily
    // down-weight float feature 1 (the unpenalized favourite) so the greedy search
    // avoids it.
    let params = isolating_params(|p| {
        p.feature_weights = vec![1.0, 0.1, 1.0, 1.0];
    });
    check_scenario("feature_weights", &params);
}

#[test]
fn penalty_oracle_first_feature_use() {
    // SUBTRACTIVE first-use penalty (PenalizeBestSplits): penalize the FIRST use of
    // float feature 1 (x penalties_coefficient) while it is unused in the model.
    let params = isolating_params(|p| {
        p.first_feature_use_penalties = vec![0.0, 5.0, 0.0, 0.0];
        p.penalties_coefficient = 1.0;
    });
    check_scenario("first_use", &params);
}

#[test]
fn penalty_oracle_per_object() {
    // SUBTRACTIVE per-object penalty (PenalizeBestSplits): penalize float feature 1
    // by penalty x coefficient x whole-fold doc count while it is globally unused.
    let params = isolating_params(|p| {
        p.per_object_feature_penalties = vec![0.0, 0.1, 0.0, 0.0];
        p.penalties_coefficient = 1.0;
    });
    check_scenario("per_object", &params);
}
