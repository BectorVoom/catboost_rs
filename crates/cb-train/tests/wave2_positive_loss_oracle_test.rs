//! Wave-2 positive-domain / link regression-loss training oracle (LOSS-03 / Plan
//! 06.1-02): train a plain boosted regression model under each positive-domain /
//! link loss — Poisson (exp-link), Tweedie{variance_power}, MAPE (der2=0) — and
//! gate per-tree splits, per-tree leaf values, per-iteration staged approximants,
//! and (Poisson/Tweedie) final Predictions against the committed upstream catboost
//! 1.2.10 `{poisson,tweedie,mape}` fixtures at <= 1e-5.
//!
//! All three train on a POSITIVE-target variant of the frozen `numeric_tiny`
//! corpus (the additive shift `y_pos = y - min(y) + 1.0` recorded in each
//! fixture's config.json `target_shift`); Poisson/Tweedie/MAPE require a strictly
//! positive target.
//!
//! Per-loss leaf method is PINNED per the fixture (RESEARCH Pitfall 2/3/5):
//!   - Poisson : Newton, leaf_estimation_iterations:1 (override upstream 10)
//!   - Tweedie : Newton (variance_power=1.5)
//!   - MAPE    : Gradient (der2=0 so Newton is undefined — Pitfall 5)
//!
//! The Poisson exp-link is validated by BOTH stages (Open Q1 / Pitfall 4):
//!   - StagedApprox compares the RAW staged approx (RawFormulaVal).
//!   - Predictions compares `exp(raw)` via the production
//!     `cb_model::apply_prediction_type(PredictionType::Exponent, ..)` over the
//!     final-iteration staged approx (NOT a hand-rolled exp).
//! Tweedie Predictions are RAW (no Exponent — A4): the exp lives inside the der.
//!
//! All share the D-07 isolating discipline (depth 2, 5 iterations, learning_rate
//! 0.1, l2_leaf_reg 3.0, no sampling / no random strength, boost_from_average=
//! false, score_function=L2) so a divergence is attributable to the loss math
//! alone.
//!
//! Integration test (under `tests/`) so it can depend on `cb-oracle` / `cb-model`.
//!
//! RED until Tasks 2-4 land the three losses + their dispatch wiring — the
//! `Loss::{Poisson,Tweedie,Mape}` variants do not exist yet, so this file will not
//! compile against the current `cb_compute::Loss`. That RED state is the Task-1
//! acceptance signal.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_backend::CpuBackend;
use cb_compute::{LeafMethod, Loss};
use cb_model::{apply_prediction_type, PredictionType};
use cb_oracle::{compare_stage, load_f64_vec, load_model_json, Stage};
use cb_train::{train, BoostParams, EBootstrapType, EOverfittingDetectorType, Model};
use ndarray::Array2;
use ndarray_npy::read_npy;
use serde_json::Value;

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

/// The positive target column `y_pos = y - min(y) + 1.0`, reproducing the exact
/// additive `target_shift` the fixture recorded in its config.json.
fn load_positive_y(scenario: &str) -> Vec<f64> {
    let y = load_f64_vec(&fixture("inputs/numeric_tiny/y.npy")).unwrap();
    let cfg: Value = serde_json::from_str(
        &std::fs::read_to_string(fixture(&format!("{scenario}/config.json"))).unwrap(),
    )
    .unwrap();
    let shift = cfg["target_shift"].as_f64().unwrap();
    y.into_iter().map(|v| v + shift).collect()
}

fn base_params(loss: Loss, leaf_method: LeafMethod) -> BoostParams {
    BoostParams {
        loss,
        iterations: 5,
        depth: 2,
        learning_rate: 0.1,
        l2_leaf_reg: 3.0,
        random_strength: 0.0,
        boost_from_average: false,
        leaf_method,
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

fn train_scenario(loss: Loss, leaf_method: LeafMethod, scenario: &str) -> (Model, Vec<f64>) {
    let columns = load_feature_columns();
    let target = load_positive_y(scenario);
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
        &base_params(loss, leaf_method),
        Some(&mut staged),
    )
    .unwrap_or_else(|e| panic!("{scenario}: training failed: {e:?}"));
    (model, staged)
}

/// Gate Splits / LeafValues / StagedApprox for a scenario. Returns the staged
/// approx (flat, n_iterations * n_rows) so the Predictions stage can slice the
/// final iteration.
fn check_train_stages(loss: Loss, leaf_method: LeafMethod, scenario: &str) -> (Model, Vec<f64>) {
    let (model, staged) = train_scenario(loss, leaf_method, scenario);
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

/// The final-iteration raw staged approx (last `n_rows` of the flat staged vec).
fn final_raw_approx(staged: &[f64], n_rows: usize) -> Vec<f64> {
    staged[staged.len() - n_rows..].to_vec()
}

#[test]
fn wave2_oracle_poisson() {
    // Poisson: Newton, leaf_estimation_iterations:1 PINNED (override upstream 10).
    let (_model, staged) = check_train_stages(Loss::Poisson, LeafMethod::Newton, "poisson");

    // Predictions stage (exp-link, Open Q1 / Pitfall 4): the Poisson final
    // prediction is `exp(raw)`, applied via the production Exponent transform over
    // the final-iteration RAW staged approx (NOT a hand-rolled exp).
    let n_rows = load_feature_columns()[0].len();
    let raw = final_raw_approx(&staged, n_rows);
    let predictions = apply_prediction_type(PredictionType::Exponent, &raw);

    let expected_predictions = load_f64_vec(&fixture("poisson/predictions.npy")).unwrap();
    compare_stage(Stage::Predictions, &expected_predictions, &predictions)
        .unwrap_or_else(|e| panic!("poisson: exp-link predictions diverged: {e:?}"));
}

#[test]
fn wave2_oracle_tweedie() {
    // Tweedie{variance_power=1.5}: Newton; exp lives INSIDE the der (raw approx).
    let (_model, staged) =
        check_train_stages(Loss::Tweedie { variance_power: 1.5 }, LeafMethod::Newton, "tweedie");

    // Predictions are RAW (no Exponent — A4): the final raw approx IS the
    // prediction. Compare against the fixture's RawFormulaVal predictions.
    let n_rows = load_feature_columns()[0].len();
    let predictions = final_raw_approx(&staged, n_rows);

    let expected_predictions = load_f64_vec(&fixture("tweedie/predictions.npy")).unwrap();
    compare_stage(Stage::Predictions, &expected_predictions, &predictions)
        .unwrap_or_else(|e| panic!("tweedie: raw predictions diverged: {e:?}"));
}

#[test]
fn wave2_oracle_mape() {
    // MAPE: der2=0 so Newton is undefined (Pitfall 5) -> Gradient leaf. No
    // Predictions stage (raw approx is the prediction; gated via StagedApprox).
    check_train_stages(Loss::Mape, LeafMethod::Gradient, "mape");
}
