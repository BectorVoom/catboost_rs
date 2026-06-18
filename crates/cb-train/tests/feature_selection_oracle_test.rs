//! FEAT-05 / SC-2 — recursive feature selection partition oracle.
//!
//! Drives `cb_train::select_features` (the recursive-elimination loop, 06.6-08
//! Task 1) end-to-end and asserts that the `{selected_features,
//! eliminated_features}` partition EXACTLY matches the committed catboost 1.2.10
//! `select_features(...)` output — for BOTH the `RecursiveByShapValues` backend
//! and a FeatureEffect backend (`RecursiveByPredictionValuesChange`). This is a
//! DISCRETE oracle (set/order equality of feature indices), not a float
//! tolerance.
//!
//! The importance backends are the Gate-C `cb-model` methods (`shap_values`,
//! `prediction_values_change`), injected as `cb_train::select_features`'s
//! `ImportanceRanker` callbacks. `cb-model` is a DEV-dependency of `cb-train`
//! (cycle-exempt — the normal build graph is `cb-model -> cb-train`), so this
//! test can name the real importances while the orchestration loop in `cb-train`
//! source stays cb-model-free (06.6-08 module docs / D-6.6-03).
//!
//! Integration test (under `tests/`) so it can depend on `cb-oracle` + `cb-model`
//! (both dev-deps); the top-line `#![allow(...)]` mirrors the other cb-train
//! oracle tests.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_backend::CpuBackend;
use cb_compute::{LeafMethod, Loss};
use cb_model::{prediction_values_change, shap_values, Model as CbModel};
use cb_oracle::load_f64_vec;
use cb_train::{
    select_features, BoostParams, EBootstrapType, EBoostingType, EFeaturesSelectionAlgorithm,
    EOverfittingDetectorType,
};
use ndarray::Array2;
use ndarray_npy::read_npy;
use serde_json::Value;

const SEED: u64 = 0;

/// Resolve a path under `cb-oracle/fixtures/` from cb-train's manifest dir.
fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join(rel)
}

/// Load the dataset as per-feature `f32` SoA columns.
fn load_columns() -> Vec<Vec<f32>> {
    let x: Array2<f32> = read_npy(fixture("feature_selection/X.npy"))
        .unwrap_or_else(|e| panic!("feature_selection/X.npy must load as f32 [N,F]: {e:?}"));
    (0..x.ncols()).map(|fi| x.column(fi).to_vec()).collect()
}

fn load_target() -> Vec<f64> {
    load_f64_vec(&fixture("feature_selection/y.npy"))
        .unwrap_or_else(|e| panic!("feature_selection/y.npy must load: {e:?}"))
}

/// Parsed fixture config (params + the captured partitions + borders).
struct FixtureCfg {
    borders: Vec<Vec<f64>>,
    features_for_select: Vec<usize>,
    shap_selected: Vec<usize>,
    shap_eliminated: Vec<usize>,
    shap_steps: usize,
    shap_num_to_select: usize,
    pvc_selected: Vec<usize>,
    pvc_eliminated: Vec<usize>,
    pvc_steps: usize,
    pvc_num_to_select: usize,
}

fn usize_vec(v: &Value) -> Vec<usize> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|e| e.as_u64().expect("u64") as usize)
        .collect()
}

fn load_cfg() -> FixtureCfg {
    let text = std::fs::read_to_string(fixture("feature_selection/config.json"))
        .unwrap_or_else(|e| panic!("config.json must load: {e:?}"));
    let c: Value = serde_json::from_str(&text).expect("config.json must parse");
    let borders = c["feature_borders"]
        .as_array()
        .expect("feature_borders array")
        .iter()
        .map(|row| {
            row.as_array()
                .expect("border row")
                .iter()
                .map(|b| b.as_f64().expect("f64 border"))
                .collect::<Vec<f64>>()
        })
        .collect();
    FixtureCfg {
        borders,
        features_for_select: usize_vec(&c["features_for_select"]),
        shap_selected: usize_vec(&c["shap_values"]["selected_features"]),
        shap_eliminated: usize_vec(&c["shap_values"]["eliminated_features"]),
        shap_steps: c["shap_values"]["steps"].as_u64().unwrap() as usize,
        shap_num_to_select: c["shap_values"]["num_features_to_select"].as_u64().unwrap() as usize,
        pvc_selected: usize_vec(&c["prediction_values_change"]["selected_features"]),
        pvc_eliminated: usize_vec(&c["prediction_values_change"]["eliminated_features"]),
        pvc_steps: c["prediction_values_change"]["steps"].as_u64().unwrap() as usize,
        pvc_num_to_select: c["prediction_values_change"]["num_features_to_select"]
            .as_u64()
            .unwrap() as usize,
    }
}

/// The isolating RMSE / Plain / no-bootstrap / random_strength=0 config the
/// fixture pins (matches `feature_selection/config.json` `params`).
fn params() -> BoostParams {
    BoostParams {
        loss: Loss::Rmse,
        iterations: 20,
        depth: 4,
        learning_rate: 0.1,
        l2_leaf_reg: 3.0,
        random_strength: 0.0,
        boost_from_average: true,
        leaf_method: LeafMethod::Gradient,
        bootstrap_type: EBootstrapType::No,
        subsample: 1.0,
        bagging_temperature: 0.0,
        random_seed: SEED,
        od_type: EOverfittingDetectorType::None,
        od_pval: 0.0,
        od_wait: 0,
        use_best_model: false,
        eval_metric: None,
        auto_learning_rate: false,
        one_hot_max_size: cb_train::one_hot_max_size_default(),
        permutation_count: 1,
        fold_len_multiplier: 2.0,
        simple_ctr: cb_train::simple_ctr_default(),
        simple_ctr_priors: cb_train::simple_ctr_priors_default(),
        counter_calc_method: cb_train::counter_calc_method_default(),
        boosting_type: EBoostingType::Plain,
        max_ctr_complexity: cb_train::max_ctr_complexity_default(),
        combinations_ctr: cb_train::combinations_ctr_default(),
        combinations_ctr_priors: cb_train::combinations_ctr_priors_default(),
        score_function: cb_compute::EScoreFunction::L2,
        has_time: false,
        feature_weights: cb_train::feature_weights_default(),
        first_feature_use_penalties: cb_train::first_feature_use_penalties_default(),
        per_object_feature_penalties: cb_train::per_object_feature_penalties_default(),
        penalties_coefficient: cb_train::penalties_coefficient_default(),
        monotone_constraints: cb_train::monotone_constraints_default(),
        grow_policy: cb_train::grow_policy_default(),
        max_leaves: cb_train::max_leaves_default(),
        min_data_in_leaf: cb_train::min_data_in_leaf_default(),
    }
}

/// RMSE finalError over the raw approx (regression objective; lower = better).
fn rmse_final_error(approx: &[f64], labels: &[f64]) -> f64 {
    if approx.is_empty() {
        return 0.0;
    }
    let mut acc = 0.0_f64;
    for (a, t) in approx.iter().zip(labels.iter()) {
        let d = a - t;
        acc += d * d;
    }
    (acc / approx.len() as f64).sqrt()
}

/// FEAT-05: `RecursiveByShapValues` partition matches catboost 1.2.10.
#[test]
fn select_features_shap_values_partition_matches_upstream() {
    let columns = load_columns();
    let target = load_target();
    let cfg = load_cfg();
    let borders = cfg.borders.clone();

    // SHAP backend ranker (upstream EliminateFeaturesBasedOnShapValues): rank a
    // feature by the loss INCREASE caused by removing its SHAP contribution from
    // the approx — the feature whose removal least worsens the loss (the smallest
    // increase) is eliminated first. `select_features` eliminates the LOWEST
    // score first, so the score IS the loss-change-on-removal directly.
    let labels = target.clone();
    let bd = borders.clone();
    let ranker = move |trained: &cb_train::Model, sub: &[Vec<f32>], n_local: usize| -> Vec<f64> {
        let sub_borders: Vec<Vec<f64>> = (0..n_local).map(|i| bd[i].clone()).collect();
        let model = CbModel::from_trained(trained, sub_borders);
        let approx = cb_model::predict_raw(&model, sub);
        let shap = shap_values(&model, sub, n_local);
        let base = rmse_final_error(&approx, &labels);
        (0..n_local)
            .map(|f| {
                let approx_f: Vec<f64> = (0..approx.len())
                    .map(|o| approx[o] - shap.get(o).and_then(|r| r.get(f)).copied().unwrap_or(0.0))
                    .collect();
                rmse_final_error(&approx_f, &labels) - base
            })
            .collect()
    };

    let result = select_features(
        &CpuBackend,
        &columns,
        &borders,
        &target,
        &[],
        &params(),
        &cfg.features_for_select,
        cfg.shap_num_to_select,
        EFeaturesSelectionAlgorithm::RecursiveByShapValues,
        cfg.shap_steps,
        /* train_final_model = */ false,
        &ranker,
    )
    .unwrap_or_else(|e| panic!("select_features (Shap) failed: {e:?}"));

    assert_eq!(
        result.selected_features, cfg.shap_selected,
        "SHAP selected_features must match catboost 1.2.10 EXACTLY"
    );
    assert_eq!(
        result.eliminated_features, cfg.shap_eliminated,
        "SHAP eliminated_features (in elimination order) must match catboost 1.2.10 EXACTLY"
    );
}

/// FEAT-05: `RecursiveByPredictionValuesChange` (FeatureEffect backend) partition
/// matches catboost 1.2.10.
#[test]
fn select_features_prediction_values_change_partition_matches_upstream() {
    let columns = load_columns();
    let target = load_target();
    let cfg = load_cfg();
    let borders = cfg.borders.clone();

    // FeatureEffect backend ranker (upstream EliminateFeaturesBasedOnFeatureEffect):
    // rank by the per-feature PredictionValuesChange effect; the smallest effect
    // is eliminated first (upstream StableSortBy ascending then drops the front).
    let bd = borders.clone();
    let ranker = move |trained: &cb_train::Model, _sub: &[Vec<f32>], n_local: usize| -> Vec<f64> {
        let sub_borders: Vec<Vec<f64>> = (0..n_local).map(|i| bd[i].clone()).collect();
        let model = CbModel::from_trained(trained, sub_borders);
        let mut eff = prediction_values_change(&model);
        // The PVC vector spans the model's used features; pad to n_local so an
        // unused candidate column ranks lowest (effect 0) and is eliminated first.
        eff.resize(n_local, 0.0);
        eff
    };

    let result = select_features(
        &CpuBackend,
        &columns,
        &borders,
        &target,
        &[],
        &params(),
        &cfg.features_for_select,
        cfg.pvc_num_to_select,
        EFeaturesSelectionAlgorithm::RecursiveByPredictionValuesChange,
        cfg.pvc_steps,
        /* train_final_model = */ false,
        &ranker,
    )
    .unwrap_or_else(|e| panic!("select_features (PVC) failed: {e:?}"));

    assert_eq!(
        result.selected_features, cfg.pvc_selected,
        "PVC selected_features must match catboost 1.2.10 EXACTLY"
    );
    assert_eq!(
        result.eliminated_features, cfg.pvc_eliminated,
        "PVC eliminated_features (in elimination order) must match catboost 1.2.10 EXACTLY"
    );
}
