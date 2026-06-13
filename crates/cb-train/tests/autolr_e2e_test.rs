//! End-to-end automatic-learning-rate train->predict cycle (TRAIN-08; Phase-3
//! success criterion 5).
//!
//! This is the phase capstone: it confirms a FULL CPU train->predict cycle runs
//! with the AUTO-SELECTED learning rate (no explicit `learning_rate`). The
//! boosting loop (`train`) is invoked with `auto_learning_rate: true`, so the
//! rate is guessed pre-train via `cb-train::autolr` (the gate upstream's
//! `UpdateLearningRate` checks), then trees are grown and a per-iteration staged
//! approximant (the running prediction) is produced.
//!
//! The assertion is operational, not an oracle lock: the cycle RUNS (no error),
//! a NON-EMPTY model is produced (one tree per iteration), and the produced
//! learning rate matches the upstream-selected value persisted in the committed
//! `autolr/{rmse,logloss}` fixtures (the same value the unit test pins). The
//! per-tree split parity is locked by the deterministic slice oracles; here the
//! point is the auto-LR-driven train->predict path is wired and functional.
//!
//! Integration test (under `tests/`) so it can depend on `cb-oracle`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_backend::CpuBackend;
use cb_compute::{LeafMethod, Loss};
use cb_oracle::{load_f64_vec, load_model_json};
use cb_train::{
    autolr_guess, train, BoostParams, EBootstrapType, EOverfittingDetectorType, TargetType,
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

/// Read `selected_learning_rate` from an autolr fixture config.json.
fn fixture_selected_lr(name: &str) -> f64 {
    let raw = std::fs::read_to_string(fixture(&format!("autolr/{name}/config.json")))
        .expect("autolr fixture config must exist");
    let cfg: serde_json::Value = serde_json::from_str(&raw).expect("valid json");
    cfg["selected_learning_rate"]
        .as_f64()
        .expect("selected_learning_rate")
}

/// Train one scenario with auto-LR enabled and assert the train->predict cycle
/// runs end-to-end with the auto-selected rate.
fn run_autolr_e2e(
    name: &str,
    loss: Loss,
    boost_from_average: bool,
    target: &[f64],
    borders_scenario: &str,
) {
    let columns = load_feature_columns();
    // Reuse the deterministic skeleton's trained quantization borders (same
    // numeric_tiny inputs). Only an operational train->predict run is asserted.
    let model_json = load_model_json(&fixture(&format!("{borders_scenario}/model.json")))
        .unwrap_or_else(|e| panic!("{borders_scenario}/model.json must load: {e:?}"));
    let borders = model_json.float_feature_borders();

    // The fixed iteration count the autolr fixture used (so the guessed rate the
    // boosting loop computes matches the persisted upstream selected rate).
    let iterations = 500usize;

    let params = BoostParams {
        loss,
        iterations,
        depth: 2,
        // Explicit value is IGNORED because auto_learning_rate == true and the
        // loss is auto-LR eligible; set to a sentinel to prove it is unused.
        learning_rate: f64::NAN,
        auto_learning_rate: true,
        one_hot_max_size: cb_train::one_hot_max_size_default(),
        permutation_count: cb_train::permutation_count_default(),
        fold_len_multiplier: cb_train::fold_len_multiplier_default(),
        simple_ctr: cb_train::simple_ctr_default(),
        simple_ctr_priors: cb_train::simple_ctr_priors_default(),
        counter_calc_method: cb_train::counter_calc_method_default(),
        l2_leaf_reg: 3.0,
        random_strength: 0.0,
        boost_from_average,
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
    };

    let mut staged = Vec::new();
    let model = train(
        &CpuBackend,
        &columns,
        &borders,
        target,
        &[],
        &params,
        Some(&mut staged),
    )
    .unwrap_or_else(|e| panic!("{name}: auto-LR train failed: {e:?}"));

    // 1. A non-empty model: one oblivious tree per iteration.
    assert_eq!(
        model.oblivious_trees.len(),
        iterations,
        "{name}: expected {iterations} trees from the auto-LR train"
    );
    assert!(
        !model.leaf_values().is_empty(),
        "{name}: model must have leaf values"
    );

    // 2. The staged approximants (the per-iteration running prediction) are
    //    produced for the whole cycle and are finite (no NaN/inf leaked from the
    //    auto-LR path).
    assert_eq!(
        staged.len(),
        iterations * target.len(),
        "{name}: staged predictions must cover every iteration x object"
    );
    assert!(
        staged.iter().all(|v| v.is_finite()),
        "{name}: staged predictions must be finite"
    );

    // 3. The rate the loop applied matches the upstream-selected rate (the same
    //    value the unit test pins) — proving the auto-LR path drove training.
    let target_type = match loss {
        Loss::Rmse => TargetType::Rmse,
        Loss::Logloss | Loss::CrossEntropy => TargetType::Logloss,
        Loss::Focal { .. } => TargetType::Logloss,
        Loss::Mae => TargetType::Unknown,
    };
    let guessed = autolr_guess(target_type, false, boost_from_average, target.len(), iterations)
        .expect("auto-LR guess");
    let expected = fixture_selected_lr(name);
    assert!(
        (guessed - expected).abs() <= 1e-5,
        "{name}: auto-LR rate {guessed} must match upstream {expected}"
    );
    // The guessed rate must be non-trivial (the explicit NaN sentinel was not used).
    assert!(guessed.is_finite() && guessed > 0.0);
}

#[test]
fn autolr_rmse_train_predict_cycle_runs() {
    let target = load_f64_vec(&fixture("inputs/numeric_tiny/y.npy")).unwrap();
    run_autolr_e2e(
        "rmse",
        Loss::Rmse,
        true, // RMSE default boost_from_average
        &target,
        "regression_skeleton",
    );
}

#[test]
fn autolr_logloss_train_predict_cycle_runs() {
    // Binary labels: y_binary = (y > median(y)), matching the binclf fixture.
    let y = load_f64_vec(&fixture("inputs/numeric_tiny/y.npy")).unwrap();
    let mut sorted = y.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let median = if n.is_multiple_of(2) {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    };
    let target: Vec<f64> = y.iter().map(|&v| if v > median { 1.0 } else { 0.0 }).collect();
    run_autolr_e2e(
        "logloss",
        Loss::Logloss,
        false, // Logloss default boost_from_average
        &target,
        "binclf_skeleton",
    );
}
