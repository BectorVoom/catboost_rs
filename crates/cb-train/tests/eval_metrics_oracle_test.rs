//! Per-iteration eval-set metric oracle (TRAIN-07 / D-10).
//!
//! Locks the PER-ITERATION `eval_metric` values, PER EVAL SET, against the
//! committed upstream catboost 1.2.10 `eval_metrics/{rmse,logloss}` fixtures
//! (each trained with TWO eval sets and an explicit `eval_metric`). For each
//! scenario the Rust loop ([`train_with_eval_sets`]) trains over the shared train
//! set with both eval sets attached, collects the per-set per-iteration metric
//! curve into an [`EvalMetricHistory`], and asserts each set's curve matches
//! upstream's `get_evals_result()[validation_k][eval_metric]` at <= 1e-5
//! (`compare_stage(Stage::Predictions, …)`).
//!
//! This supersedes the Plan 05 inline eval-set loss STUB: the metric now flows
//! through `cb-train::metrics` (`eval_metric` defaulting to the objective,
//! multi-eval-set logging), and the curves are oracle-locked rather than only
//! feeding the stop decision.
//!
//! DETERMINISTIC CONFIG (D-07): `bootstrap_type=No`, `random_strength=0`, fixed
//! seed, 12 iterations — short enough that the per-iteration eval metric stays
//! within 1e-5 of upstream (the eval-prediction boundary-routing residual the
//! overfit oracle documents only perturbs the longer ~32+-iteration curves).
//!
//! Integration test (under `tests/`) so it can depend on `cb-oracle`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_backend::CpuBackend;
use cb_compute::{LeafMethod, Loss};
use cb_oracle::{compare_stage, load_f64_vec, load_model_json, Stage};
use cb_train::{
    train_with_eval_sets, BoostParams, EBootstrapType, EOverfittingDetectorType, EvalMetric,
    EvalMetricHistory, EvalSet,
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

/// Load a 2-D `.npy` matrix as per-feature `f32` SoA columns.
fn load_columns(rel: &str) -> Vec<Vec<f32>> {
    let x: Array2<f64> = read_npy(fixture(rel)).unwrap_or_else(|e| panic!("{rel} must load: {e:?}"));
    (0..x.ncols())
        .map(|fi| x.column(fi).iter().map(|&v| v as f32).collect())
        .collect()
}

/// Train one eval_metrics scenario with BOTH eval sets attached and return the
/// produced per-eval-set per-iteration metric history.
fn train_eval_metrics(
    name: &str,
    loss: Loss,
    eval_metric: EvalMetric,
    boost_from_average: bool,
) -> EvalMetricHistory {
    let x_train = load_columns("inputs/eval_metrics/X_train.npy");
    let x_eval0 = load_columns("inputs/eval_metrics/X_eval0.npy");
    let x_eval1 = load_columns("inputs/eval_metrics/X_eval1.npy");
    let suffix = name; // "rmse" / "logloss" — the per-loss target file suffix.
    let y_train =
        load_f64_vec(&fixture(&format!("inputs/eval_metrics/y_train_{suffix}.npy"))).unwrap();
    let y_eval0 =
        load_f64_vec(&fixture(&format!("inputs/eval_metrics/y_eval0_{suffix}.npy"))).unwrap();
    let y_eval1 =
        load_f64_vec(&fixture(&format!("inputs/eval_metrics/y_eval1_{suffix}.npy"))).unwrap();

    // Borders from this scenario's model.json (the trained quantization).
    let model_json = load_model_json(&fixture(&format!("eval_metrics/{name}/model.json"))).unwrap();
    let borders = model_json.float_feature_borders();

    let params = BoostParams {
        loss,
        iterations: 12,
        depth: 3,
        learning_rate: 0.3,
        l2_leaf_reg: 3.0,
        random_strength: 0.0,
        boost_from_average,
        leaf_method: LeafMethod::Gradient,
        bootstrap_type: EBootstrapType::No,
        subsample: 1.0,
        bagging_temperature: 0.0,
        random_seed: 0,
        // No early stopping — the full budget runs so every iteration is locked.
        od_type: EOverfittingDetectorType::None,
        od_pval: 0.0,
        od_wait: 0,
        use_best_model: false,
        eval_metric: Some(eval_metric),
        auto_learning_rate: false,
        one_hot_max_size: cb_train::one_hot_max_size_default(),
        permutation_count: cb_train::permutation_count_default(),
        fold_len_multiplier: cb_train::fold_len_multiplier_default(),
    };

    let sets = [
        EvalSet {
            feature_values: &x_eval0,
            target: &y_eval0,
        },
        EvalSet {
            feature_values: &x_eval1,
            target: &y_eval1,
        },
    ];
    let mut history = EvalMetricHistory::new(2);
    train_with_eval_sets(
        &CpuBackend,
        &x_train,
        &borders,
        &y_train,
        &[],
        &params,
        None,
        &sets,
        Some(&mut history),
    )
    .expect("eval_metrics training succeeds");
    history
}

/// Assert the per-iteration metric curve for one eval set matches the committed
/// upstream history at <= 1e-5.
fn assert_eval_curve(name: &str, history: &EvalMetricHistory, set_idx: usize) {
    let file = format!("eval_metrics/{name}/eval{set_idx}_metric.npy");
    let expected = load_f64_vec(&fixture(&file)).unwrap();
    let actual = &history.per_set[set_idx];
    compare_stage(Stage::Predictions, &expected, actual).unwrap_or_else(|e| {
        panic!("{name} eval set {set_idx} per-iteration metric diverged: {e:?}")
    });
}

#[test]
fn eval_metrics_oracle_rmse_both_sets() {
    let history = train_eval_metrics("rmse", Loss::Rmse, EvalMetric::Rmse, true);
    assert_eq!(history.len(), 2, "rmse: two eval sets tracked");
    assert_eval_curve("rmse", &history, 0);
    assert_eval_curve("rmse", &history, 1);
}

#[test]
fn eval_metrics_oracle_logloss_both_sets() {
    let history = train_eval_metrics("logloss", Loss::Logloss, EvalMetric::Logloss, false);
    assert_eq!(history.len(), 2, "logloss: two eval sets tracked");
    assert_eval_curve("logloss", &history, 0);
    assert_eval_curve("logloss", &history, 1);
}

/// `eval_metric` left unset (`None`) defaults to the objective and produces the
/// SAME curve as the explicit metric (the default-to-objective rule, TRAIN-07).
#[test]
fn eval_metric_defaults_to_objective_curve() {
    let explicit = train_eval_metrics("rmse", Loss::Rmse, EvalMetric::Rmse, true);
    // Re-run with eval_metric = None (default) and compare the primary curve.
    let x_train = load_columns("inputs/eval_metrics/X_train.npy");
    let x_eval0 = load_columns("inputs/eval_metrics/X_eval0.npy");
    let x_eval1 = load_columns("inputs/eval_metrics/X_eval1.npy");
    let y_train = load_f64_vec(&fixture("inputs/eval_metrics/y_train_rmse.npy")).unwrap();
    let y_eval0 = load_f64_vec(&fixture("inputs/eval_metrics/y_eval0_rmse.npy")).unwrap();
    let y_eval1 = load_f64_vec(&fixture("inputs/eval_metrics/y_eval1_rmse.npy")).unwrap();
    let borders = load_model_json(&fixture("eval_metrics/rmse/model.json"))
        .unwrap()
        .float_feature_borders();
    let params = BoostParams {
        loss: Loss::Rmse,
        iterations: 12,
        depth: 3,
        learning_rate: 0.3,
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
        eval_metric: None, // default => RMSE (the objective)
        auto_learning_rate: false,
        one_hot_max_size: cb_train::one_hot_max_size_default(),
        permutation_count: cb_train::permutation_count_default(),
        fold_len_multiplier: cb_train::fold_len_multiplier_default(),
    };
    let sets = [
        EvalSet {
            feature_values: &x_eval0,
            target: &y_eval0,
        },
        EvalSet {
            feature_values: &x_eval1,
            target: &y_eval1,
        },
    ];
    let mut defaulted = EvalMetricHistory::new(2);
    train_with_eval_sets(
        &CpuBackend,
        &x_train,
        &borders,
        &y_train,
        &[],
        &params,
        None,
        &sets,
        Some(&mut defaulted),
    )
    .unwrap();
    assert_eq!(
        defaulted.per_set, explicit.per_set,
        "eval_metric=None must default to the objective (RMSE) curve"
    );
}
