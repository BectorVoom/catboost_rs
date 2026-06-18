//! CrossEntropy + Focal training oracle (LOSS-01 / D-09): train a plain boosted
//! binclf model under each new loss and gate per-tree splits, per-tree leaf
//! values, and per-iteration staged approximants against the committed upstream
//! catboost 1.2.10 `loss_extra/{cross_entropy,focal}` fixtures at <= 1e-5.
//!
//! Target definitions match the fixture configs:
//!   - cross_entropy: soft label `y_soft = sigmoid(standardize(y))` (a
//!     probability in `[0,1]` — CrossEntropy's distinguishing input).
//!   - focal:         binary label `y_binary = (y > median(y))`, with
//!     `alpha = 0.25`, `gamma = 2.0`.
//!
//! Both use `boost_from_average = false` (bias 0), depth 2, 5 iterations,
//! learning_rate 0.1, l2_leaf_reg 3.0, no sampling / no random strength — the
//! D-07 isolating discipline so a divergence is attributable to the loss math.
//!
//! A standalone unit test pins the der1/der2 numeric values at known
//! `(approx, target)` points as the in-env gate independent of the fixtures.
//!
//! Integration test (under `tests/`) so it can depend on `cb-oracle`; the
//! top-line `#![allow(...)]` mirrors the other cb-train oracle tests.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_backend::CpuBackend;
use cb_compute::{
    cross_entropy_der1, cross_entropy_der2, focal_der1, focal_der2, sigmoid, LeafMethod, Loss,
};
use cb_oracle::{compare_stage, load_f64_vec, load_model_json, Stage};
use cb_train::{train, BoostParams, EBootstrapType, EOverfittingDetectorType, Model};
use ndarray::Array2;
use ndarray_npy::read_npy;

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

fn load_y() -> Vec<f64> {
    load_f64_vec(&fixture("inputs/numeric_tiny/y.npy")).unwrap()
}

/// Sample standard deviation reference is not needed: standardize uses the
/// population mean/std (matching numpy `(y - mean)/std`, ddof=0).
fn standardize(y: &[f64]) -> Vec<f64> {
    let n = y.len() as f64;
    let mean = y.iter().sum::<f64>() / n;
    let var = y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();
    y.iter().map(|v| (v - mean) / std).collect()
}

/// CrossEntropy soft label: `y_soft = sigmoid(standardize(y))`.
fn cross_entropy_target() -> Vec<f64> {
    standardize(&load_y()).iter().map(|&z| sigmoid(z)).collect()
}

/// Focal binary label: `y_binary = (y > median(y))`.
fn focal_target() -> Vec<f64> {
    let y = load_y();
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

fn base_params(loss: Loss) -> BoostParams {
    BoostParams {
        loss,
        iterations: 5,
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
        monotone_constraints: cb_train::monotone_constraints_default(),
        grow_policy: cb_train::grow_policy_default(),
        max_leaves: cb_train::max_leaves_default(),
        min_data_in_leaf: cb_train::min_data_in_leaf_default(),
    }
}

fn train_scenario(loss: Loss, target: &[f64]) -> (Model, Vec<f64>) {
    let columns = load_feature_columns();
    let scenario = match loss {
        Loss::CrossEntropy => "loss_extra/cross_entropy",
        Loss::Focal { .. } => "loss_extra/focal",
        _ => panic!("unexpected loss in loss_oracle_test"),
    };
    let model_json = load_model_json(&fixture(&format!("{scenario}/model.json")))
        .unwrap_or_else(|e| panic!("{scenario}/model.json must load: {e:?}"));
    let borders = model_json.float_feature_borders();

    let mut staged = Vec::new();
    let model = train(
        &CpuBackend,
        &columns,
        &borders,
        target,
        &[],
        &base_params(loss),
        Some(&mut staged),
    )
    .unwrap_or_else(|e| panic!("{scenario}: training failed: {e:?}"));
    (model, staged)
}

fn check_scenario(loss: Loss, scenario: &str, target: &[f64]) {
    let (model, staged) = train_scenario(loss, target);
    let model_json = load_model_json(&fixture(&format!("{scenario}/model.json"))).unwrap();

    compare_stage(Stage::Splits, &model_json.split_borders(), &model.split_borders())
        .unwrap_or_else(|e| panic!("{scenario}: splits diverged: {e:?}"));
    compare_stage(Stage::LeafValues, &model_json.leaf_values(), &model.leaf_values())
        .unwrap_or_else(|e| panic!("{scenario}: leaf values diverged: {e:?}"));

    let expected_staged = load_f64_vec(&fixture(&format!("{scenario}/staged.npy"))).unwrap();
    compare_stage(Stage::StagedApprox, &expected_staged, &staged)
        .unwrap_or_else(|e| panic!("{scenario}: staged approx diverged: {e:?}"));
}

#[test]
fn loss_oracle_cross_entropy() {
    check_scenario(Loss::CrossEntropy, "loss_extra/cross_entropy", &cross_entropy_target());
}

#[test]
fn loss_oracle_focal() {
    check_scenario(
        Loss::Focal { alpha: 0.25, gamma: 2.0 },
        "loss_extra/focal",
        &focal_target(),
    );
}

/// In-env der1/der2 numeric gate (independent of the fixtures): CrossEntropy is
/// identical to Logloss (`der1 = target - sigmoid(approx)`, `der2 = -p(1-p)`).
#[test]
fn cross_entropy_der_values() {
    // approx = 0 -> p = 0.5; target = 0.7 (a SOFT label, CrossEntropy-specific).
    let p = sigmoid(0.0);
    assert!((cross_entropy_der1(0.0, 0.7) - (0.7 - p)).abs() < 1e-12);
    assert!((cross_entropy_der2(0.0, 0.7) - (-p * (1.0 - p))).abs() < 1e-12);
    // approx = 1.0 -> p = sigmoid(1.0).
    let p1 = sigmoid(1.0);
    assert!((cross_entropy_der1(1.0, 0.3) - (0.3 - p1)).abs() < 1e-12);
    assert!((cross_entropy_der2(1.0, 0.3) - (-p1 * (1.0 - p1))).abs() < 1e-12);
}

/// In-env Focal der1/der2 numeric gate at a known point, transcribing the
/// reference formula directly (`error_functions.h:1684-1709`) so the closed-form
/// helper is pinned independently of the training fixture.
#[test]
fn focal_der_values() {
    let (alpha, gamma) = (0.25_f64, 2.0_f64);
    let approx = 0.5_f64;
    let target = 1.0_f64;

    // Reference transcription.
    let p = (1.0 / (1.0 + (-approx).exp())).clamp(1e-13, 1.0 - 1e-13);
    let at = alpha; // target == 1
    let pt = p;
    let y = 2.0 * target - 1.0;
    let want_der1 = -(at * y * (1.0 - pt).powf(gamma) * (gamma * pt * pt.ln() + pt - 1.0));
    let u = at * y * (1.0 - pt).powf(gamma);
    let du = -at * y * gamma * (1.0 - pt).powf(gamma - 1.0);
    let v = gamma * pt * pt.ln() + pt - 1.0;
    let dv = gamma * pt.ln() + gamma + 1.0;
    let want_der2 = -((du * v + u * dv) * y * (pt * (1.0 - pt)));

    assert!((focal_der1(approx, target, alpha, gamma) - want_der1).abs() < 1e-12);
    assert!((focal_der2(approx, target, alpha, gamma) - want_der2).abs() < 1e-12);

    // Negative-class branch (target == 0): at = 1-alpha, pt = 1-p, y = -1.
    let target0 = 0.0_f64;
    let at0 = 1.0 - alpha;
    let pt0 = 1.0 - p;
    let y0 = -1.0;
    let want0 = -(at0 * y0 * (1.0 - pt0).powf(gamma) * (gamma * pt0 * pt0.ln() + pt0 - 1.0));
    assert!((focal_der1(approx, target0, alpha, gamma) - want0).abs() < 1e-12);
}
