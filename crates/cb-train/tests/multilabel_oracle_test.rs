//! Wave-2 multilabel training oracle (LOSS-02 / Plan 06.2-04): train plain-boosted
//! multilabel models under `MultiLogloss` and `MultiCrossEntropy` — the SAME
//! upstream `TMultiCrossEntropyError` per-dimension DIAGONAL der path
//! (`error_functions.h:781-820`; dispatch `tensor_search_helpers.cpp:236-238`),
//! two enum names — and gate per-tree splits, per-tree leaf values, the
//! per-iteration staged N-dim approx, AND the per-dimension sigmoid probability
//! predictions against the committed upstream catboost 1.2.10 fixtures at <= 1e-5.
//!
//! Two scenarios:
//!   - multilogloss      : `Loss::MultiLogloss`. Binary {0,1} label columns.
//!   - multicrossentropy : `Loss::MultiCrossEntropy`. The SAME der path (trained on
//!     the same binary label matrix here, which exercises the identical
//!     `multi_crossentropy_ders`; MultiCrossEntropy additionally admits soft [0,1]
//!     targets, validated separately).
//!
//! Both are SEPARABLE (each label dimension independent), reusing the scalar
//! sigmoid + scalar Newton leaf step per dimension (no softmax coupling). Both pin
//! `score_function=Cosine`, `leaf_method=Newton`, `leaf_estimation_iterations=1`
//! (Pitfall 2 — the upstream multilabel default is Newton with 10 iters, pinned to
//! 1), depth 2, 5 iterations, learning_rate 0.1, l2_leaf_reg 3.0, bootstrap_type=No,
//! random_strength=0, thread_count=1. The 3-label binary target is derived from
//! numeric_tiny (y>median, X0>median, X1>median) — the same rule the generator pins.
//!
//! Layout notes (RESEARCH A4 / Pitfall 6):
//!   - the trainer's leaf values are DIMENSION-MAJOR (leaf_values[d*n_leaves+l]);
//!     the fixture model.json is LEAF-MAJOR (leaf_values[l*dim+d]) — transposed
//!     per tree before comparing.
//!   - the trainer's staged buffer is DIMENSION-MAJOR per iteration
//!     (approx[d*n+i]); the fixture staged.npy is OBJECT-MAJOR (n,dim) per
//!     iteration — transposed before comparing.
//!   - the multilabel target is DIM-MAJOR (target[d*n+i]); the trainer derives the
//!     object count from the feature columns and the label width from
//!     target.len()/n.
//!
//! Integration test (under `tests/`) so it can depend on `cb-oracle`. NO `#[ignore]`,
//! NO weakened tolerance.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_backend::CpuBackend;
use cb_compute::{LeafMethod, Loss};
use cb_model::{apply_multiclass_prediction, MultiClassKind, PredictionType};
use cb_oracle::{compare_stage, load_f64_vec, load_model_json, Stage};
use cb_train::{train, BoostParams, EBootstrapType, EOverfittingDetectorType, Model};
use ndarray::Array2;
use ndarray_npy::read_npy;

const LABEL_COUNT: usize = 3;

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

/// The 3-label binary multilabel target, DIM-MAJOR `target[d*n + i]`. Reproduces the
/// generator's `_multilabel_target` rule: three NESTED binary thresholds on `y` at
/// its 0.25 / 0.5 / 0.75 quantiles — label k = (y > quantile(y,[.25,.5,.75])[k]).
/// Nested on a single signal so the multi-dim split score has a CLEAR winner per
/// level (no symmetric cross-feature tie). Returned dim-major (label 0 column, then
/// label 1, then label 2).
fn load_multilabel_target() -> Vec<f64> {
    let y = load_f64_vec(&fixture("inputs/numeric_tiny/y.npy")).unwrap();
    let q0 = quantile_linear(&y, 0.25);
    let q1 = quantile_linear(&y, 0.50);
    let q2 = quantile_linear(&y, 0.75);
    let n = y.len();
    let mut out = Vec::with_capacity(LABEL_COUNT * n);
    // Dim-major: label 0 column (>q25), then label 1 (>q50), then label 2 (>q75).
    out.extend(y.iter().map(|&v| if v > q0 { 1.0 } else { 0.0 }));
    out.extend(y.iter().map(|&v| if v > q1 { 1.0 } else { 0.0 }));
    out.extend(y.iter().map(|&v| if v > q2 { 1.0 } else { 0.0 }));
    out
}

/// numpy's default `quantile` (linear interpolation): sort, position `q*(n-1)`,
/// interpolate between the floor and ceil ranks.
fn quantile_linear(data: &[f64], q: f64) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted[0];
    }
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    sorted[lo] + (sorted[hi] - sorted[lo]) * frac
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
        leaf_method: LeafMethod::Newton,
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
        // MultiLabel default split-score function is Cosine (NOT the scalar L2).
        score_function: cb_compute::EScoreFunction::Cosine,
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

fn train_scenario(loss: Loss, scenario: &str) -> (Model, Vec<f64>) {
    let columns = load_feature_columns();
    let target = load_multilabel_target();
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

/// Transpose the trainer's DIMENSION-MAJOR per-tree leaf values
/// (`leaf_values[d*n_leaves + l]`) into the LEAF-MAJOR layout
/// (`leaf_values[l*dim + d]`) the fixture model.json stores (Pitfall 6), flattened
/// in tree order.
fn leaf_values_leaf_major(model: &Model) -> Vec<f64> {
    let dim = model.approx_dimension;
    let mut out = Vec::new();
    for tree in &model.oblivious_trees {
        let total = tree.leaf_values.len();
        let n_leaves = if dim == 0 { total } else { total / dim };
        for l in 0..n_leaves {
            for d in 0..dim {
                out.push(tree.leaf_values[d * n_leaves + l]);
            }
        }
    }
    out
}

/// Transpose the trainer's DIMENSION-MAJOR staged buffer (per iteration
/// `approx[d*n + i]`) into the OBJECT-MAJOR `(iterations, n, dim)` layout the
/// fixture staged.npy stores (A4).
fn staged_object_major(staged: &[f64], dim: usize, n: usize) -> Vec<f64> {
    if dim == 0 || n == 0 {
        return staged.to_vec();
    }
    let per_iter = dim * n;
    let iterations = staged.len() / per_iter;
    let mut out = Vec::with_capacity(staged.len());
    for it in 0..iterations {
        let base = it * per_iter;
        for i in 0..n {
            for d in 0..dim {
                out.push(staged[base + d * n + i]);
            }
        }
    }
    out
}

/// Final-iteration raw approx (dimension-major `approx[d*n+i]`) extracted from the
/// staged buffer — the input to the prediction transform.
fn final_raw_approx(staged: &[f64], dim: usize, n: usize) -> Vec<f64> {
    let per_iter = dim * n;
    let iterations = staged.len() / per_iter;
    let base = (iterations - 1) * per_iter;
    staged[base..base + per_iter].to_vec()
}

fn check_multilabel(loss: Loss, scenario: &str) {
    let (model, staged) = train_scenario(loss, scenario);
    let model_json = load_model_json(&fixture(&format!("{scenario}/model.json"))).unwrap();
    let dim = model.approx_dimension;
    assert_eq!(dim, LABEL_COUNT, "{scenario}: expected {LABEL_COUNT} label dimensions");
    let n = load_multilabel_target().len() / LABEL_COUNT;

    // Stage 1: Splits (the multi-dim single-shared-accumulator split score).
    compare_stage(Stage::Splits, &model_json.split_borders(), &model.split_borders())
        .unwrap_or_else(|e| panic!("{scenario}: splits diverged: {e:?}"));

    // Stage 2: LeafValues (transpose dim-major -> leaf-major to match model.json).
    compare_stage(
        Stage::LeafValues,
        &model_json.leaf_values(),
        &leaf_values_leaf_major(&model),
    )
    .unwrap_or_else(|e| panic!("{scenario}: leaf values diverged: {e:?}"));

    // Stage 3: StagedApprox (transpose dim-major -> object-major to match staged.npy).
    let expected_staged = load_f64_vec(&fixture(&format!("{scenario}/staged.npy"))).unwrap();
    compare_stage(
        Stage::StagedApprox,
        &expected_staged,
        &staged_object_major(&staged, dim, n),
    )
    .unwrap_or_else(|e| panic!("{scenario}: staged approx diverged: {e:?}"));

    // Stage 4: Predictions (per-dimension sigmoid Probability, object-major).
    let raw = final_raw_approx(&staged, dim, n);
    let predictions = apply_multiclass_prediction(
        PredictionType::Probability,
        MultiClassKind::MultiLabel,
        &raw,
        dim,
        &model.class_to_label,
    );
    let expected_pred = load_f64_vec(&fixture(&format!("{scenario}/predictions.npy"))).unwrap();
    compare_stage(Stage::Predictions, &expected_pred, &predictions)
        .unwrap_or_else(|e| panic!("{scenario}: predictions diverged: {e:?}"));
}

#[test]
fn multilogloss_per_stage_oracle() {
    check_multilabel(Loss::MultiLogloss, "multilogloss");
}

#[test]
fn multicrossentropy_per_stage_oracle() {
    check_multilabel(Loss::MultiCrossEntropy, "multicrossentropy");
}
