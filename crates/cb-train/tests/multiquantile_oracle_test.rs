//! Wave-3 MultiQuantile training oracle (LOSS-03 multi-output / Plan 06.2-05):
//! train a plain-boosted multi-output regression model under `MultiQuantile`
//! (`Loss::MultiQuantile { alpha, delta }`) — K INDEPENDENT Quantile dimensions
//! (D-6.2-05) — and gate per-tree splits, per-tree leaf values, the per-iteration
//! staged N-dim approx, AND the RAW per-quantile predictions against the committed
//! upstream catboost 1.2.10 `multiquantile` fixture at <= 1e-5.
//!
//! MultiQuantile is fully SEPARABLE: each dimension `d` reuses the scalar quantile
//! der with its OWN level `alpha[d]` (the shared `delta`) and the 6.1 Exact
//! weighted-`alpha[d]`-quantile leaf path (`exact_leaf_delta`, reused VERBATIM per
//! dimension — leaf.rs UNCHANGED). `approx_dimension = alpha.len()`. The leaf method
//! is `Exact` (the single-host CPU default, Pitfall 3); `der2 = 0` per dimension.
//! Predictions are RAW per-quantile (identity — no link transform).
//!
//! Pins (the generator mirrors these): `loss_function=MultiQuantile:alpha=0.1,0.5,
//! 0.9`, `leaf_estimation_method=Exact`, `leaf_estimation_iterations=1`,
//! `score_function=L2`, depth 2, 5 iterations, learning_rate 0.1, l2_leaf_reg 3.0,
//! `bootstrap_type=No`, `random_strength=0`, `boost_from_average=false`,
//! `thread_count=1`. Trained on the RAW `numeric_tiny` target (MultiQuantile admits
//! the full real line, like the 6.1 quantile fixtures).
//!
//! Layout notes (RESEARCH A4 / Pitfall 6):
//!   - the trainer's leaf values are DIMENSION-MAJOR (leaf_values[d*n_leaves+l]);
//!     the fixture model.json is LEAF-MAJOR (leaf_values[l*dim+d]) — transposed
//!     per tree before comparing.
//!   - the trainer's staged buffer is DIMENSION-MAJOR per iteration
//!     (approx[d*n+i]); the fixture staged.npy is OBJECT-MAJOR (n,dim) per
//!     iteration — transposed before comparing.
//!   - the MultiQuantile target stays PER-OBJECT (length n): every dimension
//!     predicts a quantile of the SAME scalar target (unlike the dim-major
//!     multilabel target). `approx_dimension = alpha.len()`.
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

/// The per-quantile alpha vector — mirrors the generator's MULTIQUANTILE_ALPHAS.
const ALPHAS: [f64; 3] = [0.1, 0.5, 0.9];
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

/// The RAW `numeric_tiny` regression target (MultiQuantile admits the full real
/// line — no positive shift). Per-object length `n` (every quantile dimension
/// predicts a quantile of this SAME scalar target).
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
        // MultiQuantile Exact leaf (weighted alpha[d]-quantile per dimension).
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
        // MultiQuantile fixture pins the regression-skeleton L2 split score.
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

#[test]
fn multiquantile_per_stage_oracle() {
    let loss = Loss::MultiQuantile {
        alpha: ALPHAS.to_vec(),
        delta: QUANTILE_DELTA,
    };
    let scenario = "multiquantile";
    let (model, staged) = train_scenario(loss, scenario);
    let model_json = load_model_json(&fixture(&format!("{scenario}/model.json"))).unwrap();
    let dim = model.approx_dimension;
    assert_eq!(dim, ALPHAS.len(), "{scenario}: expected {} quantile dimensions", ALPHAS.len());
    let n = load_regression_target().len();

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

    // Stage 4: Predictions (RAW per-quantile, identity — RawFormulaVal transposed
    // dim-major -> object-major). MultiQuantile has no link transform; `kind` is
    // ignored by the RawFormulaVal arm.
    let raw = final_raw_approx(&staged, dim, n);
    let predictions = apply_multiclass_prediction(
        PredictionType::RawFormulaVal,
        MultiClassKind::OneVsAll,
        &raw,
        dim,
        &model.class_to_label,
    );
    let expected_pred = load_f64_vec(&fixture(&format!("{scenario}/predictions.npy"))).unwrap();
    compare_stage(Stage::Predictions, &expected_pred, &predictions)
        .unwrap_or_else(|e| panic!("{scenario}: predictions diverged: {e:?}"));
}
