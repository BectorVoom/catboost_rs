//! LOSS-07 self-oracle (training-dispatch half): a built-in loss reimplemented AS
//! a Rust [`cb_compute::CustomObjective`] and trained through the `Loss::Custom`
//! `compute_gradients` dispatch must reproduce that built-in's EXISTING per-stage
//! oracle <= 1e-5 — proving the trait dispatch is FAITHFUL with NO new upstream
//! fixture.
//!
//! The chosen built-in is **CrossEntropy** (`loss_extra/cross_entropy`): its der1
//! / der2 are EXACTLY Logloss (`der1 = target - sigmoid(approx)`, `der2 =
//! -p*(1-p)`; the stronger der2 — `cb-compute::loss`). We reimplement that math
//! as `LoglossCustom`, train it through `Loss::Custom`, and gate per-tree splits,
//! per-tree leaf values, and per-iteration staged approximants against the
//! committed catboost 1.2.10 `loss_extra/cross_entropy` fixture (the SAME fixture
//! the built-in `loss_oracle_test` already gates). If the custom dispatch is
//! faithful, the custom-path model is bit-identical to the frozen oracle.
//!
//! The trait-CONTRACT half (the `Arc<dyn>` handle mechanics + per-object der
//! parity) lives in `cb-compute/tests/custom_objective_test.rs`; this test owns
//! the END-TO-END training dispatch (which needs the `cb-backend` train loop that
//! `cb-compute` cannot depend on, D-03).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;
use std::sync::Arc;

use cb_backend::CpuBackend;
use cb_compute::{sigmoid, CustomObjective, CustomObjectiveHandle, LeafMethod, Loss};
use cb_core::{CbError, CbResult};
use cb_oracle::{compare_stage, load_f64_vec, load_model_json, Stage};
use cb_train::{train, BoostParams, EBootstrapType, EOverfittingDetectorType, Model};
use ndarray::Array2;
use ndarray_npy::read_npy;

/// Logloss/CrossEntropy reimplemented as a custom objective: der1 = `target -
/// sigmoid(approx)`, der2 = `-p*(1-p)` (the exact built-in math from
/// `cb-compute::loss`). The per-object weight (empty == 1.0) is applied AFTER the
/// unweighted scalar der, matching the built-in convention.
struct LoglossCustom;

impl CustomObjective for LoglossCustom {
    fn calc_ders_range(
        &self,
        approxes: &[f64],
        targets: &[f64],
        weights: &[f64],
        ders: &mut [(f64, f64)],
    ) -> CbResult<()> {
        if approxes.len() != targets.len() || approxes.len() != ders.len() {
            return Err(CbError::Degenerate(
                "LoglossCustom: approx/target/ders length mismatch".to_owned(),
            ));
        }
        if !weights.is_empty() && weights.len() != approxes.len() {
            return Err(CbError::Degenerate(
                "LoglossCustom: weights length mismatch".to_owned(),
            ));
        }
        for (i, ((&a, &t), d)) in approxes
            .iter()
            .zip(targets.iter())
            .zip(ders.iter_mut())
            .enumerate()
        {
            let p = sigmoid(a);
            let w = if weights.is_empty() { 1.0 } else { weights[i] };
            *d = (w * (t - p), w * (-p * (1.0 - p)));
        }
        Ok(())
    }
}

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

/// Population standardize (`(y - mean)/std`, ddof=0) — matches the
/// `loss_oracle_test` cross_entropy target construction.
fn standardize(y: &[f64]) -> Vec<f64> {
    let n = y.len() as f64;
    let mean = y.iter().sum::<f64>() / n;
    let var = y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();
    y.iter().map(|v| (v - mean) / std).collect()
}

/// CrossEntropy soft label `y_soft = sigmoid(standardize(y))` — the same target
/// the `loss_extra/cross_entropy` fixture was generated against.
fn cross_entropy_target() -> Vec<f64> {
    standardize(&load_y()).iter().map(|&z| sigmoid(z)).collect()
}

/// The cross_entropy fixture's EXACT training config (depth 2, 5 iters, lr 0.1,
/// l2 3.0, no sampling, `boost_from_average=false`,
/// `leaf_estimation_method=Gradient` / `leaf_estimation_iterations=1` per the
/// committed `config.json`). The leaf method is pinned to Gradient to MATCH the
/// frozen fixture — the self-oracle reproduces the EXISTING per-stage oracle, so
/// the training config must be identical (the custom path's own Newton default
/// is a separate concern, exercised only when the caller does not pin a leaf
/// method). Mirrors `loss_oracle_test::base_params`.
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

fn train_with(loss: Loss, target: &[f64]) -> (Model, Vec<f64>) {
    let columns = load_feature_columns();
    let model_json = load_model_json(&fixture("loss_extra/cross_entropy/model.json"))
        .unwrap_or_else(|e| panic!("cross_entropy/model.json must load: {e:?}"));
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
    .unwrap_or_else(|e| panic!("custom-objective training failed: {e:?}"));
    (model, staged)
}

/// The self-oracle: Logloss-as-`Loss::Custom` reproduces the EXISTING
/// `loss_extra/cross_entropy` per-stage oracle (splits, leaf values, staged
/// approx) <= 1e-5. CrossEntropy IS Logloss math, so a faithful custom dispatch
/// is bit-identical to the frozen upstream fixture.
#[test]
fn custom_objective_reproduces_cross_entropy_oracle() {
    let target = cross_entropy_target();
    let custom = Loss::Custom(CustomObjectiveHandle::new(Arc::new(LoglossCustom)));
    let (model, staged) = train_with(custom, &target);

    let model_json = load_model_json(&fixture("loss_extra/cross_entropy/model.json")).unwrap();

    compare_stage(
        Stage::Splits,
        &model_json.split_borders(),
        &model.split_borders(),
    )
    .unwrap_or_else(|e| panic!("custom-objective splits diverged from cross_entropy oracle: {e:?}"));

    compare_stage(
        Stage::LeafValues,
        &model_json.leaf_values(),
        &model.leaf_values(),
    )
    .unwrap_or_else(|e| {
        panic!("custom-objective leaf values diverged from cross_entropy oracle: {e:?}")
    });

    let expected_staged =
        load_f64_vec(&fixture("loss_extra/cross_entropy/staged.npy")).unwrap();
    compare_stage(Stage::StagedApprox, &expected_staged, &staged).unwrap_or_else(|e| {
        panic!("custom-objective staged approx diverged from cross_entropy oracle: {e:?}")
    });
}

/// Cross-check: the custom objective produces the SAME trained model as the
/// built-in `Loss::CrossEntropy` (the in-tree path it reimplements) on the same
/// data — splits, leaf values, and staged approx are bit-identical. This anchors
/// the dispatch fidelity directly to the built-in, independent of the fixture.
#[test]
fn custom_objective_matches_builtin_cross_entropy() {
    let target = cross_entropy_target();
    let (builtin_model, builtin_staged) = train_with(Loss::CrossEntropy, &target);
    let custom = Loss::Custom(CustomObjectiveHandle::new(Arc::new(LoglossCustom)));
    let (custom_model, custom_staged) = train_with(custom, &target);

    compare_stage(
        Stage::LeafValues,
        &builtin_model.leaf_values(),
        &custom_model.leaf_values(),
    )
    .unwrap_or_else(|e| panic!("custom vs built-in CrossEntropy leaf values diverged: {e:?}"));
    compare_stage(Stage::StagedApprox, &builtin_staged, &custom_staged)
        .unwrap_or_else(|e| panic!("custom vs built-in CrossEntropy staged approx diverged: {e:?}"));
}
