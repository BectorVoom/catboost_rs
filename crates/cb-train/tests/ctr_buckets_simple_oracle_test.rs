//! E16 / SPEC-CTRT-06 (parity half) + SPEC-CTRT-12, acceptance A1 — Buckets
//! CTR end-to-end ≤1e-5 vs `catboost==1.2.10`.
//!
//! # What this locks
//!
//! 1. **≤1e-5 parity** for `simple_ctr = Buckets`, through the production
//!    `train_cat` → `predict_raw_cat` path — the online Buckets prefix
//!    (`N[b]` / `Total`) AND the per-class candidate expansion together.
//! 2. **The per-column `target_border_idx` genuinely reaches the model**: the
//!    committed fixture's generator proved upstream splits at BOTH indices
//!    (`idxs == {0, 1}`), so a model whose every split reports index `0` means
//!    the whole-tree constant crept back in.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_backend::CpuBackend;
use cb_compute::{LeafMethod, Loss};
use cb_data::stringify_int_category;
use cb_model::Model as CbModel;
use cb_oracle::{compare_stage, load_f64_vec, load_model_json, Stage};
use cb_train::{
    train_cat, BoostParams, EBootstrapType, ECtrType, EOverfittingDetectorType,
};
use ndarray::Array2;
use ndarray_npy::read_npy;

const SCENARIO: &str = "ctr_buckets_simple";

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join(rel)
}

/// The two categorical columns as SoA `Vec<String>`, stringified via
/// `stringify_int_category` — the A4 plain-integer form upstream's Pool hashed
/// when the fixture was generated.
fn load_cat_columns() -> Vec<Vec<String>> {
    let x: Array2<i32> = read_npy(fixture(&format!("{SCENARIO}/X_cat.npy")))
        .unwrap_or_else(|e| panic!("{SCENARIO}/X_cat.npy must load as int32 [N,2]: {e:?}"));
    (0..x.ncols())
        .map(|fi| {
            x.column(fi)
                .iter()
                .map(|&code| stringify_int_category(i64::from(code)))
                .collect()
        })
        .collect()
}

/// The fixture's pinned config, EVERY field explicit (Pitfall-6 discipline: a
/// changed builder default must not silently alter what this gate exercises).
fn buckets_params() -> BoostParams {
    BoostParams {
        loss: Loss::Logloss,
        iterations: 10,
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
        one_hot_max_size: 1,
        permutation_count: 1,
        fold_len_multiplier: 2.0,
        simple_ctr: ECtrType::Buckets,
        simple_ctr_priors: vec![0.5],
        counter_calc_method: cb_train::counter_calc_method_default(),
        boosting_type: cb_train::boosting_type_default(),
        max_ctr_complexity: 1,
        combinations_ctr: cb_train::combinations_ctr_default(),
        combinations_ctr_priors: vec![0.5],
        score_function: cb_train::score_function_default(),
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

/// Train the fixture through production `train_cat`.
fn fit() -> (
    cb_train::Model,
    cb_train::BakedCtrData,
    Vec<Vec<String>>,
    Vec<Vec<f64>>,
) {
    let cat_cols = load_cat_columns();
    let model_json = load_model_json(&fixture(&format!("{SCENARIO}/model.json")))
        .unwrap_or_else(|e| panic!("{SCENARIO}/model.json must load: {e:?}"));
    let borders = model_json.float_feature_borders();
    let target = load_f64_vec(&fixture(&format!("{SCENARIO}/y.npy"))).unwrap();

    let (trained, baked) = train_cat(
        &CpuBackend,
        &[], // categorical-only fixture: zero float columns
        &borders,
        &cat_cols,
        &target,
        &[],
        &buckets_params(),
        None,
    )
    .unwrap_or_else(|e| panic!("Buckets-CTR training failed: {e:?}"));

    (trained, baked, cat_cols, borders)
}

#[test]
fn buckets_simple_predictions_match_upstream_within_1e_minus_5() {
    let expected = load_f64_vec(&fixture(&format!("{SCENARIO}/predictions.npy"))).unwrap();
    let (trained, baked, cat_cols, borders) = fit();

    let model = CbModel::from_trained(&trained, borders)
        .with_ctr_data(cb_model::CtrData::from_baked(&baked));
    let actual = cb_model::predict_raw_cat(&model, &[], &cat_cols);

    assert_eq!(
        actual.len(),
        expected.len(),
        "prediction count must match upstream"
    );
    assert!(
        actual.iter().any(|v| *v != actual[0]),
        "predictions are constant — the gate would be vacuous"
    );

    let max_div = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    compare_stage(Stage::Predictions, &expected, &actual).unwrap_or_else(|e| {
        panic!("Buckets-CTR predictions diverged from upstream (max |diff| = {max_div:e}): {e:?}")
    });
    println!("buckets_simple max |diff| = {max_div:e}");
}

#[test]
fn buckets_model_carries_a_split_at_target_border_idx_one() {
    let (trained, _baked, _cat_cols, _borders) = fit();

    let idxs: Vec<usize> = trained
        .oblivious_trees
        .iter()
        .flat_map(|t| t.ctr_splits.iter())
        .map(|s| s.target_border_idx)
        .collect();

    assert!(
        !idxs.is_empty(),
        "the fixture must produce CTR splits — zero splits would make this vacuous"
    );
    assert!(
        idxs.contains(&1),
        "the per-column target_border_idx must reach CtrSplitSpec: upstream's \
         committed model splits at BOTH indices (the generator's idxs == {{0, 1}} \
         guard), but every trained split reports {idxs:?} — the whole-tree \
         constant 0 is back"
    );
}
