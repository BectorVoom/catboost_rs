//! E17 / SPEC-CTRT-10 (parity half), acceptance A5 — mixed simple-vs-combination
//! CTR routing, end-to-end ≤1e-5 vs `catboost==1.2.10`.
//!
//! # What this locks
//!
//! The `is_simple` discriminator routes each candidate to ITS side's type AND
//! prior — `simple_ctr = Buckets:Prior=0.5` versus
//! `combinations_ctr = Counter:Prior=0.25` — through materialization, scoring,
//! the bake, and prediction. Types and priors differ on BOTH axes, so any
//! cross-routing (one side's config governing the other) fails both the baked
//! table assertions and the numeric gate.
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

const SCENARIO: &str = "ctr_mixed_simple_vs_combo";

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join(rel)
}

/// The three categorical columns as SoA `Vec<String>`, stringified via
/// `stringify_int_category` — the A4 plain-integer form upstream's Pool hashed
/// when the fixture was generated.
fn load_cat_columns() -> Vec<Vec<String>> {
    let x: Array2<i32> = read_npy(fixture(&format!("{SCENARIO}/X_cat.npy")))
        .unwrap_or_else(|e| panic!("{SCENARIO}/X_cat.npy must load as int32 [N,3]: {e:?}"));
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
fn mixed_params() -> BoostParams {
    BoostParams {
        loss: Loss::Logloss,
        iterations: 10,
        depth: 3,
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
        max_ctr_complexity: 2,
        combinations_ctr: ECtrType::Counter,
        combinations_ctr_priors: vec![0.25],
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
        extra: Default::default(),
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
        &mixed_params(),
        None,
    )
    .unwrap_or_else(|e| panic!("mixed simple-vs-combo CTR training failed: {e:?}"));

    (trained, baked, cat_cols, borders)
}

#[test]
fn mixed_simple_vs_combo_predictions_match_upstream_within_1e_minus_5() {
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
        panic!(
            "mixed simple-vs-combo predictions diverged from upstream \
             (max |diff| = {max_div:e}): {e:?}"
        )
    });
    println!("mixed_simple_vs_combo max |diff| = {max_div:e}");
}

#[test]
fn simple_projections_are_baked_as_buckets_and_combinations_as_counter() {
    let (_trained, baked, _cat_cols, _borders) = fit();

    let mut has_simple = false;
    let mut has_combo = false;
    for t in &baked.tables {
        let members = t.projection.cat_features().len();
        if members == 1 {
            has_simple = true;
            assert_eq!(
                t.ctr_type,
                ECtrType::Buckets.as_i8(),
                "a SIMPLE projection must bake with simple_ctr (Buckets)"
            );
            assert_eq!(
                t.prior_num, 0.5,
                "a SIMPLE projection must bake with simple_ctr_priors, got {}",
                t.prior_num
            );
        } else {
            has_combo = true;
            assert_eq!(
                t.ctr_type,
                ECtrType::Counter.as_i8(),
                "a COMBINATION projection must bake with combinations_ctr (Counter)"
            );
            assert_eq!(
                t.prior_num, 0.25,
                "a COMBINATION projection must bake with combinations_ctr_priors, got {}",
                t.prior_num
            );
        }
    }
    // Anti-vacuity: the model must genuinely carry BOTH kinds (the generator's
    // guard proved upstream does).
    assert!(
        has_simple && has_combo,
        "both a simple and a combination table must be baked \
         (has_simple = {has_simple}, has_combo = {has_combo})"
    );
}
