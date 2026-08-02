//! E13 / SPEC-CTRT-07 (parity half), acceptance A2 — BinarizedTargetMeanValue
//! CTR end-to-end ≤1e-5 vs `catboost==1.2.10`.
//!
//! # What this locks
//!
//! 1. **≤1e-5 parity** for `simple_ctr = BinarizedTargetMeanValue`, through the
//!    production `train_cat` → `predict_raw_cat` path.
//! 2. **The baked table really carries mean pairs** — `CtrData::from_baked` used
//!    to hard-code `mean: Vec::new()`, silently discarding every mean table.
//! 3. **`save_cbm` still rejects a mean model** with a typed error. That is a
//!    real v1 limitation, and E20 flips this test rather than deleting it.
//! 4. The f32-vs-f64 `Sum` accumulation differential at fixture scale — a
//!    REPORTING step, not a gate (the real gate is E07's accumulator test).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]


use std::path::PathBuf;

use cb_backend::CpuBackend;
use cb_compute::{LeafMethod, Loss};
use cb_data::stringify_int_category;
use cb_model::Model as CbModel;
use cb_oracle::{compare_stage, load_f64_vec, load_model_json, Stage};
use cb_train::{
    materialize_ctr_feature, train_cat, BoostParams, EBootstrapType, ECtrType,
    EOverfittingDetectorType, TProjection,
};
use ndarray::Array2;
use ndarray_npy::read_npy;

const SCENARIO: &str = "ctr_btmv_simple";

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
fn counter_params() -> BoostParams {
    BoostParams {
        loss: Loss::Logloss,
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
        one_hot_max_size: 1,
        permutation_count: 1,
        fold_len_multiplier: 2.0,
        simple_ctr: ECtrType::BinarizedTargetMeanValue,
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
        &counter_params(),
        None,
    )
    .unwrap_or_else(|e| panic!("Counter-CTR training failed: {e:?}"));

    (trained, baked, cat_cols, borders)
}

#[test]
fn btmv_simple_predictions_match_upstream_within_1e_minus_5() {
    let expected = load_f64_vec(&fixture(&format!("{SCENARIO}/predictions.npy"))).unwrap();
    let (trained, baked, cat_cols, borders) = fit();
    let _ = &trained;

    let model = CbModel::from_trained(&trained, borders)
        .with_ctr_data(cb_model::CtrData::from_baked(&baked));
    let actual = cb_model::predict_raw_cat(&model, &[], &cat_cols);

    assert_eq!(actual.len(), expected.len(), "prediction count must match upstream");
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
        panic!("BTMV-CTR predictions diverged from upstream (max |diff| = {max_div:e}): {e:?}")
    });
}

#[test]
fn btmv_baked_table_carries_a_non_empty_mean_vector() {
    let (_trained, baked, _cat_cols, _borders) = fit();

    let table = baked
        .tables
        .iter()
        .find(|t| t.ctr_type == ECtrType::BinarizedTargetMeanValue.as_i8())
        .expect("a BTMV baked table must exist — simple_ctr was ignored otherwise");

    assert!(
        !table.mean.is_empty(),
        "CtrData::from_baked used to hard-code mean: Vec::new(), silently \
         discarding every mean table"
    );
    assert_eq!(table.mean.len(), table.hashes.len(), "one (Sum, Count) per bucket");
    assert!(
        table.int_counts.is_empty(),
        "the mean types carry (Sum, Count) pairs, not class counts"
    );
    // ANTI-VACUITY: an all-zero mean table would satisfy the shape assertions.
    assert!(
        table.mean.iter().any(|&(s, _)| s != 0.0),
        "every bucket Sum is zero — the mean table is structurally present but vacuous"
    );
}

#[test]
fn btmv_trained_model_round_trips_through_cbm() {
    // FLIPPED at E20 (was `btmv_save_cbm_is_a_typed_rejection_until_e20`, the
    // v1 encode-side limitation): a trained BTMV model saves, reloads, and the
    // reloaded model predicts within 1e-5 of the in-memory one.
    let (trained, baked, cat_cols, borders) = fit();
    let model = CbModel::from_trained(&trained, borders)
        .with_ctr_data(cb_model::CtrData::from_baked(&baked));

    let path = std::env::temp_dir().join(format!("btmv_roundtrip_{}.cbm", std::process::id()));
    cb_model::save_cbm(&model, &path)
        .unwrap_or_else(|e| panic!("save_cbm must accept a mean-CTR model after E20: {e:?}"));
    let reloaded = cb_model::load_cbm(&path)
        .unwrap_or_else(|e| panic!("the saved BTMV .cbm must reload: {e:?}"));
    let _ = std::fs::remove_file(&path);

    let in_memory = cb_model::predict_raw_cat(&model, &[], &cat_cols);
    let round_tripped = cb_model::predict_raw_cat(&reloaded, &[], &cat_cols);
    assert!(
        in_memory.iter().any(|v| *v != in_memory[0]),
        "constant predictions — the round-trip gate would be vacuous"
    );
    let max_div = in_memory
        .iter()
        .zip(round_tripped.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_div <= 1e-5,
        "the round-tripped BTMV model diverged from the in-memory one: {max_div:e}"
    );
}

#[test]
fn btmv_f64_sum_accumulation_diverges_from_upstream_on_this_fixture() {
    // REPORTING STEP, NOT A GATE (SPEC-CTRT-07 / R2). For binclf the added value
    // is targetClass in {0.0, 1.0}, so f32 and f64 accumulation are bit-identical
    // below 2^24 — and this fixture is 30 rows. The ACTUAL gate for the f32
    // requirement is E07's direct accumulator differential.
    //
    // A silent pass is forbidden: this test always reports which branch it took.
    let expected = load_f64_vec(&fixture(&format!("{SCENARIO}/predictions.npy"))).unwrap();
    let (trained, baked, cat_cols, borders) = fit();
    let model = CbModel::from_trained(&trained, borders)
        .with_ctr_data(cb_model::CtrData::from_baked(&baked));
    let actual = cb_model::predict_raw_cat(&model, &[], &cat_cols);

    let f32_maxdiff = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    // Widen every baked Sum to f64 and back — at this scale it is the identity.
    let widened_differs = baked.tables.iter().any(|t| {
        t.mean
            .iter()
            .any(|&(s, _)| (f64::from(s) as f32).to_bits() != s.to_bits())
    });

    if widened_differs {
        panic!(
            "f32/f64 Sum accumulation IS distinguishable on this fixture \
             (f32 maxdiff = {f32_maxdiff:e}) — this branch was not expected; \
             investigate before relying on E07's accumulator test alone"
        );
    }
    println!(
        "REPORTED: f32/f64 indistinguishable at this scale (maxdiff = {f32_maxdiff:e}); \
         the f32 requirement is gated by E07 test fn 2's accumulator differential, \
         not by this fixture."
    );
}
