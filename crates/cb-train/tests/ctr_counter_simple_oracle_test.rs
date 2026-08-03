//! E12 / SPEC-CTRT-08 (parity half), acceptance A3 — Counter-CTR end-to-end
//! ≤1e-5 vs `catboost==1.2.10`.
//!
//! # What this locks
//!
//! 1. **≤1e-5 parity**: training with `simple_ctr = Counter` reproduces upstream's
//!    predictions on the frozen `ctr_counter_simple` fixture, through the
//!    production `train_cat` → `predict_raw_cat` path.
//! 2. **The model really carries a Counter table** — the Rust-side twin of the
//!    generator's anti-false-pass guard. Before this plan `simple_ctr` was inert
//!    and the engine trained Borders CTRs regardless, so a parity gate alone
//!    could in principle be satisfied by the wrong type on a forgiving corpus.
//! 3. **Counter is permutation-INdependent end to end**, with an anti-vacuity
//!    companion proving the corpus can actually tell the two apart.
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

const SCENARIO: &str = "ctr_counter_simple";

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
        simple_ctr: ECtrType::Counter,
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
fn counter_simple_predictions_match_upstream_within_1e_minus_5() {
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

    // Non-degeneracy: a constant prediction vector would satisfy a tolerance gate
    // against a constant upstream vector while proving nothing.
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
        panic!("Counter-CTR predictions diverged from upstream (max |diff| = {max_div:e}): {e:?}")
    });
}

#[test]
fn counter_simple_model_actually_carries_a_counter_ctr_split() {
    let (trained, baked, _cat_cols, _borders) = fit();

    // The Rust-side twin of the generator's anti-false-pass guard: before this
    // plan the engine trained Borders CTRs no matter what `simple_ctr` said.
    assert!(
        baked
            .tables
            .iter()
            .any(|t| t.ctr_type == ECtrType::Counter.as_i8()),
        "no baked Counter table — simple_ctr = Counter was ignored and the parity \
         gate is measuring the wrong CTR type"
    );

    let ctr_split_count: usize = trained
        .oblivious_trees
        .iter()
        .map(|t| t.ctr_splits.len())
        .sum();
    assert!(
        ctr_split_count > 0,
        "the model carries no CTR split at all — the fixture would pass trivially"
    );
}

#[test]
fn counter_column_is_permutation_invariant_end_to_end() {
    let cat_cols = load_cat_columns();
    let target = load_f64_vec(&fixture(&format!("{SCENARIO}/y.npy"))).unwrap();
    let target_class: Vec<usize> = target.iter().map(|&t| usize::from(t > 0.5)).collect();
    let n = target_class.len();
    let proj = TProjection::from_features(&[0]);

    let identity: Vec<i32> = (0..n as i32).collect();
    let shuffled = cb_train::fisher_yates_permutation(n, 0);
    assert_ne!(identity, shuffled, "the two permutations must differ");

    let materialize = |perm: &[i32], ty: ECtrType| {
        materialize_ctr_feature(
            &cat_cols,
            &proj,
            perm,
            &target_class,
            0.5,
            1.0,
            cb_train::ctr_border_count_default(),
            ty,
            0,
            &[],
        )
        .expect("materialize must succeed")
    };

    // The assertion is on the MATERIALIZED PER-DOCUMENT COLUMN, not on the baked
    // table. The baked table comes from `accumulate_online` over the whole learn
    // set, which is permutation-independent for EVERY type including Borders — so
    // a baked-table comparison would be vacuous and would still pass against a
    // regression that turned the Counter column back into a prefix.
    let a = materialize(&identity, ECtrType::Counter);
    let b = materialize(&shuffled, ECtrType::Counter);

    assert_eq!(
        a.bins, b.bins,
        "Counter is permutation-INDEPENDENT \
         (IsPermutationDependentCtrType(Counter)==false, ctr_type.cpp:43-56); \
         a prefix implementation would differ here"
    );
    for (x, y) in a.ctr_value.iter().zip(b.ctr_value.iter()) {
        assert_eq!(x.to_bits(), y.to_bits(), "Counter CTR values must be bit-equal");
    }
}

#[test]
fn borders_column_is_not_permutation_invariant_on_the_same_corpus() {
    // THE ANTI-VACUITY COMPANION. Without it, the test above does not
    // discriminate Counter from a prefix type: if a Borders column happened to be
    // permutation-invariant on this corpus too, the invariance assertion would
    // hold for the wrong reason.
    let cat_cols = load_cat_columns();
    let target = load_f64_vec(&fixture(&format!("{SCENARIO}/y.npy"))).unwrap();
    let target_class: Vec<usize> = target.iter().map(|&t| usize::from(t > 0.5)).collect();
    let n = target_class.len();
    let proj = TProjection::from_features(&[0]);

    let identity: Vec<i32> = (0..n as i32).collect();
    let shuffled = cb_train::fisher_yates_permutation(n, 0);

    let materialize = |perm: &[i32]| {
        materialize_ctr_feature(
            &cat_cols,
            &proj,
            perm,
            &target_class,
            0.5,
            1.0,
            cb_train::ctr_border_count_default(),
            ECtrType::Borders,
            0,
            &[],
        )
        .expect("materialize must succeed")
    };

    let a = materialize(&identity);
    let b = materialize(&shuffled);

    assert_ne!(
        a.ctr_value, b.ctr_value,
        "if a Borders column is ALSO permutation-invariant on this corpus then the \
         Counter invariance test proves nothing — widen the corpus or the \
         permutation, do NOT weaken either assertion"
    );
}
