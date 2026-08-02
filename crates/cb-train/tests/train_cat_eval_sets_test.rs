//! E21 (enabling task for SPEC-CTRT-17): `EvalSet` carries categorical columns
//! and `train_cat_with_eval_sets` exists, so `counter_calc_method = Full` —
//! which counts learn + EVERY eval set into the Counter bucket totals
//! (`online_ctr.cpp:716-729`) — becomes structurally expressible at all.
//!
//! Behavior is UNCHANGED here (E22 threads the method into the tally); this
//! task pins the API shape, the typed length-mismatch rejection, and that the
//! existing numeric eval-set paths stay byte-identical with `cat_columns: &[]`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use cb_backend::CpuBackend;
use cb_compute::{LeafMethod, Loss};
use cb_train::{
    train_cat_with_eval_sets, BoostParams, EBootstrapType, ECtrType, EOverfittingDetectorType,
    EvalSet,
};

fn cat_params() -> BoostParams {
    BoostParams {
        loss: Loss::Logloss,
        iterations: 3,
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

/// A 30-row learn set with one cat column (cardinality 6) whose label follows
/// the category, plus a 20-row eval set of the same shape.
fn learn_and_eval() -> (Vec<Vec<String>>, Vec<f64>, Vec<Vec<String>>, Vec<f64>) {
    let learn_cat: Vec<String> = (0..30).map(|i| format!("c{}", i % 6)).collect();
    let learn_y: Vec<f64> = (0..30).map(|i| f64::from(u8::from(i % 6 < 3))).collect();
    let eval_cat: Vec<String> = (0..20).map(|i| format!("c{}", i % 6)).collect();
    let eval_y: Vec<f64> = (0..20).map(|i| f64::from(u8::from(i % 6 < 3))).collect();
    (vec![learn_cat], learn_y, vec![eval_cat], eval_y)
}

#[test]
fn train_cat_with_eval_sets_accepts_categorical_eval_columns() {
    let (learn_cats, learn_y, eval_cats, eval_y) = learn_and_eval();
    let weights = vec![1.0_f64; 30];

    let eval_sets = vec![EvalSet {
        feature_values: &[],
        target: &eval_y,
        cat_columns: &eval_cats,
    }];
    assert_eq!(eval_sets[0].cat_columns.len(), 1, "the cat column is carried");

    let (model, baked) = train_cat_with_eval_sets(
        &CpuBackend,
        &[],
        &[],
        &learn_cats,
        &learn_y,
        &weights,
        &cat_params(),
        None,
        &eval_sets,
        None,
    )
    .expect("categorical training with an eval set must succeed");

    assert!(!model.oblivious_trees.is_empty(), "trees must train");
    assert!(!baked.tables.is_empty(), "the Counter table must bake");
}

#[test]
fn eval_set_cat_column_length_mismatch_is_a_typed_error() {
    let (learn_cats, learn_y, _eval_cats, eval_y) = learn_and_eval();
    let weights = vec![1.0_f64; 30];

    // 19 cat values for a 20-row eval target — a length mismatch.
    let short_cat: Vec<String> = (0..19).map(|i| format!("c{}", i % 6)).collect();
    let bad_eval = vec![short_cat];
    let eval_sets = vec![EvalSet {
        feature_values: &[],
        target: &eval_y,
        cat_columns: &bad_eval,
    }];

    let result = train_cat_with_eval_sets(
        &CpuBackend,
        &[],
        &[],
        &learn_cats,
        &learn_y,
        &weights,
        &cat_params(),
        None,
        &eval_sets,
        None,
    );
    match result {
        Err(cb_core::CbError::LengthMismatch { .. }) => {}
        other => panic!("a cat-column/target length mismatch must be a typed LengthMismatch, got {other:?}"),
    }
}
