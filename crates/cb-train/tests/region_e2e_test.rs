//! Region grow-policy END-TO-END oracle (GPUT-18 / D-03a, Plan 12-02 Task 2).
//!
//! Trains with `grow_policy=Region` on a pinned, perfectly-separable synthetic
//! fixture (the "Region OUT" rejection is now LIFTED), lifts the trained region path
//! into the canonical `cb_model::Model` (`TreeVariant::Region`), and locks:
//!   - training PRODUCES `region_trees` (oblivious / non-symmetric stay EMPTY);
//!   - the region path is the FROZEN depth-2 structure the CPU grower oracle
//!     (`region_grow_test.rs`) pins — a d+1-leaf path, NOT a `2^d` node graph;
//!   - `predict_raw` reproduces the frozen reference `[2, 2, 0, 0, -3, -3]` ≤1e-5
//!     (this separable fixture is fit exactly in one lr=1 iteration);
//!   - training is deterministic (identical predictions on re-train).
//!
//! Lives under `tests/` (integration) so `cb_train` is the SAME external crate
//! instance `cb_model` links — a src-mounted unit test would hit the dev-dep
//! diamond (two `cb_train` versions). Mirrors `non_symmetric_grower_oracle_test.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing, clippy::float_cmp)]

use cb_backend::CpuBackend;
use cb_compute::{EScoreFunction, LeafMethod, Loss};
use cb_model::Model as CbModel;
use cb_train::{
    train, BoostParams, EBootstrapType, EGrowPolicy, EOverfittingDetectorType, Split,
};

/// The pinned, perfectly-separable fixture (matches `region_grow_test.rs`): `f0`
/// separates the three gradient groups, RMSE der1 == `-target`.
fn fixture() -> (Vec<Vec<f32>>, Vec<Vec<f64>>, Vec<f64>) {
    let columns = vec![
        vec![0.0_f32, 0.0, 1.0, 1.0, 2.0, 2.0], // f0
        vec![0.0_f32, 1.0, 0.0, 1.0, 0.0, 1.0], // f1 (unused by the grown path)
    ];
    let borders = vec![vec![0.5_f64, 1.5], vec![0.5_f64]];
    // RMSE der1 at approx 0 is `-target`; target = [2,2,0,0,-3,-3] yields der1 =
    // [-2,-2,0,0,3,3] (the grower fixture). One lr=1 iteration fits it exactly.
    let target = vec![2.0_f64, 2.0, 0.0, 0.0, -3.0, -3.0];
    (columns, borders, target)
}

/// A Region-policy [`BoostParams`]: one lr=1 iteration, no regularization / bootstrap
/// / averaging, so the separable fixture is fit exactly and the structure matches the
/// grower oracle (scaled_l2 == 0).
fn region_params() -> BoostParams {
    BoostParams {
        loss: Loss::Rmse,
        iterations: 1,
        depth: 3,
        learning_rate: 1.0,
        l2_leaf_reg: 0.0,
        random_strength: 0.0,
        boost_from_average: false,
        leaf_method: LeafMethod::Gradient,
        bootstrap_type: EBootstrapType::No,
        subsample: 1.0,
        bagging_temperature: 0.0,
        random_seed: 42,
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
        score_function: EScoreFunction::L2,
        has_time: false,
        feature_weights: cb_train::feature_weights_default(),
        first_feature_use_penalties: cb_train::first_feature_use_penalties_default(),
        per_object_feature_penalties: cb_train::per_object_feature_penalties_default(),
        penalties_coefficient: cb_train::penalties_coefficient_default(),
        monotone_constraints: cb_train::monotone_constraints_default(),
        grow_policy: EGrowPolicy::Region,
        max_leaves: cb_train::max_leaves_default(),
        min_data_in_leaf: cb_train::min_data_in_leaf_default(),
    }
}

#[test]
fn region_grow_policy_trains_and_applies_to_the_frozen_reference() {
    let (columns, borders, target) = fixture();
    let params = region_params();

    let model = train(&CpuBackend, &columns, &borders, &target, &[], &params, None)
        .unwrap_or_else(|e| panic!("region training failed: {e:?}"));

    // Region training PRODUCES region_trees; oblivious / non-symmetric stay EMPTY
    // (a model is EITHER all-oblivious OR all-non-sym OR all-region).
    assert!(
        model.oblivious_trees.is_empty() && model.non_symmetric_trees.is_empty(),
        "grow_policy=Region must not populate oblivious / non-symmetric trees"
    );
    assert_eq!(model.region_trees.len(), 1, "one region tree per iteration");

    // FROZEN structure: depth-2 path (2 levels), EXACTLY 3 leaves (d+1, NOT 2^d).
    let rt = &model.region_trees[0];
    assert_eq!(rt.splits, vec![Split { feature: 0, border: 1.5 }, Split { feature: 0, border: 0.5 }]);
    assert_eq!(rt.directions, vec![false, true]);
    assert_eq!(rt.one_hot, vec![false, false]);
    assert_eq!(rt.leaf_values.len(), rt.splits.len() + 1, "d+1 leaves, never 2^d");
    assert_eq!(rt.leaf_values.len(), 3);

    // Lift into the canonical model and apply.
    let cb_model = CbModel::from_trained(&model, borders.clone());
    assert_eq!(cb_model.region_trees.len(), 1);
    assert!(cb_model.oblivious_trees.is_empty() && cb_model.non_symmetric_trees.is_empty());

    let preds = cb_model::predict_raw(&cb_model, &columns);
    // FROZEN reference: the separable fixture is fit exactly in one lr=1 iteration.
    let expected = [2.0, 2.0, 0.0, 0.0, -3.0, -3.0];
    assert_eq!(preds.len(), expected.len());
    for (i, (&p, &e)) in preds.iter().zip(expected.iter()).enumerate() {
        assert!(
            (p - e).abs() <= 1e-5,
            "object {i}: region prediction {p} != frozen reference {e} (>1e-5)"
        );
    }
}

#[test]
fn region_grow_policy_training_is_deterministic() {
    let (columns, borders, target) = fixture();
    let params = region_params();
    let a = train(&CpuBackend, &columns, &borders, &target, &[], &params, None)
        .expect("first region train");
    let b = train(&CpuBackend, &columns, &borders, &target, &[], &params, None)
        .expect("second region train");
    let ca = CbModel::from_trained(&a, borders.clone());
    let cb = CbModel::from_trained(&b, borders.clone());
    assert_eq!(
        cb_model::predict_raw(&ca, &columns),
        cb_model::predict_raw(&cb, &columns),
        "region training must be deterministic"
    );
}
