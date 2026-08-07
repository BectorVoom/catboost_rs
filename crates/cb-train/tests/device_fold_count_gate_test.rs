//! GDC-01 (T01): `begin_device_training` must receive the REAL
//! `learning_folds_for_cycle`, never a hardcoded `1`. A CPU-only mock records the
//! `fold_count` the boosting loop hands it.
//!
//! V-1/B-1 correction (PLAN §1): for a NON-CTR fit
//! `learning_fold_count(pc, false) == 1` for every `pc`, so a "no CTR +
//! `permutation_count > 1`" positive case is vacuous. And a CTR fit is not yet
//! `device_host_eligible` (the CTR clause relaxes in GDC-11/T14), so
//! `begin_device_training` is never reached with CTR candidates in this task's
//! wave. The positive `fold_count > 1` observation is therefore asserted at the
//! source (`learning_fold_count`) here, and end-to-end by T14's
//! `multi_permutation_ctr_declines_to_device` once the CTR clause opens.

use std::cell::Cell;

use cb_compute::{
    Derivatives, DeviceGrownTree, EScoreFunction, FamilyTreeArgs, LeafMethod, Loss, Runtime,
};
use cb_core::{CbError, CbResult};
use cb_train::{
    boosting_type_default, combinations_ctr_default, combinations_ctr_priors_default,
    counter_calc_method_default, feature_weights_default, first_feature_use_penalties_default,
    fold_len_multiplier_default, grow_policy_default, has_time_default, learning_fold_count,
    max_ctr_complexity_default, max_leaves_default, min_data_in_leaf_default,
    monotone_constraints_default, per_object_feature_penalties_default, permutation_count_default,
    score_function_default, simple_ctr_default, simple_ctr_priors_default, train, BoostParams,
    EBootstrapType, EOverfittingDetectorType,
};

/// A CPU-only seam mock that records the `fold_count` argument handed to
/// `begin_device_training`, accepts the session, and grows a canned depth-1 tree
/// (the `device_seam_test.rs` pattern).
struct FoldCountRecorder {
    recorded: Cell<Option<usize>>,
}

impl Runtime for FoldCountRecorder {
    fn compute_gradients(
        &self,
        _loss: &Loss,
        _approx: &[f64],
        _target: &[f64],
        _approx_dimension: usize,
    ) -> CbResult<Derivatives> {
        Err(CbError::Degenerate(
            "compute_gradients must not be called on the device path".to_owned(),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn begin_device_training(
        &self,
        _loss: &Loss,
        _depth: usize,
        _boosting_type_is_plain: bool,
        fold_count: usize,
        _score_function: EScoreFunction,
        _bins_feature_major: &[u32],
        _weight: &[f64],
        _n: usize,
        _n_features: usize,
        _n_bins: usize,
        _learning_rate: f64,
        _scaled_l2: f64,
        _config: &cb_compute::DeviceTrainConfig,
    ) -> CbResult<bool> {
        self.recorded.set(Some(fold_count));
        Ok(true)
    }

    fn grow_tree_on_device(
        &self,
        _approx: &[f64],
        _target: &[f64],
        _sample: &[f64],
        _family: Option<&FamilyTreeArgs<'_>>,
    ) -> CbResult<Option<DeviceGrownTree>> {
        Ok(Some(DeviceGrownTree {
            splits: vec![(0, 1, false)],
            leaf_values: vec![2.0, -3.0],
            approx_dim: 1,
            leaf_of: Vec::new(),
            step_nodes: Vec::new(),
            node_id_to_leaf_id: Vec::new(),
            region_path: Vec::new(),
        }))
    }

    fn end_device_training(&self) -> CbResult<()> {
        Ok(())
    }
}

/// One float feature with three ascending borders.
fn feature_borders() -> Vec<Vec<f64>> {
    vec![vec![0.5, 1.5, 2.5]]
}

/// Four objects on the lone float feature.
fn feature_columns() -> Vec<Vec<f32>> {
    vec![vec![0.0, 1.0, 2.0, 3.0]]
}

/// A device-eligible RMSE / depth-1 / Plain numeric config (mirrors
/// `device_seam_test.rs::device_params`).
fn device_params() -> BoostParams {
    BoostParams {
        loss: Loss::Rmse,
        iterations: 1,
        depth: 1,
        learning_rate: 0.1,
        auto_learning_rate: false,
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
        one_hot_max_size: cb_train::one_hot_max_size_default(),
        permutation_count: permutation_count_default(),
        fold_len_multiplier: fold_len_multiplier_default(),
        simple_ctr: simple_ctr_default(),
        simple_ctr_priors: simple_ctr_priors_default(),
        counter_calc_method: counter_calc_method_default(),
        boosting_type: boosting_type_default(),
        max_ctr_complexity: max_ctr_complexity_default(),
        combinations_ctr: combinations_ctr_default(),
        combinations_ctr_priors: combinations_ctr_priors_default(),
        score_function: score_function_default(),
        has_time: has_time_default(),
        feature_weights: feature_weights_default(),
        first_feature_use_penalties: first_feature_use_penalties_default(),
        per_object_feature_penalties: per_object_feature_penalties_default(),
        penalties_coefficient: cb_train::penalties_coefficient_default(),
        monotone_constraints: monotone_constraints_default(),
        grow_policy: grow_policy_default(),
        max_leaves: max_leaves_default(),
        min_data_in_leaf: min_data_in_leaf_default(),
    }
}

/// D-04 regression: a plain (non-CTR) device-eligible fit still hands
/// `fold_count == 1` to the backend, whatever `permutation_count` is
/// (`learning_fold_count(pc, false) == 1`).
#[test]
fn plain_fit_still_passes_fold_count_one() {
    for pc in [1_usize, 4] {
        let mock = FoldCountRecorder {
            recorded: Cell::new(None),
        };
        let params = BoostParams {
            permutation_count: pc,
            ..device_params()
        };
        let target = vec![1.0, 2.0, 3.0, 4.0];
        train(
            &mock,
            &feature_columns(),
            &feature_borders(),
            &target,
            &[],
            &params,
            None,
        )
        .expect("device fit must succeed");
        assert_eq!(
            mock.recorded.get(),
            Some(1),
            "non-CTR fit (permutation_count={pc}) must hand fold_count == 1"
        );
    }
}

/// The source of the threaded value: with CTR candidates present the real
/// learning-fold count is `max(1, pc - 1)`, not `1`. (The end-to-end observation
/// through `begin_device_training` requires the GDC-11 CTR clause relaxation and
/// lives in `device_ctr_gate_test.rs`.)
#[test]
fn learning_fold_count_is_the_threaded_source() {
    assert_eq!(learning_fold_count(4, true), 3);
    assert_eq!(learning_fold_count(2, true), 1);
    assert_eq!(learning_fold_count(1, true), 1);
    assert_eq!(learning_fold_count(4, false), 1);
    assert_eq!(learning_fold_count(1, false), 1);
}
