//! Structural validation of the GPUT-01 device grow seam wiring in cb-train's
//! boosting loop (Plan 10-08 Task 2). A CPU-only MOCK [`Runtime`] returns a canned
//! depth-1 [`DeviceGrownTree`] so the `(feature, bin_id) -> border` join
//! (`border = feature_borders[feature][bin_id]`, Pattern 4), the per-fit
//! all-or-nothing gate (D-10-01 / T-10-23), and the `bin_id` range-check (T-10-22)
//! are exercised WITHOUT a GPU. The authoritative device oracle is Kaggle CUDA
//! (10-09, human-gated); this test locks the host-side fold correctness.
//!
//! The mock's `compute_gradients` deliberately ERRORS so any test path that reaches
//! the CPU grower fails loudly — proving the device branch was taken when the mock
//! accepts the session (`begin -> Ok(true)`), and proving the CPU fallback is
//! selected when it declines (`begin -> Ok(false)`).

use cb_compute::{Derivatives, DeviceGrownTree, EScoreFunction, LeafMethod, Loss, Runtime};
use cb_core::{CbError, CbResult};
use cb_train::{
    boosting_type_default, combinations_ctr_default, combinations_ctr_priors_default,
    counter_calc_method_default, feature_weights_default, first_feature_use_penalties_default,
    fold_len_multiplier_default, grow_policy_default, has_time_default, leaf_index,
    max_ctr_complexity_default, max_leaves_default, min_data_in_leaf_default,
    monotone_constraints_default, per_object_feature_penalties_default, permutation_count_default,
    score_function_default, simple_ctr_default, simple_ctr_priors_default, train, BoostParams,
    EBootstrapType, EOverfittingDetectorType,
};

/// A configurable CPU-only device seam test double.
struct DeviceMock {
    /// What `begin_device_training` returns (`Ok(true)` accepts the device path).
    accept_begin: bool,
    /// What `grow_tree_on_device` returns each iteration (`None` -> `Ok(None)`).
    grow: Option<DeviceGrownTree>,
}

impl Runtime for DeviceMock {
    fn compute_gradients(
        &self,
        _loss: &Loss,
        _approx: &[f64],
        _target: &[f64],
        _approx_dimension: usize,
    ) -> CbResult<Derivatives> {
        // Reaching the CPU grower on a device-accepted fit is a wiring bug — fail
        // loudly so the "device branch was taken" assertion is real.
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
        _fold_count: usize,
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
        Ok(self.accept_begin)
    }

    fn grow_tree_on_device(
        &self,
        _approx: &[f64],
        _target: &[f64],
    ) -> CbResult<Option<DeviceGrownTree>> {
        Ok(self.grow.clone())
    }

    fn end_device_training(&self) -> CbResult<()> {
        Ok(())
    }
}

/// One float feature with three ascending borders (bin ids 0..=2 valid).
fn feature_borders() -> Vec<Vec<f64>> {
    vec![vec![0.5, 1.5, 2.5]]
}

/// Four objects on the lone float feature.
fn feature_columns() -> Vec<Vec<f32>> {
    vec![vec![0.0, 1.0, 2.0, 3.0]]
}

/// A device-eligible RMSE / depth-1 / Plain numeric config (`boost_from_average =
/// false` so the bias is `0`, keeping the staged assertion a pure tree
/// contribution). Every host-only complexity (CTR / ordered / penalties / monotone
/// / sampling / perturbation / eval) is left at its default-off value so the device
/// host-eligibility gate passes.
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

#[test]
fn device_seam_folds_depth1_tree_via_bin_border_join() {
    // Device returns split (feature 0, bin_id 1) -> border must resolve to
    // feature_borders[0][1] = 1.5; UN-scaled leaf values [2.0, -3.0].
    let dev_tree = DeviceGrownTree {
        splits: vec![(0, 1)],
        leaf_values: vec![2.0, -3.0],
        leaf_of: Vec::new(),
        step_nodes: Vec::new(),
        node_id_to_leaf_id: Vec::new(),
    };
    let mock = DeviceMock {
        accept_begin: true,
        grow: Some(dev_tree),
    };
    let borders = feature_borders();
    let columns = feature_columns();
    let target = vec![1.0, 2.0, 3.0, 4.0];
    let mut staged = Vec::new();
    let model = train(
        &mock,
        &columns,
        &borders,
        &target,
        &[],
        &device_params(),
        Some(&mut staged),
    )
    .expect("device fit must succeed");

    // Exactly one device-grown oblivious tree; no non-symmetric trees.
    assert_eq!(model.oblivious_trees.len(), 1, "one device tree");
    assert!(model.non_symmetric_trees.is_empty());
    let tree = &model.oblivious_trees[0];

    // bin_id -> border join (Pattern 4).
    assert_eq!(tree.splits.len(), 1);
    assert_eq!(tree.splits[0].feature, 0);
    assert!(
        (tree.splits[0].border - 1.5).abs() < 1e-12,
        "border must resolve to feature_borders[0][1]=1.5, got {}",
        tree.splits[0].border
    );

    // Leaf values are the device's UN-scaled leaves * learning_rate (0.1), no
    // pairwise centering for RMSE (D-04 leaf-update contract).
    assert_eq!(tree.leaf_values.len(), 2);
    assert!((tree.leaf_values[0] - 0.2).abs() < 1e-12);
    assert!((tree.leaf_values[1] + 0.3).abs() < 1e-12);

    // Staged approx = bias (0) + per-object leaf contribution, using the SAME
    // forward-bit leaf assignment (value > 1.5) the fold applied.
    assert!(model.bias.abs() < 1e-12, "boost_from_average=false -> bias 0");
    assert_eq!(staged.len(), columns[0].len());
    for (i, &v) in columns[0].iter().enumerate() {
        let leaf = leaf_index(&[f64::from(v) > 1.5]);
        let expected = tree.leaf_values[leaf];
        assert!(
            (staged[i] - expected).abs() < 1e-12,
            "object {i}: staged {} != expected {expected}",
            staged[i]
        );
    }
}

#[test]
fn device_all_or_nothing_rejects_none_after_begin() {
    // begin accepts (Ok(true)) but grow returns Ok(None): mixing a CPU-grown tree
    // into a device-grown model is forbidden (D-10-01 / T-10-23) -> typed error.
    let mock = DeviceMock {
        accept_begin: true,
        grow: None,
    };
    let target = vec![1.0, 2.0, 3.0, 4.0];
    let err = train(
        &mock,
        &feature_columns(),
        &feature_borders(),
        &target,
        &[],
        &device_params(),
        None,
    )
    .expect_err("Ok(None) after a committed device fit must error");
    match err {
        CbError::Degenerate(msg) => assert!(
            msg.contains("all-or-nothing"),
            "unexpected message: {msg}"
        ),
        other => panic!("expected Degenerate all-or-nothing error, got {other:?}"),
    }
}

#[test]
fn device_bin_id_out_of_range_is_typed_error() {
    // Feature 0 has 3 borders (valid bin ids 0..=2); bin_id 5 is out of range and
    // must surface a typed OutOfRange error, never a panic / raw index (T-10-22).
    let dev_tree = DeviceGrownTree {
        splits: vec![(0, 5)],
        leaf_values: vec![1.0, -1.0],
        leaf_of: Vec::new(),
        step_nodes: Vec::new(),
        node_id_to_leaf_id: Vec::new(),
    };
    let mock = DeviceMock {
        accept_begin: true,
        grow: Some(dev_tree),
    };
    let target = vec![1.0, 2.0, 3.0, 4.0];
    let err = train(
        &mock,
        &feature_columns(),
        &feature_borders(),
        &target,
        &[],
        &device_params(),
        None,
    )
    .expect_err("out-of-range bin_id must error");
    match err {
        CbError::OutOfRange(msg) => assert!(msg.contains("bin_id 5"), "unexpected message: {msg}"),
        other => panic!("expected OutOfRange error, got {other:?}"),
    }
}

#[test]
fn device_declines_nonzero_starting_bias_boost_from_average() {
    // CR-01 regression: `boost_from_average: true` on RMSE (the CatBoostBuilder
    // default) makes `starting_approx` the target mean (2.5 here) — a non-zero
    // bias the device session cannot seed (it always starts resident approx at
    // zero). The host gate must therefore DECLINE the device path and fall back
    // to the CPU grower, which calls the mock's `compute_gradients` sentinel.
    // Reaching THAT error proves the CPU fallback (D-04) was taken even though
    // the mock's `begin` would have accepted the device path.
    let mock = DeviceMock {
        accept_begin: true,
        grow: Some(DeviceGrownTree {
            splits: vec![(0, 1)],
            leaf_values: vec![2.0, -3.0],
            leaf_of: Vec::new(),
            step_nodes: Vec::new(),
            node_id_to_leaf_id: Vec::new(),
        }),
    };
    let params = BoostParams {
        boost_from_average: true,
        ..device_params()
    };
    // Non-zero target mean -> non-zero starting bias.
    let target = vec![1.0, 2.0, 3.0, 4.0];
    let err = train(
        &mock,
        &feature_columns(),
        &feature_borders(),
        &target,
        &[],
        &params,
        None,
    )
    .expect_err("non-zero starting bias must route to the CPU grower");
    match err {
        CbError::Degenerate(msg) => assert!(
            msg.contains("compute_gradients must not be called"),
            "expected the CPU-path sentinel (bias fallback), got: {msg}"
        ),
        other => panic!("expected the CPU-path compute_gradients sentinel, got {other:?}"),
    }
}

#[test]
fn device_declines_newton_leaf_method_on_covered_loss() {
    // CR-02 regression: Newton leaf estimation on a device-covered loss (Logloss)
    // diverges from the device grower's `calc_average` (Gradient) formula because
    // `der2 = -p(1-p)` varies per object. The device grower has no Newton arm, so
    // the host gate must DECLINE and fall back to the CPU grower, hitting the
    // mock's `compute_gradients` sentinel. `boost_from_average` stays false (bias
    // 0) so CR-01 is NOT the reason for the fallback — the leaf method is.
    let mock = DeviceMock {
        accept_begin: true,
        grow: Some(DeviceGrownTree {
            splits: vec![(0, 1)],
            leaf_values: vec![2.0, -3.0],
            leaf_of: Vec::new(),
            step_nodes: Vec::new(),
            node_id_to_leaf_id: Vec::new(),
        }),
    };
    let params = BoostParams {
        loss: Loss::Logloss,
        leaf_method: LeafMethod::Newton,
        ..device_params()
    };
    // Binary target for Logloss.
    let target = vec![0.0, 1.0, 0.0, 1.0];
    let err = train(
        &mock,
        &feature_columns(),
        &feature_borders(),
        &target,
        &[],
        &params,
        None,
    )
    .expect_err("Newton leaf method must route to the CPU grower");
    match err {
        CbError::Degenerate(msg) => assert!(
            msg.contains("compute_gradients must not be called"),
            "expected the CPU-path sentinel (Newton fallback), got: {msg}"
        ),
        other => panic!("expected the CPU-path compute_gradients sentinel, got {other:?}"),
    }
}

#[test]
fn device_declined_begin_falls_back_to_cpu_path() {
    // begin declines (Ok(false)): the fit must use the CPU grower, which calls the
    // mock's compute_gradients -> the sentinel error. Reaching THAT error proves the
    // CPU fallback path (D-04) was taken rather than the device branch.
    let mock = DeviceMock {
        accept_begin: false,
        grow: Some(DeviceGrownTree {
            splits: vec![(0, 1)],
            leaf_values: vec![2.0, -3.0],
            leaf_of: Vec::new(),
            step_nodes: Vec::new(),
            node_id_to_leaf_id: Vec::new(),
        }),
    };
    let target = vec![1.0, 2.0, 3.0, 4.0];
    let err = train(
        &mock,
        &feature_columns(),
        &feature_borders(),
        &target,
        &[],
        &device_params(),
        None,
    )
    .expect_err("declined device begin must route to the CPU grower");
    match err {
        CbError::Degenerate(msg) => assert!(
            msg.contains("compute_gradients must not be called"),
            "expected the CPU-path sentinel, got: {msg}"
        ),
        other => panic!("expected the CPU-path compute_gradients sentinel, got {other:?}"),
    }
}
