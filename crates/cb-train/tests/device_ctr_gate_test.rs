//! GDC-11 (T14): the `device_host_eligible` CTR clauses are RELAXED — a
//! single-permutation, simple-Borders CTR fit with float columns now COMMITS to
//! the device (counting wrapper, anti-false-pass), while a
//! `permutation_count > 1` CTR fit still declines (the `learning_folds_for_cycle
//! == 1` guard, made real by GDC-01/T01 — acceptance scenario 3).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing, clippy::float_cmp)]

use std::path::PathBuf;

use cb_compute::{EScoreFunction, LeafMethod, Loss};
use cb_train::{BoostParams, EBootstrapType, EOverfittingDetectorType};

#[cfg_attr(not(any(feature = "rocm", feature = "cuda")), allow(dead_code))]
fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join("ctr_device_mixed")
        .join(rel)
}

/// The `ctr_device_mixed` pinned config (mirrors its `config.json` params).
fn ctr_params(permutation_count: usize) -> BoostParams {
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
        permutation_count,
        fold_len_multiplier: 2.0,
        simple_ctr: cb_train::ECtrType::Borders,
        simple_ctr_priors: vec![0.5],
        counter_calc_method: cb_train::counter_calc_method_default(),
        boosting_type: cb_train::boosting_type_default(),
        max_ctr_complexity: 1,
        combinations_ctr: cb_train::combinations_ctr_default(),
        combinations_ctr_priors: cb_train::combinations_ctr_priors_default(),
        score_function: EScoreFunction::L2,
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

#[cfg(any(feature = "rocm", feature = "cuda"))]
mod device {
    use std::cell::Cell;

    use super::{ctr_params, fixture};
    use cb_backend::GpuBackend;
    use cb_compute::{
        DeviceGrownTree, DeviceTrainConfig, EScoreFunction, FamilyTreeArgs, Loss, Runtime,
    };
    use cb_core::CbResult;
    use cb_data::stringify_int_category;
    use cb_train::train_cat;
    use ndarray::{Array1, Array2};
    use ndarray_npy::read_npy;

    pub struct CountingGpu {
        pub inner: GpuBackend,
        pub grown: Cell<usize>,
    }

    impl Runtime for CountingGpu {
        fn compute_gradients(
            &self,
            loss: &Loss,
            approx: &[f64],
            target: &[f64],
            dim: usize,
        ) -> CbResult<cb_compute::Derivatives> {
            self.inner.compute_gradients(loss, approx, target, dim)
        }

        #[allow(clippy::too_many_arguments)]
        fn begin_device_training(
            &self,
            loss: &Loss,
            depth: usize,
            plain: bool,
            fold_count: usize,
            score_function: EScoreFunction,
            bins: &[u32],
            weight: &[f64],
            n: usize,
            n_features: usize,
            n_bins: usize,
            lr: f64,
            scaled_l2: f64,
            config: &DeviceTrainConfig,
        ) -> CbResult<bool> {
            self.inner.begin_device_training(
                loss, depth, plain, fold_count, score_function, bins, weight, n, n_features,
                n_bins, lr, scaled_l2, config,
            )
        }

        fn grow_tree_on_device(
            &self,
            approx: &[f64],
            target: &[f64],
            sample: &[f64],
        family: Option<&FamilyTreeArgs<'_>>,
        ) -> CbResult<Option<DeviceGrownTree>> {
            let out = self.inner.grow_tree_on_device(approx, target, sample, family)?;
            if out.is_some() {
                self.grown.set(self.grown.get() + 1);
            }
            Ok(out)
        }

        fn end_device_training(&self) -> CbResult<()> {
            self.inner.end_device_training()
        }
    }

    pub fn load_inputs() -> (Vec<Vec<f32>>, Vec<Vec<f64>>, Vec<Vec<String>>, Vec<f64>) {
        let x: Array2<f32> = read_npy(fixture("X.npy")).expect("X.npy loads");
        let columns: Vec<Vec<f32>> = (0..x.ncols()).map(|fi| x.column(fi).to_vec()).collect();
        let borders_arr: Array2<f64> = read_npy(fixture("borders.npy")).expect("borders.npy");
        let borders: Vec<Vec<f64>> =
            (0..borders_arr.nrows()).map(|r| borders_arr.row(r).to_vec()).collect();
        let cat: Array1<i32> = read_npy(fixture("X_cat.npy")).expect("X_cat.npy");
        let cat_columns =
            vec![cat.iter().map(|&c| stringify_int_category(i64::from(c))).collect()];
        let target: Vec<f64> = {
            let y: Array1<f64> = read_npy(fixture("y.npy")).expect("y.npy");
            y.to_vec()
        };
        (columns, borders, cat_columns, target)
    }

    pub fn run(permutation_count: usize, expect_device: bool, label: &str) {
        let (columns, borders, cat_columns, target) = load_inputs();
        let params = ctr_params(permutation_count);
        let gpu = CountingGpu { inner: GpuBackend::default(), grown: Cell::new(0) };
        train_cat(&gpu, &columns, &borders, &cat_columns, &target, &[], &params, None)
            .unwrap_or_else(|e| panic!("[{label}] CTR train failed: {e:?}"));
        let expected = if expect_device { params.iterations } else { 0 };
        assert_eq!(
            gpu.grown.get(),
            expected,
            "[{label}] expected {expected} device grows (permutation_count={permutation_count})"
        );
        println!("[{label}] device grows = {}", gpu.grown.get());
    }
}

/// Acceptance: a single-permutation simple-Borders CTR fit COMMITS to the device.
#[test]
fn single_permutation_ctr_commits_to_device() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    device::run(1, true, "ctr-single-perm");
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    {
        let _ = ctr_params(1);
        eprintln!("SKIP single_permutation_ctr_commits_to_device: needs rocm/cuda");
    }
}

/// Acceptance scenario 3: `permutation_count > 1` + CTR still declines to CPU —
/// the `learning_folds_for_cycle == 1` guard fires (real value 3, not the old
/// hardcoded 1, GDC-01). This test must FAIL if T01 is reverted.
#[test]
fn multi_permutation_ctr_declines_to_device() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    device::run(4, false, "ctr-multi-perm");
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    {
        let _ = ctr_params(4);
        eprintln!("SKIP multi_permutation_ctr_declines_to_device: needs rocm/cuda");
    }
}
