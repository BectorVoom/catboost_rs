//! GDC-05 (T07): the `device_host_eligible` weight-uniformity clause is REMOVED —
//! a fit with a genuinely non-uniform weight vector must now COMMIT to the device
//! on every covered grow policy (SymmetricTree / Depthwise / Lossguide / Region),
//! observed the anti-false-pass way: a counting wrapper around the real
//! `GpuBackend` records one `grow_tree_on_device` `Some` per iteration (a silent
//! CPU fallback records zero and fails loudly).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing, clippy::float_cmp)]

use cb_compute::{EScoreFunction, LeafMethod, Loss};
use cb_train::{BoostParams, EBootstrapType, EGrowPolicy, EOverfittingDetectorType};

/// The `device_nonsym_fit_test` clear-margin fixture: feature 0 a 32-bin ramp,
/// feature 1 a low-gain spread, step target on feature 0.
#[cfg_attr(not(any(feature = "rocm", feature = "cuda")), allow(dead_code))]
fn fixture() -> (Vec<Vec<f32>>, Vec<Vec<f64>>, Vec<f64>) {
    let n = 64usize;
    let f0: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let f1: Vec<f32> = (0..n).map(|i| (i % 7) as f32).collect();
    let borders0: Vec<f64> = (0..31).map(|k| k as f64 + 0.5).collect();
    let borders1: Vec<f64> = (0..6).map(|k| k as f64 + 0.5).collect();
    let target: Vec<f64> = (0..n).map(|i| if i <= 15 { 1.0 } else { -1.0 }).collect();
    (vec![f0, f1], vec![borders0, borders1], target)
}

/// NON-uniform per-object weights (the clause this test proves removed).
#[cfg_attr(not(any(feature = "rocm", feature = "cuda")), allow(dead_code))]
fn weights(n: usize) -> Vec<f64> {
    (0..n).map(|i| [1.0, 2.0, 1.0, 3.0][i % 4]).collect()
}

/// A device-eligible weighted config: RMSE / Plain / fold-1 / bias-0 / Gradient.
fn weighted_params(grow_policy: EGrowPolicy) -> BoostParams {
    BoostParams {
        loss: Loss::Rmse,
        iterations: 2,
        depth: 3,
        learning_rate: 0.3,
        l2_leaf_reg: 3.0,
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
        grow_policy,
        max_leaves: cb_train::max_leaves_default(),
        min_data_in_leaf: cb_train::min_data_in_leaf_default(),
        extra: Default::default(),
    }
}

#[cfg(any(feature = "rocm", feature = "cuda"))]
mod device {
    use std::cell::Cell;

    use super::{fixture, weighted_params, weights};
    use cb_backend::GpuBackend;
    use cb_compute::{
        DeviceGrownTree, DeviceTrainConfig, EScoreFunction, FamilyTreeArgs, Loss, Runtime,
    };
    use cb_core::CbResult;
    use cb_train::{train, EGrowPolicy};

    /// The `bootstrap_dev_oracle_test` counting-wrapper precedent: delegates every
    /// seam method to the real `GpuBackend`, counting committed device grows.
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

    pub fn run(grow_policy: EGrowPolicy, label: &str) {
        let (columns, borders, target) = fixture();
        let params = weighted_params(grow_policy);
        let w = weights(target.len());
        assert!(w.iter().any(|&x| x != 1.0), "[{label}] weights must be non-uniform");

        let gpu = CountingGpu { inner: GpuBackend::default(), grown: Cell::new(0) };
        train(&gpu, &columns, &borders, &target, &w, &params, None)
            .unwrap_or_else(|e| panic!("[{label}] weighted device train failed: {e:?}"));

        assert_eq!(
            gpu.grown.get(),
            params.iterations,
            "[{label}] a non-uniform-weight fit must COMMIT to the device (one device \
             grow per iteration) — zero means the removed weight clause silently \
             resurrected a CPU fallback"
        );
        println!("[{label}] weighted fit committed to device: {} grows", gpu.grown.get());
    }
}

#[test]
fn non_uniform_weights_commit_to_device_symmetric() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    device::run(EGrowPolicy::SymmetricTree, "sym-weighted");
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    {
        let _ = weighted_params(EGrowPolicy::SymmetricTree);
        eprintln!("SKIP non_uniform_weights_commit_to_device_symmetric: needs rocm/cuda");
    }
}

#[test]
fn non_uniform_weights_commit_to_device_depthwise() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    device::run(EGrowPolicy::Depthwise, "depthwise-weighted");
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    {
        let _ = weighted_params(EGrowPolicy::Depthwise);
        eprintln!("SKIP non_uniform_weights_commit_to_device_depthwise: needs rocm/cuda");
    }
}

#[test]
fn non_uniform_weights_commit_to_device_lossguide() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    device::run(EGrowPolicy::Lossguide, "lossguide-weighted");
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    {
        let _ = weighted_params(EGrowPolicy::Lossguide);
        eprintln!("SKIP non_uniform_weights_commit_to_device_lossguide: needs rocm/cuda");
    }
}

#[test]
fn non_uniform_weights_commit_to_device_region() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    device::run(EGrowPolicy::Region, "region-weighted");
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    {
        let _ = weighted_params(EGrowPolicy::Region);
        eprintln!("SKIP non_uniform_weights_commit_to_device_region: needs rocm/cuda");
    }
}
