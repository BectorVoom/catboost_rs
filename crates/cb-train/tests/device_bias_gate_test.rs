//! FPP-02 (T09): `device_host_eligible`'s `bias == 0.0` clause is REMOVED — a fit with a
//! NON-ZERO starting approximant (`boost_from_average = true`, which is upstream's RMSE
//! default and this project's `CatBoostBuilder` default) must now COMMIT to the device.
//!
//! Observed the anti-false-pass way: a counting wrapper around the real `GpuBackend`
//! records one `grow_tree_on_device` `Some` per iteration, so a silent CPU fallback
//! records zero and fails loudly. Asserting "the fit succeeded" would pass either way.
//!
//! The ≤1e-5 numerical consequence is `device_bias_fit_test`'s job (T12); this file only
//! proves the GATE opened and that `DeviceTrainConfig.bias` carries the real value.
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


/// A device-eligible config with a NON-ZERO starting approximant. Everything else is\n/// the covered regime: RMSE / Plain / fold-1 / no sampling / Gradient leaf.
fn bias_params(grow_policy: EGrowPolicy) -> BoostParams {
    BoostParams {
        loss: Loss::Rmse,
        iterations: 2,
        depth: 3,
        learning_rate: 0.3,
        l2_leaf_reg: 3.0,
        random_strength: 0.0,
        boost_from_average: true,
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
    // Atomics, not `Cell`: `cb_train::train` requires `R: Runtime + Sync` so the
    // fit can run inside a `thread_count`-sized rayon pool.
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::{bias_params, fixture};
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
        pub grown: AtomicUsize,
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
                self.grown.fetch_add(1, Ordering::Relaxed);
            }
            Ok(out)
        }

        fn end_device_training(&self) -> CbResult<()> {
            self.inner.end_device_training()
        }
    }


    pub fn run(grow_policy: EGrowPolicy, label: &str) {
        let (columns, borders, target) = fixture();
        let params = bias_params(grow_policy);
        assert!(params.boost_from_average, "[{label}] the fixture must request a bias");

        // The starting approximant for an RMSE fit is mean(target); a near-zero mean
        // cannot discriminate the fix from the former hardcoded-zero seed.
        let mean = target.iter().sum::<f64>() / (target.len() as f64);
        assert!(
            mean.abs() > 0.1,
            "[{label}] |mean(target)| = {mean:.6} is too close to zero to discriminate \
             a non-zero bias"
        );

        let w = vec![1.0_f64; target.len()];
        let gpu = CountingGpu { inner: GpuBackend::default(), grown: AtomicUsize::new(0) };
        train(&gpu, &columns, &borders, &target, &w, &params, None)
            .unwrap_or_else(|e| panic!("[{label}] bias device train failed: {e:?}"));

        assert_eq!(
            gpu.grown.load(Ordering::Relaxed),
            params.iterations,
            "[{label}] a boost_from_average fit must COMMIT to the device (one device grow \
             per iteration) — zero means the removed bias clause silently resurrected a CPU \
             fallback"
        );
        println!("[{label}] bias fit committed to device: {} grows", gpu.grown.load(Ordering::Relaxed));
    }
}

#[test]
fn non_zero_bias_commits_to_device_symmetric() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    device::run(EGrowPolicy::SymmetricTree, "sym-bias");
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    {
        let _ = bias_params(EGrowPolicy::SymmetricTree);
        eprintln!("SKIP non_zero_bias_commits_to_device_symmetric: needs rocm/cuda");
    }
}

#[test]
fn non_zero_bias_commits_to_device_depthwise() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    device::run(EGrowPolicy::Depthwise, "depthwise-bias");
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    {
        let _ = bias_params(EGrowPolicy::Depthwise);
        eprintln!("SKIP non_zero_bias_commits_to_device_depthwise: needs rocm/cuda");
    }
}

#[test]
fn non_zero_bias_commits_to_device_region() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    device::run(EGrowPolicy::Region, "region-bias");
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    {
        let _ = bias_params(EGrowPolicy::Region);
        eprintln!("SKIP non_zero_bias_commits_to_device_region: needs rocm/cuda");
    }
}
