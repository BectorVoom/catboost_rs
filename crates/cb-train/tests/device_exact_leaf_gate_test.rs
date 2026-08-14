//! FPP-06 (T10): `device_host_eligible` admits `LeafMethod::Exact` for the {Mae,
//! Quantile} intersection, and STILL declines it for every other loss.
//!
//! Both directions matter and both are asserted. Admitting too much is the dangerous
//! direction: a gate that opened for LogCosh would apply the Gradient `calc_average` leaf
//! to a fit whose leaf is an order statistic — wrong, and strictly worse than today's
//! correct CPU fallback.
//!
//! Observed the anti-false-pass way: a counting wrapper around the real `GpuBackend`
//! records one `grow_tree_on_device` `Some` per iteration, so a silent CPU fallback
//! records zero. The ≤1e-5 numerical consequence is `device_exact_leaf_fit_test`'s job.
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


/// A device-eligible config requesting the EXACT order-statistic leaf over `loss`.\n/// Everything else is the covered regime: Plain / fold-1 / no sampling / bias 0.
fn exact_params(loss: Loss) -> BoostParams {
    BoostParams {
        loss: loss,
        iterations: 2,
        depth: 3,
        learning_rate: 0.3,
        l2_leaf_reg: 3.0,
        random_strength: 0.0,
        boost_from_average: false,
        leaf_method: LeafMethod::Exact,
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
        grow_policy: EGrowPolicy::SymmetricTree,
        max_leaves: cb_train::max_leaves_default(),
        min_data_in_leaf: cb_train::min_data_in_leaf_default(),
        extra: Default::default(),
    }
}


#[cfg(any(feature = "rocm", feature = "cuda"))]
mod device {
    use std::cell::Cell;

    use super::{exact_params, fixture};
    use cb_backend::GpuBackend;
    use cb_compute::{
        DeviceGrownTree, DeviceTrainConfig, EScoreFunction, FamilyTreeArgs, Loss, Runtime,
    };
    use cb_core::CbResult;
    use cb_train::train;

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


    /// Train an Exact-leaf fit over `loss` and return how many trees the DEVICE grew.
    pub fn device_grows(loss: Loss, label: &str) -> usize {
        let (columns, borders, target) = fixture();
        let params = exact_params(loss);
        let w = vec![1.0_f64; target.len()];
        let gpu = CountingGpu { inner: GpuBackend::default(), grown: Cell::new(0) };
        train(&gpu, &columns, &borders, &target, &w, &params, None)
            .unwrap_or_else(|e| panic!("[{label}] exact-leaf train failed: {e:?}"));
        let grown = gpu.grown.get();
        println!("[{label}] exact-leaf fit device grows: {grown}/{}", params.iterations);
        grown
    }

    /// Same, but with an explicit leaf method — so the Gradient D-04 regression can be
    /// asserted as a real device commit rather than as a property of the params builder.
    pub fn device_grows_with(
        loss: Loss,
        leaf_method: cb_compute::LeafMethod,
        label: &str,
    ) -> usize {
        let (columns, borders, target) = fixture();
        let mut params = exact_params(loss);
        params.leaf_method = leaf_method;
        let w = vec![1.0_f64; target.len()];
        let gpu = CountingGpu { inner: GpuBackend::default(), grown: Cell::new(0) };
        train(&gpu, &columns, &borders, &target, &w, &params, None)
            .unwrap_or_else(|e| panic!("[{label}] train failed: {e:?}"));
        let grown = gpu.grown.get();
        println!("[{label}] device grows: {grown}/{}", params.iterations);
        grown
    }

    pub fn iterations() -> usize {
        exact_params(Loss::Mae).iterations
    }
}

#[test]
fn exact_leaf_mae_commits_to_device() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    {
        let grown = device::device_grows(Loss::Mae, "exact-mae");
        assert_eq!(
            grown,
            device::iterations(),
            "Exact/MAE is in the admitted intersection and must COMMIT to the device — \
             zero grows means the relaxed leaf-method clause silently resurrected a CPU \
             fallback"
        );
    }
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    eprintln!("SKIP exact_leaf_mae_commits_to_device: needs rocm/cuda");
}

#[test]
fn exact_leaf_quantile_commits_to_device() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    {
        let loss = Loss::Quantile { alpha: 0.7, delta: 1e-6 };
        let grown = device::device_grows(loss, "exact-quantile07");
        assert_eq!(
            grown,
            device::iterations(),
            "Exact/Quantile is in the admitted intersection and must COMMIT to the device"
        );
    }
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    eprintln!("SKIP exact_leaf_quantile_commits_to_device: needs rocm/cuda");
}

#[test]
fn exact_leaf_logcosh_cannot_reach_the_device_at_all() {
    // THE DANGEROUS DIRECTION, asserted honestly.
    //
    // LogCosh is CPU-LEGAL under Exact (`validate_leaf_method` admits it) but
    // device-UNCOVERED (`map_leaf_method` has no LogCosh arm), so a gate that admitted it
    // would apply the Gradient `calc_average` leaf to an order-statistic fit.
    //
    // It turns out LogCosh is stopped even EARLIER than the gate: `GpuBackend` has no
    // LogCosh derivative kernel at all, so `compute_gradients` refuses the fit with a
    // typed `OutOfRange` before any grow decision is made. That is a *stronger* guarantee
    // than "the gate declines", and this test pins it — including the fact that it is a
    // clean typed refusal naming the loss, not a silent wrong-leaf device fit.
    //
    // The gate-level decision for LogCosh is separately pinned, device-free, by
    // `device_exact_leaf_config_test::exact_logcosh_declines_because_the_device_does_not_cover_it`.
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    {
        use cb_backend::GpuBackend;
        use cb_train::train;

        let (columns, borders, target) = fixture();
        let params = exact_params(Loss::LogCosh);
        let w = vec![1.0_f64; target.len()];
        let err = train(&GpuBackend::default(), &columns, &borders, &target, &w, &params, None)
            .expect_err("LogCosh has no GPU derivative kernel; the fit must refuse");
        let msg = format!("{err}");
        assert!(
            msg.contains("LogCosh"),
            "the refusal must name the unsupported loss, got: {msg}"
        );
        println!("[exact-logcosh] refused before any grow: {msg}");
    }
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    eprintln!("SKIP exact_leaf_logcosh_cannot_reach_the_device_at_all: needs rocm/cuda");
}

#[test]
fn gradient_leaf_still_commits_to_device() {
    // D-04: the overwhelmingly common Gradient path must be untouched by the relaxation.
    // Asserted as a real device commit, not as a property of the params builder.
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    {
        let grown = device::device_grows_with(
            Loss::Rmse,
            cb_compute::LeafMethod::Gradient,
            "gradient-rmse",
        );
        assert_eq!(
            grown,
            device::iterations(),
            "a Gradient/RMSE fit must still commit to the device after the exact-leaf \
             relaxation — the common path is unchanged (D-04)"
        );
    }
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    eprintln!("SKIP gradient_leaf_still_commits_to_device: needs rocm/cuda");
}
