//! FPP-13 (T11): the `bootstrap_type × grow_policy` cross-product is relaxed — the three
//! HOST-sampled bootstrap types (Bayesian / Bernoulli / MVS) are now device-eligible on
//! EVERY covered grow policy, not just SymmetricTree.
//!
//! The restriction existed because the Region and non-symmetric growers IGNORED the
//! per-object multiplier, so the backend declined those combinations rather than silently
//! dropping the sample. FPP-12 (T08) gave both growers real SPLIT-SCORING channels.
//!
//! POISSON deliberately keeps the SymmetricTree restriction and this file asserts that:
//! it is not host-sampled at all but DEVICE-resident, and only the oblivious arm opens the
//! resident sampler — admitting it elsewhere would commit a fit the session then declines.
//!
//! Observed the anti-false-pass way: a counting wrapper around the real `GpuBackend`
//! records one `grow_tree_on_device` `Some` per iteration, so a silent CPU fallback
//! records zero. The ≤1e-5 numerical consequence is `device_nonsym_bootstrap_test`'s job.
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


/// A device-eligible SAMPLED config: one of the three host-sampled bootstrap types
/// over an arbitrary grow policy. Everything else is the covered regime.
fn sampled_params(grow_policy: EGrowPolicy, bootstrap_type: EBootstrapType) -> BoostParams {
    BoostParams {
        loss: Loss::Rmse,
        iterations: 3,
        depth: 3,
        learning_rate: 0.3,
        l2_leaf_reg: 3.0,
        random_strength: 0.0,
        boost_from_average: false,
        leaf_method: LeafMethod::Gradient,
        bootstrap_type: bootstrap_type,
        subsample: 0.66,
        bagging_temperature: 1.0,
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
    }
}


#[cfg(any(feature = "rocm", feature = "cuda"))]
mod device {
    use std::cell::Cell;

    use super::{fixture, sampled_params};
    use cb_backend::GpuBackend;
    use cb_compute::{
        DeviceGrownTree, DeviceTrainConfig, EScoreFunction, FamilyTreeArgs, Loss, Runtime,
    };
    use cb_core::CbResult;
    use cb_train::{train, EBootstrapType, EGrowPolicy};

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


    /// Train a sampled fit expected to REFUSE, returning the error text.
    pub fn train_err(grow_policy: EGrowPolicy, bootstrap_type: EBootstrapType) -> String {
        let (columns, borders, target) = fixture();
        let params = sampled_params(grow_policy, bootstrap_type);
        let w = vec![1.0_f64; target.len()];
        let err = train(&GpuBackend::default(), &columns, &borders, &target, &w, &params, None)
            .expect_err("this configuration must refuse, not train");
        format!("{err}")
    }

    /// Train a sampled fit and return how many trees the DEVICE grew.
    pub fn device_grows(
        grow_policy: EGrowPolicy,
        bootstrap_type: EBootstrapType,
        label: &str,
    ) -> (usize, usize) {
        let (columns, borders, target) = fixture();
        let params = sampled_params(grow_policy, bootstrap_type);
        let w = vec![1.0_f64; target.len()];
        let gpu = CountingGpu { inner: GpuBackend::default(), grown: Cell::new(0) };
        train(&gpu, &columns, &borders, &target, &w, &params, None)
            .unwrap_or_else(|e| panic!("[{label}] sampled train failed: {e:?}"));
        let grown = gpu.grown.get();
        println!("[{label}] device grows: {grown}/{}", params.iterations);
        (grown, params.iterations)
    }
}

/// The cross-product that FPP-13 opens: three host-sampled types × the two non-symmetric
/// policies × Region. Every cell must COMMIT.
#[test]
fn host_sampled_types_commit_on_every_covered_grow_policy() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    {
        for policy in [EGrowPolicy::Depthwise, EGrowPolicy::Lossguide, EGrowPolicy::Region] {
            for bootstrap in [
                EBootstrapType::Bayesian,
                EBootstrapType::Bernoulli,
                EBootstrapType::Mvs,
            ] {
                let label = format!("{policy:?}-{bootstrap:?}");
                let (grown, iterations) = device::device_grows(policy, bootstrap, &label);
                assert_eq!(
                    grown, iterations,
                    "[{label}] a host-sampled fit must COMMIT to the device on this grow \
                     policy — zero grows means the relaxed cross-product clause silently \
                     resurrected a CPU fallback"
                );
            }
        }
    }
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    {
        let _ = sampled_params(EGrowPolicy::Depthwise, EBootstrapType::Bayesian);
        eprintln!("SKIP host_sampled_types_commit_on_every_covered_grow_policy: needs rocm/cuda");
    }
}

/// D-04: the oblivious × sampled cells that were ALREADY covered must stay covered.
#[test]
fn host_sampled_symmetric_still_commits() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    {
        for bootstrap in [
            EBootstrapType::Bayesian,
            EBootstrapType::Bernoulli,
            EBootstrapType::Mvs,
        ] {
            let label = format!("SymmetricTree-{bootstrap:?}");
            let (grown, iterations) =
                device::device_grows(EGrowPolicy::SymmetricTree, bootstrap, &label);
            assert_eq!(grown, iterations, "[{label}] the pre-existing cell must stay covered");
        }
    }
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    eprintln!("SKIP host_sampled_symmetric_still_commits: needs rocm/cuda");
}

/// Poisson stays SymmetricTree-ONLY — the deliberate non-relaxation.
#[test]
fn poisson_stays_symmetric_only() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    {
        let (grown, iterations) =
            device::device_grows(EGrowPolicy::SymmetricTree, EBootstrapType::Poisson, "poisson-sym");
        assert_eq!(grown, iterations, "Poisson on SymmetricTree must still commit");

        // …and must NOT be trainable anywhere else. Poisson is DEVICE-resident and only
        // the oblivious arm opens the resident sampler, so off SymmetricTree there is no
        // sampler at all — not on the device, and not on the CPU either (upstream rejects
        // Poisson on the CPU task type outright). The fit therefore REFUSES with a typed
        // error, which is stronger than a CPU fallback: falling back would silently train
        // an unsampled model under a config that asked for sampling.
        for policy in [EGrowPolicy::Depthwise, EGrowPolicy::Lossguide, EGrowPolicy::Region] {
            let label = format!("poisson-{policy:?}");
            let err = device::train_err(policy, EBootstrapType::Poisson);
            assert!(
                err.contains("poisson") && err.contains("SymmetricTree"),
                "[{label}] Poisson off the oblivious grow must refuse with a typed error \
                 naming the requirement, got: {err}"
            );
        }
    }
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    eprintln!("SKIP poisson_stays_symmetric_only: needs rocm/cuda");
}
