//! End-to-end device sign-off for `bootstrap_type = Poisson`.
//!
//! The KERNEL's correctness is settled elsewhere and more strongly: `cb-backend`'s
//! `poisson_bootstrap_oracle_test` holds the `#[cube]` draw bit-for-bit against
//! `cb-oracle/fixtures/bootstrap_poisson/`, frozen by a host-compiled verbatim
//! transcription of upstream's CUDA `PoissonBootstrapImpl`. What THIS suite adds is the
//! wiring: that those weights actually reach the split histogram of a real fit, that the
//! fit is reproducible, and that λ is a live parameter rather than a constant.
//!
//! There is deliberately no "device vs CPU" comparison here, and none is possible:
//! upstream CatBoost rejects Poisson on the CPU task type, so this repo's CPU grower
//! refuses it too. That refusal is itself asserted (in `device_bootstrap_parity_test`),
//! and it is why the anti-false-pass burden falls entirely on the checks below.
//!
//! # Anti-false-pass discipline
//!
//! A Poisson suite could pass while proving nothing in four ways, all closed here:
//!
//! 1. **The fit silently ran on the CPU grower.** Impossible — the CPU grower errors on
//!    Poisson — but a *silently unsampled device* fit is possible, so [`CountingGpu`]
//!    asserts every tree came back from `grow_tree_on_device`.
//! 2. **The sample was drawn and then dropped before the histogram.** Closed by
//!    requiring the Poisson fit to differ MATERIALLY from `bootstrap_type = No`.
//! 3. **λ never reached the kernel** (a hard-coded rate would still "sample"). Closed by
//!    requiring two different `subsample` values to produce different models.
//! 4. **Sampling destroyed the model** — "different" is not "correct". Closed by
//!    requiring the sampled fit to still beat the constant predictor by a wide margin.
//!
//! Runs on the REAL device only (rocm / cuda); `GpuBackend` is not compiled under the
//! `cpu` feature, so cpu/wgpu print a SKIP line instead of passing silently.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp
)]

use cb_compute::{EScoreFunction, LeafMethod, Loss};
use cb_train::{BoostParams, EBootstrapType, EGrowPolicy, EOverfittingDetectorType};

/// A non-separable regression fixture: `n` objects × `n_features` quantized ramps
/// (32 bins), target a smooth nonlinear combination no single tree fits exactly.
#[cfg_attr(not(any(feature = "rocm", feature = "cuda")), allow(dead_code))]
fn fixture(n: usize, n_features: usize) -> (Vec<Vec<f32>>, Vec<Vec<f64>>, Vec<f64>) {
    let mut columns = Vec::with_capacity(n_features);
    let mut borders = Vec::with_capacity(n_features);
    for f in 0..n_features {
        let col: Vec<f32> = (0..n).map(|i| ((i * 7 + f * 13) % 32) as f32).collect();
        columns.push(col);
        borders.push((0..31).map(|k| k as f64 + 0.5).collect::<Vec<f64>>());
    }
    let target: Vec<f64> = (0..n)
        .map(|i| {
            let a = f64::from(columns[0][i]);
            let b = f64::from(columns[1 % n_features][i]);
            (a * 0.31).sin() + (b * 0.17).cos() * 0.5 + ((i % 11) as f64) * 0.05
        })
        .collect();
    (columns, borders, target)
}

/// Device-eligible OBLIVIOUS params (bias 0, unit weights, `random_strength = 0`,
/// Gradient leaves, SymmetricTree — every `device_host_eligible` precondition).
#[cfg_attr(not(any(feature = "rocm", feature = "cuda")), allow(dead_code))]
fn params_with(
    bootstrap_type: EBootstrapType,
    subsample: f64,
    iterations: usize,
    random_seed: u64,
) -> BoostParams {
    BoostParams {
        loss: Loss::Rmse,
        iterations,
        depth: 6,
        learning_rate: 0.3,
        l2_leaf_reg: 3.0,
        random_strength: 0.0,
        boost_from_average: false,
        leaf_method: LeafMethod::Gradient,
        bootstrap_type,
        subsample,
        bagging_temperature: 0.0,
        random_seed,
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
    }
}

#[cfg(any(feature = "rocm", feature = "cuda"))]
mod device {
    use super::{fixture, params_with};
    use cb_backend::GpuBackend;
    use cb_compute::{
        FamilyTreeArgs,
        DeviceGrownTree, DeviceTrainConfig, Derivatives, EScoreFunction, Loss, Runtime,
    };
    use cb_core::CbResult;
    use cb_model::Model as CbModel;
    use cb_train::{train, BoostParams, EBootstrapType};
    use std::cell::Cell;

    /// The real [`GpuBackend`], instrumented to PROVE the device grow ran and to record
    /// the sample length the seam received (0 for Poisson — the draw is device-resident).
    struct CountingGpu {
        inner: GpuBackend,
        grown: Cell<usize>,
        begun: Cell<usize>,
        last_sample_len: Cell<usize>,
    }

    impl CountingGpu {
        fn new() -> Self {
            Self {
                inner: GpuBackend::default(),
                grown: Cell::new(0),
                begun: Cell::new(0),
                last_sample_len: Cell::new(usize::MAX),
            }
        }
    }

    impl Runtime for CountingGpu {
        fn compute_gradients(
            &self,
            loss: &Loss,
            approx: &[f64],
            target: &[f64],
            approx_dimension: usize,
        ) -> CbResult<Derivatives> {
            self.inner.compute_gradients(loss, approx, target, approx_dimension)
        }

        #[allow(clippy::too_many_arguments)]
        fn begin_device_training(
            &self,
            loss: &Loss,
            depth: usize,
            boosting_type_is_plain: bool,
            fold_count: usize,
            score_function: EScoreFunction,
            bins_feature_major: &[u32],
            weight: &[f64],
            n: usize,
            n_features: usize,
            n_bins: usize,
            learning_rate: f64,
            scaled_l2: f64,
            config: &DeviceTrainConfig,
        ) -> CbResult<bool> {
            let accepted = self.inner.begin_device_training(
                loss,
                depth,
                boosting_type_is_plain,
                fold_count,
                score_function,
                bins_feature_major,
                weight,
                n,
                n_features,
                n_bins,
                learning_rate,
                scaled_l2,
                config,
            )?;
            if accepted {
                self.begun.set(self.begun.get() + 1);
            }
            Ok(accepted)
        }

        fn grow_tree_on_device(
            &self,
            approx: &[f64],
            target: &[f64],
            sample: &[f64],
        family: Option<&FamilyTreeArgs<'_>>,
        ) -> CbResult<Option<DeviceGrownTree>> {
            self.last_sample_len.set(sample.len());
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

    const N: usize = 8192;
    const NF: usize = 8;
    const ITERS: usize = 12;

    /// Train on the device and return the in-sample predictions, asserting the device
    /// actually grew every tree.
    fn fit(label: &str, params: &BoostParams) -> Vec<f64> {
        let (columns, borders, target) = fixture(N, NF);
        let gpu = CountingGpu::new();
        let model = train(&gpu, &columns, &borders, &target, &[], params, None)
            .unwrap_or_else(|e| panic!("[{label}] device train failed: {e:?}"));
        assert_eq!(gpu.begun.get(), 1, "[{label}] the device session must be accepted");
        assert_eq!(
            gpu.grown.get(),
            params.iterations,
            "[{label}] every tree must be grown on device"
        );
        if params.bootstrap_type == EBootstrapType::Poisson {
            assert_eq!(
                gpu.last_sample_len.get(),
                0,
                "[{label}] Poisson draws device-resident; the host must pass an EMPTY sample"
            );
        }
        cb_model::predict_raw(&CbModel::from_trained(&model, borders.clone()), &columns)
    }

    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(&x, &y)| (x - y).abs()).fold(0.0, f64::max)
    }

    fn rmse(pred: &[f64], target: &[f64]) -> f64 {
        (pred.iter().zip(target).map(|(&p, &t)| (p - t) * (p - t)).sum::<f64>()
            / pred.len() as f64)
            .sqrt()
    }

    /// The Poisson sample must REACH the split histogram: a Poisson fit and an
    /// unsampled fit on identical data must differ materially. If the weights were
    /// drawn and then dropped, these two would be bit-identical.
    pub fn poisson_sampling_changes_the_model() {
        let sampled = fit("poisson", &params_with(EBootstrapType::Poisson, 0.8, ITERS, 42));
        let plain = fit("no", &params_with(EBootstrapType::No, 1.0, ITERS, 42));
        let delta = max_abs_diff(&sampled, &plain);
        assert!(
            delta > 1e-3,
            "Poisson sampling changed the model by only {delta:.3e}; the device weights are \
             not reaching the split histogram"
        );
        println!("[poisson e2e] sampled vs unsampled max|dpred| = {delta:.3e}");
    }

    /// λ is a LIVE parameter. `GetPoissonLambda() = -log(1 - subsample)`, so a different
    /// `subsample` must yield a different draw and a different model. A kernel that
    /// ignored λ (or hard-coded Poisson(1)) would produce identical models here.
    pub fn subsample_changes_the_poisson_draw() {
        let a = fit("poisson-0.5", &params_with(EBootstrapType::Poisson, 0.5, ITERS, 42));
        let b = fit("poisson-0.9", &params_with(EBootstrapType::Poisson, 0.9, ITERS, 42));
        let delta = max_abs_diff(&a, &b);
        assert!(
            delta > 1e-3,
            "subsample 0.5 and 0.9 produced models differing by only {delta:.3e}; lambda is \
             not reaching the kernel"
        );
        println!("[poisson e2e] subsample 0.5 vs 0.9 max|dpred| = {delta:.3e}");
    }

    /// The seed is a LIVE parameter too: a different `random_seed` seeds a different
    /// device seed buffer and must move the model.
    pub fn seed_changes_the_poisson_draw() {
        let a = fit("poisson-seed1", &params_with(EBootstrapType::Poisson, 0.8, ITERS, 1));
        let b = fit("poisson-seed2", &params_with(EBootstrapType::Poisson, 0.8, ITERS, 2));
        let delta = max_abs_diff(&a, &b);
        assert!(
            delta > 1e-4,
            "two random seeds produced models differing by only {delta:.3e}; the seed buffer \
             is not derived from random_seed"
        );
        println!("[poisson e2e] seed 1 vs 2 max|dpred| = {delta:.3e}");
    }

    /// Run-to-run determinism: identical params must give bit-identical predictions.
    /// The seed buffer is host-derived and the kernel's per-thread streams are
    /// independent, so there is no atomics-ordering nondeterminism to absorb — the
    /// budget here is EXACT equality, tighter than WR01-S13's ≤1e-7.
    pub fn poisson_is_run_to_run_deterministic() {
        let params = params_with(EBootstrapType::Poisson, 0.8, ITERS, 7);
        let a = fit("poisson-run1", &params);
        let b = fit("poisson-run2", &params);
        assert_eq!(a, b, "Poisson device fit is not bit-reproducible across runs");
        println!("[poisson e2e] run-to-run max|dpred| = 0 (bit-identical)");
    }

    /// "Different" is not "correct": the Poisson-sampled model must still LEARN, and
    /// its trees must be non-degenerate. A sampler that zeroed every weight would
    /// produce a flat model that trivially differs from the unsampled one.
    pub fn poisson_model_still_learns() {
        let (_columns, _borders, target) = fixture(N, NF);
        let pred = fit("poisson-quality", &params_with(EBootstrapType::Poisson, 0.8, ITERS, 42));
        let mean = target.iter().sum::<f64>() / target.len() as f64;
        let baseline: Vec<f64> = vec![mean; target.len()];
        let sampled_rmse = rmse(&pred, &target);
        let baseline_rmse = rmse(&baseline, &target);
        assert!(
            sampled_rmse < baseline_rmse * 0.5,
            "Poisson-sampled RMSE {sampled_rmse:.4} is not materially better than the \
             constant predictor {baseline_rmse:.4}; the sample is degenerate"
        );
        // A flat model would have zero prediction spread.
        let spread = pred.iter().cloned().fold(f64::MIN, f64::max)
            - pred.iter().cloned().fold(f64::MAX, f64::min);
        assert!(spread > 0.1, "Poisson model predictions are nearly constant (spread {spread:.3e})");
        println!(
            "[poisson e2e] rmse {sampled_rmse:.4} vs constant-predictor {baseline_rmse:.4} \
             (spread {spread:.3})"
        );
    }

    /// `subsample >= 1` makes upstream's `GetPoissonLambda()` return `-1`, which zeroes
    /// EVERY sample weight. That must fail loudly, not train a model on nothing.
    pub fn poisson_rejects_degenerate_subsample() {
        let (columns, borders, target) = fixture(1024, 4);
        let params = params_with(EBootstrapType::Poisson, 1.0, 3, 42);
        let err = train(&CountingGpu::new(), &columns, &borders, &target, &[], &params, None)
            .expect_err("subsample = 1.0 with Poisson must be rejected, not trained on zeros");
        let msg = err.to_string();
        assert!(msg.contains("subsample"), "unexpected error text: {msg}");
        println!("[poisson e2e] degenerate subsample rejected: {msg}");
    }
}

#[cfg(any(feature = "rocm", feature = "cuda"))]
#[test]
fn poisson_sampling_changes_the_model() {
    device::poisson_sampling_changes_the_model();
}

#[cfg(any(feature = "rocm", feature = "cuda"))]
#[test]
fn poisson_subsample_changes_the_draw() {
    device::subsample_changes_the_poisson_draw();
}

#[cfg(any(feature = "rocm", feature = "cuda"))]
#[test]
fn poisson_seed_changes_the_draw() {
    device::seed_changes_the_poisson_draw();
}

#[cfg(any(feature = "rocm", feature = "cuda"))]
#[test]
fn poisson_is_run_to_run_deterministic() {
    device::poisson_is_run_to_run_deterministic();
}

#[cfg(any(feature = "rocm", feature = "cuda"))]
#[test]
fn poisson_model_still_learns() {
    device::poisson_model_still_learns();
}

#[cfg(any(feature = "rocm", feature = "cuda"))]
#[test]
fn poisson_rejects_degenerate_subsample() {
    device::poisson_rejects_degenerate_subsample();
}

#[cfg(not(any(feature = "rocm", feature = "cuda")))]
#[test]
fn device_poisson_bootstrap_skipped_without_a_device() {
    // Anti-false-pass: `GpuBackend` is not compiled under the `cpu`/`wgpu` features, and
    // Poisson has no CPU path at all, so there is nothing to assert here. Print rather
    // than silently pass, so a cpu-feature run cannot be mistaken for device evidence.
    println!("SKIP device_poisson_bootstrap: needs rocm/cuda");
}
