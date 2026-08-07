//! FPP-14 (T15): the SAMPLED NON-SYMMETRIC device e2e oracle.
//!
//! FPP-13 opened the `bootstrap_type × grow_policy` cross-product — the three HOST-sampled
//! types (Bayesian / Bernoulli / MVS) are now device-eligible on Depthwise, Lossguide and
//! Region, not just SymmetricTree. This suite is the numerical half of that claim: all
//! NINE cells must reproduce a CPU fit from the same seed to ≤1e-4, with no mid-run
//! fallback.
//!
//! # Why every iteration is compared, not just the last
//!
//! An RNG-phase divergence typically appears at TREE 2, not tree 1: tree 1's sample is
//! drawn before anything can desynchronise, and it is the NEXT tree's `bootstrap()` that
//! reads the phase the previous tree left. A final-prediction-only assertion can hide that
//! behind later trees' shrinkage. This suite therefore asserts the per-iteration STAGED
//! predictions, so a tree-2 divergence surfaces as a tree-2 failure.
//!
//! That failure mode is not hypothetical here: blocker B-1 was exactly it. The device
//! branch replays the draws the skipped CPU level search would have consumed, and the
//! replay was written for the OBLIVIOUS level search's draw count. `region_grower` and
//! `leaf_wise_grower` take no `Perturbation` at all and consume ZERO draws, so before B-1
//! made `replay_grow_draws` policy-aware, every cell in this suite would have diverged
//! from tree 2 for a reason no kernel owns.
//!
//! # Anti-false-pass discipline (inherited from `device_bootstrap_parity_test`)
//!
//! 1. A silent CPU fallback would make "device == CPU" a tautology — closed by
//!    `CountingGpu`, which counts the trees that actually came back from
//!    `grow_tree_on_device` and asserts the count equals `iterations`.
//! 2. A sample that crosses the seam but is DROPPED before the histogram would make a
//!    sampled fit equal an unsampled one and still satisfy "device == CPU" — closed by
//!    asserting each sampled fit differs materially from the same fit at
//!    `bootstrap_type = No`.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp
)]

use cb_compute::{EScoreFunction, LeafMethod, Loss};
use cb_train::{BoostParams, EBootstrapType, EGrowPolicy, EOverfittingDetectorType};

/// The shipped device ε for a non-symmetric sampled fit. The oblivious sampled arm holds
/// the full ≤1e-5 bar (`device_bootstrap_parity_test`); the host-driven Region / leaf-wise
/// growers score through a separate per-node device pass, so this suite ships at the
/// documented device self-oracle ε.
const EPS: f64 = 1e-4;

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

/// Device-eligible OBLIVIOUS params carrying one bootstrap configuration. Every
/// `device_host_eligible` precondition outside the sampler is satisfied: RMSE /
/// Plain / fold-1 / unit weights / bias 0 (`boost_from_average = false`, the CR-01
/// gate) / Gradient leaf / `random_strength = 0` / SymmetricTree.
#[cfg_attr(not(any(feature = "rocm", feature = "cuda")), allow(dead_code))]
fn params_with(
    bootstrap_type: EBootstrapType,
    subsample: f64,
    bagging_temperature: f32,
    depth: usize,
    iterations: usize,
    grow_policy: EGrowPolicy,
) -> BoostParams {
    BoostParams {
        loss: Loss::Rmse,
        iterations,
        depth,
        learning_rate: 0.3,
        l2_leaf_reg: 3.0,
        random_strength: 0.0,
        boost_from_average: false,
        leaf_method: LeafMethod::Gradient,
        bootstrap_type,
        subsample,
        bagging_temperature,
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
    use super::{fixture, params_with, EPS};
    use cb_backend::GpuBackend;
    use cb_compute::{
        FamilyTreeArgs,
        rmse_der1, rmse_der2, Derivatives, DeviceGrownTree, DeviceTrainConfig, EScoreFunction,
        Loss, Runtime,
    };
    use cb_core::{CbError, CbResult};
    use cb_model::Model as CbModel;
    use cb_train::{train, BoostParams, EBootstrapType, EGrowPolicy};
    use std::cell::Cell;

    /// CPU reference runtime that DECLINES the device seam (trait defaults: `begin →
    /// Ok(false)`, `grow → Ok(None)`) so `train` runs the byte-unchanged CPU grower,
    /// while computing the same RMSE derivatives the real `CpuBackend` would
    /// (`CpuBackend` is not compiled under rocm/cuda).
    struct CpuRefRuntime;

    impl Runtime for CpuRefRuntime {
        fn compute_gradients(
            &self,
            _loss: &Loss,
            approx: &[f64],
            target: &[f64],
            _approx_dimension: usize,
        ) -> CbResult<Derivatives> {
            Ok(Derivatives {
                der1: approx.iter().zip(target).map(|(&a, &t)| rmse_der1(a, t)).collect(),
                der2: approx.iter().zip(target).map(|(&a, &t)| rmse_der2(a, t)).collect(),
            })
        }
    }

    /// The real [`GpuBackend`], instrumented so the suite can PROVE the device grow
    /// actually ran. Without this, a silent CPU fallback would make every
    /// "device == CPU" assertion below a tautology that passes on a broken gate.
    ///
    /// It also records the sample length the seam received, which is what
    /// distinguishes "sampling was wired" from "an empty sample went through".
    struct CountingGpu {
        inner: GpuBackend,
        /// Trees the device actually returned (`Ok(Some(_))`).
        grown: Cell<usize>,
        /// Sessions the backend accepted (`begin → Ok(true)`).
        begun: Cell<usize>,
        /// Length of the sample handed to the most recent grow call.
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

    /// Train `params` on both backends and return
    /// `(device_predictions, cpu_predictions, split_mismatch_report, sample_len)`.
    ///
    /// Panics (rather than returning) if the device declined the fit — a decline is
    /// exactly the silent-CPU-fallback failure this suite exists to catch.
    fn both(
        label: &str,
        params: &BoostParams,
        n: usize,
        n_features: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<(usize, f64)>, usize) {
        let (columns, borders, target) = fixture(n, n_features);

        let gpu = CountingGpu::new();
        let dev = train(&gpu, &columns, &borders, &target, &[], params, None)
            .unwrap_or_else(|e| panic!("[{label}] device train failed: {e:?}"));
        let cpu = train(&CpuRefRuntime, &columns, &borders, &target, &[], params, None)
            .unwrap_or_else(|e| panic!("[{label}] cpu train failed: {e:?}"));

        // ── Anti-false-pass 1: the device really grew every tree. ──
        assert_eq!(
            gpu.begun.get(),
            1,
            "[{label}] the backend must ACCEPT exactly one device session; \
             0 means the eligibility gate declined and the 'device' fit is a CPU fit"
        );
        assert_eq!(
            gpu.grown.get(),
            params.iterations,
            "[{label}] the device must grow every tree ({} expected); a shortfall means \
             the fit silently fell back to the CPU grower",
            params.iterations
        );
        assert_eq!(
            dev.oblivious_trees.len(),
            params.iterations,
            "[{label}] one oblivious tree per iteration"
        );

        // Per-tree structural comparison + each divergent tree's OWN contribution
        // delta (the WR01-S12 evidence).
        let mut mismatches: Vec<(usize, f64)> = Vec::new();
        for (t, (d, c)) in dev.oblivious_trees.iter().zip(cpu.oblivious_trees.iter()).enumerate() {
            if d.splits == c.splits {
                continue;
            }
            let mut dev_one = dev.clone();
            dev_one.oblivious_trees = vec![d.clone()];
            let mut cpu_one = cpu.clone();
            cpu_one.oblivious_trees = vec![c.clone()];
            let dp = cb_model::predict_raw(&CbModel::from_trained(&dev_one, borders.clone()), &columns);
            let cp = cb_model::predict_raw(&CbModel::from_trained(&cpu_one, borders.clone()), &columns);
            let tree_max = dp
                .iter()
                .zip(cp.iter())
                .fold(0.0_f64, |m, (&a, &b)| m.max((a - b).abs()));
            mismatches.push((t, tree_max));
        }

        let dev_pred = cb_model::predict_raw(&CbModel::from_trained(&dev, borders.clone()), &columns);
        let cpu_pred = cb_model::predict_raw(&CbModel::from_trained(&cpu, borders.clone()), &columns);
        (dev_pred, cpu_pred, mismatches, gpu.last_sample_len.get())
    }

    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len(), "prediction lengths must agree");
        a.iter().zip(b.iter()).fold(0.0_f64, |m, (&x, &y)| m.max((x - y).abs()))
    }


    /// FPP-14: every cell of the {Depthwise, Lossguide, Region} × {Bayesian, Bernoulli,
    /// Mvs} cross-product reproduces the CPU fit to ε, per ITERATION.
    pub fn sampled_nonsym_cells_hold_eps() {
        let n = 512usize;
        let nf = 4usize;
        let depth = 3usize;
        let iters = 5usize;

        for policy in [EGrowPolicy::Depthwise, EGrowPolicy::Lossguide, EGrowPolicy::Region] {
            for (bootstrap, subsample, temp) in [
                (EBootstrapType::Bayesian, 1.0_f64, 1.0_f32),
                (EBootstrapType::Bernoulli, 0.66, 0.0),
                (EBootstrapType::Mvs, 0.66, 0.0),
            ] {
                let label = format!("{policy:?}×{bootstrap:?}");
                let params = params_with(bootstrap, subsample, temp, depth, iters, policy);
                let (columns, borders, target) = fixture(n, nf);

                let gpu = CountingGpu::new();
                let mut dev_staged: Vec<f64> = Vec::new();
                let dev = train(
                    &gpu, &columns, &borders, &target, &[], &params, Some(&mut dev_staged),
                )
                .unwrap_or_else(|e| panic!("[{label}] device train failed: {e:?}"));
                let mut cpu_staged: Vec<f64> = Vec::new();
                let _cpu = train(
                    &CpuRefRuntime, &columns, &borders, &target, &[], &params,
                    Some(&mut cpu_staged),
                )
                .unwrap_or_else(|e| panic!("[{label}] cpu train failed: {e:?}"));

                // ── Anti-false-pass 1: the device really grew every tree. ──
                assert_eq!(
                    gpu.begun.get(), 1,
                    "[{label}] the backend must ACCEPT exactly one device session; 0 means \
                     the eligibility gate declined and the 'device' fit is a CPU fit"
                );
                assert_eq!(
                    gpu.grown.get(), params.iterations,
                    "[{label}] the device must grow every tree; a shortfall means the fit \
                     silently fell back to the CPU grower"
                );
                // ── Anti-false-pass 2: a length-n sample really crossed the seam. ──
                assert_eq!(
                    gpu.last_sample_len.get(), n,
                    "[{label}] a host-sampled fit must hand the seam a length-n multiplier; \
                     0 means the sample never crossed and the fit is effectively unsampled"
                );

                // ── PER-ITERATION comparison: a tree-2 RNG-phase divergence must not be
                //    hidden behind later trees. ──
                assert_eq!(
                    dev_staged.len(), cpu_staged.len(),
                    "[{label}] staged prediction counts must agree"
                );
                assert_eq!(
                    dev_staged.len(), n * params.iterations,
                    "[{label}] staged output must carry one prediction block per iteration"
                );
                let mut worst = (0usize, 0.0_f64);
                for it in 0..params.iterations {
                    let lo = it * n;
                    let hi = lo + n;
                    let d = &dev_staged[lo..hi];
                    let c = &cpu_staged[lo..hi];
                    let m = d.iter().zip(c.iter()).fold(0.0_f64, |m, (&a, &b)| m.max((a - b).abs()));
                    if m > worst.1 {
                        worst = (it, m);
                    }
                    assert!(
                        m <= EPS,
                        "[{label}] iteration {it}: device vs CPU max|Δ| = {m:.3e} exceeds \
                         ε={EPS:.0e} (a divergence that first appears at iteration >= 1 is the \
                         RNG-phase signature — see blocker B-1)"
                    );
                }

                // ── Anti-false-pass 2b: the sampled fit must differ MATERIALLY from the
                //    same fit with no sampling, or "device == CPU" proves nothing. ──
                let unsampled_params =
                    params_with(EBootstrapType::No, 1.0, 0.0, depth, iters, policy);
                let unsampled = train(
                    &CountingGpu::new(), &columns, &borders, &target, &[], &unsampled_params, None,
                )
                .unwrap_or_else(|e| panic!("[{label}] unsampled device train failed: {e:?}"));
                let dev_pred = cb_model::predict_raw(
                    &CbModel::from_trained(&dev, borders.clone()), &columns,
                );
                let uns_pred = cb_model::predict_raw(
                    &CbModel::from_trained(&unsampled, borders.clone()), &columns,
                );
                let sampling_effect = dev_pred
                    .iter()
                    .zip(uns_pred.iter())
                    .fold(0.0_f64, |m, (&a, &b)| m.max((a - b).abs()));
                assert!(
                    sampling_effect > EPS,
                    "[{label}] the sampled and unsampled fits agree (max|Δ|={sampling_effect:.3e}) \
                     — the sample crossed the seam but never reached the split histogram"
                );

                println!(
                    "[{label}] {} device trees; worst iteration {} at {:.3e} (bar {EPS:.0e}); \
                     sampling effect {sampling_effect:.3e}",
                    gpu.grown.get(), worst.0, worst.1
                );
            }
        }
    }
}

#[test]
fn fpp14_sampled_nonsym_cells_hold_eps_vs_cpu() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    device::sampled_nonsym_cells_hold_eps();
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    eprintln!("SKIP fpp14_sampled_nonsym_cells_hold_eps_vs_cpu: needs rocm/cuda");
}
