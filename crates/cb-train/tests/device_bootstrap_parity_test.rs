//! WR-01 device bootstrap parity — the phase's committed device sign-off.
//!
//! Covers four specifications on the REAL device:
//!
//! - `WR01-S11` — the BASE oblivious device grower holds ≤1e-5 against the CPU
//!   grower at `bootstrap_type = No`. Everything else here is meaningless if this
//!   fails, so it is asserted first and at the real bar (ε = 1e-5, never the
//!   shipped ε = 1e-4).
//! - `WR01-S12` — device/CPU split selection is allowed to diverge, but ONLY when
//!   the divergence is numerically inert. The device histogram is a fixed-point
//!   `Atomic<u64>` accumulator (`round(v · 2^30)`) while the CPU sums exact `f64`,
//!   so the two disagree by up to ~4.66e-10 per object per channel — enough to flip
//!   a near-tie. Split-structure equality can therefore never be an unconditional
//!   assertion; what IS assertable is that each divergent tree's own contribution
//!   still agrees to ≤1e-5, i.e. the tie-break picked an equivalent partition.
//! - `WR01-S15` — with Bernoulli / Bayesian / MVS sampling active, the device fit
//!   reproduces the CPU fit to ≤1e-5. This is the phase's headline claim.
//! - Poisson's backend contract — the device TRAINS it (upstream's GPU task type
//!   does) while the CPU grower refuses it with upstream's own wording. This supersedes
//!   `WR01-S16`'s placeholder "rejected on every backend"; the kernel's own bit-for-bit
//!   upstream gate lives in cb-backend's `poisson_bootstrap_oracle_test`.
//!
//! # Anti-false-pass discipline
//!
//! Two ways this suite could pass while proving nothing, both closed here:
//!
//! 1. **The device path silently falls back to the CPU grower.** Then "device ==
//!    CPU" is a tautology. Closed by [`device::CountingGpu`], which wraps the real
//!    `GpuBackend` and counts the trees that actually came back from
//!    `grow_tree_on_device`; every scenario asserts that count equals `iterations`.
//! 2. **The sample crosses the seam but is dropped before the histogram.** Then a
//!    sampled fit equals an unsampled one and "device == CPU" still holds. Closed
//!    by asserting each sampled fit differs MATERIALLY from the same fit with
//!    `bootstrap_type = No`.
//!
//! Runs on the REAL device only (rocm / cuda); `GpuBackend` is not compiled under
//! the `cpu` feature, so cpu/wgpu print a SKIP line instead of passing silently.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp
)]

use cb_compute::{EScoreFunction, LeafMethod, Loss};
use cb_train::{BoostParams, EBootstrapType, EGrowPolicy, EOverfittingDetectorType};

/// The ≤1e-5 project parity bar. NEVER relax this to the shipped device ε = 1e-4:
/// the whole point of WR-01 is that device sampling reaches the REAL bar.
#[cfg_attr(not(any(feature = "rocm", feature = "cuda")), allow(dead_code))]
const EPS: f64 = 1e-5;

/// A non-separable regression fixture large enough for per-tree round-off to
/// accumulate and for sampling to actually change which objects are scored:
/// `n` objects × `n_features` quantized ramps (32 bins each), target a smooth
/// nonlinear combination no single tree fits exactly.
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
        grow_policy: EGrowPolicy::SymmetricTree,
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
    use cb_train::{train, BoostParams, EBootstrapType};
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

    /// `WR01-S11` + `WR01-S12`: the base grower at `bootstrap_type = No`.
    pub fn base_grower_holds_1e5() {
        for &(n, nf, depth, iters) in
            &[(512usize, 4usize, 3usize, 5usize), (2048, 8, 6, 10), (20000, 16, 6, 20)]
        {
            let label = format!("BASE n={n} nf={nf} d={depth} it={iters}");
            let params = params_with(EBootstrapType::No, 1.0, 0.0, depth, iters);
            let (dev, cpu, mismatches, sample_len) = both(&label, &params, n, nf);

            // `bootstrap_type = No` ⇒ NO sample crosses the seam (WR01-S3: the grow
            // stays byte-identical to the pre-WR-01 path).
            assert_eq!(
                sample_len, 0,
                "[{label}] an unsampled fit must send an EMPTY sample across the seam"
            );

            let max_d = max_abs_diff(&dev, &cpu);
            println!(
                "[{label}] max|Δpred|={max_d:.3e} split_mismatched_trees={}/{iters}",
                mismatches.len()
            );
            assert!(
                max_d <= EPS,
                "[{label}] base device grower must hold <= {EPS:e} vs CPU, got {max_d:.3e}"
            );

            // WR01-S12: a divergent split is ACCEPTABLE only if that tree's own
            // contribution still agrees to <= EPS — i.e. the fixed-point histogram
            // flipped a near-tie between equivalent partitions rather than choosing a
            // materially different tree.
            for (t, tree_max) in &mismatches {
                println!("[{label}]   tree {t} split-mismatch: max|Δcontribution|={tree_max:.3e}");
                assert!(
                    *tree_max <= EPS,
                    "[{label}] tree {t} took a DIFFERENT split whose contribution differs by \
                     {tree_max:.3e} > {EPS:e}; that is a genuine structural divergence, not a \
                     fixed-point tie-break (WR01-S12)"
                );
            }
        }
    }

    /// `WR01-S15`: device == CPU at ≤1e-5 for each sampled bootstrap type, AND the
    /// sample demonstrably changed the model.
    pub fn sampled_types_hold_1e5() {
        let (n, nf, depth, iters) = (20000usize, 16usize, 6usize, 20usize);
        let scenarios: &[(&str, EBootstrapType, f64, f32)] = &[
            ("bernoulli", EBootstrapType::Bernoulli, 0.8, 0.0),
            ("bayesian", EBootstrapType::Bayesian, 1.0, 1.0),
            ("mvs", EBootstrapType::Mvs, 0.8, 0.0),
        ];

        // The unsampled reference this suite compares against to prove the sample was
        // not silently dropped.
        let (unsampled_dev, _, _, _) = both(
            "NO-REF",
            &params_with(EBootstrapType::No, 1.0, 0.0, depth, iters),
            n,
            nf,
        );

        for &(name, bt, subsample, temp) in scenarios {
            let label = format!("{name} n={n} nf={nf} d={depth} it={iters}");
            let params = params_with(bt, subsample, temp, depth, iters);
            let (dev, cpu, mismatches, sample_len) = both(&label, &params, n, nf);

            // WR01-S4: an n-length sample really crossed the seam.
            assert_eq!(
                sample_len, n,
                "[{label}] a sampled fit must send an n-length multiplier across the seam"
            );

            let max_d = max_abs_diff(&dev, &cpu);
            println!(
                "[{label}] max|Δpred(device,cpu)|={max_d:.3e} split_mismatched_trees={}/{iters} \
                 first_mismatched_trees={:?}",
                mismatches.len(),
                mismatches.iter().map(|(t, _)| *t).take(6).collect::<Vec<_>>(),
            );
            assert!(
                max_d <= EPS,
                "[{label}] device must reproduce the CPU sampled fit to <= {EPS:e}, \
                 got {max_d:.3e}"
            );

            for (t, tree_max) in &mismatches {
                assert!(
                    *tree_max <= EPS,
                    "[{label}] tree {t} split-mismatch contribution {tree_max:.3e} > {EPS:e} \
                     (WR01-S12)"
                );
            }

            // ── Anti-false-pass 2: the sample actually reached the histogram. ──
            // If the multiplier were dropped, this sampled fit would be identical to
            // the unsampled one and the ≤1e-5 assertion above would prove nothing.
            let vs_unsampled = max_abs_diff(&dev, &unsampled_dev);
            println!("[{label}] max|Δpred(sampled,unsampled)|={vs_unsampled:.3e}");
            assert!(
                vs_unsampled > EPS,
                "[{label}] the sampled device fit is indistinguishable from the UNSAMPLED fit \
                 (max|Δ|={vs_unsampled:.3e} <= {EPS:e}); the per-tree multiplier never reached \
                 the split histogram, so the parity assertion above is vacuous"
            );
        }
    }

    /// `WR01-S13`: the device leaf reduce's nondeterminism stays inside budget.
    ///
    /// `partition_update_kernel` merges leaf stats with a NAKED float atomic, which is
    /// order-dependent: tree STRUCTURE is bit-identical run to run, but leaf values and
    /// predictions carry ulp-level variance. That is a designed trade (it is what keeps
    /// the histogram fast), so the honest form of the claim is a measured BUDGET, not an
    /// assumption of determinism.
    ///
    /// Budget: ≤1e-7 on `max|Δpred|` across repeated identical fits — two orders
    /// stricter than the ≤1e-5 parity bar, so run-to-run jitter can never be what
    /// consumes the parity margin. Asserted WITH sampling active, because the host
    /// multiplier adds two more elementwise device products per tree and therefore more
    /// opportunity for reordering.
    pub fn run_to_run_jitter_within_budget() {
        const RUNS: usize = 5;
        const JITTER_BUDGET: f64 = 1e-7;
        let (n, nf, depth, iters) = (20000usize, 16usize, 6usize, 10usize);

        for &(name, bt, subsample, temp) in &[
            ("no", EBootstrapType::No, 1.0_f64, 0.0_f32),
            ("bernoulli", EBootstrapType::Bernoulli, 0.8, 0.0),
            ("mvs", EBootstrapType::Mvs, 0.8, 0.0),
        ] {
            let label = format!("jitter/{name}");
            let params = params_with(bt, subsample, temp, depth, iters);
            let mut baseline: Option<Vec<f64>> = None;
            let mut worst = 0.0_f64;
            for run in 0..RUNS {
                let (pred, _, _, _) = both(&label, &params, n, nf);
                match baseline.as_ref() {
                    None => baseline = Some(pred),
                    Some(b) => worst = worst.max(max_abs_diff(b, &pred)),
                }
                let _ = run;
            }
            println!("[{label}] run-to-run max|Δpred| over {RUNS} fits = {worst:.3e}");
            assert!(
                worst <= JITTER_BUDGET,
                "[{label}] run-to-run jitter {worst:.3e} exceeds the {JITTER_BUDGET:e} budget; \
                 the float-atomic leaf reduce is eating into the 1e-5 parity margin and the \
                 reduce would need converting to fixed-point atomics (WR01-S13)"
            );
        }
    }

    /// Poisson's backend contract, superseding `WR01-S16`'s "rejected everywhere".
    ///
    /// WR-01 rejected Poisson on every backend because it had no CPU-semantics parity
    /// target. That was a placeholder, not upstream's behaviour: upstream CatBoost
    /// TRAINS Poisson on the GPU task type and REFUSES it on the CPU one
    /// (`bootstrap_options.cpp:29`). Now that the device kernel is a verbatim
    /// transcription of upstream's `PoissonBootstrapImpl`, gated bit-for-bit in
    /// `cb-backend`'s `poisson_bootstrap_oracle_test`, this suite asserts the SAME
    /// asymmetry — the device trains it, the CPU grower refuses it with upstream's own
    /// wording.
    pub fn poisson_trains_on_device_and_is_refused_on_cpu() {
        let (n, nf) = (2048usize, 8usize);
        let (columns, borders, target) = fixture(n, nf);
        let params = params_with(EBootstrapType::Poisson, 0.8, 0.0, 4, 3);

        let gpu = CountingGpu::new();
        let dev = train(&gpu, &columns, &borders, &target, &[], &params, None)
            .expect("Poisson must TRAIN on the device backend (upstream's GPU task type does)");
        // The device really grew every tree — otherwise "it trained" would mean a CPU
        // fallback that cannot even express Poisson.
        assert_eq!(gpu.begun.get(), 1, "the device session must be accepted for Poisson");
        assert_eq!(
            gpu.grown.get(),
            params.iterations,
            "every Poisson tree must be grown ON DEVICE"
        );
        // The host must NOT have sampled: Poisson is drawn device-resident, so an empty
        // sample crosses the seam. A non-empty one would mean double sampling.
        assert_eq!(
            gpu.last_sample_len.get(),
            0,
            "Poisson is drawn device-resident; the host must pass an EMPTY sample"
        );
        assert_eq!(dev.oblivious_trees.len(), params.iterations);

        let cpu_err = train(&CpuRefRuntime, &columns, &borders, &target, &[], &params, None)
            .expect_err("Poisson must be REFUSED on the CPU grower, as upstream refuses it");
        assert!(
            matches!(cpu_err, CbError::Degenerate(_)),
            "CPU Poisson rejection must stay CbError::Degenerate, got {cpu_err:?}"
        );
        assert!(
            cpu_err.to_string().contains("poisson bootstrap is not supported on CPU"),
            "the CPU rejection should carry upstream's wording, got: {cpu_err}"
        );
        println!("[poisson] device trained {} trees; CPU refused: {cpu_err}", gpu.grown.get());
    }
}

#[cfg(any(feature = "rocm", feature = "cuda"))]
#[test]
fn wr01_base_device_grower_holds_1e5_vs_cpu() {
    device::base_grower_holds_1e5();
}

#[cfg(any(feature = "rocm", feature = "cuda"))]
#[test]
fn wr01_device_sampled_bootstrap_holds_1e5_vs_cpu() {
    device::sampled_types_hold_1e5();
}

#[cfg(any(feature = "rocm", feature = "cuda"))]
#[test]
fn wr01_device_run_to_run_jitter_within_budget() {
    device::run_to_run_jitter_within_budget();
}

#[cfg(any(feature = "rocm", feature = "cuda"))]
#[test]
fn poisson_trains_on_device_and_is_refused_on_cpu() {
    device::poisson_trains_on_device_and_is_refused_on_cpu();
}

#[cfg(not(any(feature = "rocm", feature = "cuda")))]
#[test]
fn wr01_device_bootstrap_parity_skipped_without_a_device() {
    // Anti-false-pass: `GpuBackend` is not compiled under the `cpu`/`wgpu` features,
    // so there is nothing to assert here. Print rather than silently pass, so a
    // cpu-feature run cannot be mistaken for device evidence.
    println!("SKIP wr01_device_bootstrap_parity: needs rocm/cuda");
}
