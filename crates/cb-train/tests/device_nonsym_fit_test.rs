//! Device Depthwise / Lossguide non-symmetric grow END-TO-END oracle (GPUT-18, Plan 12-03
//! Task 3). Trains the SAME clear-margin fixture through `cb_train::train()` twice — once on the
//! real device backend (`GpuBackend`, which commits the fit to the Plan-03 device non-sym grow)
//! and once on a CPU-declining reference runtime (the `leaf_wise_grower` path) — and locks:
//!
//!   - the DEVICE fit folds through the Plan-03 device-fold non-sym arm into
//!     `non_symmetric_trees` (NON-EMPTY), with `oblivious_trees` EMPTY — proving the widened
//!     Depthwise / Lossguide gate fires the NonSymmetricTree fold arm, NOT a degenerate
//!     ObliviousTree (the checker BLOCKER this plan closes);
//!   - the device model's predictions reproduce the CPU `leaf_wise_grower`-grown model within
//!     ε=1e-4 (both lifted into `cb_model` and applied).
//!
//! Runs on the REAL device only (rocm gfx1100 in-env / cuda) — the cubecl-cpu backend cannot
//! JIT the per-node score/argmin over the grow's subset shapes, and `GpuBackend` is not even
//! compiled under the `cpu` feature — so cpu/wgpu SKIP (the WR-01 anti-false-pass convention).
//! Kaggle CUDA ε=1e-4 sign-off is deferred to Plan 09; the in-env oracle is the local gate.
//! Lives under `tests/` (integration) so `cb_train` is the SAME external crate instance
//! `cb_model` links (the dev-dep diamond a src test would hit, 12-02 SUMMARY Deviation 2).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing, clippy::float_cmp)]

use cb_compute::{EScoreFunction, LeafMethod, Loss};
use cb_train::{BoostParams, EBootstrapType, EGrowPolicy, EOverfittingDetectorType};

/// A clear-margin fixture: feature 0 is a 32-bin ramp (`value = obj index`), feature 1 is an
/// unused low-gain spread; the target is a clean step on feature 0 (bins `<= 15` → `+1`, else
/// `-1`), so RMSE der1 == target and the leaf-wise grower splits feature 0 at the separating
/// border, both pure children stopping (gain 0). Device and CPU agree on structure + leaves.
// Consumed only by the rocm/cuda `device` module; the cpu-skip build never calls it.
#[cfg_attr(not(any(feature = "rocm", feature = "cuda")), allow(dead_code))]
fn fixture() -> (Vec<Vec<f32>>, Vec<Vec<f64>>, Vec<f64>) {
    let n = 64usize;
    let f0: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let f1: Vec<f32> = (0..n).map(|i| (i % 7) as f32).collect();
    // 31 borders at k+0.5 → 32 bins; value i lands in bin min(i, 31).
    let borders0: Vec<f64> = (0..31).map(|k| k as f64 + 0.5).collect();
    let borders1: Vec<f64> = (0..6).map(|k| k as f64 + 0.5).collect();
    let target: Vec<f64> = (0..n).map(|i| if i <= 15 { 1.0 } else { -1.0 }).collect();
    (vec![f0, f1], vec![borders0, borders1], target)
}

/// A device-eligible non-symmetric [`BoostParams`]: RMSE / Plain / fold-1 / unit-weight /
/// bias-0 / Gradient leaf, so the fit commits to the device grow. `grow_policy` is the caller's.
fn nonsym_params(grow_policy: EGrowPolicy) -> BoostParams {
    BoostParams {
        loss: Loss::Rmse,
        iterations: 2,
        depth: 3,
        learning_rate: 1.0,
        l2_leaf_reg: 0.0,
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
    }
}

/// The device-backed end-to-end body. Compiled ONLY for the real GPU backends: `GpuBackend`
/// (the device grow) exists only under wgpu/cuda/rocm, and this oracle runs only where the
/// device grow actually executes (rocm/cuda — wgpu has no f64 and cpu cannot JIT the scorer).
#[cfg(any(feature = "rocm", feature = "cuda"))]
mod device {
    use super::{fixture, nonsym_params};
    use cb_backend::GpuBackend;
    use cb_compute::{rmse_der1, rmse_der2, Derivatives, Loss, Runtime};
    use cb_core::CbResult;
    use cb_model::Model as CbModel;
    use cb_train::{train, EGrowPolicy};

    /// A CPU reference runtime that DECLINES the device grow (so `train` uses the byte-unchanged
    /// `leaf_wise_grower` path) and computes the same RMSE gradients the real `CpuBackend` does
    /// (`CpuBackend` itself is not compiled under the rocm/cuda feature). Every device-seam method
    /// inherits the trait default (`begin → Ok(false)`, `grow → Ok(None)`).
    struct CpuRefRuntime;

    impl Runtime for CpuRefRuntime {
        fn compute_gradients(
            &self,
            _loss: &Loss,
            approx: &[f64],
            target: &[f64],
            _approx_dimension: usize,
        ) -> CbResult<Derivatives> {
            // The fixture is RMSE / single-dimension: der1 = target - approx, der2 = -1.
            let der1: Vec<f64> =
                approx.iter().zip(target).map(|(&a, &t)| rmse_der1(a, t)).collect();
            let der2: Vec<f64> =
                approx.iter().zip(target).map(|(&a, &t)| rmse_der2(a, t)).collect();
            Ok(Derivatives { der1, der2 })
        }
    }

    pub fn run(grow_policy: EGrowPolicy, label: &str) {
        let (columns, borders, target) = fixture();
        let params = nonsym_params(grow_policy);

        // DEVICE fit (GpuBackend commits to the Plan-03 device non-sym grow).
        let dev = train(&GpuBackend::default(), &columns, &borders, &target, &[], &params, None)
            .unwrap_or_else(|e| panic!("[{label}] device non-sym train failed: {e:?}"));
        // CPU reference fit (declines device → leaf_wise_grower).
        let cpu = train(&CpuRefRuntime, &columns, &borders, &target, &[], &params, None)
            .unwrap_or_else(|e| panic!("[{label}] cpu non-sym train failed: {e:?}"));

        // The DEVICE fit folds into non_symmetric_trees (NOT a degenerate ObliviousTree).
        assert!(
            dev.oblivious_trees.is_empty(),
            "[{label}] device non-sym fit must NOT produce oblivious trees (degenerate-ObliviousTree regression)"
        );
        assert_eq!(
            dev.non_symmetric_trees.len(),
            params.iterations,
            "[{label}] device fit must fold one NonSymmetricTree per iteration into non_symmetric_trees"
        );
        assert!(dev.region_trees.is_empty(), "[{label}] device non-sym fit must not touch region_trees");

        // The CPU reference is likewise all-non-symmetric.
        assert!(cpu.oblivious_trees.is_empty(), "[{label}] cpu reference must be all-non-symmetric");
        assert_eq!(cpu.non_symmetric_trees.len(), params.iterations);

        // Predictions reproduce the CPU-grown model within ε=1e-4 (both lifted into cb_model).
        let dev_model = CbModel::from_trained(&dev, borders.clone());
        let cpu_model = CbModel::from_trained(&cpu, borders.clone());
        assert!(
            dev_model.non_symmetric_trees.len() == params.iterations
                && dev_model.oblivious_trees.is_empty()
        );

        let dev_pred = cb_model::predict_raw(&dev_model, &columns);
        let cpu_pred = cb_model::predict_raw(&cpu_model, &columns);
        assert_eq!(dev_pred.len(), cpu_pred.len());
        let mut max_abs = 0.0_f64;
        for (i, (&d, &c)) in dev_pred.iter().zip(cpu_pred.iter()).enumerate() {
            let abs = (d - c).abs();
            max_abs = max_abs.max(abs);
            assert!(
                abs <= 1e-4,
                "[{label}] obj {i}: device pred {d} vs cpu {c} exceeds ε=1e-4 (abs={abs:.3e})"
            );
        }
        let target_err = dev_pred
            .iter()
            .zip(target.iter())
            .map(|(&p, &t)| (p - t).abs())
            .fold(0.0_f64, f64::max);
        println!(
            "[{label}] device vs cpu max |Δpred| = {max_abs:.3e}; device vs target max |Δ| = {target_err:.3e}"
        );
        assert!(
            target_err <= 1e-4,
            "[{label}] the separable step fixture must be fit within ε=1e-4 (target_err={target_err:.3e})"
        );
    }
}

#[test]
fn device_depthwise_fit_reproduces_cpu_leaf_wise() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    device::run(EGrowPolicy::Depthwise, "depthwise");
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    {
        let _ = nonsym_params(EGrowPolicy::Depthwise);
        eprintln!("SKIP device_depthwise_fit_reproduces_cpu_leaf_wise: needs rocm/cuda");
    }
}

#[test]
fn device_lossguide_fit_reproduces_cpu_leaf_wise() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    device::run(EGrowPolicy::Lossguide, "lossguide");
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    {
        let _ = nonsym_params(EGrowPolicy::Lossguide);
        eprintln!("SKIP device_lossguide_fit_reproduces_cpu_leaf_wise: needs rocm/cuda");
    }
}
