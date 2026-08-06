//! GDC-08 (T09): the weighted NON-symmetric device e2e oracles. Each of
//! Depthwise / Lossguide / Region commits to the device under the frozen
//! NON-uniform weights and reproduces a CPU `CpuRefRuntime` reference fit at
//! ε=1e-4; Depthwise ADDITIONALLY matches upstream `catboost==1.2.10`
//! (`weighted_device_nonsym/predictions.npy`) at ≤1e-5 (the T03 planning
//! decision: one upstream fixture per verified fix point, the remaining
//! policies vs the CPU reference).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing, clippy::float_cmp)]

use cb_compute::{EScoreFunction, LeafMethod, Loss};
use cb_train::{BoostParams, EBootstrapType, EGrowPolicy, EOverfittingDetectorType};

/// The `weighted_device_nonsym` pinned config, parameterized over the policy.
fn weighted_params(grow_policy: EGrowPolicy) -> BoostParams {
    BoostParams {
        loss: Loss::Rmse,
        iterations: 3,
        depth: 3,
        learning_rate: 0.3,
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
        one_hot_max_size: cb_train::one_hot_max_size_default(),
        permutation_count: 1,
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
    use std::path::PathBuf;

    use super::weighted_params;
    use cb_backend::GpuBackend;
    use cb_compute::{rmse_der1, rmse_der2, Derivatives, Loss, Runtime};
    use cb_core::CbResult;
    use cb_model::Model as CbModel;
    use cb_oracle::load_f64_vec;
    use cb_train::{train, EGrowPolicy};
    use ndarray::Array2;
    use ndarray_npy::read_npy;

    fn fixture(rel: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("cb-oracle")
            .join("fixtures")
            .join("weighted_device_nonsym")
            .join(rel)
    }

    /// Declines the device seam (trait defaults) and computes RMSE gradients —
    /// the CPU reference (`CpuBackend` is not compiled under rocm/cuda).
    struct CpuRefRuntime;

    impl Runtime for CpuRefRuntime {
        fn compute_gradients(
            &self,
            _loss: &Loss,
            approx: &[f64],
            target: &[f64],
            _approx_dimension: usize,
        ) -> CbResult<Derivatives> {
            let der1: Vec<f64> =
                approx.iter().zip(target).map(|(&a, &t)| rmse_der1(a, t)).collect();
            let der2: Vec<f64> =
                approx.iter().zip(target).map(|(&a, &t)| rmse_der2(a, t)).collect();
            Ok(Derivatives { der1, der2 })
        }
    }

    #[allow(clippy::type_complexity)]
    fn load_inputs() -> (Vec<Vec<f32>>, Vec<Vec<f64>>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let x: Array2<f32> = read_npy(fixture("X.npy")).expect("X.npy loads");
        let columns: Vec<Vec<f32>> = (0..x.ncols()).map(|fi| x.column(fi).to_vec()).collect();
        let borders_arr: Array2<f64> = read_npy(fixture("borders.npy")).expect("borders.npy");
        let borders: Vec<Vec<f64>> =
            (0..borders_arr.nrows()).map(|r| borders_arr.row(r).to_vec()).collect();
        let target = load_f64_vec(&fixture("y.npy")).unwrap();
        let weights = load_f64_vec(&fixture("weights.npy")).unwrap();
        let expected = load_f64_vec(&fixture("predictions.npy")).unwrap();
        (columns, borders, target, weights, expected)
    }

    pub fn run(grow_policy: EGrowPolicy, upstream: bool, label: &str) {
        let (columns, borders, target, weights, expected) = load_inputs();
        assert!(weights.iter().any(|&w| (w - 1.0).abs() > 1e-12));
        let params = weighted_params(grow_policy);

        let dev = train(&GpuBackend::default(), &columns, &borders, &target, &weights, &params, None)
            .unwrap_or_else(|e| panic!("[{label}] weighted device train failed: {e:?}"));
        let cpu = train(&CpuRefRuntime, &columns, &borders, &target, &weights, &params, None)
            .unwrap_or_else(|e| panic!("[{label}] weighted cpu reference train failed: {e:?}"));

        // Arm routing per policy (no silent CPU-fallback tree shape).
        match grow_policy {
            EGrowPolicy::Region => {
                assert_eq!(dev.region_trees.len(), params.iterations, "[{label}] region arm");
                assert!(dev.oblivious_trees.is_empty() && dev.non_symmetric_trees.is_empty());
            }
            _ => {
                assert_eq!(
                    dev.non_symmetric_trees.len(),
                    params.iterations,
                    "[{label}] nonsym arm"
                );
                assert!(dev.oblivious_trees.is_empty() && dev.region_trees.is_empty());
            }
        }

        let dev_model = CbModel::from_trained(&dev, borders.clone());
        let cpu_model = CbModel::from_trained(&cpu, borders.clone());
        let dev_pred = cb_model::predict_raw(&dev_model, &columns);
        let cpu_pred = cb_model::predict_raw(&cpu_model, &columns);
        let mut max_vs_cpu = 0.0_f64;
        for (i, (&d, &c)) in dev_pred.iter().zip(cpu_pred.iter()).enumerate() {
            let abs = (d - c).abs();
            max_vs_cpu = max_vs_cpu.max(abs);
            assert!(
                abs <= 1e-4,
                "[{label}] obj {i}: device {d} vs cpu reference {c} exceeds ε=1e-4 (|Δ|={abs:.3e})"
            );
        }

        let mut max_vs_upstream = f64::NAN;
        if upstream {
            max_vs_upstream = 0.0;
            for (i, (&d, &e)) in dev_pred.iter().zip(expected.iter()).enumerate() {
                let abs = (d - e).abs();
                max_vs_upstream = max_vs_upstream.max(abs);
                assert!(
                    abs <= 1e-5,
                    "[{label}] obj {i}: device {d} vs upstream {e} exceeds ≤1e-5 (|Δ|={abs:.3e})"
                );
            }
        }
        println!(
            "[{label}] max |Δ| vs cpu = {max_vs_cpu:.3e} (1e-4); vs upstream = {max_vs_upstream:.3e}"
        );
    }
}

#[test]
fn device_weighted_depthwise_matches_upstream() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    device::run(EGrowPolicy::Depthwise, true, "depthwise-weighted-e2e");
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    {
        let _ = weighted_params(EGrowPolicy::Depthwise);
        eprintln!("SKIP device_weighted_depthwise_matches_upstream: needs rocm/cuda");
    }
}

#[test]
fn device_weighted_lossguide_matches_cpu() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    device::run(EGrowPolicy::Lossguide, false, "lossguide-weighted-e2e");
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    {
        let _ = weighted_params(EGrowPolicy::Lossguide);
        eprintln!("SKIP device_weighted_lossguide_matches_cpu: needs rocm/cuda");
    }
}

#[test]
fn device_weighted_region_matches_cpu() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    device::run(EGrowPolicy::Region, false, "region-weighted-e2e");
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    {
        let _ = weighted_params(EGrowPolicy::Region);
        eprintln!("SKIP device_weighted_region_matches_cpu: needs rocm/cuda");
    }
}
