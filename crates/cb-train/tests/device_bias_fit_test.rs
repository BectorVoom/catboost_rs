//! FPP-04 (T12): the NON-ZERO-BIAS device e2e oracle — a real
//! `cb_train::train(&GpuBackend::default(), …)` fit over the frozen `bias_device_sym/`
//! fixture COMMITS to the device (arm-routing assertion, no silent CPU fallback) and its
//! predictions match upstream `catboost==1.2.10` at ≤1e-5.
//!
//! This is the end-to-end proof of the whole bias track: FPP-01 seeded the resident
//! approximant from `DeviceTrainConfig.bias` and FPP-02 removed the gate clause, but only
//! a fit against real upstream output can show the two compose to the parity bar. The
//! fixture's generator asserts bias-on vs bias-off differ by 1.42, so a device that
//! ignored the bias would miss by ~5 orders of magnitude, not by rounding.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing, clippy::float_cmp)]

use cb_compute::{EScoreFunction, LeafMethod, Loss};
use cb_train::{BoostParams, EBootstrapType, EGrowPolicy, EOverfittingDetectorType};

/// The `bias_device_sym` pinned config (mirrors its `config.json` params).\n/// `boost_from_average: true` is the one field that differs from the weighted fixture.
fn bias_params(grow_policy: EGrowPolicy) -> BoostParams {
    BoostParams {
        loss: Loss::Rmse,
        iterations: 3,
        depth: 3,
        learning_rate: 0.3,
        l2_leaf_reg: 3.0,
        random_strength: 0.0,
        boost_from_average: true,
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
        extra: Default::default(),
    }
}


#[cfg(any(feature = "rocm", feature = "cuda"))]
mod device {
    use std::path::PathBuf;

    use super::bias_params;
    use cb_backend::GpuBackend;
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
            .join("bias_device_sym")
            .join(rel)
    }

    pub fn run() {
        let x: Array2<f32> = read_npy(fixture("X.npy")).expect("X.npy loads");
        let columns: Vec<Vec<f32>> = (0..x.ncols()).map(|fi| x.column(fi).to_vec()).collect();
        let borders_arr: Array2<f64> = read_npy(fixture("borders.npy")).expect("borders.npy");
        let borders: Vec<Vec<f64>> =
            (0..borders_arr.nrows()).map(|r| borders_arr.row(r).to_vec()).collect();
        let target = load_f64_vec(&fixture("y.npy")).unwrap();
        let expected = load_f64_vec(&fixture("predictions.npy")).unwrap();

        // The starting approximant IS mean(y); a near-zero mean could not discriminate a
        // seeded bias from the former hardcoded zero.
        let mean = target.iter().sum::<f64>() / (target.len() as f64);
        assert!(
            mean.abs() > 0.5,
            "the fixture's |mean(y)| = {mean:.6} must exceed 0.5 or this oracle is vacuous"
        );

        let params = bias_params(EGrowPolicy::SymmetricTree);
        assert!(params.boost_from_average, "the whole point of this fixture");

        let trained = train(
            &GpuBackend::default(),
            &columns,
            &borders,
            &target,
            &[],
            &params,
            None,
        )
        .unwrap_or_else(|e| panic!("bias sym device train failed: {e:?}"));

        // Arm routing: the device oblivious grower fired (no CPU fallback shape).
        assert_eq!(trained.oblivious_trees.len(), params.iterations);
        assert!(trained.non_symmetric_trees.is_empty() && trained.region_trees.is_empty());

        let model = CbModel::from_trained(&trained, borders);
        let actual = cb_model::predict_raw(&model, &columns);
        assert_eq!(actual.len(), expected.len());
        let mut max_abs = 0.0_f64;
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            let abs = (a - e).abs();
            max_abs = max_abs.max(abs);
            assert!(
                abs <= 1e-5,
                "obj {i}: bias device prediction {a} vs upstream {e} exceeds ≤1e-5 \
                 (|Δ|={abs:.3e})"
            );
        }
        println!("[device-bias-sym-e2e] max |Δpred| = {max_abs:.3e} (bar 1e-5)");
    }
}

#[test]
fn device_bias_symmetric_fit_matches_upstream() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    device::run();
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    {
        let _ = bias_params(cb_train::EGrowPolicy::SymmetricTree);
        eprintln!("SKIP device_bias_symmetric_fit_matches_upstream: needs rocm/cuda");
    }
}
