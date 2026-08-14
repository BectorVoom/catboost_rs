//! FPP-08 (T13): the EXACT-leaf device e2e oracle — real
//! `cb_train::train(&GpuBackend::default(), …)` fits over the frozen
//! `exact_leaf_device/{mae,quantile07}/` fixtures COMMIT to the device and their
//! predictions match upstream `catboost==1.2.10` at ≤1e-5.
//!
//! BOTH arms run, and that is the point: a device path that computed an Exact order
//! statistic but silently ignored `quantile_alpha` would pass the MAE arm alone. The
//! fixtures' generator asserts the two differ by 0.432, so the α=0.7 arm is a real
//! discriminator, and each arm additionally differs from its Gradient sibling by >0.67 —
//! so a device that fell back to `calc_average` would miss by orders of magnitude.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing, clippy::float_cmp)]

use cb_compute::{EScoreFunction, LeafMethod, Loss};
use cb_train::{BoostParams, EBootstrapType, EGrowPolicy, EOverfittingDetectorType};

/// The `exact_leaf_device/*` pinned config (mirrors their `config.json` params).\n/// `leaf_estimation_method = Exact` over the arm's own loss; everything else is the\n/// same covered regime as the weighted fixture.
fn exact_params(loss: Loss) -> BoostParams {
    BoostParams {
        loss: loss,
        iterations: 3,
        depth: 3,
        learning_rate: 0.3,
        l2_leaf_reg: 3.0,
        random_strength: 0.0,
        boost_from_average: false,
        leaf_method: LeafMethod::Exact,
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
        grow_policy: EGrowPolicy::SymmetricTree,
        max_leaves: cb_train::max_leaves_default(),
        min_data_in_leaf: cb_train::min_data_in_leaf_default(),
        extra: Default::default(),
    }
}


#[cfg(any(feature = "rocm", feature = "cuda"))]
mod device {
    use std::path::PathBuf;

    use super::exact_params;
    use cb_backend::GpuBackend;
    use cb_compute::Loss;
    use cb_model::Model as CbModel;
    use cb_oracle::load_f64_vec;
    use cb_train::train;
    use ndarray::Array2;
    use ndarray_npy::read_npy;

    fn fixture(arm: &str, rel: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("cb-oracle")
            .join("fixtures")
            .join("exact_leaf_device")
            .join(arm)
            .join(rel)
    }

    /// Fit one arm on the device and return its predictions, asserting the ≤1e-5 bar.
    pub fn run(arm: &str, loss: Loss) -> Vec<f64> {
        let x: Array2<f32> = read_npy(fixture(arm, "X.npy")).expect("X.npy loads");
        let columns: Vec<Vec<f32>> = (0..x.ncols()).map(|fi| x.column(fi).to_vec()).collect();
        let borders_arr: Array2<f64> =
            read_npy(fixture(arm, "borders.npy")).expect("borders.npy");
        let borders: Vec<Vec<f64>> =
            (0..borders_arr.nrows()).map(|r| borders_arr.row(r).to_vec()).collect();
        let target = load_f64_vec(&fixture(arm, "y.npy")).unwrap();
        let expected = load_f64_vec(&fixture(arm, "predictions.npy")).unwrap();

        let params = exact_params(loss);
        let trained = train(
            &GpuBackend::default(),
            &columns,
            &borders,
            &target,
            &[],
            &params,
            None,
        )
        .unwrap_or_else(|e| panic!("[{arm}] exact-leaf device train failed: {e:?}"));

        // Arm routing: the device oblivious grower fired (no CPU fallback shape).
        assert_eq!(trained.oblivious_trees.len(), params.iterations, "[{arm}] tree count");
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
                "[{arm}] obj {i}: exact-leaf device prediction {a} vs upstream {e} \
                 exceeds ≤1e-5 (|Δ|={abs:.3e})"
            );
        }
        println!("[device-exact-{arm}-e2e] max |Δpred| = {max_abs:.3e} (bar 1e-5)");
        actual
    }
}

#[test]
fn device_exact_leaf_fits_match_upstream_and_alpha_is_load_bearing() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    {
        let mae = device::run("mae", Loss::Mae);
        let q07 = device::run(
            "quantile07",
            Loss::Quantile { alpha: 0.7, delta: 1e-6 },
        );

        // The α discriminator: if the device ignored `quantile_alpha` and always took the
        // median, both arms would predict identically — and BOTH would still have to be
        // wrong against their own upstream fixture to be caught above. Asserting the gap
        // here catches it directly.
        let max_delta = mae
            .iter()
            .zip(q07.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_delta > 1e-6,
            "the MAE and Quantile:alpha=0.7 device fits agree (max|Δ|={max_delta:.3e}) — \
             quantile_alpha is not reaching the device leaf"
        );
        println!("[device-exact-e2e] alpha discrimination max|Δ| = {max_delta:.6}");
    }
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    {
        let _ = exact_params(Loss::Mae);
        eprintln!(
            "SKIP device_exact_leaf_fits_match_upstream_and_alpha_is_load_bearing: needs rocm/cuda"
        );
    }
}
