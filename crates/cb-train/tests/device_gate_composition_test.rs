//! GDC-19 (T23): cross-gap composition regression tests — verification only, no
//! production code of its own (a failing assertion here is a T07/T14 bug).
//!
//! 1. `ordered_plus_ctr_still_declines` — the Ordered clause is UNTOUCHED (D5):
//!    `boosting_type=Ordered`, alone or with CTR categoricals, still declines to
//!    the CPU path exactly as before this phase (acceptance scenario 5).
//! 2. `weighted_plus_ctr_admits_together` — the POSITIVE composition: non-uniform
//!    weights × CTR commit to the device TOGETHER and match upstream
//!    `catboost==1.2.10` (`predictions_weighted.npy`) at ≤1e-5 (scenario 6).
//! 3. `depthwise_plus_bayesian_bootstrap_commits` — the bootstrap × grow-policy
//!    cross-product, which was excluded when this file was written and is now COVERED
//!    (FPP-12/FPP-13). Kept here as a composition guard: relaxing that cross-product must
//!    not disturb the two exclusions above.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing, clippy::float_cmp)]

use cb_compute::{EScoreFunction, LeafMethod, Loss};
use cb_train::{
    BoostParams, EBootstrapType, EBoostingType, EGrowPolicy, EOverfittingDetectorType,
};

/// The `ctr_device_mixed` config, parameterized for the composition cases.
fn base_params() -> BoostParams {
    BoostParams {
        loss: Loss::Logloss,
        iterations: 5,
        depth: 2,
        learning_rate: 0.1,
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
        one_hot_max_size: 1,
        permutation_count: 1,
        fold_len_multiplier: 2.0,
        simple_ctr: cb_train::ECtrType::Borders,
        simple_ctr_priors: vec![0.5],
        counter_calc_method: cb_train::counter_calc_method_default(),
        boosting_type: cb_train::boosting_type_default(),
        max_ctr_complexity: 1,
        combinations_ctr: cb_train::combinations_ctr_default(),
        combinations_ctr_priors: cb_train::combinations_ctr_priors_default(),
        score_function: EScoreFunction::L2,
        has_time: false,
        feature_weights: cb_train::feature_weights_default(),
        first_feature_use_penalties: cb_train::first_feature_use_penalties_default(),
        per_object_feature_penalties: cb_train::per_object_feature_penalties_default(),
        penalties_coefficient: cb_train::penalties_coefficient_default(),
        monotone_constraints: cb_train::monotone_constraints_default(),
        grow_policy: cb_train::grow_policy_default(),
        max_leaves: cb_train::max_leaves_default(),
        min_data_in_leaf: cb_train::min_data_in_leaf_default(),
    }
}

#[cfg(any(feature = "rocm", feature = "cuda"))]
mod device {
    use std::cell::Cell;
    use std::path::PathBuf;

    use super::base_params;
    use cb_backend::GpuBackend;
    use cb_compute::{
        DeviceGrownTree, DeviceTrainConfig, EScoreFunction, FamilyTreeArgs, Loss, Runtime,
    };
    use cb_core::CbResult;
    use cb_data::stringify_int_category;
    use cb_model::Model as CbModel;
    use cb_oracle::load_f64_vec;
    use cb_train::{train, train_cat, BoostParams};
    use ndarray::{Array1, Array2};
    use ndarray_npy::read_npy;

    fn fixture(rel: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("cb-oracle")
            .join("fixtures")
            .join("ctr_device_mixed")
            .join(rel)
    }

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

    #[allow(clippy::type_complexity)]
    fn load_mixed() -> (Vec<Vec<f32>>, Vec<Vec<String>>, Vec<f64>) {
        let x: Array2<f32> = read_npy(fixture("X.npy")).expect("X.npy loads");
        let columns: Vec<Vec<f32>> = (0..x.ncols()).map(|fi| x.column(fi).to_vec()).collect();
        let cat: Array1<i32> = read_npy(fixture("X_cat.npy")).expect("X_cat.npy");
        let cat_columns =
            vec![cat.iter().map(|&c| stringify_int_category(i64::from(c))).collect()];
        let target = load_f64_vec(&fixture("y.npy")).unwrap();
        (columns, cat_columns, target)
    }

    fn load_borders(name: &str) -> Vec<Vec<f64>> {
        let arr: Array2<f64> = read_npy(fixture(name)).expect("borders load");
        (0..arr.nrows()).map(|r| arr.row(r).to_vec()).collect()
    }

    /// GDC-19.1: Ordered (+ CTR) still declines — the Ordered clause is untouched.
    pub fn ordered_plus_ctr_declines() {
        let (columns, cat_columns, target) = load_mixed();
        let borders = load_borders("borders.npy");
        let params = BoostParams {
            boosting_type: cb_train::EBoostingType::Ordered,
            ..base_params()
        };
        let gpu = CountingGpu { inner: GpuBackend::default(), grown: Cell::new(0) };
        train_cat(&gpu, &columns, &borders, &cat_columns, &target, &[], &params, None)
            .expect("ordered+ctr CPU fit must succeed");
        assert_eq!(
            gpu.grown.get(),
            0,
            "Ordered × CTR must still decline to CPU (the D5-untouched clause)"
        );
    }

    /// GDC-19.2 (POSITIVE): weighted × CTR admits together, ≤1e-5 vs upstream.
    pub fn weighted_plus_ctr_admits() {
        let (columns, cat_columns, target) = load_mixed();
        let borders = load_borders("borders_weighted.npy");
        let weights = load_f64_vec(&fixture("weights.npy")).unwrap();
        let expected = load_f64_vec(&fixture("predictions_weighted.npy")).unwrap();
        assert!(weights.iter().any(|&w| (w - 1.0).abs() > 1e-12));
        let params = base_params();
        let gpu = CountingGpu { inner: GpuBackend::default(), grown: Cell::new(0) };
        let (trained, baked) = train_cat(
            &gpu, &columns, &borders, &cat_columns, &target, &weights, &params, None,
        )
        .expect("weighted+ctr device fit must succeed");
        assert_eq!(
            gpu.grown.get(),
            params.iterations,
            "weighted × CTR must COMMIT to the device together (positive composition)"
        );
        let model = CbModel::from_trained(&trained, borders)
            .with_ctr_data(cb_model::CtrData::from_baked(&baked));
        let actual = cb_model::predict_raw_cat(&model, &columns, &cat_columns);
        let mut max_abs = 0.0_f64;
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            let abs = (a - e).abs();
            max_abs = max_abs.max(abs);
            assert!(
                abs <= 1e-5,
                "obj {i}: weighted×CTR device pred {a} vs upstream {e} (|Δ|={abs:.3e})"
            );
        }
        println!("[weighted×ctr] device grows = {}, max |Δpred| = {max_abs:.3e}", gpu.grown.get());
    }

    /// GDC-19.3: an exclusion untouched by this phase is STILL excluded.
    pub fn depthwise_plus_bayesian_commits() {
        let n = 64usize;
        let f0: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let borders0: Vec<f64> = (0..31).map(|k| k as f64 + 0.5).collect();
        let target: Vec<f64> = (0..n).map(|i| if i <= 15 { 1.0 } else { -1.0 }).collect();
        let params = BoostParams {
            loss: Loss::Rmse,
            grow_policy: cb_train::EGrowPolicy::Depthwise,
            bootstrap_type: cb_train::EBootstrapType::Bayesian,
            bagging_temperature: 1.0,
            simple_ctr_priors: cb_train::simple_ctr_priors_default(),
            simple_ctr: cb_train::simple_ctr_default(),
            one_hot_max_size: cb_train::one_hot_max_size_default(),
            max_ctr_complexity: cb_train::max_ctr_complexity_default(),
            fold_len_multiplier: cb_train::fold_len_multiplier_default(),
            ..base_params()
        };
        let gpu = CountingGpu { inner: GpuBackend::default(), grown: Cell::new(0) };
        train(&gpu, &[f0], &[borders0], &target, &[], &params, None)
            .expect("depthwise+bayesian fit must succeed");
        // FPP-13 (T11): this assertion is INVERTED. Depthwise × Bayesian used to decline
        // because the non-symmetric grower IGNORED the per-object bootstrap multiplier, so
        // the backend refused rather than silently drop it. FPP-12 (T08) gave the grower
        // real SPLIT-SCORING channels, and FPP-13 removed the
        // `bootstrap_type × grow_policy == SymmetricTree` restriction, so this cell now
        // COMMITS. The full cross-product is covered by device_nonsym_bootstrap_gate_test.
        assert_eq!(
            gpu.grown.get(),
            params.iterations,
            "Depthwise × Bayesian is device-eligible since FPP-13 and must COMMIT"
        );
    }
}

#[test]
fn ordered_plus_ctr_still_declines() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    device::ordered_plus_ctr_declines();
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    {
        let _ = (base_params(), EBoostingType::Ordered);
        eprintln!("SKIP ordered_plus_ctr_still_declines: needs rocm/cuda");
    }
}

#[test]
fn weighted_plus_ctr_admits_together() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    device::weighted_plus_ctr_admits();
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    {
        let _ = base_params();
        eprintln!("SKIP weighted_plus_ctr_admits_together: needs rocm/cuda");
    }
}

#[test]
fn depthwise_plus_bayesian_bootstrap_commits() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    device::depthwise_plus_bayesian_commits();
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    {
        let _ = (base_params(), EGrowPolicy::Depthwise, EBootstrapType::Bayesian);
        eprintln!("SKIP depthwise_plus_bayesian_bootstrap_still_declines: needs rocm/cuda");
    }
}
