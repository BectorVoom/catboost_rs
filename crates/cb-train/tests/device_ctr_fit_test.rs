//! GDC-12 (T15): the CTR device e2e oracle — a real `train_cat(&GpuBackend, …)`
//! fit on the mixed float+cat `ctr_device_mixed` fixture COMMITS to the device
//! and its predictions match upstream `catboost==1.2.10` at ≤1e-5, structure AND
//! leaf values. This bar is the one a structure-only leaf-gather implementation
//! FAILS (research pitfall #2): the fixture's structure and averaging
//! permutations genuinely diverge (asserted below), so an implementation that
//! gathers leaf values over the structure bins produces different predictions.
//!
//! # DCTR-19 (T20): why `oblivious_trees.len()` is NOT a device-commit assertion
//!
//! Until T20 this file asserted only `trained.oblivious_trees.len() ==
//! params.iterations` plus `non_symmetric_trees.is_empty() && region_trees.is_empty()`.
//! **Both hold for a pure-CPU fit**: the CPU oblivious grower produces exactly
//! `iterations` symmetric trees and no non-symmetric/Region trees, so those
//! assertions are satisfied whether or not the fit ever reached the GPU. The
//! ≤1e-5 prediction bar does not close the gap either — the CPU path is *more*
//! accurate against the upstream oracle, not less. Measured on this fixture with
//! the device gate forced closed (`&& false` in
//! `cb_train::boosting::ctr_types_are_device_covered`): the whole test still
//! **passed**, printing `max |Δpred| = 1.388e-17` (the CPU-fallback value) in
//! place of the device path's `4.483e-11`. That is the R-8 false-pass class this
//! phase exists to eliminate.
//!
//! The only sound evidence is counting the calls the boosting loop actually makes
//! into `Runtime::grow_tree_on_device`. The `CountingGpu` wrapper below is copied
//! **verbatim** from the canonical copy at
//! `crates/cb-train/tests/device_ctr_gate_test.rs:82-138` (the wrapper block of
//! that file's `:60-170` device module); keep the two in sync.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing, clippy::float_cmp)]

use cb_compute::{EScoreFunction, LeafMethod, Loss};
use cb_train::{BoostParams, EBootstrapType, EOverfittingDetectorType};

/// The `ctr_device_mixed` pinned config (mirrors its `config.json` params).
fn ctr_params() -> BoostParams {
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
        extra: Default::default(),
    }
}

#[cfg(any(feature = "rocm", feature = "cuda"))]
mod device {
    use std::cell::Cell;
    use std::path::PathBuf;

    use super::ctr_params;
    use cb_backend::GpuBackend;
    use cb_compute::{
        DeviceGrownTree, DeviceTrainConfig, EScoreFunction, FamilyTreeArgs, Loss, Runtime,
    };
    use cb_core::CbResult;
    use cb_data::stringify_int_category;
    use cb_model::Model as CbModel;
    use cb_oracle::load_f64_vec;
    use cb_train::{averaging_ctr_permutation, create_shuffled_indices, train_cat};
    use ndarray::{Array1, Array2};
    use ndarray_npy::read_npy;

    /// Anti-false-pass device-commitment counter (DCTR-19 / R-8). Copied
    /// **verbatim** from `device_ctr_gate_test.rs:82-138`; every override
    /// forwards to `self.inner`, and `grown` counts only the
    /// `grow_tree_on_device` calls that actually returned a device tree.
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

    fn fixture(rel: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("cb-oracle")
            .join("fixtures")
            .join("ctr_device_mixed")
            .join(rel)
    }

    pub fn run() {
        let x: Array2<f32> = read_npy(fixture("X.npy")).expect("X.npy loads");
        let columns: Vec<Vec<f32>> = (0..x.ncols()).map(|fi| x.column(fi).to_vec()).collect();
        let borders_arr: Array2<f64> = read_npy(fixture("borders.npy")).expect("borders.npy");
        let borders: Vec<Vec<f64>> =
            (0..borders_arr.nrows()).map(|r| borders_arr.row(r).to_vec()).collect();
        let cat: Array1<i32> = read_npy(fixture("X_cat.npy")).expect("X_cat.npy");
        let cat_columns: Vec<Vec<String>> =
            vec![cat.iter().map(|&c| stringify_int_category(i64::from(c))).collect()];
        let target = load_f64_vec(&fixture("y.npy")).unwrap();
        let expected = load_f64_vec(&fixture("predictions.npy")).unwrap();
        let n = target.len();
        let params = ctr_params();

        // Fixture-permutation-divergence guard (SPEC GDC-12's Unresolved item):
        // the structure order (the learn-set shuffle S) and the averaging order
        // (S ∘ P_avg) must differ at this fixture's (n, seed), or this oracle
        // could not discriminate a structure-only leaf gather.
        let structure = create_shuffled_indices(n, params.random_seed);
        let averaging = averaging_ctr_permutation(n, 1, params.random_seed);
        assert_ne!(
            structure, averaging,
            "structure and averaging permutations coincide at (n={n}, seed={}) — \
             the fixture cannot discriminate pitfall #2",
            params.random_seed
        );

        // DEVICE fit — driven through `CountingGpu` so the fit's commitment to the
        // device is *counted*, not inferred (DCTR-19). The shape assertions below
        // (oblivious tree count, no non-symmetric/Region trees) are kept as a
        // grow-policy guard only; see this file's module doc for why they are not
        // device evidence.
        let gpu = CountingGpu { inner: GpuBackend::default(), grown: Cell::new(0) };
        let (trained, baked) = train_cat(
            &gpu,
            &columns,
            &borders,
            &cat_columns,
            &target,
            &[],
            &params,
            None,
        )
        .unwrap_or_else(|e| panic!("device CTR train failed: {e:?}"));
        assert_eq!(trained.oblivious_trees.len(), params.iterations);
        assert!(trained.non_symmetric_trees.is_empty() && trained.region_trees.is_empty());
        let n_ctr_splits: usize = trained
            .oblivious_trees
            .iter()
            .map(|t| t.ctr_splits.len())
            .sum();
        assert!(n_ctr_splits >= 1, "the trained model must contain ≥1 CTR split");

        let model = CbModel::from_trained(&trained, borders)
            .with_ctr_data(cb_model::CtrData::from_baked(&baked));
        let actual = cb_model::predict_raw_cat(&model, &columns, &cat_columns);
        assert_eq!(actual.len(), expected.len());
        let mut max_abs = 0.0_f64;
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            let abs = (a - e).abs();
            max_abs = max_abs.max(abs);
            assert!(
                abs <= 1e-5,
                "obj {i}: device CTR prediction {a} vs upstream {e} exceeds ≤1e-5 \
                 (|Δ|={abs:.3e})"
            );
        }
        println!(
            "[device-ctr-e2e] {} CTR splits; max |Δpred| = {max_abs:.3e} (bar 1e-5)",
            n_ctr_splits
        );

        // DCTR-19: the device-commitment assertion. Deliberately placed AFTER the
        // ≤1e-5 loop and its println, so that the §2.5 mutation check observes,
        // in one run, that the prediction bar STILL PASSES while this assertion
        // fails — the executable proof that the prediction bar alone (and the
        // oblivious-tree-count assertion above) cannot detect a CPU fallback.
        assert_eq!(
            gpu.grown.get(),
            params.iterations,
            "the fit did not commit to the device: {} of {} trees were grown on \
             device (the ≤1e-5 bar above passed regardless — R-8)",
            gpu.grown.get(),
            params.iterations
        );
        println!("[device-ctr-e2e] device grows = {}", gpu.grown.get());
    }
}

#[test]
fn device_ctr_fit_matches_upstream_predictions() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    device::run();
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    {
        let _ = ctr_params();
        eprintln!("SKIP device_ctr_fit_matches_upstream_predictions: needs rocm/cuda");
    }
}
