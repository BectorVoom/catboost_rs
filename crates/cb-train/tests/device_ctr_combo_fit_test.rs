//! DCTR-17 (T19): the COMBINATION-CTR device e2e oracle — a real
//! `train_cat(&CountingGpu, …)` fit on the two-cat `ctr_device_combo` fixture COMMITS to
//! the device and its predictions match upstream `catboost==1.2.10` at ≤1e-5.
//!
//! This is the end-to-end proof of DCTR-15/16/17: `ctr_types_are_device_covered` no longer
//! carries the projection-arity conjunct, `build_device_ctr_config` emits one `member_bins`
//! entry per projection member, and the device applies the SAME per-level combination
//! eligibility rule the CPU does (`resident_combination_eligible`, D-1/T17) plus the same
//! eligibility filter on `maxCount` (`resident_eligible_max_bucket_count`, D-2/T18). Before
//! that, a 2-member combination reached the device as ONE member's raw bucket column, so the
//! device scored the combination split from the wrong bins — wrong, not merely worse.
//!
//! The fixture's generator asserts its model genuinely contains a ≥2-member CTR
//! projection (its target carries an XOR interaction between the two cat columns that
//! neither explains alone), and this test re-asserts the same property on the trained
//! Rust model — a fit that silently degraded to simple projections cannot pass.
//!
//! It also inherits the GDC-12 bar: the structure and averaging permutations genuinely
//! diverge, so a structure-only leaf gather still fails here.
//!
//! # Why this test was `#[ignore]`d, and what un-ignoring it required (R-8)
//!
//! Until T19 the file carried `#[ignore = "FPP-11 ESCALATED …"]` whose rationale claimed the
//! fit "runs on the CPU grower and the arm-routing assertion below would fail". That was
//! **factually wrong**: the only routing assertions were `oblivious_trees.len() ==
//! params.iterations` and `non_symmetric_trees.is_empty() && region_trees.is_empty()`, both
//! of which the **CPU** oblivious grower satisfies. Run with `--ignored` the test PASSED, at
//! `max |Δpred| = 1.388e-17` in 0.01 s — the CPU-fallback fingerprint. It was the R-8
//! false-pass class in its purest form: a green e2e that measured the CPU path while its name
//! and its doc claimed the device.
//!
//! **`oblivious_trees.len() == iterations` is NOT a device-commit assertion.** The only one
//! that is, is [`device::CountingGpu`]'s `grown.get() == params.iterations` (GLOBALS §2.2.6),
//! and it is what T00's gate-pin migration named as the condition for re-opening the arity
//! conjunct. A small `max |Δpred|` is never evidence on its own: on this very fixture the CPU
//! fallback scores `1.388e-17` and the device `2.082e-17`, i.e. **falling back makes the
//! number BETTER**, and on `ctr_device_buckets`/`ctr_device_btmv` the two paths print the
//! *identical* delta. Only `grown` and the runtime (≈1.6 s device vs 0.01 s CPU) held across
//! every device e2e in this phase.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing, clippy::float_cmp)]

use cb_compute::{EScoreFunction, LeafMethod, Loss};
use cb_train::{BoostParams, EBootstrapType, EOverfittingDetectorType};

/// The `ctr_device_combo` pinned config (mirrors its `config.json` params).
/// The two deltas from `ctr_device_mixed` are `max_ctr_complexity` and `combinations_ctr`.
fn combo_params() -> BoostParams {
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
        max_ctr_complexity: 2,
        combinations_ctr: cb_train::ECtrType::Borders,
        combinations_ctr_priors: vec![0.5],
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

    use super::combo_params;
    use cb_backend::GpuBackend;
    use cb_compute::{
        DeviceGrownTree, DeviceTrainConfig, EScoreFunction, FamilyTreeArgs, Loss, Runtime,
    };
    use cb_core::CbResult;
    use cb_data::stringify_int_category;
    use cb_model::Model as CbModel;
    use cb_oracle::load_f64_vec;
    use cb_train::{averaging_ctr_permutation, create_shuffled_indices, train_cat};
    use ndarray::Array2;
    use ndarray_npy::read_npy;

    /// The device-commitment counter (GLOBALS §2.2.6). Copied **verbatim** from
    /// `crates/cb-train/tests/device_ctr_gate_test.rs` (the canonical copy) — keep in sync:
    /// every override forwards to `self.inner: GpuBackend` and only `grow_tree_on_device`
    /// counts, and only when it returns `Some` (a `None` is the device declining a tree).
    ///
    /// This wrapper is the ONLY thing in this file that can distinguish a device fit from a
    /// CPU-fallback fit. See the module doc for why the prediction delta cannot.
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
            .join("ctr_device_combo")
            .join(rel)
    }

    pub fn run() {
        let x: Array2<f32> = read_npy(fixture("X.npy")).expect("X.npy loads");
        let columns: Vec<Vec<f32>> = (0..x.ncols()).map(|fi| x.column(fi).to_vec()).collect();
        let borders_arr: Array2<f64> = read_npy(fixture("borders.npy")).expect("borders.npy");
        let borders: Vec<Vec<f64>> =
            (0..borders_arr.nrows()).map(|r| borders_arr.row(r).to_vec()).collect();
        // TWO cat columns — a combination projection needs at least two members.
        let cat: Array2<i32> = read_npy(fixture("X_cat.npy")).expect("X_cat.npy is [N,2]");
        assert_eq!(cat.ncols(), 2, "the combo fixture must ship two cat columns");
        let cat_columns: Vec<Vec<String>> = (0..cat.ncols())
            .map(|c| {
                cat.column(c)
                    .iter()
                    .map(|&v| stringify_int_category(i64::from(v)))
                    .collect()
            })
            .collect();
        let target = load_f64_vec(&fixture("y.npy")).unwrap();
        let expected = load_f64_vec(&fixture("predictions.npy")).unwrap();
        let n = target.len();
        let params = combo_params();

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

        // DEVICE fit, driven through the counting wrapper. The shape assertions below
        // (`oblivious_trees.len()`, the empty non-symmetric/region lists) are kept, but they
        // are NOT the device evidence — the CPU oblivious grower satisfies all of them (R-8).
        // `gpu.grown` is, and it is asserted at the END of this function; see there for why.
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

        // COMBINATION vacuity guard (T13's finding, generalised to this track): `≥1 CTR
        // split` is satisfied by a fit that chose only SIMPLE projections, which is exactly
        // the silent degradation DCTR-17 exists to exclude — and it is what the module doc
        // has always claimed this test re-asserts. Make it executable: at least one chosen
        // CTR split must carry a ≥2-member projection.
        let arities: Vec<usize> = trained
            .oblivious_trees
            .iter()
            .flat_map(|t| t.ctr_splits.iter())
            .map(|spec| spec.projection.cat_features().len())
            .collect();
        let n_combination_splits = arities.iter().filter(|&&a| a >= 2).count();
        assert!(
            n_combination_splits >= 1,
            "the trained model must contain ≥1 COMBINATION (≥2-member) CTR split — chosen \
             CTR-split arities were {arities:?}. A fit that silently degraded to simple \
             projections reproduces this fixture's predictions perfectly well, so the ≤1e-5 \
             bar below cannot see it."
        );

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
            "[device-ctr-combo-e2e] {n_ctr_splits} CTR splits ({n_combination_splits} of them \
             ≥2-member combinations); max |Δpred| = {max_abs:.3e} (bar 1e-5)"
        );

        // DEVICE COMMITMENT — deliberately AFTER the ≤1e-5 loop, not right after the fit.
        // Under the §2.5 mutation that closes the gate, the run then prints the passing
        // `max |Δpred|` line BEFORE panicking here, so a single mutated run yields both
        // halves of DCTR-17's completion evidence ("the fit fell back to the CPU AND the
        // prediction bar still passed"). Fail-fast ordering hides exactly that. Do not
        // "fix" this ordering. (Same rationale as `device_ctr_fit_test`, T20.)
        assert_eq!(
            gpu.grown.get(),
            params.iterations,
            "the combination-CTR fit must COMMIT to the device: expected {} device grows, \
             got {}. `oblivious_trees.len() == iterations` above does not say this — the CPU \
             oblivious grower satisfies it too (R-8), which is why this test was a false pass \
             for as long as it was `#[ignore]`d with a wrong rationale.",
            params.iterations,
            gpu.grown.get()
        );
        println!("[device-ctr-combo-e2e] device grows = {}", gpu.grown.get());
    }
}

/// DCTR-17 (T19): the combination-CTR device e2e, **un-ignored**.
///
/// The `#[ignore]` this test carried until T19 is gone, together with its rationale — which
/// was factually wrong and must NOT be restored verbatim if this ever has to be rolled back
/// (see the module doc: the assertions it claimed "would fail" are all satisfied by the CPU
/// grower, so the ignored test passed under `--ignored` while measuring the CPU path).
///
/// What replaces it is a real bar: [`device::CountingGpu`]'s `grown.get() == iterations`
/// alongside the ≤1e-5 upstream comparison. The measured device value on this fixture is
/// `≈2.082e-17`, deliberately DIFFERENT from the CPU fallback's `1.388e-17` — but note that
/// difference is corroboration, not the evidence (on two other fixtures in this phase the two
/// paths print the same delta). `grown` is the evidence.
#[test]
fn device_ctr_combo_fit_matches_upstream_predictions() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    device::run();
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    {
        let _ = combo_params();
        eprintln!("SKIP device_ctr_combo_fit_matches_upstream_predictions: needs rocm/cuda");
    }
}
