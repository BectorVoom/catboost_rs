//! DCTR-10 (T12): the **Counter** CTR device e2e oracle — a real
//! `train_cat(&CountingGpu, …)` fit with `simple_ctr = Counter:Prior=0.5` on the
//! frozen `ctr_device_counter` fixture COMMITS to the device and its predictions
//! match upstream `catboost==1.2.10` at ≤1e-5, structure AND leaf values.
//!
//! Counter is the FIRST admitted CTR type that is **not** an online class prefix:
//! its numerator is the whole-set bucket total and its denominator the CONSTANT
//! MAX bucket total (`online_ctr.cpp:503-562`, `:934-936`), so it is
//! permutation-INdependent (`IsPermutationDependentCtrType(Counter) == false`,
//! `ctr_type.cpp:43-56`). That is why the device path routes it through its own
//! `launch_counter_ctr_resident` (DCTR-09 / T11) — which takes neither a
//! permutation nor a target class — instead of the class-prefix kernel, whose
//! host guard rejects `ctr_type == Counter` outright.
//!
//! # The prior is pinned on BOTH sides, and this oracle CANNOT police it
//!
//! Upstream's DEFAULT Counter prior is `0/1`, not `0.5/1`
//! (`cat_feature_options.cpp:118-138`, mirrored by
//! `cb_train::ctr::ECtrType::default_priors`). The fixture pins
//! `simple_ctr = ["Counter:Prior=0.5"]` (asserted by
//! `cb-oracle/tests/ctr_device_counter_fixture_smoke_test.rs`) and this file pins
//! the matching `simple_ctr_priors = vec![0.5]` below.
//!
//! **That pin is the only guard.** T06 measured on this fixture's data that
//! `Prior=0.5`, the default `0/1` and `Prior=3` all produce **bit-identical**
//! upstream predictions — upstream compensates by shifting the CTR border
//! (8.999999 at prior 0.5 and 0, 3.999999 at prior 3), and with 16 CTR buckets
//! over integer counts the shifted value never crosses a bin boundary. So the
//! ≤1e-5 bar below would stay green under a Rust-side prior mismatch on THIS
//! fixture. Do not delete or "simplify" the explicit `simple_ctr_priors` pin on
//! the strength of a passing run.
//!
//! # Why the assertions are ordered this way (DCTR-19 / R-8, and T20's measurement)
//!
//! `oblivious_trees.len() == iterations` and the ≤1e-5 bar are BOTH satisfied by a
//! pure-CPU fallback — on the sibling `ctr_device_mixed` fixture, forcing the gate
//! closed (`&& false` in `cb_train::boosting::ctr_types_are_device_covered`) makes
//! the printed `max |Δpred|` *improve* from `4.483e-11` (device) to `1.388e-17`
//! (CPU) and collapses the runtime from ~1.9 s to ~0.01 s; on `ctr_device_buckets`
//! (T10) the two paths print the SAME delta. **Neither a small delta nor a
//! delta that differs from the CPU one is evidence of device commitment.** The
//! only sound evidence is counting the calls the boosting loop makes into
//! `Runtime::grow_tree_on_device`, which is what the `CountingGpu` wrapper below
//! does.
//!
//! That assertion is placed deliberately **after** the ≤1e-5 loop and its
//! `println!`, so a single §2.5 mutation run shows both halves of the required
//! evidence at once: the prediction bar still passing while the commitment
//! assertion fails. Do not "fix" the ordering to fail fast.
//!
//! The `CountingGpu` wrapper is copied **verbatim** from the canonical copy at
//! `crates/cb-train/tests/device_ctr_gate_test.rs:82-138`; keep the copies in sync
//! (GLOBALS §2.2.6).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing, clippy::float_cmp)]

use cb_compute::{EScoreFunction, LeafMethod, Loss};
use cb_train::{BoostParams, EBootstrapType, EOverfittingDetectorType};

/// The `ctr_device_counter` pinned config (mirrors its `config.json` params
/// exactly). Only `simple_ctr` differs from the `ctr_device_mixed` template in
/// `device_ctr_gate_test.rs:24-66`.
///
/// `simple_ctr_priors = [0.5]` is pinned EXPLICITLY here to match the fixture's
/// `"simple_ctr": ["Counter:Prior=0.5"]`. Counter's *default* prior is `0/1`, so
/// omitting this line would be a silent recipe divergence — and, per this
/// module's doc, one the ≤1e-5 comparison could not detect on this fixture.
///
/// `counter_calc_method` is the canonical `SkipTest` default (`boosting.rs:511-513`,
/// matching the fixture's `"counter_calc_method": "SkipTest"`). It widens the
/// counted sample range to the eval sets when `Full`, which is structurally moot
/// here — this fit has no eval set at all (T13 pins that boundary).
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
        simple_ctr: cb_train::ECtrType::Counter,
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
            .join("ctr_device_counter")
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

        // The prior pin, asserted rather than merely written (T06's finding): the
        // fixture's `config.json` carries `"simple_ctr": ["Counter:Prior=0.5"]`,
        // and a mismatch here is invisible to the ≤1e-5 bar on this data.
        assert_eq!(
            params.simple_ctr_priors,
            vec![0.5],
            "the Counter prior must be pinned EXPLICITLY to match the frozen fixture \
             (`Counter:Prior=0.5`); Counter's DEFAULT prior is 0/1, and on this fixture \
             upstream compensates a prior change by shifting the CTR border, so the ≤1e-5 \
             comparison below cannot detect the mismatch"
        );

        // Fixture-permutation-divergence guard (GLOBALS §2.2): the structure
        // order (the learn-set shuffle S) and the averaging order (S ∘ P_avg)
        // must differ at this fixture's (n, seed), or this oracle could not
        // discriminate a structure-only leaf gather (research pitfall #2).
        //
        // NOTE the Counter-specific subtlety: the Counter STATISTIC is itself
        // permutation independent, so the two materializations produce equal CTR
        // columns. What the guard still buys is the leaf-value gather — the
        // averaging half feeds `assign_leaf_over_ctr_columns`, and the float
        // splits of this fit are permutation sensitive.
        let structure = create_shuffled_indices(n, params.random_seed);
        let averaging = averaging_ctr_permutation(n, 1, params.random_seed);
        assert_ne!(
            structure, averaging,
            "structure and averaging permutations coincide at (n={n}, seed={}) — \
             the fixture cannot discriminate pitfall #2",
            params.random_seed
        );

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
        .unwrap_or_else(|e| panic!("device Counter CTR train failed: {e:?}"));
        assert_eq!(trained.oblivious_trees.len(), params.iterations);
        assert!(trained.non_symmetric_trees.is_empty() && trained.region_trees.is_empty());

        // Vacuity guard: a fit that chose no CTR split at all would exercise
        // nothing this task ships.
        let n_ctr_splits: usize =
            trained.oblivious_trees.iter().map(|t| t.ctr_splits.len()).sum();
        assert!(n_ctr_splits >= 1, "the trained model must contain ≥1 CTR split");

        // The chosen splits must be COUNTER. Descriptor COUNT does not
        // discriminate the CTR type (COORDINATOR-FINDINGS T07: a `Borders` or
        // `Buckets` model on this data also yields one descriptor), so the type is
        // asserted explicitly — otherwise a gate that silently kept routing the
        // class-prefix numerator would satisfy every other assertion here.
        let counter = cb_train::ECtrType::Counter.as_i8();
        for tree in &trained.oblivious_trees {
            for spec in &tree.ctr_splits {
                assert_eq!(
                    spec.ctr_type, counter,
                    "a chosen CTR split carries ctr_type {} — this fit is configured \
                     `simple_ctr = Counter`, so every chosen CTR split must be Counter ({counter})",
                    spec.ctr_type
                );
                // Counter does not binarize the target at all:
                // `target_border_count(Counter, 2) == 1` (`ctr_helper.h:35-42`), so
                // the numerator selector is structurally 0 on every Counter column.
                assert_eq!(
                    spec.target_border_idx, 0,
                    "Counter emits exactly ONE column per (projection, prior); a non-zero \
                     numerator selector means the materialization loop changed shape"
                );
            }
        }
        println!("[device-ctr-counter-e2e] {n_ctr_splits} Counter CTR splits");

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
                "obj {i}: device Counter CTR prediction {a} vs upstream {e} exceeds \
                 ≤1e-5 (|Δ|={abs:.3e})"
            );
        }
        println!("[device-ctr-counter-e2e] max |Δpred| = {max_abs:.3e} (bar 1e-5)");

        // DCTR-10's second half. AFTER the ≤1e-5 loop on purpose — see this
        // file's module doc.
        assert_eq!(
            gpu.grown.get(),
            params.iterations,
            "the fit did not commit to the device: {} of {} trees were grown on \
             device (the ≤1e-5 bar above passed regardless — R-8)",
            gpu.grown.get(),
            params.iterations
        );
        println!("[device-ctr-counter-e2e] device grows = {}", gpu.grown.get());
    }
}

#[test]
fn device_ctr_counter_fit_commits_and_matches_upstream() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    device::run();
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    {
        let _ = ctr_params();
        eprintln!("SKIP device_ctr_counter_fit_commits_and_matches_upstream: needs rocm/cuda");
    }
}
