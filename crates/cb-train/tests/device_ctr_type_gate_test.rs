//! DCTR-11 (T13): the **P1/P3 boundary** — a Counter CTR fit with a non-empty
//! **eval set** routes to the CPU, whatever `counter_calc_method` says.
//!
//! # What this file pins, and at which layer
//!
//! Since DCTR-10 (T12) the device CTR **type** list admits `Counter`
//! (`cb_train::boosting::ctr_types_are_device_covered` names
//! `ECtrType::Counter`), so a Counter fit is genuinely device-eligible *as far as
//! the CTR type is concerned*. The decline pinned here therefore comes from a
//! **different layer**: the `&& eval_sets.is_empty()` conjunct of
//! `device_host_eligible` in `crates/cb-train/src/boosting.rs` (the whole-fit
//! host-eligibility conjunction; `boosting.rs:4620` at the time of writing —
//! locate it by the clause text, the line drifts). Nothing in the CTR type
//! list, the backend `ctr_covered` admission list, or the kernel dispatch is
//! involved.
//!
//! # `counter_calc_method = Full` is STRUCTURALLY MOOT on a device fit
//!
//! `Full`'s only effect anywhere in this crate is to assemble
//! `counter_full_eval_columns` — the per-cat-column concatenation
//! `eval[0].cat_columns[c] ++ eval[1].cat_columns[c] ++ …`, mirroring upstream's
//! learn-plus-every-test-set hash array (`online_ctr.cpp:714-729`) — which is
//! threaded into `materialize_ctr_feature` as `extra_cat_columns` and consumed by
//! `online_counter_column`'s `extra_bins`. That assembly lives in `train_inner`
//! immediately after the `counter_calc_skip_test` binding (`boosting.rs:4231-4249`
//! at the time of writing) and is **built purely out of `eval_sets`**: with
//! `eval_sets` empty it collapses to `Vec::new()` through the
//! `cols.iter().all(Vec::is_empty)` arm, i.e. **`Full ≡ SkipTest`**.
//!
//! And `eval_sets` are reachable only through `train_cat_with_eval_sets`
//! (PLAN §6 C-1): `train_cat`'s sixth argument is `weights`, **not** an eval-set
//! list, so a test written against `train_cat` asserts on a structurally-empty
//! eval-set slice and passes vacuously. Every fit here goes through
//! `train_cat_with_eval_sets`, and the two "no eval set" cells pass a genuinely
//! empty `&[]` **in the eval-set slot**.
//!
//! ⇒ **P1 ships no eval-widening device code, and needs none.** The two facts
//! together mean the device path can never observe `counter_calc_method`: when it
//! would matter (eval sets present) the fit is not on the device at all.
//!
//! # The 2×2, and why it is four cells rather than two
//!
//! The task's two required cells (`Full` + eval ⇒ CPU, `SkipTest` + no eval ⇒
//! device) differ in **two** factors at once, so on their own they cannot say
//! *which* factor causes the decline. The full `{SkipTest, Full} × {no eval set,
//! eval set}` square does:
//!
//! | | no eval set | eval set |
//! |---|---|---|
//! | `SkipTest` | **device** (`grown == iterations`) | **CPU** (`grown == 0`) |
//! | `Full` | **device** (`grown == iterations`) | **CPU** (`grown == 0`) |
//!
//! Reading down a column: `counter_calc_method` moves nothing (it is moot).
//! Reading across a row: the eval set moves everything (it is the layer). That is
//! the test-side half of the layer pin; the §2.5 mutation on
//! `&& eval_sets.is_empty()` (recorded in `notes/T13.md`) is the production-side
//! half.
//!
//! # P3 WILL INVERT THIS
//!
//! **The two `*_declines_to_cpu` cells are BOUNDARY PINS, not requirements.** P3
//! lands device eval-set support and the Counter `Full` widening; at that point
//! the correct assertion for both becomes `grown == params.iterations` and these
//! two tests must be **flipped**, not preserved. A boundary pin that does not say
//! it is a boundary reads to the next phase as a requirement, and someone will
//! defend it. The two `*_commits` cells are ordinary requirements and stay.
//!
//! # Why `grown`, and why the non-vacuity guards
//!
//! `oblivious_trees.len() == iterations` and any prediction bar are satisfied by a
//! pure-CPU fallback — measured three times in this phase, in both directions
//! (`notes/COORDINATOR-FINDINGS.md`, T20/T10/T12). The only sound routing evidence
//! is counting the calls the boosting loop makes into `Runtime::grow_tree_on_device`,
//! which is what the `CountingGpu` wrapper below does. A `grown == 0` assertion has
//! the mirror-image hazard: it also passes when the fit **never ran** or ran without
//! any CTR work at all. Every cell therefore additionally asserts that the fit
//! succeeded, grew all `iterations` trees, chose ≥1 CTR split, and that every chosen
//! CTR split really is `Counter` — so `grown == 0` can only mean "routed to the CPU".
//!
//! The `CountingGpu` wrapper is copied **verbatim** from the canonical copy at
//! `crates/cb-train/tests/device_ctr_gate_test.rs:82-138`; keep the copies in sync
//! (GLOBALS §2.2.6). This is the fifth copy in the tree.
//!
//! **T21 extends this file** with the remaining DCTR-03 surviving-clause boundary
//! pins (see PLAN §5.3's ownership table); T22 owns
//! `device_ctr_combo_types_diff_test.rs` and does not touch this file.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing, clippy::float_cmp)]

use cb_compute::{EScoreFunction, LeafMethod, Loss};
use cb_train::{BoostParams, CounterCalcMethod, EBootstrapType, EOverfittingDetectorType};

/// First row of the `ctr_device_counter` fixture used as the EVAL half.
///
/// **The LEARN side is the WHOLE frozen fixture** (all 64 rows), i.e. byte-for-byte
/// the learn set `device_ctr_counter_fit_test` (DCTR-10) proved commits to the
/// device with 5 Counter CTR splits and `max |Δpred| = 1.388e-17`. The eval half is
/// the fixture's trailing rows, carried purely so the eval-set slice is genuinely
/// non-empty and genuinely carries the categorical column `counter_calc_method =
/// Full` would widen the Counter tally with. **Nothing in this file compares
/// predictions or metrics**, so the eval rows overlapping the learn rows is
/// immaterial here — what matters is that the four cells differ in exactly the two
/// factors under test and in nothing else.
///
/// A genuinely disjoint 32-row learn half was tried first (the task text's
/// literal reading) and **rejected on measurement**: at 32 rows this recipe's
/// float splits beat every Counter CTR candidate, so three of the four cells grew
/// models with **zero** CTR splits and the non-vacuity guard in [`device::assert_route`]
/// fired. A CTR-routing pin whose fit contains no CTR split proves nothing. See
/// `.planning/plans/device-ctr-full-coverage/notes/T13.md` §Red for the verbatim
/// failure.
const EVAL_HALF_START: usize = 32;

/// The `ctr_device_counter` pinned config, parameterized on `counter_calc_method`
/// — otherwise identical to `device_ctr_counter_fit_test.rs`'s `ctr_params()`,
/// which is the configuration DCTR-10 proved commits to the device and reproduces
/// upstream at ≤1e-5.
///
/// `simple_ctr_priors = [0.5]` is pinned EXPLICITLY (matching the fixture's
/// `"simple_ctr": ["Counter:Prior=0.5"]`); Counter's *default* prior is `0/1`, so
/// omitting it is a silent recipe divergence (COORDINATOR-FINDINGS T06/T12).
/// Nothing in this file compares predictions, so nothing here could detect that —
/// which is exactly why the pin is written out rather than defaulted.
fn ctr_params(counter_calc_method: CounterCalcMethod) -> BoostParams {
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
        counter_calc_method,
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

    use super::{ctr_params, EVAL_HALF_START};
    use cb_backend::GpuBackend;
    use cb_compute::{
        DeviceGrownTree, DeviceTrainConfig, EScoreFunction, FamilyTreeArgs, Loss, Runtime,
    };
    use cb_core::CbResult;
    use cb_data::stringify_int_category;
    use cb_train::{
        averaging_ctr_permutation, create_shuffled_indices, train_cat_with_eval_sets,
        CounterCalcMethod, EvalSet,
    };
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

    /// The frozen `ctr_device_counter` fixture, loaded as a LEARN set (all rows —
    /// DCTR-10's exact learn set) plus an EVAL half (rows [`EVAL_HALF_START`]..).
    /// The fixture is read-only here — nothing is regenerated (GLOBALS §2.6).
    ///
    /// `borders` is the fixture's frozen float-border set: it is supplied to the
    /// fit, never recomputed from the rows, so the eval slice cannot perturb the
    /// quantization.
    struct Split {
        learn_columns: Vec<Vec<f32>>,
        borders: Vec<Vec<f64>>,
        learn_cat: Vec<Vec<String>>,
        learn_target: Vec<f64>,
        eval_columns: Vec<Vec<f32>>,
        eval_cat: Vec<Vec<String>>,
        eval_target: Vec<f64>,
    }

    fn load_split() -> Split {
        let x: Array2<f32> = read_npy(fixture("X.npy")).expect("X.npy loads");
        let borders_arr: Array2<f64> = read_npy(fixture("borders.npy")).expect("borders.npy");
        let borders: Vec<Vec<f64>> =
            (0..borders_arr.nrows()).map(|r| borders_arr.row(r).to_vec()).collect();
        let cat: Array1<i32> = read_npy(fixture("X_cat.npy")).expect("X_cat.npy");
        let target = cb_oracle::load_f64_vec(&fixture("y.npy")).unwrap();
        let n = target.len();
        assert!(
            EVAL_HALF_START > 0 && EVAL_HALF_START < n,
            "the eval half must be non-empty and must not be the whole fixture: \
             n={n}, EVAL_HALF_START={EVAL_HALF_START}"
        );

        let learn_columns: Vec<Vec<f32>> =
            (0..x.ncols()).map(|fi| x.column(fi).to_vec()).collect();
        let eval_columns: Vec<Vec<f32>> = (0..x.ncols())
            .map(|fi| x.column(fi).iter().skip(EVAL_HALF_START).copied().collect())
            .collect();
        let cat_strings: Vec<String> =
            cat.iter().map(|&c| stringify_int_category(i64::from(c))).collect();
        let eval_cat =
            vec![cat_strings.iter().skip(EVAL_HALF_START).cloned().collect::<Vec<_>>()];
        let learn_cat = vec![cat_strings];
        let eval_target = target.iter().skip(EVAL_HALF_START).copied().collect::<Vec<_>>();
        let learn_target = target;

        Split {
            learn_columns,
            borders,
            learn_cat,
            learn_target,
            eval_columns,
            eval_cat,
            eval_target,
        }
    }

    /// Which grower the fit is expected to route to.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Route {
        /// `grown == params.iterations` — every tree came off the device.
        Device,
        /// `grown == 0` — the fit fell back to the byte-unchanged CPU grower.
        Cpu,
    }

    /// Run one cell of the 2×2 and assert its route.
    ///
    /// Every cell trains through [`train_cat_with_eval_sets`] (PLAN §6 C-1), on
    /// the SAME learn set, with the SAME params except `counter_calc_method`, so
    /// the only two things that vary across the four cells are the two factors
    /// under test.
    pub fn assert_route(
        tag: &str,
        method: CounterCalcMethod,
        with_eval_set: bool,
        expect: Route,
    ) {
        let split = load_split();
        let params = ctr_params(method);
        assert_eq!(
            params.counter_calc_method, method,
            "the cell must actually carry the `counter_calc_method` it claims"
        );
        assert_eq!(
            params.simple_ctr_priors,
            vec![0.5],
            "the Counter prior must stay pinned to the frozen fixture's \
             `Counter:Prior=0.5`; Counter's DEFAULT prior is 0/1"
        );

        // Fixture-permutation-divergence guard (GLOBALS §2.2): the structure order
        // (the learn-set shuffle S) and the averaging order (S ∘ P_avg) must differ
        // at this half's (n, seed), or a structure-only leaf gather could not be
        // discriminated (research pitfall #2).
        let n_learn = split.learn_target.len();
        assert_ne!(
            create_shuffled_indices(n_learn, params.random_seed),
            averaging_ctr_permutation(n_learn, 1, params.random_seed),
            "structure and averaging permutations coincide at (n={n_learn}, seed={}) — \
             this split cannot discriminate pitfall #2",
            params.random_seed
        );

        // THE factor under test. `&[]` here is the EVAL-SET slot of
        // `train_cat_with_eval_sets`, not `train_cat`'s `weights` slot (C-1).
        let eval_sets: Vec<EvalSet> = if with_eval_set {
            vec![EvalSet {
                feature_values: &split.eval_columns,
                target: &split.eval_target,
                cat_columns: &split.eval_cat,
            }]
        } else {
            Vec::new()
        };
        assert_eq!(
            eval_sets.is_empty(),
            !with_eval_set,
            "the eval-set slice must be genuinely {} for this cell — a test that thinks it \
             passed an eval set but did not is exactly the vacuous pass C-1 warns about",
            if with_eval_set { "NON-EMPTY" } else { "EMPTY" }
        );
        if with_eval_set {
            assert_eq!(eval_sets[0].target.len(), split.eval_target.len());
            assert!(
                !eval_sets[0].cat_columns.is_empty()
                    && !eval_sets[0].cat_columns[0].is_empty(),
                "the eval set must carry the categorical column `counter_calc_method = Full` \
                 would widen the Counter tally with — an eval set without it could not \
                 exercise the P3 behaviour this boundary defers"
            );
        }

        let gpu = CountingGpu { inner: GpuBackend::default(), grown: Cell::new(0) };
        let (trained, _baked) = train_cat_with_eval_sets(
            &gpu,
            &split.learn_columns,
            &split.borders,
            &split.learn_cat,
            &split.learn_target,
            &[],
            &params,
            None,
            &eval_sets,
            None,
        )
        .unwrap_or_else(|e| panic!("[{tag}] Counter CTR train failed: {e:?}"));

        // Non-vacuity, in order of increasing specificity. `grown == 0` is also
        // what a fit that never ran, or ran with no CTR work at all, would report;
        // these four assertions are what make `Route::Cpu` mean "routed to the CPU"
        // rather than "nothing happened".
        assert_eq!(
            trained.oblivious_trees.len(),
            params.iterations,
            "[{tag}] the fit must have grown all {} trees",
            params.iterations
        );
        assert!(trained.non_symmetric_trees.is_empty() && trained.region_trees.is_empty());
        let n_ctr_splits: usize =
            trained.oblivious_trees.iter().map(|t| t.ctr_splits.len()).sum();
        assert!(
            n_ctr_splits >= 1,
            "[{tag}] the trained model must contain ≥1 CTR split — without one this cell \
             says nothing about CTR routing"
        );
        let counter = cb_train::ECtrType::Counter.as_i8();
        for tree in &trained.oblivious_trees {
            for spec in &tree.ctr_splits {
                assert_eq!(
                    spec.ctr_type, counter,
                    "[{tag}] a chosen CTR split carries ctr_type {} — this fit is configured \
                     `simple_ctr = Counter`, and the DEVICE TYPE LIST ADMITS Counter since \
                     DCTR-10 (T12), which is precisely why a decline here must come from the \
                     eval-set layer and not from the type list",
                    spec.ctr_type
                );
            }
        }

        let grown = gpu.grown.get();
        println!(
            "[device-ctr-type-gate] {tag}: method={method:?} eval_sets={} \
             ctr_splits={n_ctr_splits} grown={grown}/{}",
            eval_sets.len(),
            params.iterations
        );
        match expect {
            Route::Device => assert_eq!(
                grown, params.iterations,
                "[{tag}] the fit did not commit to the device: {grown} of {} trees were grown \
                 on device. With NO eval set, `counter_calc_method` is structurally moot \
                 (`Full` collapses to `SkipTest`) and a Counter fit is device-covered (DCTR-10)",
                params.iterations
            ),
            Route::Cpu => assert_eq!(
                grown, 0,
                "[{tag}] the fit committed {grown} of {} trees to the DEVICE, but a non-empty \
                 eval set must decline via the `eval_sets.is_empty()` conjunct of \
                 `device_host_eligible` (boosting.rs). P1 ships no device eval-set support and \
                 no Counter `Full` widening; if this is now intentional, this is the P3 \
                 inversion and the assertion must be FLIPPED, not deleted",
                params.iterations
            ),
        }
    }
}

/// **P3 WILL INVERT THIS.** Boundary pin, not a requirement — see the module doc.
///
/// `counter_calc_method = Full` **and** a non-empty eval set ⇒ the fit routes to
/// the CPU, via `device_host_eligible`'s `eval_sets.is_empty()` conjunct. This is
/// the cell that proves P1 ships no eval-widening device code: the one
/// configuration in which `Full` is observable at all is also the one that never
/// reaches the device.
#[test]
fn counter_full_with_eval_set_declines_to_cpu() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    device::assert_route(
        "full+eval",
        CounterCalcMethod::Full,
        true,
        device::Route::Cpu,
    );
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    {
        let _ = ctr_params(CounterCalcMethod::Full);
        eprintln!("SKIP counter_full_with_eval_set_declines_to_cpu: needs rocm/cuda");
    }
}

/// **P3 WILL INVERT THIS.** Boundary pin, not a requirement — see the module doc.
///
/// The layer isolation, test side: the eval set alone declines the fit even under
/// the default `SkipTest`. Together with `counter_full_without_eval_set_commits`
/// this says the decline is caused by the eval set and **not** by
/// `counter_calc_method`.
#[test]
fn counter_skiptest_with_eval_set_declines_to_cpu() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    device::assert_route(
        "skiptest+eval",
        CounterCalcMethod::SkipTest,
        true,
        device::Route::Cpu,
    );
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    {
        let _ = ctr_params(CounterCalcMethod::SkipTest);
        eprintln!("SKIP counter_skiptest_with_eval_set_declines_to_cpu: needs rocm/cuda");
    }
}

/// The complement required by DCTR-11: without an eval set the very same learn
/// half, params and fixture **do** commit to the device. Without this cell the two
/// negative tests above could pass because the split data, the params or the
/// fixture were broken rather than because of the routing decision.
#[test]
fn counter_skiptest_without_eval_set_commits() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    device::assert_route(
        "skiptest+noeval",
        CounterCalcMethod::SkipTest,
        false,
        device::Route::Device,
    );
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    {
        let _ = ctr_params(CounterCalcMethod::SkipTest);
        eprintln!("SKIP counter_skiptest_without_eval_set_commits: needs rocm/cuda");
    }
}

/// The layer isolation's other half: `Full` **without** an eval set commits to the
/// device exactly like `SkipTest` does — the executable form of "`Full ≡ SkipTest`
/// when `eval_sets` is empty", and therefore of "`counter_calc_method` is
/// structurally moot on a device fit".
#[test]
fn counter_full_without_eval_set_commits() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    device::assert_route(
        "full+noeval",
        CounterCalcMethod::Full,
        false,
        device::Route::Device,
    );
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    {
        let _ = ctr_params(CounterCalcMethod::Full);
        eprintln!("SKIP counter_full_without_eval_set_commits: needs rocm/cuda");
    }
}
