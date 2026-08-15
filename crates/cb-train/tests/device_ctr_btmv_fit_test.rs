//! DCTR-14 (T16): the **BinarizedTargetMeanValue** CTR device e2e oracle — a real
//! `train_cat(&CountingGpu, …)` fit with `simple_ctr = BinarizedTargetMeanValue:Prior=0.5`
//! on the frozen `ctr_device_btmv` fixture COMMITS to the device and its predictions
//! match upstream `catboost==1.2.10` at ≤1e-5, structure AND leaf values.
//!
//! BTMV is the fourth and last CPU-legal CTR type P1 admits. Unlike `Counter` it IS a
//! permutation-dependent read-before-increment online prefix
//! ([`cb_train::ECtrType::is_online_prefix`] is `true`), but unlike `Borders`/`Buckets` its
//! numerator is not a class COUNT: it is a running **float** `TCtrMeanHistory::Sum`
//! (`online_ctr.h:373`), which is why the device gives it its own accumulator
//! (`cb-backend`'s `btmv_ctr_prefix_kernel` / `launch_btmv_ctr_resident`, DCTR-12 / T14)
//! whose per-bucket history stays **f32-wide by parity contract**. The class-prefix
//! launcher's host guard still rejects the BTMV discriminant outright, so a BTMV column can
//! never silently receive the Borders numerator.
//!
//! # What this oracle CANNOT discriminate (read before trusting a green run)
//!
//! DCTR-13 / T15 measured that at binary classification the device BTMV column and the
//! device `Borders@0` column emit **identical cindex bins** — the addend is
//! `targetClass / 1 ∈ {0, 1}`, so `Sum == N[1]` and `Count == Total` exactly
//! (`online_ctr.cpp:467`/`:762`; `SIMPLE_CLASSES_COUNT == 2`,
//! `cb-train/src/ctr/online.rs:52`). ⇒ **a device path that routed this fit's columns
//! through the Borders numerator would produce the same predictions and pass the ≤1e-5 bar
//! below.** The bar is therefore an upstream-parity check, not a routing check.
//!
//! The routing is pinned separately and structurally:
//!
//! * every chosen CTR split's `ctr_type` is asserted to be `BinarizedTargetMeanValue` here
//!   (T07: the descriptor COUNT does not discriminate the type — `Borders:Prior=0.5` on
//!   this data also yields a single descriptor at `target_border_idx = 0`);
//! * `cb-backend`'s dispatch cannot confuse the two: `launch_btmv_ctr_resident` takes a
//!   `divisor` and returns a `ResidentCtrMean` (an f32 `sum` channel), while
//!   `launch_ordered_ctr_resident` takes `(ctr_type, target_border_idx)` and returns a
//!   `ResidentCtr` (an integer `good` channel) **and rejects `ctr_type == 2`**;
//! * the accumulator itself is proved against the CPU `online_mean_prefix` by DCTR-12's
//!   kernel self-oracle, whose f32-width detector runs at a synthetic `divisor = 3` because
//!   at binclf the two widths are bit-identical (PLAN §6 C-2, measured in `notes/T14.md`).
//!
//! # The prior is pinned on BOTH sides
//!
//! BTMV's DEFAULT prior set is the `{0/1, 0.5/1, 1/1}` TRIPLE
//! (`cat_feature_options.cpp:118-138`, mirrored by
//! `cb_train::ECtrType::default_priors`), not the fixture's single `Prior=0.5`. Omitting
//! `simple_ctr_priors` would therefore materialize THREE CTR columns against a fixture
//! whose `model.json` carries exactly one descriptor. The fixture pins
//! `simple_ctr = ["BinarizedTargetMeanValue:Prior=0.5"]` (asserted by
//! `cb-oracle/tests/ctr_device_btmv_fixture_smoke_test.rs`) and this file pins the matching
//! `simple_ctr_priors = vec![0.5]`, asserted rather than merely written.
//!
//! # Why the assertions are ordered this way (DCTR-19 / R-8, and T20's measurement)
//!
//! `oblivious_trees.len() == iterations` and the ≤1e-5 bar are BOTH satisfied by a
//! pure-CPU fallback — on `ctr_device_mixed`, forcing the gate closed (`&& false` in
//! `cb_train::boosting::ctr_types_are_device_covered`) makes the printed `max |Δpred|`
//! *improve* from `4.483e-11` (device) to `1.388e-17` (CPU) and collapses the runtime from
//! ~1.9 s to ~0.01 s; on `ctr_device_buckets` (T10) the two paths print the SAME delta.
//! **Neither a small delta nor a delta that differs from the CPU one is evidence of device
//! commitment.** The only sound evidence is counting the calls the boosting loop makes into
//! `Runtime::grow_tree_on_device`, which is what the `CountingGpu` wrapper below does.
//!
//! That assertion is placed deliberately **after** the ≤1e-5 loop and its `println!`, so a
//! single §2.5 mutation run shows both halves of the required evidence at once: the
//! prediction bar still passing while the commitment assertion fails. Do not "fix" the
//! ordering to fail fast.
//!
//! The `CountingGpu` wrapper is copied **verbatim** from the canonical copy at
//! `crates/cb-train/tests/device_ctr_gate_test.rs:82-138`; keep the copies in sync
//! (GLOBALS §2.2.6).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing, clippy::float_cmp)]

use cb_compute::{EScoreFunction, LeafMethod, Loss};
use cb_train::{BoostParams, EBootstrapType, EOverfittingDetectorType};

/// The `ctr_device_btmv` pinned config (mirrors its `config.json` params exactly). Only
/// `simple_ctr` differs from the `ctr_device_counter` / `ctr_device_mixed` templates.
///
/// `simple_ctr_priors = [0.5]` is pinned EXPLICITLY here to match the fixture's
/// `"simple_ctr": ["BinarizedTargetMeanValue:Prior=0.5"]`. BTMV's *default* prior set is
/// the three-element `{0/1, 0.5/1, 1/1}`, so omitting this line would materialize three CTR
/// columns instead of the fixture's one — a silent recipe divergence.
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
        simple_ctr: cb_train::ECtrType::BinarizedTargetMeanValue,
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
    // Atomics, not `Cell`: `cb_train::train` requires `R: Runtime + Sync` so the
    // fit can run inside a `thread_count`-sized rayon pool.
    use std::sync::atomic::{AtomicUsize, Ordering};
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
        pub grown: AtomicUsize,
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
                self.grown.fetch_add(1, Ordering::Relaxed);
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
            .join("ctr_device_btmv")
            .join(rel)
    }

    /// The largest number of documents sharing one categorical bucket.
    ///
    /// T04's non-vacuity dial, inherited through T14/T15: `calc_normalization`'s `norm` and
    /// the CTR denominator `count + 1` COINCIDE at `count == 1`, and a BTMV accumulator
    /// whose per-bucket history never advances past its second document cannot separate
    /// "read before increment" from "read after". The device BTMV statistic is only
    /// meaningfully exercised when some bucket holds ≥3 documents.
    fn max_bucket_occupancy(codes: &[i32]) -> usize {
        let mut counts: Vec<usize> = Vec::new();
        for &c in codes {
            let idx = usize::try_from(c).unwrap_or(0);
            if counts.len() <= idx {
                counts.resize(idx + 1, 0);
            }
            counts[idx] += 1;
        }
        counts.into_iter().max().unwrap_or(0)
    }

    pub fn run() {
        let x: Array2<f32> = read_npy(fixture("X.npy")).expect("X.npy loads");
        let columns: Vec<Vec<f32>> = (0..x.ncols()).map(|fi| x.column(fi).to_vec()).collect();
        let borders_arr: Array2<f64> = read_npy(fixture("borders.npy")).expect("borders.npy");
        let borders: Vec<Vec<f64>> =
            (0..borders_arr.nrows()).map(|r| borders_arr.row(r).to_vec()).collect();
        let cat: Array1<i32> = read_npy(fixture("X_cat.npy")).expect("X_cat.npy");
        let cat_codes: Vec<i32> = cat.iter().copied().collect();
        let cat_columns: Vec<Vec<String>> =
            vec![cat_codes.iter().map(|&c| stringify_int_category(i64::from(c))).collect()];
        let target = load_f64_vec(&fixture("y.npy")).unwrap();
        let expected = load_f64_vec(&fixture("predictions.npy")).unwrap();
        let n = target.len();
        let params = ctr_params();

        // The prior pin, asserted rather than merely written: the fixture's `config.json`
        // carries `"simple_ctr": ["BinarizedTargetMeanValue:Prior=0.5"]`, while BTMV's
        // DEFAULT prior set is the `{0, 0.5, 1}` triple.
        assert_eq!(
            params.simple_ctr_priors,
            vec![0.5],
            "the BTMV prior must be pinned EXPLICITLY to match the frozen fixture \
             (`BinarizedTargetMeanValue:Prior=0.5`); BTMV's DEFAULT prior set is the \
             three-element {{0/1, 0.5/1, 1/1}}, which would materialize three CTR columns \
             against a one-descriptor model"
        );

        // T04's ≥3-documents-per-bucket dial, asserted rather than assumed (the same guard
        // `cb-backend`'s BTMV kernel oracles carry). `norm` and the CTR denominator coincide
        // at `count == 1`, so a fixture whose buckets hold ≤2 documents exercises the BTMV
        // accumulator's distinguishing behaviour barely at all.
        let occupancy = max_bucket_occupancy(&cat_codes);
        assert!(
            occupancy >= 3,
            "the frozen fixture must drive ≥3 documents through some categorical bucket \
             (measured max occupancy {occupancy}) — below that the BTMV running (Sum, Count) \
             history never advances past the point where `calc_normalization`'s norm and the \
             CTR denominator coincide"
        );

        // Fixture-permutation-divergence guard (GLOBALS §2.2): the structure order (the
        // learn-set shuffle S) and the averaging order (S ∘ P_avg) must differ at this
        // fixture's (n, seed), or this oracle could not discriminate a structure-only leaf
        // gather (research pitfall #2). BTMV IS permutation dependent
        // (`is_online_prefix() == true`), so unlike Counter the two materializations produce
        // genuinely different CTR columns here.
        let structure = create_shuffled_indices(n, params.random_seed);
        let averaging = averaging_ctr_permutation(n, 1, params.random_seed);
        assert_ne!(
            structure, averaging,
            "structure and averaging permutations coincide at (n={n}, seed={}) — \
             the fixture cannot discriminate pitfall #2",
            params.random_seed
        );

        let gpu = CountingGpu { inner: GpuBackend::default(), grown: AtomicUsize::new(0) };
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
        .unwrap_or_else(|e| panic!("device BTMV CTR train failed: {e:?}"));
        assert_eq!(trained.oblivious_trees.len(), params.iterations);
        assert!(trained.non_symmetric_trees.is_empty() && trained.region_trees.is_empty());

        // Vacuity guard: a fit that chose no CTR split at all would exercise nothing this
        // task ships.
        let n_ctr_splits: usize =
            trained.oblivious_trees.iter().map(|t| t.ctr_splits.len()).sum();
        assert!(n_ctr_splits >= 1, "the trained model must contain ≥1 CTR split");

        // The chosen splits must be BTMV. This is the ONLY routing assertion in this file
        // that the ≤1e-5 bar cannot make for itself: DCTR-13 measured that BTMV and
        // `Borders@0` emit IDENTICAL bins at binclf, so a gate that kept routing the
        // class-prefix numerator would satisfy every prediction assertion below. Descriptor
        // COUNT does not discriminate the type either (COORDINATOR-FINDINGS T07).
        let btmv = cb_train::ECtrType::BinarizedTargetMeanValue.as_i8();
        for tree in &trained.oblivious_trees {
            for spec in &tree.ctr_splits {
                assert_eq!(
                    spec.ctr_type, btmv,
                    "a chosen CTR split carries ctr_type {} — this fit is configured \
                     `simple_ctr = BinarizedTargetMeanValue`, so every chosen CTR split must \
                     be BTMV ({btmv})",
                    spec.ctr_type
                );
                // BTMV does not binarize the target at all:
                // `target_border_count(BinarizedTargetMeanValue, 2) == 1`
                // (`ctr_helper.h:35-42`), so the numerator selector is structurally 0 on
                // every BTMV column.
                assert_eq!(
                    spec.target_border_idx, 0,
                    "BTMV emits exactly ONE column per (projection, prior); a non-zero \
                     numerator selector means the materialization loop changed shape"
                );
            }
        }
        println!("[device-ctr-btmv-e2e] {n_ctr_splits} BTMV CTR splits; max bucket occupancy = {occupancy}");

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
                "obj {i}: device BTMV CTR prediction {a} vs upstream {e} exceeds \
                 ≤1e-5 (|Δ|={abs:.3e})"
            );
        }
        println!("[device-ctr-btmv-e2e] max |Δpred| = {max_abs:.3e} (bar 1e-5)");

        // DCTR-14's second half. AFTER the ≤1e-5 loop on purpose — see this file's module
        // doc.
        assert_eq!(
            gpu.grown.load(Ordering::Relaxed),
            params.iterations,
            "the fit did not commit to the device: {} of {} trees were grown on \
             device (the ≤1e-5 bar above passed regardless — R-8)",
            gpu.grown.load(Ordering::Relaxed),
            params.iterations
        );
        println!("[device-ctr-btmv-e2e] device grows = {}", gpu.grown.load(Ordering::Relaxed));
    }
}

#[test]
fn device_ctr_btmv_fit_commits_and_matches_upstream() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    device::run();
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    {
        let _ = ctr_params();
        eprintln!("SKIP device_ctr_btmv_fit_commits_and_matches_upstream: needs rocm/cuda");
    }
}
