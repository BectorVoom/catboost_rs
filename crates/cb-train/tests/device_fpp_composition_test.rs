//! FPP-17 (T20): CROSS-GAP composition guards for this phase's gate relaxations.
//!
//! Waves 1–2 relaxed four independent clauses (bias, exact leaf, combination CTR, the
//! bootstrap × grow-policy cross-product). Each has its own oracle; what those cannot show
//! is that a relaxation did not WIDEN some unrelated exclusion as a side effect. That is
//! what this file is for.
//!
//! Every assertion is on the OBSERVABLE — did the device actually grow trees, counted by a
//! `CountingGpu` wrapper — never on a specific clause. A decline may legitimately fire at
//! either the host gate (`device_host_eligible`) or the backend session's family gate, and
//! a test pinned to one of them would break for the wrong reason when coverage moves
//! between the two.
//!
//! The prior phase's `device_gate_composition_test.rs` is left untouched; this is its
//! sibling for the FPP relaxations.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing, clippy::float_cmp)]

use cb_compute::{EScoreFunction, LeafMethod, Loss};
use cb_train::{
    BoostParams, EBootstrapType, EOverfittingDetectorType,
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
        extra: Default::default(),
    }
}

#[cfg(any(feature = "rocm", feature = "cuda"))]
mod device {
    // Atomics, not `Cell`: `cb_train::train` requires `R: Runtime + Sync` so the
    // fit can run inside a `thread_count`-sized rayon pool.
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::path::PathBuf;

    use super::base_params;
    use cb_backend::GpuBackend;
    use cb_compute::{
        DeviceGrownTree, DeviceTrainConfig, EScoreFunction, FamilyTreeArgs, Loss, Runtime,
    };
    use cb_core::CbResult;
    use cb_data::stringify_int_category;
    use cb_oracle::load_f64_vec;
    use cb_compute::LeafMethod;
    use cb_train::{train, train_cat, BoostParams, EBootstrapType};
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


    /// FPP-17.1 (POSITIVE): bias × weighted × CTR admit TOGETHER.
    ///
    /// All three are independently correct after this phase — the bias seeds the resident
    /// approx (FPP-01), the weighted der channel is the prior phase's, and simple-Borders
    /// CTR is covered — and nothing about their union is unimplemented. A union that
    /// declined would mean one relaxation had narrowed another's reach.
    pub fn bias_x_weighted_x_ctr_admits_together() {
        let (columns, cat_columns, target) = load_mixed();
        let borders = load_borders("borders_weighted.npy");
        let weights = load_f64_vec(&fixture("weights.npy")).unwrap();
        assert!(weights.iter().any(|&w| (w - 1.0).abs() > 1e-12), "weights must be non-uniform");
        let params = BoostParams {
            boost_from_average: true,
            ..base_params()
        };
        let gpu = CountingGpu { inner: GpuBackend::default(), grown: AtomicUsize::new(0) };
        let (trained, _baked) = train_cat(
            &gpu, &columns, &borders, &cat_columns, &target, &weights, &params, None,
        )
        .expect("bias × weighted × CTR device fit must succeed");
        assert_eq!(
            gpu.grown.load(Ordering::Relaxed),
            params.iterations,
            "bias × weighted × CTR must COMMIT to the device together — a decline means one \
             relaxation narrowed another's reach"
        );
        assert_eq!(trained.oblivious_trees.len(), params.iterations);
        println!("[bias×weighted×ctr] device grows = {}", gpu.grown.load(Ordering::Relaxed));
    }

    /// FPP-17.2: exact leaf × CTR still declines. FPP-06 admitted Exact for {Mae,
    /// Quantile} on the FLOAT path only; the backend's exact-leaf family gate still
    /// requires `ctr.is_none()`, and combining them was never implemented.
    pub fn exact_leaf_x_ctr_still_declines() {
        let (columns, cat_columns, target) = load_mixed();
        let borders = load_borders("borders.npy");
        let params = BoostParams {
            loss: Loss::Mae,
            leaf_method: LeafMethod::Exact,
            ..base_params()
        };
        let gpu = CountingGpu { inner: GpuBackend::default(), grown: AtomicUsize::new(0) };
        train_cat(&gpu, &columns, &borders, &cat_columns, &target, &[], &params, None)
            .expect("exact-leaf × CTR CPU fit must succeed");
        assert_eq!(
            gpu.grown.load(Ordering::Relaxed),
            0,
            "exact leaf × CTR must still decline to the CPU grower — FPP-06 widened the \
             leaf-method clause only, not the CTR family gate"
        );
    }

    /// FPP-17.3: exact leaf × sampling still declines. FPP-13 opened
    /// bootstrap × grow_policy, NOT bootstrap × leaf_method; the backend's exact-leaf
    /// family gate still requires `bootstrap_type == No`.
    pub fn exact_leaf_x_sampling_still_declines() {
        let (columns, _cat, target) = load_mixed();
        let borders = load_borders("borders.npy");
        let params = BoostParams {
            loss: Loss::Mae,
            leaf_method: LeafMethod::Exact,
            bootstrap_type: EBootstrapType::Bernoulli,
            subsample: 0.66,
            ..base_params()
        };
        let gpu = CountingGpu { inner: GpuBackend::default(), grown: AtomicUsize::new(0) };
        train(&gpu, &columns, &borders, &target, &[], &params, None)
            .expect("exact-leaf × sampling CPU fit must succeed");
        assert_eq!(
            gpu.grown.load(Ordering::Relaxed),
            0,
            "exact leaf × sampling must still decline — FPP-13 opened bootstrap × \
             grow_policy, not bootstrap × leaf_method"
        );
    }

    /// FPP-17.4: one-hot × CTR still declines (SPEC-OH-26, entirely untouched by this
    /// phase).
    ///
    /// **RETAINED BY DESIGN (DCTR-03) — not a P2/P3 boundary.** Every other negative in
    /// this phase's suite marks a coverage boundary a later phase will invert; this one
    /// does not. `SPEC.md` DCTR-03 keeps the `one_hot_bins.is_empty()` device conjunct
    /// (`crates/cb-train/src/boosting.rs`, the CTR arm of `device_host_eligible`)
    /// **indefinitely**, as defence in depth behind SPEC-OH-26, so that the day the CPU
    /// gains a three-way candidate union one-hot × CTR cannot silently reach the device
    /// untested. Do not read this test as a scheduled inversion and flip it.
    ///
    /// **It asserts the OBSERVABLE, not a layer.** Two independent layers reject the
    /// mixed pool — SPEC-OH-26's typed `CbError::Unsupported` in `train_inner` (which
    /// fires today, hence `expect_err`) and the device conjunct above (which fires if
    /// SPEC-OH-26 is ever relaxed) — and `grown == 0` is asserted alongside, so the test
    /// survives either layer moving. T21 measured both: with SPEC-OH-26 alone disabled
    /// the pool trains at `grown == 0`; with both disabled it commits at `grown == 5`
    /// (`.planning/plans/device-ctr-full-coverage/notes/T21.md` §3).
    ///
    /// This needs a GENUINELY MIXED pool — some cat columns routed one-hot AND some routed
    /// to CTR. `ctr_device_mixed/` has a single cat column, so any `one_hot_max_size` there
    /// produces an all-one-hot or an all-CTR pool and the case is vacuous (an all-one-hot
    /// pool legitimately commits, which is not what this guard is about). The two-cat
    /// `ctr_device_combo/` fixture (cardinalities 3 and 4) with `one_hot_max_size = 3`
    /// routes the first column one-hot and the second to CTR — the real mixed shape
    /// SPEC-OH-26 rejects.
    pub fn one_hot_x_ctr_still_declines() {
        let (columns, cat_columns, target, borders) = load_combo();
        assert_eq!(cat_columns.len(), 2, "the mixed shape needs two cat columns");
        let params = BoostParams {
            one_hot_max_size: 3,
            ..base_params()
        };
        let gpu = CountingGpu { inner: GpuBackend::default(), grown: AtomicUsize::new(0) };
        let err = train_cat(&gpu, &columns, &borders, &cat_columns, &target, &[], &params, None)
            .expect_err("a mixed one-hot/CTR pool must be REFUSED, not trained");
        let msg = format!("{err}");
        assert!(
            msg.contains("one-hot") && msg.contains("CTR"),
            "the refusal must name both routings: {msg}"
        );
        assert!(
            msg.contains("one_hot_max_size"),
            "the refusal must name the knob that resolves it: {msg}"
        );
        assert_eq!(
            gpu.grown.load(Ordering::Relaxed),
            0,
            "…and nothing may have been grown on the device before the refusal"
        );
        // SPEC-OH-26 rejects the mixed pool OUTRIGHT rather than declining to the CPU
        // grower — a stronger guarantee than this guard was written to expect, and the
        // one that actually holds. Recorded so a future relaxation to a silent CPU
        // fallback is a visible change, not an unnoticed weakening.
        println!("[one-hot×ctr] refused outright: {msg}");
    }

    /// Load the two-cat `ctr_device_combo/` fixture: `(float columns, cat columns, target,
    /// borders)`.
    #[allow(clippy::type_complexity)]
    fn load_combo() -> (Vec<Vec<f32>>, Vec<Vec<String>>, Vec<f64>, Vec<Vec<f64>>) {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("cb-oracle")
            .join("fixtures")
            .join("ctr_device_combo");
        let x: Array2<f32> = read_npy(dir.join("X.npy")).expect("combo X.npy");
        let columns: Vec<Vec<f32>> = (0..x.ncols()).map(|fi| x.column(fi).to_vec()).collect();
        let cat: Array2<i32> = read_npy(dir.join("X_cat.npy")).expect("combo X_cat.npy is [N,2]");
        let cat_columns: Vec<Vec<String>> = (0..cat.ncols())
            .map(|c| {
                cat.column(c)
                    .iter()
                    .map(|&v| stringify_int_category(i64::from(v)))
                    .collect()
            })
            .collect();
        let target = load_f64_vec(&dir.join("y.npy")).unwrap();
        let arr: Array2<f64> = read_npy(dir.join("borders.npy")).expect("combo borders.npy");
        let borders = (0..arr.nrows()).map(|r| arr.row(r).to_vec()).collect();
        (columns, cat_columns, target, borders)
    }

    /// FPP-17.5: multi-permutation × CTR still declines — the
    /// `learning_folds_for_cycle == 1` guard is untouched by FPP-09.
    ///
    /// **P3 WILL INVERT THIS (C-2), and only for the anchored `pc=4`/`seed=0` family
    /// (R-17).** Boundary marker, not a requirement: when P3 lands device structure-fold
    /// cycling the correct assertion becomes `grown == params.iterations` and this test
    /// must be FLIPPED, not defended. The sibling pin is
    /// `device_ctr_gate_test::multi_permutation_ctr_declines_to_device`; T21 re-ran and
    /// mutation-checked that one (`notes/T21.md` §3) rather than duplicating the work
    /// here.
    pub fn multi_permutation_x_ctr_still_declines() {
        let (columns, cat_columns, target) = load_mixed();
        let borders = load_borders("borders.npy");
        let params = BoostParams {
            permutation_count: 4,
            ..base_params()
        };
        let gpu = CountingGpu { inner: GpuBackend::default(), grown: AtomicUsize::new(0) };
        train_cat(&gpu, &columns, &borders, &cat_columns, &target, &[], &params, None)
            .expect("multi-permutation × CTR CPU fit must succeed");
        assert_eq!(
            gpu.grown.load(Ordering::Relaxed),
            0,
            "multi-permutation × CTR must still decline — the single-permutation guard is \
             untouched by the projection-arity work"
        );
    }

    /// FPP-17.6: `random_strength != 0` still declines. This exclusion has nothing to do
    /// with any clause this phase touched, so it is the control: if it moved, something
    /// widened the gate far beyond its brief.
    pub fn random_strength_still_declines() {
        let (columns, _cat, target) = load_mixed();
        let borders = load_borders("borders.npy");
        let params = BoostParams {
            random_strength: 1.0,
            ..base_params()
        };
        let gpu = CountingGpu { inner: GpuBackend::default(), grown: AtomicUsize::new(0) };
        train(&gpu, &columns, &borders, &target, &[], &params, None)
            .expect("random_strength CPU fit must succeed");
        assert_eq!(
            gpu.grown.load(Ordering::Relaxed),
            0,
            "random_strength != 0 must still decline — it is this suite's control clause"
        );
    }
}

#[test]
fn fpp17_bias_x_weighted_x_ctr_admits_together() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    device::bias_x_weighted_x_ctr_admits_together();
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    {
        let _ = base_params();
        eprintln!("SKIP fpp17_bias_x_weighted_x_ctr_admits_together: needs rocm/cuda");
    }
}

#[test]
fn fpp17_exact_leaf_x_ctr_still_declines() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    device::exact_leaf_x_ctr_still_declines();
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    eprintln!("SKIP fpp17_exact_leaf_x_ctr_still_declines: needs rocm/cuda");
}

#[test]
fn fpp17_exact_leaf_x_sampling_still_declines() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    device::exact_leaf_x_sampling_still_declines();
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    eprintln!("SKIP fpp17_exact_leaf_x_sampling_still_declines: needs rocm/cuda");
}

#[test]
fn fpp17_one_hot_x_ctr_still_declines() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    device::one_hot_x_ctr_still_declines();
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    eprintln!("SKIP fpp17_one_hot_x_ctr_still_declines: needs rocm/cuda");
}

#[test]
fn fpp17_multi_permutation_x_ctr_still_declines() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    device::multi_permutation_x_ctr_still_declines();
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    eprintln!("SKIP fpp17_multi_permutation_x_ctr_still_declines: needs rocm/cuda");
}

#[test]
fn fpp17_random_strength_still_declines() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    device::random_strength_still_declines();
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    eprintln!("SKIP fpp17_random_strength_still_declines: needs rocm/cuda");
}
