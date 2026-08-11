//! `T22-OBS-1` — a CTR fit's **CTR-FREE** trees must estimate their leaves from the SAME
//! derivative trajectory on the device as on the CPU.
//!
//! # The defect this file exists to detect
//!
//! A covered CTR fit runs **two** derivative trajectories by construction
//! (`gpu_runtime/session.rs`, GDC-10 / T10 §1):
//!
//! * the **resident** device approx/der1, advanced over the STRUCTURE partition — the CPU's
//!   learning-fold-0 approx (`UpdateLearningFold`), which drives the next tree's split SEARCH;
//! * the caller's **main** host approx, advanced over the AVERAGING partition — the trajectory
//!   the LEAF VALUES must be estimated from (`boosting.rs`'s `leaf_value_leaf_of` +
//!   `lv_weighted_der1`).
//!
//! The device's returned leaf values are reconciled onto the second trajectory by the
//! averaging-permutation gather at the end of `grow_tree_on_device`. That gather used to be
//! gated on *"this tree chose ≥1 CTR split"* — so a CTR fit's tree that happened to pick only
//! FLOAT splits silently returned the RESIDENT (learning-fold) estimate instead. Same splits,
//! same partition, **different der source** ⇒ a ~1e-3 leaf-value divergence against the CPU
//! grower, which then contaminates every later tree through the main approx.
//!
//! Measured on `ctr_device_combo` at the fixture's own `simple_ctr = combinations_ctr =
//! Borders`, merely run to 30 iterations, BEFORE the fix:
//!
//! ```text
//! trees  0..22   1e-17 … 2.4e-17   every tree carries ≥1 CTR split
//! tree   23      7.824e-4          ctr_splits == 0   ◀
//! tree   24      1.269e-5          (contaminated by 23 via the main approx)
//! tree   25      1.223e-3          ctr_splits == 0   ◀
//! trees  26,27   1.30e-5 / 1.27e-5 (contaminated)
//! tree   28      1.943e-3          ctr_splits == 0   ◀
//! tree   29      1.296e-3          ctr_splits == 0   ◀
//! ```
//!
//! AFTER the fix every tree, CTR-carrying and CTR-free alike, agrees to ≤ 1.4e-17.
//!
//! # Why no existing test caught it
//!
//! **Every committed device CTR fixture stops at 5 iterations**, and on this corpus the first
//! CTR-free tree is #23. A short-horizon differential cannot observe the branch at all. This
//! file's load-bearing guard is therefore not the leaf comparison — it is
//! [`device::run`]'s assertion that **both arms actually contain ≥1 CTR-free tree**. Without
//! that guard, lowering `iterations` back to 5 would turn this file green *and vacuous*, which
//! is exactly the state the whole device CTR suite was in. NEVER weaken it; if the corpus or
//! the search changes so that 30 iterations no longer produce a CTR-free tree, raise the
//! horizon rather than dropping the guard.
//!
//! # Relationship to `device_ctr_combo_types_diff_test`
//!
//! T22's differential is the CTR-type × combination surface and deliberately sits *below* the
//! first CTR-free tree (its module doc says so, and it prints its CTR-free tree count). This
//! file is the complementary one: same two-arm harness, one CTR configuration, run PAST that
//! boundary. It compares split sequences too, because a leaf-value comparison over two
//! different trees says nothing.
//!
//! GLOBALS §2.2: five allow-attrs; everything device-touching inside
//! `#[cfg(any(feature = "rocm", feature = "cuda"))] mod device`; no `use cb_backend::CpuBackend`
//! (not compiled under `rocm`); SKIP by printing on cpu/wgpu rather than `#[ignore]`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp
)]

use cb_compute::{EScoreFunction, LeafMethod, Loss};
use cb_train::{BoostParams, EBootstrapType, ECtrType, EOverfittingDetectorType};

/// The `ctr_device_combo` params — the SHIPPED configuration (`simple_ctr` and
/// `combinations_ctr` both `Borders`, prior `0.5`), with only `iterations` raised so the fit
/// reaches its first CTR-free tree. The frozen `X` / `X_cat` / `y` / `borders` are byte-
/// untouched (R-12); nothing about the fixture is tuned to make this pass.
///
/// Both prior lists are pinned explicitly for the same reason T22 pins them: the default prior
/// list is per-type, so an implicit list would silently change the materialized CTR column
/// count if the type ever changed.
fn ctr_free_params(iterations: usize) -> BoostParams {
    BoostParams {
        loss: Loss::Logloss,
        iterations,
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
        simple_ctr: ECtrType::Borders,
        simple_ctr_priors: vec![0.5],
        counter_calc_method: cb_train::counter_calc_method_default(),
        boosting_type: cb_train::boosting_type_default(),
        max_ctr_complexity: 2,
        combinations_ctr: ECtrType::Borders,
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

    use super::ctr_free_params;
    use cb_backend::GpuBackend;
    use cb_compute::{
        Derivatives, DeviceGrownTree, DeviceTrainConfig, EScoreFunction, FamilyTreeArgs, Loss,
        Runtime,
    };
    use cb_core::CbResult;
    use cb_data::stringify_int_category;
    use cb_train::{averaging_ctr_permutation, create_shuffled_indices, train_cat};
    use ndarray::Array2;
    use ndarray_npy::read_npy;

    /// The device-commitment counter (GLOBALS §2.2.6). Copied **verbatim** from
    /// `crates/cb-train/tests/device_ctr_gate_test.rs` (the canonical copy) — keep in sync:
    /// every override forwards to `self.inner: GpuBackend` and only `grow_tree_on_device`
    /// counts, and only when it returns `Some` (a `None` is the device declining a tree).
    /// NINTH copy.
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
        ) -> CbResult<Derivatives> {
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

    /// The CPU reference arm. Overrides **only** `compute_gradients`, forwarding to a real
    /// `GpuBackend` so BOTH arms consume bit-identical derivatives and the differential
    /// isolates the GROWER rather than the gradient kernel. Every device-seam method inherits
    /// the `cb_compute::Runtime` trait default (`begin_device_training → Ok(false)`,
    /// `grow_tree_on_device → Ok(None)`) — the `device_nonsym_fit_test.rs` precedent.
    struct CpuRefRuntime {
        inner: GpuBackend,
    }

    impl Runtime for CpuRefRuntime {
        fn compute_gradients(
            &self,
            loss: &Loss,
            approx: &[f64],
            target: &[f64],
            dim: usize,
        ) -> CbResult<Derivatives> {
            self.inner.compute_gradients(loss, approx, target, dim)
        }
        // NOTHING else is overridden — see the doc above.
    }

    /// The same counting wrapper shape as [`CountingGpu`], wrapping [`CpuRefRuntime`], so the
    /// reference arm's `grown == 0` / `accepted_begins == 0` are OBSERVATIONS rather than
    /// assumptions. Without it this file would be a device-vs-device tautology (the R-8 class).
    pub struct CountingCpu {
        inner: CpuRefRuntime,
        pub grown: Cell<usize>,
        pub begins: Cell<usize>,
        pub accepted_begins: Cell<usize>,
    }

    impl Runtime for CountingCpu {
        fn compute_gradients(
            &self,
            loss: &Loss,
            approx: &[f64],
            target: &[f64],
            dim: usize,
        ) -> CbResult<Derivatives> {
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
            self.begins.set(self.begins.get() + 1);
            let accepted = self.inner.begin_device_training(
                loss, depth, plain, fold_count, score_function, bins, weight, n, n_features,
                n_bins, lr, scaled_l2, config,
            )?;
            if accepted {
                self.accepted_begins.set(self.accepted_begins.get() + 1);
            }
            Ok(accepted)
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
    }

    fn fixture(rel: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("cb-oracle")
            .join("fixtures")
            .join("ctr_device_combo")
            .join(rel)
    }

    /// The full identity of a chosen CTR split — the same key `assign_leaf_over_ctr_columns`
    /// uses. `shift`/`scale` are excluded: prior-derived, so they add no discrimination.
    #[derive(Debug, Clone, PartialEq)]
    struct CtrSig {
        projection: Vec<usize>,
        ctr_type: i8,
        prior_num: f64,
        prior_denom: f64,
        target_border_idx: usize,
        border: f64,
    }

    fn ctr_sigs(tree: &cb_train::ObliviousTree) -> Vec<CtrSig> {
        tree.ctr_splits
            .iter()
            .map(|s| CtrSig {
                projection: s.projection.cat_features().to_vec(),
                ctr_type: s.ctr_type,
                prior_num: s.prior_num,
                prior_denom: s.prior_denom,
                target_border_idx: s.target_border_idx,
                border: s.border,
            })
            .collect()
    }

    /// The device-vs-CPU leaf bar for this file.
    ///
    /// It is deliberately far tighter than T22's D-07 `1e-4`: both arms compute their leaf
    /// values by the SAME `calc_average(Σ w·der1, Σ w, l2)` over the SAME partition from the
    /// SAME derivative trajectory, so the only residual is f64 summation order. Measured max
    /// over 30 trees on `gfx1151`: **1.4e-17**. The defect this file detects was 7.8e-4 —
    /// eight orders above this bar and eight orders above the measurement, so the bar
    /// discriminates without being flaky.
    const LEAF_EPS: f64 = 1e-9;

    pub fn run(iterations: usize) {
        let x: Array2<f32> = read_npy(fixture("X.npy")).expect("X.npy loads");
        let columns: Vec<Vec<f32>> = (0..x.ncols()).map(|fi| x.column(fi).to_vec()).collect();
        let borders_arr: Array2<f64> = read_npy(fixture("borders.npy")).expect("borders.npy");
        let borders: Vec<Vec<f64>> =
            (0..borders_arr.nrows()).map(|r| borders_arr.row(r).to_vec()).collect();
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
        let target = cb_oracle::load_f64_vec(&fixture("y.npy")).unwrap();
        let n = target.len();
        let params = ctr_free_params(iterations);

        // Fixture-permutation-divergence guard (GLOBALS §2.2). This one is load-bearing HERE
        // in a way it is not elsewhere: the whole defect is "which permutation's derivative
        // trajectory did the leaf estimate come from", so if the structure and averaging
        // permutations coincided the two trajectories would be the same and the file could
        // not distinguish them at all.
        let structure = create_shuffled_indices(n, params.random_seed);
        let averaging = averaging_ctr_permutation(n, 1, params.random_seed);
        assert_ne!(
            structure, averaging,
            "structure and averaging permutations coincide at (n={n}, seed={}) — the two \
             derivative trajectories would then be identical and this file could not \
             discriminate the leaf-estimation trajectory at all",
            params.random_seed
        );

        // ---- arm 1: the DEVICE grower ----
        let gpu = CountingGpu { inner: GpuBackend::default(), grown: Cell::new(0) };
        let (dev, _dev_baked) =
            train_cat(&gpu, &columns, &borders, &cat_columns, &target, &[], &params, None)
                .unwrap_or_else(|e| panic!("device CTR train failed: {e:?}"));

        // ---- arm 2: the CPU grower, same inputs, same gradients ----
        let cpu = CountingCpu {
            inner: CpuRefRuntime { inner: GpuBackend::default() },
            grown: Cell::new(0),
            begins: Cell::new(0),
            accepted_begins: Cell::new(0),
        };
        let (host, _host_baked) =
            train_cat(&cpu, &columns, &borders, &cat_columns, &target, &[], &params, None)
                .unwrap_or_else(|e| panic!("cpu CTR train failed: {e:?}"));

        let dev_ctr_free =
            dev.oblivious_trees.iter().filter(|t| t.ctr_splits.is_empty()).count();
        let cpu_ctr_free =
            host.oblivious_trees.iter().filter(|t| t.ctr_splits.is_empty()).count();
        let dev_ctr: usize = dev.oblivious_trees.iter().map(|t| t.ctr_splits.len()).sum();
        let cpu_ctr: usize = host.oblivious_trees.iter().map(|t| t.ctr_splits.len()).sum();

        // ---- the arms are really what they claim to be ----
        assert_eq!(
            gpu.grown.get(),
            params.iterations,
            "the CTR fit must COMMIT to the device: expected {} device grows, got {}. \
             `oblivious_trees.len() == iterations` does not say this — the CPU oblivious \
             grower satisfies it too (R-8).",
            params.iterations,
            gpu.grown.get()
        );
        assert_eq!(
            cpu.grown.get(),
            0,
            "the reference arm must run the CPU grower — a `CpuRefRuntime` that committed to \
             the device would make this differential a device-vs-device tautology"
        );
        assert_eq!(
            cpu.accepted_begins.get(),
            0,
            "the reference arm's `begin_device_training` must inherit the trait default \
             `Ok(false)`; it accepted {} of {} sessions",
            cpu.accepted_begins.get(),
            cpu.begins.get()
        );
        assert_eq!(dev.oblivious_trees.len(), params.iterations);
        assert_eq!(host.oblivious_trees.len(), params.iterations);

        // ---- THE load-bearing vacuity guard: the horizon really does reach a CTR-free
        //      tree, on BOTH arms. See the module doc — every committed device CTR fixture
        //      stops at 5 iterations, where this guard would fail. NEVER weaken it. ----
        assert!(
            dev_ctr_free >= 1 && cpu_ctr_free >= 1,
            "no CTR-FREE tree in {iterations} iterations (device {dev_ctr_free} / cpu \
             {cpu_ctr_free}) — this file asserts nothing without one, because the branch \
             under test is exactly `a CTR fit's tree that chose zero CTR splits`. Raise the \
             horizon; do NOT drop this guard."
        );
        // …and the fit is genuinely a CTR fit, not a float fit that trivially has no CTR
        // trees (which would satisfy the guard above for the wrong reason).
        assert!(
            dev_ctr >= 1 && cpu_ctr >= 1,
            "both arms must also contain ≥1 CTR split (device {dev_ctr}, cpu {cpu_ctr}) — \
             otherwise every tree is CTR-free and the fit is not exercising the CTR path"
        );

        println!(
            "[device-ctr-free-tree-leaf] iters={iterations} \
             CTR splits: device {dev_ctr} / cpu {cpu_ctr} | \
             CTR-free trees: device {dev_ctr_free} / cpu {cpu_ctr_free} | \
             device grows = {}, cpu device-grows = {} (begins {} / accepted {})",
            gpu.grown.get(),
            cpu.grown.get(),
            cpu.begins.get(),
            cpu.accepted_begins.get()
        );

        // ---- split-sequence equality FIRST: a leaf comparison over two different trees
        //      would be meaningless, and a structural divergence must not be reported as a
        //      leaf-value divergence. ----
        for (ti, (d, h)) in
            dev.oblivious_trees.iter().zip(host.oblivious_trees.iter()).enumerate()
        {
            assert_eq!(
                d.splits, h.splits,
                "tree {ti}: the FLOAT split sequence diverges between the device and CPU \
                 growers"
            );
            assert_eq!(
                ctr_sigs(d),
                ctr_sigs(h),
                "tree {ti}: the CTR split sequence diverges between the device and CPU growers"
            );
            assert_eq!(d.one_hot_splits, h.one_hot_splits, "tree {ti}: one-hot splits diverge");
        }

        // ---- the assertion this file exists for: leaf values agree on EVERY tree, and in
        //      particular on the CTR-FREE ones. Reported per tree with its CTR-split count so
        //      a failure names the branch directly. ----
        let mut max_all = 0.0_f64;
        let mut max_ctr_free = 0.0_f64;
        for (ti, (d, h)) in
            dev.oblivious_trees.iter().zip(host.oblivious_trees.iter()).enumerate()
        {
            assert_eq!(d.leaf_values.len(), h.leaf_values.len(), "tree {ti} leaf count");
            let ctr_free = d.ctr_splits.is_empty();
            for (li, (&a, &b)) in d.leaf_values.iter().zip(h.leaf_values.iter()).enumerate() {
                let abs = (a - b).abs();
                max_all = max_all.max(abs);
                if ctr_free {
                    max_ctr_free = max_ctr_free.max(abs);
                }
                assert!(
                    abs <= LEAF_EPS,
                    "tree {ti} leaf {li} ({}): device {a} vs cpu {b}, |Δ| = {abs:.3e} exceeds \
                     ε = {LEAF_EPS:.0e}. On a CTR-FREE tree this is `T22-OBS-1`: the device \
                     returned its RESIDENT (learning-fold) leaf estimate instead of gathering \
                     over the caller's main/averaging derivative trajectory — see \
                     `gpu_runtime/session.rs`'s unconditional CTR leaf gather.",
                    if ctr_free { "CTR-FREE" } else { "carries CTR splits" }
                );
            }
        }
        println!(
            "[device-ctr-free-tree-leaf] max |Δleaf| = {max_all:.3e} over all {iterations} \
             trees, {max_ctr_free:.3e} over the {dev_ctr_free} CTR-FREE tree(s) (bar \
             {LEAF_EPS:.0e})"
        );
    }
}

/// `T22-OBS-1` at 30 iterations — the lowest round horizon on `ctr_device_combo` that reaches
/// a CTR-free tree (the first is #23; 4 of the 30 trees are CTR-free, and 3 more are
/// contaminated by them through the main approx).
#[test]
fn ctr_free_trees_of_a_ctr_fit_match_the_cpu_grower_leaf_values() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    device::run(30);
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    {
        let _ = ctr_free_params(30);
        eprintln!(
            "SKIP ctr_free_trees_of_a_ctr_fit_match_the_cpu_grower_leaf_values: needs rocm/cuda"
        );
    }
}
