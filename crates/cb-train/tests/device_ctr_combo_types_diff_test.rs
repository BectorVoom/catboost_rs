//! DCTR-20 (T22): the **combination × {Buckets, Counter, BTMV}** cross-product detector.
//!
//! P1's delivered coverage is simple × {Borders, Buckets, Counter, BTMV} (DCTR-08/10/14)
//! and combination × **Borders only** (DCTR-17 — `ctr_device_combo`'s frozen `config.json`
//! pins `combinations_ctr: ["Borders:Prior=0.5"]`). Nothing exercised combination ×
//! {Buckets, Counter, BTMV}, and the first thing that cross-product hits is the
//! `bucket_counts` fallback for a ≥2-member column (T17) and the per-level `eligible_max`
//! filter (T18). This file is that detector.
//!
//! # What this is, and what it is NOT
//!
//! It is a **device-grower-vs-CPU-grower self-oracle at ε = 1e-4** (D-07), NOT an upstream
//! comparison: it reuses `ctr_device_combo`'s frozen `X`/`X_cat`/`y`/`borders` unchanged
//! (R-12 — no new fixture, no seed search) and never reads `predictions.npy`. It therefore
//! proves **device == CPU**, not device == upstream. The upstream half of the chain is the
//! two shipped CPU oracles `ctr_mixed_simple_vs_combo_oracle_test` and
//! `tensor_ctr_e2e_oracle_test`, which T22's validation block runs alongside this file.
//!
//! Because it is a self-oracle and never compares against `predictions.npy`, its TRAINING
//! PARAMS are tunable (R-12 freezes the fixture's data artifacts, not the params a
//! device-vs-CPU differential drives them with). The frozen arrays are byte-untouched.
//!
//! # The comparison surface, and why it is the chosen-split sequence and NOT predictions
//!
//! Two independent measurements say a **prediction** comparison cannot detect a routing
//! defect on either non-Borders arm:
//!
//! * **Buckets** (T10 §2): at `Prior = 0.5` and binclf,
//!   `ctr(Buckets@0) + ctr(Buckets@1) = (total + 1)/(total + 1) = 1` and
//!   `calc_normalization(0.5) = (0, 1)`, so the `b = 0` and `b = 1` columns binarize to
//!   MIRRORED bins. The device selecting `[0]×8` and the CPU selecting `[0,1,0,0,1,1,0,1]`
//!   is then *the same oblivious partition with one level bit flipped*, and both score
//!   `2.776e-17` against upstream.
//! * **BTMV** (T15/T16): at binclf `Sum == N[1]` and `Count == Total`, so BTMV and
//!   `Borders@0` emit IDENTICAL cindex bins. A device path that kept routing BTMV columns
//!   through the Borders numerator is prediction-identical on EVERY binclf fixture.
//!
//! So this file compares the **chosen-split identity sequence** per tree — `splits`,
//! `ctr_splits` (on the full `(projection, ctr_type, prior_num, prior_denom,
//! target_border_idx, border)` identity `assign_leaf_over_ctr_columns` keys on),
//! `one_hot_splits` and `level_kinds` — plus leaf values at ε = 1e-4.
//!
//! ## Why the Buckets arm runs at a prior ≠ 0.5 (stated UP FRONT, not discovered)
//!
//! The mirror above is not merely a prediction blind spot: it makes the two Buckets columns
//! of a `(projection, prior)` pair score EXACTLY equal, so device-vs-CPU enumeration order
//! decides which one wins and a raw split-sequence comparison would diverge **for a benign
//! reason** — and, symmetrically, an agreement at `Prior = 0.5` would be uninformative. The
//! coordinator's consolidated T10 §2 + T16 finding offers two remedies; this file takes the
//! second, at its source: the **Buckets arm drives `combinations_ctr_priors = [0.25]`**, a
//! prior ≠ 0.5, at which `ctr(b0) + ctr(b1) = (total + 0.5)/(total + 1) ≠ 1` and the two
//! columns are no longer mirrors. That is strictly better than weakening the assertion,
//! which is exactly the class of change GLOBALS §2.5 exists to prevent.
//!
//! **Counter and BTMV keep `Prior = 0.5`** (the task text's value): `ECtrType::
//! target_border_count(2)` is `1` for both (`ctr/mod.rs`, `GetTargetBorderCount`,
//! `ctr_helper.h:34-42`), so each `(projection, prior)` yields exactly ONE column and there
//! is no mirrored twin to tie with. The T10 §2 degeneracy is structurally absent there.
//!
//! ## What this file honestly CANNOT say
//!
//! T15's binclf BTMV ≡ Borders@0 identity holds at the **cindex-column** level, so if the
//! device ever routed a BTMV column to the Borders numerator, BOTH arms' splits would still
//! agree and this differential would stay green. That specific defect is covered by three
//! other layers (T16 §2): the per-split `ctr_type` assertion below, the launchers'
//! incompatible signatures (a swap is a compile error), and DCTR-12's kernel self-oracle.
//! A cindex-column-level differential would see it; it is not reachable from a `cb-train`
//! integration test without new production readback seams, and T22 adds no production code.
//!
//! # A divergence this file MEASURED but does not own — `T22-OBS-1`, now FIXED elsewhere
//!
//! While climbing the guard-4 escalation ladder for the Counter arm (below), longer fits on
//! this corpus reached the first tree whose greedy search chooses **no CTR split at all**.
//! On exactly those trees the device and CPU leaf VALUES diverged by ~1e-3 while their split
//! sequences stayed identical (`tree 23 → 1.069e-3`, `tree 27 → 1.280e-3` at 30 iterations;
//! reproducing unchanged on the shipped `combinations_ctr = Borders` configuration). Every
//! committed device CTR fixture stops at 5 iterations, where every tree still carries a CTR
//! split, which is why nothing had caught it.
//!
//! **Root cause and fix (post-phase):** the device's averaging-permutation leaf-value gather
//! (`gpu_runtime/session.rs`) was gated on *"this tree chose ≥1 CTR split"*, so a CTR fit's
//! CTR-free tree returned the RESIDENT learning-fold leaf estimate instead of the main /
//! averaging one. The gate is now removed — the gather is unconditional on a covered CTR fit —
//! and the divergence is ≤1.4e-17 on every tree. The detector is
//! `crates/cb-train/tests/device_ctr_free_tree_leaf_test.rs`, which runs this same corpus at 30
//! iterations and asserts that the horizon really does reach a CTR-free tree.
//!
//! T22's own arms are unchanged: each is still configured to sit strictly BELOW the first
//! CTR-free tree, and each run still PRINTS its CTR-free tree count so that is visible rather
//! than assumed. This file's green was never, and still is not, the evidence for CTR-free
//! trees — that is the other file's job.
//!
//! # The Buckets arm's horizon — `T22-OBS-2`, addressed elsewhere
//!
//! The Buckets arm below ships at 5 iterations because at longer horizons the raw
//! `CtrSplitSpec`-identity comparison false-reds on a benign `b=0`/`b=1` tie that survives a
//! prior ≠ 0.5 (see `T22-OBS-2`; measured at tree 12 of a 20-iteration run). The long-horizon
//! statement is made instead by
//! `crates/cb-train/tests/device_ctr_buckets_long_horizon_diff_test.rs`, over a
//! partition-invariant projection of the tree. Keep this arm strict and short; keep that one
//! invariant and long.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp
)]

use cb_compute::{EScoreFunction, LeafMethod, Loss};
use cb_train::{BoostParams, EBootstrapType, ECtrType, EOverfittingDetectorType};

/// The `ctr_device_combo` params, with the CTR types, the combination prior, `iterations` and
/// `depth` parameterised (the last two are the guard-4 escalation ladder's rungs 1 and 2).
///
/// `simple_ctr` stays `Borders` on every shipped arm, so the fit MIXES a simple and a
/// combination column — which is what makes D-2's `eligible_max` filter load-bearing (a fit
/// whose columns were all one arity could not observe a per-level eligibility difference at
/// all). It is a parameter only because the Counter arm's ladder investigation drove it to
/// `Counter` as a diagnostic; that probe is recorded in `notes/T22.md` and did not ship.
///
/// `simple_ctr_priors` / `combinations_ctr_priors` are BOTH pinned explicitly: the default
/// prior list is per-type and inverts between the types under test (Counter's default is
/// the single `0/1`; BTMV's is the `{0, 0.5, 1}` TRIPLE — T16 §5), so an implicit list
/// would silently change the number of materialized CTR columns per arm.
fn diff_params(
    simple_ctr: ECtrType,
    combinations_ctr: ECtrType,
    combination_prior: f64,
    iterations: usize,
    depth: usize,
) -> BoostParams {
    BoostParams {
        loss: Loss::Logloss,
        iterations,
        depth,
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
        simple_ctr,
        simple_ctr_priors: vec![0.5],
        counter_calc_method: cb_train::counter_calc_method_default(),
        boosting_type: cb_train::boosting_type_default(),
        max_ctr_complexity: 2,
        combinations_ctr,
        combinations_ctr_priors: vec![combination_prior],
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

    use super::diff_params;
    use cb_backend::GpuBackend;
    use cb_compute::{
        Derivatives, DeviceGrownTree, DeviceTrainConfig, EScoreFunction, FamilyTreeArgs, Loss,
        Runtime,
    };
    use cb_core::CbResult;
    use cb_data::stringify_int_category;
    use cb_train::{averaging_ctr_permutation, create_shuffled_indices, train_cat, ECtrType};
    use ndarray::Array2;
    use ndarray_npy::read_npy;

    /// The device-commitment counter (GLOBALS §2.2.6). Copied **verbatim** from
    /// `crates/cb-train/tests/device_ctr_gate_test.rs` (the canonical copy) — keep in sync:
    /// every override forwards to `self.inner: GpuBackend` and only `grow_tree_on_device`
    /// counts, and only when it returns `Some` (a `None` is the device declining a tree).
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

    /// The CPU reference arm. It overrides **only** `compute_gradients`, forwarding to a real
    /// `GpuBackend` so BOTH arms consume bit-identical derivatives and the differential
    /// isolates the GROWER rather than the gradient kernel. Every device-seam method inherits
    /// the `cb_compute::Runtime` trait default (`begin_device_training → Ok(false)`,
    /// `grow_tree_on_device → Ok(None)`, `end_device_training → Ok(())`,
    /// `runtime.rs`'s default bodies) — the shipped precedent is
    /// `crates/cb-train/tests/device_nonsym_fit_test.rs`'s `CpuRefRuntime`.
    ///
    /// `CpuBackend` itself is not compiled under `rocm`/`cuda` (GLOBALS §2.2.3), which is why
    /// the gradients are borrowed from `GpuBackend` rather than from the CPU backend.
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

    /// The same counting wrapper shape as [`CountingGpu`], but wrapping [`CpuRefRuntime`].
    ///
    /// Its purpose is the risk guardrail T22's task text names: a `CpuRefRuntime` that
    /// accidentally committed to the device would make this whole file a device-vs-device
    /// tautology (the R-8 class, one level up). `grown == 0` is a real observation — this
    /// wrapper really does call through to `CpuRefRuntime::grow_tree_on_device` once per
    /// tree, and counts every `Some` it returns.
    pub struct CountingCpu {
        inner: CpuRefRuntime,
        pub grown: AtomicUsize,
        pub begins: AtomicUsize,
        pub accepted_begins: AtomicUsize,
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
            self.begins.fetch_add(1, Ordering::Relaxed);
            let accepted = self.inner.begin_device_training(
                loss, depth, plain, fold_count, score_function, bins, weight, n, n_features,
                n_bins, lr, scaled_l2, config,
            )?;
            if accepted {
                self.accepted_begins.fetch_add(1, Ordering::Relaxed);
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
                self.grown.fetch_add(1, Ordering::Relaxed);
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

    /// The full identity of a chosen CTR split — the SAME key
    /// `assign_leaf_over_ctr_columns` uses (`boosting.rs`). `shift`/`scale` are deliberately
    /// excluded: they are derived from `prior_num` by the bake, so including them would add
    /// no discrimination.
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

    /// The tree's per-level split-KIND sequence, canonicalised.
    ///
    /// `ObliviousTree::level_kinds` is an OPTIONAL encoding: its own doc says it is EMPTY
    /// when a tree's levels are all one kind, and that consumers then fall back to the
    /// kind-grouped order (`splits`, then `ctr_splits`, then `one_hot_splits`) — which is
    /// byte-identical to pre-`level_kinds` behaviour (SPEC-OH-31). The device and CPU growers
    /// exercise that option DIFFERENTLY: on an all-float tree the device leaves the vector
    /// empty while the CPU emits `[Float(0), Float(1)]`. Both decode to the same tree, so
    /// this canonicalises to the decoded sequence rather than comparing the encoding.
    ///
    /// The equivalence is not assumed: [`assert_level_kinds_agree`] first checks that an
    /// EMPTY vector really does belong to a single-kind tree, which is the precondition the
    /// documented fallback rests on. A `Ctr` level's `border` is deliberately not carried
    /// here — it is already compared, per split, by [`ctr_sigs`].
    fn kind_sequence(tree: &cb_train::ObliviousTree) -> Vec<(u8, usize)> {
        use cb_train::LevelKind;
        if !tree.level_kinds.is_empty() {
            return tree
                .level_kinds
                .iter()
                .map(|k| match *k {
                    LevelKind::Float(i) => (0u8, i),
                    LevelKind::Ctr { ctr_idx, .. } => (1u8, ctr_idx),
                    LevelKind::OneHot(i) => (2u8, i),
                })
                .collect();
        }
        (0..tree.splits.len())
            .map(|i| (0u8, i))
            .chain((0..tree.ctr_splits.len()).map(|i| (1u8, i)))
            .chain((0..tree.one_hot_splits.len()).map(|i| (2u8, i)))
            .collect()
    }

    /// How many distinct split KINDS a tree actually uses.
    fn kinds_used(tree: &cb_train::ObliviousTree) -> usize {
        usize::from(!tree.splits.is_empty())
            + usize::from(!tree.ctr_splits.is_empty())
            + usize::from(!tree.one_hot_splits.is_empty())
    }

    /// Compare the decoded per-level kind sequences, after verifying that any EMPTY
    /// `level_kinds` really is the documented single-kind fallback (an empty vector on a
    /// tree whose levels DO interleave would be a genuine defect, not an encoding choice).
    fn assert_level_kinds_agree(label: &str, ti: usize, d: &cb_train::ObliviousTree, h: &cb_train::ObliviousTree) {
        for (arm, t) in [("device", d), ("cpu", h)] {
            if t.level_kinds.is_empty() {
                assert!(
                    kinds_used(t) <= 1,
                    "[{label}] {arm} tree {ti}: `level_kinds` is EMPTY on a tree whose levels \
                     interleave ({} float / {} ctr / {} one-hot) — the documented fallback \
                     (SPEC-OH-31) only applies to a single-kind tree, so this is a real defect",
                    t.splits.len(),
                    t.ctr_splits.len(),
                    t.one_hot_splits.len()
                );
            }
        }
        assert_eq!(
            kind_sequence(d),
            kind_sequence(h),
            "[{label}] tree {ti}: the per-level split KIND sequence diverges (kind tags: \
             0 = float, 1 = ctr, 2 = one-hot)"
        );
    }

    /// `(total CTR splits, ≥2-member CTR splits)` over a whole model.
    fn ctr_split_counts(trees: &[cb_train::ObliviousTree]) -> (usize, usize) {
        let total: usize = trees.iter().map(|t| t.ctr_splits.len()).sum();
        let combos = trees
            .iter()
            .flat_map(|t| t.ctr_splits.iter())
            .filter(|s| s.projection.cat_features().len() >= 2)
            .count();
        (total, combos)
    }

    /// One arm of the differential, for `combinations_ctr = ctr_type` at `prior`, over
    /// `iterations` trees of `depth` levels (T22's guard-4 escalation ladder — see each
    /// `#[test]`'s doc for the rung that arm reached and why).
    pub fn run(
        simple_ctr: ECtrType,
        ctr_type: ECtrType,
        prior: f64,
        iterations: usize,
        depth: usize,
        label: &str,
    ) {
        let x: Array2<f32> = read_npy(fixture("X.npy")).expect("X.npy loads");
        let columns: Vec<Vec<f32>> = (0..x.ncols()).map(|fi| x.column(fi).to_vec()).collect();
        let borders_arr: Array2<f64> = read_npy(fixture("borders.npy")).expect("borders.npy");
        let borders: Vec<Vec<f64>> =
            (0..borders_arr.nrows()).map(|r| borders_arr.row(r).to_vec()).collect();
        let cat: Array2<i32> = read_npy(fixture("X_cat.npy")).expect("X_cat.npy is [N,2]");
        assert_eq!(cat.ncols(), 2, "[{label}] the combo fixture must ship two cat columns");
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
        let params = diff_params(simple_ctr, ctr_type, prior, iterations, depth);

        // The prior is load-bearing for the Buckets arm (see the module doc) — pin it on the
        // params actually handed to both fits, not just in prose.
        assert_eq!(params.combinations_ctr_priors, vec![prior]);
        assert_eq!(params.simple_ctr, simple_ctr);
        assert_eq!(params.combinations_ctr, ctr_type);

        // Fixture-permutation-divergence guard (GLOBALS §2.2): the structure order and the
        // averaging order must genuinely differ, or a structure-only leaf gather would pass.
        let structure = create_shuffled_indices(n, params.random_seed);
        let averaging = averaging_ctr_permutation(n, 1, params.random_seed);
        assert_ne!(
            structure, averaging,
            "[{label}] structure and averaging permutations coincide at (n={n}, seed={}) — \
             the fixture cannot discriminate a structure-only leaf gather",
            params.random_seed
        );

        // ---- arm 1: the DEVICE grower ----
        let gpu = CountingGpu { inner: GpuBackend::default(), grown: AtomicUsize::new(0) };
        let (dev, _dev_baked) =
            train_cat(&gpu, &columns, &borders, &cat_columns, &target, &[], &params, None)
                .unwrap_or_else(|e| panic!("[{label}] device combination-CTR train failed: {e:?}"));

        // ---- arm 2: the CPU grower, same inputs, same gradients ----
        let cpu = CountingCpu {
            inner: CpuRefRuntime { inner: GpuBackend::default() },
            grown: AtomicUsize::new(0),
            begins: AtomicUsize::new(0),
            accepted_begins: AtomicUsize::new(0),
        };
        let (host, _host_baked) =
            train_cat(&cpu, &columns, &borders, &cat_columns, &target, &[], &params, None)
                .unwrap_or_else(|e| panic!("[{label}] cpu combination-CTR train failed: {e:?}"));

        let (dev_ctr, dev_combo) = ctr_split_counts(&dev.oblivious_trees);
        let (cpu_ctr, cpu_combo) = ctr_split_counts(&host.oblivious_trees);
        // The CTR-FREE tree count is printed, not asserted: it is the boundary of the
        // separately-reported `T22-OBS-1` divergence (see the module doc — the divergence
        // itself is now FIXED, and owned by `device_ctr_free_tree_leaf_test`). Every arm here
        // is configured to sit strictly below it, and a reader must be able to see that
        // rather than take it on
        // trust.
        let dev_ctr_free = dev.oblivious_trees.iter().filter(|t| t.ctr_splits.is_empty()).count();
        let cpu_ctr_free = host.oblivious_trees.iter().filter(|t| t.ctr_splits.is_empty()).count();
        println!(
            "[device-ctr-combo-types-diff/{label}] prior={prior} iters={iterations} depth={depth} \
             device: {dev_ctr} CTR splits ({dev_combo} ≥2-member) | \
             cpu: {cpu_ctr} CTR splits ({cpu_combo} ≥2-member) | \
             CTR-free trees: device {dev_ctr_free} / cpu {cpu_ctr_free} | \
             device grows = {}, cpu device-grows = {} (begins {} / accepted {})",
            gpu.grown.load(Ordering::Relaxed),
            cpu.grown.load(Ordering::Relaxed),
            cpu.begins.load(Ordering::Relaxed),
            cpu.accepted_begins.load(Ordering::Relaxed)
        );

        // ---- (1) the device arm really COMMITTED, and the CPU arm really did NOT ----
        assert_eq!(
            gpu.grown.load(Ordering::Relaxed),
            params.iterations,
            "[{label}] the combination × {ctr_type:?} fit must COMMIT to the device: expected \
             {} device grows, got {}. `oblivious_trees.len() == iterations` does not say this \
             — the CPU oblivious grower satisfies it too (R-8).",
            params.iterations,
            gpu.grown.load(Ordering::Relaxed)
        );
        assert_eq!(
            cpu.grown.load(Ordering::Relaxed),
            0,
            "[{label}] the reference arm must run the CPU grower — a `CpuRefRuntime` that \
             committed to the device would make this differential a device-vs-device \
             tautology (R-8, one level up)"
        );
        assert_eq!(
            cpu.accepted_begins.load(Ordering::Relaxed),
            0,
            "[{label}] the reference arm's `begin_device_training` must inherit the trait \
             default `Ok(false)`; it accepted {} of {} sessions",
            cpu.accepted_begins.load(Ordering::Relaxed),
            cpu.begins.load(Ordering::Relaxed)
        );

        // ---- (4) vacuity guards, BEFORE the equality (an equality over two empty split
        //          lists is trivially satisfied) ----
        assert_eq!(dev.oblivious_trees.len(), params.iterations);
        assert_eq!(host.oblivious_trees.len(), params.iterations);
        assert!(dev.non_symmetric_trees.is_empty() && dev.region_trees.is_empty());
        assert!(host.non_symmetric_trees.is_empty() && host.region_trees.is_empty());
        assert!(
            dev_ctr >= 1 && cpu_ctr >= 1,
            "[{label}] both arms must contain ≥1 CTR split (device {dev_ctr}, cpu {cpu_ctr}) \
             — a CTR-free differential asserts nothing"
        );
        assert!(
            dev_combo >= 1 && cpu_combo >= 1,
            "[{label}] both arms must contain ≥1 COMBINATION (≥2-member) CTR split (device \
             {dev_combo}, cpu {cpu_combo}). Without one the combination path is untested and \
             this differential is trivially satisfied. See T22's guard-4 escalation ladder \
             (raise iterations, then depth, then sweep the prior); NEVER weaken this guard."
        );

        // Routing hygiene (COORDINATOR-FINDINGS T07): the descriptor COUNT does not
        // discriminate the CTR type, so assert the type per chosen split. Simple projections
        // are `simple_ctr` (Borders); ≥2-member projections must be the type under test.
        for (arm, trees) in [("device", &dev.oblivious_trees), ("cpu", &host.oblivious_trees)] {
            for (ti, tree) in trees.iter().enumerate() {
                for sig in ctr_sigs(tree) {
                    let expect = if sig.projection.len() >= 2 {
                        ctr_type.as_i8()
                    } else {
                        simple_ctr.as_i8()
                    };
                    assert_eq!(
                        sig.ctr_type, expect,
                        "[{label}] {arm} tree {ti}: a {}-member projection carried ctr_type \
                         {}, expected {expect} ({sig:?})",
                        sig.projection.len(),
                        sig.ctr_type
                    );
                }
            }
        }

        // ---- (2) SPLIT-SEQUENCE EQUALITY, per tree, over `ObliviousTree`'s ordered
        //          surfaces. This is the assertion predictions cannot make. ----
        for (ti, (d, h)) in dev.oblivious_trees.iter().zip(host.oblivious_trees.iter()).enumerate()
        {
            assert_eq!(
                d.splits, h.splits,
                "[{label}] tree {ti}: the FLOAT split sequence diverges between the device and \
                 CPU growers"
            );
            assert_eq!(
                ctr_sigs(d),
                ctr_sigs(h),
                "[{label}] tree {ti}: the CTR split sequence diverges between the device and \
                 CPU growers (full `CtrSplitSpec` identity: projection, ctr_type, prior_num, \
                 prior_denom, target_border_idx, border)"
            );
            assert_eq!(
                d.one_hot_splits, h.one_hot_splits,
                "[{label}] tree {ti}: the ONE-HOT split sequence diverges"
            );
            assert_level_kinds_agree(label, ti, d, h);
        }

        // ---- (3) leaf values within ε = 1e-4 (D-07's device-vs-CPU bar, NOT the 1e-5
        //          upstream bar — this is a self-oracle) ----
        let mut max_leaf_delta = 0.0_f64;
        for (ti, (d, h)) in dev.oblivious_trees.iter().zip(host.oblivious_trees.iter()).enumerate()
        {
            assert_eq!(d.leaf_values.len(), h.leaf_values.len(), "[{label}] tree {ti} leaf count");
            for (li, (&a, &b)) in d.leaf_values.iter().zip(h.leaf_values.iter()).enumerate() {
                let abs = (a - b).abs();
                max_leaf_delta = max_leaf_delta.max(abs);
                assert!(
                    abs <= 1e-4,
                    "[{label}] tree {ti} leaf {li}: device {a} vs cpu {b} exceeds ε=1e-4 \
                     (|Δ|={abs:.3e})"
                );
            }
        }
        println!(
            "[device-ctr-combo-types-diff/{label}] split sequences IDENTICAL across \
             {} trees; max |Δleaf| = {max_leaf_delta:.3e} (bar 1e-4)",
            params.iterations
        );
    }
}

/// DCTR-20 — combination × **Buckets**, at `Prior = 0.25`.
///
/// The prior is ≠ 0.5 deliberately and up front: see the module doc's "Why the Buckets arm
/// runs at a prior ≠ 0.5". At `Prior = 0.5` the `b = 0` / `b = 1` columns are exact mirrors,
/// score exactly equal, and the winner is decided by enumeration order — so a raw
/// split-sequence comparison would diverge for a benign reason and an agreement would be
/// uninformative (T10 §2).
///
/// **Guard-4 ladder rung 0**: green at the fixture's own `iterations = 5, depth = 2`, with
/// 8 CTR splits of which 3 are ≥2-member combinations. No escalation was needed.
#[test]
fn combination_ctr_non_borders_types_match_the_cpu_grower_buckets() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    device::run(ECtrType::Borders, ECtrType::Buckets, 0.25, 5, 2, "buckets");
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    {
        let _ = diff_params(ECtrType::Borders, ECtrType::Buckets, 0.25, 5, 2);
        eprintln!(
            "SKIP combination_ctr_non_borders_types_match_the_cpu_grower_buckets: needs rocm/cuda"
        );
    }
}

/// DCTR-20 — combination × **Counter**, at `Prior = 0.5`, **guard-4 ladder rung 1**
/// (`iterations` 5 → **20**; `depth` stays 2).
///
/// `target_border_count(2) == 1` for Counter, so a `(projection, prior)` pair yields exactly
/// ONE column: there is no mirrored twin and the T10 §2 degeneracy is structurally absent.
///
/// # Why this arm needed the ladder, and exactly how far it climbed
///
/// At the fixture's own `iterations = 5, depth = 2` this arm chose **zero** ≥2-member CTR
/// splits — on BOTH arms, so a search outcome and not a device defect. That is the
/// non-defect guard-4 failure T22's task text predicts, and the cause is structural: the
/// combination Counter column carries only the JOINT FREQUENCY of the two cat columns, which
/// on this corpus correlates with the target at r = 0.16, against `simple_ctr = Borders`
/// columns that encode the target statistic directly. Measured rungs, all at
/// `simple_ctr = Borders` unless noted:
///
/// ```text
///  5 iters / depth 2   →  8 CTR splits,   0 ≥2-member   (rung 0, the fixture's own params)
/// 10 iters / depth 2   → 10 CTR splits,   0 ≥2-member   (rung 1)
/// 10 iters / depth 3   → 10 CTR splits,   0 ≥2-member   (rung 2)
/// priors 0 / .25 / 1 / 2 at 10 iters / depth 3          (rung 3) — 0 ≥2-member at every prior
/// 15 iters / depth 2   → 15 CTR splits,   0 ≥2-member
/// 20 iters / depth 2   → 24 CTR splits,   4 ≥2-member   ◀ SHIPPED
/// ```
///
/// (A further diagnostic, `simple_ctr = Counter` too, at 30 iters / depth 3 and 100 iters /
/// depth 2, is recorded in `notes/T22.md`; it did not ship.) `20` is the LOWEST rung-1 value
/// that satisfies guard 4, which is the ladder's "climb in order" discipline — and it also
/// keeps this arm strictly below the first CTR-free tree (#23), the boundary of the separately
/// reported `T22-OBS-1` divergence the module doc records. **Guard 4 itself was never
/// weakened**; the params moved, which the task text explicitly authorises for this
/// self-oracle (the frozen `X`/`X_cat`/`y`/`borders` are byte-untouched, R-12).
#[test]
fn combination_ctr_non_borders_types_match_the_cpu_grower_counter() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    device::run(ECtrType::Borders, ECtrType::Counter, 0.5, 20, 2, "counter");
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    {
        let _ = diff_params(ECtrType::Borders, ECtrType::Counter, 0.5, 20, 2);
        eprintln!(
            "SKIP combination_ctr_non_borders_types_match_the_cpu_grower_counter: needs rocm/cuda"
        );
    }
}

/// DCTR-20 — combination × **BinarizedTargetMeanValue**, at `Prior = 0.5`.
///
/// `target_border_count(2) == 1` for BTMV too. Note the module doc's honest limitation: this
/// arm cannot detect a BTMV→Borders numerator swap, because T15's binclf identity makes the
/// two bin-identical; three other layers cover that.
///
/// **Guard-4 ladder rung 0**: green at the fixture's own `iterations = 5, depth = 2`, with
/// 8 CTR splits of which 3 are ≥2-member combinations. No escalation was needed.
#[test]
fn combination_ctr_non_borders_types_match_the_cpu_grower_btmv() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    device::run(ECtrType::Borders, ECtrType::BinarizedTargetMeanValue, 0.5, 5, 2, "btmv");
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    {
        let _ = diff_params(ECtrType::Borders, ECtrType::BinarizedTargetMeanValue, 0.5, 5, 2);
        eprintln!(
            "SKIP combination_ctr_non_borders_types_match_the_cpu_grower_btmv: needs rocm/cuda"
        );
    }
}
