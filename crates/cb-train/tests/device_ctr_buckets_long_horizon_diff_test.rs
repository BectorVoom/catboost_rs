//! `T22-OBS-2` — the LONG-HORIZON combination × **Buckets** device-vs-CPU differential, over a
//! **partition-invariant projection** of the tree instead of the raw split identity.
//!
//! # The problem this file solves
//!
//! A `(projection, prior)` pair of the **Buckets** CTR yields one column per target border —
//! for a binary target, `b = 0` and `b = 1`. Those two columns are ordinally
//! anti-monotone: `ctr(b0) + ctr(b1) = (total + 2·prior)/(total + 1)`, so their binarized bins
//! satisfy `bin(b0) + bin(b1) ≈ const`. Whenever a threshold pair lands so that
//! `bin_b0 > s` is the exact complement of `bin_b1 > t`, the two candidate splits induce the
//! **same partition of the objects** (up to which side is "left"), score EXACTLY equal, and the
//! greedy winner is decided by candidate-enumeration order alone.
//!
//! T22 hit this. Its Buckets arm compares the raw `CtrSplitSpec` identity, and took the first
//! of the coordinator's two remedies — a prior ≠ 0.5, which removes the exact algebraic mirror
//! (`ctr(b0) + ctr(b1) = 1`). `T22-OBS-2` is the finding that **this is necessary but not
//! sufficient**: at `Prior = 0.25`, 20 iterations, depth 2, tree 12 level 1 the device picks
//! `([0,1], Buckets, target_border_idx = 0, border 11.999999)` and the CPU picks
//! `([0,1], Buckets, target_border_idx = 1, border 0.999999)` — a benign tie, verified
//! independent of the D-2 eligibility filter. A prior ≠ 0.5 kills the mirror IDENTITY but not
//! the ordinal anti-monotonicity, and with ~12 combination buckets over 15 CTR bins many
//! threshold pairs still induce identical partitions.
//!
//! ⇒ the coordinator's OTHER remedy is the one that scales: **a genuinely partition-invariant
//! projection of the split set**. That is what this file implements.
//!
//! # The projection
//!
//! A split set's only observable effect on the training data is the PARTITION of the objects
//! it induces. So project each tree to that partition, and compare the projections.
//!
//! The per-object, per-tree leaf assignment at TRAINING time is reachable without any new
//! production seam: `train_cat`'s `staged_out` records the main (averaging-fold) approx after
//! every iteration, so tree `t`'s per-object contribution is
//! `staged[t·n + i] − staged[(t−1)·n + i]` — which is exactly `leaf_values[leaf_of(i)]` for
//! that tree's LEAF-VALUE partition. [`partition_labels`] then relabels objects by
//! first-occurrence of their contribution value, yielding a canonical labelling of the induced
//! partition.
//!
//! That labelling is invariant to everything the tie can change and to nothing else:
//!
//! | change | label vector |
//! |---|---|
//! | a level's bit complemented (`b=0`/`b=1` swap over an equivalent column) | **unchanged** — the leaf VALUES permute with their members, so each object keeps its value |
//! | leaves renumbered / levels reordered | **unchanged** |
//! | an object actually routed to a different group | **changes** |
//!
//! The invariance and the discrimination are not asserted by prose:
//! [`partition_labels_are_invariant_to_leaf_relabelling_but_not_to_regrouping`] pins both on
//! constructed input, and runs on every backend (no device needed).
//!
//! # What this file does NOT claim
//!
//! Where two Buckets columns genuinely induce the same partition on this corpus, **nothing
//! computable from this corpus can tell them apart** — a `b=0`/`b=1` swap there is not a
//! detectable defect, it is a re-description of the same tree. The projection is blind to
//! exactly that and to nothing more; it is the maximum discrimination available, not a
//! weakening. The complementary short-horizon statement — the strict, ordered
//! `CtrSplitSpec`-identity comparison — is `device_ctr_combo_types_diff_test`'s Buckets arm at
//! 5 iterations, which stays as it is. The two files are meant to be read together.
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

/// Recover each object's LEAF from its per-tree contribution, by matching the contribution
/// against the tree's own `leaf_values`.
///
/// The obvious alternative — grouping objects by exact `f64` equality of their contribution —
/// does NOT work, and the reason is worth recording: the contribution is recovered as
/// `staged[t] − staged[t−1]`, and `(a + v) − a` is not bit-exactly `v`. Two objects in the same
/// leaf therefore recover contributions that differ in the last bits, and exact grouping splits
/// a 4-leaf tree into 5–6 spurious groups (measured, before this was anchored). Matching
/// against the tree's actual leaf values removes the whole problem: the recovered value is
/// within one ulp-of-the-approx of exactly one leaf value.
///
/// Returns `None` if any object's contribution is not within `tol` of some leaf value — that
/// would mean the recovered quantity is not a leaf value at all, and the caller must fail
/// loudly rather than silently compare nonsense.
///
/// Leaves carrying numerically identical values (e.g. two empty leaves at `0.0`) collapse to
/// the first such index. That is conservative in the safe direction — it can only ever HIDE a
/// difference, never invent one — and the per-object value comparison in [`device::run`] is the
/// independent second check.
fn leaf_assignment(contrib: &[f64], leaf_values: &[f64], tol: f64) -> Option<Vec<usize>> {
    contrib
        .iter()
        .map(|&c| {
            let mut best: Option<(usize, f64)> = None;
            for (l, &v) in leaf_values.iter().enumerate() {
                let d = (c - v).abs();
                if best.is_none_or(|(_, bd)| d < bd) {
                    best = Some((l, d));
                }
            }
            best.filter(|&(_, d)| d <= tol).map(|(l, _)| l)
        })
        .collect()
}

/// Canonical labelling of a partition: relabel by order of first occurrence, discarding the
/// leaf INDICES themselves.
///
/// This is the projection proper. It is invariant to leaf renumbering — which is exactly what
/// a complemented level bit does (`leaf ^= 1 << l`, with the leaf values permuting alongside) —
/// and to nothing else: two objects change labels relative to each other if and only if they
/// stopped (or started) sharing a leaf.
fn canonical_labels(leaf_of: &[usize]) -> Vec<usize> {
    let mut labels = Vec::with_capacity(leaf_of.len());
    let mut seen: Vec<usize> = Vec::new();
    for &l in leaf_of {
        let idx = match seen.iter().position(|&s| s == l) {
            Some(i) => i,
            None => {
                seen.push(l);
                seen.len() - 1
            }
        };
        labels.push(idx);
    }
    labels
}

/// The projection's two defining properties, pinned on constructed input rather than asserted
/// in prose. Backend-independent — this is the part of the file that runs everywhere.
///
/// The first case is a faithful simulation of the `T22-OBS-2` tie: a depth-2 tree whose LEVEL 1
/// bit is complemented (the `b=0` ↔ `b=1` swap over an equivalent column). Each object's leaf
/// index changes — `leaf ^= 1 << 1` — and the leaf VALUES permute with their members, because a
/// leaf's value is estimated from the objects in it. So every object keeps its contribution,
/// and the projection must be blind to the swap while the raw leaf-index sequence is not.
#[test]
fn the_partition_projection_absorbs_a_complemented_level_but_not_a_regrouping() {
    // Eight objects over a depth-2 tree; leaf `l` carries value `values[l]`.
    let leaf_of: [usize; 8] = [0, 0, 1, 1, 2, 2, 3, 3];
    let values: [f64; 4] = [-0.5, 0.25, 0.75, -1.25];

    // Complement level 1: every object's leaf index flips bit 1, and the value vector permutes
    // the same way, so the per-object contribution is untouched.
    let swapped_leaf_of: Vec<usize> = leaf_of.iter().map(|&l| l ^ 0b10).collect();
    let swapped_values: Vec<f64> = (0..4).map(|l: usize| values[l ^ 0b10]).collect();

    // The raw description really did change …
    assert_ne!(
        leaf_of.to_vec(),
        swapped_leaf_of,
        "the simulation is vacuous unless the leaf INDICES actually differ"
    );
    // … and the projection is blind to it, which is the whole point.
    assert_eq!(
        canonical_labels(&leaf_of),
        canonical_labels(&swapped_leaf_of),
        "the partition projection must be invariant to a complemented level bit — that is the \
         benign `T22-OBS-2` tie it exists to absorb"
    );

    // Discrimination: regrouping the objects must change the labels. Without this the
    // projection could be trivially constant and still satisfy the invariance above.
    assert_ne!(
        canonical_labels(&leaf_of),
        canonical_labels(&[0, 1, 0, 1, 2, 3, 2, 3]),
        "the projection must still detect a genuine change in which objects share a group — \
         otherwise it is not a differential at all"
    );
    assert_eq!(canonical_labels(&leaf_of), vec![0, 0, 1, 1, 2, 2, 3, 3]);
    assert_eq!(canonical_labels(&[3, 3, 1]), vec![0, 0, 1]);

    // `leaf_assignment` recovers the leaf from a contribution perturbed by exactly the kind of
    // last-bit noise `staged[t] − staged[t−1]` introduces, and composes with `canonical_labels`
    // to reproduce the invariance end to end.
    let noisy: Vec<f64> =
        leaf_of.iter().enumerate().map(|(i, &l)| values[l] + (i as f64) * 1e-16).collect();
    assert_eq!(
        leaf_assignment(&noisy, &values, 1e-9).expect("every contribution matches a leaf"),
        leaf_of.to_vec()
    );
    let noisy_swapped: Vec<f64> = swapped_leaf_of
        .iter()
        .enumerate()
        .map(|(i, &l)| swapped_values[l] + (i as f64) * 1e-16)
        .collect();
    assert_eq!(
        canonical_labels(&leaf_assignment(&noisy, &values, 1e-9).unwrap()),
        canonical_labels(&leaf_assignment(&noisy_swapped, &swapped_values, 1e-9).unwrap())
    );

    // A contribution that is NOT a leaf value must be reported, never silently snapped.
    assert!(
        leaf_assignment(&[42.0], &values, 1e-9).is_none(),
        "a contribution that matches no leaf value must fail the recovery, not snap to the \
         nearest leaf"
    );
}

/// The `ctr_device_combo` params at the `T22-OBS-2` configuration: `simple_ctr = Borders`,
/// `combinations_ctr = Buckets`, `combinations_ctr_priors = [0.25]`, 20 iterations, depth 2.
///
/// The prior stays ≠ 0.5 even though the projection no longer NEEDS it. `T22-OBS-2`'s finding
/// is that a prior ≠ 0.5 is *necessary but not sufficient*, not that it is useless: at
/// `Prior = 0.5` the two Buckets columns are exact mirrors on EVERY threshold pair, so every
/// tree's Buckets level would be a tie and the arm would carry no information about which
/// column the device chose. Keeping the prior at 0.25 leaves most levels genuinely
/// discriminating, and the projection absorbs the residual ties.
///
/// Both prior lists are pinned explicitly (T22's reason: the default list is per-type, so an
/// implicit list would silently change the materialized CTR column count per arm).
fn buckets_params(prior: f64, iterations: usize, depth: usize) -> BoostParams {
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
        simple_ctr: ECtrType::Borders,
        simple_ctr_priors: vec![0.5],
        counter_calc_method: cb_train::counter_calc_method_default(),
        boosting_type: cb_train::boosting_type_default(),
        max_ctr_complexity: 2,
        combinations_ctr: ECtrType::Buckets,
        combinations_ctr_priors: vec![prior],
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

    use super::{buckets_params, canonical_labels, leaf_assignment};
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
    /// `crates/cb-train/tests/device_ctr_gate_test.rs` (the canonical copy) — keep in sync.
    /// TENTH copy.
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

    /// The CPU reference arm — overrides ONLY `compute_gradients` so both arms consume
    /// bit-identical derivatives and the differential isolates the GROWER. Every device-seam
    /// method inherits the `cb_compute::Runtime` trait default.
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
    }

    /// The counting wrapper around [`CpuRefRuntime`], so `grown == 0` /
    /// `accepted_begins == 0` are observations rather than assumptions (the R-8 guardrail).
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

    /// The raw `CtrSplitSpec` identity T22's Buckets arm compares — kept here ONLY so this file
    /// can report how many trees the strict comparison would have rejected. Nothing is asserted
    /// on it; see the module doc.
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

    /// Per-tree, per-object contribution `leaf_values[leaf_of(i)]`, recovered by differencing
    /// the staged main approx. Tree 0's predecessor is the starting approx, which is the
    /// all-zero vector here (`boost_from_average = false`, no bias) — and in any case is the
    /// SAME constant in both arms, and [`partition_labels`] is invariant to an additive
    /// constant, so the two comparisons below are unaffected either way.
    fn per_tree_contributions(staged: &[f64], n: usize, iterations: usize) -> Vec<Vec<f64>> {
        (0..iterations)
            .map(|t| {
                (0..n)
                    .map(|i| {
                        let cur = staged.get(t * n + i).copied().unwrap_or(0.0);
                        let prev = if t == 0 {
                            0.0
                        } else {
                            staged.get((t - 1) * n + i).copied().unwrap_or(0.0)
                        };
                        cur - prev
                    })
                    .collect()
            })
            .collect()
    }

    /// Device-vs-CPU bar on the per-object contribution. Both arms compute it by the same
    /// `calc_average` over the same partition from the same derivative trajectory, so the
    /// residual is f64 summation order only; measured max on `gfx1151` is ~1e-17.
    const CONTRIB_EPS: f64 = 1e-9;

    pub fn run(prior: f64, iterations: usize, depth: usize) {
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
        let params = buckets_params(prior, iterations, depth);
        assert_eq!(params.combinations_ctr_priors, vec![prior]);
        assert_eq!(params.combinations_ctr, cb_train::ECtrType::Buckets);

        // Fixture-permutation-divergence guard (GLOBALS §2.2): the structure order and the
        // averaging order must genuinely differ, or a structure-only leaf gather would pass.
        let structure = create_shuffled_indices(n, params.random_seed);
        let averaging = averaging_ctr_permutation(n, 1, params.random_seed);
        assert_ne!(
            structure, averaging,
            "structure and averaging permutations coincide at (n={n}, seed={}) — the fixture \
             cannot discriminate a structure-only leaf gather",
            params.random_seed
        );

        // ---- arm 1: the DEVICE grower ----
        let gpu = CountingGpu { inner: GpuBackend::default(), grown: Cell::new(0) };
        let mut dev_staged: Vec<f64> = Vec::new();
        let (dev, _dev_baked) = train_cat(
            &gpu,
            &columns,
            &borders,
            &cat_columns,
            &target,
            &[],
            &params,
            Some(&mut dev_staged),
        )
        .unwrap_or_else(|e| panic!("device Buckets train failed: {e:?}"));

        // ---- arm 2: the CPU grower, same inputs, same gradients ----
        let cpu = CountingCpu {
            inner: CpuRefRuntime { inner: GpuBackend::default() },
            grown: Cell::new(0),
            begins: Cell::new(0),
            accepted_begins: Cell::new(0),
        };
        let mut cpu_staged: Vec<f64> = Vec::new();
        let (host, _host_baked) = train_cat(
            &cpu,
            &columns,
            &borders,
            &cat_columns,
            &target,
            &[],
            &params,
            Some(&mut cpu_staged),
        )
        .unwrap_or_else(|e| panic!("cpu Buckets train failed: {e:?}"));

        // ---- the arms are really what they claim to be ----
        assert_eq!(
            gpu.grown.get(),
            params.iterations,
            "the Buckets fit must COMMIT to the device: expected {} device grows, got {}",
            params.iterations,
            gpu.grown.get()
        );
        assert_eq!(
            cpu.grown.get(),
            0,
            "the reference arm must run the CPU grower — otherwise this is a device-vs-device \
             tautology (R-8)"
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
        assert_eq!(
            dev_staged.len(),
            n * params.iterations,
            "the device arm's staged output must carry one full approx per iteration — the \
             whole projection is derived from it"
        );
        assert_eq!(cpu_staged.len(), n * params.iterations);

        // ---- vacuity, BEFORE the equality: the combination Buckets path is actually
        //      exercised. A projection equality over two CTR-free models says nothing. ----
        let combos = |trees: &[cb_train::ObliviousTree]| -> (usize, usize) {
            let total: usize = trees.iter().map(|t| t.ctr_splits.len()).sum();
            let two_plus = trees
                .iter()
                .flat_map(|t| t.ctr_splits.iter())
                .filter(|s| s.projection.cat_features().len() >= 2)
                .count();
            (total, two_plus)
        };
        let (dev_ctr, dev_combo) = combos(&dev.oblivious_trees);
        let (cpu_ctr, cpu_combo) = combos(&host.oblivious_trees);
        assert!(
            dev_ctr >= 1 && cpu_ctr >= 1,
            "both arms must contain ≥1 CTR split (device {dev_ctr}, cpu {cpu_ctr})"
        );
        assert!(
            dev_combo >= 1 && cpu_combo >= 1,
            "both arms must contain ≥1 COMBINATION (≥2-member) Buckets split (device \
             {dev_combo}, cpu {cpu_combo}). Without one the combination Buckets path is \
             untested and this differential is trivially satisfied. Escalate the horizon \
             (iterations, then depth) rather than weakening this guard."
        );
        // Every ≥2-member projection must really carry the Buckets type — the descriptor
        // COUNT does not discriminate the CTR type (COORDINATOR-FINDINGS T07).
        for (arm, trees) in [("device", &dev.oblivious_trees), ("cpu", &host.oblivious_trees)] {
            for (ti, tree) in trees.iter().enumerate() {
                for sig in ctr_sigs(tree) {
                    let expect = if sig.projection.len() >= 2 {
                        cb_train::ECtrType::Buckets.as_i8()
                    } else {
                        cb_train::ECtrType::Borders.as_i8()
                    };
                    assert_eq!(
                        sig.ctr_type, expect,
                        "[{arm}] tree {ti}: a {}-member projection carried ctr_type {}, \
                         expected {expect} ({sig:?})",
                        sig.projection.len(),
                        sig.ctr_type
                    );
                }
            }
        }

        // ---- the raw strict comparison, REPORTED not asserted (module doc) ----
        let strict_divergent: Vec<usize> = dev
            .oblivious_trees
            .iter()
            .zip(host.oblivious_trees.iter())
            .enumerate()
            .filter(|(_, (d, h))| ctr_sigs(d) != ctr_sigs(h) || d.splits != h.splits)
            .map(|(ti, _)| ti)
            .collect();

        println!(
            "[device-ctr-buckets-long-horizon] prior={prior} iters={iterations} depth={depth} \
             device: {dev_ctr} CTR splits ({dev_combo} ≥2-member) | cpu: {cpu_ctr} ({cpu_combo}) \
             | trees whose RAW split identity diverges: {} {strict_divergent:?} \
             | device grows = {}, cpu device-grows = {}",
            strict_divergent.len(),
            gpu.grown.get(),
            cpu.grown.get()
        );

        // ---- (1) the PARTITION-INVARIANT PROJECTION, per tree ----
        let dev_contrib = per_tree_contributions(&dev_staged, n, params.iterations);
        let cpu_contrib = per_tree_contributions(&cpu_staged, n, params.iterations);
        let mut distinct_groups = 0usize;
        for (ti, (d, h)) in dev_contrib.iter().zip(cpu_contrib.iter()).enumerate() {
            let dv = &dev.oblivious_trees[ti].leaf_values;
            let hv = &host.oblivious_trees[ti].leaf_values;
            let dl = canonical_labels(&leaf_assignment(d, dv, CONTRIB_EPS).unwrap_or_else(|| {
                panic!(
                    "device tree {ti}: a recovered per-object contribution matches NO leaf \
                     value within {CONTRIB_EPS:.0e} — the staged differencing does not \
                     reproduce `leaf_values[leaf_of(i)]`, so the projection below would be \
                     comparing nonsense. leaf_values = {dv:?}"
                )
            }));
            let hl = canonical_labels(&leaf_assignment(h, hv, CONTRIB_EPS).unwrap_or_else(|| {
                panic!(
                    "cpu tree {ti}: a recovered per-object contribution matches NO leaf value \
                     within {CONTRIB_EPS:.0e}. leaf_values = {hv:?}"
                )
            }));
            distinct_groups = distinct_groups.max(dl.iter().copied().max().unwrap_or(0) + 1);
            assert_eq!(
                dl, hl,
                "tree {ti}: the device and CPU growers induce DIFFERENT partitions of the \
                 training objects. This is a real routing divergence — it is exactly what the \
                 partition-invariant projection is NOT allowed to absorb (a benign `b=0`/`b=1` \
                 Buckets tie leaves the partition untouched; see the module doc). Raw split \
                 identity diverged on trees {strict_divergent:?}."
            );
        }
        // A tree that puts every object in one group would make the equality above vacuous.
        assert!(
            distinct_groups >= 2,
            "every tree collapsed the objects into a single group — the projection equality is \
             then trivially satisfied and this file asserts nothing"
        );

        // ---- (2) the per-object CONTRIBUTION VALUES, the independent check that covers the
        //          one way the projection is conservative (two leaves with equal values) ----
        let mut max_delta = 0.0_f64;
        for (ti, (d, h)) in dev_contrib.iter().zip(cpu_contrib.iter()).enumerate() {
            for (i, (&a, &b)) in d.iter().zip(h.iter()).enumerate() {
                let abs = (a - b).abs();
                max_delta = max_delta.max(abs);
                assert!(
                    abs <= CONTRIB_EPS,
                    "tree {ti} object {i}: device contribution {a} vs cpu {b}, |Δ| = {abs:.3e} \
                     exceeds ε = {CONTRIB_EPS:.0e}"
                );
            }
        }
        println!(
            "[device-ctr-buckets-long-horizon] partitions IDENTICAL across {iterations} trees \
             (≤ {distinct_groups} groups/tree); max |Δcontribution| = {max_delta:.3e} (bar \
             {CONTRIB_EPS:.0e})"
        );
    }
}

/// `T22-OBS-2` — combination × **Buckets** at `Prior = 0.25`, **20 iterations / depth 2**:
/// the exact configuration at which T22 measured the surviving `b=0`/`b=1` tie (tree 12,
/// level 1), and which its own arm therefore could not ship. Under the partition-invariant
/// projection the differential holds at that horizon.
#[test]
fn combination_buckets_matches_the_cpu_grower_at_a_long_horizon_under_a_partition_projection() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    device::run(0.25, 20, 2);
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    {
        let _ = buckets_params(0.25, 20, 2);
        eprintln!(
            "SKIP combination_buckets_matches_the_cpu_grower_at_a_long_horizon_under_a_\
             partition_projection: needs rocm/cuda"
        );
    }
}

/// The same projection at `Prior = 0.5`, where the two Buckets columns are EXACT mirrors on
/// every threshold pair, so essentially every Buckets level is a tie and the raw identity
/// comparison is uninformative by construction (T10 §2 — the degeneracy that forced T22 to a
/// prior ≠ 0.5 in the first place).
///
/// This is the arm the strict comparison cannot express at all. It is deliberately kept
/// SEPARATE from the 0.25 arm rather than replacing it: at this prior the projection is blind
/// to which of the mirrored columns each grower chose, so on its own it would be a weaker
/// statement. Together the two arms say: the growers agree observably at both priors, and at
/// 0.25 they additionally agree on almost every raw identity.
#[test]
fn combination_buckets_matches_the_cpu_grower_at_the_mirrored_prior() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    device::run(0.5, 20, 2);
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    {
        let _ = buckets_params(0.5, 20, 2);
        eprintln!(
            "SKIP combination_buckets_matches_the_cpu_grower_at_the_mirrored_prior: needs \
             rocm/cuda"
        );
    }
}
