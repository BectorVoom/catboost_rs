//! DCTR-16 / D-2 — the **behavioural** detector for `eligible_max`'s eligibility filter
//! (`SPEC.md` risk **R-20**).
//!
//! # What R-20 was, and what closes it
//!
//! D-2 is one line of `crates/cb-backend/src/gpu_runtime/mod.rs`: pass C's `maxCount` input
//! must be maxed over the **eligible** CTR columns only
//! (`resident_eligible_max_bucket_count`), never over every column, mirroring upstream's
//! per-level `CalcMaxFeatureValueCount` (`greedy_tensor_search.cpp:1070-1088`, v1.2.10) and
//! the CPU's `eligible_max_bucket_count` (`cb-train/src/tree.rs`). T18 shipped it with unit
//! tests on the filtered fold and a source-level scan of the call site; `SPEC.md` DCTR-16
//! calls that "acceptable", but the plan checker amended R-20 to demand a test whose RESULT
//! changes when D-2 is un-wired. Three attempts on the frozen `ctr_device_combo` corpus
//! (T19's combo e2e, T22's split-sequence differential at its shipped configuration, and the
//! same differential at a 20-iteration horizon) all came back **byte-identical**, so R-20
//! stayed open with the honest note that "no committed fixture moves under it".
//!
//! **This file is that missing detector.** Un-wire D-2 and it fails at tree 0; wire it back
//! and it passes. The verbatim failure is transcribed in
//! `.planning/plans/device-ctr-full-coverage/notes/R20-CLOSURE.md`.
//!
//! # The mechanism, and therefore why every parameter below is load-bearing
//!
//! At **level 0** of every tree `chosen_ctr_projections` is empty, so §4.6's predicate makes
//! every ≥2-member (combination) column INELIGIBLE — upstream's `AddTreeCtrs` cannot even
//! construct such a candidate at level 0 (`baseProj.IsEmpty()` skip). D-2 is what keeps that
//! column's larger `bucket_count` out of `maxCount`. Level 0 is also the only level where the
//! difference can survive: from level 1 on, `phantom_max` (the float-partition × cat-bucket
//! pair count) is folded in outside the filter (⚠ C-16) and typically dominates both values.
//!
//! The consumer is `(1 + count/maxCount)^(-model_size_reg)`, applied to every CTR candidate
//! whose `(ctr_type, projection)` group is not yet used. It is **increasing in `maxCount`**,
//! so an unfiltered (inflated) `maxCount` raises EVERY CTR candidate's weight at once — which
//! is why the reachable flip is CTR-vs-**float**, a single threshold, rather than the much
//! rarer CTR-vs-CTR near-tie.
//!
//! On this pool the arithmetic is exact and is asserted below, not merely described:
//!
//! ```text
//!   simple [0] bucket_count = 5     simple [1] bucket_count = 5
//!   combination [0,1] bucket_count = 25          (5 x 5, every pair observed)
//!
//!   maxCount, FILTERED (correct, = the CPU's)  = 5   ->  weight = (1 + 5/5)^-0.5  = 0.70711
//!   maxCount, UNFILTERED (the D-2 defect)      = 25  ->  weight = (1 + 5/25)^-0.5 = 0.91287
//!
//!   => every level-0 CTR candidate's weighted gain moves up by a factor of 1.291.
//! ```
//!
//! A 29 % band is wide enough that a level-0 float candidate lands inside it on this data,
//! and it does: with D-2 un-wired the device picks a CTR split at tree 0 level 0 where the
//! CPU picks `Float(0)`, and the whole model diverges from there.
//!
//! ## ⚠ Do NOT "tidy" these parameters back towards the other CTR tests' values
//!
//! * **`k1 = k2 = 5` categories per column, with all 25 pairs observed.** This is the whole
//!   detector. It is what makes `maxCount` unfiltered/filtered = **5x**, hence the 29 % band.
//!   The frozen `ctr_device_combo` fixture's ratio is only ~3x (3/4 simple vs <=12 combined)
//!   and it demonstrably does **not** discriminate (T19 §5, T22 §4.1) — so a "let us just
//!   reuse the existing fixture" edit silently re-opens R-20. `PERTURBATION` below asserts
//!   the ratio is real before anything else runs.
//! * **`max_ctr_complexity = 2` and exactly two cat features.** With `1` there is no
//!   combination column, the filter is the identity, and this file becomes vacuous.
//! * **`one_hot_max_size = 1`.** Raising it to >= 5 turns both cat columns into one-hot
//!   features, no CTR column is materialized at all, and the fit stops being a CTR fit.
//! * **`n = 300`, `NF = 3` float features, the generator constants and `DATA_SEED = 0`.**
//!   Chosen by an explicit sweep (40 seeds x cardinalities x effect sizes, recorded in
//!   `notes/R20-CLOSURE.md`); they place a level-0 float candidate inside the band. Most
//!   configurations do not, which is exactly why R-20 stayed open for three tasks.
//! * **`iterations = 5, depth = 2`.** The divergence is at tree 0, so 5 is ample; the sweep
//!   confirmed the same MATCH/DIFF verdict at 3, 5, 6, 8 and 10 iterations, so this is not a
//!   knife-edge horizon.
//! * **`model_size_reg` is NOT a knob here.** It is fixed at the canonical default `0.5`
//!   (`boosting.rs`'s `model_size_reg_default`) on BOTH the device (`DeviceCtrConfig
//!   ::model_size_reg`) and the CPU grower, and `BoostParams` does not expose it. Raising it
//!   would widen the band further, but it is unreachable from a fit — so the cardinality
//!   ratio above is the only available amplifier, and it carries the whole detector.
//!
//! # Why device-vs-CPU rather than device-vs-device
//!
//! The CPU grower is the reference precisely because it **always** filters
//! (`eligible_max_bucket_count`). So this file is simultaneously a D-2 detector and a
//! cross-implementation parity pin: an un-wired D-2 makes the device disagree with the CPU,
//! which is the defect stated in its own terms. Predictions are deliberately NOT the
//! comparison surface — a weight-driven winner flip can be nearly loss-neutral, and T22's
//! module doc records two more reasons a prediction comparison is blind on CTR fits.
//!
//! The pool is **synthetic and generated in-test** (no new frozen fixture — R-12) and there
//! is therefore no upstream `predictions.npy` to compare against; the upstream half of the
//! chain is the shipped `ctr_mixed_simple_vs_combo_oracle_test` /
//! `tensor_ctr_e2e_oracle_test` / `device_ctr_combo_fit_test` set.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp
)]

use cb_compute::{EScoreFunction, LeafMethod, Loss};
use cb_train::{BoostParams, EBootstrapType, ECtrType, EOverfittingDetectorType};

/// Objects in the synthetic pool.
pub const N: usize = 300;
/// Float features (feature 0 carries the effect that competes with the CTR candidate).
pub const NF: usize = 3;
/// Categories per categorical column. `K * K == 25` distinct pairs is the 5x `maxCount`
/// ratio this detector runs on — see the module doc.
pub const K: usize = 5;
/// Generator seed (swept; see the module doc and `notes/R20-CLOSURE.md`).
pub const DATA_SEED: u64 = 0;
/// Float-effect scale in the label generator.
pub const FLOAT_COEF: f64 = 0.5;
/// Categorical-effect scale in the label generator.
pub const CAT_COEF: f64 = 1.0;

/// The fit parameters. Every CTR-relevant field is pinned explicitly (never a `*_default()`)
/// so a change to a default cannot silently move this detector off its configuration.
pub fn eligible_max_params() -> BoostParams {
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
        // 1 => neither 5-category column can become a one-hot feature; both stay CTR.
        one_hot_max_size: 1,
        permutation_count: 1,
        fold_len_multiplier: 2.0,
        simple_ctr: ECtrType::Borders,
        simple_ctr_priors: vec![0.5],
        counter_calc_method: cb_train::counter_calc_method_default(),
        boosting_type: cb_train::boosting_type_default(),
        // 2 => the {0,1} combination column EXISTS. At 1 this file is vacuous.
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
        extra: Default::default(),
    }
}

/// The generated pool. No fixture is read or written (R-12): the arrays are a pure function
/// of the constants above, so the detector is reproducible without a frozen artifact.
pub struct SyntheticPool {
    /// `columns[f]` — float feature `f`'s per-object values.
    pub columns: Vec<Vec<f32>>,
    /// `borders[f]` — 15 quantile borders for float feature `f` (the CatBoost default count).
    pub borders: Vec<Vec<f64>>,
    /// The two categorical columns, as the raw strings `train_cat` takes.
    pub cat_columns: Vec<Vec<String>>,
    /// Binary labels.
    pub target: Vec<f64>,
}

/// A 64-bit LCG (Knuth's MMIX constants) — deterministic on every platform and dependency
/// free, so the pool cannot drift with an `rand` version bump.
fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

/// Build the pool: `NF` uniform float features, two independent `K`-category columns, and a
/// Bernoulli label whose logit mixes a per-category effect with the float features. The
/// mixture is what puts a level-0 float candidate near — but on the correct side of — the
/// weighted CTR candidate.
#[cfg_attr(not(any(feature = "rocm", feature = "cuda")), allow(dead_code))]
pub fn make_pool() -> SyntheticPool {
    let mut s = DATA_SEED
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(12345);
    let mut columns: Vec<Vec<f32>> = (0..NF).map(|_| Vec::with_capacity(N)).collect();
    let mut c1v: Vec<usize> = Vec::with_capacity(N);
    let mut c2v: Vec<usize> = Vec::with_capacity(N);
    let mut target: Vec<f64> = Vec::with_capacity(N);

    let e1: Vec<f64> = (0..K).map(|_| lcg(&mut s) * 2.0 - 1.0).collect();
    let e2: Vec<f64> = (0..K).map(|_| lcg(&mut s) * 2.0 - 1.0).collect();

    for _ in 0..N {
        let mut fs = Vec::with_capacity(NF);
        for c in columns.iter_mut() {
            let v = lcg(&mut s);
            c.push(v as f32);
            fs.push(v);
        }
        let a = (lcg(&mut s) * K as f64).floor() as usize % K;
        let b = (lcg(&mut s) * K as f64).floor() as usize % K;
        c1v.push(a);
        c2v.push(b);
        let mut logit = CAT_COEF * (e1[a] + e2[b]);
        for (j, v) in fs.iter().enumerate() {
            let w = if j == 0 { 1.0 } else { 1.0 / (j as f64 + 1.0) };
            logit += FLOAT_COEF * w * (v - 0.5);
        }
        let p = 1.0 / (1.0 + (-logit).exp());
        target.push(if lcg(&mut s) < p { 1.0 } else { 0.0 });
    }

    let borders: Vec<Vec<f64>> = columns
        .iter()
        .map(|col| {
            let mut v: Vec<f32> = col.clone();
            v.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in a generated column"));
            let mut out: Vec<f64> = Vec::new();
            for i in 1..=15 {
                let idx = ((v.len() * i) / 16).min(v.len() - 1);
                let b = f64::from(v[idx]);
                if out.last().is_none_or(|l| *l < b) {
                    out.push(b);
                }
            }
            out
        })
        .collect();

    let cat_columns = vec![
        c1v.iter()
            .map(|&v| cb_data::stringify_int_category(v as i64))
            .collect(),
        c2v.iter()
            .map(|&v| cb_data::stringify_int_category(v as i64))
            .collect(),
    ];

    SyntheticPool {
        columns,
        borders,
        cat_columns,
        target,
    }
}

/// The EXACT `bucket_count` the split search sees for `projection`, obtained by running the
/// production materializer (`materialize_ctr_feature`) over this pool — not a host-side
/// re-derivation of the cardinality. `bucket_count` is
/// `TOnlineCtrUniqValuesCounts::Count` (`ctrs.h:50`), the `count` input of
/// `(1 + count/maxCount)^(-model_size_reg)`.
#[cfg_attr(not(any(feature = "rocm", feature = "cuda")), allow(dead_code))]
pub fn bucket_count_of(pool: &SyntheticPool, members: &[usize]) -> usize {
    let permutation: Vec<i32> = (0..pool.target.len() as i32).collect();
    let target_class: Vec<usize> = pool.target.iter().map(|&y| y as usize).collect();
    cb_train::materialize_ctr_feature(
        &pool.cat_columns,
        &cb_train::TProjection::from_features(members),
        &permutation,
        &target_class,
        /* prior_num = */ 0.5,
        /* prior_denom = */ 1.0,
        /* ctr_border_count = */ 15,
        ECtrType::Borders,
        /* target_border_idx = */ 0,
        /* extra_cat_columns = */ &[],
    )
    .expect("materialize_ctr_feature over the synthetic pool")
    .bucket_count
}

/// `GetCatFeatureWeight`'s core, re-stated here so the printed band is computed rather than
/// asserted from prose (`cb_train::tree::cat_feature_weight` is private).
#[cfg_attr(not(any(feature = "rocm", feature = "cuda")), allow(dead_code))]
pub fn cat_feature_weight(count: usize, max_count: usize, model_size_reg: f64) -> f64 {
    (1.0 + count as f64 / max_count as f64).powf(-model_size_reg)
}

/// The canonical `model_size_reg` both growers run at. It is not a `BoostParams` field; see
/// the module doc.
pub const MODEL_SIZE_REG: f64 = 0.5;

#[cfg(any(feature = "rocm", feature = "cuda"))]
mod device {
    use std::cell::Cell;

    use cb_backend::GpuBackend;
    use cb_compute::{
        Derivatives, DeviceGrownTree, DeviceTrainConfig, EScoreFunction, FamilyTreeArgs, Loss,
        Runtime,
    };
    use cb_core::CbResult;
    use cb_train::{averaging_ctr_permutation, create_shuffled_indices, train_cat};

    use super::{
        bucket_count_of, cat_feature_weight, eligible_max_params, make_pool, MODEL_SIZE_REG,
    };

    /// The device-commitment counter (GLOBALS §2.2.6). Copied **verbatim** from
    /// `crates/cb-train/tests/device_ctr_gate_test.rs` (the canonical copy) — keep in sync:
    /// every override forwards to `self.inner: GpuBackend` and only `grow_tree_on_device`
    /// counts, and only when it returns `Some` (a `None` is the device declining a tree).
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

    /// The CPU reference arm. It overrides **only** `compute_gradients`, forwarding to a real
    /// `GpuBackend` so BOTH arms consume bit-identical derivatives and the differential
    /// isolates the GROWER. Every device-seam method inherits the `cb_compute::Runtime` trait
    /// default (`begin_device_training -> Ok(false)`, `grow_tree_on_device -> Ok(None)`), the
    /// `device_nonsym_fit_test.rs` / `device_ctr_combo_types_diff_test.rs` precedent.
    /// `CpuBackend` itself is not compiled under `rocm`/`cuda` (GLOBALS §2.2.3).
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

    /// [`CountingGpu`]'s shape wrapping [`CpuRefRuntime`], so "the reference arm really ran
    /// the CPU grower" is an OBSERVATION (`grown == 0`, `accepted_begins == 0`) and not an
    /// assumption. Without it this file could be a device-vs-device tautology.
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

    /// The full identity of a chosen CTR split — the key `assign_leaf_over_ctr_columns`
    /// itself uses. `shift`/`scale` are prior-derived and add no discrimination.
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

    fn ctr_split_counts(trees: &[cb_train::ObliviousTree]) -> (usize, usize) {
        let total: usize = trees.iter().map(|t| t.ctr_splits.len()).sum();
        let combos = trees
            .iter()
            .flat_map(|t| t.ctr_splits.iter())
            .filter(|s| s.projection.cat_features().len() >= 2)
            .count();
        (total, combos)
    }

    pub fn run() {
        let pool = make_pool();
        let params = eligible_max_params();
        let n = pool.target.len();

        // ---- PERTURBATION: the two arms' `maxCount` must genuinely DIFFER at level 0 ----
        // Asserted BEFORE anything is trained. If the ineligible combination's bucket count
        // does not exceed the simple columns' max, the filtered and unfiltered `eligible_max`
        // coincide, the input perturbation is zero, and no split-sequence comparison could
        // possibly detect D-2 — which is precisely the state the `ctr_device_combo` corpus
        // left R-20 in for three tasks. See the module doc.
        let bc_a = bucket_count_of(&pool, &[0]);
        let bc_b = bucket_count_of(&pool, &[1]);
        let bc_ab = bucket_count_of(&pool, &[0, 1]);
        let filtered_max = bc_a.max(bc_b);
        let unfiltered_max = filtered_max.max(bc_ab);
        let w_filtered = cat_feature_weight(filtered_max, filtered_max, MODEL_SIZE_REG);
        let w_unfiltered = cat_feature_weight(filtered_max, unfiltered_max, MODEL_SIZE_REG);
        println!(
            "[device-ctr-eligible-max-diff] bucket_counts: [0]={bc_a} [1]={bc_b} [0,1]={bc_ab} \
             | level-0 maxCount filtered={filtered_max} unfiltered={unfiltered_max} \
             | cat_feature_weight {w_filtered:.5} -> {w_unfiltered:.5} \
             (band x{:.3} at model_size_reg={MODEL_SIZE_REG})",
            w_unfiltered / w_filtered
        );
        assert!(
            bc_ab > filtered_max,
            "the {}-member combination's bucket_count ({bc_ab}) must strictly EXCEED the \
             simple columns' max ({filtered_max}), or the filtered and unfiltered `maxCount` \
             coincide and this differential cannot detect D-2 at all",
            2
        );
        assert!(
            w_unfiltered / w_filtered > 1.25,
            "the cat-feature-weight band this detector runs on collapsed to x{:.3} (was x1.291 \
             at 5/5/25 bucket counts). Restore the pool's cardinalities — see the module doc's \
             \"Do NOT tidy these parameters\".",
            w_unfiltered / w_filtered
        );

        // Structure-vs-averaging permutation divergence guard (GLOBALS §2.2): without it a
        // structure-only leaf gather would pass.
        assert_ne!(
            create_shuffled_indices(n, params.random_seed),
            averaging_ctr_permutation(n, 1, params.random_seed),
            "structure and averaging permutations coincide at (n={n}, seed={}) — the pool \
             cannot discriminate a structure-only leaf gather",
            params.random_seed
        );

        // ---- arm 1: the DEVICE grower ----
        let gpu = CountingGpu {
            inner: GpuBackend::default(),
            grown: Cell::new(0),
        };
        let (dev, _) = train_cat(
            &gpu,
            &pool.columns,
            &pool.borders,
            &pool.cat_columns,
            &pool.target,
            &[],
            &params,
            None,
        )
        .unwrap_or_else(|e| panic!("device CTR train failed: {e:?}"));

        // ---- arm 2: the CPU grower, same inputs, same gradients ----
        let cpu = CountingCpu {
            inner: CpuRefRuntime {
                inner: GpuBackend::default(),
            },
            grown: Cell::new(0),
            begins: Cell::new(0),
            accepted_begins: Cell::new(0),
        };
        let (host, _) = train_cat(
            &cpu,
            &pool.columns,
            &pool.borders,
            &pool.cat_columns,
            &pool.target,
            &[],
            &params,
            None,
        )
        .unwrap_or_else(|e| panic!("cpu CTR train failed: {e:?}"));

        let (dev_ctr, dev_combo) = ctr_split_counts(&dev.oblivious_trees);
        let (cpu_ctr, cpu_combo) = ctr_split_counts(&host.oblivious_trees);
        println!(
            "[device-ctr-eligible-max-diff] device: {dev_ctr} CTR splits ({dev_combo} \
             >=2-member) | cpu: {cpu_ctr} CTR splits ({cpu_combo} >=2-member) | device grows \
             = {}, cpu device-grows = {} (begins {} / accepted {})",
            gpu.grown.get(),
            cpu.grown.get(),
            cpu.begins.get(),
            cpu.accepted_begins.get()
        );

        // ---- (1) the device arm really COMMITTED; the CPU arm really did NOT ----
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

        // ---- (2) vacuity guards, BEFORE the equality ----
        assert_eq!(dev.oblivious_trees.len(), params.iterations);
        assert_eq!(host.oblivious_trees.len(), params.iterations);
        assert!(dev.non_symmetric_trees.is_empty() && dev.region_trees.is_empty());
        assert!(host.non_symmetric_trees.is_empty() && host.region_trees.is_empty());
        assert!(
            dev_ctr >= 1 && cpu_ctr >= 1,
            "both arms must contain >=1 CTR split (device {dev_ctr}, cpu {cpu_ctr}) — a \
             CTR-free differential asserts nothing"
        );
        assert!(
            dev_combo >= 1 && cpu_combo >= 1,
            "both arms must contain >=1 COMBINATION (>=2-member) CTR split (device \
             {dev_combo}, cpu {cpu_combo}); without one the combination column is materialized \
             but never exercised. NEVER weaken this guard — raise `iterations` instead."
        );

        // ---- (3) SPLIT-SEQUENCE EQUALITY, per tree. THIS is the D-2 detector: with D-2
        //          un-wired the device's level-0 winner at tree 0 flips from `Float(0)` to a
        //          CTR split and this assertion fails. ----
        for (ti, (d, h)) in dev
            .oblivious_trees
            .iter()
            .zip(host.oblivious_trees.iter())
            .enumerate()
        {
            assert_eq!(
                d.splits, h.splits,
                "tree {ti}: the FLOAT split sequence diverges between the device and CPU \
                 growers. If the device chose a CTR split where the CPU chose a float, check \
                 D-2 first: an UNFILTERED `eligible_max` inflates `maxCount`, raises every \
                 CTR candidate's `(1 + count/maxCount)^-0.5` weight, and lets a CTR candidate \
                 overtake the float winner (DCTR-16 / R-20)"
            );
            assert_eq!(
                ctr_sigs(d),
                ctr_sigs(h),
                "tree {ti}: the CTR split sequence diverges between the device and CPU growers \
                 (full identity: projection, ctr_type, prior_num, prior_denom, \
                 target_border_idx, border)"
            );
            assert_eq!(
                d.one_hot_splits, h.one_hot_splits,
                "tree {ti}: the ONE-HOT split sequence diverges (both must be empty here)"
            );
        }

        // ---- (4) leaf values within eps = 1e-4 (D-07's device-vs-CPU bar; this is a
        //          self-oracle, not the 1e-5 upstream bar) ----
        let mut max_leaf_delta = 0.0_f64;
        for (ti, (d, h)) in dev
            .oblivious_trees
            .iter()
            .zip(host.oblivious_trees.iter())
            .enumerate()
        {
            assert_eq!(d.leaf_values.len(), h.leaf_values.len(), "tree {ti} leaf count");
            for (li, (&a, &b)) in d.leaf_values.iter().zip(h.leaf_values.iter()).enumerate() {
                let abs = (a - b).abs();
                max_leaf_delta = max_leaf_delta.max(abs);
                assert!(
                    abs <= 1e-4,
                    "tree {ti} leaf {li}: device {a} vs cpu {b} exceeds eps=1e-4 (|Δ|={abs:.3e})"
                );
            }
        }
        println!(
            "[device-ctr-eligible-max-diff] split sequences IDENTICAL across {} trees; \
             max |Δleaf| = {max_leaf_delta:.3e} (bar 1e-4)",
            params.iterations
        );
    }
}

/// DCTR-16 / D-2 / **R-20** — the behavioural detector.
///
/// Un-wire `resident_eligible_max_bucket_count` at pass C's `eligible_max` call site (i.e.
/// restore the pre-T18 `cs.bucket_counts.iter().copied().max().unwrap_or(1).max(1)`) and this
/// test fails at tree 0 on the FLOAT split sequence: the device picks a CTR split at level 0
/// where the CPU picks `Float(0)`. The verbatim failure is in
/// `.planning/plans/device-ctr-full-coverage/notes/R20-CLOSURE.md`.
#[test]
fn an_ineligible_combination_must_not_inflate_the_level_0_max_bucket_count() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    device::run();
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    {
        let _ = eligible_max_params();
        eprintln!(
            "SKIP an_ineligible_combination_must_not_inflate_the_level_0_max_bucket_count: \
             needs rocm/cuda"
        );
    }
}
