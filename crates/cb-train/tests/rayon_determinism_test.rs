//! PERF-03 determinism gate (Phase 21, Plan 05): the rayon-parallelized CPU
//! split search MUST produce a BYTE-IDENTICAL trained model on repeated runs.
//!
//! The per-level histogram build + border scoring is parallelized over INDEPENDENT
//! features (`into_par_iter` / `par_chunks_mut` in `tree.rs`) with an ordered
//! `collect` and each per-bin fold staying the sequential `cb_core::sum_f64`. That
//! makes the parallel sections deterministic BY CONSTRUCTION — no cross-feature
//! float reduction, no unordered merge (21-RESEARCH Pitfall 5). This test is the
//! empirical guard: train the same fixture twice under the live (multi-threaded)
//! rayon pool and assert the two [`Model`]s are equal field-for-field (structure +
//! leaf values + leaf weights, `Model: PartialEq`). A flicker here would mean the
//! merge is NOT feature-independent.
//!
//! Covers BOTH parallelized grow paths:
//! - `SymmetricTree` → `select_level_plain` (parallel border scoring) +
//!   `GrowScratch::new` (parallel binning).
//! - `Depthwise` → `best_split_for_leaf` (parallel per-leaf binning).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp
)]

use cb_compute::{rmse_der1, rmse_der2, Derivatives, EScoreFunction, LeafMethod, Loss, Runtime};
use cb_core::CbResult;
use cb_train::{train, BoostParams, EBootstrapType, EGrowPolicy, EOverfittingDetectorType, Model};

/// Pure host-CPU runtime (declines the device grow → the byte-unchanged host
/// oblivious/leaf-wise boosting loop runs), matching `perf_baseline_test.rs`.
struct CpuHostRuntime;
impl Runtime for CpuHostRuntime {
    fn compute_gradients(
        &self,
        _loss: &Loss,
        approx: &[f64],
        target: &[f64],
        _approx_dimension: usize,
    ) -> CbResult<Derivatives> {
        let der1 = approx.iter().zip(target).map(|(&a, &t)| rmse_der1(a, t)).collect();
        let der2 = approx.iter().zip(target).map(|(&a, &t)| rmse_der2(a, t)).collect();
        Ok(Derivatives { der1, der2 })
    }
}

/// Deterministic continuous workload (splitmix64 hash features, linear target) —
/// the same generator shape as `perf_baseline_test.rs`. Enough features (`nf`) so
/// the parallel-over-features build/scoring actually forks work across the pool.
fn gen(n: usize, nf: usize, nbins: usize) -> (Vec<Vec<f32>>, Vec<Vec<f64>>, Vec<f64>) {
    let mut cols = Vec::with_capacity(nf);
    for f in 0..nf {
        let col: Vec<f32> = (0..n)
            .map(|i| {
                let mut z = (i as u64)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add((f as u64).wrapping_mul(0xD1B5_4A32_D192_ED03));
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                ((z >> 11) as f64 / (1u64 << 53) as f64) as f32
            })
            .collect();
        cols.push(col);
    }
    let borders: Vec<Vec<f64>> =
        (0..nf).map(|_| (1..nbins).map(|k| k as f64 / nbins as f64).collect()).collect();
    let target: Vec<f64> = (0..n)
        .map(|i| {
            let mut acc = 0.0;
            for f in 0..nf.min(5) {
                acc += (cols[f][i] as f64) * (if f % 2 == 0 { 1.0 } else { -1.0 });
            }
            acc
        })
        .collect();
    (cols, borders, target)
}

fn params(grow_policy: EGrowPolicy, depth: usize, iterations: usize) -> BoostParams {
    BoostParams {
        loss: Loss::Rmse,
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
        random_seed: 42,
        od_type: EOverfittingDetectorType::None,
        od_pval: 0.0,
        od_wait: 0,
        use_best_model: false,
        eval_metric: None,
        auto_learning_rate: false,
        one_hot_max_size: cb_train::one_hot_max_size_default(),
        permutation_count: cb_train::permutation_count_default(),
        fold_len_multiplier: cb_train::fold_len_multiplier_default(),
        simple_ctr: cb_train::simple_ctr_default(),
        simple_ctr_priors: cb_train::simple_ctr_priors_default(),
        counter_calc_method: cb_train::counter_calc_method_default(),
        boosting_type: cb_train::boosting_type_default(),
        max_ctr_complexity: cb_train::max_ctr_complexity_default(),
        combinations_ctr: cb_train::combinations_ctr_default(),
        combinations_ctr_priors: cb_train::combinations_ctr_priors_default(),
        score_function: EScoreFunction::L2,
        has_time: false,
        feature_weights: cb_train::feature_weights_default(),
        first_feature_use_penalties: cb_train::first_feature_use_penalties_default(),
        per_object_feature_penalties: cb_train::per_object_feature_penalties_default(),
        penalties_coefficient: cb_train::penalties_coefficient_default(),
        monotone_constraints: cb_train::monotone_constraints_default(),
        grow_policy,
        max_leaves: cb_train::max_leaves_default(),
        min_data_in_leaf: cb_train::min_data_in_leaf_default(),
    }
}

fn train_once(grow_policy: EGrowPolicy, depth: usize, iterations: usize) -> Model {
    // Enough features (12) + rows (2000) that the parallel-over-features work is
    // real; enough bins (64) that each feature does non-trivial scanning.
    let (cols, borders, target) = gen(2_000, 12, 64);
    let p = params(grow_policy, depth, iterations);
    train(&CpuHostRuntime, &cols, &borders, &target, &[], &p, None)
        .unwrap_or_else(|e| panic!("train failed: {e:?}"))
}

/// SymmetricTree: exercises the parallel `select_level_plain` border scoring +
/// `GrowScratch::new` binning. Two runs under the live rayon pool must be equal.
#[test]
fn symmetric_tree_is_byte_identical_across_runs_under_rayon() {
    let a = train_once(EGrowPolicy::SymmetricTree, 6, 15);
    let b = train_once(EGrowPolicy::SymmetricTree, 6, 15);
    assert_eq!(
        a, b,
        "SymmetricTree model differs across two runs — parallel histogram \
         build/scoring is NOT deterministic (21-RESEARCH Pitfall 5)"
    );
    // A non-trivial model (real trees + real leaf values) so the equality is
    // meaningful, not a vacuous empty-model match.
    assert_eq!(a.oblivious_trees.len(), 15, "expected 15 boosting trees");
    assert!(
        a.oblivious_trees.iter().any(|t| !t.splits.is_empty()),
        "expected non-degenerate trees with splits"
    );
}

/// Depthwise: exercises the parallel `best_split_for_leaf` per-leaf binning. Two
/// runs under the live rayon pool must be equal.
#[test]
fn depthwise_leaf_wise_is_byte_identical_across_runs_under_rayon() {
    let a = train_once(EGrowPolicy::Depthwise, 6, 15);
    let b = train_once(EGrowPolicy::Depthwise, 6, 15);
    assert_eq!(
        a, b,
        "Depthwise model differs across two runs — parallel per-leaf binning is \
         NOT deterministic (21-RESEARCH Pitfall 5)"
    );
    assert_eq!(
        a.non_symmetric_trees.len(),
        15,
        "expected 15 non-symmetric (leaf-wise) boosting trees"
    );
    assert!(
        a.non_symmetric_trees.iter().any(|t| !t.splits.is_empty()),
        "expected non-degenerate leaf-wise trees with splits"
    );
}
