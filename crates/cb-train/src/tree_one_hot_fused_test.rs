//! SPEC-OH-06 / SPEC-OH-07 — the FUSED one-hot-aware level search.
//!
//! The frozen `grow_one_hot_tree` / `select_level_one_hot` pair (`tree.rs`) is
//! the CORRECTNESS REFERENCE: it re-scans the whole dataset per candidate
//! (`score_candidate_any`), which is O(candidates · n) and unusable in
//! production, but it is unambiguous. The fused path added by SPEC-OH-06 scores
//! one-hot candidates through the SAME per-feature histogram machinery the
//! floats use (build at level 0, subtraction-trick derive after) — a pure
//! SPEEDUP, not a different algorithm.
//!
//! These tests pin that equivalence: same data, same params, identical chosen
//! splits and identical per-object leaf assignment. If they ever diverge, the
//! fused path has drifted and the reference is right.
//!
//! Sibling `#[path]` mount of `tree.rs` (source/test separation, CLAUDE.md).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp
)]

use super::{
    greedy_tensor_search_oblivious_perturbed, grow_one_hot_tree, AnySplit, FeatureMatrix, LevelKind,
    OneHotSplit, Split,
};
use cb_compute::EScoreFunction;

const N: usize = 200;
const SCALED_L2: f64 = 3.0;

/// 2 float columns (3 borders each) + 2 one-hot bin columns (cardinality 2 and
/// 3), with deterministic derivatives that make several levels genuinely prefer
/// a categorical split.
struct Pool {
    floats: Vec<Vec<f32>>,
    borders: Vec<Vec<f64>>,
    cat_bins: Vec<Vec<u32>>,
    der1: Vec<f64>,
    weight: Vec<f64>,
}

fn pool() -> Pool {
    let floats = vec![
        (0..N).map(|i| (i % 7) as f32 / 7.0).collect::<Vec<f32>>(),
        (0..N).map(|i| (i % 11) as f32 / 11.0).collect::<Vec<f32>>(),
    ];
    let borders = vec![vec![0.2, 0.5, 0.8], vec![0.25, 0.5, 0.75]];
    // Cardinality 4 and 3 — deliberately NOT binary. On a BINARY column the
    // candidates `cat == 0` and `cat == 1` induce the SAME partition (they are
    // complements), so their scores are mathematically identical and the winner
    // is decided by last-bit summation noise. Since the fused path sums in a
    // different (subtraction-trick) order than the per-candidate rescan
    // reference, such a column makes the equivalence comparison a test of
    // floating-point association rather than of the algorithm.
    let cat_bins = vec![
        (0..N).map(|i| (i % 4) as u32).collect::<Vec<u32>>(),
        (0..N).map(|i| (i % 3) as u32).collect::<Vec<u32>>(),
    ];
    // A signal carried by THREE INDEPENDENT axes — cat column 0, cat column 1,
    // and float column 0 — so a depth-3 tree has a distinct, well-separated
    // winner at every level. Without that separation a level's argmax is
    // decided by last-bit summation noise (measured: 1.1e-13 on a score of
    // ~6.8e2 for two mathematically identical candidates), which the fused and
    // reference paths legitimately resolve differently because they sum in
    // different orders.
    let der1: Vec<f64> = (0..N)
        .map(|i| {
            let a = f64::from(i % 4 == 2); // cat column 0
            let b = f64::from(i % 3 == 1); // cat column 1
            let c = f64::from(i % 7 >= 4); // float column 0 (value (i%7)/7 > 0.5)
            3.0 * a - 2.0 * b + 4.0 * c + ((i % 5) as f64) * 0.1 - 0.4
        })
        .collect();
    let weight = vec![1.0_f64; N];
    Pool {
        floats,
        borders,
        cat_bins,
        der1,
        weight,
    }
}

fn matrix(p: &Pool) -> FeatureMatrix<'_> {
    FeatureMatrix {
        feature_values: &p.floats,
        feature_borders: &p.borders,
        cat_bins: &p.cat_bins,
    }
}

/// Rebuild the level-ordered `Vec<AnySplit>` from a `GrownTree`'s kind-grouped
/// vectors — the inverse of the split-back the grower performs (SPEC-OH-07).
fn level_ordered(tree: &super::GrownTree) -> Vec<AnySplit> {
    if tree.level_kinds.is_empty() {
        return tree.splits.iter().copied().map(AnySplit::Float).collect();
    }
    tree.level_kinds
        .iter()
        .filter_map(|k| match *k {
            LevelKind::Float(i) => tree.splits.get(i).copied().map(AnySplit::Float),
            LevelKind::OneHot(i) => tree.one_hot_splits.get(i).copied().map(AnySplit::OneHot),
            LevelKind::Ctr { .. } => None,
        })
        .collect()
}

/// SPEC-OH-06 — the fused search reproduces the frozen reference grower EXACTLY:
/// the same level-ordered split list and the same per-object leaf assignment.
#[test]
fn fused_one_hot_search_matches_the_frozen_reference_grower() {
    let p = pool();
    let m = matrix(&p);
    let depth = 3;

    let reference = grow_one_hot_tree(
        &m,
        &p.der1,
        &p.weight,
        SCALED_L2,
        depth,
        N,
        EScoreFunction::L2,
    )
    .expect("the frozen reference grower must succeed");

    let fused = greedy_tensor_search_oblivious_perturbed(
        &m,
        &p.der1,
        &p.weight,
        SCALED_L2,
        depth,
        N,
        /* perturb */ None,
        EScoreFunction::L2,
        /* penalties */ None,
    )
    .expect("the fused grower must succeed");

    let fused_levels = level_ordered(&fused);
    assert_eq!(
        fused_levels, reference.splits,
        "the fused level-ordered splits must equal the frozen reference's"
    );
    assert_eq!(
        fused.leaf_of, reference.leaf_of,
        "and the per-object leaf assignment must match element-for-element"
    );

    // The pool is chosen so this is not a vacuous float-only agreement.
    assert!(
        fused_levels
            .iter()
            .any(|s| matches!(s, AnySplit::OneHot(_))),
        "the pool must actually elect a one-hot split, else the test proves nothing"
    );
}

/// SPEC-OH-06 — the CPU twin of the device [C2] last-bin guard: when the HIGHEST
/// cat bin is the winning equality value, it must be selectable. An
/// off-by-one exclusion (the float path legitimately drops the last BORDER,
/// which has no right side) would silently make the top category unsplittable.
#[test]
fn fused_one_hot_search_selects_the_last_bin_when_it_wins() {
    // One float column with no useful signal, one 3-valued cat column whose
    // HIGHEST bin (2) carries the entire signal.
    let floats = vec![vec![0.5_f32; N]];
    let borders = vec![vec![0.25, 0.75]];
    let cat_bins = vec![(0..N).map(|i| (i % 3) as u32).collect::<Vec<u32>>()];
    let der1: Vec<f64> = (0..N)
        .map(|i| if i % 3 == 2 { 5.0 } else { -1.0 })
        .collect();
    let weight = vec![1.0_f64; N];
    let m = FeatureMatrix {
        feature_values: &floats,
        feature_borders: &borders,
        cat_bins: &cat_bins,
    };

    let fused = greedy_tensor_search_oblivious_perturbed(
        &m, &der1, &weight, SCALED_L2, 1, N, None, EScoreFunction::L2, None,
    )
    .expect("grow");

    assert_eq!(
        level_ordered(&fused),
        vec![AnySplit::OneHot(OneHotSplit {
            feature: 0,
            value: 2
        })],
        "the HIGHEST cat bin must be a selectable equality candidate"
    );
}

/// SPEC-OH-31 — a matrix with NO cat columns takes exactly the pre-one-hot
/// path: `one_hot_splits` empty and `level_kinds` EMPTY (not a filled all-Float
/// vector), which is what keeps every float-only consumer byte-identical.
#[test]
fn float_only_growth_leaves_the_one_hot_vectors_empty() {
    let p = pool();
    let no_cats: Vec<Vec<u32>> = Vec::new();
    let m = FeatureMatrix {
        feature_values: &p.floats,
        feature_borders: &p.borders,
        cat_bins: &no_cats,
    };
    let fused = greedy_tensor_search_oblivious_perturbed(
        &m,
        &p.der1,
        &p.weight,
        SCALED_L2,
        3,
        N,
        None,
        EScoreFunction::L2,
        None,
    )
    .expect("grow");

    assert!(fused.one_hot_splits.is_empty());
    assert!(
        fused.level_kinds.is_empty(),
        "a float-only tree must leave level_kinds EMPTY (the legacy shape)"
    );
    assert_eq!(fused.splits.len(), 3);
}

/// SPEC-OH-07 — every one-hot level is recorded in `level_kinds` with an index
/// that actually addresses `one_hot_splits`, and the two kind-grouped vectors
/// together reconstruct the full depth (no level is dropped).
#[test]
fn level_kinds_index_both_vectors_and_cover_every_level() {
    let p = pool();
    let m = matrix(&p);
    let depth = 4;
    let fused = greedy_tensor_search_oblivious_perturbed(
        &m,
        &p.der1,
        &p.weight,
        SCALED_L2,
        depth,
        N,
        None,
        EScoreFunction::L2,
        None,
    )
    .expect("grow");

    assert_eq!(fused.level_kinds.len(), depth, "one kind per level");
    assert_eq!(
        fused.splits.len() + fused.one_hot_splits.len(),
        depth,
        "the kind-grouped vectors partition the levels exactly"
    );
    assert_eq!(level_ordered(&fused).len(), depth, "no level is dropped");
    for k in &fused.level_kinds {
        match *k {
            LevelKind::Float(i) => assert!(i < fused.splits.len()),
            LevelKind::OneHot(i) => assert!(i < fused.one_hot_splits.len()),
            LevelKind::Ctr { .. } => panic!("this grower emits no CTR levels"),
        }
    }
}

/// SPEC-OH-27 (T01b branch b) — one-hot candidates are NEVER enumerated while
/// `perturb.is_some()`: `train_inner`'s gate fires first, and the perturbed
/// level search itself refuses rather than silently changing its draw count.
#[test]
fn one_hot_candidates_are_never_enumerated_under_perturbation() {
    use cb_core::TFastRng64;

    let p = pool();
    let m = matrix(&p);
    let mut rng = TFastRng64::from_seed(0);
    let perturb = super::Perturbation {
        rng: &mut rng,
        score_st_dev: 1.0,
    };

    let got = greedy_tensor_search_oblivious_perturbed(
        &m,
        &p.der1,
        &p.weight,
        SCALED_L2,
        2,
        N,
        Some(perturb),
        EScoreFunction::L2,
        None,
    );
    assert!(
        got.is_err(),
        "a one-hot matrix under perturbation must be refused, not silently scored \
         with a desynchronised draw stream"
    );

    // …while the SAME perturbed search over a float-only matrix is unaffected.
    let no_cats: Vec<Vec<u32>> = Vec::new();
    let m2 = FeatureMatrix {
        feature_values: &p.floats,
        feature_borders: &p.borders,
        cat_bins: &no_cats,
    };
    let mut rng2 = TFastRng64::from_seed(0);
    let perturb2 = super::Perturbation {
        rng: &mut rng2,
        score_st_dev: 1.0,
    };
    assert!(greedy_tensor_search_oblivious_perturbed(
        &m2,
        &p.der1,
        &p.weight,
        SCALED_L2,
        2,
        N,
        Some(perturb2),
        EScoreFunction::L2,
        None,
    )
    .is_ok());
}

/// The float-only fused path is unchanged by the added one-hot machinery: with
/// no cat columns the chosen splits equal what the pre-one-hot search chose,
/// which we pin by construction — a single-float pool has exactly one right
/// answer per level and it must still be a `Split`, never an `AnySplit::OneHot`.
#[test]
fn float_only_levels_are_still_float_splits() {
    let floats = vec![(0..N).map(|i| i as f32 / N as f32).collect::<Vec<f32>>()];
    let borders = vec![vec![0.25, 0.5, 0.75]];
    let der1: Vec<f64> = (0..N).map(|i| if i * 2 < N { -1.0 } else { 1.0 }).collect();
    let weight = vec![1.0_f64; N];
    let no_cats: Vec<Vec<u32>> = Vec::new();
    let m = FeatureMatrix {
        feature_values: &floats,
        feature_borders: &borders,
        cat_bins: &no_cats,
    };
    let fused = greedy_tensor_search_oblivious_perturbed(
        &m, &der1, &weight, SCALED_L2, 1, N, None, EScoreFunction::L2, None,
    )
    .expect("grow");
    assert_eq!(
        fused.splits,
        vec![Split {
            feature: 0,
            border: 0.5
        }]
    );
}

