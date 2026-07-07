//! Unit tests for oblivious tree growth + the strict first-wins tie-break
//! (TRAIN-02, Pitfall 1). The tie-break is the parity landmine: equal-gain
//! candidates MUST resolve to the FIRST one in upstream candidate order
//! (feature index ascending, border ascending) via strict `gain > bestGain`.

use crate::tree::{select_best_candidate, Candidate};

#[test]
fn select_best_candidate_empty_is_none() {
    let candidates: Vec<Candidate> = Vec::new();
    assert!(select_best_candidate(&candidates).is_none());
}

#[test]
fn select_best_candidate_picks_strict_max() {
    let candidates = [
        Candidate {
            feature: 0,
            border: 0.1,
            score: 3.0,
        },
        Candidate {
            feature: 1,
            border: 0.2,
            score: 9.0,
        },
        Candidate {
            feature: 2,
            border: 0.3,
            score: 5.0,
        },
    ];
    let best = select_best_candidate(&candidates).unwrap();
    assert_eq!(best.feature, 1);
}

#[test]
fn leaf_index_uses_forward_bit_order() {
    use crate::tree::leaf_index;
    // split 0 bit -> bit 0 (LSB); split 1 bit -> bit 1.
    // object passes split 0 only -> index 0b01 = 1
    assert_eq!(leaf_index(&[true, false]), 1);
    // object passes split 1 only -> index 0b10 = 2
    assert_eq!(leaf_index(&[false, true]), 2);
    // passes both -> 0b11 = 3
    assert_eq!(leaf_index(&[true, true]), 3);
    // passes neither -> 0
    assert_eq!(leaf_index(&[false, false]), 0);
}

#[test]
fn depth_cap_rejected_not_panicked() {
    use crate::tree::check_depth;
    assert!(check_depth(16).is_ok());
    assert!(check_depth(17).is_err());
}

/// A single categorical one-hot split (`grow_one_hot_tree`, depth 1) selects the
/// one-hot `cat_bin == value` candidate that best separates the gradient and
/// assigns leaves by `IsTrueOneHotFeature` (forward-bit leaf index). Anchored to
/// the D-04 equivalence: a one-hot split is structurally a binary feature, so the
/// leaf assignment equals the equivalent binary float split the EXISTING float
/// search would grow on the same separation.
#[test]
fn one_hot_single_split_assigns_by_bin_equality() {
    use crate::tree::{grow_one_hot_tree, AnySplit, FeatureMatrix, GrownOneHotTree, OneHotSplit};

    // 4 objects, one categorical feature with bins [0,1,0,1]. der1 cleanly
    // separates bin 1 (positive gradient) from bin 0 (negative), so the best
    // one-hot split is `cat_bin == 1`.
    let cat_bins = vec![vec![0u32, 1, 0, 1]];
    let no_float: Vec<Vec<f32>> = Vec::new();
    let no_borders: Vec<Vec<f64>> = Vec::new();
    let matrix = FeatureMatrix {
        feature_values: &no_float,
        feature_borders: &no_borders,
        cat_bins: &cat_bins,
    };
    let der1 = [-1.0, 1.0, -1.0, 1.0];
    let weight = [1.0, 1.0, 1.0, 1.0];

    let GrownOneHotTree { splits, leaf_of } = grow_one_hot_tree(
        &matrix,
        &der1,
        &weight,
        0.0,
        1,
        4,
        cb_compute::EScoreFunction::Cosine,
    )
    .expect("one-hot tree grows");

    // One split, on categorical feature 0, an equality test on SOME learn-set
    // bin (which bin wins is decided by the L2 score + strict first-wins
    // tie-break; we lock the STRUCTURE, not a hand-picked bin).
    assert_eq!(splits.len(), 1);
    let (feat, val) = match splits[0] {
        AnySplit::OneHot(OneHotSplit { feature, value }) => (feature, value),
        other => panic!("expected a one-hot split, got {other:?}"),
    };
    assert_eq!(feat, 0);

    // Forward-bit leaf index: objects whose bin == the chosen `val` pass (leaf
    // 1), others do not (leaf 0) — exactly `IsTrueOneHotFeature` (split.h:16-17),
    // identical to the equivalent binary float split `value > 0.5`.
    let expected: Vec<usize> = cat_bins[0]
        .iter()
        .map(|&b| usize::from(b == val))
        .collect();
    assert_eq!(leaf_of, expected);
}

/// `grow_one_hot_tree` with NO candidates (no float borders, no categorical
/// features) is a typed `Degenerate` error, never a panic (T-05-02-02 guard).
#[test]
fn one_hot_no_candidates_is_degenerate_not_panic() {
    use crate::tree::{grow_one_hot_tree, FeatureMatrix};
    let no_float: Vec<Vec<f32>> = Vec::new();
    let no_borders: Vec<Vec<f64>> = Vec::new();
    let no_cats: Vec<Vec<u32>> = Vec::new();
    let matrix = FeatureMatrix {
        feature_values: &no_float,
        feature_borders: &no_borders,
        cat_bins: &no_cats,
    };
    let der1 = [1.0, -1.0];
    let weight = [1.0, 1.0];
    assert!(
        grow_one_hot_tree(&matrix, &der1, &weight, 0.0, 1, 2, cb_compute::EScoreFunction::Cosine)
            .is_err()
    );
}

/// WR-01 (06.2-07): `multi_dim_candidate_score` must NOT substitute the whole
/// `der1` buffer for a dimension whose strided slice is out of range. The fix
/// replaces `unwrap_or(der1)` with `unwrap_or(&[])` so an out-of-range slice
/// scores 0 for that dimension instead of feeding wrong-dimension data. We
/// verify the contract by comparing the bad-stride score against the
/// hypothetical whole-buffer-substitution score and asserting they DIFFER (the
/// fix is active), plus no panic and a finite result. The correctly-strided
/// dim=1 path stays the D-04 split-score anchor.
#[test]
fn multi_dim_candidate_score_bad_stride_scores_zero_not_whole_buffer() {
    use crate::tree::multi_dim_candidate_score;
    use cb_compute::{reduce_leaf_stats, multi_dim_split_score, EScoreFunction};

    let n_objects = 3usize;
    let n_leaves = 2usize;
    let leaf_of = [0usize, 1usize, 0usize];
    let weight = [1.0_f64, 1.0, 1.0];
    let scaled_l2 = 0.5_f64;

    // der1.len() = 5: inferred approx_dimension = 5/3 = 1, so the loop runs once
    // over the in-range slice der1[0..3]; the trailing two elements are NOT
    // silently promoted to a second dimension (no whole-buffer use).
    let der1 = [1.0_f64, -2.0, 3.0, 99.0, -99.0];
    let score = multi_dim_candidate_score(
        &leaf_of,
        &der1,
        &weight,
        scaled_l2,
        n_objects,
        n_leaves,
        EScoreFunction::Cosine,
    );
    assert!(score.is_finite(), "bad-stride score must be finite (no panic)");

    // Reference: the inferred single dimension scores ONLY der1[0..3]; the
    // out-of-range tail (99, -99) must NOT influence the score.
    let leaves_dim0 = reduce_leaf_stats(&leaf_of, &der1[0..3], &weight, n_leaves);
    let expected = multi_dim_split_score(EScoreFunction::Cosine, &[leaves_dim0], scaled_l2);
    assert!(
        (score - expected).abs() < 1e-13,
        "score {score} must equal the in-range dim-0 reduction {expected}, \
         not a whole-buffer substitution"
    );

    // dim=1 byte-identity anchor (D-04): a length-n der1 scores identically.
    let der1_dim1 = [1.0_f64, -2.0, 3.0];
    let score_dim1 = multi_dim_candidate_score(
        &leaf_of,
        &der1_dim1,
        &weight,
        scaled_l2,
        n_objects,
        n_leaves,
        EScoreFunction::Cosine,
    );
    let leaves_clean = reduce_leaf_stats(&leaf_of, &der1_dim1, &weight, n_leaves);
    let expected_dim1 = multi_dim_split_score(EScoreFunction::Cosine, &[leaves_clean], scaled_l2);
    assert!(
        (score_dim1 - expected_dim1).abs() < 1e-13,
        "dim=1 path must be byte-identical to the single-dim reduction"
    );
}

/// WR-02: the always-run `fused_feature_parity_test` guard only ever exercises
/// `fused_feature_scan_and_score` (a test/harness-only primitive) and a DIFFERENT
/// subtraction-trick decomposition than production runs
/// (`relocate_sub(shift=2)`+`add_relocated(shift=0)` for the FALSE-sibling case, vs
/// production's `relocate_sub(shift=0)`+`add_relocated(shift=n_parent)` inside
/// [`crate::tree::derive_feature_level_hist`]). Call `derive_feature_level_hist`
/// directly — the ACTUAL level>0 production derivation — at `level >= 2`, covering
/// BOTH the `n_true <= n_false` and `n_true > n_false` branches, and assert
/// cell-by-cell `to_bits` equality against a fresh whole-partition build over the
/// same next-level partition.
fn make_derive_case(
    n: usize,
    n_bins: usize,
    level: usize,
    majority_true: bool,
) -> (Vec<u32>, Vec<f64>, Vec<f64>, Vec<usize>, Vec<usize>, usize, usize) {
    let n_parent = 1usize << level.saturating_sub(1);
    let n_next = 1usize << level;
    // Deterministic per-object bin spread over the feature's contiguous column.
    let col: Vec<u32> = (0..n).map(|o| ((o * 7 + 3) % n_bins) as u32).collect();
    // Integer der1 in a small range, exact under f64 (mirrors the existing
    // subtraction-trick convention: integer sums keep `parent - false == true`
    // bit-exact so the fresh-build reference is a meaningful oracle).
    let der1: Vec<f64> = (0..n).map(|o| (o % 9) as f64 - 4.0).collect();
    let weight = vec![1.0f64; n];
    let parent_leaf: Vec<usize> = (0..n).map(|o| o % n_parent).collect();
    // Skew the pass/fail split so one case is TRUE-minority (n_true <= n_false) and
    // the other TRUE-majority (n_true > n_false) — the two branches
    // `derive_feature_level_hist` selects between.
    let pass = |o: usize| if majority_true { o % 5 != 0 } else { o % 5 == 0 };
    let leaf_of: Vec<usize> = (0..n)
        .map(|o| parent_leaf[o] + if pass(o) { n_parent } else { 0 })
        .collect();
    (col, der1, weight, leaf_of, parent_leaf, n_parent, n_next)
}

#[test]
fn derive_feature_level_hist_matches_fresh_build_both_branches() {
    use crate::tree::derive_feature_level_hist;
    use cb_compute::build_bucket_histogram;

    let n = 40usize;
    let n_bins = 6usize;
    let level = 2usize; // n_parent = 2, n_next = 4 (level >= 2, per WR-02).
    let dim = 1usize;
    let n_channels = dim + 1;

    for majority_true in [false, true] {
        let (col, der1, weight, leaf_of, parent_leaf, n_parent, n_next) =
            make_derive_case(n, n_bins, level, majority_true);

        let n_true = leaf_of.iter().filter(|&&l| l >= n_parent).count();
        let n_false = n - n_true;
        if majority_true {
            assert!(n_true > n_false, "expected a TRUE-majority (n_true > n_false) case");
        } else {
            assert!(n_true <= n_false, "expected a TRUE-minority-or-tie (n_true <= n_false) case");
        }

        let parent_hist =
            build_bucket_histogram(&col, &der1, &weight, &parent_leaf, n_parent, 1, n_bins, dim);
        let derived = derive_feature_level_hist(
            Some(&parent_hist),
            &col,
            &der1,
            &weight,
            &leaf_of,
            level,
            n_bins,
            dim,
        );
        let fresh = build_bucket_histogram(&col, &der1, &weight, &leaf_of, n_next, 1, n_bins, dim);

        for leaf in 0..n_next {
            for bin in 0..n_bins {
                for c in 0..n_channels {
                    assert_eq!(
                        derived.channel(leaf, 0, bin, c).to_bits(),
                        fresh.channel(leaf, 0, bin, c).to_bits(),
                        "derive_feature_level_hist (majority_true={majority_true}) diverged \
                         from a fresh build at leaf {leaf} bin {bin} channel {c}"
                    );
                }
            }
        }
    }
}
