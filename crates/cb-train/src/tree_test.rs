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
