//! Unit tests for oblivious tree growth + the strict first-wins tie-break
//! (TRAIN-02, Pitfall 1). The tie-break is the parity landmine: equal-gain
//! candidates MUST resolve to the FIRST one in upstream candidate order
//! (feature index ascending, border ascending) via strict `gain > bestGain`.

use crate::tree::{combination_ctr_eligible, select_best_candidate, Candidate};
use crate::TProjection;

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

// ===========================================================================
// ORD-06-01/02 — `combination_ctr_eligible` (tree-structure-scoped combination-CTR
// candidate eligibility). Mirrors upstream `AddTreeCtrs`'s `seenProj` /
// `baseProj.IsEmpty()` gate (`greedy_tensor_search.cpp:503-568`), restricted to
// this codebase's categorical-only `TProjection` (see SPEC.md §1 "Codebase-
// specific simplification").
// ===========================================================================

/// AT-ORD06-01a: an empty `used_projections` (the tree root, no split chosen yet)
/// makes ANY combination projection ineligible.
#[test]
fn combination_ineligible_when_no_ctr_used_empty() {
    let projection = TProjection::from_features(&[0, 1]);
    assert!(!combination_ctr_eligible(&projection, &[]));
}

/// AT-ORD06-01b / ORD-06-02 scenario 2: a non-empty `used_projections` whose
/// single member is UNRELATED to the candidate (not a subset with a length gap of
/// exactly one) still yields `false` — this also stands in for "a tree with only
/// `Float` splits chosen," which never contributes a `Ctr`-derived entry to
/// `used_projections` at all (so the caller passes the same empty/unrelated
/// shape either way).
#[test]
fn combination_ineligible_when_used_is_unrelated() {
    let used = TProjection::single(5);
    let projection = TProjection::from_features(&[2, 3]);
    assert!(!combination_ctr_eligible(&projection, &[&used]));
}

/// ORD-06-02 scenario 1: `used_projections = [{0}]`, `projection = {0,1}` → `true`
/// (`{0}` has 1 member, `{0,1}` has 2, and `{0}` is a subset of `{0,1}`).
#[test]
fn combination_eligible_extends_simple_ctr() {
    let used = TProjection::single(0);
    let projection = TProjection::from_features(&[0, 1]);
    assert!(combination_ctr_eligible(&projection, &[&used]));
}

/// ORD-06-02 scenario 3: a length gap of TWO (`{0}` vs `{0,1,2}`) is NOT a
/// legitimate one-feature extension — `false`.
#[test]
fn combination_ineligible_length_gap_two() {
    let used = TProjection::single(0);
    let projection = TProjection::from_features(&[0, 1, 2]);
    assert!(!combination_ctr_eligible(&projection, &[&used]));
}

/// ORD-06-02 scenario 4: TWO distinct simple CTRs chosen (`{0}`, `{1}`) — the
/// candidate `{0,1}` is eligible via EITHER one (`any(...)` semantics).
#[test]
fn combination_eligible_via_any_of_multiple_used() {
    let used0 = TProjection::single(0);
    let used1 = TProjection::single(1);
    let projection = TProjection::from_features(&[0, 1]);
    assert!(combination_ctr_eligible(&projection, &[&used0, &used1]));
}

/// ORD-06-02 scenario 5: extending an already-chosen COMBINATION CTR (`{0,1}`) by
/// one more feature (`{0,1,2}`) is a legitimate extension — `true`.
#[test]
fn combination_eligible_extends_existing_combination() {
    let used = TProjection::from_features(&[0, 1]);
    let projection = TProjection::from_features(&[0, 1, 2]);
    assert!(combination_ctr_eligible(&projection, &[&used]));
}

/// ORD-06-02 scenario 6: a projection is never eligible against ITSELF (length
/// gap 0, not 1) — `false`.
#[test]
fn combination_ineligible_against_itself() {
    let used = TProjection::from_features(&[0, 1]);
    let projection = TProjection::from_features(&[0, 1]);
    assert!(!combination_ctr_eligible(&projection, &[&used]));
}

// ===========================================================================
// ORD-06-04 — `max_bucket_count` scoped to the per-level ELIGIBLE candidate set
// (plan-checker CRITICAL finding: `max_bucket_count` is a `cat_feature_weight`
// scoring INPUT and must use the SAME eligibility rule ORD-06-03 applies to
// `scored`, not the tree-wide unfiltered `ctr_features` list).
// ===========================================================================

use crate::ctr::CtrFeatureColumn;
use crate::tree::eligible_max_bucket_count;

/// A minimal synthetic [`CtrFeatureColumn`] carrying only `projection` and
/// `bucket_count` — the two fields [`eligible_max_bucket_count`] reads.
fn column_with_bucket_count(projection: TProjection, bucket_count: usize) -> CtrFeatureColumn {
    CtrFeatureColumn {
        projection,
        ctr_type: 0,
        prior_num: 0.5,
        prior_denom: 1.0,
        bins: Vec::new(),
        ctr_value: Vec::new(),
        bucket_count,
    }
}

/// AT-ORD06-04a: `chosen` empty (level 0) — the combination `{0,1}`
/// (bucket_count 20) is INELIGIBLE (ORD-06-01's root-level gate), so
/// `max_bucket_count` is the max over the two SIMPLE columns only: `5`, NOT `20`.
#[test]
fn max_bucket_count_excludes_ineligible_combination_at_root() {
    let cols = [
        column_with_bucket_count(TProjection::single(0), 5),
        column_with_bucket_count(TProjection::single(1), 4),
        column_with_bucket_count(TProjection::from_features(&[0, 1]), 20),
    ];
    let used_projections: Vec<&TProjection> = Vec::new();
    assert_eq!(eligible_max_bucket_count(&cols, &used_projections), 5);
}

/// AT-ORD06-04b: the SAME columns, but a `Ctr` split on projection `{0}` has
/// already been chosen — `{0,1}` is now ELIGIBLE per ORD-06-02, so
/// `max_bucket_count` correctly includes its `bucket_count` of `20`.
#[test]
fn max_bucket_count_includes_combination_once_eligible() {
    let cols = [
        column_with_bucket_count(TProjection::single(0), 5),
        column_with_bucket_count(TProjection::single(1), 4),
        column_with_bucket_count(TProjection::from_features(&[0, 1]), 20),
    ];
    let already_chosen = TProjection::single(0);
    let used_projections: Vec<&TProjection> = vec![&already_chosen];
    assert_eq!(eligible_max_bucket_count(&cols, &used_projections), 20);
}

/// AT-ORD06-04c (regression lock): `ctr_features` contains ONLY simple columns
/// (no combination at all) — the filter is a no-op, `max_bucket_count` is
/// IDENTICAL to the pre-fix unconditional `.max()` over every column.
#[test]
fn max_bucket_count_unchanged_for_all_simple_columns() {
    let cols = [
        column_with_bucket_count(TProjection::single(0), 5),
        column_with_bucket_count(TProjection::single(1), 4),
        column_with_bucket_count(TProjection::single(2), 9),
    ];
    let used_projections: Vec<&TProjection> = Vec::new();
    assert_eq!(eligible_max_bucket_count(&cols, &used_projections), 9);
}

// ===========================================================================
// ORD-07-01 — `phantom_mixed_bucket_count`: the number of DISTINCT
// `(current-partition-leaf, cat-value)` pairs actually observed in the learn
// sample, for ONE CTR-eligible categorical feature. Mirrors upstream's
// `binAndOneHotFeaturesTree`-derived phantom projection (`AddTreeCtrs`,
// `greedy_tensor_search.cpp:517-522` builds this base; `CalcMaxFeatureValueCount`,
// `:1097-1115` consumes its bucket count) — never itself a scoreable candidate
// in this codebase (categorical-only `TProjection`, ORD-06's simplification),
// it exists ONLY to correctly size `max_bucket_count` (SPEC.md ORD-07-01).
// ===========================================================================

use crate::tree::phantom_mixed_bucket_count;

/// AT-ORD07-01a: all 4 `(leaf, cat_bucket)` combinations are distinct → `4`.
#[test]
fn phantom_count_all_distinct() {
    let leaf_of = [0usize, 0, 1, 1];
    let cat_bucket = [0u32, 1, 0, 1];
    assert_eq!(phantom_mixed_bucket_count(&leaf_of, &cat_bucket), 4);
}

/// AT-ORD07-01b: a single leaf, 3 distinct cat values (one repeated) → `3`.
#[test]
fn phantom_count_single_leaf_repeated_value() {
    let leaf_of = [0usize, 0, 0, 0];
    let cat_bucket = [0u32, 1, 2, 0];
    assert_eq!(phantom_mixed_bucket_count(&leaf_of, &cat_bucket), 3);
}

/// AT-ORD07-01c: no objects → `0` (never panics, never divides).
#[test]
fn phantom_count_empty() {
    let leaf_of: [usize; 0] = [];
    let cat_bucket: [u32; 0] = [];
    assert_eq!(phantom_mixed_bucket_count(&leaf_of, &cat_bucket), 0);
}

/// AT-ORD07-01d: the SAME `(leaf, cat_bucket)` pair repeated across many objects
/// is counted ONCE (distinct pairs, not object count).
#[test]
fn phantom_count_repeated_pair_counted_once() {
    let leaf_of = [0usize, 0, 0];
    let cat_bucket = [5u32, 5, 5];
    assert_eq!(phantom_mixed_bucket_count(&leaf_of, &cat_bucket), 1);
}

// ===========================================================================
// ORD-07-02 — `phantom_bucket_gate`: the phantom contribution applies at a
// level iff `chosen` contains `>= 1` `CtrAwareSplit::Float` entry, mirroring
// `binAndOneHotFeaturesTree`'s non-empty condition (`greedy_tensor_search.cpp:
// 517-522`), which depends ONLY on chosen Float/one-hot splits — independent
// of, and additive to, ORD-06-04's existing "already-chosen `Ctr` projection"
// gating (SPEC.md ORD-07-02).
// ===========================================================================

use crate::tree::{phantom_bucket_gate, CtrAwareSplit, Split};

/// AT-ORD07-02a: `chosen = []` (level 0) → `false` — matches upstream's
/// `baseProj.IsEmpty()` skip and today's already-correct level-0 behavior.
#[test]
fn phantom_gate_false_when_chosen_empty() {
    let chosen: Vec<CtrAwareSplit> = Vec::new();
    assert!(!phantom_bucket_gate(&chosen));
}

/// AT-ORD07-02b: one simple CTR chosen, ZERO float splits → `false` —
/// `binAndOneHotFeaturesTree` remains empty regardless of CTR choices.
#[test]
fn phantom_gate_false_when_only_ctr_chosen() {
    let chosen = [CtrAwareSplit::Ctr { col: 0, border: 10.0 }];
    assert!(!phantom_bucket_gate(&chosen));
}

/// AT-ORD07-02c: one float split, zero CTR splits (the fixture's ACTUAL
/// level-1 state) → `true`.
#[test]
fn phantom_gate_true_when_float_chosen() {
    let chosen = [CtrAwareSplit::Float(Split {
        feature: 1,
        border: -0.2014,
    })];
    assert!(phantom_bucket_gate(&chosen));
}

/// AT-ORD07-02d: mixed (Float + Ctr) → `true` (only needs `>= 1` Float;
/// presence of a Ctr split doesn't disable it).
#[test]
fn phantom_gate_true_when_mixed() {
    let chosen = [
        CtrAwareSplit::Float(Split {
            feature: 1,
            border: -0.2014,
        }),
        CtrAwareSplit::Ctr { col: 0, border: 10.0 },
    ];
    assert!(phantom_bucket_gate(&chosen));
}

// ===========================================================================
// ORD-07-03 (AT-ORD07-03a) — `max_bucket_count_with_phantom` wires ORD-07-01/02
// into ORD-06-04's `eligible_max_bucket_count` output via a single outer
// `.max(...)`, mirroring tree0's exact level-0/1/2 states from the `fstr_ctr`
// fixture (SPEC.md §5 ORD-07-03's worked table: 5, 10, 20).
// ===========================================================================

use crate::tree::{assign_leaves_ctr_aware, max_bucket_count_with_phantom, FeatureMatrix};

/// Builds a float column of `n_leaves * per_leaf` objects where object group `g`
/// (of `per_leaf` consecutive objects) takes float value `((g >> feature_bit) &
/// 1) as f32` — i.e. bit `feature_bit` of the group index. With border `0.5` this
/// makes `passes_float` reproduce bit `feature_bit` of the forward-bit leaf index.
fn bit_float_column(n_leaves: usize, per_leaf: usize, feature_bit: u32) -> Vec<f32> {
    (0..n_leaves)
        .flat_map(|g| {
            let bit = ((g >> feature_bit) & 1) as f32;
            std::iter::repeat(bit).take(per_leaf)
        })
        .collect()
}

/// Builds a per-object cat-bucket column where each `per_leaf`-sized group
/// repeats a fixed pattern of `distinct_per_leaf` distinct values (padded with
/// the pattern's first value to fill `per_leaf`), so the TOTAL distinct
/// `(leaf, cat_bucket)` pair count is exactly `n_leaves * distinct_per_leaf`
/// (leaves never collide with each other's cat values in the pair count).
fn cat_column_with_distinct_per_leaf(
    n_leaves: usize,
    per_leaf: usize,
    distinct_per_leaf: usize,
) -> Vec<u32> {
    (0..n_leaves)
        .flat_map(|_| (0..per_leaf).map(|i| (i % distinct_per_leaf) as u32))
        .collect()
}

/// AT-ORD07-03a part 3 (regression lock): `chosen = []` (level 0) →
/// `max_bucket_count == 5`, UNCHANGED from ORD-06-04's current output — the
/// phantom gate is off, so the (irrelevant) `cat_eligible_buckets` inputs are
/// never consulted.
#[test]
fn max_bucket_count_unchanged_at_level0() {
    let values: Vec<Vec<f32>> = Vec::new();
    let borders: Vec<Vec<f64>> = Vec::new();
    let matrix = FeatureMatrix::new(&values, &borders);
    let chosen: Vec<CtrAwareSplit> = Vec::new();
    let cat_eligible_buckets: Vec<Vec<u32>> = vec![vec![0, 1, 2], vec![9, 9, 9]];
    let leaf_of = assign_leaves_ctr_aware(&matrix, &[], &chosen, 3);
    let result =
        max_bucket_count_with_phantom(&matrix, &[], &chosen, 3, 5, &cat_eligible_buckets, &leaf_of);
    assert_eq!(result, 5);
}

/// AT-ORD07-03a part 1: tree0's level-1 state (`chosen=[Float(1)@-0.2014]`) —
/// `cat_eligible_buckets` constructed so `phantom_mixed_bucket_count` yields `10`
/// for cat0 and `8` for cat1 → `max_bucket_count == max(5, 10, 8) == 10`.
#[test]
fn max_bucket_count_includes_phantom_at_level1() {
    let n_leaves = 2usize;
    let per_leaf = 5usize;
    let n = n_leaves * per_leaf;
    let feature0 = bit_float_column(n_leaves, per_leaf, 0);
    let values = vec![feature0];
    let borders = vec![vec![0.5_f64]];
    let matrix = FeatureMatrix::new(&values, &borders);
    let chosen = [CtrAwareSplit::Float(Split {
        feature: 0,
        border: 0.5, // the CHOSEN split's border drives assign_leaves_ctr_aware's
                     // `value > border` test directly (independent of the
                     // candidate-enumeration `borders` vector above) — it must
                     // match `bit_float_column`'s 0.0/1.0 group values.
    })];
    let cat0 = cat_column_with_distinct_per_leaf(n_leaves, per_leaf, 5); // -> 10
    let cat1 = cat_column_with_distinct_per_leaf(n_leaves, per_leaf, 4); // -> 8
    let cat_eligible_buckets = vec![cat0, cat1];

    let leaf_of = assign_leaves_ctr_aware(&matrix, &[], &chosen, n);
    let result =
        max_bucket_count_with_phantom(&matrix, &[], &chosen, n, 5, &cat_eligible_buckets, &leaf_of);
    assert_eq!(result, 10);
}

/// AT-ORD07-03a part 2: tree0's level-2 state
/// (`chosen=[Float(1)@-0.2014, Float(0)@0.561]`) — phantom counts `20`/`16` →
/// `max_bucket_count == max(5, 20, 16) == 20`.
#[test]
fn max_bucket_count_includes_phantom_at_level2() {
    let n_leaves = 4usize;
    let per_leaf = 5usize;
    let n = n_leaves * per_leaf;
    let feature0 = bit_float_column(n_leaves, per_leaf, 0);
    let feature1 = bit_float_column(n_leaves, per_leaf, 1);
    let values = vec![feature0, feature1];
    let borders = vec![vec![0.5_f64], vec![0.5_f64]];
    let matrix = FeatureMatrix::new(&values, &borders);
    let chosen = [
        CtrAwareSplit::Float(Split {
            feature: 0,
            border: 0.5,
        }),
        CtrAwareSplit::Float(Split {
            feature: 1,
            border: 0.5,
        }),
    ];
    let cat0 = cat_column_with_distinct_per_leaf(n_leaves, per_leaf, 5); // -> 20
    let cat1 = cat_column_with_distinct_per_leaf(n_leaves, per_leaf, 4); // -> 16
    let cat_eligible_buckets = vec![cat0, cat1];

    let leaf_of = assign_leaves_ctr_aware(&matrix, &[], &chosen, n);
    let result =
        max_bucket_count_with_phantom(&matrix, &[], &chosen, n, 5, &cat_eligible_buckets, &leaf_of);
    assert_eq!(result, 20);
}

/// Regression test for the bug the phantom partition MUST NOT reproduce: a `Ctr`
/// split chosen BEFORE a `Float` split in the same tree must NOT contribute a bit
/// to the phantom partition (`binAndOneHotFeaturesTree.BinFeatures` is
/// `currentTree.GetBinFeatures()` — Float/one-hot splits ONLY,
/// `greedy_tensor_search.cpp:517-522`). `chosen = [Ctr, Float]`: the FULL
/// partition is bijective over 4 objects (every object its own leaf, so the
/// buggy "count over the full chosen partition" always yields `4` — one object
/// per leaf, so every `(leaf, cat_bucket)` pair is trivially distinct regardless
/// of `cat_bucket`'s actual values), while the CORRECT Float-only partition
/// groups objects `{0,1}` and `{2,3}` into 2 leaves, so with `cat_bucket =
/// [7,7,8,8]` (one distinct cat value per leaf) the correct phantom count is `2`.
#[test]
fn max_bucket_count_phantom_excludes_ctr_split_from_partition() {
    let n = 4usize;
    // feature0: obj0,obj1 -> 0.0 (fails border 0.5); obj2,obj3 -> 1.0 (passes).
    let feature0 = vec![0.0_f32, 0.0, 1.0, 1.0];
    let values = vec![feature0];
    let borders = vec![vec![0.5_f64]];
    let matrix = FeatureMatrix::new(&values, &borders);

    let ctr_col = CtrFeatureColumn {
        projection: TProjection::single(0),
        ctr_type: 0,
        prior_num: 0.5,
        prior_denom: 1.0,
        bins: vec![0, 1, 0, 1], // obj0,obj2 fail border 0.5; obj1,obj3 pass.
        ctr_value: Vec::new(),
        bucket_count: 2,
    };
    let ctr_features = [ctr_col];

    // Ctr chosen FIRST (bit 0), Float chosen SECOND (bit 1) — the exact ordering
    // the buggy code mishandled: a Ctr split earlier in the tree than the Float
    // split whose phantom contribution is being sized.
    let chosen = [
        CtrAwareSplit::Ctr { col: 0, border: 0.5 },
        CtrAwareSplit::Float(Split { feature: 0, border: 0.5 }),
    ];
    // Full chosen (Ctr+Float) partition is bijective: leaves [0,1,2,3].
    let chosen_leaf_of = assign_leaves_ctr_aware(&matrix, &ctr_features, &chosen, n);
    assert_eq!(chosen_leaf_of, vec![0, 1, 2, 3]);

    let cat_eligible_buckets = vec![vec![7u32, 7, 8, 8]];
    let result = max_bucket_count_with_phantom(
        &matrix,
        &ctr_features,
        &chosen,
        n,
        1, // eligible_max, deliberately low so the phantom term drives the result
        &cat_eligible_buckets,
        &chosen_leaf_of,
    );
    // Correct (Float-only partition {0,1},{2,3}): 1 distinct cat value per leaf -> 2.
    // The bug (full Ctr+Float partition, bijective): every object its own leaf -> 4.
    assert_eq!(result, 2);
}
