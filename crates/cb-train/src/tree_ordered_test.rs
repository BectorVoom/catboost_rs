//! Unit tests for the ORD-02 ordered split-scoring subsystem
//! ([`greedy_tensor_search_oblivious_ordered`] and helpers).
//!
//! These lock the structural heart of Ordered boosting (per-segment ordered L2
//! score over the learning fold's `BodyTailArr`, summed across segments) as a
//! standalone subsystem, independent of the train loop (05-10 wires it):
//!
//! 1. **Degeneration anchor** — a single full-span segment `[(n, n)]` + identity
//!    permutation makes the segment-summed ordered score reduce to the plain
//!    whole-fold L2 score, so the ordered level picks the SAME split as the plain
//!    `greedy_tensor_search_oblivious` search (falsifiable anchor).
//! 2. **Per-segment scaled L2** — for a hand-derived 3-segment scenario the
//!    per-segment `scaledL2 = l2 * (body_sum_weight / body_finish)` is asserted to
//!    the EXACT value (`scoring.cpp:746-748`).
//! 3. **Strict first-wins** — two equal-score candidates pick the FIRST in
//!    enumeration order (feature asc, border asc; `>` not `>=`, Pitfall 1).
//! 4. **Bounds** — a permutation index out of range returns `CbError::Degenerate`
//!    (T-05-08-01; no panic / OOB).

use cb_compute::{l2_split_score, reduce_leaf_stats, scale_l2_reg, LeafStats};
use cb_core::CbError;

use crate::fold::{body_sum_weights, body_tail_segments};
use crate::tree::{
    greedy_tensor_search_oblivious, greedy_tensor_search_oblivious_ordered, score_candidate_ordered,
    select_level_ordered, FeatureMatrix, Split,
};

/// A small numeric-only fixture: 6 objects, 2 float features, hand-chosen
/// derivatives so the ordered and plain searches are both well-defined.
fn fixture() -> (Vec<Vec<f32>>, Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    // feature 0: 0,1,2,3,4,5 ; feature 1: 5,4,3,2,1,0
    let feature_values = vec![
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        vec![5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
    ];
    // candidate borders (ascending) per feature.
    let feature_borders = vec![vec![1.5, 3.5], vec![1.5, 3.5]];
    let der1 = vec![1.0, -2.0, 3.0, -1.0, 2.0, -3.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    (feature_values, feature_borders, der1, weight)
}

#[test]
fn single_full_span_segment_has_an_empty_tail_and_scores_zero() {
    // REWRITTEN with the ORD-SCORE fix. This test previously asserted that a single
    // full-span segment `[(n, n)]` degenerates to the PLAIN search. That property was
    // an artifact of the bug it was written against: the old implementation merged the
    // body and tail rows into ONE stat pair, which made the ordered score collapse to
    // the plain formula whenever the segment spanned everything.
    //
    // Upstream does NOT do that. `CalcStatsKernel`'s `!isPlainMode` arm
    // (`scoring.cpp:291-309`) fills `SumDelta`/`Count` from `[begin, BodyFinish)` and
    // `SumWeightedDelta`/`SumWeight` from `[BodyFinish, TailFinish)`, the latter
    // GUARDED by `if (tailFinishInRange > bt.BodyFinish)`. At `BodyFinish ==
    // TailFinish == n` that guard is false, no `SumWeightedDelta` is ever accumulated,
    // and `AddLeafOrdered`'s `avg * SumWeightedDelta` is 0 for every leaf.
    //
    // So the correct property is that a full-span segment scores ZERO -- there are no
    // tail rows to evaluate the gain on. Asserting the old equality would re-introduce
    // the bug.
    let (fv, fb, der1, weight) = fixture();
    let n = der1.len();
    let matrix = FeatureMatrix::new(&fv, &fb);
    let identity: Vec<i32> = (0..n as i32).collect();
    let candidate = Split { feature: 0, border: 1.5 };

    let score = score_candidate_ordered(
        &matrix,
        &[],
        candidate,
        &der1,
        &weight,
        &identity,
        &[(n, n)],
        &[n as f64],
        2.0,
        n,
    )
    .expect("ordered score");

    assert!(
        score.abs() < 1e-12,
        "a full-span segment has an EMPTY tail, so the ordered gain must be 0; got {score}"
    );
}

#[test]
fn ordered_score_estimates_on_the_body_and_scores_on_the_tail() {
    // REWRITTEN with the ORD-SCORE fix (see the sibling test for why the old
    // "equals the plain whole-fold score" assertion encoded the bug).
    //
    // The defining property of ordered boosting, asserted directly against a
    // hand-computed reference: for segment `(body_finish, tail_finish)` each leaf's
    // value is estimated on the BODY rows `[0, body_finish)` and multiplied by the
    // TAIL rows' derivative sum `[body_finish, tail_finish)` --
    // `TL2ScoreCalcer::AddLeafOrdered` (`score_calcers.cpp:36-49`). A document
    // therefore never contributes to the leaf value that scores it.
    let (fv, fb, der1, weight) = fixture();
    let n = der1.len();
    assert!(n >= 4, "fixture must have enough rows to split a body from a tail");
    let matrix = FeatureMatrix::new(&fv, &fb);
    let identity: Vec<i32> = (0..n as i32).collect();
    let candidate = Split { feature: 0, border: 1.5 };
    let body_finish = n / 2;
    let l2 = 2.0;
    let bsw: f64 = cb_core::sum_f64(&weight[..body_finish]);

    let got = score_candidate_ordered(
        &matrix, &[], candidate, &der1, &weight, &identity, &[(body_finish, n)], &[bsw], l2, n,
    )
    .expect("ordered score");

    // Hand-computed reference over the SAME two ranges.
    let leaf_of: Vec<usize> = (0..n)
        .map(|obj| usize::from(f64::from(fv[0][obj]) > 1.5))
        .collect();
    let scaled_l2 = scale_l2_reg(l2, bsw, body_finish);
    let mut expected = 0.0_f64;
    for leaf in 0..2usize {
        let body_sum: f64 = (0..body_finish)
            .filter(|&i| leaf_of[i] == leaf)
            .map(|i| der1[i])
            .sum();
        let body_cnt: f64 = (0..body_finish)
            .filter(|&i| leaf_of[i] == leaf)
            .map(|i| weight[i])
            .sum();
        let tail_sum: f64 = (body_finish..n)
            .filter(|&i| leaf_of[i] == leaf)
            .map(|i| der1[i])
            .sum();
        expected += cb_compute::calc_average(body_sum, body_cnt, scaled_l2) * tail_sum;
    }

    assert!(
        (got - expected).abs() < 1e-12,
        "ordered score {got} must equal the body-estimated / tail-scored reference {expected}"
    );

    // Discrimination: the ordered score must NOT coincide with the plain whole-fold
    // score, or this test would pass against the very bug it replaces.
    let stats: Vec<LeafStats> = reduce_leaf_stats(&leaf_of, &der1, &weight, 2);
    let plain = l2_split_score(&stats, scaled_l2);
    assert!(
        (got - plain).abs() > 1e-12,
        "ordered and plain scores coincide ({got} vs {plain}) — the body/tail split is \
         not being applied, which is exactly the bug this test replaces"
    );
}

#[test]
fn multi_segment_per_segment_scaled_l2_is_l2_times_bsw_over_body_finish() {
    // Hand-derived 3-segment scenario: assert each segment's scaledL2 is EXACTLY
    // l2 * (body_sum_weight / body_finish) (scoring.cpp:746-748), the precise
    // arithmetic the ordered score uses per segment — NOT a subjective check.
    //
    // For n = 4, multiplier = 2.0 the body/tail boundaries are [1, 2, 4] →
    // segments [(1, 2), (2, 4)]. To force a THIRD segment we use n = 6,
    // multiplier = 1.5: boundaries = [1, ceil(1.5)=2, ceil(3)=3, ceil(4.5)=5,
    // ceil(7.5)=8 capped 6] = [1, 2, 3, 5, 6] → 4 segments. Take the first three.
    let n = 6usize;
    let mult = 1.5f64;
    let segments = body_tail_segments(n, mult);
    // Weighted: w = [2,2,2,2,2,2] so body_sum_weight = 2 * body_finish.
    let weight = vec![2.0f64; n];
    let bsw = body_sum_weights(n, mult, &weight);
    assert!(segments.len() >= 3, "need at least 3 segments, got {segments:?}");

    let l2 = 3.0;
    // Segment 0: (1, 2), body_finish = 1, body_sum_weight = 2 → scaledL2 = 3 * (2/1) = 6.
    // Segment 1: (2, 3), body_finish = 2, body_sum_weight = 4 → scaledL2 = 3 * (4/2) = 6.
    // Segment 2: (3, 5), body_finish = 3, body_sum_weight = 6 → scaledL2 = 3 * (6/3) = 6.
    for (idx, &(body_finish, _tail)) in segments.iter().take(3).enumerate() {
        let body_sum_weight = bsw[idx];
        let expected = l2 * (body_sum_weight / body_finish as f64);
        let got = scale_l2_reg(l2, body_sum_weight, body_finish);
        assert!(
            (got - expected).abs() < 1e-12,
            "segment {idx} ({body_finish}): scaledL2 got {got} expected {expected}"
        );
    }

    // Also assert the EXACT numeric values for the hand-derived weights above.
    assert!((scale_l2_reg(l2, bsw[0], segments[0].0) - 6.0).abs() < 1e-12);
    assert!((scale_l2_reg(l2, bsw[1], segments[1].0) - 6.0).abs() < 1e-12);
    assert!((scale_l2_reg(l2, bsw[2], segments[2].0) - 6.0).abs() < 1e-12);
}

#[test]
fn multi_segment_ordered_search_runs_and_can_differ_from_plain() {
    // The full public ordered search over the real multi-segment dynamic body/tail
    // produces a valid tree (the structural difference ORD-02 requires; the chosen
    // split MAY differ from the plain whole-fold choice).
    let (fv, fb, der1, weight) = fixture();
    let n = der1.len();
    let matrix = FeatureMatrix::new(&fv, &fb);
    let identity: Vec<i32> = (0..n as i32).collect();

    let ordered = greedy_tensor_search_oblivious_ordered(
        &matrix, &der1, &weight, &identity, 2.0, 2.0, 2, n,
    )
    .expect("ordered search");
    assert_eq!(ordered.splits.len(), 2, "depth-2 tree has 2 splits");
    assert_eq!(ordered.leaf_of.len(), n, "leaf_of is object-order, length n");
    // leaf indices are within 0..2^depth.
    assert!(ordered.leaf_of.iter().all(|&l| l < 4));
}

#[test]
fn strict_first_wins_on_equal_score_pair() {
    // Two candidates with identical ordered scores must resolve to the FIRST in
    // enumeration order (feature asc, border asc), via the strict `>` discipline
    // shared with select_best_candidate. We construct a symmetric fixture where
    // feature 0 and feature 1 produce identical leaf partitions (hence identical
    // scores), and assert the search picks feature 0 (the first).
    //
    // feature 0: 0,0,0,1,1,1 ; feature 1: 0,0,0,1,1,1 (identical columns) →
    // border 0.5 on either gives the SAME 3/3 partition and the SAME score.
    let fv = vec![
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
    ];
    let fb = vec![vec![0.5], vec![0.5]];
    let der1 = vec![1.0, 1.0, 1.0, -1.0, -1.0, -1.0];
    let weight = vec![1.0; 6];
    let n = 6;
    let matrix = FeatureMatrix::new(&fv, &fb);
    let identity: Vec<i32> = (0..n as i32).collect();

    let tree = greedy_tensor_search_oblivious_ordered(
        &matrix, &der1, &weight, &identity, 1.0, 2.0, 1, n,
    )
    .expect("ordered search");
    assert_eq!(
        tree.splits[0].feature, 0,
        "equal-score candidates must pick the FIRST (feature 0) — strict first-wins"
    );
}

#[test]
fn permutation_index_out_of_range_returns_degenerate() {
    // A permutation referencing a document index >= n must surface
    // CbError::Degenerate, never an OOB panic (T-05-08-01).
    let (fv, fb, der1, weight) = fixture();
    let n = der1.len();
    let matrix = FeatureMatrix::new(&fv, &fb);
    // A permutation with an out-of-range index (n, which is >= n).
    let bad_perm: Vec<i32> = vec![0, 1, 2, 3, 4, n as i32];
    let segments = vec![(n, n)];
    let seg_bsw = vec![n as f64];

    let err = score_candidate_ordered(
        &matrix,
        &[],
        Split { feature: 0, border: 1.5 },
        &der1,
        &weight,
        &bad_perm,
        &segments,
        &seg_bsw,
        1.0,
        n,
    );
    assert!(
        matches!(err, Err(CbError::Degenerate(_))),
        "out-of-range permutation index must return Degenerate, got {err:?}"
    );
}

#[test]
fn negative_permutation_index_returns_degenerate() {
    // A negative permutation index is also rejected (no `as usize` wrap to a huge
    // value, no OOB).
    let (fv, fb, der1, weight) = fixture();
    let n = der1.len();
    let matrix = FeatureMatrix::new(&fv, &fb);
    let bad_perm: Vec<i32> = vec![0, 1, 2, 3, 4, -1];
    let segments = vec![(n, n)];
    let seg_bsw = vec![n as f64];

    let err = score_candidate_ordered(
        &matrix,
        &[],
        Split { feature: 0, border: 1.5 },
        &der1,
        &weight,
        &bad_perm,
        &segments,
        &seg_bsw,
        1.0,
        n,
    );
    assert!(matches!(err, Err(CbError::Degenerate(_))));
}
