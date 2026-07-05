//! Unit tests for the host-side ordered bucket reduction (D-02/D-05) and the
//! CPU split-finding histogram primitives (PERF-01/PERF-02, Phase 21).

use crate::histogram::{
    bin_of, build_bucket_histogram, collect_leaf_residuals, reduce_leaf_der2, reduce_leaf_stats,
    scan_border_to_leaf_stats, BucketHistogram, LeafStats,
};
use crate::runtime::EScoreFunction;
use crate::score::{l2_split_score, multi_dim_split_score};

#[test]
fn reduce_leaf_stats_groups_by_leaf() {
    // 4 objects, 2 leaves. leaf 0 = {obj0, obj2}, leaf 1 = {obj1, obj3}.
    let leaf_of = [0usize, 1, 0, 1];
    let der1 = [1.0, 2.0, 3.0, 4.0];
    let weight = [1.0, 1.0, 1.0, 1.0];
    let stats = reduce_leaf_stats(&leaf_of, &der1, &weight, 2);
    assert_eq!(stats.len(), 2);
    assert_eq!(
        stats[0],
        LeafStats {
            sum_weighted_delta: 4.0, // 1.0 + 3.0
            sum_weight: 2.0,
        }
    );
    assert_eq!(
        stats[1],
        LeafStats {
            sum_weighted_delta: 6.0, // 2.0 + 4.0
            sum_weight: 2.0,
        }
    );
}

#[test]
fn reduce_leaf_stats_empty_leaf_is_zero() {
    // All objects fall in leaf 0; leaf 1 stays empty -> zero stats.
    let leaf_of = [0usize, 0, 0];
    let der1 = [1.0, 2.0, 3.0];
    let weight = [1.0, 1.0, 1.0];
    let stats = reduce_leaf_stats(&leaf_of, &der1, &weight, 2);
    assert_eq!(stats[1], LeafStats::default());
    assert_eq!(stats[0].sum_weighted_delta, 6.0);
    assert_eq!(stats[0].sum_weight, 3.0);
}

#[test]
fn reduce_leaf_stats_ordered_determinism() {
    // The reduction must fold in ascending object order (D-05). Build a leaf
    // whose member order matters under the naive sequential sum: the adversarial
    // [1e16, 1.0, -1e16] sequence sums to 0.0 left-to-right (the 1.0 is lost).
    let leaf_of = [0usize, 0, 0];
    let der1 = [1e16, 1.0, -1e16];
    let weight = [1.0, 1.0, 1.0];
    let stats = reduce_leaf_stats(&leaf_of, &der1, &weight, 1);
    // Object order preserved -> sequential fold loses the 1.0, exactly as
    // cb_core::sum_f64 (the parity contract) does.
    assert_eq!(stats[0].sum_weighted_delta, 0.0);
}

#[test]
fn reduce_leaf_der2_groups_by_leaf() {
    // leaf 0 = {obj0, obj2}, leaf 1 = {obj1, obj3}; weighted der2 per object.
    let leaf_of = [0usize, 1, 0, 1];
    let weighted_der2 = [-1.0, -0.5, -1.0, -0.25];
    let d2 = reduce_leaf_der2(&leaf_of, &weighted_der2, 2);
    assert_eq!(d2.len(), 2);
    assert!((d2[0] - (-2.0)).abs() < 1e-12); // -1 + -1
    assert!((d2[1] - (-0.75)).abs() < 1e-12); // -0.5 + -0.25
}

#[test]
fn reduce_leaf_der2_empty_leaf_is_zero() {
    let leaf_of = [0usize, 0];
    let weighted_der2 = [-1.0, -1.0];
    let d2 = reduce_leaf_der2(&leaf_of, &weighted_der2, 2);
    assert_eq!(d2[1], 0.0);
    assert!((d2[0] - (-2.0)).abs() < 1e-12);
}

#[test]
fn collect_leaf_residuals_groups_members() {
    // leaf 0 = {obj0, obj2}, leaf 1 = {obj1}. residuals widen through f32.
    let leaf_of = [0usize, 1, 0];
    let residuals = [1.5_f64, -2.0, 3.25];
    let weight = [1.0, 1.0, 1.0];
    let members = collect_leaf_residuals(&leaf_of, &residuals, &weight, 2);
    assert_eq!(members.len(), 2);
    assert_eq!(members[0].0, vec![1.5_f32, 3.25_f32]);
    assert_eq!(members[0].1, vec![1.0_f64, 1.0_f64]);
    assert_eq!(members[1].0, vec![-2.0_f32]);
}

// ---------------------------------------------------------------------------
// CPU split-finding histogram primitives (Phase 21, Task 1: build + subtraction).
//
// Test-local benign exactly-representable fixtures so the histogram/bin
// regrouping is bit-exact to the object-order `reduce_leaf_stats` fold; the
// adversarial-ULP tie-flip risk (RESEARCH Pitfall 1) is gated by the downstream
// oracle suite, not these unit primitives.
// ---------------------------------------------------------------------------

#[test]
fn bin_of_matches_strict_greater_split() {
    let borders = [1.0_f64, 3.0, 5.0];
    // count of borders strictly less than value (upper-bound).
    assert_eq!(bin_of(&borders, 0.5), 0); // below min
    assert_eq!(bin_of(&borders, 1.0), 0); // equal to border -> lower bucket (1.0 > 1.0 is false)
    assert_eq!(bin_of(&borders, 2.0), 1);
    assert_eq!(bin_of(&borders, 3.0), 1); // equal -> lower
    assert_eq!(bin_of(&borders, 4.0), 2);
    assert_eq!(bin_of(&borders, 5.0), 2); // equal -> lower
    assert_eq!(bin_of(&borders, 6.0), 3); // above max
    // Consistency with passes_float's strict `>`: object passes border k iff k < bin.
    for &v in &[0.5_f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0] {
        let bin = bin_of(&borders, v);
        for (k, &brd) in borders.iter().enumerate() {
            let passes = f64::from(v) > brd;
            assert_eq!(passes, k < bin, "value {v} border#{k}={brd}");
        }
    }
}

#[test]
fn build_bucket_histogram_sums_match_reduce_leaf_stats() {
    // 6 objects, 1 leaf, 2 features; benign integer der/weight so bin-grouped
    // reassociation is bit-exact to the object-order fold.
    let n = 6;
    let leaf_of = vec![0usize; n];
    let der1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let weight = vec![1.0, 1.0, 2.0, 1.0, 2.0, 1.0];
    let n_features = 2;
    let n_bins = 4;
    // Feature-major bins: feature 0 in [0..3], feature 1 in [0..2].
    let bins: Vec<u32> = vec![
        // feature 0
        0, 1, 2, 3, 1, 0, // feature 1
        0, 0, 1, 1, 2, 2,
    ];
    let hist = build_bucket_histogram(&bins, &der1, &weight, &leaf_of, 1, n_features, n_bins, 1);
    assert_eq!(hist.n_leaves(), 1);
    assert_eq!(hist.n_features(), 2);
    assert_eq!(hist.n_bins(), 4);
    assert_eq!(hist.n_channels(), 2); // approx_dimension(1) + weight

    let reduced = reduce_leaf_stats(&leaf_of, &der1, &weight, 1);
    // For each feature, the per-leaf sum across all bins reproduces reduce_leaf_stats.
    for feature in 0..n_features {
        let d: f64 = (0..n_bins).map(|b| hist.channel(0, feature, b, 0)).sum();
        let w: f64 = (0..n_bins).map(|b| hist.channel(0, feature, b, 1)).sum();
        assert_eq!(d, reduced[0].sum_weighted_delta, "feature {feature} delta");
        assert_eq!(w, reduced[0].sum_weight, "feature {feature} weight");
    }
}

/// Build a histogram over an arbitrary object subset (object order preserved).
fn build_subset_hist(
    subset: &[usize],
    bins_all: &[u32],
    der1: &[f64],
    weight: &[f64],
    n_all: usize,
    n_features: usize,
    n_bins: usize,
) -> BucketHistogram {
    let m = subset.len();
    let mut bins = vec![0u32; n_features * m];
    let mut der = vec![0.0_f64; m];
    let mut w = vec![0.0_f64; m];
    let leaf = vec![0usize; m];
    for (j, &o) in subset.iter().enumerate() {
        der[j] = der1[o];
        w[j] = weight[o];
        for f in 0..n_features {
            bins[f * m + j] = bins_all[f * n_all + o];
        }
    }
    build_bucket_histogram(&bins, &der, &w, &leaf, 1, n_features, n_bins, 1)
}

#[test]
fn bucket_histogram_remove_equals_fresh_sibling() {
    // Subtraction trick (Pitfall 2): parent.remove(childA) == fresh-built sibling B.
    let n = 6;
    let der1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let n_features = 1;
    let n_bins = 4;
    let bins_all: Vec<u32> = vec![0, 1, 2, 3, 1, 0];
    let a = [0usize, 2, 4];
    let b = [1usize, 3, 5];
    let all: Vec<usize> = (0..n).collect();

    let parent = build_subset_hist(&all, &bins_all, &der1, &weight, n, n_features, n_bins);
    let child = build_subset_hist(&a, &bins_all, &der1, &weight, n, n_features, n_bins);
    let sibling_fresh = build_subset_hist(&b, &bins_all, &der1, &weight, n, n_features, n_bins);

    let sibling = parent.remove(&child);
    assert_eq!(sibling, sibling_fresh);
}

/// Local transcription of the forward-bit `leaf_index` (tree.rs:284): split `i`
/// occupies bit `i` (so the appended candidate takes the highest bit) — the
/// reference the histogram prefix scan must reproduce WITHOUT depending on
/// cb-train (cb-train depends on cb-compute; importing it would be circular).
fn leaf_index_ref(passes: &[bool]) -> usize {
    let mut idx = 0usize;
    for (i, &p) in passes.iter().enumerate() {
        if p {
            idx |= 1usize << i;
        }
    }
    idx
}

/// Feature-major bin matrix for `feature_values`/`feature_borders` (bin_of per cell).
fn bin_matrix(feature_values: &[Vec<f32>], feature_borders: &[Vec<f64>], n: usize) -> Vec<u32> {
    let n_features = feature_values.len();
    let mut bins = vec![0u32; n_features * n];
    for f in 0..n_features {
        for obj in 0..n {
            bins[f * n + obj] = bin_of(&feature_borders[f], feature_values[f][obj]) as u32;
        }
    }
    bins
}

#[test]
fn scan_border_matches_rescan_scalar() {
    // 8 objects, feature 0 = chosen split, feature 1 = candidate borders.
    let n = 8;
    let der1 = vec![1.0, -2.0, 3.0, 0.5, -1.5, 2.0, 4.0, -0.5];
    let weight = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let borders0 = vec![2.0_f64];
    let borders1 = vec![1.0_f64, 3.0];
    let f0 = vec![1.0_f32, 3.0, 2.5, 0.0, 4.0, 1.5, 3.5, 2.0];
    let f1 = vec![0.5_f32, 2.0, 4.0, 1.0, 3.5, 0.0, 2.5, 5.0];
    let feature_values = vec![f0.clone(), f1.clone()];
    let feature_borders = vec![borders0.clone(), borders1.clone()];
    let n_features = 2;
    let n_bins = borders0.len().max(borders1.len()) + 1; // uniform per-feature bin count

    // Parent partition from the chosen split on feature 0.
    let leaf_of: Vec<usize> = (0..n)
        .map(|o| usize::from(f64::from(f0[o]) > borders0[0]))
        .collect();
    let bins = bin_matrix(&feature_values, &feature_borders, n);
    let hist = build_bucket_histogram(&bins, &der1, &weight, &leaf_of, 2, n_features, n_bins, 1);

    let scaled_l2 = 1.0;
    for (b, &brd) in borders1.iter().enumerate() {
        // Histogram path.
        let per_dim = scan_border_to_leaf_stats(&hist, 1, b, 1);
        let hist_leaves = &per_dim[0];
        // Rescan reference: chosen(feature0) ++ candidate(feature1, border b).
        let leaf_of_cand: Vec<usize> = (0..n)
            .map(|o| {
                let p0 = f64::from(f0[o]) > borders0[0];
                let p1 = f64::from(f1[o]) > brd;
                leaf_index_ref(&[p0, p1])
            })
            .collect();
        let ref_leaves = reduce_leaf_stats(&leaf_of_cand, &der1, &weight, 4);
        assert_eq!(hist_leaves, &ref_leaves, "LeafStats border {b}");
        // Candidate score bit-exact.
        let hist_score = l2_split_score(hist_leaves, scaled_l2);
        let ref_score = l2_split_score(&ref_leaves, scaled_l2);
        assert_eq!(hist_score, ref_score, "score border {b}");
    }
}

#[test]
fn scan_border_matches_rescan_multiclass() {
    // approx_dimension = 2; der1 dimension-major (der[d*n + obj]).
    let n = 8;
    let der_d0 = [1.0, -2.0, 3.0, 0.5, -1.5, 2.0, 4.0, -0.5];
    let der_d1 = [-0.5, 1.5, -1.0, 2.0, 0.5, -3.0, 1.0, 2.5];
    let mut der1 = Vec::with_capacity(2 * n);
    der1.extend_from_slice(&der_d0);
    der1.extend_from_slice(&der_d1);
    let weight = vec![1.0_f64; n];
    let borders0 = vec![2.0_f64];
    let borders1 = vec![1.0_f64, 3.0];
    let f0 = vec![1.0_f32, 3.0, 2.5, 0.0, 4.0, 1.5, 3.5, 2.0];
    let f1 = vec![0.5_f32, 2.0, 4.0, 1.0, 3.5, 0.0, 2.5, 5.0];
    let feature_values = vec![f0.clone(), f1.clone()];
    let feature_borders = vec![borders0.clone(), borders1.clone()];
    let n_features = 2;
    let n_bins = borders0.len().max(borders1.len()) + 1;

    let leaf_of: Vec<usize> = (0..n)
        .map(|o| usize::from(f64::from(f0[o]) > borders0[0]))
        .collect();
    let bins = bin_matrix(&feature_values, &feature_borders, n);
    let hist = build_bucket_histogram(&bins, &der1, &weight, &leaf_of, 2, n_features, n_bins, 2);

    let scaled_l2 = 1.0;
    for (b, &brd) in borders1.iter().enumerate() {
        let per_dim = scan_border_to_leaf_stats(&hist, 1, b, 2);
        // Rescan reference per dimension.
        let leaf_of_cand: Vec<usize> = (0..n)
            .map(|o| {
                let p0 = f64::from(f0[o]) > borders0[0];
                let p1 = f64::from(f1[o]) > brd;
                leaf_index_ref(&[p0, p1])
            })
            .collect();
        let ref_dim0 = reduce_leaf_stats(&leaf_of_cand, &der_d0, &weight, 4);
        let ref_dim1 = reduce_leaf_stats(&leaf_of_cand, &der_d1, &weight, 4);
        assert_eq!(&per_dim[0], &ref_dim0, "dim0 border {b}");
        assert_eq!(&per_dim[1], &ref_dim1, "dim1 border {b}");
        // Cross-dimension Cosine score bit-exact through the UNCHANGED score math.
        let ref_per_dim = vec![ref_dim0, ref_dim1];
        let hist_score = multi_dim_split_score(EScoreFunction::Cosine, &per_dim, scaled_l2);
        let ref_score = multi_dim_split_score(EScoreFunction::Cosine, &ref_per_dim, scaled_l2);
        assert_eq!(hist_score, ref_score, "multiclass score border {b}");
    }
}
