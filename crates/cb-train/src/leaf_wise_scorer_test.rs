//! `best_split_for_leaf` histogram-scorer equivalence (Plan 21-03, PERF-02). The
//! leaf-wise scoring core was rewritten to score each candidate from a per-leaf
//! [`cb_compute::BucketHistogram`] + `O(n_bins)` prefix scan instead of a
//! per-candidate `reduce_leaf_stats` full-subset rescan. This locks that the
//! rewritten core returns BYTE-FOR-BYTE the pre-rewrite result (chosen split,
//! bit-exact gain, and the left/right document partition) on benign fixtures —
//! the direct regression gate the Depthwise / Lossguide / Region growers inherit
//! through this shared core.
//!
//! Sibling `#[path]` mount (source/test separation, CLAUDE.md) of `tree.rs`, so it
//! can exercise the private `best_split_for_leaf` / `split_score` / `LeafBestSplit`.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp
)]

use cb_compute::{reduce_leaf_stats, EScoreFunction, MINIMAL_SCORE};

use super::{best_split_for_leaf, split_score, FeatureMatrix, Split};

/// The PRE-REWRITE algorithm, transcribed verbatim: for each (feature ascending,
/// border ascending) candidate, build the 2-leaf `{value <= border, value > border}`
/// partition restricted to `docs`, reduce it with `reduce_leaf_stats`, score it with
/// the configured calcer, keep the strict `>` first-wins best, subtract the unsplit
/// baseline, and gate on `gain >= 1e-9`. Returns the chosen split, its gain, and the
/// (left = FALSE, right = TRUE) document partition — exactly what the rewritten
/// histogram-backed core must reproduce bit-for-bit.
fn reference_best_split(
    matrix: &FeatureMatrix,
    docs: &[usize],
    der1: &[f64],
    weight: &[f64],
    scaled_l2: f64,
    min_data_in_leaf: usize,
    score_function: EScoreFunction,
) -> Option<(Split, f64, Vec<usize>, Vec<usize>)> {
    if docs.len() < min_data_in_leaf || docs.len() < 2 {
        return None;
    }
    let der1_sub: Vec<f64> = docs.iter().map(|&i| der1[i]).collect();
    let weight_sub: Vec<f64> = docs.iter().map(|&i| weight[i]).collect();

    let base_leaf_of = vec![0usize; docs.len()];
    let base_stats = reduce_leaf_stats(&base_leaf_of, &der1_sub, &weight_sub, 1);
    let baseline = split_score(score_function, &base_stats, scaled_l2);

    let mut best: Option<(usize, f64, Vec<bool>)> = None;
    let mut best_score = MINIMAL_SCORE;
    for feature in 0..matrix.n_features() {
        for &border in &matrix.feature_borders[feature] {
            let passes: Vec<bool> = docs
                .iter()
                .map(|&obj| f64::from(matrix.feature_values[feature][obj]) > border)
                .collect();
            let leaf_of: Vec<usize> = passes.iter().map(|&p| usize::from(p)).collect();
            let stats = reduce_leaf_stats(&leaf_of, &der1_sub, &weight_sub, 2);
            let score = split_score(score_function, &stats, scaled_l2);
            if score > best_score {
                best_score = score;
                best = Some((feature, border, passes));
            }
        }
    }

    let (feature, border, passes) = best?;
    let gain = best_score - baseline;
    if gain < 1e-9 {
        return None;
    }
    let mut left = Vec::new();
    let mut right = Vec::new();
    for (&obj, &p) in docs.iter().zip(passes.iter()) {
        if p {
            right.push(obj);
        } else {
            left.push(obj);
        }
    }
    Some((Split { feature, border }, gain, left, right))
}

/// A benign fixture: integer-valued der1 / weight so every `sum_f64` fold is EXACT,
/// making the histogram scorer's nested bin-order sum bit-identical to the rescan's
/// flat object-order sum (the ULP tie-flip of Pitfall 1 cannot occur here — that
/// adversarial case is gated by the full oracle suite).
fn fixture() -> (Vec<Vec<f32>>, Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let feature_values = vec![
        vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], // f0
        vec![7.0_f32, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0], // f1
    ];
    let feature_borders = vec![vec![1.5_f64, 3.5, 5.5], vec![2.5_f64, 4.5]];
    let der1 = vec![-3.0_f64, -2.0, -1.0, 1.0, 2.0, -1.0, 3.0, -2.0];
    let weight = vec![1.0_f64; 8];
    (feature_values, feature_borders, der1, weight)
}

fn assert_matches_reference(docs: &[usize], score_function: EScoreFunction) {
    let (fv, fb, der1, weight) = fixture();
    let matrix = FeatureMatrix::new(&fv, &fb);
    let got = best_split_for_leaf(&matrix, docs, &der1, &weight, 0.0, 1, score_function);
    let want = reference_best_split(&matrix, docs, &der1, &weight, 0.0, 1, score_function);
    match (got, want) {
        (Some(g), Some((split, gain, left, right))) => {
            assert_eq!(g.split, split, "chosen split diverged from the rescan");
            assert_eq!(g.gain, gain, "gain must be BIT-EXACT to the rescan");
            assert_eq!(g.left_docs, left, "left (FALSE) partition diverged");
            assert_eq!(g.right_docs, right, "right (TRUE) partition diverged");
        }
        (None, None) => {}
        (g, w) => panic!(
            "split-existence mismatch: histogram={:?} vs rescan={:?}",
            g.is_some(),
            w.is_some()
        ),
    }
}

/// Cosine (the default CPU calcer) on a document SUBSET: the histogram-backed core
/// reproduces the rescan's split, gain, and partition bit-for-bit.
#[test]
fn best_split_for_leaf_matches_rescan_cosine_subset() {
    assert_matches_reference(&[0, 2, 3, 4, 6, 7], EScoreFunction::Cosine);
}

/// L2 on the full document set: same bit-for-bit equivalence under the second
/// shipped CPU calcer.
#[test]
fn best_split_for_leaf_matches_rescan_l2_full() {
    assert_matches_reference(&[0, 1, 2, 3, 4, 5, 6, 7], EScoreFunction::L2);
}

/// A degenerate leaf (uniform gradient → every candidate gain below the `1e-9`
/// cutoff) yields `None` from BOTH the histogram core and the rescan.
#[test]
fn best_split_for_leaf_degenerate_leaf_is_none_like_rescan() {
    let fv = vec![vec![0.0_f32, 1.0, 2.0, 3.0]];
    let fb = vec![vec![0.5_f64, 1.5, 2.5]];
    let der1 = vec![1.0_f64; 4]; // uniform ⇒ no beneficial split
    let weight = vec![1.0_f64; 4];
    let matrix = FeatureMatrix::new(&fv, &fb);
    let docs = [0usize, 1, 2, 3];
    let got = best_split_for_leaf(&matrix, &docs, &der1, &weight, 0.0, 1, EScoreFunction::Cosine);
    let want = reference_best_split(
        &matrix,
        &docs,
        &der1,
        &weight,
        0.0,
        1,
        EScoreFunction::Cosine,
    );
    assert!(got.is_none(), "degenerate leaf must not split");
    assert!(want.is_none(), "reference agrees the degenerate leaf must not split");
}
