//! Host-side ordered bucket reduction — the parity-critical step that folds the
//! backend's per-object scatter contributions into per-bin / per-leaf totals
//! through `cb_core::sum_f64` in canonical object order (D-02/D-05). The
//! `cb-backend` kernel does ONLY the order-independent per-object work; THIS is
//! where the order-sensitive SUM happens, so the 1e-5 oracle bar stays
//! deterministic.
//!
//! # Source of truth
//!
//! `catboost/private/libs/algo/score_calcers.cpp` / `online_predictor.h` —
//! `TBucketStats { SumWeightedDelta, SumWeight }`. Each leaf/bucket accumulates
//! the per-object first-derivative ("weighted delta") and weight; the L2 score
//! calcer (`score.rs`) and the Gradient leaf delta (`leaf.rs`) consume these
//! reduced totals.
//!
//! # Summation routing (D-07 / D-08)
//!
//! Every bin total is produced by [`cb_core::sum_f64`] over the per-object
//! contributions GATHERED in canonical object order. No raw iterator-sum or
//! zero-seeded float fold is spelled here (D-08); the gather builds an ordered
//! `Vec` and hands it to the single sanctioned primitive.

use cb_core::sum_f64;

/// A single leaf/bucket's reduced statistics (`TBucketStats` analogue): the
/// summed first-derivative ("weighted delta") and the summed weight of its member
/// objects.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LeafStats {
    /// Σ der1[i] over the leaf's member objects (the "weighted delta"). In the
    /// unweighted path the per-object weight is folded into `der1` already, so
    /// this is the plain derivative sum.
    pub sum_weighted_delta: f64,
    /// Σ weight[i] over the leaf's member objects (the leaf's object count in the
    /// unweighted path).
    pub sum_weight: f64,
}

/// Reduce per-object contributions into one bucket per leaf index.
///
/// `leaf_of[i]` is object `i`'s leaf index (`0..n_leaves`); `der1[i]` its
/// first-derivative contribution; `weight[i]` its weight. For each leaf the
/// member objects are gathered in ascending object order and summed via
/// [`cb_core::sum_f64`] (D-05 canonical order), producing a [`LeafStats`] per
/// leaf. `der1`, `weight`, and `leaf_of` MUST be the same length (`n` objects);
/// any object whose leaf index is `>= n_leaves` is ignored defensively rather
/// than panicking (the trainer guarantees valid indices).
#[must_use]
pub fn reduce_leaf_stats(
    leaf_of: &[usize],
    der1: &[f64],
    weight: &[f64],
    n_leaves: usize,
) -> Vec<LeafStats> {
    // Gather each leaf's per-object contributions in ascending object order, then
    // fold each gathered Vec through the single sanctioned reduction primitive so
    // the SUM order is exactly upstream's thread_count==1 object order (D-05).
    let mut delta_members: Vec<Vec<f64>> = vec![Vec::new(); n_leaves];
    let mut weight_members: Vec<Vec<f64>> = vec![Vec::new(); n_leaves];

    for (i, &leaf) in leaf_of.iter().enumerate() {
        if leaf >= n_leaves {
            continue;
        }
        // der1/weight are parallel to leaf_of; a missing entry is treated as 0.0
        // (defensive — the trainer passes equal-length slices).
        let d = der1.get(i).copied().unwrap_or(0.0);
        let w = weight.get(i).copied().unwrap_or(0.0);
        if let Some(slot) = delta_members.get_mut(leaf) {
            slot.push(d);
        }
        if let Some(slot) = weight_members.get_mut(leaf) {
            slot.push(w);
        }
    }

    (0..n_leaves)
        .map(|leaf| {
            let deltas = delta_members.get(leaf).map_or(&[][..], Vec::as_slice);
            let weights = weight_members.get(leaf).map_or(&[][..], Vec::as_slice);
            LeafStats {
                sum_weighted_delta: sum_f64(deltas),
                sum_weight: sum_f64(weights),
            }
        })
        .collect()
}

/// Reduce per-object weighted second derivatives into one `Σ der2*weight` per
/// leaf index, through `cb_core::sum_f64` in canonical object order (D-05).
///
/// This is the Newton-method companion to [`reduce_leaf_stats`]: the boosting
/// loop needs the leaf's summed second derivative for `newton_leaf_delta`'s
/// `-sum_der2 + scaledL2` denominator. `weighted_der2[i]` is object `i`'s
/// `der2 * weight` (the elementwise product the host computes); `leaf_of[i]` its
/// leaf index. Objects whose leaf index is `>= n_leaves` are ignored defensively
/// (the trainer guarantees valid indices). Kept separate from [`LeafStats`] so the
/// score path (which never reads `der2`) is untouched.
#[must_use]
pub fn reduce_leaf_der2(
    leaf_of: &[usize],
    weighted_der2: &[f64],
    n_leaves: usize,
) -> Vec<f64> {
    let mut members: Vec<Vec<f64>> = vec![Vec::new(); n_leaves];
    for (i, &leaf) in leaf_of.iter().enumerate() {
        if leaf >= n_leaves {
            continue;
        }
        let d = weighted_der2.get(i).copied().unwrap_or(0.0);
        if let Some(slot) = members.get_mut(leaf) {
            slot.push(d);
        }
    }
    (0..n_leaves)
        .map(|leaf| {
            let ds = members.get(leaf).map_or(&[][..], Vec::as_slice);
            sum_f64(ds)
        })
        .collect()
}

/// Reduce per-object contributions into per-leaf [`LeafStats`] for a **second-order
/// score function** (`NewtonL2` / `NewtonCosine`, `IsSecondOrderScoreFunction`,
/// `enum_helpers.cpp:830-847`): the `sum_weighted_delta` slot carries the summed
/// gradient exactly as [`reduce_leaf_stats`], but the `sum_weight` slot carries the
/// summed **positive hessian** `Σ(-der2*weight)` instead of the weight count.
///
/// # Newton fill mechanism (D-6.4-03, RESEARCH Strand 1 "How der2 threads in")
///
/// The Newton split-score variants reuse the L2/Cosine score FORMULA verbatim
/// (`pointwise_scores.cu:504-521`); only the histogram FILL differs — the calcer's
/// `weight` slot is the summed hessian rather than the object/weight count. RMSE
/// `der2 = -1` and Logloss `der2 = -p(1-p)` are both ≤0, so the per-object
/// `der2 * weight` is negated to a POSITIVE hessian (Pitfall 2) before summation.
/// `weighted_der2[i]` is object `i`'s `der2 * weight`; it is negated here so the
/// caller passes the raw (≤0) weighted second derivative — the SAME column
/// [`reduce_leaf_der2`] consumes for the Newton leaf delta.
///
/// `LeafStats` is UNCHANGED (this is a different FILL, not a different struct); the
/// first-order functions (`Cosine`/`L2`/`SolarL2`/`LOOL2`/`SatL2`) keep calling
/// [`reduce_leaf_stats`] (the `Σ weight` path) so the shipped 05-19 Task A
/// L2-vs-Cosine split lock is byte-identical (no-regression, Pitfall 3). Both folds
/// route through [`cb_core::sum_f64`] in canonical object order (D-05/D-08).
#[must_use]
pub fn reduce_leaf_stats_newton(
    leaf_of: &[usize],
    der1: &[f64],
    weighted_der2: &[f64],
    n_leaves: usize,
) -> Vec<LeafStats> {
    // Reuse the existing der1 fold (via `reduce_leaf_der2`, a plain Σ over each
    // leaf's member column through `sum_f64`) for `sum_weighted_delta` (gradient
    // sum), and the same fold over the weighted der2 column for the hessian, NEGATED
    // to a positive hessian per leaf.
    let sum_delta = reduce_leaf_der2(leaf_of, der1, n_leaves);
    let sum_der2 = reduce_leaf_der2(leaf_of, weighted_der2, n_leaves);
    (0..n_leaves)
        .map(|leaf| {
            let sum_weighted_delta = sum_delta.get(leaf).copied().unwrap_or(0.0);
            // der2 is ≤0 for RMSE/Logloss; negate to the positive hessian the Newton
            // denominator wants (Pitfall 2). `reduce_leaf_der2` already summed via
            // sum_f64, so negating the total is order-faithful.
            let positive_hessian = -sum_der2.get(leaf).copied().unwrap_or(0.0);
            LeafStats {
                sum_weighted_delta,
                sum_weight: positive_hessian,
            }
        })
        .collect()
}

/// Gather each leaf's per-member residuals (as `f32`, matching upstream's
/// `TVector<float> leafSamples`) and weights, in ascending object order — the
/// input the Exact-method quantile (`exact_leaf_delta`) consumes per leaf.
///
/// `leaf_of[i]` is object `i`'s leaf index; `residuals[i] = target_i - approx_i`
/// (the host computes the residual in `f64`, this widens it through `f32` to match
/// upstream's float sample buffer); `weight[i]` its object weight. Returns, per
/// leaf, the `(residuals, weights)` member vectors. Objects whose leaf index is
/// `>= n_leaves` are ignored defensively. No float SUM of derivatives is spelled
/// here (the only later accumulation is the quantile's weight scan inside
/// [`crate::leaf::exact_leaf_delta`]).
#[must_use]
pub fn collect_leaf_residuals(
    leaf_of: &[usize],
    residuals: &[f64],
    weight: &[f64],
    n_leaves: usize,
) -> Vec<(Vec<f32>, Vec<f64>)> {
    let mut out: Vec<(Vec<f32>, Vec<f64>)> = vec![(Vec::new(), Vec::new()); n_leaves];
    for (i, &leaf) in leaf_of.iter().enumerate() {
        if leaf >= n_leaves {
            continue;
        }
        let r = residuals.get(i).copied().unwrap_or(0.0) as f32;
        let w = weight.get(i).copied().unwrap_or(1.0);
        if let Some(slot) = out.get_mut(leaf) {
            slot.0.push(r);
            slot.1.push(w);
        }
    }
    out
}
