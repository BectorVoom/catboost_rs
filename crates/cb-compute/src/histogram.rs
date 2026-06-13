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
