//! Oblivious (symmetric) tree growth — `GreedyTensorSearchOblivious` and the
//! strict first-wins split tie-break (TRAIN-02).
//!
//! # Source of truth
//!
//! `catboost/private/libs/algo/greedy_tensor_search.cpp`:
//! - `:1189-1259` — for `curDepth in 0..MaxDepth` exactly ONE split is selected
//!   per level via `SelectBestCandidate` and applied across the whole level; a
//!   depth-`d` tree has `d` splits and `2^d` leaves.
//! - `:948-966` — `SelectBestCandidate` uses strict `if (gain > bestGain)` over a
//!   FIXED candidate-iteration order (feature index ascending, border ascending
//!   within feature). The FIRST candidate reaching the max wins; later
//!   equal-gain candidates do NOT replace it (Pitfall 1). Do NOT sort by score;
//!   do NOT use `>=`.
//!
//! The split score is the L2 `AddLeafPlain` fold over the candidate's leaves
//! (`cb_compute::l2_split_score`), computed over the leaf statistics reduced
//! host-side via `cb_core::sum_f64` (D-02/D-05) by `cb_compute::reduce_leaf_stats`.
//!
//! # Leaf indexing
//!
//! An object's leaf index is the `d`-bit number formed by its split outcomes,
//! split `i` contributing bit `i` (forward bit order, verified against the
//! upstream `model.json` leaf ordering): `idx |= (passes_split_i << i)`.
//!
//! # Fallibility
//!
//! Depth is capped at [`MAX_DEPTH`] (upstream `MaxDepth`); a larger depth is a
//! [`CbError::DepthExceeded`] BEFORE any `2^depth` allocation (T-03-01-02). No
//! `unwrap`/`expect`/raw float fold (deny-lints + D-08).

use cb_compute::{l2_split_score, reduce_leaf_stats, LeafStats, MINIMAL_SCORE};
use cb_core::{CbError, CbResult};

// Tests live in dedicated sibling files (source/test separation, CLAUDE.md /
// AGENTS.md — no test body in this production file). Mounted as CHILD modules of
// `tree` so the canonical filters select them: `cargo test -p cb-train tree::`
// selects all tree tests; `cargo test -p cb-train tree::tie_break` selects the
// Pitfall-1 tie-break tests (mounted at `tree::tie_break`).
#[cfg(test)]
#[path = "tree_test.rs"]
mod general;

#[cfg(test)]
#[path = "tree_tie_break_test.rs"]
mod tie_break;

/// Maximum supported tree depth (upstream `MaxDepth`). Capping `depth <= 16`
/// keeps `2^depth` within `usize` and bounds leaf-buffer allocation.
pub const MAX_DEPTH: usize = 16;

/// One split in an oblivious tree: a `value > border` test on one float feature.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Split {
    /// The float feature this split tests.
    pub feature: usize,
    /// The split border (threshold); an object passes when `value > border`.
    pub border: f64,
}

/// A scored candidate split during the greedy search. `score` is the L2 split
/// score (`l2_split_score`) of applying this split across the current level.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candidate {
    /// Candidate float feature index.
    pub feature: usize,
    /// Candidate split border.
    pub border: f64,
    /// L2 split score of this candidate.
    pub score: f64,
}

/// The grown oblivious tree's structure: the ordered splits and the per-object
/// leaf assignment (`leaf_of[i]` in `0..2^depth`). Leaf VALUES are computed by
/// the boosting loop (`cb-train::boosting`) from these assignments.
#[derive(Debug, Clone, PartialEq)]
pub struct GrownTree {
    /// The `depth` ordered splits (one per level).
    pub splits: Vec<Split>,
    /// Per-object leaf index (`0..2^depth`), object order.
    pub leaf_of: Vec<usize>,
}

/// Reject a depth that exceeds [`MAX_DEPTH`] before any `2^depth` allocation.
///
/// # Errors
/// [`CbError::DepthExceeded`] if `depth > MAX_DEPTH`.
pub fn check_depth(depth: usize) -> CbResult<()> {
    if depth > MAX_DEPTH {
        Err(CbError::DepthExceeded {
            depth,
            max: MAX_DEPTH,
        })
    } else {
        Ok(())
    }
}

/// The leaf index for an object given its per-split outcomes (`passes[i]` is
/// whether the object passes split `i`): forward bit order, split `i` -> bit `i`.
#[must_use]
pub fn leaf_index(passes: &[bool]) -> usize {
    let mut idx = 0usize;
    for (i, &p) in passes.iter().enumerate() {
        if p {
            idx |= 1usize << i;
        }
    }
    idx
}

/// Select the best candidate split with the strict first-wins tie-break
/// (`greedy_tensor_search.cpp:948-966`): iterate `candidates` in the given order
/// (the caller supplies upstream order — feature ascending, border ascending) and
/// keep the FIRST candidate whose score strictly exceeds the running best
/// (`score > best`). Returns `None` for an empty candidate list.
///
/// Strict `>` is load-bearing: a `>=` would pick the LATER equal-gain candidate
/// and diverge from upstream (Pitfall 1).
#[must_use]
pub fn select_best_candidate(candidates: &[Candidate]) -> Option<Candidate> {
    let mut best: Option<Candidate> = None;
    let mut best_score = MINIMAL_SCORE;
    for &candidate in candidates {
        // STRICT `>` (NOT `>=`): first-wins on equal gain.
        if candidate.score > best_score {
            best_score = candidate.score;
            best = Some(candidate);
        }
    }
    best
}

/// Per-object access to one feature's `f32` value, for the `value > border` test.
/// `values[feature][object]` SoA layout (object order preserved for D-05).
pub struct FeatureMatrix<'a> {
    /// `feature_values[f]` is feature `f`'s per-object `f32` column.
    pub feature_values: &'a [Vec<f32>],
    /// `feature_borders[f]` is the ascending candidate borders for feature `f`
    /// (the model's float-feature borders).
    pub feature_borders: &'a [Vec<f64>],
}

impl FeatureMatrix<'_> {
    /// Number of float features.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.feature_values.len()
    }

    /// Whether object `obj` passes the split `value > border` on `feature`.
    /// Out-of-range indices return `false` defensively (the trainer passes valid
    /// indices).
    #[must_use]
    fn passes(&self, feature: usize, obj: usize, border: f64) -> bool {
        self.feature_values
            .get(feature)
            .and_then(|col| col.get(obj))
            .is_some_and(|&v| f64::from(v) > border)
    }
}

/// Assign every object to a leaf given the chosen `splits` (forward bit order).
fn assign_leaves(matrix: &FeatureMatrix, splits: &[Split], n_objects: usize) -> Vec<usize> {
    (0..n_objects)
        .map(|obj| {
            let passes: Vec<bool> = splits
                .iter()
                .map(|s| matrix.passes(s.feature, obj, s.border))
                .collect();
            leaf_index(&passes)
        })
        .collect()
}

/// Score one candidate split applied across the CURRENT level: extend the
/// already-chosen `splits` with the candidate, assign leaves, reduce per-leaf
/// stats (ordered, via `cb_compute::reduce_leaf_stats`), and fold the L2 score.
fn score_candidate(
    matrix: &FeatureMatrix,
    chosen: &[Split],
    candidate: Split,
    der1: &[f64],
    weight: &[f64],
    scaled_l2: f64,
    n_objects: usize,
) -> f64 {
    let mut splits = chosen.to_vec();
    splits.push(candidate);
    let n_leaves = 1usize << splits.len();
    let leaf_of = assign_leaves(matrix, &splits, n_objects);
    let stats: Vec<LeafStats> = reduce_leaf_stats(&leaf_of, der1, weight, n_leaves);
    l2_split_score(&stats, scaled_l2)
}

/// Grow one oblivious tree of depth `depth` with the strict first-wins greedy
/// search (`GreedyTensorSearchOblivious`).
///
/// For each level `0..depth`, enumerate candidate splits in upstream order
/// (feature index ascending, then border ascending within the feature), score
/// each via the L2 calcer, and select the best with [`select_best_candidate`]
/// (strict `>`). The one chosen split is applied across the whole level. Returns
/// the `depth` splits and the final per-object leaf assignment.
///
/// # Errors
/// - [`CbError::DepthExceeded`] if `depth > MAX_DEPTH` (before allocation).
/// - [`CbError::Degenerate`] if a level has no candidate split at all (no
///   feature has any border), so no tree can be grown.
pub fn greedy_tensor_search_oblivious(
    matrix: &FeatureMatrix,
    der1: &[f64],
    weight: &[f64],
    scaled_l2: f64,
    depth: usize,
    n_objects: usize,
) -> CbResult<GrownTree> {
    check_depth(depth)?;

    let mut chosen: Vec<Split> = Vec::with_capacity(depth);

    for _level in 0..depth {
        // Enumerate candidates in upstream order: feature ascending, border
        // ascending within feature (the borders are stored ascending).
        let mut candidates: Vec<Candidate> = Vec::new();
        for feature in 0..matrix.n_features() {
            let borders = matrix.feature_borders.get(feature).map_or(&[][..], Vec::as_slice);
            for &border in borders {
                let score = score_candidate(
                    matrix,
                    &chosen,
                    Split { feature, border },
                    der1,
                    weight,
                    scaled_l2,
                    n_objects,
                );
                candidates.push(Candidate {
                    feature,
                    border,
                    score,
                });
            }
        }

        let best = select_best_candidate(&candidates).ok_or_else(|| {
            CbError::Degenerate(
                "no candidate split available (no feature has any border)".to_owned(),
            )
        })?;
        chosen.push(Split {
            feature: best.feature,
            border: best.border,
        });
    }

    let leaf_of = assign_leaves(matrix, &chosen, n_objects);
    Ok(GrownTree {
        splits: chosen,
        leaf_of,
    })
}
