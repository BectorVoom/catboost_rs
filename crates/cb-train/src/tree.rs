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

use cb_compute::{l2_split_score, random_score_instance, reduce_leaf_stats, LeafStats, MINIMAL_SCORE};
use cb_core::{CbError, CbResult, TFastRng64};

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
    // The unperturbed path is `random_strength == 0` with no RNG draws — exactly
    // the first-slice behaviour. Delegate to the perturbed search with `None`.
    greedy_tensor_search_oblivious_perturbed(
        matrix, der1, weight, scaled_l2, depth, n_objects, None,
    )
}

/// The `random_strength` split-score perturbation state threaded through the
/// greedy search (`TRandomScore` / `SetBestScore` / `SelectBestCandidate`,
/// TRAIN-05). When supplied, every candidate score is perturbed by a normal draw
/// in the EXACT upstream order; when `None`, the search is the deterministic
/// first-slice path with zero RNG draws.
pub struct Perturbation<'a> {
    /// The persistent training RNG (`LearnProgress->Rand`). Consumed in upstream
    /// draw order: per level one `gen_rand` for the per-feature reseed seed, then
    /// one `std_normal` per candidate feature in `SelectBestCandidate`.
    pub rng: &'a mut TFastRng64,
    /// `scoreStDev` (`CalcScoreStDev`): the perturbation magnitude for this tree
    /// (`random_strength * derivativesStDevFromZero * modelSizeMultiplier`).
    pub score_st_dev: f64,
}

/// Grow one oblivious tree with the OPTIONAL `random_strength` perturbation
/// (`GreedyTensorSearchOblivious` + `SetBestScore`/`SelectBestCandidate`).
///
/// Per level the upstream CPU single-host draw order is reproduced EXACTLY when
/// `perturb` is `Some` (Pitfall 3):
///
/// 1. `randSeed = Rand.GenRand()` — one draw from the main RNG (`CalcScores`,
///    `greedy_tensor_search.cpp:884`).
/// 2. `SetBestScore` (per candidate FEATURE `taskIdx`, `tensor_search_helpers.cpp:716`):
///    a FRESH `TFastRng64::from_seed(randSeed + taskIdx).advance(10)`, then per
///    border `instance = score + std_normal(featRng) * scoreStDev`, keeping the
///    border with the strict-max `instance` as that feature's `BestScore`
///    (Val = winning raw score, StDev = scoreStDev).
/// 3. `SelectBestCandidate` (`:948-966`): per feature ONE
///    `instance = BestScore.GetInstance(Rand)` from the MAIN RNG, then strict
///    `gain > bestGain` first-wins. `gain = instance - scoreBeforeSplit`; since
///    `scoreBeforeSplit` is the same constant for every candidate at a level and
///    there are no feature weights here, the argmax of `gain` equals the argmax
///    of `instance` — so the strict `instance > best` first-wins is exact.
///
/// When `perturb` is `None`, no RNG is touched and the search is the plain
/// strict-`>` L2 argmax (the first-slice / `random_strength == 0` path).
///
/// # Errors
/// - [`CbError::DepthExceeded`] if `depth > MAX_DEPTH` (before allocation).
/// - [`CbError::Degenerate`] if a level has no candidate split at all.
pub fn greedy_tensor_search_oblivious_perturbed(
    matrix: &FeatureMatrix,
    der1: &[f64],
    weight: &[f64],
    scaled_l2: f64,
    depth: usize,
    n_objects: usize,
    mut perturb: Option<Perturbation<'_>>,
) -> CbResult<GrownTree> {
    check_depth(depth)?;

    let mut chosen: Vec<Split> = Vec::with_capacity(depth);

    for _level in 0..depth {
        let best = match perturb.as_mut() {
            None => select_level_plain(matrix, &chosen, der1, weight, scaled_l2, n_objects)?,
            Some(p) => {
                select_level_perturbed(matrix, &chosen, der1, weight, scaled_l2, n_objects, p)?
            }
        };
        chosen.push(best);
    }

    let leaf_of = assign_leaves(matrix, &chosen, n_objects);
    Ok(GrownTree {
        splits: chosen,
        leaf_of,
    })
}

/// One level of the UNPERTURBED search: enumerate candidates in upstream order
/// (feature ascending, border ascending), score each via the L2 calcer, and pick
/// the strict first-wins best. No RNG draws (the first-slice path).
fn select_level_plain(
    matrix: &FeatureMatrix,
    chosen: &[Split],
    der1: &[f64],
    weight: &[f64],
    scaled_l2: f64,
    n_objects: usize,
) -> CbResult<Split> {
    let mut candidates: Vec<Candidate> = Vec::new();
    for feature in 0..matrix.n_features() {
        let borders = matrix.feature_borders.get(feature).map_or(&[][..], Vec::as_slice);
        for &border in borders {
            let score = score_candidate(
                matrix,
                chosen,
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
        CbError::Degenerate("no candidate split available (no feature has any border)".to_owned())
    })?;
    Ok(Split {
        feature: best.feature,
        border: best.border,
    })
}

/// One level of the PERTURBED search reproducing the upstream two-pass draw order
/// (`SetBestScore` then `SelectBestCandidate`, Pitfall 3). See
/// [`greedy_tensor_search_oblivious_perturbed`] for the draw contract.
fn select_level_perturbed(
    matrix: &FeatureMatrix,
    chosen: &[Split],
    der1: &[f64],
    weight: &[f64],
    scaled_l2: f64,
    n_objects: usize,
    perturb: &mut Perturbation<'_>,
) -> CbResult<Split> {
    let std_dev = perturb.score_st_dev;

    // (1) randSeed = Rand.GenRand() — one main-RNG draw per level (CalcScores).
    let rand_seed = perturb.rng.gen_rand();

    // (2) SetBestScore: per candidate FEATURE (taskIdx ascending) reseed a fresh
    //     RNG and pick the strict-best border WITHIN the feature by perturbed
    //     instance. The chosen border's RAW score is that feature's BestScore.Val.
    //     A feature with no border is not a candidate (taskIdx skips it), matching
    //     upstream where empty candidate lists produce no task.
    let mut feature_best: Vec<Option<(f64, f64)>> = Vec::with_capacity(matrix.n_features());
    let mut task_idx: u64 = 0;
    for feature in 0..matrix.n_features() {
        let borders = matrix.feature_borders.get(feature).map_or(&[][..], Vec::as_slice);
        if borders.is_empty() {
            feature_best.push(None);
            continue;
        }
        // TRestorableFastRng64(randSeed + taskIdx); rand.Advance(10).
        let mut feat_rng = TFastRng64::from_seed(rand_seed.wrapping_add(task_idx));
        feat_rng.advance(10);
        task_idx += 1;

        let mut best_instance = MINIMAL_SCORE;
        let mut best_border: f64 = 0.0;
        let mut best_raw: f64 = MINIMAL_SCORE;
        for &border in borders {
            let raw = score_candidate(
                matrix,
                chosen,
                Split { feature, border },
                der1,
                weight,
                scaled_l2,
                n_objects,
            );
            // scoreInstance = scoreWoNoise + std_normal(featRng) * scoreStDev.
            let instance = random_score_instance(raw, std_dev, &mut feat_rng);
            // Strict `>` first-wins on the per-feature border (SetBestScore).
            if instance > best_instance {
                best_instance = instance;
                best_border = border;
                best_raw = raw;
            }
        }
        feature_best.push(Some((best_border, best_raw)));
    }

    // (3) SelectBestCandidate: per feature ONE GetInstance(Rand) from the MAIN
    //     RNG; strict `gain > bestGain` first-wins (scoreBeforeSplit cancels in
    //     the argmax, no feature weights here).
    let mut best_gain = f64::NEG_INFINITY;
    let mut chosen_split: Option<Split> = None;
    for (feature, slot) in feature_best.iter().enumerate() {
        let &Some((border, raw)) = slot else {
            continue;
        };
        let instance = random_score_instance(raw, std_dev, perturb.rng);
        if instance > best_gain {
            best_gain = instance;
            chosen_split = Some(Split { feature, border });
        }
    }

    chosen_split.ok_or_else(|| {
        CbError::Degenerate("no candidate split available (no feature has any border)".to_owned())
    })
}
