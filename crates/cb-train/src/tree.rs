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
///
/// This is the canonical float-split type shared with `cb-model` (`pub use
/// cb_train::Split`). The ORD-04 one-hot categorical split is a SEPARATE type
/// ([`OneHotSplit`]) confined to the categorical tree-growth path so this widely
/// re-used struct stays byte-for-byte unchanged (no cross-crate literal churn).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Split {
    /// The float feature this split tests.
    pub feature: usize,
    /// The split border (threshold); an object passes when `value > border`.
    pub border: f64,
}

/// One one-hot categorical split: a `cat_bin == value` equality test on one
/// categorical feature (`ESplitType::OneHotFeature`,
/// `IsTrueOneHotFeature(featureValue, splitValue) = featureValue == splitValue`,
/// `catboost/libs/model/split.h:16-17`). The bin is the first-seen perfect-hash
/// bin (`cb_data::PerfectHash`) of the object's categorical value (ORD-04).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OneHotSplit {
    /// The categorical feature this split tests.
    pub feature: usize,
    /// The categorical bin this split tests equality against (`splitValue`).
    pub value: u32,
}

/// A split in the ORD-04 categorical tree-growth path: either a float
/// `value > border` ([`Split`]) or a one-hot `cat_bin == value` ([`OneHotSplit`]).
/// Used only inside the categorical-aware search ([`grow_one_hot_tree`]); the
/// numeric first-slice path keeps using bare [`Split`] unchanged.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnySplit {
    /// A float threshold split.
    Float(Split),
    /// A one-hot categorical equality split.
    OneHot(OneHotSplit),
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

/// Per-object access to feature values for split tests: float `value > border`
/// columns plus optional one-hot categorical bin columns (`cat_bins`). SoA
/// layout, object order preserved for D-05.
pub struct FeatureMatrix<'a> {
    /// `feature_values[f]` is float feature `f`'s per-object `f32` column.
    pub feature_values: &'a [Vec<f32>],
    /// `feature_borders[f]` is the ascending candidate borders for float feature
    /// `f` (the model's float-feature borders).
    pub feature_borders: &'a [Vec<f64>],
    /// `cat_bins[c]` is categorical feature `c`'s per-object first-seen
    /// perfect-hash bin column (`cb_data::PerfectHash`), for one-hot
    /// `cat_bin == value` splits (ORD-04). Empty for the numeric-only first
    /// slice (no categorical features).
    pub cat_bins: &'a [Vec<u32>],
}

impl<'a> FeatureMatrix<'a> {
    /// Construct a numeric-only matrix (no categorical features). The
    /// backward-compatible constructor for the float first-slice callers.
    #[must_use]
    pub fn new(feature_values: &'a [Vec<f32>], feature_borders: &'a [Vec<f64>]) -> Self {
        Self {
            feature_values,
            feature_borders,
            cat_bins: &[],
        }
    }

    /// Number of float features.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.feature_values.len()
    }

    /// Number of categorical features (one-hot bin columns).
    #[must_use]
    pub fn n_cat_features(&self) -> usize {
        self.cat_bins.len()
    }

    /// Whether object `obj` passes the float split `value > border` on float
    /// feature `feature`. Out-of-range indices return `false` defensively (the
    /// trainer passes valid indices).
    #[must_use]
    fn passes_float(&self, feature: usize, obj: usize, border: f64) -> bool {
        self.feature_values
            .get(feature)
            .and_then(|col| col.get(obj))
            .is_some_and(|&v| f64::from(v) > border)
    }

    /// Whether object `obj` passes the one-hot split `cat_bin == value` on
    /// categorical feature `feature` (`IsTrueOneHotFeature`, `split.h:16-17`).
    /// Out-of-range indices return `false` defensively.
    #[must_use]
    fn passes_one_hot(&self, feature: usize, obj: usize, value: u32) -> bool {
        self.cat_bins
            .get(feature)
            .and_then(|col| col.get(obj))
            .is_some_and(|&bin| bin == value)
    }

    /// Whether object `obj` passes the float [`Split`] `split`.
    #[must_use]
    fn passes(&self, split: &Split, obj: usize) -> bool {
        self.passes_float(split.feature, obj, split.border)
    }

    /// Whether object `obj` passes the [`AnySplit`] `split` (float or one-hot).
    #[must_use]
    fn passes_any(&self, split: &AnySplit, obj: usize) -> bool {
        match split {
            AnySplit::Float(s) => self.passes_float(s.feature, obj, s.border),
            AnySplit::OneHot(s) => self.passes_one_hot(s.feature, obj, s.value),
        }
    }
}

/// Assign every object to a leaf given the chosen float `splits` (forward bit
/// order). The numeric first-slice path.
fn assign_leaves(matrix: &FeatureMatrix, splits: &[Split], n_objects: usize) -> Vec<usize> {
    (0..n_objects)
        .map(|obj| {
            let passes: Vec<bool> = splits.iter().map(|s| matrix.passes(s, obj)).collect();
            leaf_index(&passes)
        })
        .collect()
}

/// Assign every object to a leaf given the chosen [`AnySplit`] list (float or
/// one-hot, forward bit order). The ORD-04 categorical path.
fn assign_leaves_any(matrix: &FeatureMatrix, splits: &[AnySplit], n_objects: usize) -> Vec<usize> {
    (0..n_objects)
        .map(|obj| {
            let passes: Vec<bool> = splits.iter().map(|s| matrix.passes_any(s, obj)).collect();
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
/// — FLOAT features (feature ascending, border ascending; `AddFloatFeatures`)
/// THEN ONE-HOT categorical features (feature ascending, bin ascending;
/// `AddOneHotFeatures`, `greedy_tensor_search.cpp:171-197`) — score each via the
/// L2 calcer, and pick the strict first-wins best. No RNG draws (the first-slice
/// / D-04 path).
///
/// Each scored [`Candidate`] is paired with the concrete [`Split`] it represents
/// (float or one-hot) at the same index, so the strict first-wins winner maps
/// back to the actual split. The `Candidate::border` carries the float border for
/// a float candidate and the categorical bin (as `f64`) for a one-hot candidate,
/// preserving the ascending iteration order the strict `>` tie-break relies on.
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

// ===========================================================================
// ORD-04 categorical one-hot tree growth (D-04 isolation path)
// ===========================================================================
//
// The categorical-aware greedy search: identical L2 split math and strict
// first-wins tie-break to the float `select_level_plain` above, but the
// candidate set ALSO includes one-hot `cat_bin == value` splits
// (`AddOneHotFeatures`, greedy_tensor_search.cpp:171-197). It rides the EXISTING
// plain boosting unchanged: NO permutation and NO RNG draws (random_strength=0).
// A one-hot split is structurally a binary feature, so a one-hot-only model is
// the same oblivious tree the float path would grow on the equivalent binary
// columns — the load-bearing D-04 parity property the oracle test locks.

/// The grown categorical oblivious tree: ordered [`AnySplit`] splits (float or
/// one-hot) plus the per-object leaf assignment. The categorical analog of
/// [`GrownTree`].
#[derive(Debug, Clone, PartialEq)]
pub struct GrownOneHotTree {
    /// The `depth` ordered splits (float or one-hot), one per level.
    pub splits: Vec<AnySplit>,
    /// Per-object leaf index (`0..2^depth`), object order.
    pub leaf_of: Vec<usize>,
}

/// The distinct categorical bins present in `column`, in ASCENDING bin order
/// (the one-hot candidate enumeration order, bin asc). Checked access only.
fn distinct_bins_ascending(column: &[u32]) -> Vec<u32> {
    let mut seen: Vec<u32> = Vec::new();
    for &bin in column {
        if !seen.contains(&bin) {
            seen.push(bin);
        }
    }
    seen.sort_unstable();
    seen
}

/// Score one [`AnySplit`] candidate applied across the CURRENT level (the
/// categorical analog of [`score_candidate`]): extend the chosen splits with the
/// candidate, assign leaves ([`assign_leaves_any`]), reduce per-leaf stats
/// ordered, and fold the L2 score.
fn score_candidate_any(
    matrix: &FeatureMatrix,
    chosen: &[AnySplit],
    candidate: AnySplit,
    der1: &[f64],
    weight: &[f64],
    scaled_l2: f64,
    n_objects: usize,
) -> f64 {
    let mut splits = chosen.to_vec();
    splits.push(candidate);
    let n_leaves = 1usize << splits.len();
    let leaf_of = assign_leaves_any(matrix, &splits, n_objects);
    let stats: Vec<LeafStats> = reduce_leaf_stats(&leaf_of, der1, weight, n_leaves);
    l2_split_score(&stats, scaled_l2)
}

/// One level of the categorical UNPERTURBED search: enumerate FLOAT candidates
/// (feature asc, border asc; `AddFloatFeatures`) THEN ONE-HOT candidates
/// (categorical feature asc, bin asc; `AddOneHotFeatures`,
/// `greedy_tensor_search.cpp:171-197`), score each via the L2 calcer, and pick
/// the strict first-wins best over that FIXED order. No RNG draws (the D-04
/// path). The winning candidate's concrete [`AnySplit`] is recovered from a
/// vector kept in lockstep with the scores.
fn select_level_one_hot(
    matrix: &FeatureMatrix,
    chosen: &[AnySplit],
    der1: &[f64],
    weight: &[f64],
    scaled_l2: f64,
    n_objects: usize,
) -> CbResult<AnySplit> {
    let mut scored: Vec<(AnySplit, f64)> = Vec::new();

    // FLOAT candidates first (AddFloatFeatures), feature asc / border asc.
    for feature in 0..matrix.n_features() {
        let borders = matrix.feature_borders.get(feature).map_or(&[][..], Vec::as_slice);
        for &border in borders {
            let split = AnySplit::Float(Split { feature, border });
            let score =
                score_candidate_any(matrix, chosen, split, der1, weight, scaled_l2, n_objects);
            scored.push((split, score));
        }
    }

    // ONE-HOT candidates next (AddOneHotFeatures), categorical feature asc / bin
    // asc. One candidate per distinct learn-set bin (the one-hot expansion); the
    // `crate::candidates` routing already gated which categorical features reach
    // here (`1 < cardinality <= one_hot_max_size`).
    for feature in 0..matrix.n_cat_features() {
        let bins = matrix.cat_bins.get(feature).map_or(&[][..], Vec::as_slice);
        for value in distinct_bins_ascending(bins) {
            let split = AnySplit::OneHot(OneHotSplit { feature, value });
            let score =
                score_candidate_any(matrix, chosen, split, der1, weight, scaled_l2, n_objects);
            scored.push((split, score));
        }
    }

    // Strict first-wins over the FIXED enumeration order (floats then one-hots),
    // identical to `select_best_candidate` (strict `>`, NOT `>=`; Pitfall 1).
    let mut best: Option<AnySplit> = None;
    let mut best_score = MINIMAL_SCORE;
    for &(split, score) in &scored {
        if score > best_score {
            best_score = score;
            best = Some(split);
        }
    }
    best.ok_or_else(|| {
        CbError::Degenerate(
            "no candidate split available (no float border and no one-hot categorical)".to_owned(),
        )
    })
}

/// Grow one oblivious tree of depth `depth` over the FLOAT + ONE-HOT candidate
/// set (ORD-04 / D-04), with the strict first-wins greedy search. Each level
/// selects one [`AnySplit`] (float or one-hot) applied across the whole level;
/// a depth-`d` tree has `d` splits and `2^d` leaves. Rides the EXISTING plain
/// boosting math unchanged — NO permutation, NO RNG draws.
///
/// # Errors
/// - [`CbError::DepthExceeded`] if `depth > MAX_DEPTH` (before allocation).
/// - [`CbError::Degenerate`] if a level has no candidate split at all.
pub fn grow_one_hot_tree(
    matrix: &FeatureMatrix,
    der1: &[f64],
    weight: &[f64],
    scaled_l2: f64,
    depth: usize,
    n_objects: usize,
) -> CbResult<GrownOneHotTree> {
    check_depth(depth)?;
    let mut chosen: Vec<AnySplit> = Vec::with_capacity(depth);
    for _level in 0..depth {
        let best = select_level_one_hot(matrix, &chosen, der1, weight, scaled_l2, n_objects)?;
        chosen.push(best);
    }
    let leaf_of = assign_leaves_any(matrix, &chosen, n_objects);
    Ok(GrownOneHotTree {
        splits: chosen,
        leaf_of,
    })
}
