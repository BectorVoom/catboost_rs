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

use cb_compute::{
    bin_of, build_bucket_histogram, calculate_pairwise_score,
    compute_der_sums as compute_pairwise_der_sums, compute_pair_weight_statistics,
    cosine_split_score, l2_split_score, multi_dim_split_score, random_score_instance,
    reduce_leaf_stats, scale_l2_reg, scan_border_to_leaf_stats, scan_borders_to_leaf_stats,
    BucketHistogram, EScoreFunction, GroupSpan, LeafStats, MINIMAL_SCORE,
};

/// Dispatch the configured split-score calcer over reduced leaf statistics.
///
/// catboost CPU supports exactly Cosine (default) and L2 (`score_calcers.h`);
/// the three additional FIRST-ORDER variants (`SolarL2`/`LOOL2`/`SatL2`) are GPU-only
/// upstream (D-6.4-06) and route through the single
/// [`cb_compute::multi_dim_split_score`] seam (D-6.4-03 single code path) wrapped as a
/// one-dimension call so the dim=1 fold is byte-identical to the scalar score. Every
/// reachable variant here computes its per-leaf term from the first-order stats
/// (`sum_weighted_delta` = gradient sum, `sum_weight` = weight count).
///
/// The second-order (Newton) score functions `NewtonL2` / `NewtonCosine` are NOT
/// reachable on this path: `validate_score_function` (CR-01) rejects them at train
/// time because the CPU scoring path produces only the first-order weight-count
/// reduction, so they would silently degrade to L2 / Cosine. They remain in the
/// `EScoreFunction` enum (the score FORMULA seam in `multi_dim_split_score`) for the
/// future GPU der2-hessian-fill path, but cannot be selected for CPU training.
#[inline]
fn split_score(score_function: EScoreFunction, leaves: &[LeafStats], scaled_l2: f64) -> f64 {
    match score_function {
        // Hot path: the two shipped CPU functions stay on the dedicated scalar calcers
        // so the 05-19 Task A L2-vs-Cosine split lock is byte-identical (no-regression).
        EScoreFunction::Cosine => cosine_split_score(leaves, scaled_l2),
        EScoreFunction::L2 => l2_split_score(leaves, scaled_l2),
        // The five GPU-only variants reuse the single multi-dim seam at dim=1.
        EScoreFunction::SolarL2
        | EScoreFunction::NewtonL2
        | EScoreFunction::NewtonCosine
        | EScoreFunction::LOOL2
        | EScoreFunction::SatL2 => {
            let per_dim = [leaves.to_vec()];
            multi_dim_split_score(score_function, &per_dim, scaled_l2)
        }
    }
}
use cb_core::{CbError, CbResult, TFastRng64};

use crate::fold::{body_sum_weights, body_tail_segments};

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

#[cfg(test)]
#[path = "tree_ordered_test.rs"]
mod ordered;

#[cfg(test)]
#[path = "tree_pairwise_test.rs"]
mod pairwise;

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

/// A trainer-side tensor / combination CTR split spec (ORD-05 / D-05): a
/// `ctr_value > border` test on a materialized CTR feature value computed over
/// the combined categorical [`crate::TProjection`]. The trainer persists one of
/// these per chosen CTR split so [`crate::Model`] → `cb_model::Model::from_trained`
/// can lift it into the canonical model's CTR-split representation. The CTR
/// VALUE math underneath is the 05-04/05/06 online accumulation keyed on the
/// combined projection hash (consumed, not re-derived here).
///
/// `prior_num` / `prior_denom` carry the CTR prior (`Borders:Prior=0.5` pins
/// `prior_num = 0.5`, `prior_denom = 1`, RESEARCH A6); `ctr_type` is the i8
/// discriminant of the baked `CtrValueTable` type (the SAME values as
/// `crate::ECtrType` / `cb_model::ECtrType`); `target_border_idx` selects the
/// Buckets per-class numerator (default `0`).
#[derive(Debug, Clone, PartialEq)]
pub struct CtrSplitSpec {
    /// The combined categorical projection (sorted cat-feature member set).
    pub projection: crate::TProjection,
    /// The CTR type i8 discriminant of the baked table this split tests.
    pub ctr_type: i8,
    /// The CTR prior numerator (`PriorNum`).
    pub prior_num: f64,
    /// The CTR prior denominator (`PriorDenom`).
    pub prior_denom: f64,
    /// The Buckets per-class numerator selector (default `0`).
    pub target_border_idx: usize,
    /// The CTR-value threshold; the split passes when `ctr_value > border`.
    pub border: f64,
    /// The inference `Shift` derived from the prior (`calc_normalization(prior_num)`
    /// → `shift`); `0.0` for the in-scope `Borders:Prior=0.5/1`. Threaded into
    /// `cb_model::CtrSplit.shift` so the apply path scales the CTR value into the
    /// same baked-border space (Plan 05-14). Defaults to `0.0` until the bake sets
    /// it on a chosen split.
    pub shift: f64,
    /// The inference `Scale` derived from `ctr_border_count / norm`
    /// (`Borders:Prior=0.5/1` → `15/1 = 15`). Threaded into
    /// `cb_model::CtrSplit.scale`. Defaults to `1.0` until the bake sets it.
    pub scale: f64,
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
    /// The chosen tensor / combination CTR splits (ORD-05), one
    /// [`CtrSplitSpec`] per level a CTR candidate won. EMPTY for the float-only /
    /// one-hot / ordered searches (so their consumers keep compiling unchanged).
    /// The boosting loop persists these onto the [`crate::ObliviousTree`] and
    /// (Plan 05-13 Task 2) REASSIGNS `leaf_of` over the averaging-fold CTR column
    /// using these chosen borders for LEAF-VALUE estimation.
    pub ctr_splits: Vec<CtrSplitSpec>,
    /// The per-level chosen-split kinds in level order: each entry records whether
    /// level `d` is a float split (and its index into [`Self::splits`]) or a CTR
    /// split (and its index into [`Self::ctr_splits`] plus the chosen CTR-value
    /// border the `ctr_bin > border` test uses). Empty for the searches that do
    /// not mix CTR candidates. Drives the forward-bit leaf index when float and
    /// CTR levels interleave, and lets Plan 05-13 Task 2 rebuild `leaf_of` over the
    /// averaging-fold CTR column in the correct level order.
    pub level_kinds: Vec<LevelKind>,
    /// Per-NODE `(left_subtree_diff, right_subtree_diff)` offsets for a TRUE
    /// non-symmetric tree (FEAT-06 / D-6.6-04, `TNonSymmetricTreeStepNode`). EMPTY
    /// for every oblivious / one-hot / ordered / CTR search (those leave the
    /// `splits` + forward-bit `leaf_of` representation untouched, so existing
    /// callers compile and behave unchanged — the empty-parallel-vector precedent of
    /// `ctr_splits` / `level_kinds`). When NON-empty this `GrownTree` is a
    /// non-symmetric node graph produced by [`leaf_wise_grower`]: `splits[i]` is the
    /// split at node `i`, a `(0, 0)` entry marks a terminal (leaf) node, and an
    /// interior node's chosen diff is ADDED to the current node index to reach the
    /// next node. `cb_model::Model::from_trained` lifts this into
    /// `TreeVariant::NonSymmetric`.
    pub step_nodes: Vec<(u16, u16)>,
    /// Per-NODE index into the distinct leaf list for a non-symmetric tree
    /// (`NonSymmetricNodeIdToLeafId`). Only meaningful for terminal nodes (a
    /// `(0, 0)` `step_nodes` entry). EMPTY for the oblivious searches. Same length
    /// as `step_nodes` when populated.
    pub node_id_to_leaf_id: Vec<u32>,
    /// Per-LEVEL CONTINUE direction for a REGION path (GPUT-18 / D-03a). EMPTY for
    /// every oblivious / non-symmetric search. When NON-empty this `GrownTree` is a
    /// region PATH (NOT a node graph): `splits[level]` is the level's split, the
    /// walk descends while `(value > border) == region_directions[level]`, and
    /// `leaf_of[obj]` is the object's terminal bin (`0..=depth`). Length equals
    /// `splits.len()` (== depth). `cb_model::Model::from_trained` lifts this into
    /// `TreeVariant::Region`.
    pub region_directions: Vec<bool>,
    /// Per-LEVEL one-hot flag for a REGION path (`feature.OneHotFeature`). EMPTY for
    /// non-region searches; all `false` for the CPU float grower (which emits
    /// `value > border` splits only). Same length as `region_directions`.
    pub region_one_hot: Vec<bool>,
}

/// One level's chosen-split kind in [`GrownTree::level_kinds`] (ORD-05): a float
/// split or a CTR split. Records the index back into the parallel `splits` /
/// `ctr_splits` vectors plus, for CTR levels, the chosen CTR-value `border` the
/// forward-bit `ctr_bin > border` leaf test uses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LevelKind {
    /// This level is a float `value > border` split; the payload is its index
    /// into [`GrownTree::splits`].
    Float(usize),
    /// This level is a CTR `ctr_bin > border` split; the payload is its index into
    /// [`GrownTree::ctr_splits`] and the chosen CTR-value border.
    Ctr {
        /// Index into [`GrownTree::ctr_splits`].
        ctr_idx: usize,
        /// The chosen CTR-value border (`ctr_bin > border` passes).
        border: f64,
    },
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
///
/// Retained as the REFERENCE per-candidate scorer (the per-object-order rescan) —
/// the documented analog base for [`score_candidate_ordered`] /
/// [`score_candidate_ctr_aware`] / [`score_candidate_any`], and (via
/// [`multi_dim_candidate_score`]) the equivalence anchor the WR-01 unit test
/// pins. The oblivious plain/perturbed grow path no longer calls it: it now
/// scores from the per-level [`BucketHistogram`] prefix scan (PERF-01), so this
/// function is not on the hot path.
#[allow(dead_code)]
fn score_candidate(
    matrix: &FeatureMatrix,
    chosen: &[Split],
    candidate: Split,
    der1: &[f64],
    weight: &[f64],
    scaled_l2: f64,
    n_objects: usize,
    score_function: EScoreFunction,
) -> f64 {
    let mut splits = chosen.to_vec();
    splits.push(candidate);
    let n_leaves = 1usize << splits.len();
    let leaf_of = assign_leaves(matrix, &splits, n_objects);
    multi_dim_candidate_score(&leaf_of, der1, weight, scaled_l2, n_objects, n_leaves, score_function)
}

/// Score one candidate's leaf partition over the (possibly multi-dimensional)
/// dimension-major `der1` buffer via the SINGLE shared cross-dimension accumulator
/// ([`cb_compute::multi_dim_split_score`], RESEARCH "Multi-dim split-score
/// reduction"). `der1` is `der1[d*n_objects + i]` of length
/// `approx_dimension * n_objects`; `weight` is per-object (length `n_objects`,
/// shared across dimensions). The leaf partition `leaf_of` is shared across
/// dimensions (the oblivious structure is one tree).
///
/// The `approx_dimension` is inferred from `der1.len() / n_objects` (so the call
/// sites do not all need a new argument): for every scalar / binary loss `der1` is
/// length `n_objects`, the inferred dimension is `1`, the outer loop runs once, and
/// the resulting score is BYTE-IDENTICAL to the prior single-column
/// `split_score` (D-04 anchor, Pitfall 1). For multiclass `der1` is `k*n_objects`,
/// so each dimension's per-leaf stats are fed into the shared accumulator and the
/// transform is applied once (Cosine couples num/den inside the sqrt).
fn multi_dim_candidate_score(
    leaf_of: &[usize],
    der1: &[f64],
    weight: &[f64],
    scaled_l2: f64,
    n_objects: usize,
    n_leaves: usize,
    score_function: EScoreFunction,
) -> f64 {
    let approx_dimension = if n_objects == 0 {
        1
    } else {
        (der1.len() / n_objects).max(1)
    };
    // Per-dimension per-leaf stats: dimension `d` reduces over its own slice
    // `der1[d*n .. d*n+n]` against the SHARED `leaf_of` / per-object `weight`. At
    // dim=1 this is exactly one `reduce_leaf_stats(leaf_of, der1, weight, …)`.
    let per_dim_leaves: Vec<Vec<LeafStats>> = (0..approx_dimension)
        .map(|d| {
            let base = d * n_objects;
            // WR-01 (06.2-07): on a stride mismatch (der1.len() not a multiple of
            // n_objects, so this dimension's slice is out of range) fall back to an
            // EMPTY slice — scoring 0 for this dimension — NOT `der1` (the whole
            // buffer), which would silently feed a wrong-length, wrong-dimension
            // slice into reduce_leaf_stats and produce a plausible-but-wrong score.
            // The caller (compute_gradients) validates the shape upstream, so the
            // correctly-strided path is unchanged (dim=1 byte-identical, D-04).
            let der1_d = der1.get(base..base + n_objects).unwrap_or(&[]);
            reduce_leaf_stats(leaf_of, der1_d, weight, n_leaves)
        })
        .collect();
    multi_dim_split_score(score_function, &per_dim_leaves, scaled_l2)
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
    score_function: EScoreFunction,
) -> CbResult<GrownTree> {
    // The unperturbed path is `random_strength == 0` with no RNG draws — exactly
    // the first-slice behaviour. Delegate to the perturbed search with `None`
    // perturbation and `None` penalties (the byte-identical pre-6.6 path).
    greedy_tensor_search_oblivious_perturbed(
        matrix, der1, weight, scaled_l2, depth, n_objects, None, score_function, None,
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
    score_function: EScoreFunction,
    penalties: Option<&FeaturePenalties<'_>>,
) -> CbResult<GrownTree> {
    check_depth(depth)?;

    let mut chosen: Vec<Split> = Vec::with_capacity(depth);

    // The reusable across-level grow state (the `TLearnContext` analogue): builds
    // the quantized bin matrix once and holds the current level's histogram, so the
    // O(n) binning + ordered reduction is paid ONCE per level (not once per
    // candidate). Level 0's histogram (single root leaf) is built here.
    let mut scratch = GrowScratch::new(matrix, der1, weight, n_objects);

    for level in 0..depth {
        let best = match perturb.as_mut() {
            None => select_level_plain(matrix, &scratch, scaled_l2, score_function, penalties)?,
            Some(p) => {
                select_level_perturbed(matrix, &scratch, scaled_l2, p, score_function, penalties)?
            }
        };
        // Derive the NEXT level's histogram (subtraction trick) before scoring it;
        // the last level has no successor, so skip its (unused) transition.
        if level + 1 < depth {
            scratch.advance(matrix, der1, weight, &best, n_objects);
        }
        chosen.push(best);
    }

    let leaf_of = assign_leaves(matrix, &chosen, n_objects);
    Ok(GrownTree {
        splits: chosen,
        leaf_of,
        ctr_splits: Vec::new(),
        level_kinds: Vec::new(),
        step_nodes: Vec::new(),
        node_id_to_leaf_id: Vec::new(),
        region_directions: Vec::new(),
        region_one_hot: Vec::new(),
    })
}

/// The reusable across-level grow state (the `TLearnContext` analogue, CONTEXT
/// §Parallelism / PERF-03). It replaces the per-candidate `assign_leaves`
/// + `reduce_leaf_stats` full-dataset rescan with a per-LEVEL histogram:
///
/// - `bins`: the quantized per-feature bin matrix, built ONCE at construction
///   (feature-major `bins[feature * n_objects + obj]`, via [`bin_of`]).
/// - `leaf_of`: the current-partition leaf index per object, maintained
///   INCREMENTALLY (once per chosen split, forward-bit) instead of reassigned per
///   candidate.
/// - `hist`: the current level's per-`(leaf, feature, bin)` [`BucketHistogram`].
///   Each candidate is scored by an `O(n_bins)` prefix scan over this histogram
///   ([`scan_borders_to_leaf_stats`]) fed to the UNCHANGED score math — the O(n)
///   binning cost is paid ONCE per level, not once per candidate (PERF-01).
///
/// The histogram is a fixed-size buffer reused across levels (no per-candidate
/// allocation storm, Spike 004); each level transition derives the next level's
/// histogram from the current one with the SUBTRACTION TRICK
/// ([`GrowScratch::advance`]).
struct GrowScratch {
    /// Feature-major quantized bins: `bins[feature * n_objects + obj]`.
    bins: Vec<u32>,
    /// Current-partition leaf index per object (`0..1 << level`), incremental.
    leaf_of: Vec<usize>,
    /// The current level's per-`(leaf, feature, bin)` histogram.
    hist: BucketHistogram,
    /// Number of float features histogrammed.
    n_features: usize,
    /// Global bins per feature (`max(n_borders) + 1`).
    n_bins: usize,
    /// Approx dimension (`der1.len() / n_objects`, `1` for scalar/binary losses).
    approx_dim: usize,
    /// Current level (`n_leaves == 1 << level`).
    level: usize,
}

impl GrowScratch {
    /// Build the bin matrix and the level-0 (single root leaf) histogram once.
    fn new(matrix: &FeatureMatrix, der1: &[f64], weight: &[f64], n_objects: usize) -> Self {
        let n_features = matrix.n_features();
        // Global bin width = max(n_borders)+1 over features; a feature with fewer
        // borders simply leaves the upper bins empty (the scan walks only its own
        // borders, so the extra width is inert — Pitfall 4 boundary via `bin_of`).
        let n_bins = (0..n_features)
            .map(|f| matrix.feature_borders.get(f).map_or(0, Vec::len) + 1)
            .max()
            .unwrap_or(1);
        let mut bins = vec![0u32; n_features.saturating_mul(n_objects)];
        for f in 0..n_features {
            let borders = matrix.feature_borders.get(f).map_or(&[][..], Vec::as_slice);
            let col = matrix.feature_values.get(f).map_or(&[][..], Vec::as_slice);
            for obj in 0..n_objects {
                let v = col.get(obj).copied().unwrap_or(0.0);
                if let Some(slot) = bins.get_mut(f * n_objects + obj) {
                    *slot = bin_of(borders, v) as u32;
                }
            }
        }
        let approx_dim = if n_objects == 0 {
            1
        } else {
            (der1.len() / n_objects).max(1)
        };
        let leaf_of = vec![0usize; n_objects];
        let hist =
            build_bucket_histogram(&bins, der1, weight, &leaf_of, 1, n_features, n_bins, approx_dim);
        Self {
            bins,
            leaf_of,
            hist,
            n_features,
            n_bins,
            approx_dim,
            level: 0,
        }
    }

    /// `a + b` composed from the sanctioned whole-histogram subtraction
    /// ([`BucketHistogram::remove`] is the only combinator the type exposes):
    /// `a + b = a − (0 − b)`. Used to reunite the FALSE-child (low slots) and
    /// TRUE-child (high slots) halves of a level transition, whose non-zero cells
    /// are disjoint. Each `remove` is a cell-wise f64 op (O(cells), independent of
    /// `n`), so no per-object rescan is introduced.
    fn hist_add(a: &BucketHistogram, b: &BucketHistogram) -> BucketHistogram {
        let zero = a.remove(a);
        let neg_b = zero.remove(b);
        a.remove(&neg_b)
    }

    /// Advance from level `L` to `L+1` after `split` is chosen. Updates `leaf_of`
    /// incrementally (forward-bit: the new split occupies the highest bit, so
    /// `leaf' = leaf + (passes ? 1<<L : 0)` — byte-identical to
    /// `assign_leaves(chosen ++ split)`) and derives the next level's histogram
    /// with the SUBTRACTION TRICK (upstream `FixUpStats`, `scoring.cpp:315`): build
    /// the SMALLER sibling directly and derive the LARGER = parent − smaller. Both
    /// siblings land in their canonical forward-bit slots (false child = parent
    /// leaf `p`, true child = `p + n_parent`), so the derived histogram is
    /// scannable exactly like a freshly-built one.
    fn advance(
        &mut self,
        matrix: &FeatureMatrix,
        der1: &[f64],
        weight: &[f64],
        split: &Split,
        n_objects: usize,
    ) {
        let n_parent = 1usize << self.level;
        let n_next = n_parent << 1;
        let dim = self.approx_dim;
        let nf = self.n_features;
        let nb = self.n_bins;
        // Objects whose leaf index is >= n_leaves are ignored by the builder — the
        // sentinel places an object OUT of the histogram (used to build one sibling).
        let sentinel = usize::MAX;

        let passes: Vec<bool> = (0..n_objects).map(|o| matrix.passes(split, o)).collect();
        let n_true = passes.iter().filter(|&&p| p).count();
        let n_false = n_objects.saturating_sub(n_true);

        let parent = &self.leaf_of;
        let pget = |o: usize| parent.get(o).copied().unwrap_or(0);
        let pass = |o: usize| passes.get(o).copied().unwrap_or(false);
        let build = |assign: &[usize], nl: usize| {
            build_bucket_histogram(&self.bins, der1, weight, assign, nl, nf, nb, dim)
        };

        let next_hist = if self.hist.n_leaves() != n_parent {
            // Defensive: histogram shape drifted from the tracked partition — fall
            // back to a fresh full rebuild (still O(n)/level, T-21-05).
            let next_leaf: Vec<usize> = (0..n_objects)
                .map(|o| pget(o) + if pass(o) { n_parent } else { 0 })
                .collect();
            build(&next_leaf, n_next)
        } else if n_true <= n_false {
            // TRUE is the smaller sibling → build it directly in its high slots;
            // derive the FALSE (larger) child = parent − true in its low slots.
            let true_high: Vec<usize> = (0..n_objects)
                .map(|o| if pass(o) { pget(o) + n_parent } else { sentinel })
                .collect();
            let true_low: Vec<usize> = (0..n_objects)
                .map(|o| if pass(o) { pget(o) } else { sentinel })
                .collect();
            let base_low: Vec<usize> = (0..n_objects).map(pget).collect();
            let h_true_high = build(&true_high, n_next);
            let h_true_low = build(&true_low, n_next);
            let h_base_low = build(&base_low, n_next);
            // FixUpStats: false child (larger) = parent − true sibling (smaller).
            let h_false_low = h_base_low.remove(&h_true_low);
            Self::hist_add(&h_false_low, &h_true_high)
        } else {
            // FALSE is the smaller sibling → build it directly in its low slots;
            // derive the TRUE (larger) child = parent − false in its high slots.
            let false_low: Vec<usize> = (0..n_objects)
                .map(|o| if pass(o) { sentinel } else { pget(o) })
                .collect();
            let false_high: Vec<usize> = (0..n_objects)
                .map(|o| if pass(o) { sentinel } else { pget(o) + n_parent })
                .collect();
            let base_high: Vec<usize> = (0..n_objects).map(|o| pget(o) + n_parent).collect();
            let h_false_low = build(&false_low, n_next);
            let h_false_high = build(&false_high, n_next);
            let h_base_high = build(&base_high, n_next);
            // FixUpStats: true child (larger) = parent − false sibling (smaller).
            let h_true_high = h_base_high.remove(&h_false_high);
            Self::hist_add(&h_false_low, &h_true_high)
        };

        // Incrementally advance the partition (forward-bit, new split highest bit).
        // Computed into a fresh buffer so the `parent`/`pget` immutable borrow above
        // is released before the write-back.
        let next_leaf_of: Vec<usize> = (0..n_objects)
            .map(|o| pget(o) + if pass(o) { n_parent } else { 0 })
            .collect();
        self.leaf_of = next_leaf_of;
        self.hist = next_hist;
        self.level += 1;
    }
}

/// One level of the UNPERTURBED search: enumerate candidates in upstream order
/// (FLOAT features, feature ascending × border ascending; `AddFloatFeatures`),
/// score each from the current level's [`BucketHistogram`] via an `O(n_bins)`
/// prefix scan ([`scan_borders_to_leaf_stats`]) fed to the UNCHANGED
/// [`multi_dim_split_score`], and pick the strict first-wins best. No RNG draws
/// (the first-slice / D-04 path).
///
/// This replaces the per-candidate `assign_leaves` + `reduce_leaf_stats` rescan
/// (PERF-01): the histogram is built once per level in [`GrowScratch`], and every
/// candidate's `&[LeafStats]` is derived from it by the prefix scan. The candidate
/// enumeration ORDER, the FEAT-04 penalty insertion point, and the strict `>`
/// first-wins tie-break are byte-for-byte unchanged.
fn select_level_plain(
    matrix: &FeatureMatrix,
    scratch: &GrowScratch,
    scaled_l2: f64,
    score_function: EScoreFunction,
    penalties: Option<&FeaturePenalties<'_>>,
) -> CbResult<Split> {
    // FEAT-04 penalties are applied ONLY when an active (non-no-op) context is
    // supplied; otherwise the candidate scores stay byte-identical to the pre-6.6
    // path (D-6.6-05 — gate on the param being non-empty).
    let active_penalties = penalties.filter(|p| !p.is_noop());
    let dim = scratch.approx_dim;
    let mut candidates: Vec<Candidate> = Vec::new();
    for feature in 0..matrix.n_features() {
        let borders = matrix.feature_borders.get(feature).map_or(&[][..], Vec::as_slice);
        if borders.is_empty() {
            continue;
        }
        // ONE O(n_bins) prefix scan per border, derived from the per-level
        // histogram (not a per-candidate dataset rescan).
        let scans = scan_borders_to_leaf_stats(&scratch.hist, feature, borders.len(), dim);
        for (bi, &border) in borders.iter().enumerate() {
            let per_dim = scans.get(bi).map_or(&[][..], Vec::as_slice);
            // UNCHANGED score math over the scan-derived per-dimension LeafStats —
            // the same call the prior `multi_dim_candidate_score` made (dim=1 is
            // byte-identical to the scalar split, D-04).
            let raw = multi_dim_split_score(score_function, per_dim, scaled_l2);
            // FEAT-04: multiplicative feature_weight + subtractive first-use /
            // per-object penalties (the two upstream insertion points). The no-op
            // context leaves `raw` untouched.
            let score = match active_penalties {
                Some(p) => p.penalize(feature, raw),
                None => raw,
            };
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
///
/// Only the SOURCE of each candidate's RAW score changes (histogram prefix scan
/// instead of the per-candidate rescan); the per-level `gen_rand`, the per-feature
/// `from_seed(rand_seed + task_idx).advance(10)` reseed, the one `std_normal` per
/// border, the per-feature `feature_best` bookkeeping, and the per-feature main-RNG
/// `GetInstance` are byte-for-byte the same draw sequence and count (Pitfall 3).
fn select_level_perturbed(
    matrix: &FeatureMatrix,
    scratch: &GrowScratch,
    scaled_l2: f64,
    perturb: &mut Perturbation<'_>,
    score_function: EScoreFunction,
    penalties: Option<&FeaturePenalties<'_>>,
) -> CbResult<Split> {
    let std_dev = perturb.score_st_dev;
    // FEAT-04 penalties apply at the per-feature SelectBestCandidate stage (pass 3
    // below) — the multiplicative feature weight scales the gain and the
    // subtractive first-use / per-object penalties act on the per-feature best
    // score (`PenalizeBestSplits`). A no-op (or absent) context leaves the score
    // untouched, so the unpenalized perturbed path is byte-identical (D-6.6-05).
    let active_penalties = penalties.filter(|p| !p.is_noop());
    let dim = scratch.approx_dim;

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

        // ONE O(n_bins) prefix scan per border, from the per-level histogram — the
        // per-border loop shape (and thus the RNG draw count/order) is untouched.
        let scans = scan_borders_to_leaf_stats(&scratch.hist, feature, borders.len(), dim);
        let mut best_instance = MINIMAL_SCORE;
        let mut best_border: f64 = 0.0;
        let mut best_raw: f64 = MINIMAL_SCORE;
        for (bi, &border) in borders.iter().enumerate() {
            let per_dim = scans.get(bi).map_or(&[][..], Vec::as_slice);
            let raw = multi_dim_split_score(score_function, per_dim, scaled_l2);
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
        // FEAT-04: penalize the per-feature best RAW score before the noise instance
        // (multiplicative weight + subtractive first-use/per-object). No-op context
        // ⇒ `raw` unchanged.
        let penalized_raw = match active_penalties {
            Some(p) => p.penalize(feature, raw),
            None => raw,
        };
        let instance = random_score_instance(penalized_raw, std_dev, perturb.rng);
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
// FEAT-06 leaf-wise grower (Depthwise + Lossguide) — TRUE non-symmetric trees
// (D-6.6-04). ONE policy-parameterized core mirroring upstream's shared
// leaf-wise scoring core (`leafwise_scoring.cpp`) + the two policy growers
// (`GreedyTensorSearchDepthwise` :1509, `GreedyTensorSearchLossguide` :1806)
// dispatched in `GreedyTensorSearch`. The split queue / expansion strategy is the
// ONLY thing that differs by policy — the per-leaf candidate enumeration (float
// feature ascending × border ascending), `score_candidate`-style scoring, and the
// strict first-wins tie-break are REUSED verbatim from the oblivious path.
// ===========================================================================

/// The leaf-wise expansion strategy (FEAT-06). Maps from `EGrowPolicy`'s two
/// non-symmetric arms at the dispatch site; `SymmetricTree` / `Region` never reach
/// here (`SymmetricTree` dispatches to the literal oblivious grower, `Region` is
/// rejected up front by `validate_grow_policy`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeafWisePolicy {
    /// Level-order expansion (`GreedyTensorSearchDepthwise`): split every eligible
    /// leaf at the current level before descending, bounded by `max_depth`.
    Depthwise,
    /// Best-gain priority-queue expansion (`GreedyTensorSearchLossguide`): pop the
    /// highest-gain splittable leaf, split it, enqueue its children; stop when the
    /// queue empties or the structure reaches `max_leaves` leaves.
    Lossguide,
}

/// The best per-leaf split found by [`best_split_for_leaf`]: the chosen split and
/// its GAIN (the 2-leaf split score minus the unsplit single-leaf score). `None`
/// when the leaf cannot be split (too few docs, no candidate, or gain `< 1e-9`).
struct LeafBestSplit {
    split: Split,
    gain: f64,
    /// The left-child document indices (split FALSE: `value <= border`).
    left_docs: Vec<usize>,
    /// The right-child document indices (split TRUE: `value > border`).
    right_docs: Vec<usize>,
}

/// The single-leaf unsplit L2 score over a document subset (`der1` / `weight`
/// reduced into ONE leaf bucket), the baseline the per-leaf split gain subtracts.
fn unsplit_leaf_score(
    docs: &[usize],
    der1: &[f64],
    weight: &[f64],
    scaled_l2: f64,
    score_function: EScoreFunction,
) -> f64 {
    let leaf_of: Vec<usize> = vec![0; docs.len()];
    let der1_sub: Vec<f64> = docs.iter().map(|&i| der1.get(i).copied().unwrap_or(0.0)).collect();
    let weight_sub: Vec<f64> = docs.iter().map(|&i| weight.get(i).copied().unwrap_or(0.0)).collect();
    let stats = reduce_leaf_stats(&leaf_of, &der1_sub, &weight_sub, 1);
    split_score(score_function, &stats, scaled_l2)
}

/// Find the best split for ONE leaf's document subset (FEAT-06 shared leaf-wise
/// scoring core). Enumerates candidates in the SAME upstream order the oblivious
/// path uses (float feature ascending, border ascending) and scores each 2-leaf
/// `{value <= border, value > border}` partition from a per-leaf
/// [`BucketHistogram`] built ONCE over this leaf's docs + an `O(n_bins)` prefix
/// scan ([`scan_borders_to_leaf_stats`]) fed to the UNCHANGED score calcer — the
/// histogram scorer (PERF-02), replacing the old per-candidate `reduce_leaf_stats`
/// full-subset rescan. Keeps the strict first-wins best
/// ([`select_best_candidate`] semantics — strict `>`). The reported GAIN is
/// `best_2leaf_score - unsplit_score`; the leaf is splittable only when
/// `docs.len() >= min_data_in_leaf`, a candidate exists, and `gain >= 1e-9`
/// (`greedy_tensor_search.cpp:1582,1635` — the shared `gain < 1e-9` cutoff).
fn best_split_for_leaf(
    matrix: &FeatureMatrix,
    docs: &[usize],
    der1: &[f64],
    weight: &[f64],
    scaled_l2: f64,
    min_data_in_leaf: usize,
    score_function: EScoreFunction,
) -> Option<LeafBestSplit> {
    if docs.len() < min_data_in_leaf || docs.len() < 2 {
        return None;
    }
    let baseline = unsplit_leaf_score(docs, der1, weight, scaled_l2, score_function);

    // Per-leaf BucketHistogram over THIS leaf's document subset — a SINGLE parent
    // leaf (`n_leaves == 1`). Local object index `local` maps to global
    // `docs[local]`; the subset's der1 / weight are gathered in ascending
    // local-object (== ascending doc) order so the reduction order stays the
    // canonical D-05 object order. The leaf-wise / region growers score
    // single-dimension der1 (`approx_dim == 1` — `der1` indexed directly by object,
    // the existing scalar contract, `boosting.rs:3778,3800`), so the histogram
    // carries one delta channel + one shared weight channel.
    let n_docs = docs.len();
    let n_features = matrix.n_features();
    let approx_dim = 1usize;

    // Global bin width = max(n_borders)+1 across features; a feature with fewer
    // borders leaves its upper bins empty (the per-feature scan walks only its own
    // borders — Pitfall 4 boundary via `bin_of`). Mirrors `GrowScratch::new`.
    let n_bins = (0..n_features)
        .map(|f| matrix.feature_borders.get(f).map_or(0, Vec::len) + 1)
        .max()
        .unwrap_or(1);

    // Restricted der1 / weight for this leaf's docs (object order preserved so the
    // reduction order is deterministic).
    let der1_sub: Vec<f64> = docs.iter().map(|&i| der1.get(i).copied().unwrap_or(0.0)).collect();
    let weight_sub: Vec<f64> =
        docs.iter().map(|&i| weight.get(i).copied().unwrap_or(0.0)).collect();

    // Feature-major quantized bins for the doc subset: `bins[feature*n_docs+local]`.
    let mut bins = vec![0u32; n_features.saturating_mul(n_docs)];
    for feature in 0..n_features {
        let borders = matrix.feature_borders.get(feature).map_or(&[][..], Vec::as_slice);
        let col = matrix.feature_values.get(feature).map_or(&[][..], Vec::as_slice);
        for (local, &obj) in docs.iter().enumerate() {
            let v = col.get(obj).copied().unwrap_or(0.0);
            if let Some(slot) = bins.get_mut(feature * n_docs + local) {
                *slot = bin_of(borders, v) as u32;
            }
        }
    }

    // ONE object-order pass builds the parent leaf's histogram; every candidate's
    // [false, true] children then come from the `O(n_bins)` prefix scan of THIS
    // histogram, replacing the per-candidate `assign_leaves` + `reduce_leaf_stats`
    // full-subset rescan (PERF-02). The histogram is fresh-built per leaf (still
    // `O(docs)`/leaf — the target complexity). The two-children subtraction trick
    // (`BucketHistogram::remove`) applies where a parent histogram is threaded to
    // its children; here the per-leaf-independent grower architecture fresh-builds
    // each child instead (histogram threading / scratch reuse is 21-05 scope), so
    // this stays the "fresh-build both" branch.
    let leaf_of_local = vec![0usize; n_docs];
    let hist = build_bucket_histogram(
        &bins,
        &der1_sub,
        &weight_sub,
        &leaf_of_local,
        1,
        n_features,
        n_bins,
        approx_dim,
    );

    let mut best: Option<Candidate> = None;
    let mut best_score = MINIMAL_SCORE;
    // Candidate enumeration: float features ascending, borders ascending (the
    // upstream `AddFloatFeatures` order the oblivious `select_level_plain` uses).
    for feature in 0..n_features {
        let borders = matrix.feature_borders.get(feature).map_or(&[][..], Vec::as_slice);
        if borders.is_empty() {
            continue;
        }
        // ONE `O(n_bins)` prefix scan per border from the per-leaf histogram: the
        // FALSE child (`bins <= border`) is leaf index 0, the TRUE child
        // (`bins > border`) is leaf index 1 — byte-for-byte the leaf order the old
        // `reduce_leaf_stats(leaf_of = passes as 0/1, .., 2)` produced.
        let scans = scan_borders_to_leaf_stats(&hist, feature, borders.len(), approx_dim);
        for (bi, &border) in borders.iter().enumerate() {
            // dim == 1: `per_dim[0]` is `[false_stats, true_stats]` — the exact
            // `&[LeafStats]` the pre-rewrite `split_score` consumed.
            let per_dim = scans.get(bi).map_or(&[][..], Vec::as_slice);
            let stats = per_dim.first().map_or(&[][..], Vec::as_slice);
            let score = split_score(score_function, stats, scaled_l2);
            // STRICT `>` first-wins (Pitfall 1): the FIRST candidate reaching the
            // running max wins; later equal-gain candidates do NOT replace it.
            if score > best_score {
                best_score = score;
                best = Some(Candidate { feature, border, score });
            }
        }
    }

    let cand = best?;
    let gain = best_score - baseline;
    if gain < 1e-9 {
        return None;
    }
    // Partition the leaf's docs by the chosen split: FALSE (value <= border) →
    // left, TRUE (value > border) → right. This is the SAME left/right convention
    // upstream's `AddSplit` uses (left = test-false subtree, right = test-true).
    let mut left_docs = Vec::new();
    let mut right_docs = Vec::new();
    for &obj in docs {
        if matrix.passes_float(cand.feature, obj, cand.border) {
            right_docs.push(obj);
        } else {
            left_docs.push(obj);
        }
    }
    Some(LeafBestSplit {
        split: Split { feature: cand.feature, border: cand.border },
        gain,
        left_docs,
        right_docs,
    })
}

/// One internal or leaf node in the non-symmetric node graph under construction.
enum BuiltNode {
    /// An interior node: its split and the node ids of its left / right children.
    Interior { split: Split, left: usize, right: usize },
    /// A terminal (leaf) node. The distinct-leaf id is assigned in node order at
    /// finalization (`node_to_leaf`), so the variant carries no payload.
    Leaf,
}

/// Grow ONE non-symmetric tree with the leaf-wise grower (FEAT-06 / D-6.6-04).
///
/// `policy` selects the expansion strategy (the ONLY policy-dependent behaviour):
///
/// - [`LeafWisePolicy::Depthwise`] — level-order: split every eligible leaf at the
///   current level (in order), descend, repeat for `0..max_depth`
///   (`GreedyTensorSearchDepthwise`, `greedy_tensor_search.cpp:1509-1700`).
/// - [`LeafWisePolicy::Lossguide`] — best-gain priority queue: pop the highest-gain
///   splittable leaf, split it, enqueue its children's best splits; stop when the
///   queue empties OR the structure reaches `max_leaves` leaves; each leaf is
///   gated on `leaf_depth < max_depth` AND `gain >= 1e-9`
///   (`greedy_tensor_search.cpp:1806-1938`). Queue penalties (FEAT-04) are applied
///   at ENQUEUE time only and NEVER recomputed when a feature first becomes used —
///   the upstream non-recompute is replicated exactly (Pitfall 4,
///   `greedy_tensor_search.cpp:1875-1882`); since the in-scope leaf-wise fixtures
///   carry NO feature penalties, the queue gain is the unpenalized split gain.
///
/// Both policies SHARE [`best_split_for_leaf`] (the candidate enumeration + scoring
/// + strict first-wins core) — exactly upstream's one shared scoring core behind two
/// policy growers. Returns a [`GrownTree`] whose non-symmetric fields (`splits` in
/// node order, `step_nodes`, `node_id_to_leaf_id`) are populated and whose `leaf_of`
/// maps each object to its DISTINCT-leaf index (`0..leaf_count`) for the leaf-value
/// estimation (06.6-05 reconciles leaf VALUES + the apply pointer-walk).
///
/// # Errors
/// - [`CbError::DepthExceeded`] if `max_depth > MAX_DEPTH`.
/// - [`CbError::Degenerate`] if the root cannot be split at all (no candidate /
///   gain below the cutoff), so no non-symmetric tree can be grown.
pub fn leaf_wise_grower(
    policy: LeafWisePolicy,
    matrix: &FeatureMatrix,
    der1: &[f64],
    weight: &[f64],
    scaled_l2: f64,
    max_depth: usize,
    max_leaves: usize,
    min_data_in_leaf: usize,
    n_objects: usize,
    score_function: EScoreFunction,
) -> CbResult<GrownTree> {
    check_depth(max_depth)?;

    // The flat node graph being built, in `AddSplit` order (root first). Each entry
    // is finalized to a `BuiltNode` once its kind (interior vs leaf) is known.
    let mut nodes: Vec<BuiltNode> = Vec::new();
    // Per-node split (interior nodes only; leaf nodes get a placeholder filtered out
    // by the (0,0) step entry). Indexed by node id.
    let mut node_split: Vec<Option<Split>> = Vec::new();
    // Per-node owning document subset, consumed when the node is split.
    let mut node_docs: Vec<Vec<usize>> = Vec::new();
    let mut node_depth: Vec<usize> = Vec::new();

    let root_docs: Vec<usize> = (0..n_objects).collect();

    // Register the root node (id 0).
    let new_node = |nodes: &mut Vec<BuiltNode>,
                    node_split: &mut Vec<Option<Split>>,
                    node_docs: &mut Vec<Vec<usize>>,
                    node_depth: &mut Vec<usize>,
                    docs: Vec<usize>,
                    depth: usize|
     -> usize {
        let id = nodes.len();
        nodes.push(BuiltNode::Leaf); // provisional; promoted to Interior on split
        node_split.push(None);
        node_docs.push(docs);
        node_depth.push(depth);
        id
    };

    let root = new_node(
        &mut nodes,
        &mut node_split,
        &mut node_docs,
        &mut node_depth,
        root_docs,
        0,
    );

    // `leaf_owner[obj]` = the node id of the leaf object `obj` currently lands in.
    let mut leaf_owner: Vec<usize> = vec![root; n_objects];

    // Split node `id` (interior) given its best split, registering two children and
    // routing its docs. Returns (left_child_id, right_child_id).
    let do_split = |nodes: &mut Vec<BuiltNode>,
                        node_split: &mut Vec<Option<Split>>,
                        node_docs: &mut Vec<Vec<usize>>,
                        node_depth: &mut Vec<usize>,
                        leaf_owner: &mut [usize],
                        id: usize,
                        bs: &LeafBestSplit|
     -> (usize, usize) {
        let depth = node_depth.get(id).copied().unwrap_or(0) + 1;
        let left = new_node(
            nodes,
            node_split,
            node_docs,
            node_depth,
            bs.left_docs.clone(),
            depth,
        );
        let right = new_node(
            nodes,
            node_split,
            node_docs,
            node_depth,
            bs.right_docs.clone(),
            depth,
        );
        if let Some(slot) = nodes.get_mut(id) {
            *slot = BuiltNode::Interior { split: bs.split, left, right };
        }
        if let Some(slot) = node_split.get_mut(id) {
            *slot = Some(bs.split);
        }
        for &obj in &bs.left_docs {
            if let Some(o) = leaf_owner.get_mut(obj) {
                *o = left;
            }
        }
        for &obj in &bs.right_docs {
            if let Some(o) = leaf_owner.get_mut(obj) {
                *o = right;
            }
        }
        (left, right)
    };

    match policy {
        LeafWisePolicy::Depthwise => {
            // Level-order: the current level's leaf node ids, oldest first.
            let mut current_level: Vec<usize> = vec![root];
            for _cur_depth in 0..max_depth {
                let mut next_level: Vec<usize> = Vec::new();
                for &leaf in &current_level {
                    let docs = node_docs.get(leaf).cloned().unwrap_or_default();
                    if let Some(bs) = best_split_for_leaf(
                        matrix,
                        &docs,
                        der1,
                        weight,
                        scaled_l2,
                        min_data_in_leaf,
                        score_function,
                    ) {
                        let (l, r) = do_split(
                            &mut nodes,
                            &mut node_split,
                            &mut node_docs,
                            &mut node_depth,
                            &mut leaf_owner,
                            leaf,
                            &bs,
                        );
                        next_level.push(l);
                        next_level.push(r);
                    }
                    // gain < 1e-9 / too few docs → leaf stays terminal (continue).
                }
                if next_level.is_empty() {
                    break;
                }
                current_level = next_level;
            }
        }
        LeafWisePolicy::Lossguide => {
            // Max-priority-queue of (gain, node_id, best_split). A BinaryHeap on an
            // ordered (OrderedF64, seq) key gives best-gain-first with a stable
            // first-enqueued tie-break (deterministic, matching the single-host
            // queue order). `leaf_count` starts at 1 (the root leaf) and grows by 1
            // per split.
            use std::cmp::Ordering;
            use std::collections::BinaryHeap;

            struct QItem {
                gain: f64,
                seq: u64,
                node: usize,
                best: LeafBestSplit,
            }
            impl PartialEq for QItem {
                fn eq(&self, other: &Self) -> bool {
                    self.gain == other.gain && self.seq == other.seq
                }
            }
            impl Eq for QItem {}
            impl Ord for QItem {
                fn cmp(&self, other: &Self) -> Ordering {
                    // Higher gain first; on equal gain the EARLIER seq (enqueued
                    // first) wins — so reverse seq ordering (smaller seq = greater).
                    match self.gain.partial_cmp(&other.gain).unwrap_or(Ordering::Equal) {
                        Ordering::Equal => other.seq.cmp(&self.seq),
                        ord => ord,
                    }
                }
            }
            impl PartialOrd for QItem {
                fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                    Some(self.cmp(other))
                }
            }

            let mut heap: BinaryHeap<QItem> = BinaryHeap::new();
            let mut seq: u64 = 0;
            let mut leaf_count: usize = 1;

            // Enqueue the root's best split (if any). A leaf is only considered when
            // its depth < max_depth (the upstream `needSplit` gate).
            let enqueue = |heap: &mut BinaryHeap<QItem>,
                               seq: &mut u64,
                               node_docs: &[Vec<usize>],
                               node_depth: &[usize],
                               node: usize| {
                if node_depth.get(node).copied().unwrap_or(0) >= max_depth {
                    return;
                }
                let Some(docs) = node_docs.get(node) else {
                    return;
                };
                if let Some(bs) = best_split_for_leaf(
                    matrix,
                    docs,
                    der1,
                    weight,
                    scaled_l2,
                    min_data_in_leaf,
                    score_function,
                ) {
                    heap.push(QItem { gain: bs.gain, seq: *seq, node, best: bs });
                    *seq += 1;
                }
            };

            enqueue(&mut heap, &mut seq, &node_docs, &node_depth, root);

            while let Some(item) = heap.pop() {
                if leaf_count >= max_leaves {
                    break;
                }
                let (l, r) = do_split(
                    &mut nodes,
                    &mut node_split,
                    &mut node_docs,
                    &mut node_depth,
                    &mut leaf_owner,
                    item.node,
                    &item.best,
                );
                // One leaf became two: net +1 leaf.
                leaf_count += 1;
                enqueue(&mut heap, &mut seq, &node_docs, &node_depth, l);
                enqueue(&mut heap, &mut seq, &node_docs, &node_depth, r);
            }
        }
    }

    // ── Finalize the flat node graph → step_nodes + node_id_to_leaf_id ──────────
    // Assign distinct leaf ids in node-id order (terminal nodes only). `splits` is
    // one entry per node (interior nodes carry their split; a terminal node carries
    // a placeholder split filtered out by its (0,0) step entry, matching the
    // `tree_splits` indexed-by-node contract in cb-model::NonSymmetricTree).
    let node_count = nodes.len();
    let mut step_nodes: Vec<(u16, u16)> = Vec::with_capacity(node_count);
    // `u32::MAX` is the INTERIOR-node sentinel, matching the cbm serializer's
    // `distinct_leaves` filter (cbm.rs:200 counts only ids `!= u32::MAX` as real
    // leaves). Interior slots stay at this sentinel; only `BuiltNode::Leaf` arms
    // overwrite their slot with a real distinct leaf id. Without this, interior
    // nodes default to `0`, get mis-counted as leaves, and `save_cbm` fails with
    // `ModelError::SchemaVersion` (CR-02).
    let mut node_id_to_leaf_id: Vec<u32> = vec![u32::MAX; node_count];
    let mut splits: Vec<Split> = Vec::with_capacity(node_count);
    let mut next_leaf_id: u32 = 0;
    // Map node id → distinct leaf id for terminal nodes (assigned in node order).
    let mut node_to_leaf: Vec<Option<u32>> = vec![None; node_count];
    for (id, node) in nodes.iter().enumerate() {
        match node {
            BuiltNode::Interior { split, left, right } => {
                splits.push(*split);
                // Interior nodes carry the `u32::MAX` interior sentinel explicitly
                // (the init already covers it; this keeps the intent local to the
                // arm and matches the cbm.rs:200 `distinct_leaves` filter — CR-02).
                if let Some(slot) = node_id_to_leaf_id.get_mut(id) {
                    *slot = u32::MAX;
                }
                // step diffs are (child_id - this_id), the upstream
                // `LeftSubtreeDiff` / `RightSubtreeDiff` offsets added to walk down.
                // Children are registered after their parent, so both diffs are
                // positive. Convert via checked `u16::try_from` (NOT a bare
                // `as u16`) so an out-of-u16-range diff at extreme `max_depth`
                // surfaces as a typed `CbError::OutOfRange` instead of silently
                // truncating the topology (CR-01).
                let ld = u16::try_from(*left - id).map_err(|_| {
                    CbError::OutOfRange(format!(
                        "non-symmetric step-node child diff {} at node {} exceeds \
                         u16 range (max_depth too large for the flat node encoding)",
                        *left - id,
                        id
                    ))
                })?;
                let rd = u16::try_from(*right - id).map_err(|_| {
                    CbError::OutOfRange(format!(
                        "non-symmetric step-node child diff {} at node {} exceeds \
                         u16 range (max_depth too large for the flat node encoding)",
                        *right - id,
                        id
                    ))
                })?;
                step_nodes.push((ld, rd));
            }
            BuiltNode::Leaf => {
                // Placeholder split for a leaf node (never read — its step entry is
                // (0,0)); reuse a zeroed Split so `splits` stays node-indexed.
                splits.push(Split { feature: 0, border: 0.0 });
                step_nodes.push((0, 0));
                if let Some(slot) = node_to_leaf.get_mut(id) {
                    *slot = Some(next_leaf_id);
                }
                if let Some(slot) = node_id_to_leaf_id.get_mut(id) {
                    *slot = next_leaf_id;
                }
                next_leaf_id += 1;
            }
        }
    }

    if next_leaf_id < 2 {
        // The root never split → no non-symmetric tree (degenerate, matching the
        // oblivious `Degenerate` contract for a level with no candidate split).
        return Err(CbError::Degenerate(
            "leaf-wise grower produced no split (root gain below the 1e-9 cutoff or \
             too few documents)"
                .to_owned(),
        ));
    }

    // `leaf_of[obj]` = the DISTINCT leaf id object `obj` lands in (for leaf-value
    // estimation). Every object's owning node is terminal by construction.
    let leaf_of: Vec<usize> = leaf_owner
        .iter()
        .map(|&node| node_to_leaf.get(node).and_then(|o| *o).unwrap_or(0) as usize)
        .collect();

    Ok(GrownTree {
        splits,
        leaf_of,
        ctr_splits: Vec::new(),
        level_kinds: Vec::new(),
        step_nodes,
        node_id_to_leaf_id,
        region_directions: Vec::new(),
        region_one_hot: Vec::new(),
    })
}

/// Grow ONE region PATH tree (GPUT-18 / D-03a — upstream's `TRegionModel`, the
/// single largest Phase-12 lift; "Region OUT" gap now closed on the CPU).
///
/// A region is an oblivious-like PATH, NOT a binary node graph: at each level the
/// current path FRONTIER (the deepest not-yet-terminal leaf) is scored and its best
/// split is taken; ONE child continues as the next frontier and the OTHER becomes a
/// terminal leaf. A depth-`d` region has `d` levels and exactly `d + 1` leaves
/// (`ComputeOptimalSplitsRegion`, `greedy_search_helper.cpp::SelectLeavesToSplit` —
/// `MaxLeaves = MaxDepth + 1`). This does NOT emit `step_nodes` /
/// `node_id_to_leaf_id` (that is the non-symmetric node-graph finalization — Pitfall
/// 2); it emits `region_directions` + `region_one_hot` + a `leaf_of` in bin order.
///
/// **Continue direction** (upstream `SelectLeavesToSplit`): among the two children
/// of the frontier split, the one with the BETTER (higher-gain) best-split is the
/// next frontier; the other diverges. The recorded per-level direction is `true`
/// when the CONTINUING child is the `value > border` (passes) child, else `false`
/// (so the apply walk `(value > border) == direction` descends into the frontier).
/// Ties prefer the passes-child (`>=`), a deterministic, reproducible rule.
///
/// **Leaf bins.** `leaf_of[obj]` is the object's terminal bin: `k` for an object
/// that diverged at level `k` (`0 <= k < depth`), or `depth` for an object that
/// matched every continue direction. This is EXACTLY the `AddRegionImpl` walk's
/// `bin`, so the emitted `leaf_of` and the apply path agree by construction.
///
/// # Errors
/// - [`CbError::DepthExceeded`] if `max_depth > MAX_DEPTH`.
/// - [`CbError::Degenerate`] if the ROOT cannot be split (no candidate / gain below
///   the `1e-9` cutoff), so no region path exists (mirrors the leaf-wise contract).
#[allow(clippy::too_many_arguments)]
pub fn region_grower(
    matrix: &FeatureMatrix,
    der1: &[f64],
    weight: &[f64],
    scaled_l2: f64,
    max_depth: usize,
    min_data_in_leaf: usize,
    n_objects: usize,
    score_function: EScoreFunction,
) -> CbResult<GrownTree> {
    check_depth(max_depth)?;

    // The current path frontier's owning document subset (root = all objects).
    let mut frontier: Vec<usize> = (0..n_objects).collect();
    // Per-level chosen split + continue direction (`region_directions`), in walk
    // order. `depth == splits.len()`.
    let mut splits: Vec<Split> = Vec::new();
    let mut directions: Vec<bool> = Vec::new();
    // Per-bin terminal document subsets: `terminal_docs[k]` diverged at level `k`.
    // The final frontier (bin == depth) is pushed after the loop.
    let mut terminal_docs: Vec<Vec<usize>> = Vec::new();

    for _level in 0..max_depth {
        // Score the frontier's best split. `best_split_for_leaf` enforces
        // `gain >= 1e-9` + `min_data_in_leaf`, so a beneficial split is required to
        // extend the path (upstream `Score < 0` gate; the root splits only when a
        // beneficial candidate exists — a degenerate root errors below).
        let Some(bs) = best_split_for_leaf(
            matrix,
            &frontier,
            der1,
            weight,
            scaled_l2,
            min_data_in_leaf,
            score_function,
        ) else {
            break;
        };

        // Continue direction: the child whose OWN best split has the higher gain is
        // the next frontier (upstream picks the lower-Score == higher-gain child to
        // split next). A non-splittable child scores `NEG_INFINITY`. Ties prefer the
        // passes (right) child (`>=`), deterministic.
        let right_gain = best_split_for_leaf(
            matrix,
            &bs.right_docs,
            der1,
            weight,
            scaled_l2,
            min_data_in_leaf,
            score_function,
        )
        .map_or(f64::NEG_INFINITY, |b| b.gain);
        let left_gain = best_split_for_leaf(
            matrix,
            &bs.left_docs,
            der1,
            weight,
            scaled_l2,
            min_data_in_leaf,
            score_function,
        )
        .map_or(f64::NEG_INFINITY, |b| b.gain);

        splits.push(bs.split);
        if right_gain >= left_gain {
            // The passes (`value > border`) child continues → direction `true`; the
            // not-passes child (left_docs) diverges into this level's terminal bin.
            directions.push(true);
            terminal_docs.push(bs.left_docs);
            frontier = bs.right_docs;
        } else {
            directions.push(false);
            terminal_docs.push(bs.right_docs);
            frontier = bs.left_docs;
        }
    }

    if splits.is_empty() {
        // The root never split → no region path (degenerate, matching the leaf-wise
        // `Degenerate` contract for a root with no beneficial candidate).
        return Err(CbError::Degenerate(
            "region grower produced no split (root gain below the 1e-9 cutoff or too \
             few documents)"
                .to_owned(),
        ));
    }

    let depth = splits.len();
    // Assign each object its terminal bin (`AddRegionImpl` bin). Objects in
    // `terminal_docs[k]` diverged at level `k` → bin `k`; the surviving `frontier`
    // matched every direction → bin `depth`. Default `0` is overwritten for every
    // object (each object lands in exactly one terminal set).
    let mut leaf_of: Vec<usize> = vec![0usize; n_objects];
    for (bin, docs) in terminal_docs.iter().enumerate() {
        for &obj in docs {
            if let Some(slot) = leaf_of.get_mut(obj) {
                *slot = bin;
            }
        }
    }
    for &obj in &frontier {
        if let Some(slot) = leaf_of.get_mut(obj) {
            *slot = depth;
        }
    }

    Ok(GrownTree {
        splits,
        leaf_of,
        ctr_splits: Vec::new(),
        level_kinds: Vec::new(),
        step_nodes: Vec::new(),
        node_id_to_leaf_id: Vec::new(),
        region_directions: directions,
        // CPU float grower emits `value > border` splits only (no one-hot).
        region_one_hot: vec![false; depth],
    })
}

// ===========================================================================
// ORD-02 ordered split-scoring subsystem (the structural heart of Ordered
// boosting — per-segment ordered L2 score over the learning fold's BodyTailArr)
// ===========================================================================
//
// # Why Ordered trees differ from Plain
//
// The Plain search (`select_level_plain`) scores every candidate on the WHOLE
// fold (`reduce_leaf_stats` over all objects, one `l2_split_score`). Ordered
// boosting instead scores a candidate by SUMMING its per-segment ordered L2
// score across the learning fold's `BodyTailArr` (`scoring.cpp:746-760`
// `CalculateNonPairwiseScore`: `for bodyTailIdx in 0..GetBodyTailCount()` the
// score calcer's `AddLeaf` is additive, so the final candidate score is the SUM
// over segments). Each segment `(body_finish, tail_finish)` uses a per-segment
// `scaledL2 = l2 * (BodySumWeight / BodyFinish)` (`scoring.cpp:746-748`,
// `online_predictor.h::ScaleL2Reg` = [`scale_l2_reg`]).
//
// # Per-segment stats
//
// `scoring.cpp:283-308` `CalcStatsKernel` non-plain branch: the BODY range
// `[0, body_finish)` contributes `WeightedDerivatives` and the TAIL range
// `[body_finish, tail_finish)` contributes `SampleWeightedDerivatives` — both
// into the SAME per-leaf stats for that segment. Under the in-scope
// `ordered_boost` fixture `random_strength == 0` ⇒ NO bootstrap perturbation ⇒
// `SampleWeightedDerivatives == WeightedDerivatives` (no RNG draws here, the
// D-11 multi-tree Box-Muller drift does NOT apply). So a segment accumulates the
// SAME per-object `der`/`weight` over BOTH the body rows and the tail rows, in
// permutation order, into the candidate's leaf buckets, then folds
// `l2_split_score(stats, scaledL2_segment)`.
//
// The leaf VALUES (downstream, 05-10) still come from `CalcLeafValuesSimple` on
// the AVERAGING fold exactly as Plain (STATE.md re-scope note), so `leaf_of` is
// assigned over the object order (forward-bit `leaf_index`) like the Plain path.

/// Per-segment ordered leaf statistics for ONE candidate split: accumulate the
/// candidate's per-leaf `(sum_weighted_delta, sum_weight)` over the segment's
/// BODY rows `[0, body_finish)` then TAIL rows `[body_finish, tail_finish)`, both
/// walked in `permutation` order (`scoring.cpp:283-308` non-plain branch). Under
/// `random_strength == 0` the tail's `SampleWeightedDerivatives` equals the body's
/// `WeightedDerivatives`, so both ranges accumulate the SAME `der`/`weight`.
///
/// Bounds are checked: a permutation index out of range, or a `der`/`weight`
/// slice shorter than the document it indexes, returns [`CbError::Degenerate`]
/// (T-05-08-01/02 — no raw index, no panic). The final per-leaf sums route
/// through [`reduce_leaf_stats`] so the fold order is the sanctioned `sum_f64`
/// object order (D-08, T-05-08-03).
fn ordered_segment_leaf_stats(
    leaf_of: &[usize],
    der1: &[f64],
    weight: &[f64],
    permutation: &[i32],
    body_finish: usize,
    tail_finish: usize,
    n_leaves: usize,
) -> CbResult<Vec<LeafStats>> {
    // Gather this segment's member objects (body ∪ tail) in permutation order,
    // along with their parallel der/weight, then reduce via the sanctioned
    // primitive so the sum order is the canonical object order (D-08).
    let mut seg_leaf_of: Vec<usize> = Vec::new();
    let mut seg_der: Vec<f64> = Vec::new();
    let mut seg_weight: Vec<f64> = Vec::new();

    let n = permutation.len();
    let upper = tail_finish.min(n);
    // Walk [0, tail_finish): the BODY rows [0, body_finish) then the TAIL rows
    // [body_finish, tail_finish) are accumulated identically (random_strength == 0
    // ⇒ SampleWeightedDerivatives == WeightedDerivatives), so a single contiguous
    // walk over [0, tail_finish) is exact. `body_finish` is retained in the
    // signature to document the body/tail split the segment represents.
    let _ = body_finish;
    for p in 0..upper {
        let Some(&doc_i) = permutation.get(p) else {
            return Err(CbError::Degenerate(
                "ordered score: permutation index out of range".to_owned(),
            ));
        };
        if doc_i < 0 {
            return Err(CbError::Degenerate(
                "ordered score: negative permutation index".to_owned(),
            ));
        }
        let doc = doc_i as usize;
        let (Some(&leaf), Some(&d)) = (leaf_of.get(doc), der1.get(doc)) else {
            return Err(CbError::Degenerate(
                "ordered score: leaf_of / der shorter than permutation".to_owned(),
            ));
        };
        let w = if weight.is_empty() {
            1.0
        } else {
            match weight.get(doc) {
                Some(&w) => w,
                None => {
                    return Err(CbError::Degenerate(
                        "ordered score: weight shorter than permutation".to_owned(),
                    ))
                }
            }
        };
        seg_leaf_of.push(leaf);
        seg_der.push(d);
        seg_weight.push(w);
    }

    Ok(reduce_leaf_stats(&seg_leaf_of, &seg_der, &seg_weight, n_leaves))
}

// DEFERRAL (Phase 21 → Phase 22): the ordered-boosting split search
// (`score_candidate_ordered` / `select_level_ordered` below) is INTENTIONALLY left
// on its per-segment `assign_leaves` + `reduce_leaf_stats` rescan this phase. The
// Phase-21 histogram rewrite (PERF-01/02) targets the DEFAULT oblivious
// plain/perturbed path (`select_level_plain` / `select_level_perturbed`); a
// per-segment prefix-histogram for the ordered path is additional scope (each
// `(body_finish, tail_finish)` segment needs its own bucket accumulation). Parity
// is trivially preserved because this code is untouched; the residual ordered-path
// slowness is tracked for Phase 22 (RESEARCH Open Question 1 / A2).

/// Score one candidate split with the ORDERED per-segment sum (the ordered analog
/// of [`score_candidate`]): extend `chosen` with `candidate`, assign leaves over
/// the object order, then for each `(body_finish, tail_finish)` segment compute
/// the per-segment `scaledL2 = l2 * (body_sum_weight / body_finish)`
/// ([`scale_l2_reg`], `scoring.cpp:746-748`), fold `l2_split_score`, and SUM
/// across all segments (`scoreCalcer->AddLeaf` additive over `bodyTailIdx`,
/// `scoring.cpp:746-760`).
///
/// `segments` and `seg_body_sum_weights` are paired index-for-index (segment `s`
/// uses `segments[s]` and `seg_body_sum_weights[s]`). A segment with
/// `body_finish == 0` would divide by zero in the scaled L2; [`scale_l2_reg`]
/// guards that by returning `l2` directly (T-05-08-02), and `body_tail_segments`
/// never emits a zero `body_finish` (floor `SelectMinBatchSize ≥ 1`).
///
/// The per-segment sums route through [`reduce_leaf_stats`]/[`l2_split_score`]
/// (D-08); the cross-segment sum is a short accumulation of the per-segment
/// scores, folded via `sum_f64` for the sanctioned reduction discipline.
#[allow(clippy::too_many_arguments)]
fn score_candidate_ordered(
    matrix: &FeatureMatrix,
    chosen: &[Split],
    candidate: Split,
    der1: &[f64],
    weight: &[f64],
    permutation: &[i32],
    segments: &[(usize, usize)],
    seg_body_sum_weights: &[f64],
    l2_leaf_reg: f64,
    n_objects: usize,
) -> CbResult<f64> {
    let mut splits = chosen.to_vec();
    splits.push(candidate);
    let n_leaves = 1usize << splits.len();
    let leaf_of = assign_leaves(matrix, &splits, n_objects);

    let mut segment_scores: Vec<f64> = Vec::with_capacity(segments.len());
    for (idx, &(body_finish, tail_finish)) in segments.iter().enumerate() {
        let body_sum_weight = seg_body_sum_weights.get(idx).copied().unwrap_or(0.0);
        // Per-segment scaled L2 = l2 * (BodySumWeight / BodyFinish)
        // (scoring.cpp:746-748). scale_l2_reg guards body_finish == 0.
        let scaled_l2_segment = scale_l2_reg(l2_leaf_reg, body_sum_weight, body_finish);
        let stats = ordered_segment_leaf_stats(
            &leaf_of,
            der1,
            weight,
            permutation,
            body_finish,
            tail_finish,
            n_leaves,
        )?;
        segment_scores.push(l2_split_score(&stats, scaled_l2_segment));
    }
    Ok(cb_core::sum_f64(&segment_scores))
}

/// One level of the ORDERED search: enumerate candidates in the SAME upstream
/// order as [`select_level_plain`] (float feature ascending, border ascending —
/// the in-scope `ordered_boost` fixture is numeric-only), score each via the
/// segment-summed ordered L2 ([`score_candidate_ordered`]), and pick the strict
/// first-wins best via the SAME [`select_best_candidate`] discipline (strict `>`,
/// feature asc then border asc; Pitfall 1). No RNG draws (`random_strength == 0`).
#[allow(clippy::too_many_arguments)]
fn select_level_ordered(
    matrix: &FeatureMatrix,
    chosen: &[Split],
    der1: &[f64],
    weight: &[f64],
    permutation: &[i32],
    segments: &[(usize, usize)],
    seg_body_sum_weights: &[f64],
    l2_leaf_reg: f64,
    n_objects: usize,
) -> CbResult<Split> {
    let mut candidates: Vec<Candidate> = Vec::new();
    for feature in 0..matrix.n_features() {
        let borders = matrix.feature_borders.get(feature).map_or(&[][..], Vec::as_slice);
        for &border in borders {
            let score = score_candidate_ordered(
                matrix,
                chosen,
                Split { feature, border },
                der1,
                weight,
                permutation,
                segments,
                seg_body_sum_weights,
                l2_leaf_reg,
                n_objects,
            )?;
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

/// Grow one oblivious tree with the ORDERED per-segment split-scoring subsystem
/// (the structural heart of ORD-02). Per level `0..depth`, each candidate is
/// scored by SUMMING its per-segment ordered L2 score across the learning fold's
/// `body_tail_segments(n_objects, fold_len_multiplier)` (each segment's
/// `scaledL2 = l2 * (BodySumWeight / BodyFinish)`), then the strict first-wins
/// best is chosen ([`select_best_candidate`], `>` not `>=`). After `depth` levels
/// `leaf_of` is assigned over the object order (forward-bit `leaf_index`) so the
/// downstream leaf-value estimation (05-10) runs on the averaging fold exactly as
/// Plain (`CalcLeafValuesSimple`).
///
/// `permutation` is the learning fold's object permutation (the order the body/
/// tail rows are walked); `der1`/`weight` are object-order parallel slices. The
/// per-segment `BodySumWeight` is derived from `body_sum_weights(n_objects,
/// fold_len_multiplier, weight)` (the same fold machinery 05-03 locked).
///
/// # Degeneration anchor
///
/// At a single full-span segment `[(n, n)]` (the plain `body_tail_segments`
/// degenerate case) AND the identity permutation, the segment-summed ordered
/// score reduces to the plain whole-fold L2 score, so this produces the SAME
/// splits as [`greedy_tensor_search_oblivious`] (falsifiable degeneration anchor,
/// unit-locked in `tree_ordered_test.rs`).
///
/// # Errors
/// - [`CbError::DepthExceeded`] if `depth > MAX_DEPTH` (before allocation).
/// - [`CbError::Degenerate`] if a level has no candidate split, or a permutation
///   index / body-tail boundary is out of range (T-05-08-01/02).
#[allow(clippy::too_many_arguments)]
pub fn greedy_tensor_search_oblivious_ordered(
    matrix: &FeatureMatrix,
    der1: &[f64],
    weight: &[f64],
    permutation: &[i32],
    l2_leaf_reg: f64,
    fold_len_multiplier: f64,
    depth: usize,
    n_objects: usize,
) -> CbResult<GrownTree> {
    check_depth(depth)?;

    // The learning fold's BodyTailArr for this object count + multiplier, and the
    // per-segment body prefix weights (fold.rs, 05-03 — consume, do not re-port).
    let segments = body_tail_segments(n_objects, fold_len_multiplier);
    if segments.is_empty() {
        return Err(CbError::Degenerate(
            "ordered search: empty body/tail segments (n_objects == 0)".to_owned(),
        ));
    }
    let seg_body_sum_weights = body_sum_weights(n_objects, fold_len_multiplier, weight);

    let mut chosen: Vec<Split> = Vec::with_capacity(depth);
    for _level in 0..depth {
        let best = select_level_ordered(
            matrix,
            &chosen,
            der1,
            weight,
            permutation,
            &segments,
            &seg_body_sum_weights,
            l2_leaf_reg,
            n_objects,
        )?;
        chosen.push(best);
    }

    let leaf_of = assign_leaves(matrix, &chosen, n_objects);
    Ok(GrownTree {
        splits: chosen,
        leaf_of,
        ctr_splits: Vec::new(),
        level_kinds: Vec::new(),
        step_nodes: Vec::new(),
        node_id_to_leaf_id: Vec::new(),
        region_directions: Vec::new(),
        region_one_hot: Vec::new(),
    })
}

// ===========================================================================
// ORD-05 CTR-feature scoring in the oblivious search (the STRUCTURE half)
// ===========================================================================
//
// The CTR-aware oblivious search enumerates, at each level, BOTH the float
// candidates (the existing `select_level_plain` order, unchanged) AND — for each
// materialized `CtrFeatureColumn` — one candidate per CTR border (the value test
// `ctr_bin > border`, borders `0..ctr_border_count`). Every candidate is scored
// with the SAME `l2_split_score` over `reduce_leaf_stats` the float path uses (no
// forked scoring math). The fixed candidate-iteration order is FLOAT-then-CTR
// (upstream enumerates float features, THEN AddTreeCtrs), and the strict
// first-wins (`> best`) tie-break is reused verbatim (Pitfall 1).
//
// A winning CTR split is recorded as a `CtrSplitSpec` carrying the chosen
// CTR-value border + the column's prior num/denom PAIR; `GrownTree.level_kinds`
// records which level each split occupies so the forward-bit leaf index assigns
// CTR bits (`ctr_bin > border`) and float bits in the correct level order.
//
// IMPORTANT: this search computes the STRUCTURE only. The `grown.leaf_of` it
// returns is the structure partition over the IDENTITY-fold CTR column; Plan
// 05-13 Task 2 REPLACES the leaf_of used for leaf VALUES with the averaging-fold
// partition (boosting.rs), driven by the chosen `ctr_splits` + `level_kinds`.

/// A unified candidate split in the CTR-aware oblivious search: a float
/// `value > border` ([`Split`]) or a CTR `ctr_bin > border` over one materialized
/// [`crate::ctr::CtrFeatureColumn`] (identified by its index into the supplied
/// column slice). Used only inside [`greedy_tensor_search_oblivious_with_ctr`].
#[derive(Debug, Clone, Copy, PartialEq)]
enum CtrAwareSplit {
    /// A float threshold split.
    Float(Split),
    /// A CTR split: the column index and the CTR-value border (`ctr_bin > border`).
    Ctr { col: usize, border: f64 },
}

/// Whether object `obj` passes the [`CtrAwareSplit`] given the materialized CTR
/// columns. A float split delegates to [`FeatureMatrix::passes_float`]; a CTR
/// split tests `ctr_bin > border` over the column's quantized `bins` (the
/// forward-bit CTR test). Out-of-range indices return `false` defensively.
fn passes_ctr_aware(
    matrix: &FeatureMatrix,
    ctr_features: &[crate::ctr::CtrFeatureColumn],
    split: &CtrAwareSplit,
    obj: usize,
) -> bool {
    match split {
        CtrAwareSplit::Float(s) => matrix.passes_float(s.feature, obj, s.border),
        CtrAwareSplit::Ctr { col, border } => ctr_features
            .get(*col)
            .and_then(|c| c.bins.get(obj))
            .is_some_and(|&bin| f64::from(bin) > *border),
    }
}

/// Assign every object to a leaf given the chosen [`CtrAwareSplit`] list (float or
/// CTR, forward bit order). The ORD-05 structure path.
fn assign_leaves_ctr_aware(
    matrix: &FeatureMatrix,
    ctr_features: &[crate::ctr::CtrFeatureColumn],
    splits: &[CtrAwareSplit],
    n_objects: usize,
) -> Vec<usize> {
    (0..n_objects)
        .map(|obj| {
            let passes: Vec<bool> = splits
                .iter()
                .map(|s| passes_ctr_aware(matrix, ctr_features, s, obj))
                .collect();
            leaf_index(&passes)
        })
        .collect()
}

/// Build the ONE per-level [`BucketHistogram`] the CTR-aware level search scores
/// every candidate from — over the CURRENT partition (`chosen`) and the UNION of
/// the FLOAT feature columns (bins via [`bin_of`] over `feature_borders`) AND the
/// materialized CTR bin columns ([`crate::ctr::CtrFeatureColumn::bins`], the
/// integer CTR bins fed DIRECTLY as their own histogram bin — the `ctr_bin > b`
/// test is the same prefix-scan boundary as a float `bins > b`, RESEARCH
/// §Coverage / Pitfall 4). This replaces the per-candidate `assign_leaves_ctr_aware`
/// + `reduce_leaf_stats` full-dataset rescan (PERF-02): the `O(n)` binning +
/// ordered reduction is paid ONCE per level, then every candidate comes from an
/// `O(n_bins)` prefix scan ([`score_candidate_ctr_aware`]).
///
/// The combined histogram lays the float features first (indices `0..n_float`,
/// matching [`FeatureMatrix::n_features`]) then the CTR columns
/// (`n_float + col`), so a float candidate scans feature `feature` and a CTR
/// candidate on column `col` scans feature `n_float + col`. The partition is the
/// SAME [`assign_leaves_ctr_aware`] forward-bit leaf order the once-per-tree final
/// assignment uses, so the scan's `[false, true]` children are byte-for-byte the
/// leaves `reduce_leaf_stats(assign_leaves_ctr_aware(chosen ++ candidate))`
/// produced. The CTR path scores single-dimension der1 (`approx_dim == 1`, the
/// `reduce_leaf_stats` + `split_score` scalar contract) so the histogram carries
/// one delta channel + one shared weight channel.
fn build_ctr_aware_histogram(
    matrix: &FeatureMatrix,
    ctr_features: &[crate::ctr::CtrFeatureColumn],
    ctr_border_count: usize,
    chosen: &[CtrAwareSplit],
    der1: &[f64],
    weight: &[f64],
    n_objects: usize,
) -> BucketHistogram {
    let n_float = matrix.n_features();
    let n_ctr = ctr_features.len();
    let n_features = n_float + n_ctr;
    let approx_dim = 1usize;

    // Global bin width across BOTH kinds of feature. Float feature `f` needs
    // `n_borders(f) + 1` bins. A CTR column needs enough bins for (a) its highest
    // actual bin value and (b) the scan's highest border index `ctr_border_count - 1`
    // (the `.get(0..=border)` false-side slice must stay in range even when no
    // object reaches that bin) — `ctr_border_count + 1` bounds both since CTR bins
    // live in `[0, ctr_border_count]`. Extra empty upper bins contribute `0.0` and
    // are inert (they only widen the scan, never change a sum — Pitfall 4).
    let float_max = (0..n_float)
        .map(|f| matrix.feature_borders.get(f).map_or(0, Vec::len) + 1)
        .max()
        .unwrap_or(0);
    let ctr_actual = ctr_features
        .iter()
        .flat_map(|c| c.bins.iter().copied())
        .max()
        .map_or(0usize, |m| m as usize + 1);
    let ctr_max = if n_ctr > 0 {
        ctr_actual.max(ctr_border_count + 1)
    } else {
        0
    };
    let n_bins = float_max.max(ctr_max).max(1);

    // Feature-major quantized bins: float columns via `bin_of`, then the CTR
    // columns' integer bins fed DIRECTLY (bin == ctr bin value).
    let mut bins = vec![0u32; n_features.saturating_mul(n_objects)];
    for f in 0..n_float {
        let borders = matrix.feature_borders.get(f).map_or(&[][..], Vec::as_slice);
        let col = matrix.feature_values.get(f).map_or(&[][..], Vec::as_slice);
        for obj in 0..n_objects {
            let v = col.get(obj).copied().unwrap_or(0.0);
            if let Some(slot) = bins.get_mut(f * n_objects + obj) {
                *slot = bin_of(borders, v) as u32;
            }
        }
    }
    for c in 0..n_ctr {
        let ctr_bins = ctr_features.get(c).map_or(&[][..], |col| col.bins.as_slice());
        let f = n_float + c;
        for obj in 0..n_objects {
            let b = ctr_bins.get(obj).copied().unwrap_or(0);
            if let Some(slot) = bins.get_mut(f * n_objects + obj) {
                *slot = b;
            }
        }
    }

    // The current partition (chosen splits), forward-bit leaf order — the SAME
    // partition `assign_leaves_ctr_aware` yields, so the scan's canonical
    // `[false, true]` leaf order matches the pre-rewrite rescan exactly.
    let leaf_of = assign_leaves_ctr_aware(matrix, ctr_features, chosen, n_objects);
    let n_leaves = 1usize << chosen.len();
    build_bucket_histogram(
        &bins, der1, weight, &leaf_of, n_leaves, n_features, n_bins, approx_dim,
    )
}

/// Score one CTR-aware candidate from the PRE-BUILT per-level histogram (the
/// histogram-backed replacement for the per-candidate `assign_leaves_ctr_aware` +
/// `reduce_leaf_stats` rescan). The candidate is identified by its combined-histogram
/// `hist_feature` (float feature `f`, or `n_float + col` for a CTR column) and
/// `border_index`; an `O(n_bins)` prefix scan
/// ([`scan_border_to_leaf_stats`]) yields the split's `[false, true]` child
/// [`LeafStats`] — `bins <= border` FALSE, `bins > border` TRUE, byte-for-byte the
/// `reduce_leaf_stats(assign_leaves_ctr_aware(chosen ++ candidate))` leaf order —
/// fed to the UNCHANGED [`split_score`] (`approx_dim == 1`, the same scalar calcer
/// the float path uses; NOT a forked scorer).
fn score_candidate_ctr_aware(
    hist: &BucketHistogram,
    hist_feature: usize,
    border_index: usize,
    scaled_l2: f64,
    score_function: EScoreFunction,
) -> f64 {
    let scan = scan_border_to_leaf_stats(hist, hist_feature, border_index, 1);
    let stats = scan.first().map_or(&[][..], Vec::as_slice);
    split_score(score_function, stats, scaled_l2)
}

/// One level of the CTR-aware UNPERTURBED search: enumerate FLOAT candidates
/// (feature asc, border asc; `AddFloatFeatures`) THEN CTR candidates (column asc,
/// border asc; `AddTreeCtrs`) — one CTR candidate per border `0..ctr_border_count`
/// for each column — score each via the SHARED L2 calcer, and pick the strict
/// first-wins best over that FIXED float-then-CTR order
/// ([`select_best_candidate`] discipline, strict `>`, Pitfall 1). The winning
/// candidate's concrete [`CtrAwareSplit`] is recovered from a vector kept in
/// lockstep with the scores.
/// The `model_size_reg` cat-feature weight for a NEW CTR projection
/// (`GetCatFeatureWeight`, greedy_tensor_search.cpp:925-928):
/// `(1 + count / maxCount)^(-model_size_reg)`. With the default `model_size_reg =
/// 0.5` a projection whose distinct-bucket `count` equals `maxCount` is weighted
/// `2^-0.5 ≈ 0.707`, while a low-cardinality projection is weighted nearer `1.0`.
/// `model_size_reg == 0` ⇒ weight `1.0` (no penalty).
#[must_use]
fn cat_feature_weight(count: usize, max_count: usize, model_size_reg: f64) -> f64 {
    if model_size_reg == 0.0 || max_count == 0 {
        return 1.0;
    }
    let ratio = count as f64 / max_count as f64;
    (1.0 + ratio).powf(-model_size_reg)
}

/// The MULTIPLICATIVE float-feature gain weight (`GetSplitFeatureWeight`,
/// greedy_tensor_search.cpp:980-988; the float-feature analogue of
/// [`cat_feature_weight`], FEAT-04). A split on float feature `feature` scales its
/// candidate gain by `feature_weights[feature]`. An out-of-range index (including
/// the EMPTY default vector) falls back to `1.0` — a no-op multiplier that leaves
/// the candidate score byte-identical to the pre-6.6 path (D-6.6-05). Access via
/// `.get(..).copied()` (no `indexing_slicing`, no panic — T-06.6-01).
#[must_use]
pub fn feature_weight(feature: usize, feature_weights: &[f64]) -> f64 {
    feature_weights.get(feature).copied().unwrap_or(1.0)
}

/// The FEAT-04 feature-penalty context threaded through the plain oblivious search
/// (`SelectBestCandidate` multiplicative weight + the `PenalizeBestSplits`
/// subtractive pass, `feature_penalties_calcer.cpp:191-205`). A `None` context (or
/// one whose vectors are all empty) leaves every candidate score numerically
/// untouched, so the default oblivious path is byte-identical (D-6.6-05). Each
/// vector is indexed by FLOAT feature id via `.get(..)` (no panic — T-06.6-01).
pub struct FeaturePenalties<'a> {
    /// MULTIPLICATIVE per-feature gain weights (`feature_weights`); empty ⇒ `1.0`
    /// per feature.
    pub feature_weights: &'a [f64],
    /// SUBTRACTIVE per-feature first-use penalties (`first_feature_use_penalties`);
    /// applied (× `penalties_coefficient`) only while the feature is unused. Empty
    /// ⇒ `0.0`.
    pub first_feature_use_penalties: &'a [f64],
    /// SUBTRACTIVE per-feature per-object penalties (`per_object_feature_penalties`);
    /// applied (× `penalties_coefficient` × `unused_doc_count`) only while the
    /// feature is globally unused on the symmetric path. Empty ⇒ `0.0`.
    pub per_object_feature_penalties: &'a [f64],
    /// The shared penalty scaling coefficient (`penalties_coefficient`).
    pub penalties_coefficient: f64,
    /// Per-FLOAT-feature "already used anywhere in the model built so far" flags
    /// (`used_features[f] == true` ⇒ feature `f` has been split on by a prior tree
    /// in this boosting run). Drives the "only when unused" gate of BOTH subtractive
    /// penalties (`feature_penalties_calcer.cpp` — the penalty fires until the
    /// feature first becomes used). An out-of-range index reads as unused (`false`).
    pub used_features: &'a [bool],
    /// The whole-fold object count used as the per-object penalty's `unused_doc_count`
    /// on the symmetric path (RESEARCH Pitfall 6 — the per-object term counts EVERY
    /// document when the feature is globally unused).
    pub doc_count: usize,
}

impl FeaturePenalties<'_> {
    /// `true` when every penalty knob is at its no-op default (all vectors empty),
    /// so the search can skip the penalty arithmetic and stay byte-identical to the
    /// pre-6.6 path (D-6.6-05 — gate on the param being non-empty, mirroring the
    /// `has_ctr` discriminator discipline).
    #[must_use]
    fn is_noop(&self) -> bool {
        self.feature_weights.is_empty()
            && self.first_feature_use_penalties.is_empty()
            && self.per_object_feature_penalties.is_empty()
    }

    /// Apply the FEAT-04 penalties to one candidate's `raw_score` for float feature
    /// `feature`, returning the penalized score. Transcribes the two upstream
    /// insertion points (RESEARCH "Penalty application"):
    ///
    /// 1. MULTIPLICATIVE on the gain (`gain *= GetSplitFeatureWeight(...)`,
    ///    greedy_tensor_search.cpp:980-988). With `scoreBeforeSplit` constant across
    ///    a level and no other feature weights, the candidate `raw_score` is the
    ///    pre-gain quantity the strict-`>` argmax ranks; scaling it by the
    ///    feature weight matches the upstream gain scaling.
    /// 2. SUBTRACTIVE first-use + per-object (`score -= ... × penalties_coefficient`,
    ///    feature_penalties_calcer.cpp:191-205), each fired ONLY while the feature is
    ///    unused so far. The per-object term multiplies by the whole-fold
    ///    `doc_count` (symmetric path, RESEARCH Pitfall 6).
    ///
    /// All sums route through [`cb_core::sum_f64`] (D-08); the per-object product is
    /// a single multiply (no reduction). Empty vectors ⇒ identity (`1.0` / `0.0`).
    #[must_use]
    fn penalize(&self, feature: usize, raw_score: f64) -> f64 {
        // (1) Multiplicative feature weight on the (pre-gain) candidate score.
        let weighted = raw_score * feature_weight(feature, self.feature_weights);

        // (2) Subtractive penalties — only while the feature is globally unused.
        let already_used = self.used_features.get(feature).copied().unwrap_or(false);
        if already_used {
            return weighted;
        }
        let first_use = self
            .first_feature_use_penalties
            .get(feature)
            .copied()
            .unwrap_or(0.0);
        let per_object = self
            .per_object_feature_penalties
            .get(feature)
            .copied()
            .unwrap_or(0.0);
        // first_use * coeff  +  per_object * coeff * unused_doc_count.
        let first_use_term = first_use * self.penalties_coefficient;
        let per_object_term =
            per_object * self.penalties_coefficient * self.doc_count as f64;
        // Deterministic-order accumulation of the subtracted terms (D-08).
        let subtracted = cb_core::sum_f64(&[first_use_term, per_object_term]);
        weighted - subtracted
    }
}

#[allow(clippy::too_many_arguments)]
fn select_level_ctr_aware(
    matrix: &FeatureMatrix,
    ctr_features: &[crate::ctr::CtrFeatureColumn],
    ctr_border_count: usize,
    chosen: &[CtrAwareSplit],
    der1: &[f64],
    weight: &[f64],
    scaled_l2: f64,
    n_objects: usize,
    model_size_reg: f64,
    score_function: EScoreFunction,
) -> CbResult<CtrAwareSplit> {
    let mut scored: Vec<(CtrAwareSplit, f64)> = Vec::new();

    // Build the ONE per-level combined float+CTR histogram over the current
    // partition; every candidate below is then scored by an `O(n_bins)` prefix
    // scan of it (PERF-02 — no per-candidate dataset rescan). The candidate
    // ENUMERATION ORDER, the `cat_feature_weight` insertion point, and the strict
    // `>` first-wins tie-break are byte-for-byte unchanged.
    let n_float = matrix.n_features();
    let hist = build_ctr_aware_histogram(
        matrix,
        ctr_features,
        ctr_border_count,
        chosen,
        der1,
        weight,
        n_objects,
    );

    // FLOAT candidates first (AddFloatFeatures), feature asc / border asc.
    for feature in 0..n_float {
        let borders = matrix.feature_borders.get(feature).map_or(&[][..], Vec::as_slice);
        for (border_index, &border) in borders.iter().enumerate() {
            let split = CtrAwareSplit::Float(Split { feature, border });
            let score = score_candidate_ctr_aware(
                &hist, feature, border_index, scaled_l2, score_function,
            );
            scored.push((split, score));
        }
    }

    // The `model_size_reg` cat-feature-weight penalty inputs (GetCatFeatureWeight,
    // greedy_tensor_search.cpp:908-932 + CalcMaxFeatureValueCount:1070-1088):
    //   * `max_bucket_count` = max distinct-bucket count over ALL CTR candidate
    //     columns (the candidates this level scores).
    //   * a projection ALREADY split in this tree (`chosen`) is exempt (weight 1.0)
    //     — the penalty only down-weights NEW projections, so a second border on an
    //     already-used simple CTR is never penalized while a new combination CTR is.
    let max_bucket_count = ctr_features
        .iter()
        .map(|c| c.bucket_count)
        .max()
        .unwrap_or(1)
        .max(1);
    let used_projections: Vec<&crate::TProjection> = chosen
        .iter()
        .filter_map(|s| match s {
            CtrAwareSplit::Ctr { col, .. } => {
                ctr_features.get(*col).map(|c| &c.projection)
            }
            CtrAwareSplit::Float(_) => None,
        })
        .collect();

    // CTR candidates next (AddTreeCtrs), column asc / border asc. One candidate per
    // CTR-value border in `0..ctr_border_count`; the `ctr_bin > border` test
    // borders are the integer bucket thresholds the materialized `bins` are
    // quantized into (a border `b` ⇔ bucket > `b`, i.e. bucket ≥ `b + 1`).
    for col in 0..ctr_features.len() {
        // The cat-feature weight for this column's projection (1.0 if already used,
        // else (1 + count/maxCount)^(-model_size_reg)).
        let cat_weight = match ctr_features.get(col) {
            Some(column) => {
                let already_used = used_projections.iter().any(|p| **p == column.projection);
                if already_used {
                    1.0
                } else {
                    cat_feature_weight(column.bucket_count, max_bucket_count, model_size_reg)
                }
            }
            None => 1.0,
        };
        // The CTR column occupies combined-histogram feature index `n_float + col`;
        // border `border_idx` scans the `ctr_bin > border_idx` boundary directly.
        let hist_feature = n_float + col;
        for border_idx in 0..ctr_border_count {
            let border = border_idx as f64;
            let split = CtrAwareSplit::Ctr { col, border };
            let score = cat_weight
                * score_candidate_ctr_aware(
                    &hist, hist_feature, border_idx, scaled_l2, score_function,
                );
            scored.push((split, score));
        }
    }

    // Strict first-wins over the FIXED float-then-CTR enumeration order, identical
    // to `select_best_candidate` (strict `>`, NOT `>=`; Pitfall 1).
    let mut best: Option<CtrAwareSplit> = None;
    let mut best_score = MINIMAL_SCORE;
    for &(split, score) in &scored {
        if score > best_score {
            best_score = score;
            best = Some(split);
        }
    }
    best.ok_or_else(|| {
        CbError::Degenerate(
            "no candidate split available (no float border and no CTR candidate)".to_owned(),
        )
    })
}

/// Grow one oblivious tree of depth `depth` over the FLOAT + CTR candidate set
/// (ORD-05 / D-05, the STRUCTURE half), with the strict first-wins greedy search.
///
/// At each level `0..depth`, both the float candidates (`select_level_plain`
/// order, unchanged) and — for each materialized [`crate::ctr::CtrFeatureColumn`]
/// — one candidate per CTR border `0..ctr_border_count` (the `ctr_bin > border`
/// test) are scored with the SHARED [`l2_split_score`]/[`reduce_leaf_stats`], and
/// the strict first-wins (`> best`) winner over the FIXED FLOAT-then-CTR order is
/// chosen. A winning CTR split is recorded as a [`CtrSplitSpec`] (carrying the
/// chosen CTR-value border + the column's prior num/denom PAIR + projection +
/// ctr_type); `GrownTree.level_kinds` records each level's kind so the forward-bit
/// leaf index assigns CTR bits and float bits in the correct level order.
///
/// `ctr_features` are the IDENTITY-learning-fold materialized CTR columns
/// (structure search); `target_border_idx` is the Buckets per-class numerator
/// selector carried onto each chosen `CtrSplitSpec` (default `0`).
///
/// IMPORTANT — this computes the STRUCTURE only. `grown.leaf_of` is the structure
/// partition; Plan 05-13 Task 2 REASSIGNS leaf_of over the averaging-fold CTR
/// column (boosting.rs) using `grown.ctr_splits` + `grown.level_kinds` for
/// LEAF-VALUE estimation. When `ctr_features` is empty this is the plain float
/// search (byte-identical structure to `greedy_tensor_search_oblivious`), with
/// empty `ctr_splits` / `level_kinds`.
///
/// # Errors
/// - [`CbError::DepthExceeded`] if `depth > MAX_DEPTH` (before allocation).
/// - [`CbError::Degenerate`] if a level has no candidate split at all.
#[allow(clippy::too_many_arguments)]
pub fn greedy_tensor_search_oblivious_with_ctr(
    matrix: &FeatureMatrix,
    ctr_features: &[crate::ctr::CtrFeatureColumn],
    ctr_border_count: usize,
    der1: &[f64],
    weight: &[f64],
    scaled_l2: f64,
    depth: usize,
    n_objects: usize,
    target_border_idx: usize,
    model_size_reg: f64,
    score_function: EScoreFunction,
) -> CbResult<GrownTree> {
    check_depth(depth)?;

    let mut chosen: Vec<CtrAwareSplit> = Vec::with_capacity(depth);
    for _level in 0..depth {
        let best = select_level_ctr_aware(
            matrix,
            ctr_features,
            ctr_border_count,
            &chosen,
            der1,
            weight,
            scaled_l2,
            n_objects,
            model_size_reg,
            score_function,
        )?;
        chosen.push(best);
    }

    let leaf_of = assign_leaves_ctr_aware(matrix, ctr_features, &chosen, n_objects);

    // Split the chosen unified splits back into the parallel `splits` / `ctr_splits`
    // vectors, recording each level's kind so the forward-bit leaf index (and Plan
    // 05-13 Task 2's averaging-fold reassignment) can rebuild the partition in
    // level order.
    let mut splits: Vec<Split> = Vec::new();
    let mut ctr_splits: Vec<CtrSplitSpec> = Vec::new();
    let mut level_kinds: Vec<LevelKind> = Vec::with_capacity(chosen.len());
    for split in &chosen {
        match split {
            CtrAwareSplit::Float(s) => {
                level_kinds.push(LevelKind::Float(splits.len()));
                splits.push(*s);
            }
            CtrAwareSplit::Ctr { col, border } => {
                // Recover the projection / prior PAIR / ctr_type from the winning
                // column; the chosen CTR-value border is the structure threshold
                // (Plan 05-14 reconciles Scale/Shift so apply compares in the same
                // space). A missing column index is a degenerate internal error.
                let column = ctr_features.get(*col).ok_or_else(|| {
                    CbError::Degenerate(
                        "ctr search: chosen CTR column index out of range".to_owned(),
                    )
                })?;
                level_kinds.push(LevelKind::Ctr {
                    ctr_idx: ctr_splits.len(),
                    border: *border,
                });
                ctr_splits.push(CtrSplitSpec {
                    projection: column.projection.clone(),
                    ctr_type: column.ctr_type,
                    prior_num: column.prior_num,
                    prior_denom: column.prior_denom,
                    target_border_idx,
                    border: *border,
                    // Default Shift/Scale at structure-search time; the train_cat
                    // bake (Plan 05-14) overwrites these on the chosen splits with
                    // the calc_normalization(prior_num)-derived (Shift, Scale).
                    shift: 0.0,
                    scale: 1.0,
                });
            }
        }
    }

    Ok(GrownTree {
        splits,
        leaf_of,
        ctr_splits,
        level_kinds,
        step_nodes: Vec::new(),
        node_id_to_leaf_id: Vec::new(),
        region_directions: Vec::new(),
        region_one_hot: Vec::new(),
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
    score_function: EScoreFunction,
) -> f64 {
    let mut splits = chosen.to_vec();
    splits.push(candidate);
    let n_leaves = 1usize << splits.len();
    let leaf_of = assign_leaves_any(matrix, &splits, n_objects);
    let stats: Vec<LeafStats> = reduce_leaf_stats(&leaf_of, der1, weight, n_leaves);
    split_score(score_function, &stats, scaled_l2)
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
    score_function: EScoreFunction,
) -> CbResult<AnySplit> {
    let mut scored: Vec<(AnySplit, f64)> = Vec::new();

    // FLOAT candidates first (AddFloatFeatures), feature asc / border asc.
    for feature in 0..matrix.n_features() {
        let borders = matrix.feature_borders.get(feature).map_or(&[][..], Vec::as_slice);
        for &border in borders {
            let split = AnySplit::Float(Split { feature, border });
            let score = score_candidate_any(
                matrix, chosen, split, der1, weight, scaled_l2, n_objects, score_function,
            );
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
            let score = score_candidate_any(
                matrix, chosen, split, der1, weight, scaled_l2, n_objects, score_function,
            );
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
    score_function: EScoreFunction,
) -> CbResult<GrownOneHotTree> {
    check_depth(depth)?;
    let mut chosen: Vec<AnySplit> = Vec::with_capacity(depth);
    for _level in 0..depth {
        let best = select_level_one_hot(
            matrix, &chosen, der1, weight, scaled_l2, n_objects, score_function,
        )?;
        chosen.push(best);
    }
    let leaf_of = assign_leaves_any(matrix, &chosen, n_objects);
    Ok(GrownOneHotTree {
        splits: chosen,
        leaf_of,
    })
}

// ===========================================================================
// LOSS-04 pairwise split-scoring search (the SPLIT-SELECTION half for
// `*Pairwise` losses — `IsPairwiseScoring`)
// ===========================================================================
//
// # Why a separate split path
//
// `*Pairwise` losses (`PairLogitPairwise`, `YetiRankPairwise`) score candidate
// splits through upstream's dedicated `TPairwiseScoreCalcer` /
// `CalculatePairwiseScore` (`pairwise_scoring.cpp`), NOT the pointwise
// L2/Cosine der histogram the float `select_level_plain` path reuses
// (`greedy_tensor_search.cpp:680-690`:
// `if (IsPairwiseScoring(loss)) scoreCalcer.Reset(new TPairwiseScoreCalcer)`).
// The 06.3-13/14 verification isolated this as the SPLIT divergence cause for
// PairLogitPairwise (upstream f0@1.628 vs cb-train f1@1.816 at tree-0 split 1).
//
// Plan 06.3-15 built the scorer primitive in `cb-compute::pairwise_scoring`
// (`compute_der_sums` + `compute_pair_weight_statistics` +
// `calculate_pairwise_score`); this path WIRES it into the greedy oblivious
// level search. Per candidate float feature: bucket the docs by that feature's
// borders, build the `[leaf][bucket]` der sums + `[leaf][leaf][bucket]`
// pair-weight statistics over the CURRENT leaf assignment and the per-tree
// global competitor pairs, call `calculate_pairwise_score` (one score per
// border), and emit one [`Candidate`] per border. The strict first-wins
// [`select_best_candidate`] tie-break is reused VERBATIM (feature asc, border
// asc; Pitfall 1).
//
// # Regularization
//
// `l2_diag_reg = params.l2_leaf_reg` is the RAW L2
// (`scoring.cpp:809,844`: `l2Regularizer = ObliviousTreeOptions->L2Reg`, passed
// UNSCALED to `CalculatePairwiseScore` — the same RAW l2 the pairwise LEAF path
// uses, NOT the `sumAllWeights`-scaled `scaled_l2` the pointwise path uses).
// `pairwise_bucket_weight_prior_reg = bayesian_matrix_reg` default `0.1`
// (`oblivious_tree_options.cpp:16` `PairwiseNonDiagReg`).

/// The bucket index of a float `value` against ascending `borders`: the number
/// of borders the value is strictly greater than (`value > border`), mirroring
/// upstream's float-feature bin (`bucketCount = borders.len() + 1`,
/// `pairwise_scoring.h:169`). A doc with `bucket > splitId` passes the candidate
/// split at border index `splitId` (consistent with [`FeatureMatrix::passes_float`]).
#[must_use]
fn float_bucket_of(value: f32, borders: &[f64]) -> usize {
    let v = f64::from(value);
    borders.iter().filter(|&&b| v > b).count()
}

/// Flatten the per-tree group competitor adjacency into the GLOBAL
/// `(winner_global, loser_global, weight)` pair list the cb-compute pairwise
/// scorer consumes (`TFlatPairsInfo`). `competitors[winner_local]` lists the
/// losers `winner_local` is preferred over; the group-local indices are lifted
/// to global object indices by adding the group's `begin` offset — the SAME
/// lift `pairwise_leaves::compute_pairwise_weight_sums` performs (group asc,
/// winner asc, competitor order — the parity contract).
#[must_use]
fn flatten_global_pairs(groups: &[GroupSpan]) -> Vec<(usize, usize, f64)> {
    let mut pairs: Vec<(usize, usize, f64)> = Vec::new();
    for group in groups {
        let begin = group.begin;
        for (winner_local, comps) in group.competitors.iter().enumerate() {
            let winner_global = begin + winner_local;
            for competitor in comps {
                let loser_global = begin + competitor.id;
                pairs.push((winner_global, loser_global, competitor.weight));
            }
        }
    }
    pairs
}

/// Score every candidate (feature, border) for ONE level via the pairwise
/// scorer and return the per-border scores keyed by feature
/// (`CalculatePairwiseScore`, `pairwise_scoring.cpp:140-232`, `OneFeature`).
///
/// For each candidate float feature, bucket the docs against the feature's
/// borders (`bucket_count = borders.len() + 1`), build the `[leaf][bucket]`
/// der sums (over the SAME weighted der1 the pairwise leaf path uses) and the
/// `[leaf][leaf][bucket]` pair-weight statistics (over the global pairs), then
/// `calculate_pairwise_score` returns `bucket_count - 1` scores (one per border).
#[allow(clippy::too_many_arguments)]
fn score_pairwise_feature(
    matrix: &FeatureMatrix,
    feature: usize,
    leaf_of: &[usize],
    leaf_count: usize,
    der1: &[f64],
    global_pairs: &[(usize, usize, f64)],
    l2_diag_reg: f64,
    pairwise_bucket_weight_prior_reg: f64,
    n_objects: usize,
) -> CbResult<Vec<f64>> {
    let borders = matrix.feature_borders.get(feature).map_or(&[][..], Vec::as_slice);
    let bucket_count = borders.len() + 1;
    let values = matrix.feature_values.get(feature).map_or(&[][..], Vec::as_slice);

    let bucket_of: Vec<usize> = (0..n_objects)
        .map(|obj| {
            let v = values.get(obj).copied().unwrap_or(0.0_f32);
            float_bucket_of(v, borders)
        })
        .collect();

    let der_sums =
        compute_pairwise_der_sums(der1, leaf_count, bucket_count, leaf_of, &bucket_of)?;
    let pair_weight_statistics = compute_pair_weight_statistics(
        global_pairs,
        leaf_count,
        bucket_count,
        leaf_of,
        &bucket_of,
    )?;
    calculate_pairwise_score(
        &der_sums,
        &pair_weight_statistics,
        bucket_count,
        l2_diag_reg,
        pairwise_bucket_weight_prior_reg,
    )
}

/// One level of the PAIRWISE search: enumerate candidates in the SAME upstream
/// order as [`select_level_plain`] (float feature ascending, border ascending),
/// score each border via the pairwise scorer ([`score_pairwise_feature`]), and
/// pick the strict first-wins best ([`select_best_candidate`], strict `>`;
/// Pitfall 1). The `*Pairwise` corpus is float-only and `boosting_type = Plain`
/// (`IsPlainOnlyModeLoss`), so there are no ordered / CTR / perturbation draws.
#[allow(clippy::too_many_arguments)]
fn select_level_pairwise(
    matrix: &FeatureMatrix,
    chosen: &[Split],
    leaf_of: &[usize],
    der1: &[f64],
    global_pairs: &[(usize, usize, f64)],
    l2_diag_reg: f64,
    pairwise_bucket_weight_prior_reg: f64,
    n_objects: usize,
) -> CbResult<Split> {
    let leaf_count = 1usize << chosen.len();
    let mut candidates: Vec<Candidate> = Vec::new();
    for feature in 0..matrix.n_features() {
        let borders = matrix.feature_borders.get(feature).map_or(&[][..], Vec::as_slice);
        if borders.is_empty() {
            continue;
        }
        let scores = score_pairwise_feature(
            matrix,
            feature,
            leaf_of,
            leaf_count,
            der1,
            global_pairs,
            l2_diag_reg,
            pairwise_bucket_weight_prior_reg,
            n_objects,
        )?;
        // scores[j] is the score of splitting at border index `j`; enumerate in
        // border-ascending order so the strict first-wins tie-break matches
        // upstream's candidate iteration (feature asc, border asc).
        for (border_idx, &border) in borders.iter().enumerate() {
            let score = scores.get(border_idx).copied().unwrap_or(MINIMAL_SCORE);
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

/// Grow one oblivious tree of depth `depth` with the PAIRWISE split-scoring
/// subsystem (the SPLIT-SELECTION half of LOSS-04 for `*Pairwise` losses). Per
/// level `0..depth`, each candidate (feature, border) is scored via the
/// cb-compute pairwise scorer over the CURRENT leaf assignment + the per-tree
/// global competitor pairs, then the strict first-wins best is chosen
/// ([`select_best_candidate`], `>` not `>=`). After `depth` levels `leaf_of` is
/// assigned over the object order (forward-bit [`leaf_index`]) exactly like the
/// plain path, so the downstream pairwise leaf-value estimation
/// (`pairwise_leaves.rs`) runs over the same partition.
///
/// `der1` is the per-object pairwise weighted der1 (the SAME buffer fed to the
/// pairwise leaf path); `groups` carries the per-tree competitor adjacency.
/// `l2_diag_reg = params.l2_leaf_reg` (RAW, NOT `scaled_l2`);
/// `pairwise_bucket_weight_prior_reg = bayesian_matrix_reg` default `0.1`.
///
/// # Errors
/// - [`CbError::DepthExceeded`] if `depth > MAX_DEPTH` (before allocation).
/// - [`CbError::Degenerate`] if a level has no candidate split at all.
/// - [`CbError::OutOfRange`] if a competitor/leaf/bucket index from the trainer
///   trust boundary is out of range (propagated from the cb-compute scorer).
#[allow(clippy::too_many_arguments)]
pub fn greedy_tensor_search_oblivious_pairwise(
    matrix: &FeatureMatrix,
    der1: &[f64],
    groups: &[GroupSpan],
    l2_diag_reg: f64,
    pairwise_bucket_weight_prior_reg: f64,
    depth: usize,
    n_objects: usize,
) -> CbResult<GrownTree> {
    check_depth(depth)?;

    let global_pairs = flatten_global_pairs(groups);
    let mut chosen: Vec<Split> = Vec::with_capacity(depth);

    for _level in 0..depth {
        // The current leaf assignment over the already-chosen splits (forward-bit
        // leaf index); leaf_count = 2^chosen.len().
        let leaf_of = assign_leaves(matrix, &chosen, n_objects);
        let best = select_level_pairwise(
            matrix,
            &chosen,
            &leaf_of,
            der1,
            &global_pairs,
            l2_diag_reg,
            pairwise_bucket_weight_prior_reg,
            n_objects,
        )?;
        chosen.push(best);
    }

    let leaf_of = assign_leaves(matrix, &chosen, n_objects);
    Ok(GrownTree {
        splits: chosen,
        leaf_of,
        ctr_splits: Vec::new(),
        level_kinds: Vec::new(),
        step_nodes: Vec::new(),
        node_id_to_leaf_id: Vec::new(),
        region_directions: Vec::new(),
        region_one_hot: Vec::new(),
    })
}

#[cfg(test)]
#[path = "region_grow_test.rs"]
mod region_grow_test;

#[cfg(test)]
#[path = "leaf_wise_scorer_test.rs"]
mod leaf_wise_scorer;
