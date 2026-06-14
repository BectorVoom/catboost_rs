//! `TFold` body/tail prefix state machine and multi-permutation fold creation
//! (ORD-01) — the anti-leakage bookkeeping ordered boosting and ordered CTR are
//! computed over. Transcribed from upstream catboost 1.2.10.
//!
//! # Source of truth
//!
//! `catboost/private/libs/algo/fold.cpp:35-41` (`SelectMinBatchSize`,
//! `SelectTailSize`), `fold.cpp:148-198` (`BuildDynamicFold` growing body/tail
//! loop), `fold.cpp:222-285` (`BuildPlainFold` single-span path), and
//! `catboost/private/libs/algo/learn_context.cpp:38-90`
//! (`IsPermutationNeeded` / `CountLearningFolds` / fold creation).
//!
//! # The two fold shapes
//!
//! - **Dynamic (ordered)** — [`body_tail_boundaries`]: a sequence of growing
//!   `(body_finish, tail_finish)` segments. A tail document's approximant is
//!   estimated on the BODY prefix only and never depends on itself
//!   (`approx_calcer.cpp:566-600`), so the prefix boundaries are the linchpin
//!   the per-object ordered-approx oracle catches off-by-ones in. The boundary
//!   math is exactly: `leftPartLen` starts at `SelectMinBatchSize(n)`; each
//!   segment has `bodyFinish = leftPartLen`, `tailFinish =
//!   min(ceil(leftPartLen * fold_len_multiplier), n)`; `leftPartLen =
//!   tailFinish` until `leftPartLen >= n` (`fold.cpp:148-198`).
//! - **Plain** — [`plain_fold_body_tail`]: a SINGLE body/tail spanning the whole
//!   fold (`bodyFinish == tailFinish == n`, `fold.cpp:268-274`). This is the
//!   path the one-hot / Plain slices (05-02) ride; it is kept intact here.
//!
//! # Fold count
//!
//! [`learning_fold_count`] reproduces `CountLearningFolds = max(1,
//! permutation_count - 1)` when a learning permutation is needed
//! (`learn_context.cpp:48-49`); [`create_folds`] builds those learning folds
//! PLUS one averaging fold (`learn_context.cpp` fold-creation loop), each with
//! its own permutation drawn IN ORDER from a single persistent RNG (mirroring
//! the bootstrap.rs continuous-stream discipline). E.g. `permutation_count = 2`
//! → 1 learning fold + 1 averaging fold (RESEARCH Open Q2).
//!
//! # Parity discipline
//!
//! `permutation_count` and `fold_len_multiplier` are pinned EXPLICITLY on
//! [`crate::BoostParams`] (never auto-selected, RESEARCH Pitfall 6; defaults
//! `permutation_count = 4`, `fold_len_multiplier = 2.0`). All prefix arithmetic
//! is checked (`usize::try_from` / `checked_*` / capped at `n`; the growth loop
//! is strictly monotone so it terminates — T-05-03-01). Any float sum routes
//! through `cb_core::sum_f64` (D-08). No `unwrap`/`expect`/panic/raw index, no
//! `anyhow`.

use cb_core::{sum_f64, TFastRng64};

use crate::permutation::{permutations, shuffle_in_place};

// Tests live in a dedicated sibling file (source/test separation, CLAUDE.md /
// AGENTS.md), mounted as a child module so `cargo test -p cb-train fold` and
// `... fold_prefix` select them.
#[cfg(test)]
#[path = "fold_test.rs"]
mod tests;

/// `SelectMinBatchSize` (`fold.cpp:35-37`): the initial body prefix length.
/// `learn_sample_count > 500 ? min(100, n / 50) : 1`.
#[must_use]
pub fn select_min_batch_size(learn_sample_count: usize) -> usize {
    if learn_sample_count > 500 {
        usize::min(100, learn_sample_count / 50)
    } else {
        1
    }
}

/// `SelectTailSize` (`fold.cpp:39-41`): `ceil(old_size * multiplier)`. The
/// multiplier is `fold_len_multiplier` (default `2.0`). Computed in `f64` to
/// match upstream's `ceil(double)`, then narrowed back to a doc count. A
/// non-finite or negative product (a degenerate `multiplier`) clamps to `0` so
/// the caller's `min(_, n)` keeps the result a valid in-`[0, n]` count rather
/// than panicking on the cast.
#[must_use]
pub fn select_tail_size(old_size: usize, multiplier: f64) -> usize {
    let product = (old_size as f64) * multiplier;
    let ceiled = product.ceil();
    if ceiled.is_finite() && ceiled >= 0.0 {
        // `ceiled` is a non-negative finite integer-valued f64; the `as usize`
        // cast saturates large values (defensive — real fold sizes are tiny).
        ceiled as usize
    } else {
        0
    }
}

/// The dynamic (ordered) fold body/tail boundary sequence
/// (`fold.cpp:148-198`), ungrouped (no query/group structure — the in-scope
/// object-order path).
///
/// Returns the `leftPartLen` sequence: the initial `SelectMinBatchSize(n)`
/// followed by each segment's `tailFinish`, i.e. exactly the committed
/// `body_tail_boundaries.npy` schema. For `n = 30`, `multiplier = 2.0` this is
/// `[1, 2, 4, 8, 16, 30]` (initial `1`; tails `2, 4, 8, 16`, then `ceil(16*2) =
/// 32` capped at `30`). The final entry always equals `n` (the last segment's
/// tail is capped at `n`, terminating the growth).
///
/// `n == 0` returns an empty sequence; `n == 1` returns `[1]` (a single segment
/// body=tail=1 — `SelectMinBatchSize(1) = 1 == n`, so the growth loop runs once
/// and stops). The loop is strictly monotone (each `tailFinish > leftPartLen`
/// until the `n` cap), so it always terminates (T-05-03-01).
#[must_use]
pub fn body_tail_boundaries(n: usize, multiplier: f64) -> Vec<usize> {
    if n == 0 {
        return Vec::new();
    }
    // `leftPartLen = UpdateSize(SelectMinBatchSize(n), …)`; ungrouped UpdateSize
    // is just `min(size, n)`.
    let mut left_part_len = usize::min(select_min_batch_size(n), n);
    let mut boundaries = vec![left_part_len];
    // `while (BodyTailArr.empty() || leftPartLen < n)` — at least one segment,
    // then grow until the body prefix covers the whole fold.
    while left_part_len < n {
        // tailFinish = min(SelectTailSize(leftPartLen, mult), n).
        let tail_finish = usize::min(select_tail_size(left_part_len, multiplier), n);
        // Defensive monotonicity guard: a degenerate `multiplier <= 1.0` could
        // fail to grow the prefix; force progress to `n` so the loop always
        // terminates (upstream's real multiplier is > 1, default 2.0).
        let tail_finish = if tail_finish <= left_part_len {
            n
        } else {
            tail_finish
        };
        boundaries.push(tail_finish);
        left_part_len = tail_finish;
    }
    boundaries
}

/// The dynamic fold's `(body_finish, tail_finish)` segment pairs
/// (`fold.cpp:157-174`), derived from [`body_tail_boundaries`]. Segment `s` has
/// `body_finish = boundaries[s]` and `tail_finish = boundaries[s + 1]`. For
/// `n = 30`, `multiplier = 2.0`: `[(1,2), (2,4), (4,8), (8,16), (16,30)]`.
#[must_use]
pub fn body_tail_segments(n: usize, multiplier: f64) -> Vec<(usize, usize)> {
    let boundaries = body_tail_boundaries(n, multiplier);
    boundaries
        .windows(2)
        .filter_map(|w| match w {
            [body, tail] => Some((*body, *tail)),
            _ => None,
        })
        .collect()
}

/// The PLAIN-boosting single body/tail (`BuildPlainFold`, `fold.cpp:268-274`):
/// one segment spanning the whole fold, `body_finish == tail_finish == n`. This
/// is the path the one-hot / Plain slices ride (no ordered prefixes). `n == 0`
/// yields `(0, 0)`.
#[must_use]
pub fn plain_fold_body_tail(n: usize) -> (usize, usize) {
    (n, n)
}

/// `CountLearningFolds` (`learn_context.cpp:48-49`): `max(1, permutation_count -
/// 1)` learning folds when a learning permutation is needed, else `1`.
///
/// `permutation_needed_for_learning` is upstream's `IsPermutationNeeded(hasTime,
/// hasCtrs, isOrderedBoosting, isAveragingFold=false)` for the learning folds
/// (`learn_context.cpp:38-46`): true when the data has CTRs (any cat feature
/// over `one_hot_max_size`) OR ordered boosting is on (and not a time-ordered
/// dataset). The caller supplies that decision; this function is the pure
/// fold-count arithmetic. `permutation_count == 0` still yields at least `1`
/// (the `max(1, …)` floor; `0 - 1` is guarded by `saturating_sub`).
#[must_use]
pub fn learning_fold_count(permutation_count: usize, permutation_needed_for_learning: bool) -> usize {
    if permutation_needed_for_learning {
        usize::max(1, permutation_count.saturating_sub(1))
    } else {
        1
    }
}

/// One created fold: the object permutation it is built over and its
/// body/tail boundary sequence. An averaging fold uses the PLAIN single-span
/// body/tail; a learning fold under ordered boosting uses the dynamic growing
/// body/tail.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fold {
    /// The object permutation (original index at each learn-order position).
    pub permutation: Vec<i32>,
    /// The `leftPartLen` boundary sequence (see [`body_tail_boundaries`]); for a
    /// plain/averaging fold this is `[n]` (the single full span).
    pub body_tail_boundaries: Vec<usize>,
    /// Whether this is the averaging fold (`true`) or a learning fold (`false`).
    pub is_averaging: bool,
}

/// Creates the full fold set for one training run: `learning_fold_count(...)`
/// learning folds PLUS one averaging fold (`learn_context.cpp` fold-creation
/// loop).
///
/// # Draw-order contract (the parity linchpin)
///
/// WHEN a learning permutation is needed (`permutation_needed_for_learning`,
/// i.e. `hasCtrs` OR ordered boosting), `Folds[0]` is the IDENTITY permutation
/// `[0,1,…,n-1]` consuming ZERO RNG draws — mirroring upstream's `shuffle =
/// foldIdx != 0` (`learn_context.cpp:524`, `fold.cpp:54`). Every SUBSEQUENT fold
/// (learning folds `1..learning_fold_count`, then the averaging fold) consumes
/// ONE Fisher-Yates draw IN ORDER from the SAME persistent
/// `TFastRng64::from_seed(random_seed)` (the upstream persistent
/// `TRestorableFastRng64` — never reseeded per fold). Because the lone learning
/// `Folds[0]` draws nothing, the averaging fold's shuffle is the FIRST seeded
/// draw: for `permutation_count = 1` the averaging-fold permutation byte-equals
/// `fisher_yates_permutation(n, random_seed)` index-for-index
/// (`IsAverageFoldPermuted = hasCtrs`, `learn_context.cpp:575-577`). Plan 05-13
/// pulls that averaging-fold permutation (via `find(|f| f.is_averaging)`) for the
/// leaf-value materialization.
///
/// WHEN a learning permutation is NOT needed (the numeric / Plain-no-CTR path),
/// the legacy continuous-stream draw order is preserved EXACTLY: every fold
/// (learning then averaging) draws one Fisher-Yates shuffle in order from the
/// persistent RNG, byte-identical to before this change (the numeric oracles
/// stay byte-for-byte).
///
/// `dynamic_body_tail` selects the learning folds' body/tail shape: `true` (the
/// ordered-boosting path) gives each learning fold the growing dynamic
/// body/tail; `false` (the plain path) gives every fold the single full span.
/// The AVERAGING fold always uses the plain single span (it is the
/// non-ordered, whole-dataset fold).
///
/// The total fold count is `learning_fold_count + 1`. `permutation_count = 2` →
/// 1 learning + 1 averaging fold (RESEARCH Open Q2).
#[must_use]
pub fn create_folds(
    n: usize,
    permutation_count: usize,
    permutation_needed_for_learning: bool,
    dynamic_body_tail: bool,
    fold_len_multiplier: f64,
    random_seed: u64,
) -> Vec<Fold> {
    let learning_folds = learning_fold_count(permutation_count, permutation_needed_for_learning);
    // One permutation per learning fold + one for the averaging fold.
    let total_folds = learning_folds.saturating_add(1);

    if !permutation_needed_for_learning {
        // NUMERIC / Plain-no-CTR path: preserve the legacy continuous-stream
        // draws (learning fold then averaging), byte-identical to before — there
        // is no CTR/ordered draw stream to align with upstream's identity-Folds[0]
        // rule, so the prior behavior is kept verbatim (regression anchor).
        let perms = permutations(n, total_folds, random_seed);
        return perms
            .into_iter()
            .enumerate()
            .map(|(idx, permutation)| {
                build_fold(idx, learning_folds, permutation, n, dynamic_body_tail, fold_len_multiplier)
            })
            .collect();
    }

    // LEARNING-PERMUTATION-NEEDED path (hasCtrs OR ordered boosting): drive a
    // SINGLE persistent rng directly. Fold 0 (idx == 0) is the IDENTITY
    // (zero draws, `shuffle = foldIdx != 0`); every subsequent fold takes one
    // Fisher-Yates draw from the SAME held rng IN ORDER. The persistent rng is
    // never reseeded per fold.
    //
    // PRE-AVERAGING DRAW (Plan 05-14): upstream advances its persistent
    // `LearnProgress->Rand` by exactly ONE `GenRand()` between the identity
    // learning Folds[0] and the AveragingFold's `Shuffle` (the averaging shuffle
    // starts at RNG call-count 1, NOT 0). Without this the averaging permutation
    // is `fisher_yates(n, seed)` (call-count 0) which yields the WRONG leaf-value
    // partition (`[6,0,11,13]` instead of the upstream `[6,0,7,17]` for the
    // tensor_ctr_e2e config). Advancing one draw reproduces upstream's
    // AveragingFold permutation and the leaf-weight partition `[6,0,7,17]` /
    // tree0 leaf values `[-0.0333,0,-0.005,0.0275]` bit-for-bit.
    let mut rng = TFastRng64::from_seed(random_seed);
    let mut first_real_shuffle = true;
    (0..total_folds)
        .map(|idx| {
            let permutation: Vec<i32> = if idx == 0 {
                // Identity Folds[0]: ZERO draws on the persistent rng.
                (0..n).map(|i| i as i32).collect()
            } else {
                if first_real_shuffle {
                    // One pre-averaging GenRand draw (upstream call-count 1).
                    rng.gen_rand();
                    first_real_shuffle = false;
                }
                shuffle_in_place(n, &mut rng)
            };
            build_fold(idx, learning_folds, permutation, n, dynamic_body_tail, fold_len_multiplier)
        })
        .collect()
}

/// Assembles one [`Fold`] from its already-drawn `permutation`, selecting the
/// body/tail shape: the averaging fold (`idx == learning_folds`) and the plain
/// path (`!dynamic_body_tail`) use the single full span `[n]`; a dynamic
/// learning fold uses the growing [`body_tail_boundaries`].
fn build_fold(
    idx: usize,
    learning_folds: usize,
    permutation: Vec<i32>,
    n: usize,
    dynamic_body_tail: bool,
    fold_len_multiplier: f64,
) -> Fold {
    let is_averaging = idx == learning_folds;
    let body_tail_boundaries = if is_averaging || !dynamic_body_tail {
        // Plain single span: [n] (body == tail == n).
        vec![n]
    } else {
        body_tail_boundaries(n, fold_len_multiplier)
    };
    Fold {
        permutation,
        body_tail_boundaries,
        is_averaging,
    }
}

/// The per-fold body-prefix summed weights (`fold.cpp:170-172`
/// `bodySumWeight`): for each dynamic segment, the sum of the first
/// `body_finish` learn weights. Routed through the sanctioned ordered
/// [`sum_f64`] (D-08). For unweighted training (`weights` empty) each segment's
/// body weight is its `body_finish` count (upstream's `? bodyFinish` branch).
///
/// Exposed so the ordered-approx slice can feed the exact body-weight
/// normalization without re-deriving the prefix math. Returns one weight per
/// dynamic segment.
#[must_use]
pub fn body_sum_weights(n: usize, multiplier: f64, weights: &[f64]) -> Vec<f64> {
    body_tail_segments(n, multiplier)
        .into_iter()
        .map(|(body_finish, _tail_finish)| {
            if weights.is_empty() {
                body_finish as f64
            } else {
                let prefix = weights.get(..body_finish).unwrap_or(weights);
                sum_f64(prefix)
            }
        })
        .collect()
}
