//! Per-fold object permutation (ORD-01, D-03 linchpin) — the modern
//! Fisher-Yates shuffle over the Phase-1 [`cb_core::TFastRng64`], reproducing
//! upstream catboost 1.2.10's EXACT permutation index-for-index.
//!
//! # Source of truth
//!
//! `util/random/shuffle.h:24-32` (the modern Fisher-Yates `Shuffle`),
//! `util/random/common_ops.h:48-91` (`GenUniform`/`Uniform`), and
//! `catboost/private/libs/algo/fold.cpp:53-96` (`InitPermutationData`:
//! ungrouped, block-size-1 path resolves to the plain per-object shuffle for
//! `N < 1000`).
//!
//! # Why this is the D-03 linchpin
//!
//! Both our Rust impl and upstream are seeded by the bit-exact `TFastRng64`
//! port, so the generated permutation is reproducible to the exact index — NOT
//! a `1e-5` value comparison. A single-index mismatch must be rejected BEFORE
//! any value stage (online CTR / ordered approx) is allowed to run: a prefix or
//! CTR value computed under the wrong order is meaningless. The
//! `permutation_oracle_test.rs` integration test asserts integer-exact equality
//! against the committed `permutation_fold{k}.npy` via
//! `cb_oracle::compare_permutation` (`Stage::Permutation`).
//!
//! # Exact draw order (the parity contract)
//!
//! Transcribed from `shuffle.h:28-30`:
//!
//! ```text
//! const size_t sz = end - begin;
//! for (size_t i = 1; i < sz; ++i) {
//!     DoSwap(*(begin + i), *(begin + gen.Uniform(i + 1)));
//! }
//! ```
//!
//! The array starts as the identity `[0, 1, …, n-1]`. For each `i` from `1` to
//! `n-1` (in that order), ONE draw `j = gen.Uniform(i + 1)` is taken on the
//! generator (range `[0, i]`, inclusive), then elements `i` and `j` are
//! swapped. Index `0` is never the active swap target (the loop starts at `1`),
//! matching upstream exactly. We MIRROR the bootstrap.rs draw-phase discipline
//! (one documented draw per loop step, in upstream order) — see `bootstrap.rs`
//! `PRE_TREE_DRAWS`.
//!
//! The generator is `TFastRng64::from_seed(seed)` where `seed` is the per-fold
//! permutation seed (the training `random_seed` for fold 0; the persistent RNG
//! advances across folds upstream — see [`permutations`]).
//!
//! # Block awareness (`fold_permutation_block`)
//!
//! Upstream's `DefaultFoldPermutationBlockSize = min(256, docCount/1000 + 1)`
//! (`defaults_helper.h`) is `1` for every `N < 1000` (RESEARCH Open Q3). The
//! ungrouped path in `InitPermutationData` (`fold.cpp:53-60`) forces
//! `PermutationBlockSize = 1` whenever the feature subset is non-consecutive,
//! and a trivial grouping with block size 1 is exactly the per-object shuffle
//! above. [`PERMUTATION_BLOCK_SIZE_THRESHOLD`] documents the `N < 1000`
//! boundary; this module implements the block-size-1 (per-object) shuffle that
//! every in-scope fixture (`N == 30`) exercises. Block sizes `> 1` (only
//! reachable at `N >= 1000`) shuffle whole contiguous blocks of `blockSize`
//! objects as a unit and are out of scope for this slice — [`fold_block_size`]
//! reports the upstream block size so a future slice can extend the shuffle.
//!
//! # Parity discipline
//!
//! All draws go through [`cb_core::TFastRng64::uniform`] ONLY (the bit-exact
//! `NPrivate::GenUniform` port — never re-port the RNG, never `rand`). Checked
//! access only: no `unwrap`/`expect`/panic/raw index, no `anyhow`. Permutation
//! indices are emitted as `Vec<i32>` to match the upstream `int32` `.npy`
//! schema (D-02); callers widen to `i64` for the integer-exact comparator.

use cb_core::TFastRng64;

// Tests live in a dedicated sibling file (source/test separation, CLAUDE.md /
// AGENTS.md — no test body in this production file), mounted as a child module
// so `cargo test -p cb-train permutation` selects them.
#[cfg(test)]
#[path = "permutation_test.rs"]
mod tests;

/// Document-count boundary below which the fold permutation block size is `1`
/// (the per-object shuffle): `DefaultFoldPermutationBlockSize =
/// min(256, docCount/1000 + 1)` (`defaults_helper.h`) equals `1` for every
/// `N < 1000` (RESEARCH Open Q3). Every in-scope Wave-0 fixture has `N == 30`,
/// well under this threshold.
pub const PERMUTATION_BLOCK_SIZE_THRESHOLD: usize = 1000;

/// The upstream default fold permutation block size for `doc_count` objects:
/// `min(256, doc_count / 1000 + 1)` (`defaults_helper.h`
/// `DefaultFoldPermutationBlockSize`). Returns `1` for every `N < 1000` (the
/// per-object shuffle this module implements). Exposed so a later slice can
/// detect when the block-aware (`block > 1`) path is required (`N >= 1000`).
#[must_use]
pub fn fold_block_size(doc_count: usize) -> usize {
    usize::min(256, doc_count / PERMUTATION_BLOCK_SIZE_THRESHOLD + 1)
}

/// Generates one fold's object permutation via the modern Fisher-Yates shuffle
/// (`shuffle.h:24-32`) over a `TFastRng64::from_seed(seed)`, block size 1
/// (per-object; the `N < 1000` path).
///
/// The result is the permutation `p` where `p[k]` is the original object index
/// placed at learn-order position `k` — identical to upstream's
/// `fold->LearnPermutation` index array and the committed
/// `permutation_fold{k}.npy` (integer-exact, `Stage::Permutation`, D-03).
///
/// Draw order (the parity contract, `shuffle.h:28-30`): identity init, then for
/// `i` in `1..n` ONE draw `j = rng.uniform(i + 1)` followed by `swap(i, j)`.
/// `n <= 1` returns the trivial identity with NO draws (the loop body never
/// runs), matching upstream.
#[must_use]
pub fn fisher_yates_permutation(n: usize, seed: u64) -> Vec<i32> {
    let mut rng = TFastRng64::from_seed(seed);
    shuffle_in_place(n, &mut rng)
}

/// The modern Fisher-Yates shuffle (`shuffle.h:28-30`) over an ALREADY-seeded
/// generator, so a caller driving a persistent multi-fold RNG can keep the
/// draw stream continuous across folds (see [`permutations`]). Identity init,
/// then for `i` in `1..n`: `j = rng.uniform(i + 1)`; `swap(i, j)`. Uses checked
/// `swap` semantics over a `Vec` (no raw index; `Vec::swap` is bounds-safe and
/// the indices `i`, `j` are both `< n` by construction — `j = uniform(i+1)` is
/// in `[0, i]`).
///
/// Exposed `pub(crate)` so [`crate::fold::create_folds`] can drive a single
/// persistent `TFastRng64` directly — emitting the IDENTITY for the lone
/// learning `Folds[0]` (ZERO draws, upstream's `shuffle = foldIdx != 0`) and
/// taking ONE shuffle for each subsequent fold from the SAME held rng. The
/// public `permutations` / `fisher_yates_permutation` API is unchanged.
pub(crate) fn shuffle_in_place(n: usize, rng: &mut TFastRng64) -> Vec<i32> {
    // Identity `[0, 1, …, n-1]`. `i32` matches the upstream `int32` index width
    // and the `.npy` schema (D-02); indices fit comfortably for in-scope N.
    let mut v: Vec<i32> = (0..n).map(|idx| idx as i32).collect();
    // shuffle.h:28 — `for (size_t i = 1; i < sz; ++i)`.
    for i in 1..n {
        // shuffle.h:29 — `gen.Uniform(i + 1)` draws ONE value in `[0, i]`.
        // `i + 1 >= 2 > 0`, so `uniform` never hits the degenerate zero-bound
        // case; the draw is exactly upstream's `GenUniform` result.
        let bound = (i as u64).wrapping_add(1);
        let j = rng.uniform(bound) as usize;
        // shuffle.h:29 — `DoSwap(*(begin + i), *(begin + j))`. Both `i` and `j`
        // are `< n`, so this is a checked, in-bounds swap (no panic).
        v.swap(i, j);
    }
    v
}

/// Generates `permutation_count` fold permutations from a single persistent
/// `TFastRng64::from_seed(random_seed)`, advancing the draw stream CONTINUOUSLY
/// across folds (the upstream `TRestorableFastRng64 rand` shared by every
/// fold's `InitPermutationData` call — `learn_context.cpp` fold-creation loop).
///
/// Fold `k`'s permutation consumes exactly `n - 1` draws (one per Fisher-Yates
/// step, for `n > 1`); the next fold continues from the resulting RNG phase —
/// never reseeded per fold. This mirrors the bootstrap.rs persistent-RNG
/// discipline (the stream is continuous, documented per draw).
///
/// `permutation_count == 0` yields an empty `Vec` (no folds); `permutation_count
/// == 1` yields exactly fold 0 (the only permutation the in-scope Wave-0
/// fixtures pin). Fold-creation knobs (`LearningFoldCount` / the averaging fold)
/// live in [`crate::fold`]; this function is the raw permutation generator the
/// fold machinery layers prefixes over.
#[must_use]
pub fn permutations(n: usize, permutation_count: usize, random_seed: u64) -> Vec<Vec<i32>> {
    let mut rng = TFastRng64::from_seed(random_seed);
    (0..permutation_count)
        .map(|_| shuffle_in_place(n, &mut rng))
        .collect()
}
