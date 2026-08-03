//! WR-01 (TASK-05): host-side replay of the RNG draws the CPU oblivious grow
//! consumes per tree, so the DEVICE grow branch leaves the persistent
//! [`cb_core::TFastRng64`] in the SAME phase the CPU branch would.
//!
//! # Why a replay exists at all
//!
//! The device branch in [`crate::boosting`] skips the CPU level search entirely,
//! but the per-tree bootstrap sample is still drawn on the HOST from the ONE
//! persistent, CONTINUOUS training RNG (Design A / `[DECISION D1]`). Tree `k+1`'s
//! sample therefore depends on the RNG phase tree `k` left behind. If the device
//! branch consumed fewer draws than the CPU branch, every tree after the first
//! would sample from a different phase than upstream — and the ≤1e-5
//! device-vs-upstream bar (`WR01-S15`) would fail from tree 2 onward for a reason
//! that has nothing to do with the kernels.
//!
//! # The draw shape being replayed
//!
//! Whenever sampling is active (`draws_active` in `boosting.rs`) the CPU grower
//! runs the PERTURBED level search — `perturb` is `Some` even at
//! `random_strength == 0`, where `score_st_dev == 0.0` makes the perturbation a
//! numeric no-op but the draws still happen (`boosting.rs` `perturb` construction
//! and `cb_compute::random_score_instance`, which calls `std_normal`
//! unconditionally). Per LEVEL, on the MAIN stream, in this exact order:
//!
//! 1. `n_features × gen_rand_real1()` — the per-level RSM candidate reselection
//!    (`SelectFeaturesForScoring`, `tree.rs` level loop), drawn for EVERY listed
//!    float feature including border-less ones.
//! 2. `1 × gen_rand()` — the `CalcScores` `randSeed`
//!    (`select_level_perturbed`). The per-feature `TRestorableFastRng64(randSeed
//!    + taskIdx)` streams branch off this value and do NOT touch the main stream,
//!    so they are correctly absent from the replay.
//! 3. `n_features × std_normal()` — `SelectBestCandidate`'s one `GetInstance`
//!    normal per listed feature, border-less features included.
//!
//! [`std_normal`] consumes a VARIABLE number of `gen_rand_real1` draws (Marsaglia
//! polar rejection: 2, 4, 6, …). That is why this module CALLS it instead of
//! advancing the stream by a counted amount — a count would be wrong for exactly
//! the seeds that reject a pair. `PRE_TREE_DRAWS` / `POST_TREE_EXTRA_DRAWS` and
//! the `bootstrap()` call itself are NOT replayed here: the device branch shares
//! those code paths with the CPU branch.

use cb_core::{std_normal, TFastRng64};

// Source/test separation (CLAUDE.md): the tests live in a sibling file mounted as
// a child module, never inline in this production file.
#[cfg(test)]
#[path = "device_draw_replay_test.rs"]
mod tests;

/// Consume, on `rng`, exactly the draws the CPU oblivious grow would have
/// consumed for ONE tree of `depth` levels over `n_features` listed float
/// features, leaving the stream in the phase the next tree's `bootstrap()` call
/// expects (`WR01-S7`).
///
/// `n_features` is the LISTED float feature count (`FeatureMatrix::n_features()`
/// on the CPU side / `matrix.n_features()` at the call site) — border-less
/// "unused-but-quantized" features are listed and DO draw, so callers must not
/// filter them out.
///
/// A `depth` or `n_features` of `0` consumes nothing (the CPU grower's loops are
/// likewise empty). All draws route through [`cb_core::TFastRng64`] only; the
/// returned unit carries no value because the DRAWS, not their values, are the
/// observable effect.
pub fn replay_grow_draws(rng: &mut TFastRng64, depth: usize, n_features: usize) {
    for _level in 0..depth {
        // (1) Per-level RSM candidate reselection: one `GenRandReal1` per listed
        //     float feature, unconditionally (the always-select Rsm==1 regime
        //     still draws — instrumented upstream ground truth, `tree.rs:596-614`).
        for _ in 0..n_features {
            let _ = rng.gen_rand_real1();
        }
        // (2) The `CalcScores` randSeed — ONE main-stream `GenRand` per level.
        //     The per-feature `randSeed + taskIdx` sub-streams branch off it and
        //     never touch the main stream (`select_level_perturbed` step 2).
        let _ = rng.gen_rand();
        // (3) `SelectBestCandidate`: ONE `GetInstance` normal per listed feature
        //     off the MAIN stream, border-less features included. CALLED, never
        //     counted — `std_normal`'s uniform-draw count is data-dependent
        //     (rejection sampling), so a counted advance would desynchronise on
        //     every seed that rejects a pair.
        for _ in 0..n_features {
            let _ = std_normal(rng);
        }
    }
}
