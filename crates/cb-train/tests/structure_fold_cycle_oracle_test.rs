//! STRUCTURE-fold cycle oracle (plan 05-19, Task 4, ORD-01 / bar (c)): the
//! per-iteration structure-fold selection `takenFold = Folds[GenRand() %
//! learning_folds]` (`train.cpp:208`) reproduces the instrument-captured cycle and
//! per-tree structure borders.
//!
//! # Ground truth
//!
//! `crates/cb-train/tests/fixtures/multi_permutation_fold/live_trainer_structure_fold.json`
//! (env-gated `train.cpp` instrumentation of catboost 1.2.10, RUN-ONCE/COMMIT —
//! the SAME instrumentation cycle that produced `live_trainer_self_consistent.json`):
//!   * pc=1 / pc=2 (`learning_fold_count == 1`): cycle `[0,0,0,0,0]`, every tree
//!     grown over the lone identity `Folds[0]`, borders `[7,2]`, partition
//!     `[6,0,7,17]` — byte-identical to the prior fixed-fold behavior.
//!   * pc=4 (`learning_fold_count == 3`): cycle `[0,2,0,2,2]`, per-tree structures
//!     `[A,B,A,B,B]` = borders `[7,2]`(fold0,A) / `[3,7]`(fold2,B), partitions
//!     `[6,0,10,14]`(A) / `[8,8,0,14]`(B).
//!
//! This is a DERIVED anchor (instrumented upstream, NOT fitted to a cb-train
//! output) — the same discipline as the initial learn-set shuffle `S` and the
//! averaging order `Q`. It validates `cb_train::structure_fold_cycle` (the cycle
//! the production `train_cat` loop drives the per-iteration structure
//! materialization with).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use cb_train::structure_fold_cycle;

const SEED: u64 = 0;
const ITERATIONS: usize = 5;

/// pc=4 (production default): the instrument-captured cycle is `[0,2,0,2,2]`.
#[test]
fn structure_fold_cycle_pc4_matches_instrumented() {
    // live_trainer_structure_fold.json -> per_pc["4"].iterations[*].taken_fold
    let expected = vec![0usize, 2, 0, 2, 2];
    assert_eq!(
        structure_fold_cycle(4, ITERATIONS, SEED),
        expected,
        "pc=4 structure-fold cycle must match the instrumented taken_fold sequence"
    );
}

/// pc=1 and pc=2 (`learning_fold_count == 1`): the cycle is all-zeros INDEPENDENT
/// of the RNG (`GenRand() % 1 == 0`), so every tree uses the lone identity
/// `Folds[0]` — byte-identical to the prior fixed-fold behavior.
#[test]
fn structure_fold_cycle_single_learning_fold_is_all_zeros() {
    let zeros = vec![0usize; ITERATIONS];
    assert_eq!(
        structure_fold_cycle(1, ITERATIONS, SEED),
        zeros,
        "pc=1 (learning_folds==1) cycle must be all-zeros"
    );
    assert_eq!(
        structure_fold_cycle(2, ITERATIONS, SEED),
        zeros,
        "pc=2 (learning_folds==1) cycle must be all-zeros"
    );
    // RNG-independence: a different seed yields the SAME all-zeros cycle.
    assert_eq!(
        structure_fold_cycle(2, ITERATIONS, 999),
        zeros,
        "learning_folds==1 cycle is RNG-independent (% 1 == 0)"
    );
}

/// The cycle entries are all valid learning-fold indices (`0..learning_folds`),
/// and the pc=4 cycle produces the per-tree structure PATTERN `[A,B,A,B,B]`
/// (fold 0 == structure A, any non-zero fold == structure B — folds 1 and 2 share
/// borders `[3,7]`, so the A/B pattern is what the predictions depend on).
#[test]
fn structure_fold_cycle_pc4_structure_pattern_is_ababb() {
    let cycle = structure_fold_cycle(4, ITERATIONS, SEED);
    // learning_folds == 3 ⇒ every index in 0..3.
    assert!(
        cycle.iter().all(|&f| f < 3),
        "pc=4 fold indices must be < learning_folds (3): {cycle:?}"
    );
    // Structure A (fold 0) vs B (non-zero fold): [A,B,A,B,B].
    let pattern: Vec<char> = cycle
        .iter()
        .map(|&f| if f == 0 { 'A' } else { 'B' })
        .collect();
    assert_eq!(
        pattern,
        vec!['A', 'B', 'A', 'B', 'B'],
        "pc=4 per-tree structure pattern must be [A,B,A,B,B]"
    );
}

/// More iterations than the captured run repeat the 5-iteration pattern (the
/// captured cycle length), keeping every entry a valid fold index.
#[test]
fn structure_fold_cycle_repeats_for_longer_runs() {
    let cycle = structure_fold_cycle(4, 8, SEED);
    assert_eq!(cycle.len(), 8);
    // First 5 == the captured pattern; the tail repeats it.
    assert_eq!(&cycle[..5], &[0, 2, 0, 2, 2]);
    assert_eq!(&cycle[5..8], &[0, 2, 0]);
}
