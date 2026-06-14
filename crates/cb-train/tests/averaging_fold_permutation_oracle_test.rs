//! Standalone integer-exact AveragingFold-permutation DRAW-ORDER oracle (ORD-05,
//! Plan 05-12 — the D-03-style de-risk linchpin gating Plan 05-13's leaf-value
//! materialization).
//!
//! # Why this gate exists
//!
//! Under `boosting_type=Plain` WITH `hasCtrs=true`, upstream catboost 1.2.10
//! builds the lone learning `Folds[0]` as the IDENTITY permutation (no shuffle,
//! ZERO RNG draws — `shuffle = foldIdx != 0`, `learn_context.cpp:524` /
//! `fold.cpp:54`) and the AveragingFold as the FIRST real seeded Fisher-Yates
//! draw (`IsAverageFoldPermuted = hasCtrs`, `learn_context.cpp:575-577`). The
//! research (05-CTR-LEAF-VALUE-RESEARCH.md) proved tree0's leaf values are
//! estimated on the AveragingFold's SHUFFLED permutation, so a CTR partition
//! computed under the WRONG permutation is meaningless — exactly the discipline
//! of the existing D-03 permutation gate.
//!
//! This oracle locks the cb-train AveragingFold permutation (for the
//! `tensor_ctr_e2e` config: N=30, seed=0, permutation_count=1, hasCtrs=true)
//! to the research-reproduced draw order BEFORE any leaf-value or e2e value
//! stage runs (Plan 05-13/05-14). It is self-consistent: the expected
//! permutation is derived from the PRODUCTION `fisher_yates_permutation` (not a
//! committed `.npy`), so no fixture is touched.
//!
//! # Reconciling the STATE.md 05-12 blocker note
//!
//! The blocker note recorded the executor's offline reverse-engineering: real
//! learn perm = `fisher_yates(30,0)`, averaging perm = `permutations(30,2,0)[1]`
//! under the OLD all-shuffle scheme. POST-FIX (Plan 05-12 Task 1) the STRUCTURE
//! (learning) fold is the IDENTITY and the AVERAGING fold takes the FIRST draw =
//! `fisher_yates_permutation(30,0)` — because the identity learning fold now
//! draws nothing, the averaging shuffle is the first seeded draw. The two views
//! reconcile: the byte sequence the OLD scheme assigned to the learning fold0 is
//! now the averaging fold's permutation.
//!
//! # The two-materialization roles (research's "Summary of the two materializations")
//!
//! - the IDENTITY learning fold is the STRUCTURE-search permutation;
//! - the AVERAGING fold (this oracle's subject) is the LEAF-VALUE permutation.
//!
//! Integer-exact comparison only (`Stage::Permutation`, the D-03 comparator —
//! NOT a 1e-5 value check). Runs unconditionally — NO `#[ignore]`.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use cb_oracle::{compare_permutation, Stage};
use cb_train::{create_folds, fisher_yates_permutation, Fold};

/// The `tensor_ctr_e2e` fixture parameters (matched to
/// `tensor_ctr_e2e_oracle_test.rs`).
const FIXTURE_N: usize = 30;
const FIXTURE_SEED: u64 = 0;
const FIXTURE_PERMUTATION_COUNT: usize = 1;
const FOLD_LEN_MULTIPLIER: f64 = 2.0;

/// Build the `tensor_ctr_e2e` fold set: hasCtrs ⇒ a learning permutation is
/// needed, Plain ⇒ `dynamic_body_tail=false` (the online-CTR prefix uses the
/// single full span). permutation_count=1 ⇒ 1 learning fold + 1 averaging fold.
fn tensor_ctr_e2e_folds() -> Vec<Fold> {
    create_folds(
        FIXTURE_N,
        FIXTURE_PERMUTATION_COUNT,
        /* permutation_needed_for_learning = */ true,
        /* dynamic_body_tail = */ false,
        FOLD_LEN_MULTIPLIER,
        FIXTURE_SEED,
    )
}

/// The integer-exact D-03-style gate: the AveragingFold permutation byte-equals
/// `fisher_yates_permutation(30, 0)` index-for-index — it is the FIRST seeded
/// Fisher-Yates draw (drawn AFTER the zero-draw identity learning fold). This is
/// the LEAF-VALUE permutation Plan 05-13 materializes the AveragingFold CTR over.
#[test]
fn averaging_fold_permutation_is_first_seeded_draw() {
    let folds = tensor_ctr_e2e_folds();

    let averaging = folds
        .iter()
        .find(|f| f.is_averaging)
        .expect("an averaging fold must exist (1 learning + 1 averaging)");

    // The research-reproduced relationship: post-fix the averaging fold is the
    // first seeded draw, so it equals a fresh-seed Fisher-Yates over the SAME
    // seed (the identity learning Folds[0] consumed zero draws).
    let expected: Vec<i64> = fisher_yates_permutation(FIXTURE_N, FIXTURE_SEED)
        .into_iter()
        .map(i64::from)
        .collect();
    let actual: Vec<i64> = averaging.permutation.iter().map(|&v| i64::from(v)).collect();

    // Integer-exact (Stage::Permutation, D-03) — NOT a 1e-5 value tolerance.
    compare_permutation(&expected, &actual).unwrap_or_else(|e| {
        panic!("AveragingFold permutation diverged from fisher_yates_permutation(30, 0) [{:?}]: {e}", Stage::Permutation)
    });
}

/// The STRUCTURE-search permutation: the FIRST non-averaging (learning) fold is
/// the IDENTITY `[0..30]` (zero draws), per upstream's `shuffle = foldIdx != 0`.
#[test]
fn learning_fold_is_identity_zero_draws() {
    let folds = tensor_ctr_e2e_folds();

    let learning = folds
        .iter()
        .find(|f| !f.is_averaging)
        .expect("a learning fold must exist (the structure-search permutation)");

    let identity: Vec<i64> = (0..FIXTURE_N as i64).collect();
    let actual: Vec<i64> = learning.permutation.iter().map(|&v| i64::from(v)).collect();

    // Integer-exact (Stage::Permutation, D-03): the structure fold draws nothing.
    compare_permutation(&identity, &actual).unwrap_or_else(|e| {
        panic!("learning Folds[0] is not the identity [0..30] [{:?}]: {e}", Stage::Permutation)
    });
}

/// Cross-check the two-materialization invariant: the STRUCTURE (identity)
/// permutation and the LEAF-VALUE (averaging) permutation are DISTINCT — the
/// whole reason structure search and leaf estimation diverge in the research.
#[test]
fn structure_and_leaf_value_permutations_are_distinct() {
    let folds = tensor_ctr_e2e_folds();
    let learning = folds.iter().find(|f| !f.is_averaging).expect("learning fold");
    let averaging = folds.iter().find(|f| f.is_averaging).expect("averaging fold");
    assert_ne!(
        learning.permutation, averaging.permutation,
        "the structure (identity) and leaf-value (averaging) permutations must differ"
    );
}
