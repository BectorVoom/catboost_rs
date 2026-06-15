//! Integer-exact oracle for the upstream INITIAL LEARN-SET SHUFFLE `S` (plan 05-19, ORD-01).
//!
//! `S` is catboost's `CreateShuffledIndices` (`permutation.h:84`), invoked by
//! `ShuffleLearnDataIfNeeded` (`preprocess.cpp:183`) as the FIRST consumer of
//! `TRestorableFastRng64(random_seed)` (zero pre-draws, `train_model.cpp:1057-1058`). The exact
//! `S` for the `tensor_ctr_e2e` fixture (n=30, seed=0) was DERIVED by direct trainer
//! instrumentation (the `learn_set_shuffle` event: `pre_shuffle_callcount == 0`, 29 draws), NOT
//! fitted. The earlier reconstructed `S=[5,2,6,...]` (05-18 SUMMARY / research draft) was a
//! cat+y reconstruction error; the captured ground truth below is authoritative and equals
//! `fisher_yates_permutation(30, 0)` — i.e. cb-train's existing primitive already reproduces it.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use cb_train::{create_shuffled_indices, fisher_yates_permutation};

/// The instrumented upstream ground truth: ShuffleLearnDataIfNeeded subset indices for n=30,
/// random_seed=0 (`learn_set_shuffle` event, plan 05-19). `S[k]` = original object index placed
/// at shuffled position `k`.
const UPSTREAM_S_N30_SEED0: [i32; 30] = [
    8, 12, 5, 18, 14, 28, 13, 17, 29, 25, 7, 24, 26, 10, 3, 11, 6, 19, 27, 15, 23, 4, 22, 2, 21,
    20, 16, 0, 1, 9,
];

#[test]
fn create_shuffled_indices_reproduces_upstream_s_integer_exact() {
    let s = create_shuffled_indices(30, 0);
    assert_eq!(
        s, UPSTREAM_S_N30_SEED0,
        "create_shuffled_indices(30,0) must equal the instrumented upstream learn-set shuffle S"
    );
}

#[test]
fn s_equals_fisher_yates_same_algorithm_zero_pre_draws() {
    // Upstream CreateShuffledIndices == iota + shuffle.h Fisher-Yates over the seed-0 RNG with
    // zero pre-draws — identical to fisher_yates_permutation. Documents the resolved mechanism:
    // the prior `S != fisher_yates(30,0)` claim was a reconstruction artifact, not a real divergence.
    assert_eq!(create_shuffled_indices(30, 0), fisher_yates_permutation(30, 0));
}

#[test]
fn s_is_a_bijection_over_0_n() {
    let s = create_shuffled_indices(30, 0);
    let mut seen = [false; 30];
    for &v in &s {
        let idx = usize::try_from(v).unwrap();
        assert!(idx < 30, "index in range");
        assert!(!seen[idx], "S is a bijection (no repeats)");
        seen[idx] = true;
    }
}

#[test]
fn s_trivial_for_degenerate_sizes() {
    // n<=1: upstream Shuffle loop (i in 1..n) never runs -> identity, ZERO draws.
    assert_eq!(create_shuffled_indices(0, 0), Vec::<i32>::new());
    assert_eq!(create_shuffled_indices(1, 0), vec![0]);
}
