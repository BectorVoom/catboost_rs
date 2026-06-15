//! Unit tests for the Fisher-Yates fold permutation ([`crate::permutation`]).
//!
//! These lock the local algebra (identity edge cases, a hand-traced small-N
//! draw sequence, block-size boundary, and the persistent-RNG fold stream). The
//! integer-exact upstream lock vs the committed `permutation_fold{k}.npy` lives
//! in the `permutation_oracle_test.rs` integration test (D-03).

use super::{
    averaging_ctr_permutation, create_shuffled_indices, fisher_yates_permutation, fold_block_size,
    permutations, PERMUTATION_BLOCK_SIZE_THRESHOLD,
};
use cb_core::TFastRng64;

/// A permutation of `0..n` is a bijection: every index appears exactly once.
fn is_permutation_of_range(v: &[i32], n: usize) -> bool {
    if v.len() != n {
        return false;
    }
    let mut seen = vec![false; n];
    for &x in v {
        if x < 0 || (x as usize) >= n {
            return false;
        }
        let slot = &mut seen[x as usize];
        if *slot {
            return false;
        }
        *slot = true;
    }
    seen.iter().all(|&b| b)
}

#[test]
fn n0_is_empty_no_draws() {
    // N == 0: the loop `1..0` never runs; the identity is empty.
    assert_eq!(fisher_yates_permutation(0, 0), Vec::<i32>::new());
    assert_eq!(fisher_yates_permutation(0, 42), Vec::<i32>::new());
}

#[test]
fn n1_is_identity_no_draws() {
    // N == 1: the loop `1..1` never runs; the single element stays put, so the
    // permutation is identity REGARDLESS of seed (no draw is taken).
    assert_eq!(fisher_yates_permutation(1, 0), vec![0]);
    assert_eq!(fisher_yates_permutation(1, 12345), vec![0]);
}

#[test]
fn known_small_n_draw_sequence_seed42() {
    // Hand-anchored small-N sequence: seed=42, N=5 -> [4 2 0 3 1].
    // This is the 05-01 self-oracle anchor (harness [4 2 0 3 1] ==
    // cb-core::TFastRng64 Fisher-Yates), so it cross-locks the draw order /
    // generator wiring against the already-bitstream-verified RNG.
    assert_eq!(fisher_yates_permutation(5, 42), vec![4, 2, 0, 3, 1]);
}

#[test]
fn draw_order_matches_manual_uniform_replay() {
    // Independently replay shuffle.h:28-30 against the raw RNG: identity init,
    // for i in 1..n draw uniform(i+1) and swap(i, j). The module must produce
    // the SAME array — proving it consumes the generator in the exact upstream
    // draw order, not merely *some* permutation.
    let n = 8;
    let seed = 7;
    let mut rng = TFastRng64::from_seed(seed);
    let mut expected: Vec<i32> = (0..n as i32).collect();
    for i in 1..n {
        let j = rng.uniform((i as u64) + 1) as usize;
        expected.swap(i, j);
    }
    assert_eq!(fisher_yates_permutation(n, seed), expected);
}

#[test]
fn output_is_always_a_valid_permutation() {
    for &n in &[2usize, 3, 5, 16, 30, 64] {
        for &seed in &[0u64, 1, 42, 999] {
            let p = fisher_yates_permutation(n, seed);
            assert!(
                is_permutation_of_range(&p, n),
                "n={n} seed={seed} produced a non-bijection: {p:?}"
            );
        }
    }
}

#[test]
fn block_size_is_one_below_threshold() {
    // DefaultFoldPermutationBlockSize = min(256, N/1000 + 1) == 1 for N < 1000.
    assert_eq!(fold_block_size(0), 1);
    assert_eq!(fold_block_size(30), 1);
    assert_eq!(fold_block_size(PERMUTATION_BLOCK_SIZE_THRESHOLD - 1), 1);
    // At and beyond the threshold the block grows (out-of-scope path, but the
    // arithmetic is locked so a future slice can branch on it).
    assert_eq!(fold_block_size(PERMUTATION_BLOCK_SIZE_THRESHOLD), 2);
    assert_eq!(fold_block_size(2000), 3);
    // Capped at 256.
    assert_eq!(fold_block_size(10_000_000), 256);
}

#[test]
fn permutations_fold0_equals_single_fisher_yates() {
    // permutation_count == 1 must reproduce exactly fold 0 from the same seed
    // (the only fold the in-scope fixtures pin).
    let n = 30;
    let seed = 0;
    let folds = permutations(n, 1, seed);
    assert_eq!(folds.len(), 1);
    assert_eq!(folds[0], fisher_yates_permutation(n, seed));
}

#[test]
fn permutations_zero_count_is_empty() {
    assert!(permutations(30, 0, 0).is_empty());
}

#[test]
fn averaging_ctr_permutation_reproduces_self_consistent_q_pc1() {
    // ORD-01 / bar (c): Q = [S[p] for p in P_avg] must equal the trainer-captured
    // `object_permutation_Q` (live_trainer_self_consistent.json, pc1.averaging_fold).
    // pc1: permutation_count == 1 ⇒ learning_folds == max(1, 1-1) == 1.
    let q_expected: [i32; 30] = [
        7, 19, 20, 18, 13, 2, 17, 1, 14, 28, 12, 3, 8, 22, 4, 29, 26, 16, 24, 27, 5, 23, 21, 10, 9,
        15, 11, 25, 0, 6,
    ];
    assert_eq!(
        averaging_ctr_permutation(30, 1, 0),
        q_expected.to_vec(),
        "pc1 averaging CTR order Q must match the self-consistent fixture"
    );
}

#[test]
fn averaging_ctr_permutation_reproduces_self_consistent_q_pc4() {
    // pc4: permutation_count == 4 ⇒ learning_folds == max(1, 4-1) == 3. The
    // averaging shuffle is the 3rd full shuffle on the persistent stream (S is the
    // 0th), NOT the 05-17 single-`gen_rand`-per-fold compensating hack.
    let q_expected: [i32; 30] = [
        24, 27, 11, 9, 6, 26, 8, 17, 15, 0, 14, 18, 28, 19, 3, 20, 25, 23, 29, 2, 13, 1, 16, 21, 5,
        10, 4, 22, 7, 12,
    ];
    assert_eq!(
        averaging_ctr_permutation(30, 3, 0),
        q_expected.to_vec(),
        "pc4 averaging CTR order Q must match the self-consistent fixture"
    );
}

#[test]
fn averaging_ctr_permutation_composes_s_with_p_avg() {
    // Structural identity: Q[k] == S[P_avg[k]], where S is shuffle #0 and P_avg is
    // shuffle #learning_folds on the SAME persistent stream.
    let n = 30;
    let seed = 0;
    let lf = 3;
    let s = create_shuffled_indices(n, seed);
    let p_avg = permutations(n, lf + 1, seed)[lf].clone();
    let expected: Vec<i32> = p_avg.iter().map(|&p| s[p as usize]).collect();
    assert_eq!(averaging_ctr_permutation(n, lf, seed), expected);
    // And it is always a valid permutation of 0..n.
    assert!(is_permutation_of_range(
        &averaging_ctr_permutation(n, lf, seed),
        n
    ));
}

#[test]
fn averaging_ctr_permutation_edge_cases() {
    // n <= 1 ⇒ identity (no draws).
    assert_eq!(averaging_ctr_permutation(0, 1, 0), Vec::<i32>::new());
    assert_eq!(averaging_ctr_permutation(1, 3, 0), vec![0]);
}

#[test]
fn permutations_stream_is_continuous_across_folds() {
    // The persistent RNG advances continuously across folds: fold 1 is NOT a
    // fresh from_seed(seed) shuffle — it continues from fold 0's RNG phase.
    // Replay the continuous stream manually and require an exact match, AND
    // require fold 1 to DIFFER from a fresh-seed fold-0 shuffle (so we know the
    // stream really advanced rather than being reseeded).
    let n = 16;
    let seed = 123;
    let count = 3;
    let got = permutations(n, count, seed);
    assert_eq!(got.len(), count);

    let mut rng = TFastRng64::from_seed(seed);
    for fold in got.iter() {
        let mut expected: Vec<i32> = (0..n as i32).collect();
        for i in 1..n {
            let j = rng.uniform((i as u64) + 1) as usize;
            expected.swap(i, j);
        }
        assert_eq!(fold, &expected);
    }

    // Fold 1 continues the stream, so it must not equal a reseeded fold-0 draw.
    assert_ne!(
        got[1],
        fisher_yates_permutation(n, seed),
        "fold 1 must continue the stream, not reseed"
    );
}
