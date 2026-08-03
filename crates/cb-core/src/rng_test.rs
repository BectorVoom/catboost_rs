//! Bitstream-oracle tests for [`crate::rng::TFastRng64`].
//!
//! Every expected value here is transcribed verbatim from the vendored upstream
//! unit test `catboost-master/util/random/fast_ut.cpp` (suite `TTestFastRng`).
//! These vectors are the parity oracle: the Rust port must reproduce CatBoost's
//! raw PRNG bitstream exactly (INFRA-05). Kept in a dedicated `*_test.rs` file
//! per the source/test separation rule (D-17); no `#[cfg(test)] mod` lives in
//! `rng.rs`.

use crate::error::CbError;
use crate::rng::TFastRng64;

/// fast_ut.cpp `Test3`: `TFastRng64 rng(17); rng.GenRand() == 14895365814383052362`.
#[test]
fn test3_from_seed_17_first_gen_rand() {
    let mut rng = TFastRng64::from_seed(17);
    assert_eq!(rng.gen_rand(), 14_895_365_814_383_052_362_u64);
}

/// fast_ut.cpp `Test2`: `TFastRng64 rng(0, 1, 2, 3)`, then `Uniform(100)` twenty
/// times yields the committed `R1[]` sequence.
#[test]
fn test2_new_0_1_2_3_uniform_100_sequence() {
    const EXPECTED: [u64; 20] = [
        37, 43, 76, 17, 12, 87, 60, 4, 83, 47, 57, 81, 28, 45, 66, 74, 18, 17, 18, 75,
    ];

    let mut rng = TFastRng64::new(0, 1, 2, 3);
    for &expected in &EXPECTED {
        assert_eq!(rng.uniform(100), expected);
    }
}

/// fast_ut.cpp `TestAdvance` (64-bit half): advancing one generator by 100 equals
/// calling `GenRand()` 100 times on an identical generator; their next outputs match.
#[test]
fn test_advance_parity_with_100_gen_rand_calls() {
    let mut stepped = TFastRng64::new(0, 1, 2, 3);
    let mut advanced = TFastRng64::new(0, 1, 2, 3);

    for _ in 0..100 {
        stepped.gen_rand();
    }
    advanced.advance(100);

    assert_eq!(stepped.gen_rand(), advanced.gen_rand());
}

/// fast_ut.cpp `TestAdvanceBoundaries`: `Advance(0)` is a no-op; `Advance(1)`
/// equals a single `GenRand()` step. (Extra coverage on the 64-bit generator.)
#[test]
fn test_advance_boundaries_zero_is_noop_one_is_single_step() {
    // Advance(0) must not change the stream.
    let mut baseline = TFastRng64::new(0, 1, 2, 3);
    let mut zero_advanced = TFastRng64::new(0, 1, 2, 3);
    zero_advanced.advance(0);
    assert_eq!(baseline.gen_rand(), zero_advanced.gen_rand());

    // Advance(1) equals one GenRand() step: after a single step on `stepped`,
    // its next output must equal `one_advanced`'s first output.
    let mut stepped = TFastRng64::new(0, 1, 2, 3);
    let mut one_advanced = TFastRng64::new(0, 1, 2, 3);
    stepped.gen_rand();
    one_advanced.advance(1);
    assert_eq!(stepped.gen_rand(), one_advanced.gen_rand());
}

/// The `Uniform` precondition (bound > 0): `try_uniform(.., 0)` returns
/// `Err(CbError::InvalidBound)` and never panics; `try_uniform(.., 100)` on
/// `new(0, 1, 2, 3)` returns `Ok(37)` (first value of the `Test2` sequence).
#[test]
fn try_uniform_rejects_zero_bound_without_panicking() {
    let mut rng = TFastRng64::new(0, 1, 2, 3);
    match rng.try_uniform(0) {
        Err(CbError::InvalidBound { bound }) => assert_eq!(bound, 0),
        other => panic!("expected Err(InvalidBound), got {other:?}"),
    }
}

#[test]
fn try_uniform_valid_bound_matches_uniform_first_value() {
    let mut rng = TFastRng64::new(0, 1, 2, 3);
    assert_eq!(rng.try_uniform(100), Ok(37));
}

/// `GenRandReal1` (common_ops.h:19,99) for a ui64 engine is
/// `(GenRand() >> 11) * (1.0 / (2^53 - 1))`. From `from_seed(17)` the first
/// `GenRand()` is `14895365814383052362` (fast_ut.cpp `Test3`), so the first
/// `gen_rand_real1()` is exactly `(14895365814383052362 >> 11) / 9007199254740991`.
#[test]
fn gen_rand_real1_from_seed_17_matches_to_rand_real1() {
    let mut rng = TFastRng64::from_seed(17);
    let expected = (14_895_365_814_383_052_362u64 >> 11) as f64 * (1.0 / 9_007_199_254_740_991.0);
    let got = rng.gen_rand_real1();
    assert_eq!(got, expected);
    // The draw lies in the closed unit interval [0, 1].
    assert!((0.0..=1.0).contains(&got));
}

/// ORCH-03-S2 (TASK-02): [`TFastRng64::from_raw_state`] restores a generator that
/// reproduces the ORIGINAL's `gen_rand` stream bit-for-bit from the captured
/// position, and carries the captured `call_count` forward.
///
/// This is the snapshot/resume parity contract: a training run persists
/// `(raw_state(), call_count())`, and the resumed run must consume the SAME draw
/// sequence the straight-through run would have consumed. Exercised over several
/// `(seed, offset, continue_draws)` triples so a constructor that accidentally
/// re-derives from a seed (rather than restoring the raw state) cannot pass at
/// offset 0 alone.
#[test]
fn from_raw_state_reproduces_the_gen_rand_stream_bit_for_bit() {
    for &(seed, offset, continue_draws) in &[
        (17u64, 0usize, 8usize),
        (17, 1, 8),
        (0, 5, 16),
        (42, 37, 24),
        (u64::MAX, 129, 4),
    ] {
        let mut original = TFastRng64::from_seed(seed);
        for _ in 0..offset {
            original.gen_rand();
        }

        let mut restored = TFastRng64::from_raw_state(original.raw_state(), original.call_count());
        assert_eq!(
            restored.call_count(),
            original.call_count(),
            "restored call_count must equal the captured one (seed={seed}, offset={offset})"
        );
        assert_eq!(
            restored.raw_state(),
            original.raw_state(),
            "restored raw state must equal the captured one (seed={seed}, offset={offset})"
        );

        for draw in 0..continue_draws {
            assert_eq!(
                restored.gen_rand(),
                original.gen_rand(),
                "restored stream diverged at draw {draw} (seed={seed}, offset={offset})"
            );
        }
        assert_eq!(
            restored.call_count(),
            original.call_count(),
            "call_count must advance in lockstep (seed={seed}, offset={offset})"
        );
    }
}

/// The restore is a pure state transplant: `raw_state` alone (WITHOUT the
/// original seed) is sufficient. A generator restored at offset `M` must NOT
/// match a fresh `from_seed` generator's stream — otherwise the test above could
/// pass with a constructor that ignored `raw_state` entirely.
#[test]
fn from_raw_state_differs_from_a_fresh_generator_at_a_nonzero_offset() {
    let mut original = TFastRng64::from_seed(17);
    for _ in 0..11 {
        original.gen_rand();
    }
    let mut restored = TFastRng64::from_raw_state(original.raw_state(), original.call_count());
    let mut fresh = TFastRng64::from_seed(17);
    assert_ne!(
        restored.gen_rand(),
        fresh.gen_rand(),
        "a restored generator at offset 11 must not replay the stream from the start"
    );
}
