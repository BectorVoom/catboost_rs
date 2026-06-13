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
