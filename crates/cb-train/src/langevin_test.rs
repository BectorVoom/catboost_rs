//! Unit tests for the Langevin noise primitives.
//!
//! The end-to-end parity is the facade oracle's job
//! (`crates/catboost-rs/tests/langevin_oracle_test.rs`). What is checked HERE is
//! the part that oracle cannot isolate: the BLOCK-SEEDED stream, whose defining
//! property is that the block size is a compile-time constant rather than the
//! thread count. That is what makes upstream's `thread_count` numerically inert,
//! and it is invisible in any single fit.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use cb_core::{std_normal, TFastRng64};

use crate::langevin::{
    add_noise_to_derivatives, add_noise_to_leaf_der_sums, add_noise_to_leaf_newton_sums,
    langevin_noise_rate, LANGEVIN_BLOCK_SIZE,
};

/// `CalcLangevinNoiseRate(dt, lr) = sqrt(2.0 / lr / dt)` — including the `float`
/// narrowing of both arguments.
#[test]
fn noise_rate_matches_the_upstream_formula() {
    let (dt, lr) = (100.0_f64, 0.3_f64);
    let expected = (2.0 / f64::from(lr as f32) / f64::from(dt as f32)).sqrt();
    assert!((langevin_noise_rate(dt, lr) - expected).abs() < 1e-15);
}

/// Higher temperature ⇒ SMALLER noise. A reciprocal slip would still produce
/// "noise" and still pass a single-cell parity check.
#[test]
fn noise_rate_falls_as_temperature_rises() {
    let lr = 0.3;
    assert!(langevin_noise_rate(1.0, lr) > langevin_noise_rate(100.0, lr));
    assert!(langevin_noise_rate(100.0, lr) > langevin_noise_rate(10_000.0, lr));
    // ∝ 1/sqrt(dt): a 100x temperature is a 10x smaller coefficient.
    let ratio = langevin_noise_rate(1.0, lr) / langevin_noise_rate(100.0, lr);
    assert!((ratio - 10.0).abs() < 1e-9, "expected a 10x ratio, got {ratio}");
}

/// A zero temperature is a no-op — no noise AND no RNG consumed.
#[test]
fn zero_temperature_leaves_the_derivatives_untouched() {
    let mut ders = vec![1.0, -2.0, 3.5];
    let before = ders.clone();
    add_noise_to_derivatives(&mut ders, 0.0, 0.3, 12345);
    assert_eq!(ders, before);
}

/// The defining property: block `b` covers `[b*128, min((b+1)*128, n))` and is
/// driven by ONE `TFastRng64::from_seed(seed + b)` drawn sequentially. This
/// reconstructs the expected values independently of the implementation.
#[test]
fn derivative_noise_uses_a_per_block_seed_of_128() {
    let n = 300; // 3 blocks: 128 + 128 + 44
    let (dt, lr, seed) = (100.0_f64, 0.3_f64, 987_654_321_u64);
    let coef = langevin_noise_rate(dt, lr);

    let mut actual = vec![0.0_f64; n];
    add_noise_to_derivatives(&mut actual, dt, lr, seed);

    let mut expected = vec![0.0_f64; n];
    let mut idx = 0;
    let mut block_idx = 0_u64;
    while idx < n {
        let mut rng = TFastRng64::from_seed(seed.wrapping_add(block_idx));
        let end = usize::min(idx + LANGEVIN_BLOCK_SIZE, n);
        while idx < end {
            expected[idx] = coef * std_normal(&mut rng);
            idx += 1;
        }
        block_idx += 1;
    }
    assert_eq!(actual, expected);
}

/// The blocking must be by the CONSTANT 128, not by the buffer length: element
/// 128 starts a NEW stream, so it cannot equal what a single-stream draw would
/// have produced there.
#[test]
fn the_block_boundary_restarts_the_stream() {
    let n = LANGEVIN_BLOCK_SIZE * 2;
    let (dt, lr, seed) = (1.0_f64, 0.3_f64, 42_u64);

    let mut blocked = vec![0.0_f64; n];
    add_noise_to_derivatives(&mut blocked, dt, lr, seed);

    // One continuous stream over the whole range — what a "seed once" reading of
    // the upstream code would produce.
    let coef = langevin_noise_rate(dt, lr);
    let mut rng = TFastRng64::from_seed(seed);
    let single: Vec<f64> = (0..n).map(|_| coef * std_normal(&mut rng)).collect();

    assert_eq!(
        blocked[..LANGEVIN_BLOCK_SIZE],
        single[..LANGEVIN_BLOCK_SIZE],
        "block 0 shares the base seed, so its values must agree"
    );
    assert_ne!(
        blocked[LANGEVIN_BLOCK_SIZE], single[LANGEVIN_BLOCK_SIZE],
        "element {LANGEVIN_BLOCK_SIZE} must start a FRESH stream seeded from seed+1"
    );
}

/// A buffer shorter than one block is a single block — the common small-fixture
/// case, and the one where a blocking bug would hide.
#[test]
fn a_short_buffer_is_one_block() {
    let (dt, lr, seed) = (100.0_f64, 0.3_f64, 7_u64);
    let coef = langevin_noise_rate(dt, lr);
    let mut actual = vec![0.0_f64; 10];
    add_noise_to_derivatives(&mut actual, dt, lr, seed);

    let mut rng = TFastRng64::from_seed(seed);
    let expected: Vec<f64> = (0..10).map(|_| coef * std_normal(&mut rng)).collect();
    assert_eq!(actual, expected);
}

/// Leaf-sum noise: ONE rng for all leaves, scaled per leaf by
/// `sqrt(sum_weight + scaled_l2)`, and a leaf under the `1e-9` weight threshold
/// is SKIPPED WITHOUT DRAWING — the skip is part of the stream, so a later leaf's
/// noise depends on how many earlier leaves were empty.
#[test]
fn leaf_der_sum_noise_skips_empty_leaves_without_drawing() {
    let (dt, lr, l2, seed) = (100.0_f64, 0.3_f64, 3.0_f64, 555_u64);
    let coef = langevin_noise_rate(dt, lr);
    let weights = vec![10.0, 0.0, 20.0, 0.0, 30.0];

    let mut sums = vec![0.0_f64; 5];
    add_noise_to_leaf_der_sums(&mut sums, &weights, dt, lr, l2, seed);

    let mut rng = TFastRng64::from_seed(seed);
    let mut expected = vec![0.0_f64; 5];
    for (leaf, &w) in weights.iter().enumerate() {
        if w < 1e-9 {
            continue; // no draw
        }
        expected[leaf] = coef * (w + l2).sqrt() * std_normal(&mut rng);
    }
    assert_eq!(sums, expected);
    assert_eq!(sums[1], 0.0, "an empty leaf gets no noise");
    assert_eq!(sums[3], 0.0, "an empty leaf gets no noise");
}

/// Removing an empty leaf must SHIFT the later leaves' noise. This is what proves
/// the skip is stream-affecting rather than a cosmetic zeroing.
#[test]
fn an_empty_leaf_shifts_the_later_draws() {
    let (dt, lr, l2, seed) = (100.0_f64, 0.3_f64, 3.0_f64, 555_u64);

    let mut with_gap = vec![0.0_f64; 3];
    add_noise_to_leaf_der_sums(&mut with_gap, &[10.0, 0.0, 30.0], dt, lr, l2, seed);

    let mut no_gap = vec![0.0_f64; 3];
    add_noise_to_leaf_der_sums(&mut no_gap, &[10.0, 20.0, 30.0], dt, lr, l2, seed);

    assert_ne!(
        with_gap[2], no_gap[2],
        "leaf 2's draw must depend on whether leaf 1 consumed one"
    );
}

/// The Newton variant scales by `sqrt(|sum_der2| + l2)` but still SKIPS on the
/// summed WEIGHT — two different quantities, and swapping them would change both
/// the scale and the stream.
#[test]
fn newton_leaf_noise_scales_on_der2_but_skips_on_weight() {
    let (dt, lr, l2, seed) = (100.0_f64, 0.3_f64, 3.0_f64, 99_u64);
    let coef = langevin_noise_rate(dt, lr);
    let weights = vec![10.0, 0.0, 20.0];
    // Deliberately NEGATIVE der2 (the usual sign for a maximisation Hessian) so
    // the absolute value matters.
    let der2 = vec![-4.0, -100.0, -9.0];

    let mut sums = vec![0.0_f64; 3];
    add_noise_to_leaf_newton_sums(&mut sums, &der2, &weights, dt, lr, l2, seed);

    let mut rng = TFastRng64::from_seed(seed);
    let mut expected = vec![0.0_f64; 3];
    for (leaf, &w) in weights.iter().enumerate() {
        if w < 1e-9 {
            continue;
        }
        expected[leaf] = coef * (der2[leaf].abs() + l2).sqrt() * std_normal(&mut rng);
    }
    assert_eq!(sums, expected);
    assert_eq!(sums[1], 0.0, "leaf 1 is skipped on WEIGHT despite a large |der2|");
}
