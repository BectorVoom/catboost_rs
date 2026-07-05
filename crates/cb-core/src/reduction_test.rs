//! Order-lock property tests for [`crate::reduction`].
//!
//! These tests pin the naive-sequential summation order that the ≤ 1e-5 oracle
//! gate depends on (DATA-07, threat T-02-01). The adversarial
//! `[1e16, 1.0, -1e16]` case is the canary: a pairwise or Kahan reduction would
//! return `1.0`, the sequential fold returns `0.0`. Kept in a dedicated
//! `*_test.rs` file per the source/test separation rule (D-17); no
//! `#[cfg(test)] mod` lives in `reduction.rs`.

use crate::reduction::{scatter_add_f64, sum_f32_in_f64, sum_f64};

/// Order-lock: the naive left-to-right fold of `[1e16, 1.0, -1e16]` is `0.0`
/// because `1e16 + 1.0 == 1e16` in `f64`, then `1e16 - 1e16 == 0.0`. A
/// pairwise/Kahan sum would instead preserve the `1.0`. This asserts the
/// sequential contract exactly.
#[test]
fn sum_f64_naive_order_loses_small_term() {
    assert_eq!(sum_f64(&[1e16, 1.0, -1e16]), 0.0);
}

/// Empty slice sums to the additive identity `0.0`.
#[test]
fn sum_f64_empty_is_zero() {
    assert_eq!(sum_f64(&[]), 0.0);
}

/// A small known vector equals its left-to-right running sum (exact in `f64`
/// for these integer-valued operands).
#[test]
fn sum_f64_small_vector_equals_running_sum() {
    let values = [1.0, 2.0, 3.0, 4.0, 5.0];
    let mut expected = 0.0_f64;
    for &v in &values {
        expected += v;
    }
    assert_eq!(sum_f64(&values), expected);
    assert_eq!(sum_f64(&values), 15.0);
}

/// `sum_f32_in_f64` widens each `f32` to `f64` and folds sequentially, returning
/// an `f64`.
#[test]
fn sum_f32_in_f64_accumulates_in_f64() {
    let values: [f32; 4] = [0.5, 1.5, 2.0, -1.0];
    assert_eq!(sum_f32_in_f64(&values), 3.0_f64);
}

/// `sum_f32_in_f64` of an empty slice is `0.0`.
#[test]
fn sum_f32_in_f64_empty_is_zero() {
    assert_eq!(sum_f32_in_f64(&[]), 0.0);
}

/// Scatter-adding a slice's values in ascending order into a zeroed one-element
/// accumulator is byte-for-byte identical to `sum_f64` of the same slice — the
/// scatter form is the same left-to-right `+=` fold, just addressed by index. The
/// adversarial `[1e16, 1.0, -1e16]` canary must reproduce `0.0` exactly.
#[test]
fn scatter_add_matches_sum_f64() {
    for values in [
        &[1.0_f64, 2.0, 3.0, 4.0, 5.0][..],
        &[1e16, 1.0, -1e16][..],
        &[][..],
        &[-2.5, 0.25, 7.0][..],
    ] {
        let mut acc = [0.0_f64];
        for &v in values {
            scatter_add_f64(&mut acc, 0, v);
        }
        assert_eq!(acc[0], sum_f64(values), "scatter fold must equal sum_f64");
    }
}

/// An out-of-range index is a defensive no-op: nothing is written, no panic, and
/// the in-range slots are untouched.
#[test]
fn scatter_add_out_of_range_is_noop() {
    let mut acc = [1.0_f64, 2.0];
    scatter_add_f64(&mut acc, 2, 99.0); // len == 2, idx 2 is OOB
    scatter_add_f64(&mut acc, usize::MAX, 99.0);
    assert_eq!(acc, [1.0, 2.0]);
    // A valid index still folds normally.
    scatter_add_f64(&mut acc, 1, 0.5);
    assert_eq!(acc, [1.0, 2.5]);
}

/// The `f64` accumulator preserves a small term that an `f32` accumulator would
/// lose, demonstrating the widening contract: `16_777_216_f32 + 1.0_f32` rounds
/// back to `16_777_216` in `f32`, but the running `f64` sum keeps the `1.0`.
#[test]
fn sum_f32_in_f64_widens_before_adding() {
    let values: [f32; 2] = [16_777_216.0, 1.0];
    assert_eq!(sum_f32_in_f64(&values), 16_777_217.0_f64);
}
