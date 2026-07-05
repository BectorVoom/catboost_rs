//! The single sanctioned deterministic float-summation primitive (DATA-07 /
//! D-07). Every float sum anywhere in catboost-rs MUST route through this
//! module; all other raw `.sum()` / `.fold(0.0, +)` over floats in library
//! crates are banned by `scripts/check-no-raw-float-sum.sh` (D-08).
//!
//! # Order is the contract
//!
//! Upstream CatBoost accumulates weights and class totals as a plain
//! left-to-right `double` running sum under `thread_count == 1` (one block, no
//! per-thread partials):
//!
//! - `library/cpp/grid_creator/binarization.cpp:803-815` — `totalWeight +=
//!   weight` iterated over the sorted values.
//! - `private/libs/target/calc_class_weights.cpp:36-54` — a `double` per-block
//!   accumulation where, under `thread_count == 1`, `blocks == 1`, so the sum
//!   degenerates to one sequential fold.
//!
//! Contract: **sequential `f64` fold, `thread_count == 1`, NO compensated
//! summation** (no Kahan, no pairwise, no `.sum()`/`.fold(0.0, +)`). Any
//! reordering of additions perturbs the result on adversarial inputs and breaks
//! the ≤ 1e-5 oracle gate everywhere downstream (RESEARCH Pitfall 1, threat
//! T-02-01). This file is the *only* place a hand-written summation loop is
//! allowed to exist.

/// Sum `f64` values in strict left-to-right order into an `f64` accumulator.
///
/// This is the naive sequential reduction — it deliberately does **not** use
/// Kahan or pairwise summation. On the adversarial input `[1e16, 1.0, -1e16]`
/// it returns `0.0` (the running sum loses the `1.0`), exactly as a single-block
/// upstream `double` accumulation does.
#[must_use]
pub fn sum_f64(values: &[f64]) -> f64 {
    let mut acc = 0.0_f64;
    for &v in values {
        acc += v;
    }
    acc
}

/// The sanctioned SCATTER form of the same left-to-right `f64` fold as
/// [`sum_f64`]: fold `value` into the accumulator slot at `idx` with a single
/// sequential `+=`.
///
/// Folding a cell's members via repeated `scatter_add_f64` in ascending object
/// order yields **byte-identical** bits to `sum_f64(&members)` — it is the exact
/// same naive running sum, just addressed by slot index instead of gathered into a
/// contiguous slice first. It exists so histogram accumulation can honor the D-08
/// "no raw float fold outside this file" ban WITHOUT allocating a per-cell `Vec`
/// (PERF-03): the scatter-add builder holds one flat accumulator and folds directly
/// into it. Like [`sum_f64`] there is NO compensation (no Kahan/pairwise); the
/// order is the parity contract.
///
/// An out-of-range `idx` is a defensive no-op (no raw indexing — workspace deny
/// `indexing_slicing`; the caller guarantees valid indices, a mismatch is a bug,
/// not a panic condition).
pub fn scatter_add_f64(acc: &mut [f64], idx: usize, value: f64) {
    if let Some(slot) = acc.get_mut(idx) {
        *slot += value;
    }
}

/// Sum `f32` values, accumulating in an `f64` accumulator, in strict
/// left-to-right order. Each `f32` is widened to `f64` before being added, then
/// folded sequentially — matching upstream's `double`-accumulator weight sums
/// over `float` columns. No compensated summation.
#[must_use]
pub fn sum_f32_in_f64(values: &[f32]) -> f64 {
    let mut acc = 0.0_f64;
    for &v in values {
        acc += f64::from(v);
    }
    acc
}
