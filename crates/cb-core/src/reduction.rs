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
