//! Plan 11-01 Task 2 — CPU oracle cross-check of the depth-6 correctness fixture.
//!
//! Proves the offline generator's `bench/fixtures/expected_depth6_tree.json` is
//! bit-consistent (≤1e-5) with the cb-compute CPU oracle that the Phase-11 device
//! path (Plans 02–04) must match at ε=1e-4. This is the Wave-0 test foundation:
//! a WRONG fixture (silent bad reference, threat T-11-01-01) fails here.
//!
//! Method (self-contained, no `.npy` parser / no X routing in Rust): the generator
//! emits, per arm, the per-object `leaf_of` / `der1` / `weight` / `weighted_der2`
//! arrays it derived from the 6-level split sequence. This test reduces them with
//! `reduce_leaf_stats` (Σder1 / Σweight) and `reduce_leaf_der2` (Σ(der2·weight)) in
//! the canonical object order (D-05), recomputes each leaf value with `calc_average`
//! (RMSE) / `newton_leaf_delta` (Logloss) using `scale_l2_reg` for the regularizer,
//! and asserts it matches the fixture's stored leaf value to ≤1e-5.
//!
//! # A1 / A2 assumption locks
//! - **A1** — the fixture MUST pin `leaf_estimation_iterations == 1` (single
//!   closed-form Newton step; the CPU oracle has no iterative walker). Asserted below.
//! - **A2** — the split score is the Cosine function with channel-0 == Σweight (NOT
//!   Σder2); the Logloss Newton hessian enters only the leaf value. Asserted below so
//!   a later device change to der2-in-score surfaces as a failure, not silent drift.
//!
//! Integration test under `tests/` (source/test separation, CLAUDE.md — NO inline
//! `#[cfg(test)] mod tests` in any production source file).

// Test-only: the workspace restriction lints (unwrap/expect/panic/indexing) are
// denied in production but exempted in test code (matches custom_objective_test.rs).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use cb_compute::{
    calc_average, newton_leaf_delta, reduce_leaf_der2, reduce_leaf_stats, scale_l2_reg,
};
use serde_json::Value;

/// The committed depth-6 fixture, embedded at compile time (path relative to THIS
/// source file: `crates/cb-compute/tests/` → repo root → `bench/fixtures/`).
const FIXTURE: &str = include_str!("../../../bench/fixtures/expected_depth6_tree.json");

/// Parity tolerance — the ≤1e-5 CPU-oracle bar (the device ε=1e-4 bar is looser still).
const TOL: f64 = 1e-5;

/// The `l2_leaf_reg` the fixture was generated with (CatBoost default 3.0).
const L2: f64 = 3.0;

fn f64_vec(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("f64 element"))
        .collect()
}

fn usize_vec(v: &Value) -> Vec<usize> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_u64().expect("u64 element") as usize)
        .collect()
}

#[test]
fn depth6_fixture_matches_cpu_oracle() {
    let doc: Value = serde_json::from_str(FIXTURE).expect("parse depth-6 fixture json");
    let config = &doc["config"];

    // --- A1: single closed-form Newton step pinned in the fixture. ---
    let iters = config["leaf_estimation_iterations"]
        .as_u64()
        .expect("config.leaf_estimation_iterations");
    assert_eq!(
        iters, 1,
        "A1 violated: fixture must pin leaf_estimation_iterations == 1 (single \
         closed-form Newton step; the cb-compute oracle has no iterative walker)"
    );

    // --- A2: Cosine score with channel-0 == Σweight pinned in the fixture. ---
    assert_eq!(
        config["score_function"].as_str(),
        Some("Cosine"),
        "A2 violated: fixture score_function must be pinned to Cosine"
    );
    assert_eq!(
        config["score_channel0"].as_str(),
        Some("sum_weight"),
        "A2 violated: fixture split score channel-0 must be Σweight (NOT Σder2); the \
         Logloss Newton hessian enters only the leaf value"
    );

    // Both arms carry a full 6-level oblivious split sequence.
    assert_eq!(config["depth"].as_u64(), Some(6), "fixture depth must be 6");
    assert_eq!(
        doc["rmse"]["splits"].as_array().map(Vec::len),
        Some(6),
        "RMSE arm must have a 6-level split sequence"
    );
    assert_eq!(
        doc["logloss"]["splits"].as_array().map(Vec::len),
        Some(6),
        "Logloss arm must have a 6-level split sequence"
    );

    check_arm(&doc["rmse"], false);
    check_arm(&doc["logloss"], true);
}

/// Reduce the arm's per-object arrays with the cb-compute oracle and assert every
/// recomputed leaf value matches the fixture's stored leaf value to ≤1e-5.
///
/// `newton == true` uses `reduce_leaf_der2` + `newton_leaf_delta` (Logloss);
/// otherwise `calc_average` on `reduce_leaf_stats` (RMSE).
fn check_arm(arm: &Value, newton: bool) {
    let leaf_of = usize_vec(&arm["leaf_of"]);
    let der1 = f64_vec(&arm["der1"]);
    let weight = f64_vec(&arm["weight"]);
    let weighted_der2 = f64_vec(&arm["weighted_der2"]);
    let leaf_values = f64_vec(&arm["leaf_values"]);
    let n_leaves = arm["n_leaves"].as_u64().expect("n_leaves") as usize;
    let stored_scaled_l2 = arm["scaled_l2"].as_f64().expect("scaled_l2");

    assert_eq!(n_leaves, 1 << 6, "depth-6 arm must have 64 leaves");
    assert_eq!(leaf_values.len(), n_leaves, "one stored value per leaf");
    assert_eq!(der1.len(), leaf_of.len(), "der1 parallel to leaf_of");
    assert_eq!(weight.len(), leaf_of.len(), "weight parallel to leaf_of");
    assert_eq!(
        weighted_der2.len(),
        leaf_of.len(),
        "weighted_der2 parallel to leaf_of"
    );

    // The scaled L2 the oracle applies — must match the fixture's stored value
    // (scale_l2_reg == l2 for the unit-weight path, Σweight == doc_count).
    let total_weight: f64 = weight.iter().sum();
    let scaled_l2 = scale_l2_reg(L2, total_weight, weight.len());
    assert!(
        (scaled_l2 - stored_scaled_l2).abs() <= 1e-9,
        "scaled_l2 mismatch: oracle {scaled_l2} vs fixture {stored_scaled_l2}"
    );

    // Reduce Σder1 / Σweight per leaf in canonical object order (D-05).
    let stats = reduce_leaf_stats(&leaf_of, &der1, &weight, n_leaves);

    if newton {
        // Σ(der2·weight) per leaf — the Newton denominator channel.
        let sum_der2 = reduce_leaf_der2(&leaf_of, &weighted_der2, n_leaves);
        for leaf in 0..n_leaves {
            let got = newton_leaf_delta(stats[leaf].sum_weighted_delta, sum_der2[leaf], scaled_l2);
            let want = leaf_values[leaf];
            assert!(
                (got - want).abs() <= TOL,
                "Logloss leaf {leaf}: newton_leaf_delta={got} vs fixture={want} \
                 (Δ={})",
                (got - want).abs()
            );
        }
    } else {
        for leaf in 0..n_leaves {
            let got = calc_average(stats[leaf].sum_weighted_delta, stats[leaf].sum_weight, scaled_l2);
            let want = leaf_values[leaf];
            assert!(
                (got - want).abs() <= TOL,
                "RMSE leaf {leaf}: calc_average={got} vs fixture={want} (Δ={})",
                (got - want).abs()
            );
        }
    }
}
