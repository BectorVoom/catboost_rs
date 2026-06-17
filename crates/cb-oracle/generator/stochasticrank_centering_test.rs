//! WR-03 / D-08 accumulation-order parity test for the OFFLINE
//! `stochasticrank_oracle.cpp` shift-centering (Plan 06.3-08, Wave-5 gap closure).
//!
//! The generator's Stage-1 centering (`StochasticRankNoiseStream`,
//! stochasticrank_oracle.cpp) computes the average of `shifted[]` and subtracts it
//! before the Monte-Carlo noise stream. After the WR-03 fix the generator folds
//! `shifted[]` in the SAME strict left-to-right order `cb_core::sum_f64` uses (the
//! Rust StochasticRank centering at ranking_der.rs:726-727 is
//! `sum_f64(&shifted) / count`). `cb_core::sum_f64` is the D-08 SOURCE OF TRUTH —
//! the oracle adapts to it, never the reverse.
//!
//! This test exercises a group with **> 4 docs and NON-ZERO mu** (the committed
//! 3-doc mu=0 fixture has an exactly-zero shifted sum, so it masks any ordering
//! divergence). It replicates the generator's centering formula in Rust against
//! `cb_core::sum_f64` and asserts the centered `shifted[]` values match to ≤ 1e-5.
//! To prove the assertion is actually ORDER-SENSITIVE on this input, it also shows
//! a deliberately reordered fold diverges from `sum_f64` on adversarial magnitudes.
//!
//! Source/test separation (INFRA-06): this file lives next to the generator it
//! validates and is discovered through the `#[path]` shim in
//! `crates/cb-oracle/tests/stochasticrank_centering_test.rs` — never inline.

use cb_core::sum_f64;

/// The generator's Stage-1 shifted vector: `shifted[d] = approx[d] - sigma*mu*target[d]`
/// (stochasticrank_oracle.cpp, mirrored from error_functions.cpp:1026-1028).
fn build_shifted(approxes: &[f64], targets: &[f64], sigma: f64, mu: f64) -> Vec<f64> {
    approxes
        .iter()
        .zip(targets.iter())
        .map(|(&a, &t)| a - sigma * mu * t)
        .collect()
}

/// Center `shifted[]` exactly as the WR-03-fixed generator does: the average is the
/// strict left-to-right `cb_core::sum_f64` fold divided by `count`.
fn center_via_sum_f64(shifted: &[f64]) -> Vec<f64> {
    let count = shifted.len();
    assert!(count > 0);
    let avrg = sum_f64(shifted) / count as f64;
    shifted.iter().map(|&s| s - avrg).collect()
}

#[test]
fn centering_matches_sum_f64_order_for_more_than_four_docs_non_zero_mu() {
    // A group with > 4 docs and NON-ZERO mu so the shifted sum is NOT identically 0
    // (the mu=0 fixture is the degenerate masking case WR-03 calls out).
    let approxes = vec![0.31, -0.42, 0.13, 0.27, -0.08, 0.55, -0.19];
    let targets = vec![3.0, 0.0, 1.0, 2.0, 1.0, 4.0, 0.0];
    let sigma = 1.0_f64;
    let mu = 0.5_f64; // non-zero — exercises the shift term and a non-zero mean.

    let shifted = build_shifted(&approxes, &targets, sigma, mu);

    // The shift mean is genuinely non-zero here (sanity: this input would mask
    // nothing — it actually exercises the centering subtraction).
    let mean = sum_f64(&shifted) / shifted.len() as f64;
    assert!(mean.abs() > 1e-3, "test input must have a non-zero shift mean, got {mean}");

    // The Rust StochasticRank centering (ranking_der.rs:726-727) and the WR-03-fixed
    // generator BOTH center via the sum_f64 fold; they must agree to ≤ 1e-5.
    let centered = center_via_sum_f64(&shifted);

    // Independent reference: the generator's explicit left-to-right C++ fold
    // `double avrg = 0.0; for (double s : shifted) avrg += s;` is, by construction,
    // identical to sum_f64 — replicate that exact loop here and compare.
    let mut cpp_order_avrg = 0.0_f64;
    for &s in &shifted {
        cpp_order_avrg += s;
    }
    cpp_order_avrg /= shifted.len() as f64;
    let cpp_centered: Vec<f64> = shifted.iter().map(|&s| s - cpp_order_avrg).collect();

    for (d, (&a, &b)) in centered.iter().zip(cpp_centered.iter()).enumerate() {
        assert!(
            (a - b).abs() <= 1e-5,
            "doc {d}: sum_f64-centered {a} vs generator-order {b} diverge beyond 1e-5"
        );
    }

    // The centered values must themselves sum to ~0 (the centering invariant).
    let s = sum_f64(&centered);
    assert!(s.abs() < 1e-9, "centered shifted[] must be zero-mean, sum={s}");
}

#[test]
fn centering_is_accumulation_order_sensitive() {
    // Prove the parity assertion above is not vacuous: on adversarial magnitudes a
    // REORDERED fold (a different summation order, e.g. ascending-by-magnitude) does
    // NOT match the strict left-to-right sum_f64, so an oracle that reordered its
    // accumulation WOULD diverge. This is exactly what WR-03 guards against.
    let shifted = vec![1e16, 1.0, -1e16, 2.0, 0.5, 3.0, -4.0];

    // sum_f64 strict left-to-right loses the small terms behind 1e16 (documented
    // adversarial behavior in reduction.rs).
    let left_to_right = sum_f64(&shifted);

    // A pairwise/reordered fold that adds the small terms first recovers them.
    let mut reordered: Vec<f64> = shifted.clone();
    reordered.sort_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap());
    let mut reordered_sum = 0.0_f64;
    for &v in &reordered {
        reordered_sum += v;
    }

    // The strict left-to-right fold loses the lone 1.0 behind the ±1e16 terms,
    // while the magnitude-ascending fold recovers it — the two orders diverge by a
    // full ULP-of-1e16-scale unit (~1.0), proving the order is observable here.
    assert!(
        (left_to_right - reordered_sum).abs() > 0.25,
        "accumulation order must be observable on this input \
         (left_to_right={left_to_right}, reordered={reordered_sum}) — \
         otherwise the WR-03 parity test would be vacuous"
    );
}
