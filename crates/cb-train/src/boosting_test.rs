//! Unit tests for the plain boosting loop's leaf-delta computation
//! ([`crate::boosting::compute_leaf_deltas`]), focused on the RESEARCH Pattern 3
//! Exact-alpha threading (Plan 06.1-03 / D-6.1-05): the Exact leaf branch must
//! thread the ACTIVE loss's `(alpha, delta)` into `exact_leaf_delta`, NOT the
//! hardcoded `QUANTILE_ALPHA` / `QUANTILE_DELTA` median constants.
//!
//! These are falsifiable regression catches: a revert of the threading (back to
//! the unconditional hardcoded 0.5) flips `quantile_alpha07_threads_alpha`.
//!
//! Dedicated test file (CLAUDE.md source/test separation — no inline
//! `#[cfg(test)]` in production source). Mounted via `#[path]` from `boosting.rs`,
//! so it can reach the private `compute_leaf_deltas`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use cb_compute::{exact_leaf_delta, LeafMethod, Loss, QUANTILE_ALPHA, QUANTILE_DELTA};

use super::compute_leaf_deltas;

/// Run the Exact-leaf branch of `compute_leaf_deltas` over a single leaf whose
/// per-member residuals are exactly `residuals` (we feed `approx = 0`, `target =
/// residuals`, so the internal `target - approx` recovers them), unit weights, and
/// return the single leaf delta. `der2`/`weighted_der1` are unused by the Exact
/// branch (it works off the residuals), so they are filled trivially.
fn exact_single_leaf(loss: Loss, residuals: &[f64]) -> f64 {
    exact_single_leaf_dim(loss, residuals, 0)
}

/// As [`exact_single_leaf`] but for a specific output dimension index `dim_index`
/// (the MultiQuantile per-dimension `alpha[dim_index]` selector).
fn exact_single_leaf_dim(loss: Loss, residuals: &[f64], dim_index: usize) -> f64 {
    let n = residuals.len();
    let leaf_of = vec![0_usize; n]; // every object in leaf 0.
    let weighted_der1 = vec![0.0_f64; n];
    let der2 = vec![0.0_f64; n];
    let weights = vec![1.0_f64; n];
    let approx = vec![0.0_f64; n];
    let target = residuals.to_vec();

    let deltas = compute_leaf_deltas(
        LeafMethod::Exact,
        &loss,
        &leaf_of,
        &weighted_der1,
        &der2,
        &weights,
        &approx,
        &target,
        /* scaled_l2 */ 0.0,
        /* n_leaves */ 1,
        dim_index,
    );
    assert_eq!(deltas.len(), 1);
    deltas[0]
}

#[test]
fn quantile_alpha07_threads_alpha_not_hardcoded_half() {
    // Residuals [1,2,3,4,5], unit weights: the weighted 0.5-quantile is 3, the
    // weighted 0.7-quantile is 4 (DISTINCT). If the Exact branch threaded the
    // active Quantile{0.7} alpha, the leaf delta is the 0.7-quantile; if it
    // regressed to the hardcoded 0.5, it would be the 0.5-quantile — so this is a
    // falsifiable threading catch.
    let residuals = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let alpha = 0.7;
    let delta = QUANTILE_DELTA;

    let delta_07 = exact_single_leaf(Loss::Quantile { alpha, delta }, &residuals);

    // Anchor: the alpha-general exact_leaf_delta at alpha=0.7 (leaf.rs UNCHANGED).
    let residuals_f32: Vec<f32> = residuals.iter().map(|&r| r as f32).collect();
    let weights = vec![1.0_f64; residuals.len()];
    let expected_07 = exact_leaf_delta(&residuals_f32, &weights, alpha, delta);
    assert!(
        (delta_07 - expected_07).abs() < 1e-12,
        "Exact branch must thread alpha=0.7: got {delta_07}, expected {expected_07}"
    );

    // Sanity: the 0.7-quantile differs from the 0.5-quantile here, so the test
    // genuinely distinguishes threaded-0.7 from hardcoded-0.5.
    let expected_05 = exact_leaf_delta(&residuals_f32, &weights, 0.5, delta);
    assert!(
        (expected_07 - expected_05).abs() > 0.5,
        "test corpus must separate the 0.7- and 0.5-quantiles (got 0.7={expected_07}, 0.5={expected_05})"
    );
}

#[test]
fn quantile_alpha05_equals_mae_exact_leaf() {
    // MAE == Quantile{alpha=0.5, delta=1e-6} at the Exact-leaf level: the threaded
    // Quantile{0.5} leaf delta must equal the Mae leaf delta (which threads the
    // hardcoded QUANTILE_ALPHA/QUANTILE_DELTA == 0.5/1e-6) bit-for-bit.
    let residuals = [-2.5_f64, 0.0, 1.0, 3.25, 7.0, -4.5];

    let mae_delta = exact_single_leaf(Loss::Mae, &residuals);
    let q05_delta = exact_single_leaf(
        Loss::Quantile {
            alpha: QUANTILE_ALPHA,
            delta: QUANTILE_DELTA,
        },
        &residuals,
    );
    assert_eq!(
        mae_delta, q05_delta,
        "MAE Exact leaf must equal Quantile{{0.5}} Exact leaf (byte-stable)"
    );
}

#[test]
fn multiquantile_threads_per_dimension_alpha() {
    // MultiQuantile (D-6.2-05): the Exact leaf for dimension `d` must thread
    // alpha[d] (each dimension is an independent quantile). With alpha=[0.3,0.7],
    // dimension 0 takes the weighted 0.3-quantile and dimension 1 the weighted
    // 0.7-quantile of the SAME residuals — DISTINCT values. A regression that used
    // a single fixed alpha (or alpha[0] for every dim) flips dimension 1.
    let residuals = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let delta = QUANTILE_DELTA;
    let alpha = vec![0.3_f64, 0.7];

    let d0 = exact_single_leaf_dim(
        Loss::MultiQuantile {
            alpha: alpha.clone(),
            delta,
        },
        &residuals,
        0,
    );
    let d1 = exact_single_leaf_dim(
        Loss::MultiQuantile {
            alpha: alpha.clone(),
            delta,
        },
        &residuals,
        1,
    );

    // Each dimension must equal the alpha-general exact_leaf_delta at its own alpha.
    let residuals_f32: Vec<f32> = residuals.iter().map(|&r| r as f32).collect();
    let weights = vec![1.0_f64; residuals.len()];
    let expected_d0 = exact_leaf_delta(&residuals_f32, &weights, 0.3, delta);
    let expected_d1 = exact_leaf_delta(&residuals_f32, &weights, 0.7, delta);
    assert!((d0 - expected_d0).abs() < 1e-12, "dim 0 must thread alpha[0]=0.3");
    assert!((d1 - expected_d1).abs() < 1e-12, "dim 1 must thread alpha[1]=0.7");
    assert!(
        (d0 - d1).abs() > 0.5,
        "the two quantile levels must produce distinct leaf deltas (got d0={d0}, d1={d1})"
    );
}

#[test]
fn multiquantile_alpha07_dimension_equals_scalar_quantile07_leaf() {
    // The degenerate-equivalence anchor at the leaf level (D-6.2-05): a
    // MultiQuantile dimension at alpha=0.7 must produce the SAME Exact leaf delta as
    // the scalar Quantile{0.7} path (leaf.rs reused verbatim per dimension).
    let residuals = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let delta = QUANTILE_DELTA;

    let mq = exact_single_leaf_dim(
        Loss::MultiQuantile {
            alpha: vec![0.7],
            delta,
        },
        &residuals,
        0,
    );
    let scalar = exact_single_leaf(Loss::Quantile { alpha: 0.7, delta }, &residuals);
    assert_eq!(
        mq, scalar,
        "MultiQuantile{{[0.7]}} dimension-0 leaf must equal scalar Quantile{{0.7}} leaf"
    );
}
