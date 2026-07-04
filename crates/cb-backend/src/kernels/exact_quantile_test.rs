//! Self-oracle for the device Exact weighted-quantile leaf delta (Phase 12 Plan 05, GPUT-19,
//! D-09): [`crate::kernels::exact_quantile::device_exact_leaf_delta`] must match the ≤1e-5 CPU
//! reference `cb_compute::exact_leaf_delta` within ε=1e-4 for the Quantile / MAE / MAPE family
//! — the order-statistic leaf method, DISTINCT from the Newton der2 path (Pitfall 6).
//!
//! Source/test separation is mandatory (CLAUDE.md / AGENTS.md): the device pipeline lives in
//! the production `kernels::exact_quantile` module; ALL assertions / `.unwrap()` / indexing
//! live here. The oracle is the ACTUAL CPU reference `cb_compute::exact_leaf_delta` (a normal
//! cb-backend dep — the landmine is cb-TRAIN, not cb-compute; the PRODUCTION module transcribes
//! its semantics inline, this TEST calls it directly as ground truth).
//!
//! # A4 (the Exact objective set) — CONFIRMED against `leaf.rs`
//!
//! `cb_compute::leaf.rs` routes MAE and Quantile{alpha,delta} through the SAME
//! `exact_leaf_delta` (the weighted alpha-quantile). MAPE is the SAME weighted quantile with
//! the caller's `weightsWithTargets[i] = weights[i]/max(1,|target[i]|)` transform applied
//! host-side (upstream `ComputeWeightsWithTargets`) — so all three of {Quantile, MAE, MAPE}
//! reduce to `exact_leaf_delta`, and the fixtures below exercise each by feeding the
//! appropriate (residuals, weights) pair. No MAPE-specific optimum path exists (A4 resolved).
//!
//! Runs over the generic [`crate::SelectedRuntime`]. Like the `kernels::sort` /
//! `kernels::segmented_sort_test` oracles the multi-kernel device sort/scan composition is
//! validated on ROCm in-env (gfx1100 wave32); the cpu backend cannot execute the composition
//! (documented, same by-design limitation as the sort oracle).

use crate::kernels::exact_quantile::device_exact_leaf_delta;

/// The ε=1e-4 device-vs-CPU bar (D-09; looser than the CPU ref's own ≤1e-5, per the GPU bar).
const EXACT_TOL: f64 = 1e-4;

/// Max abs divergence between the device Exact leaf delta and the CPU `exact_leaf_delta`
/// (single scalar per leaf — the `grow_loop::max_divergence` reporter shape, D-7.5-05).
fn max_divergence(device: f64, baseline: f64) -> f64 {
    (device - baseline).abs()
}

/// Run BOTH the device pipeline and the CPU reference on the same (residuals, weights, α, δ)
/// and assert the abs divergence is within ε=1e-4. Returns the device value for logging.
fn assert_exact(residuals: &[f32], weights: &[f64], alpha: f64, delta: f64, label: &str) -> f64 {
    let device = device_exact_leaf_delta(residuals, weights, alpha, delta).unwrap();
    let baseline = cb_compute::exact_leaf_delta(residuals, weights, alpha, delta);
    let div = max_divergence(device, baseline);
    println!("[exact {label}] device={device:.8} cpu={baseline:.8} abs_div={div:.3e}");
    assert!(
        div <= EXACT_TOL,
        "device Exact leaf delta diverged from exact_leaf_delta for {label}: \
         device={device} cpu={baseline} abs_div={div:.3e} > {EXACT_TOL:.0e}"
    );
    device
}

#[test]
fn exact_quantile_median_unit_weights_matches_cpu() {
    // MAE / Quantile{0.5}: unit weights, odd count → the median residual.
    assert_exact(&[2.0, -3.0, -1.0], &[1.0, 1.0, 1.0], 0.5, 1e-6, "median odd");
    // Even count → the linear-search first-crossing (upstream picks the crossing element).
    assert_exact(&[4.0, -2.0, 1.0, -5.0], &[1.0, 1.0, 1.0, 1.0], 0.5, 1e-6, "median even");
}

#[test]
fn exact_quantile_weighted_matches_cpu() {
    // Weighted quantile: the heavy element pulls the median toward it.
    assert_exact(&[5.0, 0.0], &[1.0, 3.0], 0.5, 1e-6, "weighted 2-elem");
    assert_exact(&[10.0, -4.0, 2.0, 7.0], &[0.5, 2.0, 1.0, 3.0], 0.5, 1e-6, "weighted 4-elem");
}

#[test]
fn exact_quantile_nonmedian_alpha_matches_cpu() {
    // Quantile{alpha != 0.5}: the α=0.25 and α=0.75 order statistics.
    let residuals = [3.0f32, -1.0, 5.0, -4.0, 2.0, 0.0, -2.0];
    let weights = [1.0f64; 7];
    assert_exact(&residuals, &weights, 0.25, 1e-6, "alpha 0.25");
    assert_exact(&residuals, &weights, 0.75, 1e-6, "alpha 0.75");
    assert_exact(&residuals, &weights, 0.9, 1e-4, "alpha 0.9 delta 1e-4");
}

#[test]
fn exact_quantile_mape_weights_with_targets_matches_cpu() {
    // MAPE (A4): the weighted median with weightsWithTargets[i] = weight_i / max(1, |target_i|).
    // Build targets/approx, form the MAPE weights host-side, feed BOTH paths the SAME pair.
    let targets = [12.0f64, 0.5, -8.0, 3.0, 20.0];
    let approx = [10.0f64, 1.0, -5.0, 2.0, 18.0];
    let residuals: Vec<f32> = targets
        .iter()
        .zip(approx.iter())
        .map(|(&t, &a)| (t - a) as f32)
        .collect();
    let weights: Vec<f64> = targets.iter().map(|&t| 1.0 / f64::max(1.0, t.abs())).collect();
    assert_exact(&residuals, &weights, 0.5, 1e-6, "mape weighted median");
}

#[test]
fn exact_quantile_edge_cases_match_cpu() {
    // Empty leaf → 0.0 (CalcSampleQuantile empty guard).
    let empty: [f32; 0] = [];
    let empty_w: [f64; 0] = [];
    let dev = device_exact_leaf_delta(&empty, &empty_w, 0.5, 1e-6).unwrap();
    assert_eq!(dev, 0.0, "empty leaf must be 0.0");
    assert_eq!(dev, cb_compute::exact_leaf_delta(&empty, &empty_w, 0.5, 1e-6));

    // alpha <= 0 → the min residual (CalcSampleQuantile:113-115).
    assert_exact(&[5.0, -2.0, 3.0], &[1.0, 1.0, 1.0], 0.0, 1e-6, "alpha zero -> min");

    // Single element → that element (± delta adjustment matching the CPU ref).
    assert_exact(&[7.5], &[1.0], 0.5, 1e-6, "single element");

    // Duplicate residuals (equal-weight ties exercise the less/equal delta test).
    assert_exact(&[2.0, 2.0, 2.0, 5.0], &[1.0, 1.0, 1.0, 1.0], 0.5, 1e-6, "duplicate residuals");
}

#[test]
fn exact_quantile_larger_leaf_matches_cpu() {
    // A larger leaf (n >> CUBE_DIM) with varied signs/magnitudes and non-uniform weights.
    let n = 200usize;
    let residuals: Vec<f32> = (0..n)
        .map(|k| ((k as f32) * 0.37).sin() * 12.0 - 3.0)
        .collect();
    let weights: Vec<f64> = (0..n).map(|k| 0.5 + ((k % 5) as f64) * 0.4).collect();
    assert_exact(&residuals, &weights, 0.5, 1e-6, "large median");
    assert_exact(&residuals, &weights, 0.3, 1e-6, "large alpha 0.3");
}
