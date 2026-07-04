//! Serial self-oracle for the Phase 12 Plan 04 (GPUT-18, D-03a) device Region grow
//! (`kernels::region_device::grow_region_tree`). Runs the SAME pinned fixture that Plan 02's
//! `cb_train::region_grow_test.rs` froze as the ≤1e-5 CPU Region reference through the DEVICE
//! Region path and asserts:
//!
//! - PATH STRUCTURE is EXACT: the per-level `(feature, bin, expected_direction, one_hot)`
//!   `region_path` and the per-object terminal bin `leaf_of` match the frozen CPU Region
//!   reference bit-for-bit (a depth-`d` Region has EXACTLY `d + 1` leaves — the `2^d` failure
//!   signal for the "Region is a node graph" bug is asserted against).
//! - LEAF VALUES match within ε=1e-4 (`max_divergence`) vs the frozen `calc_average` reference
//!   (transcribed inline — cb-backend cannot `use cb_train`, the feature-unification landmine).
//!
//! The frozen fixture (Plan 02 SUMMARY): `f0` bins `[0,0,1,1,2,2]` (borders `[0.5, 1.5]`), `f1`
//! bins `[0,1,0,1,0,1]` (unused by the grown path), der1 `[-2,-2,0,0,3,3]`, unit weights,
//! `scaled_l2 = 0`, Cosine score. The grown path: level 0 `f0 > 1.5` continue=`false` (peels the
//! `+3` pair into bin 0), level 1 `f0 > 0.5` continue=`true` (bin 1 = `{o0,o1}`), survivors
//! `{o2,o3}` → bin 2. Depth 2, 3 leaves, `leaf_of = [1,1,2,2,0,0]`, leaf values `[3, -2, 0]`.
//!
//! Runs over `SelectedRuntime`, but — like the non-sym grow oracle — the cubecl-cpu backend
//! cannot JIT the per-frontier score/argmin over these subset shapes (an `elem.rs` visitor
//! panic), so it SKIPS on cpu/wgpu and validates on the real device in-env (rocm gfx1100), the
//! WR-01 anti-false-pass convention. Kaggle CUDA ε=1e-4 sign-off is deferred to Plan 09.

use cb_compute::calc_average;

use crate::kernels::region_device::grow_region_tree;
use crate::kernels::SCORE_FN_COSINE;

/// Max abs / rel divergence over two equal-length buffers (the `grow_loop::max_divergence`
/// reporter shape). A length mismatch yields a sentinel infinite divergence (WR-06).
fn max_divergence(device: &[f64], baseline: &[f64]) -> (f64, f64) {
    if device.len() != baseline.len() {
        return (f64::INFINITY, f64::INFINITY);
    }
    let mut max_abs = 0.0_f64;
    let mut max_rel = 0.0_f64;
    for (&d, &b) in device.iter().zip(baseline) {
        let abs = (d - b).abs();
        let rel = if b.abs() > 0.0 { abs / b.abs() } else { abs };
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
    }
    (max_abs, max_rel)
}

/// The pinned Plan-02 Region fixture as feature-major quantized bins.
fn fixture() -> (Vec<f64>, Vec<f64>, Vec<u32>) {
    // der1 = target - approx (RMSE, from-zero): [-2,-2,0,0,3,3].
    let der1 = vec![-2.0_f64, -2.0, 0.0, 0.0, 3.0, 3.0];
    let weight = vec![1.0_f64; 6];
    // cindex feature-major: f0 bins then f1 bins (n=6, n_features=2). The bin VALUES are
    // {0,1,2} (f0) / {0,1} (f1); n_bins is padded to 32 below because the device
    // `pointwise_hist2` fill only dispatches line sizes {2,32,64,128,256} — the empty upper
    // buckets contribute nothing, so the argmax picks the SAME frozen splits.
    let cindex: Vec<u32> = vec![
        0, 0, 1, 1, 2, 2, // f0
        0, 1, 0, 1, 0, 1, // f1
    ];
    (der1, weight, cindex)
}

#[test]
fn region_device_reproduces_frozen_cpu_region_path() {
    // The device split scorer runs real GPU kernels; the cubecl-cpu backend cannot JIT the
    // per-frontier score/argmin over these subset shapes, so SKIP on cpu/wgpu and validate on
    // the real device in-env (rocm gfx1100) — the WR-01 anti-false-pass convention shared with
    // the non-sym grow oracle. Kaggle CUDA ε sign-off is Plan 09's.
    if !cfg!(any(feature = "rocm", feature = "cuda")) {
        println!("[region] SKIP: device Region grow needs a real GPU backend (rocm/cuda)");
        return;
    }
    const EPS: f64 = 1e-4;
    let (der1, weight, cindex) = fixture();
    let n = 6usize;
    // Padded to a device-dispatchable line size (empty upper buckets, same argmax).
    let n_bins = 32usize;
    let n_features = 2usize;
    let max_depth = 3usize;
    let min_data_in_leaf = 1usize;
    let scaled_l2 = 0.0_f64;

    let dev = grow_region_tree(
        &der1, &weight, &cindex, n, n_bins, n_features, max_depth, min_data_in_leaf, scaled_l2,
        SCORE_FN_COSINE,
    )
    .expect("device Region grow must succeed on the frozen Plan-02 fixture");

    // (A) PATH STRUCTURE — EXACT vs the frozen CPU Region reference. Level 0: f0 > 1.5 (bin 1),
    // continue = false; level 1: f0 > 0.5 (bin 0), continue = true. Both float splits (one_hot
    // false). Depth 2 → 3 leaves (NEVER 2^depth == 4, the node-graph failure signal).
    assert_eq!(
        dev.region_path,
        vec![(0u32, 1u32, false, false), (0u32, 0u32, true, false)],
        "device Region path must match the frozen CPU Region reference (per-level feature/bin/direction/one_hot)"
    );
    assert_eq!(dev.region_path.len(), 2, "depth-2 region has 2 path levels");
    assert_eq!(
        dev.leaf_values.len(),
        dev.region_path.len() + 1,
        "a depth-d Region has EXACTLY d+1 leaves, never 2^d (node-graph failure signal)"
    );
    assert_eq!(dev.leaf_values.len(), 3);

    // Region is a PATH, NOT a node graph — the non-symmetric carrier must stay empty.
    assert!(dev.step_nodes.is_empty(), "Region must not emit a node graph");
    assert!(dev.node_id_to_leaf_id.is_empty(), "Region must not emit a node-graph leaf map");

    // (B) PER-OBJECT TERMINAL BIN — EXACT vs the frozen `leaf_of = [1,1,2,2,0,0]`.
    assert_eq!(
        dev.leaf_of,
        vec![1u32, 1, 2, 2, 0, 0],
        "device Region per-object terminal bin must match the frozen CPU walk"
    );

    // (C) LEAF VALUES — within ε=1e-4 vs the frozen `calc_average` reference. Bin order:
    // bin0 = {o4,o5} der[3,3], bin1 = {o0,o1} der[-2,-2], bin2 = {o2,o3} der[0,0].
    let expected_leaf_values = vec![
        calc_average(6.0, 2.0, scaled_l2),  // bin 0: sum der = 3+3
        calc_average(-4.0, 2.0, scaled_l2), // bin 1: sum der = -2-2
        calc_average(0.0, 2.0, scaled_l2),  // bin 2: sum der = 0+0
    ];
    let (abs, rel) = max_divergence(&dev.leaf_values, &expected_leaf_values);
    println!(
        "[region] depth={} leaves={}; leaf-value max abs_div={abs:.3e} rel_div={rel:.3e} (bar={EPS:.0e})",
        dev.region_path.len(),
        dev.leaf_values.len(),
    );
    assert!(
        abs <= EPS || rel <= EPS,
        "device Region leaf values exceeded ε=1e-4: abs={abs:.3e} rel={rel:.3e}"
    );
}
