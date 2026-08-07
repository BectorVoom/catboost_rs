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

/// GDC-04 (T06): the SESSION-level weighted Region oracle. The weighted-der
/// substitution lives in `GpuTrainSession::grow_one`'s Region arm (caller-side
/// `w·der1`; `grow_region_tree` itself is deliberately untouched), so this drives
/// the SESSION with a NON-uniform weight and asserts it reproduces a DIRECT
/// `grow_region_tree` call fed the weighted product — and that the result genuinely
/// differs from the raw-der grow (the discriminator a pre-fix session fails).
#[test]
fn region_session_weighted_feeds_weighted_der() {
    if !cfg!(any(feature = "rocm", feature = "cuda")) {
        println!("[region-weighted] SKIP: device Region grow needs a real GPU backend (rocm/cuda)");
        return;
    }
    let (der1, _unit_weight, cindex) = fixture();
    let weight = vec![1.0_f64, 2.0, 1.0, 2.0, 1.0, 2.0];
    assert!(weight.iter().any(|&w| (w - 1.0).abs() > 1e-12));
    let n = 6usize;
    let n_bins = 32usize;
    let n_features = 2usize;
    let max_depth = 3usize;
    let min_data_in_leaf = 1usize;
    let scaled_l2 = 0.0_f64;

    // Session: RMSE from a zero approx ⇒ der1 == target, Region policy.
    let target = der1.clone();
    let config = cb_compute::DeviceTrainConfig {
        grow_policy: cb_compute::DeviceGrowPolicy::Region,
        min_data_in_leaf,
        ..cb_compute::DeviceTrainConfig::default()
    };
    let mut session = crate::gpu_runtime::GpuTrainSession::begin(
        &cb_compute::Loss::Rmse,
        max_depth,
        true,
        1,
        cb_compute::EScoreFunction::Cosine,
        &cindex,
        &weight,
        n,
        n_features,
        n_bins,
        0.3,
        scaled_l2,
        &config,
    )
    .expect("begin must not error on a covered Region config")
    .expect("a covered Region config must open a session");
    let dev = session
        .grow_one(&vec![0.0_f64; n], &target, &[])
        .expect("session Region grow must succeed");

    // Reference: the SAME grower fed the weighted product directly.
    let weighted: Vec<f64> = der1.iter().zip(weight.iter()).map(|(&d, &w)| d * w).collect();
    let expected = grow_region_tree(
        &weighted, &weight, &weighted, &weight, &cindex, n, n_bins, n_features, max_depth,
        min_data_in_leaf, scaled_l2, SCORE_FN_COSINE,
    )
    .expect("direct weighted Region grow must succeed");

    assert_eq!(dev.region_path, expected.region_path, "session must grow the weighted path");
    assert_eq!(dev.leaf_of, expected.leaf_of, "session must route by the weighted path");
    let (abs, _rel) = max_divergence(&dev.leaf_values, &expected.leaf_values);
    assert!(
        abs <= 1e-12,
        "session Region leaf values must equal the weighted direct grow (abs={abs:.3e})"
    );

    // Discriminator: the RAW-der grow (what a pre-fix session computed) must differ
    // in leaf values — otherwise this fixture cannot detect the regression.
    let raw = grow_region_tree(
        &der1, &weight, &der1, &weight, &cindex, n, n_bins, n_features, max_depth,
        min_data_in_leaf, scaled_l2, SCORE_FN_COSINE,
    )
    .expect("raw-der Region grow must succeed");
    let (raw_abs, _r) = max_divergence(&dev.leaf_values, &raw.leaf_values);
    assert!(
        raw_abs > 1e-4,
        "the weighted and raw-der Region grows coincide (abs={raw_abs:.3e}) — pick a \
         fixture where the weight matters"
    );
    drop(session);
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
        &der1, &weight, &der1, &weight, &cindex, n, n_bins, n_features, max_depth,
        min_data_in_leaf, scaled_l2, SCORE_FN_COSINE,
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

// ─── FPP-12 (T08): the host bootstrap sample reaches the Region grower's SCORE channels ──

/// FPP-12: with a length-`n` `sample`, the Region device grower scores splits over
/// `der1 ⊙ sample` / `weight ⊙ sample` while LEAF values keep using the UNSAMPLED channels
/// — `Runtime::grow_tree_on_device`'s contract verbatim.
///
/// # Why the reference multiplies twice
///
/// The `der1` that reaches `grow_region_tree` has already been through
/// `host_weighted_der1`, so on a weighted × sampled fit the score channel is
/// `w · der1 · s` and the score weight is `w · s`. That mirrors the oblivious resident
/// arm's nested `fold_weights_resident(fold_weights_resident(der1, weight), sample)`. A
/// reference that multiplied once would be chasing a phantom.
#[test]
fn region_device_matches_cpu_with_nontrivial_sample() {
    if !cfg!(any(feature = "rocm", feature = "cuda")) {
        println!("[region-sampled] SKIP: device Region grow needs a real GPU backend (rocm/cuda)");
        return;
    }
    const EPS: f64 = 1e-4;
    // NOT the 6-object frozen Plan-02 fixture: its der1 is CONSTANT within every terminal
    // bin, so a weighted leaf average equals the plain one and assertion (B) below can
    // never fire. This test needs der1 to VARY inside a leaf, so it builds a wider ramp
    // fixture from the shared primitives instead.
    let n = 64usize;
    let n_features = 3usize;
    let n_bins = 32usize;
    let max_depth = 3usize;
    let min_data_in_leaf = 1usize;
    let der1 = crate::kernels::test_fixtures::ramp_centred(n);
    let weight = crate::kernels::test_fixtures::weight_mod5(n);
    let cindex = crate::kernels::test_fixtures::cindex_feature_major(n, n_features, n_bins);
    let scaled_l2 = cb_compute::scale_l2_reg(3.0, cb_core::sum_f64(&weight), n);
    // A non-trivial multiplier: ~30% DROPPED objects (0.0) plus up- and down-weighted
    // ones. The zeros are the discriminating part — a dropped object must contribute
    // NOTHING to any split histogram, so a grower that ignored the sample picks a
    // different split.
    let sample: Vec<f64> = (0..n)
        .map(|i| match i % 10 {
            0 | 3 | 7 => 0.0,
            1 | 4 => 0.5,
            2 | 5 | 8 => 1.0,
            _ => 2.0,
        })
        .collect();
    assert!(
        sample.iter().any(|&s| s == 0.0) && sample.iter().any(|&s| s > 1.0),
        "the sample must have both dropped and up-weighted objects or this is vacuous"
    );

    let score_der1: Vec<f64> = der1.iter().zip(sample.iter()).map(|(&d, &s)| d * s).collect();
    let score_weight: Vec<f64> =
        weight.iter().zip(sample.iter()).map(|(&w, &s)| w * s).collect();

    let sampled = grow_region_tree(
        &der1, &weight, &score_der1, &score_weight, &cindex, n, n_bins, n_features, max_depth,
        min_data_in_leaf, scaled_l2, SCORE_FN_COSINE,
    )
    .expect("sampled Region grow must succeed");

    // (A) The STRUCTURE must be the one a grower fed the SAMPLED channels for BOTH roles
    // would pick — i.e. the sample genuinely reached the scorer.
    let all_sampled = grow_region_tree(
        &score_der1, &score_weight, &score_der1, &score_weight, &cindex, n, n_bins, n_features,
        max_depth, min_data_in_leaf, scaled_l2, SCORE_FN_COSINE,
    )
    .expect("all-sampled Region grow must succeed");
    assert_eq!(
        sampled.region_path, all_sampled.region_path,
        "the split path must be decided by the SAMPLED score channels"
    );
    assert_eq!(sampled.leaf_of, all_sampled.leaf_of, "routing follows the sampled path");

    // …and (A) is only meaningful if the sampled path DIFFERS from the unsampled one.
    // Without this, a fixture where both paths coincide would pass (A) vacuously — the
    // sample could have been dropped on the floor.
    let unsampled = grow_region_tree(
        &der1, &weight, &der1, &weight, &cindex, n, n_bins, n_features, max_depth,
        min_data_in_leaf, scaled_l2, SCORE_FN_COSINE,
    )
    .expect("unsampled Region grow must succeed");
    assert_ne!(
        sampled.region_path, unsampled.region_path,
        "the sampled and unsampled Region paths coincide — the sample never reached the          scorer, or this fixture cannot detect it"
    );

    // (B) The LEAF VALUES must come from the UNSAMPLED channels, so they must DIFFER from
    // the all-sampled grow — that difference IS the contract.
    let (leaf_abs, _r) = max_divergence(&sampled.leaf_values, &all_sampled.leaf_values);
    assert!(
        leaf_abs > EPS,
        "sampled and all-sampled leaf values coincide (abs={leaf_abs:.3e}) — leaf \
         estimation must NOT see the sample; pick a fixture where it matters"
    );

    // (C) …and they must equal `calc_average` over each terminal bin's RAW der/weight,
    // computed independently here.
    let mut bins: Vec<Vec<usize>> = vec![Vec::new(); sampled.leaf_values.len()];
    for (obj, &bin) in sampled.leaf_of.iter().enumerate() {
        if let Some(slot) = bins.get_mut(bin as usize) {
            slot.push(obj);
        }
    }
    let expected_leaves: Vec<f64> = bins
        .iter()
        .map(|docs| {
            let ds: Vec<f64> = docs.iter().map(|&i| der1[i]).collect();
            let ws: Vec<f64> = docs.iter().map(|&i| weight[i]).collect();
            calc_average(cb_core::sum_f64(&ds), cb_core::sum_f64(&ws), scaled_l2)
        })
        .collect();
    let (abs, rel) = max_divergence(&sampled.leaf_values, &expected_leaves);
    println!("[region-sampled] leaf oracle: abs={abs:.3e} rel={rel:.3e} (bar={EPS:.0e})");
    assert!(
        abs <= EPS || rel <= EPS,
        "sampled Region leaf values must be calc_average over the RAW channels: \
         abs={abs:.3e} rel={rel:.3e}"
    );
}

/// D-04: an unsampled grow (score channels == leaf channels) must reproduce the frozen
/// Plan-02 routing exactly — the score-channel split changed nothing for it.
#[test]
fn region_device_empty_sample_is_byte_unchanged() {
    if !cfg!(any(feature = "rocm", feature = "cuda")) {
        println!("[region-sampled] SKIP: needs rocm/cuda");
        return;
    }
    let (der1, weight, cindex) = fixture();
    let dev = grow_region_tree(
        &der1, &weight, &der1, &weight, &cindex, 6, 32, 2, 3, 1, 0.0, SCORE_FN_COSINE,
    )
    .expect("unsampled Region grow must succeed");
    assert_eq!(
        dev.leaf_of,
        vec![1, 1, 2, 2, 0, 0],
        "the frozen unsampled Region routing must be unchanged by the score-channel split"
    );
}
