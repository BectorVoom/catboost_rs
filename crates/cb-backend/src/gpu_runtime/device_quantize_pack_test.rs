//! QPACK-01 bit-exact self-oracle: the device quantize+pack fast path
//! ([`crate::gpu_runtime::cindex::fill_packed_cindex_on_device`] driving
//! [`crate::kernels::quantize_pack_feature_kernel`]) must produce packed cindex words
//! BIT-IDENTICAL to the host pipeline (host `partition_point` quantize →
//! [`crate::gpu_runtime::cindex::pack_cindex`]) — integer equality, tighter than any
//! float bar (D-07). The ground truth is the host pipeline over the SAME columns and
//! borders; the device fusion is the system under test.
//!
//! Lives as a `#[cfg(test)] mod` inside `gpu_runtime` (not `tests/`) because
//! `plan_cindex` / `fill_packed_cindex_on_device` / `pack_cindex` are `pub(crate)` —
//! same visibility rationale as the `device_float_only_identity_test` sibling. All
//! `#[test]` / `.unwrap()` / indexing live here per the source/test separation rule.

use crate::gpu_runtime::cindex::{fill_packed_cindex_on_device, pack_cindex, plan_cindex};

/// Deterministic pseudo-random f32 column (LCG — no rand dep) in roughly [-4, 4),
/// with every `nan_every`-th value replaced by NaN when `nan_every > 0`.
fn synth_column(n: usize, seed: u32, nan_every: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|i| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            if nan_every > 0 && i % nan_every == 0 {
                f32::NAN
            } else {
                ((state >> 8) as f32 / (1u32 << 21) as f32) - 4.0
            }
        })
        .collect()
}

/// Ascending f32-exact borders (each an exact `f64::from(f32)`, mirroring the real
/// border builder's midpoint construction) spanning the synth column's value range.
fn synth_borders(count: usize) -> Vec<f64> {
    (0..count)
        .map(|k| f64::from(-3.5_f32 + (7.0_f32 * (k as f32 + 0.5) / count as f32)))
        .collect()
}

/// The host reference quantizer — the same `partition_point` count
/// `cb_train::boosting::quantize_feature_major` performs (that fn is not reachable
/// from this crate; the expression is small enough that replicating it here is the
/// oracle, with the comparison done in f64 exactly like the host).
///
/// Including the host's `nan_to_top_bin` branch: a trailing `f32::MAX` border is
/// upstream's `nan_mode=Max` sentinel, and a NaN then belongs in the TOP bin rather
/// than in bin 0. Replicating that here is what makes this oracle able to catch a
/// device kernel that ignores the sentinel.
fn host_quantize(columns: &[Vec<f32>], borders: &[Vec<f64>], n: usize) -> Vec<u32> {
    let mut bins = vec![0u32; columns.len() * n];
    for (f, col) in columns.iter().enumerate() {
        let b = &borders[f];
        let nan_to_top_bin = b.last().is_some_and(|&x| x == f64::from(f32::MAX));
        for (i, &v) in col.iter().enumerate() {
            let value = f64::from(v);
            bins[f * n + i] = if nan_to_top_bin && value.is_nan() {
                b.len() as u32
            } else {
                b.partition_point(|&bb| value > bb) as u32
            };
        }
    }
    bins
}

/// `synth_borders` with upstream's `nan_mode=Max` SENTINEL appended.
fn synth_borders_with_nan_sentinel(count: usize) -> Vec<f64> {
    let mut b = synth_borders(count);
    b.push(f64::from(f32::MAX));
    b
}

/// Host `plan_cindex` and the packer's own geometry can never disagree: the TCFeature
/// tables must be identical (this is the shared-plan refactor's no-drift pin, and it
/// runs on every backend — no device needed).
#[test]
fn plan_geometry_matches_packer_geometry() {
    let n = 257usize;
    let n_features = 7usize;
    let n_bins_line = 64usize;
    let n_buckets = vec![n_bins_line; n_features];
    let one_hot = vec![false; n_features];
    let borders: Vec<Vec<f64>> = (0..n_features).map(|f| synth_borders(3 + 8 * f)).collect();
    let columns: Vec<Vec<f32>> = (0..n_features)
        .map(|f| synth_column(n, 42 + f as u32, 0))
        .collect();
    let bins = host_quantize(&columns, &borders, n);

    let plan = plan_cindex(&n_buckets, &one_hot, n).unwrap();
    let packed = pack_cindex(&bins, &n_buckets, &one_hot, n).unwrap();
    assert_eq!(plan.features, packed.features, "plan vs packer TCFeature drift");
    assert_eq!(
        plan.num_groups * n,
        packed.words.len(),
        "plan vs packer word-count drift"
    );
}

/// The core QPACK-01 pin: device-fused quantize+pack words == host quantize → pack
/// words, bit-exact, across mixed border widths (including an EMPTY border list — the
/// constant/ignored-feature case), NaN-bearing columns (NaN → bin 0 on both sides),
/// and multiple word groups.
///
/// Runs on EVERY backend, cubecl-cpu included: the kernel is one bounds-guarded thread
/// per object with no `CUBE_COUNT` grid-stride (the builtin cubecl-cpu rejects), which
/// is what makes this oracle verifiable in the default `cargo test` lane.
#[test]
fn device_quantize_pack_words_are_bit_identical_to_host_pipeline() {
    let device = <crate::SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <crate::SelectedRuntime as cubecl::Runtime>::client(&device);

    // 1000 objects (not a multiple of the 256-lane cube), 12 features at 6 bits each
    // (padded line width 64) → 5 features/word → 3 word groups; border widths 0..=63;
    // feature 4 carries NaNs.
    let n = 1000usize;
    let n_features = 12usize;
    let n_bins_line = 64usize;
    let n_buckets = vec![n_bins_line; n_features];
    let one_hot = vec![false; n_features];
    let borders: Vec<Vec<f64>> = (0..n_features)
        .map(|f| {
            if f == 2 {
                Vec::new() // empty border list: bin constantly 0, field still packed
            } else {
                synth_borders((5 * f + 1).min(63))
            }
        })
        .collect();
    let columns: Vec<Vec<f32>> = (0..n_features)
        .map(|f| synth_column(n, 1000 + f as u32, if f == 4 { 7 } else { 0 }))
        .collect();

    // Host pipeline (the oracle): quantize in f64, pack on host.
    let bins = host_quantize(&columns, &borders, n);
    let host_packed = pack_cindex(&bins, &n_buckets, &one_hot, n).unwrap();

    // Device pipeline (system under test): plan + fused quantize/pack kernel.
    let plan = plan_cindex(&n_buckets, &one_hot, n).unwrap();
    let words_h = fill_packed_cindex_on_device(&client, &columns, &borders, &plan, n).unwrap();
    let bytes = client.read_one(words_h).unwrap();
    let device_words: &[u32] = bytemuck::cast_slice(&bytes);

    assert_eq!(
        device_words.len(),
        host_packed.words.len(),
        "device word buffer length mismatch"
    );
    for (i, (&d, &h)) in device_words.iter().zip(host_packed.words.iter()).enumerate() {
        assert_eq!(
            d, h,
            "packed word {i} diverges: device {d:#010x} vs host {h:#010x} \
             (group {}, obj {})",
            i / n,
            i % n
        );
    }
}

/// `nan_mode=Max` on the RAW device channel. Upstream encodes the NaN treatment in
/// the border list itself — an appended `f32::MAX` SENTINEL — and a NaN must then
/// land in the TOP bin. The device kernel's bin is a count of `v > borders[k]`, and
/// IEEE makes every one of those compares false for a NaN, so the natural device
/// answer is bin 0: the `nan_mode=Min` answer, for a caller who asked for `Max`.
///
/// This is not hypothetical. The raw channel is offered on exactly the float-only
/// SymmetricTree pool a NaN column routes through, and until the kernel's
/// `nan_to_top` branch existed a device `Max` fit silently trained a `Min` model
/// while the CPU path (already sentinel-aware) trained the right one — the two
/// diverging with no error raised anywhere.
///
/// Both directions are pinned: the sentinel column must match the host, AND the two
/// modes must actually DISAGREE on the NaN rows. Without the second assertion a
/// kernel that ignored the sentinel entirely would still pass a host comparison run
/// against an equally-broken oracle.
#[test]
fn nan_sentinel_column_bins_to_the_top_on_the_device() {
    let device = <crate::SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <crate::SelectedRuntime as cubecl::Runtime>::client(&device);

    let n = 512usize;
    let n_features = 4usize;
    let n_bins_line = 64usize; // 6 bits: comfortably above the 21 bins used here
    let n_buckets = vec![n_bins_line; n_features];
    let one_hot = vec![false; n_features];

    // Every column carries NaNs (every 5th row) so the sentinel branch is exercised
    // across word groups, not just in one isolated feature.
    let columns: Vec<Vec<f32>> = (0..n_features)
        .map(|f| synth_column(n, 7000 + f as u32, 5))
        .collect();

    let borders_min: Vec<Vec<f64>> = (0..n_features).map(|_| synth_borders(20)).collect();
    let borders_max: Vec<Vec<f64>> = (0..n_features)
        .map(|_| synth_borders_with_nan_sentinel(20))
        .collect();

    let plan = plan_cindex(&n_buckets, &one_hot, n).unwrap();

    let mut device_by_mode = Vec::new();
    for borders in [&borders_min, &borders_max] {
        let host_packed =
            pack_cindex(&host_quantize(&columns, borders, n), &n_buckets, &one_hot, n).unwrap();
        let words_h = fill_packed_cindex_on_device(&client, &columns, borders, &plan, n).unwrap();
        let bytes = client.read_one(words_h).unwrap();
        let device_words: &[u32] = bytemuck::cast_slice(&bytes);
        assert_eq!(
            device_words,
            &host_packed.words[..],
            "device quantize+pack diverges from the sentinel-aware host oracle"
        );
        device_by_mode.push(device_words.to_vec());
    }

    // Discrimination: Min and Max must produce DIFFERENT packed words. If they
    // coincide, the sentinel is being ignored on both sides and the equality above
    // proves nothing.
    assert_ne!(
        device_by_mode[0], device_by_mode[1],
        "nan_mode Min and Max produced identical packed cindex words — the f32::MAX \
         sentinel is not reaching the device quantizer, so the equality assertions \
         above are vacuous"
    );
}
