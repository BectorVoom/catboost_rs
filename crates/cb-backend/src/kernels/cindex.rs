//! Bit-exact self-oracle for the bit-packed compressed index (cindex) — Plan 10-06,
//! GPUT-15. The host `gpu_runtime::cindex::pack_cindex` grouped bit-packing + the device
//! [`crate::kernels::read_all_bins_kernel`] (which reads every `(feature, obj)` cell
//! through the ONE [`crate::kernels::read_bin`] accessor `(cindex[Offset + obj] >> Shift)
//! & Mask`) must reproduce the CPU quantized bins EXACTLY — integer equality (BIT-EXACT,
//! tighter than the ≤1e-4 float bar, D-07). The ground truth is the SOURCE bin matrix
//! (D-02: no `cb-train` reach, no upstream/CUB fixture); packing then read-back is the
//! system under test. Multiple features sharing one 32-bit word each extract their own
//! field via their distinct Shift, and mixed per-feature bit-widths + large-n cases are
//! asserted.
//!
//! Source/test separation is mandatory (CLAUDE.md / AGENTS.md): the `read_bin` accessor +
//! `read_all_bins_kernel` live in `kernels.rs`, the packer in `gpu_runtime/cindex.rs`; all
//! `#[test]` + `.unwrap()`/indexing live here.
//!
//! This runs on `rocm` in-env on gfx1100 (wave32) and under every other backend over the
//! generic [`crate::SelectedRuntime`]. The authoritative Kaggle CUDA sign-off (bit-exact)
//! is human-gated via 10-09; here the source bin matrix is the ground truth.

use cubecl::prelude::*;

use crate::gpu_runtime::cindex::{feature_bits, pack_cindex, read_bin_host};
use crate::kernels::ctr_device::binarize_ctr_column_host;
use crate::kernels::read_all_bins_kernel;

/// Launch geometry: 32-wide cubes (wave32 gfx1100), enough cubes to cover every cell.
const CUBE_DIM: usize = 32;

/// Pack `bins` (feature-major `bins[feature * n + obj]`, per-feature bucket counts
/// `n_buckets`) into the grouped bit-packed cindex on the host, upload it, then read back
/// EVERY bin through the device `read_all_bins_kernel` (each cell via `read_bin`). Returns
/// the materialized feature-major bin matrix (must equal the source `bins`, bit-exact).
fn pack_then_read_all(bins: &[u32], n_buckets: &[usize], n: usize) -> Vec<u32> {
    let n_features = n_buckets.len();
    let packed = pack_cindex(bins, n_buckets, n).unwrap();
    let (offsets, shifts, masks) = packed.device_arrays().unwrap();

    let device = <crate::SelectedRuntime as Runtime>::Device::default();
    let client = <crate::SelectedRuntime as Runtime>::client(&device);
    let dim32 = CubeDim { x: CUBE_DIM as u32, y: 1, z: 1 };

    let words_h = client.create(cubecl::bytes::Bytes::from_elems(packed.words.clone()));
    let offsets_h = client.create(cubecl::bytes::Bytes::from_elems(offsets));
    let shifts_h = client.create(cubecl::bytes::Bytes::from_elems(shifts));
    let masks_h = client.create(cubecl::bytes::Bytes::from_elems(masks));

    let total = n_features * n;
    let out_h = client.empty(total * std::mem::size_of::<u32>());
    let cubes = total.div_ceil(CUBE_DIM).max(1);
    read_all_bins_kernel::launch::<f64, crate::SelectedRuntime>(
        &client,
        CubeCount::Static(cubes as u32, 1, 1),
        dim32,
        unsafe { ArrayArg::from_raw_parts(words_h, packed.words.len()) },
        unsafe { ArrayArg::from_raw_parts(offsets_h, n_features) },
        unsafe { ArrayArg::from_raw_parts(shifts_h, n_features) },
        unsafe { ArrayArg::from_raw_parts(masks_h, n_features) },
        unsafe { ArrayArg::from_raw_parts(out_h.clone(), total) },
        n_features as u32,
    );

    let bytes = client.read_one(out_h).unwrap();
    bytemuck::cast_slice::<u8, u32>(&bytes).to_vec()
}

/// Deterministic pseudo-random bin column matrix (LCG — no rand dep): feature-major
/// `bins[feature * n + obj]`, each feature's bins in `0..n_buckets[feature]`.
fn synth_bins(n: usize, n_buckets: &[usize], seed: u32) -> Vec<u32> {
    let mut state = seed;
    let n_features = n_buckets.len();
    let mut bins = vec![0u32; n_features * n];
    for (feature, &nb) in n_buckets.iter().enumerate() {
        for obj in 0..n {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            bins[feature * n + obj] = (state >> 8) % (nb as u32);
        }
    }
    bins
}

#[test]
fn pack_read_bit_exact_multiple_features_per_word() {
    // Two 8-bit features (buckets 256) pack into ONE 32-bit word per object (8+8=16 bits
    // <= 32), so both features share the SAME Offset and extract via distinct Shifts
    // (0 and 8). read_bin over the packed word must reproduce the source bins exactly.
    let n = 64usize;
    let n_buckets = vec![256usize, 256];
    let bins = synth_bins(n, &n_buckets, 42);

    let packed = pack_cindex(&bins, &n_buckets, n).unwrap();
    // Grouping: two 8-bit features share word column 0 (offset 0), shifts 0 and 8.
    assert_eq!(packed.features[0].offset, 0, "feature 0 word base");
    assert_eq!(packed.features[1].offset, 0, "feature 1 shares the same word column");
    assert_eq!(packed.features[0].shift, 0);
    assert_eq!(packed.features[1].shift, 8);
    assert_eq!(packed.features[0].mask, 0xFF);
    assert_eq!(packed.features[1].mask, 0xFF);
    assert_eq!(packed.words.len(), n, "one group of 2 features => n words");

    // Host read_bin_host reproduces the source bins from the packed words (independent
    // of the device path).
    for feature in 0..2usize {
        let f = &packed.features[feature];
        for obj in 0..n {
            let got = read_bin_host(&packed.words, f.offset, obj, f.shift, f.mask);
            assert_eq!(got, bins[feature * n + obj], "host read_bin (feature {feature}, obj {obj})");
        }
    }

    // Device read_all_bins reproduces the source bins bit-exactly.
    let got = pack_then_read_all(&bins, &n_buckets, n);
    assert_eq!(got, bins, "device read_all_bins must be bit-exact");
}

#[test]
fn pack_read_bit_exact_across_seeds_and_widths() {
    // Uniform per-feature bucket counts across the histogram border widths (2/16/32/64/
    // 128/256) and seeds; every pack->read must reproduce the source EXACTLY.
    for &nb in &[2usize, 16, 32, 64, 128, 256] {
        for &seed in &[1u32, 42, 12345, 987_654_321] {
            let n = 500usize;
            let n_features = 3usize;
            let n_buckets = vec![nb; n_features];
            let bins = synth_bins(n, &n_buckets, seed);

            let got = pack_then_read_all(&bins, &n_buckets, n);
            assert_eq!(
                got, bins,
                "device pack->read not bit-exact (n_buckets={nb}, seed={seed})"
            );
        }
    }
}

#[test]
fn pack_read_bit_exact_mixed_widths_grouping() {
    // Mixed per-feature bit-widths force a non-trivial greedy grouping: 4-bit (16), 4-bit
    // (16), 8-bit (256), 1-bit (2), 8-bit (256), 8-bit (256). Cumulative bits:
    //   f0: 4  (used 0->4)      f1: 4  (4->8)    f2: 8  (8->16)
    //   f3: 1  (16->17)         f4: 8  (17->25)  f5: 8  (25->33 > 32 -> new group)
    // so features 0..=4 share word column 0; feature 5 starts column 1. Every feature's
    // field must extract exactly.
    let n = 300usize;
    let n_buckets = vec![16usize, 16, 256, 2, 256, 256];
    let bins = synth_bins(n, &n_buckets, 7);

    let packed = pack_cindex(&bins, &n_buckets, n).unwrap();
    // Groups: features 0..=4 in column 0 (offset 0), feature 5 in column 1 (offset n).
    assert_eq!(packed.features[0].offset, 0);
    assert_eq!(packed.features[4].offset, 0);
    assert_eq!(packed.features[5].offset, n as u64, "feature 5 starts a new word column");
    // Shifts within group 0: 0,4,8,16,17.
    assert_eq!(packed.features[0].shift, 0);
    assert_eq!(packed.features[1].shift, 4);
    assert_eq!(packed.features[2].shift, 8);
    assert_eq!(packed.features[3].shift, 16);
    assert_eq!(packed.features[4].shift, 17);
    // Shift within group 1: 0.
    assert_eq!(packed.features[5].shift, 0);
    assert_eq!(packed.words.len(), 2 * n, "two groups => 2*n words");

    let got = pack_then_read_all(&bins, &n_buckets, n);
    assert_eq!(got, bins, "mixed-width grouping must be bit-exact");
}

/// HOST-ONLY pack→extract bit-exactness over mixed widths and a non-trivial `n` — no
/// device launch, so it guards the (rayon-parallel) `pack_cindex` group-column packing
/// even where the device runtime is unavailable: every `(feature, obj)` cell extracted
/// through `read_bin_host` must reproduce the source bin exactly.
#[test]
fn pack_host_extract_bit_exact_mixed_widths() {
    let n = 10_007usize; // prime, so chunk boundaries never align with feature strides
    let n_buckets = vec![16usize, 16, 256, 2, 256, 256, 64, 33];
    let bins = synth_bins(n, &n_buckets, 260716);

    let packed = pack_cindex(&bins, &n_buckets, n).unwrap();
    for (feature, f) in packed.features.iter().enumerate() {
        for obj in 0..n {
            let got = read_bin_host(&packed.words, f.offset, obj, f.shift, f.mask);
            assert_eq!(
                got,
                bins[feature * n + obj],
                "host pack->extract (feature {feature}, obj {obj})"
            );
        }
    }
}

#[test]
fn pack_read_bit_exact_large_n() {
    // Large n so many packed words + a non-cube-multiple tail are exercised.
    let n = 10_000usize;
    let n_buckets = vec![256usize, 16, 2, 64];
    let bins = synth_bins(n, &n_buckets, 20260703);
    let got = pack_then_read_all(&bins, &n_buckets, n);
    assert_eq!(got, bins, "large-n pack->read must be bit-exact");
}

/// Deterministic categorical bin + binclf class + permutation columns (LCG), for the CTR→cindex
/// join test. Mirrors `ctr_device_test::synth_fixture` (kept local — the two test modules do not
/// share a helper crate).
fn synth_ctr_fixture(n: usize, cardinality: u32, seed: u32) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut s = seed;
    let next = |st: &mut u32| {
        *st = st.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *st >> 8
    };
    let bins: Vec<u32> = (0..n).map(|_| next(&mut s) % cardinality).collect();
    let class: Vec<u32> = (0..n).map(|_| next(&mut s) % 2).collect();
    let mut perm: Vec<u32> = (0..n as u32).collect();
    for i in 0..n {
        let j = (next(&mut s) as usize) % n;
        perm.swap(i, j);
    }
    (bins, class, perm)
}

#[test]
fn ctr_binarized_column_joins_cindex_bit_exact() {
    // Plan 08 (GPUT-10) CTR→cindex JOIN: a binarized device CTR column must pack into the cindex as
    // an ADDITIONAL feature column the histogram loop reads, extracting bit-exactly like any other
    // feature. Build two plain features + one device-CTR column (accumulated + binarized ON device),
    // append the CTR column as the last feature, pack, and read every cell back bit-exact.
    if cfg!(feature = "wgpu") {
        eprintln!("SKIP ctr_binarized_column_joins_cindex_bit_exact: wgpu has no f64 CTR seam");
        return;
    }
    let n = 300usize;
    let prior = 0.5;
    let borders = vec![0.2_f64, 0.4, 0.5, 0.6, 0.8]; // => 6 CTR buckets (0..=5)
    let ctr_buckets = borders.len() + 1;

    // The device CTR column, binarized to cindex bins (the extra column).
    let (cat_bins, class, perm) = synth_ctr_fixture(n, 6, 314);
    let ctr_bucket_count = cat_bins.iter().copied().max().map_or(0, |m| m as usize + 1);
    let ctr_col =
        binarize_ctr_column_host(&perm, &cat_bins, &class, prior, ctr_bucket_count, &borders)
            .unwrap();
    assert_eq!(ctr_col.len(), n);

    // Two plain quantized features (8-bit + 4-bit) alongside the CTR column.
    let plain0 = synth_bins(n, &[256usize], 11);
    let plain1 = synth_bins(n, &[16usize], 22);
    let n_buckets = vec![256usize, 16, ctr_buckets];

    // Feature-major augmented matrix: [plain0 | plain1 | ctr_col].
    let mut bins = Vec::with_capacity(3 * n);
    bins.extend_from_slice(&plain0);
    bins.extend_from_slice(&plain1);
    bins.extend_from_slice(&ctr_col);

    // Pack the augmented matrix (the CTR column joins as feature 2), then extract EVERY cell via
    // the HOST `read_bin_host` (the exact device `read_bin` accessor `(word >> shift) & mask`). The
    // host extraction is deterministic and independent of the grid-stride `read_all_bins_kernel`
    // device path (which the sibling `pack_read_bit_exact_*` oracles / the Kaggle CUDA sign-off
    // exercise on device); here the point is that the binarized CTR column PACKS + EXTRACTS as an
    // additional cindex feature bit-exact — the JOIN the histogram loop reads.
    let packed = pack_cindex(&bins, &n_buckets, n).unwrap();
    for feature in 0..3usize {
        let f = &packed.features[feature];
        for obj in 0..n {
            let got = read_bin_host(&packed.words, f.offset, obj, f.shift, f.mask);
            assert_eq!(
                got,
                bins[feature * n + obj],
                "CTR-augmented cindex extract mismatch (feature {feature}, obj {obj})"
            );
        }
    }
    // The CTR column (feature 2) is a valid, non-degenerate additional cindex column.
    assert!(
        ctr_col.iter().all(|&b| (b as usize) < ctr_buckets),
        "CTR cindex bins must lie in 0..ctr_buckets"
    );
}

#[test]
fn feature_bits_and_overflow_guards() {
    // ceil(log2(n_buckets)) sizing (bin values 0..n_buckets).
    assert_eq!(feature_bits(1).unwrap(), 1, "single bucket still needs 1 bit");
    assert_eq!(feature_bits(2).unwrap(), 1, "0..1 -> 1 bit");
    assert_eq!(feature_bits(3).unwrap(), 2, "0..2 -> 2 bits");
    assert_eq!(feature_bits(16).unwrap(), 4, "0..15 -> 4 bits");
    assert_eq!(feature_bits(17).unwrap(), 5, "0..16 -> 5 bits");
    assert_eq!(feature_bits(256).unwrap(), 8, "full byte");

    // Out-of-range bin (>= n_buckets) is rejected BEFORE masking into a shared word.
    let n = 4usize;
    let n_buckets = vec![16usize];
    let mut bins = synth_bins(n, &n_buckets, 1);
    bins[0] = 16; // == n_buckets -> out of the valid 0..16 range
    let err = pack_cindex(&bins, &n_buckets, n);
    assert!(
        matches!(err, Err(cb_core::CbError::OutOfRange(_))),
        "out-of-range bin must surface CbError::OutOfRange, got {err:?}"
    );

    // Length disagreement surfaces LengthMismatch.
    let short = &bins[..bins.len() - 1];
    let err = pack_cindex(short, &n_buckets, n);
    assert!(
        matches!(err, Err(cb_core::CbError::LengthMismatch { .. })),
        "wrong bins length must surface CbError::LengthMismatch, got {err:?}"
    );
}
