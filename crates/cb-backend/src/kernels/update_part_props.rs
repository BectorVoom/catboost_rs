//! Self-oracle for the per-partition stat aggregation primitive (Plan 10-05, GPUT-16):
//! `update_part_props_kernel` sums a stat channel per partition into the `2^depth`
//! partition totals the depth-1 leaf-value step reads back. The device Σstat must match
//! an INLINE ORDERED SERIAL per-partition sum (D-02: no `cb-train` reach, no upstream
//! fixture) — each partition folded through `cb_core::sum_f64` in ascending object order
//! (the frozen host reduction), and every object's partition id asserted in range
//! (generalizing the `pointwise_hist` `host_reference_hist2` range discipline).
//!
//! Accumulation is in f64 (the 10-03 deterministic reduce winner: fixed-order tree
//! reduce, no float atomics), so the fold is deterministic; the asserted bounds are
//! generous/run-stable (NOT the GPU-06 epsilon, which is signed off on Kaggle CUDA via
//! 10-09).
//!
//! Source/test separation is mandatory (CLAUDE.md / AGENTS.md): the kernel lives in
//! `kernels.rs`; all assertions + `.unwrap()`/indexing live here. This runs on `rocm`
//! in-env on gfx1100 (wave32) and under every other backend over the generic
//! [`crate::SelectedRuntime`].

use cubecl::prelude::*;

use crate::kernels::update_part_props_kernel;

/// Generous relative bound for an f32 device Σstat vs the f64 CPU baseline.
const F32_REL_TOL: f64 = 1e-3;
/// Generous absolute bound for an f32 device Σstat vs the f64 CPU baseline.
const F32_ABS_TOL: f64 = 1e-3;
/// Generous relative bound for an f64 device Σstat vs the f64 CPU baseline.
const F64_REL_TOL: f64 = 1e-9;
/// Looser absolute bound for the large-partition f64 accumulation case.
const F64_ABS_TOL: f64 = 1e-6;

/// Launch `update_part_props_kernel::<F>` (one cube per partition) and read back the
/// `num_parts` per-partition sums. `part_offsets` has `num_parts + 1` entries (the
/// contiguous partition boundary array the 10-04 partition primitives produce).
fn run_update_part_props<F>(stat: &[F], part_offsets: &[u32]) -> Vec<F>
where
    F: Float + CubeElement + bytemuck::Pod,
{
    let num_parts = part_offsets.len() - 1;
    let device = <crate::SelectedRuntime as Runtime>::Device::default();
    let client = <crate::SelectedRuntime as Runtime>::client(&device);

    let stat_h = client.create(cubecl::bytes::Bytes::from_elems(stat.to_vec()));
    let off_h = client.create(cubecl::bytes::Bytes::from_elems(part_offsets.to_vec()));
    let props_h = client.empty(num_parts * std::mem::size_of::<F>());

    update_part_props_kernel::launch::<F, crate::SelectedRuntime>(
        &client,
        CubeCount::Static(num_parts as u32, 1, 1),
        CubeDim { x: 32u32, y: 1, z: 1 },
        unsafe { ArrayArg::from_raw_parts(stat_h, stat.len()) },
        unsafe { ArrayArg::from_raw_parts(off_h, part_offsets.len()) },
        unsafe { ArrayArg::from_raw_parts(props_h.clone(), num_parts) },
    );

    let bytes = client.read_one(props_h).unwrap();
    bytemuck::cast_slice::<u8, F>(&bytes).to_vec()
}

/// Inline ORDERED serial per-partition Σstat reference (D-02). Every object's partition
/// id (derived from the contiguous `part_offsets`) is asserted in `0..num_parts` before
/// it is gathered (the `pointwise_hist` range discipline generalized to partitions), then
/// each partition's members are folded through `cb_core::sum_f64` in ascending object
/// order — the frozen host reduction the device fold is checked against.
fn cpu_update_part_props(stat: &[f64], part_offsets: &[u32]) -> Vec<f64> {
    let num_parts = part_offsets.len() - 1;
    let n = stat.len();
    assert_eq!(
        part_offsets[0], 0,
        "part_offsets must start at 0 (contiguous partition cover)"
    );
    assert_eq!(
        part_offsets[num_parts] as usize, n,
        "part_offsets must end at n (contiguous partition cover)"
    );

    // Derive object → partition and assert every partition id is in range (range
    // discipline). part_offsets is non-decreasing and covers [0, n).
    let mut part_of = vec![0usize; n];
    for p in 0..num_parts {
        let a = part_offsets[p] as usize;
        let b = part_offsets[p + 1] as usize;
        assert!(a <= b, "part_offsets non-decreasing (p={p})");
        assert!(p < num_parts, "partition id {p} out of range 0..{num_parts}");
        for slot in part_of.iter_mut().take(b).skip(a) {
            *slot = p;
        }
    }

    // Fold each partition's members in ascending object order through cb_core::sum_f64.
    (0..num_parts)
        .map(|p| {
            let members: Vec<f64> = (0..n)
                .filter(|&i| part_of[i] == p)
                .map(|i| stat[i])
                .collect();
            cb_core::sum_f64(&members)
        })
        .collect()
}

/// Deterministic pseudo-random stat value in `[-1, 1)` (LCG — no rand dep).
fn synth_stat_f64(n: usize, seed: u32) -> Vec<f64> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f64 / (1u32 << 24) as f64) * 2.0 - 1.0
        })
        .collect()
}

/// Contiguous partition offsets from per-partition sizes (the exclusive-prefix boundary
/// array the 10-04 partition primitives emit); has `sizes.len() + 1` entries.
fn offsets_from_sizes(sizes: &[usize]) -> Vec<u32> {
    let mut offsets = vec![0u32];
    let mut acc = 0u32;
    for &sz in sizes {
        acc += sz as u32;
        offsets.push(acc);
    }
    offsets
}

fn max_divergence(device: &[f64], baseline: &[f64]) -> (f64, f64) {
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    for (d, b) in device.iter().zip(baseline) {
        let abs = (d - b).abs();
        max_abs = max_abs.max(abs);
        let denom = b.abs().max(1e-300);
        max_rel = max_rel.max(abs / denom);
    }
    (max_abs, max_rel)
}

#[test]
fn update_part_props_behaviour_example() {
    // stat [1,2,3,4,5] with partition sizes [2,3] (offsets {0,2,5}) → partition 0 = 1+2 =
    // 3, partition 1 = 3+4+5 = 12.
    let stat = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let offsets = offsets_from_sizes(&[2, 3]);
    assert_eq!(offsets, vec![0u32, 2, 5]);
    let dev = run_update_part_props(&stat, &offsets);
    assert_eq!(dev, vec![3.0_f64, 12.0], "per-partition Σstat behaviour example");
}

#[test]
fn update_part_props_handles_empty_partition() {
    // A middle partition with zero members must yield exactly 0 (start == end).
    let stat = vec![10.0_f64, 20.0, 30.0];
    let offsets = offsets_from_sizes(&[1, 0, 2]); // {0,1,1,3}
    assert_eq!(offsets, vec![0u32, 1, 1, 3]);
    let dev = run_update_part_props(&stat, &offsets);
    assert_eq!(dev, vec![10.0_f64, 0.0, 50.0], "empty partition → 0");
}

#[test]
fn update_part_props_matches_ordered_serial_f64() {
    // Varied partition sizes, several > CUBE_DIM=32 so the grid-stride intra-partition
    // fold is exercised. f64 channel.
    let sizes = [1usize, 5, 32, 33, 64, 100, 7, 50, 0, 3];
    let offsets = offsets_from_sizes(&sizes);
    let n: usize = sizes.iter().sum();
    let stat = synth_stat_f64(n, 2024);

    let dev = run_update_part_props(&stat, &offsets);
    let dev_f64: Vec<f64> = dev.iter().map(|&v| v as f64).collect();
    let baseline = cpu_update_part_props(&stat, &offsets);

    let (max_abs, max_rel) = max_divergence(&dev_f64, &baseline);
    assert!(
        max_abs <= F64_ABS_TOL && max_rel <= F64_REL_TOL,
        "f64 update_part_props diverged: max_abs={max_abs:e}, max_rel={max_rel:e}"
    );
}

#[test]
fn update_part_props_matches_ordered_serial_f32() {
    // Same geometry, f32 channel (accumulation still widens to f64 in-kernel; the bound
    // is the generous f32-vs-f64-baseline bound, NOT the signed-off epsilon).
    let sizes = [3usize, 40, 64, 1, 17, 80];
    let offsets = offsets_from_sizes(&sizes);
    let n: usize = sizes.iter().sum();
    let stat_f64 = synth_stat_f64(n, 7);
    let stat_f32: Vec<f32> = stat_f64.iter().map(|&v| v as f32).collect();

    let dev = run_update_part_props(&stat_f32, &offsets);
    let dev_f64: Vec<f64> = dev.iter().map(|&v| v as f64).collect();
    // Baseline folds the SAME f32 values (widened) so the comparison isolates the fold,
    // not the f64→f32 quantization of the inputs.
    let base_inputs: Vec<f64> = stat_f32.iter().map(|&v| v as f64).collect();
    let baseline = cpu_update_part_props(&base_inputs, &offsets);

    let (max_abs, max_rel) = max_divergence(&dev_f64, &baseline);
    assert!(
        max_abs <= F32_ABS_TOL && max_rel <= F32_REL_TOL,
        "f32 update_part_props diverged: max_abs={max_abs:e}, max_rel={max_rel:e}"
    );
}
