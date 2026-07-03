//! Self-oracle for the TDataPartition {Offset,Size} update (Plan 10-04, GPUT-16):
//! from a SORTED partition-id array the offset/size update must match an INLINE SERIAL
//! partition-bookkeeping reference (D-02: no `cb-train` reach, no upstream fixture),
//! including an EMPTY-partition case. Integer bookkeeping ⇒ the match is BIT-EXACT.
//!
//! Source/test separation is mandatory (CLAUDE.md / AGENTS.md): the kernels live in
//! `kernels.rs`; all assertions + `.unwrap()`/indexing live here.
//!
//! This runs on `rocm` in-env on gfx1100 (wave32) and under every other backend over
//! the generic [`crate::SelectedRuntime`]. The authoritative Kaggle CUDA sign-off
//! (bit-exact) is human-gated via 10-09.

use cubecl::prelude::*;

use crate::kernels::{
    full_scan_into, key_head_flag_kernel, update_partition_offsets_kernel,
    update_partition_sizes_kernel,
};

/// 32-wide cubes (wave32 gfx1100); enough cubes to cover every element.
const CUBE_DIM: usize = 32;

/// Compute each partition's `{Offset, Size}` from a SORTED `part_ids` array over the
/// selected runtime: head flags (10-01 `key_head_flag`) → exclusive `full_scan`
/// (run index per element) → offsets scatter → sizes expand. `num_partitions` sizes the
/// output; partitions absent from `part_ids` come back `{0, 0}` (well-defined empty).
fn run_partition_update(part_ids: &[u32], num_partitions: usize) -> (Vec<u32>, Vec<u32>) {
    let n = part_ids.len();
    let device = <crate::SelectedRuntime as Runtime>::Device::default();
    let client = <crate::SelectedRuntime as Runtime>::client(&device);
    let dim32 = CubeDim { x: CUBE_DIM as u32, y: 1, z: 1 };
    let n_cubes = n.div_ceil(CUBE_DIM).max(1);
    let count = CubeCount::Static(n_cubes as u32, 1, 1);

    // Distinct-run count (host scalar; positions are computed on-device, mirroring
    // reduce-by-key's host-scalar num_segments).
    let mut num_runs = 0usize;
    for i in 0..n {
        if i == 0 || part_ids[i] != part_ids[i - 1] {
            num_runs += 1;
        }
    }

    // Outputs seeded to 0: empty partitions keep {Offset:0, Size:0}.
    let offsets_h = client.create(cubecl::bytes::Bytes::from_elems(vec![0u32; num_partitions]));
    let sizes_h = client.create(cubecl::bytes::Bytes::from_elems(vec![0u32; num_partitions]));

    let part_ids_h = client.create(cubecl::bytes::Bytes::from_elems(part_ids.to_vec()));

    // Phase 1: key-run head flags (reuse the 10-03 key_head_flag kernel; f64 channel).
    let flags_h = client.empty(n * std::mem::size_of::<f64>());
    key_head_flag_kernel::launch::<f64, crate::SelectedRuntime>(
        &client,
        count.clone(),
        dim32,
        unsafe { ArrayArg::from_raw_parts(part_ids_h.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(flags_h.clone(), n) },
    );

    // Phase 2: exclusive scan of the flags → per-element run index (10-01 reuse,
    // device-resident). `flags_h` is cloned since `full_scan_into` consumes its input.
    let run_ids_h = full_scan_into::<f64>(&client, flags_h.clone(), n, false).unwrap();

    // Phase 3: offsets scatter + compact run_keys/run_starts.
    let run_keys_h = client.empty(num_runs * std::mem::size_of::<u32>());
    let run_starts_h = client.empty(num_runs * std::mem::size_of::<u32>());
    update_partition_offsets_kernel::launch::<f64, crate::SelectedRuntime>(
        &client,
        count.clone(),
        dim32,
        unsafe { ArrayArg::from_raw_parts(part_ids_h, n) },
        unsafe { ArrayArg::from_raw_parts(flags_h, n) },
        unsafe { ArrayArg::from_raw_parts(run_ids_h, n) },
        unsafe { ArrayArg::from_raw_parts(offsets_h.clone(), num_partitions) },
        unsafe { ArrayArg::from_raw_parts(run_keys_h.clone(), num_runs) },
        unsafe { ArrayArg::from_raw_parts(run_starts_h.clone(), num_runs) },
    );

    // Phase 4: per-run sizes (one lane per run).
    let runs_cubes = num_runs.div_ceil(CUBE_DIM).max(1);
    update_partition_sizes_kernel::launch::<f64, crate::SelectedRuntime>(
        &client,
        CubeCount::Static(runs_cubes as u32, 1, 1),
        dim32,
        unsafe { ArrayArg::from_raw_parts(run_keys_h, num_runs) },
        unsafe { ArrayArg::from_raw_parts(run_starts_h, num_runs) },
        unsafe { ArrayArg::from_raw_parts(sizes_h.clone(), num_partitions) },
        n as u32,
        num_runs as u32,
    );

    let ob = client.read_one(offsets_h).unwrap();
    let sb = client.read_one(sizes_h).unwrap();
    (
        bytemuck::cast_slice::<u8, u32>(&ob).to_vec(),
        bytemuck::cast_slice::<u8, u32>(&sb).to_vec(),
    )
}

/// Inline serial partition-bookkeeping reference (D-02): walk contiguous equal-id runs
/// of the SORTED `part_ids`, recording each partition's contiguous `{Offset, Size}`.
/// Absent partitions keep `{0, 0}`.
fn cpu_partition_update(part_ids: &[u32], num_partitions: usize) -> (Vec<u32>, Vec<u32>) {
    let mut offsets = vec![0u32; num_partitions];
    let mut sizes = vec![0u32; num_partitions];
    let n = part_ids.len();
    let mut i = 0usize;
    while i < n {
        let p = part_ids[i] as usize;
        let start = i;
        while i < n && part_ids[i] as usize == p {
            i += 1;
        }
        offsets[p] = start as u32;
        sizes[p] = (i - start) as u32;
    }
    (offsets, sizes)
}

#[test]
fn partition_update_behaviour_example() {
    // Sorted partition-id array [0,0,0,1,1] → partition 0 {Offset:0,Size:3},
    // partition 1 {Offset:3,Size:2}.
    let part_ids = vec![0u32, 0, 0, 1, 1];
    let (offsets, sizes) = run_partition_update(&part_ids, 2);
    assert_eq!(offsets, vec![0u32, 3], "offsets for [0,0,0,1,1]");
    assert_eq!(sizes, vec![3u32, 2], "sizes for [0,0,0,1,1]");
}

#[test]
fn partition_update_handles_empty_partitions() {
    // Partition 1 is EMPTY (no elements): it must come back {Offset:0, Size:0} while the
    // populated partitions 0 and 2 are correct.
    let part_ids = vec![0u32, 0, 2, 2, 2];
    let (offsets, sizes) = run_partition_update(&part_ids, 3);
    assert_eq!(offsets, cpu_partition_update(&part_ids, 3).0);
    assert_eq!(sizes, cpu_partition_update(&part_ids, 3).1);
    // Explicitly: 0 → {0,2}; 1 (empty) → {0,0}; 2 → {2,3}.
    assert_eq!(offsets, vec![0u32, 0, 2]);
    assert_eq!(sizes, vec![2u32, 0, 3]);
}

#[test]
fn partition_update_matches_serial_large_n() {
    // n >> CUBE_DIM with varied run sizes (some > CUBE_DIM) and a trailing empty
    // partition, vs the inline serial reference.
    let run_sizes = [40usize, 1, 33, 100, 5, 64, 7, 50, 200];
    let mut part_ids: Vec<u32> = Vec::new();
    for (p, &sz) in run_sizes.iter().enumerate() {
        for _ in 0..sz {
            part_ids.push(p as u32);
        }
    }
    // Declare one extra partition beyond the last populated id → a trailing empty.
    let num_partitions = run_sizes.len() + 1;

    let (d_off, d_sz) = run_partition_update(&part_ids, num_partitions);
    let (e_off, e_sz) = cpu_partition_update(&part_ids, num_partitions);

    assert_eq!(d_off, e_off, "offsets mismatch at n={}", part_ids.len());
    assert_eq!(d_sz, e_sz, "sizes mismatch at n={}", part_ids.len());
    // The trailing declared-but-absent partition is the well-defined empty {0,0}.
    assert_eq!(d_off[num_partitions - 1], 0);
    assert_eq!(d_sz[num_partitions - 1], 0);
}
