//! Self-oracle for the sort/reorder primitives (Plan 10-04, GPUT-16): the stable
//! single-bit reorder (`reorder_one_bit`) and the LSD radix sort composed from it must
//! match INLINE SERIAL stable references — a stable partition-by-bit and a stable sort
//! (D-02: no `cb-train` reach, no upstream/CUB fixture). Integer keys ⇒ the match is
//! BIT-EXACT (tighter than the ≤1e-4 float bar); a duplicate-key stability case and an
//! `n >> CUBE_DIM` case are asserted.
//!
//! Source/test separation is mandatory (CLAUDE.md / AGENTS.md): the kernels live in
//! `kernels.rs`; all assertions + `.unwrap()`/indexing live here.
//!
//! This runs on `rocm` in-env on gfx1100 (wave32) and under every other backend over
//! the generic [`crate::SelectedRuntime`]. The authoritative Kaggle CUDA sign-off
//! (bit-exact) is human-gated via 10-09; here the inline serial reference is the
//! ground truth.

use cubecl::prelude::*;

use crate::kernels::{full_scan_into, radix_bit_flag_kernel, reorder_one_bit_scatter_kernel};

/// Launch geometry: 32-wide cubes (wave32 gfx1100), enough cubes to cover every
/// element (mirrors the `key_head_flag` / `segment_offset_scatter` launch geometry —
/// one bounds-guarded write per lane).
const CUBE_DIM: usize = 32;

/// Run the STABLE single-bit reorder of `keys` (with paired `values`) at `bit` over the
/// selected runtime: bit-flag kernel → 10-01 exclusive `full_scan` (onesBefore) →
/// stable scatter. `total_zeros` (count of `bit==0` keys) is order-invariant, so it is
/// computed host-side once (mirrors reduce-by-key's host-scalar `num_segments`).
fn run_reorder_one_bit(keys: &[u32], values: &[u32], bit: u32) -> (Vec<u32>, Vec<u32>) {
    let n = keys.len();
    assert_eq!(n, values.len(), "keys/values length mismatch");
    let device = <crate::SelectedRuntime as Runtime>::Device::default();
    let client = <crate::SelectedRuntime as Runtime>::client(&device);
    let dim32 = CubeDim { x: CUBE_DIM as u32, y: 1, z: 1 };
    let n_cubes = n.div_ceil(CUBE_DIM).max(1);
    let count = CubeCount::Static(n_cubes as u32, 1, 1);

    let keys_h = client.create(cubecl::bytes::Bytes::from_elems(keys.to_vec()));
    let values_h = client.create(cubecl::bytes::Bytes::from_elems(values.to_vec()));

    // Phase A: per-element bit flags (f64 scan channel — exact for the 0/1 sums).
    let flags_h = client.empty(n * std::mem::size_of::<f64>());
    radix_bit_flag_kernel::launch::<f64, crate::SelectedRuntime>(
        &client,
        count.clone(),
        dim32,
        unsafe { ArrayArg::from_raw_parts(keys_h.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(flags_h.clone(), n) },
        bit,
    );

    // Phase B: exclusive scan of the flags → onesBefore (10-01 reuse, device-resident).
    let ones_before_h = full_scan_into::<f64>(&client, flags_h, n, false).unwrap();

    // total_zeros: order-invariant count of keys with bit == 0.
    let total_zeros = keys.iter().filter(|&&k| (k >> bit) & 1 == 0).count() as u32;

    // Phase C: stable scatter.
    let out_keys_h = client.empty(n * std::mem::size_of::<u32>());
    let out_values_h = client.empty(n * std::mem::size_of::<u32>());
    reorder_one_bit_scatter_kernel::launch::<f64, crate::SelectedRuntime>(
        &client,
        count,
        dim32,
        unsafe { ArrayArg::from_raw_parts(keys_h, n) },
        unsafe { ArrayArg::from_raw_parts(values_h, n) },
        unsafe { ArrayArg::from_raw_parts(ones_before_h, n) },
        unsafe { ArrayArg::from_raw_parts(out_keys_h.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(out_values_h.clone(), n) },
        bit,
        total_zeros,
    );

    let kb = client.read_one(out_keys_h).unwrap();
    let vb = client.read_one(out_values_h).unwrap();
    (
        bytemuck::cast_slice::<u8, u32>(&kb).to_vec(),
        bytemuck::cast_slice::<u8, u32>(&vb).to_vec(),
    )
}

/// LSD radix sort composed from the single-bit reorder: ping-pong the resident
/// key/value buffers, applying one STABLE single-bit reorder per bit from LSB up to the
/// highest set bit of any key. Because each pass is stable, the composition is a stable
/// full sort. Only the FINAL buffers are read back (device-resident across passes);
/// `total_zeros[bit]` is order-invariant so it is computed host-side per bit.
fn run_radix_sort(keys: &[u32], values: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let n = keys.len();
    assert_eq!(n, values.len(), "keys/values length mismatch");
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let device = <crate::SelectedRuntime as Runtime>::Device::default();
    let client = <crate::SelectedRuntime as Runtime>::client(&device);
    let dim32 = CubeDim { x: CUBE_DIM as u32, y: 1, z: 1 };
    let n_cubes = n.div_ceil(CUBE_DIM).max(1);
    let count = CubeCount::Static(n_cubes as u32, 1, 1);

    // Number of LSD passes = bit-width of the largest key (0 keys ⇒ already sorted).
    let max_key = keys.iter().copied().max().unwrap_or(0);
    let num_bits = if max_key == 0 { 0 } else { 32 - max_key.leading_zeros() };

    let mut cur_keys = client.create(cubecl::bytes::Bytes::from_elems(keys.to_vec()));
    let mut cur_vals = client.create(cubecl::bytes::Bytes::from_elems(values.to_vec()));
    let mut alt_keys = client.empty(n * std::mem::size_of::<u32>());
    let mut alt_vals = client.empty(n * std::mem::size_of::<u32>());

    for bit in 0..num_bits {
        // Bit flags on the CURRENT key order.
        let flags_h = client.empty(n * std::mem::size_of::<f64>());
        radix_bit_flag_kernel::launch::<f64, crate::SelectedRuntime>(
            &client,
            count.clone(),
            dim32,
            unsafe { ArrayArg::from_raw_parts(cur_keys.clone(), n) },
            unsafe { ArrayArg::from_raw_parts(flags_h.clone(), n) },
            bit,
        );
        // Exclusive scan → onesBefore (10-01 reuse, device-resident).
        let ones_before_h = full_scan_into::<f64>(&client, flags_h, n, false).unwrap();
        // Order-invariant zero count for this bit.
        let total_zeros = keys.iter().filter(|&&k| (k >> bit) & 1 == 0).count() as u32;
        // Stable scatter into the alternate buffers.
        reorder_one_bit_scatter_kernel::launch::<f64, crate::SelectedRuntime>(
            &client,
            count.clone(),
            dim32,
            unsafe { ArrayArg::from_raw_parts(cur_keys.clone(), n) },
            unsafe { ArrayArg::from_raw_parts(cur_vals.clone(), n) },
            unsafe { ArrayArg::from_raw_parts(ones_before_h, n) },
            unsafe { ArrayArg::from_raw_parts(alt_keys.clone(), n) },
            unsafe { ArrayArg::from_raw_parts(alt_vals.clone(), n) },
            bit,
            total_zeros,
        );
        std::mem::swap(&mut cur_keys, &mut alt_keys);
        std::mem::swap(&mut cur_vals, &mut alt_vals);
    }

    let kb = client.read_one(cur_keys).unwrap();
    let vb = client.read_one(cur_vals).unwrap();
    (
        bytemuck::cast_slice::<u8, u32>(&kb).to_vec(),
        bytemuck::cast_slice::<u8, u32>(&vb).to_vec(),
    )
}

/// Inline serial STABLE partition-by-bit reference (D-02): `bit==0` keys first (input
/// order preserved), then `bit==1` keys (input order preserved). Paired values move.
fn cpu_stable_partition_by_bit(keys: &[u32], values: &[u32], bit: u32) -> (Vec<u32>, Vec<u32>) {
    let mut out_keys = Vec::with_capacity(keys.len());
    let mut out_vals = Vec::with_capacity(keys.len());
    for (&k, &v) in keys.iter().zip(values) {
        if (k >> bit) & 1 == 0 {
            out_keys.push(k);
            out_vals.push(v);
        }
    }
    for (&k, &v) in keys.iter().zip(values) {
        if (k >> bit) & 1 == 1 {
            out_keys.push(k);
            out_vals.push(v);
        }
    }
    (out_keys, out_vals)
}

/// Inline serial STABLE sort reference (D-02): a stable sort of the (key, value) pairs
/// by key. `sort_by_key` is stable, so equal keys keep their input order.
fn cpu_stable_sort(keys: &[u32], values: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let mut idx: Vec<usize> = (0..keys.len()).collect();
    idx.sort_by_key(|&i| keys[i]);
    let out_keys = idx.iter().map(|&i| keys[i]).collect();
    let out_vals = idx.iter().map(|&i| values[i]).collect();
    (out_keys, out_vals)
}

#[test]
fn reorder_one_bit_behaviour_example() {
    // keys [3,1,2,0] at bit 0: zeros (even) [2,0] keep order, then ones (odd) [3,1] →
    // [2,0,3,1]. Values track their keys (use the input index as the value payload).
    let keys = vec![3u32, 1, 2, 0];
    let values = vec![0u32, 1, 2, 3];
    let (dk, dv) = run_reorder_one_bit(&keys, &values, 0);
    assert_eq!(dk, vec![2u32, 0, 3, 1], "reorder_one_bit keys at bit 0");
    // The paired values follow their keys: 2→idx2, 0→idx3, 3→idx0, 1→idx1.
    assert_eq!(dv, vec![2u32, 3, 0, 1], "reorder_one_bit values track keys");
}

#[test]
fn reorder_one_bit_matches_serial_stable_partition_large_n() {
    // n = 5000 >> CUBE_DIM: a stable single-bit reorder vs the inline serial stable
    // partition, across several bit positions.
    let n = 5000usize;
    let keys: Vec<u32> = (0..n).map(|k| (k as u32).wrapping_mul(2654435761).wrapping_add(7) & 0xFFFF).collect();
    let values: Vec<u32> = (0..n as u32).collect();
    for &bit in &[0u32, 1, 3, 7, 15] {
        let (dk, dv) = run_reorder_one_bit(&keys, &values, bit);
        let (ek, ev) = cpu_stable_partition_by_bit(&keys, &values, bit);
        assert_eq!(dk, ek, "reorder keys mismatch at bit {bit}, n={n}");
        assert_eq!(dv, ev, "reorder values mismatch at bit {bit}, n={n}");
    }
}

#[test]
fn radix_sort_behaviour_example() {
    // LSD radix sort of [5,3,9,1,7] → [1,3,5,7,9]; values track their keys.
    let keys = vec![5u32, 3, 9, 1, 7];
    let values = vec![0u32, 1, 2, 3, 4];
    let (dk, dv) = run_radix_sort(&keys, &values);
    assert_eq!(dk, vec![1u32, 3, 5, 7, 9], "radix sort keys");
    // 1→idx3, 3→idx1, 5→idx0, 7→idx4, 9→idx2.
    assert_eq!(dv, vec![3u32, 1, 0, 4, 2], "radix sort values track keys");
}

#[test]
fn radix_sort_is_stable_on_duplicate_keys() {
    // Duplicate keys MUST keep their input order (T-10-11). The value payload is the
    // input index, so a stable sort yields ascending value runs within each equal-key
    // group.
    let keys = vec![2u32, 1, 2, 1, 2, 0, 1];
    let values = vec![0u32, 1, 2, 3, 4, 5, 6]; // input indices
    let (dk, dv) = run_radix_sort(&keys, &values);

    // Keys sorted.
    let (ek, ev) = cpu_stable_sort(&keys, &values);
    assert_eq!(dk, ek, "radix sort keys must match the stable serial sort");
    assert_eq!(dv, ev, "radix sort must be STABLE (values match the stable serial sort)");

    // Explicitly: key 0 → [5]; key 1 → indices [1,3,6] ascending; key 2 → [0,2,4].
    assert_eq!(dk, vec![0u32, 1, 1, 1, 2, 2, 2]);
    assert_eq!(dv, vec![5u32, 1, 3, 6, 0, 2, 4]);
}

#[test]
fn radix_sort_matches_serial_stable_sort_large_n() {
    // n = 5000 >> CUBE_DIM, many duplicate keys (mod 97) to stress stability at scale.
    let n = 5000usize;
    let keys: Vec<u32> = (0..n).map(|k| ((k as u32).wrapping_mul(1103515245).wrapping_add(12345) >> 3) % 97).collect();
    let values: Vec<u32> = (0..n as u32).collect();

    let (dk, dv) = run_radix_sort(&keys, &values);
    let (ek, ev) = cpu_stable_sort(&keys, &values);

    assert_eq!(dk, ek, "radix sort keys mismatch at n={n}");
    assert_eq!(dv, ev, "radix sort not stable at n={n}");
}
