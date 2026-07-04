//! Self-oracle for the SEGMENTED radix-sort primitive (Phase 12 Plan 05, Open Q1 / A1,
//! GPUT-19): [`crate::kernels::exact_quantile::segmented_radix_sort`] must sort keys+values
//! STABLY and ASCENDING *within each flag-delimited segment* and NEVER mix elements across a
//! segment boundary. Integer keys ⇒ the per-segment match against a serial stable sort is
//! BIT-EXACT (tighter than the ≤1e-4 float bar, D-07).
//!
//! Source/test separation is mandatory (CLAUDE.md / AGENTS.md): the primitive lives in the
//! production `kernels::exact_quantile` module; ALL assertions / `.unwrap()` / indexing live
//! here. The serial reference is transcribed INLINE — no `cb-train` reach, no upstream/CUB
//! fixture (D-02). Runs over the generic [`crate::SelectedRuntime`] (cpu default + rocm
//! in-env on gfx1100 wave32).

use crate::kernels::exact_quantile::segmented_radix_sort;

/// Inline serial per-segment STABLE sort reference (D-02): sort each `[seg_start, seg_end)`
/// slice by key (stable — equal keys keep input order), never crossing a segment boundary.
/// `head_flags[i] == 1` marks a segment start; `head_flags[0]` is assumed `1`.
fn cpu_segmented_stable_sort(
    head_flags: &[u32],
    keys: &[u32],
    values: &[u32],
) -> (Vec<u32>, Vec<u32>) {
    let n = keys.len();
    let mut out_keys = vec![0u32; n];
    let mut out_vals = vec![0u32; n];
    let mut seg_start = 0usize;
    let mut i = 1usize;
    while i <= n {
        let boundary = i == n || head_flags[i] == 1;
        if boundary {
            let mut order: Vec<usize> = (seg_start..i).collect();
            // Stable sort by key (sort_by_key is stable).
            order.sort_by_key(|&k| keys[k]);
            for (off, &src) in order.iter().enumerate() {
                out_keys[seg_start + off] = keys[src];
                out_vals[seg_start + off] = values[src];
            }
            seg_start = i;
        }
        i += 1;
    }
    (out_keys, out_vals)
}

#[test]
fn segmented_sort_behaviour_example() {
    // Two segments: [0..3) = [5,3,9] → [3,5,9]; [3..6) = [2,8,1] → [1,2,8]. Values track keys.
    let head = vec![1u32, 0, 0, 1, 0, 0];
    let keys = vec![5u32, 3, 9, 2, 8, 1];
    let values = vec![0u32, 1, 2, 3, 4, 5]; // input indices
    let (dk, dv) = segmented_radix_sort(&head, &keys, &values).unwrap();
    assert_eq!(dk, vec![3u32, 5, 9, 1, 2, 8], "per-segment sorted keys");
    // Segment 0: 3→idx1, 5→idx0, 9→idx2. Segment 1: 1→idx5, 2→idx3, 8→idx4.
    assert_eq!(dv, vec![1u32, 0, 2, 5, 3, 4], "values track keys, no cross-segment mix");
}

#[test]
fn segmented_sort_no_cross_segment_mixing() {
    // A LARGER key in an EARLIER segment must stay in its segment (never bleed into a later
    // segment even though it is globally larger). Segment 0 holds 100; segment 1 holds 1.
    let head = vec![1u32, 0, 1, 0];
    let keys = vec![100u32, 50, 1, 2];
    let values = vec![0u32, 1, 2, 3];
    let (dk, dv) = segmented_radix_sort(&head, &keys, &values).unwrap();
    // Segment 0 sorts to [50,100] (the 100 stays in segment 0), segment 1 to [1,2].
    assert_eq!(dk, vec![50u32, 100, 1, 2], "the large key stays in its own segment");
    assert_eq!(dv, vec![1u32, 0, 2, 3]);
}

#[test]
fn segmented_sort_stable_on_duplicate_keys() {
    // Duplicate keys within a segment MUST keep input order (stable). Value payload = index.
    let head = vec![1u32, 0, 0, 0, 0];
    let keys = vec![2u32, 1, 2, 1, 2];
    let values = vec![0u32, 1, 2, 3, 4];
    let (dk, dv) = segmented_radix_sort(&head, &keys, &values).unwrap();
    let (ek, ev) = cpu_segmented_stable_sort(&head, &keys, &values);
    assert_eq!(dk, ek, "keys match the serial stable sort");
    assert_eq!(dv, ev, "STABLE: duplicate-key groups keep input order");
    // Explicitly: key 1 → [1,3] ascending; key 2 → [0,2,4].
    assert_eq!(dk, vec![1u32, 1, 2, 2, 2]);
    assert_eq!(dv, vec![1u32, 3, 0, 2, 4]);
}

#[test]
fn segmented_sort_matches_serial_many_varied_segments() {
    // Many segments of DIFFERING lengths (incl. length-1), duplicate keys (mod 53) to stress
    // per-segment stability + boundaries at scale.
    let n = 900usize;
    let keys: Vec<u32> = (0..n)
        .map(|k| ((k as u32).wrapping_mul(2654435761).wrapping_add(7) >> 4) % 53)
        .collect();
    let values: Vec<u32> = (0..n as u32).collect();
    // Segment starts at 0 and then irregular strides (lengths 1,2,3,...).
    let mut head = vec![0u32; n];
    head[0] = 1;
    let mut pos = 0usize;
    let mut seg_len = 1usize;
    while pos < n {
        head[pos] = 1;
        pos += seg_len;
        seg_len = (seg_len % 7) + 1;
    }

    let (dk, dv) = segmented_radix_sort(&head, &keys, &values).unwrap();
    let (ek, ev) = cpu_segmented_stable_sort(&head, &keys, &values);
    assert_eq!(dk, ek, "segmented sort keys mismatch (cross-segment mixing or wrong order)");
    assert_eq!(dv, ev, "segmented sort not stable / mixed segments at n={n}");
}
