---
phase: 10-gpu-foundations-runtime-seam-session-residency-device-primit
plan: 04
subsystem: infra
tags: [cubecl, gpu, rocm, radix-sort, single-bit-reorder, tdatapartition, fill-transform, gput-16]

# Dependency graph
requires:
  - phase: 10-01 (device primitives)
    provides: full_scan_into two-level cross-cube exclusive scan (reused for onesBefore in reorder + run-index in partitions), kernels/scan.rs oracle harness shape
  - phase: 10-03 (reduce family)
    provides: key_head_flag_kernel (reused for partition run-head detection), segment_offset_scatter host-scalar num_segments precedent, kernels/reduce.rs launcher shape
  - phase: 07 (kernels.rs)
    provides: histogram_scatter_kernel per-object scatter shape copied for reorder/fill/gather
provides:
  - radix_bit_flag_kernel + reorder_one_bit_scatter_kernel (stable single-bit reorder over the 10-01 exclusive scan; scatter to zeroesBefore / total_zeros+onesBefore)
  - LSD radix sort composed from the single-bit reorder (device-resident ping-pong, one pass per bit, host-scalar order-invariant total_zeros)
  - update_partition_offsets_kernel + update_partition_sizes_kernel (TDataPartition {Offset,Size} from a sorted partition-id array via head-flag → exclusive scan → offsets scatter → per-run size expand)
  - fill_kernel, gather_kernel, vector_add/sub/mul/div_kernel (one-write-per-lane generic-float transforms)
affects: [11-histograms (depth>1 partition {Offset,Size}), later sort/partition/transform consumers]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Stable single-bit reorder = bit-flag kernel → 10-01 exclusive full_scan (onesBefore) → stable scatter to (i-onesBefore) / (total_zeros+onesBefore); total_zeros host-scalar order-invariant"
    - "LSD radix = device-resident ping-pong of single-bit reorders, one pass per bit up to the highest set bit; only the final buffers read back"
    - "TDataPartition update = key_head_flag → exclusive full_scan (run index) → offsets scatter keyed by partition VALUE (gap-safe) + compact run_keys/run_starts → per-run size expand; host-scalar num_runs; empty partitions host-seeded {0,0}"
    - "CubeCL nested-index mutable write (sizes[run_keys[r]]) must extract the inner index to a local first (macro mis-borrows the inner array as mut otherwise)"

key-files:
  created:
    - crates/cb-backend/src/kernels/sort.rs
    - crates/cb-backend/src/kernels/partitions.rs
    - crates/cb-backend/src/kernels/fill_transform.rs
  modified:
    - crates/cb-backend/src/kernels.rs

key-decisions:
  - "Radix-2 LSD (one bit per pass) as the composition from reorder_one_bit — directly satisfies both the must-have 'composed from it' and the plan's 'per-digit histogram → exclusive scan → scatter' (a 1-bit digit IS that pipeline). Device-resident ping-pong keeps handles on device across passes."
  - "total_zeros (reorder) and num_runs (partitions) are host scalars: they only SIZE buffers / select the scatter branch and are order-invariant; the parity-critical positions (onesBefore, run starts) are computed on-device — mirrors 10-03 reduce-by-key's host-scalar num_segments."
  - "update_partition_offsets scatters keyed by the partition VALUE (offsets[part_ids[i]]) not the run index, so gaps (empty partitions) are gap-safe; the run index from the scan is consumed as the COMPACT run_keys/run_starts index (key_link honoured — partitions update consumes the 10-01 scan)."
  - "Empty partitions come back {Offset:0, Size:0} via host-seeded output buffers (well-defined per plan); the inline serial reference produces the identical seed so the oracle is self-consistent."

patterns-established:
  - "Sort/partition/transform primitive oracles mirror kernels/scan.rs: launch over SelectedRuntime, inline serial reference (D-02), integer primitives asserted BIT-EXACT, small + n>>CUBE_DIM cases"

requirements-completed: [GPUT-16]

# Metrics
duration: ~40min
completed: 2026-07-03
status: complete
---

# Phase 10 Plan 04: Sort/Reorder + TDataPartition Update + Fill/Transform Primitives Summary

**From-scratch CubeCL sort/partition/transform substrate: a stable single-bit reorder (`reorder_one_bit`) built on the 10-01 exclusive scan and an LSD radix sort composed from it (device-resident ping-pong), the TDataPartition `{Offset,Size}` update (head-flag → exclusive scan → offsets scatter → per-run size expand, gap-safe for empty partitions), and the trivial fill/gather/vector-arithmetic transforms — each with a separate-file inline serial self-oracle (integer primitives BIT-EXACT), all green on rocm gfx1100 in-env.**

## Performance
- **Duration:** ~40 min
- **Completed:** 2026-07-03
- **Tasks:** 3
- **Files modified:** 4 (1 modified, 3 created)

## Accomplishments
- **Stable single-bit reorder** (`radix_bit_flag_kernel` + `reorder_one_bit_scatter_kernel`): extract `(key>>bit)&1` into a 0/1 float flag → 10-01 exclusive `full_scan` (onesBefore) → stable scatter to `i - onesBefore` (bit 0) or `total_zeros + onesBefore` (bit 1). Behaviour example `[3,1,2,0]` at bit 0 → `[2,0,3,1]` verified; paired values track keys.
- **LSD radix sort** composed from the single-bit reorder: device-resident ping-pong over the resident key/value handles, one stable pass per bit up to the highest set bit, only the final buffers read back. `total_zeros[bit]` is order-invariant so it is host-computed per bit. `[5,3,9,1,7]` → `[1,3,5,7,9]` verified; duplicate-key **stability** (T-10-11) asserted at n=7 and n=5000.
- **TDataPartition `{Offset,Size}` update** (`update_partition_offsets_kernel` + `update_partition_sizes_kernel`): reuse 10-03 `key_head_flag_kernel` → exclusive `full_scan` (run index per element) → offsets scatter (keyed by partition value → gap-safe) emitting compact `run_keys`/`run_starts` → per-run size expand. Behaviour `[0,0,0,1,1]` → part 0 `{0,3}`, part 1 `{3,2}` verified; **empty-partition** case `[0,0,2,2,2]` → part 1 `{0,0}` verified; n>>CUBE_DIM with runs>CUBE_DIM + trailing empty vs serial.
- **fill / gather / vector arithmetic** (`fill_kernel`, `gather_kernel`, `vector_add/sub/mul/div_kernel`): one-write-per-lane bounds-guarded generic-float kernels (mirror `histogram_scatter_kernel`). Elementwise self-oracles on small + n>>CUBE_DIM; full validation is transitive through the depth-1 tree + cindex (D-01).
- **rocm gfx1100 in-env:** sort 5/5, partitions 3/3, fill_transform 3/3 green; no regression (scan 17/17, reduce 8/8 re-run green).

## Task Commits
1. **Task 1: stable single-bit reorder + LSD radix sort** — `bc64d72` (feat)
2. **Task 2: TDataPartition offset/size update** — `e9c37f1` (feat)
3. **Task 3: fill / gather / vector-arithmetic transforms** — `f4a700a` (feat)

_Kernel + oracle committed together per task (the self-oracle cannot compile without the kernel it exercises — GPU-kernel TDD constraint; tdd_mode inactive for this phase, per 10-01/10-03 precedent)._

## Files Created/Modified
- `crates/cb-backend/src/kernels.rs` — added `radix_bit_flag_kernel`, `reorder_one_bit_scatter_kernel`, `update_partition_offsets_kernel`, `update_partition_sizes_kernel`, `fill_kernel`, `gather_kernel`, `vector_add/sub/mul/div_kernel`; mounted `kernels::sort`, `kernels::partitions`, `kernels::fill_transform`.
- `crates/cb-backend/src/kernels/sort.rs` — NEW: `run_reorder_one_bit`/`run_radix_sort` launchers, inline serial `cpu_stable_partition_by_bit`/`cpu_stable_sort` references, 5 tests (behaviour + stability + large-n).
- `crates/cb-backend/src/kernels/partitions.rs` — NEW: `run_partition_update` launcher, inline serial `cpu_partition_update` reference, 3 tests (behaviour + empty + large-n).
- `crates/cb-backend/src/kernels/fill_transform.rs` — NEW: `run_fill`/`run_gather`/`run_vector` launchers, inline serial elementwise references, 3 tests.

## Decisions Made
- **Radix-2 LSD (one bit per pass) as the composition from `reorder_one_bit`.** A 1-bit digit IS the plan's "per-digit histogram → exclusive scan → scatter" (the histogram of a 1-bit digit is exactly the ones-flag, its exclusive scan is onesBefore). This satisfies both the must-have ("an LSD radix sort composed from it") and the `<action>` text with the minimal, provably-stable construction. Device-resident ping-pong keeps handles on device across passes (only the final read-back).
- **Host-scalar `total_zeros` / `num_runs`.** Both only SIZE buffers or select the scatter branch and are order-invariant; the parity-critical positions (onesBefore, run starts) are on-device. Mirrors 10-03 reduce-by-key's host-scalar `num_segments` — recorded, not a scope cut.
- **Offsets scatter keyed by partition VALUE (gap-safe).** `offsets[part_ids[i]] = i` at each head means absent partitions are never written and keep their well-defined seed; the run index from the exclusive scan is consumed as the COMPACT `run_keys`/`run_starts` index, so the `key_link` ("partitions update consumes the 10-01 scan") is honoured with no dead scan output.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] CubeCL nested-index mutable write rejected**
- **Found during:** Task 2 (`update_partition_sizes_kernel`)
- **Issue:** `sizes[run_keys[r as usize] as usize] = end - start;` failed to compile — the `#[cube]` macro mis-borrowed the inner `run_keys` array as mutable (`E0596: cannot borrow *run_keys as mutable`).
- **Fix:** Extract the inner index to a local first (`let p = run_keys[r as usize]; sizes[p as usize] = end - start;`).
- **Files modified:** crates/cb-backend/src/kernels.rs
- **Committed in:** `e9c37f1` (Task 2 commit)

**2. [Rule 1 - Bug] Debug-build multiply overflow in the sort oracle's test-data generator**
- **Found during:** Task 1 (`reorder_one_bit_matches_serial_stable_partition_large_n`)
- **Issue:** `(k as u32) * 2654435761` overflowed in the debug test profile (panic in the TEST harness, not the kernel — the other 4 sort tests, incl. radix at n=5000, passed).
- **Fix:** `(k as u32).wrapping_mul(2654435761)`.
- **Files modified:** crates/cb-backend/src/kernels/sort.rs
- **Committed in:** `bc64d72` (Task 1 commit — caught and fixed before the commit)

**Total deviations:** 2 auto-fixed (1 blocking CubeCL macro quirk, 1 test-harness bug). Same public primitive surface and oracle discipline as planned.

## Known Stubs
None — all primitives are wired and oracle-verified; no placeholder/mock data paths.

## Threat Flags
None beyond the plan's `<threat_model>`. T-10-09 (scatter target OOB) mitigated by exclusive-scan-derived positions bounded to `[0,n)` + a bounds guard on every access; T-10-10 (portability UB) mitigated by no `-inf` literal + rocm smoke green; T-10-11 (silent instability) mitigated by the duplicate-key stability assertion vs a serial stable sort at n=7 and n=5000.

## Next Phase Readiness
- The stable single-bit reorder, LSD radix sort, `TDataPartition {Offset,Size}` update, and fill/gather/vector-arithmetic complete the from-scratch primitive substrate alongside scan (10-01) and reduce (10-03) — Phase 11 depth>1 histograms key on the partition `{Offset,Size}` layout.
- Human-gated acceptance still open (per plan): Kaggle CUDA authoritative sort/reorder/partitions ≤1e-4 / bit-exact via the 10-09 bench harness — not in-CI.

## Self-Check: PASSED
- Files: `crates/cb-backend/src/kernels.rs`, `crates/cb-backend/src/kernels/sort.rs`, `crates/cb-backend/src/kernels/partitions.rs`, `crates/cb-backend/src/kernels/fill_transform.rs` — all FOUND.
- Commits: `bc64d72`, `e9c37f1`, `f4a700a` — all FOUND.
- Acceptance greps: `reorder_one_bit` present in kernels.rs; `update_partition_offsets_kernel` present; `fill_kernel` present.
- rocm gfx1100 in-env: sort 5/5, partitions 3/3, fill_transform 3/3 green; scan 17/17, reduce 8/8 (no regression).

---
*Phase: 10-gpu-foundations-runtime-seam-session-residency-device-primit*
*Completed: 2026-07-03*
