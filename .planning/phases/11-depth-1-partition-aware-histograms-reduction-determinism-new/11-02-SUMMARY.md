---
phase: 11-depth-1-partition-aware-histograms-reduction-determinism-new
plan: 02
subsystem: cb-backend
tags: [cubecl, kernels, histogram, partition-aware, subtraction-trick, fixed-point, determinism, gpu, rocm]

# Dependency graph
requires:
  - phase: 11-depth-1-partition-aware-histograms-reduction-determinism-new
    plan: 01
    provides: "cb-compute reduce_leaf_stats CPU leaf-keyed scatter oracle + depth-6 fixture A1/A2 pinning"
  - phase: 10-gpu-foundations-runtime-seam-session-residency-device-primit
    provides: "LOCKED fixed-point Atomic<u64> accumulator (block_reduce_fixedpoint_kernel, REDUCE_FIXEDPOINT_SCALE_F64 k=30, SPIKE-REDUCTION §5b); read_bin packed-cindex accessor; partition_split/partition_update kernels"
provides:
  - "partition_hist2_nonbinary_kernel — fullPass=false leaf-keyed pointwise_hist2 filling 2^level slots (part * leaf_stride cell index), fixed-point Atomic<u64> deterministic accumulate (GPUT-06), channel-0 = weight/hessian (statId 0), channel-1 = der1"
  - "subtract_histograms_kernel — parent − smaller per fixed-point cell, statId==0 weight-channel max(0) clamp (upstream SubstractHistogramsImpl)"
  - "launch_partition_hist2_into / launch_subtract_histograms_into — checked_shl/checked_mul buffer sizing + leaf_of/indices/cindex value-range guards -> typed CbError before launch; read_fixedpoint_hist_f64 decode helper"
  - "grow_loop::partition_hist::partition_aware_hist_matches_cpu_scatter — kernel self-oracle vs reduce_leaf_stats <=1e-4 (levels 2 and 6) on gfx1100"
affects: [11-03, 11-04, 11-05, BENCH-02]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Partition-aware histogram = depth-1 pointwise_hist2 + a leaf/partition stride prepended to the FROZEN interleaved cell index (upstream §6.4 leaf-wise tensor layout)"
    - "GPUT-06 determinism = the LOCKED fixed-point Atomic<u64> path reused via fixedpoint_encode/decode #[cube] helpers — NO new reduction strategy, NO naked f64 fetch_add"
    - "Subtraction trick in FLOAT space (decode/subtract/encode) is bit-exact below 2^53 (both operands are exact integer sums of quantized contributions) and reuses only proven CubeCL ops (cast_from/round) to dodge the i64-arithmetic build risk"
    - "Slot-base-offset addressing (parent_base/smaller_base) lets one multi-slot buffer serve parent + child siblings — the D-04 memory-lean parent-resident reuse"

key-files:
  created: []
  modified:
    - crates/cb-backend/src/kernels.rs
    - crates/cb-backend/src/gpu_runtime/mod.rs
    - crates/cb-backend/src/kernels/grow_loop.rs

key-decisions:
  - "Channel layout = channel-0 weight/hessian (statId 0), channel-1 der1 — per the must-have; makes the subtraction max(0) clamp key literally on statId==0 (cell index ≡ 0 mod HIST_CHANNELS), matching upstream SubstractHistogramsImpl. (The plan's parenthetical 'consistent with depth-1' was factually inverted — the depth-1 kernel is der1-first; the binding must-have + subtraction-clamp semantics win.)"
  - "Subtraction performed in FLOAT (decode->subtract->encode) rather than raw i64: reuses only CubeCL ops already proven on gfx1100 (f64::cast_from/i64::cast_from/u64::cast_from/f64::round, float compare) — avoids unexercised i64 subtract/compare JIT risk; exact below 2^53, so ≤1e-4 is met with margin (observed bit-consistent)"
  - "leaf_of passed as a host &[u32] slice to launch_partition_hist2_into (uploaded + value-range-validated host-side) — the kernel-level self-oracle frames a 'known leaf_of assignment'; the resident-handle grow-loop wiring is Plan 03's job"
  - "Self-oracle compares the device histogram's feature-0 bin sum per leaf against reduce_leaf_stats (per-leaf total) — every object contributes exactly one feature-0 bin, so the bin sum equals the leaf total; directly exercises the named CPU oracle at both level 2 and depth-6 (64) slots"

requirements-completed: [GPUT-05, GPUT-06]

# Metrics
duration: 45min
completed: 2026-07-03
status: complete
---

# Phase 11 Plan 02: Partition-aware pointwise_hist2 + subtraction trick + deterministic accumulator Summary

**The `fullPass=false` leaf-keyed `pointwise_hist2` (2^level slots), the histogram subtraction trick (parent − smaller, weight-channel max(0) clamp), and the LOCKED fixed-point `Atomic<u64>` accumulator (GPUT-06) — the depth>1 device capability (GPUT-05), oracle-proven ≤1e-4 vs the CPU leaf-keyed scatter on real gfx1100.**

## Performance

- **Duration:** ~45 min
- **Completed:** 2026-07-03
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments
- `partition_hist2_nonbinary_kernel<F>` clones the depth-1 non-binary fill but routes every object into `leaf_of[obj]`'s slot, prepending the leaf stride to the FROZEN interleaved cell: `cell = part * (n_features*n_bins*HIST_CHANNELS) + (feature*n_bins+bin)*HIST_CHANNELS + channel` (upstream §6.4). Bins read through the one `read_bin` packed-cindex accessor.
- **GPUT-06 wired here:** the histogram merge uses the LOCKED deterministic fixed-point `Atomic<u64>` path (`fixedpoint_encode` = `round(v·2^30) → i64 → u64` bits, `REDUCE_FIXEDPOINT_SCALE_F64` k=30) — replacing the depth-1 kernel's naked f64 `fetch_add` (the accepted non-deterministic source). Integer atomic add is exact + order-independent ⇒ byte-identical run to run.
- `subtract_histograms_kernel<F>` derives the larger sibling as `parent − smaller` per cell (decoded to float, subtracted as upstream `SubstractHistogramsImpl` does on float `TBucketStats`, re-encoded), clamping the `statId==0` weight/hessian channel to `max(0)` (the LANDMINE guard — tiny negative weights would poison the score denominator). Slot-base offsets support the D-04 parent-resident reuse.
- `launch_partition_hist2_into` / `launch_subtract_histograms_into` validate ALL buffer sizing (`checked_shl` for `2^level`, `checked_mul` for the per-leaf line and total, `checked_add` for slot offsets) AND the `leaf_of[obj] < 2^level` / `indices[i] < n` / `cindex bin < n_bins` value ranges → typed `CbError::OutOfRange`/`LengthMismatch` BEFORE launch (T-11-02-01/02). `read_fixedpoint_hist_f64` decodes the u64 handle.
- `partition_aware_hist_matches_cpu_scatter` (grow_loop.rs) proves the fill + subtraction match `cb_compute::reduce_leaf_stats` to ≤1e-4 at level 2 (4 partitions) AND a depth-6-slot case (64 partitions) on real gfx1100, and asserts the weight channel is ≥0 after the clamp.

## Task Commits

1. **Tasks 1+2: partition-aware pointwise_hist2 + subtraction trick + launch fns** - `8e6dbf3` (feat)
2. **Task 3: kernel self-oracle vs CPU leaf-keyed scatter** - `7971e06` (test)

## Files Created/Modified
- `crates/cb-backend/src/kernels.rs` — added `fixedpoint_encode`/`fixedpoint_decode` `#[cube]` helpers, `partition_hist2_nonbinary_kernel<F>`, `subtract_histograms_kernel<F>` (modified).
- `crates/cb-backend/src/gpu_runtime/mod.rs` — added `launch_partition_hist2_into`, `launch_subtract_histograms_into`, `read_fixedpoint_hist_f64`; extended kernel imports + `REDUCE_FIXEDPOINT_SCALE_F64` import (modified).
- `crates/cb-backend/src/kernels/grow_loop.rs` — added the `partition_hist` self-oracle module + gpu_runtime imports (modified).

## Deviations from Plan

**Process (not a deviation rule): Tasks 1 and 2 committed together.** The subtraction kernel depends on Task 1's `fixedpoint_decode` `#[cube]` helper and both kernels + their launch fns were co-developed in the same two files; they form one interdependent build unit (both `rocm`-build-verify green as a unit — the shared verify for Tasks 1/2). Task 3 (the test) is a separate commit.

**[Plan-latitude] Subtraction computed in float, not raw i64 fixed-point.** The plan says "hist[fromId] -= hist[whatId]"; upstream `SubstractHistogramsImpl` subtracts float `TBucketStats`. Doing the subtraction in float (decode→subtract→encode) is (a) faithful to upstream, (b) bit-exact below 2^53 since both operands are exact integer sums of quantized contributions, and (c) reuses only CubeCL ops already proven on gfx1100 — avoiding the unexercised i64-subtract/compare JIT path (AGENTS.md build-risk avoidance). No CubeCL build errors occurred, so the error-guideline protocol was not triggered.

**[Plan-latitude] Channel-0 = weight, channel-1 = der1.** The must-have truth ("channel-0 = weight/hessian and channel-1 = Σder1") and the subtraction `statId==0` clamp semantics agree on weight-first; the plan action's parenthetical "consistent with the depth-1 layout" is factually inverted (the depth-1 `pointwise_hist2_nonbinary_kernel` is der1-first). Followed the binding must-have + clamp semantics so the clamp keys literally on `cell % HIST_CHANNELS == 0`.

## Issues Encountered
None. The `#[cube]` kernels JIT-compiled on gfx1100 first try (no `-inf` literal — the recalled landmine avoided; fixed-point helpers reuse the proven `block_reduce_fixedpoint_kernel` cast/round vocabulary). The self-oracle passed first run at both levels with the weight-channel clamp holding.

## Next Phase Readiness
- GPUT-05 (depth>1 device histogram capability) and GPUT-06 (deterministic accumulator) are now kernel-level complete and oracle-proven. Plan 03 can wire `launch_partition_hist2_into` + the subtraction trick into the depth>1 grow loop (replacing the depth-1 MVP's typed forward-dependency error), threading `leaf_of` as a resident handle instead of a host slice.
- A2 caveat carried forward: channel-0 = Σweight here (matching the depth-6 fixture's pinned score-channel). If Plan 04's Logloss-Newton path adopts a der2-in-channel-0 variant, it passes that as the `weight` argument (the layout is caller-parameterized) — no kernel change needed.

---
*Phase: 11-depth-1-partition-aware-histograms-reduction-determinism-new*
*Completed: 2026-07-03*

## Self-Check: PASSED
- Created file present: 11-02-SUMMARY.md
- Both task commits present: 8e6dbf3, 7971e06
- Source artifacts present: partition_hist2_nonbinary_kernel (kernels.rs), launch_partition_hist2_into (gpu_runtime/mod.rs)
