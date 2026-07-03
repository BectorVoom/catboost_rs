---
phase: 10-gpu-foundations-runtime-seam-session-residency-device-primit
plan: 03
subsystem: infra
tags: [cubecl, gpu, rocm, segmented-reduce, reduce-by-key, deterministic-reduction, fixed-point-atomics, gput-16]

# Dependency graph
requires:
  - phase: 10-01 (device primitives)
    provides: full_scan / full_scan_into two-level cross-cube scan (reused for reduce-by-key key-run detection), block_reduce_kernel intra-cube fold, kernels/reduce.rs oracle harness
  - phase: 07.6
    provides: AtomicFinalizePath capability-gate + HostSumFallback precedent, cb_core::sum_f64 sanctioned ordered baseline
provides:
  - segmented_reduce_kernel (one cube per segment, f64 accumulation, fixed-order tree reduce — deterministic)
  - reduce_by_key device pipeline (key_head_flag_kernel + exclusive full_scan_into + segment_offset_scatter_kernel + reduce_by_key_kernel) → compacted (keys, f64 sums)
  - block_reduce_fixedpoint_kernel (round(v*2^30)→i64→u64 integer atomics; exact, order-independent) + REDUCE_FIXEDPOINT_SCALE_F64
  - 3 selectable deterministic scalar finalize strategies (fixed-order tree / host-sum / fixed-point u64 atomic) with a 32-launch zero-run-to-run-spread + capability-path variance harness
  - SPIKE-REDUCTION.md (candidate err+determinism table, per-backend viability, winner recommendation feeding Phase 11)
affects: [10-05 update_part_props, 11-histograms]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "One-cube-per-segment reduce with f64 accumulation + fixed-order shared-mem tree reduce (no cross-cube contention ⇒ deterministic without atomics)"
    - "reduce-by-key = host-scalar num_segments + on-device flag→exclusive-scan→offset-scatter→per-run sum (reuses 10-01 full_scan_into device-resident)"
    - "Fixed-point u64 integer-atomic finalize (round(v*2^30)→i64→u64 bits) for order-independent exact cross-cube determinism (manual 09_fixedpoint_atomics)"
    - "Selectable finalize strategy that REPORTS which path ran (capability downgrade explicit, never silent)"

key-files:
  created:
    - .planning/phases/10-gpu-foundations-runtime-seam-session-residency-device-primit/SPIKE-REDUCTION.md
  modified:
    - crates/cb-backend/src/kernels.rs
    - crates/cb-backend/src/kernels/reduce.rs

key-decisions:
  - "reduce-by-key derives num_segments (a scalar count) host-side while the boundary POSITIONS and per-run SUMS are computed on-device — the parity-critical work stays on device; num_segments only sizes the output buffers"
  - "Introduced a local ReduceFinalizeStrategy enum in kernels/reduce.rs rather than extending gpu_runtime::AtomicFinalizePath (which is NOT in files_modified): the strategy launchers are the test-side harness, so the reported-strategy enum lives with them; the existing AtomicFinalizePath contract is still asserted by the pre-existing atomic-finalize test"
  - "Per-segment reduces ship the fixed-order f64 tree reduce (no contention → deterministic, no atomic dependency); the scalar/histogram accumulator winner is fixed-point u64 atomics (single-pass device-resident) with the tree reduce as capability fallback"

patterns-established:
  - "Deterministic finalize is a SELECTABLE, self-reporting strategy verified by a 32-launch byte-identical variance harness"

requirements-completed: [GPUT-16]

# Metrics
duration: ~45min
completed: 2026-07-03
status: complete
---

# Phase 10 Plan 03: Reduce Family (Segmented-Reduce + Reduce-by-Key) + Reduction-Determinism Spike Summary

**From-scratch CubeCL reduce family — segmented-reduce and an on-device reduce-by-key pipeline (both f64-accumulated, fixed-order, deterministic) — plus the D-03/D-04 reduction-determinism spike delivering 3 selectable deterministic finalize strategies (fixed-order tree / host-sum / fixed-point u64 atomics) with a 32-launch zero-run-to-run-spread + capability-path variance harness, all green on rocm gfx1100 in-env, and a SPIKE-REDUCTION.md recommending the winner feeding Phase 11.**

## Performance
- **Duration:** ~45 min
- **Completed:** 2026-07-03
- **Tasks:** 3
- **Files modified:** 3 (2 modified, 1 created)

## Accomplishments
- **Segmented-reduce** (`segmented_reduce_kernel`): one cube per segment over `seg_offsets` (num_segments+1 boundary array), f64 accumulation regardless of channel float type, fixed-order shared-mem tree reduce → deterministic. Behaviour example `[1,2,3,4]` / offsets `{0,2,4}` → `[3,7]` verified.
- **Reduce-by-key** device pipeline: `key_head_flag_kernel` (phase 1) → exclusive `full_scan_into` of flags (phase 2, **10-01 scan reused** for key-run detection, device-resident) → `segment_offset_scatter_kernel` (phase 3) → `reduce_by_key_kernel` (per-run key + f64 sum). Emits a compacted `(keys, sums)` list. Behaviour example keys `[a,a,b,b,b]` values `[1×5]` → keys `[a,b]` sums `[2,3]` verified.
- **3 selectable deterministic finalize strategies** for the scalar cross-cube reduce: (a) fixed-order recursive tree reduce, (b) block-then-host-sum (`HostSumFallback`), (c) fixed-point u64 atomics (`block_reduce_fixedpoint_kernel`, `round(v·2³⁰)→i64→u64`). Each self-reports the path it ran.
- **Variance harness** (`reduce_finalize_strategies_are_deterministic_and_report_path`): 32 launches per strategy, asserts **byte-identical** results (zero run-to-run spread — T-10-06) and that the reported strategy matches the device's advertised capability (a silent switch FAILS — T-10-07).
- **Serial self-oracles** (D-02, inline, no cb-train reach): per-segment sum and group-by-sum baselined via `cb_core::sum_f64`; f32+f64, with segments/runs > CUBE_DIM to exercise the grid-stride intra-segment fold.
- **SPIKE-REDUCTION.md**: candidate set, in-env err+determinism table, per-backend viability, CUDA err+ms rows marked TBD-awaiting-Kaggle, and a firm recommendation.
- **rocm gfx1100 in-env:** 8/8 `reduce` tests green (3 new); 17/17 `scan` green (no regression from the `full_scan_into` visibility widening).

## Task Commits
1. **Task 1: Segmented-reduce + reduce-by-key kernels + oracle** — `ba7ae5c` (feat)
2. **Task 2: Deterministic finalize strategies + variance harness** — `b7f7763` (feat)
3. **Task 3: SPIKE-REDUCTION.md report + recommendation** — `5382fe5` (docs)

_Kernel + oracle committed together per task (the self-oracle cannot compile without the kernel it exercises — GPU-kernel TDD constraint; tdd_mode inactive for this phase, per 10-01 precedent)._

## Files Created/Modified
- `crates/cb-backend/src/kernels.rs` — added `segmented_reduce_kernel`, `key_head_flag_kernel`, `segment_offset_scatter_kernel`, `reduce_by_key_kernel`, `block_reduce_fixedpoint_kernel`, `REDUCE_FIXEDPOINT_SCALE_F64`; widened `full_scan_into` to `pub(crate)`.
- `crates/cb-backend/src/kernels/reduce.rs` — added `run_segmented_reduce`/`cpu_segmented_reduce`, `run_reduce_by_key`/`cpu_reduce_by_key`, `ReduceFinalizeStrategy` + 3 strategy launchers (`run_fixed_order_tree_reduce`, `run_host_sum_finalize`, `run_fixedpoint_reduce`) + `tree_reduce_into` + `device_supports_u64_atomic_add` + `assert_zero_spread`; 3 new tests.
- `.planning/phases/10-.../SPIKE-REDUCTION.md` — the reduction-determinism spike report.

## Key Measured Finding
**gfx1100 ADVERTISES `Atomic<u64>` add** (even though it does NOT advertise `Atomic<f64>` add — Phase 7.6). The plan anticipated the fixed-point path would only exercise on CUDA with gfx1100 falling back to `HostSum`; in practice the fixed-point u64 kernel (`f64::round` + `Atomic<u64>::fetch_add`) **JITs cleanly and runs on-device in-env on gfx1100**, byte-exact with zero spread. This validates the manual's wide-integer rationale (`09_fixedpoint_atomics.md §5`) and makes the fixed-point strategy the recommended histogram-accumulator winner on BOTH in-env backends (not CUDA-only).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Widened `full_scan_into` to `pub(crate)`**
- **Found during:** Task 1 (reduce-by-key device pipeline)
- **Issue:** The plan requires reduce-by-key key-run detection "reusing the 10-01 full/segmented scan"; the device-resident exclusive scan of the flag array needs `full_scan_into` (the handle-in/handle-out level), but it was a private `fn`. The public `full_scan` does a host round-trip, which would break device residency between the flag kernel and the offset scatter.
- **Fix:** Widened `full_scan_into` to `pub(crate)` with a doc note; the reduce-by-key launcher calls it directly, keeping flags → seg-ids on-device.
- **Files modified:** `crates/cb-backend/src/kernels.rs`
- **Committed in:** `ba7ae5c`

### Scope-respecting choice (not a functional deviation)

**2. Local `ReduceFinalizeStrategy` enum instead of extending `AtomicFinalizePath`**
- The plan text suggested "extending the existing `AtomicFinalizePath` enum". That enum lives in `crates/cb-backend/src/gpu_runtime/mod.rs`, which is **not** in this plan's `files_modified` (kernels.rs, reduce.rs, SPIKE-REDUCTION.md). The strategy launchers + variance harness are the test-side oracle, so the reported-strategy enum (`ReduceFinalizeStrategy`) was introduced **locally in `kernels/reduce.rs`** with the launchers — respecting the stated file surface and avoiding a cross-module change. The must-have (2-3 selectable deterministic strategies that report which ran, no silent switch) is fully satisfied; the pre-existing `AtomicFinalizePath` contract is still exercised by the untouched `block_reduce_atomic_finalize_matches_cpu_sum_and_reports_variance` test.

**Total deviations:** 1 auto-fixed (Rule 3) + 1 scope-respecting choice. Same public primitive surface and oracle discipline as planned.

## Design Notes / Scope
- **reduce-by-key `num_segments` is host-scalar:** the launcher counts distinct key runs host-side (only to size the output buffers); the boundary POSITIONS (flag → scan → scatter) and the per-run SUMS are computed on-device. The parity-critical device work is the sums.
- **Per-segment vs scalar finalize:** segmented/by-key reduces have one cube per segment (no cross-cube contention) → the fixed-order f64 tree reduce is already deterministic and ships as-is. The fixed-point u64 atomic strategy targets the many-cubes-contend-on-one-cell case (Phase 11 histogram accumulator).

## Known Stubs
None — all primitives are wired and oracle-verified; no placeholder/mock data paths.

## Threat Flags
None beyond the plan's `<threat_model>`. T-10-06 (nondeterminism) mitigated by the zero-spread assertion over 32 launches; T-10-07 (silent capability downgrade) mitigated by the reported-strategy assertion; T-10-08 (portability UB) mitigated by no `-inf` literal + rocm smoke green.

## Next Phase Readiness
- Segmented-reduce, reduce-by-key, and the deterministic finalize strategies are ready for 10-05 `update_part_props` and Phase 11 histograms.
- SPIKE-REDUCTION.md recommendation (fixed-point u64 atomics for the histogram accumulator, fixed-order tree reduce as capability fallback) is Phase 11's step-0 input.
- Human-gated acceptance still open (per plan): Kaggle CUDA authoritative segmented-reduce + reduce-by-key ≤1e-4 and per-candidate err+ms (fills the SPIKE §4 TBD rows) via the 10-09 bench harness — not in-CI.

## Self-Check: PASSED
- Files: `crates/cb-backend/src/kernels.rs`, `crates/cb-backend/src/kernels/reduce.rs`, `.planning/phases/10-.../SPIKE-REDUCTION.md` — all FOUND.
- Commits: `ba7ae5c`, `b7f7763`, `5382fe5` — all FOUND.
- Acceptance: `reduce_by_key` present in kernels.rs; `Recommendation` present in SPIKE-REDUCTION.md.
- rocm gfx1100 in-env: `cargo test -p cb-backend --no-default-features --features rocm reduce` → 8 passed, 0 failed.

---
*Phase: 10-gpu-foundations-runtime-seam-session-residency-device-primit*
*Completed: 2026-07-03*
