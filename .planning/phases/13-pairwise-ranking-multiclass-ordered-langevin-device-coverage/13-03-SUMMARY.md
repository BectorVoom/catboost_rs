---
phase: 13-pairwise-ranking-multiclass-ordered-langevin-device-coverage
plan: 03
subsystem: infra
tags: [cubecl, gpu, ranking, query-grouping, segmented-sort, fixed-point-reduction, gput-22]

# Dependency graph
requires:
  - phase: 13-02
    provides: device batched f64 Cholesky solver + the serial #[cube] kernel skeleton this reuses
  - phase: 12-07
    provides: mvs_device inline PCG RNG transcription + wgpu-reject pattern
  - phase: 10-04
    provides: exact_quantile::segmented_radix_sort (reused for in-query sampling)
  - phase: 10-03
    provides: REDUCE_FIXEDPOINT_SCALE_F64 k=30 deterministic fixed-point reduction
provides:
  - Shared device query-grouping kernel surface (query_helper.rs) amortized across all 5 query/listwise objectives
  - ComputeGroupIds / ComputeGroupMeans / ComputeGroupMax / RemoveGroupMeans on device (fixed-point group der/weight sums)
  - CreateSortKeys + shuffle_within_queries_host (reuses segmented_radix_sort; no second sort)
  - FillTakenDocsMask / FillQueryEndMask / ComputeSampledSizes (SampledQuerySize ≥2 floor)
  - Self-oracle vs CPU ranking_der group reductions at ε=1e-4
affects: [13-04, 13-05, ranking, query-listwise-objectives]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Serial #[cube] group reduction over query offsets with k=30 fixed-point integer accumulation for determinism"
    - "In-query shuffle composes CreateSortKeys (u32 random low-32 keys) with the existing segmented_radix_sort under query head-flags"
    - "Backend-split test gating: serial u32/f64 kernels hard-assert on cpu; plane-scan (segmented sort) + numeric ε skip off rocm/cuda (WR-01)"

key-files:
  created:
    - crates/cb-backend/src/kernels/query_helper.rs
    - crates/cb-backend/src/kernels/query_helper_test.rs
  modified:
    - crates/cb-backend/src/kernels.rs

key-decisions:
  - "Serial (unit-0) group reductions with wrapping k=30 fixed-point integer accumulation (REDUCE_FIXEDPOINT_SCALE_F64) — deterministic and satisfies the T-13-06 not-f64-atomic-add contract without a real Atomic<u64> (gfx1100 has no f64 atomic-add anyway); matches the mvs_device serial precedent"
  - "CreateSortKeys emits the u32 random low-32 sort key; the conceptual (qid<<32)|random high bits are supplied implicitly by the segmented sort's query head-flags, so no second/global sort is hand-rolled"
  - "Contiguity test device-gated: segmented_radix_sort's underlying full_scan uses plane_inclusive_sum, unsupported on the cpu backend"

patterns-established:
  - "query_helper device grouping substrate — one resident grouping infra amortized across QueryRMSE/QuerySoftMax/QueryCrossEntropy/YetiRank/PFound-F (Plans 04–05)"

requirements-completed: [GPUT-22]

# Metrics
duration: ~35min
completed: 2026-07-04
status: complete
---

# Phase 13 Plan 03: Shared Device Query-Grouping Infrastructure Summary

**Device query-grouping kernel surface (group ids/weighted-means/max, per-query bias removal, in-query random sort keys, taken-docs + query-end masks) — group der/weight sums via the k=30 fixed-point deterministic path, in-query sampling reusing the existing segmented radix sort, self-oracled against the CPU `ranking_der` group reductions at ε=1e-4.**

## Performance

- **Duration:** ~35 min
- **Completed:** 2026-07-04
- **Tasks:** 2
- **Files modified:** 3 (2 created, 1 modified)

## Accomplishments
- New `query_helper.rs` with the full grouping kernel surface: `ComputeGroupIds`, `ComputeGroupMeans` (weighted, fixed-point), `ComputeGroupMax`, `RemoveGroupMeans` (doc-parallel), `CreateSortKeys` (inline PCG), `FillTakenDocsMask`, `FillQueryEndMask`, `ComputeSampledSizes` (`SampledQuerySize` ≥2 floor).
- Der/weight group SUMS route through the k=30 fixed-point `REDUCE_FIXEDPOINT_SCALE_F64` integer accumulation — deterministic (T-13-06), never f64 atomic-add.
- In-query random shuffle (`shuffle_within_queries_host`) composes `CreateSortKeys` with the EXISTING `exact_quantile::segmented_radix_sort` under per-query head-flags — no second sort algorithm introduced.
- Self-oracle (`query_helper_test.rs`, 6 tests) vs an independent inline serial CPU reference of the `cb_compute::ranking_der` group reductions (baselined via `cb_core::sum_f64`) at ε=1e-4 — all 6 green on the cpu backend.
- No `cb-train` dependency added (feature-unification landmine avoided); RNG transcribed inline; f64/u64 wgpu reject at every entry point; no `-inf` literal in any `#[cube]` body.

## Task Commits

1. **Task 1: Device query-grouping kernels** - `8393b31` (feat)
2. **Task 2: Self-oracle vs CPU ranking_der group reductions** - `2a77373` (test)

_Note: this plan's tasks are marked `tdd="true"`; `tdd_mode` is disabled project-wide (config.json), so each task was landed as a single atomic commit (kernel, then oracle) rather than split RED/GREEN commits._

## Files Created/Modified
- `crates/cb-backend/src/kernels/query_helper.rs` - The `#[cube]` grouping kernels + host readback/launch wrappers (fixed-point weighted means, group max, bias removal, sort keys, masks, sampled sizes, in-query shuffle).
- `crates/cb-backend/src/kernels/query_helper_test.rs` - Self-oracle: group means/max/bias-removal vs CPU (ε=1e-4, numeric assert gated to rocm/cuda), query-contiguity (device-gated), `SampledQuerySize` ≥2 floor + mask coverage (backend-independent).
- `crates/cb-backend/src/kernels.rs` - Registered `pub(crate) mod query_helper` + `#[cfg(test)] mod query_helper_test`.

## Decisions Made
- **Serial fixed-point group reduction over warp-per-query.** The PATTERNS analog described a warp-per-query WarpReduce; the shipped kernels use the simpler serial (unit-0) accumulation established by `mvs_device.rs`, with wrapping k=30 fixed-point integer sums for the der/weight reductions. This is deterministic, satisfies the T-13-06 "not f64 atomic-add" contract, references `REDUCE_FIXEDPOINT_SCALE_F64`, and defers the warp-parallel geometry as a perf follow-up (perf is an MVP concern per the Plan 06/07 precedent). The covered fixtures are small (single-block).
- **`CreateSortKeys` emits u32 low-32 keys; qid high bits are implicit in the segmentation.** `segmented_radix_sort` keeps queries contiguous by construction (per-segment slice sort under query head-flags), so the conceptual `(qid<<32)|random` global key never needs a 64-bit global sort — the low 32 random bits shuffle within a query while the head-flags preserve query order.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Device-gated the in-query contiguity test**
- **Found during:** Task 2 (self-oracle)
- **Issue:** `segmented_radix_sort`'s underlying `full_scan` uses `plane_inclusive_sum`, which the CPU cubecl backend does not implement — it panics/produces a non-permutation on the default `cpu` test run.
- **Fix:** Added a `device_backend_active()` skip guard to `create_sort_keys_keep_queries_contiguous` (WR-01), matching the established mvs/cholesky/segmented-sort device-only gating. The backend-independent invariants (`SampledQuerySize` floor, taken/query-end masks) still hard-assert on cpu.
- **Files modified:** crates/cb-backend/src/kernels/query_helper_test.rs
- **Verification:** `cargo test -p cb-backend --lib query_helper` → 6/6 pass on cpu.
- **Committed in:** `2a77373` (Task 2 commit)

**2. [Rule 2 - Missing coverage] Added a mask-coverage test (Test 5)**
- **Found during:** Task 2
- **Issue:** `FillTakenDocsMask` / `FillQueryEndMask` are required plan artifacts (provides) but were otherwise unexercised (dead-code warnings, matching the codebase's next-plan-API pattern).
- **Fix:** Added `taken_and_query_end_masks_match_cpu` — a backend-independent (serial u32) hard-assert against a CPU reference, genuinely exercising both masks on cpu.
- **Files modified:** crates/cb-backend/src/kernels/query_helper_test.rs
- **Verification:** Test green on cpu.
- **Committed in:** `2a77373` (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (1 blocking, 1 missing coverage)
**Impact on plan:** Both necessary for a green cpu-backend test run and genuine artifact coverage. No scope creep — all artifacts as specified.

## Issues Encountered
- **CubeCL `.len()` returns `usize`, not `u32`.** The initial kernel bodies used `u32` group counters; CubeCL's `Array::len()` is `usize`, so group loop counters were switched to `usize` (indexing offsets read `u32` values, cast where writing to `u32` arrays). Resolved during Task 1 compile.
- **Full `cargo test -p cb-backend --lib` shows 60 pre-existing failures on the cpu backend** (exact_quantile, sort, segmented_sort, reduce-atomic) — device-only tests that use plane/device features unsupported on cpu and hard-fail off rocm/cuda by design. NOT caused by this task (SCOPE BOUNDARY); logged to `deferred-items.md`. The plan's verify commands (`query_helper` / `query_helper_test`) pass 6/6.

## Deferred Issues
- **Kaggle CUDA ε=1e-4 sign-off** for the group reductions is deferred to Plan 10 (per the plan's `success_criteria`); the numeric ε assertions are device-gated and validated in-env on rocm/cuda by the orchestrator (the cpu run records-only, WR-01).
- **Warp-per-query parallelization** of the group reductions is a perf follow-up (shipped serial, correctness-first).

## Next Phase Readiness
- The shared query-grouping substrate is ready for amortization by Plans 04 (QueryRMSE/QuerySoftMax/QueryCrossEntropy) and 05 (YetiRank/PFound-F).
- Plan 10 will run the per-family Kaggle CUDA ε=1e-4 + BENCH-02 sign-off (human-gated).

---
*Phase: 13-pairwise-ranking-multiclass-ordered-langevin-device-coverage*
*Completed: 2026-07-04*

## Self-Check: PASSED
- FOUND: crates/cb-backend/src/kernels/query_helper.rs
- FOUND: crates/cb-backend/src/kernels/query_helper_test.rs
- FOUND: .planning/phases/13-.../13-03-SUMMARY.md
- FOUND commit 8393b31 (Task 1), 2a77373 (Task 2)
