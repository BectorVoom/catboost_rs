---
phase: 10-gpu-foundations-runtime-seam-session-residency-device-primit
plan: 01
subsystem: infra
tags: [cubecl, gpu, rocm, prefix-scan, segmented-scan, device-primitives, gput-16]

# Dependency graph
requires:
  - phase: 07-cuda-structural-parity (7.1 reduce/scan primitives)
    provides: block_scan_kernel single-cube prefix scan + kernels/scan.rs self-oracle harness
  - phase: 07.1
    provides: BLOCK_REDUCE_SHMEM comptime shared-mem sizing + wave-agnostic plane-op scan structure
provides:
  - Cross-cube two-level full prefix scan (inclusive + exclusive) over arbitrary n via recursive block-sum exclusive scan
  - block_scan_total_kernel (per-block scan + block totals) and add_block_offset_kernel (phase-3 offset add)
  - full_scan / full_scan_into public launcher over SelectedRuntime (resident handles, no intermediate read-back)
  - Flag-array segmented scan (segmented_scan_kernel, TSegmentedSum pair-combiner, inclusive + exclusive)
  - Serial-CPU self-oracles at n >> CUBE_DIM (scan) and multi-segment/full-cube (segmented) in separate test files
affects: [10-02 sort, 10-03 compression, 10-04 partitions, 10-05 update_part_props, 11-histograms]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Two-level (per-block scan -> exclusive scan of block sums -> add block offset) cross-cube scan, recursion on block sums for arbitrary n"
    - "Block-total emitted from the last VALID unit of each block (flag-independent inclusive_val = scanned_in_plane + carry)"
    - "Segmented Hillis-Steele: if flag[tid]==0 { val+=left }; flag |= left (register snapshot before in-place shared write)"

key-files:
  created:
    - crates/cb-backend/src/kernels/segmented_scan.rs
  modified:
    - crates/cb-backend/src/kernels.rs
    - crates/cb-backend/src/kernels/scan.rs

key-decisions:
  - "Recursive two-level scan (not single-cube mid-scan): the plan's 'single cube suffices' assumption fails at n=100_000 with CUBE_DIM=32 (~3125 block sums >> 32); recursing the exclusive scan of block sums covers arbitrary n"
  - "Reuse block_scan_kernel geometry for the base case and a dedicated block_scan_total_kernel for the block-total emission so the offset exactly matches the local scan (self-consistent, no second input read)"
  - "Segmented scan single-cube scope documented (n <= CUBE_DIM), mirroring block_scan_kernel Open Q2; cross-cube segmented carry deferred as the named forward dependency (plan-sanctioned)"

patterns-established:
  - "Cross-cube full scan launcher living in kernels.rs, driven from the kernels/scan.rs oracle over SelectedRuntime"
  - "Segmented pair-combiner scan with exact 0.0/1.0 flag tests (assigned, never float-accumulated)"

requirements-completed: [GPUT-16]

# Metrics
duration: ~35min
completed: 2026-07-03
status: complete
---

# Phase 10 Plan 01: GPU Device Primitives — Cross-Cube Full Scan + Segmented Scan Summary

**From-scratch CubeCL device-primitive substrate: a recursive two-level cross-cube prefix scan (inclusive + exclusive, correct at n=100_000 >> CUBE_DIM) and a flag-array segmented scan, each with a separate-file serial-CPU self-oracle, all green on rocm gfx1100 in-env.**

## Performance

- **Duration:** ~35 min
- **Started:** 2026-07-03
- **Completed:** 2026-07-03
- **Tasks:** 2
- **Files modified:** 3 (2 modified, 1 created)

## Accomplishments
- Two-level cross-cube full prefix scan generalizing the single-cube `block_scan_kernel` to arbitrary `n` (RESEARCH Open Q2 closed): `block_scan_total_kernel` (phase 1) + recursive exclusive scan of block sums (phase 2) + `add_block_offset_kernel` (phase 3), exposed via `full_scan` over `SelectedRuntime`.
- Flag-array segmented scan (`segmented_scan_kernel`) with the upstream `TSegmentedSum` pair-combiner semantics (segment-boundary reset), inclusive + exclusive comptime variants.
- Serial-CPU self-oracles in separate test files (D-01/D-02): `kernels/scan.rs` extended with n=100_000 inclusive/exclusive/f32 cases + behaviour example + edge cases; new `kernels/segmented_scan.rs` with behaviour example, multi-segment inclusive+exclusive, and full-cube (n=32) boundary cases.
- rocm gfx1100 in-env smoke green: 17/17 (`scan` filter covers scan + segmented_scan), no regression to the existing single-cube tests.

## Task Commits

Each task was committed atomically:

1. **Task 1: Cross-cube full prefix scan (inclusive + exclusive)** - `56015eb` (feat)
2. **Task 2: Flag-array segmented scan** - `580d0d9` (feat)

_Note: kernel + oracle committed together per task — the self-oracle cannot compile without the kernel it exercises (GPU-kernel TDD constraint; tdd_mode inactive for this phase)._

## Files Created/Modified
- `crates/cb-backend/src/kernels.rs` - Added `SCAN_CUBE_DIM` const, `block_scan_total_kernel`, `add_block_offset_kernel`, `full_scan_into` (recursive), `full_scan` (public launcher), `segmented_scan_kernel`; mounted `kernels::segmented_scan`; imported `CbError`/`CbResult`/`Handle`.
- `crates/cb-backend/src/kernels/scan.rs` - Extended oracle: `full_scan` behaviour example, edge cases (empty/n=1), and n=100_000 inclusive/exclusive/f32 cases vs inline serial prefix reference with generous run-stable bounds.
- `crates/cb-backend/src/kernels/segmented_scan.rs` - New self-oracle: `run_segmented_scan` launcher (single-cube scope), inline serial `cpu_segmented_inclusive`/`cpu_segmented_exclusive` references, `max_divergence`, and 4 tests.

## Decisions Made
- **Recursive two-level scan over the plan's "single cube suffices" mid-scan.** At n=100_000 with CUBE_DIM=32 there are ~3125 block sums — far more than one cube can scan. The exclusive scan of block sums recurses through `full_scan_into` (100000 → 3125 → 98 → 4 → base case), covering arbitrary `n`. This is strictly more general than the plan text and fully correct; recorded as a Rule-1/Rule-3 correctness generalization (see Deviations).
- **Dedicated `block_scan_total_kernel` for block-total emission** (rather than a separate `block_reduce` pass): the block total is derived as the flag-independent inclusive prefix (`scanned_in_plane + carry`) at the last valid unit, so the per-block offset is exactly self-consistent with the local scan and the input is read once.
- **Segmented scan single-cube scope documented**, mirroring the `block_scan_kernel` Open Q2 note. The cross-cube segmented carry (block tail sum propagated into the next block only up to its first segment head) is named as the forward dependency — a plan-sanctioned choice (the `<action>` permits "documenting the single-cube scope explicitly as scan.rs does"), not a silent scope cut.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Correctness] Two-level scan generalized to arbitrary n via recursion**
- **Found during:** Task 1 (Cross-cube full prefix scan)
- **Issue:** The plan's action states "an exclusive scan of block_sums (single cube suffices since n_blocks is small at bench scale)". At the plan's own oracle scale (n=100_000, CUBE_DIM=32) there are ~3125 block sums, which a single 32-wide cube cannot scan — a single-cube mid-scan would silently produce wrong prefixes for n > CUBE_DIM^2.
- **Fix:** Made `full_scan_into` recursive: phase 2 calls `full_scan_into(block_sums, ...)` with the exclusive flag, terminating once the block sums fit one cube. Fully correct for arbitrary `n`.
- **Files modified:** crates/cb-backend/src/kernels.rs
- **Verification:** n=100_000 inclusive/exclusive/f32 oracles pass on rocm gfx1100 (rel div within generous run-stable bounds; a missing offset would diverge by orders of magnitude).
- **Committed in:** `56015eb` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 correctness generalization)
**Impact on plan:** The deviation makes the delivered primitive strictly more general and correct than the literal plan text; the plan's own acceptance scale (n=100_000) requires it. No scope creep — same public surface (`full_scan`) and same oracle.

## Issues Encountered
- Initial compile error: `input.len()` is `usize` in the cube context, so `n - 1u32` failed to typecheck. Fixed by using `n - 1usize` in the last-valid-unit block-total guard. Resolved before the first commit.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- `full_scan` (cross-cube, inclusive/exclusive) and `segmented_scan_kernel` are ready as the substrate for 10-02 radix sort / single-bit reorder, 10-03 compression, 10-04 partitions, and 10-05 update_part_props.
- Forward dependency (documented, not blocking this plan): cross-cube segmented carry for `segmented_scan_kernel` (reuses the same two-level pattern + a head-seen mask) if a later consumer needs segmented scan over n > CUBE_DIM.
- Human-gated acceptance still open per plan: Kaggle CUDA authoritative oracle (full-scan + segmented-scan ≤1e-4 vs serial reference) via the 10-09 bench harness — not in-CI.

## Self-Check: PASSED
- Files: `crates/cb-backend/src/kernels.rs`, `crates/cb-backend/src/kernels/scan.rs`, `crates/cb-backend/src/kernels/segmented_scan.rs` — all FOUND.
- Commits: `56015eb`, `580d0d9` — both FOUND.
- Acceptance greps: scan oracle contains n=100_000 case (>> 4x CUBE_DIM); `segmented_scan` present in kernels.rs.
- rocm gfx1100 in-env: `cargo test -p cb-backend --no-default-features --features rocm scan` → 17 passed, 0 failed.

---
*Phase: 10-gpu-foundations-runtime-seam-session-residency-device-primit*
*Completed: 2026-07-03*
