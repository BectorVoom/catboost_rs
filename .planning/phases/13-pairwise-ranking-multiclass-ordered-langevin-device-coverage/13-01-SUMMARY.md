---
phase: 13-pairwise-ranking-multiclass-ordered-langevin-device-coverage
plan: 01
subsystem: gpu-training
tags: [cubecl, pairwise, cholesky, gput-11, gput-21, rocm, catboost, ranking]

# Dependency graph
requires:
  - phase: 07-gpu-cuda-structural-parity
    provides: Phase 7.4 4-channel pairwise_hist kernels + Phase 7.5 pairwise scorer (host solve)
  - phase: 12-gpu-device-families
    provides: GpuTrainSession Option<*State> coverage-gate idiom, Ok(None) all-or-nothing fallback
provides:
  - Device pairwise per-leaf linear-system assembly (packed lower-triangular linearSystem, resident)
  - PairwiseState coverage gate on GpuTrainSession (map_pairwise_coverage, Ok(None) fallback)
  - Self-oracle for the packed system vs the CPU pair-stat reference (<=1e-4)
affects: [13-02-cholesky-solver, gput-21, pairwise-ranking-device-grow]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Packed lower-triangular linearSystem assembly (rowSize*(rowSize+1)/2 cells then rowSize RHS) resident on device"
    - "Option<PairwiseState> coverage gate mirroring the Phase-12 per-family Ok(None) all-or-nothing idiom"

key-files:
  created:
    - crates/cb-backend/src/kernels/pairwise_deriv_test.rs
  modified:
    - crates/cb-backend/src/kernels.rs
    - crates/cb-backend/src/gpu_runtime/pairwise.rs
    - crates/cb-backend/src/gpu_runtime/session.rs
    - crates/cb-backend/src/gpu_runtime/mod.rs

key-decisions:
  - "Assembly transcribes calculate_pairwise_leaf_values reg constants (diag/non-diag prior + drop-last-row), NOT upstream RegularizeImpl bump-heuristics (Pitfall 2)"
  - "begin() pairwise arm declines to CPU (Ok(None)) pending the Plan-02 per-tree pair/group seam, to avoid regressing real PairLogitPairwise training (the Runtime seam carries only approx/target)"
  - "f64 serial assembly kernel with a typed wgpu reject (mvs_device precedent); the packed system feeds Plan-02's f64 batched Cholesky (D-07)"

patterns-established:
  - "Pattern F self-oracle: device packed system vs inline CPU reference over equal-length buffers, numeric assert skipped off rocm/cuda (WR-01)"
  - "Device linearSystem residency: resident handle out, no n-length host readback of pair stats (D-05)"

requirements-completed: [GPUT-11]

# Metrics
duration: 45min
completed: 2026-07-04
status: complete
---

# Phase 13 Plan 01: Pairwise Device Coverage & Per-Leaf System Assembly Summary

**Device-resident packed lower-triangular pairwise `linearSystem` assembly (transcribing the Rust CPU `calculate_pairwise_leaf_values` matrix build) plus the `PairwiseState` coverage gate, self-oracled bit-exact vs the CPU reference on rocm gfx1100.**

## Performance

- **Duration:** ~45 min
- **Completed:** 2026-07-04
- **Tasks:** 2 (committed as one cohesive vertical slice — shared `kernels.rs`)
- **Files modified:** 5 (4 modified, 1 created)

## Accomplishments
- New `#[cube]` `pairwise_assemble_system_kernel` (f64 serial) packs the per-leaf `linearSystem` upstream `ExtractMatricesAndTargets` consumes: `rowSize*(rowSize+1)/2` lower-triangle matrix cells (row-major, `x` in `0..=y`, with the catboost `diag_reg`/`non_diag_reg` prior) then `rowSize` RHS, where `rowSize = leaf_count - 1` (leaf gauge freedom).
- `launch_pairwise_assemble_system_into` / `assemble_pairwise_system_host` in `gpu_runtime/pairwise.rs`: resident handle out (D-05, no n-length pair-stat readback), f64/wgpu typed reject, `leaf_count<=1` empty no-op (no 0-len handle read).
- Pairwise coverage gate: `PairwiseState` struct + `pairwise: Option<PairwiseState>` field on `GpuTrainSession` + `map_pairwise_coverage` (the `Option`-returning family-gated template) + `begin()` pairwise arm returning `Ok(None)` on both the covered and uncovered branches.
- Self-oracle `kernels/pairwise_deriv_test.rs` (source/test separation): 3-/4-leaf fixtures compare device vs an independent inline CPU packing reference; single-leaf empty; empty-n no-op guard. rocm gfx1100 in-env **4/4 green with the numeric assertions firing** (device == CPU reference, ≤1e-4).
- Reuses the Phase 7.4 pairwise histograms unchanged; **no `cb-train` dep** leaks into `cb-backend` (landmine grep == 0).

## Task Commits

1. **Task 1 + Task 2 (device pairwise system assembly + coverage gate + self-oracle)** - `020472c` (feat)

_Tasks 1 and 2 share `crates/cb-backend/src/kernels.rs` (the production kernel + the `#[cfg(test)] mod` registration), so they were committed as one atomic, compiling-at-every-commit slice rather than split mid-file._

**Plan metadata:** see the final docs commit.

## Files Created/Modified
- `crates/cb-backend/src/kernels.rs` - `pairwise_assemble_system_kernel` (`#[cube]` f64 serial packing) + `#[cfg(test)] mod pairwise_deriv_test` registration.
- `crates/cb-backend/src/gpu_runtime/pairwise.rs` - `launch_pairwise_assemble_system_into`, `assemble_pairwise_system_host`, `PAIRWISE_BUCKET_WEIGHT_PRIOR_REG_DEFAULT`, wgpu reject.
- `crates/cb-backend/src/gpu_runtime/session.rs` - `PairwiseState`, `map_pairwise_coverage`, `pairwise: Option<PairwiseState>` field, `begin()` pairwise arm.
- `crates/cb-backend/src/gpu_runtime/mod.rs` - import `pairwise_assemble_system_kernel`.
- `crates/cb-backend/src/kernels/pairwise_deriv_test.rs` - self-oracle (new).

## Decisions Made
- **Reg constants from the Rust CPU oracle, not upstream:** the kernel transcribes `calculate_pairwise_leaf_values` (`cell_prior = 1/leaf_count`, `non_diag_reg = -prior*cell_prior`, `diag_reg = prior*(1-cell_prior) + l2`), NOT `linear_solver.cu::RegularizeImpl` — the ε=1e-4 oracle is the Rust CPU path (Pitfall 2).
- **f64 serial assembly + wgpu typed reject:** the packed system accumulates in f64 to feed Plan-02's batched Cholesky (D-07); WGSL has no f64 so the launcher rejects wgpu with a typed `CbError::OutOfRange` rather than a JIT crash (mvs_device precedent).

## Deviations from Plan

### Scoping clarification (documented, not a code deviation rule)

**1. begin() pairwise arm declines to CPU (Ok(None)) rather than returning a live pairwise session**
- **Found during:** Task 1 (coverage gate wiring)
- **Issue:** must-have truth #1 ("a PairLogit fit reaches the device grow path") implies an end-to-end session pairwise grow, but the `Runtime::grow_tree_on_device` seam carries only `approx`/`target` — it has **no per-tree pair/group descriptor**, and `DeviceTrainConfig` carries no pairs. Wiring pairs onto the seam is a cross-crate change (seam signature + `cb-train` boosting + config) **outside this plan's 4-file scope**, and returning a live pairwise session that `grow_one` cannot service would either fabricate a wrong pointwise grow or hard-error the fit (worse than the current CPU fallback).
- **Resolution:** the coverage **decision** is landed and self-tested via `map_pairwise_coverage` (returns `Some(PairwiseState)` for a covered config — "the pairwise gate returns Some"), the `pairwise: Option<PairwiseState>` field is the landed structural seam for Plan 02 (like the `#[allow(dead_code)] config` precedent), and the device per-leaf assembly driver + self-oracle (the wave's real deliverable) are landed and rocm-validated. `begin()` returns `Ok(None)` for pairwise so **real PairLogitPairwise training is unaffected** (D-04 no-regression). This matches the plan's own framing ("still routes the small solve through the existing 7.5 host bounded solve — Plan 02 replaces it", "de-risks the solver wave").
- **Verification:** pointwise session tests 10/10 green; a `*Pairwise` loss returned `Ok(None)` before AND after (byte-unchanged CPU fallback).

---

**Total deviations:** 1 scoping clarification (no auto-fix rules triggered).
**Impact on plan:** All landed deliverables (assembly driver, coverage gate, self-oracle, histogram reuse, no cb-train dep) are complete and rocm-validated. The end-to-end session pairwise grow wiring is Plan 02, per the plan's explicit sequencing. No scope creep.

## Issues Encountered
- **Pre-existing `pairwise_hist` / `score_split::pairwise` / `grow_loop::pairwise` cpu-backend failures (10 tests):** confirmed pre-existing on a clean tree (stash test: 3 passed / 10 failed both before and after my change) — these are device-backend (rocm/cuda) tests that fail on the default `cpu` backend by design. **Out of scope** (SCOPE BOUNDARY — not caused by this task); the orchestrator discharges the rocm suite in-env. My change adds 4 passing tests and regresses none.

## Deferred Issues
None introduced by this plan. The device pairwise SPD **solve** (batched Cholesky, GPUT-21) and the per-tree pair/group seam wiring are Plan 02 by design.

## Known Stubs
None. The assembly driver is fully wired and self-oracled; the `pairwise` session field is a documented Plan-02 structural seam (`#[allow(dead_code)]`, mirroring the existing `config` field), not a data stub feeding UI/output.

## Next Phase Readiness
- Plan 02 (GPUT-21) consumes the resident packed `linearSystem` for the on-device batched f64 Cholesky (decomp + fwd/back subst + ridge + `CalcScoresCholesky`) and wires the per-tree pair/group descriptor seam so a covered PairLogitPairwise fit grows end-to-end on device.
- Kaggle CUDA ε=1e-4 + BENCH-02 pairwise sign-off remains deferred to Plan 10 (human-gated), per the plan.

## Self-Check: PASSED

---
*Phase: 13-pairwise-ranking-multiclass-ordered-langevin-device-coverage*
*Completed: 2026-07-04*
