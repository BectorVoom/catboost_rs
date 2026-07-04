---
phase: 13-pairwise-ranking-multiclass-ordered-langevin-device-coverage
plan: 06
subsystem: infra
tags: [cubecl, gpu, multiclass, newton, cholesky, leaf-solve, device-seam]

# Dependency graph
requires:
  - phase: 13-05
    provides: device stochastic-ranking pair (prior wave; sequencing dependency only)
  - phase: 12
    provides: DeviceGrownTree seam struct (plain host carrier), device grow-tree seam over SelectedRuntime
  - phase: 13-02
    provides: crate::kernels::cholesky_solve serial batched f64 SPD solver idiom (launcher + host readback)
provides:
  - "DeviceGrownTree.approx_dim: usize + leaf_values reinterpreted as a leaf_count × approx_dim row-major block (scalar byte-unchanged at approx_dim==1, GPUT-14/D-04)"
  - "crate::kernels::multi_newton — K-dim Newton der2 block-leaf solve (coupled full K×K softmax + diagonal per-component), transcribing solve_symmetric_newton inline"
  - "block apply reference approx[d*n+i] += lr*leaf_block[leaf_of[i]*K+d] routed through the existing multi-output CPU apply layout"
affects: [13-07, multiclass, multi-output, multi-target, langevin]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "DeviceGrownTree carrier extension (plain host usize field, =1 for scalar paths, byte-unchanged) — same idiom as step_nodes/region_path"
    - "Serial #[cube] batched block solve mirroring kernels/cholesky_solve.rs (unit-0 loop, f64 non-atomic per-matrix arithmetic, wgpu typed reject, host readback oracle)"

key-files:
  created:
    - crates/cb-compute/src/runtime_test.rs
    - crates/cb-backend/src/kernels/multi_newton.rs
    - crates/cb-backend/src/kernels/multi_newton_test.rs
  modified:
    - crates/cb-compute/src/runtime.rs
    - crates/cb-compute/src/lib.rs
    - crates/cb-backend/src/kernels.rs
    - crates/cb-backend/src/kernels/apply_leaf_delta.rs
    - crates/cb-backend/src/gpu_runtime/session.rs
    - crates/cb-backend/src/kernels/nonsym_grow.rs
    - crates/cb-backend/src/kernels/region_device.rs
    - crates/cb-train/tests/device_seam_test.rs

key-decisions:
  - "Diagonal mode consumes the SAME packed lower-triangular der2 buffer as coupled mode, reading only the packed diagonal entries (uniform launcher input; == solve_symmetric_newton with k=1 per component)"
  - "Block apply reuses the existing scalar apply_leaf_delta_kernel per dimension (multilogit.cu applies one dimension at a time) rather than adding a new K-dim scatter kernel"
  - "approx_dim=1 set at ALL scalar construction sites (3 in cb-backend, 5 in cb-train test) so leaf_values bytes are IDENTICAL to the pre-Phase-13 flat vector (D-04)"

patterns-established:
  - "Multi-output leaf block = leaf_count × approx_dim row-major; dimension d of leaf l is leaf_values[l*approx_dim+d]; scalar path collapses to the flat vector"

requirements-completed: [GPUT-12]

# Metrics
duration: ~35min
completed: 2026-07-04
status: complete
---

# Phase 13 Plan 06: Multi-output prerequisite sub-wave (block leaves + K-dim Newton solve) Summary

**DeviceGrownTree extended with approx_dim block leaves (scalar byte-unchanged) plus a serial CubeCL K-dim Newton der2 block-leaf solve — coupled full K×K softmax vs diagonal per-component — self-oracled ≤1e-4 against solve_symmetric_newton.**

## Performance

- **Duration:** ~35 min
- **Completed:** 2026-07-04T08:53Z
- **Tasks:** 3
- **Files modified:** 11 (3 created, 8 modified)

## Accomplishments
- `DeviceGrownTree` carries `approx_dim: usize`; `leaf_values` documented + used as a `leaf_count × approx_dim` row-major block, byte-unchanged at `approx_dim==1` (GPUT-14 / D-04).
- New `multi_newton.rs` `#[cube]` batched block solve: packed lower-triangular hessian reconstruction, `maxTrace`/`adjustedL2` f32 regularization, `M = -(H − adjustedL2·I)`, inline Cholesky block solve, `res = -x`; coupled/diagonal dispatch via a mode flag; non-PD pivot → zeros (T-13-11).
- Block apply extension `approx[d*n+i] += lr*leaf_block[leaf_of[i]*K+d]` routed through the existing scalar multi-output apply, plus a K=1-collapses-to-scalar oracle.
- Self-oracle (coupled K=3 softmax, diagonal K=2, non-PD zeros, scalar K=1 no-regression) passes; numeric asserts ≤1e-4 vs `solve_symmetric_newton` are hard on rocm/cuda, record-only on cpu (WR-01 anti-false-pass).

## Task Commits

1. **Task 1: DeviceGrownTree block-leaf extension + block apply** - `33cc187` (feat)
2. **Task 2: K-dim Newton der2 block solve kernel (coupled + diagonal)** - `ef1a9a5` (feat)
3. **Task 3: Self-oracle — K-dim block solve vs solve_symmetric_newton** - `a78befe` (test)

## Files Created/Modified
- `crates/cb-compute/src/runtime.rs` - `DeviceGrownTree.approx_dim` field + block-leaf layout docs
- `crates/cb-compute/src/runtime_test.rs` - block reinterpretation + scalar-collapse + apply-layout self-oracle
- `crates/cb-compute/src/lib.rs` - mount `runtime_test`
- `crates/cb-backend/src/kernels/multi_newton.rs` - K-dim Newton der2 block solve kernel + launcher + host readback wrapper
- `crates/cb-backend/src/kernels/multi_newton_test.rs` - device-vs-CPU block-solve self-oracle
- `crates/cb-backend/src/kernels.rs` - register `multi_newton` + `multi_newton_test`
- `crates/cb-backend/src/kernels/apply_leaf_delta.rs` - block apply reference + device run + K=1 no-regression
- `crates/cb-backend/src/gpu_runtime/session.rs`, `kernels/nonsym_grow.rs`, `kernels/region_device.rs` - `approx_dim: 1` at scalar construction sites
- `crates/cb-train/tests/device_seam_test.rs` - `approx_dim: 1` at 5 test construction sites

## Decisions Made
- Diagonal mode reuses the coupled mode's packed der2 input (reads only diagonal entries) for a uniform launcher signature; equals per-component `solve_symmetric_newton(k=1)`.
- Block apply reuses the scalar `apply_leaf_delta_kernel` per dimension rather than introducing a new K-dim scatter kernel — faithful to `multilogit.cu` (one dimension at a time) and keeps Task 1 to the listed files.
- `approx_dim=1` at every scalar construction site keeps the oblivious/non-symmetric/Region emissions byte-identical (D-04).

## Deviations from Plan

None - plan executed exactly as written. (One in-test host-helper guard was added for the `d==0` packed-diagonal-index underflow; the kernel runs the same expression in wrapping GPU arithmetic and is unaffected — this is a test-only correctness guard, not a plan deviation.)

## Issues Encountered
- The `diag_index` host helper in the test panicked on `d==0` (`d-1` usize underflow in debug). Guarded with an early `return 0`. The `#[cube]` kernel evaluates the identical expression in non-overflow-checked GPU arithmetic (K=1 diagonal fixture passed before the fix), so the kernel needed no change.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- The multi-output representation (`DeviceGrownTree.approx_dim` block leaves) and the K-dim Newton block solve (`crate::kernels::multi_newton::solve_multi_newton_host` / `launch_multi_newton_solve`) are ready to be consumed by Plan 07 (the multiclass / multi-output family), which wires the der/weight sums into the block solve and emits `approx_dim > 1` trees.
- Device numeric sign-off (ε=1e-4 asserts on real CUDA) is deferred to Plan 10's Kaggle CUDA gate — the cpu-backend run is record-only by design (WR-01). rocm/cuda in-env execution not run in this executor.

## Self-Check: PASSED
- All created files present (runtime_test.rs, multi_newton.rs, multi_newton_test.rs, 13-06-SUMMARY.md).
- All task commits present in git (33cc187, ef1a9a5, a78befe).
- cb-compute / cb-train / cb-backend all `cargo check --tests` clean; no cb-train dep in cb-backend.
