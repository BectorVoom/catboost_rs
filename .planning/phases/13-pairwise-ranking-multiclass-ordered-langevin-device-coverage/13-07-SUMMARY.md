---
phase: 13-pairwise-ranking-multiclass-ordered-langevin-device-coverage
plan: 07
subsystem: infra
tags: [cubecl, gpu, multiclass, multi-output, newton, block-leaf, coverage-gate, device-seam]

# Dependency graph
requires:
  - phase: 13-06
    provides: DeviceGrownTree approx_dim block leaves + crate::kernels::multi_newton K-dim Newton der2 block solve (coupled/diagonal)
  - phase: 12
    provides: GpuTrainSession coverage-gate idiom (Option<*State> family-gated Pattern A), DeviceGrownTree seam
provides:
  - "crate::gpu_runtime::multiclass — multi-output block-leaf device DRIVER (grow_multiclass_block): assembles per-object der via existing cb-compute der fns, accumulates per-leaf sum_der (K) + sum_der2_packed (K(K+1)/2) via ordered cb_core::sum_f64, dispatches the Plan-06 device K-dim Newton block solve (coupled softmax vs diagonal separable), emits the leaf_count × K row-major DeviceGrownTree block"
  - "GpuTrainSession multiclass coverage gate (Option<MulticlassState> + map_multiclass_coverage): a covered multi-output loss declines to Ok(None)→CPU pending the per-tree shared multi-dim grow seam (pairwise/ranking precedent, D-04 no-regression)"
  - "map_multiclass_objective: loss → MulticlassObjective classification (coupled ONLY for MultiClass softmax; diagonal for OneVsAll/MultiCrossEntropy/MultiRMSE/RMSEWithUncertainty)"
affects: [13-08, 13-09, 13-10, langevin, multi-output]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Multi-output block driver = per-object der assembly (cb-compute) + ordered per-leaf reduction (cb_core::sum_f64, D-08 same order as compute_softmax_leaf_deltas) + batched device multi_newton solve → leaf_count × K row-major block (DeviceGrownTree contract)"
    - "Coverage-gate Pattern A extended to the multi-output family: Option<MulticlassState> records the coverage DECISION; begin() declines to CPU pending the forward-dependency grow seam (pairwise/ranking precedent)"

key-files:
  created:
    - crates/cb-backend/src/gpu_runtime/multiclass.rs
    - crates/cb-backend/src/gpu_runtime/multiclass_test.rs
  modified:
    - crates/cb-backend/src/gpu_runtime/mod.rs
    - crates/cb-backend/src/gpu_runtime/session.rs

key-decisions:
  - "The block driver batches the device multi_newton solve over ALL leaves at once (batch = n_leaves, system-major sum_der[leaf*K+d]) so the read-back IS the leaf_count × K row-major DeviceGrownTree block directly — no transpose"
  - "Diagonal (separable) losses fill ONLY the packed diagonal entries of the per-object der2_packed buffer (off-diagonal = 0); the device diagonal mode reads only those, matching per-component solve_symmetric_newton(k==1) — the Plan-06 uniform-launcher contract"
  - "Covered multi-output configs decline to Ok(None)→CPU (like pairwise/ranking): grow_one has no multi-dim (approx_dim>1) arm, so returning Ok(Some) would fabricate a SCALAR grow on a multi-output fit. The per-tree shared multi-dim grow seam is a forward dependency"
  - "MultiRMSE has no Loss variant yet — classified (MulticlassObjective::MultiRmse, diagonal) so the arm lands when the variant is added; the three self-oracle fixtures use MultiClass/RMSEWithUncertainty/MultiClassOneVsAll (both hessian structures covered, RESEARCH A2)"

patterns-established:
  - "Multi-output leaf estimation on device = one shared tree structure (shared leaf_of across K dims) + per-leaf K-dim Newton block solve; coupled full K×K ONLY for softmax, diagonal per-component for the separable losses"

requirements-completed: [GPUT-12]

# Metrics
duration: ~45min
completed: 2026-07-04
status: complete
---

# Phase 13 Plan 07: Multiclass / multi-output / uncertainty device coverage Summary

**Wired the multi-output loss family (MultiClass softmax, MultiClassOneVsAll, MultiLogloss/MultiCrossEntropy, MultiRMSE, RMSEWithUncertainty) onto the Plan-06 block-leaf + K-dim Newton machinery: a device block-leaf DRIVER that grows ONE shared tree, solves a K-dim der2 block per leaf (coupled softmax vs diagonal separable), and emits the leaf_count × K row-major DeviceGrownTree block — self-oracled ≤1e-4 vs the CPU solve_symmetric_newton multi-output leaf values — plus the multiclass coverage gate.**

## Performance
- **Duration:** ~45 min
- **Completed:** 2026-07-04
- **Tasks:** 2
- **Files:** 4 (2 created, 2 modified)

## Accomplishments
- New `gpu_runtime/multiclass.rs`: `map_multiclass_objective` (loss → coupled/diagonal classification), `assemble_multiclass_ders` (per-object der from the EXISTING `cb-compute` der fns — `softmax_ders`/`multiclass_onevsall_ders`/`multi_crossentropy_ders`/`rmse_with_uncertainty_ders` — the device transcribes the emission, not the der), and `grow_multiclass_block` (ordered per-leaf `sum_der`/`sum_der2_packed` reduction → batched device `launch_multi_newton_solve` → `leaf_count × K` row-major block).
- `session.rs`: `MulticlassState` coverage-gate field (`Option`) + `map_multiclass_coverage` (SymmetricTree, depth≥1, Plain, single fold, all-or-nothing family default); `begin()` intercepts multi-output losses and returns `Ok(None)` (declines to CPU pending the per-tree shared multi-dim grow seam — the pairwise/ranking precedent, never a fabricated scalar grow on a multi-output fit).
- Coupled full-block solve is used ONLY for `MultiClass` softmax; the four separable losses use the diagonal per-component path (RESEARCH Pitfall 3, VERIFIED and asserted).
- Self-oracle (`multiclass_test.rs`): coupled softmax K=3, diagonal RMSEWithUncertainty K=2 (distinct row-0/row-1 hessian), diagonal MultiClassOneVsAll K=3 block leaves == CPU `solve_symmetric_newton` multi-output leaf values ≤1e-4; objective/coupled-dispatch classification; covered + uncovered configs both `Ok(None)`. Numeric ε asserts are device-gated (hard on rocm/cuda, record-only on cpu — WR-01 anti-false-pass), the whole file gated off wgpu.

## Task Commits
1. **Task 1: Multi-output device driver + coverage gate (coupled + diagonal dispatch)** - `1ed441e` (feat)
2. **Task 2: Frozen multiclass fixtures + end-to-end block self-oracle** - `a10abf8` (test)

## Files Created/Modified
- `crates/cb-backend/src/gpu_runtime/multiclass.rs` (NEW) — the multi-output block-leaf driver + objective classification + der assembly.
- `crates/cb-backend/src/gpu_runtime/multiclass_test.rs` (NEW) — the coupled + two-diagonal block self-oracle + gate assertions.
- `crates/cb-backend/src/gpu_runtime/mod.rs` — register `pub(crate) mod multiclass` + `#[cfg(test)] mod multiclass_test`.
- `crates/cb-backend/src/gpu_runtime/session.rs` — `MulticlassState` + `map_multiclass_coverage` + the `begin()` multi-output intercept branch + the `multiclass: Option<MulticlassState>` field.

## Decisions Made
- Batched device solve over ALL leaves at once (system-major `sum_der[leaf*K+d]`) so the read-back IS the row-major block — no transpose.
- Diagonal losses fill only the packed diagonal entries (off-diagonal 0); the device diagonal mode reads only those (Plan-06 uniform-launcher contract, == `solve_symmetric_newton(k==1)` per component).
- Covered multi-output configs decline to `Ok(None)`→CPU (grow_one has no `approx_dim>1` arm; the per-tree shared multi-dim grow seam is a forward dependency), exactly like the pairwise/ranking gates — never a fabricated scalar grow.

## Deviations from Plan
None - plan executed exactly as written. (MultiRMSE has no `Loss` variant yet, so it is classified for the arm but the third self-oracle fixture uses MultiClassOneVsAll per the plan's explicit "MultiRMSE (or MultiClassOneVsAll)" allowance — both hessian structures are covered.)

## Issues Encountered
- `cargo test -p cb-backend --lib` (full crate, default `cpu` backend) reports 60 pre-existing failures — ALL panics inside `cubecl-cpu-0.10.0/src/compiler/visitor/elem.rs` for OTHER device kernels (reduce-atomic, radix/segmented sort, sat score). These are the CubeCL CPU-backend JIT choking on device kernels validated in-env on rocm/cuda only (CLAUDE.md "GPU tests on rocm only"); NONE are `multiclass` and NONE touch code changed this plan. The 7 `multiclass` tests pass on cpu (the serial f64 block solve JITs cleanly; numeric asserts are record-only on cpu, hard on rocm/cuda).
- Root disk at ~100% caused integration-test link failures for `cb-train`; cleared `target/debug/incremental` (freed ~3G) and verified `cb-train --lib` + `cb-compute --lib` (188 passed) + `cb-backend --tests` compile clean. Per-crate verification per project memory ("disk pressure & full-suite verification").

## Verification
- `cargo test -p cb-backend --lib multiclass` → 7 passed, 0 failed.
- `grep -v '^#' crates/cb-backend/Cargo.toml | grep -c 'cb-train'` == 0 (no cb-train dep).
- No infinity literal in any executable body (the sole `-inf` occurrence is the doc stating the discipline).
- `cb-compute --lib` 188 passed; `cb-train --lib` + `cb-backend --tests` compile clean.
- Kaggle CUDA numeric sign-off (ε=1e-4 on real device) deferred to Plan 10 (the phase gate) — the cpu-backend run is record-only by design (WR-01). rocm/cuda in-env execution not run in this executor.

## Next Phase Readiness
- The multi-output block driver (`grow_multiclass_block`) + coverage decision (`MulticlassState`) are ready for the per-tree shared multi-dim grow seam wiring (the forward dependency: `Runtime::grow_tree_on_device` must carry the K-dimensional approx / block leaf), and for the Plan-10 Kaggle CUDA sign-off.
- GPUT-12 device coverage is COMPLETE at the block-emission level (5 multi-output losses via one shared tree + block leaves; coupled/diagonal dispatch correct; self-oracled ≤1e-4).

## Self-Check: PASSED
- Created files present: `multiclass.rs`, `multiclass_test.rs`, `13-07-SUMMARY.md`.
- Task commits present in git: `1ed441e`, `a10abf8`.
- `cb-backend --tests` compiles clean; no cb-train dep; multiclass self-oracle 7/7 green on cpu.
