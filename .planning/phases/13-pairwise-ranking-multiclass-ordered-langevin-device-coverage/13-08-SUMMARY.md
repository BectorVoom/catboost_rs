---
phase: 13-pairwise-ranking-multiclass-ordered-langevin-device-coverage
plan: 08
subsystem: gpu_runtime
tags: [gpu, ordered-boosting, residency, GPUT-13, self-oracle]
requires:
  - "cb-backend gpu_runtime::launch_apply_leaf_delta_into (resident approx add, GPUT-03)"
  - "cb-compute::gradient_leaf_delta (calc_average, the shared leaf average)"
  - "GpuTrainSession coverage-gate template (pairwise/ranking/multiclass Option<*State> precedent)"
provides:
  - "GpuTrainSession.ordered: Option<OrderedState> — the landed ordered-boosting coverage seam"
  - "gpu_runtime::ordered::ordered_approx_delta — inline body/tail approximant (body rows keep 0)"
  - "gpu_runtime::ordered::accumulate_ordered_trajectory — device-resident per-permutation trajectory"
affects:
  - "GpuTrainSession::begin coverage gate (ordered !plain branch classifies then declines to CPU)"
tech-stack:
  added: []
  patterns:
    - "Ok(None) family coverage gate (all-or-nothing per family, D-10-01)"
    - "resident device handle folded via apply_leaf_delta (identity leaf map + unit rate), one final read-back"
    - "inline CPU-ref transcription (no cb-train dep); f64 wgpu typed reject"
key-files:
  created:
    - crates/cb-backend/src/gpu_runtime/ordered.rs
    - crates/cb-backend/src/gpu_runtime/ordered_test.rs
  modified:
    - crates/cb-backend/src/gpu_runtime/session.rs
    - crates/cb-backend/src/gpu_runtime/mod.rs
decisions:
  - "Ordered grow declines to CPU (Ok(None)) — the per-tree ordered permutation-descriptor grow seam is a forward dependency, mirroring the pairwise/ranking/multiclass gate precedent; the driver + self-oracle land the residency mechanism this plan"
  - "The ordered per-object delta is computed HOST-side (inherently sequential permutation scan); only the resident trajectory FOLD runs on device via apply_leaf_delta — satisfying the no-per-iteration-n-length-readback contract (one final read-back)"
  - "OrderedState carries the covered der kernel (Gradient/RMSE simple approximant) as the coverage decision, like RankingState.objective"
metrics:
  duration_min: 12
  completed: 2026-07-04
  tasks: 2
  files_created: 2
  files_modified: 2
  commits: 2
status: complete
---

# Phase 13 Plan 08: Ordered Boosting Device Coverage (GPUT-13) Summary

Ordered boosting's per-permutation historical approx trajectory is now reproduced device-resident: the device driver folds each tree's anti-leakage body/tail approximant into a resident trajectory handle via `apply_leaf_delta` (one final read-back, no per-iteration n-length crossing), self-oracled bit-for-bit against the frozen CPU `ordered_approx_delta_simple` at ε=1e-4.

## What was built

- **`gpu_runtime/ordered.rs` (NEW, 226 LOC)** — the ordered device driver:
  - `OrderedTree` — one boosting iteration's frozen descriptor (leaf_of, der, weights, learn permutation, body/tail boundary, leaf count, scaled_l2).
  - `ordered_approx_delta` — ONE tree's ordered per-object approximant delta, transcribing `cb_train::boosting::ordered_approx_delta_simple` INLINE (no cb-train dep). Seeds the body prefix, walks the tail in permutation order (add-then-read the running per-leaf `gradient_leaf_delta` = `calc_average`), body rows keep delta 0.
  - `accumulate_ordered_trajectory` — the resident driver: init a zero trajectory handle ONCE, fold each tree's ordered delta into it ON DEVICE via `launch_apply_leaf_delta_into` (identity leaf map + unit rate → `trajectory[i] += delta[i]`), reading back exactly ONCE at the end. f64 wgpu typed reject.
- **`gpu_runtime/session.rs` (MODIFY)** — `OrderedState` struct + `map_ordered_coverage` gate + `ordered: Option<OrderedState>` field. `begin(...)` intercepts the ordered (`!boosting_type_is_plain`) branch, classifies coverage, then declines to CPU (`Ok(None)`) — the Plain path stays byte-unchanged.
- **`gpu_runtime/ordered_test.rs` (NEW, 343 LOC)** — 5 tests: frozen-fixture device-vs-CPU trajectory parity (ε=1e-4, device-gated / cpu record-only WR-01); body-rows-keep-0 anti-leakage; residency (N-iteration resident accumulation equals the sum of single-tree trajectories, proving cross-iteration handle persistence); `begin()` declines ordered to CPU (covered + uncovered) while the Plain path still opens a session; a `gradient_leaf_delta` numeric anchor.
- **`gpu_runtime/mod.rs` (MODIFY)** — registered `pub(crate) mod ordered` + `#[cfg(test)] mod ordered_test`.

## Verification

- `cargo test -p cb-backend --lib gpu_runtime::ordered_test` → 5/5 green.
- `cargo test -p cb-train ordered` → green (exit 0).
- `cargo check --tests -p cb-backend` → clean (my files 0 warnings; 4 pre-existing `nonsym_grow` mut warnings out of scope).
- Acceptance greps: cb-train dep count in `cb-backend/Cargo.toml` == 0; `apply_leaf_delta` present in ordered.rs; no infinity literal in code (only the doc comment references `-inf`).

## Deviations from Plan

None — plan executed as written. The ordered grow declines to CPU exactly as the pairwise/ranking/multiclass gates in this phase do (the per-tree ordered permutation-descriptor grow seam is a documented forward dependency); the driver + residency mechanism + self-oracle are the landed deliverables per the plan's success criteria.

## Deferred / out of scope

- Pre-existing cpu-backend failures `kernels::score_split::scan::cumulative_matches_host_ordered_reference` and `kernels::grow_loop::partition::update_matches_ordered_reference` (they only match the `ordered` name filter). Confirmed failing at `13d54ba~1` (before any Plan-08 change) — a known cpu-backend device-vs-host n=1 false-compare (WR-01 family), NOT caused by this plan. Not fixed (scope boundary).
- Kaggle CUDA ε=1e-4 sign-off for the ordered trajectory is human-gated in Plan 10 (device numeric asserts fire only on rocm/cuda; the in-env default `cpu` backend records-only).

## Threat surface

No new trust boundaries. T-13-15 (ordered per-iteration launch geometry DoS) mitigated: checked casts, bounded launch via the existing `apply_leaf_delta` grid, resident trajectory (no per-iteration realloc — one handle reused across iterations). T-13-16 (coverage gate) held: `Ok(None)` never fabricates; the frozen CPU trajectory is the reference. No external package added (T-13-SC).

## Self-Check: PASSED

- FOUND: crates/cb-backend/src/gpu_runtime/ordered.rs
- FOUND: crates/cb-backend/src/gpu_runtime/ordered_test.rs
- FOUND commit 13d54ba (feat 13-08 driver)
- FOUND commit f897ae7 (test 13-08 self-oracle)
