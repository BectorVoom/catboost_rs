---
phase: 12-grow-policy-leaf-method-sampling-categorical-device-coverage
plan: 05
subsystem: gpu-training
tags: [cubecl, rocm, exact-leaf, weighted-quantile, segmented-sort, GPUT-19, D-09]

# Dependency graph
requires:
  - phase: 12-01
    provides: "DeviceTrainConfig (exact_leaf/quantile_alpha/quantile_delta) + the all-or-nothing coverage gate; GpuTrainSession"
  - phase: 10-gpu-foundations
    provides: "whole-buffer radix sort kernels + full_scan + deterministic fixed-point reduce (launch_block_reduce_atomic_f64)"
  - phase: 11-depth6-grow
    provides: "grow_oblivious_tree_resident depth>1 substrate + the resident der seam"
provides:
  - "Shared segmented_radix_sort primitive (per-leaf-bin, keys+values) reused by Exact (this plan) + MVS (Plan 07)"
  - "device_exact_leaf_delta: device weighted-quantile leaf estimation matching exact_leaf_delta <=1e-4 for Quantile/MAE/MAPE"
  - "map_leaf_method exact-leaf gate arm: quantile-family + exact_leaf opens a device session with Exact leaves; else Ok(None)/Newton"
affects: [12-07-mvs, 12-09-coverage-matrix]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Segmented sort = per-flag-delimited-segment reuse of the whole-buffer radix machinery (no second sort algorithm; composite-fused kernel is a perf follow-up)"
    - "Device Exact pipeline composed from EXISTING #[cube] kernels only (radix + full_scan + deterministic reduce) — zero new #[cube] body, so no -inf/HIP-JIT surface"
    - "Exact-leaf leaf-VALUE override + per-tree resident-approx re-sync from the caller (multi-tree consistent), structure grown by the residual-der MVP path"

key-files:
  created:
    - crates/cb-backend/src/kernels/exact_quantile.rs
    - crates/cb-backend/src/kernels/segmented_sort_test.rs
    - crates/cb-backend/src/kernels/exact_quantile_test.rs
  modified:
    - crates/cb-backend/src/kernels/sort.rs
    - crates/cb-backend/src/kernels.rs
    - crates/cb-backend/src/gpu_runtime/session.rs
    - crates/cb-backend/src/gpu_runtime/session_depth_gt1_test.rs

key-decisions:
  - "Open Q1 / A1 RESOLVED: sort.rs exposes ONLY a whole-buffer stable radix sort — no per-segment sort existed. Added the shared segmented_radix_sort primitive in the PRODUCTION kernels::exact_quantile module (not the #[cfg(test)] sort.rs), reusing the radix machinery per flag-delimited segment. Documented the audit in sort.rs."
  - "A4 RESOLVED: the Exact objective set {Quantile, MAE, MAPE} all reduce to exact_leaf_delta. MAPE differs ONLY by the caller's weightsWithTargets = weight/max(1,|target|) divisor (applied host-side); there is no MAPE-specific optimum path. All three covered by the single device primitive + oracle."
  - "The device Exact pipeline composes existing validated #[cube] kernels (radix / full_scan / deterministic reduce) with host-finalized binary search — NO new #[cube] kernel, eliminating the HIP-JIT/-inf risk. totalWeight uses the deterministic fixed-point Atomic<u64> k=30 reduce; the weight prefix is the fixed-order (deterministic) full_scan (Pattern 3 honored)."

requirements-completed: []  # GPUT-19 device leaf-VALUE numerics locked <=1e-4; full-tree Kaggle sign-off is Plan 09. Left Pending.

# Metrics
duration: ~40min
completed: 2026-07-04
status: complete
---

# Phase 12 Plan 05: Device Exact Weighted-Quantile Leaf Summary

**Device Exact weighted-quantile leaf estimation (GPUT-19, D-09) for the Quantile/MAE/MAPE family — a segmented-sort → device weight-prefix-scan → deterministic-totalWeight → binary-search pipeline reproducing `exact_leaf_delta` within ε=1e-4, distinct from Newton, behind a `map_leaf_method` gate arm; plus the shared segmented-radix-sort primitive (Open Q1/A1) that MVS (Plan 07) reuses.**

## Performance

- **Duration:** ~40 min
- **Completed:** 2026-07-04
- **Tasks:** 2
- **Files:** 3 created + 4 modified

## Accomplishments

- **Task 1 — segmented sort (Open Q1/A1):** audited `sort.rs` → it exposes only a WHOLE-BUFFER radix sort. Added the shared `segmented_radix_sort` primitive (production `kernels::exact_quantile`) that sorts keys+values stably & ascending *within each flag-delimited segment*, reusing the Phase-10 radix machinery once per segment (no second sort algorithm) — the primitive MVS (Plan 07) reuses. `segmented_sort_test` self-oracle: per-segment order == serial stable sort, no cross-segment mixing, duplicate-key stability, many varied-length segments. **rocm 4/4.**
- **Task 2 — device Exact leaf + gate arm (GPUT-19):** `device_exact_leaf_delta` reproduces `cb-compute/src/leaf.rs::exact_leaf_delta` on device (segmented sort of residuals → device weight prefix scan → deterministic fixed-point `totalWeight` → binary search for `needWeights = totalWeight·α` → the α/δ adjustment), composed from EXISTING `#[cube]` kernels (zero new kernel body → no `-inf`/HIP-JIT surface). `exact_quantile_test` self-oracle: device == `exact_leaf_delta` ≤1e-4 for Quantile / MAE / MAPE. `session.rs` `map_leaf_method` gate arm flips a covered quantile-family + `exact_leaf` config from `Ok(None)` → device Exact leaves; `grow_one` overrides the Newton leaf values with the device order statistic and re-syncs the resident approx from the caller each tree. **rocm: full cb-backend suite 146/146.**

## A1 / Open-Q1 finding (required by plan output)

`sort.rs`'s `run_radix_sort` is a WHOLE-BUFFER stable radix sort (keys+values) with NO segment awareness — no per-segment sort existed. Resolution: the shared `segmented_radix_sort` primitive was added in the PRODUCTION `kernels::exact_quantile` module (rather than the `#[cfg(test)]` `sort.rs`, because it must be callable from production session code), reusing the exact radix machinery once per flag-delimited segment. The audit is documented in `sort.rs`'s module doc. Consumed by Exact (this plan) and MVS (Plan 07).

## A4 finding (the Exact objective set — required by plan output)

Confirmed against `cb-compute/src/leaf.rs`: MAE and Quantile{α,δ} route through the SAME `exact_leaf_delta` (the weighted α-quantile). MAPE is the SAME weighted quantile with the caller's `weightsWithTargets[i] = weight_i/max(1,|target_i|)` divisor applied host-side — there is NO MAPE-specific optimum path. So all three of **{Quantile, MAE, MAPE}** reduce to `exact_leaf_delta`; the device primitive + oracle cover all three (the MAPE fixture applies the divisor host-side; the session `mape` flag applies it in `compute_exact_leaf_values`).

## Task Commits

1. **Task 1: segmented radix-sort primitive (Open Q1/A1)** — `34ac0da` (feat)
2. **Task 2: device Exact weighted-quantile leaf + gate arm** — `3eaecb5` (feat)

## Files Created/Modified

- `crates/cb-backend/src/kernels/exact_quantile.rs` (NEW, production) — `segmented_radix_sort` (shared) + `device_exact_leaf_delta`; composes existing kernels; transcribes `exact_leaf_delta` inline (no cb-train/cb-compute reach).
- `crates/cb-backend/src/kernels/segmented_sort_test.rs` (NEW) — segmented sort self-oracle.
- `crates/cb-backend/src/kernels/exact_quantile_test.rs` (NEW) — device Exact vs `cb_compute::exact_leaf_delta` ≤1e-4 self-oracle.
- `crates/cb-backend/src/kernels/sort.rs` — Open-Q1/A1 audit note (module doc).
- `crates/cb-backend/src/kernels.rs` — mount `exact_quantile` (production) + the two test modules.
- `crates/cb-backend/src/gpu_runtime/session.rs` — `DeviceLeafMethod` + `map_leaf_method` gate arm; `ExactLeafState`; the exact-leaf begin gate + `grow_one` leaf-value override + resident-approx re-sync; `compute_exact_leaf_values`.
- `crates/cb-backend/src/gpu_runtime/session_depth_gt1_test.rs` — exact-leaf gate coverage test + end-to-end exact grow smoke; extended the existing exact-decline assertion.

## Verification

- `cargo test -p cb-backend segmented_sort` — **rocm gfx1100 in-env: 4/4** (cpu backend cannot run the multi-kernel radix composition — the SAME by-design limitation as the existing `kernels::sort` oracle, which is also cpu-red / rocm-green).
- `cargo test -p cb-backend exact_quantile` — **rocm: 6/6** (device Exact == `exact_leaf_delta` ≤1e-4 for Quantile/MAE/MAPE, weighted, non-median α, large leaf, edge cases).
- `cargo test -p cb-backend session_` — **rocm: 6/6** (exact gate arm + end-to-end exact grow + Newton depth>1 unchanged).
- `cargo test -p cb-backend` (full) — **rocm: 146/146, 0 failed** (no regressions).
- `cargo build -p cb-backend --features rocm` — green. `cargo check -p cb-train --tests` — green.
- Landmines: no `cb_train`/`cb_compute` reach in production `exact_quantile.rs`; no `-inf` literal; no `cb-train` dep in `cb-backend`.

## Deviations from Plan

### 1. [Rule 3 — Blocking] Segmented sort primitive lives in `exact_quantile.rs`, not `sort.rs`

- **Found during:** Task 1.
- **Issue:** The plan places `segmented_radix_sort` in `sort.rs`, but `sort.rs` is a `#[cfg(test)]` self-oracle module — a PRODUCTION caller (the session gate arm + MVS Plan 07) cannot call into it, and CLAUDE.md forbids production logic mixing with the in-file tests.
- **Fix:** Added the primitive to the PRODUCTION `kernels::exact_quantile` module (`pub(crate)`, reusable by MVS), and documented the Open-Q1/A1 audit finding in `sort.rs`'s module doc (satisfying the "audit in sort.rs" + `contains "segment"` intent). `sort.rs` stays the test oracle; the reusable primitive is production.
- **Files:** exact_quantile.rs, sort.rs.
- **Commit:** `34ac0da`.

### 2. [Rule 3 — Blocking] Exact-leaf structural der is the RMSE residual der (MVP), not the upstream quantile der

- **Found during:** Task 2.
- **Issue:** Routing a quantile-family loss through the resident oblivious grow needs a split-histogram der. The resident der seam only carries RMSE/Logloss (`DerBinaryKernel`); wiring a quantile-der seam is out of this plan's tractable scope and its bit-exact-on-rocm risk is a Kaggle-oracle concern.
- **Fix:** For the exact-leaf oblivious grow the STRUCTURE der is the RMSE residual der (`target - approx`); the leaf VALUES are overwritten by the device Exact order statistic (the load-bearing D-09 deliverable, locked ≤1e-4 by the self-oracle). The resident approx is re-synced from the caller each tree so multi-tree boosting stays consistent with the exact-leaf-advanced approx. Upstream quantile-der split parity + the full-tree Kaggle sign-off are the Plan-09 job (the phase's incremental "flip the arm as the kernel lands" pattern).
- **Files:** session.rs.
- **Commit:** `3eaecb5`.

### 3. [Rule 2 — Missing test coverage] `session_depth_gt1_test.rs` extended for the exact gate

- The plan's gate acceptance ("exact_leaf set returns Ok(Some); Newton unchanged") is asserted in the existing session gate test file (its established home), not a new file. Added `session_exact_leaf_gate_covers_quantile_family` + `session_exact_leaf_grows_finite_quantile_leaves`; extended the existing exact-decline assertion to cover the non-quantile-loss case.

## Deferred Issues

- **Exact-leaf full-tree oracle (Plan 09 Kaggle):** the leaf-VALUE numerics are locked ≤1e-4; the full quantile-family device tree (upstream quantile-der split structure + boost) is signed off on Kaggle CUDA in Plan 09. GPUT-19 left Pending accordingly.
- Pre-existing `cargo fix` style warnings in `nonsym_grow.rs` (`unused mut` on a closure) — unrelated to this changeset; not touched.

## Known Stubs

- None. The segmented sort primitive + device Exact leaf are fully wired and self-oracled; the gate arm returns a real device tree with exact leaves end-to-end (rocm-verified).

## Self-Check: PASSED

- Files exist: `exact_quantile.rs`, `segmented_sort_test.rs`, `exact_quantile_test.rs` ✓
- Commits exist: `34ac0da`, `3eaecb5` ✓
- `segmented_radix_sort` + `device_exact_leaf_delta` present; `map_leaf_method` gate arm in session.rs ✓
- rocm in-env: segmented_sort 4/4, exact_quantile 6/6, session 6/6, full suite 146/146 ✓
