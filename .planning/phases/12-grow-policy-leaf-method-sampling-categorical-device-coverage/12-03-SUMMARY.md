---
phase: 12-grow-policy-leaf-method-sampling-categorical-device-coverage
plan: 03
subsystem: gpu-training
status: complete
tags: [cubecl, rocm, device-grow, non-symmetric, depthwise, lossguide, leaf-wise, coverage-gate, GPUT-18]

# Dependency graph
requires:
  - phase: 12-01
    provides: "DeviceGrownTree step_nodes/node_id_to_leaf_id carrier + DeviceTrainConfig config surface + all-or-nothing coverage gate"
  - phase: 12-02
    provides: "cb_train NonSymmetricTree + cb_model non-sym apply (leaf_index_nonsym) reference"
  - phase: 11-depth6-grow
    provides: "launch_find_optimal_split_pointwise device scorer + pointwise_hist2 whole-subset path"
provides:
  - "Device Depthwise + Lossguide non-symmetric grow (grow_nonsym_tree) — structure integer-exact vs leaf_wise_grower, leaf values bit-exact (0.0 divergence) on gfx1100"
  - "boosting.rs device-grow fold arm is now shape-aware: step_nodes non-empty → NonSymmetricTree into non_symmetric_trees; oblivious arm byte-unchanged"
  - "begin_device_training trait method threads DeviceTrainConfig (Open Q2 promotion); Depthwise/Lossguide flip the device non-sym arm on end-to-end"
affects: [12-04-region-device, 12-05-exact, 12-06-bootstrap, 12-07-mvs, 12-08-ctr]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Host-light non-sym grow: per candidate leaf-node the device scorer (launch_find_optimal_split_pointwise) argmins over the node's doc subset; host does the O(1) gain + node bookkeeping (transcribed leaf_wise_grower, no cb-train dep)"
    - "Config-surface promotion: begin_device_training takes &DeviceTrainConfig, threaded boosting → gpu_backend → session (the Plan-01 deferred trait-method promotion, now that this wave owns boosting.rs)"
    - "Non-sym device session is host-DRIVEN: der1 re-derived on host from the caller's approx (RMSE target-approx / Logloss target-sigmoid), matching CPU compute_gradients bit-for-bit in the unit-weight/bias-0 covered regime"

key-files:
  created:
    - crates/cb-backend/src/kernels/nonsym_grow.rs
    - crates/cb-backend/src/kernels/nonsym_grow_test.rs
    - crates/cb-train/src/boosting_device_fold_test.rs
    - crates/cb-train/tests/device_nonsym_fit_test.rs
  modified:
    - crates/cb-backend/src/kernels.rs
    - crates/cb-backend/src/gpu_runtime/session.rs
    - crates/cb-backend/src/gpu_backend.rs
    - crates/cb-backend/src/gpu_backend_test.rs
    - crates/cb-backend/src/gpu_runtime/session_residency.rs
    - crates/cb-backend/src/gpu_runtime/session_depth_gt1_test.rs
    - crates/cb-compute/src/runtime.rs
    - crates/cb-train/src/boosting.rs
    - crates/cb-train/tests/device_seam_test.rs

key-decisions:
  - "The non-sym device path is HOST-DRIVEN (not resident): per node it slices the doc subset and calls launch_find_optimal_split_pointwise (pointwise_hist2, Atomic<f64>) which runs on cpu too — UNLIKE the resident oblivious grow (Atomic<u64>, rocm-only). Selection on device, O(1) gain + bookkeeping on host, so structure is integer-exact and Lossguide queue order is bit-identical."
  - "begin_device_training trait method PROMOTED to take &DeviceTrainConfig this wave (Plan 01 deferred it because Plan 01 could not edit boosting.rs). This wave owns boosting.rs, so the config now threads end-to-end; gpu_backend passes it through instead of constructing default()."
  - "grow_one signature gained `approx` (the non-sym arm re-derives der1 from it; the oblivious resident arm ignores it). The two resident-session test call sites pass a zero approx (unused on the oblivious arm)."
  - "GPUT-18 stays Pending: Plan 03 closes Depthwise/Lossguide device coverage; Region device (Plan 04) is the last non-sym family, so GPUT-18 is not marked complete."

patterns-established:
  - "Device non-sym grow oracles are rocm/cuda-gated (cubecl-cpu panics JITing the per-node scorer over subset shapes — elem.rs:38); the WR-01 anti-false-pass skip extends to nonsym_grow_test + device_nonsym_fit_test."

requirements-completed: []  # GPUT-18 is phase-spanning; Region device (Plan 04) remains — left Pending.

# Metrics
duration: ~110min
completed: 2026-07-04
---

# Phase 12 Plan 03: Device Depthwise/Lossguide Non-Symmetric Grow Summary

**Flipped the first two non-symmetric grow families (Depthwise level-order + Lossguide best-gain-priority) from `Ok(None)` to a real device grow: `grow_nonsym_tree` reuses the Phase-11 device split scorer per candidate leaf-node and emits a non-symmetric node graph as plain host structs, which the now-shape-aware `boosting.rs` device-fold arm materializes into `non_symmetric_trees` (NOT a degenerate `ObliviousTree`). Verified on real gfx1100: structure integer-exact and leaf values bit-exact (0.0 divergence) vs the CPU `leaf_wise_grower`, end-to-end through `cb_train::train()`.**

## Accomplishments

- **Task 1 — device non-sym grow driver (`nonsym_grow.rs`, commit `0b990fe`):** a host-light greedy grower that changes ONLY the leaf-selection order (Depthwise level-order / Lossguide priority queue) over the existing `launch_find_optimal_split_pointwise` scorer, transcribing `leaf_wise_grower`'s node bookkeeping inline (`step_nodes`/`node_id_to_leaf_id`, `u32::MAX` interior sentinel, checked `u16::try_from` diffs) — no `cb-train` dep, no `#[cube]` of its own, no `-inf` literal.
- **Task 2 — shape-aware fold (`boosting.rs`, commit `9012d5c`):** the `device_active` branch dispatches on `dev_tree.step_nodes` — empty → `ObliviousTree` (byte-identical to Plan 01), non-empty → `NonSymmetricTree` into `non_symmetric_trees` with per-object leaf assignment via the transcribed `leaf_index_nonsym` pointer-walk (`device_leaf_of_nonsym`).
- **Task 3 — gate + config threading + oracles (commits `feb7f8e`, `e6e755e`, `6216dca`):** `map_grow_policy` gate arm (Depthwise/Lossguide → device, Region → `Ok(None)`), a host-driven non-sym session path, and the `begin_device_training` config-surface promotion threaded boosting → gpu_backend → session. Self-oracle (`nonsym_grow_test.rs`) + end-to-end (`device_nonsym_fit_test.rs`) both bit-exact on gfx1100.

## Verification (rocm gfx1100 in-env + cpu)

- `cargo test -p cb-backend --no-default-features --features rocm nonsym_grow` — **4/4**: Depthwise + Lossguide × L2 + Cosine, structure INTEGER-exact (`assert_eq!` on `step_nodes`/`node_id_to_leaf_id`/per-node splits/`leaf_of`), leaf-value **max abs_div = 0.000e0** (bit-exact, bar ε=1e-4).
- `cargo test -p cb-train --no-default-features --features rocm --test device_nonsym_fit_test` — **2/2**: device Depthwise/Lossguide `train()` fit folds into `non_symmetric_trees` (oblivious EMPTY) and reproduces the CPU `leaf_wise_grower` model with **max |Δpred| = 0.000e0** (closes the degenerate-`ObliviousTree` checker BLOCKER).
- No-regression: rocm `session` 4/4, `grow_loop` 14/14, `score_split` 12/12; cpu `cb-train --lib` 237/237, `device_seam_test` 6/6, `boosting_device_fold` 2/2, `region_e2e_test` 2/2, `non_symmetric_grower_oracle_test` 1/1.
- Landmine checks: `grep "use cb_train" nonsym_grow.rs` — none; `grep "use cb_backend" boosting.rs` — none new; no `-inf` literal in the changeset.

## Deviations from Plan

### 1. [Rule 3 — Blocking] `begin_device_training` trait method promoted to take `&DeviceTrainConfig`
- **Found during:** Task 3 (wiring Depthwise/Lossguide end-to-end).
- **Issue:** Plan 01 deferred threading `DeviceTrainConfig` through the `Runtime` trait method (its sole caller is `boosting.rs`, which Plan 01 could not edit). To reach the session gate with the real grow policy, this wave (which owns `boosting.rs`) had to promote the method signature.
- **Fix:** Added `config: &DeviceTrainConfig` to `Runtime::begin_device_training` (cb-compute), the `GpuBackend` impl (`gpu_backend.rs`), the `DeviceMock` (`device_seam_test.rs`), and the boosting call site (which builds the config from `params`). `gpu_backend_test.rs` call sites updated. Files `gpu_backend.rs`/`gpu_backend_test.rs`/`device_seam_test.rs` were not in the plan's `files_modified` but are mechanically required by the signature change.
- **Commit:** `feb7f8e`.

### 2. [Rule 3 — Blocking] end-to-end oracle lives in `tests/`, not `src/`
- **Issue:** The plan lists `crates/cb-train/src/device_nonsym_fit_test.rs`, but a src-mounted test cannot instantiate `cb_model` (the `cb_train` dev-dep diamond → E0308, documented in 12-02 SUMMARY). The end-to-end predict-comparison needs `cb_model`.
- **Fix:** Placed it at `crates/cb-train/tests/device_nonsym_fit_test.rs` (integration), mirroring `region_e2e_test.rs`. It uses a local `CpuRefRuntime` (declines device, computes RMSE gradients) as the reference because `CpuBackend` is `#[cfg(feature="cpu")]` and unavailable under the rocm feature.
- **Commit:** `feb7f8e`.

### 3. [Rule 1 — Bug] gate/coverage tests updated for the now-covered policies
- **Issue:** `session_depth_gt1_gate_declines_uncovered` (Plan 01) asserted a Depthwise config DECLINES; `gpu_backend_device_grow_uncovered_falls_back` used depth=2 (no longer uncovered after Plan 01) as its "uncovered" case.
- **Fix:** Assert Depthwise/Lossguide now OPEN a session and Region still declines; switched the gpu_backend "uncovered" case to `fold_count=2`.
- **Commit:** `feb7f8e` (+ `6216dca` for the session test).

### 4. [Rule 1 — Bug] end-to-end fixture uses `lr=0.3` (not `1.0`)
- **Issue:** At `lr=1.0` the separable step fixture is fit EXACTLY in one tree → the 2nd iteration has all-zero der1 → root gain `< 1e-9` → both growers legitimately `Degenerate`.
- **Fix:** `lr=0.3` leaves a residual so both trees grow; the oracle asserts device-vs-CPU bit-exact + sign tracks target (dropped the too-strict exact-target check).
- **Commit:** `e6e755e`.

## Known Stubs

None. The Region device arm is explicitly deferred to Plan 04 (Region grow_policy still returns `Ok(None)` at the gate — documented, wave-owned).

## Self-Check: PASSED

- Files exist: `nonsym_grow.rs`, `nonsym_grow_test.rs`, `boosting_device_fold_test.rs`, `device_nonsym_fit_test.rs` — all FOUND.
- Commits exist: `0b990fe`, `9012d5c`, `feb7f8e`, `e6e755e`, `6216dca` — all FOUND.
- rocm gfx1100 in-env: nonsym_grow 4/4 (0.0 divergence), device_nonsym_fit 2/2 (0.0 divergence) ✓
- No `use cb_train` in nonsym_grow.rs; no new `use cb_backend` in boosting.rs; no `-inf` literal ✓
