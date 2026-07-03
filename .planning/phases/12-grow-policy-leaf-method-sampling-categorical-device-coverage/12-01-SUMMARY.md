---
phase: 12-grow-policy-leaf-method-sampling-categorical-device-coverage
plan: 01
subsystem: gpu-training
tags: [cubecl, rocm, device-grow, oblivious-tree, non-symmetric, seam, coverage-gate, GPUT-18]

# Dependency graph
requires:
  - phase: 11-depth6-grow
    provides: "grow_oblivious_tree_resident depth>1 partition-aware substrate (Atomic<u64> fixed-point histogram), the resident GpuTrainSession"
  - phase: 10-gpu-foundations
    provides: "DeviceGrownTree Runtime seam, begin/grow_one/end device-training trait methods"
provides:
  - "Depth>1 Plain/fold1/RMSE/covered-score configs now REACH the device grow via GpuTrainSession::begin (A3 gap closed)"
  - "DeviceGrownTree.step_nodes + node_id_to_leaf_id non-symmetric node-graph carrier (plain host structs) for Plan-03/04 fold arms"
  - "DeviceTrainConfig single plain-host config surface (grow-policy/sampling/exact/CTR) with an all-or-nothing coverage gate"
affects: [12-02-region, 12-03-nonsym, 12-04-region-device, 12-05-exact, 12-06-bootstrap, 12-07-mvs, 12-08-ctr]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Config-surface widening via one plain-host DeviceTrainConfig struct instead of per-wave arg-list growth (Open Q2)"
    - "All-or-nothing coverage gate: is_covered_regime() → Ok(None) → byte-unchanged CPU grower for any not-yet-covered family flag (D-10-01)"
    - "Non-symmetric node graph crosses the Runtime seam as plain host Vec<(u16,u16)>/Vec<u32>, never a cubecl type (T-10-04)"

key-files:
  created:
    - crates/cb-backend/src/gpu_runtime/session_depth_gt1_test.rs
    - .planning/phases/12-grow-policy-leaf-method-sampling-categorical-device-coverage/deferred-items.md
  modified:
    - crates/cb-compute/src/runtime.rs
    - crates/cb-compute/src/lib.rs
    - crates/cb-backend/src/gpu_runtime/session.rs
    - crates/cb-backend/src/gpu_backend.rs
    - crates/cb-backend/src/gpu_runtime/session_residency.rs
    - crates/cb-backend/src/gpu_runtime/mod.rs

key-decisions:
  - "A3 audit finding: depth>1 was ALREADY device-covered by the Phase-11 substrate (grow_oblivious_tree_resident loops 0..depth); ONLY the session begin gate force-declined it. No grow-path wiring was added — the fix was a one-condition gate relaxation."
  - "DeviceTrainConfig is threaded into GpuTrainSession::begin (where the gate lives), NOT into the Runtime::begin_device_training trait method — that method's sole caller is cb_train::boosting, which Plan 01's success criteria forbid editing (same-wave conflict with Plan 02). The backend impl constructs DeviceTrainConfig::default() internally; trait-method promotion is deferred to the wave that owns boosting.rs."
  - "GPUT-18 is NOT marked complete: Plan 01 delivers only the foundation (seam carrier + gate + config surface). The actual on-device Depthwise/Lossguide/Region policies land in Plans 02/03/04; GPUT-18 stays Pending until those close."

patterns-established:
  - "Device-only grow tests SKIP (not panic) on cpu/wgpu via `if !cfg!(any(feature=rocm,feature=cuda))` — the WR-01 anti-false-pass convention, now applied to the residency oracle too."

requirements-completed: []  # GPUT-18 is phase-spanning; Plan 01 is foundational only — left Pending.

# Metrics
duration: ~22min
completed: 2026-07-03
status: complete
---

# Phase 12 Plan 01: Device-Coverage Foundation Summary

**Closed the A3 depth>1 gap with a one-condition session-gate relaxation (the Phase-11 substrate already grew depth>1 on device), and laid the two seam carriers every later Phase-12 wave stands on: the non-symmetric node-graph fields on `DeviceGrownTree` and the single plain-host `DeviceTrainConfig` config surface with an all-or-nothing coverage gate.**

## Performance

- **Duration:** ~22 min
- **Completed:** 2026-07-03T22:44Z
- **Tasks:** 3
- **Files modified:** 6 (+2 created)

## Accomplishments

- **A3 gap closed (Task 1):** the `begin` gate's `depth != 1` force-decline is gone — a depth>1 Plain/fold1/RMSE/covered-score config now flows through `grow_oblivious_tree_resident` (the already-shipped Phase-11 partition-aware substrate) instead of falling back to CPU. A depth-6 session self-oracle proves the session grow matches a direct `grow_oblivious_tree` call bit-for-bit on integer splits + `leaf_of`, leaf values within ε=1e-4.
- **Non-symmetric seam carrier (Task 2):** `DeviceGrownTree` gained `step_nodes: Vec<(u16,u16)>` + `node_id_to_leaf_id: Vec<u32>` (mirroring `cb_train::tree::GrownTree` verbatim), EMPTY on the oblivious path (byte-unchanged, D-04) — the fields Plans 03/04 will fill to fold `NonSymmetricTree`/`RegionTree` in `boosting.rs`.
- **Config surface (Task 3):** `DeviceTrainConfig` (+ `DeviceGrowPolicy`/`DeviceBootstrapType`/`DeviceCtrConfig`) is a single plain-host struct carrying grow-policy/sampling/exact/CTR knobs; `is_covered_regime()` drives an all-or-nothing gate so any non-default family flag returns `Ok(None)` until its wave flips the arm on.

## A3 Audit Finding (required by plan output)

**Was depth>1 already reachable?** The *device grow* was — the *session gate* was not.
`grow_oblivious_tree_resident` (mod.rs:2717) already loops `for level in 0..depth`,
keying each level's fill on the resident `leaf_of` via `launch_partition_hist2_resident_into`
(the Phase-11 GPUT-05 partition-aware path). The ONLY thing declining depth>1 was
`session.rs` `begin` at the former `if depth != 1 || ...` gate. **No new dispatch wiring
was added** — Task 1 was a pure gate relaxation (`depth != 1` → `depth == 0`) plus a
self-oracle test. This matches the plan's "if the audit shows the resident path already
handles depth>1, scope to the gate only" branch.

## Task Commits

1. **Task 1: Audit A3 + relax the session gate** - `49541ac` (feat)
2. **Task 2: Extend DeviceGrownTree with the non-symmetric node graph** - `c51122f` (feat)
3. **Task 3: DeviceTrainConfig host struct + config gate** - `8973b47` (feat)

## Files Created/Modified

- `crates/cb-compute/src/runtime.rs` - `DeviceGrownTree` non-sym fields; `DeviceTrainConfig` + `DeviceGrowPolicy`/`DeviceBootstrapType`/`DeviceCtrConfig` + `is_covered_regime()`. All plain host types (no cubecl).
- `crates/cb-compute/src/lib.rs` - export the four new config/policy types.
- `crates/cb-backend/src/gpu_runtime/session.rs` - gate relaxation (depth>1 covered); `begin` takes `&DeviceTrainConfig`, declines on non-default family flags, stores it.
- `crates/cb-backend/src/gpu_backend.rs` - constructs `DeviceTrainConfig::default()` and threads it into `GpuTrainSession::begin` (Runtime trait method + boosting.rs untouched).
- `crates/cb-backend/src/gpu_runtime/session_residency.rs` - inverted the stale depth>1-declines assertion; added the WR-01 cpu/wgpu skip guard; config call-site threading.
- `crates/cb-backend/src/gpu_runtime/session_depth_gt1_test.rs` - NEW self-oracle: depth-6 session grow == direct grow; gate declines uncovered incl. non-default config.
- `crates/cb-backend/src/gpu_runtime/mod.rs` - mount the new test module.

## Verification

- `cargo test -p cb-backend session_` — **cpu:** 4/4 (grow arm skips per WR-01, gate arm runs); **rocm gfx1100 in-env:** 4/4 (real depth-6 device grow matches direct grow within ε=1e-4).
- `cargo build -p cb-compute -p cb-backend -p cb-train` — green.
- Landmine check: `grep -rn "use cb_train" crates/cb-backend/src` adds nothing (the one hit is prose in a comment).

## Deviations from Plan

### 1. [Rule 3 — Blocking] DeviceTrainConfig threaded into `GpuTrainSession::begin`, not the `Runtime` trait method

- **Found during:** Task 3.
- **Issue:** The plan's Task 3 says "thread config into `begin_device_training`", but the plan's NOTE + success criteria HARD-require Plan 01 to NOT edit `boosting.rs` (the sole caller of `Runtime::begin_device_training`, owned by same-wave Plan 02). Adding a param to the trait method would force a `boosting.rs` edit.
- **Fix:** Threaded `&DeviceTrainConfig` into `GpuTrainSession::begin` (where the coverage gate actually lives) and had the backend's `begin_device_training` impl construct `DeviceTrainConfig::default()` internally. The trait method signature and `boosting.rs` are untouched; trait-method promotion is deferred to the wave that owns `boosting.rs`. All Task-3 acceptance criteria (non-default config → `Ok(None)`, tested directly via `GpuTrainSession::begin`) are satisfied.
- **Files modified:** session.rs, gpu_backend.rs.
- **Commit:** `8973b47`.

### 2. [Rule 1 — Bug] session_residency oracle panicked on cpu instead of skipping

- **Found during:** Task 3 verification (`cargo test -p cb-backend gpu_runtime`).
- **Issue:** `session_residency_matches_cpu_multi_tree_boosting` (a depth-1 device-grow oracle) lacked the cpu/wgpu skip guard its sibling depth6 grow tests carry, so on the default `cpu` feature it PANICKED ("partition-aware histogram fill requires Atomic<u64>") rather than skipping. Confirmed PRE-EXISTING (the original pre-Plan-12 source fails identically on cpu) — but it blocked Plan 01's own verify command and lives in a file Plan 01 already edits.
- **Fix:** Added the standard WR-01 `if !cfg!(any(feature=rocm,feature=cuda)) { skip }` guard, matching the convention.
- **Files modified:** session_residency.rs.
- **Commit:** `8973b47`.

## Deferred Issues

- Pre-existing clippy `indexing_slicing` denials in `crates/cb-backend/src/cpu_runtime.rs:696` and `:1025` (surfaced only by `cargo clippy`, not build/test; unrelated file, not in this plan's changeset). Logged in `deferred-items.md` for a future hardening plan. My changed production files are clippy-clean.

## Known Stubs

- `DeviceCtrConfig` is an intentional empty placeholder struct — Plan 08 fills it with CTR kind/prior/target-binarization. Its mere presence (`ctr: Some(_)`) already routes to `Ok(None)` in Plan 01, so it is inert until then. Documented, wave-owned resolution — not a blocking stub.

## Self-Check: PASSED

- Files exist: `session_depth_gt1_test.rs`, `deferred-items.md` ✓
- Commits exist: `49541ac`, `c51122f`, `8973b47` ✓
- `step_nodes`/`node_id_to_leaf_id` on `DeviceGrownTree` ✓; `DeviceTrainConfig` defined + exported ✓
- rocm 4/4 session tests green in-env ✓
