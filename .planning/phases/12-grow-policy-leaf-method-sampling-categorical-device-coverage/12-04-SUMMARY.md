---
phase: 12-grow-policy-leaf-method-sampling-categorical-device-coverage
plan: 04
subsystem: gpu-training / grow-policy
status: complete
tags: [cubecl, rocm, device-grow, region, path-model, GPUT-18, coverage-gate]

# Dependency graph
requires:
  - phase: 12-01
    provides: "DeviceGrownTree carrier + DeviceTrainConfig + all-or-nothing coverage gate"
  - phase: 12-02
    provides: "cb_train region_grower + cb_model RegionTree/region_leaf apply — the frozen ≤1e-5 CPU Region oracle this device kernel reproduces"
  - phase: 12-03
    provides: "nonsym_grow device_best_split_for_node per-frontier scoring spine + shape-aware boosting device-fold dispatch"
provides:
  - "cb_backend region_device::grow_region_tree — host-driven device Region PATH grow (walk-until-diverge, MaxLeaves = depth+1), structure EXACT vs frozen CPU Region, leaf values bit-exact (0.0 divergence) on gfx1100"
  - "DeviceGrownTree.region_path carrier ((feature,bin,expected_direction,one_hot) per level) — the Region dispatch key, EMPTY for oblivious/non-sym"
  - "session.rs Region gate arm (Ok(None)→device) + RegionState + grow_one Region dispatch; Region device-eligible in boosting.rs"
  - "boosting.rs device-fold Region arm — region_path DeviceGrownTree → RegionTree into region_trees (NOT a degenerate ObliviousTree)"
affects: [12-05-exact, 12-06-bootstrap, 12-07-mvs, 12-08-ctr, 12-09-kaggle-signoff]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Region device grow REUSES the non-sym per-frontier scorer (device_best_split_for_node, exposed pub(crate)) — only the SELECTION differs (single path, higher-child-gain continue direction), transcribing region_grower's SelectLeavesToSplit rule inline (no cb-train dep)"
    - "Region is a PATH carrier (region_path, d entries → d+1 leaves), dispatched BEFORE step_nodes/oblivious in both the device-fold and CPU folds — never a 2^d node graph"
    - "Padded n_bins to a device-dispatchable line size (32) for the small frozen fixture — pointwise_hist2 only dispatches {2,32,64,128,256}; empty upper buckets do not perturb the argmax"

key-files:
  created:
    - crates/cb-backend/src/kernels/region_device.rs
    - crates/cb-backend/src/kernels/region_device_test.rs
    - crates/cb-train/tests/device_region_fit_test.rs
  modified:
    - crates/cb-compute/src/runtime.rs
    - crates/cb-backend/src/kernels.rs
    - crates/cb-backend/src/kernels/nonsym_grow.rs
    - crates/cb-backend/src/gpu_runtime/session.rs
    - crates/cb-backend/src/gpu_runtime/session_depth_gt1_test.rs
    - crates/cb-train/src/boosting.rs
    - crates/cb-train/tests/device_seam_test.rs

key-decisions:
  - "Region device grow is HOST-DRIVEN like the non-sym grow (per-frontier device split scorer over a host doc-subset, pointwise_hist2 Atomic<f64>) — NOT resident, so it does NOT need Atomic<u64> and is unaffected by the in-env resident-histogram capability regression (deferred-items 12-06). Both region oracles RAN in-env on gfx1100 bit-exact."
  - "The self-oracle hardcodes the frozen Plan-02 path ([(0,1,false,false),(0,0,true,false)], leaf_of [1,1,2,2,0,0]) + computes expected leaf values via cb_compute::calc_average in-test (cb-backend cannot use cb_train — the feature-unification landmine); the PATH structure is the real oracle, leaf values via the deterministic calc_average formula."
  - "GPUT-18 stays Pending: Region is the last non-sym family device grow, but the authoritative ε=1e-4 GPU correctness sign-off is the human-gated Kaggle CUDA notebook (Plan 09). In-env rocm + CPU self-oracle is the local gate only."
  - "n_bins padded 3→32 in the self-oracle fixture (bin VALUES unchanged {0,1,2}) because pointwise_hist2 only dispatches line sizes {2,32,64,128,256}; the e2e ramp fixture already uses 32."

requirements-completed: []  # GPUT-18 phase-spanning; Kaggle CUDA sign-off (Plan 09) still pending.

# Metrics
duration: ~50min
completed: 2026-07-04
tasks: 2
files: 10
---

# Phase 12 Plan 04: Device Region Grow-Policy Path Summary

**Flipped the Region grow policy from `Ok(None)` to a real device grow (GPUT-18 / D-03a — the single largest Phase-12 lift, second half of the Region two-step). `region_device::grow_region_tree` grows one Region PATH on device (walk-until-diverge, `MaxLeaves = MaxDepth+1`) by REUSING the Plan-03 non-symmetric per-frontier device split scorer with a Region SELECTION, emits a plain-host `region_path` across the seam, and the now-three-way `boosting.rs` device-fold arm materializes it into `region_trees` (NOT a degenerate `ObliviousTree`). Verified on real gfx1100: path structure EXACT and leaf values bit-exact (0.0 divergence) vs the frozen Plan-02 CPU Region reference, end-to-end through `cb_train::train()`.**

## Accomplishments

**Task 1 — device Region kernel + carrier + session gate (commit `edd6452`)**
- `DeviceGrownTree.region_path: Vec<(u32,u32,bool,bool)>` — the per-level `(feature, bin, expected_direction, one_hot)` walk-until-diverge carrier (`d` entries → `d+1` leaves, NOT a `2^d` node graph); plain host type, filled at every existing construction site (empty for oblivious/non-sym).
- `region_device.rs::grow_region_tree` composes `nonsym_grow::device_best_split_for_node` (exposed `pub(crate)`) per frontier, selecting the Region continue direction (the child whose own best split has the higher gain continues; deterministic `>=` tie prefers the passes child — the SAME rule as `region_grower`). No `#[cube]` of its own (composes the finite-sentinel `launch_find_optimal_split_pointwise`); host-only `f64::NEG_INFINITY`; no `cb-train`/`cb-model` dep.
- `session.rs`: `region_active` gate arm (family-default all-or-nothing: no subsampling/MVS/exact/CTR/leaf-cap), `RegionState` (host bins+weights, der1 re-derived per tree), `grow_one` Region dispatch, and the resident line-size dispatch check skipped for Region (it scores via `pointwise_hist2`).

**Task 2 — self-oracle + boosting Region arm + e2e (commit `2ba4d9e`)**
- `boosting.rs` device-fold **Region arm** (dispatched FIRST on `region_path` non-empty, mirroring the CPU fold order): resolves each level's `(feature,bin)` to a `Split` via the range-checked `feature_borders` join, computes per-object leaf via the transcribed `region_leaf` walk (`bin=0; for level { split = one_hot ? val==border : val>border; if split != expected_direction break; bin+=1 }`, checked access → malformed halts, no panic), normalizes + folds a `RegionTree` into `region_trees`. Region added to `device_host_eligible` + `device_grow_policy`.
- `region_device_test.rs`: the frozen Plan-02 fixture through the device path — `region_path` == `[(0,1,false,false),(0,0,true,false)]` EXACT, `leaf_of` == `[1,1,2,2,0,0]` EXACT, `step_nodes`/`node_id_to_leaf_id` empty (path not graph), leaf values ≤1e-4 vs the `calc_average` reference. gfx1100: `abs_div=0.000e0`.
- `device_region_fit_test.rs`: a device Region `cb_train::train()` fit folds into `region_trees` (oblivious/non-sym EMPTY, each tree `d+1` leaves) and reproduces the CPU `region_grower` model. gfx1100: `max |Δpred| = 0.000e0` (closes the degenerate-`ObliviousTree` regression).
- `session_depth_gt1_test`: flipped the stale Plan-03 "Region declines until Plan 04" assertion to "Region now covered" (Rule 1).

## Verification (rocm gfx1100 in-env + cpu)

- `cargo test -p cb-backend --features rocm region_device` — **1/1**: device Region path structure EXACT vs frozen CPU Region reference, leaf-value **max abs_div = 0.000e0** (bar ε=1e-4).
- `cargo test -p cb-train --features rocm --test device_region_fit_test` — **1/1**: device Region fit → `region_trees` reproduces CPU `region_grower` with **max |Δpred| = 0.000e0**; oblivious/non-sym EMPTY.
- `cargo build -p cb-backend --features rocm` — green (ROCm smoke).
- No-regression: rocm `session` 10/10 (after the stale-assertion fix), `nonsym_grow` 4/4, `grow_loop` 14/14, `device_nonsym_fit` 2/2; cpu `cb-train --lib` 237/237, `device_seam_test` 6/6, `region_e2e_test` 2/2, `non_symmetric_grower_oracle_test` 1/1, `cb-model` all green.
- Landmine checks: no `use cb_train`/`use cb_model` in `region_device.rs`; no `-inf` literal / no `#[cube]` in the module; no new `use cb_backend` in `boosting.rs`; no `mod tests` in production source.

## Deviations from Plan

### 1. [Rule 1 — Bug] Stale Plan-03 "Region declines" assertion flipped
- **Found during:** Task 2 rocm no-regression run.
- **Issue:** `session_depth_gt1_test::session_depth_gt1_gate_declines_uncovered` asserted `!open(Region)` with the message "no device kernel until Plan 04" — now stale, since Plan 04 covers Region. It failed once the gate flipped.
- **Fix:** Flipped to `open(Region)` == true with an updated "Region now covered (Plan 04)" message. `session_depth_gt1_test.rs` was not in the plan's `files_modified` but the assertion is mechanically required by the coverage change.
- **Commit:** `2ba4d9e`.

### 2. [Rule 3 — Blocking] Self-oracle fixture n_bins padded 3 → 32
- **Found during:** Task 2 first rocm run of the self-oracle (`Degenerate("pointwise_hist2 one-byte non-binary fill expects n_bins in {32,64,128,256} ... got 3")`).
- **Issue:** The frozen Plan-02 fixture has 3 bins (borders `[0.5,1.5]`), but the device `pointwise_hist2` fill only dispatches line sizes `{2,32,64,128,256}`.
- **Fix:** Padded `n_bins` to 32 in the self-oracle (the bin VALUES stay `{0,1,2}`; the empty upper buckets contribute nothing, so the argmax picks the SAME frozen splits — verified bit-exact). The e2e ramp fixture already uses 32.
- **Commit:** `2ba4d9e`.

### 3. [Rule 3 — Blocking] End-to-end oracle lives in `tests/`, not `src/`
- **Issue:** The plan lists `crates/cb-train/src/device_region_fit_test.rs`, but a src-mounted test cannot instantiate `cb_model` (the `cb_train` dev-dep diamond → E0308, the 12-02/12-03 precedent).
- **Fix:** Placed it at `crates/cb-train/tests/device_region_fit_test.rs` (integration), mirroring `device_nonsym_fit_test.rs` (local `CpuRefRuntime` reference since `CpuBackend` is not compiled under the rocm feature). Same `lr=0.3` partial-fit rationale as Plan 03.

## Known Stubs

None. The one-hot Region walk branch (`val == border`) is transcribed but inert (the device float grower always emits `one_hot=false`); it is carried for structural fidelity / future categorical-Region parity, not a stub.

## Threat Flags

None — numerical GPU kernel work only, no new network/auth/upload/untrusted-input surface. The T-12-07 (bounded `0..depth` loop + checked leaf index) and T-12-02 (deterministic reduction, inherited from the composed scorer) mitigations are present.

## Self-Check: PASSED

- Files exist: `region_device.rs`, `region_device_test.rs`, `device_region_fit_test.rs` — all FOUND.
- Commits exist: `edd6452`, `2ba4d9e` — both FOUND.
- rocm gfx1100 in-env: region_device 1/1 (0.0 divergence), device_region_fit 1/1 (0.0 divergence) ✓
- No `use cb_train`/`use cb_model` in region_device.rs; no `-inf`/`#[cube]`; no new `use cb_backend` in boosting.rs ✓
