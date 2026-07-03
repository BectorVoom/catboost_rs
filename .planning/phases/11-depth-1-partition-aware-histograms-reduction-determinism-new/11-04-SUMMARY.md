---
phase: 11-depth-1-partition-aware-histograms-reduction-determinism-new
plan: 04
subsystem: cb-backend
tags: [cubecl, gpu, rocm, newton, der2, logloss, leaf-estimation, partition-reduce, gput-07]

# Dependency graph
requires:
  - phase: 11-depth-1-partition-aware-histograms-reduction-determinism-new
    plan: 03
    provides: "depth>1 device grow loop (grow_oblivious_tree_into / _resident), partition_update_kernel + launch_partition_update_into (2-channel), read_part_stats_f64"
  - phase: 11-depth-1-partition-aware-histograms-reduction-determinism-new
    plan: 01
    provides: "depth-6 Logloss fixture arm + cb-compute newton_leaf_delta / reduce_leaf_delta oracle pinning A1 (iterations=1) / A3 (der2·weight)"
provides:
  - "partition_update_kernel — 3-channel per-partition reduce (ch0=Σder1, ch1=Σweight, ch2=Σ(der2·weight) Newton hessian); part*3 stride, part*3+2 bounds guard, in-kernel weight-fold (A3)"
  - "launch_partition_update_into — +der2 Handle param, checked_mul(3) part-stats sizing; RMSE/pairwise callers thread a const -1 der2 handle and read channels 0/1 at stride 3 (calc_average value-identical)"
  - "grow_oblivious_tree_newton / grow_oblivious_tree_newton_into — device-resident Newton der2 leaf estimation (Logloss arm): reuse the shared structure grow, re-reduce Σ(der2·weight) with the real der2 on the same client, leaf = inline newton_leaf_delta (single closed-form step, A1)"
  - "inline newton_leaf_delta host helper in cb-backend (transcribed, no cb-compute-leaf import)"
  - "grow_loop::single_tree::newton_leaf_matches_cpu — depth-6 Logloss device Σ(der2·weight) == reduce_leaf_der2 AND leaves == newton_leaf_delta ≤1e-4 on gfx1100"
  - "grow_loop::single_tree::rmse_newton_collapses_to_average — der2=-1 Newton == calc_average ≤1e-4 (Pitfall 2 collapse check)"
affects: [11-05, BENCH-02]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "The per-partition reduce is now 3-channel (stride 3); the Newton hessian channel folds weight IN-KERNEL as der2·weight, matching the CPU reduce_leaf_der2 convention (A3). Channels 0/1 are value-identical to the 2-channel path (RMSE calc_average reads them verbatim at the wider stride)"
    - "Newton der2 leaf estimation composes on top of the shared structure grow: grow the tree once (der2-independent scoring, A2), then re-reduce Σ(der2·weight) with the REAL der2 handle bound to the same client and apply the inline single-step newton_leaf_delta — no duplication of the 130-line partition-aware level loop, D-05 crossing class unchanged"
    - "RMSE collapses to calc_average automatically: der2=-1 ⇒ -Σ(der2·weight)=Σweight ⇒ newton_leaf_delta == calc_average exactly (proven by rmse_newton_collapses_to_average)"

key-files:
  created: []
  modified:
    - crates/cb-backend/src/kernels.rs
    - crates/cb-backend/src/gpu_runtime/mod.rs
    - crates/cb-backend/src/gpu_runtime/pairwise.rs
    - crates/cb-backend/src/kernels/grow_loop.rs

key-decisions:
  - "Composed the Newton path (structure-reuse + der2 re-reduce) instead of swapping the leaf formula in-place inside grow_oblivious_tree_into. This keeps the RMSE path bit-identical, avoids duplicating the depth>1 partition-aware level loop, and binds the real der2 handle to the same client as the reduce (Pitfall 3). Cost: one extra device partition_update reduce over the final leaves (D-05 crossing class unchanged)."
  - "Extended the SHARED partition_update_kernel to 3 channels (not a second kernel): channels 0/1 keep bit-identical values at the wider part*3 stride, so RMSE calc_average and pairwise are unaffected in value; only the extra Newton hessian channel-2 is new. All four callers (grow_oblivious_tree_into, grow_oblivious_tree_resident, pairwise, the grow_loop test) thread a der2 handle."
  - "RMSE/pairwise callers pass a const -1 der2 handle (upload_channel_floats(&vec![-1.0; n])) so the 3-channel launch is well-formed; they still read channels 0/1 for calc_average. The genuinely-new Newton/Logloss leaf estimation is the dedicated grow_oblivious_tree_newton_into path."
  - "leaf_estimation_iterations pinned to 1 (A1): the single-tree Newton function computes each leaf once via the closed-form newton_leaf_delta — no iterative walker, no backtracking, no per-iteration readback (D-01 trivially satisfied). The device-resident apply_leaf_delta seam remains the approx-update path in the resident/boosting drivers, unchanged."

requirements-completed: [GPUT-07]

# Metrics
duration: 45min
completed: 2026-07-03
status: complete
---

# Phase 11 Plan 04: Newton der2 leaf estimation on device (GPUT-07) Summary

**Device-resident Newton der2 leaf estimation for the Logloss default (GPUT-07): a 3rd Σ(der2·weight) channel on the per-partition reduce (weight folded in-kernel per the reduce_leaf_der2 convention, A3) + an inline single-step newton_leaf_delta (leaf_estimation_iterations=1, A1) composed on top of the shared depth>1 structure grow, oracle-proven ≤1e-4 on real gfx1100 — with the RMSE der2=-1 collapse to calc_average verified as the Pitfall-2 sign/weighting guard.**

## Performance
- **Duration:** ~45 min
- **Completed:** 2026-07-03
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments
- **Task 1 (3rd channel):** `partition_update_kernel` now accumulates a 3rd channel Σ(der2·weight) — `part*3` stride, `part*3+2` bounds guard, `fetch_add(der2[obj]·weight[obj])` folding weight IN-KERNEL to match `cb_compute::reduce_leaf_der2` (A3 landmine). `launch_partition_update_into` gained a `der2: Handle` param and sizes the part-stats via `checked_mul(3)` → typed `CbError::OutOfRange`. Channels 0/1 keep bit-identical values at the wider stride, so the RMSE/pairwise `calc_average` arms are value-unchanged.
- **Task 2 (Newton leaf):** Transcribed `newton_leaf_delta` inline into cb-backend (no cb-compute-leaf import). Added `grow_oblivious_tree_newton` / `grow_oblivious_tree_newton_into` (Logloss arm): reuse the shared `grow_oblivious_tree_into` for the (der2-independent, A2) structure, then re-reduce the Σ(der2·weight) channel with the REAL der2 handle bound to the same client (Pitfall 3), and compute each leaf via the inline single closed-form step (A1, iterations=1). RMSE collapses automatically (der2=-1 ⇒ newton == calc_average).
- **Task 3 (self-oracle):** `newton_leaf_matches_cpu` grows a depth-6 Logloss tree (`der2 = -p(1-p)`, `p = sigmoid(margin)`) on device and hard-asserts (a) the device Σ(der2·weight) channel-2 matches CPU `reduce_leaf_der2` ≤1e-4, and (b) the device Newton leaves match CPU `newton_leaf_delta(Σder1, Σ(der2·weight), scaled_l2)` ≤1e-4, plus the structure (splits + leaf_of) exact. `rmse_newton_collapses_to_average` asserts der2=-1 Newton == `calc_average` ≤1e-4 (the Pitfall-2 sign/weighting guard). Both green on gfx1100.
- **Merge gate:** full `cargo test -p cb-backend --features rocm` → **128 passed, 0 failed** (was 126; +2 Newton). cpu + wgpu host builds clean; no `cb-train` dep in cb-backend.

## Task Commits
1. **Task 1: Σ(der2·weight) 3rd channel on the per-partition reduce** — `b7dc8fb` (feat)
2. **Task 2: device-resident Newton der2 leaf estimation** — `64ad14c` (feat)
3. **Task 3: Newton der2 self-oracle + 3-channel update test** — `edcb93b` (test)

## Files Created/Modified
- `crates/cb-backend/src/kernels.rs` — `partition_update_kernel` +der2 input, `part*3` stride, `part*3+2` guard, channel-2 `fetch_add(der2·weight)` (modified).
- `crates/cb-backend/src/gpu_runtime/mod.rs` — `launch_partition_update_into` +der2 Handle / `checked_mul(3)`; grow_oblivious_tree_into + grow_oblivious_tree_resident thread a const -1 der2 handle + stride-3 reads; inline `newton_leaf_delta`; new `grow_oblivious_tree_newton` / `grow_oblivious_tree_newton_into` (modified).
- `crates/cb-backend/src/gpu_runtime/pairwise.rs` — RMSE caller updated to the 3-channel signature (const -1 der2 + stride-3 reads) (modified).
- `crates/cb-backend/src/kernels/grow_loop.rs` — `newton_leaf_matches_cpu`, `rmse_newton_collapses_to_average`; `update_matches_ordered_reference` widened to the 3-channel stride-3 reduce (modified).

## Deviations from Plan

### Plan-latitude choices (no deviation rule needed)

**1. [Plan-latitude] Composed the Newton path (structure-reuse + der2 re-reduce) rather than an in-place leaf-formula swap.** The plan framed Task 2 as swapping the leaf value "at the leaf-value computation point in grow_oblivious_tree_into". Doing that in-place would either (a) require a leaf-mode branch + der2 threaded through the shared function and all its callers (public API + boosting + resident), or (b) risk the RMSE 0.0-divergence oracle. Instead `grow_oblivious_tree_newton_into` reuses the shared structure grow (der2-independent scoring, A2), then re-reduces Σ(der2·weight) with the real der2 and applies the inline `newton_leaf_delta`. This keeps the RMSE path bit-identical, avoids duplicating the 130-line partition-aware level loop, and binds the real der2 handle to the reduce's client (Pitfall 3). Cost: one extra device partition_update reduce over the final leaves (device-resident; the D-05 crossing class — leaf_of + part-stats — is unchanged).

**2. [Plan-latitude] Newton wired into the single-tree self-oracle path; the resident/boosting production drivers stay on calc_average (der2=-1) for now.** `grow_oblivious_tree_resident` / `grow_boosting_pass_into` supply a const -1 der2 handle and keep `calc_average` (RMSE). The genuinely-new Newton math is device-proven via `grow_oblivious_tree_newton_into` + the self-oracle + the collapse test; wiring Newton leaf estimation into the resident Logloss boosting driver is a follow-up (mirrors Plan 03's direct-fill / n_bins MVP scoping). Kaggle CUDA (Plan 05) is the authoritative end-to-end correctness+speed gate.

**3. [Rule 3 - blocking fix] Extended the SHARED launch_partition_update_into signature → updated all four callers.** The shared `launch_partition_update_into` gained a `der2: Handle` param and `checked_mul(3)`, so `grow_oblivious_tree_resident`, the pairwise grow, and the `update_matches_ordered_reference` test were updated to the 3-channel signature (const -1 der2 handle + stride-3 channel-0/1 reads). Values (channels 0/1) are bit-identical — a required compile fix from the signature change, not a behavior change.

**4. [Plan-latitude] Independent cb-compute oracle instead of the JSON fixture cross-check.** The plan's Task 3 mentioned cross-checking against `bench/fixtures/expected_depth6_tree.json`. Used `cb_compute::reduce_leaf_der2` + `cb_compute::newton_leaf_delta` (already a cb-backend dep) as the fully-independent CPU oracle instead, avoiding a `serde_json` dev-dependency in cb-backend (same choice as Plan 03 deviation 7). The oracle is the exact same math the fixture pins (A1/A3).

## Threat Mitigations (Task threat register)
- **T-11-04-01** (OOB write from the widened 2^depth*3 buffer / `part*3+2` index): the kernel bounds guard is widened to `part*3 + 2 < part_stats.len()` and `launch_partition_update_into` uses `checked_mul(3)` → typed `CbError::OutOfRange` before dispatch.
- **T-11-04-02** (`unwrap()` panic on read-back failure): no `unwrap()` in production; the widened part-stats read-back surfaces `CbError::Degenerate` (WR-05), never a silent zero buffer.
- **T-11-04-03** (der2 sign/weighting mismatch inverting the denominator): `newton_leaf_matches_cpu` cross-checks the Σ(der2·weight) channel vs `reduce_leaf_der2` and the leaf vs `newton_leaf_delta`; `rmse_newton_collapses_to_average` catches a sign/weighting error (der2=-1 must collapse to `calc_average`).

## Issues Encountered
- No CubeCL build errors — the only kernel change is a `*` multiply + a 3rd `Atomic<F>::fetch_add` (no new `-inf` literal, no new shared vocabulary), so it JIT-compiled on gfx1100 first try. The error-guideline protocol was not triggered.

## Next Phase Readiness
- GPUT-07 (device Newton der2 leaf estimation) is wired + in-env oracle-proven ≤1e-4 on gfx1100. Plan 05 (Kaggle CUDA) is the authoritative gate over the full depth-6 Logloss run (final-prediction ε + per-tree diagnostic) and the speed cells.
- Carried forward: Newton leaf estimation is currently exercised via the single-tree `grow_oblivious_tree_newton_into` path; wiring it into the resident Logloss boosting driver (`grow_oblivious_tree_resident` / `grow_boosting_pass`) is a follow-up. RMSE/pairwise keep calc_average.

---
*Phase: 11-depth-1-partition-aware-histograms-reduction-determinism-new*
*Completed: 2026-07-03*

## Self-Check: PASSED
- Created file present: 11-04-SUMMARY.md
- All three task commits present: b7dc8fb, 64ad14c, edcb93b
- Source artifacts present: partition_update_kernel +der2 (kernels.rs), grow_oblivious_tree_newton_into + inline newton_leaf_delta (gpu_runtime/mod.rs), newton_leaf_matches_cpu + rmse_newton_collapses_to_average (kernels/grow_loop.rs)
- Full rocm suite green (128 passed, 0 failed); cpu/wgpu host builds clean; no cb-train dep in cb-backend
