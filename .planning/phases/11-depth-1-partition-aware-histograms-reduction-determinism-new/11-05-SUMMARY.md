---
phase: 11-depth-1-partition-aware-histograms-reduction-determinism-new
plan: 05
subsystem: testing
tags: [kaggle, cuda, oracle, bench-02, depth-6, gput-14, gput-06, human-gated, checkpoint]

# Dependency graph
requires:
  - phase: 11-depth-1-partition-aware-histograms-reduction-determinism-new
    plan: 03
    provides: "depth>1 device grow loop + partition_hist_reduce_zero_spread + depth6_rmse_grow_matches_cpu CUDA grow-loop split-agreement oracle"
  - phase: 11-depth-1-partition-aware-histograms-reduction-determinism-new
    plan: 04
    provides: "device Newton der2 leaf estimation (Logloss arm) + newton_leaf_matches_cpu oracle"
  - phase: 11-depth-1-partition-aware-histograms-reduction-determinism-new
    plan: 01
    provides: "expected_depth6_tree.json (byte-unchanged CPU reference-of-record, A1/A2 pinned) + X_depth6_speed.npy + SPEED_CONFIG"
provides:
  - "bench/cuda_oracle.ipynb depth-6 section: Gate A (device 1-tree vs committed CPU ref, base-free, <=1e-4 RMSE+Logloss), Gate B (device full-run vs Rust cpu-wheel preds), per-tree split-agreement + run-to-run spread diagnostic, depth-6 BENCH-02 speed cells, depth-6 structured report"
  - "bench/RESULTS.md dated ## Phase 11 depth-6 sign-off block (all correctness+speed fields PENDING the human Kaggle run)"
affects: [phase-11-verification, GPUT-14, GPUT-06, BENCH-02]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Device bar ε=1e-4 (GPUT-14); CPU reference-of-record stays ≤1e-5 (Plan 01), byte-unchanged (D-04)"
    - "Base-free centered prediction comparison (device vs committed fixture) cancels the RMSE mean prior / Logloss logit prior — a CUDA-alone gate needing no cpu wheel for the single-tree A1 config"
    - "Compile-time backend selection (CLAUDE.md) ⇒ device (cuda) and CPU (cpu) wheels cannot coexist in one process: the literal device-vs-CPU-path full-run gate (Gate B) loads cpu-wheel preds saved from a separate kernel; PENDING if absent (never fabricated)"
    - "Correctness is BLOCKING before any speed number (T-10-25); every measured cell is filled only from the human-gated Kaggle run"

key-files:
  created: []
  modified:
    - bench/cuda_oracle.ipynb
    - bench/RESULTS.md

key-decisions:
  - "Gate A (CUDA-alone, byte-unchanged fixture) is the primary always-runnable blocking gate: device single depth-6 tree (iterations=1 == fixture A1) vs the committed Plan-01 CPU reference, compared base-free so no cpu wheel is needed and the CPU reference is not retrained (D-04)"
  - "Gate B (device full-run vs the Rust cpu-feature-wheel preds) is the literal 'device vs Rust CPU path' over a 200-iter depth-6 run; because backends are compile-time it loads cpu_pred_{rmse,logloss}.npy produced by a separate cpu-wheel kernel, and reports PENDING (not fabricated) when absent"
  - "Logloss raw margin derived via predict_proba->logit: CatBoostClassifier.predict takes only (x) and returns class labels (no prediction_type kwarg); base-free centering makes the logit prior cancel"
  - "Per-tree diagnostic = (1) the in-tree cb-backend CUDA grow-loop split-agreement oracles (depth6_rmse_grow_matches_cpu / newton_leaf_matches_cpu / partition_hist_reduce_zero_spread) for per-level split exactness + zero spread, plus (2) a device run-to-run spread curve over growing prefixes that localizes any compounding split-flip to the tree where the spread first steps up (Pitfall 4)"

requirements-completed: []
requirements-pending: [GPUT-14, GPUT-06, BENCH-02]

# Metrics
duration: 35min
completed: 2026-07-03
status: awaiting-human-checkpoint
---

# Phase 11 Plan 05: Kaggle CUDA depth-6 oracle harness Summary

**Extended `bench/cuda_oracle.ipynb` with the authoritative depth-6 GPU oracle — a blocking
final-prediction ε=1e-4 gate (RMSE + Logloss, device vs the Rust CPU path), a per-tree
split-agreement + run-to-run spread diagnostic that localizes compounding drift (GPUT-06 /
D-05), and depth-6 BENCH-02 speed cells — plus a dated Phase-11 sign-off skeleton in
`bench/RESULTS.md` with every number field explicitly PENDING the human-gated Kaggle CUDA
run (no fabricated results).**

## Status: AWAITING HUMAN CHECKPOINT (Task 3)

Tasks 1–2 (all preparable work) are complete and committed. **Task 3 is a
`checkpoint:human-verify` (`gate="blocking-human"`)**: the Kaggle CUDA notebook run is the
sole authoritative GPU oracle for both correctness and speed and CANNOT be executed in-env
(no local CUDA; gfx1100 lacks the f64 atomic-add smoke path). The plan is **NOT complete**
until the human runs the notebook and fills the RESULTS.md Phase-11 block.

## Performance
- **Duration:** ~35 min (preparable work; excludes the human Kaggle run)
- **Completed (Tasks 1–2):** 2026-07-03
- **Tasks:** 2 of 3 (Task 3 human-gated)
- **Files modified:** 2 (+ deferred-items.md)

## Accomplishments
- **Task 1 (GPUT-14 gate + GPUT-06 diagnostic):** Added a self-contained depth-6 section to
  `bench/cuda_oracle.ipynb`:
  - **Gate A (BLOCKING, CUDA-alone):** device single depth-6 tree (`iterations=1`, the
    fixture's pinned A1 config) vs the committed Plan-01 CPU reference, compared **base-free**
    (centered prediction differences, so the RMSE mean prior / Logloss logit prior cancels),
    asserting `max|Δ| ≤ 1e-4` for **RMSE and Logloss**. Needs no cpu wheel and does not
    retrain the CPU reference (D-04).
  - **Gate B:** the literal "device vs Rust CPU path" over a 200-iter depth-6 run, loading
    `cpu_pred_{rmse,logloss}.npy` from a separate cpu-feature-wheel kernel (compile-time
    backends); PENDING when absent.
  - **Per-tree diagnostic:** runs the in-tree `cb-backend` CUDA grow-loop oracles
    (`depth6_rmse_grow_matches_cpu`, `newton_leaf_matches_cpu`,
    `partition_hist_reduce_zero_spread`) for per-level split `(feature,bin)` exactness +
    zero run-to-run spread, plus a device run-to-run spread curve over growing prefixes that
    localizes a compounding split-flip to the tree where the spread first steps up (Pitfall 4).
- **Task 2 (BENCH-02):** Added depth-6 device train-only speed cells (RMSE + Logloss,
  warm-run/JIT-excluded/queue-drained on the regenerated large-n `SPEED_CONFIG`), host-CPU
  (cpu wheel) + official CatBoost GPU (`task_type='GPU'`, depth=6) comparison, and a depth-6
  structured report cell. Appended a dated `## Phase 11` block to `bench/RESULTS.md` with a
  copy-paste run template — all correctness + speed fields marked **TBD/PENDING** until the
  Kaggle run, with correctness recorded as blocking-before-speed.

## Task Commits
1. **Task 1: depth-6 final-ε=1e-4 gate + per-tree diagnostic** — `9351d21` (feat)
2. **Task 2: depth-6 BENCH-02 speed cells + RESULTS.md sign-off skeleton** — `da32d10` (feat)
3. **Task 3: human-gated Kaggle CUDA run** — PENDING (checkpoint; see below)

## Files Created/Modified
- `bench/cuda_oracle.ipynb` — appended 9 depth-6 cells (1 header md, load/helper, cpu-ref
  helper, Gate A+B, per-tree diagnostic, speed header md, device speed, CPU/CatBoost-GPU
  baselines, structured report). Depth-1 cells left byte-unchanged (D-04). (modified)
- `bench/RESULTS.md` — dated `## Phase 11 — depth-6 CUDA sign-off` block + run template, all
  fields PENDING/TBD (no fabricated numbers). (modified)
- `.planning/phases/11-.../deferred-items.md` — logged a pre-existing depth-1 cell-6 bug
  (out of scope). (modified)

## Deviations from Plan

### Plan-latitude / correctness choices (no deviation rule needing user sign-off)

**1. [Plan-latitude] Two-gate structure (Gate A CUDA-alone + Gate B cpu-wheel).** The plan
asks for "device vs Rust CPU path" ≤1e-4. Backends are compile-time (CLAUDE.md), so the
device (cuda) and CPU (cpu) `catboost_rs` wheels cannot be imported in one process. Gate A
compares the device tree to the committed byte-unchanged CPU reference-of-record via
base-free centering (always runnable on CUDA alone, matches the fixture's A1 iterations=1
config); Gate B is the literal full-run device-vs-cpu-wheel comparison, loading preds saved
from a separate cpu-wheel kernel (PENDING if absent). Both are legitimate "CPU path"; Gate A
is the blocking always-runnable gate, Gate B the extended full-run confirmation.

**2. [Rule 1 - Correctness] Logloss margin via `predict_proba`→logit, not
`prediction_type='RawFormulaVal'`.** `CatBoostClassifier.predict` in `catboost-rs-py` takes
only `(x)` and returns class labels — there is no `prediction_type` kwarg. The new cells
derive the raw margin as `logit(predict_proba[:,1])`; base-free centering cancels the prior.
The pre-existing depth-1 cell 6 still uses the unsupported `prediction_type` kwarg — left
byte-unchanged (D-04, depth-1 reference) and logged to `deferred-items.md` (SCOPE BOUNDARY:
not caused by this task).

**3. [Plan-latitude] Per-tree diagnostic composed from the Rust CUDA grow-loop oracles + a
Python prefix spread curve.** The Python binding exposes no staged/per-tree API, so per-tree
split-agreement is evidenced by the in-tree CUDA grow-loop oracles (per-level split exact vs
CPU + zero spread), and the "first divergent tree" localization is a device run-to-run
spread curve over growing prefixes (a step-change pinpoints the originating tree, Pitfall 4).

## Known Stubs / Pending
- **RESULTS.md Phase-11 numbers are all PENDING/TBD** — this is intentional and required:
  they are filled ONLY from the human-gated Kaggle run (Task 3), never fabricated
  (threat T-11-05-01).
- **Gate B** reports PENDING unless a cpu-feature-wheel run has produced
  `cpu_pred_{rmse,logloss}.npy`. Gate A is the always-runnable blocking gate.

## Threat Mitigations (Task threat register)
- **T-11-05-01** (fabricated/unverified numbers): every RESULTS.md number is TBD/PENDING and
  filled only from the human Kaggle run; correctness is blocking-before-speed; the notebook
  asserts the gate before timing (T-10-25).
- **T-11-05-02** (non-deterministic reduce passing-local/failing-aggregate): the per-tree
  run-to-run spread curve + the LOCKED deterministic reduce (Plans 02/03) localize any
  compounding drift to the originating tree.
- **T-11-SC** (package installs): none added/removed; the notebook uses official CatBoost GPU
  + the existing harness only.

## User Setup Required (Task 3 — the human checkpoint)
Run the depth-6 section of `bench/cuda_oracle.ipynb` on a Kaggle CUDA GPU:
1. Build/install the `--features cuda` wheel + confirm `nvidia-smi` (Step-0 cells).
2. Run the depth-6 gate cell — confirm Gate A ε=1e-4 PASSES for RMSE + Logloss (blocking).
3. (Optional, literal full-run) build a `--features cpu` wheel in a fresh kernel, run the
   `BUILD_CPU_REF=True` helper to emit `cpu_pred_*.npy`, then re-run the gate for Gate B.
4. Run the per-tree diagnostic cell — confirm the split-agreement oracle PASSES and the
   run-to-run spread shows no step-change (no compounding split-flip drift).
5. Run the depth-6 speed cells (warm-run/JIT-excluded, train-only) — record device vs
   host-CPU vs official CatBoost GPU.
6. Paste the structured-report numbers into the `## Phase 11` block of `bench/RESULTS.md`
   (correctness first, then speed). Do NOT fabricate.

## Next Phase Readiness
- Phase 11 verification is gated on this human Kaggle run. Once the human confirms the
  ε=1e-4 gate PASSES (RMSE + Logloss), the per-tree diagnostic shows no compounding drift,
  and the depth-6 speed is logged, GPUT-14 / GPUT-06 / BENCH-02 are discharged and the plan
  can be marked complete + requirements checked off.

---
*Phase: 11-depth-1-partition-aware-histograms-reduction-determinism-new*
*Tasks 1–2 completed: 2026-07-03 · Task 3 awaiting human-gated Kaggle CUDA run*
