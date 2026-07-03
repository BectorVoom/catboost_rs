---
status: testing
phase: 10-gpu-foundations-runtime-seam-session-residency-device-primit
source: [10-VERIFICATION.md]
started: 2026-07-03T00:00:00Z
updated: 2026-07-03T00:00:00Z
---

## Current Test

number: 1
name: Kaggle CUDA correctness + speed sign-off (authoritative GPU oracle of record)
expected: |
  On a Kaggle CUDA instance, build the `--features cuda` wheel and run
  `bench/cuda_oracle.ipynb` end-to-end. The notebook must:
    - Pass the BLOCKING primitive + bit-packed cindex correctness oracles vs the
      committed CPU-reference fixtures (bit-exact / integer-equal).
    - Pass the depth-1 device-vs-CPU-reference RMSE & Logloss correctness gate
      (RMSE ~1e-9, Logloss ≤1e-5; depth-1 whole-dataset level-0 histogram is the
      exact CPU score) and set GATE_PASSED=true.
    - ONLY THEN report warm-run/JIT-excluded, train-only wall-clock speed on the
      large-n (~1e6×50) workload: device path vs host-CPU baseline (BENCH-01/02).
    - Fill the SPIKE-REDUCTION.md §4 reduce err/ms rows and bench/RESULTS.md from
      the measured run (currently intentional TBD placeholders — nothing fabricated).
  ROCm in-env is smoke-only and does NOT satisfy this item; only the CUDA run is
  authoritative for this milestone.
awaiting: user response

## Tests

### 1. Kaggle CUDA correctness + speed sign-off (authoritative GPU oracle of record)
expected: |
  Build `--features cuda` wheel + run `bench/cuda_oracle.ipynb` on Kaggle CUDA.
  BLOCKING primitive + cindex oracles pass bit-exact; depth-1 RMSE ~1e-9 /
  Logloss ≤1e-5 gate passes (GATE_PASSED=true); then warm-run train-only speed
  (device vs host-CPU baseline) recorded into RESULTS.md + SPIKE-REDUCTION.md §4.
  This is the milestone's sole authoritative GPU numeric oracle; harness is
  structurally complete and reproducible in-env, numbers deferred to this run.
result: [pending]

### 2. WR-01 — disclosed non-blocking leaf_of read-back wiring caveat (accept/schedule decision)
expected: |
  Human accept-or-schedule decision on a self-disclosed, non-correctness wiring
  discrepancy: `grow_oblivious_tree_resident`
  (crates/cb-backend/src/gpu_runtime/mod.rs:~2104) reads back an n-length
  `leaf_of` buffer on every production tree — one extra device→host crossing
  beyond ROADMAP.md Success-Criterion-4's literal wording ("only O(1) BestSplit
  + 2^depth partition stats cross per level"). It does NOT affect model
  correctness and is already disclosed in 10-REVIEW.md (WR-01) and the 10-07
  SUMMARY as deferred debt. Decide: accept as-is for the depth-1 MVP, or schedule
  closure (e.g. Phase 11 depth>1 partition-aware refactor where leaf_of becomes
  device-resident anyway).
result: [pending]

## Summary

total: 2
passed: 0
issues: 0
pending: 2
skipped: 0
blocked: 0

## Gaps
