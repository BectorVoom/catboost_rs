---
phase: 14-comprehensive-kaggle-cuda-speed-benchmark-parity-sign-off
plan: 02
subsystem: bench
tags: [kaggle, cuda, bench03, catboost-gpu, sign-off, oracle]
status: complete
requires:
  - bench/phase13_cuda_oracle/oracle.py (cloned structure)
  - crates/cb-train/tests/bench_grow_speed_test.rs (gen() workload source)
provides:
  - bench/phase14_cuda_signoff/oracle.py (Kaggle CUDA driver: Part A pre-flight + Part C CatBoost-GPU arm)
  - bench/phase14_cuda_signoff/kernel-metadata.json (Phase-14 kernel descriptor)
  - bench/phase14_cuda_signoff/oracle_gen_test.py (offline gen() reproduction test)
affects:
  - Plan 14-03 (human-gated Kaggle run consuming this driver; bench03-result.json feeds BENCH-03-SIGNOFF.md)
tech-stack:
  added: []
  patterns:
    - "Module-level importable gen() with lazy numpy import; all run-time work under if __name__ == '__main__'"
    - "Correctness-gated-before-speed: Part A blocking pre-flight sys.exit(2) before Part C"
key-files:
  created:
    - bench/phase14_cuda_signoff/oracle.py
    - bench/phase14_cuda_signoff/kernel-metadata.json
    - bench/phase14_cuda_signoff/oracle_gen_test.py
  modified: []
decisions:
  - "Reproduce bench_grow_speed_test.rs::gen() in numpy (orchestrator decision 2 / A2), NOT generator.py's 1e6x50 workload — the aggregated device/CPU numbers came from gen()"
  - "Region CatBoost-GPU cell = N/A (no official Region grow_policy), never a fabricated proxy"
  - "border_count set explicitly to 32 (GPU default is 128) to match the bench"
metrics:
  duration: ~2 min
  completed: 2026-07-05
  tasks: 2
  files: 3
---

# Phase 14 Plan 02: Kaggle CUDA driver (BENCH-03 Part A pre-flight + Part C CatBoost-GPU arm) Summary

Authored the human-gated Phase-14 Kaggle CUDA driver as a minimal, additive clone of the proven `phase13_cuda_oracle/oracle.py`: Part A keeps the per-family Rust device self-oracle as a BLOCKING correctness pre-flight (D-04), and a new Part C times official `catboost` `task_type='GPU'` on an exact numpy reproduction of `bench_grow_speed_test.rs::gen()`, emitting `bench03-result.json`. Proven offline via a 6-test `gen()` reproduction suite.

## What Was Built

- **`bench/phase14_cuda_signoff/oracle.py`** — the driver. All Kaggle-run work lives in `main()` under `if __name__ == "__main__"`; a module-level `gen(n, nf=20, nbins=32)` with a lazy `import numpy` reproduces the Rust workload (uint64 wrapping hash `(i*2654435761 + f*40503) % nbins` as float32; `+/-1` target on `X[:,0] + 0.5*X[:,1%nf] > nbins*0.75`). Part A reuses the five Phase-13 family device self-oracles under `--no-default-features --features cuda`; a failed roll-up writes a SOME-FAIL verdict and `sys.exit(2)` **before** Part C. Part C builds the frozen `CatBoostRegressor` (`task_type='GPU'`, RMSE, L2, depth 6, iters 20, lr 0.3, l2 0.0, `bootstrap_type='No'`, `border_count=32`, seed 42, `grow_policy='Depthwise'`), warm-fits an untimed 2000-row slice, times a single train-only fit for n in {10k,100k,300k}, and records Region as `catboost_gpu_s=None`/`grow_policy_used="N/A"`. Emits `bench03-result.json` + `bench03-result.md`.
- **`bench/phase14_cuda_signoff/kernel-metadata.json`** — fresh Phase-14 descriptor (`id: yensen2/catboost-rs-phase14-cuda-signoff`, `dataset_sources: [yensen2/catboost-rs-phase14-src]`), `enable_gpu:true` + `enable_internet:true` (internet is load-bearing for the Part C `pip install -q catboost` fallback).
- **`bench/phase14_cuda_signoff/oracle_gen_test.py`** — standalone pytest (source/test separation per CLAUDE.md) proving gen() shape `(n x 20)` float32, integer bins in `[0,32)`, determinism across calls, `+/-1` target with both classes, and spot-checked hash-formula cells. Uses `pytest.importorskip("numpy")` to skip cleanly without numpy; loads the sibling module via importlib without running `main()`.

## Tasks Completed

| Task | Name | Commit | Files |
| ---- | ---- | ------ | ----- |
| 1 | Kaggle CUDA driver (oracle.py) | 69dcf27 | bench/phase14_cuda_signoff/oracle.py |
| 2 | Kernel descriptor + offline gen() test | d769778 | kernel-metadata.json, oracle_gen_test.py |

## Verification

- `python3 -m py_compile bench/phase14_cuda_signoff/oracle.py` — succeeds.
- Import in-env (no GPU, no numpy at import time) exposes module-level `gen` and `main`; the run body is guarded, so nothing executes on import.
- Frozen config grep-verified: `task_type`, `border_count`, `bootstrap_type='No'`, `score_function='L2'`, `grow_policy='Depthwise'` all present.
- `python3 -m pytest bench/phase14_cuda_signoff/oracle_gen_test.py -x -q` — 6 passed offline.
- kernel-metadata.json parses; `enable_gpu` and `enable_internet` both true; Phase-14 id.

## Key Design Notes

Four informational divergences are documented in the oracle.py header and echoed into `bench03.divergences` (per D-01 the CatBoost-GPU column is informational, not a gate): (1) Region has no official grow_policy → N/A; (2) GPU `border_count` default 128 set to 32; (3) CatBoost `fit()` wall-clock includes on-device quantization while catboost-rs times only the grow loop — NOT subtracted; (4) integer-binned columns fed as float32 with `border_count=32`. The gen() numpy reproduction is exact because 32 divides 2^64, so Rust's 64-bit wrapping mod-then-mod-32 equals the plain `% 32` used in the spot-check assertions.

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs

None. Part C leaves `catboost_gpu_s=None` only on Region (N/A by design) or on a real fit failure — never a fabricated placeholder value. This is the do-not-fabricate discipline (D-04), not a stub.

## Scope Boundary (per D-04)

This plan writes the SCRIPT only. It does NOT flip GPUT-14 to satisfied, does NOT write `bench/RESULTS.md`, and does NOT run the notebook — the actual human-gated Kaggle CUDA run is Plan 14-03.

## Self-Check: PASSED

- FOUND: bench/phase14_cuda_signoff/oracle.py
- FOUND: bench/phase14_cuda_signoff/kernel-metadata.json
- FOUND: bench/phase14_cuda_signoff/oracle_gen_test.py
- FOUND commit: 69dcf27
- FOUND commit: d769778
