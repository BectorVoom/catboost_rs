---
phase: 13-pairwise-ranking-multiclass-ordered-langevin-device-coverage
plan: 10
subsystem: gpu-training
tags: [kaggle-cuda, coverage-matrix, bench-02, human-gate, gput-11, gput-21, gput-22, gput-12, gput-13, gput-20]

# Dependency graph
requires:
  - phase: 13-pairwise-ranking-multiclass-ordered-langevin-device-coverage
    provides: five landed device families (Plans 01–09) — pairwise/ranking/multiclass/ordered/langevin der-drivers + solvers + self-oracles + coverage-gate seams
  - phase: 10-gpu-foundations
    provides: Kaggle CUDA oracle + speed harness (BENCH-01/02), bench/generator.py D-06 single-source workload
  - phase: 12-gpu-device-families
    provides: reusable Kaggle CUDA pipeline (kaggle CLI, PROVEN on Tesla P100) + SC-5 coverage-matrix precedent
provides:
  - Phase-13 Kaggle CUDA notebook (bench/kaggle_cuda_phase13.ipynb) — per-family correctness gate + BENCH-02 anchor
  - Phase-13 GPU coverage matrix (13-COVERAGE-MATRIX.md) — SC-5 per-family correctness/speed/Ok(None) status (PENDING-KAGGLE)
affects: [phase-14-aggregate-signoff, gput-14-standing-gate]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Per-family device self-oracle as the authoritative CUDA correctness gate (cargo test --features cuda over *_test.rs)"
    - "Ok(None) sub-operation BENCH-02 treatment: no per-family end-to-end device train loop (grow seam forward dependency); speed captured by the shared grow-loop anchor (Phase-12 precedent)"
    - "PENDING-KAGGLE anti-fabrication scaffold (T-13-19): correctness gates before any speed number; no CUDA value entered until human-reported"

key-files:
  created:
    - bench/kaggle_cuda_phase13.ipynb
    - .planning/phases/13-pairwise-ranking-multiclass-ordered-langevin-device-coverage/13-COVERAGE-MATRIX.md
  modified:
    - .planning/STATE.md
    - .planning/ROADMAP.md

key-decisions:
  - "All five families decline to Ok(None) at session begin() this phase (grow seam forward dependency), so BENCH-02 per-family is captured-by-grow-loop, NOT a fabricated per-family end-to-end speedup (Phase-12 sub-operation precedent)"
  - "Per-family correctness ε=1e-4 device-vs-Rust-CPU is the recordable Phase-13 gate; the notebook runs each family's existing device self-oracle under --features cuda (same oracles green on rocm in-env)"
  - "No CUDA numbers fabricated — every matrix cell is PENDING-KAGGLE / _PENDING_ until the human-gated run reports (T-13-19)"

metrics:
  duration: 30min
  completed: 2026-07-04
  tasks_completed: 1
  tasks_total: 2
  files_created: 2
  files_modified: 2

status: pending-human-gate
requirements-completed: []
requirements-pending-kaggle: [GPUT-11, GPUT-21, GPUT-22, GPUT-12, GPUT-13, GPUT-20]
---

# Phase 13 Plan 10: Kaggle CUDA Per-Family Coverage Sign-Off (Human-Gated) Summary

**Scaffolded the Phase-13 human-gated CUDA authority: `bench/kaggle_cuda_phase13.ipynb`
(per-family correctness gate + BENCH-02 grow-loop anchor for all five families) and
`13-COVERAGE-MATRIX.md` (SC-5 per-family correctness/speed/`Ok(None)` status), both with
honest `PENDING-KAGGLE` placeholders — Task 2 (the real Kaggle CUDA run producing ε=1e-4 +
BENCH-02 numbers) is a BLOCKING human action and is NOT complete.**

## What landed (Task 1 — autonomous)

- **`bench/kaggle_cuda_phase13.ipynb`** (11 cells, valid nbformat 4) — extends the Phase-10/12
  harness: reuses the Step-0 `--features cuda` wheel build + `nvidia-smi` gate + `generator.py`
  D-06 workload, then adds one correctness+BENCH-02 cell group covering all five families via a
  `TEST_MAP` loop:
  - **Correctness gate (BLOCKING):** runs each family's in-tree device self-oracle under
    `cargo test --no-default-features --features cuda -p cb-backend` (pairwise: `pairwise_deriv`
    + `cholesky_solve`; ranking: `query_helper` + `ranking_det` + `ranking_stoch`; multiclass:
    `multiclass` + `multi_newton`; ordered: `ordered`; langevin: `langevin`) — the device path
    vs the Rust CPU reference on the CUDA runtime, ε=1e-4. `--nocapture` surfaces each oracle's
    printed max divergence for transcription.
  - **BENCH-02:** honest `Ok(None)` treatment — no per-family end-to-end device train loop this
    phase (grow seam forward dependency), so it times the shared depth-6 device grow-loop anchor
    (warm-run/JIT-excluded, queue-drained) and records each family's status as
    `captured-by-grow-loop`, not a fabricated per-family number.
- **`13-COVERAGE-MATRIX.md`** (SC-5) — a row per family with columns: in-env self-oracle (ε,
  convenience), Kaggle CUDA correctness (`PENDING-KAGGLE`), BENCH-02 (`captured-by-grow-loop` /
  `PENDING-KAGGLE`), and authoritative gate state (`Ok(None) → CPU fallback (PENDING-KAGGLE)`).
  Includes the two verbatim result tables (correctness + BENCH-02) with `_PENDING_` cells, the
  `Ok(None)` reality section, SC coverage, and anti-fabrication footer notes.

## What is PENDING (Task 2 — BLOCKING human action, NOT complete)

The actual Kaggle CUDA notebook RUN on real NVIDIA hardware (no CUDA in-env) producing the
per-family ε=1e-4 correctness sign-off and the BENCH-02 speed measurement is a **blocking human
action**. No correctness or speed number has been recorded. See the checkpoint below.

- **GPUT-11 / GPUT-21 (pairwise)** — PENDING-KAGGLE correctness + BENCH-02
- **GPUT-22 (ranking)** — PENDING-KAGGLE correctness + BENCH-02 (QueryCrossEntropy independently `Ok(None)`, Open Q3)
- **GPUT-12 (multiclass)** — PENDING-KAGGLE correctness + BENCH-02
- **GPUT-13 (ordered)** — PENDING-KAGGLE correctness + BENCH-02
- **GPUT-20 (langevin)** — PENDING-KAGGLE correctness + BENCH-02

None of these requirements are marked complete — they remain PENDING-KAGGLE until real measured
results are transcribed into `13-COVERAGE-MATRIX.md`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Root filesystem 100% full blocked `git commit`**
- **Found during:** Task 1 commit.
- **Issue:** `/` (242G) was at 100% (156K free); the harness temp filesystem and git could not
  write. The catboost `target/` dir was 73G.
- **Fix:** Removed the regenerable Cargo **incremental** compilation cache
  (`target/*/incremental`, 1.8G) — safe, cargo regenerates it; redirected `TMPDIR`/
  `CLAUDE_CODE_TMPDIR` to `/run/user/1000` (tmpfs with room). No source or build artifact of
  record deleted. (Note per MEMORY: incremental wipe can make rust-analyzer diagnostics stale —
  trust `cargo check`, not the editor.)
- **Files modified:** none (environment only).

## Known Stubs

None — the two artifacts are intentionally scaffolds with clearly-marked `PENDING-KAGGLE` /
`_PENDING_` result cells per the plan (`Do NOT fabricate CUDA numbers`); the human-gated run
fills them. This is the plan's designed output, not an unresolved stub.

## Self-Check: PASSED

- FOUND: `bench/kaggle_cuda_phase13.ipynb` (valid nbformat 4, 11 cells)
- FOUND: `.planning/phases/13-.../13-COVERAGE-MATRIX.md`
- FOUND commit: `5d80637` (docs(13-10) scaffold)
