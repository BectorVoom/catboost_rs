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
  duration: 90min
  completed: 2026-07-04
  tasks_completed: 2
  tasks_total: 2
  files_created: 2
  files_modified: 4

status: complete
requirements-completed: [GPUT-11, GPUT-21, GPUT-22, GPUT-12, GPUT-13, GPUT-20]
kaggle-kernel: yensen2/catboost-rs-phase13-cuda-oracle
kaggle-src-dataset: yensen2/catboost-rs-phase13-src
gpu-box: Tesla P100-PCIE-16GB, driver 580.159.04, nvidia-smi CUDA 13.0, nvcc 12.8, 16384 MiB
---

# Phase 13 Plan 10: Kaggle CUDA Per-Family Coverage Sign-Off Summary

**COMPLETE. Scaffolded the Phase-13 CUDA authority (`bench/kaggle_cuda_phase13.ipynb` +
`13-COVERAGE-MATRIX.md`), then executed the sign-off on a real Tesla P100 by driving the
`kaggle` CLI (coordinator-authorized): kernel `yensen2/catboost-rs-phase13-cuda-oracle` ran all
five families' device self-oracles under `--features cuda` (correctness_verdict ALL-PASS) plus
the BENCH-02 grow loop (bench_verdict OK, 23.9×–36.6× device≫CPU). All six requirements
(GPUT-11/21/22/12/13/20) now have real CUDA ε≤1e-4 sign-off; every matrix cell is transcribed
verbatim from the run artifacts — no fabricated numbers.**

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

## What landed (Task 2 — Kaggle CUDA sign-off, executed via the `kaggle` CLI)

Coordinator authorized me to drive the `kaggle` CLI (account `yensen2`, already authenticated).
Reusing the PROVEN Phase-12 pipeline (script kernel + source dataset, NOT repo-clone):

- Built a lean **1.8M source tarball** via `git archive HEAD Cargo.toml Cargo.lock
  rust-toolchain.toml crates` (git-tracked source only — excludes the 73G `target/` + untracked
  venv bloat) → Kaggle dataset **`yensen2/catboost-rs-phase13-src`**.
- Wrote a single combined script kernel **`yensen2/catboost-rs-phase13-cuda-oracle`** (GPU +
  internet, one cold Rust build in **12m 49s**, `--release --no-default-features --features cuda`)
  that runs Part A (per-family correctness self-oracles) then Part B (BENCH-02 grow loop),
  pushed via `kaggle kernels push`, polled to completion, retrieved via `kaggle kernels output`.
- **Box:** Tesla P100-PCIE-16GB, driver 580.159.04, nvidia-smi CUDA 13.0, nvcc release 12.8
  (`/usr/local/cuda-12.8`), 16384 MiB.

**Correctness (device vs Rust CPU on CUDA, ε=1e-4) — correctness_verdict: ALL-PASS:**

| Family | Req | Tests | Max divergence |
|--------|-----|-------|----------------|
| Pairwise | GPUT-11, GPUT-21 | 8/8 pass | packed linear-system + batched f64 Cholesky, all asserts ≤1e-4 |
| Ranking | GPUT-22 | 14/14 pass | pfound_f der2 **0.000e0**, yetirank der2 **0.000e0** |
| Multiclass | GPUT-12 | 9/9 pass | K-dim Newton block leaves ≤1e-4 |
| Ordered | GPUT-13 | 10/10 pass | scan + partition_update **abs/rel_div 0.000e0** (bound 1e-9) |
| Langevin | GPUT-20 | 3/3 pass | seed42 **1.110e-16**, seed2024 **2.220e-16**, draw-count **4.441e-16** |

**BENCH-02 grow loop (bench_verdict: OK, dev/cpu trees = 20/20), device_s / cpu_s / speedup:**

| family | n=10k | n=100k | n=300k |
|--------|-------|--------|--------|
| depthwise | 0.1080 / 2.6645 / **24.66×** | 0.9167 / 30.3894 / **33.15×** | 2.9717 / 101.5605 / **34.18×** |
| region | 0.1310 / 3.1296 / **23.89×** | 0.9867 / 36.1485 / **36.64×** | 3.2888 / 111.6311 / **33.94×** |

Range **23.9×–36.6×** device≫CPU (grows with n), consistent with the Phase-12 P100 anchor
(30–42×). Per-family sessions remain `Ok(None)` (per-tree grow seam is a forward dependency), so
this shared grow-loop is the honest BENCH-02 anchor, not a per-family end-to-end train time — the
per-family standalone loops aggregate in Phase 14. All six requirements
(GPUT-11/21/22/12/13/20) are marked complete in REQUIREMENTS.md on the strength of this real
CUDA sign-off.

## No-regression check (D-04 / GPUT-14)

`cargo test -p cb-compute` → **all pass**. `cargo test -p cb-train --no-fail-fast` → green except
**one stale Phase-12 test** (`monotone_non_symmetric_and_region_are_typed_errors`, Region arm):
it asserts `grow_policy=Region` on CPU is a typed error ("Region OUT", D-6.6-04), but **Phase 12
built the CPU Region path** so Region now trains (confirmed: `region_e2e_test` 2/2 pass). This is
NOT a Phase-13 regression and NOT caused by 13-10 (docs/notebook only) — logged as **DI-13-01**
in `deferred-items.md` for a Phase-12 monotone-test update. Phase-13 families all decline to
`Ok(None)` → the CPU numeric path is byte-unchanged (D-04 intent satisfied).

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

**2. [Authorized scope change] Task 2 executed by agent, not human**
- The plan gated Task 2 as a BLOCKING human action (no CUDA in-env). The coordinator explicitly
  authorized me to drive the pre-authenticated `kaggle` CLI myself (outward action + GPU quota
  granted). I ran the sign-off end-to-end on real hardware and transcribed only real measured
  numbers — the anti-fabrication invariant (T-13-19) held.

**3. [Pipeline choice] Reused Phase-12 script-kernel + source-dataset pattern**
- The scaffolded `bench/kaggle_cuda_phase13.ipynb` assumed a repo-clone notebook. For reliability
  I instead reused the PROVEN Phase-12 pattern (Python script kernel + `git archive` source
  dataset), combining correctness + BENCH-02 into ONE kernel to share a single Rust build. The
  notebook remains in-repo as the documented harness; the executed kernel
  (`yensen2/catboost-rs-phase13-cuda-oracle`) is its script-form equivalent.

## Deferred Issues

- **DI-13-01** — stale Phase-12 test `monotone_non_symmetric_and_region_are_typed_errors`
  (Region arm). Out of 13-10 scope; see `deferred-items.md` + the no-regression section above.

## Known Stubs

None — the two artifacts are intentionally scaffolds with clearly-marked `PENDING-KAGGLE` /
`_PENDING_` result cells per the plan (`Do NOT fabricate CUDA numbers`); the human-gated run
fills them. This is the plan's designed output, not an unresolved stub.

## Self-Check: PASSED

- FOUND: `bench/kaggle_cuda_phase13.ipynb` (valid nbformat 4, 11 cells)
- FOUND: `.planning/phases/13-.../13-COVERAGE-MATRIX.md` (tables filled from real P100 run)
- FOUND: Kaggle kernel `yensen2/catboost-rs-phase13-cuda-oracle` — status COMPLETE, ALL-PASS
- FOUND: run artifacts (`result.json` / `result.md` / kernel log) — verdicts ALL-PASS / OK
- FOUND commit: `5d80637` (docs(13-10) scaffold)
- Correctness ALL-PASS + BENCH-02 OK transcribed verbatim; GPUT-11/21/22/12/13/20 marked complete
- No-regression: cb-compute all pass; cb-train green except stale DI-13-01 (documented, out of scope)
