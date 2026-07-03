---
phase: 10-gpu-foundations-runtime-seam-session-residency-device-primit
plan: 09
subsystem: bench
tags: [gpu, cuda, bench, oracle, BENCH-01, BENCH-02, D-06, D-05, D-10-09]

# Dependency graph
requires:
  - phase: 10-08 (device grow seam wired into fit)
    provides: the depth-1 device path reachable from public fit() that this notebook oracles end-to-end
  - phase: 10-01..06 (primitive + cindex + reduce family)
    provides: the standalone primitive/cindex/reduce oracles this notebook runs as the CUDA blocking gate
provides:
  - "bench/generator.py: one seeded generator (D-06) sourcing BOTH the depth-1 <=1e-5 correctness fixture AND the ~1e6x50 speed workload"
  - "bench/fixtures/ committed small-n fixtures + serial CPU-path expected values + sha256 commit-discipline manifest"
  - "bench/cuda_oracle.ipynb: committed Kaggle CUDA notebook, correctness-gate-then-warm-run-speed with structured report"
  - "bench/RESULTS.md: human sign-off log with the D-10-09 large-n pinning escalation"
affects:
  - "Phases 11->13: reuse this notebook + RESULTS log as the standing per-phase CUDA correctness+speed discipline (BENCH-02)"
  - "SPIKE-REDUCTION.md section 4: CUDA reduce err/ms rows filled from the notebook reduce cell"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "One seeded numpy generator (legacy RandomState -> cross-version-stable bytes) is the SINGLE source (D-06) for both the small-n correctness fixture and the large-n speed workload; only small-n fixtures are committed, large-n is regenerated from seed."
    - "Correctness is a BLOCKING gate before any speed cell: a module-level GATE_PASSED flag is set only after primitive+cindex (bit-exact) and depth-1 RMSE/Logloss (<=1e-5) all pass; the speed cell asserts it (T-10-25, no fast-but-wrong number)."
    - "Warm-run/JIT-excluded train-only timing: warm one untimed fit, then time a fresh fit and force a read-back/predict to DRAIN CubeCL's lazy queue before stopping the clock."

key-files:
  created:
    - bench/generator.py
    - bench/fixtures/README.md
    - bench/fixtures/ (17 committed fixtures + manifest.json)
    - bench/cuda_oracle.ipynb
    - bench/RESULTS.md
  modified:
    - .planning/phases/10-gpu-foundations-runtime-seam-session-residency-device-primit/SPIKE-REDUCTION.md

key-decisions:
  - "D-06 single source: generate(n_rows, n_features, seed) feeds both fixtures; CORRECTNESS_CONFIG=2000x10, SPEED_CONFIG=1e6x50 (tunable above the launch-overhead break-even)."
  - "Legacy numpy.random.RandomState (Mersenne) not default_rng: the legacy stream is stable across numpy versions, so committed fixtures reproduce bit-for-bit on the Kaggle image regardless of its numpy version."
  - "Logloss depth-1 reference pinned to FIRST-ORDER calc_average leaves, NOT Newton der2 (Newton is Phase 11 / GPUT-07) -- the single most likely reason a naive oracle misses <=1e-5 (RESEARCH line 318 / CONTEXT scope anchor)."
  - "numpy reference bordering is uniform-quantile: a self-consistent anchor for the standalone cindex/primitive fixtures, NOT a bit-parity claim vs CatBoost GreedyLogSum; the AUTHORITATIVE depth-1 oracle is device-vs-Rust-CPU (both use the Rust bordering)."
  - "D-10-09 surfaced explicitly (not silently assumed): depth-1 device>=CPU is achievable only at large n; at small n it is physically infeasible; BENCH-02's depth-1 bar is pinned to the large-n workload; the Kaggle run is the arbiter of the crossover."
  - "Kaggle numbers are NOT fabricated: all measured cells are TBD placeholders filled by the human-gated run; ROCm in-env is smoke-only, not a gate."

requirements-completed: [BENCH-01, BENCH-02]

# Metrics
duration: ~6min
completed: 2026-07-03
status: complete
---

# Phase 10 Plan 09: Kaggle CUDA Oracle & Speed Harness Summary

**Established the authoritative, reproducible Kaggle CUDA harness (BENCH-01) and stood up the standing per-phase correctness+speed discipline (BENCH-02).** One seeded numpy generator (`bench/generator.py`, D-06) is the single source for BOTH the small-n depth-1 `<=1e-5` correctness fixture AND the large-n (~1e6x50) wall-clock speed workload — using the legacy `RandomState` Mersenne stream so committed fixtures reproduce bit-for-bit on the Kaggle image across numpy versions. The generator also emits the serial CPU-path expected values (scan / segmented scan / stable sort / reduce-by-key / segmented-reduce / bit-packed cindex, plus a first-order `calc_average` depth-1 Cosine-score tree for RMSE and Logloss) into `bench/fixtures/`, gated by a sha256 commit-discipline manifest (`--write`/`--check`). The committed `bench/cuda_oracle.ipynb` runs, in order: build the `--features cuda` wheel + `nvidia-smi` → BLOCKING primitive+cindex oracles (`cargo test --features cuda`, `<=1e-4`/bit-exact) → depth-1 RMSE+Logloss device-vs-CPU-reference (`<=1e-5`) → a `GATE_PASSED` flag → warm-run/JIT-excluded train-only speed on the large-n workload draining the lazy queue → structured report. The speed cell **asserts the gate passed** so no fast-but-wrong number is ever emitted (T-10-25). `bench/RESULTS.md` is the human sign-off log with the D-10-09 escalation pinned prominently (depth-1 device`>=`CPU only at large n; small-n infeasibility surfaced, not assumed), and `SPIKE-REDUCTION.md` §4 is wired to the notebook's reduce cell as the CUDA err/ms fill source. No Kaggle numbers are fabricated — the run is human-gated.

## Performance
- **Duration:** ~6 min
- **Completed:** 2026-07-03
- **Tasks:** 2
- **Files:** 6 (5 created incl. fixtures dir, 1 modified)

## Accomplishments

- **Seeded generator + committed fixtures (Task 1, D-06/D-05).** `bench/generator.py` parameterizes `n_rows`/`n_features`/`seed`; `generate()` mirrors `benchmark.py`'s seeded `randn` + linear-plus-noise target. `write_fixtures()` emits 17 committed small-n fixtures (correctness inputs + serial expected values + a bit-packed cindex reference + a first-order depth-1 tree reference for RMSE and Logloss) and a sha256 `manifest.json`; `--check` regenerates into a temp dir and diffs shas (drift = fail). Determinism proven in-env: two `--write` runs are byte-identical (`--check OK — 17 fixtures reproduce bit-for-bit`).
- **cuda_oracle.ipynb correctness-gate-then-speed (Task 2, BENCH-01).** 13 ordered cells; the notebook imports `bench/generator.py` (key_link, D-06) for both the correctness fixture and the large-n speed workload. Primitive/cindex oracles run as `cargo test --features cuda -- --nocapture` (BLOCKING, raises on non-zero return); depth-1 RMSE/Logloss compare device `fit()`/`predict` against the committed `expected_depth1_tree.json` at `<=1e-5` and `assert`-halt on miss; `GATE_PASSED` unlocks the warm-run/queue-drained train-only speed cell (device vs host-CPU-wheel vs official CatBoost `task_type='GPU'`); a structured-report cell prints correctness-first-then-speed rows.
- **RESULTS.md sign-off log + D-10-09 escalation (Task 2, BENCH-02).** `bench/RESULTS.md` records the escalation prominently ("depth-1 device`>=`CPU only at large n; at small n physically infeasible regardless of optimization — bar pinned to the large-n workload; Kaggle run is the arbiter") and carries a per-run template (correctness gate table then large-n speed table) that every later phase (11→13) appends to. ROCm in-env is documented as smoke-only, not a gate.
- **SPIKE-REDUCTION §4 wired.** The notebook's `--nocapture` reduce cell + a dedicated "Fill SPIKE-REDUCTION.md §4" markdown cell instruct the human to transcribe the CUDA err/ms for reduce candidates (a)/(b)/(c); §4 now names the notebook + RESULTS.md as the authoritative fill source (still TBD — awaiting Kaggle, not fabricated).

## Task Commits
1. **Task 1: seeded generator + committed fixtures** — `aa6ed6e` (feat)
2. **Task 2: cuda_oracle.ipynb + RESULTS.md + SPIKE-REDUCTION wiring** — `f8b30f5` (feat)

## Deviations from Plan
None functional — the plan executed as written. Scope-clarifying notes (documented above as key-decisions, not correctness deviations):

1. **Primitive/cindex oracles run as Rust `cargo test --features cuda`, not Python.** The scan/sort/reduce/cindex kernels live in `cb-backend` and are not exposed through the Python wheel (which exposes Regressor/Classifier/Ranker). The notebook therefore runs them as the in-tree device tests (the same oracles that run on ROCm in-env, now on CUDA) as the blocking gate, and uses the Python wheel only for the depth-1 fit + speed. This is the faithful realization of "standalone primitive oracles vs their serial references" given the compile-time feature boundary.
2. **Host-CPU speed baseline requires a separate cpu-feature wheel.** Cargo features are compile-time (CLAUDE.md — no runtime switching), so a single wheel cannot hold both the cuda device path and the cpu path. The notebook builds/installs the cuda wheel for the device number and documents that the host-CPU baseline comes from a cpu-feature wheel run pasted into RESULTS.md. Surfaced explicitly, not silently assumed.

## Known Stubs
None in the code-flow sense. The `TBD` cells in `bench/RESULTS.md` and `SPIKE-REDUCTION.md` §4 and the `cpu_s = None` / measured-value placeholders in the notebook are **intentional human-fill placeholders** for the human-gated Kaggle CUDA run — the plan explicitly forbids fabricating them ("the run is human-gated ... leave measured-value cells as clearly-marked TBD/placeholder"). They are wired and reproducible; only the external measurement is pending.

## Threat Flags
None beyond the plan's `<threat_model>`. T-10-25 (fast-but-wrong number) mitigated: correctness is a BLOCKING `GATE_PASSED` gate that the speed cell asserts; warm untimed fit + lazy-queue drain before timing. T-10-26 (misleading small-n speed claim) mitigated: D-10-09 escalation recorded prominently in RESULTS.md, depth-1 bar pinned to the large-n workload, no fabricated Kaggle numbers. T-10-SC (installs) accept: only already-present audited packages (maturin/numpy/catboost); no new external package.

## Next Phase Readiness
- Phases 11→13 reuse `bench/cuda_oracle.ipynb` + `bench/RESULTS.md` as the standing per-phase CUDA correctness+speed discipline (BENCH-02) — each phase appends a dated run block.
- The authoritative BENCH-01/02 sign-off (primitive+cindex bit-exact, depth-1 RMSE/Logloss `<=1e-5`, then large-n device-vs-CPU speed) remains the human-gated Kaggle CUDA run; ROCm in-env is smoke-only.
- SPIKE-REDUCTION §4 CUDA reduce err/ms rows are ready to be filled from the same run.

## Self-Check: PASSED
- Files: `bench/generator.py`, `bench/fixtures/README.md`, `bench/fixtures/manifest.json`, `bench/cuda_oracle.ipynb`, `bench/RESULTS.md`, `SPIKE-REDUCTION.md` — all FOUND.
- Commits: `aa6ed6e` (feat 10-09 generator+fixtures), `f8b30f5` (feat 10-09 notebook+RESULTS) — both FOUND in git log.
- Verifies: `generator.py` parses + deterministic (`--check OK, 17 fixtures reproduce bit-for-bit`); notebook JSON valid (13 cells) with generator import (key_link); `RESULTS.md` contains "large n" + D-10-09.

---
*Phase: 10-gpu-foundations-runtime-seam-session-residency-device-primit*
*Completed: 2026-07-03*
