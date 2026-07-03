---
phase: 11-depth-1-partition-aware-histograms-reduction-determinism-new
plan: 01
subsystem: testing
tags: [fixtures, oracle, numpy, depth-6, newton, logloss, rmse, cb-compute]

# Dependency graph
requires:
  - phase: 10-gpu-foundations-runtime-seam-session-residency-device-primit
    provides: "bench/generator.py seeded synthetic generator + depth-1 fixture pattern; cb-compute reduce_leaf_stats/reduce_leaf_der2/newton_leaf_delta/calc_average/scale_l2_reg"
provides:
  - "bench/fixtures/expected_depth6_tree.json — depth-6 RMSE + Logloss correctness fixture (6-level oblivious split sequence, 64 raw-delta leaf values per arm, per-object leaf_of/der1/weight/weighted_der2 reduction inputs, config block pinning A1/A2)"
  - "bench/fixtures/X_depth6_speed.npy — large-n (10k x 50) depth-6 speed workload from the same seed, sha-manifested, for BENCH-02"
  - "serial_depth6_tree() reference in bench/generator.py"
  - "cb-compute depth6_reference_test.rs — CPU-oracle cross-check locking A1 (iterations==1) and A2 (Cosine score, channel-0 = Σweight)"
affects: [11-02, 11-03, 11-04, 11-05, BENCH-02, device-self-oracle, kaggle-cuda-harness]

# Tech tracking
tech-stack:
  added: [serde_json (cb-compute dev-dependency only)]
  patterns:
    - "One seeded generator emits BOTH the ≤1e-5 correctness fixture and the large-n speed workload (D-03/D-06)"
    - "Fixture emits per-object reduction arrays so the Rust cross-check needs no .npy parser or X routing"
    - "A1/A2 assumptions pinned as fixture config facts AND asserted in the CPU-oracle test (drift → test failure, not silent divergence)"

key-files:
  created:
    - crates/cb-compute/tests/depth6_reference_test.rs
    - bench/fixtures/expected_depth6_tree.json
    - bench/fixtures/X_depth6_speed.npy
  modified:
    - bench/generator.py
    - bench/fixtures/manifest.json
    - crates/cb-compute/Cargo.toml

key-decisions:
  - "A1 pinned: leaf_estimation_iterations = 1 (single closed-form Newton step; no iterative walker in the CPU oracle)"
  - "A2 pinned: split score = Cosine with channel-0 = Σweight (NOT Σder2); the Logloss Newton hessian Σ(der2·weight) enters only the leaf value"
  - "leaf_values stored as RAW pre-learning-rate deltas so the Rust cross-check compares directly to calc_average / newton_leaf_delta output"
  - "Committed X_depth6_speed.npy sized at a representative-committable 10k x 50 (~2MB); the full ~1e6 x 50 speed run is regenerated on the fly from SPEED_CONFIG at BENCH-02 (never committed)"
  - "serde_json added as a dev-dependency only — cb-compute production surface stays cubecl-free / anyhow-free (D-03/D-14)"

patterns-established:
  - "Depth-N oblivious reference: per-level single-best Cosine split summed across current 2^level partitions, forward-bit routing, empty split sides permitted (l2>0 keeps avg finite)"
  - "Bare `python generator.py` writes the sibling fixtures/ dir; --write/--check remain for out-of-tree emission and the sha diff"

requirements-completed: [GPUT-05, GPUT-07]

# Metrics
duration: 20min
completed: 2026-07-03
status: complete
---

# Phase 11 Plan 01: Depth-6 correctness fixture + CPU-oracle cross-check Summary

**Depth-6 RMSE (calc_average) + Logloss (single-step newton_leaf_delta) fixtures from one seeded generator, pinning leaf_estimation_iterations=1 (A1) and Cosine/Σweight scoring (A2), proven bit-consistent with the cb-compute CPU oracle to ≤1e-5.**

## Performance

- **Duration:** ~20 min
- **Completed:** 2026-07-03
- **Tasks:** 2
- **Files modified:** 6 (2 created fixtures, 1 created test, 3 modified)

## Accomplishments
- `serial_depth6_tree` extends the Phase-10 generator to a per-level Cosine best-split oblivious reference: RMSE leaves via `calc_average`, Logloss leaves via a single closed-form `newton_leaf_delta = Σder1 / (−Σ(der2·weight) + scaled_l2)` step.
- `expected_depth6_tree.json` emitted with both arms (6-level split sequence + 64 raw-delta leaf values each), per-object reduction inputs, and a `config` block pinning A1 (`leaf_estimation_iterations == 1`) and A2 (`score_function = Cosine`, `score_channel0 = sum_weight`).
- `X_depth6_speed.npy` large-n depth-6 speed workload emitted from the same seed and sha-manifested (BENCH-02).
- `depth6_reference_test.rs` reduces the fixture with `reduce_leaf_stats` + `reduce_leaf_der2` (canonical object order, D-05) and recomputes each leaf with `calc_average` / `newton_leaf_delta`, asserting ≤1e-5 and locking A1/A2 (T-11-01-01 mitigated).

## Task Commits

1. **Task 1: Extend the synthetic generator to depth-6 RMSE + Logloss fixtures** - `d692289` (feat)
2. **Task 2: CPU oracle cross-check of the depth-6 fixture** - `735aa86` (test)

_Both tasks were TDD: Task 1's plan-verify (fixture assertions) and Task 2's test are the RED/GREEN gates; each landed once green._

## Files Created/Modified
- `bench/generator.py` - Added `serial_depth6_tree` + `_cosine_split_score` helper, DEPTH6/LEAF_ESTIMATION_ITERATIONS/DEPTH6_SCORE_FUNCTION/DEPTH6_SCORE_CHANNEL0/DEPTH6_SPEED_CONFIG constants, depth-6 emission + speed workload in `write_fixtures`, manifest metadata, bare-run default now writes the sibling `fixtures/` dir.
- `bench/fixtures/expected_depth6_tree.json` - The depth-6 correctness fixture (created).
- `bench/fixtures/X_depth6_speed.npy` - Large-n depth-6 speed workload (created).
- `bench/fixtures/manifest.json` - sha256 + config for the two new fixtures (modified).
- `crates/cb-compute/tests/depth6_reference_test.rs` - CPU-oracle cross-check (created).
- `crates/cb-compute/Cargo.toml` - serde_json dev-dependency (modified).

## Decisions Made
See `key-decisions` frontmatter. Highlights: A1/A2 pinned as fixture facts and test assertions; raw-delta leaf values for direct oracle comparison; committed speed workload sized representatively (10k x 50) with the full run regenerated on the fly at BENCH-02; serde_json confined to dev-dependencies.

## Deviations from Plan

None requiring a deviation rule. Two plan-latitude choices exercised within the plan's own guidance:

1. **X_depth6_speed.npy sizing.** The plan's `SPEED_CONFIG` (~1e6 x 50, ~200MB) is impractical to commit and the fixtures/README already mandates regenerating the large workload on the fly. Per the plan's "e.g." latitude and D-06, the committed artifact is a representative 10k x 50 (~2MB) seed workload; the full run scales via `SPEED_CONFIG` on Kaggle. Documented in the generator + manifest.
2. **Bare `python generator.py` default.** The plan's verify invokes `python generator.py` (no args) then opens `fixtures/expected_depth6_tree.json`. The prior no-arg path only ran a determinism smoke, so the no-arg default now also writes the sibling `fixtures/` dir (determinism assertion preserved; `--write`/`--check` unchanged).

## Issues Encountered
None. The per-partition Cosine scorer permits empty split sides (the `l2>0` denominator keeps `avg = 0/(0+l2)` finite), so no depth-6 level degenerates into a "no valid split" error despite 64 leaves over 2000 objects.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Wave-0 test foundation complete: Plans 11-02/03/04 device self-oracles and the 11-05 Kaggle harness now have an unambiguous ≤1e-5 target with A1/A2 pinned.
- A2 caveat for downstream: this fixture pins split-score channel-0 = Σweight (matching the depth-1 reference and CPU oracle). RESEARCH flagged a possible der2-in-score device variant; if a device plan adopts it, the pinned assertion in `depth6_reference_test.rs` will flag the divergence for a deliberate re-pin.

---
*Phase: 11-depth-1-partition-aware-histograms-reduction-determinism-new*
*Completed: 2026-07-03*

## Self-Check: PASSED
- All created files present: generator.py, expected_depth6_tree.json, X_depth6_speed.npy, depth6_reference_test.rs
- Both task commits present: d692289, 735aa86
