---
phase: 01-workspace-lint-discipline-oracle-harness
plan: 03
subsystem: infra
tags: [cargo-workspace, feature-gates, oracle-generator, catboost, fixtures, per-stage-comparator, source-test-separation, ci, rust, python]

# Dependency graph
requires:
  - phase: 01-01
    provides: "cb-core/cb-oracle scaffolds, workspace clippy deny table + [workspace.dependencies], assert_abs_close + Stage enum, load_f64_vec/load_config, OracleError, ci.yml CPU lane, check-no-anyhow.sh"
  - phase: 01-02
    provides: "cb-core::TFastRng64 (no direct dependency in this plan, but completes cb-core)"
provides:
  - "Six stub crates completing the day-one 8-crate workspace: cb-data, cb-compute (cubecl-free, D-03), cb-backend (sole feature-gated runtime-alias owner cpu/wgpu/cuda/rocm, D-02), cb-train, cb-model, catboost-rs (published Builder facade, D-04)"
  - "cb-backend::SelectedRuntime — compile-time-selected inert () placeholder, one cfg arm per backend, zero runtime dispatch"
  - "Python-first build-time oracle generator (pinned catboost==1.2.10 + numpy==1.26.4, thread_count=1, fixed seed) — gen_inputs.py + gen_fixtures.py + README"
  - "Frozen shared INPUT corpus (numeric_tiny 50x4, numeric_categorical 50x(3+2), grouped_ranking 60/12-groups) in hybrid config.json + np.float64 .npy format (D-09/D-11)"
  - "regression_skeleton per-stage expected-OUTPUT fixtures: borders.npy (33), borders_per_feature.npy, model.json (splits+leaf_values), staged.npy (500), predictions.npy (50)"
  - "cb-oracle::compare_stage(Stage, expected, actual) — per-stage 1e-5 comparator API tagging StageDiverged/StageLengthMismatch (INFRA-04)"
  - "Falsifiable boundary-gate proof: unit tests (9e-6 -> Ok, 2e-5 -> Err) + integration test perturbing REAL committed catboost==1.2.10 predictions"
  - "scripts/check-source-test-separation.sh — INFRA-06/D-17 grep gate, wired into CI alongside cb-backend feature-gate build steps"
affects: [02-pool-quantization, 03-train-loop, 04-model-facade, 07-cubecl-backend, all-later-phases]

# Tech tracking
tech-stack:
  added: ["catboost==1.2.10 (build-time generator only)", "numpy==1.26.4 (build-time generator only)"]
  patterns: [feature-gated-compile-time-runtime-alias, cubecl-free-pure-generic-compute-crate, python-first-build-time-oracle-generator-never-in-ci, hybrid-npy-plus-config-json-fixtures, per-stage-comparator-with-stage-tagged-errors, falsifiable-tolerance-gate-on-real-oracle-data, grep-based-source-test-separation-gate]

key-files:
  created:
    - crates/cb-data/Cargo.toml
    - crates/cb-data/src/lib.rs
    - crates/cb-compute/Cargo.toml
    - crates/cb-compute/src/lib.rs
    - crates/cb-backend/Cargo.toml
    - crates/cb-backend/src/lib.rs
    - crates/cb-train/Cargo.toml
    - crates/cb-train/src/lib.rs
    - crates/cb-model/Cargo.toml
    - crates/cb-model/src/lib.rs
    - crates/catboost-rs/Cargo.toml
    - crates/catboost-rs/src/lib.rs
    - crates/cb-oracle/generator/requirements.txt
    - crates/cb-oracle/generator/gen_inputs.py
    - crates/cb-oracle/generator/gen_fixtures.py
    - crates/cb-oracle/generator/README.md
    - crates/cb-oracle/fixtures/inputs/numeric_tiny/{config.json,X.npy,y.npy}
    - crates/cb-oracle/fixtures/inputs/numeric_categorical/{config.json,X.npy,cat.npy,y.npy}
    - crates/cb-oracle/fixtures/inputs/grouped_ranking/{config.json,X.npy,group_id.npy,y.npy}
    - crates/cb-oracle/fixtures/regression_skeleton/{config.json,borders.npy,borders_per_feature.npy,model.json,staged.npy,predictions.npy}
    - crates/cb-oracle/tests/per_stage_oracle_test.rs
    - scripts/check-source-test-separation.sh
  modified:
    - crates/cb-oracle/src/compare.rs
    - crates/cb-oracle/src/error.rs
    - crates/cb-oracle/src/lib.rs
    - crates/cb-oracle/src/compare_test.rs
    - .github/workflows/ci.yml
    - .gitignore
    - Cargo.lock

key-decisions:
  - "Pinned numpy==1.26.4 alongside catboost==1.2.10 (D-07): catboost 1.2.10 requires numpy<2.0; 1.26.4 is the stable in-range release and installed cleanly under Python 3.12"
  - "get_borders() returns a dict {feature_index: [borders]} in 1.2.10; flattened to a single f64 borders.npy in ascending feature-index order + a borders_per_feature.npy count array so the flat vector is splittable (layout documented in config.json + README)"
  - "staged_predict flattened stage-major (stage 0 rows, then stage 1, ...) into one f64 staged.npy; layout recorded in config.json"
  - "compare_stage adds dedicated StageDiverged/StageLengthMismatch variants (rather than reusing the untagged Diverged/LengthMismatch) so a per-stage failure names the stage that drifted; assert_abs_close stays the audited untagged primitive"
  - "catboost_info/ training-log dir emitted by CatBoostRegressor.fit is generator output, not a fixture — untracked and gitignored"

patterns-established:
  - "cb-backend is the single compile-time runtime-alias owner: one #[cfg(feature=...)] arm per cpu/wgpu/cuda/rocm defining pub type SelectedRuntime = (), guarded with not(...) so default+explicit feature combos never double-define; no runtime match"
  - "cb-compute carries NO cubecl/backend dependency (D-03); the concrete runtime is bound only in cb-backend"
  - "Python oracle generator is build-time only: pinned versions, thread_count=1 + fixed seed for determinism, np.float64 with dtype asserts, fixtures committed frozen, generator NEVER installed/run in CI (D-12)"
  - "compare_stage is the per-stage INFRA-04 API; proven by a falsifiable 1e-5 boundary gate (9e-6 Ok / 2e-5 Err) on BOTH synthetic and REAL committed oracle data — not a self-equality check"
  - "Source/test separation enforced by a grep script distinguishing the allowed 'mod x_test;' declaration from the forbidden inline 'mod tests {' body"

requirements-completed: [INFRA-01, INFRA-03, INFRA-04, INFRA-06]

# Metrics
duration: 9min
completed: 2026-06-13
---

# Phase 1 Plan 03: Stub Crates, Oracle Generator & Per-Stage Comparator Summary

**Completed the day-one 8-crate workspace (six stubs incl. cb-backend's compile-time cpu/wgpu/cuda/rocm runtime alias and a cubecl-free cb-compute), stood up a pinned Python-first oracle generator (catboost==1.2.10, thread_count=1) producing a frozen input corpus + per-stage regression_skeleton fixtures, and delivered the `compare_stage` per-stage 1e-5 comparator API proven falsifiable on REAL committed catboost oracle predictions, with source/test separation enforced in CI.**

## Performance

- **Duration:** ~9 min
- **Completed:** 2026-06-13
- **Tasks:** 3
- **Files modified:** 35+ created, 7 modified

## Accomplishments

- **INFRA-01 complete:** six stub crates (`cb-data`, `cb-compute`, `cb-backend`, `cb-train`, `cb-model`, `catboost-rs`) join Plan-01's `cb-core`/`cb-oracle` to form the full day-one 8-crate fine-grained workspace (D-01). `cb-backend` is the sole feature-gated runtime-alias owner — `cpu`/`wgpu`/`cuda`/`rocm` each compile a `pub type SelectedRuntime = ()` placeholder selected purely at compile time (D-02); `cb-compute` carries NO cubecl dependency (D-03).
- **INFRA-03 generation side:** a pinned Python-first build-time generator (`catboost==1.2.10` + `numpy==1.26.4`, `thread_count=1`, fixed seeds) synthesizes and freezes the shared input corpus (`numeric_tiny`, `numeric_categorical`, `grouped_ranking`) and the `regression_skeleton` per-stage expected-output fixtures (borders, model.json splits+leaf_values, staged, predictions) in the hybrid `config.json` + `np.float64` `.npy` format (D-09/D-11), all committed frozen and NEVER run in CI (D-12).
- **INFRA-04 comparator API:** `compare_stage(Stage, expected, actual)` wraps `assert_abs_close` at `1e-5` and tags failures with the `Stage` (new `StageDiverged`/`StageLengthMismatch` variants). Proven FALSIFIABLE in both directions by unit tests (9e-6 → Ok, 2e-5 → Err for `Predictions` AND `Borders`) and an integration test that perturbs REAL committed `catboost==1.2.10` predictions and asserts the gate fires — also proving the borders+predictions READ pipeline on real oracle data. Comparison against Rust-COMPUTED actuals is explicitly deferred to cb-train (P3) / cb-model (P4) (documented in-test).
- **INFRA-06:** `scripts/check-source-test-separation.sh` greps production `crates/*/src/**/*.rs` (excluding `*_test.rs`) for inline `#[cfg(test)] mod ... {` bodies (allowing the `mod x_test;` declaration form), proven falsifiable via an injected probe, and wired into the CI gate set alongside `cb-backend` `wgpu`/`cuda`/`rocm` feature-gate build steps. CI remains CPU-only with no generator/catboost install.

## Task Commits

1. **Task 1: Six stub crates completing the 8-crate workspace** — `66ca150` (feat)
2. **Task 2: Python oracle generator + frozen corpus + per-stage fixtures** — `f47fef8` (feat); **`ac34581`** (fix: untrack/gitignore `catboost_info/` training logs)
3. **Task 3: compare_stage API + falsifiable 1e-5 gate + CI extension** — `0f75f90` (feat)

## Decisions Made

- Pinned `numpy==1.26.4` with `catboost==1.2.10` (catboost requires numpy<2.0); installed cleanly under Python 3.12.
- `get_borders()` returns a per-feature dict in 1.2.10 → flattened to one f64 `borders.npy` (ascending feature index) + a `borders_per_feature.npy` count array; layout documented in `config.json` and README.
- `staged_predict` flattened stage-major into a single f64 `staged.npy`.
- Added dedicated `StageDiverged`/`StageLengthMismatch` variants so per-stage failures name the drifting stage, keeping `assert_abs_close` as the untagged audited primitive.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Untracked and gitignored the `catboost_info/` training-log directory**
- **Found during:** Task 2 (after running `gen_fixtures.py`, before commit verification)
- **Issue:** `CatBoostRegressor.fit` writes a `catboost_info/` directory of training TSV/event logs into the generator working dir. It was generator output (not a consumed fixture) but got staged in the Task 2 commit (`f47fef8`).
- **Fix:** `git rm --cached` the directory, deleted it from disk, and added `crates/cb-oracle/generator/catboost_info/` to `.gitignore`.
- **Files modified:** `.gitignore` (and untracked `catboost_info/`)
- **Verification:** `git status` shows `catboost_info` clean; fixtures remain tracked.
- **Committed in:** `ac34581`

---

**Total deviations:** 1 auto-fixed (1 blocking). No architectural changes; no scope creep. Crate boundaries, feature gates, fixture format, comparator API, separation gate, and CI shape are exactly as planned.

## Issues Encountered

- The `cargo build -p cb-backend --features wgpu` verify uses `--no-default-features` so only one backend feature is active; the `cb-backend` `cfg` arms additionally guard each non-cpu arm with `not(...)` of the higher-priority features so a `default + explicit feature` combination never double-defines `SelectedRuntime`. No build issues observed.

## Known Stubs

The six crates created here are **intentional stubs** per D-01/D-05 — they exist to fix the workspace shape and feature gates on day one; their real surfaces ship in later phases:
- `cb-data` (Pool/quantization) → Phase 2
- `cb-compute` (generic R/F seam) → Phase 3
- `cb-train` (boosting loop) → Phase 3
- `cb-model` (serialize/SHAP) → Phase 4
- `catboost-rs` (Builder facade) → Phase 4
- `cb-backend::SelectedRuntime` is an inert `()` placeholder; the real CubeCL runtime is wired in Phase 7.

These are documented in each crate's module doc-comment with the realizing phase. No stub blocks this plan's goal (workspace completeness + oracle harness), so all are intentional and phase-mapped.

## Threat Flags

None — no new security surface beyond the plan's threat model. Mitigations applied as planned: `thread_count=1` + fixed seed + frozen fixtures (T-01-07); explicit `np.float64` + dtype asserts in the generator, `read_npy` into `Array1<f64>` on the Rust side (T-01-08); falsifiable 9e-6/2e-5 boundary gate proving the comparator actually gates (T-01-10); pinned canonical `catboost`/`numpy`, generator never in CI (T-01-SC); source/test separation grep gate (T-01-09).

## User Setup Required

None for consumers. To **regenerate** fixtures (build-time only, never required for CI/test): `cd crates/cb-oracle/generator && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/python gen_inputs.py && .venv/bin/python gen_fixtures.py` (documented in the generator README).

## Next Phase Readiness

- Full 8-crate workspace builds; `cb-backend` compiles under each of cpu/wgpu/cuda/rocm; `cb-compute` cubecl-free.
- `cb-oracle::compare_stage` + the frozen input corpus + `regression_skeleton` fixtures are ready for Phase 2/3/4 to feed Rust-computed actuals into per-stage comparisons.
- Verified green: `cargo build --workspace`, `cargo build -p cb-backend --no-default-features --features {wgpu,cuda,rocm}`, `cargo clippy --workspace --lib -- -D warnings`, `cargo test --workspace` (cb-oracle 15 unit + 3 per-stage integration + 1 skeleton integration), `bash scripts/check-no-anyhow.sh`, `bash scripts/check-source-test-separation.sh`.

## Self-Check: PASSED

All spot-checked created files exist on disk; all four plan commits (66ca150, f47fef8, ac34581, 0f75f90) are present in git history.

---
*Phase: 01-workspace-lint-discipline-oracle-harness*
*Completed: 2026-06-13*
