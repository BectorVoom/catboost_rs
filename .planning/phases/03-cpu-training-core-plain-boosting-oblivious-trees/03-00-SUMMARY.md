---
phase: 03-cpu-training-core-plain-boosting-oblivious-trees
plan: 00
subsystem: infra
tags: [cubecl, bytemuck, gradient-boosting, oracle, model-json, catboost, generics-float, cpu-runtime]

# Dependency graph
requires:
  - phase: 01-workspace-lint-discipline-oracle-harness
    provides: cb-backend SelectedRuntime placeholder, cb-oracle fixture/compare harness, deny-lints, cb-core::sum_f64
  - phase: 02-data-layer-pool-quantization-reduction
    provides: numeric_tiny frozen input corpus, regression_skeleton fixture scaffold, ordered-reduction invariant
provides:
  - "CubeCL CPU compute seam (cubecl 0.10.0 + bytemuck) wired into cb-backend only (D-03)"
  - "First #[cube] gradient_kernel<F: Float> running on CpuRuntime (RESEARCH Open Q2 closed)"
  - "cb-oracle::model_json parser: load_model_json/ModelJson/ObliviousTree/SplitJson + split_borders/leaf_values/bias extractors"
  - "Frozen RMSE (regression_skeleton) + Logloss (binclf_skeleton) training-oracle fixtures with simplified isolating params (D-07)"
  - "Wave-0 Nyquist sign-off in 03-VALIDATION.md (nyquist_compliant: true, wave_0_complete: true)"
affects: [cb-train, cb-compute, cb-backend, "Phase 3 Plan 01 slice_first_oracle", "Phase 7 GPU backends"]

# Tech tracking
tech-stack:
  added: ["cubecl 0.10.0 (features=[cpu])", "bytemuck 1.x (features=[extern_crate_std])"]
  patterns:
    - "Kernels do order-independent elementwise work ONLY; parity-critical sums host-side via cb-core::sum_f64 (D-02/D-05/D-06)"
    - "cubecl lives in cb-backend ONLY; cb-compute stays generic/cubecl-free (D-03)"
    - "model.json parsed via serde Deserialize + fallible OracleError loader (mirrors fixture.rs), no unwrap in production"
    - "Simplified isolating params (bootstrap_type=No, random_strength=0, score_function=L2, leaf_estimation_iterations=1) for first-slice oracle (D-07)"

key-files:
  created:
    - crates/cb-backend/src/kernels.rs
    - crates/cb-backend/src/kernels/gradient.rs
    - crates/cb-oracle/src/model_json.rs
    - crates/cb-oracle/src/model_json_test.rs
    - crates/cb-oracle/fixtures/binclf_skeleton/{model.json,staged.npy,predictions.npy,config.json}
  modified:
    - Cargo.toml
    - crates/cb-backend/Cargo.toml
    - crates/cb-backend/src/lib.rs
    - crates/cb-oracle/src/lib.rs
    - crates/cb-oracle/src/error.rs
    - crates/cb-oracle/generator/gen_fixtures.py
    - crates/cb-oracle/fixtures/regression_skeleton/*
    - .planning/phases/03-cpu-training-core-plain-boosting-oblivious-trees/03-VALIDATION.md

key-decisions:
  - "CubeCL CpuRuntime stood up now (D-01): SelectedRuntime = cubecl::cpu::CpuRuntime under the cpu feature"
  - "Test module mounted at path kernels::gradient (kernels.rs + kernels/gradient.rs) so `cargo test kernels::gradient` selects the spike, preserving source/test separation"
  - "score_function=L2 resolved for Open Q1 (simplest first-slice split math)"
  - "binclf_skeleton labels derived deterministically as y>median(numeric_tiny.y); Logloss staged stored as RawFormulaVal raw logits (A5/Pitfall 6)"
  - "cubecl 0.10.0 launch API: ArrayArg::from_raw_parts(Handle, len) (2 args, no turbofish); read_one(Handle)->Result<Bytes>"

patterns-established:
  - "Pattern 1: #[cube(launch)] generics-float kernels in cb-backend, host-side ordered reductions elsewhere"
  - "Pattern 2: oracle model.json parser exporting Vec<f64> extractors for compare_stage(Stage::Splits|LeafValues)"

requirements-completed: []

# Metrics
duration: ~75min
completed: 2026-06-13
---

# Phase 3 Plan 00: CubeCL Seam + model.json Parser + Training-Oracle Fixtures Summary

**Stood up the CubeCL `CpuRuntime` with a generics-float `#[cube]` gradient kernel, added a `cb-oracle::model_json` splits/leaves/bias parser, and froze simplified-isolating-param RMSE + Logloss training oracles — flipping the Wave-0 Nyquist gate to signed-off.**

## Performance

- **Duration:** ~75 min
- **Started:** 2026-06-13T16:05Z (approx)
- **Completed:** 2026-06-13T07:21:47Z
- **Tasks:** 4
- **Files modified/created:** 16

## Accomplishments

- Wired `cubecl 0.10.0` (features=[cpu]) + `bytemuck` into the workspace and `cb-backend` ONLY (D-03), replacing the `cpu` arm placeholder with the real `cubecl::cpu::CpuRuntime` (D-01).
- Proved the CubeCL seam (RESEARCH Open Q2 closed): `gradient_kernel<F: Float>` (RMSE der1 = `target − approx`, order-independent elementwise, no reduction per D-02) compiles under deny-lints and runs on `CpuRuntime`, with f32 + f64 per-element outputs matching a host reference.
- Added `cb-oracle::model_json` — `load_model_json` + `ModelJson`/`ObliviousTree`/`SplitJson` Deserialize structs and `split_borders()`/`leaf_values()`/`bias()` extractors returning `Vec<f64>` for `compare_stage(Stage::Splits|LeafValues, …)`, with `OracleError::MalformedModel` on bad shape (no `unwrap` in production).
- Extended `gen_fixtures.py` with shared `ISOLATING_PARAMS` (D-07/A1/A2/A4) and a new `binclf_skeleton` (Logloss) scenario mirroring the now-simplified `regression_skeleton` (RMSE); both frozen with `bootstrap_type=No`, `random_strength=0`, `l2_leaf_reg=3.0`, `depth=2`, `lr=0.1`, `iterations=5`, `leaf_estimation_iterations=1`, `score_function=L2`, `leaf_estimation_method=Gradient`, pinned seed, `thread_count=1`, explicit `boost_from_average` (True RMSE / False Logloss).
- Recorded the Wave-0 Nyquist sign-off in `03-VALIDATION.md` (`nyquist_compliant: true`, `wave_0_complete: true`, sign-off boxes checked).

## Task Commits

Each task was committed atomically:

1. **Task 1: Install CubeCL + bytemuck and prove the #[cube] CpuRuntime seam** — `1070675` (feat)
2. **Task 2: Add model.json parser to cb-oracle** — `00bd2a4` (feat)
3. **Task 3: Extend the fixture generator to emit RMSE + Logloss training oracles** — `63d57b3` (feat)
4. **Task 4: Record the Wave-0 Nyquist sign-off in 03-VALIDATION.md** — `dd2fac5` (docs)

## Files Created/Modified

- `Cargo.toml` — added `cubecl` + `bytemuck` to `[workspace.dependencies]`.
- `crates/cb-backend/Cargo.toml` — `cubecl.workspace = true` + `bytemuck.workspace = true` (D-03 scope).
- `crates/cb-backend/src/lib.rs` — `SelectedRuntime = cubecl::cpu::CpuRuntime`; `pub mod kernels`.
- `crates/cb-backend/src/kernels.rs` — `#[cube(launch)] gradient_kernel<F: Float>` (production module; declares `#[cfg(test)] mod gradient;`).
- `crates/cb-backend/src/kernels/gradient.rs` — CpuRuntime launch + bytemuck transfer + per-element host-reference asserts (f32/f64).
- `crates/cb-oracle/src/model_json.rs` — parser, structs, extractors, `bias()`.
- `crates/cb-oracle/src/model_json_test.rs` — parses `regression_skeleton/model.json`, asserts the `2^depth` leaf invariant, extractor lengths, finite bias.
- `crates/cb-oracle/src/error.rs` — new `MalformedModel` variant.
- `crates/cb-oracle/src/lib.rs` — module registration + `pub use` exports.
- `crates/cb-oracle/generator/gen_fixtures.py` — `ISOLATING_PARAMS`, `gen_binclf_skeleton()`, run-once-commit header, RMSE regenerated with simplified params.
- `crates/cb-oracle/fixtures/regression_skeleton/*` — regenerated (depth=2, 5 trees, simplified params).
- `crates/cb-oracle/fixtures/binclf_skeleton/{model.json,staged.npy,predictions.npy,config.json}` — new Logloss training oracle.
- `.planning/.../03-VALIDATION.md` — Nyquist sign-off.

## Decisions Made

- **CubeCL CpuRuntime now (D-01):** `SelectedRuntime` aliases `cubecl::cpu::CpuRuntime` under `cpu`; GPU arms stay `()` until Phase 7.
- **Test path `kernels::gradient`:** kept the plan-pinned `kernels.rs` production file (artifact grep for `#[cube`) and placed the spike in `kernels/gradient.rs`, declared from `kernels.rs` via a single `#[cfg(test)] mod gradient;` line (permitted multi-file structure; no embedded test body). This makes the canonical filter `cargo test kernels::gradient` select the two spike tests.
- **cubecl 0.10.0 launch API discovered empirically:** `ArrayArg::from_raw_parts(handle, len)` takes the `Handle` by value with no turbofish; `read_one(handle) -> Result<Bytes, ServerError>`. The output `Handle` is cloned for the launch arg so the original remains readable.
- **score_function=L2 (Open Q1 RESOLVED):** simplest first-slice split-score math.
- **binclf labels:** `y > median(numeric_tiny.y)` (deterministic, reuses the frozen feature matrix — no new input corpus); staged stored as `RawFormulaVal` logits (A5/Pitfall 6).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] cubecl 0.10.0 host-launch API signature mismatch**
- **Found during:** Task 1 (kernel spike test build)
- **Issue:** Initial test used the manual's `ArrayArg::from_raw_parts::<F>(&handle, n, 1)` (3 args + turbofish + by-ref) and `client.read_one(handle.binding())`. cubecl 0.10.0's actual signatures are `from_raw_parts(handle: Handle, length: usize)` (2 args, by value, no turbofish) and `read_one(handle: Handle) -> Result<Bytes, ServerError>`.
- **Resolution:** Per AGENTS.md, consulted the CubeCL error guideline (`cubecl_error_solution_guide/mismatched types.md`) first; it covers in-kernel `#[cube]` macro errors, but my kernel compiled cleanly — the errors were host-side API drift. Verified the exact 0.10.0 signatures by reading the installed crate sources (`cubecl-core`/`cubecl-runtime` 0.10.0), then aligned the call: pass `Handle` by value, drop the turbofish, clone the output handle for the launch arg, and `unwrap()` the `Result<Bytes>` (test-only, lints exempted).
- **Files modified:** crates/cb-backend/src/kernels/gradient.rs
- **Verification:** `cargo test -p cb-backend kernels::gradient` → 2 passed (f32 + f64, per-element match).
- **Committed in:** 1070675 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking — version-pinned API alignment).
**Impact on plan:** Necessary to compile/run the CubeCL spike on the installed cubecl version. No scope creep; the kernel logic and all other plan instructions were followed exactly.

## Issues Encountered

- **Test-module path vs. source/test separation:** getting the test to live in a dedicated file AND be selectable by the canonical filter `kernels::gradient` required several layout iterations (`include!` + `#[path]` traversal through a non-existent `kernels/` dir failed). Resolved cleanly with the idiomatic `kernels.rs` + `kernels/gradient.rs` module pair, declared via a single `#[cfg(test)] mod gradient;` line (the permitted explicit-separate-module structure — no embedded `mod tests {…}` body in production).

## User Setup Required

None — no external service configuration required. (`cubecl`/`bytemuck` are vetted per RESEARCH § Package Legitimacy Audit — Tracel-AI/Burn ecosystem; no blocking human checkpoint, T-03-00-SC.)

## Next Phase Readiness

- All Wave-1 test assets are in place: the compiling+running CubeCL seam, the `model.json` parser, and the frozen `regression_skeleton`/`binclf_skeleton` training oracles.
- The Nyquist Wave-0 gate is signed off, unblocking **Plan 01 (`slice_first_oracle`)** which gates TRAIN-01/02/03.
- No open blockers from this plan. Later-slice oracle scenarios (leaf_methods/bootstrap/regularization/overfit/eval_metrics/autolr) remain owned by their own Wave-1+ slices.

## Self-Check: PASSED

All claimed files exist on disk (kernels.rs, kernels/gradient.rs, model_json.rs, model_json_test.rs, binclf_skeleton/{model.json,staged.npy}, regression_skeleton/staged.npy) and all four task commits (1070675, 00bd2a4, 63d57b3, dd2fac5) are present in git history.

---
*Phase: 03-cpu-training-core-plain-boosting-oblivious-trees*
*Completed: 2026-06-13*
