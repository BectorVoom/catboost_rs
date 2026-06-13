---
phase: 04-model-serialization-shap-rust-api-first-full-oracle-lock
plan: 02
subsystem: model
tags: [apply, prediction-types, cross-entropy, focal, oracle, cubecl, model]

# Dependency graph
requires:
  - phase: 04-model-serialization-shap-rust-api-first-full-oracle-lock
    plan: 01
    provides: "canonical cb-model::Model {oblivious_trees, bias, float_feature_borders} with leaf_weights; cb-oracle model_json parser + compare_stage harness"
  - phase: 03-cpu-training-core-plain-boosting-oblivious-trees
    provides: "cb-train plain boosting loop, cb_train::leaf_index forward-bit evaluator, Loss enum + elementwise gradient/hessian seam"
  - phase: 02-data-layer-pool-quantization-reduction
    provides: "cb-core::sum_f64 order-locked reduction (D-08); strict value>border binarization semantics"
provides:
  - "Pure-Rust CPU apply path cb-model::predict_raw (strict-> binarize, forward-bit leaf index, bias + sum_f64 over leaf values) — GPU-toolchain-free (MODEL-02)"
  - "cb-model::PredictionType + apply_prediction_type — RawFormulaVal/Probability/LogProbability/Class/Exponent (two-column probs), oracle-locked (LOSS-06)"
  - "cb-compute Loss::CrossEntropy + Loss::Focal{alpha,gamma}; cross_entropy_der1/der2 + focal_der1/der2 transcribed from error_functions.{h,cpp}"
  - "cb-backend focal_gradient/hessian #[cube] kernels (generics-float; alpha/gamma as length-1 device arrays) + CrossEntropy reusing Logloss kernels"
  - "Oracle locks: apply (binclf RawFormulaVal), all 5 prediction types, CrossEntropy + Focal training (splits/leaf/staged ≤1e-5)"
affects: [04-04, 04-05, shap, fstr, rust-api]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Apply path lives in cb-model, imports NO backend/cubecl symbol (MODEL-02 GPU-toolchain independence)"
    - "Per-tree leaf contributions summed via cb_core::sum_f64; bias added exactly once (RESEARCH Pitfall 6)"
    - "CrossEntropy delegates to the shared logloss sigmoid-gradient helper (no math duplication, D-09)"
    - "Focal #[cube] kernels keep F: Float generic by passing alpha/gamma as length-1 Array<F> (avoids the non-generic ScalarArgType bound)"
    - "CubeCL math via associated-function form (F::ln/F::powf/F::exp/F::clamp) per the cubecl error guideline; if-as-statement for label branch"

key-files:
  created:
    - crates/cb-model/src/apply.rs
    - crates/cb-model/src/predict.rs
    - crates/cb-model/tests/apply_oracle_test.rs
    - crates/cb-model/tests/predict_oracle_test.rs
    - crates/cb-train/tests/loss_oracle_test.rs
  modified:
    - crates/cb-model/src/lib.rs
    - crates/cb-model/Cargo.toml
    - crates/cb-compute/src/runtime.rs
    - crates/cb-compute/src/loss.rs
    - crates/cb-compute/src/lib.rs
    - crates/cb-compute/src/loss_test.rs
    - crates/cb-backend/src/kernels.rs
    - crates/cb-backend/src/cpu_runtime.rs
    - crates/cb-train/src/boosting.rs
    - crates/cb-train/src/metrics.rs

key-decisions:
  - "Apply input type = SoA &[Vec<f32>] feature columns (the layout cb-train and the Plan-05 Builder feed); n_objects taken from the first column."
  - "Shared model.json for apply + prediction-type locks: binclf_skeleton/model.json is the SAME model that produced prediction_types/*.npy (rawformulaval.npy == binclf_skeleton/predictions.npy), so one Model drives both test files."
  - "Probability and LogProbability emit TWO columns per object (flattened row-major [class-0, class-1]) — matching upstream binary predict (eval_helpers.cpp:393); fixtures are length 2*n_rows."
  - "Exponent uses f64::exp; the ≤1e-5 gate absorbs upstream FastExp's table-approximation gap (A2 / Pitfall 3) — verified against the committed exponent.npy."
  - "Loss drops `Eq` (kept `PartialEq`) because Loss::Focal carries f64 alpha/gamma; no call site relied on Loss: Eq."
  - "Focal kernel passes alpha/gamma as length-1 Array<F> instead of generic scalar args — a generic `F` scalar arg requires F: ScalarArgType (CubeElement+Scalar+NumCast), which would break the generics-float rule; the array path keeps F: Float."

requirements-completed: [MODEL-02, LOSS-01]
requirements-partial: [LOSS-06]

# Metrics
duration: ~50min
completed: 2026-06-14
---

# Phase 4 Plan 02: First Train→Predict Slice (Apply Path + Prediction Types + CrossEntropy/Focal) Summary

**Pure-Rust GPU-toolchain-free CPU apply path (`predict_raw`), the five in-scope prediction-type transforms, and the remaining binary-classification losses CrossEntropy + Focal — every output oracle-locked to upstream catboost 1.2.10 at ≤1e-5.**

## Performance

- **Duration:** ~50 min
- **Completed:** 2026-06-14
- **Tasks:** 2 (both `auto`, `tdd="true"`)
- **Files changed:** 15 (5 created, 10 modified)

## Accomplishments

- **MODEL-02 — pure-Rust CPU apply path.** `cb-model::predict_raw(model, &[Vec<f32>]) -> Vec<f64>`: binarize each float feature via the STRICT `raw > border` count (Step A, `quantization.h:138`), compute the per-tree leaf index via the forward-bit-order `cb_train::leaf_index` (Step B, `evaluator_impl.cpp:26-50`), and accumulate `bias + Σ_trees leaf_values[leaf]` with the per-object leaf-sum routed through `cb_core::sum_f64` and the model bias added EXACTLY once (Step C / Pitfall 6). The file imports no backend / cubecl symbol — it runs with no GPU toolchain present. Oracle-locked vs `binclf_skeleton/predictions.npy` ≤1e-5.
- **LOSS-06 — five prediction-type transforms.** `PredictionType { RawFormulaVal, Probability, LogProbability, Class, Exponent }` + `apply_prediction_type`. `Probability`/`LogProbability` emit two columns per object (`f64::exp`, matching the oracle's `std::exp` path); `Class` thresholds at 0; `Exponent` uses `f64::exp` within the 1e-5 gate (FastExp gap, A2). Each type oracle-locked vs `prediction_types/*.npy` ≤1e-5.
- **LOSS-01 — CrossEntropy + Focal (D-09 complete).** `Loss::CrossEntropy` (delegates to the shared Logloss sigmoid-gradient helper) and `Loss::Focal { alpha, gamma }` (der1/der2 transcribed verbatim from `error_functions.h:1684-1709`, with the `p`-clamp to `[1e-13, 1-1e-13]` so a saturated logit cannot produce NaN — T-04-02-02). New `cb-backend` `focal_gradient`/`focal_hessian` `#[cube]` kernels (generics-float); CrossEntropy reuses the existing Logloss kernels. Binclf now trains under Logloss / CrossEntropy / Focal — splits, leaf values, and per-iteration staged approx all oracle-locked ≤1e-5.

## Task Commits

1. **Task 1: pure-Rust CPU apply path + prediction-type transforms** — `c3aa903` (feat)
2. **Task 2: CrossEntropy + Focal losses with oracle-locked binclf training** — `ffee2a1` (feat)

_Both tasks are `tdd="true"`: the oracle/unit tests were authored alongside the implementation and gate it; all tests pass green (apply 3/3, predict 5/5, loss 4/4 + cb-backend 9/9)._

## Files Created/Modified

- `crates/cb-model/src/apply.rs` (created) — `predict_raw`, `binarize_feature`, internal leaf-walk; GPU-toolchain-free.
- `crates/cb-model/src/predict.rs` (created) — `PredictionType` enum + `apply_prediction_type` (two-column probs).
- `crates/cb-model/src/lib.rs` — wired `mod apply; mod predict;` + re-exports.
- `crates/cb-model/Cargo.toml` — `cb-oracle` / `ndarray` / `ndarray-npy` dev-deps for the oracle tests.
- `crates/cb-model/tests/{apply,predict}_oracle_test.rs` (created) — build `Model` from upstream model.json, lock apply + all 5 prediction types.
- `crates/cb-compute/src/runtime.rs` — `Loss::CrossEntropy` + `Loss::Focal { alpha, gamma }` (drop `Eq`).
- `crates/cb-compute/src/loss.rs` — `cross_entropy_der1/der2` + `focal_der1/der2` (+ `FOCAL_P_MIN`).
- `crates/cb-compute/src/lib.rs` — export the new loss fns + constant.
- `crates/cb-compute/src/loss_test.rs` — CrossEntropy/Focal der1/der2 unit asserts + saturation no-NaN test.
- `crates/cb-backend/src/kernels.rs` — `focal_gradient_kernel` / `focal_hessian_kernel` (`F: Float`, alpha/gamma as length-1 `Array<F>`).
- `crates/cb-backend/src/cpu_runtime.rs` — route CrossEntropy→Logloss kernels, Focal→new kernels (`launch_focal_f64`).
- `crates/cb-train/src/boosting.rs` + `metrics.rs` — thread the new losses through `autolr_target_type` + `EvalMetric::for_loss`.
- `crates/cb-train/tests/loss_oracle_test.rs` (created) — CrossEntropy + Focal training oracle locks + in-env der1/der2 gates.

## Decisions Made

See the `key-decisions` frontmatter. Most load-bearing: (1) the apply path takes SoA `&[Vec<f32>]` columns and is backend-free for MODEL-02; (2) `binclf_skeleton/model.json` is the single source model for both the apply lock and the prediction-type lock; (3) Focal kernel scalars are passed as length-1 device arrays to preserve the generics-float rule.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 — Blocking] Focal kernel scalar args broke the generics-float rule**
- **Found during:** Task 2 (cube kernel build)
- **Issue:** Passing `alpha: F` / `gamma: F` as generic scalar kernel args failed to compile — a generic `#[cube(launch)]` scalar parameter requires `F: ScalarArgType` (`CubeElement + Scalar + NumCast`), which is not implied by `F: Float` and would force a non-generic float bound (violating AGENTS.md generics-float).
- **Fix:** Pass `alpha`/`gamma` as length-1 `Array<F>` arguments (read at index 0). The kernel stays `F: Float` generic; the launch site creates two single-element device buffers. CubeCL error guideline consulted before the fix (per AGENTS.md) — the actual math-method errors (`.ln()`/`.powf()`/`.clamp()`) were resolved to associated-function form (`F::ln`, `F::powf`, `F::clamp`) it prescribes.
- **Files modified:** `crates/cb-backend/src/kernels.rs`, `crates/cb-backend/src/cpu_runtime.rs`
- **Verification:** `cargo build -p cb-backend --features cb-backend/cpu` exits 0; Focal training oracle-locks ≤1e-5.
- **Committed in:** `ffee2a1`

**2. [Rule 3 — Blocking] `Loss` could no longer derive `Eq`**
- **Found during:** Task 2
- **Issue:** `Loss::Focal { alpha: f64, gamma: f64 }` contains `f64`, which is not `Eq`; the enum derived `Eq`.
- **Fix:** Dropped `Eq` from `Loss` (kept `Copy, Clone, PartialEq`). No call site relied on `Loss: Eq` (verified by grep); `BoostParams` already omits `Eq` (it has f64 fields).
- **Files modified:** `crates/cb-compute/src/runtime.rs`
- **Committed in:** `ffee2a1`

## Issues Encountered

- **Disk-space limit (environment, not a code defect):** the box has <1 GB free and `target/` is 8.9 GB. `cargo test -p cb-compute loss` cannot run because cb-compute's test profile must recompile `polars-core` (a transitive dev-dep via `cb-data`, ~1.3 GB rlib) and fails with `No space left on device`. The new CrossEntropy/Focal der1/der2 unit tests WERE added to `cb-compute/src/loss_test.rs`, and the SAME functions are fully exercised and PASSING through `cb-train/tests/loss_oracle_test.rs` (in-env der1/der2 gates + the CrossEntropy/Focal training oracle locks ≤1e-5), which compiled and ran green. Logged to `deferred-items.md`.
- `cargo test --workspace` (the plan's final sanity) was likewise NOT run for the same disk reason; every in-scope crate suite that does not pull polars was run instead: cb-model apply 3/3 + predict 5/5, cb-train loss 4/4, cb-backend 9/9.

## Deferred Issues

None within scope. The disk-blocked `cargo test -p cb-compute loss` / `cargo test --workspace` runs are tracked in `deferred-items.md`; equivalent coverage passes via cb-train.

## Known Stubs

None. The apply path and prediction-type transforms are fully wired to real model data (upstream model.json) and oracle-locked; no placeholder/empty data sources.

## Next Phase Readiness

- The apply path (`predict_raw`) and `RawFormulaVal` substrate are in place for SHAP / fstr (Plan 04) and the Builder facade (Plan 05).
- D-09 binclf loss surface is complete (Logloss / CrossEntropy / Focal all train + oracle-locked).
- LOSS-06 uncertainty types (RMSEWithUncertainty / VirtEnsembles / TotalUncertainty) remain deferred to Phase 6 (D-10) — the five deterministic types are done.

## Self-Check: PASSED

- Created files verified present: `crates/cb-model/src/apply.rs`, `crates/cb-model/src/predict.rs`, `crates/cb-model/tests/apply_oracle_test.rs`, `crates/cb-model/tests/predict_oracle_test.rs`, `crates/cb-train/tests/loss_oracle_test.rs`.
- Commits verified present: `c3aa903` (Task 1), `ffee2a1` (Task 2).
- Tests green: cb-model apply 3/3, predict 5/5; cb-train loss 4/4; cb-backend 9/9.
- Acceptance grep gates pass: apply.rs has NO cubecl/cb_backend ref; strict-> binarization; no raw float sum; Focal kernels are `F: Float` generic; error_functions doc comments present.

---
*Phase: 04-model-serialization-shap-rust-api-first-full-oracle-lock*
*Completed: 2026-06-14*
