---
phase: 03-cpu-training-core-plain-boosting-oblivious-trees
plan: 01
subsystem: testing
tags: [gradient-boosting, oblivious-trees, cubecl, cpu-runtime, oracle, rmse, logloss, generics-float, l2-score]

# Dependency graph
requires:
  - phase: 03-00
    provides: "CubeCL CpuRuntime seam (gradient_kernel), cb-oracle::model_json parser, frozen regression_skeleton (RMSE) + binclf_skeleton (Logloss) training oracles"
  - phase: 02-data-layer-pool-quantization-reduction
    provides: "QuantizedPool typed-width bins, cb-core::sum_f64 ordered reduction primitive, D-08 raw-sum grep gate"
provides:
  - "cb-compute abstract Runtime/Float boundary (cubecl-free, D-03) with Loss/Derivatives + coarse compute_gradients op (D-04)"
  - "cb-compute host math: RMSE/Logloss der1/der2 + sigmoid (loss), CalcAverage/ScaleL2Reg/gradient_leaf_delta (leaf), ordered LeafStats reduction (histogram), L2 AddLeafPlain split score + MINIMAL_SCORE (score)"
  - "cb-backend CpuBackend impl of cb_compute::Runtime launching elementwise gradient/hessian/scatter #[cube] kernels, returning UN-reduced buffers (D-02)"
  - "cb-train GreedyTensorSearchOblivious (one split/level, 2^depth leaves) with strict gain>bestGain first-wins tie-break (Pitfall 1) + depth cap"
  - "cb-train plain boosting loop: boost_from_average init, Gradient leaf estimation, lr-scaled leaf values, staged approximants"
  - "slice_first_oracle: Splits + LeafValues + StagedApprox <=1e-5 for BOTH RMSE and Logloss (TRAIN-01/02/03 Gradient)"
affects: [cb-train, cb-compute, cb-backend, "Phase 3 Plan 02 leaf_methods (Newton/Exact/Simple)", "Phase 3 Plan 04 bootstrap", "Phase 4 .cbm serialization/apply/SHAP", "Phase 7 GPU backends"]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Generic R: Runtime boundary in cb-compute (cubecl-free, D-03); cb-backend implements it via CubeCL CpuRuntime; GPU runtimes attach additively in Phase 7"
    - "Kernels scatter elementwise only; every parity-critical SUM finalized host-side via cb-core::sum_f64 in canonical object order (D-02/D-05)"
    - "Oblivious tree leaf index = forward bit order (split i -> bit i); model.json leaf_values are already learning_rate-scaled and added directly to staged approx"
    - "Strict gain > bestGain first-wins split tie-break over upstream candidate order (feature asc, border asc) — Pitfall 1"

key-files:
  created:
    - crates/cb-compute/src/runtime.rs
    - crates/cb-compute/src/loss.rs
    - crates/cb-compute/src/loss_test.rs
    - crates/cb-compute/src/histogram.rs
    - crates/cb-compute/src/histogram_test.rs
    - crates/cb-compute/src/score.rs
    - crates/cb-compute/src/score_test.rs
    - crates/cb-compute/src/leaf.rs
    - crates/cb-compute/src/leaf_test.rs
    - crates/cb-backend/src/cpu_runtime.rs
    - crates/cb-backend/src/cpu_runtime_test.rs
    - crates/cb-backend/src/kernels/scatter.rs
    - crates/cb-train/src/tree.rs
    - crates/cb-train/src/tree_test.rs
    - crates/cb-train/src/tree_tie_break_test.rs
    - crates/cb-train/src/boosting.rs
    - crates/cb-train/tests/slice_first_oracle_test.rs
  modified:
    - crates/cb-compute/Cargo.toml
    - crates/cb-compute/src/lib.rs
    - crates/cb-backend/Cargo.toml
    - crates/cb-backend/src/lib.rs
    - crates/cb-backend/src/kernels.rs
    - crates/cb-train/Cargo.toml
    - crates/cb-train/src/lib.rs
    - crates/cb-core/src/error.rs
    - crates/cb-oracle/src/model_json.rs
    - crates/cb-oracle/src/lib.rs

key-decisions:
  - "Leaf index forward bit order verified against model.json (split i sets bit i); model.json leaf_values are already learning_rate-scaled — boosting stores lr*delta and adds directly to staged approx"
  - "Runtime trait kept minimal for the first slice (single coarse compute_gradients op returning UN-reduced Derivatives); histogram scatter/score live as host functions in cb-compute, added to the trait additively in later slices (D-04 leaves backend decomposition free)"
  - "scaledL2 = l2*(sumAllWeights/docCount); unweighted path -> scaledL2 == l2; Gradient leaf delta = CalcAverage(sumDer, sumWeight, scaledL2)"
  - "L2 split score = sum over level leaves of avg*sumDer (avg = CalcAverage), strict-max select; depth-d level rescored across all 2^level leaves with candidate applied across the level"
  - "Added CbError::DepthExceeded (depth>16) and CbError::Degenerate (no candidate split / empty input) — guards, never panic (T-03-01-01/02)"
  - "Extended cb-oracle::model_json with FeaturesInfoJson/FloatFeatureJson + float_feature_borders() accessor so the oracle test feeds the model's per-feature borders to the trainer as candidate split borders"

patterns-established:
  - "Pattern 1: cb-compute abstract Runtime/Float seam (cubecl-free) implemented by cb-backend CpuBackend; host orchestration finalizes ordered sums"
  - "Pattern 2: tie-break tests mounted as a tree::tie_break child module (dedicated file) so the canonical filter cargo test -p cb-train tree::tie_break selects them while honoring source/test separation"
  - "Pattern 3: train->predict oracle gates Splits/LeafValues/StagedApprox via compare_stage for BOTH RMSE and Logloss in one integration test"

requirements-completed: [TRAIN-01, TRAIN-02, TRAIN-03]

# Metrics
duration: 20min
completed: 2026-06-13
---

# Phase 3 Plan 01: First End-to-End CPU Train Slice (RMSE + Logloss) Summary

**Stood up the generic cb-compute Runtime/Float boundary, the cb-backend CubeCL CpuRuntime trait impl, and the cb-train plain boosting loop that grows symmetric oblivious trees with Gradient leaf estimation — oracle-locked on per-tree splits, leaf values, and per-iteration staged approximants to <=1e-5 for BOTH RMSE (regression) and Logloss (binary classification).**

## Performance

- **Duration:** ~20 min
- **Started:** 2026-06-13T07:31:37Z
- **Completed:** 2026-06-13T07:52:06Z
- **Tasks:** 4
- **Files created/modified:** 27

## Accomplishments

- **cb-compute boundary (cubecl-free, D-03):** abstract `Runtime`/`Float` traits + `Loss`/`Derivatives` (D-04), and the host-side parity math — RMSE/Logloss der1/der2 + sigmoid (`loss`), `CalcAverage`/`ScaleL2Reg`/`gradient_leaf_delta` with the `count>0` guard (`leaf`), the ordered `LeafStats` bucket reduction routed through `cb_core::sum_f64` (`histogram`), and the L2 `AddLeafPlain` split score with `MINIMAL_SCORE = NEG_INFINITY` (`score`). 19 unit tests green.
- **cb-backend CpuRuntime impl (D-01/D-03):** `CpuBackend impl cb_compute::Runtime`, launching the elementwise `#[cube]` gradient/Logloss-hessian kernels and the new histogram-scatter kernel (per-object `der1*weight`, NO in-kernel reduction, D-02/D-05) on `CpuRuntime`, returning UN-reduced per-object buffers for the host to fold. 8 tests green under deny-lints.
- **cb-train oblivious tree growth (TRAIN-02):** `GreedyTensorSearchOblivious` — one split per level applied across the whole level, `2^depth` leaves — with the strict `gain > bestGain` first-wins tie-break over the exact upstream candidate order (feature asc, border asc) (Pitfall 1), depth capped `<=16` against `2^depth` overflow.
- **cb-train plain boosting loop (TRAIN-01) + Gradient leaf estimation (TRAIN-03):** `boost_from_average` starting approx (target mean for RMSE / 0 for Logloss, stored as `Model.bias`, Pitfall 2), per-iteration `compute_gradients -> grow tree -> lr-scaled Gradient leaf values -> approx update`, staged approximants recorded per iteration.
- **First-slice oracle (D-08):** `slice_first_oracle` trains on `regression_skeleton` (RMSE) and `binclf_skeleton` (Logloss) and gates `Stage::Splits`, `Stage::LeafValues`, `Stage::StagedApprox` against the Plan-00 fixtures at `<=1e-5` for BOTH losses. `cargo test --workspace` green (wave merge gate).

## Task Commits

Each task was committed atomically:

1. **Task 1: cb-compute boundary — Runtime/Float traits, loss, histogram, score, leaf** — `0bd740b` (feat, TDD)
2. **Task 2: cb-backend CpuRuntime impl + histogram-scatter kernel** — `661eec6` (feat)
3. **Task 3: cb-train oblivious tree growth + strict first-wins tie-break** — `3134858` (feat, TDD)
4. **Task 4: plain boosting loop + first-slice oracle (RMSE + Logloss)** — `3c155eb` (feat, TDD)

_TDD tasks: tests and implementation were committed together per task (single atomic commit each), with the test files written alongside the implementation in the same commit._

## Files Created/Modified

- `crates/cb-compute/src/runtime.rs` — abstract `Runtime`/`Float` traits, `Loss`, `Derivatives` (cubecl-free, D-03/D-04).
- `crates/cb-compute/src/loss.rs` (+`_test`) — RMSE/Logloss der1/der2, sigmoid (error_functions, Pitfall 6).
- `crates/cb-compute/src/leaf.rs` (+`_test`) — CalcAverage (count>0 guard), ScaleL2Reg, gradient_leaf_delta (online_predictor.h).
- `crates/cb-compute/src/histogram.rs` (+`_test`) — `LeafStats` (TBucketStats analogue) + ordered `reduce_leaf_stats` via cb_core::sum_f64.
- `crates/cb-compute/src/score.rs` (+`_test`) — L2 AddLeafPlain split score, `MINIMAL_SCORE = NEG_INFINITY`.
- `crates/cb-compute/src/lib.rs`, `Cargo.toml` — module wiring + cb-core/cb-data deps (no cubecl, D-03).
- `crates/cb-backend/src/kernels.rs` — added logloss gradient/hessian kernels + histogram_scatter_kernel (scatter only).
- `crates/cb-backend/src/kernels/scatter.rs` — scatter-kernel host-reference tests.
- `crates/cb-backend/src/cpu_runtime.rs` (+`_test`) — `CpuBackend impl cb_compute::Runtime`.
- `crates/cb-backend/src/lib.rs`, `Cargo.toml` — mod wiring + cb-compute/cb-core deps.
- `crates/cb-train/src/tree.rs` (+`tree_test.rs`, `tree_tie_break_test.rs`) — GreedyTensorSearchOblivious, select_best_candidate, leaf_index, check_depth.
- `crates/cb-train/src/boosting.rs` — `train<R: Runtime>`, `BoostParams`, `Model`, `ObliviousTree`.
- `crates/cb-train/tests/slice_first_oracle_test.rs` — RMSE + Logloss train->predict oracle.
- `crates/cb-train/src/lib.rs`, `Cargo.toml` — re-exports + dep wiring (cb-oracle/approx/ndarray dev-deps).
- `crates/cb-core/src/error.rs` — new `DepthExceeded` / `Degenerate` CbError variants.
- `crates/cb-oracle/src/model_json.rs`, `lib.rs` — FeaturesInfoJson/FloatFeatureJson + `float_feature_borders()`.

## Decisions Made

- **Leaf index forward bit order + pre-scaled leaf values:** verified against `regression_skeleton/model.json` that split `i` sets bit `i` and that the stored `leaf_values` already include the `learning_rate` factor (the staged approx adds them directly). The boosting loop stores `lr * gradient_leaf_delta` and updates `approx += leaf_value[leaf(i)]`, matching the oracle exactly.
- **Minimal Runtime trait for the first slice:** the trait exposes one coarse op (`compute_gradients` returning UN-reduced `Derivatives`); histogram scatter and split scoring live as host functions in `cb-compute` (`reduce_leaf_stats`, `l2_split_score`) for the first slice. D-04 lets the backend own internal decomposition, so wider trait ops are added additively in later slices without reshaping `cb-train`.
- **L2 split score over the level:** at depth `level`, each candidate is rescored across all `2^level` current-level leaves with the candidate applied across the level (`avg*sumDer` per leaf, `avg = CalcAverage`), then strict-max select. Verified to reproduce tree-0 splits (feat3@0.3005, feat0@0.5757) of the oracle.
- **New CbError variants** (`DepthExceeded`, `Degenerate`) for the depth cap and degenerate-split paths — guards that return errors, never panic (T-03-01-01/02).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Extended cb-oracle::model_json with float-feature borders**
- **Found during:** Task 4 (slice_first_oracle test)
- **Issue:** The oracle test must feed the trainer the model's per-feature candidate borders, but `ModelJson` (from Plan 00) exposed only `oblivious_trees` + `scale_and_bias`; the `features_info.float_features[].borders` were unparsed, so the test could not supply candidate borders.
- **Fix:** Added `FeaturesInfoJson`/`FloatFeatureJson` Deserialize structs + a `float_feature_borders()` accessor returning `Vec<Vec<f64>>`; exported them from `cb-oracle::lib`. The existing `model_json_test` still passes (the new fields deserialize from the same fixtures).
- **Files modified:** crates/cb-oracle/src/model_json.rs, crates/cb-oracle/src/lib.rs
- **Verification:** `cargo test -p cb-oracle` green (18 tests); `slice_first_oracle` consumes the borders and passes.
- **Committed in:** 3c155eb (Task 4 commit)

**2. [Rule 2 - Missing Critical] Added DepthExceeded / Degenerate CbError variants**
- **Found during:** Task 3 (tree growth) — carried into the boosting loop in Task 4
- **Issue:** The plan calls for depth-cap and degenerate-leaf errors but `CbError` had no suitable variant; without them the depth cap or an empty/degenerate level would have to panic, violating the deny-lints and the T-03-01-01/02 mitigations.
- **Fix:** Added `CbError::DepthExceeded { depth, max }` and `CbError::Degenerate(String)` to `cb-core::error`; `check_depth` and the tree/boosting degenerate paths return them.
- **Files modified:** crates/cb-core/src/error.rs
- **Verification:** `cargo test -p cb-core` green (21 tests); `depth_cap_rejected_not_panicked` test passes.
- **Committed in:** 3134858 (Task 3 commit)

---

**Total deviations:** 2 auto-fixed (1 blocking, 1 missing-critical).
**Impact on plan:** Both were necessary to complete the planned oracle test and the planned depth-cap/degenerate guards. No scope creep — both are required by the plan's own acceptance criteria.

## Issues Encountered

- **D-08 raw-sum grep false positives in doc comments:** the `check-no-raw-float-sum.sh` grep matched the literal `.sum()` / `.fold(0.0` strings inside explanatory doc comments in `histogram.rs`, `score.rs`, `tree.rs`, and `lib.rs`. Resolved by rephrasing the prose ("raw iterator-sum or zero-seeded float fold") so the grep sees no banned literal; the actual code already routes every sum through `cb_core::sum_f64`.
- **Test-filter selection for `tree::tie_break` and `slice_first_oracle`:** the plan's verify commands use those exact filters. Mounted the tie-break tests as a `tree::tie_break` child module (dedicated file, source/test separation preserved) and prefixed the oracle test functions with `slice_first_oracle_` so both canonical filters select correctly.
- **Disk-space exhaustion during `cargo test --workspace`:** the full workspace build pulls in `cubecl-cpu`'s heavy `tracel-mlir-sys` (MLIR) transitive dep, filling the disk (100%) mid-build and corrupting `cb-backend`'s incremental cache. Resolved by clearing `target/debug/incremental` + the stale `cb-backend` deps and rebuilding; the workspace then built and all tests passed. No code change required — a host environment constraint, not a plan defect. (Noted for the verifier: GPU-free CPU builds still compile the MLIR optimizer dep.)

## User Setup Required

None — no external service configuration required. The CubeCL/bytemuck stack was vetted and wired in Plan 00; no new dependencies were added this plan.

## Next Phase Readiness

- The full tree/leaf/score math surface and the generic `R: Runtime` compute seam are proven end-to-end and oracle-locked for RMSE + Logloss — TRAIN-01/02/03 (Gradient) are complete.
- Plan 02 (leaf methods Newton/Exact/Simple, D-09) plugs into the same `cb-compute::leaf` module and the established boosting loop; the `Runtime` trait can widen additively for histogram/eval ops without reshaping `cb-train`.
- Plan 04+ (bootstrap/regularization/overfit/eval/auto-LR) attach to the same loop; the `bootstrap.rs`/`overfit.rs`/`autolr.rs` modules from the research structure remain unimplemented (their own later slices).
- No open blockers from this plan. Host disk pressure is an environment concern (the MLIR transitive dep is large) — not a code blocker.

## Self-Check: PASSED

All claimed files exist on disk (runtime.rs, loss.rs, leaf.rs, score.rs, histogram.rs, cpu_runtime.rs, tree.rs, boosting.rs, slice_first_oracle_test.rs and their tests) and all four task commits (0bd740b, 661eec6, 3134858, 3c155eb) are present in git history. `cargo test --workspace` is green; `slice_first_oracle` passes Splits/LeafValues/StagedApprox at <=1e-5 for both RMSE and Logloss; D-08 raw-sum grep clean; cb-compute is cubecl-free (verified via `cargo tree`).

---
*Phase: 03-cpu-training-core-plain-boosting-oblivious-trees*
*Completed: 2026-06-13*
