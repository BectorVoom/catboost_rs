---
phase: 03-cpu-training-core-plain-boosting-oblivious-trees
plan: 02
subsystem: testing
tags: [gradient-boosting, leaf-estimation, newton, exact-quantile, simple, mae, oracle, cubecl, generics-float]

# Dependency graph
requires:
  - phase: 03-01
    provides: "cb-compute Runtime/Float boundary + Gradient leaf delta (leaf.rs/histogram.rs), cb-train plain boosting loop (boosting.rs), cb-backend CpuBackend gradient/hessian kernels, slice_first oracle harness"
  - phase: 03-00
    provides: "cb-oracle model_json parser (split_borders/leaf_values/float_feature_borders), compare_stage Stage::{Splits,LeafValues,StagedApprox}, gen_fixtures.py generator + numeric_tiny corpus"
  - phase: 02-data-layer-pool-quantization-reduction
    provides: "cb-core::sum_f64 ordered reduction primitive, D-08 raw-sum grep gate"
provides:
  - "cb-compute::leaf all four leaf methods: newton_leaf_delta (sumDer/(-sumDer2+scaledL2), guarded), simple_leaf_delta (== gradient, A6), exact_leaf_delta (weighted sample quantile of leaf residuals), plus LeafMethod enum"
  - "cb-compute::histogram reduce_leaf_der2 (ordered Σ der2*weight per leaf) + collect_leaf_residuals (per-leaf f32 residual+weight members) for Newton/Exact"
  - "cb-compute Loss::Mae + mae_der1/mae_der2 (Quantile alpha=0.5,delta=1e-6) + QUANTILE_ALPHA/QUANTILE_DELTA"
  - "cb-backend mae_gradient_kernel (#[cube] generics-float) + CpuBackend Loss::Mae dispatch"
  - "cb-train BoostParams.leaf_method + compute_leaf_deltas four-way dispatch over the boosting loop (D-05 ordered sums)"
  - "leaf_methods_oracle: Splits+LeafValues+StagedApprox <=1e-5 for Gradient(RMSE)/Newton(Logloss)/Exact(MAE)/Simple(RMSE) — TRAIN-03 complete (D-09)"
affects: [cb-train, cb-compute, cb-backend, "Phase 3 Plan 04 bootstrap", "Phase 3 Plan 05 regularization", "Phase 4 .cbm serialization/apply", "Phase 7 GPU backends"]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Leaf-method dispatch as a host-side function (compute_leaf_deltas) over LeafMethod; closed-form methods consume reduced sums, Exact consumes per-leaf residual members — all reductions via cb_core::sum_f64 (D-05)"
    - "Newton needs a per-leaf Σ der2*weight companion to LeafStats; added as a separate reduce_leaf_der2 rather than widening LeafStats so the score path (which never reads der2) is untouched"
    - "Exact is impossible upstream for RMSE/Logloss (catboost_options.cpp:346) so its oracle uses MAE; its leaf delta is a rank statistic (weighted median), not an L2 average — scaled_l2 is unused for Exact"
    - "Per-method oracle isolation (D-07 discipline carried forward): one loss/method per scenario so a divergence is attributable to a single method's leaf math; Newton uses Logloss (der2=-p(1-p)) where it is genuinely distinct from Gradient"

key-files:
  created:
    - crates/cb-train/tests/leaf_methods_oracle_test.rs
    - crates/cb-oracle/fixtures/leaf_methods/gradient/{model.json,staged.npy,config.json}
    - crates/cb-oracle/fixtures/leaf_methods/newton/{model.json,staged.npy,config.json}
    - crates/cb-oracle/fixtures/leaf_methods/exact/{model.json,staged.npy,config.json}
    - crates/cb-oracle/fixtures/leaf_methods/simple/{model.json,staged.npy,config.json}
  modified:
    - crates/cb-compute/src/leaf.rs
    - crates/cb-compute/src/leaf_test.rs
    - crates/cb-compute/src/histogram.rs
    - crates/cb-compute/src/histogram_test.rs
    - crates/cb-compute/src/loss.rs
    - crates/cb-compute/src/loss_test.rs
    - crates/cb-compute/src/runtime.rs
    - crates/cb-compute/src/lib.rs
    - crates/cb-backend/src/kernels.rs
    - crates/cb-backend/src/cpu_runtime.rs
    - crates/cb-backend/src/cpu_runtime_test.rs
    - crates/cb-train/src/boosting.rs
    - crates/cb-train/tests/slice_first_oracle_test.rs
    - crates/cb-oracle/generator/gen_fixtures.py

key-decisions:
  - "Newton leaf delta = sumDer/(-sumDer2+scaledL2) (CalcDeltaNewtonBody, online_predictor.h:162-170); denominator guarded (>0) returns 0.0 on degenerate/empty leaf or der2==0 loss — never div-by-zero/NaN (T-03-02-01)"
  - "Simple == Gradient leaf delta (A6 RESOLVED): upstream CalcLeafDeltasSimple routes ELeavesEstimation::Simple through the Gradient branch; verified bit-identical leaf values vs Gradient for these params"
  - "Exact leaf delta = CalcOneDimensionalOptimumConstApprox -> weighted sample quantile (MAE alpha=0.5,delta=1e-6): CalcSampleQuantileLinearSearch (stable-sort residuals asc, accumulate weight, first >= totalWeight*alpha-DBL_EPSILON) + CalculateWeightedTargetQuantile delta adjustment. <100 samples uses the linear search (binary-search path deferred until a larger leaf appears)"
  - "Exact is rejected upstream for RMSE/Logloss (catboost_options.cpp:346 — only Quantile/MAE/MAPE/... support it), so the exact scenario trains MAE; Newton is mathematically == Gradient for RMSE (der2==-1), so the newton scenario trains Logloss to exercise Newton distinctly"
  - "Added Loss::Mae + mae_der1/mae_der2 + a mae_gradient_kernel so Exact (and the MAE oracle) can train end-to-end through the same Runtime/boosting loop — required by the plan's own acceptance criteria (oracle-lock all four), not scope creep"

patterns-established:
  - "Pattern 1: compute_leaf_deltas(method, ...) — single dispatch point for the four leaf methods in the boosting loop; closed-form (Gradient/Newton/Simple) vs rank-statistic (Exact) branches share the leaf_of partition and ordered reductions"
  - "Pattern 2: reduce_leaf_der2 / collect_leaf_residuals as additive companions to reduce_leaf_stats — new leaf inputs are added without disturbing the score path's LeafStats contract"
  - "Pattern 3: per-method oracle scenario (one loss + one method each) keeps a divergence attributable to a single method; the generator records the transcribed formula in each scenario's config.json"

requirements-completed: [TRAIN-03]

# Metrics
duration: 12min
completed: 2026-06-13
---

# Phase 3 Plan 02: Leaf Estimation Methods (Newton / Exact / Simple) Summary

**Completed TRAIN-03 by adding Newton, Exact, and Simple leaf-estimation methods alongside the first-slice Gradient method, each oracle-locked on per-tree splits, leaf values, and per-iteration staged approximants to <=1e-5 against upstream catboost 1.2.10 (Newton via Logloss, Exact via MAE's weighted-median optimum, Simple == Gradient per A6).**

## Performance

- **Duration:** ~12 min
- **Started:** 2026-06-13T08:03:04Z
- **Completed:** 2026-06-13T08:15:07Z
- **Tasks:** 2
- **Files created/modified:** 14 source/test/generator files + 12 committed fixture files

## Accomplishments

- **Four leaf-estimation methods (D-09, TRAIN-03):** `cb-compute::leaf` now exposes `LeafMethod::{Gradient, Newton, Simple, Exact}` and the three new deltas:
  - `newton_leaf_delta(sum_der, sum_der2, scaled_l2) = sum_der / (-sum_der2 + scaled_l2)` (`CalcDeltaNewtonBody`), with a guarded `> 0` denominator returning `0.0` on a degenerate/empty leaf or a `der2 == 0` loss (T-03-02-01 — never div-by-zero/NaN).
  - `simple_leaf_delta == gradient_leaf_delta` (A6 resolved — upstream `CalcLeafDeltasSimple` dispatches `Simple` through the Gradient branch).
  - `exact_leaf_delta(residuals, weights, alpha, delta)` — the weighted sample quantile (`CalcSampleQuantileLinearSearch`: stable-sort residuals ascending, accumulate weight, first value `>= totalWeight*alpha - DBL_EPSILON`) plus the `CalculateWeightedTargetQuantile` alpha/delta adjustment.
- **Newton/Exact leaf inputs:** `reduce_leaf_der2` (ordered `Σ der2*weight` per leaf via `sum_f64`, D-05) and `collect_leaf_residuals` (per-leaf `f32` residual + weight members matching upstream's `TVector<float> leafSamples`), added as additive companions to the existing `reduce_leaf_stats` so the score path is untouched.
- **MAE loss + kernel:** added `Loss::Mae`, `mae_der1` (signed half-quantile with a `1e-6` deadzone) / `mae_der2` (`0`), and a `#[cube]` generics-float `mae_gradient_kernel` (CubeCL conditionals-manual mutable-variable pattern) wired into `CpuBackend`'s `Loss::Mae` dispatch — so Exact trains end-to-end through the same `Runtime` boundary.
- **Boosting-loop dispatch:** `BoostParams.leaf_method` and `compute_leaf_deltas` route the per-tree leaf step through the selected method; every leaf reduction goes through `cb_core::sum_f64` (D-05). The default Gradient path is byte-unchanged.
- **Four-method oracle (D-09):** `leaf_methods_oracle_test` trains each method on its scenario (gradient→RMSE, newton→Logloss, exact→MAE, simple→RMSE) and gates `Stage::Splits`, `Stage::LeafValues`, `Stage::StagedApprox` against the new Plan-02 fixtures at `<=1e-5`. The first-slice `slice_first_oracle` (RMSE+Logloss Gradient) still passes — no regression. `cargo test --workspace` green; D-08 / source-test / anyhow CI grep gates all green.

## Task Commits

1. **Task 1: leaf_methods oracle fixtures + transcribe Exact/Simple/Newton bodies** — `5a21df6` (test)
2. **Task 2: implement Newton/Exact/Simple leaf deltas + method dispatch; oracle-lock all four** — `772f694` (feat, TDD: tests + impl in one atomic commit per the prior-wave convention)

## Files Created/Modified

- `crates/cb-oracle/generator/gen_fixtures.py` — `gen_leaf_methods()` emits `leaf_methods/{gradient,newton,exact,simple}` (each `model.json` + `staged.npy` + `config.json`), pinning one `leaf_estimation_method` + the first-slice simplified isolating params; the transcribed Newton/Exact/Simple formulas are recorded in the function doc comment and each scenario's `config.json`.
- `crates/cb-compute/src/leaf.rs` (+`leaf_test.rs`) — `LeafMethod`, `newton_leaf_delta`, `simple_leaf_delta`, `exact_leaf_delta` (+ `DBL_EPSILON`).
- `crates/cb-compute/src/histogram.rs` (+`histogram_test.rs`) — `reduce_leaf_der2`, `collect_leaf_residuals`.
- `crates/cb-compute/src/loss.rs` (+`loss_test.rs`) — `mae_der1`/`mae_der2`, `QUANTILE_ALPHA`/`QUANTILE_DELTA`.
- `crates/cb-compute/src/runtime.rs` — `Loss::Mae` variant.
- `crates/cb-compute/src/lib.rs` — re-exports for the new symbols.
- `crates/cb-backend/src/kernels.rs` — `mae_gradient_kernel`.
- `crates/cb-backend/src/cpu_runtime.rs` (+`cpu_runtime_test.rs`) — `BinaryKernel::MaeGradient` + `Loss::Mae` dispatch + host-reference test.
- `crates/cb-train/src/boosting.rs` — `BoostParams.leaf_method`, `compute_leaf_deltas` dispatch.
- `crates/cb-train/tests/leaf_methods_oracle_test.rs` — the four-method train→predict oracle.
- `crates/cb-train/tests/slice_first_oracle_test.rs` — updated for the new `leaf_method` field and the exhaustive `Loss` match.

## Decisions Made

- **MAE loss added to satisfy the Exact oracle.** Upstream rejects `leaf_estimation_method=Exact` for RMSE/Logloss (`catboost_options.cpp:346` — Exact is only valid for Quantile/GroupQuantile/MultiQuantile/MAE/MAPE/...). The plan's acceptance criteria require Exact oracle-locked end-to-end, so `Loss::Mae` + its gradient kernel were added — the minimum needed to train MAE through the existing `Runtime`/boosting loop. Documented as a deviation (Rule 2/3 — required to complete the task as specified), not scope creep.
- **Newton scenario uses Logloss, not RMSE.** For RMSE `der2 == -1` so `-sum_der2 == sum_weight`, making Newton's denominator identical to Gradient's; the Newton oracle would not exercise the Newton formula distinctly. Logloss (`der2 = -p(1-p)`) makes Newton genuinely distinct and is the meaningful lock.
- **`reduce_leaf_der2` / `collect_leaf_residuals` kept separate from `LeafStats`.** Widening `LeafStats` with a `sum_der2` field would touch the score-path struct literals and the `add_leaf_plain` contract; a separate reducer is additive and leaves the split-scoring path byte-identical.
- **Exact `<100`-sample linear search only.** The numeric_tiny leaves never reach 100 members, so only `CalcSampleQuantileLinearSearch` is ported; the `>=100` binary-search branch is deferred until a corpus needs it (documented in the `exact_leaf_delta` doc comment).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2/3 - Missing Critical / Blocking] Added `Loss::Mae` + MAE gradient kernel to make the Exact oracle trainable**
- **Found during:** Task 1 (probing which `leaf_estimation_method` values catboost 1.2.10 accepts).
- **Issue:** The plan assumes all four methods can be oracle-locked on the existing RMSE/Logloss scenarios, but upstream rejects `Exact` for RMSE/Logloss (`catboost_options.cpp:346`). Without a Quantile-family loss the Exact method cannot be trained end-to-end, so its `Stage::LeafValues`/`Stage::StagedApprox` oracle (the plan's own acceptance criterion) is unreachable.
- **Fix:** Added `Loss::Mae`, `mae_der1`/`mae_der2` (`TQuantileError` alpha=0.5, delta=1e-6), a `#[cube]` `mae_gradient_kernel`, and the `CpuBackend` dispatch; the `exact` scenario trains MAE whose leaf delta is the weighted median of leaf residuals.
- **Files modified:** `runtime.rs`, `loss.rs`(+test), `kernels.rs`, `cpu_runtime.rs`(+test).
- **Verification:** `leaf_methods_oracle_exact` passes Splits/LeafValues/StagedApprox at <=1e-5; `mae_gradients_match_host_reference` green.
- **Committed in:** `772f694` (Task 2).

**2. [Rule 1 - Bug] Rewrote `exact_leaf_delta` indexing to satisfy the `indexing_slicing` deny-lint**
- **Found during:** Task 2 (`cargo clippy` after first implementation).
- **Issue:** The initial Exact implementation used `residuals[0]`, `&residuals[1..]`, and `elements[len-1]` — three `clippy::indexing_slicing` deny-lint violations in production code (T-03-02-01 / CLAUDE.md no-panic discipline).
- **Fix:** Replaced with an iterator min (`f64::INFINITY` seed) and `Vec::last().map_or(...)`; no behavior change.
- **Files modified:** `crates/cb-compute/src/leaf.rs`.
- **Verification:** `cargo clippy -p cb-compute --lib` clean; all leaf unit + oracle tests still pass.
- **Committed in:** `772f694` (Task 2).

### Non-code adjustments

- **Reverted metadata-only churn in the two pre-existing skeleton `model.json` files.** Re-running `gen_fixtures.py` rewrote `model_guid`/`train_finish_time` in `regression_skeleton/model.json` and `binclf_skeleton/model.json` (splits/leaf_values unchanged); reverted both to keep the first-slice oracle byte-stable. Documented to make the no-op explicit.

---

**Total deviations:** 2 auto-fixed (1 missing-critical/blocking, 1 bug) + 1 documented no-op revert.
**Impact on plan:** Deviation 1 was required by the plan's own acceptance criteria (oracle-lock all four methods); deviation 2 is a lint-compliance rewrite with no behavior change. No scope creep.

## Issues Encountered

- **Verify-command filters do not select the new tests as written.** The plan's `cargo test -p cb-compute leaf::` selects 0 tests — the leaf tests live in the `leaf_test` module (source/test separation), so the correct filter is `cargo test -p cb-compute leaf_test::` (or the unqualified `cargo test -p cb-compute`). The oracle filter `cargo test -p cb-train leaf_methods_oracle` works as written. The full `cargo test -p cb-compute` (33 tests) and `cargo test --workspace` were used as the authoritative gates.
- **`Loss` match exhaustiveness.** Adding `Loss::Mae` made the existing `slice_first_oracle_test.rs` `match loss` non-exhaustive (compile error); fixed by folding `Mae` into the `Rmse` arm (slice_first never uses MAE). Caught and fixed before the Task-2 commit.

## Known Stubs

None — all four leaf methods are fully wired and oracle-locked. The Exact `>=100`-sample binary-search quantile path is intentionally deferred (documented in `exact_leaf_delta`); the Phase-3 corpora never reach 100 leaf members, so it cannot be exercised yet and is a future-corpus additive item, not a stub blocking this plan's goal.

## Next Phase Readiness

- TRAIN-03 (all four leaf-estimation methods) is complete and oracle-locked; the `compute_leaf_deltas` dispatch and the `LeafMethod` enum are the stable surface later slices build on.
- Plan 04 (bootstrap/sampling) and Plan 05 (regularization) attach to the same boosting loop; `Loss::Mae` and the MAE kernel are now available if a later slice needs a robust-regression oracle.
- The `Runtime` trait can still widen additively (histogram/eval ops) without reshaping `cb-train`.
- No open blockers. Host disk pressure from the MLIR transitive dep (noted in Plan 01) remains an environment concern, not a code blocker; the full workspace built and tested green this run.

## Self-Check: PASSED

All claimed files exist on disk and both task commits are present in git history (verified below). `cargo test --workspace` is green (cb-backend 9, cb-compute 33, cb-core 21, cb-data 68, cb-oracle 18 + oracle tests, cb-train 6 unit + 4 leaf_methods + 2 slice_first); `leaf_methods_oracle` passes Splits/LeafValues/StagedApprox at <=1e-5 for all four methods; Gradient/slice_first unchanged (no regression); D-08 raw-sum, source/test-separation, and anyhow CI grep gates all green; no `unwrap`/`expect`/raw float fold in the new production code.

---
*Phase: 03-cpu-training-core-plain-boosting-oblivious-trees*
*Completed: 2026-06-13*
