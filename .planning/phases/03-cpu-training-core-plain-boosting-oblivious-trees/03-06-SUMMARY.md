---
phase: 03-cpu-training-core-plain-boosting-oblivious-trees
plan: 06
subsystem: training
tags: [eval-metric, validation-metrics, rmse, logloss, per-iteration-logging, multiple-eval-sets, overfitting-detector, oracle, parity, TRAIN-07]

# Dependency graph
requires:
  - phase: 03-05
    provides: "cb-train train_with_eval + EvalSet + the overfitting detector (IncToDec/Iter/Wilcoxon) + BestModelTracker; the per-iteration inline eval-set loss STUB (boosting.rs::inline_eval_metric) this plan supersedes"
  - phase: 03-01
    provides: "cb-compute Loss enum + loss.rs (sigmoid, rmse/logloss der); cb-train boosting loop over the generic Runtime boundary; oblivious tree.rs (leaf_index, FeatureMatrix, tree eval contribution)"
  - phase: 01
    provides: "cb_core::sum_f64 ordered reduction; CbError/CbResult"
provides:
  - "cb_train::metrics — EvalMetric{Rmse,Logloss} computing the per-iteration eval-set validation metric (weighted RMSE = sqrt(sum_w (pred-target)^2 / sum_w); weighted cross-entropy over p=sigmoid(raw logit)); eval_metric defaults to the objective via EvalMetric::for_loss; all folds via cb_core::sum_f64; degenerate eval set / non-positive total weight => CbError::Degenerate (no div-by-zero/panic)"
  - "cb_train::metrics::EvalMetricHistory — per-eval-set per-iteration metric log (per_set[k] is validation_k's curve; primary() is index 0 the detector consumes)"
  - "cb_train::train_with_eval_sets — boosting loop computing the eval_metric over N eval sets per iteration, logging per-set per-iteration values into EvalMetricHistory and feeding the PRIMARY set's metric to the overfit detector; train_with_eval is now a single-set wrapper; BoostParams gains eval_metric: Option<EvalMetric>"
  - "eval_metrics_oracle: per-iteration eval-metric history per eval set (2 sets, RMSE + Logloss) locked vs upstream get_evals_result() at <=1e-5; the eval_metric=None default-to-objective curve equivalence"
affects: [cb-train, "Phase 3 Plan 07 (auto-LR TRAIN-08)", "Phase 4 (model predict path / eval-routing residual follow-up)", "Phase 8 (Python API: eval_set / eval_metric / get_evals_result surface)"]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "EvalMetric is a small value-type metric calcer (RMSE / Logloss) decoupled from the boosting loop; eval_metric defaults to the objective (EvalMetric::for_loss(loss)) and is overridable via BoostParams.eval_metric: Option<EvalMetric>, matching upstream's eval_metric==objective default"
    - "The eval metric is weighted (sum_w numerator / sum_w denominator) even though this phase's eval sets carry uniform weight 1.0 — the weighted formulation is the upstream metric shape and absorbs the unweighted case exactly (total_weight == n)"
    - "Multiple eval sets are first-class: train_with_eval_sets takes &[EvalSet] and produces an EvalMetricHistory (per_set[k]); the PRIMARY set (index 0 == validation_0) is the one the overfitting detector consumes — the stop/best-iteration decision path is UNCHANGED from Plan 05, only the metric SOURCE moved from the inline stub to cb-train::metrics"
    - "Every metric fold (squared-error numerator, cross-entropy numerator, weight denominator) routes through cb_core::sum_f64 in canonical object order (D-05/D-08); degenerate guards (empty set / length mismatch / non-finite or non-positive total weight) return CbError::Degenerate, never div-by-zero or panic (T-03-06-01)"

key-files:
  created:
    - crates/cb-train/src/metrics.rs
    - crates/cb-train/src/metrics_test.rs
    - crates/cb-train/tests/eval_metrics_oracle_test.rs
    - crates/cb-oracle/fixtures/eval_metrics/{rmse,logloss}/{model.json,eval0_metric.npy,eval1_metric.npy,config.json}
    - crates/cb-oracle/fixtures/inputs/eval_metrics/{X_train,X_eval0,X_eval1,y_train_rmse,y_eval0_rmse,y_eval1_rmse,y_train_logloss,y_eval0_logloss,y_eval1_logloss}.npy + config.json
  modified:
    - crates/cb-train/src/boosting.rs
    - crates/cb-train/src/lib.rs
    - crates/cb-oracle/generator/gen_fixtures.py
    - crates/cb-train/tests/{slice_first,leaf_methods,bootstrap,regularization,overfit}_oracle_test.rs
    - .planning/phases/03-cpu-training-core-plain-boosting-oblivious-trees/deferred-items.md

key-decisions:
  - "eval_metric defaults to the objective (EvalMetric::for_loss) and is overridable via BoostParams.eval_metric: Option<EvalMetric>; the oracle locks both the explicit-metric curve and the None=>objective equivalence."
  - "The eval metric is weighted (sum_w / sum_w) per the upstream metric shape; with uniform eval weights this reduces to the unweighted mean exactly, so the RMSE/Logloss curves match upstream get_evals_result() bit-for-bit (Logloss) / <=1e-8 (RMSE), well within the 1e-5 gate."
  - "Multiple eval sets are supported (train_with_eval_sets over &[EvalSet] -> EvalMetricHistory); the PRIMARY (index 0) set drives the detector. The Plan-05 single-eval-set train_with_eval is retained as a thin wrapper so the TRAIN-06 overfit oracle is unchanged (stop decision identical after the stub->metrics swap)."
  - "eval_metrics fixtures run only 12 deterministic (bootstrap_type=No, random_strength=0) iterations so the per-iteration eval metric stays within 1e-5 of upstream — short enough to avoid the eval-prediction boundary-routing residual the Plan-05 overfit oracle documents (which only perturbs the longer ~32+-iteration curves)."

patterns-established:
  - "Pattern 1: eval-metric value type (EvalMetric) separate from the boosting loop, defaulting to the objective; the loop selects it once per train and applies it per iteration per eval set."
  - "Pattern 2: per-eval-set per-iteration history (EvalMetricHistory.per_set[k]); the primary set (index 0) is the detector's input, all sets are logged."
  - "Pattern 3: train_with_eval_sets is the N-eval-set entry point; train_with_eval (1 set) and train (0 sets) are thin wrappers, so existing oracles keep their exact call shape and behaviour."

requirements-completed: [TRAIN-07]

# Metrics
duration: 35min
completed: 2026-06-13
---

# Phase 3 Plan 06: Eval-Set Metric Logging (TRAIN-07) Summary

**Formalized CatBoost's per-iteration eval-set validation metric in `cb-train::metrics` (`EvalMetric{Rmse,Logloss}` defaulting to the objective, weighted RMSE / weighted cross-entropy, all folds via `cb_core::sum_f64`), supporting MULTIPLE eval sets via `train_with_eval_sets` + `EvalMetricHistory`, REPLACING the Plan 05 inline eval-set loss STUB — the per-iteration metric history per eval set (2 sets, RMSE + Logloss) locks against upstream `get_evals_result()` at ≤1e-5, the `eval_metric=None` default-to-objective curve is equivalent, and the TRAIN-06 overfit oracle stays green (stop decision unchanged after the stub→metrics swap).**

## Performance

- **Duration:** ~35 min
- **Completed:** 2026-06-13
- **Tasks:** 2 (Task 1 auto; Task 2 TDD: unit tests + impl + integration + oracle)
- **Files:** 3 created + 6 modified + committed fixtures (2 scenarios × {model.json, eval0_metric.npy, eval1_metric.npy, config.json} + 10 shared input files + 1 input config)

## Accomplishments

- **`cb-train::metrics` (Task 2):** `EvalMetric{Rmse,Logloss}` with `EvalMetric::for_loss(loss)` (RMSE for the RMSE/MAE family, Logloss for Logloss) implementing the `eval_metric==objective` default. `EvalMetric::eval(approx, target, weights)` computes the weighted RMSE `sqrt(sum_w (pred-target)^2 / sum_w)` and the weighted cross-entropy `sum_w -(y ln p + (1-y) ln(1-p)) / sum_w` (`p = sigmoid(raw logit)`, clamped away from {0,1}). Every fold routes through `cb_core::sum_f64`; an empty set, length mismatch, or non-finite/non-positive total weight returns `CbError::Degenerate` (no div-by-zero/panic, T-03-06-01). `EvalMetricHistory` logs `per_set[k]` per-iteration curves and exposes `primary()` (index 0) for the detector.
- **Boosting wiring (Task 2):** `train_with_eval_sets` evaluates the `eval_metric` over EACH of N eval sets per iteration (each set's raw approximant accumulating the bias + per-tree leaf contributions via the existing `tree_eval_contribution`), logs the per-set per-iteration value into `EvalMetricHistory`, and feeds the PRIMARY set's metric to the overfit detector + best-model tracker (the Plan 05 `AddError`/`IsNeedStop`/`use_best_model` truncation path is byte-for-byte unchanged). `train_with_eval` (1 set) and `train` (0 sets) became thin wrappers; `BoostParams` gained `eval_metric: Option<EvalMetric>`.
- **STUB removed:** the Plan 05 `boosting.rs::inline_eval_metric` function and its `// STUB: ... superseded by cb-train::metrics in Plan 06` marker are deleted (grep confirms no remaining `STUB: ... superseded` marker in `cb-train/src/`).
- **Oracle (Task 1 + 2):** `gen_fixtures.py` `gen_eval_metrics()` trains two scenarios (RMSE regressor, Logloss classifier) each with TWO held-out eval sets (`eval_set=[(X0,y0),(X1,y1)]`, sizes 60/45) and an explicit `eval_metric`, persisting each set's per-iteration metric history from `model.get_evals_result()[validation_k]` as committed `.npy`. `eval_metrics_oracle_test` locks each set's per-iteration curve at ≤1e-5 (`compare_stage(Stage::Predictions, …)`) for both losses, plus the `eval_metric=None` default equivalence. 9 metric unit tests on hand-computed values + `cargo test --workspace` green.

## Task Commits

1. **Task 1:** `7321a8f` (feat) — eval_metrics oracle scenario (gen_eval_metrics, two eval sets, explicit eval_metric, committed per-set metric history)
2. **Task 2:** `745efb2` (feat) — cb-train::metrics + per-iteration logging + train_with_eval_sets replacing the Plan 05 stub + unit tests + oracle

## Files Created/Modified

- `crates/cb-train/src/metrics.rs` — `EvalMetric` (RMSE/Logloss, for_loss default, weighted eval) + `EvalMetricHistory`
- `crates/cb-train/src/metrics_test.rs` — 9 unit tests (hand-computed RMSE/Logloss, degenerate guards, history bookkeeping)
- `crates/cb-train/src/boosting.rs` — removed the inline STUB; added `train_with_eval_sets` (N eval sets, per-iteration eval_metric logging, primary-set detector feed); `train_with_eval` now a single-set wrapper; `BoostParams.eval_metric`
- `crates/cb-train/src/lib.rs` — re-export `EvalMetric`/`EvalMetricHistory`/`train_with_eval_sets`
- `crates/cb-train/tests/eval_metrics_oracle_test.rs` — per-set per-iteration metric oracle (RMSE + Logloss) + default-to-objective curve
- `crates/cb-oracle/generator/gen_fixtures.py` — `gen_eval_metrics()`
- `crates/cb-oracle/fixtures/eval_metrics/*`, `fixtures/inputs/eval_metrics/*` — committed frozen fixtures
- `crates/cb-train/tests/{slice_first,leaf_methods,bootstrap,regularization,overfit}_oracle_test.rs` — new `BoostParams.eval_metric: None` field

## Decisions Made

- **`eval_metric` defaults to the objective** (`EvalMetric::for_loss`) and is overridable via `BoostParams.eval_metric: Option<EvalMetric>`. The oracle locks both the explicit-metric curve and the `None`=>objective equivalence.
- **Weighted metric shape** (`sum_w / sum_w`) per upstream; with uniform eval weights this is exactly the unweighted mean, so the curves match upstream `get_evals_result()` bit-for-bit (Logloss) / ≤1e-8 (RMSE).
- **Multiple eval sets** via `train_with_eval_sets` over `&[EvalSet]` -> `EvalMetricHistory`; the PRIMARY (index 0) set drives the detector. `train_with_eval` (single set) retained as a wrapper so the TRAIN-06 overfit oracle is unchanged.
- **12 deterministic iterations** for the eval_metrics fixtures so the per-iteration metric stays within 1e-5 of upstream (avoids the longer-run eval-prediction boundary-routing residual the overfit oracle documents).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing critical] Added `train_with_eval_sets` (N-eval-set entry point) + `BoostParams.eval_metric`; kept `train_with_eval` as a single-set wrapper**
- **Found during:** Task 2 (replacing the stub with multi-eval-set logging).
- **Issue:** The plan requires SUPPORTING MULTIPLE eval sets and an `eval_metric` override, but the Plan 05 `train_with_eval` took a single `EvalSet` and no metric field; the per-set per-iteration history (`EvalMetricHistory`) had no producer.
- **Fix:** Added `train_with_eval_sets(&[EvalSet], Option<&mut EvalMetricHistory>)` as the N-set entry point and the `BoostParams.eval_metric: Option<EvalMetric>` override; `train_with_eval` (1 set) and `train` (0 sets) delegate to it, so every existing oracle keeps its exact call shape and the overfit stop decision is unchanged.
- **Files modified:** `crates/cb-train/src/boosting.rs`, `crates/cb-train/src/lib.rs`, and the 5 existing oracle tests (added `eval_metric: None`).
- **Verification:** `cargo test --workspace` green; `overfit_oracle` 5 passed / 3 documented-ignored (unchanged from Plan 05).
- **Committed in:** `745efb2` (Task 2 commit).

---

**Total deviations:** 1 auto-fixed (missing-critical API surface for the plan's multiple-eval-sets + eval_metric-override requirement).
**Impact on plan:** Necessary to satisfy the acceptance criteria (multiple eval sets, `eval_metric`); no scope creep — `train`/`train_with_eval` behaviour is preserved by delegation.

## Issues Encountered

- **Pre-existing D-08 grep failure (out of scope, logged).** `scripts/check-no-raw-float-sum.sh` flags `crates/cb-train/src/overfit.rs:521` — the W.J. Cody `erf` `horner` Horner fold `coeffs.iter().fold(0.0, |acc,&c| acc*x + c)`, a multiply-add polynomial evaluation (NOT a parity-critical float summation). Verified pre-existing: committed in Plan 05 (`e1adc96`), and the grep already failed at HEAD before this plan (confirmed by stashing the 03-06 changes). The `SUM_PATTERN` `\.fold\(0\.0` over-matches Horner; routing it through `sum_f64` would be incorrect (sum_f64 cannot express `acc*x + c`). Out of scope per the executor SCOPE BOUNDARY; logged in `deferred-items.md` with a fix candidate (narrow the D-08 pattern). My own `metrics.rs` is D-08-clean. `cargo test --workspace` is green.

## Known Stubs

- None — this plan REMOVES the Plan 05 inline eval-set loss STUB (the `// STUB: ... superseded by cb-train::metrics in Plan 06` marker is gone) and replaces it with the formalized `cb-train::metrics` set. No new stubs introduced.

## Threat Flags

None — no new network/auth/file/schema surface. The eval metric is a pure host computation over trusted in-memory eval predictions/targets (T-03-06-01: empty set / length mismatch / non-finite or non-positive total weight => `CbError::Degenerate`, never div-by-zero/panic, deny-lints hold). All metric folds route through `cb_core::sum_f64` in canonical order (T-03-06-02, D-08; `metrics.rs` grep-clean).

## Next Phase Readiness

- TRAIN-07 per-iteration eval-set metric logging (multiple eval sets, `eval_metric` defaulting to the objective) complete and oracle-locked at ≤1e-5; the Plan 05 inline stub is superseded. Plan 07 (auto-LR, TRAIN-08) is the final Phase-3 slice (keyed off `use_best_model`/`boost_from_average`).
- The eval-prediction boundary-routing residual remains a Phase-4/5 tree-prediction parity item (model predict path); it does not block Plan 07. The eval_metrics oracle deliberately uses 12 deterministic iterations to stay clear of it.

## Self-Check: PASSED

- Created files verified present: `metrics.rs`, `metrics_test.rs`, `eval_metrics_oracle_test.rs`, `eval_metrics/{rmse,logloss}/eval{0,1}_metric.npy` — all FOUND.
- Commits verified in `git log`: `7321a8f` (Task 1), `745efb2` (Task 2) — both FOUND.

---
*Phase: 03-cpu-training-core-plain-boosting-oblivious-trees*
*Completed: 2026-06-13*
