---
phase: 03-cpu-training-core-plain-boosting-oblivious-trees
verified: 2026-06-13T12:00:00Z
status: gaps_found
score: 4/5
overrides_applied: 0
gaps:
  - truth: "Bootstrap/sampling and regularization are seeded by TFastRng64 and reproduce upstream draws (random_strength combined with non-No bootstrap)"
    status: failed
    reason: "CR-01: boosting.rs:590 feeds the sampled/control-masked derivative vector (score_weighted_der1) into score_st_dev instead of the full un-sampled fold derivative vector (weighted_der1). Upstream's CalcDerivativesStDevFromZeroPlainBoosting operates on the full AveragingFold weighted derivatives. This is masked by all current oracle fixtures (every random_strength scenario pins bootstrap_type=No, making score_weighted_der1 == weighted_der1), but constitutes a latent parity break for any run combining random_strength != 0 with bootstrap_type != No."
    artifacts:
      - path: "crates/cb-train/src/boosting.rs"
        issue: "Line 590: score_st_dev(params.random_strength, &score_weighted_der1, model_length) should use &weighted_der1"
      - path: "crates/cb-oracle/generator/gen_fixtures.py"
        issue: "All random_strength fixtures pin bootstrap_type=No (lines 868-882), masking the bug — a cross-scenario fixture (random_strength + Bernoulli/MVS) is needed to gate the fix"
    missing:
      - "Change boosting.rs:590 to use &weighted_der1 instead of &score_weighted_der1 for score_st_dev"
      - "Add a cross-scenario oracle fixture (e.g. random_strength=1.0 + bootstrap_type=Bernoulli) to the generator and lock it <=1e-5 end-to-end"
---

# Phase 3: CPU Training Core — Plain Boosting & Oblivious Trees — Verification Report

**Phase Goal:** A user can train a plain-boosted model of symmetric oblivious trees on the CPU and have every per-tree split, leaf value, and per-iteration approximant match upstream to ≤1e-5.
**Verified:** 2026-06-13T12:00:00Z
**Status:** gaps_found — 1 BLOCKER (CR-01 latent parity break)
**Re-verification:** No — initial verification

## Summary

The phase delivers a substantially complete and oracle-tested CPU training core. All eight TRAIN requirements are implemented; `cargo test --workspace` passes with 0 failures. The vast majority of the oracle surface — all four leaf estimation methods (RMSE/Logloss/MAE), the full plain boosting loop, No/Bernoulli/MVS bootstrap, l2_leaf_reg, detector decisions for all three overfit types, per-iteration eval metrics for multiple eval sets, and auto-LR — locks at ≤1e-5 against upstream catboost 1.2.10.

**One BLOCKER prevents full sign-off:** the code review (03-REVIEW.md) identified CR-01, a latent parity break in the random_strength+sampling interaction, confirmed by reading the source. The fix is trivial (one line), but the current oracle suite does not detect it because every `random_strength` fixture pins `bootstrap_type=No`.

Six `#[ignore]`d tests (multi-tree Bayesian bootstrap, multi-tree random_strength/bagging_temp, and three overfit end-to-end stop iterations) are assessed as acceptable documented deferrals — they are isolated to a tree-1+ RNG-phase drift and an eval-prediction boundary-routing sensitivity, both escalated to Phase 4/5. They do not affect the BLOCKER determination.

---

## Step 1 — Phase Context

**Goal (ROADMAP.md):** A user can train a plain-boosted model of symmetric oblivious trees on the CPU and have every per-tree split, leaf value, and per-iteration approximant match upstream to ≤1e-5.

**Requirements:** TRAIN-01, TRAIN-02, TRAIN-03, TRAIN-04, TRAIN-05, TRAIN-06, TRAIN-07, TRAIN-08
**Plans:** 8 plans (03-00 through 03-07), 8 waves, all marked complete in ROADMAP.md.

---

## Step 2 — Must-Haves (from ROADMAP Success Criteria)

| # | Success Criterion |
|---|------------------|
| SC-1 | `R: Runtime` + `F: Float` boundary in `cb-compute`, cpu backend with `SelectedRuntime = CpuRuntime`, histogram/gradient/scan/reduction kernels run |
| SC-2 | Plain boosting loop (iterations, learning_rate, depth) builds symmetric oblivious trees with all four leaf methods; per-tree split + leaf-value oracles ≤1e-5 |
| SC-3 | Bootstrap/sampling (Poisson/Bayesian/Bernoulli/MVS/No, subsample) and regularization (l2_leaf_reg, random_strength, bagging_temperature) seeded by TFastRng64, reproduce upstream draws |
| SC-4 | Overfitting detection/early stopping (Wilcoxon/IncToDec/Iter, od_pval/od_wait, use_best_model) and per-iteration eval-set metric logging (multiple eval sets, eval_metric) behave correctly |
| SC-5 | Automatic learning-rate selection matches upstream; first end-to-end CPU train→predict cycle runs |

---

## Step 3 — Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | Runtime/Float boundary established; CpuRuntime kernels run | VERIFIED | `crates/cb-compute/src/runtime.rs:73` exports `pub trait Runtime`; `cb-backend` wires `SelectedRuntime = cubecl::cpu::CpuRuntime`; 9 kernel tests pass; D-03 confirmed (cb-compute/Cargo.toml contains no cubecl reference) |
| 2 | Plain boosting loop + symmetric oblivious trees + four leaf methods (Gradient/Newton/Exact/Simple) ≤1e-5 | VERIFIED | `slice_first_oracle_{regression,binclf}` pass; `leaf_methods_oracle_{gradient,newton,exact,simple}` all pass; boosting.rs=737 lines, tree.rs=414, leaf.rs=233 — substantive |
| 3 | Bootstrap/sampling seeded by TFastRng64 reproduces upstream; regularization (l2, random_strength, bagging_temp) matches upstream | FAILED (BLOCKER) | No/Bernoulli/MVS oracle-locked ≤1e-5 end-to-end (PASS); l2 multi-tree locked (PASS); Bayesian first-tree + draw-sequence locked (PASS, multi-tree #[ignore]d documented deferral); BUT: `boosting.rs:590` uses `score_weighted_der1` instead of `weighted_der1` for `score_st_dev` — latent parity break when `random_strength != 0` + `bootstrap_type != No`, masked because all current `random_strength` fixtures pin `bootstrap_type=No` |
| 4 | Overfitting detection (all three types) + eval-set metric logging (multiple eval sets) behave correctly | VERIFIED | `overfit_oracle_{inctodec,wilcoxon,iter,use_best_model}_decision` all pass (detector math locked against upstream eval curve); `overfit_oracle_iter_end_to_end` passes; `eval_metrics_oracle_{rmse,logloss}_both_sets` pass; 3 end-to-end stops #[ignore]d (eval-prediction boundary-routing, Phase 4/5 scope) |
| 5 | Auto-LR matches upstream; end-to-end CPU train→predict cycle runs | VERIFIED | `autolr_{rmse,logloss}_train_predict_cycle_runs` pass; `autolr_test.rs` pins upstream rates (RMSE 0.044808, Logloss 0.005413) ≤1e-5; autolr.rs=149 lines with `guess()` formula and coefficient table |

**Score: 4/5 truths verified**

---

## Step 4 — Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/cb-compute/src/runtime.rs` | Abstract `R: Runtime` + `F: Float` traits (no cubecl) | VERIFIED | 87 lines; `pub trait Runtime` at line 73; cb-compute/Cargo.toml confirms no cubecl dep |
| `crates/cb-compute/src/loss.rs` | RMSE + Logloss + MAE der1/der2 | VERIFIED | 108 lines; all 8 loss unit tests pass |
| `crates/cb-compute/src/leaf.rs` | All four leaf-delta methods (Gradient/Newton/Exact/Simple) | VERIFIED | 233 lines; all 11 leaf unit tests pass |
| `crates/cb-compute/src/histogram.rs` | LeafStats reduction + reduce_leaf_der2 + collect_leaf_residuals | VERIFIED | 154 lines; 6 histogram tests pass |
| `crates/cb-compute/src/score.rs` | L2 split score + random_strength perturbation (score_st_dev, random_score_instance) | VERIFIED (with CR-01 caveat) | 121 lines; score unit tests pass; `score_st_dev` function present but is called with wrong input in boosting.rs (CR-01) |
| `crates/cb-backend/src/cpu_runtime.rs` | impl Runtime for CpuBackend over CubeCL CpuRuntime | VERIFIED | Exists; 5 kernel tests pass |
| `crates/cb-backend/src/kernels.rs` | #[cube] gradient/hessian/scatter kernels (generics-float) | VERIFIED | `#[cube` confirmed; f32/f64 kernel tests pass |
| `crates/cb-train/src/boosting.rs` | Plain boosting loop (iterations/lr/depth/boost_from_average/autolr) | VERIFIED (CR-01 caveat) | 737 lines; imports `Runtime`, `bootstrap`, `autolr`; wired; CR-01 bug on line 590 |
| `crates/cb-train/src/tree.rs` | GreedyTensorSearchOblivious + strict first-wins tie-break | VERIFIED | 414 lines; tie-break tests pass |
| `crates/cb-train/src/bootstrap.rs` | Poisson/Bayesian/Bernoulli/MVS/No over TFastRng64 + per-block reseed | VERIFIED | 439 lines; `EBootstrapType` defined; draw-sequence unit tests pass |
| `crates/cb-train/src/overfit.rs` | IncToDec/Iter/Wilcoxon + BestModelTracker + use_best_model | VERIFIED | 522 lines; all three types present; detector decision tests pass |
| `crates/cb-train/src/metrics.rs` | RMSE + Logloss eval metrics + EvalMetricHistory | VERIFIED | 196 lines; per-set/per-iteration logging tests pass |
| `crates/cb-train/src/autolr.rs` | TAutoLRParamsGuesser coefficient table + guess() formula | VERIFIED | 149 lines; 7 autolr unit tests pass |

---

## Step 5 — Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `boosting.rs` | `cb_compute::Runtime` | trait method calls on `R: Runtime` | WIRED | `use ... Runtime` import on line 30; `R: Runtime` generic parameter on all train functions |
| `boosting.rs` | `cb-train::bootstrap` | per-iteration `bootstrap()` call | WIRED | `use crate::bootstrap::{bootstrap, ...}` line 35; called in the per-tree loop |
| `boosting.rs` | `cb-train::autolr` | pre-train `autolr::guess()` when `auto_learning_rate` | WIRED | `use crate::autolr::{self, TargetType}` line 34; called at train entry |
| `boosting.rs` | `cb-train::overfit::OverfittingDetector` | per-iteration AddError + IsNeedStop | WIRED | confirmed by passing overfit oracle tests |
| `boosting.rs` | `cb-train::metrics::EvalMetric` | per-iteration eval over each eval set | WIRED | confirmed by passing eval_metrics oracle tests |
| `cb-compute::score` | `cb_core::std_normal` (via `score.rs::random_score_instance`) | normal draw per candidate | WIRED | score unit tests pass |
| `cb-compute::histogram` | `cb_core::sum_f64` | ordered host-side bin-total reduction (D-05) | WIRED | confirmed by oracle locks; D-08 grep gate clean |
| `boosting.rs:590` | `score_st_dev` with FULL weighted derivatives | should use `&weighted_der1` | NOT_WIRED (CR-01) | Uses `&score_weighted_der1` (sampled/masked) instead of `&weighted_der1` (full fold) — upstream uses the full fold |

---

## Step 4b — Data-Flow Trace (Level 4)

The training loop (`train_with_eval_sets`) is the primary data-producing function. Key data flows verified:

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|--------------------|--------|
| `boosting.rs` train loop | `approx` (staged approximants) | iterative tree leaf accumulation | Yes — per-tree leaf_deltas applied | FLOWING |
| `boosting.rs` leaf path | `weighted_der1` | `backend.compute_gradients(...)` via Runtime trait | Yes — CpuBackend launches RMSE/Logloss/MAE gradient kernel | FLOWING |
| `boosting.rs` score path (CR-01) | `score_weighted_der1` (used for std_dev) | masked derivative vector | Flows but is WRONG source — should be `weighted_der1` | HOLLOW (wrong source) |
| `autolr.rs` guess | `learning_rate` | `coefficients()` const table + exp/ln/round formula | Yes — formula driven by object_count/iter_count | FLOWING |
| `metrics.rs` eval metric | per-iteration RMSE/Logloss values | `cb_core::sum_f64` over eval set predictions | Yes — weighted reduction over oracle-accurate predictions | FLOWING |

---

## Step 7 — Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|---------|--------|
| `crates/cb-train/src/boosting.rs` | 590 | `score_st_dev(..., &score_weighted_der1, ...)` — wrong input vector for std-dev magnitude | BLOCKER | Systematic parity break when `random_strength != 0` AND `bootstrap_type != No` combined; numerator under-counts because masked entries are zeroed rather than excluded, so std_dev is biased low |

No `TBD`/`FIXME`/`XXX` markers found in any phase-3 modified production files. No `TODO`/`HACK`/`PLACEHOLDER` markers found. No `return null`/`return {}`/stub return patterns found in production paths. No embedded `mod tests` in production source files (source/test separation honored throughout).

---

## Step 7b — Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Gradient kernel runs on CpuRuntime | `cargo test -p cb-backend kernels::gradient` (inferred from full run) | 2 tests pass (f32 + f64) | PASS |
| slice_first RMSE + Logloss oracle ≤1e-5 | `cargo test -p cb-train slice_first_oracle` | 2 tests pass | PASS |
| All four leaf methods oracle ≤1e-5 | `cargo test -p cb-train leaf_methods_oracle` | 4 tests pass | PASS |
| Overfit detector decision locks exactly | `cargo test -p cb-train overfit_oracle` | 5 pass, 3 ignored (boundary-routing residual) | PASS |
| Auto-LR e2e train→predict cycle | `cargo test -p cb-train autolr_e2e` | 2 tests pass | PASS |
| 0 test failures workspace-wide | `cargo test --workspace` | 0 failures, 6 ignored | PASS |

Full test summary from `cargo test --workspace`: **0 failures, 6 ignored, all other tests PASS.**

The 6 `#[ignore]`d tests:
1. `bootstrap_oracle_bayesian` — Bayesian multi-tree residual (structural, first-tree locked, deferred)
2. `regularization_oracle_random_strength` — random_strength multi-tree RNG-phase drift (deferred)
3. `regularization_oracle_bagging_temp` — bagging_temp multi-tree (inherits TRAIN-04 Bayesian residual)
4. `overfit_oracle_inctodec_end_to_end` — eval-prediction boundary-routing sensitivity (Phase 4/5)
5. `overfit_oracle_wilcoxon_end_to_end` — same root cause
6. `overfit_oracle_use_best_model_end_to_end` — same root cause

All 6 carry the correct `#[ignore = "..."]` message with a reason. None of these constitutes a new BLOCKER beyond CR-01 — see reasoning in Gaps Summary below.

---

## Step 6 — Requirements Coverage

| Requirement | Phase 3 Plans | Description | Status | Evidence |
|-------------|-------------|-------------|--------|---------|
| TRAIN-01 | 03-01 | Plain gradient boosting train loop (iterations, learning_rate, depth) | SATISFIED | `boosting.rs` implements the loop; `slice_first_oracle` passes |
| TRAIN-02 | 03-01 | Symmetric (oblivious) decision trees | SATISFIED | `tree.rs` implements `greedy_tensor_search_oblivious`; tie-break tests + oracle pass |
| TRAIN-03 | 03-01, 03-02 | Leaf value estimation — Gradient/Newton/Exact/Simple | SATISFIED | All four methods in `leaf.rs`; `leaf_methods_oracle` passes all four |
| TRAIN-04 | 03-03 | Bootstrap/sampling — Poisson/Bayesian/Bernoulli/MVS/No; subsample | SATISFIED (qualified) | No/Bernoulli/MVS end-to-end locked; Poisson CPU-rejected (upstream-faithful); Bayesian first-tree + draw-sequence locked; multi-tree residual documented deferred |
| TRAIN-05 | 03-04 | Regularization — l2_leaf_reg, random_strength, bagging_temperature | PARTIALLY SATISFIED | l2 fully locked; random_strength first-tree locked; bagging_temp first-tree locked; CR-01 latent bug means combined random_strength+sampling is WRONG — BLOCKER |
| TRAIN-06 | 03-05 | Overfitting detection / early stopping (Wilcoxon/IncToDec/Iter, od_pval/od_wait, use_best_model) | SATISFIED | Detector decision locked for all 3 types; Iter end-to-end passes; 3 end-to-end stops deferred (boundary-routing, Phase 4/5) |
| TRAIN-07 | 03-06 | Eval-set metrics per iteration (multiple eval sets, eval_metric) | SATISFIED | `eval_metrics_oracle` passes for RMSE + Logloss × 2 eval sets; eval_metric default-to-objective passes |
| TRAIN-08 | 03-07 | Automatic learning-rate selection from dataset size | SATISFIED | `autolr.rs` coefficient table + formula; upstream rates matched ≤1e-5; e2e train→predict passes |

All 8 Phase 3 requirements are claimed in the plans. All map correctly to their declared plans. No orphaned requirements.

---

## Human Verification Required

None — all phase behaviors have automated oracle verification. The one manual check listed in 03-VALIDATION.md (CubeCL CpuRuntime executes kernels) is satisfied by the kernel tests passing under `cargo test`.

---

## Gaps Summary

### BLOCKER: CR-01 — score_st_dev uses sampled derivatives instead of full-fold derivatives

**Root cause:** `crates/cb-train/src/boosting.rs:590` passes `&score_weighted_der1` to `score_st_dev()`. The correct input is `&weighted_der1` (the full, un-sampled AveragingFold weighted derivatives).

**Why it is a BLOCKER, not a deferral:**
- The phase goal is "every per-tree split, leaf value, and per-iteration approximant match upstream to ≤1e-5"
- SC-3 requires that regularization (random_strength) and sampling (bootstrap) reproduce upstream draws
- When `random_strength != 0` AND `bootstrap_type != No` are combined, the `scoreStDev` magnitude is systematically biased low (zeroed entries in the numerator, full-n denominator), producing wrong split scores and wrong tree structure — failing the ≤1e-5 gate
- The bug is confirmed by code review CR-01 and by reading the source directly; the masking is incidental (all oracle fixtures happen to pin `bootstrap_type=No`)
- This is not a deferral to a later phase — TRAIN-05 is the phase that claims random_strength and the fix belongs here

**Fix (1 line + 1 oracle fixture):**
1. Change `boosting.rs:590`: `score_st_dev(params.random_strength, &score_weighted_der1, model_length)` → `score_st_dev(params.random_strength, &weighted_der1, model_length)`
2. Add a cross-scenario oracle (e.g. `random_strength=1.0` + `bootstrap_type=Bernoulli`) to the generator and lock it ≤1e-5 end-to-end to confirm the fix

**Assessment of the 6 #[ignore]d tests vs BLOCKER threshold:**
- The three multi-tree stochastic residuals (Bayesian bootstrap tree-1+, random_strength tree-1+, bagging_temp tree-1+) are rooted in a tree-1+ RNG-phase drift from variable-length draw loops. The code correctly produces the FIRST tree, correctly draws the bootstrap weights (unit-validated), and correctly computes leaf values on the full fold. The residual is a draw-order accounting gap in the MULTI-TREE case, already tracked as D-11 / Open Q4 / Phase 5 work. These are legitimate deferrals.
- The three overfit end-to-end stop tests are blocked by an eval-prediction boundary-routing sensitivity (objects within ~1e-7 of a split border routing to the other leaf in the eval path). This is a tree-PREDICTION parity issue (Phase 4 model apply), not a TRAIN-06 detector defect — the detector decision tests prove the TRAIN-06 math is exact. Legitimate Phase 4/5 deferral.
- CR-01 differs from all six of these: it is not a numerics residual near a tolerance boundary, not a draw-order accounting issue, and not an eval-prediction routing sensitivity. It is a wrong-variable bug that, combined with a different bootstrap type, produces structurally incorrect tree choices.

---

_Verified: 2026-06-13_
_Verifier: Claude (gsd-verifier)_
