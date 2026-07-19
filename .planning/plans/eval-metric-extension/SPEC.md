---
title: EvalMetric extension — MAE / MAPE / Quantile (Min-optimized flat metrics)
status: draft
format: markdown
spec_version: 1
updated_at: 2026-07-18T00:00:00Z
source_requirements:
  - "User: Implement features of CatBoost that have not yet been implemented in catboost_rs."
  - "User decision: add an ORCH-04 metric-extension prerequisite so FSTR-02 can target the full numeric loss set."
  - ".planning/plans/fstr-02-loss-change/ (dependent feature)"
treefinder_pending:
  collection: UNRESOLVED
  document_id: UNRESOLVED
  note: "TreeFinder MCP unavailable; local SPEC is authoritative draft. This SPEC is a PREREQUISITE for FSTR-02."
enables:
  - "FSTR-02 multi-loss LossFunctionChange (needs MAE/Quantile/MAPE final-error via EvalMetric)"
---

# EvalMetric extension — MAE / MAPE / Quantile

## 1. Context

`cb_train::EvalMetric` (crates/cb-train/src/metrics.rs:64) is the standalone-metric enum ORCH-04
exposes via `calc_metric`/`eval_metric`. Its flat (non-ranking) arms today are `Rmse`, `Logloss`,
`Msle`, and `Custom(CustomMetricHandle)` `[VERIFIED: CODEGRAPH crates/cb-train/src/calc_metrics.rs:157-169
parse_metric arms rmse/logloss/msle]`. MAE, MAPE, and Quantile are listed in `DEFERRED_METRICS`
and rejected by `parse_metric` with a typed "unknown metric" error
`[VERIFIED: CODEGRAPH calc_metrics.rs:233-238 unknown-name arm cites DEFERRED_METRICS]`.

FSTR-02 (multi-loss LossFunctionChange) needs the model metric's `GetFinalError` for these
Min-optimized numeric losses, sourced through the ORCH-04 surface (the facade supplies it as a
closure to `cb-model`; the metric implementations must live in `cb-train`). This SPEC adds the three
Min-optimized flat metrics so FSTR-02 (and the standalone `eval_metric` surface) can use them.

The flat-metric contract is `EvalMetric::eval(approx, target, weight) -> CbResult<f64>` returning the
final error (`GetFinalError` = `error_sum / weight_sum`, folded via `cb_core::sum_f64`, D-08)
`[VERIFIED: CODEGRAPH calc_metrics.rs:291 metric.eval(...); metrics.rs uses sum_f64]`. Adding a metric
means: (1) an enum arm, (2) an `eval` arm with the final-error math, (3) a `parse_metric` name arm,
(4) removal from `DEFERRED_METRICS` + updated unknown-name help text.

Final-error math (Min-optimized; all `is_max_optimal == false`):
- **MAE:** `mean |approx − target|` (weighted: `Σ w·|a−t| / Σ w`).
- **MAPE:** `mean |approx − target| / max(|target|, ε)` — upstream guards the zero-target divisor
  `[INFERRED: upstream metric.cpp MAPE; confirm ε convention at fixture time]`.
- **Quantile(alpha):** mean pinball loss `t≥a ? alpha·(t−a) : (alpha−1)·(t−a)` = `t≥a ?
  alpha·(t−a) : (1−alpha)·(a−t)`; default `alpha=0.5` (= 0.5·MAE). `QUANTILE_ALPHA` exists in
  cb-compute `[VERIFIED: CODEGRAPH crates/cb-train/src/boosting.rs:31 imports QUANTILE_ALPHA]`.

## 2. Scope and non-goals

**In scope:** add `EvalMetric::{Mae, Mape, Quantile { alpha: f64 }}` (Min-optimized), their `eval`
final-error implementations, `parse_metric` arms (`"mae"`, `"mape"`, `"quantile"` with an `alpha`
param defaulting to 0.5), removal from `DEFERRED_METRICS`, and a ≤1e-5 oracle per metric vs
`catboost.utils.eval_metric`.

**Non-goals:** wiring these as TRAINING losses (this is metric-only — FSTR-02 uses `eval`, not
training); ranking/grouped variants; Max-optimized metrics; changing `Rmse`/`Logloss`/`Msle`
behavior; any `EvalMetric::Copy` reinstatement (the enum is already non-`Copy` due to `Custom`).

## 3. Dependencies

- `cb_train::EvalMetric` enum + `eval` + `parse_metric` + `DEFERRED_METRICS`
  `[VERIFIED: CODEGRAPH metrics.rs:64, calc_metrics.rs:151/233]`.
- `cb_core::{sum_f64, CbError, CbResult}` (already used) `[VERIFIED: CODEGRAPH metrics.rs:35]`.
- ORCH-04 surface (`calc_metric`/`eval_metric`) automatically covers the new arms via the flat
  `metric.eval` path (non-ranking) `[VERIFIED: CODEGRAPH calc_metrics.rs:281-293]`.
- `Params::parse` / `p.f64_or` param helpers for the Quantile `alpha` `[VERIFIED: CODEGRAPH
  calc_metrics.rs:155,190 f64_or usage]`.
- Oracle: `crates/cb-oracle` harness; extend the `calc_metrics` / `eval_metrics` fixture generator
  `[VERIFIED: CODEGRAPH crates/cb-oracle/generator/gen_ranking_fixtures.py CALC_METRICS_DIR]`.
- Blast radius to verify: `EvalMetric` has 10 callers incl. `boosting.rs` (train-time eval_metric),
  `calc_metrics.rs`, `lib.rs`; `parse_metric` has 3 callers. Any exhaustive `match EvalMetric`
  (e.g. `eval`, `eval_one_group`, `is_ranking`, `use_group_weight`) MUST gain arms — verify each via
  CodeGraph before editing `[VERIFIED: CODEGRAPH blast radius; metrics.rs:494-518 eval_one_group is
  an exhaustive match that will need the new flat arms routed to the flat path or rejected]`.

## 4. Typed contracts

```rust
// crates/cb-train/src/metrics.rs — new EvalMetric arms
pub enum EvalMetric {
    Rmse, Logloss, Msle,
    Mae,                       // NEW — Min-optimized
    Mape,                      // NEW — Min-optimized
    Quantile { alpha: f64 },   // NEW — Min-optimized; default alpha 0.5
    // … existing ranking arms + Custom(CustomMetricHandle) …
}

// EvalMetric::eval(&self, approx: &[f64], target: &[f64], weight: &[f64]) -> CbResult<f64>
//   Mae  => Σ w·|a−t| / Σ w
//   Mape => Σ w·|a−t|/max(|t|,ε) / Σ w
//   Quantile{alpha} => Σ w·pinball(a,t,alpha) / Σ w
// (unit weights when `weight` is empty; reject length mismatch / empty set / non-positive
//  weight-sum with CbError::Degenerate, mirroring the existing flat arms.)
```

```rust
// crates/cb-train/src/calc_metrics.rs — new parse_metric arms
"mae" => { p.reject_unknown(name, &[])?; EvalMetric::Mae }
"mape" => { p.reject_unknown(name, &[])?; EvalMetric::Mape }
"quantile" => { p.reject_unknown(name, &["alpha"])?; EvalMetric::Quantile { alpha: p.f64_or("alpha", 0.5, name)? } }
// remove MAE/MAPE/Quantile from DEFERRED_METRICS; update the unknown-name help string.
```

## 5. Failure-isolated behavioral specifications

### EM-01 — MAE final error
- **Responsibility:** `EvalMetric::Mae.eval` returns weighted mean absolute error.
- **Input:** `approx`, `target`, `weight` (empty ⇒ unit).
- **Output:** `Σ w·|a−t| / Σ w`; `CbError::Degenerate` on length mismatch / empty / non-positive weight-sum.
- **Given/When/Then:** Given approx=[1,3], target=[2,2], unit weight; When eval; Then `(1+1)/2 == 1.0`.
- **Acceptance:** unit `mae_eval` in `metrics_test.rs`.

### EM-02 — MAPE final error (zero-target guard)
- **Responsibility:** `EvalMetric::Mape.eval` returns weighted mean absolute percentage error, guarding `target==0`.
- **Input:** as EM-01.
- **Output:** `Σ w·|a−t|/max(|t|,ε) / Σ w`; never divides by zero; typed error on degenerate lengths.
- **Given/When/Then:** Given a target containing 0; When eval; Then a finite value (no NaN/Inf), matching upstream ≤1e-5.
- **Acceptance:** unit `mape_eval` + `mape_zero_target_finite`.

### EM-03 — Quantile(alpha) final error + alpha parse
- **Responsibility:** `EvalMetric::Quantile{alpha}.eval` returns the mean pinball loss; `alpha` defaults to 0.5, parsed from `quantile:alpha=..`.
- **Input:** `approx`, `target`, `weight`, `alpha`.
- **Output:** `Σ w·pinball / Σ w`; at `alpha=0.5` equals `0.5·MAE`.
- **Given/When/Then:** Given `quantile:alpha=0.9` string; When parsed+eval'd; Then the asymmetric pinball value matches a hand calc; and `parse_metric("quantile")` yields `alpha=0.5`.
- **Acceptance:** unit `quantile_eval_default` + `quantile_eval_alpha` + `quantile_parse_alpha`.

### EM-04 — parse_metric recognition + DEFERRED_METRICS update
- **Responsibility:** `parse_metric` accepts `mae`/`mape`/`quantile[:alpha]` (case-insensitive) and no longer lists them as deferred; unknown-name help text updated; every exhaustive `match EvalMetric` in the crate compiles with the new arms.
- **Input:** metric descriptor strings.
- **Output:** the correct `EvalMetric`; `reject_unknown` still fires on bogus params (e.g. `mae:top=2`).
- **Given/When/Then:** Given `"MAE"`, `"mape"`, `"Quantile:alpha=0.3"`; When parsed; Then the matching variants. Given `"quantile:beta=1"`; Then `CbError::Degenerate` (unknown param). Given `is_ranking(Mae)`; Then false (flat path).
- **Acceptance:** unit `parse_mae_mape_quantile` + `parse_rejects_bad_quantile_param` + confirm `is_ranking` false for all three; and a compile check that `eval_one_group`/`use_group_weight`/any exhaustive match handle the new arms (flat metrics route to the flat `eval`, never the grouped seam — mirror the `Rmse|Logloss|Msle|Custom` arm at metrics.rs:513).
- **Acceptance (regression):** existing `calc_metrics_flat_oracle_test`, `eval_metrics_oracle_test` stay green.

### EM-05 — Oracle parity per metric
- **Responsibility:** ≤1e-5 parity vs `catboost.utils.eval_metric` for MAE, MAPE, Quantile(0.5) and Quantile(alpha≠0.5).
- **Input:** frozen `label`/`approx`(+optional `weight`) `.npy` + expected values from `catboost==1.2.10`.
- **Output:** Rust `eval_metric` matches ≤1e-5.
- **Given/When/Then:** Given the extended `calc_metrics`/`eval_metrics` fixtures; When evaluated; Then max|diff| ≤ 1e-5.
- **Acceptance:** `crates/cb-train/tests/{calc_metrics_flat_oracle_test.rs or eval_metrics_oracle_test.rs}` extended with the three metrics.

## 6. Acceptance scenarios

1. `eval_metric("MAE", label, approx)` == upstream ≤1e-5 (EM-01/EM-05).
2. `eval_metric("MAPE", …)` with a zero in target is finite and == upstream ≤1e-5 (EM-02/EM-05).
3. `eval_metric("Quantile:alpha=0.9", …)` == upstream ≤1e-5; `Quantile` default == 0.5·MAE (EM-03/EM-05).
4. `parse_metric("mae:top=2")` → typed error; RMSE/Logloss/MSLE unchanged (EM-04).

## 7. Impact scope

- **local (cb-train):** `metrics.rs` (enum arms + `eval` arms + any exhaustive `match EvalMetric`
  incl. `eval_one_group` at :494-518 which must route the new flat arms to the flat path or the
  `Rmse|Logloss|Msle|Custom` reject arm), `calc_metrics.rs` (`parse_metric` arms + `DEFERRED_METRICS`
  + `is_ranking` stays false for them). Tests in mounted `metrics_test.rs` / `calc_metrics_test.rs`.
- **cross-module:** none required — ORCH-04's `calc_metric`/`eval_metric`/facade `eval_metrics`
  automatically cover the new arms via the flat path (no signature change).
- **tests:** unit + extended oracle; extend the fixture generator.
- No schema/persistence/wire/config impact. Adding enum variants is source-compatible for the crate
  but breaks any EXHAUSTIVE external `match EvalMetric` — verify no downstream crate matches it
  exhaustively (facade/py consume via strings, not the enum) via CodeGraph.

## 8. Compatibility and migration

Additive enum variants. Internal exhaustive matches gain arms in the same change. No public
signature change; the standalone metric surface simply accepts three more names. No wire change.

## 9. Risks and open questions

- **R1 (MAPE ε):** the exact zero-target guard convention (skip-zero vs `max(|t|,ε)`) is
  `[UNVERIFIED — sparse checkout]`; confirm against `catboost==1.2.10` output at fixture time (an
  all-nonzero-target fixture sidesteps it, but include one zero-target case to pin the guard).
- **R2 (Quantile weight/alpha):** confirm upstream Quantile metric's exact weighting and whether
  `alpha` is the only param; `f64_or("alpha", 0.5)` assumed.
- **R3 (exhaustive matches):** the decisive implementation risk is missing an exhaustive `match
  EvalMetric` arm (compile error) — enumerate them all via CodeGraph first (`eval`, `eval_one_group`,
  `is_ranking`, `use_group_weight`, any Debug/print). EM-04 makes this a gate.

## 10. Traceability and sources

- CodeGraph: `cb-train/src/metrics.rs:{64 EvalMetric, 494-518 eval_one_group}`,
  `cb-train/src/calc_metrics.rs:{151 parse_metric, 233 DEFERRED_METRICS, 250 is_ranking, 281 calc_metric}`,
  `cb-compute/src/custom.rs (CustomMetric is_max_optimal)`, `boosting.rs:31 QUANTILE_ALPHA`.
- Local: `crates/cb-oracle/generator/gen_ranking_fixtures.py` (CALC_METRICS_DIR), fixtures
  `calc_metrics`/`eval_metrics`.
- Downstream: `.planning/plans/fstr-02-loss-change/SPEC.md` (this SPEC unblocks its full loss set).
