---
title: FSTR-02 — multi-loss LossFunctionChange (numeric, first slice)
status: draft
format: markdown
spec_version: 1
updated_at: 2026-07-18T00:00:00Z
source_requirements:
  - "User: Implement features of CatBoost that have not yet been implemented in catboost_rs."
  - ".planning/plans/next-feature-research/research.md §2 Candidate 3"
treefinder_pending:
  collection: UNRESOLVED
  document_id: UNRESOLVED
  note: "TreeFinder MCP unavailable; local SPEC is authoritative draft."
ordering_constraint:
  after:
    - "in-flight FSTR-01 CTR change to crates/cb-model/src/fstr.rs (uncommitted) — merge-conflict gate"
    - "eval-metric-extension prerequisite (.planning/plans/eval-metric-extension/) — provides MAE/MAPE/Quantile final-error via cb_train::EvalMetric"
  reason: "This SPEC edits fstr.rs (sequence AFTER FSTR-01) AND needs MAE/MAPE/Quantile in EvalMetric (only RMSE is buildable today; MAE/MAPE/Quantile are in cb-train DEFERRED_METRICS). User decision: add the EvalMetric extension as a prerequisite rather than narrowing FSTR-02 to RMSE-only."
---

# FSTR-02 — multi-loss LossFunctionChange

## 1. Context

`get_feature_importance(type='LossFunctionChange', data=pool)` scores each feature by how
much removing its SHAP contribution degrades the model's own loss metric's `GetFinalError`.
Today `cb_model::loss_function_change` **hard-codes binary Logloss** as the final error
`[VERIFIED: CODEGRAPH crates/cb-model/src/fstr.rs:788-848 — logloss_final_error only]`:

```
base = logloss_final_error(approx, labels)
score_f = logloss_final_error(approx − shap_f, labels) − base   // Logloss is Min-optimized
```

So a model trained with RMSE / MAE / Quantile gets a Logloss-based (wrong) importance.

**Layering (CORRECTED — PLAN-CHECK CRITICAL).** The earlier "cb-model depends only on cb-core"
claim was WRONG. The real dependency direction is: **`cb-model` depends on `cb-train`** as a
normal `[dependencies]` entry (`cb-model/Cargo.toml:26`; `cb-model/src/model.rs` does
`pub use cb_train::Split`), while `cb-train → cb-model` is a **dev-dependency only**
`[VERIFIED: LOCAL crates/cb-model/Cargo.toml:26,63; crates/cb-train/Cargo.toml:39-45; CODEGRAPH
cb-model/src/model.rs pub use cb_train]`. So `cb_model::loss_function_change` **could** call
`cb_train::EvalMetric::eval` directly — there is NO circular-dependency barrier.

**Design decision (unchanged, re-justified).** We still INJECT the final-error as a
caller-supplied closure `Fn(&[f64],&[f64]) -> f64`, NOT because layering forbids the direct call,
but because it keeps `fstr.rs` **metric-agnostic and unit-testable** without constructing an
`EvalMetric`, and it lets the **facade** own the `loss: &str → EvalMetric` mapping (reusing
ORCH-04's `eval_metric`). This is a decoupling/cleanliness choice, not a necessity. (Alternative
considered: cb-model calls `cb_train::EvalMetric` directly and takes `loss: &str` — rejected to
keep fstr metric-free, but it is now known to be viable.)

`ORCH-04` provides exactly the needed final-error surface: `EvalMetric::eval(approx, target,
weight) -> CbResult<f64>` (flat final error) and `calc_metrics::calc_metric`
`[VERIFIED: CODEGRAPH crates/cb-train/src/metrics.rs eval; calc_metrics.rs:281]`.

## 2. Scope and non-goals

**In scope:** generalize `loss_function_change` to accept an injected final-error
closure `Fn(&[f64] approx, &[f64] labels) -> f64`; keep the existing SHAP-subtraction and
per-feature difference logic byte-identical; wire the facade
`feature_importance_with_data(LossFunctionChange, pool, loss)` to pick the closure from the
model's trained loss (numeric, **Min-optimized** losses: RMSE, MAE, Quantile, MAPE, Logloss);
a ≤1e-5 oracle per loss over numeric-only models.

**PREREQUISITE (user decision).** MAE / MAPE / Quantile are NOT yet in `cb_train::EvalMetric`
(they sit in `DEFERRED_METRICS`) — only RMSE + Logloss are buildable through the ORCH-04 reuse
path today `[VERIFIED: CODEGRAPH crates/cb-train/src/calc_metrics.rs:233 DEFERRED_METRICS]`. The
**`eval-metric-extension`** feature (`.planning/plans/eval-metric-extension/`) adds those three
Min-optimized flat metrics and MUST land before FSTR-02's full-loss oracle (FL-04) can pass. FSTR-02
may implement FL-01/FL-02/FL-03 against RMSE+Logloss first, but its FL-04 loss set (MAE/MAPE/Quantile)
is gated on the prerequisite. **Loss source (Q1 resolved):** `cb_model::Model` does NOT store the
trained loss name, so the facade `feature_importance_with_data` takes the loss/metric name as an
explicit `loss: &str` argument (layering-safe) `[VERIFIED: CODEGRAPH cbm.rs read_class_to_label reads
only class labels from InfoMap; PLAN-CHECK/planner Q1]`.

**Non-goals (explicit):**
- CTR / categorical models (needs CTR-aware SHAP — a separate hidden dependency); numeric only.
- **Max-optimized** metrics (AUC, Accuracy, R²) where the importance sign inverts — first slice
  supports Min-optimized final errors only; Max metrics are a typed "unsupported loss" error.
- Multi-dimension (multiclass) LossFunctionChange.
- Changing the default Logloss behavior for existing binary models (must stay ≤1e-5 identical).

## 3. Dependencies

- `cb_model::fstr::{loss_function_change (signature change), shap_values, logloss_final_error}`
  `[VERIFIED: CODEGRAPH fstr.rs:788-848]`.
- `cb_core::sum_f64` (already used) `[VERIFIED: CODEGRAPH fstr.rs]`.
- Facade: `cb_train::EvalMetric` + `cb_train::calc_metrics::calc_metric` for the closure body;
  the model's trained loss name (from params/model_info). The facade already calls
  `cb_model::loss_function_change` in `feature_importance_with_data`
  `[VERIFIED: CODEGRAPH crates/catboost-rs/src/model.rs:177-191]`.
- How the facade learns the model's loss: TBD — either the loss is passed by the caller, or read
  from the model's stored params. **[UNVERIFIED: does cb_model::Model expose the trained loss
  name?]** Resolve at plan time via CodeGraph on `Model` / `model_info` / InfoMap; if absent,
  the facade method takes the loss/metric name as an explicit argument (see §9 Q1).
- Oracle: existing `crates/cb-oracle/fixtures/fstr_loss_change/gen_fixtures.py` extended per loss
  `[VERIFIED: LOCAL crates/cb-oracle/fixtures/fstr_loss_change/gen_fixtures.py]`.

## 4. Typed contracts

```rust
// crates/cb-model/src/fstr.rs  (CHANGED signature — metric-agnostic)

/// Per-feature LossFunctionChange importance. `final_error(approx, labels)` computes the
/// model metric's `GetFinalError` for a set of raw approxes; the caller supplies it so this
/// crate stays free of the metric implementations (cb-train). The metric MUST be Min-optimized
/// (smaller == better) so `score = final_error(approx − shap_f) − final_error(approx)` is the
/// importance verbatim; Max-optimized metrics are the caller's responsibility to reject.
#[must_use]
pub fn loss_function_change<F: Fn(&[f64], &[f64]) -> f64>(
    model: &Model,
    cols: &[Vec<f32>],
    labels: &[f64],
    n_features: usize,
    final_error: F,
) -> Vec<f64>;
```

Back-compat: keep a thin `loss_function_change_logloss(...)` (or a default) that calls the
generic form with `logloss_final_error`, so the current Logloss path and its callers/tests stay
green byte-for-byte `[VERIFIED: CODEGRAPH fstr.rs:788 caller in lib.rs; fstr_oracle_test.rs]`.

Facade:
```rust
// crates/catboost-rs/src/model.rs  (feature_importance_with_data LossFunctionChange arm)
// picks the Min-optimized final-error closure for the model's numeric loss and passes it in;
// returns CatBoostError::Unsupported (or Train) for a Max-optimized / out-of-scope loss.
```

## 5. Failure-isolated behavioral specifications

### FL-01 — Injected final-error closure (metric-agnostic core)
- **Responsibility:** `loss_function_change` computes importance using the supplied closure, not a
  hard-coded Logloss.
- **Input:** `model`, `cols`, `labels`, `n_features`, `final_error: F`.
- **Output:** `Vec<f64>` length `n_features`; `score_f == final_error(approx−shap_f) − final_error(approx)`.
- **Given/When/Then:** Given a closure that returns RMSE final error; When called; Then each score
  equals the RMSE degradation from removing feature f (verified against a hand-computed value on a tiny model).
- **Acceptance:** unit `loss_change_uses_injected_final_error` in `fstr_test.rs`.
- **Out of scope:** facade loss selection (FL-03).

### FL-02 — Logloss back-compat unchanged
- **Responsibility:** the default/Logloss path yields byte-identical results to the pre-change fn.
- **Input:** binary model + labels; the retained `logloss_final_error`.
- **Output:** identical to today's `loss_function_change` output.
- **Given/When/Then:** Given the existing `fstr_oracle_test` Logloss fixture; When run through the
  refactored code; Then results are unchanged (≤1e-5, in practice bit-identical).
- **Acceptance:** existing `crates/cb-model/tests/fstr_oracle_test.rs` stays green (regression gate).

### FL-03 — Facade selects the model's numeric loss (Min-optimized only)
- **Responsibility:** the facade maps the model's trained loss → the matching `EvalMetric` final-error
  closure; rejects Max-optimized / unsupported losses with a typed error.
- **Input:** `FeatureImportanceType::LossFunctionChange`, `pool` (+ loss name — source per §3/Q1).
- **Output:** `Vec<(usize,usize,f64)>` on a supported loss; `Err(CatBoostError::…)` otherwise.
- **Given/When/Then:** Given an RMSE-trained numeric model; When `feature_importance_with_data`;
  Then the closure is RMSE and the scores match upstream ≤1e-5. Given an AUC-optimized request;
  Then a typed unsupported-loss error (no silent Logloss fallback).
- **Acceptance:** facade test `loss_change_rmse_facade` + `loss_change_rejects_max_metric`.

### FL-04 — Oracle parity per numeric loss
- **Responsibility:** ≤1e-5 parity vs `get_feature_importance('LossFunctionChange', data=pool)` for
  each in-scope loss (RMSE, MAE, Quantile, MAPE at least).
- **Input:** frozen numeric `.cbm` per loss + pool + labels; upstream importance frozen to expected.
- **Output:** Rust scores match ≤1e-5 per loss.
- **Given/When/Then:** Given each per-loss fixture; When computed; Then max|diff| ≤ 1e-5.
- **Acceptance:** `crates/cb-model/tests/fstr_oracle_test.rs` (or a new
  `fstr_loss_change_multi_oracle_test.rs`) over extended `fstr_loss_change/` fixtures.

## 6. Acceptance scenarios

1. RMSE numeric model → importances match upstream ≤1e-5 (FL-01/FL-03/FL-04).
2. MAE / Quantile / MAPE numeric models → match ≤1e-5 (FL-04).
3. Existing binary Logloss model → unchanged from today (FL-02).
4. AUC-optimized request → typed unsupported error, not a wrong Logloss number (FL-03).

## 7. Impact scope

- **local (cb-model):** signature change to `loss_function_change` (+ retained Logloss wrapper);
  `fstr.rs` edit — **conflicts with the uncommitted FSTR-01 change to the same file**
  `[VERIFIED: LOCAL git status M crates/cb-model/src/fstr.rs]`. **Ordering: implement AFTER FSTR-01
  lands.** `lib.rs` re-export update if the public signature changes.
- **cross-module (facade):** `feature_importance_with_data` LossFunctionChange arm now selects a
  metric closure; needs the model's loss name (source per §3). Callers of the changed cb-model fn:
  `crates/cb-model/src/lib.rs` (re-export) and the facade — update both; verify via CodeGraph
  blast radius (`loss_function_change` has 2 callers `[VERIFIED: CODEGRAPH blast radius]`).
- **tests:** `fstr_test.rs` (unit), `fstr_oracle_test.rs` (regression + new per-loss), fixture gen.
- No schema/persistence/event impact.

## 8. Compatibility and migration

The public `loss_function_change` signature CHANGES (adds a closure param). Because it has only 2
in-repo callers `[VERIFIED: CODEGRAPH]`, migrate both in the same change; retain a Logloss-defaulted
convenience wrapper so no behavior regresses. No wire/schema change.

## 9. Risks and open questions

- **R1 (merge conflict):** hard-blocks parallel work with in-flight FSTR-01 on `fstr.rs`. Gate on
  FSTR-01 landing. `[VERIFIED: LOCAL git status]`
- **R2 (sign/direction):** Max-optimized metrics invert the importance sign. First slice = Min only,
  with an explicit reject for Max metrics. Do NOT silently apply Logloss.
- **Q1 (loss source):** does `cb_model::Model` expose the trained loss/metric name (params / InfoMap)?
  `[UNVERIFIED]` — resolve via CodeGraph at plan time. If not, the facade method must accept the
  loss/metric name explicitly (still layering-safe). Decide before FL-03.
- **R3 (weights):** upstream `GetFinalError` is weighted; first slice assumes unit weights (as the
  current Logloss path does). Note weighted pools as out of scope until a weighted fixture exists.

## 10. Traceability and sources

- Research: `.planning/plans/next-feature-research/research.md` §2 Candidate 3, §4.
- CodeGraph: `cb-model/src/fstr.rs:{214 pvc, 788-848 loss_function_change/logloss_final_error}`,
  `catboost-rs/src/model.rs:169-193`, `cb-train/src/metrics.rs:64/eval`,
  `cb-train/src/calc_metrics.rs:281 calc_metric`.
- Local: `crates/cb-model/Cargo.toml:26,63` (cb-model → cb-train is a NORMAL dep; cb-train → cb-model is dev-only — the corrected layering fact),
  `crates/cb-oracle/fixtures/fstr_loss_change/gen_fixtures.py`.
- Memory: fstr01-ord06-ord07 chain (FSTR-01 in flight), fstr03 plan (clippy/test-mount/oracle recipe).
