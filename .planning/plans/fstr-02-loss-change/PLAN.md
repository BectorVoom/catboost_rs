---
title: FSTR-02 — multi-loss LossFunctionChange (TDD implementation plan)
plan_of: .planning/plans/fstr-02-loss-change/SPEC.md
status: complete
format: markdown
plan_version: 3
updated_at: 2026-07-19T00:00:00Z
gsd_used: false
spec_ids: [FL-01, FL-02, FL-03, FL-04]
supported_losses: [RMSE, MAE, MAPE, Quantile, Logloss]
treefinder_pending:
  collection: UNRESOLVED
  document_id: UNRESOLVED
  note: "TreeFinder MCP unavailable in this session; local SPEC + this PLAN are the authoritative drafts."
ordering_gate:
  wave: 0
  prerequisites:
    - id: GATE-A
      condition: "In-flight FSTR-01 CTR change to crates/cb-model/src/fstr.rs is COMMITTED/landed."
      reason: "This plan edits the same fstr.rs region; sequencing after FSTR-01 avoids merge conflicts."
      blocks: [FL-01, FL-02, FL-03, FL-04a, FL-04b]
    - id: GATE-B
      condition: "eval-metric-extension feature (.planning/plans/eval-metric-extension/) landed — MAE/MAPE/Quantile are Min-optimized flat variants in cb_train::EvalMetric."
      reason: "MAE/MAPE/Quantile are in cb-train DEFERRED_METRICS today; the MAE/MAPE/Quantile facade arms + oracle cannot compile/pass until they exist."
      blocks: [FL-03-mmq-arms, FL-04b]
---

# FSTR-02 — multi-loss LossFunctionChange — TDD Plan (v3)

> ## Execution status (2026-07-19) — ✅ COMPLETE (FL-01..FL-04b), verified green
> Both Wave-0 gates were satisfied in the working tree (FSTR-01 `fstr.rs` change
> present; `eval-metric-extension` landed + green), so the FULL loss set shipped:
> - **FL-01 (T1):** `loss_function_change` now takes a `Fn(&[f64],&[f64])->f64`
>   final-error closure; `loss_function_change_logloss` wrapper retained;
>   `lib.rs` re-exports it. Unit `loss_change_uses_injected_final_error` green.
> - **FL-02 (T2):** `fstr_oracle_test` Logloss call migrated to the wrapper;
>   binclf LFC parity unchanged.
> - **FL-03 (T3):** facade `feature_importance_with_data` gained `loss: &str`; a
>   private `eval_metric_for_loss` allow-list `{RMSE,MAE,MAPE,Quantile,Logloss}`
>   → `EvalMetric` closure; `CatBoostError::UnsupportedLoss(String)` added +
>   `catboost-rs-py` `to_pyerr` arm + `errors_test`. Facade
>   `fstr_loss_change_facade_test` (3 tests) green; a Max/ranking/MSLE/unknown/
>   bad-param loss → typed `UnsupportedLoss` (no silent fallback).
> - **FL-04a/b (T4a/T4b):** per-loss oblivious REGRESSOR `.cbm` fixtures
>   (RMSE/MAE/MAPE/Quantile:alpha=0.5) generated from `catboost==1.2.10`; the
>   generalized core reproduces each upstream `LossFunctionChange` vector ≤1e-5
>   via INDEPENDENT hand-written closures (cb-model) AND via `EvalMetric`
>   (facade). Cross-check: Quantile(0.5) LFC == 0.5·MAE LFC exactly.
> - **Gates:** `cargo test -p cb-model --lib` 102/0; `--test fstr_oracle_test`
>   10/0 (4 new per-loss); `cargo test -p catboost-rs` 18/0 (3 facade LFC);
>   `cargo clippy -p cb-model --lib` + `-p catboost-rs --lib` 0 errors;
>   `cargo check -p catboost-rs-py` clean (E0004 trap closed).
> - **py runtime tests** blocked ONLY by the pre-existing `python3.14` link
>   (`mold: library not found: python3.14`) — environmental, not this change;
>   `cargo check` is the authoritative exhaustiveness gate.
> - **Files:** `crates/cb-model/src/{fstr,fstr_test,lib}.rs`,
>   `crates/cb-model/tests/fstr_oracle_test.rs`,
>   `crates/catboost-rs/src/{model,error}.rs`,
>   `crates/catboost-rs/tests/fstr_loss_change_facade_test.rs`,
>   `crates/catboost-rs-py/src/{errors,errors_test}.rs`,
>   `crates/cb-oracle/fixtures/fstr_loss_change/gen_fixtures.py` + frozen
>   `{rmse,mae,mape,quantile}_{model.cbm,model.json,loss_function_change.npy,X.npy,y.npy}`.
> **Code-review round (2026-07-19, high effort):** 7 verified findings.
> Fixed: [0] py `to_pyerr` UnsupportedLoss now surfaces full `Display` (not the
> bare name); [1] a bad *param* on a supported loss appends the `parse_metric`
> reason to the `UnsupportedLoss` message; [3] the allow-list base name is now
> `trim`med so `" RMSE "` is accepted; [5] dropped the redundant `label().to_vec()`
> (pass `pool.label()` directly). Documented (inherent limits, no code fix
> possible without model loss metadata / a weighted fixture): [2] bare `"Quantile"`
> resolves `alpha=0.5` — caller must pass the trained alpha; [4] per-object pool
> weights are ignored (unit-weight first slice, SPEC R3). Left: [6] `rmse_final_error`
> duplicated across two separate test binaries (clean cross-binary dedup isn't
> possible; isolation acceptable). Added regression tests
> `loss_change_accepts_whitespaced_name` + `loss_change_bad_param_message_carries_parse_reason`.
> Post-fix gates: cb-model 102/0 lib + 10/0 oracle, catboost-rs 20/0, clippy 0/0,
> `cargo check -p catboost-rs-py` clean.
> **Not committed** (working-tree change; awaiting operator's commit decision).

Plan built goal-backward from the acceptance outcomes in
`.planning/plans/fstr-02-loss-change/SPEC.md`. No production code is written here.

**v3 changes (Plan-Checker fixes, verified against source):** (1) the layering premise was wrong
— `cb-model` DOES depend on `cb-train`; the injected closure is now justified as a
decoupling/testability choice, not a circular-dependency workaround (§0, FL-04a). (2) FL-03
facade tests move to a `tests/*.rs` integration file (catboost-rs mounts unit tests via `lib.rs`;
no `model_test.rs`). (3) Quantile wording corrected (absent from `EvalMetric`; only MAE/MAPE are
in `DEFERRED_METRICS`). The task/wave/gate structure, Q1, and the `UnsupportedLoss`/py-trap
decision are unchanged (Checker verified them SOUND).

**v2 change (user decision on B2):** FSTR-02 is NOT narrowed to RMSE-only. The full loss set
`{RMSE, MAE, MAPE, Quantile, Logloss}` is kept; MAE/MAPE/Quantile are delivered by a new
PREREQUISITE feature `eval-metric-extension` (its own planner is running). The Wave-0 gate now
carries TWO prerequisites, and FL-04 is split into an ungated RMSE+Logloss slice (FL-04a) and a
gated MAE/MAPE/Quantile slice (FL-04b).

---

## 0. Verified facts driving this plan (CodeGraph / Read)

- **Decoupling design (a choice, NOT a hard requirement — layering premise corrected in v3).**
  `cb-model` DOES depend on `cb-train` as a normal `[dependencies]` entry
  (`cb-model/Cargo.toml:26`, `default-features = false`; its full dep set is `cb-core` + `cb-data`
  + `cb-train`, NOT "cb-core only"), and it already re-exports `pub use cb_train::Split`
  (`model.rs:31`). So calling `cb_train::EvalMetric` from `fstr.rs` is **NOT circular and IS
  possible** — the earlier "circular dependency" justification was wrong. The **injected
  `Fn(&[f64], &[f64]) -> f64` final-error closure is kept as a deliberate decoupling/testability
  choice**: it keeps `fstr.rs` metric-agnostic (unit-testable without constructing a metric) and
  lets the facade own loss selection. The facade (`catboost-rs`) supplies the closure.
  `[VERIFIED: Read crates/cb-model/Cargo.toml:23-29; crates/cb-model/src/model.rs:31 pub use cb_train::Split]`
- **Current signature is Logloss-hardcoded.** `loss_function_change(model, cols, labels,
  n_features)` calls `logloss_final_error` for base + per-feature.
  `[VERIFIED: CODEGRAPH crates/cb-model/src/fstr.rs:788-848]`
- **Blast radius of `loss_function_change` = 2 in-repo call sites:** the `pub use fstr::{…}`
  re-export in `crates/cb-model/src/lib.rs:38-39`, and the facade
  `feature_importance_with_data` arm in `crates/catboost-rs/src/model.rs:177-191`. Covering
  test `crates/cb-model/tests/fstr_oracle_test.rs`. No PyO3 caller of feature-importance.
  `[VERIFIED: CODEGRAPH blast radius; grep catboost-rs-py]`
- **Facade method is self-contained** — defined/used ONLY in `crates/catboost-rs/src/model.rs`;
  adding a `loss` argument is fully contained. `[VERIFIED: grep -rln feature_importance_with_data]`
- **fstr.rs already mounts its sibling test:** `#[cfg(test)] #[path="fstr_test.rs"] mod tests;`
  at `fstr.rs:854-855`; `fstr_test.rs` exists. `[VERIFIED: Read fstr.rs:854; ls fstr_test.rs]`
- **ORCH-04 flat metrics available TODAY = {RMSE, Logloss, MSLE}** (+ ranking + program-only
  `Custom`). **MAE and MAPE are named in `DEFERRED_METRICS`; Quantile is NOT listed there but is
  likewise absent from `EvalMetric` entirely** — all three are unavailable today (gating stays
  correct).
  `[VERIFIED: CODEGRAPH crates/cb-train/src/calc_metrics.rs:22-24 DEFERRED_METRICS (MAE, MAPE), :151-241
  parse_metric; crates/cb-train/src/metrics.rs:64-151 EvalMetric variants]` → the
  `eval-metric-extension` prerequisite (GATE-B) supplies the missing three.
- **No `is_max_optimal` on `EvalMetric`** (only on the `CustomMetric` trait,
  `cb-compute/src/custom.rs:111`). → the facade rejects out-of-scope/Max-optimized losses via an
  explicit **allow-list**, not a direction query. Satisfies FL-03's "no silent Logloss fallback".

### Q1 — RESOLVED (kept): `cb_model::Model` does NOT store the trained loss name.
`Model` (`model.rs:271-313`) has no loss field; `decode_cbm` reads only `class_to_label` from
`InfoMap` (`cbm.rs:894-929`). → the facade `feature_importance_with_data` takes an explicit
`loss: &str` argument. `[VERIFIED: CODEGRAPH model.rs:271-313; cbm.rs:827-929]`

### Error-variant decision (FL-03) — ADD `CatBoostError::UnsupportedLoss(String)` + py edits.
`CatBoostError` (`crates/catboost-rs/src/error.rs`) is NOT `#[non_exhaustive]`, and
`catboost-rs-py::to_pyerr` (`crates/catboost-rs-py/src/errors.rs:113-135`) is an **exhaustive
`match FacadeError { … }` with no wildcard** → adding a variant is **E0004** in the py crate
until a matching arm is added. `[VERIFIED: Read errors.rs:113-135; ls errors_test.rs → exists]`

**Decision: add the dedicated variant** (an unknown/unsupported loss name is a distinct,
Python-catchable, user-input condition — reusing `FeatureMismatch` would misname it, and
reusing `Train`/`Model` would bury a user-input error as an internal one, breaking the
"no silent fallback / clear typed error" intent). Cost paid in FL-03: add the `to_pyerr` arm
`UnsupportedLoss(m) => CatBoostValueError::new_err(m.clone())` (bad-input value error, mirroring
`FeatureMismatch`/`PartialDependence`), extend `errors_test.rs`, and add
`cargo build -p catboost-rs-py` to FL-03 validation.

---

## 1. Typed contracts

### 1a. cb-model (`crates/cb-model/src/fstr.rs`) — generalized core + retained wrapper
```rust
#[must_use]
pub fn loss_function_change<F: Fn(&[f64], &[f64]) -> f64>(
    model: &Model, cols: &[Vec<f32>], labels: &[f64], n_features: usize, final_error: F,
) -> Vec<f64>;                       // score_f = final_error(approx − shap_f) − final_error(approx)

#[must_use]
pub fn loss_function_change_logloss(
    model: &Model, cols: &[Vec<f32>], labels: &[f64], n_features: usize,
) -> Vec<f64> { loss_function_change(model, cols, labels, n_features, logloss_final_error) }
```
`logloss_final_error` (`fstr.rs:833`) retained as the wrapper's closure. `lib.rs:38-39` adds
`loss_function_change_logloss`. Metric MUST be Min-optimized (caller enforces).

### 1b. facade (`crates/catboost-rs/src/model.rs`) — explicit loss + full allow-list
```rust
pub fn feature_importance_with_data(
    &self, importance_type: FeatureImportanceType, pool: &Pool, loss: &str,   // NEW: Q1
) -> Result<Vec<(usize, usize, f64)>, CatBoostError>;
```
LossFunctionChange arm: `loss` (case-insensitive) → `EvalMetric` via a private helper covering
the FULL set `{RMSE, MAE, MAPE, Quantile, Logloss}`; validate the base call with `?`, then pass
a non-panicking closure `|approx, labels| metric.eval(approx, labels, &[]).unwrap_or(f64::NAN)`
(or `calc_metric(&metric, labels, approx, &[], &[])`) into `cb_model::loss_function_change`.
Unknown / Max-optimized / ranking loss → `Err(CatBoostError::UnsupportedLoss(loss.into()))`.
RMSE+Logloss arms compile today; MAE/MAPE/Quantile arms reference EvalMetric variants that exist
only post-GATE-B (delivered as `FL-03-mmq-arms`).

### 1c. facade error (`crates/catboost-rs/src/error.rs`) — new variant
```rust
#[error("unsupported loss for LossFunctionChange: {0}")]
UnsupportedLoss(String),
```

---

## 2. Task graph (execution waves)

```text
Wave 0  GATE-A: FSTR-01 CTR fstr.rs landed        (blocks FL-01, FL-02, FL-03, FL-04a, FL-04b)
        GATE-B: eval-metric-extension landed       (blocks FL-03 MAE/MAPE/Quantile arms, FL-04b)
                         │
Wave 1  TASK-FL-01-CORE  (cb-model generalize + wrapper + lib + unit)       [needs GATE-A]
                         │  (same crate/files → sequential)
        TASK-FL-02-REGRESSION  (migrate fstr_oracle_test Logloss → wrapper)  [needs GATE-A]
                         │
Wave 2  TASK-FL-03-FACADE  ∥  TASK-FL-04a-ORACLE                             [both need FL-01]
          │  (RMSE+Logloss arms + UnsupportedLoss + py trap;                  (FL-04a: RMSE+Logloss)
          │   MAE/MAPE/Quantile arms sub-part FL-03-mmq-arms needs GATE-B)
                         │
Wave 3  TASK-FL-04b-ORACLE  (MAE/MAPE/Quantile fixtures + oracle)   [needs FL-01 + GATE-B + FL-03-mmq-arms]
```

Spec → task coverage: **FL-01 → T1 · FL-02 → T2 · FL-03 → T3 (incl. FL-03-mmq-arms) ·
FL-04 → T4a + T4b**. Every FL id covered; graph acyclic.

---

## 3. TASK-FL-01-CORE — inject the final-error closure (FL-01)

- **Goal / completion:** `loss_function_change` takes a `Fn(&[f64],&[f64])->f64` closure used
  for both base and per-feature errors; `loss_function_change_logloss` reproduces today's
  behavior; a unit test proves the closure (not hard-coded Logloss) drives the result.
  `cargo test -p cb-model --lib` green; clippy clean on changed code.
- **Prerequisites:** GATE-A (FSTR-01 committed); Q1 resolved (done).
- **Files:** Modify `crates/cb-model/src/fstr.rs` (generalize + wrapper; keep
  `logloss_final_error`); Modify `crates/cb-model/src/lib.rs:38-39` (re-export wrapper);
  Test `crates/cb-model/src/fstr_test.rs` (mounted at `fstr.rs:854-855`).
- **TDD:**
  1. **Red** — `loss_change_uses_injected_final_error` in `fstr_test.rs`: tiny numeric `Model`,
     RMSE-style closure `sqrt(mean((a-t)^2))`, assert each score == `final_error(approx−shap_f)
     − final_error(approx)` recomputed in-test. Fails vs the old 4-arg Logloss-only fn.
     Run: `cargo test -p cb-model --lib loss_change_uses_injected`.
  2. **Green** — add `final_error: F`; replace the two `logloss_final_error(...)` calls with
     `final_error(...)`; add wrapper; update re-export. Same command.
  3. **Refactor** — reductions stay on `cb_core::sum_f64` (D-08); no `unwrap`/`expect`/indexing.
     Run: `cargo clippy -p cb-model --lib --no-deps`.
  4. **Verify** — `cargo test -p cb-model --lib`.
- **Completion:** [ ] Red fails for "result Logloss-derived, not closure-derived". [ ] closure
  unit test green. [ ] wrapper exists + re-exported. [ ] clippy clean.
- **Guardrail:** do NOT touch the FSTR-01 CTR PVC/interaction paths in the same file.

---

## 4. TASK-FL-02-REGRESSION — Logloss back-compat unchanged (FL-02)  [REGRESSION GATE]

- **Goal / completion:** existing Logloss oracle stays green + bit-identical after the signature
  change; the oracle test's calls migrate to `loss_function_change_logloss`; parity ≤1e-5.
- **Prerequisites:** TASK-FL-01-CORE (same crate/file; sequential).
- **Files:** Modify `crates/cb-model/tests/fstr_oracle_test.rs` (change the
  `loss_function_change(model, cols, labels, n)` call sites → `loss_function_change_logloss(...)`;
  import updated).
- **TDD:**
  1. **Red** — after FL-01, the 4-arg call no longer compiles; that compile failure is the red
     signal pinpointing the back-compat surface. Run: `cargo test -p cb-model --test fstr_oracle_test`.
  2. **Green** — migrate call sites to the wrapper; numeric assertions unchanged. Same command.
  3. **Refactor** — none (test-only); keep the test `#![allow(clippy::…)]` header.
  4. **Verify** — `cargo test -p cb-model --test fstr_oracle_test` (oblivious + non-symmetric LFC ≤1e-5).
- **Completion:** [ ] compiles only via the retained wrapper. [ ] all prior LFC assertions ≤1e-5.
  [ ] no fixture regenerated (frozen-fixture rule).

---

## 5. TASK-FL-03-FACADE — facade selects the loss, rejects the rest (FL-03)

- **Goal / completion:** `feature_importance_with_data(.., loss)` maps the FULL Min-optimized set
  `{RMSE, MAE, MAPE, Quantile, Logloss}` → the matching `EvalMetric` closure and returns scores;
  an unknown/Max-optimized loss → `CatBoostError::UnsupportedLoss` (never a silent Logloss
  number). RMSE+Logloss arms + the reject path land now; the MAE/MAPE/Quantile arms
  (**sub-part FL-03-mmq-arms**) land with GATE-B.
- **Prerequisites:** TASK-FL-01-CORE (new cb-model signature). FL-03-mmq-arms additionally needs
  GATE-B. Independent of T2/T4 (disjoint files) → parallel within Wave 2.
- **Files:**
  - Modify `crates/catboost-rs/src/model.rs:169-193` (add `loss: &str`; allow-list helper;
    closure via `metric.eval`/`calc_metric`; reject arm).
  - Modify `crates/catboost-rs/src/error.rs` (add `UnsupportedLoss(String)`).
  - **Modify `crates/catboost-rs-py/src/errors.rs:113-135`** (add
    `FacadeError::UnsupportedLoss(m) => CatBoostValueError::new_err(m.clone())` — the E0004 trap fix).
  - Test `crates/catboost-rs-py/src/errors_test.rs` (assert `UnsupportedLoss` → `CatBoostValueError`).
  - Test **`crates/catboost-rs/tests/fstr_loss_change_facade_test.rs`** (NEW integration-test file).
    This is the facade's established pattern for data-bearing tests — matching
    `tests/onnx_facade_test.rs`, `tests/partial_dependence_facade_test.rs`, `tests/builder_oracle_test.rs`.
    NOTE: `catboost-rs` mounts UNIT tests via `lib.rs` (`#[cfg(test)] mod error_test; mod metrics_test;
    mod onnx_test;` at `lib.rs:57-62`) and has NO `model_test.rs`, so a sibling `#[path]` mount in
    `model.rs` would silently run 0 tests — use the `tests/` integration file instead. Cases:
    `loss_change_rmse_facade`, `loss_change_rejects_max_metric`.
- **TDD:**
  1. **Red** — `loss_change_rejects_max_metric`: `feature_importance_with_data(LossFunctionChange,
     &pool, "AUC")` ⇒ `Err(CatBoostError::UnsupportedLoss(_))`. `loss_change_rmse_facade`: load
     frozen `rmse_model.cbm` (FL-04a fixtures; gate the numeric assert on fixture presence like
     `fstr_oracle_test.rs`) ⇒ scores match `rmse_loss_function_change.npy` ≤1e-5. Fails: no `loss`
     arg / no `UnsupportedLoss` variant → and `cargo build -p catboost-rs-py` fails E0004 until the
     `to_pyerr` arm is added. Run: `cargo test -p catboost-rs` + `cargo build -p catboost-rs-py`.
  2. **Green** — add `loss` arg + `UnsupportedLoss` variant + the `to_pyerr` arm; build the
     allow-list helper (RMSE+Logloss active now; MAE/MAPE/Quantile arms behind FL-03-mmq-arms);
     validate the base call with `?`; non-panicking closure. Migrate the single existing call site.
     Run: `cargo test -p catboost-rs` + `cargo build -p catboost-rs-py`.
  3. **Refactor** — factor name→`EvalMetric` into a private `Result<EvalMetric, CatBoostError>`
     helper. Run: `cargo clippy -p catboost-rs --lib --no-deps`.
  4. **Verify** — `cargo test -p catboost-rs`; `cargo build -p catboost-rs-py`;
     `cargo test -p catboost-rs-py errors` (py error mapping).
- **FL-03-mmq-arms (gated on GATE-B):** add the `MAE`/`MAPE`/`Quantile` name→`EvalMetric` arms
  once those variants exist; extend `loss_change_*_facade` asserts to those losses (uses FL-04b
  fixtures). Run: `cargo test -p catboost-rs`.
- **Completion:** [ ] `UnsupportedLoss` added + `to_pyerr` arm + `errors_test.rs` green +
  `cargo build -p catboost-rs-py` passes. [ ] a Max/unknown loss ⇒ `UnsupportedLoss` (no
  fallback). [ ] RMSE+Logloss route correctly; RMSE facade parity ≤1e-5. [ ] (post-GATE-B)
  MAE/MAPE/Quantile arms active. [ ] clippy clean.
- **Guardrail:** allow-list is exactly `{RMSE, MAE, MAPE, Quantile, Logloss}`; MAE/MAPE/Quantile
  arms compile only post-GATE-B; everything else falls into the reject arm.

---

## 6. TASK-FL-04a-ORACLE — RMSE + Logloss numeric parity (FL-04a, ungated)

- **Goal / completion:** a frozen RMSE-trained numeric `.cbm` + its upstream
  `get_feature_importance('LossFunctionChange', data=pool)` vector exist under
  `crates/cb-oracle/fixtures/fstr_loss_change/`; a cb-model oracle test reproduces RMSE ≤1e-5 via
  a hand-written RMSE closure `sqrt(sum_f64(sq)/n)` passed to the generalized
  `loss_function_change`. (cb-model tests CAN use `cb-train` — it is a normal dependency — but a
  hand-written closure is PREFERRED here as an INDEPENDENT reimplementation: a stronger oracle
  than routing through the very same `cb_train::EvalMetric::Rmse` the facade uses.) Logloss parity
  already covered by FL-02 in the same binary.
- **Prerequisites:** TASK-FL-01-CORE. Only GATE-A (no GATE-B). Parallel with T3.
- **Files:**
  - Modify `crates/cb-oracle/fixtures/fstr_loss_change/gen_fixtures.py` — add `gen_rmse_lfc()`:
    `CatBoostRegressor(loss_function="RMSE", thread_count=1, bootstrap_type="No", random_seed=0,
    **isolating params)` on `numeric_tiny`; save `rmse_model.cbm`, `rmse_X.npy`, `rmse_y.npy`,
    `rmse_loss_function_change.npy`; extend `config.json` + `artifacts`.
  - Create/commit FROZEN: `rmse_model.cbm`, `rmse_X.npy`, `rmse_y.npy`,
    `rmse_loss_function_change.npy`.
  - Test `crates/cb-model/tests/fstr_oracle_test.rs` — add
    `loss_function_change_rmse_matches_upstream_within_tol` (load via `load_cbm`; RMSE closure
    `sqrt(sum_f64(sq)/n)`; compare ≤1e-5; gate on fixture presence).
- **Oracle env (one-time, frozen):**
  `uv venv --python 3.12 && uv pip install catboost==1.2.10 'numpy<2'` then
  `.venv/bin/python crates/cb-oracle/fixtures/fstr_loss_change/gen_fixtures.py`.
- **TDD:** Red (test references missing fixtures → fails) → Green (generate+commit frozen
  artifacts, implement independent RMSE closure) → Refactor (share `model_from_cbm` helper) →
  Verify `cargo test -p cb-model --test fstr_oracle_test` (RMSE ≤1e-5 AND FL-02 Logloss still green).
- **Completion:** [ ] frozen RMSE fixtures committed + in `config.json`. [ ] RMSE parity ≤1e-5.
  [ ] Logloss (FL-02) still green in the same binary.
- **Guardrail:** numeric-only, unit weights (SPEC R3); no CTR/categorical fixture (SPEC §2).

---

## 7. TASK-FL-04b-ORACLE — MAE/MAPE/Quantile parity (FL-04b, gated on GATE-B)

- **Goal / completion:** frozen MAE / MAPE / Quantile-trained numeric `.cbm` models + their
  upstream LossFunctionChange vectors exist; the oracle reproduces each ≤1e-5. In cb-model this
  needs a hand-written MAE/MAPE/Quantile final-error closure; in the facade (FL-03-mmq-arms) it
  routes through the newly-added `EvalMetric` variants.
- **Prerequisites:** TASK-FL-01-CORE **+ GATE-B** (MAE/MAPE/Quantile in `EvalMetric`) **+
  FL-03-mmq-arms** (facade arms). Blocked until `eval-metric-extension` lands.
- **Files:**
  - Modify `gen_fixtures.py` — add `gen_mae_lfc()` / `gen_mape_lfc()` / `gen_quantile_lfc()`
    (`loss_function="MAE"` / `"MAPE"` / `"Quantile:alpha=0.5"`; frozen `.cbm` + `*_X/_y` +
    `*_loss_function_change.npy`); extend `config.json` + `artifacts`.
  - Create/commit FROZEN per-loss `.cbm` + `.npy`.
  - Test `crates/cb-model/tests/fstr_oracle_test.rs` (or new
    `fstr_loss_change_multi_oracle_test.rs`) — one `..._matches_upstream_within_tol` per loss with
    the matching hand-written final-error closure; ≤1e-5; gate on fixture presence.
  - Test (facade) — extend FL-03's facade asserts to MAE/MAPE/Quantile via the new arms.
- **TDD:** Red (per-loss test/facade arm missing) → Green (generate+commit frozen fixtures,
  add the EvalMetric-routed facade arms + independent cb-model closures) → Refactor (table-drive
  the per-loss oracle) → Verify `cargo test -p cb-model --test fstr_oracle_test` +
  `cargo test -p catboost-rs`.
- **Completion:** [ ] frozen MAE/MAPE/Quantile fixtures committed. [ ] each parity ≤1e-5 in both
  cb-model (closure) and facade (EvalMetric) paths. [ ] FL-04a + FL-02 still green.
- **Guardrail:** Quantile alpha must match the fixture's trained alpha; numeric-only, unit weights.

---

## 8. Blockers / open questions

- **GATE-A (VERIFIED, active):** uncommitted FSTR-01 CTR change to `fstr.rs`; every fstr.rs edit
  waits until it lands. `[VERIFIED: git status M crates/cb-model/src/fstr.rs]`
- **GATE-B (VERIFIED prerequisite, tracked externally):** MAE/MAPE are named in cb-train
  `DEFERRED_METRICS` and Quantile is absent from `EvalMetric` entirely — the
  `eval-metric-extension` feature must land first to unblock FL-03-mmq-arms + FL-04b.
  `[VERIFIED: CODEGRAPH calc_metrics.rs:22-24; metrics.rs:64-151]` — **user has approved
  this prerequisite path (B2 is a dependency, not a narrowing).**
- **Q1 — RESOLVED:** `Model` does not expose the loss name → facade takes explicit `loss: &str`.
- **Error-variant — DECIDED:** add `CatBoostError::UnsupportedLoss(String)` + `to_pyerr` arm →
  `CatBoostValueError` + `errors_test.rs` + `cargo build -p catboost-rs-py`. (Reuse of an
  existing variant rejected: `FeatureMismatch` misnames it; `Train`/`Model` bury a user-input
  error as internal.)

---

## 9. Validation command reference

- `cargo test -p cb-model --lib`                          (FL-01 unit)
- `cargo test -p cb-model --test fstr_oracle_test`        (FL-02 regression + FL-04a/b oracle)
- `cargo clippy -p cb-model --lib --no-deps`              (deny-lints, changed code)
- `cargo test -p catboost-rs`                             (FL-03 facade + reject)
- `cargo clippy -p catboost-rs --lib --no-deps`           (deny-lints, facade)
- `cargo build -p catboost-rs-py`                         (FL-03 E0004 trap — MUST pass)
- `cargo test -p catboost-rs-py errors`                   (FL-03 py error mapping)
- Oracle env: `uv venv --python 3.12 && uv pip install catboost==1.2.10 'numpy<2'`

Lint gate is **clippy, not build** (scope with `-p <crate> --no-deps`); the ONE build gate that
matters here is `cargo build -p catboost-rs-py` (the exhaustive-match trap). Test-location
conventions DIFFER per crate: **cb-model** unit tests live in a mounted sibling `*_test.rs`
(verify the `#[cfg(test)] #[path=…] mod tests;` mount in the prod file, else 0 tests run
silently — e.g. `fstr.rs:854-855`); **catboost-rs** mounts unit tests via `lib.rs`
(`mod …_test;`, `lib.rs:57-62`) and puts data-bearing FACADE tests in `tests/*.rs` integration
files (FL-03 uses `tests/fstr_loss_change_facade_test.rs`).

---

## 10. Traceability

- SPEC: `.planning/plans/fstr-02-loss-change/SPEC.md` (FL-01..FL-04; frontmatter now lists both
  prerequisites; §2 keeps the full loss set with the `eval-metric-extension` prerequisite).
- Prerequisite: `.planning/plans/eval-metric-extension/SPEC.md` (adds MAE/MAPE/Quantile to
  `cb_train::EvalMetric`).
- Research: `.planning/plans/next-feature-research/research.md` §2 Candidate 3, §4, §5.
- Code evidence: `crates/cb-model/Cargo.toml:23-29` (dep set = cb-core + cb-data + cb-train);
  `crates/cb-model/src/fstr.rs:{788-848, 854-855}`; `crates/cb-model/src/lib.rs:38-39`;
  `crates/cb-model/src/model.rs:{31 pub use cb_train::Split, 271-313}`;
  `crates/cb-model/src/cbm.rs:{827,894-929}`;
  `crates/catboost-rs/src/model.rs:169-193`; `crates/catboost-rs/src/error.rs`;
  `crates/catboost-rs/src/lib.rs:57-62` (unit-test mounts); `crates/catboost-rs/tests/` (facade
  integration-test precedent: onnx_facade_test / partial_dependence_facade_test / builder_oracle_test);
  `crates/catboost-rs-py/src/errors.rs:113-135`; `crates/catboost-rs-py/src/errors_test.rs`;
  `crates/cb-train/src/calc_metrics.rs:{22-24,151-241,281-293}`;
  `crates/cb-train/src/metrics.rs:64-151`;
  `crates/cb-oracle/fixtures/fstr_loss_change/gen_fixtures.py`.
- GSD: **not used** (no GSD skill/command/workflow/agent invoked).
