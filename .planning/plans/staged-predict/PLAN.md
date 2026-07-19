---
title: staged_predict — TDD implementation plan
plan_for: .planning/plans/staged-predict/SPEC.md
status: draft
format: markdown
updated_at: 2026-07-18T00:00:00Z
gsd_used: false
spec_ids: [SP-01, SP-02, SP-03, SP-04]
tasks: [T1-STAGED-CORE, T2-STAGED-SCHEDULE, T3-STAGED-FACADE, T4-STAGED-ORACLE]
---

# staged_predict — TDD Implementation Plan

Goal-backward plan for the draft SPEC `staged_predict — per-tree-prefix cumulative
prediction`. Each task is one Red→Green→Refactor cycle over a single failure-isolated
behavior. This is a **plan only** — no production code is authored here. No GSD
skill/command/workflow/agent was used to produce it.

## Verified facts (CodeGraph + Read, this session)

- `predict_raw_one(model, features, cat_values) -> f64` sums
  `bias + Σ_tree tree.leaf_values[leaf_index_for(...)]`
  `[VERIFIED: CODEGRAPH crates/cb-model/src/apply.rs:318-355]`. The oblivious arm is a
  `map` over `model.oblivious_trees` reading `tree.leaf_values.get(leaf).copied().unwrap_or(0.0)`.
- `leaf_index_for(model, tree, features, cat_values) -> usize` is a **private** fn in
  `apply.rs` (`:208-215`), dimension-agnostic, reusable in-module
  `[VERIFIED: CODEGRAPH apply.rs:208]`.
- `predict_raw` (`:370`) / `predict_raw_cat` (`:386`) are the public scalar apply; scalar
  models take the byte-identical `predict_raw_one` path (`approx_dimension <= 1`)
  `[VERIFIED: CODEGRAPH apply.rs:394]`. These MUST stay byte-identical (additive change only).
- `sum_f64` is the order-locked accumulator used by every apply path
  `[VERIFIED: CODEGRAPH apply.rs:354 uses sum_f64]`.
- `apply.rs` mounts its unit tests via `#[cfg(test)] #[path = "region_apply_test.rs"] mod
  region_apply_test;` at the file tail (`:812-814`)
  `[VERIFIED: READ apply.rs:812-814]`. There is **no** `apply_test.rs` sibling today
  `[VERIFIED: LOCAL ls]` — a new mount block is required or 0 tests run.
- `lib.rs` re-exports the apply surface at `:26-30`
  (`pub use apply::{ ... predict_raw, predict_raw_cat, predict_raw_multi, ... };`)
  `[VERIFIED: READ lib.rs:26-30]`.
- `cb_model::Model` fields are **all `pub`**: `oblivious_trees`, `non_symmetric_trees`,
  `region_trees`, `bias`, `float_feature_borders`, `ctr_data: Option<CtrData>`,
  `approx_dimension`, `class_to_label` `[VERIFIED: READ model.rs:271-313]`. The facade can
  read every guard field through `Model::as_canonical() -> &cb_model::Model`.
- Facade `catboost_rs::Model` wraps `inner: cb_model::Model`; `feature_columns(&self, pool)`
  checks float-count → `CatBoostError::FeatureMismatch` and narrows to `Vec<Vec<f32>>`
  (`crates/catboost-rs/src/model.rs:60-73`); `predict(pool)` = `predict_with(pool,
  RawFormulaVal)` (`:100-102`); `as_canonical()` at `:45` `[VERIFIED: READ/CODEGRAPH]`.
- `CatBoostError` (facade) variants today: `Train`, `Model`, `Io`, `Deserialize`,
  `SchemaVersion`, `FeatureMismatch`, `PartialDependence`, `Export`
  `[VERIFIED: READ crates/catboost-rs/src/error.rs:33-85]` — no "unsupported model" variant yet.
- Oracle harness: `cb_oracle::{assert_abs_close, load_f64_vec}`, `ndarray_npy::read_npy`,
  fixtures resolved via `env!("CARGO_MANIFEST_DIR")/../cb-oracle/fixtures/...`; PDP oracle test
  is the closest template (loads a frozen `model.cbm`, reads `inputs/numeric_tiny/X.npy`,
  compares to `*.npy` at `TOL = 1e-5`) `[VERIFIED: READ partial_dependence_oracle_test.rs]`.
- Fixture generators are pinned-seed offline Python (`catboost==1.2.10`, `thread_count=1`,
  `bootstrap_type="No"`), run from a 3.12 venv `[VERIFIED: READ fstr_loss_change/gen_fixtures.py]`.

## Conventions enforced by every task

- **No production code in this plan.** Tasks describe edits; the implementer writes them.
- **`anyhow` banned in `cb-model`** (D-14); use `cb_model::ModelError` / plain returns.
- **Lint gate is clippy, not build**: new code must pass `cargo clippy -p <crate> --lib
  --no-deps` under the workspace deny-lints (no `unwrap`/`expect`/`panic`/`indexing_slicing`).
  Use checked `.get(...).copied().unwrap_or(...)` and `sum_f64`, never raw `[]` indexing.
- **Source/test separation (MANDATORY)**: unit tests live in a sibling `*_test.rs` mounted
  with `#[cfg(test)] #[path = "X_test.rs"] mod tests;` — never `mod tests {}` inline.
- **Additive only**: `predict_raw` / `predict_raw_cat` / `predict_raw_one` stay byte-identical.
- **Frozen-fixture rule**: float-only oblivious model, committed `.cbm` + `.npy`; never
  regenerate at test time (CatBoost quantization is run-to-run nondeterministic).
- **Oracle env**: `uv venv --python 3.12 && uv pip install catboost==1.2.10 'numpy<2'`.

## Validation commands (repository-valid)

- `cargo test -p cb-model --lib` — apply.rs sibling unit tests (T1, T2).
- `cargo test -p cb-model --test staged_predict_oracle_test` — SP-04 integration oracle (T4).
- `cargo clippy -p cb-model --lib --no-deps` — deny-lint gate on new cb-model code (T1, T2).
- `cargo test -p catboost-rs` — facade unit tests (T3).
- `cargo clippy -p catboost-rs --lib --no-deps` — deny-lint gate on facade code (T3).
- `cargo build -p catboost-rs-py` — proves the new `CatBoostError::UnsupportedModel` variant did
  not break the exhaustive `to_pyerr` match (E0004); the py crate is NOT compiled by `cargo test
  -p catboost-rs` (T3).
- `cargo test -p catboost-rs-py --lib` — py `errors_test.rs` `to_pyerr` mapping assertion (T3).

## Execution waves

- **Wave 1:** `T1-STAGED-CORE`
- **Wave 2:** `T2-STAGED-SCHEDULE` (after T1 — same fn/file)
- **Wave 3 (parallel):** `T3-STAGED-FACADE` ∥ `T4-STAGED-ORACLE` (both need the finished
  `predict_raw_staged` + lib re-export from T2; they touch disjoint files — T3 edits
  `catboost-rs/src/{model.rs,error.rs}` + `catboost-rs-py/src/errors{,_test}.rs` + a new
  `catboost-rs/tests/` file; T4 edits only `cb-oracle/fixtures/` + a new `cb-model/tests/`
  file — no write conflict). If T4-R1 corrects T2's schedule, T3's facade tests must be re-run.

```text
T1-STAGED-CORE -> T2-STAGED-SCHEDULE -> T3-STAGED-FACADE
                                     \-> T4-STAGED-ORACLE
```

## Spec-ID → task coverage

| Spec | Task(s) |
|------|---------|
| SP-01 Single-stage prefix == truncated apply | T1-STAGED-CORE |
| SP-02 Stage schedule (start/end/period) | T2-STAGED-SCHEDULE |
| SP-03 Facade wiring + guard + feature-count | T3-STAGED-FACADE |
| SP-04 Oracle parity vs `model.staged_predict` | T4-STAGED-ORACLE |

---

## T1-STAGED-CORE — prefix accumulation equals truncated apply

- **Spec:** SP-01
- **Goal / observable completion:** a new `pub fn predict_raw_staged` exists in `apply.rs`
  and, for a single stage covering the first `k` oblivious trees, returns one row whose
  `row[i] == bias + Σ_{t<k} tree_t.leaf_values[leaf_index_for(t, x_i)]`; at `k == T` the row
  equals `predict_raw` exactly. Unit tests pass; clippy clean.
- **Prerequisites:** none.
- **Files:**
  - Modify: `crates/cb-model/src/apply.rs` — add `predict_raw_staged` (sibling to
    `predict_raw_cat`); add a tail mount block `#[cfg(test)] #[path = "staged_predict_test.rs"]
    mod staged_predict_test;` next to the existing `region_apply_test` mount (`:812-814`).
  - Create: `crates/cb-model/src/staged_predict_test.rs` — the mounted unit-test module.
  - Modify: `crates/cb-model/src/lib.rs:26-30` — add `predict_raw_staged` to the
    `pub use apply::{ ... }` re-export list.
  - Reuse (no edit): private `leaf_index_for` (`:208`), `sum_f64`, `Model.{oblivious_trees,
    bias}`.
- **Signature (from SPEC §4):**
  `pub fn predict_raw_staged(model, feature_values: &[Vec<f32>], ntree_start, ntree_end,
  eval_period) -> Vec<Vec<f64>>`. In T1 exercise it only in the single-stage regime
  (`ntree_start=0, ntree_end=k, eval_period=k`); the full schedule is T2.
- **Red:**
  - Test `staged_prefix_matches_truncated_apply` in `staged_predict_test.rs`.
  - Setup: build a small scalar oblivious `Model` in-test (mirror how `region_apply_test.rs`
    constructs a `Model`; **confirm the exact constructor pattern via CodeGraph
    `codegraph_explore "region_apply_test Model oblivious_trees ObliviousTree"` before writing**),
    with `T` (e.g. 3) oblivious trees and a distinct `bias`.
  - Input: one float row (as SoA `Vec<Vec<f32>>`), `k = T-1`, single stage.
  - Expected: the row equals a hand-rolled prefix sum `bias + Σ_{t<k} leaf_t`; and a second
    assertion with `k = T` equals `predict_raw(&model, &cols)` element-wise (`assert_abs`).
  - Initial failure: `predict_raw_staged` does not exist → compile error (fail closed).
- **Green (minimal intent):** implement the per-object prefix sum by iterating only
  `model.oblivious_trees.iter().take(end)`, gathering each object's row with checked `.get`
  (NaN-pad short columns exactly as `predict_raw_cat` does at `:404-407`), summing per-tree
  leaf values via `sum_f64`, then `+ model.bias` once. Return a single-row `Vec<Vec<f64>>` for
  the single stage. Do NOT implement multi-stage stepping (that is T2). Ignore
  `non_symmetric`/`region` arms (scalar-oblivious scope; guarded at the facade in T3).
- **Refactor:** factor a private `prefix_row(model, &row, &cats, end) -> f64` helper if it
  clarifies reuse; keep `predict_raw` byte-identical. Regression scope: `cargo test -p cb-model
  --lib` (all apply unit tests).
- **Validation:** `cargo test -p cb-model --lib` · `cargo clippy -p cb-model --lib --no-deps`.
- **Completion evidence:** both unit assertions green; clippy reports no new deny-lint;
  `predict_raw`/`predict_raw_cat` diff is add-only.
- **Parallelization:** none (Wave 1, gates T2).

---

## T2-STAGED-SCHEDULE — stage schedule (ntree_start / ntree_end / eval_period)

- **Spec:** SP-02
- **Goal / observable completion:** `predict_raw_staged` produces exactly the upstream stage
  tree-counts and one cumulative row per stage: `ntree_end == 0` ⇒ `oblivious_trees.len()`;
  `eval_period == 0` ⇒ `1`; stages step by `eval_period` and **always include `ntree_end`**;
  `ntree_start >= end` ⇒ empty `Vec`. Each stage row equals a direct prefix apply at that count.
- **Prerequisites:** T1-STAGED-CORE (extends the same fn).
- **Files:**
  - Modify: `crates/cb-model/src/apply.rs` — generalize `predict_raw_staged` from single-stage
    to the full schedule (add the stage-count generator; reuse the T1 per-object prefix helper).
  - Modify: `crates/cb-model/src/staged_predict_test.rs` — add schedule tests.
- **Red:**
  - Test `staged_schedule_boundaries` in `staged_predict_test.rs`.
  - Setup: scalar oblivious `Model` with `T = 10` trees.
  - Input/expected (three sub-cases, one assert block each):
    1. `start=0, end=0, period=3` ⇒ stage tree-counts `{3, 6, 9, 10}` (last == full `end`); each
       row equals a direct prefix apply at that count (compare against a helper that re-sums the
       first `k` trees, or against `predict_raw_staged(.., 0, k, k)` single-stage from T1).
    2. `period=1` ⇒ `T` rows; final row == `predict_raw(&model, &cols)`.
    3. `start >= end` (e.g. `start=10, end=5`) ⇒ empty `Vec`.
  - Initial failure: T1's single-stage impl returns one row regardless of `period` ⇒ sub-case 1
    asserts row-count `4` but gets `1` (fail on count/inclusion, the SP-02 principal reason).
- **Green (minimal intent):** compute `end = if ntree_end == 0 { oblivious_trees.len() } else {
  ntree_end.min(len) }`, `step = eval_period.max(1)`; generate stage counts starting at
  `ntree_start.saturating_add(step)` stepping by `step` while `< end`, then always push `end`;
  if `ntree_start >= end` return `Vec::new()`. For each stage count evaluate every object's
  prefix (reuse the T1 helper) into a row. All arithmetic saturating/checked; no indexing.
- **Refactor:** extract `stage_counts(start, end, step) -> Vec<usize>` as a pure private fn so
  the schedule is independently reasoned about; keep per-object accumulation shared with T1.
  Regression scope: `cargo test -p cb-model --lib`.
- **Validation:** `cargo test -p cb-model --lib` · `cargo clippy -p cb-model --lib --no-deps`.
- **Completion evidence:** all three schedule sub-cases green; T1 test still green;
  `predict_raw` unchanged.
- **Note (R1):** the schedule here encodes the SPEC's *stated* upstream rule (step by period,
  always include `end`). T4 empirically confirms it against real `model.staged_predict` output
  and, if upstream differs (e.g. includes an initial stage or a different first count), T4
  reports the delta and this fn's `stage_counts` is corrected **before T4 is marked done**.
- **Parallelization:** none (Wave 2; gates T3 and T4).

---

## T3-STAGED-FACADE — facade `staged_predict` + scalar-oblivious guard + feature-count

- **Spec:** SP-03 (and Acceptance scenario 3 guard, SPEC §6/R2)
- **Goal / observable completion:** `catboost_rs::Model::staged_predict(pool, ntree_start,
  ntree_end, eval_period)` exists, narrows the pool to f32 columns (reusing `feature_columns`),
  applies `None` defaults (`0/0/1`), rejects a float-count mismatch with
  `CatBoostError::FeatureMismatch`, rejects a non-scalar / non-oblivious / CTR model with a
  typed error, and — for a matching scalar-oblivious pool with defaults — `stages.last()` equals
  `predict(pool)` exactly.
- **Prerequisites:** T2-STAGED-SCHEDULE (needs `cb_model::predict_raw_staged` + lib re-export).
- **Files:**
  - Modify: `crates/catboost-rs/src/model.rs` — add `pub fn staged_predict(&self, pool, start:
    Option<usize>, end: Option<usize>, period: Option<usize>) -> Result<Vec<Vec<f64>>,
    CatBoostError>`; import `predict_raw_staged` in the `use cb_model::{...}` block (`:17-21`).
  - Modify: `crates/catboost-rs/src/error.rs` — add the variant
    `CatBoostError::UnsupportedModel(String)` (thiserror `#[error("unsupported model: {0}")]`)
    for the guard. (Reusing `FeatureMismatch` would be semantically wrong; `Export` is
    ONNX-specific.) **NOT source-compatible downstream:** `CatBoostError` is **not**
    `#[non_exhaustive]`, and `catboost-rs-py::to_pyerr` (`crates/catboost-rs-py/src/errors.rs:113-135`)
    is an **exhaustive `match` with no wildcard** — adding a variant is a compile error (E0004)
    in the `catboost-rs-py` crate `[VERIFIED: READ errors.rs:113-135]`. The two edits below are
    therefore part of this task, not optional.
  - Modify: `crates/catboost-rs-py/src/errors.rs` — add an arm to the exhaustive `to_pyerr`
    match: `FacadeError::UnsupportedModel(m) => CatBoostValueError::new_err(m.clone())`
    (taxonomy: a guard rejection where the model is the bad input — mirrors the `Export`
    guard-rejection variants and `PartialDependence`, which map to `CatBoostValueError`, per
    the `to_pyerr` doc-comment `:105-111`). Without this arm the workspace fails to build.
  - Modify: `crates/catboost-rs-py/src/errors_test.rs` — add an assertion that
    `to_pyerr(&FacadeError::UnsupportedModel("...".into()))` yields a `CatBoostValueError`
    (mirror the existing per-variant `to_pyerr` assertions).
  - Create: `crates/catboost-rs/tests/staged_predict_facade_test.rs` — facade tests (integration
    test dir, mirroring `crates/catboost-rs/tests/partial_dependence_facade_test.rs`).
  - Reuse (no edit): `feature_columns` (`:60`), `as_canonical` (`:45`), `predict` (`:100`).
- **Guard predicate (facade, via `as_canonical()`):** reject when any of
  `inner.approx_dimension > 1`, `!inner.non_symmetric_trees.is_empty()`,
  `!inner.region_trees.is_empty()`, or `inner.ctr_data.is_some()` → `Err(UnsupportedModel(...))`.
  All four fields are `pub` `[VERIFIED: model.rs:271-313]`. Check the guard **before**
  `feature_columns` so the scope error is deterministic. (CTR is excluded because the
  cb-model float path `predict_raw_staged` passes no cat columns; a CTR model would silently
  drop its CTR splits.)
- **Red:**
  - Test A `staged_predict_facade_last_equals_predict`: load/construct a scalar-oblivious facade
    `Model` + matching `Pool`; call `staged_predict(pool, None, None, None)`; assert
    `stages.last().unwrap()` equals `model.predict(pool)?` element-wise (`assert_abs`, tol 0 or
    1e-12). Initial failure: method does not exist → compile error.
  - Test B `staged_predict_feature_mismatch`: pool with wrong float width ⇒
    `Err(CatBoostError::FeatureMismatch(_))` (`matches!`).
  - Test C `staged_predict_rejects_non_scalar_oblivious`: a multi-dim (or non-symmetric/CTR)
    facade model ⇒ `Err(CatBoostError::UnsupportedModel(_))`. (Source the non-scalar model the
    same way an existing facade/multiclass test does — **confirm via CodeGraph
    `codegraph_explore "multiclass Model facade test load_cbm fixture"` before writing**.)
- **Green (minimal intent):** default `start=0, end=0, period=1`; run the guard; `let cols =
  self.feature_columns(pool)?;`; `Ok(predict_raw_staged(self.as_canonical(), &cols, start, end,
  period))`. No transform (RawFormulaVal only, per SPEC R3).
- **Refactor:** if the guard grows, extract a private `fn ensure_scalar_oblivious(&self) ->
  Result<(), CatBoostError>`. Regression scope: `cargo test -p catboost-rs`.
- **Validation:** `cargo test -p catboost-rs` · `cargo clippy -p catboost-rs --lib --no-deps`
  · `cargo build -p catboost-rs-py` (proves the new error variant did not break the exhaustive
  `to_pyerr` match — E0004) · `cargo test -p catboost-rs-py --lib` (the `errors_test.rs`
  `UnsupportedModel → CatBoostValueError` assertion). The `catboost-rs-py` build/test is
  MANDATORY here: `cargo test -p catboost-rs` alone never compiles the py crate, so omitting it
  would mark T3 done while the workspace does not build.
- **Completion evidence:** tests A/B/C green; new error variant compiles; `cargo build -p
  catboost-rs-py` succeeds with the new `to_pyerr` arm; py `errors_test.rs` assertion green; no
  existing facade test regressed.
- **Parallelization:** Wave 3, parallel with T4 (disjoint files — T3 touches
  `catboost-rs/src/{model.rs,error.rs}`, `catboost-rs-py/src/errors{,_test}.rs`, and a new
  `catboost-rs/tests/` file; T4 touches only `cb-oracle/fixtures/` + a new `cb-model/tests/`
  file, so no write conflict). **Re-run coupling:** if T4-R1 corrects T2's stage schedule after
  T3 has run, T3's facade tests (esp. `staged_predict_facade_last_equals_predict`, which asserts
  `stages.last() == predict`) must be **re-run** — they exercise the same schedule and run in
  parallel with T4.

---

## T4-STAGED-ORACLE — parity vs upstream `model.staged_predict` (≤ 1e-5)

- **Spec:** SP-04 (and R1 schedule confirmation)
- **Goal / observable completion:** a frozen float-only `.cbm` + fixed matrix + upstream
  staged-prediction matrix are committed under `crates/cb-oracle/fixtures/staged_predict/`;
  an integration test replays `predict_raw_staged` and matches the upstream matrix per stage
  at `≤ 1e-5`; and the exact upstream stage indexing/inclusion has been **empirically
  confirmed** and the Rust schedule (T2) aligned to it.
- **Prerequisites:** T2-STAGED-SCHEDULE (for the Rust side). Independent of T3.
- **Files:**
  - Create: `crates/cb-oracle/fixtures/staged_predict/gen_fixtures.py` — pinned-seed offline
    generator (`catboost==1.2.10`, `thread_count=1`, `bootstrap_type="No"`, float-only
    `CatBoostRegressor` or binclf; reuse `inputs/numeric_tiny/X.npy` per the PDP fixture).
  - Create (generated, committed): `staged_predict/model.cbm`, `staged_predict/config.json`,
    and one expected `.npy` per fixtured schedule (e.g. `staged_period1.npy`,
    `staged_period3.npy`), each shaped `[n_stages, n_objects]` (document the axis order in the
    test).
  - Create: `crates/cb-model/tests/staged_predict_oracle_test.rs` — integration oracle
    (template: `crates/cb-model/tests/partial_dependence_oracle_test.rs`), using
    `cb_model::{load_cbm, predict_raw_staged}`, `cb_oracle::{assert_abs_close}`,
    `ndarray_npy::read_npy`, `TOL = 1e-5`, `#![allow(clippy::unwrap_used, ...)]` top-line like
    the sibling oracle tests.
- **R1 empirical-confirmation step (MANDATORY, do this in `gen_fixtures.py` first):**
  1. Fit the float model; call `model.staged_predict(X, prediction_type='RawFormulaVal',
     eval_period=k)` for `k ∈ {1, 3}` and capture each result as a list/array.
  2. Record the produced **shape** (number of stages) and, by comparing stage `j` to
     `model.predict(X, ntree_end=c_j)` for candidate counts `c_j`, determine the exact
     tree-count each stage corresponds to (does it start at `eval_period` or `0`? is there an
     initial empty/bias stage? is `ntree_end` always included?). Write these confirmed counts
     into `config.json` (a `stage_tree_counts` field per schedule).
  3. If the confirmed counts differ from T2's `stage_counts`, STOP and correct
     `predict_raw_staged` (T2) before finalizing this task; note the correction in the SPEC R1
     line. If they match, proceed.
- **Red:**
  - Test `staged_predict_matches_upstream` in `staged_predict_oracle_test.rs`.
  - Setup: `load_cbm(fixture("staged_predict/model.cbm"))`; `numeric_tiny` X → f32 SoA columns
    (same helper as the PDP oracle test).
  - Input/expected: for each fixtured `eval_period k`, call `predict_raw_staged(&model, &cols,
    0, 0, k)`, flatten to `[n_stages][n_objects]`, and compare each stage row to the matching
    row of the committed upstream `.npy` via `assert_abs_close(expected_row, actual_row, 1e-5)`.
    Also assert the stage **count** equals the confirmed `stage_tree_counts.len()`.
  - Initial failure: the fixture files / test do not exist → the test (and its fixture load)
    fail closed; once files land, any schedule/value mismatch diverges at 1e-5.
- **Green (minimal intent):** commit the generated fixtures; ensure the Rust schedule matches
  the confirmed upstream counts (correcting T2 if R1 revealed a delta). No production logic is
  authored in this task beyond a possible T2 schedule correction driven by R1.
- **Refactor:** none expected in prod; keep the test helper for fixture-path resolution local
  and mirrored from the PDP oracle test.
- **Validation:** `cargo test -p cb-model --test staged_predict_oracle_test`. (Fixture
  regeneration is offline: `uv venv --python 3.12 && uv pip install catboost==1.2.10 'numpy<2'`
  then `.venv/bin/python crates/cb-oracle/fixtures/staged_predict/gen_fixtures.py` — run once,
  commit artifacts; never regenerate at test time.)
- **Completion evidence:** per-stage `max|diff| ≤ 1e-5` for every fixtured schedule; the
  confirmed `stage_tree_counts` recorded in `config.json` and matched by the Rust output; R1
  resolved (SPEC R1 line updated from UNVERIFIED to the confirmed rule).
- **Parallelization:** Wave 3, parallel with T3 (disjoint files).

---

## Consistency check

- Every SP maps to ≥1 task (SP-01→T1, SP-02→T2, SP-03→T3, SP-04→T4); every task maps back to a
  SP. Dependency graph is acyclic (T1→T2→{T3,T4}).
- Each task has Red / Green / Refactor / Validation and exact files+symbols; all referenced
  symbols were CodeGraph/Read-verified, and every created path is marked **Create**.
- `predict_raw` / `predict_raw_cat` / `predict_raw_one` remain byte-identical (additive fn +
  additive re-export + additive facade method). The new `CatBoostError::UnsupportedModel`
  variant is NOT source-compatible downstream (`CatBoostError` is not `#[non_exhaustive]` and
  `catboost-rs-py::to_pyerr` is an exhaustive match), so T3 pairs it with the required
  `to_pyerr` arm + py test and a `cargo build -p catboost-rs-py` gate.
- Two residual empirical unknowns are explicitly routed through T4-R1 (upstream stage
  indexing) and through the "confirm via CodeGraph before writing" notes in T1/T3 (the exact
  in-test `Model` constructor and the non-scalar fixture source) — neither blocks planning.

## Unresolved blockers

- **None blocking.** One item is *deferred by design to T4*: the exact upstream
  `staged_predict` first-stage/inclusion rule (SPEC R1) is confirmed empirically at fixture
  time, and T2's `stage_counts` is corrected there if it differs — this is a planned task step,
  not a plan-time blocker.
- TreeFinder MCP is unavailable in this environment, so the SPEC/PLAN remain the authoritative
  local drafts (SPEC front-matter `treefinder_pending: UNRESOLVED`); no external spec store was
  synchronized.
