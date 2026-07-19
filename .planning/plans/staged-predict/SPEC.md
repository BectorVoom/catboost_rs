---
title: staged_predict — per-tree-prefix cumulative prediction
status: draft
format: markdown
spec_version: 1
updated_at: 2026-07-18T00:00:00Z
source_requirements:
  - "User: Implement features of CatBoost that have not yet been implemented in catboost_rs."
  - ".planning/plans/next-feature-research/research.md §2 Candidate 2"
treefinder_pending:
  collection: UNRESOLVED
  document_id: UNRESOLVED
  note: "TreeFinder MCP unavailable; local SPEC is authoritative draft. Upstream Python signature INFERRED (sparse checkout)."
---

# staged_predict — per-tree-prefix cumulative prediction

## 1. Context

CatBoost's `staged_predict(data, prediction_type, ntree_start, ntree_end, eval_period)`
yields predictions evaluated over an increasing prefix of the ensemble — the raw approx after
`ntree_start`, `ntree_start+eval_period`, … up to `ntree_end` trees. It underpins learning-curve
analysis and early-stopping introspection. It is **not** exposed today: the facade has no
`staged_predict` `[VERIFIED: LOCAL grep staged crates/catboost-rs → empty]`.

The scalar apply loop sums `bias + Σ_tree tree.leaf_values[leaf_index_for(...)]`
`[VERIFIED: CODEGRAPH crates/cb-model/src/apply.rs:318 predict_raw_one, :208 leaf_index_for]`.
A prefix prediction is the same accumulation truncated to the first `k` trees. `leaf_index_for`
is already dimension-agnostic and reusable `[VERIFIED: CODEGRAPH apply.rs:208]`.

**Prefix semantics.** For a scalar oblivious model with `T` trees, the raw approx at prefix `k`
is `bias + Σ_{t<k} tree_t.leaf_values[leaf_index_for(t, x)]`. The stages are the prefixes
`k ∈ {ntree_start? or eval_period, 2·eval_period, …, ntree_end}` following upstream
(`ntree_end == 0` means "all trees"; stage boundaries step by `eval_period`, always including
`ntree_end`). This SPEC scopes the **first slice to scalar (`approx_dimension == 1`),
oblivious, RawFormulaVal** predictions; multi-dim and non-oblivious are non-goals.

## 2. Scope and non-goals

**In scope:** a `cb_model` free function returning per-stage raw approx for scalar oblivious
models over a float feature matrix; a `catboost-rs` facade method returning a `Vec<Vec<f64>>`
(one inner vec per stage, each length `n_objects`); the stage schedule
(`ntree_start`, `ntree_end`, `eval_period`) with upstream defaults; a ≤1e-5 oracle vs
`model.staged_predict`.

**Non-goals:** multiclass / multi-dimension staged output; non-symmetric / Region models;
CTR/categorical models (float-only first slice); non-raw prediction types (Probability/Class
transforms) — deferred (the facade may layer `apply_prediction_type` per stage later);
Python surface (follow-up).

## 3. Dependencies

- `cb_model::apply::{leaf_index_for (fn, currently private), predict_raw_one}` — the prefix fn
  reuses the per-tree leaf accumulation. `leaf_index_for` is currently a private fn in
  `apply.rs` `[VERIFIED: CODEGRAPH apply.rs:208]`; the new prefix fn lives in the SAME module so
  it can call it without widening visibility.
- `cb_model::Model` fields `oblivious_trees`, `bias`, `approx_dimension`
  `[VERIFIED: CODEGRAPH model.rs:271-313]`.
- facade `Model` wrapper + `feature_columns` narrowing + `CatBoostError::FeatureMismatch`
  `[VERIFIED: CODEGRAPH crates/catboost-rs/src/model.rs:60-93]`.
- Oracle harness ≤1e-5; frozen float `.cbm` + `staged_predict` expected matrix
  `[VERIFIED: LOCAL crates/cb-oracle/fixtures/prediction_types]`.
- NO new external crate.

## 4. Typed contracts

```rust
// crates/cb-model/src/apply.rs  (add; sibling to predict_raw)

/// Raw approx over an increasing prefix of the ensemble (SCALAR oblivious models).
/// Returns one row per stage; each row is length `n_objects` in object order.
/// Stages are the tree counts `min(ntree_start + eval_period, ntree_end), …, ntree_end`,
/// where `ntree_end == 0` is treated as `oblivious_trees.len()`. `eval_period == 0`
/// is treated as `1`. An out-of-range `ntree_start >= end` yields an empty Vec.
///
/// # Contract (UNGUARDED — caller's responsibility)
/// This function does NOT validate model shape: it only accumulates
/// `model.oblivious_trees` over the scalar leaf-value path. On a multi-dimension
/// (`approx_dimension > 1`), non-symmetric, Region, or CTR model it returns
/// SILENTLY WRONG output (dropped dimensions / ignored trees), not an error. The
/// scalar-oblivious guard lives at the facade (SP-03); any direct `cb-model` caller
/// MUST ensure the model is scalar + oblivious + float-only before calling.
#[must_use]
pub fn predict_raw_staged(
    model: &Model,
    feature_values: &[Vec<f32>],
    ntree_start: usize,
    ntree_end: usize,
    eval_period: usize,
) -> Vec<Vec<f64>>;
```

Facade:
```rust
// crates/catboost-rs/src/model.rs
/// Cumulative raw predictions over tree prefixes (scalar oblivious models).
/// `ntree_start`/`ntree_end`/`eval_period` default to `0`/`0`/`1` (all trees, step 1)
/// when `None`. Returns one Vec<f64> per stage.
/// # Errors
/// [`CatBoostError::FeatureMismatch`] if `pool`'s float-feature count differs from the model's.
pub fn staged_predict(
    &self,
    pool: &Pool,
    ntree_start: Option<usize>,
    ntree_end: Option<usize>,
    eval_period: Option<usize>,
) -> Result<Vec<Vec<f64>>, CatBoostError>;
```

## 5. Failure-isolated behavioral specifications

### SP-01 — Single-stage prefix equals truncated apply
- **Responsibility:** raw approx over the first `k` trees equals the full-apply accumulation truncated to `k`.
- **Input:** scalar oblivious `&Model`, float matrix, `ntree_start=0, ntree_end=k, eval_period=k`.
- **Output:** one stage row; `row[i] == bias + Σ_{t<k} tree_t contribution` for object i.
- **Given/When/Then:** Given a model with `T` trees and `k<T`; When staged with a single stage at `k`;
  Then the row equals a hand-rolled prefix sum; and at `k==T` it equals `predict_raw` exactly.
- **Acceptance:** `staged_prefix_matches_truncated_apply` in `apply_test.rs` (or `staged_predict_test.rs`).
- **Out of scope:** multi-stage schedule (SP-02).

### SP-02 — Stage schedule (ntree_start / ntree_end / eval_period)
- **Responsibility:** produce exactly the stage tree-counts upstream produces, always including `ntree_end`.
- **Input:** `ntree_start`, `ntree_end` (0 ⇒ all), `eval_period` (0 ⇒ 1).
- **Output:** correct number/order of stage rows.
- **Given/When/Then:** Given `T=10, start=0, end=0, period=3`; When staged; Then stages at tree
  counts `{3,6,9,10}` (last stage always the full `end`); each row is the cumulative approx at that count.
- **Acceptance:** `staged_schedule_boundaries` (assert row count and each row's equivalence to a
  direct prefix apply at that count). Include edge cases `period=1` (T rows) and `start>=end` (empty).
- **Out of scope:** facade wiring (SP-03).

### SP-03 — Facade wiring + feature-count guard
- **Responsibility:** expose `staged_predict` on the facade; narrow the pool to f32 columns;
  reject a float-feature-count mismatch with `FeatureMismatch`; apply `None` defaults.
- **Input:** `&Pool`, optional stage params.
- **Output:** `Result<Vec<Vec<f64>>, CatBoostError>`.
- **Given/When/Then:** Given a pool whose float width != model's; When `staged_predict`; Then
  `Err(FeatureMismatch)`. Given a matching pool with defaults; Then `stages.last()` equals
  `predict(pool)` exactly.
- **Acceptance:** facade test `staged_predict_facade_last_equals_predict` + `staged_predict_feature_mismatch`.

### SP-04 — Oracle parity vs `model.staged_predict`
- **Responsibility:** ≤1e-5 parity against upstream staged predictions.
- **Input:** frozen float `.cbm` + fixed matrix; upstream
  `model.staged_predict(X, prediction_type='RawFormulaVal', eval_period=k)` frozen as a matrix.
- **Output:** Rust staged rows match the upstream matrix ≤1e-5 (aligning stage indexing to
  upstream's — verify upstream's first stage / inclusion rule at fixture time).
- **Given/When/Then:** Given the fixture; When Rust staged predicts; Then per-stage max|diff| ≤ 1e-5.
- **Acceptance:** `crates/cb-model/tests/staged_predict_oracle_test.rs` over
  `crates/cb-oracle/fixtures/staged_predict/`.

## 6. Acceptance scenarios

1. `eval_period=1` on a `T`-tree model → `T` stages; final stage == `predict_raw` (SP-01/SP-02).
2. `eval_period=3, T=10` → stages at `{3,6,9,10}` matching upstream ≤1e-5 (SP-02/SP-04).
3. Non-symmetric or multiclass model → out of first-slice scope; facade returns a typed
   `CatBoostError` (add a guard: `approx_dimension>1` or non-empty non_symmetric/region ⇒ error).
4. Feature-count mismatch → `FeatureMismatch` (SP-03).

## 7. Impact scope

- **local:** add `predict_raw_staged` to `crates/cb-model/src/apply.rs` (additive; existing
  `predict_raw`/`predict_raw_cat` byte-unchanged), re-export in `lib.rs`. Add unit tests.
  `apply.rs` is NOT among the in-flight uncommitted files (`fstr.rs`, `tree.rs` are)
  `[VERIFIED: LOCAL git status]`.
- **cross-module:** facade method in `crates/catboost-rs/src/model.rs`.
- **tests:** unit (mounted sibling), integration oracle, new fixture dir.
- No schema/persistence/event/config impact.

## 8. Compatibility and migration

Purely additive: a new public fn + facade method. No change to any existing apply path.

## 9. Risks and open questions

- **R1 (stage indexing):** `[CONFIRMED — T4 empirical, catboost==1.2.10]`. Upstream
  `model.staged_predict(X, prediction_type='RawFormulaVal', eval_period=k)` produces stage
  tree-counts stepping by `k` starting at `k` and ALWAYS including `ntree_end` as the final
  stage: for a `T=10` model, `k=1 -> {1,2,…,10}` (10 stages), `k=3 -> {3,6,9,10}` (4 stages).
  Each stage `j` equals `model.predict(X, ntree_end=stage_tree_counts[j])` (matched to 1e-9).
  There is NO initial empty/bias-only stage. SP-02's `stage_counts(start, end, step)` (first at
  `start+step`, always push `end`) matched this exactly — **no correction was required**.
  Recorded in `crates/cb-oracle/fixtures/staged_predict/config.json` (`stage_tree_counts`).
- **R2 (scalar-only):** first slice is `approx_dimension==1`. A guard must reject multi-dim /
  non-oblivious at the facade with a typed error (scenario 3), else the truncation would silently
  drop dimensions. Captured in SP-03 guard.
- **R3 (prediction_type):** only RawFormulaVal in scope. Probability/Class per-stage transforms
  are a follow-up; document so callers don't assume sigmoid-applied stages.

## 10. Traceability and sources

- Research: `.planning/plans/next-feature-research/research.md` §2 Candidate 2, §4.
- CodeGraph: `cb-model/src/apply.rs:{208 leaf_index_for, 318 predict_raw_one, 370 predict_raw}`,
  `catboost-rs/src/model.rs:{60-113 predict/feature_columns}`.
- Local: `crates/cb-oracle/fixtures/prediction_types`; `crates/cb-model/Cargo.toml`.
- Memory: fstr03 plan (clippy gate, test-mount, uv oracle recipe), ctr-model-loading (frozen fixtures).
