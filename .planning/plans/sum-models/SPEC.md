---
title: sum_models — weighted model merge (float-only oblivious, first slice)
status: draft
format: markdown
spec_version: 1
updated_at: 2026-07-18T00:00:00Z
source_requirements:
  - "User: Implement features of CatBoost that have not yet been implemented in catboost_rs."
  - ".planning/plans/next-feature-research/research.md §2 Candidate 1 (RECOMMENDED)"
treefinder_pending:
  collection: UNRESOLVED
  document_id: UNRESOLVED
  note: "TreeFinder MCP not available in this session; local SPEC is authoritative draft. Upstream C++/Python signatures INFERRED (sparse catboost-master checkout — only 3 files)."
---

# sum_models — weighted model merge

## 1. Context

CatBoost exposes `sum_models(models, weights=None, ctr_merge_policy=...)` which combines
N trained models into ONE by scaling each model's leaf contributions by its weight and
summing the ensembles (trees concatenated, biases summed). It is used for blending /
ensembling and for incremental model composition. It does **not** exist in catboost_rs
today `[VERIFIED: LOCAL grep -rli sum_model crates/ → empty]`.

The canonical in-memory model is `cb_model::Model`
`[VERIFIED: CODEGRAPH crates/cb-model/src/model.rs:271-313]`:

```
pub struct Model {
    pub oblivious_trees: Vec<ObliviousTree>,          // boosting order
    pub non_symmetric_trees: Vec<NonSymmetricTree>,   // EMPTY for oblivious models
    pub region_trees: Vec<RegionTree>,                // EMPTY for oblivious models
    pub bias: f64,                                     // starting approx; NO separate scale field
    pub float_feature_borders: Vec<Vec<f64>>,
    pub ctr_data: Option<CtrData>,                     // None for numeric-only models
    pub approx_dimension: usize,                       // 1 for scalar/binary
    pub class_to_label: Vec<f64>,                      // EMPTY for scalar/binary
}
```

`ObliviousTree = { splits: Vec<ModelSplit>, leaf_values: Vec<f64>, leaf_weights: Vec<f64> }`
`[VERIFIED: CODEGRAPH crates/cb-model/src/cbm.rs:1086]`. Apply computes
`bias + Σ_tree leaf_values[leaf_index]` with no scale multiply
`[VERIFIED: CODEGRAPH crates/cb-model/src/apply.rs:318 predict_raw_one]`, confirming the
canonical model bakes scale into leaf values (scale == 1.0 for standard trained/loaded
float models).

**Merge math (float-only, scale==1, first slice).** For models `m_i` with weights `w_i`,
the summed model's raw prediction must equal `Σ_i w_i · rawpredict(m_i, x)`. Since
`rawpredict(m_i) = bias_i + Σ_tree contribution`, and every tree contribution is linear in
its leaf value, the summed model is:
- `oblivious_trees` = concatenation over `i` of each `m_i.oblivious_trees` with **every leaf
  value multiplied by `w_i`** (splits/structure unchanged);
- `bias` = `Σ_i w_i · bias_i` (reduced via `cb_core::sum_f64`, D-08);
- `float_feature_borders` = the shared borders (all inputs must agree — see SM-04);
- `approx_dimension`, `class_to_label` carried through from the (identical) inputs;
- `ctr_data = None`, `non_symmetric_trees = []`, `region_trees = []` (all guarded out).

## 2. Scope and non-goals

**In scope (first slice):** N ≥ 1 **oblivious, float-only** (`ctr_data == None`),
scale==1 models with **identical** `float_feature_borders`, identical `approx_dimension`,
and identical `class_to_label`; a `cb_model` free function; a `catboost-rs` facade method;
an optional Python `catboost_rs.sum_models`; a ≤1e-5 oracle vs `catboost.sum_models`.

**Non-goals (explicit):** CTR / categorical model merge (`ctr_merge_policy`) — deferred to
a second slice; non-symmetric (Lossguide/Depthwise) or Region models; models with differing
float-feature borders or feature counts; merging models of differing `approx_dimension`;
merging models of differing `class_to_label`; text/embedding models. Each of these **is a
typed `ModelError::Merge`**, never a silent wrong merge (SM-04).

**Non-default scale — an UNCHECKABLE assumption, NOT a typed error.** The canonical
`cb_model::Model` has **no scale field**: `decode_cbm` folds the wire `Scale` away and apply
is `bias + Σ leaf` with no scale multiply `[VERIFIED: CODEGRAPH crates/cb-model/src/model.rs:272,
apply.rs:354; PLAN-CHECK sum-models MAJOR]`. At the `sum_models` boundary a non-unit scale is
therefore **unrecoverable and undetectable** — it cannot be a typed-error case. The first slice
**assumes** scale==1 (the standard trained/loaded float model) and relies on the SM-07 oracle as
the only backstop. This is an inherited apply-path limitation, not a defect introduced here.

## 3. Dependencies

- `cb_core::sum_f64` — deterministic ordered reduction (D-08) `[VERIFIED: CODEGRAPH crates/cb-model/src/fstr.rs uses sum_f64]`.
- `cb_model::{Model, ObliviousTree}` — read-only construction `[VERIFIED: CODEGRAPH]`.
- `cb_model::error::ModelError` — add a variant OR reuse `Serialize`/`Deserialize`? See §4:
  new variant `ModelError::Merge(String)` `[VERIFIED: CODEGRAPH crates/cb-model/src/error.rs:16-52]`.
- facade `catboost-rs::CatBoostError` + `Model` wrapper (`from_canonical`, `as_canonical`)
  `[VERIFIED: CODEGRAPH crates/catboost-rs/src/model.rs:38-47]`.
- Oracle: `cb_oracle` compare harness ≤1e-5; frozen float `.cbm` fixtures
  `[VERIFIED: LOCAL crates/cb-oracle/fixtures/model_serde]`.
- NO new external crate.

## 4. Typed contracts

```rust
// crates/cb-model/src/model_sum.rs  (NEW FILE)

/// Combine `models` into one weighted-sum model. `weights[i]` scales model i's
/// leaf contributions; when `weights` is empty, every model gets weight 1.0.
///
/// # Errors
/// `ModelError::Merge` if: `models` is empty; `weights` is non-empty and its length
/// != models.len(); any model is non-oblivious (non_symmetric_trees / region_trees
/// non-empty); any model carries `ctr_data` (Some); the models disagree on
/// `float_feature_borders`, `approx_dimension`, or `class_to_label`.
pub fn sum_models(models: &[&Model], weights: &[f64]) -> Result<Model, ModelError>;
```

New error variant:
```rust
// crates/cb-model/src/error.rs
/// Two or more models cannot be merged: empty input, weight/model count mismatch,
/// an unsupported model kind (CTR / non-oblivious), or incompatible feature/output
/// structure. Surfaced loudly instead of emitting a wrong-valued merged model.
#[error("models cannot be merged: {0}")]
Merge(String),
```

Facade:
```rust
// crates/catboost-rs/src/model.rs
/// Combine several models into one weighted-sum model (D-07 analogue).
/// # Errors
/// [`CatBoostError::Model`] on an unmergeable set (see cb_model::sum_models).
pub fn sum_models(models: &[&Model], weights: Option<&[f64]>) -> Result<Model, CatBoostError>;
```
(`CatBoostError::Model` already wraps `ModelError` `[VERIFIED: CODEGRAPH crates/catboost-rs/src/model.rs load_cbm maps ModelError]`.)

## 5. Failure-isolated behavioral specifications

### SM-01 — Single-model weight scaling (leaf arithmetic)
- **Responsibility:** scale one model's every oblivious leaf value by a scalar weight.
- **Input:** `&Model` (oblivious, float-only), `w: f64`.
- **Output:** a `Model` whose `oblivious_trees[t].leaf_values[l] == w * input.leaf_values`,
  `bias == w * input.bias`, everything else structurally identical.
- **Given/When/Then:** Given a 1-model input with weight `w`; When summed; Then
  `rawpredict(result, x) == w * rawpredict(input, x)` for all x (≤1e-12 numerically).
- **Acceptance:** unit test `sum_models_single_scales_leaves` in `model_sum_test.rs`.
- **Out of scope:** multi-model concat (SM-02).

### SM-02 — Tree concatenation of N compatible models
- **Responsibility:** produce `oblivious_trees` = concatenation of each weighted input's trees.
- **Input:** `&[&Model]` (≥2, compatible), `weights`.
- **Output:** `result.oblivious_trees.len() == Σ_i models[i].oblivious_trees.len()`, in
  input order, each tree's splits byte-identical to its source (only leaf values scaled).
- **Given/When/Then:** Given 2 models with `a` and `b` trees; When summed; Then result has
  `a+b` trees and `rawpredict(result,x) == w0*rawpredict(m0,x)+w1*rawpredict(m1,x)`.
- **Acceptance:** `sum_models_concats_trees`.
- **Out of scope:** bias arithmetic (SM-03), validation (SM-04).

### SM-03 — Weighted bias sum
- **Responsibility:** `result.bias == Σ_i w_i · m_i.bias` via `cb_core::sum_f64`.
- **Input:** `&[&Model]`, `weights`.
- **Output:** exact weighted sum (deterministic reduction order = input order).
- **Given/When/Then:** Given models with biases `[b0,b1]` and weights `[w0,w1]`; When summed;
  Then `result.bias == sum_f64(&[w0*b0, w1*b1])`.
- **Acceptance:** `sum_models_sums_bias`.

### SM-04 — Compatibility validation → typed error
- **Responsibility:** reject unmergeable inputs with `ModelError::Merge`, never a wrong merge.
- **Input:** any `&[&Model]`, any `weights`.
- **Output/typed error:** `Err(ModelError::Merge(_))` on each rejected case; `Ok` only when
  every precondition holds.
- **Given/When/Then (each its own test case):**
  - empty `models` → Err;
  - `weights.len() != models.len()` (and weights non-empty) → Err;
  - a model with non-empty `non_symmetric_trees` or `region_trees` → Err;
  - a model with `ctr_data.is_some()` → Err;
  - models with differing `float_feature_borders` → Err;
  - models with differing `approx_dimension` → Err;
  - models with differing `class_to_label` → Err.
- **Acceptance:** `sum_models_rejects_*` (one assertion per case).
- **Invariant:** never panics; all access checked `.get` (clippy deny-lints).

### SM-05 — Facade wiring (`catboost_rs::Model::sum_models`)
- **Responsibility:** expose the merge on the published facade; map `ModelError → CatBoostError`;
  default `weights=None` to all-ones.
- **Input:** `&[&catboost_rs::Model]`, `Option<&[f64]>`.
- **Output:** `Result<catboost_rs::Model, CatBoostError>` wrapping the canonical result.
- **Given/When/Then:** Given two facade models; When `Model::sum_models([&a,&b], None)`; Then
  the wrapped model predicts `rawpredict(a)+rawpredict(b)`.
- **Acceptance:** facade test `sum_models_facade_roundtrip`.

### SM-06 — Python surface (`catboost_rs.sum_models`) [OPTIONAL, sequence last]
- **Responsibility:** module-level Python function mirroring upstream signature
  `sum_models(models, weights=None)` returning a fitted estimator wrapping the merged model.
- **Input:** a Python list of fitted estimators, optional list of floats.
- **Output:** an estimator whose `.predict` equals the weighted sum.
- **Given/When/Then:** Given two fitted `CatBoostRegressor`s; When `sum_models([a,b])`; Then
  `.predict(X) ≈ a.predict(X)+b.predict(X)` (≤1e-5).
- **Acceptance:** py test (marks: needs built extension). May be deferred if the Python
  module wiring is out of the first slice's budget — note as a follow-up.
- **Out of scope:** `ctr_merge_policy` kwarg.

### SM-07 — Oracle parity vs `catboost.sum_models`
- **Responsibility:** prove ≤1e-5 parity against real CatBoost's own merged model.
- **Input:** frozen float `.cbm` models m0,m1 + weights + a fixed feature matrix; upstream
  `catboost.sum_models([m0,m1],[w0,w1]).predict(X)` frozen to `expected.npy`.
- **Output:** `predict_raw(sum_models([m0,m1],[w0,w1]), X)` matches `expected` ≤1e-5.
- **Given/When/Then:** Given the frozen fixture; When the Rust merge predicts; Then max|diff| ≤ 1e-5.
- **Acceptance:** integration test `crates/cb-model/tests/model_sum_oracle_test.rs` over
  `crates/cb-oracle/fixtures/model_sum/`.

## 6. Acceptance scenarios

1. `w=[1,1]`: merged prediction == sum of the two models' raw predictions (SM-02/SM-07).
2. `w=[0.3,0.7]`: weighted blend matches upstream ≤1e-5 (SM-01/SM-03/SM-07).
3. Single model, `w=[2.0]`: predictions doubled (SM-01).
4. CTR model in input → typed `Merge` error (SM-04).
5. Mismatched borders → typed `Merge` error (SM-04).

## 7. Impact scope

- **local:** new `crates/cb-model/src/model_sum.rs`, new `ModelError::Merge` variant, `lib.rs`
  re-export. No edit to `apply.rs`, `fstr.rs`, or `tree.rs` (zero overlap with in-flight work)
  `[VERIFIED: LOCAL git status — fstr.rs & tree.rs uncommitted]`.
- **cross-module:** facade method in `crates/catboost-rs/src/model.rs`; optional Python fn.
- **tests:** `model_sum_test.rs` (unit, mounted), `model_sum_oracle_test.rs` (integration),
  new fixture dir `crates/cb-oracle/fixtures/model_sum/`.
- No persistence/schema/migration/event/config impact. `ModelError::Merge` is an additive enum
  variant — existing exhaustive matches on `ModelError` (if any) must add the arm; verify via
  CodeGraph blast radius at plan time.

## 8. Compatibility and migration

Purely additive. Adding `ModelError::Merge` does not change existing serialization or apply.
No `.cbm`/json wire change. No breaking API change.

## 9. Risks and open questions

- **R1 (scale field):** confirmed the canonical `Model` has no scale field; standard float
  models are scale==1. If a loaded model ever carries a non-1 baked scale (it should not),
  the merge would be wrong. **Mitigation:** first slice only accepts models produced by this
  workspace's train/load path (scale==1); no scale field exists to check, so document the
  assumption and rely on the oracle to catch any deviation. `[INFERRED from apply.rs having no scale]`
- **R2 (upstream signature):** exact `catboost.sum_models` arg names/defaults and the
  `ctr_merge_policy` default are `[UNVERIFIED — sparse checkout]`; confirm against
  github.com/catboost core.py at fixture-generation time.
- **R3 (leaf_weights):** whether merged `leaf_weights` must be scaled for downstream fstr on
  the merged model. First slice predicts only; leaf_weights carried unscaled — note as a
  known limitation if the merged model is later fed to PredictionValuesChange.
- **Q1:** should `weights=[]` (empty) mean all-ones (upstream `weights=None`)? Adopt yes.

## 10. Traceability and sources

- Research: `.planning/plans/next-feature-research/research.md` §0, §2 Candidate 1, §4, §5.
- CodeGraph: `cb-model/src/{model.rs:71-313, cbm.rs:1086-1120, apply.rs:318-370, error.rs},
  catboost-rs/src/model.rs:38-274`.
- Local: `crates/cb-model/Cargo.toml` (cb-core-only dep), `crates/cb-oracle/fixtures/model_serde`.
- Memory: `ctr-model-loading.md` (frozen CTR/float fixture rule), fstr03 plan (clippy gate,
  test-mount, uv oracle recipe).
