---
title: Categorical-feature (cat_features/CTR) routing through the CatBoostBuilder facade
status: draft
format: markdown
spec_version: 1
updated_at: 2026-07-31T00:00:00Z
source_requirements:
  - "User request (2026-07-31): implement all train catboost_rs parameters; first slice = cat_features/CTR facade routing"
  - ".planning/plans/catboost-builder-cat-features-routing/research.md"
  - "Locked user decisions (2026-07-31): predict-side wiring IN scope; cat_features as fit() kwarg IN scope; CTR priors exposed in lockstep with CTR type"
---

# Categorical-feature (cat_features/CTR) routing through the `CatBoostBuilder` facade

## 1. Context

`catboost-rs` already contains a complete, oracle-verified categorical/CTR training
engine (`cb_train::train_cat`) and a complete, oracle-verified CTR-aware inference
path (`cb_model::predict_raw_cat` + `cb_model::CtrData`). Neither is reachable from
the public facade.

- `CatBoostBuilder::fit` reads only `pool.float_features()` and unconditionally calls
  the float-only `cb_train::train`; `pool.cat_features()` is never read
  `[VERIFIED: LOCAL crates/catboost-rs/src/builder.rs:334-402]`.
- `CatBoostBuilder::boost_params()` pins every CTR-related `BoostParams` field to its
  upstream default with an explicit "inert here" comment, and exposes no setter for
  any of them `[VERIFIED: LOCAL crates/catboost-rs/src/builder.rs:284-317]`.
- `Model::predict_with` calls the float-only `predict_raw`
  `[VERIFIED: LOCAL crates/catboost-rs/src/model.rs:120-128]`.
- `crates/catboost-rs-py/src/params.rs` lists `cat_features`, `one_hot_max_size`,
  `max_ctr_complexity`, `simple_ctr`, `combinations_ctr`, `counter_calc_method` in
  `VOCABULARY` but NOT in `IMPLEMENTED`, so `validate_params` rejects each as a
  "parity gap" `[VERIFIED: LOCAL crates/catboost-rs-py/src/params.rs:42-188,290-318]`.

Consequently CatBoost's namesake capability — categorical features — is unusable
through both the Rust facade and the Python bindings, despite the engine being
present and tested.

### The silent-wrongness hazard (drives SPEC-CATF-12)

`predict_raw(model, fv)` is defined as `predict_raw_cat(model, fv, &[])`
`[VERIFIED: LOCAL crates/cb-model/src/apply.rs:370-372]`. Inside `predict_raw_cat`,
an absent categorical value resolves via `col.get(obj).cloned().unwrap_or_default()`
— the empty string — and `feature_columns()` validates only the FLOAT width
`[VERIFIED: LOCAL crates/cb-model/src/apply.rs:410-414; crates/catboost-rs/src/model.rs:94-107]`.
A CTR model scored through today's `predict()` therefore evaluates every
`ModelSplit::Ctr` against an empty-string category and returns **numerically wrong
predictions with no error**. Wiring `fit()` without wiring `predict()` would ship
exactly this footgun, contradicting the repo's stated "never silently wrong" honesty
policy `[VERIFIED: LOCAL crates/catboost-rs-py/src/params.rs:1-18]`.

## 2. Scope and non-goals

### In scope

1. `CatBoostBuilder` setters for `one_hot_max_size`, `max_ctr_complexity`,
   `simple_ctr` + `simple_ctr_priors`, `combinations_ctr` + `combinations_ctr_priors`,
   `counter_calc_method`, each defaulting to the `*_default()` function
   `boost_params()` currently calls inline (behavior-preserving when unset).
2. `CatBoostBuilder::fit` branches on `pool.cat_features().is_empty()`: float-only path
   unchanged; categorical path calls `cb_train::train_cat` and attaches the returned
   `BakedCtrData` via `cb_model::CtrData::from_baked` + `Model::with_ctr_data`.
3. `catboost_rs` re-exports the setter parameter types (`ECtrType`,
   `CounterCalcMethod`) so downstream crates can name them.
4. `Model::predict_with` (and therefore `predict`/`predict_proba`) routes to
   `cb_model::predict_raw_cat` with the pool's categorical columns when the model
   carries baked `ctr_data`; a categorical-width mismatch is a typed error.
5. `crates/catboost-rs-py/src/params.rs` promotes the five CTR scalar/enum kwargs from
   rejected-parity-gap to `IMPLEMENTED`, with range/enum validation.
6. `cat_features` accepted as a `fit(X, y, cat_features=[...])` kwarg on the Python
   estimators, in addition to the existing `Pool(..., cat_features=[...])` path.
7. An oracle test driving the **public Python API** end to end
   (`Pool(..., cat_features=...)` → `fit()` → `predict()`) at ≤1e-5 against upstream
   `catboost==1.2.10`.

### Non-goals

- Any change to `cb_train::train_cat`'s internal correctness. The combination-CTR
  candidate-gating bug (ORD-06/ORD-07) is tracked separately in
  `.planning/phases/24-ctr-split-search-correctness/` and MUST NOT be addressed here
  `[VERIFIED: LOCAL .planning/phases/24-ctr-split-search-correctness/]`.
- CTR `.cbm`/`.json` save/load — already shipped (Phase 23).
- Arrow/Polars categorical ingestion — hard-rejected today by design at
  `crates/catboost-rs-py/src/ingest_py.rs:338-342`; unchanged here.
- CTR-awareness for `shap_values`, `staged_predict`, and ONNX/CoreML export.
  `staged_predict` and `save_onnx` already reject CTR models with a typed
  `UnsupportedModel` error `[VERIFIED: LOCAL crates/catboost-rs/src/model.rs:138-160]`
  — that existing rejection is the correct behavior and must be preserved, not
  extended.
- The remaining facade-wiring parameters (`od_type`, `monotone_constraints`,
  penalties, `grow_policy`, `boosting_type`, …) and the genuinely-missing engine
  parameters (`nan_mode`, `class_weights`, `rsm`, …) — separate, later plans in the
  same initiative.

## 3. Dependencies

All required symbols already exist, are public, and are compiled today. No new
external crate dependency is introduced.

| Symbol | Location | Role |
|---|---|---|
| `cb_train::train_cat` | `crates/cb-train/src/boosting.rs:2149` | Categorical-aware training entrypoint; returns `CbResult<(Model, BakedCtrData)>` |
| `cb_train::train` | `crates/cb-train/src/boosting.rs:1950` | Existing float-only path (unchanged) |
| `cb_model::CtrData::from_baked` | `crates/cb-model/src/ctr_data.rs:313` | Trainer output → inference-ready CTR tables |
| `cb_model::Model::with_ctr_data` | `crates/cb-model/src/ctr_data.rs:288` | Attaches CTR tables to the canonical model |
| `cb_model::predict_raw_cat` | `crates/cb-model/src/apply.rs:386` | CTR-aware apply |
| `cb_data::Pool::cat_features` | `crates/cb-data/src/pool.rs:152` | `-> &[Vec<String>]`, already populated end-to-end |
| `cb_train::{ECtrType, CounterCalcMethod}` | `crates/cb-train/src/lib.rs:46,49` | Setter parameter types |
| `cb_train::{one_hot_max_size,max_ctr_complexity,simple_ctr,simple_ctr_priors,combinations_ctr,combinations_ctr_priors,counter_calc_method}_default` | `cb-train` | Current pinned values; become the new setter defaults |

`CatBoostError` requires **no new variant**: `train_cat` returns the same
`CbResult` shape as `train`, covered by the existing
`CatBoostError::Train(#[from] cb_core::CbError)`
`[VERIFIED: LOCAL crates/catboost-rs/src/error.rs:31-34]`.

## 4. Typed contracts

```rust
// crates/catboost-rs/src/builder.rs — new setters (existing #[must_use] consuming style)
impl CatBoostBuilder {
    #[must_use] pub fn one_hot_max_size(mut self, one_hot_max_size: u32) -> Self;
    #[must_use] pub fn max_ctr_complexity(mut self, max_ctr_complexity: usize) -> Self;
    #[must_use] pub fn simple_ctr(mut self, simple_ctr: ECtrType) -> Self;
    #[must_use] pub fn simple_ctr_priors(mut self, priors: Vec<f64>) -> Self;
    #[must_use] pub fn combinations_ctr(mut self, combinations_ctr: ECtrType) -> Self;
    #[must_use] pub fn combinations_ctr_priors(mut self, priors: Vec<f64>) -> Self;
    #[must_use] pub fn counter_calc_method(mut self, m: CounterCalcMethod) -> Self;
}

// crates/catboost-rs/src/model.rs — categorical companion to `feature_columns`
impl Model {
    /// `Ok(cat columns)` when the pool's categorical width matches what the model
    /// expects; `Err(CatBoostError::FeatureMismatch)` otherwise.
    fn cat_columns(&self, pool: &Pool) -> Result<Vec<Vec<String>>, CatBoostError>;
    /// True when the canonical model carries baked CTR tables.
    fn is_ctr_model(&self) -> bool;
}
```

Exact `train_cat` call shape (from the proven pattern at
`crates/cb-train/tests/tensor_ctr_e2e_oracle_test.rs:232-250`):

```rust
let (trained, baked) = train_cat(
    &backend, &feature_values, &feature_borders,
    pool.cat_features(), pool.label(), pool.weights(), &params, None,
)?;
let canonical = cb_model::Model::from_trained(&trained, feature_borders)
    .with_ctr_data(cb_model::CtrData::from_baked(&baked));
```

## 5. Failure-isolated behavioral specifications

Each specification below has one behavioral responsibility, so a failing acceptance
test has one primary cause.

---

### SPEC-CATF-01 — `one_hot_max_size` setter
**Status:** draft
**Input:** `u32`. **Output:** `Self`.
**Given** a default `CatBoostBuilder`, **when** `.one_hot_max_size(k)` is called,
**then** `boost_params().one_hot_max_size == k`.
**Invariant:** an unset builder yields `one_hot_max_size_default()` — the value pinned
today.
**Out of scope:** whether the value changes training output (that is CATF-08).

### SPEC-CATF-02 — `max_ctr_complexity` setter
**Status:** draft
**Input:** `usize`. **Output:** `Self`.
**Given** a default builder, **when** `.max_ctr_complexity(k)`, **then**
`boost_params().max_ctr_complexity == k`; unset ⇒ `max_ctr_complexity_default()`.

### SPEC-CATF-03 — `simple_ctr` + `simple_ctr_priors` setters (lockstep)
**Status:** draft
**Input:** `ECtrType` / `Vec<f64>`. **Output:** `Self`.
**Given** a default builder, **when** either setter is called, **then** the matching
`BoostParams` field reflects it; unset ⇒ `simple_ctr_default()` /
`simple_ctr_priors_default()`.
**Rationale (locked decision):** the priors setter ships with the type setter so a
non-default `ECtrType` cannot silently pair with priors tuned for a different type.

### SPEC-CATF-04 — `combinations_ctr` + `combinations_ctr_priors` setters (lockstep)
**Status:** draft
Same contract as CATF-03 over `combinations_ctr` / `combinations_ctr_priors`.

### SPEC-CATF-05 — `counter_calc_method` setter
**Status:** draft
**Input:** `CounterCalcMethod`. **Output:** `Self`; unset ⇒
`counter_calc_method_default()`.

### SPEC-CATF-06 — `boost_params()` default-equivalence
**Status:** draft
**Given** a `CatBoostBuilder::new(loss)` on which NO new setter from CATF-01..05 was
called, **when** `boost_params()` is invoked, **then** every one of the seven CTR
fields equals the value produced by today's inline `*_default()` calls.
**Why isolated:** this is the single guard against the refactor (inline default →
builder field) silently changing a default. Its failure cause is distinct from any
individual setter's.

### SPEC-CATF-07 — float-only `fit()` no-regression (D-04)
**Status:** draft
**Input:** a `Pool` with `cat_features().is_empty() == true`.
**Given** such a pool, **when** `fit()` runs, **then** the produced model is
byte-identical to the model produced before this change, and `cb_train::train` (not
`train_cat`) is the entrypoint used.
**Acceptance:** `crates/catboost-rs/tests/builder_oracle_test.rs`,
`cv_oracle_test.rs`, and `grid_search_oracle_test.rs` pass **unmodified**.

### SPEC-CATF-08 — categorical `fit()` routes to `train_cat` and bakes CTR data
**Status:** draft
**Input:** a `Pool` with non-empty `cat_features()`.
**Output:** `Result<Model, CatBoostError>` whose canonical model has
`ctr_data.is_some()`.
**Given** a pool carrying categorical columns, **when** `fit()` runs, **then**
`cb_train::train_cat` is called with `pool.cat_features()` and the returned
`BakedCtrData` is attached via `CtrData::from_baked`.
**Errors:** a `train_cat` failure surfaces as `CatBoostError::Train` (existing
`#[from] cb_core::CbError`; no new variant).
**Anti-false-pass guard:** the test MUST assert `ctr_data.is_some()` AND that at least
one tree contains a `ModelSplit::Ctr` — otherwise "it trained" is satisfied by the
float path silently ignoring the categorical columns.

### SPEC-CATF-09 — setter parameter types are nameable downstream
**Status:** draft
**Given** an external crate depending only on `catboost_rs`, **when** it names
`catboost_rs::ECtrType` / `catboost_rs::CounterCalcMethod`, **then** it compiles.
(Today neither is re-exported; `EBootstrapType` at `lib.rs:61` is the precedent.)

### SPEC-CATF-10 — categorical width validation on the predict path
**Status:** draft
**Input:** `&Pool`. **Output:** `Result<Vec<Vec<String>>, CatBoostError>`.
**Given** a CTR model expecting `n` categorical features, **when** a pool with
`m != n` categorical columns is supplied, **then**
`CatBoostError::FeatureMismatch` is returned, naming both counts.
**Given** `m == n`, **then** the pool's categorical columns are returned unchanged.

### SPEC-CATF-11 — `predict_with` is CTR-aware
**Status:** draft
**Input:** `(&Pool, PredictionType)`. **Output:** `Result<Vec<f64>, CatBoostError>`.
**Given** a model with `ctr_data.is_some()` and a width-matching pool, **when**
`predict_with` runs, **then** `cb_model::predict_raw_cat` is called with the pool's
categorical columns and the result matches upstream within 1e-5.
**Given** a model with `ctr_data.is_none()`, **then** the existing `predict_raw` call
is used and output is byte-identical to today (D-04).

### SPEC-CATF-12 — a CTR model may never be silently mis-scored
**Status:** draft
**Given** a model with `ctr_data.is_some()`, **when** `predict`/`predict_with`/
`predict_proba` is called with a pool carrying **zero** categorical columns, **then** a
typed `CatBoostError::FeatureMismatch` is returned.
**Why isolated:** today this case returns wrong numbers with no error (see §1). This
specification exists to make that specific silent failure impossible, and its
acceptance test must assert the error — not merely that some result was produced.

### SPEC-CATF-13 — the five CTR scalar/enum kwargs are accepted from Python
**Status:** draft
**Input:** `one_hot_max_size: int`, `max_ctr_complexity: int`, `simple_ctr: str`,
`combinations_ctr: str`, `counter_calc_method: str` (plus the two priors kwargs if
upstream names them; otherwise priors stay Rust-only).
**Output:** a configured `CatBoostBuilder`; or `CatBoostParameterError` on an
out-of-range number or unrecognized enum string.
**Given** `CatBoostRegressor(max_ctr_complexity=2)`, **when** `fit` runs, **then** no
parity-gap error is raised and the value reaches `BoostParams`.
**Given** `simple_ctr="NotACtrType"`, **then** `CatBoostParameterError` names the
offending value and lists the accepted variants.
**Constraint:** reuse the existing `get_with_aliases` / `check_range` / `parse_*`
machinery (`parse_bootstrap_type` is the shape precedent); do not hand-roll
validation.

### SPEC-CATF-14 — `cat_features` accepted as a `fit()` kwarg
**Status:** draft
**Input:** `fit(X, y, cat_features: Optional[list[int | str]])`.
**Output:** a fitted estimator.
**Given** `CatBoostRegressor().fit(df, y, cat_features=[2])`, **when** ingestion runs,
**then** the resulting `Pool` carries that column in `cat_features()`, matching what
`Pool(df, y, cat_features=[2])` produces.
**Given** `x` is already a native `Pool` AND `cat_features` is passed, **then** the
behavior follows the existing WR-04 `Pool`-fast-path convention documented at
`crates/catboost-rs-py/src/estimator.rs:236-252` — the `Pool` is the single source of
truth. The plan MUST make this either a documented ignore consistent with `y`'s
existing treatment, or an explicit error; it may not be left ambiguous.
**Out of scope:** Arrow/Polars sources (still rejected upstream of this path).

### SPEC-CATF-15 — parameter-registry introspection stays truthful
**Status:** draft
**Given** `catboost_rs._param_status(name)` for each of the six promoted kwargs,
**then** it returns `"IMPLEMENTED"`.
**And** `crates/catboost-rs-py/tests/test_params.py::test_every_upstream_param_is_in_registry`
still passes (every upstream kwarg classified), as does
`test_known_not_yet_param_rejected_as_parity_gap` — the latter exercises
`nan_mode="Min"`, which is NOT in this change's promoted set and must remain rejected
`[VERIFIED: LOCAL crates/catboost-rs-py/tests/test_params.py:53-72]`.

### SPEC-CATF-16 — public-Python-API categorical oracle at ≤1e-5
**Status:** draft
**Given** a frozen fixture dataset with categorical columns and a pinned upstream
configuration, **when** the same configuration is trained and predicted through
`catboost_rs` (`Pool(..., cat_features=...)` → `fit` → `predict`), **then** every
prediction matches the upstream `catboost==1.2.10` reference within 1e-5.
**Constraint:** the comparison must use the model's **own**
`float_feature_borders()`, never borders shared across configurations — CatBoost
quantization borders are not stable across configurations
`[VERIFIED: memory wr01-device-bootstrap-shipped, trap #2]`.
**Constraint:** mirror `gen_fixtures.py`'s isolating discipline — `thread_count=1`,
fixed `random_seed`, `verbose=False` — and pin every builder default whose value
differs from catboost's raw dict-API default (notably `random_strength`)
`[VERIFIED: memory cv-orch01-random-strength-fixture]`.

### SPEC-CATF-17 — the oracle fixture is frozen, not regenerated
**Status:** draft
**Given** the CTR fixture generated for CATF-16, **then** it is committed under
`crates/cb-oracle/fixtures/` and consumed from disk; CI must not regenerate it.
**Rationale:** upstream CatBoost quantization is run-to-run nondeterministic, so a
regenerated CTR fixture would produce spurious failures
`[VERIFIED: memory ctr-model-loading]`.

---

## 6. Acceptance scenarios

| # | Scenario | Expected | Specs |
|---|---|---|---|
| A1 | Fit a float-only pool through the facade | Byte-identical to pre-change; existing oracle tests pass unmodified | CATF-07 |
| A2 | Fit a pool with categorical columns | Model carries `ctr_data`; ≥1 `ModelSplit::Ctr` present | CATF-08 |
| A3 | Fit categorical, then predict with the same pool | Matches upstream ≤1e-5 | CATF-11, CATF-16 |
| A4 | Predict a CTR model with a cat-free pool | Typed `FeatureMismatch`, never a silent number | CATF-12 |
| A5 | Predict a CTR model with the wrong cat width | Typed `FeatureMismatch` naming both counts | CATF-10 |
| A6 | `CatBoostRegressor(max_ctr_complexity=2).fit(...)` | Accepted; value reaches `BoostParams` | CATF-13 |
| A7 | `simple_ctr="Bogus"` | `CatBoostParameterError` listing valid variants | CATF-13 |
| A8 | `fit(df, y, cat_features=[2])` | Equivalent to `Pool(df, y, cat_features=[2])` | CATF-14 |
| A9 | `_param_status` for the six promoted kwargs | `"IMPLEMENTED"`; `nan_mode` still `KNOWN_NOT_YET` | CATF-15 |
| A10 | Builder with no CTR setter called | `boost_params()` CTR fields equal today's pinned defaults | CATF-06 |

## 7. Impact scope

**Classification: cross-module** (three crates: `catboost-rs`, `catboost-rs-py`,
plus read-only reuse of `cb-train`/`cb-model`/`cb-data`).

### Must change
- `crates/catboost-rs/src/builder.rs` — struct fields, `new()`, seven setters,
  `boost_params()`, `fit()`.
- `crates/catboost-rs/src/lib.rs` — re-exports; register any new `*_test.rs` module.
- `crates/catboost-rs/src/model.rs` — `cat_columns()`, `is_ctr_model()`,
  `predict_with()`.
- `crates/catboost-rs-py/src/params.rs` — `IMPLEMENTED`, `parse_ctr_type`,
  `parse_counter_calc_method`, `make_builder` wiring.
- `crates/catboost-rs-py/src/{estimator.rs,regressor.rs,classifier.rs,ranker.rs}` —
  `cat_features` fit-kwarg threading through `data_to_pool` → `ingest_to_owned`
  (which already accepts a `cat_features` argument, currently passed `None` at
  `estimator.rs:247`).

### Callers inheriting the new behavior with no signature change
`crates/catboost-rs/src/cv.rs:407` and `crates/catboost-rs/src/grid_search.rs:405,515`
each call `builder.fit(pool)` with an unmodified `&Pool`, so both gain categorical
routing automatically. Both must be re-run as regression gates (CATF-07).

### Verification only (must not be modified)
`cb-train` (`train_cat`, `train_inner`), `cb-model` (`predict_raw_cat`, `CtrData`),
`cb-data` (`Pool`), `crates/catboost-rs-py/src/ingest_py.rs`.

> **AMENDMENT (F08 / PLAN-CHECK revision #3, option 1) — `cb-model` is no longer
> verification-only.** Resolving CRITICAL-3 ("derived cat width is a data-dependent
> lower bound enforced as equality") requires the model to carry the pool's
> **DECLARED** categorical width rather than one derived from the splits it happened
> to choose. `crates/cb-model/src/model.rs` therefore gains:
>
> - a `cat_feature_count: usize` field (runtime-only — neither the `.cbm` encoder nor
>   `json.rs`'s serde shape writes or reads it, guarded by
>   `adding_the_cat_feature_count_does_not_change_cbm_bytes`, so every frozen
>   byte-identity baseline stays valid);
> - `#[non_exhaustive]` plus `Model::new` and the `with_*` builder surface, so this
>   field and every future one cost external crates nothing.
>
> The `#[non_exhaustive]` change forced a **one-time mechanical migration** of every
> `Model` struct literal outside `cb-model` (24 sites: 19 in `cb-model/tests`, 4 in
> `catboost-rs`, 1 in `cb-train/tests/ctr_split_scoring_test.rs`) and the addition of
> `cat_feature_count: 0` to 22 literals inside it. **No assertion was added, removed,
> weakened or reworded at any site** — verified by a zero-balance diff audit over
> assertion/panic lines. `predict_raw_cat`, `CtrData`, `train_cat` and `train_inner`
> remain untouched, so the rest of this section stands.

## 8. Compatibility and migration

- **Additive only.** No `BoostParams` field is added or removed — only the source of
  their values changes from inline defaults to builder state (CATF-06 guards
  equivalence).
- **No serialization change.** `Model`/`CtrData` shapes are untouched; CTR `.cbm`
  round-trip already shipped in Phase 23.
- **Python surface widens, never narrows:** six kwargs move from rejected to accepted;
  no previously accepted kwarg changes meaning.
- **Deliberate behavior change 1:** `predict()` on a CTR model with a cat-free pool
  moves from *silently wrong numbers* to a typed error (CATF-12). This is a fix, but
  it is technically observable; it must be called out in the change description.
  **AMENDED (F11/F12, SPEC-CATF-Δ7):** the same change applies to a **ONE-HOT**
  model, which has no `ctr_data` at all — `predict_raw(m, fv)` is
  `predict_raw_cat(m, fv, &[])`, so every one-hot split silently evaluated to
  `false`. The predicate is `needs_cat_columns()`, not `is_ctr_model()`, and it
  guards **four** entrypoints: `predict`, `predict_with`, `predict_proba` and
  `staged_predict` (the last via a new `ensure_scalar_oblivious` arm). It also
  reaches `partial_dependence` and
  `feature_importance_with_data(PredictionValuesChange)` (F13).
- **Deliberate behavior change 2 (F14, SPEC-CATF-Δ6):** `cv()`, `grid_search()`
  and `randomized_search()` on a pool with categorical columns now return a typed
  `CatBoostError::UnsupportedModel` **before any fold is fitted**, instead of
  fitting every fold and then failing from inside it — or, worse, with
  `ErrorScore::Value(NaN)` (the sklearn default), absorbing every candidate's
  failure into `error_score` and RETURNING a `SearchResult` with all-NaN scores
  and an arbitrary `best_index`. A categorical pool is a *configuration* failure,
  not a *candidate* failure, so mapping it to `error_score` is category
  confusion. `randomized_search` shares `run_over` and therefore the identical
  hazard, so it is guarded too. Float-only pools are entirely unaffected
  (`cv_on_a_float_only_pool_is_unchanged`).
- **Backend-agnostic:** `train_cat` is `R: Runtime`-generic exactly like `train`, so
  every existing backend feature (`cpu`/`wgpu`/`cuda`/`rocm`) compiles unchanged with
  no new feature gating.

## 9. Risks and open questions

| # | Risk | Mitigation | Spec |
|---|---|---|---|
| R1 | Refactoring inline defaults into builder fields silently changes a default | Dedicated default-equivalence test | CATF-06 |
| R2 | Categorical `fit()` "passes" because the float path ignored the cat columns | Assert `ctr_data.is_some()` AND a `ModelSplit::Ctr` exists | CATF-08 |
| R3 | CTR model silently mis-scored on the predict path | Typed `FeatureMismatch` required | CATF-12 |
| R4 | Non-default `ECtrType` paired with mismatched default priors | Priors setters ship in lockstep | CATF-03, CATF-04 |
| R5 | Oracle comparison uses shared quantization borders | Use each model's own `float_feature_borders()` | CATF-16 |
| R6 | Fixture regenerated in CI → nondeterministic failures | Freeze and commit the fixture | CATF-17 |
| R7 | Scope leak into the ORD-06/07 combination-CTR gating bug | Explicit non-goal; if the oracle fails ONLY for `max_ctr_complexity > 1`, stop and report — do not fix Phase 24's bug here | §2 |
| R8 | `test_params.py` coverage test goes stale | Update it in the same change | CATF-15 |

### Open questions for the planner
1. Does upstream CatBoost expose `simple_ctr_priors`/`combinations_ctr_priors` as
   standalone Python kwargs, or only as embedded components of the `simple_ctr`
   string grammar (e.g. `"Borders:Prior=0/1"`)? If the latter, CATF-13 covers the
   Rust-side setters only and the Python surface exposes the string grammar or
   defers priors. **Must be resolved against the vendored upstream `core.py` before
   implementing CATF-13.**
2. Which estimators receive the `cat_features` fit-kwarg — all four
   (`Regressor`/`Classifier`/`Ranker`/base) or only those whose upstream counterpart
   has it? Resolve from the same vendored `core.py`.
3. `Pool` + `cat_features` fit-kwarg collision: documented-ignore (consistent with
   `y`'s existing treatment) vs. explicit error. CATF-14 requires a decision, not a
   default.

## 10. Traceability and sources

- Research: `.planning/plans/catboost-builder-cat-features-routing/research.md`
  (`[VERIFIED: LOCAL]`, includes CodeGraph-verified symbol/caller evidence).
- Locked user decisions, 2026-07-31: predict-side wiring in scope; `cat_features`
  fit-kwarg in scope; CTR priors exposed in lockstep with CTR type.
- Verified source anchors: `crates/catboost-rs/src/builder.rs:259-402`;
  `crates/catboost-rs/src/model.rs:94-128`; `crates/catboost-rs/src/error.rs:31-34`;
  `crates/cb-model/src/apply.rs:370-416`; `crates/cb-train/src/boosting.rs:1950,2149`;
  `crates/cb-data/src/pool.rs:152`; `crates/catboost-rs-py/src/params.rs:42-188,290-318`;
  `crates/catboost-rs-py/src/estimator.rs:236-252`;
  `crates/catboost-rs-py/tests/test_params.py:53-72`.
- Environment: repo `.venv` has `catboost==1.2.10` and `pytest 9.1.1`
  `[VERIFIED: local command]`. The `.venv-py8` referenced in older session notes does
  not exist in this checkout.
- TreeFinder: no indexed document covers this feature (index holds 3 unrelated
  documents). This SPEC is a **pending TreeFinder update** — no write target could be
  safely identified, so it is staged locally here.

## 11. Closing verification (outside this plan's task scope)

Per project convention this batch closes with the full oracle suite executed on a
CUDA GPU runner (Google Colab T4, `~/.local/bin/colab` CLI; Kaggle `boomvector` as
fallback). That run is a release gate for the batch, not a task in this plan.
