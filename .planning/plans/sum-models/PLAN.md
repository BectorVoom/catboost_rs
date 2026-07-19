---
title: sum_models — weighted model merge (float-only oblivious, first slice) — TDD PLAN
plan_for_spec: .planning/plans/sum-models/SPEC.md
status: draft
format: markdown
plan_version: 1
updated_at: 2026-07-18T00:00:00Z
gsd_used: false
source_evidence:
  - ".planning/plans/sum-models/SPEC.md"
  - ".planning/plans/next-feature-research/research.md"
  - "CLAUDE.md (source/test separation; clippy deny-lints; anyhow ban in cb-model)"
---

# TDD Implementation Plan — `sum_models`

Plan FROM `.planning/plans/sum-models/SPEC.md`. Goal-backward: each task derives one
observable success condition from a SPEC acceptance scenario and closes exactly one
Red → Green → Refactor cycle. No production code is authored in this document.

## 0. Verified ground truth (CodeGraph / Read, this session)

- `cb_model::Model` fields: `oblivious_trees: Vec<ObliviousTree>`, `non_symmetric_trees`,
  `region_trees`, `bias: f64`, `float_feature_borders: Vec<Vec<f64>>`,
  `ctr_data: Option<CtrData>`, `approx_dimension: usize`, `class_to_label: Vec<f64>`
  `[VERIFIED: CODEGRAPH crates/cb-model/src/model.rs:272-313]`. No scale field.
- `ObliviousTree { splits: Vec<ModelSplit>, leaf_values: Vec<f64>, leaf_weights: Vec<f64> }`
  `[VERIFIED: CODEGRAPH crates/cb-model/src/model.rs:248+]`. `leaf_values` is the
  DIMENSION-MAJOR flat buffer `leaf_values[d*n_leaves+l]`; scaling EVERY element by `w`
  is correct for any `approx_dimension` `[VERIFIED: CODEGRAPH model.rs:301-305]`.
- Apply is `bias + Σ_tree leaf_values[leaf]`, NO scale multiply; per-dim Σ routes through
  `sum_f64` `[VERIFIED: CODEGRAPH crates/cb-model/src/apply.rs:305-372, 603-627]`.
- `predict_raw(model: &Model, feature_values: &[Vec<f32>]) -> Vec<f64>`
  `[VERIFIED: CODEGRAPH apply.rs:370]`.
- `cb_core::sum_f64(values: &[f64]) -> f64` — sanctioned left-to-right fold (D-08)
  `[VERIFIED: CODEGRAPH crates/cb-core/src/reduction.rs:32]`. `cb-model` already depends
  on `cb-core` `[VERIFIED: Read crates/cb-model/Cargo.toml]`.
- `ModelError` (thiserror) variants: `Deserialize`, `SchemaVersion`, `Serialize`, `Json`,
  `Core`, `Io` `[VERIFIED: CODEGRAPH crates/cb-model/src/error.rs:16-52]`. **No exhaustive
  `match ModelError` exists anywhere** — the facade uses `#[from]` and
  `catboost-rs-py to_pyerr` matches only the OUTER `FacadeError::Model(m)` arm and calls
  `m.to_string()` `[VERIFIED: CODEGRAPH catboost-rs-py/src/errors.rs:113-135 + grep: no
  `=>` arm over `ModelError::` in facade/py]`. **Adding `ModelError::Merge` is fully
  additive — zero downstream match-arm updates required** (this resolves SPEC §7's
  open blast-radius question).
- `catboost_rs::Model { inner: cb_model::Model }` with `from_canonical(inner) -> Self`
  and `as_canonical(&self) -> &cb_model::Model` (both usable to unwrap/wrap), plus
  `predict(&self, pool) -> Result<Vec<f64>, CatBoostError>`, `load_cbm`
  `[VERIFIED: CODEGRAPH crates/catboost-rs/src/model.rs:30-103]`.
- `catboost_rs::CatBoostError::Model(#[from] cb_model::ModelError)`
  `[VERIFIED: Read crates/catboost-rs/src/error.rs:44-45]`.
- `cb-model/src/lib.rs` module list + `pub use` block: add `mod model_sum;` and
  `pub use model_sum::sum_models;` `[VERIFIED: Read crates/cb-model/src/lib.rs:14-52]`.
- Fixture precedent: `crates/cb-oracle/fixtures/model_serde/regression/{config.json,
  model.cbm,model.json,predictions.npy}` + generator idiom in
  `crates/cb-oracle/fixtures/fstr_loss_change/gen_fixtures.py` (`.venv/bin/python`,
  `catboost==1.2.10`, pinned seed, `thread_count=1`, `bootstrap_type="No"`, inputs at
  `crates/cb-oracle/fixtures/inputs/numeric_tiny/{X,y}.npy`)
  `[VERIFIED: Read + ls]`.

## 1. Repository conventions this plan MUST honor

- **Source/test separation (MANDATORY):** production `.rs` files contain NO
  `#[cfg(test)] mod tests { … }` body. Unit tests live in a sibling `*_test.rs` mounted
  from the prod file via `#[cfg(test)] #[path = "model_sum_test.rs"] mod tests;`.
  Omitting the mount silently runs 0 tests.
- **Clippy is the lint gate, not build:** `unwrap`/`expect`/`panic`/`indexing_slicing`
  are workspace-denied. All slice access via checked `.get`/`.get_mut`; no `[]` indexing;
  no `unwrap`/`expect` in production paths. Scope lint to the crate:
  `cargo clippy -p cb-model --lib --no-deps` (the workspace is broadly red in untouched
  files).
- **`anyhow` is banned in `cb-model` (D-14).** Errors are typed `ModelError` only.
- **Deterministic sums via `cb_core::sum_f64` (D-08).** The bias reduction MUST use it;
  no raw `.sum()`/`.fold(0.0,+)` over floats.
- **Frozen-fixture rule:** CatBoost quantization is run-to-run nondeterministic → freeze
  the upstream `.cbm` models + `expected.npy`; use FLOAT-ONLY models for the oracle.

## 2. Execution waves & dependency order

```text
Wave 1:  TASK-01  (error variant + module bootstrap + empty-input reject)
Wave 2:  TASK-02  (single-model weight scaling)                depends TASK-01
Wave 3:  TASK-03  (N-model tree concatenation)                 depends TASK-02
Wave 4:  TASK-04  (weighted bias sum via sum_f64)              depends TASK-02
Wave 5:  TASK-05  (compatibility validation → typed error)     depends TASK-02
Wave 6:  TASK-06  (facade Model::sum_models)                   depends TASK-05
Wave 7:  TASK-08  (SM-07 oracle parity test)                   depends TASK-05, TASK-07
Optional TASK-09  (SM-06 Python surface)                       depends TASK-06

Parallel track (any time, no Rust-file conflict):
         TASK-07  (SM-07 frozen fixtures + gen_fixtures.py)    no code prereq
```

**Linear critical path:** `TASK-01 → 02 → 05 → 06 → 08`.
**Write-conflict note:** TASK-02/03/04/05 all edit `model_sum.rs` + `model_sum_test.rs`,
so they are **sequential** (not parallel) despite 03/04/05 sharing only TASK-02 as a
prerequisite. TASK-07 touches only Python + a new fixtures dir → parallel with everything.
TASK-06 touches only facade files; TASK-09 only `catboost-rs-py`.

## 3. Spec-ID → Task coverage map

| SPEC ID | Behavior | Task(s) |
|---|---|---|
| SM-01 | single-model weight scaling (leaf + bias) | TASK-02 |
| SM-02 | N-model tree concatenation | TASK-03 |
| SM-03 | weighted bias sum via `sum_f64` | TASK-04 |
| SM-04 | compatibility validation → `ModelError::Merge` | TASK-01 (empty), TASK-05 (rest) |
| SM-05 | facade `catboost_rs::Model::sum_models` | TASK-06 |
| SM-06 | Python `catboost_rs.sum_models` (OPTIONAL) | TASK-09 |
| SM-07 | oracle parity ≤1e-5 vs `catboost.sum_models` | TASK-07 (fixtures), TASK-08 (test) |

Every task maps back to ≥1 SPEC ID; every SPEC ID maps to ≥1 task.

---

## TASK-01 — Bootstrap module + `ModelError::Merge` + empty-input rejection

- **SPEC refs:** SM-04 (empty-`models` case), scaffolds SM-01/02/03/05/07.
- **Goal / observable completion:** `cb_model::sum_models(&[], &[])` returns
  `Err(ModelError::Merge(_))`; the new module, its mounted test file, and the `lib.rs`
  re-export exist and compile clean under clippy.
- **Depends on:** none (Wave 1).
- **Files:**
  - Create: `crates/cb-model/src/model_sum.rs` — `pub fn sum_models(models: &[&Model],
    weights: &[f64]) -> Result<Model, ModelError>`; at top of file mount tests:
    `#[cfg(test)] #[path = "model_sum_test.rs"] mod tests;`.
  - Create: `crates/cb-model/src/model_sum_test.rs` — `use super::*;` unit tests.
  - Modify: `crates/cb-model/src/error.rs` — add variant
    `#[error("models cannot be merged: {0}")] Merge(String)` to `enum ModelError`.
  - Modify: `crates/cb-model/src/lib.rs` — add `mod model_sum;` (with the other `mod`s,
    lines 14-24) and `pub use model_sum::sum_models;` (in the `pub use` block, lines 26-52).
- **Red:**
  - Test `sum_models_rejects_empty` in `model_sum_test.rs`: call `sum_models(&[], &[])`;
    assert the result is `Err` and `matches!(err, ModelError::Merge(_))`.
  - Expected initial failure: **compile error** — `model_sum` module / `sum_models` /
    `ModelError::Merge` do not yet exist (`E0432`/`E0433`/unknown-variant).
- **Green (minimal intent):** add the `Merge` variant; create `model_sum.rs` with
  `sum_models` that (a) if `models.is_empty()` returns
  `Err(ModelError::Merge("no models to sum".into()))`; (b) otherwise returns a
  placeholder `Ok(models[0].clone())` accessed via `models.first()` +
  `.ok_or_else(...)` (NO `[]` indexing) — correct scaling/concat/bias arrive in
  TASK-02/03/04. Wire the `lib.rs` `mod` + `pub use`.
- **Refactor constraints:** keep the empty-check first; no behavior beyond empty
  rejection + clone placeholder. Regression scope: `cargo test -p cb-model --lib`.
- **Validation:**
  - `cargo test -p cb-model --lib sum_models`
  - `cargo clippy -p cb-model --lib --no-deps` (must be clean for the new file)
- **Completion evidence:** `sum_models_rejects_empty` passes; clippy clean; `Merge`
  variant visible in `ModelError`; `cb_model::sum_models` importable.
- **Parallelization:** none (Wave 1 root).

---

## TASK-02 — SM-01: single-model weight scaling (leaf values + bias)

- **SPEC refs:** SM-01; resolves Q1 (empty `weights` ⇒ all-ones).
- **Goal / observable completion:** `sum_models(&[&m], &[w])` yields a model whose every
  `oblivious_trees[t].leaf_values[l] == w * m…leaf_values[l]` and
  `bias == w * m.bias`, everything else structurally identical; and with empty `weights`
  the effective weight is `1.0`.
- **Depends on:** TASK-01.
- **Files:** Modify `crates/cb-model/src/model_sum.rs`,
  `crates/cb-model/src/model_sum_test.rs`.
- **Red:**
  - Test `sum_models_single_scales_leaves`: build a tiny float-only oblivious `Model`
    in-memory (2 trees, `approx_dimension = 1`, `ctr_data = None`, non-symmetric/region
    empty, small `float_feature_borders`, known `leaf_values`, `bias = 0.5`). Call
    `sum_models(&[&m], &[2.0])`. Assert each result leaf value ≈ `2.0 * source` (≤1e-12),
    `result.bias ≈ 1.0`, `result.oblivious_trees.len() == m.oblivious_trees.len()`, and
    each tree's `splits == source.splits`.
  - Add case in same test (or `sum_models_empty_weights_defaults_ones`): `sum_models(&[&m],
    &[])` equals `sum_models(&[&m], &[1.0])` (leaf/bias unchanged from source).
  - Expected initial failure: assertion fails — TASK-01 placeholder clones without scaling,
    so `result.leaf_values == source` (not `2×`) and `bias == 0.5` (not `1.0`).
- **Green (minimal intent):** compute an effective-weights vec: if `weights.is_empty()`
  use `vec![1.0; models.len()]`, else use `weights` (length validation deferred to
  TASK-05). For the single-model path, clone the model then map every tree's
  `leaf_values` element to `w * v` (checked iteration, no `[]`), and set
  `bias = w * m.bias`. Carry `float_feature_borders`, `approx_dimension`,
  `class_to_label`, `leaf_weights` UNCHANGED (R3: `leaf_weights` unscaled — this slice
  predicts only). Keep `ctr_data = None`, `non_symmetric_trees = []`, `region_trees = []`.
- **Refactor constraints:** factor a private `scaled_tree(tree: &ObliviousTree, w: f64)
  -> ObliviousTree` helper reused by TASK-03. No `[]`/`unwrap`. Regression:
  `cargo test -p cb-model --lib`.
- **Validation:** `cargo test -p cb-model --lib sum_models`;
  `cargo clippy -p cb-model --lib --no-deps`.
- **Completion evidence:** both single-model tests pass; helper `scaled_tree` present;
  clippy clean.
- **Parallelization:** none (shares `model_sum.rs` with TASK-03/04/05).

---

## TASK-03 — SM-02: N-model tree concatenation

- **SPEC refs:** SM-02.
- **Goal / observable completion:** for ≥2 compatible models,
  `result.oblivious_trees.len() == Σ_i models[i].oblivious_trees.len()` in input order,
  each concatenated tree's `splits` byte-identical to its source and its `leaf_values`
  scaled by that model's weight; `rawpredict(result,x) == Σ_i w_i·rawpredict(m_i,x)`.
- **Depends on:** TASK-02.
- **Files:** Modify `crates/cb-model/src/model_sum.rs`,
  `crates/cb-model/src/model_sum_test.rs`.
- **Red:**
  - Test `sum_models_concats_trees`: build two compatible float-only models `m0` (a trees)
    and `m1` (b trees) sharing identical `float_feature_borders`/`approx_dimension`/
    `class_to_label`. Call `sum_models(&[&m0,&m1], &[0.5,1.5])`. Assert
    `result.oblivious_trees.len() == a + b`; the first `a` trees' `splits` equal `m0`'s and
    their leaves equal `0.5×`; the next `b` equal `m1`'s scaled `1.5×`. Then assert
    end-to-end: for a fixed `&[Vec<f32>]` input,
    `predict_raw(&result, x) ≈ 0.5*predict_raw(&m0,x) + 1.5*predict_raw(&m1,x)` (≤1e-9)
    — note this also exercises bias summation, which lands correct in TASK-04; keep the
    bias of both inputs `0.0` in THIS test so the concat assertion is isolated from SM-03.
  - Expected initial failure: `result.oblivious_trees.len() == a` (TASK-02 handles only
    `models[0]`), so the length assertion fails.
- **Green (minimal intent):** iterate `models.iter().zip(effective_weights)`, and for each
  `(m, w)` extend the result `oblivious_trees` with `m.oblivious_trees.iter().map(|t|
  scaled_tree(t, w))`. Take structural fields (`float_feature_borders`,
  `approx_dimension`, `class_to_label`) from the first model (checked `.first()`).
- **Refactor constraints:** no `[]`/`unwrap`; single pass building the concatenated vec.
  Regression: `cargo test -p cb-model --lib`.
- **Validation:** `cargo test -p cb-model --lib sum_models`;
  `cargo clippy -p cb-model --lib --no-deps`.
- **Completion evidence:** `sum_models_concats_trees` passes; result tree count = sum.
- **Parallelization:** none (shares `model_sum.rs`).

---

## TASK-04 — SM-03: weighted bias sum via `cb_core::sum_f64`

- **SPEC refs:** SM-03.
- **Goal / observable completion:** `result.bias == sum_f64(&[w_0*b_0, …, w_{n-1}*b_{n-1}])`
  in input order (deterministic D-08 reduction).
- **Depends on:** TASK-02.
- **Files:** Modify `crates/cb-model/src/model_sum.rs`,
  `crates/cb-model/src/model_sum_test.rs`.
- **Red:**
  - Test `sum_models_sums_bias`: two compatible models with `bias` `b0 = 0.25`,
    `b1 = -0.75`, weights `[2.0, 4.0]`. Call `sum_models`. Assert
    `result.bias == cb_core::sum_f64(&[2.0*0.25, 4.0*-0.75])` (bit-exact `==`, not approx —
    determinism is the contract).
  - Expected initial failure: after TASK-03, `result.bias` is taken from the first model's
    scaled bias only (or `0.0`), not the weighted sum → assertion fails.
- **Green (minimal intent):** build `let biased: Vec<f64> = models.iter().zip(weights)
  .map(|(m,&w)| w * m.bias).collect();` then `result.bias = cb_core::sum_f64(&biased);`.
  Remove any per-model bias assignment introduced earlier so bias is set once here.
- **Refactor constraints:** MUST route through `cb_core::sum_f64` (no raw fold). No
  `[]`/`unwrap`. Regression: `cargo test -p cb-model --lib` (re-run TASK-03's end-to-end
  assertion with non-zero biases now that SM-03 is live).
- **Validation:** `cargo test -p cb-model --lib sum_models`;
  `cargo clippy -p cb-model --lib --no-deps`.
- **Completion evidence:** `sum_models_sums_bias` passes bit-exact; `sum_f64` referenced.
- **Parallelization:** none (shares `model_sum.rs`).

---

## TASK-05 — SM-04: compatibility validation → `ModelError::Merge`

- **SPEC refs:** SM-04 (all cases except empty, which is TASK-01).
- **Goal / observable completion:** `sum_models` returns `Err(ModelError::Merge(_))` for
  every incompatible input and `Ok` only when all preconditions hold; never panics.
- **Depends on:** TASK-02 (needs the working merge to guard).
- **Files:** Modify `crates/cb-model/src/model_sum.rs`,
  `crates/cb-model/src/model_sum_test.rs`.
- **Red (one assertion/`#[test]` per rejection case; principal failure reason = "an
  incompatible model is not rejected with a typed `Merge` error"):**
  - `sum_models_rejects_weight_count_mismatch`: `weights` non-empty, `weights.len() !=
    models.len()` → `Err(Merge)`.
  - `sum_models_rejects_non_oblivious`: a model with non-empty `non_symmetric_trees` OR
    `region_trees` → `Err(Merge)`.
  - `sum_models_rejects_ctr_model`: a model with `ctr_data.is_some()` → `Err(Merge)`.
  - `sum_models_rejects_border_mismatch`: models with differing `float_feature_borders`
    → `Err(Merge)`.
  - `sum_models_rejects_approx_dim_mismatch`: differing `approx_dimension` → `Err(Merge)`.
  - `sum_models_rejects_class_to_label_mismatch`: differing `class_to_label` → `Err(Merge)`.
  - Expected initial failure: pre-guard code either returns `Ok` with a wrong-valued merge
    or (for the non-oblivious/ctr cases) silently drops data → the `matches!(…, Err(
    ModelError::Merge(_)))` assertions fail.
- **Green (minimal intent):** at the top of `sum_models`, after the empty-check, validate
  in order and return `Err(ModelError::Merge(msg))` on the first failure:
  (1) if `!weights.is_empty() && weights.len() != models.len()`;
  (2) for each model: `!non_symmetric_trees.is_empty() || !region_trees.is_empty()` →
      reject (non-oblivious); `ctr_data.is_some()` → reject (CTR unsupported this slice);
  (3) take the first model as reference (checked `.first()`) and reject any model whose
      `float_feature_borders`, `approx_dimension`, or `class_to_label` differs
      (`!=` on the owned `Vec`/`usize`). Only then run the TASK-02/03/04 merge. Every
      message names the offending precondition.
- **Refactor constraints:** factor a private `validate(models, weights) ->
  Result<(), ModelError>` returning the reference model index/handle; no `[]`/`unwrap`;
  no panic on any path. Regression: `cargo test -p cb-model --lib`.
- **Validation:** `cargo test -p cb-model --lib sum_models`;
  `cargo clippy -p cb-model --lib --no-deps`.
- **Completion evidence:** all six `sum_models_rejects_*` tests + TASK-01's empty test
  pass; the happy-path tests (TASK-02/03/04) still pass; clippy clean.
- **Parallelization:** none (shares `model_sum.rs`). **This completes the `cb_model` core
  function** — TASK-06 and TASK-08 gate on it.

---

## TASK-06 — SM-05: facade `catboost_rs::Model::sum_models`

- **SPEC refs:** SM-05.
- **Goal / observable completion:** `catboost_rs::Model::sum_models(&[&a,&b], None)`
  returns a facade `Model` whose `predict` equals the weighted sum of the inputs' raw
  predictions; `weights: None` ⇒ all-ones; a `ModelError::Merge` surfaces as
  `CatBoostError::Model`.
- **Depends on:** TASK-05 (full core function).
- **Files:**
  - Modify: `crates/catboost-rs/src/model.rs` — add associated fn to `impl Model`:
    `pub fn sum_models(models: &[&Model], weights: Option<&[f64]>) ->
    Result<Model, CatBoostError>`; add `sum_models` to the existing
    `use cb_model::{…}` import list (lines 17-21).
  - Create: `crates/catboost-rs/src/model_sum_test.rs` (or extend an existing facade test
    module) mounted per the source/test-separation rule; if `model.rs` has no mount yet,
    add `#[cfg(test)] #[path = "model_sum_test.rs"] mod sum_models_tests;`.
- **Red:**
  - Test `sum_models_facade_roundtrip`: load/construct two compatible facade `Model`s
    (build canonical models in-memory and wrap via `Model::from_canonical`, or load two
    frozen `.cbm` via `Model::load_cbm`). Call `Model::sum_models(&[&a,&b], None)`; then
    for a fixed `Pool`, assert `merged.predict(&pool)? ≈ a.predict(&pool)? +
    b.predict(&pool)?` (≤1e-9). Add `sum_models_facade_maps_error`: pass an incompatible
    pair and assert `matches!(err, CatBoostError::Model(_))`.
  - Expected initial failure: compile error — `Model::sum_models` does not exist on the
    facade.
- **Green (minimal intent):** in the facade fn, map `models` to
  `Vec<&cb_model::Model>` via `m.as_canonical()`, resolve `weights` (`None` → `&[]`, which
  the core treats as all-ones), call `cb_model::sum_models(&canon, w)?` (the `?` converts
  `ModelError` → `CatBoostError::Model` via the existing `#[from]`), and wrap the result
  with `Model::from_canonical`.
- **Refactor constraints:** no `[]`/`unwrap`; the fn body is a thin adapter (unwrap →
  call → wrap). Regression: `cargo test -p catboost-rs`.
- **Validation:** `cargo test -p catboost-rs sum_models`;
  `cargo clippy -p catboost-rs --lib --no-deps`.
- **Completion evidence:** both facade tests pass; `Model::sum_models` public.
- **Parallelization:** independent of TASK-07/08 files; depends on TASK-05.

---

## TASK-07 — SM-07 fixtures: frozen upstream models + `gen_fixtures.py`

- **SPEC refs:** SM-07 (data half). NO Rust production code.
- **Goal / observable completion:** a committed, frozen fixture set exists under
  `crates/cb-oracle/fixtures/model_sum/` produced by a pinned generator, ready for the
  oracle test: two float-only oblivious `.cbm` models, the feature matrix, per-weight
  expected predictions, and a `config.json`.
- **Depends on:** none (parallel track); logically precedes TASK-08.
- **Files (create):**
  - `crates/cb-oracle/fixtures/model_sum/gen_fixtures.py` — mirrors
    `fstr_loss_change/gen_fixtures.py`: `.venv/bin/python`, `import catboost as cb`,
    `catboost==1.2.10`, `numpy<2`, pinned `random_seed`, `thread_count=1`,
    `bootstrap_type="No"`, `boost_from_average` explicit; inputs from
    `crates/cb-oracle/fixtures/inputs/numeric_tiny/{X,y}.npy`.
  - `crates/cb-oracle/fixtures/model_sum/m0.cbm`, `m1.cbm` — two float-only
    `CatBoostRegressor`s (numeric features only; **no** cat features → deterministic,
    `ctr_data` absent), saved `format="cbm"`.
  - `crates/cb-oracle/fixtures/model_sum/X.npy` — the fixed eval matrix (f64).
  - `crates/cb-oracle/fixtures/model_sum/expected_m0.npy` — upstream
    `m0.predict(X, prediction_type="RawFormulaVal")` (per-model baseline, isolates
    apply from merge arithmetic in TASK-08).
  - `crates/cb-oracle/fixtures/model_sum/expected_m1.npy` — upstream
    `m1.predict(X, prediction_type="RawFormulaVal")`.
  - `crates/cb-oracle/fixtures/model_sum/expected_w_1_1.npy` — upstream
    `cb.sum_models([m0,m1],[1.0,1.0]).predict(X, prediction_type="RawFormulaVal")`.
  - `crates/cb-oracle/fixtures/model_sum/expected_w_03_07.npy` — upstream
    `cb.sum_models([m0,m1],[0.3,0.7]).predict(X, prediction_type="RawFormulaVal")`.
  - `crates/cb-oracle/fixtures/model_sum/config.json` — records `catboost_version`,
    input dataset, the two weight vectors, seeds, the fixture filenames, AND the
    explicit `"prediction_type": "RawFormulaVal"` used for EVERY expected `.npy`
    (mirror `model_serde/regression/config.json` shape).
- **Prediction-type pin (MANDATORY):** the Rust side compares via `predict_raw`
  (`RawFormulaVal`), so EVERY upstream expected — per-model AND summed — MUST be
  generated with `prediction_type="RawFormulaVal"` (never the default probability/class
  for a classifier). `CatBoostRegressor` already returns raw values, but pass the kwarg
  explicitly so the fixture is unambiguous and record it in `config.json`.
- **Generation recipe (run once, offline; NOT part of `cargo build`):**
  - `uv venv --python 3.12 && uv pip install catboost==1.2.10 'numpy<2'`
  - `.venv/bin/python crates/cb-oracle/fixtures/model_sum/gen_fixtures.py`
- **R2 check (SPEC §9):** at generation time, confirm the real `catboost.sum_models`
  signature/defaults (`sum_models(models, weights=None, ctr_merge_policy=…)`) against the
  installed `catboost==1.2.10` (`help(cb.sum_models)`); record the confirmed signature in
  `config.json`. Use only float-only models so `ctr_merge_policy` is irrelevant.
- **Red / Green:** N/A (fixture data task). "Red" analogue: TASK-08 cannot run until these
  files exist. "Green": the generator writes all listed artifacts and they are committed.
- **Validation:** re-running the generator reproduces byte-stable `.npy` expecteds for the
  frozen `.cbm` (predictions on a saved model are deterministic even though quantization
  is not — that is why the `.cbm` is FROZEN, not regenerated per test run).
- **Completion evidence:** all listed files present; `config.json` documents the confirmed
  upstream signature + weights.
- **Parallelization:** fully parallel with all Rust tasks (touches only Python + a new
  fixtures dir).

---

## TASK-08 — SM-07: oracle parity test (≤1e-5) vs `catboost.sum_models`

- **SPEC refs:** SM-07 (test half). Covers acceptance scenarios 1 & 2.
- **Goal / observable completion:** an integration test loads the frozen `m0.cbm`/`m1.cbm`,
  runs `cb_model::sum_models` for both weight vectors, predicts on `X.npy`, and asserts
  `max|rust − expected| ≤ 1e-5` against each frozen `expected_*.npy` — AND first asserts
  each input model's own `predict_raw` matches its frozen `expected_m{0,1}.npy`, so a
  failure isolates a merge-arithmetic defect from an apply defect.
- **Depends on:** TASK-05 (core fn) AND TASK-07 (fixtures).
- **Files:** Create `crates/cb-model/tests/model_sum_oracle_test.rs` (integration test;
  uses the `cb-oracle` dev-dependency already present in `cb-model`).
- **Red:**
  - Test `sum_models_oracle_inputs_apply` (per-model sanity, runs FIRST): load both `.cbm`
    via `cb_model::load_cbm`, load `X.npy` into `&[Vec<f32>]` columns, assert
    `predict_raw(&m0, &cols)` matches `expected_m0.npy` and `predict_raw(&m1, &cols)`
    matches `expected_m1.npy`, each `|diff| ≤ 1e-5`. This proves apply itself is correct
    BEFORE any merge, so if the summed assertions below fail, the defect is provably in
    `sum_models` arithmetic, not `predict_raw`.
  - Test `sum_models_oracle_w_1_1` and `sum_models_oracle_w_03_07`: build `&[&Model]`, call
    `cb_model::sum_models(&[&m0,&m1], &w)?`, `predict_raw(&merged, &cols)`, load the
    matching summed `expected_w_*.npy`, assert element-wise `|diff| ≤ 1e-5` via the
    `cb_oracle` compare helper (or `approx`).
  - Expected initial failure: before TASK-07 the fixtures are absent (file-open error) /
    before TASK-05 the merge is wrong (summed diff > 1e-5 while the per-model sanity test
    stays green — the isolation signal). With both done, the assertions are the live gate.
- **Green (minimal intent):** N/A for production code — this task only authors the test.
  It passes once TASK-05 + TASK-07 are complete and correct. If it fails with both
  complete, the defect is in the core merge (reopen TASK-02/03/04/05), NOT in this test.
- **Refactor constraints:** load columns via checked iteration (no `[]`); reuse
  `cb_oracle` npy readers as `fstr_loss_change`/`model_serde` oracle tests do.
- **Validation:** `cargo test -p cb-model --test model_sum_oracle_test`.
- **Completion evidence:** both oracle tests pass at ≤1e-5.
- **Parallelization:** gated on TASK-05 + TASK-07; independent of TASK-06/09.

---

## TASK-09 — SM-06 (OPTIONAL, DEFERRABLE): Python `catboost_rs.sum_models`

- **SPEC refs:** SM-06. **Explicitly deferrable** — may ship as a follow-up if the first
  slice's budget excludes the Python module wiring (SPEC SM-06 note).
- **Goal / observable completion:** a module-level Python function
  `catboost_rs.sum_models(models, weights=None)` returns a fitted estimator whose
  `.predict(X)` equals the weighted sum of the inputs' predictions (≤1e-5).
- **Depends on:** TASK-06 (facade fn).
- **Files:**
  - Modify: `crates/catboost-rs-py/src/` (the estimator/module surface, e.g. a new
    `sum_models` `#[pyfunction]` registered in the `#[pymodule]`, mirroring how
    `load_model_path` / estimators wrap facade `Model`; unwrap each Python estimator's
    facade `Model`, call `catboost_rs::Model::sum_models`, wrap the result via
    `CatBoostEstimator::from_model` — `[VERIFIED: CODEGRAPH estimator.rs:56 from_model,
    296 load_model_path]`). Errors convert via `PyCbError`/`to_pyerr` (the existing
    `FacadeError::Model` arm already handles it — no `to_pyerr` change needed).
  - Create: a Python test `crates/catboost-rs-py/tests/test_sum_models.py` (marked
    "needs built extension").
- **Red:** `test_sum_models_predict`: fit two `CatBoostRegressor`s on a small `X,y`;
  `merged = catboost_rs.sum_models([a, b])`; assert
  `merged.predict(X) ≈ a.predict(X) + b.predict(X)` (≤1e-5). Initial failure:
  `AttributeError` — `catboost_rs.sum_models` not registered.
- **Green (minimal intent):** add the `#[pyfunction]`, register it in the module, unwrap →
  call facade `Model::sum_models` → wrap in the estimator.
- **Refactor constraints:** GIL discipline per D-11 (own inputs before any detach);
  reuse `PyCbError`. No `ctr_merge_policy` kwarg (out of scope).
- **Validation:** build the extension (`maturin develop` in the py crate's venv) then
  `pytest crates/catboost-rs-py/tests/test_sum_models.py`. **Deferrable:** if not built
  this slice, record as a tracked follow-up; TASK-01..08 are the shippable core.
- **Completion evidence:** the py test passes against the built extension, OR the task is
  explicitly deferred in the phase progress notes.
- **Parallelization:** depends on TASK-06; independent of TASK-07/08.

---

## 4. Consistency self-check

- Every SPEC ID (SM-01..SM-07) maps to ≥1 task; every task cites ≥1 SPEC ID (§3).
- Dependency graph is acyclic; critical path `01→02→05→06→08`; `07` parallel; `09` optional.
- Each implementation task (01–06, 08–09) has Red / Green / Refactor / Validation; TASK-07
  is a data-only task with an explicit generation recipe and no production code.
- All validation commands are repository-verified: `cargo test -p cb-model --lib`,
  `cargo test -p cb-model --test model_sum_oracle_test`,
  `cargo clippy -p cb-model --lib --no-deps`, `cargo test -p catboost-rs`.
- Every referenced production symbol/path exists and is marked Create vs Modify.
- `ModelError::Merge` is additive with ZERO downstream match-arm updates (verified §0).
- No production code authored here. No GSD skill/command/workflow/agent used.

## 5. Unresolved blockers / assumptions carried from SPEC

- **A1 (scale field — UNCHECKABLE assumption, NOT a typed error):** per SPEC §2
  reconciliation, the canonical `Model` has **no scale field**, so a non-default baked
  scale is **impossible to detect at merge time** and MUST NOT be a `ModelError::Merge`
  case. No task asserts a scale typed-error check — the SM-04 validation set (TASK-01
  empty; TASK-05 weight-count / non-oblivious / CTR / border / approx_dimension /
  class_to_label) contains **no scale arm**. The first slice simply *assumes* scale==1
  (true for all workspace-trained/loaded float models); the oracle (TASK-08) is the only
  backstop for any deviation. This intentionally differs from the border/dim/class
  mismatches, which ARE checkable fields and therefore ARE typed errors.
- **A2 (R2 upstream signature):** exact `catboost.sum_models` defaults confirmed at
  TASK-07 generation time against installed `catboost==1.2.10`; recorded in `config.json`.
- **A3 (R3 leaf_weights):** merged `leaf_weights` carried UNSCALED (predict-only slice);
  documented limitation if the merged model is later fed to PredictionValuesChange.
- **A4 (Q1):** empty `weights` ⇒ all-ones (adopted; tested in TASK-02).
- **No hard blockers.** TreeFinder MCP is unavailable this session (per SPEC front-matter);
  the local SPEC is the authoritative draft — no TreeFinder sync attempted or claimed.
