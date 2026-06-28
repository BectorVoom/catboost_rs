# Phase 8: Python Bindings, Dual API & Packaging - Pattern Map

**Mapped:** 2026-06-21
**Files analyzed:** 19 new (1 modified)
**Analogs found:** 14 / 19 (5 are net-new PyO3/packaging surfaces with no in-repo analog → use RESEARCH.md patterns)

This phase adds a new PyO3 `cdylib` workspace member (`crates/catboost-rs-py`) that wraps the existing `catboost-rs` facade into a CatBoost-mirror Python package. The single richest in-repo source of patterns is the **`catboost-rs` facade crate** (`lib.rs`, `builder.rs`, `model.rs`, `error.rs`, its `Cargo.toml`) plus the **`cb-data` ingestion seam** (`ingest/mod.rs`, `ingest/owned.rs`, `ingest/arrow.rs`) and the **`cb-backend` feature table**. PyO3-specific surfaces (`#[pyclass]`, `create_exception!`, maturin `pyproject.toml`, pytest suite) have NO in-repo analog and must follow RESEARCH.md §Architecture Patterns 1–4.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/catboost-rs-py/Cargo.toml` | config (workspace member, cdylib+rlib, cpu/rocm features) | n/a | `crates/cb-backend/Cargo.toml` (features) + `crates/catboost-rs/Cargo.toml` (facade deps + `[lints]`) | exact (features) + role-match (deps) |
| `crates/catboost-rs-py/src/lib.rs` | PyO3 `#[pymodule]` entrypoint | request-response | `crates/catboost-rs/src/lib.rs` (facade module root, re-exports) | role-match (module-root composition); PyO3 `#[pymodule]` body has no analog |
| `crates/catboost-rs-py/src/errors.rs` | error-mapping module (`create_exception!` + `From<CatBoostError> for PyErr`) | transform | `crates/catboost-rs/src/error.rs` (typed `thiserror` enum = the SOURCE to map from) | exact (the variant-by-variant source) |
| `crates/catboost-rs-py/src/params.rs` | param-vocabulary registry (D-07) + kwargs→Builder map | transform | `crates/catboost-rs/src/builder.rs` (setter surface = the kwargs targets; `*_default()` defaults) | role-match (kwargs map target) |
| `crates/catboost-rs-py/src/estimator.rs` | shared `#[pyclass]` base logic (params store, get/set, tags, clone, NotFitted) | request-response | `crates/catboost-rs/src/builder.rs` (field-store + setter idiom) | partial (store idiom); sklearn-contract `#[pyclass]` glue has no analog |
| `crates/catboost-rs-py/src/classifier.rs` | `#[pyclass]` estimator wrapper (`CatBoostClassifier`) | request-response | `crates/catboost-rs/src/builder.rs` + `model.rs` (`fit`→`predict_proba`) | role-match (wraps Builder+Model); `#[pyclass]` glue has no analog |
| `crates/catboost-rs-py/src/regressor.rs` | `#[pyclass]` estimator wrapper (`CatBoostRegressor`) | request-response | `crates/catboost-rs/src/builder.rs` + `model.rs` (`fit`→`predict`) | role-match |
| `crates/catboost-rs-py/src/ranker.rs` | `#[pyclass]` estimator wrapper (`CatBoostRanker`) | request-response | `crates/catboost-rs/src/builder.rs` + `model.rs` (group_id/pairs path) | role-match |
| `crates/catboost-rs-py/src/pool.rs` | native `Pool` `#[pyclass]` wrapper | CRUD / batch | `crates/cb-data/src/ingest/owned.rs` (`OwnedColumns` `with_*` builder → `into_pool`) | exact (the build target) |
| `crates/catboost-rs-py/src/ingest_py.rs` | input-ingestion adapter (NumPy/Pandas/Arrow/Polars → `IngestSource`) | transform / file-I/O-like | `crates/cb-data/src/ingest/{owned.rs,arrow.rs}` (`impl IngestSource` + `into_pool` validation) | role-match (new `impl IngestSource`); zero-copy borrow glue has no analog |
| `crates/catboost-rs-py/src/*_test.rs` | test (Rust unit) | n/a | `crates/catboost-rs/src/error_test.rs` (separated `_test.rs`, `#[cfg(test)] mod` in root) | exact |
| `crates/catboost-rs-py/pyproject.toml` | packaging config (maturin) | n/a | NONE (first Python packaging surface in repo) | no analog → RESEARCH §Standard Stack |
| `crates/catboost-rs-py/tests/test_check_estimator.py` | test (pytest, sklearn gate) | n/a | NONE | no analog → RESEARCH §Validation |
| `crates/catboost-rs-py/tests/test_oracle_parity.py` | test (pytest, ≤1e-5) | n/a | NONE (Rust-side oracle harness exists in `cb-oracle`, not Python) | no analog → RESEARCH §Validation |
| `crates/catboost-rs-py/tests/test_ingestion.py` | test (pytest) | n/a | NONE | no analog |
| `crates/catboost-rs-py/tests/test_errors.py` | test (pytest) | n/a | NONE | no analog |
| `crates/catboost-rs-py/tests/test_free_threaded.py` | test (pytest, 3.13t) | n/a | NONE | no analog |
| `crates/catboost-rs-py/tests/conftest.py` | test fixtures | n/a | NONE | no analog |
| `Cargo.toml` (workspace root) | config (MODIFIED — members glob already covers `crates/*`) | n/a | self (`members = ["crates/*"]` already includes the new crate) | exact — likely NO edit needed |

## Pattern Assignments

### `crates/catboost-rs-py/Cargo.toml` (config — cdylib+rlib, cpu/rocm features)

**Analog A (feature table — exact):** `crates/cb-backend/Cargo.toml` lines 26-38. This is the canonical "forward backend selection, never pin cpu unconditionally" pattern (the prior-phase feature-unification landmine). The new crate forwards to the facade's passthrough, which forwards to `cb-backend`:

```toml
[features]
# Compile-time backend selection only — NO runtime dispatch (D-02). Each backend
# feature forwards to its cubecl facade feature: selection lives HERE, not on the
# workspace cubecl pin, so `--no-default-features --features rocm` is cpu-free.
default = ["cpu"]
cpu = ["cubecl/cpu"]
wgpu = ["cubecl/wgpu"]
cuda = ["cubecl/cuda"]
rocm = ["cubecl/hip"]
```

For the binding, this becomes `cpu = ["catboost-rs/<cpu-passthrough>"]` / `rocm = ["catboost-rs/<rocm-passthrough>"]` — NEVER an unconditional `cpu`. (Note: confirm at planning whether `catboost-rs` facade currently re-exposes backend features as passthroughs; if not, that passthrough is a small facade `Cargo.toml` addition the planner must call out.)

**Analog B (facade deps + lints — exact):** `crates/catboost-rs/Cargo.toml` lines 7-19:

```toml
[package]
name = "catboost-rs"
version = "0.1.0"
edition = "2021"

[lints]
workspace = true              # <-- mandatory: opts into the workspace clippy gate

[dependencies]
cb-data = { path = "../cb-data" }
thiserror.workspace = true
```

Every workspace member uses `edition = "2021"`, `version = "0.1.0"`, `[lints] workspace = true`. The new crate MUST carry `[lints] workspace = true` so the `unwrap`/`expect`/`panic`/`indexing_slicing` deny lints apply to PyO3 glue too (CONTEXT §Established Patterns). Add the `[lib] crate-type = ["cdylib", "rlib"]` block and the pyo3/numpy/pyo3-arrow deps per RESEARCH §Standard Stack (lines 105-123).

---

### `crates/catboost-rs-py/src/lib.rs` (PyO3 `#[pymodule]` entrypoint)

**Analog:** `crates/catboost-rs/src/lib.rs` (the whole file, 47 lines) — the module-root composition + re-export idiom, plus the mandatory crate-level test-allow attribute and the `#[cfg(test)] mod *_test` declaration:

```rust
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing))]
//! crate-level doc comment

mod builder;
mod error;
mod model;

pub use builder::CatBoostBuilder;
pub use error::CatBoostError;
pub use model::Model;

#[cfg(test)]
mod error_test;
```

**Copy:** the `#![cfg_attr(test, allow(...))]` header verbatim (the in-code test exemption required because manifest-level per-crate overrides are forbidden when `lints.workspace = true` — see workspace `Cargo.toml` comment lines 4-7), the `mod`/`pub use` composition, and the `#[cfg(test)] mod <name>_test;` declarations.

**No analog (use RESEARCH Pattern 3 + §Recommended Project Structure):** the `#[pymodule]` body registering classes + exceptions and the `gil_used = false` flag — see RESEARCH lines 293-296, 349.

---

### `crates/catboost-rs-py/src/errors.rs` (error-mapping — PYAPI-05)

**Analog (the mapping SOURCE — exact):** `crates/catboost-rs/src/error.rs` lines 32-71. This is the typed enum the `From<CatBoostError> for PyErr` impl matches on, variant by variant:

```rust
#[derive(Debug, thiserror::Error)]
pub enum CatBoostError {
    #[error("training error: {0}")]
    Train(#[from] cb_core::CbError),
    #[error("model error: {0}")]
    Model(#[from] cb_model::ModelError),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("malformed model: {0}")]
    Deserialize(String),
    #[error("unsupported model schema: {0}")]
    SchemaVersion(String),
    #[error("feature mismatch: {0}")]
    FeatureMismatch(String),
}
```

**Mapping guidance (one Rust variant → one Python exception, PYAPI-05):** the variant set is exactly six — `Train`, `Model`, `Io`, `Deserialize`, `SchemaVersion`, `FeatureMismatch`. Per RESEARCH Pattern 3 (lines 290-310): `FeatureMismatch` → `CatBoostValueError`, `Io` → `PyIOError`, `Deserialize`/`SchemaVersion` → a deserialize/`CatBoostError`, `Train`/`Model` → base `CatBoostError`. The binding ADDS `CatBoostParameterError` (D-05/D-07 — has no facade-enum source; it is raised by `params.rs` validation, not converted from `CatBoostError`) and `NotFittedError` (sklearn parity). Use `create_exception!` for the Python exception types (RESEARCH lines 293-296).

**Error-test analog:** `crates/catboost-rs/src/error_test.rs` lines 1-45 shows the variant-conversion assertion style (`let public: CatBoostError = core.into(); match public { ... }`) — mirror it for the `From<…> for PyErr` direction in `errors_test.rs`.

---

### `crates/catboost-rs-py/src/params.rs` (registry + kwargs→Builder map — D-07)

**Analog (kwargs targets — role-match):** `crates/catboost-rs/src/builder.rs`. The setter surface (lines 113-245) IS the set of params that map to "IMPLEMENTED" in the D-07 registry; the `new()` defaults (lines 89-109) are the default values the Python signature mirrors:

```rust
// builder.rs:90-108 — the IMPLEMENTED params + their default values:
loss: Loss::Rmse,            // loss_function (default RMSE)
iterations: 1000,            // iterations / n_estimators / num_trees / num_boost_round
depth: 6,                    // depth / max_depth
learning_rate: 0.03,         // learning_rate / eta
l2_leaf_reg: 3.0,            // l2_leaf_reg / reg_lambda
random_strength: 0.0,        // random_strength
bootstrap_type: EBootstrapType::No,  // bootstrap_type
subsample: 1.0,              // subsample
bagging_temperature: 0.0,    // bagging_temperature
random_seed: 0,              // random_seed / random_state
border_count: <default 254>, // border_count / max_bin
score_function: <Cosine>,    // score_function
```

```rust
// builder.rs:125-138 — the two callback setters the Python custom_loss/custom_metric bridge targets:
pub fn custom_objective(mut self, objective: Arc<dyn CustomObjective>) -> Self { ... }
pub fn custom_metric(mut self, metric: Arc<dyn CustomMetric>) -> Self { ... }
```

**Key insight for the registry (D-07 case (a) vs (b)):** every upstream `core.py` param that HAS a matching Builder setter above is `IMPLEMENTED`; every upstream param WITHOUT one (e.g. `od_wait`, `nan_mode`, CTR knobs, GPU-only knobs) is `KNOWN-NOT-YET` (a parity gap, error case (a)); anything not in upstream's ~130-param vocabulary is `UNKNOWN` (case (b), suggest closest match). The full upstream seed is RESEARCH lines 394-418 (incl. sklearn-alias params, line 418). The `builder.rs` `boost_params()` method (lines 251-311) also reveals which params are currently *pinned to defaults and inert* (CTR/permutation/one-hot) — these are `KNOWN-NOT-YET` for the numeric facade path.

---

### `crates/catboost-rs-py/src/pool.rs` (native `Pool` `#[pyclass]`)

**Analog (build target — exact):** `crates/cb-data/src/ingest/owned.rs` lines 35-205. The Python `Pool.__init__(data, label, cat_features, ...)` maps directly onto `OwnedColumns::new(float_features, label)` + the `with_*` chain → `into_pool()`:

```rust
// owned.rs:40-102 — the builder the Python Pool wraps:
OwnedColumns::new(float_features, label)
    .with_cat_features(...).with_text_features(...).with_embedding_features(...)
    .with_weights(...).with_group_id(...).with_subgroup_id(...)
    .with_pairs(...).with_baseline(...)
// then:
let pool = owned.into_pool()?;   // IngestSource seam (owned.rs:147-204) — length validation here
```

The upstream `Pool.__init__` signature to mirror is RESEARCH lines 421-428 (`data, label, cat_features, text_features, embedding_features, weight, group_id, subgroup_id, pairs, baseline, feature_names, thread_count, ...`). The `cb_data::Pool` read accessors (`float_features`/`cat_features`/`label`/`weights`/`group_id`/`subgroup_id`/`pairs`/`baseline`) back any Python-side getters.

**Validation pattern to inherit:** `owned.rs` `into_pool()` (lines 147-204) does ALL column-length consistency checks via `check_len` (lines 122-132) returning typed `CbError::LengthMismatch` — never panics, never indexes. The Python adapter feeds owned columns into THIS seam so it inherits that validation for free (do not re-implement length checks).

---

### `crates/catboost-rs-py/src/ingest_py.rs` (NumPy/Pandas/Arrow/Polars adapters — D-10/D-11/D-12)

**Analog (the seam to converge on — role-match):** `crates/cb-data/src/ingest/mod.rs` lines 39-50 (the `IngestSource` trait) + `ingest/owned.rs` (the owned impl) + `ingest/arrow.rs` (the Arrow impl, `ArrowColumns` lines 121-148). The mod.rs doc comment (lines 5-8) EXPLICITLY anticipates this phase:

```rust
// ingest/mod.rs:5-8
//! At Phase 8 a borrowed / zero-copy view plugs into the same seam by adding
//! another `impl IngestSource`, without touching `Pool` (D-02).

pub trait IngestSource {
    /// # Errors
    /// Returns `LengthMismatch` ... `OutOfRange` ...
    fn into_pool(self) -> CbResult<Pool>;
}
```

```rust
// ingest/arrow.rs:121-148 — the existing borrowed-source → Pool adapter to model the NumPy/Arrow path on:
pub struct ArrowColumns { ... }
impl ArrowColumns { pub fn new(...) -> Self { ... } }
impl IngestSource for ArrowColumns {
    fn into_pool(self) -> CbResult<Pool> { ... }   // validate, then build Pool
}
```

**Strategy:** the NumPy/Pandas/Arrow/Polars adapters borrow the Python buffer (`numpy::PyReadonlyArray` / `pyo3_arrow::PyTable`), validate dtype=float32 + contiguity (D-12 → `CatBoostValueError`), COPY into an `OwnedColumns` (D-11: own BEFORE `py.detach()`), then call the existing `into_pool()`. Do NOT invent a new ingestion seam (CONTEXT §Reusable Assets). The zero-copy borrow + own-before-detach glue itself has NO in-repo analog → RESEARCH Patterns 2 & 4 (lines 273-322).

---

### `crates/catboost-rs-py/src/{classifier,regressor,ranker}.rs` (estimator `#[pyclass]` wrappers)

**Analog (the wrapped Rust surface — role-match):** `crates/catboost-rs/src/builder.rs` `fit()` (lines 326-357) + `crates/catboost-rs/src/model.rs` predict surface (lines 85-111). The `#[pyclass].fit()` validates params, ingests to a `Pool`, then drives:

```rust
// builder.rs:326-357 (abridged) — what each estimator's fit() calls under py.detach():
let model = CatBoostBuilder::new()
    .loss(...).iterations(...).depth(...).learning_rate(...)  // from kwargs map (params.rs)
    .fit(&pool)?;                                             // -> Result<Model, CatBoostError>
```

```rust
// model.rs:85-111 — the predict surface each class exposes:
pub fn predict_with(&self, pool: &Pool, prediction_type: PredictionType) -> Result<Vec<f64>, CatBoostError>
pub fn predict(&self, pool: &Pool) -> Result<Vec<f64>, CatBoostError>            // Regressor.predict
pub fn predict_proba(&self, pool: &Pool) -> Result<Vec<f64>, CatBoostError>      // Classifier.predict_proba
```

Classifier → `predict_proba`/`predict`; Regressor → `predict`; Ranker → `predict` over a group_id/pairs pool. `score`/SHAP/importance map to `model.rs` `feature_importance*`/`shap_values` (lines 119-192). The `#[pyclass]` store-verbatim + get/set/tags/clone/NotFitted glue has NO in-repo analog → RESEARCH Pattern 1 (lines 236-271).

---

### `crates/catboost-rs-py/src/*_test.rs` (Rust unit tests)

**Analog (exact):** `crates/catboost-rs/src/error_test.rs` + the `#[cfg(test)] mod error_test;` declaration in `lib.rs` line 46-47. The mandatory source/test-separation rule (CLAUDE.md) is realized as separate `<name>_test.rs` files declared `#[cfg(test)] mod <name>_test;` in the module root — NOT inline `#[cfg(test)] mod tests {}`. Test files may freely `panic!`/`unwrap` (covered by the crate-level `#![cfg_attr(test, allow(...))]`).

## Shared Patterns

### Workspace clippy gate (`unwrap`/`expect`/`panic`/`indexing_slicing` = deny)
**Source:** workspace `Cargo.toml` lines 9-14 + the per-crate opt-in `[lints] workspace = true` (`crates/catboost-rs/Cargo.toml:7-8`) + the in-code test exemption (`crates/catboost-rs/src/lib.rs:1`).
**Apply to:** the new crate's `Cargo.toml` (`[lints] workspace = true`) AND `src/lib.rs` (the `#![cfg_attr(test, allow(...))]` header). PyO3 glue is NOT exempt — every fallible path returns `PyResult`/`Result<_, CatBoostError>`, never `unwrap`/`panic` across the FFI boundary (RESEARCH §Security V12).

```rust
// crates/catboost-rs/src/lib.rs:1 — copy this header verbatim into the new lib.rs:
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing))]
```

### Typed-error → boundary mapping
**Source:** `crates/catboost-rs/src/error.rs` (the 6-variant `thiserror` enum) + `error_test.rs` (the conversion-assertion style).
**Apply to:** `errors.rs` (`From<CatBoostError> for PyErr`) and every estimator/pool method (return `PyResult`). One Rust variant → one Python exception (PYAPI-05).

### Backend feature forwarding (never pin cpu)
**Source:** `crates/cb-backend/Cargo.toml` lines 26-38 (the per-feature `cubecl/<facade>` forward) + workspace `Cargo.toml` comment on feature-unification (the landmine).
**Apply to:** the new crate's `[features]` — `cpu`/`rocm` forward to the facade passthrough; `default = ["cpu"]` but NEVER an unconditional `cpu` dependency (CONTEXT §Established Patterns: the backend-feature-unification landmine).

### `IngestSource::into_pool()` validation seam
**Source:** `crates/cb-data/src/ingest/{mod.rs,owned.rs,arrow.rs}`.
**Apply to:** `pool.rs` and `ingest_py.rs` — converge all Python inputs onto an `OwnedColumns`/new `impl IngestSource` and call `into_pool()` to inherit length/range validation; do not re-implement shape checks or invent a new seam.

### Builder-is-Rust-only / Python-is-mirror
**Source:** `crates/catboost-rs/src/builder.rs` (the Builder stays) + CONTEXT D-01.
**Apply to:** all estimator classes — Python kwargs map to Builder setters internally; the Builder is never exposed to Python.

## No Analog Found

Files with no close match in the codebase (planner uses RESEARCH.md patterns instead):

| File | Role | Data Flow | Reason | Use Instead |
|------|------|-----------|--------|-------------|
| `crates/catboost-rs-py/pyproject.toml` | maturin packaging config | n/a | First Python packaging surface in repo | RESEARCH §Standard Stack lines 125-141 |
| `src/lib.rs` `#[pymodule]` body | PyO3 module registration | n/a | No PyO3 code exists in repo yet | RESEARCH §Recommended Project Structure + Pattern 3 |
| estimator `#[pyclass]` sklearn glue (get/set/tags/clone/NotFitted) | sklearn contract in Rust | n/a | No `#[pyclass]` in repo | RESEARCH Pattern 1 (lines 236-271), Pitfall 2/5 |
| zero-copy borrow + own-before-detach glue | GIL/buffer safety | n/a | No PyO3 buffer handling in repo | RESEARCH Pattern 2 & 4 (lines 273-322), Pitfall 3 |
| `tests/test_*.py` + `conftest.py` | pytest suite | n/a | No Python test harness in repo (oracle harness is Rust `cb-oracle`) | RESEARCH §Validation Architecture (lines 514-553) |

## Metadata

**Analog search scope:** `crates/catboost-rs/{Cargo.toml,src/*}`, `crates/cb-data/src/ingest/*`, `crates/cb-data/src/pool.rs`, `crates/cb-backend/Cargo.toml`, workspace-root `Cargo.toml`.
**Files scanned:** 9 (full read: facade `lib.rs`/`error.rs`/`builder.rs`/`Cargo.toml`, `cb-backend/Cargo.toml`, `ingest/mod.rs`, `ingest/owned.rs`; targeted: `model.rs` predict surface, `pool.rs`/`arrow.rs` signatures).
**Pattern extraction date:** 2026-06-21

## PATTERN MAPPING COMPLETE

**Phase:** 8 - Python Bindings, Dual API & Packaging
**Files classified:** 19 (18 new + 1 possibly-modified workspace root)
**Analogs found:** 14 / 19

### Coverage
- Files with exact analog: 6 (Cargo.toml features/deps, errors.rs source, pool.rs build target, `*_test.rs`, lib.rs header, workspace-root members glob)
- Files with role-match analog: 8 (lib.rs composition, params.rs, estimators ×3, ingest_py.rs, estimator.rs base)
- Files with no analog: 5 (pyproject.toml, `#[pymodule]` body, sklearn `#[pyclass]` glue, zero-copy GIL glue, pytest suite)

### Key Patterns Identified
- Every workspace member: `edition="2021"`, `[lints] workspace=true`, plus the in-code `#![cfg_attr(test, allow(...))]` header — the clippy deny-gate (`unwrap`/`expect`/`panic`/`indexing_slicing`) applies to PyO3 glue too.
- The `catboost-rs` facade is the EXACT bind target: `CatBoostBuilder` setters (= D-07 IMPLEMENTED kwargs + default values) + `Model::predict*`/`shap_values`/`feature_importance*`; the binding never reaches into `cb-*` internals.
- The `cb_data::ingest::IngestSource` seam already anticipates Phase 8 (mod.rs doc): all Python inputs converge onto a new `impl IngestSource` / `OwnedColumns` and call `into_pool()` to inherit length/range validation — no new ingestion seam, no re-implemented shape checks.
- Backend features forward (`cpu`/`rocm` → facade passthrough → `cb-backend`'s `cubecl/<facade>`); NEVER pin `cpu` unconditionally (feature-unification landmine).
- The 6-variant `CatBoostError` `thiserror` enum is the source for the one-variant→one-PyErr PYAPI-05 mapping; `CatBoostParameterError`/`NotFittedError` are binding-only additions (no facade-enum source).
- Source/test separation: `<name>_test.rs` files declared `#[cfg(test)] mod <name>_test;` in the module root — never inline `mod tests {}`.

### File Created
`.planning/phases/08-python-bindings-dual-api-packaging/08-PATTERNS.md`

### Ready for Planning
Pattern mapping complete. The planner can reference each analog (file + line range) directly in PLAN.md action sections; the 5 no-analog PyO3/packaging surfaces fall back to RESEARCH.md §Architecture Patterns 1–4 and §Standard Stack.
