# TDD Plan — FSTR-03b (surface partial dependence: facade + Python)

> ## Execution status (2026-07-16) — ✅ COMPLETE
> - **F1 facade** — `CatBoostError::PartialDependence(#[from] PdpError)`;
>   `Model::partial_dependence(&self, pool, features)`; `lib.rs` re-exports
>   `PartialDependence`/`PdpError`. Test `partial_dependence_facade_test.rs` (FAC-01
>   oracle ≤1e-5 vs fixtures + FAC-02 error mapping) — **2 passed**; facade clippy
>   clean; full `cargo test -p catboost-rs` = 0 fail.
> - **F2/F3 Python** — `to_pyerr` arm; shared `partial_dependence_py` helper;
>   `partial_dependence(x, features)` `#[pymethod]` on regressor/classifier/ranker
>   returning `{features, grids, values}`. **`cargo check -p catboost-rs-py` passes**;
>   my edited files are clippy-clean. **Runtime NOT tested** — catboost-rs-py links
>   against an absent python3.14 (pre-existing env blocker); the 9 `cargo clippy`
>   errors are all in untouched `ingest_py.rs`/`params.rs` (pre-existing).
> - **Files:** `crates/catboost-rs/src/{error,model,lib}.rs`,
>   `crates/catboost-rs/tests/partial_dependence_facade_test.rs`;
>   `crates/catboost-rs-py/src/{errors,estimator,regressor,classifier,ranker}.rs`.


**Spec:** `./SPEC.md` (FAC-01/02, PY-01). **Crates:** `catboost-rs` (facade),
`catboost-rs-py`. **cb-model core is unchanged.**

Validation:
```
cargo test  -p catboost-rs --test partial_dependence_facade_test   # FAC-01/02
cargo clippy -p catboost-rs --lib --no-deps                        # facade lint gate
cargo check -p catboost-rs-py                                      # PY-01 (link is py3.14-blocked)
```

## F1 — facade error variant + method (FAC-01/02)
- **Red:** new `crates/catboost-rs/tests/partial_dependence_facade_test.rs`:
  - `facade_single_and_pair_match_fixture`: `Model::load_cbm(partial_dependence/model.cbm)`;
    build a `Pool` from `numeric_tiny/X.npy` via `OwnedColumns::new(float_cols_f64,
    label).into_pool()`; assert `model.partial_dependence(&pool, &[3]).values` ==
    `pdp_single_values.npy` and `&[0,3]` == `pdp_pair_values.npy` (≤1e-5, via
    `cb_oracle::assert_abs_close`).
  - `facade_maps_errors`: a pool with the wrong float-feature width → `Err(FeatureMismatch)`;
    `features=[0,0]` → `Err(PartialDependence(PdpError::DuplicateFeature{..}))`;
    `features=[99]` → `Err(PartialDependence(PdpError::FeatureIndexOutOfRange{..}))`.
  - Expected fail: `partial_dependence` method + `PartialDependence` variant absent.
- **Green:**
  - `error.rs`: add `#[error("partial dependence error: {0}")] PartialDependence(#[from]
    cb_model::PdpError)`.
  - `model.rs`: import `cb_model::{partial_dependence, PartialDependence}`; add
    `pub fn partial_dependence(&self, pool: &Pool, features: &[usize])
    -> Result<PartialDependence, CatBoostError>` = `feature_columns(pool)?` then
    `Ok(cb_model::partial_dependence(&self.inner, &cols, features)?)`.
  - `lib.rs`: `pub use cb_model::{PartialDependence, PdpError};`.
- **Refactor:** none (mirrors `feature_importance_with_data`).

## F2 — Python `to_pyerr` arm (compile gate)
- `crates/catboost-rs-py/src/errors.rs`: add
  `FacadeError::PartialDependence(e) => CatBoostValueError::new_err(e.to_string()),`
  to the exhaustive `to_pyerr` match (a bad-input value error, like `FeatureMismatch`).
  Without it `cargo check -p catboost-rs-py` fails (non-exhaustive match).

## F3 — Python adapter method (PY-01, compile-verified)
- `estimator.rs`: shared `pub(crate) fn partial_dependence_py(model:&Model, py, x:&Bound<PyAny>,
  features: Vec<usize>) -> PyResult<PyObject>`: `data_to_pool(py,x,None)?`;
  `py.detach(|| model.partial_dependence(&pool, &features)).map_err(PyCbError)?`;
  build a `dict{features, grids:[PyArray1<f64>…], values:PyArray1<f64>}`.
- `regressor.rs`, `classifier.rs`, `ranker.rs`: add
  `fn partial_dependence(&self, py, x, features) -> PyResult<PyObject>` mirroring
  `predict` (not-fitted guard → `not_fitted_err`; delegate to the shared helper).
- **Verify:** `cargo check -p catboost-rs-py` compiles. (Runtime tests blocked: py3.14 link.)

## Traceability
| Task | Spec | Kind |
|------|------|------|
| F1 | FAC-01/02 | facade integration + facade unit |
| F2 | PY-01 (compile) | exhaustiveness |
| F3 | PY-01 | `cargo check` |
