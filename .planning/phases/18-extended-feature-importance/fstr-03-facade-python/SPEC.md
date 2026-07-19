---
title: "FSTR-03b — Surface partial dependence through the facade + Python"
status: draft
format: markdown
spec_version: 1
updated_at: 2026-07-16T00:00:00Z
phase: 18
requirement_ids:
  - FSTR-03
source_requirements:
  - "Follow-up to fstr-03-partial-dependence (cb-model core, shipped): expose it upward."
---

# FSTR-03b — Surface partial dependence upward

> **Draft.** Additive exposure of the already-shipped, oracle-verified
> `cb_model::partial_dependence` (see `../fstr-03-partial-dependence/`) through the
> published `catboost-rs` facade and the `catboost-rs-py` Python estimators. No new
> algorithm; no change to the cb-model core.

## 1. Context

`cb_model::partial_dependence(model, columns, features) -> Result<PartialDependence,
PdpError>` is implemented and oracle-locked ≤1e-5 vs `catboost==1.2.10`
`[VERIFIED: LOCAL crates/cb-model/src/partial_dependence.rs; tests green]`. It is NOT
reachable from the published `catboost-rs` facade `Model` (which currently exposes
predict / shap / feature_importance) nor from the Python estimators. This slice adds
those two thin adapters.

## 2. Scope / non-goals

- **In:** a facade `Model::partial_dependence(&self, pool, features)`; a Python
  `partial_dependence(self, X, features)` on the three estimators; error mapping;
  re-exports.
- **Out:** any change to the PD algorithm, grid, or oracle (done upstream); a
  plotting/figure API; categorical/flat-index PD (still deferred).

## 3. Dependencies (verified)

| Dependency | Interface | Evidence |
|-----------|-----------|----------|
| Core PD | `cb_model::partial_dependence`, `PartialDependence`, `PdpError` | `[VERIFIED: CODEGRAPH crates/cb-model/src/lib.rs:44]` |
| Pool→columns | facade `Model::feature_columns(pool)` (checks n_float, → `FeatureMismatch`) | `[VERIFIED: CODEGRAPH crates/catboost-rs/src/model.rs:59]` |
| Pool build (test) | `catboost_rs::OwnedColumns::new(float_features, label).into_pool()` | `[VERIFIED: CODEGRAPH crates/cb-data/src/ingest/owned.rs:40,172]` |
| Facade error | `CatBoostError` (exhaustive `to_pyerr` match must gain an arm) | `[VERIFIED: LOCAL crates/catboost-rs-py/src/errors.rs:105-114]` |
| Py Pool | `data_to_pool(py, x, y)` | `[VERIFIED: CODEGRAPH crates/catboost-rs-py/src/estimator.rs:235]` |

## 4. Typed contracts

```rust
// facade (catboost-rs)
impl Model {
    pub fn partial_dependence(&self, pool: &Pool, features: &[usize])
        -> Result<cb_model::PartialDependence, CatBoostError>;
}
// new CatBoostError variant:
//   PartialDependence(#[from] cb_model::PdpError)  -> Python CatBoostValueError
```

Python (per estimator, sklearn-adjacent): `partial_dependence(self, X, features)
-> dict` with keys `features: list[int]`, `grids: list[np.ndarray[f64]]`,
`values: np.ndarray[f64]` (row-major). `NotFittedError` if unfitted; a bad request
→ `CatBoostValueError`.

## 5. Failure-isolated specs

- **FAC-01 (facade happy path):** `Model::partial_dependence(pool, &[f])` /
  `&[f1,f2]` returns the SAME `PartialDependence` as `cb_model::partial_dependence`
  on the pool's projected columns; values match the committed
  `partial_dependence/pdp_*_values.npy` ≤1e-5. *One failure reason: facade
  delegation/column-projection.*
- **FAC-02 (facade error mapping):** a wrong-width `pool` → `FeatureMismatch`; an
  invalid `features` (arity/out-of-range/duplicate) → `PartialDependence(PdpError)`.
- **PY-01 (python adapter, compile-verified only):** `partial_dependence(X,
  features)` builds a Pool via `data_to_pool`, calls the facade, returns the dict;
  errors map through `to_pyerr`. **Cannot be run in this env** (catboost-rs-py links
  against an absent python3.14 — `[VERIFIED: LOCAL memory catboost-rs-preexisting-test-failures]`);
  verified by `cargo check -p catboost-rs-py` only.

## 6. Acceptance

| Scenario | Spec | Kind | Bar |
|----------|------|------|-----|
| facade single/pair == fixture | FAC-01 | integration (facade) | ≤1e-5 |
| wrong-width pool → FeatureMismatch; bad features → PartialDependence | FAC-02 | integration | typed Err |
| python method compiles + maps errors | PY-01 | `cargo check` only | compiles |

## 7. Impact

`cross-module` (catboost-rs + catboost-rs-py; cb-model unchanged). New: facade
method + `CatBoostError::PartialDependence` + re-exports; `to_pyerr` arm; a shared
python helper + one `#[pymethod]` per estimator. No serialization/format change.

## 8. Risks

1. **[VERIFIED] Python untestable here** (py3.14 link). Mitigation: `cargo check`
   compile-verification + mirror the exact `predict` adapter pattern; mark PY-01
   unverified-at-runtime.
2. **[INFERRED] Exhaustive `to_pyerr`** — the new `CatBoostError` variant MUST get an
   arm or the py crate fails to compile (caught by `cargo check`).
