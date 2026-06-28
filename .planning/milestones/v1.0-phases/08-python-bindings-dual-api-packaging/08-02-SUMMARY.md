---
phase: 08-python-bindings-dual-api-packaging
plan: 02
subsystem: api
tags: [pyo3, errors, param-registry, taxonomy, validation]

# Dependency graph
requires:
  - phase: 08-python-bindings-dual-api-packaging
    provides: "08-01 walking skeleton (CatBoostRegressor fit/predict over NumPy via the facade)"
  - phase: 04-builder-facade
    provides: "CatBoostBuilder setter surface + typed CatBoostError six-variant enum"
provides:
  - "PYAPI-05 typed-exception taxonomy: CatBoostError base + CatBoostParameterError / CatBoostValueError / NotFittedError"
  - "to_pyerr: one facade CatBoostError variant -> one specific Python exception"
  - "D-07 param-vocabulary registry (119 upstream params tagged IMPLEMENTED/KNOWN_NOT_YET) + sklearn alias map"
  - "validate_params: fit()-time rejection of parity-gap + unknown (typo) kwargs (D-05/D-06)"
  - "make_builder: typed kwargs->CatBoostBuilder map for all IMPLEMENTED params"
  - "_param_status pyfunction (registry introspection)"
affects: [08-03, 08-04, 08-05, 08-06, 08-07]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Orphan-legal error mapping: local PyCbError newtype + to_pyerr free fn (NOT impl From<foreign> for PyErr)"
    - "Multiple-inheritance exception via dynamic type(name,(bases),{}) (create_exception! is single-parent)"
    - "PyOnceLock cache for the dynamically-built NotFittedError type"
    - "Registry honesty: KNOWN_NOT_YET params rejected (parity gap) not silently ignored (T-08-05)"

key-files:
  created:
    - crates/catboost-rs-py/src/errors.rs
    - crates/catboost-rs-py/src/errors_test.rs
    - crates/catboost-rs-py/src/params.rs
    - crates/catboost-rs-py/src/params_test.rs
    - crates/catboost-rs-py/tests/test_errors.py
    - crates/catboost-rs-py/tests/test_params.py
  modified:
    - crates/catboost-rs-py/src/lib.rs
    - crates/catboost-rs-py/src/regressor.rs
    - crates/catboost-rs-py/src/estimator.rs
    - crates/catboost-rs-py/Cargo.toml

key-decisions:
  - "Orphan rule (E0117): used a local PyCbError newtype + to_pyerr(&err) free fn instead of the plan's literal impl From<catboost_rs::CatBoostError> for PyErr (both types foreign)"
  - "NotFittedError needs TWO bases (CatBoostError + ValueError) which create_exception! cannot express; built dynamically via Python type(name, bases, dict) and cached in a PyOnceLock"
  - "Deserialize/SchemaVersion map to CatBoostValueError (malformed/unsupported model = value error), per plan action"
  - "IMPLEMENTED = the 14 params with a CatBoostBuilder setter; loss_function restricted to the built-in default-arg losses (RMSE/Logloss/CrossEntropy/MAE/LogCosh), parametric losses deferred"

requirements-completed: [PYAPI-05, PYAPI-03]

# Metrics
duration: ~40min
completed: 2026-06-23
---

# Phase 8 Plan 02: Error Taxonomy + Param Registry Summary

**The binding is now honest about what it supports: every facade `CatBoostError` variant maps to a specific catchable Python exception (PYAPI-05), and the full 119-param upstream vocabulary is validated at `fit()` (D-06) so a known-but-unimplemented param (`nan_mode`) is rejected as a parity gap, a typo (`iteratons`) suggests `iterations`, and sklearn aliases (`n_estimators`/`max_depth`/`reg_lambda`) resolve.**

## Performance

- **Duration:** ~40 min
- **Tasks:** 2 (both `auto` + `tdd`)
- **Files created:** 6
- **Files modified:** 4
- **Tests:** 21 Rust unit tests + 16 pytest, all green

## Accomplishments

### Task 1 — Typed exception taxonomy (PYAPI-05) — commit `688c8c4`

- `errors.rs`: `create_exception!` taxonomy — `CatBoostError` (base, subclasses `PyException`), `CatBoostParameterError` (parent `CatBoostError`), `CatBoostValueError` (parent `CatBoostError`). `NotFittedError` is built **dynamically** with the two bases `(CatBoostError, ValueError)` (so `except CatBoostError` AND sklearn's `check_is_fitted` `ValueError` path both catch it) — `create_exception!` only supports a single parent. Cached in a `PyOnceLock`.
- Orphan-legal mapping (the prior attempt's E0117 landmine): `to_pyerr(&CatBoostError) -> PyErr` free function performs the six-variant match; a local `PyCbError` newtype provides `impl From<PyCbError> for PyErr` for `.map_err(PyCbError)?` call sites. No illegal `impl From<foreign> for foreign`.
- Variant mapping: `FeatureMismatch` / `Deserialize` / `SchemaVersion` -> `CatBoostValueError`; `Io` -> `PyIOError`; `Train` / `Model` -> base `CatBoostError`.
- `regressor.rs`: not-fitted `predict` raises `NotFittedError`; ingest failure raises `CatBoostValueError`; facade errors routed through `to_pyerr`.

### Task 2 — Param-vocabulary registry + fit()-time validation (D-05/D-06/D-07) — commit `12b2cb6`

- `params.rs`: the **119-name** upstream `CatBoostClassifier.__init__` vocabulary transcribed verbatim from `core.py:5333`, each tagged `IMPLEMENTED` (has a `CatBoostBuilder` setter) or `KNOWN_NOT_YET` (parity gap). Alias map resolves sklearn/xgboost names to their canonical target before tag lookup.
- `validate_params` runs at the TOP of `fit()` (D-06, before ingest): IMPLEMENTED -> ok; KNOWN_NOT_YET -> `CatBoostParameterError` flagging a parity gap; UNKNOWN -> `CatBoostParameterError` with a Levenshtein closest-match suggestion. No silent acceptance (threat T-08-05).
- `make_builder` applies every IMPLEMENTED param (alias-resolved) with correct typed extraction, plus enum-string parsers for `loss_function` / `score_function` / `bootstrap_type` / `leaf_estimation_method`.
- `_param_status` pyfunction exposes the registry to the Python coverage test.

## Parity-Gap Inventory (for the verifier)

**Registry coverage: 119 upstream params, 0 UNKNOWN.** Split:

- **IMPLEMENTED — 14 canonical setters:** `iterations`, `learning_rate`, `depth`, `l2_leaf_reg`, `loss_function`, `border_count`, `random_seed`, `random_strength`, `bagging_temperature`, `bootstrap_type`, `subsample`, `score_function`, `boost_from_average`, `leaf_estimation_method`.
- **IMPLEMENTED via alias — 9:** `max_depth`->depth, `n_estimators`/`num_trees`/`num_boost_round`->iterations, `random_state`->random_seed, `reg_lambda`->l2_leaf_reg, `objective`->loss_function, `eta`->learning_rate, `max_bin`->border_count. (`_param_status` reports these IMPLEMENTED -> 23 names total report IMPLEMENTED.)
- **KNOWN_NOT_YET — 96:** every other upstream param (e.g. `nan_mode`, `od_wait`, `od_type`, `rsm`, all CTR knobs `simple_ctr`/`combinations_ctr`/`max_ctr_complexity`/..., `class_weights`, `auto_class_weights`, `one_hot_max_size`, `grow_policy`, `min_data_in_leaf`, `monotone_constraints`, `text_features`/`embedding_features`/`tokenizers`/`dictionaries`, `task_type`/`devices`, `eval_metric`, `custom_loss`/`custom_metric`, `colsample_bylevel`->`rsm`, ...). These are rejected at `fit()` as parity gaps — future plans (08-03 ingestion, 08-04 classifier/ranker, later loss/CTR plans) will move subsets to IMPLEMENTED.

Note: `loss_function` is IMPLEMENTED but only the built-in default-argument losses parse today (RMSE, Logloss, CrossEntropy, MAE, LogCosh); a parametric loss string (e.g. `Quantile:alpha=0.9`) currently raises `CatBoostParameterError`. Parametric-loss parsing is deferred (08-04+).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Orphan rule (E0117): cannot `impl From<catboost_rs::CatBoostError> for PyErr`**
- **Found during:** Task 1 (anticipated by the retry note).
- **Issue:** Both `catboost_rs::CatBoostError` and `pyo3::PyErr` are foreign to the binding crate; the plan's literal Pattern-3 `impl From<catboost_rs::CatBoostError> for PyErr` is an orphan-rule violation.
- **Fix:** Local `PyCbError` newtype with `impl From<PyCbError> for PyErr` (orphan-legal) + a `to_pyerr(&err)` free function as the single conversion chokepoint. Call sites use `.map_err(|e| to_pyerr(&e))` / `.map_err(PyCbError)?`. The `key_links` pattern (`From<catboost_rs::CatBoostError>|impl From`) is satisfied by the newtype `impl From`.
- **Files:** `errors.rs`, `regressor.rs`.

**2. [Rule 3 - Blocking] `NotFittedError` needs multiple inheritance that `create_exception!` cannot express**
- **Found during:** Task 1.
- **Issue:** The plan requires `NotFittedError` to subclass BOTH `CatBoostError` (taxonomy) AND `ValueError` (sklearn parity). PyO3's `create_exception!` accepts only a single parent.
- **Fix:** Built `NotFittedError` dynamically with `type("NotFittedError", (CatBoostError, ValueError), {})` at module init, cached in a `PyOnceLock`; a `not_fitted_err(py, msg)` helper raises it. Both subclass relationships are asserted in `test_errors.py`.
- **Files:** `errors.rs`, `lib.rs`.

**3. [Rule 3 - Blocking] PyO3 0.29 `FromPyObject` has two lifetimes + an associated `Error`**
- **Found during:** Task 2 (generic `get::<T>` helper).
- **Issue:** `T: FromPyObject<'py>` (one lifetime) no longer compiles; 0.29 is `FromPyObject<'a, 'py>` with an associated `Error` requiring `PyErr: From<T::Error>`.
- **Fix:** Bound `T: FromPyObject<'py, 'py>, PyErr: From<<T as FromPyObject<'py,'py>>::Error>`.
- **Files:** `params.rs`.

**4. [Rule 3 - Blocking] dev-dependencies for the Rust error test**
- **Found during:** Task 1 (`errors_test.rs` needs to construct `Train(CbError)` / `Model(ModelError)` payloads).
- **Issue:** `cb-core` / `cb-model` are NOT re-exported by the facade, so the test could not build those two variants.
- **Fix:** Added `cb-core` / `cb-model` as `[dev-dependencies]` (test-only).
- **Files:** `Cargo.toml`.

**Total deviations:** 4 auto-fixed (all Rule 3 - blocking), all necessary to compile under PyO3 0.29 / the orphan rule. No scope change.

## Threat Mitigations Applied

- **T-08-05 (silently-wrong model):** `validate_params` rejects KNOWN_NOT_YET / UNKNOWN params at `fit()` — verified by `test_known_not_yet_param_rejected_as_parity_gap` + `validate_rejects_known_not_yet_as_parity_gap`.
- **T-08-07 (DoS via registry/Levenshtein):** `levenshtein` uses checked iteration (no indexing on user input length); `closest_match` iterates a static slice. `[lints] workspace=true` deny gate holds.

## Known Stubs

- `loss_function` parsing covers only the 5 built-in default-arg losses; parametric loss strings raise a typed error (deferred 08-04+). This is documented above, not a silent stub.
- 96 KNOWN_NOT_YET params are deliberate, honestly-rejected parity gaps (the whole point of D-07), not stubs.

## Self-Check: PASSED

- `crates/catboost-rs-py/src/errors.rs` — FOUND
- `crates/catboost-rs-py/src/params.rs` — FOUND
- `crates/catboost-rs-py/src/errors_test.rs` — FOUND
- `crates/catboost-rs-py/src/params_test.rs` — FOUND
- `crates/catboost-rs-py/tests/test_errors.py` — FOUND
- `crates/catboost-rs-py/tests/test_params.py` — FOUND
- commit `688c8c4` (Task 1) — FOUND
- commit `12b2cb6` (Task 2) — FOUND
- `cargo test -p catboost-rs-py` 21/21 + `pytest tests/` 16/16 — GREEN

---
*Phase: 08-python-bindings-dual-api-packaging*
*Completed: 2026-06-23*
