---
phase: 08-python-bindings-dual-api-packaging
plan: 04
subsystem: api
tags: [pyo3, classifier, ranker, oracle-parity, load-model]

# Dependency graph
requires:
  - phase: 08-python-bindings-dual-api-packaging
    provides: "08-01/08-02 CatBoostRegressor + error taxonomy; 08-03 ingest_to_owned / data_to_pool / native Pool"
  - phase: 04-model-serialization
    provides: "facade Model::load_cbm / load_json typed-error loaders (never panic)"
provides:
  - "CatBoostClassifier #[pyclass]: fit / predict (class labels, PredictionType::Class) / predict_proba ((n,2)); defaults loss to Logloss when user sets neither loss_function nor objective (D-05)"
  - "CatBoostRanker #[pyclass]: fit over a group_id-bearing Pool (rejects missing group_id with actionable CatBoostValueError, T-08-14) / predict (raw ranking scores)"
  - "load_model(path) staticmethod on CatBoostRegressor + CatBoostClassifier (dispatch .json->load_json else load_cbm; maps Deserialize/SchemaVersion->CatBoostValueError, T-08-12)"
  - "Shared estimator helpers hoisted to estimator.rs: data_to_pool, load_model_path, EstimatorBase::from_model"
  - "Python-surface oracle-parity gate <=1e-5 vs offline catboost 1.2.10 (model_serde regression + binclf), hermetic"
affects: [08-05, 08-06, 08-07]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Estimator trio shares one ingest chokepoint (estimator::data_to_pool) — prep for the 08-05 sklearn contract"
    - "load_model = a fitted estimator with params={} + model=Some(loaded) (EstimatorBase::from_model); no training"
    - "Classification default loss applied in fit() AFTER make_builder, only when neither loss_function nor objective present (honors explicit CrossEntropy etc.)"
    - "Classifier predict_proba reshapes the facade's flat [class-0,class-1] row-major output to (n,2) via chunks_exact(2)"
    - "Oracle parity: model_serde fixtures store RawFormulaVal; reg compared to predict directly, binclf raw logit compared via predict_proba[:,1] == sigmoid(raw) and predict == (raw>0)"

key-files:
  created:
    - crates/catboost-rs-py/src/classifier.rs
    - crates/catboost-rs-py/src/ranker.rs
    - crates/catboost-rs-py/tests/test_native_api.py
    - crates/catboost-rs-py/tests/test_oracle_parity.py
  modified:
    - crates/catboost-rs-py/src/estimator.rs
    - crates/catboost-rs-py/src/regressor.rs
    - crates/catboost-rs-py/src/lib.rs
    - crates/catboost-rs-py/tests/conftest.py

key-decisions:
  - "predict_proba shape convention = (n, 2) with [P(class 0), P(class 1)] per row (upstream binary form), reshaped from the facade's flat two-column Probability output."
  - "Classifier predict returns class labels via PredictionType::Class (shape (n,), values in {0.0,1.0})."
  - "Classification default loss = Logloss, applied in fit() only when the user set neither loss_function nor its objective alias, so an explicit classification loss is honored."
  - "Oracle parity fixtures = crates/cb-oracle/fixtures/model_serde/{regression,binclf} (catboost 1.2.10, score_function=L2, RawFormulaVal) with the shared numeric_tiny input matrix. Both store RawFormulaVal, so binclf parity is asserted via predict_proba[:,1]==sigmoid(raw) (and predict==(raw>0)) rather than a stored proba vector — observed bit-exact (max abs diff 0.0)."
  - "load_model is a #[staticmethod] (mirrors upstream load_model call form) returning a fitted estimator; the single deterministic oracle path, no re-fit fallback."

requirements-completed: [PYAPI-03]

# Metrics
duration: ~5min
completed: 2026-06-23
---

# Phase 8 Plan 04: CatBoost-Native Classifier + Ranker + Oracle Parity Summary

**A user can now fit a `CatBoostClassifier` (defaulting to Logloss) and get `(n,)` class labels + `(n,2)` probabilities, fit a `CatBoostRanker` on a `group_id` `Pool` and get `(n,)` ranking scores (with a group-less dataset rejected by an actionable `CatBoostValueError`), and the whole Python surface is parity-locked: `CatBoostRegressor`/`CatBoostClassifier.load_model(path)` load the offline catboost 1.2.10 reference `.cbm`/`.json` and reproduce its predictions to within 1e-5 (observed bit-exact) — hermetically, with no live `catboost` import and no re-fit fallback.**

## Performance

- **Duration:** ~5 min
- **Tasks:** 2 (Task 1 `auto`+`tdd`, Task 2 `auto`)
- **Files created:** 4; **modified:** 4
- **Tests:** 22 Rust unit tests + 37 pytest (27 prior + 6 native-api + 4 oracle-parity), all green

## Accomplishments

### Task 1 — CatBoostClassifier + CatBoostRanker #[pyclass] — commit `7f4e93a`

- **`classifier.rs`** — `#[pyclass] CatBoostClassifier` mirroring the regressor's store-verbatim / validate / ingest / detach / fit structure, with:
  - `fit` defaulting the builder loss to `Loss::Logloss` (a classification loss — the loss SELECTS the task, D-05) ONLY when the user supplied neither `loss_function` nor its `objective` alias.
  - `predict` -> `PredictionType::Class` (class labels, `(n,)`).
  - `predict_proba` -> `PredictionType::Probability`, reshaped from the facade's flat row-major `[class-0, class-1]` output to `(n, 2)` (upstream binary convention).
- **`ranker.rs`** — `#[pyclass] CatBoostRanker`: `fit` validates `group_id` presence on the materialized facade `Pool` (`pool.group_id().is_empty()`) and rejects a group-less dataset with an actionable `CatBoostValueError` (threat T-08-14); `predict` -> `Model::predict` (raw ranking scores). sklearn presentation deferred to 08-05 per RESEARCH 483-487.
- **`estimator.rs`** — hoisted the duplicated `data_to_pool` chokepoint out of `regressor.rs` so all three estimators ingest identically; added `load_model_path` (extension dispatch + `PyCbError` mapping) and `EstimatorBase::from_model` (fitted base with empty params).
- **`regressor.rs`** — refactored onto the shared `data_to_pool`, switched its error mapping to the `PyCbError` newtype, and gained a `load_model` `#[staticmethod]`.
- **`lib.rs`** — registered `CatBoostClassifier` + `CatBoostRanker`.
- **`conftest.py`** — added `toy_classification` (binary), `toy_ranking` (10 groups x 6), and the `oracle_regression` / `oracle_binclf` fixture-path fixtures (hermetic, repo-root-relative).
- **`test_native_api.py`** — 6 tests: classifier label/proba shape + default-loss; ranker grouped fit/predict + two group-less rejection paths (bare array, group-less Pool).

### Task 2 — Python-surface oracle parity <=1e-5 — commit `e35acf2`

- The `load_model` constructors landed in Task 1 (source committed there); Task 2 adds **`test_oracle_parity.py`** (4 tests) exercising them as the single deterministic oracle path:
  - **Regression:** `CatBoostRegressor.load_model(model.cbm | model.json).predict(X)` vs the stored `RawFormulaVal` `predictions.npy` — `atol=1e-5, rtol=0`.
  - **Classification:** `CatBoostClassifier.load_model(model.cbm).predict_proba(X)[:,1]` vs `sigmoid(raw)`, `[:,0]` vs `1-sigmoid(raw)`, and `predict(X)` vs `(raw > 0)` — `atol=1e-5, rtol=0`.
- **Fixtures:** `crates/cb-oracle/fixtures/model_serde/{regression,binclf}` (catboost 1.2.10, `score_function=L2`, `RawFormulaVal`) over the shared `inputs/numeric_tiny/X.npy` (50x4) — reused, not regenerated.
- **Observed parity:** bit-exact (max abs diff `0.0`) for both regression `predict` and classification `predict_proba[:,1]` vs `sigmoid(raw)`, far inside the 1e-5 bar.
- **Hermetic:** the test imports only `numpy` + `catboost_rs` and reads frozen fixture files; it does NOT `import catboost`. No re-fit fallback.

## predict_proba Shape Convention (recorded per plan output spec)

`CatBoostClassifier.predict_proba(X)` returns a NumPy `float64` array shaped **`(n, 2)`** with `[P(class 0), P(class 1)]` per row (the upstream binary convention), reshaped from the facade `Model::predict_with(PredictionType::Probability)` flat row-major two-column output. `predict(X)` returns class LABELS shaped `(n,)` via `PredictionType::Class`.

## Fixtures Used for Parity (recorded per plan output spec)

- Input matrix: `crates/cb-oracle/fixtures/inputs/numeric_tiny/X.npy` (50 rows x 4 float features, loaded as float32 C-contiguous).
- Regression: `crates/cb-oracle/fixtures/model_serde/regression/{model.cbm, model.json, predictions.npy}` (RMSE, `boost_from_average=true`, `score_function=L2`, `RawFormulaVal`).
- Classification: `crates/cb-oracle/fixtures/model_serde/binclf/{model.cbm, predictions.npy}` (Logloss, `boost_from_average=false`, `score_function=L2`, `RawFormulaVal` logits).

## Threat Mitigations Applied

- **T-08-12 (malformed .cbm/.json via load_model):** `load_model_path` routes through the facade `Model::load_cbm`/`load_json` (Phase-4 typed, never-panic loaders); a `Deserialize`/`SchemaVersion` error maps via `PyCbError`/`to_pyerr` to `CatBoostValueError`.
- **T-08-13 (classifier proba shape/threshold drift -> silent wrong labels):** the oracle-parity test pins `predict_proba[:,1] == sigmoid(raw)` and `predict == (raw>0)` numerically against catboost 1.2.10; `test_native_api` pins the `(n,2)` shape + rows-sum-to-1 simplex.
- **T-08-14 (ranker fit without group_id):** `fit` checks `pool.group_id().is_empty()` and returns an actionable typed `CatBoostValueError` before any training; verified by two native-api tests (bare array + group-less Pool).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] unused `to_pyerr` import after switching regressor to `PyCbError`**
- **Found during:** Task 1 (`cargo check`).
- **Issue:** refactoring the regressor's error mapping to the `PyCbError` newtype (`.map_err(PyCbError)`) left `to_pyerr` imported but unused (a warning).
- **Fix:** dropped `to_pyerr` from the `regressor.rs` import list.
- **Files:** `crates/catboost-rs-py/src/regressor.rs`.
- **Committed in:** `7f4e93a`.

### Plan-deviation note (classification oracle comparison)

The plan's Task 2 wording (`py_pred = est.predict(X)` compared to `ref_vec`) literally applies to the regression fixture. For the classification fixture the stored reference vector is `RawFormulaVal` (raw logits), not class labels — so comparing the classifier's label output (`predict`) to it directly would be incorrect. The done-criterion (Python predictions reproduce the catboost 1.2.10 reference vector to <=1e-5 via the deterministic load_model path) is satisfied by comparing `predict_proba[:,1]` to `sigmoid(raw_ref)` (the exact upstream logit->probability map) AND `predict` to `(raw_ref > 0)`. Both are bit-exact. This is the correct numeric pinning of the classifier surface to the stored RawFormulaVal vector, not a fixture change — fixtures were reused as-is.

**Total deviations:** 1 auto-fixed (Rule 1 - warning) + 1 documented comparison-method clarification. No scope change, no fixture regeneration.

## Known Stubs

None. The classifier/regressor/ranker fit/predict surfaces are wired to the real facade; `load_model` loads real reference models; the oracle test reads real frozen fixtures. (Pre-existing 08-03 documented Pool signature-parity stubs for `text_features`/`embedding_features`/`pairs`/`feature_names` are unchanged and out of this plan's scope.)

## Self-Check: PASSED

- `crates/catboost-rs-py/src/classifier.rs` — FOUND
- `crates/catboost-rs-py/src/ranker.rs` — FOUND
- `crates/catboost-rs-py/tests/test_native_api.py` — FOUND
- `crates/catboost-rs-py/tests/test_oracle_parity.py` — FOUND
- commit `7f4e93a` (Task 1) — FOUND
- commit `e35acf2` (Task 2) — FOUND
- `cargo test -p catboost-rs-py --features cpu` 22/22 + `pytest tests/` 37/37 — GREEN
- Oracle parity max abs diff `0.0` (<= 1e-5) for regression predict + classification predict_proba[:,1]

---
*Phase: 08-python-bindings-dual-api-packaging*
*Completed: 2026-06-23*
