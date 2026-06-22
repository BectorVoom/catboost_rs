---
phase: 08-python-bindings-dual-api-packaging
plan: 03
subsystem: api
tags: [pyo3, ingestion, numpy, pandas, arrow, polars, gil-safety, pool]

# Dependency graph
requires:
  - phase: 08-python-bindings-dual-api-packaging
    provides: "08-01 NumPy-only ingest (numpy_to_owned) + 08-02 CatBoostValueError taxonomy"
  - phase: 02-data-ingestion
    provides: "cb_data::ingest IngestSource / OwnedColumns::into_pool seam + arrow adapter"
provides:
  - "ingest_to_owned: single multi-source entry (NumPy / Pandas / Arrow / Polars) -> OwnedColumns"
  - "Strict D-12 rejection of float64 / non-contiguous / ambiguous-object / nullable inputs as CatBoostValueError"
  - "Own-before-detach (D-11/PYAPI-06) as a code property: every branch copies into OwnedColumns before returning"
  - "Native Pool #[pyclass] (PYAPI-03) mirroring upstream Pool.__init__, with num_row/num_col + inherited length validation"
  - "Estimator fit/predict accept EITHER a framework object OR a native Pool (data_to_pool)"
  - "OwnedColumns::feature_shape() validation-free shape accessor (cb-data)"
affects: [08-04, 08-05, 08-06, 08-07]

# Tech tracking
tech-stack:
  added: [arrow 59.0.0 (umbrella crate, workspace-pinned)]
  patterns:
    - "Multi-source dispatch: Arrow-capsule (pyarrow Table OR Polars DataFrame) -> Pandas DataFrame -> NumPy ndarray, in that order"
    - "One shared pyo3-arrow AnyRecordBatch path serves both pyarrow and Polars (__arrow_c_stream__)"
    - "Own-before-detach at every ingest branch (D-11): no PyReadonlyArray / Arrow capsule borrow crosses py.detach()"
    - "Pool stores OwnedColumns; into_pool() length validation deferred to to_pool() (cheap __new__)"
    - "data_to_pool: x.cast::<Pool>() (PyO3 0.29 rename of downcast) else ingest_to_owned"

key-files:
  created:
    - crates/catboost-rs-py/src/pool.rs
    - crates/catboost-rs-py/tests/test_ingestion.py
  modified:
    - crates/catboost-rs-py/src/ingest_py.rs
    - crates/catboost-rs-py/src/ingest_py_test.rs
    - crates/catboost-rs-py/src/regressor.rs
    - crates/catboost-rs-py/src/lib.rs
    - crates/catboost-rs-py/Cargo.toml
    - crates/catboost-rs-py/tests/test_smoke.py
    - crates/cb-data/src/ingest/owned.rs

key-decisions:
  - "Arrow path validates Float32 (not the cb-data ArrowColumns Float64 adapter): the Python boundary requires float32 (D-12), so the binding reads Float32Array directly out of the pyo3-arrow RecordBatch and copies f32->f64 into OwnedColumns; it does NOT route through cb_data::ingest::arrow::ArrowColumns (which is Float64-only)."
  - "Length-mismatch (label shorter than features) is caught fail-fast at the shared ingest seam (label_to_owned) carrying the same typed CatBoostValueError; metadata-column mismatches (weight/group_id/baseline) still inherit OwnedColumns::into_pool()'s check via Pool::to_pool()."
  - "Pool::__new__ ingests the supported optional columns (cat_features/weight/group_id/subgroup_id/baseline); text/embedding/pairs/feature_names are accepted in the upstream-mirror signature but not yet ingested (later plan)."
  - "Added the `arrow` umbrella crate (workspace-pinned 59.0.0 — the SAME version pyo3-arrow 0.19 links) as a direct dep so the binding can validate/read RecordBatch columns."

requirements-completed: [PYAPI-04, PYAPI-06, PYAPI-03]

# Metrics
duration: ~12min
completed: 2026-06-23
---

# Phase 8 Plan 03: Multi-Source Ingestion + Native Pool Summary

**A user can now fit/predict from a NumPy array, a Pandas DataFrame, a pyarrow Table, or a Polars DataFrame — all converging on the existing `OwnedColumns::into_pool()` seam with equal predictions for equal data — while float64 / non-contiguous / ambiguous-object / nullable inputs are rejected with an actionable `CatBoostValueError`, every buffer is copied into owned Rust memory before any GIL release (PYAPI-06 as a code property), and a native `Pool(data, label, cat_features=...)` mirrors upstream `Pool.__init__`.**

## Performance

- **Duration:** ~12 min
- **Tasks:** 2 (both `auto` + `tdd`)
- **Files created:** 2; **modified:** 7
- **Tests:** 22 Rust unit tests + 27 pytest, all green

## Accomplishments

### Task 1 — Multi-source ingest + strict validation + own-before-detach — commit `c9937e0`

- `ingest_to_owned(py, x, y, cat_features)` is the single entry point, dispatching on the Python type:
  1. **Arrow PyCapsule** (`__arrow_c_stream__` / `__arrow_c_array__`) — pyarrow `Table` AND Polars `DataFrame` share one path (`arrow_to_owned` via `pyo3_arrow::input::AnyRecordBatch` → `PyTable` → `into_inner()` RecordBatches). Each column is validated `Float32` + `null_count == 0` (D-12 / T-08-10) and copied f32→f64.
  2. **Pandas DataFrame** (duck-typed `columns`/`dtypes`/`to_numpy`) — numeric columns materialize via `.to_numpy(dtype=float32)` then route through the strict NumPy copy; an object/string column NOT listed in `cat_features` is rejected by name with a `cat_features` suggestion (T-08-10); a declared-categorical column is read as owned strings into the cat slot.
  3. **NumPy ndarray** — the 08-01 path, now raising the typed `CatBoostValueError` on float64 / non-contiguous with the actionable message.
- **Own-before-detach (D-11 / T-08-08):** every branch copies into an owned `OwnedColumns` (`Vec<Vec<f64>>` + `Vec<f64>` + `Vec<Vec<String>>`) BEFORE returning — no `PyReadonlyArray` or Arrow capsule borrow escapes. The Rust unit tests assert the returned value moves into `into_pool()` (compiles iff it borrows nothing from Python).
- The regressor `fit`/`predict` now route through `ingest_to_owned` (later replaced by `data_to_pool` in Task 2).

### Task 2 — Native Pool #[pyclass] (PYAPI-03) — commit `14c413e`

- `pool.rs`: `#[pyclass] Pool` storing `OwnedColumns` + cached `(n_rows, n_cols)`. `#[new]` mirrors upstream `Pool.__init__` (`data, label, cat_features, text_features, embedding_features, weight, group_id, subgroup_id, pairs, baseline, feature_names, thread_count`). `data`+`label` route through `ingest_to_owned`; supported optional columns attach via the `OwnedColumns::with_*` chain.
- `num_row` / `num_col` getters backed by the cached shape (from the new `OwnedColumns::feature_shape()` — validation-free).
- Crate-internal `Pool::to_pool()` clones the owned columns and calls `into_pool()`, **inheriting** cb-data's length / range validation (T-08-11) — never re-implemented; surfaced as `CatBoostValueError`.
- Estimator `fit`/`predict` accept EITHER a framework object OR a native `Pool` via `data_to_pool` (`x.cast::<Pool>()` — PyO3 0.29's rename of `downcast`). `fit`'s `y` is now optional (a `Pool` carries its own label).
- `Pool` registered in the `#[pymodule]`.
- cb-data: added `OwnedColumns::feature_shape() -> (n_rows, n_cols)` reading the float-feature matrix shape without triggering validation.

## Own-Before-Detach Call Sites (for the 08-06 free-threaded test to target)

Every site below COPIES the Python buffer into owned Rust memory before the compute `py.detach()`:

- `crates/catboost-rs-py/src/ingest_py.rs`
  - `numpy_to_owned` → `numpy_matrix_to_cols` (NumPy float32 matrix copy)
  - `label_to_owned` (NumPy float32 label copy — shared by all paths)
  - `pandas_to_owned` (numeric block → `numpy_matrix_to_cols`; cat block → `.astype(str).tolist()` owned strings)
  - `arrow_to_owned` (RecordBatch `Float32Array` copy f32→f64)
- `crates/catboost-rs-py/src/regressor.rs` → `data_to_pool` (the chokepoint both `fit` and `predict` use; for a `Pool` arg it calls `Pool::to_pool()` which clones owned columns)
- `crates/catboost-rs-py/src/pool.rs` → `Pool::new` (ingests + owns at construction) and `Pool::to_pool` (clones owned columns; no Python borrow)

The `fit`/`predict` `py.detach(|| ...)` closures receive only `&Pool` (fully owned). No borrow is alive across the detach (PYAPI-06 buffer-safety code property).

## Pool Signature Coverage vs Upstream

Upstream `Pool.__init__` (RESEARCH 421-428) vs this slice:

- **Ingested:** `data`, `label`, `cat_features`, `weight`, `group_id`, `subgroup_id`, `baseline`.
- **Accepted in signature (upstream parity) but not yet ingested:** `text_features`, `embedding_features`, `pairs`, `feature_names`, `thread_count` (bound + ignored; a later plan wires them).
- **Not yet present:** the remaining upstream Pool kwargs (`column_description`, `graph`, `delimiter`, `has_header`, `ignore_csv_quoting`, `group_weight`, `pairs_weight`, `timestamp`, `feature_tags`) — these are CSV-loading / advanced-metadata knobs deferred to later plans.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] `arrow` crate not a direct dependency**
- **Found during:** Task 1 (compiling the Arrow path).
- **Issue:** `pyo3_arrow::PyTable::into_inner()` yields `arrow` `RecordBatch` / `SchemaRef`, but the binding crate did not depend on `arrow`, so `use arrow::array::...` failed (E0433).
- **Fix:** Added `arrow.workspace = true` (the workspace pin is 59.0.0 — the SAME major pyo3-arrow 0.19 links, so the `RecordBatch`/`Float32Array`/`DataType` types are identical, no version skew).
- **Files:** `crates/catboost-rs-py/Cargo.toml`.
- **Committed in:** `c9937e0`.

**2. [Rule 1 - Bug] PyO3 0.29 renamed `downcast` → `cast`**
- **Found during:** Task 2 (`data_to_pool` Pool detection).
- **Issue:** `Bound::downcast` does not exist in PyO3 0.29; the method is `cast`.
- **Fix:** Used `x.cast::<crate::pool::Pool>()`.
- **Files:** `crates/catboost-rs-py/src/regressor.rs`.
- **Committed in:** `14c413e`.

**3. [Rule 1 - Bug] Smoke test caught stdlib `ValueError`, not the typed `CatBoostValueError`**
- **Found during:** Task 2 verification (full pytest run).
- **Issue:** 08-01's ingest raised the bare `PyValueError`; 08-03 upgrades the ingest path to the typed `CatBoostValueError` (which subclasses `CatBoostError`, NOT stdlib `ValueError`, per the 08-02 taxonomy). The existing `test_smoke.py::test_float64_rejected` caught `ValueError` and so no longer matched. This is a direct, in-scope consequence of the Task-1 ingest upgrade.
- **Fix:** Updated the smoke test to catch `catboost_rs.CatBoostValueError`. (`test_predict_before_fit_raises` still catches `ValueError` correctly — `NotFittedError` subclasses both.)
- **Files:** `crates/catboost-rs-py/tests/test_smoke.py`.
- **Committed in:** `14c413e`.

### Supporting change

**`OwnedColumns::feature_shape()` (cb-data)** — added a small validation-free public accessor returning `(n_rows, n_cols)` from the float-feature matrix, so the Python `Pool` getters (`num_row`/`num_col`) report shape without an eager `into_pool()` rebuild. Pure addition; no existing behavior changed (cb-data still compiles + tests green).

**Total deviations:** 3 auto-fixed (1 Rule 3 - blocking, 2 Rule 1 - bug) + 1 supporting addition. No scope change.

## Threat Mitigations Applied

- **T-08-08 (use-after-free across detach):** every ingest branch copies into `OwnedColumns` before returning; `data_to_pool` / `Pool::to_pool` hand only owned data to the `py.detach()` compute. Asserted by `ingest_py_test.rs` ownership tests.
- **T-08-09 (silent float64→float32 coercion):** float64 rejected with a "float32" message (no coercion); verified by `rejects_float64_with_actionable_message` (Rust) + `test_float64_rejected_actionable` (pytest).
- **T-08-10 (nullable Arrow / ambiguous object):** Arrow null_count==0 enforced; Pandas object column without `cat_features` rejected by name; verified by `test_arrow_nullable_column_rejected_actionable` + `test_pandas_object_column_rejected_actionable`.
- **T-08-11 (length-mismatched Pool column):** `Pool::to_pool()` inherits `OwnedColumns::into_pool()` length checks; label-vs-features caught fail-fast at ingest; verified by `test_pool_length_mismatch_rejected`.

## Known Stubs

- `Pool` accepts `text_features` / `embedding_features` / `pairs` / `feature_names` / `thread_count` in its signature (upstream parity) but does NOT yet ingest them — they are bound-and-ignored placeholders documented in the Pool signature coverage section. A later plan (08-04+) wires text/embedding/pairs. This is a documented, plan-scheduled stub, not a silent one; it does not block this plan's goal (multi-source fit/predict + a native Pool over the supported columns).
- Arrow/Polars `cat_features` (declared-categorical Arrow columns) are not yet ingested — a non-Float32 Arrow column is rejected. Pandas categorical columns ARE ingested (via the object-column path). Arrow categorical ingestion is deferred.

## Self-Check: PASSED

- `crates/catboost-rs-py/src/pool.rs` — FOUND
- `crates/catboost-rs-py/tests/test_ingestion.py` — FOUND
- `crates/catboost-rs-py/src/ingest_py.rs` (ingest_to_owned) — FOUND
- commit `c9937e0` (Task 1) — FOUND
- commit `14c413e` (Task 2) — FOUND
- `cargo test -p catboost-rs-py --features cpu` 22/22 + `pytest tests/` 27/27 — GREEN

---
*Phase: 08-python-bindings-dual-api-packaging*
*Completed: 2026-06-23*
