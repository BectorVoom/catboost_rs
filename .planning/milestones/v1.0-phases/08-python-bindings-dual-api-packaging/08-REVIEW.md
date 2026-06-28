---
phase: 08-python-bindings-dual-api-packaging
reviewed: 2026-06-23T00:33:38Z
depth: standard
files_reviewed: 41
files_reviewed_list:
  - .github/workflows/python-wheels.yml
  - crates/catboost-rs-py/Cargo.toml
  - crates/catboost-rs-py/FREE_THREADING.md
  - crates/catboost-rs-py/PACKAGING.md
  - crates/catboost-rs-py/pyproject-rocm.toml
  - crates/catboost-rs-py/pyproject.toml
  - crates/catboost-rs-py/src/classifier.rs
  - crates/catboost-rs-py/src/errors.rs
  - crates/catboost-rs-py/src/errors_test.rs
  - crates/catboost-rs-py/src/estimator.rs
  - crates/catboost-rs-py/src/estimator_test.rs
  - crates/catboost-rs-py/src/ingest_py.rs
  - crates/catboost-rs-py/src/ingest_py_test.rs
  - crates/catboost-rs-py/src/lib.rs
  - crates/catboost-rs-py/src/params.rs
  - crates/catboost-rs-py/src/params_test.rs
  - crates/catboost-rs-py/src/pool.rs
  - crates/catboost-rs-py/src/ranker.rs
  - crates/catboost-rs-py/src/regressor.rs
  - crates/catboost-rs-py/tests/conftest.py
  - crates/catboost-rs-py/tests/test_check_estimator.py
  - crates/catboost-rs-py/tests/test_errors.py
  - crates/catboost-rs-py/tests/test_free_threaded.py
  - crates/catboost-rs-py/tests/test_ingestion.py
  - crates/catboost-rs-py/tests/test_native_api.py
  - crates/catboost-rs-py/tests/test_oracle_parity.py
  - crates/catboost-rs-py/tests/test_params.py
  - crates/catboost-rs-py/tests/test_smoke.py
  - crates/catboost-rs/Cargo.toml
  - crates/catboost-rs/src/builder.rs
  - crates/cb-backend/src/gpu_backend.rs
  - crates/cb-backend/src/gpu_backend_test.rs
  - crates/cb-backend/src/lib.rs
  - crates/cb-data/src/ingest/owned.rs
  - crates/cb-model/Cargo.toml
  - crates/cb-train/Cargo.toml
findings:
  critical: 1
  warning: 7
  info: 4
  total: 12
status: issues_found
---

# Phase 8: Code Review Report

**Reviewed:** 2026-06-23T00:33:38Z
**Depth:** standard
**Files Reviewed:** 41
**Status:** issues_found

## Summary

This phase delivers the PyO3 binding crate (`catboost-rs-py`) exposing the
`catboost_rs` Python module — the regressor/classifier/ranker trio, a native
`Pool`, multi-source ingestion (NumPy/Pandas/Arrow/Polars), a typed-exception
taxonomy, a param-vocabulary registry, and abi3 CPU + ROCm wheel packaging. It
also adds the generic `GpuBackend` (08-08 gap) wiring the GPU der seam into the
facade train path.

The code is carefully written, well-documented, and largely `unwrap`-free in
production paths. However the adversarial pass surfaced one BLOCKER (a
silent feature-reorder in the Pandas ingest path that breaks the
positional `cat_features` contract and silently corrupts feature alignment for
mixed-type DataFrames), plus seven warnings centered on edge-case robustness
(empty/odd-length reshape, feature-unification in workspace tests, an
unvalidated `data_to_pool` Pool fast-path that skips strict-input checks), and
four info-level quality items.

No injection, secrets, unsafe-deserialization, or memory-safety defects were
found. The `unsafe`-free Rust boundary and own-before-detach discipline hold.

## Critical Issues

### CR-01: Pandas ingest silently REORDERS feature columns, breaking positional `cat_features` and numeric column order

**File:** `crates/catboost-rs-py/src/ingest_py.rs:199-273`
**Issue:**
`pandas_to_owned` partitions the DataFrame into two independent blocks —
`numeric_names` (appended in column order) and `cat_names` (appended in
`cat_features`-index order) — then builds `OwnedColumns` with the numeric block
as the float-feature matrix and the cat block separately. This SILENTLY reorders
features relative to the original DataFrame whenever a categorical column sits
between numeric columns.

Concretely, for a DataFrame `[a, b, c]` with `cat_features=[1]` (column `b`):
- The float matrix becomes `[a, c]` (original indices 0 and 2 collapsed to
  positions 0 and 1).
- The user passed `cat_features=[1]` meaning "column index 1 is categorical",
  but after the split the surviving numeric features no longer occupy their
  original indices, so any *other* index-based contract (a second
  `cat_features` entry, `ignored_features`, monotone constraints, feature
  weights, SHAP feature attribution, `feature_names_` alignment) refers to the
  wrong column. The float feature at apply-time position 1 is now `c`, not the
  original column 1.

Because the reorder is deterministic, train→predict on the *same* DataFrame are
self-consistent, which is exactly why this passes the existing
numeric-only tests (`test_pandas_matches_numpy` has no cat columns) and will not
be caught until a user supplies a real mixed-type frame — at which point the
model trains on misaligned features with no error. This is a data-corruption /
silent-wrong-result defect (the project's "oracle-tested to 1e-5" bar cannot
hold if column identity is scrambled).

Additionally, the categorical block is dropped entirely on the Arrow/Polars
path (`arrow_to_owned` ignores `_cat_features`) while accepted on Pandas,
so the same logical DataFrame ingested via two sources yields different feature
sets — another silent divergence.

**Fix:** Preserve original column order. Build a single ordered list of columns
and assign each to its float-or-cat slot while tracking the ORIGINAL positional
index, so `cat_features` indices remain meaningful and numeric features keep
their source positions. For example, iterate columns once and push into a
position-tagged structure, or reject mixed numeric/categorical interleaving with
an explicit "categorical columns must be trailing" error until ordered
ingestion is implemented:

```rust
// Sketch: keep original order, map each column to a typed slot by original idx.
// Reject silent reordering — either preserve positions or error out.
for idx in 0..n_cols {
    let col_name = columns.get_item(idx)?;
    if cats.contains(&idx) {
        // record as cat at ORIGINAL position idx
    } else {
        // record as numeric at ORIGINAL position idx
    }
}
// then materialize float_cols / cat_cols so downstream feature index N
// corresponds to original column N (not the post-partition position).
```
At minimum, add a test with `cat_features` pointing at a non-trailing column and
assert feature alignment, and make the Arrow path consistent (reject or ingest
`cat_features` identically) rather than silently dropping it.

## Warnings

### WR-01: `predict_proba` reshape via `chunks_exact(2)` silently truncates an odd-length flat vector

**File:** `crates/catboost-rs-py/src/classifier.rs:133-134`
**Issue:** `flat.chunks_exact(2).map(<[f64]>::to_vec).collect()` discards any
trailing element when `flat.len()` is odd. The facade currently guarantees two
values per object for `Probability`, so today it is even — but this is an
unchecked assumption: if `predict_with(Probability)` ever returns a single
column (e.g. a multiclass or degenerate model) the binding would silently drop
the last object's probabilities and return an `(n-? , 2)` array shorter than the
input, with no error. A silent shape mismatch is worse than a raised error.
**Fix:** Assert the invariant before reshaping:
```rust
if flat.len() % 2 != 0 {
    return Err(CatBoostValueError::new_err(format!(
        "probability output length {} is not divisible by 2 (expected (n, 2))",
        flat.len()
    )));
}
let rows: Vec<Vec<f64>> = flat.chunks_exact(2).map(<[f64]>::to_vec).collect();
```

### WR-02: `PyArray2::from_vec2` on an empty `rows` yields shape `(0, 0)`, not `(0, 2)`

**File:** `crates/catboost-rs-py/src/classifier.rs:133-134`
**Issue:** When `predict_proba` is called on a zero-row input, `rows` is empty
and `PyArray2::from_vec2(py, &[])` produces a `(0, 0)` array. The oracle test
(`test_oracle_parity.py:53`) and any downstream consumer assert
`proba.shape == (n, 2)`; a `(0, 2)` contract is violated for the empty case,
which can break `np.concatenate`/`vstack` pipelines that rely on the column
count. **Fix:** Special-case the empty input to construct an explicit
`(0, 2)`-shaped array (e.g. via `PyArray2::zeros(py, [0, 2], false)`), or
document and test the empty-input shape.

### WR-03: workspace-wide `cargo test --features rocm` re-unifies `cpu` via un-pinned dev-dependencies (the documented landmine)

**File:** `crates/cb-model/Cargo.toml:46`, `crates/cb-train/Cargo.toml:39`, `crates/catboost-rs-py/Cargo.toml:43`, `crates/cb-oracle/Cargo.toml:33,35`
**Issue:** The production dependency graph correctly pins every backend-bearing
dep with `default-features = false` (the feature-unification landmine called out
in `cb-backend/Cargo.toml`). But several `[dev-dependencies]` pull
`cb-backend` / `cb-model` WITHOUT `default-features = false`
(`cb-model` dev-dep `cb-backend`, `cb-train` dev-dep `cb-model`,
`catboost-rs-py` dev-dep `cb-model`, `cb-oracle` dev-deps). Each of these
re-enables `cb-backend`'s default `cpu` feature. Under a workspace-level
`cargo test --features rocm` (the in-env GPU validation command per project
memory), feature unification then activates BOTH `cpu` and `rocm` in the build
graph — which silently breaks the rocm runtime selection (`SelectedRuntime`
resolves to `CpuRuntime` because `cpu` wins the cfg precedence in
`cb-backend/src/lib.rs:62`). This is the exact failure mode the production
pins were designed to prevent; it is merely pushed into the test graph.
**Fix:** Either pin the dev-deps with `default-features = false` and an explicit
feature where a backend is genuinely needed for the test, or document/enforce
that rocm validation runs per-crate (`-p <crate>`), never `--workspace`. Given
the project has already been burned by this (MEMORY: "feature unification breaks
rocm runtime"), pinning the dev-deps is the durable fix.

### WR-04: `data_to_pool` Pool fast-path bypasses the strict D-12 input validation applied to framework objects

**File:** `crates/catboost-rs-py/src/estimator.rs:211-222`, `crates/catboost-rs-py/src/pool.rs:147-152`
**Issue:** When `x` is a native `Pool`, `data_to_pool` returns
`pool_ref.borrow().to_pool()` directly, which runs ONLY the
`OwnedColumns::into_pool()` length check. A `Pool` constructed earlier from a
float64 / non-contiguous / nullable source already had those rejected at
construction, so this is mostly safe — but the `y` argument is silently dropped
on the Pool path (documented), and there is no guard that the Pool's feature
width matches the fitted model on `predict` until the facade's
`FeatureMismatch` fires deep in `predict_with`. The asymmetry (strict checks for
NumPy, only length checks for Pool) means a malformed Pool surfaces a different,
later error than the equivalent NumPy input. **Fix:** This is acceptable if
intentional, but document the dropped-`y` and width-defer behavior at the
`data_to_pool` call sites and add a test that a width-mismatched Pool yields the
same typed `CatBoostValueError` as a width-mismatched NumPy array, so the
error surface is consistent across input kinds.

### WR-05: Param registry validates names but never validates VALUE ranges; nonsensical numeric params reach training unguarded

**File:** `crates/catboost-rs-py/src/params.rs:286-468`
**Issue:** `validate_params` only checks that each kwarg *name* is in the
vocabulary and Implemented; `make_builder` then extracts values by type but
applies no range validation. A user passing `iterations=0`, `depth=0` or a huge
`depth`, `learning_rate=-1.0`, `subsample=5.0`, or `l2_leaf_reg=-3.0` is accepted
and forwarded to the builder. Depending on the train loop this either trains a
degenerate model silently or surfaces a low-level error far from the param that
caused it. Upstream CatBoost rejects out-of-range params at construction with a
clear message. **Fix:** Add range/domain validation in `make_builder` (or a
dedicated validator) for the bounded params (`0 < learning_rate`,
`0 < subsample <= 1`, `depth >= 1`, `l2_leaf_reg >= 0`, `iterations >= 1`),
raising `CatBoostParameterError` with the offending name/value before training.

### WR-06: `accuracy_score` compares rounded f64 labels with `f64::EPSILON`, which is too tight for non-trivial label magnitudes

**File:** `crates/catboost-rs-py/src/estimator.rs:182`
**Issue:** `(t.round() - p.round()).abs() < f64::EPSILON` works for `0.0`/`1.0`
binary labels (the current classifier output), but `f64::EPSILON` (~2.2e-16) is
the gap near 1.0; for any label/round result with magnitude > ~2 the
representable gap exceeds `EPSILON`, so two equal integer-valued f64s could in
principle compare unequal after subtraction rounding, and more importantly the
intent (integer equality) is obscured. **Fix:** Compare the rounded integers
directly: `(t.round() as i64) == (p.round() as i64)` (guarding NaN/overflow), or
use a tolerance of `0.5`. The current form is correct only for the binary case
and is a latent bug if multiclass labels arrive.

### WR-07: `const_der_host` discards the device handle it allocates, performing a pointless device round-trip

**File:** `crates/cb-backend/src/gpu_backend.rs:58-67`
**Issue:** `const_der_host` calls `const_der_handle(value, n)?` purely for its
error side-effect, immediately discards the result with `let _ =`, then returns
a host `vec![value; n]`. The doc comment claims the read-back surfaces a
`CbError::Degenerate` "never a silent all-zero buffer", but no read-back of the
handle occurs — the returned vector is the host-materialized constant, NOT the
device buffer. So the stated safety property (device read-back validation) is
not actually exercised; the `const_der_handle` call is dead work whose only
effect is a possible allocation error. Either the comment is wrong or the
intended read-back is missing. **Fix:** Either drop the `const_der_handle` call
entirely (and correct the comment to say the constant is host-materialized), or
actually read the handle back and use that buffer so the documented validation
holds. As written it is misleading and wastes a device allocation per call.

## Info

### IN-01: `pandas_n_rows` row count diverges from `feature_shape` for categorical-only frames

**File:** `crates/catboost-rs-py/src/ingest_py.rs:266-272`, `crates/cb-data/src/ingest/owned.rs:113-117`
**Issue:** For a DataFrame with no numeric columns, `float_cols` is empty, so
`OwnedColumns::feature_shape()` reports `n_rows = 0` (it reads the first float
column). `Pool.num_row` would then report 0 even though `pandas_n_rows`
correctly computed the true row count for label alignment. Edge case; only bites
all-categorical pools (not currently a fixture path). **Fix:** Derive
`feature_shape` row count from the max of all feature kinds + label, or have
`Pool` cache `pandas_n_rows`.

### IN-02: `closest_match` fallback `.unwrap_or("iterations")` is unreachable dead defaulting

**File:** `crates/catboost-rs-py/src/params.rs:277`
**Issue:** `VOCABULARY` is a non-empty `const`, so `min_by_key` always returns
`Some`; the `.unwrap_or("iterations")` branch is dead. Harmless, but the literal
`"iterations"` as a typo suggestion for an unrelated name would be misleading if
it ever fired. **Fix:** Acceptable as defensive code; optionally
`expect("VOCABULARY is non-empty")` would document the invariant more honestly,
or leave a comment.

### IN-03: `arrow_to_owned` takes `_py` and `_cat_features` that are entirely unused

**File:** `crates/catboost-rs-py/src/ingest_py.rs:295-300`
**Issue:** Both leading params are unused (prefixed `_`). `_cat_features`
silently dropping categorical declarations is the consistency gap noted in
CR-01; `_py` is signature-parity dead weight. **Fix:** Once CR-01 is addressed,
`_cat_features` should be consumed; drop `_py` if the Arrow path never needs the
token.

### IN-04: `pyproject-rocm.toml` documents a manual `cp pyproject-rocm.toml pyproject.toml` swap with no automation

**File:** `crates/catboost-rs-py/pyproject-rocm.toml:16-21`
**Issue:** The rocm distribution build relies on a hand-copied pyproject swap and
an `LD_PRELOAD`/`ROCM_PATH` runtime incantation documented only in comments. This
is error-prone (a stale `pyproject.toml` left in a checkout would publish the
wrong distribution name) and unguarded by CI (correctly, per D-06). **Fix:**
Provide a small build wrapper script (or maturin `--config-file` if supported)
that selects the rocm pyproject without mutating the tracked file, so a
half-applied swap cannot silently mis-name the wheel.

---

_Reviewed: 2026-06-23T00:33:38Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
