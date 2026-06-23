---
phase: 08-python-bindings-dual-api-packaging
fixed_at: 2026-06-23T00:33:38Z
review_path: .planning/phases/08-python-bindings-dual-api-packaging/08-REVIEW.md
iteration: 1
findings_in_scope: 12
fixed: 12
skipped: 0
status: all_fixed
---

# Phase 8: Code Review Fix Report

**Fixed at:** 2026-06-23
**Source review:** .planning/phases/08-python-bindings-dual-api-packaging/08-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 12 (fix_scope = all)
- Fixed: 12
- Skipped: 0

All fixes were applied in an isolated git worktree and committed atomically (one
commit per finding, except WR-01/WR-02 which share the same `predict_proba` code
block). Each fix was verified with a scoped `cargo check` (Tier 2 syntax/type
check) on the affected crate(s); GPU-gated code (WR-07) was checked under the
`wgpu` feature, and the shell wrapper (IN-04) was `bash -n` syntax-checked.

## Fixed Issues

### CR-01: Pandas ingest silently REORDERS feature columns

**Files modified:** `crates/catboost-rs-py/src/ingest_py.rs`
**Commit:** b17de96
**Applied fix:** The Pandas path now tracks whether a categorical column has been
seen and REJECTS a numeric column that appears AFTER a categorical one (an
interleaved / non-trailing categorical block) with an actionable
`CatBoostValueError` — instead of silently splitting numeric/cat into independent
blocks that scramble positional feature indices. `OwnedColumns` stores float and
cat as two separate blocks with no per-original-index ordering, so requiring
trailing categorical columns is the correct way to keep `cat_features` (and every
other positional contract) meaningful until ordered ingestion exists. The Arrow
path now REJECTS a non-empty `cat_features` rather than silently dropping it, so an
Arrow/Polars table and the equivalent Pandas DataFrame never diverge in their
feature set. (This also consumed the previously-unused `_cat_features` Arrow param,
partially addressing IN-03.)

### WR-01: `predict_proba` reshape silently truncates an odd-length flat vector

**Files modified:** `crates/catboost-rs-py/src/classifier.rs`
**Commit:** d7f4f64
**Applied fix:** Assert `flat.len() % 2 == 0` before `chunks_exact(2)`, raising a
typed `CatBoostValueError` on an odd output instead of silently dropping the last
object's probabilities.

### WR-02: `PyArray2::from_vec2` on empty `rows` yields `(0, 0)`, not `(0, 2)`

**Files modified:** `crates/catboost-rs-py/src/classifier.rs`
**Commit:** d7f4f64 (shared with WR-01 — same code block)
**Applied fix:** Special-case empty input to return `PyArray2::zeros(py, [0, 2],
false)`, preserving the `(n, 2)` column-count contract for `np.concatenate` /
`vstack` consumers.

### WR-03: workspace `cargo test --features rocm` re-unifies `cpu` via dev-deps

**Files modified:** `crates/cb-model/Cargo.toml`, `crates/cb-train/Cargo.toml`,
`crates/catboost-rs-py/Cargo.toml`, `crates/cb-oracle/Cargo.toml`
**Commit:** 6707b92
**Applied fix:** Pinned every backend-bearing dev-dependency (`cb-backend`,
`cb-model`, `cb-train`) with `default-features = false` so they no longer re-enable
cb-backend's default `cpu` feature. `cb-oracle` has no production backend dep and
no feature passthrough, so a `[features]` block (`default = ["cpu"]`, plus
rocm/wgpu/cuda passthroughs) was added — its integration tests still get a cpu
backend under per-crate `cargo test -p cb-oracle`, while a workspace `--features
rocm` selects rocm WITHOUT unifying cpu. All four affected crates' test targets
verified to compile.

### WR-04: `data_to_pool` Pool fast-path bypasses strict input validation

**Files modified:** `crates/catboost-rs-py/src/estimator.rs`
**Commit:** dff85c4
**Applied fix:** Documented the intentional error-surface asymmetry — the Pool
fast-path runs only the length check (feature-width mismatch defers to the facade's
`FeatureMismatch` in `predict_with`) and silently ignores `y` (the Pool carries its
own label) — both in the function doc and at the call site. (Documentation fix, as
the reviewer noted the behavior is acceptable if intentional; a width-mismatch
parity test was NOT added — see "Notes" below.)

### WR-05: Param registry validates names but never validates VALUE ranges

**Files modified:** `crates/catboost-rs-py/src/params.rs`
**Commit:** 4239bb7
**Applied fix:** Added a `check_range` helper and range-validated the bounded
numeric params in `make_builder` (`iterations >= 1`, `depth` in `[1, 16]`,
`0 < learning_rate <= 1`, `l2_leaf_reg >= 0`, `random_strength >= 0`, `border_count`
in `[1, 65535]`, `0 < subsample <= 1`), plus NaN/inf rejection, raising
`CatBoostParameterError` at the offending param before training.

### WR-06: `accuracy_score` compares rounded f64 labels with `f64::EPSILON`

**Files modified:** `crates/catboost-rs-py/src/estimator.rs`
**Commit:** 9c696bd
**Applied fix:** Compare `(t.round() as i64) == (p.round() as i64)` with a
`is_finite` guard, making the integer-equality intent explicit and correct for any
label magnitude (the previous `< f64::EPSILON` form was correct only for 0.0/1.0
binary labels).

### WR-07: `const_der_host` discards the device handle it allocates

**Files modified:** `crates/cb-backend/src/gpu_backend.rs`
**Commit:** bb7be61
**Applied fix:** Removed the dead `const_der_handle(value, n)?` call (its result was
discarded with `let _ =`, never read back, so the documented `Degenerate`
validation was never exercised — it only wasted a device allocation) and its
now-unused import, and corrected the doc comment to state the constant is
host-materialized. Verified under the `wgpu` feature (the module is gated to
wgpu/cuda/rocm).

### IN-01: `pandas_n_rows` row count diverges from `feature_shape` for cat-only frames

**Files modified:** `crates/cb-data/src/ingest/owned.rs`
**Commit:** 2671b32
**Applied fix:** `feature_shape()` now derives `n_rows` from the first feature kind
that has a column (float → cat → text → embedding), so an all-categorical
`OwnedColumns` reports its true row count rather than 0.

### IN-02: `closest_match` fallback `.unwrap_or("iterations")` is dead defaulting

**Files modified:** `crates/catboost-rs-py/src/params.rs`
**Commit:** bf19f9c
**Applied fix:** Replaced `.unwrap_or("iterations")` with `.expect(...)` documenting
the non-empty-`VOCABULARY` invariant honestly (the reviewer-suggested option). The
`expect` is on a provably-unreachable branch in a hint-generation path, consistent
with the project's `unwrap` ban (which targets fallible production paths).

### IN-03: `arrow_to_owned` takes `_py` and `_cat_features` that are unused

**Files modified:** `crates/catboost-rs-py/src/ingest_py.rs`
**Commit:** 2d64ec6 (and b17de96 for `_cat_features`)
**Applied fix:** `_cat_features` was consumed in the CR-01 fix (the Arrow path now
rejects non-empty `cat_features`). The remaining unused `_py` token was dropped from
the `arrow_to_owned` signature and its single call site.

### IN-04: `pyproject-rocm.toml` documents a manual swap with no automation

**Files modified:** `crates/catboost-rs-py/build-rocm-wheel.sh` (new),
`crates/catboost-rs-py/pyproject-rocm.toml`
**Commit:** 06704fa
**Applied fix:** Added `build-rocm-wheel.sh`, a wrapper that swaps
`pyproject-rocm.toml` in for `pyproject.toml` transactionally — it backs up the cpu
pyproject, runs maturin, and ALWAYS restores the original on exit
(success/error/interrupt via a `trap`), so a half-applied swap can never persist and
mis-name the wheel. Updated the doc comment to point at the wrapper. `bash -n`
syntax-checked.

## Notes for human verification

- **WR-05 (range validation)** is a logic change (new rejection conditions). The
  ranges chosen mirror upstream CatBoost's documented bounds (`depth <= 16`,
  `border_count <= 65535`, `0 < learning_rate <= 1`, `0 < subsample <= 1`), but a
  developer should confirm these bounds match the project's intended param contract
  and that no existing valid fixture/test passes an out-of-range value (none was
  found among the reviewed tests, but the full param test suite was not executed).
- **WR-04** was addressed as documentation only (per the reviewer's "acceptable if
  intentional"). The suggested width-mismatch parity test (asserting a malformed
  Pool yields the same typed error as a malformed NumPy array) was NOT added; it is
  left as a follow-up test task since the error surfaces in the facade's
  `predict_with`, outside this binding crate.

---

_Fixed: 2026-06-23_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
