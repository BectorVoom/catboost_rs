---
title: "FSTR-03 — Partial-Dependence Feature Importance (one or two features)"
status: draft
format: markdown
spec_version: 1
updated_at: 2026-07-16T00:00:00Z
phase: 18
requirement_ids:
  - FSTR-03
source_requirements:
  - ".planning/REQUIREMENTS.md:31 (FSTR-03) — git-recovered (commit a82289c); NOT in the working tree. Confirm the canonical revision before flipping the requirement checkbox."
  - ".planning/ROADMAP.md:131 (Phase 18 Success Criterion 3) — git-recovered (commit a82289c); not in the working tree."
pageindex_target: "catboost_rs/SPEC.md (PageIndex folder id cmrhcxbtm000104jr3i5jzm0m, indexed 2026-07-12, status=completed). PENDING re-index of THIS hardened revision — see §10 (the MCP's process_document ingests PDFs and offers no in-place Markdown upsert)."
---

# FSTR-03 — Partial-Dependence Feature Importance

> **Draft.** Not approved / not implemented. This spec decomposes FSTR-03 into
> failure-isolated behavioral specifications for TDD (see `PLAN.md`). No production
> code is authored by this document.

## 1. Context

catboost-rs computes several feature-importance modes in `cb-model/src/fstr.rs`
(`prediction_values_change`, `interaction`, `loss_function_change`)
`[VERIFIED: CODEGRAPH crates/cb-model/src/fstr.rs:123,288,490]`. **Partial
dependence does not exist** — no `partial_dependence` symbol appears anywhere in
the Rust workspace, and the vendored `catboost-master/` C++ tree contains **no**
partial-dependence source either `[VERIFIED: LOCAL grep 'partial.dependence|PartialDependence'
→ 0 hits in crates/ and catboost-master/]`. Upstream partial dependence is the
Python utility `catboost.CatBoost.plot_partial_dependence(pool, features)`, added
after this vendored snapshot `[VERIFIED: CONTEXT7 /catboost/catboost plot_partial_dependence tutorial]`.

Because the capability is **greenfield**, it carries **no bit-exact regression
surface** to preserve during red→green — unlike FSTR-01/02, whose numeric paths
must stay byte-identical `[VERIFIED: LOCAL crates/cb-model/src/fstr.rs:14-17]`. This is
why FSTR-03 is the recommended first slice of Phase 18
`[VERIFIED: LOCAL .planning/research/unimplemented-parity-research.md:190-205]`.

**Upstream definition (the target to match).** For a target feature set `x_S`
and the complementary features `x_C`, partial dependence is
`f_{x_S}(x_S) = (1/n) · Σ_{i=1..n} f(x_S, x_C^{(i)})`: the target feature(s) are
pinned to a grid value while every other feature keeps its actual per-object
dataset value, the model is applied (`RawFormulaVal`), and the result is averaged
over all `n` dataset objects `[VERIFIED: CONTEXT7 /catboost/catboost "partial dependence
function estimated by calculating averages of the model's predictions across the
training data"]`. A one-feature plot is a 1-D curve over that feature's grid; a
two-feature plot is a 2-D grid `[VERIFIED: CONTEXT7 /catboost/catboost plot_partial_dependence]`.

## 2. Scope and non-goals

### In scope (this slice)
- Partial dependence for **one** float feature over its grid.
- Partial dependence for **two** float features over the Cartesian product grid.
- Grid derivation from the model's stored float-feature borders, matching
  upstream's x-axis grid.
- Typed rejection of unsupported inputs (out-of-range index, non-float /
  categorical target feature, >2 features, empty dataset).
- CPU-only, pure-Rust apply path (`predict_raw`) — no GPU, no new crate.
- Oracle fixture + integration test at the `1e-5` CPU parity bar.

### Non-goals
- **Categorical / CTR / text / embedding target features** — deferred; rejected
  with a typed error in this slice (mirrors upstream's export-guard pattern).
  `[INFERRED: keeps the first slice float-only and failure-isolated; broader kinds
  are a follow-up spec]`
- Three-or-more-feature partial dependence (upstream itself is 1–2)
  `[VERIFIED: CONTEXT7 /catboost/catboost "typically generated for one or two features"]`.
- Plotting / visualization (the Rust surface returns data, not a figure).
- Non-oblivious grid special-casing beyond what `predict_raw` already handles
  (the apply path already spans both tree variants
  `[VERIFIED: CODEGRAPH crates/cb-model/src/apply.rs:370]`).
- FSTR-01 (Interaction on CTR) and FSTR-02 (LossFunctionChange) — separate Phase-18 slices.

## 3. Dependencies

| Dependency | Typed interface | Evidence |
|-----------|-----------------|----------|
| Model apply (RawFormulaVal) | `predict_raw(model: &Model, feature_values: &[Vec<f32>]) -> Vec<f64>` (SoA: `feature_values[f]` = float feature `f`'s per-object `f32` column) | `[VERIFIED: CODEGRAPH crates/cb-model/src/apply.rs:370]` |
| Float-feature grid source | **`pub` field** `cb_model::Model.float_feature_borders: Vec<Vec<f64>>` (per-feature ascending borders; empty inner vecs preserved so index == float-feature index) | `[VERIFIED: CODEGRAPH crates/cb-model/src/model.rs:293 (field on the struct at :272)]` |
| Float-feature count | `n_float = model.float_feature_borders.len()` (the model exposes NO other flat-feature layout — see §4 index-space note) | `[VERIFIED: CODEGRAPH crates/catboost-rs/src/model.rs:51-53 `n_float_features` reads `float_feature_borders.len()`]` |
| Silent NaN-pad hazard | `predict_raw_cat` gathers each object row via `col.get(obj).copied().unwrap_or(f32::NAN)` — a **short or missing** column silently reads `NaN` (no error), which would corrupt the PD average. Motivates the PDP-05 `MalformedColumns` guard. | `[VERIFIED: CODEGRAPH crates/cb-model/src/apply.rs:404-407]` |
| Deterministic mean fold | `cb_core::sum_f64(&[f64]) -> f64` (sequential left-to-right `f64` fold, upstream order, D-08 — never `.sum()`) | `[VERIFIED: CODEGRAPH crates/cb-core/src/reduction.rs:32]` |
| Oracle comparator | `cb_oracle::compare::assert_abs_close(expected: &[f64], actual: &[f64], tol: f64) -> Result<(), OracleError>` — **returns a `Result`, never panics**; oracle tests propagate/`unwrap` it under the test-only lint allow | `[VERIFIED: CODEGRAPH crates/cb-oracle/src/compare.rs:46]` |
| Fixture recipe | upstream `catboost==1.2.10` in a venv, `.npy` + `config.json` committed under `crates/cb-oracle/fixtures/<name>/` | `[VERIFIED: LOCAL crates/cb-oracle/fixtures/advanced_fstr/gen_fixtures.py:1-70]` |
| Oracle test harness pattern | integration test under `crates/cb-model/tests/` with top `#![allow(clippy::unwrap_used, expect_used, panic, indexing_slicing)]`, `const TOL: f64 = 1e-5`, a `fixture()` path helper, and `ndarray_npy::read_npy` / `cb_oracle::load_f64_vec` loaders | `[VERIFIED: LOCAL crates/cb-model/tests/advanced_fstr_oracle_test.rs:18-49]` |

**Layering:** all work lives in `cb-model` (already depends on `cb-core`); no
`cb-train`/`cb-backend`/`cb-compute` edge is added, so no CubeCL feature-unification
risk `[VERIFIED: LOCAL .planning/research/unimplemented-parity-research.md:47,182]`.

## 4. Typed contracts

### Feature-index space (load-bearing — read first)

`features` and `columns` are **both indexed in float-feature-index space**:
index `f ∈ 0..n_float` where `n_float = model.float_feature_borders.len()`, and
`columns[f]` is that float feature's per-object `f32` column (exactly the SoA
layout `predict_raw` consumes). **This is the only index space the API can
honor**, because `cb_model::Model` stores *no* flat-feature / feature-kind map —
its only feature metadata is `float_feature_borders` (float features) and
`ctr_data: Option<CtrData>` (whether CTR splits exist)
`[VERIFIED: CODEGRAPH crates/cb-model/src/model.rs:272-313 — Model has no flat-feature
kind list]`. For the **numeric-only fixture model in scope**, the float-feature
index is *identical* to upstream `plot_partial_dependence(pool, features)`'s
flat feature index (the pool has only float features), so the oracle comparison
is apples-to-apples. `[INFERRED: numeric-only ⇒ float-index == flat-index]`

**Consequence for validation:** a *categorical / CTR / text / embedding* target
feature **cannot be expressed** through this float-index column API and **cannot
be detected** from `Model` metadata, so there is no `UnsupportedFeatureKind`
error — such an index is simply out of the float range (`FeatureIndexOutOfRange`).
Categorical / flat-index partial dependence (which needs the internal→regular
feature-index remap) is a **deferred follow-up** (an FSTR-01-adjacent concern),
explicitly out of this slice. `[VERIFIED: CODEGRAPH crates/cb-model/src/model.rs:272-313]`

```rust
/// Result of a partial-dependence computation for one or two features.
pub struct PartialDependence {
    /// The 1 or 2 target float-feature indices, in the order requested.
    pub features: Vec<usize>,
    /// Per-target-feature ascending grid values (`grids.len() == features.len()`).
    pub grids: Vec<Vec<f64>>,
    /// Averaged RawFormulaVal, row-major over the Cartesian product of `grids`.
    /// Single feature: `values.len() == grids[0].len()`.
    /// Two features:   `values.len() == grids[0].len() * grids[1].len()`,
    ///                 index `a*grids[1].len() + b` = (grids[0][a], grids[1][b]).
    pub values: Vec<f64>,
}

/// Typed failure at the partial-dependence boundary (no panic, no unwrap).
/// Every variant is REACHABLE and IMPLEMENTABLE from `Model` + `columns` alone
/// (see §5 PDP-05 for the one-Red-test-per-arm mapping).
#[derive(Debug, thiserror::Error)]
pub enum PdpError {
    /// A requested feature index is >= the model's float-feature count. Also the
    /// (only) outcome for a categorical/CTR target, which has no float index.
    #[error("feature index {index} out of range (model has {n_float} float features)")]
    FeatureIndexOutOfRange { index: usize, n_float: usize },
    /// `features.len()` is not 1 or 2 (upstream partial dependence is 1–2 features).
    #[error("partial dependence supports 1 or 2 features, got {requested}")]
    UnsupportedFeatureArity { requested: usize },
    /// A two-feature request named the SAME float feature twice; a 2-D PD surface
    /// over `(f, f)` is degenerate. Conservative API precondition of this slice.
    #[error("partial dependence over two features requires distinct features; feature {index} was given twice")]
    DuplicateFeature { index: usize },
    /// `columns` does not match the model's float-feature layout: the count is not
    /// `n_float`, or the columns are ragged (unequal lengths). Guards the silent
    /// NaN-pad hazard (§3) — a short/narrow column would otherwise average garbage.
    #[error("columns malformed: expected {expected_float_features} equal-length float columns, got {actual}")]
    MalformedColumns { expected_float_features: usize, actual: String },
    /// The dataset has no objects (`columns` empty, or every column length 0).
    #[error("dataset has no objects")]
    EmptyDataset,
}

/// Public entry point (composes PDP-05 validation + PDP-02 grid + PDP-01 engine).
/// `columns[f]` is float feature `f`'s per-object `f32` column (SoA, as `predict_raw`);
/// `columns.len()` MUST equal `model.float_feature_borders.len()` (validated).
pub fn partial_dependence(
    model: &Model,
    columns: &[Vec<f32>],
    features: &[usize],
) -> Result<PartialDependence, PdpError>;
```

> Exact placement of `PdpError` (a dedicated enum in the new module vs. new
> `cb_model::ModelError` variants) is a plan-time wiring choice — see `PLAN.md`
> Task 0. It does not change the behavioral contract. `[INFERRED]`
>
> **Change note (2026-07-16 hardening):** the earlier draft's
> `UnsupportedFeatureKind { index }` variant was **removed** — it is not
> implementable (Model exposes no feature-kind map) nor reachable (a non-float
> index is caught by `FeatureIndexOutOfRange`). It is replaced by two
> implementable, individually-testable guards — `DuplicateFeature` and
> `MalformedColumns` — the latter closing the verified silent-NaN-pad hole (§3).

## 5. Failure-isolated behavioral specifications

Each specification below has one behavioral responsibility, one trigger, an
explicit dependency boundary, and one primary cause of acceptance-test failure.

---

### PDP-01 — Single-feature averaging engine over an explicit grid

- **Status:** implemented (2026-07-16; unit AT-01a/b/c green in `partial_dependence_test.rs`)
- **Responsibility:** given a grid, compute the averaged RawFormulaVal for each
  grid point. *Isolates the averaging math from grid derivation.*
- **Preconditions (guaranteed by PDP-05 before the engine runs):** `columns.len()
  == n_float` (`= model.float_feature_borders.len()`); every column has the same
  length `n >= 1` (rectangular, non-empty); `feature < n_float`. Because these
  hold, `predict_raw` never NaN-pads a short/missing column (§3 hazard closed).
- **Input:** `model: &Model`, `columns: &[Vec<f32>]`, `feature: usize`, `grid: &[f64]`.
- **Output:** `Vec<f64>`, `len == grid.len()`.
- **Dependencies:** `predict_raw` (apply.rs:370), `cb_core::sum_f64`.
- **Behavior (Given/When/Then):**
  - **Given** a model and `n` object columns, **when** the engine is asked for
    grid point `grid[k]`, **then** it forms a working column set identical to
    `columns` except that the whole target column is set to `grid[k] as f32`,
    calls `predict_raw`, and returns `sum_f64(preds) / n` for that `k`.
  - **Given** a grid of length `m`, **then** output length is exactly `m`, in grid order.
- **Invariants / side effects:** pure; the caller's `columns` are not mutated
  (the override is on a working copy / view). Every mean folds through
  `sum_f64`, never a raw `.sum()` (D-08).
- **Acceptance tests (unit, self-consistent — no oracle needed):**
  - AT-01a: for a grid value equal to a constant `v`, the engine result equals
    `mean(predict_raw(model, columns_with_target_column_all_v))` computed
    independently in the test → proves the override+average wiring.
  - AT-01b: output length == grid length; grid order preserved.
  - AT-01c: `columns` argument is unchanged after the call (no mutation).
- **Out of scope:** deriving the grid; two features; validation.
- **Traceability:** `[VERIFIED: CONTEXT7 averaging formula]`,
  `[VERIFIED: CODEGRAPH apply.rs:370]`.

---

### PDP-02 — Per-bin grid derivation from model borders (single float feature)

- **Status:** implemented (2026-07-16; unit `grid_for_feature_is_per_bin_representatives` green)
- **Responsibility:** produce the per-bin representative grid for a float feature.
  *Grid transform RESOLVED (was the one open convention question).*
- **Preconditions:** `feature` is a valid float feature index.
- **Input:** `model: &Model`, `feature: usize`.
- **Output:** `Vec<f64>` ascending grid, length `n_borders + 1` (one per bin).
- **Dependencies:** `model.float_feature_borders[feature]` (model.rs:293).
- **RESOLVED transform (ground-truth, `catboost==1.2.10`).** Upstream computes PD
  **per BIN**: borders `b_0 < … < b_{k-1}` define `k+1` bins
  (`(-inf,b_0], (b_0,b_1], …, (b_{k-1},+inf)`), and `plot_partial_dependence`
  returns one value per bin (`_calc_partial_dependence`, `core.py:4041`). Since a
  prediction depends only on which bin the feature lands in, the bin-`i`
  representative is any interior point; we use
  `[b_0 - 1, (b_0+b_1)/2, …, (b_{k-2}+b_{k-1})/2, b_{k-1} + 1]` (length `k+1`,
  strictly ascending). Feeding `grid[i]` through the PDP-01 engine reproduces
  upstream bin `i` (verified <1e-15 on numeric_tiny). There is **no upstream
  numeric x-grid** to compare against (upstream's x-axis is bin indices with
  interval tick text), so PDP-02 is a **unit** test on the deterministic
  transform, while the *values* it feeds are oracle-locked by PDP-03/04.
  `[VERIFIED: LOCAL catboost 1.2.10 core.py:4033-4055; empirical dev <1e-15]`
- **Behavior (Given/When/Then):**
  - **Given** feature `f` with borders `b_0 < … < b_{k-1}`, **then** the grid is
    the `k+1` per-bin representatives above (ascending).
  - **Given** a feature the model never split on (empty borders), **then** the
    grid is the benign single point `[0.0]` (the feature does not affect
    prediction; upstream rejects such a feature outright — we do not). `[INFERRED]`
- **Invariants:** ascending, finite, length `n_borders + 1`.
- **Acceptance tests (unit):**
  - AT-02: `grid_for_feature` == `[b0-1, midpoints…, b_last+1]` for a 3-border and
    a 1-border feature; `[0.0]` for empty borders; strictly ascending.
- **Out of scope:** averaging; two features.
- **Traceability:** `[VERIFIED: CODEGRAPH model.rs:293]`,
  `[VERIFIED: LOCAL catboost==1.2.10 core.py:4041 `_calc_partial_dependence` per-bin]`.

---

### PDP-03 — Single-feature partial dependence (public API, end-to-end)

- **Status:** implemented (2026-07-16; oracle `single_feature_pdp_matches_upstream` green ≤1e-5)
- **Responsibility:** compose PDP-02 grid + PDP-01 engine behind the public
  `partial_dependence(model, columns, &[f])` and return `PartialDependence`.
- **Preconditions:** valid float feature; non-empty rectangular `columns`.
- **Input:** `model`, `columns: &[Vec<f32>]`, `features: &[usize]` of length 1.
- **Output:** `Ok(PartialDependence { features: [f], grids: [grid], values })`,
  `values.len() == grid.len()`.
- **Dependencies:** PDP-01, PDP-02.
- **Behavior:** **Given** the fixture model + `numeric_tiny` columns and the
  chosen feature `f`, **when** `partial_dependence(model, cols, &[f])` runs,
  **then** `values` matches upstream `plot_partial_dependence`'s y-data for `f`
  within `1e-5`, and `grids[0]` matches PDP-02.
- **Acceptance tests (oracle):**
  - AT-03a: `values` == `partial_dependence/pdp_single_values.npy` (≤1e-5 via `assert_abs_close`).
  - AT-03b: `grids[0]` == `pdp_single_grid.npy`; `features == [f]`.
- **Out of scope:** two features; validation.
- **Traceability:** `[VERIFIED: CONTEXT7 plot_partial_dependence single feature]`.

---

### PDP-04 — Two-feature partial dependence (2-D grid)

- **Status:** implemented (2026-07-16; oracle `pair_feature_pdp_matches_upstream` green ≤1e-5)
- **Responsibility:** partial dependence over the Cartesian product of two
  float features' grids.
- **Preconditions:** two distinct valid float feature indices; non-empty columns.
- **Input:** `model`, `columns`, `features: &[usize]` of length 2 `[f1, f2]`.
- **Output:** `Ok(PartialDependence { features: [f1,f2], grids: [g1,g2], values })`,
  `values.len() == g1.len()*g2.len()`, row-major (`f1` outer, `f2` inner).
- **Dependencies:** PDP-02 (per feature), PDP-01 generalized to override two columns.
- **Behavior:** **Given** the fixture model and features `[f1,f2]`, **when**
  computed, **then** for each `(a,b)` the value equals the mean over objects of
  `predict_raw` with column `f1` pinned to `g1[a]` and column `f2` to `g2[b]`,
  and the flattened row-major `values` matches upstream's 2-D pdp within `1e-5`.
- **Acceptance tests (oracle):**
  - AT-04a: `values` == `partial_dependence/pdp_pair_values.npy` (≤1e-5).
  - AT-04b: `grids == [pdp_pair_grid0.npy, pdp_pair_grid1.npy]`; row-major order
    verified by a shape/stride assertion.
- **Out of scope:** validation; >2 features.
- **Traceability:** `[VERIFIED: CONTEXT7 plot_partial_dependence two features → 2-D heatmap]`.

---

### PDP-05 — Typed rejection of unsupported inputs

- **Status:** implemented (2026-07-16; unit AT-05a..e green in `partial_dependence_test.rs`)
- **Responsibility:** reject invalid requests with a typed `PdpError`, never
  panic/unwrap (workspace denies `unwrap_used`/`panic`/`indexing_slicing`).
  Every arm is reachable and implementable from `Model` + `columns` alone.
- **Input / Output:** as `partial_dependence(...) -> Result<_, PdpError>`.
- **Dependencies:** `Model` metadata only (`float_feature_borders.len()` = `n_float`).
- **Deterministic check order (must be honored so each Red test isolates one arm):**
  1. **arity** — `features.len() ∉ {1, 2}` → `UnsupportedFeatureArity { requested }`.
  2. **column shape** — `columns.len() != n_float` (this includes the empty
     `columns == []` case, since `0 != n_float` for the in-scope `n_float >= 1`),
     or columns are ragged (unequal lengths) → `MalformedColumns { expected_float_features: n_float, actual }`.
  3. **empty dataset** — shape is correct (`columns.len() == n_float`) but every
     column has length 0 → `EmptyDataset`. (Ordered after shape so a width/ragged
     mismatch is `MalformedColumns` and only a correctly-shaped, zero-row dataset
     is `EmptyDataset`. `columns == []` is therefore `MalformedColumns`, NOT
     `EmptyDataset` — resolves the prior AT-05b/c overlap.)
  4. **feature range** — any `features[k] >= n_float` → `FeatureIndexOutOfRange
     { index: features[k], n_float }` (first offending index, in request order).
  5. **duplicate (2-feature only)** — `features == [f, f]` → `DuplicateFeature { index: f }`.
- **Behavior (Given/When/Then), each independently testable:**
  - **Given** `features.len() == 0` or `== 3`, **then** `Err(UnsupportedFeatureArity { requested })`.
  - **Given** `columns == []`, `columns.len() != n_float`, or ragged columns,
    **then** `Err(MalformedColumns { expected_float_features: n_float, .. })`.
  - **Given** `columns.len() == n_float` with every column length 0, **then**
    `Err(EmptyDataset)`.
  - **Given** valid rectangular non-empty columns (`columns.len() == n_float`,
    length `n >= 1`) and `features = [i]` with `i >= n_float`, **then**
    `Err(FeatureIndexOutOfRange { index: i, n_float })`.
  - **Given** valid rectangular non-empty columns and `features = [f, f]` (both in
    range), **then** `Err(DuplicateFeature { index: f })`.
- **Invariants:** no partial computation on any error path; the message names the
  offending input; a valid request never returns a `PdpError`.
- **Acceptance tests (unit):** AT-05a (arity), AT-05b (malformed columns —
  includes `columns == []`, wrong width, and ragged), AT-05c (empty dataset:
  `n_float` columns each length 0), AT-05d (out-of-range), AT-05e (duplicate) —
  one per arm; all buildable against a numeric-only model (no categorical model
  needed). **AT-05d/AT-05e must pass valid rectangular non-empty `columns`** so
  checks 1–3 succeed and the range/duplicate check is the one under test.
- **Out of scope:** the happy-path math (PDP-01/03/04); categorical/flat-index
  support (deferred, see §4 index-space note).
- **Traceability:** `[VERIFIED: LOCAL Cargo.toml:10-14 restriction lints]`,
  `[VERIFIED: CODEGRAPH crates/cb-model/src/apply.rs:404-407 NaN-pad → MalformedColumns guard]`,
  `[VERIFIED: CODEGRAPH crates/cb-model/src/model.rs:272-313 no feature-kind map → no UnsupportedFeatureKind arm]`,
  `[INFERRED: DuplicateFeature is a conservative API precondition, not asserted upstream behavior]`.

## 6. Acceptance scenarios (roll-up)

| Scenario | Spec | Kind | Oracle artifact | Bar |
|----------|------|------|-----------------|-----|
| Engine averages a constant grid == direct mean | PDP-01 | unit | — | exact |
| Per-bin grid == `[b0-1, midpoints…, b_last+1]` | PDP-02 | unit | — (no upstream numeric x-grid) | exact |
| Single-feature curve == upstream | PDP-03 | oracle | `pdp_single_values.npy` | ≤1e-5 |
| Two-feature surface == upstream | PDP-04 | oracle | `pdp_pair_values.npy` | ≤1e-5 |
| Invalid inputs → typed errors (arity / malformed columns / empty / out-of-range / duplicate) | PDP-05 | unit | — | typed `Err` (5 arms) |

## 7. Impact scope

- **Classification:** `local` (single crate `cb-model`). `[VERIFIED: CODEGRAPH deps]`
- **New symbols:** `partial_dependence`, `PartialDependence`, `PdpError`, plus a
  new module (`cb-model/src/partial_dependence.rs`) + test file
  (`partial_dependence_test.rs`) + oracle test
  (`crates/cb-model/tests/partial_dependence_oracle_test.rs`).
- **Modified:** `crates/cb-model/src/lib.rs` (add `mod partial_dependence;` and a
  `pub use`) `[VERIFIED: LOCAL crates/cb-model/src/lib.rs:14-22]`.
- **New oracle fixture dir:** `crates/cb-oracle/fixtures/partial_dependence/`
  (`gen_fixtures.py`, `config.json`, `.npy`, model `.cbm`/`.json`).
- **Callers/consumers affected:** none existing (additive public API). Facade
  (`catboost-rs`) / Python (`catboost-rs-py`) surfacing is a **later** DX task,
  not this slice. `[INFERRED]`
- **Tests:** new unit + oracle only; no shipped fixture is touched → no
  bit-exact re-baseline risk. `[VERIFIED: greenfield, no existing pdp symbol]`
- **Build/operational:** none (CPU host path; validates under
  `cargo test -p cb-model -p cb-oracle`, avoiding the env-red cb-backend/py suites).

## 8. Compatibility and migration

Additive only — new public items in `cb-model`. No serialization format, no
existing signature, no fixture changes. No migration needed. `[INFERRED]`

## 9. Risks and open questions

1. **[RESOLVED] Exact grid transform (PDP-02).** Upstream computes PD **per BIN**
   (`n_borders+1` bins), NOT per border value. The grid is the per-bin
   representatives `[b0-1, midpoints…, b_last+1]` (see PDP-02). Confirmed by
   reading `catboost==1.2.10` `core.py:4033-4055` and reproducing `all_predictions`
   to <1e-15 via the engine. `[VERIFIED: LOCAL core.py:4041; empirical]`
2. **[RESOLVED] How the generator extracts truth from `plot_partial_dependence`.**
   `plot_partial_dependence(data, features, plot=False)` **returns
   `(all_predictions, fig)`** — `all_predictions` IS the oracle array
   (`_calc_partial_dependence`); **no figure parsing needed**. `plot=False` also
   avoids the notebook-only render path. The generator dumps `all_predictions`
   verbatim (1-D) / C-order-flattened (2-D). No hand-averaging.
   `[VERIFIED: LOCAL catboost 1.2.10 core.py:4041,4055]`
3. **[RESOLVED] Feature-index space.** `partial_dependence` indexes the **float**
   feature space (`0..model.float_feature_borders.len()`), because `cb_model::Model`
   exposes no flat-feature/kind map `[VERIFIED: CODEGRAPH crates/cb-model/src/model.rs:272-313]`.
   The numeric-only fixture model makes float-index == upstream flat-index, so the
   internal→regular remap (an FSTR-01 concern) never arises and categorical PD is a
   deferred follow-up (see §4 index-space note). Owner: T3.
4. **[INFERRED] `catboost` not installed in this env** and vendored snapshot
   lacks the utility → fixtures are generated **offline** in a pinned venv and
   committed; CI/dev consume committed `.npy` (same pattern as every fixture
   family). `[VERIFIED: LOCAL advanced_fstr/gen_fixtures.py:1-21]`
5. **[VERIFIED] Silent NaN-pad hazard closed by PDP-05.** `predict_raw` reads a
   missing/short column as `NaN` `[VERIFIED: CODEGRAPH crates/cb-model/src/apply.rs:404-407]`;
   the `MalformedColumns` guard (PDP-05 check 2) requires `columns.len() == n_float`
   and rectangular columns before any apply, so the PD average is never computed
   over silently-padded garbage.
6. **[INFERRED] Grid `f64 → f32` cast.** The engine overrides the target column
   with `grid[k] as f32` (the column storage `predict_raw` already uses), so the
   cast introduces **no** precision hazard beyond what upstream apply itself has;
   exact parity is still adjudicated by the ≤1e-5 oracle, not by reasoning about
   the cast. `[VERIFIED: CODEGRAPH crates/cb-model/src/apply.rs:370 f32 columns]`

## 10. Traceability and sources

- **Requirement:** `[VERIFIED: LOCAL .planning/REQUIREMENTS.md:31]`,
  `[VERIFIED: LOCAL .planning/ROADMAP.md:126-131]`.
- **Upstream behavior:** `[VERIFIED: CONTEXT7 /catboost/catboost plot_partial_dependence
  (averaging formula, 1-/2-feature, grid from float-split borders)]`.
- **Rust seams:** `[VERIFIED: CODEGRAPH crates/cb-model/src/apply.rs:370 predict_raw]`,
  `[VERIFIED: CODEGRAPH crates/cb-model/src/model.rs:293 float_feature_borders]`,
  `[VERIFIED: CODEGRAPH crates/cb-model/src/fstr.rs:123,288,490 existing fstr entry points]`,
  `[VERIFIED: LOCAL crates/cb-model/src/lib.rs:14-22 module decls]`.
- **Oracle harness:** `[VERIFIED: CODEGRAPH crates/cb-oracle/src/compare.rs:46,84]`,
  `[VERIFIED: LOCAL crates/cb-oracle/fixtures/advanced_fstr/{config.json,gen_fixtures.py}]`.
- **Constraints:** `[VERIFIED: LOCAL Cargo.toml:10-14]`, `[VERIFIED: LOCAL CLAUDE.md
  source/test separation + no-unwrap]`.
- **Absence proof:** `[VERIFIED: LOCAL grep partial_dependence → 0 hits in crates/ and catboost-master/]`.
- **PageIndex:** this SPEC **is already indexed** — `[VERIFIED: PAGEINDEX
  catboost_rs/SPEC.md (folder id cmrhcxbtm000104jr3i5jzm0m, status=completed,
  indexed 2026-07-12)]` — correcting the earlier draft's claim that no corpus
  applied. **Pending PageIndex update (this hardened revision):** the MCP's
  `process_document` ingests PDFs/files as *new* documents (no `doc_id` overwrite
  / in-place Markdown upsert), so re-processing would create a **duplicate**; the
  human owner should re-index `SPEC.md` into folder `catboost_rs` (replacing the
  2026-07-12 doc) out-of-band. No duplicate was created and the existing doc was
  NOT removed by the planner. `[VERIFIED: TOOL process_document schema — url/file
  ingestion only]`
