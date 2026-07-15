---
title: "FSTR-03 — Partial-Dependence Feature Importance (one or two features)"
status: draft
format: markdown
spec_version: 1
updated_at: 2026-07-12T00:00:00Z
phase: 18
requirement_ids:
  - FSTR-03
source_requirements:
  - ".planning/REQUIREMENTS.md:31 (FSTR-03)"
  - ".planning/ROADMAP.md:131 (Phase 18 Success Criterion 3)"
pageindex_target: "PENDING — no writable PageIndex spec corpus is indexed for this repo (see §10). This file is the local authoritative draft."
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
| Float-feature grid source | `Model.float_feature_borders: Vec<Vec<f64>>` (per-feature ascending borders) | `[VERIFIED: CODEGRAPH crates/cb-model/src/model.rs:293]` |
| Deterministic mean fold | `cb_core::sum_f64(&[f64]) -> f64` (upstream iteration order, D-08) | `[VERIFIED: LOCAL crates/cb-model/src/fstr.rs:49,55]` |
| Oracle comparator | `cb_oracle::compare::assert_abs_close(expected, actual, 1e-5)` / `compare_stage(stage, …)` | `[VERIFIED: CODEGRAPH crates/cb-oracle/src/compare.rs:46,84]` |
| Fixture recipe | upstream `catboost==1.2.10` in a venv, `.npy` + `config.json` committed under `crates/cb-oracle/fixtures/<name>/` | `[VERIFIED: LOCAL crates/cb-oracle/fixtures/advanced_fstr/gen_fixtures.py:1-70]` |

**Layering:** all work lives in `cb-model` (already depends on `cb-core`); no
`cb-train`/`cb-backend`/`cb-compute` edge is added, so no CubeCL feature-unification
risk `[VERIFIED: LOCAL .planning/research/unimplemented-parity-research.md:47,182]`.

## 4. Typed contracts

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
#[derive(Debug, thiserror::Error)]
pub enum PdpError {
    #[error("feature index {index} out of range (model has {n_float} float features)")]
    FeatureIndexOutOfRange { index: usize, n_float: usize },
    #[error("feature {index} is not a float feature; partial dependence is float-only in this release")]
    UnsupportedFeatureKind { index: usize },
    #[error("partial dependence supports 1 or 2 features, got {requested}")]
    UnsupportedFeatureArity { requested: usize },
    #[error("dataset has no objects")]
    EmptyDataset,
}

/// Public entry point (composes PDP-02 grid + PDP-01 engine).
/// `columns[f]` is float feature `f`'s per-object `f32` column (SoA, as `predict_raw`).
pub fn partial_dependence(
    model: &Model,
    columns: &[Vec<f32>],
    features: &[usize],
) -> Result<PartialDependence, PdpError>;
```

> Exact placement of `PdpError` (a dedicated enum in the new module vs. new
> `cb_model::ModelError` variants) is a plan-time wiring choice — see `PLAN.md`
> Task 0. It does not change the behavioral contract. `[INFERRED]`

## 5. Failure-isolated behavioral specifications

Each specification below has one behavioral responsibility, one trigger, an
explicit dependency boundary, and one primary cause of acceptance-test failure.

---

### PDP-01 — Single-feature averaging engine over an explicit grid

- **Status:** draft
- **Responsibility:** given a grid, compute the averaged RawFormulaVal for each
  grid point. *Isolates the averaging math from grid derivation.*
- **Preconditions:** `columns` non-empty and rectangular; `feature < columns.len()`;
  `feature` is a float feature.
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

### PDP-02 — Default grid derivation from model borders (single float feature)

- **Status:** draft
- **Responsibility:** produce the x-axis grid for a float feature, **matching
  upstream's grid**. *Isolates the one genuine open convention question.*
- **Preconditions:** `feature` is a valid float feature index.
- **Input:** `model: &Model`, `feature: usize`.
- **Output:** `Vec<f64>` ascending grid.
- **Dependencies:** `Model.float_feature_borders[feature]` (model.rs:293).
- **Behavior (Given/When/Then):**
  - **Given** feature `f` with stored borders `b_0 < … < b_{k-1}`, **when** the
    grid is derived, **then** it equals the upstream `plot_partial_dependence`
    x-grid for `f` on the same model. The **exact transform** (borders verbatim
    vs. consecutive midpoints vs. borders-plus-endpoints) is **TBD** — resolved
    empirically in TDD by comparing to the committed `pdp_single_grid.npy`
    (resolution owner: fixture generator run under pinned `catboost==1.2.10`).
    `[UNVERIFIED: exact grid transform not derivable from vendored source or docs]`
  - **Given** a feature the model never split on (empty borders), **then** the
    grid is empty and `values` is empty (no crash). `[INFERRED]`
- **Invariants:** ascending, deduplicated, finite.
- **Acceptance tests (oracle):**
  - AT-02a: derived grid equals `partial_dependence/pdp_single_grid.npy` element-for-element
    (≤1e-5; grid values are borders, so exact or near-exact).
- **Out of scope:** averaging; two features.
- **Traceability:** `[VERIFIED: CODEGRAPH model.rs:293]`,
  `[VERIFIED: CONTEXT7 grid derives from float-split borders]`,
  `[UNVERIFIED: precise transform → oracle-resolved]`.

---

### PDP-03 — Single-feature partial dependence (public API, end-to-end)

- **Status:** draft
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

- **Status:** draft
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

- **Status:** draft
- **Responsibility:** reject invalid requests with a typed `PdpError`, never
  panic/unwrap (workspace denies `unwrap_used`/`panic`/`indexing_slicing`).
- **Input / Output:** as `partial_dependence(...) -> Result<_, PdpError>`.
- **Dependencies:** none beyond `Model` metadata (`float_feature_borders.len()`).
- **Behavior (Given/When/Then), each independently testable:**
  - **Given** `features = [i]` with `i >= n_float`, **then** `Err(FeatureIndexOutOfRange { index: i, n_float })`.
  - **Given** a target feature that is categorical / CTR-only (not a float
    feature), **then** `Err(UnsupportedFeatureKind { index })`.
  - **Given** `features.len() == 0` or `> 2`, **then** `Err(UnsupportedFeatureArity { requested })`.
  - **Given** `columns` empty or a zero-length first column, **then** `Err(EmptyDataset)`.
- **Invariants:** no partial computation on the error paths; message names the offending input.
- **Acceptance tests (unit):** AT-05a..d, one per arm above.
- **Out of scope:** the happy-path math (PDP-01/03/04).
- **Traceability:** `[VERIFIED: LOCAL Cargo.toml:10-14 restriction lints]`,
  `[INFERRED: float-only guard mirrors upstream export rejection]`.

## 6. Acceptance scenarios (roll-up)

| Scenario | Spec | Kind | Oracle artifact | Bar |
|----------|------|------|-----------------|-----|
| Engine averages a constant grid == direct mean | PDP-01 | unit | — | exact |
| Derived grid == upstream x-grid | PDP-02 | oracle | `pdp_single_grid.npy` | ≤1e-5 |
| Single-feature curve == upstream | PDP-03 | oracle | `pdp_single_values.npy` | ≤1e-5 |
| Two-feature surface == upstream | PDP-04 | oracle | `pdp_pair_values.npy` | ≤1e-5 |
| Invalid inputs → typed errors | PDP-05 | unit | — | typed `Err` |

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

1. **[UNVERIFIED] Exact grid transform (PDP-02).** Borders-verbatim vs.
   midpoints vs. borders+endpoints is not derivable from vendored source or
   docs. *Resolution:* the fixture dumps upstream's x-grid; the PDP-02 RED test
   compares against it and GREEN adjusts the transform until it matches. Owner:
   fixture-generation task (T3) + PDP-02 task (T4).
2. **[UNVERIFIED] How the generator extracts truth from `plot_partial_dependence`.**
   The upstream call returns a figure, not arrays. *Resolution:* extract grid +
   values from the returned figure object's data (e.g. `fig.data[...] .x/.y` for
   the plotly backend) under pinned `catboost==1.2.10`; confirm the exact
   attribute path when authoring `gen_fixtures.py`. Owner: T3. Do **not** compute
   the truth with our own averaging loop — the oracle must be upstream's output.
3. **[INFERRED] Feature-index space.** `partial_dependence` indexes the **float**
   feature space (as `float_feature_borders` / `predict_raw` columns do). Choose
   a numeric-only fixture model so float-index == flat-index and the open
   internal→regular remap (an FSTR-01 concern) never arises. Owner: T3.
4. **[INFERRED] `catboost` not installed in this env** and vendored snapshot
   lacks the utility → fixtures are generated **offline** in a pinned venv and
   committed; CI/dev consume committed `.npy` (same pattern as every fixture
   family). `[VERIFIED: LOCAL advanced_fstr/gen_fixtures.py:1-21]`

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
- **PageIndex:** no writable indexed spec corpus applies to this in-repo code
  question (the MCP corpus is document/PDF-oriented, not the `.planning/` tree);
  this draft is the local authoritative artifact. **Pending PageIndex update:**
  none required unless the team indexes `.planning/phases/**` — if so, upsert this
  file as document id `phase-18/fstr-03-partial-dependence` with `status: draft`.
