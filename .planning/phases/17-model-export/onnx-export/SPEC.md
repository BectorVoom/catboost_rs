---
title: "EXPORT-01 — ONNX Export for Float-Only Oblivious Models"
status: draft
format: markdown
spec_version: 1
updated_at: 2026-07-17T00:00:00Z
phase: 17
requirement_ids:
  - EXPORT-01
source_requirements:
  - ".planning/ROADMAP.md:108-120 (Phase 17) — git-recovered (commit a82289c); NOT in the working tree (deleted in commit 4cfd88c). Confirm the canonical revision before flipping the requirement checkbox."
  - ".planning/REQUIREMENTS.md@a82289c (EXPORT-01/02/03) — git-recovered, same commit."
pageindex_target: "catboost_rs (PageIndex folder id cmrhcxbtm000104jr3i5jzm0m). No document indexed yet for Phase 17 — this SPEC is PENDING initial PageIndex ingestion (see §10)."
---

# EXPORT-01 — ONNX Export for Float-Only Oblivious Models

> **Draft.** Not approved / not implemented. This spec decomposes EXPORT-01 (the
> first slice of Phase 17: Model Export) into failure-isolated behavioral
> specifications for TDD (see `PLAN.md`). No production code is authored by this
> document.

> **Plan-checker status (see `PLAN-CHECK.md` for full detail):** 3 checker
> passes run (this planning process's cap). Pass 1 found 1 CRITICAL + 1
> MAJOR + 2 MINOR; pass 2 confirmed all 4 fixed but surfaced 1 new MAJOR + 2
> new MINOR; pass 3 confirmed those 3 fixed but surfaced 1 new CRITICAL (the
> `cb-model` dev-dependency gap affecting T6's Python wiring). Because the
> pass cap was reached, that final CRITICAL fix (PLAN.md T6-0,
> `crates/catboost-rs-py/Cargo.toml`'s dependency promotion) was applied
> directly to these artifacts WITHOUT a 4th independent checker pass. **T0
> through T5 (the `cb-model`-only ONNX guard/graph-builder/serializer) are
> fully checker-verified. T6/T7 (facade + Python surfacing) carry one
> checker-identified, directly-applied-but-not-independently-re-verified
> fix and should get a human/checker second look at implementation time,
> specifically around `crates/catboost-rs-py/Cargo.toml` and the WR-03
> feature-unification rationale for the promoted dependency.**

## 1. Context

Phase 18 (Extended Feature Importance — Interaction, LossFunctionChange,
Partial Dependence) is now fully implemented in `cb-model/src/fstr.rs` and
`cb-model/src/partial_dependence.rs`
`[VERIFIED: CODEGRAPH crates/cb-model/src/fstr.rs:61-72,288,490; crates/cb-model/src/partial_dependence.rs]`,
confirming Phase 17 (Model Export — ONNX + CoreML) is the correct next
not-started phase in the recovered roadmap's execution order
(15→16→17→18→19→20→21→21.5→22; 15/16/18/21/21.5 done, 17/19/20/22 not started)
`[VERIFIED: LOCAL git show a82289c:.planning/ROADMAP.md]`. There is currently
**zero** ONNX or CoreML code anywhere in the workspace — no `cb-export` crate,
no `onnx`/`coreml`/`prost`/`protobuf` reference in any `Cargo.toml` or
`Cargo.lock` `[VERIFIED: LOCAL grep -rli "onnx|coreml" crates/, grep -rn
"prost|protobuf" crates/*/Cargo.toml Cargo.lock → 0 hits]`.

**Research.** A dedicated research pass fetched and read upstream's live ONNX
exporter source (`catboost/libs/model/model_export/{onnx_helpers.cpp,.h,
model_exporter.cpp}`, `scale_and_bias.h`) and mapped it against this port's
current `cb_model::Model` data model; the full findings are recorded at
`.planning/phases/17-model-export/onnx-export/research.md` and are cited
throughout this spec as `[VERIFIED: WEB ...]` / `[CODEGRAPH ...]`. This
research's scope was itself narrowed to EXPORT-01 only (CoreML/EXPORT-02 and
the numeric oracle harness/EXPORT-03 were explicitly deferred there); this SPEC
inherits that same narrowing.

**Upstream definition (the target to match).** A user can export a
float-only, oblivious, identity-scale trained model to ONNX via the
`ai.onnx.ml` `TreeEnsembleRegressor` / `TreeEnsembleClassifier` (+`ZipMap`)
operators, with unsupported models (categorical, CTR, text, embedding,
non-oblivious) rejected by a typed error mirroring upstream's own export guard
`[VERIFIED: LOCAL git show a82289c:.planning/ROADMAP.md Phase 17 Success
Criterion 1]`.

**Scope-defining decisions locked before this spec was written** (each was a
genuine judgment call the research flagged as blocking the API shape; resolved
by explicit user sign-off, not silently defaulted):

1. **Classifier signal — explicit parameter.** `cb_model::Model` carries no
   loss-function/objective metadata, so a 1-dimensional model is ambiguously
   either a regressor or a binary classifier
   `[VERIFIED: CODEGRAPH crates/cb-model/src/model.rs:266-313 — no loss/objective
   field on Model]`. Resolved: the export function takes an explicit
   `is_classifier: bool` supplied by the caller (who DOES know — the facade/
   PyO3 wrapper type). **(User decision, this session.)**
2. **Crate placement — `cb-model` submodule.** `crates/cb-model/src/export/
   onnx.rs`, NOT a new `cb-export` crate — following every existing precedent
   for a new read-only `Model` surface (`fstr.rs`, `shap.rs`,
   `partial_dependence.rs`, `cbm.rs`, `json.rs`)
   `[VERIFIED: LOCAL crates/cb-model/src/lib.rs:14-23]`. The roadmap's cited
   "STACK.md vs ARCHITECTURE.md disagreement" could not be corroborated
   against current file content or git history
   `[VERIFIED: LOCAL .planning/codebase/{STACK,ARCHITECTURE}.md content;
   git log --all -- .planning/codebase/STACK.md .planning/codebase/ARCHITECTURE.md
   → one unrelated commit]`. **(User decision, this session.)**
3. **Facade + Python wiring — included in this slice.** Unlike FSTR-03 (whose
   `cb-model`-only plan was split from a separate facade+Python plan), this
   slice adds `catboost-rs::Model::save_onnx` AND PyO3-exposed `save_onnx`
   methods on `CatBoostRegressor`/`CatBoostClassifier` in the SAME TDD cycle
   set. **(User decision, this session.)** Note: `catboost-rs-py` currently
   exposes **no** save/export method of any kind (`save_cbm`/`save_json` are
   Rust-facade-only) `[VERIFIED: LOCAL grep -rn "save_cbm|save_json|fn save"
   crates/catboost-rs-py/src/*.rs → 0 hits]` — this spec's Python surface is
   therefore the FIRST model-saving method exposed to Python, not a mirror of
   an existing one.
4. **Return shape — path-based.** `export_onnx(&Model, &Path, is_classifier:
   bool) -> Result<(), OnnxExportError>`, mirroring `save_cbm`/`save_json`'s
   existing `(&self, path: &Path) -> Result<(), CatBoostError>` shape exactly
   `[VERIFIED: CODEGRAPH crates/cb-model/src/cbm.rs save_cbm; crates/catboost-rs/src/model.rs:224-227]`.
   **(User decision, this session.)**

## 2. Scope and non-goals

### In scope (this slice)

- Export a float-only, oblivious, identity-scale `cb_model::Model` to a
  well-formed ONNX file via `ai.onnx.ml` `TreeEnsembleRegressor` (regression /
  `is_classifier=false`) or `TreeEnsembleClassifier`+`ZipMap` (binary
  classification / `is_classifier=true`, `approx_dimension == 1`).
- Multiclass classifier export (`approx_dimension > 1`) — structurally
  unambiguous (dimension alone selects `TreeEnsembleClassifier`), included as
  a natural consequence of the graph-builder's dimension handling, but NOT
  independently oracle-tested in this slice (no multiclass ONNX fixture is in
  scope; flagged as a fast-follow).
- Typed rejection (no panic) of: any model containing a `ModelSplit::Ctr`
  split, any model with `ctr_data.is_some()`, any model with a non-empty
  `non_symmetric_trees` or `region_trees` (i.e. anything that is not
  purely-oblivious).
- Pinned `ir_version=3`, `ai.onnx.ml` opset `2` (matching upstream exactly).
- `cb-model` submodule (`export/onnx.rs`), `catboost-rs` facade method
  (`Model::save_onnx`), and `catboost-rs-py` PyO3 methods
  (`CatBoostRegressor::save_onnx`, `CatBoostClassifier::save_onnx`).
- Structural unit tests (hand-built small trees; no upstream numeric oracle
  needed for this slice — see §9 Risk 1 on why numeric oracle is deferred).

### Non-goals

- **CoreML export (EXPORT-02)** — a wholly separate exporter/schema; not
  investigated beyond confirming its upstream source file paths exist
  `[VERIFIED: WEB raw.githubusercontent.com/.../coreml_helpers.h,.cpp — HTTP 200]`.
- **Numeric oracle validation against ONNX Runtime (EXPORT-03)** — deferred;
  `onnxruntime` is not yet in `crates/cb-oracle/generator/requirements.txt`
  `[VERIFIED: LOCAL crates/cb-oracle/generator/requirements.txt]`. This slice's
  acceptance bar is **structural** correctness (graph shape, node counts,
  attribute values match what the transcribed algorithm computes,
  independently verified by hand-computed expected values on small
  hand-built trees — see PDP-03-style precedent in FSTR-03's PDP-02 unit
  test), not "matches official CatBoost's own ONNX export to a numeric
  tolerance." That numeric comparison is EXPORT-03's job, in a later plan.
- **CatBoostRanker `save_onnx`** — `crates/catboost-rs-py/src/ranker.rs` is
  out of scope for this slice unless the plan finds it trivially reachable via
  the same `EstimatorBase`; not assumed here (ranker models have
  `approx_dimension == 1` RawFormulaVal semantics identical to a regressor,
  so if included it follows the `is_classifier=false` regressor path with no
  new graph-builder logic — a plan-time scoping call, not a spec blocker).
- **Non-identity `Scale`** — this port's `Model` has no `Scale` concept at all
  (only `bias: f64`, upstream's `Bias` with `Scale` permanently `1.0`)
  `[VERIFIED: CODEGRAPH crates/cb-model/src/model.rs:266-313 — no Scale field]`,
  so the "identity-scale" guard is implemented defensively (for forward
  compatibility) but is vacuously true for every `Model` constructible today.
- **Text / embedding feature detection** — `cb_model::Model` has no
  text/embedding feature representation at all
  `[VERIFIED: CODEGRAPH crates/cb-model/src/model.rs — no text/embedding field]`;
  a model built from a Pool with text/embedding features simply cannot be
  round-tripped through this port's canonical `Model` at all, so there is
  nothing for the ONNX guard to additionally reject beyond CTR/categorical
  (this SIMPLIFIES the guard vs. upstream's four separate checks — see §4).

### Open or Conflicting Requirements

None outstanding — all four blocking open questions the research raised were
resolved by explicit user decision (§1, items 1–4) before this spec was
written.

## 3. Dependencies

| Dependency | Typed interface | Evidence |
|---|---|---|
| Canonical model | `cb_model::Model { oblivious_trees, non_symmetric_trees, region_trees, bias, float_feature_borders, ctr_data: Option<CtrData>, approx_dimension, class_to_label }` | `[VERIFIED: CODEGRAPH crates/cb-model/src/model.rs:266-313]` |
| Split representation | `ModelSplit::{Float(Split{feature,border}), Ctr(CtrSplit)}` per tree, `ObliviousTree{splits, leaf_values, leaf_weights}` | `[VERIFIED: CODEGRAPH crates/cb-model/src/model.rs:65-98,253-264]` |
| Binarize semantics (source of truth for `BRANCH_GT`) | `binarize_feature(raw: f64, borders: &[f64]) -> usize` — bin = count of borders `b` with `raw > b` (strict) | `[VERIFIED: CODEGRAPH crates/cb-model/src/apply.rs:49-51]` |
| Leaf-index bit order (source of truth for the ONNX depth-reversal) | forward bit order: `splits[i]` contributes bit `i` of the leaf index (lowest-index split closest to the leaves) | `[VERIFIED: CODEGRAPH crates/cb-model/src/apply.rs:14-15 doc comment; crates/cb-model/src/partial_dependence.rs:18-29 "Float-only scope (index space)"]` |
| Upstream ONNX tree-walk (source of truth for the graph shape) | `AddTree`/`ConvertTreeToOnnxGraph`: ONNX depth `d` node = `tree.splits[len-1-d]` (REVERSED); complete-binary-tree node indexing `2*i+1`/`2*i+2`; `BRANCH_GT`; `base_values` populated only when `!IsZeroBias()` | `[VERIFIED: WEB raw.githubusercontent.com/catboost/catboost/master/catboost/libs/model/model_export/onnx_helpers.cpp]` |
| Upstream export guard ordering (source of truth for typed-error precondition order) | `ExportModel`/`SerializeFullModelToOnnxStream`: `CB_ENSURE_SCALE_IDENTITY`, `HasCategoricalFeatures`, `HasTextFeatures`, `HasEmbeddingFeatures`, `IsOblivious` guards, checked before any graph is built | `[VERIFIED: WEB raw.githubusercontent.com/catboost/catboost/master/catboost/libs/model/model_export/model_exporter.cpp]` |
| Identity-scale definition | `TScaleAndBias::IsIdentity()` = `Scale==1.0 && IsZeroBias()`; `CB_ENSURE_SCALE_IDENTITY` checks `Scale==1.0` ONLY (bias is separately handled via `base_values`, not part of the identity-scale guard) | `[VERIFIED: WEB raw.githubusercontent.com/catboost/catboost/master/catboost/libs/model/scale_and_bias.h]` |
| ONNX domain constant | `AI_ONNX_ML_DOMAIN = "ai.onnx.ml"` | `[VERIFIED: WEB raw.githubusercontent.com/catboost/catboost/master/contrib/libs/onnx/onnx/common/constants.h]` |
| Protobuf encode | `prost::Message::encode(&self, buf) -> Result<(), EncodeError>` on a hand-built, `prost`-derived `onnx::ModelProto` value, generated once from the vendored `.proto` and committed (mirrors `flatbuffers`' committed-`flatc`-output convention) | `[VERIFIED: LOCAL crates/cb-model/src/lib.rs:51-89 "committed files under src/generated/ are unmodified flatc output"]`, `[VERIFIED: CRATES.IO api/v1/crates/prost — 0.14.4, tokio-rs org]` |
| Typed error pattern | `#[derive(Debug, thiserror::Error)] pub enum OnnxExportError { ... }` mirroring `PdpError`/`ModelError` | `[VERIFIED: CODEGRAPH crates/cb-model/src/partial_dependence.rs:75-107; crates/cb-model/src/error.rs:16-52]` |
| Facade error wiring | `catboost_rs::CatBoostError::Export(#[from] cb_model::OnnxExportError)`, same `#[from]` pattern as `PartialDependence(#[from] cb_model::PdpError)` | `[VERIFIED: CODEGRAPH crates/catboost-rs/src/error.rs:72-78]` |
| Facade save-to-path pattern | `pub fn save_cbm(&self, path: &Path) -> Result<(), CatBoostError> { save_cbm(&self.inner, path)?; Ok(()) }` | `[VERIFIED: CODEGRAPH crates/catboost-rs/src/model.rs:219-227]` |
| PyO3 estimator pattern | `self.base.model.as_ref().ok_or_else(not_fitted_err)?`, `py.detach(\|\| ...)` around the Rust call, `.map_err(PyCbError)` | `[VERIFIED: CODEGRAPH crates/catboost-rs-py/src/regressor.rs:98-114; crates/catboost-rs-py/src/classifier.rs:89-105]` |

**Layering:** all new production code lives in `cb-model` (submodule) →
`catboost-rs` (facade method) → `catboost-rs-py` (PyO3 methods); no
`cb-train`/`cb-backend`/`cb-compute` edge is touched, so the CubeCL
feature-unification landmine (`never add a cb-train dependency to
cb-backend`) does not apply here `[VERIFIED: LOCAL research.md §Constraints]`.

## 4. Typed contracts

### Guard predicate — what makes a `Model` "float-only, oblivious"

`cb_model::Model`'s split representation has exactly two variants
(`ModelSplit::Float`, `ModelSplit::Ctr`) and no distinct one-hot-categorical
or text/embedding representation at all
`[VERIFIED: CODEGRAPH crates/cb-model/src/model.rs:70-98]`. This SIMPLIFIES
the EXPORT-01 guard relative to upstream's four separate checks
(`HasCategoricalFeatures`/`HasTextFeatures`/`HasEmbeddingFeatures`/
non-oblivious): in this codebase, "float-only AND oblivious" reduces to a
single structural predicate reachable from `Model` alone:

```rust
fn is_onnx_exportable(model: &Model) -> bool {
    model.non_symmetric_trees.is_empty()
        && model.region_trees.is_empty()
        && model.ctr_data.is_none()
        && model
            .oblivious_trees
            .iter()
            .all(|t| t.splits.iter().all(|s| matches!(s, ModelSplit::Float(_))))
}
```

Per **Do Not Hand-Roll** (research.md §"Do Not Hand-Roll"), this predicate
(or a `Model::is_float_only_oblivious(&self) -> bool` method hoisted onto
`model.rs` for reuse by a future CoreML guard) is the SINGLE chokepoint; no
second CTR/categorical detector may be written elsewhere in this slice.

### Public API

```rust
/// Typed failure at the ONNX-export boundary (no panic, no unwrap, no raw
/// indexing — workspace-denied restriction lints).
#[derive(Debug, thiserror::Error)]
pub enum OnnxExportError {
    /// The model contains at least one CTR split, or carries baked `ctr_data`
    /// — upstream's `HasCategoricalFeatures`-equivalent guard for this port's
    /// data model (a CTR split is the ONLY categorical-derived construct
    /// `Model` can represent).
    #[error("model uses categorical/CTR features, which ONNX export does not support")]
    CategoricalFeaturesUnsupported,

    /// The model has at least one non-symmetric (Lossguide/Depthwise) tree —
    /// upstream's `IsOblivious()` guard.
    #[error("model contains non-symmetric (Lossguide/Depthwise) trees, which ONNX export does not support")]
    NonObliviousTreesUnsupported,

    /// The model has at least one region-path tree — upstream's
    /// `IsOblivious()` guard (Region trees are a separate, non-oblivious
    /// variant in this port's `TreeVariant`).
    #[error("model contains region-path trees, which ONNX export does not support")]
    RegionTreesUnsupported,

    /// Failed to encode the built ONNX graph to protobuf bytes.
    #[error("ONNX protobuf encode error: {0}")]
    Encode(#[from] prost::EncodeError),

    /// Underlying I/O error while writing the `.onnx` file.
    #[error("ONNX export I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Export `model` to a well-formed ONNX file at `path`.
///
/// `is_classifier` selects `TreeEnsembleClassifier`+`ZipMap`
/// (`post_transform="LOGISTIC"` for `approx_dimension==1`, `"SOFTMAX"` for
/// `approx_dimension>1`) when `true`, or `TreeEnsembleRegressor`
/// (`post_transform="NONE"`) when `false`. The caller supplies this because
/// `Model` carries no loss-function/objective metadata to infer it from
/// (§1 decision 1).
///
/// # Errors
/// [`OnnxExportError::CategoricalFeaturesUnsupported`] /
/// [`OnnxExportError::NonObliviousTreesUnsupported`] /
/// [`OnnxExportError::RegionTreesUnsupported`] if the guard rejects `model`;
/// [`OnnxExportError::Encode`] / [`OnnxExportError::Io`] on a downstream
/// failure. Never panics.
pub fn export_onnx(model: &Model, path: &Path, is_classifier: bool) -> Result<(), OnnxExportError>;
```

Facade (`catboost-rs`):

```rust
impl Model {
    /// Export to ONNX (EXPORT-01). `is_classifier` selects
    /// `TreeEnsembleClassifier`+`ZipMap` vs `TreeEnsembleRegressor` — see
    /// `cb_model::export_onnx`.
    ///
    /// # Errors
    /// [`CatBoostError::Export`] on an unsupported model (categorical/CTR,
    /// non-oblivious) or a downstream encode/I/O failure.
    pub fn save_onnx(&self, path: &Path, is_classifier: bool) -> Result<(), CatBoostError> {
        cb_model::export_onnx(&self.inner, path, is_classifier)?;
        Ok(())
    }
}
```

New `CatBoostError` arm:

```rust
/// An [`crate::Model::save_onnx`] export failed — an unsupported model
/// (categorical/CTR, non-oblivious) or a downstream encode/I/O error. Carries
/// the typed `cb-model` [`cb_model::OnnxExportError`].
#[error("ONNX export error: {0}")]
Export(#[from] cb_model::OnnxExportError),
```

PyO3 (`catboost-rs-py`), one method per estimator type, each hardcoding its
own `is_classifier` (resolving §1 decision 1 at the PyO3 layer with NO
Python-facing parameter, since the wrapper TYPE already knows):

```rust
// CatBoostRegressor
fn save_onnx(&self, py: Python<'_>, path: &str) -> PyResult<()> {
    let model = self.base.model.as_ref().ok_or_else(|| not_fitted_err(py, "...save_onnx"))?;
    py.detach(|| model.save_onnx(Path::new(path), /* is_classifier = */ false))
        .map_err(PyCbError)?;
    Ok(())
}

// CatBoostClassifier
fn save_onnx(&self, py: Python<'_>, path: &str) -> PyResult<()> {
    let model = self.base.model.as_ref().ok_or_else(|| not_fitted_err(py, "...save_onnx"))?;
    py.detach(|| model.save_onnx(Path::new(path), /* is_classifier = */ true))
        .map_err(PyCbError)?;
    Ok(())
}
```

> Exact placement of `OnnxExportError` variant ordering / message wording is a
> plan-time wiring choice; it does not change the behavioral contract.
> `[INFERRED]`

## 5. Failure-isolated behavioral specifications

Each specification below has one behavioral responsibility, one trigger, an
explicit dependency boundary, and one primary cause of acceptance-test
failure.

---

### EXPORT-01a — Guard: typed rejection of unsupported models

- **Responsibility:** reject a model this exporter cannot represent, BEFORE
  any graph-building code runs, with the specific typed error. *Isolates the
  precondition check from the graph-building algorithm.*
- **Input:** `model: &Model`.
- **Output:** `Result<(), OnnxExportError>` (via the internal guard function;
  publicly observed through `export_onnx`'s early return).
- **Dependencies:** `Model.ctr_data`, `Model.non_symmetric_trees`,
  `Model.region_trees`, `ModelSplit` variant matching. No I/O, no protobuf.
- **Deterministic check order** (mirrors upstream's guard-before-build
  ordering `[VERIFIED: WEB model_exporter.cpp]`; each order slot must be
  independently testable):
  1. **non-oblivious (non-symmetric)** — `!model.non_symmetric_trees.is_empty()`
     → `Err(NonObliviousTreesUnsupported)`.
  2. **non-oblivious (region)** — `!model.region_trees.is_empty()` →
     `Err(RegionTreesUnsupported)`.
  3. **categorical/CTR** — `model.ctr_data.is_some()` OR any
     `ObliviousTree.splits` contains a `ModelSplit::Ctr` → `Err(CategoricalFeaturesUnsupported)`.
  4. Otherwise `Ok(())`.
- **Behavior (Given/When/Then):**
  - **Given** a model with `non_symmetric_trees` non-empty, **then**
    `Err(NonObliviousTreesUnsupported)`, regardless of what `region_trees` or
    `ctr_data` also contain (order slot 1 wins).
  - **Given** a model with `region_trees` non-empty and `non_symmetric_trees`
    empty, **then** `Err(RegionTreesUnsupported)`.
  - **Given** an all-oblivious model containing at least one
    `ModelSplit::Ctr` split (in any tree), **then**
    `Err(CategoricalFeaturesUnsupported)`.
  - **Given** an all-oblivious, all-`ModelSplit::Float` model with
    `ctr_data.is_some()` (baked tables present even though no live split
    references them — a defensive belt-and-suspenders case), **then**
    `Err(CategoricalFeaturesUnsupported)`.
  - **Given** an all-oblivious, all-float model with `ctr_data: None`,
    **then** `Ok(())`.
- **Invariants / side effects:** pure; no partial file is ever written on a
  guard failure (the guard runs to completion strictly before
  `export_onnx` opens/creates the output path).
- **Acceptance tests (unit, hand-built `Model` values — no oracle needed):**
  - AT-01a-1: non-symmetric-tree model → `Err(NonObliviousTreesUnsupported)`.
  - AT-01a-2: region-tree model → `Err(RegionTreesUnsupported)`.
  - AT-01a-3: oblivious model with a `ModelSplit::Ctr` split → `Err(CategoricalFeaturesUnsupported)`.
  - AT-01a-4: oblivious all-float model with `ctr_data: Some(..)` but no CTR
    split present → `Err(CategoricalFeaturesUnsupported)`.
  - AT-01a-5: oblivious all-float model, `ctr_data: None` → `Ok(())`.
  - AT-01a-6 (order): a model with BOTH `non_symmetric_trees` non-empty AND a
    CTR split present → `Err(NonObliviousTreesUnsupported)` (slot 1, not
    slot 3) — proves the deterministic order, not just "is an error."
- **Out of scope:** the graph-building algorithm (EXPORT-01b/c/d); I/O
  failures (EXPORT-01e).
- **Traceability:** `[VERIFIED: WEB model_exporter.cpp guard ordering]`,
  `[VERIFIED: CODEGRAPH crates/cb-model/src/model.rs:266-313]`.

---

### EXPORT-01b — Oblivious tree → ONNX node arrays (single tree, structural)

- **Responsibility:** transcribe ONE `ObliviousTree` into the `ai.onnx.ml`
  tree-ensemble node-array fragment (`nodes_treeids`, `nodes_nodeids`,
  `nodes_featureids`, `nodes_modes`, `nodes_values`, `nodes_truenodeids`,
  `nodes_falsenodeids`, plus the flat leaf-value contribution for this tree).
  *Isolates the reversed-depth-walk + complete-binary-tree-indexing algorithm
  from the whole-model assembly (EXPORT-01c) and from serialization
  (EXPORT-01e).*
- **Preconditions:** the tree has already passed EXPORT-01a (every split is
  `ModelSplit::Float`).
- **Input:** `tree: &ObliviousTree`, `tree_id: i64` (this tree's index within
  the ensemble).
- **Output:** the per-tree node-array fragment (a plain Rust struct/tuple
  bundling the seven parallel arrays above — exact struct name is a
  plan-time choice).
- **Dependencies:** none beyond the tree's own `splits`/`leaf_values` (no
  `Model`-level state needed for a single tree).
- **Behavior (Given/When/Then)** — transcribing
  `[VERIFIED: WEB onnx_helpers.cpp AddTree]`, cross-checked against this
  port's own documented forward-bit-order convention
  `[VERIFIED: CODEGRAPH crates/cb-model/src/apply.rs:14-15]`:
  - **Given** a tree of depth `k` (`splits.len() == k`, `leaf_values.len()
    == 2^k`), **when** building the ONNX depth-`d` internal node (`d` in
    `0..k`), **then** its split is `tree.splits[k - 1 - d]` (REVERSED index
    order: ONNX root = the LAST element of `splits`, matching this port's
    own convention that the LOWEST-index split is evaluated closest to the
    leaves and must therefore map to the DEEPEST ONNX level).
  - **Given** internal node index `i` in the standard complete-binary-tree
    numbering (root `i=0`), **then** its false-child is node `2*i+1` and its
    true-child is `2*i+2`, `nodes_mode = "BRANCH_GT"` (`value > border`,
    matching this port's own `binarize_feature`'s strict `>` with NO
    translation `[VERIFIED: CODEGRAPH crates/cb-model/src/apply.rs:49-51]`),
    `nodes_featureids[i] = split.feature`, `nodes_values[i] = split.border`.
  - **Given** the `2^k` leaf node ids (the final complete-binary-tree level),
    **then** `nodes_modes = "LEAF"` for each, dummy split fields (feature id
    `0`, value `0.0`, true/false child `0`), and the per-leaf CONTRIBUTION
    value is read directly from `tree.leaf_values` IN ORDER — this port's
    `ObliviousTree.leaf_values` is ALREADY canonical forward-bit-order, which
    is the SAME sequential order the ONNX complete-binary-tree leaf walk
    visits, so no reordering/permutation is applied.
  - **Given** a depth-0 tree (`splits.len() == 0`, a single leaf value —
    degenerate but representable), **then** the tree contributes exactly one
    `LEAF` node (node id `0`) and no internal nodes.
- **Invariants:** node-id space is `0..(2^(k+1) - 1)` (a complete binary
  tree with `2^k` leaves has `2^k - 1` internal nodes); every array has the
  same length (total node count for this tree); `nodes_treeids` is `tree_id`
  repeated for every node in this fragment.
- **Acceptance tests (unit, structural — hand-built small trees, hand-computed
  expected arrays, independent of any oracle fixture):**
  - AT-01b-1: a hand-built depth-2 tree with 3 DISTINCT (feature, border)
    splits at 3 distinct positions — assert `nodes_featureids`/`nodes_values`
    per depth level equal the EXPECTED reversed mapping (this is the
    dedicated regression test for the "reversed split-order" pitfall the
    research flagged as HIGH risk: getting it backwards produces a
    structurally-valid-but-numerically-wrong tree, undetectable without this
    test).
  - AT-01b-2: leaf contribution values equal `tree.leaf_values` verbatim, in
    order (no permutation).
  - AT-01b-3: `nodes_mode` is `"BRANCH_GT"` for every internal node and
    `"LEAF"` for every terminal node; child-index arithmetic (`2*i+1`,
    `2*i+2`) verified for a depth-3 tree.
  - AT-01b-4: depth-0 (single-leaf) tree produces exactly one `LEAF` node,
    zero internal nodes.
- **Out of scope:** multi-tree assembly (EXPORT-01c); `base_values`/bias
  (EXPORT-01c); classifier post-processing (EXPORT-01d); serialization
  (EXPORT-01e).
- **Traceability:** `[VERIFIED: WEB onnx_helpers.cpp AddTree]`.

---

### EXPORT-01c — Whole-ensemble regressor graph assembly

- **Responsibility:** compose EXPORT-01b's per-tree fragments across every
  tree in the model into one `TreeEnsembleRegressor` `NodeProto`, with the
  bias correctly gated into (or omitted from) `base_values`. *Isolates
  cross-tree assembly + the bias-gating pitfall from the single-tree
  transcription (EXPORT-01b).*
- **Preconditions:** guard (EXPORT-01a) passed; `is_classifier == false`.
- **Input:** `model: &Model` (guard-passed).
- **Output:** an `ai.onnx.ml` `TreeEnsembleRegressor` node's attribute set
  (all seven `nodes_*` arrays concatenated across trees with correct
  per-tree `tree_id`s, `target_ids`/`target_weights`/`target_nodeids` for the
  leaf contributions, `n_targets=1`, `post_transform="NONE"`, `base_values`).
- **Dependencies:** EXPORT-01b (per tree), `Model.bias`, `Model.oblivious_trees`.
- **Behavior (Given/When/Then)**
  `[VERIFIED: WEB onnx_helpers.cpp ConvertTreeToOnnxGraph, TTreesAttributes]`:
  - **Given** `model.oblivious_trees` of length `T`, **when** the graph is
    assembled, **then** each tree `t` in `0..T` contributes its EXPORT-01b
    fragment with `tree_id = t`, and the concatenated arrays preserve
    boosting order (`Model.oblivious_trees` iteration order).
  - **Given** `model.bias == 0.0`, **then** the `base_values` attribute is
    ABSENT from the emitted node (not present-and-`[0.0]`) — mirrors
    upstream's `TTreesAttributes` only allocating `base_values` when
    `!IsZeroBias()`.
  - **Given** `model.bias != 0.0`, **then** `base_values = [model.bias]`
    (single-target regressor: one bias value).
  - **Given** any model in scope, **then** `n_targets = 1`,
    `post_transform = "NONE"`, `op_type = "TreeEnsembleRegressor"`, domain
    `"ai.onnx.ml"`.
- **Invariants:** array lengths across all seven `nodes_*` fields (plus
  `target_ids`/`target_nodeids`/`target_weights`) are mutually consistent
  (same total node / leaf count derived from summing EXPORT-01b fragments).
- **Acceptance tests (unit, structural):**
  - AT-01c-1: a 2-tree model → assembled `nodes_treeids` contains exactly the
    two tree ids, each tree's node block matches its independently-computed
    EXPORT-01b fragment.
  - AT-01c-2: `bias == 0.0` → `base_values` attribute absent (assert by
    attribute name lookup returning `None`, not by checking a zero value).
  - AT-01c-3: `bias == 2.5` → `base_values == [2.5]`.
  - AT-01c-4: `op_type`/`domain`/`post_transform`/`n_targets` are exactly the
    fixed values above for every model.
- **Out of scope:** classifier assembly (EXPORT-01d); serialization
  (EXPORT-01e).
- **Traceability:** `[VERIFIED: WEB onnx_helpers.cpp ConvertTreeToOnnxGraph + TTreesAttributes constructor (conditional base_values allocation)]`.

---

### EXPORT-01d — Classifier graph assembly (`TreeEnsembleClassifier` + `ZipMap`)

- **Responsibility:** when `is_classifier == true`, compose the same
  per-tree fragments (EXPORT-01b) into a `TreeEnsembleClassifier` node
  (`post_transform` selected by dimension, `classlabels_int64s`, the
  binary-classifier asymmetric-bias trick) followed by a separate `ZipMap`
  node. *Isolates classifier-specific attribute/post-processing logic from
  the shared tree-walk (EXPORT-01b) and the regressor assembly (EXPORT-01c).*
- **Preconditions:** guard (EXPORT-01a) passed; `is_classifier == true`.
- **Input:** `model: &Model` (guard-passed), `is_classifier: true`.
- **Output:** a `TreeEnsembleClassifier` node's attribute set PLUS a
  downstream `ZipMap` node consuming its probability output.
- **Dependencies:** EXPORT-01b (per tree), `Model.approx_dimension`,
  `Model.class_to_label`, `Model.bias`.
- **Behavior (Given/When/Then)**
  `[VERIFIED: WEB onnx_helpers.cpp IsClassifierModel, ConvertTreeToOnnxGraph classifier branch, GetClassLabels]`:
  - **Given** `approx_dimension == 1` (binary), **then**
    `post_transform = "LOGISTIC"`, `classlabels_int64s` is derived from
    `model.class_to_label` (numeric labels only — this port's
    `class_to_label: Vec<f64>` has no string-label representation, so
    `classlabels_strings` is unreachable in this slice), and `base_values`
    (when bias is non-zero) is the upstream binary-classifier
    ASYMMETRIC-bias pair `[-model.bias, +model.bias]` (`class_ids` `0` and
    `1`) — NOT the single-value regressor form.
  - **Given** `approx_dimension > 1` (multiclass), **then**
    `post_transform = "SOFTMAX"`, and the per-class leaf contributions are
    laid out per upstream's multi-dim target-weight convention. **Load-bearing
    indexing note (added after plan-checker review):** upstream's `AddTree`
    walks the flat `leafValue` pointer **leaf-major** — `leaf_values[leaf *
    dim + class]` (class is the inner/fastest-varying index)
    `[VERIFIED: WEB onnx_helpers.cpp:428-479 AddTree]`. This port's
    `ObliviousTree.leaf_values` is **dimension-major** —
    `leaf_values[class * n_leaves + leaf]` (dimension is the OUTER index;
    at `dim==1` the two layouts coincide, which is why the binary path
    needs no remap) `[VERIFIED: CODEGRAPH crates/cb-model/src/model.rs:299-306]`.
    The multiclass per-tree fragment builder MUST therefore read
    `tree.leaf_values[class * n_leaves + leaf]` when emitting the `class_id
    == class` contribution for `leaf` — a naive "iterate `leaf_values` in
    the single-dim order" reuse of EXPORT-01b's leaf transcription would
    silently transpose leaf/class and corrupt every multiclass export
    without any structural symptom (the graph still loads and runs, just
    computes the wrong class scores).
  - **Given** the `TreeEnsembleClassifier` node's `probability_tensor`
    output, **then** a SEPARATE `ZipMap` node (same `ai.onnx.ml` domain)
    consumes it and produces the graph's final `probabilities` output;
    `ZipMap` is never fused into the same node as the tree-ensemble op.
    **Output shape note (added after plan-checker review):** upstream
    declares the `probability_tensor` `ValueInfoProto`'s second dimension as
    `dims==1 ? 2 : dims` — i.e. even a 1-dimensional (binary) model's
    probability tensor is declared width-2 (matching the synthesized
    `class_ids` `[0, 1]`), never width-1
    `[VERIFIED: WEB onnx_helpers.cpp:526-531 ConvertTreeToOnnxGraph]`.
- **Invariants:** `classlabels_int64s.len() == max(2, model.class_to_label.len())`
  for the binary case (`class_to_label` may be empty for a plain binary
  Logloss model — falls back to `[0, 1]`, `[INFERRED]`); the `ZipMap` node's
  input name matches the classifier node's probability output name exactly
  (a dangling/mismatched edge is a structural bug, not a numeric one, and
  must be caught by a unit test, not deferred to EXPORT-03).
- **Acceptance tests (unit, structural):**
  - AT-01d-1: `approx_dimension == 1`, `bias == 1.0` → `base_values ==
    [-1.0, 1.0]` (asymmetric pair, NOT `[1.0]`) — dedicated regression test
    for this pitfall (research §"Common Pitfalls" item on bias handling).
  - AT-01d-2: `approx_dimension == 1` → `post_transform == "LOGISTIC"`.
  - AT-01d-3: `approx_dimension > 1` (a small hand-built 2-tree, 3-class
    model) → `post_transform == "SOFTMAX"`.
  - AT-01d-3b (added after plan-checker review — closes the "post_transform
    passes but values are transposed" gap): for the SAME 3-class hand-built
    model, assert the emitted `class_weights` (or equivalent per-leaf/
    per-class attribute array) values equal
    `tree.leaf_values[class * n_leaves + leaf]` read out via the
    dimension-major formula above — hand-compute the expected array in the
    test from the tree's `leaf_values` directly (not round-tripped through
    the exporter itself), so a leaf/class transposition bug is caught even
    though `post_transform` alone would not catch it.
  - AT-01d-4: the emitted graph contains a `ZipMap` node whose input name ==
    the `TreeEnsembleClassifier` node's probability output name.
  - AT-01d-5: `classlabels_int64s == [0, 1]` when `class_to_label` is empty
    (plain binary Logloss); `== class_to_label` (cast to `i64`) when
    populated.
- **Out of scope:** regressor assembly (EXPORT-01c, mutually exclusive by
  `is_classifier`); serialization (EXPORT-01e).
- **Traceability:** `[VERIFIED: WEB onnx_helpers.cpp classifier branch + TTreesAttributes]`.

---

### EXPORT-01e — Metadata + serialization + file write (public entry point)

- **Responsibility:** wrap EXPORT-01a (guard) → EXPORT-01c/d (graph
  assembly, selected by `is_classifier`) → metadata (`ir_version`,
  `opset_import`) → `prost` encode → file write, behind the public
  `export_onnx(model, path, is_classifier)` entry point.
- **Preconditions:** none beyond what EXPORT-01a checks (this is the
  outermost composition).
- **Input:** `model: &Model`, `path: &Path`, `is_classifier: bool`.
- **Output:** `Result<(), OnnxExportError>`; on success, `path` contains a
  well-formed serialized `onnx::ModelProto`.
- **Dependencies:** EXPORT-01a/c/d, `prost::Message::encode`,
  `std::fs::write` (or `File::create` + `Write`).
- **Behavior (Given/When/Then):**
  - **Given** a model that fails the guard, **then** the typed guard error is
    returned and NO file is created at `path` (not even an empty one).
  - **Given** a guard-passing model, **then** the emitted `ModelProto` has
    `ir_version == 3` and exactly one `opset_import` entry
    `{domain: "ai.onnx.ml", version: 2}`
    `[VERIFIED: WEB onnx_helpers.cpp InitMetadata]`.
  - **Given** a guard-passing model and `is_classifier == false`, **then**
    the graph's sole computation node is the EXPORT-01c
    `TreeEnsembleRegressor`.
  - **Given** a guard-passing model and `is_classifier == true`, **then**
    the graph contains the EXPORT-01d `TreeEnsembleClassifier` node followed
    by its `ZipMap` node.
  - **Given** a successful build, **then** `prost::Message::encode` never
    fails for a well-formed `ModelProto` built by this code path (an
    `Encode` error is only reachable via a buffer-capacity failure, not a
    logic error in this slice) — the error arm exists for defensive
    completeness, not because a reachable failure mode was found.
  - **Given** an unwritable `path` (e.g. a nonexistent parent directory),
    **then** `Err(OnnxExportError::Io(_))`, no panic.
- **Invariants:** the guard ALWAYS runs to completion before ANY byte is
  written to `path` (matches upstream's check-before-build ordering,
  research §"Error, security, and failure behavior").
- **Acceptance tests:**
  - AT-01e-1 (unit): a guard-failing model → `path` is NOT created (assert
    `!path.exists()` after the call, using a tempdir).
  - AT-01e-2 (unit): a guard-passing regressor model → the written file,
    when re-decoded via `prost::Message::decode` back into a `ModelProto`,
    round-trips `ir_version`, `opset_import`, and the `TreeEnsembleRegressor`
    node's attributes exactly as asserted in EXPORT-01c's unit tests (proves
    the serialize/write step is lossless, independent of any oracle).
  - AT-01e-3 (unit): same round-trip check for a guard-passing classifier
    model, asserting the `TreeEnsembleClassifier` + `ZipMap` pair survives
    the encode/decode round trip, AND (added after plan-checker review)
    that the decoded `probability_tensor` output's `ValueInfoProto` second
    dimension is `2` for a binary (`approx_dimension==1`) model — not `1` —
    per the output-shape note above.
  - AT-01e-4 (unit): an unwritable path → `Err(Io(_))`.
- **Out of scope:** numeric prediction-match validation against ONNX Runtime
  (EXPORT-03, deferred per §2).
- **Traceability:** `[VERIFIED: WEB onnx_helpers.cpp InitMetadata]`.

---

### EXPORT-01f — Facade + Python surfacing

- **Responsibility:** expose `export_onnx` through `catboost-rs::Model::save_onnx`
  and PyO3 `save_onnx` methods on `CatBoostRegressor`/`CatBoostClassifier`,
  each supplying its own fixed `is_classifier` value.
- **Preconditions:** EXPORT-01a–e implemented and green.
- **Input (facade):** `&self` (a fitted/loaded `catboost_rs::Model`),
  `path: &Path`, `is_classifier: bool`.
- **Input (PyO3, per estimator):** `&self` (a fitted estimator), `path: &str`
  — NO `is_classifier` parameter (the wrapper type supplies it internally).
- **Output:** `Result<(), CatBoostError>` (facade) / `PyResult<()>` (PyO3).
- **Dependencies:** EXPORT-01e (`cb_model::export_onnx`), the new
  `CatBoostError::Export` arm, the existing `not_fitted_err`/`py.detach`/
  `PyCbError` PyO3 conventions.
- **Behavior (Given/When/Then):**
  - **Given** a facade `Model` wrapping a guard-passing canonical model,
    **when** `.save_onnx(path, is_classifier)` is called, **then** it
    delegates to `cb_model::export_onnx(&self.inner, path, is_classifier)`
    and maps any `OnnxExportError` through `CatBoostError::Export` via `?`.
  - **Given** an UNFITTED `CatBoostRegressor`/`CatBoostClassifier` PyO3
    object, **when** `.save_onnx(path)` is called, **then** a
    `NotFittedError` is raised (mirroring every other estimator method's
    `not_fitted_err` guard) BEFORE any Rust export code runs.
  - **Given** a fitted `CatBoostRegressor`, **when** `.save_onnx(path)` is
    called, **then** it calls `model.save_onnx(Path::new(path), false)`
    under `py.detach`.
  - **Given** a fitted `CatBoostClassifier`, **when** `.save_onnx(path)` is
    called, **then** it calls `model.save_onnx(Path::new(path), true)`
    under `py.detach`.
  - **Given** a guard failure surfacing through either PyO3 method,
    **then** it is mapped through the existing `PyCbError` wrapper (same
    pattern as every other fallible PyO3 method), not a raw string error,
    into the SPECIFIC exception class named below (not an arbitrary/default
    one).
- **Invariants:** `to_pyerr` in `crates/catboost-rs-py/src/errors.rs` is an
  EXHAUSTIVE match over all `CatBoostError` variants with NO wildcard arm
  `[VERIFIED: CODEGRAPH crates/catboost-rs-py/src/errors.rs:104-116 — 6 arms
  covering exactly the 7 current CatBoostError variants]`; adding
  `CatBoostError::Export` as an 8th variant WILL NOT COMPILE until a
  matching arm is added — this is REQUIRED, not conditional (see §9 risk 7).
  **Target exception mapping (added after plan-checker review, pass 2 —
  the mapping must preserve the existing value-error/io-error/internal-error
  taxonomy `to_pyerr`'s doc comment establishes), one arm per
  `OnnxExportError` sub-variant:**
  - `CategoricalFeaturesUnsupported` / `NonObliviousTreesUnsupported` /
    `RegionTreesUnsupported` → `CatBoostValueError` (a bad-input value
    error, exactly like `PartialDependence`'s existing mapping
    `[VERIFIED: CODEGRAPH crates/catboost-rs-py/src/errors.rs:113-115]`
    — the model itself is the "bad input" to the export operation).
  - `Io` → `PyIOError` (mirrors the top-level `CatBoostError::Io` arm's own
    mapping exactly).
  - `Encode` → base `CatBoostError` (an internal/unexpected failure, mirrors
    `Train`/`Model`'s mapping — not user-input-driven).
- **Acceptance tests:**
  - AT-01f-1a (facade unit/integration): `catboost_rs::Model::save_onnx` on a
    guard-passing loaded `.cbm` fixture writes a file that round-trips per
    AT-01e-2.
  - AT-01f-1b (facade unit): **[CORRECTED after plan-checker review, pass
    2 — the original "load a CTR fixture" guidance does not work on this
    branch, and the first-pass fallback ("add a `#[cfg(test)]`-only
    constructor... if not reachable") does not work either.]**
    `crates/cb-model`'s `.cbm`/`model.json` deserializers
    (`reconstruct_model` in `cbm.rs`, `from_doc` in `json.rs`)
    unconditionally set `ctr_data: None` and never construct
    `ModelSplit::Ctr` regardless of the source file's content — CTR-model
    *loading* is separate, not-yet-merged work (`feat/23-ctr-model-loading`)
    — so NO currently-loadable fixture can exercise the CTR-rejection path
    `[VERIFIED: CODEGRAPH crates/cb-model/src/cbm.rs:494-612;
    crates/cb-model/src/json.rs:633-765 — zero ModelSplit::Ctr construction]`.
    This test MUST instead hand-construct a `cb_model::Model` containing a
    literal `ModelSplit::Ctr` split (or `ctr_data: Some(..)`) directly in
    Rust — the exact same technique EXPORT-01a's AT-01a-3/4 already use —
    and wrap it via `Model::from_canonical`
    (`crates/catboost-rs/src/model.rs:38`), which is `pub(crate)`
    `[VERIFIED: CODEGRAPH crates/catboost-rs/src/model.rs:38]`. A
    `pub(crate)` item is reachable from an INTERNAL `#[cfg(test)]` module
    compiled as part of the `catboost-rs` crate itself, but is **NOT**
    reachable from `crates/catboost-rs/tests/` (a separate integration-test
    binary that links the library's normal, non-`cfg(test)` build) —
    confirmed by the existing precedent that
    `crates/catboost-rs/tests/partial_dependence_facade_test.rs` only ever
    constructs a `Model` via the genuinely-`pub` `Model::load_cbm`. **This
    test MUST therefore live as an internal `#[cfg(test)]`-mounted module
    inside the `catboost-rs` crate**, mirroring the existing
    `crates/catboost-rs/src/lib.rs:50-51` `mod error_test;` precedent
    exactly (e.g. a new `mod onnx_test;` mounted the same way) — NOT in
    `crates/catboost-rs/tests/`, and NOT behind a speculative new
    `#[cfg(test)]`-only production constructor (that fallback is deleted;
    it does not solve the `tests/`-crate-boundary problem and is
    unnecessary for the internal-module location). AT-01f-1a (the
    happy-path test, which only needs the already-`pub` `Model::load_cbm`)
    is unaffected and may stay in `crates/catboost-rs/tests/` per the
    existing precedent.
  - AT-01f-2 (PyO3, via the Python test suite): an unfitted
    `CatBoostRegressor().save_onnx(path)` raises `NotFittedError`.
  - AT-01f-3 (PyO3): a fitted `CatBoostRegressor` (numeric-only data)
    `.save_onnx(path)` succeeds and the file exists and is non-empty.
  - AT-01f-4 (PyO3): a fitted `CatBoostClassifier` (numeric-only,
    Logloss-default) `.save_onnx(path)` succeeds; the exported graph (loaded
    back via `prost` in a Rust-side helper, or structurally asserted from
    Python if a lightweight parser is available) contains a
    `TreeEnsembleClassifier`+`ZipMap` pair, not a `TreeEnsembleRegressor`.
  - AT-01f-5 (Rust, `crates/catboost-rs-py/src/errors.rs` unit test — added
    after plan-checker review, pass 2, mirroring that file's existing
    per-variant coverage style): `to_pyerr(&CatBoostError::Export(cb_model::OnnxExportError::CategoricalFeaturesUnsupported))`
    produces a `CatBoostValueError`; the `Io` sub-variant produces a
    `PyIOError`; the `Encode` sub-variant produces the base `CatBoostError`
    — one assertion per sub-variant, per the mapping above.
- **Out of scope:** `CatBoostRanker.save_onnx` (§2 non-goals);
  `catboost-rs-py`'s existing test-suite conventions/tooling are reused
  as-is, not modified.
- **Traceability:** `[VERIFIED: CODEGRAPH crates/catboost-rs/src/model.rs:224-227
  save_cbm pattern; crates/catboost-rs-py/src/{regressor,classifier}.rs
  not_fitted_err/py.detach pattern]`.

## 6. Acceptance scenarios (roll-up)

| Scenario | Spec | Kind | Bar |
|---|---|---|---|
| Non-oblivious (non-symmetric) model rejected | EXPORT-01a | unit | typed `Err`, deterministic order slot 1 |
| Non-oblivious (region) model rejected | EXPORT-01a | unit | typed `Err`, order slot 2 |
| CTR-split / baked-CTR-data model rejected | EXPORT-01a | unit | typed `Err`, order slot 3 |
| Float-only oblivious model passes guard | EXPORT-01a | unit | `Ok(())` |
| Reversed split-order tree walk matches hand-computed expectation | EXPORT-01b | unit | exact |
| Leaf values transcribed verbatim, no permutation | EXPORT-01b | unit | exact |
| `BRANCH_GT` mode + complete-binary-tree child indexing | EXPORT-01b | unit | exact |
| Multi-tree regressor assembly, boosting order preserved | EXPORT-01c | unit | exact |
| Zero-bias → `base_values` ABSENT | EXPORT-01c | unit | exact (attribute presence, not value) |
| Non-zero-bias → `base_values == [bias]` | EXPORT-01c | unit | exact |
| Binary classifier → `base_values == [-bias, +bias]`, `LOGISTIC` | EXPORT-01d | unit | exact |
| Multiclass classifier → `SOFTMAX` | EXPORT-01d | unit | exact |
| `ZipMap` node wired to classifier's probability output | EXPORT-01d | unit | exact |
| `ir_version=3`, `opset_import={ai.onnx.ml, 2}` pinned | EXPORT-01e | unit | exact |
| Guard failure → no file written | EXPORT-01e | unit | exact |
| Encode/decode round trip is lossless (regressor + classifier) | EXPORT-01e | unit | exact |
| Unwritable path → typed I/O error | EXPORT-01e | unit | typed `Err` |
| Facade `save_onnx` delegates + maps errors | EXPORT-01f | unit/integration | exact |
| Unfitted PyO3 estimator → `NotFittedError` | EXPORT-01f | integration (pytest) | exact |
| Fitted `CatBoostRegressor.save_onnx` succeeds, non-empty file | EXPORT-01f | integration (pytest) | exact |
| Fitted `CatBoostClassifier.save_onnx` emits classifier graph | EXPORT-01f | integration (pytest) | exact |
| `Export` sub-variants map to the correct Python exception class | EXPORT-01f | unit (`errors_test.rs`) | exact |

## 7. Impact scope

- **Classification:** `local` (new code confined to `cb-model` → `catboost-rs`
  → `catboost-rs-py`, a straight-line dependency chain already in place; no
  new crate, no `cb-train`/`cb-backend` edge). `[VERIFIED: CODEGRAPH deps]`
- **New symbols:** `export_onnx`, `OnnxExportError`, `is_onnx_exportable`
  (or `Model::is_float_only_oblivious`) in a new `crates/cb-model/src/export/
  onnx.rs` (+ `export/mod.rs` if directory-shaped); `Model::save_onnx` in
  `crates/catboost-rs/src/model.rs`; `CatBoostError::Export` in
  `crates/catboost-rs/src/error.rs`; `save_onnx` PyO3 methods in
  `crates/catboost-rs-py/src/{regressor,classifier}.rs`.
- **New vendored/generated file:** `crates/cb-model/src/generated/onnx_generated.rs`
  (once-generated `prost` bindings for the ONNX proto messages actually used
  — `ModelProto`, `GraphProto`, `NodeProto`, `AttributeProto`,
  `ValueInfoProto`, `TypeProto*`, `OperatorSetIdProto`, `TensorProto`
  element-type enum). Committed, never hand-edited (mirrors the FlatBuffers
  convention `[VERIFIED: LOCAL crates/cb-model/src/lib.rs:51-89]`).
- **Modified:**
  - `crates/cb-model/src/lib.rs` — `mod export;` + `pub use export::{export_onnx, OnnxExportError};`.
  - `crates/cb-model/Cargo.toml` — add `prost` dependency (new; §Dependency Analysis in research.md).
  - `crates/catboost-rs/src/model.rs` — add `save_onnx`.
  - `crates/catboost-rs/src/error.rs` — add `CatBoostError::Export` arm.
  - `crates/catboost-rs-py/src/regressor.rs`, `.../classifier.rs` — add `save_onnx` PyO3 method each.
  - `crates/catboost-rs-py/src/errors.rs` — add the `Export` arm to `to_pyerr`.
  - `crates/catboost-rs-py/Cargo.toml` — **[ADDED after plan-checker
    review, pass 3, applied without a 4th checker pass — see PLAN.md T6-0]**
    promote `cb-model` from `[dev-dependencies]` to `[dependencies]`
    (`default-features = false` preserved), since `to_pyerr`'s new `Export`
    arm names `cb_model::OnnxExportError` from production code, not just
    `errors_test.rs`. Without this, `cargo test -p catboost-rs-py` passes
    (dev-deps are visible to test builds) but the real `cargo build -p
    catboost-rs-py` / `maturin` wheel build fails to compile.
- **May change (plan-time judgment call, not required by this spec):**
  `crates/catboost-rs-py/src/ranker.rs` (see §2 non-goals — a natural
  extension, not required).
- **Verification only:** `crates/cb-model/src/{apply.rs,model.rs}` — read,
  never modified. Existing `cb-model`/`catboost-rs`/`catboost-rs-py` test
  suites must continue passing unchanged (purely additive).
- **Deferred, not this slice's responsibility:** `crates/cb-oracle/generator/requirements.txt`
  needs `onnxruntime` added for EXPORT-03 — explicitly NOT touched here.
- **Tests:** all new; no shipped fixture is touched → no bit-exact
  re-baseline risk anywhere in the existing ≤1e-5 CPU oracle suite.
- **Build/operational:** adds one new runtime dependency (`prost`) to
  `cb-model` and everything depending on it (`catboost-rs`, `catboost-rs-py`);
  no new system/build-time tool is required if the `prost` bindings are
  committed rather than generated at build time (mirrors the `flatbuffers`
  precedent — `protoc`/`prost-build`/`protox` run ONCE, offline, out of the
  `cargo build` graph).

## 8. Compatibility and migration

Additive only — new public items in `cb-model`/`catboost-rs`/`catboost-rs-py`,
one new `CatBoostError` variant (downstream `match`es are documented as
needing to stay robust to new variants,
`[VERIFIED: CODEGRAPH crates/catboost-rs/src/error.rs:27-31]`), one new
runtime dependency (`prost`). No serialization format change to `.cbm`/
`model.json`, no existing signature changed, no fixture changes. No migration
needed. `[INFERRED]`

## 9. Risks and open questions

1. **[ACCEPTED SCOPE LIMIT] No numeric oracle in this slice.** This spec's
   acceptance bar is structural (hand-computed expected node arrays,
   round-trip lossy-free encode/decode), not "matches official CatBoost's
   ONNX export in ONNX Runtime." The research explicitly deferred
   `onnxruntime` wiring to EXPORT-03 since it is not yet a pinned oracle
   dependency `[VERIFIED: LOCAL crates/cb-oracle/generator/requirements.txt]`.
   Risk: a structurally-valid-but-numerically-wrong export (e.g. a
   OFF-BY-ONE in the depth-reversal that still produces a valid tree shape)
   could pass this slice's tests and only be caught later, in EXPORT-03.
   Mitigation: AT-01b-1 is specifically designed as an independent,
   hand-computed structural check of the reversal (not merely "loads
   without erroring"), directly targeting this risk.
2. **[OPEN, non-blocking] Multiclass classifier target-weight layout.**
   EXPORT-01d describes multiclass inclusion at a high level but the exact
   `target_ids`/`target_weights` per-class fan-out for `TreeEnsembleClassifier`
   was not independently re-verified against upstream source with the same
   depth as the binary case in the underlying research (§ "Out of Scope" in
   research.md explicitly flagged multiclass as "structurally straightforward
   ... not exercised by this narrow slice's fixtures"). The plan should
   either (a) budget a short structural read of upstream's multiclass
   `AddTree` branch before implementing AT-01d-3, or (b) explicitly mark
   multiclass support `#[ignore]`/deferred if the reading surfaces
   unexpected complexity. Not a blocker for the binary/regressor path.
3. **[RESOLVED] Classifier signal.** §1 decision 1 — explicit `is_classifier`
   parameter, threaded from the PyO3 wrapper type at the outermost layer.
4. **[RESOLVED] Crate placement.** §1 decision 2 — `cb-model` submodule.
5. **[RESOLVED] Facade/Python inclusion.** §1 decision 3 — included this slice.
6. **[RESOLVED] Return shape.** §1 decision 4 — path-based, mirrors `save_cbm`.
7. **[CONFIRMED — required step, not conditional] `PyCbError` needs an
   explicit new match arm.** Plan-checker review verified `to_pyerr` in
   `crates/catboost-rs-py/src/errors.rs:104-116` is an EXHAUSTIVE `match`
   over all 7 current `CatBoostError` variants with NO wildcard arm
   `[VERIFIED: CODEGRAPH crates/catboost-rs-py/src/errors.rs:88-117]`. Adding
   `CatBoostError::Export` as an 8th variant WILL FAIL TO COMPILE in
   `catboost-rs-py` until a matching arm is added — this is a certain,
   compiler-enforced requirement of EXPORT-01f, not a "verify and maybe fix"
   hedge. The plan MUST add the `Export` arm to `to_pyerr` as part of its
   Green step.
8. **[INFERRED] `prost` vendoring workflow is synthesis-by-analogy.** The
   research rated this MEDIUM confidence: the commit-once-generated-bindings
   pattern is proven for FlatBuffers in this repo, but was not independently
   prototyped for `prost`/ONNX in this research pass. The plan's Task 0
   should treat "generate + commit `onnx_generated.rs`" as its own
   verifiable prerequisite step (confirm the generated file compiles and
   exposes the needed message types) before any guard/graph-builder code is
   written.

## 10. Traceability and sources

- **Requirement:** `[VERIFIED: LOCAL git show a82289c:.planning/ROADMAP.md
  Phase 17]`, `[VERIFIED: LOCAL git show a82289c:.planning/REQUIREMENTS.md
  EXPORT-01]` (both git-recovered; deleted from the working tree in commit
  `4cfd88c`).
- **Research:** `.planning/phases/17-model-export/onnx-export/research.md`
  (this session) — upstream exporter structure, current `cb-model` gap
  analysis, dependency/crate-placement analysis, all Common Pitfalls.
- **Upstream behavior (fetched and read live, not paraphrased from training
  data):**
  `[VERIFIED: WEB https://raw.githubusercontent.com/catboost/catboost/master/catboost/libs/model/model_export/onnx_helpers.cpp]`,
  `[VERIFIED: WEB https://raw.githubusercontent.com/catboost/catboost/master/catboost/libs/model/model_export/onnx_helpers.h]`,
  `[VERIFIED: WEB https://raw.githubusercontent.com/catboost/catboost/master/catboost/libs/model/model_export/model_exporter.cpp]`,
  `[VERIFIED: WEB https://raw.githubusercontent.com/catboost/catboost/master/catboost/libs/model/scale_and_bias.h]`,
  `[VERIFIED: WEB https://raw.githubusercontent.com/catboost/catboost/master/contrib/libs/onnx/onnx/common/constants.h]`.
- **Rust seams:** `[VERIFIED: CODEGRAPH crates/cb-model/src/model.rs:65-313]`,
  `[VERIFIED: CODEGRAPH crates/cb-model/src/apply.rs:14-15,49-51]`,
  `[VERIFIED: CODEGRAPH crates/cb-model/src/partial_dependence.rs:75-107 PdpError precedent]`,
  `[VERIFIED: CODEGRAPH crates/cb-model/src/error.rs:16-52 ModelError precedent]`,
  `[VERIFIED: CODEGRAPH crates/catboost-rs/src/model.rs:219-227 save_cbm precedent]`,
  `[VERIFIED: CODEGRAPH crates/catboost-rs/src/error.rs:27-78]`,
  `[VERIFIED: CODEGRAPH crates/catboost-rs-py/src/{regressor,classifier}.rs]`.
- **Dependency legitimacy:** `[VERIFIED: CRATES.IO api/v1/crates/prost — 0.14.4, tokio-rs org]`
  (research.md §Standard Stack / §Dependency Analysis).
- **Absence proof:** `[VERIFIED: LOCAL grep -rli "onnx|coreml" crates/ → 0 hits;
  grep -rn "prost|protobuf" crates/*/Cargo.toml Cargo.lock → 0 hits;
  grep -rn "save_cbm|save_json|fn save" crates/catboost-rs-py/src/*.rs → 0 hits]`.
- **PageIndex:** no document indexed yet for Phase 17. **Pending PageIndex
  update:** this SPEC should be ingested into the `catboost_rs` folder (id
  `cmrhcxbtm000104jr3i5jzm0m`) as a NEW document (the MCP's `process_document`
  has no in-place Markdown upsert, per the FSTR-03 SPEC's same note) — the
  human owner should do this out-of-band. No PageIndex write was attempted by
  this planning session.
