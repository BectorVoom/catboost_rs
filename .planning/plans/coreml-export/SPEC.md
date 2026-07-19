---
title: EXPORT-02 — CoreML export (float-only oblivious, first slice)
status: draft
format: markdown
spec_version: 1
updated_at: 2026-07-18T00:00:00Z
source_requirements:
  - "User: Implement features of CatBoost that have not yet been implemented in catboost_rs."
  - ".planning/plans/next-feature-research/research.md §2 Candidate 4"
treefinder_pending:
  collection: UNRESOLVED
  document_id: UNRESOLVED
  note: "TreeFinder MCP unavailable; local SPEC is authoritative draft. CoreML .proto schema + upstream export layout INFERRED (sparse checkout)."
---

# EXPORT-02 — CoreML export

## 1. Context

CatBoost can export a model to Apple CoreML `.mlmodel` (protobuf). catboost_rs already ships a
float-only oblivious **ONNX** exporter with the exact structural pattern to mirror
`[VERIFIED: CODEGRAPH crates/cb-model/src/export/onnx.rs:333 build_regressor_node, :544 export_onnx]`:
build per-tree node fragments, encode a prost `Message`, write bytes. CoreML export is **missing**
— only a `mod.rs` placeholder, no `coreml.rs` `[VERIFIED: LOCAL grep -i coreml crates/ → empty]`.
This completes Phase 17 (EXPORT-02).

The ONNX exporter guards to float-only, oblivious, scalar models (`is_onnx_exportable`)
`[VERIFIED: CODEGRAPH crates/cb-model/src/export/onnx.rs:99 is_onnx_exportable`], iterates
`model.oblivious_trees`, and emits `base_values` only when `model.bias != 0.0`
`[VERIFIED: CODEGRAPH onnx.rs:333-372]`. CoreML uses its own protobuf schema
(`TreeEnsembleRegressor` / `Model.proto`), encoded via the existing `prost = 0.14.4`
`[VERIFIED: LOCAL crates/cb-model/Cargo.toml prost 0.14.4]`, with a committed generated module
mirroring `src/generated/onnx_generated.rs`.

**Verification caveat (accepted):** no Apple CoreML runtime exists on this Linux host, so parity is
a **structural round-trip** (encode → re-decode the emitted protobuf → assert the tree
structure/leaf values/thresholds match the source model), NOT the ≤1e-5 numeric bar. This mirrors
the ROADMAP's "export uses an export-specific tolerance" framing `[VERIFIED: LOCAL research §2 C4]`.

## 2. Scope and non-goals

**In scope:** a `crates/cb-model/src/export/coreml.rs` exporter for **float-only, oblivious,
scalar** models (same guard as ONNX); a committed CoreML protobuf schema (minimal subset:
`Model`, `TreeEnsembleRegressor`, nodes/thresholds/leaf values, bias as `base_values`); a facade
`Model::save_coreml`; optional Python `save_coreml`; a structural round-trip test + a golden-bytes
regression test.

**Non-goals (explicit):** classifier CoreML output (regressor first); categorical / CTR / one-hot
models; non-oblivious (Lossguide/Depthwise/Region) models; multi-dimension (multiclass) output;
numeric ≤1e-5 parity via an Apple runtime (no runtime on host); any new external crate beyond the
already-present `prost`.

## 3. Dependencies

- `cb_model::export::onnx::{is_onnx_exportable-equivalent guard, build_tree_nodes fragments}` —
  reuse the guard logic and per-tree fragment shape `[VERIFIED: CODEGRAPH onnx.rs:99,305-373]`.
- `cb_model::Model` fields `oblivious_trees`, `bias`, `float_feature_borders`, `approx_dimension`
  `[VERIFIED: CODEGRAPH model.rs:271-313]`.
- `prost::Message::encode` (already used by ONNX) `[VERIFIED: CODEGRAPH onnx.rs:628]`.
- A committed generated CoreML schema module (build-time codegen OR hand-committed prost structs,
  mirroring `src/generated/onnx_generated.rs`) `[VERIFIED: LOCAL crates/cb-model/src/generated/onnx_generated.rs]`.
- `cb_model::export::OnnxExportError` analogue → new `CoreMlExportError` (or a shared
  `ExportError`) `[VERIFIED: CODEGRAPH onnx.rs:57 OnnxExportError]`.
- Facade needs a NEW `CatBoostError::CoreMlExport(#[from] CoreMlExportError)` variant — the
  existing `CatBoostError::Export` is ONNX-typed and cannot be reused (see §4 + PLAN-CHECK A/B)
  `[VERIFIED: CODEGRAPH crates/catboost-rs/src/error.rs:84]`.

**Note on "scale":** the canonical `Model` has NO model-level scale field (`pub scale` at
model.rs:62 is a CTR-split field, unrelated) `[VERIFIED: CODEGRAPH model.rs:271-313; PLAN-CHECK
coreml MINOR D]`. Scale is therefore NOT a guardable condition — it is silently assumed 1.0, same
as the ONNX exporter. Do not spec an "identity-scale" rejection: there is nothing to test.

## 4. Typed contracts

```rust
// crates/cb-model/src/export/coreml.rs  (NEW FILE)

/// Export a float-only, oblivious, scalar REGRESSOR model to a CoreML `.mlmodel`
/// protobuf at `path`.
/// # Errors
/// `CoreMlExportError::Unsupported` if the model is categorical/CTR, non-oblivious, or multi-dim
/// (scale is NOT checkable — no model-level scale field, assumed 1.0);
/// `CoreMlExportError::Encode` / `::Io` on a downstream failure. Never panics.
pub fn export_coreml(model: &Model, path: &Path) -> Result<(), CoreMlExportError>;
```

Facade:
```rust
// crates/catboost-rs/src/model.rs
/// Export to CoreML (EXPORT-02): float-only oblivious regressor models only.
/// # Errors
/// [`CatBoostError::CoreMlExport`] on an unsupported model or a downstream encode/I/O failure.
pub fn save_coreml(&self, path: &Path) -> Result<(), CatBoostError>;
```

**Error-variant wiring (verified against PLAN-CHECK CRITICAL A/B).** `CatBoostError::Export`
is `Export(#[from] cb_model::OnnxExportError)` — hard-typed to ONNX, so it **cannot** carry a
CoreML error `[VERIFIED: CODEGRAPH crates/catboost-rs/src/error.rs:84]`. A **new**
`CatBoostError::CoreMlExport(#[from] cb_model::CoreMlExportError)` variant is required. Because
`CatBoostError` is **not** `#[non_exhaustive]` and `catboost-rs-py::to_pyerr` matches it
**exhaustively with no wildcard** `[VERIFIED: CODEGRAPH crates/catboost-rs-py/src/errors.rs:114-135]`,
adding the variant is NOT purely additive: `to_pyerr` MUST gain a matching arm (mirroring the ONNX
inner match) or the `catboost-rs-py` crate fails to compile (E0004). Additionally, the new
`export_coreml` / `CoreMlExportError` symbols MUST be re-exported at the `cb_model` crate root
(`crates/cb-model/src/lib.rs` `pub use export::{…}`, currently ONNX-only) and at
`crates/catboost-rs/src/lib.rs`, or the facade cannot name them
`[VERIFIED: CODEGRAPH crates/cb-model/src/lib.rs:37]`.

## 5. Failure-isolated behavioral specifications

### CM-01 — Exportability guard → typed error
- **Responsibility:** reject non-float-only / non-oblivious / multi-dim models (scale is NOT
  checked — no model-level scale field; assumed 1.0, same as ONNX).
- **Input:** any `&Model`.
- **Output:** `Ok` only for a float-only oblivious scalar model; else
  `Err(CoreMlExportError::Unsupported(_))`.
- **Given/When/Then:** Given a CTR model (`ctr_data.is_some()`) or a non-symmetric model; When
  `export_coreml`; Then a typed unsupported error (no file written).
- **Acceptance:** `coreml_rejects_unsupported` in `coreml_test.rs`.
- **Out of scope:** node encoding (CM-02).

### CM-02 — Tree → CoreML node encoding
- **Responsibility:** each oblivious tree becomes CoreML `TreeEnsembleRegressor` nodes with correct
  feature indices, `branch_on_value_greater_than` thresholds, and leaf `evaluation_value`s; bias as
  `base_values` only when `bias != 0.0` (mirroring ONNX's conditional).
- **Input:** float-only oblivious `&Model`.
- **Output:** a `Model` protobuf whose decoded node arrays reproduce the source trees' structure and leaf values.
- **Given/When/Then:** Given a known 2-tree model; When exported and re-decoded; Then node
  thresholds == split borders, leaf values == `leaf_values`, tree count == `oblivious_trees.len()`.
- **Acceptance:** `coreml_nodes_match_source` (round-trip decode).
- **Out of scope:** file I/O + facade (CM-03).

### CM-03 — Encode + write + facade wiring
- **Responsibility:** encode to bytes, write to `path`; expose `save_coreml` mapping `CoreMlExportError → CatBoostError::Export`.
- **Input:** a supported `&Model`, a path.
- **Output:** a `.mlmodel` file that re-decodes to a valid `Model` proto; facade returns `Ok(())`.
- **Given/When/Then:** Given a supported facade model; When `save_coreml(path)`; Then the file exists,
  re-decodes, and its tree structure matches (CM-02). Given a CTR facade model; Then `Err(Export)`.
- **Acceptance:** facade test `save_coreml_roundtrip` + `save_coreml_rejects_ctr`.

### CM-04 — Golden-bytes regression [structural pin]
- **Responsibility:** pin the exact emitted bytes for a fixed tiny model so encoding drift is caught.
- **Input:** a frozen tiny float `.cbm`.
- **Output:** emitted `.mlmodel` bytes equal a committed golden file (or a decoded-field golden JSON).
- **Given/When/Then:** Given the frozen model; When exported; Then bytes == golden (regenerated
  deliberately, never silently).
- **Acceptance:** `crates/cb-model/tests/coreml_export_test.rs` (or sibling) with a committed golden.
- **Note:** this is the closest available "oracle" absent an Apple runtime; see §9 R1.

## 6. Acceptance scenarios

1. Float-only 3-tree regressor → `.mlmodel` re-decodes; nodes/leaves match source (CM-02/CM-03).
2. Zero-bias model → no `base_values`; non-zero bias → `base_values=[bias]` (CM-02).
3. CTR / non-symmetric / multiclass model → typed unsupported error (CM-01/CM-03).
4. Golden bytes stable across runs (CM-04).

## 7. Impact scope

- **local:** new `crates/cb-model/src/export/coreml.rs`, new committed
  `src/generated/coreml_generated.rs` (+ optional `.proto` + build.rs codegen), new
  `CoreMlExportError`, `export/mod.rs` re-export. No edit to `onnx.rs` internals beyond possibly
  sharing the tree-fragment builder (keep ONNX byte-identical). No overlap with in-flight
  `fstr.rs`/`tree.rs` `[VERIFIED: LOCAL git status]`.
- **cross-module:** facade `save_coreml` in `crates/catboost-rs/src/model.rs`; optional Python method.
- **external:** true numeric validation would need Apple's CoreML runtime — OUT of scope; structural
  round-trip + golden bytes only.
- **tests:** unit (mounted sibling), round-trip/golden integration, a frozen tiny fixture.
- **build:** if `.proto` codegen is used, `build.rs` gains a prost-build step; prefer committing the
  generated module (as ONNX does) to avoid a `protoc` host dependency — decide at plan time.

## 8. Compatibility and migration

Mostly additive (new module + facade method + new error type + generated schema module; ONNX export
untouched) with **two required edits to existing exhaustive surfaces** (PLAN-CHECK A/B):
1. `crates/cb-model/src/lib.rs` `pub use export::{…}` must add `export_coreml, CoreMlExportError`
   (and `crates/catboost-rs/src/lib.rs` likewise), or the symbols stay private and the facade
   cannot name them.
2. Adding `CatBoostError::CoreMlExport` forces a new arm in the **exhaustive, non-wildcard**
   `catboost-rs-py::to_pyerr` match (`crates/catboost-rs-py/src/errors.rs:114-135`), else the py
   crate fails to compile (E0004). No `.cbm`/json wire change; no ONNX behavior change.

## 9. Risks and open questions

- **R1 (no numeric oracle):** the ≤1e-5 bar is not achievable on this host (no Apple runtime). This
  is the WEAKEST-verified candidate; CM-04 golden bytes + CM-02 structural round-trip are the
  substitute. Flag explicitly to the user — this feature ships without true predict-parity here.
- **R2 (schema source):** the exact CoreML `Model.proto` / `TreeEnsembleRegressor` field layout and
  the `default_value` / `missing_value_tracks_right_child` semantics are `[UNVERIFIED — sparse
  checkout, no local coremltools]`. Must be sourced from Apple's coremltools `Model.proto` at plan
  time; getting a field wrong yields a structurally-valid but semantically-wrong `.mlmodel` that the
  host cannot detect. HIGH-attention item.
- **R3 (codegen vs commit):** committing generated prost structs avoids a `protoc` build dependency
  but must be regenerated deliberately on schema change. Prefer commit (ONNX precedent).
- **Q1:** classifier CoreML output — deferred; first slice is regressor only.

## 10. Traceability and sources

- Research: `.planning/plans/next-feature-research/research.md` §2 Candidate 4, §3 (prost 0.14.4), §6.
- CodeGraph: `cb-model/src/export/onnx.rs:{57 OnnxExportError, 99 is_onnx_exportable, 305-373
  fragments/build_regressor_node, 544-631 export_onnx}`, `cb-model/src/generated/onnx_generated.rs`,
  `catboost-rs/src/model.rs:271 save_onnx`.
- Local: `crates/cb-model/Cargo.toml` (prost 0.14.4), `crates/cb-model/src/export/mod.rs`.
- External (to fetch at plan time): Apple coremltools `mlmodel` protobuf (`Model.proto`,
  `TreeEnsemble.proto`) — `[UNVERIFIED — not local]`.
