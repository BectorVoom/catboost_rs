---
title: EXPORT-02 — CoreML export (float-only oblivious regressor) — TDD PLAN
plan_for: .planning/plans/coreml-export/SPEC.md
status: draft
format: markdown
plan_version: 1
updated_at: 2026-07-18T00:00:00Z
spec_ids: [CM-01, CM-02, CM-03, CM-04]
tasks: [CM-R0, CM-01, CM-02, CM-03, CM-04, CM-PY]
no_gsd: true
verification_caveat: "No Apple CoreML runtime on this Linux host → NO ≤1e-5 numeric oracle. Verification is STRUCTURAL round-trip (encode→re-decode→assert) + a GOLDEN-BYTES regression pin. No numeric oracle is invented."
---

# EXPORT-02 — CoreML export: goal-backward TDD plan

**Provenance / GSD:** This plan was authored WITHOUT any GSD skill, command, workflow, or
sub-agent. No production code was written; the sole artifact is this file. All symbols below were
verified via CodeGraph; the CoreML schema field layout was sourced live from Apple's coremltools
`.proto` files (see §Schema Evidence).

**Mirror source (verified):** the CoreML exporter mirrors the shipped ONNX exporter
`crates/cb-model/src/export/onnx.rs` — the exact pattern is: guard the model
(`is_onnx_exportable` :99), build per-tree node fragments (`build_tree_nodes` :178 →
`TreeNodeFragment` :149, `SharedNodeArrays` :294), assemble one ensemble node
(`build_regressor_node` :333), encode a prost `Message` and write bytes (`export_onnx` :544-631).
`base_values` is emitted ONLY when `model.bias != 0.0` (:361). The generated schema is a
COMMITTED prost module (`src/generated/onnx_generated.rs`, wired via `generated_module!` in
`lib.rs:97`), generated out-of-band by `protox` + `prost-build` (NO `protoc` host dependency),
never hand-edited. CoreML follows this precedent exactly.

---

## Schema Evidence (sourced live — resolves SPEC §9 R2, the HIGH-attention risk)

Fetched from `github.com/apple/coremltools` (`mlmodel/format/`) at plan time. These field tags MUST
be re-pinned to an exact coremltools tag/commit in CM-R0 (the header of `coreml_generated.rs` must
record the tag, mirroring `onnx_generated.rs:1-24`). Fetched-from-`main` layout:

`TreeEnsemble.proto`:
```protobuf
message TreeEnsembleRegressor {
    TreeEnsembleParameters treeEnsemble = 1;
    TreeEnsemblePostEvaluationTransform postEvaluationTransform = 2;   // regressor: NoTransform (=0) — CONFIRM enum in CM-R0
}
message TreeEnsembleParameters {
    repeated TreeNode nodes = 1;
    uint64 numPredictionDimensions = 2;        // scalar regressor → 1
    repeated double basePredictionValue = 3;   // ← model.bias goes here (see mapping note)
}
message TreeNode {
    uint64 treeId = 1;
    uint64 nodeId = 2;
    TreeNodeBehavior nodeBehavior = 3;         // internal: BranchOnValueGreaterThan(=3); leaf: LeafNode(=6)
    uint64 branchFeatureIndex = 10;
    double branchFeatureValue = 11;            // ← split border
    uint64 trueChildNodeId = 12;
    uint64 falseChildNodeId = 13;
    bool   missingValueTracksTrueChild = 14;
    repeated EvaluationInfo evaluationInfo = 20;   // leaf value(s)
    double relativeHitRate = 30;
}
enum TreeNodeBehavior {
    BranchOnValueLessThanEqual = 0; BranchOnValueLessThan = 1;
    BranchOnValueGreaterThanEqual = 2; BranchOnValueGreaterThan = 3;
    BranchOnValueEqual = 4; BranchOnValueNotEqual = 5; LeafNode = 6;
}
message EvaluationInfo { uint64 evaluationIndex = 1; double evaluationValue = 2; }
```

`Model.proto`:
```protobuf
message Model {
    int32 specificationVersion = 1;            // CONFIRM min spec version supporting TreeEnsembleRegressor in CM-R0
    ModelDescription description = 2;
    oneof Type { TreeEnsembleRegressor treeEnsembleRegressor = 302; /* ... */ }
}
message ModelDescription {
    repeated FeatureDescription input = 1;
    repeated FeatureDescription output = 10;
    string predictedFeatureName = 11;
    Metadata metadata = 100;
}
message FeatureDescription { string name = 1; string shortDescription = 2; FeatureType type = 3; }
```
`FeatureTypes.proto` (`FeatureType` oneof, esp. `multiArrayType` / `doubleType` and the array
`dataType` enum) — NOT yet fetched; **CM-R0 must fetch and pin it** before writing field mappings.

**CRITICAL MAPPING NOTES (get one wrong → structurally-valid but semantically-wrong `.mlmodel`
the host cannot detect — SPEC §9 R2):**
1. **Branch direction & child sense.** ONNX uses `BRANCH_GT` with `false=left(2i+1)`,
   `true=right(2i+2)` (`build_tree_nodes` :218-223). CoreML's `BranchOnValueGreaterThan` +
   `trueChildNodeId`/`falseChildNodeId` MUST reproduce the SAME routing:
   "feature > branchFeatureValue → trueChild". Verify against CatBoost's own CoreML exporter
   convention in CM-R0; do not assume ONNX and CoreML use the same child ordering.
2. **Bias placement.** ONNX puts bias in the `base_values` attribute; CoreML's analogue is
   `TreeEnsembleParameters.basePredictionValue` (repeated double). Emit `[bias]` ONLY when
   `bias != 0.0` (mirror the ONNX conditional at :361) — do NOT emit `[0.0]`.
3. **Leaf value channel.** CoreML carries the leaf output in `EvaluationInfo{evaluationIndex=0,
   evaluationValue=leaf}` on the LEAF node, NOT in a separate `target_weights` array like ONNX.
4. **postEvaluationTransform** for a raw regressor is `NoTransform` — confirm the enum name/number.
5. `missingValueTracksTrueChild` — CatBoost NaN routing: confirm the value CatBoost's exporter
   sets (SPEC notes upstream `missing_value_tracks_right_child`); pick the value matching this
   port's apply-time NaN routing, default `false` if unconfirmed, and record the assumption.

---

## Symbol / path verification (CodeGraph + local)

| Item | Location | Status |
|---|---|---|
| `Model{oblivious_trees,bias,float_feature_borders,ctr_data,non_symmetric_trees,region_trees,approx_dimension,class_to_label}` | `crates/cb-model/src/model.rs:271-313` | VERIFIED CODEGRAPH |
| `ObliviousTree{splits,leaf_values,leaf_weights}` | `crates/cb-model/src/model.rs:254-264` | VERIFIED CODEGRAPH |
| ONNX guard `is_onnx_exportable` (order: non-sym → region → CTR) | `onnx.rs:99-115` | VERIFIED CODEGRAPH |
| ONNX per-tree fragment `build_tree_nodes`/`TreeNodeFragment`/`SharedNodeArrays` | `onnx.rs:178-326` | VERIFIED CODEGRAPH |
| ONNX ensemble assembler + `base_values` conditional | `build_regressor_node` `onnx.rs:333-373` | VERIFIED CODEGRAPH |
| ONNX encode+write | `export_onnx` `onnx.rs:544-631` (`prost::Message::encode` :628, `std::fs::write` :629) | VERIFIED CODEGRAPH |
| `OnnxExportError` (thiserror; `Encode(#[from] EncodeError)`, `Io(#[from] io::Error)`) | `onnx.rs:57-93` | VERIFIED CODEGRAPH |
| export module wiring | `crates/cb-model/src/export/mod.rs:9-11` (`mod onnx; pub use onnx::{export_onnx, OnnxExportError};`) | VERIFIED LOCAL |
| mod.rs doc already anticipates `coreml.rs` sibling | `export/mod.rs:3-7` | VERIFIED LOCAL |
| generated-module macro + wiring | `lib.rs:74-97` (`generated_module!(onnx_generated, "generated/onnx_generated.rs")`) | VERIFIED LOCAL |
| generated dir contents | `src/generated/{model,features,ctr_data,onnx}_generated.rs` | VERIFIED LOCAL |
| facade `save_onnx` → maps export err → `CatBoostError::Export` | `crates/catboost-rs/src/model.rs:271-274` | VERIFIED CODEGRAPH |
| `cb_model` crate-root re-export (export mod is PRIVATE) | `crates/cb-model/src/lib.rs:18` `mod export;`, `:37` `pub use export::{export_onnx, OnnxExportError};` | VERIFIED CODEGRAPH — basis of CRITICAL A |
| facade error hard-typed to ONNX (NOT boxed) | `crates/catboost-rs/src/error.rs:84` `Export(#[from] cb_model::OnnxExportError)`; `catboost-rs/src/lib.rs:43` `pub use cb_model::OnnxExportError;` | VERIFIED CODEGRAPH — basis of CRITICAL B (needs NEW `CoreMlExport` variant) |
| Python `to_pyerr` is EXHAUSTIVE (no wildcard) | `crates/catboost-rs-py/src/errors.rs:113-135` (outer `match err` over all variants; inner ONNX match `:125-134`) | VERIFIED LOCAL — new variant → `E0004` until an arm is added |
| Python facades `save_onnx` | `catboost-rs-py/src/{regressor.rs:98, classifier.rs:177}` | VERIFIED CODEGRAPH |
| ONNX facade integration test precedent | `crates/catboost-rs/tests/onnx_facade_test.rs` | VERIFIED LOCAL |
| ONNX sibling unit test precedent | `crates/cb-model/src/export/onnx_test.rs` (mounted `#[cfg(test)] #[path=..]`) | VERIFIED (blast radius) |
| `prost = 0.14.4` present; `protox` used for onnx codegen | `crates/cb-model/Cargo.toml`; `onnx_generated.rs:1-3` | VERIFIED LOCAL |

**No existing CoreML symbols** (`grep -i coreml` empty) — this is greenfield, purely additive; zero
overlap with in-flight `fstr.rs`/`tree.rs`.

---

## Lint / test conventions (must satisfy)

- Lint gate is **clippy, not build**: `unwrap`/`expect`/`panic`/`indexing_slicing` are DENIED in
  new production code. Mirror ONNX's defensive style (`i64::try_from(..).unwrap_or(i64::MAX)`,
  `.get(..).copied().unwrap_or(..)`, `checked_*`). `anyhow` is BANNED in `cb-model` — use
  `thiserror`.
- **Source/test separation (CLAUDE.md, mandatory):** NO `#[cfg(test)] mod tests` inside a prod
  file. Unit tests live in a sibling `coreml_test.rs` mounted from `coreml.rs` via
  `#[cfg(test)] #[path = "coreml_test.rs"] mod tests;`. Golden/integration tests live in
  `crates/cb-model/tests/coreml_export_test.rs`. Omitting the mount silently runs 0 tests.
- The committed `coreml_generated.rs` is UNMODIFIED generator output under the `generated_module!`
  lint-exemption umbrella (never hand-edited).

---

## Execution waves & dependency graph

```text
Wave 1:  CM-R0 (schema)  ┊  CM-01 (guard+error, parallel — coreml.rs w/o schema import)
Wave 2:  CM-02 (node encoding + round-trip)      [needs CM-R0 ∧ CM-01]
Wave 3:  CM-03 (encode/write + facade wiring)     [needs CM-02]
Wave 4:  CM-04 (golden-bytes pin)                 [needs CM-03]
         CM-PY (optional Python save_coreml)      [needs CM-03] — DEFERRABLE
```
```text
CM-R0 ─┐
       ├─> CM-02 ─> CM-03 ─> CM-04
CM-01 ─┘                 └─> CM-PY (optional)
```
Acyclic. CM-R0 and CM-01 both touch `crates/cb-model/src/lib.rs` but in DISTINCT, non-overlapping
hunks — CM-R0 adds `generated_module!(coreml_generated, …)` after `:97`; CM-01 extends the re-export
at `:37`. They may run in parallel with a coordinated merge of those two hunks, PROVIDED: (a) CM-01's
`coreml.rs` does not yet `use crate::coreml_generated` (that import is introduced in CM-02), and
(b) the `:37` re-export line naming `export_coreml`/`CoreMlExportError` lands together with CM-01's
`coreml.rs` (so the crate root compiles — the re-export cannot precede the symbols it names). If
concurrent editing of `lib.rs` is undesirable, serialize CM-R0 → CM-01; the plan does not depend on
their concurrency.

**Spec-ID → task coverage:** CM-01→[CM-01]; CM-02→[CM-02]; CM-03→[CM-03]; CM-04→[CM-04].
CM-R0 is enabling infrastructure for CM-02/03/04 (no standalone SPEC behavior). Every SPEC id
(CM-01..CM-04) maps to ≥1 task; every task maps back to ≥1 SPEC id (CM-R0/CM-PY explicitly noted as
infra/optional).

---

## TASK CM-R0 — Source & commit the CoreML prost schema module
**Spec:** enabling infra for CM-02/03/04 (resolves SPEC §9 R2/R3).
**Goal / observable completion:** a committed `crates/cb-model/src/generated/coreml_generated.rs`
containing prost structs for `Model`, `ModelDescription`, `FeatureDescription`, `FeatureType`
(the `multiArray`/`double` arms + array `dataType` enum), `TreeEnsembleRegressor`,
`TreeEnsembleParameters`, `TreeNode`, `TreeNodeBehavior`, `EvaluationInfo`,
`TreeEnsemblePostEvaluationTransform` — wired via `generated_module!` in `lib.rs`, compiling under
`cargo build -p cb-model`. Its header records the exact coremltools tag/commit (mirroring
`onnx_generated.rs:1-24`).
**Prerequisites:** none (Wave 1).
**Files:**
- Create: `crates/cb-model/src/generated/coreml_generated.rs` (committed, unmodified generator output).
- Modify: `crates/cb-model/src/lib.rs` — add `generated_module!(coreml_generated, "generated/coreml_generated.rs");` after line 97, and extend the doc list (lib.rs:70-73 style) with the pinned tag.
- (Codegen input, out-of-band, NOT committed to build): vendored `.proto` files under a scratch dir; regenerate via `protox::compile` + `prost_build::Config::compile_fds` exactly as ONNX did. **Prefer committing structs (ONNX precedent) — do NOT add a `build.rs` `protoc` step** (avoids a host `protoc` dependency; SPEC §9 R3).
**Research step (do FIRST, resolves R2):** WebFetch, pinned to a specific coremltools tag (e.g. a
released `x.y` tag, NOT `main`):
`mlmodel/format/{Model,TreeEnsemble,FeatureTypes,DataStructures,Parameters}.proto`. Confirm: the
`Model.specificationVersion` minimum that supports `treeEnsembleRegressor` (oneof tag 302);
`FeatureType.multiArrayType` + array `dataType` (Double vs Float32); `TreeEnsemblePostEvaluationTransform`
enum (`NoTransform`); and CatBoost's own CoreML exporter child-ordering convention (mapping note 1).
**TDD sequence:**
- *Red:* add a compile-smoke unit test in `coreml_test.rs` (mounted in CM-01/CM-02) that constructs
  `coreml_generated::Model::default()` and `TreeNode::default()` and asserts field defaults — fails
  to compile until the module exists/wires.
- *Green:* commit the generated module + `lib.rs` wiring until it compiles and the smoke test passes.
- *Refactor:* none (generated file is not hand-edited); only trim the vendored `.proto` set to the
  transitive closure actually needed by `Model`+`TreeEnsembleRegressor`.
**Validation:** `cargo build -p cb-model`; `cargo test -p cb-model --lib` (smoke test);
`cargo clippy -p cb-model --lib --no-deps` (generated module is lint-exempt via the macro).
**Completion evidence:** module compiles; header pins tag/commit; smoke test green.
**Parallelization:** parallel with CM-01.

---

## TASK CM-01 — Exportability guard → typed `CoreMlExportError`  [SPEC CM-01]
**Goal / observable completion:** `export_coreml` rejects non-float-only / non-oblivious / region /
multi-dim models with a typed `CoreMlExportError::Unsupported(_)` (variant set mirroring
`OnnxExportError`), writing NO file; accepts a float-only oblivious scalar model.
**MINOR D (resolved):** there is NO "identity-scale" guard. The canonical `Model`
(`model.rs:271-313`) carries NO model-level `scale` field (the `pub scale: f64` at `:62` is a
CTR-split field); scale is baked at load time, so "non-identity-scale" is not representable and NO
guard is written or needed. Do not invent a scale field. Confirm during CM-01 that load bakes scale.
**Prerequisites:** none for the guard+error themselves (Wave 1, parallel with CM-R0). NOTE: keep
`coreml.rs` free of `use crate::coreml_generated` in this task so it compiles before CM-R0 lands.
**Files:**
- Create: `crates/cb-model/src/export/coreml.rs` — `pub enum CoreMlExportError` (thiserror;
  `Encode(#[from] prost::EncodeError)`, `Io(#[from] std::io::Error)`, plus `Unsupported`
  variants: `NonObliviousTreesUnsupported`, `RegionTreesUnsupported`,
  `CategoricalFeaturesUnsupported`, and a `MultiDimUnsupported`/`NonScalar` variant for
  `approx_dimension > 1`); private `fn is_coreml_exportable(model: &Model) -> Result<(), CoreMlExportError>`.
  Mirror `onnx.rs:99-115` check order (non-sym → region → CTR), and ADD the `approx_dimension > 1`
  rejection (regressor-first, SPEC §2 non-goals).
- Create: `crates/cb-model/src/export/coreml_test.rs` (sibling unit tests); mount from `coreml.rs`
  bottom with `#[cfg(test)] #[path = "coreml_test.rs"] mod tests;`.
- Modify: `crates/cb-model/src/export/mod.rs` — `mod coreml;` + `pub use coreml::{export_coreml, CoreMlExportError};`.
  (`export_coreml`'s body may be a `todo!`-free stub returning the guard result until CM-02; do NOT
  use `todo!`/`unimplemented!`/`panic!` — return `Ok(())` after the guard, or a placeholder
  `CoreMlExportError` per the test, to keep clippy green.)
- **Modify (CRITICAL A — crate-root re-export): `crates/cb-model/src/lib.rs:37`** — the `export`
  module is PRIVATE (`mod export;` at `lib.rs:18`); the ONLY public path is the explicit re-export
  at `:37` `pub use export::{export_onnx, OnnxExportError};`. Extend it to
  `pub use export::{export_onnx, OnnxExportError, export_coreml, CoreMlExportError};` so the facade
  can name `cb_model::CoreMlExportError` / `cb_model::export_coreml` at the crate root. Editing only
  `export/mod.rs` is INSUFFICIENT (verified: `catboost-rs/src/lib.rs:43` + `error.rs:84` reference
  `cb_model::OnnxExportError` at the crate root, which works only because of `lib.rs:37`).
**TDD sequence:**
- *Red:* `coreml_rejects_unsupported` — build a CTR model (`ctr_data = Some(..)` or a
  `ModelSplit::Ctr` split), a non-symmetric model, a region model, and a `approx_dimension = 2`
  model; assert each returns the matching `Err(CoreMlExportError::..)` and that no file is written.
  Fails (module/guard absent).
- *Green:* implement `is_coreml_exportable` in the ONNX order + the multi-dim check; wire the guard
  as the first line of `export_coreml`.
- *Refactor:* if the guard logic is byte-identical to ONNX's, consider a shared free-fn; otherwise
  keep separate to preserve ONNX byte-identity (SPEC §7 "keep ONNX byte-identical"). Prefer separate
  to avoid touching `onnx.rs`.
- *Verify:* the four rejection cases + one accept case (guard returns `Ok`) pass.
**Validation:** `cargo test -p cb-model --lib` (runs mounted `coreml_test`);
`cargo clippy -p cb-model --lib --no-deps` (no unwrap/expect/panic/indexing).
**Completion evidence:** `coreml_rejects_unsupported` green; `export/mod.rs` re-exports both symbols.
**Parallelization:** parallel with CM-R0; blocks CM-02.

---

## TASK CM-02 — Tree → CoreML node encoding + round-trip decode  [SPEC CM-02]
**Goal / observable completion:** for a float-only oblivious `&Model`, `export_coreml` builds a
`coreml_generated::Model` whose decoded `TreeEnsembleParameters.nodes` reproduce, per tree:
correct `branchFeatureIndex`, `branchFeatureValue == split.border`, child routing identical to the
ONNX fragment (feature > value → trueChild), and leaf `EvaluationInfo.evaluationValue ==
leaf_values[j]`; `TreeEnsembleParameters.basePredictionValue == [bias]` iff `bias != 0.0` (else
empty); tree count == `oblivious_trees.len()`; `numPredictionDimensions == 1`.
**Prerequisites:** CM-R0 (schema) ∧ CM-01 (module + guard). Introduce `use crate::coreml_generated
as coreml;` here.
**Files:**
- Modify: `crates/cb-model/src/export/coreml.rs` — add `fn build_tree_nodes(tree, tree_id) ->
  Vec<coreml::TreeNode>` (port `onnx.rs:178-237` node/leaf numbering `2i+1`/`2i+2`, forward-bit
  leaf order, reversed split-order `splits[k-1-depth]`, defensive `.get`/`try_from`), and a
  `build_regressor(model) -> coreml::TreeEnsembleRegressor` assembler (port `onnx.rs:333-373`
  concatenation + the `bias != 0.0` conditional → `basePredictionValue`). Return a
  `coreml::Model` builder (no file I/O yet).
- Modify: `crates/cb-model/src/export/coreml_test.rs` — add `coreml_nodes_match_source`.
**TDD sequence:**
- *Red:* `coreml_nodes_match_source` — construct a deterministic 2-tree, depth-2 float-only `Model`
  in-code (known borders + leaf_values), build the proto, `prost::Message::encode` then
  `coreml::Model::decode` the bytes back, and assert: node thresholds == borders, leaf
  `evaluationValue`s == `leaf_values` in forward-bit order, tree count == 2, child ids match the
  ONNX `2i+1/2i+2` scheme, `nodeBehavior` ∈ {BranchOnValueGreaterThan, LeafNode}. Add a zero-bias
  vs non-zero-bias pair asserting `basePredictionValue` presence/absence. Fails (encoder absent).
- *Green:* implement `build_tree_nodes` + `build_regressor` minimally to satisfy the decode
  assertions.
- *Refactor:* factor the shared per-tree numbering if it clarifies; keep the leaf-value channel
  (`EvaluationInfo`) distinct from ONNX's `target_weights`.
- *Verify:* round-trip decode assertions green for both bias cases.
**Validation:** `cargo test -p cb-model --lib`; `cargo clippy -p cb-model --lib --no-deps`.
**Completion evidence:** `coreml_nodes_match_source` green; decoded structure == source model.
**Parallelization:** blocks CM-03. No parallel sibling.

---

## TASK CM-03 — Encode + write + facade `save_coreml`  [SPEC CM-03]
**Goal / observable completion:** `export_coreml(model, path)` encodes the `coreml::Model` to bytes
and writes `path`; facade `Model::save_coreml(path)` maps `CoreMlExportError` via a NEW
`CatBoostError::CoreMlExport` variant and returns `Ok(())` for a supported model,
`Err(CatBoostError::CoreMlExport(..))` for a CTR model. The written file re-decodes to a valid
`coreml::Model` whose trees match CM-02. After this task the WHOLE workspace still compiles
(`cb-model` + `catboost-rs` + `catboost-rs-py`).
**Prerequisites:** CM-02.

**CRITICAL B (resolved) — error wiring, NOT a reuse of `Export`:** `CatBoostError::Export` is
hard-typed `Export(#[from] cb_model::OnnxExportError)` (`catboost-rs/src/error.rs:84`) — thiserror
cannot attach a second `#[from]` of a different type, and `CoreMlExportError` cannot construct an
`OnnxExportError`. Therefore a NEW variant is REQUIRED. `CatBoostError` is NOT `#[non_exhaustive]`
and `catboost-rs-py::to_pyerr` (`errors.rs:113-135`) matches ALL variants with NO wildcard, so the
new variant breaks that exhaustive match (`E0004`) until a matching arm is added. The plan's earlier
"purely additive / no broken matches" framing is CORRECTED: this task deliberately touches the
facade error enum + the Python error mapping, and both are in this task's file list + validation.

**Files:**
- Modify: `crates/cb-model/src/export/coreml.rs` — finish `pub fn export_coreml(model: &Model,
  path: &Path) -> Result<(), CoreMlExportError>`: guard → build `coreml::Model` (with
  `ModelDescription`: one input `FeatureDescription` named `"features"` typed as the
  multiArray/float per CM-R0, one output `"predictions"`, `predictedFeatureName="predictions"`;
  `specificationVersion` from CM-R0) → `prost::Message::encode` → `std::fs::write` (mirror
  `onnx.rs:627-629`). **Determinism (MAJOR C):** do NOT populate any protobuf `map<…>` field —
  leave `Metadata`/`userDefined` EMPTY (prost renders maps as nondeterministic `HashMap`); do NOT
  embed `env!("CARGO_PKG_VERSION")`, build time, or any timestamp (the ONNX exporter embeds the
  version at `onnx.rs:622` — DO NOT copy that idiom into any CoreML field).
- Modify: `crates/catboost-rs/src/error.rs` — add a NEW variant
  `CoreMlExport(#[from] cb_model::CoreMlExportError)` alongside `Export(..)` at `:84` (do NOT reuse
  `Export`). This gives `save_coreml`'s `?` its `From` conversion.
- Modify (CRITICAL A tail): `crates/catboost-rs/src/lib.rs:43` — add `pub use cb_model::CoreMlExportError;`
  next to the existing `pub use cb_model::OnnxExportError;`.
- Modify: `crates/catboost-rs/src/model.rs` — add `pub fn save_coreml(&self, path: &Path) ->
  Result<(), CatBoostError>` next to `save_onnx` (:271); body `export_coreml(&self.inner, path)?;
  Ok(())` (the `?` now maps via the new `#[from]` on `CoreMlExport`).
- Modify: `crates/catboost-rs-py/src/errors.rs` `to_pyerr` (`:113-135`) — add an OUTER arm
  `FacadeError::CoreMlExport(e) => match e { … }` with an INNER per-`CoreMlExportError`-variant
  mapping mirroring the ONNX arm at `:125-134` (guard-rejection variants →
  `CatBoostValueError`; `Io` → `PyIOError`; `Encode` → base `CatBoostError`). Update the doc block
  at `:104-112` to mention CoreML. Without this, `catboost-rs-py` fails to compile (`E0004`).
- Modify: `crates/catboost-rs-py/src/errors_test.rs` (or the crate's error test file) — add a case
  asserting `to_pyerr` maps each `CoreMlExport` sub-variant to the expected Python exception type
  (mirror the existing ONNX `Export` test cases).
- Create: `crates/cb-model/tests/coreml_export_test.rs` — integration `save_coreml_roundtrip`
  (write to a `tempfile`/scratch path, re-read, `coreml::Model::decode`, assert tree structure) and
  a `cb-model`-level `export_coreml_writes_file` if a facade-free variant is preferred.
- Create: `crates/catboost-rs/tests/coreml_facade_test.rs` — `save_coreml_rejects_ctr` (facade
  returns `Err(CatBoostError::CoreMlExport(..))` for a CTR model) mirroring `onnx_facade_test.rs`.
**TDD sequence:**
- *Red:* `save_coreml_roundtrip` writes+re-decodes and asserts structure; `save_coreml_rejects_ctr`
  asserts the typed facade error; the `to_pyerr` test asserts each CoreML sub-variant mapping. All
  fail (write path + facade + variant + py arm absent).
- *Green:* implement encode+write; add the `CoreMlExport` variant + `save_coreml`; add the
  `to_pyerr` outer+inner arm + its test; add the `catboost-rs/src/lib.rs:43` re-export.
- *Refactor:* dedupe the encode+write tail with a tiny local helper if warranted; keep `onnx.rs`
  untouched.
- *Verify:* file exists, re-decodes, structure matches; CTR facade path errors; `to_pyerr` exhaustive
  match compiles.
**Validation:** `cargo test -p cb-model --test coreml_export_test`;
`cargo test -p catboost-rs` (facade + rejection);
**`cargo build -p catboost-rs-py`** (proves the new variant did NOT break the exhaustive `to_pyerr`
match) and the crate's error test; `cargo clippy -p cb-model --lib --no-deps`.
**Completion evidence:** round-trip + facade rejection tests green; `catboost-rs-py` builds with the
new arm; determinism preconditions (no maps, no version/timestamp) held.
**Parallelization:** blocks CM-04 and CM-PY.

---

## TASK CM-04 — Golden-bytes regression pin  [SPEC CM-04]
**Goal / observable completion:** the exact emitted `.mlmodel` bytes for a FROZEN tiny float model
equal a committed golden artifact; drift in encoding fails the test. This is the closest available
"oracle" absent an Apple runtime (SPEC §9 R1) — explicitly NOT a numeric ≤1e-5 check.
**Prerequisites:** CM-03.
**Determinism preconditions (MAJOR C — HARD REQUIREMENTS, enforced by CM-03's builder):**
1. The emitted bytes MUST contain NO protobuf `map<…>` field. prost renders maps as `HashMap` whose
   iteration order (hence wire encoding) is nondeterministic run-to-run. Concretely: leave CoreML
   `Metadata.userDefined` (and any other map field) EMPTY. A single populated map entry makes this
   test flake.
2. The emitted bytes MUST NOT embed `env!("CARGO_PKG_VERSION")`, build timestamp, or any wall-clock
   value (unlike the ONNX exporter, which sets `producer_version` at `onnx.rs:622`). Such a field
   would break the golden on every unrelated crate version bump.
CM-04 MUST add an explicit assertion/comment recording both preconditions; if either is violated the
golden is unusable as a regression pin.
**Files:**
- Create: fixture dir `crates/cb-model/tests/fixtures/coreml_export/` (co-located with the
  integration test, since the golden is SELF-generated, not a CatBoost oracle):
  - a deterministic tiny model source — PREFER an in-code constructed `Model` (identical bytes
    every run, no `.cbm`-load variance); the frozen-`.cbm` alternative (SPEC §5 CM-04) is acceptable
    only if a committed float-only `.cbm` is reused (avoid generating a fresh one — CatBoost
    quantization is run-to-run nondeterministic, per project memory). Decision: **in-code Model**.
  - `golden.mlmodel` — the committed expected bytes (regenerated ONLY deliberately).
- Modify: `crates/cb-model/tests/coreml_export_test.rs` — add `coreml_golden_bytes_stable`: build
  the frozen model, `export_coreml` to a scratch path, read bytes, `assert_eq!` against
  `include_bytes!("fixtures/coreml_export/golden.mlmodel")`. Include a commented, guarded
  regenerate helper (env-flag gated) so the golden is never silently rewritten.
**TDD sequence:**
- *Red:* add `coreml_golden_bytes_stable` referencing a not-yet-committed golden → fails
  (missing file / mismatch).
- *Green:* run the exporter once, inspect+commit the emitted bytes as `golden.mlmodel`; test passes.
- *Refactor:* document in the test header WHEN and HOW to regenerate (schema change → CM-R0 re-pin →
  regenerate golden deliberately).
- *Verify:* two consecutive `cargo test` runs produce identical bytes (determinism); confirm by
  inspection that the emitted bytes carry NO map field and NO version/timestamp string (MAJOR C
  preconditions above).
**Validation:** `cargo test -p cb-model --test coreml_export_test`;
`cargo clippy -p cb-model --lib --no-deps`.
**Completion evidence:** `coreml_golden_bytes_stable` green; `golden.mlmodel` committed.
**Parallelization:** terminal on the main path.

---

## TASK CM-PY — (OPTIONAL, DEFERRABLE) Python `save_coreml`
**Spec:** extends CM-03 to the Python surface (SPEC §2 "optional Python `save_coreml`").
**Goal / observable completion:** `CatBoostRegressor.save_coreml(path)` exposed via PyO3, mirroring
`regressor.rs:98-108` `save_onnx` (NotFitted guard + `py.detach(|| model.save_coreml(..))` +
`PyCbError` map).
**Prerequisites:** CM-03. **Deferrable** — not required for phase-17 core completion.
**Files:** Modify `crates/catboost-rs-py/src/regressor.rs` (add `save_coreml`); (classifier omitted
— regressor-first, SPEC §2 non-goals).
**TDD sequence:** *Red* a py-side unfitted-error + fitted-write test; *Green* add the method;
*Refactor* share the not-fitted message helper; *Verify* the method writes a decodable file.
**Validation:** `cargo test -p catboost-rs-py` (or the crate's py test harness).
**Parallelization:** parallel with CM-04; both depend only on CM-03.

---

## Consistency check (self-audit)

- Every SPEC id CM-01..CM-04 maps to exactly one primary task; CM-R0 (infra) and CM-PY (optional)
  are explicitly labeled. ✔
- Each task has Red / Green / Refactor / Verify. ✔
- Dependency graph acyclic; waves declared; only CM-R0∥CM-01 and CM-04∥CM-PY run parallel (no write
  conflicts). ✔
- All referenced existing paths/symbols verified (CodeGraph/local table above); all NEW files marked
  `Create`. ✔
- Lint discipline (no unwrap/expect/panic/indexing; `anyhow` banned) and source/test separation
  (sibling `coreml_test.rs` mount + `tests/coreml_export_test.rs`) stated per task. ✔
- **Full blast radius captured (post PLAN-CHECK):** crate-root re-exports at `cb-model/src/lib.rs:37`
  (CM-01) and `catboost-rs/src/lib.rs:43` (CM-03); NEW `CatBoostError::CoreMlExport` variant at
  `catboost-rs/src/error.rs:84` (CM-03); exhaustive `to_pyerr` + `errors_test.rs` in
  `catboost-rs-py` (CM-03); `cargo build -p catboost-rs-py` in CM-03 validation. NO reuse of the
  ONNX-typed `Export` variant; NO "purely additive / no broken matches" claim — CM-03 deliberately
  edits the facade + Python error enums and its validation proves the exhaustive matches still
  compile. ✔
- No production code written by this plan. ✔ No GSD used. ✔

## Unresolved blockers / risks (carry to implementation)

- **R2 — schema field layout (HIGH, partially resolved).** Top-level + TreeEnsemble tags sourced
  live and recorded above; STILL to pin in CM-R0: `FeatureTypes.proto` (`multiArrayType`/`dataType`),
  `TreeEnsemblePostEvaluationTransform` enum, minimum `specificationVersion`, and CatBoost's own
  CoreML child-ordering + `missingValueTracks*` convention. A wrong field/direction yields a
  structurally-valid but semantically-wrong `.mlmodel` this host cannot detect.
- **R1 — no numeric oracle (ACCEPTED).** No Apple CoreML runtime on Linux → NO ≤1e-5 numeric
  parity. Verification is structural round-trip (CM-02/CM-03) + golden bytes (CM-04) ONLY. This
  feature ships without true predict-parity on this host; surface this to the user.
- **R3 — codegen vs commit (DECIDED).** Commit generated prost structs (ONNX precedent); do NOT add
  a `build.rs` `protoc` step. Re-pin tag/commit in the `coreml_generated.rs` header on any schema
  change.
- **Open scope Q1 — classifier CoreML output** deferred (regressor-first), consistent with SPEC.
