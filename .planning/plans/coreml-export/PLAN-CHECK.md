## Plan Check Result

**Verdict:** ISSUES_FOUND
**Goal:** Implement CoreML `.mlmodel` export for float-only oblivious scalar regressor models, mirroring the existing ONNX exporter (acceptance CM-01..CM-04 + §6). Accepted caveat: no Apple runtime → structural round-trip + golden bytes, not ≤1e-5 numeric parity.
**Plan:** `.planning/plans/coreml-export/PLAN.md` (EXPORT-02, plan_version 1)

### Summary
- The plan is well-structured, the ONNX mirror symbols it cites are all real and correctly located, the committed-generated-module approach matches precedent, and the R1/R2 caveats are honestly flagged (verification confidence is NOT overstated).
- However the wiring/blast-radius analysis is wrong in two load-bearing places: (1) the plan never updates `cb-model`'s crate-root re-export, so `export_coreml`/`CoreMlExportError` are unreachable from the facade; (2) the SPEC's "map `CoreMlExportError → CatBoostError::Export`" is impossible and any new `CatBoostError` variant breaks an exhaustive match in `catboost-rs-py`. Both are compile-breaks that contradict the plan's "purely additive / no broken matches" claim. Verdict ISSUES_FOUND.

### Specification Coverage
- [x] CM-01 exportability guard → typed error: TASK CM-01 (`coreml_rejects_unsupported`). Guard mirror verified: `is_onnx_exportable` `onnx.rs:99-115` (order non-sym→region→CTR) exists; plan correctly ADDS `approx_dimension>1` (ONNX guard does not check it). Evidence OK.
- [x] CM-02 tree→node encoding + `basePredictionValue` iff `bias!=0.0`: TASK CM-02. ONNX conditional verified at `onnx.rs:361`; child scheme `false=2i+1 / true=2i+2` verified `onnx.rs:220-223`. Round-trip decode is a sound structural check.
- [x] CM-03 encode+write+facade: TASK CM-03. `export_onnx` encode/write verified `onnx.rs:627-629`; facade `save_onnx` verified `catboost-rs/src/model.rs:271`. BUT wiring is under-specified — see CRITICAL A/B.
- [x] CM-04 golden bytes: TASK CM-04, in-code model choice is correct (avoids nondeterministic `.cbm` quantization). Determinism not fully guarded — see MAJOR C.
- [x] §9 R2 UNVERIFIED items (FeatureTypes.proto, postEvaluationTransform enum, specificationVersion, child-ordering/missingValueTracks) explicitly deferred to CM-R0. Satisfied.
- [~] SPEC "identity-scale" guard: no corresponding top-level `Model` field — see MINOR D.

### CodeGraph Evidence
- `is_onnx_exportable` — `crates/cb-model/src/export/onnx.rs:99` (private). Order non-sym→region→CTR confirmed; no `approx_dimension`/`scale` check. Plan's ADD of a multi-dim reject is correct.
- `build_tree_nodes` — `onnx.rs:178`; `false_node_ids=2i+1`, `true_node_ids=2i+2`, mode `BRANCH_GT` (`:218-223`); reversed split order `splits[k-1-depth]` (`:205-208`). Matches plan mapping note 1.
- `build_regressor_node` — `onnx.rs:333`; `base_values` emitted only when `model.bias != 0.0` (`:361`). Matches CM-02.
- `export_onnx` — `onnx.rs:544`; `prost::Message::encode` (`:628`) + `std::fs::write` (`:629`); NOTE it embeds `producer_version: env!("CARGO_PKG_VERSION")` (`:622`) — relevant to MAJOR C.
- `OnnxExportError` — `onnx.rs:57`, thiserror, `Encode(#[from] EncodeError)` + `Io(#[from] io::Error)` + 4 unsupported variants. Mirror target for `CoreMlExportError`.
- Generated-module precedent — `lib.rs:74-97`, `generated_module!(onnx_generated, "generated/onnx_generated.rs")` at `:97`; `onnx_generated.rs:1-24` header confirms out-of-band `protox::compile` + `prost_build`, committed, no `protoc`. CM-R0 mirror is valid.
- `cb-model` crate-root export — `crates/cb-model/src/lib.rs:18` `mod export;` (PRIVATE) and `:37` `pub use export::{export_onnx, OnnxExportError};`. This is the ONLY public path; `cb_model::export::*` is not reachable. Basis of CRITICAL A.
- Facade error — `crates/catboost-rs/src/error.rs:84` `Export(#[from] cb_model::OnnxExportError)` (hard-typed to ONNX; NOT boxed/string). `catboost-rs/src/lib.rs:43` `pub use cb_model::OnnxExportError`. Basis of CRITICAL B.
- Python error mapping — `crates/catboost-rs-py/src/errors.rs:113-135`: `to_pyerr` outer `match err` is EXHAUSTIVE over all 8 `CatBoostError` variants (no `_` arm), and inner `FacadeError::Export(e) => match e { OnnxExportError::… }` is exhaustive over all 6 ONNX variants. Basis of CRITICAL B.
- `Model` struct — `crates/cb-model/src/model.rs:271-313`: fields are `oblivious_trees, non_symmetric_trees, region_trees, bias, float_feature_borders, ctr_data, approx_dimension, class_to_label`. NO model-level `scale` field (the `pub scale: f64` at `:62` belongs to the CTR-split struct). Basis of MINOR D.

### Issues

#### [CRITICAL] A — CoreML symbols never re-exported at `cb_model` crate root → facade cannot compile
- **Plan location:** TASK CM-01 (Files: modifies only `export/mod.rs`); TASK CM-03 (facade uses `export_coreml` + `#[from] cb_model::CoreMlExportError`).
- **Requirement:** CM-03 facade wiring.
- **Evidence:** `cb-model/src/lib.rs:18` declares `mod export;` (private); the only public surface is the explicit re-export at `:37` `pub use export::{export_onnx, OnnxExportError};`. The facade error arm `error.rs:84` and `catboost-rs/src/lib.rs:43` both reference `cb_model::OnnxExportError` (crate-root path), which works ONLY because of `:37`. The plan adds `pub use coreml::{export_coreml, CoreMlExportError}` to `export/mod.rs` but never edits `lib.rs:37`, so those symbols resolve to `cb_model::export::…` which is not public.
- **Failure scenario:** CM-03 adds `export_coreml(&self.inner, path)?` and `Export`-style `#[from] cb_model::CoreMlExportError` in `catboost-rs`; both fail to resolve (`E0603 module 'export' is private` / unresolved import). `catboost-rs` does not compile.
- **Impact:** Phase blocked at CM-03; invalid intermediate state.
- **Required revision:** Add to CM-R0/CM-01 a modification of `crates/cb-model/src/lib.rs:37` to `pub use export::{export_onnx, OnnxExportError, export_coreml, CoreMlExportError};` (and to `crates/catboost-rs/src/lib.rs:43` a `pub use cb_model::CoreMlExportError;`, mirroring the ONNX re-export).

#### [CRITICAL] B — `CoreMlExportError → CatBoostError::Export` is impossible; a new variant breaks an exhaustive `to_pyerr` match (plan's "no broken matches" is false)
- **Plan location:** TASK CM-03 ("map `CoreMlExportError → CatBoostError::Export`"; "ADD that `From`/`#[from]` arm alongside the ONNX one"; "verify `CatBoostError::Export` already wraps a boxed/string source"); SPEC §5 CM-03 and §8 ("purely additive… no broken matches").
- **Requirement:** CM-03 facade error mapping.
- **Evidence:** `CatBoostError::Export` is `Export(#[from] cb_model::OnnxExportError)` (`error.rs:84`) — hard-typed to ONNX, NOT boxed/string as the plan assumes; thiserror cannot attach a second `#[from]` of a different type to the same variant, and `CoreMlExportError` cannot construct an `OnnxExportError`. A separate variant (e.g. `CoreMlExport(#[from] cb_model::CoreMlExportError)`) is therefore required. Adding any new `CatBoostError` variant makes the OUTER `match err` in `catboost-rs-py/src/errors.rs:114-135` (`to_pyerr`) non-exhaustive (no wildcard arm) → `E0004`.
- **Failure scenario:** After CM-03 introduces the new variant, `catboost-rs-py` fails to compile until `to_pyerr` gains a `FacadeError::CoreMlExport(e) => …` arm (and, for user-facing correctness, an inner match over `CoreMlExportError` variants mirroring the ONNX one at `:125-134`). CM-PY (deferred) covers only the Python `save_coreml` *method*, not this error mapping, so deferring CM-PY does NOT avoid the break.
- **Impact:** Workspace build break in `catboost-rs-py`; contradicts SPEC §8. The plan's blast-radius table omits `catboost-rs-py/src/errors.rs`.
- **Required revision:** (1) Correct SPEC §5 CM-03 / §8 to a NEW `CatBoostError::CoreMlExport(#[from] cb_model::CoreMlExportError)` variant (not reuse of `Export`). (2) Add to CM-03 a required edit of `crates/catboost-rs-py/src/errors.rs` `to_pyerr`: an outer arm for the new variant plus an inner per-`CoreMlExportError`-variant mapping (mirror `:125-134`), and add covering cases in `errors_test.rs`. (3) Remove/repair the "purely additive, no broken matches" claim.

#### [MAJOR] C — Golden-bytes determinism not fully guarded (protobuf maps + version/timestamp strings)
- **Plan location:** TASK CM-04 ("two consecutive runs produce identical bytes"); TASK CM-03 (builds `ModelDescription`/`Model`).
- **Evidence:** prost renders protobuf `map<…>` fields as `HashMap` (nondeterministic iteration → nondeterministic encoding). CoreML `Metadata` carries a `userDefined` string map; `ModelDescription` may carry metadata. Separately, the ONNX exporter embeds `producer_version: env!("CARGO_PKG_VERSION")` (`onnx.rs:622`) — copying that idiom into any CoreML field makes the golden change on every crate version bump.
- **Failure scenario:** If CM-03 populates any map field (even one entry) or embeds the crate version, `coreml_golden_bytes_stable` (CM-04) becomes flaky run-to-run or breaks on unrelated version bumps.
- **Impact:** CM-04 (the only substitute "oracle" given R1) becomes unreliable.
- **Required revision:** In CM-03/CM-04 explicitly forbid populating any protobuf map field (leave `Metadata`/`userDefined` empty) and forbid embedding `CARGO_PKG_VERSION`, build time, or any timestamp in the emitted bytes; add an assertion/comment in CM-04 documenting these determinism preconditions.

#### [MINOR] D — SPEC "identity-scale" guard has no corresponding top-level `Model` field
- **Plan location:** SPEC §2/§4/§5 CM-01 ("non-identity-scale → Err"); TASK CM-01 guard list (non-sym→region→CTR→approx_dim, no scale check).
- **Evidence:** `Model` (`model.rs:271-313`) has no model-level `scale` field; `pub scale: f64` at `:62` is a CTR-split field. The canonical model applies no top-level scale, so "non-identity-scale" is not representable and CM-01 neither checks nor can check it.
- **Impact:** The stated CM-01 output ("non-identity-scale → Err") maps to no task and is vacuously satisfied — harmless but a spec/plan inconsistency that could confuse an implementer into inventing a nonexistent field.
- **Required revision:** Drop the "identity-scale" clause from CM-01, or restate it as "N/A — the canonical `Model` carries no model-level scale (baked at load); no guard needed," and confirm during CM-01 that load bakes scale.

### Implementation Order Review
1. CM-R0 ∥ CM-01 (Wave 1): valid, no write conflict — BUT add the `lib.rs:37` re-export edit (CRITICAL A) to whichever of these lands the `coreml` module publicly; it must precede CM-03.
2. CM-02 (needs CM-R0 ∧ CM-01): valid.
3. CM-03 (needs CM-02): must ALSO edit `catboost-rs/src/lib.rs:43` (re-export) and `catboost-rs-py/src/errors.rs` + `errors_test.rs` (CRITICAL B) in the same wave, otherwise the workspace does not build after this task.
4. CM-04 (needs CM-03) with the determinism preconditions of MAJOR C.
5. CM-PY optional/deferrable — but note the error-mapping half of B is NOT deferrable with it.
Graph is otherwise acyclic and correctly sequenced.

### Potential Bugs
- Adding a `CatBoostError` variant without updating `to_pyerr` → non-exhaustive match compile error (CRITICAL B). Mitigation: update `errors.rs`/`errors_test.rs` in CM-03.
- Populated protobuf map / version string → nondeterministic golden bytes (MAJOR C). Mitigation: forbid maps + version/timestamp fields.
- Child-ordering divergence (ONNX vs CoreML `trueChildNodeId`/`falseChildNodeId` sense): plan already flags this as a CM-R0 verification (mapping note 1); keep it a hard gate — a wrong direction is undetectable on this host (R2).

### Required Plan Revisions
1. Add crate-root re-export of `export_coreml` + `CoreMlExportError` at `cb-model/src/lib.rs:37` (and `CoreMlExportError` at `catboost-rs/src/lib.rs:43`). (CRITICAL A)
2. Replace "map to `CatBoostError::Export`" with a NEW `CatBoostError::CoreMlExport(#[from] cb_model::CoreMlExportError)` variant; add required edits to `catboost-rs-py/src/errors.rs` `to_pyerr` (+ `errors_test.rs`) to keep the exhaustive matches compiling; correct the SPEC §8 "no broken matches" claim. (CRITICAL B)
3. Add explicit golden-bytes determinism preconditions (no protobuf maps, no version/timestamp). (MAJOR C)
4. Remove or reword the "identity-scale" guard clause. (MINOR D)

### Unverified Items
- CoreML `.proto` field tags / enum values (FeatureTypes.proto `multiArrayType`+`dataType`, `TreeEnsemblePostEvaluationTransform` `NoTransform`, minimum `specificationVersion` for oneof 302, CatBoost's own child-ordering + `missingValueTracks*` convention): correctly left UNVERIFIED and assigned to CM-R0 (external, not in repo; TreeFinder MCP unavailable). Not resolvable here; the plan's deferral is acceptable but these remain hard gates before any byte is emitted.

---

## Plan Check Result — PASS 2 (re-review of revised SPEC/PLAN)

**Verdict:** PASS
**Goal:** CoreML `.mlmodel` export for float-only oblivious scalar regressor models, mirroring the ONNX exporter (CM-01..CM-04). Accepted caveat: no Apple runtime → structural round-trip + golden bytes, not ≤1e-5 numeric parity.
**Plan:** `.planning/plans/coreml-export/PLAN.md` (EXPORT-02, plan_version 1, revised) + `SPEC.md` (revised §4 error-wiring, §8 compatibility, dropped identity-scale).

### Summary
- All four prior findings (2 CRITICAL, 1 MAJOR, 1 MINOR) are resolved in the revised SPEC/PLAN, and every structural claim was re-verified live against on-disk source via CodeGraph. No new breakage introduced by the revisions. Verdict PASS.

### Re-verification of each prior finding

**[CRITICAL A] — RESOLVED (verified).** The revised plan adds the crate-root re-export edits.
- CodeGraph confirms `crates/cb-model/src/lib.rs:18` is `mod export;` (private) and `:37` is `pub use export::{export_onnx, OnnxExportError};` — the ONLY public path, exactly as the fix targets.
- PLAN CM-01 Files now explicitly modifies `lib.rs:37` → `pub use export::{export_onnx, OnnxExportError, export_coreml, CoreMlExportError};` (PLAN lines 239-245), and CM-03 modifies `crates/catboost-rs/src/lib.rs:43` (verified current content `pub use cb_model::OnnxExportError;`) to add `pub use cb_model::CoreMlExportError;` (PLAN line 330-331).
- SPEC §4 (lines 96-107) and §8 (lines 174-180) state both edits as required. Ordering is handled: the `:37` re-export naming `export_coreml`/`CoreMlExportError` lands with CM-01's `coreml.rs` stub (which defines those symbols), so the re-export never precedes the symbols it names (PLAN lines 165-172). Correct.

**[CRITICAL B] — RESOLVED (verified).** The revised plan introduces a NEW variant and updates the exhaustive Python match.
- CodeGraph confirms `error.rs:84` `Export(#[from] cb_model::OnnxExportError)` is hard-typed to ONNX. A second `#[from]` of a different source type on the same variant is impossible in thiserror, and `CoreMlExportError` cannot construct an `OnnxExportError` — so reuse of `Export` is genuinely impossible; a new variant is required. (Two distinct variants each carrying a different `#[from]` source type is valid — no From-impl conflict.) Verified.
- CodeGraph confirms `to_pyerr` (`errors.rs:113-135`) is exhaustive with NO `_` arm; the outer `match err` over `FacadeError` and the inner `FacadeError::Export(e) => match e {…}` over all ONNX variants are both exhaustive → any new `CatBoostError` variant triggers E0004 until an arm is added.
- PLAN CM-03 now: (1) adds `CoreMlExport(#[from] cb_model::CoreMlExportError)` at `error.rs:84` (lines 327-329); (2) adds an OUTER `FacadeError::CoreMlExport(e) => match e {…}` arm with an INNER per-variant mapping mirroring the ONNX arm at `:125-134` (lines 335-339); (3) adds covering cases in `errors_test.rs` (lines 340-342); (4) runs `cargo build -p catboost-rs-py` in validation (line 360) to prove the exhaustive match still compiles. SPEC §8 (lines 174-180) drops the old "purely additive / no broken matches" framing and names both required edits. Fully resolved.

**[MAJOR C] — RESOLVED (verified).** CM-04 now carries hard determinism preconditions (PLAN lines 373-382): (1) NO protobuf `map<…>` field (leave `Metadata.userDefined` empty — prost renders maps as nondeterministic `HashMap`); (2) NO `env!("CARGO_PKG_VERSION")`/timestamp/wall-clock value, explicitly "unlike the ONNX exporter, which sets `producer_version` at `onnx.rs:622`". CodeGraph confirms `onnx.rs:622` is exactly `producer_version: env!("CARGO_PKG_VERSION").to_owned()`. CM-03's builder note (lines 324-326) enforces the same at construction time. CM-04 adds an explicit assertion/comment recording both preconditions. Resolved.

**[MINOR D] — RESOLVED (verified).** No task checks a model-level scale field. CodeGraph confirms the `Model` struct (`model.rs:272-313`) has fields `oblivious_trees, non_symmetric_trees, region_trees, bias, float_feature_borders, ctr_data, approx_dimension, class_to_label` — NO model-level `scale`. SPEC §3 note (lines 68-71), §4 (lines 82-83), §5 CM-01 (lines 112-113) and PLAN CM-01 "MINOR D (resolved)" block (lines 219-222) all state scale is not representable/guardable and no guard is written. The identity-scale rejection is dropped. Resolved.

### CM-R0 schema gates — still held (verified honest)
- The UNVERIFIED CoreML `.proto` items remain hard gates BEFORE any byte is emitted: PLAN CM-R0 research step (lines 195-200) and "Unresolved blockers R2" (lines 448-452) require pinning `FeatureTypes.proto` (`multiArrayType`/`dataType`), `TreeEnsemblePostEvaluationTransform` (`NoTransform`), minimum `specificationVersion` for oneof tag 302, and CatBoost's own child-ordering + `missingValueTracks*` convention, all to an exact coremltools tag recorded in the `coreml_generated.rs` header. SPEC §9 R2 (lines 189-191) flags the same as HIGH-attention.
- The no-numeric-oracle limitation (R1) is not overstated: verification is explicitly structural round-trip (CM-02/CM-03) + golden bytes (CM-04) only, with the ≤1e-5 bar declared unachievable on this host and to be surfaced to the user (SPEC §1 caveat, §9 R1; PLAN header + R1). Honest, not overstated.

### No new issues introduced
- Two distinct `#[from]` variants (`Export` for ONNX, `CoreMlExport` for CoreML) coexist without From-impl conflict — source types differ. Verified.
- CM-01's stub `export_coreml` returns the guard result (no `todo!`/`panic!`), so the crate-root re-export at `:37` names real symbols and `cb-model` compiles before CM-R0's schema module lands (the coreml_generated import is deferred to CM-02). Consistent.
- `model.rs` `save_coreml` gets its `?`→`CatBoostError` conversion from the new `#[from]` on `CoreMlExport`; `export_coreml`/`CoreMlExportError` are reachable via the CM-01/CM-03 re-exports. Wiring is complete and correctly sequenced.
- Nit (non-blocking, implementer detail): CM-03 should add `export_coreml` to the `use cb_model::{…}` import list in `catboost-rs/src/model.rs:17-21`; this is implied by the named edit ("body `export_coreml(&self.inner, path)?`") and lands in a file already in CM-03's list. Not a plan defect.

### Final verdict
All prior BLOCKER/CRITICAL/MAJOR/MINOR findings resolved; CM-R0 schema/oracle gates preserved; no regressions in the revision. **PASS.**
