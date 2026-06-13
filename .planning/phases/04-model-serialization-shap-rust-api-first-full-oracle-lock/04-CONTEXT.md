# Phase 4: Model, Serialization, SHAP & Rust API (First Full Oracle Lock) - Context

**Gathered:** 2026-06-13
**Status:** Ready for planning

<domain>
## Phase Boundary

The first **complete vertical slice** — train → serialize → load → predict/explain — oracle-locked end-to-end (≤1e-5 vs upstream CatBoost **1.2.10**) for **numeric binary classification + regression**, exposed through the public `catboost-rs` Builder API. This is where the internal `cb-train::Model` becomes a real **serializable, loadable, explainable, publicly-callable** model. Deliverables (MODEL-01/02/03/04/06, LOSS-01, LOSS-06, RAPI-01/02):

- **`.cbm` (FlatBuffers) serialization** — save/load round-trip + load upstream-produced 1.2.10 `.cbm` + upstream loads ours (bidirectional interop).
- **CPU inference/apply path** — independent of any GPU toolchain (MODEL-02).
- **JSON model export** matching the upstream `model.json` schema (MODEL-06).
- **Regular SHAP** values (MODEL-04) + **feature importance** PredictionValuesChange & Interaction (MODEL-03, partial).
- **Binary-classification training** Logloss + CrossEntropy + Focal (LOSS-01), and **prediction types** RawFormulaVal/Probability/LogProbability/Class/Exponent (LOSS-06, core subset).
- **Public `CatBoostBuilder` Builder API** (RAPI-01) + typed `thiserror` `CatBoostError` enum (RAPI-02), driving a full numeric binclf + regression train→serialize→predict oracle pass ≤1e-5.

**NOT in this phase:** ordered boosting / ordered CTR / categoricals (Phase 5); multiclass / ranking / full regression-loss matrix / text+embedding / advanced fstr (SHAP interaction, PredictionDiff, SAGE) / uncertainty (Phase 6); GPU backends (Phase 7); Python bindings (Phase 8).

</domain>

<decisions>
## Implementation Decisions

### `.cbm` Serialization (MODEL-01)
- **D-01: `flatbuffers` Rust crate + `flatc`-generated bindings, committed.** Generate Rust bindings from the vendored upstream schema (`catboost-master/catboost/libs/model/flatbuffers/{model,features,ctr_data}.fbs`) via `flatc`; commit the generated code. Chosen for exact schema parity, zero-copy reads (memory-efficiency-first constraint), and as the only sane path to reading upstream `.cbm` byte-for-byte. `flatc` becomes a dev/build tool. (Rejected: hand-written schema structs — drift risk; fully hand-rolled wire-format parser — reimplements FlatBuffers, high parity risk.)
- **D-02: Cross-version load bar = catboost 1.2.10 only.** Load `.cbm` produced by the pinned oracle version and apply ≤1e-5. Matches the existing oracle pin; broader version-range tolerance is a later hardening pass, not this phase.
- **D-03: Correctness bar = semantic round-trip + bidirectional interop, NOT byte-identity.** Assert: (a) **our save → our load** reproduces the `Model` exactly; (b) **we load an upstream-produced 1.2.10 `.cbm`** and apply ≤1e-5 vs upstream predictions; (c) **upstream CatBoost loads OUR `.cbm`** and predicts ≤1e-5. Byte-identical output is explicitly out of scope (couples us to FlatBuffers builder internals + uncontrolled metadata like training-param blobs/timestamps).
- **D-04: JSON export targets the upstream `model.json` schema.** Emit the structure CatBoost produces (`oblivious_trees`, `leaf_values`, `scale_and_bias`, `float_features`, …) so the existing `cb-oracle` `model_json.rs` parser doubles as a round-trip oracle and we get real interop. (Rejected: self-defined minimal JSON — not interoperable, no parser reuse.)

### Public Rust API (RAPI-01 / RAPI-02)
- **D-05: Single unified `CatBoostBuilder`.** Takes all params; the **loss function determines** classification vs regression (CatBoost-native style, matches RAPI-01 verbatim). No typed `Classifier`/`Regressor` split this phase (that sklearn flavor is a Phase-8 Python concern).
- **D-06: `predict(pool, PredictionType)` enum core + ergonomic shorthands.** A single enum-selected `predict` entry point (CatBoost-native semantics) PLUS convenience shorthands (`predict_proba()`, `predict()`) from day one.
- **D-07: Serialization + explainability are methods on `Model`.** `Model::save_cbm/load_cbm/save_json/load_json` and `model.shap_values(pool)` / `model.feature_importance(type)` — one cohesive object (CatBoost-native ergonomics), internally delegating to `cb-model`.
- **D-08: Public `CatBoostError` (thiserror) wraps `cb-core::CbError`.** `catboost-rs` defines its own `CatBoostError` with new variants (`Io`, `Deserialize`, `SchemaVersion`, `FeatureMismatch`, …) and a `#[from] CbError` arm for training/data errors. Internal crates keep `cb-core::CbError` — this preserves `CbError`'s `Clone`/`PartialEq`/`Eq` (needed for `Result`-equality test asserts; `io::Error` is neither `Clone` nor `PartialEq`).

### Losses & Prediction Types (LOSS-01 / LOSS-06)
- **D-09: Train & oracle-lock ALL THREE binary-clf losses this phase.** Logloss (carried from Phase 3) + **CrossEntropy** (probability/weighted targets — shares most of Logloss's math) + **Focal** (its own γ-weighted gradient/hessian + dedicated oracle fixture). Regression loss for the lock = **RMSE** (carried from Phase 3). User chose full LOSS-01 breadth.
- **D-10: Prediction types in scope = RawFormulaVal, Probability, LogProbability, Class, Exponent.** Uncertainty types (RMSEWithUncertainty, VirtEnsembles, TotalUncertainty) **deferred to Phase 6** with LOSS-08 — they require uncertainty models that don't exist yet.

### SHAP & Feature Importance (MODEL-03 / MODEL-04)
- **D-11: Regular `EShapCalcType` only; oracle = full per-object SHAP matrix ≤1e-5.** Lock the per-object × (n_features+1) matrix (trailing column = bias/expected-value term), and assert the **local-accuracy invariant** `sum(shap) == prediction`. Strongest correctness signal; stays Python-reachable (D-13).
- **D-12: Feature importance in scope = PredictionValuesChange + Interaction.** `PredictionValuesChange` (model-intrinsic) + `Interaction` (fstr pairwise interaction importance — **NOT** SHAP interaction values, which is MODEL-05/Phase 6). **`LossFunctionChange` is DEFERRED out of Phase 4** — listed in MODEL-03's text but absent from success criterion 3; grouped with advanced-fstr work. **Coverage adjustment:** MODEL-03 is only partially satisfied in Phase 4; its `LossFunctionChange` portion shifts to a later phase (roadmap/requirements should reflect this at transition).

### Oracle Strategy (continues Phase 3 D-11)
- **D-13: Python-reachable oracles only — no C++ instrumentation.** Pinned `catboost==1.2.10`, `thread_count=1`, frozen committed fixtures, per-stage `compare_stage`. New Phase-4 fixtures, all reachable from the Python API: an **upstream-produced `.cbm`** binary (for load-parity), `get_feature_importance(type=ShapValues / PredictionValuesChange / Interaction, data=Pool)`, and `predict(prediction_type=…)` outputs. No upstream C++ build this phase.

### Claude's Discretion (parity-dictated — research reads upstream and reproduces)
- Exact `.cbm` blob framing around the FlatBuffers payload (magic/header, separately-serialized large-array sections) — `catboost/libs/model/model_export/`, model serializer internals.
- Exact CPU **apply / tree-evaluation** procedure for oblivious trees (float-feature binarization at apply time, leaf-index assembly, `scale_and_bias` application) — `catboost/libs/model/`.
- Exact **Regular SHAP** algorithm (per-tree contribution recursion, mean-feature-value baselines, multi-tree aggregation) — `catboost/libs/fstr/shap_values.cpp`, `shap_prepared_trees.cpp`.
- Exact **PredictionValuesChange** and **Interaction** importance formulas — `catboost/libs/fstr/`.
- CrossEntropy and **Focal** gradient/hessian definitions (Focal `focal_alpha`/`focal_gamma` params) — `error_functions.*`, `ders_holder.h`.
- Prediction-type transforms (sigmoid for Probability, log-prob, `Exponent`, class threshold) — `catboost/libs/model/` apply path.
- Where the canonical `Model` type lives post-Phase-4 (re-home `cb-train::Model` into `cb-model` as the serializable canonical model vs `cb-model` operating on `cb-train::Model`) — resolve to avoid a dependency cycle; `cb-model` is the natural home for the serialize/apply/SHAP surface.
- `flatbuffers` crate version (latest stable per CLAUDE.md) and `flatc` invocation (build-time vs committed-generated).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project & Roadmap
- `.planning/PROJECT.md` — core value, constraints (memory-efficiency first-class, `thiserror`/`anyhow`, latest crate versions), oracle strategy.
- `.planning/ROADMAP.md` § "Phase 4: Model, Serialization, SHAP & Rust API (First Full Oracle Lock)" — goal + 5 success criteria this phase is judged against.
- `.planning/REQUIREMENTS.md` — MODEL-01/02/03/04/06, LOSS-01, LOSS-06, RAPI-01/02 requirement text + traceability.
- `.planning/phases/03-cpu-training-core-plain-boosting-oblivious-trees/03-CONTEXT.md` — compute seam (D-01/02/03), Python-reachable oracle pattern (D-11), the trained `Model`/`ObliviousTree`/`Split` representation this phase serializes/applies/explains, `cb-core::sum_f64` reduction invariant.
- `.planning/phases/01-workspace-lint-discipline-oracle-harness/01-CONTEXT.md` — crate map, `catboost-rs` published-facade naming (D-04), oracle pin 1.2.10, fixture format/layout, `thread_count=1` determinism.
- `.planning/phases/02-data-layer-pool-quantization-reduction/02-CONTEXT.md` — `Pool`/`QuantizedPool` (apply-time float binarization input), cat-hash, reduction primitive + CI-grep ban.

### Vendored Reference & Oracle Source (catboost-master/, version 1.2.10)
- `catboost-master/catboost/libs/model/flatbuffers/model.fbs`, `features.fbs`, `ctr_data.fbs` — the `.cbm` FlatBuffers schema `flatc` generates Rust bindings from (D-01).
- `catboost-master/catboost/libs/model/` — model storage, the `.cbm` blob framing around the FlatBuffers payload, and the CPU apply/tree-evaluation path (MODEL-01/02).
- `catboost-master/catboost/libs/model/model_export/` — JSON model export (`model.json`) schema target (D-04, MODEL-06).
- `catboost-master/catboost/libs/fstr/shap_values.cpp`, `shap_values.h`, `shap_prepared_trees.cpp`, `.h` — Regular SHAP algorithm (MODEL-04, D-11).
- `catboost-master/catboost/libs/fstr/` (PredictionValuesChange / Interaction importance) — feature importance (MODEL-03, D-12). NOTE: `shap_interaction_values.*`, `shap_exact.*`, `independent_tree_shap.*` are **Phase 6** (MODEL-05) — do not implement here.
- `catboost-master/catboost/private/libs/algo_helpers/error_functions.cpp`, `.h`, `ders_holder.h` — CrossEntropy + Focal gradient/hessian (D-09, LOSS-01).
- `catboost-master/catboost/private/libs/options/loss_description.*`, `enums.h` — `EShapCalcType`, prediction-type enum, Focal params (D-09/D-10/D-11).

### CubeCL constraint (MODEL-02)
- `AGENTS.md` (project root) + `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md` — the CPU apply path must remain GPU-toolchain-independent; respect the Phase-3 D-02/D-03 seam (kernels + cubecl stay in `cb-backend`; `cb-compute`/apply path stay cubecl-free where applicable).

### Process / Project Rules
- `CLAUDE.md` (project root) — constraints, naming, mandatory source/test separation, latest-crate-versions rule.
- `.planning/codebase/CONVENTIONS.md`, `.planning/codebase/TESTING.md` — Rust lint/error/test conventions and the source/test-separation rule.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `crates/cb-model/` — **stub** (`lib.rs` doc comment only; `anyhow` intentionally absent). This phase fills it with `.cbm`/JSON serialization, the CPU apply path, and SHAP/fstr. Natural home for the canonical serializable `Model`.
- `crates/catboost-rs/` — **stub** published Builder facade. This phase fills it with `CatBoostBuilder` (D-05), `predict` (D-06), `Model` methods (D-07), and the public `CatBoostError` enum (D-08).
- `crates/cb-train/src/boosting.rs` — `Model { oblivious_trees: Vec<ObliviousTree>, bias: f64 }`, `ObliviousTree { splits, leaf_values }`, with `split_borders()` / `leaf_values()` accessors. This is the trained representation Phase 4 serializes, applies, and explains (re-home decision is Claude's discretion, D above).
- `crates/cb-oracle/src/model_json.rs` — already PARSES upstream `model.json` (`ObliviousTree`, `ModelJson`, `scale_and_bias=[1,[bias]]`). Reuse as the JSON round-trip oracle (D-04). Live `.npy` fixture infra + ≤1e-5 `compare_stage` API extends to SHAP matrices / predictions.
- `crates/cb-core/` — `CbError`/`CbResult`, `TFastRng64`, order-locked `sum_f64`/`sum_f32_in_f64`. Apply-time and SHAP sums route through `sum_f64` (D-08 CI-grep ban still applies).
- `crates/cb-data/` — `Pool`/`QuantizedPool` + apply-time float binarization (border lookup) and cat-hash; the apply path consumes these.

### Established Patterns
- Source/test separation mandatory: dedicated `*_test.rs`, no inline `#[cfg(test)]`; test-lint exemption via `#![cfg_attr(test, allow(...))]`.
- `thiserror` in libraries; `anyhow` structurally banned from library crates (CI grep). The new public `CatBoostError` is `thiserror`.
- Oracle determinism: pinned seed, `thread_count=1`, frozen committed fixtures (generator does not run in CI). Phase 4 adds `.cbm` binary + SHAP/importance/prediction-type fixtures (D-13).
- All float summation inside `cb-core::sum_f64` (D-08 grep gate) — applies to the apply path and SHAP.

### Integration Points
- `cb-train::Model` (oblivious trees + bias) → Phase-4 `cb-model` serialize/apply/SHAP → `catboost-rs` Builder facade. The Builder `fit(&pool)` drives `cb-train`; `Model` methods drive `cb-model`.
- The CPU apply path (MODEL-02) must run with no GPU toolchain present — keep it off the `cb-backend` cubecl kernels (or behind a CPU-only path), honoring Phase-3 D-02/D-03.
- The `.cbm`/JSON formats and the apply path established here are the substrate Phases 5/6 extend (categoricals/CTR sections in `ctr_data.fbs`, multiclass leaf dimensions, advanced fstr).

</code_context>

<specifics>
## Specific Ideas

- User chose **maximum interop without byte-identity** for `.cbm` (D-03): three-way semantic parity (ours↔ours, upstream→ours, ours→upstream) is the bar, deliberately avoiding the rabbit-hole of matching FlatBuffers builder byte layout and uncontrolled metadata.
- User chose **breadth on the binclf loss surface** (D-09): all three of Logloss/CrossEntropy/Focal trained & oracle-locked this phase, accepting Focal's extra γ-weighted gradient/hessian + fixture cost — consistent with the Phase-3 pattern of proving a broad math surface against narrow isolating params.
- User chose **per-object SHAP** locking (D-11) over aggregate importance — catching per-object sign/allocation errors that cancel in the mean — and explicitly wants the local-accuracy invariant asserted.
- User favored **CatBoost-native ergonomics** throughout the Rust surface (unified builder, enum-based predict, methods on `Model`) — the sklearn-flavored split is reserved for the Python layer in Phase 8.

</specifics>

<deferred>
## Deferred Ideas

- **LossFunctionChange feature importance** (part of MODEL-03's text) — deferred out of Phase 4; grouped with advanced fstr. Roadmap/requirements coverage should note MODEL-03 is only partially delivered here.
- **SHAP interaction values, PredictionDiff, SAGE** (MODEL-05) — Phase 6.
- **Uncertainty prediction types** (RMSEWithUncertainty, VirtEnsembles, TotalUncertainty) — Phase 6 with LOSS-08.
- **Broader `.cbm` cross-version load tolerance** (beyond 1.2.10) — later hardening pass.
- **Byte-identical `.cbm` output** — explicitly rejected as a goal (D-03).

None of the above are scope creep — all are explicitly later-phase items surfaced while bounding Phase 4.

</deferred>

---

*Phase: 4-Model, Serialization, SHAP & Rust API (First Full Oracle Lock)*
*Context gathered: 2026-06-13*
