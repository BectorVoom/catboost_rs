# Phase 4: Model, Serialization, SHAP & Rust API (First Full Oracle Lock) - Research

**Researched:** 2026-06-13
**Domain:** Model serialization (FlatBuffers `.cbm` + JSON), CPU tree-evaluation/apply, TreeSHAP, feature importance, binary-classification losses, public Rust Builder API — all oracle-locked ≤1e-5 vs upstream CatBoost 1.2.10
**Confidence:** HIGH (every format/algorithm verified by reading the vendored upstream C++ at `catboost-master/`)

## Summary

Phase 4 turns the internal `cb-train::Model` (oblivious trees + bias) into a serializable, loadable, explainable, publicly-callable model. The five sub-domains — `.cbm`/JSON serialization, CPU apply, SHAP, feature importance, and the public Builder API — are all **parity-dictated**: the correct behavior is whatever the vendored upstream C++ does, and this research reads that source directly rather than guessing. Every claim below is `[VERIFIED]` against a specific file/line in `catboost-master/` (the literal oracle), which is the strongest possible source for a parity rewrite.

The single most important structural finding: **the current `cb-train::Model` does not carry per-leaf weights, and SHAP / PredictionValuesChange / Interaction all require them.** Upstream stores `LeafWeights:[double]` in the FlatBuffers model (the per-leaf sum of training-document weights). The trainer accumulates these as `leafWeights[leafIndex] += rowWeight` (`approx_calcer.cpp:160`); for unweighted training a leaf weight is just its document count. Phase 4 must (a) capture leaf weights during/after training, (b) add them to the canonical model, (c) serialize them in both `.cbm` and `model.json`, and (d) consume them in all three fstr/SHAP algorithms. Treat this as a Wave-0 / first-task structural change, because three downstream success criteria depend on it.

**Primary recommendation:** Re-home the canonical serializable `Model` into `cb-model` (carrying `oblivious_trees`, `bias`, and **leaf_weights**), generate Rust FlatBuffers bindings from the vendored `.fbs` via `flatc` and commit them, transcribe the four reference algorithms (apply, TreeSHAP recursion, PredictionValuesChange, Interaction) line-for-line from the cited C++, and build the `catboost-rs` Builder facade on top. Route every float sum through `cb-core::sum_f64` (D-08 grep gate). The `.cbm` blob framing is trivial (magic + size + FlatBuffers payload); the hard parts are SHAP and exact prediction-type transforms.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**`.cbm` Serialization (MODEL-01)**
- **D-01: `flatbuffers` Rust crate + `flatc`-generated bindings, committed.** Generate Rust bindings from the vendored upstream schema (`catboost-master/catboost/libs/model/flatbuffers/{model,features,ctr_data}.fbs`) via `flatc`; commit the generated code. Chosen for exact schema parity, zero-copy reads, and as the only sane path to reading upstream `.cbm` byte-for-byte. `flatc` becomes a dev/build tool. (Rejected: hand-written schema structs; fully hand-rolled wire-format parser.)
- **D-02: Cross-version load bar = catboost 1.2.10 only.** Load `.cbm` produced by the pinned oracle version and apply ≤1e-5.
- **D-03: Correctness bar = semantic round-trip + bidirectional interop, NOT byte-identity.** Assert: (a) our save → our load reproduces the `Model` exactly; (b) we load an upstream-produced 1.2.10 `.cbm` and apply ≤1e-5; (c) upstream CatBoost loads OUR `.cbm` and predicts ≤1e-5. Byte-identical output is explicitly out of scope.
- **D-04: JSON export targets the upstream `model.json` schema.** Emit the structure CatBoost produces (`oblivious_trees`, `leaf_values`, `scale_and_bias`, `float_features`, …) so the existing `cb-oracle` `model_json.rs` parser doubles as a round-trip oracle. (Rejected: self-defined minimal JSON.)

**Public Rust API (RAPI-01 / RAPI-02)**
- **D-05: Single unified `CatBoostBuilder`.** Takes all params; the loss function determines classification vs regression (CatBoost-native style). No typed `Classifier`/`Regressor` split this phase.
- **D-06: `predict(pool, PredictionType)` enum core + ergonomic shorthands.** A single enum-selected `predict` entry point PLUS convenience shorthands (`predict_proba()`, `predict()`) from day one.
- **D-07: Serialization + explainability are methods on `Model`.** `Model::save_cbm/load_cbm/save_json/load_json` and `model.shap_values(pool)` / `model.feature_importance(type)` — one cohesive object internally delegating to `cb-model`.
- **D-08: Public `CatBoostError` (thiserror) wraps `cb-core::CbError`.** `catboost-rs` defines its own `CatBoostError` with new variants (`Io`, `Deserialize`, `SchemaVersion`, `FeatureMismatch`, …) and a `#[from] CbError` arm. Internal crates keep `cb-core::CbError` (preserves `Clone`/`PartialEq`/`Eq` for Result-equality test asserts; `io::Error` is neither).

**Losses & Prediction Types (LOSS-01 / LOSS-06)**
- **D-09: Train & oracle-lock ALL THREE binary-clf losses this phase.** Logloss (from Phase 3) + CrossEntropy (probability/weighted targets — shares Logloss's math) + Focal (its own γ-weighted gradient/hessian + dedicated oracle fixture). Regression loss for the lock = RMSE (from Phase 3).
- **D-10: Prediction types in scope = RawFormulaVal, Probability, LogProbability, Class, Exponent.** Uncertainty types deferred to Phase 6.

**SHAP & Feature Importance (MODEL-03 / MODEL-04)**
- **D-11: Regular `EShapCalcType` only; oracle = full per-object SHAP matrix ≤1e-5.** Lock the per-object × (n_features+1) matrix (trailing column = bias/expected-value term), and assert the local-accuracy invariant `sum(shap) == prediction`.
- **D-12: Feature importance in scope = PredictionValuesChange + Interaction.** `LossFunctionChange` is DEFERRED out of Phase 4. MODEL-03 is only partially satisfied here.

**Oracle Strategy**
- **D-13: Python-reachable oracles only — no C++ instrumentation.** Pinned `catboost==1.2.10`, `thread_count=1`, frozen committed fixtures, per-stage `compare_stage`. New fixtures: an upstream-produced `.cbm` binary, `get_feature_importance(type=ShapValues / PredictionValuesChange / Interaction, data=Pool)`, and `predict(prediction_type=…)` outputs. No upstream C++ build this phase.

### Claude's Discretion (parity-dictated — research reads upstream and reproduces)
- Exact `.cbm` blob framing around the FlatBuffers payload. **→ RESOLVED below (HIGH confidence).**
- Exact CPU apply / tree-evaluation procedure for oblivious trees. **→ RESOLVED below (HIGH).**
- Exact Regular SHAP algorithm. **→ RESOLVED below (HIGH).**
- Exact PredictionValuesChange and Interaction importance formulas. **→ RESOLVED below (HIGH).**
- CrossEntropy and Focal gradient/hessian definitions. **→ RESOLVED below (HIGH).**
- Prediction-type transforms. **→ RESOLVED below (HIGH).**
- Where the canonical `Model` type lives post-Phase-4. **→ Recommendation: re-home into `cb-model` (see Architecture).**
- `flatbuffers` crate version + `flatc` invocation. **→ RESOLVED below (`flatbuffers = "25.12.19"`, `flatc` build-time OR committed; flatc NOT currently installed — see Environment Availability).**

### Deferred Ideas (OUT OF SCOPE)
- LossFunctionChange feature importance — deferred; roadmap should note MODEL-03 partial.
- SHAP interaction values, PredictionDiff, SAGE (MODEL-05) — Phase 6.
- Uncertainty prediction types (RMSEWithUncertainty, VirtEnsembles, TotalUncertainty) — Phase 6.
- Broader `.cbm` cross-version load tolerance (beyond 1.2.10) — later hardening.
- Byte-identical `.cbm` output — explicitly rejected as a goal.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| MODEL-01 | Native `.cbm` (FlatBuffers) serialization — save/load, cross-version, load upstream files | Blob framing decoded (`CBM1` magic + ui32 size + FlatBuffers `TModelCore`); schema in `model.fbs`/`features.fbs`; `flatbuffers` crate OK; `flatc` codegen path |
| MODEL-02 | CPU inference/apply path (GPU-toolchain-independent) | Scalar reference algorithm `CalcIndexesBasic` + `BinarizeFloatsNonSse` transcribed; must NOT touch `cb-backend` cubecl kernels (keep in `cb-model`, pure Rust) |
| MODEL-03 | Feature importance — PredictionValuesChange, Interaction (LossFunctionChange deferred) | `CalcEffect` (feature_str.h:233) + `CalcMostInteractingFeatures` algorithms transcribed; both need LeafWeights |
| MODEL-04 | SHAP values (Regular `EShapCalcType`) | Full TreeSHAP recursion + prepared-trees (subtree weights, mean values) transcribed; needs LeafWeights |
| MODEL-06 | JSON model export | `model.json` field names + structure decoded; `cb-oracle::model_json` reused as round-trip oracle |
| LOSS-01 | Binary classification — Logloss, CrossEntropy, Focal | Exact der1/der2 for all three transcribed from `error_functions.{h,cpp}` |
| LOSS-06 | Prediction types — Probability, LogProbability, Class, RawFormulaVal, Exponent | Exact transforms from `eval_helpers.cpp::PrepareEval` + `eval_processing.h` |
| RAPI-01 | Rust Builder API — `CatBoostBuilder::new()...fit(&pool) -> Model`, predict | Builder over `cb-train::train` + `cb-model` Model methods |
| RAPI-02 | Typed `thiserror` error enum | `CatBoostError` wrapping `CbError` (D-08) |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Train (`fit`) | `cb-train` (compute via `cb-compute`/`cb-backend`) | — | Already built in Phase 3; Builder drives it |
| Canonical serializable `Model` | `cb-model` | `cb-train` (produces raw trees) | `cb-model` is the natural home for serialize/apply/SHAP; avoids dependency cycle |
| `.cbm` / JSON (de)serialization | `cb-model` | `flatbuffers` crate, `serde_json` | Pure I/O + format; no compute |
| CPU apply / tree-evaluation | `cb-model` (pure Rust) | `cb-data` (QuantizedPool / borders) | MODEL-02 requires GPU-toolchain independence — must NOT route through `cb-backend` cubecl |
| SHAP + feature importance | `cb-model` | `cb-core::sum_f64` | Model-intrinsic, operate on serialized model + leaf weights |
| Public Builder + `CatBoostError` | `catboost-rs` (facade) | `cb-train`, `cb-model` | The single published crate; wraps internal `CbError` |
| Loss der1/der2 (CrossEntropy/Focal) | `cb-compute` (`Loss` enum) + `cb-backend` (kernels) | — | Extends Phase-3 elementwise gradient/hessian seam |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `flatbuffers` (Rust runtime) | `25.12.19` | Zero-copy reader/builder for `.cbm` `TModelCore` | Official Google crate; the only correct runtime for flatc-generated Rust; wire-compatible with upstream's vendored 24.3.25 (FlatBuffers wire format is stable across versions) |
| `flatc` (CLI, dev/build tool) | ≥ 24.3.25 (match or newer than vendored) | Generate Rust bindings from the vendored `.fbs` | D-01 mandates flatc-generated, committed bindings |
| `serde` / `serde_json` | `1.0.228` / `1.0.150` | `model.json` export/import | Already in workspace; `cb-oracle::model_json` already uses serde for parsing |
| `cb-core::sum_f64` | (internal) | All float summation in apply + SHAP | D-08 grep gate bans any other summation |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `ndarray` / `ndarray-npy` | `0.17.2` / `0.10.0` | Read `.npy` oracle fixtures (SHAP matrices, predictions) | Test-side; already wired in `cb-oracle` |
| `thiserror` | `2.0.18` | `CatBoostError` enum (RAPI-02) | Public facade error type |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `flatc`-generated bindings | `planus` crate (pure-Rust flatc) | `planus` avoids the external `flatc` dependency but generates a different API and is not what D-01 chose; only consider if `flatc` install proves intractable in CI |
| Committed generated bindings | `build.rs` codegen | Committing (D-01) is correct: deterministic, no flatc needed at consumer build time; `build.rs` would require flatc on every build machine |

**Installation:**
```bash
# flatc is NOT currently installed (see Environment Availability). One-time, dev/CI:
#   Debian/Ubuntu: apt-get install flatbuffers-compiler   (verify version >= 24.3.25)
#   or build from github.com/google/flatbuffers at a tag >= v24.3.25
# Generate (run once, commit output):
flatc --rust -o crates/cb-model/src/generated \
  catboost-master/catboost/libs/model/flatbuffers/model.fbs \
  catboost-master/catboost/libs/model/flatbuffers/features.fbs \
  catboost-master/catboost/libs/model/flatbuffers/ctr_data.fbs
# Add to crates/cb-model/Cargo.toml:  flatbuffers = "25.12.19"
```

**Version verification (2026-06-13):**
- `flatbuffers` crate: latest stable `25.12.19` `[VERIFIED: cargo search]`, published, repo `github.com/google/flatbuffers`, 1.27M weekly downloads `[VERIFIED: package-legitimacy check OK]`.
- Vendored upstream FlatBuffers: `24.3.25` `[VERIFIED: contrib/libs/flatbuffers/include/flatbuffers/base.h:142-144]`. The Rust crate's higher version number reflects FlatBuffers' date-based monorepo versioning; the binary wire format is forward/backward compatible, so `25.12.19` reads schemas authored for `24.3.25` `[CITED: flatbuffers.dev — format stability]`.

## Package Legitimacy Audit

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| `flatbuffers` | crates.io | since 2016-06 | 1.27M/wk | github.com/google/flatbuffers | OK | Approved |

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

`flatc` is an external CLI binary (not a crate); it is the official FlatBuffers compiler from `github.com/google/flatbuffers`. The planner should add a `checkpoint:human-verify` (or an install task) for the `flatc` install since it is **not present** on this machine (see Environment Availability). All other crates (`serde`, `serde_json`, `thiserror`, `ndarray`, `ndarray-npy`) are already in the workspace and were vetted in earlier phases.

## Architecture Patterns

### System Architecture Diagram

```
                         ┌──────────────────────────────────────────────┐
   user (Rust)  ───────► │            catboost-rs (facade)              │
                         │  CatBoostBuilder::new().loss(...).iterations │
                         │     .depth(...)...fit(&pool) -> Model        │
                         │  Model::{predict, predict_proba,             │
                         │     save_cbm/load_cbm, save_json/load_json,  │
                         │     shap_values, feature_importance}         │
                         │  CatBoostError (thiserror, #[from] CbError)  │
                         └───────┬───────────────────────┬──────────────┘
                                 │ fit                    │ Model methods
                                 ▼                        ▼
                  ┌──────────────────────┐   ┌────────────────────────────────────┐
                  │  cb-train::train     │   │  cb-model (canonical Model)         │
                  │  (Phase 3 boosting)  │   │  ┌──────────────────────────────┐   │
                  │  produces trees +    │──►│  │ Model { trees, bias,         │   │
                  │  LEAF WEIGHTS (NEW)  │   │  │   leaf_weights, float_borders}│  │
                  └──────────┬───────────┘   │  └──────────────────────────────┘   │
                             │               │   ┌─────────┐  ┌──────────┐         │
        loss der1/der2 ◄─────┘               │   │ .cbm    │  │ model.json│        │
   ┌──────────────────────┐                  │   │ (FB)    │  │ (serde)  │         │
   │ cb-compute Loss enum │                  │   └─────────┘  └──────────┘         │
   │  +cb-backend kernels │                  │   ┌─────────────────────────────┐   │
   │  CrossEntropy/Focal  │                  │   │ CPU apply (pure Rust):      │   │
   └──────────────────────┘                  │   │  binarize floats → bin idx  │   │
                                             │   │  → leaf index → leaf sum    │   │
   input: Pool / QuantizedPool ─────────────┼──►│  → +bias → prediction-type  │   │
   (cb-data: borders, binarization)         │   │     transform               │   │
                                             │   ├─────────────────────────────┤   │
                                             │   │ SHAP (TreeSHAP recursion,   │   │
                                             │   │  subtreeWeights+meanValues  │   │
                                             │   │  from leaf_weights)         │   │
                                             │   │ PredictionValuesChange      │   │
                                             │   │ Interaction (pairwise)      │   │
                                             │   └─────────────────────────────┘   │
                                             └─────────────────────────────────────┘
                       all float sums ──► cb-core::sum_f64 (D-08 grep gate)
```

### Recommended Project Structure
```
crates/cb-model/src/
├── lib.rs              # re-exports; module wiring
├── model.rs           # canonical Model { oblivious_trees, bias, leaf_weights, float_feature_borders }
├── generated/         # flatc --rust output, committed (model_generated.rs, features_generated.rs, ctr_data_generated.rs)
├── cbm.rs             # .cbm blob framing: magic + size + FBSerialize/FBDeserialize
├── json.rs            # model.json export/import (serde), targeting upstream schema
├── apply.rs           # CPU tree-evaluation (binarize → leaf index → leaf sum → bias)
├── predict.rs         # prediction-type transforms (RawFormulaVal/Probability/LogProbability/Class/Exponent)
├── shap.rs            # TreeSHAP recursion + prepared-trees (subtree weights, mean values)
├── fstr.rs            # PredictionValuesChange + Interaction
└── *_test.rs          # dedicated test files (source/test separation, mandatory)

crates/catboost-rs/src/
├── lib.rs
├── builder.rs         # CatBoostBuilder (D-05)
├── model.rs           # Model facade methods (D-06/D-07) delegating to cb-model
└── error.rs           # CatBoostError (thiserror, D-08)
```

### Pattern 1: `.cbm` Blob Framing
**What:** The `.cbm` file is NOT pure FlatBuffers — it is a tiny framing wrapper around the FlatBuffers `TModelCore` payload.
**When to use:** `save_cbm` / `load_cbm`.
**Layout** `[VERIFIED: model.cpp:1113-1163 (Save), 1179-1228 (Load)]`:
```
offset 0:  4 bytes  = "CBM1"  (the ui32 model format descriptor, little-endian POD)
offset 4:  4 bytes  = ui32 LE = size of the FlatBuffers core blob
offset 8:  N bytes  = FlatBuffers TModelCore buffer (root_type TModelCore)
offset 8+N:          = optional model parts (CTR provider / text / embedding) — OUT OF SCOPE Phase 4
```
- Magic: `static const char MODEL_FILE_DESCRIPTOR_CHARS[4] = {'C','B','M','1'}` reinterpreted as `ui32` `[VERIFIED: model.cpp:41,49-51]`. On little-endian x86_64 the on-disk bytes are literally `C B M 1`.
- Size: `SaveSize`/`LoadSize` is **fixed ui32 LE** for sizes < 0xffffffff, with a `0xffffffff` escape followed by a `ui64` for ≥4GB `[VERIFIED: util/ysaveload.h:277-296]`. NOT varint. Phase-4 models are tiny, so always the 4-byte form.
- Phase 4 (numeric-only, no CTR) writes zero model parts → `ModelPartIds` is empty/absent. On load, an empty `ModelPartIds` means stop after the core blob.
- `FormatVersion` string inside `TModelCore` MUST equal `"FlabuffersModel_v1"` (sic — the upstream typo is canonical) `[VERIFIED: model.cpp:53,1167]`. Our writer must emit it; our reader must check it.

### Pattern 2: CPU Apply / Tree-Evaluation (oblivious, numeric-only)
**What:** Map raw float features → bin indices → per-tree leaf index → accumulate leaf values → add bias → prediction-type transform.
**Reference (scalar, the algorithm to transcribe):** `CalcIndexesBasic` `[VERIFIED: cpu/evaluator_impl.cpp:16-52]` and `BinarizeFloatsNonSse` `[VERIFIED: cpu/quantization.h:81-140]`.

**Step A — binarize each float feature to a bin index** `[VERIFIED: quantization.h:130-138]`:
```
binValue(feature) = count of borders b where (rawValue > b)      // STRICT greater-than
// NaN handling: if HasNans, substitute nanSubstitutionValue first (Min → -inf-ish / Max).
```
**Step B — compute the per-tree leaf index** `[VERIFIED: evaluator_impl.cpp:26-50]`:
```
leafIndex = 0
for depth in 0..treeSize:                       // forward bit order (matches Phase-3 finding)
    border = repackedBin[depth].SplitIdx        // for the common ≤254-borders case: borderId+1
    featureBin = binValue(repackedBin[depth].FeatureIndex)
    bit = (featureBin >= border)                // NaN-free float: XorMask=0
    leafIndex |= (bit << depth)
```
**Step C — accumulate + bias** `[VERIFIED: evaluator_impl.cpp:155-172, eval_processing.h:179 ApplyScaleAndBias]`:
```
raw = bias                                       // scale_and_bias bias term
for each tree: raw += leafValues[treeFirstLeafOffset + leafIndex]
// leafValues are ALREADY learning_rate-scaled (Phase-3 finding); add directly.
```
- TRepackedBin construction `[VERIFIED: model.cpp:560-573]`: float split → `FeatureIndex = bucketIdx`, `SplitIdx = (borderId % 254) + 1`, `XorMask = 0`. `MAX_VALUES_PER_BIN = 254` `[VERIFIED: model.h:72]`. For any feature with ≤254 borders (the Phase-4 case), one bucket per feature and `SplitIdx = borderId + 1`.
- The strict `rawValue > border` (Step A) plus `>=` in Step B together reproduce upstream's `<`/`<=` border semantics already characterized in Phase 2. Reuse `cb-data` border lookup but verify the strict-`>` count matches.

### Pattern 3: Prediction-Type Transforms
**What:** Convert the raw `approx` (RawFormulaVal logit) to the requested output. **Use the Python-reachable dispatcher's math**, because D-13 fixtures come from the Python `predict(prediction_type=…)` API.
**Reference:** `PrepareEval` `[VERIFIED: eval_helpers.cpp:352-496]` (single-dimension / binary path):

| PredictionType | Formula (binary, 1-dim) | exp used | Source |
|----------------|-------------------------|----------|--------|
| `RawFormulaVal` | identity (= raw approx) | — | eval_helpers.cpp:490 |
| `Probability` | `1 / (1 + exp(-approx))` | **`std::exp`** (vector overload) | eval_helpers.cpp:391 → eval_processing.h:103-110 |
| `LogProbability` | 2 columns: `[-log(1+exp(approx)), -log(1+exp(-approx))]` | **`std::exp`** | eval_helpers.cpp:393 → CalcLogSigmoid (eval_processing.h:131-141) |
| `Class` | `approx > binClassLogitThreshold` (default `0`) | — | eval_helpers.cpp:413-414 |
| `Exponent` | `exp(approx)` | **`FastExp` (table/SSE/AVX)** | eval_helpers.cpp:420 → CalcExponent (eval_processing.h:30-33) |

**Critical nuance:** the Python `predict` path's `Probability`/`LogProbability` use the **`std::exp`** vector overloads (`CalcSigmoid(approx[0])` / `CalcLogSigmoid(approx[0])`, eval_processing.h:103-110/153-161) — Rust can match these exactly with `f64::exp`. But `Exponent` uses `CalcExponent` → `FastExpWithInfInplace`, whose AVX2/SSE2 implementations are a **table-based approximation** (`library/cpp/fast_exp/fast_exp.cpp:33-49`), NOT `std::exp`. The ≤1e-5 tolerance must absorb FastExp's approximation error for `Exponent`. See Pitfall 3.
- `LogProbability` returns **two columns** (class-0 and class-1 log-probs) `[VERIFIED: eval_helpers.cpp:393]`. The Rust API and fixtures must produce both.
- `Class` threshold = `model.GetBinClassLogitThreshold()` `[VERIFIED: eval_helpers.cpp:329]`; default 0 unless a probability border was set. For Phase-4 fixtures with no custom border, threshold = 0.

### Pattern 4: Regular TreeSHAP
**What:** Per-object Shapley feature contributions via the Lundberg TreeSHAP polynomial-time recursion.
**Reference:** `CalcObliviousInternalShapValuesForLeafRecursive` + `ExtendFeaturePath` + `UnwindFeaturePath` + `UpdateShapByFeaturePath` `[VERIFIED: shap_values.cpp:26-320, 493-548]`, plus prepared-trees `CalcSubtreeWeightsForTree` + `CalcMeanValueForTree` `[VERIFIED: shap_prepared_trees.cpp:25-222]`.

**Prepared-trees precompute (per tree):**
- `subtreeWeights[depth][node]` — bottom-up sum of **leaf weights**; leaves = `LeafWeights`, internal = sum of children `[VERIFIED: shap_prepared_trees.cpp:177-222]`.
- `meanValue[dim] = (Σ leafValue·leafWeight) / subtreeWeights[0][0]` — the per-tree weighted-average leaf value = `averageTreeApprox` baseline `[VERIFIED: shap_prepared_trees.cpp:25-67]`.

**Per-leaf recursion (feature-path machinery) — transcribe verbatim:**
- `ExtendFeaturePath(path, zeroFrac, oneFrac, feature)` `[VERIFIED: shap_values.cpp:44-64]`: appends element with `weight = (pathLen==0 ? 1 : 0)`, then back-propagates `weight[i+1] += oneFrac·weight[i]·(i+1)/(L+1)` and `weight[i] = zeroFrac·weight[i]·(L-i)/(L+1)`.
- `UnwindFeaturePath(path, eraseIdx)` `[VERIFIED: shap_values.cpp:66-104]`: inverse of extend; two branches on `oneFrac == 0` (use `FuzzyEquals`).
- Recursion `[VERIFIED: shap_values.cpp:196-320]`: at each split, `hotCoefficient = subtreeWeights[d+1][goNode]/subtreeWeights[d][node]`, `coldCoefficient = subtreeWeights[d+1][skipNode]/subtreeWeights[d][node]`; go-branch carries `oneFrac=newOnePathsFraction`, skip-branch carries `oneFrac=0`. At the leaf, `UpdateShapByFeaturePath` distributes `coefficient = weightSum·(oneFrac−zeroFrac)` × `(leafValue − averageTreeApprox)` to each feature.

**Final matrix assembly** `[VERIFIED: shap_values.cpp:1030-1055]`:
- Output shape per document: `[approxDim][featureCount + 1]`. Trailing column index `featureCount` = `Σ_trees meanValue[dim]` + model `bias[dim]`.
- **Local-accuracy invariant (D-11 assert):** `Σ_f shap[dim][f]  (including trailing column)  ==  RawFormulaVal prediction[dim]`. This is the strongest correctness check.
- SHAP requires `scale == 1` `[VERIFIED: shap_values.cpp:1620 CB_ENSURE_SCALE_IDENTITY]` — always true in Phase 4 (scale never set).
- `combinationClass` indirection (binFeatureCombinationClass) is identity for numeric-only models (no CTR combinations) — each float bin-feature maps to its own feature `[VERIFIED: shap recursion uses binFeatureCombinationClass; for numeric-only it's 1:1]`.

### Pattern 5: PredictionValuesChange (feature importance)
**Reference:** `CalcEffect` `[VERIFIED: feature_str.h:233-270]`:
```
for each tree, for each split-feature, for each leaf pair (leaf, leaf ^ (1<<featureBit)) with inverted > leaf:
    count1 = leafWeight[leaf];  count2 = leafWeight[inverted]
    if count1==0 or count2==0: skip
    for each dim:
        avrg = (val1·count1 + val2·count2) / (count1+count2)
        dif  = (val1-avrg)²·count1 + (val2-avrg)²·count2
        res[srcFeature] += dif
ConvertToPercents(res)   // normalize so Σ = 100
```
Needs `LeafWeights`. Output normalized to percentages summing to 100.

### Pattern 6: Interaction (feature importance)
**Reference:** `CalcMostInteractingFeatures` + `CalcFeatureInteraction` `[VERIFIED: feature_str.cpp:190-270, calc_fstr.cpp:314-441]`. Pairwise: for each tree and each pair of split-features, the weighted-variance contribution attributable to the pair, aggregated across trees, normalized to percent. **NOT** SHAP interaction values (that is MODEL-05/Phase 6). Returns `Vec<(feature_i, feature_j, score)>`.

### Anti-Patterns to Avoid
- **Routing the CPU apply path through `cb-backend` cubecl kernels.** MODEL-02 requires GPU-toolchain independence; keep apply/SHAP pure Rust in `cb-model`. The Phase-3 D-02/D-03 seam already keeps `cb-compute` cubecl-free — apply must stay off cubecl entirely (it is not even an elementwise-kernel candidate).
- **Hand-rolling a FlatBuffers parser.** D-01 forbids it; use flatc-generated bindings.
- **Computing SHAP/importance without leaf weights.** All three algorithms divide by / weight on `LeafWeights`; absent weights give wrong baselines and NaNs (`count1==0` short-circuits silently mask the bug). Capture leaf weights first.
- **Using `FastExp` for Probability/LogProbability.** The Python oracle uses `std::exp` there; mixing in FastExp adds avoidable error. Only `Exponent` uses FastExp.
- **Assuming byte-identical `.cbm`.** D-03 explicitly targets semantic round-trip, not bytes; FlatBuffers builder field ordering and ModelInfo metadata (timestamps, training params) are uncontrolled.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| FlatBuffers (de)serialization | Custom wire-format parser | `flatc`-generated bindings + `flatbuffers` crate | Reimplements a stable binary format; high parity/maintenance risk (D-01) |
| JSON parse of `model.json` | New parser | Reuse `cb-oracle::model_json` + extend with `leaf_weights` | Already parses the upstream schema; doubles as round-trip oracle (D-04) |
| Float summation in apply/SHAP | Naive `iter().sum()` | `cb-core::sum_f64` / `sum_f32_in_f64` | D-08 grep gate; order-locked f64 accumulator for ≤1e-5 parity |
| Border binarization | New quantizer | `cb-data` border lookup (verify strict-`>` count) | Phase-2 GreedyLogSum borders + binarization already oracle-locked |
| sigmoid/exp | Custom | `f64::exp` for Probability/LogProbability; FastExp-equivalent only if matching Exponent | The Python oracle's exact functions are known (see Pattern 3) |

**Key insight:** For a parity rewrite, "don't hand-roll" means "don't reinterpret" — transcribe the cited upstream algorithm line-for-line and verify against the oracle, rather than re-deriving the math. The vendored C++ IS the spec.

## Runtime State Inventory

Not a rename/refactor/migration phase — this is additive feature work. **Omitting the migration table.** One structural carry-over worth flagging (not runtime state, but a model-representation change): the canonical `Model` gains a `leaf_weights` field and is re-homed `cb-train::Model` → `cb-model::Model`. Existing Phase-3 fixtures (`regression_skeleton`, `binclf_skeleton`, etc.) are unaffected (they assert splits/leaf-values/staged-approx, which are unchanged). New Phase-4 fixtures are additive.

## Common Pitfalls

### Pitfall 1: Missing leaf weights break SHAP and feature importance silently
**What goes wrong:** `cb-train::Model` has no `leaf_weights`. SHAP `subtreeWeights`/`meanValue` and `CalcEffect` all need them; without them you get zeros (the `count1==0||count2==0` and `subtreeWeights[0][0]==0` paths short-circuit) and a confidently-wrong all-zero importance/SHAP that may even pass a weak test.
**Why it happens:** Leaf weights are a training-time byproduct (`leafWeights[leafIndex] += rowWeight`, `approx_calcer.cpp:160`) that Phase 3 didn't need and didn't keep.
**How to avoid:** First Phase-4 task: capture per-leaf summed weights during/after training and add `leaf_weights: Vec<Vec<f64>>` (one inner vec per tree, length `2^depth`) to the canonical `Model`. For unweighted Phase-4 fixtures, a leaf weight == its training document count. Serialize in both `.cbm` (`LeafWeights:[double]` flat array) and `model.json` (`leaf_weights` per tree).
**Warning signs:** SHAP trailing column ≠ prediction; importance all zeros or NaN; `sum(shap) != prediction`.

### Pitfall 2: `model.json` `leaf_weights` flat layout vs per-tree
**What goes wrong:** In `.cbm` FlatBuffers, `LeafWeights:[double]` is a single flat array across all trees (offset per tree via `TreeFirstLeafOffsets`). In `model.json` it's nested per tree (`oblivious_trees[i].leaf_weights`). Mixing the layouts corrupts SHAP.
**Why it happens:** Two serializations, two shapes `[VERIFIED: model.fbs:35 flat; json_model_helpers.cpp:304 per-tree]`.
**How to avoid:** Store canonically as per-tree `Vec<Vec<f64>>`; flatten for `.cbm`, nest for JSON.
**Warning signs:** Off-by-tree SHAP after a `.cbm`↔JSON round-trip.

### Pitfall 3: `Exponent` prediction type uses FastExp (approximation), not `std::exp`
**What goes wrong:** Matching `Exponent` with `f64::exp` produces a tiny systematic difference vs the oracle (which uses the table-based `fast_exp` via AVX2/SSE2).
**Why it happens:** `CalcExponent` → `FastExpWithInfInplace` whose vectorized path is a 4-iteration table approximation `[VERIFIED: fast_exp.cpp:33-49]`. (Its scalar fallback uses `std::exp`, but x86_64 oracle machines take the vectorized path.)
**How to avoid:** Expect ≤1e-5 to absorb FastExp error for `Exponent`; if a fixture diverges, either (a) port the `fast_exp` table algorithm, or (b) generate the `Exponent` fixture and confirm the gap is < 1e-5 with plain `f64::exp`. Probability/LogProbability are SAFE with `f64::exp` (oracle uses `std::exp` there).
**Warning signs:** `Exponent` fixture diverges at ~1e-6–1e-7 while Probability passes exactly.

### Pitfall 4: `binClassLogitThreshold` / probability border
**What goes wrong:** `Class` output uses a logit threshold that defaults to 0 but becomes `-log(1/p - 1)` if a probability border is configured `[VERIFIED: eval_processing.cpp:32]`.
**How to avoid:** Phase-4 fixtures should not set a custom border (threshold = 0). Document the param so the Builder doesn't silently diverge if someone sets it.

### Pitfall 5: FlatBuffers `FormatVersion` string is a typo — must be reproduced
**What goes wrong:** Writing `"FlatbuffersModel_v1"` (corrected spelling) makes upstream reject our `.cbm` ("Unsupported model format").
**Why it happens:** Upstream's canonical string is `"FlabuffersModel_v1"` (missing the second `t`) `[VERIFIED: model.cpp:53]`.
**How to avoid:** Emit the exact upstream string. Add a test asserting the literal.

### Pitfall 6: `boost_from_average` → `scale_and_bias` mapping for serialization
**What goes wrong:** The Phase-3 `Model.bias` (starting approx from `boost_from_average`) must serialize into the FlatBuffers `Bias:double` / `MultiBias:[double]` and `model.json` `scale_and_bias = [scale, [bias,...]]` with `scale = 1` `[VERIFIED: model.fbs:41-43; json scale_and_bias = [1, [bias]] per cb-oracle parser]`. Apply re-adds it via `ApplyScaleAndBias`. Getting the bias into the wrong field, or double-counting it (in leaf values AND bias), breaks predictions.
**How to avoid:** Single source of truth: `bias` lives in `scale_and_bias`; leaf values are bias-free. Existing `cb-oracle::model_json::bias()` reads `scale_and_bias[1][0]` — match that.
**Warning signs:** Predictions off by exactly `bias`; or off by `2·bias`.

### Pitfall 7: NaN handling at apply time
**What goes wrong:** Float features with `HasNans` substitute a value before binarization (`AsFalse`/`AsTrue`/`AsIs` per `NanValueTreatment`) `[VERIFIED: quantization.h:104-110, features.fbs:5-9,17]`. Ignoring this diverges on any NaN-bearing column.
**How to avoid:** Phase-4 numeric fixtures are NaN-free per Phase-3 `numeric_tiny`; keep them NaN-free unless explicitly testing NaN, then honor `NanValueTreatment`. Carry the per-feature `HasNans`/`NanValueTreatment` into the model and serialize them.

## Code Examples

### `.cbm` framing (write)
```rust
// Source: catboost-master/.../model.cpp:1113-1163 (verified)
// magic + ui32 size + FlatBuffers TModelCore. No model parts in Phase 4.
fn write_cbm(model_core_fb: &[u8], out: &mut impl Write) -> Result<(), CatBoostError> {
    out.write_all(b"CBM1")?;                                  // model.cpp:41,49-51
    let size: u32 = u32::try_from(model_core_fb.len())
        .map_err(|_| CatBoostError::SchemaVersion("core > 4GiB".into()))?;
    out.write_all(&size.to_le_bytes())?;                      // util/ysaveload.h:277-284 (fixed ui32 LE)
    out.write_all(model_core_fb)?;                            // FB TModelCore with FormatVersion="FlabuffersModel_v1"
    Ok(())
}
```

### Oblivious leaf index (apply)
```rust
// Source: catboost-master/.../cpu/evaluator_impl.cpp:26-50 + quantization.h:130-138 (verified)
fn leaf_index(bin_values: &[u8], repacked: &[RepackedBin]) -> usize {
    let mut idx = 0usize;
    for (depth, rb) in repacked.iter().enumerate() {
        let feature_bin = bin_values[rb.feature_index as usize];   // strict-> count, see binarize
        let bit = (feature_bin >= rb.split_idx) as usize;          // XorMask=0 for NaN-free float
        idx |= bit << depth;                                       // forward bit order
    }
    idx
}
// binarize: bin = borders.iter().filter(|&&b| raw > b).count() as u8   // STRICT > (quantization.h:138)
```

### CrossEntropy / Logloss der1/der2 (identical math)
```rust
// Source: error_functions.cpp:304-336 CalcCrossEntropyDerRangeImpl (verified)
// p = 1 - 1/(1+exp(approx)) == sigmoid(approx)
// der1 = target - p          (target ∈ {0,1} for Logloss, ∈ [0,1] for CrossEntropy)
// der2 = -p*(1-p)
```

### Focal der1/der2
```rust
// Source: error_functions.h:1684-1709 TFocalError (verified). alpha∈(0,1), gamma>0.
// p  = 1/(1+exp(-approx));  p = clamp(p, 1e-13, 1-1e-13)
// at = (target==1) ? alpha : 1-alpha
// pt = (target==1) ? p : 1-p
// y  = 2*target - 1
// der1 = -( at*y*pow(1-pt, gamma) * (gamma*pt*log(pt) + pt - 1) )
// der2: u=at*y*pow(1-pt,gamma); du=-at*y*gamma*pow(1-pt,gamma-1);
//       v=gamma*pt*log(pt)+pt-1; dv=gamma*log(pt)+gamma+1;
//       der2 = -( (du*v + u*dv) * y * (pt*(1-pt)) )
// NOTE: uses std exp/pow/log — Rust f64 matches directly.
```

### SHAP feature-path extend (the polynomial weight machinery)
```rust
// Source: shap_values.cpp:44-64 ExtendFeaturePath (verified) — transcribe exactly
fn extend(old: &[Elem], zero_frac: f64, one_frac: f64, feature: i32) -> Vec<Elem> {
    let l = old.len();
    let mut p = old.to_vec();
    p.push(Elem { feature, zero_frac, one_frac, weight: if l == 0 { 1.0 } else { 0.0 } });
    for i in (0..l).rev() {
        let wi = p[i].weight;
        p[i + 1].weight += one_frac * wi * (i as f64 + 1.0) / (l as f64 + 1.0);
        p[i].weight = zero_frac * wi * (l as f64 - i as f64) / (l as f64 + 1.0);
    }
    p
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| n/a (greenfield) | flatc-generated committed bindings | this phase | Deterministic builds, no flatc at consumer build time |
| Model without weights (Phase 3) | Model with per-leaf weights | this phase | Enables SHAP + importance without a reference dataset |

**Deprecated/outdated:** None relevant. The `.fbs` schema and apply algorithm are stable in upstream 1.2.10. The `CalcSigmoid(TConstArrayRef)` `// TODO uncomment` comment in `eval_processing.h:104` indicates upstream intends to migrate the vector path to FastExp eventually — but in 1.2.10 it still uses `std::exp`, which is what the pinned oracle produces.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `flatbuffers` crate `25.12.19` reads schemas authored for flatc `24.3.25` (wire-format stable) | Standard Stack | LOW — FlatBuffers format stability is well-documented; if wrong, pin flatc and crate to matching majors |
| A2 | `Exponent` ≤1e-5 parity holds with `f64::exp` despite oracle's FastExp | Pitfall 3 | MEDIUM — if the gap exceeds 1e-5, must port the `fast_exp` table; verify when generating the Exponent fixture |
| A3 | For numeric-only models, `binFeatureCombinationClass` is identity (each float bin-feature = its own feature) in SHAP | Pattern 4 | LOW — true when no CTR/combination features exist (Phase 4 has none); verify in the first SHAP fixture |
| A4 | Unweighted leaf weight == training document count per leaf | Pitfall 1 | LOW — direct from `leafWeights[idx] += rowWeight` with rowWeight=1; verify against an upstream `model.json` `leaf_weights` |
| A5 | `flatc` can be installed in CI at version ≥ 24.3.25 | Environment Availability | MEDIUM — if CI can't install flatc, committed bindings still work (D-01), only regeneration is blocked; fallback = `planus` |

**These five assumptions need confirmation during planning/execution.** A2 and A5 are the material ones — fold them into Wave-0 verification.

## Open Questions

1. **Does the Builder need to recompute leaf weights, or capture them during the boosting loop?**
   - What we know: weights are accumulated per leaf during `CalcLeafValues` (`approx_calcer.cpp:160`); the Phase-3 trainer computes leaf membership already.
   - What's unclear: cheapest insertion point in the existing `cb-train::train` loop.
   - Recommendation: capture during training (the leaf-index assignment already runs); store per-tree `Vec<f64>` alongside `leaf_values`. Recomputing post-hoc would require re-binarizing the training pool.

2. **`flatc` provisioning for CI.**
   - What we know: flatc is not installed locally; D-01 wants committed bindings (so consumers don't need flatc).
   - Recommendation: install flatc once (dev), commit `generated/`, and gate regeneration behind a documented manual step. Do NOT add flatc to the per-build path. (A5.)

3. **MultiBias vs Bias field for 1-dim models.**
   - What we know: schema has both `Bias:double` and `MultiBias:[double]` (`model.fbs:42-43`).
   - Recommendation: for 1-dim (binclf/regression) write the single `Bias`; confirm by inspecting an upstream-produced 1.2.10 `.cbm` for a 1-dim model (the load-parity fixture will reveal which field upstream populates).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `flatc` (FlatBuffers compiler) | D-01 binding generation | ✗ | — | Commit generated bindings (one-time gen on a machine with flatc); or `planus` pure-Rust codegen |
| `flatbuffers` crate | `.cbm` runtime | ✓ (crates.io) | 25.12.19 | — |
| Python `catboost==1.2.10` | D-13 fixture generation | ✗ (not importable) | — | Generator runs offline; fixtures are committed and NOT regenerated in CI (matches D-13). Generator venv exists at `crates/cb-oracle/generator/.venv` |
| Python 3.12 | generator | ✓ | 3.12.3 | — |
| Rust (workspace) | all | ✓ | latest stable | — |
| `serde`/`serde_json`/`ndarray`/`ndarray-npy`/`thiserror` | JSON + fixtures + errors | ✓ | workspace-pinned | — |

**Missing dependencies with no fallback:** none (all have a viable path).
**Missing dependencies with fallback:**
- `flatc`: install once + commit bindings (preferred), or switch to `planus`. The planner should add an explicit "install flatc and generate bindings" task (or a `checkpoint:human-verify` if the install must happen on the user's machine).
- Python `catboost`: fixtures must be (re)generated where it can be installed; the existing generator (`gen_fixtures.py`) shows the exact pattern (`thread_count=1`, fixed `random_seed`, `save_model(format='json')`, `predict(prediction_type=…)`, `get_feature_importance(type=…, data=Pool)`). New fixtures: an upstream `.cbm` (`save_model(format='cbm')`), SHAP matrix (`get_feature_importance(type='ShapValues', data=Pool)`), PredictionValuesChange + Interaction, and per-prediction-type outputs.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` (workspace) + `approx` 0.5 for float asserts + `cb-oracle::compare_stage` (≤1e-5 gate) |
| Config file | none (cargo built-in); CI clippy gate is `--lib` per Phase-1 |
| Quick run command | `cargo test -p cb-model` (or `-p catboost-rs`) |
| Full suite command | `cargo test --workspace` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MODEL-01 | our save→load reproduces Model; load upstream `.cbm` apply ≤1e-5; upstream loads ours ≤1e-5 | integration (oracle) | `cargo test -p cb-model cbm` | ❌ Wave 0 (needs upstream `.cbm` fixture + tests) |
| MODEL-02 | apply runs with no GPU toolchain; predictions ≤1e-5 | integration | `cargo test -p cb-model apply` | ❌ Wave 0 |
| MODEL-03 | PredictionValuesChange + Interaction ≤1e-5 | integration (oracle) | `cargo test -p cb-model fstr` | ❌ Wave 0 (needs importance fixtures) |
| MODEL-04 | per-object SHAP matrix ≤1e-5; `sum(shap)==prediction` | integration (oracle) | `cargo test -p cb-model shap` | ❌ Wave 0 (needs SHAP `.npy` fixture) |
| MODEL-06 | JSON export round-trips via `cb-oracle::model_json`; matches upstream schema | integration | `cargo test -p cb-model json` | ⚠️ parser exists; needs `leaf_weights` extension + export tests |
| LOSS-01 | Logloss/CrossEntropy/Focal train ≤1e-5 (splits/leaf/staged) | integration (oracle) | `cargo test -p cb-train loss` | ❌ Wave 0 (CrossEntropy + Focal fixtures) |
| LOSS-06 | RawFormulaVal/Probability/LogProbability/Class/Exponent ≤1e-5 | integration (oracle) | `cargo test -p cb-model predict` | ❌ Wave 0 (per-type prediction fixtures) |
| RAPI-01 | `CatBoostBuilder...fit(&pool)->Model`, predict end-to-end | integration | `cargo test -p catboost-rs builder` | ❌ Wave 0 |
| RAPI-02 | `CatBoostError` variants + `#[from] CbError`; Result equality | unit | `cargo test -p catboost-rs error` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test -p <crate>` for the touched crate + `cargo clippy --lib` (restriction lints).
- **Per wave merge:** `cargo test --workspace`.
- **Phase gate:** full suite green + the five success criteria oracle-locked before `/gsd-verify-work`. **Disk caution (from STATE.md): `cargo test --workspace` pulls cubecl-cpu's heavy MLIR dep — watch disk headroom.**

### Wave 0 Gaps
- [ ] `crates/cb-model/src/cbm_test.rs` — covers MODEL-01 (round-trip + bidirectional interop)
- [ ] `crates/cb-model/src/apply_test.rs` — covers MODEL-02 (apply ≤1e-5)
- [ ] `crates/cb-model/src/shap_test.rs` — covers MODEL-04 (SHAP matrix + local accuracy)
- [ ] `crates/cb-model/src/fstr_test.rs` — covers MODEL-03 (PredictionValuesChange + Interaction)
- [ ] `crates/cb-model/src/json_test.rs` — covers MODEL-06
- [ ] `crates/cb-model/src/predict_test.rs` — covers LOSS-06
- [ ] `crates/catboost-rs/src/builder_test.rs`, `error_test.rs` — covers RAPI-01/02
- [ ] New committed fixtures (generated offline, D-13): upstream `.cbm` (1-dim binclf + regression); SHAP `.npy`; PredictionValuesChange/Interaction `.npy`; per-prediction-type `.npy`; CrossEntropy + Focal training fixtures (model.json + staged.npy + predictions.npy)
- [ ] Leaf-weights capture in `cb-train::train` (structural prerequisite for SHAP/fstr) — first task
- [ ] `cb-oracle::model_json` extension: add `leaf_weights` per tree

*Existing infra reused: `cb-oracle::compare_stage` (≤1e-5), `.npy` fixture readers, `model_json` parser, generator scaffold (`gen_fixtures.py`).*

## Security Domain

`security_enforcement: true` in config, but this phase has **no authentication, session, access-control, network, or untrusted-input-from-the-internet surface** — it is a numerical library reading model files the user already trusts. The one relevant control is **input validation on deserialization** (a `.cbm`/JSON file could be malformed or hostile).

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes | Validate `.cbm` magic, size bounds, and FlatBuffers buffer before trusting it; upstream uses `flatbuffers::Verifier` with depth/table limits `[VERIFIED: model.cpp:1191]`. The Rust `flatbuffers` crate's generated `root_as_*` performs verification — use the verifying accessor, not unchecked. Return typed `CatBoostError::Deserialize`/`SchemaVersion`, never panic/`unwrap` (CLAUDE.md + lint gate). |
| V6 Cryptography | no | — (no crypto; never hand-roll) |

### Known Threat Patterns for {Rust model-file parsing}

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malformed `.cbm` size field → huge allocation / OOB read | Denial of Service | Bound the declared core size against actual file length before allocating; use the verifying FlatBuffers accessor (upstream caps depth=64, tables=256M, model.cpp:1191) |
| Truncated / corrupt FlatBuffers buffer | Tampering | `flatbuffers` crate verifier rejects; map failure to `CatBoostError::Deserialize` |
| Untrusted `model.json` (deeply nested / huge) | DoS | `serde_json` is safe by default; no `unsafe`; lints forbid `indexing_slicing`/`unwrap` so OOB on malformed arrays returns typed errors |
| Integer overflow on offsets/sizes | Tampering | Use checked conversions (`u32::try_from`, `usize` bounds) — already enforced by the `indexing_slicing = deny` lint |

## Project Constraints (from CLAUDE.md)

- **`thiserror` in libraries; `anyhow` banned** from library crates (CI grep). `cb-model` and `catboost-rs` use `thiserror`. (`cb-model` stub currently notes "anyhow intentionally absent" — keep it that way.)
- **`unwrap()`/`expect()`/`panic`/`indexing_slicing` denied** in production (workspace clippy). Apply/SHAP index-heavy code must use checked access or restructure (e.g., iterators, `get`) — this is non-trivial for the SHAP recursion; budget for it.
- **Source/test separation mandatory:** dedicated `*_test.rs`, no inline `#[cfg(test)]`; test-lint exemption via `#![cfg_attr(test, allow(...))]` (already the pattern).
- **Memory efficiency first-class:** prefer zero-copy reads of `.cbm` (FlatBuffers enables this), minimize allocations in apply (reuse buffers across documents, mirror QuantizedPool SoA reuse).
- **Latest crate versions:** `flatbuffers = "25.12.19"` (latest stable, verified).
- **Builder pattern on the Rust side:** `CatBoostBuilder` (D-05).
- **No C API / FFI:** pure Rust; do NOT link upstream `libcatboostmodel` — reimplement apply in Rust.
- **CubeCL rules (AGENTS.md):** apply path stays off cubecl (MODEL-02). If any cubecl build error arises (it shouldn't in this phase), consult `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/cubecl_error_guideline.md` before fixing.

## Sources

### Primary (HIGH confidence — vendored upstream C++, the literal oracle)
- `catboost-master/catboost/libs/model/model.cpp:41,49-53,1113-1285` — `.cbm` magic, format string, Save/Load framing
- `catboost-master/util/ysaveload.h:277-296` — SaveSize/LoadSize (fixed ui32 LE, not varint)
- `catboost-master/catboost/libs/model/flatbuffers/{model,features,ctr_data}.fbs` — `.cbm` schema (TModelCore, TModelTrees with LeafValues/LeafWeights/Bias)
- `catboost-master/catboost/libs/model/cpu/evaluator_impl.cpp:16-52,155-172` — scalar apply (CalcIndexesBasic, CalculateLeafValues)
- `catboost-master/catboost/libs/model/cpu/quantization.h:81-140` — apply-time float binarization (strict `>` count)
- `catboost-master/catboost/libs/model/model.cpp:560-573` + `model.h:72` — TRepackedBin construction, MAX_VALUES_PER_BIN=254
- `catboost-master/catboost/libs/model/eval_processing.h:92-260` + `catboost/libs/eval_result/eval_helpers.cpp:352-496` — prediction-type transforms (Probability/LogProbability/Class/Exponent), `std::exp` vs FastExp
- `catboost-master/library/cpp/fast_exp/fast_exp.cpp:33-49` — FastExp table approximation (Exponent)
- `catboost-master/catboost/libs/fstr/shap_values.cpp:26-320,493-548,1030-1055` — TreeSHAP recursion + bias column + local accuracy
- `catboost-master/catboost/libs/fstr/shap_prepared_trees.cpp:25-222` — subtree weights + mean values (need LeafWeights)
- `catboost-master/catboost/libs/fstr/feature_str.h:190-284` + `calc_fstr.cpp:94-441` — PredictionValuesChange (CalcEffect) + Interaction
- `catboost-master/catboost/private/libs/algo_helpers/error_functions.{h,cpp}` — CrossEntropy (304-336), Focal (1669-1711)
- `catboost-master/catboost/private/libs/algo/approx_calcer.cpp:154-160` — leaf weight accumulation
- `catboost-master/catboost/libs/model/model_export/json_model_helpers.cpp:160-526` — model.json field names/structure
- `catboost-master/catboost/private/libs/options/enums.h:253-295` — EPredictionType, EFstrType, ECalcTypeShapValues

### Secondary (HIGH — existing project code)
- `crates/cb-train/src/boosting.rs` — current Model (no leaf_weights), Split, BoostParams
- `crates/cb-compute/src/runtime.rs` — Loss enum (Rmse/Logloss/Mae)
- `crates/cb-oracle/src/{model_json.rs,compare.rs,fixture.rs}` — JSON parser, compare_stage, fixture infra
- `crates/cb-oracle/generator/gen_fixtures.py` — D-13 fixture generation pattern

### Tertiary (verified via tool)
- `cargo search flatbuffers` → `25.12.19` latest; `gsd-tools package-legitimacy check --ecosystem crates flatbuffers` → OK (1.27M/wk, github.com/google/flatbuffers)
- `flatc --version` → not installed; `python3 -c "import catboost"` → not importable

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — flatbuffers crate verified (cargo search + legitimacy seam); flatc requirement explicit
- `.cbm` framing: HIGH — read Save/Load + SaveSize line-by-line
- CPU apply: HIGH — scalar reference algorithm transcribed from evaluator_impl + quantization.h
- Prediction transforms: HIGH — exact functions identified incl. the std::exp-vs-FastExp distinction
- SHAP: HIGH — full recursion + prepared-trees read; one MEDIUM assumption (combinationClass identity) to verify in first fixture
- Feature importance: HIGH — CalcEffect + Interaction transcribed
- Losses: HIGH — CrossEntropy + Focal der1/der2 read directly
- Leaf-weights gap: HIGH — confirmed absent in cb-train::Model and required by all fstr/SHAP paths
- Exponent FastExp parity: MEDIUM — A2, verify ≤1e-5 holds with f64::exp when generating the fixture

**Research date:** 2026-06-13
**Valid until:** 2026-07-13 (stable — vendored upstream is pinned at 1.2.10 and does not move; only the `flatbuffers` crate version could drift)
