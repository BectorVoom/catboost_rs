---
phase: 04-model-serialization-shap-rust-api-first-full-oracle-lock
verified: 2026-06-14T08:00:00Z
status: passed
score: 5/5 must-haves verified
overrides_applied: 0
---

# Phase 4: Model, Serialization, SHAP & Rust API Verification Report

**Phase Goal:** The first complete vertical slice — train -> serialize -> load -> predict/explain — is oracle-locked end-to-end for numeric binary classification and regression, exposed through the public Rust Builder API.
**Verified:** 2026-06-14T08:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Native `.cbm` (FlatBuffers) serialization round-trips, and an upstream CatBoost 1.2.10 model can be loaded and applied ≤1e-5 | VERIFIED | `cbm_oracle_test.rs`: `cbm_round_trip_reproduces_model`, `cbm_load_upstream_binclf_applies_within_tol`, `cbm_load_upstream_regression_applies_within_tol` — all 9 cbm tests pass. `cbm.rs` has CBM1 magic, `FlabuffersModel_v1` constant (correct upstream typo), size-bound read, verifying `root_as_tmodel_core` (no `_unchecked`). |
| 2 | CPU inference/apply path runs independently of any GPU toolchain; JSON model export available | VERIFIED | `apply.rs` imports no cubecl/cb-backend (grep confirms empty). `json.rs` implements `save_json`/`load_json` with upstream schema. `json_oracle_test.rs`: upstream binclf and regression loads apply ≤1e-5. |
| 3 | SHAP values (Regular EShapCalcType) and feature importance (PredictionValuesChange, Interaction) match upstream ≤1e-5 | VERIFIED | `shap_oracle_test.rs`: `shap_matrix_matches_upstream_within_tol` + local-accuracy invariant (`Σshap == predict_raw`) pass for every object. `fstr_oracle_test.rs`: `prediction_values_change_matches_upstream_within_tol`, `interaction_matches_upstream_within_tol`, `prediction_values_change_sums_to_100` all pass. Real `.npy` fixtures in `cb-oracle/fixtures/feature_importance/`. LossFunctionChange deliberately absent from `fstr.rs` (D-12; grep confirms). MODEL-03 correctly marked `[~]` (partial) in REQUIREMENTS.md. |
| 4 | Binary classification (Logloss, CrossEntropy, Focal) and prediction types (Probability, LogProbability, Class, RawFormulaVal, Exponent) produce outputs matching upstream ≤1e-5 | VERIFIED | `predict_oracle_test.rs`: all 5 prediction types pass vs `prediction_types/*.npy`. `loss_oracle_test.rs` (cb-train): `loss_oracle_cross_entropy` and `loss_oracle_focal` pass ≤1e-5. `loss.rs` cites `error_functions.cpp:317-340` / `error_functions.h:1684-1709`. LOSS-06 partial (uncertainty types D-10 deferred to Phase 6) honestly marked `[~]` in REQUIREMENTS.md. |
| 5 | The `catboost-rs` Builder API (`CatBoostBuilder::new()...fit(&pool) -> Model`, predict) drives a full numeric-only binclf + regression train→serialize→predict oracle pass ≤1e-5 | VERIFIED | `builder_oracle_test.rs`: `builder_binclf_full_cycle` and `builder_regression_full_cycle` both pass — Rust-to-Rust round-trip determinism AND upstream 1.2.10 `predictions.npy` oracle (`compare_stage` ≤1e-5) run unconditionally (no `#[ignore]`). `CatBoostBuilder` has `new()`/setters/`fit()`; `Model` facade exposes `predict_with`/`predict`/`predict_proba`/`save_cbm`/`load_cbm`/`save_json`/`load_json`/`shap_values`/`feature_importance`. Public API used exclusively in oracle test. |

**Score:** 5/5 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/cb-model/src/model.rs` | Canonical `Model { oblivious_trees, bias, float_feature_borders }` with `leaf_weights` | VERIFIED | Struct present, 96 lines, has all fields, `ObliviousTree` with `leaf_weights: Vec<f64>` |
| `crates/cb-model/src/apply.rs` | Pure-Rust CPU apply path, GPU-toolchain-free | VERIFIED | 109 lines, strict `> b` binarization, `sum_f64`, no cubecl/cb-backend import |
| `crates/cb-model/src/predict.rs` | PredictionType enum + transforms | VERIFIED | 103 lines, all 5 deterministic variants, two-column LogProbability/Probability |
| `crates/cb-model/src/cbm.rs` | `.cbm` FlatBuffers framing | VERIFIED | 445 lines, CBM1 magic, `FlabuffersModel_v1`, verifying accessor, size-bounded read |
| `crates/cb-model/src/json.rs` | model.json export/import on upstream schema | VERIFIED | 251 lines, nested leaf_weights per tree, `scale_and_bias=[1,[bias]]` |
| `crates/cb-model/src/error.rs` | ModelError (thiserror) with Deserialize/SchemaVersion/Io/Json/Core | VERIFIED | Exists; mirrored in catboost-rs CatBoostError |
| `crates/cb-model/src/shap.rs` | Regular TreeSHAP with prepared-trees and local accuracy | VERIFIED | 515 lines, `subtree_weights`, `mean_value`, `sum_f64`, checked `.get/.get_mut` throughout |
| `crates/cb-model/src/fstr.rs` | PredictionValuesChange + Interaction (no LossFunctionChange) | VERIFIED | 203 lines, `leaf_weights` used, `count==0` guards, `LossFunctionChange` absent |
| `crates/cb-model/src/generated/model_generated.rs` | flatc-generated FlatBuffers bindings | VERIFIED | Machine-generated header present (`// automatically generated by the FlatBuffers compiler`), compiles with `flatbuffers 25.12.19` |
| `crates/catboost-rs/src/builder.rs` | `CatBoostBuilder` with `new()`/setters/`fit()` | VERIFIED | 266 lines, all 14+ chained setters, `fit(&pool) -> Result<Model, CatBoostError>` |
| `crates/catboost-rs/src/model.rs` | Model facade with predict/save/load/shap/importance | VERIFIED | 187 lines, all 9 required methods, `feature_columns()` FeatureMismatch guard |
| `crates/catboost-rs/src/error.rs` | Public CatBoostError (thiserror, #[from] CbError) | VERIFIED | 71 lines, `Train(#[from] CbError)`, `Model(#[from] ModelError)`, `Io`, `Deserialize`, `SchemaVersion`, `FeatureMismatch`; no `Clone`/`PartialEq` |
| `crates/cb-train/tests/leaf_weights_oracle_test.rs` | Leaf weights oracle lock | VERIFIED | 147 lines, 2/2 tests pass |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `catboost-rs/src/builder.rs` | `cb_train::train` | `fit(&pool)` maps builder fields to BoostParams | WIRED | `fit` calls `cb_train::train` with computed borders + BoostParams |
| `catboost-rs/src/model.rs` | `cb_model` | predict/save/load/shap/importance delegation | WIRED | All methods delegate; `feature_columns` guard guards predict/shap |
| `catboost-rs/src/error.rs` | `cb_core::CbError` | `#[from] arm` | WIRED | `Train(#[from] cb_core::CbError)` confirmed in error.rs |
| `crates/cb-model/src/cbm.rs` | `crates/cb-model/src/generated` | flatc-generated TModelCore bindings | WIRED | `root_as_tmodel_core` import from generated module |
| `crates/cb-model/src/apply.rs` | `cb_core::sum_f64` | leaf-sum accumulation | WIRED | `use cb_core::sum_f64;` + called in `predict_raw_one` |
| `crates/cb-model/src/apply.rs` | `Model.float_feature_borders` | strict `value > border` binarization | WIRED | `borders.iter().filter(|&&b| raw > b).count()` confirmed |
| `crates/cb-model/src/shap.rs` | `Model.leaf_weights` | subtree-weight + mean-value baselines | WIRED | `calc_subtree_weights(leaf_weights, ...)` + `mean_value` confirmed |
| `crates/cb-model/src/fstr.rs` | `Model.leaf_weights` | leaf-pair weighted variance | WIRED | `tree.leaf_weights.get(leaf_idx)` confirmed |
| `crates/cb-train/src/boosting.rs` | `ObliviousTree.leaf_weights` | per-leaf row-weight accumulation | WIRED | `accumulate_leaf_weights` called in train loop, stored on ObliviousTree |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|-------------------|--------|
| `builder_oracle_test.rs` / `Model.predict_with` | `predictions` | `cb_train::train` -> `cb_model::predict_raw` | Yes — real boosting over real `numeric_tiny` fixtures | FLOWING |
| `cbm_oracle_test.rs` / `load_cbm` | Model fields | Real upstream catboost 1.2.10 `model.cbm` fixture | Yes — 4 files committed in `cb-oracle/fixtures/model_serde/` | FLOWING |
| `shap_oracle_test.rs` / `shap_values` | SHAP matrix | `leaf_weights` from trained model; `shap_values.npy` fixture | Yes — real fixture, real model data, local-accuracy invariant passes | FLOWING |
| `fstr_oracle_test.rs` / `prediction_values_change` | importance scores | `leaf_weights` from trained model; `prediction_values_change.npy` | Yes — real fixture, Σ=100 verified | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| cb-model 29 tests (apply/cbm/fstr/json/predict/shap) | `cargo test -p cb-model` | 29/29 pass, 0 failed, 0 ignored | PASS |
| catboost-rs 7 tests (5 error unit + 2 builder oracle integration) | `cargo test -p catboost-rs` | 7/7 pass, 0 failed, 0 ignored | PASS |
| cb-train leaf weights oracle (2 tests) | `cargo test -p cb-train --test leaf_weights_oracle_test` | 2/2 pass | PASS |
| cb-train CrossEntropy + Focal oracle (4 tests) | `cargo test -p cb-train --test loss_oracle_test` | 4/4 pass | PASS |
| Workspace source compilation | `cargo check --workspace` | `Finished dev profile in 0.14s` | PASS |

### Probe Execution

Step 7c: No phase-declared probes found. No `probe-*.sh` scripts in `scripts/`. SKIPPED.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| MODEL-01 | 04-01, 04-03 | Native `.cbm` FlatBuffers serialization — save/load, cross-version compatible | SATISFIED | `cbm.rs` + `cbm_oracle_test.rs` upstream 1.2.10 binclf + regression load ≤1e-5; round-trip; malformed-input typed errors |
| MODEL-02 | 04-02 | CPU inference/apply path independent of GPU toolchain | SATISFIED | `apply.rs` has no cubecl/cb-backend import; oracle-locked via `apply_oracle_test.rs` |
| MODEL-03 | 04-04 | Feature importance — PVC + Interaction (LossFunctionChange deferred D-12) | PARTIALLY SATISFIED | PVC + Interaction oracle-locked ≤1e-5. LossFunctionChange absent (D-12). REQUIREMENTS.md correctly marks `[~]`, ROADMAP correctly states PARTIAL. Phase 4 note and D-12 are consistent. |
| MODEL-04 | 04-04 | SHAP values (Regular EShapCalcType) | SATISFIED | `shap.rs` + `shap_oracle_test.rs` — oracle ≤1e-5 + local-accuracy invariant |
| MODEL-06 | 04-03 | JSON model export (interop minimum) | SATISFIED | `json.rs` + `json_oracle_test.rs` — round-trip + upstream load ≤1e-5 |
| LOSS-01 | 04-02 | Binary classification — Logloss, CrossEntropy, Focal | SATISFIED | `cb-compute/src/loss.rs` CrossEntropy/Focal der1/der2 with `error_functions` citations; oracle-locked 4/4 in `loss_oracle_test.rs` |
| LOSS-06 | 04-02 | Prediction types — 5 deterministic types (uncertainty deferred D-10) | PARTIALLY SATISFIED | RawFormulaVal/Probability/LogProbability/Class/Exponent oracle-locked vs `prediction_types/*.npy`. Uncertainty types (RMSEWithUncertainty/VirtEnsembles/TotalUncertainty) deferred to Phase 6 per D-10. Honestly marked `[~]` in REQUIREMENTS.md. |
| RAPI-01 | 04-05 | Rust Builder-pattern public API — `CatBoostBuilder::new()...fit(&pool) -> Model`, predict | SATISFIED | `builder.rs` + `model.rs` in catboost-rs; full method set verified; oracle 2/2 |
| RAPI-02 | 04-05 | Typed thiserror error enum across the public surface | SATISFIED | `error.rs` in catboost-rs — all 6 variants; `#[from] CbError`; no Clone/PartialEq; 5/5 unit asserts |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `crates/cb-model/src/generated/*.rs` | Many | `unwrap()` in unsafe flatc-generated accessors | INFO | These are machine-generated FlatBuffers accessor stubs (standard flatbuffers Rust codegen pattern); `unwrap()` is inside `unsafe` blocks and follows the FlatBuffers crate contract; production code in `cbm.rs` uses verifying accessors only. Lint allows properly declared in `lib.rs`. NOT a production code violation. |

No `TBD`, `FIXME`, or `XXX` markers found in any Phase 4 production source files. No `TODO`/`HACK`/`PLACEHOLDER` found. No stub return values in production paths. All floats accumulated via `cb_core::sum_f64` in modified files (D-08 gate confirmed by grep).

### Human Verification Required

None. All must-haves are mechanically verifiable and were verified.

### Deferred Items

Items honestly deferred per documented design decisions, addressed in later phases:

| # | Item | Addressed In | Evidence |
|---|------|-------------|----------|
| 1 | LossFunctionChange feature importance (MODEL-03 sub-item) | Phase 6 | ROADMAP Phase 6 Success Criteria 5: "feature selection (recursive by PredictionValuesChange/LossFunctionChange/ShapValues)" + FEAT-05 requirement |
| 2 | Uncertainty prediction types (LOSS-06 sub-items: RMSEWithUncertainty, VirtEnsembles, TotalUncertainty) | Phase 6 | ROADMAP Phase 6 SC 4: "Uncertainty estimation (RMSEWithUncertainty, virtual ensembles)" + LOSS-08 requirement |

Both deferred items are explicitly documented in their requirement entries (`[~]` status) and in the ROADMAP Phase 4 note: "MODEL-03 is only PARTIALLY delivered this phase" and in each plan's SUMMARY. No false `[x]` completion marks.

### Gaps Summary

No gaps. All 5 ROADMAP Success Criteria are VERIFIED against the codebase:

1. **SC1 (MODEL-01):** `.cbm` round-trips and upstream 1.2.10 load works ≤1e-5. 9/9 cbm tests pass.
2. **SC2 (MODEL-02/MODEL-06):** CPU apply path is GPU-toolchain-free (grep confirmed). JSON export round-trips through cb-oracle parser ≤1e-5.
3. **SC3 (MODEL-04/MODEL-03):** SHAP matrix oracle-locked ≤1e-5 with local accuracy. PVC + Interaction oracle-locked ≤1e-5. LossFunctionChange deliberately deferred (D-12) and honestly marked partial — not a false claim of completion.
4. **SC4 (LOSS-01/LOSS-06):** CrossEntropy, Focal der1/der2 oracle-locked. All 5 deterministic prediction types oracle-locked. Uncertainty types deferred (D-10) and honestly partial.
5. **SC5 (RAPI-01/RAPI-02):** Builder API with full method set. End-to-end binclf + regression train->serialize->load->predict unconditionally oracle-locked ≤1e-5 vs upstream 1.2.10. 7/7 catboost-rs tests pass.

The MODEL-03 PARTIAL delivery is **correctly tracked** as `[~]` in REQUIREMENTS.md, stated in the ROADMAP Phase 4 Note field, and documented in the 04-04 SUMMARY. It is NOT falsely marked `[x]`.

---

_Verified: 2026-06-14T08:00:00Z_
_Verifier: Claude (gsd-verifier)_
