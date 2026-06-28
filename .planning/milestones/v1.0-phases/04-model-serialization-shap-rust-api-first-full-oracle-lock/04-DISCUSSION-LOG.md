# Phase 4: Model, Serialization, SHAP & Rust API (First Full Oracle Lock) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-13
**Phase:** 4-Model, Serialization, SHAP & Rust API (First Full Oracle Lock)
**Areas discussed:** .cbm serialization strategy, Public Builder API shape, First-lock losses & predict types, SHAP & feature-importance scope

---

## .cbm serialization strategy

### Q1 — FlatBuffers tooling
| Option | Description | Selected |
|--------|-------------|----------|
| flatbuffers crate + flatc-generated | flatbuffers Rust crate; flatc-generated bindings from vendored .fbs, committed; zero-copy reads | ✓ |
| flatbuffers crate, hand-written schema | flatbuffers crate but hand-transcribe .fbs structs | |
| Fully hand-rolled reader/writer | Parse/emit FlatBuffers bytes manually | |

**User's choice:** flatbuffers crate + flatc-generated.

### Q2 — Cross-version load scope
| Option | Description | Selected |
|--------|-------------|----------|
| Pin to 1.2.10 only | Load .cbm from the pinned oracle version, apply ≤1e-5 | ✓ |
| Forward-compatible best-effort | Tolerate other schema versions, no fixtures | |
| Multi-version oracle fixtures | Commit .cbm from several versions | |

**User's choice:** Pin to 1.2.10 only.

### Q3 — Serialization correctness bar
| Option | Description | Selected |
|--------|-------------|----------|
| Semantic round-trip + interop | ours→ours exact; load upstream→apply ≤1e-5; upstream loads ours→predict ≤1e-5 | ✓ |
| Byte-identical output too | Plus exact byte match to upstream | |
| Round-trip only (no upstream interop) | Only our save→load | |

**User's choice:** Semantic round-trip + bidirectional interop (no byte-identity).

### Q4 — JSON export target
| Option | Description | Selected |
|--------|-------------|----------|
| Match upstream model.json schema | Reuse cb-oracle model_json parser as round-trip oracle | ✓ |
| Self-defined minimal JSON | catboost-rs-native JSON | |
| You decide | Decide during research | |

**User's choice:** Match upstream model.json schema.

---

## Public Builder API shape

### Q1 — Builder structure
| Option | Description | Selected |
|--------|-------------|----------|
| Unified CatBoostBuilder | One builder; loss determines task | ✓ |
| Typed Classifier/Regressor builders | Separate sklearn-like builders | |
| Unified builder + thin typed wrappers | Core builder + convenience wrappers | |

**User's choice:** Unified CatBoostBuilder.

### Q2 — Predict API
| Option | Description | Selected |
|--------|-------------|----------|
| predict(pool, PredictionType) | Single enum-selected entry point | |
| Separate typed methods | predict_raw/predict_proba/predict_class | |
| Both: enum core + shorthands | Enum core PLUS predict_proba()/predict() shorthands | ✓ |

**User's choice:** Both — enum core + shorthands.

### Q3 — Model API surface (save/load + explainability)
| Option | Description | Selected |
|--------|-------------|----------|
| Methods on Model | save_cbm/load_cbm/save_json/load_json + shap_values/feature_importance on Model | ✓ |
| Free functions / separate module | Standalone serialize::/fstr:: functions | |
| You decide | Decide during planning | |

**User's choice:** Methods on Model.

### Q4 — Public error type
| Option | Description | Selected |
|--------|-------------|----------|
| Public enum wrapping CbError | catboost-rs CatBoostError with new variants + #[from] CbError | ✓ |
| Extend cb-core::CbError | Add variants to core (risks Clone/PartialEq/Eq) | |
| You decide | Decide during planning | |

**User's choice:** Public CatBoostError enum wrapping CbError.

---

## First-lock losses & predict types

### Q1 — Binary-clf losses trained this phase
| Option | Description | Selected |
|--------|-------------|----------|
| Logloss only | Lock with Logloss + RMSE; defer CrossEntropy/Focal | |
| Logloss + CrossEntropy | Add CrossEntropy; defer Focal | |
| All three (Logloss + CrossEntropy + Focal) | Complete LOSS-01 (Focal needs γ-weighted ders + oracle) | ✓ |

**User's choice:** All three.

### Q2 — Prediction types in scope
| Option | Description | Selected |
|--------|-------------|----------|
| Core types; defer uncertainty | RawFormulaVal/Probability/LogProbability/Class/Exponent; uncertainty → Phase 6 | ✓ |
| Core types only (minimal) | RawFormulaVal/Probability/Class only | |
| Attempt all LOSS-06 types | Include uncertainty types now | |

**User's choice:** Core types; defer uncertainty to Phase 6.

---

## SHAP & feature-importance scope

### Q1 — LossFunctionChange in scope?
| Option | Description | Selected |
|--------|-------------|----------|
| Defer LossFunctionChange | Ship PredictionValuesChange + Interaction + Regular SHAP | ✓ |
| Include LossFunctionChange now | All three importances | |
| You decide | Decide during research | |

**User's choice:** Defer LossFunctionChange (coverage adjustment vs MODEL-03 text).

### Q2 — SHAP oracle granularity
| Option | Description | Selected |
|--------|-------------|----------|
| Per-object SHAP matrix ≤1e-5 | Full per-object × (n_features+1) matrix incl. bias; local-accuracy invariant | ✓ |
| Aggregate importance only | mean(|SHAP|) only | |
| You decide | Decide during research | |

**User's choice:** Per-object SHAP matrix ≤1e-5.

---

## Claude's Discretion

- `.cbm` blob framing around the FlatBuffers payload; exact CPU apply / tree-evaluation procedure.
- Regular SHAP algorithm internals; PredictionValuesChange / Interaction importance formulas.
- CrossEntropy & Focal gradient/hessian definitions (Focal params); prediction-type transforms.
- Where the canonical `Model` type lives post-Phase-4 (re-home into `cb-model` vs operate on `cb-train::Model`).
- `flatbuffers` crate version + `flatc` invocation strategy.

## Deferred Ideas

- LossFunctionChange feature importance (part of MODEL-03) → later phase / advanced fstr.
- SHAP interaction values, PredictionDiff, SAGE (MODEL-05) → Phase 6.
- Uncertainty prediction types (RMSEWithUncertainty/VirtEnsembles/TotalUncertainty) → Phase 6 / LOSS-08.
- Broader `.cbm` cross-version load tolerance → later hardening.
- Byte-identical `.cbm` output → explicitly rejected as a goal.
