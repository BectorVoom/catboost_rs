# Feature Research

**Domain:** Gradient boosting ML library (full Rust rewrite of CatBoost, parity target)
**Researched:** 2026-06-13
**Confidence:** HIGH (grounded in vendored CatBoost C++ source at `catboost-master/`)

## Scope Note

v1 target is **full feature parity** with CatBoost. "Table stakes" here means *parity-critical core* — the things a CatBoost user assumes work identically. "Differentiators" are CatBoost's *signature* algorithmic capabilities that distinguish it from XGBoost/LightGBM and are the reason someone picks CatBoost at all (so they are also parity-critical, but called out separately because they drive the architecture). "Anti-features" are surfaces the project has *explicitly excluded* (per `.planning/PROJECT.md` Out of Scope).

All feature names and enum values below are verified against the vendored source:
- Loss/metric surface: `catboost/private/libs/options/enums.h` → `enum class ELossFunction` (90+ values)
- Categorical CTR: `catboost/private/libs/ctr_description/ctr_type.h`, `options/cat_feature_options.h`
- Boosting mode: `options/enums.h` → `enum EBoostingType { Ordered, Plain }`
- Text/embedding calcers: `enum class EFeatureCalcerType { BoW, NaiveBayes, BM25, LDA, KNN }`
- Model export: `libs/model/enums.h` → `enum class EModelType { CatboostBinary, AppleCoreML, Cpp, Python, Json, Onnx, Pmml, CPUSnapshot }`
- Fstr: `enum class EFstrType { PredictionValuesChange, LossFunctionChange, ShapValues, ShapInteractionValues, Interaction, PredictionDiff, SageValues, ... }`

## Feature Landscape

### Table Stakes (Parity-Critical Core)

Features a CatBoost user assumes work identically. Missing/divergent = not a drop-in replacement.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Symmetric (oblivious) decision trees | The core CatBoost tree structure; same split + threshold across a whole tree level | HIGH | `EGrowPolicy::SymmetricTree` (default). Underpins **everything** — model format, fast SIMD inference, leaf indexing are all built around it. Must be the first tree primitive built. |
| Gradient boosting train loop (CPU) | Core of the library | HIGH | Iterative additive trees; `learning_rate`, `iterations`, `depth`. Leaf estimation via `ELeavesEstimation { Gradient, Newton, Exact, Simple }`. |
| Binary classification (Logloss, CrossEntropy) | Most common use case | MEDIUM | `Logloss`, `CrossEntropy`, `Focal`, `CtrFactor`. Sigmoid + probability output. |
| Multiclass classification | Standard | MEDIUM-HIGH | `MultiClass` (softmax), `MultiClassOneVsAll`. Multi-dim leaf values; `MultiLogloss`/`MultiCrossEntropy` for multilabel. |
| Regression | Standard | MEDIUM | `RMSE`, `MAE`, `Quantile`, `MultiQuantile`, `LogCosh`, `Huber`, `Poisson`, `Tweedie`, `MAPE`, `MSLE`, `Lq`, `Expectile`, `LogLinQuantile`, `MedianAbsoluteError`, `SMAPE`, `RMSPE`, `Cox`, `MultiRMSE`. |
| Pool abstraction (data input) | Native CatBoost API; required for drop-in | HIGH | Holds features (float/cat/text/embedding), label, weights, group_id, subgroup_id, pairs, baseline. Quantized + raw variants (`libs/data/`, `private/libs/quantization/`). Memory-efficiency mandate makes the quantized representation central. |
| Feature quantization (binarization) | Speed + the way splits are chosen | HIGH | `TBinarizationOptions`, border selection methods, `border_count`. Pre-binning floats into bucket indices. Inputs feed every tree split. |
| Missing value handling | Real datasets have NaNs | LOW-MEDIUM | `ENanMode { Min, Max, Forbidden }` — NaN routed to min/max bucket or rejected. |
| Feature/object weights, class weights | Standard training control | LOW | Per-object `weight`, per-class weights, `EAutoClassWeightsType { Balanced, SqrtBalanced, None }`. |
| Overfitting detection / early stopping | Expected default behavior | MEDIUM | `EOverfittingDetectorType { None, Wilcoxon, IncToDec, Iter }`, `od_pval`, `od_wait`, `use_best_model`. (`libs/overfitting_detector/`) |
| Eval set / validation metrics during training | Standard monitoring | MEDIUM | Multiple eval sets, `eval_metric`, custom metric list, per-iteration metric logging. (`libs/metrics/`) |
| Prediction types | Drop-in API parity | LOW-MEDIUM | `EPredictionType { Probability, LogProbability, Class, RawFormulaVal, Exponent, RMSEWithUncertainty, VirtEnsembles, TotalUncertainty }`. |
| Feature importance | Users expect `get_feature_importance()` | MEDIUM | `EFstrType::PredictionValuesChange` (default), `LossFunctionChange`, `Interaction`. (`libs/fstr/`) |
| SHAP values | Now table stakes for any GBM | HIGH | `EFstrType::ShapValues`; `EShapCalcType { Regular, Approximate, Exact }`. CatBoost has an exact poly-time SHAP for trees. |
| Model serialization (native .cbm) | Save/load a trained model | HIGH | FlatBuffers-based binary format (`libs/model/flatbuffers/`). Cross-version compatibility is a PROJECT requirement. |
| Model load/predict (inference) | Already partially present in vendored Rust crate | MEDIUM | The existing `catboost`/`catboost-sys` crates do inference via FFI; the rewrite reimplements this natively in Rust. |
| Bootstrap / sampling | Standard regularization | MEDIUM | `EBootstrapType { Poisson, Bayesian, Bernoulli, MVS, No }`, `subsample`, `ESamplingUnit { Object, Group }`. |
| L2 regularization, random strength | Standard regularization | LOW | `l2_leaf_reg`, `random_strength`, `bagging_temperature`. |
| Learning-rate schedule / auto LR | Expected ergonomics | LOW | Auto learning-rate selection from dataset size; constant + decay support. |
| sklearn-compatible Python API | PROJECT requirement | MEDIUM | `fit`/`predict`/`predict_proba`/`score`; `CatBoostClassifier`, `CatBoostRegressor`, `CatBoostRanker`, `CatBoost`. |
| CatBoost-native Python API | PROJECT requirement | MEDIUM-HIGH | `Pool`, full CatBoost parameter-name parity. Verified classes in `python-package/catboost/core.py`. |
| Python input: NumPy / Pandas / Arrow / Polars | PROJECT requirement | MEDIUM | Zero-copy ingestion where possible; Arrow path exists in upstream (`python-package/catboost/arrow.cpp`). |

### Differentiators (CatBoost Signature Capabilities)

These are *why CatBoost exists* and are still parity-critical, but they drive the architecture and are the hardest/most novel pieces to replicate exactly.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Ordered boosting** | The flagship anti-leakage algorithm; reduces prediction shift / target leakage | VERY HIGH | `EBoostingType::Ordered` vs `Plain`. Maintains multiple models over random permutations so each example's gradient uses only "past" examples. Hardest correctness target for oracle parity; deeply coupled to permutation machinery and CTR computation. |
| **Ordered target statistics (ordered CTR)** | Leakage-free categorical encoding — CatBoost's defining feature | VERY HIGH | `ECtrType { Borders, Buckets, BinarizedTargetMeanValue, FloatTargetMeanValue, Counter, FeatureFreq }` with priors. Uses the same permutation as ordered boosting. (`private/libs/ctr_description/`, `options/cat_feature_options.h`) |
| **Feature combinations (tensor CTRs)** | Automatic categorical crosses encoded as CTRs | VERY HIGH | `SimpleCtrs`, `CombinationCtrs`, `PerFeatureCtrs`, `MaxTensorComplexity`. Combinatorial growth controlled by `max_ctr_complexity`. Inflates model size + inference complexity. |
| **One-hot for low-cardinality categoricals** | Avoids CTR overhead when categories are few | LOW-MEDIUM | `OneHotMaxSize` / `one_hot_max_size` threshold — categoricals with ≤ N values use one-hot splits instead of CTRs. |
| **Native categorical handling end-to-end** | No manual encoding needed by the user | HIGH | String categoricals hashed (`libs/cat_feature/`); combined CTR + one-hot pipeline. A major DX advantage over XGBoost/LightGBM. |
| **Text features** | Built-in NLP without external preprocessing | HIGH | Tokenization → `EFeatureCalcerType { BoW, NaiveBayes, BM25 }`; `text_processing` / dictionary config (`private/libs/text_features/`, `text_processing/`, `options/text_processing_options.h`). |
| **Embedding features** | Use precomputed vectors directly | HIGH | `EFeatureCalcerType { LDA, KNN }` over embedding columns (`private/libs/embedding_features/` — `lda.h`, `knn.h`). |
| **Ranking (pairwise + listwise)** | First-class learning-to-rank | HIGH | `YetiRank`, `YetiRankPairwise`, `PairLogit`, `PairLogitPairwise`, `QueryRMSE`, `QuerySoftMax`, `QueryCrossEntropy`, `LambdaMart`, `StochasticRank`, `StochasticFilter`, `GroupQuantile`. Needs group_id/subgroup_id + pairs in Pool. Ranking metrics: `NDCG`, `DCG`, `MAP`, `MRR`, `ERR`, `PFound`, `PrecisionAt`, `RecallAt`, `QueryAUC`. |
| **Uncertainty estimation** | Probabilistic predictions, rare in GBMs | MEDIUM-HIGH | `RMSEWithUncertainty`, virtual ensembles (`VirtEnsembles`, `TotalUncertainty` prediction types). |
| **Multi-permutation machinery** | Underlies ordered boosting + ordered CTR | VERY HIGH | Shared infrastructure (`fold_count` permutations). Build once; both signature features depend on it. |
| **Grow policies beyond symmetric** | Flexibility for non-oblivious trees | HIGH | `EGrowPolicy { SymmetricTree, Lossguide, Depthwise, Region }`. Lossguide/Depthwise diverge from the oblivious structure and complicate the model format. |
| **Score functions** | Split-scoring variants | MEDIUM | `EScoreFunction { SolarL2, Cosine, NewtonL2, NewtonCosine, LOOL2, SatL2, L2 }`. |
| **Custom objectives / metrics** | Extensibility | HIGH | `PythonUserDefinedPerObject`, `PythonUserDefinedMultiTarget`, `UserPerObjMetric`, `UserQuerywiseMetric`. In Rust: a trait + Python callback bridge via PyO3. |
| **Monotone constraints** | Enforce monotonic feature→prediction | MEDIUM | `options/monotone_constraints.h`; per-feature +1/-1/0. |
| **Feature penalties / per-object penalties** | Cost-aware feature use | MEDIUM | `options/feature_penalties_options.h`. |
| **GPU training via CubeCL** | PROJECT differentiator vs upstream (upstream is CUDA-only) | VERY HIGH | Multi-backend (`cuda`/`rocm`/`wgpu`/`cpu`) at compile time via Cargo features + generic CubeCL runtime. Upstream `catboost/cuda/` is the algorithmic reference. See dependency notes — feature set differs from CPU. |
| **SHAP interaction values + advanced fstr** | Deep explainability | HIGH | `ShapInteractionValues`, `PredictionDiff`, `SageValues`, `InternalInteraction`, `Interaction`. |
| **Feature selection** | Automated feature pruning | MEDIUM | `EFeaturesSelectionAlgorithm { RecursiveByPredictionValuesChange, RecursiveByLossFunctionChange, RecursiveByShapValues }`, grouping `Individual` / `ByTags` (`libs/features_selection/`). |

### Anti-Features (Explicitly Excluded — per PROJECT.md Out of Scope)

These exist in upstream CatBoost but are deliberately NOT being built. Documented to prevent scope re-creep.

| Feature | Why Requested | Why Problematic (for this project) | Alternative |
|---------|---------------|-------------------------------------|-------------|
| C API / C FFI layer (`libs/model_interface/c_api.h`) | Upstream's universal binding surface | PROJECT mandates PyO3 direct bindings only; an extra unsafe C ABI is redundant and a maintenance burden | PyO3 bindings call native Rust directly |
| R bindings (`R-package/`) | Upstream supports R | Out of scope: "Rust and Python only for this milestone" | None this milestone |
| CLI application (`catboost/app/`) | Upstream ships a `catboost` CLI | Out of scope: Rust + Python only | Use the Rust/Python APIs |
| JVM / Scala, .NET, Node.js bindings | Upstream ships them | Out of scope: Rust + Python only | None this milestone |
| Model export to CoreML / ONNX / PMML / C++ / Python source | Upstream `EModelType` supports them | Not needed for a drop-in CatBoost replacement; large surface, low value here | Native `.cbm` (CatboostBinary) + JSON for interop; defer others |
| Mobile / embedded targets (`CMakeLists.android-*`) | Upstream cross-compiles to Android/ARM mobile | Out of scope: desktop + server only | x86_64 / aarch64 server-class targets |
| Real-time / online / streaming training | Sometimes requested | Out of scope: batch training only | Batch retrain |
| Distributed multi-node training (`private/libs/distributed/`) | Upstream supports it | Not in active scope; very high complexity, low priority for v1 | Single-node CPU/GPU |
| CUDA-direct GPU inference (`EnableGPUEvaluation`) | Upstream C API path | Replaced by CubeCL strategy; the upstream CUDA inference path is not the chosen abstraction | CubeCL multi-backend kernels |
| MonoForest / model analysis tooling, dataset statistics CLI | Upstream extras | Peripheral to a drop-in library; defer | Defer to v2+ if demanded |

## Feature Dependencies

```
Symmetric (oblivious) trees
    └──underpins──> Native model format (.cbm / FlatBuffers)
    └──underpins──> Fast inference (load/predict)
    └──underpins──> Feature importance / SHAP (tree-structure traversal)

Feature quantization (binarization)
    └──required-by──> Tree split search
    └──required-by──> GPU training (quantized buckets are the GPU data layout)

Pool abstraction
    └──required-by──> ALL training & inference (data carrier)
    └──carries──> group_id/subgroup_id/pairs ──required-by──> Ranking losses
    └──carries──> cat/text/embedding columns ──required-by──> CTR / text / embedding calcers

Multi-permutation machinery (fold permutations)
    └──required-by──> Ordered boosting (EBoostingType::Ordered)
    └──required-by──> Ordered target statistics (ordered CTR)

Ordered target statistics (CTR)
    └──requires──> Categorical hashing (libs/cat_feature)
    └──requires──> Target/label data (CtrsNeedTargetData)
    └──enhanced-by──> Feature combinations (tensor CTRs)
    └──alternative──> One-hot (one_hot_max_size threshold) for low cardinality

Gradient boosting train loop
    └──requires──> Quantization + Trees + Loss/gradient functions + Leaf estimation
    └──enhanced-by──> Bootstrap/sampling, L2 reg, overfitting detector

Loss functions (ELossFunction)
    └──determine──> gradient/hessian + leaf estimation + default eval metric

SHAP / fstr
    └──requires──> trained model + tree structure
    └──ShapValues required-by──> RecursiveByShapValues feature selection

GPU training (CubeCL)
    └──requires──> quantized Pool data layout + generic-float kernels
    └──reuses──> same loss/tree algorithms as CPU (parity must hold across backends)

Python API (PyO3)
    └──requires──> stable native Rust API (Builder pattern) first
    └──sklearn + CatBoost-native──> both wrap the same core
```

### Dependency Notes

- **Symmetric trees underpin everything.** The oblivious-tree structure dictates the model serialization format, the SIMD-friendly inference path, and how SHAP/importance traverse the model. Build this primitive first; it has no upstream dependency but everything depends on it.
- **Quantization gates both CPU split search and GPU data layout.** GPU kernels consume the quantized bucket representation, so quantization must be stable before serious GPU work.
- **Multi-permutation machinery is the shared root of both signature features.** Ordered boosting and ordered CTR both consume the same fold-permutation infrastructure. Build it once, validate it, then layer both features on top.
- **Ordered CTR requires target data and categorical hashing.** `CtrsNeedTargetData` gates CTR computation; string categoricals must be hashed first (`libs/cat_feature/`).
- **Ranking requires Pool group/pair metadata.** YetiRank/PairLogit/QueryRMSE need group_id, subgroup_id, and pairs carried by the Pool — Pool must support these columns before ranking losses land.
- **Python API depends on a stable Rust core.** Both the sklearn-compatible and CatBoost-native surfaces wrap the same Rust engine, so the Builder-pattern Rust API should stabilize first.
- **GPU vs CPU feature parity is a constraint, not just a feature.** Upstream historically has feature gaps between CPU and GPU (e.g., some CTR types like `FeatureFreq` are GPU-only per source comments; certain losses/options are CPU-only — see `TCpuOnlyOption`/`TGpuOnlyOption` in `cat_feature_options.h`). The rewrite must decide per-feature which backends support it; oracle parity (≤1e-5) must hold on whatever backends claim support.

## MVP Definition

Given the full-parity v1 target, "MVP" here = the minimal vertical slice that proves the architecture before fanning out across the full loss/feature matrix.

### Launch With (v1 core slice)

- [ ] Symmetric oblivious trees + native `.cbm` (FlatBuffers) load/predict — the structural foundation
- [ ] Feature quantization (float binarization, border selection)
- [ ] Pool abstraction (float + cat + label + weights + group_id/pairs scaffolding)
- [ ] Plain boosting train loop with Logloss + RMSE (binary clf + regression)
- [ ] **Ordered boosting** + **ordered CTR** + one-hot threshold + feature combinations — the signature slice; without these it is not CatBoost
- [ ] Overfitting detector + early stopping + eval-set metrics
- [ ] Feature importance + SHAP values (Regular)
- [ ] Rust Builder API + PyO3 bindings (sklearn + CatBoost-native) for the above
- [ ] Oracle test harness wiring (random inputs vs upstream, ≤1e-5)

### Add After Validation (v1.x — completes parity)

- [ ] Full loss/metric matrix: multiclass, multilabel, all regression variants, quantile/Tweedie/Poisson/Huber/etc.
- [ ] Ranking suite (YetiRank, PairLogit, QueryRMSE, LambdaMart) + ranking metrics
- [ ] Text features (tokenization, BoW/NaiveBayes/BM25)
- [ ] Embedding features (LDA, KNN calcers)
- [ ] Alternative grow policies (Lossguide, Depthwise, Region)
- [ ] GPU training via CubeCL (`rocm` first per test mandate, then `wgpu`/`cuda`)
- [ ] Uncertainty estimation / virtual ensembles
- [ ] Custom objectives/metrics (Rust trait + Python callback)
- [ ] Monotone constraints, feature penalties, feature selection
- [ ] SHAP interaction values + advanced fstr (LossFunctionChange, PredictionDiff, SAGE)
- [ ] Arrow/Polars input paths, JSON model export

### Future Consideration (v2+ / likely out of scope)

- [ ] Distributed multi-node training — explicitly not in active scope
- [ ] Additional model export formats (ONNX/CoreML/PMML/C++/Python source) — anti-features for now
- [ ] MonoForest / model analysis tooling

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Symmetric trees + `.cbm` load/predict | HIGH | HIGH | P1 |
| Quantization | HIGH | HIGH | P1 |
| Pool abstraction | HIGH | HIGH | P1 |
| Plain boosting (Logloss, RMSE) | HIGH | HIGH | P1 |
| Ordered boosting + ordered CTR + combinations | HIGH | VERY HIGH | P1 |
| Overfitting detector / early stopping | HIGH | MEDIUM | P1 |
| Feature importance + SHAP (Regular) | HIGH | HIGH | P1 |
| Rust Builder API + PyO3 (sklearn + native) | HIGH | HIGH | P1 |
| Oracle test harness | HIGH | MEDIUM | P1 |
| Full loss/metric matrix (multiclass, regression variants) | HIGH | HIGH | P2 |
| Ranking suite + ranking metrics | MEDIUM-HIGH | HIGH | P2 |
| Text features | MEDIUM | HIGH | P2 |
| Embedding features | MEDIUM | HIGH | P2 |
| GPU training via CubeCL | HIGH | VERY HIGH | P2 |
| Alt grow policies (Lossguide/Depthwise/Region) | MEDIUM | HIGH | P2 |
| Uncertainty / virtual ensembles | MEDIUM | MEDIUM-HIGH | P2 |
| Custom objectives/metrics | MEDIUM | HIGH | P2 |
| Monotone constraints / penalties / feature selection | MEDIUM | MEDIUM | P3 |
| SHAP interaction + advanced fstr | MEDIUM | HIGH | P3 |
| Arrow/Polars input, JSON export | MEDIUM | MEDIUM | P3 |
| Distributed training, extra export formats | LOW | VERY HIGH | P3 (likely excluded) |

**Priority key:**
- P1: Must have for the v1 core slice (proves architecture + signature features)
- P2: Required to reach full parity; add after the core slice is oracle-validated
- P3: Parity tail / nice-to-have; some overlap with anti-features

## Competitor Feature Analysis

| Feature | CatBoost (upstream, the oracle) | XGBoost / LightGBM | Our Approach |
|---------|----------------------------------|--------------------|--------------|
| Categorical handling | Native ordered CTR + one-hot + combinations | Manual encoding or basic native (LightGBM) | Replicate ordered CTR + combinations exactly (signature parity) |
| Tree structure | Symmetric/oblivious by default | Depthwise/leafwise | Symmetric-first, then add Lossguide/Depthwise/Region for parity |
| Anti-leakage | Ordered boosting (unique) | None | Replicate ordered boosting — the hardest oracle target |
| Text/embedding | Built-in calcers | None / external | Replicate BoW/NaiveBayes/BM25 + LDA/KNN |
| GPU | CUDA only | CUDA / OpenCL (LightGBM) | CubeCL multi-backend (cuda/rocm/wgpu) — a project differentiator over upstream |
| Bindings | C API + Py/R/JVM/.NET/Node | Many | PyO3-only (no C API); Rust-native + Python (sklearn + CatBoost-native) |
| Explainability | SHAP + interaction + SAGE | SHAP | SHAP (Regular) in v1; interaction/SAGE in tail |

## Sources

- Vendored CatBoost C++ source (oracle + algorithmic reference): `catboost-master/`
  - `catboost/private/libs/options/enums.h` — `ELossFunction`, `EBoostingType`, `EGrowPolicy`, `EBootstrapType`, `EScoreFunction`, `ELeavesEstimation`, `EOverfittingDetectorType`, `ENanMode`, `EPredictionType`, `EFeatureCalcerType`, `EFeaturesSelectionAlgorithm`, `EAutoClassWeightsType`, `ESamplingUnit` (HIGH confidence — read directly)
  - `catboost/private/libs/ctr_description/ctr_type.h` — `ECtrType`
  - `catboost/private/libs/options/cat_feature_options.h` — CTR descriptions, `OneHotMaxSize`, `MaxTensorComplexity`, CPU/GPU-only options
  - `catboost/private/libs/embedding_features/{lda.h,knn.h}` — embedding calcers
  - `catboost/libs/model/enums.h` — `EModelType` export formats
  - `catboost/libs/fstr/`, `EFstrType` — feature importance / SHAP types
  - `catboost/python-package/catboost/core.py` — `Pool`, `CatBoost`, `CatBoostClassifier/Regressor/Ranker`, `EShapCalcType`
  - Lib layout: `catboost/libs/`, `catboost/private/libs/`, `catboost/cuda/`
- Project scope/boundaries: `.planning/PROJECT.md` (Active + Out of Scope)
- Component map: `.planning/codebase/ARCHITECTURE.md`, `.planning/codebase/STRUCTURE.md`

---
*Feature research for: gradient boosting ML library (CatBoost parity rewrite in Rust)*
*Researched: 2026-06-13*
