# Feature Research

**Domain:** Gradient-boosting library — CatBoost parity surface (Rust rewrite, v1.2 "Parity Completion & Release Readiness")
**Researched:** 2026-07-05
**Confidence:** HIGH (grounded in vendored upstream source `catboost-master/` + `docs/CATBOOST_CORE_DESIGN.md`; algorithm-level citations to file/line where available)

> Scope note: this milestone *adds* surfaces to an already-mature engine (full CPU + device-resident GPU training, `.cbm`/`.json` export, SHAP + PredictionValuesChange fstr, dual sklearn/CatBoost-native Python API — see PROJECT.md "Validated"). Only the NEW v1.2 features are researched here. Each row records how upstream CatBoost exposes/implements it and what it depends on in our existing crates.

## Feature Landscape

### Table Stakes (Users Expect These)

A "drop-in CatBoost replacement" is judged against the official Python API. These are the surfaces existing CatBoost users call routinely; missing them breaks migration.

| Feature | Why Expected (upstream surface) | Complexity | Notes |
|---------|--------------------------------|------------|-------|
| **Cross-validation `cv()`** | Module-level `catboost.cv(pool, params, fold_count, ...)` returns fold-averaged learning curves. Upstream: `CrossValidate` (`train_lib/cross_validation.cpp:343`) — disables the global overfitting detector, `PrepareCvFolds` splits (stratified / time-series / inverted / custom), runs per-fold `Train(...)` with `CalcMetricsOnly`, averages per-iteration train/test into `TCVResult` (`AverageTrain/Test`, `StdDev*`). | MEDIUM | Pure orchestration over the **existing training loop** — no new kernels. Needs fold-splitting + per-iteration metric averaging. Depends on: training core (`cb-train`), metrics, eval-set plumbing (all shipped). |
| **Hyperparameter tuning — `grid_search()` / `randomized_search()`** | `model.grid_search(grid, pool)` / `randomized_search(grid, pool, n_iter)`. Upstream: `hyperparameter_tuning.cpp` `GridSearch`/`RandomizedSearch` — quantize once & reuse (`QuantizeAndSplitDataIfNeeded`), per candidate run a single split **or** full `CrossValidate`, keep best per `EMetricBestValue`; returns `TBestOptionValuesWithCvResult` (best params + CV curves). | MEDIUM | Thin loop **on top of `cv()`**; hard dependency on CV landing first. Main work: param-grid enumeration + "quantize once, retrain many" reuse for speed. |
| **Snapshot / resume** (`snapshot_file`, `snapshot_interval`) | `fit(..., save_snapshot=True, snapshot_file=..., snapshot_interval=...)`. Upstream: `ctx->SaveProgress` every `GetSnapshotSaveInterval()` sec; `TryLoadProgress` at loop start resumes from the same iteration; snapshot = serialized `TLearnProgress` (folds, approxes, tree structs, leaf values, metric history, **RNG**). Random-seed continuity is re-applied (`train_model.cpp` core `TrainModel`). | MEDIUM–HIGH | Requires serializing the **full mutable training state** (`TLearnProgress` analog), not just the model. RNG-state reproducibility is the hard part for the ≤1e-5 bar. Depends on the training-context internals + a stable snapshot format. |
| **Standalone metrics — `eval_metrics()` / calc_metrics** | `model.eval_metrics(pool, metrics, ntree_start, ntree_end, eval_period)` → per-metric, per-staged-tree-range curves. Upstream: `eval_result.h` (`TEvalResult` = `RawValues[evalIter][dim][doc]`) + `calc_metrics.h` (`ConstructMetric`, additive metrics stream block-by-block, non-additive buffer). | LOW–MEDIUM | Reuses the **existing metric objects + staged apply** (`[treeStart,treeEnd)` predict already shipped). Mostly wiring: staged predictions → metric accumulation. |
| **Feature importance — `Interaction`** | `get_feature_importance(type='Interaction')` → ranked feature **pairs**. Upstream: `CalcInternalFeatureInteraction` / `CalcInteraction` (`calc_fstr.h:109-115`) — walks tree splits, for every ordered pair of splits within a tree accumulates a score-impact contribution (co-occurring splits and their leaf-value deltas), then `CalcFeatureInteraction` maps internal `TFeature`s back to original columns. Dataset-free. | MEDIUM | No dataset needed. Reads **existing model tree structure** (splits, leaf values). New: pairwise split-cooccurrence accumulation. Emits `TFeatureInteraction{score, firstFeature, secondFeature}`. |
| **Feature importance — `LossFunctionChange`** | `get_feature_importance(type='LossFunctionChange', data=pool)` — the CatBoost-recommended importance for ranking. Upstream: `CalcFeatureEffectLossChange` (`loss_change_fstr.h`) — needs a dataset; builds `TShapPreparedTrees`, computes per-feature loss-change via SHAP-based leaf stats + `CalcFeatureEffectLossChangeMetricStats` (metric re-evaluated with each feature's contribution removed). | HIGH | **Requires a dataset** and reuses the **already-shipped SHAP machinery** (`TShapPreparedTrees` analog) + metric evaluation. Highest-value fstr but couples SHAP + metrics + loss description. |
| **Online-HNSW KNN (FEAT-07)** | Closes the **known ≤1e-5 parity gap**: CatBoost's KNN estimated-feature calcer (`embedding_features/knn.h`) builds an **approximate** HNSW index over training embeddings (`library/cpp/online_hnsw`), votes over `closeNum` neighbors. Our current impl is brute-force-exact → per-stage residual (see `.planning/notes/knn-estimated-feature-is-online-hnsw.md`). | HIGH | Must port `library/cpp/online_hnsw` (~900 LOC) bit-for-bit to match the *approximate* neighbor set (incl. incremental/online insertion order per ordered-boosting permutation). Depends on embedding calcer path (shipped). Pure parity work, no new user API. |
| **ONNX export** | `model.save_model('m.onnx', format='onnx')`. Upstream `SerializeFullModelToOnnxStream` (`model_export/onnx_helpers.cpp`) → ONNX-ML `TreeEnsembleRegressor` / `TreeEnsembleClassifier` (+ `ZipMap` for class labels). Each oblivious tree is **expanded to a full binary decision tree** of ONNX nodes (`BRANCH_GTE`/`BRANCH_GT`). | MEDIUM–HIGH | **Hard upstream limits (verified `model_exporter.cpp:91-104`):** rejects *any* categorical (`!HasCategoricalFeatures()`), text, embedding features; requires **identity scale** (`CB_ENSURE_SCALE_IDENTITY`); oblivious trees only; class labels must be int/string (**no float labels**, `onnx_helpers.cpp:137`). → ONNX path is **float-feature models only**. Post-transform (sigmoid/softmax) is *not* baked as ONNX ops — consumer applies it; cat-feature indices are still written into `metadata_props["cat_features"]` for round-trip info only. |
| **CoreML export** | `model.save_model('m.mlmodel', format='coreml')`. Upstream `OutputModelCoreML` (`coreml_helpers.cpp`) → CoreML `TreeEnsembleRegressor` (+ optional **categorical-mapping pipeline** for one-hot cats). | MEDIUM–HIGH | Supports float **and one-hot categorical** features (via `ConfigureCategoricalMappings` pipeline stage) — richer than ONNX — but **no CTR features**, no text/embedding (cbm only), requires **identity scale** (`model_exporter.cpp:179`) and **single-dimension bias** (`coreml_helpers.cpp:260`), oblivious trees only (`≤16` levels, `2^depth` leaves, `coreml_helpers.cpp:383`). Needs a protobuf CoreML `.mlmodel` writer. |
| **Benchmark vs official CatBoost** | Not an upstream API — but the project's **Core Value** ("verifiable feature parity, oracle-tested to 10⁻⁵") is unproven to adopters without an end-to-end accuracy+speed comparison on real datasets. | MEDIUM | Depends on the whole engine + Python bindings + a CatBoost install in the harness (oracle strategy already established). Extends existing `benchmark*.py`. |
| **PyPI release readiness** | Users install `pip install catboost-rs`. Per-backend wheels (`cpu`/`rocm`/`wgpu`; `cuda` untestable locally) via maturin, CI matrix, version pinning, `abi3` (already prototyped in Phase 8). | MEDIUM | Packaging/CI, not algorithm. Depends on: Python bindings (shipped), maturin build, wheel-per-backend matrix. Gate for *any* external adoption. |
| **Docs + runnable examples** | README, API docs, migration guide (sklearn ↔ CatBoost-native ↔ catboost-rs), runnable Rust + Python examples. | LOW–MEDIUM | No engine dependency; blocks adoption. Should cover the export/orchestration surfaces added this milestone. |

### Differentiators (Competitive Advantage)

Features that strengthen the "memory-efficient Rust CatBoost" positioning beyond mere parity.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **GPU inference evaluator (device predict)** | v1.1 delivered device *training*; device *predict* completes the story — batch scoring on GPU without leaving the CubeCL runtime. Upstream has it (`model/cuda/evaluator.cu`, `EnableGPUEvaluation` in `c_api.h`). | HIGH | Depends on the **shipped device-resident primitives + quantized-index residency** (v1.1) and the tree-walk. Must reuse the CubeCL `Runtime` seam + `Ok(None)`→CPU fallback pattern. Kaggle-CUDA-gated (per PROJECT.md GPU-oracle rule). Differentiator because most "from-scratch" GB rewrites never ship GPU predict. |
| **Partial-dependence (PDP)** | `partial_dependence.cpp` — shows how prediction varies as 1–2 chosen features sweep their range; popular explainability tool, pairs with the Interaction importance. | MEDIUM | Dataset + model tree-walk over a swept feature grid; reuses staged apply. Lower priority than Interaction/LossChange but rounds out the explainability suite. |
| **Memory-efficiency-forward benchmark** | The benchmark isn't just speed/accuracy — publishing **peak-RSS vs official CatBoost** turns the first-class "memory efficiency" constraint into a marketing-grade, verifiable number. | LOW (delta on the benchmark) | Adds memory instrumentation to the benchmark harness. Directly serves Core Value. |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| **ONNX export of categorical/CTR models** | Users want "just export my full model to ONNX". | **Upstream itself refuses it** (`!HasCategoricalFeatures()` hard guard) — ONNX-ML has no CTR/target-encoding op; faking it would diverge from CatBoost and break the parity promise. | Match upstream: error clearly for cat/text/embedding models on ONNX; steer users to CoreML (one-hot) or `.cbm`/`.json`. Document the limitation up front. |
| **PMML export** | "Enterprise Java scoring wants PMML." | Explicitly **out of scope for v1.2** (PROJECT.md). Another full exporter (`pmml_helpers.cpp` ~25KB) for a niche target; dilutes the release. | Defer to a later milestone; ONNX+CoreML cover the interop need now. |
| **C++/Python source-code export** | "Emit standalone applicator code." | **Out of scope** (PROJECT.md); large surface (`cpp_exporter.cpp`, `python_exporter.cpp`), low demand for a Rust-native library whose whole point is embedding. | Defer; the Rust crate *is* the embeddable applicator. |
| **Distributed / multi-node training (MPI, multi-GPU)** | "Scale to a cluster." | **Out of scope** (PROJECT.md); upstream needs Plain-boosting-only + `MaxTensorComplexity==1` constraints, master/worker over `library/cpp/par` — huge surface, single-node scope this milestone. | Single-node CPU+GPU only; revisit in a future milestone. |
| **SAGE values / Independent (background-dataset) SHAP / Carry-Uplift** | Advanced explainability parity with every upstream fstr type. | Not in the v1.2 requirement set; each is its own algorithm (`sage_values.cpp`, `independent_tree_shap.cpp`, `carry.h`). Adding them now is scope creep against the release goal. | Ship Interaction + LossFunctionChange + PDP (the requested set); leave SAGE/Independent/Carry as backlog. |
| **Baking sigmoid/softmax into ONNX as extra ops** | "Make the ONNX model output probabilities directly." | CatBoost deliberately exports **raw formula values** — inventing a post-transform graph would break bit-parity with official CatBoost's ONNX output. | Match upstream: export raw scores; document that the consumer applies sigmoid (binary) / softmax (multiclass), same as official CatBoost. |

## Feature Dependencies

```
[grid_search / randomized_search]
    └──requires──> [cv()]
                       └──requires──> [training loop + eval-set metrics]  (SHIPPED)

[snapshot / resume] ──requires──> [serializable TLearnProgress-analog training state]  (NEW state serialization)

[eval_metrics / calc_metrics] ──requires──> [staged predict + metric objects]  (SHIPPED)

[LossFunctionChange fstr]
    └──requires──> [SHAP / TShapPreparedTrees]  (SHIPPED)
    └──requires──> [metric + loss-description evaluation]  (SHIPPED)

[Interaction fstr] ──requires──> [model tree structure + leaf values]  (SHIPPED, dataset-free)

[Partial dependence] ──requires──> [staged predict over swept feature grid]  (SHIPPED)

[GPU inference evaluator]
    └──requires──> [device-resident quantized index + primitives]  (SHIPPED v1.1)
    └──requires──> [CubeCL Runtime seam + Ok(None)->CPU fallback]  (SHIPPED pattern)

[ONNX export] ──requires──> [oblivious-tree walk + identity-scale + float-only guard]
[CoreML export] ──requires──> [oblivious-tree walk + one-hot cat mapping + protobuf writer]

[Online-HNSW KNN] ──requires──> [embedding estimator path + ordered permutation plumbing]  (SHIPPED)

[Benchmark vs official CatBoost] ──requires──> [full engine + Python bindings + CatBoost oracle install]  (SHIPPED)
[PyPI release] ──requires──> [Python bindings + maturin per-backend wheels]  (SHIPPED prototype)
[Docs/examples] ──enhances──> [PyPI release, export, orchestration]
```

### Dependency Notes

- **grid/random search requires cv():** upstream `RandomizedSearch` calls `CrossValidate` per candidate; ordering cv → tuning is mandatory in the roadmap.
- **LossFunctionChange requires SHAP:** it is computed from SHAP-based per-leaf stats, so it reuses the already-shipped SHAP `TShapPreparedTrees` analog rather than a new tree walk. Interaction, by contrast, is dataset-free and only needs the tree structure.
- **snapshot/resume is the odd one out:** it depends on serializing *mutable training state* (not the final model), including RNG — the only new *format* work in the orchestration group, and the one most exposed to the ≤1e-5 reproducibility bar.
- **GPU predict rides on v1.1:** no new device primitives needed; it reuses the resident quantized index and the `Runtime` generic seam. Correctness must be Kaggle-CUDA-signed-off (ROCm in-env is smoke only).
- **Export guards are hard parity constraints, not choices:** replicate upstream `CB_ENSURE` guards exactly (ONNX float-only + identity scale; CoreML one-hot + identity scale + single-dim bias; both oblivious-only) so exported artifacts match official CatBoost where the format is deterministic.

## MVP Definition

### Launch With (v1.2 core)

Minimum to call the milestone "parity complete + release ready".

- [ ] **Discharge v1.1 debt** — GPUT-14 aggregate ε=1e-4 Kaggle CUDA sign-off, Phase-10/11 BENCH-02 rows, RV-13-01..04 latent hazards — because the milestone explicitly carries them and they underwrite the correctness claim.
- [ ] **Online-HNSW KNN (FEAT-07)** — the one *open* ≤1e-5 parity gap; parity is the Core Value.
- [ ] **cv() + grid_search/randomized_search + snapshot/resume + eval_metrics** — the orchestration surface every migrating CatBoost user calls.
- [ ] **Interaction + LossFunctionChange feature importance** — the two explicitly-requested fstr types; LossFunctionChange is CatBoost's recommended importance.
- [ ] **ONNX + CoreML export** — the interop requirement (with upstream limitations replicated exactly).
- [ ] **Benchmark vs official CatBoost (accuracy + speed + memory) + PyPI wheels + docs/examples** — without these the parity work is invisible to adopters.

### Add After Validation (v1.2 later phases)

- [ ] **GPU inference evaluator** — high value but rides on v1.1 residency; sequence after the CPU-side orchestration/export lands so the CPU path stays the oracle.
- [ ] **Partial dependence** — completes the explainability trio once Interaction/LossChange are in.

### Future Consideration (post-v1.2)

- [ ] **SAGE / Independent SHAP / Carry-Uplift fstr** — parity completeness, low current demand.
- [ ] **PMML + C++/Python source export** — deferred interop formats.
- [ ] **Distributed / multi-GPU training** — separate milestone; needs master/worker infra.

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| v1.1 debt discharge (GPUT-14, BENCH-02, RV-13) | HIGH | MEDIUM | P1 |
| Online-HNSW KNN (FEAT-07) | HIGH | HIGH | P1 |
| cv() | HIGH | MEDIUM | P1 |
| grid/randomized search | HIGH | MEDIUM | P1 |
| snapshot / resume | HIGH | MEDIUM-HIGH | P1 |
| eval_metrics / calc_metrics | MEDIUM | LOW-MEDIUM | P1 |
| Interaction fstr | MEDIUM | MEDIUM | P1 |
| LossFunctionChange fstr | HIGH | HIGH | P1 |
| ONNX export | HIGH | MEDIUM-HIGH | P1 |
| CoreML export | MEDIUM | MEDIUM-HIGH | P1 |
| Benchmark vs official CatBoost | HIGH | MEDIUM | P1 |
| PyPI release readiness | HIGH | MEDIUM | P1 |
| Docs + examples | HIGH | LOW-MEDIUM | P1 |
| GPU inference evaluator | MEDIUM-HIGH | HIGH | P2 |
| Partial dependence | MEDIUM | MEDIUM | P2 |
| SAGE / Independent SHAP / Carry | LOW | HIGH | P3 |
| PMML / source-code export | LOW | HIGH | P3 |

**Priority key:** P1 = must-have for the v1.2 release; P2 = should-have, sequence after P1; P3 = deferred backlog.

## Competitor Feature Analysis

The relevant "competitor" is **official CatBoost itself** (the parity oracle); XGBoost/LightGBM are secondary reference points for the interop/DX surfaces.

| Feature | Official CatBoost | XGBoost / LightGBM | Our Approach (catboost-rs v1.2) |
|---------|-------------------|--------------------|---------------------------------|
| ONNX export | float-only, identity-scale, oblivious, int labels (`onnx_helpers.cpp`) | XGBoost via `onnxmltools`/converters; broad | **Match CatBoost exactly**, incl. the guards — parity over coverage |
| CoreML export | one-hot cats + pipeline, identity-scale, single-dim bias | limited/none | Match CatBoost's one-hot pipeline; no CTR |
| cv / grid / random search | `cv`, `grid_search`, `randomized_search` | sklearn `GridSearchCV` wraps them | Native `cv()`/`grid_search()`/`randomized_search()` mirroring CatBoost signatures |
| snapshot/resume | `snapshot_file`/`snapshot_interval` on `fit` | LightGBM continued training; different | Serialize training-state analog; RNG-continuity to hold ≤1e-5 |
| Interaction / LossFunctionChange fstr | both, plus SHAP/SAGE/Independent | XGBoost gain/weight/cover; SHAP via `shap` | Ship Interaction + LossFunctionChange + PDP; defer SAGE/Independent |
| GPU inference | `EnableGPUEvaluation` CUDA path | XGBoost GPU predictor | CubeCL device predict over v1.1 residency, `Ok(None)`→CPU fallback |
| KNN estimated feature | **online HNSW** (approximate) | n/a | Port `online_hnsw` for bit-exact approximate parity |
| Packaging | PyPI wheels (CUDA/CPU) | PyPI/conda | maturin per-backend wheels (`cpu`/`rocm`/`wgpu`; `cuda` build-only) |

## Sources

- `docs/CATBOOST_CORE_DESIGN.md` — §"Trained Model Representation, Serialization, and Export" (Export Formats table, CTR-in-model), §"Training Orchestration & Driver Layer" (CV / hyperparameter tuning / snapshots), §"Inference API, Feature Importance (fstr), and Language Bindings" (fstr dispatcher table, SHAP types, eval_result/calc_metrics), §"Text and Embedding Features" (KNN = HNSW). [HIGH — repo design doc]
- `catboost-master/catboost/libs/model/model_export/model_exporter.cpp:91-104,153-195` — ONNX guards (`!HasCategoricalFeatures/Text/Embedding`, `CB_ENSURE_SCALE_IDENTITY`), CoreML oblivious/identity-scale guards. [HIGH — upstream source]
- `catboost-master/catboost/libs/model/model_export/onnx_helpers.cpp` — TreeEnsembleRegressor/Classifier + ZipMap, int/string-labels-only (`:137`), cat_features metadata_props. [HIGH]
- `catboost-master/catboost/libs/model/model_export/coreml_helpers.cpp` — one-hot categorical mapping pipeline, single-dim bias (`:260`), `≤16` levels / `2^depth` leaves (`:383`). [HIGH]
- `catboost-master/catboost/libs/fstr/calc_fstr.h`, `loss_change_fstr.h` — `CalcInteraction`/`CalcInternalFeatureInteraction`, `CalcFeatureEffectLossChange`, `GetFeatureImportances` dispatcher. [HIGH]
- `catboost-master/catboost/libs/fstr/partial_dependence.h`, `library/cpp/online_hnsw` (upstream HNSW impl), `catboost/libs/model/cuda/evaluator.cu` (GPU evaluator). [HIGH — file existence + role]
- `.planning/PROJECT.md` (v1.2 Active requirements + Out of Scope), `.planning/notes/knn-estimated-feature-is-online-hnsw.md` (KNN parity root cause). [HIGH — project state]

---
*Feature research for: CatBoost parity surface (v1.2 export / orchestration / extended fstr / GPU predict / online-HNSW / adoption)*
*Researched: 2026-07-05*
