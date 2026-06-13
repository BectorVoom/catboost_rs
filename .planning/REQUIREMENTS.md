# Requirements: catboost-rs

**Defined:** 2026-06-13
**Core Value:** A memory-efficient, Rust-native CatBoost implementation with verifiable feature parity (oracle-tested ≤10⁻⁵), embeddable in Rust and droppable into both scikit-learn and existing CatBoost Python pipelines.

> **Scope note:** v1 target is *full feature parity* with upstream CatBoost. "v1" therefore covers the complete parity surface, sequenced (per research) as a narrow oracle-passing core slice first, then additive widening. Each requirement is verified against the vendored `catboost-master/` source as oracle (absolute error ≤ 10⁻⁵ on the CPU path; a separately-stated looser tolerance on GPU).

## v1 Requirements

### Infrastructure & Oracle Harness

- [x] **INFRA-01**: Modular Cargo workspace with feature-gated backend crates (`cpu`/`wgpu`/`cuda`/`rocm`) and clear separation of responsibilities
- [x] **INFRA-02**: Lint discipline enforced in library crates — deny `unwrap`/`expect`/`panic`/`indexing_slicing`; `thiserror` in libraries, `anyhow` only at binding/app/test edges; CI check that `anyhow` is absent from core library code
- [x] **INFRA-03**: Oracle test harness — randomly generated inputs validated against upstream CatBoost outputs to ≤10⁻⁵, with frozen committed fixtures, pinned seed/version, and single-thread determinism
- [x] **INFRA-04**: Intermediate per-stage oracle tooling — compare quantization borders, per-tree splits, leaf values, and per-iteration approximants (not just final predictions)
- [x] **INFRA-05**: Exact port of CatBoost's `TFastRng64` PRNG, bitstream-oracle-validated against the C++ generator for a fixed seed
- [x] **INFRA-06**: Source and test code strictly separated (no inline `#[cfg(test)]` mixed with production logic)

### Data Layer (Pool & Quantization)

- [x] **DATA-01**: `Pool` abstraction — float/categorical/text/embedding columns, label, weights, group_id, subgroup_id, pairs, baseline
- [x] **DATA-02**: `QuantizedPool` — columnar SoA `u8`/`u16` bin storage with pre-allocated buffers reused across rounds (memory efficiency)
- [x] **DATA-03**: `GreedyLogSum` border selection, per-feature border set oracle-validated (including NaN/duplicate columns, `<`/`<=` semantics)
- [x] **DATA-04**: Missing-value handling — `NanMode` (Min/Max/Forbidden)
- [ ] **DATA-05**: Categorical feature hashing
- [ ] **DATA-06**: Zero-copy NumPy ingestion and Arrow/Polars ingestion with dtype/contiguity validation; copy-in path for training
- [x] **DATA-07**: Single audited deterministic reduction utility matching the C++ `double` accumulator type and summation order
- [ ] **DATA-08**: Per-object / per-class weights and auto class weights (Balanced/SqrtBalanced)

### CPU Training Core

- [ ] **TRAIN-01**: Plain gradient boosting train loop (`iterations`, `learning_rate`, `depth`)
- [ ] **TRAIN-02**: Symmetric (oblivious) decision trees — the core CatBoost tree structure
- [ ] **TRAIN-03**: Leaf value estimation — Gradient, Newton, Exact, Simple
- [ ] **TRAIN-04**: Bootstrap / sampling — Poisson, Bayesian, Bernoulli, MVS, No; `subsample`; object/group sampling units
- [ ] **TRAIN-05**: Regularization — `l2_leaf_reg`, `random_strength`, `bagging_temperature`
- [ ] **TRAIN-06**: Overfitting detection and early stopping — Wilcoxon/IncToDec/Iter, `od_pval`/`od_wait`, `use_best_model`
- [ ] **TRAIN-07**: Eval-set validation metrics logged per iteration (multiple eval sets, `eval_metric`)
- [ ] **TRAIN-08**: Automatic learning-rate selection from dataset size

### Ordered Algorithms (Signature Features)

- [ ] **ORD-01**: Multi-permutation fold machinery (`fold_count` permutations, `TFold`-equivalent bookkeeping)
- [ ] **ORD-02**: Ordered boosting (`EBoostingType::Ordered`) with exact prefix boundaries, per-object intermediate oracle
- [ ] **ORD-03**: Ordered target statistics / CTR — `Borders`, `Buckets`, `BinarizedTargetMeanValue`, `FloatTargetMeanValue`, `Counter`, `FeatureFreq` with priors
- [ ] **ORD-04**: One-hot encoding for low-cardinality categoricals (`one_hot_max_size` threshold)
- [ ] **ORD-05**: Feature combinations (tensor CTRs) — `SimpleCtrs`/`CombinationCtrs`, `max_ctr_complexity` control

### Losses, Metrics & Prediction

- [ ] **LOSS-01**: Binary classification — Logloss, CrossEntropy, Focal
- [ ] **LOSS-02**: Multiclass (MultiClass softmax, MultiClassOneVsAll) and multilabel (MultiLogloss, MultiCrossEntropy)
- [ ] **LOSS-03**: Regression matrix — RMSE, MAE, Quantile, MultiQuantile, LogCosh, Huber, Poisson, Tweedie, MAPE, MSLE, Lq, Expectile, etc.
- [ ] **LOSS-04**: Ranking losses — YetiRank(/Pairwise), PairLogit(/Pairwise), QueryRMSE, QuerySoftMax, LambdaMart, StochasticRank
- [ ] **LOSS-05**: Ranking metrics — NDCG, DCG, MAP, MRR, ERR, PFound, PrecisionAt, RecallAt, QueryAUC
- [ ] **LOSS-06**: Prediction types — Probability, LogProbability, Class, RawFormulaVal, Exponent, RMSEWithUncertainty, VirtEnsembles, TotalUncertainty
- [ ] **LOSS-07**: Custom objectives/metrics — Rust trait + Python callback bridge
- [ ] **LOSS-08**: Uncertainty estimation — RMSEWithUncertainty, virtual ensembles
- [ ] **LOSS-09**: Score functions — SolarL2, Cosine, NewtonL2, NewtonCosine, LOOL2, SatL2, L2

### Advanced Feature Types

- [ ] **FEAT-01**: Text features — tokenization → BoW, NaiveBayes, BM25 calcers
- [ ] **FEAT-02**: Embedding features — LDA, KNN calcers
- [ ] **FEAT-03**: Monotone constraints (per-feature +1/-1/0)
- [ ] **FEAT-04**: Feature penalties / per-object penalties
- [ ] **FEAT-05**: Feature selection — recursive by PredictionValuesChange / LossFunctionChange / ShapValues
- [ ] **FEAT-06**: Alternative grow policies — Lossguide, Depthwise, Region

### Model, Serialization & Explainability

- [ ] **MODEL-01**: Native `.cbm` (FlatBuffers) serialization — save/load, cross-version compatible, load upstream-produced `.cbm` files
- [ ] **MODEL-02**: CPU inference/apply path (independent of the GPU toolchain)
- [ ] **MODEL-03**: Feature importance — PredictionValuesChange, LossFunctionChange, Interaction
- [ ] **MODEL-04**: SHAP values (Regular `EShapCalcType`)
- [ ] **MODEL-05**: SHAP interaction values + advanced fstr — ShapInteractionValues, PredictionDiff, SAGE
- [ ] **MODEL-06**: JSON model export (interop minimum)

### GPU Backends (CubeCL)

- [ ] **GPU-01**: CubeCL compute kernels generic over `R: Runtime` and `F: Float` — histogram, gradient/hessian, scan, reductions
- [ ] **GPU-02**: Compile-time backend selection via Cargo features (`cpu`/`wgpu`/`cuda`/`rocm`) through a single `cfg`-gated type alias — zero runtime dispatch
- [ ] **GPU-03**: `rocm`/HIP backend validated on AMD hardware (wavefront-64 safe); GPU tests run on `rocm`
- [ ] **GPU-04**: `wgpu` backend for dev machines without ROCm/CUDA
- [ ] **GPU-05**: `cuda` backend — compile-gated, untested locally
- [ ] **GPU-06**: Documented GPU tolerance — `rocm` results within a separately-stated epsilon vs the Rust CPU path (with sign-off)

### Rust Public API

- [ ] **RAPI-01**: Rust Builder-pattern public API — `CatBoostBuilder::new()...fit(&pool) -> Model`, predict
- [ ] **RAPI-02**: Typed `thiserror` error enum across the public surface

### Python Bindings & Packaging

- [ ] **PYAPI-01**: PyO3 + maturin per-backend wheels (`cpu` + `rocm` minimum), `abi3-py312`, Python ≥ 3.12
- [ ] **PYAPI-02**: scikit-learn compatible API — `fit`/`predict`/`predict_proba`/`score`/`get_params`/`set_params`; passes `check_estimator`
- [ ] **PYAPI-03**: CatBoost-native API — `Pool`, `CatBoostClassifier`/`Regressor`/`Ranker`, full parameter-name parity and default values
- [ ] **PYAPI-04**: Python input — NumPy, Pandas, Arrow, Polars with dtype/contiguity validation
- [ ] **PYAPI-05**: Typed `thiserror` → specific Python exception mapping with actionable messages
- [ ] **PYAPI-06**: Free-threaded-aware design — no GIL reliance for buffer safety (copy/quantize under GIL before release)

## v2 Requirements

Deferred beyond the parity milestone. Tracked, not in the current roadmap.

### Extended

- **EXT-01**: MonoForest / model-analysis tooling and dataset-statistics utilities
- **EXT-02**: Additional input/scale conveniences surfaced by user demand after v1

## Out of Scope

Explicitly excluded (anti-features). Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| C API / C FFI layer | PyO3 direct bindings only; redundant unsafe ABI |
| R / JVM-Scala / .NET / Node.js bindings | Rust and Python only this milestone |
| CLI application | Rust + Python APIs only |
| Model export to CoreML / ONNX / PMML / C++ / Python source | Not needed for a drop-in replacement; native `.cbm` + JSON suffice |
| Mobile / embedded targets | Desktop and server workloads only |
| Real-time / online / streaming training | Batch training only |
| Distributed multi-node training | Single-node CPU/GPU only; very high complexity |
| CUDA-direct GPU inference (upstream C path) | Replaced by the CubeCL multi-backend strategy |

## Traceability

Each v1 requirement maps to exactly one phase. See `.planning/ROADMAP.md` for phase detail.

| Requirement | Phase | Status |
|-------------|-------|--------|
| INFRA-01 | Phase 1 | Complete |
| INFRA-02 | Phase 1 | Complete |
| INFRA-03 | Phase 1 | Complete |
| INFRA-04 | Phase 1 | Complete |
| INFRA-05 | Phase 1 | Complete |
| INFRA-06 | Phase 1 | Complete |
| DATA-01 | Phase 2 | Complete |
| DATA-02 | Phase 2 | Complete |
| DATA-03 | Phase 2 | Complete |
| DATA-04 | Phase 2 | Complete |
| DATA-05 | Phase 2 | Pending |
| DATA-06 | Phase 2 | Pending |
| DATA-07 | Phase 2 | Complete |
| DATA-08 | Phase 2 | Pending |
| TRAIN-01 | Phase 3 | Pending |
| TRAIN-02 | Phase 3 | Pending |
| TRAIN-03 | Phase 3 | Pending |
| TRAIN-04 | Phase 3 | Pending |
| TRAIN-05 | Phase 3 | Pending |
| TRAIN-06 | Phase 3 | Pending |
| TRAIN-07 | Phase 3 | Pending |
| TRAIN-08 | Phase 3 | Pending |
| MODEL-01 | Phase 4 | Pending |
| MODEL-02 | Phase 4 | Pending |
| MODEL-03 | Phase 4 | Pending |
| MODEL-04 | Phase 4 | Pending |
| MODEL-06 | Phase 4 | Pending |
| LOSS-01 | Phase 4 | Pending |
| LOSS-06 | Phase 4 | Pending |
| RAPI-01 | Phase 4 | Pending |
| RAPI-02 | Phase 4 | Pending |
| ORD-01 | Phase 5 | Pending |
| ORD-02 | Phase 5 | Pending |
| ORD-03 | Phase 5 | Pending |
| ORD-04 | Phase 5 | Pending |
| ORD-05 | Phase 5 | Pending |
| LOSS-02 | Phase 6 | Pending |
| LOSS-03 | Phase 6 | Pending |
| LOSS-04 | Phase 6 | Pending |
| LOSS-05 | Phase 6 | Pending |
| LOSS-07 | Phase 6 | Pending |
| LOSS-08 | Phase 6 | Pending |
| LOSS-09 | Phase 6 | Pending |
| FEAT-01 | Phase 6 | Pending |
| FEAT-02 | Phase 6 | Pending |
| FEAT-03 | Phase 6 | Pending |
| FEAT-04 | Phase 6 | Pending |
| FEAT-05 | Phase 6 | Pending |
| FEAT-06 | Phase 6 | Pending |
| MODEL-05 | Phase 6 | Pending |
| GPU-01 | Phase 7 | Pending |
| GPU-02 | Phase 7 | Pending |
| GPU-03 | Phase 7 | Pending |
| GPU-04 | Phase 7 | Pending |
| GPU-05 | Phase 7 | Pending |
| GPU-06 | Phase 7 | Pending |
| PYAPI-01 | Phase 8 | Pending |
| PYAPI-02 | Phase 8 | Pending |
| PYAPI-03 | Phase 8 | Pending |
| PYAPI-04 | Phase 8 | Pending |
| PYAPI-05 | Phase 8 | Pending |
| PYAPI-06 | Phase 8 | Pending |

**Coverage:**

- v1 requirements: 62 total
- Mapped to phases: 62 ✓
- Unmapped: 0

**Per-phase counts:** Phase 1: 6 · Phase 2: 8 · Phase 3: 8 · Phase 4: 9 · Phase 5: 5 · Phase 6: 14 · Phase 7: 6 · Phase 8: 6 (= 62)

---
*Requirements defined: 2026-06-13*
*Last updated: 2026-06-13 after roadmap creation (traceability populated, 62/62 mapped; corrected v1 count from stale 57)*
