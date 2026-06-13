# Roadmap: catboost-rs

## Overview

catboost-rs is a numerically-exact Rust rewrite of CatBoost, oracle-tested to ≤1e-5 against the vendored upstream C++ implementation on the CPU path. The journey is sequenced as a series of oracle-passing vertical slices, narrowest-first: lay down the entire architecture (workspace, lint discipline, oracle harness, the exact `TFastRng64` PRNG port) before any algorithm is written; build the data layer (Pool, `GreedyLogSum` quantization, the single audited reduction utility) that everything depends on; establish the generic `R: Runtime` boundary with the CPU plain-boosting core and oblivious trees; lock the first full train→serialize→predict slice end-to-end against the oracle; then add the highest-risk parity slice (ordered boosting, ordered CTR, categoricals); widen to the full loss/feature matrix; add GPU backends additively on the locked generic boundary; and finally wrap the stable Rust API with dual PyO3 Python bindings and per-backend wheels. CPU is fully oracle-passing before GPU. Python is strictly downstream of a stable Rust Builder API. Each phase must be oracle-passing before the next begins.

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Workspace, Lint Discipline & Oracle Harness** - Foundational infrastructure, intermediate-oracle tooling, and the bitstream-exact `TFastRng64` port (completed 2026-06-13)
- [ ] **Phase 2: Data Layer — Pool, Quantization & Reduction** - `Pool`/`QuantizedPool`, oracle-validated `GreedyLogSum` borders, audited deterministic reduction
- [ ] **Phase 3: CPU Training Core — Plain Boosting & Oblivious Trees** - The generic `R: Runtime` boundary, plain boosting loop, symmetric trees, leaf estimation, early stopping
- [ ] **Phase 4: Model, Serialization, SHAP & Rust API (First Full Oracle Lock)** - `.cbm` serialize/apply, SHAP/fstr, binary-clf + regression end-to-end ≤1e-5, Builder API
- [ ] **Phase 5: Ordered Boosting, Ordered CTR & Categoricals (High-Risk Parity Slice)** - Multi-permutation folds, ordered boosting, ordered CTR, one-hot, feature combinations
- [ ] **Phase 6: Full Loss & Feature Parity** - Multiclass/regression/ranking losses, text/embedding features, uncertainty, advanced fstr, custom objectives
- [ ] **Phase 7: GPU Backends via CubeCL** - `rocm`/`wgpu`/`cuda` kernels on the locked generic boundary, documented GPU tolerance
- [ ] **Phase 8: Python Bindings, Dual API & Packaging** - PyO3 dual sklearn + CatBoost-native API, NumPy/Pandas/Arrow/Polars input, per-backend wheels

## Phase Details

### Phase 1: Workspace, Lint Discipline & Oracle Harness

**Goal**: The entire project scaffolding, parity-testing infrastructure, and the exact PRNG port exist so that every subsequent algorithm is born oracle-gated and lint-clean.
**Mode:** mvp
**Depends on**: Nothing (first phase)
**Requirements**: INFRA-01, INFRA-02, INFRA-03, INFRA-04, INFRA-05, INFRA-06
**Success Criteria** (what must be TRUE):

  1. A modular Cargo workspace builds with all backend crates stubbed and feature-gated (`cpu`/`wgpu`/`cuda`/`rocm`), and `cargo build`/`cargo clippy` pass on the skeleton.
  2. Library crates deny `unwrap`/`expect`/`panic`/`indexing_slicing`, and a CI check fails the build if `anyhow` appears in core library (non-test) code.
  3. The oracle harness runs against frozen, committed upstream-CatBoost fixtures (pinned seed/version, `thread_count=1`) and can assert per-stage (borders, splits, leaf values, per-iteration approximants) — not just final predictions — to ≤1e-5.
  4. The Rust `TFastRng64` port reproduces the C++ generator's raw bitstream exactly for a fixed seed (bitstream-oracle-validated).
  5. Source and test code are strictly separated (no inline `#[cfg(test)]` in production modules), enforced as a convention from the first commit.**Plans**: 3 plans in 2 waves

**Wave 1**

- [x] 01-01-PLAN.md — Walking Skeleton: workspace + lint/anyhow gates + cb-core(error) + cb-oracle(fixture/compare) + one committed .npy oracle pass + CPU CI lane

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 01-02-PLAN.md — Exact TFastRng64 PRNG port in cb-core, bitstream-validated against vendored fast_ut.cpp vectors
- [x] 01-03-PLAN.md — Six feature-gated stub crates + Python catboost==1.2.10 oracle generator + frozen input corpus + per-stage comparator proof + source/test-separation gate

### Phase 2: Data Layer — Pool, Quantization & Reduction

**Goal**: The leaf data crate everything depends on exists and is oracle-locked, so no downstream tree can be poisoned by a border or summation-order divergence.
**Mode:** mvp
**Depends on**: Phase 1
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04, DATA-05, DATA-06, DATA-07, DATA-08
**Success Criteria** (what must be TRUE):

  1. A `Pool` holds float/categorical/text/embedding columns plus label, weights, group_id, subgroup_id, pairs, and baseline; `QuantizedPool` stores `u8`/`u16` bin indices in columnar SoA with buffers reusable across rounds.
  2. `GreedyLogSum` border selection produces a per-feature border set that matches upstream exactly (including NaN/duplicate columns and `<`/`<=` assignment semantics), validated by the intermediate oracle.
  3. Missing-value handling (`NanMode` Min/Max/Forbidden) and categorical feature hashing match upstream behavior.
  4. A single audited deterministic reduction utility matches the C++ `double` accumulator type and summation order, and is the only summation primitive used in the codebase.
  5. NumPy is ingested zero-copy and Arrow/Polars with dtype/contiguity validation; per-object/per-class weights and auto class weights (Balanced/SqrtBalanced) are computed correctly.

**Plans**: 5 plans in 5 waves

**Wave 1**

- [x] 02-01-PLAN.md — Foundation: cb-core reduction primitive + D-08 CI-grep gate + Wave-0 oracle fixtures (numeric_nan/borders/cat-hash/class-weights) resolving Assumptions A1–A5

**Wave 2** *(blocked on Wave 1)*

- [ ] 02-02-PLAN.md — Pool (owned columns + IngestSource seam) + GreedyLogSum borders oracle-locked on numeric_tiny

**Wave 3** *(blocked on Wave 2)*

- [ ] 02-03-PLAN.md — NanMode sentinel + strict value>border + QuantizedPool SoA width enum + pool.quantize driver, oracle-locked on numeric_nan

**Wave 4** *(blocked on Wave 3)*

- [ ] 02-04-PLAN.md — CityHash64 port + CalcCatFeatureHash + first-seen perfect-hash remap, oracle-locked on the categorical corpus

**Wave 5** *(blocked on Wave 4)*

- [ ] 02-05-PLAN.md — Arrow/Polars ingestion (typed CbError taxonomy) + Balanced/SqrtBalanced auto class weights, oracle-locked; full workspace suite green

### Phase 3: CPU Training Core — Plain Boosting & Oblivious Trees

**Goal**: A user can train a plain-boosted model of symmetric oblivious trees on the CPU and have every per-tree split, leaf value, and per-iteration approximant match upstream to ≤1e-5.
**Mode:** mvp
**Depends on**: Phase 2
**Requirements**: TRAIN-01, TRAIN-02, TRAIN-03, TRAIN-04, TRAIN-05, TRAIN-06, TRAIN-07, TRAIN-08
**Success Criteria** (what must be TRUE):

  1. The generic `R: Runtime` + `F: Float` compute boundary is established in `cb-compute` with the `cpu` backend (`SelectedRuntime = CpuRuntime`) and the histogram/gradient/scan/reduction kernels run.
  2. A plain gradient-boosting train loop (`iterations`, `learning_rate`, `depth`) builds symmetric oblivious trees with leaf estimation (Gradient, Newton, Exact, Simple), and per-tree split + leaf-value intermediate oracles pass ≤1e-5 vs C++.
  3. Bootstrap/sampling (Poisson, Bayesian, Bernoulli, MVS, No; `subsample`; object/group units) and regularization (`l2_leaf_reg`, `random_strength`, `bagging_temperature`) are seeded by the Phase 1 `TFastRng64` port and reproduce upstream draws.
  4. Overfitting detection / early stopping (Wilcoxon/IncToDec/Iter, `od_pval`/`od_wait`, `use_best_model`) and per-iteration eval-set metric logging (multiple eval sets, `eval_metric`) behave correctly.
  5. Automatic learning-rate selection from dataset size matches upstream, and a first end-to-end CPU train→predict cycle runs.

**Plans**: TBD

### Phase 4: Model, Serialization, SHAP & Rust API (First Full Oracle Lock)

**Goal**: The first complete vertical slice — train → serialize → load → predict/explain — is oracle-locked end-to-end for numeric binary classification and regression, exposed through the public Rust Builder API.
**Mode:** mvp
**Depends on**: Phase 3
**Requirements**: MODEL-01, MODEL-02, MODEL-03, MODEL-04, MODEL-06, LOSS-01, LOSS-06, RAPI-01, RAPI-02
**Success Criteria** (what must be TRUE):

  1. Native `.cbm` (FlatBuffers) serialization round-trips, and a model produced by upstream CatBoost can be loaded and applied (cross-version compatible).
  2. The CPU inference/apply path runs independently of any GPU toolchain, and JSON model export is available for interop.
  3. SHAP values (Regular `EShapCalcType`) and feature importance (PredictionValuesChange, Interaction) match upstream.
  4. Binary classification (Logloss, CrossEntropy, Focal) and prediction types (Probability, LogProbability, Class, RawFormulaVal, Exponent, etc.) produce outputs matching upstream ≤1e-5.
  5. The `catboost-rs` Builder API (`CatBoostBuilder::new()...fit(&pool) -> Model`, predict) with a typed `thiserror` error enum drives a full numeric-only binary-clf + regression train→serialize→predict oracle pass ≤1e-5 vs C++.

**Plans**: TBD

### Phase 5: Ordered Boosting, Ordered CTR & Categoricals (High-Risk Parity Slice)

**Goal**: CatBoost's defining anti-leakage algorithms — ordered boosting and ordered CTR — plus native categorical handling produce models matching upstream ≤1e-5, with per-object intermediate oracles confirming no silent leakage.
**Mode:** mvp
**Depends on**: Phase 4
**Requirements**: ORD-01, ORD-02, ORD-03, ORD-04, ORD-05
**Success Criteria** (what must be TRUE):

  1. Multi-permutation fold machinery (`fold_count` permutations, `TFold`-equivalent bookkeeping) is seeded by `TFastRng64` and reproduces upstream permutations exactly.
  2. `EBoostingType::Ordered` trains with exact prefix boundaries and the exact prior formula `(sumTarget + prior) / (sumCount + priorWeight)`, and a per-object target-statistic intermediate oracle passes (no leakage signature in train metrics).
  3. Ordered CTR computes `Borders`, `Buckets`, `BinarizedTargetMeanValue`, `FloatTargetMeanValue`, `Counter`, and `FeatureFreq` with priors, matching upstream.
  4. One-hot encoding for low-cardinality categoricals (`one_hot_max_size` threshold) selects the correct encoding path.
  5. Feature combinations (tensor CTRs — `SimpleCtrs`/`CombinationCtrs`, `max_ctr_complexity` control) produce models matching upstream ≤1e-5 on categorical datasets.

**Plans**: TBD
**Research flag**: NEEDS DEEPER RESEARCH before planning — line-by-line read of `approx_calcer.cpp` + `online_ctr.*`; design the per-object intermediate-oracle schema (which values to extract and compare) before writing implementation code.

### Phase 6: Full Loss & Feature Parity

**Goal**: The full CatBoost loss/metric and advanced-feature surface is reached additively, each loss and feature type passing its own oracle before the next is added.
**Mode:** mvp
**Depends on**: Phase 5
**Requirements**: LOSS-02, LOSS-03, LOSS-04, LOSS-05, LOSS-07, LOSS-08, LOSS-09, FEAT-01, FEAT-02, FEAT-03, FEAT-04, FEAT-05, FEAT-06, MODEL-05
**Success Criteria** (what must be TRUE):

  1. Multiclass (MultiClass softmax, MultiClassOneVsAll), multilabel (MultiLogloss, MultiCrossEntropy), and the full regression matrix (RMSE, MAE, Quantile, MultiQuantile, LogCosh, Huber, Poisson, Tweedie, MAPE, MSLE, Lq, Expectile, etc.) each pass their oracle ≤1e-5.
  2. Ranking losses (YetiRank/Pairwise, PairLogit/Pairwise, QueryRMSE, QuerySoftMax, LambdaMart, StochasticRank) and ranking metrics (NDCG, DCG, MAP, MRR, ERR, PFound, PrecisionAt, RecallAt, QueryAUC) work over group_id/subgroup_id/pairs.
  3. Text features (tokenization → BoW, NaiveBayes, BM25) and embedding features (LDA, KNN calcers) produce upstream-matching encodings.
  4. Uncertainty estimation (RMSEWithUncertainty, virtual ensembles), score functions (SolarL2, Cosine, NewtonL2, NewtonCosine, LOOL2, SatL2, L2), and custom objectives/metrics (Rust trait + Python callback bridge) work.
  5. Monotone constraints, feature penalties, feature selection (recursive by PredictionValuesChange/LossFunctionChange/ShapValues), alternative grow policies (Lossguide, Depthwise, Region), and advanced fstr (ShapInteractionValues, PredictionDiff, SAGE) match upstream.

**Plans**: TBD

### Phase 7: GPU Backends via CubeCL

**Goal**: GPU training runs on the `rocm`/`wgpu`/`cuda` backends purely additively on the locked generic boundary, within a documented and signed-off GPU tolerance versus the Rust CPU path.
**Mode:** mvp
**Depends on**: Phase 6
**Requirements**: GPU-01, GPU-02, GPU-03, GPU-04, GPU-05, GPU-06
**Success Criteria** (what must be TRUE):

  1. CubeCL kernels generic over `R: Runtime` and `F: Float` (histogram, gradient/hessian, scan, reductions) compile and run, with `cb-core`/`cb-model` unchanged from their Phase 3–6 form.
  2. Compile-time backend selection via Cargo features (`cpu`/`wgpu`/`cuda`/`rocm`) flows through a single `cfg`-gated type alias with zero runtime dispatch.
  3. The `rocm`/HIP backend is validated on AMD hardware (wavefront-64 safe; no warp-size assumptions), and GPU tests run on `rocm`.
  4. The `wgpu` backend runs on dev machines without ROCm/CUDA, and the `cuda` backend compiles behind its feature gate (untested locally).
  5. A documented GPU tolerance is established and signed off: `rocm` results fall within a separately-stated epsilon vs the Rust CPU path (not vs the C++ CPU oracle).

**Plans**: TBD
**Research flag**: NEEDS DEEPER RESEARCH before planning — spike `cubecl-hip` kernel coverage (histogram atomics, prefix scan, reductions) at cubecl 0.10.0; validate wavefront-64 reduction determinism; match `cubecl-hip-sys` HIP version to the test machine; set the concrete GPU epsilon and get sign-off.

### Phase 8: Python Bindings, Dual API & Packaging

**Goal**: Python ML practitioners can drop catboost-rs into existing scikit-learn or CatBoost workflows via a dual-surface PyO3 binding distributed as per-backend wheels.
**Mode:** mvp
**Depends on**: Phase 7
**Requirements**: PYAPI-01, PYAPI-02, PYAPI-03, PYAPI-04, PYAPI-05, PYAPI-06
**Success Criteria** (what must be TRUE):

  1. The scikit-learn-compatible API (`fit`/`predict`/`predict_proba`/`score`/`get_params`/`set_params`) passes `check_estimator`.
  2. The CatBoost-native API (`Pool`, `CatBoostClassifier`/`Regressor`/`Ranker`) has full parameter-name parity and matching default values with upstream.
  3. Python input accepts NumPy, Pandas, Arrow, and Polars with dtype/contiguity validation, copying/quantizing under the GIL before release (free-threaded-aware; no GIL reliance for buffer safety).
  4. Typed `thiserror` errors map to specific Python exceptions with actionable messages.
  5. Per-backend wheels (`cpu` + `rocm` minimum) build via `maturin --features <backend>` with `abi3-py312` on Python ≥ 3.12.

**Plans**: TBD
**Research flag**: NEEDS RESEARCH before planning — confirm current PyO3/maturin `abi3`/`abi3t` status for Python 3.12–3.15 (PEP 803: `abi3t` only on 3.15+; free-threaded 3.12–3.14 needs version-specific wheels); verify the pinned `pyo3 0.28.3` + `rust-numpy 0.28` + `ndarray 0.17` triad before committing the ABI strategy.

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Workspace, Lint & Oracle Harness | 3/3 | Complete   | 2026-06-13 |
| 2. Data Layer — Pool, Quantization & Reduction | 1/5 | In Progress|  |
| 3. CPU Training Core — Plain Boosting & Trees | 0/TBD | Not started | - |
| 4. Model, Serialization, SHAP & Rust API | 0/TBD | Not started | - |
| 5. Ordered Boosting, Ordered CTR & Categoricals | 0/TBD | Not started | - |
| 6. Full Loss & Feature Parity | 0/TBD | Not started | - |
| 7. GPU Backends via CubeCL | 0/TBD | Not started | - |
| 8. Python Bindings, Dual API & Packaging | 0/TBD | Not started | - |
