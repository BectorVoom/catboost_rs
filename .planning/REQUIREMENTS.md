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
- [x] **DATA-05**: Categorical feature hashing
- [x] **DATA-06**: Zero-copy NumPy ingestion and Arrow/Polars ingestion with dtype/contiguity validation; copy-in path for training
- [x] **DATA-07**: Single audited deterministic reduction utility matching the C++ `double` accumulator type and summation order
- [x] **DATA-08**: Per-object / per-class weights and auto class weights (Balanced/SqrtBalanced)

### CPU Training Core

- [x] **TRAIN-01**: Plain gradient boosting train loop (`iterations`, `learning_rate`, `depth`)
- [x] **TRAIN-02**: Symmetric (oblivious) decision trees — the core CatBoost tree structure
- [~] **TRAIN-03**: Leaf value estimation — Gradient (done, Plan 01); Newton, Exact, Simple (Plan 02)
- [x] **TRAIN-04**: Bootstrap / sampling — Poisson, Bayesian, Bernoulli, MVS, No; `subsample`; object/group sampling units (Plan 03; No/Bernoulli/MVS oracle-locked ≤1e-5, Poisson CPU-rejected per upstream, Bayesian first-tree + draw-sequence locked)
- [x] **TRAIN-05**: Regularization — `l2_leaf_reg`, `random_strength`, `bagging_temperature`
- [x] **TRAIN-06**: Overfitting detection and early stopping — Wilcoxon/IncToDec/Iter, `od_pval`/`od_wait`, `use_best_model`
- [x] **TRAIN-07**: Eval-set validation metrics logged per iteration (multiple eval sets, `eval_metric`)
- [x] **TRAIN-08**: Automatic learning-rate selection from dataset size

### Ordered Algorithms (Signature Features)

- [x] **ORD-01**: Multi-permutation fold machinery (`fold_count` permutations, `TFold`-equivalent bookkeeping)
- [x] **ORD-02**: Ordered boosting (`EBoostingType::Ordered`) with exact prefix boundaries, per-object intermediate oracle
- [x] **ORD-03**: Ordered target statistics / CTR — `Borders`, `Buckets`, `BinarizedTargetMeanValue`, `FloatTargetMeanValue`, `Counter`, `FeatureFreq` with priors
- [x] **ORD-04**: One-hot encoding for low-cardinality categoricals (`one_hot_max_size` threshold)
- [x] **ORD-05**: Feature combinations (tensor CTRs) — `SimpleCtrs`/`CombinationCtrs`, `max_ctr_complexity` control

### Losses, Metrics & Prediction

- [x] **LOSS-01**: Binary classification — Logloss, CrossEntropy, Focal (Plan 04-02: CrossEntropy + Focal der1/der2 transcribed from `error_functions.{h,cpp}` and oracle-locked; binclf trains under all three losses with splits/leaf-values/staged-approx ≤1e-5)
- [x] **LOSS-02**: Multiclass (MultiClass softmax, MultiClassOneVsAll) and multilabel (MultiLogloss, MultiCrossEntropy) — multiclass members in 06.2-03, multilabel members in 06.2-04; all four pass per-stage oracle ≤1e-5 vs catboost 1.2.10
- [x] **LOSS-03**: Regression matrix — RMSE, MAE, Quantile, MultiQuantile, LogCosh, Huber, Poisson, Tweedie, MAPE, MSLE, Lq, Expectile, etc.
- [x] **LOSS-04**: Ranking losses — YetiRank(/Pairwise), PairLogit(/Pairwise), QueryRMSE, QuerySoftMax, LambdaMart, StochasticRank
- [x] **LOSS-05**: Ranking metrics — NDCG, DCG, MAP, MRR, ERR, PFound, PrecisionAt, RecallAt, QueryAUC — all nine eval-only on the widened EvalMetric::eval_grouped seam (06.3-05), oracle-locked ≤1e-5 vs catboost 1.2.10
- [~] **LOSS-06**: Prediction types — Probability, LogProbability, Class, RawFormulaVal, Exponent, RMSEWithUncertainty, VirtEnsembles, TotalUncertainty (Plan 04-02: the five in-scope deterministic types — RawFormulaVal/Probability/LogProbability/Class/Exponent — are implemented and oracle-locked ≤1e-5; the uncertainty types RMSEWithUncertainty/VirtEnsembles/TotalUncertainty are deferred to Phase 6 per D-10)
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

- [x] **MODEL-01**: Native `.cbm` (FlatBuffers) serialization — save/load, cross-version compatible, load upstream-produced `.cbm` files (Plan 04-03: `cb-model::cbm::{save_cbm, load_cbm}` — CBM1 magic + ui32 LE size + FlatBuffers TModelCore; semantic round-trip + upstream catboost 1.2.10 binclf/regression `.cbm` load applies ≤1e-5; malformed input → typed ModelError, never panics. Built on the 04-01 committed flatc bindings + canonical cb-model::Model)
- [x] **MODEL-02**: CPU inference/apply path (independent of the GPU toolchain) (Plan 04-02: pure-Rust `cb-model::predict_raw` — strict-> binarize, forward-bit leaf index, bias + `sum_f64` over leaf values; imports no backend/cubecl symbol; oracle-locked ≤1e-5 vs upstream)
- [~] **MODEL-03**: Feature importance — PredictionValuesChange, LossFunctionChange, Interaction (Plan 04-04: `cb-model::prediction_values_change` (CalcEffect, Σ=100) + `cb-model::interaction` (CalcMostInteractingFeatures + CalcFeatureInteraction) oracle-locked ≤1e-5 vs upstream `feature_importance/*.npy`. PARTIAL — LossFunctionChange deferred per D-12, out of scope this phase)
- [x] **MODEL-04**: SHAP values (Regular `EShapCalcType`) (Plan 04-04: `cb-model::shap_values` — regular TreeSHAP per-object [n_features+1] matrix (trailing column = Σ_trees meanValue + bias) transcribed verbatim from `shap_values.cpp` + `shap_prepared_trees.cpp`; oracle-locked ≤1e-5 vs upstream `feature_importance/shap_values.npy` AND the local-accuracy invariant Σshap == predict_raw holds for every object, D-11)
- [ ] **MODEL-05**: SHAP interaction values + advanced fstr — ShapInteractionValues, PredictionDiff, SAGE
- [x] **MODEL-06**: JSON model export (interop minimum) (Plan 04-03: `cb-model::json::{save_json, load_json}` on the upstream model.json schema — per-tree NESTED leaf_weights, scale_and_bias=[1,[bias]]; save_json round-trips through the cb-oracle model_json parser (D-04) and upstream binclf/regression model.json load applies ≤1e-5; malformed JSON → typed ModelError)

### GPU Backends (CubeCL)

- [ ] **GPU-01**: CubeCL compute kernels generic over `R: Runtime` and `F: Float` — histogram, gradient/hessian, scan, reductions
- [ ] **GPU-02**: Compile-time backend selection via Cargo features (`cpu`/`wgpu`/`cuda`/`rocm`) through a single `cfg`-gated type alias — zero runtime dispatch
- [ ] **GPU-03**: `rocm`/HIP backend validated on AMD hardware (wavefront-64 safe); GPU tests run on `rocm`
- [ ] **GPU-04**: `wgpu` backend for dev machines without ROCm/CUDA
- [ ] **GPU-05**: `cuda` backend — compile-gated, untested locally
- [ ] **GPU-06**: Documented GPU tolerance — `rocm` results within a separately-stated epsilon vs the Rust CPU path (with sign-off)

### Rust Public API

- [x] **RAPI-01**: Rust Builder-pattern public API — `CatBoostBuilder::new()...fit(&pool) -> Model`, predict
- [x] **RAPI-02**: Typed `thiserror` error enum across the public surface

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
| DATA-05 | Phase 2 | Complete |
| DATA-06 | Phase 2 | Complete |
| DATA-07 | Phase 2 | Complete |
| DATA-08 | Phase 2 | Complete |
| TRAIN-01 | Phase 3 | Complete |
| TRAIN-02 | Phase 3 | Complete |
| TRAIN-03 | Phase 3 | Complete (Gradient/Newton/Exact/Simple, Plan 02) |
| TRAIN-04 | Phase 3 | Complete (No/Bernoulli/MVS oracle-locked; Poisson CPU-rejected; Bayesian first-tree + draw-sequence, Plan 03) |
| TRAIN-05 | Phase 3 | Complete |
| TRAIN-06 | Phase 3 | Complete |
| TRAIN-07 | Phase 3 | Complete |
| TRAIN-08 | Phase 3 | Complete |
| MODEL-01 | Phase 4 | Complete (04-03: .cbm save/load, semantic round-trip + upstream 1.2.10 load ≤1e-5, malformed-input typed errors) |
| MODEL-02 | Phase 4 | Complete (04-02: pure-Rust apply path, oracle-locked ≤1e-5) |
| MODEL-03 | Phase 4 (+6.6) | Partial (04-04: PredictionValuesChange + Interaction oracle-locked ≤1e-5; LossFunctionChange completes in Phase 6.6 per D-12) |
| MODEL-04 | Phase 4 | Complete (04-04: regular TreeSHAP matrix + local-accuracy invariant, oracle-locked ≤1e-5, D-11) |
| MODEL-06 | Phase 4 | Complete (04-03: model.json export/import, round-trips through cb-oracle parser + upstream load ≤1e-5) |
| LOSS-01 | Phase 4 | Complete (04-02: CrossEntropy + Focal der1/der2 oracle-locked; binclf trains under all three losses) |
| LOSS-06 | Phase 4 (+6.4) | In progress (04-02: 5 in-scope prediction types oracle-locked; uncertainty types RMSEWithUncertainty/VirtEnsembles/TotalUncertainty complete in Phase 6.4 per D-10) |
| RAPI-01 | Phase 4 | Complete |
| RAPI-02 | Phase 4 | Complete |
| ORD-01 | Phase 5 | Complete |
| ORD-02 | Phase 5 | Complete |
| ORD-03 | Phase 5 | Complete |
| ORD-04 | Phase 5 | Complete |
| ORD-05 | Phase 5 | Complete |
| LOSS-02 | Phase 6.2 | Complete (06.2-03 multiclass + 06.2-04 multilabel) |
| LOSS-03 | Phase 6.1 (+6.2) | Complete. Scalar matrix ✓ (Wave 1 ✓ LogCosh/Lq/Huber/Expectile; Wave 2 ✓ Poisson/Tweedie/MAPE + MSLE metric-only; Wave 3 ✓ Quantile{α,δ} — MAE==Quantile{0.5}, α-threaded Exact leaf, oracle ≤1e-5; Phase 6.1 scalar LOSS-03 COMPLETE). MultiQuantile multi-output member ✓ (06.2-05 — K independent Quantile dims on the N-dim foundation, per-dim alpha[d] der + Exact weighted-alpha[d]-quantile leaf reused verbatim, leaf.rs unchanged; per-stage oracle ≤1e-5 vs catboost 1.2.10). LOSS-03 scalar+multi matrix CLOSED. |
| LOSS-04 | Phase 6.3 | In progress (06.3-01: grouped der seam landed. 06.3-02 (Wave A): **QueryRMSE + QuerySoftMax** der arms wired + trained end-to-end ≤1e-5. 06.3-03 (Wave B): **LambdaMart** trained end-to-end per-stage oracle ≤1e-5; **PairLogit/PairLogitPairwise** der + Cholesky pairwise-leaf path land + unit-tested, per-stage ORACLES DEFERRED on a pair-weight normalization gap. 06.3-04 (Wave C): **YetiRank/YetiRankPairwise + StochasticRank** der+sampler+variants+boosting-wiring LAND; the RNG STREAM (2-level TFastRng64 + Gumbel / std_normal Monte-Carlo) is VALIDATED ≤1e-5 against TWO standalone OFFLINE instrumented generators that self-oracle bit-for-bit vs cb-core::TFastRng64/std_normal; 3 per-stage RNG-draw-log oracles GREEN, NO #[ignore]/NO weakened tolerance. The end-to-end per-stage TRAINER fixtures (YetiRank/YetiRankPairwise/StochasticRank + the PairLogit follow-up) are DEFERRED on an infeasible instrumented catboost-trainer build (path c: toolchain absent + disk NO-GO; escalate-don't-weaken, D-6.3-03b; deferred-items.md). 06.3-10 (Wave-6 gap closure): the instrumented catboost 1.2.10 `_catboost.so` trainer (CB_INSTRUMENT_LOG per-leaf der1/der2 + YetiRank/StochasticRank RNG draws) **BUILT + LINKED + runs + emits the log — GO** (disk now 67GB free, no longer the link-failure regime that forced the prior path-c deferral); a sudo-free re-runnable `build_instrumented_trainer.sh` + `instrumented_trainer_STATUS.md` capture it (awaiting blocking-human GO sign-off). This GO UNBLOCKS the deferred trainer-fixture closures: 06.3-13 (PairLogit per-leaf SumDer2 oracle) + 06.3-14 (YetiRank/YetiRankPairwise/StochasticRank end-to-end per-stage fixtures + D-07 RNG-draw validation). 06.3-11 (Wave-6 gap closure, trainer-INDEPENDENT): ranking_der.rs index-panic hardening — WR-02 (the 6 raw cum_sum/cum_sum_up/cum_sum_low subscripts in `calc_dcg_metric_diff` → bounds-checked `.get(..)`, matching CR-01's pos_weights discipline) + the 42 `clippy::indexing_slicing` sites in `stochastic_rank_group_der` (42→0, all bounds-checked, NO unwrap/expect, BIT-IDENTICAL) **CLOSED** (21ddf5a/0fdd507), oracle-revalidated lambdamart 1/1 + stochasticrank 2/2 ≤1e-5, no parity regression. 06.3-15 (Wave-7 gap closure, ENABLER): the pairwise SPLIT-scorer subsystem — the Rule-4 architectural piece the 06.3-13/14 verification isolated as the SPLIT-SELECTION divergence deferring **PairLogitPairwise** + **YetiRankPairwise** — LANDS as a pure cb-compute library (03ae077/653e083): `crates/cb-compute/src/pairwise_scoring.rs` = `compute_der_sums` + `compute_pair_weight_statistics` + `calculate_pairwise_score` (OneFeature float path; `TPairwiseScoreCalcer`/`CalculatePairwiseScore`, pairwise_scoring.{h,cpp}) + `BucketPairWeightStatistics`, reusing the in-house `crate::pairwise_cholesky_solve` via a cb-compute-local leaf-solve twin (NO new crate), all reductions via `cb_core::sum_f64`, bounds-guarded → `CbError::OutOfRange`, self-oracled bit-for-bit vs hand-derived references + an independent Gaussian-elimination reference solver ≤1e-9 (cb-compute lib 131/131). Library-only — NO tree wiring (06.3-16), NO fixture, NO tolerance touched. Remaining: 13/14 must consume the 06.3-10 GO artifact to land the deferred Wave-B/C trainer-fixture oracles ≤1e-5; 16 wires the 15 scorer into tree.rs + removes #[ignore] from pairlogit_pairwise_oracle_test.rs; 17 the YetiRankPairwise fixture (truths #5/#7 still deferred). |
| LOSS-05 | Phase 6.3 | Complete (06.3-05, Wave D): all 9 ranking metrics — NDCG/DCG/MAP/MRR/ERR/PFound/PrecisionAt/RecallAt/QueryAUC — land eval-only on the widened EvalMetric::eval_grouped seam (D-6.3-05; flat eval byte-identical D-04), shared compare_docs tie-break transcribed once, per-metric oracle ≤1e-5 vs catboost 1.2.10 (default + top=2; QueryAUC Ranking+Classic). |
| LOSS-07 | Phase 6.4 | Pending (Rust trait; Python callback bridge → Phase 8 per D-09) |
| LOSS-08 | Phase 6.4 | Pending |
| LOSS-09 | Phase 6.4 | Pending |
| FEAT-01 | Phase 6.5 | Pending |
| FEAT-02 | Phase 6.5 | Pending |
| FEAT-03 | Phase 6.6 | Pending |
| FEAT-04 | Phase 6.6 | Pending |
| FEAT-05 | Phase 6.6 | Pending |
| FEAT-06 | Phase 6.6 | Pending |
| MODEL-05 | Phase 6.6 | Pending |
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
