# Roadmap: catboost-rs

## Overview

catboost-rs is a numerically-exact Rust rewrite of CatBoost, oracle-tested to ≤1e-5 against the vendored upstream C++ implementation on the CPU path. The journey is sequenced as a series of oracle-passing vertical slices, narrowest-first: lay down the entire architecture (workspace, lint discipline, oracle harness, the exact `TFastRng64` PRNG port) before any algorithm is written; build the data layer (Pool, `GreedyLogSum` quantization, the single audited reduction utility) that everything depends on; establish the generic `R: Runtime` boundary with the CPU plain-boosting core and oblivious trees; lock the first full train→serialize→predict slice end-to-end against the oracle; then add the highest-risk parity slice (ordered boosting, ordered CTR, categoricals); widen to the full loss/feature matrix; add GPU backends additively on the locked generic boundary; and finally wrap the stable Rust API with dual PyO3 Python bindings and per-backend wheels. CPU is fully oracle-passing before GPU. Python is strictly downstream of a stable Rust Builder API. Each phase must be oracle-passing before the next begins.

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Workspace, Lint Discipline & Oracle Harness** - Foundational infrastructure, intermediate-oracle tooling, and the bitstream-exact `TFastRng64` port (completed 2026-06-13)
- [x] **Phase 2: Data Layer — Pool, Quantization & Reduction** - `Pool`/`QuantizedPool`, oracle-validated `GreedyLogSum` borders, audited deterministic reduction (completed 2026-06-13)
- [x] **Phase 3: CPU Training Core — Plain Boosting & Oblivious Trees** - The generic `R: Runtime` boundary, plain boosting loop, symmetric trees, leaf estimation, early stopping (completed 2026-06-13)
- [x] **Phase 4: Model, Serialization, SHAP & Rust API (First Full Oracle Lock)** - `.cbm` serialize/apply, SHAP/fstr, binary-clf + regression end-to-end ≤1e-5, Builder API
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

- [x] 02-02-PLAN.md — Pool (owned columns + IngestSource seam) + GreedyLogSum borders oracle-locked on numeric_tiny

**Wave 3** *(blocked on Wave 2)*

- [x] 02-03-PLAN.md — NanMode sentinel + strict value>border + QuantizedPool SoA width enum + pool.quantize driver, oracle-locked on numeric_nan

**Wave 4** *(blocked on Wave 3)*

- [x] 02-04-PLAN.md — CityHash64 port + CalcCatFeatureHash + first-seen perfect-hash remap, oracle-locked on the categorical corpus (corrected cat_hash fixtures from the vendored city.cpp; the CTR-hash extraction was the wrong target)

**Wave 5** *(blocked on Wave 4)*

- [x] 02-05-PLAN.md — Arrow/Polars ingestion (typed CbError taxonomy) + Balanced/SqrtBalanced auto class weights, oracle-locked; full workspace suite green

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

**Plans**: 9 plans in 9 waves (03-08 gap-closure for CR-01)

**Wave 1**

- [x] 03-00-PLAN.md — Wave-0 foundation: install cubecl 0.10.0/bytemuck (cb-backend only, D-03), prove the #[cube] CpuRuntime gradient seam, add cb-oracle model.json parser, generate RMSE + Logloss training-oracle fixtures (simplified isolating params, D-07) — _SUMMARY 03-00 (4 tasks, Nyquist Wave-0 signed off)_

**Wave 2** *(blocked on Wave 1)*

- [x] 03-01-PLAN.md — First end-to-end slice: cb-compute R:Runtime/F:Float boundary + loss/histogram/score/leaf, cb-backend CpuRuntime impl, cb-train plain boosting + oblivious trees (Gradient leaf), oracle-locked RMSE + Logloss splits/leaves/staged ≤1e-5 — _SUMMARY 03-01 (4 tasks; TRAIN-01/02 complete, TRAIN-03 Gradient)_

**Wave 3** *(blocked on Wave 2)*

- [x] 03-02-PLAN.md — Newton/Exact/Simple leaf-estimation methods (completes TRAIN-03, D-09), each oracle-locked on leaf values

**Wave 4** *(blocked on Wave 3)*

- [x] 03-03-PLAN.md — Bootstrap/sampling (No/Bayesian/Bernoulli/MVS/Poisson, subsample) seeded by TFastRng64 with exact per-block reseed order (TRAIN-04); No/Bernoulli/MVS oracle-locked ≤1e-5 end-to-end, Poisson CPU-rejected (upstream-faithful), Bayesian first-tree + draw-sequence locked (multi-tree residual deferred) — _SUMMARY 03-03 (2 tasks; TRAIN-04 complete)_

**Wave 5** *(blocked on Wave 4)*

- [x] 03-04-PLAN.md — Full regularization: random_strength normal-draw perturbation (cb-core::normal port), bagging_temperature, l2_leaf_reg (TRAIN-05)

**Wave 6** *(blocked on Wave 5)*

- [x] 03-05-PLAN.md — Overfitting detection / early stopping (IncToDec/Iter/Wilcoxon, od_pval/od_wait, use_best_model) (TRAIN-06)

**Wave 7** *(blocked on Wave 6)*

- [x] 03-06-PLAN.md — Per-iteration eval-set metric logging (multiple eval sets, eval_metric) (TRAIN-07)

**Wave 8** *(blocked on Wave 7)*

- [x] 03-07-PLAN.md — Automatic learning-rate selection (TAutoLRParamsGuesser) + first end-to-end auto-LR train→predict (TRAIN-08)

**Wave 9** *(blocked on Wave 5; gap closure)*

- [x] 03-08-PLAN.md — Gap closure CR-01: feed score_st_dev the FULL-fold weighted_der1 (not the masked score_weighted_der1) + new cross-scenario oracle (random_strength=1.0 + Bernoulli, subsample=0.7) locking first-tree splits/leaves ≤1e-5 (TRAIN-05) — _SUMMARY 03-08 (3 tasks; CR-01 closed, fix verified vs upstream greedy_tensor_search.cpp:99; RED→GREEN locked at the cb-compute unit boundary — first-tree end-to-end cannot isolate the std-dev bias on numeric_tiny, entangled with the D-11 draw-stream residual)_

### Phase 4: Model, Serialization, SHAP & Rust API (First Full Oracle Lock)

**Goal**: The first complete vertical slice — train → serialize → load → predict/explain — is oracle-locked end-to-end for numeric binary classification and regression, exposed through the public Rust Builder API.
**Mode:** mvp
**Depends on**: Phase 3
**Requirements**: MODEL-01, MODEL-02, MODEL-03, MODEL-04, MODEL-06, LOSS-01, LOSS-06, RAPI-01, RAPI-02
**Note**: MODEL-03 is only PARTIALLY delivered this phase — PredictionValuesChange + Interaction land here; `LossFunctionChange` is deferred to a later advanced-fstr phase (D-12).
**Success Criteria** (what must be TRUE):

  1. Native `.cbm` (FlatBuffers) serialization round-trips, and a model produced by upstream CatBoost can be loaded and applied (cross-version compatible).
  2. The CPU inference/apply path runs independently of any GPU toolchain, and JSON model export is available for interop.
  3. SHAP values (Regular `EShapCalcType`) and feature importance (PredictionValuesChange, Interaction) match upstream.
  4. Binary classification (Logloss, CrossEntropy, Focal) and prediction types (Probability, LogProbability, Class, RawFormulaVal, Exponent, etc.) produce outputs matching upstream ≤1e-5.
  5. The `catboost-rs` Builder API (`CatBoostBuilder::new()...fit(&pool) -> Model`, predict) with a typed `thiserror` error enum drives a full numeric-only binary-clf + regression train→serialize→predict oracle pass ≤1e-5 vs C++.

**Plans**: 5 plans in 5 waves
Plans:
**Wave 1**

- [x] 04-01-PLAN.md — Wave-0 prerequisite: capture per-leaf weights in cb-train, re-home canonical Model into cb-model (leaf_weights + float_feature_borders), extend model_json with leaf_weights, commit flatc-generated FlatBuffers bindings, stage offline fixtures

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 04-02-PLAN.md — Pure-Rust CPU apply path (MODEL-02) + prediction-type transforms (LOSS-06) + CrossEntropy/Focal losses (LOSS-01) — _SUMMARY 04-02 (2 tasks; MODEL-02 + LOSS-01 complete, LOSS-06 partial — 5 deterministic types locked, uncertainty types deferred to Phase 6 per D-10)_

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 04-03-PLAN.md — .cbm save/load (FlatBuffers framing) + model.json export/import, semantic round-trip + upstream 1.2.10 load (MODEL-01, MODEL-06) — _SUMMARY 04-03 (2 tasks; MODEL-01 + MODEL-06 complete — .cbm + model.json round-trip + upstream binclf/regression load ≤1e-5, malformed-input typed errors V5; 1 Rule-1 fix: bias from MultiBias[0])_

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 04-04-PLAN.md — Regular TreeSHAP + local-accuracy lock (MODEL-04) + PredictionValuesChange/Interaction (MODEL-03 partial)

**Wave 5** *(blocked on Wave 4 completion)*

- [x] 04-05-PLAN.md — Public CatBoostBuilder + Model facade + CatBoostError + end-to-end binclf+regression train→serialize→load→predict oracle (RAPI-01, RAPI-02) — _SUMMARY 04-05 (2 tasks; RAPI-01 + RAPI-02 complete — CatBoostBuilder/Model/CatBoostError published facade; full numeric binclf+regression train→serialize→load→predict cycle through the public API oracle-locked ≤1e-5 vs upstream 1.2.10 (criterion 5); FeatureMismatch guard added per Rule 2)_

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

**Plans**: 12 plans in 10 waves (additive isolation ladder waves 1-6: one-hot → permutation → Plain CTR → Ordered CTR → Ordered boosting → tensor CTR; gap-closure waves 7-9: wave 7 fixed the multi-fold permutation oracle (05-07) + built the ordered split-scoring subsystem in tree.rs (05-08, the ORD-02 re-scope per STATE.md 2026-06-14); wave 8 wires ordered boosting into train() + locks the FULL multi-tree ordered e2e oracle (05-10); wave 9 wires tensor CTRs into train() (05-09) — per 05-VERIFICATION.md, the ORD-02 e2e bar is a full multi-tree hard gate, no #[ignore])

Plans:

**Wave 1**

- [x] 05-01-PLAN.md — Wave-0 oracle infra: Stage::{Permutation,OnlineCtr,OrderedApprox} + model.json ctr_data parsing + transcribed ordered_oracle.cpp (zero catboost includes) + frozen purpose-built categorical fixtures

**Wave 2** *(blocked on 05-01)*

- [x] 05-02-PLAN.md — One-hot-only first slice (ORD-04, D-04): one_hot_max_size path selection + categorical one-hot splits, oracle-locked ≤1e-5 with NO permutation present (self-oracled vs the upstream-locked float reference; commits 392fe65, da4fb30)

**Wave 3** *(blocked on 05-01, 05-02)*

- [x] 05-03-PLAN.md — Multi-permutation fold machinery (ORD-01, D-03 linchpin): TFastRng64 Fisher-Yates + TFold body/tail prefixes, permutation locked integer-exact before any value stage

**Wave 4** *(blocked on 05-01, 05-02, 05-03)*

- [x] 05-04-PLAN.md — Plain CTR (ORD-03, D-06): all six CTR types whole-set + ctr_data .cbm/model.json serde + model-side apply, locked BEFORE ordered

**Wave 5** *(blocked on 05-01, 05-03, 05-04)*

- [x] 05-05-PLAN.md — Ordered CTR + Ordered boosting (ORD-02, ORD-03 ordered): per-permutation read-before-increment + body/tail approximant, per-object intermediate oracle (indirect anchor for ordered approx)

**Wave 6** *(blocked on 05-01, 05-04, 05-05)*

- [x] 05-06-PLAN.md — Tensor / combination CTRs (ORD-05): TProjection enumeration + combined hash (ctr_provider.h CalcHash, sign-extended (ui64)(int)) + max_ctr_complexity gate; tensor CTR = the single-feature online accumulation over a combined key (D-05), oracle-locked D-03 → per-object (good,total) exact → combined OnlineCtr ≤1e-5 + model-side combined apply — _SUMMARY 05-06 (2 tasks; commits aa580ec, 659b0cc; Phase 5 additive ladder COMPLETE)_

**Wave 7** *(gap closure — blocks on existing 05-01..05-06; from 05-VERIFICATION.md gaps_found)*

- [x] 05-07-PLAN.md — GAP 3 (CR-01, ORD-01/ORD-03): fix ordered_oracle.cpp to continuous-stream multi-fold seeding (single persistent TFastRng64 across folds, matching create_folds/permutations); regenerate ordered_ctr/permutation_fold1.npy; re-key the D-03 fold-1 gate to permutations(30,2,0)[1] so the production permutations() is validated integer-exact for k≥1 (was self-consistency-only)
- [x] 05-08-PLAN.md — GAP 1 (ORD-02) Part 1/2 — ORDERED SPLIT-SCORING SUBSYSTEM: the previous 05-08 (wire approximant into leaf-update only) was UNDER-SCOPED (STATE.md 2026-06-14: ordered vs plain differ in tree STRUCTURE, not just leaf update). Build greedy_tensor_search_oblivious_ordered in tree.rs — per-segment ordered split scoring over the learning fold's BodyTailArr (segment-summed ordered L2, per-segment scaled L2 = l2*(BodySumWeight/BodyFinish), scoring.cpp:746-760), strict-first-wins preserved, degenerating to the plain search at a single full-span segment; + WR-01 dead sum_weights cleanup; unit-locked standalone

**Wave 8** *(gap closure — blocks on 05-08; shares boosting.rs + tree.rs)*

- [~] 05-10-PLAN.md — GAP 1 (ORD-02) Part 2/2 — WIRE + E2E HARD GATE: **Task 1 DONE** (eee112c) — train_with_eval_sets() branches on EBoostingType::Ordered, folds built ONCE (create_folds, FOLDS-BUILT-ONCE grep-enforced), tree STRUCTURE via greedy_tensor_search_oblivious_ordered (05-08), leaf VALUES on the averaging fold (Plain-identical); Plain unchanged; wiring test locks Ordered≠Plain. **Task 2 source DONE, oracle BLOCKED** (018c633) — gen_ordered_boost_e2e() + ordered_boost_e2e_oracle_test.rs (FULL multi-tree ≤1e-5 via cb_model::predict_raw, NO #[ignore]) authored & compile-clean, but the ordered_boost_e2e/ fixtures CANNOT be generated here (catboost==1.2.10 not importable). Oracle NOT weakened. **ORD-02 closure pending OFFLINE fixture generation** (run gen_fixtures.py on a catboost==1.2.10 machine, commit the 4 fixtures, then the e2e test must pass).

**Wave 9** *(gap closure — blocks on 05-10; shares boosting.rs + apply.rs)*

- [~] 05-09-PLAN.md — GAP 2 (ORD-05): **Tasks 1a + 1b DONE** (b2261ec, 200ffb0) — cb-model CTR-split representation RESOLVED (ModelSplit { Float, Ctr(CtrSplit) }, ObliviousTree.splits: Vec<ModelSplit>, Model.ctr_data); tensor_ctr_candidates wired into train() candidate generation under max_ctr_complexity (cb-train CtrSplitSpec + parallel ctr_splits; from_trained lifts to ModelSplit::Ctr); apply.rs evaluates ModelSplit::Ctr via combined hash (calc_cat_feature_hash + fold_cat_hash) + baked ctr_data with bounds-safe not-found→empty (T-05-09-V5) + predict_raw_cat. Split UNCHANGED; cargo check --workspace --tests clean; standalone tensor_ctr 3/3, ctr_data_roundtrip 5/5, all cb-model + cb-train float oracles green. **Task 2 source DONE, oracle BLOCKED** (10f4a92) — gen_tensor_ctr_e2e() + tensor_ctr_e2e_oracle_test.rs (FULL multi-tree ≤1e-5 via cb_model::predict_raw, NO #[ignore]) authored & compile-clean, but tensor_ctr_e2e/ fixtures CANNOT be generated here (catboost==1.2.10 not importable). Oracle NOT weakened. **ORD-05 end-to-end closure pending OFFLINE fixture generation** (run gen_fixtures.py on a catboost==1.2.10 machine, commit the 5 fixtures, then `cargo test -p cb-train --test tensor_ctr_e2e_oracle_test` must pass ≤1e-5 across all trees).

**Wave 10** *(gap closure — ORD-05 LIVE gap; blocks on 05-09's representation/apply; the categorical-CTR TRAINING pipeline `train()` never had — `cat_cardinalities` was hardcoded `&[]`, so cat columns reached the apply path but never training)*

- [ ] 05-11-PLAN.md — ORD-05 Part 1/2 — CAT INGESTION + CTR-FEATURE MATERIALIZATION: NEW cat-aware `train_cat` entry point computes OnLearnOnly per-feature cardinalities (calc_cat_feature_hash + PerfectHash) and, per tensor_ctr_candidate projection, materializes a per-document combined-projection ONLINE CTR feature (reusing online_ctr_prefix_binclf + the Borders quantizer calc_ctr_online_bin) the tree search can split on; `train()` byte-identical (14 numeric callers + all float oracles unaffected). (2 tasks)
- [ ] 05-12-PLAN.md — ORD-05 Part 2/2 — CTR-SPLIT SCORING + ctr_data BAKE + e2e HARD GATE: score materialized CTR columns into the oblivious search (shared L2 score, strict first-wins, forward-bit leaf index); bake each chosen split's whole-set CtrValueTable (build_final_ctr) into Model.ctr_data with the correct Scale/Shift (Borders prior 0.5 → Shift=0, Scale=15); thread split.shift/split.scale through apply.rs (removing the hardcoded 1.0/0.0); drive the FULL multi-tree `tensor_ctr_e2e_oracle_predictions_match_upstream` ≤1e-5 vs upstream catboost 1.2.10 through train_cat + predict_raw_cat, NO #[ignore] / NO weakened tolerance / NO fabricated fixtures. Closes ORD-05 / SC-5. (3 tasks)

**Research flag (RESOLVED)**: line-by-line read of `approx_calcer.cpp` + `online_ctr.*` complete (05-RESEARCH.md, file:line citations); per-object oracle schema designed (D-02). Research ESCALATION resolved: the D-01 TU-linking mechanism is infeasible; the user-approved **transcribe-then-self-oracle** replacement (05-CONTEXT DECISION REVISION 2026-06-14) is the mechanism used.

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
**Requirements**: PYAPI-01, PYAPI-02, PYAPI-03, PYAPI-04, PYAPI-05, PYAP
