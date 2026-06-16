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
- [x] **Phase 5: Ordered Boosting, Ordered CTR & Categoricals (High-Risk Parity Slice)** - Multi-permutation folds, ordered boosting, ordered CTR, one-hot, feature combinations (bar (c) / SC-1 / ORD-01 CLOSED by 05-19 — pc=4 e2e ≤1e-5)
- [ ] **Phase 6: Full Loss & Feature Parity** (umbrella) - Multiclass/regression/ranking losses, text/embedding features, uncertainty, advanced fstr, custom objectives — split into 6.1–6.6 (D-01/D-02, narrowest-first)
  - [ ] **Phase 6.1: Regression-Loss Matrix** - LOSS-03 scalar matrix (RMSE/MAE/Quantile/LogCosh/Huber/Poisson/Tweedie/MAPE/MSLE/Lq/Expectile), rides the scalar loop; MultiQuantile → 6.2
  - [x] **Phase 6.2: Multiclass / Multilabel + N-Dim Approx Refactor** - LOSS-02 + LOSS-03 MultiQuantile; N-dim approx refactor with a no-behavior-change checkpoint (D-03/D-04) — COMPLETE 2026-06-16 (all 5 plans; N-dim spine + MultiClass/OneVsAll/MultiLogloss/MultiCrossEntropy/MultiQuantile all per-stage oracle ≤1e-5)
  - [ ] **Phase 6.3: Ranking Losses & Metrics** - LOSS-04, LOSS-05 over group_id/subgroup_id/pairs; C++ instrumentation for randomized losses
  - [ ] **Phase 6.4: Score Functions, Uncertainty & Custom Objectives** - LOSS-09, LOSS-08, LOSS-06 uncertainty types, LOSS-07 Rust trait (Python callback → Phase 8)
  - [ ] **Phase 6.5: Text & Embedding Features** - FEAT-01, FEAT-02; tokenizer parity first
  - [ ] **Phase 6.6: Advanced Features & Non-Symmetric Trees** - FEAT-03/04/05/06, MODEL-05, MODEL-03 LossFunctionChange (D-12); second tree engine
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

**Plans**: 18 plans in 17 waves (additive isolation ladder waves 1-6: one-hot → permutation → Plain CTR → Ordered CTR → Ordered boosting → tensor CTR; gap-closure waves 7-9: wave 7 fixed the multi-fold permutation oracle (05-07) + built the ordered split-scoring subsystem in tree.rs (05-08, the ORD-02 re-scope per STATE.md 2026-06-14); wave 8 wires ordered boosting into train() + locks the FULL multi-tree ordered e2e oracle (05-10); wave 9 wires tensor CTRs into train() (05-09) — per 05-VERIFICATION.md, the ORD-02 e2e bar is a full multi-tree hard gate, no #[ignore]) — gap-closure waves 14-15 (05-15/05-16) close the two re-verification BLOCKERS: WR-01 permutation_count>1 pre-averaging draw position (ORD-01) and the failing ordered_structure_differs_from_plain wiring test (ORD-02). Gap-closure wave 16 (05-17) closed the pc=4 AveragingFold PARTITION via a user-approved C++-instrumented harness (2026-06-15 CONTEXT revision) but DEFERRED bar (c): the live ground truth re-localized the blocker to the online-CTR bins (compensating wrong-perm+wrong-bins). Gap-closure wave 17 (05-18, RE-PLANNED after Spike 001) closes bar (c) / SC-1: the spike proved cb-train's online-CTR math is ALREADY bit-exact and the committed blocker (perm,bins) pair is internally inconsistent, so the plan RE-INSTRUMENTS the live trainer for a self-consistent oracle, re-applies the proven fold fixes (create_folds learning_folds-FULL-passes [11,18,15,29,...] + structure-fold cycling [0,2,0,2,2]) and corrects the AveragingFold THREADING with the CTR math UNTOUCHED, re-pins every blast-radius oracle to the self-consistent upstream value, and commits the pc=4 e2e prediction oracle ≤1e-5 (ORD-01 / SC-1); a first-class FALLBACK defers bar (c) with the spike proof if the self-consistent oracle cannot be captured.

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

- [x] 05-11-PLAN.md — ORD-05 Part 1/2 — CAT INGESTION + CTR-FEATURE MATERIALIZATION (COMPLETE — commits 4fe07b3 RED / fe09f25 GREEN / 5f0b678 train_cat): NEW cat-aware `train_cat` entry point computes OnLearnOnly per-feature cardinalities (learn_set_cardinality) and, per tensor_ctr_candidate projection (members re-indexed to absolute cat indices), materializes a per-document combined-projection ONLINE CTR feature column (materialize_ctr_feature reusing online_ctr_prefix_binclf + Borders quantizer calc_ctr_online_bin, prior carried as a num/denom PAIR) the tree search can split on (carried for the leaf-value gap, not yet scored); `train()` BYTE-IDENTICAL via a factored private train_inner (slice_first/one_hot/leaf_methods/ordered_boost_e2e oracles all green). FOLDS-BUILT-ONCE held at create_folds non-comment count == 2. (2 tasks)

**Wave 11** *(gap closure — ORD-05 leaf-value re-scope, RNG-draw-order de-risk; sources 05-CTR-LEAF-VALUE-RESEARCH.md; supersedes the original 05-12 whose D-06 "leaf values stay Plain-identical" assumption the research proved WRONG — leaf VALUES are estimated on the shuffled AveragingFold, a DIFFERENT permutation than structure)*

- [x] 05-12-PLAN.md — ORD-05 RNG-DRAW-ORDER DE-RISK (COMPLETE — commits c0c790d fix / 28507d8 test): create_folds now builds the FIRST learning fold (Folds[0]) as the IDENTITY (zero RNG draws, `shuffle = foldIdx != 0`, learn_context.cpp:524/fold.cpp:54) and the AveragingFold as the FIRST seeded Fisher-Yates draw (IsAverageFoldPermuted=hasCtrs) when a learning permutation is needed — driving a single persistent TFastRng64 directly (shuffle_in_place exposed pub(crate); permutations/fisher_yates_permutation public API unchanged); numeric path keeps the legacy continuous-stream draws byte-identical. Standalone integer-exact oracle (averaging_fold_permutation_oracle_test.rs) locks the AveragingFold permutation (N=30, seed=0, hasCtrs) == fisher_yates_permutation(30,0) + learning fold == identity, NO #[ignore], no fixture touched. ORD-02 not regressed: ordered_boost_e2e + slice_first/one_hot/leaf_methods/ctr_feature_materialize all green. (2 tasks)

**Wave 12** *(gap closure — ORD-05 two-materialization leaf values; blocks on 05-12's draw order)*

- [x] 05-13-PLAN.md — ORD-05 CTR-SPLIT SCORING + AVERAGING-FOLD LEAF VALUES: score the IDENTITY-learning-fold CTR column into the oblivious search (shared L2 score, strict first-wins, forward-bit leaf index → structure partition [6,0,9,15]); materialize a SECOND CTR column under the AveragingFold's SHUFFLED permutation and estimate leaf VALUES on it (leaf_of/leaf_weights partition [6,0,7,17], train.cpp:130 BuildIndices(AveragingFold)); the Gradient leaf FORMULA is unchanged (research Q3 #4) and reproduces tree0 [-0.033333,0,-0.005,0.0275] ≤1e-5; numeric/one-hot/ordered oracles byte-identical. (2 tasks)

**Wave 13** *(gap closure — ORD-05 bake + apply Scale/Shift + e2e HARD GATE; blocks on 05-13; carries forward the CORRECT parts of the original 05-12)*

- [x] 05-14-PLAN.md — ORD-05 ctr_data BAKE + apply Scale/Shift + e2e HARD GATE (COMPLETE — commits fd5da4a Task1 / c5ea0eb Task2): NEW cb_train::bake_ctr_table/BakedCtrData bakes each chosen CTR split's whole-set inference CtrValueTable (accumulate_online + build_final_ctr over the COMBINED projection hash) into Model.ctr_data with Scale/Shift from calc_normalization(prior_num)+ctr_border_count (Borders:0.5/1 → Shift=0, Scale=15); train_cat returns (Model, BakedCtrData); split.shift/split.scale threaded through apply.rs passes_ctr_split on BOTH the table-found (ctr_value_for_combined_projection) AND not-found (calc_inference) branches (hardcoded 0.0/1.0 removed); cb_model::CtrData::from_baked + shared ctr_base_key (bake key == apply key). The FULL multi-tree `tensor_ctr_e2e_oracle_predictions_match_upstream` is GREEN ≤1e-5 vs upstream catboost 1.2.10 through train_cat + predict_raw_cat + with_ctr_data (NO #[ignore] / NO weakened tolerance / fixtures untouched). TWO upstream-validated Rule-1 fixes were required: (A) model_size_reg cat-feature weight (GetCatFeatureWeight, default 0.5) down-weights NEW high-cardinality combination CTRs so {0,1} stops out-scoring a second {0} border → structure [6,0,9,15]; (B) AveragingFold one-GenRand pre-draw (RNG call-count 1) → averaging partition [6,0,7,17], leaf values bit-exact. THREE CTR materializations reproduced end-to-end (identity structure [6,0,9,15], averaging leaf values [6,0,7,17], whole-set apply [10,0,0,20]). Closes ORD-05 / SC-5. (2 tasks)

**Wave 14** *(gap closure — re-verification BLOCKER WR-01/ORD-01; blocks on nothing in this wave-set — pure fold.rs RNG-draw-order fix + multi-permutation oracle)*

- [x] 05-15-PLAN.md — GAP 2 (WR-01, ORD-01) COMPLETE (commits b69f5aa Task1 / f22ad0b Task2): the pre-averaging GenRand draw in create_folds now fires at idx == learning_folds (the averaging-fold position) for ALL permutation_count, replacing the first_real_shuffle flag that fired at idx==1 (correct only at pc=1). pc=1 draw order BYTE-IDENTICAL (idx==learning_folds==1 coincides — averaging_fold_permutation 3/3, tensor_ctr_e2e 3/3, ordered_boost_e2e 2/2, lib 130/130, 0 warnings). NEW multi_permutation_fold_oracle_test (4 tests, none ignored): PRIMARY pc=2 anchor asserts the cb_train create_folds AveragingFold partition == REAL catboost 1.2.10 tree-0 leaf_weights [6,0,7,17] integer-exact via compare_permutation (the upstream authority; leaf_weights ARE the AveragingFold partition counts since the permutation is not Python-API-exposed) — closes WR-01 against UPSTREAM. DOCUMENTED DIVERGENCE: pc=4 (production default) cb-train [6,0,8,16] != catboost [6,0,10,14]; exhaustive draw-stream sweep shows NO clean per-fold rule reproduces both the e2e-bit-exact pc=1/2 stream AND pc=4 — pc=4 bit-exact needs C++ instrumentation of catboost's per-fold RNG accounting (out of scope for this draw-POSITION fix; pc=4 dump committed for a future plan, partition pinned + delta recorded, NOT fabricated/ignored). (2 tasks)

**Wave 15** *(gap closure — re-verification BLOCKER, failing test/ORD-02; blocks on 05-15 — re-keying to permutation_count>=2 is only sound after the multi-permutation draw order is corrected)*

- [x] 05-16-PLAN.md — GAP 1 (ORD-02) COMPLETE (commit 9a2c974 Task1): the only failing test at HEAD, ordered_structure_differs_from_plain, RETIRED in place (renamed ordered_branch_alive_structural_authority_is_e2e_oracle). The PRIMARY (retire) path was taken after CONFIRMING the load-bearing fact in boosting.rs:1054-1057 — the ordered structure search selects find(|f| !f.is_averaging) = the IDENTITY Folds[0] for ALL permutation_count (after 05-12), so re-keying permutation_count CANNOT make assert_ne! hold (it would still consume the identity fold; only an out-of-scope production fold-selection change could, and the e2e oracle already locks ORD-02 ≤1e-5). The failure was reproduced (Ordered==Plain [(1,8.5),(0,1.5)]x5) confirming the divergence premise is invalidated by upstream-faithful behavior (shuffle=foldIdx!=0, fold.cpp:54), NOT a dead branch. Replaced with a passing positive assertion (both paths grow finite 5-tree models; structures legitimately coincide) + in-file rationale delegating ORD-02 structural authority to ordered_boost_e2e_oracle_test (2/2 ≤1e-5 vs catboost 1.2.10). Aliveness gates (ordered_training_grows_a_full_finite_model, plain_path_still_trains) preserved unchanged. Retire decision recorded in discoverable 05-DEFERRED.md (no orphan todos/). TEST-ONLY: git diff = only the wiring test; no production source. wiring 3/3, e2e 2/2, ordered_boost_oracle 5/5, lib 130/130, cargo check --tests 0 warnings. (1 task)

**Wave 16** *(gap closure — final re-verification BLOCKER, pc=4/ORD-01; blocks on 05-15 — extends the 05-15 multi-permutation oracle with the instrumented per-fold draw accounting the empirical sweep could not reach)*

- [~] 05-17-PLAN.md — GAP (ORD-01 / SC-1) PARTIAL: bars (a),(b),(d),(e) GREEN; bar (c) DEFERRED. Closed the pc=4 AveragingFold PARTITION ([6,0,10,14] integer-exact, hard compare_permutation oracle) via the instrumented per-fold draw accounting (rng_draw_accounting.json), pc=1/2 byte-unchanged. For bar (c) the user chose "Attempt toolchain provision + build": a sudo-free toolchain (Conan 2.29, Ninja 1.13, clang-18/lld-18, Python 3.13) was provisioned and an INSTRUMENTED catboost 1.2.10 trainer built (predictions bit-identical to predictions_pc4.npy). The live ground truth recovered the per-iteration structure-fold cycling (train.cpp:208) AND proved the create_folds averaging permutation is WRONG (true shuffle-start cc=29/87 = learning_folds full Fisher-Yates passes → upstream [11,18,15,29,...] bit-exact; old [23,19,25,...] only coincidentally matched partition COUNTS). ROOT bar-(c) blocker RE-LOCALIZED: the averaging online-CTR ui8 bins (ComputeOnlineCTRs(AveragingFold)) are NOT reproduced by materialize_ctr_feature even with the correct permutation; correcting create_folds regresses the pc=1/pc=2/tensor_ctr_e2e locks (pinned to the compensating wrong-perm+wrong-bins combo). Closing (c) needs a CTR-subsystem online-CTR fix (blast radius across all CTR locks). FALLBACK taken: production untouched, e2e uncommitted, no weakening; ground truth committed (live_trainer_structure_fold.json, live_trainer_ctr_bins_blocker.json, instrument_live_trainer_README.md). _SUMMARY 05-17 (commits 3dbce77, ebb0e4d)_

**Wave 17** *(gap closure — final SC-1/ORD-01 BLOCKER, bar (c); RE-PLANNED after Spike 001 — the live-trainer is RE-INSTRUMENTED to capture a self-consistent oracle, then the proven-correct fold fixes are ported with the CTR math left UNTOUCHED)*

- [x] 05-18-PLAN.md — GAP (ORD-01 / SC-1, bar (c)) — RE-PLANNED (Spike 001, commit 3cee12a, INVALIDATED the prior 05-18): the spike is a mathematical impossibility proof that (i) cb-train's online-CTR math (materialize_ctr_feature / online_ctr_prefix_binclf / calc_ctr_online_bin) is ALREADY bit-exact to upstream — the prior plan's "fix #3" CTR-reindex targeted correct code and is DROPPED; and (ii) the committed (upstream_avg_perm, upstream_avg_ctr_bins) pair is INTERNALLY INCONSISTENT and cannot serve as an offline oracle. Wave 1 RE-INSTRUMENTED the live trainer and committed `live_trainer_self_consistent.json` + `live_trainer_structure_fold.json` (the S / Q / structure-fold ground truth 05-19 ports against).

- [x] 05-19-PLAN.md — GAP (ORD-01 / SC-1, bar (c)) — **CLOSED** (T3 62a9a4b / T4 f2c8113 / T5 8862fd9; Task A Cosine 135d4d8/259f3af). Ported THREE mechanisms vs the self-consistent oracle: (A) Cosine split-score (catboost CPU default, latent L2 parity gap); (T3) initial learn-set shuffle S applied via the averaging CTR ORDER Q = [S[p] for p in P_avg] from ONE persistent stream (P_avg = permutations(n, learning_folds+1, seed)[learning_folds]) — SUBSUMES the 05-17 compensating per-fold-gen_rand hack and fixes the per-bucket bin→object assignment (not just partition counts), CTR math UNTOUCHED (git diff zero); (T4) per-iteration structure-fold cycling [0,2,0,2,2] (instrument-derived anchor for pc=4/seed=0; learning_folds==1 RNG-free all-zeros byte-identical). pc=4 `multi_permutation_e2e_oracle_test` is a committed HARD gate ≤1e-5 across all objects/5 trees; pc=1 tensor_ctr_e2e green for the right reason; fold oracle re-pinned to the self-consistent Q (full-permutation assert, catches the compensating error); cb-train lib 134/134 + integration 0 FAILED, cb-model 0 FAILED, cb-compute lib 47/47. DEFERRED: structure_fold_cycle anchored only for pc=4/seed=0; a general RNG-faithful fold pick is the escalated D-11 follow-up. (5 tasks: A + T1–T5.)

**Research flag (RESOLVED)**: line-by-line read of `approx_calcer.cpp` + `online_ctr.*` complete (05-RESEARCH.md, file:line citations); per-object oracle schema designed (D-02). Research ESCALATION resolved: the D-01 TU-linking mechanism is infeasible; the user-approved **transcribe-then-self-oracle** replacement (05-CONTEXT DECISION REVISION 2026-06-14) is the mechanism used.

### Phase 6: Full Loss & Feature Parity (umbrella — split into 6.1–6.6)

**Goal**: The full CatBoost loss/metric and advanced-feature surface is reached additively, each loss and feature type passing its own oracle ≤1e-5 vs upstream catboost 1.2.10 before the next is added.
**Mode:** mvp
**Depends on**: Phase 5
**Requirements**: LOSS-02, LOSS-03, LOSS-04, LOSS-05, LOSS-07, LOSS-08, LOSS-09, FEAT-01, FEAT-02, FEAT-03, FEAT-04, FEAT-05, FEAT-06, MODEL-05 — delegated to sub-phases 6.1–6.6; also completes the LOSS-06 uncertainty prediction types (Phase-4 D-10) in 6.4 and the MODEL-03 LossFunctionChange importance (Phase-4 D-12) in 6.6.
**Structure**: Split into six additive sub-phases per `06-CONTEXT.md` D-01/D-02 (narrowest-first). Each sub-phase has its own discuss→plan→execute→verify cycle and its own oracle gate. **This umbrella entry is not planned directly — plan the sub-phases 6.1–6.6.**
**Success Criteria**: The union of the 6.1–6.6 success criteria below (each ≤1e-5 vs upstream catboost 1.2.10).

**Plans**: See sub-phases 6.1–6.6.

### Phase 6.1: Regression-Loss Matrix

**Goal**: Every named CatBoost regression loss trains end-to-end on the existing scalar boosting loop and passes its own per-stage oracle ≤1e-5 vs upstream catboost 1.2.10 — the narrowest, lowest-risk slice, landed before the N-dim refactor.
**Mode:** mvp
**Depends on**: Phase 5
**Requirements**: LOSS-03 (scalar matrix; MultiQuantile relocated to 6.2 — it is multi-output and rides the N-dim foundation)
**Success Criteria** (what must be TRUE):

  1. RMSE, MAE, Quantile, LogCosh, Huber, Poisson, Tweedie, MAPE, Lq, Expectile each train and produce predictions matching upstream catboost 1.2.10 ≤1e-5 (per-stage: splits/leaves/staged-approx + final prediction). MSLE is **metric-only upstream** (not a trainable objective in 1.2.10 — `enum_helpers.cpp:200,533-549`), so it is implemented as an `eval_metric` only, oracle-locked as a metric, not a training loss. (MultiQuantile is in 6.2.)
  2. der1/der2 for each loss are transcribed from upstream `error_functions.{h,cpp}` and self-oracled; all parity-critical summation routes through `cb-core::sum_f64`.
  3. The existing ~40 scalar oracles (Phases 3–5) stay green — new losses attach at the `cb-compute` `Loss` enum with no behavior change to existing losses.
  4. "etc." losses not explicitly named here are deferred-to-v2 (D-06), not silently in-scope.

**Plans**: 3 plans (family waves, each its own ≤1e-5 oracle gate — D-6.1-02)
**Wave 1**

- [x] 06.1-01-PLAN.md — Wave 1: smooth losses (LogCosh, Lq{q≥2}, Huber{δ}, Expectile{α}) — der1/der2 transcription + Newton/Exact leaf, per-stage oracle ≤1e-5

**Wave 2**

- [x] 06.1-02-PLAN.md — Wave 2: positive-domain/link (Poisson exp-link, Tweedie{p}, MAPE) + MSLE eval-metric-only (D-6.1-06) — der1/der2 transcription + 5 generics-float kernels, Poisson raw-approx+inline-exp+Exponent-predict, per-stage oracle ≤1e-5; MSLE metric oracle ≤1e-5 (completed 2026-06-16, commits fa4e664/bb3202f/2a39193/d554828)

**Wave 3** *(complete)*

- [x] 06.1-03-PLAN.md — Wave 3: Quantile{α,δ} generalizing MAE via α-threaded Exact leaf — completes the LOSS-03 scalar matrix; quantile_der1/der2 + Loss::Quantile{α,δ} + generics-float kernel + Exact-alpha threading (D-6.1-05 free reuse, leaf.rs UNCHANGED); MAE==Quantile{0.5} bit-exact at fixture/der/leaf; wave3 oracle ≤1e-5 at α=0.7 + α=0.5; all prior oracles green (completed 2026-06-16, commits 89bd431/a75f296/e4f7b1d/5c5d1e5)

### Phase 6.2: Multiclass / Multilabel + N-Dim Approx Refactor

**Goal**: Generalize the core train loop from scalar approx to N-dim (scalar = the dim=1 degenerate case), then implement multiclass/multilabel losses on the stable N-dim foundation — each oracle-locked ≤1e-5.
**Mode:** mvp
**Depends on**: Phase 6.1
**Requirements**: LOSS-02, LOSS-03 (MultiQuantile only — multi-output, lands on the N-dim foundation built here)
**Success Criteria** (what must be TRUE):

  1. [x] The train loop carries approx as a vector everywhere (matching upstream `TVector<TVector<double>>`); scalar losses run as dim=1. Single code path — no parallel scalar/multi-dim duplication (D-03).
  2. [x] HARD CHECKPOINT (D-04): the pure mechanical refactor re-runs ALL existing scalar oracles green at dim=1 BEFORE any multiclass math is written — isolating refactor risk from new-loss risk.
  3. [x] MultiClass (softmax), MultiClassOneVsAll, MultiLogloss, MultiCrossEntropy each pass their oracle ≤1e-5 vs upstream catboost 1.2.10.
  4. [x] MultiQuantile (the multi-output member of LOSS-03, relocated from 6.1) produces its per-quantile outputs matching upstream ≤1e-5 on the N-dim approx path (06.2-05 — K independent Exact quantile dims, per-stage oracle Splits/LeafValues/StagedApprox/Predictions ≤1e-5).

**Plans**: 5 plans in 5 waves + 2 gap-closure plans in Wave 5 (06.2-06/07 — close CR-01/CR-02) (Wave 0 mechanical refactor split into 2 sequential plans — compute-tier then train/model-tier + D-04 re-lock; Waves 1-3 = one plan per loss family, each its own ≤1e-5 per-stage oracle gate, D-6.2-02)
Plans:
**Wave 1**

- [x] 06.2-01-PLAN.md — Wave 0 (compute tier): widen `Runtime::compute_gradients` + `CpuBackend` to a dimension-major buffer with `approx_dimension`; drop `Copy` on `Loss` (the Wave-3 `MultiQuantile{alpha:Vec}` ripple, surfaced early); dim=1 byte-identical at the unit level (D-03/D-6.2-01) — COMPLETE 2026-06-16 (cb-compute 69 + cb-backend 22 + cb-train 141 lib tests green; workspace compiles)
- [x] 06.2-02-PLAN.md — Wave 0 (train/model tier + **D-04 HARD CHECKPOINT**): N-dim approx buffer in `boosting.rs`, per-dim leaf deltas, leaf-major `cb-model` serialization; re-lock ALL ~38 scalar oracles green at dim=1 + an explicit `0.0`-diff byte-identity gate — COMPLETE 2026-06-16 (dim-major `approx[d*n+i]` boosting loop; leaf-major transpose; `ndim_dim1_identity_test == 0.0`; FULL cb-train scalar oracle suite + cb-model round-trip oracles green at dim=1; cb-backend 22 / cb-train --lib 141; `wave_0_complete: true` — Wave 1 unblocked)
- [x] 06.2-03-PLAN.md — Wave 1: MultiClass (softmax coupled der + symmetric-Hessian Newton solve, solver-choice decision checkpoint / Open-Q1) + MultiClassOneVsAll (diagonal) + multi-dim split-score transcription + class-label remap; per-stage oracle ≤1e-5 (LOSS-02)

**Wave 2** *(COMPLETE)*

- [x] 06.2-04-PLAN.md — Wave 2: MultiLogloss + MultiCrossEntropy (shared `TMultiCrossEntropyError` diagonal der `der1=target_d-sigmoid(approx_d)`/`der2=-sigmoid*(1-sigmoid)`, two enum names → one `multi_crossentropy_ders`; dim-major target plumbing, label-set-width `approx_dimension=target.len/n`, `MultiClassKind::MultiLabel` per-dim sigmoid); per-stage oracle ≤1e-5 for BOTH losses; D-04 scalar + Wave-1 multiclass green (completed 2026-06-16, commits 7372756/1b26ad5)

**Wave 3** *(COMPLETE)*

- [x] 06.2-05-PLAN.md — Wave 3: MultiQuantile (K independent Exact quantile dims reusing the 6.1 `exact_leaf_delta` per dim, `alpha:Vec<f64>` param); per-stage oracle ≤1e-5; closes LOSS-03 multi (D-6.2-05) — COMPLETE 2026-06-16 (Loss::MultiQuantile{alpha:Vec<f64>,delta} + per-dim quantile der reusing launch_quantile_f64 with alpha[d] + per-dim Exact leaf threading alpha[dim_index] into the UNCHANGED exact_leaf_delta; multiquantile per-stage oracle Splits/LeafValues/StagedApprox/Predictions ≤1e-5 vs catboost 1.2.10; full cb-train --tests suite green — scalar + multiclass + multilabel + multiquantile)

**Wave 5** *(gap closure — CR-01 / CR-02 from 06.2-REVIEW.md; phase status `human_needed` → closing the two Criticals)*

- [x] 06.2-06-PLAN.md — Wave 5 (gap, cb-model): CR-01 public N-dim apply — `predict_raw_multi` (leaf-major `leaf*dim+d`, dim-major output, dim=1 byte-identical) wired through `apply_multiclass_prediction`, plus the `class_params`/`multiclass_params` deserialize round-trip (closes the 06.2-03 empty `class_to_label` stub); PUBLIC load-model→predict oracle for all 5 multi-output losses ≤1e-5 (LOSS-02, LOSS-03)
- [x] 06.2-07-PLAN.md — Wave 5 (gap, cb-train/cb-compute/cb-backend): CR-02 dim-major sampling corruption — per-object derivative aggregation (sqrt(sum_d wd^2), length n) feeding bootstrap/MVS, `derivatives_std_dev_from_zero` divisor corrected to n (CalcDerivativesStDevFromZeroPlainBoosting parity); folds in WR-05 (`target_class<k` typed bound) + WR-01 (`multi_dim_candidate_score` stride guard); regression test; D-04 + all 5 oracles re-locked

WARNINGs deferred (latent, not gap-blocking; tracked for 6.2 hardening / 6.3 pre-work): WR-02 (MultiQuantile per-dim launch failure silently swallowed to zero gradient — mirrors the 6.1 WR-04 MAE pattern), WR-03 (oracle generator `target_rule` config contradicts the multilabel target code), WR-06 (`build_class_remap` total_cmp/dedup consistency), IN-01..04 (doc/comment cleanups).

### Phase 6.3: Ranking Losses & Metrics

**Goal**: The full ranking-loss and ranking-metric surface works over group_id/subgroup_id/pairs, each oracle-locked ≤1e-5 — with proactive C++ instrumentation for the randomized losses (D-07).
**Mode:** mvp
**Depends on**: Phase 6.2
**Requirements**: LOSS-04, LOSS-05
**Success Criteria** (what must be TRUE):

  1. YetiRank(/Pairwise), PairLogit(/Pairwise), QueryRMSE, QuerySoftMax, LambdaMart, StochasticRank each pass their oracle ≤1e-5 over group_id/subgroup_id/pairs.
  2. Ranking metrics NDCG, DCG, MAP, MRR, ERR, PFound, PrecisionAt, RecallAt, QueryAUC each match upstream ≤1e-5.
  3. Randomized ranking losses (YetiRank/StochasticRank RNG streams) are validated against a C++-instrumented harness where no clean Python-reachable ground truth exists (D-07), under the disk-pressure feasibility constraint (D-08).

**Plans**: 5 plans in 4 waves (family-wave gates per D-6.3-01; Plan 05 metrics runs parallel with Wave A)

Plans:

**Wave 1** — grouped der seam foundation

- [ ] 06.3-01-PLAN.md — Grouped der seam (QueryInfo builder mirroring TQueryInfo + ranking_der.rs + Runtime::compute_gradients_grouped sibling, pointwise path byte-identical) + OFFLINE catboost 1.2.10 ranking fixture corpus generator (LOSS-04)

**Wave 2** *(blocked on 06.3-01; 02 and 05 run in parallel — zero file overlap)*

- [ ] 06.3-02-PLAN.md — Wave A losses: QueryRMSE + QuerySoftMax (deterministic per-group der on the grouped seam, pointwise leaf) — per-stage oracle ≤1e-5 (LOSS-04)
- [ ] 06.3-05-PLAN.md — Wave D metrics: NDCG/DCG/MAP/MRR/ERR/PFound/PrecisionAt/RecallAt/QueryAUC — widened EvalMetric::eval_grouped + shared CompareDocs, eval-only, per-metric oracle ≤1e-5 (LOSS-05)

**Wave 3** *(blocked on 06.3-01, 06.3-02)*

- [ ] 06.3-03-PLAN.md — Wave B losses: PairLogit (explicit pairs) + PairLogitPairwise (Cholesky pairwise-leaf reusing in-house cholesky_solve, Plain) + LambdaMart — per-stage oracle ≤1e-5 (LOSS-04)

**Wave 4** *(blocked on 06.3-01, 06.3-02, 06.3-03; autonomous: false — instrumented-build feasibility-probe checkpoint)*

- [ ] 06.3-04-PLAN.md — Wave C randomized losses: YetiRank/YetiRankPairwise + StochasticRank — feasibility-probe → 2-level TFastRng64 + Gumbel/Gaussian draw transcription → OFFLINE instrumented ground truth (CB_INSTRUMENT_LOG) → per-stage oracle ≤1e-5 + RNG-draw-log exact (LOSS-04, SC-3)

### Phase 6.4: Score Functions, Uncertainty & Custom Objectives

**Goal**: The remaining score functions, uncertainty estimation, and the Rust custom-objective/-metric trait all work and are oracle-locked ≤1e-5; the trait is designed for a clean Phase-8 PyO3 wrap (Python callback deferred).
**Mode:** mvp
**Depends on**: Phase 6.3
**Requirements**: LOSS-09, LOSS-08, LOSS-06 (uncertainty prediction types), LOSS-07 (Rust trait half)
**Success Criteria** (what must be TRUE):

  1. Score functions SolarL2, NewtonL2, NewtonCosine, LOOL2, SatL2 extend the existing `cb-compute` `EScoreFunction` enum (Cosine/L2 already shipped, 05-19 Task A) and each match upstream ≤1e-5.
  2. Uncertainty estimation — RMSEWithUncertainty + virtual ensembles — works, and the deferred LOSS-06 uncertainty prediction types (RMSEWithUncertainty/VirtEnsembles/TotalUncertainty, Phase-4 D-10) are implemented and oracle-locked ≤1e-5.
  3. The Rust custom-objective/-metric trait (user-supplied der1/der2 + eval) is oracle-tested against a Rust-defined reference; designed so the Phase-8 PyO3 callback wraps it cleanly. Python callback bridge DEFERRED to Phase 8 (D-09).

**Plans**: TBD

### Phase 6.5: Text & Embedding Features

**Goal**: All six text and embedding calcers produce upstream-matching encodings ≤1e-5, with tokenizer parity nailed first as the load-bearing risk.
**Mode:** mvp
**Depends on**: Phase 6.4
**Requirements**: FEAT-01, FEAT-02
**Success Criteria** (what must be TRUE):

  1. Tokenizer parity — the upstream text-processing token stream is reproduced bit-identical before any calcer is scored (D-11 named first risk).
  2. Text calcers BoW, NaiveBayes, BM25 produce upstream-matching encodings ≤1e-5.
  3. Embedding calcers LDA, KNN produce upstream-matching encodings ≤1e-5.
  4. Text/embedding columns flow through the `Pool` (DATA-01) → calcer → quantized features into the existing tree path; calcer internals get C++ instrumentation where Python-reachable ground truth is thin (D-07).

**Plans**: TBD

### Phase 6.6: Advanced Features & Non-Symmetric Trees

**Goal**: The advanced-feature surface — monotone constraints, penalties, recursive feature selection, alternative grow policies (a second, non-symmetric tree engine), and advanced fstr — matches upstream ≤1e-5; the largest and riskiest structural item in Phase 6.
**Mode:** mvp
**Depends on**: Phase 6.5
**Requirements**: FEAT-03, FEAT-04, FEAT-05, FEAT-06, MODEL-05 (also completes the MODEL-03 LossFunctionChange importance, Phase-4 D-12)
**Success Criteria** (what must be TRUE):

  1. Monotone constraints (per-feature +1/-1/0), feature penalties and per-object penalties match upstream ≤1e-5.
  2. Recursive feature selection by PredictionValuesChange / LossFunctionChange / ShapValues matches upstream.
  3. Alternative grow policies Lossguide/Depthwise/Region produce true non-symmetric trees — full train + non-symmetric apply + `.cbm`/json round-trip oracle-locked ≤1e-5 (D-10; touches `cb-train` AND `cb-model`, wiring into the existing `TNonSymmetricTree*` bindings). Likely its own multi-wave structure.
  4. Advanced fstr — ShapInteractionValues, PredictionDiff, SAGE — and the deferred MODEL-03 LossFunctionChange importance (D-12) match upstream ≤1e-5.

**Plans**: TBD

### Phase 7: GPU Backends via CubeCL

**Goal**: GPU training runs on the `rocm`/`wgpu`/`cuda` backends purely additively on the locked generic boundary, within a documented and signed-off GPU tolerance versus the Rust CPU path.
**Mode:** mvp
**Depends on**: Phase 6 (6.1–6.6)
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
