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
  - [x] **Phase 6.3: Ranking Losses & Metrics** - LOSS-04, LOSS-05 over group_id/subgroup_id/pairs; C++ instrumentation for randomized losses (completed 2026-06-17)
  - [x] **Phase 6.4: Score Functions, Uncertainty & Custom Objectives** - LOSS-09, LOSS-08, LOSS-06 uncertainty types, LOSS-07 Rust trait (Python callback → Phase 8) (completed 2026-06-18)
  - [x] **Phase 6.5: Text & Embedding Features** - FEAT-01, FEAT-02; tokenizer parity first (SC-2 BM25 per-stage CLOSED via 06.5-09 PATH-A fixture-correctness fix)
  - [x] **Phase 6.6: Advanced Features & Non-Symmetric Trees** - FEAT-03/04/05/06, MODEL-05, MODEL-03 LossFunctionChange (D-12); second tree engine (completed 2026-06-18)
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

**Plans**: 18 plans in 11 waves (5 original + 4 gap-closure 06–09 + 5 gap-closure 10–14 + 4 gap-closure-round-3 15–18 from the 8/9 re-verification; round-3 lands the pairwise split-scorer subsystem that unblocks both *Pairwise losses + closes StochasticRank end-to-end + D-07 trainer-level)

Plans:

**Wave 1** — grouped der seam foundation

- [x] 06.3-01-PLAN.md — Grouped der seam (QueryInfo builder mirroring TQueryInfo + ranking_der.rs + Runtime::compute_gradients_grouped sibling, pointwise path byte-identical) + OFFLINE catboost 1.2.10 ranking fixture corpus generator (LOSS-04) — **COMPLETE** (c1a38a9/1939ab8/441f36d): build_query_info (10 tests), calc_ders_for_queries + group_reduce_weighted + compute_gradients_grouped (9 tests, pointwise byte-identical D-04), gen_ranking_fixtures.py end-to-end-validated against .venv catboost 1.2.10; corpus inputs + QueryRMSE smoke fixture frozen. Loss arms return typed "not yet wired" until 02–05.

**Wave 2** *(blocked on 06.3-01; 02 and 05 run in parallel — zero file overlap)*

- [x] 06.3-02-PLAN.md — Wave A losses: QueryRMSE + QuerySoftMax (deterministic per-group der on the grouped seam, pointwise leaf) — per-stage oracle ≤1e-5 (LOSS-04) — **COMPLETE** (f42e3e6/6b208dd): queryrmse_der/querysoftmax_der + Loss::QueryRmse/QuerySoftMax{lambda,beta} (validate, defaults 0.01/1.0); ranking_der.rs arms (queryAvrg + max-shifted softmax der, sum_f64, empty-group/sumWTargets≤0 guards); boosting der-site branches on is_grouped_loss → compute_gradients_grouped over a per-fit QueryInfo view (train_ranking + RankingData; non-ranking site byte-identical D-04); QueryRMSE=Newton / QuerySoftMax=Gradient leaf, Cosine score (per fixture); per-stage oracles gate Splits/LeafValues/StagedApprox/Predictions ≤1e-5 vs catboost 1.2.10; QuerySoftMax fixture frozen OFFLINE. cb-compute 102/102, full cb-train suite 0 failures (D-04 no-regression). 11 new unit tests.
- [x] 06.3-05-PLAN.md — Wave D metrics: NDCG/DCG/MAP/MRR/ERR/PFound/PrecisionAt/RecallAt/QueryAUC — widened EvalMetric::eval_grouped + shared CompareDocs, eval-only, per-metric oracle ≤1e-5 (LOSS-05) — **COMPLETE** (086550d/274fbb9): all 9 per-group formulas transcribed verbatim from libs/metrics (DCG/NDCG dcg.cpp, PFound pfound.h, ERR/MRR metric.cpp, Prec/Rec/MAP precision_recall_at_k.cpp, QueryAUC auc.cpp) + the shared compare_docs tie-break (predicted desc→target asc→stable index) transcribed ONCE; EvalMetric::eval_grouped sibling seam (flat eval byte-identical D-04, ranking/non-ranking misuse → typed error); upstream defaults pinned (top=-1, decay=0.85, border=0.5, denominator=LogPosition, dcg_type=Base); group-weighted vs uniform averaging via sum_f64; IDCG≤0→1 / total_relevant==0→1 / empty→0 guards. Oracle ground truth = catboost.utils.eval_metric over a FIXED approx (eval-only, no trained model); 18 cases ≤1e-5 (default + top=2 + QueryAUC Ranking/Classic). Rule-1 fix: ranking_auc_group → direct concordance count (closed 0.48-vs-0.5). LOSS-05 / SC-2 CLOSED. cb-train lib 173/173, D-04 no-regression green.

**Wave 3** *(blocked on 06.3-01, 06.3-02)*

- [~] 06.3-03-PLAN.md — Wave B losses: PairLogit (explicit pairs) + PairLogitPairwise (Cholesky pairwise-leaf reusing in-house cholesky_solve, Plain) + LambdaMart — per-stage oracle ≤1e-5 (LOSS-04) — **LambdaMart COMPLETE; PairLogit/PairLogitPairwise machinery lands, oracle DEFERRED** (9b2606d/b25a2be/bef767d): LambdaMart NDCG per-group lambda grad + pointwise Newton leaf — per-stage oracle GREEN ≤1e-5 vs catboost 1.2.10 (unlocked by the Rule-1 newton_leaf_delta fix: divide verbatim for NEGATIVE denominators — listwise positive hessian; regression losses unaffected). PairLogit/PairLogitPairwise der over Competitors (inline exp, error_functions.h:849-866) + the Cholesky pairwise-leaf path (pairwise_leaves.rs, 2×2 + general SPD via REUSED cb_compute::pairwise_cholesky_solve + diag/nonDiag reg + MakeZeroAverage — BIT-EXACT vs pairwise_leaves_calculation_ut.cpp, no new crate) + is_pairwise_scoring leaf-routing all LAND and are unit-tested; the two PairLogit/PairLogitPairwise per-stage ORACLES are DEFERRED on a precisely-isolated pair-weight normalization gap (catboost group-relative pair weighting not yet in build_query_info — deferred-items.md; fixtures frozen; NO #[ignore]/NO weakened tolerance). Gates: cb-compute 113/113, pairwise_leaves 6/6, lambdamart_oracle 1/1, full cb-train green, cargo check --workspace --tests GREEN.

**Wave 4** *(blocked on 06.3-01, 06.3-02, 06.3-03; autonomous: false — instrumented-build feasibility-probe checkpoint)*

- [~] 06.3-04-PLAN.md — Wave C randomized losses: YetiRank/YetiRankPairwise + StochasticRank — **RNG STREAM VALIDATED ≤1e-5; end-to-end trainer fixture DEFERRED (path c)** (5a77507/6c59fde/3890089): Loss::YetiRank/YetiRankPairwise{permutations,decay} + Loss::StochasticRank{metric,sigma,mu,num_estimations} + validate; yetirank.rs sampler (2-level TFastRng64 seed + Gumbel noise + Classic decayed weights, yetirank_helpers.cpp:146-393) rides PairLogit der over SAMPLED pairs; StochasticRank DCG/NDCG Monte-Carlo der (std_normal noise + SFA, error_functions.cpp:1008-1256, der2=0) in ranking_der.rs; boosting per-iteration competitor re-sample; YetiRankPairwise → Cholesky leaf (Plain). TWO standalone OFFLINE instrumented generators (yetirank_oracle.cpp + stochasticrank_oracle.cpp, ZERO catboost includes) compile clean + RUN + SELF-ORACLE bit-for-bit vs cb-core::TFastRng64/std_normal (block_seed 12283622132691337806, std_normal(0)=[0.6337,-0.5284,-0.4408]); RNG-draw ground truth frozen; 3 per-stage oracles gate the draw log (sampled competitor weights 0.192/0.098250/0.083250 + 2-level query seed + Gaussian noise stream) ≤1e-5, NO #[ignore]/NO weakened tolerance. Feasibility-probe Task 1 → path (c) ESCALATE: instrumented catboost TRAINER build infeasible (toolchain absent + disk 95-97%/~8-12G = link-failure regime, D-6.3-03b); end-to-end per-stage trainer fixture DEFERRED, oracle tests wired to run the full gate the moment it lands (deferred-items.md / instrument_ranking_rng_README.md). decay AMBIGUITY resolved = 0.85. cargo check --workspace --tests GREEN; D-04 + Wave-A/B + cb-compute/cb-backend suites green.

**Wave 5 — gap closure** *(from 06.3-VERIFICATION.md, scoped: code fixes now, instrumented trainer build deferred; plans 06–09 are independent — zero file overlap, all parallel)*

- [x] 06.3-06-PLAN.md — CR-01 (StochasticRank NDCG calc_dcg_metric_diff reads normalized pos_weights) + WR-02 (lambdamart_ideal_ndcg via cb_core::sum_f64) — both in ranking_der.rs (LOSS-04) — **COMPLETE** (6cbe2eb/c7d4baa): CR-01 threads `pos_weights: &[f64]` into `calc_dcg_metric_diff`, reading `old_weight`/`new_weight` via `pos_weights.get(old_pos/new_pos).copied().unwrap_or(0.0)` (the SAME normalized vector that built cum_sum/cum_sum_up/cum_sum_low; upstream posWeights[oldPos]/posWeights[newPos], error_functions.cpp:1233-1234), removing the raw `1.0/ndcg_denominator(pos)` recompute + obsolete OFFLINE-closure comment → doc_diff and mid_diff now share the 1/ideal_dcg scale for NDCG groups where ideal_dcg != 1.0. WR-02 replaces the raw `score +=` loop in `lambdamart_ideal_ndcg` with collected terms reduced via `cb_core::sum_f64` (bit-identical, D-08-compliant). RED-first graded-relevance ideal_dcg != 1.0 regression test + DCG-arm-unchanged guard + lambdamart sum_f64-of-terms + trivial-window tests. cb-compute 122/122, lambdamart_oracle 1/1 (≤1e-5, no regression), stochasticrank_oracle 2/2, full cb-train suite 0 failures. Pre-existing clippy::indexing_slicing in stochastic_rank_group_der logged out-of-scope (deferred-items.md). LOSS-04 stays partially open (truths #5/#7 still deferred).
- [x] 06.3-07-PLAN.md — CR-02 (boosting.rs weighted_der1 branches on group_spans.is_some() so grouped ranking ders are not double-weighted) + non-uniform-weight regression test (LOSS-04) — **COMPLETE** (ff2fd8e/5cb1e74): `weighted_der1` now branches on `group_spans.is_some()` — grouped ranking ders (QueryRMSE `(target-approx-queryAvrg)*weight`, QuerySoftMax `beta*(-sumWTargets*p + weight*target)`) are ALREADY weight-folded so the grouped path uses `ders.der1.clone()` as-is; the pointwise path keeps `der1 * weight` (idx % n, byte-identical at dim=1, D-04). The old unconditional re-multiply double-weighted (squared weights → corrupt split scores/leaf values), invisible at the w=1.0 oracle fixtures. New `grouped_weight_regression_test.rs` (non-uniform weights [2.0,0.5,1.5,1.0,0.25] over 2 groups) gates the invariant against the public `calc_ders_for_queries` seam: grouped der == single-weighted reference (≤1e-12) AND != squared-weight value for non-unit objects → would fail against pre-CR-02 code; covers QueryRmse + QuerySoftMax. Gates: queryrmse/querysoftmax oracles ≤1e-5 (no regression), lambdamart oracle green (pair-weighted, unaffected), full cb-train --tests 0 failures. LOSS-04 stays partially open (truths #5/#7 deferred).
- [x] 06.3-08-PLAN.md — WR-01 (make_zero_average via cb_core::sum_f64) + WR-03 (stochasticrank_oracle.cpp centering aligned to sum_f64 order) + >4-doc non-zero-mu test; autonomous: false (frozen RNG groundtruth regeneration human-gated) (LOSS-04) — **COMPLETE** (c9a0f8e/6123c60): WR-01 — `make_zero_average` mean now routes through `cb_core::sum_f64(res)/n` (loop-order exception doc-comment dropped, D-08 citation added; bit-identical fold; pairwise_leaves 7/7 incl. a new sum_f64 zero-average invariant). WR-03 — `stochasticrank_oracle.cpp` Stage-1 centering replaces `std::accumulate` with an explicit left-to-right fold matching `sum_f64` (cite reduction.rs source-of-truth + ranking_der.rs:726-727); new `stochasticrank_centering_test.rs` (7-doc, NON-ZERO mu) asserts sum_f64-centered vs generator-order centering ≤1e-5 + an order-sensitivity guard proving non-vacuity (discovered via a tests/ #[path] shim, INFRA-06). Task 3 human-action checkpoint: the dependency-free generator was compiled (g++ -std=c++20 -Wall, zero catboost includes) + run; regenerated `stochasticrank_rng_groundtruth.jsonl` is BYTE-IDENTICAL to the committed fixture (empty diff — the 3-doc mu=0 corpus has an exactly-zero shifted sum, so the centering-order change is a provable no-op) → NO regeneration required, fixture untouched (awaiting human sign-off). Gates: cb-train pairwise_leaves 7/7, cb-oracle stochasticrank_centering 2/2, cb-train stochasticrank_oracle_test 2/2 (no regression). LOSS-04 stays partially open (truths #5/#7 still deferred). gsd-tools ABSENT → STATE/ROADMAP updated MANUALLY.
- [~] 06.3-09-PLAN.md — PairLogit pair-weight wiring + PairLogit/PairLogitPairwise per-stage oracle tests (LOSS-04) — **bt.PairwiseWeights WIRING LANDED; end-to-end oracle DEFERRED on leaf-der2 parity** (de78521/ae6eb67): `boosting.rs` gains `uses_pairwise_weights` (UsesPairsForCalculation = PairLogit | PairLogitPairwise | YetiRank{,Pairwise}, enum_helpers.cpp:502) + `calc_pairwise_weights` (mirrors CalcPairwiseWeights, approx_updater_helpers.h:74-89: scatter competitor.weight to BOTH winner+loser slots → per-object bt.PairwiseWeights); the per-iteration `eff_weights` feeds split-scoring `score_weights` (scoring.cpp:276-279) + `scaled_l2 = scale_l2_reg(l2, Σ eff_weights, n)` (CalcDeltaNewtonBody), the grouped Newton leaf der2 keeping UNIT weights (der already folds the pair weight — der2 analogue of CR-02). RED-first boosting_test.rs (predicate + pairwise-sum + weighted/empty-group). This advanced the PairLogit Splits oracle match index 4→6. **REFUTED the plan's Competitor.weight diagnosis (Rule-1):** fixture pairs.npy is (7,2) winner/loser only → all weights 1.0; upstream Competitor.Weight = pair.Weight verbatim (data_providers.cpp:327-329); build_query_info UNCHANGED. The two oracle tests (pairlogit_oracle_test.rs Newton/l2=3, pairlogit_pairwise_oracle_test.rs Gradient-Cholesky/l2=5) are committed + #[ignore]'d (NO tolerance weakened, D-6.3-03b). DEFERRED ROOT CAUSE = the PairLogit LEAF-der2 reduction (06.3-03 blocker): der bit-verified-identical to upstream yet per-leaf Newton denominators are mutually inconsistent with -sumDer2+C (single cross-leaf-pair leaf needs ~23 from sumDer=0.5/sumDer2=-0.25), needs the instrumented catboost trainer per-leaf SumDer2 log (toolchain/disk infeasible). deferred-items.md [06.3-03] updated with the reproduction table. Gates: cb-train --lib 189/0, queryrmse/querysoftmax/lambdamart 3/3 ≤1e-5 (no regression), pairwise_leaves 7/7, grouped_weight_regression 2/2, full cb-train 0 failures. LOSS-04 stays PARTIALLY OPEN. gsd-tools ABSENT → STATE/ROADMAP MANUAL.

**Wave 6 — gap closure round 2** *(from the 7/9 re-verification 06.3-VERIFICATION.md + 06.3-REVIEW.md; operator chose ATTEMPT THE FULL INSTRUMENTED TRAINER BUILD. DISK IS NOW 67 GB FREE / 71% — no longer the documented link-failure regime that forced the prior deferrals. Plans 11+12 are trainer-INDEPENDENT and run parallel to the build so the round is productive even on a build NO-GO; 13/14 are trainer-DEPENDENT and ESCALATE-DON'T-WEAKEN on NO-GO.)*

- [~] 06.3-10-PLAN.md — Wave 1: instrumented catboost 1.2.10 trainer build (disk-prep + clang-18/lld-18 restore + CB_INSTRUMENT_LOG per-leaf-der2 + RNG-draw patch + build_native.py driver) + build-feasibility GO/NO-GO checkpoint; autonomous: false (LOSS-04) — **Task1 COMPLETE + GO recorded; AWAITING blocking-human GO sign-off** (dae8bff/472922c): the instrumented `_catboost.so` (39.7MB) **BUILT, LINKED, runs, emits CB_INSTRUMENT_LOG**. Sudo-free re-runnable `build_instrumented_trainer.sh` (287 lines): disk-gate(≥25GB; now 67GB free)→clang-18/lld-18 via apt-get download+dpkg -x into /tmp/clang18_prefix→idempotent env-gated patch of 4 TUs (per-leaf der1/der2 approx_calcer_querywise.cpp + leaf-weight approx_calcer.cpp + YetiRank Gumbel yetirank_helpers.cpp + StochasticRank noise error_functions.cpp; no-op when unset)→build_native.py --targets _catboost vs .venv Py3.13. RC=0 after 3 auto-fixes (Rule-3 toolchain bare clang/clang++→prefix symlinks; Rule-1 awk -v '\n'→fputc(10); Rule-1 perl \" JSON→R"J(...)J" raw literals). Smoke CatBoostRanker(YetiRank).fit() wrote 58KB JSONL: 264 leaf_der + 1080 yeti_gumbel events. instrumented_trainer_STATUS.md = GO. Vendored catboost-master/ patches stay UNCOMMITTED (OFFLINE/RUN-ONCE, D-09/D-12). **UNBLOCKS 13 (PairLogit leaf-der2) + 14 (YetiRank/StochasticRank fixtures)**. NO oracle weakened, NO fixture fabricated. gsd-tools ABSENT → STATE/ROADMAP MANUAL.
- [x] 06.3-11-PLAN.md — Wave 1 (parallel, trainer-independent): ranking_der.rs hardening — REVIEW WR-02 (bounds-check cum_sum/cum_sum_up/cum_sum_low reads in calc_dcg_metric_diff) + the 42 clippy::indexing_slicing sites in stochastic_rank_group_der (deferred-items.md [06.3-06]), oracle-revalidated ≤1e-5 (LOSS-04) — **COMPLETE** (21ddf5a/0fdd507): WR-02 — the 6 raw cum_sum/cum_sum_up/cum_sum_low subscripts in BOTH mid_diff arms of `calc_dcg_metric_diff` now read via `.get(..).copied().unwrap_or(0.0)`, matching the CR-01 `pos_weights.get(..)` discipline (arrays length count+1 → bit-identical; removes the latent panic on a future top/query_top_size index desync). Task 2 — ALL 42 `clippy::indexing_slicing` sites in `stochastic_rank_group_der` converted to bounds-checked reads / `get_mut` guarded writes / iterator-zip / a `score_at(p)` closure for nested `scores[order[p]]`; NO unwrap()/expect() (CLAUDE.md ban); `count<=1` early return → `count>=2` so every `.get()` index is in-range → BIT-IDENTICAL. RED-first WR-02 tests (boundary new_pos+1==count + graded-relevance NDCG both-arms over all pos pairs). Oracle-revalidated NOT blind-fixed: `cargo clippy -p cb-compute --lib` 42→0 'indexing may panic'; lambdamart_oracle 1/1 + stochasticrank_oracle 2/2 ≤1e-5; cb-compute lib 124/124, cb-train lib 189/189 — zero parity regression. LOSS-04's two trainer-INDEPENDENT findings CLOSED independently of the 06.3-10 build outcome; LOSS-04 truths #5/#7 (end-to-end trainer fixtures + SC-3 harness) STILL DEFERRED. gsd-tools ABSENT → STATE/ROADMAP MANUAL.
- [x] 06.3-12-PLAN.md — Wave 1 (parallel, trainer-independent): REVIEW CR-01 BLOCKER (stochasticrank_oracle.cpp noise/score → %.17g) + WR-05 (non-masking mu!=0 / >4-doc unit) + regenerate full-precision stochasticrank_rng_groundtruth.jsonl + WR-04 (grouped_weight_regression guard scale-match); autonomous: false (frozen RNG groundtruth human-gated) (LOSS-04)
- [x] 06.3-13-PLAN.md — Wave 2 (depends 10): GAP 1 / truth #4 — capture instrumented per-leaf SumDer/SumDer2 log for the frozen PairLogit fixture, transcribe the der2 reduction into the pairwise-leaf path, REMOVE #[ignore] from pairlogit_oracle_test.rs + pairlogit_pairwise_oracle_test.rs (full ≤1e-5 per-stage gate); autonomous: false; escalate-don't-weaken on 06.3-10 NO-GO (LOSS-04)
- [x] 06.3-14-PLAN.md — Wave 3 (depends 10,12,13): GAP 2 / truth #5 — train + freeze YetiRank/YetiRankPairwise/StochasticRank model.json fixtures, remove absent-fixture invariants, wire full compare_stage ≤1e-5 gates + REVIEW WR-03 (YetiRank Gradient leaf eff_weights, branched on leaf method not group_spans) + GAP 3 / truth #7 trainer-half (CB_INSTRUMENT_LOG RNG draw log vs Rust sampler, D-07); autonomous: false; escalate-don't-weaken on NO-GO (LOSS-04)

> Still deferred after this gap-closure round (out of scope — instrumented catboost trainer build): YetiRank/YetiRankPairwise/StochasticRank end-to-end TRAINER fixtures + ≤1e-5 trainer gates (truth #5); SC-3 / D-07 instrumented-harness RNG validation (truth #7). LOSS-04 stays partially open.

**Wave 7 — gap closure round 3** *(from the 8/9 re-verification 06.3-VERIFICATION.md; THREE remaining gaps, all escalated un-weakened. The PairLogitPairwise + YetiRankPairwise gaps share ONE new subsystem — the pairwise SPLIT-scorer (TPairwiseScoreCalcer / CalculatePairwiseScore) — isolated precisely as a tree-0 split-selection divergence, NOT a leaf-der gap; StochasticRank's distinct per-group noise-seed model (randomSeed+group_index) closes in parallel. Plans 15 (library) → 16 (wire + PairLogitPairwise) → 17 (YetiRankPairwise) are sequential; plan 18 (StochasticRank) is independent/parallel.)*

- [x] 06.3-15-PLAN.md — Wave 1 (parallel with 18): pairwise SPLIT-scorer subsystem in cb-compute (compute_der_sums + compute_pair_weight_statistics + calculate_pairwise_score, OneFeature float path, reusing the in-house Cholesky leaf solver, all reductions via cb_core::sum_f64) — pure library, self-oracled bit-for-bit vs hand-derived references (LOSS-04) — **COMPLETE** (03ae077/653e083): `crates/cb-compute/src/pairwise_scoring.rs` lands `compute_der_sums` (ComputeDerSums, pairwise_scoring.h:52-68), `compute_pair_weight_statistics` (ComputePairWeightStatistics, h:72-103, OneFeature), `calculate_pairwise_score` (CalculatePairwiseScore + CalculateScore + UpdateWeightSumFromTotal/NonDiagStats, cpp:51-232) + `BucketPairWeightStatistics`, exported from lib.rs. Per-split leaf solve reuses `crate::pairwise_cholesky_solve` via a cb-compute-LOCAL `calculate_pairwise_leaf_values` twin (2×2 closed form + general (n-1)×(n-1) Cholesky + diag/nonDiag reg + MakeZeroAverage; cb-compute can't depend on cb-train — layering) — NO new crate. Rule-2 deviation: public fns return `CbResult` (artifact table showed bare returns) to honor T-06.3-15-01 — malformed leaf/bucket/doc index → `CbError::OutOfRange`, never panic; Rule-1: add→merge rename (should_implement_trait). All reductions via `cb_core::sum_f64` (D-08); only documented upstream-order scatter cells use raw +=/-=; degenerate Cholesky → zeros → finite 0.0 not NaN (T-06.3-15-02). Self-oracled: der-sums + pair-weight stats hand-derived bit-for-bit (both bucket branches + winner==loser skip); calculate_pairwise_score vs an INDEPENDENT Gaussian-elimination reference solver ≤1e-9; single-leaf 2×2 closed-form re-derivation. Library-ONLY: NO tree wiring (06.3-16), NO fixture, NO tolerance touched. Gates: cb-compute lib 131/131 (124 baseline + 7 new), 0 indexing_slicing/unwrap_used in the new file, no new crate in Cargo.toml, `cargo check --workspace --tests` GREEN. Symbol names match the artifact contract so 06.3-16/17 key_links resolve. gsd-tools ABSENT → STATE/ROADMAP/REQUIREMENTS updated MANUALLY.
- [x] 06.3-16-PLAN.md — Wave 2 (depends 15): GAP 1 / truth #4 — wire the pairwise scorer into tree.rs greedy search gated on is_pairwise_scoring (non-pairwise path byte-identical D-04), freeze the PairLogitPairwise fixture OFFLINE (catboost 1.2.10), REMOVE #[ignore] from pairlogit_pairwise_oracle_test.rs (full 4-stage ≤1e-5 gate); autonomous: false (LOSS-04) — **COMPLETE** (6aaa769 Task1 / 09bd53e Task3): `calculate_pairwise_score` WIRED into `crates/cb-train/src/tree.rs` greedy oblivious search (tree.rs:1463-1556) gated on `is_pairwise_scoring`; non-pairwise path byte-identical (D-04). Task 2 fixture NOT regenerated — the existing `ranking_corpus/PairLogitPairwise/{model.json,staged.npy,predictions.npy}` is GENUINE catboost 1.2.10 output (tags/v1.2.10, model_guid 7a8f259-…, train_finish_time 2026-06-16T23:14:35Z, loss PairLogitPairwise), committed bef767d (06.3-03), blocking-human APPROVED, params match the test base_params (depth 2, 5 iters, lr 0.3, l2_leaf_reg 5, Plain) — no fabricated fixture. Task 3: `#[ignore]` + the 06.3-13 deferral comment REMOVED; the full 4-stage gate (Splits|LeafValues|StagedApprox|Predictions) PASSES ≤1e-5 (`1 passed; 0 ignored`), NO tolerance weakened / no compare.rs change. Resolves the tree-0 split-1 SPLIT-SELECTION divergence (upstream f0@1.628 vs prior pointwise f1@1.816). D-04 non-regression PROVEN: pairlogit/queryrmse/querysoftmax/lambdamart/yetirank oracles all still ≤1e-5; tree_pairwise lib 15/15. gsd-tools CLI ABSENT → STATE/ROADMAP/REQUIREMENTS updated MANUALLY.
- [x] 06.3-17-PLAN.md — Wave 3 (depends 16): GAP 2 / truth #5 — YetiRankPairwise end-to-end per-stage oracle CLOSED ≤1e-5; autonomous: false (LOSS-04) — **COMPLETE** (68c8e2e/f6c2ff2/566d4d3/9552bab/d33c9a5 instrumentation+groundtruth, bc7b661 calibration+WR-02, 6a87e63/81c2772 docs): `yetirank_pairwise_end_to_end_per_stage` passes the FULL 4-stage gate (Splits|LeafValues|StagedApprox|Predictions) ≤1e-5 vs the genuine catboost 1.2.10 fixture. Built the instrumented multi-tree pairwise trainer NOW (incremental rebuilds over the persistent `/tmp/cb_build313` + clang-18) + env-gated `CB_INSTRUMENT_LOG` per-tree/per-level/per-candidate + `update_pairs`/`competitor_weight` fences (RUN-ONCE/COMMIT, D-08/D-11). TRUE root cause = **WR-02 candidate-feature undercount**: `yetirank_n_candidate_features` counted only SELECTED-border float features (3) while the trainer draws an Rsm + normal per ALL quantized float features (4) — corpus feature 2 ends unused but was a training candidate; fixed to `feature_borders.len()`. REFUTED the prior child-RNG-bypass hypothesis (the `cand_score_rng` fence: every `*Pairwise` candidate draws `dist=Normal/stdev=0` DIRECTLY on `LearnProgress->Rand`); the `YetiRankTreeSeeder` pairwise flag is now a no-op. New `yetirank_pairwise_tree_rng_oracle_test` asserts the seeder lands the per-tree call-count fences (0/34/76/108/146/186) + reproduces the deriv/learnfold/leafval recalc seeds bit-exact for all 5 trees (`cb-core TFastRng64::call_count()` mirrors `GetCallCount()`). WR-04 FIXED (typed `OutOfRange`, 8ac7893); WR-01 MASKED (bootstrap=No/random_strength=0 gate passes; non-fixture-reachable desync deferred to hardening). NO `#[ignore]`, NO tolerance weakened, NO fabricated fixture. Non-regression: YetiRank pointwise 2/2, PairLogitPairwise 1/1, PairLogit/QueryRMSE/QuerySoftMax/LambdaMart green; cb-train lib 194/194, cb-core 26/26, cb-compute 131/131, all 47 cb-train test binaries pass. The ff10d51 DEFERRED note SUPERSEDED. **YetiRankPairwise (truth #5) now CLOSED.** gsd-tools CLI ABSENT → STATE/ROADMAP/REQUIREMENTS updated MANUALLY.
- [x] 06.3-18-PLAN.md — Wave 1 (parallel with 15-17, independent): GAP 3 / truth #5+#7 — StochasticRank per-group noise-seed (randomSeed+group_index) closure — **COMPLETE** (165c531 fixture+per-tree-noise-GT / a951bb0 production fix / 75259cd test activation): `stochasticrank_end_to_end_per_stage` passes the FULL 4-stage gate (Splits|LeafValues|StagedApprox|Predictions) ≤1e-5; `stochasticrank_pertree_noise_oracle` (D-07) bit-exact vs the instrumented catboost 1.2.10 per-tree noise GT (110 events/40 streams/10 base seeds). TWO root causes: (1) per-tree noise seeding — `boosting.rs` passed the FIXED `params.random_seed` every tree; now drives StochasticRank off the same per-tree context-RNG advance as YetiRank (`YetiRankTreeSeeder`), `recalc_seeds[0]`=derivative base + `recalc_seeds[2]`=leaf-value base (distinct leaf-value der re-compute), per-group noise seed = `base+group_index`, verified == the 10 GT cluster bases. (2) per-query approx centering — the der `mean` + SFA approx projection read the per-query `approxes`, which catboost feeds GROUP-MEAN-CENTERED; isolated via an INCREMENTAL rebuild of the warm instrumented `_catboost.so` (one TU over `/tmp/cb_build313`+clang-18, `.venv` python3.13) with a new `srank_rawder` per-doc hook; `stochastic_rank_group_der` now centers at entry. Also matched catboost `Log2(x)==log(x)*M_LN2_INV`. NO `#[ignore]`/NO tolerance weakened/NO fabricated fixture. Non-regression: YetiRank 2/2, YetiRankPairwise 3/3, PairLogit{,Pairwise} 1/1, QueryRMSE/QuerySoftMax/LambdaMart green; ranking_metrics 18/18; cb-compute 131/131. **StochasticRank (truth #5+#7) CLOSED — LOSS-04 FULLY SATISFIED (gaps #1/#2/#3 all closed).** (LOSS-04)

> After Wave 7: all three remaining LOSS-04 gaps (PairLogitPairwise, YetiRankPairwise, StochasticRank end-to-end + D-07) target closure; LOSS-05 already complete (18/18). The pairwise split-scorer (15) is the single subsystem unblocking both *Pairwise losses.

### Phase 6.4: Score Functions, Uncertainty & Custom Objectives

**Goal**: The remaining score functions, uncertainty estimation, and the Rust custom-objective/-metric trait all work and are oracle-locked ≤1e-5; the trait is designed for a clean Phase-8 PyO3 wrap (Python callback deferred).
**Mode:** mvp
**Depends on**: Phase 6.3
**Requirements**: LOSS-09, LOSS-08, LOSS-06 (uncertainty prediction types), LOSS-07 (Rust trait half)
**Success Criteria** (what must be TRUE):

  1. Score functions SolarL2, NewtonL2, NewtonCosine, LOOL2, SatL2 extend the existing `cb-compute` `EScoreFunction` enum (Cosine/L2 already shipped, 05-19 Task A) and each match upstream ≤1e-5.
  2. Uncertainty estimation — RMSEWithUncertainty + virtual ensembles — works, and the deferred LOSS-06 uncertainty prediction types (RMSEWithUncertainty/VirtEnsembles/TotalUncertainty, Phase-4 D-10) are implemented and oracle-locked ≤1e-5.
  3. The Rust custom-objective/-metric trait (user-supplied der1/der2 + eval) is oracle-tested against a Rust-defined reference; designed so the Phase-8 PyO3 callback wraps it cleanly. Python callback bridge DEFERRED to Phase 8 (D-09).

**Plans**: 4 plans (Waves A/B/C, family-wave per D-6.4-01)
Plans:
**Wave 1**

- [x] 06.4-01-PLAN.md — Wave A (LOSS-09): 5 score functions (SolarL2/NewtonL2/NewtonCosine/LOOL2/SatL2) extend EScoreFunction; transcribe-then-self-oracle (GPU-only upstream, weakened-oracle caveat D-6.4-06) — COMPLETE 2026-06-17 (3 tasks, 6 files; self-oracle ≤1e-12; Cosine/L2 lock unregressed; Newton live-search der2 deferred to Phase-7 GPU)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 06.4-02-PLAN.md — Wave B (LOSS-08): Loss::RmseWithUncertainty 2-dim diagonal loss on the 6.2 N-dim spine; training per-stage oracle ≤1e-5

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 06.4-03-PLAN.md — Wave B (LOSS-06): 3 uncertainty prediction types (RmseWithUncertainty/VirtEnsembles/TotalUncertainty) + apply_virtual_ensembles; oracle ≤1e-5 (Phase-4 D-10 closed)

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 06.4-04-PLAN.md — Wave C (LOSS-07): CustomObjective/CustomMetric Rust traits via Arc<dyn> seam; self-oracle ≤1e-5 (PyO3 deferred to Phase 8, D-09)

### Phase 6.5: Text & Embedding Features

**Goal**: All six text and embedding calcers produce upstream-matching encodings ≤1e-5, with tokenizer parity nailed first as the load-bearing risk.
**Mode:** mvp
**Depends on**: Phase 6.4
**Requirements**: FEAT-01, FEAT-02
**Success Criteria** (what must be TRUE):

  1. Tokenizer parity — the upstream text-processing token stream is reproduced bit-identical before any calcer is scored (D-11 named first risk).
  2. Text calcers BoW, NaiveBayes, BM25 produce upstream-matching encodings ≤1e-5. — **CLOSED (06.5-09):** BoW/NaiveBayes per-stage ≤1e-5 (06.5-03/04); BM25 per-stage ≤1e-5 (06.5-09) after 06.5-08's PATH-A finding that the splits.npy ±1.24 "BM25 normalized borders" were the DEFAULT EMBEDDING calcer's borders mislabeled (the fixture pool wrongly included emb0). Fixture regenerated text-only → genuine O(1e-3) BM25 text-feature borders; full BM25 per-stage oracle GREEN (Splits/LeafValues from the online-estimate tree, StagedApprox/Predictions via the offline whole-set apply column).
  3. Embedding calcers LDA, KNN produce upstream-matching encodings ≤1e-5.
  4. Text/embedding columns flow through the `Pool` (DATA-01) → calcer → quantized features into the existing tree path; calcer internals get C++ instrumentation where Python-reachable ground truth is thin (D-07).

**Plans**: 9 plans in 9 waves (strict tokenizer-first ladder — D-01 GATE blocks every calcer; narrowest-first per RESEARCH: tokenizer/dictionary → BoW → NaiveBayes+BM25 → LDA (spike-led) → KNN (spike-led, highest risk) → SC-4 integration; embedding plans 05/06 are `autonomous: false`, each opens with an instrumented-GT spike that drives the f32-LAPACK / HNSW reproduction decision from real divergence numbers, any external linear-algebra crate gated behind `checkpoint:human-verify`, escalate-don't-weaken on infeasible ≤1e-5)

Plans:

**Wave 1**

- [x] 06.5-01-PLAN.md — Wave 0: re-provision + rebuild the instrumented catboost 1.2.10 trainer with text/embedding CB_INSTRUMENT_LOG hooks (token stream / dict ids / TText / calcer encodings / online order / LDA projection / KNN neighbors, D-07) + single-thread per-stage fixture corpora for all 5 calcers + the tokenizer D-01 corpus — **COMPLETE** (f411da4/0b1e4a4): 7 env-gated hooks rebuilt RC=0, smoke dump fires all 7 categories non-empty; per-stage `.npy`+`model.cbm` fixtures (model.json export forbidden for text/embedding models) for BoW/NaiveBayes/BM25/LDA/KNN thread_count=1 + D-01 instrumented tokenizer corpus; corpus 16 rows → OLB pinned 1 (A4); vendored catboost-master/ patches UNCOMMITTED (D-09/D-12). Ground truth only — no parity assertion here; Wave 0 gate GREEN, unblocks Plans 02-07

**Wave 2** *(blocked on 06.5-01 — D-01 GATE)*

- [x] 06.5-02-PLAN.md — Tokenizer + frequency dictionary + digitizer + TText (SC-1 load-bearing gate): ByDelimiter split/lowercase, deterministic (count DESC, token ASC) dictionary build, sorted-RLE TText, bit-exact vs the instrumented D-07 dump — every calcer blocks on this

**Wave 3** *(blocked on 06.5-02)*

- [x] 06.5-03-PLAN.md — BoW (target-independent, simplest slice) + the SC-4 estimated-feature integration seam (estimated floats → existing borders/quantize/tree); BoW per-stage oracle ≤1e-5 (FEAT-01) — COMPLETE: bag_of_words_compute + BiGram dictionary (deferred from SC-1, ttext bit-exact) + build_bow_estimated_features seam; 4-stage BoW oracle ≤1e-5 (Newton+Cosine, depth-2→depth-1 canonicalization); D-04 non-text path byte-identical; cb-train lib 202 + cb-data lib 106 pass

**Wave 4** *(blocked on 06.5-03)*

- [x] 06.5-04-PLAN.md — NaiveBayes + BM25 (shared online-text seam, D-03 read-before-update prefix over the TFold learn permutation); per-stage + per-prefix oracles ≤1e-5 — **COMPLETE** (d53a7a1/8eeef44): NaiveBayes (naive_bayesian.cpp:14-63) + BM25 (bm25.cpp:12-83) calcer math + online_text_prefix read-before-update loop (mirror ctr/online.rs); ONLINE estimate feeds the Plain tree (NaiveBayes border 0.590515 matches online, not offline 0.5). NaiveBayes per-stage oracle ≤1e-5 (Splits/LeafValues/StagedApprox/Predictions) + per-prefix leakage-order anchors (no-leakage doc0=0.5 + head/tail prefix-boundary vs instrumented dump). BM25 calcer math bit-exact ≤1e-5 vs independent closed-form online ref + no-leakage anchor + SC-4 quantizer. D-04 byte-identical; cb-train lib 210 + cb-compute lib 151. **Deferred (deferred-items.md):** BM25 per-stage NORMALIZED-border scale (splits.npy ±1.24 vs raw O(1e-3)) + depth-2 [7,2,0,7] = catboost estimated-feature normalization + permutation averaging (trainer concern, NOT calcer math) → FEAT-01 NOT yet fully closed (BM25 per-stage normalized borders remain)

**Wave 5** *(blocked on 06.5-04; autonomous: false — LDA landmine A1)*

- [x] 06.5-05-PLAN.md — LDA (first embedding calcer): instrumented-GT projection SPIKE → eigensolver decision (hand-roll f32 vs LAPACK crate behind checkpoint:human-verify vs documented tolerance) → LinearDA calcer + online-embedding prefix; LDA per-stage + projection oracle ≤1e-5 or human-signed-off tolerance (FEAT-02)

**Wave 6** *(blocked on 06.5-05; autonomous: false — KNN landmine A2, highest risk)*

- [x] 06.5-06-PLAN.md — KNN (last/highest-risk): instrumented per-query neighbor-id SPIKE → exact online_hnsw hand-port vs brute-force-exact L2 (NEVER a third-party HNSW crate) → neighbor-vote calcer; KNN per-stage + neighbor-id oracle ≤1e-5 or human-signed-off tolerance — completes FEAT-02 / SC-3

**Wave 7** *(blocked on 06.5-04 + 06.5-06)*

- [x] 06.5-07-PLAN.md — SC-4 integration: mixed text+embedding pool flows Pool → calcers → estimated floats → existing quantize → tree in one trained model; end-to-end per-stage oracle ≤1e-5 (no-text/no-embedding path byte-identical) — FEAT-01 + FEAT-02 terminal hard gate — **COMPLETE** (a20cd9b/2be89ba): `build_mixed_estimated_features` joins numeric + BoW text + KNN embedding into ONE float-feature layout `[numeric|text|embedding]` through the EXISTING `select_borders_greedy_logsum` quantizer→tree (SC-4, no parallel quantizer; KNN offline whole-set Plain-tree estimate; inert-when-absent D-04). SC-4 mixed end-to-end oracle: **StagedApprox + Predictions match upstream catboost 1.2.10 ≤1e-5 BIT-FOR-BIT** (text AND embedding flowing together → upstream's model); Splits/LeafValues gated structure-invariantly under a documented feature-selection tie (1 distinct split/tree + valid separating border; per-tree leaf-value MULTISET ≤1e-5 — magnitudes exact, only ambiguous leaf ORDER freed; upstream's own `splits.npy` `[0.0,0.0,0.5,0.0,0.0]` shows the tie). Mixed scoped to BoW + KNN (the two fully per-stage-closed calcers); BM25 normalized borders (06.5-04) + LDA raw-projection tolerance (06.5-05) EXCLUDED. 5 oracle tests, 0 ignored; cb-train lib 228 + cb-data 106 + cb-compute 176; D-04 no-text e2e unchanged; clippy-clean. **FEAT-02 COMPLETE.** **RESIDUAL (FEAT-01, for the phase verifier):** BM25 per-stage NORMALIZED borders (06.5-04 deferred-items.md) + general estimated-feature quantization-GRID parity (KNN vote / BoW digitization grid; surfaced by an XOR-corpus prototype) deferred to a follow-on trainer-estimated-feature-normalization slice — FEAT-01 NOT yet fully closed

**Wave 8 — gap closure** *(from 06.5-VERIFICATION.md gaps_found, 3/4 SC; the ONE open gap is the BM25 per-stage NORMALIZED-border scale, SC-2/FEAT-01. Plan 08 is an upstream-source investigation gate that records a binding A/B decision; Plan 09 branches on it. Blocks on existing 06.5-04 + 06.5-07.)*

- [x] 06.5-08-PLAN.md — Investigation GATE — **COMPLETE** (b51aa25/1849b04): **DECISION PATH-A**. Source path read end to end (`base_text_feature_estimator.h:74-88` raw O(1e-3) column → `estimated_features.cpp:204-250` `BestSplit` on raw values, no transform → `split.cpp:45-46` → `model.cpp:209`, scale-preserving) + instrumented `cb_instr_estimated_borders` RUN-ONCE dump prove the BM25 estimated-feature borders are O(1e-3), NOT ±1.24. The committed `splits.npy` ±1.24 / -0.550486 borders ALL carry `calcer_id=96AE6D4D` (default EMBEDDING calcer on `emb0`), NOT the BM25 text calcer `4559D4B0`; `model.cbm` has an `emb0` feature. **There is NO BM25 normalization — the ±1.24 is a fixture mislabel (splits.npy frozen from a text+embedding pool).** 06.5-09 = fixture-correctness fix (regenerate text-only BM25 fixtures + per-stage oracle ≤1e-5). Vendored patch UNCOMMITTED (D-09/D-12). Decision → `BM25-NORMALIZATION-DECISION.md`. `autonomous: true`

**Wave 9 — gap closure** *(blocked on 06.5-08; executes the recorded path)*

- [x] 06.5-09-PLAN.md — Close BM25 SC-2/FEAT-01 along the 06.5-08 PATH-A path — **COMPLETE** (3426145/+oracle): executed PATH-A (the recorded DECISION) as a FIXTURE-CORRECTNESS fix, NOT a normalization implementation (the PLAN.md normalization-transform wording predated 08's conclusion → DEVIATION recorded). 06.5-08 proved the splits.npy ±1.24/-0.550486 borders carry `calcer_id=96AE6D4D` (the DEFAULT EMBEDDING calcer on `emb0`), NOT the BM25 text calcer — the fixture pool wrongly included `embedding_features=[emb0]`, whose ±1.0 clouds dominated the split search. `gen_text_embedding_fixtures.py::_make_pool(text_only=True)` drops `emb0` from the text-calcer path; BM25/BoW/NaiveBayes fixtures regenerated single-thread text-only (BoW/NaiveBayes per-stage `.npy` byte-identical → zero regression; BM25 splits now the genuine O(1e-3) BM25 text-feature borders `0.00248965, 0.00127047, …`, `calcer_id=0BDFE5…`). Full BM25 per-stage oracle GREEN, 0 ignored: Splits/LeafValues ≤1e-5 from the ONLINE-estimate tree; StagedApprox/Predictions ≤1e-5 (2.9e-8) via the OFFLINE whole-set apply column (the Plain-mode online-tree / offline-apply contract `online_text.rs` documents — doc 0's online no-leakage value is 0 but its offline value routes it to the correct leaf, the one place BM25 differs from NaiveBayes). NO production trainer change (the Rust seam already produces O(1e-3) BM25 borders); NO #[ignore], NO weakened tolerance. `autonomous: false` (blocking SC-2/FEAT-01 closure checkpoint)

### Phase 6.6: Advanced Features & Non-Symmetric Trees

**Goal**: The advanced-feature surface — monotone constraints, penalties, recursive feature selection, alternative grow policies (a second, non-symmetric tree engine), and advanced fstr — matches upstream ≤1e-5; the largest and riskiest structural item in Phase 6.
**Mode:** mvp
**Depends on**: Phase 6.5
**Requirements**: FEAT-03, FEAT-04, FEAT-05, FEAT-06, MODEL-05 (also completes the MODEL-03 LossFunctionChange importance, Phase-4 D-12)
**Success Criteria** (what must be TRUE):

  1. Monotone constraints (per-feature +1/-1/0), feature penalties and per-object penalties match upstream ≤1e-5.
  2. Recursive feature selection by PredictionValuesChange / LossFunctionChange / ShapValues matches upstream.
  3. Alternative grow policies Lossguide/Depthwise produce true non-symmetric trees — full train + non-symmetric apply + `.cbm`/json round-trip oracle-locked ≤1e-5 (D-10; touches `cb-train` AND `cb-model`, wiring into the existing `TNonSymmetricTree*` bindings; its own multi-wave gate). **Region is OUT OF SCOPE** (CPU-unimplemented upstream → no ground truth) and **non-symmetric monotone is OUT OF SCOPE** (upstream rejects it) — both recorded as escalated gaps and enforced by typed-error guards, not built (06.6-RESEARCH gate 1, D-6.6-07).
  4. Advanced fstr — ShapInteractionValues, PredictionDiff, SAGE — and the deferred MODEL-03 LossFunctionChange importance (D-12) match upstream ≤1e-5.

**Plans**: 8 plans in 7 waves + 1 gap-closure plan (Gate A symmetric-features-first → Gate B non-symmetric engine multi-wave gate → Gate C advanced fstr → Gate D feature selection LAST, per D-6.6-01..03; gap-closure 06.6-09 closes the SC-3 grower→save→load→predict gap)

Plans:

**Wave 1** — Gate A (symmetric features, rides existing oblivious grower)

- [x] 06.6-01-PLAN.md — Feature/per-object penalties (FEAT-04): feature_weights (multiplicative) + first_feature_use/per_object (subtractive PenalizeBestSplits) on the oblivious grower, oracle ≤1e-5

**Wave 2** *(blocked on 06.6-01 — shares boosting.rs)*

- [x] 06.6-02-PLAN.md — Monotone constraints OBLIVIOUS-ONLY (FEAT-03): verbatim isotonic (PAVA) leaf-delta projection (calc_monotonic_leaf_deltas + build_monotonic_linear_orders, transcribing monotonic_constraint_utils.cpp + CalcMonotonicLeafDeltasSimple) + monotone_constraints BoostParams param + validate_monotone_constraints typed-error guard — monotone_oracle_test ≤1e-5 (LeafValues/StagedApprox/Predictions vs catboost 1.2.10; fixture pins model_shrink_rate=0 to isolate the PAVA). Default path byte-identical (D-6.6-05). Non-symmetric-monotone + Region grow_policy guards DEFERRED to 06.6-04 (grow_policy not yet defined; commented TODO, no silent drop). Commits 5c2761c / fb40de4

**Wave 3** — Gate B Wave-0 (non-symmetric engine contract)

- [x] 06.6-03-PLAN.md — Non-symmetric model variant + .cbm/json serde (wire existing TNonSymmetricTreeStepNode bindings) + distinct "trees" oracle parser + splits-first failing harness (FEAT-06 infra)

**Wave 4** *(blocked on 06.6-02, 06.6-03)*

- [x] 06.6-04-PLAN.md — Unified policy-parameterized leaf-wise grower (Depthwise + Lossguide; Region out) + grow_policy dispatch + from_trained lift; SPLITS oracle-locked, SymmetricTree arm byte-identical (FEAT-06)

**Wave 5** *(blocked on 06.6-03, 06.6-04)*

- [x] 06.6-05-PLAN.md — Non-symmetric pointer-walk apply + leaf values + FULL per-stage + .cbm/json round-trip oracle ≤1e-5 (FEAT-06 / SC-3 complete) — **COMPLETE** (6fcb0e8 feat / 8642618 test): `leaf_index_nonsym()` flat-node pointer walk (evaluator_impl.cpp:726-742) gated on tree variant; `predict_raw_one`/`predict_raw_multi_cat` branch per-tree, oblivious arm byte-identical (D-6.6-05). Full `non_symmetric_oracle_test` GREEN — Depthwise + Lossguide per-stage (Splits|LeafValues|StagedApprox|Predictions) ≤1e-5 vs catboost 1.2.10 + `.cbm` AND `model.json` round-trip re-predict, none `#[ignore]`d. Fixed 2 latent 06.6-03 serde bugs (Rule 1): non-symmetric `.cbm` decode under-counted Lossguide distinct leaves (one-sided `(d,0)`/`(0,d)` halts ARE leaves; GLOBAL↔LOCAL leaf-id reconciliation) + json `unflatten` one-sided-halt expansion. LeafValues gate compares the per-tree sorted multiset (representation-independent; apply equivalence locked by StagedApprox/Predictions/round-trip). Non-symmetric leaf-VALUE estimation already wired by 06.6-04 (shared fold machinery, no forked formula; Open Question 1 stays RESOLVED, no instrumented-trainer escalation). Non-regression: cb-model apply/fstr/shap/cbm oracles green, cb-train lib 228, grower SPLITS oracle 1. **FEAT-06 / SC-3 COMPLETE.** gsd-tools CLI ABSENT → STATE/ROADMAP updated MANUALLY.

**Wave 6** — Gate C (advanced fstr; 06+07 parallel, zero file overlap)

- [x] 06.6-06-PLAN.md — LossFunctionChange importance (completes MODEL-03/D-12) + non-symmetric PVC/Interaction recursion, oracle ≤1e-5 (≥1 non-symmetric case) — **COMPLETE** (c9c28b5 feat / 091ee29 test): `cb_model::loss_function_change(model, cols, labels, n_features)` transcribing `loss_change_fstr.cpp:154-356` (Logloss single-dim: `score[f] = finalError(approx − shap[·][f]) − finalError(approx)`, built on `predict_raw` + `shap_values` + additive-Logloss mean BCE) oracle-locked ≤1e-5 vs `get_feature_importance(type='LossFunctionChange', data=pool)`; `FeatureImportanceType::LossFunctionChange` variant (completes D-12). `prediction_values_change`/`interaction` generalized to non-symmetric trees (D-6.6-10): oblivious arms byte-identical (D-6.6-05); non-symmetric PVC = node-graph `CalcEffectForNonObliviousModel` (feature_str.h:149-228, two-pass triangle recursion); non-symmetric Interaction = signed DFS `CalcMostInteractingFeatures` (feature_str.cpp:226-295) → shared `CalcFeatureInteraction` scoring. Depthwise non-symmetric PVC (+Σ=100) + Interaction oracle-locked ≤1e-5; existing oblivious cases unchanged. node_count-bounded recursion + `sum_f64` (T-06.6-15/16). Facade `feature_importance` made exhaustive (Rule 3) + `feature_importance_with_data(type, pool)` for the data-bearing LFC path. **MODEL-03/D-12 COMPLETE.** gsd-tools CLI ABSENT → STATE/ROADMAP/REQUIREMENTS updated MANUALLY.
- [x] 06.6-07-PLAN.md — Non-symmetric TreeSHAP + ShapInteractionValues/PredictionDiff/SAGE (seed-match strict, no instrumentation), oracle ≤1e-5 (≥1 non-symmetric case) (MODEL-05 / SC-4 complete)

**Wave 7** — Gate D (feature selection LAST, blocks on Gate C)

- [x] 06.6-08-PLAN.md — Recursive feature selection by ShapValues / PredictionValuesChange / LossFunctionChange (NEW cb-train module, no new crate), selected/eliminated set oracle (FEAT-05 / SC-2 complete)

**Wave 8** — Gap closure (verification gaps_found 4/5, SC-3 partial)

- [ ] 06.6-09-PLAN.md — Close CR-02 sentinel mismatch: leaf_wise_grower finalization inits node_id_to_leaf_id to u32::MAX (interior sentinel) + checked u16::try_from step-node diffs (CR-01); NEW cb-model oracle test trains a non-symmetric model via the Rust grower → save_cbm → load_cbm → predict ≤1e-5 (FEAT-06 / SC-3 grower→save→load→predict complete)

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
