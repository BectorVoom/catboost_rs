# Roadmap: catboost-rs

## Milestones

- ✅ **v1.0 Core Parity** — Phases 1–8 (shipped 2026-06-28)
- ✅ **v1.1 GPU Performance** — Phases 10–14 (shipped 2026-07-05) — the boosting inner loop moved fully device-resident (CubeCL, no CUB); BENCH-03: PASS, 23.9×–42.1× vs the host-light CPU baseline on P100. Closed with accepted standing debt (GPUT-14 aggregate + Phase-10/11 BENCH-02 un-run). Full detail: `milestones/v1.1-ROADMAP.md`.
- 🚧 **v1.2 Parity Completion & Release Readiness** — Phases 15–22 (in progress) — discharge v1.1 debt + close the last CPU parity gap (online-HNSW), add ONNX/CoreML export, extended fstr, GPU inference, CV/tuning/snapshot orchestration, rewrite the CPU split search onto histograms (close the ~250–450× CPU-training speed gap found in Spike 002), then benchmark vs official CatBoost + PyPI release as the capstone.

## Phases

<details>
<summary>✅ v1.0 Core Parity (Phases 1–8) — SHIPPED 2026-06-28</summary>

- [x] Phase 1: Workspace, Lint Discipline & Oracle Harness
- [x] Phase 2: Data Layer — Pool, Quantization & Reduction
- [x] Phase 3: CPU Training Core — Plain Boosting & Oblivious Trees
- [x] Phase 4: Model Serialization, SHAP & Rust API (first full oracle lock)
- [x] Phase 5: Ordered Boosting, Ordered CTR & Categoricals
- [x] Phase 6: Full Loss & Feature Parity (6.1 regression · 6.2 multiclass/N-dim · 6.3 ranking · 6.4 score-fns/uncertainty/custom · 6.5 text/embedding · 6.6 advanced + non-symmetric)
- [x] Phase 7: GPU Backends via CubeCL — structural parity (7.1 primitives · 7.2 grad/hess · 7.3 pointwise hist · 7.4 pairwise hist · 7.5 on-device grow loop · 7.6 rocm tolerance sign-off)
- [x] Phase 8: Python Bindings, Dual API & Packaging

Full per-phase detail: `.planning/milestones/v1.0-ROADMAP.md` and `.planning/milestones/v1.0-REQUIREMENTS.md`.
61/62 v1 requirements complete; known gaps carried forward (see Backlog + `.planning/MILESTONES.md`).

</details>

<details>
<summary>✅ v1.1 GPU Performance (Phases 10–14) — SHIPPED 2026-07-05</summary>

**Milestone goal:** Move the entire boosting inner loop (histogram build, split scoring, BestSplit, partition/leaf-assignment, leaf values) onto the GPU — not just derivatives — closing the >20× gap vs official CatBoost GPU while preserving correctness. Re-scoped in place 2026-07-02 against `CATBOOST_CUDA_KERNELS_DESIGN.md` (79 `.cu` + 77 `.cuh` across 9 kernel directories; 17 → 25 requirements). All GPU kernel oracles (correctness AND speed) validated on Kaggle CUDA (ROCm in-env not a gate); device path held to ε=1e-4 vs the Rust CPU path (depth-1 tighter at ≤1e-5); CPU path byte-unchanged (D-04).

- [x] Phase 10: GPU Foundations — Runtime Seam, Session Residency, Device-Primitive Library, Compressed Index, Depth-1 + Kaggle CUDA Oracle & Speed Harness (9/9 plans) — completed 2026-07-03
- [x] Phase 11: Depth>1 Partition-Aware Histograms + Reduction Determinism + Newton Der2 (5/5 plans) — completed 2026-07-04
- [x] Phase 12: Grow-Policy, Leaf-Method, Sampling & Categorical Device Coverage (9/9 plans) — completed 2026-07-04
- [x] Phase 13: Pairwise, Ranking, Multiclass, Ordered & Langevin Device Coverage (10/10 plans) — completed 2026-07-04
- [x] Phase 14: Comprehensive Kaggle CUDA Speed Benchmark + Parity Sign-Off (3/3 plans) — completed 2026-07-05

Full per-phase detail: `.planning/milestones/v1.1-ROADMAP.md` and `.planning/milestones/v1.1-REQUIREMENTS.md`.
**Standing debt accepted at close** (formal override in `14-VERIFICATION.md`): `GPUT-14` milestone-wide aggregate sign-off Pending; Phase-10 (depth-1) + Phase-11 (depth-6) BENCH-02 Kaggle rows un-run. Per-family ≤1e-4 evidence + committed P100 runs stand. Discharged in v1.2 Phase 15. See `.planning/MILESTONES.md` → Known Gaps and STATE.md → Deferred Items.

</details>

### 🚧 v1.2 Parity Completion & Release Readiness (In Progress)

**Milestone goal:** Close the remaining CatBoost surface described in `docs/CATBOOST_CORE_DESIGN.md` / `docs/CATBOOST_CUDA_KERNELS_DESIGN.md`, discharge the v1.1 standing debt, and make catboost-rs adoption-ready — held to ≤10⁻⁵ CPU / ε=1e-4 GPU parity. Build order is **debt-first → HNSW → export → fstr → GPU-infer → orchestration → adoption/DX capstone** (research SUMMARY.md "Implications for Roadmap").

**Milestone-wide context carried into every phase:** Kaggle CUDA is the SOLE authoritative GPU oracle (ROCm in-env = non-gating smoke); export uses an **export-specific float32 tolerance** oracled vs CatBoost's own ONNX/CoreML export in the target runtime (NOT the ≤10⁻⁵ double bar); FEAT-07 oracles the per-object **neighbor set** index-for-index; GPU-infer reuses the v1.1 fixed-point-u64 deterministic reduction + `f32::MIN` sentinel; **never add a `cb-train` dependency to `cb-backend`** (feature-unification landmine); new backend-bearing crates keep `default-features=false` / no unconditional `cpu`.

- [x] **Phase 15: Debt Discharge & CUDA Oracle Re-establishment** — GPUT-14 aggregate ε=1e-4 sign-off, Phase-10/11 BENCH-02 rows, RV-13-01..04 hazards; re-establishes the trusted CUDA oracle everything downstream rests on (completed 2026-07-05)
- [ ] **Phase 16: Online-HNSW KNN Estimated-Feature Parity** — bit-for-bit port of upstream `online_hnsw` closing the last open ≤10⁻⁵ CPU parity gap (parallel with Phase 15, different crates)
- [ ] **Phase 17: Model Export — ONNX + CoreML** — read-only float-only exporters with upstream guards, oracled under an export-specific tolerance
- [ ] **Phase 18: Extended Feature Importance** — Interaction, LossFunctionChange, partial-dependence (parallel with Phase 17)
- [ ] **Phase 19: GPU Inference Evaluator** — device-side predict for the upstream subset, deterministic, Kaggle-CUDA-signed (depends on Phase 15's re-signed oracle)
- [ ] **Phase 20: Orchestration — CV, Tuning, Snapshot/Resume, calc_metrics** — new `cb-orchestrate` crate + a `BoostingCheckpoint` surface on `cb-train` (parallel with Phase 19)
- [x] **Phase 21: CPU Split-Finding Histogram Rewrite** — replace the per-candidate full-dataset rescan with per-feature bin histograms + subtraction trick + parallelism across ALL CPU grow policies (oblivious, Depthwise/Lossguide, CTR-feature scoring path), preserving ≤10⁻⁵ parity; closes the ~250–450× CPU-training slowdown (Spike 002/003/004). Must precede the Phase 22 benchmark. (completed 2026-07-05)
- [ ] **Phase 22: Adoption / DX Capstone** — benchmark vs official CatBoost, PyPI per-backend wheels, docs + examples, real-dataset validation (last — exercises Phases 17/19/20/21)

## Phase Details

### Phase 15: Debt Discharge & CUDA Oracle Re-establishment

**Goal**: The v1.1 GPU claims are backed by a single authoritative Kaggle CUDA correctness + speed record with no stitched gaps, and all latent parity hazards are resolved — re-establishing the trusted oracle every later parity/benchmark claim depends on.
**Depends on**: Phase 14 (v1.1 device families + Kaggle CUDA harness)
**Requirements**: HARD-01, HARD-02, HARD-03
**Success Criteria** (what must be TRUE):

  1. A single aggregate GPUT-14 ε=1e-4 Kaggle CUDA correctness row covers all v1.1 device families as one authoritative run and passes (HARD-01)
  2. Phase-10 (depth-1) and Phase-11 (depth-6) BENCH-02 speed rows are executed on Kaggle CUDA and the BENCH-03 aggregate is recomputed from real numbers with no stitched Phase-12/13-only gaps (HARD-02)
  3. Each RV-13-01..04 latent parity hazard is either fixed (with an oracle demonstrating the fix) or explicitly retired with recorded evidence (HARD-03)

**Plans**: 4/4 plans complete
Plans:
**Wave 1**

- [x] 15-01-PLAN.md — RV-13-01/02 ranking-der oracles (tie-order stability + weight>0 softmax max-seed) [Wave 1, HARD-03]
- [x] 15-02-PLAN.md — RV-13-03/04 oracles (n==0 empty-group guard + near-equal-border pairwise tie-break) [Wave 1, HARD-03]

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 15-03-PLAN.md — single authoritative Kaggle CUDA session (Part A ALL-PASS ε=1e-4 all 13 families + 4 RV-13; Part B 12 depth-1/depth-6 rows 29.1–40.8×) [Wave 2, HARD-01/02]

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 15-04-PLAN.md — 15-EVIDENCE.md + BENCH-03 recompute in place + REQUIREMENTS/MILESTONES/STATE bookkeeping flip [Wave 3, HARD-01/02/03]

**Context**: Mostly Kaggle CUDA job execution + contained `cb-backend`/`cb-train` fixes — high de-risking, low code-change risk. ROCm in-env is non-gating. This phase re-establishes the CUDA oracle before anyone benchmarks against it (avoids the benchmark-baseline-confusion pitfall).

### Phase 16: Online-HNSW KNN Estimated-Feature Parity

**Goal**: The KNN estimated-feature calcer returns upstream-identical neighbor sets, closing the last open ≤10⁻⁵ CPU parity gap via a bit-for-bit transcription of upstream online-HNSW.
**Depends on**: Phase 6.5 (estimated-feature calcer + frozen `text_embedding_xor/` fixture) — no v1.2 dependency; runs parallel with Phase 15 (`cb-compute` only, disjoint crates). Supersedes the deferred Phase 9 backlog.
**Requirements**: FEAT-07
**Success Criteria** (what must be TRUE):

  1. The Rust online-HNSW returns upstream-identical neighbor IDs index-for-index on the instrumented neighbor-set evidence corpus — the divergence from the current brute-force-exact calcer is reproduced, not merely "close" (FEAT-07)
  2. Both the online (`TKNNUpdatableCloud`) and offline (`TKNNCloud`) calcer paths match upstream neighbor sets bit-for-bit across the full XOR corpus, and the class-vote order matches upstream (feat0 = class-1 vote)
  3. The text+embedding+numeric XOR corpus StagedApprox + Predictions match upstream ≤10⁻⁵ with no weakened tolerance and no `#[ignore]`

**Plans**: TBD
**Context**: Bit-exact parity IS achievable — the crux is replicating upstream's construction RNG + insertion order + `TL2SqrDistance`; an off-the-shelf HNSW crate can NEVER pass (different RNG/graph). Oracle the neighbor **set**, not just the final prediction. Self-contained ~936 LOC port in `cb-compute/src/hnsw/` + one wiring change in `cb-train/estimated`.

### Phase 17: Model Export — ONNX + CoreML

**Goal**: A user can export a float-only trained model to ONNX and CoreML matching upstream's guards and correctness, with unsupported (categorical/CTR/text/embedding) models rejected by a typed error.
**Depends on**: Phase 15 preferred for a trusted parity baseline, but genuinely independent (read-only, zero seam risk) — may run parallel with Phases 15/16.
**Requirements**: EXPORT-01, EXPORT-02, EXPORT-03
**Success Criteria** (what must be TRUE):

  1. User can export a float-only oblivious identity-scale model to ONNX via `ai.onnx.ml` TreeEnsembleRegressor/Classifier (+ZipMap), and a categorical/CTR/text/embedding model is rejected with a typed error mirroring upstream's guard (EXPORT-01)
  2. User can export a float + one-hot-categorical oblivious (≤16 levels, single-dim bias, identity-scale) model to CoreML, with unsupported models rejected by a typed error (EXPORT-02)
  3. Exported ONNX predictions match official CatBoost's own ONNX export evaluated in ONNX Runtime under the export-specific float32 tolerance (RawFormulaVal + Probability as distinct cases; binary label ignored per the ORT bug); CoreML is execution-validated on an Apple runtime if available, else structurally validated with a documented gap (EXPORT-03)

**Plans**: TBD
**Context**: **Resolve the crate-placement disagreement in this phase's plan** — STACK.md recommends a new `cb-export` crate, ARCHITECTURE.md recommends a feature-gated `cb-model/export` submodule (both defensible; affects Cargo wiring, not algorithm correctness). Do NOT hold exports to the ≤10⁻⁵ double bar (float32 drift is inherent). Pin the opset. Decide the CoreML validation environment up front.

### Phase 18: Extended Feature Importance

**Goal**: A user can compute Interaction, LossFunctionChange, and partial-dependence feature importance matching CatBoost's exact accounting (not textbook formulas).
**Depends on**: Phase 4/6.6 (shipped SHAP `TShapPreparedTrees` + basic fstr) — independent single-crate work; may run parallel with Phase 17.
**Requirements**: FSTR-01, FSTR-02, FSTR-03
**Success Criteria** (what must be TRUE):

  1. User can compute `Interaction` feature importance (pairwise split-cooccurrence over tree structure, dataset-free) matching `get_feature_importance(type=Interaction)` ≤10⁻⁵, including on CTR-feature models (FSTR-01)
  2. User can compute `LossFunctionChange` importance over a supplied dataset (reusing shipped SHAP machinery + a new cubecl-free `cb-model→cb-compute` loss-derivative edge) matching upstream ≤10⁻⁵ (FSTR-02)
  3. User can compute partial-dependence for one or two features via a staged-apply sweep matching upstream's exact grid/quantization + averaging convention (FSTR-03)

**Plans**: TBD
**Context**: Oracle each type against `get_feature_importance` on models **with** categorical/CTR features (where the accounting is hardest), not just numeric fixtures. Extends `cb-model/src/fstr/` alongside existing SHAP/basic fstr.

### Phase 19: GPU Inference Evaluator

**Goal**: A user can run device-side model inference for the upstream-supported subset — deterministic, ε=1e-4 Kaggle-CUDA-signed — with everything else falling back to CPU.
**Depends on**: Phase 15 (the re-signed CUDA oracle + v1.1 primitive library must be trustworthy before a second device path is built on them). Parallel with Phase 20 (disjoint new crates).
**Requirements**: GINF-01
**Success Criteria** (what must be TRUE):

  1. User can run device-side predict for oblivious / single-dim / float-only models on RawFormulaVal|Probability|Class, matching the CPU predictor at ε=1e-4 (GINF-01)
  2. Non-oblivious / multi-dim / categorical / unsupported-prediction-type models transparently fall back to the CPU evaluator via `Ok(None)` — matching upstream's deliberately narrow subset, not exceeding it
  3. Repeated identical GPU applies are bit-identical (v1.1 fixed-point-u64 deterministic reduction reused, `f32::MIN` sentinel, `+=`/`<<` leaf-index), and the rocm suite passes in-env with no `-inf`/`|=` HIP codegen reject
  4. GPU-inference correctness (ε=1e-4 vs CPU + vs official CatBoost `EnableGPUEvaluation`) is signed off on Kaggle CUDA (ROCm in-env non-gating)

**Plans**: TBD
**Context**: New `cb-infer-gpu` crate above both `cb-model` and `cb-backend` (respects the no-cycle rule; mirrors upstream's `libs/model/cuda` separation, design-doc line 2859); model-agnostic kernels live in `cb-backend/src/kernels/infer/`. `default-features=false`, backend passthrough. **Research-phase flag:** re-verify CubeCL HIP kernel specifics before writing kernels.

### Phase 20: Orchestration — CV, Tuning, Snapshot/Resume, calc_metrics

**Goal**: A user can cross-validate, tune hyperparameters, snapshot/resume training bit-identically, and compute metrics standalone — all matching CatBoost's exact semantics.
**Depends on**: Phase 15 (trusted training core baseline). Parallel with Phase 19 (disjoint new crates). Internal: ORCH-02 hard-depends on ORCH-01.
**Requirements**: ORCH-01, ORCH-02, ORCH-03, ORCH-04
**Success Criteria** (what must be TRUE):

  1. User can run `cv()` matching CatBoost fold semantics (Classical/Inverted/TimeSeries split types, per-loss stratification defaults, group-in-fold) with fold assignment oracled per seed and a target-permutation leakage canary scoring ~chance (ORCH-01)
  2. User can run `grid_search` and `randomized_search` hyperparameter tuning built on `cv()`, reusing the existing deterministic RNG stream with no `rand` dependency (ORCH-02)
  3. User can snapshot and resume training via a versioned `BoostingCheckpoint` (serde + format-version guard) with full RNG-state continuity — straight-run vs resumed-run predictions bit-identical for plain AND ordered boosting with sampling enabled (ORCH-03)
  4. User can compute metrics on precomputed predictions standalone (`calc_metrics` / `eval_metrics`, staged) independent of a live training run (ORCH-04)

**Plans**: TBD
**Context**: New `cb-orchestrate` crate (mirrors upstream `train_lib` driver layer) + the only training-core change this milestone — a `BoostingCheckpoint` serde surface + resume entry on `cb-train`. Compute borders/CTR **inside each fold** (leakage). Snapshot the complete trainer state (iteration, all RNG/permutation state, ordered-boosting folds, approx cursors, OD history), not just the trees. **Research-phase flag:** re-verify CV fold-partition semantics against upstream `cross_validation.cpp` at plan time.

### Phase 21: CPU Split-Finding Histogram Rewrite

**Goal**: CPU training split-finding matches CatBoost's histogram/bucket-stats algorithm — per-feature bin histograms + subtraction trick + parallelism — collapsing the ~250–450× single-thread slowdown Spike 002 measured, while preserving the ≤10⁻⁵ CPU parity bar, across ALL CPU grow policies (oblivious `SymmetricTree`, non-symmetric `Depthwise`/`Lossguide`, and the online-CTR-feature scoring path).
**Depends on**: Phase 3 (CPU training core + oblivious trees) and Phase 6.6 (non-symmetric leaf-wise growers) — both shipped. Independent of the other v1.2 phases (`cb-train` / `cb-compute` only, disjoint from export/GPU-infer/orchestration crates); may run parallel with 16–20. Must precede Phase 22 (the benchmark capstone) so the CPU baseline is competitive.
**Requirements**: PERF-01, PERF-02, PERF-03
**Success Criteria** (what must be TRUE):

  1. The CPU oblivious split search builds per-feature bin histograms (`TBucketStats`: Σder1, Σweight per bin) in ONE `O(n)` pass per level plus the subtraction trick (child = parent − sibling), replacing the per-candidate full-dataset `assign_leaves`/`reduce_leaf_stats` rescan — per-tree CPU time's `border_count` scaling collapses from the pre-rewrite ~linear blow-up to a small constant factor (measured 32→254 ≈ 3.5× at n=10000/nf=20/depth=6). **NOTE (accepted with human override 2026-07-06):** the original "flat within noise across 32→254" bar is documented as algorithmically unachievable at this harness size — the residual is the irreducible `O(n_bins·n_leaves·n_features)` split-scoring arithmetic (n-independent), so flatness needs `n ≫ n_bins·n_leaves` (n≥~100k), and official CatBoost is itself ~2.1× here; see 21-VERIFICATION.md (PERF-01)
  2. All CPU grow policies (`SymmetricTree`, `Depthwise`, `Lossguide`) AND the online-CTR-feature scoring path use the histogram scorer, and EVERY shipped ≤10⁻⁵ CPU oracle fixture stays bit-exact — parity preserved via deterministic ordered bin summation (fixed-point-u64 per Phase 10/11, or per-bin ordered `sum_f64`) so the algorithm change does not perturb `sum_f64` order (PERF-02)
  3. The split search is parallelized over features/candidates (`rayon`) with reusable scratch buffers (no per-candidate allocation storm), with a documented end-to-end speedup vs the pre-rewrite baseline on the Spike-002 grid and single-thread per-core efficiency brought within a stated target factor of official CatBoost's 1-thread times (PERF-03)

**Plans**: 7/7 plans complete

- [x] 21-07-PLAN.md

**Wave 1**

- [x] 21-01-PLAN.md — cb-compute histogram foundation (BucketHistogram build + prefix scan + subtraction) + bit-exact equivalence tests (PERF-01, PERF-02)
- [x] 21-06-PLAN.md — GAP CLOSURE (PERF-01 flatness): running-prefix O(n_bins) scan (TRUE=total−prefix) + cb_core scatter_add flat-scratch build + retained-parent subtraction advance, gated by an atomic full-oracle-suite parity run + CB_PERF 32→254 re-sweep (PERF-01, PERF-02, PERF-03)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 21-02-PLAN.md — wire oblivious plain + perturbed path to histogram + GrowScratch + n_bins-flat sweep (PERF-01, PERF-02)

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 21-03-PLAN.md — extend to non-symmetric leaf-wise Depthwise/Lossguide + Region (best_split_for_leaf) (PERF-02)

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 21-04-PLAN.md — extend to the online-CTR-feature scoring path (PERF-02)

**Wave 5** *(blocked on Wave 4 completion)*

- [x] 21-05-PLAN.md — rayon parallelism + reusable scratch + determinism test + documented speedup (PERF-03)

*Scope decision (Open Question 1): the ordered-boosting path (`score_candidate_ordered`) is a boosting TYPE, not a grow policy, and is NOT in the CONTEXT scope list — it is explicitly deferred to Phase 22, left unchanged (parity trivially preserved, residual slowness surfaced). The pairwise scorer stays excluded per CONTEXT.*
**Context**: Root cause + measured evidence live in `.planning/spikes/002-perf-baseline-and-scaling`, `003-split-finding-hotpath-audit`, `004-parallelism-and-allocation-audit`. The device histogram already exists to mirror onto the host: `cb-backend/src/kernels/pointwise_hist.rs` (Phase 11 `pointwise_hist2` + subtraction trick). The crux (and the reason the current code is slow) is preserving D-05/D-08 bit-exact summation while dropping the rescan — the parity-first `sum_f64` gather-and-sum shortcut is what abandoned the histogram. Fix the ALGORITHM first (removes the `n_bins`/`n_features` blow-up + most allocations), parallelism SECOND. **never add a `cb-train` dependency to `cb-backend`** (feature-unification landmine) — transcribe the histogram logic inline into `cb-train`/`cb-compute`, do not cross the seam. Reuse the CB_PERF-gated harness `crates/cb-train/tests/perf_baseline_test.rs` + `catboost_grid.py` to measure before/after.

### Phase 22: Adoption / DX Capstone

**Goal**: The whole parity story is proven to adopters — benchmarked vs official CatBoost, released to PyPI per-backend, documented with runnable examples, and validated end-to-end on real datasets.
**Depends on**: Phases 17 (export), 19 (GPU-infer), 20 (orchestration), 21 (competitive CPU baseline) — it exercises every one of them; must come last.
**Requirements**: DX-01, DX-02, DX-03, DX-04
**Success Criteria** (what must be TRUE):

  1. An end-to-end benchmark reports accuracy, speed, and peak-RSS memory vs official CatBoost on real datasets, matched hardware/version/params — GPU numbers from Kaggle CUDA only, CPU baseline = official CatBoost (NOT the v1.1 host-light baseline) (DX-01)
  2. Per-backend wheels (`cpu`/`cuda`/`rocm`/`wgpu` as separately-named distributions) build via the `maturin-action` CI matrix as abi3-py312 with versioning/tagging, and each imports + smoke-predicts in a clean environment (DX-02)
  3. Documentation + runnable Rust and Python examples cover training, inference, export, fstr, CV/tuning, and GPU usage (DX-03)
  4. A real-world dataset validation suite exercises the new surfaces (export round-trip, CV, tuning, GPU inference) end-to-end (DX-04)

**Plans**: TBD
**Context**: Benchmark vs **official CatBoost** (matched version/hardware/params), not the seductive v1.1 host-light numbers; warm up kernels before timed regions; report medians. Gate the free-threaded concurrency claim behind an actual `python3.13t` run or document it as a code property. Ship the rocm wheel's `ROCM_PATH`/`LD_PRELOAD` requirement; never bundle the patchelf-renamed `libhiprtc`.

## Progress

**Execution Order:**
v1.2 phases execute in numeric order: 15 → 16 → 17 → 18 → 19 → 20 → 21 → 22. Parallelizable pairs (different crates): {15, 16, 17, 18} overlap; {19, 20, 21} overlap (21 is `cb-train`/`cb-compute`-only, disjoint from 19/20's crates). Firm hard-dependencies: 19 after 15; 22 after 17/19/20/21; ORCH-02 after ORCH-01 (within 20).

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1–8 (Core Parity) | v1.0 | — | Complete | 2026-06-28 |
| 10–14 (GPU Performance) | v1.1 | — | Complete | 2026-07-05 |
| 15. Debt Discharge & CUDA Oracle Re-establishment | v1.2 | 4/4 | Complete    | 2026-07-05 |
| 16. Online-HNSW KNN Estimated-Feature Parity | v1.2 | 0/TBD | Not started | - |
| 17. Model Export — ONNX + CoreML | v1.2 | 0/TBD | Not started | - |
| 18. Extended Feature Importance | v1.2 | 0/TBD | Not started | - |
| 19. GPU Inference Evaluator | v1.2 | 0/TBD | Not started | - |
| 20. Orchestration — CV, Tuning, Snapshot/Resume, calc_metrics | v1.2 | 0/TBD | Not started | - |
| 21. CPU Split-Finding Histogram Rewrite | v1.2 | 7/7 | Complete   | 2026-07-05 |
| 22. Adoption / DX Capstone | v1.2 | 0/TBD | Not started | - |

## Backlog (Deferred from v1.0)

### Phase 9: Online HNSW Estimated-Feature Parity — SUPERSEDED by Phase 16 (FEAT-07, v1.2)

**Status**: The deferred v1.0 Phase 9 backlog is now **active as Phase 16** in the v1.2 milestone (FEAT-07). Original planning context preserved at `.planning/milestones/v1.0-phases/09-online-hnsw-estimated-feature-parity/`; the port surface (`online_hnsw/base/` + `knn.{h,cpp}`), the construction-RNG crux, and the neighbor-set oracle strategy are carried into Phase 16's plan. Do NOT plan Phase 9 separately — it is Phase 16.
