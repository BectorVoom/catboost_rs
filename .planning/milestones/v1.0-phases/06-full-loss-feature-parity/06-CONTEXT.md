# Phase 6: Full Loss & Feature Parity - Context

**Gathered:** 2026-06-15
**Status:** Ready for planning

<domain>
## Phase Boundary

Reach the **complete** CatBoost loss/metric and advanced-feature surface, additively, each loss and feature type passing its own oracle ≤1e-5 vs upstream **catboost 1.2.10** (`thread_count=1`) before the next is added. Builds on the locked CPU core (Phase 3), the first full oracle slice + `.cbm`/SHAP/fstr infrastructure (Phase 4), and the ordered-algorithm/CTR/categorical machinery (Phase 5).

**Requirements (14):** LOSS-02, LOSS-03, LOSS-04, LOSS-05, LOSS-07, LOSS-08, LOSS-09, FEAT-01, FEAT-02, FEAT-03, FEAT-04, FEAT-05, FEAT-06, MODEL-05. Also completes the LOSS-06 uncertainty prediction types deferred from Phase 4 (D-10) and the MODEL-03 `LossFunctionChange` importance deferred from Phase 4 (D-12).

**Structural decision (D-01):** This phase is **split into 6 sub-phases** (roadmap restructure required — see D-01). The phase boundary below is the union of all six; each sub-phase gets its own discuss→plan→execute→verify cycle and oracle gate.

**In scope:**
- **6.1 Regression-loss matrix (LOSS-03):** RMSE, MAE, Quantile, MultiQuantile, LogCosh, Huber, Poisson, Tweedie, MAPE, MSLE, Lq, Expectile — every named loss.
- **6.2 Multiclass/multilabel (LOSS-02):** MultiClass (softmax), MultiClassOneVsAll, MultiLogloss, MultiCrossEntropy — requires the N-dim approx refactor (D-03/D-04).
- **6.3 Ranking (LOSS-04, LOSS-05):** YetiRank(/Pairwise), PairLogit(/Pairwise), QueryRMSE, QuerySoftMax, LambdaMart, StochasticRank; metrics NDCG, DCG, MAP, MRR, ERR, PFound, PrecisionAt, RecallAt, QueryAUC — over group_id/subgroup_id/pairs.
- **6.4 Score-fns / uncertainty / custom (LOSS-09, LOSS-08, LOSS-06 uncertainty types, LOSS-07 Rust half):** score functions SolarL2/Cosine/NewtonL2/NewtonCosine/LOOL2/SatL2/L2; uncertainty RMSEWithUncertainty + virtual ensembles + the deferred LOSS-06 prediction types (RMSEWithUncertainty/VirtEnsembles/TotalUncertainty); custom objective/metric **Rust trait** (Python callback deferred — D-09).
- **6.5 Text/embedding features (FEAT-01, FEAT-02):** text = tokenization + BoW + NaiveBayes + BM25; embedding = LDA + KNN calcers — all six oracle-locked.
- **6.6 Advanced features (FEAT-03, FEAT-04, FEAT-05, FEAT-06, MODEL-05):** monotone constraints, feature/per-object penalties, recursive feature selection (PredictionValuesChange/LossFunctionChange/ShapValues), grow policies Lossguide/Depthwise/Region (**non-symmetric trees** — D-10), advanced fstr (ShapInteractionValues, PredictionDiff, SAGE), and the deferred MODEL-03 LossFunctionChange importance (D-12).

**NOT in this phase:** GPU backends (Phase 7); all Python bindings incl. the LOSS-07 Python callback bridge and the sklearn/CatBoost-native API (Phase 8); any loss/metric NOT explicitly named in the success criteria — "etc." resolves to **deferred-to-v2, not silently in-scope** (D-06).

</domain>

<decisions>
## Implementation Decisions

### Decomposition & Sequencing
- **D-01: Split Phase 6 into 6 formal sub-phases (roadmap restructure).** Phase 6 spans 5 genuinely independent subsystems (14 reqs); a single mega-phase has too large a blast radius for the narrowest-first oracle philosophy. Each sub-phase gets its own discuss→plan→execute→verify cycle and its own oracle gate. **Immediate next step is `/gsd-phase`** to restructure ROADMAP.md (and the REQUIREMENTS traceability) into 6.1–6.6 before planning begins. Rejected: one phase with many waves (single mega-verification, large blast radius); hybrid checkpoint-gates (chosen split is cleaner given the subsystems are independent).
- **D-02: Additive sub-phase order (narrowest-first):** **6.1** regression-loss matrix (scalar, rides the existing loop — cheapest/lowest-risk first) → **6.2** multiclass/multilabel (the N-dim refactor) → **6.3** ranking losses+metrics → **6.4** score-fns + uncertainty + custom-obj → **6.5** text + embedding → **6.6** advanced features (monotone/penalties/selection/grow-policies/fstr). The N-dim refactor lands AFTER scalar losses are proven, so it lands on stable loss code; advanced features (incl. the non-symmetric tree engine) come last as the riskiest structural work.

### Multi-Dimensional Approx Refactor (6.2)
- **D-03: Refactor the core loop to N-dim approx, scalar = the dim=1 degenerate case.** The entire train loop is scalar today (`cb-compute::loss` der1/der2 take `approx: f64`; `cb-train::bootstrap`/leaf-estimation/`tree`/SHAP assume one dimension). Generalize so approx is always a vector (matching upstream `TVector<TVector<double>>`); scalar losses become dim=1. **Single code path, no parallel duplication** — aligns with the memory-efficiency-first goal and avoids divergence between two paths. Rejected: a separate parallel multi-dim path.
- **D-04: First wave of 6.2 is a NO-BEHAVIOR-CHANGE checkpoint.** Do the pure mechanical refactor (approx → length-1 vector everywhere) and **re-run ALL ~40 existing scalar oracles green at dim=1 BEFORE any softmax/multiclass math is written.** A subsequent divergence then provably comes from the refactor vs the new multiclass math — the two risks are isolated. This is a hard gate; multiclass loss code does not start until the refactor checkpoint is green.

### Oracle Depth & Completeness
- **D-05: Every NAMED loss/metric/feature is implemented and oracle-locked ≤1e-5.** The success-criteria lists ARE the contract for this phase. Full-parity mandate, bounded scope.
- **D-06: "etc." in the success criteria = deferred-to-v2, NOT silently in-scope.** Anything genuinely unlisted in the SC is captured as deferred, not implemented this phase. Prevents unbounded "etc." expansion. Rejected: a representative-subset-with-long-tail-deferral (leaves a parity gap); and enumerating the complete upstream registry beyond the named lists (unbounded).
- **D-07: Default oracle floor = Python-reachable per-stage parity; proactively add C++ instrumentation where helpful.** Default every sub-phase to the Phase-5 transcribe-then-self-oracle + per-stage Python-reachable parity (splits/leaves/staged approx + final prediction ≤1e-5). **PROACTIVELY build C++-instrumented harnesses for the trickier categories** that may lack a clean Python-reachable ground truth — randomized ranking losses (YetiRank/StochasticRank RNG streams), text/embedding calcer internals, recursive feature selection — rather than waiting for an escalation trigger. This deliberately goes BEYOND the Phase-5 "Python-reachable floor, escalate only" rule (the user chose proactive instrumentation). **Selective, not universal:** simple leaf-math regression losses (6.1) still ride the Python-reachable per-stage oracle — instrumentation is for categories where it materially strengthens the signal.
- **D-08 (first-class risk): C++-instrumented builds run under known disk pressure.** Root disk is ~100% full (see STATE.md Blockers + the `disk-pressure-and-full-suite-verification` and `catboost-instrumented-trainer-build` memories). The Phase-5 instrumented toolchain (sudo-free clang-18 + lld-18 + a built `_catboost`) **persists in `/tmp` and is REUSABLE** (see `instrumented-trainer-toolchain-persists` memory) — prefer incremental rebuild over a fresh full build. Every instrumented-harness task MUST treat build/link feasibility as a first-class risk: smallest instrumented unit, free disk before/after, and an explicit feasibility-probe that escalates rather than silently expanding scope.

### Cross-Phase Scope Tensions
- **D-09: LOSS-07 — Rust custom-objective/-metric trait now (6.4); Python callback bridge DEFERRED to Phase 8.** Build and oracle-test the Rust trait (user-supplied der1/der2 + eval) against a Rust-defined reference in 6.4. **Design the trait so the Phase-8 PyO3 callback wraps it cleanly, but do NOT build the Python bridge before PyO3 exists** — honors the roadmap's "Python strictly downstream of a stable Rust API" sequencing. **Captured as a Phase-8 dependency** (the Python callback half of LOSS-07). Rejected: pulling PyO3 forward into Phase 6 (violates sequencing); moving all of LOSS-07 to Phase 8 (the Rust trait belongs with the loss surface).
- **D-10: FEAT-06 — full non-symmetric tree parity (train + apply + serialize) in 6.6.** Implement Lossguide/Depthwise/Region growth producing true non-symmetric trees, the non-symmetric apply path, and `.cbm`/json round-trip — oracle-locked ≤1e-5. This is effectively a **second tree engine** and the largest single item in Phase 6; it likely warrants its own multi-wave structure within 6.6 and **touches `cb-model` (apply + serialization), not just `cb-train`.** The `TNonSymmetricTree*` FlatBuffers bindings already exist in `cb-model::generated` but have zero train/apply support today. Rejected: symmetric-equivalent-only (fails parity vs real Lossguide/Depthwise models).
- **D-11: FEAT-01/02 — all six text/embedding calcers oracle-locked in 6.5.** Text = tokenization + BoW + NaiveBayes + BM25; embedding = LDA + KNN — each producing upstream-matching encodings ≤1e-5. **Tokenizer parity (the upstream text-processing pipeline) is the first risk to nail** — the calcers depend on identical token streams. Consistent with the D-05 "every named feature" completeness bar.

### Claude's Discretion (parity-dictated — research reads upstream and reproduces)
- Exact sub-phase requirement-to-plan mapping and wave structure within each 6.x (planner decides).
- Exact der1/der2 formulas and prior/parameter defaults for every named loss/metric (transcribe from `error_functions.{h,cpp}`, `*_metrics.cpp`).
- The precise N-dim approx data layout and which kernels/host-reductions change (research reads `approx_calcer.cpp`'s approx dimensionality model; pinned by what keeps existing oracles green under D-04).
- Score-function math for the 5 new EScoreFunction variants (SolarL2/NewtonL2/NewtonCosine/LOOL2/SatL2) — extend the existing `cb-compute::EScoreFunction` enum (Cosine/L2 already shipped, 05-19 Task A).
- Whether 6.6's non-symmetric tree engine needs further internal sub-splitting (planner/research call).
- Which specific categories actually get C++ instrumentation under D-07 (decide per-category during research, gated by D-08 feasibility).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project & Roadmap
- `.planning/PROJECT.md` — core value, constraints (memory-efficiency first-class, `thiserror`/`anyhow`, latest crate versions), oracle strategy.
- `.planning/ROADMAP.md` § "Phase 6: Full Loss & Feature Parity" — goal + 5 success criteria this phase is judged against. **NOTE:** D-01 splits this into sub-phases 6.1–6.6 via `/gsd-phase` before planning.
- `.planning/REQUIREMENTS.md` — LOSS-02…05/07/08/09, FEAT-01…06, MODEL-05 requirement text + traceability; also the LOSS-06 uncertainty carve-out (Phase-4 D-10) and MODEL-03 LossFunctionChange carve-out (Phase-4 D-12) that complete here.
- `.planning/phases/05-ordered-boosting-ordered-ctr-categoricals-high-risk-parity-s/05-CONTEXT.md` — the Phase-5 oracle toolkit this phase reuses: transcribe-then-self-oracle (D-01-revision), Python-reachable fixtures, the user-approved C++-instrumentation deviation + feasibility constraint, `cb-core::sum_f64` reduction discipline.
- `.planning/phases/04-model-serialization-shap-rust-api-first-full-oracle-lock/04-CONTEXT.md` — `.cbm`/`model.json` framing + apply path (6.6 non-symmetric serialization extends this), SHAP/fstr infrastructure (MODEL-05 extends), the LOSS-06 (D-10) + LossFunctionChange (D-12) deferrals this phase closes.
- `.planning/phases/03-cpu-training-core-plain-boosting-oblivious-trees/03-CONTEXT.md` — generic `R: Runtime`/`F: Float` seam, host-ordered-reduce invariant, the scalar train loop the D-03 N-dim refactor generalizes.

### Vendored Reference & Oracle Source (catboost-master/, version 1.2.10)
- `catboost-master/catboost/libs/metrics/` — metric implementations (NDCG/DCG/MAP/MRR/ERR/PFound/PrecisionAt/RecallAt/QueryAUC for 6.3; regression metrics for 6.1).
- `catboost-master/catboost/private/libs/algo_helpers/error_functions.h`, `.../error_functions.cpp` — der1/der2 for the full loss matrix (RMSE/MAE/Quantile/Huber/Poisson/Tweedie/MAPE/MSLE/Lq/Expectile/LogCosh; MultiClass softmax; PairLogit/QueryRMSE/QuerySoftMax). Read per-loss.
- `catboost-master/catboost/private/libs/algo/approx_calcer.cpp` — the approx-dimensionality model the D-03 N-dim refactor must match (`TVector<TVector<double>>` approxes); multiclass + ranking approx updates.
- `catboost-master/catboost/private/libs/algo/score_calcers.h`, `.../score_calcers.cpp` — the 7 score functions for LOSS-09 (extends `cb-compute::EScoreFunction`).
- `catboost-master/catboost/private/libs/algo/yetirank_helpers.*`, `.../pairwise_*` — YetiRank/Pairwise/LambdaMart/StochasticRank randomized ranking (6.3; the D-07 C++-instrumentation candidates).
- `catboost-master/catboost/private/libs/text_features/`, `.../text_processing/` — tokenization + BoW/NaiveBayes/BM25 calcers (FEAT-01, 6.5).
- `catboost-master/catboost/private/libs/embedding_features/` — LDA + KNN calcers (FEAT-02, 6.5).
- `catboost-master/catboost/private/libs/algo/greedy_tensor_search.cpp`, `.../tensor_search_helpers.*`, and the non-symmetric tree growth — Lossguide/Depthwise/Region (FEAT-06, 6.6).
- `catboost-master/catboost/private/libs/options/` — `monotone_constraints`, `feature_penalties`, `per_object_feature_penalties`, `grow_policy` option parsing/defaults (FEAT-03/04/06).
- `catboost-master/catboost/private/libs/algo/features_select.cpp` (recursive feature selection by PredictionValuesChange/LossFunctionChange/ShapValues — FEAT-05).
- `catboost-master/catboost/libs/fstr/` — ShapInteractionValues, PredictionDiff, SAGE, LossFunctionChange (MODEL-05 + the MODEL-03 D-12 deferral, 6.6).
- `catboost-master/catboost/libs/model/` — `TNonSymmetricTree*` model structures + apply (6.6 apply/serialize). Rust bindings already at `crates/cb-model/src/generated/`.
- `catboost-master/catboost/libs/train_lib/` — uncertainty / virtual-ensemble training (RMSEWithUncertainty, LOSS-08; LOSS-06 uncertainty prediction types).

### Oracle Harness & Instrumentation (D-07, D-08)
- `crates/cb-oracle/` — fixture root + the transcribe-then-self-oracle generators + `compare_stage` ≤1e-5 API; new per-loss/per-feature fixtures land here, frozen-committed, generators run OFFLINE.
- `crates/cb-oracle/generator/` — existing standalone-C++/Python generator precedents (cityhash_oracle.cpp, ordered_oracle.cpp) the 6.x instrumented harnesses follow.
- Project memories: `catboost-instrumented-trainer-build` (sudo-free clang-18 + `_catboost` recipe), `instrumented-trainer-toolchain-persists` (toolchain in `/tmp`, reuse + incremental rebuild), `disk-pressure-and-full-suite-verification` (root disk ~100% full; verify per-crate).

### Existing Rust Code to Extend
- `crates/cb-compute/src/loss.rs`, `crates/cb-compute/src/runtime.rs` — the `Loss` enum + scalar der1/der2 the D-03 refactor generalizes; `EScoreFunction` (Cosine/L2) the LOSS-09 work extends.
- `crates/cb-compute/src/score.rs`, `crates/cb-train/src/tree.rs` — split-score plumbing (LOSS-09).
- `crates/cb-train/src/metrics.rs` — `EvalMetric` + `EvalMetricHistory` the ranking/regression metrics extend.
- `crates/cb-train/src/bootstrap.rs`, `crates/cb-train/src/boosting.rs` — single-dimension assumptions the D-03 refactor touches.
- `crates/cb-model/src/generated/` — `TNonSymmetricTree*` bindings (6.6); `ctr_data`/SHAP infra (MODEL-05).

### CubeCL constraint (carried forward)
- `AGENTS.md` (project root) + `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md` — any kernel work uses generics-float, lives in `cb-backend`; read the manual before writing kernel code; load `cubecl_error_guideline.md` on any build error before fixing. (Phase 6 is CPU-path; most loss/feature work is host orchestration.)

### Process / Project Rules
- `CLAUDE.md` (project root) — constraints, naming, mandatory source/test separation, latest-crate-versions rule.
- `.planning/codebase/CONVENTIONS.md`, `.planning/codebase/TESTING.md` — Rust lint/error/test conventions, source/test-separation rule.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `crates/cb-compute/src/loss.rs` — scalar `*_der1`/`*_der2` for Rmse/Logloss/CrossEntropy/Focal/Mae; the `Loss` enum (`cb-compute/src/runtime.rs:47`) is where new losses attach. The N-dim refactor (D-03) generalizes these signatures.
- `crates/cb-compute/src/runtime.rs` — `EScoreFunction` enum already has `Cosine` (default) + `L2` (05-19 Task A); LOSS-09 adds SolarL2/NewtonL2/NewtonCosine/LOOL2/SatL2.
- `crates/cb-train/src/metrics.rs` — `EvalMetric` + `EvalMetricHistory` (multi-set, `per_set: Vec<Vec<f64>>`) extend for ranking/regression metrics.
- `crates/cb-model/src/generated/` — `TNonSymmetricTreeStepNode` and related FlatBuffers bindings already committed (6.6 train/apply/serialize wires into these).
- `crates/cb-oracle/` — `compare_stage` ≤1e-5 API + frozen-fixture convention + offline generators (transcribe-then-self-oracle, D-07).

### Established Patterns
- **Everything is scalar-`approx` today** (`approx: f64` in loss math; `bootstrap.rs` comments explicitly "single-dimension"). D-03/D-04 generalize this; the D-04 no-behavior-change checkpoint protects the existing locks.
- **All parity-critical float summation routes through `cb-core::sum_f64`/`sum_f32_in_f64`** (Phase-2 D-07); D-08 CI-grep ban applies to all new loss/metric/feature accumulation.
- **Per-stage oracle (INFRA-04):** borders/splits/leaf-values/staged-approx + final prediction, all ≤1e-5 — the D-07 default floor.
- **Source/test separation mandatory** (no inline `#[cfg(test)]`); `thiserror` in libraries, `anyhow` structurally banned (CI grep).

### Integration Points
- New losses attach at the `Loss` enum + der1/der2 dispatch; the train loop's leaf-estimation/score path consumes them.
- Non-symmetric trees (6.6) connect at BOTH `cb-train` (growth) AND `cb-model` (apply + `.cbm`/json serialize) — a wider integration than prior phases.
- Text/embedding (6.5) connect at the `Pool` text/embedding columns (DATA-01) → calcer → quantized features feeding the existing tree path.
- Custom-objective trait (6.4) plugs into the same der1/der2 seam; the Phase-8 PyO3 callback (D-09) will wrap the trait.

</code_context>

<specifics>
## Specific Ideas

- The 6.2 N-dim refactor must mirror upstream's `TVector<TVector<double>>` approxes layout (research confirms exact shape).
- 6.5 tokenizer parity is the named first risk — get the upstream token stream bit-identical before scoring BoW/NaiveBayes/BM25.
- 6.6 non-symmetric trees are "effectively a second tree engine" — expect its own multi-wave structure and `cb-model` blast radius.
- Reuse the persisted `/tmp` clang-18 + `_catboost` instrumented toolchain (incremental rebuild) rather than fresh builds, under the ~100%-full-disk constraint.

</specifics>

<deferred>
## Deferred Ideas

- **LOSS-07 Python callback bridge** → Phase 8 (Python bindings). The Rust custom-objective trait ships in 6.4; the PyO3 callback that wraps it is a Phase-8 dependency (D-09). Captured so Phase 8 planning knows to close the second half of LOSS-07.
- **Any loss/metric NOT explicitly named in the Phase-6 success criteria** ("etc." in the SC lists) → v2 (D-06). Not implemented this phase; revisit against the full upstream registry post-parity.

</deferred>

---

*Phase: 6-full-loss-feature-parity*
*Context gathered: 2026-06-15*
