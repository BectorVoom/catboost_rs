# catboost-rs — Unimplemented-Feature Gap Analysis & Next-Feature Recommendation

**Research date:** 2026-07-18
**Branch:** `feat/18-fstr03-partial-dependence`
**Author:** Research agent (spec-and-TDD workflow)
**Scope:** Read-only gap analysis. No production code authored.

Every material claim carries an evidence tag: `[VERIFIED: CODEGRAPH …]`,
`[VERIFIED: LOCAL <path>]`, `[INFERRED: …]`, `[UNVERIFIED: …]`.

---

## 0. Executive Summary

catboost-rs is a **mature, near-complete** Rust rewrite. Milestones v1.0 (Core
Parity, Phases 1–8) and v1.1 (GPU Performance, Phases 10–14) are **shipped**;
v1.2 (Parity Completion & Release Readiness, Phases 15–22) is **in progress**
with Phases 15, 16, 21, 21.5 complete and Phase 23 (CTR .cbm load/save) landed
`[VERIFIED: LOCAL git log; .planning ROADMAP@a82289c]`.

The remaining unimplemented user-facing surface is small and well-bounded:

| Phase | Feature | Status |
|-------|---------|--------|
| 17 | ONNX export | **DONE** (float-only oblivious) |
| 17 | CoreML export (EXPORT-02) + ORT oracle (EXPORT-03) | **MISSING** |
| 18 | FSTR-01 Interaction (+CTR) | **DONE** |
| 18 | FSTR-03 Partial dependence | **DONE** |
| 18 | FSTR-02 LossFunctionChange (multi-loss + CTR) | **PARTIAL** (Logloss/numeric only) |
| 19 | GPU inference evaluator (GINF-01) | **MISSING** |
| 20 | CV / tuning / snapshot / **calc_metrics** (ORCH-01..04) | **MISSING** (no `cb-orchestrate` crate) |
| 22 | Adoption / DX capstone (benchmark, wheels, docs) | **MISSING** (last, exercises others) |
| 24 | CTR split-search correctness (ORD-06/ORD-07) | **IN PROGRESS** (bug-fix, draft SPEC) |

**Recommended next feature: ORCH-04 — a standalone `eval_metrics` / `calc_metrics`
surface** (compute metrics on precomputed predictions, staged, independent of a
live training run). It is the highest-value **genuinely-missing** feature that is
simultaneously self-contained, failure-isolated per metric, and backed by the
cleanest possible ≤10⁻⁵ CPU oracle (`catboost.utils.eval_metric` on fixed
predictions — no training, no nondeterminism, no SHAP/CTR dependency).

Runner-ups: (1) FSTR-02 numeric multi-loss generalization; (2) CoreML export
(EXPORT-02); (3) CV fold semantics (ORCH-01).

---

## 1. Inventory of the Rust Implementation

Workspace = 9 crates under `crates/` `[VERIFIED: LOCAL crates/; Cargo.toml
members=["crates/*"]]`:

| Crate | Responsibility (verified) |
|-------|---------------------------|
| `cb-core` | RNG, normal dist, deterministic reduction (`sum_f64`), error base `[VERIFIED: LOCAL crates/cb-core/src]` |
| `cb-data` | Pool, quantization, borders, cat-hash, NaN modes, weights, **text** (tokenizer/dictionary/digitizer/bigram) ingestion (arrow/polars/owned) `[VERIFIED: LOCAL crates/cb-data/src]` |
| `cb-compute` | Loss/derivatives, leaf estimation, histogram, score functions, pairwise scoring, ranking derivatives, **text/embedding calcers** (BM25/BoW/NaiveBayes/LDA/**online-HNSW KNN**), runtime seam `[VERIFIED: LOCAL crates/cb-compute/src]` |
| `cb-backend` | CubeCL GPU kernels + cpu_runtime: histograms (pointwise/pairwise), grow loop, non-sym grow, region, CTR device, bootstrap, MVS, Cholesky, Langevin, multi-Newton, scan/sort/reduce/scatter `[VERIFIED: LOCAL crates/cb-backend/src/kernels]` |
| `cb-train` | Boosting (plain + ordered), tree growers (oblivious/Depthwise/Lossguide/Region), CTR (online/final/bake/calc), candidates, bootstrap, folds, permutation, overfit/OD, autolr, metrics, ranking metrics, yetirank, feature_selection, projection `[VERIFIED: LOCAL crates/cb-train/src]` |
| `cb-model` | Canonical `Model`, apply/predict (incl. virtual ensembles + multiclass + CTR), `.cbm`/`.json` (de)serialize, **SHAP family**, **fstr** (PVC/LossChange/Interaction), **partial_dependence**, **ONNX export** `[VERIFIED: LOCAL crates/cb-model/src/lib.rs]` |
| `cb-oracle` | Fixture harness + `compare` (≤1e-5) + ~60 fixture families + Python/C++ generators `[VERIFIED: LOCAL crates/cb-oracle/fixtures]` |
| `catboost-rs` | Published Builder-pattern facade (`CatBoostBuilder`, `Model`, `CatBoostError`) `[VERIFIED: LOCAL crates/catboost-rs/src/lib.rs]` |
| `catboost-rs-py` | PyO3 bindings: sklearn-compatible classifier/regressor/ranker + estimator/params/pool/ingest `[VERIFIED: LOCAL crates/catboost-rs-py/src]` |

### Capabilities confirmed implemented (DONE)

- **Model load/save**: `load_cbm`/`save_cbm`/`decode_cbm`, `load_json`/`save_json`,
  incl. **CTR reconstruction** for upstream categorical `.cbm`
  `[VERIFIED: LOCAL crates/cb-model/src/lib.rs pub use cbm/json]`.
- **Apply/predict**: `predict_raw`, `predict_raw_cat`, `predict_raw_multi`,
  `apply_virtual_ensembles`, multiclass, prediction types
  `[VERIFIED: CODEGRAPH apply.rs; predict.rs]`.
- **Training**: plain + ordered boosting; oblivious + non-symmetric
  (Depthwise/Lossguide) + Region growers; one-hot; simple/tensor/combination
  CTR; monotone constraints; feature penalties; feature selection
  `[VERIFIED: LOCAL crates/cb-train + tests/*_oracle_test.rs]`.
- **Losses/metrics** (oracle-locked): RMSE, MAE, Logloss, CrossEntropy, Quantile,
  MultiQuantile, Expectile, Huber, LogCosh, MAPE, MSLE, Lq, Poisson, Tweedie,
  Focal, MultiClass/OneVsAll, MultiLogloss, MultiCrossEntropy, RMSEWithUncertainty;
  ranking: PairLogit(+Pairwise), QueryRMSE, QuerySoftMax, YetiRank(+Pairwise),
  LambdaMart, StochasticRank; ranking metrics `[VERIFIED: LOCAL crates/cb-train/tests
  loss_oracle/wave1-3/multiclass/ranking_*_oracle_test.rs; crates/cb-oracle/fixtures]`.
- **fstr**: `prediction_values_change` (+CTR), `interaction` (+CTR, FSTR-01 DONE),
  `loss_function_change` (numeric Logloss only — see §4), SHAP:
  `shap_values`, `shap_interaction_values`, `prediction_diff`, `sage_values`,
  **partial_dependence** (FSTR-03 DONE) `[VERIFIED: LOCAL crates/cb-model/src/lib.rs;
  fstr.rs; shap.rs; partial_dependence.rs]`.
- **GPU training**: full device-resident boosting inner loop (CubeCL), Kaggle-CUDA
  signed ε=1e-4 `[VERIFIED: LOCAL ROADMAP Phases 10–14 Complete]`.
- **ONNX export**: `export_onnx` for float-only oblivious identity-scale models,
  typed `OnnxExportError` guards `[VERIFIED: CODEGRAPH export/onnx.rs]`.
- **Text/embedding + online-HNSW KNN** estimated features, bit-for-bit parity
  `[VERIFIED: LOCAL ROADMAP Phase 16 Complete; crates/cb-compute/src/hnsw.rs]`.
- **Python bindings**: sklearn classifier/regressor/ranker + native Pool
  `[VERIFIED: LOCAL crates/catboost-rs-py/src]`.

---

## 2. Inventory of Planning Phases

Authoritative source is the **git-recovered ROADMAP** (`git show
a82289c:.planning/ROADMAP.md`; the file is NOT in the working tree — the
`.planning/codebase/*.md` docs ARE stale, describing the old upstream FFI
wrapper, NOT the current Rust rewrite) `[VERIFIED: LOCAL git show
a82289c:.planning/ROADMAP.md; .planning/codebase/CONCERNS.md "near-zero Rust layer"]`.

| Phase | Title | Status |
|-------|-------|--------|
| 1–8 | v1.0 Core Parity | ✅ Complete 2026-06-28 |
| 10–14 | v1.1 GPU Performance | ✅ Complete 2026-07-05 |
| 15 | Debt Discharge & CUDA Oracle Re-est. | ✅ Complete |
| 16 | Online-HNSW KNN parity (FEAT-07) | ✅ Complete 2026-07-12 |
| **17** | **Model Export — ONNX + CoreML** | 🟡 ONNX done; CoreML MISSING |
| **18** | **Extended Feature Importance** | 🟡 FSTR-01/03 done; FSTR-02 PARTIAL |
| **19** | **GPU Inference Evaluator (GINF-01)** | ❌ Not started |
| **20** | **Orchestration — CV/tuning/snapshot/calc_metrics** | ❌ Not started |
| 21 | CPU Split-Finding Histogram Rewrite | ✅ Complete |
| 21.5 | CPU Parallel-Scaling (fused hist) | ✅ Complete |
| **22** | **Adoption / DX Capstone** | ❌ Not started (last) |
| 23 | CTR model loading (out-of-roadmap) | ✅ Complete (`.cbm` CTR load+save) |
| 24 | CTR split-search correctness (ORD-06/07) | 🚧 Draft SPEC, bug-fix in progress |

Local phase-dir evidence `[VERIFIED: LOCAL .planning/phases/*/*/PLAN.md
"Execution status … COMPLETE"]`:
- `17-model-export/onnx-export` — SPEC/PLAN/research/PLAN-CHECK present; ONNX shipped
  (commit `c981e33`).
- `18-…/fstr-01-interaction-ctr` — "✅ COMPLETE, AT-FIC02d/AT-FIC03d GREEN".
- `18-…/fstr-03-partial-dependence` + `fstr-03-facade-python` — "✅ COMPLETE".
- **No `fstr-02` phase directory exists** → FSTR-02 v1.2 is UN-SPECCED
  `[VERIFIED: LOCAL ls .planning/phases/18-extended-feature-importance]`.
- `23-ctr-model-loading/{cbm-ctr-load,cbm-ctr-save}` — landed (commits
  `9015b22`, `c5ff842`).
- `24-ctr-split-search-correctness/{combination-ctr-level-gating,
  simple-ctr-cat-feature-weight}` — draft SPECs, bug discovered as FSTR-01
  side-effect; training produces a different tree structure than
  catboost 1.2.10 at the root split `[VERIFIED: LOCAL 24…/combination-ctr-level-gating/SPEC.md §1]`.

---

## 3. CatBoost Feature Surface → catboost-rs Status (the Gap Table)

Legend: DONE / PARTIAL / MISSING / PLANNED-NOT-DONE.

| CatBoost feature category | catboost-rs status | Evidence |
|---------------------------|--------------------|----------|
| `.cbm` / json model load+save | DONE | `cb-model` cbm.rs/json.rs `[VERIFIED: CODEGRAPH]` |
| CTR categorical `.cbm` load+predict | DONE | Phase 23 `[VERIFIED: LOCAL]` |
| Predict (raw/proba/class, staged, multiclass, virtual ensembles) | DONE | apply.rs/predict.rs `[VERIFIED: CODEGRAPH]` |
| Regression losses (RMSE/MAE/Quantile/Huber/…/Tweedie/Poisson) | DONE | oracle fixtures `[VERIFIED: LOCAL]` |
| Classification (Logloss/CrossEntropy/Focal) | DONE | oracle fixtures `[VERIFIED: LOCAL]` |
| Multiclass / multilabel / MultiQuantile / N-dim | DONE | multiclass/multilabel/multiquantile oracle tests `[VERIFIED: LOCAL]` |
| Ranking (PairLogit/QueryRMSE/YetiRank/LambdaMart/StochasticRank) | DONE | ranking_corpus fixtures `[VERIFIED: LOCAL]` |
| Uncertainty / virtual ensembles (RMSEWithUncertainty) | DONE | uncertainty_predict fixtures `[VERIFIED: LOCAL]` |
| Custom objective/metric | DONE | custom_objective_oracle_test `[VERIFIED: LOCAL]` |
| Text + embedding features (BM25/BoW/NaiveBayes/LDA/KNN-HNSW) | DONE | cb-compute calcers + oracle `[VERIFIED: LOCAL]` |
| Ordered boosting + ordered/tensor/combination CTR | DONE (train) | ordered_*/tensor_ctr oracle `[VERIFIED: LOCAL]` — **but** combination-CTR candidate gating has a **known bug** (Phase 24) `[VERIFIED: LOCAL 24…/SPEC.md]` |
| One-hot categorical | DONE | one_hot_oracle_test `[VERIFIED: LOCAL]` |
| Monotone constraints / feature penalties | DONE | monotone/penalty oracle `[VERIFIED: LOCAL]` |
| Non-symmetric trees (Depthwise/Lossguide/Region) | DONE | non_symmetric/region oracle `[VERIFIED: LOCAL]` |
| Recursive feature selection | DONE | feature_selection_oracle_test `[VERIFIED: LOCAL]` |
| CPU histogram split search + rayon scaling | DONE | Phase 21/21.5 `[VERIFIED: LOCAL]` |
| GPU **training** (CubeCL, all families) | DONE | Phase 10–14 `[VERIFIED: LOCAL]` |
| fstr: PredictionValuesChange (+CTR) | DONE | fstr.rs `prediction_values_change_with_data` `[VERIFIED: CODEGRAPH]` |
| fstr: Interaction (+CTR) | DONE | fstr.rs `interaction` FSTR-01 `[VERIFIED: LOCAL]` |
| fstr: ShapValues / ShapInteraction / PredictionDiff / SAGE | DONE (float) | shap.rs `[VERIFIED: CODEGRAPH]` |
| fstr: Partial dependence | DONE | partial_dependence.rs FSTR-03 `[VERIFIED: LOCAL]` |
| **fstr: LossFunctionChange (multi-loss + CTR)** | **PARTIAL** | Logloss+float only, SHAP-approx (§4) `[VERIFIED: CODEGRAPH fstr.rs:788]` |
| **ONNX export** | DONE | export/onnx.rs `[VERIFIED: CODEGRAPH]` |
| **CoreML export (EXPORT-02)** | **MISSING** | only a `// future coreml.rs` comment `[VERIFIED: LOCAL export/mod.rs:3]` |
| **Export oracle vs ONNX Runtime (EXPORT-03)** | **PARTIAL/UNVERIFIED** | ONNX exportable+guarded; ORT round-trip oracle presence not confirmed `[UNVERIFIED]` |
| **GPU inference evaluator (GINF-01)** | **MISSING** | no `cb-infer-gpu` crate `[VERIFIED: LOCAL ls crates/]` |
| **Cross-validation `cv()` (ORCH-01)** | **MISSING** | no cv symbol/crate `[VERIFIED: LOCAL grep]` |
| **grid/randomized search (ORCH-02)** | **MISSING** | none `[VERIFIED: LOCAL grep]` |
| **Snapshot/resume `BoostingCheckpoint` (ORCH-03)** | **MISSING** | none `[VERIFIED: LOCAL grep]` |
| **Standalone `eval_metrics`/`calc_metrics` (ORCH-04)** | **MISSING** | metric math exists ONLY as training-coupled `EvalMetric` `[VERIFIED: CODEGRAPH cb-train/metrics.rs; grep no standalone symbol]` |
| Benchmarks / PyPI wheels / docs (DX-01..04) | MISSING | Phase 22, last `[VERIFIED: LOCAL ROADMAP]` |

---

## 4. Deep-Dive: current `loss_function_change` (why FSTR-02 is PARTIAL)

`cb_model::loss_function_change(model, cols, labels, n_features)`
`[VERIFIED: CODEGRAPH crates/cb-model/src/fstr.rs:788-828]`:

1. **Binary-Logloss-hardcoded.** Final error is `logloss_final_error` (sigmoid +
   binary cross-entropy). No RMSE/MAE/multiclass/ranking path. Upstream
   `LossFunctionChange` uses the model's actual metric's `GetFinalError`
   (`loss_change_fstr.cpp`) `[VERIFIED: CODEGRAPH fstr.rs:832-851]`.
2. **SHAP-approximation accounting**: `approx_f[obj] = approx[obj] −
   shap[obj][feature]`, then `finalError(without f) − finalError(full)`. It relies
   on `predict_raw` + `shap_values` `[VERIFIED: CODEGRAPH fstr.rs:800-828]`.
3. **Float-only → no CTR/categorical model support.** `predict_raw` and
   `shap_values` consume float SoA columns only; `shap_values` has **no CTR-split
   handling** (`shap_values_fixed` walks numeric columns; no `ModelSplit::Ctr`
   branch) `[VERIFIED: CODEGRAPH shap.rs:524-534, 820-866]`. A CTR `.cbm` model
   cannot even be applied through this path.
4. Facade already routes it: `Model::feature_importance_with_data` →
   `cb_model::loss_function_change`; the no-data `feature_importance` returns an
   empty vector for LossFunctionChange `[VERIFIED: CODEGRAPH catboost-rs/src/model.rs:139-180]`.
5. Numeric coverage IS oracle-locked (Phase 6.6 MODEL-03): fixtures
   `fstr_loss_change/{oblivious,non_symmetric}_loss_function_change.npy`,
   test `fstr_oracle_test.rs:178` `[VERIFIED: LOCAL]`.

**FSTR-02 v1.2 remaining work** = (a) generalize final-error beyond Logloss to the
model loss; (b) support **CTR/categorical models**, which **requires CTR-aware SHAP
+ apply** (not yet implemented). (b) is a **hidden, substantial dependency** — this
is the primary reason FSTR-02 is a runner-up, not the top pick.

---

## 5. Next-Feature Selection (against the three criteria)

Criteria: (a) high-value for parity, (b) self-contained with failure-isolated
behaviors, (c) clear ≤10⁻⁵ oracle path vs C++ CatBoost.

| Candidate | (a) Value | (b) Self-contained | (c) Oracle | Verdict |
|-----------|-----------|--------------------|-----------|---------|
| **ORCH-04 `eval_metrics`/`calc_metrics`** | HIGH — ubiquitous public API, entirely absent standalone | HIGH — pure fn over (labels, approx, metric); one metric = one behavior; staged vs final isolated | **BEST** — `catboost.utils.eval_metric` on FIXED preds; no training nondeterminism; `eval_metrics` fixture dir exists | **RECOMMENDED** |
| FSTR-02 LossFunctionChange | HIGH — completes Phase 18 | MEDIUM — numeric OK, **CTR needs CTR-aware SHAP** (hidden dep) | GOOD — `get_feature_importance(type='LossFunctionChange')`, ≤1e-5 | Runner-up 1 |
| CoreML export (EXPORT-02) | MEDIUM — completes Phase 17 | HIGH — read-only | **WEAK** — needs Apple runtime; export-specific float32 tol, NOT 1e-5 | Runner-up 2 |
| CV `cv()` (ORCH-01) | HIGH | MEDIUM — trains inside folds (determinism coupling) | GOOD — fold-assignment per seed + cv results | Runner-up 3 |
| GPU inference (GINF-01) | MEDIUM | LOW — new crate, CUDA-only | POOR (this host) — needs Kaggle CUDA, ε=1e-4 | Rejected |
| Snapshot/resume (ORCH-03) | MEDIUM | MEDIUM — deep trainer-state coupling | SELF-oracle (resume==straight), not vs C++ | Rejected |
| grid/random search (ORCH-02) | MEDIUM | LOW — hard-depends on ORCH-01 | Indirect | Rejected (not a leaf) |

### Recommended next feature (one line)

> **ORCH-04 — a standalone `eval_metrics` / `calc_metrics` surface**: compute
> CatBoost metrics (staged and final) on caller-supplied predictions + labels
> (+ optional weights/group-ids), oracle-locked ≤10⁻⁵ against
> `catboost.utils.eval_metric`, exposed through the facade and Python.

### Why ORCH-04 over the runner-ups

- **Zero hidden dependencies.** The metric arithmetic already exists and is
  already oracle-tested inside the training loop (`cb_train::metrics::EvalMetric`
  covering RMSE, Logloss, AUC, plus `ranking_metrics.rs`)
  `[VERIFIED: CODEGRAPH cb-train/src/metrics.rs:64-190; ranking_metrics.rs]`. The
  feature is a thin **standalone dispatch + staged-accumulation** wrapper — it does
  NOT drag in CTR-aware SHAP (FSTR-02's trap) or a GPU/Apple runtime.
- **Pristine oracle.** Metrics on FIXED predictions are the one surface with no
  quantization/training nondeterminism (the documented cross-cutting hazard —
  "catboost quantization is run-to-run nondeterministic so CTR fixtures are
  frozen") `[VERIFIED: LOCAL MEMORY ctr-model-loading]`. A fixture is
  `label + approx + expected_metric_value`; `.venv` catboost 1.2.10 is already the
  project oracle (`fstr_loss_change/gen_fixtures.py` uses it)
  `[VERIFIED: LOCAL crates/cb-oracle/fixtures/fstr_loss_change/gen_fixtures.py]`.
- **Failure-isolated behaviors** for TDD: one behavior per metric family; final
  vs staged (per-tree-prefix) evaluation; single-dim vs multi-dim; unit vs
  weighted vs grouped. Each is independently red/green-able.
- **Naturally opens Phase 20** at its dependency-free leaf: ORCH-04 has NO
  dependency on ORCH-01/02/03 (ORCH-02 depends on ORCH-01; ORCH-04 is standalone)
  `[VERIFIED: LOCAL ROADMAP Phase 20 "Internal: ORCH-02 hard-depends on ORCH-01"]`.

### Runner-ups (explicit)

1. **FSTR-02 numeric multi-loss generalization** — completes Phase 18; single-crate
   (`cb-model/fstr.rs`); clean numeric oracle. **Defer the CTR sub-case** (needs
   CTR-aware SHAP first) or spec it as a separate slice. Choose this if closing
   Phase 18 outranks opening Phase 20.
2. **CoreML export (EXPORT-02)** — completes Phase 17; read-only, self-contained;
   but the oracle is weak on Linux (no Apple runtime) and inherently not ≤10⁻⁵.
3. **CV fold semantics (ORCH-01)** — high value and the true Phase-20 anchor, but
   couples to training determinism and has a larger oracle surface (fold assignment
   + per-fold training + cv-results table).

---

## 6. Impact & Oracle Path for the Recommended Feature (ORCH-04)

### 6.1 Rust files/symbols that would change or be created

- **Reuse (no rewrite):** `cb_train::metrics::EvalMetric` + its `final_error`/
  per-metric math, `cb_train::ranking_metrics` `[VERIFIED: CODEGRAPH
  cb-train/src/metrics.rs; ranking_metrics.rs]`. These are currently **coupled to
  the per-iteration eval path** (`boosting.rs` eval_metric) — the new surface must
  expose them **standalone** (labels+approx in, metric value out) without a live
  `Booster`.
- **New surface (crate-placement is a PLAN decision):** ROADMAP prescribes a new
  `cb-orchestrate` crate for Phase 20 `[VERIFIED: LOCAL ROADMAP Phase 20 Context]`.
  A lighter alternative is a standalone `eval_metrics` module in `cb-train` (or
  `cb-model`) surfaced through the facade. **Flag for the planner** (mirrors the
  Phase-17 `cb-export`-vs-submodule crate-placement decision).
- **Facade:** add `eval_metrics`/`calc_metrics` to `catboost-rs` (`Model` or a free
  fn) mirroring the `feature_importance_with_data` precedent
  `[VERIFIED: CODEGRAPH catboost-rs/src/model.rs:139-180]`.
- **Python:** expose on the estimators / a `utils` module, mirroring the
  `partial_dependence` Python surfacing precedent `[VERIFIED: LOCAL MEMORY
  fstr03-partial-dependence-plan "Python partial_dependence"]`.

### 6.2 C++ reference implementation

- Metric math: `catboost-master/catboost/libs/metrics/metric.cpp` (each
  `TMetric::Eval` + `GetFinalError`) `[VERIFIED: LOCAL CLAUDE.md metrics.cpp ref;
  cb-train/src/metrics.rs docstrings cite metric.cpp]`.
- Public standalone API: `catboost.utils.eval_metric(label, approx, metric,
  weight=, group_id=, thread_count=)` and `CatBoost.eval_metrics(data, metrics,
  ntree_start/end/eval_period)` (staged) `[INFERRED: upstream python-package
  core.py; confirm exact signature at plan time]`.

### 6.3 Existing oracle-test patterns to follow

- Harness: `cb_oracle::{fixture, compare}` with ≤1e-5 tolerance
  `[VERIFIED: LOCAL crates/cb-oracle/src/compare.rs, fixture.rs]`.
- Directly analogous existing tests: `eval_metrics_oracle_test.rs`,
  `ranking_metrics_oracle_test.rs`, `msle_metric_oracle_test.rs`
  `[VERIFIED: LOCAL crates/cb-train/tests]` and the `eval_metrics/{logloss,rmse}`
  fixture dir `[VERIFIED: LOCAL crates/cb-oracle/fixtures/eval_metrics]` — a
  standalone `calc_metrics` fixture extends this exact shape.
- Fixture generator pattern: pinned-seed Python writing `.npy` + `config.json`
  straight from catboost (see `fstr_loss_change/gen_fixtures.py`)
  `[VERIFIED: LOCAL]`.

### 6.4 Oracle tooling availability (verified on this host)

- `uv` is installed (`/home/user/.cargo/bin/uv`) `[VERIFIED: LOCAL which uv]`.
- Established recipe: `uv venv --python 3.12 && uv pip install catboost==1.2.10
  'numpy<2'` (system python is 3.14; catboost 1.2.10 has no 3.14 wheel)
  `[VERIFIED: LOCAL MEMORY fstr03-partial-dependence-plan "Oracle unblocked via uv"]`.

---

## 7. Dependencies

- **No new external crate needed** for ORCH-04 — all metric math is in-tree; only
  workspace-internal wiring (facade → metrics module). Follows the "always use
  existing capability first" constraint `[VERIFIED: LOCAL CLAUDE.md Dependencies]`.
- If a new `cb-orchestrate` crate is chosen, keep `default-features=false` /
  no unconditional `cpu` feature (feature-unification landmine)
  `[VERIFIED: LOCAL ROADMAP milestone-wide context; Cargo.toml cubecl note]`.

---

## 8. Common Pitfalls & Risks (for the recommended feature)

| Risk | Trigger | Consequence | Prevention/Verification |
|------|---------|-------------|-------------------------|
| Metric coupling to `Booster` | Reusing `EvalMetric` that assumes a live training context | Can't call standalone | Extract a pure `(approx, target, weight, group) -> f64` seam; unit-test without a `fit` |
| `GetFinalError` semantics | Additive metrics divide by weight-sum; some are non-additive (AUC) | Wrong value | Mirror each metric's `GetFinalError`; oracle per-metric ≤1e-5 |
| Staged evaluation | `eval_metrics` with `eval_period`/`ntree_start/end` computes per-prefix | Missing/incorrect staged column | Reuse `predict_raw`'s prefix capability; fixture with staged expected matrix |
| Summation order (D-08) | Metric reductions | Parity drift | Route sums through `cb_core::sum_f64` (project rule) `[VERIFIED: LOCAL CLAUDE.md; fstr.rs uses sum_f64]` |
| Lint gate is CLIPPY, not build | `cargo build` passes but `unwrap/expect/panic/indexing_slicing` denied | CI red | Gate new code with `cargo clippy -p <crate> --all-targets`; workspace is broadly red in untouched files → scope with `--lib --no-deps` `[VERIFIED: LOCAL MEMORY fstr03-plan gotchas]` |
| Test mount | Unit tests need `#[cfg(test)] #[path="X_test.rs"] mod tests;` in the prod `.rs` | `cargo test` silently runs 0 tests | Follow `ctr_data.rs:58-61` mount pattern `[VERIFIED: LOCAL MEMORY]` |
| Metric coverage gap | Upstream supports many metrics not yet in Rust | Incomplete parity | SPEC scopes to the implemented metric set + documents deferred metrics |
| Python can't link locally | catboost-rs-py needs python3.12, system is 3.14 | Python tests unrunnable in-env | `cargo check` compile-verify; run via uv 3.12 venv `[VERIFIED: LOCAL MEMORY]` |

Cross-cutting (not specific to ORCH-04):
- **CTR training-structure bug (Phase 24, ORD-06/07)** — do NOT rely on
  Rust-trained CTR models as fixtures; load upstream `.cbm` (Phase 23) instead
  `[VERIFIED: LOCAL 24…/SPEC.md; MEMORY ctr-model-loading "fixtures frozen"]`.
- **Stale `.planning/codebase/*.md`** — they describe the old FFI wrapper; ignore
  for architecture truth, prefer CodeGraph/current source `[VERIFIED: LOCAL
  CONCERNS.md]`.

---

## 9. Testing & Verification Strategy (recommended feature)

- **Unit:** one `#[test]` per metric family (final value on hand-computed small
  vectors), plus staged accumulation and weighted/grouped variants — in a sibling
  `*_test.rs` mounted per the project pattern.
- **Oracle (≤10⁻⁵):** new `calc_metrics` fixtures (extend `eval_metrics/`):
  `label.npy` + `approx.npy` (+ `weight.npy`/`group_id.npy`) + `expected.json` from
  `catboost.utils.eval_metric` / `model.eval_metrics(...)`; compared via
  `cb_oracle::compare`.
- **Facade/Python:** compile-checked facade test + Python parity via the uv 3.12
  venv (mirroring the FSTR-03 facade+Python precedent).
- **Commands (verified patterns):**
  - `cargo test -p cb-train --test eval_metrics_oracle_test` (existing analogue)
    `[VERIFIED: LOCAL]`
  - `cargo clippy -p <crate> --lib --no-deps` (lint gate) `[VERIFIED: LOCAL MEMORY]`
  - Oracle env: `uv venv --python 3.12 && uv pip install catboost==1.2.10 'numpy<2'`
    `[VERIFIED: LOCAL MEMORY]`

---

## 10. Open Questions (materially affect planning)

1. **Crate placement** — new `cb-orchestrate` (ROADMAP prescription) vs a lighter
   standalone `eval_metrics` module in `cb-train`/facade. Affects Cargo wiring, not
   correctness. **Planner/user decision.** `[VERIFIED: LOCAL ROADMAP Phase 20 Context]`
2. **Metric scope** — which upstream metrics to cover in the first slice
   (implemented set vs a documented subset). `[INFERRED]`
3. **EXPORT-03 status** — is there already an ONNX-Runtime round-trip oracle, or
   only structural export? Not confirmed here. `[UNVERIFIED]` — relevant only if the
   planner instead picks CoreML/EXPORT.
4. **Alternative pick** — if closing Phase 18 is prioritized over opening Phase 20,
   switch to FSTR-02 (numeric multi-loss first; CTR sub-case deferred pending
   CTR-aware SHAP). `[INFERRED]`

---

## 11. Sources

- **CodeGraph:** `codegraph_explore` over model/apply/predict/CTR/fstr/SHAP/
  partial-dependence/ONNX (82 symbols, 5 files); `fstr.rs:788` `loss_function_change`;
  `shap.rs:524/820` float-only SHAP; `catboost-rs/src/model.rs:139-180` facade
  dispatch; `cb-train/src/metrics.rs:64-190` `EvalMetric`.
- **Local files:** `Cargo.toml`; `crates/*/src` + `crates/*/tests` listings;
  `crates/cb-oracle/fixtures` (~60 families incl. `eval_metrics`, `fstr_loss_change`,
  `fstr_ctr`, `ctr_load`, `partial_dependence`); `crates/cb-model/src/lib.rs`;
  `crates/cb-model/src/export/{mod,onnx}.rs`; `crates/catboost-rs/src/lib.rs`;
  `.planning/phases/{17,18,23,24}/…/{SPEC,PLAN}.md`; `.planning/codebase/*.md` (stale).
- **Git history:** `git show a82289c:.planning/ROADMAP.md` (274 lines, full v1.2
  roadmap — file deleted from working tree); `git log --oneline` (commits
  `c981e33` ONNX, `9015b22`/`c5ff842` CTR load/save, `620dd46`/`41bd3fb` FSTR-03).
- **Project memory:** `fstr03-partial-dependence-plan.md` (v1.2 gap, gotchas, uv
  oracle recipe); `ctr-model-loading.md` (frozen CTR fixtures, quantization
  nondeterminism).
- **Tooling:** `which uv` → `/home/user/.cargo/bin/uv` (verified present).
- **Context7 CLI:** not invoked — the recommended feature depends on no new external
  library; all metric math is in-tree. (Would be used at plan time only if a new
  crate/dep is chosen.)
- **Web:** none required; upstream C++ is vendored locally under `catboost-master/`.

---

## 12. Confidence Assessment

- **HIGH:** crate inventory & implemented capabilities; phase statuses from the
  git-recovered ROADMAP + phase PLAN "Execution status" blocks; `loss_function_change`
  being Logloss/float-only; SHAP being float-only (no CTR); absence of standalone
  `eval_metrics`/`cv`/`cb-orchestrate`; `uv` oracle recipe.
- **MEDIUM:** exact upstream `eval_metric`/`eval_metrics` signatures and full
  metric-coverage list (confirm at plan time against `metric.cpp` / python core.py);
  FSTR-02's CTR sub-case requiring CTR-aware SHAP (strongly inferred from float-only
  SHAP, not from an upstream CTR-SHAP diff).
- **LOW / UNVERIFIED:** EXPORT-03 ONNX-Runtime oracle presence; precise Phase-24
  resolution timeline (does not affect the ORCH-04 recommendation).

---

## 13. Recommended next feature

**ORCH-04 — standalone `eval_metrics` / `calc_metrics`** (metrics on precomputed
predictions, staged + final, oracle-locked ≤10⁻⁵ vs `catboost.utils.eval_metric`;
reuses shipped, already-oracle-tested metric math; no SHAP/CTR/GPU/Apple dependency;
opens Phase 20 at its dependency-free leaf).

## 14. Runner-ups

1. **FSTR-02 numeric multi-loss LossFunctionChange** (completes Phase 18; defer the
   CTR sub-case, which needs CTR-aware SHAP first).
2. **CoreML export (EXPORT-02)** (completes Phase 17; weak local oracle — no Apple
   runtime, not ≤10⁻⁵).
3. **Cross-validation `cv()` (ORCH-01)** (high value; couples to training
   determinism; larger oracle surface).
