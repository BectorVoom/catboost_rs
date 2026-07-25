# catboost-rs — Next-Feature Research (Updated Gap Analysis)

**Research date:** 2026-07-18
**Branch:** `feat/18-fstr03-partial-dependence` (merged with `feat/23-ctr-model-loading`)
**Scope:** Read-only. No production code authored. Supersedes the state (not the
method) of `.planning/plans/unimplemented-features-survey/research.md`, whose top
pick (ORCH-04) has since shipped.

Evidence tags: `[VERIFIED: CODEGRAPH <path:symbol>]`, `[VERIFIED: LOCAL <path>]`,
`[INFERRED: reasoning]`, `[UNVERIFIED]`.

---

## 0. Executive Summary

catboost-rs is a **mature, near-complete** Rust rewrite. v1.0 (Phases 1–8) and
v1.1 GPU (Phases 10–14) are shipped; v1.2 (Phases 15–22) is in progress with 15,
16, 21, 21.5 complete `[VERIFIED: LOCAL git show a82289c:.planning/ROADMAP.md]`.

**Since the prior survey, the landscape moved:**

- **ORCH-04 `eval_metric`/`calc_metric` — now DONE.** The prior survey's #1
  recommendation shipped in commit `ba08aaf`
  (`feat(20/ORCH-04): standalone eval_metric/calc_metric surface`). It exists
  end-to-end: `crates/cb-train/src/calc_metrics.rs`, facade
  `crates/catboost-rs/src/metrics.rs`, Python `crates/catboost-rs-py/src/utils.rs`,
  and `crates/cb-oracle/fixtures/{calc_metrics,eval_metrics}/`. Ranking arm is
  wired (`is_ranking` → `EvalMetric::eval_grouped`)
  `[VERIFIED: LOCAL git log; crates/cb-train/src/calc_metrics.rs:250,288]`.
- **FSTR-01 CTR-aware fstr — IN FLIGHT (uncommitted on this branch).** The working
  tree has a large uncommitted change to `crates/cb-model/src/fstr.rs` (+485/-… )
  making `prediction_values_change` and `interaction` CTR-aware, plus new
  `crates/cb-model/tests/fstr_ctr_oracle_test.rs` and
  `crates/cb-oracle/fixtures/fstr_ctr/` `[VERIFIED: LOCAL git diff --stat; git diff
  crates/cb-model/src/fstr.rs header "FSTR-01: FIC-01/FIC-02/FIC-03"]`.
- **Phase 24 CTR split-search correctness (ORD-06/ORD-07) — IN FLIGHT.**
  Uncommitted changes to `crates/cb-train/src/{tree.rs(+208),boosting.rs(+21)}`,
  `tree_test.rs`, and `crates/cb-train/tests/ctr_split_scoring_test.rs`; SPECs still
  `status: draft`; the `simple-ctr-cat-feature-weight` PLAN-CHECK reached
  `Verdict: PASS` (plan approved, implementation in progress)
  `[VERIFIED: LOCAL git diff --stat; 24.../*/SPEC.md status: draft;
  simple-ctr-cat-feature-weight/PLAN-CHECK.md "Verdict: PASS"]`.

**Top recommendation: `sum_models` (model merge / weighted ensemble sum).** It is
genuinely missing, lands in a **new file** in `cb-model` with **zero overlap** with
the two in-flight workstreams, has the cleanest possible ≤10⁻⁵ oracle
(`catboost.sum_models` over frozen upstream `.cbm` models → save → predict-compare),
and decomposes into small failure-isolated behaviors (weight scaling · tree
concatenation · bias/scale sum). Runner-ups: (2) staged prediction public surface,
(3) FSTR-02 numeric multi-loss LossFunctionChange (single-crate but **merge-conflict
risk** with the in-flight fstr.rs), (4) CoreML export (completes Phase 17; weak local
oracle).

**Important caveat about the reference tree:** the task brief assumed the full
vendored C++ tree is present. It is **not**. `catboost-master/` in this working copy
is a **sparse checkout of only 3 files** (124 KB total):
`catboost/private/libs/algo/{greedy_tensor_search,train,yetirank_helpers}.cpp`
`[VERIFIED: LOCAL find catboost-master -type f]`. All other C++ reference paths
below are the canonical upstream layout `[INFERRED]`, not locally verifiable, and
must be re-confirmed against github.com/catboost/catboost at plan time.

---

## 1. Implementation-State Inventory

Workspace = 9 crates `[VERIFIED: LOCAL ls crates/]`: `cb-core`, `cb-data`,
`cb-compute`, `cb-backend`, `cb-train`, `cb-model`, `cb-oracle`, `catboost-rs`
(facade), `catboost-rs-py` (PyO3).

| Phase / Feature | Status | Evidence |
|---|---|---|
| 1–8 v1.0 Core Parity | ✅ DONE | `ROADMAP` shipped 2026-06-28 `[VERIFIED: LOCAL]` |
| 10–14 v1.1 GPU training (CubeCL) | ✅ DONE | shipped 2026-07-05 `[VERIFIED: LOCAL ROADMAP]` |
| 15 Debt discharge / CUDA oracle | ✅ DONE | `[VERIFIED: LOCAL ROADMAP]` |
| 16 Online-HNSW KNN parity (FEAT-07) | ✅ DONE | `crates/cb-compute/src/hnsw.rs` `[VERIFIED: LOCAL]` |
| 17 ONNX export (EXPORT-01) | ✅ DONE | `crates/cb-model/src/export/onnx.rs`; facade `save_onnx` `[VERIFIED: CODEGRAPH]` |
| 17 **CoreML export (EXPORT-02)** | ❌ MISSING | only `mod.rs` comment; no `coreml.rs` `[VERIFIED: LOCAL ls export/; grep -i coreml crates/]` |
| 17 Export ORT oracle (EXPORT-03) | ⚠️ UNVERIFIED | `onnx_test.rs` present; ORT round-trip not confirmed `[UNVERIFIED]` |
| 18 FSTR-03 partial dependence | ✅ DONE | `partial_dependence.rs`; facade `partial_dependence` `[VERIFIED: CODEGRAPH]` |
| 18 FSTR-01 interaction (+CTR) | 🚧 IN FLIGHT | uncommitted `fstr.rs`; `fstr_ctr_oracle_test.rs` new `[VERIFIED: LOCAL git diff]` |
| 18 **FSTR-02 LossFunctionChange (multi-loss/CTR)** | 🟡 PARTIAL | Logloss+float only `[VERIFIED: CODEGRAPH fstr.rs loss_function_change]` |
| 19 **GPU inference evaluator (GINF-01)** | ❌ MISSING | no infer crate `[VERIFIED: LOCAL ls crates/ ` grep infer` empty]` |
| 20 ORCH-04 `eval_metric`/`calc_metric` | ✅ DONE | `calc_metrics.rs`; facade `metrics.rs`; py `utils.rs` `[VERIFIED: LOCAL git log ba08aaf]` |
| 20 **CV `cv()` (ORCH-01)** | ❌ MISSING | no cv symbol `[VERIFIED: LOCAL grep cross_valid empty]` |
| 20 **grid/random search (ORCH-02)** | ❌ MISSING | none `[VERIFIED: LOCAL grep grid_search empty]` |
| 20 **snapshot/resume (ORCH-03)** | ❌ MISSING | none `[VERIFIED: LOCAL grep snapshot/checkpoint empty]` |
| 21 / 21.5 CPU histogram rewrite + scaling | ✅ DONE | `[VERIFIED: LOCAL ROADMAP]` |
| 22 **Adoption/DX capstone** | ❌ MISSING | last; exercises 17/19/20 `[VERIFIED: LOCAL ROADMAP]` |
| 23 CTR `.cbm` load + save | ✅ DONE | commits `9015b22`, `c5ff842`; `cb-model/src/{cbm,ctr_data}.rs` `[VERIFIED: LOCAL git log]` |
| 24 **CTR split-search correctness (ORD-06/07)** | 🚧 IN FLIGHT | uncommitted `tree.rs`/`boosting.rs`; SPECs draft `[VERIFIED: LOCAL git diff]` |
| **`sum_models` / model merge** | ❌ MISSING | no `sum_model`/`merge_model` symbol anywhere `[VERIFIED: LOCAL grep -rli sum_model crates/ → empty]` |
| **staged predict public API** | ❌ MISSING (internals exist) | facade has no `staged_predict`; apply.rs has prefix internals `[VERIFIED: LOCAL grep model.rs pub fn; apply.rs]` |
| **PMML export** | ❌ MISSING (unplanned) | not in ROADMAP `[INFERRED]` |

Confirmed-DONE capability breadth (unchanged from prior survey, spot-verified):
model load/save (`.cbm`/`.json`, incl. CTR reconstruction); apply/predict incl.
multiclass + virtual ensembles + CTR; the full loss/metric families; ranking;
text/embedding + HNSW; ordered boosting + ordered/tensor/combination CTR; one-hot;
monotone/penalty; non-symmetric growers; feature selection; SHAP family (float);
PVC / interaction fstr; ONNX export; GPU training
`[VERIFIED: LOCAL crates/*/src; crates/cb-oracle/fixtures ~60 families]`.

---

## 2. Candidate Features (Ranked)

Selection criteria: (a) genuinely missing, (b) **not overlapping the two in-flight
workstreams** (fstr.rs FSTR-01, tree.rs Phase-24) to avoid merge conflicts,
(c) self-contained with failure-isolated behaviors, (d) a clean ≤10⁻⁵ oracle path.

### Candidate 1 (RECOMMENDED) — `sum_models` (model merge / weighted sum)

- **What it is:** CatBoost's public `sum_models(models, weights, ctr_merge_policy)`
  — combine N trained models into one by scaling each model's leaf values by its
  weight and concatenating (summing) their trees + biases/scales into a single
  `Model`. Widely used for ensembling and blending.
  `[INFERRED: upstream catboost.sum_models]`
- **C++ reference (upstream, not in sparse checkout):**
  `catboost/libs/model/model.cpp` — `SumModels(...)` / `TModelSumBuilder`; Python
  wrapper `catboost/python-package/catboost/core.py` `sum_models(...)`. Re-confirm
  paths at plan time `[INFERRED — not locally verifiable, sparse checkout]`.
- **Target Rust crate/module:** **new file** `crates/cb-model/src/model_sum.rs`
  (+ `model_sum_test.rs`), operating on the existing `cb_model::Model` /
  `oblivious_trees` / leaf-value representation; facade `Model::sum_models` in
  `crates/catboost-rs/src/model.rs`; optional Python `catboost_rs.sum_models` in
  `catboost-rs-py`. No touch of `fstr.rs` or `tree.rs`
  `[VERIFIED: CODEGRAPH crates/cb-model/src/model.rs; apply.rs predict_raw]`.
- **Scope classification:** **local → cross-module (facade/py)**. Read/construct
  only; no training, no backend.
- **Oracle feasibility (≤10⁻⁵): STRONG.** Freeze 2–3 upstream float-only `.cbm`
  models (avoid CTR run-to-run nondeterminism — the documented frozen-fixture
  rule), call `catboost.sum_models(...)`, save result `.cbm`, and compare Rust
  `sum_models(...).predict_raw` vs the upstream-summed model's predictions on a
  fixed pool. New `crates/cb-oracle/fixtures/model_sum/` mirrors the existing
  `model_serde` fixture shape `[VERIFIED: LOCAL crates/cb-oracle/fixtures/model_serde;
  MEMORY ctr-model-loading "CTR fixtures frozen"]`.
- **Estimated risk: LOW.** Structure-compatible oblivious float models sum
  cleanly. Edge cases: mismatched feature sets, differing tree structures (kept as
  separate concatenated trees), scale/bias combination, and the CTR merge policy
  (defer CTR to a second slice — same discipline the codebase uses elsewhere).
  No overlap with in-flight code.

### Candidate 2 — Staged prediction public surface (`staged_predict`)

- **What it is:** Public per-tree-prefix prediction — evaluate the model over an
  increasing number of trees (`ntree_start`, `ntree_end`, `eval_period`) yielding a
  matrix/iterator of cumulative predictions. Upstream `CatBoost.staged_predict(...)`
  `[INFERRED: upstream core.py staged_predict]`.
- **C++ reference:** `catboost/libs/model/model.cpp` /
  `catboost/python-package/.../core.py staged_predict` `[INFERRED]`.
- **Target Rust crate/module:** `crates/cb-model/src/apply.rs` already has
  prefix-capable internals (`apply_virtual_ensembles` takes an `end`;
  `collect_leaves_statistics`) — add a `predict_raw_staged` free fn, facade
  `Model::staged_predict`, Python method `[VERIFIED: CODEGRAPH apply.rs:712,733
  "end = oblivious_trees.len()"; :647,370]`.
- **Scope classification:** **local → cross-module.** Read-only.
- **Oracle feasibility: STRONG.** `model.staged_predict(pool, eval_period=k)` on a
  frozen float `.cbm` → matrix of per-prefix predictions, compared ≤10⁻⁵.
  Failure-isolated per prefix; reuses existing `prediction_types` fixtures shape
  `[VERIFIED: LOCAL crates/cb-oracle/fixtures/prediction_types]`.
- **Estimated risk: LOW.** Some conceptual overlap with the just-shipped staged
  `calc_metric` accumulation, but the surfaces are distinct (predictions vs
  metrics). No overlap with in-flight files.

### Candidate 3 — FSTR-02 numeric multi-loss LossFunctionChange

- **What it is:** Generalize `loss_function_change` beyond hard-coded binary
  Logloss to the model's actual metric's `GetFinalError` (RMSE/MAE/Quantile/…),
  numeric-features only in the first slice. Completes Phase 18's last partial item
  `[VERIFIED: CODEGRAPH crates/cb-model/src/fstr.rs loss_function_change — Logloss
  final-error hardcoded]`.
- **C++ reference:** `catboost/private/libs/algo/../loss_change_fstr.cpp` and
  `catboost/libs/metrics/metric.cpp` (`GetFinalError`) `[INFERRED]`.
- **Target Rust crate/module:** `crates/cb-model/src/fstr.rs` + reuse
  `cb_train`/facade metric arithmetic now exposed by ORCH-04
  (`calc_metrics.rs`/`metrics.rs`) `[VERIFIED: CODEGRAPH]`.
- **Scope classification:** **local (single crate).**
- **Oracle feasibility: GOOD.** `model.get_feature_importance(type='LossFunctionChange')`
  on numeric models per loss; existing `fstr_loss_change/` fixtures + generator
  extend directly `[VERIFIED: LOCAL crates/cb-oracle/fixtures/fstr_loss_change/gen_fixtures.py]`.
- **Estimated risk: MEDIUM — MERGE CONFLICT.** `fstr.rs` is the file with the large
  uncommitted FSTR-01 CTR change. Starting FSTR-02 now risks conflicting edits;
  sequence it **after** the FSTR-01 CTR work lands. The CTR sub-case additionally
  needs CTR-aware SHAP (a hidden dependency) — defer it, ship numeric-only first.

### Candidate 4 — CoreML export (EXPORT-02)

- **What it is:** Read-only exporter to Apple CoreML `.mlmodel` for float-only
  oblivious models, mirroring the shipped ONNX exporter. Completes Phase 17.
- **C++ reference:** `catboost/libs/model/model_export/` CoreML path `[INFERRED —
  not in sparse checkout]`.
- **Target Rust crate/module:** new `crates/cb-model/src/export/coreml.rs`
  (`mod.rs` already anticipates it); would need a CoreML protobuf schema encoded
  via the existing `prost = "0.14.4"` (as ONNX does)
  `[VERIFIED: LOCAL crates/cb-model/Cargo.toml prost 0.14.4; export/mod.rs]`.
- **Scope classification:** **local → external (Apple runtime for verification).**
- **Oracle feasibility: WEAK on Linux.** No Apple runtime in-env; parity is an
  **export-specific float32 tolerance vs CatBoost's own CoreML export**, explicitly
  NOT the ≤10⁻⁵ double bar (ROADMAP milestone context)
  `[VERIFIED: LOCAL ROADMAP "export uses an export-specific float32 tolerance"]`.
- **Estimated risk: MEDIUM.** Self-contained code, but the verification story is
  the weakest of the four (structural round-trip only on this host).

### Lower-ranked / rejected (unchanged rationale)

- **CV `cv()` (ORCH-01)** — high value, the true Phase-20 anchor, but couples to
  training determinism and has a large oracle surface (fold assignment + per-fold
  training + cv-results table). Medium.
- **grid/random search (ORCH-02)** — hard-depends on ORCH-01; not a leaf. Reject.
- **snapshot/resume (ORCH-03)** — deep trainer-state coupling; self-oracle
  (resume==straight), not vs C++. Reject as "next".
- **GPU inference (GINF-01)** — new crate, CUDA-only, needs Kaggle oracle (ε=1e-4,
  not ≤1e-5); not runnable/verifiable on this host. Reject as "next".
- **PMML export** — unplanned; weak oracle; low demand. Reject.

---

## 3. Dependency Versions (relevant to candidates)

From `Cargo.toml` (workspace) and `crates/cb-model/Cargo.toml`
`[VERIFIED: LOCAL Cargo.toml; crates/cb-model/Cargo.toml]`:

- `thiserror = 2.0.18`, `anyhow = 1.0.102` (anyhow **banned** in `cb-model`, D-14),
  `serde = 1.0.228`, `serde_json = 1.0.150`.
- `ndarray = 0.17.2`, `ndarray-npy = 0.10.0` (fixture `.npy` I/O), `approx = 0.5`.
- `prost = 0.14.4` — protobuf wire encoding used by the ONNX exporter; **reusable
  for CoreML** (Candidate 4) via a committed generated schema (same pattern as
  `src/generated/onnx_generated.rs`).
- `flatbuffers = 25.12.19` — `.cbm` FlatBuffers runtime (relevant to
  `sum_models`/staged: both operate on the already-deserialized `Model`, so no new
  serialization dep needed).
- `arrow = 59.0.0`, `polars = 0.54.4`, `rayon = 1.12.0`, `cubecl = 0.10.0`,
  `bytemuck = 1` — not needed by Candidates 1–4.
- **Candidates 1, 2, 3 require NO new external crate** (in-tree `Model` +
  metric math). Candidate 4 (CoreML) requires only a generated protobuf schema on
  the existing `prost`. Aligns with the "use existing capability first / latest
  crate versions" constraint `[VERIFIED: LOCAL CLAUDE.md Constraints]`.

---

## 4. Oracle Tooling & Verification (all candidates)

- Harness: `cb_oracle::{fixture, compare}` at ≤1e-5; fixtures are pinned-seed
  Python writing `.npy` + `config.json`/`expected.json` straight from
  `catboost==1.2.10` `[VERIFIED: LOCAL crates/cb-oracle/src/compare.rs;
  crates/cb-oracle/fixtures/fstr_loss_change/gen_fixtures.py]`.
- Oracle env recipe (system python is 3.14; no catboost 3.14 wheel):
  `uv venv --python 3.12 && uv pip install catboost==1.2.10 'numpy<2'`
  `[VERIFIED: LOCAL MEMORY fstr03-partial-dependence-plan]`.
- **Frozen-fixture rule:** CatBoost quantization/CTR is run-to-run
  nondeterministic — for model-bearing fixtures, freeze upstream `.cbm`/`.npy`
  artifacts; prefer **float-only** models for the first slice of any candidate to
  keep the oracle deterministic `[VERIFIED: LOCAL MEMORY ctr-model-loading].`
- Lint gate is **clippy, not build**: `unwrap/expect/panic/indexing_slicing` are
  denied; scope new-code lint with `cargo clippy -p <crate> --lib --no-deps`
  (workspace is broadly red in untouched files) `[VERIFIED: LOCAL MEMORY
  fstr03-plan gotchas]`.
- Test-mount pattern: unit tests live in a sibling `*_test.rs` mounted via
  `#[cfg(test)] #[path="X_test.rs"] mod tests;` in the prod file — omitting the
  mount silently runs 0 tests `[VERIFIED: LOCAL crates/cb-model/src/ctr_data.rs
  mount; CLAUDE.md source/test separation rule]`.
- Verified command analogues:
  `cargo test -p cb-model --test model_serde_oracle_test` (model round-trip
  analogue for `sum_models`); `cargo test -p cb-train --test eval_metrics_oracle_test`
  `[VERIFIED: LOCAL crates/cb-oracle/fixtures/{model_serde,eval_metrics}]`.

---

## 5. Recommended Sequencing for the Planner

1. **`sum_models` (Candidate 1)** now — new file, no in-flight overlap, strongest
   oracle. First slice = structure-compatible float-only oblivious models + facade;
   defer the CTR merge policy to a second slice.
2. **Staged prediction (Candidate 2)** — independent, low risk; can be planned in
   parallel (different concern from `sum_models`).
3. **FSTR-02 numeric (Candidate 3)** — schedule **after** the uncommitted FSTR-01
   CTR change lands, to avoid `fstr.rs` merge conflicts.
4. **CoreML (Candidate 4)** — only if closing Phase 17 outranks the above; accept
   the weak local (structural-only) verification.

Decisions the planner must preserve: `anyhow` banned in `cb-model`; deterministic
sums via `cb_core::sum_f64`; float-first oracle to dodge CTR nondeterminism; new
code passes `cargo clippy` (not just build); tests in sibling `*_test.rs`.

---

## 6. Open Questions / Unknowns

1. **Sparse reference tree.** `catboost-master/` holds only 3 `.cpp` files; every
   non-listed C++ path here is `[INFERRED]` from upstream layout and must be
   re-confirmed against github.com/catboost/catboost at plan time (or by expanding
   the sparse checkout) `[VERIFIED: LOCAL find catboost-master -type f]`.
2. **`sum_models` CTR merge policy** — upstream exposes `ctr_merge_policy`; whether
   the first Rust slice supports CTR models or float-only is a scope decision
   (recommend float-only first, mirroring the ONNX exporter's float-only start).
3. **Exact upstream signatures** for `sum_models` and `staged_predict`
   (arg names/defaults) — confirm against `core.py` at plan time. `[UNVERIFIED]`
4. **EXPORT-03 ONNX-Runtime oracle** — is there already an ORT round-trip test, or
   only structural export? `onnx_test.rs` presence confirmed; ORT execution not.
   Relevant only if CoreML (Candidate 4) is chosen. `[UNVERIFIED]`
5. **In-flight completion timing** — FSTR-01 (fstr.rs) and Phase-24 (tree.rs) are
   uncommitted; their landing order gates when FSTR-02 (Candidate 3) is safe.
   `[VERIFIED: LOCAL git diff --stat]`

---

## 7. Sources

- **CodeGraph / Read:** `crates/cb-model/src/{apply.rs,fstr.rs,model.rs,export/mod.rs}`;
  `crates/catboost-rs/src/model.rs`; `crates/cb-train/src/calc_metrics.rs`;
  `crates/catboost-rs-py/src/utils.rs`.
- **Local:** `Cargo.toml`, `crates/cb-model/Cargo.toml`; `crates/*/src` + `tests`
  listings; `crates/cb-oracle/fixtures/` (~60 families incl. `calc_metrics`,
  `eval_metrics`, `fstr_ctr`, `fstr_loss_change`, `model_serde`, `ctr_load`,
  `partial_dependence`); `find catboost-master -type f` (3 files);
  `.planning/phases/{17,18,20,23,24}/…/{SPEC,PLAN,PLAN-CHECK}.md`;
  `.planning/plans/unimplemented-features-survey/research.md`.
- **Git:** `git log --oneline` (esp. `ba08aaf` ORCH-04, `c981e33` ONNX,
  `9015b22`/`c5ff842` CTR load/save); `git diff --stat` + `git diff
  crates/cb-model/src/fstr.rs` (in-flight FSTR-01 CTR); `git show
  a82289c:.planning/ROADMAP.md` (v1.2 roadmap; file deleted from working tree).
- **Project memory:** `ctr-model-loading.md` (frozen CTR fixtures),
  `fstr03-partial-dependence-plan.md` (uv oracle recipe, clippy gate, test-mount),
  `orch04-calc-metrics-plan.md`.
- **Context7 CLI:** not invoked — Candidates 1–3 need no new external library;
  Candidate 4 reuses in-tree `prost`. (Would be used only if a new dep is chosen.)
- **Web:** none required in this pass; upstream signatures deferred to plan time.

---

## 8. Confidence Assessment

- **HIGH:** ORCH-04 shipped (git log + files); FSTR-01 CTR + Phase-24 in flight
  (git diff); absence of `sum_models`/staged-predict/CoreML/CV/snapshot symbols
  (grep); crate inventory & dep versions; sparse `catboost-master` (only 3 files);
  oracle harness + env recipe + lint/test conventions.
- **MEDIUM:** `sum_models`/`staged_predict` being clean, low-risk, strongly
  oracle-able (strongly inferred from the existing `Model`/`apply` surface and the
  `model_serde`/`prediction_types` fixture precedents, not yet prototyped);
  FSTR-02's numeric slice being single-crate once fstr.rs settles.
- **LOW / UNVERIFIED:** exact upstream C++ paths & Python signatures (sparse tree);
  EXPORT-03 ORT-oracle presence; CoreML export-tolerance parity on this host.
