---
title: "ORCH-02 — grid_search / randomized_search — TDD Implementation Plan"
phase: 20-orchestration
slice: grid-random-search
plan_version: 1
status: planned
updated_at: 2026-07-19T00:00:00Z
source_spec: .planning/plans/grid-random-search/SPEC.md
source_research: "Phase-20 orchestration research pass (ORCH-01/02/03)"
depends_on_plan: .planning/plans/cv-cross-validation/PLAN.md   # ORCH-01 (draft, unimplemented) — HARD blocker
gsd_used: false
---

# ORCH-02 — TDD Implementation Plan

Plan-only artifact. No production code authored here. Every file/symbol/command
below is verified against on-disk source via CodeGraph + Read (evidence inline).
The cv, training, metric-parse, shuffle, and error seams are **reused, never
modified** (D-04 no-regression): `catboost_rs::cv`, `CatBoostBuilder::fit`,
`cb_train::parse_metric`, `cb_train::fisher_yates_permutation`,
`cb_compute::CustomMetric::is_max_optimal`.

> **HARD CROSS-PLAN BLOCKER (read first).** This plan builds ON TOP OF the sibling
> slice **ORCH-01 `cv()`** (`.planning/plans/cv-cross-validation/`), which is
> itself **draft / unimplemented**. `cv`, `CvResult`, `CvFold`, `CvType`,
> `make_cv_folds` do **NOT** yet exist in the facade — `crates/catboost-rs/src/
> lib.rs:27-30` re-exports only `CatBoostBuilder`, `CatBoostError`,
> `eval_metric`/`eval_metrics` (ORCH-04), `Model`
> `[VERIFIED: LOCAL grep crates/catboost-rs/src/lib.rs]`. Therefore:
> - **ORCH-01 must ship first.** Concretely, ORCH-01 **TASK-04** must land the
>   `CvResult` type, and ORCH-01 **TASK-06** must land the public `cv` function
>   `[VERIFIED: LOCAL .planning/plans/cv-cross-validation/PLAN.md:289-303 (TASK-04
>   CvResult), :385-434 (TASK-06 cv)]`.
> - Every ORCH-02 task except **TASK-01** (the pure `metric_is_max_optimal`
>   direction helper, which needs no `CvResult`) is **BLOCKED** on that landing.
>   The wave graph below encodes this as an explicit external gate `G-ORCH-01`.
> - Do **not** treat `cv`/`CvResult` as verified working code; they are a
>   `[DEPENDENCY CONTRACT: ORCH-01]` until ORCH-01's own oracle (its TASK-06) is
>   green.
> - **ORCH-01 is not just unshipped — it is presently UNDER ACTIVE REVISION** to
>   fix a CRITICAL Plan-Checker finding (per-fold quantization potentially
>   diverging from upstream `catboost.cv()`; see
>   `.planning/plans/cv-cross-validation/PLAN-CHECK.md`). A quantization-semantics
>   fix could plausibly change `cv()`'s parameter list, its error taxonomy, or —
>   most importantly for TASK-03's fixture-reuse claim — the FROZEN `cv/` fixture
>   values ORCH-01's own TASK-01 generates. **`G-ORCH-01` therefore has a THIRD
>   condition beyond "TASK-04/TASK-06 landed":** before TASK-02 begins, re-read
>   ORCH-01's FINAL, merged (post-fix) `cv.rs` source and its regenerated `cv/`
>   fixtures — not just this document's current citations of ORCH-01's draft
>   `PLAN.md` — and re-verify (a) `cv()`'s exact signature still matches this
>   plan's SPEC §4 typed contracts, (b) `CvResult`'s column-key convention (SPEC
>   §9 Q1) against the ACTUAL shipped code, and (c) that TASK-03's "no NEW
>   fixtures required" claim still holds against the regenerated fixtures. Treat
>   `G-ORCH-01` as unsatisfied until this checkpoint passes, even if TASK-04/06
>   have technically landed.

## 0. Goal-backward derivation

Acceptance outcomes (SPEC §6) drive the task set:

| Acceptance | Observable success | Task |
|---|---|---|
| AT-S1 | `metric_is_max_optimal` correct per metric class + Custom delegation | TASK-01 |
| AT-S2a/b | `score_candidate` best-over-iterations; `select_best` argbest + lowest-index tie-break + missing-column/empty error | TASK-02 |
| AT-S3a | `grid_search` `cv_results` == `cv(&candidates[best_index],…)`; `best_index` == manual argbest (self-consistency, explicit folds) | TASK-03 |
| AT-S3b | `refit=true` ⇒ `best_model` preds == `best_builder.fit(pool)` preds; empty candidates/metrics → typed error | TASK-03 |
| AT-S4 | `randomized_search` first-`n_iter`-of-permutation subsample; same-seed determinism; `n_iter>=M` = full; index mapped back | TASK-04 |
| AT-S5 | `catboost_rs.grid_search(...)` dict shape + cv self-consistency + bad-param raises | TASK-05 |
| AT-S6 | `catboost_rs.randomized_search(...)` determinism + subset size + shape | TASK-06 |

Reused seams (verified, do NOT modify):

- `CatBoostBuilder` `#[derive(Debug, Clone, PartialEq)]` + setters + `fit(&self,
  pool:&Pool) -> Result<Model, CatBoostError>` — `crates/catboost-rs/src/
  builder.rs:64-85,334`. Cheap clone-and-mutate per candidate. `[VERIFIED: CODEGRAPH]`
- `catboost_rs::cv(pool, builder, metrics, fold_count, shuffle,
  partition_random_seed, inverted, folds) -> Result<CvResult, CatBoostError>` +
  `CvResult { iterations: Vec<usize>, columns: BTreeMap<String, Vec<f64>> }` with
  keys `test-<M>-mean` etc. — `[DEPENDENCY CONTRACT: ORCH-01 SPEC §4,
  .planning/plans/cv-cross-validation/SPEC.md:249-281]`. **NOT yet in the tree.**
- `cb_train::parse_metric(&str) -> CbResult<EvalMetric>` (ORCH-04, shipped) —
  re-export `crates/cb-train/src/lib.rs:71`. `[VERIFIED: LOCAL]`
- `cb_train::EvalMetric` variants (flat: `Rmse|Logloss|Msle|Mae|Mape|Quantile|
  Custom`; ranking: `Ndcg|Dcg|Map|Mrr|Err|PFound|PrecisionAt|RecallAt|QueryAuc`) —
  the ranking partition mirrors the `use_group_weight` match at
  `crates/cb-train/src/metrics.rs:533-541` and the flat/ranking split in `eval`
  (`:449-459`) / `eval_one_group` (`:604-615`). `[VERIFIED: CODEGRAPH]`
- `cb_compute::CustomMetric::is_max_optimal(&self) -> bool` (via the
  `EvalMetric::Custom(CustomMetricHandle)` handle `.0`) —
  `crates/cb-compute/src/custom.rs:111-114,155`. `[VERIFIED: CODEGRAPH]`
- **No `EvalMetric` direction method exists** — the enum has only `eval`,
  `eval_grouped`, `for_loss`; direction is prose-only (`metrics.rs:77,82,86`) →
  ORCH-02 must ADD `metric_is_max_optimal`. `[VERIFIED: LOCAL grep]`
- `cb_train::fisher_yates_permutation(n: usize, seed: u64) -> Vec<i32>` over
  `cb_core::rng::TFastRng64` — re-export `crates/cb-train/src/lib.rs:75`; RNG
  `crates/cb-core/src/rng.rs:171` (`from_seed`), `:265` (`uniform`). Deterministic;
  no new `rand` dep. `[VERIFIED: LOCAL + CODEGRAPH]`
- `CatBoostError::Train(#[from] cb_core::CbError)` / `UnsupportedModel(String)` —
  `?` on a `CbResult` converts to `CatBoostError::Train`. `[VERIFIED: LOCAL cv
  PLAN.md:64-66]`
- `rayon` is a `catboost-rs` dep (`crates/catboost-rs/Cargo.toml:39`); order-
  preserving `par_iter().map(...).collect::<Result<Vec<_>,_>>()` (cv PLAN
  precedent `:405-409`). `[VERIFIED: CODEGRAPH/LOCAL]`
- Python: `sum_models` free `#[pyfunction]` (`crates/catboost-rs-py/src/
  regressor.rs:277-301`) returning `EstimatorBase::from_model`; `make_builder(
  &BTreeMap<String, Py<PyAny>>, py)` (`params.rs:451`), `validate_params`
  (`params.rs:290`), `data_to_pool(py, x, y)` (`estimator.rs:236`), `EstimatorBase`
  with `params: BTreeMap<String, Py<PyAny>>` (`estimator.rs:24`); registration
  `m.add_function(wrap_pyfunction!(regressor::sum_models, m)?)` (`lib.rs:57`);
  error chokepoint `PyCbError`/`errors::to_pyerr`. `[VERIFIED: CODEGRAPH]`
- Facade root test-mount idiom `#[cfg(test)] mod grid_search_test;`
  (`crates/catboost-rs/src/lib.rs:63-69`, cf. `mod metrics_test;`). `[VERIFIED: LOCAL]`

Selection rule (new, in `grid_search.rs`): the PRIMARY metric is `metrics[0]`;
a candidate's score is the best value over iterations of its
`columns[format!("test-{primary}-mean")]` — `min` if `!metric_is_max_optimal`,
`max` if it is; `select_best` picks the argbest (argmax for max-optimal, argmin
otherwise), lowest index on ties. This is the DOCUMENTED first-slice rule; the
oracle is decomposition-based, not upstream-`grid_search`-parity (SPEC §2/§9 Q2).

## 1. Execution order & waves

```
                       ┌─────────────────────────────────────────────────┐
Wave 0 (unblocked):    │ TASK-01  metric_is_max_optimal + grid_search.rs  │  (pure; needs NO CvResult)
                       └─────────────────────────────────────────────────┘
========================  external gate G-ORCH-01  ========================
   (ORCH-01 TASK-04 lands `CvResult`  AND  ORCH-01 TASK-06 lands `cv`
    AND the pre-TASK-02 re-verification checkpoint above passes against
    ORCH-01's FINAL post-fix cv.rs + regenerated fixtures — not its draft)
===========================================================================
Wave A (after G-ORCH-01): TASK-02  score_candidate + select_best   depends: TASK-01, G-ORCH-01(CvResult)
Wave B:                   TASK-03  grid_search facade + oracle      depends: TASK-02, G-ORCH-01(cv)  (same prod file)
Wave C:                   TASK-04  randomized_search facade         depends: TASK-03                 (same prod file)
Wave D:                   TASK-05  Python grid_search               depends: TASK-03  (parallel w/ TASK-04 — different file, no shared write)
Wave E:                   TASK-06  Python randomized_search         depends: TASK-04, TASK-05         (same file search.rs)
```

Dependency graph:

```
TASK-01 ─> TASK-02 ─> TASK-03 ─┬─> TASK-04 ─┐
   │           ▲          ▲    └─> TASK-05 ─┴─> TASK-06
   │           │          └── external gate G-ORCH-01 (cv)  [BLOCKER]
   └───────────┴───────────── external gate G-ORCH-01 (CvResult)  [BLOCKER]
```
TASK-06 requires BOTH TASK-04 (`randomized_search`) and TASK-05 (`search.rs` +
`expand_param_grid`) to have landed — it is NOT simply "after TASK-05" in a
linear chain; TASK-04 and TASK-05 form a genuine parallel wave off TASK-03,
which then rejoins at TASK-06 (corrected here after PLAN-CHECK pass 2 caught
this diagram/prose drift — the wave graph and TASK-05's own "Parallelization"
note already had this right; only this diagram and TASK-06's "Blocked by" line
below had not been updated to match).

Acyclic. **TASK-01 is parallelizable with ORCH-01's own remaining tasks** (it
touches only the NEW `crates/catboost-rs/src/grid_search.rs` + `lib.rs` and uses
no ORCH-01 symbol). TASK-02/03/04 are strictly serial (same prod file
`grid_search.rs`, write conflict). **TASK-05 depends only on TASK-03** (it calls
`catboost_rs::grid_search` alone, never `randomized_search`, and writes to a
DIFFERENT file, `crates/catboost-rs-py/src/search.rs` — corrected from an
earlier "depends: TASK-04" mis-statement caught in PLAN-CHECK) — so **TASK-05
and TASK-04 form the one intra-plan parallel wave** once TASK-03 lands (disjoint
files: `grid_search.rs` vs `search.rs`). TASK-06 is sequential after BOTH TASK-04
(needs `randomized_search`) and TASK-05 (shares `search.rs` + reuses TASK-05's
`expand_param_grid` helper).

File-ownership note (no write conflicts within a wave): TASK-01/02/03/04 →
`crates/catboost-rs/src/grid_search.rs` (+ `grid_search_test.rs`, `lib.rs`);
TASK-03/04 also add `crates/catboost-rs/tests/grid_search_oracle_test.rs`;
TASK-05/06 → `crates/catboost-rs-py/src/search.rs` (+ `lib.rs`, Python tests).
TASK-04 and TASK-05 touch disjoint crates/files (`catboost-rs` vs
`catboost-rs-py`) and neither's Green step needs the other's output, so they may
run in parallel; TASK-06 re-serializes on `search.rs`.

## 2. Spec-ID → task coverage

| Spec | Behavior | Task(s) |
|---|---|---|
| ORCH-02-S1 | `metric_is_max_optimal` direction | TASK-01 |
| ORCH-02-S2 | `score_candidate` + `select_best` | TASK-02 |
| ORCH-02-S3 | Rust facade `grid_search` (+ refit) | TASK-03 |
| ORCH-02-S4 | Rust facade `randomized_search` | TASK-04 |
| ORCH-02-S5 | Python `catboost_rs.grid_search` | TASK-05 |
| ORCH-02-S6 | Python `catboost_rs.randomized_search` | TASK-06 |

Every S1..S6 covered; every task maps back to ≥1 spec. Acyclic graph.

---

## TASK-01 — `metric_is_max_optimal` + `grid_search.rs` scaffold (ORCH-02-S1)

- **Spec refs:** ORCH-02-S1. Primary failure reason: a metric's optimization
  direction is wrong (min treated as max or vice-versa).
- **Blocked by:** nothing (the ONLY unblocked task — needs no `CvResult`).
- **Goal / completion:** `catboost_rs::grid_search::metric_is_max_optimal(
  &EvalMetric) -> bool` exists; unit tests in `grid_search_test.rs` pass; `cargo
  clippy -p catboost-rs --lib --no-deps` clean. Stands up the module scaffold the
  later facade tasks extend.
- **Files:**
  - Create: `crates/catboost-rs/src/grid_search.rs` — module doc + `use
    cb_train::EvalMetric;` + `pub fn metric_is_max_optimal`. (Do NOT yet import
    `crate::cv::CvResult` — that type does not exist until G-ORCH-01; importing it
    now would break the build. Add the `use crate::cv::CvResult;` only in TASK-02.)
    Mount tests at file end: `#[cfg(test)] mod grid_search_test;` is added at the
    CRATE ROOT (`lib.rs`), matching the facade idiom (`mod metrics_test;`).
  - Create: `crates/catboost-rs/src/grid_search_test.rs` — S1 unit tests.
  - Modify: `crates/catboost-rs/src/lib.rs` — add `mod grid_search;` + (scaffold)
    `pub use grid_search::metric_is_max_optimal;` near the ORCH-04 `metrics`
    re-export (`lib.rs:24-29`), and `#[cfg(test)] mod grid_search_test;` near
    `mod metrics_test;` (`lib.rs:65`).
- **CodeGraph/Read evidence:** `EvalMetric` variants + ranking partition
  (`metrics.rs:64-151`, `use_group_weight` match `:533-541`, flat/ranking split in
  `eval` `:449-459`); `CustomMetric::is_max_optimal` (`custom.rs:111-114`),
  reachable via `EvalMetric::Custom(CustomMetricHandle)` `.0` (`custom.rs:155`);
  no existing direction method (`[VERIFIED: LOCAL grep]`); mount idiom `lib.rs:65`.
- **Match arms (implement exactly):**
  - `EvalMetric::Rmse | Logloss | Msle | Mae | Mape | Quantile { .. }` ⇒ `false`.
  - `EvalMetric::Ndcg { .. } | Dcg { .. } | Map { .. } | Mrr { .. } | Err { .. } |
    PFound { .. } | PrecisionAt { .. } | RecallAt { .. } | QueryAuc { .. }` ⇒ `true`.
  - `EvalMetric::Custom(h)` ⇒ `h.0.is_max_optimal()`.
  Exhaustive match (no wildcard) so a future variant forces a compile error, not a
  silent wrong default. No `unwrap`/panic/indexing.
- **Red:** in `grid_search_test.rs`:
  - `direction_min_metrics` — `metric_is_max_optimal(&EvalMetric::Rmse) == false`
    and same for `Logloss`, `Msle`, `Mae`, `Mape`, `Quantile{alpha:0.5}`.
  - `direction_max_metrics` — `metric_is_max_optimal(&EvalMetric::Ndcg{top:-1,
    dcg_type:DcgMetricType::Base, denominator:DcgDenominator::LogPosition}) == true`
    and same for a `Map`, `Mrr`, `QueryAuc{auc_type:AucType::Classic}` sample.
  - `direction_custom_delegates` — build a tiny `CustomMetric` stub whose
    `is_max_optimal()` returns `true`, wrap it in `EvalMetric::Custom(
    CustomMetricHandle::new(Arc::new(stub)))`, assert the helper returns `true`.
  Expected INITIAL failure: `grid_search` module / `metric_is_max_optimal` does
  not exist ⇒ compile error (unresolved import), i.e. the test file fails to build.
- **Green:** implement the match; the three tests pass.
- **Refactor:** none beyond clarity (a single exhaustive match). Regression scope:
  `grid_search_test.rs` only; `metrics.rs`/`custom.rs` untouched (D-04); the
  `EvalMetric` blast radius (11 cb-train/facade callers) unaffected — this only
  READS variants.
- **Validation:**
  - `cargo test -p catboost-rs --lib grid_search`
  - `cargo clippy -p catboost-rs --lib --no-deps`
- **Completion evidence:** 3 S1 tests green; clippy clean; `mod grid_search;`
  compiles WITHOUT any `cv`/`CvResult` reference (proves TASK-01 is gate-free).
- **Compat/rollback:** additive module; rollback = remove the two files + the
  three `lib.rs` lines.
- **Parallelization:** may proceed immediately, in parallel with ORCH-01's
  remaining tasks (disjoint files). Blocks TASK-02.

---

## TASK-02 — `score_candidate` + `select_best` (ORCH-02-S2)

- **Spec refs:** ORCH-02-S2. Primary failure reason: candidate scoring reduces the
  cv column wrong (wrong iteration extremum / wrong argbest / bad tie-break).
- **Blocked by:** TASK-01 **AND** external gate **G-ORCH-01 (`CvResult` type must
  exist — ORCH-01 TASK-04)**. Until `crate::cv::CvResult` compiles, this task
  cannot build. `[DEPENDENCY CONTRACT: ORCH-01; VERIFIED: LOCAL cv PLAN.md:289-305]`
- **Goal / completion:** `score_candidate(&CvResult, &str) -> Result<f64,
  CatBoostError>` and `select_best(&[CvResult], &str) -> Result<usize,
  CatBoostError>` exist in `grid_search.rs`; S2 unit tests pass; clippy clean.
- **Files:**
  - Modify: `crates/catboost-rs/src/grid_search.rs` — add `use crate::cv::CvResult;
    use crate::CatBoostError; use cb_train::parse_metric;` + the two fns. Column
    key: `format!("test-{metric}-mean")` (documented coupling to ORCH-01's naming
    — verify the exact spelling against the shipped `cv()` here; if `cv()`
    canonicalizes the descriptor, apply the SAME canonicalization to `metric`
    before the lookup — SPEC §9 Q1).
  - Modify: `crates/catboost-rs/src/grid_search_test.rs` — S2 unit tests over
    hand-built `CvResult`s.
- **CodeGraph/Read evidence:** `CvResult { iterations, columns: BTreeMap<String,
  Vec<f64>> }` + column names `test-<M>-mean` `[DEPENDENCY CONTRACT: ORCH-01 SPEC
  §4:249-281]`; `parse_metric` (`cb-train/src/lib.rs:71`); `metric_is_max_optimal`
  (TASK-01); `CatBoostError::Train(#[from] CbError)` (`?`-convert a `CbResult`).
- **Algorithm (implement exactly):**
  - `score_candidate`: `let m = parse_metric(metric)?; let key =
    format!("test-{metric}-mean"); let col = cv.columns.get(&key).ok_or_else(||
    CbError::Degenerate(...))?; if col.is_empty() { return Err(...) }` then fold
    with `metric_is_max_optimal(&m)`: max ⇒ `col.iter().copied().fold(f64::MIN,
    f64::max)`; min ⇒ `fold(f64::MAX, f64::min)`. (Guard against NaN by rejecting a
    non-finite result with a typed error.) No `unwrap`/indexing — `.get`, `?`.
  - `select_best`: `if cv_results.is_empty() { return Err(...) }`; `let m =
    parse_metric(metric)?; let max = metric_is_max_optimal(&m);` iterate
    `cv_results.iter().enumerate()`, compute each `score_candidate(cv, metric)?`,
    track the best with a strict `<`/`>` comparison so the FIRST (lowest-index)
    candidate wins ties. Return the tracked index.
- **Red:** in `grid_search_test.rs`:
  - `score_min_metric` — a `CvResult` with `columns = {"test-RMSE-mean":
    [0.9,0.5,0.6]}` ⇒ `score_candidate(&cv, "RMSE")? == 0.5` (≤1e-12).
  - `score_max_metric` — `{"test-NDCG-mean": [0.2,0.8,0.6]}` ⇒
    `score_candidate(&cv, "NDCG")? == 0.8`.
  - `select_best_argmin_and_tiebreak` — three `CvResult`s with RMSE-mean best
    scores `[0.5,0.4,0.7]` ⇒ `select_best(&v, "RMSE")? == 1`; a tie
    `[0.4,0.4,0.7]` ⇒ `== 0` (lowest index).
  - `score_missing_column_errs` — a `CvResult` lacking `test-RMSE-mean` ⇒
    `score_candidate(...).is_err()`; `select_best(&[], "RMSE").is_err()`.
  Expected INITIAL failure: `score_candidate`/`select_best`/`CvResult` unresolved
  ⇒ test build fails (this is ALSO the G-ORCH-01 gate signal: it cannot even build
  until `CvResult` lands).
- **Green:** implement per the algorithm; the four tests pass.
- **Refactor:** extract `best_over_iters(col: &[f64], max: bool) -> Option<f64>`;
  reuse in both fns. No behavior change; regression scope: `grid_search_test.rs` +
  TASK-01's S1 tests still green.
- **Validation:**
  - `cargo test -p catboost-rs --lib grid_search`
  - `cargo clippy -p catboost-rs --lib --no-deps`
- **Completion evidence:** 4 S2 tests green; clippy clean.
- **Compat/rollback:** additive; rollback = remove the two fns + tests.
- **Parallelization:** sequential after TASK-01 (same prod file) AND after
  G-ORCH-01 (`CvResult`). Not parallel with TASK-03/04.

---

## TASK-03 — Rust facade `grid_search` + oracle (ORCH-02-S3)

- **Spec refs:** ORCH-02-S3 (AT-S3a self-consistency + AT-S3b refit/errors).
  Primary failure reason: orchestration/selection is wrong, or `cv_results` /
  refit diverges from the direct `cv`/`fit` composition.
- **Blocked by:** TASK-02 **AND** external gate **G-ORCH-01 (`cv` function must
  exist — ORCH-01 TASK-06)**. `[DEPENDENCY CONTRACT: ORCH-01; VERIFIED: LOCAL cv
  PLAN.md:385-434]`
- **Goal / completion:** `catboost_rs::grid_search(pool, candidates, metrics,
  fold_count, shuffle, partition_random_seed, inverted, folds, refit) ->
  Result<SearchResult, CatBoostError>` + `pub struct SearchResult` exist; the new
  oracle test passes; clippy clean.
- **Files:**
  - Modify: `crates/catboost-rs/src/grid_search.rs` — add `pub struct SearchResult`
    (`best_index`, `best_builder: CatBoostBuilder`, `cv_results: CvResult`,
    `best_model: Option<Model>`; derive only `Debug` — `Model` is not `PartialEq`)
    and `pub fn grid_search`. Body: validate `!candidates.is_empty()` &&
    `!metrics.is_empty()` (else typed `Degenerate`); `let cvs: Vec<CvResult> =
    candidates.par_iter().map(|b| cv(pool, b, metrics, fold_count, shuffle,
    partition_random_seed, inverted, folds)).collect::<Result<_,_>>()?;` (order-
    preserving); `let primary = metrics.first().ok_or_else(|| CbError::Degenerate(
    "metrics must be non-empty"))?; let best = select_best(&cvs, primary)?;`
    (checked access ONLY — no raw `metrics[0]` indexing anywhere, per the
    crate's `indexing_slicing`-deny guardrail in §3); build `best_builder =
    candidates.get(best).ok_or_else(...)?.clone()` (checked, not `candidates[best]`);
    `let best_model = if refit { Some(best_builder.fit(pool)?) } else { None };`.
    No `unwrap`/indexing anywhere in this body.
  - Modify: `crates/catboost-rs/src/lib.rs` — extend re-export to `pub use
    grid_search::{metric_is_max_optimal, grid_search, SearchResult};`.
  - Create: `crates/catboost-rs/tests/grid_search_oracle_test.rs` — integration
    (carries `#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic,
    clippy::indexing_slicing)]` as the other integration tests do).
- **CodeGraph/Read evidence:** `cv` signature + `CvResult` `[DEPENDENCY CONTRACT:
  ORCH-01 SPEC §4]`; `CatBoostBuilder::fit` (`builder.rs:334`) + `Clone`
  (`builder.rs:64`); `select_best` (TASK-02); `rayon` dep (`Cargo.toml:39`) +
  order-preserving `collect` (cv PLAN `:405-409`); the ORCH-01 `cv_oracle_test.rs`
  harness shape (`load_f64_vec`, build a `Pool` from `cv/{X,y}.npy`) — reuse the
  SAME ORCH-01 `cv/` fixtures (numeric-regression Pool + explicit `folds.json` +
  `params.json`) `[VERIFIED: LOCAL cv PLAN.md:125-176,410-419]`. No NEW fixtures
  required (self-consistency oracle).
- **Red:** `grid_search_oracle_test.rs`:
  - reuse ORCH-01's `cv/{X,y}.npy` + `folds.json` to build a `Pool` and an explicit
    `folds: Vec<Vec<usize>>`; build 3 candidate `CatBoostBuilder`s from
    `params.json` differing only in `depth ∈ {2,4,6}`;
  - `let res = grid_search(&pool, &cands, &["RMSE"], 3, false, 0, false,
    Some(&folds), false)?;`
  - `SELF-CONSISTENCY`: `let direct = cv(&pool, &cands[res.best_index], &["RMSE"],
    3, false, 0, false, Some(&folds))?;` assert `res.cv_results.columns ==
    direct.columns` (bytewise; same seams) — or `compare_stage(Stage::Predictions,
    &direct.columns["test-RMSE-mean"], &res.cv_results.columns["test-RMSE-mean"])`
    ≤1e-5 for each column;
  - `ARGBEST`: independently `cv` each candidate, `select_best` over those results,
    assert `== res.best_index`;
  - `REFIT` (AT-S3b): `let res2 = grid_search(..., refit=true)?;` assert
    `res2.best_model` is `Some`, and its `predict(&pool)` equals
    `cands[res2.best_index].fit(&pool)?.predict(&pool)` ≤1e-9;
  - `ERRORS`: `grid_search(&pool, &[], &["RMSE"], ...).is_err()` and
    `grid_search(&pool, &cands, &[], ...).is_err()`.
  Expected INITIAL failure: `grid_search`/`SearchResult`/`cv` unresolved ⇒ build
  fails (also the G-ORCH-01(`cv`) gate signal); then value mismatch if selection
  is wrong.
- **Green:** implement `grid_search`; self-consistency holds to machine precision
  (identical seams), argbest matches, refit matches, errors typed.
- **Refactor:** `grid_search` = validate → per-candidate `cv` → `select_best` →
  optional `fit`; each step one call. No new float math (selection reuses TASK-02;
  no `sum_f64` needed here). Regression scope: `cargo test -p catboost-rs` (facade
  + grid oracle) green.
- **Validation:**
  - `cargo test -p catboost-rs --test grid_search_oracle_test`
  - `cargo test -p catboost-rs --lib grid_search`
  - `cargo clippy -p catboost-rs --lib --no-deps`
- **Completion evidence:** oracle green (self-consistency ≤1e-5, argbest exact,
  refit ≤1e-9); empty-candidates/empty-metrics typed error; clippy clean.
- **Compat/rollback:** additive; rollback = remove `grid_search`+`SearchResult`+
  oracle test + revert the re-export.
- **Parallelization:** sequential after TASK-02 (same prod file) AND G-ORCH-01(cv).

---

## TASK-04 — Rust facade `randomized_search` (ORCH-02-S4)

- **Spec refs:** ORCH-02-S4. Primary failure reason: the n_iter subsampling is
  non-deterministic, samples the wrong indices, or the best index is not mapped
  back to the original slice.
- **Blocked by:** TASK-03 (same prod file; reuses the grid_search mechanism) +
  G-ORCH-01 (transitively, via `cv`).
- **Goal / completion:** `catboost_rs::randomized_search(pool, candidates, metrics,
  n_iter, fold_count, shuffle, partition_random_seed, inverted, folds, refit) ->
  Result<SearchResult, CatBoostError>` exists; subsampling unit tests + one
  integration case pass; clippy clean.
- **Files:**
  - Modify: `crates/catboost-rs/src/grid_search.rs` — add `use
    cb_train::fisher_yates_permutation;` + `pub fn randomized_search`. Body:
    validate `n_iter > 0` && `!candidates.is_empty()`; `let take = n_iter.min(
    candidates.len()); let perm = fisher_yates_permutation(candidates.len(),
    partition_random_seed);` (perm is `Vec<i32>` — cast each to `usize` with a
    checked `usize::try_from`); take the first `take` permuted indices as the
    sampled set `S`; build `sampled: Vec<CatBoostBuilder> = S.iter().map(|&i|
    candidates.get(i).cloned())…` (checked); run the SAME cv-per-candidate +
    `select_best` over `sampled`; map the sampled-best position back to the
    ORIGINAL index via `S[best_pos]`; refit `candidates[orig_best]` if `refit`.
    Reuse a shared private `fn run_over(pool, cands, metrics, …, refit) ->
    Result<(usize, CvResult, Option<Model>), CatBoostError>` extracted from
    TASK-03's `grid_search` so both facades share the loop (no duplication).
  - Modify: `crates/catboost-rs/src/grid_search_test.rs` — S4 subsampling unit
    tests (pure, over the permutation — do NOT need training).
  - Modify: `crates/catboost-rs/tests/grid_search_oracle_test.rs` — one
    integration case (`n_iter < M` end-to-end reusing the TASK-03 harness).
  - Modify: `crates/catboost-rs/src/lib.rs` — extend re-export to add
    `randomized_search`.
- **CodeGraph/Read evidence:** `fisher_yates_permutation(n, seed) -> Vec<i32>`
  (`cb-train/src/lib.rs:75`; deterministic via `TFastRng64::from_seed`,
  `rng.rs:171`); `select_best`/`SearchResult`/`grid_search` shared loop (TASK-02/03);
  order note: the permutation is the ONLY randomness (candidate order preserved).
- **Red:** in `grid_search_test.rs` (subsampling is pure — testable WITHOUT cv):
  - `sample_is_first_n_of_permutation` — for `M=6, seed=0, n_iter=3`, assert the
    sampled index set equals the first 3 of `fisher_yates_permutation(6, 0)`
    (call the same fn in the test). (If the sampling is a private helper, expose a
    `#[cfg(test)]`-visible `pub(crate) fn sample_indices(m, n_iter, seed) ->
    Vec<usize>` and assert on it directly — the failure-isolated seam.)
  - `sample_deterministic` — same `(M, n_iter, seed)` twice ⇒ equal index sets;
    a different seed ⇒ (generally) different set.
  - `sample_full_when_niter_ge_m` — `n_iter >= M` ⇒ the sample is all `M` indices
    (as a set).
  - `sample_zero_or_empty_errs` — `n_iter == 0` or empty candidates ⇒ the facade
    returns `.is_err()` (checked in the integration file where a Pool exists, or a
    pure guard test on the validate front-door).
  Then in `grid_search_oracle_test.rs`: `randomized_search(&pool, &cands, &["RMSE"],
  2, 3, false, 0, false, Some(&folds), false)?` evaluates exactly the 2 sampled
  candidates and its `best_index` is one of them, mapped to the original slice;
  same seed ⇒ identical `best_index`+`cv_results`.
  Expected INITIAL failure: `randomized_search`/`sample_indices` unresolved ⇒ build
  fails.
- **Green:** implement the sampling + shared `run_over`; tests pass.
- **Refactor:** ensure `grid_search` is `randomized_search` with
  `n_iter = candidates.len()` (or a direct call to the shared `run_over` over all
  candidates) — dedupe. No behavior change; regression scope: all grid_search unit
  + oracle green.
- **Validation:**
  - `cargo test -p catboost-rs --test grid_search_oracle_test`
  - `cargo test -p catboost-rs --lib grid_search`
  - `cargo clippy -p catboost-rs --lib --no-deps`
- **Completion evidence:** S4 subsampling tests green (determinism, first-n,
  full-when-ge, error); integration case green; clippy clean.
- **Compat/rollback:** additive; rollback = remove `randomized_search`+its tests +
  revert re-export.
- **Parallelization:** sequential after TASK-03 (same prod file).

---

## TASK-05 — Python `catboost_rs.grid_search` (ORCH-02-S5)

- **Spec refs:** ORCH-02-S5. Primary failure reason: the Python layer expands the
  grid wrong, returns the wrong dict shape, refits incorrectly, or a bad param
  aborts instead of raising.
- **Blocked by:** TASK-03 (needs the Rust `grid_search` facade only — this task
  never calls `randomized_search`) + G-ORCH-01 (transitively). Corrected from an
  earlier "TASK-04" mis-statement (PLAN-CHECK MINOR finding): TASK-05 writes to
  `crates/catboost-rs-py/src/search.rs`, a different file than TASK-04's
  `crates/catboost-rs/src/grid_search.rs`, so it may run in PARALLEL with TASK-04.
- **Goal / completion:** `catboost_rs.grid_search(estimator, param_grid, X, y=None,
  cv=3, partition_random_seed=0, shuffle=True, inverted=False, folds=None,
  metrics=None, refit=True)` returns `{"params": <best grid point>, "cv_results":
  {columns}}`; `cargo check -p catboost-rs-py` compiles; parity/structure ≤1e-5
  under the uv-3.12 venv.
- **Files:**
  - Create: `crates/catboost-rs-py/src/search.rs` — a `#[pyfunction] grid_search`.
    Steps: `data_to_pool(py, x, y)?`; read the estimator's base params
    (`EstimatorBase.params: BTreeMap<String, Py<PyAny>>`); expand `param_grid`
    (dict[str, list]) into the Cartesian product of param-override dicts (stable
    key order — sort keys for determinism); for EACH grid point, `let merged =
    base.clone(); merged.extend(grid_point); validate_params(&merged)?; let
    builder = make_builder(&merged, py)?;` → `Vec<CatBoostBuilder>` + a parallel
    `Vec<grid_point_dict>`; derive the metric list (`metrics` `str`/`list[str]`, or
    from `params["loss_function"]` default `RMSE`); convert `folds`
    (`Option<Vec<Vec<usize>>>`); `py.detach(|| catboost_rs::grid_search(&pool,
    &builders, &metric_refs, cv, shuffle, partition_random_seed, inverted,
    folds.as_deref(), refit))` then `.map_err(PyCbError)?`; build the return
    `dict`: `"params"` = `grid_points[res.best_index]`, `"cv_results"` = a dict of
    `iterations` + each column list from `res.cv_results`; if `refit`, set the
    estimator's params to the winning grid point and `fit` it on `(X, y)` (or
    attach `res.best_model`) per SPEC §9 Q3.
  - Modify: `crates/catboost-rs-py/src/lib.rs` — `mod search;` +
    `m.add_function(wrap_pyfunction!(search::grid_search, m)?)?;` near the
    `sum_models` registration (`lib.rs:57`).
  - Create: `crates/catboost-rs-py/tests/test_grid_search.py` — parity/structure +
    error tests.
- **CodeGraph/Read evidence:** `sum_models` free `#[pyfunction]` pattern
  (`regressor.rs:277-301`); `make_builder`/`validate_params` (`params.rs:451,290`);
  `data_to_pool` (`estimator.rs:236`); `EstimatorBase.params` map (`estimator.rs:24`);
  `PyCbError`/`to_pyerr` chokepoint; GIL own-before-detach discipline (copy Python
  buffers into Rust-owned `Vec`s before `py.detach`). `[VERIFIED: CODEGRAPH]`
- **Red:** `test_grid_search.py`:
  - `import catboost_rs` + `catboost_rs.grid_search` resolves;
  - structure: `res = catboost_rs.grid_search(reg, {"depth":[4,6],
    "learning_rate":[0.1,0.3]}, X, y, cv=3, folds=<fixed>, shuffle=False)` has keys
    `{"params","cv_results"}`; `res["cv_results"]` has `test-RMSE-mean` of the
    right length; `res["params"]` is one of the 4 grid points;
  - self-consistency: `catboost_rs.cv(reg-with-best-params, X, y, folds=<fixed>,
    shuffle=False)["test-RMSE-mean"]` ≈ `res["cv_results"]["test-RMSE-mean"]` ≤1e-5;
  - errors: `pytest.raises` on a KnownNotYet/unknown param in the grid, and on a
    categorical Pool (UnsupportedModel → mapped exception).
  If the uv venv is unavailable in-session, the equivalent red is `cargo check -p
  catboost-rs-py` failing to resolve `catboost_rs::grid_search` before TASK-03,
  then a compile-verified binding.
- **Green:** implement the pyfunction + registration; grid expansion + dict return
  + refit; map errors through `PyCbError`.
- **Refactor:** extract a `expand_param_grid(base, grid) -> (Vec<CatBoostBuilder>,
  Vec<PyDict>)` helper; reuse in TASK-06. Regression scope: `cargo check -p
  catboost-rs-py`; existing Python tests unaffected (additive function).
- **Validation:**
  - `cargo check -p catboost-rs-py`
  - Under uv 3.12: `uv venv --python 3.12 && uv pip install catboost==1.2.10
    'numpy<2' scikit-learn maturin pytest` then `maturin develop` + `pytest
    crates/catboost-rs-py/tests/test_grid_search.py`.
- **Completion evidence:** `cargo check` clean; Python dict shape correct;
  cv_results self-consistency ≤1e-5; bad-param `pytest.raises`.
- **Compat/rollback:** additive; rollback = remove `search.rs` + the `lib.rs`
  registration + the test.
- **Parallelization:** sequential after TASK-03; PARALLEL with TASK-04 (disjoint
  files, neither's Green step consumes the other's output). TASK-06 remains
  sequential after BOTH this task and TASK-04.

---

## TASK-06 — Python `catboost_rs.randomized_search` (ORCH-02-S6)

- **Spec refs:** ORCH-02-S6. Primary failure reason: sampling is non-deterministic
  across the FFI boundary, wrong subset size, or wrong dict shape.
- **Blocked by:** TASK-04 **AND** TASK-05 (needs `randomized_search` from
  TASK-04 AND reuses TASK-05's `expand_param_grid` helper / `search.rs` file —
  corrected after PLAN-CHECK pass 2 caught this line and the §1 ASCII graph
  both still implying a linear TASK-04→TASK-05→TASK-06 chain, which contradicted
  the already-corrected wave graph and TASK-05's own parallelization note).
- **Goal / completion:** `catboost_rs.randomized_search(estimator,
  param_distributions, X, y=None, cv=3, n_iter=10, partition_random_seed=0,
  shuffle=True, inverted=False, folds=None, metrics=None, refit=True)` returns the
  same `{"params","cv_results"}` dict; `cargo check -p catboost-rs-py` compiles;
  determinism verified under uv-3.12.
- **Files:**
  - Modify: `crates/catboost-rs-py/src/search.rs` — add `#[pyfunction]
    randomized_search`, reusing `expand_param_grid` (TASK-05) to build the full
    candidate product, then delegating to `catboost_rs::randomized_search(&pool,
    &builders, &metric_refs, n_iter, cv, shuffle, partition_random_seed, inverted,
    folds.as_deref(), refit)`; build the SAME return dict (`"params"` =
    `grid_points[res.best_index]`).
  - Modify: `crates/catboost-rs-py/src/lib.rs` — register
    `wrap_pyfunction!(search::randomized_search, m)`.
  - Create: `crates/catboost-rs-py/tests/test_randomized_search.py` — determinism +
    subset-size + shape tests.
- **CodeGraph/Read evidence:** TASK-05's `search.rs` scaffold + `expand_param_grid`;
  `catboost_rs::randomized_search` (TASK-04); registration pattern (`lib.rs:57`).
- **Red:** `test_randomized_search.py`:
  - `catboost_rs.randomized_search` resolves;
  - determinism: two calls with the SAME `partition_random_seed` on the same grid
    give identical `res["params"]` and `res["cv_results"]`;
  - subset behavior: with `n_iter=2` over a 12-point grid, the run completes and
    `res["params"]` is a valid grid point; `n_iter >= grid_size` behaves like
    `grid_search` (same best as a `grid_search` call);
  - shape: `{"params","cv_results"}` present.
  If uv unavailable, red = `cargo check -p catboost-rs-py` failing to resolve
  `catboost_rs::randomized_search` before TASK-04.
- **Green:** implement the pyfunction + registration; determinism holds (all
  randomness is the Rust `TFastRng64` seeded by `partition_random_seed`).
- **Refactor:** dedupe the dict-building with TASK-05 (`fn result_to_pydict(py,
  res, grid_points) -> PyResult<Py<PyDict>>`). Regression scope: `cargo check -p
  catboost-rs-py`; existing tests unaffected.
- **Validation:**
  - `cargo check -p catboost-rs-py`
  - Under uv 3.12: `maturin develop` + `pytest
    crates/catboost-rs-py/tests/test_randomized_search.py`.
- **Completion evidence:** `cargo check` clean; determinism + subset + shape green.
- **Compat/rollback:** additive; rollback = remove `randomized_search` + the
  `lib.rs` registration + the test.
- **Parallelization:** sequential after BOTH TASK-04 (needs `randomized_search`)
  AND TASK-05 (shares `search.rs` + `expand_param_grid`) — corrected to match
  this task's own "Blocked by" line (PLAN-CHECK pass 3 found this line had not
  been updated in step, still reading "after TASK-05" alone).

---

## 3. Cross-cutting guardrails (apply to every Rust task)

- **HARD external gate G-ORCH-01:** TASK-02..06 MUST NOT begin until ORCH-01's
  `CvResult` (its TASK-04) and `cv` (its TASK-06) are merged and green. TASK-01 is
  the only task safe to author before then. Treat `cv`/`CvResult` as a
  `[DEPENDENCY CONTRACT]`, never verified working code, until ORCH-01's oracle
  passes. `[VERIFIED: LOCAL crates/catboost-rs/src/lib.rs:27-30 no cv today;
  cv PLAN.md:289-434]`
- **Clippy gate, not build:** `unwrap`/`expect`/`panic`/`indexing_slicing` are
  DENY in prod. Gate each Rust prod change with `cargo clippy -p catboost-rs --lib
  --no-deps`. Integration tests carry `#![allow(clippy::unwrap_used,
  clippy::expect_used, clippy::panic, clippy::indexing_slicing)]`. `[VERIFIED:
  LOCAL MEMORY fstr03-plan gotchas; cv PLAN.md:491-497]`
- **Test mount:** `grid_search_test.rs` must be mounted at the facade crate root
  (`#[cfg(test)] mod grid_search_test;`, cf. `lib.rs:65 mod metrics_test;`) or
  `cargo test` silently runs 0 tests. `[VERIFIED: LOCAL lib.rs:63-69]`
- **D-08 summation:** grid/random search adds NO ad-hoc float sums; scoring is a
  min/max fold over an existing `cv` column (no mean/variance), and cv's own means
  already route through `cb_core::sum_f64`. If any averaging is ever added, route
  it through `sum_f64`.
- **D-04 no-regression:** `cv.rs`, `builder.rs`, `model.rs`, `metrics.rs`,
  `custom.rs`, `permutation.rs` are read-only. ORCH-02 only CALLS them. Confirm
  with `cargo test -p catboost-rs` after TASK-04.
- **Model-class scope:** the numeric/float-only bound is enforced by cv()'s
  existing `staged_predict`/`ensure_scalar_oblivious` guard — ORCH-02 must NOT
  re-implement it; it propagates the typed `UnsupportedModel`.
- **Determinism / no new RNG:** candidate subsampling reuses
  `cb_train::fisher_yates_permutation` (`TFastRng64`); NO `rand` crate is added.
  `[VERIFIED: LOCAL cb-train/src/lib.rs:75]`

## 4. Unresolved blockers / assumptions

1. **[BLOCKER] ORCH-01 `cv`/`CvResult` unshipped (SPEC §9).** The hard cross-plan
   gate. TASK-02..06 cannot compile until ORCH-01 TASK-04 (`CvResult`) + TASK-06
   (`cv`) land and its oracle is green. TASK-01 is unblocked and may proceed now.
   This is the single dominant risk; it is documented in the frontmatter
   (`depends_on_plan`), the wave graph (`G-ORCH-01`), and §3.
2. **cv column-key spelling (SPEC §9 Q1).** `score_candidate` builds
   `format!("test-{metric}-mean")`; if the shipped `cv()` canonicalizes descriptors
   (e.g. `"NDCG:top=2"` → `"NDCG"`), apply the SAME canonicalization. Resolved at
   TASK-02 Green by inspecting the real `cv()` output. Blocks TASK-02's exact key,
   not its shape.
3. **Upstream selection parity (SPEC §9 Q2).** Deferred non-goal; the first-slice
   oracle is decomposition-based (grid_search == cv-per-candidate + our documented
   argbest). Does not block any task.
4. **Python refit shape (SPEC §9 Q3).** In-place estimator refit vs returning the
   `best_model`; resolved at TASK-05 by inspecting the `EstimatorBase` mutation
   surface. Does not block TASK-01..04.
5. **`metrics=None` default (SPEC §9 Q4).** Derived from `loss_function` (`RMSE`);
   shares ORCH-01-S6's open question; confirm at TASK-05/06 Python-parity time.
6. **uv-3.12 venv availability in-session.** TASK-05/06 Python parity needs it; if
   unavailable, compile-verify via `cargo check -p catboost-rs-py` and defer the
   `pytest` run to the venv (FSTR-03 precedent). Not a correctness blocker for
   TASK-01..04.
7. No TreeFinder/PageIndex write target confirmed for this corpus (SPEC frontmatter
   `pageindex_pending`); the SPEC/PLAN under `.planning/plans/…` are the effective
   spec store. Not a planning blocker.

No requirement conflicts detected. No production code was authored. ORCH-02 is
gated behind ORCH-01 shipping (documented above and in SPEC §9).
