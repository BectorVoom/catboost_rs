---
title: "ORCH-02 — grid_search / randomized_search hyperparameter search"
status: draft
format: markdown
spec_version: 1
updated_at: 2026-07-19T00:00:00Z
phase: 20-orchestration
slice: grid-random-search
source_requirements:
  - "User: Draft SPEC+PLAN for ORCH-02 grid/randomized search, mirroring upstream CatBoost.grid_search(param_grid, X, y, cv=3, ...) and .randomized_search(param_distributions, X, y, cv=3, n_iter=10, ...)."
  - "User: HARD-DEPENDS on ORCH-01 cv() (sibling draft, unimplemented) — treat cv()/CvResult/CatBoostBuilder as an upstream DEPENDENCY CONTRACT (ORCH-01-S1..S6), NOT verified working code; gate ORCH-02 impl behind ORCH-01 shipping."
  - "Research: Phase-20 orchestration research pass (ORCH-01/02/03); ORCH-02 has no prior .planning scoping anywhere; grid/randomized_search are Python-only upstream (no direct C++/Rust equivalent)."
  - "Sibling precedent (shipped): .planning/phases/20-orchestration/calc-metrics/SPEC.md (ORCH-04). Sibling draft (unshipped): .planning/plans/cv-cross-validation/SPEC.md (ORCH-01)."
pageindex_pending:
  reason: "No TreeFinder/PageIndex write target confirmed in-session for the catboost-rs planning corpus; this SPEC is authored locally under .planning/plans/ (the effective spec store, matching the ORCH-01/ORCH-04 sibling convention)."
  intended_identifier: "catboost-rs / .planning/plans/grid-random-search/SPEC.md"
---

# ORCH-02 — `grid_search` / `randomized_search` Hyperparameter Search

> Draft specification. NOT approved, accepted, final, or implemented.
> Evidence tags: `[VERIFIED: CODEGRAPH …]`, `[VERIFIED: LOCAL <path>]`,
> `[VERIFIED: WEB <url>]`, `[INFERRED: …]`, `[UNVERIFIED: …]`,
> `[DEPENDENCY CONTRACT: ORCH-01 …]` (a draft, unimplemented sibling contract we
> build on but MUST NOT treat as verified working code).

---

## 1. Context

`catboost_rs` is a Rust rewrite of CatBoost, oracle-tested ≤10⁻⁵ against the
original C++ library, with a dual Rust + Python surface. Upstream exposes two
**Python-only** hyperparameter-search methods with **no direct C++/Rust
equivalent** (unlike `cv()`, which has a conceptual C++-adjacent origin):

- `CatBoost.grid_search(param_grid, X, y=None, cv=3, partition_random_seed=0,
  calc_cv_statistics=True, search_by_train_test_split=True, refit=True,
  shuffle=True, stratified=None, train_size=0.8, verbose=True, plot=False, …)`
- `CatBoost.randomized_search(param_distributions, X, y=None, cv=3, n_iter=10,
  …)`

Both return `{"params": <best_params>, "cv_results": {…}}` and, when
`refit=True`, refit the estimator on the full data with the best params
`[VERIFIED: WEB https://catboost.ai/en/docs/concepts/python-reference_catboost_grid_search
and .../python-reference_catboost_randomized_search — return dict {"params","cv_results"};
refit default True; searches the Cartesian product (grid) / n_iter samples (random) by
cv score]`. Each is, mechanically, a loop that **clones the estimator, mutates
one hyperparameter combination at a time, cross-validates it, and keeps the best
mean cv score** `[INFERRED from upstream semantics + WEB docs]`.

Every primitive ORCH-02 needs already exists OR is being introduced by the
sibling ORCH-01 slice:

- **Cheap clone-and-mutate of a candidate:** `CatBoostBuilder` is
  `#[derive(Debug, Clone, PartialEq)]` — exactly the mechanism a search needs to
  vary one hyperparameter per candidate `[VERIFIED: CODEGRAPH
  crates/catboost-rs/src/builder.rs:64-85]`.
- **Per-candidate cross-validation:** the ORCH-01 facade `catboost_rs::cv(pool,
  builder, metrics, fold_count, shuffle, partition_random_seed, inverted, folds)
  -> Result<CvResult, CatBoostError>` returning `CvResult { iterations:
  Vec<usize>, columns: BTreeMap<String, Vec<f64>> }` with columns
  `test-<Metric>-mean`, `test-<Metric>-std`, `train-<Metric>-mean`,
  `train-<Metric>-std` `[DEPENDENCY CONTRACT: ORCH-01-S5 / ORCH-01 SPEC §4;
  .planning/plans/cv-cross-validation/SPEC.md:210-281]`. **Not yet implemented:**
  `cv`/`CvResult`/`CvFold`/`CvType`/`make_cv_folds` are ABSENT from the facade
  crate root — `crates/catboost-rs/src/lib.rs:27-30` re-exports only
  `CatBoostBuilder`, `CatBoostError`, `eval_metric`/`eval_metrics` (ORCH-04),
  `Model` `[VERIFIED: LOCAL grep crates/catboost-rs/src/lib.rs — no `cv`, no
  `CvResult`]`.
- **Refit on the full pool:** `CatBoostBuilder::fit(&self, pool: &Pool) ->
  Result<Model, CatBoostError>` `[VERIFIED: CODEGRAPH
  crates/catboost-rs/src/builder.rs:334]`.
- **Metric-descriptor → `EvalMetric` (to read the search-metric direction):**
  `cb_train::parse_metric(&str) -> CbResult<EvalMetric>` (ORCH-04, shipped;
  re-exported) `[VERIFIED: LOCAL crates/cb-train/src/lib.rs:71]`.
- **Deterministic candidate subsampling for random search:**
  `cb_train::fisher_yates_permutation(n, seed) -> Vec<i32>` over the project's own
  `cb_core::rng::TFastRng64` (PCG-XSH-RR; no new `rand` dependency)
  `[VERIFIED: LOCAL crates/cb-train/src/lib.rs:75; CODEGRAPH
  crates/cb-core/src/rng.rs:171,265]`.
- **Python parameter-mutation surface:** `make_builder(params: &BTreeMap<String,
  Py<PyAny>>, py) -> PyResult<CatBoostBuilder>` converts a kwargs dict into a
  `CatBoostBuilder` through the VOCABULARY/alias-checked registry — the exact
  surface the Python layer uses to expand a param grid into candidate builders
  `[VERIFIED: CODEGRAPH crates/catboost-rs-py/src/params.rs:451`; validation
  `params.rs:290]`.

**Load-bearing gap (new work ORCH-02 must add):** there is **no metric
optimization-direction accessor on `EvalMetric`**. `is_max_optimal` (larger vs
smaller is better) exists ONLY on the `cb_compute::CustomMetric` trait
(`custom.rs:111-114`, `true`=AUC-like maximize, `false`=RMSE/Logloss minimize)
and as prose in the `EvalMetric` variant docs — the enum itself exposes no
`is_max_optimal`/`best_is_max` method `[VERIFIED: CODEGRAPH
crates/cb-compute/src/custom.rs:92-115; crates/cb-train/src/metrics.rs:77,82,86 —
doc comments only; LOCAL grep "is_max_optimal|best_is_max|higher_is_better" over
crates/cb-train/src, crates/catboost-rs/src — no method]`. Selecting the best
candidate therefore requires a small NEW direction helper (ORCH-02-S1).

**Scope decision (locked — the open research question resolved):** ORCH-02 ships
**BOTH** a thin Rust facade AND a Python surface, consistent with this project's
`sum_models` / `staged_predict` / CoreML precedent of always adding the Rust
facade as this project's own additive value even when upstream is Python-only
`[VERIFIED: LOCAL crates/catboost-rs-py/src/regressor.rs:277 sum_models is a Rust
facade fn + Python wrapper; MEMORY next-features-5plan-batch]`. Because Rust has
no dict-based param-grid concept, the Rust facade is parametrized over a
**caller-supplied `&[CatBoostBuilder]` candidate list** + a metric; the Python
layer expands `param_grid` (Cartesian product) / `param_distributions` (sampled)
into that candidate list via `make_builder`. This keeps the failure-isolating
value-selection + refit logic in Rust (unit-testable, `unwrap`-free) and the
dict/product ergonomics in Python.

**Crate placement (locked scope decision):** all ORCH-02 logic — the direction
helper, candidate scoring/selection, the `grid_search`/`randomized_search` free
functions — lives in a **new `crates/catboost-rs/src/grid_search.rs` facade
module**, surfaced through the Python bindings. No new crate. Rationale: it can
only be facade-level (it calls the facade-only `catboost_rs::cv` /
`CatBoostBuilder::fit` / `Model`; `cb-train`/`cb-model`/`cb-data` never depend on
the facade), and it is a pure-orchestration leaf over ORCH-01. This mirrors the
ORCH-01 `cv.rs` placement decision exactly
`[VERIFIED: LOCAL .planning/plans/cv-cross-validation/SPEC.md:81-93; CODEGRAPH
crates/catboost-rs/Cargo.toml:26-39 (facade → cb-core/cb-data/cb-train/cb-model +
rayon)]`.

---

## 2. Scope and Non-Goals

### In scope (first slice)

- A **metric optimization-direction helper** (new, pure): `metric_is_max_optimal(
  &EvalMetric) -> bool` — `true` iff a LARGER metric value is better (ranking
  metrics; AUC), `false` for the error metrics (RMSE/Logloss/MSLE/MAE/MAPE/
  Quantile); `Custom` delegates to the trait's `is_max_optimal`.
- A **candidate-scoring + best-selection** primitive (new, pure): reduce a
  candidate's `CvResult` to a single scalar "best cv score" for the primary
  (first) metric — the best value over iterations of its `test-<M>-mean` column
  (min if minimize, max if maximize) — and pick the argbest candidate across a
  slice of `CvResult`s (ties → lowest index, deterministic).
- A **Rust facade `grid_search`** (new glue): evaluate each supplied
  `CatBoostBuilder` candidate by calling `catboost_rs::cv(...)` once, select the
  best per the scoring primitive, and — when `refit=true` — refit the winning
  builder on the FULL pool via `.fit()`, returning a `SearchResult { best_index,
  best_builder, cv_results, best_model: Option<Model> }`.
- A **Rust facade `randomized_search`** (new glue): the SAME mechanism, but over a
  deterministically subsampled subset of `min(n_iter, candidates.len())`
  candidates chosen via `fisher_yates_permutation(candidates.len(),
  partition_random_seed)` (the project's own `TFastRng64`), then grid_search's
  select + refit.
- A **Python `catboost_rs.grid_search(estimator, param_grid, X, y=None, cv=3, …,
  refit=True)`** (new surface): expand `param_grid`'s Cartesian product into a
  `Vec<CatBoostBuilder>` via `make_builder` (base estimator params merged with
  each grid point), call the Rust facade, return `{"params": <best grid point>,
  "cv_results": {<columns>}}`; when `refit=True`, also fit the winner and return
  a fitted estimator (or set it as the result's model).
- A **Python `catboost_rs.randomized_search(estimator, param_distributions,
  X, y=None, cv=3, n_iter=10, …)`** (new surface): expand the distributions'
  Cartesian product into candidates, delegate to the Rust `randomized_search`
  (n_iter subsampling via `TFastRng64`), same return shape.
- **Self-consistency (decomposition) oracle** (primary): `grid_search`'s returned
  `cv_results` for candidate *i* equals `catboost_rs::cv(...)` called directly on
  candidate *i* (same seams ⇒ machine precision), and `best_index` equals the
  manual argbest of the per-candidate scores.
- **Determinism guarantees:** identical inputs (same pool, candidates, folds,
  seeds) ⇒ identical `best_index`, `cv_results`, and (bytewise) refit `Model`.

### Non-goals (explicit — documented, not silently dropped)

- **`stratified` / label-strata-balanced folds** — inherited non-goal from
  ORCH-01 (its `cv` first slice is non-stratified). Passing it through is a typed
  error `[DEPENDENCY CONTRACT: ORCH-01 SPEC §2 Non-goals]`.
- **`search_by_train_test_split=False` / train-test-split-only search mode**
  (upstream default splits by a single 80/20 train/test rather than full cv). The
  first slice always uses full cv via ORCH-01 `cv()`. `train_size` is ignored.
- **`plot` / `plot_file`** — no visualization surface exists.
- **`calc_cv_statistics=False`** — the first slice always computes cv statistics
  (that is the selection signal).
- **scipy-distribution `param_distributions`** (continuous distributions with a
  `.rvs()` sampler). The first slice accepts only **discrete value lists** per
  parameter (`{"depth": [4, 6, 8]}`), matching `param_grid`'s shape; a continuous
  distribution is a documented later slice. `[INFERRED — sklearn/CatBoost accept
  both; discrete-list is the minimal parity slice]`
- **Lazy random sampling** (never materializing the full product). The first
  slice expands the full candidate product in Python then subsamples `n_iter`
  in Rust via `TFastRng64` — a documented memory/simplicity tradeoff; true lazy
  per-parameter sampling is a later slice.
- **Model classes cv() cannot handle.** ORCH-01's `cv` (via `staged_predict`'s
  `ensure_scalar_oblivious` guard) supports only scalar/oblivious/float-only
  (numeric regression / binary) models. A CTR/categorical/multiclass candidate
  surfaces the typed `CatBoostError::UnsupportedModel`, propagated unchanged —
  never a wrong selection `[DEPENDENCY CONTRACT: ORCH-01 SPEC §2 Non-goals;
  CODEGRAPH crates/catboost-rs/src/model.rs staged_predict guard]`.
- **Upstream-exact best-candidate SELECTION parity** (byte-identical to
  `catboost.grid_search`'s internal choice) is NOT claimed: upstream's
  fold-assignment RNG and its by-best-iteration selection detail are not
  reproduced ≤1e-5 in-session (same deferral ORCH-01 makes for `shuffle=True`).
  The first-slice oracle is DECOMPOSITION-based (grid_search == cv-per-candidate +
  our documented argbest rule); upstream-selection parity is a later slice.
- **In-place estimator mutation as a METHOD** (`model.grid_search(...)`). The
  first slice ships a **free function** `catboost_rs.grid_search(estimator, …)`
  (matching the `sum_models` free-function precedent); the method-sugar form is a
  later slice `[VERIFIED: LOCAL crates/catboost-rs-py/src/regressor.rs:277
  sum_models is a free `#[pyfunction]`]`.
- **GPU** — search adds no GPU code; training rides whatever backend is compiled.

---

## 3. Dependencies

| Dependency | Kind | Evidence |
|-----------|------|----------|
| `catboost_rs::cv(pool, builder, metrics, fold_count, shuffle, partition_random_seed, inverted, folds) -> Result<CvResult, CatBoostError>` and `CvResult { iterations, columns }` | **cross-plan dependency (draft, unimplemented)** — per-candidate cv | `[DEPENDENCY CONTRACT: ORCH-01-S5 / SPEC §4; .planning/plans/cv-cross-validation/SPEC.md:210-281]` |
| Column-naming convention `test-<Metric>-mean` etc. produced by `cv()` | dependency (selection reads this key) | `[DEPENDENCY CONTRACT: ORCH-01-S4/S5; SPEC §4]` |
| `catboost_rs::CatBoostBuilder` `#[derive(Debug, Clone, PartialEq)]` + setters + `fit(&Pool)` | reuse (clone-and-mutate candidates; refit) | `[VERIFIED: CODEGRAPH crates/catboost-rs/src/builder.rs:64-85,334]` |
| `catboost_rs::Model` | reuse (refit result) | `[VERIFIED: LOCAL crates/catboost-rs/src/lib.rs:30]` |
| `cb_train::parse_metric(&str) -> CbResult<EvalMetric>` (ORCH-04, shipped) | reuse (metric string → variant for direction) | `[VERIFIED: LOCAL crates/cb-train/src/lib.rs:71]` |
| `cb_train::EvalMetric` variants (flat vs ranking partition; `Custom(CustomMetricHandle)`) | reuse (direction match arms) | `[VERIFIED: CODEGRAPH crates/cb-train/src/metrics.rs:64-151,533-541]` |
| `cb_compute::CustomMetric::is_max_optimal()` (via `EvalMetric::Custom` handle) | reuse (custom-metric direction) | `[VERIFIED: CODEGRAPH crates/cb-compute/src/custom.rs:111-114]` |
| `cb_train::fisher_yates_permutation(n, seed) -> Vec<i32>` over `TFastRng64` | reuse (deterministic n_iter subsampling) | `[VERIFIED: LOCAL crates/cb-train/src/lib.rs:75; CODEGRAPH crates/cb-core/src/rng.rs:171,265]` |
| `catboost_rs::CatBoostError` (`Train(#[from] cb_core::CbError)`, `UnsupportedModel`) | reuse (typed error surface) | `[VERIFIED: LOCAL cv PLAN.md:64-66; CODEGRAPH error.rs]` |
| `cb_data::Pool` (borrowed; passed through to `cv`/`fit`) | reuse | `[VERIFIED: LOCAL crates/catboost-rs/src/lib.rs:60]` |
| `catboost-rs-py`: `make_builder`, `validate_params`, `data_to_pool`, `EstimatorBase` (with `params: BTreeMap<String, Py<PyAny>>`), `PyCbError`/`to_pyerr`, `#[pyfunction]`+`wrap_pyfunction!` | new Python surface | `[VERIFIED: CODEGRAPH crates/catboost-rs-py/src/params.rs:451,290; regressor.rs:277-301; estimator.rs:24,236]` |
| `catboost==1.2.10` under `uv --python 3.12`, `numpy<2`, `scikit-learn` | new oracle-fixture generation / Python parity | `[VERIFIED: LOCAL MEMORY next-features-5plan-batch; cv PLAN.md:170]` |

**No new external crate** is required — every primitive is in-tree; candidate
subsampling reuses the project's own `TFastRng64` via `fisher_yates_permutation`,
not a new `rand` dependency `[VERIFIED: LOCAL CLAUDE.md Dependencies "use existing
capability first"]`.

**Hard cross-plan blocker (see §9):** ORCH-01 (`cv`/`CvResult`) is a sibling
**draft, unimplemented** slice. Its symbols do NOT yet exist in the facade
(`lib.rs:27-30`). Every ORCH-02 task that calls `cv()` is BLOCKED until ORCH-01
ships. ORCH-02's pure helpers (ORCH-02-S1 direction, ORCH-02-S2 selection over
hand-built `CvResult`s) can be authored against the ORCH-01 **type contract**
only once the `CvResult` type exists (i.e. once ORCH-01 TASK-04 lands `CvResult`).

---

## 4. Typed Contracts

New code lives in `crates/catboost-rs/src/grid_search.rs` (prod) with unit tests
in `crates/catboost-rs/src/grid_search_test.rs`, mounted via the facade crate's
root-mount idiom (`#[cfg(test)] mod grid_search_test;`, cf.
`crates/catboost-rs/src/lib.rs:65 mod metrics_test;`)
`[VERIFIED: LOCAL crates/catboost-rs/src/lib.rs:63-69]`.

```rust
// crates/catboost-rs/src/grid_search.rs  (ORCH-02-S1..S4)

use cb_data::Pool;
use crate::{CatBoostBuilder, CatBoostError, Model};
use crate::cv::CvResult;            // [DEPENDENCY CONTRACT: ORCH-01 CvResult]
use cb_train::{parse_metric, fisher_yates_permutation};

/// (ORCH-02-S1) Whether a LARGER value of `metric` is better.
///
/// `true` for the ranking metrics (`Ndcg/Dcg/Map/Mrr/Err/PFound/PrecisionAt/
/// RecallAt/QueryAuc`) — larger is better; `false` for the error metrics
/// (`Rmse/Logloss/Msle/Mae/Mape/Quantile`) — smaller is better; `Custom`
/// delegates to `CustomMetric::is_max_optimal`. Total function; no panic.
#[must_use]
pub fn metric_is_max_optimal(metric: &cb_train::EvalMetric) -> bool;

/// The result of a search: the winning candidate + its cv columns (+ refit model).
#[derive(Debug)]
pub struct SearchResult {
    /// Index into the supplied `candidates` slice of the winning builder.
    pub best_index: usize,
    /// A clone of the winning `CatBoostBuilder` (the selected hyperparameters).
    pub best_builder: CatBoostBuilder,
    /// The `cv()` output columns for the winning candidate (upstream `cv_results`).
    pub cv_results: CvResult,
    /// The model refit on the FULL pool with `best_builder` when `refit == true`;
    /// `None` otherwise. (Excluded from any `PartialEq`; `Model` is not `PartialEq`.)
    pub best_model: Option<Model>,
}

/// (ORCH-02-S2) Reduce one candidate's `CvResult` to its scalar "best cv score"
/// for the PRIMARY metric (`metric`, i.e. `metrics[0]`): the best value over
/// iterations of `columns["test-<metric>-mean"]` — the min if
/// `!metric_is_max_optimal`, the max if it is.
///
/// # Errors
/// [`CatBoostError::Train`] wrapping [`cb_core::CbError::Degenerate`] when the
/// expected `test-<metric>-mean` column is absent/empty, or `metric` fails to
/// `parse_metric`. Never panics.
pub fn score_candidate(cv: &CvResult, metric: &str) -> Result<f64, CatBoostError>;

/// (ORCH-02-S2) Index of the best candidate among per-candidate scores for the
/// primary `metric`. Picks the argmax when the metric is max-optimal, else the
/// argmin; ties resolve to the LOWEST index (deterministic).
///
/// # Errors
/// [`CatBoostError::Train`] on an empty `cv_results` slice or any per-candidate
/// scoring failure.
pub fn select_best(cv_results: &[CvResult], metric: &str) -> Result<usize, CatBoostError>;

/// (ORCH-02-S3) Grid search: cross-validate every candidate with `catboost_rs::cv`
/// and keep the best by the PRIMARY metric (`metrics[0]`). When `refit`, refit the
/// winner on the full `pool`.
///
/// - `candidates`: one `CatBoostBuilder` per hyperparameter combination (the
///   Python layer expands `param_grid` into this). Empty ⇒ typed error.
/// - `metrics`, `fold_count`, `shuffle`, `partition_random_seed`, `inverted`,
///   `folds`: forwarded verbatim to `catboost_rs::cv` per candidate.
///
/// # Errors
/// [`CatBoostError::Train`] for empty `candidates`/`metrics` or a bad partition;
/// [`CatBoostError::UnsupportedModel`] propagated from `cv`/`staged_predict` for an
/// out-of-class candidate; any training/eval failure surfaces typed. Never panics.
#[allow(clippy::too_many_arguments)]
pub fn grid_search(
    pool: &Pool,
    candidates: &[CatBoostBuilder],
    metrics: &[&str],
    fold_count: usize,
    shuffle: bool,
    partition_random_seed: u64,
    inverted: bool,
    folds: Option<&[Vec<usize>]>,
    refit: bool,
) -> Result<SearchResult, CatBoostError>;

/// (ORCH-02-S4) Randomized search: the same mechanism over a deterministic
/// subset of `min(n_iter, candidates.len())` candidates, chosen by
/// `fisher_yates_permutation(candidates.len(), partition_random_seed)` (project
/// `TFastRng64`). `best_index` in the returned [`SearchResult`] indexes the
/// ORIGINAL `candidates` slice (the sampled index is mapped back).
///
/// # Errors
/// As `grid_search`, plus a typed error when `n_iter == 0` or `candidates` empty.
#[allow(clippy::too_many_arguments)]
pub fn randomized_search(
    pool: &Pool,
    candidates: &[CatBoostBuilder],
    metrics: &[&str],
    n_iter: usize,
    fold_count: usize,
    shuffle: bool,
    partition_random_seed: u64,
    inverted: bool,
    folds: Option<&[Vec<usize>]>,
    refit: bool,
) -> Result<SearchResult, CatBoostError>;
```

```python
# catboost_rs.grid_search / .randomized_search (PyO3), mirroring the supported
# subset of CatBoost.grid_search / .randomized_search  (ORCH-02-S5, -S6)
def grid_search(
    estimator,                 # a fitted-or-unfitted catboost_rs estimator (base params source)
    param_grid,                # dict[str, list] -> Cartesian product of candidate param dicts
    X, y=None,                 # -> Pool via data_to_pool
    cv=3,                      # fold_count
    partition_random_seed=0,
    shuffle=True,
    inverted=False,
    folds=None,                # list[list[int]] test-row indices, or None
    metrics=None,              # str | list[str]; None -> from estimator loss_function (default RMSE)
    refit=True,
) -> dict:
    """Returns {"params": <best grid point dict>, "cv_results": {"iterations": [...],
    "test-<M>-mean": [...], ...}}. When refit=True the estimator is (re)fit on
    (X, y) with the best params before return."""

def randomized_search(
    estimator, param_distributions, X, y=None,
    cv=3, n_iter=10, partition_random_seed=0, shuffle=True, inverted=False,
    folds=None, metrics=None, refit=True,
) -> dict:
    """As grid_search, but samples n_iter candidates (deterministic, TFastRng64)."""
```

If a type does not yet exist it is labelled here as PROPOSED: `SearchResult`,
`metric_is_max_optimal`, `score_candidate`, `select_best`, `grid_search`,
`randomized_search` are all PROPOSED (new). `CvResult`/`cv` are the
`[DEPENDENCY CONTRACT: ORCH-01]` types (also not yet materialized).

---

## 5. Failure-Isolated Behavioral Specifications

Each spec has ONE primary reason a failing acceptance test would fail.

### ORCH-02-S1 — Metric optimization direction
- **Status:** implemented
- **Responsibility:** map an `EvalMetric` to its best-direction bool, nothing else.
- **Input:** `metric: &EvalMetric`. **Output:** `bool` (`true`=max is best).
- **Dependencies:** `EvalMetric` variants; `CustomMetric::is_max_optimal` (via the
  `Custom` handle).
- **Behavior:**
  - `Rmse`/`Logloss`/`Msle`/`Mae`/`Mape`/`Quantile` ⇒ `false` (minimize).
  - `Ndcg`/`Dcg`/`Map`/`Mrr`/`Err`/`PFound`/`PrecisionAt`/`RecallAt`/`QueryAuc`
    ⇒ `true` (maximize).
  - `Custom(h)` ⇒ `h.0.is_max_optimal()`.
- **Invariants:** total function; no panic; matches the documented convention
  (`custom.rs:92-114`, `metrics.rs:77,82,86`).
- **Acceptance test:** `grid_search_test.rs` unit — one assertion per variant
  class (min set, max set, custom-delegation via a tiny stub metric).
- **Non-goals:** does NOT parse strings (callers pass an `EvalMetric`; the string
  → metric hop is `cb_train::parse_metric`).
- **Traceability:** `custom.rs:111-114`; `metrics.rs:533-541` (ranking partition).

### ORCH-02-S2 — Candidate scoring + best selection
- **Status:** implemented
- **Responsibility:** reduce a `CvResult` to its primary-metric best cv score, and
  argbest across candidates; nothing else (no training).
- **Input:** `score_candidate(cv: &CvResult, metric: &str)`;
  `select_best(cv_results: &[CvResult], metric: &str)`. **Output:** `f64` / `usize`
  in `Result<_, CatBoostError>`.
- **Dependencies:** ORCH-02-S1; `cb_train::parse_metric`; the `CvResult` type
  `[DEPENDENCY CONTRACT: ORCH-01]`.
- **Behavior (Given/When/Then):**
  - Given a `CvResult` whose `columns["test-RMSE-mean"] = [0.9, 0.5, 0.6]`, When
    `score_candidate(cv, "RMSE")` (minimize), Then it returns `0.5` (the min over
    iterations).
  - Given three candidates with best-scores `[0.5, 0.4, 0.7]` for a minimize
    metric, When `select_best(&[c0,c1,c2], "RMSE")`, Then `Ok(1)`.
  - Given a maximize metric (e.g. a hand `NDCG` column `[0.2, 0.8, 0.6]`), Then
    `score_candidate` returns `0.8` and `select_best` argmaxes.
  - Ties (equal best scores) ⇒ the LOWEST index.
  - Missing `test-<metric>-mean` column / empty column / empty `cv_results` ⇒
    `Err(CatBoostError::Train(Degenerate))`.
- **Invariants:** no panic; deterministic tie-break; column key built as
  `format!("test-{metric}-mean")` MATCHING the ORCH-01 cv naming convention (a
  documented coupling — see §9 Q1).
- **Acceptance test:** `grid_search_test.rs` unit over hand-built `CvResult`s
  (min, max, tie, missing-column).
- **Traceability:** ORCH-02-S1; ORCH-01 SPEC §4 column names.

### ORCH-02-S3 — Rust facade `grid_search`
- **Status:** implemented
- **Responsibility:** orchestrate cv-per-candidate + select + optional refit;
  surface all misuse typed.
- **Input/Output:** the `grid_search(...) -> Result<SearchResult, CatBoostError>`
  contract from §4.
- **Dependencies:** `catboost_rs::cv` `[DEPENDENCY CONTRACT: ORCH-01]`,
  ORCH-02-S2, `CatBoostBuilder::fit`.
- **Behavior:**
  - Given a fixed numeric-regression `pool`, N candidate builders differing in one
    hyperparameter (e.g. `depth ∈ {2,4,6}`), `metrics=["RMSE"]`, and explicit
    `folds`, When `grid_search(..., refit=false)` runs, Then `cv_results` equals
    `catboost_rs::cv(pool, &candidates[best_index], ["RMSE"], …, Some(folds))`
    (same seams ⇒ machine precision), and `best_index == select_best(<per-candidate
    cv results>, "RMSE")`.
  - When `refit=true`, Then `best_model` is `Some(m)` where `m` is bytewise equal
    to `candidates[best_index].fit(pool)?` (deterministic refit on the FULL pool).
  - `candidates` empty or `metrics` empty ⇒ `Err(CatBoostError::Train(Degenerate))`.
  - An out-of-class candidate ⇒ `Err(CatBoostError::UnsupportedModel)` propagated
    from `cv`/`staged_predict` (never a wrong selection).
- **Invariants:** no `unwrap`/`panic`; per-candidate cv MAY run under `rayon`
  (`par_iter().map(cv).collect::<Result<Vec<_>,_>>()`) with an order-preserving
  `collect`, so the result is identical to the serial loop
  `[VERIFIED: LOCAL cv PLAN.md:405-409 rayon order-preservation precedent]`.
- **Acceptance test:** `grid_search_oracle_test.rs` (integration) — self-consistency
  vs `cv` + manual argbest + refit equality, on a small numeric Pool with explicit
  folds.
- **Traceability:** ORCH-01-S5 (`cv`); ORCH-02-S2; `builder.rs:334` (`fit`).

### ORCH-02-S4 — Rust facade `randomized_search`
- **Status:** implemented
- **Responsibility:** deterministic n_iter subsampling of candidates, then the
  grid_search select+refit mechanism.
- **Input/Output:** the `randomized_search(...) -> Result<SearchResult,
  CatBoostError>` contract from §4.
- **Dependencies:** `cb_train::fisher_yates_permutation`, ORCH-02-S3 mechanism.
- **Behavior:**
  - Given M candidates, `n_iter < M`, and a fixed `partition_random_seed`, When
    `randomized_search(...)` runs, Then it evaluates exactly the candidates whose
    indices are the first `n_iter` of `fisher_yates_permutation(M, seed)`, and the
    SAME seed yields the SAME sampled subset (determinism).
  - `n_iter >= M` ⇒ evaluates all candidates (equivalent to `grid_search`).
  - `best_index` in the result indexes the ORIGINAL `candidates` slice (sampled
    index mapped back).
  - `n_iter == 0` or `candidates` empty ⇒ `Err(CatBoostError::Train(Degenerate))`.
  - Refit behavior identical to ORCH-02-S3.
- **Invariants:** no panic; determinism seeded by `partition_random_seed`; NO new
  `rand` dependency (reuses `TFastRng64` via `fisher_yates_permutation`).
- **Acceptance test:** `grid_search_test.rs` unit for the subsampling
  (same-seed determinism, `n_iter>=M` = full, mapping-back), + one integration
  case in `grid_search_oracle_test.rs` reusing the S3 harness.
- **Traceability:** `permutation.rs:109` / `lib.rs:75`; ORCH-02-S3.

### ORCH-02-S5 — Python `catboost_rs.grid_search`
- **Status:** implemented
- **Responsibility:** PyO3 surface mirroring the supported subset of
  `CatBoost.grid_search`; expand `param_grid` → candidate builders; return the
  `{"params","cv_results"}` dict; refit.
- **Input:** `estimator`, `param_grid` (dict[str, list]), `X`, `y`, `cv`,
  `partition_random_seed`, `shuffle`, `inverted`, `folds`, `metrics`, `refit`.
  **Output:** `dict` (`{"params": <best grid point>, "cv_results": {columns}}`).
- **Dependencies:** `data_to_pool`, `make_builder`, `validate_params`,
  `catboost_rs::grid_search`, `PyCbError`; the estimator's base
  `params: BTreeMap<String, Py<PyAny>>`.
- **Behavior:**
  - `grid_search(reg, {"depth":[4,6], "learning_rate":[0.1,0.3]}, X, y, cv=3,
    folds=<fixed>, shuffle=False)` builds the 4-candidate Cartesian product
    (base params merged with each grid point via `make_builder`), returns a dict
    whose `cv_results["test-RMSE-mean"]` matches `catboost_rs.cv(...)` on the
    winner ≤1e-5, and whose `"params"` is the winning grid point.
  - `metrics=None` derives the metric from the estimator's `loss_function`
    (default `RMSE`); a `str` or `list[str]` is accepted.
  - `refit=True` ⇒ the estimator is (re)fit on `(X, y)` with the best params
    before return (mirrors upstream in-place refit).
  - A bad param / unsupported model raises a mapped `CatBoostError`/
    `CatBoostParameterError`, not a panic/abort (`make_builder`/`validate_params`
    already reject KnownNotYet/unknown params).
- **Acceptance test:** Python test under uv-3.12 / catboost 1.2.10 — structural
  (`{"params","cv_results"}` shape) + cv_results self-consistency vs
  `catboost_rs.cv` + a best-selection sanity check + `pytest.raises` on a bad
  param / categorical Pool.
- **Traceability:** `regressor.rs:277` (sum_models free-fn precedent);
  `params.rs:451,290`; `estimator.rs:236`; ORCH-01-S6 Python cv precedent.

### ORCH-02-S6 — Python `catboost_rs.randomized_search`
- **Status:** implemented
- **Responsibility:** PyO3 surface mirroring the supported subset of
  `CatBoost.randomized_search`; expand `param_distributions` → candidates;
  delegate to the Rust `randomized_search` (n_iter, seed).
- **Input:** as ORCH-02-S5 plus `n_iter` (default 10) and `param_distributions`
  (dict[str, list]). **Output:** the same `{"params","cv_results"}` dict.
- **Dependencies:** ORCH-02-S5 surface + `catboost_rs::randomized_search`.
- **Behavior:**
  - `randomized_search(reg, {"depth":[2,4,6,8], "l2_leaf_reg":[1,3,5]}, X, y,
    n_iter=5, partition_random_seed=0, folds=<fixed>, shuffle=False)` expands the
    12-candidate product, the Rust facade deterministically evaluates 5, and the
    returned `"params"` is one of the sampled grid points; the SAME seed ⇒ the
    SAME result.
  - `n_iter >= product_size` ⇒ equivalent to `grid_search`.
  - Same error mapping as ORCH-02-S5.
- **Acceptance test:** Python test under uv-3.12 — determinism (same seed ⇒ same
  `"params"`), `n_iter` subset size, structural shape.
- **Traceability:** ORCH-02-S4; ORCH-02-S5; `permutation.rs` determinism.

---

## 6. Acceptance Scenarios

| ID | Scenario | Oracle | Tolerance |
|----|----------|--------|-----------|
| AT-S1 | `metric_is_max_optimal` correct for each metric class (min set, max set, Custom delegation) | unit (hand) | exact |
| AT-S2a | `score_candidate` = best-over-iterations (min for RMSE, max for NDCG) | unit (hand `CvResult`) | ≤1e-12 |
| AT-S2b | `select_best` argbest + lowest-index tie-break + missing-column/empty error | unit | — |
| AT-S3a | `grid_search` `cv_results` == `cv(&candidates[best_index], …)`; `best_index` == manual argbest | integration (self-consistency, explicit folds) | ≤1e-5 (machine precision) |
| AT-S3b | `refit=true` ⇒ `best_model` predictions == `best_builder.fit(pool)` predictions; empty candidates/metrics → typed error | integration + unit | ≤1e-9 |
| AT-S4 | `randomized_search` samples first `n_iter` of `fisher_yates_permutation(M, seed)`; same-seed determinism; `n_iter>=M` = full; index mapped back | unit + integration | exact |
| AT-S5 | `catboost_rs.grid_search(...)` dict shape + `cv_results` self-consistency vs `catboost_rs.cv` + best sanity + bad-param raises | Python (uv 3.12) | ≤1e-5 |
| AT-S6 | `catboost_rs.randomized_search(...)` determinism + subset size + shape | Python (uv 3.12) | exact / ≤1e-5 |

Primary oracle is DECOMPOSITION/self-consistency (grid_search == cv-per-candidate
+ documented argbest). Upstream-`catboost.grid_search` SELECTION parity is a
deferred non-goal (§2, §9 Q2).

---

## 7. Impact Scope

- **`crates/catboost-rs/src/grid_search.rs`** (NEW) — `local`. New leaf module
  calling existing/ORCH-01 seams; `cv.rs`/`builder.rs`/`model.rs` are **called,
  not modified** `[VERIFIED: CODEGRAPH builder.rs:334; DEPENDENCY CONTRACT cv]`.
- **`crates/catboost-rs/src/lib.rs`** — `local`. Add `mod grid_search;` + `pub use
  grid_search::{grid_search, randomized_search, SearchResult, metric_is_max_optimal};`.
- **`crates/catboost-rs-py/src/`** — `external/public`. New `grid_search` /
  `randomized_search` `#[pyfunction]`s registered on the module (mirrors the
  `sum_models` registration `lib.rs:57`). Additive.
- **`crates/cb-oracle/fixtures/grid_search/`** (OPTIONAL, NEW) + generator arm —
  `local`. Only if a frozen upstream `catboost.grid_search` comparison fixture is
  produced; the primary oracle is self-consistency vs `cv` and needs no NEW
  fixtures beyond ORCH-01's `cv/` corpus.
- **Tests** — new `grid_search_test.rs` (unit), `grid_search_oracle_test.rs`
  (integration), Python parity test.

No persistence/schema/event/cache/config/flag impact. No public contract of an
existing symbol changes. **ORCH-01's `cv`/`CvResult` MUST exist first** — this
module does not create them.

---

## 8. Compatibility and Migration

- **Purely additive.** No existing signature, serialization format, or behavior
  changes. `CatBoostBuilder`, `Model`, `cv`, `Pool`, `parse_metric`, and
  `fisher_yates_permutation` are read-only dependencies.
- **Naming parity:** the function names `grid_search` / `randomized_search`, the
  Python `catboost_rs.grid_search` / `.randomized_search`, and the return keys
  `"params"` / `"cv_results"` mirror upstream for drop-in familiarity
  `[VERIFIED: WEB catboost.ai grid_search / randomized_search docs]`.
- No migration steps; no rollout flag. Rollback = revert the additive module +
  the two Python functions.

---

## 9. Risks and Open Questions

| Risk | Consequence | Mitigation |
|------|-------------|------------|
| **ORCH-01 (`cv`/`CvResult`) not yet shipped** | ORCH-02 cannot compile or run its cv-calling tasks | HARD cross-plan blocker: every cv-calling ORCH-02 task (S2 test build, S3, S4, S5, S6) is gated behind ORCH-01 landing `cv`/`CvResult` (ORCH-01 TASK-04 for the type, TASK-06 for `cv`). Documented in §3 + the PLAN wave graph. ORCH-02-S1 (pure direction, no `CvResult`) is the only fully-unblocked task. `[VERIFIED: LOCAL lib.rs:27-30 no cv]` |
| cv `test-<Metric>-mean` column key spelling differs from `format!("test-{metric}-mean")` | `score_candidate` reads the wrong/absent column → mis-selection or error | Coupling to ORCH-01's naming contract; VERIFY the exact key spelling (raw descriptor vs canonical metric name) against the shipped `cv()` at ORCH-02-S2 Green; if `cv()` canonicalizes (e.g. `"NDCG:top=2"` → `"NDCG"`), reuse the SAME canonicalization. `[DEPENDENCY CONTRACT: ORCH-01 SPEC §4 — column names]` |
| Upstream `grid_search` best-candidate selection detail (best-iteration vs final-iteration; primary-metric choice) differs from ours | selection-parity oracle would drift | First slice uses a DOCUMENTED rule (best-over-iterations of the primary `test-<M>-mean`) with a DECOMPOSITION oracle (grid_search == cv-per-candidate + argbest); upstream-exact selection parity is a deferred non-goal (§2). `[UNVERIFIED — deferred]` |
| Metric direction wrong for a metric not in scope of the cv first slice (ranking) | `select_best` argmax/argmin inverted | ORCH-02-S1 covers ALL `EvalMetric` variants (future-proof) even though cv's first slice only reaches the min-optimal flat metrics; unit-tested per class. `[VERIFIED: CODEGRAPH custom.rs:111-114; metrics.rs:533-541]` |
| Full-product materialization for randomized_search | memory blow-up on a huge grid | First slice expands the full product then subsamples `n_iter` in Rust via `TFastRng64` (documented tradeoff, §2 non-goal); lazy per-parameter sampling is a later slice. |
| Per-candidate training nondeterminism (quantization) | oracle flakiness | Rust `fit`/`cv` are deterministic and oracle-locked; fixed dataset + fixed folds + fixed seed; the primary oracle is self-consistency (grid_search vs cv on the SAME candidate). `[VERIFIED: LOCAL MEMORY ctr-model-loading; cv SPEC §9]` |
| `rayon` per-candidate ordering | non-reproducible `best_index` | Order-preserving `collect` (cv PLAN precedent); `best_index` is by candidate index, not completion order. `[VERIFIED: LOCAL cv PLAN.md:405-409]` |
| Lint gate is CLIPPY not build (`unwrap`/`expect`/`panic`/`indexing_slicing` denied) | CI red despite `cargo build` green | Gate new prod with `cargo clippy -p catboost-rs --lib --no-deps`; selection/scoring return typed `CatBoostError`; use `.get`/`?`/checked argmin. `[VERIFIED: LOCAL MEMORY fstr03-plan gotchas]` |
| Test-mount omission runs 0 tests silently | false green | Mount `grid_search_test.rs` (`#[cfg(test)] mod grid_search_test;`). `[VERIFIED: LOCAL lib.rs:65]` |
| Python cannot link locally (system python 3.14) | Python test unrunnable in-env | Build/run via `uv venv --python 3.12`; `cargo check -p catboost-rs-py` compile-verify. `[VERIFIED: LOCAL MEMORY fstr03-plan]` |
| Refit `Model` not `PartialEq` | `SearchResult` cannot derive `PartialEq` | `best_model` excluded from equality; refit equality is asserted via prediction comparison, not `==`. `[VERIFIED: LOCAL lib.rs:30 Model has no PartialEq re-export]` |
| **`partition_random_seed` double-duty in `randomized_search`.** The SAME seed drives BOTH `fisher_yates_permutation(candidates.len(), partition_random_seed)` (candidate subsampling) AND is forwarded into every per-candidate `cv(...)` call (fold assignment when `shuffle=true`) — two conceptually independent randomization decisions sharing one knob (PLAN-CHECK MINOR finding). | A caller changing `partition_random_seed` to get a different candidate SAMPLE also silently gets a different per-candidate fold ASSIGNMENT for every evaluated candidate — surprising, not documented. Not a correctness bug (both draws are independent, freshly-seeded `TFastRng64` streams; the relative cv comparison across candidates within one call stays fair since every candidate gets the same fold assignment). | Documented here explicitly, first slice; a later slice may add a separate `candidate_random_seed` parameter if this coupling proves confusing in practice. |
| **Three-level nested `rayon` parallelism** (candidates × per-fold via `cv()` × per-feature border selection inside `CatBoostBuilder::fit`). | Possible (low-probability) thread-pool oversubscription / diminishing returns on large candidate counts; NOT a correctness risk — rayon's work-stealing scheduler is designed to compose safely across nested `par_iter()` calls. | Acknowledged here per PLAN-CHECK finding; no additional test planned given low risk — the existing order-preserving-collect oracle already covers correctness. |

**Open questions**

1. **Exact `cv()` column-key spelling** — confirm whether `cv()` keys are the raw
   metric descriptor or a canonical name, and reuse it in `score_candidate`.
   `[UNVERIFIED — resolve at ORCH-02-S2 Green once ORCH-01 ships; DEPENDENCY
   CONTRACT ORCH-01 SPEC §4]`
2. **Upstream selection parity** — whether `catboost.grid_search` selects by best
   or final cv iteration and by which metric; deferred (decomposition oracle
   suffices for the first slice). `[UNVERIFIED — deferred]`
3. **Python refit return shape** — whether to refit the passed estimator in place
   (upstream) vs return a fresh fitted estimator; first slice refits in place to
   match upstream, verify the `EstimatorBase` mutation surface at ORCH-02-S5.
   `[INFERRED — resolve at Python-surface time]`
4. **`metrics=None` default** — the metric name derived from the estimator's
   `loss_function` (e.g. `RMSE`); shares ORCH-01-S6's open question.
   `[INFERRED — verify at Python parity time]`

---

## 10. Traceability and Sources

- **Dependency contract (draft, unimplemented):** `catboost_rs::cv` + `CvResult`
  — `.planning/plans/cv-cross-validation/SPEC.md:210-281` (ORCH-01-S5, §4) and its
  PLAN `.planning/plans/cv-cross-validation/PLAN.md` (TASK-04 lands `CvResult`,
  TASK-06 lands `cv`) `[DEPENDENCY CONTRACT: ORCH-01]`; confirmed ABSENT from the
  facade today `[VERIFIED: LOCAL crates/catboost-rs/src/lib.rs:27-30]`.
- **Reuse targets:** `crates/catboost-rs/src/builder.rs:64-85,334`
  (`CatBoostBuilder` derive + `fit`) `[VERIFIED: CODEGRAPH]`;
  `crates/catboost-rs/src/lib.rs:30` (`Model`) `[VERIFIED: LOCAL]`;
  `crates/cb-train/src/lib.rs:71` (`parse_metric` re-export),
  `crates/cb-train/src/lib.rs:75` (`fisher_yates_permutation` re-export)
  `[VERIFIED: LOCAL]`; `crates/cb-train/src/metrics.rs:64-151,533-541`
  (`EvalMetric` variants + ranking partition) `[VERIFIED: CODEGRAPH]`;
  `crates/cb-compute/src/custom.rs:111-114` (`CustomMetric::is_max_optimal`)
  `[VERIFIED: CODEGRAPH]`; `crates/cb-core/src/rng.rs:171,265` (`TFastRng64`)
  `[VERIFIED: CODEGRAPH]`.
- **Direction convention (no enum method exists):** `EvalMetric` doc comments
  reference `is_max_optimal` at `metrics.rs:77,82,86` but the enum has no such
  method — `[VERIFIED: LOCAL grep over crates/cb-train/src, crates/catboost-rs/src]`.
- **Python binding precedent:** `crates/catboost-rs-py/src/regressor.rs:277-301`
  (`sum_models` free `#[pyfunction]` returning `EstimatorBase::from_model`);
  `crates/catboost-rs-py/src/params.rs:451` (`make_builder`), `params.rs:290`
  (`validate_params`); `crates/catboost-rs-py/src/estimator.rs:24,236`
  (`EstimatorBase`, `data_to_pool`); registration `lib.rs:57` `[VERIFIED: CODEGRAPH]`;
  ORCH-01-S6 Python `cv` precedent `[VERIFIED: LOCAL cv PLAN.md:438-488]`.
- **Upstream API:** `CatBoost.grid_search` / `.randomized_search` signatures +
  `{"params","cv_results"}` return `[VERIFIED: WEB
  https://catboost.ai/en/docs/concepts/python-reference_catboost_grid_search;
  .../python-reference_catboost_randomized_search]`.
- **Sibling SPEC (house style):** `.planning/phases/20-orchestration/calc-metrics/
  SPEC.md` (ORCH-04) `[VERIFIED: LOCAL]`;
  `.planning/plans/cv-cross-validation/SPEC.md` (ORCH-01) `[VERIFIED: LOCAL]`.
- **Greenfield confirmation:** no prior ORCH-02 scoping under `.planning/`
  `[VERIFIED: LOCAL ls .planning/plans — no grid/search folder before this]`.

---

## 11. Implementation Evidence

> Rust slice IMPLEMENTED (2026-07-20). ORCH-02-S1..S4 are `implemented` and
> verified. Python slice IMPLEMENTED (2026-07-20): ORCH-02-S5 (`grid_search`)
> and ORCH-02-S6 (`randomized_search`) are now `implemented` and verified. The
> document lifecycle stays `status: draft` (implementation completion does not
> approve the document). The G-ORCH-01 gate is satisfied: ORCH-01 `cv`/`CvResult`
> shipped and are oracle-green.
>
> **Implementation evidence (ORCH-02-S5..S6, TASK-05/TASK-06):**
> - Source: `crates/catboost-rs-py/src/search.rs` (`grid_search`,
>   `randomized_search` `#[pyfunction]`s + shared `base_params`,
>   `resolve_metrics`, `expand_param_grid`, `refit_estimator`, `result_to_pydict`
>   helpers); registration in `crates/catboost-rs-py/src/lib.rs`
>   (`mod search;` + two `wrap_pyfunction!` lines). Additive only — no chokepoint
>   (`estimator.rs` / `params.rs` / `errors.rs` / `cv.rs` / facade) modified;
>   base params read via the estimator's own sklearn `get_params()`, refit done
>   in place via `set_params` + `fit` (SPEC §9 Q3, facade called with
>   `refit=false`).
> - Tests: `crates/catboost-rs-py/tests/test_grid_search.py` (5 tests — resolve,
>   structure + params membership, cv self-consistency ≤1e-5, bad-param raises,
>   in-place refit) + `crates/catboost-rs-py/tests/test_randomized_search.py`
>   (4 tests — resolve, shape + subset, same-seed determinism, `n_iter>=grid`
>   == grid_search).
> - Verification: `cargo check -p catboost-rs-py` (clean); `cargo clippy -p
>   catboost-rs-py --lib --no-deps` (search.rs introduces 0 findings; the 9
>   pre-existing errors in `ingest_py.rs`/`params.rs` are untouched baseline);
>   `.venv` (Python 3.12, catboost 1.2.10) `maturin develop` + `pytest
>   test_grid_search.py test_randomized_search.py` (9 passed); full py suite
>   98 passed / 3 pre-existing unrelated failures (coreml coremltools decode,
>   pandas ingestion).
>
> **Implementation evidence (ORCH-02-S1..S4):**
> - Source: `crates/catboost-rs/src/grid_search.rs`
>   (`metric_is_max_optimal`, `score_candidate`, `select_best`, `best_over_iters`,
>   `SearchResult`, `run_over`, `grid_search`, `sample_indices`,
>   `randomized_search`); re-exports in `crates/catboost-rs/src/lib.rs`.
> - Tests: `crates/catboost-rs/src/grid_search_test.rs` (11 unit tests —
>   direction, scoring/selection, subsampling) + `crates/catboost-rs/tests/
>   grid_search_oracle_test.rs` (4 integration tests — self-consistency vs `cv`
>   bytewise, independent argbest, refit ≤1e-9 [observed 0.0], randomized
>   determinism / mapping / `n_iter>=M`==grid, typed-error guards).
> - Verification: `cargo test -p catboost-rs --lib grid_search` (11 pass);
>   `cargo test -p catboost-rs --test grid_search_oracle_test` (4 pass);
>   `cargo test -p catboost-rs` (38 lib + all integration, no regressions);
>   `cargo clippy -p catboost-rs --lib --no-deps` (clean);
>   `cargo check -p catboost-rs --no-default-features --features wgpu` (serial
>   GPU-feature path compiles).
> - No `cb-*` / `cv.rs` / `builder.rs` / `model.rs` / `metrics.rs` / `custom.rs`
>   / `permutation.rs` source modified (D-04 no-regression).
>
> (No TreeFinder/PageIndex MCP write target confirmed in-session; this SPEC is
> the effective local spec store per the frontmatter `pageindex_pending` note.)
