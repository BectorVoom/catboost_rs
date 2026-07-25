---
title: "ORCH-01 — cv() cross-validation surface"
status: draft
format: markdown
spec_version: 1
updated_at: 2026-07-19T00:00:00Z
phase: 20-orchestration
slice: cv-cross-validation
source_requirements:
  - "User: Draft SPEC+PLAN for ORCH-01 cv() cross-validation, mirroring upstream catboost.cv(pool, params, fold_count=3, ...)."
  - "Research: Phase-20 orchestration research pass (ORCH-01/02/03); cv() is 100% greenfield."
  - "Sibling precedent (shipped): .planning/phases/20-orchestration/calc-metrics/SPEC.md (ORCH-04)."
pageindex_pending:
  reason: "No TreeFinder/PageIndex write target confirmed in-session for the catboost-rs planning corpus; this SPEC is authored locally under .planning/plans/ (the effective spec store, matching the ORCH-04 sibling's convention)."
  intended_identifier: "catboost-rs / .planning/plans/cv-cross-validation/SPEC.md"
---

# ORCH-01 — `cv()` Cross-Validation Surface

> Draft specification. NOT approved, accepted, final, or implemented.
> Evidence tags: `[VERIFIED: CODEGRAPH …]`, `[VERIFIED: LOCAL <path>]`,
> `[VERIFIED: WEB <url>]`, `[INFERRED: …]`, `[UNVERIFIED: …]`.

---

## 1. Context

`catboost_rs` is a Rust rewrite of CatBoost, oracle-tested ≤10⁻⁵ against the
original C++ library, with a dual Rust + Python surface. Upstream exposes a
free function `catboost.cv(pool, params, fold_count=3, ...)` that k-fold
cross-validates a parameter set and returns **per-iteration** aggregate metric
columns (`iterations`, `test-<Metric>-mean`, `test-<Metric>-std`,
`train-<Metric>-mean`, `train-<Metric>-std`)
`[VERIFIED: WEB https://catboost.ai/docs/en/concepts/python-reference_cv —
signature `cv(pool, params, dtrain, iterations, num_boost_round, fold_count=3,
nfold, inverted=False, partition_random_seed=0, seed, shuffle=True,
logging_level, stratified, as_pandas=True, metric_period, verbose,
verbose_eval, plot, early_stopping_rounds, folds, type='Classical',
return_models=False)`; return columns `iterations`, `<dataset>-<metric>-<stat>`]`.

There is **no** `cv`, cross-validation, or fold-splitting-for-CV surface
anywhere in the workspace — this slice is **100% greenfield**
`[VERIFIED: LOCAL grep "select_rows|fn cv\b|cross_valid|k_fold" over crates/ —
only unrelated `fold_count` boosting knobs in `cb-backend`/`cb-compute`]`. The
training surface is purely functional (`cb_train::{train,
train_with_eval_sets, train_ranking, train_cat}`, all delegating to a private
`train_inner`) `[VERIFIED: CODEGRAPH crates/cb-train/src/boosting.rs:1946,2048,
2092,2145,2259]`; there is **no** `Trainer` object holding cross-call state and
**no** existing disjoint-fold CV partitioner (`create_folds`,
`crates/cb-train/src/fold.rs:256`, builds upstream's *learning/averaging
permutation* folds for boosting internals, not train/test CV partitions)
`[VERIFIED: CODEGRAPH fold.rs:256]`.

Every primitive `cv()` needs already exists and is oracle-locked:

- **Per-fold training:** the facade `CatBoostBuilder::fit(&self, pool: &Pool) ->
  Result<Model, CatBoostError>` quantizes borders and calls `cb_train::train`
  `[VERIFIED: CODEGRAPH crates/catboost-rs/src/builder.rs:334-402]`.
- **Per-iteration prediction on a held-out fold:** `Model::staged_predict(pool,
  ntree_start, ntree_end, eval_period) -> Result<Vec<Vec<f64>>, CatBoostError>`
  returns one raw-approx row per tree-prefix stage; it is gated to scalar,
  oblivious, float-only, non-CTR models via `ensure_scalar_oblivious`
  `[VERIFIED: CODEGRAPH/Read crates/catboost-rs/src/model.rs:144-208]`.
- **Per-stage metric value:** the shipped ORCH-04 facade `catboost_rs::eval_metric(
  label, approx, metric, weight, group_id) -> Result<f64, CatBoostError>`
  `[VERIFIED: CODEGRAPH crates/catboost-rs/src/metrics.rs:44-59]`.
- **Deterministic fold shuffle:** the bit-exact `cb_train::fisher_yates_permutation(
  n, seed) -> Vec<i32>` over `cb_core::rng::TFastRng64` (the project's PCG-XSH-RR
  engine mirroring upstream) `[VERIFIED: CODEGRAPH crates/cb-train/src/
  permutation.rs:109; crates/cb-core/src/rng.rs:141-267; re-export
  crates/cb-train/src/lib.rs:75]`.
- **Order-stable reduction:** `cb_core::sum_f64` (already the workspace's
  summation chokepoint) `[VERIFIED: CODEGRAPH crates/cb-train/src/metrics.rs:292]`.

So `cv()` is **new orchestration-layer glue** that calls these existing,
oracle-locked seams N times (serial, or `rayon`-parallel per fold **when the
`cpu` feature is compiled** — see §3.1 — `rayon 1.12.0` is a `catboost-rs` dep
`[VERIFIED: LOCAL crates/catboost-rs/Cargo.toml:39 `rayon.workspace = true`;
Cargo.toml:49 `rayon = "1.12.0"`]`) plus two small new primitives (a Pool
row-subset and a fold-index partitioner).

### 1.1 Empirical resolution of the per-fold quantization question (closes a prior CRITICAL finding)

A first Plan-Checker pass raised a CRITICAL concern: `CatBoostBuilder::fit`
always re-derives `feature_borders` from whatever `pool` it receives
(`select_borders_greedy_logsum`, `builder.rs:360-365`) — does upstream
`catboost.cv()` instead share ONE global quantization across all folds, which
this plan's `fit(select_rows(train))`-per-fold architecture would then fail to
reproduce ≤1e-5? This was resolved **empirically**, not by inference, against
the real installed `catboost==1.2.10` package (`.venv`):

- **Global quantization sharing is REFUTED.** Using `catboost.cv(...,
  return_models=True)` and `model.get_borders()` on a dataset with extreme
  outliers confined to one fold's TEST block (never in that fold's train
  subset), the fold's own trained model's borders did **not** include a border
  for the outlier region — a global, full-pool quantization would have forced
  it in. Upstream quantizes each fold's TRAIN SUBSET independently, matching
  this plan's `fit(select_rows(train))` architecture exactly. **No
  shared-border seam is needed; the architecture is unchanged.**
  `[VERIFIED: LOCAL empirical experiment against .venv catboost==1.2.10,
  reproduced independently by two separate review passes]`
- **A second, more consequential divergence source was found and fixed:**
  `catboost.cv()`'s raw, dict-based native call
  (`catboost/core.py:7281-7313`) does **NOT** apply the `CatBoostRegressor`
  estimator class's Python-side default-injection (e.g. `boost_from_average`
  defaults to `True` for RMSE **only** at the estimator-class layer, never at
  the raw `_cv()` extension layer) — a `params` dict lacking that key trains
  EVERY fold with `boost_from_average` effectively `False` (bias forced to
  `0.0`), while this project's `CatBoostBuilder` defaults
  `boost_from_average: true` (`builder.rs:108`, matching the Estimator-class
  default, not the raw-dict default). Comparing this plan's `cv()` (built on
  `CatBoostBuilder`, bias = train-subset mean) against a naive
  `catboost.cv(pool, {..no boost_from_average key..})` fixture (bias forced
  to `0.0`) would therefore NEVER hit ≤1e-5, for a reason having nothing to do
  with cv's architecture — a pure **oracle-fixture-generation** hazard.
  **Mitigation (binding on TASK-01): every oracle fixture's `params` dict MUST
  explicitly set `"boost_from_average": True` (and `"bootstrap_type": "No"`,
  matching `CatBoostBuilder`'s other default, `builder.rs:110`) so the raw
  `catboost.cv()` call receives the SAME effective defaults `CatBoostBuilder`
  applies** `[VERIFIED: LOCAL empirical experiment — explicitly matching both
  keys collapses aggregate `test-RMSE-mean` divergence to ~1e-9]`.
- **ddof resolved:** `catboost.cv`'s `test-<M>-std` is **sample** std (ddof=1,
  divide by `F-1`), confirmed by matching `numpy.std(per_fold_values, ddof=1)`
  exactly against the fixture's reported `-std` column (and confirmed
  different from ddof=0). §5-S4 and the aggregation formula below are pinned
  to ddof=1, not "population std" as an earlier draft of this section stated.

**Crate placement (locked scope decision):** the row-subset lives in `cb-data`
(the only crate with `Pool`'s private constructor), and *all other* cv logic —
fold partitioning, the per-fold train/eval loop, aggregation, and the public
`cv` free function — lives in a new `crates/catboost-rs/src/cv.rs` facade
module, surfaced through the Python bindings. No new `cb-orchestrate` crate.
Rationale: only the facade can call `CatBoostBuilder::fit` /
`Model::staged_predict` (both live in `catboost-rs`, which depends on `cb-train`/
`cb-model`, never the reverse), so the loop and its orchestration MUST be
facade-level; the partition/aggregation helpers are pure and unit-tested in the
same module. This keeps the blast radius to three crates (`cb-data`,
`catboost-rs`, `catboost-rs-py`) `[VERIFIED: LOCAL Cargo dep direction —
crates/cb-data/Cargo.toml:12 (cb-data → cb-core only); crates/catboost-rs/
Cargo.toml:26-34 (facade → cb-core/cb-data/cb-train/cb-model)]`.

---

## 2. Scope and Non-Goals

### In scope (first slice)

- A **fold-index partitioner** (new, pure): non-stratified k-fold assignment of
  `n` rows into `fold_count` disjoint test blocks (train = complement), with an
  optional deterministic `shuffle` seeded by `partition_random_seed` (reusing
  `fisher_yates_permutation`), a **grouped** variant that keeps every ranking
  `group_id` whole within one fold, an `inverted` / `type` role-swap (train on
  the single test block), and support for **caller-supplied explicit `folds`**.
- A **`Pool` row-subset** (new): `Pool::select_rows(&self, &[usize]) -> Pool`
  producing a Pool over the selected rows across every populated column.
- A **per-fold train + per-iteration staged-eval loop** (new glue): for each
  fold, `CatBoostBuilder::fit` on the train sub-Pool, then `Model::staged_predict`
  on both the test and train sub-Pools, then `eval_metric` per stage — yielding,
  per metric, a per-iteration test curve and train curve.
- A **cross-fold aggregation** (new, pure): per iteration, per metric, the mean
  and SAMPLE (ddof=1) standard deviation across folds — empirically confirmed
  against upstream, §1.1 — emitted as the upstream columns
  `test-<Metric>-mean`, `test-<Metric>-std`, `train-<Metric>-mean`,
  `train-<Metric>-std`, plus an `iterations` index. Requires `F >= 2` folds
  (an empty or single-fold input is a typed error, never `NaN`).
- A **Rust facade** free function `catboost_rs::cv(pool, builder, metrics,
  fold_count, shuffle, partition_random_seed, inverted, folds) -> CvResult`.
- A **Python** `catboost_rs.cv(pool, params, fold_count=3, shuffle=True,
  partition_random_seed=0, inverted=False, folds=None, ...)` mirroring the
  supported subset of `catboost.cv`, returning a dict of columns.
- **Oracle fixtures** for a fixed numeric-regression dataset with **explicit
  fixed folds** (`catboost.cv(..., folds=<fixed>, shuffle=False)`), compared
  ≤1e-5 on the per-iteration `test-`/`train-` mean/std columns.

### Non-goals (explicit — documented, not silently dropped)

- **`stratified=True`** (label-strata-balanced folds). Deferred: needs a
  stratified partitioner; the first slice ships only non-stratified /
  grouped-non-stratified k-fold. Selecting it is a typed error.
- **Bit-exact upstream RNG-parity for `shuffle=True` fold ASSIGNMENT.** Our
  `shuffle=True` path is *deterministic* (seeded `TFastRng64` via
  `fisher_yates_permutation`) but is **NOT** claimed byte-identical to upstream
  C++ `cv`'s internal fold assignment. Establishing that parity is a separate
  slice; the DEFAULT oracle therefore uses **explicit `folds=`** to sidestep
  assignment-RNG parity entirely `[UNVERIFIED: upstream cv shuffle→fold
  assignment stream not reproduced ≤1e-5 in-session — deferred; see §9 Q1]`.
- **Categorical / CTR / text / embedding models.** `staged_predict` rejects a
  CTR/categorical/non-oblivious/multi-dim model with a typed
  `CatBoostError::UnsupportedModel` `[VERIFIED: CODEGRAPH model.rs:144-171]`, so
  the first slice supports only **scalar, oblivious, float-only** models
  (numeric regression / binary). A model outside that class surfaces the typed
  error, never a wrong result.
- **Multiclass / multilabel** (`approx_dimension > 1`) — rejected by the same
  guard.
- **`type='TimeSeries'`**, **`return_models`**, **`plot`/`plot_file`**,
  **in-cv `early_stopping_rounds`**, and **`metric_period`** (the first slice
  always emits every iteration; no stage subsampling). `type='Inverted'` /
  `inverted=True` (train-on-one-fold role swap) IS supported.
- **`as_pandas` DataFrame** return. The Rust surface returns a `CvResult`
  (columns map); the Python surface returns a **dict of columns** (a DataFrame
  wrapper is a later slice).
- **Extending the metric set.** cv() reuses exactly the ORCH-04 `parse_metric`
  metric set; an unsupported metric string is the existing typed parse error.
- **GPU** — training rides whatever backend feature is compiled; cv() adds no
  GPU code.

---

## 3. Dependencies

| Dependency | Kind | Evidence |
|-----------|------|----------|
| `catboost_rs::CatBoostBuilder::fit(&Pool) -> Result<Model, CatBoostError>` | reuse (per-fold train) | `[VERIFIED: CODEGRAPH crates/catboost-rs/src/builder.rs:334-402]` |
| `catboost_rs::Model::staged_predict(pool, start, end, period)` + `ensure_scalar_oblivious` | reuse (per-iteration test approx + model-class guard) | `[VERIFIED: CODEGRAPH/Read crates/catboost-rs/src/model.rs:144-208]` |
| `catboost_rs::Model::predict(&Pool)` | reuse (final-iteration cross-check) | `[VERIFIED: CODEGRAPH model.rs:135]` |
| `catboost_rs::eval_metric(label, approx, metric, weight, group_id)` (ORCH-04) | reuse (per-stage metric) | `[VERIFIED: CODEGRAPH crates/catboost-rs/src/metrics.rs:44-59]` |
| `cb_train::fisher_yates_permutation(n, seed) -> Vec<i32>` over `TFastRng64` | reuse (deterministic shuffle) | `[VERIFIED: CODEGRAPH crates/cb-train/src/permutation.rs:109; re-export lib.rs:75]` |
| `cb_core::sum_f64` | reuse (order-stable mean/std reduction) | `[VERIFIED: CODEGRAPH crates/cb-core/src/reduction.rs:32]` |
| `cb_data::Pool` accessors (`n_rows`, `float_features`, `cat_features`, `label`, `weights`, `group_id`, `subgroup_id`, `pairs`, `baseline`) + private `from_validated_columns` | subset target (new method needs constructor access) | `[VERIFIED: CODEGRAPH crates/cb-data/src/pool.rs:82-205]` |
| `catboost_rs::CatBoostError` (`Train(#[from] cb_core::CbError)`, `UnsupportedModel`, `FeatureMismatch`) | reuse (typed error surface) | `[VERIFIED: CODEGRAPH crates/catboost-rs/src/error.rs:37,70,113]` |
| `rayon` 1.12.0 (facade dep) | optional per-fold parallelism | `[VERIFIED: LOCAL crates/catboost-rs/Cargo.toml:39; Cargo.toml:49]` |
| `catboost-rs-py` bindings (`EstimatorBase`/`make_builder`/`data_to_pool`/`PyCbError`/`to_pyerr`) | new Python surface | `[VERIFIED: CODEGRAPH crates/catboost-rs-py/src/estimator.rs:24,204,236; params.rs:451]` |
| `catboost==1.2.10` under `uv --python 3.12`, `numpy<2` | new oracle-fixture generation | `[VERIFIED: LOCAL MEMORY next-features-5plan-batch / fstr03-plan uv-py3.12 flow]` |

**No new external crate** is required — every primitive is in-tree (satisfies
the "use existing capability first" constraint `[VERIFIED: LOCAL CLAUDE.md
Dependencies]`). In particular, fold shuffling reuses the project's own
`TFastRng64`, not a new `rand` dependency `[VERIFIED: CODEGRAPH rng.rs:141]`.

---

## 4. Typed Contracts

New pure/glue code lives in `crates/catboost-rs/src/cv.rs` (prod) with unit
tests in `crates/catboost-rs/src/cv_test.rs`, mounted via the crate-root idiom
the facade uses (`#[cfg(test)] mod cv_test;`, cf. the ORCH-04 facade's
`metrics_test` mount) `[VERIFIED: LOCAL calc-metrics PLAN.md TASK-06]`. The Pool
subset lives in `crates/cb-data/src/pool.rs` with tests in the existing
`crates/cb-data/src/pool_test.rs` (or a mounted sibling per the source/test
separation rule) `[VERIFIED: LOCAL CLAUDE.md source/test separation]`.

```rust
// crates/cb-data/src/pool.rs  (ORCH-01-S2)

impl Pool {
    /// Build a new [`Pool`] containing only `indices` rows, in the given order,
    /// across every populated column (float/cat/text/embedding features + label,
    /// weights, group_id, subgroup_id, baseline). Empty columns stay empty.
    /// Ranking `pairs` are DROPPED (row re-indexing would invalidate pair ids);
    /// the first-slice CV path is numeric/grouped, not pairwise (documented).
    /// An out-of-range index is skipped (checked access; never panics), so the
    /// result has `<= indices.len()` rows.
    #[must_use]
    pub fn select_rows(&self, indices: &[usize]) -> Pool;
}
```

```rust
// crates/catboost-rs/src/cv.rs  (ORCH-01-S1, -S3, -S4, -S5)

/// One CV fold: the row indices used to TRAIN and to TEST. Disjoint; their
/// union covers every partitioned row exactly once (Classical) — for the
/// grouped variant, boundaries fall only between whole groups.
#[derive(Debug, Clone, PartialEq)]
pub struct CvFold {
    pub train: Vec<usize>,
    pub test: Vec<usize>,
}

/// How folds are formed. `Classical` tests on one block, trains on the rest;
/// `Inverted` trains on one block, tests on the rest (upstream `inverted=True`
/// / `type='Inverted'`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CvType { Classical, Inverted }

/// (ORCH-01-S1) Partition `n` rows into `fold_count` folds. `group_id` empty ⇒
/// plain k-fold; non-empty ⇒ grouped k-fold (whole groups per fold, contiguity
/// respected). `shuffle` seeds a deterministic `fisher_yates_permutation` by
/// `seed`; `false` uses the identity order. `cv_type` selects Classical vs
/// Inverted.
///
/// # Errors
/// [`CatBoostError::Train`] wrapping [`cb_core::CbError::Degenerate`] when
/// `fold_count < 2`, `fold_count > n` (or `> group_count` when grouped), `n == 0`,
/// or `group_id` is non-empty and non-contiguous.
pub fn make_cv_folds(
    n: usize,
    group_id: &[u64],
    fold_count: usize,
    shuffle: bool,
    seed: u64,
    cv_type: CvType,
) -> Result<Vec<CvFold>, CatBoostError>;

/// The result of a cv() run: the `iterations` index plus one column per
/// `<dataset>-<metric>-<stat>` (e.g. `test-RMSE-mean`), each a per-iteration
/// vector of length `iterations`. Mirrors the upstream cv DataFrame columns.
#[derive(Debug, Clone, PartialEq)]
pub struct CvResult {
    pub iterations: Vec<usize>,
    pub columns: std::collections::BTreeMap<String, Vec<f64>>,
}

/// (ORCH-01-S5) k-fold cross-validate `builder`'s parameter set on `pool`,
/// returning per-iteration aggregate metric columns.
///
/// - `metrics`: metric descriptor strings (ORCH-04 grammar). Empty ⇒ typed error.
/// - `folds`: explicit per-fold TEST-row index sets (train = complement);
///   `Some` overrides `fold_count`/`shuffle`/`seed`. `Some(&[])` (an explicit
///   but EMPTY fold list) is ALSO a typed error, validated at entry — NOT
///   silently forwarded to zero-fold aggregation (added post-cap verification
///   pass 4/5: `sum_f64(&[])/0 == NaN` would otherwise result, violating the
///   "never NaN" invariant below).
/// - Every fold trains a fresh model with `builder`; per-iteration test/train
///   metric curves are aggregated across folds (mean + SAMPLE std, ddof=1 —
///   empirically confirmed against upstream, §1.1; requires `F >= 2` folds).
///
/// # Errors
/// [`CatBoostError::UnsupportedModel`] if a fold's model is not scalar/oblivious/
/// float-only (propagated from `staged_predict`); [`CatBoostError::Train`] for a
/// bad partition, empty `metrics`, an empty explicit `folds` list, fewer than 2
/// resulting folds, mismatched per-fold iteration counts, or any training/eval
/// failure; [`CatBoostError::FeatureMismatch`] never (same Pool schema
/// throughout). Never panics, never returns a `NaN`-bearing `Ok`.
#[allow(clippy::too_many_arguments)]
pub fn cv(
    pool: &Pool,
    builder: &CatBoostBuilder,
    metrics: &[&str],
    fold_count: usize,
    shuffle: bool,
    partition_random_seed: u64,
    inverted: bool,
    folds: Option<&[Vec<usize>]>,
) -> Result<CvResult, CatBoostError>;
```

```python
# catboost_rs.cv (PyO3), mirroring the supported subset of catboost.cv  (ORCH-01-S6)
def cv(
    pool,                      # native Pool or (X, y) framework object
    params,                    # dict -> CatBoostBuilder via make_builder
    fold_count=3,
    inverted=False,
    partition_random_seed=0,
    seed=None,                 # alias of partition_random_seed
    shuffle=True,
    folds=None,                # list[list[int]] of test-row indices, or None
    metrics=None,              # str | list[str]; None -> derived from loss_function
) -> dict[str, list[float]]:
    """Returns {'iterations': [...], 'test-<M>-mean': [...], 'test-<M>-std': [...],
    'train-<M>-mean': [...], 'train-<M>-std': [...]} — the upstream cv columns."""
```

---

## 5. Failure-Isolated Behavioral Specifications

Each spec has ONE primary reason a failing acceptance test would fail.

### ORCH-01-S1 — Fold-index partitioning
- **Status:** unimplemented
- **Responsibility:** map `(n, group_id, fold_count, shuffle, seed, cv_type)` to
  disjoint `CvFold`s, and nothing else.
- **Input:** as above. **Output:** `Result<Vec<CvFold>, CatBoostError>`.
- **Dependencies:** `cb_train::fisher_yates_permutation` (shuffle order only).
- **Behavior:**
  - Plain (`group_id` empty), `shuffle=false`, `fold_count=k`: `k` folds whose
    `test` blocks are the `k` contiguous near-equal partitions of `0..n`; each
    `train` is the exact complement; the `k` test blocks are pairwise disjoint
    and cover `0..n` exactly once.
  - `shuffle=true`: identical partition applied over the permuted order from
    `fisher_yates_permutation(n, seed)`; the SAME `seed` gives the SAME folds
    (determinism).
  - Grouped (`group_id` non-empty): every group's rows land in exactly one
    fold's test block (no group split across folds); requires contiguous
    `group_id`. **`shuffle` × grouped interaction (pinned, was previously
    unspecified):** `shuffle=true` permutes GROUP-SPAN order (which whole
    group is assigned to which fold), NEVER the row order WITHIN a group —
    each group's rows stay contiguous and in their original relative order
    both before and after `select_rows`, which is a hard precondition of
    `eval_metric`'s `eval_grouped` (non-contiguous `group_id` ⇒
    `CbError::Degenerate`, `metrics.rs:487-661`). A multi-group-per-fold
    configuration (not just the trivial 1-group-per-fold case) and a
    `shuffle=true` + grouped combination are both required test cases below.
  - `Inverted`: `train` and `test` roles swapped (train on the one block).
  - `fold_count < 2`, `> n` (or `> group_count`), `n == 0`, or non-contiguous
    `group_id` ⇒ `Err(CatBoostError::Train(Degenerate))`.
- **Invariants:** total function; no panic; disjoint+covering test blocks;
  grouped-mode output preserves each selected group's row contiguity (needed
  by `eval_grouped` downstream).
- **Acceptance test:** `cv_test.rs` unit tests (partition coverage/disjointness,
  determinism, grouped whole-group placement — including a MULTI-group-per-fold
  case, not only 1-group-per-fold — a `shuffle=true`+grouped case, inverted
  swap, error paths).
- **Traceability:** `permutation.rs:109`; `fold.rs:256` (contrast — NOT reused).

### ORCH-01-S2 — `Pool::select_rows`
- **Status:** unimplemented
- **Responsibility:** produce a Pool over the selected rows across every
  populated column; nothing else.
- **Input:** `&self`, `indices: &[usize]`. **Output:** `Pool`.
- **Dependencies:** `Pool::from_validated_columns` (private constructor).
- **Behavior (Given/When/Then):**
  - Given a Pool with `f` float columns of length `n` and a `label`/`weights`/
    `group_id`, When `select_rows(&[i0, i1, …])`, Then the result has one row per
    valid index, each float column gathered in index order, and `label`/`weights`
    /`group_id`/`subgroup_id`/`baseline` gathered identically (an empty source
    column stays empty).
  - An out-of-range index is skipped (checked `.get`), never a panic.
  - `pairs` are dropped (row re-index would invalidate ids) — documented.
- **Invariants:** `result.n_rows() == count(valid indices)`; every populated
  column length equals `result.n_rows()`.
- **Acceptance test:** `pool_test.rs` unit tests (gather correctness, empty
  columns preserved, OOB skipped).
- **Traceability:** `pool.rs:82-205`.

### ORCH-01-S3 — Per-fold train + per-iteration staged evaluation
- **Status:** unimplemented
- **Responsibility:** for one fold, produce per-metric per-iteration TEST and
  TRAIN metric curves by fitting on the train sub-Pool and staged-evaluating on
  both sub-Pools. Delegates train/predict/eval to existing seams.
- **Input:** `pool: &Pool`, `fold: &CvFold`, `builder: &CatBoostBuilder`,
  `metrics: &[&str]`. **Output:** per-metric `{test: Vec<f64>, train: Vec<f64>}`
  over iterations, as `Result<_, CatBoostError>`.
- **Dependencies:** `Pool::select_rows` (S2), `CatBoostBuilder::fit`,
  `Model::staged_predict`, `catboost_rs::eval_metric`.
- **Behavior:**
  - Given a numeric-regression `pool` and a `CvFold`, When the fold is run, Then
    `model = builder.fit(pool.select_rows(fold.train))`, `test_stages =
    model.staged_predict(pool.select_rows(fold.test), None, None, None)`,
    `train_stages = model.staged_predict(pool.select_rows(fold.train), …)`, and
    each stage `s` yields `eval_metric(sub_label, stages[s], m, sub_weight,
    sub_group_id)` for every requested metric `m`.
  - The number of stages equals the trained tree count (`iterations`); the last
    test stage's metric equals `eval_metric` on `model.predict(test_subpool)`.
  - A non-scalar/oblivious/float-only model ⇒ the typed
    `CatBoostError::UnsupportedModel` from `staged_predict` (propagated).
- **Invariants:** no panic; empty sub-fold (0 rows) ⇒ typed error, never NaN.
- **Acceptance test:** decomposition unit test in `cv_test.rs` — a hand-built
  2-fold numeric Pool, asserting the fold curve equals the manual
  `fit`+`staged_predict`+`eval_metric` composition; PLUS a degenerate 0-row
  fold case (e.g. `fold_count` at the edge of `n`) driving the "typed error,
  never NaN" invariant through the REAL `fit`/`staged_predict` error paths,
  not merely asserted in prose.
- **Traceability:** `builder.rs:334`; `model.rs:189`; `metrics.rs:44`.

### ORCH-01-S4 — Cross-fold mean/std aggregation
- **Status:** unimplemented
- **Responsibility:** per iteration, per metric, aggregate the per-fold curves
  into mean and SAMPLE (ddof=1) std; emit the upstream column names. Nothing
  else.
- **Input:** per-fold, per-metric test/train curves (all length `iterations`).
  **Output:** `Result<CvResult, CatBoostError>`.
- **Dependencies:** `cb_core::sum_f64`.
- **Behavior:**
  - Given `F` folds each with a length-`I` test curve for metric `M`, Then
    `columns["test-M-mean"][i] = mean_f(curve_f[i])` and `columns["test-M-std"][i]
    = sqrt(sum_f((curve_f[i]-mean)^2) / (F-1))` — **SAMPLE std, ddof=1**
    (empirically confirmed against `catboost.cv`'s reported `-std` columns;
    NOT population/ddof=0 as an earlier draft assumed — see §1.1), for each
    `i` and each metric; likewise for `train-`. `F == 1` (a single fold) is a
    degenerate case with no defined sample std — typed error, not division by
    zero or NaN. **`F == 0` (an empty `per_fold` slice — e.g. from an explicit
    but empty caller-supplied `folds` list) is ALSO a typed error, not
    `sum_f64(&[])/0 == NaN`** (added post-cap verification pass 4/5, a
    defense-in-depth guard independent of the TASK-06 entry-point check).
  - `iterations = [0, 1, …, I-1]`.
  - Folds with mismatched curve lengths ⇒ `Err(CatBoostError::Train(Degenerate))`
    (no ragged aggregation).
- **Invariants:** all reductions go through `cb_core::sum_f64` (D-08); no panic.
- **Acceptance test:** `cv_test.rs` unit — hand curves with a known mean/sample-std
  (ddof=1), plus an `F==1` degenerate-error case.
- **Traceability:** `cb_core::sum_f64` definition at `crates/cb-core/src/reduction.rs:32`
  (corrected citation — `crates/cb-train/src/metrics.rs:292` was a call site,
  not the definition).

### ORCH-01-S5 — Rust facade `cv`
- **Status:** unimplemented
- **Responsibility:** orchestrate S1→S3→S4 with validation and typed errors;
  handle explicit `folds` and `inverted`.
- **Input/Output:** the `cv(...) -> Result<CvResult, CatBoostError>` contract
  from §4.
- **Dependencies:** S1, S2, S3, S4 + the reused train/eval seams.
- **Behavior:**
  - Given a fixed numeric-regression `pool`, a `builder`, `metrics=["RMSE"]`, and
    explicit `folds`, When `cv(...)` runs, Then `CvResult.columns` contains
    `test-RMSE-mean/std` and `train-RMSE-mean/std`, each length `iterations`,
    matching `catboost.cv(pool, params, folds=<same>, shuffle=False)` ≤1e-5.
  - `metrics` empty ⇒ `Err(CatBoostError::Train(Degenerate))`.
  - A CTR/categorical/multiclass model ⇒ `Err(CatBoostError::UnsupportedModel)`
    (propagated from the first fold's `staged_predict`).
  - `folds = Some(...)` overrides `fold_count`/`shuffle`/`seed`.
  - `folds = Some(&[])` (explicit but EMPTY) ⇒ `Err(CatBoostError::Train(
    Degenerate))`, validated at entry BEFORE `aggregate_folds` is reached —
    never a `NaN`-bearing `Ok(CvResult)` (added post-cap verification pass 4/5).
- **Invariants:** no `unwrap`/`panic` on any path. **Rayon-parallel fold
  execution policy (pinned, was previously unexamined):** per-fold work runs
  under `rayon` (`par_iter().collect()`, order-preserving) ONLY when the
  `cpu` feature is compiled (`#[cfg(feature = "cpu")]`); under any GPU feature
  (`wgpu`/`cuda`/`rocm`) fold execution is SERIAL. Rationale: `CatBoostBuilder::fit`
  constructs a `GpuBackend::default()` holding a `cubecl::client::ComputeClient<
  SelectedRuntime>` bound to a single physical device when a GPU feature is
  compiled (`crates/cb-backend/src/gpu_backend.rs:57-63`); concurrently
  constructing/using multiple such clients from separate rayon worker threads
  against the same device is unverified and untested by this project (GPU
  tests run on `rocm` only, per CLAUDE.md), so parallel fold execution is
  scoped OUT for GPU builds in this first slice rather than left as an
  unexamined risk. The aggregated result is order-independent (byte-identical
  to serial) either way, because each fold's curves are keyed by fold index —
  this is asserted by a dedicated serial-vs-parallel byte-identity test under
  `cpu`.
- **Acceptance test:** `cv_oracle_test.rs` (integration) over new fixtures;
  plus a serial-vs-`cpu`-parallel byte-identity test, and a compile check
  under `--features wgpu` confirming the serial-only path builds.
- **Traceability:** S1–S4 + `builder.rs:334`, `model.rs:189`, `metrics.rs:44`;
  `gpu_backend.rs:57-63` (GPU concurrency rationale).

### ORCH-01-S6 — Python `catboost_rs.cv`
- **Status:** implemented
- **Implementation evidence (TASK-07, 2026-07-20):** `#[pyfunction] cv`
  (signature `cv(pool, params, fold_count=3, inverted=False,
  partition_random_seed=0, seed=None, shuffle=True, folds=None, metrics=None)`)
  in `crates/catboost-rs-py/src/cv.rs`; registered via `mod cv;` +
  `m.add_function(wrap_pyfunction!(cv::cv, m)?)?` in
  `crates/catboost-rs-py/src/lib.rs`. Own-before-detach (D-11): params dict →
  owned `BTreeMap`, `data_to_pool`/tuple-unpack, `make_builder`, metric list
  resolved (str/list or derived from `loss_function`, default `RMSE`), `seed`
  overrides `partition_random_seed`, then `py.detach(|| catboost_rs::cv(...))`
  mapped via `.map_err(PyCbError)?`; returns the `iterations` + column dict.
  Verified `cargo check -p catboost-rs-py` clean; `maturin develop` under
  `.venv` py3.12/catboost 1.2.10; `pytest crates/catboost-rs-py/tests/test_cv.py`
  6 passed — explicit-folds parity vs `catboost.cv` max|diff| ≈ 9e-9 (test/train
  RMSE mean/std), ≤1e-5. **Documented divergence (AT-S6 sub-case):** the
  "unsupported model raises" arm is correctly wired (any facade
  `UnsupportedModel` maps to `CatBoostValueError` via the `PyCbError`
  chokepoint) but is currently UNREACHABLE from the float-only Python surface —
  the present training path does not CTR-model declared `cat_features` (1-vs-2
  cat_feature runs give byte-identical curves) and `parse_loss` exposes only
  scalar losses, so no `ctr_data`/`approx_dimension>1` model is producible here;
  the two reachable mapped-exception cases (empty + unknown metric) are asserted
  green. This is pre-existing facade/cb-train behavior, out of the additive
  TASK-07 scope. The two `SPEC §8` combined-argument-override divergence is
  implemented (folds silently overrides the partition knobs; commented in
  `cv.rs`).
- **Responsibility:** PyO3 surface mirroring the supported subset of
  `catboost.cv`.
- **Input:** `pool` (native Pool or `(X, y)`), `params` (dict), `fold_count`,
  `inverted`, `partition_random_seed`, `seed`, `shuffle`, `folds`, `metrics`.
  **Output:** `dict[str, list[float]]` of the cv columns.
- **Dependencies:** `data_to_pool`, `make_builder`, `catboost_rs::cv`, `PyCbError`.
- **Behavior:**
  - `catboost_rs.cv(pool, {"iterations":10,"learning_rate":0.1}, fold_count=3,
    folds=<fixed>, shuffle=False)` returns a dict whose `test-RMSE-mean` etc.
    match `catboost.cv(...)` ≤1e-5.
  - `metrics=None` derives the metric from `params["loss_function"]`
    (default `RMSE`); a `str` or `list[str]` is accepted.
  - A bad metric / unsupported model raises a mapped `CatBoostError`, not a
    panic/abort.
- **Acceptance test:** Python parity test under uv-3.12 / catboost 1.2.10.
- **Traceability:** `estimator.rs:24,204,236`; `params.rs:451`; ORCH-04-S6
  Python precedent.

**Additional required test (closes a prior MAJOR gap — grouped k-fold was
in-scope but had zero end-to-end coverage):** an integration test running a
ranking loss + ranking metric (e.g. NDCG) through the FULL `cv()` pipeline with
grouped folds (`group_id` non-empty), asserting no error, correct aggregation,
and no spurious `eval_grouped` "non-contiguous group_id" rejection —
`cv_grouped_ranking_test.rs`.

---

## 6. Acceptance Scenarios

| ID | Scenario | Oracle | Tolerance |
|----|----------|--------|-----------|
| AT-S1a | plain k-fold: test blocks disjoint + cover `0..n`; complement train | unit (hand) | exact |
| AT-S1b | `shuffle=true` same-seed determinism; grouped whole-group placement; inverted swap; error paths (`fold_count<2`, `>n`, non-contiguous group) | unit | — |
| AT-S2 | `select_rows` gathers every column; empty columns preserved; OOB skipped | unit | exact |
| AT-S3 | one fold's per-iteration test/train curve == manual `fit`+`staged_predict`+`eval_metric` composition | unit (decomposition) | ≤1e-5 |
| AT-S4 | mean/sample-std (ddof=1) over hand curves match closed form | unit | ≤1e-12 |
| AT-S4b | `F==0` (empty per-fold slice) and `F==1` ⇒ typed error, never `NaN` | unit | — |
| AT-S5d | `cv(..., folds=Some(&[]))` (explicit empty folds) ⇒ typed error, never a `NaN`-bearing `Ok` | unit | — |
| AT-S1c | multi-group-per-fold placement (not just 1-group-per-fold) | unit | exact |
| AT-S1d | `shuffle=true` + grouped: group-span order permuted, within-group row order preserved | unit | exact |
| AT-S3b | degenerate 0-row fold ⇒ typed error via the REAL `fit`/`staged_predict` paths, never NaN | unit | — |
| AT-S5 | `cv(pool, builder, ["RMSE"], folds=<fixed>)` per-iteration `test-`/`train-` mean/std, with the oracle `params` dict EXPLICITLY setting `boost_from_average=True` + `bootstrap_type="No"` (§1.1) | `catboost.cv(pool, params, folds=<same>, shuffle=False)` fixtures | ≤1e-5 |
| AT-S5b | serial vs `cpu`-parallel (`rayon`) fold execution ⇒ byte-identical `CvResult` | unit (`cpu` feature) | exact |
| AT-S5c | grouped ranking loss/metric (e.g. NDCG) through the full `cv()` pipeline | integration (`cv_grouped_ranking_test.rs`) | — (no error, correct aggregation) |
| AT-S6 | `catboost_rs.cv(...)` dict columns vs upstream; unsupported model raises | Python (uv 3.12) | ≤1e-5 |

---

## 7. Impact Scope

- **`crates/cb-data/src/pool.rs`** — `local`. Adds `select_rows`; no existing
  method changes. `Pool` blast radius unaffected `[VERIFIED: CODEGRAPH pool.rs]`.
- **`crates/catboost-rs/src/cv.rs`** (NEW) — `local`. New leaf module calling
  existing seams; `builder.rs`/`model.rs`/`metrics.rs` are **called, not
  modified** `[VERIFIED: CODEGRAPH builder.rs:334, model.rs:189, metrics.rs:44]`.
- **`crates/catboost-rs/src/lib.rs`** — `local`. Add `mod cv;` + `pub use
  cv::{cv, CvResult, CvFold, CvType, make_cv_folds};`.
- **`crates/catboost-rs-py/src/`** — `external/public`. New `cv` `#[pyfunction]`
  on the Python module. Additive.
- **`crates/cb-oracle/fixtures/cv/`** (NEW) + generator arm — `local`. Additive
  fixtures + a `gen_*` function; existing fixtures untouched.
- **Tests** — new `cv_test.rs` (unit), `pool_test.rs` additions,
  `cv_oracle_test.rs` (integration), Python parity test.

No persistence/schema/event/cache/config/flag impact. No public contract of an
existing symbol changes.

---

## 8. Compatibility and Migration

- **Purely additive.** No existing signature, serialization format, or behavior
  changes. `CatBoostBuilder`, `Model`, `eval_metric`, `Pool`, and the training
  path are read-only dependencies (`select_rows` is a new method, not a change).
- **Naming parity:** the free-function name `cv`, the Python `catboost_rs.cv`,
  and the column names `test-<M>-mean` / `test-<M>-std` / `train-<M>-mean` mirror
  upstream `catboost.cv` for drop-in familiarity
  `[VERIFIED: WEB catboost.ai cv docs]`.
- No migration steps; no rollout flag. Rollback = revert the additive module +
  `select_rows` + fixtures.
- **Intentional divergence from upstream's combined-argument error behavior:**
  upstream `catboost.cv()` RAISES when a caller supplies both `folds` AND any
  of `fold_count`/`shuffle`/`partition_random_seed`/`inverted`
  (`core.py:7164-7167`). This plan's Rust `cv(fold_count: usize, ...,
  folds: Option<&[Vec<usize>]>)` cannot represent "unset" for `fold_count` (not
  `Option<usize>`), so `folds = Some(...)` silently overrides `fold_count`/
  `shuffle`/`seed` rather than erroring — a deliberate, documented adaptation,
  not a functional bug (the oracle fixtures never exercise the conflicting-args
  case).

---

## 9. Risks and Open Questions

| Risk | Consequence | Mitigation |
|------|-------------|------------|
| Upstream `shuffle=True` fold-assignment RNG not reproduced ≤1e-5 | shuffle-path oracle would drift | DEFAULT oracle uses explicit `folds=` (upstream-supported), sidestepping assignment RNG; our `shuffle=True` is deterministic but NOT claimed upstream-parity (§2 Non-goals) `[UNVERIFIED — deferred, non-blocking]` |
| **RESOLVED (was CRITICAL): per-fold vs. global quantization** | — | Empirically REFUTED global sharing; per-fold `fit(select_rows(train))` matches upstream's actual per-fold behavior exactly. See §1.1. `[VERIFIED: LOCAL empirical experiment]` |
| **RESOLVED (was the real root cause of an initially-observed divergence): `boost_from_average`/`bootstrap_type` default mismatch between the raw `catboost.cv()` dict API and the Estimator-class defaults `CatBoostBuilder` mirrors** | oracle fixtures would fail ≤1e-5 for a reason unrelated to cv's architecture | TASK-01's fixture-generation `params` dict MUST explicitly set `"boost_from_average": True` and `"bootstrap_type": "No"`, matching `CatBoostBuilder`'s actual defaults (`builder.rs:108,110`). See §1.1. `[VERIFIED: LOCAL empirical experiment — divergence collapses to ~1e-9]` |
| Per-fold training nondeterminism (quantization) | oracle flakiness | Rust `fit` is deterministic and already oracle-locked; fixed dataset + fixed folds + fixed seed. Cross-check the frozen-fixtures discipline `[VERIFIED: LOCAL MEMORY ctr-model-loading]` |
| **RESOLVED (was open): std ddof convention** | — | Empirically confirmed SAMPLE std, ddof=1 (`/(F-1)`), not population. §5-S4 pinned. `[VERIFIED: LOCAL empirical experiment]` |
| `staged_predict` model-class guard rejects the trained model | cv unusable on that model kind | In scope by design: numeric/float-only first slice; the typed `UnsupportedModel` is the CORRECT behavior, asserted in AT-S5/S6 `[VERIFIED: CODEGRAPH model.rs:144-171]` |
| Per-fold iteration counts differ (e.g. `use_best_model` truncation) | ragged aggregation | Guard: mismatched curve lengths ⇒ typed error (S4); first slice trains fixed `iterations`, no early stopping in cv |
| **`rayon`-parallel fold execution vs. GPU-backend concurrency safety** (was unexamined) | undefined behavior on GPU-feature builds if unsafe | Gated to `cpu` feature only (serial under any GPU feature); see §5-S5. `[VERIFIED: CODEGRAPH gpu_backend.rs:57-63 GpuBackend/ComputeClient shape]` |
| **Grouped k-fold + `shuffle` interaction and ranking end-to-end coverage** (was unspecified/untested) | in-scope grouped/ranking capability could ship broken | `shuffle` × grouped semantics now pinned (§5-S1); multi-group + shuffle+grouped unit tests + a grouped-ranking integration test added (AT-S1c/d, AT-S5c) |
| **Empty explicit `folds` list (`Some(&[])`) bypasses `make_cv_folds`'s `fold_count>=2` validation entirely** (found at post-cap verification pass 4 — `sum_f64` never rejects an empty slice, returning `0.0`, so `0.0/0 == NaN` would otherwise silently reach `Ok(CvResult)`) | silently NaN-filled result, violating the "never NaN" invariant | TWO independent guards added: an entry-point check in `cv()` (S5, §4 `# Errors`) rejecting `Some(&[])` before aggregation is ever attempted, AND a defense-in-depth `F==0` check inside `aggregate_folds` itself (S4). AT-S4b/AT-S5d. `[VERIFIED: CODEGRAPH cb-core/src/reduction.rs:32-38 sum_f64([])==0.0]` |
| Lint gate is CLIPPY not build (`unwrap`/`expect`/`panic`/`indexing_slicing` denied) | CI red despite `cargo build` green | Gate new prod with `cargo clippy -p catboost-rs --lib --no-deps` / `-p cb-data`; partition/aggregation return typed `CbError`/`CatBoostError` `[VERIFIED: LOCAL MEMORY fstr03-plan gotchas]` |
| Test-mount omission runs 0 tests silently | false green | Mount `cv_test.rs` (`#[cfg(test)] mod cv_test;`) `[VERIFIED: LOCAL calc-metrics facade mount]` |
| Python cannot link locally (system python 3.14) | Python test unrunnable in-env | Build/run via `uv venv --python 3.12`; `cargo check -p catboost-rs-py` compile-verify `[VERIFIED: LOCAL MEMORY fstr03-plan]` |
| `pairs` re-indexing after `select_rows` | invalid pair ids | First-slice CV drops `pairs` in `select_rows` (documented S2); pairwise-ranking CV is a later slice |
| `fold_count`+`folds` combined-argument behavior diverges from upstream's error | Python user surprise, not a functional bug | Documented explicitly in §8 as an intentional adaptation (Rust `fold_count: usize` cannot represent "unset") |

**Open questions**

1. **Upstream `shuffle=True` fold-assignment parity** — deferred; the first
   slice's default oracle uses explicit `folds=`. `[UNVERIFIED — resolve in a
   later slice by instrumenting upstream cv's split stream]`
2. ~~**std ddof**~~ — **RESOLVED**: sample std, ddof=1, empirically confirmed
   (§1.1, §5-S4).
3. **Metric-from-loss default** — confirm the exact upstream default metric name
   derived from `loss_function` when `metrics=None` (e.g. `RMSE` for
   regression). `[INFERRED — default to the loss's canonical metric; verify at
   Python-parity time]`

---

## 10. Traceability and Sources

- **Reuse targets:** `crates/catboost-rs/src/builder.rs:334` (`fit`);
  `crates/catboost-rs/src/model.rs:135,144,189` (`predict`,
  `ensure_scalar_oblivious`, `staged_predict`);
  `crates/catboost-rs/src/metrics.rs:44` (`eval_metric`);
  `crates/cb-train/src/permutation.rs:109` (`fisher_yates_permutation`);
  `crates/cb-core/src/rng.rs:141` (`TFastRng64`); `cb_core::sum_f64`
  (`crates/cb-core/src/reduction.rs:32` — corrected citation) — all
  `[VERIFIED: CODEGRAPH]`.
- **Subset target:** `crates/cb-data/src/pool.rs:82-205` (`from_validated_columns`
  private ctor + accessors) `[VERIFIED: CODEGRAPH]`.
- **Contrast (NOT reused):** `crates/cb-train/src/fold.rs:256` (`create_folds`
  builds boosting learning/averaging folds, not CV partitions)
  `[VERIFIED: CODEGRAPH]`.
- **Facade error mapping precedent:** `crates/catboost-rs/src/error.rs:37,113`
  (`Train(#[from] CbError)`, `UnsupportedModel`) `[VERIFIED: CODEGRAPH]`.
- **Python binding precedent:** `crates/catboost-rs-py/src/estimator.rs:24,204,236`;
  `params.rs:451` (`make_builder`) `[VERIFIED: CODEGRAPH]`; ORCH-04 `utils.eval_metric`
  submodule pattern (`.planning/phases/20-orchestration/calc-metrics/PLAN.md
  TASK-07`) `[VERIFIED: LOCAL]`.
- **Upstream API:** `catboost.cv` signature + return columns
  `[VERIFIED: WEB https://catboost.ai/docs/en/concepts/python-reference_cv]`.
- **Sibling SPEC (house style):** `.planning/phases/20-orchestration/calc-metrics/
  SPEC.md` (ORCH-04) `[VERIFIED: LOCAL]`.
- **Greenfield confirmation:** `grep "select_rows|fn cv\b|cross_valid|k_fold"`
  over `crates/` — no CV surface `[VERIFIED: LOCAL]`.

---

## 11. Implementation Evidence

> Spec+plan only — NOT implemented. Every per-spec `implementation_state` is
> `unimplemented`; the document lifecycle is `status: draft`. No production code
> was authored. (No TreeFinder/PageIndex MCP write target confirmed in-session;
> this SPEC is the effective local spec store per the frontmatter
> `pageindex_pending` note.)
