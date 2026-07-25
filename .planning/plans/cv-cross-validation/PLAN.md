---
title: "ORCH-01 — cv() cross-validation surface — TDD Implementation Plan"
phase: 20-orchestration
slice: cv-cross-validation
plan_version: 1
status: planned
updated_at: 2026-07-19T00:00:00Z
source_spec: .planning/plans/cv-cross-validation/SPEC.md
source_research: "Phase-20 orchestration research pass (ORCH-01/02/03)"
gsd_used: false
---

# ORCH-01 — TDD Implementation Plan

Plan-only artifact. No production code authored here. Every file/symbol/command
below is verified against on-disk source via CodeGraph + Read (evidence inline).
The training, staged-prediction, metric, shuffle, and summation seams are
**reused, never modified** (D-04 no-regression): `CatBoostBuilder::fit`,
`Model::staged_predict`, `catboost_rs::eval_metric`,
`cb_train::fisher_yates_permutation`, `cb_core::sum_f64`.

## 0. Goal-backward derivation

Acceptance outcomes (SPEC §6) drive the task set:

| Acceptance | Observable success | Task |
|---|---|---|
| AT-S1a/b | `make_cv_folds` disjoint+covering blocks; same-seed determinism; grouped whole-group; inverted swap; typed error paths | TASK-03 |
| AT-S2 | `Pool::select_rows` gathers every column; empty preserved; OOB skipped | TASK-02 |
| AT-S3 | one fold's per-iteration test/train curve == manual `fit`+`staged_predict`+`eval_metric` (≤1e-5) | TASK-05 |
| AT-S4 | cross-fold mean/sample-std (ddof=1) over hand curves match closed form; `F==0`/`F==1` ⇒ typed error, never NaN | TASK-04 |
| AT-S5 | `cv(..., folds=<fixed>)` per-iteration `test-`/`train-` mean/std ≤1e-5 vs `catboost.cv` fixtures; `folds=Some(&[])` ⇒ typed error, never NaN | TASK-01 (fixtures) + TASK-06 |
| AT-S6 | `catboost_rs.cv(...)` dict columns ≤1e-5 vs upstream; unsupported model raises | TASK-07 |

Reused seams (verified, do NOT modify):

- `CatBoostBuilder::fit(&self, pool: &Pool) -> Result<Model, CatBoostError>` —
  `crates/catboost-rs/src/builder.rs:334-402` (quantizes float borders via
  `select_borders_greedy_logsum`, calls `cb_train::train`). `[VERIFIED: CODEGRAPH]`
- `Model::staged_predict(&self, pool, ntree_start: Option<usize>, ntree_end:
  Option<usize>, eval_period: Option<usize>) -> Result<Vec<Vec<f64>>,
  CatBoostError>` — `crates/catboost-rs/src/model.rs:189-208`; defaults
  `0/0/1` = one stage per tree, last stage == `predict`; guarded by
  `ensure_scalar_oblivious` (`model.rs:144-171`, rejects `approx_dimension>1`,
  non-symmetric/region trees, CTR data with typed `UnsupportedModel`).
  `[VERIFIED: Read]`
- `Model::predict(&self, pool: &Pool) -> Result<Vec<f64>, CatBoostError>` —
  `model.rs:135`. `[VERIFIED: CODEGRAPH]`
- `catboost_rs::eval_metric(label:&[f64], approx:&[f64], metric:&str,
  weight:Option<&[f64]>, group_id:Option<&[u64]>) -> Result<f64, CatBoostError>`
  — `crates/catboost-rs/src/metrics.rs:44-59` (ORCH-04, shipped). `[VERIFIED: CODEGRAPH]`
- `cb_train::fisher_yates_permutation(n: usize, seed: u64) -> Vec<i32>` —
  `crates/cb-train/src/permutation.rs:109`; `pub` + re-exported at
  `crates/cb-train/src/lib.rs:75`. Uses `cb_core::rng::TFastRng64::from_seed`.
  `[VERIFIED: CODEGRAPH + LOCAL grep]`
- `cb_core::sum_f64` — the workspace summation chokepoint, DEFINED at
  `crates/cb-core/src/reduction.rs:32` (corrected citation — an earlier draft
  cited `crates/cb-train/src/metrics.rs:292`, a call site, not the
  definition); `cb-core` is a `catboost-rs` dep
  (`crates/catboost-rs/Cargo.toml:26`). `[VERIFIED: CODEGRAPH + LOCAL]`
- `Pool` private ctor `from_validated_columns` + accessors `n_rows`,
  `float_features`, `cat_features`, `text_features`, `embedding_features`,
  `label`, `weights`, `group_id`, `subgroup_id`, `pairs`, `baseline` —
  `crates/cb-data/src/pool.rs:82-205`. The ctor is `pub(crate)`, so `select_rows`
  MUST be an `impl Pool` method inside `cb-data`. `[VERIFIED: CODEGRAPH]`
- `CatBoostError::Train(#[from] cb_core::CbError)` (`error.rs:37`),
  `UnsupportedModel(String)` (`error.rs:113`) — the typed error surface.
  `?` on a `CbResult` converts to `CatBoostError::Train` directly. `[VERIFIED: CODEGRAPH]`
- Python: `EstimatorBase` (`estimator.rs:24`), `make_builder(&BTreeMap, py) ->
  PyResult<CatBoostBuilder>` (`params.rs:451`), `data_to_pool(py, x, y) ->
  PyResult<Pool>` (`estimator.rs:236`), `fit_pool` (`estimator.rs:204`); error
  chokepoint `PyCbError` / `errors::to_pyerr` (ORCH-04 TASK-07 precedent).
  `[VERIFIED: CODEGRAPH + LOCAL calc-metrics PLAN.md]`
- Contrast (NOT reused): `cb_train::create_folds` (`fold.rs:256`) builds boosting
  learning/averaging permutation folds — NOT disjoint CV partitions. `[VERIFIED: CODEGRAPH]`

Layering note (load-bearing): the per-fold loop and public `cv` MUST live in the
`catboost-rs` facade because only the facade can call `CatBoostBuilder::fit` /
`Model::staged_predict` (both facade symbols); `cb-train`/`cb-data`/`cb-model`
never depend on the facade. `Pool::select_rows` MUST live in `cb-data` (private
ctor access). Partitioning + aggregation are pure and co-located in the facade
`cv.rs` (they only need `fisher_yates_permutation` + `sum_f64`, both facade deps).

Model-class scope (load-bearing): `staged_predict`'s `ensure_scalar_oblivious`
guard bounds cv() to scalar/oblivious/float-only models — the numeric-regression
first slice. A CTR/categorical/multiclass model yields the typed
`UnsupportedModel`, asserted (not worked around) in TASK-06/07.

## 1. Execution order & waves

```
Wave A (parallel):  TASK-01 (fixtures, Python)  ∥  TASK-02 (Pool::select_rows, cb-data)  ∥  TASK-03 (make_cv_folds, facade cv.rs)
Wave B:             TASK-04 (aggregation, facade cv.rs)     depends: TASK-03  (same prod file)
Wave C:             TASK-05 (per-fold loop, facade cv.rs)   depends: TASK-02, TASK-04  (same prod file)
Wave D:             TASK-06 (public cv + oracle test)       depends: TASK-01, TASK-05
Wave E:             TASK-07 (Python catboost_rs.cv)         depends: TASK-06
```

Dependency graph:

```
TASK-01 ─┐
TASK-02 ─┼────────────────┐
TASK-03 ─> TASK-04 ─> TASK-05 ─> TASK-06 ─> TASK-07
```

Acyclic. Wave-A tasks touch disjoint files/crates (Python generator+fixtures ∥
`crates/cb-data/src/pool.rs` ∥ `crates/catboost-rs/src/cv.rs`) — safe to
parallelize. TASK-03/04/05 all edit `cv.rs` ⇒ strictly sequential (write
conflict). TASK-02 (`pool.rs`) is independent of TASK-03/04 but TASK-05 needs it.

## 2. Spec-ID → task coverage

| Spec | Behavior | Task(s) |
|---|---|---|
| ORCH-01-S1 | fold-index partitioning (`make_cv_folds`) | TASK-03 |
| ORCH-01-S2 | `Pool::select_rows` | TASK-02 |
| ORCH-01-S3 | per-fold train + staged eval loop | TASK-05 |
| ORCH-01-S4 | cross-fold mean/std aggregation | TASK-04 |
| ORCH-01-S5 | Rust facade `cv` + oracle | TASK-01 + TASK-06 |
| ORCH-01-S6 | Python `catboost_rs.cv` | TASK-07 |

Every S1..S6 covered; TASK-01 is a fixture prerequisite for S5.

---

## TASK-01 — Fixed-fold `cv/` oracle fixtures (supports ORCH-01-S5)

- **Spec refs:** ORCH-01-S5 (fixture half of AT-S5). Primary failure reason:
  fixtures do not reproduce a runnable `catboost.cv` ground truth.
- **Goal / completion:** committed frozen fixtures under
  `crates/cb-oracle/fixtures/cv/` — a fixed numeric-regression `X.npy`/`y.npy`
  (float64), an explicit fold assignment (`folds.json`: list of test-row index
  lists), the fixed params (`params.json`: `iterations`, `learning_rate`,
  `depth`, `loss_function="RMSE"`, `border_count`, **and, critically,
  `"boost_from_average": True` + `"bootstrap_type": "No"` explicitly set — see
  the empirical-investigation note in the Green step below; omitting either
  key makes the fixture NOT reproduce what `CatBoostBuilder`'s Rust defaults
  actually train**), and the upstream cv output columns as `.npy`
  (`test_rmse_mean.npy`, `test_rmse_std.npy`, `train_rmse_mean.npy`,
  `train_rmse_std.npy`, each length `iterations`) plus `summary.json`.
  Completion = files exist and `load_f64_vec` reads them.
- **Prerequisites:** none (parallel with TASK-02, TASK-03).
- **Files:**
  - Create (generator): `crates/cb-oracle/generator/gen_cv_fixtures.py` (mirror
    the `gen_ranking_fixtures.py` structure — `numpy`, float64, `summary.json`).
    Offline tool; no prod code.
  - Create (generated, committed): `crates/cb-oracle/fixtures/cv/{X,y}.npy`,
    `.../folds.json`, `.../params.json`, `.../test_rmse_mean.npy`,
    `.../test_rmse_std.npy`, `.../train_rmse_mean.npy`, `.../train_rmse_std.npy`,
    `.../summary.json`.
- **CodeGraph/Read evidence:** upstream signature `cv(pool, params, folds=,
  shuffle=, as_pandas=)` and columns `test-RMSE-mean/std`, `train-RMSE-mean/std`
  `[VERIFIED: WEB catboost.ai/docs/.../python-reference_cv]`; float64 fixture +
  `summary.json` discipline (ORCH-04 TASK-01 precedent
  `.planning/phases/20-orchestration/calc-metrics/PLAN.md:116-169`); `load_f64_vec`
  reads `.npy` as float64.
- **Red:** N/A (data-prep; its "red" is TASK-06's oracle test failing for want of
  fixtures). Guard: after generation, a `python -c` load asserts each expected
  `.npy` is a finite length-`iterations` vector.
- **Green (generation intent):** build a `catboost.Pool(X, y)`; pick an explicit
  fold split (`folds = KFold(n_splits=3, shuffle=False)` index lists or a
  hand-written list); call
  `df = catboost.cv(pool, params={"iterations":10,"learning_rate":0.1,
  "depth":4,"loss_function":"RMSE","border_count":128,
  "boost_from_average":True,"bootstrap_type":"No"}, folds=<fixed>,
  shuffle=False, as_pandas=True)`; save `df["test-RMSE-mean"]` etc. to `.npy`
  (float64).
  **Empirical finding (was SPEC §9 Q1-CRITICAL, now resolved):** an initial
  fixture WITHOUT `boost_from_average`/`bootstrap_type` explicitly set produced
  a ~0.17-absolute divergence against this plan's `cv()` — traced (via
  `catboost.cv(..., return_models=True)` + `model.get_borders()` /
  `get_scale_and_bias()` per fold) to `catboost.cv()`'s raw dict-based native
  call NOT applying the `CatBoostRegressor` estimator class's Python-side
  default-injection (`boost_from_average` silently `False`/bias-0 at the raw
  `_cv()` layer, vs. `True` at the estimator-class layer, which
  `CatBoostBuilder`'s Rust default mirrors, `builder.rs:108`). Explicitly
  setting BOTH keys in the fixture's `params` dict (matching
  `CatBoostBuilder`'s actual defaults, `builder.rs:108,110`) collapsed the
  divergence to ~1e-9. **This is now a MANDATORY fixture-generation
  requirement, not optional tuning** — a fixture missing either key does not
  reproduce what the Rust `cv()` under test actually trains, independent of
  whether the Rust implementation is correct.
  **Open-question resolution (SPEC §9 Q2, now RESOLVED):** the `-std` column
  is SAMPLE std (ddof=1, divide by `F-1`) — confirmed by matching
  `numpy.std(per_fold_values, ddof=1)` exactly against the generated
  `test-RMSE-std`, and confirmed different from `ddof=0`. Record `ddof=1` in
  `summary.json`; this pins the S4 aggregation formula (§ below). **Open-question
  (SPEC §9 Q3):** confirm the default metric derived from `loss_function="RMSE"`
  is `RMSE`; record it.
- **Refactor:** none (fixtures are frozen artifacts).
- **Systematic parity self-check (added per PLAN-CHECK pass 3 MAJOR finding —
  the `boost_from_average`/`bootstrap_type` fix closed ONE discovered
  divergence source empirically, but nothing in this task systematically rules
  out a THIRD, still-undiscovered `CatBoostBuilder`-default that upstream's raw
  `cv()` dict API also fails to inject).** Before freezing the fixture, for
  EACH fold: call `catboost.cv(..., return_models=True)` to get that fold's
  actual trained model, and separately fit a standalone
  `catboost.CatBoostRegressor(**params).fit(Pool(X[train_idx], y[train_idx]))`
  on the identical row subset; assert `model.get_borders()` (per feature) AND
  `model.get_scale_and_bias()` (bias/scale) are IDENTICAL between the two
  (not just the two now-known keys) for every fold. Any further mismatch found
  this way must be root-caused and folded into the pinned `params` dict (the
  same way `boost_from_average`/`bootstrap_type` were), BEFORE the fixture is
  frozen — do not defer a newly-found divergence past this task the way the
  first two were found only after TASK-06 was already written.
- **Validation (offline, run-once/commit):**
  - `uv venv --python 3.12 && uv pip install catboost==1.2.10 'numpy<2' scikit-learn`
  - `.venv/bin/python crates/cb-oracle/generator/gen_cv_fixtures.py`
  - Sanity: `.venv/bin/python -c "import numpy,glob;[print(p,numpy.load(p).shape) for p in glob.glob('crates/cb-oracle/fixtures/cv/*rmse*.npy')]"`
  - Parity self-check (above): `.venv/bin/python -c "<per-fold get_borders/
    get_scale_and_bias comparison script>"` — must print zero mismatches
    across every fold before the fixture is committed.
- **Completion evidence:** the listed files present + loadable; `summary.json`
  records catboost 1.2.10, params, the explicit folds, and the pinned ddof;
  the per-fold `get_borders`/`get_scale_and_bias` parity self-check reports
  zero mismatches (recorded in `summary.json` alongside the other pins).
- **Compat/rollback:** additive new fixture dir; rollback = delete dir + generator.
- **Parallelization:** parallel with TASK-02, TASK-03 (disjoint files/crates).

---

## TASK-02 — `Pool::select_rows` row subset (ORCH-01-S2)

- **Spec refs:** ORCH-01-S2. Primary failure reason: a subset column is
  mis-gathered or a length invariant breaks.
- **Goal / completion:** `cb_data::Pool::select_rows(&self, indices: &[usize]) ->
  Pool` returns a Pool over the selected rows across every populated column;
  unit tests pass; `cargo clippy -p cb-data --lib --no-deps` clean.
- **Prerequisites:** none.
- **Files:**
  - Modify: `crates/cb-data/src/pool.rs` — add `pub fn select_rows` inside
    `impl Pool` (uses the `pub(crate) from_validated_columns` ctor; gathers each
    column with checked `.get(i).cloned()`, filtering OOB).
  - Create or modify: `crates/cb-data/src/pool_test.rs` — S2 unit tests (mounted
    via the crate's existing test-mount idiom; if `pool_test.rs` does not exist,
    add it and the `#[cfg(test)] #[path="pool_test.rs"] mod tests;` mount at the
    end of `pool.rs`).
- **CodeGraph/Read evidence:** `Pool` fields + `from_validated_columns(n_rows,
  float_features, cat_features, text_features, embedding_features, label,
  weights, group_id, subgroup_id, pairs, baseline)` — `pool.rs:82-108`; accessors
  `pool.rs:110-205`; SoA layout `float_features[f][row]`. `[VERIFIED: CODEGRAPH]`
- **Red:** in `pool_test.rs`:
  - `select_rows_gathers_columns` — build a Pool (via the crate's ingest/builder
    seam or a test helper) with 2 float cols len 4, a `label`/`weights`/`group_id`;
    `select_rows(&[3,1])` yields `n_rows()==2`, each float col == `[c[3],c[1]]`,
    `label`/`weights`/`group_id` gathered identically.
  - `select_rows_preserves_empty_columns` — an unweighted Pool (`weights()`
    empty) stays empty after subset.
  - `select_rows_skips_oob` — `select_rows(&[0, 99])` yields `n_rows()==1`.
  Expected INITIAL failure: `select_rows` unresolved ⇒ test build fails.
- **Green:** implement the gather:
  `let take = |col: &[T]| indices.iter().filter_map(|&i| col.get(i).cloned()).collect()`
  per column (float cols map over `self.float_features()`), `n_rows` = the gathered
  label/first-column length (or `indices` valid count); `pairs = Vec::new()`
  (dropped, documented). Return via `from_validated_columns`.
- **Refactor:** extract a small generic `gather<T: Clone>(col, indices)` helper;
  no behavior change. Regression scope: `cargo test -p cb-data` (Pool suite).
- **Validation:**
  - `cargo test -p cb-data --lib pool`
  - `cargo clippy -p cb-data --lib --no-deps`
- **Completion evidence:** 3 S2 tests green; clippy clean; no `unwrap`/indexing.
- **Compat/rollback:** additive method; rollback = remove `select_rows` + tests.
- **Parallelization:** parallel with TASK-01, TASK-03. Blocks TASK-05.

---

## TASK-03 — Fold-index partitioner `make_cv_folds` + `cv.rs` scaffold (ORCH-01-S1)

- **Spec refs:** ORCH-01-S1. Primary failure reason: fold membership is wrong
  (overlap / gap / group split / bad determinism).
- **Goal / completion:** `catboost_rs::cv::make_cv_folds(n, group_id, fold_count,
  shuffle, seed, cv_type) -> Result<Vec<CvFold>, CatBoostError>` + the `CvFold` /
  `CvType` types exist; unit tests pass; `cargo clippy -p catboost-rs --lib
  --no-deps` clean. Stands up the `cv.rs` module scaffold every later facade task
  extends.
- **Prerequisites:** none.
- **Files:**
  - Create: `crates/catboost-rs/src/cv.rs` — module doc; `use
    crate::error::CatBoostError;` `use cb_train::fisher_yates_permutation;`;
    `pub struct CvFold`, `pub enum CvType`, `pub fn make_cv_folds`. Mount tests:
    `#[cfg(test)] mod cv_test;` (facade root-mount idiom, cf. `mod metrics_test;`).
  - Create: `crates/catboost-rs/src/cv_test.rs` — S1 unit tests.
  - Modify: `crates/catboost-rs/src/lib.rs` — add `mod cv;` + (scaffold)
    `pub use cv::{CvFold, CvType, make_cv_folds};`.
- **CodeGraph/Read evidence:** `fisher_yates_permutation(n, seed) -> Vec<i32>`
  (`permutation.rs:109`, re-export `lib.rs:75`); `CatBoostError::Train(#[from]
  cb_core::CbError)` (`error.rs:37`) ⇒ build errors as
  `CatBoostError::Train(CbError::Degenerate(..))`; contrast `create_folds`
  (`fold.rs:256`, NOT reused). `[VERIFIED: CODEGRAPH]`
- **Algorithm (implement exactly):** for the PLAIN path (`group_id` empty),
  compute the ordered index list `order` = identity `0..n` when `!shuffle`,
  else `fisher_yates_permutation(n, seed)` cast to `usize`; split `order` into
  `fold_count` contiguous near-equal blocks (block `k` = the test set). For the
  GROUPED path (`!group_id.is_empty()`), first verify contiguity (each
  distinct id appears in one contiguous run — reuse the check pattern from
  `eval_grouped`'s `group_spans`, `metrics.rs`), form the group boundaries
  (a `Vec<Range<usize>>` of row-spans, one per group, in original order), THEN
  apply `shuffle` to the GROUP-SPAN order only (`fisher_yates_permutation(
  group_count, seed)` over the group list, NOT over individual rows), then
  assign the (possibly shuffled) whole-group spans round-robin/contiguous to
  `fold_count` folds (never split a group, and NEVER reorder rows within a
  group — required downstream by `eval_grouped`'s contiguity precondition).
  `train` = complement of `test` (sorted, preserving each retained group's
  internal row order). For `CvType::Inverted`, swap `train` and `test`.
  Validate `fold_count >= 2`, `fold_count <= n` (or `<= group_count`), `n > 0`;
  else `Err(CatBoostError::Train(CbError::Degenerate(msg)))`. No
  `unwrap`/indexing — use `.get`, checked arithmetic.
- **Red:** in `cv_test.rs`:
  - `plain_folds_partition` — `make_cv_folds(6, &[], 3, false, 0, Classical)`
    yields 3 folds; concatenated+sorted `test` sets == `0..6`; each pair of test
    sets disjoint; each `train` == complement.
  - `shuffle_determinism` — same `(n, seed)` gives equal folds; a different seed
    gives a different partition.
  - `grouped_whole_group` — `group_id = [0,0,1,1,2,2]`, `fold_count=3` ⇒ each
    fold's test set is exactly one group's rows (no split).
  - `grouped_multi_group_per_fold` — a LARGER group set (e.g. 6 groups,
    `fold_count=2`) so each fold's test set spans MULTIPLE whole groups, not
    just the trivial 1-group-per-fold case; assert every group's rows remain
    contiguous and unsplit in the result.
  - `shuffle_grouped_permutes_spans_not_rows` — `group_id` non-empty,
    `shuffle=true`: assert (a) group ASSIGNMENT to folds differs from the
    `shuffle=false` result (spans are permuted), (b) each group's row order
    WITHIN its span is unchanged from the input order (rows are never permuted
    within a group), and (c) a repeat call with the same seed reproduces the
    same assignment (determinism holds for grouped mode too).
  - `inverted_swaps_roles` — `Inverted` fold `k`'s `train` == the Classical fold
    `k`'s `test`.
  - `errors` — `fold_count=1` / `fold_count=7` (>n) / `n=0` / non-contiguous
    `group_id=[0,1,0]` each `.is_err()`.
  Expected INITIAL failure: `cv`/`make_cv_folds` unresolved ⇒ test build fails.
- **Green:** implement per the algorithm; the tests pass.
- **Refactor:** extract `contiguous_group_spans(group_id) -> CbResult<Vec<Range>>`
  and `block_bounds(len, k)` helpers; single validation front-door. No behavior
  change; regression scope = `cv_test.rs` (new leaf module; `permutation.rs`
  untouched).
- **Validation:**
  - `cargo test -p catboost-rs --lib cv`
  - `cargo clippy -p catboost-rs --lib --no-deps`
- **Completion evidence:** S1 tests green; clippy clean; `mod cv;` compiles.
- **Compat/rollback:** additive module; rollback = remove `cv.rs`/`cv_test.rs` +
  the two `lib.rs` lines.
- **Parallelization:** parallel with TASK-01, TASK-02. Blocks TASK-04/05 (same
  prod file).

---

## TASK-04 — Cross-fold mean/std aggregation (ORCH-01-S4)

- **Spec refs:** ORCH-01-S4. Primary failure reason: mean/std math wrong or a
  column mis-named / mis-shaped.
- **Goal / completion:** an aggregation fn in `cv.rs`
  (`fn aggregate_folds(per_fold: &[FoldCurves], metrics: &[&str], iterations:
  usize) -> Result<CvResult, CatBoostError>`, plus `pub struct CvResult`) that
  builds `test-<M>-mean/std` + `train-<M>-mean/std` per iteration via
  `cb_core::sum_f64`. Unit tests pass; clippy clean.
- **Prerequisites:** TASK-03 (same prod file `cv.rs`).
- **Files:**
  - Modify: `crates/catboost-rs/src/cv.rs` — add `pub struct CvResult`,
    an internal `FoldCurves` type (per-metric test/train `Vec<f64>`), and
    `aggregate_folds`.
  - Modify: `crates/catboost-rs/src/cv_test.rs` — S4 unit tests.
  - Modify: `crates/catboost-rs/src/lib.rs` — extend re-export to
    `pub use cv::{CvFold, CvType, CvResult, make_cv_folds};`.
- **CodeGraph/Read evidence:** `cb_core::sum_f64` reduction chokepoint
  (`crates/cb-core/src/reduction.rs:32` — corrected citation; the earlier
  `metrics.rs:292` was a call site, not the definition); `cb-core` is a facade
  dep (`Cargo.toml:26`). Column naming `test-<M>-mean` etc.
  `[VERIFIED: WEB catboost.ai cv docs]`. `[VERIFIED: CODEGRAPH/LOCAL]`
- **Formula (implement exactly):** for each metric `M`, dataset `D ∈ {test,
  train}`, iteration `i`: `mean[i] = sum_f64(&[curve_f[i] for f]) / F`;
  `std[i] = (sum_f64(&[(curve_f[i]-mean[i])^2 for f]) / (F-1)).sqrt()` —
  **SAMPLE std, ddof=1, empirically confirmed against upstream `catboost.cv`'s
  reported `-std` columns (SPEC §1.1/§9 Q2, RESOLVED — not population/ddof=0
  as an earlier draft assumed).** `F == 1` (a single fold) has no defined
  sample std: `Err(CatBoostError::Train(Degenerate))`, not division by zero.
  Insert into `BTreeMap` under `format!("{D}-{M}-mean")` / `"-std"`.
  `iterations = (0..iterations).collect()`. Mismatched curve length across
  folds ⇒ `Err(CatBoostError::Train(Degenerate))`.
- **Red:** in `cv_test.rs`:
  - `aggregate_two_folds` — two hand curves `[[1.0,2.0],[3.0,4.0]]` for
    `test-RMSE` ⇒ `test-RMSE-mean == [2.0, 3.0]`, `test-RMSE-std == [sqrt(2.0),
    sqrt(2.0)]` (sample std, ddof=1: `sum((x-mean)^2)/(F-1)` with `F=2` ⇒
    `(1.0+1.0)/1 = 2.0` ⇒ `sqrt(2.0)`); assert ≤1e-12.
  - `aggregate_single_fold_errs` — `F == 1` ⇒ `.is_err()` (no defined sample
    std), not a panic or NaN.
  - `aggregate_zero_folds_errs` — `F == 0` (empty `per_fold` slice) ⇒
    `.is_err()`, NEVER `NaN` (`sum_f64(&[])/0 == 0.0/0.0 == NaN` if unguarded —
    found at the post-cap extra verification pass: an empty explicit
    caller-supplied `folds` list, e.g. Rust `Some(&[])` or Python `folds=[]`,
    would otherwise reach this function with zero folds and silently produce
    `NaN` columns, violating the plan's own "never NaN" invariant). This is a
    DEFENSE-IN-DEPTH guard inside `aggregate_folds` itself — do not rely
    solely on the TASK-06 entry-point guard below to prevent this.
  - `aggregate_ragged_errs` — folds of differing curve length ⇒ `.is_err()`.
  Expected INITIAL failure: `aggregate_folds`/`CvResult` unresolved ⇒ build fails.
- **Green:** implement the formula (ddof=1, fixed — no longer read from
  TASK-01's `summary.json` as a variable, since it's now a pinned constant);
  tests pass.
- **Refactor:** extract `mean_std_over_folds(&[f64 per fold]) -> (f64, f64)`;
  reuse for both datasets/metrics. No new float sum outside `sum_f64` (D-08).
- **Validation:**
  - `cargo test -p catboost-rs --lib cv`
  - `cargo clippy -p catboost-rs --lib --no-deps`
- **Completion evidence:** S4 tests green; clippy clean.
- **Compat/rollback:** additive; rollback = remove aggregation + revert re-export.
- **Parallelization:** sequential after TASK-03 (same prod file).

---

## TASK-05 — Per-fold train + staged-eval loop (ORCH-01-S3)

- **Spec refs:** ORCH-01-S3. Primary failure reason: the per-fold, per-iteration
  metric wiring (train / staged-predict / eval) produces a wrong curve.
- **Goal / completion:** an internal `fn run_fold(pool, fold, builder, metrics)
  -> Result<FoldCurves, CatBoostError>` in `cv.rs` that fits on the train
  sub-Pool and computes per-metric per-iteration test+train curves via
  `staged_predict` + `eval_metric`. A decomposition unit test passes ≤1e-5.
- **Prerequisites:** TASK-02 (`Pool::select_rows`), TASK-04 (`FoldCurves` type,
  same prod file `cv.rs`).
- **Files:**
  - Modify: `crates/catboost-rs/src/cv.rs` — add `run_fold`.
  - Modify: `crates/catboost-rs/src/cv_test.rs` — S3 decomposition test.
- **CodeGraph/Read evidence:** `CatBoostBuilder::fit` (`builder.rs:334`),
  `Model::staged_predict(pool, None, None, None)` default schedule = one stage per
  tree, last stage == `predict` (`model.rs:189-208`), `catboost_rs::eval_metric`
  (`metrics.rs:44`), `Pool::select_rows` (TASK-02), `Pool::{label,weights,
  group_id}` accessors (`pool.rs:172-186`). `[VERIFIED: CODEGRAPH/Read]`
- **Algorithm (implement exactly):** `let train_pool = pool.select_rows(&fold.train);
  let test_pool = pool.select_rows(&fold.test); let model = builder.fit(&train_pool)?;
  let test_stages = model.staged_predict(&test_pool, None, None, None)?; let
  train_stages = model.staged_predict(&train_pool, None, None, None)?;` then for
  each metric `m` and each stage row `s`: `eval_metric(test_pool.label(),
  &test_stages[s], m, Some(test_pool.weights()), Some(test_pool.group_id()))?`
  (weights/group_id via `Option` — empty slice is fine). Collect into
  `FoldCurves`. Guard `test_stages.len() == train_stages.len()` else typed error.
  No `unwrap`/indexing — iterate with `.iter()`, `?`-propagate.
- **Red:** in `cv_test.rs`:
  - `run_fold_matches_manual` — build a small numeric-regression Pool (float
    columns + `y`), a `CatBoostBuilder` with `iterations=5, depth=2`, and a
    hand `CvFold{train, test}`. Assert `run_fold(...)`'s `test` RMSE curve equals
    the MANUAL composition (`builder.fit(select_rows(train))` →
    `staged_predict(select_rows(test))` → `eval_metric` per stage) within 1e-5.
  - `run_fold_degenerate_zero_row_errs` — a `CvFold` whose `test` OR `train`
    index list is empty (0 rows) ⇒ `run_fold(...)` returns a typed
    `Err(CatBoostError::...)`, propagated from the REAL error path — never a
    panic, never a NaN curve. **Corrected attribution (post-cap extra
    verification pass):** the two cases fail via DIFFERENT seams, both real,
    neither hand-asserted: (a) an empty TRAIN fold errors inside `fit` itself
    (`cb_train::boosting`'s empty-target guard, `boosting.rs:2332-2338`,
    `CbError::Degenerate("empty target")`); (b) an empty TEST fold does NOT
    error inside `staged_predict`/`predict_raw_staged` (`cb_model::apply.rs:465-532`
    silently returns empty per-stage `Vec<f64>`s for 0 objects — verified, not
    an error path) — the typed error instead comes from the SUBSEQUENT
    `eval_metric` call on that empty `approx` (`EvalMetric::eval`'s
    `approx.is_empty()` check, `crates/cb-train/src/metrics.rs:261-299`,
    `CbError::Degenerate("eval metric: empty eval set")`). Both cases still
    correctly produce a typed `Err`, never NaN — only the ATTRIBUTION of which
    seam raises it was wrong in an earlier draft (it is not `staged_predict`
    for the test-fold case). Test BOTH cases explicitly.
  Expected INITIAL failure: `run_fold` unresolved ⇒ build fails.
  (This is a self-consistency/decomposition oracle — no upstream needed; the
  upstream ≤1e-5 parity lands in TASK-06.)
- **Green:** implement `run_fold`; the decomposition test passes exactly (same
  seams, so equality holds to machine precision).
- **Refactor:** extract `curve_over_stages(stages, label, weight, group, metric)`;
  reuse for test+train. Regression scope: `cv` unit suite + S1/S4 green.
- **Validation:**
  - `cargo test -p catboost-rs --lib cv`
  - `cargo clippy -p catboost-rs --lib --no-deps`
- **Completion evidence:** S3 decomposition test green ≤1e-5; clippy clean; no
  panic path (UnsupportedModel/empty-fold errors propagate typed).
- **Compat/rollback:** additive; rollback = remove `run_fold` + its test.
- **Parallelization:** sequential after TASK-04 (same prod file). Needs TASK-02.

---

## TASK-06 — Public facade `cv` + upstream oracle (ORCH-01-S5)

- **Spec refs:** ORCH-01-S5. Primary failure reason: orchestration is wrong or the
  per-iteration `test-`/`train-` mean/std diverges from `catboost.cv` beyond 1e-5.
- **Goal / completion:** `catboost_rs::cv(pool, builder, metrics, fold_count,
  shuffle, partition_random_seed, inverted, folds) -> Result<CvResult,
  CatBoostError>` composes S1→S3→S4; the new oracle test passes ≤1e-5 on the
  TASK-01 fixtures.
- **Prerequisites:** TASK-01 (fixtures), TASK-05 (`run_fold` + all cv.rs pieces).
- **Files:**
  - Modify: `crates/catboost-rs/src/cv.rs` — add `pub fn cv` (validate `metrics`
    non-empty; **validate an explicit `folds` argument is non-empty too when
    `Some` — `Some(&[])` ⇒ `Err(CatBoostError::Train(Degenerate))` BEFORE
    calling `aggregate_folds` (found at the post-cap extra verification pass:
    an empty explicit `folds` list would otherwise reach zero-fold aggregation
    and silently produce `NaN` via `sum_f64(&[])/0`); this is the ENTRY-POINT
    half of the fix, complementing `aggregate_folds`'s own defense-in-depth
    guard from TASK-04**; build folds from `folds` if `Some` else
    `make_cv_folds`; map `inverted` → `CvType`; run each fold via TWO
    `#[cfg]`-gated bodies — **under
    `#[cfg(feature = "cpu")]`**: `folds.par_iter().map(run_fold).collect::<
    Result<Vec<_>,_>>()?` (rayon-parallel, order preserved by `collect`);
    **under any other backend feature (`wgpu`/`cuda`/`rocm`)**:
    `folds.iter().map(run_fold).collect::<Result<Vec<_>,_>>()?` (serial —
    see the rationale below). Then aggregate via `aggregate_folds`.
    **This `#[cfg]` split is a REQUIRED fix (was an unexamined MAJOR risk):**
    `CatBoostBuilder::fit` constructs its own local `GpuBackend::default()`
    per call (`builder.rs:379-391`), which under a GPU feature holds a
    `cubecl::client::ComputeClient<SelectedRuntime>` bound to a single
    physical device (`crates/cb-backend/src/gpu_backend.rs:57-63`);
    concurrently constructing/using multiple such clients from separate rayon
    worker threads against the same device is unverified and untested by this
    project (GPU tests run on `rocm` only, per CLAUDE.md). Serializing fold
    execution under any GPU feature sidesteps this unverified concurrency
    question entirely for the first slice, at the cost of losing per-fold
    parallelism only on GPU builds (training itself is already
    backend-accelerated).
  - Create: `crates/catboost-rs/tests/cv_oracle_test.rs` — integration oracle
    (carries `#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic,
    clippy::indexing_slicing)]` as the other integration tests do). ALSO add a
    `#[cfg(feature = "cpu")]`-gated `cv_serial_vs_parallel_byte_identical` test:
    run `cv(...)` twice on the SAME inputs — once via the serial `.iter()` path
    (temporarily forced, e.g. by calling an internal `#[cfg(test)]` serial-only
    helper or by asserting the parallel path's `CvResult` equals a
    directly-computed serial `Vec<FoldCurves>` aggregation — pick whichever is
    simplest given the actual `cv.rs` structure at Green time) and once via the
    normal (parallel, under `cpu`) path; assert the two `CvResult`s are
    byte-identical (`PartialEq`), not just ≤1e-5-close.
  - Create: `crates/catboost-rs/tests/cv_grouped_ranking_test.rs` — integration
    test: a small ranking Pool (`group_id` non-empty, a ranking loss e.g.
    `YetiRank`/`PairLogit` — whichever the builder already supports — and a
    ranking metric e.g. `"NDCG"`), grouped folds (`fold_count=2` or `3`,
    `group_id` set, `shuffle` either value), asserting `cv(...)` returns `Ok`
    with correctly-shaped `test-NDCG-mean`/`-std` columns and does NOT trip
    `eval_grouped`'s "non-contiguous group_id" `Degenerate` error. Closes a
    prior MAJOR gap: grouped k-fold was in-scope but had zero end-to-end
    pipeline coverage (only the pure `make_cv_folds` partitioner was unit-tested
    in TASK-03).
  - Modify: `crates/catboost-rs/src/lib.rs` — final re-export
    `pub use cv::{cv, CvFold, CvType, CvResult, make_cv_folds};`.
- **CodeGraph/Read evidence:** `rayon` facade dep (`Cargo.toml:39`); the
  `builder.rs:353` `rayon::join` precedent shows `par_iter().collect` preserves
  order; `GpuBackend`/`GpuTrainSession` concurrency shape
  (`crates/cb-backend/src/gpu_backend.rs:57-63`,
  `crates/cb-backend/src/gpu_runtime/session.rs:634-756`); oracle harness
  pattern (`cb_oracle::{compare_stage, load_f64_vec, Stage}`) from
  `crates/cb-train/tests/*_oracle_test.rs` (ORCH-04 TASK-03 precedent).
  `[VERIFIED: CODEGRAPH/LOCAL]`
- **Red:** `cv_oracle_test.rs`:
  - load `cv/{X,y}.npy`, `folds.json`, `params.json` (now including
    `boost_from_average: true`/`bootstrap_type: "No"`, per TASK-01), and the
    expected `test_rmse_mean.npy` / `test_rmse_std.npy` / `train_rmse_mean.npy`
    / `train_rmse_std.npy`;
  - build a `Pool` (via the facade/test ingest) + a `CatBoostBuilder` from
    `params.json`, calling `.boost_from_average(true).bootstrap_type(
    EBootstrapType::No)` EXPLICITLY on the builder (per PLAN-CHECK pass 3
    MINOR finding: relying implicitly on `CatBoostBuilder::new()`'s CURRENT
    defaults, without pinning them explicitly in the test, would silently
    break this oracle if a future change to `new()`'s defaults ever landed —
    explicit pinning makes the test's dependency on these two values visible
    and self-documenting, matching what TASK-01's fixture explicitly pins on
    the Python side); call `cv(&pool, &builder, &["RMSE"], 3, false, 0, false,
    Some(&folds))`;
  - `compare_stage(Stage::Predictions, &expected_test_mean,
    &result.columns["test-RMSE-mean"])` ≤1e-5 for each of the 4 columns;
  - `cv(..., &[], ...)` (empty metrics) ⇒ `.is_err()`.
  - `cv_empty_explicit_folds_errs` — `cv(&pool, &builder, &["RMSE"], 3, false, 0,
    false, Some(&[]))` (an explicit but EMPTY `folds` list) ⇒ `.is_err()`,
    NEVER a `NaN`-bearing `Ok(CvResult)`. Added at the post-cap extra
    verification pass: closes a real gap where `Some(&[])` would otherwise
    reach `aggregate_folds` with zero folds and silently produce
    `sum_f64(&[])/0 == NaN` columns.
  Expected INITIAL failure: `cv` unresolved ⇒ build fails; then value mismatch if
  the ddof/aggregation is off (fixed by TASK-01/TASK-04 pinning) or if
  `boost_from_average`/`bootstrap_type` weren't pinned in the fixture (§1.1).
- **Green:** implement `cv`; the 4 columns match ≤1e-5.
- **Refactor:** `cv` = validate → partition → `run_fold` per fold (serial or
  `cpu`-parallel) → aggregate; keep each step a single call. Regression scope:
  `cargo test -p catboost-rs` (facade suite + cv oracle) green.
- **Validation:**
  - `cargo test -p catboost-rs --test cv_oracle_test`
  - `cargo test -p catboost-rs --test cv_grouped_ranking_test`
  - `cargo test -p catboost-rs --lib cv` (includes the serial-vs-parallel
    byte-identity test under the default `cpu` feature)
  - `cargo clippy -p catboost-rs --lib --no-deps`
  - `cargo check -p catboost-rs --no-default-features --features wgpu`
    (compile-verify the serial-only GPU-feature path builds)
- **Completion evidence:** oracle green ≤1e-5 on all 4 columns; empty-metrics
  typed error; grouped-ranking integration test green; serial-vs-parallel
  byte-identity confirmed under `cpu`; `wgpu` compile-check green; clippy clean.
- **Compat/rollback:** additive; rollback = remove `cv` + the three new test
  files + revert re-export.
- **Parallelization:** sequential after TASK-05; needs TASK-01 fixtures.

---

## TASK-07 — Python `catboost_rs.cv` (ORCH-01-S6)

- **Spec refs:** ORCH-01-S6. Primary failure reason: Python column values diverge,
  wrong dict shape, or an error aborts instead of raising.
- **Goal / completion:** `catboost_rs.cv(pool, params, fold_count=3,
  inverted=False, partition_random_seed=0, seed=None, shuffle=True, folds=None,
  metrics=None)` returns the cv columns dict; `cargo check -p catboost-rs-py`
  compiles; parity ≤1e-5 under the uv-3.12 venv.
- **Prerequisites:** TASK-06 (facade `cv`).
- **Files:**
  - Create: `crates/catboost-rs-py/src/cv.rs` — a `#[pyfunction] cv` that
    `data_to_pool(py, pool_arg, None)` (accepts native Pool or `(X,y)`),
    `make_builder(&params_map, py)`, derives the metric list (`metrics` `str`/
    `list[str]`, or from `params["loss_function"]` default `RMSE`), converts
    `folds` (`Option<Vec<Vec<usize>>>`), maps `seed`→`partition_random_seed`,
    `py.detach(|| catboost_rs::cv(...))`, `.map_err(PyCbError)?`, then builds a
    Python `dict` from `CvResult` (`iterations` + each column as a list).
    **Documented intentional divergence from upstream (SPEC §8):** upstream
    `catboost.cv()` RAISES when `folds` is combined with any of
    `fold_count`/`shuffle`/`partition_random_seed`/`inverted`
    (`core.py:7164-7167`); this binding does NOT replicate that check — `folds`
    silently overrides the others, matching the Rust `cv()` signature's own
    behavior (which cannot represent "unset" for a plain `usize` fold_count).
    Not a functional bug; documented here so a future implementer doesn't
    "fix" it unprompted and diverge from the locked SPEC decision.
  - Modify: `crates/catboost-rs-py/src/lib.rs` — `mod cv;` +
    `m.add_function(wrap_pyfunction!(cv::cv, m)?)?;` in `#[pymodule] catboost_rs`.
  - Create: `crates/catboost-rs-py/tests/test_cv.py` — parity + error tests.
- **CodeGraph/Read evidence:** `#[pyfunction]`+`wrap_pyfunction!` registration
  (ORCH-04 TASK-07 precedent, `params.rs:238`/`lib.rs`); `data_to_pool`
  (`estimator.rs:236`), `make_builder` (`params.rs:451`); `PyCbError`/`to_pyerr`
  chokepoint; GIL own-before-detach discipline (copy Python buffers into
  Rust-owned `Vec`s before `py.detach`). `[VERIFIED: CODEGRAPH + LOCAL calc-metrics PLAN.md TASK-07]`
- **Red:** `test_cv.py`:
  - `import catboost_rs` + `catboost_rs.cv` resolves;
  - parity: on the same fixed `(X, y)`, params, and explicit `folds`,
    `catboost_rs.cv(...)` columns ≈ `catboost.cv(...)` ≤1e-5 for
    `test-RMSE-mean`/`std` and `train-RMSE-mean`/`std`;
  - `pytest.raises` on empty/unknown metric and on a categorical Pool
    (UnsupportedModel → mapped exception).
  If the uv venv is unavailable in-session, the equivalent red is `cargo check -p
  catboost-rs-py` failing to resolve `catboost_rs::cv` before TASK-06, then a
  compile-verified binding.
- **Green:** implement the pyfunction + registration; scalar-vs-list metric
  branch; map errors through `PyCbError`.
- **Refactor:** dedupe the label/approx/column extraction; no behavior change.
  Regression scope: `cargo check -p catboost-rs-py`; existing Python tests
  unaffected (additive function).
- **Validation:**
  - `cargo check -p catboost-rs-py`
  - Under uv 3.12: `uv venv --python 3.12 && uv pip install catboost==1.2.10
    'numpy<2' scikit-learn maturin pytest` then `maturin develop` + `pytest
    crates/catboost-rs-py/tests/test_cv.py`.
- **Completion evidence:** `cargo check` clean; Python parity ≤1e-5 on the 4
  columns; error `pytest.raises` green.
- **Compat/rollback:** additive `cv` function; rollback = remove `cv.rs` + the
  `lib.rs` registration + the test.
- **Parallelization:** sequential after TASK-06.

---

## 3. Cross-cutting guardrails (apply to every Rust task)

- **Clippy gate, not build:** `unwrap`/`expect`/`panic`/`indexing_slicing` are
  DENY in prod. Gate each Rust prod change with `cargo clippy -p <crate> --lib
  --no-deps` (`cb-data` for TASK-02, `catboost-rs` for TASK-03..06). Integration
  tests carry `#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic,
  clippy::indexing_slicing)]`. `[VERIFIED: LOCAL MEMORY fstr03-plan gotchas]`
- **Test mount:** the unit test files must be mounted (`#[cfg(test)] mod cv_test;`
  at the facade crate root for `cv_test.rs`; `#[cfg(test)] #[path="pool_test.rs"]
  mod tests;` for `pool.rs`) or `cargo test` silently runs 0 tests. `[VERIFIED:
  LOCAL calc-metrics facade mount; CLAUDE.md source/test separation]`
- **D-08 summation:** aggregation adds NO ad-hoc float sums; all reductions route
  through `cb_core::sum_f64`.
- **D-04 no-regression:** `builder.rs`, `model.rs`, `metrics.rs`,
  `permutation.rs`, and the training path are read-only. cv() only CALLS them.
  Confirm with `cargo test -p catboost-rs` + `cargo test -p cb-data` after
  TASK-06.
- **Model-class scope:** the numeric/float-only bound is enforced by
  `staged_predict`'s existing `ensure_scalar_oblivious` guard — cv() must NOT
  re-implement it; it propagates the typed `UnsupportedModel`.

## 4. Unresolved blockers / assumptions

1. **std ddof (SPEC §9 Q2).** Population vs sample std for the `-std` columns is
   pinned by inspecting TASK-01's generated fixture (`summary.json` records it);
   TASK-04's formula and AT-S4's expected values follow that pin. Blocks the
   TASK-06 oracle value only, not the code shape.
2. **`shuffle=True` upstream fold-assignment parity (SPEC §9 Q1).** Deferred; the
   first-slice oracle uses explicit `folds=` so this does NOT block any task. Our
   `shuffle=True` path is deterministic but not upstream-parity-claimed.
3. **Metric-from-loss default (SPEC §9 Q3).** TASK-07 derives the default metric
   from `params["loss_function"]` (`RMSE` for regression); confirm at Python
   parity time. Does not block TASK-02..06 (Rust `cv` takes explicit `metrics`).
4. **uv-3.12 venv availability in-session.** TASK-01 (fixture gen) and TASK-07
   (Python parity) require it; if unavailable, fixtures are produced
   run-once/commit offline and TASK-07 is compile-verified via `cargo check` with
   the `pytest` parity deferred to the venv (FSTR-03 precedent). Not a
   correctness blocker for TASK-02..06.
5. No TreeFinder/PageIndex write target confirmed for this corpus (SPEC
   frontmatter `pageindex_pending`); the SPEC under `.planning/plans/…` is the
   effective spec store. Not a planning blocker.

No requirement conflicts detected. No production code was authored.
