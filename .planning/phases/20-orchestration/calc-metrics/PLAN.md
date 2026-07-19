---
title: "ORCH-04 — Standalone eval_metric / calc_metric surface — TDD Implementation Plan"
phase: 20-orchestration
slice: calc-metrics
plan_version: 1
status: planned
updated_at: 2026-07-18T00:00:00Z
source_spec: .planning/phases/20-orchestration/calc-metrics/SPEC.md
source_research: .planning/plans/unimplemented-features-survey/research.md
gsd_used: false
---

# ORCH-04 — TDD Implementation Plan

Plan-only artifact. No production code authored here. Every file/symbol/command
below is verified against the on-disk source via CodeGraph + Read (evidence
inline). Metric arithmetic (`EvalMetric::eval`, `EvalMetric::eval_grouped`) is
**reused, never modified** (D-04 no-regression).

## 0. Goal-backward derivation

Acceptance outcomes (from SPEC §6) drive the task set:

| Acceptance | Observable success | Task |
|---|---|---|
| AT-S1a/b | `parse_metric("NDCG:top=2:type=Exp")` → exact variant; `parse_metric("bogus")` → `Err(CbError::Degenerate)` | TASK-02 |
| AT-S2 | RMSE/Logloss/MSLE (+weighted) on fixed preds ≤1e-5 vs new `calc_metrics/` fixtures | TASK-01 (fixtures) + TASK-03 |
| AT-S3 | 9 ranking metrics (+top2, +QueryAUC Classic/Ranking) ≤1e-5 vs existing `ranking_corpus/ranking_metrics/*.npy` | TASK-04 |
| AT-S4a/b | `["RMSE","Logloss"]` → 2 values; length-mismatch / unknown metric → typed `Err` | TASK-05 |
| AT-S5 | facade `eval_metric(...)` scalar value + `CatBoostError` propagation | TASK-06 |
| AT-S6 | `catboost_rs.utils.eval_metric` scalar & list vs upstream ≤1e-5; bad string → Python exception | TASK-07 |

Reused seams (verified, do NOT modify):

- `EvalMetric` enum + 13 variants — `crates/cb-train/src/metrics.rs:64-151`
  (`Rmse` 66, `Logloss` 68, `Msle` 75, `Ndcg{top,dcg_type,denominator}` 79,
  `Dcg{…}` 88, `Map{top,border}` 98, `Mrr{top,border}` 105, `Err{top}` 112,
  `PFound{top,decay}` 117, `PrecisionAt{top,border}` 124, `RecallAt{top,border}`
  131, `QueryAuc{auc_type}` 139, `Custom(CustomMetricHandle)` 150).
  `[VERIFIED: CODEGRAPH/Read metrics.rs]`
- `EvalMetric::eval(&self, approx:&[f64], target:&[f64], weights:&[f64]) -> CbResult<f64>`
  — flat metrics; `metrics.rs:244`. Flat arms: `Rmse|Logloss|Msle|Custom`; ranking
  arms return `Err(Degenerate)` (`metrics.rs:359-369`). `[VERIFIED: Read]`
- `EvalMetric::eval_grouped(&self, approx, target, weights, group_id, subgroup_id) -> CbResult<f64>`
  — ranking metrics; `metrics.rs:397`. Empty `group_id` ⇒ one group
  (`group_spans`, `metrics.rs:536-543`); non-contiguous `group_id` ⇒
  `Err(Degenerate)` (`metrics.rs:558-562`). `[VERIFIED: Read]`
- Param enums `DcgMetricType{Base,Exp}` (`ranking_metrics.rs:34`),
  `DcgDenominator{Position,LogPosition}` (`:45`), `AucType{Classic,Ranking}`
  (`:56`). `[VERIFIED: CODEGRAPH]`
- `cb_core::{sum_f64, CbError, CbResult}` — imported at `metrics.rs:35`. Parser +
  dispatch add NO new float sums (routing only). `[VERIFIED: Read]`
- lib re-export site `crates/cb-train/src/lib.rs:70` `pub use metrics::{EvalMetric, EvalMetricHistory};`. `[VERIFIED: Read]`
- Facade error mapping precedent `crates/catboost-rs/src/model.rs:169-193`
  (`feature_importance_with_data`, `CbError`→`CatBoostError`); `CatBoostError::Train(#[from] cb_core::CbError)` exists (`crates/catboost-rs/src/error.rs`). `[VERIFIED: Read]`
- Test-mount idiom `#[cfg(test)] #[path="X_test.rs"] mod tests;` — confirmed
  `crates/cb-model/src/ctr_data.rs:65` and `crates/cb-train/src/metrics.rs:616-618`. `[VERIFIED: Read]`
- Oracle harness: `cb_oracle::{compare_stage, load_f64_vec, Stage}`, ≤1e-5 via
  `compare_stage(Stage::Predictions, expected, actual)`; scalar-as-1-elem-vec
  pattern — `crates/cb-train/tests/ranking_metrics_oracle_test.rs:47-55`. `[VERIFIED: Read]`
- Generator arm precedent `gen_metrics_eval` — `crates/cb-oracle/generator/gen_ranking_fixtures.py:320-351`
  (`from catboost.utils import eval_metric`; `arr.reshape(-1)`; `.npy` + `summary.json`). `[VERIFIED: Read]`
- Python binding surface: `#[pyfunction]` + `wrap_pyfunction!` registration in
  `#[pymodule] catboost_rs` (`crates/catboost-rs-py/src/lib.rs:44-55`,
  `params.rs:238`); error chokepoint `errors::to_pyerr(&FacadeError)` +
  `PyCbError` newtype (`errors.rs:92,113`). `[VERIFIED: Read]`

Signature-mapping note (load-bearing): upstream + SPEC order is
`(label, approx, …)` but the reused seams take `(approx, target, …)`. `calc_metric`
maps `label → target`, `approx → approx`, passes `subgroup_id = &[]`. The ranking
oracle test already proves this mapping (`eval_grouped(approx, target, &[], group_id, &[])`).

Routing predicate (new, in `calc_metrics.rs`): a metric is *ranking* iff it
matches `Ndcg|Dcg|Map|Mrr|Err|PFound|PrecisionAt|RecallAt|QueryAuc`; else *flat*
(`Rmse|Logloss|Msle|Custom`). This mirrors the `use_group_weight`/`eval_one_group`
partition already in `metrics.rs:443-517`.

## 1. Execution order & waves

```
Wave A (parallel):   TASK-01 (fixtures/gen, Python)  ∥  TASK-02 (parser + module scaffold, Rust)
Wave B:              TASK-03 (S2 flat calc_metric)          depends: TASK-01, TASK-02
Wave C:              TASK-04 (S3 grouped calc_metric)       depends: TASK-03  (same prod file)
Wave D:              TASK-05 (S4 dispatch eval_metric + re-export) depends: TASK-03, TASK-04
Wave E:              TASK-06 (S5 facade)                    depends: TASK-05
Wave F:              TASK-07 (S6 Python)                    depends: TASK-06
```

Dependency graph:

```
TASK-01 ─┐
         ├─> TASK-03 ─> TASK-04 ─> TASK-05 ─> TASK-06 ─> TASK-07
TASK-02 ─┘
```

Acyclic. Only TASK-01 ∥ TASK-02 are parallel (disjoint files/crates:
Python generator+fixtures vs `crates/cb-train/src/calc_metrics.rs`). TASK-03/04/05
all edit `calc_metrics.rs` ⇒ strictly sequential (write conflict).

## 2. Spec-ID → task coverage

| Spec | Behavior | Task(s) |
|---|---|---|
| ORCH-04-S1 | parser `&str → EvalMetric` | TASK-02 |
| ORCH-04-S2 | flat final value (RMSE/Logloss/MSLE, weighted) | TASK-01 + TASK-03 |
| ORCH-04-S3 | grouped ranking final value | TASK-04 |
| ORCH-04-S4 | dispatch + validation (`eval_metric` list) | TASK-05 |
| ORCH-04-S5 | Rust facade `eval_metric`/`eval_metrics` | TASK-06 |
| ORCH-04-S6 | Python `catboost_rs.utils.eval_metric` | TASK-07 |

Every S1..S6 covered; TASK-01 is a fixture-prep prerequisite for S2.

---

## TASK-01 — Flat `calc_metrics/` oracle fixtures (supports ORCH-04-S2)

- **Spec refs:** ORCH-04-S2 (fixture half of AT-S2).
- **Goal / completion:** committed frozen fixtures under
  `crates/cb-oracle/fixtures/calc_metrics/` — `label.npy`, `approx.npy`
  (float64), plus per-scenario expected `.npy` from `catboost.utils.eval_metric`
  for `RMSE`, `Logloss`, `MSLE`, and (if the upstream signature supports it) a
  `weight.npy` + weighted-RMSE expected — each a 1-element float64 array; plus
  `summary.json`. Completion = files exist and `load_f64_vec` reads them.
- **Prerequisites:** none (parallel with TASK-02).
- **Files:**
  - Modify: `crates/cb-oracle/generator/gen_ranking_fixtures.py` — add a
    `gen_calc_metrics_flat()` arm + a `--calc-metrics` CLI flag (mirror
    `gen_metrics_eval` at `:320-351` and the `--metrics-eval` flag at `:359-363`).
    Alternatively a new sibling generator `gen_calc_metrics.py`; keep it in the
    same generator dir. Prod-code-free (offline tool).
  - Create (generated, committed): `crates/cb-oracle/fixtures/calc_metrics/{label,approx}.npy`,
    `.../rmse.npy`, `.../logloss.npy`, `.../msle.npy`, `.../summary.json`
    (+ `.../weight.npy`, `.../rmse_weighted.npy` if `weight=` supported).
- **CodeGraph/Read evidence:** generator pattern `gen_ranking_fixtures.py:320-351`
  (`from catboost.utils import eval_metric`, `np.asarray(value).reshape(-1)`,
  float64, `summary.json`); `_assert_f64` float64 guard `gen_inputs.py:38-42`;
  fixtures loaded as float64 by `load_f64_vec` (`ranking_metrics_oracle_test.rs:38`).
- **Red:** N/A (data-prep task; its "red" is TASK-03's oracle test failing for
  want of fixtures). Guard: after generation, a one-off `python -c` load asserts
  each expected `.npy` is a finite scalar.
- **Green (generation intent):** FIXED inputs (pinned seed, NO training — the
  nondeterminism-free surface). **`label` MUST be pinned to `[0,1]`** (checker
  MAJOR): upstream Logloss requires `target ∈ [0,1]` and `catboost.utils.eval_metric`
  will raise / diverge otherwise — a `[0,1]` label simultaneously satisfies RMSE,
  Logloss, AND MSLE (`1+label>0` holds trivially). `approx` = raw model outputs
  (RAW logit for Logloss); keep `approx > -1` so the MSLE `1+approx>0` log-domain
  guard (`metrics.rs:326-331`) is satisfied. (A single shared `(label, approx)`
  pair across all three metrics is therefore valid ONLY with label∈[0,1] and
  approx>-1.) For each metric call
  `eval_metric(label, approx, "<Metric>")`, reshape to a flat float64 array,
  save. **Open-question resolution (SPEC §9 Q1):** before writing the weighted
  fixture, confirm `catboost.utils.eval_metric` accepts `weight=` in 1.2.10
  (inspect the installed `catboost/utils.py` signature under the uv venv). If
  unsupported, OMIT `weight.npy`/`rmse_weighted.npy` and record in `summary.json`
  that weighting is covered by TASK-03's hand-computed unit assertion instead
  (the weight arithmetic in `EvalMetric::eval` is deterministic — `metrics.rs:266-296`).
- **Refactor:** none (fixtures are frozen artifacts).
- **Validation (offline, run-once/commit):**
  - `uv venv --python 3.12 && uv pip install catboost==1.2.10 'numpy<2'`
  - `.venv/bin/python crates/cb-oracle/generator/gen_ranking_fixtures.py --calc-metrics`
  - Sanity: `.venv/bin/python -c "import numpy,glob; [print(p, numpy.load(p)) for p in glob.glob('crates/cb-oracle/fixtures/calc_metrics/*.npy')]"`
- **Completion evidence:** the listed `.npy`/`summary.json` files present and
  loadable; `summary.json` records catboost version 1.2.10, seed, and per-scenario
  metric string + value.
- **Compat/rollback:** purely additive new fixture dir; rollback = delete the
  dir + revert the generator arm. Existing fixtures untouched.
- **Parallelization:** parallel with TASK-02 (disjoint files: Python/fixtures vs
  Rust module). No conflict.

---

## TASK-02 — Metric-descriptor parser + module scaffold (ORCH-04-S1)

- **Spec refs:** ORCH-04-S1. Primary failure reason: parser maps a descriptor to
  the wrong/absent `EvalMetric` variant.
- **Goal / completion:** `cb_train::calc_metrics::parse_metric(&str) -> CbResult<EvalMetric>`
  exists; unit tests in `calc_metrics_test.rs` pass; `cargo clippy -p cb-train
  --lib --no-deps` clean. Also stands up the module scaffold every later Rust task
  extends.
- **Prerequisites:** none.
- **Files:**
  - Create: `crates/cb-train/src/calc_metrics.rs` — module doc + `use cb_core::{CbError, CbResult}; use crate::EvalMetric; use crate::ranking_metrics::{DcgMetricType, DcgDenominator, AucType};` + `pub fn parse_metric(...)`. Mount tests at file end:
    `#[cfg(test)] #[path = "calc_metrics_test.rs"] mod tests;`.
  - Create: `crates/cb-train/src/calc_metrics_test.rs` — S1 unit tests.
  - Modify: `crates/cb-train/src/lib.rs` — add `pub mod calc_metrics;` and (scaffold
    only for now) `pub use calc_metrics::parse_metric;` near the `metrics` re-export
    (`lib.rs:70`).
- **CodeGraph/Read evidence:** variant shapes + defaults `metrics.rs:79-150`
  (defaults: `top=-1`, `border=0.5`, `decay=0.85`, `dcg_type=Base`,
  `denominator=LogPosition`, `auc_type=Classic`); param enums `ranking_metrics.rs:34,45,56`;
  mount idiom `metrics.rs:616-618` / `ctr_data.rs:65`; the enums are re-exported at
  `lib.rs` (`AucType, DcgDenominator, DcgMetricType` used by the ranking oracle
  test import `ranking_metrics_oracle_test.rs:24`).
- **Grammar (implement exactly):** split `descr` on `':'`; head = metric name
  (ASCII case-insensitive compare); tail tokens = `key=value`. Supported names →
  variants: `rmse→Rmse`, `logloss→Logloss`, `msle→Msle`, `ndcg→Ndcg{…}`,
  `dcg→Dcg{…}`, `map→Map{…}`, `mrr→Mrr{…}`, `err→Err{…}`, `pfound→PFound{…}`,
  `precisionat→PrecisionAt{…}`, `recallat→RecallAt{…}`, `queryauc→QueryAuc{…}`.
  Keys per metric: `top`(i64), `border`(f64), `decay`(f64), `type`
  (Base|Exp for NDCG/DCG; Classic|Ranking for QueryAUC), `denominator`
  (LogPosition|Position). Unknown name, unknown/duplicate key for the metric, or
  unparseable value ⇒ `Err(CbError::Degenerate(msg))` whose message enumerates the
  supported names (and the deferred set — SPEC §9 Q2: `AUC`, `Accuracy`, `F1`,
  `Precision`, `Recall`, `R2`, `MAE`, `MAPE`, and `Custom` which is
  program-constructed only, never string-parsed). No `unwrap`/`expect`/`panic`/
  indexing — use `str::split`, `.get()`, `str::parse().map_err(...)`.
- **Red:** in `calc_metrics_test.rs`, add:
  - `parse_ndcg_with_params` — `parse_metric("NDCG:top=2:type=Exp:denominator=Position")`
    `== Ok(EvalMetric::Ndcg{top:2, dcg_type:DcgMetricType::Exp, denominator:DcgDenominator::Position})`.
  - `parse_defaults` — `parse_metric("NDCG")` `== Ok(Ndcg{top:-1,Base,LogPosition})`;
    `parse_metric("rmse")` (lowercase) `== Ok(Rmse)`.
  - `parse_queryauc` — `parse_metric("QueryAUC:type=Ranking") == Ok(QueryAuc{AucType::Ranking})`.
  - `parse_rejects_unknown` — `parse_metric("NoSuchMetric").is_err()` and
    `parse_metric("NDCG:bogus=1").is_err()`.
  Expected INITIAL failure: `calc_metrics` module/`parse_metric` does not exist ⇒
  compile error (unresolved import), i.e. the whole test file fails to build.
- **Green:** implement `parse_metric` per the grammar; the four tests pass.
- **Refactor:** extract small `parse_i64`/`parse_f64`/`parse_top`/`parse_border`
  helpers; keep one match over the lowercased name. No behavior change; regression
  scope = `calc_metrics_test.rs` only (module is a new leaf; `EvalMetric` blast
  radius — 5 cb-train callers — untouched, `metrics.rs` unmodified).
- **Validation:**
  - `cargo test -p cb-train --lib calc_metrics`
  - `cargo clippy -p cb-train --lib --no-deps`
- **Completion evidence:** 4 S1 tests green; clippy clean; `pub mod calc_metrics;`
  compiles.
- **Compat/rollback:** additive module; rollback = remove file + the two `lib.rs`
  lines.
- **Parallelization:** parallel with TASK-01. Blocks TASK-03/04/05 (they extend
  the same prod file).

---

## TASK-03 — Flat final-value `calc_metric` (ORCH-04-S2)

- **Spec refs:** ORCH-04-S2. Primary failure reason: flat routing/`GetFinalError`
  value differs from `catboost.utils.eval_metric` beyond 1e-5.
- **Goal / completion:** `cb_train::calc_metrics::calc_metric(metric:&EvalMetric,
  label:&[f64], approx:&[f64], weight:&[f64], group_id:&[u64]) -> CbResult<f64>`
  computes flat metrics; the new flat oracle test passes ≤1e-5.
- **Prerequisites:** TASK-01 (fixtures), TASK-02 (module).
- **Files:**
  - Modify: `crates/cb-train/src/calc_metrics.rs` — add `pub fn calc_metric(...)`
    with an `is_ranking(&EvalMetric)->bool` helper. **BOTH arms are implemented as
    real seam delegations in this task** (checker MINOR — no `todo!`/`unimplemented!`/
    panic stub may exist, since TASK-03's own clippy gate DENYs panic): flat arm
    routes `metric.eval(approx, label, weight)`, ranking arm routes
    `metric.eval_grouped(approx, label, weight, group_id, &[])`. S2's red oracle
    test only exercises the flat path; the ranking path is oracle-LOCKED separately
    in TASK-04 (which adds tests, not new prod logic).
  - Create: `crates/cb-train/tests/calc_metrics_flat_oracle_test.rs` — oracle harness.
  - Modify: `crates/cb-train/src/lib.rs` — extend the re-export to
    `pub use calc_metrics::{parse_metric, calc_metric};`.
- **CodeGraph/Read evidence:** `EvalMetric::eval` signature + flat arms
  `metrics.rs:244-354` (arg order `approx, target, weights`; ranking arms error at
  `:359-369`); weighting is deterministic `metrics.rs:266-296`; oracle harness
  shape (fixture() helper, `load_f64_vec`, `compare_stage(Stage::Predictions,…)`,
  `#![allow(clippy::unwrap_used,…)]` on the integration test) —
  `eval_metrics_oracle_test.rs:23-52` and `ranking_metrics_oracle_test.rs:19-55`.
- **Red:** `calc_metrics_flat_oracle_test.rs`:
  - load `calc_metrics/{label,approx}.npy` + `rmse.npy`/`logloss.npy`/`msle.npy`;
  - for each, `let got = calc_metric(&metric, &label, &approx, &[], &[]).unwrap();`
    then `compare_stage(Stage::Predictions, &expected, &[got])`.
  - If TASK-01 emitted the weighted fixture: a `rmse_weighted` case with
    `weight` loaded; ELSE a hand-computed weighted-RMSE unit assertion in
    `calc_metrics_test.rs` (deterministic small vector).
  Expected INITIAL failure: `calc_metric` unresolved ⇒ test build fails.
- **Green:** implement full routing `if is_ranking(metric) {
  metric.eval_grouped(approx, label, weight, group_id, &[]) } else {
  metric.eval(approx, label, weight) }` — a REAL delegation on BOTH arms (no panic
  stub; the ranking arm is simply not oracle-tested until TASK-04). Flat values
  match ≤1e-5.
- **Refactor:** keep `is_ranking` as the single routing predicate reused by
  TASK-04/05. No sum added (delegates to `eval`, which already routes through
  `sum_f64`). Regression scope: flat oracle + S1 tests still green;
  `metrics.rs` untouched (D-04).
- **Validation:**
  - `cargo test -p cb-train --test calc_metrics_flat_oracle_test`
  - `cargo test -p cb-train --lib calc_metrics`
  - `cargo clippy -p cb-train --lib --no-deps`
- **Completion evidence:** flat oracle green ≤1e-5 for RMSE/Logloss/MSLE (+weighted);
  clippy clean.
- **Compat/rollback:** additive; rollback = remove `calc_metric` + the test +
  revert the `lib.rs` re-export line.
- **Parallelization:** sequential after TASK-02 (same prod file). Not parallel
  with TASK-04/05.

---

## TASK-04 — Grouped ranking final-value `calc_metric` (ORCH-04-S3)

- **Spec refs:** ORCH-04-S3. Primary failure reason: grouped routing/value differs
  from the committed ranking `.npy` beyond 1e-5, or group edge-cases mis-handled.
- **Goal / completion:** `calc_metric`'s ranking arm routes
  `metric.eval_grouped(approx, label, weight, group_id, &[])`; empty `group_id`
  ⇒ single group; non-contiguous `group_id` ⇒ typed error. New ranking oracle
  test passes ≤1e-5, reusing existing fixtures (NO new fixtures).
- **Prerequisites:** TASK-03 (same prod file; extends `calc_metric`).
- **Files:**
  - Modify: `crates/cb-train/src/calc_metrics.rs` — complete the ranking arm of
    `calc_metric`.
  - Create: `crates/cb-train/tests/calc_metrics_ranking_oracle_test.rs` — mirrors
    `ranking_metrics_oracle_test.rs` but drives `calc_metric(&metric, &target,
    &approx, &[], &group_id)` instead of `eval_grouped` directly.
- **CodeGraph/Read evidence:** `eval_grouped` signature + group semantics
  `metrics.rs:397-528`; `group_spans` empty→one-group `:536-543`, non-contiguous→
  `Degenerate` `:558-562`; existing fixtures list (`ndcg,dcg,map,mrr,err,pfound,
  precision_at,recall_at,queryauc_ranking,queryauc_classic` + `_top2` + shared
  `target/approx/group_id/binary_target`) — `crates/cb-oracle/fixtures/ranking_corpus/ranking_metrics/`;
  harness `ranking_metrics_oracle_test.rs:26-55`.
- **Red:** `calc_metrics_ranking_oracle_test.rs`:
  - reuse `metric_inputs()` (load `target/approx/group_id`);
  - a `gate(metric, "<name>.npy")` that calls
    `calc_metric(&metric, &target, &approx, &[], &group_id)` and
    `compare_stage(Stage::Predictions, &expected, &[got])` ≤1e-5;
  - cover all 9 metrics at defaults + each `@k` metric at `top=2` +
    `QueryAuc{Ranking}` and `QueryAuc{Classic}` (Classic uses `binary_target`);
  - `empty_group_is_single_group` — `calc_metric(&Ndcg{…}, &target, &approx, &[], &[])`
    is `Ok` (single-group NDCG);
  - `non_contiguous_group_id_errs` — a scrambled `group_id` (e.g. `[0,1,0]`) ⇒
    `calc_metric(...).is_err()`.
  Expected INITIAL failure: ranking arm returns the flat error / wrong value ⇒
  `compare_stage` mismatch (or `is_err` fails).
- **Green:** the ranking arm delegation already exists from TASK-03
  (`metric.eval_grouped(approx, label, weight, group_id, &[])`); this task adds NO
  new prod logic unless the oracle reveals a routing defect — its deliverable is
  the ranking oracle LOCK + the two edge-case tests. (If TASK-03's ranking arm is
  found incomplete, complete it here — still a real delegation, never a stub.)
- **Refactor:** unify flat/grouped behind the single `is_ranking` predicate; no
  duplicated group logic (delegated to `eval_grouped`). Regression scope: flat
  oracle (TASK-03) + S1 (TASK-02) + ranking oracle green; `metrics.rs` untouched.
- **Validation:**
  - `cargo test -p cb-train --test calc_metrics_ranking_oracle_test`
  - `cargo test -p cb-train --test calc_metrics_flat_oracle_test`
  - `cargo clippy -p cb-train --lib --no-deps`
- **Completion evidence:** all ~20 ranking oracle cases + 2 edge cases green ≤1e-5;
  no new fixtures added.
- **Compat/rollback:** additive; rollback = revert the ranking arm + remove the
  ranking test.
- **Parallelization:** sequential after TASK-03 (same prod file).

---

## TASK-05 — Dispatch + validation `eval_metric` list form (ORCH-04-S4)

- **Spec refs:** ORCH-04-S4. Primary failure reason: parse-then-evaluate dispatch
  returns wrong arity/order or leaks a panic on misuse.
- **Goal / completion:** `cb_train::calc_metrics::eval_metric(label:&[f64],
  approx:&[f64], metrics:&[&str], weight:&[f64], group_id:&[u64]) -> CbResult<Vec<f64>>`
  parses each descriptor (S1) then evaluates (S2/S3), one value per metric in
  order; all misuse is a typed `Err`. Re-exported from `cb_train`.
- **Prerequisites:** TASK-03, TASK-04 (needs both `calc_metric` arms + `parse_metric`).
- **Files:**
  - Modify: `crates/cb-train/src/calc_metrics.rs` — add `pub fn eval_metric(...)`.
  - Modify: `crates/cb-train/src/calc_metrics_test.rs` — S4 unit tests.
  - Modify: `crates/cb-train/src/lib.rs` — final re-export
    `pub use calc_metrics::{parse_metric, calc_metric, eval_metric};`.
- **CodeGraph/Read evidence:** length-mismatch guard already inside `eval`
  (`metrics.rs:245-249`) and `eval_grouped` (`:406-410`) ⇒ dispatch surfaces the
  typed error without its own panic; re-export site `lib.rs:70`.
- **Red:** in `calc_metrics_test.rs`:
  - `dispatch_two_metrics` — `eval_metric(&label, &approx, &["RMSE","Logloss"], &[], &[])`
    returns `Ok(v)` with `v.len()==2`, `v[0]` = RMSE hand value, `v[1]` = Logloss
    hand value (small vector, deterministic).
  - `dispatch_ranking_empty_group` — a ranking descriptor with empty `group_id`
    returns `Ok` (single-group), no error.
  - `dispatch_length_mismatch_errs` — `label.len() != approx.len()` ⇒ `is_err()`.
  - `dispatch_unknown_metric_errs` — `["bogus"]` ⇒ `is_err()` (from S1).
  Expected INITIAL failure: `eval_metric` unresolved ⇒ build fails.
- **Green:** `metrics.iter().map(|m| calc_metric(&parse_metric(m)?, label, approx,
  weight, group_id)).collect::<CbResult<Vec<_>>>()`. First error short-circuits.
- **Refactor:** none beyond clarity. Regression scope: S1/S2/S3 tests +
  flat/ranking oracle all green.
- **Validation:**
  - `cargo test -p cb-train --lib calc_metrics`
  - `cargo clippy -p cb-train --lib --no-deps`
- **Completion evidence:** 4 S4 unit tests green; re-export compiles; no panic path.
- **Compat/rollback:** additive; rollback = remove `eval_metric` + tests + revert
  re-export to TASK-04 state.
- **Parallelization:** sequential after TASK-04 (same prod file).

---

## TASK-06 — Rust facade `eval_metric` / `eval_metrics` (ORCH-04-S5)

- **Spec refs:** ORCH-04-S5. Primary failure reason: facade value diverges or
  `CbError` is not mapped to `CatBoostError` / a panic crosses the boundary.
- **Goal / completion:** published `catboost_rs::eval_metric(label, approx,
  metric:&str, weight:Option<&[f64]>, group_id:Option<&[u64]>) -> Result<f64,
  CatBoostError>` and `eval_metrics(..., metrics:&[&str], ...) -> Result<Vec<f64>,
  CatBoostError>`; facade test green.
- **Prerequisites:** TASK-05 (`cb_train::calc_metrics::eval_metric` re-exported).
- **Files:**
  - Create: `crates/catboost-rs/src/metrics.rs` — the two free fns; `None` →
    `&[]`. Single-metric extraction MUST be non-panicking (checker MINOR — facade
    clippy DENYs `unwrap`/`indexing_slicing`): `eval_metrics(label, approx,
    &[metric], weight, group_id)?.into_iter().next().ok_or_else(|| /* CatBoostError
    from a CbError::Degenerate("empty metric result") */)` — `.next()` on an owned
    `Vec` iterator is index-safe, and the `ok_or_else` removes the panic path
    entirely (the underlying `eval_metric` returns exactly one value per metric, so
    this arm is unreachable but must still be typed, not `unwrap`ped).
  - Modify: `crates/catboost-rs/src/lib.rs` — `mod metrics;` +
    `pub use metrics::{eval_metric, eval_metrics};`.
  - Create: `crates/catboost-rs/src/metrics_test.rs` (mounted at crate root via
    `#[cfg(test)] mod metrics_test;` — the facade uses root-level test mounts, cf.
    `mod error_test;` `lib.rs:56`).
- **CodeGraph/Read evidence:** mapping precedent `model.rs:169-193`
  (`cb_model::...` → `CatBoostError` via `?`); `CatBoostError::Train(#[from]
  cb_core::CbError)` exists (`error.rs`) ⇒ `?` on a `CbResult` converts directly;
  `cb_train` is a facade dep (re-exports at `lib.rs:44-52`). Note: `cb_train`
  itself must be a dependency of `catboost-rs` — verify `catboost-rs/Cargo.toml`
  lists `cb-train` (the facade already re-exports `cb_train::EBootstrapType` at
  `lib.rs:52`, so the dep is present).
- **Red:** `metrics_test.rs`:
  - `facade_rmse_matches` — build a small `(label, approx)`, assert
    `eval_metric(&label, &approx, "RMSE", None, None).unwrap()` equals the
    hand-computed RMSE (≤1e-5).
  - `facade_list` — `eval_metrics(&label,&approx,&["RMSE","MSLE"],None,None).unwrap().len()==2`.
  - `facade_unknown_metric_errs` — `eval_metric(&label,&approx,"bogus",None,None).is_err()`
    and the error is a `CatBoostError` (matches `CatBoostError::Train(_)`).
  Expected INITIAL failure: `catboost_rs::eval_metric` unresolved ⇒ build fails.
- **Green:** implement both fns delegating to `cb_train::calc_metrics::eval_metric`,
  mapping `None`→`&[]`, propagating `CbError` via `?` (→ `CatBoostError::Train`).
- **Refactor:** `eval_metric` = thin wrapper over `eval_metrics`. Regression scope:
  `cargo test -p catboost-rs` (facade suite) green.
- **Validation:**
  - `cargo test -p catboost-rs`
  - `cargo clippy -p catboost-rs --lib --no-deps`
- **Completion evidence:** 3 facade tests green; `CatBoostError` propagation proven;
  no `unwrap`/`panic` on the prod path.
- **Compat/rollback:** additive free fns; no `Model`/signature change. Rollback =
  remove `metrics.rs`/`metrics_test.rs` + the two `lib.rs` lines.
- **Parallelization:** sequential after TASK-05 (needs the re-export).

---

## TASK-07 — Python `catboost_rs.utils.eval_metric` (ORCH-04-S6)

- **Spec refs:** ORCH-04-S6. Primary failure reason: Python value diverges, wrong
  return shape (scalar vs list), or a bad string aborts instead of raising.
- **Goal / completion:** `catboost_rs.utils.eval_metric(label, approx, metric,
  weight=None, group_id=None)` returns a `float` for a `str` metric and a
  `list[float]` for a `list[str]`; a bad metric string raises a `CatBoostError`
  (mapped), not a panic; `cargo check -p catboost-rs-py` compiles; parity ≤1e-5
  under the uv 3.12 venv.
- **Prerequisites:** TASK-06 (facade fns).
- **Files:**
  - Create: `crates/catboost-rs-py/src/utils.rs` — a `#[pyfunction] eval_metric`
    (accepts `metric: &Bound<PyAny>` to branch `str` vs sequence-of-`str`;
    `label/approx` → `Vec<f64>`, `weight`/`group_id` `Option` → `Vec<f64>`/`Vec<u64>`)
    delegating to `catboost_rs::eval_metric`/`eval_metrics`; errors via
    `.map_err(PyCbError)?` (chokepoint `errors::to_pyerr`). Register it inside a
    `utils` submodule (`PyModule::new(py, "utils")` + `m.add_submodule` + insert
    into `sys.modules["catboost_rs.utils"]` so `import catboost_rs.utils` works).
  - Modify: `crates/catboost-rs-py/src/lib.rs` — `mod utils;` + build/register the
    `utils` submodule in `#[pymodule] fn catboost_rs` (near `:44-55`).
  - Create: a Python parity test (e.g. `crates/catboost-rs-py/tests/test_utils_eval_metric.py`)
    that FIRST asserts the submodule import form works — `import catboost_rs.utils`
    AND `from catboost_rs.utils import eval_metric` both succeed (checker MINOR: the
    `utils` submodule is a new pattern with no in-repo precedent, so the
    `sys.modules["catboost_rs.utils"]` registration must be proven, not assumed) —
    then asserts `catboost_rs.utils.eval_metric` ≈ `catboost.utils.eval_metric`
    ≤1e-5 for RMSE/MSLE scalar + a `["RMSE","MSLE"]` list, and `pytest.raises` on a
    bad metric string.
- **CodeGraph/Read evidence:** `#[pyfunction]` + `wrap_pyfunction!` registration
  (`params.rs:238`, `lib.rs:54`); error chokepoint `errors::to_pyerr` + `PyCbError`
  newtype (`errors.rs:22,92,113`); `gil_used=false` own-before-detach discipline
  (`lib.rs:38-52`) ⇒ copy all Python buffers into Rust-owned `Vec`s before any
  `Python::detach`.
- **Red:** the Python parity test fails because `catboost_rs.utils` /
  `eval_metric` does not exist (ImportError/AttributeError). If the uv venv is
  unavailable in-session, the equivalent red is `cargo check -p catboost-rs-py`
  failing to resolve `catboost_rs::eval_metric` before TASK-06, then a
  compile-verified binding.
- **Green:** implement the pyfunction + submodule; scalar-vs-list branch on the
  `metric` arg type; map errors through `PyCbError`.
- **Refactor:** none beyond deduping the label/approx extraction. Regression scope:
  `cargo check -p catboost-rs-py`; existing Python tests unaffected (additive
  submodule).
- **Validation:**
  - `cargo check -p catboost-rs-py`
  - Under uv 3.12: `uv venv --python 3.12 && uv pip install catboost==1.2.10 'numpy<2' maturin pytest`
    then `maturin develop` + `pytest crates/catboost-rs-py/tests/test_utils_eval_metric.py`
    (matches the FSTR-03 Python precedent).
- **Completion evidence:** `cargo check` clean; Python parity ≤1e-5 scalar + list;
  bad-string `pytest.raises(CatBoostError)`.
- **Compat/rollback:** additive `utils` submodule; no change to existing
  estimators. Rollback = remove `utils.rs` + the `lib.rs` registration + the test.
- **Parallelization:** sequential after TASK-06.

---

## 3. Cross-cutting guardrails (apply to every Rust task)

- **Clippy gate, not build:** `unwrap`/`expect`/`panic`/`indexing_slicing` are
  DENY in prod. Gate each Rust prod change with `cargo clippy -p <crate> --lib
  --no-deps`. Integration tests carry `#![allow(clippy::unwrap_used,
  clippy::expect_used, clippy::panic, clippy::indexing_slicing)]` (as
  `eval_metrics_oracle_test.rs:23` / `ranking_metrics_oracle_test.rs:19` do).
- **Test mount:** the unit test file must be mounted
  (`#[cfg(test)] #[path="calc_metrics_test.rs"] mod tests;`) or `cargo test`
  silently runs 0 tests. Verified against `metrics.rs:616-618` / `ctr_data.rs:65`.
- **D-08 summation:** the parser/dispatch add NO float sums; all reductions stay
  inside `eval`/`eval_grouped`, which already route through `cb_core::sum_f64`.
- **D-04 no-regression:** `metrics.rs` and `ranking_metrics.rs` are read-only.
  `EvalMetric` blast radius (5 cb-train callers: `lib.rs`, `boosting.rs`; tests
  `metrics_test.rs`, `eval_metrics_oracle_test.rs`, `ranking_metrics_oracle_test.rs`)
  is unaffected — confirm with `cargo test -p cb-train` after TASK-05.

## 4. Unresolved blockers / assumptions

1. **`catboost.utils.eval_metric(weight=)` support (SPEC §9 Q1).** Not confirmable
   without the uv venv; resolved inside TASK-01's Green by inspecting the installed
   `catboost/utils.py` signature. Guarded fallback specified (drop weighted fixture,
   cover weighting via a deterministic hand-computed unit test in TASK-03). Does NOT
   block any other task.
2. **uv 3.12 venv availability in-session.** TASK-01 (fixture gen) and TASK-07
   (Python parity run) require it. If unavailable, TASK-01 fixtures are produced
   run-once/commit offline and TASK-07 is compile-verified via `cargo check` with
   the parity `pytest` deferred to the venv (per the FSTR-03 precedent). Not a
   correctness blocker for TASK-02..06.
3. **`cb-train` is a `catboost-rs` dependency (TASK-06).** Strongly implied — the
   facade already re-exports `cb_train::EBootstrapType` (`lib.rs:52`) — but confirm
   `crates/catboost-rs/Cargo.toml` lists `cb-train` before implementing TASK-06.
4. No PageIndex write target confirmed for this corpus (SPEC frontmatter
   `pageindex_pending`); the SPEC under `.planning/phases/.../SPEC.md` is the
   effective spec store. Not a planning blocker.

No requirement conflicts detected. No production code was authored.
