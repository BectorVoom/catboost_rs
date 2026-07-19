## Plan Check Result — Pass 2 (Re-verification)

**Verdict:** PASS
**Goal:** ORCH-04 — standalone `eval_metric`/`calc_metric` surface computing a CatBoost metric's FINAL value on caller-supplied `(label, approx, weight, group_id)`, oracle-locked ≤1e-5 vs `catboost.utils.eval_metric`; `calc_metrics` module in `cb-train`, re-exported through the `catboost-rs` facade and `catboost-rs-py`.
**Plan:** `.planning/phases/20-orchestration/calc-metrics/PLAN.md` (v1, revised)

### Summary
- All five findings from Pass 1 are resolved in the revised SPEC/PLAN. The one MAJOR issue (flat-fixture label domain) is fixed at the executable level (TASK-01 Green) and recorded in SPEC §9. The four MINORs are each addressed with a concrete, verified revision.
- No new issue was introduced by the edits. Every structural claim underpinning the plan was CodeGraph-verified in Pass 1 and remains valid — no production source changed, so the reused seams, error wiring, dependency edges, re-export sites, oracle harness, and `EvalMetric` blast radius are unaffected.
- The single remaining unverified item (`catboost.utils.eval_metric(weight=)` support) is a documented, guarded non-blocker: it is resolved inside TASK-01's Green under the venv, with a deterministic hand-computed unit-test fallback if the kwarg is absent. Accepted as non-blocking.

### Resolution status of prior issues

#### [MAJOR] Shared flat-fixture label domain — RESOLVED
- **Fix (verified):** PLAN TASK-01 Green now states "**`label` MUST be pinned to `[0,1]`** ... a `[0,1]` label simultaneously satisfies RMSE, Logloss, AND MSLE" and "keep `approx > -1` so the MSLE `1+approx>0` log-domain guard (`metrics.rs:326-331`) is satisfied" (PLAN.md:143-150).
- **Evidence:** The `[0,1]` constraint satisfies upstream Logloss's `target ∈ [0,1]` requirement; RMSE accepts any label; MSLE's `1+label>0` holds trivially for `label ≥ 0`. The `approx > -1` constraint matches the Rust MSLE guard I re-confirmed at `metrics.rs:326-331` (returns `CbError::Degenerate` when `!(1+approx>0)`). SPEC §9 risk table adds a dedicated row recording the same constraint with `[VERIFIED: CODEGRAPH metrics.rs; checker PLAN-CHECK MAJOR]`.
- **Status:** Fully resolved. AT-S2 (RMSE/Logloss/MSLE) can now be generated and oracle-locked from a single shared `(label, approx)` pair.

#### [MINOR] TASK-03 panic/`todo!` ranking stub vs clippy gate — RESOLVED
- **Fix (verified):** PLAN TASK-03 Files + Green now implement BOTH arms as real seam delegations — flat `metric.eval(approx, label, weight)`, ranking `metric.eval_grouped(approx, label, weight, group_id, &[])` — with the explicit note "no `todo!`/`unimplemented!`/panic stub may exist, since TASK-03's own clippy gate DENYs panic" (PLAN.md:245-251, 269-273). TASK-04 Green updated to "the ranking arm delegation already exists from TASK-03 ... this task adds NO new prod logic ... its deliverable is the ranking oracle LOCK + the two edge-case tests" (PLAN.md:325-329).
- **Evidence:** Both delegations target the verified seam signatures (`eval` at `metrics.rs:244`; `eval_grouped` at `metrics.rs:397`). No intermediate build state contains a panic path, so `cargo clippy -p cb-train --lib --no-deps` at the TASK-03 gate stays green.
- **Status:** Fully resolved.

#### [MINOR] Facade single-metric extraction must be non-panicking — RESOLVED
- **Fix (verified):** PLAN TASK-06 Files now specifies `eval_metrics(label, approx, &[metric], weight, group_id)?.into_iter().next().ok_or_else(|| ... CatBoostError from a CbError::Degenerate("empty metric result"))` and notes "`.next()` on an owned `Vec` iterator is index-safe, and the `ok_or_else` removes the panic path entirely" (PLAN.md:396-402).
- **Evidence:** No `unwrap`/`expect`/indexing on the prod path; the fallback maps to `CatBoostError::Train` via the confirmed `#[from] cb_core::CbError` (`error.rs:37`). Satisfies the facade clippy gate.
- **Status:** Fully resolved.

#### [MINOR] Python `utils` submodule import (new pattern) — RESOLVED
- **Fix (verified):** PLAN TASK-07 test now "FIRST asserts the submodule import form works — `import catboost_rs.utils` AND `from catboost_rs.utils import eval_metric` both succeed ... the `sys.modules["catboost_rs.utils"]` registration must be proven, not assumed — then asserts ... ≤1e-5 ... and `pytest.raises` on a bad metric string" (PLAN.md:458-465).
- **Evidence:** The registration recipe (`PyModule::new` + `add_submodule` + `sys.modules` insertion) is standard PyO3; the `#[pyfunction]`/`wrap_pyfunction!` and `PyCbError`/`to_pyerr` chokepoint it relies on were confirmed in Pass 1 (`lib.rs:54`, `params.rs:238`, `errors.rs:88,113`). The added import assertions turn the previously-assumed behavior into a tested invariant.
- **Status:** Fully resolved.

#### [MINOR] Weighted-RMSE `weight=` upstream support — ACCEPTED NON-BLOCKER
- **Status (verified):** Unchanged by design (PLAN.md:152-157, SPEC §9 Open Question 1). Resolved inside TASK-01's Green by inspecting the installed `catboost/utils.py` under the uv 3.12 venv; if `weight=` is unsupported, the weighted fixture is omitted and weighting is covered by a deterministic hand-computed unit assertion (the weight arithmetic in `EvalMetric::eval` is deterministic, `metrics.rs:266-296`).
- **Assessment:** Genuinely unverifiable in-session (no venv). The guarded fallback preserves a regression guard on the weighting arithmetic even in the worst case. Accepted as a non-blocker per the coordinator's request; the only residual is that the weighted clause of AT-S2 may be self-consistency-tested rather than oracle-locked — a documented, bounded degradation, not a correctness risk.

### CodeGraph Evidence (unchanged, still valid — no source edited)
- `EvalMetric::eval` `(approx, target, weights)` at `crates/cb-train/src/metrics.rs:244`; MSLE log-domain guard at `:326-331`. `calc_metric` maps `label→target` — correct.
- `EvalMetric::eval_grouped` `(approx, target, weights, group_id, subgroup_id)` at `metrics.rs:397`; empty-group→single (`group_spans:536-543`), non-contiguous→`Degenerate` (`:558-562`).
- `CatBoostError::Train(#[from] cb_core::CbError)` at `crates/catboost-rs/src/error.rs:37`; `cb-train` dep at `crates/catboost-rs/Cargo.toml:33`; no `metrics` module / `eval_metric` symbol collision in the facade.
- `#[pyfunction]`/`wrap_pyfunction!` (`catboost-rs-py/src/params.rs:238`, `lib.rs:54`); `PyCbError`/`to_pyerr` (`errors.rs:88,113`); `gil_used=false` (`lib.rs:42`).
- Re-export site `crates/cb-train/src/lib.rs:70`; `mod ranking_metrics;` (:30) → `crate::ranking_metrics::{DcgMetricType, DcgDenominator, AucType}` resolves.
- Oracle harness `compare_stage(Stage::Predictions, &expected, &[got])` with `expected.len()==1` — verified against `ranking_metrics_oracle_test.rs:47-55`.
- `EvalMetric` blast radius (5 cb-train callers + 3 test files) unaffected — `metrics.rs`/`ranking_metrics.rs` are read-only under this plan.

### Implementation Order Review
- Wave A `TASK-01 ∥ TASK-02` (disjoint artifacts) → `TASK-03 → TASK-04 → TASK-05 → TASK-06 → TASK-07`. Acyclic; every prerequisite is produced before it is consumed. The former panic-stub hazard between TASK-03 and TASK-04 is eliminated (both arms real from TASK-03). No intermediate state fails to build or violates the clippy gate.

### Verification Coverage
- S1: unit (parser variants/defaults/rejection). S2: new `calc_metrics/` oracle ≤1e-5 (RMSE/Logloss/MSLE) + weighted (oracle or guarded unit). S3: existing `ranking_corpus/ranking_metrics/*.npy` oracle ≤1e-5 over ~20 cases + 2 group edge-case tests. S4: dispatch arity/order + typed-error unit tests. S5: facade value + `CatBoostError::Train` propagation. S6: Python import-form + ≤1e-5 parity + `pytest.raises`. Every S1..S6 has an objective method; clippy gate + `cargo test` targets specified per task.

### Unverified Items
- `catboost.utils.eval_metric(weight=...)` support in catboost 1.2.10 — not verifiable without the uv 3.12 venv. Bounded, guarded, accepted non-blocker (see above).
- Numeric values of the not-yet-generated `calc_metrics/` flat fixtures — producible only when TASK-01 runs under the venv. Expected as part of execution, not a planning gap.
