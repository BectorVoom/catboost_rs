## Plan Check Result

**Verdict:** ISSUES_FOUND
**Goal:** Add Min-optimized flat metrics MAE / MAPE / Quantile{alpha} to `cb_train::EvalMetric` so the ORCH-04 standalone `eval_metric`/`calc_metric` surface (and FSTR-02) can evaluate them; acceptance = EM-01..EM-05 + §6.
**Plan:** `.planning/plans/eval-metric-extension/PLAN.md` (against SPEC.md)

### Summary
- The plan's decisive structural claims are CORRECT and independently CodeGraph-verified: exactly TWO E0004-hard exhaustive `match self` sites over `EvalMetric` (`eval` metrics.rs:284, `eval_one_group` metrics.rs:494); `use_group_weight` (matches!, :443), `empty_metric_default` (`_ => 0.0`, :524) and `is_ranking` (matches!, calc_metrics.rs:250) need no edit; no `Display`/`ToString` impl; `for_loss` matches over `Loss`, not `EvalMetric`; no external crate matches `EvalMetric` exhaustively.
- The enum-arm/derive, parse_metric, DEFERRED_METRICS, math, and reduction claims are all correct.
- Two defects block a clean PASS: (1) EMT-5's `flat_metrics_not_ranking` success-probe depends on EMT-2..4 eval math, contradicting the "safe to run in parallel with EMT-2..4" claim (MAJOR); (2) the EM-05 `gate()` helper hard-codes the shared `flat_inputs()` and cannot load a dedicated `label_mape.npy`, so the stated "via the existing gate(...) helper" for the R1 richer-target arbiter is inaccurate (MINOR).

### Specification Coverage
- [x] EM-01 MAE math: EMT-2 (arm) + EMT-6 (oracle). Formula `Σw|a−t|/Σw` mirrors Rmse arm (metrics.rs:285-297); folds via `sum_f64`. OK.
- [x] EM-02 MAPE + zero guard: EMT-3 (arm, provisional D(t)) + EMT-6 (R1 pin). Divisor ambiguity correctly flagged OPEN and gated by oracle (see below). OK with infra caveat.
- [x] EM-03 Quantile math + alpha parse: EMT-4 (pinball) + EMT-5 (parse alpha). `alpha=0.5 ⇒ 0.5·MAE` verified algebraically. OK.
- [x] EM-04 parse + DEFERRED + compile-safety: EMT-1 (E0004 spine) + EMT-5 (parse/DEFERRED/help). OK — but see EMT-5 parallelization defect.
- [x] EM-05 oracle ≤1e-5: EMT-6. OK with gate-helper caveat.

### CodeGraph Evidence
- `EvalMetric` (crates/cb-train/src/metrics.rs:64) — `#[derive(Debug, Clone, PartialEq)]`, non-Copy (`Custom(CustomMetricHandle)` holds `Arc`). f64 fields already precedent (`Map.border`, `PFound.decay`), so `Quantile { alpha: f64 }` is derive-consistent. Blast radius: 10 callers in cb-train (boosting.rs/lib.rs/calc_metrics.rs) only; external refs (catboost-rs/src/builder.rs, cb-compute/src/custom.rs) use it as `Option<EvalMetric>` / construct `Custom` / comments — NO exhaustive external match. SPEC §7 confirmed.
- `EvalMetric::eval` (metrics.rs:284) — exhaustive `match self` with explicit ranking-reject arm, NO wildcard ⇒ adding variants is E0004-hard. CONFIRMS site #1.
- `EvalMetric::eval_one_group` (metrics.rs:494) — `Ok(match self {...})` exhaustive; flat-reject arm at :513 `Self::Rmse | Self::Logloss | Self::Msle | Self::Custom(_)`, NO wildcard ⇒ E0004-hard. CONFIRMS site #2. Only reached from `eval_grouped` (metrics.rs:471) which only runs when `is_ranking` true, so new flat arms never reach it at runtime — routing to the reject arm is correct.
- `use_group_weight` (metrics.rs:443) — `matches!(self, Ndcg|Dcg|PFound|Err|Mrr|QueryAuc)`; non-exhaustive ⇒ new arms default `false`; no compile risk, no edit. CONFIRMED.
- `empty_metric_default` (metrics.rs:524) — `match self { PrecisionAt|RecallAt => 1.0, _ => 0.0 }` wildcard ⇒ safe, no edit. CONFIRMED.
- `is_ranking` (calc_metrics.rs:250) — `matches!` non-exhaustive ⇒ new arms `false` = flat path via `calc_metric` (calc_metrics.rs:288-292 routes to `eval`). CONFIRMED.
- `for_loss` (metrics.rs:163) — `match *loss` over `Loss` (NOT EvalMetric); produces EvalMetric values but needs no new arm. CONFIRMED not a hazard.
- `parse_metric` (calc_metrics.rs:151-241) — `match lower.as_str()` with `other =>` catch-all (:233). `reject_unknown(&self, metric, allowed)` (:54) and `f64_or(&self, key, default, metric)` (:80) signatures match the plan's proposed `p.reject_unknown(name, &["alpha"])?` / `p.f64_or("alpha", 0.5, name)?` exactly (cf. `map` arm :187-191). CONFIRMED.
- `DEFERRED_METRICS` (calc_metrics.rs:22-24) — text lists "…, MAE, MAPE" (Quantile NOT listed). Plan removes MAE/MAPE. Unknown-name help (:235-236) is a SEPARATE "supported:" list; add MAE/MAPE/Quantile there. CONFIRMED.
- Test infra: mounts `metrics.rs:616-618 → metrics_test.rs`, `calc_metrics.rs:319-321 → calc_metrics_test.rs` CONFIRMED. Oracle: `tests/calc_metrics_flat_oracle_test.rs` `gate(metric: EvalMetric, fixture_name, weight)` (:36) + `flat_inputs()` (:28); `tests/eval_metrics_oracle_test.rs` exists (regression). Generator `gen_ranking_fixtures.py:368 gen_calc_metrics_flat()`, `freeze(name, metric, **kw)` (:396), `--calc-metrics` flag (:430). CONFIRMED. No test enumerates all EvalMetric variants ⇒ EMT-1 inert arms keep `--lib` green.

### Issues

#### [MAJOR] EMT-5's `flat_metrics_not_ranking` success-probe depends on EMT-2..4, contradicting the stated parallelism
- **Plan location:** EMT-5 "Files & symbols" (`flat_metrics_not_ranking`) + §1 wave graph ("‖ EMT-5 (parallel)") + EMT-5 "Prerequisites: EMT-1 … Independent of EMT-2..4 … safe to run in parallel".
- **Requirement:** EM-04 (confirm `is_ranking` false / flat routing).
- **Evidence:** EMT-1 installs INERT eval arms returning `Err(CbError::Degenerate("… not yet implemented"))` for Mae/Mape/Quantile (plan EMT-1). `calc_metric` (calc_metrics.rs:288-292) routes a flat metric to `EvalMetric::eval`. The proposed probe asserts "`calc_metric` on the three SUCCEEDS on flat inputs … i.e. routes to `eval`". Until EMT-2/3/4 replace the inert arms, `eval(Mae|Mape|Quantile)` returns `Err`, so `calc_metric` returns `Err` — the success assertion fails. EMT-5's own Verify step runs `cargo test -p cb-train --lib`, which includes this test.
- **Failure scenario:** In the authorized Wave-2 parallel schedule, EMT-5 is authored/verified before (or interleaved with) EMT-2..4 ⇒ `flat_metrics_not_ranking` fails as a spurious Red that cannot go Green from EMT-5's own changes. This is an invalid intermediate state for the stated ordering.
- **Impact:** Broken TDD Red→Green at EMT-5; false regression signal; the "independent, parallel" claim is wrong for this test.
- **Required revision:** Either (a) move the calc_metric-success routing probe out of EMT-5 into a task that depends on EMT-2/3/4 (e.g. fold it into EMT-6 or a post-math step), keeping only the parse-only tests (`parse_mae_mape_quantile`, `parse_rejects_bad_quantile_param`) in EMT-5 (those ARE independent of eval math and parallel-safe); or (b) redefine the probe so it does not require eval success — assert flat routing via the DISTINCT error signal (during the inert phase the flat path yields the inert "…not yet implemented" message, whereas the grouped path via `eval_one_group` yields "non-ranking metric passed to the grouped seam"), or expose a thin test-only routing check. Update EMT-5's prerequisite/parallelism note accordingly.

#### [MINOR] EM-05 `gate()` helper cannot load a dedicated MAPE target; "via the existing gate(...) helper" is inaccurate for the R1 arbiter
- **Plan location:** EMT-6 "Files & symbols" (dedicated `label_mape.npy` with 0 / `0<|t|<1` / `|t|>1` rows) + "each via the existing `gate(...)` helper".
- **Requirement:** EM-02 / EM-05 / R1 pinning.
- **Evidence:** `gate(metric, fixture_name, weight)` (calc_metrics_flat_oracle_test.rs:36-44) hard-codes `let (label, approx) = flat_inputs();` and only varies `weight`. It cannot consume a different `label`/`approx` pair. The shared `label` is `rng.integers(0,2)` = {0,1} (generator :377), which DOES contain zeros (so `max(1,·)` vs `max(ε,·)` vs skip-zero already differ at the t=0 rows) but has no `0<|t|<1` row.
- **Failure scenario:** Following EMT-6 literally — freeze `label_mape.npy` AND reuse `gate()` — does not compile/work: `gate()` ignores the new label file, so MAPE is evaluated against the shared {0,1} label instead of the richer target, silently defeating the intended R1 discriminator.
- **Impact:** Low — the shared {0,1} label still distinguishes the three divisor conventions at its zero/one rows, so R1 can be pinned without a new fixture; but the plan as written is internally inconsistent and would confuse the implementer.
- **Required revision:** Either (a) drop the dedicated `label_mape.npy` and pin R1 using the shared {0,1} label (a `freeze("mape","MAPE")` scalar over it is already a valid arbiter — state this), or (b) add a parameterized helper `gate_with_inputs(label_file, approx_file, metric, fixture, weight)` (and a matching `approx_mape.npy`) and route the MAPE zero-target gate through it. Specify which.

### Implementation Order Review
1. EMT-1 first — enum arms + close both E0004 sites (eval :284, eval_one_group reject arm :513) with inert arms. Correct; blocks all. `--lib` stays green (no variant enumeration in tests — verified).
2. EMT-2 → EMT-3 → EMT-4 serialized (all edit the metrics.rs:284 `match self` block — real write-conflict). Correct.
3. EMT-5 (calc_metrics.rs) — parse/DEFERRED/help arms are genuinely independent of eval math and parallel-safe; ONLY the `flat_metrics_not_ranking` success-probe is not (see MAJOR issue). Split that probe out.
4. EMT-6 last — needs all math arms + parse; also the natural home for the routing probe from the MAJOR fix. Correct.
Graph is otherwise acyclic and prerequisites are satisfied.

### Potential Bugs
- MAPE R1 divisor (`max(|t|,ε)` vs `max(1.0,|t|)` vs skip-zero): correctly flagged OPEN and gated by EMT-6's frozen `catboost==1.2.10` scalar; EMT-3's `mape_zero_target_finite` asserts finiteness only, so it holds under any convention. Not silently assumed. Adequately handled.
- Quantile params (R2): `reject_unknown(name, &["alpha"])` rejects any non-alpha param (e.g. upstream `delta`). If `catboost.utils.eval_metric` needs another Quantile param for parity, EMT-6's Quantile(0.9) gate catches it. Low risk, flagged. Acceptable.
- `catboost.utils.eval_metric(weight=…)` support for MAE/MAPE/Quantile is assumed (generator comment says confirmed at gen time for RMSE). EMT-6 must re-confirm at freeze time for the weighted MAE/Quantile scenarios; if unsupported, drop the weighted scenarios rather than fabricate. Minor — verify at fixture time.

### Required Plan Revisions
1. Fix EMT-5: relocate the `calc_metric`-success routing probe (`flat_metrics_not_ranking`) to depend on EMT-2/3/4 (or redefine it to not require eval success); keep only parse-only tests parallel with EMT-2..4; correct the "independent / parallel" note.
2. Fix EMT-6 MAPE fixture: either pin R1 on the existing shared {0,1} label (state it is a valid arbiter) or add a `gate_with_inputs(...)` helper + `label_mape.npy`/`approx_mape.npy`; do not claim the current `gate()` helper supports a dedicated target.
3. EMT-6: re-confirm `catboost.utils.eval_metric` accepts `weight=` for MAE/Quantile at freeze time before committing weighted scenarios.

### Unverified Items
- The exact upstream `TMAPEMetric` divisor convention and Quantile weighting/param set (R1/R2) — `[UNVERIFIED — sparse checkout]`; correctly deferred to the EMT-6 oracle (the plan does not silently choose). Not a plan defect; noted for the operator.

---

## Plan Check Result — PASS 2 (re-review of revised PLAN)

**Verdict:** PASS
**Goal:** Add Min-optimized flat metrics MAE / MAPE / Quantile{alpha} to `cb_train::EvalMetric` so the ORCH-04 `eval_metric`/`calc_metric` surface (and FSTR-02) can evaluate them; acceptance = EM-01..EM-05 + §6.
**Plan:** `.planning/plans/eval-metric-extension/PLAN.md` (revised)

### Summary
- Both blocking defects from PASS 1 are resolved by the revision, and no new defect was introduced. All structural claims re-verified via CodeGraph MCP against the current working tree. Verdict flips ISSUES_FOUND → PASS.

### MAJOR (PASS 1) — RESOLVED
- EMT-5 is now **PARSE-ONLY**: it carries only `parse_mae_mape_quantile` and `parse_rejects_bad_quantile_param` plus the `DEFERRED_METRICS`/help edits (PLAN EMT-5 lines 279–329). These tests exercise `parse_metric` only (no `calc_metric`), so they go Red→Green from EMT-5's own changes (parse arms + the EMT-1 enum arms) and are genuinely parallel-safe with EMT-2..4 (which edit `metrics.rs`, a different file).
- The `flat_metrics_not_ranking` **success probe is removed from EMT-5 and moved to EMT-6** (split note lines 287–295; EMT-6 lines 378–383: the `gate(...)` calls to `calc_metric(&metric, …, &[])` "also constitute the flat-routing SUCCESS assertion deferred from EMT-5").
- The fix is **real**, CodeGraph-confirmed: `calc_metric` (calc_metrics.rs:288–292) routes a non-ranking metric to `metric.eval`; `is_ranking` (calc_metrics.rs:250, non-exhaustive `matches!`) returns `false` for the three new arms; EMT-1's inert `eval` arms (metrics.rs:284 exhaustive `match self`, no wildcard — E0004) return `Err` until EMT-2/3/4 land. Therefore any `calc_metric` **success** probe placed in EMT-5 would fail during the authorized Wave-2 parallel schedule — the earlier parallel placement was genuinely invalid, and relocation to post-math EMT-6 is correct.

### MINOR (PASS 1) — RESOLVED
- EMT-6 now REUSES the shared `flat_inputs()` `(label, approx)` and the existing `gate(metric, fixture, weight)` helper **AS-IS** (PLAN lines 341–356): no `gate_with_inputs`, no dedicated `label_mape.npy` on the default path. Verified against `calc_metrics_flat_oracle_test.rs:36–44` — `gate` takes an `EvalMetric` value directly and loads the shared `{0,1}` label from `flat_inputs()` (:28–32); it never varied `label`, so the PASS-1 inconsistency is gone.
- The `{0,1}` zero-row arbiter reasoning is **sound**: the shared label is `rng.integers(0,2)` (generator :377), so it contains `t=0` rows. At `t=0`, `max(ε,|t|)` (→ε, huge term), `max(1,|t|)` (→1), and skip-zero (row dropped) each yield a **distinct** finite frozen scalar; plain-`|t|` div-by-zero is non-finite and excluded by `freeze`'s `np.isfinite` guard (:399–400). At `t=1` all conventions coincide (D=1), so only the `t=0` rows discriminate — which is exactly why `{0,1}` pins R1. The one gap `{0,1}` cannot separate (`max(1,|t|)` vs plain-`|t|` for `0<|t|<1`) is correctly captured as the explicit contingency fallback (lines 352–356), not the default.
- Quantile scenarios pass `EvalMetric::Quantile { alpha: 0.9 }` directly to `gate()` (lines 350–351, 380) — no metric-string parsing in the oracle test; consistent with `gate`'s `EvalMetric` parameter.

### minor note (PASS 1) — RESOLVED
- EMT-6 now requires re-confirming `catboost.utils.eval_metric(weight=…)` support for MAE/Quantile **at freeze time**, with an explicit try/skip per metric and a `summary.json` note (PLAN lines 364–371). Verified this is a real gap in the current generator: `freeze` (gen_ranking_fixtures.py:396–406) raises only on a non-finite value; it has no guard for a rejected `weight=` kwarg. The plan's added try/skip is the correct instruction, and the weighted variant is omitted for any metric where `weight=` is rejected rather than emitting a bogus fixture.

### No-regression re-verification (CodeGraph, current tree)
- **E0004 site #1** — `EvalMetric::eval` (metrics.rs:284) exhaustive `match self`, no wildcard; flat arms Rmse/Logloss/Msle/Custom + ranking-reject arm (:359–369). Adding variants is compile-hard; the inert-arm placement is correct. CONFIRMED.
- **E0004 site #2** — `eval_one_group` (metrics.rs:494) `Ok(match self {...})`, flat-reject arm at :513 `Self::Rmse | Self::Logloss | Self::Msle | Self::Custom(_)`, no wildcard. Only reached from `eval_grouped` (:471) under `is_ranking==true`, so routing the new flat arms to the reject arm is correct. CONFIRMED.
- **No-edit matches** — `use_group_weight` (metrics.rs:443, `matches!` → default `false`), `empty_metric_default` (metrics.rs:524, `_ => 0.0` wildcard), `is_ranking` (calc_metrics.rs:250, `matches!` → default `false`). All safe, no edit. CONFIRMED.
- **parse pattern** — `parse_metric` (calc_metrics.rs:157) `match lower.as_str()` with `other =>` catch-all (:233); `map` arm (:186–191) uses `p.reject_unknown(name, &["top","border"])?` + `p.f64_or("border", 0.5, name)?` — byte-for-byte the pattern the plan's `quantile` arm mirrors (`&["alpha"]` + `f64_or("alpha", 0.5, name)`). CONFIRMED.
- **DEFERRED_METRICS** — text at calc_metrics.rs:23 lists "…, MAE, MAPE" (Quantile NOT listed); plan removes MAE/MAPE. Unknown-name help (:235–236) is a separate "supported:" list to extend. CONFIRMED.
- **final-error math** — MAE `Σw|a−t|/Σw` mirrors the Rmse arm shape (metrics.rs:285–297) and folds via `sum_f64` on the shared `total_weight` (:274–282); Quantile pinball reduces the same way; MAPE divisor provisional and oracle-pinned. Sound.
- **Task graph** — EMT-1 → (EMT-2→EMT-3→EMT-4 serialized on the metrics.rs:284 block ∥ EMT-5 on calc_metrics.rs) → EMT-6. Acyclic; prerequisites satisfied.
- **Validation commands real** — `calc_metrics_flat_oracle_test.rs`, `eval_metrics_oracle_test.rs`, mounted `metrics_test.rs` / `calc_metrics_test.rs`, and `gen_ranking_fixtures.py --calc-metrics` all exist on disk. CONFIRMED.

### Residual (non-blocking) observations
- EMT-5's Verify step runs the whole `cargo test -p cb-train --lib` suite; under true parallel execution it could transiently observe an in-flight EMT-2..4 Red (those tasks add their own tests to `metrics_test.rs`). This is an orchestration property of parallel TDD on a shared crate, not a defect in EMT-5's own Red→Green, and the plan already serializes the `metrics.rs` edits. No action required.

### Unverified Items (unchanged from PASS 1 — not plan defects)
- Exact upstream `TMAPEMetric` divisor convention and Quantile weighting/param set (R1/R2) — `[UNVERIFIED — sparse checkout]`; correctly deferred to the EMT-6 frozen `catboost==1.2.10` oracle, which is the designed arbiter. The plan does not silently choose.
