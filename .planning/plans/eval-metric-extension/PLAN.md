---
title: EvalMetric extension — MAE / MAPE / Quantile — TDD Implementation Plan
plan_for: .planning/plans/eval-metric-extension/SPEC.md
status: complete
format: markdown
plan_version: 1
updated_at: 2026-07-19T00:00:00Z
enables:
  - "FSTR-02 multi-loss LossFunctionChange (needs MAE/Quantile/MAPE final-error via EvalMetric)"
spec_ids: [EM-01, EM-02, EM-03, EM-04, EM-05]
task_ids: [EMT-1, EMT-2, EMT-3, EMT-4, EMT-5, EMT-6]
gsd_used: false
---

# EvalMetric extension — MAE / MAPE / Quantile — TDD Plan

Goal-backward plan for adding three Min-optimized **flat** metrics to
`cb_train::EvalMetric` — `Mae`, `Mape`, `Quantile { alpha }` — so ORCH-04's
`calc_metric`/`eval_metric` surface (and downstream FSTR-02) can evaluate them.
Metric-only: **no training-loss wiring, no ranking/grouped variants** (SPEC §2).

This is a PREREQUISITE that unblocks FSTR-02's full numeric loss set. Plan only —
**no production code is written here**; each task is an executable Red→Green→Refactor
prompt. No GSD skill/command/workflow/agent was used to produce this plan.

---

## 0. Verified anchors (CodeGraph + Read, this session)

All line numbers verified against the current working tree on
`feat/18-fstr03-partial-dependence`.

| Symbol | Location | Fact |
|---|---|---|
| `EvalMetric` enum | `crates/cb-train/src/metrics.rs:63-151` | `#[derive(Debug, Clone, PartialEq)]`, non-`Copy` (Custom holds an `Arc`). Flat arms `Rmse`/`Logloss`/`Msle`/`Custom`; rest ranking. |
| `EvalMetric::eval` (flat) | `crates/cb-train/src/metrics.rs:284` (`match self`) | **EXHAUSTIVE — E0004 risk.** Guards at :246-282 (length-mismatch/empty/non-positive-weight → `CbError::Degenerate`); `weight_at` fallback 1.0; `total_weight = sum_f64(...)`. |
| `EvalMetric::eval_one_group` | `crates/cb-train/src/metrics.rs:494` (`Ok(match self {...})`) | **EXHAUSTIVE — E0004 risk.** Flat arms rejected at :513 (`Self::Rmse \| Self::Logloss \| Self::Msle \| Self::Custom(_)`). |
| `EvalMetric::use_group_weight` | `crates/cb-train/src/metrics.rs:443` (`matches!(self, ...)`) | Non-exhaustive `matches!` → new arms default to `false`. No compile risk, no edit. |
| `EvalMetric::empty_metric_default` | `crates/cb-train/src/metrics.rs:524` (`match self`) | Has `_ => 0.0` wildcard → new arms safe; only reached by ranking path. No edit. |
| `is_ranking` | `crates/cb-train/src/calc_metrics.rs:250` (`matches!`) | Non-exhaustive → new arms default `false` = flat path. No edit; assert in test. |
| `parse_metric` | `crates/cb-train/src/calc_metrics.rs:151-241` (`match lower.as_str()`) | String match with `other =>` catch-all at :233. `p.reject_unknown`, `p.f64_or("alpha",…)` helpers available. |
| `DEFERRED_METRICS` | `crates/cb-train/src/calc_metrics.rs:22-24` | Lists "…, MAE, MAPE" — remove those two. (Quantile is NOT currently listed.) |
| unknown-name help | `crates/cb-train/src/calc_metrics.rs:235-236` | Supported list — add MAE, MAPE, Quantile. |
| `calc_metric` dispatch | `crates/cb-train/src/calc_metrics.rs:281-293` | `is_ranking(m) ? eval_grouped : eval`. New arms auto-route to `eval`. |
| lib exports | `crates/cb-train/src/lib.rs:71-72` | `pub use` of `calc_metric, eval_metric, parse_metric, EvalMetric` — no change. |
| `QUANTILE_ALPHA` | `crates/cb-compute/src/loss.rs:150` = `0.5` | Training-loss default; a `cb-compute` const. Parser default uses a **literal `0.5`** (avoid a parser→cb-compute coupling); reference `QUANTILE_ALPHA` only in a doc comment. |
| test mounts | `metrics.rs:617` → `metrics_test.rs`; `calc_metrics.rs:320` → `calc_metrics_test.rs` | Sibling `#[cfg(test)] #[path=…] mod tests;`. Omitting the mount silently runs 0 tests. |
| oracle test | `crates/cb-train/tests/calc_metrics_flat_oracle_test.rs` | Deterministic flat oracle over frozen `(label, approx)`; `gate(metric, "x.npy", &weight)` helper; `compare_stage(Stage::Predictions, …)`. **This is the EM-05 home** (no training). |
| fixture generator | `crates/cb-oracle/generator/gen_ranking_fixtures.py:368` `gen_calc_metrics_flat()` (`--calc-metrics`) | Freezes `label`/`approx`/`weight` `.npy` + per-metric scalar `.npy` from `catboost.utils.eval_metric` (RUN-ONCE/COMMIT). |

### R3 (exhaustive-match) — RESOLVED enumeration

Adding the three enum variants forces edits at exactly **two** compile-hard
(`E0004`) sites, plus zero to-verify sites:

1. **`metrics.rs:284` `EvalMetric::eval` `match self`** — add real math arms
   (`Mae`, `Mape`, `Quantile{alpha}`). **HARD.**
2. **`metrics.rs:494` `EvalMetric::eval_one_group` `Ok(match self {...})`** —
   extend the flat-reject arm at :513 to `Self::Rmse | Self::Logloss | Self::Msle
   | Self::Mae | Self::Mape | Self::Quantile { .. } | Self::Custom(_)`. **HARD.**
3. `metrics.rs:443` `use_group_weight` (`matches!`) — bool, defaults `false`. **OK, no edit.**
4. `metrics.rs:524` `empty_metric_default` (`_ => 0.0` wildcard). **OK, no edit.**
5. `calc_metrics.rs:250` `is_ranking` (`matches!`) — bool, defaults `false`. **OK, no edit** (assert in EMT-5).
6. No `Display`/`ToString`/other `impl … for EvalMetric` exists (`Debug` is derived and auto-handles new arms). **OK.**

`EMT-1` makes E0004 the driving failure (add a variant with an unhandled arm ⇒
crate won't compile) and closes both hard sites before any metric math lands.

### R1 (MAPE zero-target divisor) — OPEN, gated by EMT-6

The SPEC's §4 formula writes `|a−t| / max(|t|, ε)`. Upstream CatBoost
`TMAPEMetric` plausibly divides by `max(1.0, |t|)` — a **materially different**
convention for `0 < |t| < 1` rows. This is `[UNVERIFIED — sparse checkout]` and
**must not be silently chosen**. EMT-6 freezes a MAPE target vector containing
(a) a zero row, (b) a `0 < |t| < 1` row, and (c) a `|t| > 1` row so that
`max(ε,|t|)`, `max(1,|t|)`, and plain `|t|`+skip-zero are all mutually
distinguishable; the frozen `catboost==1.2.10` scalar is the arbiter. EMT-3's
`Mape` arm is implemented against a stated hypothesis and **finalized in EMT-6**
once the fixture pins the convention.

---

## 1. Task graph & execution waves

```text
Wave 1:  EMT-1  (compile-safety spine: enum arms + both E0004 matches)
Wave 2:  EMT-2 -> EMT-3 -> EMT-4   (serialized: all edit metrics.rs `eval` match — write conflict)
         ‖ EMT-5                    (parallel: edits calc_metrics.rs — different file)
Wave 3:  EMT-6  (oracle fixtures + gate; depends on EMT-2,3,4,5)
```

- **EMT-2/EMT-3/EMT-4 share the `metrics.rs:284` `match self` block** (different
  arms of one match) ⇒ **write-conflict**; run them sequentially, not in parallel.
- **EMT-5 is PARSE-ONLY, edits `calc_metrics.rs` only** and needs only the enum
  arms from EMT-1 (not the eval math) ⇒ safe to run in parallel with EMT-2..4. Its
  tests never call `calc_metric` (which would hit EMT-1's inert `Err` arms) — the
  flat-routing SUCCESS assertion is held by EMT-6 (see EMT-5 split note).
- **EMT-6 needs every math arm + the parse arm** ⇒ last; it also carries the
  flat-routing success assertion (`calc_metric` on Mae/Mape/Quantile succeeds,
  routing to `eval`) that could not live in EMT-5.

Spec→task coverage:

| Spec | Tasks |
|---|---|
| EM-01 (MAE math) | EMT-1 (arm slot), **EMT-2**, EMT-6 |
| EM-02 (MAPE math + zero guard) | EMT-1, **EMT-3**, **EMT-6** (pins R1) |
| EM-03 (Quantile math + alpha) | EMT-1, **EMT-4** (math), **EMT-5** (parse), EMT-6 |
| EM-04 (parse + DEFERRED + compile-safety) | **EMT-1** (compile-safety), **EMT-5** (parse/DEFERRED), EMT-6 (flat-routing success probe) |
| EM-05 (oracle ≤1e-5) | **EMT-6** |

Every task maps to ≥1 spec; every spec maps to ≥1 task. Graph is acyclic.

---

## 2. Global validation commands

Run the narrowest first; the last two are the regression gate.

```bash
cargo test  -p cb-train --lib                                  # unit (metrics_test.rs + calc_metrics_test.rs)
cargo test  -p cb-train --test calc_metrics_flat_oracle_test   # EM-05 flat oracle (deterministic, no training)
cargo test  -p cb-train --test eval_metrics_oracle_test        # regression: train-time eval curves unaffected
cargo clippy -p cb-train --lib --no-deps                       # deny-lints: no unwrap/expect/panic/indexing_slicing
```

Oracle regen (EMT-6 only, RUN-ONCE / COMMIT the `.npy`; CI only reads them):

```bash
uv venv --python 3.12 && uv pip install catboost==1.2.10 'numpy<2'
.venv/bin/python crates/cb-oracle/generator/gen_ranking_fixtures.py --calc-metrics
```

Guardrails honoured by every task: no `unwrap`/`expect`/`panic`/indexing in
production (clippy-gated, not build-gated); every fold via `cb_core::sum_f64`
(D-08); tests only in mounted sibling `*_test.rs`; enum additions are additive.

---

## EMT-1 — Compile-safety spine (enum arms + both E0004 matches)  [EM-04 / R3]

**Goal / observable completion:** `cb-train` compiles and `--lib` tests stay green
with the three new variants present. `EvalMetric::Mae`, `EvalMetric::Mape`,
`EvalMetric::Quantile { alpha: f64 }` exist; `eval` has three arms that return a
typed "not yet implemented" `CbError::Degenerate` (inert placeholders, flipped to
real math by EMT-2..4); `eval_one_group` rejects all three via its flat-reject arm.

**Prerequisites:** none.

**Files & symbols:**
- Modify `crates/cb-train/src/metrics.rs`:
  - enum `EvalMetric` (:63-151) — add `Mae`, `Mape`, `Quantile { alpha: f64 }`
    with doc comments noting "Min-optimized, `is_max_optimal == false`, flat".
  - `eval` `match self` (:284) — add three arms, each
    `Self::Mae|Self::Mape|Self::Quantile{..} => Err(CbError::Degenerate("<metric> eval not yet implemented".to_owned()))`
    (temporary; each is replaced by its metric task).
  - `eval_one_group` reject arm (:513) — extend to include
    `Self::Mae | Self::Mape | Self::Quantile { .. }`.
- No edit to `use_group_weight` (:443), `empty_metric_default` (:524), or
  `is_ranking` (calc_metrics.rs:250) — verify each still compiles/behaves (bool
  `false` / wildcard).

**TDD sequence:**
1. **Red** — add ONLY the enum variants first; run `cargo build -p cb-train` and
   observe **E0004 non-exhaustive `match`** at `metrics.rs:284` and `:494`. That
   compile failure IS the Red for the compile-safety spine (R3).
2. **Green** — add the inert `eval` arms and extend the `eval_one_group` reject
   arm; `cargo build -p cb-train` succeeds; `cargo test -p cb-train --lib` stays
   green (no new tests yet).
3. **Refactor** — none beyond doc comments; keep arm ordering next to the flat
   arms (Rmse/Logloss/Msle) for readability.
4. **Verify** — `cargo test -p cb-train --lib` green; `cargo clippy -p cb-train
   --lib --no-deps` clean; grep-confirm no other `match … EvalMetric` /
   `impl … for EvalMetric` site remains unhandled.

**Completion evidence:** clean build + green `--lib` + clippy clean with the
three variants present and inert. **Parallelization:** none — blocks all.

---

## EMT-2 — MAE final error  [EM-01]

**Goal / observable completion:** `EvalMetric::Mae.eval(approx,target,weight)`
returns `Σ w·|a−t| / Σ w`; degenerate inputs → `CbError::Degenerate` (reusing the
shared :246-282 guards). `mae_eval` (and weighted/degenerate) pass.

**Prerequisites:** EMT-1.

**Files & symbols:**
- Add tests to `crates/cb-train/src/metrics_test.rs` (mounted at metrics.rs:617):
  `mae_eval`, `mae_eval_weighted`, `mae_rejects_length_mismatch`.
- Modify `crates/cb-train/src/metrics.rs` `eval` `match self` (:284) — replace the
  `Self::Mae` inert arm with the weighted-MAE fold via `cb_core::sum_f64` (mirror
  the `Rmse` arm shape at :285-297: build `weight_at(i) * (a-t).abs()` vec, divide
  by the already-computed `total_weight`).

**TDD sequence:**
1. **Red** — `mae_eval`: `approx=[1.0,3.0]`, `target=[2.0,2.0]`, empty weight ⇒
   expect `1.0`. Fails (inert arm returns `Err`). Run `cargo test -p cb-train --lib`.
2. **Green** — implement the `Self::Mae` arm; test passes.
3. **Refactor** — factor a shared `weighted_mean(|closure|)` helper only if it
   cleanly serves EMT-3/EMT-4 too (optional; no behavior change).
4. **Verify** — `cargo test -p cb-train --lib`; `cargo clippy -p cb-train --lib --no-deps`.

**Completion evidence:** the three MAE unit tests green; clippy clean.
**Parallelization:** serialize with EMT-3/EMT-4 (same `eval` match block).

---

## EMT-3 — MAPE final error + zero-target guard  [EM-02]  (R1 provisional)

**Goal / observable completion:** `EvalMetric::Mape.eval` returns the weighted MAPE
with a zero-target-safe divisor (no NaN/Inf); `mape_eval` + `mape_zero_target_finite`
pass. The exact divisor convention is a **stated hypothesis, finalized in EMT-6**.

**Prerequisites:** EMT-1.

**Files & symbols:**
- Add tests to `metrics_test.rs`: `mape_eval` (hand calc on an all-nonzero target,
  consistent with the chosen divisor), `mape_zero_target_finite` (a `0.0` in
  target ⇒ result `is_finite()`, no NaN/Inf), `mape_rejects_length_mismatch`.
- Modify `metrics.rs` `eval` `match self` — replace the `Self::Mape` inert arm with
  `Σ w·|a−t| / D(t) / Σ w`, where `D(t)` is the guarded divisor.

**R1 decision gate (do NOT silently choose):**
- Implement `D(t)` behind a single clearly-commented expression. **Hypothesis for
  first Green:** `D(t) = t.abs().max(1.0)` (candidate upstream `TMAPEMetric`
  convention). Alternatives to keep in view: `t.abs().max(EPS)` (SPEC §4 wording)
  and skip-zero-rows.
- The `mape_zero_target_finite` unit test asserts **finiteness only** (not a
  specific value) so it holds under any of the three conventions. The exact
  numeric value assertion lives in EMT-6 against the frozen upstream scalar; if
  EMT-6 diverges, correct `D(t)` here and update `mape_eval`'s expected value.

**TDD sequence:**
1. **Red** — `mape_zero_target_finite`: target contains `0.0` ⇒ inert arm returns
   `Err` (fails). Run `--lib`.
2. **Green** — implement the `Self::Mape` arm with the guarded divisor; tests pass.
3. **Refactor** — reuse the EMT-2 mean helper if present.
4. **Verify** — `--lib` + clippy.

**Completion evidence:** MAPE unit tests green; a `// R1: divisor convention pinned
by calc_metrics_flat_oracle_test::mape*` comment marks the finalization point.
**Parallelization:** serialize with EMT-2/EMT-4.

---

## EMT-4 — Quantile(alpha) final error  [EM-03 math]

**Goal / observable completion:** `EvalMetric::Quantile{alpha}.eval` returns the
weighted mean pinball loss `Σ w·pinball / Σ w`, `pinball = t≥a ? alpha·(t−a) :
(1−alpha)·(a−t)`; at `alpha=0.5` equals `0.5·MAE`. `quantile_eval_default`,
`quantile_eval_alpha` pass. (Parse of `alpha` is EMT-5.)

**Prerequisites:** EMT-1.

**Files & symbols:**
- Add tests to `metrics_test.rs`: `quantile_eval_default`
  (`Quantile{alpha:0.5}` == `0.5·Mae` on the EMT-2 vectors),
  `quantile_eval_alpha` (`alpha=0.9`, asymmetric hand calc).
- Modify `metrics.rs` `eval` `match self` — replace the `Self::Quantile { alpha }`
  inert arm with the pinball fold via `cb_core::sum_f64` (bind `alpha` by ref /
  deref, mirroring the ranking arms' `*param` style, since `self` is matched by
  reference).
- Doc comment references `cb_compute::QUANTILE_ALPHA` (=0.5) as the shared default;
  the arm itself uses the runtime `alpha` field.

**TDD sequence:**
1. **Red** — `quantile_eval_default` ⇒ inert arm `Err` (fails). Run `--lib`.
2. **Green** — implement the `Self::Quantile` pinball arm; tests pass.
3. **Refactor** — reuse the EMT-2 mean helper.
4. **Verify** — `--lib` + clippy.

**Completion evidence:** both Quantile eval unit tests green; clippy clean.
**Parallelization:** serialize with EMT-2/EMT-3.

---

## EMT-5 — parse_metric recognition + DEFERRED_METRICS update  [EM-04 / EM-03 parse]

**Goal / observable completion:** `parse_metric` accepts `mae`, `mape`, and
`quantile[:alpha=..]` (case-insensitive), defaulting `alpha=0.5`; bogus params
(`mae:top=2`, `quantile:beta=1`) → `CbError::Degenerate`; MAE/MAPE removed from
`DEFERRED_METRICS`; unknown-name help lists the three. Existing flat-oracle
regression stays green.

> **Split (per Plan-Check MAJOR):** EMT-5 is PARSE-ONLY and parallel-safe after
> EMT-1. It does NOT include a `calc_metric` success probe, because EMT-1's inert
> `eval` arms return `Err` until EMT-2/3/4 land the math — so any test asserting
> `calc_metric(Mae|Mape|Quantile)` **succeeds** cannot go Green from EMT-5's own
> changes. The routing/success assertion (that the three route to the flat `eval`
> and succeed) is verified in **EMT-6** once the math exists (its oracle gates
> exercise exactly that flat path). The `is_ranking`-is-false property is instead
> asserted *structurally* here via the parse result plus a code-reading note, not
> via a runtime success probe.

**Prerequisites:** EMT-1 (enum arms). Parse needs no eval math ⇒ **parallel-safe
with EMT-2..4** (edits `calc_metrics.rs`, a different file from the `metrics.rs`
`eval` block).

**Files & symbols:**
- Add tests to `crates/cb-train/src/calc_metrics_test.rs` (mounted at
  calc_metrics.rs:320) — **parse-only, no `calc_metric` success probe**:
  `parse_mae_mape_quantile` (`"MAE"`→`Mae`, `"mape"`→`Mape`,
  `"Quantile"`→`Quantile{alpha:0.5}`, `"Quantile:alpha=0.3"`→`Quantile{alpha:0.3}`),
  `parse_rejects_bad_quantile_param` (`"quantile:beta=1"` → `Err`; `"mae:top=2"` →
  `Err`). (`is_ranking` is private; the flat-routing SUCCESS assertion is deferred
  to EMT-6 — see split note above.)
- Modify `crates/cb-train/src/calc_metrics.rs`:
  - `parse_metric` `match` (:157) — add:
    `"mae" => { p.reject_unknown(name, &[])?; EvalMetric::Mae }`,
    `"mape" => { p.reject_unknown(name, &[])?; EvalMetric::Mape }`,
    `"quantile" => { p.reject_unknown(name, &["alpha"])?; EvalMetric::Quantile { alpha: p.f64_or("alpha", 0.5, name)? } }`.
  - `DEFERRED_METRICS` (:23) — remove `MAE, MAPE`.
  - unknown-name help (:235) — add `MAE, MAPE, Quantile` to the supported list.

**TDD sequence:**
1. **Red** — `parse_mae_mape_quantile` ⇒ `parse_metric` returns the unknown-name
   `Err` today (fails). Run `--lib`.
2. **Green** — add the three parse arms; edit DEFERRED_METRICS + help string.
   Tests pass.
3. **Refactor** — none (arms mirror the existing pattern).
4. **Verify** — `cargo test -p cb-train --lib`; `cargo test -p cb-train --test
   calc_metrics_flat_oracle_test` (RMSE/Logloss/MSLE regression unchanged);
   clippy clean.

**Completion evidence:** parse-accept + reject unit tests green; flat oracle
(RMSE/Logloss/MSLE) regression green. (No `calc_metric` success probe here.)
**Parallelization:** parallel with EMT-2..4 (different file).

---

## EMT-6 — Oracle parity per metric + R1/R2 pinning  [EM-05]

**Goal / observable completion:** `calc_metric` matches `catboost.utils.eval_metric`
≤1e-5 for MAE, MAPE, Quantile(0.5), Quantile(0.9). The MAPE zero-target divisor
(R1) and Quantile weighting (R2) are empirically pinned; EMT-3's `D(t)` finalized.

**Prerequisites:** EMT-2, EMT-3, EMT-4, EMT-5.

**EM-05 gate decision (per Plan-Check MINOR — option (a) chosen):** REUSE the
existing shared `flat_inputs()` `(label, approx)` pair; the `gate(metric,
fixture, weight)` helper (calc_metrics_flat_oracle_test.rs:36) is used AS-IS,
unchanged. No `gate_with_inputs` variant, no dedicated `label_mape.npy`. This is
valid because `label` is pinned to `{0,1}` and therefore **already contains zero
rows** — the zero rows are the R1 arbiter: `max(ε,|t|)` (zero→ε, huge term),
`max(1,|t|)` (zero→1), and skip-zero (drops the row from the denominator count)
each produce a **distinct** frozen upstream scalar, so the `{0,1}` label pins the
zero-target convention without a richer vector. All scenarios pass an
`EvalMetric` value directly to `gate()` (including `EvalMetric::Quantile { alpha:
0.9 }`), so no metric-string parsing is needed in the oracle test.
- (Fallback, only if EMT-6 Green finds the zero-row scalar under-determines
  `D(t)` — e.g. `max(1,|t|)` vs plain-`|t|` for fractional targets, which `{0,1}`
  cannot separate: THEN add a single `gate_with_inputs(inputs, metric, fixture,
  weight)` helper plus one `label_mape.npy` with a `0<|t|<1` row. Treat this as a
  contingency, not the default path.)

**Files & symbols:**
- Modify `crates/cb-oracle/generator/gen_ranking_fixtures.py`
  `gen_calc_metrics_flat()` (:368) — reuse the existing frozen `label`/`approx`/
  `weight` `.npy`; add scenarios via the existing `freeze(name, metric, **kw)`
  helper: `freeze("mae","MAE")`, `freeze("mape","MAPE")`,
  `freeze("quantile_default","Quantile")`, `freeze("quantile_a90","Quantile:alpha=0.9")`.
  - **Weighted scenarios (`mae_weighted`, `quantile_default_weighted` via
    `weight=weight`): FIRST re-confirm at freeze time that `catboost.utils.
    eval_metric` accepts `weight=` for MAE and Quantile in `catboost==1.2.10`**
    (it is confirmed for RMSE here, but MAE/Quantile support must be verified
    before committing those `.npy` — the generator's `freeze` already raises on a
    non-finite result; add an explicit try/skip so a `weight=`-unsupported metric
    does not silently emit a bogus fixture). Omit the weighted variant for any
    metric where `weight=` is rejected, and note it in `summary.json`.
  - Keep all writes offline/RUN-ONCE; commit the resulting `.npy` under
    `crates/cb-oracle/fixtures/calc_metrics/`.
- Modify `crates/cb-train/tests/calc_metrics_flat_oracle_test.rs`: add
  `mae_matches_upstream`, `mape_matches_upstream` (the `{0,1}` label's zero rows
  ARE the zero-target case — the R1 arbiter), `quantile_default_matches_upstream`,
  `quantile_alpha90_matches_upstream`, plus `mae_weighted_matches_upstream` /
  `quantile_default_weighted_matches_upstream` **only if** the corresponding
  weighted fixture was frozen — each via the existing `gate(...)` helper with an
  `EvalMetric` constructed directly (e.g. `EvalMetric::Quantile { alpha: 0.9 }`).
  These gates call `calc_metric(&metric, …, &[])` and thereby also constitute the
  **flat-routing SUCCESS assertion** deferred from EMT-5 (the three metrics route
  to the flat `eval` and succeed; `is_ranking` is `false`).

**TDD sequence:**
1. **Red** — generate the fixtures (uv recipe above), add the gate tests; run
   `cargo test -p cb-train --test calc_metrics_flat_oracle_test`. Any MAPE mismatch
   here is the R1 signal.
2. **Green** — if MAPE diverges, correct EMT-3's `D(t)` to the convention the
   fixture demands and update EMT-3's `mape_eval` expected value; if Quantile
   diverges, reconcile the R2 weighting. Re-run until all gates pass ≤1e-5.
3. **Refactor** — deduplicate any gate boilerplate; keep fixtures minimal.
4. **Verify** — full gate: `cargo test -p cb-train --lib`, `--test
   calc_metrics_flat_oracle_test`, `--test eval_metrics_oracle_test` (regression),
   `cargo clippy -p cb-train --lib --no-deps`.

**Completion evidence:** all new oracle gates green ≤1e-5; R1 convention recorded
in a code comment + this plan's blocker section resolved; both regression tests green.
**Parallelization:** none — final task.

---

## 3. Completion criteria (phase)

- [x] EMT-1: crate compiles with the three variants; both E0004 sites closed; `--lib` green.
- [x] EMT-2: MAE eval == weighted `Σw|a−t|/Σw`; unit tests green.
- [x] EMT-3: MAPE eval finite under zero target; unit tests green (value provisional).
- [x] EMT-4: Quantile eval == pinball; default == `0.5·MAE`; unit tests green.
- [x] EMT-5: parse accepts mae/mape/quantile[:alpha]; DEFERRED/help updated; flat oracle regression green.
- [x] EMT-6: MAE/MAPE/Quantile(0.5)/Quantile(0.9) ≤1e-5 vs upstream; R1 pinned; R2 confirmed.
- [x] No production `unwrap`/`expect`/`panic`/indexing (clippy clean).
- [x] No training-loss wiring, no ranking variants, no `Rmse`/`Logloss`/`Msle` behavior change.

> ## Execution status (2026-07-19) — ✅ COMPLETE (EMT-1..6), verified green
> The full slice is implemented in the working tree and re-verified this session:
> - **Unit** `cargo test -p cb-train --lib` = 283 passed / 0 failed, incl. the 14
>   new MAE/MAPE/Quantile/parse tests.
> - **Oracle** `cargo test -p cb-train --test calc_metrics_flat_oracle_test` = 10
>   passed (6 new gates ≤1e-5 vs `catboost==1.2.10`): `mae`, `mape` (R1 arbiter),
>   `quantile_default`, `quantile_alpha90`, `mae_weighted`, `quantile_default_weighted`.
> - **Regression** `--test eval_metrics_oracle_test` = 3 passed (RMSE/Logloss/MSLE
>   train-time curves unchanged).
> - **Lint** `cargo clippy -p cb-train --lib --no-deps` clean; no
>   `unwrap`/`expect`/`panic`/indexing in the new arms.
> - **R1 RESOLVED:** `D(t) = max(1.0, |t|)` — the frozen `mape.npy` scalar under the
>   `{0,1}` label uniquely selects this over `max(EPS,|t|)`/skip-zero.
> - **Files:** `crates/cb-train/src/{metrics,metrics_test,calc_metrics,calc_metrics_test}.rs`,
>   `crates/cb-train/tests/calc_metrics_flat_oracle_test.rs`,
>   `crates/cb-oracle/generator/gen_ranking_fixtures.py`, and the committed fixtures
>   under `crates/cb-oracle/fixtures/calc_metrics/` (mae/mape/quantile*.npy + summary.json).
> **Not committed** (working-tree change; awaiting operator's commit decision).

## 4. Unresolved blockers to surface to the operator

- **R1 (MAPE divisor) — RESOLVED (EMT-6).** `D(t) = max(1.0, |t|)` — the upstream
  `TMAPEMetric` convention. Pinned against the frozen `catboost==1.2.10` scalar
  `mape.npy = 0.8651406907243623`: the `{0,1}` label's zero-target rows make the
  three candidate conventions mutually distinct, and only `max(1.0,|t|)` reproduces
  upstream (~1e-16); SPEC §4's `max(|t|, ε)` explodes to ~1e37 and skip-zero
  undershoots (0.761). No code change to EMT-3's arm was needed — the provisional
  hypothesis was correct; only the in-arm comment was updated to mark R1 RESOLVED.
- **R2 (Quantile weighting/alpha) — LOW.** `f64_or("alpha",0.5)` assumed sole param;
  weighting assumed `Σw·pinball/Σw`. Confirmed by EMT-6's Quantile(0.9) gate.
- **R3 (exhaustive matches) — RESOLVED.** Exactly two E0004-hard sites
  (`metrics.rs:284` eval, `metrics.rs:494` eval_one_group); `use_group_weight`,
  `empty_metric_default`, `is_ranking` need no edit; no `Display` impl. Closed by EMT-1.
