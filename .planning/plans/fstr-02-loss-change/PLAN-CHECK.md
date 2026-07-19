## Plan Check Result

**Verdict:** ISSUES_FOUND
**Goal:** Generalize `LossFunctionChange` beyond hard-coded Logloss to the model's actual Min-optimized numeric loss (RMSE, MAE, MAPE, Quantile, Logloss) via a facade-injected final-error closure, keeping cb-model free of the metric implementations.
**Plan:** `.planning/plans/fstr-02-loss-change/PLAN.md` (v2)

### Summary
- The plan is well structured, every acceptance behavior (FL-01..FL-04 + §6) has a Red task, the Q1 resolution and the `UnsupportedLoss`/py-error handling are correct and fully specified, and both gates (GATE-A, GATE-B) are validly declared.
- BUT the "decisive" layering premise that the whole design is justified on is FACTUALLY INVERTED. `cb-model` depends on `cb-train` as a NORMAL build-graph dependency (not the reverse), so `cb_model::fstr::loss_function_change` CAN call `cb_train::EvalMetric` directly with no circular dependency. The injected-closure design still compiles and reaches the goal, but its stated rationale, several `[VERIFIED]` citations, and FL-04a's "cb-model tests cannot use cb-train" justification are wrong and must be corrected.
- One MAJOR execution gap: the facade crate mounts unit tests in `lib.rs` (`#[cfg(test)] mod *_test;`), not via `#[path]` inside source files; there is no `model_test.rs` and no mount for it. FL-03's facade tests can silently not run unless the mount is added in `lib.rs`.

### Specification Coverage
- [x] FL-01 injected closure core: TASK-FL-01-CORE + Red `loss_change_uses_injected_final_error`. Verified the current 4-arg Logloss-hardcoded `loss_function_change` (fstr.rs:788-827) and `logloss_final_error` (fstr.rs:833-848).
- [x] FL-02 Logloss back-compat: TASK-FL-02-REGRESSION migrates `crates/cb-model/tests/fstr_oracle_test.rs` to `loss_function_change_logloss`; existing frozen fixtures present (`fstr_loss_change/{oblivious,non_symmetric}_loss_function_change.npy`).
- [x] FL-03 facade selects loss / rejects Max: TASK-FL-03-FACADE adds `loss: &str`, allow-list, `UnsupportedLoss`, py arm. Verified the facade arm (catboost-rs/src/model.rs:177-191) and `pool.label()` usage.
- [x] FL-04 oracle per loss: FL-04a (RMSE+Logloss, ungated) + FL-04b (MAE/MAPE/Quantile, GATE-B). Fixture generator `fstr_loss_change/gen_fixtures.py` exists.
- [x] §6 scenarios 1-4 map to FL-01/03/04, FL-04, FL-02, FL-03 respectively.

### CodeGraph / Local Evidence
- `loss_function_change` (crates/cb-model/src/fstr.rs:788) — real in-repo callers: the `pub use` re-export in `crates/cb-model/src/lib.rs`, and the facade `feature_importance_with_data` in `crates/catboost-rs/src/model.rs:180`; covering test `crates/cb-model/tests/fstr_oracle_test.rs`. The `crates/cb-train/src/feature_selection.rs` grep hit is a DOC COMMENT only (line 17); `select_features` uses an injected `ranker` closure, not a direct call — NOT a caller. Plan's "2 callers + migrate both + test" is correct.
- `feature_importance_with_data` (catboost-rs/src/model.rs:169) — only caller/definition is within `catboost-rs/src/model.rs`; no PyO3 binding calls it (`grep feature_importance` in catboost-rs-py = empty). Adding `loss: &str` is fully contained. Verified.
- `Model` (crates/cb-model/src/model.rs:271-313) — fields are oblivious_trees, non_symmetric_trees, region_trees, bias, float_feature_borders, ctr_data, approx_dimension, class_to_label. NO loss/objective field. `decode_cbm` (cbm.rs:827-892) reads only `class_to_label` from InfoMap. Q1 resolution (facade takes explicit `loss: &str`) is SOUND.
- `to_pyerr` (crates/catboost-rs-py/src/errors.rs:113-135) — exhaustive `match err { … }` with NO wildcard. `CatBoostError` (catboost-rs/src/error.rs:32-85) is NOT `#[non_exhaustive]`. Adding `UnsupportedLoss` => E0004 in the py crate until a `to_pyerr` arm is added. Plan's FL-03 (arm + `errors_test.rs` + `cargo build -p catboost-rs-py`) correctly covers this.
- GATE-A: `git status --porcelain crates/cb-model/src/fstr.rs` => ` M …` (uncommitted). Valid and active.
- GATE-B: `EvalMetric` (cb-train/src/metrics.rs:64) — no `Mae`/`Mape`/`Quantile` arms; `is_ranking` (calc_metrics.rs:250) flat metrics false; `calc_metric` (calc_metrics.rs:281) routes flat via `metric.eval`. `DEFERRED_METRICS` text lists "…MAE, MAPE" (NOT Quantile). Quantile is absent from `EvalMetric` entirely (`grep Quantile` in cb-train shows only `Loss::Quantile`, the training-loss enum). eval-metric-extension SPEC §4 adds all three, so gating FL-04b on GATE-B is correct; FL-04a (RMSE+Logloss exist today) is correctly ungated.

### Issues

#### [CRITICAL] The "decisive" layering premise is inverted — cb-model DEPENDS ON cb-train (normal dep)
- **Plan location:** PLAN §0 "Layering (decisive)"; SPEC §1 and §3; FL-04a §6 ("cb-model tests cannot use cb-train").
- **Requirement:** "cb-model cannot depend on cb-train" is the stated justification for the injected-closure contract.
- **Evidence:** `crates/cb-model/Cargo.toml` line 26 lists `cb-train = { path = "../cb-train", default-features = false }` under `[dependencies]` (NOT dev-deps); line 63 comment: "cb-train is already a NORMAL dependency above". `crates/cb-model/src/model.rs` uses `pub use cb_train::Split;` (line 31), `cb_train::TProjection` (line 45), `cb_train::Model` (line 316) in PRODUCTION source. Conversely `crates/cb-train/Cargo.toml` lists `cb-model` only as a DEV-dependency (lines 39-45: "A DEV-dependency ONLY — the normal cb-train build graph never links cb-model (cb-model depends on cb-train in the build graph…)"). The plan's `[VERIFIED: LOCAL crates/cb-model/Cargo.toml — dependency is cb-core only]` and `[VERIFIED: CODEGRAPH … cb-train dep on cb-model]` citations are false; the real direction is cb-core <- cb-train <- cb-model.
- **Failure scenario:** No build failure — the injected-closure design still compiles and works. The failure is one of reasoning/evidence integrity: an implementer operating on the plan's mental model will (a) be told a circular dependency exists that does not, (b) be told cb-model tests "cannot use cb-train" when cb-train is fully available in cb-model's tests (FL-04a's stated rationale for an independent RMSE closure is false — the independent closure is still fine, just not "required"), and (c) be blocked from noticing a simpler valid alternative (call `cb_train::EvalMetric` directly inside `fstr.rs`, taking `loss: &str`/`&EvalMetric`, dropping the facade closure plumbing).
- **Impact:** The core design decision rests on a false, `[VERIFIED]`-tagged structural claim. The plan remains executable, but cannot be approved with a decisive premise contradicted by the actual dependency graph.
- **Required revision:** Correct SPEC §1/§3 and PLAN §0 to state the real direction (cb-model depends on cb-train; cb-train depends on cb-model only as a dev-dep). Re-justify the injected-`Fn` closure on DECOUPLING grounds ("keep fstr.rs free of cb-train's metric enum even though the dependency edge exists"), or consciously adopt the simpler direct-`EvalMetric` design. Fix FL-04a's rationale: cb-model tests CAN use cb-train; the independent RMSE closure is a chosen stronger check, not a forced one.

#### [MAJOR] Facade unit tests (FL-03) may silently not run — wrong test-mount mechanism for the catboost-rs crate
- **Plan location:** PLAN §5 TASK-FL-03-FACADE, "Test crates/catboost-rs sibling mounted *_test.rs (e.g. model_test.rs — verify the `#[cfg(test)] #[path=…] mod tests;` mount exists in model.rs, else 0 tests run)".
- **Requirement:** FL-03 acceptance = `loss_change_rmse_facade` + `loss_change_rejects_max_metric` must execute.
- **Evidence:** `crates/catboost-rs/src/model.rs` has NO `#[cfg(test)]`/`#[path]`/`mod tests` (grep exit 1). The crate mounts test modules in `lib.rs`: `#[cfg(test)] mod error_test;` / `mod metrics_test;` / `mod onnx_test;` (lib.rs:57-62), with sibling files `src/{error,metrics,onnx}_test.rs`. There is no `model_test.rs` and no mount for it.
- **Failure scenario:** An implementer following the plan looks for a `#[path]` mount inside `model.rs` (none exists), and if they create `model_test.rs` assuming a mount is present, the file is never compiled and 0 facade tests run — FL-03's reject/parity assertions pass vacuously.
- **Impact:** FL-03 acceptance unverifiable; a Max-metric that is NOT rejected, or an RMSE facade parity break, would go undetected.
- **Required revision:** FL-03 must (1) create `crates/catboost-rs/src/model_test.rs`, and (2) add `#[cfg(test)] mod model_test;` to `crates/catboost-rs/src/lib.rs` after line 62, mirroring the existing `error_test`/`metrics_test`/`onnx_test` mounts. Drop the incorrect "`#[path=…]` mount in model.rs" instruction.

#### [MINOR] Inaccurate claim that Quantile is "in DEFERRED_METRICS"
- **Plan location:** PLAN §0 (line 61-64) and §8 GATE-B.
- **Evidence:** `DEFERRED_METRICS` (calc_metrics.rs:22-24) text is "…AUC, Accuracy, F1, Precision, Recall, R2, MAE, MAPE" — Quantile is NOT listed. Quantile is simply absent from `EvalMetric` entirely (no enum arm, no `parse_metric` arm), so `parse_metric("quantile")` would fail as an unknown name.
- **Impact:** None on gating — Quantile is still unavailable today and GATE-B (eval-metric-extension §4) adds `EvalMetric::Quantile { alpha }`, so FL-04b's gate on GATE-B is correct. Wording only.
- **Required revision:** State that MAE/MAPE are in `DEFERRED_METRICS` and Quantile is absent from `EvalMetric` altogether; both are delivered by GATE-B.

#### [MINOR] "cb-model depends only on cb-core" understates the dep set
- **Evidence:** cb-model/Cargo.toml `[dependencies]` also lists `cb-data` (line 29) and `cb-train` (line 26). Same root cause as the CRITICAL; fix alongside it.

### Implementation Order Review
1. GATE-A (FSTR-01 fstr.rs committed) before any fstr.rs edit — correct; verified fstr.rs still `M`.
2. TASK-FL-01-CORE before TASK-FL-02-REGRESSION — correct: both touch the same crate/file, serialized; FL-02's Red is the compile break of the migrated 4-arg call sites.
3. Wave 2: TASK-FL-03-FACADE ∥ TASK-FL-04a-ORACLE — disjoint files (catboost-rs/src + catboost-rs-py/src vs cb-oracle/fixtures + cb-model/tests), safe to parallelize; both need FL-01. Correct.
4. TASK-FL-04b-ORACLE after FL-01 + GATE-B + FL-03-mmq-arms — correct (EvalMetric MAE/MAPE/Quantile must exist first).
   No cycle; the ordering graph is valid. The only ordering-relevant correction is the test-mount step inside FL-03 (MAJOR above).

### Potential Bugs
- LossFunctionChange formula parity per loss: the `finalError(approx − shap_f) − finalError(approx)` structure is oracle-proven only for Logloss today. RMSE/MAE/MAPE/Quantile parity depends on upstream using the same document-importance aggregation for those metrics. This is exactly what FL-04a/FL-04b `≤1e-5` fixtures gate, so it is acceptable as planned — but the plan should keep the oracle as the source of truth and NOT assert per-metric correctness by construction. (No revision required beyond keeping FL-04 fixtures as the gate.)
- MAPE zero-target divisor: handled in the GATE-B prerequisite (eval-metric-extension EM-02), not in FSTR-02. Ensure the FL-04b MAPE fixture includes a zero/near-zero target row so the guard is exercised end-to-end through LossFunctionChange, not only in the metric unit test.

### Required Plan Revisions
1. Correct the dependency-direction statements (SPEC §1/§3, PLAN §0) and remove/replace the false `[VERIFIED]` citations; re-justify the injected-closure design on decoupling grounds or adopt the direct-`EvalMetric` alternative.
2. Fix FL-04a's "cb-model tests cannot use cb-train" rationale (cb-train IS available in cb-model tests).
3. FL-03: create `crates/catboost-rs/src/model_test.rs` and add `#[cfg(test)] mod model_test;` to `catboost-rs/src/lib.rs` (after line 62), matching the crate's lib.rs mount convention; remove the "`#[path]` mount in model.rs" instruction.
4. Correct the DEFERRED_METRICS / "depends only on cb-core" wording (minor).
5. (Optional) FL-04b MAPE fixture: include a zero-target row.

### Unverified Items
- None material. All structural claims exercised by the plan (caller blast radius, Model fields, to_pyerr exhaustiveness, CatBoostError exhaustiveness, EvalMetric metric set, dependency direction, GATE-A git state, facade test-mount convention) were verified against the current sources. GATE-B (eval-metric-extension) is a declared, checker-pending external prerequisite; FSTR-02 correctly gates FL-04b/FL-03-mmq-arms on it and does not assume the metrics already exist — acceptable per the review scope.

---

## Plan Check Result — PASS 2 (re-review of revised SPEC + PLAN v3)

**Verdict:** PASS
**Goal:** Generalize `LossFunctionChange` beyond hard-coded Logloss to the model's actual Min-optimized numeric loss (RMSE, MAE, MAPE, Quantile, Logloss) via a facade-injected final-error closure, keeping cb-model's `fstr.rs` free of the metric implementations.
**Plan:** `.planning/plans/fstr-02-loss-change/PLAN.md` (v3) + `SPEC.md` (spec_version 1, §1 revised)

### Summary
- All three PASS-1 issues are resolved and independently re-verified against current sources. No new issues introduced; the two-prerequisite gate, FL-04a/FL-04b split, Q1, and the `UnsupportedLoss`/E0004 py-trap remain intact and correct.

### CRITICAL (layering premise) — RESOLVED
- **Real dependency direction re-verified via Read:** `crates/cb-model/Cargo.toml:26` lists `cb-train = { path = "../cb-train", default-features = false }` under `[dependencies]` (NORMAL), plus `cb-core:23` and `cb-data:29` — the full dep set. `crates/cb-train/Cargo.toml:45` lists `cb-model` under `[dev-dependencies]` only, with the inline comment (lines 39-41) "cb-model depends on cb-train in the build graph; dev-deps are exempt from the cycle." Direction is `cb-core <- cb-train <- cb-model`, cb-train->cb-model dev-only. Confirmed.
- **SPEC §1 (lines 37-51)** now explicitly labels the earlier "cb-model depends only on cb-core / circular dependency" claim as WRONG, states the corrected direction with a `[VERIFIED: LOCAL cb-model/Cargo.toml:26,63; cb-train/Cargo.toml:39-45]` citation, and re-justifies the injected closure as a decoupling/testability choice ("NOT because layering forbids the direct call"), explicitly noting the direct-`EvalMetric` alternative is now known viable. The false `[VERIFIED]` "cb-core only" / "cb-train dep on cb-model" citations are gone.
- **PLAN §0 (lines 51-60)** matches: "cb-model DOES depend on cb-train … the earlier 'circular dependency' justification was wrong … injected closure kept as a deliberate decoupling/testability choice." Dep set stated as "cb-core + cb-data + cb-train, NOT 'cb-core only'."
- **FL-04a rationale corrected (PLAN §6, lines 266-270):** now reads "cb-model tests CAN use cb-train — it is a normal dependency — but a hand-written closure is PREFERRED here as an INDEPENDENT reimplementation: a stronger oracle." The false "cb-model tests cannot use cb-train" is removed; the independent RMSE closure is correctly framed as a chosen stronger check, not a forced one.

### MAJOR (facade test-mount) — RESOLVED
- **catboost-rs unit-test mount convention re-verified via Read:** `crates/catboost-rs/src/lib.rs:57-62` mounts `#[cfg(test)] mod error_test; mod metrics_test; mod onnx_test;` — there is NO `model_test.rs` and no `#[path]` mount anywhere in `model.rs`.
- **Integration-test precedent re-verified via ls:** `crates/catboost-rs/tests/` contains `onnx_facade_test.rs`, `partial_dependence_facade_test.rs`, `builder_oracle_test.rs` — the established home for data-bearing facade tests.
- **PLAN §5 (lines 227-233)** now places FL-03's facade tests in a NEW integration file `crates/catboost-rs/tests/fstr_loss_change_facade_test.rs`, explicitly warning that a sibling `#[path]` mount in `model.rs` "would silently run 0 tests." The prior incorrect "`#[path]` mount / `model_test.rs`" instruction is gone. Cases `loss_change_rmse_facade` + `loss_change_rejects_max_metric` retained. Traceability (§10) lists the same precedent files. Correct.

### MINOR (Quantile / dep-set wording) — RESOLVED
- **PLAN §0 (lines 74-79)** now states: "MAE and MAPE are named in `DEFERRED_METRICS`; Quantile is NOT listed there but is likewise absent from `EvalMetric` entirely." SPEC §2 (line 68) and PLAN §8 (lines 326-327) match. "depends only on cb-core" phrasing dropped throughout (SPEC §1 line 40 flags it WRONG; PLAN §0 line 54 gives the full three-crate dep set).

### No-regression re-verification
- **Two-prerequisite Wave-0 gate:** PLAN frontmatter `ordering_gate` (lines 15-25) carries GATE-A (FSTR-01 uncommitted fstr.rs) and GATE-B (eval-metric-extension). GATE-A re-verified live: `git status` still shows `M crates/cb-model/src/fstr.rs` (uncommitted). Intact.
- **FL-04a / FL-04b split:** §6 (RMSE+Logloss, ungated, only GATE-A) and §7 (MAE/MAPE/Quantile, gated GATE-B + FL-03-mmq-arms). Acyclic wave graph (§2). Intact.
- **Q1 explicit `loss: &str`:** re-verified `Model` (model.rs:271-313) has no loss/objective field (fields: oblivious/non_symmetric/region_trees, bias, float_feature_borders, ctr_data, approx_dimension, class_to_label). Facade takes explicit `loss: &str` (SPEC §2; PLAN §1b line 125). Intact.
- **UnsupportedLoss + to_pyerr E0004 fix:** re-verified `crates/catboost-rs/src/error.rs:33` `pub enum CatBoostError` has NO `#[non_exhaustive]`; `crates/catboost-rs-py/src/errors.rs:113-135` `to_pyerr` is an exhaustive `match err { … }` with NO wildcard (adding a variant => E0004 until an arm is added). `errors_test.rs` exists. PLAN §5 (lines 224-226) + §0 (lines 89-102) add the `UnsupportedLoss(m) => CatBoostValueError::new_err(m.clone())` arm, extend `errors_test.rs`, and gate on `cargo build -p catboost-rs-py`. Intact and correct.
- **Blast radius:** re-verified via CodeGraph — `loss_function_change` (fstr.rs:788) still Logloss-hardcoded (4-arg), covered by `crates/cb-model/tests/fstr_oracle_test.rs`; callers are the `lib.rs` re-export + the facade arm. Migration plan (both call sites + retained `loss_function_change_logloss` wrapper) unchanged and correct.

### Unverified Items
- None material. GATE-B (eval-metric-extension) remains a declared external prerequisite the plan correctly gates on rather than assuming; that is by design, not a verification gap.

### Verdict rationale
Every mandatory requirement maps to a concrete task with a Red gate; all structural claims (dependency direction, Model fields, exhaustive `to_pyerr`, non-`#[non_exhaustive]` error enum, EvalMetric metric set, blast radius, facade test conventions, GATE-A git state) are CodeGraph/Read-verified; the two prior evidence defects and the test-location defect are corrected; no regression in the retained decisions. **PASS.**
