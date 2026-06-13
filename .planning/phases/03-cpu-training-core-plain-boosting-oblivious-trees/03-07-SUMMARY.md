---
phase: 03-cpu-training-core-plain-boosting-oblivious-trees
plan: 07
subsystem: training
tags: [auto-learning-rate, autolr, TAutoLRParamsGuesser, rmse, logloss, oracle, parity, train-predict, capstone, TRAIN-08]

# Dependency graph
requires:
  - phase: 03-06
    provides: "cb-train boosting loop (train/train_with_eval/train_with_eval_sets), BoostParams (loss/iterations/depth/learning_rate/l2_leaf_reg/boost_from_average/use_best_model/eval_metric), the full deterministic train->predict path + oracle harness"
  - phase: 03-01
    provides: "cb-compute Loss enum (Rmse/Logloss/Mae); the generic Runtime boundary + CpuBackend the boosting loop drives"
  - phase: 01
    provides: "cb_core::CbError/CbResult (OutOfRange/Degenerate); cb_core::sum_f64 (not needed here — auto-LR is a pure scalar)"
provides:
  - "cb_train::autolr — TAutoLRParamsGuesser port: const CPU coefficient table {A,B,C,D} keyed by (TargetType{Rmse|Logloss|Unknown}, useBestModel, boostFromAverage) + the exp/log/round formula `lr = round(min(exp(A*ln N+B) * exp(C*ln iter+D)/exp(C*ln 1000+D), 0.5), 6)`; guess() guards object_count>0 / iter_count>0 (T-03-07-01) and returns Err for Unknown-target keys (matches upstream NeedToUpdate==false)"
  - "cb_train re-exports: autolr_guess, autolr_coefficients, TargetType"
  - "BoostParams.auto_learning_rate: bool — opt-in to pre-train auto-LR; when true AND the loss is auto-LR eligible, train_with_eval_sets guesses learning_rate before the loop (upstream UpdateLearningRate gate); explicit learning_rate is then ignored"
  - "autolr/{rmse,logloss} oracle scenario — upstream get_all_params()['learning_rate'] (RMSE 0.044808, Logloss 0.005413) + all keying inputs persisted; gen_autolr() in gen_fixtures.py"
  - "Phase-3 success criterion 5: a first FULL end-to-end CPU train->predict cycle runs with the auto-selected rate (autolr_e2e_test, RMSE + Logloss)"
affects: [cb-train, "Phase 4 (Builder API / model serialization — auto-LR is a pre-train default to surface)", "Phase 7 (GPU auto-LR rows already transcribed in upstream; CPU-only exposed now)", "Phase 8 (Python API: learning_rate=None default-to-auto behaviour)"]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "auto-LR is a PURE host scalar (exp/ln/round) — no float SUM, so no cb_core::sum_f64 routing is needed (D-08 grep is about summation; autolr.rs is grep-clean)"
    - "the coefficient table is a `match`-based const lookup returning Option<[f64;4]> (None == key absent == no guess), mirroring weights.rs' const-table + #[must_use] pure-fn style; a None lookup is the Rust analogue of upstream THashMap::contains in NeedToUpdate"
    - "the activation gate (learning_rate/leaf_estimation_method/leaf_estimation_iterations/l2_leaf_reg all unset) is collapsed to a single BoostParams.auto_learning_rate flag — BoostParams carries concrete f64 values for the other three params, so 'all four unset' is a host-caller decision expressed as the one opt-in flag; an Unknown-target loss (e.g. MAE) keeps its explicit rate (NeedToUpdate==false)"
    - "the effective learning rate is resolved ONCE at train entry into a local `learning_rate` (not a params mutation, since params is &BoostParams) and both downstream uses (model_length for random-strength, leaf-value scaling) read the local"

key-files:
  created:
    - crates/cb-train/src/autolr.rs
    - crates/cb-train/src/autolr_test.rs
    - crates/cb-train/tests/autolr_e2e_test.rs
    - crates/cb-oracle/fixtures/autolr/rmse/config.json
    - crates/cb-oracle/fixtures/autolr/logloss/config.json
  modified:
    - crates/cb-train/src/boosting.rs
    - crates/cb-train/src/lib.rs
    - crates/cb-oracle/generator/gen_fixtures.py
    - crates/cb-train/tests/{slice_first,bootstrap,leaf_methods,regularization,overfit,eval_metrics}_oracle_test.rs

key-decisions:
  - "The four-param activation gate is expressed as a single BoostParams.auto_learning_rate opt-in flag (BoostParams holds concrete values for leaf_estimation_method/iterations/l2_leaf_reg, so 'unset' is a caller decision). When set and the loss is in the table, learning_rate is guessed pre-train; otherwise the explicit value is used unchanged."
  - "Only the CPU coefficient rows are exposed (this phase is CPU-only). The GPU rows exist upstream and are transcribed in comments/research for Phase 7 but are not in the Rust table — TargetType has no TaskType axis since CPU is the only target."
  - "guess() returns Err (OutOfRange) on object_count==0 / iter_count==0 rather than computing ln(0)=-inf (T-03-07-01), and Err (Degenerate) for an Unknown-target key; boosting.rs treats the Degenerate (no-row) case as 'keep explicit rate' (NeedToUpdate==false) and propagates the OutOfRange guard."
  - "The autolr oracle uses iterations=500 (!= 1000) so the custIter/defIter ratio is genuinely exercised (defIter is fixed at iter=1000 in the formula); learn_object_count=50 (numeric_tiny). use_best_model defaults False (no eval set); boost_from_average defaults True for RMSE / False for Logloss (Pitfall 2)."
  - "The end-to-end test reuses the deterministic skeleton model.json quantization borders (same numeric_tiny inputs) — it is an OPERATIONAL train->predict assertion (cycle runs, non-empty model, finite staged predictions, applied rate == upstream selected rate), NOT a per-tree splits oracle (that parity is already locked by the slice oracles)."

patterns-established:
  - "Pattern 1: pure-scalar parity util (autolr) as a const match-table + #[must_use] guess() returning CbResult — the option-defaults analogue of weights.rs; oracle-locked on the persisted upstream get_all_params() value at <=1e-5."
  - "Pattern 2: a pre-train param-resolution flag on BoostParams (auto_learning_rate) resolved once into a local before the loop — the template for future auto-defaulted params (leaf_estimation_iterations auto-force, etc.)."
  - "Pattern 3: an operational end-to-end capstone test (train->predict cycle runs + applied-rate parity) distinct from the per-stage splits/leaves oracle — proves the wired path, not just the formula."

requirements-completed: [TRAIN-08]

# Metrics
duration: 7min
completed: 2026-06-13
---

# Phase 3 Plan 07: Automatic Learning-Rate Selection (TRAIN-08) Summary

**Ported CatBoost's `TAutoLRParamsGuesser` into `cb-train::autolr` — the `const` CPU coefficient table `{A,B,C,D}` keyed by `(target, useBestModel, boostFromAverage)` plus the `exp/log/round` formula — wired into the boosting loop as a pre-train `BoostParams.auto_learning_rate` opt-in (upstream's gate where `learning_rate`/`leaf_estimation_method`/`leaf_estimation_iterations`/`l2_leaf_reg` are all unset); the guessed rate matches upstream `get_all_params()['learning_rate']` at ≤1e-5 (RMSE 0.044808, Logloss 0.005413), and a FIRST FULL end-to-end CPU train→predict cycle runs with the auto-selected rate — closing TRAIN-08 and Phase-3 success criterion 5 (the phase capstone). `cargo test --workspace` is green with no oracle regression.**

## Performance

- **Duration:** ~7 min
- **Completed:** 2026-06-13
- **Tasks:** 2 (Task 1 auto: oracle scenario; Task 2 TDD: RED unit test → GREEN impl + wiring + end-to-end)
- **Files:** 3 created (`autolr.rs`, `autolr_test.rs`, `autolr_e2e_test.rs`) + committed fixtures (`autolr/{rmse,logloss}/config.json`) + 8 modified (`boosting.rs`, `lib.rs`, `gen_fixtures.py`, 6 oracle tests with the new `auto_learning_rate: false` field)

## Accomplishments

- **`cb-train::autolr` (Task 2):** `coefficients(target, use_best_model, boost_from_average) -> Option<[f64;4]>` is the `const` CPU table (8 rows: RMSE × {bestModel,bfa} and Logloss × {bestModel,bfa}) transcribed verbatim from `options_helper.cpp:198-219`; `guess(target, use_best_model, boost_from_average, learn_object_count, iter_count) -> CbResult<f64>` implements `custIter=exp(C·ln iter+D)`, `defIter=exp(C·ln 1000+D)`, `defLR=exp(A·ln N+B)`, `lr=round(min(defLR·custIter/defIter, 0.5), 6)`. `round_to` matches upstream `Round` (`round(x·1e6)/1e6`, half-away-from-zero). Zero `object_count`/`iter_count` returns `CbError::OutOfRange` (T-03-07-01, never `ln(0)`); an Unknown-target key returns `CbError::Degenerate` (matches `NeedToUpdate==false`). No `unwrap`/`expect`/`panic`/indexing (deny-lints hold).
- **Boosting wiring (Task 2):** `BoostParams` gained `auto_learning_rate: bool`. At `train_with_eval_sets` entry, when the flag is set the effective `learning_rate` is guessed once from `(autolr_target_type(loss), use_best_model, boost_from_average, n, iterations)`; a `Degenerate` (no-row) result keeps the explicit rate (`NeedToUpdate==false`), an `OutOfRange` guard propagates. The local `learning_rate` replaces both prior `params.learning_rate` reads (the `model_length` random-strength term and the leaf-value scaling), so the auto-selected rate drives the whole loop. The default (`auto_learning_rate: false`) is behaviourally identical to before — all six existing oracle tests keep their exact rates and stay green.
- **Oracle (Task 1):** `gen_autolr()` trains an RMSE regressor and a Logloss classifier WITHOUT setting any of the four gating params (so `TAutoLRParamsGuesser` fires), with `iterations=500` over `numeric_tiny` (50 objects), and persists `get_all_params()['learning_rate']` (RMSE 0.044808, Logloss 0.005413) plus every keying input (target_type, use_best_model, boost_from_average, learn_object_count, iterations) into `autolr/{rmse,logloss}/config.json`. Python-reachable only (D-11).
- **Parity + capstone (Task 2):** `autolr_test.rs` (7 unit tests) pins the two RESEARCH example coeff rows, the two upstream fixture rates at ≤1e-5, the 0.5 cap, the Unknown-target no-row case, and the zero-count guard. `autolr_e2e_test.rs` runs the FIRST full CPU train→predict cycle with `auto_learning_rate: true` (explicit `learning_rate` set to `NaN` to prove it is unused) for both losses, asserting the cycle runs, a 500-tree non-empty model is produced, all staged predictions are finite, and the applied rate matches the upstream selected rate — Phase-3 success criterion 5.

## Task Commits

1. **Task 1:** `59d8074` (feat) — autolr oracle scenario (`gen_autolr`, RMSE + Logloss, gating params unset, persisted `get_all_params()['learning_rate']` + keying inputs)
2. **Task 2 RED:** `d6036cd` (test) — failing `autolr_test` (coefficients/guess RED stubs return None/Err)
3. **Task 2 GREEN:** `8842119` (feat) — `TAutoLRParamsGuesser` port + `BoostParams.auto_learning_rate` pre-train wiring + end-to-end auto-LR train→predict test; 6 oracle `BoostParams` literals updated

## Files Created/Modified

- `crates/cb-train/src/autolr.rs` — `TargetType` enum, `coefficients()` const CPU table, `round_to`, `guess()`
- `crates/cb-train/src/autolr_test.rs` — 7 unit tests (RESEARCH rows, fixture parity, cap, Unknown, zero-count)
- `crates/cb-train/tests/autolr_e2e_test.rs` — end-to-end auto-LR train→predict cycle (RMSE + Logloss)
- `crates/cb-train/src/boosting.rs` — `BoostParams.auto_learning_rate`, `autolr_target_type()`, pre-train guess resolving the effective `learning_rate`
- `crates/cb-train/src/lib.rs` — `mod autolr` + re-exports (`autolr_guess`, `autolr_coefficients`, `TargetType`)
- `crates/cb-oracle/generator/gen_fixtures.py` — `gen_autolr()` + `AUTOLR` path + `main()` wiring
- `crates/cb-oracle/fixtures/autolr/{rmse,logloss}/config.json` — committed frozen fixtures
- `crates/cb-train/tests/{slice_first,bootstrap,leaf_methods,regularization,overfit,eval_metrics}_oracle_test.rs` — new `auto_learning_rate: false` field on every `BoostParams` literal (behaviour unchanged)

## Decisions Made

- **The four-param activation gate is a single `auto_learning_rate` opt-in flag.** `BoostParams` holds concrete `f64`s for `leaf_estimation_method`/`iterations`/`l2_leaf_reg`, so "all four unset" is a host-caller decision; the flag is the explicit opt-in. When set and the loss is in the table, `learning_rate` is guessed; otherwise the explicit value is used.
- **CPU rows only.** This phase is CPU-only, so `TargetType` has no `TaskType` axis and only the 8 CPU coefficient rows are in the Rust table; the GPU rows (upstream + research) are deferred to Phase 7.
- **`guess()` is fallible.** Zero counts → `CbError::OutOfRange` (never `ln(0)=-inf`, T-03-07-01); Unknown-target key → `CbError::Degenerate`, which the loop treats as "keep explicit rate" (`NeedToUpdate==false`).
- **Oracle uses `iterations=500`** (≠ 1000) so the `custIter/defIter` ratio is genuinely exercised; `learn_object_count=50`.
- **The end-to-end test is operational, not a splits oracle** — it reuses the skeleton borders (same inputs) and asserts the cycle runs + the applied rate == the upstream selected rate; per-tree parity is already locked by the deterministic slice oracles.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing critical] Added `BoostParams.auto_learning_rate` opt-in and updated all six existing oracle `BoostParams` literals**

- **Found during:** Task 2 (wiring the guess into the boosting loop).
- **Issue:** The plan requires the auto-LR guess to fire ONLY when the four gating params are unset, but `BoostParams` carries concrete (non-`Option`) values for `learning_rate`/`l2_leaf_reg` etc., so there was no "unset" signal to gate on. Without an explicit opt-in, the loop could not know whether the caller wanted the auto rate or the literal one.
- **Fix:** Added a single `auto_learning_rate: bool` field expressing the upstream gate decision at the host boundary; the loop guesses only when it is `true` AND the loss is auto-LR eligible. Every existing `BoostParams` construction (6 oracle tests, 7 literals) got `auto_learning_rate: false`, preserving their exact rates and oracle results.
- **Files modified:** `crates/cb-train/src/boosting.rs`, `crates/cb-train/src/lib.rs`, and the 6 oracle test files.
- **Verification:** `cargo test -p cb-train` (39 lib + all oracle integration tests green, prior `#[ignore]`s unchanged); `cargo test --workspace` green.
- **Committed in:** `8842119` (Task 2 GREEN commit).

---

**Total deviations:** 1 auto-fixed (the activation-gate opt-in needed because `BoostParams` has no `Option`-typed gating fields).
**Impact on plan:** Necessary to satisfy the "active ONLY when the four params are unset" acceptance criterion within the existing concrete-field `BoostParams`; no scope creep — the default flag value preserves every prior behaviour exactly.

## Issues Encountered

- **Pre-existing D-08 grep failure (out of scope, unchanged from Plan 06).** `scripts/check-no-raw-float-sum.sh` still flags `crates/cb-train/src/overfit.rs:521` (the W.J. Cody `erf` Horner `fold(0.0, |acc,&c| acc*x + c)`, a multiply-add polynomial, NOT a parity-critical summation), documented in `deferred-items.md` from Plan 06. My new `autolr.rs` involves no float SUM and is grep-clean; `cargo test --workspace` is green. No new D-08 surface introduced.

## Known Stubs

- None. The Plan 02 `autolr.rs` RED stub (`coefficients` → `None`, `guess` → `Err`) was fully replaced by the GREEN implementation in the same plan (TDD cycle); no placeholder remains. The `learning_rate: f64::NAN` sentinel in the e2e test is intentional (proves the explicit rate is unused under auto-LR), not a stub.

## Threat Flags

None — no new network/auth/file/schema surface. Auto-LR is a pure host scalar over trusted in-memory params/counts. T-03-07-01 (ln(0)) is mitigated by the `> 0` guards returning `CbError::OutOfRange`; T-03-07-02 (wrong coefficient row) is mitigated by the exact `(target, useBestModel, boostFromAverage)` keying pinned by the unit test (two RESEARCH rows + two upstream-derived values).

## Next Phase Readiness

- TRAIN-08 automatic learning-rate selection complete and oracle-locked at ≤1e-5; Phase-3 success criterion 5 (first full end-to-end CPU train→predict cycle with the auto-selected rate) is met. This is the FINAL Phase-3 plan — the CPU plain-boosting training core (TRAIN-01..08) is now complete: boosting loop, oblivious trees, four leaf methods, bootstrap/sampling, full regularization, overfitting detection, eval-set logging, and auto-LR, all oracle-locked.
- Phase 4 (Builder API / serialization) inherits `BoostParams.auto_learning_rate` as the pre-train default to surface; Phase 8 (Python API) maps `learning_rate=None` to this flag. The GPU coefficient rows are transcribed upstream and ready for Phase 7. The known `#[ignore]`d multi-tree stochastic oracles and 3 overfit end-to-end oracles remain Phase-4/5 follow-ups (not regressed by this plan).

## Self-Check: PASSED

- Created files verified present: `autolr.rs`, `autolr_test.rs`, `autolr_e2e_test.rs`, `autolr/{rmse,logloss}/config.json` — all FOUND.
- Commits verified in `git log`: `59d8074` (Task 1), `d6036cd` (RED), `8842119` (GREEN) — all FOUND.
- `cargo test --workspace` green; `autolr` unit tests (7) + e2e (2) pass; no oracle regression.

---
*Phase: 03-cpu-training-core-plain-boosting-oblivious-trees*
*Completed: 2026-06-13*
