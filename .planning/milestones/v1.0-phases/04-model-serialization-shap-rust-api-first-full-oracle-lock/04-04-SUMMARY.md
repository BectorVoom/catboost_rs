---
phase: 04-model-serialization-shap-rust-api-first-full-oracle-lock
plan: 04
subsystem: model
tags: [shap, treeshap, feature-importance, fstr, prediction-values-change, interaction, oracle]

# Dependency graph
requires:
  - phase: 04-model-serialization-shap-rust-api-first-full-oracle-lock
    plan: 01
    provides: "canonical cb-model::Model with per-tree leaf_weights (SHAP subtree weights + CalcEffect counts); cb-oracle model_json parser + leaf_weights"
  - phase: 04-model-serialization-shap-rust-api-first-full-oracle-lock
    plan: 02
    provides: "cb-model::predict_raw RawFormulaVal apply path (the local-accuracy invariant target)"
  - phase: 02-data-layer-pool-quantization-reduction
    provides: "cb-core::sum_f64 order-locked reduction (D-08)"
provides:
  - "cb-model::shap_values(model, cols, n_features) -> Vec<Vec<f64>> — regular TreeSHAP per-object [n_features+1] matrix (trailing column = Σ_trees meanValue + bias) with the local-accuracy invariant"
  - "cb-model::prediction_values_change(model) -> Vec<f64> — PredictionValuesChange importance (percent, Σ=100)"
  - "cb-model::interaction(model) -> Vec<(usize, usize, f64)> — pairwise Interaction importance (percent, sorted descending)"
  - "cb-model::FeatureImportanceType enum (PredictionValuesChange, Interaction)"
  - "Oracle locks: SHAP matrix + local accuracy; PredictionValuesChange + Interaction — all vs upstream catboost 1.2.10 feature_importance/*.npy <=1e-5"
affects: [04-05, rust-api]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Verbatim transcription of the upstream TreeSHAP polynomial recursion (ExtendFeaturePath / UnwindFeaturePath / CalcObliviousInternalShapValuesForLeafRecursive / UpdateShapByFeaturePath) + prepared-trees (subtree weights, mean value) — index-heavy but indexing_slicing-clean via checked .get/.get_mut"
    - "Local-accuracy invariant (Σ_columns shap == predict_raw) used as the in-env correctness gate that needs no fixture (D-11)"
    - "Numeric-only combinationClass identity (A3): a split's source feature IS its float feature index — no internal->regular feature remap needed for PVC/Interaction"
    - "feature_importance fixtures share the model_serde/binclf model.json (same boost_from_average=False + ISOLATING_PARAMS + numeric_tiny + seed) — one model.json drives apply, SHAP, and both importances"

key-files:
  created:
    - crates/cb-model/src/shap.rs
    - crates/cb-model/src/fstr.rs
    - crates/cb-model/tests/shap_oracle_test.rs
    - crates/cb-model/tests/fstr_oracle_test.rs
  modified:
    - crates/cb-model/src/lib.rs

key-decisions:
  - "shap_values signature takes the SoA &[Vec<f32>] columns (same input layout as predict_raw) plus an explicit n_features (the SHAP-matrix feature-block width), so the trailing expected-value column lands at the upstream-matching index."
  - "prediction_values_change derives n_features = max split feature index + 1 (matches the upstream importance-vector width for numeric_tiny = 4); plan signature is the no-arg prediction_values_change(model)."
  - "interaction uses an insertion-ordered Vec of (pair, sum) rather than a hash map for deterministic, reproducible iteration; final scores = sum/totalEffect*100, sorted descending (upstream StableSort rbegin/rend)."
  - "The literal token for the deferred loss-change importance is kept OUT of cb-model/src and tests (D-12 scope-guard grep returns empty); the deferral is documented by the D-12 reference and the FeatureImportanceType enum's omission."

requirements-completed: [MODEL-04]
requirements-partial: [MODEL-03]

# Metrics
duration: ~10min
completed: 2026-06-13
---

# Phase 4 Plan 04: TreeSHAP + Feature Importance Summary

**Regular TreeSHAP (the per-object `[n_features+1]` SHAP matrix with the local-accuracy invariant) plus PredictionValuesChange and Interaction feature importance — all transcribed verbatim from upstream catboost 1.2.10 and oracle-locked to the `feature_importance/*.npy` fixtures at <= 1e-5. Completes the explain leg of the train -> serialize -> load -> predict/explain slice.**

## Performance

- **Duration:** ~10 min
- **Completed:** 2026-06-13
- **Tasks:** 2 (both `auto`, `tdd="true"`)
- **Files changed:** 5 (4 created, 1 modified)

## Accomplishments

- **MODEL-04 — regular TreeSHAP.** `cb-model::shap_values(model, cols, n_features) -> Vec<Vec<f64>>`. Per tree, a prepared-trees precompute builds `subtree_weights[depth][node]` bottom-up from `leaf_weights` (leaves = weights, internal = child sum) and `mean_value = (Σ leafValue·leafWeight)/subtree_weights[0][0]` = the `averageTreeApprox` baseline (`shap_prepared_trees.cpp:25-67,177-223`). The Lundberg feature-path machinery — `ExtendFeaturePath`, `UnwindFeaturePath` (both branches on `FuzzyEquals(1+oneFrac, 1+0)`), the per-leaf recursion (`hotCoefficient`/`coldCoefficient` from subtree weights; go-branch `oneFrac=newOnePathsFraction`, skip-branch `oneFrac=0`), and `UpdateShapByFeaturePath` (coefficient `= weightSum·(oneFrac−zeroFrac)·(leafValue−averageTreeApprox)`) — is transcribed verbatim from `shap_values.cpp:44-320`. The per-object output is `[n_features+1]` with the trailing column = `Σ_trees meanValue + bias` (`shap_values.cpp:1030-1055`). **Oracle-locked vs `feature_importance/shap_values.npy` <= 1e-5, and the local-accuracy invariant `Σ_columns shap == predict_raw` holds for every object (D-11 — the strongest check).**
- **MODEL-03 (partial) — PredictionValuesChange.** `cb-model::prediction_values_change(model) -> Vec<f64>` transcribes `CalcEffect` (`feature_str.h:233-270`): per tree, per split-level bit, per leaf pair `(leaf, leaf^(1<<bit))` with `inverted > leaf`, `count1/count2 = leaf_weights`, the `count==0` short-circuit, `avrg=(v1·c1+v2·c2)/(c1+c2)`, `dif=(v1−avrg)²·c1+(v2−avrg)²·c2`, `res[srcFeature] += dif`, then `ConvertToPercents` (Σ=100). **Oracle-locked vs `feature_importance/prediction_values_change.npy` <= 1e-5; sums to 100 in-env.**
- **MODEL-03 (partial) — Interaction.** `cb-model::interaction(model) -> Vec<(usize, usize, f64)>` transcribes `CalcMostInteractingFeatures` (`feature_str.cpp:190-223`) + `CalcFeatureInteraction` (`calc_fstr.cpp:343-414`): per tree, per pair of split levels, `delta = Σ_leaf sign·leafValue` (`sign = (var1 XOR var2)?+1:−1`), accumulate `|delta|` into the sorted source-feature pair (skip equal source features), then `score = sum/totalEffect·100`, sorted descending. **Oracle-locked vs `feature_importance/interaction.npy` <= 1e-5.**
- **FeatureImportanceType enum** (`PredictionValuesChange`, `Interaction`) added; the loss-change importance is intentionally omitted (D-12, out of scope).

## Task Commits

1. **Task 1 (RED): failing TreeSHAP oracle + local-accuracy test** — `75f264d` (test)
2. **Task 1 (GREEN): regular TreeSHAP with prepared-trees + local accuracy** — `5124288` (feat)
3. **Task 2 (RED): failing PredictionValuesChange + Interaction oracle test** — `235df81` (test)
4. **Task 2 (GREEN): PredictionValuesChange + Interaction importance** — `5ca22f8` (feat)

_Both tasks followed TDD: RED (`75f264d` / `235df81`) -> GREEN (`5124288` / `5ca22f8`). No REFACTOR commit was needed — both verbatim transcriptions matched the oracle on the first GREEN run._

## Files Created/Modified

- `crates/cb-model/src/shap.rs` (created) — regular TreeSHAP: `fuzzy_equals`, `Elem` feature-path struct, `extend_feature_path`/`unwind_feature_path`/`update_shap_by_feature_path`, `calc_subtree_weights`/`calc_mean_value`, the `shap_recurse` recursion, `document_leaf_index`, and the public `shap_values`. All folds via `cb_core::sum_f64`; checked `.get`/`.get_mut` only.
- `crates/cb-model/src/fstr.rs` (created) — `FeatureImportanceType` enum, `prediction_values_change` (CalcEffect + ConvertToPercents), `interaction` (pairwise weighted-variance + percent normalization). `count==0`/`total==0` guards (T-04-04-03).
- `crates/cb-model/src/lib.rs` — wired `mod shap; mod fstr;` + re-exports (`shap_values`, `prediction_values_change`, `interaction`, `FeatureImportanceType`).
- `crates/cb-model/tests/shap_oracle_test.rs` (created) — SHAP matrix vs upstream <= 1e-5; local accuracy on the upstream model AND on a hand-built in-env model (no fixture).
- `crates/cb-model/tests/fstr_oracle_test.rs` (created) — PVC + Interaction vs upstream <= 1e-5; PVC Σ=100 in-env.

## Decisions Made

See the `key-decisions` frontmatter. Most load-bearing: (1) `feature_importance/*.npy` were produced by the SAME model as `model_serde/binclf/model.json` (identical `CatBoostClassifier(boost_from_average=False, **ISOLATING_PARAMS)` on `numeric_tiny`, same seed), so that committed `model.json` (which already carries `leaf_weights`) drives all four feature-importance oracle assertions in-env without a new fixture; (2) the numeric-only combinationClass identity (A3) lets PVC/Interaction key directly on the float feature index with no internal->regular remap.

## Deviations from Plan

None within scope — both algorithms were transcribed verbatim from the cited upstream sources and locked on the first GREEN run.

Minor wording adjustment (not a behavior change): the literal token naming the deferred loss-change importance is kept out of `crates/cb-model/src` and the test files so the D-12 scope-guard grep (`grep -n 'LossFunctionChange' crates/cb-model/src` returns nothing) is satisfied; the deferral is documented via the `D-12` reference and the `FeatureImportanceType` enum's deliberate omission.

## Issues Encountered

- **Disk-space limit (environment, not a code defect):** the box is ~100% full (<1 GB free). Per the environment constraints, NO `cargo test --workspace` / `cargo check --workspace --tests` was run (those recompile polars-core at the link step and fail with "No space left on device"). Verified per-crate instead: the full cb-model test suite (`cargo test -p cb-model`) runs all seven test binaries — apply 3, cbm 9, fstr 3, json 6, predict 5, shap 3, lib 0 (29/29 pass) — and `cargo clippy -p cb-model --lib` is clean. cb-oracle is not a polars consumer, so the cb-model test profile links without polars.

## Deferred Issues

None within scope.

## Known Stubs

None. Both SHAP and feature importance are wired to real model data (the upstream `model_serde/binclf/model.json`, including its `leaf_weights`) and oracle-locked against upstream catboost 1.2.10 fixtures; no placeholder/empty data sources.

## Threat Flags

None. All three STRIDE entries from the plan's threat model are mitigated as specified:
- **T-04-04-01** (index-heavy SHAP recursion): every subtree-node / feature-path access is checked `.get`/`.get_mut`; the recursion bounds match the `2^depth` tree shape — no OOB, `indexing_slicing` deny satisfied.
- **T-04-04-02** (div-by-`subtree_weights[0][0]==0`): `calc_mean_value` guards the divisor with `fuzzy_equals(1+total, 1+0)` and the recursion's `hot/cold` coefficients guard the parent weight — no NaN leak.
- **T-04-04-03** (PVC div-by-`(c1+c2)==0` / interaction div-by-`total==0`): the `count1==0 || count2==0` short-circuit (verbatim upstream) and the `total==0` guards in `convert_to_percents` / `interaction` prevent div-by-zero.
No NEW network/auth/file surface was introduced.

## Regression Guard

New `pub fn shap_values`, `pub fn prediction_values_change`, `pub fn interaction`, and a NEW `pub enum FeatureImportanceType` — all additive. `FeatureImportanceType` has no external `match` consumers (referenced only by its own module + `lib.rs`). All seven cb-model test binaries recompile and pass (29/29); no shared enum/type or existing `pub` signature changed, so no Wave-2-style non-exhaustive-match regression is possible.

## Next Phase Readiness

- The explain leg (SHAP + PredictionValuesChange + Interaction) is in place for Plan 05 (the Rust Builder facade can expose `shap_values` / feature importances directly).
- SHAP interaction values (MODEL-05) and the loss-change importance (D-12) remain deferred to Phase 6 — `interaction` here is the pairwise weighted-variance importance, NOT SHAP interaction values.

## Self-Check: PASSED

- Created files verified present: `crates/cb-model/src/shap.rs`, `crates/cb-model/src/fstr.rs`, `crates/cb-model/tests/shap_oracle_test.rs`, `crates/cb-model/tests/fstr_oracle_test.rs`.
- Commits verified present: `75f264d` (SHAP RED), `5124288` (SHAP GREEN), `235df81` (fstr RED), `5ca22f8` (fstr GREEN).
- Tests green: cb-model shap 3/3, fstr 3/3 (full suite 29/29); `cargo test -p cb-model shap` and `cargo test -p cb-model fstr` both exit 0.
- Grep gates pass: shap.rs/fstr.rs use `leaf_weights`/`subtree`/`count`; no raw float sum (all via `sum_f64`); no `unwrap()`/`expect(`; `grep 'LossFunctionChange' crates/cb-model/src` returns nothing (D-12); `cargo clippy -p cb-model --lib` clean.

---
*Phase: 04-model-serialization-shap-rust-api-first-full-oracle-lock*
*Completed: 2026-06-13*
