---
phase: 04-model-serialization-shap-rust-api-first-full-oracle-lock
plan: 05
subsystem: rust-api
tags: [facade, builder, public-error, predict, serialize, shap, fstr, oracle, rust-api]

# Dependency graph
requires:
  - phase: 04-model-serialization-shap-rust-api-first-full-oracle-lock
    plan: 01
    provides: "canonical cb-model::Model {oblivious_trees, bias, float_feature_borders, leaf_weights}; Model::from_trained"
  - phase: 04-model-serialization-shap-rust-api-first-full-oracle-lock
    plan: 02
    provides: "cb-model::predict_raw apply path + PredictionType/apply_prediction_type; Loss CrossEntropy/Focal"
  - phase: 04-model-serialization-shap-rust-api-first-full-oracle-lock
    plan: 03
    provides: "cb-model::{save_cbm,load_cbm,save_json,load_json}; typed cb-model::ModelError"
  - phase: 04-model-serialization-shap-rust-api-first-full-oracle-lock
    plan: 04
    provides: "cb-model::{shap_values, prediction_values_change, interaction, FeatureImportanceType}"
  - phase: 03-cpu-training-core-plain-boosting-oblivious-trees
    provides: "cb-train::train plain boosting loop + BoostParams; cb-backend::CpuBackend runtime; cb-oracle compare_stage harness"
  - phase: 02-data-layer-pool-quantization-reduction
    provides: "cb-data::Pool + OwnedColumns/IngestSource seam; select_borders_greedy_logsum binarizer"
provides:
  - "catboost-rs::CatBoostBuilder (D-05): new() + chained #[must_use] setters + fit(&pool) -> Result<Model, CatBoostError>; loss selects clf vs regression"
  - "catboost-rs::Model facade (D-06/D-07): predict/predict_proba/predict_with (enum core), save_cbm/load_cbm/save_json/load_json, shap_values, feature_importance"
  - "catboost-rs::CatBoostError (D-08/RAPI-02): thiserror; #[from] CbError + ModelError + io::Error; Deserialize/SchemaVersion/FeatureMismatch; not Clone/PartialEq"
  - "Re-exports: PredictionType, FeatureImportanceType, Loss, LeafMethod, EBootstrapType, Pool, OwnedColumns, IngestSource"
  - "End-to-end oracle lock: full numeric binclf + regression train->serialize->load->predict cycle through the PUBLIC API matches upstream catboost 1.2.10 <=1e-5 (ROADMAP Phase-4 criterion 5)"
affects: [05, 06, python-bindings, rust-api]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Published facade wraps the five internal cb-* crates; anyhow structurally absent (D-14/D-15) — thiserror-only library"
    - "Builder maps its fields onto cb_train::BoostParams; fit computes per-feature borders from the pool via select_borders_greedy_logsum, trains over CpuBackend, lifts into cb_model::Model::from_trained"
    - "Public CatBoostError copies the cb-oracle/cb-model io::Error #[from] tradeoff (drops Clone/PartialEq) — internal CbError keeps them"
    - "Facade predict/shap check pool float-feature count vs model and return CatBoostError::FeatureMismatch (T-04-05-02) — no OOB across the public boundary"

key-files:
  created:
    - crates/catboost-rs/src/builder.rs
    - crates/catboost-rs/src/model.rs
    - crates/catboost-rs/src/error.rs
    - crates/catboost-rs/src/error_test.rs
    - crates/catboost-rs/tests/builder_oracle_test.rs
  modified:
    - crates/catboost-rs/src/lib.rs
    - crates/catboost-rs/Cargo.toml
    - Cargo.lock

key-decisions:
  - "Builder fit computes borders from the pool (select_borders_greedy_logsum); for numeric_tiny those borders reproduce upstream's border selection EXACTLY, so the full public-API cycle is oracle-locked <=1e-5 unconditionally (no #[ignore] needed) for BOTH binclf and regression."
  - "predict_with(pool, PredictionType) is the D-06 enum core; predict() = RawFormulaVal, predict_proba() = Probability shorthands."
  - "feature_importance(FeatureImportanceType) -> Vec<(usize,usize,f64)> unifies PVC (per-feature percent, second index = same feature) and Interaction (sorted pairs) under one return shape."
  - "Builder defaults mirror catboost 1.2.10 (depth=6, lr=0.03, l2=3.0, iterations=1000, no sampling); the oracle test overrides them to the model_serde config.json params (depth=2, 5 iters, lr=0.1)."
  - "CatBoostError does NOT derive Clone/PartialEq (io::Error #[from] blocks them, D-08); documented in the module comment."

requirements-completed: [RAPI-01, RAPI-02]

# Metrics
duration: ~10min
completed: 2026-06-13
---

# Phase 4 Plan 05: Public catboost-rs Facade Summary

**The published `catboost-rs` facade closing the phase's first full vertical slice: a single unified `CatBoostBuilder` (loss selects clf vs regression), a cohesive `Model` carrying predict/predict_proba/save/load/shap/feature_importance, and the public typed `CatBoostError` (thiserror, `#[from] CbError`) — with the full numeric binclf + regression train -> serialize -> load -> predict cycle oracle-locked to upstream catboost 1.2.10 <= 1e-5 through the public API ONLY (ROADMAP Phase-4 criterion 5).**

## Performance

- **Duration:** ~10 min
- **Completed:** 2026-06-13
- **Tasks:** 2 (both `auto`)
- **Files changed:** 8 (5 created, 3 modified)

## Accomplishments

- **RAPI-01 — unified Builder + Model facade (D-05/D-06/D-07).** `CatBoostBuilder::new()` + chained `#[must_use]` setters (`loss`, `iterations`, `depth`, `learning_rate`, `auto_learning_rate`, `l2_leaf_reg`, `random_strength`, `boost_from_average`, `leaf_method`, `bootstrap_type`, `subsample`, `bagging_temperature`, `random_seed`, `border_count`) + `fit(&pool) -> Result<Model, CatBoostError>`. `fit` narrows the pool's SoA float columns to `f32`, computes each feature's quantization borders via `cb_data::select_borders_greedy_logsum`, runs `cb_train::train` over `cb_backend::CpuBackend`, and lifts the result into the canonical `cb_model::Model` (carrying `leaf_weights` + `float_feature_borders`). The `loss` field selects the task (regression on the raw label vs classification on the binary label) — no typed `Classifier`/`Regressor` split (D-05). The facade `Model` exposes the D-06 enum-core `predict_with(pool, PredictionType)` plus `predict()` (RawFormulaVal) / `predict_proba()` (Probability) shorthands, and the D-07 `save_cbm`/`load_cbm`/`save_json`/`load_json`/`shap_values`/`feature_importance(type)` methods, each delegating to `cb-model`. A wrong-width pool returns `CatBoostError::FeatureMismatch` (T-04-05-02) — no out-of-bounds access crosses the public boundary.
- **RAPI-02 — public typed `CatBoostError` (D-08).** `#[derive(Debug, thiserror::Error)] pub enum CatBoostError` with `Train(#[from] cb_core::CbError)`, `Model(#[from] cb_model::ModelError)`, `Io(#[from] std::io::Error)`, plus the facade's own boundary variants `Deserialize(String)` / `SchemaVersion(String)` / `FeatureMismatch(String)`. It does NOT derive `Clone`/`PartialEq` — the `io::Error` `#[from]` arm blocks them — exactly the tradeoff `cb-oracle`/`cb-model` already accepted (documented in the module comment). `anyhow` is structurally absent (D-14/D-15): the published facade is a `thiserror`-only library. Five unit asserts lock the variant set and the `#[from]` conversions (`CbError`/`io::Error`/`ModelError` -> `CatBoostError`, including `?`-propagation).
- **ROADMAP Phase-4 criterion 5 — end-to-end oracle lock.** `builder_oracle_test.rs` drives the FULL public-API slice for BOTH a numeric binary-classification (`Loss::Logloss`, boost_from_average=false) and a numeric regression (`Loss::Rmse`, boost_from_average=true) fixture: build a `Pool` via `OwnedColumns` -> `CatBoostBuilder::fit` -> `save_cbm`/`load_cbm` and `save_json`/`load_json` -> `predict`. It asserts (1) Rust<->Rust round-trip determinism (reload reproduces the fit model's predictions exactly via `compare_stage(Stage::Predictions, ...)`) AND (2) the reloaded model's predictions match the committed upstream catboost 1.2.10 `predictions.npy` <= 1e-5. The builder's fit-from-pool borders reproduce upstream's border selection for `numeric_tiny`, so the upstream oracle leg runs UNCONDITIONALLY (no `#[ignore]` was needed). The prediction path uses the published facade ONLY — no `cb-train`/`cb-model` import (only the `cb-oracle` test harness for `compare_stage`/npy loading).

## Task Commits

1. **Task 1: CatBoostBuilder + public CatBoostError + Model facade methods** — `6099f4a` (feat)
2. **Task 2: end-to-end binclf + regression train->serialize->load->predict oracle** — `93720a6` (test)

## Files Created/Modified

- `crates/catboost-rs/src/builder.rs` (created) — `CatBoostBuilder` (D-05): defaults + chained setters + `boost_params()` mapping + `fit(&pool)` (border computation, train over `CpuBackend`, lift into `cb_model::Model`).
- `crates/catboost-rs/src/model.rs` (created) — facade `Model` wrapping `cb_model::Model`: `predict_with`/`predict`/`predict_proba`, `shap_values`, `feature_importance`, `save_cbm`/`load_cbm`/`save_json`/`load_json`, `feature_columns` (FeatureMismatch guard).
- `crates/catboost-rs/src/error.rs` (created) — public `CatBoostError` (thiserror; `#[from]` CbError + ModelError + io::Error; Deserialize/SchemaVersion/FeatureMismatch; no Clone/PartialEq).
- `crates/catboost-rs/src/error_test.rs` (created) — 5 unit asserts (RAPI-02 variants + `#[from]` conversions + `?`-propagation).
- `crates/catboost-rs/src/lib.rs` (modified) — module wiring + re-exports (`CatBoostBuilder`, `Model`, `CatBoostError`, `PredictionType`, `FeatureImportanceType`, `Loss`, `LeafMethod`, `EBootstrapType`, `Pool`, `OwnedColumns`, `IngestSource`); `#[cfg(test)] mod error_test`.
- `crates/catboost-rs/Cargo.toml` (modified) — wired cb-core/cb-data/cb-compute/cb-backend/cb-train/cb-model + thiserror deps; cb-oracle/ndarray/ndarray-npy dev-deps; anyhow kept absent.
- `crates/catboost-rs/tests/builder_oracle_test.rs` (created) — end-to-end binclf + regression public-API oracle (Rust<->Rust round-trip determinism + upstream <= 1e-5).
- `Cargo.lock` (modified) — new dependency edges for the facade crate.

## Decisions Made

See the `key-decisions` frontmatter. Most load-bearing: the builder's fit-from-pool borders (`select_borders_greedy_logsum`) reproduce upstream's border selection for `numeric_tiny` exactly, so the upstream <= 1e-5 oracle leg runs unconditionally for BOTH binclf and regression rather than being `#[ignore]`'d — the plan anticipated a possible border divergence and allowed an `#[ignore]` fallback, but it was not needed.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 — Missing critical functionality] FeatureMismatch guard on predict/shap (T-04-05-02 mitigation)**
- **Found during:** Task 1 (Model facade)
- **Issue:** The threat model assigns `mitigate` to T-04-05-02 (predict on a pool with the wrong feature count). Without a check, a too-narrow pool would either silently mispredict or read out-of-range columns.
- **Fix:** Added `Model::feature_columns(pool)` which checks `pool.n_float_features() == model.n_float_features()` and returns `CatBoostError::FeatureMismatch` otherwise; all predict / shap entry points funnel through it.
- **Files modified:** `crates/catboost-rs/src/model.rs`, `crates/catboost-rs/src/error.rs`
- **Verification:** `cargo build -p catboost-rs` + `cargo clippy -p catboost-rs --lib` clean; the typed variant is asserted in `error_test.rs`.
- **Committed in:** `6099f4a`

**Total deviations:** 1 auto-fixed (Rule 2, a planned threat-register mitigation). No architectural changes, no scope creep — `FeatureMismatch` is already a named variant in the plan's artifact spec.

## Issues Encountered

- **Disk-full at LINK (environment, not a code defect).** Per the environment constraints the box is ~100% full (<1.5 GB free). The first `cargo test -p catboost-rs --test builder_oracle_test` failed at the LINK step with mold reporting "Disk full?" / "Bus error" (NOT a compile error). Reclaimed space by `rm -rf target/debug/incremental` and deleting stale `target/debug/deps/.mold-*` partial-link temp files (the largest reclaim, ~1 GB; no `cargo clean`), then the link and tests succeeded. NO `cargo test --workspace` / `cargo check --workspace --tests` was run (those recompile polars at link and disk-fail); verified per-crate instead: `cargo test -p catboost-rs` runs the lib unit tests (5/5) + the builder oracle integration test (2/2) = 7/7, and `cargo clippy -p catboost-rs --lib` is clean.

## Deferred Issues

None within scope. The disk-blocked `cargo test --workspace` final sanity is tracked under the same environment caveat as Plans 02-04; the in-scope public-API surface is fully covered by the per-crate runs above.

## Known Stubs

None. The Builder is wired to the real `cb-train` boosting loop and `cb-model` (de)serialize/apply/explain paths; the facade carries no placeholder/empty data sources. The end-to-end oracle lock against upstream catboost 1.2.10 (<= 1e-5) proves the whole slice is real.

## Threat Flags

None. All three STRIDE entries from the plan's threat model are mitigated as specified:
- **T-04-05-01** (malformed file via `load_cbm`/`load_json`): the facade delegates to the Plan-03 validated deserializers and maps `cb_model::ModelError` -> `CatBoostError::Model` via `#[from]` — no panic crosses the public boundary.
- **T-04-05-02** (predict on a wrong-width pool): the `feature_columns` guard returns `CatBoostError::FeatureMismatch` (typed) before any column access — no OOB.
- **T-04-05-03** (repudiation): accepted (numerical library; no auth/audit surface).
No NEW network/auth/file surface was introduced beyond the file I/O already in `cb-model`.

## Regression Guard

New `pub` symbols only (`CatBoostBuilder`, facade `Model`, `CatBoostError`) plus re-exports in the new `catboost-rs` facade crate. No shared enum/type or existing `pub` signature was changed (`Loss`/`PredictionType`/`FeatureImportanceType`/`EBootstrapType` are CONSUMED, never altered). A workspace grep for `match` on the re-exported enums in other `crates/*/tests` and `crates/*/src` found no external consumers, so no Wave-2-style non-exhaustive-match regression is possible. catboost-rs builds + all its test binaries pass.

## Next Phase Readiness

- The first full vertical slice (train -> serialize -> load -> predict/explain) is closed through the published `catboost-rs` API and oracle-locked end-to-end <= 1e-5. Phase 5/6 extend the facade (CTR / text / embedding features, multiclass, uncertainty prediction types, loss-change importance) and the Python (PyO3/maturin) bindings layer over this exact surface.
- The Builder's param surface mirrors the in-scope `BoostParams`; later phases add the remaining knobs (eval set / early stopping / use_best_model are present in `BoostParams` but not yet surfaced on the facade — a deliberate first-slice scope boundary).

## Self-Check: PASSED

- Created files verified present: `crates/catboost-rs/src/builder.rs`, `crates/catboost-rs/src/model.rs`, `crates/catboost-rs/src/error.rs`, `crates/catboost-rs/src/error_test.rs`, `crates/catboost-rs/tests/builder_oracle_test.rs`.
- Commits verified present: `6099f4a` (Task 1), `93720a6` (Task 2).
- Tests green: catboost-rs lib unit 5/5 (error_test), builder oracle integration 2/2 (binclf + regression, Rust<->Rust round-trip + upstream <= 1e-5); `cargo clippy -p catboost-rs --lib` clean.
- Acceptance grep gates pass: Builder `new`/`fit`/`struct`; facade `predict`/`predict_proba`/`save_cbm`/`load_cbm`/`save_json`/`load_json`/`shap_values`/`feature_importance`; `CatBoostError` variants + `#[from] CbError`; `compare_stage` in the oracle test; no `anyhow`; no internal cb-train/cb-model import on the test prediction path.

---
*Phase: 04-model-serialization-shap-rust-api-first-full-oracle-lock*
*Completed: 2026-06-13*
