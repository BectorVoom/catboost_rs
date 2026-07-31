# Phase Research: Wire categorical-feature (cat_features/CTR) routing through `CatBoostBuilder`

## Research Summary
- **Goal**: `CatBoostBuilder::fit` (crates/catboost-rs/src/builder.rs) must branch to the categorical-aware `cb_train::train_cat` entrypoint when the `Pool` carries categorical columns, and must surface the CTR config knobs (`cat_features` selection is implicit — it's Pool-carried; `one_hot_max_size`, `max_ctr_complexity`, `simple_ctr`, `combinations_ctr`, `counter_calc_method`) as builder setters, then plumb them through `catboost-rs-py/src/params.rs` (move 6 kwargs from `KNOWN_NOT_YET` to `IMPLEMENTED`).
- **Recommended approach**: `fit()` should check `pool.cat_features().is_empty()`; float-only path stays byte-identical (call `train`); non-empty path calls `train_cat`, attaches the returned `BakedCtrData` via `cb_model::CtrData::from_baked` + `Model::with_ctr_data`, exactly mirroring the already-proven pattern in `crates/cb-train/tests/tensor_ctr_e2e_oracle_test.rs`.
- **Most important constraint**: this phase's scope, as literally requested, is "wire … through the `CatBoostBuilder` facade" (fit/train side). But **`catboost_rs::Model::predict` / `predict_with` / `predict_proba` currently NEVER call `cb_model::predict_raw_cat`** — they call the float-only `predict_raw` unconditionally (`crates/catboost-rs/src/model.rs:120-137`). A model trained via a future cat-aware `fit()` would carry `ctr_data`, but the facade's own `predict()` would silently ignore it (or, if `feature_columns()`'s width check doesn't reject it, produce WRONG predictions rather than erroring). **The planner must decide whether wiring `predict()`/`predict_with()` CTR-awareness is in-scope for this phase or an explicit hand-off to a follow-up phase** — training a model the facade cannot correctly score back is not a safe deliverable on its own.
- **Highest-risk findings**: (1) the predict-side gap above; (2) `catboost-rs-py`'s Arrow/Polars ingestion path explicitly REJECTS non-empty `cat_features` today (`ingest_py.rs:338-342`) — only the Pandas/NumPy-object path supports it, and categorical columns there must be TRAILING; (3) `CatBoostBuilder::boost_params()` currently pins ALL CTR-related `BoostParams` fields to upstream defaults with an explicit code comment that "the numeric facade path bakes no CTR table … inert here" (builder.rs:292-303) — every one of those pinnings needs an accompanying builder setter, not just a `cat_features`-presence branch.

## Phase Requirements

### In Scope (as requested by the calling agent)
- `CatBoostBuilder::fit` branches between `cb_train::train` (float-only, byte-identical for the no-cat-features case) and `cb_train::train_cat` (categorical-aware) based on whether `pool.cat_features()` is non-empty.
- `CatBoostBuilder` gains setters (or equivalent) for `one_hot_max_size`, `max_ctr_complexity`, `simple_ctr`, `combinations_ctr`, `counter_calc_method` (all currently hardcoded in `boost_params()`).
- `crates/catboost-rs-py/src/params.rs` moves `cat_features`, `one_hot_max_size`, `max_ctr_complexity`, `simple_ctr`, `combinations_ctr`, `counter_calc_method` from `KNOWN_NOT_YET` (i.e., absent from `IMPLEMENTED`) into `IMPLEMENTED`, with type extraction + range/enum validation matching the existing pattern (`check_range`, `parse_*` helpers).
- A new oracle test through the **public Python API** (`catboost_rs.CatBoostClassifier`/`Regressor`, `Pool(..., cat_features=[...])`, `.fit()`), asserting ≤1e-5 against a fresh upstream catboost 1.2.10 fixture (the `.venv` at repo root has real `catboost==1.2.10` installed, matching `CATBOOST_VERSION` in `crates/cb-oracle/generator/gen_fixtures.py:130`).

### Acceptance Criteria (inferred; not found as a locked spec — see Open Questions)
- No existing oracle test regresses (`cargo test -p catboost-rs -p catboost-rs-py`, `pytest` under `.venv`).
- The float-only path (`pool.cat_features()` empty) remains **byte-identical** to today's `train()` call — this is a hard invariant the codebase repeatedly calls "D-04 no-regression" elsewhere.
- `params.rs::VOCABULARY`/`IMPLEMENTED` coverage test (referenced in code comments as "the param-coverage test") must still assert every upstream kwarg is known; moving 6 names from `KnownNotYet` to `Implemented` needs matching test updates.

### Out of Scope (explicitly, per the calling agent's task framing)
- No production code changes were made by this research pass — evidence-gathering only.
- Whether `Model::predict`/`shap_values` become CTR-aware is NOT explicitly requested, but is a hard blocking dependency — see Research Summary and Open Questions.

### Open or Conflicting Requirements
- No SPEC.md / phase document for this exact feature was found in `.planning/phases/` or TreeFinder — this phase has no locked, pre-existing spec. `.planning/phases/24-ctr-split-search-correctness/` covers CTR *training correctness* (ORD-06/ORD-07 bug fixes to the already-existing `train_cat`/CTR engine itself), NOT the `CatBoostBuilder` facade wiring. `.planning/phases/23-ctr-model-loading/` covers CTR `.cbm` file load/save, again not facade wiring for training. This means the planner is defining a NEW phase/spec from scratch, informed by this research.
- Whether `predict()`/`predict_proba()`/`shap_values()`/`feature_importance_with_data` need CTR-awareness in this same phase, or a follow-up phase, is unresolved and should be an explicit decision point in planning.

## Project Constraints
- `unwrap()` strictly prohibited in production code (CLAUDE.md); `cb-train`/`cb-model`/`cb-data` consistently follow this (checked `.get()`/typed `CbError` patterns throughout files read).
- Source/test separation is mandatory — no `#[cfg(test)] mod tests` embedded in production files; tests live in sibling `_test.rs`/`tests/*_oracle_test.rs` files. Confirmed pattern: `crates/cb-train/tests/tensor_ctr_e2e_oracle_test.rs` is a separate file `[VERIFIED: LOCAL crates/cb-train/tests/tensor_ctr_e2e_oracle_test.rs]`.
- Oracle parity bar: ≤ 1e-5 vs upstream CatBoost — used pervasively (`compare_stage`, `SCORE_BOUND`) `[VERIFIED: LOCAL crates/cb-train/tests/tensor_ctr_e2e_oracle_test.rs:261]`.
- Builder pattern (`#[must_use]`, consuming `mut self -> Self` setters) is the established `CatBoostBuilder` convention — every existing setter in `builder.rs:119-253` follows this exact shape `[VERIFIED: LOCAL crates/catboost-rs/src/builder.rs:119-253]`.
- `catboost-rs-py/src/params.rs`'s "honesty policy" (D-05/D-07, threat T-08-05): a kwarg is Implemented / KnownNotYet(rejected) / Unknown(rejected+suggestion) — **never silently ignored** `[VERIFIED: LOCAL crates/catboost-rs-py/src/params.rs:1-18]`.
- GPU tests restricted to `rocm` feature only (CLAUDE.md) — not directly relevant here since `train_cat` is CPU-backend-generic (`R: Runtime`) but noted as a project-wide rule.

## Current Project Architecture

### Relevant subsystems and boundaries
- `cb-data::Pool` (`crates/cb-data/src/pool.rs`) — the canonical dataset container. Re-exported verbatim as `catboost_rs::Pool` (`crates/catboost-rs/src/lib.rs:66`, `pub use cb_data::Pool;`) — there is NO separate `catboost-rs`-level Pool type; the facade IS `cb_data::Pool` `[VERIFIED: CODEGRAPH crates/catboost-rs/src/lib.rs:66]`.
- `cb_train::train` (float-only, `boosting.rs:1950`) vs `cb_train::train_cat` (`boosting.rs:2149`) vs the shared internal `train_inner` (`boosting.rs:2263`, takes `cat_columns: &[Vec<String>]` and a `RankingData<'a>`). `train_cat` is a thin wrapper: `train_inner(..., cat_columns, ..., &[], None, RankingData::default())` — no eval sets, no ranking, matching `train`'s wrapper shape `[VERIFIED: CODEGRAPH crates/cb-train/src/boosting.rs:2149-2172]`.
- `cb_model::Model` — canonical model; `with_ctr_data(CtrData)` attaches baked CTR tables; `cb_model::CtrData::from_baked(&BakedCtrData)` converts the trainer's output into inference-ready form `[VERIFIED: LOCAL crates/cb-model/src/ctr_data.rs:288,313; CODEGRAPH tensor_ctr_e2e_oracle_test.rs:249-251]`.
- `cb_model::predict_raw_cat(&Model, &[Vec<f32>], &[Vec<String>])` — the CTR-aware apply entrypoint, already implemented and exported from `cb_model::lib.rs` `[VERIFIED: LOCAL crates/cb-model/src/lib.rs:30]`. **`catboost_rs::Model::predict`/`predict_with` never call this** — only the plain `predict_raw` (float-only) `[VERIFIED: LOCAL crates/catboost-rs/src/model.rs:16-22,120-137]`.

### Existing data/control flow
1. Python (`Pool(x, y, cat_features=[...])`, `crates/catboost-rs-py/src/pool.rs:59-98`) → `ingest_to_owned(py, data, label, cat_features.as_deref())` (`ingest_py.rs`) → for a Pandas/NumPy-object frame, categorical columns are read as raw strings into `OwnedColumns.with_cat_features(...)` (`ingest_py.rs:302`); for Arrow/Polars, a non-empty `cat_features` is REJECTED with `CatBoostValueError` (`ingest_py.rs:338-342`) `[VERIFIED: LOCAL crates/catboost-rs-py/src/ingest_py.rs]`.
2. `OwnedColumns::into_pool()` → `cb_data::Pool` — `Pool.cat_features() -> &[Vec<String>]` already carries this data end-to-end `[VERIFIED: CODEGRAPH crates/cb-data/src/pool.rs:152-156]`.
3. `CatBoostBuilder::fit(&self, pool: &Pool)` (`builder.rs:334-402`) — reads ONLY `pool.float_features()` (via `rayon::join` for `feature_values`+`feature_borders`), then calls `train(&backend, &feature_values, &feature_borders, pool.label(), pool.weights(), &params, None)`. `pool.cat_features()` is **never read** in `fit()` today `[VERIFIED: LOCAL crates/catboost-rs/src/builder.rs:334-402]`.
4. Contrast: `Model::feature_importance_with_data` (a DIFFERENT facade method, not `fit`/`predict`) already DOES read `pool.cat_features()` at line 371 to call `prediction_values_change_with_data(&self.inner, &columns, &cat_columns)` — so there is PRIOR ART in this exact crate for a facade method reading `pool.cat_features()` and passing it to a CTR-aware cb-model function `[VERIFIED: LOCAL crates/catboost-rs/src/model.rs:369-378]`.

### Existing reusable implementations (must not be reimplemented)
- `cb_train::train_cat` — the entire categorical-aware training entrypoint already exists, tested, oracle-verified.
- `cb_model::CtrData::from_baked`, `Model::with_ctr_data`, `predict_raw_cat` — the entire CTR-aware inference path already exists.
- `cb_train::{simple_ctr_default, simple_ctr_priors_default, counter_calc_method_default, max_ctr_complexity_default, combinations_ctr_default, combinations_ctr_priors_default, one_hot_max_size_default}` — all default-value functions already exist and are the values `boost_params()` currently pins to; the new setters should default to these SAME functions when unset (`#[must_use] pub fn new()` pattern), preserving current behavior when the user does not override.
- `cb_train::ECtrType`, `cb_train::CounterCalcMethod`, `cb_train::TProjection`, `cb_train::BakedCtrData` — all already `pub use`d from `cb_train::lib.rs` (`crates/cb-train/src/lib.rs:46,49,80,101`) `[VERIFIED: LOCAL crates/cb-train/src/lib.rs]`. NOT currently re-exported from `catboost_rs::lib.rs` — will need new `pub use` lines there (alongside the existing `pub use cb_train::EBootstrapType;` at line 61) for the setters' parameter types to be nameable by Python-binding code and downstream users.

### Current conventions and patterns
- Builder setters: `#[must_use] pub fn field_name(mut self, field_name: T) -> Self { self.field_name = field_name; self }` — see every setter in `builder.rs:119-253`.
- `boost_params()` maps 1:1 builder fields → `BoostParams` fields, with an explanatory comment on every field still pinned to a default (builder.rs:284-317) — the planner should follow this same "map with comment" discipline, removing the "inert here" comments once a setter exists.
- `params.rs::make_builder` pattern: `if let Some(v) = get_with_aliases::<T>(params, py, "name")? { check_range(...)?; builder = builder.setter(v); }` for scalars/enums; string enums go through a local `parse_*` helper (`parse_loss`, `parse_score_function`, `parse_bootstrap_type`, `parse_leaf_method`) returning `PyResult<T>` — the new `simple_ctr`/`combinations_ctr` (ECtrType) and `counter_calc_method` (CounterCalcMethod) kwargs will need analogous `parse_ctr_type`/`parse_counter_calc_method` helpers.
- `cat_features` itself is NOT a `BoostParams` scalar — it's implicit in `pool.cat_features()` being non-empty; the Python `cat_features` kwarg is *already* consumed at `Pool()`/ingestion time (`pool.rs`, `ingest_py.rs`), NOT at `fit()`-kwarg time. Moving `"cat_features"` from `KNOWN_NOT_YET` to `IMPLEMENTED` in `params.rs` is therefore semantically different from the other 5 — it likely means: accept `cat_features=[...]` as an `Estimator.fit(x, y, cat_features=...)` kwarg (sklearn-native convenience, matching upstream `CatBoostClassifier.fit(X, y, cat_features=...)`) that gets forwarded into `data_to_pool`'s ingestion call, NOT a `CatBoostBuilder` setter. **This needs planner attention — verify the current `estimator.rs`/`data_to_pool` signature can even accept a `cat_features` override at `fit()` time versus only at `Pool()` construction time.**

## Standard Stack
No new external dependency is needed — this phase is 100% internal wiring across already-existing, already-tested internal crates (`cb-train`, `cb-model`, `cb-data`, `catboost-rs`, `catboost-rs-py`). All "library" work is Rust internal API surfacing.

| Component | Version/Location | Existing/Proposed | Purpose | Notes |
|---|---|---|---|---|
| `cb_train::train_cat` | `crates/cb-train/src/boosting.rs:2149` | Existing | Cat-aware training entrypoint | Already oracle-tested via `tensor_ctr_e2e_oracle_test.rs`, `plain_ctr_oracle_test.rs`, `ordered_ctr_oracle_test.rs`, `one_hot_oracle_test.rs` |
| `cb_model::predict_raw_cat` / `CtrData` | `crates/cb-model/src/{apply.rs,ctr_data.rs}` | Existing | Cat-aware inference | Not yet wired into `catboost_rs::Model::predict` |
| `catboost` (Python) | 1.2.10, in `.venv` at repo root `[VERIFIED: local .venv/bin/python3 -c "import catboost; print(catboost.__version__)" -> 1.2.10]` | Existing (oracle-only, test dependency) | Ground-truth fixture generation | Matches `CATBOOST_VERSION = "1.2.10"` pinned in `crates/cb-oracle/generator/gen_fixtures.py:130` |

## Dependency Analysis
- **Direct**: none new. All symbols needed (`train_cat`, `CtrData`, `predict_raw_cat`, `ECtrType`, `CounterCalcMethod`, `TProjection`, `BakedCtrData`) are already compiled, public, and exported from their owning crates.
- **Transitive/peer**: `catboost-rs-py`'s PyO3 extraction (`FromPyObject`) is the only new-surface concern — new kwarg types (`Vec<usize>` for `cat_features` at fit-kwarg-level if pursued, `String`→`ECtrType`/`CounterCalcMethod` enum parsing) reuse the existing `get_with_aliases::<T>` + `parse_*` machinery; no new PyO3 API needed.
- **Runtime/build/system**: none. No CUDA/GPU/CubeCL involvement — `train_cat` is `R: Runtime`-generic exactly like `train`, so it already compiles under every existing backend feature (`cpu`/`wgpu`/`cuda`/`rocm`) with no additional feature-gating work.
- **Compatibility/migration**: `BoostParams` is a plain struct with named fields (not `#[non_exhaustive]`) — adding no new fields (all CTR fields already exist on `BoostParams`), so no struct-shape migration is needed; only the VALUES the builder feeds in change from hardcoded defaults to builder-state-driven values.
- **Additions/removals**: none needed.

## Recommended Architecture and Implementation Pattern

### Prescribed approach
1. **`CatBoostBuilder`**: add fields `one_hot_max_size: u32`, `max_ctr_complexity: usize`, `simple_ctr: ECtrType`, `simple_ctr_priors: Vec<f64>` (or leave priors pinned — see Open Questions), `combinations_ctr: ECtrType`, `combinations_ctr_priors: Vec<f64>`, `counter_calc_method: CounterCalcMethod`, each defaulted in `new()` to the SAME `*_default()` function `boost_params()` currently calls directly (byte-identical default behavior). Add one `#[must_use] pub fn <name>(mut self, ...) -> Self` setter per field, following the existing style exactly.
2. **`boost_params()`**: replace each `xxx_default()` call with `self.xxx` (now sourced from builder state, defaulting to the same value).
3. **`fit()`**: branch on `pool.cat_features().is_empty()`:
   - Empty ⇒ EXACT existing code path (`train(...)`), byte-identical output — this is the D-04 no-regression invariant that must be tested explicitly (e.g., re-run `builder_oracle_test.rs`/`model_serde` fixtures unchanged).
   - Non-empty ⇒ call `train_cat(&backend, &feature_values, &feature_borders, pool.cat_features(), pool.label(), pool.weights(), &params, None)`, then `cb_model::Model::from_trained(&trained, feature_borders).with_ctr_data(cb_model::CtrData::from_baked(&baked_ctr_data))`, wrapped via `Model::from_canonical`. This is a straight lift of the pattern already proven in `tensor_ctr_e2e_oracle_test.rs:232-250`.
4. **`catboost_rs::lib.rs`**: add `pub use cb_train::{ECtrType, CounterCalcMethod};` (or similar) so downstream (Python bindings, direct Rust users) can name the new setter parameter types.
5. **`catboost-rs-py/src/params.rs`**: move the 6 names into `IMPLEMENTED`; add `parse_ctr_type(&str) -> PyResult<ECtrType>` and `parse_counter_calc_method(&str) -> PyResult<CounterCalcMethod>` helpers mirroring `parse_bootstrap_type`/`parse_leaf_method`; wire each into `make_builder`. For `one_hot_max_size`/`max_ctr_complexity` (numeric), add `check_range` calls analogous to `border_count`. `cat_features` needs a SEPARATE design decision (see Open Questions) since it is not a `BoostParams` scalar.
6. **Predict-side decision** (must be resolved before/at planning time, not deferred silently): either (a) extend `Model::predict`/`predict_with` to branch to `predict_raw_cat` when `self.inner.ctr_data.is_some()` and thread `pool.cat_features()` through `feature_columns`-adjacent plumbing, mirroring the `feature_importance_with_data` prior art, OR (b) explicitly scope this phase to `fit()`-only and file a follow-up phase for predict-side CTR wiring, with `Model::predict` on a CTR model either erroring clearly or documented as a known gap. Silently shipping a `fit()` that produces CTR models the facade's own `predict()` cannot correctly score is a footgun inconsistent with the project's "never silently wrong" ethos (see `params.rs`'s own "Honesty policy" doctrine).

### Component responsibilities
- `CatBoostBuilder` (catboost-rs): CTR/cat config surface + routing decision (float vs cat path) at `fit()`.
- `cb_train::train_cat`/`train_inner`: all actual CTR training math (untouched, reused verbatim).
- `cb_model::CtrData`/`predict_raw_cat`: all actual CTR inference math (reused verbatim; new caller sites only).
- `catboost-rs-py::params.rs`: kwarg vocabulary/validation/type-mapping only — no training logic.

### Integration points
- `crates/catboost-rs/src/builder.rs` (`CatBoostBuilder` struct + impl, `boost_params()`, `fit()`)
- `crates/catboost-rs/src/lib.rs` (new `pub use` re-exports)
- `crates/catboost-rs-py/src/params.rs` (`IMPLEMENTED`, `VOCABULARY` unchanged, `make_builder`, new `parse_*` helpers)
- Possibly `crates/catboost-rs-py/src/estimator.rs` (`data_to_pool` / a `fit(x, y, cat_features=...)` sklearn-style kwarg — needs verification, see Open Questions)
- Possibly `crates/catboost-rs/src/model.rs` (`Model::predict`/`predict_with`) — see predict-side decision above.

### Data and control flow
`Pool` (already cat-carrying) → `CatBoostBuilder::fit` → branch on `pool.cat_features().is_empty()` → `train` or `train_cat` → `Model::from_trained` (+`with_ctr_data` on the cat path) → facade `Model`. No new data shapes; `cat_columns: &[Vec<String>]` is exactly `pool.cat_features()` with no transformation needed at the call site (verified against the `train_cat` signature and the `Pool::cat_features()` return type — both `&[Vec<String>]`).

### Error, security, and failure behavior
- No new error variants appear strictly required: `train_cat` returns the SAME `CbResult<(Model, BakedCtrData)>` shape `train` returns for `Model` alone; `CatBoostError`'s existing `#[from] CbError` / `Train` variant should already cover it (verify at plan time — not independently confirmed in this pass, see Open Questions).
- Range/enum validation for the new numeric/enum kwargs at the Python boundary should reuse `check_range`/`parse_*` exactly like existing params (WR-05 discipline) — do not skip domain validation for the new kwargs.
- Interleaved/non-trailing categorical columns are ALREADY rejected at ingestion time (`ingest_py.rs:240-263`) — this phase does not need to re-validate that; it is upstream of `Pool` construction.
- Arrow/Polars + non-empty `cat_features` is ALREADY rejected at ingestion (`ingest_py.rs:338-342`) — also upstream of this phase's scope; no new validation needed here, just be aware categorical training can currently only be reached via the Pandas/NumPy-object ingestion path.

## Project Impact Scope

### Must Change
- `crates/catboost-rs/src/builder.rs` — `CatBoostBuilder` struct (new fields), `new()` (new defaults), `boost_params()` (read builder state not hardcoded defaults), new setter methods, `fit()` (branch train/train_cat + attach CtrData). Reason: this is the literal, explicitly-requested change. Downstream: `crates/catboost-rs/src/cv.rs:407` and `crates/catboost-rs/src/grid_search.rs:405,515` call `builder.fit(pool)` — they get the new behavior automatically with NO signature change needed (both already pass `&Pool` through unmodified) `[VERIFIED: CODEGRAPH crates/catboost-rs/src/builder.rs "CatBoostBuilder ... 16 callers"]`.
- `crates/catboost-rs/src/lib.rs` — add re-exports for the new setter parameter types (`ECtrType`, `CounterCalcMethod`, and confirm `EBootstrapType`-style visibility is sufficient or whether `TProjection`/`BakedCtrData` also need exposure).
- `crates/catboost-rs-py/src/params.rs` — `IMPLEMENTED` list (+6 entries), new `parse_*` helpers, `make_builder` wiring. Reason: explicitly requested; currently these 6 names sit in `VOCABULARY` but NOT `IMPLEMENTED`, so `validate_params` rejects them today with a "parity gap" `CatBoostParameterError` `[VERIFIED: LOCAL crates/catboost-rs-py/src/params.rs:42-188]`.

### May Change
- `crates/catboost-rs-py/src/estimator.rs` — IF `cat_features` is to be accepted as a `.fit(x, y, cat_features=...)` kwarg (sklearn convention) rather than only at `Pool()` construction time, `data_to_pool`'s `ingest_to_owned(py, x, y, None)` call at line ~243 would need a `cat_features` parameter threaded through from the estimator's `fit` signature. Needs planner decision (see Open Questions).
- `crates/catboost-rs/src/model.rs` — `Model::predict`/`predict_with`/`predict_proba`/`shap_values` — IF predict-side CTR-awareness is pulled into this phase's scope (recommended but not explicitly requested — see Research Summary risk #1).
- `crates/catboost-rs/src/error.rs` — IF a new `CatBoostError` variant is needed to distinguish a CTR-training-specific failure mode from the existing generic `Train` variant (not confirmed necessary; verify against `CbError`'s variant set at plan time).

### Verification Only
- `crates/cb-train/src/boosting.rs` (`train_cat`, `train_inner`) — reused, unmodified.
- `crates/cb-model/src/{apply.rs,ctr_data.rs}` (`predict_raw_cat`, `CtrData`) — reused, unmodified.
- `crates/cb-data/src/pool.rs`, `crates/cb-data/src/ingest/owned.rs` — `Pool.cat_features()`/`OwnedColumns.with_cat_features()` already correct end-to-end; only need to be READ from, not changed.
- `crates/catboost-rs-py/src/ingest_py.rs`, `crates/catboost-rs-py/src/pool.rs` — the `cat_features=[...]` Pool-construction-time kwarg already works today; verify it continues to work once `fit()` starts consuming `pool.cat_features()`.
- Existing test suites: `crates/cb-train/tests/{tensor_ctr_oracle_test.rs,tensor_ctr_e2e_oracle_test.rs,plain_ctr_oracle_test.rs,ordered_ctr_oracle_test.rs,one_hot_oracle_test.rs}`, `crates/catboost-rs/tests/{builder_oracle_test.rs,cv_oracle_test.rs,grid_search_oracle_test.rs}`, `crates/catboost-rs-py/tests/{test_ingestion.py,test_params.py,test_oracle_parity.py}` — must all continue passing (regression gate); `builder_oracle_test.rs`/`cv_oracle_test.rs`/`grid_search_oracle_test.rs` specifically exercise `CatBoostBuilder::fit` and must be re-run to confirm the float-only branch stays byte-identical.

### Explicitly Out of Scope
- Any change to `train_cat`'s internal correctness — Phase 24 (`24-ctr-split-search-correctness`) is a separate, already-tracked, in-progress bug-fix effort (ORD-06/ORD-07 combination-CTR candidate-gating bug) `[VERIFIED: LOCAL .planning/phases/24-ctr-split-search-correctness/combination-ctr-level-gating; .planning/plans/unimplemented-features-survey/research.md:33]`. This phase must NOT attempt to fix that bug; it only routes to the existing (partially-buggy-at-the-margins) engine.
- `.cbm`/`.json` CTR model save/load — already shipped (Phase 23) `[VERIFIED: LOCAL .planning/plans/unimplemented-features-survey/research.md:119,146]`.
- Arrow/Polars categorical ingestion support — explicitly rejected today by design (`ingest_py.rs:320-342`); out of scope for this phase unless the planner deliberately expands scope.

## Do Not Hand-Roll
- Do NOT reimplement CTR computation, online/final CTR accumulation, or combined-projection hashing — `cb_train::{train_cat, TProjection, materialize_ctr_feature, bake_ctr_table}` already do this, oracle-verified.
- Do NOT reimplement CTR-aware inference/apply — `cb_model::{predict_raw_cat, CtrData::from_baked}` already do this.
- Do NOT reimplement kwarg validation/alias-resolution/Levenshtein-suggestion machinery in `params.rs` — reuse `get_with_aliases`, `check_range`, `resolve_alias`, `closest_match` exactly as the existing 14 `IMPLEMENTED` params do.
- Do NOT reimplement Pandas/NumPy categorical ingestion — `ingest_py.rs`'s existing trailing-categorical-column contract is already correct and tested; this phase only needs to READ `pool.cat_features()`, not touch ingestion.

## Common Pitfalls and Risks

| # | Trigger | Consequence | Prevention | Verification |
|---|---|---|---|---|
| 1 | Forgetting the predict-side gap (Model::predict never calls predict_raw_cat) | A CTR model is trainable via the new `fit()` but `predict()` either silently mis-scores it (if `feature_columns()`'s width check doesn't catch a categorical-only model, e.g. `n_float_features()==0` matching `pool.n_float_features()==0` trivially) or errors unhelpfully | Explicit planner decision + test: fit a cat model then immediately call `.predict()` through the facade and assert either correct CTR-aware output or a clear typed error | Add an assertion in the new oracle test that `model.predict(&cat_pool)` is exercised, not just training |
| 2 | Treating `cat_features` kwarg identically to the other 5 (BoostParams-scalar) kwargs in `params.rs` | `cat_features` is NOT a `BoostParams` field — it's consumed at Pool-construction/ingestion time, not fit-kwarg time; naively adding it to `IMPLEMENTED`+`make_builder` with no matching `CatBoostBuilder` setter will not compile / will be a no-op | Design `cat_features`-as-fit-kwarg as a SEPARATE code path (threading into `data_to_pool`/`ingest_to_owned`), not a `CatBoostBuilder` setter | Grep `BoostParams` fields — confirm no `cat_features` field exists; confirm `make_builder`'s `CatBoostBuilder` has no matching setter needed for it |
| 3 | Breaking the float-only byte-identical invariant | Changing `fit()`'s float path in the process of adding the branch could silently alter `feature_values`/`feature_borders` computation order or defaults | Keep the `pool.cat_features().is_empty()` TRUE branch's code IDENTICAL (same `rayon::join`, same `train(...)` call signature) to today; only add a new FALSE branch | Re-run `crates/catboost-rs/tests/builder_oracle_test.rs`, `cv_oracle_test.rs`, `grid_search_oracle_test.rs` unmodified and confirm all still pass |
| 4 | `simple_ctr_priors`/`combinations_ctr_priors` left pinned while `simple_ctr`/`combinations_ctr` (the TYPE) becomes a setter | Setting `simple_ctr = ECtrType::Counter` while `simple_ctr_priors` stays at the `Borders`-tuned default priors could produce a nonsensical CTR (wrong prior shape/semantics for the new type) | Either expose priors setters in lockstep with the type setters, or validate/derive matching default priors per selected CTR type inside the builder | Add a builder-level unit test setting a non-default `simple_ctr` and asserting priors are consistent (or documented as still-pinned with a clear doc-comment caveat) |
| 5 | Arrow/Polars ingestion + `cat_features` | A user might expect `cat_features` support across every ingestion source once `fit()` wires categorical training; Arrow/Polars still hard-rejects it | Document explicitly (docstring/error message) that categorical training is Pandas/NumPy-object-only for now; do not silently change Arrow/Polars behavior in this phase | `crates/catboost-rs-py/src/ingest_py.rs:338-342` unchanged; existing rejection test (if any) still passes |
| 6 | `params.rs`'s param-coverage introspection test going stale | The code comments reference "the param-coverage test" asserting every upstream kwarg is known via `_param_status` — moving 6 names from KnownNotYet to Implemented changes that test's expected classification | Locate and update the specific test asserting `_param_status` results (search `test_params.py` / a Rust test) as part of the same change | `pytest crates/catboost-rs-py/tests/test_params.py` (or wherever this lives) after the change |
| 7 | New oracle fixture generation nondeterminism | Project memory (`ctr-model-loading.md`) explicitly documents that "catboost quantization is run-to-run nondeterministic so CTR fixtures are frozen" — a fresh fixture generated for this phase's new Python-API oracle test must be committed/frozen, not regenerated per-CI-run | Generate once with a pinned `random_seed`, `thread_count=1` (per `gen_fixtures.py`'s `ISOLATING_PARAMS` convention), commit the fixture files under `crates/cb-oracle/fixtures/` | Compare `git diff` after two independent local regenerations with the same seed to confirm stability before committing |

## Testing and Verification Strategy

### Unit tests
- `crates/catboost-rs/src/builder.rs`-adjacent (if a `builder_test.rs` exists — not directly located in this pass; check `crates/catboost-rs/src/*_test.rs` naming convention per CLAUDE.md) covering new setters' default-preservation (calling `.simple_ctr(default)` should equal not calling it at all) and the `fit()` branch selection logic.

### Integration/contract tests
- Extend or add a Rust integration test analogous to `crates/catboost-rs/tests/builder_oracle_test.rs`, but using a categorical fixture (reuse `tensor_ctr_e2e/` fixture data under `crates/cb-oracle/fixtures/`, OR generate a fresh Pandas-ingestion-shaped fixture) through the PUBLIC `catboost_rs::CatBoostBuilder`/`Pool` facade (not `cb_train` directly), asserting ≤1e-5 vs upstream.

### End-to-end/regression tests
- New Python-level pytest (e.g., `crates/catboost-rs-py/tests/test_categorical.py` or extend `test_oracle_parity.py`) driving `catboost_rs.Pool(X, y, cat_features=[...])` → `catboost_rs.CatBoostClassifier(...).fit(pool)` → `.predict(...)`, compared against a real `catboost==1.2.10` (`.venv`) reference run with matching hyperparameters (mirror the `ISOLATING_PARAMS` discipline from `gen_fixtures.py`: `thread_count=1`, fixed `random_seed`, `verbose=False`).
- Re-run existing suites for regression: `crates/catboost-rs/tests/{builder_oracle_test.rs,cv_oracle_test.rs,grid_search_oracle_test.rs}`, `crates/cb-train/tests/{tensor_ctr_oracle_test.rs,tensor_ctr_e2e_oracle_test.rs,plain_ctr_oracle_test.rs,ordered_ctr_oracle_test.rs,one_hot_oracle_test.rs}`.

### Migration/data checks
- Not applicable — no schema/serialization format change; `BoostParams`/`Model`/`CtrData` shapes are unchanged, only NEW call sites reading already-existing fields.

### Security/performance/operational checks
- No new attack surface beyond standard kwarg-value validation (range/enum) already required by WR-05 discipline in `params.rs`.
- No performance regression expected on the float-only path (branch is a single `is_empty()` check before existing code executes unchanged).

### Exact project commands
- `cargo test -p catboost-rs -p catboost-rs-py -p cb-train -p cb-model` (adjust to actual workspace crate names; confirm via `cargo metadata` or `Cargo.toml` `[workspace] members` at plan time — not independently re-verified in this pass).
- `.venv/bin/pytest crates/catboost-rs-py/tests/` (the `.venv` at repo root has `catboost==1.2.10` and presumably `pytest` installed — confirm `pytest` presence with `.venv/bin/python3 -m pytest --version` at plan time; not directly re-verified here beyond confirming `catboost` itself is importable).
- `cargo clippy --workspace` (project convention per Clippy-defaults note in CLAUDE.md; `params.rs` comments reference "clippy-not-build lint gate" per project memory).

## Planning Guidance
- **Suggested ordering**: (1) `CatBoostBuilder` setters + `boost_params()` wiring + `fit()` branch (pure Rust, testable in isolation via `cb-train`-fixture-backed Rust integration test) → (2) `catboost_rs::lib.rs` re-exports → (3) `params.rs` vocabulary/parsing wiring (depends on step 2's re-exports for setter parameter types) → (4) resolve and implement the predict-side decision (either wire `Model::predict`/`predict_with` for CTR-awareness, or explicitly document/gate it as a follow-up) → (5) new Python-level oracle test + fixture generation/freezing.
- **Task dependencies**: `params.rs` work depends on `builder.rs` setters existing (needs concrete setter names/types to call). The Python oracle test depends on ALL of the above being functional end-to-end, including whichever predict-side decision is made in step 4 — a training-only oracle test (fit + inspect `Model.as_canonical().ctr_data`) is possible without step 4, but a full `fit → predict → compare` Python oracle test is NOT possible without step 4 being resolved.
- **Decisions the planner must preserve**: the float-only path must remain byte-identical (existing D-04 no-regression discipline is a repo-wide invariant, not specific to this phase); the "never silently ignore a kwarg" honesty policy in `params.rs` must be preserved for any newly-introduced kwarg semantics (esp. `cat_features`-as-fit-kwarg, if pursued).
- **Items requiring a spike or explicit user/planner decision before implementation**:
  1. Is predict-side (`Model::predict`, `shap_values`, etc.) CTR-awareness IN this phase's scope, or deferred? (Recommended: at minimum, `Model::predict`/`predict_with` should be in scope, since otherwise `fit()` produces facade-unusable models — this is a strong recommendation from this research, not a locked decision.)
  2. Should `cat_features` be exposed as a `.fit(x, y, cat_features=[...])` sklearn-style kwarg (in addition to the existing `Pool(..., cat_features=[...])` constructor kwarg), or is the existing `Pool`-level kwarg sufficient to satisfy "wire cat_features … through the facade"? This determines whether `estimator.rs`/`data_to_pool` needs changes.
  3. Should `simple_ctr_priors`/`combinations_ctr_priors` also become setters (full CTR config exposure) or stay pinned while only the `ECtrType` selector becomes configurable (partial exposure, risking pitfall #4 above)?
  4. Confirm whether a NEW `CatBoostError` variant is needed for cat-training-specific failures, or whether the existing `#[from] CbError` blanket coverage suffices — not independently verified in this research pass.

## Open Questions
- Does `CatBoostError` already have adequate `#[from] CbError` coverage for whatever new failure modes `train_cat` can surface that `train` cannot (e.g., a perfect-hash `u32::MAX` overflow mentioned in `train_cat`'s doc comment at `boosting.rs:2146-2147`)? Not independently verified — `crates/catboost-rs/src/error.rs` was not read in this pass.
- Where exactly does the "param-coverage test" referenced in `params.rs`'s comments live (`_param_status` consumer)? Not located in this pass; needed so the planner's params.rs change updates the right test.
- Is `pytest` installed in `.venv` (confirmed `catboost` is; `pytest` presence assumed from the existing `crates/catboost-rs-py/tests/__pycache__/*.pytest-*.pyc` artifacts implying a working pytest loop was run against SOME environment, possibly `.venv`, possibly a different one no longer present) — the previously-memory-referenced `.venv-py8` directory does NOT exist in the current repo checkout; only `.venv` was found. This is a correction to stale session memory, not a project fact — verify the ACTIVE test-running venv at plan time.
- Does a `crates/catboost-rs/src/builder_test.rs` (unit-test file, per the project's `_test.rs` suffix convention) already exist for `CatBoostBuilder`? Not directly confirmed in this pass (only `crates/catboost-rs/tests/builder_oracle_test.rs`, an integration test, was read).

## Sources
- Project documents inspected via local filesystem (TreeFinder's index for this workspace was SPARSE — only 3 documents indexed workspace-wide: `snapshot-resume/{PLAN,SPEC}.md` and `xgboost-rust-rewrite/SPEC.md`; none relevant to this phase. `search_hierarchy`/`document_list` confirmed no indexed CTR/cat/builder documents — local `.planning/` filesystem inspection was used instead as the authoritative source for prior-phase context) `[VERIFIED: TREE_FINDER document_list — 3 total documents, none topically relevant]`.
- `[VERIFIED: LOCAL .planning/plans/unimplemented-features-survey/research.md]` — full survey (446 lines) confirming Phase 23 (CTR `.cbm` load/save) DONE, Phase 24 (CTR split-search correctness, ORD-06/07) IN PROGRESS, CTR training itself DONE (with the known Phase-24 bug), and (separately) SHAP/predict-side being float-only with NO CTR handling — corroborates this research's predict-side-gap finding independently.
- `[VERIFIED: LOCAL .planning/phases/24-ctr-split-search-correctness/]`, `[VERIFIED: LOCAL .planning/phases/23-ctr-model-loading/]` — directory listings confirming these are distinct, already-tracked efforts, not this phase's scope.
- `[VERIFIED: CODEGRAPH]` queries: "CatBoostBuilder fit builder.rs boost_params", "train_cat train boosting.rs cb-train categorical", "Pool cat_features ingest_py pool.rs categorical columns", "BakedCtrData train_inner CtrConfig simple_ctr combinations_ctr materialize_ctr_feature bake_ctr_table cb-train lib.rs exports", "struct BoostParams simple_ctr combinations_ctr counter_calc_method one_hot_max_size max_ctr_complexity default functions" — each returned verbatim, line-numbered source with blast-radius/caller data.
- `[VERIFIED: LOCAL]` file reads: `crates/catboost-rs/src/builder.rs` (full `CatBoostBuilder` impl + `fit()`), `crates/catboost-rs/src/model.rs` (lines 1-400), `crates/catboost-rs-py/src/params.rs` (full file, 514 lines), `crates/cb-train/tests/tensor_ctr_e2e_oracle_test.rs` (full file), `crates/catboost-rs/tests/builder_oracle_test.rs` (partial), `crates/catboost-rs-py/tests/test_ingestion.py` (partial), `crates/catboost-rs-py/src/{estimator.rs,ingest_py.rs,pool.rs}` (grep + targeted reads).
- `[VERIFIED: bash grep/find]`: `crates/cb-train/src/lib.rs` public export lines for `ECtrType`/`CounterCalcMethod`/`train_cat`/`TProjection`/`BakedCtrData`; `crates/cb-model/src/lib.rs` export lines for `predict_raw_cat`/`CtrData`; `crates/catboost-rs/src/lib.rs` re-export lines (confirming `ECtrType`/`CounterCalcMethod` NOT yet re-exported there); test file listing under `crates/cb-train/tests/` (46 oracle test files enumerated).
- `[VERIFIED: local command]` `.venv/bin/python3 -c "import catboost; print(catboost.__version__)"` → `1.2.10`, confirming the real upstream oracle reference package is installed and importable in the repo's `.venv`.

## Confidence Assessment
- **HIGH**: `train_cat`'s existence/signature/location (`boosting.rs:2149`); `fit()`'s current float-only behavior and exact line range (`builder.rs:334-402`); `Pool.cat_features()` already carrying data end-to-end from Pandas/NumPy ingestion through to `cb_data::Pool`; `params.rs`'s exact 3-state registry and the 6 target kwargs' current `KnownNotYet` status (all directly read from source, not inferred); `Model::predict` never calling `predict_raw_cat` (directly read); Phase 23/24 being separate, already-tracked efforts (directly read from `.planning/`); `.venv` having real `catboost==1.2.10` installed (directly executed and verified).
- **MEDIUM**: the exact shape a `cat_features`-as-fit-kwarg design should take (inferred from upstream CatBoost's own `fit(X, y, cat_features=...)` convention referenced in `params.rs`'s docstring, not independently confirmed against `estimator.rs`'s current `fit` signature in full); whether a new `CatBoostError` variant is needed (not fully traced through `error.rs`); the exact commands for `cargo test`/`pytest` invocation (inferred from repo conventions, not run in this pass beyond the single `.venv` catboost-import check).
- **LOW**: whether `.venv` (vs. some other environment) is the one CI/the maintainer actually uses for Python oracle tests — the `.venv-py8` name referenced in prior session memory does not exist in this checkout, and this discrepancy was not resolved (flagged as an Open Question rather than presented as fact); the exact location of the "param-coverage test" mentioned in `params.rs` comments (not located).
