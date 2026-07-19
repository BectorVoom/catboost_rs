# Evidence Ledger — FSTR-03 Partial Dependence

Research report consumed: `.planning/research/unimplemented-parity-research.md`
(Research Agent, 2026-07-12). Read in full before planning.

## Requirement / roadmap
- `[VERIFIED: LOCAL .planning/REQUIREMENTS.md:31]` — FSTR-03: "User can compute
  partial-dependence for one or two features (staged-apply sweep)".
- `[VERIFIED: LOCAL .planning/REQUIREMENTS.md:94-96]` — FSTR-01/02/03 → Phase 18, Pending.
- `[VERIFIED: LOCAL .planning/ROADMAP.md:126-131]` — Phase 18 Success Criterion 3:
  match "upstream's exact grid/quantization + averaging convention".

## Upstream behavior (Context7)
- `[VERIFIED: CONTEXT7 /catboost/catboost "plot_partial_dependence"]` — averaging
  formula `f_{x_S}(x_S) = (1/n) Σ_i f(x_S, x_C^{(i)})`; one feature → 1-D curve,
  two features → 2-D heatmap; grids relate to float-split border structure.
  Source: `catboost/tutorials/model_analysis/plot_partial_dependence_tutorial.ipynb`.
- **Not derivable:** exact x-grid transform (borders vs midpoints vs endpoints)
  and the figure-object data-extraction path — flagged `[UNVERIFIED]` in SPEC §9,
  resolved empirically at fixture-gen / T4.

## Absence proofs (justify "unimplemented")
- `[VERIFIED: LOCAL grep -rn 'partial_dependence|PartialDependence' crates/]` → 0 hits.
- `[VERIFIED: LOCAL grep -rin 'partial' catboost-master/catboost/{libs,python-package}]` → 0 hits
  (vendored snapshot predates the utility).
- `[VERIFIED: LOCAL python3 -c 'import catboost']` → ModuleNotFoundError (env has no
  catboost → fixtures generated offline & committed, per existing pattern).

## Rust seams (CodeGraph / Read) — re-verified 2026-07-16 for the hardening pass
- `[VERIFIED: CODEGRAPH crates/cb-model/src/apply.rs:370]` — `predict_raw(model,
  &[Vec<f32>]) -> Vec<f64>` (thin wrapper over `predict_raw_cat(.., &[])`; SoA
  columns indexed by float-feature index, RawFormulaVal, pure-Rust CPU path).
- `[VERIFIED: CODEGRAPH crates/cb-model/src/apply.rs:404-407]` — `predict_raw_cat`
  gathers each object row via `col.get(obj).copied().unwrap_or(f32::NAN)`: a
  **short/missing column silently reads NaN** (no error). This is the hazard the
  new PDP-05 `MalformedColumns` guard closes.
- `[VERIFIED: CODEGRAPH crates/cb-model/src/model.rs:272-313]` — `cb_model::Model`
  struct: `pub float_feature_borders: Vec<Vec<f64>>` at **:293** (grid source,
  confirmed a public field), `pub ctr_data: Option<CtrData>`, `pub approx_dimension:
  usize`, `pub class_to_label: Vec<f64>`. **No flat-feature / feature-kind map
  exists** → a categorical target cannot be expressed or detected via the
  float-index API → the earlier `UnsupportedFeatureKind` arm is unimplementable and
  was removed (SPEC §4 change-note).
- `[VERIFIED: CODEGRAPH crates/catboost-rs/src/model.rs:51-53]` — `n_float_features()`
  = `float_feature_borders.len()` (the canonical float-feature count); and
  `feature_columns()` (:59-66) is the existing precedent that validates
  `pool.n_float_features() == model.n_float_features()` before apply — the
  facade-level analogue of the new `MalformedColumns` boundary guard.
- `[VERIFIED: CODEGRAPH crates/cb-core/src/reduction.rs:32]` — `sum_f64(&[f64]) ->
  f64`, the sanctioned sequential D-08 fold (never `.sum()`).
- `[VERIFIED: LOCAL crates/cb-model/src/fstr.rs:49,55,123,288,490]` — existing fstr
  entry points (numeric-only) + `cb_core::sum_f64` D-08 fold discipline.
- `[VERIFIED: LOCAL crates/cb-model/src/lib.rs:14-22]` — module declarations
  (`mod apply/fstr/model/...`); new `mod partial_dependence;` slots here.

## Oracle harness
- `[VERIFIED: CODEGRAPH crates/cb-oracle/src/compare.rs:46]` — `assert_abs_close(exp:
  &[f64], act: &[f64], tol: f64) -> Result<(), OracleError>` — **returns a `Result`,
  never panics**; `!(diff <= tol)` also rejects NaN/Inf (NaN-safe gate). `:84` —
  `compare_stage` pinned at `1e-5` (D-12). Oracle tests handle the `Result`
  (`.expect(...)` under the test-only lint allow, or `-> Result` test fns).
- `[VERIFIED: LOCAL crates/cb-model/tests/advanced_fstr_oracle_test.rs:18-49]` —
  the oracle integration-test harness the new `partial_dependence_oracle_test.rs`
  mirrors: top `#![allow(clippy::unwrap_used, expect_used, panic, indexing_slicing)]`,
  `const TOL: f64 = 1e-5`, `fixture(rel)` path helper, `ndarray_npy::read_npy` +
  `cb_oracle::load_f64_vec` loaders.
- `[VERIFIED: LOCAL crates/cb-oracle/fixtures/advanced_fstr/config.json]` — fixture
  config shape (`artifacts`, `catboost_version:"1.2.10"`, `input_dataset`, seeds,
  `thread_count:1`, `note`).
- `[VERIFIED: LOCAL crates/cb-oracle/fixtures/advanced_fstr/gen_fixtures.py:1-70]` —
  generator recipe (load `inputs/numeric_tiny`, train pinned model, dump
  `get_feature_importance(...)` truth to `.npy`).
- `[VERIFIED: LOCAL crates/cb-oracle/generator/gen_fixtures.py:120,145-175]` —
  `CATBOOST_VERSION="1.2.10"`, ISOLATING_PARAMS (deterministic, thread_count=1).
- `[VERIFIED: LOCAL crates/cb-model/tests/advanced_fstr_oracle_test.rs:100-170]` —
  oracle integration-test pattern (load `.cbm`, compute Rust importance, load
  `.npy`, assert ≤ TOL).

## Constraints
- `[VERIFIED: LOCAL Cargo.toml:10-14]` — workspace denies `unwrap_used`,
  `expect_used`, `panic`, `indexing_slicing`.
- `[VERIFIED: LOCAL CLAUDE.md]` — source/test separation (no `mod tests` in
  production `.rs`); tests in `_test.rs` / `tests/`; thiserror(lib)+anyhow(app).
- `[VERIFIED: LOCAL memory catboost-rs-preexisting-test-failures.md]` — env-red
  suites to ignore (cb-backend MLIR, cb-train monotone, catboost-rs-py py3.14 link).

## Layering / impact
- `[VERIFIED: LOCAL .planning/research/unimplemented-parity-research.md:47,182]` —
  cb-model → cb-core only for this slice; no cb-train/cb-backend edge → no CubeCL
  feature-unification risk; impact = `local`.

## PageIndex (corrected 2026-07-16)
- `[VERIFIED: PAGEINDEX get_folder_structure + browse_documents]` — the SPEC **is
  indexed**: folder `catboost_rs` (id `cmrhcxbtm000104jr3i5jzm0m`) contains one
  document, `SPEC.md`, status `completed`, created 2026-07-12, described as the
  single/two-feature partial-dependence spec. This **corrects** the prior
  `[INFERRED]` claim that no corpus applied.
- `[VERIFIED: TOOL process_document schema]` — the write path (`process_document`)
  ingests PDFs/files as **new** documents (params: `url`, `folder_id`; no `doc_id`
  overwrite), so it cannot upsert the existing Markdown in place — re-processing
  would create a duplicate. Planner therefore left PageIndex untouched (no
  duplicate, no `remove_document`) and recorded a **pending human re-index** of the
  hardened `SPEC.md` into folder `catboost_rs` (SPEC §10).

## Git-recovered planning docs (canonical-revision caveat)
- `[VERIFIED: LOCAL git ls-files .planning]` — `.planning/REQUIREMENTS.md`,
  `.planning/ROADMAP.md`, `.planning/research/` are **not in the working tree**;
  the requirement/roadmap citations were recovered from commit `a82289c`. Confirm
  the canonical revision before flipping the FSTR-03 requirement checkbox (T7
  bookkeeping).
