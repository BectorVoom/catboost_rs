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

## Rust seams (CodeGraph / Read)
- `[VERIFIED: CODEGRAPH crates/cb-model/src/apply.rs:370]` — `predict_raw(model,
  &[Vec<f32>]) -> Vec<f64>` (SoA columns, RawFormulaVal, pure-Rust CPU path).
- `[VERIFIED: CODEGRAPH crates/cb-model/src/model.rs:293]` — `Model.float_feature_borders:
  Vec<Vec<f64>>` (per-feature ascending borders → grid source).
- `[VERIFIED: LOCAL crates/cb-model/src/fstr.rs:49,55,123,288,490]` — existing fstr
  entry points + `cb_core::sum_f64` D-08 fold discipline.
- `[VERIFIED: LOCAL crates/cb-model/src/lib.rs:14-22]` — module declarations
  (`mod apply/fstr/model/...`); new `mod partial_dependence;` slots here.

## Oracle harness
- `[VERIFIED: CODEGRAPH crates/cb-oracle/src/compare.rs:46]` — `assert_abs_close(exp,
  act, tol)`; `:84` — `compare_stage` pinned at `1e-5` (D-12); NaN-safe gate.
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

## PageIndex
- `[INFERRED]` No writable PageIndex spec corpus is indexed for this `.planning/`
  code tree (the MCP corpus is document/PDF-oriented). SPEC.md is the local
  authoritative draft; pending upsert documented in SPEC §10 only if the team
  later indexes `.planning/phases/**`.
