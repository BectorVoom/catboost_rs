# TDD Implementation Plan — FSTR-03 Partial Dependence

**Phase:** 18 (Extended Feature Importance) · **Slice:** FSTR-03
**Spec:** `./SPEC.md` (specs PDP-01..PDP-05) · **Requirement:** FSTR-03
**Crate:** `cb-model` (+ oracle fixture in `cb-oracle`) · **Impact:** `local`
**Parity bar:** `1e-5` (CPU, D-12) via `cb_oracle::compare::assert_abs_close`.

> Executor contract: strict Red → Green → Refactor per task. One spec per TDD
> cycle. **Source/test separation is mandatory** — no `#[cfg(test)] mod tests` in
> production `.rs`; unit tests go in `crates/cb-model/src/partial_dependence_test.rs`,
> integration/oracle tests in `crates/cb-model/tests/`. **No `unwrap`/`expect`/
> `panic`/`indexing_slicing`** in production (workspace-denied `[VERIFIED: LOCAL
> Cargo.toml:10-14]`). Every mean folds through `cb_core::sum_f64`, never `.sum()`
> (D-08). Do **not** mark any task complete during planning.

## Validation commands (host CPU; avoids env-red suites)

```
cargo test -p cb-model                     # unit + oracle for this slice
cargo test -p cb-model -p cb-oracle        # + comparator
cargo build -p cb-model                    # restriction-lint gate (unwrap/panic/indexing denied)
```
Known-red suites to ignore (pre-existing, environmental): `cb-backend --lib`
(CubeCL MLIR), `cb-train monotone_*`, `catboost-rs-py` (python3.14 link)
`[VERIFIED: LOCAL memory catboost-rs-preexisting-test-failures.md]`. Fixture
regeneration needs upstream `catboost==1.2.10` in a venv (offline; artifacts committed).

## Task graph (dependencies, not file order)

```
T0 scaffold ──┬─> T1 (PDP-01 engine, unit) ──┐
              ├─> T2 (PDP-05 validation, unit)│
              └─> T3 (fixture gen, enabler) ──┼─> T4 (PDP-02 grid, oracle) ─> T5 (PDP-03 single, oracle) ─> T6 (PDP-04 pair, oracle) ─> T7 refactor/wire
```
- **Parallelizable:** T1, T2, T3 after T0.
- **Serial spine:** T3 → T4 → T5 → T6 (each oracle builds on the prior).

---

## T0 — Scaffold module + error type (enabler, no spec)

- **Goal:** create the module and typed error so later tasks compile.
- **Files:**
  - create `crates/cb-model/src/partial_dependence.rs` — `PartialDependence`
    struct, `PdpError` enum (§4 contract), empty `pub fn partial_dependence(...)`
    returning `todo`-free `Err(PdpError::EmptyDataset)` placeholder is **not**
    allowed (no stubs that pass); instead leave the fn **unimplemented at test
    time only** by making T1/T2 the first cycles that force its body. Concretely:
    define the types + a private engine signature; the public fn is authored in T1/T3.
  - create empty `crates/cb-model/src/partial_dependence_test.rs`.
  - edit `crates/cb-model/src/lib.rs:14-22` — add `mod partial_dependence;` and
    (at T7) `pub use partial_dependence::{PartialDependence, PdpError, partial_dependence};`.
  - **Decision (record in code doc):** place `PdpError` in the new module (keeps
    the slice self-contained) rather than extending `cb_model::ModelError`.
    `[INFERRED from SPEC §4 note]`
- **Validation:** `cargo build -p cb-model` compiles with the new (unused) module.
- **Completion evidence:** module + test file exist; lib.rs declares the module.
- **Refactor constraints:** none yet.

## T1 — PDP-01 single-feature averaging engine (unit)

- **Spec:** PDP-01. **Depends on:** T0.
- **Red** — in `partial_dependence_test.rs`:
  - test `engine_averages_constant_grid_equals_direct_mean` (AT-01a): build a
    small model via the existing test-model helper (or load the fixture model
    once available; for the unit test use a hand-built or `predict_raw`-referenced
    model), pick feature `f` and a 1-point grid `[v]`; expected = mean over
    objects of `predict_raw(model, columns_with_column_f_all_v)` computed inline;
    assert engine output `[0]` equals it (bit-exact — same fold).
  - test `engine_output_length_and_order` (AT-01b): 3-point grid → len 3, order preserved.
  - test `engine_does_not_mutate_columns` (AT-01c): clone-compare `columns` after call.
  - **Expected initial failure:** engine fn absent / returns nothing → compile
    error then assertion failure. Record the failing output.
- **Green:** implement the private engine
  `fn pdp_curve_single(model, columns, feature, grid) -> Vec<f64>`: for each
  grid point, build a working `Vec<Vec<f32>>` = clone of `columns` with column
  `feature` overwritten to `grid[k] as f32` for all objects, call `predict_raw`,
  push `sum_f64(&preds) / n_objects`. Use checked `.get`, no indexing.
- **Refactor:** hoist the working-buffer allocation out of the grid loop (reuse
  one buffer, reset the target column per point) — keep behavior byte-identical;
  re-run T1 tests.
- **Validation:** `cargo test -p cb-model partial_dependence` (unit subset green).
- **Completion evidence:** AT-01a/b/c pass; no `.sum()`/`unwrap` introduced.

## T2 — PDP-05 typed input validation (unit)

- **Spec:** PDP-05. **Depends on:** T0. **Parallel with:** T1, T3.
- **Red** — unit tests, one per arm (AT-05a..d):
  - `rejects_out_of_range_feature` → `Err(FeatureIndexOutOfRange{..})`.
  - `rejects_non_float_feature` → `Err(UnsupportedFeatureKind{..})` (construct/load
    a model with a categorical/CTR feature, or assert via the float-index bound
    if the fixture is numeric-only — see note).
  - `rejects_bad_arity` for `len 0` and `len 3` → `Err(UnsupportedFeatureArity{..})`.
  - `rejects_empty_dataset` for `columns=[]` and `columns=[vec![]]` → `Err(EmptyDataset)`.
  - **Expected initial failure:** validation absent → wrong/no `Err`.
- **Green:** implement the guard block at the top of `partial_dependence(...)`:
  arity check → dataset non-empty check → per-feature range+kind check, returning
  the matching `PdpError` before any compute. Derive `n_float` from
  `model.float_feature_borders.len()`; "float feature" = index `< n_float`.
- **Refactor:** extract a `validate(model, columns, features) -> Result<usize /*n_obj*/, PdpError>`.
- **Note (owner: executor):** if the committed fixture model is numeric-only, the
  `UnsupportedFeatureKind` test needs a model that actually has a non-float
  feature; reuse an existing categorical fixture model (e.g. from
  `one_hot_cat`/`plain_ctr`) rather than adding a new one. `[VERIFIED: LOCAL
  crates/cb-oracle/fixtures/{one_hot_cat,plain_ctr}]`
- **Validation:** `cargo test -p cb-model partial_dependence`.
- **Completion evidence:** AT-05a..d pass.

## T3 — Oracle fixture generation (enabler artifact)

- **Spec:** enables PDP-02/03/04. **Depends on:** T0 (naming only). **Blocking for:** T4–T6.
- **Files (new):** `crates/cb-oracle/fixtures/partial_dependence/`
  - `gen_fixtures.py` — modeled on `advanced_fstr/gen_fixtures.py`
    `[VERIFIED: LOCAL crates/cb-oracle/fixtures/advanced_fstr/gen_fixtures.py:1-70]`:
    load `inputs/numeric_tiny` X/y; train a **numeric-only** `CatBoostRegressor`
    with the pinned ISOLATING params (`bootstrap_type="No"`, `depth=2`,
    `iterations=5`, `l2_leaf_reg=3.0`, `learning_rate=0.1`, `random_seed=0`,
    `random_strength=0`, `score_function="L2"`, `thread_count=1`, `verbose=False`)
    `[VERIFIED: LOCAL crates/cb-oracle/generator/gen_fixtures.py:145-175]`; save
    `model.cbm` + `model.json`.
  - Call `fig = model.plot_partial_dependence(pool, <f>)` for a chosen single
    feature `f`, and `plot_partial_dependence(pool, [f1, f2])` for a pair; extract
    the **upstream** grid + values from the returned figure object and dump:
    `pdp_single_grid.npy`, `pdp_single_values.npy`, `pdp_pair_grid0.npy`,
    `pdp_pair_grid1.npy`, `pdp_pair_values.npy` (row-major, `float64`).
    **Resolve the exact figure-data extraction path** against installed
    `catboost==1.2.10` (SPEC §9 risk 2) — do NOT recompute truth with a hand
    averaging loop.
  - `config.json` — `catboost_version:"1.2.10"`, `input_dataset:"numeric_tiny"`,
    `single_feature: f`, `pair_features:[f1,f2]`, seeds, `thread_count:1`, `note`,
    and the artifact list (mirror `advanced_fstr/config.json` shape).
    `[VERIFIED: LOCAL crates/cb-oracle/fixtures/advanced_fstr/config.json]`
- **Red/Green/Refactor:** N/A (data artifact). **Verification:** re-load each
  `.npy` in Python and assert finite, correct dtype/shape; commit artifacts.
- **Validation:** `python crates/cb-oracle/fixtures/partial_dependence/gen_fixtures.py`
  under a venv with `catboost==1.2.10`; `.npy` files present and loadable.
- **Completion evidence:** committed `.cbm`/`.json`/`.npy`/`config.json`; the
  single-feature `pdp_single_grid.npy` is what PDP-02 (T4) locks against.
- **Rollback note:** artifacts are additive; deletion is safe (no consumer until T4).

## T4 — PDP-02 grid derivation (oracle)

- **Spec:** PDP-02. **Depends on:** T3.
- **Red** — `crates/cb-model/tests/partial_dependence_oracle_test.rs`:
  - `derived_grid_matches_upstream` (AT-02a): load `model.cbm`, derive grid for
    `single_feature`, load `pdp_single_grid.npy`, `assert_abs_close(grid, npy, 1e-5)`.
  - **Expected initial failure:** grid derivation absent / wrong transform →
    length or value mismatch. **Record the exact mismatch** — it reveals the
    transform (borders-verbatim vs midpoints vs endpoints).
- **Green:** implement `fn grid_for_feature(model, feature) -> Vec<f64>` from
  `model.float_feature_borders[feature]` (checked `.get`), applying the transform
  that reproduces the dumped grid. Start with **borders-verbatim**; if the RED
  mismatch shows a shift, switch to the empirically-correct transform (consecutive
  midpoints / +min/max endpoints) until AT-02a passes. Document the resolved
  convention in the fn doc + update SPEC §9 risk 1 to `[VERIFIED]`.
- **Refactor:** none beyond naming; keep ascending/dedup invariant.
- **Validation:** `cargo test -p cb-model --test partial_dependence_oracle_test derived_grid`.
- **Completion evidence:** AT-02a green; SPEC risk 1 resolved and recorded.

## T5 — PDP-03 single-feature end-to-end (oracle)

- **Spec:** PDP-03. **Depends on:** T1, T4.
- **Red** — same oracle test file:
  - `single_feature_pdp_matches_upstream` (AT-03a): `partial_dependence(model,
    cols, &[single_feature])?.values` vs `pdp_single_values.npy` at `1e-5`.
  - `single_feature_pdp_grid_and_meta` (AT-03b): `grids[0]` == `pdp_single_grid.npy`,
    `features == [single_feature]`.
  - **Expected initial failure:** public fn not yet composing grid+engine.
- **Green:** author the happy path of `partial_dependence(...)`: after PDP-05
  validation, for `features.len()==1` → `grid = grid_for_feature`; `values =
  pdp_curve_single(model, columns, f, &grid)`; return `PartialDependence`.
- **Refactor:** ensure the working-buffer reuse from T1 is used; no dup logic.
- **Validation:** `cargo test -p cb-model --test partial_dependence_oracle_test`.
- **Completion evidence:** AT-03a/b green at `1e-5`.

## T6 — PDP-04 two-feature partial dependence (oracle)

- **Spec:** PDP-04. **Depends on:** T5.
- **Red** — oracle test:
  - `pair_feature_pdp_matches_upstream` (AT-04a): `partial_dependence(model, cols,
    &[f1,f2])?.values` vs `pdp_pair_values.npy` at `1e-5`.
  - `pair_feature_grid_and_rowmajor` (AT-04b): `grids == [grid0, grid1]`;
    `values.len() == grid0.len()*grid1.len()`; assert a known
    `(a,b)->a*len1+b` element to lock row-major order.
  - **Expected initial failure:** two-feature path absent.
- **Green:** generalize the engine to `pdp_curve_pair(model, columns, (f1,f2),
  (g1,g2)) -> Vec<f64>`: nested grid loop overriding **both** columns; push
  `sum_f64(&preds)/n` row-major (`f1` outer, `f2` inner). Route `features.len()==2`
  through it in `partial_dependence(...)`.
- **Refactor:** factor a shared override helper used by both single and pair
  engines (override k columns to k constants) to remove duplication; keep folds
  byte-identical.
- **Validation:** `cargo test -p cb-model --test partial_dependence_oracle_test`.
- **Completion evidence:** AT-04a/b green.

## T7 — Refactor, public export, docs, full-slice gate

- **Depends on:** T1–T6.
- **Steps:**
  - `crates/cb-model/src/lib.rs` — `pub use partial_dependence::{PartialDependence,
    PdpError, partial_dependence};`.
  - Module-level doc comment on `partial_dependence.rs` transcribing the upstream
    averaging formula + the **resolved** grid convention (cite CONTEXT7 +
    resolved T4 finding); note float-only scope + typed rejections.
  - Confirm no `.sum()`/`unwrap`/`expect`/indexing crept in (grep the new file).
- **Validation (full slice):**
  ```
  cargo build -p cb-model
  cargo test -p cb-model
  cargo test -p cb-model -p cb-oracle
  ```
- **Completion evidence:** all PDP-01..05 acceptance tests green; restriction
  lints clean; SPEC §9 risks 1–2 marked resolved. Then flip REQUIREMENTS.md
  FSTR-03 checkbox + ROADMAP/STATE bookkeeping (bookkeeping only — outside TDD).

## Traceability (task → spec → acceptance)

| Task | Spec | Acceptance tests | Kind |
|------|------|------------------|------|
| T0 | (enabler) | compiles | — |
| T1 | PDP-01 | AT-01a/b/c | unit |
| T2 | PDP-05 | AT-05a/b/c/d | unit |
| T3 | (enabler) | fixtures loadable | artifact |
| T4 | PDP-02 | AT-02a | oracle |
| T5 | PDP-03 | AT-03a/b | oracle |
| T6 | PDP-04 | AT-04a/b | oracle |
| T7 | all | full-slice green | gate |

Every SPEC acceptance behavior has a Red task; every task references ≥1 spec ID.
