# TDD Implementation Plan — FSTR-03 Partial Dependence

> ## Execution status (2026-07-16) — ✅ COMPLETE (all of T0–T7)
> The oracle blocker was lifted by installing `catboost==1.2.10` in a `uv`-managed
> CPython-3.12 venv, so the full slice shipped:
> - **T0** scaffold + `#[path]` unit-test mount · **T1** PDP-01 engine · **T2**
>   PDP-05 validation · **T3** fixtures generated from real upstream · **T4** PDP-02
>   per-bin grid · **T5** PDP-03 single-feature oracle · **T6** PDP-04 pair oracle ·
>   **T7** `pub use` export + resolved module docs.
> - **Tests:** 9 unit (`cargo test -p cb-model partial_dependence`) + 2 oracle
>   (`--test partial_dependence_oracle_test`) green; full `cargo test -p cb-model` =
>   0 failures. New code restriction-lint clean (`cargo clippy -p cb-model --lib
>   --no-deps` → 0 errors on `partial_dependence.rs`).
> - **Grid transform RESOLVED:** upstream PD is **per BIN** (`n_borders+1` bins);
>   grid = `[b0-1, midpoints…, b_last+1]`. Oracle truth =
>   `plot_partial_dependence(pool, features, plot=False)[0]` (== `_calc_partial_dependence`),
>   reproduced ≤1e-5. Specs **PDP-01..05 all implemented**.
> - **Files:** `crates/cb-model/src/partial_dependence.rs`,
>   `partial_dependence_test.rs`, `lib.rs` (+2), new fixture dir
>   `crates/cb-oracle/fixtures/partial_dependence/` (`gen_fixtures.py`, `model.cbm`,
>   `model.json`, `pdp_single_values.npy`, `pdp_pair_values.npy`, `config.json`).
>
> **Not done (out of slice, unchanged):** facade (`catboost-rs`) / Python
> (`catboost-rs-py`) surfacing = later DX task; the FSTR-03 requirement checkbox in
> the git-recovered `.planning/REQUIREMENTS.md` (off-tree) — confirm canonical
> revision before flipping. Pre-existing/unrelated: `cargo clippy -p cb-model
> --all-targets` still trips on `cb-oracle/src/model_json.rs:161` +
> `tests/ctr_data_roundtrip_test.rs` (baseline-reproduced) — gate new cb-model code
> with `--lib --no-deps`.



**Phase:** 18 (Extended Feature Importance) · **Slice:** FSTR-03
**Spec:** `./SPEC.md` (specs PDP-01..PDP-05) · **Requirement:** FSTR-03
**Crate:** `cb-model` (+ oracle fixture in `cb-oracle`) · **Impact:** `local`
**Parity bar:** `1e-5` (CPU, D-12) via `cb_oracle::compare::assert_abs_close`.

> Executor contract: strict Red → Green → Refactor per task. One spec per TDD
> cycle. **Source/test separation is mandatory** — no inline `#[cfg(test)] mod
> tests { … }` *body* in production `.rs`; unit tests go in a sibling
> `crates/cb-model/src/partial_dependence_test.rs` wired via the sanctioned
> `#[cfg(test)] #[path = "partial_dependence_test.rs"] mod tests;` mount (see T0,
> mirroring `ctr_data.rs:58-61`/`region_apply_test.rs`), integration/oracle tests
> in `crates/cb-model/tests/`. **No `unwrap`/`expect`/
> `panic`/`indexing_slicing`** in production (workspace-denied `[VERIFIED: LOCAL
> Cargo.toml:10-14]`). Every mean folds through `cb_core::sum_f64`, never `.sum()`
> (D-08). Do **not** mark any task complete during planning.

## Validation commands (host CPU; avoids env-red suites)

```
cargo test -p cb-model                     # unit + oracle for this slice
cargo test -p cb-model -p cb-oracle        # + comparator
cargo clippy -p cb-model --all-targets     # RESTRICTION-LINT GATE (unwrap/expect/panic/indexing denied)
cargo build -p cb-model                    # compile check only — does NOT enforce the clippy restriction lints
```
> **Lint-gate correction:** the workspace restriction lints
> (`unwrap_used/expect_used/panic/indexing_slicing`) are **clippy** lints; they are
> inert under `cargo build`/`rustc` and are ONLY enforced by `cargo clippy`
> `[VERIFIED: LOCAL Cargo.toml:10-14 + crates/cb-model/Cargo.toml:7-8 `[lints]
> workspace = true`]`. Use `cargo clippy` as the gate; `cargo build` passing does
> NOT prove the slice is free of `.unwrap()`/indexing.
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
    struct, `PdpError` enum with the **five** §4 variants (`FeatureIndexOutOfRange`,
    `UnsupportedFeatureArity`, `DuplicateFeature`, `MalformedColumns`,
    `EmptyDataset` — NO `UnsupportedFeatureKind`, which was removed as
    unimplementable per §4 change-note). A stub `partial_dependence(...)` that
    passes any test is **not** allowed (no `Ok`/`Err` placeholder that a Red test
    would accept); instead define the types + a private engine signature only, and
    let T1/T2/T5 force the public fn's body. `[VERIFIED: SPEC §4 PdpError]`
  - create empty `crates/cb-model/src/partial_dependence_test.rs`.
  - **MOUNT the unit-test file** — append to the bottom of
    `crates/cb-model/src/partial_dependence.rs` the sanctioned source/test-separation
    mount (NOT an embedded test body):
    ```rust
    #[cfg(test)]
    #[path = "partial_dependence_test.rs"]
    mod tests;
    ```
    This mirrors `crates/cb-model/src/ctr_data.rs:59-61` and `apply.rs:741`
    `[VERIFIED: LOCAL crates/cb-model/src/ctr_data.rs:58-61]`. **Without this mount
    the compiler never sees `partial_dependence_test.rs` and `cargo test` runs ZERO
    unit tests while reporting success** — a silent false-green that defeats the
    Red phase for T1/T2.
  - edit `crates/cb-model/src/lib.rs:14-22` — add `mod partial_dependence;` and
    (at T7) `pub use partial_dependence::{PartialDependence, PdpError, partial_dependence};`.
  - **Decision (record in code doc):** place `PdpError` in the new module (keeps
    the slice self-contained) rather than extending `cb_model::ModelError`.
    `[INFERRED from SPEC §4 note]`
- **Validation:** `cargo build -p cb-model` compiles with the new module; the
  `#[path]` mount makes `cargo test -p cb-model` actually pick up (empty)
  `partial_dependence_test.rs`.
- **Completion evidence:** module + test file exist and are MOUNTED (the `#[path]`
  line is present); lib.rs declares the module.
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
  - **Precondition (relied upon, enforced by PDP-05/T2):** `columns.len() ==
    n_float` and rectangular, so `predict_raw` never NaN-pads a short/missing
    column `[VERIFIED: CODEGRAPH crates/cb-model/src/apply.rs:404-407]`. The engine
    is a private fn; the public `partial_dependence` runs `validate(...)` (T2)
    first, so the engine is only ever reached with validated inputs. For the T1
    *unit* tests that call the engine directly, construct `columns` that already
    satisfy the precondition (width == the test model's `n_float`).
- **Refactor:** hoist the working-buffer allocation out of the grid loop (reuse
  one buffer, reset the target column per point) — keep behavior byte-identical;
  re-run T1 tests.
- **Validation:** `cargo test -p cb-model partial_dependence` (unit subset green).
- **Completion evidence:** AT-01a/b/c pass; no `.sum()`/`unwrap` introduced.

## T2 — PDP-05 typed input validation (unit)

- **Spec:** PDP-05. **Depends on:** T0. **Parallel with:** T1, T3.
- **Test model (T3-independent):** these arms only read `model.float_feature_borders.len()`,
  so build a small **in-code** `cb_model::Model` with `n_float >= 2` (a couple of
  non-empty border vecs, empty trees) rather than loading the T3 fixture — this
  keeps T2 genuinely parallel with T3. NO categorical model is needed (a categorical
  target has no float index and cannot be expressed via the float-index API, SPEC §4).
- **Red** — unit tests, one per arm (AT-05a..e):
  - `rejects_bad_arity` for `features.len()==0` and `==3` → `Err(UnsupportedFeatureArity{ requested })` (AT-05a).
  - `rejects_malformed_columns` (AT-05b) for **three** shapes — `columns == []`,
    `columns.len() != n_float` (wrong width), and ragged (unequal-length) columns —
    each → `Err(MalformedColumns{ expected_float_features, .. })`. (`columns == []`
    is malformed, NOT empty, because `0 != n_float`; SPEC §5 check order.)
  - `rejects_empty_dataset` (AT-05c) for exactly `n_float` columns each of length 0
    (correct width, zero rows) → `Err(EmptyDataset)`.
  - **Both AT-05b and AT-05c pass a valid-arity `features`** (e.g. `&[0]`) so check 1
    (arity) does not preempt the column-shape / emptiness check under test.
  - `rejects_out_of_range_feature` (AT-05d): pass **valid rectangular non-empty
    columns** (`n_float` columns, length `n>=1`) with `features=[n_float]` →
    `Err(FeatureIndexOutOfRange{ index, n_float })` (so checks 1–3 pass and the
    range check is under test).
  - `rejects_duplicate_feature_pair` (AT-05e): pass valid rectangular non-empty
    columns with `features=[f, f]` (`f` in range) → `Err(DuplicateFeature{ index: f })`.
  - **Expected initial failure:** validation absent → wrong/no `Err`.
- **Green:** implement `fn validate(model, columns, features) -> Result<usize /*n_obj*/, PdpError>`
  and call it at the very top of `partial_dependence(...)`, honoring the SPEC §5
  **deterministic check order**: (1) arity → (2) column shape (`columns.len() ==
  n_float` AND rectangular) → (3) empty dataset (`n_obj == 0`) → (4) per-feature
  range (first `features[k] >= n_float`, in request order) → (5) duplicate
  (2-feature `[f,f]`). Derive `n_float = model.float_feature_borders.len()`. Return
  `n_obj` (the common column length) on success. Use checked `.get`/iterators — no
  indexing, no `unwrap`. This `validate` is the single guard the T1 engine relies
  on to never NaN-pad `[VERIFIED: CODEGRAPH crates/cb-model/src/apply.rs:404-407]`.
- **Refactor:** keep `validate` as the one reusable entry guard for both the
  single- and pair-feature paths (T5/T6 call the same `partial_dependence`).
- **Validation:** `cargo test -p cb-model partial_dependence`.
- **Completion evidence:** AT-05a..e pass; the removed `UnsupportedFeatureKind`
  arm has no test (it no longer exists — SPEC §4 change-note).

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
    `pdp_pair_grid1.npy`, `pdp_pair_values.npy` (`float64`). **`pdp_pair_values.npy`
    MUST be dumped in the SPEC §4 row-major order — `grid0`/`f1` OUTER, `grid1`/`f2`
    INNER, i.e. flat index `a*len(grid1)+b` = `(grid0[a], grid1[b])`** — so any
    transpose disagreement with the Rust engine surfaces as a real AT-04a oracle
    failure, not a silent pass. Reshape/flatten explicitly (e.g.
    `np.asarray(surface, dtype=np.float64).reshape(len(grid0), len(grid1)).ravel(order="C")`)
    and record the axis→feature mapping in `config.json`.
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
- **Oracle-test harness (applies to T4/T5/T6, mirror the verified pattern):**
  `crates/cb-model/tests/partial_dependence_oracle_test.rs` opens with
  `#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]`,
  a `const TOL: f64 = 1e-5;`, a `fixture(rel)` path helper joining
  `../cb-oracle/fixtures/…`, and loads `.npy` via `ndarray_npy::read_npy` /
  `cb_oracle::load_f64_vec` — exactly as `advanced_fstr_oracle_test.rs`
  `[VERIFIED: LOCAL crates/cb-model/tests/advanced_fstr_oracle_test.rs:18-49]`.
  **`assert_abs_close` returns `Result<(), OracleError>` — it does NOT panic**
  `[VERIFIED: CODEGRAPH crates/cb-oracle/src/compare.rs:46]`; assert success with
  `.expect("… within TOL")` (permitted under the top-of-file test allow) or make
  the test fn return `Result<(), OracleError>`.
- **Red** — in that file:
  - `derived_grid_matches_upstream` (AT-02a): load the model via
    `cb_model::load_cbm(fixture("partial_dependence/model.cbm"))` — which returns a
    `cb_model::Model` directly and is proven to parse an upstream `catboost==1.2.10`
    `.cbm` `[VERIFIED: LOCAL crates/cb-model/tests/advanced_fstr_oracle_test.rs:23,93
    load_cbm on an upstream fixture]` (alternatively `cb_oracle::load_model_json` on
    `model.json` + manual `Model` build, the binclf pattern at :53-55) — derive the
    grid for `single_feature`, load `pdp_single_grid.npy`, then
    `assert_abs_close(&grid, &npy, TOL).expect("grid within TOL")`.
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
  - Confirm no `.sum()`/`unwrap`/`expect`/indexing crept in — grep the new file
    AND run the clippy gate below (the grep is a smell check; `cargo clippy` is the
    authoritative enforcement).
- **Validation (full slice):**
  ```
  cargo clippy -p cb-model --all-targets   # restriction-lint gate (authoritative)
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
| T2 | PDP-05 | AT-05a/b/c/d/e | unit |
| T3 | (enabler) | fixtures loadable | artifact |
| T4 | PDP-02 | AT-02a | oracle |
| T5 | PDP-03 | AT-03a/b | oracle |
| T6 | PDP-04 | AT-04a/b | oracle |
| T7 | all | full-slice green | gate |

Every SPEC acceptance behavior has a Red task; every task references ≥1 spec ID.
