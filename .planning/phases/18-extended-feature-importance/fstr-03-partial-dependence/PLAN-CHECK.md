# Plan Check — FSTR-03 Partial Dependence

**Checker agent:** `plan-checker` (independent; all dependency/symbol/caller claims
verified through the CodeGraph MCP server + direct source reads).
**Artifacts checked:** `SPEC.md`, `PLAN.md`, `SOURCES.md` (this folder).
**Latest verdict:** ✅ **PASS** (pass 2 of 2).
**Date:** 2026-07-16.

---

## Verdict history

| Pass | Verdict | Blocking issues |
|------|---------|-----------------|
| 1 | ISSUES_FOUND | 3 × MAJOR (execution-blocking) + 5 × MINOR |
| 2 | **PASS** | 0 blocking; 2 × MINOR wording nits (now applied) |

The plan is **checker-approved for implementation**. No blocking issue remains.

---

## Pass 1 — ISSUES_FOUND (3 MAJOR, resolved)

### MAJOR #1 — unit-test file was never mounted → silent false-green
- **Evidence:** this crate wires sibling unit-test files via an explicit
  `#[cfg(test)] #[path = "X_test.rs"] mod tests;` mount in the production `.rs`
  (`crates/cb-model/src/ctr_data.rs:58-61`, `apply.rs:740-742`). The draft T0 only
  created `partial_dependence_test.rs` + declared `mod partial_dependence;` — it
  never added the mount, so the compiler never sees the test file and
  `cargo test -p cb-model` runs **zero** PDP-01/PDP-05 unit tests while reporting
  success, defeating the Red phase.
- **Resolution:** PLAN T0 now mandates appending the mount to
  `crates/cb-model/src/partial_dependence.rs`; completion evidence requires the
  mount line present.

### MAJOR #2 — `cargo build` is not the restriction-lint gate
- **Evidence:** the workspace `unwrap_used/expect_used/panic/indexing_slicing`
  denials are **clippy** lints (`Cargo.toml:10-14` + `cb-model/Cargo.toml:7-8
  [lints] workspace = true`); inert under `cargo build`/rustc. Checker reproduced
  empirically: `.unwrap()` compiles clean under `cargo build` (EXIT 0) and only
  fails under `cargo clippy` (EXIT 101). The draft used `cargo build` as the
  "restriction-lint gate", so the slice's central safety constraint went unchecked.
- **Resolution:** validation block + T0 + T7 now use
  `cargo clippy -p cb-model --all-targets` as the authoritative gate;
  `cargo build` relabeled "compile check only".

### MAJOR #3 — PDP-05 AT-05c was self-contradictory (`columns == []`)
- **Evidence:** SPEC §5 deterministic order runs column-shape (check 2) BEFORE
  emptiness (check 3), and the in-scope model has `n_float >= 1`. So `columns == []`
  (`0 != n_float`) is `MalformedColumns`, but the draft listed it under
  `EmptyDataset` (AT-05c) — a Red test that can never go green.
- **Resolution:** SPEC §5 check-2 explicitly folds `columns == []` into
  `MalformedColumns`; `EmptyDataset` (check 3, AT-05c) restricted to correct-width
  zero-row datasets. PLAN T2 `rejects_malformed_columns`/`rejects_empty_dataset`
  realigned accordingly.

### MINORs (pass 1, addressed)
- T2 now builds a **T3-independent in-code Model** (`n_float >= 2`) for the
  validation arms.
- AT-05d/AT-05e now state they pass valid rectangular non-empty columns so checks
  1–3 succeed and the range/duplicate check is the one under test.
- T4 loads via `cb_model::load_cbm(model.cbm)` (returns a `Model` directly; proven
  on an upstream `.cbm` at `advanced_fstr_oracle_test.rs:23,93`); `model.json`
  fallback noted.
- T3 documents the explicit row-major dump order for `pdp_pair_values.npy`
  (grid0/f1 outer, grid1/f2 inner) with a `reshape(...).ravel(order="C")` recipe.
- Empty-borders / f64→f32-cast edges reviewed — acceptable, oracle-adjudicated.

---

## Pass 2 — PASS

All three MAJOR fixes re-verified landed; the five PDP-05 arms
(AT-05a arity, AT-05b malformed, AT-05c empty, AT-05d out-of-range, AT-05e
duplicate) each re-traced as reachable and isolated under the deterministic check
order; no acceptance behavior lost its Red task. Hardening decisions re-affirmed
via CodeGraph: `Model` (model.rs:272) has no feature-kind map → `UnsupportedFeatureKind`
correctly removed; `predict_raw` (apply.rs:370) SoA float-indexed with NaN-pad at
:404-407 → `columns.len()==n_float`+rectangular is the correct guard;
`assert_abs_close` (compare.rs:46) returns `Result`; `sum_f64` (reduction.rs:32) is
the sanctioned fold; greenfield (0 grep hits), no module/symbol collision.

### Residual MINOR nits (raised in pass 2, now applied)
1. PLAN executor-contract line reworded so "no `#[cfg(test)] mod tests`" reads as
   "no inline `mod tests { … }` **body**; use the `#[path]` sibling mount" — removes
   the apparent contradiction with the T0 mount.
2. AT-05b/AT-05c now state they pass a valid-arity `features` (e.g. `&[0]`) so
   check 1 doesn't preempt the shape/emptiness check under test.

---

## Unverified items — now RESOLVED during implementation (2026-07-16)
- **Exact PDP-02 grid transform** — RESOLVED: upstream PD is **per BIN**
  (`n_borders+1` bins), grid = `[b0-1, midpoints…, b_last+1]`. Confirmed against
  `catboost==1.2.10` `core.py:4033-4055` and reproduced ≤1e-5 by the oracle tests.
- **`plot_partial_dependence` extraction path** — RESOLVED: `plot=False` returns
  `(all_predictions, fig)`; `all_predictions` is the oracle array (no figure
  parsing). Installed `catboost==1.2.10` via a `uv` CPython-3.12 venv to generate
  the fixtures.

## Post-implementation note
Slice fully implemented (T0–T7); specs PDP-01..05 all green (9 unit + 2 oracle
tests; oracle parity ≤1e-5 vs real upstream). The design refinement above (per-bin
grid; PDP-02 became a unit test since upstream exposes no numeric x-grid) did not
change the public typed contract (`values.len() == ∏ grids[f].len()` still holds).
