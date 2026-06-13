---
phase: 04-model-serialization-shap-rust-api-first-full-oracle-lock
plan: 03
subsystem: model
tags: [cbm, flatbuffers, model-json, serde, serialization, oracle, security-v5]

# Dependency graph
requires:
  - phase: 04-model-serialization-shap-rust-api-first-full-oracle-lock
    plan: 01
    provides: "canonical cb-model::Model {oblivious_trees, bias, float_feature_borders, per-tree leaf_weights}; committed flatc TModelCore/TModelTrees bindings (root_as_tmodel_core); cb-oracle::model_json parser + compare_stage"
  - phase: 04-model-serialization-shap-rust-api-first-full-oracle-lock
    plan: 02
    provides: "cb-model::predict_raw CPU apply path (drives the load-parity oracle checks)"
provides:
  - "cb-model::cbm::{save_cbm, load_cbm, decode_cbm} — native .cbm (CBM1 magic + ui32 LE size + FlatBuffers TModelCore) with global bin-feature split encoding, flat LeafValues/LeafWeights, MultiBias[0]-aware bias read"
  - "cb-model::json::{save_json, load_json, decode_json} — model.json export/import on the upstream schema (per-tree NESTED leaf_weights, scale_and_bias=[1,[bias]])"
  - "cb-model::error::ModelError (thiserror: Deserialize / SchemaVersion / Json / Core / Io) — typed, panic-free deserialization errors (Security V5)"
  - "Oracle locks: .cbm semantic round-trip + upstream 1.2.10 binclf/regression .cbm load <=1e-5; model.json round-trip through cb-oracle parser + upstream binclf/regression model.json load <=1e-5; malformed-input rejection (cbm + json)"
affects: [04-04, 04-05, shap, fstr, rust-api, model-serialization]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Upstream global bin-feature split encoding (CalcBinFeatures order): TreeSplits[i] indexes a flat (feature,border) list built from FloatFeatures borders in order; save maps Split->global index, load decodes back"
    - "1-dim bias read prefers MultiBias[0] (upstream 1.2.10 stores the boost_from_average start there), falls back to scalar Bias (what save_cbm writes) — ours->ours round-trip preserved, upstream->ours regression load correct"
    - "VERIFYING root_as_tmodel_core only (no _unchecked); declared ui32 size BOUNDED against actual file length before slicing; magic via checked .get(0..4)"
    - "model.json leaf_weights NESTED per tree (Pitfall 2); scale_and_bias=[1,[bias]] (Pitfall 6); the existing cb-oracle parser doubles as the round-trip schema oracle (D-04)"

key-files:
  created:
    - crates/cb-model/src/cbm.rs
    - crates/cb-model/src/error.rs
    - crates/cb-model/src/json.rs
    - crates/cb-model/tests/cbm_oracle_test.rs
    - crates/cb-model/tests/json_oracle_test.rs
  modified:
    - crates/cb-model/src/lib.rs
    - crates/cb-model/Cargo.toml

key-decisions:
  - "DEVIATION (Rule 1, auto-fixed): read bias from MultiBias[0] not just the scalar Bias field — upstream catboost 1.2.10 stores the single-target bias in the MultiBias vector, leaving scalar Bias at 0.0. Without this the regression .cbm/json load diverged by exactly the bias (0.315). binclf (bias 0) passed regardless."
  - "Borders are f32 on the .cbm wire (the schema type), f64 canonical/JSON; the ours->ours assert_eq round-trip uses f32-exact borders (0.5/1.5/2.5). LeafValues/LeafWeights are f64 on the wire so they round-trip exactly. Borders not f32-exact (e.g. upstream) are covered by the apply-parity round-trip + upstream-load checks, not assert_eq."
  - "save_cbm writes the scalar Bias (Open Q3: single Bias for 1-dim); load_cbm prefers MultiBias when present — symmetric for our files, correct for upstream files."
  - "json.rs split_index is a per-tree positional value (self-consistent), not upstream's global split-pool index — the JSON apply/round-trip never needs the global index (that is the .cbm concern)."
  - "Upstream-load cases run IN-ENV (not #[ignore]): real catboost 1.2.10 fixtures model_serde/{binclf,regression}/model.{cbm,json}+predictions.npy on numeric_tiny exist under cb-oracle/fixtures."

requirements-completed: [MODEL-01, MODEL-06]

# Metrics
duration: ~8min
completed: 2026-06-13
---

# Phase 4 Plan 03: Native .cbm + model.json Serialization Summary

**Native `.cbm` (FlatBuffers `TModelCore` with `CBM1` framing) and `model.json` (upstream schema) save/load for the canonical `cb-model::Model`, oracle-locked to catboost 1.2.10 — semantic round-trip, upstream binclf/regression load `<=1e-5`, and malformed-input rejection via a typed `ModelError` (never panics, Security V5).**

## Performance

- **Duration:** ~8 min
- **Completed:** 2026-06-13
- **Tasks:** 2 (both `auto`)
- **Files changed:** 7 (5 created, 2 modified), +1173 lines

## Accomplishments

- **MODEL-01 — native `.cbm` (de)serialization.** `cb-model::cbm::{save_cbm, load_cbm, decode_cbm}`. Writer emits `CBM1` magic + ui32 LE core size + a FlatBuffers `TModelCore` (`FormatVersion = "FlabuffersModel_v1"` — the canonical typo, Pitfall 5) carrying global `TreeSplits` (upstream `CalcBinFeatures` bin-feature index order), per-tree `TreeSizes`/`TreeStartOffsets`, the flat `LeafValues`/`LeafWeights` arrays (one tree slice per offset, Pitfall 2), `FloatFeatures` borders (f32), `ApproxDimension=1`, and the single `Bias`. Reader validates the magic with checked `.get(0..4)`, BOUNDS the declared ui32 size against the actual file length before slicing, parses with the VERIFYING `root_as_tmodel_core` (never `_unchecked`), reconstructs the canonical `Model` (splits decoded from the rebuilt flat bin-feature list, leaf values/weights sliced per tree, bias from `MultiBias[0]` with scalar `Bias` fallback). Round-trips semantically and loads the upstream 1.2.10 `binclf` + `regression` `.cbm` with `predict_raw` `<=1e-5`.
- **MODEL-06 — `model.json` export/import.** `cb-model::json::{save_json, load_json, decode_json}` with serde structs matching the upstream field names verbatim: `features_info.float_features[].borders`, `oblivious_trees[]` with per-tree NESTED `leaf_values`/`leaf_weights` (Pitfall 2), and `scale_and_bias = [1, [bias]]` (Pitfall 6). `save_json` output parses back through `cb_oracle::model_json::load_model_json` to a matching model (D-04 — the parser is the schema oracle); `load_json` on the upstream `binclf`/`regression` `model.json` applies `<=1e-5`.
- **Security V5 — typed, panic-free deserialization.** `cb-model::error::ModelError` (thiserror: `Deserialize`/`SchemaVersion`/`Json`/`Core`/`Io`). Every malformed-input path returns a typed error and NEVER panics: bad `.cbm` magic, oversized/short declared size, truncated/corrupt FlatBuffers buffer, short header (<8 bytes), wrong `FormatVersion`, garbage JSON, wrong-shape JSON, and a malformed `scale_and_bias` — each asserted by a dedicated test. No `unwrap`/`expect`/raw-index in the production path (workspace deny-lints satisfied; verifying accessor only).

## Task Commits

1. **Task 1: .cbm save/load (FlatBuffers framing) + validated deserialization** — `a9a3de5` (feat)
2. **Task 2: model.json export/import on the upstream schema** — `63ab7a3` (feat)

## Files Created/Modified

- `crates/cb-model/src/cbm.rs` (created) — `.cbm` framing + FlatBuffers `TModelCore` save/load; global bin-feature split encoding; bounded/verified deserialization; `MultiBias[0]`-aware bias read.
- `crates/cb-model/src/error.rs` (created) — `ModelError` (thiserror; mirrors the cb-oracle `#[from] io::Error` tradeoff that drops Clone/PartialEq).
- `crates/cb-model/src/json.rs` (created) — `model.json` export/import; serde structs on the upstream schema; nested `leaf_weights`; `scale_and_bias=[1,[bias]]`; panic-free bias read.
- `crates/cb-model/src/lib.rs` — wired `mod cbm; mod error; mod json;` + re-exports.
- `crates/cb-model/Cargo.toml` — added `thiserror`/`serde`/`serde_json` (workspace-pinned).
- `crates/cb-model/tests/cbm_oracle_test.rs` (created) — round-trip (`assert_eq!` + apply parity), upstream binclf+regression `.cbm` load `<=1e-5`, malformed-input (bad magic/oversized/truncated/short header), FormatVersion literal.
- `crates/cb-model/tests/json_oracle_test.rs` (created) — save_json through the cb-oracle parser, schema-shape asserts (nested leaf_weights, `[1,[bias]]`), full save->load round-trip, upstream binclf+regression `model.json` load `<=1e-5`, malformed-json typed errors.

## Decisions Made

See `key-decisions` frontmatter. Most load-bearing: (1) bias read prefers `MultiBias[0]` (upstream 1.2.10 1-dim form) over the scalar `Bias`, which is what made the regression load parity pass; (2) f32 border wire type means the `assert_eq!` round-trip uses f32-exact borders while non-exact-border models (upstream) are covered by apply-parity + upstream-load locks; (3) the upstream-load cases run in-env against real 1.2.10 fixtures rather than being `#[ignore]`'d.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Bias read from MultiBias[0], not just the scalar Bias field**
- **Found during:** Task 1 (regression upstream-load oracle initially diverged by exactly 0.3151967525482178 = the model bias).
- **Issue:** The plan said "bias from Bias" (Open Q3: write single Bias for 1-dim). But upstream catboost 1.2.10 STORES the single-target bias (e.g. a regression `boost_from_average` start) in the `MultiBias` VECTOR, leaving the scalar `Bias` field at its 0.0 default. Reading only scalar `Bias` returned 0 and the loaded regression model under-predicted by the bias on every object. (binclf has bias 0, so it passed regardless and masked the bug.)
- **Fix:** Added `read_bias` — prefer `MultiBias[0]` when the vector is non-empty, fall back to the scalar `Bias`. `save_cbm` still writes the scalar `Bias` (Open Q3), so ours->ours round-trip is unaffected; upstream->ours regression now loads correctly.
- **Files modified:** `crates/cb-model/src/cbm.rs`
- **Verification:** `cbm_load_upstream_regression_applies_within_tol` + `json_load_upstream_regression_applies_within_tol` pass `<=1e-5`; round-trip `assert_eq!` still holds.
- **Committed in:** `a9a3de5`

**Total deviations:** 1 auto-fixed (Rule 1). No architectural changes, no scope creep — the fix is purely a reader correctly honoring upstream's bias storage location.

## Issues Encountered

- **Disk-space limit (environment, not a code defect):** the box is ~100% full (<1.5 GB free). Per the environment constraints, NO `cargo test --workspace` / `cargo check --workspace --tests` was run (those recompile polars-core at the link step and fail with "No space left on device"). Verified per-crate instead: `cargo test -p cb-model` runs all four cb-model test binaries (apply 3, cbm 9, json 6, predict 5 — 23/23 pass) and `cargo clippy -p cb-model --lib` is clean. cb-oracle is not a polars consumer, so the cb-model test profile links without polars.

## Deferred Issues

None within scope.

## Known Stubs

None. Both formats are wired to real model data and oracle-locked against upstream catboost 1.2.10 fixtures; no placeholder/empty data sources. CTR / text / embedding model parts and multiclass leaf dimensions are explicitly OUT OF SCOPE this phase (Phases 5/6 extend the `TModelTrees` sections that this writer leaves absent).

## Threat Flags

None. The two trust boundaries in the plan's threat model (`load_cbm` and `load_json`) are exactly the surfaces implemented, and all five STRIDE entries (T-04-03-01..05) are mitigated as specified: size-bound before slicing, verifying accessor, magic + FormatVersion checks, serde_json safe-by-default, checked `try_from`/`.get` conversions. No NEW network/auth/file surface was introduced.

## Regression Guard

No shared enum/type or `pub` signature was changed (only NEW pub fns + a NEW `ModelError` type + private json structs). All four cb-model test binaries recompile and pass; the only external `ModelError::` match is the new cbm test. No Wave-2-style non-exhaustive-match regression possible.

## Next Phase Readiness

- The `.cbm` / `model.json` substrate is in place for Plan 04 (SHAP / fstr consume `leaf_weights`, now round-tripped in both formats) and Plan 05 (the Builder facade's save/load).
- Phases 5/6 extend the `TModelTrees` sections this writer leaves absent (CTR `CtrFeatures`, `OneHotFeatures`, `TextFeatures`, `EmbeddingFeatures`) and the multi-dim `MultiBias`/`ApproxDimension > 1` path (the reader already prefers `MultiBias[0]`, so the 1-dim seam is forward-compatible).

## Self-Check: PASSED

- Created files verified present: `crates/cb-model/src/cbm.rs`, `crates/cb-model/src/error.rs`, `crates/cb-model/src/json.rs`, `crates/cb-model/tests/cbm_oracle_test.rs`, `crates/cb-model/tests/json_oracle_test.rs`.
- Commits verified present: `a9a3de5` (Task 1), `63ab7a3` (Task 2).
- Tests green: cb-model cbm 9/9, json 6/6, apply 3/3, predict 5/5 (23/23 total).
- Grep gates pass: `FlabuffersModel_v1` present, `FlatbuffersModel_v1` absent; `root_as_tmodel_core` verifying (no `_unchecked`); no `unwrap()`/raw-index in cbm.rs/json.rs production; `cargo clippy -p cb-model --lib` clean.

---
*Phase: 04-model-serialization-shap-rust-api-first-full-oracle-lock*
*Completed: 2026-06-13*
