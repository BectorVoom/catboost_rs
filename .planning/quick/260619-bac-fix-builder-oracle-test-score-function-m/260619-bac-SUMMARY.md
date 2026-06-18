---
phase: quick-260619-bac
plan: 01
subsystem: catboost-rs (public Builder facade)
tags: [builder, oracle, score-function, RAPI-01]
requires:
  - cb_compute::EScoreFunction
  - cb_train::BoostParams.score_function
provides:
  - CatBoostBuilder::score_function (public setter)
  - catboost_rs::EScoreFunction (re-export)
affects:
  - crates/catboost-rs/src/builder.rs
  - crates/catboost-rs/src/lib.rs
  - crates/catboost-rs/tests/builder_oracle_test.rs
tech-stack:
  added: []
  patterns: ["#[must_use] consuming-self builder setter", "single-sourced default via score_function_default()"]
key-files:
  created: []
  modified:
    - crates/catboost-rs/src/builder.rs
    - crates/catboost-rs/src/lib.rs
    - crates/catboost-rs/tests/builder_oracle_test.rs
decisions:
  - "Default builder behavior stays Cosine via score_function_default(); only the test opts into L2."
  - "EScoreFunction re-exported from catboost-rs (via cb_compute) so callers never depend on cb-compute directly."
metrics:
  duration: ~8 min
  completed: 2026-06-19
  tasks: 2
  files: 3
---

# Phase quick-260619-bac Plan 01: Fix builder_oracle_test score-function mismatch Summary

Exposed a public `.score_function(EScoreFunction)` setter on `CatBoostBuilder`, re-exported `EScoreFunction` from the published crate, and had the oracle test opt into `L2` to match its upstream fixtures — closing the last red oracle on the public facade (both `builder_regression_full_cycle` and `builder_binclf_full_cycle` now pass ≤1e-5).

## What Was Built

**Task 1 — Expose `.score_function()` and re-export `EScoreFunction` (commit `f8617f6`)**
- `builder.rs`: added `EScoreFunction` to the `cb_compute` import block; added a `score_function: EScoreFunction` field to `CatBoostBuilder`; initialized it in `new()` via `score_function_default()` (Cosine preserved, single-sourced); added a `#[must_use] pub fn score_function(...)` setter matching the existing setter style; replaced the hardcoded `score_function: score_function_default()` in `boost_params()` with `score_function: self.score_function` and updated the comment.
- `lib.rs`: extended the `pub use cb_compute::{...}` re-export to include `EScoreFunction` and updated the adjacent doc comment to mention the score-function knob.

**Task 2 — Opt the test into L2 + fix the false docstring (commit `3ff254d`)**
- `builder_oracle_test.rs`: imported `EScoreFunction` from `catboost_rs` (proving the re-export); added `.score_function(EScoreFunction::L2)` to `configured_builder`; corrected the module/inline docstrings that falsely claimed the facade's computed greedy-logsum borders "reproduce upstream's border selection exactly" — they DIFFER from the fixture's pinned counts but cancel through the trained tree structure once `score_function = L2`; added `score_function=L2` to the documented config summary.

## Verification

- `cargo build -p catboost-rs` — succeeds.
- `cargo test -p catboost-rs --test builder_oracle_test`:
  ```
  running 2 tests
  test builder_regression_full_cycle ... ok
  test builder_binclf_full_cycle ... ok
  test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.26s
  ```
  Both pass the upstream ≤1e-5 oracle leg (Stage::Predictions). The disk-pressure link-failure caveat did NOT materialize — the targeted test binary linked and ran in-env. No false pass claimed; this is a real green.
- `grep -nE '\.unwrap\(\)|\.expect\(|panic!' crates/catboost-rs/src/builder.rs` → CLEAN (no banned constructs added to production builder).
- `cb_train::score_function_default()` unchanged (still Cosine). A bare `CatBoostBuilder::new()` remains Cosine.
- `EScoreFunction` reachable as `catboost_rs::EScoreFunction`.

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None.

## Memory Note Update

`builder-oracle-test-preexisting-failure` updated to RESOLVED (score-function setter exposed; test opts into L2; oracle green), per the todo's "Done when" clause.

## Self-Check: PASSED

- Files: builder.rs, lib.rs, builder_oracle_test.rs all FOUND.
- Commits: f8617f6 (feat), 3ff254d (test) both FOUND in git log.
