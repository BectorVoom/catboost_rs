---
title: Fix builder_oracle_test facade/fixture score-function mismatch
date: 2026-06-19
priority: high
blocked_by: none (unblocked 2026-06-19 — isolation DONE)
direction: expose .score_function() setter + test opts into L2
---

# Fix builder_oracle_test (facade Cosine vs fixture L2)

Resolve the long-standing `builder_oracle_test` failure
(`builder_regression_full_cycle` + `builder_binclf_full_cycle`, diverge at
Predictions). Root cause confirmed: the facade trains with `score_function=Cosine`
(catboost's true CPU default, no public setter) while the `model_serde/{binclf,
regression}` fixtures were generated with `score_function=L2`. Full analysis:
`.planning/notes/builder-oracle-score-function-root-cause.md`.

## Gate — CLEARED

Isolation done (`/gsd-debug borders-vs-score-fn-builder`, 2026-06-19): **score
function is the sole cause; borders are benign.** No border reconciliation needed.

## Chosen fix (VERIFIED direction)

**Expose a `.score_function(EScoreFunction)` setter** on `CatBoostBuilder`; have
`builder_oracle_test`'s `configured_builder` call `.score_function(EScoreFunction::L2)`
to match the fixtures. Keep `score_function_default() = Cosine` (catboost CPU default).

Implementation sketch:
- Add a `score_function: EScoreFunction` field to `CatBoostBuilder`, defaulting to
  `score_function_default()` (Cosine), with a `.score_function(self, v)` builder
  method. Wire it through `boost_params()` (replaces the hardcoded
  `score_function_default()` at `builder.rs:286`).
- `EScoreFunction` is in `cb_compute`; re-export it from `catboost-rs` so callers can
  name `L2` without depending on `cb-compute` directly.
- In `builder_oracle_test.rs::configured_builder`, add `.score_function(L2)`.
- While here, fix the **false docstring** in `builder_oracle_test.rs` claiming
  computed borders match upstream for `numeric_tiny` (they don't — 49 vs 2/2/0/3 —
  just benign).

Expected result: both `builder_*_full_cycle` converge ≤1e-5 (measured 2.4e-8 /
2.8e-9 in the isolation run).

Rejected alternative: regenerate fixtures to Cosine — loses the L2 oracle lock that
per-crate cb-train/cb-compute oracles already use, and re-pins multiple artifacts.

## Done when

- Both `builder_*_full_cycle` tests pass ≤1e-5 through the public facade only.
- The chosen direction is recorded; memory note
  `builder-oracle-test-preexisting-failure` updated to "resolved".
