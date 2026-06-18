---
title: Fix builder_oracle_test facade/fixture score-function mismatch
date: 2026-06-19
priority: high
blocked_by: builder-oracle-borders-isolation
---

# Fix builder_oracle_test (facade Cosine vs fixture L2)

Resolve the long-standing `builder_oracle_test` failure
(`builder_regression_full_cycle` + `builder_binclf_full_cycle`, diverge at
Predictions). Root cause confirmed: the facade trains with `score_function=Cosine`
(catboost's true CPU default, no public setter) while the `model_serde/{binclf,
regression}` fixtures were generated with `score_function=L2`. Full analysis:
`.planning/notes/builder-oracle-score-function-root-cause.md`.

## Gate

**Do NOT start until `builder-oracle-borders-isolation` confirms whether borders are
a second source of drift.** The chosen direction depends on that result.

## Candidate fixes

- **A — Regenerate fixtures to Cosine** (preferred if score-fn is sole cause):
  re-train `model_serde/{binclf,regression}` upstream with `score_function=Cosine`
  and refresh `predictions.npy` + `config.json`. Locks the facade's correct default;
  no public-API change. Requires the instrumented/upstream catboost trainer (see
  memory `catboost-instrumented-trainer-build` /
  `instrumented-trainer-toolchain-persists`).
- **B — Expose `.score_function()` setter** on `CatBoostBuilder` and pin `L2` in the
  test. Keeps fixtures; widens public API; oracle-locks a non-default config.

If the borders ALSO diverge, the fix must additionally reconcile border selection
(facade pool-computed vs fixture-pinned) — likely fold the fixture's pinned borders
into the regeneration so both come from one upstream run.

## Done when

- Both `builder_*_full_cycle` tests pass ≤1e-5 through the public facade only.
- The chosen direction is recorded; memory note
  `builder-oracle-test-preexisting-failure` updated to "resolved".
