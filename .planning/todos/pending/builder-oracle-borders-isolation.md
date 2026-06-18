---
title: Isolate whether pool-computed borders also diverge in builder_oracle_test
date: 2026-06-19
priority: high
blocks: builder-oracle-fix
---

# Isolate score-function vs borders for builder_oracle_test

The facade Cosine-vs-fixture-L2 score-function mismatch is the confirmed prime cause
of the `builder_oracle_test` prediction divergence (see
`.planning/notes/builder-oracle-score-function-root-cause.md`). But it is **not
proven to be the sole cause**. Verify the borders are not a second source of drift
**before** picking a fix direction.

## Goal

Empirically isolate the divergence into score-function and/or borders.

## Approach (cheapest first)

1. **Borders check (read-only-ish):** compare the facade's
   `select_borders_greedy_logsum(col, 254, false)` output for `numeric_tiny`'s
   columns against the pinned `float_feature_borders` in
   `crates/cb-oracle/fixtures/model_serde/regression/model.json`. If they match
   bit-for-bit, borders are exonerated and the docstring claim holds.
2. **Score-function check:** add a temporary `score_function` hook to the facade (or
   call `cb_train::train` directly with `EScoreFunction::L2` + pinned borders) and
   re-run the full cycle. If predictions converge to ≤1e-5, score function is the
   sole cause.

## Done when

- We can state which of {score_function, borders} drives the divergence and by how
  much, with evidence.
- The fix direction (regenerate-to-Cosine vs expose-L2-setter) is chosen on that
  evidence and recorded in `builder-oracle-fix`.

## Note on environment

Root disk ~100% full; cb-compute test profile may not link (see memory
`disk-pressure-and-full-suite-verification`). Prefer per-crate / minimal harness
over full-workspace builds. Consider running this as `/gsd-spike` or `/gsd-debug`.
