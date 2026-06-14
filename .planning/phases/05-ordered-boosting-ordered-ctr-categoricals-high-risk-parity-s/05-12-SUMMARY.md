---
phase: 05-ordered-boosting-ordered-ctr-categoricals-high-risk-parity-s
plan: 12
subsystem: testing
tags: [ordered-ctr, fold-permutation, fisher-yates, tfastrng64, draw-order, oracle, parity]

# Dependency graph
requires:
  - phase: 05-11
    provides: cat-aware train_cat + materialize_ctr_feature/CtrFeatureColumn; FOLDS-BUILT-ONCE (create_folds at 2 sites in boosting.rs)
  - phase: 05-03/05-07
    provides: Fisher-Yates fold permutation over TFastRng64 + continuous-stream permutations() (D-03 linchpin)
provides:
  - "create_folds builds Folds[0] as the IDENTITY (zero RNG draws) and the AveragingFold as the first seeded Fisher-Yates draw when a learning permutation is needed (hasCtrs OR ordered boosting), matching upstream shuffle = foldIdx != 0"
  - "permutation::shuffle_in_place exposed pub(crate) so create_folds drives a single persistent TFastRng64 directly"
  - "standalone integer-exact AveragingFold-permutation draw-order oracle (averaging == fisher_yates_permutation(30,0); learning fold == identity) — the D-03-style linchpin gating Plan 05-13 leaf-value materialization"
affects: [05-13, 05-14, ordered-ctr, leaf-value-materialization, tensor_ctr_e2e]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Identity-Folds[0] draw discipline: when a learning permutation is needed, Folds[0] consumes ZERO RNG draws and the averaging fold is the first seeded draw (single persistent TFastRng64 driven directly, never reseeded per fold)"
    - "Self-consistent draw-order oracle: expected permutation derived from the PRODUCTION fisher_yates_permutation (no committed .npy fixture), integer-exact via Stage::Permutation"

key-files:
  created:
    - crates/cb-train/tests/averaging_fold_permutation_oracle_test.rs
  modified:
    - crates/cb-train/src/fold.rs
    - crates/cb-train/src/fold_test.rs
    - crates/cb-train/src/permutation.rs

key-decisions:
  - "create_folds splits into two branches: !permutation_needed_for_learning keeps the legacy continuous-stream permutations() draws byte-identical (numeric regression anchor); the learning-permutation-needed branch drives a single persistent TFastRng64 directly (identity Folds[0] + first-seeded averaging fold)"
  - "shuffle_in_place exposed pub(crate) (not a new public symbol); public permutations / fisher_yates_permutation API unchanged"
  - "Oracle is self-consistent (derived from production fisher_yates_permutation, NOT a committed fixture) — reconciles the STATE.md 05-12 blocker note: post-fix the structure (learning) fold is identity and the averaging fold takes the first draw = fisher_yates(30,0)"

patterns-established:
  - "Identity-Folds[0] (shuffle = foldIdx != 0) on the learning-permutation-needed path; averaging fold is the leaf-value permutation, identity learning fold is the structure-search permutation (research two-materialization roles)"

requirements-completed: [ORD-05]

# Metrics
duration: ~16min
completed: 2026-06-14
---

# Phase 5 Plan 12: AveragingFold Draw-Order De-Risk Summary

**Identity-`Folds[0]` (zero RNG draws) `create_folds` + a standalone integer-exact oracle locking the AveragingFold permutation to `fisher_yates_permutation(30,0)` — the D-03-style linchpin that aligns cb-train's TFastRng64 draw stream with upstream before any leaf-value stage runs.**

## Performance

- **Duration:** ~16 min
- **Started:** 2026-06-14T15:57:00Z (approx)
- **Completed:** 2026-06-14
- **Tasks:** 2
- **Files modified:** 4 (3 modified, 1 created)

## Accomplishments

- **`create_folds` reworked (Task 1):** WHEN a learning permutation is needed (`hasCtrs` OR ordered boosting), the lone learning `Folds[0]` is now the IDENTITY `[0..n]` consuming ZERO RNG draws (upstream `shuffle = foldIdx != 0`, `learn_context.cpp:524` / `fold.cpp:54`), and every subsequent fold — including the AveragingFold — takes one Fisher-Yates draw IN ORDER from a single persistent `TFastRng64::from_seed(seed)`. For `permutation_count=1` the averaging-fold permutation byte-equals `fisher_yates_permutation(n, seed)` (it is the first seeded draw). The numeric / Plain-no-CTR path keeps the legacy continuous-stream draws byte-identical.
- **`shuffle_in_place` exposed `pub(crate)`** so `create_folds` can drive the held rng directly; the public `permutations` / `fisher_yates_permutation` API is unchanged.
- **Draw-order contract documented** on `create_folds` (identity Folds[0] / averaging = first seeded draw).
- **Standalone integer-exact oracle authored (Task 2):** `averaging_fold_permutation_oracle_test.rs` locks the `tensor_ctr_e2e` config (N=30, seed=0, permutation_count=1, hasCtrs) — the AveragingFold permutation == `fisher_yates_permutation(30,0)` index-for-index via `cb_oracle::compare_permutation` (`Stage::Permutation`), the learning fold == identity, and the two permutations are distinct. Runs unconditionally, NO `#[ignore]`, no committed fixture touched.
- **ORD-02 not regressed:** the `ordered_boost_e2e` oracle (and slice_first / one_hot / leaf_methods / ctr_feature_materialize) stay green under the identity-Folds[0] change — upstream's ordered structure search also runs on the identity Folds[0] for permutation_count=1.

## Task Commits

1. **Task 1: Identity Folds[0] + averaging fold first seeded draw** - `c0c790d` (fix) — TDD RED tests added to `fold_test.rs`, GREEN via reworked `create_folds` + `pub(crate) shuffle_in_place`; committed as a single fix commit (RED tests and GREEN impl land together in the sibling unit-test file).
2. **Task 2: Lock the AveragingFold permutation draw order (integer-exact)** - `28507d8` (test)

## Files Created/Modified

- `crates/cb-train/src/fold.rs` - `create_folds` reworked: identity Folds[0] / first-seeded averaging fold on the learning-permutation-needed path; legacy continuous-stream draws preserved on the numeric path; new private `build_fold` helper; documented draw-order contract.
- `crates/cb-train/src/permutation.rs` - `shuffle_in_place` made `pub(crate)` (doc updated).
- `crates/cb-train/src/fold_test.rs` - new draw-order unit tests (identity Folds[0], averaging-is-first-draw, multi-permutation note); the legacy continuous-stream test re-keyed to the numeric (`needed=false`) path.
- `crates/cb-train/tests/averaging_fold_permutation_oracle_test.rs` - NEW standalone integer-exact AveragingFold draw-order oracle (3 tests).

## Decisions Made

- **Two-branch `create_folds`:** the numeric / Plain-no-CTR path (`!permutation_needed_for_learning`) keeps the exact legacy `permutations()` continuous-stream draws (byte-identical regression anchor); only the learning-permutation-needed path adopts the identity-Folds[0] + first-seeded-averaging draw discipline.
- **`shuffle_in_place` `pub(crate)`, not public:** avoids widening the public API; `permutations` / `fisher_yates_permutation` are unchanged.
- **Self-consistent oracle:** the expected permutation is derived from the production `fisher_yates_permutation`, not a committed `.npy`, so no fixture is touched and the gate cannot drift from the production RNG.
- **STATE.md 05-12 blocker note reconciled:** post-fix the STRUCTURE (learning) fold is the identity and the AVERAGING fold takes the first draw = `fisher_yates(30,0)`; the byte sequence the OLD all-shuffle scheme assigned to learning fold0 is now the averaging fold's permutation.

## Deviations from Plan

None - plan executed exactly as written.

The plan flagged a possible CRITICAL regression in the ORD-02 ordered path and the cat-CTR learn permutation (both previously took the first non-averaging fold, which now becomes the identity for permutation_count=1). As the plan anticipated, the `ordered_boost_e2e` oracle stayed GREEN — confirming upstream's ordered structure search also runs on the identity `Folds[0]` and the fix did not regress ORD-02. No revert or ordered-path change was needed.

## Issues Encountered

None. The TDD RED phase correctly showed the two new draw-order tests failing against the old all-shuffle `create_folds` (learning fold0 = `fisher_yates(30,0)` = `[8,12,5,…]`, exactly the byte sequence that becomes the averaging fold's permutation post-fix), and the GREEN rework turned them green without disturbing any oracle.

## Known Stubs

None. This plan is a draw-order de-risk gate; no UI/data stubs introduced.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- The AveragingFold draw order is now locked integer-exact — Plan 05-13 can pull the averaging-fold permutation (`find(|f| f.is_averaging)`) for the leaf-value materialization knowing it byte-matches upstream.
- The identity learning `Folds[0]` is the structure-search permutation; the averaging fold is the leaf-value permutation (the two-materialization split the research identified).
- No e2e gate yet (that is 05-14); this plan closes only the ORD-05 de-risk gate.

## Self-Check: PASSED

- FOUND: `crates/cb-train/tests/averaging_fold_permutation_oracle_test.rs`
- FOUND: `crates/cb-train/src/fold.rs`
- FOUND: `.planning/phases/05-…/05-12-SUMMARY.md`
- FOUND commit: `c0c790d` (Task 1)
- FOUND commit: `28507d8` (Task 2)

---
*Phase: 05-ordered-boosting-ordered-ctr-categoricals-high-risk-parity-s*
*Completed: 2026-06-14*
