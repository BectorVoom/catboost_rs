---
phase: 05-ordered-boosting-ordered-ctr-categoricals-high-risk-parity-s
plan: 07
subsystem: testing
tags: [oracle, permutation, ordered-ctr, fisher-yates, tfastrng64, gap-closure, cr-01]

# Dependency graph
requires:
  - phase: 05 (plan 05-03)
    provides: Fisher-Yates fold permutation over TFastRng64 (D-03 linchpin); permutation_fold0.npy anchor
  - phase: 05 (plan 05-05)
    provides: ordered_ctr fixture + ordered_ctr_oracle_test.rs (the per-fold-reseed fold-1 gate this plan corrects)
provides:
  - Continuous-stream multi-fold permutation oracle (GenMultiFoldPermutations) in ordered_oracle.cpp
  - Regenerated permutation_fold1.npy = permutations(30, 2, 0)[1] (upstream-faithful second fold)
  - D-03 fold-1 gate keyed on cb_train::permutations(30, 2, 0)[1], validating production permutations() for k=0 AND k=1
affects: [ordered-ctr, ordered-boosting, fold-machinery, phase-05-verification]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Continuous-stream multi-fold permutation: ONE persistent TFastRng64::FromSeed across all folds (never reseeded per fold), matching upstream learn_context.cpp shared TRestorableFastRng64"

key-files:
  created: []
  modified:
    - crates/cb-oracle/generator/ordered_oracle.cpp
    - crates/cb-oracle/fixtures/ordered_ctr/permutation_fold1.npy
    - crates/cb-oracle/fixtures/ordered_ctr/config.json
    - crates/cb-train/tests/ordered_ctr_oracle_test.rs

key-decisions:
  - "Harness GenMultiFoldPermutations draws ALL folds from one persistent TFastRng64 (single FromSeed; continuous GenRand stream); FisherYatesPermutation kept for the fold-0 single-fold CTR path"
  - "Only permutation_fold1.npy overwritten — fold 0 verified byte-identical; ctr_*/ordered_approx/body_tail fixtures untouched (their committed values derive from uncommitted D-09 inputs)"
  - "D-03 fold-1 gate re-keyed to production permutations(30, 2, 0)[1] (integer-exact compare_permutation, NOT comparator-relaxed); asserts permutations(30,2,0)[0] == fisher_yates_permutation(30,0)"

patterns-established:
  - "Pattern: multi-fold permutation fixtures are continuous-stream (single advancing RNG), not per-fold reseed — fold k = permutations(N, foldCount, seed)[k]"

requirements-completed: [ORD-01, ORD-03]

# Metrics
duration: 9min
completed: 2026-06-14
---

# Phase 5 Plan 07: Continuous-Stream Multi-Fold Permutation Oracle (CR-01) Summary

**Fixed the multi-fold permutation oracle to draw all folds from one continuously-advancing TFastRng64, regenerated permutation_fold1.npy as the upstream-faithful second fold, and re-keyed the D-03 fold-1 gate to validate the production `permutations(30, 2, 0)[1]` integer-exact — closing the CR-01 BLOCKER where fold k>0 was never validated against upstream.**

## Performance

- **Duration:** ~9 min
- **Started:** 2026-06-14T08:09Z (approx)
- **Completed:** 2026-06-14
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Replaced the per-fold reseed defect (`foldSeed = seed + k`, old lines 391-393) with `GenMultiFoldPermutations(n, foldCount, seed)` — a single `TFastRng64::FromSeed(seed)` whose Fisher-Yates draw stream advances continuously across folds, never re-seeded. Mirrors `permutation.rs::permutations` / `fold.rs::create_folds` and upstream `learn_context.cpp`'s shared `TRestorableFastRng64`.
- Regenerated `permutation_fold1.npy` from the corrected continuous-stream harness; verified fold 0 is **byte-identical** before/after (only fold 1 overwritten) and fold 1 now differs from the old per-fold-reseed bytes.
- Re-keyed the D-03 fold-1 gate in `ordered_ctr_oracle_test.rs` to `cb_train::permutations(30, 2, 0)[1]` (integer-exact `compare_permutation`), and added the assertion `permutations(30, 2, 0)[0] == fisher_yates_permutation(30, 0)` (continuous-stream fold 0 == single-fold fold 0). The production `permutations()` is now validated for k=0 AND k=1, removing the silent self-consistency-only hole for k>0.
- All 3 `ordered_ctr_oracle_test` tests green; harness compiles with `g++ -std=c++20`, no `seed + (ui64)k` reseed remains, zero catboost includes.

## Task Commits

Each task was committed atomically:

1. **Task 1: Continuous-stream multi-fold seeding + regenerate permutation_fold1.npy** - `1f2293f` (fix)
2. **Task 2: Re-key D-03 fold-1 gate to permutations(30, 2, 0)[1]** - `adc6a03` (test)

## Files Created/Modified
- `crates/cb-oracle/generator/ordered_oracle.cpp` - Added `ShuffleInPlace` (over an already-seeded generator) + `GenMultiFoldPermutations` (single persistent RNG across folds); replaced the per-fold reseed loop in `main`. `FisherYatesPermutation` retained for the fold-0 single-fold CTR path.
- `crates/cb-oracle/fixtures/ordered_ctr/permutation_fold1.npy` - Regenerated as the continuous-stream second fold (`permutations(30, 2, 0)[1]`).
- `crates/cb-oracle/fixtures/ordered_ctr/config.json` - Added `fold_seeding` note documenting the continuous-stream discipline.
- `crates/cb-train/tests/ordered_ctr_oracle_test.rs` - Fold-1 gate re-keyed to `permutations(30, 2, 0)[1]`; doc-comment rewritten to record CR-01 closure; both fold gates use integer-exact `compare_permutation`.

## Decisions Made
- **Only `permutation_fold1.npy` overwritten.** Fold 0 was confirmed byte-identical to the existing committed fixture before any write, so it was left untouched. The `ctr_good_count` / `ctr_total_count` / `ctr_value` / `ordered_approx_iter0` fixtures derive from the uncommitted (D-09) `cat_bin` / `target_class` / `der` stdin inputs; they were NOT regenerated/overwritten (my regeneration used placeholder inputs purely to extract fold 1 — the permutations depend only on N/foldCount/seed, not on those streams).
- **`FisherYatesPermutation` kept** for the fold-0 single-fold CTR derivation (`perm0`), since fold 0 of the continuous stream equals a fresh-seed single fold — the existing online-CTR path is unchanged.
- **No comparator relaxation.** Both fold gates remain integer-exact `compare_permutation` (`Stage::Permutation`), not `compare_stage` 1e-5.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None. The continuous-stream discipline was already established and documented in the production `permutation.rs`, so the harness fix transcribed it directly; fold-0 byte-identity and the production-vs-fixture integer-exact match for both folds were confirmed by the green test run.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- CR-01 BLOCKER closed: the D-03 contract ("reproduce upstream permutations exactly") now holds for fold k ≥ 1, not just fold 0.
- ORD-01 multi-permutation fold machinery is validated against the upstream-faithful continuous-stream seeding; the production `permutations()` is no longer self-consistency-only for k>0.
- Remaining Phase 5 gap-closure plans (05-08, 05-09) are independent of this fix.

## Self-Check: PASSED

- FOUND: `crates/cb-oracle/generator/ordered_oracle.cpp`
- FOUND: `crates/cb-oracle/fixtures/ordered_ctr/permutation_fold1.npy`
- FOUND: `crates/cb-oracle/fixtures/ordered_ctr/config.json`
- FOUND: `crates/cb-train/tests/ordered_ctr_oracle_test.rs`
- FOUND commit: `1f2293f` (Task 1)
- FOUND commit: `adc6a03` (Task 2)

---
*Phase: 05-ordered-boosting-ordered-ctr-categoricals-high-risk-parity-s*
*Completed: 2026-06-14*
