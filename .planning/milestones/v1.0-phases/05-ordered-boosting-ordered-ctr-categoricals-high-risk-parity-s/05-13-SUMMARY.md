---
phase: 05-ordered-boosting-ordered-ctr-categoricals-high-risk-parity-s
plan: 13
subsystem: training
tags: [ordered-ctr, ctr-feature, two-materialization, averaging-fold, leaf-values, oblivious-search, parity]

# Dependency graph
requires:
  - phase: 05-12
    provides: identity-Folds[0] create_folds + AveragingFold first-seeded draw (find(|f| f.is_averaging)); integer-exact averaging-permutation oracle
  - phase: 05-11
    provides: train_cat + materialize_ctr_feature / CtrFeatureColumn (combined-projection online CTR); FOLDS-BUILT-ONCE
  - phase: 05-08/05-10
    provides: greedy_tensor_search_oblivious_ordered + Plain/Ordered dispatch in train_inner
provides:
  - "greedy_tensor_search_oblivious_with_ctr: CTR-aware oblivious search scoring CtrFeatureColumn candidates alongside float (shared l2_split_score, strict first-wins, forward-bit leaf index), recording the winning CtrSplitSpec"
  - "GrownTree.ctr_splits + GrownTree.level_kinds (LevelKind::{Float,Ctr}) carrying the chosen CTR splits and per-level kind for forward-bit + averaging-fold reassignment"
  - "train_inner two-materialization: structure CTR column under the identity learning fold + a SECOND leaf-value CTR column under the AveragingFold shuffled permutation; leaf_of + leaf_weights for leaf VALUES computed from the averaging-fold column via assign_leaf_of_averaging"
affects: [05-14, ordered-ctr, leaf-value-materialization, tensor_ctr_e2e]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Two-materialization CTR: structure search scores the IDENTITY-fold CTR column; leaf VALUES are estimated on a SECOND CTR column materialized under the AveragingFold's SHUFFLED permutation (research Q1/Q3, train.cpp:130 BuildIndices(AveragingFold)) — the Gradient leaf FORMULA is unchanged"
    - "CTR-aware oblivious search: FLOAT-then-CTR fixed candidate order, strict > best first-wins, one CTR candidate per border 0..ctr_border_count, recorded as CtrSplitSpec + LevelKind"

key-files:
  created:
    - crates/cb-train/tests/ctr_split_scoring_test.rs
  modified:
    - crates/cb-train/src/tree.rs
    - crates/cb-train/src/boosting.rs
    - crates/cb-train/src/lib.rs

key-decisions:
  - "CTR-aware search is a NEW variant (greedy_tensor_search_oblivious_with_ctr) reusing the shared l2_split_score/reduce_leaf_stats (NOT a forked scorer); the chosen splits split back into parallel splits/ctr_splits with a level_kinds map so float + CTR levels interleave in forward-bit order"
  - "Leaf VALUES reassigned over the averaging-fold CTR column (assign_leaf_of_averaging) keyed by projection match against the index-aligned averaging columns; the Gradient leaf formula is untouched (research Q3 #4)"
  - "The averaging-fold reassignment is GATED on has_ctr (non-empty materialized CTR features); off the CTR path leaf_value_leaf_of == grown.leaf_of byte-identical, so numeric/one-hot/ordered oracles are provably unaffected"
  - "The chosen grown.ctr_splits are persisted onto ObliviousTree on the CTR path (replacing the candidate-only ctr_splits_for_tree emission); ctr_splits_for_tree retained for the no-CTR path"

patterns-established:
  - "Structure = identity learning fold partition [6,0,9,15]; leaf VALUES = averaging fold partition [6,0,7,17]; the two-materialization distinction the research reproduced bit-exact"

requirements-completed: []

# Metrics
duration: ~30min
completed: 2026-06-14
---

# Phase 5 Plan 13: Two-Materialization CTR Leaf Values Summary

**The CTR-aware oblivious search scores the identity-fold CTR column into the tree STRUCTURE (shared L2 score, strict first-wins, forward-bit leaf index, recorded as a `CtrSplitSpec`), and a SECOND CTR column materialized under the AveragingFold's shuffled permutation drives the per-object `leaf_of` + `leaf_weights` for LEAF-VALUE estimation — the unchanged Gradient formula over the averaging partition reproduces tree0 `[-0.033333, 0, -0.005, 0.0275]` ≤1e-5.**

## Performance

- **Duration:** ~30 min
- **Completed:** 2026-06-14
- **Tasks:** 2
- **Files modified:** 4 (3 modified, 1 created)

## Accomplishments

- **Task 1 — CTR scored into the oblivious search (STRUCTURE):** `greedy_tensor_search_oblivious_with_ctr` in `tree.rs` enumerates FLOAT candidates (unchanged order) THEN one CTR candidate per border `0..ctr_border_count` for each materialized `CtrFeatureColumn`, scoring EVERY candidate with the SAME `l2_split_score` over `reduce_leaf_stats` the float path uses (no forked math), with the strict first-wins (`> best`, never `>=`) over the fixed FLOAT-then-CTR order. A winning CTR split is recorded as a `CtrSplitSpec` carrying the chosen CTR-value border + the column's prior num/denom PAIR + projection + ctr_type. `GrownTree` gained `ctr_splits: Vec<CtrSplitSpec>` and `level_kinds: Vec<LevelKind>` (default empty for the float-only / one-hot / ordered searches, so their consumers compile unchanged); `level_kinds` records each level's kind (`Float(idx)` / `Ctr{ctr_idx,border}`) so the forward-bit leaf index assigns CTR bits (`ctr_bin > border`) and float bits in the correct level order. The tensor_ctr_e2e-style single-feature column reproduces the structure partition `[6,0,9,15]` (leaf1 empty — both levels split the same single CTR feature).
- **Task 2 — second averaging-fold materialization (LEAF VALUES):** `train_inner` now pulls BOTH the structure permutation (the lone learning `Folds[0]`, identity for pc=1 after 05-12) AND the AveragingFold's shuffled permutation (`find(|f| f.is_averaging)`) from ONE `create_folds` call, materializes a SECOND CTR column per projection under the averaging permutation (the SAME `materialize_ctr_feature`, the averaging permutation input), grows the STRUCTURE via `greedy_tensor_search_oblivious_with_ctr` over the identity-fold columns, then REASSIGNS the per-object `leaf_of` over the averaging-fold columns (`assign_leaf_of_averaging`, keyed by projection match) for leaf-VALUE estimation. `compute_leaf_deltas` (Gradient) + `accumulate_leaf_weights` + the per-iteration approx update all run over this averaging-fold `leaf_of`; the leaf FORMULA is untouched. The chosen `grown.ctr_splits` are persisted onto `ObliviousTree`.
- **Numeric path provably unaffected:** the averaging-fold reassignment is gated on `has_ctr` (non-empty materialized CTR features). Off the CTR path `leaf_value_leaf_of == grown.leaf_of` byte-identical, and the Plain/Ordered dispatch is unchanged — the numeric / one-hot / ordered / leaf-methods oracles stay byte-for-byte (all green).
- **Research result reproduced in tests:** `second_materialization_differs_from_structure` proves the identity-fold vs averaging-fold CTR columns differ; `averaging_partition_reproduces_tree0_leaf_values` proves the unchanged Gradient formula over the `[6,0,7,17]` averaging partition reproduces tree0 `[-0.033333, 0, -0.005, 0.0275]` ≤1e-5.

## Task Commits

1. **Task 1: Score the structure-search CTR column into the oblivious search** — `3b07352` (feat)
2. **Task 2: Second averaging-fold materialization + leaf values** — `b16ac28` (feat)

## Files Created/Modified

- `crates/cb-train/src/tree.rs` — NEW `greedy_tensor_search_oblivious_with_ctr` + `CtrAwareSplit` (internal) + `assign_leaves_ctr_aware` / `score_candidate_ctr_aware` / `select_level_ctr_aware`; `GrownTree` extended with `ctr_splits` + `level_kinds`; NEW `LevelKind` enum. Existing float-only / ordered `GrownTree` constructions updated with empty defaults.
- `crates/cb-train/src/boosting.rs` — `train_inner` two-materialization: capture both `cat_learn_permutation` + `cat_averaging_permutation` from one `create_folds`; materialize structure (`materialized_ctr_features`) + averaging (`averaging_ctr_features`) columns over shared `absolute_projections`; CTR-aware structure search gated on `has_ctr`; NEW `assign_leaf_of_averaging`; `leaf_value_leaf_of` drives `compute_leaf_deltas` / `accumulate_leaf_weights` / approx update; persist `grown.ctr_splits`.
- `crates/cb-train/src/lib.rs` — export `greedy_tensor_search_oblivious_with_ctr` + `LevelKind`.
- `crates/cb-train/tests/ctr_split_scoring_test.rs` — NEW (6 tests): Task 1 CTR-wins / tie-break / forward-bit / structure-partition `[6,0,9,15]`; Task 2 second-materialization-differs / averaging-partition leaf-value tree0 `[-0.033333,0,-0.005,0.0275]` ≤1e-5.

## Decisions Made

- **CTR-aware search reuses the shared L2 scorer** (`l2_split_score` over `reduce_leaf_stats`) — no forked scoring math; the chosen unified splits split back into parallel `splits` / `ctr_splits` with a `level_kinds` map.
- **Leaf VALUES reassigned over the averaging-fold column** keyed by projection match against the index-aligned averaging columns; the Gradient leaf formula is untouched (research Q3 #4 — the formula was already correct).
- **`has_ctr` gate** keeps the numeric / one-hot / ordered paths byte-identical (leaf_value_leaf_of == structure leaf_of off the CTR path).
- **Chosen `grown.ctr_splits` persisted** onto `ObliviousTree` (replacing the candidate-only emission); `ctr_splits_for_tree` retained for the no-CTR path (returns empty there).

## Deviations from Plan

None — plan executed exactly as written. Tasks 1 and 2 delivered the CTR-aware structure search and the two-materialization leaf-value path with the planned source assertions and verify commands all green.

## Deferred Issues

These are PRE-EXISTING failures present at the parent commit `0f603a1` (verified by checkout), NOT caused by this plan, and explicitly OUT OF SCOPE for 05-13 (the plan states "The ctr_data bake + apply Scale/Shift + e2e hard gate are Plan 05-14"). Neither is in this plan's verify list.

- **`tensor_ctr_e2e_oracle_test::tensor_ctr_e2e_oracle_predictions_match_upstream`** — the FULL multi-tree e2e gate. It calls `train` (NOT `train_cat`) with empty `cat_columns`, so it has no CTR candidates and grows a CTR-less, border-less tree → `Degenerate("no candidate split available")`. Plan 05-14 rewires this gate to `train_cat` + bakes the `ctr_data` / Scale-Shift through `apply.rs`. Failing identically before and after this plan.
- **`ordered_boost_wiring_test::ordered_structure_differs_from_plain`** — an Ordered-path falsifiability test asserting Ordered structure ≠ Plain. After 05-12's identity-`Folds[0]` change, Ordered structure on permutation_count=1 runs on the identity fold and matches Plain for this dataset, so the assertion no longer holds. Unrelated to CTRs; failing identically before this plan. (Note: `ordered_boost_e2e_oracle_test` — the real ORD-02 ≤1e-5 gate — stays GREEN.)

## Known Stubs

None. The structure CTR-value borders are the chosen `0..ctr_border_count` thresholds (real, not placeholder); the chosen `CtrSplitSpec.border` is the structure threshold (Plan 05-14 reconciles Scale/Shift so apply compares in the same space — documented, not a stub).

## Threat Flags

None — no new network/auth/file surface; the scoring test is synthetic + self-consistent (no committed fixture touched). The threat register's `mitigate` dispositions are satisfied: the averaging vs structure partition swap is caught by `second_materialization_differs_from_structure` + the `[6,0,7,17]`-vs-`[6,0,9,15]` partition tests; the float-then-CTR strict first-wins tie-break is locked by `tie_break_float_then_ctr_first_wins`; bins/leaf indices use checked `.get` only.

## Next Phase Readiness

- The two-materialization leaf-value path is in place: structure on the identity fold, leaf VALUES on the averaging fold, the chosen `CtrSplitSpec` set persisted onto `ObliviousTree`.
- Plan 05-14 picks up the `build_final_ctr` `ctr_data` bake + thread `Scale`/`Shift` through `apply.rs` and rewires the `tensor_ctr_e2e` hard gate to `train_cat` for the FULL multi-tree ≤1e-5 closure.

## Self-Check: PASSED

- FOUND: `crates/cb-train/tests/ctr_split_scoring_test.rs`
- FOUND: `crates/cb-train/src/tree.rs`
- FOUND: `crates/cb-train/src/boosting.rs`
- FOUND commit: `3b07352` (Task 1)
- FOUND commit: `b16ac28` (Task 2)

---
*Phase: 05-ordered-boosting-ordered-ctr-categoricals-high-risk-parity-s*
*Completed: 2026-06-14*
