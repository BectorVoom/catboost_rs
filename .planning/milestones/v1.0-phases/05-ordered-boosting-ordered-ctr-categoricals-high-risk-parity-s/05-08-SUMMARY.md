---
phase: 05-ordered-boosting-ordered-ctr-categoricals-high-risk-parity-s
plan: 08
subsystem: cpu-training-ordered-boosting
tags: [ordered-boosting, split-scoring, tree-structure, ORD-02, WR-01, parity]
requires:
  - "cb-train::fold::body_tail_segments / body_sum_weights (05-03)"
  - "cb-train::fold permutation conventions (05-07 / CR-01)"
  - "cb-compute::{reduce_leaf_stats, l2_split_score, scale_l2_reg, LeafStats}"
provides:
  - "cb_train::greedy_tensor_search_oblivious_ordered (ordered per-segment split search)"
  - "tree.rs::score_candidate_ordered / ordered_segment_leaf_stats / select_level_ordered (private helpers)"
affects:
  - "05-10 (wires greedy_tensor_search_oblivious_ordered into train() + e2e ordered oracle)"
tech-stack:
  added: []
  patterns:
    - "segment-summed ordered L2 score over the learning fold's BodyTailArr (scoring.cpp:746-760)"
    - "per-segment scaledL2 = l2 * (BodySumWeight / BodyFinish) via scale_l2_reg"
    - "strict first-wins (>) tie-break reused verbatim from the Plain path"
key-files:
  created:
    - crates/cb-train/src/tree_ordered_test.rs
  modified:
    - crates/cb-train/src/tree.rs
    - crates/cb-train/src/lib.rs
    - crates/cb-train/src/boosting.rs
decisions:
  - "Ordered split scoring walks [0, tail_finish) as a single contiguous range per segment: under the in-scope ordered_boost fixture random_strength=0, so the tail's SampleWeightedDerivatives == the body's WeightedDerivatives — body and tail accumulate identical der/weight, making a single walk exact."
  - "leaf_of is assigned over object order (forward-bit leaf_index) exactly as Plain, because final leaf VALUES still come from CalcLeafValuesSimple on the averaging fold (STATE.md re-scope note); only the tree STRUCTURE differs."
  - "WR-01: the dead body_sum_weight param is kept in the signature (9 params, 05-05/05-10 depend on it) but _-prefixed since the simple Gradient delta never reads the running total."
metrics:
  duration: ~25m
  completed: 2026-06-14
  tasks: 2
  files: 4
---

# Phase 05 Plan 08: Ordered Split-Scoring Subsystem Summary

Built the structural heart of Ordered boosting — a per-segment ordered L2 split-scoring search (`greedy_tensor_search_oblivious_ordered`) that scores each candidate by SUMMING its per-segment ordered L2 score across the learning fold's `BodyTailArr`, the tree-STRUCTURE difference between Ordered and Plain that the previous under-scoped 05-08 lacked (ORD-02). Plus the WR-01 dead-code cleanup in `ordered_approx_delta_simple`.

## What Was Built

**Task 1 — ordered split-scoring subsystem (`tree.rs`, commit 8204d61):**
- `greedy_tensor_search_oblivious_ordered(matrix, der1, weight, permutation, l2_leaf_reg, fold_len_multiplier, depth, n_objects) -> CbResult<GrownTree>`: per level, each candidate's score is the SUM over the learning fold's `body_tail_segments(n, fold_len_multiplier)` of its per-segment ordered L2 score, each segment using `scaledL2 = l2 * (BodySumWeight / BodyFinish)` (`scoring.cpp:746-760` `CalculateNonPairwiseScore`, additive `AddLeaf` across `bodyTailIdx`). The strict first-wins (`>`) best is chosen via the SAME `select_best_candidate` as Plain. `leaf_of` is assigned over object order (Plain-identical) for downstream averaging-fold leaf-value estimation.
- `score_candidate_ordered` (private): extends `chosen` with the candidate, assigns leaves, folds `l2_split_score` per segment, sums across segments via `sum_f64`.
- `ordered_segment_leaf_stats` (private): per-segment per-leaf `(sum_weighted_delta, sum_weight)` over `[0, tail_finish)` in permutation order (random_strength=0 ⇒ tail `SampleWeightedDerivatives` == body `WeightedDerivatives`), reduced via `reduce_leaf_stats` (D-08). All accesses checked (`permutation.get` / `der.get` / `weight.get`, negative-index guard) → `CbError::Degenerate` on OOR.
- `select_level_ordered` (private): enumerates float candidates (feature asc, border asc) and picks strict first-wins.
- Re-exported `greedy_tensor_search_oblivious_ordered` from `lib.rs` (05-10 compile precondition).
- New sibling `tree_ordered_test.rs` (8 units): degeneration anchor (single full-span segment + identity perm ⇒ same splits AND same per-candidate score as Plain), exact per-segment scaled-L2 for a hand-derived 3-segment weighted scenario, multi-segment search smoke, strict first-wins on an equal-score pair, OOR and negative permutation index → Degenerate.

**Task 2 — WR-01 cleanup (`boosting.rs`, commit 809953e):**
- Removed the dead `let mut sum_weights = body_sum_weight;` running total, its per-tail `sum_weights += w;`, and the `let _ = sum_weights;` discard. The simple Gradient delta reads only the per-leaf running weight; the accumulator was computed but never read → `approx_delta` is byte-identical.
- `body_sum_weight` param is now unused → `_`-prefixed (signature/order unchanged, 9 params); doc comment de-rationalized.

## Verification Results

All plan `<verify>` / `<verification>` commands pass:
- `cargo test -p cb-train --lib tree::` — 15 passed (incl. 8 `tree::ordered` units, all green).
- `cargo test -p cb-train --test slice_first_oracle_test` — 2/2 (numeric Plain unchanged).
- `cargo test -p cb-train --test one_hot_oracle_test` — 3/3 (one-hot Plain unchanged).
- `cargo test -p cb-train --test ordered_boost_oracle_test` — 5/5 (WR-01 byte-identical).
- `cargo test -p cb-train --lib` — 128/128.
- `grep -c greedy_tensor_search_oblivious_ordered crates/cb-train/src/lib.rs` — 1 (re-exported).
- `grep -c sum_weights crates/cb-train/src/boosting.rs` — 0; `grep -c "let _ = sum_weights" ...` — 0; `pub fn ordered_approx_delta_simple` — 1 (9 params).
- `cargo check --workspace --tests` — clean.
- `cargo clippy -p cb-train --lib` — no new warnings on tree.rs / boosting.rs (only pre-existing bootstrap.rs excessive-precision, out of scope).
- No `unwrap`/`expect`/`panic`/`anyhow` added to production `tree.rs`.

Per the disk-pressure constraint, no full `cargo test --workspace` link was run; only the plan's per-crate commands were used.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] clippy `too_many_arguments` on the new public fn**
- **Found during:** Task 1
- **Issue:** `greedy_tensor_search_oblivious_ordered` has 8 parameters; clippy's default `too_many_arguments` (7) warns.
- **Fix:** Added `#[allow(clippy::too_many_arguments)]`, matching the existing codebase convention (`ordered_approx_delta_simple` already uses it). Signature kept as designed (the params are the irreducible inputs: matrix, der1, weight, permutation, l2, multiplier, depth, n_objects).
- **Files modified:** crates/cb-train/src/tree.rs
- **Commit:** 8204d61

Otherwise the plan executed as written.

## Known Stubs

None. The subsystem is fully implemented and unit-locked; it is not yet wired into `train()` — that is the explicit scope of 05-10 (the public fn + re-export are the handoff, verified present).

## Self-Check: PASSED

- FOUND: crates/cb-train/src/tree.rs
- FOUND: crates/cb-train/src/tree_ordered_test.rs
- FOUND: crates/cb-train/src/lib.rs
- FOUND: crates/cb-train/src/boosting.rs
- FOUND commit 8204d61 (Task 1)
- FOUND commit 809953e (Task 2)
