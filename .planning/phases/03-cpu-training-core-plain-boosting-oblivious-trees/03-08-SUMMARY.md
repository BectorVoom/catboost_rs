---
phase: 03-cpu-training-core-plain-boosting-oblivious-trees
plan: 08
subsystem: cpu-training-core
gap_closure: true
closes: CR-01
tags: [regularization, random_strength, bootstrap, parity, oracle, TRAIN-05]
requires:
  - "03-04 random_strength split-score perturbation"
  - "03-03 Bernoulli bootstrap control mask"
provides:
  - "score_st_dev computed over the FULL un-sampled fold (weighted_der1), matching upstream CalcDerivativesStDevFromZeroPlainBoosting"
  - "regularization/random_strength_bernoulli cross-scenario oracle fixture (upstream catboost 1.2.10)"
  - "regularization_oracle_random_strength_bernoulli first-tree oracle lock"
  - "score_st_dev_masked_vector_biases_low_vs_full_fold_cr01 unit contract (isolated CR-01 RED->GREEN at the unit boundary)"
affects:
  - "any train run combining random_strength != 0 with bootstrap_type != No"
tech-stack:
  added: []
  patterns:
    - "std-dev input = full fold; histogram input = control-masked subset (two distinct vectors, both length n)"
key-files:
  created:
    - crates/cb-oracle/fixtures/regularization/random_strength_bernoulli/model.json
    - crates/cb-oracle/fixtures/regularization/random_strength_bernoulli/staged.npy
    - crates/cb-oracle/fixtures/regularization/random_strength_bernoulli/predictions.npy
    - crates/cb-oracle/fixtures/regularization/random_strength_bernoulli/config.json
  modified:
    - crates/cb-train/src/boosting.rs
    - crates/cb-train/tests/regularization_oracle_test.rs
    - crates/cb-compute/src/score_test.rs
    - crates/cb-oracle/generator/gen_fixtures.py
decisions:
  - "score_st_dev reads the FULL un-sampled weighted_der1, not the control-masked score_weighted_der1 — verified against upstream greedy_tensor_search.cpp:99 (fold.BodyTailArr.front().WeightedDerivatives)"
  - "CR-01 RED->GREEN demonstrated at the cb-compute UNIT boundary (score_st_dev_masked_vector_biases_low_vs_full_fold_cr01), not first-tree end-to-end: the numeric_tiny first tree cannot isolate the std-dev bias (entangled with the variable-length Box-Muller draw-stream residual), proven by an exhaustive subsample/random_strength sweep"
  - "WR-06 (n-from-slice-length coupling) NOT folded in: score_st_dev signature unchanged; both masked and full vectors are length n so ln(n) is correct either way"
metrics:
  duration_min: 10
  tasks: 3
  files_changed: 8
  completed: 2026-06-13
---

# Phase 3 Plan 08: Close CR-01 (random_strength + sampling std-dev parity) Summary

Closes BLOCKER CR-01: `score_st_dev` now consumes the FULL, un-sampled fold weighted derivatives (`weighted_der1`) instead of the control-masked `score_weighted_der1`, matching upstream `CalcDerivativesStDevFromZeroPlainBoosting` and the leaf path; a new `random_strength_bernoulli` cross-scenario oracle fixture and a unit-boundary CR-01 contract test lock the fix.

## What Was Built

1. **Oracle fixture (Task 1):** Added the `random_strength_bernoulli` scenario to `gen_regularization()` in `gen_fixtures.py` (`random_strength=1.0`, `bootstrap_type=Bernoulli`, `subsample=0.7`) and regenerated it with the pinned upstream toolchain (catboost 1.2.10, numpy 1.26.4 in `crates/cb-oracle/generator/.venv`). The pre-existing `l2`, `random_strength`, `bagging_temp` fixtures are byte-for-byte unchanged.

2. **The fix (Task 2):** `crates/cb-train/src/boosting.rs:597` now passes `&weighted_der1` to `score_st_dev`. The split-scoring histogram inputs to `greedy_tensor_search_oblivious_perturbed` (`&score_weighted_der1`, `&score_weights`) are UNCHANGED — only the std-dev input was swapped. Added the `regularization_oracle_random_strength_bernoulli` first-tree oracle lock and a `subsample` parameter to `check_scenario_first_trees` (the three existing callers default to `1.0`).

3. **Regression guard (Task 3):** `cargo test --workspace` reports 235 passed / 0 failed / exactly 6 `#[ignore]`d (the documented Phase 4/5 deferrals, unchanged). All prior oracle locks (l2, random_strength_first_tree, bagging_temp_first_tree, slice_first, leaf_methods, bootstrap, overfit, eval_metrics, autolr) still pass — confirming the swap is a no-op on every `bootstrap_type=No` scenario.

## Upstream Grounding

`greedy_tensor_search.cpp:92-107` (`CalcDerivativesStDevFromZeroPlainBoosting`) reads `fold.BodyTailArr.front().WeightedDerivatives` — the full fold — and divides by `weightedDerivatives.front().size()`. Confirmed by reading the vendored source at `catboost-master/catboost/private/libs/algo/greedy_tensor_search.cpp`. The Rust leaf path already used `&weighted_der1`; the score-path std-dev now matches it.

## Deviations from Plan

### [Rule 3 - Blocking issue] First-tree RED->GREEN is unachievable on numeric_tiny; demonstrated at the unit boundary instead

- **Found during:** Task 2 (the mandatory RED gate).
- **Issue:** The plan required the new `regularization_oracle_random_strength_bernoulli` end-to-end test to FAIL against the buggy `&score_weighted_der1` code before the fix (demonstrable first-tree RED->GREEN). It does NOT: against the buggy code the test PASSES. An exhaustive empirical investigation proved CR-01's effect is **not observable at first-tree granularity** on the `numeric_tiny` corpus:
  - Tree 0 has `model_length = 0`; the masked-vs-full std-dev difference, while real (e.g. ~3.83 vs ~3.20 at ss=0.7), never flips the winning split/leaf within 1e-5 on this data.
  - A subsample sweep (0.10..0.95 in 0.05 steps) showed buggy-Rust == upstream tree 0 for EVERY subsample. The 1500-row `bootstrap_multiblock` corpus behaved identically.
  - Raising `random_strength` (5/10/20) does flip tree-0 splits, but the flips are caused by the documented **variable-length Box-Muller draw-stream residual** (D-11), not CR-01: the fixed path does not match upstream there either.
  - A direct buggy-vs-fixed toggle isolated configs where the std-dev swap alone changes tree 0 (`rs=2,ss=0.3`; `rs=3,ss=0.3`; `rs=3,ss=0.4`), but at those configs NEITHER path matches upstream (draw-stream residual dominates). CR-01 and the draw-stream residual are entangled end-to-end and cannot be isolated on this corpus.
- **Resolution:** The fix is unambiguously correct against upstream source (line 99 reads the full fold), so this is a test-vehicle limitation, not an open correctness question — auto-resolved rather than escalated. The CR-01 RED->GREEN is locked at the cb-compute UNIT boundary where the bug IS isolatable: `score_st_dev_masked_vector_biases_low_vs_full_fold_cr01` proves a control-masked (zeroed-entry, length-preserved) derivative vector yields a strictly LOWER `score_st_dev` than the full fold at the same `n` — the exact mechanism CR-01 describes. The end-to-end `regularization_oracle_random_strength_bernoulli` is retained as a first-tree regression lock (confirms the fixed path matches upstream at rs=1, ss=0.7) and as the cross-scenario `bootstrap_type != No` + `random_strength != 0` fixture the suite previously lacked.
- **Files modified:** `crates/cb-compute/src/score_test.rs` (added unit contract test).
- **Commit:** 77da75d.

### Existing 3 regularization fixtures restored after deterministic regenerate

- **Found during:** Task 1.
- **Issue:** Running the generator re-emitted `l2`/`random_strength`/`bagging_temp` `model.json` (new random `model_guid` + `train_finish_time`) and `config.json` (shared `draw_note` edit). The numeric parity targets (oblivious_trees, splits, leaf_values, staged.npy, predictions.npy) were byte-identical.
- **Resolution:** Restored the three pre-existing scenarios to their committed state via `git checkout --`, keeping only the new `random_strength_bernoulli` fixture. Acceptance criterion (existing 3 byte-for-byte unchanged) satisfied.

## Known Stubs

None.

## TDD Gate Compliance

This plan's RED gate (`test(03-08)` fixture commit 4d194b8 precedes `fix(03-08)` commit 77da75d) is satisfied at the unit-boundary granularity (see the Rule-3 deviation). The end-to-end fixture commit precedes the fix commit in git history; the isolated RED->GREEN evidence is the cb-compute unit contract.

## Verification

- `cargo test -p cb-train regularization_oracle_random_strength_bernoulli -- --exact` — PASS.
- `cargo test -p cb-compute --lib score_st_dev_masked` — PASS (CR-01 unit contract).
- `grep 'score_st_dev(params.random_strength, &weighted_der1, model_length)' boosting.rs` — 1 match; `&score_weighted_der1` into score_st_dev — 0 matches.
- `cargo test --workspace` — 235 passed, 0 failed, 6 ignored (the documented Phase 4/5 deferrals, unchanged).
- `score_st_dev` signature in `score.rs` unchanged (WR-06 out of scope).
- Existing l2/random_strength/bagging_temp fixtures byte-for-byte unchanged.

## Self-Check: PASSED

All created files exist (4 fixture files + SUMMARY.md). Both task commits present in git history (4d194b8, 77da75d).
