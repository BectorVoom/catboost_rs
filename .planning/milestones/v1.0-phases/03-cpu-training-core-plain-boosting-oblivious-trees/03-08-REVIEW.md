---
phase: 03-cpu-training-core-plain-boosting-oblivious-trees
reviewed: 2026-06-13T00:00:00Z
depth: standard
files_reviewed: 4
files_reviewed_list:
  - crates/cb-train/src/boosting.rs
  - crates/cb-compute/src/score_test.rs
  - crates/cb-train/tests/regularization_oracle_test.rs
  - crates/cb-oracle/generator/gen_fixtures.py
findings:
  critical: 0
  warning: 0
  info: 2
  total: 2
status: clean
---

# Phase 03-08: Code Review Report

**Reviewed:** 2026-06-13
**Depth:** standard
**Files Reviewed:** 4
**Status:** clean

## Summary

This is a focused adversarial review of the CR-01 gap-closure change (plan 03-08).
The substantive production change is a single argument swap at
`crates/cb-train/src/boosting.rs:597`: `score_st_dev` now reads the full,
un-sampled fold `&weighted_der1` instead of the control-masked
`&score_weighted_der1`. The remaining files are oracle/test infrastructure.

I focused the FORCE stance on the four points requested. Every claim in the
change was traced through the implementation, verified numerically where the test
asserts a constant, and cross-checked against the surrounding histogram/leaf
paths. The change is surgically correct, the histogram inputs were correctly left
unchanged, the test-helper signature change was threaded to all three callers
without an argument-count or argument-order regression, project conventions
(no `unwrap()`/`expect()` in production, source/test separation) are honored, and
the Python generator scenario is internally consistent with the test and the
committed fixture. No BLOCKER or WARNING findings. Two INFO observations follow.

### Verification performed (correctness, point 1)

- **Std-dev input swap is correct and complete.** Full audit of every
  derivative-vector usage in `boosting.rs`:
  - `score_st_dev(..., &weighted_der1, ...)` at line 597 — the fix; full fold.
  - `greedy_tensor_search_oblivious_perturbed(..., &score_weighted_der1, &score_weights, ...)`
    at lines 607-608 — the SCORE HISTOGRAM path, correctly STILL masked. This is
    the `sampledDocs` restriction and must remain masked; it was left unchanged.
  - `compute_leaf_deltas(..., &weighted_der1, &ders.der2, &weights, ...)` at line
    623 — the LEAF path; correctly full fold. The std-dev now matches the leaf
    path's input, which is the upstream invariant
    (`CalcDerivativesStDevFromZeroPlainBoosting` reads
    `fold.BodyTailArr.front().WeightedDerivatives`).
  - `bootstrap(..., &weighted_der1, ...)` at line 550 — MVS reads the full
    weighted derivatives; correct and unchanged.
  - The masked `score_weighted_der1` (lines 566-571) and `score_weights`
    (lines 572-577) construction is unchanged and still zeroes control-false
    entries while preserving length — exactly the `sampledDocs` semantics.
- **The std-dev math is correct.** `derivatives_std_dev_from_zero`
  (`score.rs:76-83`) computes `sqrt(sum(wd_i^2)/n)` over the full vector length;
  the masked input would shrink the numerator while `n` (denominator AND the
  `ln(n)` model-size multiplier) stays fixed, biasing the result low whenever any
  object is dropped — exactly the CR-01 break. The fix removes that bias.
- **Unit-test numeric constants verified independently:**
  - `derivatives_std_dev_from_zero([1,-2,3,-0.5]) = 1.8874586088176875` ✓
  - `score_st_dev(1.0, wd, 0.2) = 1.4459398718996799` ✓
  - first-tree multiplier `n/(1+n) = 0.8` at `model_length=0`, product
    `dsdz*0.8 = 1.50996688705415` ✓
  - CR-01 contract test: full `sqrt(3.5625)=1.8874...` > masked
    `sqrt(2.5)=1.5811...` ✓ (the strict-inequality assertion holds).

### Verification performed (test-helper threading, point 2)

- `check_scenario_first_trees` gained a 7th parameter `subsample: f64`
  (`regularization_oracle_test.rs:148`), wired into `BoostParams.subsample` at
  line 165 (was hardcoded `1.0`).
- All three callers pass the new argument with correct positional order
  `(scenario, n_trees, l2_leaf_reg, random_strength, bootstrap_type, bagging_temperature, subsample)`:
  - `random_strength_first_tree` (211-219): `…, No, 0.0, 1.0` — subsample 1.0 ✓
  - `random_strength_bernoulli` (239-247): `…, Bernoulli, 0.0, 0.7` — subsample 0.7 ✓
  - `bagging_temp_first_tree` (283-291): `…, Bayesian, 0.5, 1.0` — subsample 1.0 ✓
- `grep` confirms exactly these three callers exist; no caller was missed, no
  argument-count regression.

### Verification performed (conventions, point 3)

- No `#[cfg(test)] mod tests` in either production file; `score_test` is a
  separate file gated `#[cfg(test)] mod score_test;` in `lib.rs:50-51`. Source/test
  separation per CLAUDE.md is honored.
- No `unwrap()`/`expect()` introduced in production. The only `unwrap` tokens in
  `boosting.rs` are `unwrap_or(0.0)` / `unwrap_or_default()` / `unwrap_or_else`
  (infallible combinators, pre-existing) and a doc-comment mention — none are
  panicking `.unwrap()`. The integration test file legitimately allows
  `clippy::unwrap_used` via the file-level `#![allow(...)]` (tests, not
  production), consistent with project conventions.

### Verification performed (Python generator, point 4)

- The new `random_strength_bernoulli` scenario (gen_fixtures.py:899-907) sets
  `l2_leaf_reg=3.0, random_strength=1.0, bootstrap_type="Bernoulli",
  subsample=0.7`. The committed `config.json` confirms these exact params plus
  the shared isolating set (depth=2, lr=0.1, iterations=3, seed=0, thread_count=1).
- The scenario is appended to the existing `scenarios` list and flows through the
  unchanged per-scenario loop (save_model / staged / predictions / config) — no
  new code path, no divergence from the established pattern.
- The fixture was trained with `iterations=3`; the Rust test gates only the first
  tree (`n_trees=1`). This is intentional and consistent — `check_scenario_first_trees`
  compares only trees `0..n_trees`, and the CR-01 bias is observable on tree 0.

## Info

### IN-01: `subsample=0.7` documented as "Ignored by `No`/`Bayesian`" yet passed for those scenarios

**File:** `crates/cb-train/tests/regularization_oracle_test.rs:211-219, 283-291`
(in conjunction with `boosting.rs:94-96`)
**Issue:** The two non-Bernoulli first-tree callers explicitly pass
`subsample = 1.0`. Per the `BoostParams::subsample` doc comment
("`1.0` disables subsampling. Ignored by `No`/`Bayesian`"), the value is inert for
those scenarios, so `1.0` is harmless and correct. This is noted only because the
parameter is now caller-visible: a future reader could mistake `subsample` as
meaningful for the `No`/`Bayesian` first-tree tests. No behavioral impact.
**Fix:** Optional — none required. If desired, a one-line comment at the `No`/
`Bayesian` call sites noting "subsample inert for this bootstrap_type" would
prevent confusion. Not worth a code change on its own.

### IN-02: CR-01 fix is not gated end-to-end on the oracle corpus (acknowledged residual)

**File:** `crates/cb-compute/src/score_test.rs:101-150` and
`crates/cb-train/tests/regularization_oracle_test.rs:222-248`
**Issue:** The production fix (full-fold std-dev) is locked by the cb-compute UNIT
contract test (`score_st_dev_masked_vector_biases_low_vs_full_fold_cr01`), which
proves masked < full at the boundary where the bug is observable. The end-to-end
`regularization_oracle_random_strength_bernoulli` oracle test gates only tree 0's
splits + leaf values, and the test/comment candidly state the std-dev difference
"never flips a tree-0 split" on `numeric_tiny` (it is entangled with the
variable-length Box-Muller draw residual, D-11 / Open Q4). So the oracle test does
NOT independently prove the fix's numeric magnitude on real data — it proves the
draw-order/structure is unbroken, while the unit test carries the magnitude
contract. This is a transparent and reasonable testing strategy given the
documented tree-1+ RNG-phase residual, but a reviewer should be aware that the
"oracle gate" for the magnitude is the unit test, not the end-to-end fixture.
**Fix:** None required for this slice — the split is documented in the test
headers and SUMMARY. Tracked under D-11 / Open Q4 for Phase 5 C++ instrumentation.

---

_Reviewed: 2026-06-13_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
