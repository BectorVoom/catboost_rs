---
phase: 03-cpu-training-core-plain-boosting-oblivious-trees
reviewed: 2026-06-13T00:00:00Z
depth: standard
files_reviewed: 41
files_reviewed_list:
  - crates/cb-backend/Cargo.toml
  - crates/cb-backend/src/cpu_runtime.rs
  - crates/cb-backend/src/cpu_runtime_test.rs
  - crates/cb-backend/src/kernels.rs
  - crates/cb-backend/src/kernels/gradient.rs
  - crates/cb-backend/src/kernels/scatter.rs
  - crates/cb-backend/src/lib.rs
  - crates/cb-compute/Cargo.toml
  - crates/cb-compute/src/histogram.rs
  - crates/cb-compute/src/leaf.rs
  - crates/cb-compute/src/lib.rs
  - crates/cb-compute/src/loss.rs
  - crates/cb-compute/src/runtime.rs
  - crates/cb-compute/src/score.rs
  - crates/cb-core/src/error.rs
  - crates/cb-core/src/lib.rs
  - crates/cb-core/src/normal.rs
  - crates/cb-core/src/reduction.rs
  - crates/cb-core/src/rng.rs
  - crates/cb-oracle/src/error.rs
  - crates/cb-oracle/src/lib.rs
  - crates/cb-oracle/src/model_json.rs
  - crates/cb-oracle/generator/gen_fixtures.py
  - crates/cb-train/Cargo.toml
  - crates/cb-train/src/autolr.rs
  - crates/cb-train/src/boosting.rs
  - crates/cb-train/src/bootstrap.rs
  - crates/cb-train/src/lib.rs
  - crates/cb-train/src/metrics.rs
  - crates/cb-train/src/overfit.rs
  - crates/cb-train/src/tree.rs
findings:
  critical: 1
  warning: 6
  info: 5
  total: 12
status: issues_found
---

# Phase 3: Code Review Report

**Reviewed:** 2026-06-13
**Depth:** standard
**Files Reviewed:** 41 (production + tests; tests inspected for reliability only)
**Status:** issues_found

## Summary

The phase implements the CPU plain-boosting core: oblivious-tree growth, four leaf-estimation methods, four bootstrap samplers, random-strength perturbation, overfitting detection, eval metrics, auto-LR, and the CubeCL CPU runtime. Overall code quality is high: parity discipline (`cb_core::sum_f64` routing, D-08) is consistently honored, every error path returns `CbError` rather than panicking, source/test separation is respected (all `mod tests;` are `#[path]` declarations to sibling files), and there are no `unwrap`/`expect`/`panic` in production code.

The adversarial pass surfaced one BLOCKER (a derivative-set divergence in the random-strength std-dev that will break parity once perturbation and sampling are combined), several WARNINGs around defensive-fallback masking of real bugs and hot-loop allocation, and some INFO items. The documented `overfit.rs` Horner fold was assessed and is genuinely a multiply-add (`acc * x + c`), not a summation — safe. The 6 `#[ignore]`d multi-tree oracle tests are treated as documented deferrals per the brief.

## Critical Issues

### CR-01: `scoreStDev` is computed over the SAMPLED/control-masked derivatives, diverging from upstream's full-fold `derivativesStDevFromZero`

**File:** `crates/cb-train/src/boosting.rs:588-597` (and `crates/cb-compute/src/score.rs:76-111`)
**Issue:** The random-strength perturbation magnitude is computed as
```rust
let std_dev = score_st_dev(params.random_strength, &score_weighted_der1, model_length);
```
where `score_weighted_der1` is the *sampled, control-masked* derivative vector (zeroed for control-false objects, scaled by `sample_weights`). Upstream's `CalcDerivativesStDevFromZeroPlainBoosting` (`greedy_tensor_search.cpp:92-107`) computes the RMS over the **full AveragingFold** weighted derivatives — the un-sampled `weighted_der1` — exactly as leaf estimation does (boosting.rs already correctly uses the raw `weighted_der1` for the leaf path at line 613-623, and documents at lines 557-565 that sample weights affect "ONLY the SPLIT SCORING ... LEAF VALUES are estimated on the FULL, UN-sampled ... derivatives"). `derivativesStDevFromZero` is part of the per-tree scale, not the per-leaf score, and is not gated by `sampledDocs`.

Because `derivatives_std_dev_from_zero` also divides by the *full* length `n` (score.rs:82) while the masked vector has zeroed entries, the numerator is too small whenever any object is dropped/down-weighted, so `scoreStDev` is systematically biased low when `bootstrap_type != No` is combined with `random_strength != 0`.

This does not surface in the current oracle because every `random_strength` fixture pins `bootstrap_type='No'` (gen_fixtures.py:868-882), making `score_weighted_der1 == weighted_der1`. It is a latent parity break for any sampled-and-perturbed run.

**Fix:** Use the full-fold weighted derivatives for the std-dev, matching the leaf path:
```rust
let std_dev = score_st_dev(params.random_strength, &weighted_der1, model_length);
```
and confirm against an upstream MVS/Bernoulli + random_strength fixture before closing.

## Warnings

### WR-01: MAE gradient kernel uses a different deadzone boundary than the host `mae_der1`, risking f32/f64 cross-backend divergence

**File:** `crates/cb-backend/src/kernels.rs:60-74` vs `crates/cb-compute/src/loss.rs:89-99`
**Issue:** Host `mae_der1` selects the gradient by `val.abs() < delta -> 0; val > 0.0 -> alpha; else -(1-alpha)`. The kernel instead uses `val > delta -> alpha; val < -delta -> -(1-alpha); else 0`. For a residual in the open interval `(0, delta]` (specifically `val == delta` exactly, or `val` in `(0, delta)` reached differently), the two disagree: host returns `alpha` for any `val >= delta` *and* for `val` in `(0, delta)` it returns `0` (deadzone), whereas the kernel only returns `alpha` for `val > delta` strictly. At the exact boundary `val == delta` host yields `alpha`, kernel yields `0`. The kernel is the production path (CpuBackend), the host fn is only a reference; the divergence is small and boundary-only, but it is a genuine semantic mismatch between two transcriptions of the same upstream `TQuantileError::CalcDer`.
**Fix:** Make the kernel branch structure identical to the host: deadzone on `|val| < delta`, then `val > 0` (not `val > delta`) selects `alpha`. Confirm which boundary upstream `error_functions.h:485-489` uses and align both to it.

### WR-02: Hot-loop per-candidate allocation in `score_candidate` / `reduce_leaf_stats` / `assign_leaves`

**File:** `crates/cb-train/src/tree.rs:164-194`, `crates/cb-compute/src/histogram.rs:49-87`
**Issue:** Memory efficiency is a first-class project constraint (CLAUDE.md). For every candidate border, at every tree level, of every tree, `score_candidate` does: `chosen.to_vec()` (tree.rs:188), a full `assign_leaves` pass allocating a `Vec<bool>` per object (tree.rs:167), and `reduce_leaf_stats` which allocates `2 * n_leaves` inner `Vec<f64>`s and pushes per object (histogram.rs:58-75). With F features × B borders × depth levels × iterations, this is O(candidates × n) allocations in the hottest training path. Upstream uses pre-allocated histogram buffers and incremental updates.
**Fix:** Reuse scratch buffers across candidates (a single reusable `leaf_of`/stats buffer cleared per candidate), or accumulate into fixed-size `[LeafStats; n_leaves]` arrays without per-leaf `Vec`s. Not a correctness bug, but a direct violation of the memory-efficiency constraint in the inner loop.

### WR-03: Bernoulli / MVS subsample short-circuit on `>= 1.0` diverges from upstream `== 1` draw accounting

**File:** `crates/cb-train/src/bootstrap.rs:183-184`, `bootstrap.rs:288-289`
**Issue:** `set_sampled_control` and `mvs_sample_weights` early-return all-`true`/all-`1.0` for `subsample >= 1.0`. Upstream's `BernoulliSampleRate == 1` (and MVS `SampleRate == 1`) early-return is keyed on `== 1`. For `subsample > 1.0` (out-of-range but not rejected here), upstream would still draw `GenRandReal1()` per object (comparison `u < rate` always true) and advance the RNG, whereas this code draws nothing — desynchronizing the continuous persistent RNG for all subsequent trees. `subsample` is not range-validated anywhere in `BoostParams`.
**Fix:** Either validate `0.0 < subsample <= 1.0` at the boosting entry (return `CbError::OutOfRange`), or make the short-circuit exactly `subsample == 1.0` to match upstream draw counting. Prefer input validation so an out-of-range value cannot silently mis-phase the RNG.

### WR-04: Defensive `unwrap_or` fallbacks silently mask programmer errors instead of surfacing them

**File:** `crates/cb-train/src/bootstrap.rs:168`, `crates/cb-train/src/boosting.rs:281`, `crates/cb-compute/src/histogram.rs:67-68,110,146-147`
**Issue:** Multiple hot paths swallow what would be programmer errors. `generate_random_weights` uses `weights.get_mut(begin..end).unwrap_or(&mut [])` (bootstrap.rs:168) — a slice-range miss silently produces an empty slice and leaves those objects' weights at the `1.0` default, corrupting the Bayesian weight vector without any signal. `tree_eval_contribution` returns `0.0` for an out-of-range leaf (boosting.rs:281). `reduce_leaf_stats`/`reduce_leaf_der2`/`collect_leaf_residuals` silently treat a length-mismatched `der1`/`weight` as `0.0`/`1.0`. These are documented as "defensive — the trainer passes valid indices," but they convert contract violations into silent numeric corruption rather than a `CbError`, defeating the ≤1e-5 oracle gate's ability to localize a bug.
**Fix:** For internal invariants that "cannot" be violated, prefer `debug_assert!`-backed checks plus an explicit early `CbError::Degenerate` on the public boundary, rather than `unwrap_or` that produces plausible-but-wrong numbers. At minimum, validate `der1.len() == weight.len() == leaf_of.len()` once at the top of `reduce_leaf_stats`.

### WR-05: `uniform()` returns `0` for a zero bound, fabricating a valid-looking index from an invalid request

**File:** `crates/cb-core/src/rng.rs:227-230`
**Issue:** `uniform(0)` returns `0` via `try_uniform(0).unwrap_or(0)`. Upstream's `Uniform` asserts `max > 0` (`Y_ABORT_UNLESS`). Returning `0` (a valid-looking index) means a caller that mistakenly passes a zero bound gets a silently wrong draw AND does not advance the RNG (try_uniform returns before any `gen_rand`), which would desync the parity stream. The doc acknowledges this trade-off, but a fabricated `0` is more dangerous than the documented panic it replaces because it corrupts results silently.
**Fix:** Since no current caller uses `uniform` (only `try_uniform`/`gen_rand_real1` are consumed), either remove the infallible `uniform` entirely or have it draw-and-discard to keep RNG phase, and document loudly that `0` is a sentinel. Best: keep only `try_uniform` and force callers to handle the error.

### WR-06: `derivatives_std_dev_from_zero` / `model_size_multiplier` denominator uses the masked vector length, not the object count

**File:** `crates/cb-compute/src/score.rs:76-95`, called from `crates/cb-train/src/boosting.rs:590`
**Issue:** Coupled to CR-01: `derivatives_std_dev_from_zero` divides the sum of squares by `weighted_der1.len()`, and `model_size_multiplier` takes `weighted_der1.len()` as `n` for `ln(n)`. When the score-path masked vector is passed (current behavior), the length still equals `n` (entries are zeroed, not removed), so `ln(n)` is correct but the RMS numerator is wrong (CR-01). If a future refactor instead *filters* control-false objects (changing the length), `ln(n)` would also break. The `n`/length coupling is fragile.
**Fix:** Pass `n` (object count) explicitly to `score_st_dev` rather than deriving it from the derivative slice length, and feed the full-fold derivatives (CR-01), decoupling the two concerns.

## Info

### IN-01: `histogram_scatter_kernel` is dead in the production runtime path

**File:** `crates/cb-backend/src/kernels.rs:102-111`
**Issue:** The scatter kernel (`contrib[i] = der1[i] * weight[i]`) is defined, exported, and unit-tested, but `CpuBackend::compute_gradients` never launches it — the boosting loop computes `der1 * weight` host-side (boosting.rs:531-536). It is currently a tested-but-unused kernel.
**Fix:** Either wire it into the runtime trait (a `histogram_scatter` op) or annotate it as an intentional forward-looking primitive in the module docs so reviewers do not flag it as orphaned.

### IN-02: `EvalMetricHistory::new` is constructed twice on the single-eval-set path

**File:** `crates/cb-train/src/boosting.rs:360,492-494`
**Issue:** `train_with_eval` builds a `history` (line 360), passes `history.as_mut()` into `train_with_eval_sets`, which then *re-creates* it via `*h = EvalMetricHistory::new(eval_sets.len())` (line 492-494). The first allocation is immediately discarded. Harmless, slightly wasteful and confusing.
**Fix:** Let `train_with_eval_sets` own the history initialization and have `train_with_eval` pass an empty placeholder, or size it once.

### IN-03: MAE eval metric silently falls back to RMSE reporting

**File:** `crates/cb-train/src/metrics.rs:59-64`
**Issue:** `EvalMetric::for_loss(Loss::Mae)` returns `Self::Rmse`. A user training with MAE and an eval set gets an RMSE eval curve, which then drives the overfitting detector and `use_best_model`. This is documented as a deliberate Phase-3 limitation, but it means the stop decision for an MAE run is made on the wrong metric.
**Fix:** Acceptable as a documented deferral; ensure it is surfaced to the user (warning log) rather than silent, or block MAE + eval-driven early stopping until the MAE metric lands.

### IN-04: `exact_leaf_delta` uses `partial_cmp(...).unwrap_or(Equal)` to sort residuals containing potential NaN

**File:** `crates/cb-compute/src/leaf.rs:189`
**Issue:** Residuals `target - approx` are sorted with `partial_cmp().unwrap_or(Equal)`. If any residual is NaN (e.g. a NaN target leaking through), the sort treats it as equal-to-everything, producing an arbitrary order and a meaningless quantile, silently. Upstream operates on `TVector<float>` where NaN would also misbehave, but here the failure is silent.
**Fix:** Not a blocker for clean inputs; consider asserting finiteness of residuals at the leaf boundary so a NaN surfaces as `CbError::Degenerate` rather than a garbage median.

### IN-05: `wilcoxon` block-grouping reads `relative_equal` between abs-equal-sorted *signed* values

**File:** `crates/cb-train/src/overfit.rs:369-388,406-412`
**Issue:** The block extension compares `relative_equal(next_val, first_val)` on the *signed* values after sorting by absolute value. Two entries with equal magnitude but opposite sign (e.g. `+x` and `-x`) have `|x - (-x)| = 2|x|`, which is NOT `< eps * |x|`, so they correctly fall into different blocks — but the intent (group by equal *magnitude*) is expressed via signed comparison, which only coincidentally works because abs-sorting places equal-magnitude opposite-sign values adjacently and `relative_equal` then separates them. This is correct for the upstream semantics (ranks are by magnitude, signs only decide the `w +=` contribution) but the code reads as if it groups signed values. Verify against `detail.h` that opposite-sign equal-magnitude ties are meant to be separate rank blocks.
**Fix:** Add a comment clarifying that signed inequality here is intentional, or compare `next_val.abs()` vs `first_val.abs()` to match the stated "group by |value|" intent.

---

_Reviewed: 2026-06-13_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
