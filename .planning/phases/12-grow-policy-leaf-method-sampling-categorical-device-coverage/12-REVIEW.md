---
phase: 12-grow-policy-leaf-method-sampling-categorical-device-coverage
reviewed: 2026-07-04T05:01:42Z
depth: standard
files_reviewed: 40
files_reviewed_list:
  - crates/cb-backend/src/gpu_backend.rs
  - crates/cb-backend/src/gpu_backend_test.rs
  - crates/cb-backend/src/gpu_runtime/mod.rs
  - crates/cb-backend/src/gpu_runtime/session.rs
  - crates/cb-backend/src/gpu_runtime/session_depth_gt1_test.rs
  - crates/cb-backend/src/gpu_runtime/session_residency.rs
  - crates/cb-backend/src/kernels.rs
  - crates/cb-backend/src/kernels/bootstrap_device.rs
  - crates/cb-backend/src/kernels/bootstrap_device_test.rs
  - crates/cb-backend/src/kernels/cindex.rs
  - crates/cb-backend/src/kernels/ctr_device.rs
  - crates/cb-backend/src/kernels/ctr_device_test.rs
  - crates/cb-backend/src/kernels/exact_quantile.rs
  - crates/cb-backend/src/kernels/exact_quantile_test.rs
  - crates/cb-backend/src/kernels/mvs_device.rs
  - crates/cb-backend/src/kernels/mvs_device_test.rs
  - crates/cb-backend/src/kernels/nonsym_grow.rs
  - crates/cb-backend/src/kernels/nonsym_grow_test.rs
  - crates/cb-backend/src/kernels/region_device.rs
  - crates/cb-backend/src/kernels/region_device_test.rs
  - crates/cb-backend/src/kernels/segmented_sort_test.rs
  - crates/cb-backend/src/kernels/sort.rs
  - crates/cb-compute/src/lib.rs
  - crates/cb-compute/src/runtime.rs
  - crates/cb-core/src/rng.rs
  - crates/cb-model/src/apply.rs
  - crates/cb-model/src/cbm.rs
  - crates/cb-model/src/json.rs
  - crates/cb-model/src/lib.rs
  - crates/cb-model/src/model.rs
  - crates/cb-model/src/region_apply_test.rs
  - crates/cb-train/src/boosting.rs
  - crates/cb-train/src/boosting_device_fold_test.rs
  - crates/cb-train/src/lib.rs
  - crates/cb-train/src/region_grow_test.rs
  - crates/cb-train/src/tree.rs
  - crates/cb-train/tests/device_nonsym_fit_test.rs
  - crates/cb-train/tests/device_region_fit_test.rs
  - crates/cb-train/tests/device_seam_test.rs
  - crates/cb-train/tests/region_e2e_test.rs
findings:
  critical: 0
  warning: 4
  info: 3
  total: 7
status: issues_found
---

# Phase 12: Code Review Report

**Reviewed:** 2026-07-04T05:01:42Z
**Depth:** standard
**Files Reviewed:** 40
**Status:** issues_found

## Summary

This phase adds device-coverage for grow policies (Depthwise / Lossguide / Region),
leaf methods (Exact weighted quantile), sampling (Bernoulli / Bayesian / Poisson /
MVS), and categorical CTR accumulation, plus the boosting-loop fold arms and model
serialization for Region trees.

The code is unusually defensive: every `#[cube]` launch validates lengths/value
ranges before dispatch, read-back failures map to typed `CbError::Degenerate` rather
than silent zero buffers, all host indexing uses checked `.get`, and overflow-prone
products are `checked_mul`. I traced the RNG transcriptions (`bootstrap_device.rs`,
`mvs_device.rs`) against `cb-core::rng::TFastRng64` primitive-for-primitive — the
per-block reseed, `fix_seq`, `advance(10)`, and `gen_rand` high/low ordering all match.
The Region / non-symmetric grow structure, the `leaf_of` walk, and the apply-side
`region_leaf` / `leaf_index_nonsym` walks are mutually consistent.

I found **no reachable correctness or security BLOCKER**. The most significant finding
is a **wiring gap**: the Phase-12 sampling / exact-leaf / CTR session machinery is
essentially unreachable from the real `train()` entry point because the boosting loop
never populates those config fields (WR-01). The remaining warnings are latent
robustness / round-trip gaps in code paths not exercised by the current covered regime.

## Warnings

### WR-01: Device sampling / exact-leaf / CTR is never reachable from `train()`

**File:** `crates/cb-train/src/boosting.rs:3060-3103` (config build) and `:3004-3048` (host-eligibility gate)
**Issue:** The production `device_config` is built as:
```rust
let device_config = DeviceTrainConfig {
    grow_policy: device_grow_policy,
    max_leaves: /* Some only for Lossguide */,
    min_data_in_leaf: params.min_data_in_leaf,
    ..DeviceTrainConfig::default()   // bootstrap_type=No, mvs_lambda=None,
                                     // exact_leaf=false, ctr=None, sample_rate=1.0
};
```
No production code path ever sets `bootstrap_type`, `mvs_lambda`, `exact_leaf`,
`ctr`, `sample_rate`, or `rng_seed` to a non-default value (confirmed by grep across
`cb-train`/`cb-model`). Additionally, `device_host_eligible` independently requires
`matches!(params.bootstrap_type, EBootstrapType::No)`, `params.random_strength == 0.0`,
and `matches!(params.leaf_method, Gradient | Simple)`. Consequently the entire
Plan-05/06/07/08 session apparatus —
`ExactLeafState`/`compute_exact_leaf_values` (`session.rs:278,1156`),
`BootstrapState` + `launch_bootstrap_weights_resident` (`bootstrap_device.rs`),
`MvsState` + `launch_mvs_weights_resident` (`mvs_device.rs`),
`build_ctr_cindex_columns` + the CTR gate (`session.rs:155,685-719`), and
`device_score_stddev` (`bootstrap_device.rs:502`) — is **dead from the training entry
point**. It is exercised only by the `#[cfg(test)]` self-oracles, never end-to-end.
If Phase 12 claims these features are delivered on the device training path, that claim
is not substantiated by any reachable call site.
**Fix:** Either (a) thread the sampling/exact/CTR knobs from `params` into
`device_config` and relax the corresponding `device_host_eligible` predicates (mirroring
the grow-policy wiring already done at `:3065-3081`), or (b) if end-to-end wiring is
intentionally deferred to a later plan (Kaggle CUDA sign-off per project memory),
document these session arms as test-only / pending and gate them behind an explicit
"not yet wired" marker so reviewers/consumers do not assume they are active in `train()`.

### WR-02: `launch_ordered_ctr_resident` has no bin-value-range guard (device OOB risk)

**File:** `crates/cb-backend/src/kernels/ctr_device.rs:229-294`
**Issue:** The kernel indexes the per-bucket scratch as `counts[2*bucket + class]`
where `bucket = bins[doc]` and `counts` has length `2 * bucket_count`. The launch
wrapper validates only the *lengths* of `perm`/`bins`/`class` (`:238`), not that every
`bins[doc] < bucket_count` (nor `class[doc] < 2`). A `bins` value `>= bucket_count`
produces an out-of-bounds device read/store (UB), contradicting the module's own claim
("every index derives from a bounds-validated host bucket count", `:133`). The sibling
histogram seam (`gpu_runtime/mod.rs:616-633`) and `session::begin` (`session.rs:675`)
both guard bin values host-side before launch; this path does not. Current callers
(`build_ctr_cindex_columns` single-member `max+1` sizing, and `combine_projection_bins`
dense remap) happen to keep bins in range, so it is latent — but the `pub(crate)`
contract and the test-reachable surface make it a real gap.
**Fix:** Add a host guard before dispatch, matching the histogram pattern:
```rust
if let Some(&bad) = bins.iter().find(|&&b| (b as usize) >= bucket_count) {
    return Err(CbError::OutOfRange(format!(
        "ctr bin value {bad} >= bucket_count ({bucket_count})")));
}
if let Some(&bad) = class.iter().find(|&&c| c > 1) {
    return Err(CbError::OutOfRange(format!("ctr class {bad} not in {{0,1}}")));
}
```

### WR-03: Region JSON serialize silently drops non-float levels, desyncing the path

**File:** `crates/cb-model/src/json.rs:512-523`
**Issue:** Region levels are serialized with `filter_map(|lvl| lvl.split.as_float()...)`.
A `ModelSplit::Ctr` level is silently dropped from the wire form while `leaf_values`
(length `depth + 1`) is emitted unchanged. On deserialize (`:670-704`) the level count
is now smaller than `leaf_values.len() - 1`, so the walk's terminal bin can never reach
the highest leaves — a silent round-trip corruption. The float-only invariant holds for
the current CPU/device float grower, so this is latent, but a `filter_map` that can
change the level count without a corresponding `leaf_values` adjustment is a data-loss
hazard.
**Fix:** Either reject a Region tree containing a non-float level with a typed
`ModelError` during serialize (so the corruption surfaces loudly), or serialize CTR
levels through a full round-trip schema. At minimum, assert
`levels.len() == leaf_values_bins - 1` before emitting.

### WR-04: Region JSON does not preserve `leaf_weights` when length mismatches

**File:** `crates/cb-model/src/json.rs:692-697`
**Issue:** On deserialize, `leaf_weights` is kept only when
`t.leaf_weights.len() == n_leaves`, otherwise replaced with `vec![0.0; n_leaves]`.
Combined with WR-03 (a dropped level changes the effective `n_leaves` relative to what
was serialized), a valid Region model can silently lose its per-leaf weights on
round-trip, degrading any downstream consumer that uses `leaf_weights` (SHAP / fstr /
leaf-weight introspection). This is defensive against a ragged buffer but masks the
underlying inconsistency rather than reporting it.
**Fix:** When `t.leaf_weights` is non-empty but length-mismatched, surface a typed
`ModelError::Deserialize` instead of silently zeroing; only zero-fill when
`leaf_weights` was legitimately absent from the wire form.

## Info

### IN-01: `compute_exact_leaf_values` MAPE weight ignores the object weight

**File:** `crates/cb-backend/src/gpu_runtime/session.rs:1171-1175`
**Issue:** For MAPE the weight is `1.0 / max(1.0, |t|)`, but upstream's
`weightsWithTargets[i] = weight_i / max(1, |target_i|)` folds in the object weight
`ex.weight[i]`. This is only correct for the unit-weight covered regime. The exact
gate (`exact_covered`, `session.rs:556`) does not itself assert unit weights (it relies
on `device_host_eligible`, which does — but that gate is currently unreachable per
WR-01, so a direct `GpuTrainSession` test with non-unit weights would diverge).
**Fix:** Use `ex.weight.get(i).copied().unwrap_or(1.0) / f64::max(1.0, t.abs())` for the
MAPE branch so the weighted case is correct if the exact path is ever wired with weights.

### IN-02: Poisson host stream advance is not draw-count-aligned

**File:** `crates/cb-backend/src/gpu_runtime/session.rs:1018-1023`
**Issue:** After a Poisson device draw the host advances the continuous stream by
exactly `self.n` (`bs.rng.advance(self.n)`), but the Knuth-Poisson kernel
(`bootstrap_device.rs:244-278`) consumes a *variable* number of `gen_rand` draws per
object. The per-tree base state is therefore not aligned to the actual draws consumed.
Poisson has no CPU oracle and is validated for determinism only (same seed ⇒ same
weights), and the advance is deterministic, so this is acceptable under the stated
scope — but the stream phase is arbitrary rather than draw-faithful, which will matter
if a Poisson oracle is ever added.
**Fix:** If Poisson parity is ever required, have the kernel emit its consumed-draw
count (or advance the stream on-device) so the host advance matches actual consumption.

### IN-03: Exact arm computes and discards a resident der1 each tree

**File:** `crates/cb-backend/src/gpu_runtime/session.rs:954-961` and `:1119-1120`
**Issue:** In the exact-leaf arm, `self.der1_h`/`self.approx_h` are set from the
freshly re-uploaded caller approx (`:954-961`), then `grow_oblivious_tree_resident`
overwrites `self.approx_h`/`self.der1_h` again with the Newton-updated device state
(`:1119-1120`) — which is immediately discarded on the next call's re-sync. The extra
device der launch is correct but wasted. Not a bug (the re-sync guarantees correctness),
noted for clarity since the two writes to the resident handles look contradictory.
**Fix:** None required for correctness; optionally skip the `:1119-1120` writeback when
`self.exact_leaf.is_some()` since the values are never read.

---

_Reviewed: 2026-07-04T05:01:42Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
