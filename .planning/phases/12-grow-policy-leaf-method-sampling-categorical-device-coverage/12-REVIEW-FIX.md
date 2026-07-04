---
phase: 12-grow-policy-leaf-method-sampling-categorical-device-coverage
fixed_at: 2026-07-04T05:01:42Z
review_path: .planning/phases/12-grow-policy-leaf-method-sampling-categorical-device-coverage/12-REVIEW.md
iteration: 1
findings_in_scope: 7
fixed: 7
skipped: 0
status: all_fixed
---

# Phase 12: Code Review Fix Report

**Fixed at:** 2026-07-04T05:01:42Z
**Source review:** .planning/phases/12-grow-policy-leaf-method-sampling-categorical-device-coverage/12-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 7 (all — critical/warning/info)
- Fixed: 7
- Skipped: 0

## Fixed Issues

### WR-02: `launch_ordered_ctr_resident` has no bin-value-range guard (device OOB risk)

**Files modified:** `crates/cb-backend/src/kernels/ctr_device.rs`
**Commit:** c048cd7
**Applied fix:** Added a host-side guard before device dispatch that rejects any
`bins[doc] >= bucket_count` (the kernel indexes `counts[2*bucket + class]`, length
`2 * bucket_count`) and any `class[doc] > 1`, returning `CbError::OutOfRange`. Mirrors
the histogram seam's host-side bin guard, closing the latent OOB/UB path on the
`pub(crate)` / test-reachable surface. Exactly the guard suggested in the review.

### WR-03: Region JSON serialize silently drops non-float levels, desyncing the path

**Files modified:** `crates/cb-model/src/json.rs`, `crates/cb-model/src/error.rs`
**Commit:** 45e25be
**Applied fix:** Added a new `ModelError::Serialize(String)` variant and made `to_doc`
fallible (`-> Result<ModelJsonDoc, ModelError>`). The Region-tree serialization now
rejects any non-float split level loudly instead of silently `filter_map`-dropping it,
and asserts `per_dim_leaves == levels.len() + 1` before emitting. `save_json` propagates
the error via `?`. This surfaces the round-trip corruption instead of emitting a document
whose level count desyncs from `leaf_values`.

### WR-04: Region JSON does not preserve `leaf_weights` when length mismatches

**Files modified:** `crates/cb-model/src/json.rs`
**Commit:** 45e25be (committed together with WR-03 — both are Region JSON round-trip
fixes in the same file, and WR-04's condition directly references the WR-03 corruption;
file-granularity atomic commits cannot separate them)
**Applied fix:** On deserialize, `leaf_weights` is now zero-filled ONLY when it was
legitimately absent from the wire form (empty). A NON-EMPTY but length-mismatched
`leaf_weights` now returns `ModelError::Deserialize` instead of being silently zeroed,
so a corrupt/desynced model is reported rather than degrading downstream consumers
(SHAP / fstr / leaf-weight introspection).

### IN-01: `compute_exact_leaf_values` MAPE weight ignores the object weight

**Files modified:** `crates/cb-backend/src/gpu_runtime/session.rs`
**Commit:** b8f9bb7
**Applied fix:** The MAPE branch now computes
`ex.weight.get(i).copied().unwrap_or(1.0) / f64::max(1.0, t.abs())`, folding in the
object weight to match upstream's `weightsWithTargets[i] = weight_i / max(1, |target_i|)`.
The unit-weight covered regime is unchanged; the weighted case is now correct if the
exact path is ever wired with non-unit weights. The existing doc comment already
described the correct upstream formula, so code and doc now agree.

### WR-01: Device sampling / exact-leaf / CTR is never reachable from `train()`

**Files modified:** `crates/cb-train/src/boosting.rs`
**Commit:** d02800b
**Applied fix:** Chose review option (b) — document the deferral — consistent with
project memory (GPU MVP delivers derivatives only; sampling/exact/CTR end-to-end wiring
is intentionally deferred to a later plan pending Kaggle CUDA sign-off). Added an
explicit "NOT YET WIRED (test-only / pending Kaggle CUDA sign-off)" marker on the
`..DeviceTrainConfig::default()` line enumerating the deliberately-defaulted knobs
(`bootstrap_type`, `mvs_lambda`, `exact_leaf`, `ctr`, `sample_rate`, `rng_seed`), naming
the test-only session apparatus, and noting that `device_host_eligible` independently
excludes any pool that would need those arms (so a real fit falls back to the CPU grower
rather than silently reaching an unwired arm). Reviewers/consumers are told not to assume
these features are active on the device `train()` path.

### IN-02: Poisson host stream advance is not draw-count-aligned

**Files modified:** `crates/cb-backend/src/gpu_runtime/session.rs`
**Commit:** 5320ed0
**Applied fix:** The review states this is acceptable under current scope (Poisson is
determinism-validated only, no CPU oracle) and the fix is conditional on ever adding a
Poisson parity oracle. Rather than change deterministic behavior, added a warning comment
on the Bernoulli/Poisson advance branch documenting that Bernoulli's `n`-draw advance is
draw-faithful while Poisson (Knuth, variable draw count) is a deterministic-but-arbitrary
phase, with the concrete remediation (emit consumed-draw count / advance on-device) noted
for whoever adds the oracle.

### IN-03: Exact arm computes and discards a resident der1 each tree

**Files modified:** `crates/cb-backend/src/gpu_runtime/session.rs`
**Commit:** 5320ed0 (committed together with IN-02 — both are comment-only clarifications
in `session.rs`)
**Applied fix:** The review states "None required for correctness"; the suggested skip is
an optional micro-optimization on the test-only exact arm. Altering resident-handle
control flow on a device path that cannot be exercised in this verification environment
(no GPU) risks a semantic regression the syntax-only verification tiers cannot catch.
Applied the review's request for clarity instead: added a comment at the
`self.approx_h`/`self.der1_h` writeback explaining that the exact arm re-syncs both
handles from the caller approx on the next call (so the Newton writeback is correct but
redundant there), and that the writeback is kept unconditional so the non-exact
Newton-resident arm keeps its carried state.

## Skipped Issues

None — all in-scope findings were fixed.

---

_Fixed: 2026-07-04T05:01:42Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
