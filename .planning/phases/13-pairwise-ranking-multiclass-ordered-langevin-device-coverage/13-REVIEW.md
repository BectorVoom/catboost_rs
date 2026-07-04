---
phase: 13-pairwise-ranking-multiclass-ordered-langevin-device-coverage
reviewed: 2026-07-04T11:36:03Z
depth: standard
files_reviewed: 28
files_reviewed_list:
  - crates/cb-backend/src/gpu_runtime/mod.rs
  - crates/cb-backend/src/gpu_runtime/multiclass.rs
  - crates/cb-backend/src/gpu_runtime/multiclass_test.rs
  - crates/cb-backend/src/gpu_runtime/ordered.rs
  - crates/cb-backend/src/gpu_runtime/ordered_test.rs
  - crates/cb-backend/src/gpu_runtime/pairwise.rs
  - crates/cb-backend/src/gpu_runtime/ranking.rs
  - crates/cb-backend/src/gpu_runtime/ranking_det_test.rs
  - crates/cb-backend/src/gpu_runtime/ranking_stoch_test.rs
  - crates/cb-backend/src/gpu_runtime/session.rs
  - crates/cb-backend/src/kernels.rs
  - crates/cb-backend/src/kernels/apply_leaf_delta.rs
  - crates/cb-backend/src/kernels/cholesky_solve.rs
  - crates/cb-backend/src/kernels/cholesky_solve_test.rs
  - crates/cb-backend/src/kernels/langevin.rs
  - crates/cb-backend/src/kernels/langevin_test.rs
  - crates/cb-backend/src/kernels/multi_newton.rs
  - crates/cb-backend/src/kernels/multi_newton_test.rs
  - crates/cb-backend/src/kernels/nonsym_grow.rs
  - crates/cb-backend/src/kernels/pairwise_deriv_test.rs
  - crates/cb-backend/src/kernels/query_helper.rs
  - crates/cb-backend/src/kernels/query_helper_test.rs
  - crates/cb-backend/src/kernels/region_device.rs
  - crates/cb-compute/src/lib.rs
  - crates/cb-compute/src/runtime.rs
  - crates/cb-compute/src/runtime_test.rs
  - crates/cb-train/tests/device_seam_test.rs
  - bench/kaggle_cuda_phase13.ipynb
findings:
  critical: 0
  warning: 3
  info: 3
  total: 6
status: issues_found
---

# Phase 13: Code Review Report

**Reviewed:** 2026-07-04T11:36:03Z
**Depth:** standard
**Files Reviewed:** 28
**Status:** issues_found

## Summary

Phase 13 lands five new device families (pairwise Cholesky, deterministic + stochastic
ranking, multi-output/multiclass block-Newton, ordered-boosting trajectory, Langevin noise)
as derivative drivers + `#[cube]` kernels + self-oracles. The code is uniformly high quality
for the criteria provable locally: no `unwrap`/`expect`/`panic`/`unreachable`/`todo` in any
production (non-`_test`) file, no `-inf` literals in `#[cube]` bodies (finite `f32::MIN` and
runtime-passed brackets are used throughout), consistent overflow-checked length arithmetic,
typed `CbError` guards before every launch, wgpu f64/u64 rejection on every f64 seam, and
`.clone()`-per-launch handle discipline bound to one client. Source/test separation is honored
(the `kernels/*.rs` self-oracle modules are `#[cfg(test)] mod`, matching the established project
convention that omits the `_test` suffix for cfg(test) submodules).

**Key structural fact that bounds severity:** every Phase-13 device family is deliberately
gated to `Ok(None)` at the session coverage gate (`session.rs::begin` and the `map_*_coverage`
helpers) — the per-tree grow seam that would carry the query/pair/permutation/K-dim descriptors
is a documented forward dependency, so a covered fit *declines to the byte-unchanged CPU
grower*. The new drivers are therefore exercised ONLY by the self-oracles (which per project
memory passed on real P100 CUDA at ε≤1e-4). Consequently the defects below are **latent**: real
correctness/robustness gaps that will bite once the grow seam is wired, but currently masked by
well-separated frozen fixtures and the CPU-fallback gate. No BLOCKER-tier defect is reachable in
a shipping path today; the WARNINGs are latent parity and device-safety hazards that should be
fixed before the grow seam is wired (consistent with the deferred RV-13-01..04 debt in memory).

## Warnings

### WR-01: YetiRank/PFound-F in-query sort tie-break diverges from upstream stable-descending sort

**File:** `crates/cb-backend/src/gpu_runtime/ranking.rs:736-770` (`descending_order_per_query`)
**Issue:** The per-permutation descending order is produced by a stable *ascending* radix sort
followed by `seg.reverse()` per query (lines 760-768). Reversing a stable-ascending order does
NOT reproduce a stable-*descending* sort when perturbed values tie: tied documents emerge in
**reversed original-index order**, whereas upstream `yetirank_helpers.cpp:326-331` uses a stable
descending sort that keeps ties in original index order. Because `CalcWeightsClassic`
(lines 869-883) pairs *adjacent* documents in this order, a tie-order flip changes which doc is
winner vs loser and thus the sampled Classic pair weights — a direct parity divergence from
`yetirank_sample_pairs`. Today this is masked only by the "well-separated frozen fixture → no
ties" assumption documented at lines 734-735; any duplicate/near-duplicate `exp(approx)·ratio`
(e.g., identical approx + identical draw, common with tied features) reaches it.
**Fix:** Produce a genuine stable-descending order rather than reverse-of-stable-ascending —
radix-sort ascending on the bitwise-complemented key (`!bits` for non-negative f64), or after
reversing re-stabilize ties by ascending original index:
```rust
// Complement the radix key so ONE stable ascending pass already yields the descending
// order with tie order preserved (matching StableSort descending on expApprox):
let ord: Vec<u64> = perturbed.iter().map(|&v| !v.to_bits()).collect();
// ... single stable radix on `ord`, no per-query reverse ...
```

### WR-02: `assemble_multiclass_ders` silently substitutes 0 for a short `target`/`weight` instead of a typed length error

**File:** `crates/cb-backend/src/gpu_runtime/multiclass.rs:143-242`
**Issue:** Only `approx` is length-checked (lines 158-164). For the separable objectives the
per-object der reads `target.get(d * n + i)` (MultiCrossEntropy / MultiRMSE expect a dim-major
`target` of length `k·n`) and `weight.get(i)` (expects length `n`), each falling back to
`0.0`/`1.0` via `unwrap_or` when out of range (lines 179, 201-227). A caller passing a too-short
`target`/`weight` therefore gets a *silently wrong* der (leaf block computed from zeros) rather
than the `CbError::LengthMismatch` the sibling seams (`launch_pointwise_hist2_into`,
`query_rmse_ders_host`) raise — defeating the phase-wide "typed error, never a silent wrong
result" contract.
**Fix:** Validate the objective-specific `target` length and `weight` length up front:
```rust
let expected_target = match objective {
    MulticlassObjective::MultiCrossEntropy | MulticlassObjective::MultiRmse => k_n,
    _ => n, // per-object class column (Softmax/OneVsAll) or scalar target (RMSEWithUncertainty)
};
if target.len() != expected_target {
    return Err(CbError::LengthMismatch { column: "target".into(), expected: expected_target, actual: target.len() });
}
if !weight.is_empty() && weight.len() != n {
    return Err(CbError::LengthMismatch { column: "weight".into(), expected: n, actual: weight.len() });
}
```

### WR-03: Ordered-trajectory apply can read `delta` out of bounds on device when a tree's permutation is shorter than `n`

**File:** `crates/cb-backend/src/gpu_runtime/ordered.rs:180-226` (`accumulate_ordered_trajectory`)
**Issue:** `ordered_approx_delta(tree)` returns a vector of length `tree.permutation.len()`
(ordered.rs:96, 102), folded via
`launch_apply_leaf_delta_into(&client, trajectory_h, identity_h, &delta, 1.0, n)` with an
identity leaf map of length `n` (lines 202-217). The apply kernel computes
`trajectory[i] += lr * delta[identity[i]]` for `i in 0..n`, indexing `delta[i]` up to `n-1`. The
launcher (`gpu_runtime/mod.rs:2688-2738`) only guards `n == 0 || leaf_values.is_empty()` — it
does NOT verify `delta.len() >= n` (nor that `leaf_of[i] < n_leaves` in general). If any tree has
`permutation.len() < n`, the device gathers past the end of the `delta` buffer (out-of-bounds
device read / UB). Nothing in `accumulate_ordered_trajectory` enforces `permutation.len() == n`.
**Fix:** Guard the per-tree invariant before the resident apply:
```rust
for tree in trees {
    if tree.permutation.len() != n {
        return Err(CbError::LengthMismatch {
            column: "ordered permutation".into(), expected: n, actual: tree.permutation.len(),
        });
    }
    let delta = ordered_approx_delta(tree)?;
    ...
}
```
(and/or add a `leaf_values.len() >= n` guard in `launch_apply_leaf_delta_into`).

## Info

### IN-01: Deterministic fixed-point group reduction can silently overflow i64 for large `value·weight`

**File:** `crates/cb-backend/src/kernels/query_helper.rs:143-154` (`compute_group_means_kernel`)
**Issue:** The order-independent group-mean numerator/denominator accumulate
`u64::cast_from(i64::cast_from(f64::round(prod * REDUCE_FIXEDPOINT_SCALE_F64)))` with
`scale = 2^30`. For `|value·weight| ≳ 8.6e9` the product exceeds `i64::MAX`, and the `f64 → i64`
cast result is backend-defined (saturate/UB), silently corrupting the group mean. The covered
regime (small mean-removed residuals) stays far inside this, and it mirrors the accepted
phase-10/11 fixed-point pattern that has passed oracles — informational only, but no host guard
rejects an out-of-range magnitude before the kernel runs.
**Fix (optional hardening):** document/enforce a magnitude precondition on the ranking inputs,
or reject residuals whose `|value·weight|·scale` would exceed `i64::MAX` with a typed error.

### IN-02: Misleading identifier and redundant multiply in the stochastic ranking der

**File:** `crates/cb-backend/src/gpu_runtime/ranking.rs:868,900`
**Issue:** `let order_global = descending_order_per_query(seg, &local_offsets)?;` actually holds
*local* (`0..qs`) indices, not global doc ids — the name invites a future `begin`-offset bug.
Separately, `let w_f32 = 1.0_f32 * cwl / denom;` carries a no-op `1.0_f32 *` (the covered
`queryWeight == 1.0`); harmless but obscures intent.
**Fix:** rename `order_global` → `order_local`; drop the `1.0_f32 *` factor (or bind
`query_weight` explicitly).

### IN-03: In-kernel `q_offsets.len() - 1` relies entirely on the host non-empty guarantee

**File:** `crates/cb-backend/src/gpu_runtime/ranking.rs:178,269`; `crates/cb-backend/src/kernels/query_helper.rs:107,137,175,244,278,295`
**Issue:** Every serial grouping kernel computes `let n_groups = q_offsets.len() - 1;` inside the
`#[cube]` body. On an empty `q_offsets` this underflows (device wrapping) and would drive an
unbounded loop. All host wrappers currently reject empty `q_offsets`
(`validate_ranking_inputs`, plus `saturating_sub` in the `*_host` wrappers) before launch, so it
is not reachable — noted only because the invariant lives entirely on the host side and is easy
to break when a new caller is added.
**Fix (optional):** have callers pass `n_groups` explicitly rather than recomputing `len() - 1`
device-side.

---

_Reviewed: 2026-07-04T11:36:03Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
