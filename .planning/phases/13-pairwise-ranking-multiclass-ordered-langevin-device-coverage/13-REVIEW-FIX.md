---
phase: 13-pairwise-ranking-multiclass-ordered-langevin-device-coverage
fixed_at: 2026-07-04T12:00:00Z
review_path: .planning/phases/13-pairwise-ranking-multiclass-ordered-langevin-device-coverage/13-REVIEW.md
iteration: 1
findings_in_scope: 6
fixed: 5
skipped: 1
status: partial
---

# Phase 13: Code Review Fix Report

**Fixed at:** 2026-07-04T12:00:00Z
**Source review:** .planning/phases/13-pairwise-ranking-multiclass-ordered-langevin-device-coverage/13-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 6 (fix_scope = all — Warning + Info)
- Fixed: 5 (WR-01, WR-02, WR-03, IN-01, IN-02)
- Skipped: 1 (IN-03)

All five Phase-13 device families remain gated to `Ok(None)` → CPU fallback; none of these
fixes changes the coverage gate. They harden the latent drivers that the self-oracles exercise,
so the kernels are reached only through the frozen self-oracle fixtures. Every fix was verified
with `cargo check -p cb-backend` (clean: only the pre-existing dead-code warnings on the
coverage-gated families remain). GPU (rocm/cuda) self-oracles cannot be run in this environment;
re-validation of the affected oracles on real hardware (P100 CUDA at ε≤1e-4) is a follow-up.

## Fixed Issues

### WR-01: YetiRank/PFound-F in-query sort tie-break diverges from upstream stable-descending sort

**Files modified:** `crates/cb-backend/src/gpu_runtime/ranking.rs`
**Commit:** fde6130
**Applied fix:** Replaced the reverse-of-stable-ascending descending order in
`descending_order_per_query` with a genuine stable-descending order: the 64-bit radix key is now
the bitwise complement `!v.to_bits()` of each non-negative perturbed value, so a single stable
2-pass ascending radix already yields the descending value order while keeping tied documents in
ORIGINAL index order (matching upstream `yetirank_helpers.cpp:326-331`). Removed the per-query
`seg.reverse()` that flipped ties into reversed-index order. Docstring updated to describe the
complemented-key rationale.

**Requires human verification:** this is an algorithm/logic change to the parity-critical sort.
`cargo check` confirms it compiles, but semantic correctness (bit-for-bit parity with the CPU
`yetirank_sample_pairs` reference, especially on tied `exp(approx)·ratio`) must be re-confirmed
against the ranking self-oracles on real GPU hardware before the grow seam is wired.

### WR-02: `assemble_multiclass_ders` silently substitutes 0 for a short `target`/`weight`

**Files modified:** `crates/cb-backend/src/gpu_runtime/multiclass.rs`
**Commit:** 644d075
**Applied fix:** Added up-front objective-specific length validation before the per-object loop.
`target` must be `k·n` for MultiCrossEntropy / MultiRmse (dimension-major label matrix) and `n`
for Softmax / OneVsAll / RmseWithUncertainty (per-object column) — mismatches now raise
`CbError::LengthMismatch { column: "target", .. }`. A non-empty `weight` whose length is not `n`
raises `CbError::LengthMismatch { column: "weight", .. }`. Used an exhaustive `match` (no
wildcard) so a future objective variant forces a deliberate length decision. Existing self-oracle
fixtures (Softmax/OneVsAll target len n, RmseWithUncertainty target len n) pass unchanged.

### WR-03: Ordered-trajectory apply can read `delta` out of bounds when a tree's permutation is shorter than `n`

**Files modified:** `crates/cb-backend/src/gpu_runtime/ordered.rs`
**Commit:** c7c756f
**Applied fix:** Added a per-tree guard `tree.permutation.len() != n → CbError::LengthMismatch`
at the top of the `accumulate_ordered_trajectory` loop, before `ordered_approx_delta` and the
identity-mapped resident device apply. This prevents the device from gathering `delta[i]` for
`i in 0..n` past the end of a shorter `delta` buffer (out-of-bounds device read / UB). The
self-oracle fixture uses an 8-element permutation with n=8, so the guard does not fire on valid
oracles.

### IN-01: Deterministic fixed-point group reduction can silently overflow i64 for large `value·weight`

**Files modified:** `crates/cb-backend/src/kernels/query_helper.rs`
**Commit:** 74ad627
**Applied fix:** Added an optional-hardening host precondition in `compute_group_means_host`
(non-wgpu path): before launching `compute_group_means_kernel`, reject any input where
`|value·weight|` or `|weight|` exceeds `i64::MAX / REDUCE_FIXEDPOINT_SCALE_F64` (≈8.59e9) with a
typed `CbError::OutOfRange`. This turns the previously backend-defined f64→i64 cast overflow into
an explicit up-front error. The covered mean-removed-residual regime stays far inside the bound,
so oracles are unaffected.

### IN-02: Misleading identifier and redundant multiply in the stochastic ranking der

**Files modified:** `crates/cb-backend/src/gpu_runtime/ranking.rs`
**Commit:** 2e8d6c3
**Applied fix:** Renamed `order_global` → `order_local` (the value holds local `0..qs` indices,
not global doc ids) at its binding and both use sites. Dropped the no-op `1.0_f32 *` factor from
`let w_f32 = 1.0_f32 * cwl / denom;`, replacing it with `let w_f32 = cwl / denom;` plus a comment
noting the covered `queryWeight == 1.0`. Pure clarity change — no numeric behavior change.

## Skipped Issues

### IN-03: In-kernel `q_offsets.len() - 1` relies entirely on the host non-empty guarantee

**File:** `crates/cb-backend/src/gpu_runtime/ranking.rs:178,269`;
`crates/cb-backend/src/kernels/query_helper.rs:107,137,175,244,278,295`
**Reason:** skipped — the suggested fix (pass `n_groups` explicitly instead of recomputing
`q_offsets.len() - 1` device-side) requires changing the `#[cube]` kernel signatures of ~9
grouping kernels across two files plus every matching host launcher. `#[cube]` signature changes
cannot be validated in this environment (rocm/cuda GPU tests do not run here), and the review
classifies this as optional/informational: the underflow is already unreachable because every
host wrapper rejects an empty `q_offsets` (`validate_ranking_inputs` + `saturating_sub` in the
`*_host` wrappers) before launch. Forcing an unvalidatable multi-kernel signature refactor for a
non-reachable Info finding carries more risk than the latent invariant it removes. Recommended to
fold into the grow-seam wiring work, when the ranking kernels are being re-touched and can be
re-run on GPU.
**Original issue:** Every serial grouping kernel computes `let n_groups = q_offsets.len() - 1;`
inside the `#[cube]` body; on an empty `q_offsets` this underflows (device wrapping) and would
drive an unbounded loop. Not reachable today because the invariant is enforced host-side.

---

_Fixed: 2026-07-04T12:00:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
