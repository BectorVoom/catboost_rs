---
phase: 11-depth-1-partition-aware-histograms-reduction-determinism-new
fixed_at: 2026-07-03T00:00:00Z
review_path: .planning/phases/11-depth-1-partition-aware-histograms-reduction-determinism-new/11-REVIEW.md
iteration: 1
findings_in_scope: 7
fixed: 7
skipped: 0
status: all_fixed
---

# Phase 11: Code Review Fix Report

**Fixed at:** 2026-07-03
**Source review:** .planning/phases/11-depth-1-partition-aware-histograms-reduction-determinism-new/11-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 7 (fix_scope = all — Warning + Info)
- Fixed: 7
- Skipped: 0

All work was done in an isolated git worktree and each fix committed atomically. The
GPU code paths (rocm/cuda) cannot be exercised in this cpu-only environment, so the
behavioral fix (WR-02 capability gate) was verified on the cpu path (it correctly returns
the new typed error) and the remaining rocm-runtime confirmation is noted as a residual.
`cargo build -p cb-backend` and `cargo check -p cb-backend --tests` are green after every
fix.

## Fixed Issues

### WR-01: Subtraction trick computed-then-discarded, wrong sibling pairing, misleading self-oracle

**Files modified:** `crates/cb-backend/src/gpu_runtime/mod.rs`
**Commit:** 3d58c68
**Applied fix:** Chose review option (a) — the safe, behavior-preserving path. Removed the
discarded per-level `launch_subtract_histograms_into` calls (result was bound to `_bigger`
and thrown away) from BOTH `grow_oblivious_tree_into` and `grow_oblivious_tree_resident`,
and dropped the now-unused `parent_hist_h` / `per_leaf` bookkeeping and the unrealized
"memory-lean" D-04 claim. Re-numbered the per-level step comments. The subtraction kernel
and its launcher are RETAINED (still unit-tested standalone) with a
`#[cfg_attr(not(test), allow(dead_code))]` attribute and a note explaining it is not yet
wired into a scored path. This eliminates the dead code, the latent wrong-pairing bug (the
`2k`/`2k+1` pairing that never matched the forward-bit routing is gone), and the false
"memory-lean" claim without touching the correct direct-fill scored path. Option (b) (wire
the derived `bigger` into the scorer with corrected forward-bit pairing) was NOT taken: it
is a behavioral GPU change that cannot be validated in this cpu-only environment.

### WR-02: depth-1/2 grow & boosting tests unguarded through the Atomic<u64> fill; production performs no u64-atomic capability gate

**Files modified:** `crates/cb-core/src/error.rs`, `crates/cb-backend/src/gpu_runtime/mod.rs`, `crates/cb-backend/src/kernels/grow_loop.rs`
**Commit:** b54aa3a
**Applied fix:**
- **Production gate:** Added a `device_supports_u64_atomic_add<R>()` helper (mirroring the
  existing `device_supports_f64_atomic_add`) and gated `launch_partition_hist2_resident_into`
  — the single choke point that BOTH `grow_oblivious_tree_into` (via
  `launch_partition_hist2_into`) and `grow_oblivious_tree_resident` route through — to return
  a new typed `CbError::Unsupported` BEFORE launch when the backend lacks `Atomic<u64>` add.
- **Error type:** Added `CbError::Unsupported(String)` to `cb-core` (all existing matches on
  `CbError` carry a catch-all arm, so the new variant does not break exhaustiveness).
- **Test guards:** Added the same `if !cfg!(any(feature = "rocm", feature = "cuda")) { …
  return; }` SKIP guard used by the depth-6 tests to the four flagged tests
  (`matches_cpu_greedy_search`, `cosine_matches_cpu_cosine_greedy_search`,
  `depth_gt_one_is_device_covered`, `matches_cpu_multi_tree_boosting`) PLUS a fifth
  same-class test found during verification (`run_to_run_structure_stability_reported`,
  which also routes `grow_boosting_pass` through the u64 fill and was already failing on cpu).
- **Verification:** Confirmed on the cpu backend that `device_supports_u64_atomic_add` returns
  `false` and the gate now surfaces the clean `Unsupported` error (previously the kernel launch
  failed less cleanly). The five guarded tests now SKIP instead of erroring on cpu.
- **Residual (needs rocm confirmation):** That the gate PASSES on rocm/cuda (so real training
  still runs) rests on gfx1100 advertising `Atomic<u64>` add — consistent with the existing
  `kernels::reduce` u64-atomic path and the Phase 10/11 memory notes, but not re-run here.

### WR-03: Reduction determinism covers only the fill; leaf-stat reduce still uses float atomics

**Files modified:** `crates/cb-backend/src/kernels.rs`, `bench/RESULTS.md`
**Commit:** 027b086
**Applied fix:** Chose review option (b) — scope the guarantee explicitly rather than change
the GPU accumulation strategy (option (a) is a behavioral device change requiring rocm/cuda
validation not available here). Added a "Determinism SCOPE (WR-03)" doc block to
`partition_update_kernel` stating that its naked float atomic makes leaf VALUES / predictions
non-bit-deterministic (ulp-level float-order variance, within ε=1e-4) while tree STRUCTURE
remains bit-identical, and added a matching scoping note to `bench/RESULTS.md` clarifying that
the "0 spread" gate applies to structure, not predictions.

### WR-04: Fixed-point encode silently wraps above ~2^33 with no guard

**Files modified:** `crates/cb-backend/src/kernels.rs`
**Commit:** 027b086
**Applied fix:** Documented the `|Σ| < 2^33` fixed-point RANGE precondition (tighter than the
`2^53` float-exactness bound already mentioned) next to `REDUCE_FIXEDPOINT_SCALE_F64` and on
`fixedpoint_encode`, including the safe-headroom rationale for the committed workloads and the
host-side magnitude estimate a large-`n`/large-magnitude fixture would need. A live in-kernel
guard is not added (a `#[cube]` kernel cannot surface a typed error); the doc places the
precondition on the caller. A host-side pre-launch magnitude check is left as a documented
follow-up rather than speculatively added without a fixture that needs it.

### IN-01: depth-1 serial reference skips degenerate splits while depth-6/device permit them

**Files modified:** `bench/generator.py`
**Commit:** 97a4f39
**Applied fix:** Chose the review's "document why" option (the pinned `expected_depth1_tree.json`
is Kaggle/human-gated, so changing reference admissibility that could alter it is out of scope
here). Added a comment at the `wl <= 0.0 or wr <= 0.0: continue` skip explaining that depth-1
uses a deliberately stricter rule than `_cosine_split_score`/the device, that the two agree
whenever no degenerate-side candidate wins (true for the committed 2000×10 gaussian fixture),
and exactly what to change (drop the `continue`, score the empty side with the zero-average
fold, regenerate the fixture) if a future fixture can produce a degenerate winner.

### IN-02: Misleading capability-gating comment in the resident partition-fill launcher

**Files modified:** `crates/cb-backend/src/gpu_runtime/mod.rs`
**Commit:** 5555855
**Applied fix:** Updated the comment at the fill's kernel-launch block so it now accurately
describes the real gate added for WR-02 — the launcher itself checks the advertised
`Atomic<u64>` capability and returns `CbError::Unsupported` before the launch — instead of
claiming an absent "caller gates this path" contract.

### IN-03: Per-object fixed-point rounding makes the device histogram a quantized approximation

**Files modified:** `bench/RESULTS.md`
**Commit:** 027b086
**Applied fix:** No fixture change is required (as the finding states). Added a
"Fixed-point quantization headroom vs n (IN-03)" note to `bench/RESULTS.md` recording the
`n·2^-31` per-bin error bound (≈2.3e-7 at n=500, ≈4.7e-4 at n≈1e6), the shrinking margin at
large `n`, and the instruction to track effective histogram error vs `n` in the Kaggle
per-tree diagnostic so the 1e-4 margin at ~1e6 rows is confirmed rather than assumed.

## Notes

- **Commit grouping:** WR-03, WR-04, and IN-03 are all documentation-only findings that ended
  up editing the same two files (`crates/cb-backend/src/kernels.rs` and `bench/RESULTS.md`).
  Because the commit tool stages whole files, they were committed together in `027b086` rather
  than as three separate commits; the per-finding intent is preserved in the doc text and this
  report.
- **Pre-existing cpu test failures (NOT caused by these fixes):** A full
  `cargo test -p cb-backend --lib` run on the cpu backend shows ~51 failures in
  `kernels::{reduce,scan,sort,score_split}` and 3 in `kernels::grow_loop::{partition,pairwise}`
  (`leaf_of_matches_cpu_leaf_index`, `update_matches_ordered_reference`,
  `matches_cpu_pairwise_grow`). These launch GPU primitive kernels directly on the cpu backend
  and were failing before any change in this session — none of the three grow_loop failures
  route through the WR-02-gated launcher, so the production gate is provably not their cause.
  They are outside the review's findings (the review flagged only the histogram-fill-routing
  tests) and are consistent with the project's rocm-in-env validation model.

---

_Fixed: 2026-07-03_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
