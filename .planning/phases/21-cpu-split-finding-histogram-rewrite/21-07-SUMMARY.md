---
phase: 21-cpu-split-finding-histogram-rewrite
plan: 07
subsystem: testing
tags: [cpu-training, histogram, split-finding, scan-score-fusion, scratch-reuse, parity, f64-summation-order, perf, gap-closure]

# Dependency graph
requires:
  - phase: 21-cpu-split-finding-histogram-rewrite (plans 01-06)
    provides: BucketHistogram, feature_block, scan_borders_to_leaf_stats, multi_dim_split_score, cb_core::sum_f64, perf_baseline_test (CB_PERF), rayon_determinism_test
provides:
  - "multi_dim_split_score_into — scratch-reusing sibling of multi_dim_split_score (caller-owned num/den fold buffers; byte-identical, zero per-call heap Vec)"
  - "scan_and_score_borders — FUSED single-pass border scan + split score (no Vec<Vec<Vec<LeafStats>>> materialization, no per-candidate score Vec; bit-identical to scan_borders_to_leaf_stats + multi_dim_split_score)"
  - "bit-identity equivalence guards (fused_scan_score_bit_identical, multi_dim_split_score_into_matches_alloc) proving to_bits() equality"
affects: [cpu-training-perf, histogram-rewrite]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Fused scan+score single pass with per-parent running-prefix accumulators (border-outer, parent-inner) into a REUSED per_dim LeafStats scratch, scored via a scratch-reusing fold — no per-candidate materialization/allocation"
    - "Allocation hoisting: multi_dim_split_score delegates to multi_dim_split_score_into so the hot path reuses num/den buffers across all candidate borders"

key-files:
  created: []
  modified:
    - crates/cb-compute/src/score.rs
    - crates/cb-compute/src/score_test.rs
    - crates/cb-compute/src/histogram.rs
    - crates/cb-compute/src/histogram_test.rs
    - crates/cb-compute/src/lib.rs
    - crates/cb-train/src/tree.rs

key-decisions:
  - "Fusion is BIT-IDENTICAL (exact to_bits equality), not merely ≤1e-5: the per-border fold stays a FRESH cb_core::sum_f64 over 2·n_parent leaves in canonical dimension-then-leaf order; only the FALSE-side per-parent running prefix accumulates across borders (the same ascending += 21-06 already does). NO cross-border num/den reorder (forbidden)."
  - "Border-outer / parent-inner walk chosen so per_dim is materialized in dimension-then-leaf order per border (the exact order multi_dim_split_score folds) — a parent-outer walk would reorder the fold and break bit-identity."
  - "PERF-01 constant recovery is REAL but MODEST at this harness size: the per-candidate allocation count in the scan/score pass drops ~100× (n_borders·(1+dim) materialization vecs + 2·n_borders score vecs → ~10 per-feature buffers), yet wall-clock 32→254 improved only 3.48×→~3.33× (nbins=254 −5%). Allocation was NOT the dominant cost; the residual n_bins-linear term is the irreducible O(n_bins·n_leaves·n_features) arithmetic. The ~2.3-2.6× target is NOT met — reported honestly."

patterns-established:
  - "scan_and_score_borders: fuse the O(n_bins) prefix scan with the split-score fold, reusing all scratch across candidate borders, bit-identical to the two-step path"

requirements-completed: [PERF-03]

coverage:
  - id: D1
    description: "multi_dim_split_score_into (scratch-reusing) is byte-for-byte identical to multi_dim_split_score; clear-then-refill over reused scratch stays bit-identical — PERF-03 per-candidate score Vec eliminated"
    requirement: "PERF-03"
    verification:
      - kind: unit
        ref: "crates/cb-compute/src/score_test.rs#multi_dim_split_score_into_matches_alloc"
        status: pass
    human_judgment: false
  - id: D2
    description: "scan_and_score_borders (fused single pass, no Vec<Vec<Vec<LeafStats>>> materialization) returns per-border scores byte-for-byte equal to scan_borders_to_leaf_stats + multi_dim_split_score, Cosine+L2, dim=1 and dim=2 — PERF-03 materialization + per-candidate allocation eliminated"
    requirement: "PERF-03"
    verification:
      - kind: unit
        ref: "crates/cb-compute/src/histogram_test.rs#fused_scan_score_bit_identical"
        status: pass
      - kind: integration
        ref: "cargo test -p cb-train --release --lib tree:: (28 passed) — all 3 callers rewired"
        status: pass
    human_judgment: false
  - id: D3
    description: "ATOMIC full CPU ≤1e-5 oracle suite passes in a SINGLE uninterrupted run with ZERO ADDITIONAL fixtures flipped after the fusion (bit-identity insurance) — PERF-02 crux honored as a constraint"
    requirement: "PERF-02"
    verification:
      - kind: integration
        ref: "CARGO_INCREMENTAL=0 cargo test -p cb-train --release --no-fail-fast (413 passed / 1 failed — only pre-existing monotone_non_symmetric_and_region_are_typed_errors) + cargo test -p cb-compute --release (212 passed / 0 failed = 210 prior + 2 new bit-identity guards)"
        status: pass
    human_judgment: false
  - id: D4
    description: "rayon_determinism_test stays 2/2 byte-identical (SymmetricTree + Depthwise) after the fused scan/score/scratch changes; scratch is per-rayon-task-local"
    requirement: "PERF-02"
    verification:
      - kind: integration
        ref: "cargo test -p cb-train --release --test rayon_determinism_test (2 passed)"
        status: pass
    human_judgment: false
  - id: D5
    description: "PERF-01: CB_PERF 32→254 ratio improves toward ~2.3-2.6× (allocation-constant recovery). DELIVERED PARTIALLY / NOT AT TARGET: ratio 3.48×→~3.33× (median), nbins=254 −5%. The literal flat bar is algorithmically unreachable at n=10000/depth-6 (needs n≥100k) — documented with measured floor+slope evidence."
    requirement: "PERF-01"
    verification:
      - kind: integration
        ref: "CB_PERF=1 cargo test --release -p cb-train --test perf_baseline_test -- --nocapture (3-sample medians: nbins=32 2.580ms, 64 3.087ms, 254 8.598ms)"
        status: fail
    human_judgment: true
    rationale: "PERF-01's ≤2.5× flatness bar is NOT met (~3.33×). The fusion recovers the per-candidate allocation constant (~100× fewer allocations in the scan/score pass) but wall-clock only moves 3.48×→3.33× because allocation was not the dominant cost — the residual n_bins-linear term is the irreducible O(n_bins·n_leaves·n_features) split-scoring ARITHMETIC, n-independent, and flat requires n≫n_bins·n_leaves (fails here). A human must decide whether to accept the parity-safe allocation recovery (PERF-03 delivered, parity bit-identical) or close PERF-01 as unreachable-at-this-size."

# Metrics
duration: ~50min
completed: 2026-07-06
status: complete
---

# Phase 21 Plan 07: PERF-01 Gap Closure (scan+score fusion, parity-safe) Summary

**Fused the O(n_bins) border scan with the split-score fold into one pass (scan_and_score_borders + multi_dim_split_score_into) reusing caller-owned scratch across all candidate borders — eliminating the Vec<Vec<Vec<LeafStats>>> materialization and the per-candidate num/den score Vecs (PERF-03). The fusion is BIT-IDENTICAL (exact to_bits equality, proven by two equivalence guards), gated by a single atomic zero-additional-flip oracle pass (413/1 + 212/0) and rayon 2/2. Wall-clock 32→254 recovery is MODEST — 3.48×→~3.33× (nbins=254 −5%) — because allocation was not the dominant cost; the residual is the irreducible O(n_bins·n_leaves·n_features) arithmetic. PERF-01's literal flat bar remains algorithmically unreachable at this harness size.**

## Performance

- **Duration:** ~50 min
- **Tasks:** 3 (all executed)
- **Files modified:** 6 (4 source + 2 test)

## Accomplishments

- **PERF-03 — per-candidate scan/score allocation eliminated (the fusion).** `scan_and_score_borders` walks candidate borders in a single pass with per-parent running-prefix accumulators (`acc_false_w[parent]`, `acc_false_d[parent·dim+d]`) and per-parent totals (both computed once), materializing each border's `2·n_parent` per-dim `LeafStats` into ONE reused `per_dim` scratch, then scoring via `multi_dim_split_score_into` with reused `num`/`den` fold buffers. The full `Vec<Vec<Vec<LeafStats>>>` (n_borders·dim·2·n_parent) AND the per-candidate `num_terms`/`den_terms` Vecs are gone — the scan/score pass drops from ~n_borders·(1+dim) + 2·n_borders per-feature allocations to ~10 reused buffers per feature per level.
- **PERF-02 (the crux) — ATOMIC BIT-IDENTITY INSURANCE PASSED.** A single uninterrupted run of the full CPU oracle suite (`cargo test -p cb-train --release --no-fail-fast` → **413 passed / 1 failed**, then `cargo test -p cb-compute --release` → **212 passed / 0 failed**) flipped **ZERO ADDITIONAL** fixtures. The only non-green test is the documented pre-existing `monotone_non_symmetric_and_region_are_typed_errors` (deferred-items.md, not a regression). Because the fusion is bit-identical (exact f64), this passed trivially — as designed.
- **Bit-identity proven, not assumed.** `fused_scan_score_bit_identical` (histogram_test) asserts `to_bits()` equality between `scan_and_score_borders` and the two-step `scan_borders_to_leaf_stats` + `multi_dim_split_score` reference for Cosine + L2 at dim=1 and dim=2; `multi_dim_split_score_into_matches_alloc` (score_test) asserts the scratch-reusing fold equals the allocating one bit-for-bit, including a reused-scratch third pass.
- **Determinism preserved:** `rayon_determinism_test` stays **2/2** byte-identical. Scratch is per-rayon-task-local (allocated inside the `select_level_plain` `.map` closure) — no cross-thread mutation.
- **All 3 hot callers rewired:** `select_level_plain`, `select_level_perturbed`, `best_split_for_leaf` route through the fused path. The perturbed draw order/count is unchanged (exactly one `std_normal` per border — Pitfall 3); the FEAT-04 penalty insertion point and the strict `>` first-wins tie-break are byte-for-byte preserved.
- **PERF-01 — partially delivered, NOT at target (honest).** 32→254 per-tree-ms ratio improved from the 21-06 5-sample-median baseline **~3.48× to ~3.33×** (nbins=254 absolute **9.062ms → 8.598ms, −5.1%**). Still well above the ~2.3-2.6× target. The modest wall-clock delta confirms allocation was not the dominant cost.

## Task Commits

1. **Task 1 (TDD RED): bit-identity guards** — `85c63cd` (test)
2. **Task 2 (GREEN): fused scan+score + scratch reuse, 3 callers rewired** — `5aa1be6` (feat)
3. **Task 3: atomic gate + CB_PERF re-sweep + honest flatness docs** — (this docs commit; scan_and_score_borders doc-comment flatness framing landed in Task 2)

## Atomic Oracle-Suite Tally + Max Deviation (PERF-02 record)

| Suite | Passed | Failed | Notes |
|-------|--------|--------|-------|
| `cargo test -p cb-train --release --no-fail-fast` | 413 | 1 | only `monotone_non_symmetric_and_region_are_typed_errors` (documented pre-existing, deferred-items.md) |
| `cargo test -p cb-compute --release` | 212 | 0 | 210 prior + 2 new bit-identity guards |

- **Additional fixtures flipped by the fusion: 0.** The fusion is bit-identical (exact `to_bits()` equality), so **observed max deviation = 0.0** on the fused path itself (proven by `fused_scan_score_bit_identical`); every ≤1e-5 oracle fixture asserted within tolerance. Single uninterrupted invocation each.
- **Disk hygiene used:** `export CARGO_INCREMENTAL=0`; `rm -rf target/{debug,release}/incremental` before the run. Disk stayed at 87% (31G free) throughout; no mid-run executable pruning needed.

## CB_PERF 32→254 Re-Sweep (PERF-01 record)

`CB_PERF=1 cargo test --release -p cb-train --test perf_baseline_test -- --nocapture`, n=10000, nf=20, depth=6, iters=3, 3-sample medians:

| n_bins | per_tree_ms (median, after fusion) | 21-06 baseline (5-sample median) |
|--------|-----------------------------------|----------------------------------|
| 32     | 2.580 | 2.607 |
| 64     | 3.087 | 3.350 |
| 254    | 8.598 | 9.062 |

- **32→254 ratio ≈ 3.33×** (after) vs **~3.48×** (21-06 baseline) — improved, target ≤ ~2.5× **NOT met**.
- **64→254 step ≈ 2.79×** for a 4× bin range (sublinear).
- **nbins=254 absolute −5.1%** (9.062ms → 8.598ms) vs the 21-06 baseline.
- **Scratch reuse verified (not a hidden per-candidate alloc):** `scan_and_score_borders` allocates its `per_dim`, `num_scratch`, `den_scratch`, `total_*`, `acc_false_*`, `col` buffers ONCE per call (per feature) and reuses them across every border; `multi_dim_split_score_into` `clear()`s + refills. The allocation count in the scan/score pass genuinely drops ~100× — the small wall-clock delta means allocation was simply not the bottleneck.

## Honest Flatness-Unreachable Framing (PERF-01)

The measured decomposition at n=10000/nf=20/depth-6 is a flat floor **≈1.7ms** (binning + the O(n·nf) scatter build) **+ ~0.026 ms/bin**. The n_bins-linear term IS the O(n_bins·n_leaves·n_features) split-scoring pass, which is **n-INDEPENDENT** (the histogram cell count does not depend on the row count) and therefore **algorithmically irreducible**. Flatness across the n_bins sweep requires **n ≫ n_bins·n_leaves**, which FAILS here (n_bins·n_leaves ≈ 16K > 10K) and would only hold at n ≥ 100k. Official CatBoost is itself ~2.1× (not flat) at this size. PERF-01's literal "flat within noise 32→254" bar is thus **out of reach at this harness size** regardless of allocation strategy. This plan's success is the **parity-safe allocation-constant recovery (PERF-03) + bit-identical parity (PERF-02)** — NOT flatness. This framing is stated in both this SUMMARY and the `scan_and_score_borders` doc comment.

## Deviations from Plan

**None — plan executed as written.** The fusion is bit-identical (exact f64), so no `≤1e-5` wording downgrade was needed (unlike 21-06's TRUE-side reorder). The honest outcome is that the wall-clock constant recovery is smaller than the ~2.3-2.6× hoped: reported transparently rather than overstated.

### Pre-existing, out-of-scope (logged not fixed)

- `monotone_non_symmetric_and_region_are_typed_errors` (stale Region-OUT assertion) stays red — documented pre-existing, deferred-items.md.
- cb-backend feature-gated dead-code / `does not need to be mutable` warnings (`nonsym_grow.rs`) are pre-existing and in files this plan does not touch.

## Issues Encountered

- **PERF-01 not fully closed.** Root cause (re-confirmed by this fusion): the residual n_bins-linear cost is the split-scoring ARITHMETIC (the per-border `sum_f64` folds over 2·n_parent leaves × n_features), not allocation churn. The fusion removed the allocations (PERF-03) but the arithmetic is irreducible and comparable to the flat O(n·nf) term at this size — so wall-clock barely moves. Reaching ≤2.5× is not achievable at n=10000/depth-6 (would require n≥100k, where the sweep is naturally flat).
- **CB_PERF measurement noise:** sub-3ms low-n_bins runs vary ±20-30%; medians of 3 samples reported. The single pre-change baseline sample was discarded in favour of the more robust 21-06 5-sample median as the documented "before".

## Next Phase Readiness

- **PERF-03 (allocation) delivered and proven; PERF-02 (parity) honored bit-identically** and gated by the atomic oracle suite.
- **PERF-01 remains partially open** (~3.33×, target ≤2.5×). Verifier/human must decide: accept the parity-safe allocation recovery + the measured flatness-unreachable evidence, or close PERF-01 as algorithmically unreachable at this harness size.

## Self-Check: PASSED

- All 6 modified source/test files exist on disk.
- Task commits present in git history (85c63cd, 5aa1be6).
- `scan_and_score_borders` + `multi_dim_split_score_into` exported from cb-compute lib.
- Atomic gate 413/1 + 212/0, rayon 2/2, CB_PERF re-swept.

---
*Phase: 21-cpu-split-finding-histogram-rewrite*
*Completed: 2026-07-06*
