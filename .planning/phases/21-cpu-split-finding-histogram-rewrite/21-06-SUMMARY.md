---
phase: 21-cpu-split-finding-histogram-rewrite
plan: 06
subsystem: testing
tags: [cpu-training, histogram, split-finding, prefix-scan, subtraction-trick, parity, f64-summation-order, perf]

# Dependency graph
requires:
  - phase: 21-cpu-split-finding-histogram-rewrite (plans 01-05)
    provides: BucketHistogram, build_bucket_histogram, scan_border_to_leaf_stats, GrowScratch::advance, cb_core::sum_f64, perf_baseline_test (CB_PERF), rayon_determinism_test
provides:
  - "cb_core::scatter_add_f64 — sanctioned object-order scatter-add primitive (D-07/D-08 without per-cell Vec)"
  - "Flat-scratch build_bucket_histogram (single reused f64 accumulator, zero per-cell heap Vecs — PERF-03)"
  - "Single-pass O(n_bins) running-prefix border scan (FALSE = running prefix bit-identical; TRUE = total − prefix, the authorized reorder)"
  - "BucketHistogram::add / relocate / relocate_sub / add_relocated (single-allocation histogram-algebra primitives)"
  - "Retained-parent subtraction advance (larger sibling derived from self.hist; only the smaller sibling built — WR-04)"
  - "ATOMIC full-oracle parity proof authorizing the TRUE-side summation-order change (PERF-02 crux)"
affects: [cpu-training-perf, histogram-rewrite, future-scan-scoring-fusion]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Scatter-add accumulation via a sanctioned cb_core primitive (no per-cell gather Vec, D-08-safe)"
    - "Running-prefix single-pass border scan with TRUE = total − FALSE-prefix (upstream CalcScoresForLeaf complement)"
    - "Fused single-allocation histogram algebra (relocate_sub, add_relocated) for the subtraction trick"

key-files:
  created: []
  modified:
    - crates/cb-core/src/reduction.rs
    - crates/cb-core/src/reduction_test.rs
    - crates/cb-core/src/lib.rs
    - crates/cb-compute/src/histogram.rs
    - crates/cb-compute/src/histogram_test.rs
    - crates/cb-train/src/tree.rs

key-decisions:
  - "TRUE-side primary strategy (total − prefix) HELD — the atomic zero-flip oracle pass authorized it; no fallback (descending suffix / fixed-point-u64) was needed"
  - "PERF-01 flatness lever partially delivered: ratio improved 4.3x -> ~3.4x but NOT at the ≤2.5x target; residual is inherent O(n_bins·n_leaves·n_features) scan + subtraction-trick cell-algebra + per-candidate scoring, comparable to (not dominated by) the flat O(n·n_features) term at n=10000/depth-6"
  - "Optimized the scan/advance CONSTANT factor (contiguous-block reads, 5->3 per-level allocations, single-pass add) as the plan's 'investigate the residual n_bins term' step — all parity-safe (bit-identical)"

patterns-established:
  - "scatter_add_f64: SCATTER form of sum_f64 — repeated ascending scatter-add into a slot == sum_f64 of those members, bit-for-bit"
  - "TRUE = total − FALSE running prefix as the O(n_bins) border-scan complement, gated by the atomic oracle suite"

requirements-completed: [PERF-02, PERF-03]

coverage:
  - id: D1
    description: "cb_core::scatter_add_f64 primitive + flat-scratch build_bucket_histogram (no per-cell heap Vec; bit-identical to gather-then-sum_f64) — PERF-03 build allocation eliminated"
    requirement: "PERF-03"
    verification:
      - kind: unit
        ref: "crates/cb-core/src/reduction_test.rs#scatter_add_matches_sum_f64, scatter_add_out_of_range_is_noop"
        status: pass
      - kind: unit
        ref: "crates/cb-compute/src/histogram_test.rs#build_flat_scatter_equals_gather"
        status: pass
    human_judgment: false
  - id: D2
    description: "Single-pass running-prefix O(n_bins) border scan (FALSE bit-identical, TRUE = total − prefix) + retained-parent subtraction advance (WR-03/WR-04); no per-border Vec, advance builds one small sibling not three full histograms — PERF-03 scan/advance allocation eliminated"
    requirement: "PERF-03"
    verification:
      - kind: unit
        ref: "crates/cb-compute/src/histogram_test.rs#running_prefix_scan_matches_per_border_reference, scan_border_matches_rescan_scalar, scan_border_matches_rescan_multiclass"
        status: pass
      - kind: unit
        ref: "crates/cb-train/src/tree.rs tree:: lib tests (28 passed)"
        status: pass
    human_judgment: false
  - id: D3
    description: "ATOMIC full CPU ≤1e-5 oracle suite passes in a SINGLE uninterrupted run with ZERO fixtures flipped after the TRUE-side reorder is enabled — the gate that authorizes the summation-order change (PERF-02 crux)"
    requirement: "PERF-02"
    verification:
      - kind: integration
        ref: "CARGO_INCREMENTAL=0 cargo test -p cb-train --release --no-fail-fast (413 passed / 1 failed) + cargo test -p cb-compute --release (210 passed / 0 failed); only pre-existing monotone_non_symmetric_and_region_are_typed_errors failed"
        status: pass
    human_judgment: false
  - id: D4
    description: "rayon_determinism_test stays 2/2 byte-identical (SymmetricTree + Depthwise) after the scan/scratch/advance changes"
    requirement: "PERF-02"
    verification:
      - kind: integration
        ref: "cargo test -p cb-train --release --test rayon_determinism_test (2 passed)"
        status: pass
    human_judgment: false
  - id: D5
    description: "PERF-01: CB_PERF n_bins sweep per-tree time FLAT within noise across border_count 32→254 (target ratio ≤ ~2.5×, 64→254 step ≤ ~1.7×). DELIVERED PARTIALLY: ratio improved from the verifier's ~4.3× to ~3.4× (median), 64→254 = 2.71× (sublinear), nbins=254 absolute −33% — but NOT at the ≤2.5× / ≤1.7× target."
    requirement: "PERF-01"
    verification:
      - kind: integration
        ref: "CB_PERF=1 cargo test --release -p cb-train --test perf_baseline_test -- --nocapture (5-sample medians: nbins=32 2.607ms, 64 3.350ms, 254 9.062ms)"
        status: fail
    human_judgment: true
    rationale: "PERF-01's ≤2.5× flatness bar is NOT met (achieved ~3.4×). A human must decide whether the substantial improvement (4.3×→3.4×, −33% at nbins=254) is acceptable, or whether the remaining n_bins-linear residual (scan + per-candidate scoring + subtraction-trick cell-algebra, all O(n_bins·n_leaves·n_features)) warrants a further scan/scoring FUSION pass (invasive, would re-touch the parity crux and require re-running the atomic gate)."

# Metrics
duration: ~55min
completed: 2026-07-06
status: complete
---

# Phase 21 Plan 06: PERF-01 Gap Closure (histogram data-layer flatness) Summary

**TRUE-side running-prefix reorder AUTHORIZED by a single atomic zero-flip ≤1e-5 oracle pass (413+210 tests, only the documented pre-existing monotone test non-green); scatter-add build + single-pass scan + 3-alloc subtraction advance eliminate the per-level allocation (PERF-03) and cut the n_bins ratio 4.3×→~3.4× — improved but SHORT of the ≤2.5× PERF-01 target.**

## Performance

- **Duration:** ~55 min
- **Started:** 2026-07-05T21:00Z (approx)
- **Completed:** 2026-07-05T21:56Z
- **Tasks:** 3 (all executed)
- **Files modified:** 6 source/test + 1 planning (deferred-items.md)

## Accomplishments
- **PERF-02 (the crux) — ATOMIC PARITY GATE PASSED.** A single uninterrupted run of the full CPU oracle suite (`cargo test -p cb-train --release --no-fail-fast` → **413 passed / 1 failed**, then `cargo test -p cb-compute --release` → **210 passed / 0 failed**) flipped **ZERO** oracle fixtures across losses/CTR/ranking (lambdamart, pairlogit)/ordered/multiclass/multilabel/multiquantile. The only non-green test is the documented pre-existing `monotone_non_symmetric_and_region_are_typed_errors` (a stale Region-OUT assertion, deferred-items.md — not a regression). **This authorizes the TRUE-side `total − prefix` summation-order reorder deferred three times (21-01→02→05).** The primary strategy held; no fallback needed.
- **PERF-03 — per-level histogram allocation eliminated.** `build_bucket_histogram` scatter-adds into ONE flat `f64` accumulator via the new `cb_core::scatter_add_f64` (no `Vec<Vec<f64>>` per-cell gather); the border scan carries a single running prefix with no per-border Vec; `advance` builds only the SMALLER sibling (one O(n) scatter) and derives the larger from the retained parent — 3 total-sized allocations per level, down from the old 3× from-scratch rebuild.
- **Determinism preserved:** `rayon_determinism_test` stays **2/2** byte-identical (SymmetricTree + Depthwise).
- **PERF-01 — partially delivered (NOT at target).** The 32→254 per-tree-ms ratio improved from the verifier's ~4.3× to **~3.4×** (5-sample median 9.062/2.607); the 64→254 step is **2.71×** (sublinear vs the 4× bin range); nbins=254 absolute dropped **13.5ms → ~9.0ms (−33%)**. Still above the ≤2.5× / ≤1.7× bar.
- **CR-01 doc honesty:** the reordered TRUE-side scan/advance claims are downgraded to "≤1e-5 oracle-equivalent"; legitimately-still-byte-identical claims (FALSE-side prefix, scatter build, forward-bit leaf ORDER, relocate memcpy) keep their wording.

## Task Commits

1. **Task 1 (TDD): scatter_add_f64 + flat-scratch build** — `cbd0a5b` (test, RED) → `fedf7c1` (feat, GREEN)
2. **Task 2: running-prefix scan + retained-parent advance** — `6dd1da8` (feat)
3. **Task 3: CR-01 wording + PERF-01 residual optimization** — `c60ea9b` (docs, CR-01) → `97aba66` (perf: contiguous-block scan + 3-alloc advance)

**Plan metadata:** (this docs commit)

## Files Created/Modified
- `crates/cb-core/src/reduction.rs` — `scatter_add_f64` (sanctioned scatter form of `sum_f64`)
- `crates/cb-core/src/reduction_test.rs` — `scatter_add_matches_sum_f64`, `scatter_add_out_of_range_is_noop`
- `crates/cb-core/src/lib.rs` — re-export `scatter_add_f64`
- `crates/cb-compute/src/histogram.rs` — flat-scratch build; single-pass running-prefix scan (FALSE prefix / TRUE = total − prefix); `feature_block`, `add`, `relocate`, `relocate_sub`, `add_relocated`
- `crates/cb-compute/src/histogram_test.rs` — `build_flat_scatter_equals_gather`, `running_prefix_scan_matches_per_border_reference`
- `crates/cb-train/src/tree.rs` — 3-alloc `advance` (retained-parent subtraction), single-pass `hist_add`, CR-01 wording

## Atomic Oracle-Suite Tally + Max Deviation (PERF-02 record)

| Suite | Passed | Failed | Notes |
|-------|--------|--------|-------|
| `cargo test -p cb-train --release --no-fail-fast` | 413 | 1 | only `monotone_non_symmetric_and_region_are_typed_errors` (documented pre-existing, deferred-items.md) |
| `cargo test -p cb-compute --release` | 210 | 0 | — |

- **Fixtures flipped by the reorder: 0.** Every ≤1e-5 oracle fixture asserted within its tolerance; **observed max deviation < the fixtures' ≤1e-5 bar** (no fixture reported a deviation exceeding its tolerance — the gate is defined as zero-flip, which is met). No large tie-flip / structure divergence (Pitfall 1) occurred.
- **Disk hygiene used:** `export CARGO_INCREMENTAL=0`; `rm -rf target/{debug,release}/incremental` before the run. Disk stayed at 79–80% throughout; no mid-run executable pruning was needed.
- **TRUE-side strategy that passed:** PRIMARY `true = total − acc_false` (FALSE = ascending running prefix, `total = sum_f64(bins)` once per parent/channel). No fallback (descending suffix / fixed-point-u64) required.

## CB_PERF 32→254 Re-Sweep (PERF-01 record)

`CB_PERF=1 cargo test --release -p cb-train --test perf_baseline_test -- --nocapture`, n=10000, nf=20, depth=6, iters=3, 5-sample medians:

| n_bins | per_tree_ms (median) |
|--------|----------------------|
| 32     | 2.607 |
| 64     | 3.350 |
| 254    | 9.062 |

- **32→254 ratio ≈ 3.48×** (measurement-noisy; observed 2.4×–3.5× across runs) — improved from the verifier's ~4.3×, target ≤ ~2.5× **NOT met**.
- **64→254 step ≈ 2.71×** for a 4× bin increase (sublinear, but target ≤ ~1.7× **NOT met**).
- **nbins=254 absolute −33%** (13.5ms → ~9.0ms) vs the pre-optimization measurement.

## Decisions Made
- Kept the PRIMARY `total − prefix` TRUE-side (the atomic gate authorized it); no fallback.
- Treated the "investigate the residual n_bins term" clause as an in-scope directive and optimized the scan/advance CONSTANT (contiguous-block reads via `feature_block`; single-pass `add`; fused `relocate_sub` / `add_relocated`; advance 5→3 allocations) — all bit-identical, verified by the equivalence tests and the atomic gate.

## Deviations from Plan

### Auto-fixed / In-scope investigation

**1. [Rule 3 - Blocking on PERF-01 goal] Scan/advance constant-factor optimization**
- **Found during:** Task 3 (CB_PERF re-sweep — first measurement gave 3.25×, above target).
- **Issue:** The initial Task-1/2 implementation (per-cell `channel()` accessor gather; `hist_add` via 3× `remove`; advance building the smaller sibling twice + relocating the parent) left the 32→254 ratio at ~3.3–4.1×.
- **Fix:** Added `BucketHistogram::feature_block` (contiguous per-(parent,feature) block read), `add` (single-pass a+b replacing the a−(0−b) triple-remove), and fused `relocate_sub`/`add_relocated`; rewrote `advance` to build only the smaller sibling once and derive the larger via `relocate_sub` (5→3 per-level allocations). All bit-identical (IEEE: `−b+s == s−b`, `self+relocated` same operand order).
- **Files modified:** crates/cb-compute/src/histogram.rs, crates/cb-train/src/tree.rs
- **Verification:** all histogram + tree equivalence tests green; rayon determinism 2/2; the atomic oracle gate zero-flip.
- **Committed in:** 97aba66

**2. [Pre-existing, out-of-scope — logged not fixed] `scripts/check-no-raw-float-sum.sh` red**
- The D-08 backstop script exits 1 on pre-existing `usize` sums (`score.rs`, `update_part_props.rs`) and doc-comment `.sum()` mentions (`leaf.rs`, cb-backend kernels) — all on HEAD before this session, in files this plan does not touch. The file this plan rewrites (`histogram.rs`) is CLEAN of any raw float fold (`build_bucket_histogram` routes through `cb_core::scatter_add_f64`). Logged in `deferred-items.md`. Task 1's intent ("no raw float fold introduced in cb-compute") is satisfied.

---

**Total deviations:** 1 in-scope optimization (Rule 3, PERF-01 goal) + 1 pre-existing out-of-scope logged.
**Impact on plan:** The optimization is parity-safe and necessary to move PERF-01 toward target. No scope creep beyond the plan's "investigate the residual" directive.

## Issues Encountered
- **PERF-01 not fully closed.** Root cause (investigated): the residual n_bins-linear cost is the border SCAN + per-candidate SCORING + subtraction-trick CELL-ALGEBRA, all O(n_bins·n_leaves·n_features). At n=10000/depth-6 these are COMPARABLE to (not dominated by) the flat O(n·n_features) binning/build term — so the plan's truth ("binning dominates the residual scan") does not hold at this harness size; even upstream CatBoost is ~2.1× (not flat) here. Reaching ≤2.5× would require fusing the scan with the score calcer (no `out` materialization), which is invasive, re-touches the ≤1e-5 parity crux, and would require re-running the (expensive) atomic gate. Deferred to a human decision (coverage D5, human_judgment).
- **CB_PERF measurement noise:** sub-10ms runs at low n_bins vary ±30%; medians of 5 samples reported.

## Next Phase Readiness
- **PERF-02 (parity crux) and PERF-03 (allocation) are fully delivered and proven.** The TRUE-side reorder is now authorized and locked behind the atomic oracle gate.
- **PERF-01 remains partially open** (~3.4×, target ≤2.5×). Verifier/human must decide: accept the −33%/4.3×→3.4× improvement, or schedule a scan+score fusion follow-up (the only remaining lever, invasive).
- Pre-existing `monotone_non_symmetric_and_region_are_typed_errors` and the D-08 script false-positives remain out-of-scope backlog (deferred-items.md).

## Self-Check: PASSED
- All modified source files exist on disk.
- All 5 task commits present in git history (cbd0a5b, fedf7c1, 6dd1da8, c60ea9b, 97aba66).
- `cb_core::scatter_add_f64` exported from cb-core lib.

---
*Phase: 21-cpu-split-finding-histogram-rewrite*
*Completed: 2026-07-06*
