---
phase: 21-cpu-split-finding-histogram-rewrite
plan: 01
subsystem: cb-compute
tags: [histogram, split-finding, parity, cpu-training, bucket-stats]
requires:
  - cb_core::sum_f64 (the single sanctioned ordered reduction, D-05/D-08)
  - cb-compute::LeafStats / reduce_leaf_stats (the frozen leaf-value/score reduction)
  - cb-compute::score (l2_split_score / multi_dim_split_score — UNCHANGED)
provides:
  - cb-compute::BucketHistogram (per-(leaf,feature,bin) 2+-channel TBucketStats analogue)
  - cb-compute::build_bucket_histogram (one-object-order-pass ordered build)
  - cb-compute::BucketHistogram::remove (subtraction trick / TBucketStats::Remove)
  - cb-compute::bin_of (upper-bound binning consistent with passes_float strict >)
  - cb-compute::scan_border_to_leaf_stats / scan_borders_to_leaf_stats (O(n_bins) prefix scan → LeafStats)
affects:
  - crates/cb-compute/src/histogram.rs
  - crates/cb-compute/src/histogram_test.rs
  - crates/cb-compute/src/lib.rs
tech-stack:
  added: []
  patterns:
    - "gather-then-fold-via-sum_f64 generalized from leaf to (leaf,feature,bin)"
    - "frozen flat layout mirrors device pointwise_hist.rs (transcribed, not imported)"
    - "prefix scan combines buckets via sum_f64 in ascending bin order (upstream CalcScoresForLeaf)"
key-files:
  created: []
  modified:
    - crates/cb-compute/src/histogram.rs
    - crates/cb-compute/src/histogram_test.rs
    - crates/cb-compute/src/lib.rs
decisions:
  - "scan returns per-DIMENSION canonical-leaf-order LeafStats (Vec<Vec<LeafStats>>) directly consumable by multi_dim_split_score; scan_borders_to_leaf_stats wraps it per border ([border][dim][leaf])"
  - "bit-exact equivalence proven on BENIGN exactly-representable fixtures; the adversarial-ULP tie-flip risk (Pitfall 1) is delegated to the downstream oracle suite (wired in 21-02..05)"
  - "subtraction trick implemented as plain f64 per-cell -= (matches upstream Remove rounding, parity-faithful)"
metrics:
  duration_min: 16
  completed: 2026-07-05
  tasks: 2
  files_modified: 3
  new_tests: 5
  commits: 2
status: complete
---

# Phase 21 Plan 01: CPU Split-Finding Histogram Primitives Summary

Landed the parity-critical CPU histogram data-production layer in `cb-compute` —
a per-`(leaf,feature,bin)` 2+-channel `TBucketStats` histogram, its subtraction
trick, upper-bound binning, and the `O(n_bins)` prefix scan to `LeafStats` —
behind bit-exact equivalence unit tests against the current
`reduce_leaf_stats`/`score_candidate` path, on scalar AND multiclass fixtures,
with the score math and leaf-value path untouched.

## What Was Built

**Task 1 (commit `f1519f1`) — build + subtraction primitives:**
- `BucketHistogram`: flat `Vec<f64>` in the frozen
  `((leaf*n_features+feature)*n_bins+bin)*n_channels+channel` layout (mirrors the
  device `pointwise_hist.rs:44-49`, transcribed — no `cb-backend` dependency).
  `n_channels = approx_dimension + 1` (per-dim `Σ der1` channels + one shared
  `Σ weight` channel). Private fields with `.channel()`/shape accessors; no raw
  indexing.
- `build_bucket_histogram`: ONE object-order pass that gathers each cell's
  contributions in ascending object order then folds each through
  `cb_core::sum_f64` — exactly the `reduce_leaf_stats` gather-then-fold shape
  generalized from `leaf` to `(leaf,feature,bin)`.
- `BucketHistogram::remove`: the subtraction trick (`TBucketStats::Remove`), plain
  per-cell f64 `-=`, shape-guarded (defensive clone on mismatch, no panic).
- `bin_of`: count of borders strictly less than the value (upper-bound), so a
  split at border `b` puts false=`bins<=b` / true=`bins>b`, consistent with
  `FeatureMatrix::passes_float`'s strict `>` (guards Pitfall 4).

**Task 2 (commit `3dc5d97`) — prefix scan + equivalence proof:**
- `scan_border_to_leaf_stats`: `O(n_bins)` scan splitting every parent leaf into
  its false/true children in canonical forward-bit leaf order
  (`leaf = parent + (candidate ? n_parent : 0)`), per dimension, ready for the
  UNCHANGED `split_score`/`multi_dim_split_score`. Buckets combined via `sum_f64`
  in ascending bin order (matches upstream `CalcScoresForLeaf`).
- `scan_borders_to_leaf_stats`: convenience wrapper over all borders
  (`[border][dim][leaf]`).

## Tests (5 new, all bit-exact `==`)

1. `bin_of_matches_strict_greater_split` — boundary/equal/below-min/above-max bins;
   asserts `passes ⇔ k < bin` against `value > border` for every border.
2. `build_bucket_histogram_sums_match_reduce_leaf_stats` — per-leaf sum across all
   bins of a feature == `reduce_leaf_stats` (Σder1 and Σweight), per feature.
3. `bucket_histogram_remove_equals_fresh_sibling` — `parent.remove(childA)` ==
   freshly-built sibling B histogram, cell-for-cell (`assert_eq!` on the struct).
4. `scan_border_matches_rescan_scalar` — for each candidate border, scan-derived
   `Vec<LeafStats>` and `l2_split_score` == `assign_leaves(chosen++cand) +
   reduce_leaf_stats` rescan (with a chosen depth-1 parent split → 4 leaves).
5. `scan_border_matches_rescan_multiclass` — `approx_dimension=2`, per-dimension
   `LeafStats` == per-dim rescan and cross-dimension Cosine
   `multi_dim_split_score` bit-exact.

## Verification

- `cargo test -p cb-compute histogram` → 11 passed (6 pre-existing + 5 new).
- `cargo test -p cb-compute` (full crate, lib + integration) → 193 + 9 + 5 + 1 all
  green, 0 failed — no regression to the leaf-value / score / other paths.
- `crates/cb-compute/src/histogram.rs` is clippy-clean (0 warnings attributable to
  the new code; the one `too_many_arguments` allow is documented on
  `build_bucket_histogram` because its 8 params mirror the frozen device contract).
- Acceptance greps: no `use cb_backend` / `cb-backend` seam and no `rayon`/`cubecl`
  in `crates/cb-compute/Cargo.toml` `[dependencies]` (only prose in comments).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Freed the full output-capture tmpfs to run tests**
- **Found during:** Task 1 verification.
- **Issue:** the root disk (`/dev/nvme0n1p7`, the known disk-pressure filesystem)
  was at 100%, so the harness's stdout-capture directory hit `ENOSPC` and every
  `cargo`/`df` invocation failed before producing output.
- **Fix:** removed stale large files under `/tmp/claude-1000/**` (freed ~4.6G);
  all subsequent builds/tests/clippy ran normally. No project files touched.
- **Files modified:** none (environment only).

### Deliberate interpretation (not a functional deviation)

- The plan's `artifacts_this_phase_produces` sketches
  `scan_borders_to_leaf_stats(...) -> Vec<Vec<LeafStats>>`. To match the score
  consumer cleanly, the primary primitive is `scan_border_to_leaf_stats(border) ->
  Vec<Vec<LeafStats>>` (`[dim][leaf]`, directly fed to `multi_dim_split_score`),
  and `scan_borders_to_leaf_stats(n_borders) -> Vec<Vec<Vec<LeafStats>>>`
  (`[border][dim][leaf]`) is the all-borders wrapper. Both are exported. The
  checked `must_haves.artifacts` (histogram.rs provides build + prefix-scan +
  Remove, contains `sum_f64`) are satisfied.

## Out-of-Scope Discoveries (deferred, NOT fixed)

- **Pre-existing crate-wide clippy warnings** (`doc_lazy_continuation`,
  `neg_cmp_op_on_partial_ord`, `manual_assign_ops`, `excessive_precision`, …) in
  `cb-data` and other `cb-compute` files (`embedding_calcers.rs`, `lda_linalg.rs`,
  `leaf.rs`, `loss.rs`, `ranking_der.rs`, `runtime.rs`) surface under a newer
  clippy than these files were written against. They are unrelated to this task
  and block a whole-crate `-D warnings` run; the NEW histogram code is clean.
  Logged for a future lint-sweep; not touched (scope boundary).
- **`scripts/check-no-raw-float-sum.sh` already fails on the current tree** (it
  flags usize `.sum()` and `.sum()`-in-comments across many pre-existing files).
  The new production code routes every FLOAT sum through `cb_core::sum_f64` and
  adds no float `.sum()`; the script's pre-existing false positives are unchanged
  and out of scope.

## TDD Gate Compliance

Both tasks are `tdd="true"`, but plan `type` is `execute` (not `type: tdd`) and
`tdd_mode` is off in config, so per-task RED→GREEN commit gating is not enforced.
Implementation and its bit-exact tests were developed and verified together, then
committed as two atomic per-task `feat` commits (Task 1 = build/subtraction/bin_of
+ their tests; Task 2 = prefix scan + equivalence tests). Each commit compiles and
its tests pass in isolation (Task-1 commit verified with 9 histogram tests before
Task-2 code was restored).

## Notes for Downstream Plans (21-02..05)

- `scan_border_to_leaf_stats` is O(n_bins) per border but recomputes prefix/suffix
  per border (O(n_bins²) per leaf); this is fine for the primitive/tests. The
  scratch-buffer reuse + running-prefix optimization + rayon parallelism are
  explicitly deferred to the wiring/perf waves (PERF-03, plan 21-05).
- The parity crux is proven at the unit level on benign fixtures; the ULP tie-flip
  (Pitfall 1) and subtraction-rounding (Pitfall 2) risks are gated by the full CPU
  oracle suite once the grow loop is wired (21-02+).

## Self-Check: PASSED

All modified files and both task commits (f1519f1, 3dc5d97) verified present.
