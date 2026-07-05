---
phase: 21-cpu-split-finding-histogram-rewrite
plan: 03
subsystem: cb-train
tags: [histogram, split-finding, leaf-wise, depthwise, lossguide, region, subtraction-trick, cpu-training, perf]
requires:
  - cb-compute::BucketHistogram / build_bucket_histogram / scan_borders_to_leaf_stats / bin_of (Plan 21-01)
  - cb-compute::split_score seam (multi_dim_split_score / l2 / cosine — UNCHANGED score math)
  - cb-core::sum_f64 (the sanctioned ordered reduction, folded inside the histogram build/scan)
provides:
  - cb-train::best_split_for_leaf histogram-backed leaf-wise scoring core (per-leaf BucketHistogram + O(n_bins) prefix scan, no per-candidate rescan)
  - Depthwise / Lossguide (leaf_wise_grower) + Region (region_grower) now score every candidate through the histogram scorer via the shared core
affects:
  - crates/cb-train/src/tree.rs
  - crates/cb-train/src/leaf_wise_scorer_test.rs
tech-stack:
  added: []
  patterns:
    - "per-leaf BucketHistogram (n_leaves=1) built ONCE over the leaf's doc subset + O(n_bins) prefix scan replaces the per-candidate assign/reduce_leaf_stats full-subset rescan"
    - "scan splits the single parent leaf into [false, true] children (leaf 0 = value<=border, leaf 1 = value>border) — byte-for-byte the reduce_leaf_stats(leaf_of=passes,..,2) leaf order"
    - "leaf-wise / region growers stay per-leaf-independent (each child fresh-builds its own histogram, O(docs)/leaf) — the fresh-build-both branch; parent->child histogram threading (subtraction trick reuse) is 21-05 scope"
key-files:
  created:
    - crates/cb-train/src/leaf_wise_scorer_test.rs
  modified:
    - crates/cb-train/src/tree.rs
decisions:
  - "best_split_for_leaf builds the per-leaf histogram with approx_dim=1 (the leaf-wise/region growers score single-dimension der1 indexed directly by object — the existing scalar contract at boosting.rs:3778,3800); the histogram carries one delta channel + one weight channel so the dim=1 scan output is byte-identical to the old reduce_leaf_stats(..,2) stats"
  - "unsplit_leaf_score (the gain baseline) left UNCHANGED — still one reduce_leaf_stats over the doc subset; only the per-candidate scoring loop was converted"
  - "partition of the chosen split recomputes passes_float(cand) instead of caching the per-candidate passes Vec (cheaper; same bits)"
  - "leaf_wise_grower (Depthwise level-order / Lossguide priority queue) and region_grower frontier logic are byte-for-byte unchanged — the diff is confined to best_split_for_leaf's body + its doc comment + the new sibling test mount"
metrics:
  duration_min: 35
  completed: 2026-07-05
  tasks: 2
  files_modified: 1
  new_tests: 3
  commits: 1
status: complete
---

# Phase 21 Plan 03: Histogram-Backed Leaf-Wise CPU Split Search Summary

Converted the shared leaf-wise scoring core — `best_split_for_leaf` in
`crates/cb-train/src/tree.rs`, used by `leaf_wise_grower` for BOTH `Depthwise` and
`Lossguide` and reused by `region_grower` — to score every candidate split from a
per-leaf `BucketHistogram` + `O(n_bins)` prefix scan (`scan_borders_to_leaf_stats`)
fed to the UNCHANGED score calcer, replacing the per-candidate `assign` +
`reduce_leaf_stats` full-subset rescan (PERF-02). Because `best_split_for_leaf` is
the single shared scoring core, converting it once covers Depthwise, Lossguide, AND
Region. The candidate enumeration order, the strict `>` first-wins tie-break, the
`gain < 1e-9` cutoff, the `min_data_in_leaf` / `docs.len() >= 2` gates, the
`unsplit_leaf_score` baseline, and the left(false)/right(true) partition are all
byte-for-byte unchanged; the non-symmetric and region oracle suites stay bit-exact.

## What Was Built (commit `3723b3a`)

**Rewritten `best_split_for_leaf` (the leaf-wise / region core):**
- Builds ONE `BucketHistogram` over THIS leaf's document subset — a SINGLE parent
  leaf (`n_leaves = 1`) — via `build_bucket_histogram`. Local object index maps to
  the global `docs[local]`; the subset's `der1` / `weight` are gathered in ascending
  local-object (== ascending doc) order so the reduction stays canonical D-05 order.
  Feature-major bins `bins[feature*n_docs+local]` are quantized with `bin_of` (the
  upper-bound binning consistent with `passes_float`'s strict `>`, Pitfall 4).
  `approx_dim = 1` (the leaf-wise/region growers score single-dimension der1 — the
  existing scalar contract), `n_bins = max(n_borders)+1` across features (mirrors
  `GrowScratch::new`; a feature with fewer borders leaves its upper bins empty).
- For each candidate `(feature ascending, border ascending)` reads the split's
  `[false_stats, true_stats]` from the `O(n_bins)` prefix scan of that per-leaf
  histogram (`scan_borders_to_leaf_stats(&hist, feature, n_borders, 1)`): leaf index
  0 = FALSE child (`bins <= border`), leaf index 1 = TRUE child (`bins > border`) —
  byte-for-byte the leaf order the old `reduce_leaf_stats(leaf_of = passes as 0/1,
  .., 2)` produced. The `dim == 1` scan slice `per_dim[0]` is the exact
  `&[LeafStats]` the pre-rewrite `split_score` consumed.
- Keeps the strict `>` first-wins running max, the `unsplit_leaf_score` baseline, the
  `gain < 1e-9` reject, the `docs.len() < min_data_in_leaf || < 2` guard, and the
  left(false)/right(true) doc partition (recomputed via `passes_float` on the chosen
  split) unchanged.

**Subtraction trick disposition (must_have truth #4).** The two-children
subtraction trick (`BucketHistogram::remove`) applies where a parent histogram is
threaded to its children. The leaf-wise / region grower architecture is
per-leaf-independent — each leaf (and each child) calls `best_split_for_leaf`, which
fresh-builds its own histogram over its doc subset (still `O(docs)` per leaf, the
target complexity). This is the plan's explicit "otherwise fresh-build both" branch.
Parent→child histogram threading (which would let the larger child derive as
`parent.remove(smaller)` and collapse the per-leaf rebuild) requires the scratch
reuse / histogram plumbing that is 21-05 scope, and changing the grower to thread
histograms would alter the expansion strategy this plan prohibits. So the
subtraction primitive remains available and exercised on the oblivious path (21-02
`GrowScratch::advance`) but is intentionally NOT threaded through the leaf-wise
grower here.

**New sibling test file `leaf_wise_scorer_test.rs`** (`#[path]`-mounted as
`tree::leaf_wise_scorer`, the sanctioned source/test-separation pattern — NOT a
`mod tests` block): transcribes the PRE-REWRITE rescan algorithm verbatim as a
reference and asserts the rewritten histogram core reproduces it BIT-FOR-BIT
(chosen split, `==`-exact gain, left/right partition) on benign integer-valued
Cosine (doc subset) and L2 (full set) fixtures, plus degenerate-leaf `None` parity.

## Verification

### Task 1 — leaf-wise / region parity gate: PASS, bit-exact

- `cargo test -p cb-train --test non_symmetric_grower_oracle_test` →
  `non_symmetric_depthwise_grower_splits_match_upstream` PASS (Depthwise splits
  match upstream; Lossguide shares the same core).
- `cargo test -p cb-train --test region_e2e_test` → 2/2 PASS
  (`region_grow_policy_trains_and_applies_to_the_frozen_reference`,
  `..._training_is_deterministic`).
- `cargo test -p cb-train --lib tree::` → 28/28 PASS, including the 3 new
  `tree::leaf_wise_scorer::*` equivalence tests, the 4 `tree::region_grow_test::*`
  region-structure tests, and the tie-break / ordered / pairwise unit tests.
- `grep -rnE 'use +cb_backend|cb_backend::' crates/cb-train/src/tree.rs` →
  **NO-NEW-BACKEND-SEAM** (no code-level reach into cb-backend kernels; the Phase-8
  Cargo.toml feature-passthrough dep is untouched).

### Task 2 — full-suite PERF-02 gate for the leaf-wise + region wave: PASS

- `cargo test -p cb-compute` → **208/208 green** (193 + 9 + 5 + 1, 0 failed) — the
  histogram primitives + score/leaf paths regress nothing.
- `cargo test -p cb-train` → every integration binary + the 240-test lib suite
  green, with a SINGLE exception: the pre-existing, documented, out-of-scope
  `monotone_oracle_test::monotone_non_symmetric_and_region_are_typed_errors` (see
  Deviations). No fixture flipped: the only failure is a boolean policy-rejection
  assertion unrelated to histogram scoring, already failing on the pre-21-02
  baseline.
- clippy: `crates/cb-train/src/tree.rs` and `leaf_wise_scorer_test.rs` are
  **clippy-clean (zero warnings attributable to the new code)**. The whole-crate
  `cargo clippy -p cb-train --all-targets -- -D warnings` gate is blocked ONLY by
  PRE-EXISTING dependency-crate warnings (`cb-data/src/text/bigram_dictionary.rs`
  `doc_lazy_continuation`; `cb-oracle/src/compare.rs` `neg_cmp_op_on_partial_ord`
  and `cb-oracle/src/model_json.rs` `indexing_slicing`) — the same pre-existing
  crate-wide lint debt the 21-01 SUMMARY logged, none of it in this plan's files.

### Policies converted (PERF-02 coverage)

| Grow policy   | Grower            | Scored via                        | Oracle          |
|---------------|-------------------|-----------------------------------|-----------------|
| SymmetricTree | select_level_*    | histogram (Plan 21-02)            | full suite ✓    |
| Depthwise     | leaf_wise_grower  | histogram (best_split_for_leaf)   | non_symmetric ✓ |
| Lossguide     | leaf_wise_grower  | histogram (best_split_for_leaf)   | non_symmetric ✓ |
| Region        | region_grower     | histogram (best_split_for_leaf)   | region_e2e ✓    |

All CPU grow policies now score every candidate through the histogram scorer, with
every ≤1e-5 non-symmetric / region oracle fixture bit-exact.

## Deviations from Plan

### 1. [Out of scope — pre-existing, logged] Stale monotone Region-rejection test

- **Found during:** Task 2 full-suite run.
- **Issue:** `monotone_oracle_test::monotone_non_symmetric_and_region_are_typed_errors`
  asserts `grow_policy=Region` is rejected with a typed error ("Region OUT",
  D-6.6-04). This assertion is STALE — `validate_grow_policy` (`boosting.rs:1369`)
  documents that the Region-OUT rejection was LIFTED by GPUT-18/D-03a; Region now
  grows on CPU via `region_grower` (proven by the passing `region_e2e_test`). The
  test therefore fails because `train(..)` returns `Ok` where it expects `Err`.
- **Why not this plan's regression:** it is a boolean policy-rejection assertion, not
  a numeric fixture, and has NOTHING to do with histogram scoring; it was already
  failing on the pre-21-02 baseline (verified + logged by Plan 21-02) and is the
  SAME single non-green test. My change is bit-exact-preserving (proven by the
  Cosine/L2 rescan-equivalence unit tests + the green non_symmetric/region oracles).
- **Disposition:** NOT fixed (scope boundary — Region test maintenance, not the
  split-finding rewrite). Already recorded in the phase `deferred-items.md`.

### 2. [Deliberate interpretation] Subtraction-trick threading deferred (fresh-build both)

- The two-children subtraction trick is honored via the plan's explicit
  "otherwise fresh-build both" branch: each per-leaf histogram is fresh-built
  (`O(docs)`/leaf, the target complexity). Threading a parent histogram to its
  children (so the larger child derives via `BucketHistogram::remove`) would require
  the scratch-reuse plumbing scoped to 21-05 and would change the leaf-wise
  expansion structure this plan prohibits. See "Subtraction trick disposition"
  above. Consistent with the project note that full PERF-01 flatness / scratch reuse
  is a 21-05 deliverable.

## Known Stubs

None — the histogram scoring path is fully wired for the leaf-wise / region core.
Every candidate's `[false, true]` `LeafStats` come from the real per-leaf histogram
prefix scan; no placeholder / empty-data flow.

## Self-Check: PASSED

- `crates/cb-train/src/tree.rs` present with `best_split_for_leaf` building a
  per-leaf `build_bucket_histogram` + `scan_borders_to_leaf_stats` (verified).
- `crates/cb-train/src/leaf_wise_scorer_test.rs` present and mounted as
  `tree::leaf_wise_scorer` (3 tests pass).
- Commit `3723b3a` present in git log.
