---
phase: 21-cpu-split-finding-histogram-rewrite
plan: 04
subsystem: cb-train
tags: [histogram, split-finding, ctr, online-ctr, subtraction-trick, cpu-training, perf]
requires:
  - cb-compute::BucketHistogram / build_bucket_histogram / scan_border_to_leaf_stats / bin_of (Plan 21-01)
  - cb-compute::split_score seam (l2_split_score / cosine_split_score — UNCHANGED score math)
  - cb-core::sum_f64 (the sanctioned ordered reduction, folded inside the histogram build/scan)
  - cb-train::assign_leaves_ctr_aware / CtrAwareSplit / cat_feature_weight (unchanged enumeration + penalty)
provides:
  - cb-train::build_ctr_aware_histogram (per-level combined float+CTR BucketHistogram over the current partition)
  - histogram-backed score_candidate_ctr_aware / select_level_ctr_aware (no per-candidate assign+reduce rescan)
affects:
  - crates/cb-train/src/tree.rs
tech-stack:
  added: []
  patterns:
    - "CTR bin columns fold into the SAME 2-channel histogram as float features: CtrFeatureColumn.bins are the histogram bins DIRECTLY (bin == ctr bin value), so the `ctr_bin > border` prefix-scan boundary is identical to a float `bins > border`"
    - "one per-level combined histogram over BOTH float columns (bin_of over feature_borders) AND CTR bin columns (indices n_float..n_float+n_ctr) replaces the per-candidate assign_leaves_ctr_aware + reduce_leaf_stats full-dataset rescan"
    - "global n_bins = max(float n_borders+1, ctr_border_count+1, actual max ctr bin+1) — extra empty upper bins sum to 0.0 and are inert"
key-files:
  created: []
  modified:
    - crates/cb-train/src/tree.rs
decisions:
  - "score_candidate_ctr_aware REWRITTEN (not deleted) into a histogram-backed per-candidate scorer taking (hist, hist_feature, border_index): one O(n_bins) scan_border_to_leaf_stats fed to the UNCHANGED split_score — keeps the named symbol the plan artifacts reference"
  - "approx_dim == 1 for the CTR histogram (the CTR path scores single-dimension der1 via reduce_leaf_stats + split_score, the existing scalar contract); the histogram carries one delta channel + one shared weight channel"
  - "histogram built FRESH once per level over assign_leaves_ctr_aware(chosen) — the O(n) binning/reduction is paid once per level (not per candidate); subtraction-trick threading across the CTR level transition is deferred (the 21-02/21-03 'fresh-build' branch), consistent with 21-05 scratch-reuse scope"
  - "assign_leaves_ctr_aware KEPT unchanged — reused for the histogram partition AND the once-per-tree final leaf assignment / leaf-VALUE path"
metrics:
  duration_min: 30
  completed: 2026-07-05
  tasks: 2
  files_modified: 1
  new_tests: 0
  commits: 1
status: complete
---

# Phase 21 Plan 04: Histogram-Backed CTR-Aware CPU Split Search Summary

Converted the LAST in-scope CPU scoring path — the online-CTR-feature scoring path
(`greedy_tensor_search_oblivious_with_ctr` via `select_level_ctr_aware` /
`score_candidate_ctr_aware` in `crates/cb-train/src/tree.rs`) — to score every
candidate (float AND CTR) from ONE per-level `BucketHistogram` + `O(n_bins)` prefix
scan (`scan_border_to_leaf_stats`) fed to the UNCHANGED `split_score`, replacing the
per-candidate `assign_leaves_ctr_aware` + `reduce_leaf_stats` full-dataset rescan
(PERF-02). CTR candidates fold into the SAME 2-channel histogram as float features:
a `Ctr{col,border}` split tests `ctr_bin > border` over an integer bin column, which
is structurally the same prefix-scan boundary as a float `bins > border`, so the CTR
columns are simply additional histogram feature rows (RESEARCH §Coverage). The
float-then-CTR enumeration order, the `cat_feature_weight` multiplier insertion
point, and the strict `>` first-wins tie-break are byte-for-byte unchanged; every
shipped ≤1e-5 CTR oracle fixture stays bit-exact.

## What Was Built (commit `a268e82`)

**`build_ctr_aware_histogram` (new):** builds ONE per-level combined histogram over
the current partition (`assign_leaves_ctr_aware(chosen)`, forward-bit leaf order,
`n_leaves = 1 << level`) covering the UNION of:
- the FLOAT feature columns (indices `0..n_float`), quantized with `bin_of` over
  `feature_borders` (the upper-bound binning consistent with `passes_float`'s strict
  `>`, Pitfall 4);
- the materialized CTR bin columns (indices `n_float + col`), whose integer
  `CtrFeatureColumn.bins` are fed DIRECTLY as their histogram bin (bin == ctr bin
  value), so `scan_border_to_leaf_stats` at border index `k` splits exactly
  `ctr_bin <= k` (FALSE) / `ctr_bin > k` (TRUE) — mirroring `passes_ctr_aware`'s CTR
  branch (`ctr_bin > border`).

Global `n_bins = max(float n_borders+1, ctr_border_count+1, actual-max-ctr-bin+1,
1)`. The `ctr_border_count + 1` term guarantees the scan's highest CTR border index
(`ctr_border_count - 1`) has an in-range `.get(0..=border)` FALSE-side slice even
when no object reaches that bin (a degenerate all-FALSE split); the actual-max term
guards CTR bins that reach `ctr_border_count`. Extra empty upper bins contribute
`0.0` and are inert. `approx_dim == 1` (the CTR path's scalar `reduce_leaf_stats` +
`split_score` contract).

**`score_candidate_ctr_aware` (rewritten):** now takes `(hist, hist_feature,
border_index, scaled_l2, score_function)` and reads the split's `[false, true]`
child `LeafStats` from ONE `O(n_bins)` `scan_border_to_leaf_stats(hist, hist_feature,
border_index, 1)` (dim 0), fed to the UNCHANGED `split_score`. The canonical
`[false, true]` leaf order (`bins <= border` FALSE at index `parent`, `bins > border`
TRUE at index `parent + n_parent`) is byte-for-byte the leaves
`reduce_leaf_stats(assign_leaves_ctr_aware(chosen ++ candidate))` produced.

**`select_level_ctr_aware` (rewritten body):** builds the combined histogram ONCE,
then scores candidates in the EXISTING fixed order — FLOAT (feature asc × border
asc, scanning histogram feature `feature`) THEN CTR (column asc × `border_idx` asc,
scanning histogram feature `n_float + col`). The `cat_feature_weight` machinery
(`max_bucket_count`, `used_projections` exemption, `cat_weight * raw` insertion
point) and the strict `>` first-wins over the FIXED float-then-CTR order are
untouched.

**Kept unchanged:** `passes_ctr_aware`, `assign_leaves_ctr_aware` (reused for the
histogram partition AND the once-per-tree final leaf assignment / leaf-VALUE path),
`cat_feature_weight`, `CtrAwareSplit`, and `greedy_tensor_search_oblivious_with_ctr`'s
signature + split-recovery/`CtrSplitSpec` logic. No `cb-backend` seam introduced.

## Verification

### Task 1 — CTR parity gate: PASS, bit-exact

- `cargo test -p cb-train --test plain_ctr_oracle_test --test tensor_ctr_oracle_test
  --test ctr_split_scoring_test` → **16/16 PASS** (3 + 3 + 10), including
  `ctr_candidate_wins_over_uninformative_float`, `tie_break_float_then_ctr_first_wins`,
  `forward_bit_leaf_index_mixed_float_and_ctr`, and
  `single_feature_ctr_structure_partition_6_0_9_15` — the CTR chosen-split /
  tie-break / partition equalities.
- Additional CTR fixtures bit-exact: `s_order_ctr_bins_oracle_test` (2/2, incl. the
  pc=4 averaging CTR bins), `tensor_ctr_e2e_oracle_test` (3/3, incl.
  `oracle_predictions_match_upstream`), `ordered_ctr_oracle_test` (3/3 — the ordered
  scorer itself is unchanged, so the ordered-CTR structure stays green).
- `grep -rnE 'use +cb_backend|cb_backend::' crates/cb-train/src/tree.rs` →
  **NO-NEW-BACKEND-SEAM** (no code-level reach into cb-backend kernels; the Phase-8
  Cargo.toml feature-passthrough dep is untouched).

### Task 2 — full-suite PERF-02 gate for the CTR wave: PASS

- `cargo test -p cb-train --no-fail-fast` → every integration binary + the 240-test
  lib suite green, with a SINGLE exception: the pre-existing, documented,
  out-of-scope `monotone_oracle_test::monotone_non_symmetric_and_region_are_typed_errors`
  (see Deviations). No CTR (or any other) fixture flipped.
- `cargo test -p cb-compute --no-fail-fast` → **208/208 green** (193 + 5 + 1 + 9 + 0,
  0 failed) — the histogram primitives regress nothing.
- clippy: `crates/cb-train/src/tree.rs` is **clippy-clean** (zero warnings
  attributable to the new `build_ctr_aware_histogram` / rewritten
  `score_candidate_ctr_aware` / `select_level_ctr_aware`). The whole-crate
  `cargo clippy -p cb-train --all-targets -- -D warnings` gate is blocked ONLY by the
  SAME PRE-EXISTING dependency-crate lint debt the 21-01/21-03 SUMMARYs logged
  (`cb-oracle` `neg_cmp_op_on_partial_ord` + `indexing_slicing`; `cb-data`
  `doc_lazy_continuation`) — none of it in this plan's files.

### PERF-02 coverage — all in-scope CPU scoring paths now histogram-backed

| Grow / scoring path            | Scored via                             | Landed  |
|--------------------------------|----------------------------------------|---------|
| Oblivious SymmetricTree        | histogram (select_level_plain/perturbed) | 21-02   |
| Depthwise / Lossguide          | histogram (best_split_for_leaf)         | 21-03   |
| Region                         | histogram (best_split_for_leaf)         | 21-03   |
| **Online-CTR-feature scoring** | **histogram (select_level_ctr_aware)**  | **21-04** |

Every in-scope CPU scoring path (oblivious, Depthwise, Lossguide, Region, CTR) now
scores every candidate through the histogram scorer, with every ≤1e-5 oracle fixture
bit-exact. (Ordered-boosting and pairwise remain on their dedicated paths by design —
RESEARCH Open Questions 1 & 3; explicitly out of Phase-21 scope.)

## Deviations from Plan

### 1. [Out of scope — pre-existing, logged] Stale monotone Region-rejection test

- **Found during:** Task 2 full-suite run.
- **Issue:** `monotone_oracle_test::monotone_non_symmetric_and_region_are_typed_errors`
  asserts `grow_policy=Region` is rejected with a typed error ("Region OUT",
  D-6.6-04). This assertion is STALE — `validate_grow_policy` (`boosting.rs:1369`)
  documents that the Region-OUT rejection was LIFTED by GPUT-18/D-03a; Region now
  grows on CPU. It fails because `train(..)` returns `Ok` where it expects `Err`.
- **Why not this plan's regression:** it is a boolean policy-rejection assertion with
  NOTHING to do with CTR (or histogram) scoring; it was already failing on the
  pre-21-02 baseline (verified + logged by BOTH Plan 21-02 and 21-03) and is the SAME
  single non-green test. My change is confined to the CTR-aware level search and is
  bit-exact-preserving (proven by the full green CTR oracle suite).
- **Disposition:** NOT fixed (scope boundary — Region test maintenance, not the
  split-finding rewrite). Already recorded in the phase `deferred-items.md`.

### 2. [Deliberate interpretation] Subtraction-trick threading deferred (fresh-build per level)

- The CTR-aware histogram is fresh-built ONCE per level over the current partition
  (`O(n)`/level — the target complexity, since the `O(n)` is per-level not
  per-candidate). Threading a parent histogram to its CTR-aware children (so the
  larger child derives via `BucketHistogram::remove`) requires the scratch-reuse
  plumbing scoped to 21-05 and would complicate the CTR level-transition; the
  subtraction primitive remains exercised on the oblivious path
  (`GrowScratch::advance`, 21-02). This is the same "fresh-build" branch 21-03 took
  for the leaf-wise/region core, consistent with the project note that full PERF-01
  flatness / scratch reuse is a 21-05 deliverable. The plan's PERF-02 must_have (score
  from a per-level histogram, NOT a per-candidate rescan) is fully met.

## Known Stubs

None — the CTR-aware histogram scoring path is fully wired. Every candidate's
`[false, true]` `LeafStats` come from the real per-level combined float+CTR histogram
prefix scan; no placeholder / empty-data flow.

## Threat Flags

None — no new network endpoints, auth paths, file access, or trust-boundary schema
changes. The change is an in-process numerical refactor of the CTR-aware split
scorer; the threat register's T-21-10/11/12 (bin off-by-one, tie-flip, histogram
width) are the mitigations verified by the bit-exact CTR oracle suite above.

## Self-Check: PASSED

- `crates/cb-train/src/tree.rs` present with `build_ctr_aware_histogram` +
  histogram-backed `score_candidate_ctr_aware` (`scan_border_to_leaf_stats`) +
  rewritten `select_level_ctr_aware` (verified).
- Commit `a268e82` present in git log.
