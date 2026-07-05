---
phase: 21-cpu-split-finding-histogram-rewrite
plan: 02
subsystem: cb-train
tags: [histogram, split-finding, oblivious, grow-scratch, subtraction-trick, cpu-training, perf]
requires:
  - cb-compute::BucketHistogram / build_bucket_histogram / scan_borders_to_leaf_stats / bin_of / BucketHistogram::remove (Plan 21-01)
  - cb-compute::multi_dim_split_score (UNCHANGED score math)
  - cb-core::sum_f64 (the sanctioned ordered reduction, folded inside the histogram build/scan)
provides:
  - cb-train::GrowScratch (reusable per-level bin matrix + incremental leaf_of + per-level histogram — the TLearnContext analogue)
  - histogram-backed select_level_plain / select_level_perturbed (no per-candidate rescan)
affects:
  - crates/cb-train/src/tree.rs
tech-stack:
  added: []
  patterns:
    - "per-level BucketHistogram + O(n_bins) prefix scan replaces per-candidate assign_leaves+reduce_leaf_stats rescan"
    - "subtraction trick (BucketHistogram::remove) derives the level transition (smaller sibling built, larger = parent - smaller)"
    - "GrowScratch reused across levels (bins built once, leaf_of incremental, histogram fixed-size)"
key-files:
  created: []
  modified:
    - crates/cb-train/src/tree.rs
decisions:
  - "score_candidate / multi_dim_candidate_score RETAINED (score_candidate marked #[allow(dead_code)]) as the reference per-candidate scorer for the WR-01 unit test + ordered/CTR/one-hot doc analogs; the oblivious plain/perturbed path no longer calls them"
  - "Bins matrix uses a GLOBAL n_bins = max(n_borders)+1 across features (features with fewer borders leave upper bins empty; the per-feature scan walks only its own borders)"
  - "Subtraction advance derives via BucketHistogram::remove; sibling halves reunited with an add composed from remove (a + b = a - (0 - b)) since BucketHistogram exposes only remove — O(cells), n-independent"
  - "Ordered-boosting path (score_candidate_ordered/select_level_ordered) intentionally left on its rescan (Phase-22 deferral, code comment added)"
metrics:
  duration_min: 95
  completed: 2026-07-05
  tasks: 2
  files_modified: 1
  new_tests: 0
  commits: 1
status: complete
---

# Phase 21 Plan 02: Histogram-Backed Oblivious CPU Split Search Summary

Rewired the DEFAULT and dominant CPU grow path — the oblivious `SymmetricTree`
plain and perturbed level search (`select_level_plain` / `select_level_perturbed`
in `crates/cb-train/src/tree.rs`) — to score every candidate from a per-level
`BucketHistogram` + `O(n_bins)` prefix scan (`scan_borders_to_leaf_stats`) fed to
the UNCHANGED `multi_dim_split_score`, replacing the per-candidate
`assign_leaves` + `reduce_leaf_stats` full-dataset rescan. A reusable `GrowScratch`
pays the `O(n)` binning cost ONCE per level; each level transition derives the next
histogram with the subtraction trick. The full cb-train oracle suite stays
bit-exact (PERF-02), and per-tree time drops ~13–18× vs the old rescan. **PERF-01
full flatness is NOT yet met** — see Deviations.

## What Was Built (commit `3ed3125`)

**`GrowScratch` (the `TLearnContext` analogue):**
- `bins: Vec<u32>` — the quantized per-feature bin matrix (feature-major
  `bins[feature*n_objects+obj]`), built ONCE at construction via `bin_of` (the
  upper-bound binning consistent with `passes_float`'s strict `>`, Pitfall 4).
- `leaf_of: Vec<usize>` — the current-partition leaf index, maintained
  INCREMENTALLY (once per chosen split, forward-bit `leaf' = leaf + (passes?1<<L:0)`
  — byte-identical to `assign_leaves(chosen ++ split)`), not reassigned per
  candidate.
- `hist: BucketHistogram` — the current level's per-`(leaf,feature,bin)` histogram
  (level-0 root histogram built in `new`).
- `advance()` — the level transition: updates `leaf_of` and derives the next
  histogram with the SUBTRACTION TRICK (upstream `FixUpStats`): build the SMALLER
  sibling directly, derive the LARGER = parent − smaller via
  `BucketHistogram::remove`. Both siblings land in their canonical forward-bit slots
  (false child = leaf `p`, true child = `p + n_parent`), so the derived histogram is
  scannable exactly like a fresh build.

**`select_level_plain` (histogram-backed):** enumerates candidates in the EXACT
prior order (float feature ascending × border ascending), scores each border from
`scan_borders_to_leaf_stats(&scratch.hist, feature, n_borders, dim)` →
`multi_dim_split_score` (the same call the old `multi_dim_candidate_score` made,
dim=1 byte-identical), applies the FEAT-04 penalty at the same insertion point, and
keeps `select_best_candidate`'s strict `>` first-wins. No `score_candidate` call.

**`select_level_perturbed` (RNG-order preserving):** only the SOURCE of each
candidate's RAW score changes (histogram scan instead of rescan). Every RNG draw is
byte-for-byte in place (Pitfall 3): per-level `gen_rand`, per-feature
`from_seed(rand_seed+task_idx).advance(10)`, one `std_normal` per border via
`random_score_instance`, per-feature `feature_best` bookkeeping, per-feature main-RNG
`GetInstance`. The per-border loop shape (hence draw count/order) is untouched.

**Ordered-boosting deferral:** a code comment above `score_candidate_ordered`
documents that the ordered path is intentionally left on its per-segment rescan
this phase (Phase-22 scope, RESEARCH Open Question 1 / A2); its parity is trivially
preserved because it is untouched.

## Verification

### PERF-02 — parity (THE gate): PASS, bit-exact

Full `cargo test -p cb-train` oracle suite run in batches (disk-pressure workaround,
see below) — **all 56 integration test binaries + 25 tree lib unit tests pass
bit-exact**, including:

- Task-1 gate: `loss_oracle_test`, `overfit_oracle_test`, `regularization_oracle_test`
  (incl. `regularization_oracle_random_strength_first_tree` — perturbed path).
- Task-2 gate: `penalty_oracle_test` (perturbed RNG), `multiclass_oracle_test`,
  `multilabel_oracle_test` (multi-dim histogram path).
- Untouched paths stay green: CTR (`plain_ctr`, `tensor_ctr`, `ordered_ctr`,
  `ctr_split_scoring`, `s_order_ctr_bins`, `tensor_ctr_e2e`, `ctr_feature_materialize`),
  ordered (`ordered_boost`, `ordered_boost_e2e`, `ordered_boost_wiring`), non-symmetric
  (`non_symmetric_grower`), pairwise/ranking (`pairlogit*`, `yetirank*`, `lambdamart`,
  `queryrmse`, `querysoftmax`, `ranking_metrics`, `stochasticrank`), region/device
  (`region_e2e`, `device_seam`, `device_nonsym_fit`, `device_region_fit`), and the
  full loss/feature/permutation/bootstrap/wave families.
- `tree::` lib tests (25): tie-break (strict `>`), small-tree structure, the WR-01
  `multi_dim_candidate_score_bad_stride` test, ordered/pairwise/region unit tests.

Acceptance greps: `grep -rnE 'use +cb_backend|cb_backend::' crates/cb-train/src/tree.rs`
→ **NO-NEW-BACKEND-SEAM** (no code-level cb-backend reach; the Cargo.toml
feature-passthrough dep is untouched). `select_level_plain`/`_perturbed` no longer
call `score_candidate`.

### PERF-01 — n_bins sweep: PARTIAL (13–18× faster, but NOT flat)

`CB_PERF=1 cargo test --release -p cb-train --test perf_baseline_test` (n=10000,
nf=20, depth=6, iters=3), per-tree ms, OLD (pre-21-02 rescan) vs NEW (histogram):

| border_count | OLD ms | NEW ms | speedup |
|--------------|--------|--------|---------|
| 16           | 263.6  | 30.7   | 8.6×    |
| 32           | 549.8  | 41.8   | 13.2×   |
| 64           | 1120.7 | 65.9   | 17.0×   |
| 128          | 2275.1 | 126.9  | 17.9×   |
| 254          | 4407.2 | 339.9  | 13.0×   |

The per-candidate full-dataset rescan (the Spike-002 dominant slowdown) is
ELIMINATED — a 13–18× per-tree win. **But the n_bins SLOPE is unchanged**: 32→254 is
~8.0× on OLD and ~8.1× on NEW, so per-tree time still scales ~linearly with
`border_count` — the "flat within noise 32→254" PERF-01 acceptance is NOT met. See
Deviations for the root cause and why full flatness lands in Plan 21-05.

## Deviations from Plan

### 1. [PERF-01 flatness NOT met — deferred to 21-05, cross-plan consistent]

- **Found during:** Task 2 perf sweep.
- **Issue:** the must_have "Per-tree CPU time is flat within noise across
  border_count 32→254" is NOT achieved; per-tree time still scales ~linearly with
  `n_bins` (same slope as the old rescan, ~8× for 8× bins).
- **Root cause:** the residual `n_bins` scaling comes from the histogram DATA layer,
  not the (now-eliminated) rescan:
  1. `scan_borders_to_leaf_stats` (Plan 21-01) is **O(n_bins²) per feature per
     level** — it recomputes the false/true prefix sums fresh for every border
     (each `scan_border` does `sum_f64` over `bins 0..=border` and
     `border+1..n_bins`). The **running-prefix optimization that makes this
     O(n_bins) was EXPLICITLY DEFERRED to Plan 21-05** by the 21-01 SUMMARY
     ("scratch-buffer reuse + running-prefix optimization + rayon parallelism are
     explicitly deferred to the wiring/perf waves, PERF-03, plan 21-05"). At
     border_count=254 this O(n_bins²) scan dominates the per-tree cost.
  2. `build_bucket_histogram` allocates a fresh `Vec<Vec<f64>>` of size
     `n_leaves·n_features·n_bins·n_channels` per build (the per-level allocation is
     **O(n_bins)**); the scratch-buffer REUSE that removes this is also 21-05 scope
     (the 21-01 SUMMARY notes the primitive "allocates fresh for clarity").
- **Why not fixed here:** achieving true O(n_bins) scoring requires either the
  running-prefix scan (21-05) or changing the scan's `sum_f64` fold to a
  `total − prefix` subtraction — the latter changes the exact bits of the `true`
  child sum (Pitfall 2), risks tie-flips against the just-verified bit-exact oracle
  suite, and is precisely the summation-strategy decision the RESEARCH flags for the
  perf wave (Open Question 2). Forcing it into 21-02 would trade the green PERF-02
  gate for a speculative flatness gain. The DOMINANT Spike-002 pathology (the
  per-candidate `O(n)` rescan multiplying `n × n_bins`) IS closed here; the remaining
  `n_bins` term is a smaller constant that 21-05's running-prefix + scratch reuse
  (+ rayon) removes.
- **Disposition:** PERF-01 is PARTIALLY satisfied (algorithm + subtraction trick
  landed, 13–18× win); full flatness is a 21-05 deliverable, consistent with 21-01's
  explicit deferral. Recorded here and in the phase notes for the verifier/planner.

### 2. [Rule 3 - Blocking] Freed disk to run the oracle suite

- **Found during:** Task 1/2 verification.
- **Issue:** the root disk (the known disk-pressure filesystem) hit 100%; statically
  linked test binaries (~500 MB each, polars/cubecl/arrow) exhausted space, so
  `mold` failed with "Disk full?" and the output-capture tmpfs hit ENOSPC.
- **Fix:** cleared `target/debug/incremental` (2.2 G) and deleted stale linked test
  executables in `target/debug/deps` (the `.rlib`s — the expensive artifacts — were
  preserved; only final executables relink). Ran the 56-binary oracle suite in
  batches, freeing large executables between batches. No project files touched.
- **Files modified:** none (environment only).

### 3. [Out of scope — pre-existing, logged] Stale monotone Region test

- `monotone_oracle_test::monotone_non_symmetric_and_region_are_typed_errors` asserts
  `grow_policy=Region` is rejected with a typed error ("Region OUT", D-6.6-04). This
  is STALE: `boosting.rs:1369` (`validate_grow_policy`) documents the Region-OUT
  rejection was LIFTED by GPUT-18/D-03a — Region now grows on CPU. The test FAILS on
  the pre-21-02 baseline too (verified by stashing the 21-02 tree.rs change and
  re-running), so it is unrelated to the oblivious histogram rewrite and out of Phase
  21 scope (Region is not the oblivious path). Logged to `deferred-items.md`; NOT
  fixed (scope boundary). This is the ONLY non-green test in the whole cb-train suite.

## Known Stubs

None — the histogram scoring path is fully wired (no placeholder/empty-data flow;
every candidate's LeafStats come from the real per-level histogram).

## Design Notes for Plan 21-05 (perf wave)

- The single biggest flatness lever is replacing the O(n_bins²)
  `scan_borders_to_leaf_stats` with a **running-prefix scan** (O(n_bins) per feature
  per level): maintain a running false-child accumulator as a `sum_f64`-equivalent
  left fold (exact for the false side); the true side needs a parity-preserving
  suffix strategy (either a right-fold buffer or the sanctioned fixed-point-u64
  `total − prefix`, gated by the oracle suite — RESEARCH Open Question 2).
- The subtraction `advance` currently does 3 `build_bucket_histogram` calls per level
  (build smaller sibling low+high + parent base, derive larger via `remove`) because
  `BucketHistogram` is opaque (private fields; only `remove` is exposed — no cell
  construction, no `add`). Scratch-buffer reuse (21-05) should let this collapse to
  in-place per-cell subtraction, removing the per-level O(n_bins) allocation.
- Then rayon over independent features (Pitfall 5: per-feature-independent, ordered
  `collect`) for the core-count factor.

## Self-Check: PASSED

- `crates/cb-train/src/tree.rs` present with `GrowScratch` + `scan_borders_to_leaf_stats` wiring.
- Commit `3ed3125` present in git log.
- SUMMARY + deferred-items present.
