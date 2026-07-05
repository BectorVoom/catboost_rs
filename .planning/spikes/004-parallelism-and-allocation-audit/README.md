---
spike: 004
name: parallelism-and-allocation-audit
type: standard
validates: "Given CatBoost saturates all cores with minimal per-iteration allocation, when we profile cb-train's boosting loop, then we pinpoint single-threaded sections, redundant CTR/quantization recompute, and per-iteration memory churn"
verdict: CONTRIBUTING-CAUSES-CONFIRMED
related: [002, 003]
tags: [perf, cpu, parallelism, rayon, allocation]
---

# Spike 004: parallelism-and-allocation-audit

## What This Validates

The *constant-factor* causes layered on top of the algorithmic one (Spike 003):
threading and per-iteration memory churn. CatBoost saturates all cores via a
`LocalExecutor` thread pool and reuses scratch buffers (`TLearnContext`); does
`catboost_rs`?

## Results — VERDICT: CONTRIBUTING CAUSES CONFIRMED

### 1. CPU training is 100% single-threaded

`grep -rl rayon crates/cb-train/src crates/cb-compute/src crates/cb-model/src`
→ **NONE**. The oblivious hot path (`tree.rs`) contains no `par_iter`, no thread
pool, no `std::thread`. (The earlier "46 parallelism signals" were the word
"threaded" in doc comments and `thread_count` in the RNG — not real parallelism.)

The design doc (§ step 3c) mandates the opposite: "`CalcScores` → `CalcBestScore`
runs, **in parallel over candidates**, `CalcStatsAndScores`", and `TLearnContext`
owns "the `LocalExecutor` (thread pool)". CatBoost's default gap already reflects
this: at n=20 000 it is **19.2 ms/tree at 1 thread but 4.98 ms/tree at 16 threads**
(~3.9× from cores). catboost_rs cannot claw any of that back — it is pinned to one
core. This multiplies the Spike 002 single-thread gap (~217×) up to the ~840×
default-vs-default gap.

### 2. Per-candidate allocation churn (the OOM/constant-factor amplifier)

Every `score_candidate` call (`n_features × n_bins` times **per level**) allocates:

| Allocation | Site | Per candidate |
|---|---|---|
| `chosen.to_vec()` | `tree.rs:429` | 1 × `Vec<Split>` |
| `Vec<bool>` per object | `assign_leaves` `tree.rs:399` | **n_objects** small Vecs |
| `delta_members`, `weight_members` | `histogram.rs:58-59` | 2 × `Vec<Vec<f64>>` (n_leaves inner Vecs) |
| gathered leaf Vecs (grown by push) | `histogram.rs:69-74` | n_objects pushes w/ reallocation |

At the baseline level (n=10 000, nf=20, nbins=128) that is ≈ `20·128 = 2 560`
candidates × (10 000 `Vec<bool>` + nested gather Vecs) = **tens of millions of
heap allocations per level**, ~hundreds of millions per tree. This is why the
first full-grid release run was killed (exit 1, no panic — allocator/OOM pressure
on a heavier row), and it inflates the per-`n` constant well beyond the raw
`O(n·nf·nbins·depth)` op count.

The design doc's `TLearnContext` explicitly holds **reusable scratch buffers**
(`SampledDocs`, `SmallestSplitSideDocs`, `PrevTreeLevelStats`, `ScratchCache`) so
the inner loop allocates ~nothing. catboost_rs allocates the world per candidate.

### 3. Recompute — not the primary factor here

The benchmark used numeric-only data (no CTRs), so CTR recompute is not implicated
in these numbers. Quantization borders are passed in once. The dominant waste is
the per-candidate rescan + allocation, not recompute. (A CTR-path perf spike would
be a reasonable frontier follow-up but was out of this grid's scope.)

## Recommended Fix (for the real build — not done in this spike)

1. **Parallelize over candidates / features** with `rayon` in `select_level_*`
   once the histogram fix (Spike 003) lands — histogram accumulation and per-bin
   scoring are embarrassingly parallel. Keep the final reduction ordered for
   parity (per-feature independent, deterministic merge).
2. **Reuse scratch buffers** across candidates/levels/iterations (a `TLearnContext`
   analogue): one `leaf_of: Vec<usize>` updated incrementally, fixed-size
   histogram arrays, no nested `Vec<Vec<f64>>`. Eliminates the per-candidate
   allocation storm.
3. Ordering: fix the algorithm (Spike 003) first — it removes the `n_bins` /
   `n_features` blow-up and most allocations; then parallelism buys the remaining
   ~core-count factor.

## Investigation Trail

- `grep` for rayon/par_iter across training crates → none; confirmed the hot loop
  is a plain sequential `for feature { for border { ... } }`.
- Read `assign_leaves` + `reduce_leaf_stats` allocation sites; cross-referenced the
  release run's silent exit-1 (no panic) as allocator/OOM pressure.
- Cross-referenced CatBoost's own 1-thr vs 16-thr numbers (Spike 002 grid) to size
  the parallelism component (~3.9× at n=20 000).
