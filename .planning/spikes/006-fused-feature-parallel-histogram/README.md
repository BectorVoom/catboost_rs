---
spike: 006
name: fused-feature-parallel-histogram
type: standard
validates: "Given 005 pinned the parallel ceiling to the SERIAL histogram accumulation being left out of the parallel region, when we fuse accumulate+score into one parallel-over-features pass (CatBoost's CalcStatsAndScores shape), then per-level scaling jumps from ~1.7x to ~5x at 16 threads AND the per-border scores stay byte-identical to the current two-pass path"
verdict: VALIDATED
related: [002, 003, 004, 005]
tags: [perf, cpu, parallelism, rayon, histogram, fused, calcstatsandscores, parity]
---

# Spike 006: fused-feature-parallel-histogram

## What This Validates

005 proved the parallel-scaling ceiling is structural: the O(n*nf) accumulation
(`build_bucket_histogram`) runs SERIAL, outside the parallel region, so ~41% of per-level
work is serial => Amdahl caps the tree-grow at ~2.2x and the observed number is even lower
(1.5-1.7x) because the only parallel pass is `nf` coarse scoring tasks. 005-C proved a
feature-outer parallel build is byte-identical to the serial one.

This spike prototypes the fix — CatBoost's **`CalcStatsAndScores`** shape (design doc
step 3c: "in parallel over candidates, `CalcStatsAndScores`") — and measures whether it
actually recovers scaling while holding the `<= 1e-5` parity bar (in fact byte-identity).

## Approach

Two per-level strategies, benchmarked head-to-head at 1/2/4/8/16 threads via local
`rayon::ThreadPool`s:

- **`two_pass`** (mimics current `tree.rs`): serial `build_bucket_histogram` over ALL
  features (rayon-free), then `(0..nf).into_par_iter()` scoring via `scan_and_score_borders`.
- **`fused`** (the fix): `(0..nf).into_par_iter()`, each task builds its OWN 1-feature
  histogram (the O(n) column scan — a contiguous slice thanks to feature-major storage)
  AND scores it. The accumulation is now INSIDE the parallel region; the serial phase
  disappears.

Both reuse the production `build_bucket_histogram` + `scan_and_score_borders`, so the
per-border scores are directly comparable. Parity is a byte-identity assert (`f64::to_bits`).

## How to Run

```bash
CB_PERF=1 cargo test -p cb-train --release --test spike006_fused_parallel_test -- --nocapture
```

## Results — VERDICT: VALIDATED

16-core box. Raw records in `results.txt`.

### Parity (the gate)

```
part=parity n=10000 nf=20 nbins=128 n_leaves=32  byte_identical=true
part=parity n=40000 nf=8  nbins=254 n_leaves=16  byte_identical=true
```
Fused per-border scores are **byte-for-byte identical** to the two-pass path on both a
wide-feature and a low-feature/high-bin shape. **No oracle re-baseline, no fixed-point.**

### Scaling — baseline n=10 000, nf=20, nbins=128, n_leaves=32

| threads | two_pass ms | two_pass speedup | fused ms | fused speedup |
|---------|-------------|------------------|----------|---------------|
| 1  | 0.891 | 1.00 | 0.919 | 1.00 |
| 2  | 0.734 | 1.21 | 0.479 | 1.92 |
| 4  | 0.722 | 1.23 | 0.331 | 2.77 |
| 8  | 0.594 | 1.50 | 0.259 | 3.54 |
| 16 | 0.536 | **1.66** | 0.183 | **5.01** |

- Fused reaches **5.0x @16 threads** vs two-pass **1.66x** — decisively past the 2.2x
  Amdahl ceiling the serial histogram imposed. The serial fraction is gone.
- **Absolute:** fused is **2.9x faster** per level at 16 threads (0.183 vs 0.536 ms) and
  even ~equal at 1 thread (0.92 vs 0.89 — negligible overhead from per-feature builds).

### Scaling — other shapes

- **low-nf (n=10k, nf=8):** fused 4.27x @8t but dips to 3.30x @16t — only 8 tasks can't
  fill 16 threads. two_pass regresses to 1.25x. => low-nf needs within-feature
  parallelism (spike 007).
- **large-n (n=40k, nf=20):** fused 4.22x @8t, dips to 2.98x @16t; two_pass never beats
  1.29x. The @16t dip is per-task buffer alloc **inside the closure** contending on the
  global allocator — i.e. the reusable-scratch requirement (007), not a structural limit.

## Investigation Trail

- Built `run_two_pass` (serial build + parallel score) and `run_fused` (parallel
  build+score per feature), both on production `build_bucket_histogram` /
  `scan_and_score_borders`, so scores are directly comparable.
- Parity asserted first (the approach's gate) on two contrasting shapes — passed
  byte-identical, confirming 005-C's feature-outer = serial claim end-to-end through the
  scorer.
- Swept threads with local pools; fused scaling ~3x the two-pass speedup at 16t.
- Observed fused's own @16t dip at low-nf / large-n => isolated to task granularity +
  in-closure allocation, both explicitly 007 scope. NOT a parity or structural blocker.

## Signal for the Build (the real integration)

1. **Restructure `select_level_plain` (and `_perturbed`) to fuse accumulate+score per
   feature** in the existing `(0..n_features).into_par_iter()` — each task builds its
   feature's histogram and scores it. Delete the separate serial `build_bucket_histogram`
   call from the per-level hot path. Parity is preserved byte-for-byte (proven here), so
   this is a refactor, not a numerics change — the existing oracle suite is the guard.
2. **Keep the subtraction trick per-feature.** The current whole-partition
   `build_bucket_histogram` + `relocate_sub` (`tree.rs:745-796`) must move INTO the
   per-feature task: each feature builds the smaller child's stats and subtracts to get
   the sibling. Parallelization is orthogonal to the sub-trick; both compose.
3. **Reuse per-thread scratch** (a `TLearnContext` analogue) instead of allocating the
   per-feature histogram buffer inside the `.map` closure — this removes the >8-thread
   allocator contention seen at large-n. Use `rayon`'s `map_init` or a thread-local
   scratch pool sized `n_leaves * n_bins * n_channels`.
4. **Defer low-nf / >8-thread scaling to spike 007** (within-feature row-block
   parallelism + fixed-point-u64 for parity — the only regime where byte-identity is lost
   and upstream re-verification is required).
