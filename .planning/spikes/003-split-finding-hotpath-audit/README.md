---
spike: 003
name: split-finding-hotpath-audit
type: standard
validates: "Given the design doc's quantized-histogram + oblivious-tree pipeline, when we trace cb-train's actual CPU split-finding, then we determine whether the fast histogram path is on the hot loop or there is exact-scan / per-iteration recompute divergence"
verdict: ROOT-CAUSE-CONFIRMED
related: [002, 004]
tags: [perf, cpu, histogram, tree, split-finding, design-parity]
---

# Spike 003: split-finding-hotpath-audit

## What This Validates

Whether the CPU split-finding hot path implements the histogram / `TBucketStats`
algorithm that `docs/CATBOOST_CORE_DESIGN.md` (§ *The Tree-Growing Pipeline*,
step 3c) describes — or diverges into a naive per-candidate full-dataset rescan.

## Results — VERDICT: ROOT CAUSE CONFIRMED

**The CPU oblivious split search does NOT build histograms. For every
`(feature, border)` candidate it re-partitions the entire dataset and re-reduces
it from scratch.** This is a direct divergence from the project's own design
document, and it is the dominant root cause of the Spike 002 slowdown.

### The dispatched hot path (default `SymmetricTree`, no CTR)

`boosting.rs:3914` → `greedy_tensor_search_oblivious_perturbed`
→ per level `select_level_plain` (`tree.rs:608`)
→ **for every feature, for every border** → `score_candidate` (`tree.rs:419`):

```rust
fn score_candidate(matrix, chosen, candidate, der1, weight, ...) -> f64 {
    let mut splits = chosen.to_vec();     // (1) alloc: clone chosen splits
    splits.push(candidate);
    let leaf_of = assign_leaves(matrix, &splits, n_objects);  // (2) O(n) rescan, Vec<bool>/obj
    multi_dim_candidate_score(&leaf_of, der1, weight, ...)    // (3) O(n) re-reduce, nested Vecs
}
```

- `assign_leaves` (`tree.rs:396`) loops over **all n objects**, and for each
  re-evaluates **every already-chosen split** (`matrix.passes`), allocating a
  `Vec<bool>` per object.
- `reduce_leaf_stats` (`cb-compute/src/histogram.rs:49`) is called per candidate
  and allocates `2 × n_leaves` nested `Vec<f64>`, pushes all n objects, then sums.

So per level: **`O(n_features · n_bins · n_objects · level)`** work, repeated for
every candidate. The `O(n_objects)` pass is paid **once per candidate** instead of
once per level.

### What the design doc says the code SHOULD do (§ step 3c)

> "For each split threshold it **accumulates `TBucketStats { SumWeightedDelta,
> SumWeight, SumDelta, Count }`** over the sampled objects…" — i.e. a **histogram**:
> one `O(n)` pass per feature bins the objects; scoring each of the feature's
> borders is then `O(n_bins)` from the bucket sums, with **no `n_objects` factor**.

> *Subtraction trick:* "one child's bucket stats are derived by **subtracting** the
> sibling's from the parent's … avoiding a rescan."

The design doc mandates histograms + the subtraction trick; the code implements
neither on CPU.

### The irony: the histogram already exists — but only on the GPU path

`crates/cb-backend/src/kernels/pointwise_hist.rs` (Phase 11 `pointwise_hist2` +
subtraction trick) is a real per-feature bin-histogram accumulator — but it is
wired **only into the device grow** (`grow_tree_on_device`). When the backend
declines (every CPU build), the loop falls back to the naive host rescan above.
The CPU product never gets a histogram.

### Why it is this way (the design mistake)

The host reduction was written **parity-first**: `histogram.rs` gathers each
leaf's contributions into per-leaf `Vec`s and sums them through the single
sanctioned `cb_core::sum_f64` primitive to reproduce CatBoost's `thread_count==1`
float-summation order bit-exactly (D-05 / D-08, the ≤1e-5 oracle bar). The
simplest correct way to hit that bar was "assign every object, gather, sum" — per
candidate. Correct, but it discards the histogram structure and re-does the
`O(n)` work `n_features × n_bins` times per level. The design doc describes the
right algorithm; the CPU implementation took the parity shortcut and never came
back for the histogram.

## Recommended Fix (for the real build — not done in this spike)

1. **Build per-feature histograms once per level** on CPU: a single `O(n)` pass
   accumulates `(feature, bin) → {Σder1, Σweight}` for the current leaf partition;
   score all borders of a feature from its histogram in `O(n_bins)`. Mirrors
   `pointwise_hist.rs` on the host. Expected: collapses the `n_bins` and
   `n_features` linear blow-up measured in Spike 002.
2. **Subtraction trick**: derive the larger child's histogram by subtracting the
   smaller sibling from the parent — halves per-level accumulation.
3. **Preserve the ≤1e-5 parity bar**: accumulate bins with a deterministic ordered
   sum (fixed-point u64 accumulators already proven in Phase 10/11, or per-bin
   ordered `sum_f64`) so bit-exactness survives the algorithm change. This is the
   crux — the fix must keep D-05/D-08 parity while dropping the per-candidate
   rescan.

## Investigation Trail

- Traced dispatch: `boosting.rs` calls `greedy_tensor_search_oblivious_perturbed`
  (3914), `_with_ctr` (3854), `leaf_wise_grower` (3800). Confirmed the default
  symmetric/no-CTR arm reaches `select_level_plain`.
- Read `select_level_plain` / `score_candidate` / `assign_leaves` /
  `reduce_leaf_stats` — confirmed full-object rescan + allocation per candidate.
- A comment at `tree.rs:2292` calls this the "L2/Cosine der histogram" path, but
  no per-feature bin histogram is built there — it is the per-candidate leaf
  reducer. Misnomer, not a histogram.
- Confirmed `pointwise_hist.rs` exists but is device-only.
