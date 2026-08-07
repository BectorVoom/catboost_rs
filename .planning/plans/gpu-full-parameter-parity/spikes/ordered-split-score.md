# T21 — per-segment ordered split-score device kernel design note

**Deliverable for FPP-18.** Supersedes the Track-O paragraph in `WAVE7-SPIKES.md`.

## Headline: gate G1 was fired on a false premise — Track O is SMALL

`WAVE7-SPIKES.md` escalated Track O under G1 with this reasoning:

> The split score for a candidate must therefore be accumulated **per permutation segment**,
> not over the whole pool. […] there is no per-segment accumulation kernel […] A per-segment
> split-score kernel is net-new CubeCL work: a segmented fill whose segment boundaries are the
> fold prefix lengths, plus a segmented argmin, plus its own self-oracle.

**Two of those three claims are wrong**, and the error is checkable in one function.

### Correction 1 — the segments are NESTED PREFIXES, not disjoint ranges

`ordered_segment_leaf_stats` (`crates/cb-train/src/tree.rs:2294`) takes `body_finish` and
`tail_finish` and then **discards `body_finish`**:

```rust
    // crates/cb-train/src/tree.rs:2317
    let _ = body_finish;
    for p in 0..upper {            // upper = tail_finish.min(n)
```

with the reason given in the comment immediately above it: because `random_strength == 0` on
this path, `SampleWeightedDerivatives == WeightedDerivatives`, so body and tail rows are
accumulated *identically* and "a single contiguous walk over `[0, tail_finish)` is exact."

So segment `s` is not the range `[b_s, b_{s+1})`. It is the **prefix `[0, b_{s+1})`** of the
learn permutation. `body_finish` survives only in the regularizer, via
`scale_l2_reg(l2, body_sum_weight_s, body_finish_s)` (`tree.rs:2405`).

For `n = 30, multiplier = 2.0` the boundaries are `[1,2,4,8,16,30]` (`fold.rs:135`), so the
five "segments" are the five prefixes of length **2, 4, 8, 16, 30**.

### Correction 2 — no segmented FILL kernel is needed. None. The existing fill already does it

`launch_partition_hist2_resident_into` walks an `indices` array — documented at
`gpu_runtime/mod.rs:883` as "*(length `n`, the object visiting order)*" — and indexes
`der1[obj]`, `weight[obj]`, `cindex[feature*n + obj]`, `leaf_of[obj]` by `indices[i]`, **not**
by `i`. The resident session merely happens to pass the IDENTITY permutation (`mod.rs:2084`).

Therefore:

> **Passing the learn permutation as `indices` and `n = boundaries[s+1]` yields the
> prefix histogram for segment `s` exactly, with zero kernel changes.**

`der1` / `weight` / `cindex` / `leaf_of` stay full length `n` (they are indexed by object id);
only the *visiting count* shrinks. This is the whole segmented-fill problem, solved by an
argument change.

### Correction 3 — the cost is ~2n, not S·n

Boundaries grow geometrically (`select_tail_size` = `ceil(size * multiplier)`, default 2.0),
so summing the prefix lengths from the largest down:

```
n + n/2 + n/4 + … < 2n
```

**All S prefix fills together cost less than twice a single full fill.** S itself is
`O(log₂(n / SelectMinBatchSize(n)))` — 5 at n=30, ~10 at n=100 000.

### Correction 4 — no segmented ARGMIN either

The argmin is over candidates, and the candidate set does not change per segment. The segment
sum happens *inside* one candidate's score, before any comparison. So the existing
single-argmin structure stands; only the per-candidate score body gains an outer loop.

## What is actually net-new: ONE kernel, ~25 lines

`find_optimal_split_partition_kernel` (`crates/cb-backend/src/kernels.rs:4565`) computes, per
candidate `c`, a `score_acc` by looping `part` over partitions and folding bins into
`left_*`/`right_*`. The ordered variant wraps that in an outer segment loop:

```
for seg in 0..n_segments:
    lambda = scaled_l2[seg]                     # per-segment scale_l2_reg
    base   = seg * seg_stride                   # seg_stride = n_parts * leaf_stride
    for part in 0..n_parts:  … unchanged …      # reads bin_sums[base + cell]
```

Everything else — the grid-stride candidate sweep, the `f32::MIN` sentinel, the shared-mem
block argmin, the `real_folds` eligibility bound, the strict-`>`/lowest-index tie-break, the
host-side across-block reduction — is **reused verbatim**.

### The L2-only simplification (do not skip this)

`score_candidate_ordered` hard-codes `l2_split_score` (`tree.rs:2415`). The ordered path is
**L2-only**, which is what makes the segment loop a plain accumulator: Cosine would need a
*per-segment* denominator (`num_s / sqrt(den_s)`) and could not share one `score_acc`.

Any other `score_function` on an ordered fit **declines to CPU**; it is never approximated.

## Answers to FPP-18's five questions

| # | Question | Answer |
|---|---|---|
| 1 | Segment descriptor across the seam? | **Nothing crosses the per-tree seam.** `body_tail_boundaries` / `body_sum_weights` are pure functions of `(n, multiplier, weights)` — per-FIT constant. They live in `DeviceTrainConfig` (V-3's convention), not `FamilyTreeArgs`. |
| 2 | Segmented fill kernel? | **None.** Reuse `launch_partition_hist2_resident_into` with `indices = permutation`, `n = boundaries[s+1]`. |
| 3 | Segmented argmin? | **None.** The segment sum is inside one candidate's score; the argmin is unchanged. |
| 4 | Launch shape | Unchanged: `num_cubes = pass_candidates.div_ceil(CUBE_DIM)`, `CubeDim{x: CUBE_DIM}`. The `bin_sums` arg grows to `S * n_parts * leaf_stride`; `scaled_l2` grows from a length-1 to a length-S `Array<F>`. |
| 5 | Size estimate | **~25 net-new `#[cube]` lines** (one outer loop + a per-segment lambda/base) plus ~120 host driver lines. Against G1's "> ~3 days-equivalent" trigger: **does not fire.** |

## Deliberate simplifications (correctness before speed)

1. **Subtraction trick OFF on the ordered arm** (`filter_mask = 0` at every level). The trick is
   valid per segment, but the prefix fills already total < 2n; leaving it off removes the
   parent-chain invariant from the new path's correctness argument. Revisit only under a
   measured profile.
2. **Permutation uploaded once per fit**, not per tree (it is fold state, not tree state).
3. **`n_bins` family selection is unchanged** — the ordered fill dispatches the same
   `{32,64,128,256}` arms.

## Risks

- **Float accumulation order differs from CPU.** CPU computes `sum_f64(per-leaf terms)` per
  segment then `sum_f64(segment scores)`; the kernel accumulates one running `score_acc` over
  `(seg, part, left/right)`. Split CHOICE is what must match (integer equality); the summed
  score is held to ε=1e-4, which is T22's stated bar.
- **A prefix boundary of 0.** `select_min_batch_size` returns ≥ 1, and `body_tail_boundaries`
  returns `[]` only for `n == 0`, which the grow path short-circuits. Guard anyway.
- **`indices` shorter than the routing kernel expects.** `launch_partition_split_*` requires
  `indices` to cover `0..n` (`mod.rs:2084`). The prefix trick applies to the **fill only**;
  routing must keep the identity/full-length array. Do not cross-wire these.

## Verdict

**G1 does not fire. Proceed to T22.** The escalation in `WAVE7-SPIKES.md` is superseded; its
Track-O row should read "implemented" once T22–T24 land.
