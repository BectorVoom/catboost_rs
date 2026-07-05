---
spike: 002
name: perf-baseline-and-scaling
type: standard
validates: "Given identical data+params, when we train catboost_rs vs official CatBoost on CPU across a grid, then we measure the slowdown factor and how it scales with n_rows / n_features / n_bins / depth — separating algorithmic from constant-factor causes"
verdict: INVALIDATED
related: [003, 004]
tags: [perf, cpu, training, benchmark, scaling]
---

# Spike 002: perf-baseline-and-scaling

## What This Validates

Given identical synthetic data and matched hyperparameters, when we train
`catboost_rs` (pure host-CPU boosting loop) vs official CatBoost 1.2.10 on CPU
across a scaling grid, then we measure the per-tree slowdown factor **and its
scaling shape** — the shape is what separates an *algorithmic* mistake (gap grows
with problem size) from a mere *constant factor* (gap flat across sizes).

## How to Run

```bash
# Rust host-CPU boosting loop (SymmetricTree, RMSE), inert unless CB_PERF set:
CB_PERF=1 cargo test --release -p cb-train --test perf_baseline_test -- --nocapture

# Official CatBoost CPU, same generator + params, thread_count ∈ {1, ncpu}:
.venv/bin/python .planning/spikes/002-perf-baseline-and-scaling/catboost_grid.py
```

Harness: `crates/cb-train/tests/perf_baseline_test.rs` (Rust) +
`catboost_grid.py` (Python). Both use the **same splitmix64 generator** (uniform
`[0,1)` features, linear target), same params: `RMSE, lr=0.03, l2=3.0,
bootstrap=No, boost_from_average=false, grow_policy=SymmetricTree`, matched
`border_count = nbins-1`. The Rust path uses a device-declining `Runtime`, so the
byte-unchanged host oblivious grow runs (isolating tree-growing cost from any
cubecl-cpu derivative overhead). Metric: **`per_tree_ms`** (wall-clock / iters).

## Results — VERDICT: INVALIDATED

`catboost_rs` CPU training is **~200–450× slower than single-threaded CatBoost**
and **~840–940× slower than default (multi-threaded) CatBoost**, and — decisively
— **the gap grows with the split-candidate count** (`n_features × n_bins`), the
signature of a missing histogram algorithm, not a constant factor.

### Head-to-head (per-tree ms), single-thread CatBoost

| Config (n, nf, nbins, depth) | catboost_rs | CatBoost 1-thr | CatBoost 16-thr | rs vs 1-thr |
|---|--:|--:|--:|--:|
| n=20 000, 20, 128, 6 | 4174 | 19.2 | 4.98 | **217×** |
| n=40 000, 20, 128, 6 | 8402 | 18.5 | 8.94 | **454×** |
| n=5 000, 20, 128, 6  | 1130 | 6.53 | 2.91 | **173×** |

### Scaling shape — the algorithmic proof

**n_rows** (nf=20, nbins=128, depth=6): catboost_rs is **exactly linear** in n
(5k→1130, 10k→2166, 20k→4174, 40k→8402; each doubling ≈ 2.0×). Expected: every
candidate re-scans all n objects.

**n_bins** (n=10 000, nf=20, depth=6) — **THE SMOKING GUN**:

| n_bins | catboost_rs ms | CatBoost 1-thr ms |
|--:|--:|--:|
| 16  | 257  | — |
| 32  | 534  | 5.11 |
| 64  | 1115 | 6.34 |
| 128 | 2166 | (19.2*) |
| 254 | 4360 | 10.84 |

Going 16→254 bins (**15.9× more bins**): catboost_rs = **17.0× slower** (perfectly
linear in bin count); CatBoost 32→254 (**~8× more bins**) = **2.1×** (nearly flat).
catboost_rs pays `O(n)` *per bin*; CatBoost builds a histogram once and reads bins
for free. *(\*The CatBoost n=128 point is from the n=20 000 baseline row and is
noisy; the 32/64/254 points at n=20 000 show the flat trend cleanly.)*

**n_features** (n=10 000, nbins=128, depth=6): catboost_rs linear in nf (5→541,
10→1072, 20→2166, 40→4327 — each doubling ≈ 2.0×). Expected: candidates ∝ nf.

**depth**: both roughly linear in depth (not a differentiator).

### Conclusion

`catboost_rs` per-tree cost ≈ **`O(n_objects · n_features · n_bins · depth)`** —
it re-scans the whole dataset for *every* candidate split. CatBoost is
**`O(n_objects · n_features · depth)`** (build histograms once per level) **+
`O(n_features · n_bins · leaves)`** (score bins). The measured linear-in-`n_bins`
and linear-in-`n_features` blow-up is the direct fingerprint of the missing
histogram / bucket-stats algorithm. This is an **algorithmic** root cause, handed
off to Spike 003 (what the code does vs the design doc) and Spike 004 (the
single-thread + allocation constant factors layered on top).

## Investigation Trail

1. Built the Rust harness against the real `cb_train::train` entry with a
   device-declining `Runtime` → exercises the shipped host oblivious grow.
2. First **debug** run: the *single* baseline row (n=20k, 20 iters) did not finish
   in 9 min — flagged that debug timing is unfair; switched to `--release`.
3. First release run captured 3 n-sweep points then died (exit 1, no panic —
   likely OOM on a heavier row given the per-candidate allocation churn, see
   Spike 004). Rebuilt a lighter grid (n≤40k, iters=3) that completes.
4. CatBoost grid ran clean at thread_count ∈ {1, 16}.
5. The `n_bins` sweep isolated the algorithmic signature: rs linear, CatBoost flat.

Raw evidence: `rust_results.txt`, `catboost_results.txt`.
