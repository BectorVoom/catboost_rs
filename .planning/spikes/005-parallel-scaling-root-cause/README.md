---
spike: 005
name: parallel-scaling-root-cause
type: standard
validates: "Given the histogram rewrite closed the per-core gap (3.35x) but 1->16-thread speedup is only ~1.5-1.7x (vs CatBoost's 2.0-3.0x), when we decompose the per-level work into its serial and parallel phases and sweep thread counts, then we prove the rayon-free O(n*nf) `build_bucket_histogram` accumulation is the Amdahl ceiling AND that a feature-outer restructuring parallelizes it with byte-identical output (no oracle re-baseline)"
verdict: ROOT-CAUSE-CONFIRMED
related: [002, 003, 004]
tags: [perf, cpu, parallelism, rayon, histogram, amdahl, scaling, parity]
---

# Spike 005: parallel-scaling-root-cause

## What This Validates

Spikes 002-004 diagnosed the *algorithmic* slowness (full-rescan split search) and it
was fixed in Phase 21 (per-core gap now median **3.35x**, residual is the irreducible
`O(n_bins*n_leaves*nf)` split-scoring constant). The remaining gap is **parallel
efficiency**: catboost-rs speeds up only **~1.5-1.7x** from 1->16 threads while CatBoost
gets **2.0-3.0x** at these sizes (n=10k-20k, nf=20). The all-cores gap (3.74x) is now
*wider* than the per-core gap (3.35x) purely because of threading. Spike 004 predicted
exactly this and deferred it ("parallelize over candidates/features once the histogram
fix lands").

This spike pins the **exact structural root cause** in the current code and proves the
fix path is parity-safe *before* we commit to it.

---

## Research — Root Cause (grounded in code + `docs/CATBOOST_CORE_DESIGN.md`)

### The design doc's parallelism model (what CatBoost does)

`CATBOOST_CORE_DESIGN.md` step **3c** (the tree-growing pipeline):

> **c. Score every split** — `CalcScores` -> **`CalcBestScore`** runs, **in parallel over
> candidates**, **`CalcStatsAndScores`**. For each split threshold it accumulates
> `TBucketStats { SumWeightedDelta, SumWeight, SumDelta, Count }` over the sampled
> objects... An `IScoreCalcer` then converts those sums into a scalar gain.

The decisive word is **`CalcStatsAndScores`** — in CatBoost, one function *both accumulates
the bucket stats (the `O(n)` pass) AND scores them*, and the whole thing runs **inside the
parallel-over-candidates region**. Each candidate is one feature (`AddFloatFeatures`), so
the expensive `O(n)` accumulation for feature *f* happens on the thread that owns candidate
*f*. `TLearnContext` owns "the `LocalExecutor` (thread pool)" threaded through every call.
=> **CatBoost puts the expensive O(n) accumulation *inside* the parallel region.**

### What catboost-rs does (the divergence)

catboost-rs split the per-level work into **two separate passes** with the parallel
boundary in the wrong place:

1. **Pass 1 — accumulate (SERIAL).** `cb_compute::build_bucket_histogram`
   (`crates/cb-compute/src/histogram.rs:564`) builds the *entire* histogram (all features)
   in **one monolithic object-outer loop**:
   ```rust
   for obj in 0..n_objects {              // line 591 — SERIAL
       ...
       for feature in 0..n_features {     // line 597 — feature-inner
           let bin = bins[feature * n_objects + obj] ...;
           scatter_add_f64(&mut data, cell_base + d, dval);   // ascending-object-order += fold
       }
   }
   ```
   This file is **deliberately rayon-free** (module header, `histogram.rs:165`, "D-03").
   Cost: `O(n * nf)`, single-threaded.

2. **Pass 2 — score (PARALLEL).** `select_level_plain` (`crates/cb-train/src/tree.rs:833`)
   parallelizes the *scoring*:
   ```rust
   let candidates = (0..matrix.n_features())
       .into_par_iter()               // line 834 — the ONLY real parallelism on the hot path
       .map(|feature| scan_and_score_borders(&scratch.hist, feature, ...))
       ...
   ```
   Cost: `O(nf * n_bins * n_leaves)`, parallel over `nf` tasks.

**catboost-rs parallelized the cheap pass and left the expensive pass serial.** This is the
exact inverse of the design doc.

### Why this caps speedup at ~1.9x (Amdahl)

At the baseline point (n=10 000, nf=20, n_bins=128, depth 6 => n_leaves up to 64):

| Phase | Work | Threaded? | Order of magnitude |
|-------|------|-----------|--------------------|
| accumulate (`build_bucket_histogram`) | `n * nf` | **NO (serial)** | 10000*20 = 2.0e5 |
| score (`scan_and_score_borders` x nf) | `nf * n_bins * n_leaves` | yes | 20*128*64 = 1.6e5 |

Roughly **half the per-level work is the serial accumulation**. Amdahl with serial fraction
f ~= 0.5 and p = 16 threads:
```
speedup_max = 1 / (f + (1-f)/p) = 1 / (0.5 + 0.5/16) = 1 / 0.531 ~= 1.88x
```
That is dead-on the observed **1.5-1.7x** (the accumulation fraction is even *higher* for
shallower trees, where `n_leaves` is small and scoring is cheap — dragging the average down).
The serial `build_bucket_histogram` is the Amdahl ceiling; adding cores cannot break it.

### Second-order contributors (measured in 005-B/C, fixed in 006/007)

- **Too few, too-coarse tasks.** Pass 2 spawns exactly `nf` tasks (20). On 16 threads that
  is a 20/16 load imbalance (some threads do 2 tasks, some 1 => ~2x tail) plus per-task
  scratch **allocated inside the `.map` closure** (`tree.rs:848`) contending on the global
  allocator. So even the parallel pass scales sub-linearly.
- **Fork-join per level.** `into_par_iter` is re-entered every level (`depth` x `n_trees`
  fork-joins), each paying rayon latch overhead against tiny tasks.

### The parity constraint — and why the fix is NOT blocked by it

The serial accumulation exists **for a reason**: the `<= 1e-5` oracle bar requires the
byte-identical **ascending-object-order `sum_f64` fold** (`histogram.rs:585-588`:
"folding a cell's members by repeated scatter-add in this object-outer / feature-inner order
is byte-identical to gathering them and calling `sum_f64`"). Naive row-block parallelism
reorders that sum and breaks byte-identity.

**Key insight this spike verifies:** you do **not** need to reorder anything. Because bins are
stored **feature-major** (`bins[feature * n_objects + obj]`, `histogram.rs:598`) and each
histogram cell `(leaf, feature, bin)` belongs to **exactly one feature**, restructuring the
loop to **feature-outer / object-inner** and parallelizing **over features** gives every cell
its members in the **same ascending object order, within a single thread** =>
**byte-for-byte identical output, zero cross-thread float reduction, no fixed-point, no oracle
re-baseline.** As a bonus it turns the current cache-hostile strided bins read (object-outer
over feature-major storage = stride `n_objects`) into a contiguous per-feature scan.

Fixed-point-u64 accumulation (the Phase 10/11 GPU histogram winner, order-independent) is only
needed for the **low-nf / within-feature row-block** regime where per-feature tasks starve
(nf < cores) — that is spike **007**, and *there* parity must be re-checked against the
upstream oracle (not against current rs output).

---

## Experiment

`crates/cb-train/tests/spike005_parallel_scaling_test.rs` (CB_PERF-gated, `--release`),
three parts printing greppable `RSBENCH005 ...` records:

- **005-A end-to-end scaling curve.** Train with the device-declining `CpuHostRuntime`
  inside a local `rayon::ThreadPool` of size 1,2,4,8,16. Report `per_tree_ms` and
  `speedup` vs 1 thread, across a depth sweep (depth 2/4/6) to show the serial fraction
  grows as trees get shallow.
- **005-B phase split.** At the baseline shape, microbench `build_bucket_histogram` (serial)
  vs the full per-feature `scan_and_score_borders` loop (parallel pass). Report the measured
  serial fraction `f` and the Amdahl-predicted 16-thread ceiling `1/(f+(1-f)/16)`.
- **005-C parity escape-hatch PoC.** Build the histogram **feature-parallel** (feature-outer,
  each feature into a compact per-feature buffer, assembled into the frozen layout) and
  `assert_eq!` the full `data` vector against serial `build_bucket_histogram` — proving
  **byte-identity** — while timing it to show the serial phase parallelizes.

## How to Run

```bash
CB_PERF=1 cargo test -p cb-train --release --test spike005_parallel_scaling_test -- --nocapture
```

## What to Expect

- 005-A: speedup plateaus around **1.5-1.9x** at 16 threads, worse at shallow depth.
- 005-B: `f` ~= 0.4-0.6; Amdahl ceiling ~= 1.7-2.1x — matching 005-A (root cause confirmed).
- 005-C: **byte-identical** parity assertion passes; feature-parallel build is faster on 16
  threads (the escape hatch is real).

## Observability

`RSBENCH005 part=A|B|C key=val ...` single-line records. Raw run captured in `results.txt`.

## Investigation Trail

- Read `CATBOOST_CORE_DESIGN.md` step 3c: `CalcStatsAndScores` fuses accumulate+score
  *inside* parallel-over-candidates. catboost-rs separates them and parallelizes only score.
- Confirmed `build_bucket_histogram` is object-outer/serial/rayon-free (`histogram.rs:591,165`)
  and `select_level_plain` parallelizes only scoring (`tree.rs:834`).
- Confirmed feature-major bin storage (`bins[feature*n_objects+obj]`) => feature-outer
  parallelism is disjoint-cell and parity-preserving.

## Results — VERDICT: ROOT CAUSE CONFIRMED (+ parity-safe fix path proven)

16-core box. `CB_PERF=1 cargo test -p cb-train --release --test
spike005_parallel_scaling_test`. Raw records in `results.txt` (two runs, consistent).

### 005-A — end-to-end scaling curve reproduces the symptom

| depth | 1t | 2t | 4t | 8t | 16t (speedup) |
|-------|----|----|----|----|---------------|
| 2 | 1.00 | 1.45 | 1.86 | 1.90 | **1.89** |
| 4 | 1.00 | 1.56 | 1.89 | 1.72 | **1.87** |
| 6 | 1.00 | 1.33 | 1.71 | 1.60 | **1.56** |

- 16-thread speedup lands at **1.56-1.9x**, exactly the reported symptom (~1.5-1.7x).
- Speedup **plateaus at 4-8 threads and REGRESSES past it** (depth 6: 1.71@4t ->
  1.60@8t -> 1.56@16t). Extra cores make it *worse* — a fork-join + too-few-tasks
  signature, not just a ceiling. Deeper trees (more serial-heavy levels early) scale
  worst.

### 005-B — the serial accumulation is a first-order ceiling

`t_build_ms=0.42` (serial `build_bucket_histogram`) vs `t_score_ms=0.58` (parallel-pass
work) => **serial_fraction = 0.41**, Amdahl 16-thread ceiling = **2.20x**.

The rayon-free O(n*nf) accumulation alone caps the whole tree-grow at ~2.2x no matter
the core count. **This is the primary root cause.**

### The two effects compound

Observed end-to-end (1.56-1.9x) is *below* the histogram-only Amdahl ceiling (2.2x),
so there are **two** compounding losses, both traceable to the same structural mistake:
1. **Serial accumulation** (Amdahl ~2.2x cap) — accumulation left out of the parallel
   region.
2. **Weak parallel pass** — only `nf=20` coarse tasks on 16 threads (load imbalance),
   scratch allocated *inside* the closure, and a fresh fork-join per level; this
   plateaus/regresses past 4-8 threads and keeps actual below the 2.2x ceiling.

Both are fixed by the same move CatBoost makes (`CalcStatsAndScores`): **fuse
accumulate+score into one well-partitioned parallel-over-features region**, so the
serial phase disappears (serial_fraction -> ~0, ceiling jumps) *and* the tasks carry
enough work.

### 005-C — the fix is parity-safe (the decisive result)

`parity_byte_identical = TRUE`. The feature-outer / object-inner parallel build is
**bit-for-bit identical** (`f64::to_bits` equality on every cell) to the serial
`build_bucket_histogram`, and ran **1.2-1.45x faster** even as a naive PoC (nf=20,
per-feature buffer alloc + serial assemble — a fused integration will do better).

=> Parallelizing the accumulation **needs no fixed-point, no oracle re-baseline, no
`<= 1e-5` re-verification**: because bins are feature-major and each cell belongs to one
feature, per-feature parallelism preserves the exact ascending-object-order `sum_f64`
fold. The parity wall that justified the serial `build_bucket_histogram` (D-03) simply
does not apply to *feature*-parallelism. (Fixed-point u64 is only needed for the
low-nf / within-feature row-block regime — deferred to spike 007.)

## Signal for the Build

- **006 (the fix):** fuse accumulate+score into ONE `into_par_iter` over features
  (CatBoost `CalcStatsAndScores` shape) with per-task reusable scratch — kills the
  serial phase *and* the coarse-task waste. Feature-outer restructuring is byte-exact
  (proven here), so this is a parity-free refactor, not a numerics change.
- Watch the **>8-thread regression**: at nf=20 the win saturates by ~8 threads; finer
  task granularity (007: row-block within feature, fixed-point for parity) is what
  unlocks >8-thread scaling and the low-nf regime.
