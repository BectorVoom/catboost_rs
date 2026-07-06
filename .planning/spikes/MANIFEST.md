# Spike Manifest

## Idea

Determine whether catboost 1.2.10's training-time `ComputeOnlineCTRs(AveragingFold)`
ui8 bins can be reproduced bit-exact OFFLINE in pure Rust/Python from committed
fixtures — the feasibility gate for closing Phase-5 bar (c) (pc=4 / SC-1 / ORD-01
production-default AveragingFold parity). The answer drives the choice between
re-planning 05-18 for a real offline CTR port vs. a live-instrumentation path vs.
deferring bar (c).

## Idea (Spikes 002–004)

Root-cause why `catboost_rs` CPU **training** is slower than official CatBoost CPU,
and identify design mistakes against `docs/CATBOOST_CORE_DESIGN.md`. Measure the
slowdown and its scaling shape (algorithmic vs constant-factor), audit the CPU
split-finding hot path against the design doc's quantized-histogram + oblivious-tree
pipeline, and profile parallelism / per-iteration allocation. Investigation only —
produces a verified root-cause report with recommended fixes.

## Requirements

- An offline parity oracle for the AveragingFold online CTR must be derived from a
  **self-consistent** `(permutation, bins)` pair. The currently-committed pair
  (`upstream_avg_perm` + `upstream_avg_ctr_bins_avg_order`) is internally
  inconsistent under the upstream algorithm and is NOT a valid oracle (Spike 001).
- cb-train's `materialize_ctr_feature` / online-prefix / ui8 quantization must NOT
  be "fixed" to chase the committed bins — it is already bit-exact to the upstream
  C++ algorithm (Spike 001). Any re-plan must target the oracle/ground-truth, not
  this code.

### CPU training performance (Spikes 002–004)

- The CPU oblivious split search MUST be reimplemented as a per-feature **bin
  histogram** (`TBucketStats`: Σder1, Σweight per bin) built with ONE `O(n)` pass
  per level, plus the **subtraction trick** — replacing the current per-candidate
  full-dataset rescan (`score_candidate`/`assign_leaves`). This is the dominant
  root cause (~200–450× single-thread slowdown; gap scales linearly with n_bins ×
  n_features). A device histogram already exists to mirror: `pointwise_hist.rs`.
- The histogram fix MUST preserve the ≤1e-5 parity bar: accumulate bins with a
  deterministic ordered sum (fixed-point u64, proven in Phase 10/11, or per-bin
  ordered `sum_f64`) so bit-exactness survives the algorithm change. The current
  slowness is a *direct consequence* of the parity-first gather-and-sum shortcut —
  the fix must keep parity while dropping the rescan.
- After the histogram lands, the split search SHOULD be parallelized over
  features/candidates (rayon) and use reusable scratch buffers (a `TLearnContext`
  analogue) — CatBoost gets ~3.9× from cores at n=20k, and the current per-candidate
  allocation storm (~10^8 allocs/tree) is what OOM-killed the full benchmark grid.

### Parallel scaling (Spike 005 — the histogram landed, this is the new bottleneck)

- The per-level accumulation MUST be moved INTO the parallel region. The current split —
  serial `build_bucket_histogram` (O(n·nf), rayon-free by D-03) followed by a
  parallel-only scoring pass — leaves ~41% of the work serial, capping the whole
  tree-grow at ~2.2× (Amdahl) regardless of core count. The fix is CatBoost's
  `CalcStatsAndScores` shape: **fuse accumulate+score into one `into_par_iter` over
  features** so accumulation is threaded and the tasks carry real work.
- The fused parallel build MUST be **feature-outer / object-inner**, parallelized OVER
  FEATURES. Because bins are feature-major and each histogram cell belongs to exactly
  one feature, this preserves the exact ascending-object-order `sum_f64` fold per cell
  — Spike 005-C proved it **byte-for-byte identical** to the serial build. This is a
  parity-FREE refactor: NO fixed-point, NO oracle re-baseline, NO ≤1e-5 re-verification.
- Per-task scratch MUST be reusable (not allocated inside the `.map` closure) to remove
  the fork-join-per-level allocation churn that makes 16 threads slower than 8 at nf=20.
- Fixed-point-u64 order-independent accumulation (Phase 10/11 GPU winner) is ONLY needed
  for the low-nf / within-feature ROW-BLOCK regime (nf < cores). There — and only there —
  byte-identity is lost and parity must be re-verified against the UPSTREAM oracle, not
  against current rs output. (Deferred: Spike 007.)

## Spikes

| # | Name | Type | Validates | Verdict | Tags |
|---|------|------|-----------|---------|------|
| 001 | online-ctr-averaging-fold-offline | standard | Offline reproduction of `ComputeOnlineCTRs(AveragingFold)` ui8 bins bit-exact from committed inputs | ✗ NOT-ACHIEVABLE (committed oracle pair proven internally inconsistent; cb-train CTR code already correct) | ctr, parity, online-ctr, averaging-fold, phase-05, ord-01, pc4, bar-c |
| 002 | perf-baseline-and-scaling | standard | Given identical data+params, when we train rs vs official CPU across a grid, then we measure slowdown factor + scaling with n_rows/n_features/n_bins/depth (algorithmic vs constant-factor) | ✗ INVALIDATED (catboost_rs CPU train is ~200–450× slower single-thread, ~840–940× vs default multi-thread; gap grows LINEARLY with n_bins and n_features → algorithmic, not constant factor) | perf, cpu, training, benchmark, scaling |
| 003 | split-finding-hotpath-audit | standard | Given the design doc's histogram+oblivious-tree pipeline, when we trace cb-train's CPU split-finding, then we determine whether the fast histogram path is on the hot loop or there is exact-scan/recompute divergence | ✓ ROOT-CAUSE CONFIRMED (CPU `select_level_plain`/`score_candidate` re-scans ALL objects per (feature,border) candidate — no `TBucketStats` histogram, no subtraction trick; diverges from design doc §3c; the real `pointwise_hist.rs` histogram is GPU-only) | perf, cpu, histogram, tree, split-finding, design-parity |
| 004 | parallelism-and-allocation-audit | standard | Given CatBoost saturates all cores with minimal per-iter allocation, when we profile the boosting loop, then we pinpoint single-threaded sections, CTR/quantization recompute, and per-iteration memory churn | ⚠ CONTRIBUTING CAUSES CONFIRMED (zero rayon in cb-train/compute/model → 100% single-threaded; per-candidate allocation storm: Vec<bool>/obj + nested Vec<Vec<f64>> gather, ~10^8 allocs/tree, caused the release-run OOM) | perf, cpu, parallelism, rayon, allocation |
| 005 | parallel-scaling-root-cause | standard | Given the histogram rewrite closed the per-core gap (3.35×) but 1→16-thread speedup is only ~1.5-1.7× (vs CatBoost 2-3×), when we decompose per-level work into serial/parallel phases and sweep threads, then we pin the exact structural cause AND prove the fix is parity-safe | ✓ ROOT-CAUSE CONFIRMED (rayon-free O(n·nf) `build_bucket_histogram` accumulation is left OUT of the parallel region → serial_fraction≈0.41, Amdahl 16t ceiling≈2.2×; observed 1.56-1.9× is even lower because the only parallel pass is nf=20 coarse tasks that plateau/regress past 4-8 threads. CatBoost fuses accumulate+score in `CalcStatsAndScores` *inside* parallel-over-candidates. **Fix is parity-FREE**: feature-outer parallel build is byte-identical to serial — proven `parity_byte_identical=true`, no oracle re-baseline) | perf, cpu, parallelism, rayon, histogram, amdahl, scaling, parity |
| 006 | fused-feature-parallel-histogram | standard | Given 005 pinned the ceiling to serial accumulation, when we fuse accumulate+score into one parallel-over-features pass (CatBoost `CalcStatsAndScores` shape), then per-level scaling jumps ~1.7×→~5× at 16t while scores stay byte-identical | ✓ VALIDATED (fused reaches **5.0× @16t** vs two-pass 1.66×, **2.9× faster absolute**; parity `byte_identical=true` on nf=20 AND nf=8/nbins=254 → parity-free refactor. Residual @16t dip at low-nf/large-n = task granularity + in-closure alloc → spike 007. Build signal: fuse into `select_level_plain`/`_perturbed`, keep sub-trick per-feature, reuse per-thread scratch via `map_init`) | perf, cpu, parallelism, rayon, histogram, fused, calcstatsandscores, parity |
