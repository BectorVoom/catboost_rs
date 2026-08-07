# SPD-03 resolution — catboost-rs now beats official CatBoost GPU on EVERY device-active cell

**Branch:** `worktree-gpu-host-prep-speed` @ `3e7d886` (grid provenance)
**Runner:** Kaggle, account `boomvector`, **Tesla P100-PCIE-16GB** — the same grid,
protocol and hardware class as the 260808 baseline that recorded the 1M cells at
**0.445×** (2.2× slower than official).

## The verdict

`kaggle-output-260808-r5/result.json`, all five device-activated cells
(`device_activated == true`, probe-confirmed), median of the timed repeats after
the per-cell untimed warm run — the same protocol as every earlier round:

| cell | official (GPU) | catboost_rs (GPU) | ratio official/rs |
|---|---|---|---|
| `SymmetricTree\|RMSE\|unw\|noctr\|300k` | 1.288 s | 0.727 s | **1.77× — WIN** |
| `SymmetricTree\|Logloss\|unw\|noctr\|300k` | 1.507 s | 0.710 s | **2.12× — WIN** |
| `SymmetricTree\|RMSE\|unw\|noctr\|1000k` | 1.794 s | 1.706 s | **1.05× — WIN** |
| `SymmetricTree\|Logloss\|unw\|noctr\|1000k` | 2.009 s | 1.698 s | **1.18× — WIN** |
| `SHOWCASE-bias\|RMSE\|unw\|noctr\|300k` | 1.323 s | 0.632 s | **2.09× — WIN** |

Progression of the 1M RMSE cell (the stubborn one) across the rounds, identical
protocol: **0.445× → 0.70× → 0.91× → 1.05×**; 1M Logloss: **0.455× → 0.74× →
1.03× → 1.18×**. The 300k cells went from roughly at-parity to 1.8–2.1× wins.

## What closed the gap (each attributed on-hardware before it was built)

1. **Wave 2 — the serial host fold + JIT** (`eae8164`). The 2128 ms unattributed
   fit-train gap was the per-iteration HOST fold (serial 1M-object leaf walk +
   approx update + leaf-weight bucketing); parallelized bit-exactly. The
   1222 ms `begin-raw` was first-launch JIT compilation: CubeCL's disk
   compilation cache turned on (off by default) + a fit-entry background warm-up
   thread that compiles the fit's kernel variants during host prep.
2. **Wave 3 — f32 ingest cache** (`38c9873`). NumPy input is already f32; the
   ingest now attaches a bit-exact f32 SoA cache so fit-prep skips its full
   re-narrowing pass (differential test pins bit-identical predictions).
   The same round's one-buffer QPACK upload REGRESSED (455 vs 179 ms) and was
   reverted in wave 4 — a reminder that transfer-path guesses must be measured.
3. **Wave 4 — consume the device's own leaf assignment** (`81b4ee8`). The seam
   had ALWAYS returned `DeviceGrownTree.leaf_of` (crossed with the part-stats
   every tree) in the same forward-bit convention the host walk recomputed;
   using it collapsed the fold from 535 → 94 ms.
4. **Wave 5 — cache-blocked ingest transpose** (this run's commit). The
   per-column NumPy gather re-read the row-major matrix once per column
   (~3.2 GB of traffic); 16-column groups read it effectively once
   (py-ingest 405 → 317 ms measured here, and the timed cells gained more than
   the diag shows since the diag carries profiling fences).

## What remains (honest residuals)

- `fit-prep` is now the largest single host term (~785 ms on the 4-vCPU Kaggle
  host, ~all of it border building at 3.1 s CPU-time across 4 threads). The
  radix sort (wave 4) was a wash on this host — the cost is memory contention
  across the four rayon workers (the 200k-sample Fisher–Yates and gathers), not
  comparator overhead. Further cuts need a contention-aware design, measured on
  the cloud host, not locally.
- These wins cover the DEVICE-ACTIVE grid (unweighted, no-CTR, SymmetricTree
  float cells). Weighted / CTR / Depthwise cells still decline to CPU or N/A —
  device coverage of those is tracked elsewhere
  (`gpu-string-param-oracle-coverage`), and nothing here changes their status.
- The cold first-fit (fresh machine, empty cache) is ~4.0 s at 1M — better than
  the 6.9 s baseline (warm-up + cache) but still above official's cold fit;
  every later process on the machine is disk-cache warm.

## Protocol notes

- Grid timings are median-of-repeats after an untimed in-process warm fit —
  JIT-warm on both sides, the same fairness as every earlier round.
- The diag logs (`diag/diag_cold.txt`, `diag/diag_repeat.txt`) carry the
  per-stage `CB_GPU_PROF` attribution (now including `py-ingest`,
  `begin-inner`, `qpack-fill`, `tree-host`) for the next investigation.
- Every parity surface this work touched is pinned by tests: the fold
  parallelization and device-leaf_of consumption by the full ROCm oracle
  suites, the f32 cache by a bit-exact differential test, the radix sort by an
  adversarial border-set equality test, and the ingest transpose is a pure
  reordering of byte-identical writes.
