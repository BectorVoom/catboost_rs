# `bench/RESULTS.md` — Kaggle CUDA sign-off log (BENCH-01 / BENCH-02)

This is the **human sign-off log** for the authoritative per-phase Kaggle CUDA run
(`bench/cuda_oracle.ipynb`). Every later GPU phase (11 → 13) appends a dated block here
so each kernel is **both correctness-tested and speed-measured on CUDA** — not merely
benchmarked at the end (the BENCH-02 standing discipline).

- **Correctness is a BLOCKING gate.** No speed number is valid unless the primitive +
  cindex (bit-exact) and depth-1 RMSE/Logloss (`<=1e-5`) oracles ALL pass first
  (T-10-25). The notebook halts on any failure, so a fast-but-wrong number can never be
  recorded here.
- **ROCm in-env is smoke-only — NOT a gate.** The authoritative oracle of record is this
  human-gated Kaggle CUDA run.
- **Do NOT fabricate numbers.** All `TBD` cells below are filled from the actual notebook
  output. An empty measured cell means the run has not happened yet.

---

## D-10-09 ESCALATION (pinned — read before trusting any depth-1 speed number)

> **Depth-1 device >= CPU is achievable ONLY at large n (~1e5–1e6+ rows). At small n it
> is physically infeasible regardless of optimization.** A depth-1 stump is the single
> most launch-overhead-bound workload in the entire milestone: per tree the device does a
> handful of kernel launches over trivial per-object work, while the CPU grows the stump
> in microseconds. No kernel fusion or residency makes a GPU launch + driver round-trip
> competitive at small n — **this is physics, not a tuning gap.** Device wins only at
> **large n** where the `O(n·features)` histogram parallelizes enough to amortize the
> fixed launch latency.
>
> **Therefore BENCH-02's depth-1 speed bar is PINNED to the large-n synthetic workload**
> (D-06 `SPEED_CONFIG`, ~1e6×50, tunable above the break-even). **The Kaggle run is the
> arbiter of the crossover.** If, after large-n fused/resident optimization, depth-1
> device still does not beat CPU even at large n, record the measured crossover (or its
> absence) below and let the user decide whether D-10-09 stands for depth-1 (depth-6 in
> Phase 11 is where device dominance is unambiguous). The small-n infeasibility is
> **surfaced explicitly here, never silently assumed.**

---

## Run log

### Run template — copy this block per Kaggle run

```
### Run YYYY-MM-DD — phase <N>, <GPU model>, CUDA <ver>, driver <ver>
Signed-off-by: <name>

Correctness gate (BLOCKING — all must PASS):
# Bar semantics (match fixtures/README.md + cuda_oracle.ipynb): integer/INDEX
# outputs (sort perm, reduce-by-key KEYS, cindex) are BIT-EXACT; FLOAT reduces
# accumulate in f64 and match the serial CPU ref within <=1e-4 (they are also
# byte-identical run-to-run per kernels/reduce.rs, but that is DETERMINISM, not
# CPU bit-parity — do not record a tighter bar than the primitive's stated one).
| Oracle                          | Bar        | Result | max|err| |
|---------------------------------|------------|--------|----------|
| primitive: scan                 | <=1e-4     | TBD    | TBD      |
| primitive: segmented scan       | <=1e-4     | TBD    | TBD      |
| primitive: radix sort / reorder | bit-exact  | TBD    | TBD      |
| primitive: reduce-by-key (keys) | bit-exact  | TBD    | TBD      |
| primitive: reduce-by-key (vals) | <=1e-4     | TBD    | TBD      |
| primitive: segmented-reduce     | <=1e-4     | TBD    | TBD      |
| primitive: update_part_props    | <=1e-4     | TBD    | TBD      |
| cindex bit-pack (GPUT-15)       | bit-exact  | TBD    | TBD      |
| depth-1 RMSE (calc_average)     | <=1e-5     | TBD    | TBD      |
| depth-1 Logloss (calc_average)  | <=1e-5     | TBD    | TBD      |
GATE: TBD (PASS/FAIL)

Speed (only if GATE PASSED — large n SPEED_CONFIG, warm-run/JIT-excluded, queue drained):
| Workload (n_rows x n_features)  | device (cuda) s | host-CPU s | CatBoost GPU s | device>=CPU? |
|---------------------------------|-----------------|------------|----------------|--------------|
| depth-1, ~1e6 x 50, 100 iters   | TBD             | TBD        | TBD            | TBD          |
Crossover note (n where device first beats CPU, or "not reached"): TBD
```

---

### Run (pending first Kaggle CUDA execution)

_No authoritative Kaggle CUDA run recorded yet._ This harness is committed and
reproducible; a human runs `bench/cuda_oracle.ipynb` on a Kaggle CUDA instance and pastes
a filled copy of the template above here, then fills the `SPIKE-REDUCTION.md` §4 CUDA
`err`/`ms` rows for reduce candidates (a)/(b)/(c) from the same run.

Correctness gate: **TBD** · Speed (large n): **TBD**

---

## Phase 11 — depth-6 CUDA sign-off (GPUT-14 / GPUT-06 / BENCH-02)

Phase 11 adds the **depth-6** partition-aware grow loop + Newton der2 leaf estimation. The
authoritative gate is the depth-6 section of `bench/cuda_oracle.ipynb`, run on Kaggle CUDA.
The **device bar is ε=1e-4** (standing GPUT-14 bar); the **CPU reference-of-record stays
≤1e-5** (Plan 01 `expected_depth6_tree.json`, byte-unchanged — D-04). **Correctness is
BLOCKING before any speed number.** ROCm in-env (Plans 02–04) is smoke-only, **not** the
gate (in-env gfx1100 already shows structure bit-exact + zero run-to-run spread; Kaggle CUDA
is the oracle of record).

> **Determinism scope (WR-03, Phase 11 review).** The "zero run-to-run spread" / "0 spread"
> gate is scoped to tree **STRUCTURE** (the per-level split `(feature, bin)` sequence and the
> per-object `leaf_of`), which derives from the fixed-point `Atomic<u64>` histogram fill
> (GPUT-06) and is therefore bit-identical run to run. **Leaf VALUES (and thus predictions)
> are NOT bit-deterministic:** the leaf-stat reduce (`partition_update_kernel`) still merges
> with a naked float atomic, so leaf values carry ulp-level run-to-run float-order variance.
> That variance stays far inside the ε=1e-4 device bar (it does not compound past tolerance
> over the tested tree counts), so it is not a correctness defect — but a strict
> bit-reproducibility claim for predictions is out of scope until `partition_update_kernel`
> is routed through the same fixed-point integer-atomic accumulate as the histogram fill. The
> "Per-tree run-to-run spread" row below measures the first divergent STRUCTURE prefix.

> **Fixed-point quantization headroom vs n (IN-03, Phase 11 review).** The device histogram
> sums `Σ round(vᵢ·2^30)/2^30`, not the CPU `sum_f64(vᵢ)`: each per-object contribution is
> rounded to the `2^30` grid before the integer atomic add. Per-term error is ≤ 2^-31, so
> the per-bin error is bounded by `n·2^-31` (≈2.3e-7 at n=500, ≈4.7e-4 at n≈1e6). This is
> comfortably inside the ε=1e-4 grow bar for the committed sizes, but the margin SHRINKS with
> `n` and the full-run preds compound it across ~200 trees. No fixture change is required —
> but the Kaggle per-tree diagnostic should track the effective histogram error vs `n` so the
> 1e-4 margin at ~1e6 rows is CONFIRMED, not assumed.

**All fields below are PENDING the human-gated Kaggle run — do NOT fabricate.** Fill them
from the notebook's `PHASE 11 DEPTH-6 CUDA STRUCTURED REPORT` cell (correctness first).

```
### Run YYYY-MM-DD — phase 11 (depth-6), <GPU model>, CUDA <ver>, driver <ver>
Signed-off-by: <name>

Correctness gate (BLOCKING — device bar 1e-4 vs the Rust CPU path; CPU ref-of-record 1e-5):
| Oracle (depth-6)                                    | Bar       | Result | max|err| |
|-----------------------------------------------------|-----------|--------|----------|
| Gate A: device 1-tree RMSE    vs CPU ref (base-free)| <=1e-4    | TBD    | TBD      |
| Gate A: device 1-tree Logloss vs CPU ref (base-free)| <=1e-4    | TBD    | TBD      |
| Gate B: device full-run RMSE    vs cpu-wheel preds  | <=1e-4    | TBD    | TBD      |
| Gate B: device full-run Logloss vs cpu-wheel preds  | <=1e-4    | TBD    | TBD      |
| Split-agreement oracle (CUDA grow-loop, per-level)  | exact     | TBD    | —        |
| Per-tree run-to-run spread (first divergent prefix) | 0 spread  | TBD    | TBD      |
GATE: TBD (PASS/FAIL)   first divergent tree (if any): TBD

Speed (only if GATE PASSED — depth-6 SPEED_CONFIG ~1e6x50, warm-run/JIT-excluded, train-only):
| Workload (depth-6, ITERS x n_rows x n_features)     | device (cuda) s | host-CPU s | CatBoost GPU s | device>=CPU? |
|-----------------------------------------------------|-----------------|------------|----------------|--------------|
| depth-6 RMSE,    200 x ~1e6 x 50                     | TBD             | TBD        | TBD            | TBD          |
| depth-6 Logloss, 200 x ~1e6 x 50                     | TBD             | TBD        | TBD            | TBD          |
Notes: host-CPU is a separate cpu-feature wheel run (compile-time features); CatBoost GPU is
task_type='GPU' depth=6. Correctness blocks before any speed number is recorded.
```

### Phase 11 run (pending first Kaggle CUDA execution)

_No authoritative Phase-11 depth-6 Kaggle CUDA run recorded yet._ A human runs the depth-6
section of `bench/cuda_oracle.ipynb` on Kaggle CUDA, confirms the blocking ε=1e-4 gate for
RMSE + Logloss and the per-tree diagnostic (no compounding split-flip drift), records the
depth-6 speed numbers, and pastes a filled copy of the template above here.

Correctness gate (depth-6): **TBD** · Speed (depth-6, large n): **TBD**

---

### Run 2026-07-04 — phase 12, Tesla P100-PCIE-16GB, CUDA 12.8 (V12.8.93), driver 580.159.04
Signed-off-by: automated Kaggle CLI run (kernel `yensen2/catboost-rs-phase12-cuda-oracle`)

Method: each Phase-12 family's existing device self-oracle (`*_test.rs`) run under
`cargo test --release --no-default-features --features cuda` on the CUDA `SelectedRuntime`
(device path vs the inline Rust CPU reference). Correctness is the BLOCKING gate.

Correctness gate (BLOCKING — all PASS, bar ε=1e-4):
| Family (Req)                         | Oracle                                   | Result | max\|div\|                |
|--------------------------------------|------------------------------------------|--------|---------------------------|
| Depthwise/Lossguide (GPUT-18)        | nonsym_grow_test 4/4                      | PASS   | 0.000e0 (bit-exact)       |
| Region (GPUT-18)                     | region_device_test 1/1                   | PASS   | 0.000e0 (bit-exact)       |
| Exact Quantile/MAE/MAPE (GPUT-19)    | exact_quantile+segmented_sort 10/10      | PASS   | 0.000e0                   |
| Bootstrap (GPUT-09)                  | bootstrap_device_test 5/5                | PASS   | Bernoulli exact; Bayes 2.384e-7 |
| MVS (GPUT-17)                        | mvs_device_test 3/3                      | PASS   | 6.66e-16 … 4.44e-15       |
| CTR ordered/one-hot/tensor (GPUT-10) | ctr_device_test 5/5                      | PASS   | good/total exact, val ≤1e-4 |
| e2e device fit non-sym (GPUT-18)     | device_nonsym_fit_test 2/2 (cb-train)    | PASS   | full-fit pred parity      |
| e2e device fit Region (GPUT-18)      | device_region_fit_test 1/1 (cb-train)    | PASS   | full-fit pred parity      |

Verdict: **ALL-PASS — 31 device tests, 0 failed.** Provenance: `bench/phase12_cuda_oracle/`.
Note: this also incidentally exercises the resident depth>1 grow path on CUDA (u64/f64
atomics present), which the in-env ROCm runtime currently cannot run (Atomic<u64> regression).

BENCH-02 speed (train-only, warm, JIT-excluded, depth=6 / 20 iters / 20 feat / 32 bins,
device GpuBackend vs host-CPU boosting loop in one --features cuda binary):
| family    | n       | device_s | cpu_s   | speedup |
|-----------|---------|----------|---------|---------|
| depthwise | 10,000  | 0.083    | 2.511   | 30.3x   |
| depthwise | 100,000 | 0.746    | 29.842  | 40.0x   |
| depthwise | 300,000 | 2.568    | 101.930 | 39.7x   |
| region    | 10,000  | 0.101    | 3.178   | 31.3x   |
| region    | 100,000 | 0.875    | 36.817  | 42.1x   |
| region    | 300,000 | 2.872    | 113.303 | 39.5x   |
Grow-loop device dominance 30-42x across n. Sub-op families (Exact/bootstrap/MVS/CTR)
are device-resident within this same loop (no standalone train loop to isolate). Provenance:
bench/phase12_cuda_oracle/bench02-result.json.
