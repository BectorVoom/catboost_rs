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
| Oracle                          | Bar        | Result | max|err| |
|---------------------------------|------------|--------|----------|
| primitive: scan                 | <=1e-4     | TBD    | TBD      |
| primitive: segmented scan       | <=1e-4     | TBD    | TBD      |
| primitive: radix sort / reorder | bit-exact  | TBD    | TBD      |
| primitive: reduce-by-key        | <=1e-4     | TBD    | TBD      |
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
