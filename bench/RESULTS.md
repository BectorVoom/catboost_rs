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
