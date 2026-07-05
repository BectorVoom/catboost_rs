# `bench/RESULTS.md` — Kaggle CUDA sign-off log (BENCH-01 / BENCH-02)

This is the **human sign-off log** for the authoritative per-phase Kaggle CUDA run.
Every later GPU phase (11 → 15) appends a dated block here so each kernel is **both
correctness-tested and speed-measured on CUDA** — not merely benchmarked at the end
(the BENCH-02 standing discipline).

As of **Phase 15 (2026-07-05)** the authoritative record is ONE single-session Tesla P100
CUDA run (`bench/phase15_cuda_oracle/result.json`, `single_session: true`, driven by
`bench/phase15_cuda_oracle/oracle.py`). It supersedes the old primitive-by-primitive
notebook and the multi-session `aggregate.py` stitching: correctness is measured as
per-family device self-oracles under `--features cuda`, and BOTH depth-1 and depth-6
BENCH-02 speed rows are emitted from the same kernel session (D-04/D-05/D-07). The
depth-1 and depth-6 blocks below are filled from that one run.

- **Correctness is a BLOCKING gate.** No speed number is valid unless the device
  correctness oracles ALL pass first (T-10-25). The harness `sys.exit(2)`s on any Part A
  failure before timing, so a fast-but-wrong number can never be recorded here.
- **ROCm in-env is smoke-only — NOT a gate.** The authoritative oracle of record is the
  single-session Kaggle CUDA run.
- **Do NOT fabricate numbers.** Every filled cell below traces to
  `bench/phase15_cuda_oracle/result.json`. An empty measured cell means the run has not
  happened yet.

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
> and **the Kaggle run is the arbiter of the crossover.** The single-session Phase-15 run
> measured the depth-1 rows at n = 100k / 300k / 1M and **recorded the crossover** (device
> first beats CPU at n=100000, the smallest n tested) — recorded, never forced to a
> device≥CPU pass (A4). The small-n infeasibility is **surfaced explicitly here, never
> silently assumed.**

---

## Run log

### Run template — copy this block per Kaggle run

```
### Run YYYY-MM-DD — phase <N>, <GPU model>, CUDA <ver>, driver <ver>
Signed-off-by: <name>

Correctness gate (BLOCKING — all must PASS):
# Bar semantics (match fixtures/README.md): integer/INDEX outputs (sort perm,
# reduce-by-key KEYS, cindex) are BIT-EXACT; FLOAT reduces accumulate in f64 and
# match the serial CPU ref within <=1e-4 (device bar) / <=1e-5 (CPU ref-of-record).
| Oracle                          | Bar        | Result       | max|err|      |
|---------------------------------|------------|--------------|---------------|
| primitive: scan                 | <=1e-4     | <PASS/FAIL>  | <value>       |
| primitive: radix sort / reorder | bit-exact  | <PASS/FAIL>  | <value>       |
| depth-1 RMSE (calc_average)     | <=1e-5     | <PASS/FAIL>  | <value>       |
| depth-1 Logloss (calc_average)  | <=1e-5     | <PASS/FAIL>  | <value>       |
GATE: <PASS/FAIL>

Speed (only if GATE PASSED — warm-run/JIT-excluded, queue drained):
| Workload (n_rows x n_features)  | device (cuda) s | host-CPU s | CatBoost GPU s | device>=CPU? |
|---------------------------------|-----------------|------------|----------------|--------------|
| <workload>                      | <value>         | <value>    | <value>        | <yes/no>     |
Crossover note (n where device first beats CPU, or "not reached"): <value>
```

---

### Run 2026-07-05 — phase 15 (depth-1), Tesla P100-PCIE-16GB, CUDA release 12.8, driver 580.159.04
Signed-off-by: orchestrator-driven single-session Kaggle CUDA run (`bench/phase15_cuda_oracle/`, `single_session: true`)

This depth-1 record and the depth-6 record below both come from ONE authoritative
single-session P100 CUDA run (`bench/phase15_cuda_oracle/result.json`). The single-session
harness (`oracle.py`) runs correctness as per-family device self-oracles under
`--features cuda` at the ε=1e-4 device bar (CPU reference-of-record stays ≤1e-5, D-04) and
gates BEFORE any Part B timing (D-05). `correctness_verdict = ALL-PASS`; all 13 v1.1 device
families `exit==0` / `ran_any_tests==true`; `rv13_oracles_expected == rv13_oracles_seen`
(4/4).

Correctness gate (BLOCKING — Part A ALL-PASS, ε=1e-4 device bar; each cell traces to a result.json family divergence):
| Oracle / primitive coverage                             | Bar        | Result | max\|div\|                          |
|---------------------------------------------------------|------------|--------|-------------------------------------|
| primitive: scan / segmented-scan (ordered family)       | <=1e-4     | PASS   | 0.000e0 (bit-exact, bound 1e-9)     |
| primitive: radix sort / reorder (exact_quantile+segmented_sort) | bit-exact | PASS | 0.000e0 (10/10)                |
| primitive: update_part_props (ordered partition_update) | <=1e-4     | PASS   | 0.000e0 (bit-exact, bound 1e-9)     |
| cindex bit-pack (GPUT-15)                               | bit-exact  | PASS   | consumed by every device grow family (nonsym/region/ordered); no isolated row this session |
| depth-1 RMSE  (nonsym depthwise-l2 leaf-values)         | <=1e-5     | PASS   | 0.000e0                             |
| depth-1 Logloss (nonsym depthwise-cosine leaf-values)   | <=1e-5     | PASS   | 0.000e0                             |
GATE: PASS  (correctness_verdict = ALL-PASS; 13/13 families exit==0/ran_any_tests; 4/4 RV-13 oracles seen)

_Coverage note: reduce-by-key / segmented-reduce are Phase-10 device primitives exercised
inside the family device paths above; the single-session Phase-15 FAMILIES filter set runs
them within the family self-oracles rather than as standalone rows (their bit-exact status
is unchanged from the committed Phase-10 runs). Every value in the table above is quoted
verbatim from `result.json` family `divergences`._

Speed (Part B — ran only because GATE PASSED; warm-run / JIT-excluded / lazy-CubeCL-queue drained / median-of-3, 20 iters / 20 feat / 32 bins):
| depth-1 workload (n_rows) | device (cuda) s | host-CPU s | CatBoost GPU s | speedup | device>=CPU? |
|---------------------------|-----------------|------------|----------------|--------:|--------------|
| depthwise, n=100000       | 0.4813          | 14.9418    | 0.6857         | 31.05×  | yes          |
| depthwise, n=300000       | 1.4986          | 49.6221    | 0.7858         | 33.11×  | yes          |
| depthwise, n=1000000      | 6.0235          | 196.0425   | 0.9400         | 32.55×  | yes          |
| region,    n=100000       | 0.7273          | 29.5788    | N/A            | 40.67×  | yes          |
| region,    n=300000       | 2.3273          | 94.8537    | N/A            | 40.76×  | yes          |
| region,    n=1000000      | 9.7365          | 381.3196   | N/A            | 39.16×  | yes          |
Crossover note (n where device first beats CPU): **n=100000** (depth-1 depthwise, the
smallest n tested — recorded, not gated per A4/D-10-09). Region `catboost_gpu_s = N/A` (no
official CatBoost Region grow_policy — never proxied). `bench_verdict: OK`.

Correctness gate: **PASS** · Speed (depth-1, large n): **device 31.05×–40.76× vs host CPU on every row**

---

## Phase 11 — depth-6 CUDA sign-off (GPUT-14 / GPUT-06 / BENCH-02)

Phase 11 adds the **depth-6** partition-aware grow loop + Newton der2 leaf estimation. The
**device bar is ε=1e-4** (standing GPUT-14 bar); the **CPU reference-of-record stays
≤1e-5** (Plan 01 `expected_depth6_tree.json`, byte-unchanged — D-04). **Correctness is
BLOCKING before any speed number.** The depth-6 rows below are filled from the SAME
single-session Phase-15 P100 CUDA run as the depth-1 block above (the depth lever is
`BENCH_DEPTH`, default 6 — both depth rows ran in one kernel session, D-07).

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
> single-session run observed the depth-6 nonsym leaf-values **bit-exact (0.000e0)** at the
> committed fixture sizes.

> **Fixed-point quantization headroom vs n (IN-03, Phase 11 review).** The device histogram
> sums `Σ round(vᵢ·2^30)/2^30`, not the CPU `sum_f64(vᵢ)`: each per-object contribution is
> rounded to the `2^30` grid before the integer atomic add. Per-term error is ≤ 2^-31, so
> the per-bin error is bounded by `n·2^-31` (≈2.3e-7 at n=500, ≈4.7e-4 at n≈1e6). This is
> comfortably inside the ε=1e-4 grow bar for the committed sizes. The single-session depth-6
> rows (n up to 300k) reported leaf-value divergence 0.000e0, confirming the margin at the
> tested sizes.

### Run 2026-07-05 — phase 15 (depth-6), Tesla P100-PCIE-16GB, CUDA release 12.8, driver 580.159.04
Signed-off-by: orchestrator-driven single-session Kaggle CUDA run (`bench/phase15_cuda_oracle/`, `single_session: true`)

Correctness gate (BLOCKING — device bar 1e-4 vs the Rust CPU path; CPU ref-of-record 1e-5; each cell traces to a result.json family divergence):
| Oracle (depth-6)                                            | Bar       | Result | max\|div\|                     |
|-------------------------------------------------------------|-----------|--------|--------------------------------|
| Gate A: device leaf-values RMSE arm (nonsym depthwise-l2)   | <=1e-4    | PASS   | 0.000e0 (bit-exact)            |
| Gate A: device leaf-values Logloss arm (nonsym depthwise-cosine) | <=1e-4 | PASS  | 0.000e0 (bit-exact)            |
| Gate A: lossguide leaf-values (l2 / cosine)                 | <=1e-4    | PASS   | 0.000e0 (bit-exact)            |
| Gate B: device full-run e2e fit (device_nonsym_fit, cb-train)| <=1e-4   | PASS   | full-fit pred parity (2/2)     |
| Gate B: device full-run e2e fit Region (device_region_fit)  | <=1e-4    | PASS   | full-fit pred parity (1/1)     |
| Split-agreement oracle (nonsym grow-loop, per-level)        | exact     | PASS   | 0.000e0 (leaf-values bit-exact)|
GATE: PASS   first divergent tree (if any): none (all leaf-value divergences 0.000e0)

Speed (Part B — depth-6, warm-run / JIT-excluded / queue drained / median-of-3, 20 iters / 20 feat / 32 bins):
| depth-6 workload (n_rows) | device (cuda) s | host-CPU s | CatBoost GPU s | speedup | device>=CPU? |
|---------------------------|-----------------|------------|----------------|--------:|--------------|
| depthwise, n=10000        | 0.0921          | 2.8278     | 0.6864         | 30.70×  | yes          |
| depthwise, n=100000       | 0.8499          | 31.4309    | 0.7167         | 36.98×  | yes          |
| depthwise, n=300000       | 2.6964          | 108.6983   | 0.8325         | 40.31×  | yes          |
| region,    n=10000        | 0.1117          | 3.2557     | N/A            | 29.15×  | yes          |
| region,    n=100000       | 0.9214          | 37.2071    | N/A            | 40.38×  | yes          |
| region,    n=300000       | 2.9998          | 118.4237   | N/A            | 39.48×  | yes          |
Notes: host-CPU is the same `--features cuda` binary's host-CPU boosting loop; CatBoost GPU
is `task_type='GPU'` depth=6 (informational, Region N/A). `depth6_ge20x: true`;
`bench_verdict: OK`. Correctness blocked before any speed number was recorded (D-05).

Correctness gate (depth-6): **PASS** · Speed (depth-6, large n): **device 29.15×–40.38× vs host CPU on every row**

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

> **Superseded (2026-07-05, Phase 15):** the Phase-12 (and Phase-13) per-session BENCH-02
> numbers are retained above as history, but the AUTHORITATIVE BENCH-02 rows are now the
> single-session Phase-15 depth-1 + depth-6 blocks (one GPU / one driver / one seed /
> `single_session: true`). See `bench/BENCH-03-SIGNOFF.md` for the recomputed single-session
> BENCH-03 aggregate.
