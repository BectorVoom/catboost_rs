# `bench/BENCH-03-SIGNOFF.md` — Milestone-closing CUDA speed-parity sign-off (BENCH-03)

This is the signed-off proof that the device-resident training path **reverses** the
original >20× host-light slowdown. As of **Phase 15 (2026-07-05)** it is recomputed from
ONE authoritative single-session Tesla P100 CUDA run
(`bench/phase15_cuda_oracle/result.json`, `single_session: true`) rather than stitched from
the earlier per-phase sessions: correctness (Part A ALL-PASS, ε=1e-4), the 12 device-vs-host
BENCH-02 rows (Part B), and the informational official-CatBoost-GPU arm (Part C) all come
from the **same GPU / driver / seed**. This discharges the v1.1 GPUT-14 aggregate gap and
the un-run Phase-10 (depth-1) / Phase-11 (depth-6) BENCH-02 rows (see *Standing debt —
discharged*, below).

## Standing disciplines (copied verbatim from `bench/RESULTS.md`)

- **Correctness is a BLOCKING gate.** No speed number is valid unless the device correctness
  oracles ALL pass first (T-10-25). The single-session harness `sys.exit(2)`s on any Part A
  failure before timing, so a fast-but-wrong number can never be recorded here.
- **ROCm in-env is smoke-only — NOT a gate.** The authoritative oracle of record is the
  single-session Kaggle CUDA run.
- **Do NOT fabricate numbers.** Every cell below traces to
  `bench/phase15_cuda_oracle/result.json`. Region `catboost_gpu_s` stays `N/A`
  (informational, never a proxy).

---

## Verdict

# BENCH-03: PASS

**Every one of the 12 single-session device rows is `>=20x` vs the host-CPU baseline (D-01
hard gate).** The measured device-vs-host-CPU speedups span **29.1× … 40.8×** across the
depthwise + region families at depth-1 (n = 100k / 300k / 1M) and depth-6 (n = 10k / 100k /
300k) — comfortably above the 20× bar on every row (`bench_verdict: OK`, `depth6_ge20x:
true`, GE20X gate = 20.0).

**Baseline reversed.** The gate is signed off against the **pre-Phase-10 host-light CPU
baseline** documented in [`.planning/notes/gpu-training-host-light-root-cause.md`](../.planning/notes/gpu-training-host-light-root-cause.md):
before Phase 10 the device grew trees **host-light** — derivatives on-device but the entire
greedy tree-search inner loop ran serially on the host CPU, making the GPU path **>20×
slower than pure CPU** (an architectural MVP boundary, not a bug). The device-resident grow
loop landed across Phases 10–13 now runs that inner loop on-device, and the aggregate below
shows the original **>20× *slowdown* is demonstrably reversed into a 29–41× *speedup***.

**Correctness passed as a blocking gate this session (Part A ALL-PASS, ε=1e-4).** The
single-session P100 CUDA run (2026-07-05) ran all **13 v1.1 device families** under
`--features cuda` as a **BLOCKING pre-gate** before any timing: every family `exit==0` /
`ran_any_tests==true`, named divergences bit-exact (`max abs_div=0.000e0`; the only nonzero
deltas are the inherently-stochastic bootstrap `2.384e-7` and MVS `~1e-15`, all far under
the 1e-4 bar), and all **4 RV-13 oracles seen == expected** (`tie_order_matches_cpu_stable_descending`,
`softmax_weight_max_seed`, `empty_group_means_no_fault`, `pairwise_near_equal_border_tiebreak`)
→ `correctness_verdict = ALL-PASS`. This single aggregate row IS the GPUT-14 sign-off (no
longer per-family-only).

---

## Aggregate speed matrix (single session)

All device / host-CPU / speedup / `>=20x?` columns are the Part B rows from the one
single-session run; the `catboost_gpu_s (informational)` column is the Part C arm of the
**same** session (not a separate run). Measurement protocol (all rows): warm-run /
JIT-excluded / train-only wall-clock, lazy CubeCL queue drained before the clock stops;
median-of-3, 20 iterations, 20 features, 32 bins.

| depth | family | n | device_cuda_s | host_cpu_s | speedup_vs_host_cpu | >=20x? | catboost_gpu_s (informational) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | depthwise | 100000  | 0.4813 | 14.9418  | 31.045× | yes | 0.6857 |
| 1 | depthwise | 300000  | 1.4986 | 49.6221  | 33.112× | yes | 0.7858 |
| 1 | depthwise | 1000000 | 6.0235 | 196.0425 | 32.546× | yes | 0.9400 |
| 1 | region    | 100000  | 0.7273 | 29.5788  | 40.669× | yes | N/A |
| 1 | region    | 300000  | 2.3273 | 94.8537  | 40.757× | yes | N/A |
| 1 | region    | 1000000 | 9.7365 | 381.3196 | 39.164× | yes | N/A |
| 6 | depthwise | 10000   | 0.0921 | 2.8278   | 30.704× | yes | 0.6864 |
| 6 | depthwise | 100000  | 0.8499 | 31.4309  | 36.982× | yes | 0.7167 |
| 6 | depthwise | 300000  | 2.6964 | 108.6983 | 40.312× | yes | 0.8325 |
| 6 | region    | 10000   | 0.1117 | 3.2557   | 29.147× | yes | N/A |
| 6 | region    | 100000  | 0.9214 | 37.2071  | 40.381× | yes | N/A |
| 6 | region    | 300000  | 2.9998 | 118.4237 | 39.477× | yes | N/A |

**Rows:** 12 — depth-1 (n = 100k / 300k / 1M) and depth-6 (n = 10k / 100k / 300k), each ×
{depthwise, region}. **All 12 rows `>=20x`** (min 29.147× at region depth-6 n=10k; max
40.757× at region depth-1 n=300k). **Crossover:** depth-1 depthwise device first beats CPU
at **n=100000** (`crossover.note: "device first beats CPU at n=100000"`) — recorded, not
gated (A4/D-10-09).

### Provenance (single session — D-08)

- **All rows from ONE session:** `bench/phase15_cuda_oracle/result.json`,
  `provenance.single_session = true`. No mixed-session stitching, no cell sourced from the
  older Phase-12/13 sessions.
- **Hardware/env:** Tesla P100-PCIE-16GB (driver 580.159.04, 16384 MiB), `nvcc` release
  12.8, seed 42. `correctness_verdict = ALL-PASS`, `bench_verdict = OK`,
  `catboost_gpu_verdict = OK`.
- **catboost_gpu_s** — Part C of the same session, official `catboost==1.2.10`
  `task_type='GPU'` on the reproduced `gen()` synthetic workload; **informational only**
  (see Divergence notes).

---

## Divergence notes (D-01 / D-08 — informational, NOT a milestone gate)

The `catboost_gpu_s` column is **informational context only**. Per D-01 the sole blocking
BENCH-03 criterion is the `>=20x`-vs-host-CPU-baseline gate above; matching a mature
C++/CUDA library's absolute throughput is an explicit **stretch goal, not the definition of
done**. The head-to-head is recorded transparently, with these documented divergences
(from `result.json.catboost_gpu.divergences`):

1. **Region CatBoost-GPU = `N/A`.** Official CatBoost has no `Region` grow policy, so there
   is no comparable arm to time — the cell is `N/A`, **never a proxy or forced number**.
   Region's device-vs-host-CPU speedup gate still passes on its own single-session rows.
2. **`border_count` 128 → 32.** CatBoost's GPU default `border_count` is 128; it was set to
   **32** to match the bench's quantization grid (integer-binned 0..31 columns fed as
   float32 with `border_count=32`).
3. **Quantization-cost asymmetry (informational — NOT subtracted).** CatBoost `fit()`
   wall-clock **includes** on-device quantization, while catboost-rs times **only** the grow
   loop. This asymmetry is **documented and left un-adjusted** (D-08): the CatBoost-GPU
   numbers are not corrected downward to "match" what catboost-rs measures, precisely
   because the column is informational rather than a gate.

Because the official-CatBoost-GPU numbers reflect a broader (quantization-inclusive)
wall-clock on a different policy surface, they are **not directly comparable** to the
catboost-rs grow-loop numbers and are **not** used to pass or fail the milestone.

---

## Standing debt — DISCHARGED here (Phase 15, D-04/D-08/D-11)

The two items that this sign-off previously carried as open v1.1 debt are now **discharged
by the single-session Phase-15 run**:

- **GPUT-14 aggregate correctness sign-off — DISCHARGED.** The single session ran all 13
  v1.1 device families in ONE `--features cuda` invocation with `correctness_verdict =
  ALL-PASS` at ε=1e-4 and 4/4 RV-13 oracles seen. This IS the milestone-wide aggregate row
  that was previously "evidenced per-family only." `.planning/REQUIREMENTS.md` HARD-01 is
  flipped to complete; `.planning/MILESTONES.md` Known Gap and `.planning/STATE.md` Deferred
  Item are cleared.
- **Phase-10 (depth-1) + Phase-11 (depth-6) BENCH-02 rows — DISCHARGED.** Both depth rows
  now exist as real single-session measurements (`bench/RESULTS.md` depth-1 + depth-6
  blocks; the 12 rows above). The aggregate is **no longer stitched** from the committed
  Phase-12/13 numbers — every row here traces to the one Phase-15 `result.json`. HARD-02 is
  flipped to complete.

No BENCH-03 cell is sourced only from the Phase-12/13 sessions where a Phase-10/11 depth row
is now required. The older per-session numbers survive as history in `bench/RESULTS.md`
(marked superseded).

---

## Sources

- [`bench/phase15_cuda_oracle/result.json`](phase15_cuda_oracle/result.json) — the single
  authoritative record: `correctness_verdict: ALL-PASS`, per-family + RV-13 divergences,
  `bench02.depth_rows` (12 rows) + `crossover` + `provenance.single_session: true` + Part C
  `catboost_gpu` arm.
- [`bench/phase15_cuda_oracle/oracle.py`](phase15_cuda_oracle/oracle.py) — the
  correctness-blocks-speed single-session runner (Part A gate → Part B timing → Part C
  informational CatBoost-GPU arm).
- [`bench/RESULTS.md`](RESULTS.md) — the depth-1 + depth-6 sign-off blocks filled from the
  same session.
- [`.planning/notes/gpu-training-host-light-root-cause.md`](../.planning/notes/gpu-training-host-light-root-cause.md) — the pre-Phase-10 host-light baseline this sign-off reverses.
- Historical (superseded): `bench/phase12_cuda_oracle/bench02-result.json`,
  `bench/phase13_cuda_oracle/result.json`, `bench/phase14_cuda_signoff/` — retained as prior
  per-session evidence, no longer load-bearing for the aggregate.
