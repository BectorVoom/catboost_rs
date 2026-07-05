# `bench/BENCH-03-SIGNOFF.md` — Milestone-closing CUDA speed-parity sign-off (BENCH-03)

This is the **terminal deliverable of the v1.1 GPU-performance milestone**: the signed-off
proof that the device-resident training path **reverses** the original >20× host-light
slowdown. It **aggregates** the already-committed per-phase BENCH-02 Kaggle CUDA speed
checks (Phase 12 + Phase 13, real Tesla P100) into one head-to-head matrix, adds the
**informational** official-CatBoost-GPU column from one new human-gated Phase-14 CUDA run,
and labels every number with its source session so mixed-session provenance is explicit.

## Standing disciplines (copied verbatim from `bench/RESULTS.md`)

- **Correctness is a BLOCKING gate.** No speed number is valid unless the primitive +
  cindex (bit-exact) and depth-1 RMSE/Logloss (`<=1e-5`) oracles ALL pass first
  (T-10-25). The notebook halts on any failure, so a fast-but-wrong number can never be
  recorded here.
- **ROCm in-env is smoke-only — NOT a gate.** The authoritative oracle of record is this
  human-gated Kaggle CUDA run.
- **Do NOT fabricate numbers.** All `TBD` cells below are filled from the actual notebook
  output. An empty measured cell means the run has not happened yet.

---

## Verdict

# BENCH-03: PASS

**Every one of the 12 aggregated device rows is `>=20x` vs the host-CPU baseline (D-01 hard
gate).** The measured device-vs-host-CPU speedups span **23.9× … 42.1×** across the
depthwise + region families at n = 10k / 100k / 300k — comfortably above the 20× bar on
every row.

**Baseline reversed.** The gate is signed off against the **pre-Phase-10 host-light CPU
baseline** documented in [`.planning/notes/gpu-training-host-light-root-cause.md`](../.planning/notes/gpu-training-host-light-root-cause.md):
before Phase 10 the device grew trees **host-light** — derivatives on-device but the entire
greedy tree-search inner loop ran serially on the host CPU, making the GPU path **>20×
slower than pure CPU** (an architectural MVP boundary, not a bug). The device-resident grow
loop landed across Phases 10–13 now runs that inner loop on-device, and the aggregate below
shows the original **>20× *slowdown* is demonstrably reversed into a 24–42× *speedup***.

**Correctness pre-flight passed as a blocking gate this session (D-04, pre-flight only).**
The one new Phase-14 Kaggle CUDA run (P100, 2026-07-05) ran the Rust device self-oracle
correctness gate under `--features cuda` as a **BLOCKING pre-flight** before any timing:
all 5 CUDA families passed (44 device self-oracle tests, 0 failed) — `correctness_verdict =
ALL-PASS`. Per D-04 this phase treats device correctness as *already established* by the
Phase-12/13 real-P100 runs and re-confirms it only as a pre-flight; it does **not** formally
flip GPUT-14, and does **not** backfill the `bench/RESULTS.md` depth-1/depth-6 oracle table
(see *Standing debt*, below).

---

## Aggregate speed matrix

Device / host-CPU / speedup / `>=20x?` columns are produced by
[`bench/phase14_cuda_signoff/aggregate.py`](phase14_cuda_signoff/aggregate.py) over the
committed per-phase BENCH-02 JSON (Phase-12 `bench02-result.json` + Phase-13 `result.json`).
The `catboost_gpu_s (informational)` column is merged from the new Phase-14 run
[`bench/phase14_cuda_signoff/bench03-result.json`](phase14_cuda_signoff/bench03-result.json).
Measurement protocol (all rows): warm-run / JIT-excluded / train-only wall-clock, lazy
CubeCL queue drained before the clock stops; depth=6, 20 iterations, 20 features, 32 bins.

| family | n | device_cuda_s | host_cpu_s | speedup_vs_host_cpu | >=20x? | catboost_gpu_s (informational) | source run (GPU, date, session) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| depthwise | 10000 | 0.0828 | 2.5109 | 30.333× | yes | 0.6733 | device+CPU: Phase-12 P100 (2026-07-04); CatBoost-GPU: Phase-14 P100 (2026-07-05) |
| depthwise | 100000 | 0.7463 | 29.8424 | 39.987× | yes | 0.7052 | device+CPU: Phase-12 P100 (2026-07-04); CatBoost-GPU: Phase-14 P100 (2026-07-05) |
| depthwise | 300000 | 2.5678 | 101.9296 | 39.696× | yes | 0.8181 | device+CPU: Phase-12 P100 (2026-07-04); CatBoost-GPU: Phase-14 P100 (2026-07-05) |
| region | 10000 | 0.1014 | 3.1776 | 31.329× | yes | N/A | device+CPU: Phase-12 P100 (2026-07-04); CatBoost-GPU: N/A (no Region policy in official CatBoost) |
| region | 100000 | 0.8749 | 36.8173 | 42.080× | yes | N/A | device+CPU: Phase-12 P100 (2026-07-04); CatBoost-GPU: N/A (no Region policy in official CatBoost) |
| region | 300000 | 2.8719 | 113.3030 | 39.452× | yes | N/A | device+CPU: Phase-12 P100 (2026-07-04); CatBoost-GPU: N/A (no Region policy in official CatBoost) |
| depthwise | 10000 | 0.1080 | 2.6645 | 24.664× | yes | 0.6733 | device+CPU: Phase-13 P100 (2026-07-04); CatBoost-GPU: Phase-14 P100 (2026-07-05) |
| depthwise | 100000 | 0.9167 | 30.3894 | 33.151× | yes | 0.7052 | device+CPU: Phase-13 P100 (2026-07-04); CatBoost-GPU: Phase-14 P100 (2026-07-05) |
| depthwise | 300000 | 2.9717 | 101.5605 | 34.176× | yes | 0.8181 | device+CPU: Phase-13 P100 (2026-07-04); CatBoost-GPU: Phase-14 P100 (2026-07-05) |
| region | 10000 | 0.1310 | 3.1296 | 23.888× | yes | N/A | device+CPU: Phase-13 P100 (2026-07-04); CatBoost-GPU: N/A (no Region policy in official CatBoost) |
| region | 100000 | 0.9867 | 36.1485 | 36.635× | yes | N/A | device+CPU: Phase-13 P100 (2026-07-04); CatBoost-GPU: N/A (no Region policy in official CatBoost) |
| region | 300000 | 3.2888 | 111.6311 | 33.943× | yes | N/A | device+CPU: Phase-13 P100 (2026-07-04); CatBoost-GPU: N/A (no Region policy in official CatBoost) |

**Rows:** 12 — Phase-12 (depthwise + region) and Phase-13 (depthwise + region), each at
n = 10k / 100k / 300k. **All 12 rows `>=20x`.** The Phase-12 and Phase-13 depthwise rows at
the same `n` share the same informational `catboost_gpu_s`, because there is exactly **one**
Phase-14 CatBoost-GPU timing session (D-03) — it is not re-run per prior phase.

### Provenance (D-03 — mixed-session, explicit)

- **device_cuda_s / host_cpu_s** — from the committed per-phase BENCH-02 runs, **not**
  re-measured this session: rows 1–6 are **Phase-12 P100 (2026-07-04)**
  (`bench/phase12_cuda_oracle/bench02-result.json`); rows 7–12 are **Phase-13 P100
  (2026-07-04)** (`bench/phase13_cuda_oracle/result.json`). Both were the same
  `--features cuda` device `GpuBackend` vs host-CPU boosting loop in one binary.
- **catboost_gpu_s** — from the **one new Phase-14 P100 session (2026-07-05)**
  (`bench/phase14_cuda_signoff/bench03-result.json`), official `catboost==1.2.10`
  `task_type='GPU'` on the reproduced `gen()` synthetic workload.
- **Hardware/env of the Phase-14 session:** Tesla P100-PCIE-16GB (driver 580.159.04, 16384
  MiB), `nvcc` release 12.8, `correctness_verdict = ALL-PASS`, `catboost_gpu_verdict = OK`.

---

## Divergence notes (D-01 — informational, NOT a milestone gate)

The `catboost_gpu_s` column is **informational context only**. Per D-01 the sole blocking
BENCH-03 criterion is the `>=20x`-vs-host-CPU-baseline gate above; matching a mature
C++/CUDA library's absolute throughput is an explicit **stretch goal, not the definition of
done**. The head-to-head is recorded transparently, with these documented divergences:

1. **Region CatBoost-GPU = `N/A`.** Official CatBoost has no `Region` grow policy, so there
   is no comparable arm to time — the cell is `N/A`, **never a proxy or forced number**
   (Pitfall 4). Region's device-vs-host-CPU speedup gate still passes on its own committed
   Phase-12/13 numbers.
2. **`border_count` 128 → 32.** CatBoost's GPU default `border_count` is 128; it was set to
   **32** to match the bench's quantization grid (integer-binned 0..31 columns fed as
   float32 with `border_count=32`).
3. **Quantization-cost asymmetry (informational — NOT subtracted).** CatBoost `fit()`
   wall-clock **includes** on-device quantization, while catboost-rs times **only** the grow
   loop. This asymmetry is **documented and left un-adjusted** (D-01): the CatBoost-GPU
   numbers are not corrected downward to "match" what catboost-rs measures, precisely
   because the column is informational rather than a gate.

Because the official-CatBoost-GPU numbers reflect a broader (quantization-inclusive)
wall-clock on a different policy surface, they are **not directly comparable** to the
catboost-rs grow-loop numbers and are **not** used to pass or fail the milestone.

---

## Standing debt — NOT closed here (D-04)

The following are **explicitly out of scope** for this speed-only sign-off, to be resolved
at the milestone-close audit or a dedicated follow-up — flagged here so the debt is surfaced,
not silently assumed closed:

- **GPUT-14 remains `Pending` in `.planning/REQUIREMENTS.md`.** This phase re-confirms device
  correctness only as a **blocking pre-flight** (Part A ALL-PASS this session) and treats it
  as *already established* by the Phase-12/13 real-P100 runs. It does **NOT** formally flip
  GPUT-14 to satisfied — correctness bookkeeping is not this phase's job.
- **The `bench/RESULTS.md` depth-1 / depth-6 `TBD` oracle table is NOT backfilled.** Those
  `TBD` cells stay `TBD` (do-not-fabricate). Phase 14 appends **only one cross-link line** to
  `RESULTS.md` pointing at this sign-off; no existing TBD table is modified, filled, or
  reformatted.

These two items are standing correctness-bookkeeping debt for the milestone-close audit, not
a defect in the BENCH-03 speed sign-off.

---

## Sources

- [`bench/phase14_cuda_signoff/aggregate.py`](phase14_cuda_signoff/aggregate.py) — stitches the committed device/host-CPU/speedup/`>=20x` matrix.
- [`bench/phase14_cuda_signoff/bench03-result.json`](phase14_cuda_signoff/bench03-result.json) — the new Phase-14 human-gated Kaggle CUDA run (Part A correctness pre-flight + Part C informational CatBoost-GPU timings; Region N/A).
- [`bench/phase12_cuda_oracle/bench02-result.json`](phase12_cuda_oracle/bench02-result.json), [`bench/phase13_cuda_oracle/result.json`](phase13_cuda_oracle/result.json) — the aggregated device/host-CPU sources.
- [`.planning/notes/gpu-training-host-light-root-cause.md`](../.planning/notes/gpu-training-host-light-root-cause.md) — the pre-Phase-10 host-light baseline this sign-off reverses.
