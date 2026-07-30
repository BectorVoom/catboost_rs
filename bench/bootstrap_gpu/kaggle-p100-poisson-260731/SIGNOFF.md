# GPU-only Poisson bootstrap — Kaggle P100 sign-off

- **GPU:** Tesla P100-PCIE-16GB, driver 580.159.04, 16 GiB · **CUDA:** 12.8 · **rustc:** 1.97.1
- **Date:** 2026-07-31 · **Kernel:** `boomvector/catboost-rs-poisson-p100` (v2)
- **Accelerator pinned** via `machine_shape: "NvidiaTeslaP100"` — Kaggle assigned exactly that.
- **Verdict:** **ORACLE-PASS** — 7/7 suites green, 37 tests passed, 0 failed.
- Raw artifacts: `report.md`, `result.json`, `run.log` (all emitted by the run itself).

## What Poisson is, and why its oracle is unusual

`bootstrap_type=Poisson` is upstream CatBoost's **GPU-only** sampler. Its CPU validator
rejects it outright, which this run confirmed empirically rather than by reading the
source — official CatBoost with `task_type="CPU", bootstrap_type="Poisson"` raised:

```
catboost/private/libs/options/bootstrap_options.cpp:29:
Error: poisson bootstrap is not supported on CPU
```

So there is no CPU sampler to mirror, and the per-object weights upstream's CUDA kernel
draws are not exposed by any public API. Neither task type can produce a reference.
Upstream's CUDA kernel **is** the specification.

The gate is therefore bit-for-bit agreement against
`cb-oracle/generator/poisson_bootstrap_oracle.cpp` — a verbatim host transcription of
upstream's `PoissonBootstrapImpl` + `random_gen.cuh`, compiled by `g++` and frozen into
`cb-oracle/fixtures/bootstrap_poisson/`. Different program, language, compiler and
processor from the `#[cube]` kernel under test, so agreement is evidence rather than a
tautology.

## Oracle

| suite | passed | failed | blocking |
|---|---|---|---|
| Poisson upstream oracle (CUDA, bit-for-bit) | 8 | 0 | yes |
| Poisson device e2e (CUDA) | 6 | 0 | yes |
| bias-0 device vs UPSTREAM CatBoost (CUDA) | 1 | 0 | yes |
| WR-01 device bootstrap parity (CUDA) | 4 | 0 | yes |
| device bootstrap kernels (CUDA) | 16 | 0 | no |
| Poisson parallel-draw speed (CUDA) | 1 | 0 | no |
| sampled fits run at device speed (CUDA) | 1 | 0 | no |

Bit-for-bit vs upstream over three launch geometries × two consecutive draws (the second
draw gates the in-place seed carry-over that makes consecutive trees continue their
per-thread streams):

| scenario | n | seeds | λ | stride | result |
|---|---|---|---|---|---|
| `one_pass`  |  1000 | 65536 | 1.078810 |  1024 | bit-for-bit |
| `grid_wrap` |  4096 |  1024 | 1.609438 |  1024 | bit-for-bit |
| `wide`      | 20000 | 65536 | 1.609438 | 20224 | bit-for-bit |

**Regression control.** The Poisson work refactored the shared session sampler, so the
other three arms were re-gated against upstream CatBoost 1.2.10: `no`, `bayesian`,
`bernoulli` and `mvs` each hold splits + leaf values + staged approximants within 1e-5
over **all 3 trees**.

## Speed — 300000 × 50, depth 6, 30 iters, RMSE

Every row proven device-resident (`CB_GPU_PROF` per-tree lines), not assumed.

| bootstrap | catboost_rs | CatBoost GPU | CatBoost CPU | rs / CB-GPU |
|---|---|---|---|---|
| **Poisson**   | **1.085 s** | 1.257 s | *rejected* | **0.863× (16% faster)** |
| No        | 1.085 s | 1.200 s | 1.855 s | 0.905× |
| Bernoulli | 1.253 s | 1.221 s | 1.889 s | 1.026× |
| Bayesian  | 1.460 s | 1.254 s | 1.953 s | 1.164× |
| MVS       | 1.705 s | 1.250 s | 1.765 s | 1.364× |

Poisson is the **fastest arm and the largest win over upstream**, and that is structural
rather than lucky: it is the only sampler drawn entirely on device. The other three
mirror upstream's *CPU* samplers, whose randomness is a single sequential `TFastRng64`
stream — reproducing it bit-for-bit forbids reordering, so they pay a per-tree host
sample. Poisson has no CPU semantics to preserve, so faithfulness and speed coincide.

The CatBoost CPU cell for Poisson is empty because upstream refuses it there. That is the
correct result, and catboost_rs mirrors the same asymmetry.

Sampling costs essentially nothing at this shape (`ratio_vs_No = 0.99×` on the in-repo
speed suite, 60000 × 24): the draw is device-resident with no host round-trip, and no
read-back.

## Kernel-level

| measurement | P100 | local ROCm gfx1151 |
|---|---|---|
| parallel Poisson draw vs serial stream draw, n=2M | **12.3×** | 10.2× |

## Anti-false-pass checks that passed

- Sampled vs unsampled model differs (`max|Δpred| = 2.303e-2`) — the weights reach the
  split histogram rather than being drawn and dropped.
- `subsample` 0.5 vs 0.9 differ (`3.266e-2`) — λ is a live parameter, not a constant.
- `random_seed` 1 vs 2 differ (`2.277e-2`) — the seed buffer really derives from it.
- Run-to-run `max|Δpred| = 0` — bit-identical across fits.
- Model still learns: RMSE 0.1602 vs constant-predictor 0.6963.
- `subsample >= 1` rejected: upstream's `GetPoissonLambda()` returns −1 there, which
  would zero every weight.
- The host passes an **empty** sample for Poisson, so double-sampling is structurally
  impossible.

## Reproducing

```bash
# 1. source payload -> private Kaggle dataset
tar -czf catboost_rs_src.tar.gz --exclude=target --exclude=.git --exclude=.venv-py8 \
    crates bench scripts Cargo.toml Cargo.lock rust-toolchain.toml CLAUDE.md
kaggle datasets create -p .            # dataset-metadata.json -> boomvector/catboost-rs-poisson-src

# 2. kernel, accelerator pinned
kaggle kernels push -p . --accelerator NvidiaTeslaP100
```

**Gotcha worth keeping:** Kaggle mounts datasets at
`/kaggle/input/datasets/<owner>/<slug>/`, **not** `/kaggle/input/<slug>/`, and it
auto-decompresses uploaded archives. `poisson_bench_kaggle.py` searches recursively for a
directory containing `crates/` + `Cargo.toml` and handles both shapes; run v1 of this
kernel failed on exactly this (cleanly, at the source guard, before any timing).
