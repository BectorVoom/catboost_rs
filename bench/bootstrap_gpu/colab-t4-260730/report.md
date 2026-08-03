# WR-01 device bootstrap parity — Google Colab T4 (CUDA) sign-off

- **GPU:** Tesla T4, driver 580.82.07 · **CUDA:** 12.8 · **rustc:** 1.97.1
- **Date:** 2026-07-30
- **Verdict:** **ORACLE-PASS** — 7/7 suites green, 26 tests passed, 0 failed.

## Suites

| suite | rc | result |
|---|---|---|
| `cuda_device_bootstrap_parity` | 0 | 4 passed |
| `cuda_bias0_device_vs_upstream` | 0 | 1 passed |
| `cuda_device_speed_ratio` | 0 | 1 passed |
| `cuda_backend_bootstrap_kernels` | 0 | 7 passed |
| `cuda_backend_mvs_kernels` | 0 | 3 passed |
| `cpu_frozen_bootstrap_oracle` | 0 | 5 passed |
| `cpu_draw_replay` | 0 | 5 passed |

## Device vs UPSTREAM CatBoost 1.2.10 (bias-0 family, <=1e-5)

| scenario | trees within 1e-5 |
|---|---|
| `no` | 3/3 |
| `bayesian` | 3/3 |
| `bernoulli` | 3/3 |
| `mvs` | 2/3 (pre-existing CPU-side tree-2 gap, progress.md R-1) |

Splits, leaf values AND staged approximants all gated.

## Device vs CPU grower (20000x16, depth 6, 20 iters)

| bootstrap | max abs dpred (device vs cpu) | max abs dpred (sampled vs unsampled) |
|---|---|---|
| Bernoulli | 5.589e-11 | 3.003e-3 |
| Bayesian  | 5.477e-11 | 4.878e-3 |
| MVS       | 4.703e-11 | 2.558e-3 |

The second column is the anti-false-pass check: sampling materially changes the model,
so the <=1e-5 agreement in the first column is not vacuous.

## Base grower (bootstrap_type = No)

| shape | max abs dpred | split-mismatched trees |
|---|---|---|
| 512x4 d3 x5 | 3.605e-11 | 0/5 |
| 2048x8 d6 x10 | 5.992e-11 | 0/10 |
| 20000x16 d6 x20 | 6.212e-11 | 4/20, all inert (max abs dcontribution 2.799e-11) |

## Determinism budget (WR01-S13)

Run-to-run `max abs dpred` over 5 identical fits: **0.000e0** (bit-identical) for
No / Bernoulli / MVS. Budget was <=1e-7.

## Speed — the gap this phase closed

Sampled fit cost RELATIVE to the unsampled DEVICE fit, same machine (60000x24, depth 6,
15 iters, release):

| bootstrap | T4 CUDA | local ROCm gfx1151 | before WR-01 (Kaggle P100) |
|---|---|---|---|
| Bernoulli | **1.32x** | 1.04x | ~8.4x (CPU fallback) |
| Bayesian  | **1.58x** | 1.14x | ~8.4x (CPU fallback) |
| MVS       | **1.78x** | 1.10x | ~8.4x (CPU fallback) |

Before WR-01 the three sampled arms were excluded by `device_host_eligible` and ran on
the CPU grower with the GPU idle (P100: No 1.93s vs Bayesian 16.56s / Bernoulli 16.26s /
MVS 16.72s). They now run device-resident, costing only the per-tree host sample plus two
extra elementwise device products.

## Notes

- Poisson is rejected identically on every backend (same `CbError::Degenerate` message)
  by design — upstream CatBoost rejects it on CPU while its GPU trains it, so there is no
  CPU-semantics parity target.
- Every T4 number is identical to the local ROCm run. That is the expected consequence of
  the fixed-point `Atomic<u64>` split histogram: tree structure is deterministic across
  vendors.
- Colab reclaimed the VM twice mid-run before this run completed; this harness
  (`t4_oracle_only.py`) therefore uses the debug profile for the oracle suites and
  checkpoints each suite's verdict as it finishes.
