# Colab T4 — parameter-wireup wave GPU speed

**Date:** 2026-08-15 · **GPU:** Tesla T4, 15360 MiB · **Host:** 2 vCPU Intel Xeon @ 2.00 GHz,
12 GB · **official catboost:** 1.2.10 · **tree under test:** commit `15a1dd3`, uploaded as a
`git archive` tarball (not a clone — the branch is local-only, which is why `result.json`
records `commit: unknown`; the archive was cut from `15a1dd3`)

**Recipe:** 300_000 × 50, 30 iterations, depth 6, `border_count=128`, RMSE, SymmetricTree,
`bootstrap_type=No`, `random_strength=0`, `boost_from_average=False`, Gradient leaves, L2
score — identical on both sides. Median of 3 repeats. Device activation OBSERVED per cell
via `CB_GPU_PROF` tree lines, asserted in both directions.

---

## Headline: the device-decline cliff is ~37×

| cell | device? | catboost-rs | vs baseline | official CatBoost GPU |
|---|---|---|---|---|
| `baseline` | yes | **1.10 s** | 1.00× | 1.94 s |
| `model_shrink_rate=0.2` | **NO** | **40.70 s** | **36.97×** | *rejected — see below* |
| `leaf_estimation_iterations=3` | **NO** | **40.64 s** | **36.92×** | 2.08 s |

Setting either parameter turns a 1.1-second GPU fit into a 41-second CPU fit. Both declines
are correct — the alternative is a silently wrong model, which is what
`string_param_device_routing_test` exists to prevent — but the cost is now on the record
instead of being a surprise.

### Bound the 37× before quoting it

This host has **2 vCPUs**. The ratio is a product of two things: a fast device baseline AND
a weak CPU fallback. On a many-core host the fallback is much cheaper and the ratio much
smaller — the same measurement on the local ROCm rig (16-core host, integrated gfx1151 GPU)
gave a central estimate of only **1.26–1.40×**, and could not even separate it from noise.

So the honest range is **"between ~1.3× and ~37×, depending on how fast your GPU is
relative to your CPU"**. The T4 figure is the one a Colab or single-GPU-cloud-instance user
actually experiences; the ROCm figure is what a workstation with a weak GPU and a strong CPU
sees. Neither is *the* number.

### The two decline cells are NOT the same story

- **`model_shrink_rate`** — official CatBoost **refuses it on GPU outright**:
  > `catboost/private/libs/options/json_helper.h:185: Error: change of option
  > model_shrink_rate is unimplemented for task type GPU and was not default in previous run`

  So upstream cannot do this on a GPU at all. catboost-rs is strictly **more capable** here:
  it trains a correct model, just on the CPU path. Declining is the right call and there is
  no gap to close.

  This is exactly why `GPU_UNSUPPORTED_BORDER_TYPES` was left EMPTY and rejections are
  discovered on the box rather than declared in advance — a guessed `N/A` would have hidden
  a real capability difference behind an assumption.

- **`leaf_estimation_iterations`** — official handles it **on GPU in 2.08 s**. This is a
  genuine capability gap on our side: the multi-step accumulate-and-recompute leaf estimator
  is CPU-only here, so we pay 40.64 s for what upstream does on-device in 2.08 s. **Recorded
  as future work**, not dressed up: the correct fix is a device-side multi-step estimator,
  after which the decline clause in `device_host_eligible` can be dropped.

---

## B. Border-build cost

Every border type stayed device-eligible — quantization is a host-side border choice with no
eligibility clause of its own, which is what `every_border_type_matches_cpu_on_device`
asserts.

| `feature_border_type` | catboost-rs | vs `GreedyLogSum` | official GPU | absolute DP delta |
|---|---|---|---|---|
| `GreedyLogSum` | 1.09 s | 1.00× | 2.04 s | — |
| `Uniform` | 1.29 s | 1.18× | 1.60 s | — |
| `UniformAndQuantiles` | 1.44 s | 1.32× | 1.85 s | — |
| `Median` | 1.45 s | 1.33× | 1.97 s | — |
| `GreedyMinEntropy` | 1.47 s | 1.35× | 1.94 s | — |
| **`MaxLogSum`** | **53.86 s** | **49.38×** | 49.98 s | cb-rs 52.77 s / official 47.94 s |
| **`MinEntropy`** | **57.42 s** | **52.64×** | 51.65 s | cb-rs 56.33 s / official 49.61 s |

The exact-DP types dominate everything else on a 2-vCPU host — 57 s against a 1.1 s
baseline. Compare the **absolute deltas**, not the ratios: we pay 56.3 s where official pays
49.6 s, so we are ~14 % slower at exact-DP border building on this host. On the 16-core local
box the same comparison was 3.83 s against 3.89 s — within 2 %. Both engines' DP is
essentially serial, so a 2-vCPU host magnifies it ~14× and exposes a modest constant-factor
difference that 16 cores hide.

Two practical consequences:

1. `border_count=128` here (vs 254 in the CPU study) and the cost is still ~50 s on a weak
   host. The exact-DP types are a poor default on a small cloud instance.
2. It is one-time PREP — see `../FINDINGS-cpu-border-cost.md`, where the delta is shown flat
   across a 16× change in iteration count. A 30-iteration fit is the worst case for this
   ratio; a 1000-iteration fit amortizes it away.

---

## C. `nan_mode` control — free, as designed

| cell | device? | catboost-rs | official GPU |
|---|---|---|---|
| `nan_mode=Min` | yes | 1.16 s | 1.88 s |
| `nan_mode=Max` | yes | **1.17 s** | 1.93 s |

`Max` costs 0.01 s over `Min` — within noise. This is the control that mattered most: `Max`
adds a per-object sentinel branch to the quantizer on BOTH the host and the QPACK-01 device
kernel (the fix in commit `54862c1`), and a per-object change assumed to be free is exactly
how a regression ships unnoticed. It is free. Measured, not assumed.

---

## Incidental: catboost-rs beats official CatBoost GPU on every device-eligible cell

| cell | catboost-rs | official | speedup |
|---|---|---|---|
| `baseline` | 1.10 s | 1.94 s | **1.76×** |
| `nan_mode=Min` | 1.16 s | 1.88 s | 1.62× |
| `nan_mode=Max` | 1.17 s | 1.93 s | 1.65× |
| `GreedyLogSum` | 1.09 s | 2.04 s | 1.87× |
| `Uniform` | 1.29 s | 1.60 s | 1.24× |

Not this wave's claim and not tuned for — it falls out of the pre-existing device work
(see `bench/RESULTS.md`) — but recorded because the grid measured it on a matched recipe with
device activation observed.
