# catboost-rs param-wave GPU speed

- GPU: `Tesla T4, 15360 MiB`
- commit: `unknown`
- official catboost: `1.2.10`
- shape: 300000 x 50, 30 iterations, depth 6, border_count 128, 3 repeats

## A. The device-decline cliff

`model_shrink_rate != 0` and `leaf_estimation_iterations > 1` route the fit to the CPU grower. Both look like ordinary knobs; this is what setting one costs on a GPU box.

| cell | device? | catboost-rs median | vs baseline | official GPU median |
|---|---|---|---|---|
| `baseline` | yes | 1.10s | 1.00x | 1.94s |
| `model_shrink_rate=0.2` | NO (CPU grower) | 40.70s | 36.97x | CatBoostError: catboost/private/libs/options/json_helper.h:185: Error: change of option model_shrink_rate is unimplemented for task type GPU and was not default in previous run |
| `leaf_estimation_iterations=3` | NO (CPU grower) | 40.64s | 36.92x | 2.08s |

## B. Border-build cost

| feature_border_type | device? | catboost-rs median | vs GreedyLogSum | official GPU median |
|---|---|---|---|---|
| `Median` | yes | 1.45s | 1.33x | 1.97s |
| `GreedyLogSum` | yes | 1.09s | 1.00x | 2.04s |
| `UniformAndQuantiles` | yes | 1.44s | 1.32x | 1.85s |
| `MinEntropy` | yes | 57.42s | 52.64x | 51.65s |
| `MaxLogSum` | yes | 53.86s | 49.38x | 49.98s |
| `Uniform` | yes | 1.29s | 1.18x | 1.60s |
| `GreedyMinEntropy` | yes | 1.47s | 1.35x | 1.94s |

## C. nan_mode (control)

Expected to be free. Measured because it is expected to be free: `Max` adds a per-object sentinel branch to the quantizer on BOTH the host and the device kernel, and an assumed-free per-object change is exactly how a regression ships unnoticed.

| cell | device? | catboost-rs median | official GPU median |
|---|---|---|---|
| `nan_mode=Min` | yes | 1.16s | 1.88s |
| `nan_mode=Max` | yes | 1.17s | 1.93s |

## Disciplines

- Device activation is OBSERVED per cell via `CB_GPU_PROF` tree lines, in both directions: a cell expected to commit that shows none, AND a cell expected to decline that shows some, are both harness failures.
- Both sides get the same explicit recipe; a recipe official CatBoost GPU cannot express is `N/A` with the reason, never swapped for another.
- Median/min/max over repeats; a ratio range spanning 1.0 is *within noise*.
- A failed build or cell yields an error row, never an invented number.
