# catboost-rs full-parameter GPU speed grid

GPU: Tesla P100-PCIE-16GB, 16384 MiB

official catboost: 1.2.10


`ratio = median(official) / median(catboost_rs)`; **> 1.0 means catboost-rs is faster**. A cell whose min/max ratio spread crosses 1.0 is labelled *within noise* and is NOT claimed as a win.


| cell | device? | official (s) | catboost-rs (s) | ratio | spread | verdict |
|---|---|---|---|---|---|---|
| SymmetricTree|RMSE|unw|noctr|300k | True | 1.244 | 0.772 | 1.61x | 1.55–1.95 | catboost-rs faster |
| SymmetricTree|RMSE|unw|ctr|300k | False | N/A | N/A | N/A | N/A | N/A — CatBoostError("'data' is numpy array of floating point numerical type, it means  |
| SymmetricTree|RMSE|w|noctr|300k | False | N/A | N/A | N/A | N/A | N/A — TypeError("CatBoostRegressor.fit() got an unexpected keyword argument 'sample_we |
| SymmetricTree|RMSE|w|ctr|300k | False | N/A | N/A | N/A | N/A | N/A — CatBoostError("'data' is numpy array of floating point numerical type, it means  |
| SymmetricTree|Logloss|unw|noctr|300k | True | 1.438 | 0.745 | 1.93x | 1.77–1.98 | catboost-rs faster |
| SymmetricTree|Logloss|unw|ctr|300k | False | N/A | N/A | N/A | N/A | N/A — CatBoostError("'data' is numpy array of floating point numerical type, it means  |
| SymmetricTree|Logloss|w|noctr|300k | False | N/A | N/A | N/A | N/A | N/A — TypeError("CatBoostClassifier.fit() got an unexpected keyword argument 'sample_w |
| SymmetricTree|Logloss|w|ctr|300k | False | N/A | N/A | N/A | N/A | N/A — CatBoostError("'data' is numpy array of floating point numerical type, it means  |
| Depthwise|RMSE|unw|noctr|300k | False | N/A | N/A | N/A | N/A | N/A — CatBoostError('degenerate training input: pointwise_hist2 one-byte non-binary fi |
| Depthwise|RMSE|unw|ctr|300k | False | N/A | N/A | N/A | N/A | N/A — CatBoostError("'data' is numpy array of floating point numerical type, it means  |
| Depthwise|RMSE|w|noctr|300k | False | N/A | N/A | N/A | N/A | N/A — TypeError("CatBoostRegressor.fit() got an unexpected keyword argument 'sample_we |
| Depthwise|RMSE|w|ctr|300k | False | N/A | N/A | N/A | N/A | N/A — CatBoostError("'data' is numpy array of floating point numerical type, it means  |
| Depthwise|Logloss|unw|noctr|300k | False | N/A | N/A | N/A | N/A | N/A — CatBoostError('degenerate training input: pointwise_hist2 one-byte non-binary fi |
| Depthwise|Logloss|unw|ctr|300k | False | N/A | N/A | N/A | N/A | N/A — CatBoostError("'data' is numpy array of floating point numerical type, it means  |
| Depthwise|Logloss|w|noctr|300k | False | N/A | N/A | N/A | N/A | N/A — TypeError("CatBoostClassifier.fit() got an unexpected keyword argument 'sample_w |
| Depthwise|Logloss|w|ctr|300k | False | N/A | N/A | N/A | N/A | N/A — CatBoostError("'data' is numpy array of floating point numerical type, it means  |
| SymmetricTree|RMSE|unw|noctr|1000k | True | 1.666 | 1.827 | 0.91x | 0.88–0.95 | official faster |
| SymmetricTree|RMSE|unw|ctr|1000k | False | N/A | N/A | N/A | N/A | N/A — CatBoostError("'data' is numpy array of floating point numerical type, it means  |
| SymmetricTree|RMSE|w|noctr|1000k | False | N/A | N/A | N/A | N/A | N/A — TypeError("CatBoostRegressor.fit() got an unexpected keyword argument 'sample_we |
| SymmetricTree|RMSE|w|ctr|1000k | False | N/A | N/A | N/A | N/A | N/A — CatBoostError("'data' is numpy array of floating point numerical type, it means  |
| SymmetricTree|Logloss|unw|noctr|1000k | True | 1.896 | 1.836 | 1.03x | 1.01–1.07 | catboost-rs faster |
| SymmetricTree|Logloss|unw|ctr|1000k | False | N/A | N/A | N/A | N/A | N/A — CatBoostError("'data' is numpy array of floating point numerical type, it means  |
| SymmetricTree|Logloss|w|noctr|1000k | False | N/A | N/A | N/A | N/A | N/A — TypeError("CatBoostClassifier.fit() got an unexpected keyword argument 'sample_w |
| SymmetricTree|Logloss|w|ctr|1000k | False | N/A | N/A | N/A | N/A | N/A — CatBoostError("'data' is numpy array of floating point numerical type, it means  |
| Depthwise|RMSE|unw|noctr|1000k | False | N/A | N/A | N/A | N/A | N/A — CatBoostError('degenerate training input: pointwise_hist2 one-byte non-binary fi |
| Depthwise|RMSE|unw|ctr|1000k | False | N/A | N/A | N/A | N/A | N/A — CatBoostError("'data' is numpy array of floating point numerical type, it means  |
| Depthwise|RMSE|w|noctr|1000k | False | N/A | N/A | N/A | N/A | N/A — TypeError("CatBoostRegressor.fit() got an unexpected keyword argument 'sample_we |
| Depthwise|RMSE|w|ctr|1000k | False | N/A | N/A | N/A | N/A | N/A — CatBoostError("'data' is numpy array of floating point numerical type, it means  |
| Depthwise|Logloss|unw|noctr|1000k | False | N/A | N/A | N/A | N/A | N/A — CatBoostError('degenerate training input: pointwise_hist2 one-byte non-binary fi |
| Depthwise|Logloss|unw|ctr|1000k | False | N/A | N/A | N/A | N/A | N/A — CatBoostError("'data' is numpy array of floating point numerical type, it means  |
| Depthwise|Logloss|w|noctr|1000k | False | N/A | N/A | N/A | N/A | N/A — TypeError("CatBoostClassifier.fit() got an unexpected keyword argument 'sample_w |
| Depthwise|Logloss|w|ctr|1000k | False | N/A | N/A | N/A | N/A | N/A — CatBoostError("'data' is numpy array of floating point numerical type, it means  |
| SHOWCASE-bias|RMSE|unw|noctr|300k | True | 1.225 | 0.695 | 1.76x | 1.63–1.78 | catboost-rs faster |
| SHOWCASE-sampled-nonsym|RMSE|unw|noctr|300k | False | N/A | N/A | N/A | N/A | N/A — CatBoostError('degenerate training input: pointwise_hist2 one-byte non-binary fi |

## Caveats

- Device activation is OBSERVED per cell via `CB_GPU_PROF` tree lines, not assumed. A cell without them is a CPU row and is excluded from any claim.
- Both sides receive the SAME explicit recipe; official CatBoost's GPU default `bootstrap_type=Bayesian` would otherwise do strictly more work per tree and inflate the ratio.
- Both shapes are at or above the D-10-09 crossover (n = 100_000). Below it the device cannot win — launch-overhead physics, not a tuning gap.
- Combination-CTR cells are deliberately ABSENT: FPP-11 is escalated and combination projections are device-ineligible, so such a cell would silently measure a CPU fit.
- The headline holds for the axes measured here ONLY; it is not a claim about CatBoost GPU in general.
