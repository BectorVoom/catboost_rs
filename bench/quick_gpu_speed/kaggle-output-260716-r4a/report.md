# catboost-rs quick GPU-vs-GPU speed check (informal)

GPU: Tesla P100-PCIE-16GB, 580.159.04, 16384 MiB  
driver: 580.159.04  nvcc: release 12.8  
build_ok: True  import_ok: True  catboost: 1.2.10

## Wall-clock (300k x 50, depth 6, 30 iters)

| arm | seconds |
|---|---|
| catboost_rs RMSE | 1.2118 |
| catboost_rs Logloss | 1.1107 |
| official CatBoost GPU RMSE | 1.2423 |
| official CatBoost GPU Logloss | 1.2676 |
| sklearn HGB (CPU) RMSE | 2.1776 |
| sklearn HGB (CPU) Logloss | 2.7635 |
| sklearn HGB under cuml.accel RMSE | 2.2296 |
| XGBoost GPU hist RMSE | 1.0230 |
| XGBoost GPU hist Logloss | 0.9766 |

## Speedup (competitor / catboost_rs; >1 => catboost_rs faster)

| competitor | RMSE | Logloss |
|---|---|---|
| official CatBoost GPU | 1.0252x | 1.1413x |
| sklearn HGB (CPU) | 1.7970x | 2.4881x |
| sklearn HGB under cuml.accel | 1.8399x | N/A |
| XGBoost GPU hist | 0.8442x | 0.8793x |

## Train-set quality (comparability check across tree shapes)

| arm | metric | value |
|---|---|---|
| catboost_official_gpu_logloss | train_logloss | 0.488568 |
| catboost_official_gpu_rmse | train_rmse | 4.309055 |
| catboost_rs_logloss | train_logloss | 0.607269 |
| catboost_rs_rmse | train_rmse | 4.307126 |
| sklearn_hgb_logloss | train_logloss | 0.503607 |
| sklearn_hgb_rmse | train_rmse | 4.411695 |
| xgboost_gpu_hist_logloss | train_logloss | 0.486318 |
| xgboost_gpu_hist_rmse | train_rmse | 4.246603 |

## cuML note (checked against docs.rapids.ai, 2026-07-16)

cuml.accel does NOT accelerate sklearn's HistGradientBoosting* (its sklearn.ensemble coverage is RandomForest only), so 'cuML histogram gradient boosting' executes sklearn's CPU implementation. The 'sklearn HGB under cuml.accel' arm above measures exactly that (cuml present: '26.02.000').

## Stage attribution (CB_GPU_PROF fit — fenced, NOT the timed number)

| stage | total ms |
|---|---|
| derive | 82.17 |
| elapsed | 2091.28 |
| fill | 485.61 |
| leaf_apply_der | 59.7 |
| score | 169.89 |
| split | 57.54 |
| stats_read | 144.89 |

## Device-activation caveat (always included)

Device activation is NOT directly instrumented or observable from the Python surface in this informal check: catboost_rs exposes no log line or public attribute indicating whether the GPU tree-growth loop actually ran for a given .fit(). This bench satisfies every known device_host_eligible precondition BY CONSTRUCTION (see the preconditions map above), but that is a static/documented audit, not a runtime proof. A silent CPU fallback therefore cannot be 100% ruled out from the Python surface alone. If a catboost_rs timing lands in the same ballpark as a known host-CPU reference rather than a device-fast number, treat it as a possible silent CPU fallback.
