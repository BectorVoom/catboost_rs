# catboost-rs quick GPU-vs-GPU speed check (informal)

GPU: Tesla P100-PCIE-16GB, 580.159.04, 16384 MiB  
driver: 580.159.04  nvcc: release 12.8  
build_ok: True  import_ok: True  catboost: 1.2.10

## Wall-clock (300k x 50, depth 6, 30 iters)

| arm | seconds |
|---|---|
| catboost_rs RMSE | 1.4425 |
| catboost_rs Logloss | 1.4945 |
| official CatBoost GPU RMSE | 1.4252 |
| official CatBoost GPU Logloss | 1.4080 |
| sklearn HGB (CPU) RMSE | 3.3102 |
| sklearn HGB (CPU) Logloss | 3.8655 |
| sklearn HGB under cuml.accel RMSE | 3.2231 |
| XGBoost GPU hist RMSE | 1.2094 |
| XGBoost GPU hist Logloss | 1.2411 |

## Speedup (competitor / catboost_rs; >1 => catboost_rs faster)

| competitor | RMSE | Logloss |
|---|---|---|
| official CatBoost GPU | 0.9880x | 0.9421x |
| sklearn HGB (CPU) | 2.2948x | 2.5865x |
| sklearn HGB under cuml.accel | 2.2344x | N/A |
| XGBoost GPU hist | 0.8384x | 0.8304x |

## Train-set quality (comparability check across tree shapes)

| arm | metric | value |
|---|---|---|
| catboost_official_gpu_logloss | train_logloss | 0.488567 |
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
| derive | 101.86 |
| elapsed | 2889.02 |
| fill | 838.14 |
| lds | 1.18 |
| leaf_apply_der | 63.61 |
| multicopy | 1.69 |
| score | 210.5 |
| split | 72.6 |
| stats_read | 180.15 |

## Device-activation caveat (always included)

Device activation is NOT directly instrumented or observable from the Python surface in this informal check: catboost_rs exposes no log line or public attribute indicating whether the GPU tree-growth loop actually ran for a given .fit(). This bench satisfies every known device_host_eligible precondition BY CONSTRUCTION (see the preconditions map above), but that is a static/documented audit, not a runtime proof. A silent CPU fallback therefore cannot be 100% ruled out from the Python surface alone. If a catboost_rs timing lands in the same ballpark as a known host-CPU reference rather than a device-fast number, treat it as a possible silent CPU fallback.
