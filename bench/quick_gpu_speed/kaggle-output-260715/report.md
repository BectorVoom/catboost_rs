# catboost-rs quick GPU-vs-GPU speed check (informal)

GPU: Tesla P100-PCIE-16GB, 580.159.04, 16384 MiB  
driver: 580.159.04  nvcc: release 12.8  
build_ok: True  import_ok: True  catboost: 1.2.10

## Wall-clock (300k x 50, depth 6, 30 iters)

| arm | seconds |
|---|---|
| catboost_rs RMSE | 1.5362 |
| catboost_rs Logloss | 1.6101 |
| official CatBoost GPU RMSE | 1.2667 |
| official CatBoost GPU Logloss | 1.3163 |

## Speedup (official / catboost_rs; >1 => catboost_rs faster)

| loss | official_over_rs |
|---|---|
| RMSE | 0.8246x |
| Logloss | 0.8175x |

## Stage attribution (CB_GPU_PROF fit — fenced, NOT the timed number)

| stage | total ms |
|---|---|
| derive | 79.55 |
| elapsed | 2873.12 |
| fill | 591.63 |
| leaf_apply_der | 54.21 |
| score | 177.73 |
| split | 189.74 |
| stats_read | 153.28 |

## Device-activation caveat (always included)

Device activation is NOT directly instrumented or observable from the Python surface in this informal check: catboost_rs exposes no log line or public attribute indicating whether the GPU tree-growth loop actually ran for a given .fit(). This bench satisfies every known device_host_eligible precondition BY CONSTRUCTION (see the preconditions map above), but that is a static/documented audit, not a runtime proof. A silent CPU fallback therefore cannot be 100% ruled out from the Python surface alone. If a catboost_rs timing lands in the same ballpark as a known host-CPU reference rather than a device-fast number, treat it as a possible silent CPU fallback.
