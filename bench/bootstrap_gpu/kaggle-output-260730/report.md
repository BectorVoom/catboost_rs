# catboost-rs — bootstrap_type oracle + learning speed (CUDA GPU)

- GPU: `Tesla P100-PCIE-16GB, 580.159.04, 16384 MiB`
- Commit: `5a5068a4d6dff9eff43126a08a1ab8dfbfa20a8a` (`fix/bootstrap-rng-draw-accounting`)
- RNG-fix marker present: **True**
- Verdict: **ORACLE-PASS**

## Part A — oracle

| suite | rc | passed | failed | blocking |
|---|---|---|---|---|
| cb-train bootstrap parity (CPU, all 4 types) | 0 | 5 | 0 | True |
| cb-train regularization parity (CPU) | 101 | 5 | 1 | False |
| cb-backend device bootstrap kernels (CUDA) | 0 | 7 | 0 | False |
| cb-backend device MVS kernels (CUDA) | 0 | 3 | 0 | False |

## Part B — learning speed by bootstrap_type (300000x50, depth 6, 30 iters, RMSE)

| bootstrap_type | rs on GPU? | catboost_rs (s) | CatBoost GPU (s) | CatBoost CPU (s) | rs/cbGPU | rs train RMSE |
|---|---|---|---|---|---|---|
| No | True | 1.9283 | 1.3105 | 1.8603 | 1.471 | 4.307126 |
| Bayesian | False | 16.5562 | 1.2762 | 1.9301 | 12.973 | 4.309573 |
| Bernoulli | False | 16.2594 | 1.3242 | 1.7332 | 12.279 | 4.309293 |
| MVS | False | 16.7216 | 1.3818 | 1.7759 | 12.101 | 4.306912 |
| Poisson | False | None | 1.3267 | None | None | None |

## Caveats (never dropped)

- **only_No_is_gpu_eligible**: device_host_eligible (crates/cb-train/src/boosting.rs) hard-requires bootstrap_type == No AND random_strength == 0.0. Bayesian/Bernoulli/MVS/Poisson rows are CPU-grower numbers measured on a GPU machine; the GPU is idle for them.
- **device_activation_not_observable**: No Python-visible signal proves the device grow path ran for a given fit; a silent CPU fallback cannot be ruled out from this surface.
