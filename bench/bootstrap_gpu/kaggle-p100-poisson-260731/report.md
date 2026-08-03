# GPU-only Poisson bootstrap — oracle + speed (Kaggle CUDA)

- GPU: `Tesla P100-PCIE-16GB, 580.159.04, 16384 MiB` (P100 requested: True)
- verdict: **ORACLE-PASS**
- catboost: 1.2.10
- toolchain: `rustc 1.97.1 (8bab26f4f 2026-07-14)
cargo 1.97.1 (c980f4866 2026-06-30)
Cuda compilation tools, release 12.8, V12.8.93
Build cuda_12.8.r12.8/compiler.35583870_0`

## Oracle

| suite | passed | failed | blocking |
|---|---|---|---|
| Poisson upstream oracle (CUDA, bit-for-bit) | 8 | 0 | True |
| Poisson device e2e (CUDA) | 6 | 0 | True |
| bias-0 device vs UPSTREAM CatBoost (CUDA) | 1 | 0 | True |
| WR-01 device bootstrap parity (CUDA) | 4 | 0 | True |
| device bootstrap kernels (CUDA) | 16 | 0 | False |
| Poisson parallel-draw speed (CUDA) | 1 | 0 | False |
| sampled fits run at device speed (CUDA) | 1 | 0 | False |

### Measured evidence

```
test kernels::poisson_bootstrap_oracle_test::poisson_grid_wrap_matches_upstream_bit_for_bit ... [poisson oracle grid_wrap] n=4096 seeds=1024 lambda=1.609438 stride=1024 rounds=2 mean=1.5806 — bit-for-bit vs upstream
test kernels::poisson_bootstrap_oracle_test::poisson_one_pass_matches_upstream_bit_for_bit ... [poisson oracle one_pass] n=1000 seeds=65536 lambda=1.078810 stride=1024 rounds=2 mean=1.0740 — bit-for-bit vs upstream
test kernels::poisson_bootstrap_oracle_test::poisson_wide_matches_upstream_bit_for_bit ... [poisson oracle wide] n=20000 seeds=65536 lambda=1.609438 stride=20224 rounds=2 mean=1.6067 — bit-for-bit vs upstream
test poisson_is_run_to_run_deterministic ... [poisson e2e] run-to-run max|dpred| = 0 (bit-identical)
test poisson_model_still_learns ... [poisson e2e] rmse 0.1602 vs constant-predictor 0.6963 (spread 2.151)
test poisson_rejects_degenerate_subsample ... [poisson e2e] degenerate subsample rejected: value out of range: Poisson bootstrap needs subsample in (0, 1): upstream's GetPoissonLambda returns -1 for subsample >= 1, which zeroes every sample weight (got subsample = 1)
test poisson_sampling_changes_the_model ... [poisson e2e] sampled vs unsampled max|dpred| = 2.303e-2
test poisson_seed_changes_the_draw ... [poisson e2e] seed 1 vs 2 max|dpred| = 2.277e-2
test poisson_subsample_changes_the_draw ... [poisson e2e] subsample 0.5 vs 0.9 max|dpred| = 3.266e-2
test bootstrap_dev_device_matches_upstream ... [device] bootstrap_dev/no: splits + leaf values + staged within 1e-5 of upstream over all 3 trees
[device] bootstrap_dev/bayesian: splits + leaf values + staged within 1e-5 of upstream over all 3 trees
[device] bootstrap_dev/bernoulli: splits + leaf values + staged within 1e-5 of upstream over all 3 trees
[device] bootstrap_dev/mvs: splits + leaf values + staged within 1e-5 of upstream over all 3 trees
test poisson_trains_on_device_and_is_refused_on_cpu ... [poisson] device trained 3 trees; CPU refused: degenerate training input: poisson bootstrap is not supported on CPU (upstream CatBoost rejects it on the CPU task type). It requires the device grow path: build with the `cuda` or `rocm` backend feature and a device-eligible configuration (grow_policy = SymmetricTree, random_strength = 0, unit object weights, boost_from_average = false, Gradient/Simple leaves, no CTR / eval sets / groups)
test wr01_base_device_grower_holds_1e5_vs_cpu ... [BASE n=512 nf=4 d=3 it=5] max|Δpred|=3.605e-11 split_mismatched_trees=0/5
[BASE n=2048 nf=8 d=6 it=10] max|Δpred|=5.992e-11 split_mismatched_trees=0/10
[BASE n=20000 nf=16 d=6 it=20] max|Δpred|=6.212e-11 split_mismatched_trees=4/20
[BASE n=20000 nf=16 d=6 it=20]   tree 0 split-mismatch: max|Δcontribution|=2.799e-11
[BASE n=20000 nf=16 d=6 it=20]   tree 4 split-mismatch: max|Δcontribution|=1.847e-11
[BASE n=20000 nf=16 d=6 it=20]   tree 9 split-mismatch: max|Δcontribution|=2.405e-11
[BASE n=20000 nf=16 d=6 it=20]   tree 19 split-mismatch: max|Δcontribution|=2.121e-11
test wr01_device_run_to_run_jitter_within_budget ... [jitter/no] run-to-run max|Δpred| over 5 fits = 0.000e0
[jitter/bernoulli] run-to-run max|Δpred| over 5 fits = 0.000e0
[jitter/mvs] run-to-run max|Δpred| over 5 fits = 0.000e0
test wr01_device_sampled_bootstrap_holds_1e5_vs_cpu ... [bernoulli n=20000 nf=16 d=6 it=20] max|Δpred(device,cpu)|=5.589e-11 split_mismatched_trees=2/20 first_mismatched_trees=[0, 3]
[bernoulli n=20000 nf=16 d=6 it=20] max|Δpred(sampled,unsampled)|=3.003e-3
[bayesian n=20000 nf=16 d=6 it=20] max|Δpred(device,cpu)|=5.477e-11 split_mismatched_trees=4/20 first_mismatched_trees=[0, 5, 17, 19]
[bayesian n=20000 nf=16 d=6 it=20] max|Δpred(sampled,unsampled)|=4.878e-3
[mvs n=20000 nf=16 d=6 it=20] max|Δpred(device,cpu)|=6.798e-11 split_mismatched_trees=4/20 first_mismatched_trees=[3, 5, 11, 17]
[mvs n=20000 nf=16 d=6 it=20] max|Δpred(sampled,unsampled)|=3.056e-3
test kernels::poisson_bootstrap_oracle_test::poisson_grid_wrap_matches_upstream_bit_for_bit ... [poisson oracle grid_wrap] n=4096 seeds=1024 lambda=1.609438 stride=1024 rounds=2 mean=1.5806 — bit-for-bit vs upstream
test kernels::poisson_bootstrap_oracle_test::poisson_one_pass_matches_upstream_bit_for_bit ... [poisson oracle one_pass] n=1000 seeds=65536 lambda=1.078810 stride=1024 rounds=2 mean=1.0740 — bit-for-bit vs upstream
test kernels::poisson_bootstrap_oracle_test::poisson_wide_matches_upstream_bit_for_bit ... [poisson oracle wide] n=20000 seeds=65536 lambda=1.609438 stride=20224 rounds=2 mean=1.6067 — bit-for-bit vs upstream
test kernels::poisson_bootstrap_speed_test::poisson_parallel_draw_outpaces_the_serial_stream_draw ... [poisson speed] n=2000000: parallel Poisson 17.1ms vs serial stream draw 210.1ms -> 12.3x
test kernels::poisson_bootstrap_speed_test::poisson_parallel_draw_outpaces_the_serial_stream_draw ... [poisson speed] n=2000000: parallel Poisson 17.4ms vs serial stream draw 210.2ms -> 12.1x
test wr01_sampled_fits_run_at_device_speed ... [speed] No        0.079s (device baseline, n=60000 nf=24)
[speed] Bernoulli 0.098s  ratio_vs_No=1.25x
[speed] Bayesian  0.110s  ratio_vs_No=1.40x
[speed] MVS       0.122s  ratio_vs_No=1.55x
[speed] Poisson   0.078s  ratio_vs_No=0.99x
```

## Speed (300000x50, depth 6, 30 iters, RMSE)

| bootstrap | rs on device? | catboost_rs s | CatBoost GPU s | CatBoost CPU s | rs/CB-GPU |
|---|---|---|---|---|---|
| Poisson | True | 1.0851 | 1.2572 | None | 0.863 |
| No | True | 1.0853 | 1.1995 | 1.8551 | 0.905 |
| Bernoulli | True | 1.2526 | 1.2208 | 1.8892 | 1.026 |
| Bayesian | True | 1.4597 | 1.2535 | 1.9533 | 1.164 |
| MVS | True | 1.7051 | 1.2504 | 1.7652 | 1.364 |

## Caveats

- **poisson_parity_basis**: Poisson cannot be gated against a CatBoost-Python run: upstream rejects it on task_type=CPU, and its per-object GPU bootstrap weights are not exposed by any public API. Its parity evidence is bit-for-bit agreement between the device kernel and a verbatim host transcription of upstream's PoissonBootstrapImpl + random_gen.cuh (three launch geometries, two consecutive draws). The CatBoost GPU columns below are a SPEED and quality comparison, not a numeric parity gate.
- **backend_asymmetry**: catboost_rs mirrors upstream's asymmetry: Poisson trains on the device and is REFUSED by the CPU grower, exactly as upstream accepts it on task_type=GPU and rejects it on task_type=CPU. The CatBoost CPU column is therefore expected to be null for the Poisson row — that is the correct result, not a failure.
- **device_residency_proven**: Part B0 proves device residency per arm with CB_GPU_PROF=1 and the per-tree device lines the resident grower emits. Arms without them are reported as CPU.
