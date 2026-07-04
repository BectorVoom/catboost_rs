# Phase-13 CUDA sign-off

GPU: Tesla P100-PCIE-16GB, 580.159.04, 16384 MiB
nvcc: release 12.8
correctness_verdict: ALL-PASS
bench_verdict: OK

## Correctness (device vs Rust CPU, eps=1e-4)
### pairwise (deriv + batched Cholesky) GPUT-11/21
exit=0 secs=773.4 ran=True
summary: ['test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 198 filtered out; finished in 1.36s', 'test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s']
  - Downloaded wgpu-core-deps-windows-linux-android v29.0.3
  - Downloaded argminmax v0.6.3
  - Compiling argminmax v0.6.3
  - warning: function `compute_group_max_host` is never used
  - 421 | pub(crate) fn compute_group_max_host(values: &[f64], q_offsets: &[u32]) -> CbResult<Vec<f64>> {
  - warning: function `query_softmax_ders_host` is never used
  - 452 | pub(crate) fn query_softmax_ders_host(
  - Finished `release` profile [optimized] target(s) in 12m 49s
  - Running unittests src/lib.rs (/tmp/target/release/deps/cb_backend-1ce8c652587b4afb)

### ranking (query grouping + det + stochastic) GPUT-22
exit=0 secs=3.5 ran=True
summary: ['test result: ok. 14 passed; 0 failed; 0 ignored; 0 measured; 192 filtered out; finished in 1.68s', 'test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s']
  - pfound_f der2: max_div = 0.000e0 (<= 0.0001)
  - yetirank der2: max_div = 0.000e0 (<= 0.0001)
  - warning: function `compute_group_max_host` is never used
  - 421 | pub(crate) fn compute_group_max_host(values: &[f64], q_offsets: &[u32]) -> CbResult<Vec<f64>> {
  - warning: function `query_softmax_ders_host` is never used
  - 452 | pub(crate) fn query_softmax_ders_host(
  - Finished `release` profile [optimized] target(s) in 0.40s
  - Running unittests src/lib.rs (/tmp/target/release/deps/cb_backend-1ce8c652587b4afb)

### multiclass (softmax der + multi-Newton) GPUT-12
exit=0 secs=2.6 ran=True
summary: ['test result: ok. 9 passed; 0 failed; 0 ignored; 0 measured; 197 filtered out; finished in 0.78s', 'test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s']
  - warning: function `compute_group_max_host` is never used
  - 421 | pub(crate) fn compute_group_max_host(values: &[f64], q_offsets: &[u32]) -> CbResult<Vec<f64>> {
  - warning: function `query_softmax_ders_host` is never used
  - 452 | pub(crate) fn query_softmax_ders_host(
  - Finished `release` profile [optimized] target(s) in 0.37s
  - Running unittests src/lib.rs (/tmp/target/release/deps/cb_backend-1ce8c652587b4afb)

### ordered (resident approx trajectory) GPUT-13
exit=0 secs=2.9 ran=True
summary: ['test result: ok. 10 passed; 0 failed; 0 ignored; 0 measured; 196 filtered out; finished in 1.09s', 'test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s']
  - [partition_update n=37] REPORTED max abs_div=0.000e0 rel_div=0.000e0 (bound=1e-9)
  - [partition_update n=1000] REPORTED max abs_div=0.000e0 rel_div=0.000e0 (bound=1e-9)
  - [scan n=37 n_bins=2 n_features=1] REPORTED max abs_div=0.000e0 rel_div=0.000e0 (bound=1e-9)
  - [scan n=1000 n_bins=2 n_features=1] REPORTED max abs_div=0.000e0 rel_div=0.000e0 (bound=1e-9)
  - [scan n=1 n_bins=2 n_features=3] REPORTED max abs_div=0.000e0 rel_div=0.000e0 (bound=1e-9)
  - [scan n=37 n_bins=2 n_features=3] REPORTED max abs_div=0.000e0 rel_div=0.000e0 (bound=1e-9)
  - [scan n=1000 n_bins=2 n_features=3] REPORTED max abs_div=0.000e0 rel_div=0.000e0 (bound=1e-9)
  - [scan n=1 n_bins=16 n_features=1] REPORTED max abs_div=0.000e0 rel_div=0.000e0 (bound=1e-9)
  - [scan n=37 n_bins=16 n_features=1] REPORTED max abs_div=0.000e0 rel_div=0.000e0 (bound=1e-9)
  - [scan n=1000 n_bins=16 n_features=1] REPORTED max abs_div=0.000e0 rel_div=0.000e0 (bound=1e-9)
  - [scan n=1 n_bins=16 n_features=3] REPORTED max abs_div=0.000e0 rel_div=0.000e0 (bound=1e-9)
  - [scan n=37 n_bins=16 n_features=3] REPORTED max abs_div=0.000e0 rel_div=0.000e0 (bound=1e-9)

### langevin (seeded Gaussian / SGLB) GPUT-20
exit=0 secs=2.1 ran=True
summary: ['test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 203 filtered out; finished in 0.33s', 'test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s']
  - [langevin seed=42 coef=0.2 n=64] max_div=1.110e-16
  - [langevin seed=2024 coef=0.011 n=200] max_div=2.220e-16
  - [langevin draw-count] value max_div=4.441e-16
  - warning: function `compute_group_max_host` is never used
  - 421 | pub(crate) fn compute_group_max_host(values: &[f64], q_offsets: &[u32]) -> CbResult<Vec<f64>> {
  - warning: function `query_softmax_ders_host` is never used
  - 452 | pub(crate) fn query_softmax_ders_host(
  - Finished `release` profile [optimized] target(s) in 0.38s
  - Running unittests src/lib.rs (/tmp/target/release/deps/cb_backend-1ce8c652587b4afb)

## BENCH-02 grow loop

| family | n | device_s | cpu_s | speedup | dev_trees | cpu_trees |
|---|---|---|---|---|---|---|
| depthwise | 10000 | 0.1080 | 2.6645 | 24.664x | 20 | 20 |
| depthwise | 100000 | 0.9167 | 30.3894 | 33.151x | 20 | 20 |
| depthwise | 300000 | 2.9717 | 101.5605 | 34.176x | 20 | 20 |
| region | 10000 | 0.1310 | 3.1296 | 23.888x | 20 | 20 |
| region | 100000 | 0.9867 | 36.1485 | 36.635x | 20 | 20 |
| region | 300000 | 3.2888 | 111.6311 | 33.943x | 20 | 20 |
