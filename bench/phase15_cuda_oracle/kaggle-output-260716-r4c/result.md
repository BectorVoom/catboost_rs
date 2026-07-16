# Phase-15 single-session CUDA oracle

GPU: Tesla P100-PCIE-16GB, 580.159.04, 16384 MiB
nvcc: release 12.8
catboost: 1.2.10
correctness_verdict: ALL-PASS
bench_verdict: OK
catboost_gpu_verdict: OK
rv13_oracles_seen: ['empty_group_means_no_fault', 'pairwise_near_equal_border_tiebreak', 'softmax_weight_max_seed', 'tie_order_matches_cpu_stable_descending']

## BENCH-02 depth-1 / depth-6 grow speed (device vs host CPU)

| depth | family | n | device_s | host_cpu_s | catboost_gpu_s | speedup | device>=CPU? |
|---|---|---|---|---|---|---|---|
| 1 | depthwise | 100000 | 0.5278 | 0.7779 | 0.7247 | 1.474x | True |
| 1 | depthwise | 300000 | 1.5630 | 2.3467 | 0.8234 | 1.501x | True |
| 1 | depthwise | 1000000 | 6.1448 | 8.1406 | 0.9864 | 1.325x | True |
| 1 | region | 100000 | 0.8279 | 1.4041 | N/A | 1.696x | True |
| 1 | region | 300000 | 2.4978 | 4.0823 | N/A | 1.634x | True |
| 1 | region | 1000000 | 9.9624 | 14.4984 | N/A | 1.455x | True |
| 6 | depthwise | 10000 | 0.1144 | 0.1451 | 0.7262 | 1.268x | True |
| 6 | depthwise | 100000 | 0.9764 | 1.6662 | 0.7418 | 1.706x | True |
| 6 | depthwise | 300000 | 2.9366 | 5.3810 | 0.8599 | 1.832x | True |
| 6 | region | 10000 | 0.1276 | 0.1677 | N/A | 1.314x | True |
| 6 | region | 100000 | 1.0285 | 2.1010 | N/A | 2.043x | True |
| 6 | region | 300000 | 3.2261 | 6.2100 | N/A | 1.925x | True |

Depth-1 crossover: device first beats CPU at n=100000
