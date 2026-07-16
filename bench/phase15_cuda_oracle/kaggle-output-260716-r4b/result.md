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
| 1 | depthwise | 100000 | 0.4010 | 0.5712 | 0.6881 | 1.424x | True |
| 1 | depthwise | 300000 | 1.2873 | 1.8189 | 0.7934 | 1.413x | True |
| 1 | depthwise | 1000000 | 5.1896 | 6.6275 | 0.9486 | 1.277x | True |
| 1 | region | 100000 | 0.6268 | 1.0324 | N/A | 1.647x | True |
| 1 | region | 300000 | 2.0074 | 3.2274 | N/A | 1.608x | True |
| 1 | region | 1000000 | 8.3678 | 11.3677 | N/A | 1.359x | True |
| 6 | depthwise | 10000 | 0.0909 | 0.1071 | 0.6720 | 1.178x | True |
| 6 | depthwise | 100000 | 0.7186 | 1.1963 | 0.7403 | 1.665x | True |
| 6 | depthwise | 300000 | 2.3923 | 4.1598 | 0.8372 | 1.739x | True |
| 6 | region | 10000 | 0.1046 | 0.1291 | N/A | 1.234x | True |
| 6 | region | 100000 | 0.7966 | 1.5406 | N/A | 1.934x | True |
| 6 | region | 300000 | 2.6452 | 4.7294 | N/A | 1.788x | True |

Depth-1 crossover: device first beats CPU at n=100000
