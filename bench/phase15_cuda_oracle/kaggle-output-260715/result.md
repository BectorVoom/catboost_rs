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
| 1 | depthwise | 100000 | 0.4428 | 0.5854 | 0.6952 | 1.322x | True |
| 1 | depthwise | 300000 | 1.4349 | 1.8750 | 0.7951 | 1.307x | True |
| 1 | depthwise | 1000000 | 6.0451 | 6.6927 | 0.9314 | 1.107x | True |
| 1 | region | 100000 | 0.7383 | 1.0634 | N/A | 1.440x | True |
| 1 | region | 300000 | 2.3657 | 3.2858 | N/A | 1.389x | True |
| 1 | region | 1000000 | 9.8132 | 11.6451 | N/A | 1.187x | True |
| 6 | depthwise | 10000 | 0.1000 | 0.1130 | 0.6884 | 1.130x | True |
| 6 | depthwise | 100000 | 0.8611 | 1.2416 | 0.7189 | 1.442x | True |
| 6 | depthwise | 300000 | 2.7084 | 4.2705 | 0.8201 | 1.577x | True |
| 6 | region | 10000 | 0.1113 | 0.1308 | N/A | 1.175x | True |
| 6 | region | 100000 | 0.9470 | 1.6157 | N/A | 1.706x | True |
| 6 | region | 300000 | 3.0024 | 4.8565 | N/A | 1.618x | True |

Depth-1 crossover: device first beats CPU at n=100000
