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
| 1 | depthwise | 100000 | 0.5531 | 0.7961 | 0.7134 | 1.439x | True |
| 1 | depthwise | 300000 | 1.6892 | 2.3698 | 0.8279 | 1.403x | True |
| 1 | depthwise | 1000000 | 6.5816 | 8.2317 | 1.0341 | 1.251x | True |
| 1 | region | 100000 | 0.8841 | 1.3768 | N/A | 1.557x | True |
| 1 | region | 300000 | 2.6732 | 4.3006 | N/A | 1.609x | True |
| 1 | region | 1000000 | 10.2832 | 14.3990 | N/A | 1.400x | True |
| 6 | depthwise | 10000 | 0.1187 | 0.1447 | 0.6984 | 1.219x | True |
| 6 | depthwise | 100000 | 1.0015 | 1.6697 | 0.7447 | 1.667x | True |
| 6 | depthwise | 300000 | 3.0778 | 5.4370 | 0.8724 | 1.767x | True |
| 6 | region | 10000 | 0.1475 | 0.1686 | N/A | 1.143x | True |
| 6 | region | 100000 | 1.1481 | 2.1528 | N/A | 1.875x | True |
| 6 | region | 300000 | 3.6317 | 6.3197 | N/A | 1.740x | True |

Depth-1 crossover: device first beats CPU at n=100000
