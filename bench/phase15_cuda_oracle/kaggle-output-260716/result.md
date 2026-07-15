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
| 1 | depthwise | 100000 | 0.3976 | 0.5971 | 0.6904 | 1.502x | True |
| 1 | depthwise | 300000 | 1.2830 | 1.9466 | 0.7966 | 1.517x | True |
| 1 | depthwise | 1000000 | 5.1396 | 6.8485 | 0.9543 | 1.332x | True |
| 1 | region | 100000 | 0.6482 | 1.0663 | N/A | 1.645x | True |
| 1 | region | 300000 | 2.0569 | 3.3628 | N/A | 1.635x | True |
| 1 | region | 1000000 | 8.5809 | 12.1982 | N/A | 1.422x | True |
| 6 | depthwise | 10000 | 0.0902 | 0.1158 | 0.6816 | 1.284x | True |
| 6 | depthwise | 100000 | 0.7525 | 1.2792 | 0.7250 | 1.700x | True |
| 6 | depthwise | 300000 | 2.3706 | 4.4101 | 0.8203 | 1.860x | True |
| 6 | region | 10000 | 0.1095 | 0.1402 | N/A | 1.280x | True |
| 6 | region | 100000 | 0.7864 | 1.6725 | N/A | 2.127x | True |
| 6 | region | 300000 | 2.6421 | 5.0379 | N/A | 1.907x | True |

Depth-1 crossover: device first beats CPU at n=100000
