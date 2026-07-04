# Deferred Items — Phase 13

Out-of-scope discoveries logged during execution (NOT fixed — pre-existing / other-scope).

## From 13-03 (query-grouping infra)

- **60 pre-existing cb-backend device-only test failures on the cpu backend** (`exact_quantile_test`, `sort::{radix,reorder}`, `segmented_sort_test`, `reduce::block_reduce_atomic_kernel_direct`). These use plane / device features (`plane_inclusive_sum`, f64/u64 atomics) unsupported by the cpu cubecl runtime and are designed to run on rocm/cuda in-env. NOT introduced by 13-03 (query_helper adds 0 failures; 6/6 query_helper tests pass on cpu). Out of scope — the orchestrator discharges the device suite on rocm in-env.
