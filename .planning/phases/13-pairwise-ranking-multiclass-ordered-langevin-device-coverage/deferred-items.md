# Deferred Items — Phase 13

Out-of-scope discoveries logged during execution (NOT fixed — pre-existing / other-scope).

## From 13-03 (query-grouping infra)

- **60 pre-existing cb-backend device-only test failures on the cpu backend** (`exact_quantile_test`, `sort::{radix,reorder}`, `segmented_sort_test`, `reduce::block_reduce_atomic_kernel_direct`). These use plane / device features (`plane_inclusive_sum`, f64/u64 atomics) unsupported by the cpu cubecl runtime and are designed to run on rocm/cuda in-env. NOT introduced by 13-03 (query_helper adds 0 failures; 6/6 query_helper tests pass on cpu). Out of scope — the orchestrator discharges the device suite on rocm in-env.

## From 13-10 (D-04/GPUT-14 no-regression check)

- **DI-13-01 — stale Phase-12 test `monotone_non_symmetric_and_region_are_typed_errors` (Region arm).** `crates/cb-train/tests/monotone_oracle_test.rs:286` asserts `grow_policy=Region` on `CpuBackend` returns a **typed error** ("Region OUT", D-6.6-04). **Phase 12 built the CPU Region grow path** (that gap was explicitly closed — build-CPU-Region-FIRST), so Region no longer errors and this assertion is now stale. Confirmed NOT a regression and NOT caused by 13-10 (docs/notebook only): `region_e2e_test` passes 2/2 (CPU Region trains); part (1) of the same test (monotone × non-symmetric → typed error) still passes; **every other cb-train test passes under `--no-fail-fast` and all cb-compute tests pass.** Phase-13 families all decline to `Ok(None)` → CPU numeric path byte-unchanged (D-04 no-regression intent satisfied). **Recommended fix (Phase-12 hardening / monotone-test update):** rewrite the Region arm to assert Region is a supported CPU policy; keep the monotone × non-sym rejection arm intact. Do NOT re-assert "Region OUT".
