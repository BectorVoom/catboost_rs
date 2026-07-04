# Deferred Items — Phase 13

Out-of-scope discoveries logged during execution (NOT fixed — pre-existing / other-scope).

## From 13-03 (query-grouping infra)

- **60 pre-existing cb-backend device-only test failures on the cpu backend** (`exact_quantile_test`, `sort::{radix,reorder}`, `segmented_sort_test`, `reduce::block_reduce_atomic_kernel_direct`). These use plane / device features (`plane_inclusive_sum`, f64/u64 atomics) unsupported by the cpu cubecl runtime and are designed to run on rocm/cuda in-env. NOT introduced by 13-03 (query_helper adds 0 failures; 6/6 query_helper tests pass on cpu). Out of scope — the orchestrator discharges the device suite on rocm in-env.

## From 13-10 (D-04/GPUT-14 no-regression check)

- **DI-13-01 — stale Phase-12 test `monotone_non_symmetric_and_region_are_typed_errors` (Region arm).** `crates/cb-train/tests/monotone_oracle_test.rs:286` asserts `grow_policy=Region` on `CpuBackend` returns a **typed error** ("Region OUT", D-6.6-04). **Phase 12 built the CPU Region grow path** (that gap was explicitly closed — build-CPU-Region-FIRST), so Region no longer errors and this assertion is now stale. Confirmed NOT a regression and NOT caused by 13-10 (docs/notebook only): `region_e2e_test` passes 2/2 (CPU Region trains); part (1) of the same test (monotone × non-symmetric → typed error) still passes; **every other cb-train test passes under `--no-fail-fast` and all cb-compute tests pass.** Phase-13 families all decline to `Ok(None)` → CPU numeric path byte-unchanged (D-04 no-regression intent satisfied). **Recommended fix (Phase-12 hardening / monotone-test update):** rewrite the Region arm to assert Region is a supported CPU policy; keep the monotone × non-sym rejection arm intact. Do NOT re-assert "Region OUT".

## From 13-REVIEW (high-effort code review of `89f9f04..HEAD`) — hardening debt

All 4 correctness findings are **PLAUSIBLE/latent** (data-dependent, not firing on the frozen fixtures or the P100 CUDA ALL-PASS) and NONE are on an end-to-end device-grow path yet (families decline to `Ok(None)`→CPU). Logged for a future GPUT-22/GPUT-11 hardening pass, not blocking Phase 13's device-coverage MVP goal. Full detail + cleanup items in `13-REVIEW.md`.

- **RV-13-01 (parity, ranking.rs:766)** — YetiRank tie-break: descending order via *reverse of a stable ascending* radix sort inverts tied perturbed values vs CPU's stable descending sort → decay-coefficient swap → der1/der2 > 1e-4 when a query has tied `exp(approx)`. Fix with a stable sort-by-f64-bits descending (also resolves RV-13-07 perf).
- **RV-13-02 (parity, ranking.rs:475)** — weighted QuerySoftMax max-shift taken over ALL docs vs CPU's max over `weight>0` docs only; diverges when the max-approx doc has weight ≤ 0.
- **RV-13-03 (robustness, query_helper.rs:449)** — `compute_group_means_host` doesn't guard `n==0`; empty-group offsets launch zero-length device buffers → possible rocm/cuda fault. Add an `n==0` short-circuit.
- **RV-13-04 (parity, pairwise.rs:1754)** — device f64 Cholesky (13-02 wire-device) can flip the host argmin tie-break vs the frozen CPU scorer; wgpu (frozen host scorer) vs cpu/rocm/cuda (device solve) may disagree on the winning split for near-equal borders.
- **Cleanup (RV-13-05..09):** dead coverage-gate `map_*_coverage` builds discarded before `Ok(None)` (session.rs:890/925/947/872/966); memory-heavy `Vec<Vec<Vec<f64>>>` multiclass leaf accumulation (multiclass.rs:261); redundant per-(query×perm) radix sort (ranking.rs:868); duplicated Cholesky `#[cube]` body (cholesky_solve.rs:139 vs multi_newton.rs:206); pervasive copy-pasted PCG/RNG + solver + residency helpers across kernel modules (langevin.rs:69 et al.).
