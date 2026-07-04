# Code Review — Phase 13 (device coverage: pairwise / ranking / multiclass / ordered / langevin)

**Scope:** `89f9f04..HEAD` (31 commits, +6921 lines, cb-backend GPU/kernel layer)
**Method:** workflow-backed high-effort review — per-angle finders + independent adversarial verifier per (file,line); 20 agents.
**Result:** 4 correctness/parity hazards (all PLAUSIBLE — latent/data-dependent, none actively firing on the frozen fixtures or the Kaggle CUDA sign-off) + 8 cleanup issues.

Phase status is **PASSED** (verifier 9/9, real P100 CUDA ALL-PASS). None of these findings block the phase's device-coverage MVP goal — the five families decline to `Ok(None)`→CPU at the session level, so the latent hazards are not on an end-to-end device-grow path yet. They are logged as **hardening debt** for a future gap/hardening pass (analogous to prior-phase WR-xx items). See `deferred-items.md` (RV-13-0x).

## Correctness / parity hazards (PLAUSIBLE — latent)

| ID | File:Line | Hazard |
|----|-----------|--------|
| RV-13-01 | `gpu_runtime/ranking.rs:766` | `descending_order_per_query` reverses a **stable ascending** radix sort → inverts relative order of **tied** perturbed values vs CPU's stable descending sort. On YetiRank/YetiRankPairwise queries with tied `exp(approx)`+f32-Gumbel, decay-coefficient assignment (`CalcWeightsClassic`) swaps → der1/der2 diverge > 1e-4. Comment at :735 only asserts the *frozen fixture* has no ties; nothing enforces it for real input. |
| RV-13-02 | `gpu_runtime/ranking.rs:475` | `query_softmax_ders_host` seeds the per-query exp shift from `compute_group_max_host` (max over **all** docs), but CPU `TQuerySoftMaxError` seeds `maxApprox` only over docs with **weight > 0** (`ranking_der.rs:257-266`). For a weighted QuerySoftMax fit where the max-approx doc has weight ≤ 0, every `exp(β·(approx−maxApprox))` term is scaled differently → breaks the parity bar. Reachable outside the uniform-weight regime. |
| RV-13-03 | `kernels/query_helper.rs:449` | `compute_group_means_host` short-circuits only on `n_groups==0`, not `n==0` → an all-empty-group offset (`q_offsets=[0,0]`) launches kernels over **zero-length device buffers**, which can fault on rocm/cuda (project HIP residency lesson). Crate-public helper is unguarded even though driver callers guard `n==0` upstream. |
| RV-13-04 | `gpu_runtime/pairwise.rs:1754` | The 13-02 wire-device decision replaced the frozen CPU parity scorer with a host-assembled + **device f64 Cholesky** solve on cpu/rocm/cuda (numeric assert skipped off-device per WR-01); only the `wgpu` branch keeps the frozen host scorer. Different f64 accumulation order can flip the host argmin **tie-break** between near-equal borders → wgpu and cpu/rocm/cuda disagree on the winning split for identical pairwise input. |

## Cleanup

| ID | File:Line | Issue |
|----|-----------|-------|
| RV-13-05 | `gpu_runtime/session.rs:890` (CONFIRMED) | Pairwise coverage gate builds a full `PairwiseState` then discards it — both match arms `return Ok(None)`, so `map_pairwise_coverage(...)`+match collapses to `return Ok(None)`. Identical dead pattern at ranking (:925), multiclass (:947), ordered (:872 `let _ordered`), langevin (:966 `let _langevin`). Risk: future wiring may assume coverage state is stored on the session when it never is. |
| RV-13-06 | `gpu_runtime/multiclass.rs:261` (CONFIRMED) | `accumulate_leaf_blocks` materializes per-leaf membership `k+pk`-fold as `Vec<Vec<Vec<f64>>>` (≈65 identical-length vecs/leaf at K=10) purely to feed ordered `sum_f64` — contradicts the first-class memory-efficiency constraint. A single `Vec<usize>` of object indices per leaf + mapped-gather sum is equivalent. |
| RV-13-07 | `gpu_runtime/ranking.rs:868` (CONFIRMED) | `descending_order_per_query` runs **two** 32-bit segmented radix passes + head-flags + ~6 fresh Vecs **per (query × permutation)** on a single-segment slice — segmentation machinery is pure overhead. A plain stable sort-by-f64-bits gives identical order far cheaper. (Same site as RV-13-01 — fixing both together is natural.) |
| RV-13-08 | `kernels/cholesky_solve.rs:139` (PLAUSIBLE) | SPD Cholesky decompose + fwd/back-solve + non-positive-pivot zeros-fallback transcribed twice (`cholesky_solve.rs:139-219` and `multi_newton.rs:206-278`), differing only in reduced size (m=n−1 vs k). A numeric correction must be duplicated or one path silently keeps wrong numerics. Extract a shared `#[cube] fn cholesky_decompose_and_solve`. |
| RV-13-09 | `kernels/langevin.rs:69` (+ others) | PCG primitives (`rotate_right_u32`, `pcg_mix`, `LCG_MULTIPLIER`, `REAL1_INV`) copy-pasted verbatim across `langevin.rs`, `query_helper.rs:62/73`, `ranking.rs:601/612`, on top of pre-existing `mvs_device`/`bootstrap_device` copies. Plus additional copy-paste of solver + residency-helper code across kernel modules. Consolidate into shared helpers. |

## Not fabricated / already-clean

- All three crates (`cb-compute`, `cb-backend`, `cb-train`) `cargo check --tests` clean.
- The `Ok(None)`→CPU coverage-gate MVP scope is the documented phase design, verified in code (`session.rs:850-955`), not a hidden gap.
- Kaggle CUDA P100 sign-off (`bench/phase13_cuda_oracle/`) internally consistent — real GPU identity, matching test counts.
