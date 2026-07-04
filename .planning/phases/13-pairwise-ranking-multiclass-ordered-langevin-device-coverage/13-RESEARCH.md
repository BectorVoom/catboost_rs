# Phase 13: Pairwise, Ranking, Multiclass, Ordered & Langevin Device Coverage - Research

**Researched:** 2026-07-04
**Domain:** GPU (CubeCL) device-resident gradient-boosting kernels — five loss/output/residency families ported onto the Phase 10/11/12 device substrate
**Confidence:** HIGH (all findings verified against the local codebase, vendored upstream CatBoost C++, `CATBOOST_CUDA_KERNELS_DESIGN.md`, and the local CubeCL manual — no external/web dependencies)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions (D-01 … D-09 — DO NOT re-litigate)
- **D-01 (all 5 families, one phase, sequenced waves — HARD COMMIT):** Attempt all five families in a single Phase 13; the planner decomposes into internal waves following the roadmap sub-order (pairwise+solver → ranking → multiclass → ordered → langevin). No formal 13.1–13.5 sub-phase split. Ambition is hard commit to all five including ordered boosting — no planned slip. Each family still lands behind its own `Ok(None)` gate (any single family failing sign-off is correctness-safe → CPU), but the phase target is full device coverage of all five.
- **D-02 (retain roadmap ordering):** Keep the roadmap's sub-order — front-loads pairwise (reuses Phase 7.4 histogram kernels), amortizes query-grouping infra across all ranking objectives next, then multiclass, then the heaviest residency (ordered), ending on the small Langevin noise layer.
- **D-03 (single tree, block leaves — mirror upstream multilogit):** Grow ONE shared tree structure per boosting step with multi-row leaf blocks — extend `DeviceGrownTree.leaf_values` from scalar `Vec<f64>` to a flat `leaf_count × approx_dim` block (carry `approx_dim`), routed through the existing multi-output CPU apply. NOT `K` independent scalar trees. This is the one representation extension the phase needs; boundary stays plain-host-structs (landmine-safe).
- **D-04 (full multi-row der2 block leaf estimation):** Device leaf estimation solves the true multi-row der2 block per leaf — **coupled** for `MultiClass` softmax, **diagonal** for the separable losses (`MultiClassOneVsAll`, `MultiLogloss`/multilabel, `MultiRMSE`, `RMSEWithUncertainty`) — matching the CPU Newton path. Reuse/extend the Phase-11 Newton der2 machinery to K-dim blocks. The der-computation losses already exist in `cb-compute`; the device work is the multi-row der2 leaf solve + block-leaf emission, not the der functions.
- **D-05 (full device residency across iterations):** The per-permutation ordered-boosting approx state stays device-resident across iterations (upstream keeps ordered approxes on GPU) — only O(1) descriptors cross the seam per level, exactly like the plain path. Chosen over the lighter "resident approx + host-computed permutation" variant to preserve true no-readback residency. Heaviest residency item in the phase.
- **D-06 (pin seed, freeze permutation in fixture):** Match the CPU ordered path (Phase 5, ≤1e-5) at ε=1e-4 by mirroring Phase 12's D-07 discipline — pin the RNG seed / permutation config, freeze the exact CPU-reference permutation + per-permutation approx trajectory in the oracle fixture, reproduce bit-for-bit on device. Deterministic and checkable at the ε bar (not a distributional/statistical check).
- **D-07 (f64 batched Cholesky + ridge, mirror upstream `linear_solver`):** The batched device Cholesky solver runs decomposition + forward/back substitution in f64 accumulation with upstream's ridge (l2) on the diagonal, matching `CalcScoresCholesky`. f64 for the solve holds ε=1e-4 across hundreds of trees even where the histogram uses the fixed-point `Atomic<u64>` reduction. Batched over leaves. Per-leaf pairwise-derivative matrix assembly is `MakePairwiseDerivatives`/`MakePointwiseDerivatives`. PairLogit pairwise histograms reuse the Phase 7.4 4-channel kernels.
- **D-08 (all 5 objectives on device this phase, incl. stochastic pair):** Cover QueryRMSE, QuerySoftMax, QueryCrossEntropy, YetiRank, and PFound-F on device this phase. The stochastic pair (YetiRank / PFound-F — in-query sampling radix sort) uses the pinned-seed / frozen-fixture discipline (D-06). Shared device query-grouping infra built once and amortized across all five objectives.
- **D-09 (device Gaussian noise on the reduced derivatives, pinned seed):** `AddLangevinNoise` adds a per-element seeded Gaussian to the reduced derivatives on device (§6.3 `langevin_utils`), layering on the existing der residency — smallest family, landed last. Pinned-seed / frozen-fixture reproduction at ε=1e-4.

### Claude's Discretion (research resolves; planner refines)
- Internal wave decomposition/ordering beyond the pinned roadmap sub-order (query-grouping infra likely a shared sub-wave before the ranking objectives; multi-output leaf-block extension a shared sub-wave before multiclass).
- Query-grouping infra mechanics (group-bias removal, in-query sampling radix-sort layout, taken-docs masks) — resolved below against §6.5/§6.6a `query_helper.cu`.
- Cholesky pivoting/ordering details and the exact ridge/`l2` term placement — resolved below against §6.3 `linear_solver` + `leaves_estimation/pairwise_oracle.h`.
- Langevin Gaussian RNG stream layout and per-element seeding — resolved below against §6.3 `langevin_utils` + the CPU `cb_core::normal::std_normal` reference.
- Which specific multiclass fixtures (class counts, coupled vs diagonal) get device oracles — resolved below.

### Deferred Ideas (OUT OF SCOPE)
- Comprehensive aggregate speed benchmark + real named datasets (Higgs/Epsilon), the >20× gap sign-off — Phase 14 (BENCH-03). Phase 13 reuses the Phase-10 synthetic generator + per-family BENCH-02 checks.
- On-device border/quantile computation (`FastGpuBorders`) — out of scope milestone-wide; host CPU quantization stays the ≤1e-5 reference.
- Formal 13.1–13.5 sub-phase split — declined (D-01).
- Deterministic-only ranking subset (drop YetiRank/PFound-F) — declined in favor of all-5 coverage (D-08); noted as lower-risk fallback if stochastic in-query sampling over-runs.
- `K` separate scalar trees for multiclass — declined (D-03).
- Estimated-feature stored-border-VALUE quantization-grid parity — NOT folded (unrelated FEAT-07/KNN, deferred Phase 9).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support (which findings enable planning) |
|----|-------------|---------------------------------------------------|
| **GPUT-11** | PairLogit pairwise-loss training path (pairwise 2×2-cell histograms) runs on device | Phase 7.4 4-channel pairwise histograms + Phase 7.5 pairwise scorer already exist; §6.3 `pairwise_hist*` / `pair_logit.cu` mapped below. Der already in `cb_compute::ranking_der`. |
| **GPUT-21** | Per-leaf pairwise-derivative matrix assembly (`MakePairwiseDerivatives`/`MakePointwiseDerivatives`) + **batched device Cholesky** (decomp + fwd/back subst + ridge + `CalcScoresCholesky`) on device ≤1e-4 | **HIGHEST RISK.** Phase 7.5 deliberately deferred the on-device SPD solve to a host `calculate_pairwise_score` (an explicit "RESEARCH Open Q3" note). This phase must move it on-device. CPU oracle = `cb_train::pairwise_leaves::calculate_pairwise_leaf_values` + `cb_compute::pairwise_cholesky_solve`. Upstream `linear_solver.cu` `RunCholeskySolver<128,256>` + `pairwise_oracle.h` mapped below. |
| **GPUT-22** | Query/listwise objectives (QueryRMSE/QuerySoftMax/QueryCrossEntropy/YetiRank/PFound-F) + device query-grouping infra ≤1e-4 | CPU der in `cb_compute::ranking_der` (`calc_ders_for_queries`, `GroupSpan`). §6.5 `query_*`, `yeti_rank_pointwise`, `pfound_f` + §6.6a `query_helper.cu` mapped below. Device radix/segmented sort + RNG substrate already exist. |
| **GPUT-12** | Multiclass/multi-target/uncertainty (multilogit multi-row der2 blocks) ≤1e-4 | D-03 `DeviceGrownTree` block-leaf extension + D-04 K-dim Newton block. Der already in `cb-compute` (`MultiClass`/`OneVsAll`/`MultiCrossEntropy`/`MultiRMSE`/`RMSEWithUncertainty`). `cb_compute::leaf::cholesky_solve`/`solve_symmetric_newton` is the CPU block oracle. §6.5 `multilogit.cu` mapped below. |
| **GPUT-13** | Ordered boosting (`EBoostingType::Ordered`) trains on device | CPU ref `cb_train::boosting::ordered_approx_delta_simple` + `create_folds`/`Fold`. Pinned-seed/frozen-trajectory fixture (D-06). |
| **GPUT-20** | Langevin/SGLB seeded Gaussian on reduced derivatives ≤1e-4 | §6.3 `langevin_utils::AddLangevinNoise`. CPU Gaussian ref = `cb_core::normal::std_normal` (Marsaglia-polar over `TFastRng64`). Device RNG substrate (PCG transcribed inline) already in `bootstrap_device.rs`. |

**Standing gates enforced by this phase (not new, carried forward):**
- **GPUT-14** — every device-covered case holds ε=1e-4 vs the Rust CPU path; host/CPU path byte-unchanged (D-04 no-regression).
- **BENCH-02** — each family timed on Kaggle CUDA as it lands (device vs host-CPU baseline, and vs official CatBoost GPU where comparable), warm-run/JIT-excluded, train-only.
</phase_requirements>

## Summary

Phase 13 is the final loss-family expansion of the v1.1 device-resident training path. Five families each flip from `Ok(None)`→CPU-fallback to `Ok(Some(DeviceGrownTree))`→device, independently, behind the same per-fit all-or-nothing fallback gate proven in Phases 10–12. **Crucially, this is not greenfield: the der1/der2 computation for all five families already exists in `cb-compute`, the CPU oracles all exist at ≤1e-5 from v1.0 (Phases 5, 6.1–6.4), and the device substrate (resident cindex, partition-aware histograms, Newton der2, deterministic fixed-point reduction, radix + segmented sort, device RNG, pairwise histograms) already exists from Phases 10/11/12/7.4/7.5.** The lift is device-side *leaf-solve + emission + grouping infra*, not new loss math.

The highest-risk item by a wide margin is **GPUT-21's on-device batched Cholesky solver**. Phase 7.5 explicitly *deferred* the SPD solve to the host (a documented "RESEARCH Open Q3" in `pairwise.rs:993` and `score_split.rs:852`): it assembles the pairwise statistics on device but runs `cb_compute::calculate_pairwise_score` / `pairwise_cholesky_solve` on the host over a bounded descriptor. GPUT-21's requirement text and D-07 now mandate the decomposition + forward/back substitution + ridge + `CalcScoresCholesky` run *on device* in f64. This is the "awkward `#[cube]` dense SPD solve" that was previously punted. It is feasible (CubeCL supports f64 generics and shared-memory/plane reductions; upstream's hand-written `RunCholeskySolver<128,256>` is one-logical-warp-per-matrix with `ShuffleReduce`), but it is where the phase's schedule risk concentrates.

Two upstream-vs-Rust structural facts the planner must internalize: (1) upstream CatBoost uses the §6.3 GPU `split_pairwise`+`linear_solver` Cholesky **only for split-scoring** (structure search), and solves the leaf-value system **on the host** via `SolveLinearSystemCholesky` (`descent_helpers.cpp:110`, driven from `pairwise_oracle.h`) — two different systems with two different regularizations. The Rust project's host-light goal (D-05) chooses to run *both* on device. (2) The ε=1e-4 oracle is the **Rust CPU path**, not upstream — so the device Cholesky must reproduce `cb_train::pairwise_leaves` semantics (system size = `leaf_count`, drop the last row → `(n−1)×(n−1)`, `MakeZeroAverage`), which already encode the regularization constants; the device kernel transcribes those, it does not re-derive upstream's.

**Primary recommendation:** Sequence the waves exactly per D-02 with two shared prerequisite sub-waves inserted (query-grouping infra before ranking objectives; `DeviceGrownTree` block-leaf + K-dim Newton extension before multiclass). Land the device batched Cholesky as its own front-loaded wave with an explicit fallback checkpoint — if the on-device `#[cube]` SPD solve over-runs, the correctness-safe interim is to keep the current 7.5 host-side bounded solve (which is already O(1)-per-level) behind `Ok(Some(...))` and defer only the *residency* purity, since correctness (ε=1e-4) is unaffected. Every family self-oracles against its existing CPU function; stochastic/ordered families use pinned-seed frozen fixtures (D-06).

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| PairLogit pairwise histograms (GPUT-11) | Device (`cb-backend` kernels) | — | Reuses Phase 7.4 4-channel `pairwise_hist` kernels; already device-resident. |
| Pairwise/pointwise per-leaf matrix assembly + batched Cholesky + score (GPUT-21) | Device (`cb-backend` `#[cube]` solver) | Host seam (O(1) descriptor out) | D-05/D-07 host-light; f64 SPD solve batched over leaves on device. Interim host fallback allowed (correctness-safe). |
| Query grouping infra: group ids/means/max, group-bias removal, in-query sampling sort, taken-docs masks (GPUT-22) | Device (`cb-backend`, shared sub-wave) | — | §6.6a `query_helper.cu`; amortized across all 5 ranking objectives. Reuses device segmented radix sort + RNG. |
| Query/listwise der + value (GPUT-22) | Device (`cb-backend` der kernels over resident approx) | `cb-compute::ranking_der` (CPU oracle) | Der math already in `cb-compute`; device transcribes per §6.5 `query_*`/`yeti_rank`/`pfound_f`. |
| Multi-output block-leaf representation + K-dim Newton der2 solve (GPUT-12) | Device (`cb-backend`) + `DeviceGrownTree` block extension (host struct) | `cb-compute::leaf::solve_symmetric_newton` (CPU oracle) | D-03/D-04; coupled softmax vs diagonal separable; routes through existing multi-output CPU apply. |
| Ordered boosting per-permutation resident approx trajectory (GPUT-13) | Device (`cb-backend` session state, resident across iterations) | Host O(1) descriptors (D-05) | Heaviest residency; frozen permutation+trajectory fixture (D-06). |
| Langevin/SGLB seeded Gaussian on reduced derivatives (GPUT-20) | Device (`cb-backend` `#[cube]` RNG kernel) | `cb_core::normal::std_normal` (CPU oracle) | Layers on the existing resident der buffer; pinned-seed frozen fixture (D-09/D-06). |
| Per-fit fallback gate + tree emission | Host seam (`cb-train::boosting` ↔ `Runtime::grow_tree_on_device`) | — | `Ok(None)` all-or-nothing (D-04); plain host structs only (no `cb-train` dep in `cb-backend`). |

## Standard Stack

No new external packages are introduced by this phase. All work extends existing workspace crates and the already-vendored CubeCL toolchain.

### Core (existing workspace crates — extended, not added)
| Crate | Role in Phase 13 | Why |
|-------|------------------|-----|
| `cb-backend` | New `#[cube]` kernels: batched Cholesky solver, query-grouping infra, K-dim Newton block, ordered resident approx, Langevin noise | The single feature-gated GPU crate; all `unsafe`/`cubecl` lives here. |
| `cb-compute` | `DeviceGrownTree` block-leaf extension (`leaf_values` → `leaf_count × approx_dim` + `approx_dim`); already holds all der functions + CPU Cholesky/Newton primitives | Seam types + der math + CPU oracles. |
| `cb-train` | Wiring the per-family device path into `train()`/`boosting`; CPU oracle references (`pairwise_leaves`, `ordered_approx_delta_simple`, `create_folds`) | Host orchestration; the frozen ≤1e-5 references. |
| `cb-core` | `TFastRng64` (PCG two-stream), `normal::std_normal` (Marsaglia-polar), `sum_f64` (canonical fold) | RNG + Gaussian + deterministic-sum references the device kernels transcribe inline. |
| `cubecl` (already a dep of `cb-backend`) | `#[cube]` kernels with `generics-float`, `Atomic<u64>`, shared memory, plane/warp reductions | GPU compute; f32/f64 generics verified supported (CubeCL manual "Switching Between f32 and f64"). |

### Supporting (already present)
| Asset | Purpose | Location |
|-------|---------|----------|
| Phase 7.4 pairwise histogram | 4-channel weight-only `(f*n_bins+bin)*4+histId` fill | `cb-backend/src/gpu_runtime/pairwise.rs`, `kernels/pairwise_hist.rs` |
| Phase 7.5 pairwise scorer | assembles pair stats device-side; currently host-solves the small Cholesky | `cb-backend/src/gpu_runtime/pairwise.rs::launch_pairwise_split_score`, `kernels/score_split.rs::pairwise` |
| Phase 11 Newton der2 | `Σder2` channel, `newton_leaf_delta`, `apply_leaf_delta` | `cb-backend/src/kernels/apply_leaf_delta.rs`, `der_seams.rs` |
| Device RNG | `TFastRng64` PCG-XSH-RR transcribed inline in `#[cube]` | `cb-backend/src/kernels/bootstrap_device.rs` |
| Device segmented radix sort | `segmented_radix_sort(head, keys, values)` keys+values per-segment | `cb-backend/src/kernels/exact_quantile.rs` (re-exported), `sort.rs` |
| Deterministic reduction | fixed-point `Atomic<u64>` k=30 (`REDUCE_FIXEDPOINT_SCALE_F64`) + fixed-order tree-reduce fallback | `cb-backend/src/kernels/reduce.rs`, `SPIKE-REDUCTION.md` |

**Installation:** none. `Cargo.toml` gains no new dependency. (Per CLAUDE.md "always use the latest crate versions" applies only if a genuinely new crate were needed — it is not here.)

## Package Legitimacy Audit

**Not applicable.** Phase 13 introduces **zero** new external packages. All work extends existing workspace crates (`cb-backend`, `cb-compute`, `cb-train`, `cb-core`) and the already-vendored `cubecl` toolchain. No `npm`/`pip`/`cargo add` of a new registry package occurs. If planning later discovers a genuinely new crate is required (not anticipated), gate its install behind a `checkpoint:human-verify` task and run `cargo search`/legitimacy check at that point.

## Architecture Patterns

### System Architecture Diagram

```
                       cb-train::train()  (HOST orchestration)
                              │
         per boosting step: compute der1/der2 (cb-compute, resident on device via 7.2 seam)
                              │
                    Runtime::grow_tree_on_device(approx, target)   ← the ONE seam (plain host structs)
                              │
        ┌─────────────────────┴─────────────────────────────────────────────┐
        │                  cb-backend  (DEVICE, feature-gated)                │
        │                                                                     │
        │   [substrate, exists]  resident cindex · partition-aware hist ·     │
        │                        subtraction trick · fixed-point reduce ·     │
        │                        radix/segmented sort · device RNG (PCG)      │
        │                                                                     │
        │   ┌── PAIRWISE (GPUT-11/21) ──────────────────────────────────┐     │
        │   │ 7.4 pairwise_hist (4-ch) → MakePairwise/PointwiseDeriv →   │     │
        │   │ assemble per-leaf SPD systems → *** batched f64 Cholesky   │     │
        │   │ (decomp + fwd/back subst + ridge) *** → CalcScoresCholesky │     │
        │   └───────────────────────────────────────────────────────────┘     │
        │   ┌── RANKING (GPUT-22) ──────────────────────────────────────┐     │
        │   │ query_helper infra: group ids/means/max, group-bias        │     │
        │   │ removal, CreateSortKeys→radix, taken-docs masks (SHARED) → │     │
        │   │ {QueryRMSE, QuerySoftMax, QueryCrossEntropy}=deterministic  │     │
        │   │ {YetiRank, PFound-F}=bootstrap+intra-query sort (pinned)    │     │
        │   └───────────────────────────────────────────────────────────┘     │
        │   ┌── MULTICLASS (GPUT-12) ───────────────────────────────────┐     │
        │   │ block leaves leaf_count×approx_dim; K-dim Newton der2 block │     │
        │   │ coupled(softmax) / diagonal(separable) solve               │     │
        │   └───────────────────────────────────────────────────────────┘     │
        │   ┌── ORDERED (GPUT-13) ──────────────────────────────────────┐     │
        │   │ per-permutation approx trajectory resident across iters;   │     │
        │   │ only O(1) descriptors cross seam (frozen fixture)          │     │
        │   └───────────────────────────────────────────────────────────┘     │
        │   ┌── LANGEVIN (GPUT-20) ─────────────────────────────────────┐     │
        │   │ AddLangevinNoise: per-element seeded Gaussian on resident   │     │
        │   │ reduced derivatives (std_normal transcribed inline)        │     │
        │   └───────────────────────────────────────────────────────────┘     │
        └─────────────────────────────────────────────────────────────────────┘
                              │
                 Ok(Some(DeviceGrownTree{splits, leaf_values[block], ...}))  OR  Ok(None)→CPU grow
                              │
            cb-train emits per-shape host model + applier (multi-output CPU apply for block leaves)
```
File-to-responsibility mapping is in the Component table above; this diagram is data-flow only.

### Recommended Wave Structure (planner refines; D-01/D-02 pinned order + 2 shared sub-waves)
```
Wave A  Pairwise histograms wiring + MakePairwise/PointwiseDerivatives assembly (reuse 7.4/7.5)   [GPUT-11]
Wave B  *** On-device batched f64 Cholesky solver: decomp + fwd/back subst + ridge + CalcScores ***[GPUT-21]  ← highest risk; own wave + fallback checkpoint
Wave C  SHARED query-grouping infra (group ids/means/max, group-bias, sort keys, taken masks)     [GPUT-22 prereq]
Wave D  Deterministic ranking objectives QueryRMSE / QuerySoftMax / QueryCrossEntropy             [GPUT-22]
Wave E  Stochastic ranking YetiRank / PFound-F (pinned-seed frozen fixture, intra-query sort)     [GPUT-22, D-08]
Wave F  SHARED DeviceGrownTree block-leaf extension + K-dim Newton der2 block machinery           [GPUT-12 prereq, D-03]
Wave G  Multiclass/multi-target/uncertainty: coupled softmax + diagonal separable block solves    [GPUT-12, D-04]
Wave H  Ordered boosting: per-permutation resident approx trajectory (frozen fixture)             [GPUT-13, D-05/06]
Wave I  Langevin/SGLB seeded Gaussian on reduced derivatives (frozen fixture)                     [GPUT-20, D-09]
Wave J  Coverage matrix doc + per-family BENCH-02 Kaggle CUDA sign-offs                           [SC-4/SC-5]
```

### Pattern 1: On-device batched Cholesky (GPUT-21 — the pivotal new kernel)
**What:** Replace Phase 7.5's host-side small solve with a `#[cube]` batched SPD solver, one logical warp/plane per matrix, f64 accumulation.
**When to use:** Pairwise split-scoring (`CalcScoresCholesky` — score = max of `βᵀy − ½βᵀAβ`) AND per-leaf pairwise leaf-value solve.
**Upstream shape** (`linear_solver.cu`, verified via `CATBOOST_CUDA_KERNELS_DESIGN.md` §6.3):
- `ExtractMatricesAndTargets` — split packed `linearSystem` (lower-triangle `rowSize*(rowSize+1)/2` then `rowSize` RHS) into matrices/targets/diag.
- `RegularizeImpl` — beta-prior/L2 ridge: bump near-zero diagonals by `averageDiag+0.1`, add `0.05*averageDiag`, `-λ0·cellPrior` off-diagonal, `λ0(1−cellPrior)+λ1` on-diagonal (rank `rowSize−1` — leaf gauge freedom).
- `CholeskyDecompositionImpl<BLOCK,RowSize,SystemSize>` — in-place batched lower-triangular Cholesky, one logical warp per matrix, `ShuffleReduce` dot products, `1e-7` pivot floor + `1e-4` tiny-pivot fallback.
- `SolveForwardImpl<TDirect|TTransposed>` — one kernel serves both fwd and back subst (transposed reverses indices).
- `CalcScoresCholeskyImpl` — score directly from the decomposition.
- `ZeroMeanImpl` — recenter leaf values to zero mean (resolve the rank-deficient gauge).
- **Orchestration:** upstream picks cuSOLVER only when `rowSize>=32 && matCount>=10000`, else the hand-written `RunCholeskySolver<128,256,REMOVE_LAST>`. **There is NO cuSOLVER dependency for this project** — the hand-written path is the model to transcribe (and `linear_cusolver_stub.cu` shows upstream itself supports a no-cuSOLVER build).

**CRITICAL numerics note (VERIFIED):** The ε=1e-4 oracle is the *Rust CPU path*, not upstream. The Rust CPU pairwise **leaf-value** solve is `cb_train::pairwise_leaves::calculate_pairwise_leaf_values` — system size = `leaf_count`, builds the `(n−1)×(n−1)` SPD matrix with `non_diag_reg = -pairwise_bucket_weight_prior_reg/n` and `diag_reg = pairwise_bucket_weight_prior_reg*(1−1/n) + l2_diag_reg`, solves via `cb_compute::pairwise_cholesky_solve`, pushes a trailing 0, then `make_zero_average` (through `cb_core::sum_f64`). The device kernel must reproduce **these exact constants and the drop-last-row + zero-average steps**, NOT upstream's `RegularizeImpl` bump-heuristics (which differ). Split-*scoring* uses the separate `cb_compute::calculate_pairwise_score` path (currently host-side in 7.5).

### Pattern 2: Shared device query-grouping infra (GPUT-22 — built once, amortized)
**What:** A shared sub-wave (Wave C) of `query_helper` kernels before any ranking objective.
**Upstream shape** (`query_helper.cu` §6.6a, verified): one warp (32 lanes) per query, `queriesPerBlock = BLOCK/32`; each lane strides its query's docs accumulating `sumTarget`/`sumWeight` (or max); `WarpReduce` → lane 0 writes `sumTarget/totalWeight` (or `queryMax`).
- `ComputeGroupIds` — scatter `qid` to each doc.
- `ComputeGroupMeans` (two overloads: `qSizes`+`offsetsBias`, or `qOffsets`), `ComputeGroupMax`.
- `RemoveGroupMeans` (doc-parallel `dst[d] -= queryMeans[qids[d]]`) — the residual feed for QueryRMSE/YetiRank.
- `CreateSortKeys` — `ui64 key = (qid << 32) | random_low_32` from per-thread LCG seeds, so a radix sort keeps queries contiguous yet shuffles within a query.
- `FillQueryEndMask`, `FillTakenDocsMask` + `SampledQuerySize(sampleRate,qSize)` (≥2 floor) — in-query sampling masks.
**Rust reuse:** the device segmented radix sort (`segmented_radix_sort`) and device RNG already exist; the CPU der oracle is `cb_compute::ranking_der::calc_ders_for_queries` over `GroupSpan`. Only O(1) group descriptors + the resident der buffer cross the seam.

### Pattern 3: Multi-output block leaves + K-dim Newton der2 (GPUT-12, D-03/D-04)
**What:** Extend `DeviceGrownTree.leaf_values` (currently scalar `Vec<f64>` length `2^depth`) to a flat `leaf_count × approx_dim` block, carry `approx_dim`. Grow ONE shared tree; solve a K-dim der2 block per leaf.
**Upstream shape** (`multilogit.cu` §6.5, verified): predictions stored column-major with stride `predictionsAlignSize` (class `k` at `predictions[idx + k*alignSize]`); der2 kernels emit **one row at a time** (`der2Row` — the lower-triangular block of that row's hessian). MultiLogit softmax over `effectiveClassCount = numClasses−1` (last class implicit logit 0), max-subtracted: `der_k = w*((target==k)−p_k)`; hessian row `−w*p_k*p_row` off-diagonal, `w*(1−p_row)*p_row` diagonal (**coupled**). OneVsAll/MultiCrossEntropy/MultiRMSE/RMSEWithUncertainty are **diagonal**.
**Rust reuse:** der functions exist in `cb-compute`; the CPU block-solve oracle is `cb_compute::leaf::solve_symmetric_newton` / `cholesky_solve` (documented for `k <= ~10` softmax blocks). Extend Phase 11 Newton der2 machinery to K-dim. Block leaves route through the existing multi-output CPU apply (`apply` at `runtime.rs:1145`, `approx[d*n + i]` layout).

### Pattern 4: Ordered boosting resident approx trajectory (GPUT-13, D-05/D-06)
**What:** Keep the per-permutation ordered approx state device-resident across boosting iterations; only O(1) descriptors cross the seam per level.
**CPU ref (VERIFIED):** `cb_train::boosting::ordered_approx_delta_simple` (`boosting.rs:687`) — the anti-leakage body/tail approximant; body rows keep delta 0 (estimation prefix). Fold machinery: `create_folds`/`Fold`, `permutation_count` (default 4 → `max(1, pc−1)` learning folds + 1 averaging fold), `fold_len_multiplier` (default 2.0), `need_shuffle`. Ordered auto-select is GPU-only upstream but this project pins `boosting_type` explicitly (never auto).
**Fixture:** freeze the exact CPU permutation + per-permutation approx trajectory; reproduce bit-for-bit on device at ε=1e-4 (deterministic, not distributional).

### Pattern 5: Langevin/SGLB seeded Gaussian (GPUT-20, D-09)
**What:** `AddLangevinNoise` — each thread advances its per-element RNG seed and adds `coefficient · NextNormal(seed)` to the resident reduced-derivative buffer, persisting the seed (`langevin_utils.cu`, verified §6.3).
**CPU ref (VERIFIED):** `cb_core::normal::std_normal` — Marsaglia-polar rejection loop over `TFastRng64::gen_rand_real1`, consuming an even/variable number of draws, `x*sqrt(-2*ln(r)/r)`. **Transcribe inline** in the `#[cube]` kernel (landmine: kernel cannot reach `cb_core`; follow the `bootstrap_device.rs` precedent that already transcribed `TFastRng64` PCG inline). Pinned-seed frozen fixture.
**Upstream leaf-estimation caveat (VERIFIED):** upstream injects Langevin noise on the reduced derivatives in `leaves_estimation` (`oracle->AddLangevinNoiseToDerivatives`), and **`pairwise_oracle.h` explicitly `CB_ENSURE`s Langevin is NOT supported for the pairwise oracle on GPU**. Langevin therefore layers on the pointwise/groupwise reduced-der path, not the pairwise leaf path.

### Anti-Patterns to Avoid
- **`K` separate scalar trees for multiclass** — declined (D-03); wrong for coupled softmax der2.
- **Adding a `cb-train` dep to `cb-backend`** — the feature-unification landmine (breaks rocm runtime); transcribe CPU refs inline (memory: Phase 7.5).
- **`-inf` float literals in `#[cube]` kernels** — `F::new(f32::NEG_INFINITY)` emits `double(-inf)` → HIP/gfx1100 JIT reject; use a finite `f32::MIN` sentinel (memory: Phase 7.5 WR-01). Host code may keep `f64::NEG_INFINITY`.
- **Non-deterministic reduction** — must use fixed-point `Atomic<u64>` k=30 + fixed-order tree-reduce fallback (SPIKE-REDUCTION §5b). Note: **the Cholesky *solve* is f64 non-atomic arithmetic (per-matrix warp work), not an atomic reduction** — D-07 explicitly allows f64 there even though the histogram uses fixed-point.
- **Reading a `Handle` through a foreign client** — bind each Handle to its allocating client (memory: Phase 7.2 HIP).
- **Fabricating a device result** — any family not yet passing Kaggle CUDA sign-off returns `Ok(None)`→CPU (T-10-05).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Dense SPD solve (CPU oracle) | New Gaussian elimination | `cb_compute::pairwise_cholesky_solve` / `cholesky_solve` / `solve_symmetric_newton` | Already the frozen ≤1e-5 parity oracle; the device kernel must MATCH it, not a fresh CPU solver. |
| Per-object Gaussian noise (CPU oracle) | New RNG/Box-Muller | `cb_core::normal::std_normal` over `TFastRng64` | Exact upstream Marsaglia-polar draw order; device transcribes inline. |
| Intra-query / per-segment sort | New device sort | `segmented_radix_sort` (`cb-backend::exact_quantile`) | Already keys+values, stable, self-oracled bit-exact. |
| Device RNG stream | New device RNG | inline `TFastRng64` PCG-XSH-RR (per `bootstrap_device.rs`) | Two-stream PCG already transcribed + validated; host advances continuous stream, device expands O(1) base state. |
| Deterministic der/weight reduction | f64 atomic-add | fixed-point `Atomic<u64>` k=30 + tree-reduce fallback | gfx1100 has no f64 atomic-add; determinism is mandatory (SPIKE-REDUCTION). |
| Pairwise 2×2 histograms | New histogram kernel | Phase 7.4 4-channel `pairwise_hist` | Already resident + self-oracled. |
| Query der math | New der kernels from scratch | `cb_compute::ranking_der` functions (transcribe into `#[cube]`) | Der already exists at ≤1e-5; only the device transcription is new. |
| Ordered approx body/tail | New ordered logic | `cb_train::boosting::ordered_approx_delta_simple` (frozen ref) | The ≤1e-5 anti-leakage reference from Phase 5. |

**Key insight:** In this phase almost nothing is genuinely new math — the loss der1/der2, the CPU Cholesky/Newton solvers, the RNG, the Gaussian, the sorts, and the pairwise histograms all already exist and are oracle-locked. The novel device work is (a) the on-device batched Cholesky `#[cube]` kernel, (b) the shared query-grouping `#[cube]` infra, (c) the K-dim Newton block solve, (d) the resident ordered trajectory, (e) the Langevin `#[cube]` kernel — each transcribing an existing frozen CPU reference.

## Common Pitfalls

### Pitfall 1: Treating GPUT-21's device Cholesky as "already done" by Phase 7.5
**What goes wrong:** Phase 7.5 assembles pairwise stats on device but runs the SPD solve/score on the *host* (`pairwise.rs:993`, `score_split.rs:852` — the explicit "RESEARCH Open Q3"). Assuming it's on-device leaves GPUT-21/D-07 unmet.
**Why it happens:** The scorer returns device-resident results and looks complete; the host solve is a small bounded step easy to miss.
**How to avoid:** Wave B must implement the `#[cube]` decomp + fwd/back subst + ridge + `CalcScoresCholesky`. Add an explicit checkpoint: if the on-device solve over-runs, keep the host bounded solve behind `Ok(Some(...))` (correctness-safe, ε unaffected) and log the residency gap for Phase 14 rather than blocking the family.

### Pitfall 2: Reproducing upstream's regularization instead of the Rust CPU path's
**What goes wrong:** Upstream `linear_solver.cu::RegularizeImpl` (bump `averageDiag+0.1`, add `0.05*averageDiag`) differs from the Rust CPU `calculate_pairwise_leaf_values` constants (`-prior/n` off-diag, `prior*(1−1/n)+l2` diag) AND from the host oracle in `matrix_per_tree_oracle_base.h` (`WriteSecondDerivatives`). Matching upstream misses the ε=1e-4 vs-Rust-CPU bar.
**Why it happens:** The design doc describes upstream; the oracle is the Rust CPU path.
**How to avoid:** Transcribe `cb_train::pairwise_leaves::calculate_pairwise_leaf_values` (leaf values) and `cb_compute::calculate_pairwise_score` (scoring) constants verbatim into the kernel; drop-last-row + `make_zero_average` included.

### Pitfall 3: Coupled vs diagonal der2 confusion in multiclass
**What goes wrong:** Using a diagonal Newton step for `MultiClass` softmax (coupled hessian) diverges; using a full coupled solve for the separable losses wastes work and may mismatch the CPU diagonal path.
**Why it happens:** All five multi-output losses share the block-leaf representation but differ in hessian structure.
**How to avoid (VERIFIED mapping):** coupled full-block solve only for `MultiClass` (softmax); diagonal for `MultiClassOneVsAll`, `MultiCrossEntropy`/multilabel, `MultiRMSE`, `RMSEWithUncertainty` (D-04). `multilogit.cu` emits der2 one row at a time — mirror that.

### Pitfall 4: RNG draw-order / count divergence in stochastic + Langevin paths
**What goes wrong:** `std_normal` consumes a *variable* (even) number of draws per sample (rejection loop); YetiRank/PFound-F perturb with `uni/(1.000001−uni)` and radix-sort per bootstrap iteration. A wrong draw order or count silently shifts every subsequent value beyond ε=1e-4.
**Why it happens:** Marsaglia-polar rejection and per-iteration re-seeding are order-sensitive; the device must match the host stream exactly.
**How to avoid:** Pin the seed, freeze the CPU reference draws/permutation/trajectory in the fixture (D-06/D-08/D-09), reproduce bit-for-bit. Host advances the continuous stream and hands the device the O(1) base state (bootstrap_device precedent).

### Pitfall 5: Ordered-boosting residency readback creeping in
**What goes wrong:** Reading the per-permutation approx trajectory back to host each iteration (or per level) defeats D-05 residency and the speed goal.
**Why it happens:** The ordered body/tail is stateful across iterations; it's tempting to compute the permutation on host.
**How to avoid:** Keep the trajectory device-resident across iterations; only O(1) descriptors cross the seam per level. Validate residency (no `n`-length readback) as part of the wave's success check.

### Pitfall 6: gfx1100 in-env green ≠ Kaggle CUDA sign-off
**What goes wrong:** rocm in-env is a compile/smoke convenience, NOT the gate. A family "passing" on gfx1100 is not signed off.
**Why it happens:** rocm is the fast local loop; CUDA is human-gated.
**How to avoid:** Each family's SC requires a Kaggle CUDA ε=1e-4 correctness sign-off AND a BENCH-02 timing, human-gated. Do not fabricate; the orchestrator discharges rocm in-env, the Kaggle notebook discharges CUDA (memory: Phase 12 pipeline proven & reusable via `kaggle` CLI).

### Pitfall 7: `#[cube]` `-inf` and foreign-client Handle landmines (recurring)
**What goes wrong:** `-inf` literals JIT-reject on HIP; a Handle read through a foreign client faults. Both invisible to `cargo check`, fail only on GPU.
**How to avoid:** finite `f32::MIN` sentinel in kernels; bind each Handle to its allocating client; run the rocm suite in-env after any `#[cube]` change.

## Code Examples

### CPU oracle to transcribe — pairwise leaf-value solve (the GPUT-21 parity target)
```rust
// Source: crates/cb-train/src/pairwise_leaves.rs:113 (VERIFIED, frozen ≤1e-5 oracle)
// system_size = leaf_count; build (n-1)x(n-1) SPD matrix, solve, push 0, zero-average.
let cell_prior = 1.0 / system_size as f64;
let non_diag_reg = -pairwise_bucket_weight_prior_reg * cell_prior;
let diag_reg = pairwise_bucket_weight_prior_reg * (1.0 - cell_prior) + l2_diag_reg;
// ... fill m=system_size-1 matrix (both triangles) ...
let mut res = cb_compute::pairwise_cholesky_solve(&matrix, &rhs).unwrap_or(vec![0.0; m]);
res.push(0.0);
make_zero_average(&mut res); // subtract mean via cb_core::sum_f64
```

### CPU oracle to transcribe — Gaussian for Langevin (the GPUT-20 parity target)
```rust
// Source: crates/cb-core/src/normal.rs:50 (VERIFIED, Marsaglia-polar)
pub fn std_normal(rng: &mut TFastRng64) -> f64 {
    loop {
        let x = rng.gen_rand_real1() * 2.0 - 1.0;
        let y = rng.gen_rand_real1() * 2.0 - 1.0;
        let r = x * x + y * y;
        if !(r > 1.0 || r <= 0.0) { return x * (-2.0 * r.ln() / r).sqrt(); }
    }
}
```

### Seam contract — `DeviceGrownTree` block-leaf extension (the D-03 change)
```rust
// Source: crates/cb-compute/src/runtime.rs:927 (VERIFIED). Extend leaf_values to a
// flat leaf_count * approx_dim block + carry approx_dim; keep all fields PLAIN HOST
// types (no cubecl/cb-backend type — feature-unification landmine T-10-04).
pub struct DeviceGrownTree {
    pub splits: Vec<(u32, u32)>,
    pub leaf_values: Vec<f64>, // TODAY length 2^depth (scalar); D-03: leaf_count * approx_dim, row-major per leaf
    // + pub approx_dim: usize  (NEW, D-03 — 1 for scalar path, byte-unchanged)
    pub leaf_of: Vec<u32>,
    pub step_nodes: Vec<(u16, u16)>,           // non-sym carrier (empty for oblivious)
    pub node_id_to_leaf_id: Vec<u32>,
    pub region_path: Vec<(u32, u32, bool, bool)>,
}
```

### Upstream device Cholesky reference (shape to transcribe into `#[cube]`)
```
// Source: CATBOOST_CUDA_KERNELS_DESIGN.md §6.3 linear_solver.cu (CITED)
// Hand-written path (NO cuSOLVER — this project has no cuSOLVER dep):
//   RunCholeskySolver<128,256,REMOVE_LAST>:
//     CholeskyDecompositionImpl<BLOCK,RowSize,SystemSize>  // 1 logical warp/matrix, ShuffleReduce, 1e-7 pivot floor, 1e-4 fallback
//     SolveForwardImpl<TDirectSystem>  then  SolveForwardImpl<TTransposedSystem>  // fwd + back subst, one kernel
//     CalcScoresCholeskyImpl  // score = max(β·y - ½·β·A·β)
//     ZeroMeanImpl            // recenter (gauge)
// REMOVE_LAST drops the last row for pfound/pure-pair systems (== Rust drop-last-row).
```

## State of the Art

| Old Approach (pre-Phase-13) | Current Approach (this phase) | When Changed | Impact |
|-----------------------------|-------------------------------|--------------|--------|
| Pairwise SPD solve on HOST (7.5 "Open Q3") | On-device batched f64 Cholesky (`#[cube]`) | Phase 13 GPUT-21 | True host-light pairwise residency; highest-risk kernel. |
| Scalar `leaf_values` (`2^depth`) | `leaf_count × approx_dim` block leaves | Phase 13 D-03 | Multi-output on device; one shared tree per step. |
| Five families return `Ok(None)`→CPU | Each flips to `Ok(Some(tree))` independently | Phase 13 | Completes device-coverage surface for Phase-14 aggregate sign-off. |
| Langevin/stochastic on CPU only | Device seeded-Gaussian + bootstrap intra-query sort | Phase 13 GPUT-20/D-08 | Full ranking + SGLB device coverage. |

**Deprecated/outdated:** none introduced. Host CPU quantization/borders stay the ≤1e-5 reference (`FastGpuBorders` out of scope). cuSOLVER is NOT used (hand-written Cholesky per `linear_cusolver_stub`).

## Runtime State Inventory

Not applicable — Phase 13 is additive device-kernel work, not a rename/refactor/migration. No stored data, live-service config, OS-registered state, secrets, or build artifacts carry a renamed string. (Verified: the phase adds kernels + one struct-field extension; no string rename.)

## Validation Architecture

`workflow.nyquist_validation` is `true` → this section applies.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` (source/test separation mandatory — kernels in `kernels/*.rs`, assertions in `*_test.rs`) |
| Config file | none (cargo workspace) |
| Quick run command | `cargo test -p cb-backend --features rocm <family>_test` (in-env gfx1100 smoke) and `cargo test -p cb-train <family>_oracle_test` |
| Full suite command | `cargo test -p cb-backend --features rocm` + `cargo test -p cb-compute -p cb-train` (per-crate; root disk pressure — verify per-crate, see MEMORY) |
| Authoritative gate | Kaggle CUDA `--features cuda` notebook (human-gated), ε=1e-4 + BENCH-02 timing per family |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | Fixture / Oracle |
|--------|----------|-----------|-------------------|------------------|
| GPUT-11 | PairLogit device histogram + der path | self-oracle | `cargo test -p cb-backend --features rocm pairwise` | Phase 7.4/7.5 pairwise + `cb_compute::ranking_der` |
| GPUT-21 | Device batched Cholesky ≤1e-4 | self-oracle vs CPU | `cargo test -p cb-backend --features rocm cholesky` + `cargo test -p cb-train pairlogit_pairwise_oracle_test` | `pairwise_cholesky_solve` / `calculate_pairwise_leaf_values` |
| GPUT-22 | 5 query objectives + grouping infra ≤1e-4 | self-oracle vs CPU | `cargo test -p cb-backend --features rocm query` + `cargo test -p cb-train queryrmse_oracle_test querysoftmax_oracle_test yetirank_pairwise_oracle_test` | `calc_ders_for_queries`; deterministic (RMSE/SoftMax/CE) vs frozen-fixture (YetiRank/PFound-F) |
| GPUT-12 | Block-leaf K-dim Newton der2 ≤1e-4 | self-oracle vs CPU | `cargo test -p cb-backend --features rocm multiclass` | `cb_compute::leaf::solve_symmetric_newton`; coupled + diagonal fixtures (see below) |
| GPUT-13 | Ordered resident trajectory ≤1e-4 | frozen-fixture | `cargo test -p cb-train ordered*` + `cargo test -p cb-backend --features rocm ordered` | `ordered_approx_delta_simple` frozen permutation + trajectory (D-06) |
| GPUT-20 | Langevin seeded Gaussian ≤1e-4 | frozen-fixture | `cargo test -p cb-backend --features rocm langevin` | `cb_core::normal::std_normal` pinned-seed draw sequence (D-09) |
| GPUT-14 | Host path byte-unchanged | regression | full `cb-train`/`cb-compute` suite | pre-Phase-13 CPU outputs (scalar `leaf_values` byte-identical at `approx_dim==1`) |
| BENCH-02 | Per-family CUDA timing | manual (Kaggle) | Kaggle CUDA notebook | device vs host-CPU baseline, warm/JIT-excluded |

### Sampling Rate
- **Per task commit:** relevant `cargo test -p cb-backend --features rocm <family>_test` (in-env gfx1100).
- **Per wave merge:** `cargo test -p cb-backend --features rocm` + affected `cb-compute`/`cb-train` oracle tests; orchestrator runs `cargo check --tests` cross-crate after any shared-type change (MEMORY: rust-analyzer diagnostics are stale — trust cargo).
- **Phase gate:** each family's Kaggle CUDA ε=1e-4 + BENCH-02 sign-off; coverage matrix documented (SC-5).

### Wave 0 Gaps
- [ ] `cb-backend/src/kernels/cholesky_solve.rs` + `cholesky_solve_test.rs` — device batched Cholesky (GPUT-21). NEW.
- [ ] `cb-backend/src/kernels/query_helper.rs` + test — shared grouping infra (GPUT-22). NEW.
- [ ] `cb-backend/src/kernels/multi_newton.rs` + test — K-dim block der2 solve (GPUT-12). NEW.
- [ ] `cb-backend/src/kernels/langevin.rs` + test — seeded Gaussian (GPUT-20). NEW.
- [ ] Ordered resident-approx session state in `gpu_runtime/session.rs` + test (GPUT-13). Extend.
- [ ] Frozen fixtures: multiclass (coupled softmax K=3, diagonal K=2/uncertainty), YetiRank/PFound-F pinned-seed, ordered permutation+trajectory, Langevin draw sequence. NEW.
- [ ] `DeviceGrownTree.approx_dim` field + block-leaf apply path assertions (D-03). Extend `cb-compute`.
- [ ] Kaggle CUDA notebook cells per family (reuse Phase-10/12 harness). Extend.

**Multiclass fixture recommendation (Claude's discretion, D-04):** at minimum one **coupled** softmax `MultiClass` fixture with `numClasses=3` (`approx_dim=3`, effective 2) to exercise the off-diagonal hessian, and one **diagonal** fixture — `RMSEWithUncertainty` (`approx_dim=2`, distinct row-0/row-1 hessian) plus one `MultiRMSE`/`MultiClassOneVsAll` (`approx_dim>=2`) to exercise the separable path. This covers both hessian structures with minimal fixtures.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust stable + cargo | all | ✓ (assumed per prior phases) | latest stable | — |
| CubeCL (`cubecl` crate, rocm/cuda/wgpu/cpu backends) | all device kernels | ✓ (already a `cb-backend` dep) | workspace-pinned | — |
| AMD gfx1100 / ROCm 7.1 in-env | compile/smoke loop | ✓ (RDNA3 wave32, no CUDA locally) | ROCm 7.1 | rocm is NOT the gate; CUDA is |
| Kaggle CUDA (P100) via `kaggle` CLI | ε=1e-4 + BENCH-02 sign-off | ✓ (pipeline PROVEN, MEMORY Phase 12) | — | human-gated notebook |
| cuSOLVER | — | ✗ (not used) | — | hand-written `RunCholeskySolver` (upstream stub confirms no-cuSOLVER build) |
| CubeCL manual | pre-kernel reading (CLAUDE.md mandate) | ✓ | `/home/user/Documents/workspace/cubecl_manual/manual/cubecl/INDEX.md` | — |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** cuSOLVER — hand-written batched Cholesky is the intended path anyway (D-07).

**CubeCL feasibility (VERIFIED against local manual):** f32/f64 generics supported ("Switching Between f32 and f64"), fixed-point atomics documented (`09_fixedpoint_atomics.md`), shared memory + plane/warp reduction + atomic-contention guidance present. f64 non-atomic arithmetic (the Cholesky solve) is supported; gfx1100 lacks f64 atomic-add (MEMORY) so the *reduction* stays fixed-point `Atomic<u64>` — but the solve is not an atomic reduction, so D-07's f64 solve is feasible.

## Security Domain

`security_enforcement` is `true` (ASVS L1). Phase 13 is a pure numerical GPU-compute library with **no** network surface, no authentication/session/access-control, no persistence of untrusted data, and no user-facing input parsing beyond already-validated training parameters/seeds.

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | no auth surface |
| V3 Session Management | no | no sessions |
| V4 Access Control | no | in-process library |
| V5 Input Validation | weak | training params/dims/seeds validated at the `cb-train` seam (existing); kernel launch geometry guarded against overflow (existing pattern in `pairwise.rs`) — extend to new kernels (checked casts, bounded grid) |
| V6 Cryptography | no | RNG is a deterministic training PRNG (`TFastRng64`), NOT security crypto — do not treat as CSPRNG |
| V7 Error Handling | yes | `thiserror`/`anyhow`, no `unwrap()` in production (CLAUDE.md); `Ok(None)` fallback never fabricates |

### Known Threat Patterns for this stack
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Kernel launch integer overflow (grid/buffer sizing) | Denial of Service | Checked casts + bounded grid (existing `pair_hist_binsums_len_checked` pattern) — apply to new kernels |
| Unbounded RNG rejection loop (`std_normal`) | Denial of Service | Bounded expected iterations (4/π≈1.27), positive-measure accept region — already reasoned (threat T-03-04-02) |
| NaN/panic from non-PD Cholesky pivot | Availability | Non-positive pivot → fall back to zeros (existing `pairwise_cholesky_solve` → `None` → zeros), pivot floor `1e-7`/`1e-4` (upstream) |

No high-severity security findings anticipated; `security_block_on: high` is unlikely to trigger for numeric-kernel work.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The on-device `#[cube]` batched Cholesky (GPUT-21) is schedule-feasible within the phase; if not, the host-side 7.5 solve stays behind `Ok(Some)` (correctness-safe) | Pattern 1 / Pitfall 1 | If both over-run, GPUT-21 residency slips to Phase 14 (correctness still met). LOW correctness risk, MEDIUM schedule risk. |
| A2 | The multiclass fixture set (coupled K=3 softmax + diagonal RMSEWithUncertainty/MultiRMSE) is sufficient device-oracle coverage for D-04 | Validation Arch / Pitfall 3 | Under-coverage of a separable variant; add a fixture. LOW. |
| A3 | Ordered per-permutation trajectory can stay fully device-resident across iterations at ε=1e-4 without host readback (D-05 chosen over lighter variant) | Pattern 4 / Pitfall 5 | If residency proves infeasible, the "resident approx + host-computed permutation" variant is the fallback (still correctness-safe). MEDIUM (heaviest lift). |
| A4 | Langevin only layers on the pointwise/groupwise reduced-der path, not pairwise (upstream `CB_ENSURE`s no pairwise-GPU Langevin) | Pattern 5 | If a fixture combines PairLogit+Langevin, it must fall back to CPU. LOW (matches upstream). |
| A5 | No new external crate is needed | Package Legitimacy / Standard Stack | If one is, gate behind `checkpoint:human-verify` + `cargo search`. LOW. |
| A6 | Rust stable/cargo/CubeCL versions are as in prior phases (not re-verified this session — no build run) | Environment | Stale toolchain; verify at execute time. LOW. |

## Open Questions

1. **On-device Cholesky score-path vs leaf-path unification (GPUT-21).**
   - What we know: split-*scoring* uses `calculate_pairwise_score`; leaf-*values* use `calculate_pairwise_leaf_values` — two systems (2*PartCount vs leaf_count), two regularizations. Upstream even splits device-solved-scoring from host-solved-leaves.
   - What's unclear: whether one `#[cube]` batched Cholesky kernel parameterized by system size + REMOVE_LAST serves both, or two kernels are cleaner.
   - Recommendation: one parameterized batched solver (matches upstream `RunCholeskySolver<...,REMOVE_LAST>`), self-oracled separately against each CPU function.

2. **Ordered boosting fold interaction with permutation_count>1.**
   - What we know: `create_folds` yields `max(1,pc−1)` learning folds + 1 averaging fold; the general fold-pick for `learning_folds>1` was deferred (D-11, Phase 5 memory).
   - What's unclear: whether the device ordered path must cover multi-permutation folds this phase or can pin `permutation_count` in fixtures.
   - Recommendation: pin the permutation config in the frozen fixture (D-06); cover single learning-fold first, document multi-fold as a follow-up if not reached.

3. **Whether QueryCrossEntropy's per-query bisection/Newton shift search needs a device kernel or can reuse the resident der seam.**
   - What we know: upstream `query_cross_entropy.cu` runs 8 bisection + 5 Newton iterations per query with a persistent-grid `atomicCAS` cursor — the most complex ranking kernel.
   - Recommendation: transcribe against `cb_compute::ranking_der` QueryCrossEntropy der; if the shift search over-runs, land QueryRMSE/QuerySoftMax first and gate QueryCrossEntropy independently (`Ok(None)`).

## Sources

### Primary (HIGH confidence — verified this session)
- `crates/cb-compute/src/runtime.rs` (`DeviceGrownTree` :927, seam `grow_tree_on_device` :1255, der functions :171–283, multi-output apply :1145) — [VERIFIED: codebase]
- `crates/cb-train/src/pairwise_leaves.rs` (`calculate_pairwise_leaf_values` :113, regularization constants, drop-last-row + `make_zero_average`) — [VERIFIED: codebase]
- `crates/cb-compute/src/leaf.rs` (`pairwise_cholesky_solve` :272, `cholesky_solve`/`solve_symmetric_newton`) — [VERIFIED: codebase]
- `crates/cb-compute/src/ranking_der.rs` (`calc_ders_for_queries` :139, `GroupSpan`, `group_reduce_weighted`) — [VERIFIED: codebase]
- `crates/cb-core/src/normal.rs` (`std_normal` :50 Marsaglia-polar) + `rng.rs` (`TFastRng64`) — [VERIFIED: codebase]
- `crates/cb-train/src/boosting.rs` (`ordered_approx_delta_simple` :687, `EBoostingType`, fold params) — [VERIFIED: codebase]
- `crates/cb-backend/src/gpu_runtime/pairwise.rs` + `kernels/score_split.rs` (7.5 host-side Cholesky "Open Q3" :991/:993/:852) — [VERIFIED: codebase]
- `crates/cb-backend/src/kernels/{bootstrap_device,sort,exact_quantile,reduce,der_seams}.rs` (device RNG, segmented radix sort, fixed-point reduce, der seam) — [VERIFIED: codebase]
- `catboost-master/catboost/cuda/methods/leaves_estimation/pairwise_oracle.h` + `matrix_per_tree_oracle_base.h` (host-solved leaf system, `HasDiagonalPart`, `WriteSecondDerivatives` regularization, Langevin `CB_ENSURE` not supported for pairwise) — [VERIFIED: upstream source]
- `CATBOOST_CUDA_KERNELS_DESIGN.md` §5.6, §6.3 (`split_pairwise`, `linear_solver`, `langevin_utils`), §6.5 (`query_*`, `yeti_rank_pointwise`, `pfound_f`, `multilogit`), §6.6a (`query_helper.cu`) — [CITED: local design doc]
- `/home/user/Documents/workspace/cubecl_manual/manual/cubecl/` (f32/f64 generics, fixed-point atomics, shared memory, plane reduction) — [VERIFIED: local manual]
- `.planning/REQUIREMENTS.md` (GPUT-11/12/13/14/20/21/22, BENCH-02/03) — [VERIFIED: codebase]
- `.planning/phases/13-.../13-CONTEXT.md` (D-01…D-09) — [VERIFIED: codebase]

### Secondary (MEDIUM confidence)
- MEMORY.md phase notes (Phase 7.2/7.4/7.5/11/12 outcomes, landmines, gfx1100 f64-atomic absence, Kaggle pipeline proven) — [CITED: session memory]

### Tertiary (LOW confidence)
- none — no web/external sources used; this phase is fully internal.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new packages; all crates + CubeCL already present and verified.
- Architecture / family→substrate mapping: HIGH — every family maps to an existing CPU oracle + existing device primitive, verified in-repo and against upstream.
- Highest-uncertainty items (Cholesky, grouping, Langevin, block der2, ordered residency): resolved to a concrete plan with named CPU references; residual risk is GPUT-21 device-Cholesky *schedule* (A1/A3), not correctness.
- Pitfalls: HIGH — drawn from verified code notes + prior-phase landmines.

**Research date:** 2026-07-04
**Valid until:** ~2026-08-03 (30 days; stable internal substrate). Re-verify toolchain versions (A6) at execute time.
