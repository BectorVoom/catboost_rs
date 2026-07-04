# Requirements: catboost-rs — v1.1 GPU Performance

**Defined:** 2026-06-28
**Re-scoped:** 2026-07-02 (in place, against `CATBOOST_CUDA_KERNELS_DESIGN.md` — 17 → 25 requirements; +GPUT-15..22; GPUT-11/12 reworded)
**Core Value:** A memory-efficient, Rust-native CatBoost implementation with verifiable feature parity (oracle-tested ≤10⁻⁵ on CPU; ε=1e-4 on GPU), embeddable in Rust and droppable into both scikit-learn and existing CatBoost Python pipelines.

> **Milestone goal:** Full CUDA device-resident training parity — move the entire boosting inner loop (histogram build, split scoring, BestSplit, partition/leaf-assignment, leaf values) onto the GPU, not just derivatives, reaching speed parity with official CatBoost GPU while preserving correctness. The >20× gap in v1.0 was the derivatives-only MVP: `grow_boosting_pass` (`crates/cb-backend/src/gpu_runtime/mod.rs:1890`) exists but is never wired into `cb_train::train`.
>
> **Authoritative kernel reference:** `CATBOOST_CUDA_KERNELS_DESIGN.md` maps the complete upstream CUDA training-kernel surface (79 `.cu` + 77 `.cuh` across 9 kernel directories) — per-file processing flow, host/device split, I/O data types, and algorithm. Every v1.1 requirement below is grounded in it, and every phase cites it. It describes the *original* CUDA engine; the v1.1 target is a CubeCL reimplementation of the same behavior. **Note (no CUB in CubeCL):** the upstream engine delegates sorts/scans/segmented reductions to NVIDIA CUB; CubeCL has no CUB, so those primitives are a from-scratch device deliverable (GPUT-16), not a wrapper.
>
> **Validation authority — ALL GPU (CUDA) kernel oracles, correctness AND speed, run on a Kaggle CUDA notebook.** CUDA is the single authoritative GPU oracle for this milestone. A reproducible Kaggle CUDA oracle/test harness is a **foundational deliverable established in Phase 10** (BENCH-01) that measures BOTH correctness AND wall-clock speed. **Speed is checked for every GPU kernel from the first phase to the last** — every phase that lands GPU kernels reports a Kaggle CUDA speed measurement (device path vs the host-CPU baseline, and vs official CatBoost GPU where a comparable config exists) alongside its correctness oracle. Speed is NOT deferred to a single end-of-milestone benchmark. There is no NVIDIA hardware in-env; the AMD/ROCm in-env GPU remains an OPTIONAL compile/smoke convenience for fast local iteration, but it is **not a gate** — no requirement is satisfied by ROCm validation alone. A Kaggle CUDA oracle/speed run is a **human-gated external step** (the user runs the notebook).
>
> **Parity bar:** GPU device path is held to **ε=1e-4 vs the Rust CPU path** on CUDA (Phase 7.6 precedent — device math is f32; bit-exact f64 ≤1e-5 is not the GPU goal), with the depth-1 device tree held tighter at ≤1e-5 where the level-0 whole-dataset histogram is the exact CPU score. The CPU path remains oracle-locked ≤10⁻⁵ and byte-unchanged (D-04 no-regression).
>
> **Landmine:** never add a `cb-train` dependency to `cb-backend` (Cargo feature unification breaks the rocm runtime) — transcribe CPU references inline; the `Runtime` seam stays CubeCL-free. Kernels remain CubeCL-portable (cuda/rocm/wgpu) so ROCm smoke-testing stays possible, but CUDA on Kaggle is the oracle of record. Note: CUDA provides f64 atomic-add (unlike gfx1100), so the atomic-free constraint is a portability nicety rather than a hard gate — but parallel-reduction **determinism** still governs the ε=1e-4 parity bar, so a deterministic reduction strategy is still required.

## v1.1 Requirements

### GPU Device-Resident Training — Seam, Residency & Foundations (GPUT)

- [x] **GPUT-01**: A `Runtime` grow-tree trait seam (`begin_device_training` / `grow_tree_on_device` returning `CbResult<Option<DeviceGrownTree>>` / `end_device_training`) exists in `cb-compute` with CubeCL-free host-typed signatures, and a `Ok(None)`→host-CPU fallback so any uncovered case stays correct.
- [x] **GPUT-02**: A `GpuTrainSession` (cb-backend-internal) owns one `ComputeClient` + all persistent device handles for the whole fit; the quantized feature matrix is uploaded once above the iteration loop (no per-tree re-upload).
- [x] **GPUT-03**: Gradients/approx stay device-resident across boosting iterations; the per-tree `der1` host read-back is eliminated; only the O(1) BestSplit descriptor + `2^depth` partition statistics cross host↔device per level (D-05).
- [x] **GPUT-15**: A bit-packed device-resident **compressed index** (cindex) with `TCFeature` Offset/Shift/Mask/OneHot addressing is built and kept resident as the single input to every histogram kernel, matching the CPU quantized layout ≤1e-4, oracle-tested on Kaggle CUDA. (Borders stay host — CPU quantization is the ≤1e-5 reference per GPUT-02; only cindex packing/residency is the device deliverable. §6.6a `gpu_data/kernel/binarize.cu`, `WriteCompressedIndex`.)
- [x] **GPUT-16**: A from-scratch **CubeCL-portable device-primitive library** — fill/transform (gather-scatter, vector arithmetic), full + segmented prefix scan, reduce / segmented-reduce / reduce-by-key, radix sort + stable single-bit reorder, bit-compression, `TDataPartition` offset/size update, and per-partition stat aggregation (`update_part_props`) — runs on device with a deterministic reduction, matching the CPU path ≤1e-4, oracle-tested on Kaggle CUDA. (No CUB in CubeCL — these are real deliverables, not wrappers. §6.1 `cuda_util/kernel`, §6.2 `cuda_util/kernel/sort`.)

### GPU Device-Resident Training — Tree Growth & Scoring (GPUT)

- [x] **GPUT-04**: A depth-1 oblivious tree is grown fully on device (RMSE/Logloss, Plain boosting, fold_count=1) and matches the CPU path ≤1e-5, oracle-tested on Kaggle CUDA.
- [x] **GPUT-05**: Partition-aware histograms (`fullPass=false`) keyed by leaf, contiguous partition reorder, and the histogram subtraction trick support depth>1 oblivious trees on device.
- [x] **GPUT-06**: A chosen reduction-determinism strategy keeps device histogram/score reductions within ε=1e-4 of the CPU path across hundreds of trees, verified on Kaggle CUDA (CUDA has f64 atomic-add, but atomicAdd ordering is still non-deterministic, so a deterministic reduction is required).
- [x] **GPUT-07**: Newton der2 leaf estimation runs on device (required for classification / Logloss default).
- [x] **GPUT-08**: The Cosine / second-order score function (the GPU default) runs on device.
- [x] **GPUT-18**: The **Depthwise, Lossguide, and Region** grow policies — per-policy leaf selection (`ComputeOptimalSplitsRegion` / `ComputeOptimalSplit` + `SelectLeavesToSplit`) and region/non-symmetric tree leaf-value apply (`AddRegion` / `ComputeNonSymmetricDecisionTreeBins`) — run on device, matching the CPU path ≤1e-4, oracle-tested on Kaggle CUDA. (GPUT-04/05 are SymmetricTree/oblivious only. §6.4, §6.6c.)
- [x] **GPUT-19**: **Exact** weighted-quantile leaf-value estimation (`exact_estimation`: needWeights = totalWeight·α, binary search over per-bin weight prefix sums) runs on device for Quantile/MAE/MAPE-family objectives, matching the CPU path ≤1e-4, oracle-tested on Kaggle CUDA. (Distinct from the Newton path in GPUT-07. §6.3 `exact_estimation.{cu,cuh}`.)

### GPU Device-Resident Training — Sampling, Losses & Coverage (GPUT)

- [x] **GPUT-09**: Bootstrap + random-strength sampling runs on device (sampling parity for non-default `bootstrap_type`).
- [x] **GPUT-17**: **Minimal Variance Sampling (MVS)** bootstrap — per-block optimal threshold on `sqrt(der²+λ)` with inverse-probability reweighting — runs on device (MVS is CatBoost's *default* GPU sampling, distinct from GPUT-09's Poisson/Bayesian/Bernoulli), matching the CPU path ≤1e-4, oracle-tested on Kaggle CUDA. (§6.1 `mvs.{cu,cuh}`.)
- [x] **GPUT-10**: CTR / permutation-dependent categorical features train on device.
- [x] **GPUT-11**: The **PairLogit** pairwise-loss training path (pairwise 2×2-cell histograms) runs on device. (Query/listwise objectives are GPUT-22; the batched solver is GPUT-21. §6.3 `pairwise_hist*`.)
- [ ] **GPUT-21**: Per-leaf **pairwise-derivative matrix assembly** (`MakePairwiseDerivatives` / `MakePointwiseDerivatives`) plus **batched device Cholesky** decomposition, forward/back substitution, ridge regularization, and score-from-decomposition (`CalcScoresCholesky`) run on device for pairwise split-scoring and leaf values, matching the CPU path ≤1e-4, oracle-tested on Kaggle CUDA. (§6.3 `split_pairwise.{cu,cuh}`, `linear_solver.{cu,cuh}`.)
- [x] **GPUT-22**: The **query-wise / listwise** objectives — QueryRMSE, QuerySoftMax, QueryCrossEntropy, YetiRank, PFound-F — with device query-grouping infrastructure (group ids/means/max, group-bias removal, in-query sampling radix sort, taken-docs masks) run on device, matching the CPU path ≤1e-4, oracle-tested on Kaggle CUDA. (Split out of the old over-broad GPUT-11. §6.5 `query_*.{cu,cuh}`, `yeti_rank_pointwise`, `pfound_f`; §6.6a `query_helper.cu`.)
- [x] **GPUT-12**: The **multiclass / multi-target / uncertainty** training path — MultiClass, OneVsAll (MultiClassOneVsAll), MultiCrossEntropy, MultiRMSE, RMSEWithUncertainty (multilogit multi-row der2 blocks) — runs on device. (§6.5 `multilogit.{cu,cuh}`.)
- [x] **GPUT-13**: Ordered boosting (`EBoostingType::Ordered`) trains on device.
- [ ] **GPUT-20**: **Stochastic Gradient Langevin Boosting** noise injection (`AddLangevinNoise`: per-element seeded Gaussian added to the reduced derivatives) runs on device, matching the CPU path ≤1e-4, oracle-tested on Kaggle CUDA. (§6.3 `langevin_utils.{cu,cuh}`.)

### GPU Standing Correctness Gate (GPUT)

- [ ] **GPUT-14**: Every device-covered training case holds ε=1e-4 vs the Rust CPU path, oracle-tested on Kaggle CUDA, and the CPU/host training paths remain byte-unchanged (D-04 no-regression) across the whole milestone.

### CUDA Oracle Harness & Performance Benchmark (BENCH)

- [x] **BENCH-01**: A reproducible Kaggle CUDA oracle/test harness — **established in Phase 10** and reused by every later phase — builds the `--features cuda` wheel and on a Kaggle CUDA notebook runs BOTH the GPU kernel **correctness** oracle (≤1e-5 for the depth-1 tree, ≤1e-4 for depth>1) AND a **wall-clock speed** measurement (warm-run/JIT-excluded, train-only), with correctness as a blocking gate before any speed number. From Phase 10 the harness measures BOTH correctness AND speed from the start. Authoritative GPU oracle; ROCm in-env is not a gate; human-gated external step.
- [x] **BENCH-02**: **Standing per-phase speed check** — first established in Phase 10 but enforced in EVERY GPU phase (analogous to how GPUT-14's ε=1e-4 gate is mapped to one phase yet enforced onward). Every phase that lands GPU kernels reports a Kaggle CUDA speed measurement for those kernels (device path vs the host-CPU baseline, and vs official CatBoost GPU where a comparable config exists), so speed is tracked incrementally across the whole milestone rather than only at the end. No phase's GPU kernels are considered done without a recorded CUDA speed check.
- [ ] **BENCH-03**: The device-resident training path demonstrably closes the >20× gap on Kaggle CUDA: a **comprehensive final** speed-parity sign-off vs official CatBoost GPU across the workload matrix is documented and signed off against the pre-Phase-10 host-light baseline, **aggregating the per-phase speed checks** (BENCH-02).

## Future Requirements (deferred)

- Multi-GPU / distributed device training (`ReduceBinary`, striped/mirrored `TCudaBuffer` mappings) — single-GPU only for v1.1.
- Device-side inference/predict acceleration beyond the existing `EnableGPUEvaluation` path (§7.1 GPU inference evaluator) — v1.1 is about *training* speed.
- On-device border/quantile computation (upstream `FastGpuBorders` / `ComputeQuantileBorders`) — host CPU quantization is the ≤1e-5 reference, uploaded once (GPUT-02/15); revisit only if host↔device quantization breaks parity.
- Autotuned per-backend kernel occupancy/cube-dim selection beyond hand-tuned defaults — opportunistic, not a v1.1 gate.

## Out of Scope (explicit exclusions)

- **Bit-exact f64 ≤1e-5 on GPU** — device math is f32; the GPU parity bar is ε=1e-4 vs the CPU path (D-04 precedent). Reason: chasing f64 bit-exactness on GPU is infeasible and not the goal.
- **ROCm as a correctness gate** — the in-env AMD/ROCm GPU is an optional compile/smoke convenience only; ALL GPU kernel oracles (correctness + speed) are validated on Kaggle CUDA. Reason: CUDA is the authoritative GPU target; no NVIDIA hardware in-env.
- **Replacing the CPU training path** — the CPU path stays the correctness reference and the device fallback. Reason: it is the oracle and the safety net behind the `Ok(None)` gate.
- **GPU inference evaluator (§7.1) / multi-GPU tree-reduce** — device predict beyond `EnableGPUEvaluation` and multi-GPU striped/mirrored reductions are deferred. Reason: v1.1 is single-GPU *training* speed.
- **DCG/NDCG as an eval metric** — an eval-surface transform, not a training gradient (its `RemoveGroupMean` step folds into GPUT-22's query infra). Reason: metric, not a device training deliverable.
- **FEAT-07 HNSW estimated-feature parity** — carried as deferred backlog (Phase 9), unrelated to GPU performance. Reason: separate correctness concern, its own future milestone.

## Traceability

*Re-mapped by the roadmapper during the 2026-07-02 re-scope. 25 requirements mapped across Phases 10–14: GPUT-01..22 (22) + BENCH-01..03 (3). 100% coverage, no orphans, no duplicates.*

| Requirement | Phase | Status |
|-------------|-------|--------|
| GPUT-01 | Phase 10 | Complete |
| GPUT-02 | Phase 10 | Complete |
| GPUT-03 | Phase 10 | Complete |
| GPUT-04 | Phase 10 | Complete |
| GPUT-08 | Phase 10 | Complete |
| GPUT-15 | Phase 10 | Complete |
| GPUT-16 | Phase 10 | Complete |
| BENCH-01 | Phase 10 | Complete |
| BENCH-02 | Phase 10 (standing — enforced in every GPU phase 10→13) | Complete |
| GPUT-05 | Phase 11 | Complete |
| GPUT-06 | Phase 11 | Complete |
| GPUT-07 | Phase 11 | Complete |
| GPUT-14 | Phase 11 (standing — enforced onward through 13) | Pending |
| GPUT-18 | Phase 12 | Complete |
| GPUT-19 | Phase 12 | Complete |
| GPUT-09 | Phase 12 | Complete |
| GPUT-17 | Phase 12 | Complete |
| GPUT-10 | Phase 12 | Complete |
| GPUT-11 | Phase 13 | Complete |
| GPUT-21 | Phase 13 | Pending |
| GPUT-22 | Phase 13 | Complete |
| GPUT-12 | Phase 13 | Complete |
| GPUT-13 | Phase 13 | Complete |
| GPUT-20 | Phase 13 | Pending |
| BENCH-03 | Phase 14 | Pending |

---
*Requirements defined: 2026-06-28 for milestone v1.1 GPU Performance*
*Re-scoped in place: 2026-07-02 against `CATBOOST_CUDA_KERNELS_DESIGN.md` — added GPUT-15 (cindex residency), GPUT-16 (device-primitive library, no CUB), GPUT-17 (MVS sampling), GPUT-18 (non-symmetric grow policies), GPUT-19 (exact leaf estimation), GPUT-20 (Langevin/SGLB), GPUT-21 (batched pairwise Cholesky solver), GPUT-22 (query/listwise ranking losses); narrowed GPUT-11 to PairLogit; widened GPUT-12 to multiclass/multi-target/uncertainty. Traceability re-populated across Phases 10–14 (5 phases).*
</content>
