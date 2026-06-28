# Project Research Summary

**Project:** catboost-rs v1.1 GPU Performance
**Domain:** Device-resident GPU gradient-boosting training (CubeCL) + speed-parity benchmarking vs official CatBoost GPU
**Researched:** 2026-06-28
**Confidence:** HIGH

## Executive Summary

The v1.1 milestone is a wiring + residency + kernel-coverage problem, not a new-technology problem. All four research agents converged on the same diagnosis: the >20x gap exists because `grow_boosting_pass` (`cb-backend/src/gpu_runtime/mod.rs:1890`) is a rocm-validated, depth-1 device grow loop that is never called from `cb_train::train` — there is no `Runtime` trait seam for on-device tree growth, so every GPU fit falls through to the host CPU growers. The derivative kernels run on-device, but then their output is read back to host per tree so the host can run ~95% of training (histogram build, split scoring, BestSplit, partition update, leaf values). The result is slower than pure CPU. The fix is not a new crate or new algorithm; it is adding one coarse trait method to `cb_compute::Runtime` and wiring the existing driver into `cb_train::train`. CubeCL 0.10.0 already provides every primitive required: persistent buffer handles, device atomics, scan/reduce, lazy async dispatch, and per-block carry for the multi-block scan.

The recommended execution order is: Phase 10 adds the coarse `Runtime` seam + `GpuTrainSession` (a new `cb-backend`-owned struct that holds the `ComputeClient` and all persistent handles for the whole fit), wires depth-1, and hoists the quantized-feature-matrix upload above the iteration loop — this alone closes most of the gap for the simplest workloads. Phase 11 adds partition-aware histograms (`fullPass=false`) for depth>1 and Newton der2 leaf estimation — the single largest kernel extension, gating real workloads (depth 6, Logloss). Phase 12 extends GPU coverage behind the same `Ok(None)`→host-CPU fallback gate (CTR, pairwise, ordered boosting, multiclass) in any sub-order. Phase 13 runs the Kaggle CUDA head-to-head, but must re-run the correctness oracle on CUDA before trusting timing numbers.

The central tensions that must be resolved early are: (1) reduction non-determinism vs the parity oracle — parallel atomics produce non-associative sums that compound over hundreds of trees and can flip splits, so the reduction strategy must be chosen in Phase 10/11 before the partition-aware histogram kernel is written (the `ε=1e-4 vs Rust CPU path` precedent from Phase 7.6 applies to GPU, not ≤1e-5); (2) the validation asymmetry — correctness is validated in-env on AMD/ROCm, but the head-to-head speed benchmark runs on CUDA (Kaggle), making the CUDA path effectively unchecked for correctness until the benchmark phase; (3) three repo-specific landmines that silently corrupt the build: never add `cb-train` as a dependency of `cb-backend` (Cargo feature unification breaks the rocm runtime), never use `-inf` float literals inside `#[cube]` kernels (HIP JIT reject on gfx1100, invisible to cpu/wgpu cargo check), and never read a `Handle` through a client other than the one that allocated it.

## Key Findings

### Recommended Stack

No new compute crates are needed. CubeCL 0.10.0 is already pinned in `Cargo.toml:38` and already exposes every required primitive. The additive additions are: optional `profile-tracy`/`tracing` features on the existing CubeCL dep (gated behind a `profiling` Cargo feature), `tracing-subscriber 0.3.x` as an optional dev dep, and `criterion 0.7.x` in `[dev-dependencies]` for Rust-side regression timing on ROCm. The Python benchmark harness uses `catboost==1.2.10` (already in `.venv`) and the existing `benchmark.py` pattern. The CUDA speed run requires a `--features cuda` wheel built via maturin and run on a Kaggle notebook — there is no NVIDIA in-env.

**Core technologies (retain — do not re-add):**
- `cubecl 0.10.0`: GPU kernel authoring + ComputeClient device memory/dispatch — already drives all Phase 7.x kernels; every residency primitive is in this version.
- `SelectedRuntime` compile-time alias: zero-cost backend switching (cuda/rocm/wgpu/cpu) from one kernel source — must not be broken.
- `bytemuck` (workspace-pinned): zero-copy host↔device byte casts for the O(1) per-level read-backs — already the established idiom.
- `catboost==1.2.10` (Python, already in `.venv`): official GPU baseline for the Kaggle speed run.

**New tooling (dev/profiling only):**
- `cubecl` features `profile-tracy` + `tracing`: nanosecond GPU kernel/JIT/alloc profiling — gate behind a `profiling` Cargo feature, never in release build.
- `criterion 0.7.x` (`[dev-dependencies]`): warm-run Rust benchmark statistics for ROCm relative-timing regression during development.

### Expected Features

**Must have (table stakes — the "GPU beats CPU" milestone, P1):**
- `Runtime` grow-tree trait seam in `cb-compute` — without it no device grow loop is reachable from `fit()`; the entire milestone gate.
- Wire depth-1 `grow_boosting_pass` into `cb_train::train` with a typed `Ok(None)`→host fallback.
- Device-resident compressed index — upload the quantized feature matrix once above the iteration loop; eliminates the dominant PCIe re-copy.
- Gradients/approx device-resident across iterations — eliminate the per-tree `der1` read-back and approx re-upload.
- Partition-aware histograms (`fullPass=false`) + contiguous partition reorder — the keystone new kernel work; unlocks depth>1, and transitively Newton/CTR/multiclass. The single largest piece of the milestone.
- Histogram subtraction trick — halves histogram work at every level; required for depth>1 to approach parity speed.
- Correctness sign-off at ε=1e-4 vs Rust CPU path (Phase 7.6 precedent) and a benchmark harness vs official CatBoost GPU on Kaggle/CUDA.

**Should have (P2 — broader coverage once depth>1 is solid):**
- Newton der2 leaf estimation on device — required for classification (Logloss default); reuses Phase 7.2 der2 handles.
- Second-order/Cosine score function on device — GPU default is Cosine not L2 (known parity gap).
- Bootstrap + random-strength noise on device — sampling parity for non-default `bootstrap_type`.
- CTR / permutation-dependent features on device — categorical workloads; large, defer until numeric depth>1 is solid.

**Defer (P3 / v2+):**
- Pairwise/ranking device path, multiclass device path, ordered-boosting device path, multi-GPU.

**Explicit anti-features (never build):**
- Per-tree re-upload of the compressed index; bulk histogram/partition read-backs from device (only O(1) BestSplit + `2^depth` part-stats cross per level); chasing bit-exact f64 ≤1e-5 on GPU; `cb-train` dependency in `cb-backend`.

### Architecture Approach

One coarse optional `Runtime` seam: three default-impl methods added to `cb_compute::Runtime` (`begin_device_training`, `grow_tree_on_device` returning `CbResult<Option<DeviceGrownTree>>`, `end_device_training`). No CubeCL types appear in the trait signature — all arguments and return types are plain host structs in `cb-compute`, mirroring exactly how `Derivatives` already crosses the seam today. Fine-grained per-stage seams were explicitly rejected by all agents: they force either CubeCL handles across the boundary (violating D-03) or per-stage bulk read-backs (destroying residency). The upstream reference (`catboost-master/catboost/cuda/methods/doc_parallel_pointwise_oblivious_tree.h`) independently confirms the coarse-seam design — upstream uses a coarse weak-learner template, not stage callbacks.

**Major components:**
1. `cb_compute::Runtime` trait (MODIFY) — three default-impl grow-tree methods; stays CubeCL-free; only change visible to `cb-train`.
2. `GpuTrainSession` (NEW in `cb-backend`) — owns ONE `ComputeClient` + all persistent handles (`compressed_index_h`, `indices_h`, `target_h`, `weight_h`, `approx_h`, `der1_h`, `bins_h`, `part_stats_h`); created once per `fit()`, owned by `GpuBackend` via `RefCell<Option<GpuTrainSession>>`.
3. `GpuBackend` (`cb-backend/src/gpu_backend.rs`) (MODIFY) — implement the seam, own the session, drive `grow_boosting_pass_into` over resident handles.
4. `grow_boosting_pass_into` / `grow_oblivious_tree_into` (`cb-backend/src/gpu_runtime/mod.rs`) (MODIFY) — convert `*_into` launchers from host-slice args to handle-based args; add partition-aware histogram variant for depth>1.
5. `train_inner::<R: Runtime>` (`cb-train/src/boosting.rs`) (MODIFY) — try device seam, fall back to host on `Ok(None)`; call begin/end around the loop.
6. Host growers (`cb-train/src/tree.rs`) (KEEP UNCHANGED) — remain the CPU path and the device fallback.

**Crossing contract (D-05, enforced per level):** only the O(1) `BestSplit` descriptor and the `2^depth` `TPartitionStatistics` cross host↔device per level; histogram, partition, per-doc routing stays device-resident.

### Critical Pitfalls

1. **Inner loop left on host** — the exact v1.0 state; the `Runtime` seam is the fix; treat GPU train time ≥ CPU train time as a hard failure gate.
2. **Per-tree/level blocking read-backs** — `read_one` inside the per-level loop drains the CubeCL queue; eliminate per-tree der read-back, enforce O(1) metadata crossings only.
3. **Per-tree re-upload of training data** — re-uploading the immutable `cindex` per tree is the dominant discrete-GPU cost; `GpuTrainSession` hoists the upload once.
4. **Non-deterministic float reduction breaks the parity oracle** — `atomicAdd` non-associativity compounds over hundreds of trees; the reduction strategy must be chosen before writing the partition-aware histogram kernel; hold GPU to `ε=1e-4 vs Rust CPU`, never widen silently.
5. **f64 atomic-add unavailable on gfx1100/RDNA3** — falls back to `HostSumFallback`, destroying device residency on ROCm, invisible to cpu/wgpu cargo check; design the histogram reduction to be atomic-free.
6. **HIP rejects `-inf` literals in `#[cube]` kernels** — use `f32::MIN` sentinel; add to every kernel-authoring checklist.
7. **`cb-backend`→`cb-train` dependency landmine** — Cargo feature unification breaks ROCm backend selection; transcribe CPU reference logic inline, never add the edge.
8. **CUDA correctness untested before timing** — CUDA path first executes on Kaggle; oracle re-run on CUDA is a blocking gate before quoting any speed number.

## Implications for Roadmap

### Phase 10: Coarse Seam + GpuTrainSession Residency + Wire Depth-1

**Rationale:** Everything else is blocked on the seam; the residency architecture must be established here because retrofitting it after depth>1 kernels are written is significantly harder; the `Ok(None)` fallback pattern must be established here so all subsequent phases land incrementally without breaking correctness.

**Delivers:** `Runtime` trait with three default-impl grow-tree methods (plain host types, no CubeCL in signature); `GpuTrainSession` with all persistent handles; `*_into` launchers converted to handle-based args; `grow_boosting_pass` wired for the MVP envelope (depth=1, Plain, fold_count=1, RMSE/Logloss) with `Ok(None)` fallback; per-tree `der1` read-back eliminated; approx device-resident across iterations. Oracle: depth-1 RMSE ≤1e-5 vs CPU on rocm in-env; CPU/host paths byte-unchanged (D-04).

**Addresses features:** Runtime grow-tree seam, wire depth-1, compressed index resident, gradients/approx resident (all P1).

**Avoids pitfalls:** 1 (inner loop on host), 2 (blocking read-backs), 3 (per-tree re-upload), 7 (landmine).

**Research flag:** Standard patterns — seam shape fully specified; mirrors established `Derivatives` seam and `GpuBackend` patterns from Phases 7-8. Sub-task: spike the reduction-determinism strategy (fixed-point i64 atomics, private-histogram merge, or two-pass segmented reduce) in Phase 10 before Phase 11's histogram kernel is written.

### Phase 11: Depth>1 Partition-Aware Histograms + Newton Der2

**Rationale:** Depth>1 is the keystone — depth 6 is the real-world default; the subtraction trick is required for parity speed at depth 6; Newton der2 is required for classification. Partition-aware histograms and contiguous partition reorder are co-dependent (histogram fill needs contiguous partitions) so they land together.

**Delivers:** Partition-aware `pointwise_hist2` variant keyed by `leaf_of[obj]` into `2^level` slots; contiguous partition reorder (`TDataPartition{Offset,Size}` layout); histogram subtraction trick (parent-resident, sibling-by-subtraction); Newton der2 leaf estimation using Phase 7.2 der2 handles; chosen reduction-determinism strategy implemented. Oracle: depth>1 trees (RMSE + Logloss) ≤1e-4 vs CPU on rocm in-env.

**Addresses features:** Partition-aware histograms/depth>1, partition reorder, histogram subtraction trick, Newton der2 leaf estimation, second-order/Cosine score (partial) — all P1/P2.

**Avoids pitfalls:** 4 (non-deterministic reduction), 5 (f64 atomic-add unavailable), 6 (HIP -inf sentinel).

**Research flag:** Needs a focused spike on reduction-determinism strategy before the histogram kernel is written. Also verify the multi-block scan carry ("Open Q2" forward dependency in the CubeCL manual) against the vendored manual before implementing.

### Phase 12: GPU Coverage Expansion (CTR / Pairwise / Ordered / Multiclass)

**Rationale:** Each feature family is independently shippable behind the `Ok(None)` fallback gate; they do not depend on each other. Recommended sub-order: bootstrap + random-strength (small, high-return), then CTR (headline use case), then pairwise (reuses Phase 7.4 kernels), then multiclass, then ordered boosting (heaviest residency). Each sub-feature lands when it passes ≤1e-4 oracle sign-off; users fall back to CPU otherwise.

**Delivers:** Feature-by-feature transition from `Ok(None)`→CPU-fallback to `Ok(Some(tree))`→device path; GPU coverage matrix documented.

**Research flag:** CTR on device has the highest uncertainty — consider a targeted research spike on `batch_binarized_ctr_calcer.h` + `ctrs/` before planning that sub-task. Pairwise partition + leaves oracle (`leaves_estimation/pairwise_oracle.h`) is under-documented relative to the pointwise path.

### Phase 13: Kaggle CUDA Benchmark + Correctness Re-Run + ε Sign-Off

**Rationale:** Benchmark is only meaningful after depth>1 is device-resident. The CUDA path is untested for correctness until this phase — oracle re-run on CUDA is a blocking gate before any speed number is quoted.

**Delivers:** CUDA oracle re-run (≤1e-4 vs Rust CPU) as blocking gate; fair head-to-head timing (warm-run, train-only, identical params/bins/data, single GPU); throughput report; final ε sign-off documentation (D-04); ROCm regression criterion bench confirming device-resident path beats pre-Phase-10 host-light.

**Research flag:** Standard patterns — protocol fully specified in STACK.md and PITFALLS.md. Execution checklist: verify CUDA backend active via `nvidia-smi`, warm one untimed fit, drain lazy CubeCL queue with a read-back/predict before stopping the clock, re-run oracle before timing.

### Phase Ordering Rationale

- Phase 10 is a strict prerequisite: the seam gates the device path; the residency architecture gates the speedup; the fallback pattern gates safe incremental coverage.
- Phase 11 is the performance keystone: depth>1 (default depth 6) is the first workload where GPU can plausibly beat CPU; the subtraction trick is also required to approach parity speed.
- Phase 12 is inherently parallel: each feature family is independently gated and deferrable; can be planned and executed in parallel sub-workstreams, or cut to bootstrap + CTR MVP if Phase 11 runs long.
- Phase 13 cannot start until CUDA correctness is established; oracle re-run is not optional.

### Research Flags

**Needs deeper research during planning:**
- Phase 11 (reduction-determinism strategy): fixed-point i64 atomics vs private-histogram merge vs two-pass segmented reduce — must be spiked before the histogram kernel is written.
- Phase 11 (multi-block scan carry): "Open Q2" in the CubeCL manual — verify against vendored docs before implementing.
- Phase 12 (CTR on device): complex upstream `batch_binarized_ctr_calcer.h` pipeline warrants a targeted research sub-task.

**Standard patterns (skip additional research):**
- Phase 10 (seam + session): architecture fully specified; mirrors established patterns from Phases 7-8.
- Phase 13 (benchmark harness): protocol fully specified; existing `benchmark.py` is the template.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | CubeCL 0.10.0 already in repo; all APIs verified against vendored manual and existing `gpu_runtime` code; no new compute crates needed. |
| Features | HIGH | Grounded in direct read of `catboost-master/catboost/cuda/` upstream reference; dependency graph validated against existing Rust kernel surface. |
| Architecture | HIGH | Coarse-seam + `GpuTrainSession` design independently confirmed by all three agents and by the upstream `TBoosting<TTarget, TWeakLearner>` template. |
| Pitfalls | HIGH | Landmines 6 and 7 are in-repo validated (Phase 7.2/7.5/7.6 retrospectives); reduction non-determinism corroborated by GPU GBDT literature. |

**Overall confidence:** HIGH

### Gaps to Address

- **Reduction-determinism strategy (Phase 11):** The right choice depends on gfx1100 performance characteristics only measurable by a targeted spike. Plan a spike sub-task in Phase 10 or as Phase 11 step 0.
- **CUDA correctness before benchmark (Phase 13):** No pre-flight validation of CUDA-specific behavior before the benchmark phase; the oracle re-run on CUDA is designed to catch this; the `Ok(None)` fallback ensures CPU correctness as the safety net.
- **Pairwise device path (Phase 12):** The pairwise partition + leaves oracle is less documented in this research; plan a targeted read of `leaves_estimation/pairwise_oracle.h` before implementation.
- **Occupancy tuning per backend (Phase 12):** Cube/plane dim tuning for new kernel families is only measurable in-env; budget profiling time per new kernel family.

## Sources

### Primary (HIGH confidence)
- In-repo: `crates/cb-backend/src/gpu_runtime/mod.rs` (grow loop, `*_into` launchers, depth-1 MVP), `crates/cb-backend/src/gpu_runtime/der_seams.rs`, `crates/cb-compute/src/runtime.rs`, `crates/cb-train/src/boosting.rs`, `crates/cb-backend/src/gpu_backend.rs`.
- Upstream CUDA reference: `catboost-master/catboost/cuda/methods/oblivious_tree_doc_parallel_structure_searcher.{h,cpp}`, `methods/doc_parallel_boosting.h`, `methods/pointwise_optimization_subsets.h`, `methods/histograms_helper.h`, `methods/pointwise_scores_calcer.h`, `methods/leaves_estimation/`, `gpu_data/compressed_index.h`, `gpu_data/gpu_structures.h`.
- CubeCL vendored manual: `11_launch_overhead_and_transfers.md`, `05_lazy_execution.md`, `08_atomic_contention.md`, `09_fixedpoint_atomics.md`, `10_grid_stride_occupancy.md`, `profiling_tools.md`, `Cubecl_plane.md`, `04_autotune_optimization.md`.
- `.planning/notes/gpu-training-host-light-root-cause.md` — root-cause analysis of the >20x gap.
- Project memory: `phase75-grow-loop-outcome`, `phase76-gpu-tolerance-signoff-outcome`, `cubecl-hip-no-inf-literal`, `phase8-python-bindings-outcome`.

### Secondary (MEDIUM confidence)
- [XGBoost GPU docs — FP non-associativity in GPU ranking](https://xgboost.readthedocs.io/en/release_1.4.0/gpu/index.html)
- [GPU-acceleration for Large-scale Tree Boosting (arXiv 1706.08359)](https://arxiv.org/pdf/1706.08359)
- [Quantized Training of Gradient Boosting Decision Trees (arXiv 2207.09682)](https://arxiv.org/pdf/2207.09682)
- [CatBoost GPU vs CPU benchmark methodology](https://github.com/catboost/benchmarks/blob/master/gpu_vs_cpu_training_speed/README.md)

---
*Research completed: 2026-06-28*
*Ready for roadmap: yes*
