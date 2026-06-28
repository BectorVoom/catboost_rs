# Requirements: catboost-rs — v1.1 GPU Performance

**Defined:** 2026-06-28
**Core Value:** A memory-efficient, Rust-native CatBoost implementation with verifiable feature parity (oracle-tested ≤10⁻⁵ on CPU; ε=1e-4 on GPU), embeddable in Rust and droppable into both scikit-learn and existing CatBoost Python pipelines.

> **Milestone goal:** Full CUDA device-resident training parity — move the entire boosting inner loop (histogram build, split scoring, BestSplit, partition/leaf-assignment, leaf values) onto the GPU, not just derivatives, reaching speed parity with official CatBoost GPU while preserving correctness. The >20× gap in v1.0 was the derivatives-only MVP: `grow_boosting_pass` (`crates/cb-backend/src/gpu_runtime/mod.rs:1890`) is rocm-validated but never wired into `cb_train::train`. Correctness is developed/validated in-env on AMD/ROCm (CubeCL kernels portable cuda/rocm/wgpu); the head-to-head speed benchmark vs official CatBoost runs on CUDA via a Kaggle notebook.
>
> **Parity bar:** GPU device path is held to **ε=1e-4 vs the Rust CPU path** (Phase 7.6 precedent — device math is f32; bit-exact f64 ≤1e-5 is not the GPU goal). The CPU path remains oracle-locked ≤10⁻⁵ and byte-unchanged (D-04 no-regression).
>
> **Landmine:** never add a `cb-train` dependency to `cb-backend` (Cargo feature unification breaks the rocm runtime) — transcribe CPU references inline; the `Runtime` seam stays CubeCL-free.

## v1.1 Requirements

### GPU Device-Resident Training (GPUT)

- [ ] **GPUT-01**: A `Runtime` grow-tree trait seam (`begin_device_training` / `grow_tree_on_device` returning `CbResult<Option<DeviceGrownTree>>` / `end_device_training`) exists in `cb-compute` with CubeCL-free host-typed signatures, and a `Ok(None)`→host-CPU fallback so any uncovered case stays correct.
- [ ] **GPUT-02**: A `GpuTrainSession` (cb-backend-internal) owns one `ComputeClient` + all persistent device handles for the whole fit; the quantized feature matrix is uploaded once above the iteration loop (no per-tree re-upload).
- [ ] **GPUT-03**: Gradients/approx stay device-resident across boosting iterations; the per-tree `der1` host read-back is eliminated; only the O(1) BestSplit descriptor + `2^depth` partition statistics cross host↔device per level (D-05).
- [ ] **GPUT-04**: A depth-1 oblivious tree is grown fully on device (RMSE/Logloss, Plain boosting, fold_count=1) and matches the CPU path ≤1e-5 on the rocm in-env GPU.
- [ ] **GPUT-05**: Partition-aware histograms (`fullPass=false`) keyed by leaf, contiguous partition reorder, and the histogram subtraction trick support depth>1 trees on device.
- [ ] **GPUT-06**: A chosen reduction-determinism strategy (atomic-free / fixed-point as required for gfx1100's lack of f64 atomic-add) keeps device histogram/score reductions within ε=1e-4 of the CPU path across hundreds of trees.
- [ ] **GPUT-07**: Newton der2 leaf estimation runs on device (required for classification / Logloss default).
- [ ] **GPUT-08**: The Cosine / second-order score function (the GPU default) runs on device.
- [ ] **GPUT-09**: Bootstrap + random-strength sampling runs on device (sampling parity for non-default `bootstrap_type`).
- [ ] **GPUT-10**: CTR / permutation-dependent categorical features train on device.
- [ ] **GPUT-11**: The pairwise/ranking loss training path runs on device.
- [ ] **GPUT-12**: The multiclass training path runs on device.
- [ ] **GPUT-13**: Ordered boosting (`EBoostingType::Ordered`) trains on device.
- [ ] **GPUT-14**: Every device-covered training case holds ε=1e-4 vs the Rust CPU path on the rocm in-env GPU, and the CPU/host training paths remain byte-unchanged (D-04 no-regression) across the whole milestone.

### Performance Benchmark (BENCH)

- [ ] **BENCH-01**: A reproducible Kaggle CUDA benchmark harness times official CatBoost GPU vs catboost-rs on identical dataset/params, with warm-run/JIT exclusion and train-only (not I/O) wall-clock measurement.
- [ ] **BENCH-02**: The correctness oracle is re-run on the CUDA backend (≤1e-4 vs the Rust CPU path) as a blocking gate before any speed number is reported — closing the in-env-ROCm vs benchmark-CUDA validation asymmetry.
- [ ] **BENCH-03**: The device-resident training path demonstrably closes the >20× gap: a speed-parity target vs official CatBoost GPU is documented and signed off, and an in-env ROCm regression bench confirms the device path beats the pre-Phase-10 host-light baseline.

## Future Requirements (deferred)

- Multi-GPU / distributed device training — single-GPU only for v1.1.
- Device-side inference/predict acceleration beyond the existing `EnableGPUEvaluation` path — v1.1 is about *training* speed.
- Autotuned per-backend kernel occupancy/cube-dim selection beyond hand-tuned defaults — opportunistic, not a v1.1 gate.

## Out of Scope (explicit exclusions)

- **Bit-exact f64 ≤1e-5 on GPU** — device math is f32; the GPU parity bar is ε=1e-4 vs the CPU path (D-04 precedent). Reason: chasing f64 bit-exactness on GPU is infeasible and not the goal.
- **CUDA in-env validation** — no NVIDIA hardware in-env; CUDA correctness + speed are validated on Kaggle. Reason: environment constraint.
- **Replacing the CPU training path** — the CPU path stays the correctness reference and the device fallback. Reason: it is the oracle and the safety net behind the `Ok(None)` gate.
- **FEAT-07 HNSW estimated-feature parity** — carried as deferred backlog (Phase 9), unrelated to GPU performance. Reason: separate correctness concern, its own future milestone.

## Traceability

<!-- Filled by the roadmapper: REQ-ID → Phase → Status -->

| Requirement | Phase | Status |
|-------------|-------|--------|
| _(populated by roadmap)_ | | |

---
*Requirements defined: 2026-06-28 for milestone v1.1 GPU Performance*
