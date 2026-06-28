# Requirements: catboost-rs — v1.1 GPU Performance

**Defined:** 2026-06-28
**Core Value:** A memory-efficient, Rust-native CatBoost implementation with verifiable feature parity (oracle-tested ≤10⁻⁵ on CPU; ε=1e-4 on GPU), embeddable in Rust and droppable into both scikit-learn and existing CatBoost Python pipelines.

> **Milestone goal:** Full CUDA device-resident training parity — move the entire boosting inner loop (histogram build, split scoring, BestSplit, partition/leaf-assignment, leaf values) onto the GPU, not just derivatives, reaching speed parity with official CatBoost GPU while preserving correctness. The >20× gap in v1.0 was the derivatives-only MVP: `grow_boosting_pass` (`crates/cb-backend/src/gpu_runtime/mod.rs:1890`) exists but is never wired into `cb_train::train`.
>
> **Validation authority — ALL GPU (CUDA) kernel oracles, correctness AND speed, run on a Kaggle CUDA notebook.** CUDA is the single authoritative GPU oracle for this milestone. A reproducible Kaggle CUDA oracle/test harness is a **foundational deliverable established in Phase 10** (BENCH-01) that measures BOTH correctness AND wall-clock speed. **Speed is checked for every GPU kernel from the first phase to the last** — from Phase 10 onward, every phase that lands GPU kernels reports a Kaggle CUDA speed measurement (device path vs the host-CPU baseline, and vs official CatBoost GPU where a comparable config exists) alongside its correctness oracle. Speed is NOT deferred to a single end-of-milestone benchmark; Phase 13 is the *comprehensive final* parity sign-off, not the first place speed is measured. There is no NVIDIA hardware in-env; the AMD/ROCm in-env GPU remains an OPTIONAL compile/smoke convenience for fast local iteration, but it is **not a gate** — no requirement is satisfied by ROCm validation alone. This eliminates the prior ROCm-correctness / CUDA-speed asymmetry. A Kaggle CUDA oracle/speed run is a **human-gated external step** (the user runs the notebook).
>
> **Parity bar:** GPU device path is held to **ε=1e-4 vs the Rust CPU path** on CUDA (Phase 7.6 precedent — device math is f32; bit-exact f64 ≤1e-5 is not the GPU goal), with the depth-1 device tree held tighter at ≤1e-5 where the level-0 whole-dataset histogram is the exact CPU score. The CPU path remains oracle-locked ≤10⁻⁵ and byte-unchanged (D-04 no-regression).
>
> **Landmine:** never add a `cb-train` dependency to `cb-backend` (Cargo feature unification breaks the rocm runtime) — transcribe CPU references inline; the `Runtime` seam stays CubeCL-free. Kernels remain CubeCL-portable (cuda/rocm/wgpu) so ROCm smoke-testing stays possible, but CUDA on Kaggle is the oracle of record. Note: CUDA provides f64 atomic-add (unlike gfx1100), so the atomic-free constraint is a portability nicety rather than a hard gate — but parallel-reduction **determinism** still governs the ε=1e-4 parity bar, so a deterministic reduction strategy is still required.

## v1.1 Requirements

### GPU Device-Resident Training (GPUT)

- [ ] **GPUT-01**: A `Runtime` grow-tree trait seam (`begin_device_training` / `grow_tree_on_device` returning `CbResult<Option<DeviceGrownTree>>` / `end_device_training`) exists in `cb-compute` with CubeCL-free host-typed signatures, and a `Ok(None)`→host-CPU fallback so any uncovered case stays correct.
- [ ] **GPUT-02**: A `GpuTrainSession` (cb-backend-internal) owns one `ComputeClient` + all persistent device handles for the whole fit; the quantized feature matrix is uploaded once above the iteration loop (no per-tree re-upload).
- [ ] **GPUT-03**: Gradients/approx stay device-resident across boosting iterations; the per-tree `der1` host read-back is eliminated; only the O(1) BestSplit descriptor + `2^depth` partition statistics cross host↔device per level (D-05).
- [ ] **GPUT-04**: A depth-1 oblivious tree is grown fully on device (RMSE/Logloss, Plain boosting, fold_count=1) and matches the CPU path ≤1e-5, oracle-tested on Kaggle CUDA.
- [ ] **GPUT-05**: Partition-aware histograms (`fullPass=false`) keyed by leaf, contiguous partition reorder, and the histogram subtraction trick support depth>1 trees on device.
- [ ] **GPUT-06**: A chosen reduction-determinism strategy keeps device histogram/score reductions within ε=1e-4 of the CPU path across hundreds of trees, verified on Kaggle CUDA (CUDA has f64 atomic-add, but atomicAdd ordering is still non-deterministic, so a deterministic reduction is required).
- [ ] **GPUT-07**: Newton der2 leaf estimation runs on device (required for classification / Logloss default).
- [ ] **GPUT-08**: The Cosine / second-order score function (the GPU default) runs on device.
- [ ] **GPUT-09**: Bootstrap + random-strength sampling runs on device (sampling parity for non-default `bootstrap_type`).
- [ ] **GPUT-10**: CTR / permutation-dependent categorical features train on device.
- [ ] **GPUT-11**: The pairwise/ranking loss training path runs on device.
- [ ] **GPUT-12**: The multiclass training path runs on device.
- [ ] **GPUT-13**: Ordered boosting (`EBoostingType::Ordered`) trains on device.
- [ ] **GPUT-14**: Every device-covered training case holds ε=1e-4 vs the Rust CPU path, oracle-tested on Kaggle CUDA, and the CPU/host training paths remain byte-unchanged (D-04 no-regression) across the whole milestone.

### CUDA Oracle Harness & Performance Benchmark (BENCH)

- [ ] **BENCH-01**: A reproducible Kaggle CUDA oracle/test harness — **established in Phase 10** and reused by every later phase — builds the `--features cuda` wheel and on a Kaggle CUDA notebook runs BOTH the GPU kernel **correctness** oracle (≤1e-5 for the depth-1 tree, ≤1e-4 for depth>1) AND a **wall-clock speed** measurement (warm-run/JIT-excluded, train-only), with correctness as a blocking gate before any speed number. From Phase 10 the harness measures BOTH correctness AND speed from the start. Authoritative GPU oracle; ROCm in-env is not a gate; human-gated external step.
- [ ] **BENCH-02**: **Standing per-phase speed check** — first established in Phase 10 but enforced in EVERY phase 10→13 (analogous to how GPUT-14's ε=1e-4 gate is mapped to one phase yet enforced onward). From Phase 10 to the last phase, every phase that lands GPU kernels reports a Kaggle CUDA speed measurement for those kernels (device path vs the host-CPU baseline, and vs official CatBoost GPU where a comparable config exists), so speed is tracked incrementally across the whole milestone rather than only at the end. No phase's GPU kernels are considered done without a recorded CUDA speed check.
- [ ] **BENCH-03**: The device-resident training path demonstrably closes the >20× gap on Kaggle CUDA: a **comprehensive final** speed-parity sign-off vs official CatBoost GPU across the workload matrix is documented and signed off against the pre-Phase-10 host-light baseline, **aggregating the per-phase speed checks** (BENCH-02) recorded in Phases 10–12.

## Future Requirements (deferred)

- Multi-GPU / distributed device training — single-GPU only for v1.1.
- Device-side inference/predict acceleration beyond the existing `EnableGPUEvaluation` path — v1.1 is about *training* speed.
- Autotuned per-backend kernel occupancy/cube-dim selection beyond hand-tuned defaults — opportunistic, not a v1.1 gate.

## Out of Scope (explicit exclusions)

- **Bit-exact f64 ≤1e-5 on GPU** — device math is f32; the GPU parity bar is ε=1e-4 vs the CPU path (D-04 precedent). Reason: chasing f64 bit-exactness on GPU is infeasible and not the goal.
- **ROCm as a correctness gate** — the in-env AMD/ROCm GPU is an optional compile/smoke convenience only; ALL GPU kernel oracles (correctness + speed) are validated on Kaggle CUDA. Reason: CUDA is the authoritative GPU target; no NVIDIA hardware in-env.
- **Replacing the CPU training path** — the CPU path stays the correctness reference and the device fallback. Reason: it is the oracle and the safety net behind the `Ok(None)` gate.
- **FEAT-07 HNSW estimated-feature parity** — carried as deferred backlog (Phase 9), unrelated to GPU performance. Reason: separate correctness concern, its own future milestone.

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| GPUT-01 | Phase 10 | Pending |
| GPUT-02 | Phase 10 | Pending |
| GPUT-03 | Phase 10 | Pending |
| GPUT-04 | Phase 10 | Pending |
| BENCH-01 | Phase 10 | Pending |
| BENCH-02 | Phase 10 (standing — enforced 10→13) | Pending |
| GPUT-05 | Phase 11 | Pending |
| GPUT-06 | Phase 11 | Pending |
| GPUT-07 | Phase 11 | Pending |
| GPUT-08 | Phase 11 | Pending |
| GPUT-14 | Phase 11 | Pending |
| GPUT-09 | Phase 12 | Pending |
| GPUT-10 | Phase 12 | Pending |
| GPUT-11 | Phase 12 | Pending |
| GPUT-12 | Phase 12 | Pending |
| GPUT-13 | Phase 12 | Pending |
| BENCH-03 | Phase 13 | Pending |

**Coverage:** 17/17 v1.1 requirements mapped (GPUT-01..14, BENCH-01..03) — no orphans, no duplicates. Phase 10 = 6 reqs (GPUT-01..04 + BENCH-01 + BENCH-02 standing), Phase 11 = 5 (GPUT-05/06/07/08/14), Phase 12 = 5 (GPUT-09..13), Phase 13 = 1 (BENCH-03). BENCH-02 is mapped to Phase 10 (where it is first established) but is a **standing per-phase speed check enforced in every phase 10→13**, mirroring the GPUT-14 standing-gate pattern; Phase 13's BENCH-03 aggregates those per-phase speed checks into the comprehensive final sign-off.

---
*Requirements defined: 2026-06-28 for milestone v1.1 GPU Performance*
*Traceability populated by roadmap: 2026-06-28 (revised — Kaggle CUDA single-oracle validation; BENCH-01 → Phase 10)*
*Traceability re-revised: 2026-06-28 (per-phase speed check — BENCH-02 → Phase 10 standing, enforced 10→13; Phase 13 = BENCH-03 aggregate only)*
