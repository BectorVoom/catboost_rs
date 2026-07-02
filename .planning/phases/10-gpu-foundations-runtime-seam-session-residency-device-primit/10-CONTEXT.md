# Phase 10: GPU Foundations — Runtime Seam, Session Residency, Device-Primitive Library, Compressed Index, Depth-1 + Kaggle CUDA Oracle & Speed Harness - Context

**Gathered:** 2026-07-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Lay the whole device-resident substrate the v1.1 milestone stands on. This phase delivers, together:

- The from-scratch **CubeCL device-primitive library** (GPUT-16, no CUB) — fill/transform, full + segmented prefix scan, reduce / segmented-reduce / reduce-by-key, radix sort + stable single-bit reorder, bit-compression, `TDataPartition` offset/size update, per-partition stat aggregation (`update_part_props`), with a deterministic reduction.
- The bit-packed **device-resident compressed index (cindex)** (GPUT-15) with `TCFeature` Offset/Shift/Mask/OneHot addressing, the single input every later histogram kernel consumes.
- The **`Runtime` grow-tree seam** (GPUT-01) wired into `cb_train::train`, plus **`GpuTrainSession` residency** (GPUT-02/03) — upload the quantized matrix once per `fit()`, keep gradients/approx device-resident, eliminate per-tree `der1` read-back.
- A **depth-1 oblivious tree** (GPUT-04, RMSE/Logloss, Plain, fold_count=1) grown fully on device with the **Cosine GPU-default score** (GPUT-08), matching the CPU path ≤1e-5.
- A reproducible **Kaggle CUDA harness** (BENCH-01) measuring BOTH correctness (blocking gate) AND wall-clock speed from day one, establishing the **standing per-phase speed check** (BENCH-02).

**Scope anchor — already LOCKED by ROADMAP/REQUIREMENTS (not re-decided in discussion):**
- Seam signatures: `begin_device_training` / `grow_tree_on_device → CbResult<Option<DeviceGrownTree>>` / `end_device_training`, CubeCL-free host-typed (GPUT-01).
- `GpuTrainSession` owns one `ComputeClient` + persistent handles for the whole fit; `RefCell<Option<…>>` on `GpuBackend`; upload-once, no per-tree re-upload (GPUT-02/03).
- `Ok(None)` → host-CPU fallback; per-fit **all-or-nothing** (D-10-01) — no mixing device-grown and CPU-grown trees in one model.
- ε bars: depth-1 device **≤1e-5**; everything else **ε=1e-4** vs Rust CPU path. CPU path stays oracle-locked ≤1e-5 and byte-unchanged (D-04 no-regression).
- Depth-1 is the MVP; **depth>1 returns `Ok(None)`** and falls back to host CPU grower (that's Phase 11).
- Logloss depth-1 pins the CPU-reference fixture to **first-order `calc_average` leaves** (Newton der2 is Phase 11).
- Only the O(1) BestSplit descriptor + `2^depth` partition stats cross host↔device per level (D-05).
- **Standing landmines:** never add a `cb-train` dep to `cb-backend` (transcribe CPU refs inline); no `-inf` float literals in `#[cube]` kernels (use `f32::MIN` sentinel); deterministic reduction required for ε=1e-4; never read a `Handle` through a client other than the one that allocated it.
- Depth-1 speed bar pinned to a **large-n** dataset (~1e5–1e6 rows, D-10-09) — depth-1 is the most launch-overhead-bound workload; device ≥ CPU is not achievable at `benchmark.py`'s 1000×20.
- All GPU oracles (correctness AND speed) are authoritative on **Kaggle CUDA**, a human-gated external run. ROCm in-env is an optional compile/smoke convenience, **not a gate**.

</domain>

<decisions>
## Implementation Decisions

### Device-Primitive Library Oracle (GPUT-16)
- **D-01 (tiered oracle):** Hybrid oracle strategy. **Standalone** Kaggle CUDA oracles for the high-risk / hard-to-isolate primitives — full scan, segmented scan, radix sort + stable single-bit reorder, reduce-by-key, per-partition stat aggregation (`update_part_props`). The **trivial** primitives (fill/transform gather-scatter + vector arithmetic, plain reduce) are covered transitively through the depth-1 tree + cindex end-to-end. Rationale: a broken scan/sort found in isolation is far cheaper than debugging it through a depth-6 histogram in a later phase; this substrate is the foundation every later kernel stands on.
- **D-02 (ground truth):** Standalone primitive oracles compare against a **self-contained serial CPU/numpy reference** computing the same generic primitive on the same random-seeded input — NOT upstream CatBoost/CUB fixtures. These are generic building blocks (a prefix-scan is a prefix-scan); a dead-simple serial reference is itself easy to trust and keeps the harness self-contained. (Note the no-`cb-train`-dep landmine: transcribe any CPU reference inline; don't reach across the crate boundary.)

### Reduction-Determinism Spike (feeds Phase 11)
- **D-03 (prototype + measure):** The spike **implements the top 2–3 candidate deterministic-reduction strategies** and measures both run-to-run correctness variance AND speed on Kaggle CUDA, then recommends — not a paper survey. The CUDA harness stands up this phase and a deterministic reduce is on SC-1's critical path anyway, so measuring now hard-de-risks Phase 11's ε=1e-4 gate. Candidate strategies to weigh include: fixed-order tree reduce, sequential block-then-host-final-sum (à la Phase 7.6 `HostSumFallback`), Kahan compensation, sorted-index accumulation.
- **D-04 (winner ships as the primitive):** The measured-best strategy **IS** the reduce / segmented-reduce / reduce-by-key implementation that lands in the Phase 10 primitive library — spike and deliverable are the same work, no throwaway reduce. The depth-1 tree + stat-agg exercise it end-to-end immediately, so the winner gets real validation on landing.

### Kaggle CUDA Harness (BENCH-01)
- **D-05 (form + fixtures):** A **committed `.ipynb` notebook** builds the `--features cuda` wheel, loads **repo-committed fixtures** (random-seeded inputs + CPU-path expected values), runs **correctness first (blocking gate)** then a **warm-run / JIT-excluded, train-only wall-clock speed** measurement, and prints a **structured report**. A notebook is the native Kaggle artifact and diffable; committed fixtures keep the CPU ≤1e-5 reference the pinned in-repo authority and make the human-gated run push-button.
- **D-06 (speed workload dataset):** A **seeded synthetic generator** (configurable `n_rows` / `n_features`, e.g. ~1e6 × 50, tunable above the launch-overhead break-even per D-10-09) produces BOTH the depth-1 ≤1e-5 correctness fixture and the large-n speed workload. No external Kaggle download; fully reproducible; correctness and speed share one generator. (No real named dataset like Higgs/Epsilon in Phase 10.)

### Compressed Index Packing (GPUT-15)
- **D-07 (exact packing from the start):** Replicate upstream `WriteCompressedIndex`'s **exact 32-bit bit-packed grouped layout** — `TCFeature` Offset/Shift/Mask/OneHot addressing packing multiple features per 32-bit word — from the start; do NOT ship a simpler one-value-per-slot layout first. Memory efficiency is a first-class project constraint, every later histogram kernel consumes this as THE input so its address arithmetic must be right once, and the CPU quantized layout is the ≤1e-4 oracle so correctness is checkable immediately regardless of packing complexity. Borders stay host (CPU quantization is the ≤1e-5 reference); only cindex packing/residency is the device deliverable.

### Claude's Discretion
- Wave decomposition/ordering (ROADMAP suggests: primitive library → cindex → seam+residency → depth-1+Cosine → Kaggle harness → reduction spike) — planner refines.
- Seam module placement in `cb-compute` (`runtime.rs` — mirrors the shipped `compute_gradients_grouped` default-impl pattern), `apply_leaf_delta` device-kernel scope, per-fit session lifecycle details, and the bin→border join (`border = feature_borders[feature][bin_id]`) — research/planning resolve, grounded in the reusable assets below.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### GPU kernel design authority (v1.1)
- `CATBOOST_CUDA_KERNELS_DESIGN.md` — the complete upstream CUDA training-kernel map (79 `.cu` + 77 `.cuh` across 9 kernel dirs), per-file processing flow, host/device split, I/O types, algorithms. Every v1.1 phase cites it. Specifically for Phase 10:
  - **§6.1 `cuda_util/kernel`** + **§6.2 `cuda_util/kernel/sort`** — the device-primitive library surface (GPUT-16; no CUB).
  - **§6.6a `gpu_data/kernel/binarize.cu`, `WriteCompressedIndex`** — the compressed-index bit-packing + `TCFeature` addressing (GPUT-15).

### Root-cause & milestone framing
- `.planning/notes/gpu-training-host-light-root-cause.md` — the >20× host-light gap root cause this milestone closes.
- `.planning/PROJECT.md` — Current Milestone (v1.1 GPU Performance) goal, target features, key context, landmines.
- `.planning/REQUIREMENTS.md` — GPUT-01/02/03/04/08/15/16, BENCH-01/02 requirement text + traceability.
- `.planning/ROADMAP.md` — Phase 10 Success Criteria, Notes, standing landmines, validation authority.

### Prior research (partial — predates the 2026-07-02 re-scope)
- `.planning/milestones/v1.1-rescope-2026-07-02-phases/10-coarse-runtime-grow-tree-seam-gputrainsession-residency-wire/10-RESEARCH.md` — seam / residency / depth-1 architecture **still valid**; but it predates GPUT-08/15/16, so re-plan/re-research must ADD the primitive-library, cindex, and Cosine-score scope.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets (~80% of depth-1 device machinery already exists, Phase 7-validated)
- `crates/cb-backend/src/gpu_runtime/mod.rs:1641` `grow_oblivious_tree_into` — grows a depth-1 device tree with L2/Cosine calcers (Phase 7.5, oracle-validated). Direct basis for GPUT-04/08.
- `crates/cb-backend/src/gpu_runtime/mod.rs:1890` `grow_boosting_pass` (+ `grow_boosting_pass_into` at :1920) — the existing-but-**UNWIRED** device grow loop to make reachable from `cb_train::train` (GPUT-01).
- `crates/cb-backend/src/gpu_runtime/der_seams.rs` — Phase 7.2 der1/der2 device seam (DerBinary/Unary/Param kernels, `const_der_handle`, no read-back `*_handle`); gradients device-resident (GPUT-03).
- `crates/cb-compute/src/runtime.rs:944` `compute_gradients_grouped` — the **default-impl seam pattern** the new `grow_tree_on_device` seam should mirror (GPUT-01).
- `crates/cb-backend/src/gpu_backend.rs:47` `pub struct GpuBackend;` — unit struct today; add `RefCell<Option<GpuTrainSession>>` for persistent handles (GPUT-02).
- `calc_average` leaf formula (first-order) — the depth-1 Logloss/RMSE CPU-reference leaf values (Newton is Phase 11).
- `crates/cb-backend/src/gpu_runtime/pairwise.rs`, `crates/cb-compute/src/histogram.rs` — existing histogram/scoring kernels the cindex will feed later.

### Established Patterns
- Generic runtime over `SelectedRuntime` (cpu/wgpu/cuda/rocm), no runtime dispatch — one impl, feature-gated.
- `Ok(None)` → CPU fallback keeps every increment oracle-safe (D-04 no-regression).
- Serial CPU/numpy self-oracle for GPU kernels (this phase's D-02) matches the project's oracle discipline.

### Integration Points
- Seam lives in `cb-compute` (`Runtime` trait); wired into `cb_train::train`'s per-tree loop; `GpuBackend`/`GpuTrainSession` implement it in `cb-backend`. The boundary crosses **plain host structs only** — the `Runtime` seam stays CubeCL-free (landmine: no `cb-train` dep in `cb-backend`).
- One small `apply_leaf_delta` device kernel keeps the approx-update on device across iterations.

</code_context>

<specifics>
## Specific Ideas

- Kaggle harness output is a **structured report** (per-primitive/per-tree pass/fail at the ε bar, then warm-run speed: device vs host-CPU baseline, and vs official CatBoost GPU where a comparable config exists).
- Synthetic generator is the single source for both correctness fixture and speed workload — parameterize `n_rows`/`n_features` so the depth-1 speed bar sits comfortably above launch-overhead break-even.
- Deterministic-reduction candidate set to prototype: fixed-order tree reduce, block-then-host-final-sum (Phase 7.6 `HostSumFallback` precedent), Kahan, sorted-index accumulation.

</specifics>

<deferred>
## Deferred Ideas

- Real named large datasets (Higgs / Epsilon) as a "realistic" speed cross-check — not in Phase 10; synthetic generator is the pinned bar. Revisit at Phase 14's comprehensive benchmark (BENCH-03) if a published-comparable number is wanted.
- On-device border/quantile computation (upstream `FastGpuBorders` / `ComputeQuantileBorders`) — explicitly out of scope; host CPU quantization is the ≤1e-5 reference, uploaded once. Revisit only if host↔device quantization breaks parity.
- Newton der2 leaf estimation, depth>1 partition-aware histograms + subtraction trick, GPUT-14 ε=1e-4 operative gate — Phase 11.
- The stale `.planning/spikes/MANIFEST.md` (Spike 001, online-CTR / Phase 5) is unrelated to Phase 10; no unpackaged Phase-10 spikes exist.

### Reviewed Todos (not folded)
None — no pending todos matched this phase's scope.

</deferred>

---

*Phase: 10-gpu-foundations-runtime-seam-session-residency-device-primitive-library-compressed-index-depth-1-kaggle-cuda-oracle-speed-harness*
*Context gathered: 2026-07-03*
