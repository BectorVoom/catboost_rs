# Phase 11: Depth>1 Partition-Aware Histograms + Reduction Determinism + Newton Der2 - Context

**Gathered:** 2026-07-03
**Status:** Ready for planning

<domain>
## Phase Boundary

The performance keystone of the v1.1 milestone: grow **real depth-6 workloads** — both **RMSE regression** and **Logloss classification** — fully on device within **ε=1e-4**, oracle-tested AND speed-measured on Kaggle CUDA. Delivers, together:

- **Partition-aware histograms** (GPUT-05) — `pointwise_hist2` keyed by `leaf_of[obj]` into `2^level` slots (`fullPass=false`), `TDataPartition{Offset,Size}` contiguous partition reorder, and the **histogram subtraction trick** (§1.4/§6.3/§6.4) supporting depth>1 oblivious trees on device.
- **Reduction determinism** (GPUT-06) — the Phase-10 spike's recommended strategy holds device histogram/score reductions within ε=1e-4 across hundreds of trees with no split flips compounding.
- **Newton der2 leaf estimation** (GPUT-07) — required for the classification / Logloss default; reuses the Phase 7.2 der2 handles.
- **GPUT-14 becomes the operative standing gate** — every device-covered case holds ε=1e-4 vs the Rust CPU path on Kaggle CUDA; CPU/host paths stay byte-unchanged (D-04). Depth-1 was held to ≤1e-5; reductions over hundreds of depth>1 trees move the bar to ε=1e-4.
- **Speed check (BENCH-02, standing)** — depth-6 RMSE and Logloss device training timed on Kaggle CUDA: device vs host-CPU baseline AND vs official CatBoost GPU (warm-run/JIT-excluded, train-only).

**Scope anchor — already LOCKED (carried forward, not re-decided in discussion):**
- **Reduction strategy CHOSEN by the Phase-10 spike (D-04 / SPIKE-REDUCTION.md §5b):** fixed-point `Atomic<u64>` accumulation (LDS privatization + fixed-point, scale `k=30`) for the **many-cubes-contend histogram accumulator**, with the **fixed-order f64 tree reduce** as the capability-fallback (backends lacking `Atomic<u64>` add report the downgrade, never silently switch). Per-segment reduces (segmented-reduce / reduce-by-key) ship the fixed-order f64 tree reduce. This is step 0 — consume it before the histogram kernel.
- **ε bars:** depth>1 device **ε=1e-4** vs the Rust CPU path (GPUT-14, operative standing gate from here through Phase 13); CPU path stays oracle-locked ≤1e-5 and byte-unchanged (D-04 no-regression).
- **Newton reuses Phase 7.2 der1/der2 device handles** (`der_seams.rs` — DerBinary/Unary/Param, `const_der_handle`, no read-back `*_handle`).
- **All-or-nothing per fit (D-10-01):** depth>1 now becomes device-covered (was `Ok(None)` in Phase 10); still no mixing device-grown and CPU-grown trees in one model. Uncovered cases fall back to the byte-unchanged host CPU grower.
- Only the O(1) BestSplit descriptor + `2^depth` partition stats cross host↔device per level (D-05).
- **Standing landmines:** never add a `cb-train` dep to `cb-backend` (transcribe CPU refs inline); no `-inf` float literals in `#[cube]` kernels (use `f32::MIN` sentinel); deterministic reduction mandatory (CUDA `atomicAdd` ordering still non-deterministic; gfx1100 lacks f64 atomic-add for the smoke path); never read a `Handle` through a client other than the one that allocated it.
- All GPU oracles (correctness AND speed) authoritative on **Kaggle CUDA**, a human-gated external run. ROCm in-env is an optional compile/smoke convenience, **not a gate**.

</domain>

<decisions>
## Implementation Decisions

### Newton der2 leaf estimation (GPUT-07)
- **D-01 (fully device-resident refinement loop):** The Logloss Newton refinement — `leaf_estimation_iterations` steps, each recomputing der1/der2 at the current approx and updating leaf values — runs **fully on device**. Reuse the Phase 7.2 der1/der2 handles + the `apply_leaf_delta` kernel; recompute ders per step on-device; **no per-iteration readback**. This preserves residency, which is the entire point of the milestone (per-iteration readback × hundreds of trees would undercut the speed goal).
- **D-02 (pin iteration count in the fixture):** Pin `leaf_estimation_iterations` from the model config and **freeze it in the CPU-reference fixture** so the device refinement matches the CPU reference exactly at the ε=1e-4 bar. (RMSE's der2 is the constant weight, so its Newton step is effectively single-step / trivial — the multi-step refinement is the Logloss path.)

### Depth-6 correctness fixture + CUDA speed workload
- **D-03 (reuse the Phase-10 synthetic generator):** Extend the Phase-10 seeded synthetic generator to **depth-6 RMSE + Logloss configs**; it produces BOTH the ≤1e-4 correctness fixture AND the large-n CUDA speed workload. Fully reproducible, no external download, one generator for correctness + speed. Real named datasets (Higgs/Epsilon) stay **deferred to Phase 14 (BENCH-03)** per the Phase-10 deferred list.

### Subtraction trick + histogram memory residency (GPUT-05)
- **D-04 (smaller-sibling-direct + parent-resident subtraction):** Compute the **smaller partition's histogram directly**, derive the larger sibling by **subtracting from the parent's resident histogram** (upstream leaf-wise builder, §6.4). Keep only **parent-level** histograms resident, not all levels. Memory-lean (memory efficiency is a first-class project constraint at depth 6: 64 leaves × features × bins × channels) and it is the speed lever that approaches parity — not always-materialize-both (~2× the histogram work).

### ε=1e-4 verification across the boosting run (GPUT-06 / SC-3 / SC-5)
- **D-05 (final ε gate + per-tree diagnostic):** Gate on **final-prediction ε=1e-4** across the full run (blocking) AND instrument a **per-tree split-agreement + run-to-run spread diagnostic** in the Kaggle oracle, so a compounding drift is caught at the tree where it starts — not just at the final aggregate. This directly evidences SC-3's "no split flips compounding over the boosting run" and gives a debugging locus if the bar is missed.

### Claude's Discretion
- **Sub-wave decomposition/ordering** — ROADMAP suggests: depth>1 histograms → reduction determinism → Newton der2. Planner refines (reduction spike winner is step 0 / already landed as the reduce primitive).
- **Newton leaf-estimation backtracking** (upstream `AnyImprovement` line-search on leaf estimation) — research flag: confirm whether the CPU reference uses backtracking at the pinned config; if so it must be mirrored on device for the ε bar. Planner/researcher resolves against the CPU path.
- Exact channel layout of the partition-aware `pointwise_hist2` (der1 + weight, plus der2 for Newton), the `2^level` slot addressing, and the contiguous `TDataPartition` reorder mechanics — grounded in the reusable assets below; research/planning resolve against §6.3/§6.4 and the Phase-10 primitives.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### GPU kernel design authority (v1.1)
- `CATBOOST_CUDA_KERNELS_DESIGN.md` — the complete upstream CUDA training-kernel map. Specifically for Phase 11:
  - **§1.4 subtraction trick** — the sibling-by-subtraction identity the depth>1 builder relies on.
  - **§6.3 `methods/kernel/pointwise_hist2`** — the partition-aware `fullPass=false` histogram kernel keyed by leaf.
  - **§6.4 `methods/greedy_subsets_searcher/kernel` leaf-wise builder** — parent-resident histogram + smaller-sibling-direct computation.
  - **`targets/kernel`** — Newton der2 leaf estimation reference (der1/der2, refinement).

### Phase-10 deliverables consumed as-is (step 0 + substrate)
- `.planning/phases/10-gpu-foundations-runtime-seam-session-residency-device-primit/SPIKE-REDUCTION.md` — **the reduction-determinism decision (D-04 winner)**: fixed-point `Atomic<u64>` accumulator (k=30) for the histogram, fixed-order tree-reduce fallback. Consume §5b as step 0 before the histogram kernel.
- `.planning/phases/10-.../10-CONTEXT.md` — Phase-10 locked scope (seam signatures, residency, cindex packing, all-or-nothing D-10-01, ε bars, landmines).
- `.planning/phases/10-.../10-RESEARCH.md` — seam / residency / depth-1 / primitive-library / cindex architecture.

### Root-cause & milestone framing
- `.planning/notes/gpu-training-host-light-root-cause.md` — the >20× host-light gap this milestone closes.
- `.planning/PROJECT.md` — Current Milestone (v1.1 GPU Performance) goal, target features, landmines.
- `.planning/REQUIREMENTS.md` — GPUT-05/06/07/14, BENCH-02 requirement text + traceability.
- `.planning/ROADMAP.md` — Phase 11 Success Criteria (1–5), Notes, standing landmines, validation authority (Kaggle CUDA).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets (Phase 7/10-validated)
- `crates/cb-backend/src/kernels/grow_loop.rs` — the depth-1 device grow loop: `launch_pointwise_hist2_handle`, `launch_partition_split_into` (per-object `leaf_of` forward-bit), `launch_partition_update_into` (per-partition Σ der1 / Σ weight reduce), `read_part_stats_f64`. The depth>1 partition-aware histogram + subtraction trick extends this.
- `crates/cb-compute/src/histogram.rs` — CPU-reference histogram scatter keyed by `leaf_of[i]` into per-leaf slots (der1, der2·weight, residuals variants) — the ≤1e-4 oracle for the device partition-aware histogram.
- `crates/cb-backend/src/gpu_runtime/der_seams.rs` — Phase 7.2 der1/**der2** device seam (DerBinary/Unary/Param, `const_der_handle`, no read-back `*_handle`) — Newton der2 reuses these handles (D-01).
- `crates/cb-backend/src/gpu_runtime/mod.rs:1641` `grow_oblivious_tree_into` / `:1890` `grow_boosting_pass` — the device grow machinery wired in Phase 10; depth>1 extends the per-level partition path.
- Phase-10 device-primitive library (scan / segmented-scan / reduce-by-key / partition-update / stat-aggregation) + the resident cindex — consumed directly by the depth>1 histogram/partition path.
- `apply_leaf_delta` device kernel (Phase 10) — reused to keep the Newton refinement approx-update on device (D-01).

### Established Patterns
- Generic runtime over `SelectedRuntime` (cpu/wgpu/cuda/rocm), no runtime dispatch — one feature-gated impl.
- `Ok(None)` → CPU fallback keeps every increment oracle-safe (D-04 no-regression); depth>1 flips from `Ok(None)` to covered this phase.
- Serial CPU self-oracle for GPU kernels; max abs/rel divergence over equal-length buffers at the ε bar (`grow_loop.rs` helper).
- Fixed-point `Atomic<u64>` deterministic accumulator + fixed-order tree-reduce fallback with explicit capability-path reporting (SPIKE-REDUCTION).

### Integration Points
- Depth>1 histogram/partition kernels live in `cb-backend` (`kernels/grow_loop.rs` + `gpu_runtime`), driven per-level through the `Runtime` grow-tree seam wired into `cb_train::train` in Phase 10. Boundary crosses **plain host structs only** (landmine: no `cb-train` dep in `cb-backend`).
- Newton refinement loop stays inside the device session (`GpuTrainSession`), reading der2 handles + writing leaf deltas via `apply_leaf_delta` — no host round-trip per iteration (D-01).

</code_context>

<specifics>
## Specific Ideas

- Partition-aware `pointwise_hist2` keyed by `leaf_of[obj]` into `2^level` slots; `TDataPartition{Offset,Size}` contiguous reorder; parent-resident sibling-by-subtraction (compute the smaller partition directly).
- Newton refinement loop fully device-resident: reuse Phase 7.2 der2 handles + `apply_leaf_delta`, recompute ders per step, `leaf_estimation_iterations` frozen in the fixture.
- Depth-6 fixture + speed workload from the extended Phase-10 synthetic generator (RMSE + Logloss configs); Kaggle oracle reports final-prediction ε AND a per-tree split-agreement / run-to-run-spread diagnostic.
- Fixed-point `Atomic<u64>` histogram accumulator (k=30) as step 0, with the fixed-order tree reduce as capability fallback (SPIKE-REDUCTION §5b).

</specifics>

<deferred>
## Deferred Ideas

- **Real named datasets (Higgs / Epsilon)** as a realistic / published-comparable speed cross-check — deferred to Phase 14's comprehensive benchmark (BENCH-03), consistent with the Phase-10 deferral. Phase 11 uses the reproducible synthetic generator.
- **Non-symmetric grow policies (Depthwise/Lossguide/Region), Exact weighted-quantile leaf estimation, bootstrap/MVS sampling, CTR/categoricals** — Phase 12 (build on this phase's depth>1 histogram/partition machinery).
- **Pairwise/ranking/multiclass/ordered/Langevin device families** — Phase 13.
- **On-device border/quantile computation** (`FastGpuBorders`) — out of scope milestone-wide; host CPU quantization stays the ≤1e-5 reference.

### Reviewed Todos (not folded)
None — no pending todos matched this phase's scope.

</deferred>

---

*Phase: 11-depth-1-partition-aware-histograms-reduction-determinism-newton-der2*
*Context gathered: 2026-07-03*
