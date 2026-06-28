# Phase 10: Coarse Runtime Grow-Tree Seam + GpuTrainSession Residency + Wire Depth-1 + Kaggle CUDA Oracle & Speed Harness - Context

**Gathered:** 2026-06-29
**Status:** Ready for planning

<domain>
## Phase Boundary

Make the existing-but-unwired device grow loop reachable from training, keep training
data device-resident across iterations, grow a **depth-1** oblivious tree fully on device
matching the CPU path, and stand up the foundational **Kaggle CUDA** correctness+speed
harness reused by Phases 11–13.

Concretely this phase delivers:
- A `Runtime` grow-tree trait seam in `cb-compute` (`begin_device_training` /
  `grow_tree_on_device → CbResult<Option<DeviceGrownTree>>` / `end_device_training`),
  CubeCL-free host-typed signatures, with `Ok(None)`→host-CPU fallback (GPUT-01).
- A `GpuTrainSession` (cb-backend-internal) owning one `ComputeClient` + all persistent
  device handles; quantized matrix uploaded once per `fit()` (GPUT-02).
- Gradients/approx device-resident across iterations; per-tree `der1` host read-back
  eliminated; only the O(1) BestSplit descriptor + `2^depth` partition stats cross
  host↔device per level (GPUT-03 / D-05).
- A depth-1 oblivious tree (RMSE/Logloss, Plain boosting, fold_count=1) grown on device,
  matching the CPU path ≤1e-5, oracle-tested on Kaggle CUDA (GPUT-04).
- The reproducible Kaggle CUDA oracle/speed harness (BENCH-01) + the standing per-phase
  speed check (BENCH-02).

**NOT in this phase (deferred to 11–13):** depth>1 partition-aware histograms, Newton der2,
CTR, pairwise/ranking, multiclass, ordered boosting, the comprehensive final speed sign-off
(BENCH-03). These return `Ok(None)` and fall back to the CPU grower.

</domain>

<decisions>
## Implementation Decisions

### Fallback granularity (Ok(None) → CPU)
- **D-10-01:** **Per-fit, all-or-nothing.** The covered/uncovered decision is made ONCE at
  `begin_device_training`. If the whole config is device-coverable → a `Some(session)` is
  returned and EVERY tree in the fit grows on device. If not → `None`, and the ENTIRE fit
  uses the CPU grower. **No mid-run mixing of device-grown and CPU-grown trees in one
  model** — chosen for parity safety (no subtle device/CPU drift compounding across the
  boosting run) and a clean, attributable speed measurement.
- **D-10-02:** Because the gate is per-fit, the covered-vs-uncovered classification lives
  where the session is created (cb-backend `GpuTrainSession` construction, surfaced through
  the `Runtime` seam). The depth-1/RMSE-or-Logloss/Plain/fold_count=1 coverage check is the
  gate; anything outside it → `None` → CPU. A covered session that nonetheless cannot grow a
  tree mid-run is a hard `CbError` (panic-free), NOT a silent CPU graft.

### Kaggle CUDA harness form (BENCH-01 / BENCH-02)
- **D-10-03:** **Script + README only.** Commit runnable Python (e.g. `bench/cuda_oracle.py`,
  reusing/extending the existing root `benchmark*.py`) plus a `bench/README.md` documenting
  the manual Kaggle notebook steps (build `--features cuda` wheel → run oracle → run speed).
  The user assembles/runs the Kaggle notebook each phase from the README. Most flexible;
  keeps all logic diffable and out of notebook cells.
- **D-10-04:** The harness runs BOTH the correctness oracle (blocking gate, ≤1e-5 depth-1)
  AND the wall-clock speed measurement (warm-run / JIT-excluded, train-only). Correctness is
  a hard gate BEFORE any speed number is reported. It is a **human-gated external step** —
  the user runs the notebook; ROCm in-env is an optional compile/smoke convenience, NOT a
  gate (no requirement satisfied by ROCm alone).
- **D-10-05:** Correctness fixtures are committed as small deterministic files generated
  in-env (the same fixture the in-env build verifies), so the Kaggle run is reproducible and
  diffable. (Planner/researcher to confirm exact fixture format + how the human sign-off is
  recorded — a committed RESULTS log is the expected pattern.)

### Reduction-determinism spike depth (SC5, feeds Phase 11)
- **D-10-06:** **Runnable on-device micro-benchmark.** Prototype ALL three candidates as
  small on-device kernels — (a) fixed-point i64 atomics, (b) private-histogram merge,
  (c) two-pass segmented reduce — and measure BOTH determinism error (vs CPU f64) AND
  wall-clock. Produce `SPIKE-REDUCTION.md` with a real err+ms comparison table and a
  recommendation that feeds Phase 11's histogram kernel.
- **D-10-07:** Run the spike on Kaggle CUDA for the authoritative numbers; the ROCm in-env
  build is a fast local smoke check. Note gfx1100 lacks f64 atomic-add (the in-env smoke may
  need the HostSumFallback path, per Phase 7.6) — the spike must record where each candidate
  is/ isn't viable per backend.

### Depth-1 oracle configs + speed framing (GPUT-04 / BENCH-02)
- **D-10-08:** Oracle BOTH **RMSE and Logloss** depth-1 (Plain, fold_count=1) ≤1e-5 on
  Kaggle CUDA — SC1 names both; exercises the der1 path AND the Logloss der seam.
- **D-10-09:** **Depth-1 device fit MUST beat (≥) the CPU wall-clock** — but the speed gate
  is measured on a **large-n dataset (~10⁵–10⁶ rows)**, NOT the small correctness fixture.
  **RESOLVED 2026-06-29 (post-research escalation):** research established that device≥CPU at
  depth-1 is physically infeasible at small n (1000×20 — launch latency exceeds total CPU
  work) but achievable at large n where the O(n·features) histogram amortizes launch latency.
  Per the user's escalation decision, the planner pins the BENCH-02 depth-1 **speed gate to a
  large-n dataset** while the ≤1e-5 **correctness oracle** still runs on the small
  deterministic fixture. This honors the "device must win" intent without promising the
  impossible. Phase 10 still carries the residency obligation (per-fit upload-once,
  approx/der1 device-resident, der1 read-back eliminated) so no upload/readback dominates the
  large-n measurement; aggressive depth-1 launch-fusion is NOT required to clear the bar at
  large n.
- **D-10-10:** Speed is measured device vs CPU. Baseline = both in-env CPU (dev iteration)
  AND a CPU run on the SAME Kaggle hardware (apples-to-apples for the official number), plus
  vs official CatBoost GPU where a comparable depth-1 config exists.

### Claude's Discretion
- Exact `DeviceGrownTree` struct fields and the precise host-typed signatures of the three
  seam methods (within GPUT-01's named shape).
- Internal `GpuTrainSession` handle layout / lifetime mechanics (reusing Phase 7.2 der
  handles + Phase 7.5 `grow_boosting_pass`).
- Fixture file format and the exact RESULTS sign-off log structure.
- Micro-benchmark kernel sizes / problem shapes for the reduction spike.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Milestone scope & requirements
- `.planning/ROADMAP.md` (Phase 10 section) — goal, depends-on, 5 success criteria, notes,
  validation-authority + per-phase-speed-check preamble.
- `.planning/REQUIREMENTS.md` — GPUT-01..04, BENCH-01, BENCH-02 (standing), the ε=1e-4 vs
  ≤1e-5 parity-bar note, D-04 (CPU byte-unchanged) / D-05 (host↔device traffic) definitions.
- `.planning/PROJECT.md` (## Current Milestone: v1.1 GPU Performance) — target features +
  the cb-train→cb-backend dependency landmine.

### Root-cause & precedent (MUST read — these define the seam to wire)
- `.planning/notes/gpu-training-host-light-root-cause.md` — the integration gap, the
  "what runs where" table, the location of the unwired device grower, MVP limits, and the
  fix constraints. This is the architectural anchor for the whole phase.

### Code seams to touch (full paths)
- `crates/cb-compute/src/runtime.rs:897` — `Runtime` trait (where the grow-tree seam is
  added alongside `compute_gradients`).
- `crates/cb-backend/src/gpu_runtime/mod.rs:1890` — `grow_boosting_pass` (the existing-but-
  unwired depth-1 device grow loop) + `grow_oblivious_tree` (1615); `grow_*_into` launchers.
- `crates/cb-backend/src/gpu_backend.rs:67-146` — current `GpuBackend` (gradients-only;
  Phase 7.2 der seam to reuse).
- `crates/cb-train/src/boosting.rs:1870` (`train<R: Runtime>`), `2101` (`train_inner`),
  `~3203` (grower dispatch where the device-grow attempt is inserted ahead of the CPU
  `greedy_tensor_search_*` growers); `2996` (`der1.clone()` host read-back to eliminate).
- `crates/catboost-rs/src/builder.rs:333-371` — `fit()` backend selection (compile-time
  feature gate; `train` already generic over `R: Runtime`).
- `crates/cb-backend/src/kernels/grow_loop.rs` — existing tests that already drive
  `grow_boosting_pass` (reference for the device-grow call shape).
- `benchmark.py`, `benchmark_fast.py`, `benchmark_small.py` (repo root) — existing speed
  scripts to extend into the committed `bench/cuda_oracle.py`.

### CubeCL (mandatory before any kernel work — per AGENTS.md)
- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md` — read before writing
  kernel code; kernels use generic-float.
- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/cubecl_error_guideline.md` —
  load on ANY CubeCL build error before attempting a fix.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`grow_boosting_pass` (gpu_runtime/mod.rs:1890)** — Phase 7.5 device grow loop: per-level
  histogram + score + split on-device, reads back only `2^depth` leaf-stats per level.
  Depth-1 MVP, no Newton der2, no CTR/pairwise/ordered/multiclass. This IS the loop to wire;
  it currently runs only from `grow_loop.rs` tests.
- **Phase 7.2 der seam** (`gpu_backend.rs`, `const_der_handle` / `*_handle` no-read-back) —
  der1/der2 already device-resident; reuse the handles for residency (GPUT-03).
- **`Runtime` trait (runtime.rs:897)** — already the generic seam `train<R: Runtime>` flows
  through; `compute_gradients` is the only required method. Grow-tree methods are added here
  with the SAME CubeCL-free host-typed discipline.
- **Existing `benchmark*.py`** — starting point for the committed CUDA bench script.

### Established Patterns
- **CubeCL-free host-typed trait signatures** in `cb-compute` — the seam must follow this
  (no CubeCL types leak through `Runtime`); cb-backend implements with CubeCL behind it.
- **Compile-time backend selection** (builder.rs) — exactly one backend feature active;
  `train` accepts any zero-sized backend; no runtime dispatch.
- **`f32::MIN` sentinel, never `-inf` literal** in `#[cube]` kernels (memory
  `cubecl-hip-no-inf-literal`) — HIP/gfx1100 JIT rejects `-inf`.
- **ε precedent from Phase 7.6** — device math is f32; depth-1 level-0 whole-dataset
  histogram IS the exact CPU score → held tighter at ≤1e-5; later depths ε=1e-4.

### Integration Points
- Insert the device-grow attempt in `train_inner`'s grower dispatch (boosting.rs ~3203),
  ahead of the CPU `greedy_tensor_search_*` chain, gated per-fit by the session from
  `begin_device_training`.
- `GpuTrainSession` is constructed once per `fit()` (above the iteration loop) and owns the
  single `ComputeClient` + persistent handles; matrix upload happens here (GPUT-02).

### Landmine (hard constraint)
- **NEVER add a `cb-train` dependency to `cb-backend`** — feature unification breaks the
  rocm runtime. Transcribe any needed CPU reference logic inline (memory
  `phase75-grow-loop-outcome`).

</code_context>

<specifics>
## Specific Ideas

- The user wants depth-1 device to genuinely WIN on wall-clock vs CPU (D-10-09) — this is a
  deliberate, informed bar above the written success criteria. Treat "depth-1 device ≥ CPU
  on Kaggle CUDA" as a Phase-10 success bar.
- **Research flag:** if research concludes depth-1 device > CPU is genuinely infeasible on
  CUDA even after fused/batched launches + persistent kernel + full residency, that must
  ESCALATE back to the user — do NOT silently relax D-10-09.
- Reduction spike is a real runnable comparison (3 kernels, err+ms table), not a memo.

</specifics>

<deferred>
## Deferred Ideas

- depth>1 partition-aware histograms, Newton der2, Cosine GPU score, reduction-determinism
  PRODUCTION kernel — **Phase 11** (the spike here only recommends; Phase 11 builds).
- CTR / pairwise / multiclass / ordered-boosting device paths — **Phase 12** (each lands
  behind the same per-fit fallback gate with its own ε=1e-4 + speed sign-off).
- Comprehensive final speed-parity sign-off vs official CatBoost GPU (BENCH-03) — **Phase 13**
  (aggregates the per-phase BENCH-02 checks).

### Reviewed Todos (not folded)
- **Estimated-feature stored-border-VALUE quantization-grid parity**
  (`estimated-feature-grid-parity.md`) — matched only on generic keywords; it is a
  cb-train KNN/CTR (online-HNSW) feature-parity item (FEAT-07 backlog), unrelated to the GPU
  grow-tree seam. NOT folded into Phase 10.

</deferred>

---

*Phase: 10-coarse-runtime-grow-tree-seam-gputrainsession-residency-wire*
*Context gathered: 2026-06-29*
