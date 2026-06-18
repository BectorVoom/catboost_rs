# Phase 7: GPU Backends via CubeCL - Context

**Gathered:** 2026-06-19
**Status:** Ready for planning

<domain>
## Phase Boundary

GPU training execution via CubeCL kernels, added **purely additively** on the
locked generic boundary (`cb-core`/`cb-model` stay frozen in their Phase 3–6
form). Kernels are generic over `R: Runtime` and `F: Float` (histogram,
gradient/hessian, scan, reductions). Backend selection is compile-time `cfg`
only through the single `SelectedRuntime` alias in `cb-backend` — zero runtime
dispatch. A documented GPU tolerance is established and signed off: `rocm`
results within a stated epsilon vs the **Rust CPU path** (not the C++ CPU
oracle).

**User steer for this phase:** "match cuda in catboost" — the CubeCL kernels are
to mirror the upstream CUDA training implementation in
`catboost-master/catboost/cuda/` as faithfully as possible (structural parity,
see D-01/D-02 below).

**Requirements in scope:** GPU-01, GPU-02, GPU-03, GPU-04, GPU-05, GPU-06.
</domain>

<decisions>
## Implementation Decisions

### CUDA Fidelity Bar
- **D-01:** Fidelity bar is **full structural parity** with `catboost/cuda/` —
  port the CUDA kernel *structure* (memory layout, blocking, reduction strategy,
  accumulation approach), not merely the algorithm/result. This is the user's
  explicit choice over "result-equivalence only."
- **D-02:** Parity scope for the kernel families is **pointwise AND pairwise
  histogram families**, plus gradient/hessian, scan, and reductions. Both the
  `pointwise_hist2*` family and the `pairwise_hist*` family are in-scope —
  including the bit-width-specialized variants (5/6/7/8-bit, half-byte, binary,
  one-hot) that the upstream CUDA layer ships.

### Reductions / Determinism
- **D-03:** GPU reductions and histogram accumulation **match CUDA's in-kernel
  atomic adds** (atomic/plane reductions on device). This is a deliberate
  departure from the CPU seam's host-side `sum_f64` fold (D-02/D-05 of Phase 3).
  Consequence: float summation order is non-deterministic, so the GPU-06 epsilon
  must be looser than the CPU path's 1e-5 and must absorb run-to-run variance.

### GPU Tolerance (GPU-06)
- **D-04:** The `rocm`-vs-Rust-CPU epsilon is established **empirically, then
  signed off by the user**. The executor/researcher measures actual divergence
  on representative fixtures on the in-env gfx1100 GPU, proposes an epsilon with
  headroom over the observed max + run-to-run variance, and the user signs off
  the concrete number in CONTEXT/VERIFICATION. No epsilon is hard-coded up front.

### GPU Coverage (this phase)
- **D-05:** The **full tree-grow loop runs on-device** (device-resident
  candidates/scores/leaves, minimal host↔device roundtrips), mirroring upstream
  CUDA's design — not a kernel-offload hybrid with host orchestration.

### Validation Strategy
- **D-06:** `rocm` in this environment (gfx1100 / RDNA3, ROCm 7.1.1, HIP 7.1) is
  the **sole GPU oracle** and the GPU-06 sign-off gate. Run locally/manually on
  this machine — **never in GitHub Actions** (standing CI constraint for ROCm).
- **D-07:** `wgpu` and `cuda` remain **compile-gated stubs** — they MUST build
  behind their feature gates but are NOT validated (no CUDA hardware exists here;
  `wgpu` is not wired as a validation gate). This satisfies GPU-04 (compiles) and
  GPU-05 (cuda compile-gated, untested locally) without claiming validation.

### Flagged Caveats (not blockers — for planner/researcher awareness)
- **D-08 (Sizing):** D-01+D-02+D-03+D-05 together amount to porting essentially
  CatBoost's **entire CUDA training engine** to CubeCL. This is far larger than a
  single MVP-mode phase. **Strong recommendation: the planner split Phase 7 into
  sub-phases** (as Phase 6 was split into 6.1–6.6) and revisit the `mvp` mode tag
  on the ROADMAP entry before planning. The fidelity bar stays as decided; only
  the sizing/organization is flagged.
- **D-09 (Wave size):** The test box is gfx1100 / **wavefront-32-native** (RDNA3),
  not wave64. GPU-03 ("wavefront-64 safe; no warp-size assumptions") is validated
  here only as the *no-warp-size-assumptions* property on wave32. Literal
  **wave64 execution is NOT validated** on this hardware (would require CDNA /
  MI-series silicon). Code must remain strictly wave-size-agnostic; the wave64
  validation gap is a documented limitation, not closed by this phase.

### Claude's Discretion
- Concrete CubeCL kernel decomposition, shared-memory/blocking details, and the
  plane/subgroup primitives used to express wave-size-agnostic reductions are
  left to research + planning, subject to D-01 (structural parity) and D-09
  (no warp-size constants).
</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Upstream CUDA reference (the parity target — D-01/D-02)
- `catboost-master/catboost/cuda/` — the full CUDA training engine; the
  structural-parity reference for this phase.
- `catboost-master/catboost/cuda/methods/kernel/` — histogram kernels:
  `pointwise_hist2*.cu/.cuh` (pointwise family + 5/6/7/8-bit, half-byte, binary
  variants) and `pairwise_hist*.cu/.cuh` (pairwise family incl. atomics &
  one-hot variants). Primary porting surface.
- `catboost-master/catboost/cuda/methods/kernel/score_calcers.cuh` — score
  computation referenced by the histogram→split-selection path.
- `catboost-master/catboost/cuda/cuda_lib/`, `…/cuda_util/` — reduction/scan
  primitives and device-memory utilities (reference for D-03 atomic reductions
  and the scan/reduction kernels).
- `catboost-master/catboost/cuda/gpu_data/`, `…/targets/`, `…/train_lib/` —
  device-resident data layout, target/derivative computation, and the on-device
  train loop (reference for D-05 full on-device loop).

### Project requirements & roadmap
- `.planning/ROADMAP.md` § "Phase 7: GPU Backends via CubeCL" — goal, success
  criteria, and the NEEDS-DEEPER-RESEARCH flag (cubecl-hip kernel coverage,
  wavefront determinism, cubecl-hip-sys HIP version match, GPU epsilon sign-off).
- `.planning/REQUIREMENTS.md` § GPU-01…GPU-06 — the requirements this phase closes.

### Current Rust compute seam (the additive boundary — must stay frozen)
- `crates/cb-backend/src/lib.rs` — owns `SelectedRuntime` cfg-alias (GPU arms are
  currently inert `()` placeholders to be replaced this phase); D-02 single-crate
  ownership of backend selection.
- `crates/cb-backend/src/kernels.rs`, `…/src/kernels/` — existing `#[cube]`
  elementwise kernels (CPU); the GPU kernels extend this additively.
- `crates/cb-backend/src/cpu_runtime.rs` — the `Runtime` impl pattern GPU
  runtimes follow.
- `crates/cb-compute/src/lib.rs` — defines the abstract `Runtime`/`Float` traits;
  MUST NOT depend on `cubecl` (D-03 of Phase 3 keeps the generic boundary
  cubecl-free).

### CubeCL authoring rules (MANDATORY per AGENTS.md)
- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md` — read
  before writing any kernel code.
- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/cubecl_error_guideline.md`
  — load immediately on any CubeCL build error before attempting a fix.
- Kernels must use `generics-float` — no hard-coded float types.

### Codebase maps (background)
- `.planning/codebase/ARCHITECTURE.md`, `…/STRUCTURE.md`, `…/TESTING.md` —
  workspace layout, crate responsibilities, oracle-test conventions.
</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `cb-backend` `SelectedRuntime` cfg-alias machinery — GPU arms already stubbed
  (`()` placeholders), ready to be filled with real CubeCL GPU runtimes.
- Existing `#[cube]` elementwise kernels in `cb-backend/src/kernels.rs` — the
  generics-float authoring pattern and `Runtime`/`Float` trait impls to mirror.
- `cb-core::sum_f64` host-fold — still used by the CPU path; GPU path deliberately
  diverges (D-03) but the CPU path remains the parity *baseline* for GPU-06.

### Established Patterns
- Compile-time backend selection only — no runtime `match` over backends (D-02).
- Additive generic boundary — `cb-compute` stays cubecl-free; `cb-core`/`cb-model`
  unchanged from Phase 3–6 (SC-1).
- Oracle/test separation — tests live in dedicated `_test.rs` files, not inline.

### Integration Points
- New GPU runtimes implement the same `Runtime`/`Float` traits `cb-compute`
  defines, selected via the `cb-backend` cfg-alias — the single integration seam.

### Environment facts (verified 2026-06-19)
- GPU: AMD gfx1100 (RDNA3, wave32-native), in `AMD Ryzen AI 7 350 w/ Radeon 860M`.
- ROCm 7.1.1 at `/opt/rocm-7.1.1`; HIP 7.1.52802; `hipcc`/`rocminfo`/`rocm-smi`
  present; `/dev/kfd` exposed. GPU visible and usable in-environment.
- No NVIDIA/CUDA hardware present.
</code_context>

<specifics>
## Specific Ideas

- "match cuda in catboost" — the guiding instruction for the whole phase. The
  upstream CUDA layer (`catboost/cuda/`) is the structural reference, not just an
  algorithmic one (D-01).
- Disk pressure is a known constraint on this machine (root ~100% full;
  per-crate verification preferred over full-workspace test links) — relevant to
  how GPU build/test verification is run.
</specifics>

<deferred>
## Deferred Ideas

- **Wave64 (CDNA / MI-series) validation** — design stays wave-size-agnostic
  (D-09), but actually running on wave64 hardware is out of reach here; defer
  literal wave64 validation to whenever CDNA hardware is available.
- **Performance/perf-tuning parity beyond structural parity** — once correctness
  + structural parity land, deeper perf matching (occupancy tuning, transfer
  elimination) can be a follow-up.
- **wgpu as a CI validation gate** — explicitly NOT done this phase (D-07);
  could be added later if an automated GPU CI runner becomes worthwhile.

### Reviewed Todos (not folded)
None — no pending todos matched this phase.
</deferred>

---

*Phase: 7-gpu-backends-via-cubecl*
*Context gathered: 2026-06-19*
