# Phase 7: GPU Backends via CubeCL - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-19
**Phase:** 07-gpu-backends-via-cubecl
**Areas discussed:** CUDA fidelity bar, Determinism vs tolerance, MVP GPU coverage, Validation without hardware

---

## CUDA Fidelity Bar

| Option | Description | Selected |
|--------|-------------|----------|
| Result-equivalence | Port the algorithm only; one generic histogram, no specialization; same numeric result within tolerance | |
| Structural parity | Port the specialized CUDA kernel family faithfully (bit-width variants, atomics, one-hot, half-byte) | ✓ |
| Result now, struct later | Result-equivalence MVP; structural/perf parity deferred to a follow-up | |

**User's choice:** Structural parity
**Notes:** Reinforces the original "match cuda in catboost" steer. Claude flagged this conflicts with MVP-mode sizing.

### Parity scope (follow-up)

| Option | Description | Selected |
|--------|-------------|----------|
| Pointwise hist family | Pointwise histogram family + grad/hess + scan + reductions; defer pairwise | |
| Pointwise + pairwise | Both pointwise AND pairwise histogram families this phase | ✓ |
| Single-width first | CUDA kernel shape but only 8-bit/general width; bit-width specializations as fast-follow | |

**User's choice:** Pointwise + pairwise
**Notes:** Most complete CUDA match; roughly doubles kernel surface and wavefront validation burden.

---

## Determinism vs Tolerance

| Option | Description | Selected |
|--------|-------------|----------|
| Match CUDA atomics | In-kernel atomic/plane reductions like CUDA; non-deterministic float order → looser epsilon | ✓ |
| Host-fold reductions | Keep CPU sum_f64 host fold; deterministic tight epsilon; diverges from CUDA structure | |
| Deterministic on-GPU | On-GPU reduction with fixed deterministic ordering; middle path | |

**User's choice:** Match CUDA atomics

### GPU-06 epsilon establishment (follow-up)

| Option | Description | Selected |
|--------|-------------|----------|
| Empirical + sign-off | Measure rocm-vs-CPU divergence on fixtures, propose epsilon w/ headroom, user signs off | ✓ |
| Fixed target now | Lock a specific epsilon (e.g. 1e-3) up front | |
| Per-kernel epsilons | Separate signed-off tolerances per kernel family | |

**User's choice:** Empirical + sign-off

---

## MVP GPU Coverage

| Option | Description | Selected |
|--------|-------------|----------|
| Kernel-offload hybrid | 4 kernel families on GPU; host keeps orchestration/split/leaf/CTR | |
| Full on-device loop | Whole tree-grow loop on GPU, device-resident, minimal roundtrips, like CUDA | ✓ |
| Hybrid now, resident later | Hybrid this phase; full residency as a perf follow-up | |

**User's choice:** Full on-device loop
**Notes:** Claude flagged the combination of all maximal choices ≈ porting CatBoost's entire CUDA training engine; recommended splitting Phase 7 into sub-phases and revisiting the mvp mode tag.

---

## Validation Without Hardware

**Mid-discussion correction from user:** "This env is rocm, so you test in rocm" and "I do not have cuda." Claude verified the environment: AMD gfx1100 (RDNA3, wave32), ROCm 7.1.1, HIP 7.1, /dev/kfd present; no NVIDIA/CUDA. This overrode the original "wgpu as proxy" framing. Claude surfaced the wave32-vs-wave64 caveat (gfx1100 is wave32-native; GPU-03's wave64 cannot be literally validated here).

| Option | Description | Selected |
|--------|-------------|----------|
| rocm + wgpu CI | rocm authoritative gate (local); wgpu-vs-CPU wired for automated CI smoke | |
| rocm only | rocm in-env is the sole GPU oracle; wgpu/cuda compile-gated stubs only | ✓ |
| rocm gate, wgpu dev-only | rocm gate; wgpu buildable/runnable for dev but not a blocking CI check | |

**User's choice:** rocm only
**Notes:** cuda = compile-gated only (no hardware), matching GPU-05. ROCm runs locally/manually in-env, never in GitHub Actions (standing constraint).

---

## Claude's Discretion

- Concrete CubeCL kernel decomposition, shared-memory/blocking details, and the plane/subgroup primitives used for wave-size-agnostic reductions — subject to structural parity (D-01) and the no-warp-size-constants rule (D-09).

## Deferred Ideas

- Wave64 (CDNA / MI-series) literal validation — design stays wave-size-agnostic; defer to when CDNA hardware is available.
- Performance/perf-tuning parity beyond structural parity — follow-up after correctness lands.
- wgpu as a CI validation gate — explicitly not done this phase; could be added later.
