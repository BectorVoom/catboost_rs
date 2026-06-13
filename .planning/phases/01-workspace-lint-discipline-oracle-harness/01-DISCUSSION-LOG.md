# Phase 1: Workspace, Lint Discipline & Oracle Harness - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-13
**Phase:** 1-Workspace, Lint Discipline & Oracle Harness
**Areas discussed:** Crate decomposition, Oracle extraction method, Fixture storage & scope, Lint enforcement mechanism

---

## Crate Decomposition

### Q1 — Granularity (first attempt, user requested clarification)
Initial proposal offered fine-grained (with `cb-backend-{cpu,wgpu,cuda,rocm}` as 4 crates) vs lean-core. User clarified: **"I want backend is only one."** Question was reformulated.

### Q1b — Backend location (after clarification)
| Option | Description | Selected |
|--------|-------------|----------|
| Backend folded into cb-compute | No separate backend crate; cb-compute owns the cfg-gated runtime alias | |
| One dedicated cb-backend crate | Single cb-backend crate isolates all CubeCL code; cb-compute stays pure generic | ✓ |
| You decide | Defer to planning | |

**User's choice:** One dedicated `cb-backend` crate.
**Notes:** Backend = one feature-gated surface (cpu/wgpu/cuda/rocm), NOT four crates. `cb-compute` keeps no CubeCL dependency. Full set: cb-core, cb-data, cb-compute, cb-backend, cb-train, cb-model, cb-oracle, catboost-rs.

### Q2 — Public crate name
| Option | Description | Selected |
|--------|-------------|----------|
| catboost-rs | Facade named catboost-rs; internal crates keep cb- prefix | ✓ |
| catboost | Cleanest drop-in, but collides with vendored upstream crate | |
| cb (uniform prefix) | Most consistent internally, least discoverable | |

**User's choice:** `catboost-rs` facade; internal crates keep `cb-` prefix.

---

## Oracle Extraction Method

| Option | Description | Selected |
|--------|-------------|----------|
| Python-first hybrid | Pinned pip catboost as primary generator; C++ instrumentation only where Python can't reach (per-phase) | ✓ |
| Instrument vendored C++ | Patch & build catboost-master to dump JSON at every stage; full internal access but heavy C++ build now | |
| Python-only, defer internals | Python package only; gaps handled reactively | |

**User's choice:** Python-first hybrid.
**Notes:** Oracle pinned to `catboost==1.2.10` (matches vendored `catboost-master` source version — derived from `version.py`, stated as a locked decision rather than asked). C++ instrumentation deferred per-phase, mainly Phase 5.

---

## Fixture Storage & Scope

### Q1 — Format
| Option | Description | Selected |
|--------|-------------|----------|
| JSON, full-precision floats | One JSON per scenario, 17-sig-digit floats; diffable | |
| Hybrid: JSON meta + binary arrays | config.json metadata + .npy arrays; compact + exact | ✓ |
| You decide | Defer to planning | |

**User's choice:** Hybrid — `config.json` + `.npy` arrays under `cb-oracle/fixtures/`.

### Q2 — Scope (shared corpus vs per-phase)
| Option | Description | Selected |
|--------|-------------|----------|
| Frozen shared input corpus | Phase 1 freezes canonical input datasets once; later phases add output fixtures | ✓ |
| Per-phase, just-in-time | Only harness + PRNG fixture in P1; each phase generates its own | |
| You decide | Defer to planning | |

**User's choice:** Frozen shared input corpus generated in Phase 1.

---

## Lint Enforcement Mechanism

### Q1 — Enforcement stack
| Option | Description | Selected |
|--------|-------------|----------|
| workspace.lints + structural anyhow ban | [workspace.lints] table + structural anyhow exclusion + CI grep backstop | ✓ |
| Per-crate attrs + cargo-deny | #![deny] per lib.rs + cargo-deny bans; more duplication | |
| You decide | Defer to planning | |

**User's choice:** `[workspace.lints]` table + structural anyhow ban + CI grep backstop.

### Q2 — CI platform (first attempt, user requested clarification)
User clarified: **"rocm backend can not test in github action."** Question was reformulated to split CPU vs ROCm lanes.

### Q2b — CI structure (after clarification)
| Option | Description | Selected |
|--------|-------------|----------|
| GH Actions (CPU) + scripted ROCm lane | GH Actions runs CPU gates; ROCm tests as standalone scripts for local/self-hosted AMD (P7) | ✓ |
| GH Actions CPU only, decide GPU lane in P7 | CPU workflow now; ROCm automation entirely deferred | |
| Self-hosted AMD runner from day one | Provision AMD runner in P1; idle until P7 | |

**User's choice:** GitHub Actions CPU lane + scripted ROCm lane (`scripts/gpu-rocm-tests.sh`) for Phase 7 self-hosted.
**Notes:** ROCm cannot run in GitHub Actions (no AMD-GPU runners) — hard constraint confirmed by user.

---

## Claude's Discretion

- Exact frozen-input-corpus dataset shapes / row counts / feature mixes.
- `TFastRng64` bitstream-capture mechanics (INFRA-05) — source at `catboost-master/util/random/fast.{h,cpp}` + `fast_ut.cpp`.
- Rust edition / MSRV / toolchain pinning specifics.
- Whether `cb-backend`'s `cpu` feature uses CubeCL's CPU runtime or a plain CPU path in early phases.
- Whether the `catboost-rs` facade is treated as a thiserror-only "core lib" or an anyhow "edge" (minor; default to lib).

## Deferred Ideas

- Self-hosted AMD CI runner provisioning → Phase 7.
- C++ instrumentation of catboost-master for training-internal oracle values → primarily Phase 5.
- Facade-as-edge vs facade-as-lib for anyhow → settle in planning.
