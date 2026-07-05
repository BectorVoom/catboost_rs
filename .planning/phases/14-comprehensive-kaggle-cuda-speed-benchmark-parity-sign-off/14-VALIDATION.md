---
phase: 14
slug: comprehensive-kaggle-cuda-speed-benchmark-parity-sign-off
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-07-05
---

# Phase 14 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | cargo test (Rust) + Python (Kaggle CUDA notebook) |
| **Config file** | none — reuses existing `bench/` harness + `crates/cb-train/tests/bench_grow_speed_test.rs` |
| **Quick run command** | `cargo check -p cb-backend -p cb-train` (in-env compile smoke) |
| **Full suite command** | Human-gated Kaggle CUDA notebook run (correctness oracle pre-flight → speed) |
| **Estimated runtime** | in-env smoke ~seconds; Kaggle CUDA run ~minutes (human-gated) |

---

## Sampling Rate

- **After every task commit:** Run `cargo check` on any touched crate (bench scripting is the primary surface; no production `cb-backend`/`cb-train` source changes expected — D-04 no-regression).
- **After every plan wave:** Aggregation/doc tasks verified by re-reading the committed per-phase `result.json` schemas and confirming the sign-off table matches them.
- **Before `/gsd-verify-work`:** The human-gated Kaggle CUDA run must have (1) passed the correctness oracle as a blocking pre-flight and (2) produced the CatBoost-GPU head-to-head numbers.
- **Max feedback latency:** in-env compile smoke < 60s; final speed sign-off gated on the human Kaggle run.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 14-01-01 | 01 | 1 | BENCH-03 | — | N/A | manual | Kaggle CUDA notebook (human-gated) | ❌ W0 | ⬜ pending |

*Planner replaces this row with the real per-task map. Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

*Existing infrastructure covers all phase requirements — reuses the Phase-10 `bench/` harness, `bench_grow_speed_test.rs`, and committed per-phase `result.json` files. No new test framework needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Device ≥20× vs host-CPU baseline across the aggregated matrix; CatBoost-GPU head-to-head recorded (informational) | BENCH-03 | Requires a human-gated Kaggle CUDA (`--features cuda`) run on real NVIDIA hardware; no CUDA GPU in-env (ROCm only, non-gating) | Run the extended Kaggle CUDA notebook: verify CUDA active via `nvidia-smi` → run correctness oracle as blocking pre-flight → warm one untimed fit → drain lazy CubeCL queue → time official CatBoost GPU on the same synthetic large-n configs → commit `result.json`; then aggregate into the BENCH-03 sign-off doc |
| Correctness oracle re-confirmed on CUDA backend (≤1e-4 vs Rust CPU; ≤1e-5 depth-1) | BENCH-03 (SC-2 pre-flight) | Same human-gated Kaggle CUDA requirement | Correctness gate must be GREEN before any speed number is quoted (do-not-fabricate) |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies (aggregation/doc tasks: in-env compile smoke + schema-match; speed tasks: human-gated Kaggle CUDA)
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (none — existing harness reused)
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s for in-env smoke
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
