---
phase: 10
slug: gpu-foundations-runtime-seam-session-residency-device-primit
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-07-03
---

# Phase 10 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust `cargo test` (per-crate; workspace) + human-gated Kaggle CUDA `.ipynb` (correctness + speed oracle of record) |
| **Config file** | none — Cargo workspace; Kaggle notebook committed under the phase/harness dir |
| **Quick run command** | `cargo test -p cb-backend` (feature-gated primitive/self-oracle unit tests) |
| **Full suite command** | `cargo test --workspace` + human-gated Kaggle CUDA notebook run (`--features cuda`) |
| **Estimated runtime** | ~seconds for CPU unit tests; Kaggle CUDA run is human-gated (out-of-band) |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p <crate>` for the touched crate
- **After every plan wave:** Run `cargo test --workspace`
- **Before `/gsd-verify-work`:** Full CPU suite green; Kaggle CUDA oracle (correctness blocking gate) discharged for GPU deliverables
- **Max feedback latency:** CPU seconds; GPU oracle is human-gated per the ROADMAP validation authority

---

## Per-Task Verification Map

> Populated by the planner/executor. GPU-kernel tasks carry a serial CPU/numpy self-oracle (D-02) and a Kaggle-CUDA authoritative run (D-01/D-05); trivial primitives are covered transitively via the depth-1 tree + cindex end-to-end.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 10-01-01 | 01 | 1 | GPUT-16 | — | N/A | unit (self-oracle) | `cargo test -p cb-backend` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Serial CPU/numpy self-oracle harness for standalone primitives (scan/segmented-scan, radix sort + stable single-bit reorder, reduce-by-key, `update_part_props`) — D-01/D-02
- [ ] Kaggle CUDA `.ipynb` scaffold (build `--features cuda` wheel, load committed seeded fixtures, correctness-first blocking gate, warm-run train-only wall-clock) — BENCH-01
- [ ] Seeded synthetic generator (one source for depth-1 ≤1e-5 correctness fixture AND large-n speed workload, ~1e6×50) — D-06

*Per-primitive/per-tree oracle stubs and fixtures are provisioned by the plans that own each requirement.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Depth-1 device tree ≤1e-5 vs CPU; primitive/cindex ≤1e-4 | GPUT-04/08/15/16 | Authoritative GPU run is Kaggle CUDA (human-gated external notebook); ROCm in-env is optional smoke, not a gate | Run the committed `.ipynb` on a Kaggle CUDA instance; correctness report must pass at the ε bar before speed is read |
| Warm-run device-vs-CPU wall-clock at large n | BENCH-01/02 | Wall-clock speed measured off-CI on real CUDA hardware | Same notebook; read the warm-run/JIT-excluded train-only speed section |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify (CPU self-oracle) or Wave 0 dependencies, plus a Kaggle-CUDA authoritative check for GPU deliverables
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (self-oracle harness, notebook scaffold, synthetic generator)
- [ ] No watch-mode flags
- [ ] Feedback latency < seconds (CPU); GPU oracle human-gated
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
