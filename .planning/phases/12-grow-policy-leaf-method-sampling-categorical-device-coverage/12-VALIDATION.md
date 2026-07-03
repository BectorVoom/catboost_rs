---
phase: 12
slug: grow-policy-leaf-method-sampling-categorical-device-coverage
status: approved
nyquist_compliant: true
wave_0_complete: false
created: 2026-07-03
approved: 2026-07-03
---

# Phase 12 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[test]` (per-crate), separate `*_test.rs` files (source/test separation is mandatory) |
| **Config file** | none — Cargo workspace; per-crate test targets |
| **Quick run command** | `cargo test -p <crate> <family_filter>` (e.g. `cargo test -p cb-backend region`) |
| **Full suite command** | `cargo test --workspace` (CPU features); ROCm smoke: `cargo test -p cb-backend --features rocm` (local convenience only) |
| **Estimated runtime** | ~60–180 seconds per crate (compile-dominated; see disk-pressure note below) |

**Authority note:** Per-family device correctness + speed sign-off is a **human-gated Kaggle CUDA `--features cuda` notebook** at ε=1e-4 (reusing the Phase-10 harness). ROCm in-env and CPU self-oracle tests are convenience gates only, NOT the correctness authority.

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p <crate>` for the touched crate (per-crate to avoid the workspace link-pressure blocker).
- **After every plan wave:** Run the family's CPU self-oracle test(s) + `cargo build -p cb-backend --features rocm` smoke.
- **Before `/gsd-verify-work`:** All CPU self-oracle tests green; each landed family carries a recorded Kaggle CUDA ε=1e-4 sign-off + BENCH-02 speed measurement.
- **Max feedback latency:** ~180 seconds (per-crate).

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 12-01-xx | 01 | 1 (Depthwise/Lossguide emit) | GPUT-18 | — | N/A | unit self-oracle | `cargo test -p cb-backend nonsym` | ❌ W0 | ⬜ pending |
| 12-02-xx | 02 | 2 (Region CPU→device) | GPUT-18 | — | N/A | unit self-oracle | `cargo test -p cb-train region && cargo test -p cb-backend region` | ❌ W0 | ⬜ pending |
| 12-03-xx | 03 | 3 (Exact quantile) | GPUT-19 | — | N/A | unit self-oracle | `cargo test -p cb-backend exact` | ❌ W0 | ⬜ pending |
| 12-04-xx | 04 | 4 (bootstrap/random-strength) | GPUT-09 | — | N/A | unit self-oracle | `cargo test -p cb-backend bootstrap` | ❌ W0 | ⬜ pending |
| 12-05-xx | 05 | 5 (MVS) | GPUT-17 | — | N/A | unit self-oracle | `cargo test -p cb-backend mvs` | ❌ W0 | ⬜ pending |
| 12-06-xx | 06 | 6 (CTR device) | GPUT-10 | — | N/A | unit self-oracle | `cargo test -p cb-backend ctr` | ❌ W0 | ⬜ pending |
| 12-xx-bench | per family | per family | BENCH-02 | — | N/A | manual (Kaggle CUDA) | recorded notebook run | ❌ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky. Exact task IDs are assigned by the planner; this map is the wave-level skeleton.*

---

## Wave 0 Requirements

- [ ] Per-family CPU self-oracle fixtures — pin RNG seed / sampling config and **freeze the exact CPU-reference sample/tree/leaf-values** in the fixture (D-07 discipline, extended to every family), so the device path is reproduced bit-for-bit at ε=1e-4.
- [ ] Region CPU oracle (Wave 2) — the CPU Region path (grower + `TreeVariant::Region` + `AddRegion`/`ComputeRegionBins` apply) must land and be ≤1e-5 oracle-locked BEFORE the device Region kernel (D-03a); this is the CPU reference the device path oracles against.
- [ ] Reuse the Phase-10 Kaggle CUDA oracle harness for each family's human-gated ε=1e-4 + BENCH-02 run.

*No new test framework install needed — Rust `#[test]` + existing per-crate oracle harness cover all phase requirements.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Per-family device correctness ε=1e-4 vs CPU | GPUT-18/19/09/17/10 | Kaggle CUDA is the sole correctness authority; no CUDA GPU in-env | Run the family's fixture through the Phase-10 `--features cuda` Kaggle notebook; assert max abs/rel divergence ≤1e-4 vs frozen CPU sample |
| Per-family device speed (device vs host-CPU baseline; vs official CatBoost GPU where comparable) | BENCH-02 | Requires real CUDA timing, warm-run/JIT-excluded | Time train-only on Kaggle CUDA as each family lands; record in the GPU coverage matrix |
| GPU coverage matrix (per-family correctness + speed) documented | SC-5 | Aggregates human-gated runs | Update the coverage-matrix doc as each family flips `Ok(None)`→device |

---

## Validation Sign-Off

- [x] All tasks have a CPU self-oracle `<automated>` verify or a Wave 0 fixture dependency
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references (per-family frozen fixtures + CPU Region oracle)
- [x] No watch-mode flags
- [x] Feedback latency < 180s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-07-03
