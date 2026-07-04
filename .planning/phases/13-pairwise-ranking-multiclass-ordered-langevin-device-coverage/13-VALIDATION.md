---
phase: 13
slug: pairwise-ranking-multiclass-ordered-langevin-device-coverage
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-07-04
---

# Phase 13 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[test]` (source/test separated per CLAUDE.md) |
| **Config file** | none — per-crate `Cargo.toml` |
| **Quick run command** | `cargo test -p cb-backend --lib` |
| **Full suite command** | `cargo test -p cb-backend -p cb-compute -p cb-train` |
| **Estimated runtime** | ~60–180 seconds (CPU self-oracle path) |

**GPU authority note:** correctness+speed sign-off is the **human-gated Kaggle CUDA `--features cuda` notebook** (ε=1e-4 + BENCH-02 timing), reusing the Phase-10 harness. ROCm in-env (`--features rocm`, gfx1100) is an optional compile/smoke convenience, **not a gate**. In-env `cargo test` runs the CPU self-oracle; per-family device sign-off is discharged externally on Kaggle CUDA.

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p cb-backend --lib`
- **After every plan wave:** Run the full suite command above
- **Before `/gsd-verify-work`:** Full CPU suite green; each landed family has a recorded Kaggle CUDA ε=1e-4 + BENCH-02 result
- **Max feedback latency:** ~180 seconds (CPU); Kaggle CUDA is asynchronous / human-gated

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 13-01-01 | 01 | 1 | GPUT-11/21 | — | N/A (numeric compute) | unit | `cargo test -p cb-backend --lib` | ❌ W0 | ⬜ pending |

*Populated by the planner/executor per plan. Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] CPU self-oracle harnesses per family (max abs/rel divergence over equal-length buffers at ε=1e-4)
- [ ] Frozen oracle fixtures: deterministic families (pairwise/query-deterministic/multiclass/ordered) and pinned-seed/frozen-fixture families (YetiRank/PFound-F, Langevin, ordered permutation trajectory)

*Existing Phase 10/11/12 test infrastructure covers the device-substrate primitives; per-family additions land with each plan.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Per-family device ε=1e-4 correctness | GPUT-11/21/22/12/13/20 | Requires real CUDA GPU (no CUDA in-env) | Run the Phase-10 Kaggle CUDA `--features cuda` notebook per family |
| Per-family BENCH-02 speed measurement | BENCH-02 | Requires real CUDA GPU; warm-run/JIT-excluded, train-only | Time device vs host-CPU baseline (and vs official CatBoost GPU where comparable) on Kaggle CUDA as each family lands |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify (CPU self-oracle) or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 180s (CPU)
- [ ] Each landed family has a recorded Kaggle CUDA ε=1e-4 + BENCH-02 result (coverage matrix, SC-5)
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
