---
phase: 10
slug: coarse-runtime-grow-tree-seam-gputrainsession-residency-wire
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-29
---

# Phase 10 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[test]` (oracle tests in separate `_test.rs` files; source/test separation mandatory) + Python (`bench/cuda_oracle.py`) for the Kaggle CUDA harness |
| **Config file** | Cargo workspace (`Cargo.toml`); no separate test config |
| **Quick run command** | `cargo test -p cb-train -p cb-backend -p cb-compute` (per-crate; cb-compute full-suite link is disk-constrained — see memory) |
| **Full suite command** | `cargo test --workspace` (CPU); ROCm smoke: `cargo test -p cb-backend --features rocm`; CUDA oracle: human-gated Kaggle `python bench/cuda_oracle.py` |
| **Estimated runtime** | ~60–180 seconds (CPU per-crate); Kaggle CUDA run is external/human-gated |

---

## Sampling Rate

- **After every task commit:** Run the relevant per-crate `cargo test` quick command
- **After every plan wave:** Run the CPU workspace suite (D-04 no-regression: full existing CPU oracle suite stays green)
- **Before `/gsd-verify-work`:** CPU full suite green in-env; ROCm smoke green in-env; Kaggle CUDA oracle (≤1e-5 depth-1) recorded by the human gate
- **Max feedback latency:** ~180 seconds (in-env); Kaggle CUDA is out-of-band

---

## Per-Task Verification Map

> Populated by the planner from PLAN.md tasks. Each task maps to a requirement (GPUT-01..04, BENCH-01, BENCH-02), an automated verify command, and (where GPU) the validation tier: CPU oracle ≤1e-5, ROCm in-env smoke (non-gating), or Kaggle CUDA (authoritative, human-gated).

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 10-01-01 | 01 | 1 | GPUT-01 | — | N/A | unit | `cargo test -p cb-compute runtime` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Seam unit-test stubs for the `Runtime` grow-tree methods (GPUT-01) in a separate `*_test.rs` file
- [ ] Depth-1 RMSE + Logloss device-vs-CPU oracle fixtures (committed, deterministic) for GPUT-04
- [ ] `bench/cuda_oracle.py` + `bench/README.md` scaffold for BENCH-01/02

*Existing oracle infrastructure (Phase 3–8 `*_test.rs` suites) covers the CPU no-regression baseline (D-04).*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Depth-1 device tree ≤1e-5 vs CPU on CUDA | GPUT-04 / BENCH-01 | No NVIDIA in-env; CUDA is the authoritative oracle | Run `bench/cuda_oracle.py` on a Kaggle CUDA notebook per `bench/README.md`; correctness gate must pass before any speed number; record in RESULTS log |
| Depth-1 device ≥ CPU wall-clock at large-n | BENCH-02 / D-10-09 | Requires CUDA hardware + large-n dataset | Kaggle CUDA: time device vs CPU on ~10⁵–10⁶-row dataset (warm-run/JIT-excluded, train-only); record in RESULTS log |
| Reduction-determinism spike err+ms table | SC5 | CUDA authoritative; gfx1100 lacks f64 atomic-add | Run the 3-candidate micro-benchmark on Kaggle CUDA (+ ROCm smoke); record in SPIKE-REDUCTION.md |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 180s (in-env)
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
