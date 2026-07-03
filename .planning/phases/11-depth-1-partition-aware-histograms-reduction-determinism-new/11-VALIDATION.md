---
phase: 11
slug: depth-1-partition-aware-histograms-reduction-determinism-new
status: approved
nyquist_compliant: true
wave_0_complete: false
created: 2026-07-03
---

# Phase 11 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from `11-RESEARCH.md` §Validation Architecture. GPU oracle authority = Kaggle CUDA (human-gated); ROCm in-env is a smoke convenience only.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[test]` (source/test separation mandatory — dedicated test files, never `mod tests` in production source) |
| **Config file** | none — Cargo workspace; `cb-backend` feature-gated (`cpu`/`wgpu`/`cuda`/`rocm`) |
| **Quick run command** | `cargo test -p cb-compute` (CPU oracle math) |
| **Full suite command** | `cargo test -p cb-backend --no-default-features --features rocm` (in-env GPU smoke) + Kaggle CUDA notebook (authoritative correctness + speed) |
| **Estimated runtime** | ~60s CPU quick; ~3–5 min rocm smoke; Kaggle run human-gated |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p cb-compute` (fast CPU oracle) + `cargo test -p cb-backend --features rocm <touched kernel>` after any `#[cube]` change (the rocm `-inf`/JIT landmine mandates an in-env GPU run — cpu/wgpu can false-pass it).
- **After every plan wave:** Run `cargo test -p cb-backend --no-default-features --features rocm` (full in-env GPU smoke).
- **Before `/gsd-verify-work`:** CPU suite green + rocm smoke green + Kaggle CUDA correctness (blocking) and speed sign-off logged in `bench/RESULTS.md`.
- **Max feedback latency:** ~60s (CPU quick path); GPU authority is asynchronous/human-gated by design.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 11-01-xx | 01 | 1 | GPUT-05, GPUT-07 | — / input-size validation | Host validates buffer sizing before device dispatch (no OOB) | unit (fixture/oracle) | `cargo test -p cb-compute leaf` (depth-6 fixture + iterations=1 assert) | ❌ W0 (extend synthetic generator to depth-6 RMSE + Logloss) | ⬜ pending |
| 11-02-xx | 02 | 2 | GPUT-05, GPUT-06 | — | Host validates buffer sizing before device dispatch (no OOB) | unit (self-oracle) | `cargo test -p cb-backend --features rocm grow_loop` (partition-aware hist + subtraction) | ❌ W0 (extend `grow_loop.rs` depth>1) | ⬜ pending |
| 11-02-xx | 02 | 2 | GPUT-06 | — | N/A | unit | `cargo test -p cb-backend --features rocm reduce` (fixed-point Atomic<u64> accumulator, zero spread) | ✅ (SPIKE-REDUCTION harness) | ⬜ pending |
| 11-03-xx | 03 | 3 | GPUT-05, GPUT-06 | — | N/A | unit | `cargo test -p cb-backend --features rocm leaf_of_matches_cpu` (depth-6 grow self-oracle + zero run-to-run spread) | ✅ (depth-1; extend to depth-6) | ⬜ pending |
| 11-04-xx | 04 | 4 | GPUT-07 | — | N/A | unit | `cargo test -p cb-backend --features rocm newton` (Σder2 channel + `newton_leaf_delta` + `apply_leaf_delta`) | ❌ W0 | ⬜ pending |
| 11-04-xx | 04 | 4 | GPUT-07 | — | N/A | unit | `cargo test -p cb-compute leaf` (`newton_leaf_delta` CPU oracle; RMSE der2=−1 collapse cross-check) | ✅ | ⬜ pending |
| 11-05-xx | 05 | 5 | GPUT-14 | — | N/A | integration | Kaggle `bench/cuda_oracle.ipynb` — final-prediction ε=1e-4 over full depth-6 run (human-gated) | ❌ W0 (extend harness) | ⬜ pending |
| 11-05-xx | 05 | 5 | GPUT-14, GPUT-06 | — | N/A | integration | Kaggle harness per-tree split-agreement + run-to-run spread diagnostic (D-05) | ❌ W0 | ⬜ pending |
| 11-05-xx | 05 | 5 | BENCH-02 | — | N/A | benchmark | Kaggle harness speed cell (warm-run/JIT-excluded) → `bench/RESULTS.md` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky. Task IDs are placeholders — see each PLAN.md for concrete IDs.*

---

## Wave 0 Requirements

- [ ] Depth-6 RMSE + Logloss self-oracle in `crates/cb-backend/src/kernels/grow_loop.rs` — covers GPUT-05/07.
- [ ] `reduce_leaf_der2` / `newton_leaf_delta` device-vs-CPU self-oracle — covers GPUT-07.
- [ ] Extend the Phase-10 synthetic generator to depth-6 RMSE + Logloss configs (D-03) — one generator produces the ≤1e-4 correctness fixture AND the large-n speed workload.
- [ ] Extend `bench/cuda_oracle.ipynb`: final-ε=1e-4 gate + per-tree split-agreement/run-to-run-spread diagnostic + depth-6 speed cells (D-05, BENCH-02).
- [ ] rocm in-env smoke assertion after each `#[cube]` change (cpu/wgpu can false-pass the `-inf`/JIT landmine).

*Existing infrastructure (Phase 7/10) covers: deterministic reduce harness, der seams, partition-split/update, `apply_leaf_delta`, cindex, two-level scan.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Final-prediction ε=1e-4 over full depth-6 run (RMSE + Logloss) on CUDA | GPUT-14 | Authoritative GPU is Kaggle CUDA (no in-env CUDA; gfx1100 lacks f64 atomic-add for the smoke path) | Run `bench/cuda_oracle.ipynb` on Kaggle; assert max abs(device − CPU) prediction ≤ 1e-4 |
| Depth-6 device vs host-CPU vs official CatBoost GPU wall-clock | BENCH-02 | Speed of record is warm-run Kaggle CUDA; not reproducible in-env | Run Kaggle speed cell (JIT-excluded, train-only); log to `bench/RESULTS.md` |
| Per-tree split-agreement + spread diagnostic | GPUT-06/GPUT-14 (D-05) | Catches compounding drift at the originating tree; needs the full boosting run on CUDA | Kaggle harness diagnostic cell; assert no split flips compounding across hundreds of trees |

*CPU-side leaf math, partition/leaf_of equivalence, and reduce determinism are all automated; only the CUDA-authoritative ε and speed gates are human-gated.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [x] Feedback latency < 60s (CPU path)
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-07-03 (aligned to the committed 5-plan structure 11-01…11-05 after plan-checker review)
