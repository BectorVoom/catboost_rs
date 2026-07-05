---
phase: 15
slug: debt-discharge-cuda-oracle-re-establishment
status: approved
nyquist_compliant: true
wave_0_complete: false
created: 2026-07-05
---

# Phase 15 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> In this phase the oracles ARE the validation. ROCm in-env = non-gating smoke; the single Kaggle CUDA session is authoritative for ε=1e-4 numeric parity.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[test]` (sibling `*_test.rs`), `approx` for float asserts |
| **Config file** | none — cargo test |
| **Quick run command** | `cargo test -p cb-backend --no-default-features --features cpu <test_name>` (compiles, CPU-skips the ε assert) |
| **Full suite command** | `cargo test -p cb-backend --no-default-features --features rocm` (in-env smoke) + Kaggle `--features cuda` session (authoritative) |
| **Estimated runtime** | ~60s in-env rocm suite; ~15min Kaggle CUDA session |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p cb-backend --no-default-features --features cpu` (compile + CPU-skip) then `--features rocm` in-env smoke.
- **After every plan wave:** Run the full `cb-backend --features rocm` suite in-env.
- **Before `/gsd-verify-work`:** The single Kaggle CUDA session ALL-PASS (correctness) before any speed number, then BENCH-03 recompute.
- **Max feedback latency:** ~60s in-env (Kaggle session is the phase gate, not per-task).

---

## Per-Task Verification Map

| Task ID | Wave | Requirement | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|-------------|-----------------|-----------|-------------------|-------------|--------|
| RV-13-01 | A | HARD-03 | N/A | unit/kernel | `cargo test -p cb-backend --features cuda tie_order` | ❌ W0 (add to `ranking_stoch_test.rs`) | ⬜ pending |
| RV-13-02 | A | HARD-03 | N/A | unit/kernel | `cargo test -p cb-backend --features cuda softmax_weight_max_seed` | ❌ W0 (add to `ranking_stoch_test.rs`) | ⬜ pending |
| RV-13-03 | A | HARD-03 | N/A | unit/kernel | `cargo test -p cb-backend --features rocm empty_group_means` | ⚠️ extends `query_helper_test.rs` | ⬜ pending |
| RV-13-04 | A | HARD-03 | N/A | unit/kernel | `cargo test -p cb-backend --features cuda pairwise_near_equal_tiebreak` | ❌ W0 (add to pairwise/cholesky test) | ⬜ pending |
| HARD-01 | B | HARD-01 | N/A | integration | Kaggle `oracle.py` Part A (44 self-oracles + 4 RV-13 oracles, ε=1e-4, one session) | ❌ W0 (`bench/phase15_cuda_oracle/oracle.py`) | ⬜ pending |
| HARD-02 | B | HARD-02 | N/A | integration | Kaggle `oracle.py` Part B (depth-1 + depth-6 BENCH-02 rows, one session) | ❌ W0 (same runner) | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] RV-13-01 tie oracle in `ranking_stoch_test.rs`
- [ ] RV-13-02 weight>0-max-seed oracle in `ranking_stoch_test.rs`
- [ ] RV-13-03 empty-group oracle extending `query_helper_test.rs`
- [ ] RV-13-04 near-equal-border oracle in the pairwise/cholesky test
- [ ] `bench/phase15_cuda_oracle/oracle.py` single-session runner (Part A correctness gate + Part B timing)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Single-session Kaggle CUDA run (one P100 / driver / seed) | HARD-01, HARD-02 | Requires GPU hardware not available in-env; verifier subagent cannot run GPU — orchestrator drives the `kaggle` CLI | Drive `kaggle` CLI: `git archive` tracked-source tarball → push notebook → background-poll `KernelWorkerStatus` until COMPLETE (~15min); auth via `~/.kaggle/access_token` |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references (RV-13-01..04 oracles + `oracle.py` — authored in 15-01/15-02/15-03)
- [x] No watch-mode flags
- [x] Feedback latency < 60s in-env
- [x] `nyquist_compliant: true` set in frontmatter

> `wave_0_complete` stays `false` until the sibling `*_test.rs` oracle files exist on disk (written during Wave 1 execution). Plan-content nyquist compliance is confirmed (plan-checker PASS, Dimension 8).

**Approval:** approved 2026-07-05
