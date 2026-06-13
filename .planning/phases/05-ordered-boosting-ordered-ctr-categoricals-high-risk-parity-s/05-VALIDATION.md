---
phase: 5
slug: ordered-boosting-ordered-ctr-categoricals-high-risk-parity-s
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-06-14
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from `05-RESEARCH.md` § "Validation Architecture". This is the project's
> highest-risk parity slice — per-object oracles are the point, not an extra.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[test]` + `cb-oracle::compare_stage` ≤1e-5 gate |
| **Config file** | none (Cargo workspace); fixtures under `crates/cb-oracle/fixtures/` |
| **Quick run command** | `cargo test -p cb-train` (also `-p cb-model`, `-p cb-oracle`) |
| **Full suite command** | `cargo test -p cb-train -p cb-model -p cb-oracle` (⚠ NOT `--workspace` — MLIR/disk; see STATE.md Blockers) |
| **Estimated runtime** | ~30s per owning-crate quick run; ~minutes per-crate suite |

---

## Sampling Rate

- **After every task commit:** Run the single owning-stage test (e.g. `cargo test -p cb-train permutation`) — < 30s.
- **After every plan wave:** Run `cargo test -p cb-train -p cb-model -p cb-oracle`.
- **Before `/gsd-verify-work`:** All ORD-01..ORD-05 oracles green — per-crate (NOT `--workspace`, MLIR/disk).
- **Max feedback latency:** 30 seconds (single-stage), minutes (per-wave).

---

## Per-Task Verification Map

> Plan/task IDs assigned by the planner. Requirement → oracle mapping is locked here; the planner
> attaches `<automated>` verify blocks to the owning task. Float comparisons ≤1e-5; integer
> num/denom and permutation indices compared EXACTLY. D-03 ordering: `Stage::Permutation` must pass
> before any value stage runs.

| Requirement | Behavior | Test Type | Automated Command | Threat Ref | File Exists | Status |
|-------------|----------|-----------|-------------------|------------|-------------|--------|
| ORD-01 | Permutation reproduces upstream Fisher-Yates exactly (per fold) | unit (exact int) | `cargo test -p cb-train permutation` | — | ❌ W0 | ⬜ pending |
| ORD-01 | TFold body/tail prefix boundaries match `SelectMinBatchSize`/`SelectTailSize` | unit | `cargo test -p cb-train fold_prefix` | — | ❌ W0 | ⬜ pending |
| ORD-02 | Per-object ordered approx per iteration ≤1e-5 (`Stage::OrderedApprox`, indirect anchor) | oracle | `cargo test -p cb-train ordered_boost_oracle` | — | ❌ W0 | ⬜ pending |
| ORD-02 | Ordered final prediction ≤1e-5 vs upstream | oracle | `cargo test -p cb-model ordered_predict_oracle` | — | ❌ W0 | ⬜ pending |
| ORD-03 | Each of 6 CTR types: per-object online num/denom (exact) + value ≤1e-5 | oracle (×6) | `cargo test -p cb-train ctr_<type>_oracle` | — | ❌ W0 | ⬜ pending |
| ORD-03 | Plain-mode CTR (whole-set) ≤1e-5 — locked BEFORE ordered (D-06) | oracle | `cargo test -p cb-train plain_ctr_oracle` | — | ❌ W0 | ⬜ pending |
| ORD-04 | One-hot path selection at `count==one_hot_max_size` (incl) and `+1` (CTR) | unit | `cargo test -p cb-train one_hot_threshold` | — | ❌ W0 | ⬜ pending |
| ORD-04 | One-hot-only model trains+predicts ≤1e-5 (no permutation present) | oracle | `cargo test -p cb-model one_hot_predict_oracle` | — | ❌ W0 | ⬜ pending |
| ORD-05 | Tensor CTR (`max_ctr_complexity`) projection enumeration + ≤1e-5 | oracle | `cargo test -p cb-model tensor_ctr_oracle` | — | ❌ W0 | ⬜ pending |
| (model) | `ctr_data` `.cbm`/`model.json` round-trip + upstream load ≤1e-5 | oracle | `cargo test -p cb-model ctr_data_roundtrip` | T-05-V5 | ❌ W0 | ⬜ pending |
| (security) | Malformed `ctr_data` blob → typed `ModelError`, never panic | unit | `cargo test -p cb-model ctr_data_malformed` | T-05-V5 | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `crates/cb-oracle/src/compare.rs` — add `Stage::Permutation`, `Stage::OnlineCtr`, `Stage::OrderedApprox` variants.
- [ ] `crates/cb-oracle/src/model_json.rs` — add `ctr_data` parsing (currently borders-only).
- [ ] `crates/cb-oracle/generator/ordered_oracle.cpp` — transcribed standalone harness (transcribe-then-self-oracle fallback; zero catboost includes).
- [ ] `crates/cb-oracle/fixtures/` — purpose-built categorical fixtures (D-08): low-card (one-hot), high-card (CTR), tiny N; per-fold permutation `.npy`; per-object num/denom/approx `.npy`; per-CTR-type config (D-07).
- [ ] D-03 ordering harness: assert `Stage::Permutation` exact before value stages run.
- [ ] Framework install: none — existing `#[test]` + `compare_stage` suffice.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Offline fixture generation (Python `catboost==1.2.10` + `ordered_oracle.cpp`) | ORD-01..05 | Generators run OFFLINE, never in CI (D-09); `.npy` outputs are committed frozen | Run generator locally with pinned `catboost==1.2.10`, `thread_count=1`; commit `.npy` under `crates/cb-oracle/fixtures/`; CI consumes frozen fixtures only |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s (single-stage)
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** planned — all ORD-01..05 requirements have <automated> verify blocks attached to owning tasks; Wave 0 (05-01) stands up every MISSING reference (Stage variants, ctr_data parsing, ordered_oracle.cpp, fixtures, D-03 ordering). wave_0_complete flips true on 05-01 execution.
