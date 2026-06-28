---
phase: 5
slug: ordered-boosting-ordered-ctr-categoricals-high-risk-parity-s
status: audited
nyquist_compliant: false
nyquist_blocked_reason: "2 e2e oracles (ORD-02 final-prediction, ORD-05 categorical train→predict) are data-blocked on offline catboost==1.2.10 fixtures; test code exists, compiles, and is not #[ignore]'d — the gap is fixture data, not test coverage"
wave_0_complete: true
created: 2026-06-14
audited: 2026-06-14
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from `05-RESEARCH.md` § "Validation Architecture". This is the project's
> highest-risk parity slice — per-object oracles are the point, not an extra.
> **Audited 2026-06-14** after execution (10 plans incl. gap-closure 05-08/09/10):
> Per-Task Map statuses below reflect ACTUAL landed tests, not planning-time placeholders.

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

> Requirement → oracle mapping. Float comparisons ≤1e-5; integer num/denom and
> permutation indices compared EXACTLY. D-03 ordering: `Stage::Permutation` passes
> before any value stage. **Automated Command / File / Status columns reflect the
> actual landed tests (audited 2026-06-14).**

| Requirement | Behavior | Test Type | Automated Command (actual) | Threat Ref | File | Status |
|-------------|----------|-----------|----------------------------|------------|------|--------|
| ORD-01 | Permutation reproduces upstream Fisher-Yates exactly (per fold, incl. fold k>0 after 05-07) | unit + oracle (exact int) | `cargo test -p cb-train --test permutation_oracle_test` | T-05-03-02 | `tests/permutation_oracle_test.rs`, `src/permutation_test.rs` | ✅ green (3/3 + lib) |
| ORD-01 | TFold body/tail prefix boundaries match `SelectMinBatchSize`/`SelectTailSize` | unit | `cargo test -p cb-train --lib fold::` | T-05-03-01 | `src/fold_test.rs` | ✅ green (lib 128/128) |
| ORD-02 | Per-object ordered approx per iteration ≤1e-5 (`Stage::OrderedApprox`, indirect anchor) | oracle | `cargo test -p cb-train --test ordered_boost_oracle_test` | T-05-05-02 | `tests/ordered_boost_oracle_test.rs` | ✅ green (5/5) |
| ORD-02 | Ordered split scoring (segment-summed L2 over BodyTailArr) + structure ≠ Plain | unit + wiring | `cargo test -p cb-train --lib tree::ordered && cargo test -p cb-train --test ordered_boost_wiring_test` | T-05-08-01..03 | `src/tree_ordered_test.rs`, `tests/ordered_boost_wiring_test.rs` | ✅ green (8 + 3/3) |
| ORD-02 | Ordered final prediction ≤1e-5 vs upstream (FULL multi-tree, no `#[ignore]`, via `cb_model::predict_raw`) | oracle (e2e) | `cargo test -p cb-train --test ordered_boost_e2e_oracle_test` | T-05-10-01/V5 | `tests/ordered_boost_e2e_oracle_test.rs` | ⚠️ PARTIAL — data-blocked (offline fixtures) |
| ORD-03 | Each CTR type: per-object online num/denom (exact) + value ≤1e-5 (plain + ordered) | oracle | `cargo test -p cb-train --test ordered_ctr_oracle_test` | T-05-05-01 | `tests/ordered_ctr_oracle_test.rs`, `src/ctr/online_test.rs`, `src/ctr/calc_ctr_test.rs` | ✅ green (3/3 + lib) |
| ORD-03 | Plain-mode CTR (whole-set) ≤1e-5 — locked BEFORE ordered (D-06) | oracle | `cargo test -p cb-train --test plain_ctr_oracle_test` | T-05-04-01/02 | `tests/plain_ctr_oracle_test.rs` | ✅ green (3/3) |
| ORD-04 | One-hot path selection at `count==one_hot_max_size` (incl) and `+1` (CTR) | unit | `cargo test -p cb-train --lib candidates::` | T-05-02-01 | `src/candidates_test.rs` | ✅ green (lib) |
| ORD-04 | One-hot-only model trains+predicts ≤1e-5 (no permutation present) | oracle | `cargo test -p cb-train --test one_hot_oracle_test` | T-05-02-02 | `tests/one_hot_oracle_test.rs` | ✅ green (4/4) |
| ORD-05 | Tensor CTR (`max_ctr_complexity`) projection enumeration + per-object ≤1e-5 | oracle | `cargo test -p cb-train --test tensor_ctr_oracle_test` | T-05-06-01/V5 | `tests/tensor_ctr_oracle_test.rs`, `src/projection_test.rs` | ✅ green (3/3 + lib) |
| ORD-05 | Tensor-CTR categorical model trains→predicts ≤1e-5 end-to-end (`ModelSplit::Ctr` via `cb_model::predict_raw`, no `#[ignore]`) | oracle (e2e) | `cargo test -p cb-train --test tensor_ctr_e2e_oracle_test` | T-05-09-V5/01/02/03 | `tests/tensor_ctr_e2e_oracle_test.rs` | ⚠️ PARTIAL — data-blocked (offline fixtures) |
| (model) | `ctr_data` `.cbm`/`model.json` round-trip + upstream load ≤1e-5 | oracle | `cargo test -p cb-model --test ctr_data_roundtrip_test` | T-05-04-V5 | `tests/ctr_data_roundtrip_test.rs` | ✅ green (5/5) |
| (security) | Malformed `ctr_data` blob → typed `ModelError`, never panic | unit | `cargo test -p cb-model --lib ctr_data` | T-05-04-V5 | `src/ctr_data_test.rs` | ✅ green (lib 15/15) |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ PARTIAL (data-blocked) · 🚫 missing*

**Coverage summary:** 11/13 rows ✅ green · 2/13 ⚠️ PARTIAL (both data-blocked on the
same offline fixture step — see Manual-Only) · 0 MISSING. Every requirement has a real,
wired test file; no test is `#[ignore]`'d.

---

## Wave 0 Requirements — COMPLETE ✅

- [x] `crates/cb-oracle/src/compare.rs` — `Stage::Permutation`, `Stage::OnlineCtr`, `Stage::OrderedApprox`, `Stage::Predictions`, `Stage::Approx` present.
- [x] `crates/cb-oracle/src/model_json.rs` — `ctr_data` parsing present (`#[serde(default)]`, typed `OracleError`).
- [x] `crates/cb-oracle/generator/ordered_oracle.cpp` — transcribed standalone harness (continuous-stream multi-fold fix landed in 05-07).
- [x] `crates/cb-oracle/fixtures/` — categorical fixtures present: `one_hot_cat`, `plain_ctr`, `ordered_ctr`, `ordered_boost`, `tensor_ctr`, `cat_hash`, + skeletons. (Exception: the two NEW e2e fixture dirs `ordered_boost_e2e/` + `tensor_ctr_e2e/` are pending offline generation — see Manual-Only.)
- [x] D-03 ordering harness: `Stage::Permutation` exact before value stages (per_stage_oracle_test.rs).
- [x] Framework install: none — existing `#[test]` + `compare_stage` suffice.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Offline fixture generation (Python `catboost==1.2.10` + `ordered_oracle.cpp`) | ORD-01..05 | Generators run OFFLINE, never in CI (D-09); `.npy` outputs are committed frozen | Run generator locally with pinned `catboost==1.2.10`, `thread_count=1`; commit `.npy` under `crates/cb-oracle/fixtures/`; CI consumes frozen fixtures only |
| **ORD-02 final-prediction e2e oracle** (`ordered_boost_e2e_oracle_test`) | ORD-02 | Test code committed & wired (no `#[ignore]`); blocked only on `ordered_boost_e2e/{config,X,y,model.json,predictions}` from offline catboost | `cd crates/cb-oracle/generator && python3 gen_fixtures.py` → commit `ordered_boost_e2e/` → `cargo test -p cb-train --test ordered_boost_e2e_oracle_test` (must pass ≤1e-5 across all 5 trees) |
| **ORD-05 categorical e2e oracle** (`tensor_ctr_e2e_oracle_test`) | ORD-05 | Test code committed & wired (no `#[ignore]`); blocked only on `tensor_ctr_e2e/{config,X_cat,y,model.json,predictions}` from offline catboost | `cd crates/cb-oracle/generator && python3 gen_fixtures.py` → commit `tensor_ctr_e2e/` → `cargo test -p cb-train --test tensor_ctr_e2e_oracle_test` (must pass ≤1e-5 across all 5 trees) |

> NOTE: the two e2e rows are **not test-coverage gaps** the Nyquist auditor can fill —
> the test source already exists and compiles. They are fixture-DATA gaps blocked on the
> manual offline catboost step, explicitly handed off during `/gsd-execute-phase 5 --gaps-only`.
> When the candidate-emission seam for tensor CTRs (the `tree.rs::CtrSplitSpec` stub noted in
> 05-09 SUMMARY) is wired alongside the ORD-05 fixture, re-run this validation to flip both
> rows to ✅ and set `nyquist_compliant: true`.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references (complete)
- [x] No watch-mode flags
- [x] Feedback latency < 30s (single-stage)
- [ ] `nyquist_compliant: true` — **NOT yet**: 2 e2e oracles data-blocked on offline fixtures (test code exists, no `#[ignore]`); flips true once `ordered_boost_e2e/` + `tensor_ctr_e2e/` fixtures are generated and both tests pass

**Approval:** AUDITED 2026-06-14 post-execution. 11/13 requirement rows green with real
landed tests; 0 MISSING. The 2 remaining rows (ORD-02 final-prediction, ORD-05 categorical
e2e) are PARTIAL — wired automated tests blocked solely on offline `catboost==1.2.10` fixture
generation (a documented Manual-Only step, not a coverage gap). `nyquist_compliant` is held
`false` until those fixtures land and both e2e tests pass, to avoid falsely signalling
completion to milestone audit.

## Validation Audit 2026-06-14
| Metric | Count |
|--------|-------|
| Requirement rows | 13 |
| Covered (green) | 11 |
| Partial (data-blocked) | 2 |
| Missing | 0 |
| Gaps auditor-fillable | 0 (test code exists for all; the 2 partials need fixture DATA, not tests) |
