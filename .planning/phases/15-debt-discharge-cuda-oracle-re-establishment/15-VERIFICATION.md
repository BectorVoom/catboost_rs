---
phase: 15-debt-discharge-cuda-oracle-re-establishment
verified: 2026-07-05T07:47:38Z
status: passed
score: 3/3 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 15: Debt Discharge & CUDA Oracle Re-establishment Verification Report

**Phase Goal:** Discharge the v1.1 standing debt and re-establish the trusted CUDA oracle everything downstream rests on — GPUT-14 aggregate ε=1e-4 sign-off, Phase-10/11 BENCH-02 speed rows, and the RV-13-01..04 latent parity hazards.
**Verified:** 2026-07-05T07:47:38Z
**Status:** passed
**Re-verification:** No — initial verification

## Note on GPU Verification Method

The verifier subagent cannot run GPU (established Phases 12–14). Per the phase's own design, the single authoritative CUDA oracle already ran on a real Tesla P100 (Kaggle) and its verbatim output is committed at `bench/phase15_cuda_oracle/result.json` (commit `734109a2c5767d677d985adafa7c357db3bb2e07`). This report verifies HARD-01/HARD-02 by direct inspection of that committed JSON (field-by-field, cross-checked against the derived `bench/BENCH-03-SIGNOFF.md` / `bench/RESULTS.md` prose) rather than by re-running Kaggle. HARD-03's four RV-13 oracles were independently re-executed in this session under `--features cpu` (see Behavioral Spot-Checks) and confirmed to compile and pass — this is direct code-level evidence, not a SUMMARY.md claim.

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | A single aggregate GPUT-14 ε=1e-4 Kaggle CUDA correctness row covers all v1.1 device families as one authoritative run and passes (HARD-01) | ✓ VERIFIED | `bench/phase15_cuda_oracle/result.json`: `correctness_verdict: "ALL-PASS"`, all 13 families (`families` dict) each `exit: 0`, `ran_any_tests: true`; `rv13_oracles_expected == rv13_oracles_seen` (4/4, exact set match, order-independent). `provenance.single_session: true`, one GPU (Tesla P100-PCIE-16GB), one driver (580.159.04), one CUDA ver (release 12.8), one seed (42) — no stitching. Committed verbatim at `734109a`. |
| 2 | Phase-10 (depth-1) and Phase-11 (depth-6) BENCH-02 speed rows are executed on Kaggle CUDA and the BENCH-03 aggregate is recomputed from real numbers with no stitched Phase-12/13-only gaps (HARD-02) | ✓ VERIFIED | `result.json.bench02.depth_rows`: 12 rows present — depth-1 {depthwise, region} × n={100000,300000,1000000} and depth-6 {depthwise, region} × n={10000,100000,300000}, each with `device_s`/`host_cpu_s`/`catboost_gpu_s`/`speedup`/`device_ge_cpu`, all `device_ge_cpu: true`, speedups 29.147×–40.757×; Region rows carry `catboost_gpu_s: null` (N/A, never fabricated). `crossover.note: "device first beats CPU at n=100000"`. `bench/BENCH-03-SIGNOFF.md` and `bench/RESULTS.md` both quote these exact numbers with zero deviation; `bench/RESULTS.md` has 0 occurrences of "TBD" on any real data row (template placeholder block at :49-72 is explicitly labeled "Run template — copy this block", not a live gap). |
| 3 | Each RV-13-01..04 latent parity hazard is either fixed (with an oracle demonstrating the fix) or explicitly retired with recorded evidence (HARD-03) | ✓ VERIFIED | All 4 oracle tests exist and independently re-run PASS in this verification session under `cargo test -p cb-backend --no-default-features --features cpu`: `tie_order_matches_cpu_stable_descending` (RV-13-01, confirmatory — record-only order assert gated behind `device_backend_active()` since `plane_inclusive_sum` panics on cubecl-cpu, documented and expected), `softmax_weight_max_seed` (RV-13-02, real fix — `compute_group_max_weighted_host` wired into `query_softmax_ders_host`), `empty_group_means_no_fault` (RV-13-03, real fix — `if n == 0 { return Ok(vec![0.0; n_groups]); }` at query_helper.rs:390, precedes `client.create` at :427), `pairwise_near_equal_border_tiebreak` (RV-13-04, real fix — `pub(crate) fn select_best_candidate` at pairwise.rs:1819, wired into `select_best_split_over_scores` at :1933, single source of truth). All 4 also listed in `result.json` `rv13_oracles_seen` inside their correct device family (ranking / pairwise). `15-EVIDENCE.md` records the four-field D-10 evidence (what diverged / oracle / fix / passing result) per hazard, matching the code. `cargo check -p cb-backend --tests` compiles clean (verified EXIT=0 in this session). |

**Score:** 3/3 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/cb-backend/src/gpu_runtime/ranking.rs` | `pub(crate) fn descending_order_per_query`, `compute_group_max_weighted_host` wired into `query_softmax_ders_host` | ✓ VERIFIED | grep confirms both symbols present at expected locations (:784, :382); no debt markers (TBD/FIXME/XXX/TODO/HACK/PLACEHOLDER) found |
| `crates/cb-backend/src/gpu_runtime/ranking_stoch_test.rs` | RV-13-01/02 oracles | ✓ VERIFIED | `tie_order_matches_cpu_stable_descending` (:190), `softmax_weight_max_seed` (:304) — both compile and pass under `--features cpu` |
| `crates/cb-backend/src/kernels/query_helper.rs` | `n==0` guard before `client.create` | ✓ VERIFIED | Guard at :390-392, `client.create` first appears at :427 — guard correctly precedes it; returns `vec![0.0; n_groups]` (right length), not `Vec::new()` |
| `crates/cb-backend/src/kernels/query_helper_test.rs` | RV-13-03 oracle | ✓ VERIFIED | `empty_group_means_no_fault` (:195), passes, asserts exact value `[0.0]` |
| `crates/cb-backend/src/gpu_runtime/pairwise.rs` | near-equal-tolerant lowest-index tie-break, extracted `pub(crate)` selector | ✓ VERIFIED | `REL_TOL: f64 = 1e-9` (:1807), `select_best_candidate` (:1819), called from `select_best_split_over_scores` at :1933 — single source of truth confirmed by direct code read |
| `crates/cb-backend/src/kernels/cholesky_solve_test.rs` | RV-13-04 oracle | ✓ VERIFIED | `pairwise_near_equal_border_tiebreak` (:269), passes, demonstrates the retired exact-`==` rule flips while the new rule agrees |
| `bench/phase15_cuda_oracle/oracle.py` | single-session runner | ✓ VERIFIED | exists, referenced/committed alongside `result.json`; not independently re-run (GPU unavailable to verifier), authoritative output already committed |
| `bench/phase15_cuda_oracle/kernel-metadata.json` | Kaggle kernel metadata | ✓ VERIFIED | exists per SUMMARY; not disputed |
| `bench/phase15_cuda_oracle/result.json` | one verdict + one JSON | ✓ VERIFIED | 612 lines, `correctness_verdict: ALL-PASS`, `provenance.single_session: true`, 12 `bench02.depth_rows`, committed at `734109a` |
| `.planning/phases/15-.../15-EVIDENCE.md` | per-hazard RV-13-01..04 evidence | ✓ VERIFIED | all 4 hazard sections present with what-diverged/oracle/fix/passing-result fields, cross-checked against code and result.json |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `ranking_stoch_test.rs` | `ranking.rs` | direct invocation of `descending_order_per_query` / `query_softmax_ders_host` / `compute_group_max_weighted_host` | ✓ WIRED | confirmed by successful compile + pass of both named tests |
| `query_helper_test.rs` | `query_helper.rs` | direct invocation of `compute_group_means_host` with `q_offsets=[0,0]` | ✓ WIRED | test passes, asserts `Ok(vec![0.0])` |
| `cholesky_solve_test.rs` | `pairwise.rs` | direct invocation of `select_best_candidate` | ✓ WIRED | test passes; production `select_best_split_over_scores` calls the SAME function (pairwise.rs:1933), confirmed by direct read — not a divergent copy |
| `bench/phase15_cuda_oracle/oracle.py` | `crates/cb-backend` | `cargo test --release --no-default-features --features cuda -p cb-backend` (Part A) including 4 RV-13 oracle names | ✓ WIRED | `result.json` families dict shows the ranking family's `rv13_oracles_seen` = 3 names, pairwise family's = 1 name, matching exactly the plan's declared routing |
| `bench/RESULTS.md` / `bench/BENCH-03-SIGNOFF.md` | `bench/phase15_cuda_oracle/result.json` | TBD cells filled from single-session JSON | ✓ WIRED | every number in both docs (31.045×, 33.112×, 32.546×, 40.669×, 40.757×, 39.164×, 30.704×, 36.982×, 40.312×, 29.147×, 40.381×, 39.477×; crossover n=100000; provenance P100/580.159.04/CUDA12.8/seed42) cross-checked byte-for-byte against `result.json` — exact match, zero discrepancy |
| `.planning/MILESTONES.md` | `bench/BENCH-03-SIGNOFF.md` | Known-Gaps rows discharged, referencing recomputed sign-off | ✓ WIRED | both v1.1 Known Gaps rows explicitly marked "DISCHARGED (v1.2 Phase 15)" with correct cross-reference |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| RV-13-01 tie-order oracle passes | `cargo test -p cb-backend --no-default-features --features cpu tie_order_matches_cpu_stable_descending -- --nocapture` | `test ... ok` — 1 passed; order assert correctly skipped on cpu (plane_inclusive_sum unsupported), record-only note printed as documented | ✓ PASS |
| RV-13-02 weight-max-seed oracle passes | `cargo test -p cb-backend --no-default-features --features cpu softmax_weight_max_seed -- --nocapture` | `test ... ok` — 1 passed; seed-selection assert ran and passed | ✓ PASS |
| RV-13-03 empty-group guard oracle passes | `cargo test -p cb-backend --no-default-features --features cpu empty_group_means_no_fault -- --nocapture` | `test ... ok` — 1 passed; `empty-group means = [0.0]` | ✓ PASS |
| RV-13-04 near-equal tie-break oracle passes | `cargo test -p cb-backend --no-default-features --features cpu pairwise_near_equal_border_tiebreak -- --nocapture` | `test ... ok` — 1 passed; `device=0 host=0 (exact flip 0->1); separated winner=1` | ✓ PASS |
| No `cb-train` dependency in `cb-backend` | `grep -c 'cb-train\|cb_train' crates/cb-backend/Cargo.toml` | `0` | ✓ PASS |
| `cargo check -p cb-backend --tests` compiles clean | `cargo check -p cb-backend --tests` | exit 0, only pre-existing unrelated warnings (unused `mut` in `nonsym_grow_test.rs`) | ✓ PASS |
| Pre-existing (non-regression) cpu-backend pairwise gap confirmed unrelated | `cargo test -p cb-backend --no-default-features --features cpu pairwise` | 9 passed, 10 failed — all traced to `not yet implemented: atomic<f64>` in `cubecl-cpu` (histogram/der-sum scatter kernels, unrelated to the RV-13-04 argmax code touched by this phase) | ℹ️ INFO (documented pre-existing debt, not a phase-15 regression) |

### Requirements Coverage

| Requirement | Source Plan(s) | Description | Status | Evidence |
|--------------|-----------------|--------------|--------|----------|
| HARD-01 | 15-03, 15-04 | Aggregate ε=1e-4 Kaggle CUDA correctness sign-off (GPUT-14) across all v1.1 device families as one authoritative row | ✓ SATISFIED | `result.json.correctness_verdict = ALL-PASS`; REQUIREMENTS.md `[x]` / Complete |
| HARD-02 | 15-03, 15-04 | Phase-10/11 BENCH-02 speed rows executed on Kaggle CUDA; BENCH-03 aggregate completed with real numbers (no stitched gaps) | ✓ SATISFIED | 12 real depth rows in `result.json`, no fabricated cells, `BENCH-03: PASS`; REQUIREMENTS.md `[x]` / Complete |
| HARD-03 | 15-01, 15-02, 15-04 | RV-13-01..04 latent parity hazards resolved (or explicitly retired with evidence) | ✓ SATISFIED | 4/4 oracles exist, pass, and are counted in the Kaggle session; REQUIREMENTS.md `[x]` / Complete |

No orphaned requirements: `.planning/REQUIREMENTS.md` maps only HARD-01/02/03 to "Phase 15" and all three appear in the combined `requirements:` fields of plans 15-01 through 15-04.

### Anti-Patterns Found

None. No `TBD`/`FIXME`/`XXX`/`TODO`/`HACK`/`PLACEHOLDER` markers found in any of the six code files modified by 15-01/15-02. `bench/RESULTS.md` has zero live `TBD` occurrences (the only `<PASS/FAIL>`/`<value>` placeholders are inside an explicitly-labeled reusable "Run template" code block, not a live data gap).

### Human Verification Required

None. The one item that would ordinarily require human judgment — whether the committed `bench/phase15_cuda_oracle/result.json` genuinely reflects a real, un-tampered Kaggle P100 run rather than a hand-authored fabrication — is addressed by the phase's own design decision (accepted for this phase per the critical_context instructions): the file is git-committed as an atomic 613-line commit (`734109a`) immediately following the runner-authoring commit (`5d07c67`), its internal structure (per-family compiler warnings, cargo build timings, `secs: 812.9` etc.) is consistent with genuine `cargo test --release` output rather than synthesized JSON, and every number quoted in `bench/BENCH-03-SIGNOFF.md`/`bench/RESULTS.md` traces back to it exactly. This is accepted as sufficient evidence per the task's explicit instruction not to attempt to re-run the Kaggle oracle.

### Gaps Summary

No gaps. All three must-haves (HARD-01, HARD-02, HARD-03) are independently verified: HARD-01/HARD-02 by direct inspection of the committed, git-traceable `result.json` and its faithful transcription into `BENCH-03-SIGNOFF.md`/`RESULTS.md`; HARD-03 by re-executing all 4 RV-13 oracle tests in this verification session (not merely trusting the SUMMARY) and confirming the production code wiring (`select_best_candidate`, `compute_group_max_weighted_host`, the `n==0` guard, `pub(crate) descending_order_per_query`) is real, not cosmetic. Bookkeeping (REQUIREMENTS.md, MILESTONES.md, STATE.md) is consistently flipped with no stray live "Pending" claims (only intentionally-preserved historical/archival records remain, as documented in 15-04-SUMMARY.md and confirmed by grep). The one pre-existing cpu-backend limitation found (`atomic<f64>` unimplemented in cubecl-cpu, affecting unrelated pairwise histogram tests) is confirmed via direct reproduction to be unrelated to this phase's RV-13-04 change and is explicitly documented as pre-existing, non-regressing debt in 15-02-SUMMARY.md.

---

*Verified: 2026-07-05T07:47:38Z*
*Verifier: Claude (gsd-verifier)*
