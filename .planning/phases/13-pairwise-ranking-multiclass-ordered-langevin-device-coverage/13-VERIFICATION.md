---
phase: 13-pairwise-ranking-multiclass-ordered-langevin-device-coverage
verified: 2026-07-04T00:00:00Z
status: passed
score: 9/9 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 13: Pairwise, Ranking, Multiclass, Ordered & Langevin Device Coverage Verification Report

**Phase Goal:** Expand the device path across the loss-family / multi-output / ordered-residency families — PairLogit pairwise (+ batched Cholesky solver), query/listwise ranking, multiclass/multi-target/uncertainty, ordered boosting, and Langevin/SGLB noise — each self-oracled and signed off ≤1e-4 on Kaggle CUDA, while every family still declines to `Ok(None)`→CPU (no incorrect device result) pending the per-tree grow-seam forward dependency.

**Verified:** 2026-07-04
**Status:** passed
**Re-verification:** No — initial verification

**Scope note (honored per phase_context):** This phase's MVP boundary is explicit and documented in-code and in `13-COVERAGE-MATRIX.md`: all five families land device der-drivers / solvers / grouping infra / noise kernels + self-oracles + structural coverage-gate seams, but `GpuTrainSession::begin()` uniformly declines to `Ok(None)`→CPU for every one of them because the per-tree grow seam (`Runtime::grow_tree_on_device`, which carries only scalar `approx`/`target` today) is a forward dependency shared by all five families. Verification below checks that the **numerics/kernels/oracles exist, are wired, and are Kaggle-CUDA-signed-off** — not that end-to-end device training is wired for these families (that is explicitly out of scope this phase, by design).

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | PairLogit pairwise device path (2×2-cell histogram reuse + per-leaf matrix assembly `MakePairwiseDerivatives`/`MakePointwiseDerivatives`) is landed, self-oracled, and Kaggle-CUDA-signed-off ≤1e-4 (GPUT-11) | ✓ VERIFIED | `crates/cb-backend/src/gpu_runtime/pairwise.rs` (2149 lines) + `PairwiseState`/`map_pairwise_coverage` in `session.rs:291-360`; self-oracle `kernels/pairwise_deriv_test.rs` — `cargo test -p cb-backend --lib` local run: 4/4 pairwise_deriv tests pass. Kaggle CUDA P100 run (`bench/phase13_cuda_oracle/result.json`): `pairwise_deriv`+`cholesky_solve` **8/8 pass, exit=0** |
| 2 | Batched device f64 Cholesky solver (decomp + fwd/back subst + ridge + `CalcScoresCholesky`) is wired into the pairwise split scorer, matching CPU ≤1e-4 (GPUT-21) | ✓ VERIFIED | `crates/cb-backend/src/kernels/cholesky_solve.rs` (494 lines) + `launch_cholesky_solve` wired at `pairwise.rs:1543` (commit `c221e7c` "wire device Cholesky solve into pairwise split scorer"); self-oracle `cholesky_solve_test.rs` 4/4 local pass; included in the same Kaggle CUDA 8/8 pass above |
| 3 | Query/listwise ranking (QueryRMSE, QuerySoftMax, QueryCrossEntropy, YetiRank, PFound-F) with device query-grouping infra runs on device, ≤1e-4 (GPUT-22) | ✓ VERIFIED | `kernels/query_helper.rs` (659 lines), `gpu_runtime/ranking.rs` (1000 lines), `RankingState`/`map_ranking_coverage` (`session.rs:389-450`); local `cargo test`: 14/14 `query_helper`+`ranking_det`+`ranking_stoch` tests pass; Kaggle CUDA: **14/14 pass, exit=0**, pfound_f/yetirank der2 max_div=0.000e0. QueryCrossEntropy is explicitly, independently gated OFF (`ranking_objective_covered==false`, no CPU der oracle exists) — documented, not fabricated as covered |
| 4 | Multiclass/multi-target/uncertainty (MultiClass, MultiClassOneVsAll, MultiCrossEntropy, MultiRMSE, RMSEWithUncertainty) block-leaf K-dim Newton der2 solve runs on device, ≤1e-4 (GPUT-12) | ✓ VERIFIED | `DeviceGrownTree.approx_dim` block-leaf extension in `cb-compute/src/runtime.rs:910-945` (scalar byte-unchanged at `approx_dim==1`); `kernels/multi_newton.rs` (451 lines) + `gpu_runtime/multiclass.rs` (383 lines) + `MulticlassState`/`map_multiclass_coverage`; local: 9/9 `multiclass`+`multi_newton` tests pass; Kaggle CUDA: **9/9 pass, exit=0**. MultiRMSE has no `Loss` variant yet — self-oracle substitutes 3 other losses across both Hessian structures, documented, not fabricated |
| 5 | Ordered boosting (`EBoostingType::Ordered`) resident per-permutation approx trajectory runs on device, reproducing the frozen CPU trajectory bit-for-bit ≤1e-4 (GPUT-13) | ✓ VERIFIED | `gpu_runtime/ordered.rs` (226 lines) + `OrderedState`/`map_ordered_coverage` (`session.rs:515-555`); local: 10/10 `ordered` tests pass; Kaggle CUDA: **10/10 pass, exit=0**, abs/rel_div 0.000e0 |
| 6 | Langevin/SGLB seeded-Gaussian noise (`AddLangevinNoise`) on the resident reduced derivatives runs on device, ≤1e-4, with PairLogit+Langevin correctly declining to CPU (GPUT-20) | ✓ VERIFIED | `kernels/langevin.rs` (287 lines) + `LangevinState`/`map_langevin_coverage`/`langevin_covered_loss` (A4 pairwise-decline); local: 3/3 `langevin` tests pass; Kaggle CUDA: **3/3 pass, exit=0**, max_div ≤4.441e-16 |
| 7 | Each family carries a recorded Kaggle CUDA BENCH-02 speed measurement as it lands (SC-4) | ✓ VERIFIED (scoped) | `13-COVERAGE-MATRIX.md` records `captured-by-grow-loop` for all five families per the documented `Ok(None)` reality (no per-family end-to-end device train loop exists yet — the grow seam is the forward dependency); the shared depth-6 grow-loop anchor is measured on the SAME Kaggle CUDA P100 run at **23.9×–36.6× device≫CPU** (`result.md` BENCH-02 table, 6 rows, depthwise/region × 3 sizes). This is the phase-declared scoped interpretation of SC-4, not an unaudited gap |
| 8 | Uncovered/not-yet-flippable configs return `Ok(None)`→CPU fallback (no incorrect device result); CPU/host path stays byte-unchanged (GPUT-14 standing no-regression); GPU coverage matrix is documented (SC-5) | ✓ VERIFIED | `session.rs:850-955` shows all five family gates uniformly `return Ok(None)` for both covered and uncovered branches (code-read, not narrative); `13-COVERAGE-MATRIX.md` (181 lines) documents per-family correctness+speed+status. No-regression confirmed by full local suite (see below) |
| 9 | Kaggle CUDA sign-off evidence is present, internally consistent, and not fabricated | ✓ VERIFIED | `bench/phase13_cuda_oracle/{result.json,result.md,log-excerpt.txt,kernel-metadata.json,oracle.py}` — real GPU identification (Tesla P100-PCIE-16GB, driver 580.159.04, CUDA 13.0, nvcc 12.8), per-family exit codes/timings/test-count summaries cross-consistent between `result.json` and `result.md`, `log-excerpt.txt` shows raw `nvidia-smi` output + `CORRECTNESS_VERDICT: ALL-PASS` / `BENCH_VERDICT: OK` timestamps consistent with a ~30-minute real run (12m49s + 11m43s release builds) |

**Score:** 9/9 truths verified (0 present-but-behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/cb-backend/src/gpu_runtime/pairwise.rs` | PairLogit device path + per-leaf matrix assembly | ✓ VERIFIED | 2149 lines, wired into `session.rs`, self-oracled |
| `crates/cb-backend/src/kernels/cholesky_solve.rs` | Batched f64 SPD Cholesky solver kernel | ✓ VERIFIED | 494 lines; `launch_cholesky_solve` called from `pairwise.rs:1543` |
| `crates/cb-backend/src/kernels/query_helper.rs` | Shared device query-grouping infra | ✓ VERIFIED | 659 lines; consumed by `ranking.rs` |
| `crates/cb-backend/src/gpu_runtime/ranking.rs` | Ranking objective device der driver (5 objectives) | ✓ VERIFIED | 1000 lines; `RankingState` wired in `session.rs` |
| `crates/cb-compute/src/runtime.rs` (`DeviceGrownTree.approx_dim`) | Multi-output block-leaf carrier extension | ✓ VERIFIED | Present at :910-945; scalar byte-unchanged at `approx_dim==1` (confirmed via passing `scalar_k1_matches_prior_scalar_newton` test) |
| `crates/cb-backend/src/kernels/multi_newton.rs` | K-dim Newton der2 block solve (coupled + diagonal) | ✓ VERIFIED | 451 lines; self-oracled vs `solve_symmetric_newton` |
| `crates/cb-backend/src/gpu_runtime/multiclass.rs` | Multi-output device driver + coverage gate | ✓ VERIFIED | 383 lines |
| `crates/cb-backend/src/gpu_runtime/ordered.rs` | Ordered-boosting resident trajectory driver | ✓ VERIFIED | 226 lines |
| `crates/cb-backend/src/kernels/langevin.rs` | AddLangevinNoise seeded-Gaussian kernel | ✓ VERIFIED | 287 lines |
| `.planning/phases/13-.../13-COVERAGE-MATRIX.md` | Per-family correctness + speed + status matrix | ✓ VERIFIED | 181 lines, all five families filled with real Kaggle CUDA numbers |
| `bench/phase13_cuda_oracle/{result.json,result.md,log-excerpt.txt}` | Kaggle CUDA sign-off artifacts | ✓ VERIFIED | Present, internally consistent, real GPU (P100) identified |
| `bench/kaggle_cuda_phase13.ipynb` | Phase-13 Kaggle CUDA notebook | ✓ VERIFIED | Present |

All ten `13-NN-SUMMARY.md` files exist and correspond to real, distinct git commits (`020472c` … `5d80637`), each touching the files declared in the matching `13-NN-PLAN.md` frontmatter (spot-checked 13-01/13-02 via `git show --stat`).

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `session.rs` (`map_pairwise_coverage`) | `pairwise.rs` | pairwise gate dispatch | ✓ WIRED | Code-read at `session.rs:876-889`; both covered/uncovered branches `Ok(None)` by design |
| `pairwise.rs` | `kernels/cholesky_solve.rs::launch_cholesky_solve` | split-score solve replacement | ✓ WIRED | `pairwise.rs:1543`; commit `c221e7c` |
| `gpu_runtime/ranking.rs` | `kernels/query_helper.rs` | group means/bias-removal/masks consumption | ✓ WIRED | `ranking.rs` imports and calls query_helper functions (confirmed via passing `ranking_det_test`/`ranking_stoch_test`) |
| `gpu_runtime/multiclass.rs` | `kernels/multi_newton.rs` | per-leaf K-dim Newton block solve | ✓ WIRED | Confirmed via passing `multiclass_test::coupled_softmax_k3_matches_cpu_multi_output` etc. |
| `cb-compute::runtime.rs (DeviceGrownTree)` | multi-output CPU apply | block leaves route through `approx[d*n+i]` | ✓ WIRED | Confirmed via `runtime_test.rs` (created by Plan 06) |
| `bench/kaggle_cuda_phase13.ipynb` | `13-COVERAGE-MATRIX.md` | notebook results transcribed | ✓ WIRED | Matrix numbers verbatim-match `bench/phase13_cuda_oracle/result.{json,md}` |

### Behavioral Spot-Checks (local CPU-backend self-oracle re-run, not Kaggle re-run)

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Compile health, all 3 crates | `cargo check --tests -p cb-compute` / `-p cb-backend` / `-p cb-train` | Clean (warnings only, no errors) | ✓ PASS |
| Phase-13 family self-oracles run and pass on cpu backend | `cargo test -p cb-backend --lib -- multi_newton multiclass langevin ordered cholesky_solve pairwise_deriv query_helper ranking` | 44 passed; 2 failed (both pre-existing, unrelated to Phase 13 — see Anti-Patterns/Regression section) | ✓ PASS (all Phase-13 tests green) |
| No CPU/host regression across cb-train | `cargo test -p cb-train --no-fail-fast` | 1 failure: `monotone_non_symmetric_and_region_are_typed_errors` (DI-13-01, pre-existing, attributed below); all other tests pass | ✓ PASS (expected single pre-existing failure) |
| No CPU/host regression across cb-compute | `cargo test -p cb-compute --no-fail-fast` | 204/204 pass | ✓ PASS |
| DI-13-01 root-cause confirmed pre-existing (not Phase 13) | `git log -1 --format=%H\ %ad -- crates/cb-train/tests/monotone_oracle_test.rs` vs `region_e2e_test.rs` | `monotone_oracle_test.rs` last touched 2026-06-18 (Phase 06.6-04); `region_e2e_test.rs` (Phase 12-02, CPU Region grower) added 2026-07-04 and passes 2/2 | ✓ CONFIRMED pre-existing, correctly attributed |

### Probe / External Authority Execution (Kaggle CUDA — not re-run, evidence audited per phase_context instruction)

| Probe | Command (as recorded) | Result | Status |
|-------|------------------------|--------|--------|
| Kaggle CUDA `--features cuda` per-family self-oracle suite | `cargo test --release --no-default-features --features cuda` (per-family filters) | exit=0 for all 5 families; 8+14+9+10+3 = 44 tests passed, 0 failed | ✓ AUDITED-CONSISTENT (real Tesla P100 run; not independently re-executable in this environment — no NVIDIA hardware, per project constraint) |
| BENCH-02 grow-loop anchor | `CB_BENCH=1 cargo test --release --features cuda --test bench_grow_speed_test` | 6/6 rows OK, 23.9×–36.6× device≫CPU | ✓ AUDITED-CONSISTENT |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | No `TBD`/`FIXME`/`XXX`/`TODO`/`HACK`/`PLACEHOLDER` markers found in any Phase-13-touched file | — | None |
| — | — | No `.unwrap()` in production code paths of Phase-13-touched files | — | None |
| `crates/cb-train/tests/monotone_oracle_test.rs:286` | 286 | Stale assertion (`grow_policy=Region must be rejected`) now contradicted by Phase 12's shipped CPU Region path | ℹ️ Info (pre-existing, not introduced by Phase 13) | Recorded as DI-13-01 in `deferred-items.md`; recommended fix is Phase-12 hardening, not a Phase-13 gap |
| `crates/cb-backend/src/kernels/grow_loop.rs::partition::update_matches_ordered_reference` and `crates/cb-backend/src/kernels/score_split.rs::scan::cumulative_matches_host_ordered_reference` | — | Pre-existing `n=1` cpu-backend divergence (device-Atomic-on-cpu-runtime edge case) | ℹ️ Info (pre-existing, last touched Phase 7.5/11, well before Phase 13) | Consistent with the "60 pre-existing cb-backend device-only test failures on the cpu backend" documented in `deferred-items.md` (from 13-03) and the "10 pre-existing device-Atomic-on-cpu-runtime failures" noted in commit `c221e7c`'s own verification note; passes clean on the real Kaggle CUDA run (`scan`/`partition_update` rows show `abs_div=0.000e0`) |

### Requirements Coverage

| Requirement | Source Plan(s) | Description | Status | Evidence |
|-------------|-----------------|-------------|--------|----------|
| GPUT-11 | 13-01 | PairLogit pairwise 2×2-cell histogram device path | ✓ SATISFIED | See Truth #1 |
| GPUT-21 | 13-01, 13-02 | Per-leaf matrix assembly + batched device Cholesky solver | ✓ SATISFIED | See Truth #2 |
| GPUT-22 | 13-03, 13-04, 13-05 | Query/listwise ranking objectives + device grouping infra | ✓ SATISFIED | See Truth #3 |
| GPUT-12 | 13-06, 13-07 | Multiclass/multi-target/uncertainty block-leaf device path | ✓ SATISFIED | See Truth #4 |
| GPUT-13 | 13-08 | Ordered boosting device residency | ✓ SATISFIED | See Truth #5 |
| GPUT-20 | 13-09 | Langevin/SGLB seeded-Gaussian noise | ✓ SATISFIED | See Truth #6 |
| BENCH-02 (this phase's slice) | 13-10 | Per-family Kaggle CUDA speed check as it lands | ✓ SATISFIED (scoped `captured-by-grow-loop`) | See Truth #7 |
| GPUT-14 (standing, spans Phases 11-14) | all | ε=1e-4 + byte-unchanged CPU path | ✓ SATISFIED for Phase-13's slice | Not fully closeable until Phase 14 by design — correctly left `[ ]` in REQUIREMENTS.md traceability, not a Phase-13 gap |
| BENCH-03 | Phase 14 (not this phase) | Comprehensive final speed-parity sign-off | N/A — out of scope | Correctly deferred |

No orphaned requirements found: `REQUIREMENTS.md`'s Phase-13 traceability row set (GPUT-11/21/22/12/13/20) matches exactly the `requirements:` fields declared across the ten plans.

### Documentation staleness (non-blocking, informational)

- `ROADMAP.md` still shows "**Plans**: 9/10 plans executed" and the phase-list status column "In Progress" for Phase 13, even though all 10 `13-NN-SUMMARY.md` files exist, all 10 plan checkboxes are `[x]`, and `STATE.md` explicitly says "Phase 13 all 10 plans executed" and "Phase 13 device coverage authoritatively SIGNED OFF." This is a stale bookkeeping field in `ROADMAP.md`, not a functional gap — recommend a one-line ROADMAP.md touch-up (`9/10`→`10/10`, `In Progress`→`Complete`) but it does not block phase-goal achievement.

### Human Verification Required

None. Every must-have was verifiable via code reads, local `cargo check`/`cargo test` execution, and audit of the already-recorded Kaggle CUDA artifacts (which this environment cannot re-run — no NVIDIA hardware — but per the phase's own explicit instruction and the task's `phase_context`, the requirement is to verify the recorded evidence is present and internally consistent, not to re-run CUDA).

### Gaps Summary

No gaps. All six requirements (GPUT-11/21/22/12/13/20) are backed by real, substantive, wired code (not stubs — files range 226–2149 lines, all compile clean, all family-specific self-oracle tests pass locally on the cpu backend), and all six are independently corroborated by a real, internally-consistent Kaggle CUDA P100 run recorded in `bench/phase13_cuda_oracle/`. The phase's own explicit MVP framing — every family declining to `Ok(None)`→CPU pending the shared per-tree grow-seam forward dependency — is verified in the actual `session.rs` code, not just asserted in prose, and this framing is exactly what `phase_context` instructed this verification to accept as in-scope.

The one real code-level anomaly found (`monotone_non_symmetric_and_region_are_typed_errors` failing) is DI-13-01, confirmed via git history to be a pre-existing Phase-12-caused staleness (Phase 12 legitimately shipped CPU Region support, invalidating a Phase-06.6-era assertion) and is explicitly out of scope for Phase 13 — the affected test file was not touched by any Phase-13 commit. Two further pre-existing cpu-backend-only test failures (`n=1` device-Atomic-on-cpu-runtime edge cases in `grow_loop.rs`/`score_split.rs`, both last modified in Phase 7.5/11) are likewise unrelated to Phase 13 and pass cleanly on the real Kaggle CUDA hardware.

---

_Verified: 2026-07-04_
_Verifier: Claude (gsd-verifier)_
