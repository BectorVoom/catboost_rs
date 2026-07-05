---
phase: 15-debt-discharge-cuda-oracle-re-establishment
plan: 03
subsystem: gpu
tags: [cuda, kaggle, p100, bench-02, single-session, parity-oracle, rv-13, speedup]

# Dependency graph
requires:
  - phase: 15-01
    provides: RV-13-01/02 ranking-der direct-invocation oracles (tie_order_matches_cpu_stable_descending, softmax_weight_max_seed)
  - phase: 15-02
    provides: RV-13-03/04 latent-hazard oracles (empty_group_means_no_fault, pairwise_near_equal_border_tiebreak)
provides:
  - One authoritative single-session Tesla-P100 CUDA record (bench/phase15_cuda_oracle/result.json) discharging HARD-01 (Part A ALL-PASS ε=1e-4) and HARD-02 (Part B depth-1/depth-6 BENCH-02 rows)
  - bench/phase15_cuda_oracle/oracle.py — correctness-blocks-speed single-session runner (Part A gate → Part B timing → Part C informational catboost-GPU arm)
  - BENCH_DEPTH env lever in crates/cb-train/tests/bench_grow_speed_test.rs (default 6) so depth-1 and depth-6 rows run in one kernel session
affects: [15-04, 15-EVIDENCE, bench03-recompute, requirements-flip, milestone-v1.2]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Correctness-blocks-speed single-session harness: Part A per-family cargo-test gate → sys.exit(2) BEFORE any Part B timing (D-04/D-05)"
    - "RV-13 oracles ride the SAME --features cuda cargo-test invocations as their family (ranking family carries tie-order/softmax/empty-group; pairwise family carries near-equal-tiebreak)"
    - "One result.json with single-session provenance (one gpu/driver/seed), superseding aggregate.py multi-session stitching — keep only the >=20x verdict shape"

key-files:
  created:
    - bench/phase15_cuda_oracle/oracle.py
    - bench/phase15_cuda_oracle/kernel-metadata.json
    - bench/phase15_cuda_oracle/result.json
  modified:
    - crates/cb-train/tests/bench_grow_speed_test.rs

key-decisions:
  - "Depth lever is an env var (BENCH_DEPTH, default 6) not a code edit — depth-6 provenance byte stays unchanged; depth-1 selected by setting BENCH_DEPTH=1 per Part B row (D-07)"
  - "Depth-1 device>=CPU is NOT gated — success is 'row executed + crossover recorded' (A4/Pitfall 5); it happened to win at every n but that is not the pass condition"
  - "Region catboost_gpu_s stays N/A (no official CatBoost Region grow_policy) — not proxied (do-not-fabricate)"
  - "No speed number recorded unless Part A ALL-PASS — Part B ran only because the correctness pre-gate passed (D-05)"

patterns-established:
  - "Single authoritative GPU record: every later parity/benchmark claim (Phases 19/21) rests on this one file, not on stitched per-family sessions"

requirements-completed: [HARD-01, HARD-02]

# Metrics
duration: ~30min
completed: 2026-07-05
status: complete
---

# Phase 15 Plan 03: Single Authoritative Kaggle CUDA Oracle Session Summary

**One Tesla P100 CUDA kernel session that gates 13 v1.1 device families + the 4 RV-13 oracles at ε=1e-4 (ALL-PASS, bit-exact) BEFORE emitting 12 BENCH-02 depth-1/depth-6 grow rows at 29.1×–40.8× device-over-CPU — the trusted record every later parity/benchmark claim rests on.**

## Performance

- **Duration:** ~30 min (orchestrator-driven Kaggle session ~15 min build+run + finalization)
- **Started:** 2026-07-05
- **Completed:** 2026-07-05
- **Tasks:** 2/2 (Task 1 runner authored; Task 2 orchestrator-driven Kaggle CUDA session)
- **Files modified:** 4 (3 created under bench/phase15_cuda_oracle/, 1 modified in cb-train)

## Accomplishments

- **HARD-01 discharged (Part A, ALL-PASS ε=1e-4):** All 13 v1.1 device families ran with `exit==0` and `ran_any_tests==true` in ONE `--features cuda` session on Tesla P100. Named divergences are bit-exact (max `abs_div=0.000e0`; nonsym leaf-values, exact-quantile all `0.000e0`; the only nonzero deltas are the inherently-stochastic bootstrap `2.384e-7` and mvs `~1e-15`, all far under the 1e-4 bar).
- **All 4 RV-13 oracles seen == expected in-session:** `tie_order_matches_cpu_stable_descending`, `softmax_weight_max_seed`, `empty_group_means_no_fault` (ranking family) and `pairwise_near_equal_border_tiebreak` (pairwise family) all rode their family's cuda cargo-test invocation and were counted — `rv13_oracles_expected == rv13_oracles_seen` (4/4).
- **HARD-02 discharged (Part B, 12 rows):** depth-1 (n=100k/300k/1M) and depth-6 (n=10k/100k/300k) × {depthwise, region} grow rows, warm-run / JIT-excluded / lazy-CubeCL-queue-drained / median-of-3. Device beats host CPU on every row: **29.1× (region depth-6 n=10k) up to 40.8× (region depth-1 n=100k)**; `bench_verdict: OK`, `depth6_ge20x: true`, GE20X gate = 20.0.
- **Crossover recorded:** depth-1 depthwise device first beats CPU at **n=100000** (the smallest n tested) — recorded, not gated (A4).
- **Single-session provenance confirmed:** one GPU (Tesla P100-PCIE-16GB), driver 580.159.04, CUDA release 12.8, seed 42, `single_session: true` — no aggregate.py-style multi-session stitching.

## Task Commits

1. **Task 1: single-session runner + BENCH_DEPTH lever** - `5d07c67` (feat) — `oracle.py` (469 lines), `kernel-metadata.json`, and the `BENCH_DEPTH` env lever in `bench_grow_speed_test.rs`.
2. **Task 2: orchestrator-driven Kaggle P100 CUDA session** - `734109a` (feat) — `result.json` (613 lines) committed verbatim from the run.

**Plan metadata:** this SUMMARY + STATE/ROADMAP bookkeeping (docs commit).

## Files Created/Modified

- `bench/phase15_cuda_oracle/oracle.py` — single-session runner: reuses the frozen `gen()` and /tmp `CARGO_TARGET_DIR` staging; Part A per-family FAMILIES gate (`cargo test --release --no-default-features --features cuda`) with the 4 RV-13 test names folded into the ranking/pairwise filter sets → `sys.exit(2)` on any fail before timing; Part B depth-1/depth-6 device+CPU timing; Part C informational catboost-GPU arm; emits ONE `result.json`.
- `bench/phase15_cuda_oracle/kernel-metadata.json` — phase15 Kaggle kernel identity, `enable_gpu: true`, phase15 dataset source.
- `bench/phase15_cuda_oracle/result.json` — the authoritative record (committed verbatim): `correctness_verdict: ALL-PASS`, per-family + RV-13 divergences, `bench02.depth_rows` (12 rows) + `crossover` + provenance.
- `crates/cb-train/tests/bench_grow_speed_test.rs` — added `BENCH_DEPTH` env lever (default 6) so depth-1 rows run by setting `BENCH_DEPTH=1` in the same kernel (see Deviations).

## Part A — Correctness Verdict (ALL-PASS, ε=1e-4)

| Family | Req | Crate | ran_any | RV-13 seen |
|--------|-----|-------|---------|------------|
| nonsym_grow (Depthwise/Lossguide) | GPUT-18 | cb-backend | ✓ | — |
| region_device | GPUT-18 | cb-backend | ✓ | — |
| exact_quantile + segmented_sort | GPUT-19 | cb-backend | ✓ | — |
| bootstrap_device | GPUT-09 | cb-backend | ✓ | — |
| mvs_device | GPUT-17 | cb-backend | ✓ | — |
| ctr_device | GPUT-10 | cb-backend | ✓ | — |
| device_nonsym_fit (e2e) | GPUT-18 | cb-train | ✓ | — |
| device_region_fit (e2e) | GPUT-18 | cb-train | ✓ | — |
| pairwise (deriv + batched Cholesky) | GPUT-11/21 | cb-backend | ✓ | pairwise_near_equal_border_tiebreak |
| ranking (query grouping + det + stochastic) | GPUT-22 | cb-backend | ✓ | tie_order_matches_cpu_stable_descending, softmax_weight_max_seed, empty_group_means_no_fault |
| multiclass (softmax der + multi-Newton) | GPUT-12 | cb-backend | ✓ | — |
| ordered (resident approx trajectory) | GPUT-13 | cb-backend | ✓ | — |
| langevin (seeded Gaussian / SGLB) | GPUT-20 | cb-backend | ✓ | — |

All 13 `exit==0`; `correctness_verdict: ALL-PASS`; `rv13_oracles_expected == rv13_oracles_seen` (4/4).

## Part B — BENCH-02 Speed Rows (device vs host CPU, median-of-3, 20 iters / 20 feat / 32 bins)

| depth | family | n | device_s | host_cpu_s | catboost_gpu_s | speedup | device≥CPU |
|------:|--------|---:|---------:|-----------:|---------------:|--------:|:---------:|
| 1 | depthwise | 100000 | 0.4813 | 14.9418 | 0.6857 | 31.05× | ✓ |
| 1 | depthwise | 300000 | 1.4986 | 49.6221 | 0.7858 | 33.11× | ✓ |
| 1 | depthwise | 1000000 | 6.0235 | 196.0425 | 0.9400 | 32.55× | ✓ |
| 1 | region | 100000 | 0.7273 | 29.5788 | N/A | 40.67× | ✓ |
| 1 | region | 300000 | 2.3273 | 94.8537 | N/A | 40.76× | ✓ |
| 1 | region | 1000000 | 9.7365 | 381.3196 | N/A | 39.16× | ✓ |
| 6 | depthwise | 10000 | 0.0921 | 2.8278 | 0.6864 | 30.70× | ✓ |
| 6 | depthwise | 100000 | 0.8499 | 31.4309 | 0.7167 | 36.98× | ✓ |
| 6 | depthwise | 300000 | 2.6964 | 108.6983 | 0.8325 | 40.31× | ✓ |
| 6 | region | 10000 | 0.1117 | 3.2557 | N/A | 29.15× | ✓ |
| 6 | region | 100000 | 0.9214 | 37.2071 | N/A | 40.38× | ✓ |
| 6 | region | 300000 | 2.9998 | 118.4237 | N/A | 39.48× | ✓ |

**Crossover:** depth-1 depthwise device first beats CPU at **n=100000** (`crossover.note: "device first beats CPU at n=100000"`). Region rows have `catboost_gpu_s = N/A` (no upstream Region grow_policy). `bench_verdict: OK`, `catboost_gpu_verdict: OK`, `depth6_ge20x: true`.

## Provenance (single session)

- **GPU:** Tesla P100-PCIE-16GB, 16384 MiB
- **Driver:** 580.159.04 · **CUDA:** release 12.8 · **Seed:** 42 · **single_session:** true
- **Date:** 2026-07-05

## Decisions Made

- **Depth lever is env-driven (`BENCH_DEPTH`, default 6), not a source edit per row** — keeps the depth-6 provenance byte-unchanged and lets one kernel run both depth rows (D-07).
- **Depth-1 device≥CPU not gated** — success is "row executed + crossover recorded" (A4/Pitfall 5). It won at every n, but that was not the pass condition.
- **Region catboost_gpu_s stays N/A** — no official CatBoost Region grow_policy; not proxied.
- **No speed number without Part A pass** — Part B ran only because the correctness pre-gate was ALL-PASS (D-05).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added `BENCH_DEPTH` env lever to bench_grow_speed_test.rs**
- **Found during:** Task 1 (authoring the single-session runner)
- **Issue:** The existing BENCH-02 bench test hard-coded grow depth at 6; the plan requires BOTH depth-1 and depth-6 rows in ONE kernel session (HARD-02), which the runner cannot select without a lever.
- **Fix:** Added `let depth: usize = std::env::var("BENCH_DEPTH")...` (default 6) so `oracle.py` sets `BENCH_DEPTH=1` for the depth-1 Part B rows and leaves it unset (=6) for depth-6.
- **Files modified:** `crates/cb-train/tests/bench_grow_speed_test.rs`
- **Verification:** default 6 preserves prior depth-6 provenance byte-for-byte; both depth rows emitted in the committed result.json.
- **Committed in:** `5d07c67` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking). **No numbers were fabricated** — every Part A verdict and Part B cell comes from the committed `result.json` produced by the actual P100 run. **Impact:** minimal, env-only lever; no scope creep.

## Issues Encountered

None. Part A passed on the first authoritative session, so Part B ran and the phase advances to Wave C without a loop-back.

## User Setup Required

None for downstream — the Kaggle CUDA session (auth via `~/.kaggle/access_token`) was orchestrator-driven in-env (the verifier subagent cannot run GPU). The authoritative artifact is now committed at `bench/phase15_cuda_oracle/result.json`.

## Next Phase Readiness

- **Wave C (15-04) input ready:** `bench/phase15_cuda_oracle/result.json` is the single trusted record for 15-EVIDENCE.md assembly + BENCH-03 in-place recompute + REQUIREMENTS/MILESTONES/STATE bookkeeping flip (HARD-01/02/03).
- No blockers. HARD-01 and HARD-02 are satisfied by this one session.

## Self-Check: PASSED

- Files verified present: `bench/phase15_cuda_oracle/oracle.py`, `bench/phase15_cuda_oracle/kernel-metadata.json`, `bench/phase15_cuda_oracle/result.json`, `crates/cb-train/tests/bench_grow_speed_test.rs`.
- Commits verified in git log: `5d07c67` (Task 1), `734109a` (Task 2).
- result.json confirms `correctness_verdict: ALL-PASS`, 4/4 RV-13 oracles seen, 12 Part B rows, single-session provenance.

---
*Phase: 15-debt-discharge-cuda-oracle-re-establishment*
*Completed: 2026-07-05*
