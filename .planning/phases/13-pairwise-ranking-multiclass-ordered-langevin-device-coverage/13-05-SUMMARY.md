---
phase: 13-pairwise-ranking-multiclass-ordered-langevin-device-coverage
plan: 05
subsystem: ranking
tags: [cubecl, gpu, ranking, yetirank, pfound-f, stochastic-sampling, pinned-seed, gput-22]

# Dependency graph
requires:
  - phase: 13-04
    provides: deterministic query-objective device der driver (gpu_runtime/ranking.rs) + RankingObjective enum + session ranking coverage gate
  - phase: 13-03
    provides: shared device query-grouping infra (query_helper)
  - phase: 12-07
    provides: mvs_device inline-PCG #[cube] transcription of TFastRng64 from_seed + gen_rand_real1 + wgpu-reject
  - phase: 12-05
    provides: segmented_radix_sort (exact_quantile.rs) — the stable segmented LSD radix primitive
  - phase: 06.3
    provides: cb_train::yetirank::sample_pairs + cb_compute::calc_ders_for_queries (YetiRank arm) CPU der reference
provides:
  - Stochastic YetiRank + PFound-F device der (gpu_runtime/ranking.rs) — in-query bootstrap sampling under pinned seed
  - yetirank_perturb_kernel (#[cube]) — inline-PCG per-query RNG re-expansion + f32 Gumbel perturbation in exact CPU draw order
  - Host segmented_radix_sort descending per-query sort + Classic decayed f32 competitor weights + pairlogit scatter der
  - RankingObjective::{YetiRank, PFoundF} covered arms + session gate routing (is_stochastic_ranking_loss / is_ranking_loss)
  - Frozen pinned-seed self-oracle (ranking_stoch_test.rs) — device der == CPU reference bit-exact on rocm gfx1100
affects: [13-06, ranking, gput-22-coverage]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "In-query bootstrap sampling: inline-PCG #[cube] kernel re-expands TFastRng64::from_seed(query_seed) and draws gen_rand_real1 per doc in EXACT CPU order (perm-major, doc-ascending), f32-casts to perturb exp(approx) by uni/(1.000001-uni) — the f32 round is load-bearing (Pitfall 4)"
    - "Host derives the per-query O(1) base RNG state (single-block derive_query_seeds transcription over cb_core::TFastRng64) — no per-iteration host RNG readback (bootstrap_device precedent)"
    - "Full-precision descending device sort of NON-NEGATIVE f64 perturbations via a 2-pass 64-bit stable LSD radix (segmented_radix_sort on lo32 then hi32 of the monotone bit pattern) + per-query reverse"
    - "f32 competitor-weight accumulation + pairlogit scatter der transcribed inline on host — reproduces cb_train sample_pairs + cb_compute pairlogit_group_der bit-for-bit (no cb-train dep)"
    - "PFound-F == YetiRankPairwise shares the SAME sampled-pair der stream as YetiRank (only the leaf path differs, decided later in boosting) — one device der core, two independently-gated arms"

key-files:
  created:
    - crates/cb-backend/src/gpu_runtime/ranking_stoch_test.rs
  modified:
    - crates/cb-backend/src/gpu_runtime/ranking.rs
    - crates/cb-backend/src/gpu_runtime/session.rs
    - crates/cb-backend/src/gpu_runtime/mod.rs

key-decisions:
  - "Followed the ACTUAL CPU algorithm over the plan's loose der-averaging phrasing: the parity contract is f32 competitor-WEIGHT accumulation across permutations then ONE pairlogit der (NOT per-iteration der averaging) — the f32 weight bit-width is load-bearing (yetirank.rs comments), so averaging der would diverge. Documented as Rule-1 fidelity fix."
  - "segmented_radix_sort reused as a 2-pass 64-bit STABLE LSD sort (lo32 then hi32 of the non-negative-f64 bit pattern) for full-precision descending order, rather than a lossy single f32-key pass — avoids sort-order flips vs the CPU f64 sort (the correctness-safe choice for the ≤1e-4 gate)."
  - "PFound-F maps to Loss::YetiRankPairwise (the GPU pfound_f pairwise-leaf arm); since YetiRankPairwise is ALSO is_pairwise_scoring it is intercepted by the session pairwise gate (also Ok(None)), but its device der driver + self-oracle still land + are exercised directly by ranking_stoch_test — the session outcome (Ok(None), grow seam a forward dependency) is identical either way."
  - "Covered ranking fit continues to decline to CPU (Ok(None)) — the der driver + self-oracle are this plan's deliverable; the per-tree query-descriptor grow seam is a forward dependency (the Plan-01 pairwise / Plan-04 deterministic-ranking precedent). RankingState is the landed structural coverage seam."

patterns-established:
  - "device in-query bootstrap sampling (inline-PCG + segmented radix sort + f32 competitor weights) — the stochastic-ranking substrate; the pinned-seed frozen-fixture discipline reused for Langevin (Plan 06+)"

requirements-completed: [GPUT-22]

# Metrics
duration: ~70min
completed: 2026-07-04
status: complete
---

# Phase 13 Plan 05: Stochastic Ranking Pair (YetiRank + PFound-F) Device Der Summary

**Completed GPUT-22 device coverage with the two STOCHASTIC listwise objectives — YetiRank (pointwise leaf) and PFound-F (`YetiRankPairwise`) — via an inline-PCG `#[cube]` in-query bootstrap-perturbation kernel that reproduces `TFastRng64::from_seed(query_seed)` + `gen_rand_real1` per doc in the EXACT CPU draw order (perm-major, doc-ascending), f32-casts to perturb `exp(approx)` by `uni/(1.000001−uni)` (the load-bearing f32 round, Pitfall 4), then a host `segmented_radix_sort` descending per-query sort + f32 Classic decayed competitor weights + pairlogit scatter der — reproducing the CPU `yetirank_sample_pairs` + `calc_ders_for_queries` reference BIT-EXACT (max_div = 0.000e0) on real rocm gfx1100 at the ε=1e-4 bar.**

## Performance

- **Duration:** ~70 min
- **Completed:** 2026-07-04
- **Tasks:** 2
- **Files modified:** 4 (1 created, 3 modified)

## Accomplishments

- **`ranking.rs` — the stochastic device der:**
  - `yetirank_perturb_kernel` (`#[cube]`, serial unit-0): per query re-expands `TFastRng64::from_seed(seeds[g])` INLINE (transcribed PCG XSH-RR, mirroring `kernels::mvs_device`), then per permutation, per doc (the CPU `for perm { for doc }` order) draws `gen_rand_real1`, casts to `f32`, and writes `perturbed[p·n + d] = exp(approx[d]) · f32(u/(1.000001−u))`. The draw stream is continuous across permutations within a query (no per-doc RNG jump needed).
  - `derive_query_seeds_inline` — the single-block `derive_query_seeds` transcription over the sanctioned `cb_core::TFastRng64` (the O(1) per-query base state; no `cb-train` dep).
  - `descending_order_per_query` — reuses `segmented_radix_sort` as a 2-pass 64-bit STABLE LSD radix (lo32 then hi32 of the monotone non-negative-f64 bit pattern) → full-precision ascending order, reversed per query for the CPU descending sort.
  - `yetirank_sample_der_core` — accumulates the f32 Classic decayed competitor weights (`0.15 · decay^k · |Δrelev|`, `AddWeight` to the higher-relevance winner) across permutations, normalizes `queryWeight · cw / permutations` (f32), then the transcribed pairlogit scatter der (`p = exp(loser)/(exp(loser)+exp(winner))`, winner raises own der / lowers each loser's).
  - `yetirank_ders_host` / `pfound_f_ders_host` — the two public arms (shared core); `RankingObjective::{YetiRank, PFoundF}` added, both `ranking_objective_covered == true`; `yetirank_draw_count` exposes the `permutations · n` draw count.
- **`session.rs`:** `is_stochastic_ranking_loss` / `is_ranking_loss`; `map_ranking_coverage` maps `Loss::YetiRank → YetiRank` and `Loss::YetiRankPairwise → PFoundF`; the `begin` ranking branch widened to `is_ranking_loss` (both still decline to CPU — the grow seam is a forward dependency).
- **`ranking_stoch_test.rs` (NEW):** the frozen pinned-seed self-oracle — device YetiRank / PFound-F der vs the FROZEN CPU reference (`yetirank_sample_pairs` + `calc_ders_for_queries`, generated offline; NON-tautological) at ε=1e-4; plus the per-query seed-chain assert and the draw-COUNT assert (24 = 4·6; a divergent count is DETECTED, not absorbed — T-13-10).

## Real-device validation (rocm gfx1100, in-env)

The ε=1e-4 numeric assertions fire only on rocm/cuda (`device_backend_active`); on real gfx1100 all 5 stochastic tests + the 3 deterministic tests pass (8/8 `ranking`):
- **YetiRank** der1 / der2: `max_div = 0.000e0` (BIT-EXACT vs the frozen CPU reference).
- **PFound-F** der1 / der2: `max_div = 0.000e0` (BIT-EXACT — same sampled-pair stream).
- Per-query seed chain + draw count (24) match the frozen CPU chain exactly.

The full integer RNG stream (`gen_rand`), the f32 Gumbel cast, the descending sort, the f32 competitor accumulation, and the pairlogit der all reproduce the CPU host reference to the last bit. (The `cpu` default run records-only, WR-01; Kaggle CUDA sign-off is deferred to Plan 10.)

## Task Commits

1. **Task 1: YetiRank + PFound-F device der kernel + host driver + session gate** — `e798d30` (feat)
2. **Task 2: frozen pinned-seed self-oracle** — `1a674fb` (test)

## Files Created/Modified

- `crates/cb-backend/src/gpu_runtime/ranking.rs` — the `yetirank_perturb_kernel` (`#[cube]`) + inline PCG helpers + `yetirank_sample_der_core` + `yetirank_ders_host` / `pfound_f_ders_host` + `derive_query_seeds_inline` + `descending_order_per_query` (segmented_radix_sort reuse); `RankingObjective::{YetiRank, PFoundF}` + covered predicate.
- `crates/cb-backend/src/gpu_runtime/session.rs` — `is_stochastic_ranking_loss` / `is_ranking_loss`; `map_ranking_coverage` stochastic arms; `begin` ranking branch widened.
- `crates/cb-backend/src/gpu_runtime/ranking_stoch_test.rs` — the frozen-fixture self-oracle (5 tests), numeric ε device-gated (WR-01).
- `crates/cb-backend/src/gpu_runtime/mod.rs` — registered `#[cfg(test)] mod ranking_stoch_test`.

## Decisions Made

- **Followed the CPU algorithm over the plan's loose der-averaging phrasing (Rule 1 fidelity).** The plan's action text says "accumulate the sampled der ... and average over iterations," but the CPU parity contract is f32 competitor-WEIGHT accumulation across permutations then ONE pairlogit der. The f32 weight bit-width is load-bearing (`yetirank.rs`: an f64 accumulation drifts ~1e-8 and flips a close split), so per-iteration der averaging would diverge. The device follows the exact CPU weight-then-der order.
- **`segmented_radix_sort` reused as a full-precision 2-pass 64-bit stable sort**, not a lossy single f32-key pass — the perturbed values are all non-negative (so their f64 bit patterns are monotone), and a lo32-then-hi32 stable LSD reproduces the CPU f64 descending order exactly (no near-tie sort flips at the ≤1e-4 gate).
- **PFound-F == `YetiRankPairwise`** shares the sampled-pair der with YetiRank; since it is also `is_pairwise_scoring` the session pairwise gate intercepts it first (also `Ok(None)`), but its device der driver + self-oracle still land and are exercised directly. The session outcome is identical.
- **Covered ranking fit declines to CPU (`Ok(None)`)** — the der driver + self-oracle are the deliverable; the per-tree query-descriptor grow seam is a forward dependency (the Plan-01 / Plan-04 precedent).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Fidelity] Weight-then-der accumulation (not per-iteration der averaging)**
- **Found during:** Task 1
- **Issue:** The plan's action phrasing ("accumulate the sampled der ... average over iterations") does not match the CPU parity contract, which accumulates f32 competitor WEIGHTS across permutations then computes ONE pairlogit der. Averaging der per iteration would break the load-bearing f32 weight bit-width and diverge from the frozen reference.
- **Fix:** Implemented the exact CPU order — f32 competitor-weight accumulation across permutations, normalize, then the single pairlogit scatter der.
- **Files modified:** crates/cb-backend/src/gpu_runtime/ranking.rs
- **Verification:** rocm gfx1100 `max_div = 0.000e0` (bit-exact) vs the frozen CPU reference.
- **Committed in:** `e798d30` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 fidelity fix aligning to the actual CPU algorithm).
**Impact on plan:** No scope change — all artifacts delivered; the fix was required for bit-exact parity.

## Deferred Issues

- **Kaggle CUDA ε=1e-4 sign-off** for the stochastic ranking der is deferred to Plan 10 (per `success_criteria`); the numeric ε assertions are device-gated and validated in-env on rocm gfx1100 by the executor (cpu records-only, WR-01).
- **Per-tree query-descriptor grow seam** (the covered-ranking device grow) remains a forward dependency (the Plan-01 pairwise / Plan-04 deterministic-ranking deferral) — a covered ranking fit declines to CPU.
- **Pre-existing `cargo clippy -p cb-backend` errors** in `exact_quantile.rs:178`, `bootstrap_device.rs:230`, `cpu_runtime.rs:696/1025` (indexing/slicing/LN_2 approximate value) predate this plan (Phase 12 commit `34ac0da`) and are OUT OF SCOPE (SCOPE BOUNDARY). `cargo build`/`cargo test` do not enforce clippy-only lints, so they do not block this plan; the new `ranking.rs` / `ranking_stoch_test.rs` code is clippy-clean (bounds-checked `.get()` host accessors throughout).

## Next Phase Readiness

- GPUT-22 device coverage is COMPLETE (all 5 query objectives: QueryRMSE, QuerySoftMax, QueryCrossEntropy [gated], YetiRank, PFound-F). The deterministic-only subset (Plans 03–04) remains the documented lower-risk fallback, but was NOT needed — the stochastic pair reproduces the frozen CPU reference bit-exact.
- The device in-query bootstrap-sampling substrate (inline-PCG + segmented radix sort + f32 competitor weights) is ready for Plan 06+ (Langevin noise reuses the pinned-seed frozen-fixture discipline).

---
*Phase: 13-pairwise-ranking-multiclass-ordered-langevin-device-coverage*
*Completed: 2026-07-04*

## Self-Check: PASSED
- FOUND: crates/cb-backend/src/gpu_runtime/ranking.rs
- FOUND: crates/cb-backend/src/gpu_runtime/ranking_stoch_test.rs
- FOUND: .planning/phases/13-.../13-05-SUMMARY.md
- FOUND commit e798d30 (Task 1), 1a674fb (Task 2)
- rocm gfx1100 in-env: 8/8 ranking tests pass; YetiRank/PFound-F der max_div = 0.000e0 (bit-exact)
