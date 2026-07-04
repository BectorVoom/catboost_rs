---
phase: 13-pairwise-ranking-multiclass-ordered-langevin-device-coverage
plan: 04
subsystem: ranking
tags: [cubecl, gpu, ranking, query-listwise, queryrmse, querysoftmax, querycrossentropy, gput-22]

# Dependency graph
requires:
  - phase: 13-03
    provides: shared device query-grouping infra (query_helper) — group means/max/ids, RemoveGroupMeans, fixed-point group reductions
  - phase: 12-07
    provides: mvs_device serial #[cube] bisection skeleton + inline PCG + wgpu-reject pattern
  - phase: 06.3
    provides: cb_compute::ranking_der::calc_ders_for_queries CPU der oracle (QueryRMSE/QuerySoftMax)
provides:
  - Deterministic query-objective device der driver (gpu_runtime/ranking.rs) over the Plan-03 grouping infra
  - QueryRMSE device der (RemoveGroupMeans residual + pointwise der1=(residual-queryAvrg)*w, der2=-w)
  - QuerySoftMax device der (ComputeGroupMax shift + weighted exp-share p)
  - QueryCrossEntropy bounded per-query bisection(8)+Newton(5) shift search, INDEPENDENTLY gated off (Open Q3)
  - GpuTrainSession ranking coverage gate (Option<RankingState>) + per-objective Ok(None) arms
  - Self-oracle vs CPU calc_ders_for_queries at ε=1e-4 (rocm gfx1100 in-env)
affects: [13-05, ranking, query-listwise-objectives]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Ranking der driver composes the Plan-03 query_helper device kernels (group means/ids/max, RemoveGroupMeans) with a per-objective pointwise/serial der #[cube] kernel"
    - "Serial per-query softmax der kernel mirrors the CPU calc_ders_for_queries ascending-doc order exactly (deterministic single-thread accumulation, no atomic)"
    - "Bounded per-query bisection+Newton shift search over a RUNTIME-passed FINITE bracket (comptime-const seed clashes with the runtime bisection var — NativeExpand<f64>)"
    - "Independent per-objective Ok(None) gate: QueryCrossEntropy deferred (Open Q3, no CPU oracle) without disabling the covered QueryRMSE/QuerySoftMax arms"

key-files:
  created:
    - crates/cb-backend/src/gpu_runtime/ranking.rs
    - crates/cb-backend/src/gpu_runtime/ranking_det_test.rs
  modified:
    - crates/cb-backend/src/gpu_runtime/session.rs
    - crates/cb-backend/src/gpu_runtime/mod.rs

key-decisions:
  - "Covered ranking fit declines to the byte-unchanged CPU grower (Ok(None)) — mirrors the Plan-01 pairwise precedent: the der driver + self-oracle land here, the per-tree query-descriptor grow seam (grow_tree_on_device carries only approx/target) is a forward dependency, so no covered-ranking session is constructed yet"
  - "QueryCrossEntropy INDEPENDENTLY deferred (Open Q3): it has no cb_compute::ranking_der CPU der oracle / Loss variant, so its bounded shift search is landed structurally (DoS-bounded root-find) and gated off (ranking_objective_covered==false) rather than shipping unverified der parity"
  - "The FINITE shift bracket is passed as RUNTIME kernel data, not the comptime const directly: seeding lo/hi from a comptime f64 makes them comptime-typed, clashing with the runtime bisection var (a NativeExpand<f64> #[cube] type error)"
  - "Covered device ranking regime is UNIFORM object weights (weights empty → 1.0 column): under uniform weights the Plan-03 ComputeGroupMax (max over all) equals the CPU max-over-(weight>0) seed, and the group-means fixed-point reduction matches sum_f64 within ε=1e-4"

patterns-established:
  - "query-objective device der driver over the shared query_helper substrate — reused by Plan 05 (YetiRank/PFound-F) for the stochastic ranking arms"

requirements-completed: [GPUT-22]

# Metrics
duration: ~45min
completed: 2026-07-04
status: complete
---

# Phase 13 Plan 04: Deterministic Query/Listwise Objective Device Driver Summary

**Deterministic query/listwise ranking der on device (QueryRMSE / QuerySoftMax / QueryCrossEntropy) — transcribing `cb_compute::ranking_der::calc_ders_for_queries` onto the device path over the Plan-03 shared query-grouping infra (group means/max, per-query bias removal, fixed-point reductions), self-oracled at ε=1e-4 on real rocm gfx1100; QueryCrossEntropy's bounded per-query bisection+Newton shift search landed behind its OWN independent `Ok(None)` gate (Open Q3) so QueryRMSE/QuerySoftMax ship regardless.**

## Performance

- **Duration:** ~45 min
- **Completed:** 2026-07-04
- **Tasks:** 3
- **Files modified:** 4 (2 created, 2 modified)

## Accomplishments
- New `gpu_runtime/ranking.rs` — the deterministic query-objective device der driver:
  - **QueryRMSE**: composes the Plan-03 `ComputeGroupMeans` (the `queryAvrg` numerator/denominator fixed-point reduction), `ComputeGroupIds`, and `RemoveGroupMeans` (`residual − queryAvrg`) with a doc-parallel `ranking_rmse_der_kernel` (`der1 = centered·w`, `der2 = −w`).
  - **QuerySoftMax**: a serial per-query `query_softmax_der_kernel` using the Plan-03 `ComputeGroupMax` shift + the weighted exp-share `p` (`error_functions.cpp:540-576`), with the `sumWTargets ≤ 0` / `weight ≤ 0` guards.
  - **QueryCrossEntropy**: a serial per-query `query_cross_entropy_shift_kernel` — a bounded bisection (8) + Newton (5) shift search over a FINITE runtime bracket solving `Σ w·sigmoid(approx + b) = Σ w·target` (T-13-07 DoS mitigation, no unbounded loop).
- `session.rs`: the ranking coverage gate — `is_deterministic_ranking_loss`, `map_ranking_coverage`, an `Option<RankingState>` field, and the `begin(...)` interception (both covered & uncovered ranking configs decline to CPU, the pairwise-gate precedent). QueryCrossEntropy is independently gated off via `ranking_objective_covered`.
- New `ranking_det_test.rs` self-oracle vs the INDEPENDENT `cb_compute::calc_ders_for_queries` over a 3-query uneven fixture — QueryRMSE + QuerySoftMax der ≤1e-4; QueryCrossEntropy gated-off flag + bounded-shift self-consistency (`F(shift) ≈ Σ w·t`).
- No `cb-train` dependency added (feature-unification landmine avoided); der constants transcribed inline; f64/u64 wgpu reject at every entry point; no `-inf` literal in any `#[cube]` body.

## Real-device validation (rocm gfx1100, in-env)
The ε=1e-4 numeric assertions fire only on rocm/cuda (`device_backend_active`); on real gfx1100 all 3 tests pass:
- **QueryRMSE**: der1 max_div = `1.40e-10` (the k=30 fixed-point `queryAvrg` residual), der2 = `0`.
- **QuerySoftMax**: der1 max_div = `5.55e-17`, der2 max_div = `2.22e-16` (bit-exact, machine epsilon).
- **QueryCrossEntropy**: shift self-consistency worst `|F − T| = 0` (exact root convergence).

All within the ε=1e-4 GPU bar. (The `cpu` default run records-only, WR-01; Kaggle CUDA sign-off is deferred to Plan 10 per the plan's `success_criteria`.)

## Task Commits

1. **Task 1 + Task 2: device ranking der driver (QueryRMSE/QuerySoftMax/QueryCrossEntropy) + coverage gate** - `641e0f0` (feat)
2. **Task 3: self-oracle vs cb_compute::ranking_der** - `e575144` (test)

_Note: Task 1 (QueryRMSE/QuerySoftMax + gate) and Task 2 (QueryCrossEntropy independent gate) are landed in ONE atomic commit — the three objectives live in one cohesive `ranking.rs` and QueryCrossEntropy's independent `Ok(None)` gate is intrinsic to the same `map_ranking_coverage` / `ranking_objective_covered` code; splitting a single file mid-function is artificial. `tdd_mode` is disabled project-wide, so each logical task lands as a single atomic commit (the Plan-03 precedent)._

## Files Created/Modified
- `crates/cb-backend/src/gpu_runtime/ranking.rs` — the `#[cube]` ranking der kernels (rmse pointwise, softmax serial, cross-entropy bounded shift search) + host wrappers consuming the Plan-03 `query_helper` group outputs. `RankingObjective` enum + `ranking_objective_covered` predicate.
- `crates/cb-backend/src/gpu_runtime/ranking_det_test.rs` — the self-oracle (3 tests) vs `cb_compute::calc_ders_for_queries`, numeric ε device-gated (WR-01).
- `crates/cb-backend/src/gpu_runtime/session.rs` — the ranking coverage gate (`Option<RankingState>` + `map_ranking_coverage` + `begin` interception).
- `crates/cb-backend/src/gpu_runtime/mod.rs` — registered `pub(crate) mod ranking` + `#[cfg(test)] mod ranking_det_test`.

## Decisions Made
- **Covered ranking declines to CPU (Plan-01 pairwise precedent).** The der driver + self-oracle are this plan's deliverable; the per-tree query-descriptor grow seam (`Runtime::grow_tree_on_device` carries only `approx`/`target` today) is a forward dependency, so a covered ranking fit returns `Ok(None)` rather than fabricating a pointwise grow. `RankingState` is the landed structural coverage-decision seam (like `PairwiseState`).
- **QueryCrossEntropy independently deferred (Open Q3).** No `cb_compute::ranking_der` CPU der oracle / `Loss` variant exists for QueryCrossEntropy yet, so its bounded shift search is landed structurally (a genuine, self-consistent root-find) and gated OFF (`ranking_objective_covered` returns `false`) — the executor does not ship unverified der parity. The independent gate keeps QueryRMSE/QuerySoftMax covered even though QueryCrossEntropy's full der is deferred.
- **Runtime FINITE shift bracket.** Seeding `lo`/`hi` from a comptime f64 const makes them comptime-typed, which then clashes when reassigned from the runtime bisection `mid` (a `f64: From<NativeExpand<f64>>` `#[cube]` type error). The host passes `QCE_SHIFT_BRACKET` in as runtime data (the `mvs_device` runtime-bracket precedent).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Runtime-passed FINITE shift bracket (comptime/runtime type clash)**
- **Found during:** Task 2 (QueryCrossEntropy shift kernel)
- **Issue:** Seeding the bisection `lo`/`hi` directly from the `const QCE_SHIFT_BRACKET: f64` made them comptime-typed; reassigning `lo = mid` (a runtime value) inside the loop produced a `#[cube]` `f64: From<NativeExpand<f64>>` compile error.
- **Fix:** Pass the bracket as a runtime `params: &Array<f64>` kernel argument (read `params[0]`), so `lo`/`hi` are runtime f64 from the start — the `mvs_device` runtime-bracket precedent.
- **Files modified:** crates/cb-backend/src/gpu_runtime/ranking.rs
- **Verification:** `cargo check -p cb-backend` compiles; `cargo test ... ranking` 3/3 green (cpu + rocm).
- **Committed in:** `641e0f0` (Task 1/2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking `#[cube]` type-expansion fix).
**Impact on plan:** Necessary for the QueryCrossEntropy kernel to compile; no scope change — all artifacts as specified.

## Deferred Issues
- **Kaggle CUDA ε=1e-4 sign-off** for the ranking der is deferred to Plan 10 (per the plan's `success_criteria`); the numeric ε assertions are device-gated and validated in-env on rocm gfx1100 by the executor (the cpu run records-only, WR-01).
- **QueryCrossEntropy full der + `Loss` variant** (Open Q3): the CPU `ranking_der` QueryCrossEntropy arm + a `Loss::QueryCrossEntropy` variant are not landed, so the device der stays gated off. Only the bounded shift search (a self-consistent root-find) ships. A future plan adds the CPU oracle + der arm and flips the coverage flag.
- **Per-tree query-descriptor grow seam** (the covered-ranking device grow) is a forward dependency, mirroring the Plan-01 pairwise grow seam deferral.

## Next Phase Readiness
- The deterministic ranking der driver + the shared `query_helper` substrate are ready for Plan 05 (YetiRank / PFound-F stochastic ranking arms, which reuse the in-query sampling + grouping infra).
- Plan 10 will run the per-family Kaggle CUDA ε=1e-4 + BENCH-02 sign-off (human-gated).

---
*Phase: 13-pairwise-ranking-multiclass-ordered-langevin-device-coverage*
*Completed: 2026-07-04*

## Self-Check: PASSED
- FOUND: crates/cb-backend/src/gpu_runtime/ranking.rs
- FOUND: crates/cb-backend/src/gpu_runtime/ranking_det_test.rs
- FOUND: .planning/phases/13-.../13-04-SUMMARY.md
- FOUND commit 641e0f0 (Task 1/2), e575144 (Task 3)
