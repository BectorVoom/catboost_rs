---
phase: 15-debt-discharge-cuda-oracle-re-establishment
plan: 01
subsystem: testing
tags: [cubecl, rocm, ranking, querysoftmax, yetirank, gpu-parity, oracle, rv-13]

# Dependency graph
requires:
  - phase: 13-cuda-der-families
    provides: "device YetiRank/PFound-F + QuerySoftMax der host-drivers (ranking.rs) that the RV-13-01/02 review flagged as latent parity hazards"
provides:
  - "RV-13-01 tie-order oracle (confirmatory): descending_order_per_query proven stable-descending-equivalent to CPU on the real gfx1100 device with deliberate exact ties"
  - "RV-13-02 fix: QuerySoftMax exp-shift seeded from the weight>0 per-query max (compute_group_max_weighted_host), matching CPU ranking_der.rs:257-266"
  - "Two non-tautological sibling oracles in ranking_stoch_test.rs (tie_order_matches_cpu_stable_descending, softmax_weight_max_seed)"
  - "pub(crate) visibility on descending_order_per_query + new pub(crate) compute_group_max_weighted_host"
affects: [15-EVIDENCE, HARD-03, device-grow-wiring, ranking-der-parity]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Frozen-CPU-reference sibling oracle with device-gated ε assert (WR-01 record-only on cpu)"
    - "Oracle teeth via direct seed-selection assert when the numeric der is shift-invariant"

key-files:
  created: []
  modified:
    - crates/cb-backend/src/gpu_runtime/ranking.rs
    - crates/cb-backend/src/gpu_runtime/ranking_stoch_test.rs

key-decisions:
  - "RV-13-01 is CONFIRMATORY (verified-stable + oracle added), not a code fix — the tie oracle passed as-is on the real gfx1100 device"
  - "RV-13-02 oracle teeth come from a direct weight>0-max seed-selection assert (1.0 not 3.0), because softmax shift-invariance makes the der numerically identical (~1e-16) regardless of seed"
  - "RV-13-01 order assert is device-gated: descending_order_per_query relies on plane_inclusive_sum (segmented_radix_sort), unsupported on the cubecl cpu backend"

patterns-established:
  - "When a fix target is numerically shift-invariant, assert the intermediate (seed) directly for non-tautological teeth, and keep the der parity assert device-gated"

requirements-completed: [HARD-03]

# Metrics
duration: ~12min
completed: 2026-07-05
status: complete
---

# Phase 15 Plan 01: RV-13-01/02 Ranking-der Parity Oracles Summary

**RV-13-01 confirmed tie-order-stable (oracle-guarded, no code churn) and RV-13-02's QuerySoftMax exp-shift re-seeded from the weight>0 per-query max to match CPU — both proven on the real gfx1100 device, 2 of 4 HARD-03 hazards discharged.**

## Performance

- **Duration:** ~12 min
- **Started:** 2026-07-05T05:29:12Z
- **Completed:** 2026-07-05T05:40:55Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- **RV-13-01 (confirmatory discharge):** wrote the tie oracle FIRST (per Pitfall 1 / A1). A 2-query fixture with deliberate exact ties (incl. a 3-way tie) checked against an INDEPENDENT CPU stable descending sort (`sort_by (value desc, index asc)`) — the device `descending_order_per_query` matched exactly on the real gfx1100 device. The existing complemented-key stable radix already preserves tie order; NO body change was needed, only a `pub(crate)` bump + a doc-anchored tie-stability contract note.
- **RV-13-02 (real fix):** replaced the weight-blind `compute_group_max_host(approx, ..)` seed inside `query_softmax_ders_host` with a new host-side `compute_group_max_weighted_host` that takes the per-query max over **weight>0 objects only** (seeding `f64::MIN` for an all-w≤0 query, relying on the intact downstream `sum_weighted_targets > 0` short-circuit), mirroring CPU `ranking_der.rs:257-266`. No `#[cube]` kernel edit (Open Q2 → host-side).
- Both oracles pass on cpu (record-only) AND on the real rocm gfx1100 device (der1 max_div 1.1e-16, der2 0.0 for RV-13-02; exact permutation equality for RV-13-01). All 7 `ranking_stoch` tests green on rocm — no regression to the existing YetiRank/PFound-F/seed/draw-count oracles.

## Task Commits

1. **Task 1: RV-13-01 tie-order oracle** - `e77561d` (test) — `pub(crate) descending_order_per_query` + `tie_order_matches_cpu_stable_descending`
2. **Task 2: RV-13-02 weight>0 max-seed fix + oracle** - `e71d6dd` (fix) — `compute_group_max_weighted_host` + `softmax_weight_max_seed`

_Task 1 is a single TDD commit: the oracle passed as-is (confirmatory), so RED and "GREEN" coincide — no defensive body change was warranted (D-01 valid discharge)._

## Files Created/Modified
- `crates/cb-backend/src/gpu_runtime/ranking.rs` — `descending_order_per_query` raised to `pub(crate)` + tie-contract doc; new `pub(crate) compute_group_max_weighted_host`; `query_softmax_ders_host` re-seeded from the weight>0 max; removed the now-unused `compute_group_max_host` import.
- `crates/cb-backend/src/gpu_runtime/ranking_stoch_test.rs` — imports for the two functions under test + `compute_group_max_weighted_host`; `cpu_stable_descending_order` reference helper; `tie_order_matches_cpu_stable_descending` (RV-13-01); `softmax_fixture` + `frozen_softmax_der1/2` (with the offline generation recipe in doc) + `softmax_weight_max_seed` (RV-13-02).

## Decisions Made
- **RV-13-01 outcome = CONFIRMATORY (for Wave C 15-EVIDENCE):** the tie oracle was green on the real device with the existing code. Per Pitfall 1 / A1 / D-01, "verified-stable + oracle added" is a valid HARD-03 discharge — no speculative churn to the working complemented-key stable radix. Recorded here explicitly for the 15-EVIDENCE per-hazard table.
- **RV-13-02 der is shift-invariant:** because the QuerySoftMax share `p` divides by `sum_exp`, a common `exp(-beta·Δmax)` factor cancels — the weight-blind and weight>0 seeds yield the SAME der to ~1e-16 for normal-range fixtures. The oracle's real TEETH are therefore the direct `compute_group_max_weighted_host` seed-selection assert (`[1.0, 0.8]`, where the pre-fix weight-blind seed would have been `3.0` on the weight-0 doc), backed by device-gated der parity. This makes the fix a semantic + exact-seed-parity alignment with CPU (not a catastrophic value correction).
- **RV-13-01 order assert is device-gated:** `descending_order_per_query` → `segmented_radix_sort` uses `plane_inclusive_sum`, which the cubecl `cpu` backend does not support (it panics in worker threads and yields garbage indices). The exact-permutation assertion runs only under `device_backend_active()` (rocm/cuda), record-only on cpu — the same WR-01 discipline the file already applies to the ε der asserts. This is a small, necessary deviation from the plan's assumption that the ordering assert "MAY run on all backends" (see Deviations).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Device-gated the RV-13-01 exact-order assertion**
- **Found during:** Task 1 (RV-13-01 tie oracle)
- **Issue:** The plan stated the exact index-equality ordering assert "MAY run on all backends." In practice `descending_order_per_query` calls `segmented_radix_sort`, which uses `plane_inclusive_sum` — UNSUPPORTED on the cubecl `cpu` backend (the kernel panics in DSD worker threads and the returned order is garbage: `[0,0,0,0,0,7,7,7,0]`). A non-gated assert hard-fails on `--features cpu`, contradicting the plan's own prohibition ("No … assert that hard-fires on the default cpu backend — must be gated behind device_backend_active()").
- **Fix:** Wrapped the exact-permutation equality (and the explicit expected `[3,0,1,2,4,7,5,6,8]`) in `if device_backend_active()`, printing a record-only note + the CPU reference on cpu. The fixture-has-ties sanity check and the full-permutation length check still run on all backends. The authoritative validation runs in-env on rocm gfx1100 (passed).
- **Files modified:** crates/cb-backend/src/gpu_runtime/ranking_stoch_test.rs
- **Verification:** `--features cpu` passes (record-only); `--features rocm` passes with exact order equality on the real device.
- **Committed in:** e77561d (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** The deviation aligns the oracle with the repo's actual cubecl-cpu capability and the file's own WR-01 device-gating convention. No scope change; both hazards discharged as specified.

## Issues Encountered
- **RV-13-02 "wrong-value bug" is shift-invariant, and the pathological regime is a shared kernel limitation.** A fixture with the weight-0 max-doc approx large enough to make the pre-fix seed *underflow* the weight>0 objects to exactly zero (gap ≳ 745) also makes the FIXED path *overflow* that weight-0 doc's `exp(beta·(a−max_a))·w = inf·0 = NaN` in the shared `#[cube]` `query_softmax_der_kernel` (the sum_exp loop does not skip w≤0). So no clean fixture demonstrates "pre-fix wrong / post-fix right" at the der level. This is why the oracle asserts the SEED directly for teeth. The residual shared inf·0-overflow robustness gap (only reachable at pathological magnitudes, and out of this plan's host-side scope) is noted here as latent — a future kernel-level hardening item, NOT a blocker for HARD-03's parity discharge.

## Known Stubs
None — both oracles exercise real code paths and assert against independent CPU references.

## Threat Flags
None — no new trust boundary, network, auth, or serialization surface (per the plan threat_model; changes are internal der host-drivers + sibling unit oracles).

## Next Phase Readiness
- **HARD-03: 2 of 4 hazards discharged.** RV-13-03 (`compute_group_means_host` n==0 guard) and RV-13-04 (pairwise Cholesky near-equal-border tie-break) remain for sibling plans in Wave A.
- **For Wave B (single-session Kaggle CUDA run):** the two new oracles (`tie_order_matches_cpu_stable_descending`, `softmax_weight_max_seed`) are ready to ride the aggregate `--features cuda` session; both hard-fire their ε/order asserts on a real device.
- **For Wave C (15-EVIDENCE):** RV-13-01 = confirmatory (verified-stable + oracle); RV-13-02 = real seed-parity fix + oracle; the RV-13-02 frozen-der offline generation recipe is documented in the test module doc.

## Self-Check: PASSED
- `crates/cb-backend/src/gpu_runtime/ranking.rs` — FOUND
- `crates/cb-backend/src/gpu_runtime/ranking_stoch_test.rs` — FOUND
- Commit `e77561d` — FOUND
- Commit `e71d6dd` — FOUND
- `pub(crate) fn descending_order_per_query` — present (grep count 1)
- `cb-train` dep in cb-backend/Cargo.toml — absent (grep count 0)
- Both oracles pass on `--features cpu` (record-only) and `--features rocm` (real device)

---
*Phase: 15-debt-discharge-cuda-oracle-re-establishment*
*Completed: 2026-07-05*
