---
phase: 15-debt-discharge-cuda-oracle-re-establishment
plan: 02
subsystem: gpu
tags: [cubecl, rocm, cuda, pairwise, cholesky, tie-break, residency-guard, parity-oracle]

# Dependency graph
requires:
  - phase: 15-01
    provides: RV-13-01/02 ranking-der direct-invocation oracle pattern (sibling *_test.rs, device_backend_active gate)
provides:
  - RV-13-03 n==0 empty-group residency guard in compute_group_means_host (no zero-length device buffer)
  - RV-13-04 near-equal-tolerant lowest-index deterministic pairwise split tie-break (select_best_candidate + REL_TOL)
  - Two direct unit oracles (empty_group_means_no_fault, pairwise_near_equal_border_tiebreak)
affects: [device-grow-seam-wiring, gput-14-aggregate, hard-03, future milestone device grow]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Host-side residency guard BEFORE client.create (never bind/read a 0-len device handle)"
    - "Near-equal-tolerant, lowest-index-deterministic argmax as single-source-of-truth pub(crate) selector shared by production + oracle"

key-files:
  created: []
  modified:
    - crates/cb-backend/src/kernels/query_helper.rs
    - crates/cb-backend/src/kernels/query_helper_test.rs
    - crates/cb-backend/src/gpu_runtime/pairwise.rs
    - crates/cb-backend/src/kernels/cholesky_solve_test.rs

key-decisions:
  - "RV-13-03 guard returns vec![0.0; n_groups] (right length AND value), never Vec::new() — placed before selected_client()/client.create (Pitfall 3, HIP residency lesson)"
  - "RV-13-04 fixed via near-equal-tolerant tie-break (A2/D-02), NOT by forcing device Cholesky to bit-match host scorer (larger scope, rejected)"
  - "REL_TOL = 1e-9 relative: ~4 orders above the observed ~1e-13 device-vs-host delta, far below any real score gap"
  - "Tie-break extracted to pub(crate) select_best_candidate so production select_best_split_over_scores and the sibling oracle share one implementation (non-tautological: oracle also runs the replaced exact-== argmax and proves it FLIPS)"

patterns-established:
  - "Demonstrating oracle: keep the replaced buggy variant (exact_argmax) in the test to prove the fix bites (asserts the old rule flips 0->1 across accumulation orders)"

requirements-completed: [HARD-03]

# Metrics
duration: 18min
completed: 2026-07-05
status: complete
---

# Phase 15 Plan 02: RV-13-03/04 Latent Parity Hazard Oracles Summary

**Closed the remaining two RV-13 latent parity hazards (HARD-03, 2 of 4) with direct unit oracles: a `n==0` residency short-circuit in `compute_group_means_host` and a near-equal-tolerant, lowest-index-deterministic pairwise split tie-break — both validated on rocm gfx1100 in-env.**

## Performance

- **Duration:** 18 min
- **Started:** 2026-07-05T (plan execution start)
- **Completed:** 2026-07-05
- **Tasks:** 2/2
- **Files modified:** 4

## Accomplishments
- **RV-13-03** — added an `n==0` short-circuit to `compute_group_means_host` that returns `vec![0.0; n_groups]` BEFORE any `selected_client()`/`client.create`, so an all-empty-group offset (`q_offsets=[0,0]`) never binds a zero-length device buffer (the project HIP residency lesson). Fault-guard authoritatively validated on rocm gfx1100 in-env (D-03).
- **RV-13-04** — replaced the exact-`==` split argmax with `select_best_candidate`, a near-equal-tolerant (`|a-b| <= REL_TOL*max(|a|,|b|,1)`), lowest-index-deterministic rule extracted to a `pub(crate)` selector. The device-Cholesky and frozen host-scorer accumulation orders (which land ~1e-13 apart on genuinely-tied borders) now select the SAME border.
- Two non-tautological oracles added as sibling `*_test.rs` tests; both pass under `cpu` and `rocm`.

## Task Commits

1. **Task 1: RV-13-03 n==0 empty-group guard + oracle** - `b7e2e52` (fix)
2. **Task 2: RV-13-04 near-equal-tolerant deterministic tie-break + oracle** - `cdb3022` (fix)

_(tdd_mode disabled in config; each task committed test+impl atomically. The RV-13-03 guard produces no clean RED on the cpu backend since the "device" is the host — its value is the rocm/cuda residency fault it prevents, which is why the oracle asserts value+length, not just no-panic.)_

## Files Created/Modified
- `crates/cb-backend/src/kernels/query_helper.rs` - `if n == 0 { return Ok(vec![0.0; n_groups]); }` guard inside `compute_group_means_host`, before the wgpu/device branches.
- `crates/cb-backend/src/kernels/query_helper_test.rs` - `empty_group_means_no_fault` oracle: asserts `compute_group_means_host([], [], [0,0]) == Ok(vec![0.0])` (length 1, value 0.0). Runs on all backends; rocm/cuda fault-guard authoritative.
- `crates/cb-backend/src/gpu_runtime/pairwise.rs` - new `pub(crate) const REL_TOL: f64 = 1e-9` + `pub(crate) fn select_best_candidate(scores, cand_idxs, rel_tol) -> u32`; `select_best_split_over_scores` rewired to call it.
- `crates/cb-backend/src/kernels/cholesky_solve_test.rs` - `pairwise_near_equal_border_tiebreak` oracle + a local `exact_argmax` (the replaced rule) used to demonstrate the flip; imports `select_best_candidate`/`REL_TOL` via the `crate::gpu_runtime` glob re-export.

## Key Decisions
- **REL_TOL = 1e-9 (relative).** Chosen just above the observed device-vs-host accumulation delta (~1e-13 absolute at score magnitude ~1e1 ⇒ ~1e-14 relative) yet far below any genuine score gap. The oracle confirms both near-equal deltas fall inside the band while well-separated borders (5.0 vs 10.0) still pick the strictly-higher score.
- **A2/D-02:** fixed the tie-break rather than forcing the two solves bit-identical (larger scope, explicitly out of bounds).
- **Single source of truth:** production and oracle call the same `select_best_candidate`; the oracle additionally exercises the retired exact-`==` argmax and asserts it FLIPS (device→0, host→1), proving the new rule is not a tautology.

## Evidence (Wave C)
- **RV-13-03 rocm in-env fault-guard (D-03, authoritative):** `cargo test -p cb-backend --no-default-features --features rocm empty_group_means_no_fault` → `device_backend_active=true`, `empty-group means = [0.0]`, PASS. No zero-length-handle fault on gfx1100.
- **RV-13-04 rocm in-env smoke (non-gating):** `pairwise_near_equal_border_tiebreak` → `device=0 host=0 (exact flip 0->1); separated winner=1 (REL_TOL=1e-9)`, PASS.
- **cpu backend:** both oracles PASS; `query_helper_test` 7/7, `cholesky_solve_test` 5/5.
- **Chosen REL_TOL:** `1e-9`.
- Kaggle CUDA authoritative-session validation of both oracles remains for the aggregate GPUT-14 run (HARD-01, plans 15-03/04) per D-03/D-04.

## Deviations from Plan

**None — plan executed as written.** The `select_best_candidate` symbol name (already referenced in existing pairwise.rs doc comments as the parity-contract name) was newly created as a `pub(crate)` fn in `cb-backend`; it is distinct from the unrelated `cb_train::tree::select_best_candidate` (different crate, different signature), no conflict.

## Deferred Issues (out of scope — pre-existing)
- `cargo test -p cb-backend --features cpu pairwise` shows 10 pre-existing FAILs in `pairwise_hist::*`, `score_split::pairwise::*`, `grow_loop::pairwise::*`: `not yet implemented: atomic<f64>` in `cubecl-cpu` (the histogram/der-sum scatter kernels have no cpu backend). These fail before reaching the argmax finalization this plan touched and require a real device backend (rocm/cuda) — they are unrelated to RV-13-03/04 and out of this plan's scope boundary. Not fixed, not regressed.

## Prohibitions Honored
- No `cb-train` dep in `cb-backend` (`grep -c 'cb-train\|cb_train' crates/cb-backend/Cargo.toml` = 0).
- No `#[cfg(test)] mod tests` embedded in `query_helper.rs` or `pairwise.rs` (source/test separation MANDATORY; oracles live in sibling `*_test.rs`).
- RV-13-03 guard returns the right-length `vec![0.0; n_groups]`, before `client.create`.
- RV-13-04 oracle uses near-equal (not well-separated) borders (Pitfall 4).

## Self-Check: PASSED
- Files verified present: query_helper.rs, query_helper_test.rs, pairwise.rs, cholesky_solve_test.rs (all modified).
- Commits verified in git log: `b7e2e52`, `cdb3022`.
