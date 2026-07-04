---
phase: 13-pairwise-ranking-multiclass-ordered-langevin-device-coverage
plan: 02
subsystem: gpu-training
tags: [cubecl, cholesky, spd-solve, pairwise, gput-21, rocm, ranking, catboost]

# Dependency graph
requires:
  - phase: 13-pairwise-ranking-multiclass-ordered-langevin-device-coverage
    provides: Plan 01 device-resident packed lower-triangular pairwise linearSystem assembly + PairwiseState coverage gate
  - phase: 07-gpu-cuda-structural-parity
    provides: Phase 7.4 4-channel pairwise histograms + Phase 7.5 pairwise scorer (host solve, "Open Q3")
provides:
  - Batched f64 device Cholesky SPD solver (#[cube] decomp + fwd/back subst + ridge + CalcScoresCholesky, drop-last-row + zero-average) matching the Rust CPU leaf-value + split-score paths at ε=1e-4
  - launch_cholesky_solve resident launcher (Leaf + Score modes) + solve_pairwise_leaf_values_host / score_pairwise_cholesky_host readback wrappers
  - Wired device solve into the pairwise split scorer (launch_pairwise_split_score) — Phase-7.5 host "Open Q3" solve closed (D-05 full residency)
affects: [13-03-query-grouping, 13-06-block-leaf-newton, gput-21, pairwise-ranking-device-grow, phase-14-aggregate-signoff]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Batched f64 SPD Cholesky #[cube] kernel (serial unit-0, per-system reused scratch: L m×m + y + x + res); non-positive pivot → zeros fallback (no NaN)"
    - "One parameterized solver serving both the leaf-value system (size leaf_count, Leaf mode) and the split-score system (size 2·PartCount, Score mode) — RESEARCH Open Q1 resolved"
    - "Host-assembled bounded per-border 2×2 systems + device batch solve; only the O(borders) descriptor crosses the seam (D-05), Open Q3 histogram cross-cube-carry untouched"

key-files:
  created:
    - crates/cb-backend/src/kernels/cholesky_solve.rs
    - crates/cb-backend/src/kernels/cholesky_solve_test.rs
  modified:
    - crates/cb-backend/src/kernels.rs
    - crates/cb-backend/src/gpu_runtime/pairwise.rs

key-decisions:
  - "Device kernel matches the CPU cholesky_solve exactly (sum<=0 → zeros fallback, NO 1e-7 pivot floor) — the ε=1e-4 oracle is the Rust CPU path, not upstream RegularizeImpl (Pitfall 2); ridge constants (cell_prior=1/n, non_diag_reg=-prior/n, diag_reg=prior*(1-1/n)+l2) transcribed inline"
  - "Human checkpoint Task 3: WIRE-DEVICE (full residency, D-05) — replaced the 7.5 host calculate_pairwise_score solve with launch_cholesky_solve at the launch_pairwise_split_score call site; wgpu (no f64) retains the host scorer (no regression)"
  - "Bounded per-border 2×2 assembly transcribed inline as the leaf_count==1 (depth-1 MVP) specialization of calculate_pairwise_score's running accumulation (no off-diagonal leaf pairs) — bit-identical inputs to the host scorer, so the device scores are byte-equal"

patterns-established:
  - "Pattern 1 (Cholesky): device batched f64 SPD solve, self-oracled bit-for-bit vs cb_compute::pairwise_cholesky_solve / calculate_pairwise_leaf_values (leaf) and calculate_score (split score)"
  - "Pattern F self-oracle: device solve vs inline CPU reference over equal-length buffers; numeric assert skipped off rocm/cuda (WR-01); degenerate non-PD zeros fallback asserted unconditionally (T-13-03)"

requirements-completed: []

# Metrics
duration: ~90min
completed: 2026-07-04
status: complete
---

# Phase 13 Plan 02: On-Device Batched f64 Cholesky Solver Summary

**A batched f64 device Cholesky SPD solver (`#[cube]` decomposition + forward/back substitution + ridge + `CalcScoresCholesky`, drop-last-row + zero-average) that matches the Rust CPU pairwise leaf-value and split-score paths bit-for-bit, WIRED into the pairwise split scorer to close the Phase-7.5 host "Open Q3" solve (D-05 full residency, GPUT-21).**

## Performance

- **Duration:** ~90 min (incl. the Task-3 human decision round-trip)
- **Completed:** 2026-07-04
- **Tasks:** 3 (2 engineering + 1 human-decision checkpoint)
- **Files modified:** 4 (2 created, 2 modified)

## Accomplishments
- New `#[cube]` `cholesky_solve_kernel` (serial, batched over per-leaf SPD systems): in-place lower-triangular decomposition (`a = L·Lᵀ`), forward `L·y=b`, back `Lᵀ·x=y`, the `calculate_pairwise_leaf_values` ridge (drop-last-row → `(n-1)×(n-1)`, push trailing 0, `make_zero_average`), and an optional `CalcScoresCholesky` score path — ONE kernel parameterized by system size + a Score/Leaf mode, serving both the leaf-value system (`leaf_count`) and the split-score system (`2·PartCount`) per RESEARCH Open Q1. f64 accumulation throughout (D-07); a non-positive pivot → zeros fallback (matches the CPU `None`, no NaN, T-13-03).
- `launch_cholesky_solve` resident launcher (returns the handle, no readback, D-05) + `solve_pairwise_leaf_values_host` / `score_pairwise_cholesky_host` readback wrappers; typed `wgpu` reject (WGSL has no f64), no `-inf` literal, no `unwrap`/`expect`/`panic`/indexing.
- Self-oracle `kernels/cholesky_solve_test.rs` (source/test separation): 3-/4-leaf leaf-value fixtures + a 2-leaf split-score fixture + a degenerate non-PD fixture, all vs inline CPU references reusing `cb_compute::pairwise_cholesky_solve` (NO `cb-train` dep). cpu-backend run **4/4 green, device-vs-CPU divergence `0e0` (bit-identical)**; the ε=1e-4 numeric assert fires only on rocm/cuda (WR-01 anti-false-pass).
- **Task 3 (human decision: wire-device)** — replaced the Phase-7.5 host `calculate_pairwise_score` solve at the `launch_pairwise_split_score` call site with the resident `launch_cholesky_solve(...)`: the bounded per-border 2×2 `(weight_sum, der_sum)` systems are host-assembled (`assemble_pairwise_score_systems_leaf1`, the `leaf_count==1` MVP specialization) and batch-solved ON DEVICE. Only the O(borders) descriptor crosses the seam; the tracked Open Q3 histogram cross-cube-carry scope is untouched. `wgpu` retains the host scorer (no regression).
- No `cb-train` dep leaked into `cb-backend` (landmine grep == 0); no `-inf` in any `#[cube]` body (grep == 0).

## Task Commits

Each task was committed atomically:

1. **Task 1: Batched f64 device Cholesky solver kernel** - `deb3b00` (feat)
2. **Task 2: Self-oracle vs CPU leaf-value + split-score solvers** - `ef4115c` (test)
3. **Task 3: Wire device solve into the pairwise split scorer (human decision: wire-device)** - `c221e7c` (feat)

**Plan metadata:** see the final docs commit.

_Tasks 1 and 2 both touch `kernels.rs`; Task 1 was committed with only the production `mod cholesky_solve` registration (compiling in isolation), and Task 2 added the `#[cfg(test)] mod cholesky_solve_test` registration + the test file — so every commit compiles._

## Files Created/Modified
- `crates/cb-backend/src/kernels/cholesky_solve.rs` (NEW) - the `#[cube]` batched f64 Cholesky solver + `launch_cholesky_solve` + host readback wrappers.
- `crates/cb-backend/src/kernels/cholesky_solve_test.rs` (NEW) - self-oracle (leaf values, split score, degenerate non-PD).
- `crates/cb-backend/src/kernels.rs` - `pub(crate) mod cholesky_solve` + `#[cfg(test)] mod cholesky_solve_test` registration.
- `crates/cb-backend/src/gpu_runtime/pairwise.rs` - `assemble_pairwise_score_systems_leaf1` + the wired device solve at the `launch_pairwise_split_score` call site (wgpu-gated host fallback).

## Decisions Made
- **Match the CPU `cholesky_solve` exactly, not upstream:** the device kernel replicates `cb_compute::leaf::cholesky_solve` (a non-positive pivot returns zeros, NO 1e-7 pivot floor) and the `calculate_pairwise_leaf_values` ridge constants inline, because the ε=1e-4 oracle is the **Rust CPU path**, not upstream `linear_solver.cu::RegularizeImpl`'s bump-heuristics (Pitfall 2). This yields bit-for-bit agreement on the cpu backend.
- **wire-device (Task 3 human decision):** full residency (D-05) — the small bounded SPD solve now runs on-device, closing the Phase-7.5 "Open Q3". Correctness is identical to the retained-host alternative; the host scorer is kept only on `wgpu` (no f64) to avoid any regression there.
- **Bounded 2×2 assembly inline:** for the depth-1 MVP (`leaf_count==1`, system_size=2) there are no off-diagonal leaf pairs, so `calculate_pairwise_score`'s running accumulation reduces to the diagonal 2×2 block — transcribed inline so the device solver receives byte-identical inputs to the host scorer.

## Deviations from Plan

### Scoping clarifications (documented, not auto-fix rules)

**1. Tasks 1 + 2 committed as separate compiling slices around the shared `kernels.rs`**
- **Found during:** Task 1/2 (both register a module in `kernels.rs`)
- **Resolution:** Task 1 committed the production `mod cholesky_solve` registration only (so `cargo test` compiles without the not-yet-written test file); Task 2 added the `#[cfg(test)]` test registration + file. Two atomic, individually-compiling commits (vs Plan-01's single combined commit) — same "compiling-at-every-commit" invariant.

**2. Task 3 wiring reproduced the bounded per-border assembly inline (leaf_count==1)**
- **Found during:** Task 3 (wire-device)
- **Issue:** `cb_compute::calculate_pairwise_score` bundles the border-scan assembly AND the host solve; to run only the SOLVE on device without a cb-compute change (out of the 4-file scope) or expanding the Open Q3 histogram scope, the bounded per-border 2×2 `(weight_sum, der_sum)` assembly was transcribed inline (the `leaf_count==1` MVP specialization — no off-diagonal leaf pairs).
- **Fix:** `assemble_pairwise_score_systems_leaf1` (host, `#[cfg(not(wgpu))]`) + `launch_cholesky_solve(Score)`; `wgpu` retains the host scorer.
- **Verification:** device scores bit-identical to `calculate_pairwise_score` for the same inputs (Task 2 proved the solve is bit-exact); pairwise-family cpu-backend failing set IDENTICAL before/after (see Issues).

---

**Total deviations:** 2 scoping clarifications (no auto-fix rules triggered).
**Impact on plan:** All landed deliverables (solver kernel, self-oracle, wired device solve) are complete and cpu-validated bit-identically. No scope creep; the Open Q3 histogram cross-cube-carry was explicitly left untouched per the coordinator's instruction.

## Issues Encountered
- **Pre-existing pairwise-family cpu-backend failures (10 tests) are NOT a regression:** the `pairwise_hist` (×7), `score_split::pairwise::{scan_matches_reference, score_matches_cpu_oracle}`, and `grow_loop::pairwise::matches_cpu_pairwise_grow` tests fail on the default `cpu` backend because the device 4-channel `Atomic<F>` histogram / der-sum path is unreliable on the CubeCL **cpu runtime** (validated only on rocm/cuda in-env). A rigorous before/after comparison (stash of the Task-3 `pairwise.rs` change) confirmed the failing SET is **byte-identical** (7 passed / 10 failed both before and after — `diff` empty except the timing line). My wiring feeds the device solver the SAME inputs the host scorer received and the solve is bit-exact, so it adds **zero** new failures. The orchestrator discharges the rocm suite in-env.
- The `cholesky_solve` self-oracle (4 tests) uses pure serial f64 arithmetic (no atomics), so it is reliable and green on the cpu backend.

## Known Stubs
None. The solver is fully wired into the pairwise split scorer; the leaf-value readback wrapper (`solve_pairwise_leaf_values_host`) is exercised by the self-oracle and available for the pairwise leaf-estimation path.

## Requirements Status
- **GPUT-21 remains OPEN (not marked complete).** The device batched Cholesky (assembly + decomp + fwd/back subst + ridge + `CalcScoresCholesky`) is landed, wired, and self-oracled bit-for-bit on the cpu backend — but the requirement's completion is gated on the **Kaggle CUDA ε=1e-4 + BENCH-02 sign-off (Plan 10, human-gated)** per the phase's standing gate. Not marking it complete avoids fabricating the CUDA authority (Pitfall 6). The requirement flips to complete when Plan 10 signs off.

## Threat Flags
None. The new kernel introduces no new trust-boundary surface (plain-host-struct seam, bounded grid, checked casts); T-13-03 (non-PD pivot → zeros, no NaN) and T-13-04 (bounded launch geometry) are mitigated as planned.

## Next Phase Readiness
- Plan 03 (SHARED device query-grouping infra, GPUT-22 prereq) is independent of this solver and can proceed.
- The batched Cholesky is reusable for the pairwise **leaf-value** estimation path (`solve_pairwise_leaf_values_host` / the resident `launch_cholesky_solve` Leaf mode) when the per-tree pair/group seam lands.
- **Human-gated:** the pairwise GPUT-21 Kaggle CUDA ε=1e-4 + BENCH-02 sign-off is Plan 10; the orchestrator discharges the rocm in-env smoke. Do NOT fabricate either.

## Self-Check: PASSED

---
*Phase: 13-pairwise-ranking-multiclass-ordered-langevin-device-coverage*
*Completed: 2026-07-04*
