---
phase: 05-ordered-boosting-ordered-ctr-categoricals-high-risk-parity-s
plan: 16
subsystem: testing
tags: [ordered-boosting, ord-02, wiring-test, identity-fold, oracle-delegation, gap-closure]

# Dependency graph
requires:
  - phase: 05 (plan 05-12)
    provides: "create_folds identity-Folds[0] semantics — the load-bearing fact that makes the ordered structure search consume object order for all permutation_count"
  - phase: 05 (plan 05-15)
    provides: "multi-permutation pre-averaging draw order corrected/locked (sequencing hygiene only)"
provides:
  - "A GREEN ordered_boost_wiring_test suite (3/3) — the only failing test at HEAD is resolved"
  - "ordered_structure_differs_from_plain RETIRED in place (renamed ordered_branch_alive_structural_authority_is_e2e_oracle) with an in-file rationale delegating ORD-02 structural authority to ordered_boost_e2e_oracle_test"
  - "05-DEFERRED.md — an auditable, discoverable record of the retire decision co-located with the phase artifacts"
affects: [phase-05-verification, phase-05-completion]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "When a unit/wiring assertion's premise is invalidated by upstream-faithful behavior (not a bug), RETIRE it in place with an in-file rationale and delegate the guarantee to the authoritative oracle — never fabricate a diverging dataset or weaken the e2e gate"

key-files:
  created:
    - .planning/phases/05-ordered-boosting-ordered-ctr-categoricals-high-risk-parity-s/05-DEFERRED.md
  modified:
    - crates/cb-train/tests/ordered_boost_wiring_test.rs

key-decisions:
  - "PRIMARY path taken (as planned): the boosting.rs:~1054 find(|f| !f.is_averaging) reading was CONFIRMED — the ordered structure search consumes the IDENTITY Folds[0] for ALL permutation_count, so re-keying the test cannot make assert_ne! hold without an out-of-scope production fold-selection change. Retire in place."
  - "ORD-02 structural-correctness authority delegated to ordered_boost_e2e_oracle_test (2/2 <=1e-5 vs catboost 1.2.10); aliveness gates ordered_training_grows_a_full_finite_model + plain_path_still_trains preserved unchanged"
  - "Retire decision recorded in 05-DEFERRED.md (discoverable, co-located); .planning/todos/ NOT created (does not exist)"

requirements-completed: [ORD-02]

# Metrics
duration: ~15min
completed: 2026-06-14
---

# Phase 5 Plan 16: Retire Invalidated Ordered-vs-Plain Wiring Assertion (GAP 1 / ORD-02) Summary

**Resolved the only failing test at HEAD by retiring the invalidated `ordered_structure_differs_from_plain` assertion in place with a documented rationale, delegating ORD-02 structural authority to the e2e oracle — no production code changed and no genuine parity guarantee lost.**

## Performance

- **Duration:** ~15 min
- **Tasks:** 1/1
- **Files modified:** 2 (1 test, 1 new DEFERRED note); 0 production source

## Accomplishments

### Task 1 — retire `ordered_structure_differs_from_plain` (commit 9a2c974)

**Investigation (PRIMARY-path confirmation).** The plan's load-bearing fact was
verified directly in source: `crates/cb-train/src/boosting.rs:1054-1057` selects
the ordered learning permutation via `find(|f| !f.is_averaging)`, which returns
`Folds[0]` = the IDENTITY learning fold for EVERY `permutation_count` (after the
05-12 identity-`Folds[0]` change). The failing test was reproduced at HEAD: under
`EBoostingType::Ordered` and `EBoostingType::Plain` on the randomness-free
synthetic dataset (`bootstrap=No`, `random_strength=0`), BOTH produce splits
`[(1, 8.5), (0, 1.5)] × 5` — identical, so `assert_ne!` fails. This confirms the
premise is invalidated by upstream-faithful behavior, NOT a dead Ordered branch.
Re-keying `permutation_count` cannot help (the ordered search still consumes the
identity fold) — making it consume a non-identity fold is an out-of-scope
production change, and the e2e oracle already locks ORD-02 ≤1e-5. PRIMARY path
taken; CONDITIONAL option (b) correctly NOT triggered.

**Retire in place.** `ordered_structure_differs_from_plain` was renamed
`ordered_branch_alive_structural_authority_is_e2e_oracle` and its failing
`assert_ne!` replaced with a passing, self-documenting body:
- a top-of-fn `//` rationale block explaining the identity-`Folds[0]` consumption
  (`boosting.rs:~1054`), the upstream-faithful `shuffle = foldIdx != 0`
  (`fold.cpp:54`) origin, and the delegation of ORD-02 structural authority to
  `ordered_boost_e2e_oracle_test`;
- positive assertions that BOTH Ordered and Plain grow full, finite 5-tree models
  (Ordered branch ALIVE) with depth-2 shape parity;
- an `assert_eq!(ordered_splits, plain_splits)` with a message naming
  `ordered_boost_e2e_oracle_test` as the authoritative ≤1e-5 ORD-02 check — the
  legitimate coincidence on this randomness-free identity-fold config.

The file header comment was also updated to reflect the retirement. The two
aliveness gates (`ordered_training_grows_a_full_finite_model`,
`plain_path_still_trains`) were left UNCHANGED.

**Auditable record.** Created
`.planning/phases/05-.../05-DEFERRED.md` with a dated entry stating WHAT was
retired, WHY (identity-fold consumption; re-keying impossible without out-of-scope
production change), and that ORD-02 structural authority rests on
`ordered_boost_e2e_oracle_test`. No `.planning/todos/` directory was created.

**Verification:**
- `cargo test -p cb-train --test ordered_boost_wiring_test` — 3/3 PASS (was 2/1 fail).
- `cargo test -p cb-train --test ordered_boost_e2e_oracle_test` — 2/2 (authoritative ORD-02, untouched).
- `cargo test -p cb-train --test ordered_boost_oracle_test` — 5/5.
- `cargo test -p cb-train --lib` — 130/130 (no regression).
- `cargo check --tests -p cb-train` — 0 errors, 0 warnings.
- `git diff --name-only` — ONLY `crates/cb-train/tests/ordered_boost_wiring_test.rs` (no production source).
- grep: `ordered_boost_e2e_oracle_test` in test = 7, in DEFERRED = 3; aliveness gates present = 1 each.

## Deviations from Plan

None — Task 1 executed exactly as the PRIMARY path specifies. The
boosting.rs read CONFIRMED the identity-fold reading, so the CONDITIONAL
re-key (option b) was correctly NOT taken.

## Known Stubs

None. No placeholder data or empty-value stubs introduced; the retired test is a
fully-evaluated, documented decision, not a stub.

## Threat Flags

None. T-05-16-01 (structural-correctness tampering) is mitigated by delegation:
the retired wiring assertion's authority rests on `ordered_boost_e2e_oracle_test`
(2/2 ≤1e-5 vs upstream) and the `ordered_training_grows_a_full_finite_model`
aliveness gate. Test-only change; no untrusted input, network, or auth path
touched.

## Self-Check: PASSED

- crates/cb-train/tests/ordered_boost_wiring_test.rs — FOUND (modified)
- .planning/phases/05-.../05-DEFERRED.md — FOUND (new)
- Commit 9a2c974 (Task 1) — FOUND
