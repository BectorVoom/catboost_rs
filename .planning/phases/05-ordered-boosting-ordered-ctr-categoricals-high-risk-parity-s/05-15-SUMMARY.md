---
phase: 05-ordered-boosting-ordered-ctr-categoricals-high-risk-parity-s
plan: 15
subsystem: testing
tags: [ordered-boosting, ctr, fold-permutation, rng-draw-order, oracle, catboost-parity]

# Dependency graph
requires:
  - phase: 05 (plans 05-12, 05-13, 05-14)
    provides: "create_folds identity-Folds[0] + AveragingFold draw machinery; the 05-14 call-count-1 pre-averaging draw and the tensor_ctr_e2e hard gate"
provides:
  - "create_folds pre-averaging GenRand fired at the averaging-fold position (idx == learning_folds) for ALL permutation_count, not only the gated pc=1"
  - "A catboost-1.2.10-anchored permutation_count>=2 AveragingFold draw-order oracle (multi_permutation_fold_oracle_test) with the pc=2 partition locked integer-exact against committed upstream leaf_weights"
  - "A committed catboost 1.2.10 multi_permutation_fold fixture (leaf_weights + full model_pc{1,2,4}.json) for pc=1,2,4"
  - "A documented pc=4 (production-default) AveragingFold draw divergence requiring C++ instrumentation — tracked, not silently passed"
affects: [phase-05-verification, phase-06, future-pc4-gap-closure]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Position-based RNG draw guard (idx == learning_folds) replacing a stateful first-shuffle flag"
    - "Upstream-anchored oracle via observable model leaf_weights (the AveragingFold partition counts) when the raw permutation is not API-exposed"

key-files:
  created:
    - crates/cb-train/tests/multi_permutation_fold_oracle_test.rs
    - crates/cb-train/tests/fixtures/multi_permutation_fold/leaf_weights.json
    - crates/cb-train/tests/fixtures/multi_permutation_fold/config.json
    - crates/cb-train/tests/fixtures/multi_permutation_fold/model_pc1.json
    - crates/cb-train/tests/fixtures/multi_permutation_fold/model_pc2.json
    - crates/cb-train/tests/fixtures/multi_permutation_fold/model_pc4.json
    - crates/cb-oracle/generator/gen_multi_permutation_fold.py
  modified:
    - crates/cb-train/src/fold.rs

key-decisions:
  - "Pre-averaging GenRand fires at idx == learning_folds (the averaging-fold position) for ALL permutation_count; pc=1 byte-stream unchanged because idx == learning_folds == 1 coincides with the prior position"
  - "Upstream anchor is catboost's observable tree-0 leaf_weights (the AveragingFold partition counts), since the internal AveragingFold->LearnPermutation is not exposed by the Python API; a wrong advance count yields a wrong partition and fails the check"
  - "pc=2 is the MANDATORY upstream-locked anchor (partition [6,0,7,17] integer-exact vs catboost); pc=4 is a documented forward-compat divergence (cb-train [6,0,8,16] vs catboost [6,0,10,14]) needing C++ RNG instrumentation — committed dump kept for a future plan"

patterns-established:
  - "Pattern: anchor RNG-draw-order parity against observable upstream model output (leaf_weights) rather than re-deriving from the same RNG primitive the implementation uses (avoids the self-oracle blind spot WR-01 flags)"
  - "Pattern: when an exhaustive sweep proves no clean draw rule reproduces all configs, PIN the current value + RECORD the upstream delta rather than fabricating a match or weakening the gate"

requirements-completed: [ORD-01]

# Metrics
duration: ~70min
completed: 2026-06-14
---

# Phase 5 Plan 15: Multi-Permutation AveragingFold Draw-Order Parity (WR-01) Summary

**Moved the pre-averaging GenRand to the averaging-fold position (idx == learning_folds) for all permutation_count and proved the pc=2 AveragingFold partition matches catboost 1.2.10 integer-exact — while uncovering and documenting a genuine pc=4 (production-default) draw divergence that needs C++ instrumentation.**

## Performance

- **Duration:** ~70 min
- **Tasks:** 2/2
- **Files modified:** 8 (1 source, 1 test, 5 fixtures, 1 generator)

## Accomplishments

### Task 1 — corrected pre-averaging draw position (commit b69f5aa)

`create_folds` previously gated the single pre-averaging `gen_rand()` on a
`first_real_shuffle` flag that fired before the FIRST learning shuffle
(`idx == 1`). That is correct only at `permutation_count=1` (where
`learning_folds == 1`, so `idx == 1` IS the averaging fold). Replaced the flag
with a position guard `idx == learning_folds` so the pre-draw fires immediately
before the AVERAGING fold's shuffle for ALL `permutation_count`, matching the
upstream fold-creation order (`learn_context.cpp:524/575-578`, `fold.cpp:43-95`:
identity `Folds[0]` consumes zero draws, learning folds `1..learning_folds` each
one Fisher-Yates pass, averaging fold last). At `permutation_count=1` the
byte-stream is unchanged (the two positions coincide). The doc comment was
updated with the upstream citations and the corrected-position contract; the
numeric / Plain-no-CTR branch is untouched.

Verification: `averaging_fold_permutation_oracle_test` 3/3, cb-train lib 130/130,
`tensor_ctr_e2e_oracle_test` 3/3 (≤1e-5), `ordered_boost_e2e_oracle_test` 2/2
(≤1e-5), `grep first_real_shuffle` 0, `grep "idx == learning_folds"` 6, 0
warnings.

### Task 2 — catboost-anchored multi-permutation oracle (commit f22ad0b)

Added `gen_multi_permutation_fold.py` (RUN-ONCE/COMMIT, imports catboost from
`.venv`) which trains the tensor_ctr_e2e config family at pc=1,2,4 and commits
catboost 1.2.10's observable tree-0 `leaf_weights` (the AveragingFold partition
counts) plus the full `model_pc{1,2,4}.json`. Added
`multi_permutation_fold_oracle_test.rs` (4 tests, none ignored):

- `multi_permutation_count_two_averaging_matches_catboost_1_2_10` (PRIMARY,
  upstream-anchored, MANDATORY): the partition the `cb_train::create_folds` pc=2
  AveragingFold permutation produces over the production online-prefix CTR equals
  catboost's committed `[6,0,7,17]` integer-exact via `compare_permutation`. This
  is the WR-01 closure: pc=2 (`learning_folds==1`) shares pc=1's draw stream
  (05-14-validated bit-exact), and the OLD first-learning-shuffle guard would
  have diverged it.
- `multi_permutation_averaging_fires_after_all_learning_shuffles` (SECONDARY
  cross-check): `create_folds` == a self-derived TFastRng64 draw stream for pc=2
  and pc=4 (self-consistency, never the authority).
- `multi_permutation_learning_fold_zero_is_identity`: Folds[0] is identity for
  pc>=2.
- `multi_permutation_count_four_partition_pinned_and_upstream_delta_recorded`:
  pins the current pc=4 partition and records the upstream delta (see Deviations).

Verification: multi_permutation 4/4, lib 130/130, tensor_ctr_e2e 3/3,
ordered_boost_e2e 2/2, averaging_fold 3/3, 0 warnings;
`grep catboost` 28, `grep #[ignore]` 0, `grep compare_permutation` 5,
fixture dir non-empty.

## Deviations from Plan

### Auto-fixed Issues

None — Tasks 1 and 2 executed as written for the mandatory pc=2 anchor.

### Scoped Divergence (pc=4 — production default)

**[Rule 4-adjacent / scope boundary] pc=4 AveragingFold partition does NOT match
catboost 1.2.10 — documented, not fabricated.**

- **Found during:** Task 2, generating/validating the pc=4 anchor.
- **Issue:** catboost 1.2.10 pc=4 tree-0 leaf_weights are `[6,0,10,14]`; the
  cb-train pc=4 AveragingFold permutation (with the corrected
  `idx == learning_folds` guard) produces `[6,0,8,16]`.
- **Investigation:** An exhaustive draw-stream sweep (leading Fisher-Yates
  shuffles 0..7 × pre-averaging GenRands 0..7, plus per-fold and one-time
  pre-draw models) showed NO single clean rule reproduces BOTH the
  e2e-bit-exact pc=1/pc=2 partition `[6,0,7,17]` AND the pc=4 `[6,0,10,14]`:
    - pc=1/pc=2 require (lf-1 learning shuffles + 1 pre-averaging GenRand) — the
      05-14-validated, e2e-bit-exact rule the corrected guard implements.
    - pc=4 `[6,0,10,14]` matches a model with `lf` FULL Fisher-Yates shuffles and
      ZERO pre-draws — an inconsistent advance count vs the pc=1/2 rule.
  This points to additional RNG consumption upstream at pc>2 (catboost's
  multi-fold structure-fold selection / per-fold CTR-grid construction draw on
  the same persistent `Rand`, not captured by the fold-creation loop alone — and
  not recoverable from the lossy partition observable).
- **Resolution:** The mandatory pc=2 anchor is locked against upstream. pc=4
  bit-exact parity needs C++ instrumentation of catboost's per-fold RNG
  accounting (the same class of escalation the original 05-12 blocker flagged),
  OUT OF SCOPE for this RNG-draw-POSITION fix. The pc=4 catboost dump
  (`model_pc4.json` / `leaf_weights.json`) is committed so a future
  C++-instrumented gap-closure plan has the upstream anchor ready. The pc=4 test
  pins the current cb-train partition and records the upstream delta WITHOUT a
  hard equality — the honest state (pc=2 upstream-locked, pc=4 cross-checked +
  flagged), neither fabricated nor ignore-attributed.
- **Files:** crates/cb-train/tests/multi_permutation_fold_oracle_test.rs,
  crates/cb-train/tests/fixtures/multi_permutation_fold/
- **Commit:** f22ad0b

## Known Stubs

None. No placeholder data or empty-value stubs were introduced; the pc=4 case is
a fully-evaluated, documented divergence (not a stub).

## Threat Flags

None. T-05-15-01 (RNG draw-order tampering) is mitigated for the gated pc=1 and
the pc=2 anchor by the integer-exact oracle against committed upstream output; no
new attack surface (internal RNG draw order + committed test fixtures only).

## Self-Check: PASSED

- crates/cb-train/src/fold.rs — FOUND (modified)
- crates/cb-train/tests/multi_permutation_fold_oracle_test.rs — FOUND
- crates/cb-train/tests/fixtures/multi_permutation_fold/leaf_weights.json — FOUND
- crates/cb-train/tests/fixtures/multi_permutation_fold/model_pc{1,2,4}.json — FOUND
- crates/cb-oracle/generator/gen_multi_permutation_fold.py — FOUND
- Commit b69f5aa (Task 1) — FOUND
- Commit f22ad0b (Task 2) — FOUND
