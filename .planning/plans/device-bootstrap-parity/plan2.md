---
plan: 2
task_id: TASK-02
phase: device-bootstrap-parity
status: pending
order: 2
wave: B
hardware: local ROCm gfx1151 (required)
depends_on: [TASK-01]
specifications: [WR01-S12]
---

# Task 2: Characterise and lock the device/CPU split tie-break order

## Objective

After this task the repository can answer, in one test run, *why* a device tree and
a CPU tree ever choose different splits: rule difference (a bug this phase must fix)
or fixed-point histogram resolution (a designed property this phase must live with).
TASK-01 measured 3/20 mismatching trees at 20000×16 while predictions still agreed
to 4.4e-16 — that is benign only until a tie is broken differently on a
*non-degenerate* split, which bootstrap sampling makes more likely.

## Specification references

- `WR01-S12` — device/CPU split selection has one documented, locked tie-break
  order. Primary failure reason: a gain tie resolved differently on a
  non-degenerate split produces a genuinely different model.

## Context and evidence

- **Device rule.** Per-block winners are folded host-side with
  `let take = gain > best_gain || (gain == best_gain && cand < best_c);`
  over the flat candidate index `cand`, decomposed as
  `feature = cand / n_bins`, `bin = cand % n_bins`
  `[VERIFIED: crates/cb-backend/src/gpu_runtime/mod.rs:3069-3078]`. The in-kernel
  per-block argmin is documented to use "the strict-`>`/lowest-index tie-break"
  `[VERIFIED: mod.rs:2974-2981]`. Padded/trailing borders are excluded both
  in-kernel and by the host belt (`cand % n_bins >= n_bins_used - 1`,
  `mod.rs:3064-3068`).
- **CPU rule.** Within a feature: `if instance > best_instance` over borders in
  ascending index order → lowest border wins a tie
  `[VERIFIED: crates/cb-train/src/tree.rs:1170-1180]`. Across features:
  `if instance > best_gain` over features in ascending order → lowest feature wins a
  tie `[VERIFIED: tree.rs:1189-1204]`. Border-less features draw one
  `random_score_instance(NEG_INFINITY, …)` and can never win
  `[VERIFIED: tree.rs:1206-1216]`.
- **Therefore the rules agree** (both are lexicographic-lowest `(feature, bin)`).
  This is the claim the test must actually prove rather than assume.
- **The numeric mechanism.** The device histogram is fixed-point: every per-object
  contribution is `round(v · 2^30) → i64 → u64` before an integer atomic add
  `[VERIFIED: kernels.rs:3889-3893, called at :4025-4026 and :4210-4211; scale
  REDUCE_FIXEDPOINT_SCALE_F64 = 2^30 at kernels.rs:2335]`. The CPU histogram sums
  exact `f64` (`build_bucket_histogram` / `derive_feature_level_hist` feeding
  `scan_and_score_borders_into`, `tree.rs:1106-1128`). Per-object quantization error
  is bounded by `2^-31 ≈ 4.66e-10`, so a bin sum over `m` objects differs from the
  CPU's by at most `m · 2^-31`.
- Score functions: the fixture uses `EScoreFunction::L2` (`cb_compute::l2_split_score`,
  `score.rs:49`) on both sides — the same formula, different histogram inputs.

## Files

- Create: `crates/cb-train/tests/device_split_tiebreak_test.rs`
- Modify: `crates/cb-backend/src/gpu_runtime/mod.rs` — doc comment ONLY, on
  `score_partition_over_binsums` (around `mod.rs:3040-3045`), recording the locked
  tie-break rule and its CPU counterpart with file:line, so the invariant is
  discoverable from the code.
- Modify: `crates/cb-train/src/tree.rs` — doc comment ONLY, on
  `select_level_perturbed` step (3) (`tree.rs:1184-1188`), pointing back at the
  device rule.

## TDD sequence

### 1. Red

Three tests, each with ONE principal failure reason:

- `tiebreak_rule_agrees_on_exact_ties` — build a fixture engineered so that two
  features have bit-identical best gains AND one feature has two borders with
  bit-identical gains (easiest construction: a duplicated feature column with the
  same borders, plus a target that makes two borders of one feature symmetric).
  Grow one depth-1 tree on the device and on `CpuRefRuntime`. Assert BOTH choose the
  same `(feature, border)` and that it is the lexicographic-lowest tied candidate.
  Expected initial failure mode if the rules ever diverge: the two chosen features
  differ.
- `split_mismatches_are_below_the_fixedpoint_floor` — run the 20000×16 d6 ×20 shape
  from TASK-01. For every tree whose split sequences differ, recompute BOTH
  candidates' CPU gains at the diverging level (reusing the CPU grower's public
  scoring path, or an inline reference over the same `(der1, weight, leaf_of)`), and
  assert `|gain(device_choice) − gain(cpu_choice)| < n_objects_in_partition * 2f64.powi(-31) * SAFETY`,
  with `SAFETY = 8.0` documented. A mismatch whose gain gap exceeds the floor is a
  hard failure. Expected initial state: passes with a large margin (record the
  actual ratio).
- `fixedpoint_floor_assertion_is_not_vacuous` — feed the same comparator an
  artificially inflated pair of gains and assert it rejects. This guards against the
  floor being so loose that test 2 can never fail.

- Run: `cargo test -p cb-train --no-default-features --features rocm --test device_split_tiebreak_test -- --nocapture --test-threads 1`

### 2. Green

- Implement the reference gain recomputation using existing public building blocks
  (`cb_compute::l2_split_score` over `LeafStats` assembled from the CPU
  `leaf_of` + `der1` + `weight`), not a new scoring formula.
- If `tiebreak_rule_agrees_on_exact_ties` fails, that is a REAL defect and the
  correct fix is in the comparator, not the test: reconcile
  `mod.rs:3069` with `tree.rs:1175`/`:1201`. Record the reconciliation in the task's
  notes and re-run TASK-01's oracle to prove no regression.
- Run: the same command.

### 3. Refactor

- Move the shared fixture builders (`fixture`, `oblivious_params`, `CpuRefRuntime`)
  used by both `device_oblivious_parity_test.rs` and this file into a shared
  `crates/cb-train/tests/common/device_fixture.rs` module included by both via
  `mod common;` — Rust integration tests do not share crates, so use the
  `tests/common/mod.rs` convention. Confirm both test targets still build and report
  the same numbers.
- Run: both device test targets.

### 4. Verify

- Run: `cargo test -p cb-train --no-default-features --features rocm --test device_split_tiebreak_test -- --nocapture`
- Run: `cargo test -p cb-train --no-default-features --features rocm --test device_oblivious_parity_test -- --nocapture` (no regression from the refactor)
- Run: `cargo test -p cb-train --test device_split_tiebreak_test` (cpu SKIP arm)
- Run: `cargo clippy -p cb-train --tests --no-deps` and
  `cargo clippy -p cb-backend --no-default-features --features rocm --lib --no-deps`
- Confirm: the doc comments on `score_partition_over_binsums` and
  `select_level_perturbed` state the same rule, each citing the other's file:line.

## Implementation steps

1. Write `tiebreak_rule_agrees_on_exact_ties` first — it is the cheapest and it is
   the one that can invalidate the whole phase.
2. Add the fixed-point floor comparator as a small private helper with its own
   doc comment deriving `m · 2^-31`.
3. Add the mismatch-attribution test over the 20000×16 shape.
4. Add the non-vacuity test.
5. Add the two production doc comments (no logic change).
6. Record in `progress.md`: the number of mismatching trees, the largest observed
   gain gap, and the floor it was compared against.

## Completion criteria

- [ ] The exact-tie test demonstrably fails if either comparator's tie direction is
      inverted (verify by temporarily flipping `cand < best_c` to `cand > best_c`).
- [ ] Every observed split mismatch is attributed below the fixed-point floor.
- [ ] The non-vacuity test passes.
- [ ] Both production doc comments record the locked rule with cross-references.
- [ ] `device_oblivious_parity_test` still green after the shared-fixture refactor.

## Risks and guardrails

- **A mismatch above the floor.** That would mean a real rule/formula divergence and
  is a STOP condition for the phase — do not proceed to TASK-06 (gate relaxation)
  until it is explained. Record it in `progress.md` as a blocker.
- **Constructing an exact tie is harder than it looks** on the device side because
  the fixed-point encode may break a tie the CPU keeps. Guard: construct the tie
  from *integer-valued* der/weight contributions (e.g. targets in `{-1, 0, 1}` and
  unit weights), which are exactly representable at scale `2^30`, so both histograms
  are bit-identical and the tie survives to the comparator.
- **Test-only shared module** — `tests/common/mod.rs` must not be picked up as its
  own test target; use the directory form (`tests/common/mod.rs`), not
  `tests/common.rs`.
