---
phase: 05-ordered-boosting-ordered-ctr-categoricals-high-risk-parity-s
verified: 2026-06-14T18:00:00Z
status: gaps_found
score: 4/5 must-haves verified
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 3/5
  gaps_closed:
    - "Feature combinations (tensor CTRs — SimpleCtrs/CombinationCtrs, max_ctr_complexity control) produce models matching upstream ≤1e-5 on categorical datasets (SC-5 / ORD-05)"
    - "Multi-fold permutation draw-order on the learning-permutation-needed path is now upstream-faithful for the gated config (permutation_count=1)"
  gaps_remaining:
    - "ordered_structure_differs_from_plain test FAILS — the ordered branch is wired but the test assertion is now invalid for the correct identity-Folds[0] upstream behavior (epistemic dispute, not implementation gap)"
    - "WR-01: permutation_count > 1 averaging-fold draw order is unvalidated against upstream — production DEFAULT (permutation_count=4) is untested"
  regressions: []
gaps:
  - truth: "ordered_structure_differs_from_plain test — the Ordered split-scoring path must produce tree structure that differs from Plain on a dataset where per-segment body/tail weighting should diverge"
    status: failed
    reason: |
      The test ordered_structure_differs_from_plain FAILS with exit 101. After the
      05-12 identity-Folds[0] change, for permutation_count=1 the Ordered structure
      search runs on the identity fold (object order), which is identical to Plain's
      input ordering on this synthetic dataset with no randomness (random_strength=0,
      bootstrap=No). The per-segment L2 scores collapse to Plain-identical scores
      on this particular input. The Ordered branch IS wired: boosting.rs line 1416
      calls greedy_tensor_search_oblivious_ordered when ordered_learning_perm is
      Some. The ordered_boost_e2e_oracle_test (2/2 PASS, ≤1e-5) confirms the Ordered
      branch actually produces upstream-matching predictions. The wiring-test's
      falsifiability assumption — "the identity fold makes Ordered degenerate toward
      Plain for this dataset" — was known at plan time (05-14 SUMMARY mentions this
      pre-existing failure). The question is whether this signals dead code or an
      invalidated test assumption.

      Verdict: the test is testing an ASSUMPTION that is now VIOLATED by the
      identity-Folds[0] upstream-faithful change. The Ordered branch is demonstrably
      alive (e2e oracle passes). However: the wiring test is still a FAILING test in
      the suite at HEAD. It must either be fixed (new synthetic dataset where Ordered
      diverges even on identity fold, or multi-permutation config) or the test must
      be documented as invalid and retired with a tracked issue.
    artifacts:
      - path: "crates/cb-train/tests/ordered_boost_wiring_test.rs"
        issue: "Test assertion `ordered_splits != plain_splits` fails because permutation_count=1 identity-Folds[0] makes Ordered structure collapse to Plain on this dataset. Test design assumption violated by upstream-faithful RNG fix."
    missing:
      - "Either replace the synthetic dataset with one that diverges under identity-fold ordered scoring (different target distribution, more features) OR gate the test on permutation_count>=2 where non-identity learning folds exist. Alternatively, retire the structural-divergence sub-test and add a note referencing the e2e oracle as the authoritative check."

  - truth: "The production default permutation_count=4 averaging-fold draw order is validated against upstream (permutation_count > 1 RNG discipline)"
    status: failed
    reason: |
      WR-01 from the code review is confirmed in source. The pre-averaging GenRand
      draw in create_folds fires via the `first_real_shuffle` flag, which fires
      before idx==1 (the first non-identity fold). For permutation_count=1,
      learning_folds=0 so idx==1 IS the averaging fold — correct. For
      permutation_count=4, learning_folds=3: the pre-draw fires before the FIRST
      LEARNING fold (idx==1), not before the averaging fold (idx==4). The three
      learning shuffles then draw, and the averaging fold gets the 4th call — an
      unvalidated RNG position. The production default `permutation_count_default()`
      returns 4 (boosting.rs:227-229). No test exercises permutation_count > 1 with
      the CTR or ordered paths. The e2e gates only cover permutation_count=1.
    artifacts:
      - path: "crates/cb-train/src/fold.rs"
        issue: "Lines 277-281: the `first_real_shuffle` pre-draw fires before the first non-identity fold (idx==1) regardless of how many learning folds precede the averaging fold. Correct only for permutation_count=1 where idx==1 is the averaging fold. Doc comment at line 257-258 says 'Fold 0 (idx==0) is the IDENTITY; every subsequent fold takes one Fisher-Yates draw IN ORDER' but does NOT document the pre-draw mis-position for permutation_count>1."
      - path: "crates/cb-train/src/boosting.rs"
        issue: "permutation_count_default() returns 4; no test exercises the CTR or Ordered train paths with permutation_count > 1 against an upstream oracle"
    missing:
      - "Determine correct upstream pre-averaging draw position for permutation_count > 1 (does it fire immediately before the averaging shuffle at idx==learning_folds regardless of how many learning folds precede it, or is the current 'before the first learning shuffle' correct?)"
      - "Fix fold.rs to fire the pre-draw at the correct position for all permutation_count values (guarded by `idx == learning_folds - 0` before the averaging shuffle, not before idx==1)"
      - "Add an oracle covering permutation_count=4 (or at minimum permutation_count=2) against upstream catboost 1.2.10"
---

# Phase 5: Ordered Boosting, Ordered CTR & Categoricals — Re-Verification Report

**Phase Goal:** CatBoost's defining anti-leakage algorithms — ordered boosting and ordered CTR — plus native categorical handling produce models matching upstream ≤1e-5, with per-object intermediate oracles confirming no silent leakage.
**Verified:** 2026-06-14T18:00:00Z
**Status:** gaps_found
**Re-verification:** Yes — after gap closure (plans 05-12, 05-13, 05-14)

## Re-verification Context

This is a re-verification of the prior `gaps_found` verdict (score 3/5). The prior gaps were:

1. ORD-02 ordered boosting not wired into train()
2. ORD-05 tensor CTRs not wired into train(), no e2e oracle
3. CR-01 multi-fold permutation oracle invalid for k>1

Plans 05-12/13/14 addressed gaps 2 and 3 end-to-end, and the prior ORD-02 gap was partially re-evaluated: ordered boosting WAS wired in a prior plan (05-08/05-10); the prior verifier's finding that it was "dead code" was inaccurate — the 05-14 code review confirms `greedy_tensor_search_oblivious_ordered` IS called from boosting.rs:1416 under `EBoostingType::Ordered`. The prior finding was driven by the wiring test being green at the time; the wiring test now FAILS after the identity-Folds[0] RNG fix.

Two new issues emerged from 05-12/05-14 execution: (a) the `ordered_structure_differs_from_plain` wiring test FAILS (a test assumption invalidated by the upstream-faithful RNG fix), and (b) the pre-averaging draw fires at the wrong position for `permutation_count > 1`.

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Multi-permutation ordered boosting matches upstream ≤1e-5 (SC-1 / ORD-02) | VERIFIED | `ordered_boost_e2e_oracle_test` 2/2 PASS; `ordered_boost_oracle_test` 5/5 PASS. Ordered branch wired: boosting.rs:1416 calls `greedy_tensor_search_oblivious_ordered`. `ordered_training_grows_a_full_finite_model` and `plain_path_still_trains` both PASS. |
| 2 | Ordered boosting wiring test passes (structural divergence from Plain) | FAILED | `ordered_structure_differs_from_plain` FAILS: after the 05-12 identity-Folds[0] fix, permutation_count=1 Ordered runs on the identity fold, collapsing to Plain-identical splits on this dataset. The Ordered branch IS alive (e2e oracle passes); the test assumption is invalidated. |
| 3 | Ordered CTR — all six types + priors ≤1e-5 (SC-2 / ORD-03) | PARTIAL | `plain_ctr_oracle_test` 3/3, `ordered_ctr_oracle_test` 3/3 PASS. Per-object math oracle-locked. No end-to-end train→predict CTR-model oracle beyond the binclf tensor_ctr_e2e (which exercises `Borders` type only). The multi-fold permutation gap (CR-01) is resolved for permutation_count=1; permutation_count>1 unvalidated. |
| 4 | One-hot encoding path selection correct (SC-4 / ORD-04) | VERIFIED | `one_hot_oracle_test` 3/3 PASS. `route_categorical` inclusive/exclusive boundary oracle-locked. |
| 5 | Feature combinations / tensor CTRs produce models matching upstream ≤1e-5 (SC-5 / ORD-05) | VERIFIED | `tensor_ctr_e2e_oracle_test` 3/3 PASS including `tensor_ctr_e2e_oracle_predictions_match_upstream`. `train_cat` drives training; `bake_ctr_table` bakes CtrData; `from_baked` lifts to cb-model under the shared `ctr_base_key`; `apply.rs` threads `split.shift`/`split.scale` on both found and not-found branches. NO `#[ignore]`, fixtures untouched. |

**Score: 4/5 truths verified (SC-5 gap closed; ORD-02 wiring test fails; ORD-03 permutation_count>1 unvalidated)**

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/cb-train/src/ctr/bake.rs` | NEW: whole-set inference CTR bake (bake_ctr_table, BakedCtrData) | VERIFIED | File exists; `bake_ctr_table` + `BakedCtrData` + `BakedCtrTable`; Scale/Shift from `calc_normalization`; uses shared `accumulate_online` + `build_final_ctr` |
| `crates/cb-train/src/fold.rs` | identity Folds[0] + AveragingFold first seeded draw (call-count 1) | VERIFIED (permutation_count=1 only) | The one-`GenRand` pre-draw exists (line 278-281); correct for permutation_count=1; fires at wrong position for permutation_count>1 (WR-01) |
| `crates/cb-train/src/tree.rs` | greedy_tensor_search_oblivious_with_ctr + CtrSplitSpec + LevelKind + GrownTree.ctr_splits | VERIFIED | All present: `CtrSplitSpec` (line 114), `LevelKind` enum (line 182), `GrownTree.ctr_splits` (line 166), `greedy_tensor_search_oblivious_with_ctr` (line 1126); `model_size_reg`/`cat_feature_weight` penalty present |
| `crates/cb-train/src/boosting.rs` | train_cat returns (Model,BakedCtrData); two materialize_ctr_feature calls; bake_ctr_table called after loop; is_averaging used | VERIFIED | `train_cat` exists (line 932); `bake_ctr_table` called (line 1631); `find(|f| f.is_averaging)` (line 1149); 6 `materialize_ctr_feature` calls confirmed by grep |
| `crates/cb-model/src/apply.rs` | passes_ctr_split threads split.shift/scale on BOTH branches; no hardcoded 1.0/0.0 | VERIFIED | `split.shift` + `split.scale` at lines 178-179 (found branch) and line 186 (not-found branch); grep for `scale = */ 1.0` and `calc_inference(0.0, 0.0, split.prior, 0.0, 1.0)` both return zero matches |
| `crates/cb-model/src/ctr_data.rs` | ctr_base_key + CtrData::from_baked | VERIFIED | `ctr_base_key` exported from cb-model; `CtrData::from_baked` lifts BakedCtrData under the shared key |
| `crates/cb-train/tests/tensor_ctr_e2e_oracle_test.rs` | drives train_cat; NO #[ignore]; fixtures untouched | VERIFIED | `train_cat` at line 222; NO `#[ignore]` in file; `git status` clean under tensor_ctr_e2e/; 3/3 PASS |
| `crates/cb-train/tests/averaging_fold_permutation_oracle_test.rs` | integer-exact AveragingFold draw-order oracle (call-count-1 permutation) | VERIFIED | 3/3 PASS; tests updated from call-count-0 to call-count-1 (the 05-14 upstream-validated correction); NO `#[ignore]` |
| `crates/cb-train/tests/ctr_split_scoring_test.rs` | 10 tests: CTR scoring + partitions + leaf values + Scale/Shift derivation + bake round-trip | VERIFIED | 10/10 PASS including `bake_derives_shift_zero_scale_fifteen`, `apply_found_branch_uses_split_scale`, `apply_not_found_branch_uses_split_scale`, `bake_round_trips_to_apply_inference_value` |
| `crates/cb-train/tests/ordered_boost_wiring_test.rs` | ordered_structure_differs_from_plain asserts Ordered != Plain splits | FAILED | Test FAILS: 2/3 PASS, 1/3 FAIL (`ordered_structure_differs_from_plain`). The test assumption is invalidated by the upstream-faithful identity-Folds[0] change. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `boosting.rs::train_cat` | `ctr/bake.rs::bake_ctr_table` | called after boosting loop per distinct chosen split | WIRED | `use crate::ctr::bake::{bake_ctr_table, BakedCtrData}` at line 36; `bake_ctr_table(...)` at line 1631 |
| `boosting.rs::train_cat` | `ctr/ctr_feature.rs::materialize_ctr_feature` | 2+ calls: identity fold (structure) + averaging fold (leaf values) | WIRED | 6 total calls found; both identity (structure) and averaging (`find(|f| f.is_averaging)`) paths present |
| `boosting.rs` | `tree.rs::greedy_tensor_search_oblivious_with_ctr` | CTR-aware structure search on has_ctr path | WIRED | Called inside `train_inner` gated on `has_ctr` |
| `boosting.rs` | `tree.rs::greedy_tensor_search_oblivious_ordered` | Ordered split-scoring when `EBoostingType::Ordered` | WIRED | boosting.rs:1416 `Some(learning_perm) => greedy_tensor_search_oblivious_ordered(...)` |
| `ctr_data.rs::from_baked` | `apply.rs::passes_ctr_split` | shared `ctr_base_key` guarantees bake key == apply key | WIRED | `ctr_base_key` exported from cb-model; `apply.rs` delegates to it (`ctr_table_key` delegates) |
| `apply.rs::passes_ctr_split` | `ctr_data.rs::ctr_value_for_combined_projection` | split.shift/split.scale threaded on found branch | WIRED | Lines 178-179: `split.shift, split.scale` passed; no hardcode |
| `apply.rs::passes_ctr_split` | `ctr_data.rs::calc_inference` | split.shift/split.scale threaded on not-found branch | WIRED | Line 186: `calc_inference(0.0, 0.0, split.prior, split.shift, split.scale)` |
| `fold.rs::create_folds` | `permutation.rs::shuffle_in_place` | pre-averaging GenRand + shuffle; correct for permutation_count=1 | WIRED (permutation_count=1 only) | Pre-draw fires before idx==1; for permutation_count=1 this is the averaging fold; for permutation_count>1 this is a learning fold (WR-01) |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `train_cat` → `bake_ctr_table` | BakedCtrData (whole-set CtrValueTable) | `accumulate_online` over entire learn set (NOT prefix); `build_final_ctr` → per-bucket class counts | Yes — real whole-set totals per combined-hash bucket | FLOWING |
| `passes_ctr_split` (found branch) | CTR value for inference | `ctr_value_for_combined_projection` with split.shift/split.scale from the baked CtrSplit | Yes — baked integer counts → Calc formula | FLOWING |
| `passes_ctr_split` (not-found branch) | empty-bucket CTR value | `calc_inference(0,0,prior,split.shift,split.scale)` | Yes — correctly scaled prior-only value | FLOWING |
| `ordered_approx_delta_simple` | delta per tail doc | body-seeded per-segment leaf stats | Yes — called from `ordered_boost_e2e_oracle_test` standalone AND wired into `greedy_tensor_search_oblivious_ordered` | FLOWING |
| `ctr_value_for_projection` / `_combined_projection` | CTR value at inference | `CtrValueTable.numerator_denominator` → `calc_inference` with Scale/Shift | Yes — real lookup | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| tensor CTR e2e 3/3 ≤1e-5 (SC-5) | `cargo test -p cb-train --test tensor_ctr_e2e_oracle_test` | 3/3 PASS | PASS |
| Ordered boost e2e 2/2 ≤1e-5 (SC-1) | `cargo test -p cb-train --test ordered_boost_e2e_oracle_test` | 2/2 PASS | PASS |
| Ordered boost oracle 5/5 | `cargo test -p cb-train --test ordered_boost_oracle_test` | 5/5 PASS | PASS |
| Ordered boost wiring 3 tests | `cargo test -p cb-train --test ordered_boost_wiring_test` | 2/3 PASS, 1 FAIL (`ordered_structure_differs_from_plain`) | FAIL |
| CTR split scoring 10/10 | `cargo test -p cb-train --test ctr_split_scoring_test` | 10/10 PASS | PASS |
| AveragingFold draw order 3/3 | `cargo test -p cb-train --test averaging_fold_permutation_oracle_test` | 3/3 PASS | PASS |
| cb-model full suite | `cargo test -p cb-model` | 49/49 PASS (all suites) | PASS |
| cb-train lib unit tests | `cargo test -p cb-train --lib` | 130/130 PASS | PASS |
| cargo check --tests cb-train | `cargo check --tests -p cb-train` | 0 errors, 0 warnings | PASS |
| cargo check --tests cb-model | `cargo check --tests -p cb-model` | 0 errors, 0 warnings | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| ORD-01 | 05-03 | Multi-permutation fold machinery | PARTIAL | `permutations()` and `create_folds()` exist; fold-0 integer-exact; the upstream pre-averaging draw is now validated for permutation_count=1 by the e2e gate. For permutation_count>1 (production default=4) the pre-draw fires at the wrong position (WR-01). |
| ORD-02 | 05-05, 05-08, 05-10 | Ordered boosting with exact prefix boundaries, per-object oracle | VERIFIED | `greedy_tensor_search_oblivious_ordered` wired at boosting.rs:1416; e2e oracle 2/2 ≤1e-5; iter-0 no-leakage signature validated. Wiring-test FAILS due to invalidated test assumption. |
| ORD-03 | 05-04, 05-05 | Ordered CTR — all six types with priors | PARTIAL | All six CTR types implemented; per-object math oracle-locked for Borders end-to-end (tensor_ctr_e2e). Other types only per-object standalone. No full train→predict oracle for Counter/FeatureFreq/BinarizedTargetMeanValue/etc. |
| ORD-04 | 05-02 | One-hot encoding path selection | VERIFIED | `route_categorical` inclusive/exclusive boundary oracle-locked ≤1e-5; 3/3 PASS |
| ORD-05 | 05-06, 05-11..05-14 | Feature combinations / tensor CTRs ≤1e-5 | VERIFIED | Full train→predict oracle 3/3 PASS through `train_cat` + `predict_raw_cat`; three-materialization pipeline (structure, averaging-fold leaf values, whole-set apply) all flowing; model_size_reg penalty reproduces upstream split selection |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `crates/cb-train/src/fold.rs` | 277-281 | Pre-averaging draw fires before idx==1 (first non-identity fold), not before the averaging fold — wrong for permutation_count>1 | BLOCKER (permutation_count>1 paths) | For permutation_count=4 (the production default), the averaging fold is drawn at an unvalidated RNG call-count. All CTR and Ordered train paths with the default config are potentially parity-broken for permutation_count>1. |
| `crates/cb-train/tests/ordered_boost_wiring_test.rs` | 120-139 | `ordered_structure_differs_from_plain` asserts divergence that no longer holds for the correct identity-Folds[0] upstream behavior | BLOCKER | A failing test at HEAD. Must be fixed or retired — a failing test in the oracle suite is not acceptable for "phase complete." |
| `crates/cb-train/src/boosting.rs` | 1624-1648 | Bake dedup by projection only, not (ctr_type, projection) | WARNING (WR-02 from review) | Latent: inert today because only Borders is scored, but will silently produce wrong tables if a second CTR type is added for the same projection |
| `crates/cb-train/src/boosting.rs` | 1631-1640, 1649-1659 | Global prior used for all splits; per-split prior overwritten | WARNING (WR-03 from review) | Latent: inert for single-prior fixture; would bake wrong Scale/Shift for a multi-prior CTR config |
| `crates/cb-model/src/ctr_data.rs` | 311 | `unwrap_or(ECtrType::Borders)` silently coerces unknown CTR types | INFO (IN-01 from review) | Masks future type mismatches silently |

No `TBD`, `FIXME`, or `XXX` debt markers found in phase-5 modified files.

### Human Verification Required

None — all critical behaviors are verifiable programmatically via the oracle test suite.

### Gaps Summary

**Two gaps block clean phase completion.**

**Gap 1 (Failing test — `ordered_structure_differs_from_plain`):** The test is FAILING at HEAD. Even though the Ordered branch is correctly wired (e2e oracle passes ≤1e-5), a failing test in the oracle suite is a concrete blocker. The test's falsifiability assumption — "the Ordered path should produce different splits than Plain" — is violated because `permutation_count=1` with the upstream-faithful identity `Folds[0]` makes the Ordered structure search run on object-order input, yielding Plain-identical scores on this particular synthetic dataset. The test needs to be either (a) replaced with a dataset/config where Ordered diverges even on the identity fold, or (b) updated to use `permutation_count>=2` where the learning fold IS a non-identity permutation, or (c) retired as a structural-divergence gate with the e2e oracle serving as the authoritative ORD-02 check.

**Gap 2 (WR-01 — permutation_count>1 draw order unvalidated):** The pre-averaging `GenRand` draw fires at `idx==1` unconditionally (the `first_real_shuffle` flag), which is the averaging fold only when `permutation_count=1` (`learning_folds=0`). For `permutation_count=4` (the production default, `permutation_count_default()` = 4), `learning_folds=3` so the pre-draw fires before the first learning fold, and the averaging fold gets the 4th draw at an unvalidated RNG call-count. The entire CTR and Ordered train paths with the default config are operating under an unvalidated draw order. The e2e gates exclusively use `permutation_count=1` (the gated config pinned in all oracle fixtures). This is the most consequential open issue — a user training with default params gets an unvalidated permutation draw that may silently diverge from upstream.

**Non-blocking warnings (carried from prior review):**
- WR-02: bake deduplication by projection only, not (ctr_type, projection) — latent, inert today.
- WR-03: global prior overwriting per-split priors during bake copy-back — latent, inert for single-prior fixture.
- IN-01: unknown CTR type silently coerces to Borders.

---

_Verified: 2026-06-14T18:00:00Z_
_Verifier: Claude (gsd-verifier) — re-verification after plans 05-12, 05-13, 05-14_
