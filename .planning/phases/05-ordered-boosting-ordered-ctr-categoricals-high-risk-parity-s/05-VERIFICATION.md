---
phase: 05-ordered-boosting-ordered-ctr-categoricals-high-risk-parity-s
verified: 2026-06-15T00:00:00Z
status: gaps_found
score: 4/5 must-haves verified
overrides_applied: 0
escalation_decision:
  date: 2026-06-15
  by: developer
  decision: "ESCALATED AS BLOCKER. The verifier returned human_needed for the pc=4 (production-default permutation_count=4) AveragingFold divergence; the developer ruled it a blocker, not an accepted residual. SC-1 ('reproduces upstream permutations exactly') has no permutation_count carve-out and pc=4 is the production default, so Phase 5 stays open with status: gaps_found. Resolve via /gsd-plan-phase 5 --gaps."
re_verification:
  previous_status: gaps_found
  previous_score: 4/5
  gaps_closed:
    - "ordered_structure_differs_from_plain FAILING test — RETIRED in place (renamed ordered_branch_alive_structural_authority_is_e2e_oracle); all 3/3 wiring tests now PASS (commit 9a2c974)"
    - "WR-01 permutation_count>1 pre-averaging draw order unvalidated — RESOLVED for pc=2: create_folds now guards the pre-averaging gen_rand() at idx==learning_folds for ALL permutation_count (commit b69f5aa); pc=2 AveragingFold partition [6,0,7,17] locked integer-exact against committed catboost 1.2.10 output (commit f22ad0b)"
  gaps_remaining:
    - "pc=4 (production default permutation_count=4) AveragingFold partition [6,0,8,16] diverges from catboost 1.2.10 [6,0,10,14] — SC-1 says 'reproduces upstream permutations exactly'; pc=4 bit-exact parity needs C++ instrumentation of catboost's per-fold RNG accounting and is documented as out-of-scope for Phase 5"
  regressions: []
gaps:
  - truth: "Multi-permutation fold machinery seeded by TFastRng64 reproduces upstream permutations exactly at the production-default permutation_count=4 (SC-1 / ORD-01)"
    status: failed
    reason: "At pc=4 (permutation_count_default() = 4, the production default) the cb-train AveragingFold partition is [6,0,8,16] while catboost 1.2.10 produces [6,0,10,14]. SC-1 says 'reproduces upstream permutations exactly' with no permutation_count carve-out, and pc=4 is the default config, so this is a parity gap at the most-used setting. pc=1 and pc=2 ARE integer-exact (locked by multi_permutation_fold_oracle_test); only pc>=4 diverges. The e2e prediction oracles (ordered_boost_e2e, tensor_ctr_e2e) are locked at pc=1, so final-prediction parity at pc=4 is unproven. 05-15's exhaustive draw-stream sweep found no clean per-fold draw rule that reproduces BOTH the e2e-bit-exact pc=1/pc=2 stream AND pc=4; the executor concluded a pc=4 bit-exact fix needs C++ instrumentation of catboost's per-fold RNG accounting. NOTE: that approach is in tension with the Phase-3 P1/D-08 'Python-reachable floor, no C++ instrumentation' decision — the gap-closure plan must reconcile this (find a Python-reachable per-fold RNG oracle, or obtain an explicit user-approved deviation from D-08, before committing to C++ instrumentation)."
    artifacts:
      - path: "crates/cb-train/src/fold.rs"
        issue: "create_folds pre-averaging draw position (idx == learning_folds, line ~309) reproduces upstream for pc=1/pc=2 but not pc=4; the per-fold RNG draw accounting for permutation_count>=4 does not match catboost's AveragingFold partition"
      - path: "crates/cb-train/tests/multi_permutation_fold_oracle_test.rs"
        issue: "multi_permutation_count_four_partition_pinned_and_upstream_delta_recorded PINS cb-train [6,0,8,16] and records the upstream delta [6,0,10,14] but does NOT assert equality — the pc=4 case is documented, not closed"
      - path: "crates/cb-train/tests/fixtures/multi_permutation_fold/"
        issue: "catboost 1.2.10 pc=4 dump (leaf_weights.json [6,0,10,14], model_pc4.json) is committed and available as the oracle target for the closure plan"
    missing:
      - "Make create_folds reproduce catboost 1.2.10's pc=4 (and ideally general permutation_count>=4) AveragingFold permutation exactly — partition must equal [6,0,10,14] for the committed fixture"
      - "Upgrade multi_permutation_count_four_... from a pinned-delta test to an integer-exact assertion vs the committed catboost pc=4 leaf_weights"
      - "Add (or extend) an e2e prediction oracle at permutation_count=4 to prove final predictions match upstream <=1e-5 once the partition is exact"
      - "Reconcile the fix path with P1/D-08: prefer a Python-reachable per-fold RNG accounting oracle; only pursue C++ instrumentation with an explicit user-approved D-08 deviation"
---

# Phase 5: Ordered Boosting, Ordered CTR & Categoricals — Re-Verification Report

**Phase Goal:** CatBoost's defining anti-leakage algorithms — ordered boosting and ordered CTR — plus native categorical handling produce models matching upstream ≤1e-5, with per-object intermediate oracles confirming no silent leakage.
**Verified:** 2026-06-15T00:00:00Z
**Status:** gaps_found (developer escalated the pc=4 divergence — see escalation_decision)
**Re-verification:** Yes — after gap closure (plans 05-15 and 05-16), superseding the 2026-06-14T18:00:00Z gaps_found report

> **Developer decision (2026-06-15):** the verifier surfaced the pc=4 AveragingFold divergence as `human_needed`. The developer ESCALATED it as a blocker rather than accepting it as a tracked residual. Phase 5 remains open at `gaps_found`; the single blocking gap is recorded in the `gaps:` frontmatter for `/gsd-plan-phase 5 --gaps`. Both prior gaps (ORD-02 wiring test, ORD-01 pc=2 draw order) are confirmed CLOSED.

## Re-verification Context

This is a re-verification of the prior `gaps_found` verdict (score 4/5, 2026-06-14). The two prior gaps were:

1. `ordered_structure_differs_from_plain` FAILING test (GAP 1 / ORD-02) — closed by 05-16 (commit 9a2c974): retired the invalidated assertion in place, delegated ORD-02 structural authority to `ordered_boost_e2e_oracle_test`.
2. WR-01: permutation_count>1 pre-averaging draw order unvalidated (ORD-01) — closed for pc=2 by 05-15 (commits b69f5aa, f22ad0b): corrected guard position to `idx == learning_folds` for all permutation_count; pc=2 partition locked integer-exact against catboost 1.2.10.

One residual issue from 05-15 was NOT closed: pc=4 (the production default, `permutation_count_default() = 4`) AveragingFold partition diverges from catboost 1.2.10. This is documented, pinned, and explicitly deferred as needing C++ RNG instrumentation.

All claimed commits are VERIFIED at HEAD:
- `b69f5aa` — fix pre-averaging draw position
- `f22ad0b` — multi-permutation AveragingFold oracle
- `9a2c974` — retire ordered_structure_differs_from_plain

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Multi-permutation fold machinery seeded by TFastRng64 reproduces upstream permutations exactly (SC-1 / ORD-01) | PARTIAL | pc=1 and pc=2 are integer-exact vs catboost 1.2.10 (multi_permutation 4/4 PASS; PRIMARY test `multi_permutation_count_two_averaging_matches_catboost_1_2_10` locks pc=2 partition [6,0,7,17]). pc=4 (production default) diverges: cb-train [6,0,8,16] vs catboost [6,0,10,14] — documented and pinned in `multi_permutation_count_four_partition_pinned_and_upstream_delta_recorded`. SC-1 says "exactly"; pc=4 is not exact. |
| 2 | EBoostingType::Ordered trains with exact prefix boundaries, per-object intermediate oracle passes with no leakage (SC-2 / ORD-02) | VERIFIED | `ordered_boost_e2e_oracle_test` 2/2 PASS ≤1e-5 (AUTHORITATIVE). `ordered_boost_oracle_test` 5/5 PASS. `ordered_boost_wiring_test` 3/3 PASS (was 2/3 FAIL). Ordered branch wired: boosting.rs:1054-1057 `find(|f| !f.is_averaging)` = Folds[0] identity learning fold; greedy_tensor_search_oblivious_ordered called at boosting.rs:1416. |
| 3 | Ordered CTR — all six types (Borders, Buckets, BinarizedTargetMeanValue, FloatTargetMeanValue, Counter, FeatureFreq) with priors ≤1e-5 (SC-3 / ORD-03) | PARTIAL | `plain_ctr_oracle_test` 3/3 PASS; `ordered_ctr_oracle_test` 3/3 PASS. `tensor_ctr_e2e_oracle_test` 3/3 PASS locks the Borders type end-to-end. Other five CTR types verified per-object standalone but no full train→predict oracle for Counter/FeatureFreq/BinarizedTargetMeanValue/FloatTargetMeanValue/Buckets beyond the prefix-binclf math. |
| 4 | One-hot encoding path selection correct for low-cardinality categoricals (SC-4 / ORD-04) | VERIFIED | `one_hot_oracle_test` 3/3 PASS. `route_categorical` inclusive/exclusive boundary oracle-locked. |
| 5 | Feature combinations (tensor CTRs) produce models matching upstream ≤1e-5 on categorical datasets (SC-5 / ORD-05) | VERIFIED | `tensor_ctr_e2e_oracle_predictions_match_upstream` PASS (full multi-tree, NO #[ignore]); 3/3 tensor_ctr_e2e PASS. THREE materializations (identity structure [6,0,9,15], averaging-fold leaf values [6,0,7,17], whole-set apply [10,0,0,20]). model_size_reg cat-feature weight + AveragingFold pre-draw Rule-1 fixes validated. |

**Score: 4/5 (SC-1 partial due to pc=4 divergence; SC-3 partial but materially oracle-locked via tensor CTR e2e; SC-2/SC-4/SC-5 VERIFIED)**

### Deferred Items

| # | Item | Addressed In | Evidence |
|---|------|-------------|----------|
| — | pc=4 AveragingFold permutation divergence | Not in current roadmap | No future phase explicitly targets this; requires C++ instrumentation of catboost's per-fold RNG accounting. Documented in 05-15 SUMMARY (Deviations), pinned in multi_permutation_fold_oracle_test.rs. Tracked residual, not silently passed. |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/cb-train/src/fold.rs` | Pre-averaging draw at idx==learning_folds (not idx==1) for ALL permutation_count | VERIFIED | Line 309: `if idx == learning_folds { rng.gen_rand(); }`. `first_real_shuffle` flag absent (grep returns 0). Citied upstream: learn_context.cpp:524/575-578, fold.cpp:43-95. |
| `crates/cb-train/tests/multi_permutation_fold_oracle_test.rs` | 4 tests, none #[ignore], pc=2 upstream-anchored | VERIFIED | 4/4 PASS; zero `#[ignore]` attributes; `compare_permutation` present 5 times; pc=2 asserts catboost 1.2.10 `[6,0,7,17]` integer-exact; pc=4 pins cb-train `[6,0,8,16]` and records upstream delta [6,0,10,14] WITHOUT a hard equality |
| `crates/cb-train/tests/fixtures/multi_permutation_fold/` | Committed catboost 1.2.10 dump (leaf_weights, model_pc{1,2,4}.json, config.json) | VERIFIED | Files exist: leaf_weights.json (confirms pc=1/2=[6,0,7,17], pc=4=[6,0,10,14]), model_pc1.json, model_pc2.json, model_pc4.json, config.json |
| `crates/cb-train/tests/ordered_boost_wiring_test.rs` | 3/3 PASS; retired sub-test has in-file rationale + delegates to e2e oracle | VERIFIED | 3/3 PASS (was 2/3 FAIL). `ordered_branch_alive_structural_authority_is_e2e_oracle` contains in-file rationale with boosting.rs:~1054 citation and `ordered_boost_e2e_oracle_test` delegation. Aliveness gates `ordered_training_grows_a_full_finite_model` and `plain_path_still_trains` PRESERVED UNCHANGED. |
| `.planning/phases/05-.../05-DEFERRED.md` | Retire decision record for ordered_structure_differs_from_plain | VERIFIED | File exists; contains dated 2026-06-14 entry; explains identity-fold consumption; notes `find(|f| !f.is_averaging)` = Folds[0]; references `ordered_boost_e2e_oracle_test` 3 times; no `.planning/todos/` created. |
| `crates/cb-train/src/boosting.rs` | ordered_learning_perm via find(!is_averaging); train_cat + bake; permutation_count_default()=4 | VERIFIED | boosting.rs:1054-1057 `find(|f| !f.is_averaging)`; boosting.rs:1416 `greedy_tensor_search_oblivious_ordered`; `permutation_count_default()` = 4 at line 227. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `fold.rs::create_folds` | `permutation.rs::shuffle_in_place` | pre-averaging gen_rand() fires at `idx == learning_folds` | WIRED | Line 309: `if idx == learning_folds { rng.gen_rand(); }` — confirmed by grep "idx == learning_folds" returning 6 hits |
| `multi_permutation_fold_oracle_test.rs` | `fixtures/multi_permutation_fold/leaf_weights.json` | `upstream_leaf_weights(2)` → catboost [6,0,7,17] vs `averaging_partition` | WIRED | pc=2 hard equality present; upstream source committed and non-empty |
| `ordered_boost_wiring_test.rs` | `ordered_boost_e2e_oracle_test.rs` | in-file rationale comment delegates ORD-02 structural authority | WIRED | `ordered_boost_e2e_oracle_test` appears 7 times in wiring test; comment cites boosting.rs:~1054 |
| `boosting.rs::train_cat` | `ctr/bake.rs::bake_ctr_table` | called after boosting loop | WIRED | `use crate::ctr::bake::{bake_ctr_table, BakedCtrData}` at line 36; `bake_ctr_table(...)` at line 1631 |
| `boosting.rs` | `tree.rs::greedy_tensor_search_oblivious_ordered` | EBoostingType::Ordered branch at boosting.rs:1416 | WIRED | Confirmed at boosting.rs:1416 `Some(learning_perm) => greedy_tensor_search_oblivious_ordered(...)` |
| `apply.rs::passes_ctr_split` | `ctr_data.rs::ctr_value_for_combined_projection` | split.shift/split.scale threaded both branches | WIRED | Confirmed in prior verification, unchanged by 05-15/05-16 |

### Data-Flow Trace (Level 4)

All data-flow traces carried forward from prior verification (unchanged by 05-15/05-16 — test-only changes):

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `train_cat` → `bake_ctr_table` | BakedCtrData | `accumulate_online` over entire learn set + `build_final_ctr` | Yes | FLOWING |
| `passes_ctr_split` (found branch) | CTR value | `ctr_value_for_combined_projection` with real split.shift/scale | Yes | FLOWING |
| `passes_ctr_split` (not-found branch) | CTR value | `calc_inference(0,0,prior,split.shift,split.scale)` | Yes | FLOWING |
| `create_folds` AveragingFold permutation | permutation array | persistent TFastRng64 at idx==learning_folds position | Yes — locked pc=2 integer-exact vs catboost | FLOWING (pc=1/pc=2); DIVERGENT (pc=4) |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| ordered_boost_wiring_test 3/3 (GAP 1 closure) | `cargo test -p cb-train --test ordered_boost_wiring_test` | 3/3 PASS (0 FAIL) | PASS |
| ordered_boost_e2e_oracle_test 2/2 ≤1e-5 (SC-2) | `cargo test -p cb-train --test ordered_boost_e2e_oracle_test` | 2/2 PASS | PASS |
| ordered_boost_oracle_test 5/5 | `cargo test -p cb-train --test ordered_boost_oracle_test` | 5/5 PASS | PASS |
| multi_permutation_fold_oracle_test 4/4 (SC-1) | `cargo test -p cb-train --test multi_permutation_fold_oracle_test` | 4/4 PASS (pc=2 upstream-exact; pc=4 pinned+divergence recorded) | PASS |
| tensor_ctr_e2e_oracle_test 3/3 ≤1e-5 (SC-5) | `cargo test -p cb-train --test tensor_ctr_e2e_oracle_test` | 3/3 PASS | PASS |
| averaging_fold_permutation_oracle_test 3/3 | `cargo test -p cb-train --test averaging_fold_permutation_oracle_test` | 3/3 PASS | PASS |
| cb-train lib unit tests | `cargo test -p cb-train --lib` | 130 passed; 0 failed; 0 ignored | PASS |
| cb-model full suite | `cargo test -p cb-model` | 3 passed; 0 failed | PASS |
| Full cb-train suite (all integration tests) | `cargo test -p cb-train` | 0 FAILED; ignored tests are pre-existing Phase 3 deferred items (Bayesian, overfit e2e, random_strength) | PASS |
| cargo check --tests cb-train | `cargo check --tests -p cb-train` | 0 errors, 0 warnings | PASS |

**Confirmed: NO failing test exists at HEAD in cb-train or cb-model.**

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| ORD-01 | 05-03, 05-15 | Multi-permutation fold machinery | PARTIAL | pc=1/pc=2 integer-exact vs catboost 1.2.10; pc=4 (production default) diverges with documented residual needing C++ instrumentation |
| ORD-02 | 05-05, 05-08, 05-10, 05-16 | Ordered boosting with exact prefix boundaries, per-object oracle | VERIFIED | e2e oracle 2/2 ≤1e-5; wiring test 3/3 PASS; branch wired at boosting.rs:1416; retire decision in 05-DEFERRED.md |
| ORD-03 | 05-04, 05-05 | Ordered CTR — all six types with priors | PARTIAL | Math oracle-locked for Borders end-to-end (tensor_ctr_e2e); other types per-object standalone only |
| ORD-04 | 05-02 | One-hot encoding path selection | VERIFIED | `one_hot_oracle_test` 3/3 PASS; boundary oracle-locked |
| ORD-05 | 05-06, 05-11..05-14 | Feature combinations / tensor CTRs ≤1e-5 | VERIFIED | `tensor_ctr_e2e_oracle_predictions_match_upstream` PASS; three-materialization pipeline flowing; no #[ignore]; fixtures untouched |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `crates/cb-train/src/boosting.rs` | 1624-1648 | Bake dedup by projection only, not (ctr_type, projection) | WARNING (WR-02) | Latent; inert today since only Borders is scored; latent bug if second CTR type added for same projection |
| `crates/cb-train/src/boosting.rs` | 1631-1640, 1649-1659 | Global prior used for all splits; per-split prior overwritten | WARNING (WR-03) | Latent; inert for single-prior fixture |
| `crates/cb-model/src/ctr_data.rs` | ~311 | `unwrap_or(ECtrType::Borders)` silently coerces unknown CTR types | INFO (IN-01) | Masks future type mismatches silently |

No `TBD`, `FIXME`, or `XXX` debt markers found in phase-5 modified files (fold.rs, boosting.rs, ordered_boost_wiring_test.rs, multi_permutation_fold_oracle_test.rs). The WARNING items (WR-02, WR-03) are carried from the prior review and remain latent/inert.

### Human Verification Required

#### 1. Accept or Escalate: pc=4 AveragingFold partition divergence vs SC-1

**Test:** Review the documented pc=4 (production default, `permutation_count_default() = 4`) AveragingFold divergence: cb-train produces `[6,0,8,16]`; catboost 1.2.10 expects `[6,0,10,14]`. The divergence is pinned in `multi_permutation_count_four_partition_pinned_and_upstream_delta_recorded` (4/4 tests pass, but the test deliberately does NOT assert equality to upstream, only records the delta). Phase 5 plans (05-15 Plan, 05-15 SUMMARY Deviations) explicitly state pc=4 needs C++ instrumentation of catboost's per-fold RNG accounting and is out-of-scope for Phase 5.

**Expected:** Developer explicitly accepts the pc=4 divergence as a tracked residual that does not block Phase 5 closure — the SC-1 wording "reproduces upstream permutations exactly" is satisfied for the gated configs (pc=1, pc=2) and the production-default (pc=4) gap is a known, committed, tracked forward-compat issue. OR: developer escalates this as a blocker requiring a Phase 5 gap-closure plan before Phase 6 begins.

**Why human:** SC-1 says "exactly" without a permutation_count carve-out. The production default is pc=4 and it diverges from upstream. Whether the oracle scope was always implicitly gated on pc=1/pc=2 (per the phase plan documents) or whether the phase goal requires pc=4 parity is a judgment call that depends on development intent, not code reading. The technical state is fully documented and honest — the ambiguity is in acceptance criteria interpretation.

### Gaps Summary

**GAP 1 (CLOSED — ordered_structure_differs_from_plain):** The prior failing test at HEAD has been retired in place with a documented rationale and in-file citation (commit 9a2c974). `ordered_boost_wiring_test` now passes 3/3. The retire decision is recorded in `05-DEFERRED.md`. ORD-02 structural authority rests on `ordered_boost_e2e_oracle_test` (2/2 PASS ≤1e-5 vs catboost 1.2.10). **This gap is CLOSED.**

**GAP 2 (CLOSED for pc=2 — WR-01 permutation_count>1):** The pre-averaging draw position is corrected to `idx == learning_folds` for all permutation_count (commit b69f5aa). pc=2 AveragingFold partition is locked integer-exact against committed catboost 1.2.10 output [6,0,7,17] (commit f22ad0b, test PRIMARY: `multi_permutation_count_two_averaging_matches_catboost_1_2_10`). **WR-01 is CLOSED for pc=2.**

**RESIDUAL — pc=4 AveragingFold divergence (not new, documented):** The production default `permutation_count=4` AveragingFold partition diverges from catboost 1.2.10 ([6,0,8,16] vs [6,0,10,14]). Exhaustive draw-stream enumeration in the 05-15 SUMMARY shows no clean per-fold rule reproduces both pc=1/pc=2 and pc=4 — pc=4 bit-exact parity needs C++ instrumentation of catboost's per-fold RNG accounting. The divergence is committed, pinned, and tracked. This is the only item requiring human acceptance.

**Non-blocking warnings (carried):**
- WR-02: bake deduplication by projection only — latent, inert today.
- WR-03: global prior overwriting per-split priors during bake — latent, inert for single-prior fixture.
- IN-01: unknown CTR type silently coerces to Borders.

---

_Verified: 2026-06-15T00:00:00Z_
_Verifier: Claude (gsd-verifier) — re-verification after plans 05-15 (WR-01/ORD-01) and 05-16 (ORD-02)_
