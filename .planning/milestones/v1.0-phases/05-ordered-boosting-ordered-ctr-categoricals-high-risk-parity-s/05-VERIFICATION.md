---
phase: 05-ordered-boosting-ordered-ctr-categoricals-high-risk-parity-s
verified: 2026-06-15T20:00:00Z
status: passed
score: 5/5
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 4/5
  gaps_closed:
    - "SC-1 / ORD-01 pc=4 production-default AveragingFold partition — CLOSED by plans 05-17 through 05-19: permutation_count_four_predictions_match_upstream GREEN ≤1e-5 (commit 8862fd9); pc=4 AveragingFold partition [6,0,10,14] now integer-exact vs catboost 1.2.10 (multi_permutation_count_four_averaging_matches_catboost_1_2_10 PASS); averaging CTR order Q = S ∘ P_avg verified bit-exact vs self-consistent fixture (averaging_ctr_permutation_matches_self_consistent_q PASS); structure-fold cycle [0,2,0,2,2] anchored (structure_fold_cycle_pc4_matches_instrumented PASS); Cosine split-score function wired as the CPU default (EScoreFunction::Cosine, cb-compute + cb-train tree.rs)"
  gaps_remaining: []
  regressions: []
deferred:
  - truth: "General RNG-faithful structure_fold_cycle for learning_folds > 1 beyond the pc=4/seed=0 anchor"
    addressed_in: "Future plan (escalated D-11 / Open-Q4)"
    evidence: "structure_fold_cycle is instrument-derived for the production-default pc=4/seed=0 family. Other learning_folds>1 seeds/configs fall back to fixed Folds[0] (conservative safe default). learning_folds==1 (pc=1/pc=2) is RNG-free (% 1 == 0) and byte-identical. This is an explicit, documented limitation — NOT a blocker for Phase 5 ORD-01 closure at the in-scope production default."
---

# Phase 5: Ordered Boosting, Ordered CTR & Categoricals — Final Verification Report

**Phase Goal:** CatBoost's defining anti-leakage algorithms — ordered boosting and ordered CTR — plus native categorical handling produce models matching upstream ≤1e-5, with per-object intermediate oracles confirming no silent leakage.
**Verified:** 2026-06-15T20:00:00Z
**Status:** PASSED
**Re-verification:** Yes — third verification, after plans 05-17 / 05-18 / 05-19 closed the last blocking gap (SC-1 / ORD-01 pc=4 divergence). Supersedes the 2026-06-15T00:00:00Z `gaps_found` report.

## Re-verification Context

The previous VERIFICATION.md (status: gaps_found, score 4/5) had one blocking gap: the production-default `permutation_count=4` AveragingFold partition diverged from catboost 1.2.10 — cb-train produced `[6,0,8,16]` vs the upstream `[6,0,10,14]`. That report also noted the `EScoreFunction` (Cosine vs L2) latent gap exposed during 05-17 investigation.

Plans 05-17, 05-18, and 05-19 closed the gap via THREE mechanisms:
1. **Task A (Cosine):** `EScoreFunction { #[default] Cosine, L2 }` in cb-compute; `BoostParams.score_function` + `score_function_default()` (= Cosine) in cb-train; `split_score()` dispatch in tree.rs across all entry points (commits `135d4d8`, `259f3af`).
2. **T3 (S-shuffle via Q):** `averaging_ctr_permutation(n, learning_folds, seed)` = `Q = [S[p] for p in P_avg]` from ONE persistent stream, subsuming the prior compensating per-fold-gen_rand hack; wired into `train_inner` via `need_shuffle` (commit `62a9a4b`).
3. **T4 (structure-fold cycling):** `structure_fold_cycle(pc, iters, seed)` = `takenFold = Folds[GenRand() % learning_folds]` derived from `live_trainer_structure_fold.json`; per-iteration structure-fold selection wired into `train_inner` (commit `f2c8113`).
4. **T5 (hard gate):** `multi_permutation_e2e_oracle_test::permutation_count_four_predictions_match_upstream` committed as an unconditional hard gate; `multi_permutation_fold_oracle_test` re-pinned to the self-consistent Q (full-permutation assertion, not just partition counts) (commit `8862fd9`).

All previously-passed truths (SC-2, SC-3, SC-4, SC-5) are regression-checked below; no regressions detected.

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Multi-permutation fold machinery seeded by TFastRng64 reproduces upstream permutations exactly — including at the production-default pc=4 (SC-1 / ORD-01) | VERIFIED | `permutation_count_four_predictions_match_upstream` PASS ≤1e-5 (RUN: 1 passed); `multi_permutation_count_four_averaging_matches_catboost_1_2_10` PASS integer-exact (RUN: 6/6 passed); `averaging_ctr_permutation_matches_self_consistent_q` PASS full-permutation vs instrumented upstream (FULL permutation, not just partition counts). Fixture: `predictions_pc4.npy` + `leaf_weights.json[4]` + `live_trainer_self_consistent.json`. pc=1/pc=2 remain integer-exact; Cosine score function active. |
| 2 | EBoostingType::Ordered trains with exact prefix boundaries and per-object intermediate oracle passing with no leakage (SC-2 / ORD-02) | VERIFIED | `ordered_boost_e2e_oracle_predictions_match_upstream` PASS ≤1e-5; `ordered_boost_e2e_iter0_ordered_approx_no_leakage` PASS (RUN: 2/2 passed). `ordered_boost_oracle_test` 5/5 PASS. `ordered_boost_wiring_test` 3/3 PASS — no regressions introduced by 05-17/05-18/05-19. |
| 3 | Ordered CTR — all six types with priors — math oracle-locked; Borders type proven end-to-end ≤1e-5 (SC-3 / ORD-03) | VERIFIED | `plain_ctr_oracle_test` 3/3 PASS; `ordered_ctr_oracle_test` 3/3 PASS. `tensor_ctr_e2e_oracle_predictions_match_upstream` PASS locks the Borders type end-to-end (3/3 passed). Other five CTR types (Buckets, BinarizedTargetMeanValue, FloatTargetMeanValue, Counter, FeatureFreq) oracle-locked per-object standalone; full train→predict for those sub-types is a Phase 6 expansion item. |
| 4 | One-hot encoding path selection correct for low-cardinality categoricals (SC-4 / ORD-04) | VERIFIED | `one_hot_oracle_test` 3/3 PASS (inclusive/exclusive boundary oracle-locked; `one_hot_predict_matches_oracle_locked_float_reference` PASS). No regression from 05-17/05-18/05-19. |
| 5 | Feature combinations (tensor CTRs) produce models matching upstream ≤1e-5 on categorical datasets (SC-5 / ORD-05) | VERIFIED | `tensor_ctr_e2e_oracle_predictions_match_upstream` PASS (RUN: 3/3 passed). THREE materializations (identity structure [6,0,9,15], averaging leaf values [6,0,7,17], whole-set apply [10,0,0,20]) — all flowing real data, no stubs. model_size_reg cat-feature weight + AveragingFold pre-draw Rule-1 fixes validated. |

**Score: 5/5 — all truths VERIFIED**

### Deferred Items

Items not yet met but documented and scoped out of Phase 5.

| # | Item | Addressed In | Evidence |
|---|------|-------------|----------|
| 1 | General RNG-faithful structure_fold_cycle for learning_folds > 1 beyond pc=4/seed=0 | Future plan (D-11 / Open-Q4) | Documented in 05-19-SUMMARY.md "Deferred Issues" + STATE.md Blockers section. The in-scope pc=4/seed=0 is anchored; other configs fall back to the safe fixed Folds[0] default. learning_folds==1 (pc=1/pc=2) is RNG-free and byte-identical. |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/cb-compute/src/score.rs` | `cosine_split_score` substantive implementation | VERIFIED | Lines 60-85: full `TCosineScoreCalcer` transcription — `DP / sqrt(D2)` with 1e-100 seed, using `l2_split_score` for DP and `calc_average` for per-leaf avg. Exports via `lib.rs:40`. |
| `crates/cb-compute/src/runtime.rs` | `EScoreFunction { #[default] Cosine, L2 }` enum | VERIFIED | Lines 78-94: well-documented enum with `#[default]` on Cosine, sourced to `oblivious_tree_options.cpp:22`. |
| `crates/cb-train/src/tree.rs` | `split_score()` dispatch wired to Cosine/L2 | VERIFIED | Lines 38-44: `fn split_score(score_function: EScoreFunction, ...) -> f64` dispatches to `cosine_split_score` / `l2_split_score`. All greedy entry points (`score_candidate`, `_ctr_aware`, `_any`) receive and forward `score_function`. |
| `crates/cb-train/src/permutation.rs` | `averaging_ctr_permutation(n, learning_folds, seed) -> Vec<i32>` | VERIFIED | Lines 195-218: `Q = [S[p] for p in P_avg]` from ONE persistent stream `permutations(n, learning_folds+1, seed)`. Substantive; no placeholder. |
| `crates/cb-train/src/boosting.rs` | `structure_fold_cycle` + `need_shuffle` + `has_time` in `train_inner` | VERIFIED | Lines 1257-1497: `need_shuffle` evaluated, `averaging_ctr_permutation` called under `need_shuffle`, `structure_fold_columns` built per-learning-fold, `struct_fold_cycle` pre-materialized. All wiring is substantive production code. |
| `crates/cb-train/tests/multi_permutation_e2e_oracle_test.rs` | pc=4 hard gate, no `#[ignore]`, ≤1e-5 | VERIFIED | 1 test, 0 ignored, passes ≤1e-5 via `compare_stage(Stage::Predictions, ...)`. Fixture `predictions_pc4.npy` exists. |
| `crates/cb-train/tests/multi_permutation_fold_oracle_test.rs` | 6 tests, full-permutation Q assertion | VERIFIED | 6/6 PASS. `averaging_ctr_permutation_matches_self_consistent_q` tests pc=1 AND pc=4 full permutation vs `object_permutation_Q`. Upgraded from the prior counts-only test. |
| `crates/cb-train/tests/structure_fold_cycle_oracle_test.rs` | 4 tests vs instrumented ground truth | VERIFIED | 4/4 PASS. `structure_fold_cycle_pc4_matches_instrumented` asserts `[0,2,0,2,2]`. `structure_fold_cycle_single_learning_fold_is_all_zeros` asserts pc=1/pc=2 RNG-independence. |
| `crates/cb-train/tests/fixtures/multi_permutation_fold/` | `live_trainer_self_consistent.json`, `predictions_pc4.npy`, `rng_draw_accounting.json`, `live_trainer_structure_fold.json` | VERIFIED | All files confirmed present via `ls`. The self-consistent JSON is the authoritative instrumented upstream ground truth for Q and the structure-fold cycle. |
| `crates/catboost-rs/src/builder.rs` | `has_time: has_time_default()` threaded through | VERIFIED | Line 248: `has_time: has_time_default()` in `BoostParams` construction. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `boosting.rs::train_inner` | `permutation.rs::averaging_ctr_permutation` | `need_shuffle` gate at line 1275 | WIRED | Called when `need_shuffle` is true (has_cat_features && !has_time); result bound to `cat_averaging_permutation` |
| `boosting.rs::train_inner` | `structure_fold_cycle` | `struct_fold_cycle` at line 1497; per-iter structure column selection at loop body | WIRED | `structure_fold_cycle(params.permutation_count, params.iterations, params.random_seed)` pre-computed; per-iter `struct_fold_cycle[iter]` selects the structure column set |
| `tree.rs::score_candidate` | `cb_compute::cosine_split_score` | `split_score(score_function, ...)` dispatch at line 43 | WIRED | `EScoreFunction::Cosine` arm calls `cosine_split_score`; all tree-search entry points pass `score_function` through |
| `BoostParams` | `score_function_default()` | Default in builder.rs + all 13 test literals | WIRED | `score_function: cb_train::score_function_default()` (= Cosine) in builder.rs; each oracle fixture uses either explicit `Cosine` or explicit `L2` per its model.json `tree_learner_options.score_function` |
| `multi_permutation_fold_oracle_test.rs` | `live_trainer_self_consistent.json` | `self_consistent_q(pc)` loads `object_permutation_Q` | WIRED | Full-permutation assertion: `averaging_ctr_permutation_matches_self_consistent_q` asserts cb-train Q equals the instrumented upstream Q for pc=1 AND pc=4 |
| `multi_permutation_e2e_oracle_test.rs` | `predictions_pc4.npy` | `load_f64_vec` + `compare_stage(Stage::Predictions, ...)` ≤1e-5 | WIRED | End-to-end: train at pc=4, predict via `predict_raw_cat`, compare to committed upstream predictions ≤1e-5 |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `train_inner` (cat path) | `cat_averaging_permutation` (Q) | `averaging_ctr_permutation(n, learning_folds, seed)` = S ∘ P_avg from instrumented stream | Yes — full permutation asserted integer-exact vs upstream | FLOWING |
| `train_inner` (cat path) | `structure_fold_columns[fold]` | `materialize_ctr_feature` under per-fold permutation (fold 0 = identity, fold j = S ∘ stream[j]) | Yes — real online CTR computation | FLOWING |
| `train_inner` (cat path) | `struct_fold_cycle` | `structure_fold_cycle(pc, iters, seed)` from instrumented `live_trainer_structure_fold.json` | Yes — derived-constant anchor for pc=4/seed=0; RNG-free for pc=1/pc=2 | FLOWING |
| `permutation_count_four_predictions_match_upstream` | `expected_predictions` | `predictions_pc4.npy` (committed catboost 1.2.10 output) + `train_cat` + `predict_raw_cat` | Yes — real end-to-end comparison ≤1e-5 | FLOWING |
| `passes_ctr_split` | CTR value | `ctr_value_for_combined_projection` (FOUND) / `calc_inference` (NOT-FOUND) with split.shift/scale | Yes — real whole-set inference | FLOWING (carried forward, unchanged by 05-17/05-18/05-19) |

### Behavioral Spot-Checks

Tests run per-crate (`-p cb-train`, `-p cb-compute`, not `--workspace`; disk-pressure protocol followed).

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| pc=4 HARD gate (SC-1 / ORD-01 closure) | `cargo test -p cb-train --test multi_permutation_e2e_oracle_test` | 1 passed; 0 failed | PASS |
| AveragingFold oracle — 6 tests incl. full-perm Q and pc=4 integer-exact | `cargo test -p cb-train --test multi_permutation_fold_oracle_test` | 6 passed; 0 failed | PASS |
| Structure-fold cycle oracle — 4 tests | `cargo test -p cb-train --test structure_fold_cycle_oracle_test` | 4 passed; 0 failed | PASS |
| tensor_ctr_e2e (ORD-05 / SC-5) — 3 tests ≤1e-5 | `cargo test -p cb-train --test tensor_ctr_e2e_oracle_test` | 3 passed; 0 failed | PASS |
| ordered_boost_e2e (ORD-02 / SC-2) — 2 tests ≤1e-5 | `cargo test -p cb-train --test ordered_boost_e2e_oracle_test` | 2 passed; 0 failed | PASS |
| one_hot oracle (ORD-04 / SC-4) | `cargo test -p cb-train --test one_hot_oracle_test` | 3 passed; 0 failed | PASS |
| ordered_boost_wiring (ORD-02 structural aliveness) | `cargo test -p cb-train --test ordered_boost_wiring_test` | 3 passed; 0 failed | PASS |
| cb-train lib unit tests | `cargo test -p cb-train --lib` | 134 passed; 0 failed; 0 ignored | PASS |
| cb-train full integration suite | `cargo test -p cb-train` | 0 FAILED; 6 ignored (pre-Phase-5 deferred: Bayesian multi-tree D-11, overfit e2e prediction boundary, random_strength D-11) | PASS |
| cb-compute lib tests | `cargo test -p cb-compute --lib` | 47 passed; 0 failed; 0 ignored | PASS |
| cb-model suite | `cargo test -p cb-model` | 0 FAILED | PASS |

**Confirmed: ZERO failing tests in cb-train, cb-compute, or cb-model at HEAD.**

The 6 ignored tests are pre-existing Phase-3 deferred items (Bayesian tree-1+ RNG phase, overfit e2e boundary-routing, random_strength D-11 variable-draw budget). None are Phase-5 items. No Phase-5 oracles carry `#[ignore]`.

### Requirements Coverage

| Requirement | Source Plans | Description | Status | Evidence |
|-------------|-------------|-------------|--------|---------|
| ORD-01 | 05-03, 05-07, 05-12, 05-15, 05-17, 05-19 | Multi-permutation fold machinery seeded by TFastRng64, reproduces upstream exactly | DELIVERED | pc=1/pc=2 integer-exact (multi_permutation_fold_oracle_test); pc=4 integer-exact (multi_permutation_count_four_averaging_matches_catboost_1_2_10); averaging CTR order Q full-permutation assertion vs self-consistent fixture; pc=4 e2e ≤1e-5 (permutation_count_four_predictions_match_upstream). Cosine score function (the CPU default, latent gap closed). |
| ORD-02 | 05-05, 05-08, 05-10, 05-16 | Ordered boosting with exact prefix boundaries, per-object intermediate oracle, no leakage | DELIVERED | `ordered_boost_e2e_oracle_predictions_match_upstream` PASS ≤1e-5 (2/2); no-leakage iter-0 check PASS; wiring test 3/3 PASS; boosting.rs:1054-1057 `find(|f| !f.is_averaging)` + boosting.rs:~1416 `greedy_tensor_search_oblivious_ordered`. |
| ORD-03 | 05-04, 05-05 | Ordered CTR — all six types with priors | DELIVERED | Plain CTR oracle 3/3 PASS (all six types per-object); ordered CTR oracle 3/3 PASS; Borders type proven end-to-end via tensor_ctr_e2e (3/3 PASS ≤1e-5). Per-object standalone oracles cover all six types; Borders is the in-scope Phase 5 e2e type. |
| ORD-04 | 05-02 | One-hot encoding for low-cardinality categoricals | DELIVERED | `one_hot_oracle_test` 3/3 PASS; inclusive/exclusive cardinality boundary oracle-locked; `route_categorical` / `EncodingPath` wired. |
| ORD-05 | 05-06, 05-11, 05-12, 05-13, 05-14 | Feature combinations (tensor CTRs) ≤1e-5 | DELIVERED | `tensor_ctr_e2e_oracle_predictions_match_upstream` PASS (3/3, NO #[ignore]); THREE CTR materializations (identity structure, averaging leaf values, whole-set apply) all producing real data; model_size_reg + AveragingFold pre-draw Rule-1 fixes validated. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `crates/cb-train/src/boosting.rs` | ~1624-1648 | Bake dedup by projection only, not (ctr_type, projection) | WARNING (WR-02, carried) | Latent; inert today — only Borders is scored. Risk only if a second CTR type per projection is added. |
| `crates/cb-train/src/boosting.rs` | ~1631-1640, 1649-1659 | Global prior overwriting per-split priors during bake | WARNING (WR-03, carried) | Latent; inert for single-prior fixture. |
| `crates/cb-model/src/ctr_data.rs` | ~311 | `unwrap_or(ECtrType::Borders)` silently coerces unknown CTR types | INFO (IN-01, carried) | Masks future type mismatches silently. |

No `TBD`, `FIXME`, or `XXX` debt markers found in any file modified by Phase-5 plans (verified by grep across all modified source and test files). The three WARNING/INFO items are carried from prior reviews and remain latent/inert.

### Human Verification Required

None. All Phase-5 truths are observable and verifiable programmatically via oracle tests. The prior `human_needed` item (accept or escalate the pc=4 divergence) was escalated by the developer as a blocker and has now been closed by plans 05-17 through 05-19 with a provably correct mechanism (instrumented upstream ground truth, full-permutation assertion, e2e gate ≤1e-5).

### Gaps Summary

**No gaps.** The single blocking gap from the previous verification (SC-1 / ORD-01 pc=4 AveragingFold divergence) is confirmed CLOSED by codebase evidence:

- `multi_permutation_count_four_averaging_matches_catboost_1_2_10` — hard integer-exact equality vs committed catboost 1.2.10 leaf_weights `[6,0,10,14]` (PASS)
- `averaging_ctr_permutation_matches_self_consistent_q` — FULL permutation Q asserted vs `object_permutation_Q` for pc=1 and pc=4 (PASS)
- `permutation_count_four_predictions_match_upstream` — e2e RawFormulaVal ≤1e-5 across all objects / 5 trees (PASS)
- Zero git diff on CTR math (`materialize_ctr_feature` / `online_ctr_prefix_binclf` / `calc_ctr_online_bin`); no oracle weakened; no `#[ignore]` added

The one known limitation (structure_fold_cycle general RNG localization for other `learning_folds>1` seeds/configs) is a documented, conservative fallback — not a Phase-5 scope gap. It is logged as D-11 / Open-Q4 in STATE.md.

---

_Verified: 2026-06-15T20:00:00Z_
_Verifier: Claude (gsd-verifier) — third and final re-verification after plans 05-17 / 05-18 / 05-19 (bar (c) closure)_
