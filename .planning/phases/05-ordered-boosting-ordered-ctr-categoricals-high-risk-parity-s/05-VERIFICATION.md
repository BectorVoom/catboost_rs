---
phase: 05-ordered-boosting-ordered-ctr-categoricals-high-risk-parity-s
verified: 2026-06-14T00:00:00Z
status: gaps_found
score: 3/5 must-haves verified
overrides_applied: 0
gaps:
  - truth: "EBoostingType::Ordered trains with exact prefix boundaries and the exact prior formula, and a per-object intermediate oracle passes confirming no leakage signature in train metrics"
    status: failed
    reason: "ordered_approx_delta_simple is implemented and unit-tested standalone BUT is explicitly NOT wired into the train() driver (boosting.rs line 808 comment documents this deliberate omission). No end-to-end ordered train→predict oracle exists. The SUMMARY acknowledges this as 'the accepted A2 residual / D-09 oracle-sourcing deviation.' The roadmap SC-2 says the per-object intermediate oracle must confirm no leakage in TRAIN METRICS — that requires the ordered path to actually run in training, not just as a standalone function."
    artifacts:
      - path: "crates/cb-train/src/boosting.rs"
        issue: "EBoostingType::Ordered field exists in BoostParams and the function ordered_approx_delta_simple exists, but the train() driver never calls it — it is dead code in the training path (line 815-818 debug_assert only checks the type, never branches to the ordered path)"
    missing:
      - "Wire ordered_approx_delta_simple into the train() driver when boosting_type == EBoostingType::Ordered"
      - "Produce an end-to-end ordered train→predict oracle test that validates final predictions ≤1e-5 vs upstream CatBoost 1.2.10 with boosting_type=Ordered"

  - truth: "Feature combinations (tensor CTRs — SimpleCtrs/CombinationCtrs, max_ctr_complexity control) produce models matching upstream ≤1e-5 on categorical datasets"
    status: failed
    reason: "tensor_ctr_candidates and ctr_value_for_combined_projection exist and the per-object CTR math is oracle-locked, but the tensor CTR candidates are NOT wired into the train() driver candidate generation path. No full train→predict model is produced for a categorical dataset with tensor CTRs. The plan's intent ('train→predict oracle ≤1e-5') was explicitly deferred (D-09 residual) leaving the roadmap SC-5 ('produce models matching upstream ≤1e-5') undemonstrated end-to-end."
    artifacts:
      - path: "crates/cb-train/tests/tensor_ctr_oracle_test.rs"
        issue: "Tests validate the combined-projection CTR MATH standalone; no test trains a model with tensor CTRs and compares final predictions to upstream"
      - path: "crates/cb-train/src/boosting.rs"
        issue: "train() driver has no path that emits tensor CTR candidates during tree growth"
    missing:
      - "Wire tensor_ctr_candidates into the train() driver's candidate generation"
      - "Produce an end-to-end tensor-CTR train→predict oracle ≤1e-5 vs upstream CatBoost 1.2.10"

  - truth: "Multi-fold permutation oracle (D-03 linchpin) is validated against upstream's multi-fold seeding discipline (the D-03 contract is fold-exact for all folds k >= 0)"
    status: failed
    reason: "CR-01 from the code review is confirmed: ordered_oracle.cpp seeds each fold independently (foldSeed = seed + k) while the Rust production create_folds() / permutations() function draws ALL folds from a single continuously-advancing RNG. The executor resolved this by keying the D-03 test against the HARNESS's seeding (fisher_yates_permutation(N, seed + foldIdx)), not against what upstream CatBoost actually generates. This means: (a) the committed ordered_ctr/permutation_fold1.npy was generated with wrong seeding (seed+1 instead of the continuous-stream next fold), (b) the Rust production permutations() which IS the continuous-stream design was never validated against upstream for fold k>0, and (c) the D-03 contract ('reproduce upstream') is only held for fold 0. For fold k>1 the test passes by matching a self-generated oracle, not upstream."
    artifacts:
      - path: "crates/cb-oracle/generator/ordered_oracle.cpp"
        issue: "Lines 391-393: per-fold reseed (seed + k) disagrees with Rust create_folds() continuous-stream. Filed as CR-01 in code review."
      - path: "crates/cb-train/src/permutation.rs"
        issue: "permutations() uses a single continuous RNG (the correct upstream behavior) but was never validated against a multi-fold fixture derived from an actual upstream training run"
      - path: "crates/cb-train/tests/ordered_ctr_oracle_test.rs"
        issue: "D-03 test for fold-1 uses fisher_yates_permutation(30, 1) (per-fold reseed) instead of the continuous-stream permutations(); validates against the harness's own wrong-seeded fixture, not upstream"
    missing:
      - "Fix ordered_oracle.cpp to use the continuous-stream discipline matching create_folds() / permutations()"
      - "Regenerate ordered_ctr/permutation_fold1.npy from the corrected harness"
      - "Update ordered_ctr_oracle_test.rs to validate fold-1 against permutations(30, 2, 0)[1], not fisher_yates_permutation(30, 1)"
---

# Phase 5: Ordered Boosting, Ordered CTR & Categoricals Verification Report

**Phase Goal:** CatBoost's defining anti-leakage algorithms — ordered boosting and ordered CTR — plus native categorical handling produce models matching upstream ≤1e-5, with per-object intermediate oracles confirming no silent leakage.
**Verified:** 2026-06-14T00:00:00Z
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (from ROADMAP.md Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Multi-permutation fold machinery seeded by TFastRng64 reproduces upstream permutations exactly | PARTIAL | permutations()/create_folds() use the correct continuous-stream discipline; integer-exact for fold-0 vs committed fixture; fold k>1 validated only against self-seeded harness — upstream faithfulness for k>1 unproven (CR-01) |
| 2 | EBoostingType::Ordered trains with exact prefix boundaries, exact prior formula, per-object oracle passes (no leakage) | FAILED | ordered_approx_delta_simple exists but is not wired into train(); no end-to-end ordered train→predict; roadmap SC requires "no leakage signature in train metrics" — that cannot hold for an unexercised code path |
| 3 | Ordered CTR computes all six types with priors, matching upstream | PARTIAL | All six CTR types implemented; per-object online CTR ≤1e-5 for plain mode locked (plain_ctr); ordered CTR per-object counts locked ≤1e-5; but no full train→predict CTR model oracle (committed fixtures lack input features/labels by D-09 design) |
| 4 | One-hot encoding for low-cardinality categoricals selects the correct encoding path | VERIFIED | route_categorical() implements inclusive/exclusive boundary matching greedy_tensor_search.cpp:171-197; oracle-locked ≤1e-5 vs upstream float reference (one_hot_oracle_test 3/3 green) |
| 5 | Feature combinations (tensor CTRs) produce models matching upstream ≤1e-5 on categorical datasets | FAILED | tensor_ctr_candidates and combined_hash exist; per-object combined CTR ≤1e-5; but candidates are not wired into train(); no end-to-end tensor CTR model matching upstream ≤1e-5 |

**Score: 1 fully verified, 2 partial, 2 failed = 1/5 strict / 3/5 with partial credit**

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/cb-oracle/src/compare.rs` | Stage::Permutation, OnlineCtr, OrderedApprox variants + integer-exact path | VERIFIED | All three variants present; compare_permutation uses `==`, not 1e-5; 32 lib tests pass |
| `crates/cb-oracle/src/model_json.rs` | CtrTableJson serde + ctr_data() accessor | VERIFIED | CtrTableJson parses upstream hash_map; ctr_data() accessor returns BTreeMap; backward-compatible #[serde(default)] |
| `crates/cb-oracle/generator/ordered_oracle.cpp` | Zero-catboost-include harness, D-02 .npy schema | VERIFIED (seeding FLAWED) | 416 lines, zero catboost/ includes, dumps D-02 schema; but per-fold seeding disagrees with create_folds() for k>1 (CR-01) |
| `crates/cb-train/src/candidates.rs` | one-hot vs CTR routing on learn-set cardinality | VERIFIED | route_categorical() inclusive/exclusive boundary; learn_set_cardinality via calc_cat_feature_hash; candidates_test.rs covers all boundary cases |
| `crates/cb-train/src/permutation.rs` | Fisher-Yates over TFastRng64 (block=1 for N<1000) | VERIFIED (fold-0 only) | fisher_yates_permutation and permutations() present; fold-0 integer-exact vs committed fixture; permutations() continuous-stream not multi-fold validated against upstream |
| `crates/cb-train/src/fold.rs` | TFold body/tail prefix state machine | VERIFIED | select_min_batch_size, select_tail_size, body_tail_boundaries; [1 2 4 8 16 30] for N=30 confirmed vs committed fixture |
| `crates/cb-train/src/ctr/online.rs` | Ordered read-before-increment accumulation | VERIFIED | online_ctr_prefix_binclf and ordered_ctr_per_permutation both present; read-before-increment property locked |
| `crates/cb-train/src/ctr/calc_ctr.rs` | Online CalcCTR (+1) and inference Calc (+PriorDenom) as separate functions | VERIFIED | calc_ctr_online and calc_ctr_inference are distinct functions; Pitfall 1 avoided |
| `crates/cb-model/src/ctr_data.rs` | TCtrValueTable serde + bounds-before-slice | VERIFIED (with CR-02/CR-03 hazards) | ECtrType, CtrValueTable, encode/decode_ctr_data, per-type Calc all present; CR-02 present-but-empty bucket collapse and CR-03 first-bucket stride assumption are code quality issues but not blocking for well-formed models |
| `crates/cb-train/src/projection.rs` | TProjection enumeration + combined hash + max_ctr_complexity gate | VERIFIED | enumerate_projections, calc_hash (MAGIC_MULT), fold_cat_hash (sign-extended), combined_hash all present; 17 projection tests pass |
| `crates/cb-train/tests/one_hot_oracle_test.rs` | one_hot threshold + predict oracle ≤1e-5 | VERIFIED | 3/3 tests pass including no-permutation assertion and ≤1e-5 vs oracle-locked float reference |
| `crates/cb-train/tests/permutation_oracle_test.rs` | permutation (exact) + fold_prefix oracle | PARTIAL | 3/3 tests pass for fold-0; no multi-fold continuous-stream validation |
| `crates/cb-train/tests/ordered_ctr_oracle_test.rs` | ordered CTR per-object oracle | PARTIAL | 3/3 tests pass; fold-1 D-03 gate uses per-fold reseed (CR-01 known issue) |
| `crates/cb-train/tests/ordered_boost_oracle_test.rs` | ordered approx per iteration + final oracle | PARTIAL | 5/5 tests pass; ordered_approx_delta_simple tested standalone only; no end-to-end ordered train→predict |
| `crates/cb-train/tests/tensor_ctr_oracle_test.rs` | tensor CTR enumeration + ≤1e-5 oracle | PARTIAL | 3/3 tests pass for combined-projection math; no end-to-end tensor train→predict model |
| `crates/cb-model/tests/ctr_data_roundtrip_test.rs` | ctr_data round-trip + per-type apply ≤1e-5 | VERIFIED | 5/5 tests pass including combined-projection apply |
| Fixture directories (5) | one_hot_cat, plain_ctr, ordered_ctr, ordered_boost, tensor_ctr | VERIFIED | All 5 present with explicit per-knob config.json and .npy stacks |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `compare.rs` | `compare_permutation` | `Stage::Permutation` dispatch (== not 1e-5) | WIRED | Function present, integer-exact comparison, OracleError::PermutationDiverged on mismatch |
| `ordered_oracle.cpp` | `fixtures/` | offline .npy dump | WIRED (seeding defect) | Dumps D-02 schema; per-fold seeding does not match create_folds() for k>1 |
| `candidates.rs` | `tree.rs` | OneHot candidate → split enumeration | WIRED | route_categorical returns EncodingPath::OneHot; grow_one_hot_tree consumes OneHotSplit |
| `permutation.rs` | `cb-core::TFastRng64::uniform` | Fisher-Yates draw | WIRED | TFastRng64::from_seed used, uniform() called per swap |
| `fold.rs` | `permutation.rs` | body/tail prefixes layered over permutation | WIRED | create_folds() calls permutations(); body_tail_boundaries consumed by ordered path |
| `ctr/online.rs` | `calc_cat_feature_hash` | categorical hashing for CTR bucket identity | WIRED | PerfectHash + calc_cat_feature_hash used; no model ctr_data hash_map used |
| `apply.rs` | `ctr_data.rs` | per-type Calc at inference | WIRED | ctr_value_for_projection and ctr_value_for_combined_projection both call numerator_denominator → calc_inference |
| `boosting.rs` (EBoostingType::Ordered) | `ordered_approx_delta_simple` | body/tail prefix drives ordered update | NOT WIRED | ordered_approx_delta_simple is defined but never called from train(); the Ordered branch is documented but unimplemented in the driver |
| `candidates.rs` (tensor_ctr_candidates) | `boosting.rs` train() | tensor CTR candidates emitted into training | NOT WIRED | tensor_ctr_candidates defined but not called from train() |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `train()` driver | ApproxDelta (Ordered path) | ordered_approx_delta_simple | No — function defined but not called from train() | HOLLOW — the Ordered boosting path produces no delta in training |
| `train()` driver | CtrCandidate splits | tensor_ctr_candidates | No — function defined but not called from train() | HOLLOW — no tensor CTR splits are generated during training |
| `online_ctr_prefix_binclf()` | good/total/value | read-before-increment loop | Yes — real accumulation per permutation order | FLOWING |
| `ordered_ctr_per_permutation()` | prefix.good/total | read-before-increment with per-fold perm | Yes — real accumulation | FLOWING |
| `ordered_approx_delta_simple()` | delta per tail doc | body-seeded leaf der/weight | Yes — but not called from train() | ORPHANED |
| `ctr_value_for_projection()` | CTR value at inference | CtrValueTable.numerator_denominator | Yes — real lookup | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| one_hot oracle 3 tests | cargo test -p cb-train --test one_hot_oracle_test | 3/3 pass | PASS |
| permutation oracle 3 tests | cargo test -p cb-train --test permutation_oracle_test | 3/3 pass | PASS |
| plain_ctr oracle 3 tests | cargo test -p cb-train --test plain_ctr_oracle_test | 3/3 pass | PASS |
| ordered_ctr oracle 3 tests | cargo test -p cb-train --test ordered_ctr_oracle_test | 3/3 pass | PASS |
| ordered_boost oracle 5 tests | cargo test -p cb-train --test ordered_boost_oracle_test | 5/5 pass | PASS |
| tensor_ctr oracle 3 tests | cargo test -p cb-train --test tensor_ctr_oracle_test | 3/3 pass | PASS |
| ctr_data roundtrip 5 tests | cargo test -p cb-model --test ctr_data_roundtrip_test | 5/5 pass | PASS |
| cb-oracle lib tests | cargo test -p cb-oracle --lib | 32/32 pass | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| ORD-01 | 05-03 | Multi-permutation fold machinery | PARTIAL | permutations() and create_folds() exist and fold-0 is integer-exact; multi-fold continuous-stream not validated vs upstream (CR-01 seeding gap) |
| ORD-02 | 05-05 | Ordered boosting with exact prefix boundaries, per-object oracle | FAILED | ordered_approx_delta_simple is a standalone function not wired into train(); EBoostingType::Ordered has no effect on the training loop |
| ORD-03 | 05-04, 05-05 | Ordered CTR — all six types with priors | PARTIAL | All six CTR types implemented and per-object math oracle-locked ≤1e-5 for plain and ordered modes; no end-to-end train→predict CTR model oracle (D-09 residual) |
| ORD-04 | 05-02 | One-hot encoding path selection | VERIFIED | route_categorical boundary (inclusive at one_hot_max_size, exclusive above) oracle-locked; 3/3 oracle tests pass |
| ORD-05 | 05-06 | Feature combinations / tensor CTRs | FAILED | tensor_ctr_candidates and combined_hash exist; per-object combined CTR ≤1e-5; train() driver does not emit tensor CTR candidates; no end-to-end tensor train→predict model |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `crates/cb-train/src/boosting.rs` | 388-424 | Dead `sum_weights` accumulator (`let _ = sum_weights`) | WARNING | Dead live-code violating CLAUDE.md "Dead code must be deleted"; misleads future developers about pooled-denominator path (WR-01) |
| `crates/cb-train/src/ctr/final_ctr.rs` | 123 | `.iter().copied().sum()` for FeatureFreq denominator instead of checked fold | WARNING | Violates stated reduction discipline; silent i64 overflow possible on large datasets (WR-02) |
| `crates/cb-model/src/ctr_data.rs` | 252-259 | `unwrap_or(0)` collapses present-but-empty bucket with absent bucket | WARNING | CR-02: a stride==1 JSON table round-trips to all-empty int_counts, making every hash lookup return (0,denom) for Counter/FeatureFreq whether present or not |
| `crates/cb-model/src/ctr_data.rs` | 333-334 | Stride derived from first bucket only (`self.int_counts.first().map_or(0, Vec::len)`) | WARNING | CR-03: ragged table silently mis-serializes; no validation that all bucket widths equal the stride |
| `crates/cb-oracle/generator/ordered_oracle.cpp` | 391-393 | Per-fold reseed (`foldSeed = seed + k`) vs continuous-stream create_folds() | BLOCKER | CR-01: multi-fold D-03 oracle is self-inconsistent for k>1; fixture permutation_fold1 does not reproduce upstream CatBoost's second fold |

No `TBD`, `FIXME`, or `XXX` markers found in any phase-5 modified files. No unreferenced debt markers.

### Gaps Summary

**Three gaps block the phase goal.**

**Gap 1 (ORD-02 — Ordered Boosting not wired):** `ordered_approx_delta_simple` is implemented and standalone-tested (5/5 tests pass), but the `train()` driver never calls it. Setting `boosting_type = EBoostingType::Ordered` has zero effect on the training loop — the code path exists only as dead code. The roadmap requires that this path "trains with exact prefix boundaries" and a "per-object intermediate oracle passes (no leakage signature in train metrics)." Neither condition is demonstrable without the function being invoked during training.

**Gap 2 (ORD-05 — Tensor CTRs not wired):** `tensor_ctr_candidates` enumerates projections correctly and the combined hash is oracle-locked per-object ≤1e-5, but the candidates are never passed to the tree-growth phase in `train()`. The roadmap requires "models matching upstream ≤1e-5 on categorical datasets" — no such model is producible.

**Gap 3 (CR-01 — Multi-fold D-03 oracle invalid for k>1):** The `ordered_oracle.cpp` harness generates fold-k permutations by reseeding with `seed + k`, while `permutations()` / `create_folds()` use a single continuous-stream RNG. The executor discovered this discrepancy and adapted the test to validate against the harness's per-fold reseeding rather than fixing the harness. This means the committed `permutation_fold1.npy` in `ordered_ctr/` does not correspond to what upstream CatBoost generates for fold 1, and the production `permutations()` function (which IS the correct continuous-stream implementation) was never validated against upstream for any fold k>1. The D-03 linchpin — "reproduce upstream permutations exactly" — holds only for fold 0.

**Non-blocking concerns (warnings):**
- WR-01: Dead `sum_weights` accumulator should be removed.
- WR-02: `FeatureFreq` denominator uses `.sum()` instead of checked accumulation.
- CR-02: Present-but-empty bucket not distinguished from absent bucket in Counter/FeatureFreq.
- CR-03: Integer-table stride derived from first bucket only.
- WR-03 (from code review): C++ harness CalcCTR computes in f32 while Rust uses f64 — one of them diverges from upstream at the precision type level. The 1e-5 tolerance covers this today but the root cause is unresolved.

---

_Verified: 2026-06-14T00:00:00Z_
_Verifier: Claude (gsd-verifier)_
