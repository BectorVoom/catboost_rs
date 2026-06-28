---
status: partial
phase: 05-ordered-boosting-ordered-ctr-categoricals-high-risk-parity-s
source: [05-01-SUMMARY.md, 05-02-SUMMARY.md, 05-03-SUMMARY.md, 05-04-SUMMARY.md, 05-05-SUMMARY.md, 05-06-SUMMARY.md, 05-07-SUMMARY.md, 05-08-SUMMARY.md, 05-09-SUMMARY.md, 05-10-SUMMARY.md]
started: 2026-06-14
updated: 2026-06-14
mvp_guard_override: "Standard outcome-based UAT — Mode:mvp goals are outcome-format, not user-story; backend parity library (user = Rust dev embedding cb-train/cb-model)"
---

## Current Test

[testing complete]

## Tests

### 1. One-hot categorical training (ORD-04)
expected: A low-card categorical model trains via the one-hot path and predicts ≤1e-5 vs upstream. `cargo test -p cb-train --test one_hot_oracle_test` → 3/3 pass.
result: pass
note: User ran the command — 3/3 green (one_hot_path_selection_boundary, no_permutation_in_one_hot_only_path, one_hot_predict_matches_oracle_locked_float_reference).

### 2. Multi-permutation folds reproducible (ORD-01)
expected: Fold permutations reproduce upstream's continuous-stream Fisher-Yates integer-exact, for fold 0 AND fold k>0 (the 05-07 CR-01 fix). `cargo test -p cb-train --test permutation_oracle_test` → 3/3 pass.
result: pass

### 3. Plain CTR — all six types (ORD-03)
expected: All six CTR types computed whole-set match upstream per-object num/denom (exact) + value ≤1e-5. `cargo test -p cb-train --test plain_ctr_oracle_test` → 3/3 pass.
result: pass

### 4. Ordered CTR — per-object, no leakage (ORD-03 ordered)
expected: Ordered (per-permutation prefix, read-before-increment) CTR matches upstream per-object; identity-permutation degeneration confirms no self-inclusion leakage. `cargo test -p cb-train --test ordered_ctr_oracle_test` → 3/3 pass.
result: pass

### 5. Ordered boosting — per-object approximant anchor (ORD-02)
expected: The ordered boosting per-object intermediate approximant matches upstream ≤1e-5 (the indirect no-leakage anchor). `cargo test -p cb-train --test ordered_boost_oracle_test` → 5/5 pass.
result: pass

### 6. Ordered split-scoring subsystem — structure differs from Plain (ORD-02 structural)
expected: The ordered tree search scores splits over the learning fold's BodyTailArr (segment-summed L2), degenerates to the Plain search at a single full-span segment + identity permutation, and produces a structure that differs from Plain on real multi-segment folds. `cargo test -p cb-train --lib tree::ordered` → 8 units + `cargo test -p cb-train --test ordered_boost_wiring_test` → 3/3.
result: pass

### 7. Tensor / combination CTRs — per-object (ORD-05)
expected: SimpleCtrs/CombinationCtrs under max_ctr_complexity enumerate combined projections and match upstream per-object combined-hash CTR ≤1e-5. `cargo test -p cb-train --test tensor_ctr_oracle_test` → 3/3 pass.
result: pass

### 8. CTR-split model representation + categorical apply (05-09 1a/1b)
expected: The model carries a first-class CTR split (`ModelSplit::Ctr`) and `predict_raw_cat` evaluates it via the baked ctr_data table; numeric `ModelSplit::Float` apply is byte-identical. `cargo test -p cb-model --test apply_oracle_test` → 3/3 pass.
result: pass

### 9. ctr_data serialize round-trip + upstream load (model)
expected: `ctr_data` round-trips through `.cbm`/`model.json` and loads from upstream ≤1e-5. `cargo test -p cb-model --test ctr_data_roundtrip_test` → 5/5 pass.
result: pass

### 10. Malformed ctr_data → typed error, never panic (security)
expected: A malformed/oversized ctr_data blob surfaces a typed `ModelError`, never a panic/OOB. `cargo test -p cb-model --lib ctr_data` (cb-model lib 15/15).
result: pass

### 11. No Plain-path regression (numeric boosting unchanged)
expected: The numeric Plain boosting path is unchanged by all the ordered/CTR additions. `cargo test -p cb-train --test slice_first_oracle_test --test leaf_methods_oracle_test` → 2/2 + 3/3 pass.
result: pass

### 12. ORD-02 final-prediction e2e oracle (FULL multi-tree)
expected: An ordered-boosting train→predict model matches upstream ≤1e-5 across all 5 trees through `cb_model::predict_raw` (no #[ignore]). `cargo test -p cb-train --test ordered_boost_e2e_oracle_test`.
result: blocked
blocked_by: third-party
reason: "Requires offline ordered_boost_e2e/ fixtures from catboost==1.2.10, not importable in this environment. Test code committed and wired (no #[ignore]); fails only on NotFound for fixture files. Generate offline then re-run."

### 13. ORD-05 categorical train→predict e2e oracle (FULL multi-tree)
expected: A tensor-CTR categorical train→predict model matches upstream ≤1e-5 across all 5 trees through `cb_model::predict_raw`/`predict_raw_cat` (no #[ignore]). `cargo test -p cb-train --test tensor_ctr_e2e_oracle_test`.
result: blocked
blocked_by: third-party
reason: "Requires offline tensor_ctr_e2e/ fixtures from catboost==1.2.10, not importable in this environment. Test code committed and wired (no #[ignore]); fails only on NotFound for fixture files. Also depends on wiring the tree.rs::CtrSplitSpec candidate-emission stub. Generate offline then re-run."

## Summary

total: 13
passed: 11
issues: 0
pending: 0
blocked: 2
skipped: 0

## Gaps

[none yet]
