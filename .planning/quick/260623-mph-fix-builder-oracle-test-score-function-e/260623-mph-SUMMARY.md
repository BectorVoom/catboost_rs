---
phase: quick-260623-mph
plan: 01
type: execute
status: complete
outcome: verify-only (no code change — fix already shipped)
date: 2026-06-23
commits: []
---

# Quick Task 260623-mph — SUMMARY

## Outcome: VERIFY-ONLY — the decided fix was already applied and is green

This task was queued from a memory note (`builder-oracle-test-preexisting-failure`)
that described the score-function fix as **"DECIDED (unapplied)."** That note was
**stale**. During planning + verification we confirmed the fix is already on `main`,
shipped by quick task **260619-bac** (2026-06-19):

| Decided element | Location | State |
|---|---|---|
| Public `.score_function(EScoreFunction)` setter | `crates/catboost-rs/src/builder.rs:249` (field `:83`, default `:114`, threaded into `BoostParams` `:307`) | ✅ present |
| Default split-score = `EScoreFunction::Cosine` (unchanged) | `crates/cb-train/src/boosting.rs:544-546` (`score_function_default()`) | ✅ unchanged |
| `EScoreFunction` re-export | `crates/catboost-rs/src/lib.rs:38` | ✅ present |
| Test opts into L2 | `crates/catboost-rs/tests/builder_oracle_test.rs:113` (`.score_function(EScoreFunction::L2)`) | ✅ present |
| Commits | `f8617f6` (setter) + `3ff254d` (test opt-in) | ✅ committed |

## Verification

```
cargo test -p catboost-rs --test builder_oracle_test
  test builder_regression_full_cycle ... ok
  test builder_binclf_full_cycle ... ok
  test result: ok. 2 passed; 0 failed
```

Both legs pass within the project oracle tolerance. The ~0.56 Cosine-vs-L2 mismatch
that originally failed the test is gone — exactly as the 2×2 isolation predicted
(L2 → ~2e-8, Cosine → ~0.56).

## Actions taken

- **No source files changed.** All `must_haves` were already satisfied (drift-detection
  found zero drift), so per the plan no code edit was applied.
- Corrected the stale memory `builder-oracle-test-preexisting-failure` to read
  **APPLIED + VERIFIED** instead of "unapplied," so this task is not re-queued again.

## Note on the second referenced item

The invocation also listed `260619-cpr-estimated-feature-stored-border-value`. That work
is likewise **already shipped** (STATE.md row `260619-cpr`, commit `e739f81`): the KNN
stored-border fix landed; only the XOR per-stage residual remains deferred, with a
definitive root cause (upstream KNN = online HNSW; closing it = porting
`library/cpp/online_hnsw`, ~936 LOC, its own phase). Nothing actionable as a quick task.
