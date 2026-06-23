---
phase: quick-260623-mph
plan: 01
type: execute
wave: 1
depends_on: []
files_modified: []
autonomous: true
requirements: [QUICK-260623-MPH]
must_haves:
  truths:
    - "CatBoostBuilder exposes a public `.score_function(EScoreFunction)` setter"
    - "The default split-score function is EScoreFunction::Cosine (unchanged)"
    - "builder_oracle_test opts into EScoreFunction::L2 to match its fixtures"
    - "builder_oracle_test passes (oracle parity within the project tolerance)"
  artifacts:
    - path: "crates/catboost-rs/src/builder.rs"
      provides: ".score_function() builder setter + score_function field defaulted to Cosine"
      contains: "pub fn score_function"
    - path: "crates/catboost-rs/tests/builder_oracle_test.rs"
      provides: "Test that opts into L2 via .score_function(EScoreFunction::L2)"
      contains: "score_function(EScoreFunction::L2)"
  key_links:
    - from: "crates/catboost-rs/src/builder.rs"
      to: "cb-train score_function_default()"
      via: "default field initializer in new()"
      pattern: "score_function_default\\(\\)"
---

<objective>
Close the DIAGNOSED-but-unapplied `builder_oracle_test` score-function fix: expose a
public `.score_function(EScoreFunction)` setter on the `CatBoostBuilder` facade (Rust
Builder pattern), keep the default at `EScoreFunction::Cosine`, and have the failing
`builder_oracle_test` opt into `EScoreFunction::L2` so it matches the L2-trained oracle
fixtures and passes the ≤1e-5 parity bar.

Purpose: The builder/facade defaulted to Cosine while the oracle fixtures were trained
with L2, producing a ~0.56 parity error (verified by 2x2 isolation: L2 → ~2e-8, Cosine
→ ~0.56). The decided fix lets the test opt into L2 without changing the public default.

Output: A confirmed-green `builder_oracle_test` with the setter in production source and
the L2 opt-in in the dedicated test file (source/test separation preserved).

## CURRENT STATE FINDING (read before executing)

During planning, the codebase was grepped and the test was run. The decided fix is
ALREADY APPLIED AND COMMITTED on `main` by a prior quick task (`260619-bac`):

- Setter present: `crates/catboost-rs/src/builder.rs:249`
  `pub fn score_function(mut self, score_function: EScoreFunction) -> Self`
  (field declared at `builder.rs:83`, defaulted via `score_function_default()` at
  `builder.rs:114`, threaded into `BoostParams` at `builder.rs:307`).
- Default unchanged: `crates/cb-train/src/boosting.rs:544-546`
  `score_function_default()` returns `cb_compute::EScoreFunction::Cosine`.
- Test opt-in present: `crates/catboost-rs/tests/builder_oracle_test.rs:113`
  `.score_function(EScoreFunction::L2)` inside `configured_builder(...)`.
- Re-export present: `crates/catboost-rs/src/lib.rs:38`
  `pub use cb_compute::{EScoreFunction, LeafMethod, Loss};`
- Commits:
  - `f8617f6 feat(quick-260619-bac): expose .score_function() on CatBoostBuilder`
  - `3ff254d test(quick-260619-bac): opt oracle test into L2 and fix false borders docstring`
- Test run at planning time: `cargo test -p catboost-rs --test builder_oracle_test`
  → `test result: ok. 2 passed; 0 failed` (`builder_regression_full_cycle`,
  `builder_binclf_full_cycle`).

Therefore this plan is a VERIFY-ONLY confirmation. The executor MUST NOT re-implement
the setter or re-edit the test if the state below already holds — doing so would be
redundant churn. Only make edits if a verification step below FAILS (drift detection).
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md

@crates/catboost-rs/src/builder.rs
@crates/catboost-rs/tests/builder_oracle_test.rs
@crates/cb-train/src/boosting.rs
@crates/catboost-rs/src/lib.rs
</context>

<tasks>

<task type="auto">
  <name>Task 1: Confirm setter, default, and test opt-in are present (drift-detect; apply only if absent)</name>
  <files>crates/catboost-rs/src/builder.rs, crates/catboost-rs/tests/builder_oracle_test.rs, crates/cb-train/src/boosting.rs, crates/catboost-rs/src/lib.rs</files>
  <action>
This is a confirmation task. The fix is already committed (see the CURRENT STATE FINDING
in the objective). Verify each of the four conditions below by reading/grepping the named
files. If ALL hold, make NO edits and proceed to verify.

If any condition is MISSING (repository drifted from the planning snapshot), apply the
minimal matching change, honoring CLAUDE.md (no `unwrap()` in production source; source/test
separation — setter stays in `builder.rs`, opt-in stays in the test file; match the existing
`pub fn <name>(mut self, ...) -> Self` builder idiom and surrounding doc-comment style):

1. SETTER (production source, `crates/catboost-rs/src/builder.rs`): a `score_function`
   field of type `EScoreFunction` on the builder struct, defaulted in `new()` via
   `score_function_default()`, threaded into `BoostParams` in `boost_params()`, and a public
   `pub fn score_function(mut self, score_function: EScoreFunction) -> Self` setter. If the
   setter is absent, add it next to the other `.with_*`/scalar setters (e.g. beside
   `border_count`) and wire the field through `new()` and `boost_params()`.

2. DEFAULT (unchanged): the effective default split-score function MUST remain
   `EScoreFunction::Cosine`. `score_function_default()` in `crates/cb-train/src/boosting.rs`
   returns `cb_compute::EScoreFunction::Cosine`. Do NOT flip this default to L2 under any
   circumstance.

3. RE-EXPORT: `EScoreFunction` is publicly re-exported from the facade crate
   (`crates/catboost-rs/src/lib.rs`) so test/consumer code can name the L2 variant.

4. TEST OPT-IN (dedicated test file, `crates/catboost-rs/tests/builder_oracle_test.rs`):
   the configured builder used by the oracle scenarios calls
   `.score_function(EScoreFunction::L2)` so training matches the L2-trained
   `model_serde/*` fixtures. If absent, add the `.score_function(EScoreFunction::L2)` call
   to the test's `configured_builder(...)` helper (do NOT introduce a `#[cfg(test)] mod tests`
   block in any production source file).
  </action>
  <verify>
    <automated>grep -n "pub fn score_function" crates/catboost-rs/src/builder.rs && grep -n "EScoreFunction::Cosine" crates/cb-train/src/boosting.rs && grep -n "score_function(EScoreFunction::L2)" crates/catboost-rs/tests/builder_oracle_test.rs</automated>
  </verify>
  <done>All three greps return a match: the setter exists in builder.rs, the default in boosting.rs is Cosine, and the test opts into L2. No production `#[cfg(test)] mod tests` was introduced; no `unwrap()` added to production source.</done>
</task>

<task type="auto">
  <name>Task 2: Run builder_oracle_test and confirm oracle parity passes</name>
  <files>crates/catboost-rs/tests/builder_oracle_test.rs</files>
  <action>
Run the oracle parity test for the facade crate and confirm it passes. The test
(`run_scenario` → `configured_builder`) trains through the public Builder with
`.score_function(EScoreFunction::L2)` and compares predictions against the L2-trained
`model_serde/*` oracle fixtures within the project tolerance (≤1e-5). Both scenarios
(`builder_regression_full_cycle`, `builder_binclf_full_cycle`) must pass. If a test
FAILS, surface the actual error (it indicates real drift or a regression in the score
function threading, NOT a planning gap) before any further edits.
  </action>
  <verify>
    <automated>cargo test -p catboost-rs --test builder_oracle_test 2>&1 | tail -5</automated>
  </verify>
  <done>`cargo test -p catboost-rs --test builder_oracle_test` reports `test result: ok. 2 passed; 0 failed`; the oracle parity error is within the ≤1e-5 bar (no ~0.56 Cosine-mismatch error).</done>
</task>

</tasks>

<verification>
- `.score_function()` setter present and public on `CatBoostBuilder` (builder.rs).
- Default split-score function remains `EScoreFunction::Cosine` (boosting.rs `score_function_default`).
- `builder_oracle_test` opts into `EScoreFunction::L2` and passes 2/2 scenarios.
- No `#[cfg(test)] mod tests` added to production source; no production `unwrap()` introduced.
</verification>

<success_criteria>
- `cargo test -p catboost-rs --test builder_oracle_test` → 2 passed, 0 failed.
- Public default score function is unchanged (Cosine); only the setter is exposed and the
  test opts into L2.
- Source/test separation preserved (setter in `builder.rs`, opt-in in the `_test.rs` file).
</success_criteria>

<output>
Create `.planning/quick/260623-mph-fix-builder-oracle-test-score-function-e/260623-mph-SUMMARY.md` when done.
Note in the summary whether edits were needed or whether the fix was already present
(committed by prior quick task `260619-bac` in commits `f8617f6` and `3ff254d`).
</output>
