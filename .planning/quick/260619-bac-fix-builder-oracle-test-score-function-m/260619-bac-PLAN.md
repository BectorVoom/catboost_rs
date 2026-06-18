---
phase: quick-260619-bac
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - crates/catboost-rs/src/builder.rs
  - crates/catboost-rs/src/lib.rs
  - crates/catboost-rs/tests/builder_oracle_test.rs
autonomous: true
requirements: [RAPI-01]

must_haves:
  truths:
    - "A caller can set the split-score function through the public CatBoostBuilder facade via .score_function(EScoreFunction::L2)"
    - "EScoreFunction is nameable from the catboost-rs crate without depending on cb-compute directly"
    - "builder_binclf_full_cycle passes the upstream <= 1e-5 oracle leg"
    - "builder_regression_full_cycle passes the upstream <= 1e-5 oracle leg"
    - "Default builder behavior remains Cosine (catboost CPU default) when .score_function is not called"
  artifacts:
    - path: "crates/catboost-rs/src/builder.rs"
      provides: "score_function field + .score_function() setter wired through boost_params()"
      contains: "fn score_function"
    - path: "crates/catboost-rs/src/lib.rs"
      provides: "EScoreFunction re-export from the published crate"
      contains: "EScoreFunction"
    - path: "crates/catboost-rs/tests/builder_oracle_test.rs"
      provides: "configured_builder opts into EScoreFunction::L2 to match the fixtures"
      contains: "score_function(EScoreFunction::L2)"
  key_links:
    - from: "crates/catboost-rs/src/builder.rs::boost_params"
      to: "self.score_function"
      via: "BoostParams.score_function assignment"
      pattern: "score_function: self\\.score_function"
    - from: "crates/catboost-rs/tests/builder_oracle_test.rs::configured_builder"
      to: "CatBoostBuilder::score_function"
      via: ".score_function(EScoreFunction::L2)"
      pattern: "score_function\\(EScoreFunction::L2\\)"
---

<objective>
Fix the long-standing `builder_oracle_test` failure (`builder_regression_full_cycle` +
`builder_binclf_full_cycle`, diverge at the Predictions stage) by exposing a
`.score_function(EScoreFunction)` setter on the public `CatBoostBuilder` facade and
having the test opt into `L2` to match its upstream fixtures.

Root cause (verified, `/gsd-debug` 2x2 isolation, 2026-06-19): the facade hardcodes
`score_function = Cosine` (catboost's true CPU default) with no public setter, while the
`model_serde/{binclf,regression}` fixtures were trained with `score_function = L2`.
Borders are exonerated. Driving training with L2 converges <= 1e-5 (measured 2.4e-8 /
2.8e-9 in the isolation run).

Purpose: closes the only remaining red oracle on the public facade; per-crate
cb-train/cb-compute oracles already lock L2. Keeps the catboost-default Cosine behavior
for bare `CatBoostBuilder::new()` runs.
Output: a `.score_function()` builder method, an `EScoreFunction` re-export, the test
opting into L2, and a corrected (currently false) borders docstring.
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@./CLAUDE.md
@.planning/todos/pending/builder-oracle-fix.md
@.planning/notes/builder-oracle-score-function-root-cause.md
@crates/catboost-rs/src/builder.rs
@crates/catboost-rs/src/lib.rs
@crates/catboost-rs/tests/builder_oracle_test.rs

Key interface facts (already verified — do NOT re-discover):
- `EScoreFunction` is defined in `cb_compute` (`crates/cb-compute/src/runtime.rs:832`),
  `#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]`, default `Cosine`, has an `L2`
  variant. It is already `pub use`'d from `cb_compute` (`crates/cb-compute/src/lib.rs:70`).
- `cb_train::BoostParams.score_function` has type `cb_compute::EScoreFunction`
  (`crates/cb-train/src/boosting.rs:305`).
- `cb_train::score_function_default() -> cb_compute::EScoreFunction` returns `Cosine`
  (`crates/cb-train/src/boosting.rs:544`). Builder already imports it
  (`crates/catboost-rs/src/builder.rs:29`). DO NOT change this default.
- The hardcoded assignment to replace is `crates/catboost-rs/src/builder.rs:286`:
  `score_function: score_function_default(),`.
- The builder struct fields end at `border_count: usize,` (`builder.rs:74`); `new()`
  initializes them through `border_count: QuantizeParams::default().border_count,`
  (`builder.rs:104`).
- `lib.rs:36` already does `pub use cb_compute::{LeafMethod, Loss};` — extend that line.
- The test's `configured_builder` is `builder_oracle_test.rs:98-109`; its import line is
  `builder_oracle_test.rs:34-36`. The false borders docstring is `builder_oracle_test.rs:16-22`.

CLAUDE.md constraints for production `builder.rs`: no `unwrap()`/`expect()`/`panic!`;
4-space indentation; trailing commas in multi-line literals; `#[must_use]` on setters
(match the existing setter style). Tests are exempt from the unwrap ban. Source/test
separation already holds (the test is a separate file).
</context>

<tasks>

<task type="auto">
  <name>Task 1: Expose .score_function() on CatBoostBuilder and re-export EScoreFunction</name>
  <files>crates/catboost-rs/src/builder.rs, crates/catboost-rs/src/lib.rs</files>
  <action>
In `builder.rs`:
1. Add `EScoreFunction` to the `cb_train` import block at the top (the block at lines
   25-32 that already imports `score_function_default`). `EScoreFunction` is re-exported
   from `cb_train` only if such a re-export exists; it is NOT — it lives in `cb_compute`.
   So instead import it from `cb_compute`: add `EScoreFunction` to the existing
   `use cb_compute::{ ... }` block (lines 21-23, alongside `LeafMethod`, `Loss`, etc.).
2. Add a field `score_function: EScoreFunction,` to the `CatBoostBuilder` struct
   (after `border_count: usize,` at line 74).
3. In `new()`, initialize it to the catboost default: add
   `score_function: score_function_default(),` after the `border_count: ...` line (line 104).
   This preserves the Cosine default — do NOT call `EScoreFunction::Cosine` literally;
   route through `score_function_default()` so the default stays single-sourced.
4. Add a `#[must_use]` builder setter matching the existing setter style (e.g. like
   `leaf_method` at lines 191-196):
   `pub fn score_function(mut self, score_function: EScoreFunction) -> Self { self.score_function = score_function; self }`
   with a one-line doc comment naming `score_function` and noting Cosine is the catboost
   CPU default while L2 is the variance-reduction alternative.
5. In `boost_params()`, replace the hardcoded line 286 `score_function: score_function_default(),`
   (and its preceding comment at lines 284-285 about "the facade does not surface it") with
   `score_function: self.score_function,` and an updated comment noting the facade now
   surfaces it via `.score_function()`, defaulting to the catboost CPU default (Cosine).

In `lib.rs`:
6. Extend the existing re-export at line 36 (`pub use cb_compute::{LeafMethod, Loss};`) to
   also re-export `EScoreFunction`, so callers name the L2/Cosine variants through the
   published crate without depending on `cb-compute`. Update the adjacent doc comment
   (lines 34-35) to mention the score-function knob.

Do NOT change `cb_train::score_function_default()` (it must stay Cosine). Do NOT add any
`unwrap()`/`expect()`/`panic!` to `builder.rs`.
  </action>
  <verify>
    <automated>cargo build -p catboost-rs 2>&1 | tail -20 && grep -q "score_function: self.score_function" crates/catboost-rs/src/builder.rs && grep -q "fn score_function" crates/catboost-rs/src/builder.rs && grep -q "EScoreFunction" crates/catboost-rs/src/lib.rs && echo OK</automated>
  </verify>
  <done>
`crates/catboost-rs/src/builder.rs` has a `score_function: EScoreFunction` field, a
`#[must_use] pub fn score_function(...)` setter, and `boost_params()` reads
`self.score_function` (no hardcoded `score_function_default()` in `boost_params`).
`crates/catboost-rs/src/lib.rs` re-exports `EScoreFunction` from `cb_compute`.
`cargo build -p catboost-rs` succeeds. `score_function_default()` (in cb-train) is unchanged.
A bare `CatBoostBuilder::new()` still resolves to Cosine via `score_function_default()`.
  </done>
</task>

<task type="auto">
  <name>Task 2: Opt the oracle test into L2, fix the false borders docstring, and run the oracle</name>
  <files>crates/catboost-rs/tests/builder_oracle_test.rs</files>
  <action>
1. Add `EScoreFunction` to the `catboost_rs` import list at lines 34-36 (the
   `use catboost_rs::{ ... }` block), so the test names `EScoreFunction::L2` through the
   published crate (NOT via a `cb_compute` import — proves the re-export works).
2. In `configured_builder` (lines 98-109), add `.score_function(EScoreFunction::L2)` to the
   chain (e.g. after `.leaf_method(LeafMethod::Gradient)`), matching the upstream
   `model_serde/*` fixtures that were trained with L2.
3. Fix the FALSE docstring. The module docstring at lines 16-22 (and the inline comment at
   lines 164-167) claims the facade's computed borders "reproduce upstream's border
   selection exactly" for `numeric_tiny`. That is false (49 computed borders vs the
   fixture's pinned 2/2/0/3) but benign. Rewrite those passages to state the truth: the
   facade computes its OWN greedy-logsum borders which DIFFER from the upstream fixture's
   pinned border counts, but those differences do not affect the final predictions once the
   split-score function matches (L2); the upstream <= 1e-5 oracle holds because both legs
   use L2 and the border differences cancel through the trained tree structure.
4. Also update the config-summary docstring at lines 26-28 to add
   `score_function=L2` to the listed training config, so the documented config matches what
   `configured_builder` now sets.
Keep all existing assertions and the two `#[test]` entry points unchanged.
  </action>
  <verify>
    <automated>cargo test -p catboost-rs --test builder_oracle_test 2>&1 | tail -30</automated>
  </verify>
  <done>
`builder_oracle_test.rs::configured_builder` calls `.score_function(EScoreFunction::L2)`
with `EScoreFunction` imported from `catboost_rs`. The false borders docstring is
corrected to state borders differ but do not affect predictions once score_function
matches. Running `cargo test -p catboost-rs --test builder_oracle_test` passes BOTH
`builder_binclf_full_cycle` and `builder_regression_full_cycle`, including the upstream
<= 1e-5 oracle leg (Stage::Predictions).

Disk-pressure caveat (memory: disk-pressure-and-full-suite-verification): the root disk
is ~100% full and the cb-compute test profile may fail to LINK. If — and only if — the
targeted test binary cannot link due to disk/linker exhaustion (not a logic/compile
error), record the exact linker error verbatim in the SUMMARY, confirm `cargo build
-p catboost-rs` (lib) still succeeds, and report honestly that the oracle could not be
executed in-env rather than claiming a pass. Do NOT fabricate a green result.
  </done>
</task>

</tasks>

<verification>
- `cargo build -p catboost-rs` succeeds (the production lib compiles with the new setter
  and re-export; no `unwrap()`/`expect()`/`panic!` added to `builder.rs`).
- `cargo test -p catboost-rs --test builder_oracle_test` runs both
  `builder_binclf_full_cycle` and `builder_regression_full_cycle`; both pass the upstream
  <= 1e-5 oracle (or, if the disk-pressure link failure occurs, the linker error is
  reported verbatim and no false pass is claimed).
- `cb_train::score_function_default()` is unchanged (still Cosine) — a bare
  `CatBoostBuilder::new()` remains Cosine.
- `EScoreFunction` is reachable as `catboost_rs::EScoreFunction`.
</verification>

<success_criteria>
- The public `CatBoostBuilder` exposes `.score_function(EScoreFunction)` and defaults to
  Cosine via `score_function_default()`.
- `EScoreFunction` is re-exported from `catboost-rs`.
- Both `builder_*_full_cycle` tests pass the upstream <= 1e-5 oracle through the public
  facade only (measured convergence ~2.4e-8 / ~2.8e-9 expected).
- The previously-false borders docstring is corrected.
- No regression: bare-default builder behavior stays Cosine; no `unwrap()`/`expect()`/
  `panic!` introduced into production `builder.rs`.
</success_criteria>

<output>
Create `.planning/quick/260619-bac-fix-builder-oracle-test-score-function-m/260619-bac-SUMMARY.md` when done.
On completion, update the memory note `builder-oracle-test-preexisting-failure` to
"resolved" (score-function setter exposed; test opts into L2; oracle green) per the todo's
"Done when" clause.
</output>
