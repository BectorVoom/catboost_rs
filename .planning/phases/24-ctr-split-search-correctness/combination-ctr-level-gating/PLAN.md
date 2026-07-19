# TDD Implementation Plan — ORD-06 Combination-CTR Level-Gating Bugfix

> **Status: COMPLETE (2026-07-19).** T0-T4 all landed and verified: 7/7
> `combination_ctr_eligible` unit tests, AT-ORD06-03c, AT-ORD06-04a/b/c green
> (`cargo test -p cb-train --lib`); `ctr_split_scoring_test`,
> `tensor_ctr_e2e_oracle_test`, `multi_permutation_e2e_oracle_test` green,
> UNCHANGED expected values; `fstr_ctr_oracle_test` (the original blocker) all
> 3 tests GREEN (`fstr_ctr_predictions_sanity_gate`,
> `interaction_matches_upstream_on_mixed_ctr_model`,
> `pvc_matches_upstream_on_mixed_ctr_model`); full `cargo test -p cb-train`
> and `cargo test -p cb-model` green modulo the pre-existing, unrelated
> `monotone_non_symmetric_and_region_are_typed_errors` failure (verified
> present on a clean stash too — see
> `catboost-rs-preexisting-test-failures` memory). `cargo clippy -p cb-train
> --all-targets` is pre-existing-red in `cb-oracle`/`cb-backend`
> (unrelated files, also reproduced on a clean stash); no new clippy findings
> in this slice's own files. Superseded by ORD-07
> (`../simple-ctr-cat-feature-weight/PLAN.md`), which lands on top of this
> fix.

> **`[UNVERIFIED: Planner Agent unavailable]`** — no project-installed agent
> named `planner` exists (`.claude/agents/` contains `code-reviewer.md`,
> `plan-checker.md`, `research-agent.md`, `specification-executor.md`,
> `specification-planner.md` — none matches the goal-backward TDD Planner
> Agent this skill calls for). This plan was authored directly by the
> spec-tdd-planner-skill session as the documented fallback, using the same
> goal-backward method, and CodeGraph-verified every symbol/file/line cited
> below. It still must pass the independent Plan Checker gate — see
> `PLAN-CHECK.md`.

**Phase:** 24 (CTR split-search correctness) · **Slice:** ORD-06
**Spec:** `./SPEC.md` (specs ORD-06-01, ORD-06-02, ORD-06-03) ·
**Requirement:** ORD-06 (extends the ORD-0x lineage; corrects an ORD-05
latent defect) · **Crate:** `cb-train` (single crate) · **Impact:** `local`
**Parity bar:** `1e-5` (CPU, D-12) via `cb_oracle::compare::assert_abs_close`,
except ORD-06-01/02's unit tests, which are exact (pure set-membership
arithmetic, no floating averaging).

> Executor contract: strict Red → Green → Refactor per task. **This fix is
> unusually small and surgical** (§1/§4 of SPEC.md — an ELIGIBILITY
> PREDICATE plus one `continue` guard inside an EXISTING loop in `tree.rs`;
> `candidates.rs` and `boosting.rs` are explicitly UNCHANGED). Resist any
> temptation to build the larger "lazy per-level materialization" machinery
> research.md's Recommended Architecture originally sketched — SPEC.md's §1
> "Architecture correction" supersedes it with a smaller, equally-correct
> design; if the actual scoring loop in `tree.rs` does not match what this
> plan describes (re-verify against the CURRENT source before starting, in
> case the file has changed since this plan was written), STOP and
> re-consult SPEC.md rather than reverting to the larger design.
>
> **Source/test separation is mandatory** — no inline `#[cfg(test)] mod
> tests { ... }` body in production `.rs`. `tree.rs` already mounts FIVE
> sibling test files (`tree_test.rs` as `mod general`, `tree_tie_break_test.rs`,
> `tree_ordered_test.rs`, `tree_pairwise_test.rs`, `region_grow_test.rs` —
> `crates/cb-train/src/tree.rs:91-105,3154-3156`); the new predicate's unit
> tests go in `tree_test.rs` (the general-purpose sibling), no new mount
> needed. Integration/oracle tests stay in `crates/cb-train/tests/`
> (`ctr_split_scoring_test.rs`) and `crates/cb-model/tests/`
> (`fstr_ctr_oracle_test.rs`, already exists).
>
> **No `unwrap`/`expect`/`panic`/`indexing_slicing`** in production
> (workspace-denied `[VERIFIED: LOCAL Cargo.toml:10-14]`). Do **not** mark
> any task complete during planning.
>
> **Regression discipline (the single highest-severity risk in this slice):**
> `cat_feature_weight`'s FORMULA, `build_ctr_aware_histogram`,
> `score_candidate_ctr_aware`, `split_score`/`l2_split_score`/
> `cosine_split_score`, and the strict `> best` tie-break are ALL CONFIRMED
> correct against upstream (SPEC.md §1) — **do not touch any of these
> formulas**. The fix is: one NEW private function
> (`combination_ctr_eligible`), one NEW `continue` guard in the existing
> "CTR candidates next" loop, AND a reorder+filter of the existing
> `max_bucket_count` computation (ORD-06-04 — a SECOND, independent
> scoring-INPUT bug the Plan Checker found: `max_bucket_count` feeds
> `cat_feature_weight` but was, before this correction, still computed over
> the unfiltered tree-wide column list — without T2.5, the primary oracle
> test is at material risk of still failing even with T1+T2 correctly
> implemented). Every other line of `select_level_ctr_aware`,
> `build_ctr_aware_histogram`, and `greedy_tensor_search_oblivious_with_ctr`
> stays byte-identical.
>
> **Frozen fixtures:** `crates/cb-oracle/fixtures/fstr_ctr/` and ALL other
> CTR fixtures (`tensor_ctr_e2e`, `multi_permutation_e2e`, `plain_ctr`,
> `ordered_ctr`, `tensor_ctr`, `one_hot_cat`) are FROZEN — upstream
> quantization is run-to-run nondeterministic, so NONE may be regenerated.
> This fix must make the ALREADY-COMMITTED `fstr_ctr` fixture pass and must
> NOT change any other fixture's expected values.

## Validation commands (host CPU)

```
cargo test -p cb-train                                          # full crate regression
cargo test -p cb-train --test tensor_ctr_e2e_oracle_test         # must stay green, UNCHANGED
cargo test -p cb-train --test multi_permutation_e2e_oracle_test  # must stay green, UNCHANGED
cargo test -p cb-train --test ctr_split_scoring_test             # must stay green + new AT-ORD06-03c
cargo test -p cb-model --test fstr_ctr_oracle_test -- --nocapture # THE target: 3 tests must go GREEN
cargo test -p cb-model                                           # full crate regression (fstr_oracle_test etc.)
cargo clippy -p cb-train --all-targets                           # RESTRICTION-LINT GATE
```
> **Lint-gate correction (recurring project gotcha):** the workspace
> restriction lints (`unwrap_used/expect_used/panic/indexing_slicing`) are
> **clippy** lints; inert under `cargo build`/`rustc`, enforced ONLY by
> `cargo clippy` `[VERIFIED: LOCAL Cargo.toml:10-14]`.

Known-red suites to ignore (pre-existing, environmental, unrelated):
`cb-backend --lib` (CubeCL MLIR), `cb-train`'s `monotone_*` tests,
`catboost-rs-py` (python3.14 link) `[PROJECT: memory
catboost-rs-preexisting-test-failures.md]`.

## Task graph (dependencies, not file order)

```
T0 verify current source ─> T1 (ORD-06-01/02, unit: combination_ctr_eligible) ─> T2 (ORD-06-03, wire the guard + AT-ORD06-03c) ─> T2.5 (ORD-06-04, max_bucket_count scoping fix — plan-checker CRITICAL) ─> T3 (oracle verification, target + regression) ─> T4 (full regression sweep + gate)
```
- **Strictly serial** — this is a small, single-file fix; there is no
  meaningful parallelization opportunity (T1 must exist before T2 and T2.5
  can call it; T2.5 reorders/filters code adjacent to what T2 touches, so it
  must follow T2 to avoid conflicting edits to the same function; T3
  requires BOTH T2 and T2.5 to have landed, since T3's primary acceptance
  test is at material risk of failing with T2 alone — see T2.5's note; T3's
  result determines whether T4 is a clean gate or requires returning to
  T1/T2/T2.5).

---

## T0 — Re-verify the current `tree.rs` source matches this plan's assumptions (enabler, no spec)

- **Goal:** confirm `select_level_ctr_aware`'s "CTR candidates next" loop and
  `used_projections` computation are still at the cited lines and still
  shaped as SPEC.md §4 describes, BEFORE writing any code — this SPEC/PLAN
  were authored against a specific read of `crates/cb-train/src/tree.rs`
  (lines ~2527-2632), and the file may have changed.
- **Steps:** `Read crates/cb-train/src/tree.rs` around lines 2527-2632;
  confirm: (a) `used_projections: Vec<&crate::TProjection>` is computed from
  `chosen` via a `filter_map` over `CtrAwareSplit::Ctr { col, .. }` (existing,
  unchanged); (b) the "CTR candidates next" loop iterates
  `for col in 0..ctr_features.len()` and unconditionally computes
  `cat_weight`/`score`/pushes to `scored` for every column, with NO existing
  eligibility filter; (c) `TProjection::is_combination()` and
  `TProjection::cat_features()` are `pub` and reachable from `tree.rs`.
- **If the source has drifted** from this description (e.g. a filter already
  exists, or the loop has been restructured), STOP and re-derive the exact
  insertion point before proceeding — do not force this plan's literal code
  onto a changed function.
- **Validation:** read-only step, no test to run; confirm via `Read`/`grep`.
- **Completion evidence:** a short confirmation note (in the PR description
  or commit message, not a file) that the source matches SPEC.md §4's
  description, OR a documented deviation if it does not.

## T1 — ORD-06-01/02: implement `combination_ctr_eligible` (unit)

- **Spec:** ORD-06-01, ORD-06-02. **Depends on:** T0.
- **Files:** `crates/cb-train/src/tree.rs` (new private function, placed near
  `cat_feature_weight` at `tree.rs:2416` or immediately before
  `select_level_ctr_aware` at `tree.rs:2527` — either is fine, Planner's
  stylistic call), `crates/cb-train/src/tree_test.rs` (new tests, `mod
  general`).
- **Red** — in `tree_test.rs`:
  - `combination_ineligible_when_no_ctr_used_empty` (AT-ORD06-01a):
    `combination_ctr_eligible(&TProjection::from_features(&[0,1]), &[])` →
    `false`.
  - `combination_ineligible_when_used_is_unrelated` (part of AT-ORD06-01b's
    intent, restated precisely): construct `used = TProjection::single(5)`
    (standing in for "a tree with only Float splits chosen, so
    `used_projections` never received a Ctr-derived entry" — the predicate
    itself cannot distinguish "genuinely no Ctr chosen" from "a Ctr chosen on
    an unrelated feature"; SPEC's degenerate case is simply the EMPTY-slice
    input, already covered by AT-ORD06-01a) →
    `combination_ctr_eligible(&TProjection::from_features(&[2,3]), &[&used])`
    → `false` (SPEC AT-ORD06-02's second scenario, "unrelated projection").
  - `combination_eligible_extends_simple_ctr` (AT-ORD06-02, scenario 1):
    `used = TProjection::single(0)`;
    `combination_ctr_eligible(&TProjection::from_features(&[0,1]), &[&used])`
    → `true`.
  - `combination_ineligible_length_gap_two` (AT-ORD06-02, scenario 3):
    `used = TProjection::single(0)`;
    `combination_ctr_eligible(&TProjection::from_features(&[0,1,2]), &[&used])`
    → `false` (length gap 2, not 1).
  - `combination_eligible_via_any_of_multiple_used` (AT-ORD06-02, scenario 4):
    `used0 = TProjection::single(0)`, `used1 = TProjection::single(1)`;
    `combination_ctr_eligible(&TProjection::from_features(&[0,1]), &[&used0,
    &used1])` → `true`.
  - `combination_eligible_extends_existing_combination` (AT-ORD06-02,
    scenario 5): `used = TProjection::from_features(&[0,1])`;
    `combination_ctr_eligible(&TProjection::from_features(&[0,1,2]), &[&used])`
    → `true`.
  - `combination_ineligible_against_itself` (AT-ORD06-02, scenario 6):
    `used = TProjection::from_features(&[0,1])`;
    `combination_ctr_eligible(&TProjection::from_features(&[0,1]), &[&used])`
    → `false` (length gap 0).
  - **Expected initial failure:** `combination_ctr_eligible` does not exist →
    compile error naming it; record it, then implement.
- **Green:** implement in `tree.rs`:
  ```rust
  #[must_use]
  fn combination_ctr_eligible(
      projection: &TProjection,
      used_projections: &[&TProjection],
  ) -> bool {
      let members = projection.cat_features();
      used_projections.iter().any(|q| {
          let q_members = q.cat_features();
          q_members.len() + 1 == members.len()
              && q_members.iter().all(|m| members.contains(m))
      })
  }
  ```
  (exact signature and body per SPEC.md §4 — no `unwrap`/indexing needed,
  pure iterator combinators throughout).
- **Refactor:** none expected — this is already a minimal, single-purpose
  function. If `members.contains` on a `&[usize]` slice reads awkwardly,
  consider `.iter().all(|m| members.iter().any(|x| x == m))` for clarity —
  behaviorally identical, purely stylistic.
- **Validation:** `cargo test -p cb-train --lib tree::general` (or whatever
  the exact module path resolves to — confirm via
  `cargo test -p cb-train combination_ctr_eligible`).
- **Completion evidence:** all 7 unit tests above green; `combination_ctr_eligible`
  is `#[must_use]`, private (`fn`, not `pub fn` — internal to `tree.rs`'s own
  search, no external caller needed per SPEC).

## T2 — ORD-06-03: wire the eligibility guard into `select_level_ctr_aware`

- **Spec:** ORD-06-03. **Depends on:** T1.
- **Files:** `crates/cb-train/src/tree.rs` (one `continue` guard added inside
  the existing "CTR candidates next" loop, `tree.rs:2589-2610`),
  `crates/cb-train/tests/ctr_split_scoring_test.rs` (new regression-fence
  test, AT-ORD06-03c).
- **Red** — in `ctr_split_scoring_test.rs` (reuse the file's existing
  `ctr_column_from_bins`-style helper and `greedy_tensor_search_oblivious_with_ctr`
  call pattern — `crates/cb-train/tests/ctr_split_scoring_test.rs:1-40` — do
  NOT hand-roll a new harness):
  - `combination_ctr_cannot_win_at_level_zero` (AT-ORD06-03c): construct a
    synthetic scenario with (a) a materialized COMBINATION
    `CtrFeatureColumn` (`projection: TProjection::from_features(&[0,1])`)
    whose bins are chosen to score ARTIFICIALLY HIGH (e.g. a perfect
    separator, so that WITHOUT the fix it would win), alongside (b) at least
    one `FeatureMatrix` float column and/or a SIMPLE CtrFeatureColumn with a
    genuinely lower score, and `chosen = &[]` (level 0, empty — no `Ctr`
    split chosen yet). Call `greedy_tensor_search_oblivious_with_ctr(...)`
    with `depth = 1` and assert the resulting tree's level-0 split is NOT
    the combination CTR (either assert it's the float/simple-CTR split that
    should legitimately win, or — if constructing a clean "legitimate
    winner" is fiddly — at minimum assert the combination candidate is
    excluded from `scored` by testing `select_level_ctr_aware` more
    directly if it is reachable from this integration test's crate boundary;
    if `select_level_ctr_aware` is private and unreachable from
    `crates/cb-train/tests/`, assert via the PUBLIC
    `greedy_tensor_search_oblivious_with_ctr`'s returned tree structure
    instead).
  - **Expected initial failure:** BEFORE T2's Green, the combination CTR
    wins (since it was constructed to score artificially high and nothing
    currently filters it at level 0) — the test fails by asserting the
    "wrong" (bug-reproducing) winner is currently chosen.
- **Green:** in `tree.rs`'s existing "CTR candidates next" loop
  (`tree.rs:2589-2610`), insert:
  ```rust
  for col in 0..ctr_features.len() {
      let Some(column) = ctr_features.get(col) else { continue };
      if column.projection.is_combination()
          && !combination_ctr_eligible(&column.projection, &used_projections)
      {
          continue;
      }
      // ... existing cat_weight / score / push(scored) logic, UNCHANGED ...
  }
  ```
  placed as the FIRST statement inside the loop body (before the existing
  `cat_weight` computation), so an ineligible combination column never
  reaches `cat_feature_weight`/`score_candidate_ctr_aware` at all — no
  wasted computation, matching upstream's "never even considered" semantics,
  not merely "considered but scored zero."
- **Refactor:** none expected — this is a single `if`/`continue` insertion.
  Re-run T1's unit tests to confirm `combination_ctr_eligible` itself is
  untouched.
- **Validation:**
  ```
  cargo test -p cb-train --test ctr_split_scoring_test
  cargo test -p cb-train --lib tree::general        # T1's unit tests still green
  ```
- **Completion evidence:** AT-ORD06-03c green (the combination no longer
  wins at level 0); T1's 7 unit tests still green (untouched).

## T2.5 — ORD-06-04: scope `max_bucket_count` to the per-level eligible set (plan-checker CRITICAL)

- **Spec:** ORD-06-04. **Depends on:** T1 (`combination_ctr_eligible`), T2
  (must land after the guard exists, though this task's own code change is
  independent of T2's `continue` insertion — both consume the same
  predicate). **MANDATORY — do NOT skip or defer to "if T3 fails."** The
  Plan Checker independently confirmed that without this task,
  `fstr_ctr_oracle_test.rs`'s sanity gate (AT-ORD06-03a) is likely to still
  fail after T1+T2 alone, because `max_bucket_count` (a `cat_feature_weight`
  scoring INPUT, `tree.rs:2576-2581`) is computed over ALL materialized CTR
  columns unconditionally — including now-ineligible combinations — and
  ORD-06-03's guard only filters `scored`'s membership, not this separate
  computation.
- **Files:** `crates/cb-train/src/tree.rs` only (reorder + filter two
  existing `let` bindings inside `select_level_ctr_aware`);
  `crates/cb-train/src/tree_test.rs` (new unit tests).
- **Red** — in `tree_test.rs` (or a small standalone test of the reordered
  computation if `max_bucket_count`'s logic is easiest to test by extracting
  it into its own tiny private helper — Planner/executor's call; either
  achieves the same Red/Green target):
  - `max_bucket_count_excludes_ineligible_combination_at_root` (AT-ORD06-04a):
    construct `ctr_features` = `[simple {0} bucket_count=5, simple {1}
    bucket_count=4, combination {0,1} bucket_count=20]`, `chosen = []` →
    the computed `max_bucket_count == 5` (NOT `20`).
  - `max_bucket_count_includes_combination_once_eligible` (AT-ORD06-04b):
    same `ctr_features`, `chosen = [Ctr(single(0))]` (making `{0,1}`
    eligible per T1's predicate) → `max_bucket_count == 20`.
  - `max_bucket_count_unchanged_for_all_simple_columns` (AT-ORD06-04c,
    regression lock): `ctr_features` = only simple columns (no combination
    at all) → `max_bucket_count` computed IDENTICALLY to the pre-fix formula
    (a plain `.max()` over all of them — proves the fix is a no-op for
    combination-free configs, e.g. any level of `plain_ctr`/`ordered_ctr`).
  - **Expected initial failure:** the current unconditional `.max()` gives
    `20` for AT-ORD06-04a (wrong — should be `5`); record this, then fix.
- **Green:** in `select_level_ctr_aware` (`tree.rs`, exact current lines to
  be re-confirmed per T0's discipline — approximately `2576-2590` as of this
  writing), REORDER the two existing `let` bindings so `used_projections` is
  computed FIRST, then filter `max_bucket_count`'s input by the SAME
  eligibility rule ORD-06-03 uses:
  ```rust
  let used_projections: Vec<&crate::TProjection> = chosen
      .iter()
      .filter_map(|s| match s {
          CtrAwareSplit::Ctr { col, .. } => ctr_features.get(*col).map(|c| &c.projection),
          CtrAwareSplit::Float(_) => None,
      })
      .collect();
  let max_bucket_count = ctr_features
      .iter()
      .filter(|c| c.projection.is_simple() || combination_ctr_eligible(&c.projection, &used_projections))
      .map(|c| c.bucket_count)
      .max()
      .unwrap_or(1)
      .max(1);
  ```
  (the ONLY change: `used_projections` moves above `max_bucket_count`, and
  `max_bucket_count`'s `.iter()` gains a `.filter(...)` clause using the SAME
  predicate T2 already uses for `scored` — do not maintain two separately-
  written copies of the eligibility rule; both call sites must use the exact
  same `combination_ctr_eligible` function).
- **Refactor:** none expected — this is a minimal reorder + filter. Re-run
  T1's and T2's tests to confirm neither regressed.
- **Validation:**
  ```
  cargo test -p cb-train --lib tree::general
  cargo test -p cb-train --test ctr_split_scoring_test
  ```
- **Completion evidence:** AT-ORD06-04a/b/c green; T1's and T2's tests still
  green (untouched).

## T3 — Oracle verification: target fixture + full CTR regression suite

- **Spec:** ORD-06-03, ORD-06-04 (AT-ORD06-03a, AT-ORD06-03b). **Depends on:** T2.5.
- **No new production code expected** — this task is the end-to-end proof
  that T1+T2's fix is BOTH necessary and sufficient, and that it does not
  perturb any currently-passing CTR fixture.
- **Steps:**
  1. `cargo test -p cb-model --test fstr_ctr_oracle_test -- --nocapture` —
     all 3 tests (`fstr_ctr_predictions_sanity_gate`,
     `interaction_matches_upstream_on_mixed_ctr_model`,
     `pvc_matches_upstream_on_mixed_ctr_model`) MUST now pass at `<= 1e-5`.
     - **If the SANITY GATE (predictions) passes but interaction/PVC still
       fail:** the divergence is now confirmed to be in FSTR-01's OWN
       attribution code (`crates/cb-model/src/fstr.rs`) — OUT OF SCOPE for
       this bugfix (FSTR-01 is a separate, already-implemented,
       already-reviewed slice per its own PLAN-CHECK.md) — stop and report
       this as a DIFFERENT, newly-surfaced issue rather than attempting to
       fix `fstr.rs` under this plan.
     - **If the SANITY GATE ITSELF still fails after T1+T2+T2.5 land
       (plan-checker MAJOR-required contingency — do NOT skip this branch):**
       do NOT conclude the eligibility diagnosis (ORD-06-01/02/03) or the
       `max_bucket_count` fix (ORD-06-04) was wrong. FIRST re-check whether
       ANY OTHER per-level-scoped scoring input still reads from the
       tree-wide static `ctr_features` instead of the eligible subset — the
       `max_bucket_count` gap (ORD-06-04) was found by the Plan Checker
       specifically because it is easy to miss (a scoring INPUT, not the
       candidate list itself); a similarly-shaped gap could exist elsewhere
       in `select_level_ctr_aware`/`build_ctr_aware_histogram` (re-read both
       functions in full, line by line, checking every place `ctr_features`
       is iterated UNFILTERED, before concluding the predicate's own
       arithmetic (ORD-06-01/02) is at fault). Only after ruling out every
       such per-level-scoped-input gap should the predicate's own logic be
       re-examined.
  2. `cargo test -p cb-train --test tensor_ctr_e2e_oracle_test --test
     multi_permutation_e2e_oracle_test --test ctr_split_scoring_test` — ALL
     must pass with expected values UNCHANGED from before this fix. If any
     expected `.npy`/hard-coded value needs to change to pass, STOP — this
     means the fix altered a fixture the SPEC explicitly required to stay a
     no-op (SPEC §5 ORD-06-03, "must be re-verified... a provable no-op");
     do NOT edit the fixture's expected values to make the test pass — that
     would be masking a regression, not fixing one. Investigate whether
     `combination_ctr_eligible`'s logic is wrong for one of these fixtures'
     specific tree-growth history instead.
  3. If step 2 reveals a genuine, expected structural change (i.e., these
     fixtures' bug was NOT actually latent-and-unreachable as SPEC §9 risk 3
     inferred, but WAS reachable and is now legitimately fixed), this is a
     MATERIAL FINDING requiring explicit escalation back to a human before
     proceeding — do not silently accept a changed oracle comparison.
- **Validation:** the exact commands in steps 1-2 above.
- **Completion evidence:** AT-ORD06-03a (3/3 fstr_ctr tests green) AND
  AT-ORD06-03b (tensor_ctr_e2e, multi_permutation_e2e, ctr_split_scoring_test
  all green, UNCHANGED expected values) both hold simultaneously.

## T4 — Full regression sweep, clippy gate, doc updates

- **Depends on:** T3.
- **Steps:**
  - Add/verify a doc comment on `combination_ctr_eligible` citing the exact
    upstream source (`AddTreeCtrs`, `greedy_tensor_search.cpp:503-568`, the
    `v1.2.10` tag) and this codebase's categorical-only simplification
    rationale (SPEC §1), matching this module's existing citation style
    (e.g. `cat_feature_weight`'s doc comment at `tree.rs:2416`).
  - Re-run the FULL existing `cb-train` test suite (`cargo test -p
    cb-train`) to confirm zero regressions beyond the specifically-targeted
    CTR suites already checked in T3.
  - Re-run the FULL `cb-model` test suite (`cargo test -p cb-model`) to
    confirm `fstr_oracle_test.rs`/`advanced_fstr_oracle_test.rs` (FSTR-01's
    OWN regression suite, unrelated to this bugfix except via the shared
    `fstr_ctr` fixture) remain green.
  - Grep the new code for `.unwrap()`/`.expect()`/`panic!`/raw `[]` indexing
    as a smell check (the AUTHORITATIVE gate is clippy, below).
- **Validation (full slice):**
  ```
  cargo clippy -p cb-train --all-targets
  cargo build -p cb-train
  cargo test -p cb-train
  cargo test -p cb-model
  ```
- **Completion evidence:** ORD-06-01/02/03 acceptance tests all green;
  `fstr_ctr_oracle_test.rs` (the original blocker) fully green; ALL existing
  CTR oracle fixtures unchanged; restriction lints clean on new code. Then
  (bookkeeping, outside TDD): note in the FSTR-01 phase artifacts
  (`.planning/phases/18-extended-feature-importance/fstr-01-interaction-ctr/`)
  that its T5/T6 blocker is resolved and those tasks can now be completed if
  not already re-run as part of this fix's T3.

## Traceability (task → spec → acceptance)

| Task | Spec | Acceptance tests | Kind |
|------|------|------------------|------|
| T0 | (enabler) | source assumptions confirmed | — |
| T1 | ORD-06-01, ORD-06-02 | 7 unit tests (see T1 Red) | unit |
| T2 | ORD-06-03 | AT-ORD06-03c | unit/integration |
| T2.5 | ORD-06-04 | AT-ORD06-04a/b/c | unit |
| T3 | ORD-06-03, ORD-06-04 | AT-ORD06-03a, AT-ORD06-03b | oracle/regression |
| T4 | all | full-slice green, zero regressions | gate |

Every SPEC acceptance behavior (§6 roll-up) has a Red task; every task
references ≥1 spec ID.
