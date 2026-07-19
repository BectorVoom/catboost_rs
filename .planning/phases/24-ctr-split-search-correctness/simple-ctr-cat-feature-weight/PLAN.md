# TDD Implementation Plan — ORD-07 Phantom Mixed-Projection `max_bucket_count`

> **Status: COMPLETE (2026-07-19).** T0-T6 all landed and verified:
> `phantom_count_*`/`phantom_gate_*` unit tests, `max_bucket_count_unchanged_at_level0`,
> `max_bucket_count_includes_phantom_at_level1`/`_at_level2` all green
> (`cargo test -p cb-train --lib`); `fstr_ctr_oracle_test` (the T5/T6 target,
> doubly-blocked by ORD-06 then ORD-07) all 3 tests GREEN; `ctr_split_scoring_test`,
> `tensor_ctr_e2e_oracle_test`, `multi_permutation_e2e_oracle_test` green,
> UNCHANGED expected values (provable no-op confirmed for both zero-float-feature
> fixtures). Full `cargo test -p cb-train` and `cargo test -p cb-model` green
> modulo the pre-existing, unrelated `monotone_non_symmetric_and_region_are_typed_errors`
> failure (reproduced on a clean stash — see `catboost-rs-preexisting-test-failures`
> memory). `cargo clippy -p cb-train --all-targets` is pre-existing-red in
> `cb-oracle`/`cb-backend` (unrelated files, also reproduced on a clean stash);
> no new clippy findings in this slice's own files. FSTR-01's T5/T6 blocker
> (`.planning/phases/18-extended-feature-importance/fstr-01-interaction-ctr/`)
> is now fully resolved.

> **`[UNVERIFIED: Planner Agent unavailable]`** — no project-installed agent
> named `planner` exists (`.claude/agents/` contains `code-reviewer.md`,
> `plan-checker.md`, `research-agent.md`, `specification-executor.md`,
> `specification-planner.md` — none matches the goal-backward TDD Planner
> Agent this skill calls for). This plan was authored directly by the
> spec-tdd-planner-skill session as the documented fallback, using the same
> goal-backward method, and CodeGraph-verified every symbol/file/line cited
> below (plus, uniquely for this slice, a LIVE `catboost==1.2.10` debug-log
> spike — see SPEC.md §1 / research.md's Addendum). It still must pass the
> independent Plan Checker gate — see `PLAN-CHECK.md`.

**Phase:** 24 (CTR split-search correctness) · **Slice:** ORD-07 (sibling to,
depends on, and does NOT modify ORD-06's already-landed fix)
**Spec:** `./SPEC.md` (specs ORD-07-01, ORD-07-02, ORD-07-03) ·
**Requirement:** ORD-07 · **Crate:** `cb-train` (`tree.rs` + `boosting.rs` +
a small `candidates.rs` addition) · **Impact:** `local`
**Parity bar:** `1e-5` (CPU, D-12) via `cb_oracle::compare::assert_abs_close`
for the oracle tests; ORD-07-01/02/03a's unit tests are exact (integer/set
arithmetic, no floating averaging).

> Executor contract: strict Red → Green → Refactor per task. **This fix
> extends ORD-06's already-landed, already-verified `max_bucket_count`
> computation — it does NOT touch `combination_ctr_eligible` or
> `eligible_max_bucket_count`'s existing logic.** The new contribution is
> ADDITIVE (a second `max(...)` term), gated independently ("chosen contains
> `>= 1` `Float` split", unrelated to ORD-06-04's "chosen contains an
> already-used `Ctr` projection" gate).
>
> **Source/test separation is mandatory** — no inline `#[cfg(test)] mod
> tests { ... }` body in production `.rs`. New unit tests for the pure
> counting/gating primitives (ORD-07-01/02) go in
> `crates/cb-train/src/tree_test.rs` (`mod general` — already mounted,
> hosts ORD-06's own unit tests). **T3 no longer writes a new hand-rolled
> hashing function** (revised per plan-checker pass 1 — it reuses the
> already-existing, already-oracle-tested `cb_data::perfect_hash_bins`
> instead), so no new `candidates_test.rs` coverage is required; if an
> OPTIONAL thin delegating wrapper is added for naming consistency, its
> one-line test goes in `crates/cb-train/src/candidates_test.rs` (already
> exists and is mounted). Integration/oracle tests stay in
> `crates/cb-train/tests/`
> (`ctr_split_scoring_test.rs`) and `crates/cb-model/tests/`
> (`fstr_ctr_oracle_test.rs`, already exists, currently RED at level 2).
>
> **No `unwrap`/`expect`/`panic`/`indexing_slicing`** in production
> (workspace-denied). Restriction-lint gate is `cargo clippy -p cb-train
> --all-targets` (NOT `cargo build`).
>
> **Regression discipline:** ORD-06's fix
> (`combination_ctr_eligible`/`eligible_max_bucket_count`, already landed
> and verified) must NOT be altered — this plan's new contribution combines
> with it via a shared outer `max(...)`, added alongside, not folded into
> its internals. `learn_set_cardinality` (existing, oracle-tested via every
> CTR fixture) must NOT be modified — and the new per-object bucket data
> must NOT come from a new hand-rolled `PerfectHash`/`calc_cat_feature_hash`
> loop either (plan-checker pass 1 MAJOR finding): it comes from calling the
> ALREADY-EXISTING, already-`pub`, already-oracle-tested
> `cb_data::perfect_hash_bins` DIRECTLY (T3), never a second, independently-
> maintained hashing implementation.
>
> **Frozen fixtures:** `crates/cb-oracle/fixtures/fstr_ctr/` and ALL other
> CTR fixtures are FROZEN — never regenerate. If ANY existing fixture's
> expected value needs to change after this fix, STOP — this plan's own
> "provable no-op" analysis (SPEC §2/§7) would have been wrong; do not patch
> a fixture to match new output.

## Validation commands (host CPU)

```
cargo test -p cb-train                                          # full crate regression
cargo test -p cb-train --test tensor_ctr_e2e_oracle_test         # must stay green, UNCHANGED (0 float features — provable no-op)
cargo test -p cb-train --test multi_permutation_e2e_oracle_test  # must stay green, UNCHANGED (0 float features)
cargo test -p cb-train --test ctr_split_scoring_test             # must stay green (model_size_reg=0.0 in all existing tests)
cargo test -p cb-model --test fstr_ctr_oracle_test -- --nocapture # THE target: 3 tests must go GREEN
cargo test -p cb-model                                           # full crate regression
cargo clippy -p cb-train --all-targets                           # RESTRICTION-LINT GATE
```

Known-red, pre-existing, environmental suites to IGNORE: `cb-backend --lib`
(CubeCL MLIR), `cb-train`'s `monotone_*` tests, `catboost-rs-py` (python3.14
link).

## Task graph (dependencies, not file order)

```
T0 verify current source ─> T1 (ORD-07-01, phantom_mixed_bucket_count, unit)
                          ─> T2 (ORD-07-02, gating predicate, unit)          [parallel with T1 — different function, no shared state]
                          ─> T3 (reuse cb_data::perfect_hash_bins — verification only, no new hashing fn)  [parallel with T1/T2 — different file]
T1 + T2 + T3 ─> T4 (ORD-07-03: wire into select_level_ctr_aware + thread plumbing from boosting.rs; AT-ORD07-03a)
T4 ─> T5 (oracle verification: AT-ORD07-03b target + AT-ORD07-03c regression)
T5 ─> T6 (full regression sweep + clippy gate + doc updates)
```
- **T1, T2, T3 are genuinely parallelizable** (pure functions in different
  files/different concerns: `tree.rs`'s counting primitive, `tree.rs`'s
  gating predicate, `candidates.rs`'s new hashing-reuse function — no shared
  mutable state, no ordering dependency between them). T1 and T2 both land
  in `tree.rs`/`tree_test.rs` — if executed by a single executor, serialize
  the actual file edits (same guidance as ORD-06's T2/T3 note) even though
  they are logically independent.
- T4 is the integration/wiring task and must wait for all three.
- T5/T6 strictly serial after T4.

---

## T0 — Re-verify current source matches this plan's assumptions (enabler, no spec)

- **Goal:** confirm ORD-06's fix is present and unaltered, and that
  `select_level_ctr_aware`/`build_ctr_aware_histogram`/`boosting.rs`'s
  `train_inner` call site still match SPEC.md §3/§4's description, BEFORE
  writing any code.
- **Steps:**
  1. `Read crates/cb-train/src/tree.rs` around `select_level_ctr_aware`
     (confirm `combination_ctr_eligible`/`eligible_max_bucket_count` are
     present, per ORD-06; confirm the exact current line numbers, which may
     have drifted from this plan's citations).
  2. Confirm `assign_leaves_ctr_aware` is a standalone, callable (not
     inlined-away) function reachable from `select_level_ctr_aware`'s
     module scope, and that calling it a SECOND time (once inside
     `build_ctr_aware_histogram`, once directly in
     `select_level_ctr_aware` for this fix) is cheap and side-effect-free
     (pure function over `matrix`/`ctr_features`/`chosen`).
  3. Confirm `cb_data::perfect_hash_bins` (`crates/cb-data/src/cat_hash.rs:471-479`)
     still exists, is `pub`, and is exported from `cb-data/src/lib.rs` — T3
     reuses it directly (plan-checker pass 1 finding: do NOT hand-roll a
     duplicate hashing function).
  4. **Visibility check (MANDATORY — plan-checker pass 1 CRITICAL finding;
     an earlier draft of this plan got this wrong):** confirm
     `select_level_ctr_aware` is private (`fn`, no `pub`) with exactly 1
     caller, AND confirm `greedy_tensor_search_oblivious_with_ctr`'s EXACT
     current visibility and full caller list via
     `grep -rn "greedy_tensor_search_oblivious_with_ctr" crates/` — as of
     this plan's writing it is `pub fn` (`tree.rs:2747`), re-exported from
     `cb-train`'s crate root (`lib.rs:102-106`), with 1 production call
     (`boosting.rs:3900`) plus 5 direct calls from the external
     integration-test crate `crates/cb-train/tests/ctr_split_scoring_test.rs`
     (lines 99, 147, 189, 246, 301). **T4 must update all 6 of these real
     call sites, not a hypothetical "7 call sites."** Re-run this grep at
     T0 time to catch any drift before starting T4.
  5. Confirm `boosting.rs`'s `train_inner` still computes
     `cat_cardinalities`/`eligible_absolute` at the cited lines (~2696-2721)
     and that `cat_columns: &[Vec<String>]` remains in scope at the
     `greedy_tensor_search_oblivious_with_ctr` call site (~3900).
- **If the source has drifted** materially from this description, STOP and
  re-derive the exact insertion points before proceeding.
- **Completion evidence:** a short confirmation note (commit message, not a
  file) that the source matches SPEC.md's description, or a documented
  deviation.

## T1 — ORD-07-01: `phantom_mixed_bucket_count` (unit)

- **Spec:** ORD-07-01. **Depends on:** T0. **Parallel with:** T2, T3.
- **Files:** `crates/cb-train/src/tree.rs` (new private function, placed
  near `eligible_max_bucket_count`), `crates/cb-train/src/tree_test.rs`
  (`mod general`, new tests).
- **Red** — in `tree_test.rs`:
  - `phantom_count_all_distinct` (AT-ORD07-01a): `leaf_of=[0,0,1,1]`,
    `cat_bucket=[0,1,0,1]` → `4`.
  - `phantom_count_single_leaf_repeated_value` (AT-ORD07-01b):
    `leaf_of=[0,0,0,0]`, `cat_bucket=[0,1,2,0]` → `3`.
  - `phantom_count_empty` (AT-ORD07-01c): `leaf_of=[]`, `cat_bucket=[]` →
    `0`.
  - `phantom_count_repeated_pair_counted_once` (AT-ORD07-01d):
    `leaf_of=[0,0,0]`, `cat_bucket=[5,5,5]` (same pair 3 times) → `1`.
  - **Expected initial failure:** `phantom_mixed_bucket_count` does not
    exist → compile error naming it.
- **Green:**
  ```rust
  #[must_use]
  fn phantom_mixed_bucket_count(leaf_of: &[usize], cat_bucket: &[u32]) -> usize {
      leaf_of
          .iter()
          .zip(cat_bucket.iter())
          .map(|(&leaf, &bucket)| (leaf, bucket))
          .collect::<std::collections::HashSet<_>>()
          .len()
  }
  ```
  (exact per SPEC §4; `.zip` naturally truncates to the shorter length if
  `leaf_of`/`cat_bucket` mismatch — no panic, no indexing).
- **Refactor:** none expected.
- **Validation:** `cargo test -p cb-train phantom_count`.
- **Completion evidence:** AT-ORD07-01a–d green.

## T2 — ORD-07-02: gating predicate (unit)

- **Spec:** ORD-07-02. **Depends on:** T0. **Parallel with:** T1, T3.
- **Files:** `crates/cb-train/src/tree.rs` (new private function),
  `crates/cb-train/src/tree_test.rs`.
- **Red** — in `tree_test.rs`:
  - `phantom_gate_false_when_chosen_empty` (AT-ORD07-02a): `chosen=[]` →
    `false`.
  - `phantom_gate_false_when_only_ctr_chosen` (AT-ORD07-02b):
    `chosen=[Ctr{col:0,border:10.0}]` → `false`.
  - `phantom_gate_true_when_float_chosen` (AT-ORD07-02c):
    `chosen=[Float(Split{feature:1,border:-0.2014})]` → `true`.
  - `phantom_gate_true_when_mixed` (AT-ORD07-02d):
    `chosen=[Float{...}, Ctr{...}]` → `true`.
  - **Expected initial failure:** the gating function does not exist.
- **Green:**
  ```rust
  #[must_use]
  fn phantom_bucket_gate(chosen: &[CtrAwareSplit]) -> bool {
      chosen.iter().any(|s| matches!(s, CtrAwareSplit::Float(_)))
  }
  ```
- **Refactor:** none expected.
- **Validation:** `cargo test -p cb-train phantom_gate`.
- **Completion evidence:** AT-ORD07-02a–d green.

## T3 — Reuse `cb_data::perfect_hash_bins` for raw per-object cat-bucket data (unit)

> **Revised per plan-checker pass 1 (MAJOR finding):** an earlier draft of
> this task proposed hand-rolling a NEW `candidates.rs::learn_set_buckets`
> function. Independent review found this would have duplicated an
> ALREADY-EXISTING, already-`pub`, already-oracle-tested function
> byte-for-byte: `cb_data::perfect_hash_bins(column: &[&str]) ->
> CbResult<Vec<u32>>` (`crates/cb-data/src/cat_hash.rs:471-479`, exported
> via `cb-data/src/lib.rs:39`, oracle-tested against a real `.npy` fixture
> in `crates/cb-data/tests/cat_hash_oracle_test.rs`). `cb-train` already
> imports from `cb_data` directly in both `candidates.rs:43` and
> `boosting.rs:35` — zero new dependency. This task now REUSES it directly
> instead of writing new hashing logic.

- **Spec:** enables ORD-07-03 (the raw per-object cat-bucket data source).
  **Depends on:** T0 (confirm `cb_data::perfect_hash_bins` still exists/is
  `pub`, per T0 step 3). **Parallel with:** T1, T2.
- **Files:** `crates/cb-train/src/boosting.rs` (call `cb_data::perfect_hash_bins`
  directly at the point `cat_eligible_buckets` is assembled in T4 — see
  below; NO new file/function needed for the counting logic itself).
- **Red/Green:** no new PRODUCTION function is written by this task — it is
  reduced to a VERIFICATION step, not a Red→Green cycle of its own:
  1. Confirm `cargo test -p cb-data cat_hashes_and_perfect_hash_bins_match_oracle`
     (or the equivalent test name in `cat_hash_oracle_test.rs` — confirm the
     exact name at execution time) is GREEN before relying on
     `cb_data::perfect_hash_bins` as this fix's data source — this IS the
     acceptance evidence for this task (AT-T3a), reusing an existing oracle
     guarantee rather than writing new hand-written unit tests that would
     only re-derive hashing correctness `cb-data`'s own suite already
     proves.
  2. If a `cb-train`-local name is still desired for documentation
     consistency with `learn_set_cardinality` (OPTIONAL, not required), add
     a THIN, ONE-LINE delegating wrapper only:
     ```rust
     fn learn_set_buckets(column: &[&str]) -> CbResult<Vec<u32>> {
         cb_data::perfect_hash_bins(column)
     }
     ```
     with a single test (AT-T3b, if this wrapper is added) asserting
     delegation (`learn_set_buckets(&col) == cb_data::perfect_hash_bins(&col)`
     for a small sample column) — NOT re-deriving hashing correctness from
     scratch. If this wrapper is skipped, T4 calls
     `cb_data::perfect_hash_bins` directly and this sub-step does not apply.
- **Refactor:** none — `learn_set_cardinality` remains completely untouched
  either way (this task never modifies it).
- **Validation:** `cargo test -p cb-data cat_hash` (confirm the existing
  oracle test is green); if the optional wrapper is added,
  `cargo test -p cb-train learn_set_buckets`.
- **Completion evidence:** `cb_data::perfect_hash_bins`'s existing oracle
  test confirmed green (AT-T3a); `learn_set_cardinality` untouched
  (`git diff` shows zero changes to `candidates.rs` from this task).

## T4 — ORD-07-03: wire the phantom contribution into `max_bucket_count`

- **Spec:** ORD-07-03. **Depends on:** T1, T2, T3.
- **Files:** `crates/cb-train/src/tree.rs`
  (`select_level_ctr_aware`/`greedy_tensor_search_oblivious_with_ctr` gain a
  new parameter; the `max_bucket_count` computation extends),
  `crates/cb-train/src/boosting.rs` (`train_inner`'s `has_ctr` call site
  computes and threads the new per-eligible-cat-feature bucket data),
  `crates/cb-train/src/tree_test.rs` (AT-ORD07-03a),
  `crates/cb-train/tests/ctr_split_scoring_test.rs` (5 call sites updated to
  pass `&[]`, per the enumerated list below — NOT a new test, a required
  compile-fix to its 5 EXISTING calls).
- **Red** — in `tree_test.rs`:
  - `max_bucket_count_includes_phantom_at_level1` (AT-ORD07-03a, part 1):
    hand-built `ctr_features` (simple `{0}` bucket_count=5, simple `{1}`
    bucket_count=4, combination `{0,1}` bucket_count=20 — mirrors the real
    fixture's cardinalities), `chosen=[Float(1)@-0.2014]`, and
    `cat_eligible_buckets` constructed so `phantom_mixed_bucket_count`
    yields `10` for cat0 and `8` for cat1 (either literal small vectors
    reproducing this, or the REAL fixture's `X_float.npy`/`X_cat.npy`
    slices, per SPEC's worked table) → the computed `max_bucket_count ==
    10`.
  - `max_bucket_count_includes_phantom_at_level2` (AT-ORD07-03a, part 2):
    SAME setup, `chosen=[Float(1)@-0.2014, Float(0)@0.561]`, phantom counts
    `20`/`16` → `max_bucket_count == 20`.
  - `max_bucket_count_unchanged_at_level0` (AT-ORD07-03a, part 3,
    regression lock): `chosen=[]` → `max_bucket_count == 5` (UNCHANGED from
    ORD-06-04's current output — phantom gate is off).
  - **Expected initial failure:** before the Green step, `max_bucket_count`
    stays at `5` for parts 1/2 too (the phantom contribution doesn't exist
    yet) — record this as the RED baseline.
- **Green:**
  1. In `boosting.rs`'s `train_inner` (~line 2696-2721 area, re-confirmed at
     T0), after computing `eligible_absolute`, compute
     `cat_eligible_buckets: Vec<Vec<u32>>` — for each `abs_idx` in
     `eligible_absolute`, call
     `cb_data::perfect_hash_bins(&cat_columns[abs_idx]...)` (T3's reused
     function; stringified the SAME way `cat_cardinalities` already does,
     reusing the existing `as_str` conversion pattern at line ~2698-2701 —
     do not invent a new stringification).
  2. Thread `cat_eligible_buckets: &[Vec<u32>]` as a new parameter through
     `greedy_tensor_search_oblivious_with_ctr` (~line 2747) down to
     `select_level_ctr_aware` (~line 2588). **`select_level_ctr_aware` is
     private with exactly 1 caller** (the `..._with_ctr` function itself) —
     low-risk. **`greedy_tensor_search_oblivious_with_ctr` is `pub`,
     re-exported from the crate root, with EXACTLY 6 real call sites that
     must ALL be updated for the crate to compile (NOT "7 call sites in a
     numeric path" — that framing was wrong, corrected per plan-checker
     pass 1 CRITICAL finding; there is no such call site, since the
     numeric-only path calls the DIFFERENT function
     `greedy_tensor_search_oblivious`, never this one):**
     1. `crates/cb-train/src/boosting.rs:3900` (production, `train_inner`'s
        `has_ctr` branch) — pass the REAL `cat_eligible_buckets` computed in
        step 1.
     2. `crates/cb-train/tests/ctr_split_scoring_test.rs:99`
     3. `crates/cb-train/tests/ctr_split_scoring_test.rs:147`
     4. `crates/cb-train/tests/ctr_split_scoring_test.rs:189`
     5. `crates/cb-train/tests/ctr_split_scoring_test.rs:246`
     6. `crates/cb-train/tests/ctr_split_scoring_test.rs:301`

        (call sites 2-6 are in an EXTERNAL integration-test crate — each
        hand-constructs synthetic `CtrFeatureColumn`s with no backing
        `cat_columns` at all; pass `&[]` at each — numerically inert since
        every existing test in this file uses `model_size_reg = 0.0`
        (`cat_feature_weight` short-circuits to `1.0` before `max_count` is
        even read) OR has zero CTR-eligible cat features for the phantom
        gate to iterate over. **Re-run the T0 grep
        (`grep -rn "greedy_tensor_search_oblivious_with_ctr" crates/`)
        immediately before this step to catch any line-number drift or a
        7th call site introduced since this plan was written** — do not
        trust this enumerated list blindly if the source has changed.)

        **One of these 5 test call sites
        (`forward_bit_leaf_index_mixed_float_and_ctr`,
        `ctr_split_scoring_test.rs:172-210`) chooses a `Float` split at
        level 0 then a `Ctr` split at level 1, so `phantom_bucket_gate`
        WILL evaluate `true` there at level 1 — still numerically inert
        (`model_size_reg = 0.0`), but the new parameter must still be
        correctly threaded through THIS call for it to compile and produce
        its existing (unchanged) expected values.**
  3. Inside `select_level_ctr_aware`, immediately after computing
     `used_projections`/`max_bucket_count` (ORD-06-04's existing code,
     UNCHANGED), add:
     ```rust
     let phantom_max = if phantom_bucket_gate(chosen) {
         let leaf_of = assign_leaves_ctr_aware(matrix, ctr_features, chosen, n_objects);
         cat_eligible_buckets
             .iter()
             .map(|buckets| phantom_mixed_bucket_count(&leaf_of, buckets))
             .max()
             .unwrap_or(0)
     } else {
         0
     };
     let max_bucket_count = max_bucket_count.max(phantom_max).max(1);
     ```
     (placed AFTER ORD-06-04's `eligible_max_bucket_count` call, combining
     via a single outer `.max(...)` — ORD-06-04's own computation and this
     new term are independent, additive contributions to the SAME final
     value; the existing `.max(1)` floor is preserved).

     **Performance note (plan-checker pass 1 acknowledgment, non-blocking):**
     this adds a SECOND call to `assign_leaves_ctr_aware` per level (the
     first is already inside `build_ctr_aware_histogram`'s own call at this
     same level) — confirmed pure/deterministic/side-effect-free, so this is
     safe, and it is a bounded PER-LEVEL cost (not a per-candidate rescan,
     unlike what this module's PERF-02 discipline elsewhere guards against),
     additionally gated to zero cost at level 0 via `phantom_bucket_gate`.
     Accepted as-is; no further optimization required by this spec.
- **Refactor:** none expected beyond ensuring the new parameter's name/type
  is consistent across all 6 call sites. Re-run T1/T2/T3's unit tests.
- **Validation:**
  ```
  cargo test -p cb-train --lib tree::general
  cargo build -p cb-train   # confirm all 6 call sites compile with the new parameter
  cargo test -p cb-train --test ctr_split_scoring_test   # the 5 external-crate call sites, unchanged expected values
  ```
- **Completion evidence:** AT-ORD07-03a (all 3 parts) green; T1/T2/T3 still
  green; `cargo build -p cb-train` compiles with zero call-site omissions;
  `ctr_split_scoring_test.rs`'s existing 5 tests (whose call sites were
  updated to compile) still pass with UNCHANGED expected values.

## T5 — Oracle verification: target fixture + full CTR regression suite

- **Spec:** ORD-07-03 (AT-ORD07-03b, AT-ORD07-03c). **Depends on:** T4.
- **No new production code expected** — this task is the end-to-end proof.
- **Steps:**
  1. `cargo test -p cb-model --test fstr_ctr_oracle_test -- --nocapture` —
     all 3 tests MUST now pass at `<= 1e-5`.
     - **If they pass:** proceed to step 2.
     - **If the sanity gate STILL fails** (residual divergence beyond this
       fix): per SPEC §9 risk 1, the residual ~1% gap identified during
       research was attributed to a border-labeling uncertainty, not a
       mechanism error — if this residual turns out to be non-negligible
       in practice, re-examine `phantom_mixed_bucket_count`'s exact
       counting rule (e.g., whether one-hot-routed features should also
       phantom-contribute, or whether the leaf-partition definition needs
       to match `assign_leaves_ctr_aware`'s exact forward-bit convention
       more precisely) BEFORE assuming the whole mechanism is wrong — the
       level-1 near-exact-tie-boundary fit (SPEC §1) is strong evidence the
       mechanism itself is correct.
     - **If interaction/PVC fail but the sanity gate passes:** the
       divergence is in FSTR-01's OWN attribution code, OUT OF SCOPE for
       this bugfix (same contingency ORD-06's PLAN already established) —
       stop and report as a separate, newly-surfaced issue.
  2. `cargo test -p cb-train --test tensor_ctr_e2e_oracle_test --test
     multi_permutation_e2e_oracle_test --test ctr_split_scoring_test` — ALL
     must pass with UNCHANGED expected values (both `tensor_ctr_e2e`/
     `multi_permutation_e2e` have ZERO float features, so `phantom_bucket_gate`
     is provably always `false` for them — a genuine no-op, not merely an
     assumed one; `ctr_split_scoring_test.rs`'s existing tests all use
     `model_size_reg=0.0`, making the weight's MAGNITUDE irrelevant there,
     though the new parameter threading must still compile and produce
     identical results). If ANY expected value changes, STOP — do not patch
     the fixture; the "provable no-op" claim was wrong and needs
     re-diagnosis.
- **Validation:** the exact commands in steps 1-2 above.
- **Completion evidence:** AT-ORD07-03b (3/3 `fstr_ctr` tests green) AND
  AT-ORD07-03c (all 3 regression suites green, unchanged) both hold
  simultaneously.

## T6 — Full regression sweep, clippy gate, doc updates

- **Depends on:** T5.
- **Steps:**
  - Add/verify doc comments on `phantom_mixed_bucket_count`,
    `phantom_bucket_gate`, and the `cb_data::perfect_hash_bins` call site
    (or the optional thin wrapper, if added) citing the exact
    upstream source (`AddTreeCtrs`'s `binAndOneHotFeaturesTree`,
    `greedy_tensor_search.cpp:517-522`; `CalcMaxFeatureValueCount`,
    `:1097-1115`) and this bugfix's live-spike provenance (research.md),
    matching this module's existing citation style.
  - Re-run the FULL `cb-train` test suite (`cargo test -p cb-train`) and
    the FULL `cb-model` test suite (`cargo test -p cb-model`) to confirm
    zero regressions beyond the specifically-targeted suites already
    checked in T5.
  - Grep the new code for `.unwrap()`/`.expect()`/`panic!`/raw `[]`
    indexing as a smell check (the AUTHORITATIVE gate is clippy, below).
- **Validation (full slice):**
  ```
  cargo clippy -p cb-train --all-targets
  cargo build -p cb-train
  cargo test -p cb-train
  cargo test -p cb-model
  ```
- **Completion evidence:** ORD-07-01/02/03 acceptance tests all green;
  `fstr_ctr_oracle_test.rs` (the original blocker, now doubly-blocked by
  ORD-06 then ORD-07) fully green; ALL existing CTR oracle fixtures
  unchanged; restriction lints clean on new code. Then (bookkeeping,
  outside TDD): note in FSTR-01's phase artifacts
  (`.planning/phases/18-extended-feature-importance/fstr-01-interaction-ctr/`)
  that its T5/T6 blocker is now fully resolved.

## Traceability (task → spec → acceptance)

| Task | Spec | Acceptance tests | Kind |
|------|------|------------------|------|
| T0 | (enabler) | source assumptions confirmed | — |
| T1 | ORD-07-01 | AT-ORD07-01a/b/c/d | unit |
| T2 | ORD-07-02 | AT-ORD07-02a/b/c/d | unit |
| T3 | (enabler for ORD-07-03) | AT-T3a (+ optional AT-T3b) | unit |
| T4 | ORD-07-03 | AT-ORD07-03a (3 parts) | unit |
| T5 | ORD-07-03 | AT-ORD07-03b, AT-ORD07-03c | oracle/regression |
| T6 | all | full-slice green, zero regressions | gate |

Every SPEC acceptance behavior (§6 roll-up) has a Red task; every task
references ≥1 spec ID.
