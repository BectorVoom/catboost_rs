# TDD Implementation Plan — FSTR-01 Interaction + PredictionValuesChange CTR Support

> ## Execution status (2026-07-18) — ✅ COMPLETE, AT-FIC02d / AT-FIC03d GREEN
> The oracle acceptance test (`crates/cb-model/tests/fstr_ctr_oracle_test.rs`)
> is fully green at `<= 1e-5`, closing FSTR-01. Three changes got it there:
>
> 1. **Decoupled from ORD-07** (user-approved): merged
>    `feat/23-ctr-model-loading` into this branch (conflict-free; commit
>    `9f41acb`) and switched the oracle test from in-process `train_cat`
>    re-training to loading upstream's own `fstr_ctr/model.cbm` via the
>    CTR-capable `load_cbm`. The sanity gate passes EXACTLY, so the T5
>    "settled fact" note (load_cbm can't produce CTR models) is obsolete on
>    this branch. ORD-07's training-side residual stays paused/documented in
>    `.planning/phases/24-ctr-split-search-correctness/simple-ctr-cat-feature-weight/`.
> 2. **`interaction()` upstream-parity fix** (SPEC §9 risk 7 materialized —
>    the agent-fetched C++ description WAS subtly wrong): upstream groups
>    split levels by border-less INTERNAL feature identity (`TFeature`),
>    skips same-internal level pairs outright, and — the actual bug — keeps
>    each internal pair's FULL score in `totalEffect` while dropping
>    self-cells only from the OUTPUT (`totalEffect += effect` sits outside
>    the cross-product loops, `calc_fstr.cpp:343-414`). The pre-fix code
>    normalized by surviving cells only, inflating every percentage by a
>    uniform 1.08138x on the fixture (upstream's five scores sum to ~92.5,
>    not 100). AT-FIC02e's expected value was re-derived accordingly
>    (25.0, not 33.33); new discriminating units AT-FIC02f/g added.
> 3. **`prediction_values_change_with_data()`** (SPEC §9 risk 8 materialized):
>    the fixture used `data=pool`, which makes upstream RECOMPUTE per-leaf
>    weights from the pool via the apply path (`CollectLeavesStatistics`,
>    `fstr/util.cpp`) — genuinely different from stored training-time
>    `leaf_weights` for online-CTR models. New `cb-model` public API:
>    `collect_leaves_statistics` (apply.rs) + `prediction_values_change_with_data`
>    (fstr.rs); the no-data mode is byte-identical to before (AT-FIC03a
>    re-asserted). New unit AT-FIC03e locks the recomputed-weights path.
>
> Full `cargo test -p cb-model` (101 lib + all integration suites) and
> `cargo test -p cb-train` green; clippy clean on all touched files.

> **Revision note (2026-07-17, post plan-checker pass #1 — verdict
> `ISSUES_FOUND`):** this revision closes all 5 required revisions from
> `PLAN-CHECK.md`: (1) MAJOR — added mandatory non-symmetric-arm hand-built
> unit test AT-FIC02e to T2, with explicit per-cross-product-cell self-pair
> guidance; (2) MINOR — T5 now states as settled fact (not a to-verify item)
> that `load_cbm` cannot produce a CTR-bearing model on this branch; (3)
> MINOR — the T2/T3 "parallel" task-graph note now explicitly distinguishes
> order-independent design from concurrent same-file editing; (4) MINOR — T4
> now pins `interaction.npy` to the existing flattened-triples convention
> instead of leaving the format undecided; (5) MINOR — T4's combination-CTR
> presence check is now a hard gate, re-asserted independently inside both
> T5 and T6's Rust tests. The "Recommended, not blocking" item (verbatim
> `v1.2.10` C++ and the PVC ordering assumption being agent-fetched/inferred,
> not independently re-verified) is now carried explicitly as SPEC §9 risks 7
> and 8, with T5/T6 given explicit fallback guidance if the oracle comparison
> surfaces either issue. See `PLAN-CHECK.md` for the full first-pass review.

> **`[UNVERIFIED: Planner Agent unavailable]`** — no project-installed agent
> named `planner` exists (`.claude/agents/` contains `code-reviewer.md`,
> `plan-checker.md`, `research-agent.md`, `specification-executor.md`,
> `specification-planner.md` — none matches the goal-backward TDD Planner
> Agent this skill calls for). This plan was authored directly by the
> spec-tdd-planner-skill session as the documented fallback, using the same
> goal-backward method (derive observable success conditions from `SPEC.md` →
> identify required artifacts/verification → convert to ordered, dependency-
> checked tasks), and CodeGraph-verified every symbol/file/line cited below.
> It still must pass the independent Plan Checker gate before being called
> ready — see `PLAN-CHECK.md`.

**Phase:** 18 (Extended Feature Importance) · **Slice:** FSTR-01
**Spec:** `./SPEC.md` (specs FIC-01, FIC-02, FIC-03) · **Requirement:** FSTR-01
**Crate:** `cb-model` (+ new oracle fixture in `cb-oracle`) · **Impact:** `local`
**Parity bar:** `1e-5` (CPU, D-12) via `cb_oracle::compare::assert_abs_close`,
except FIC-01 and the hand-built FIC-02/FIC-03 unit tests, which are exact
(pure arithmetic / hand-computed expected values, no floating averaging).

> Executor contract: strict Red → Green → Refactor per task. One spec per TDD
> cycle. **Source/test separation is mandatory** — no inline `#[cfg(test)] mod
> tests { … }` body in production `.rs`; unit tests go in a NEW sibling
> `crates/cb-model/src/fstr_test.rs` wired via the sanctioned `#[cfg(test)]
> #[path = "fstr_test.rs"] mod tests;` mount (mirrors
> `crates/cb-model/src/ctr_data.rs:58-61`); integration/oracle tests live in
> `crates/cb-model/tests/`. **No `unwrap`/`expect`/`panic`/`indexing_slicing`**
> in production (workspace-denied `[VERIFIED: LOCAL Cargo.toml:10-14]`). Every
> float fold routes through `cb_core::sum_f64` (D-08) — this slice introduces
> NO new fold beyond what already exists (redistribution is per-split scalar
> `+=`, not a new reduction), so this constraint mainly guards against
> accidentally adding one. Do **not** mark any task complete during planning.
>
> **Regression discipline (the single highest-severity risk in this slice,
> per SPEC §5 Pitfall / research.md Pitfall 1):** every task that touches
> `interaction()` or `prediction_values_change()` MUST add the new CTR logic
> as an ADDITIONAL step around the existing `float_feature()`/`Some(src_idx)`
> resolution, never by editing the arithmetic inside the existing bit-indexed
> or DFS loop bodies. The existing `fstr_oracle_test.rs` float-only assertions
> (items 1, 2, 4) are the regression gate and MUST be re-run (not skipped)
> after every Green step in T2/T5.

## Validation commands (host CPU; avoids env-red suites)

```
cargo test -p cb-model                     # unit + oracle for this slice
cargo test -p cb-model -p cb-oracle        # + comparator
cargo test -p cb-train                     # confirm tensor_ctr_e2e / multi_permutation_e2e unaffected
cargo clippy -p cb-model --all-targets     # RESTRICTION-LINT GATE (unwrap/expect/panic/indexing denied)
cargo build -p cb-model                   # compile check only — does NOT enforce the clippy restriction lints
```
> **Lint-gate correction (recurring project gotcha):** the workspace
> restriction lints (`unwrap_used/expect_used/panic/indexing_slicing`) are
> **clippy** lints; inert under `cargo build`/`rustc`, enforced ONLY by
> `cargo clippy` `[VERIFIED: LOCAL Cargo.toml:10-14; PROJECT:
> fstr-03-partial-dependence/PLAN-CHECK.md MAJOR #2]`. Pre-existing,
> UNRELATED clippy noise exists elsewhere in the workspace
> (`cb-oracle/src/model_json.rs:161`, `cb-model` `ctr_data_roundtrip_test.rs`)
> — gate NEW code specifically with `cargo clippy -p cb-model --lib --no-deps`
> if the full `--all-targets` run is noisy, but the AUTHORITATIVE full-slice
> gate (T7) is still `--all-targets` scoped to code this slice touches.

Known-red suites to ignore (pre-existing, environmental, unrelated to this
slice): `cb-backend --lib` (CubeCL MLIR), `cb-train monotone_*`,
`catboost-rs-py` (python3.14 link) `[PROJECT: memory
catboost-rs-preexisting-test-failures.md]`. Fixture generation (T4) needs
upstream `catboost==1.2.10` in an offline venv (`uv venv --python 3.12 && uv
pip install catboost==1.2.10 'numpy<2'` — the exact recipe FSTR-03 used
`[PROJECT: fstr-03-partial-dependence/PLAN.md T3]`).

## Task graph (dependencies, not file order)

```
T0 scaffold ──┬─> T1 (FIC-01, unit) ──┬─> T2 (FIC-02 unit) ──┐
              │                        └─> T3 (FIC-03 unit) ──┤
              └─> T4 (fixture gen, enabler, parallel with T1/T2/T3) ─┼─> T5 (FIC-02 oracle) ──┐
                                                                       └─> T6 (FIC-03 oracle) ──┼─> T7 refactor/wire/gate
```
- **Parallelizable:** T1 and T4 both depend only on T0 and can run together
  (T4 is a data-generation task with no Rust-code dependency on T1). T2 and
  T3 are **order-independent in DESIGN** (they touch different functions —
  `interaction()` vs `prediction_values_change()` — with no logical shared
  mutable state; CodeGraph confirms no caller/callee edge between them beyond
  both calling the shared `interaction_add`/`sum_f64` helpers, which are
  read-only reused, not modified) but **NOT safe to literally
  concurrently edit** — both tasks modify the SAME two files
  (`crates/cb-model/src/fstr.rs` and `fstr_test.rs`), just different
  functions within them (plan-checker MINOR-3). **For a single executor:**
  do T2 then T3 (or vice versa) serially, one Red→Green→Refactor cycle fully
  committed before starting the other — "parallel" here describes that
  either order is valid and neither logically depends on the other's
  content, not that they should be edited simultaneously. If genuinely
  split across two agents/sessions, serialize the actual file writes (e.g.
  T2 lands first, T3 rebases onto it) rather than editing concurrently.
- **Serial spine:** T0 → T1 → {T2, T3 serialized} → {T5 needs T2+T4, T6 needs T3+T4} → T7.

---

## T0 — Scaffold: test-file mount + shared-helper placement decision (enabler, no spec)

- **Goal:** create the sibling unit-test file and mount it, and record where
  the new FIC-01 helpers live, so later tasks compile against a stable
  structure.
- **Files:**
  - create empty `crates/cb-model/src/fstr_test.rs`.
  - append to the bottom of `crates/cb-model/src/fstr.rs` the sanctioned
    source/test-separation mount:
    ```rust
    #[cfg(test)]
    #[path = "fstr_test.rs"]
    mod tests;
    ```
    mirroring `crates/cb-model/src/ctr_data.rs:58-61`
    `[VERIFIED: CODEGRAPH crates/cb-model/src/ctr_data.rs:58-61]`. **Without
    this mount, `cargo test -p cb-model` silently runs ZERO new unit tests
    while reporting success** — the exact false-green hazard the FSTR-03 plan
    called out for its own module.
  - **Decision (record in code doc on `fstr.rs`):** FIC-01's
    `cat_feature_count`/`flat_cat_index` helpers are added as private
    functions in `fstr.rs` itself (same module as `feature_count`,
    `convert_to_percents`), NOT a new module — this slice is a small,
    self-contained extension of an existing module, unlike FSTR-03's
    greenfield module. `[INFERRED, consistent with SPEC §4/§7 "Must change:
    crates/cb-model/src/fstr.rs"]`
- **Validation:** `cargo build -p cb-model` compiles; `cargo test -p cb-model`
  picks up (empty) `fstr_test.rs` (confirm via `cargo test -p cb-model
  fstr_test -- --list` or equivalent showing 0 tests found, not a module
  missing error).
- **Completion evidence:** the `#[path]` mount line is present in `fstr.rs`;
  `fstr_test.rs` exists (empty except a doc comment).
- **Rollback:** trivially revertible (no production logic yet).

## T1 — FIC-01 combined flat feature-index resolver (unit)

- **Spec:** FIC-01. **Depends on:** T0.
- **Files:** `crates/cb-model/src/fstr.rs` (new private fns), `fstr_test.rs`
  (new tests).
- **Red** — in `fstr_test.rs`:
  - `cat_feature_count_zero_when_no_ctr_splits` (AT-FIC01a): a `Model` with
    only `ModelSplit::Float` splits in its oblivious trees (reuse the existing
    hand-built-model pattern from `ctr_split_scoring_test.rs`
    `[VERIFIED: CODEGRAPH crates/cb-model/tests/ctr_split_scoring_test.rs — hand-built split literals precedent]`,
    or construct inline) → `cat_feature_count(&model) == 0`.
  - `cat_feature_count_single_ctr_split` (AT-FIC01b): one
    `ModelSplit::Ctr(CtrSplit { projection: TProjection::single(2), .. })` at
    some split position → `cat_feature_count(&model) == 3` (max index `2` `+1`).
  - `cat_feature_count_combination_ctr_uses_max_member` (AT-FIC01c): one
    `ModelSplit::Ctr` with `projection: TProjection::from_features(&[0, 3])`
    → `cat_feature_count(&model) == 4` (NOT `2`, the member count — proves the
    "max index, not length" semantics from SPEC §5 FIC-01).
  - `cat_feature_count_max_across_both_tree_kinds` (AT-FIC01d): a model with a
    `Ctr` split (`cat_features() == [1]`) in an oblivious tree AND a DIFFERENT
    `Ctr` split (`cat_features() == [4]`) in a non-symmetric tree →
    `cat_feature_count(&model) == 5` (max across both, mirroring
    `feature_count`'s existing `oblivious_max.max(non_symmetric_max)` at
    `fstr.rs:99` `[VERIFIED: CODEGRAPH crates/cb-model/src/fstr.rs:91-99]`).
  - `flat_cat_index_is_n_float_plus_local` (unit, arithmetic-only, no `Model`
    needed): `flat_cat_index(5, 0) == 5`, `flat_cat_index(5, 2) == 7`.
  - **Expected initial failure:** `cat_feature_count`/`flat_cat_index` do not
    exist → compile error naming them; record it, then implement.
- **Green:** implement in `fstr.rs`, adjacent to `feature_count`
  (`fstr.rs:75-100`):
  ```rust
  fn cat_feature_count(model: &Model) -> usize {
      let ctr_max = |splits: &mut dyn Iterator<Item = &ModelSplit>| {
          splits
              .filter_map(|s| match s {
                  ModelSplit::Ctr(c) => c.projection.cat_features().iter().copied().max(),
                  ModelSplit::Float(_) => None,
              })
              .map(|m| m + 1)
              .max()
              .unwrap_or(0)
      };
      let oblivious_max = ctr_max(&mut model.oblivious_trees.iter().flat_map(|t| t.splits.iter()));
      let non_symmetric_max = ctr_max(&mut model.non_symmetric_trees.iter().flat_map(|t| t.tree_splits.iter()));
      oblivious_max.max(non_symmetric_max)
  }

  const fn flat_cat_index(n_float: usize, local_cat_index: usize) -> usize {
      n_float + local_cat_index
  }
  ```
  (exact signature; the iterator-closure shape above is illustrative — the
  executor may write it as two straight-line loops instead of a shared
  closure, whichever reads more like the existing `feature_count`'s style;
  behavior, not shape, is what the Red tests pin down.) Use checked/iterator
  access throughout (no `unwrap`/indexing — `.iter().copied().max()` and
  `Option` combinators only).
- **Refactor:** if `feature_count` and `cat_feature_count` share enough
  structure, consider whether a shared private helper is warranted — but do
  NOT touch `feature_count`'s existing body/behavior (regression lock); a
  refactor here is optional and must re-run T1 + the existing
  `fstr_oracle_test.rs` afterward if attempted.
- **Validation:** `cargo test -p cb-model fstr_test`.
- **Completion evidence:** AT-FIC01a–d green; `flat_cat_index` arithmetic
  test green.

## T2 — FIC-02 `interaction()` CTR-aware pairwise attribution (unit)

- **Spec:** FIC-02. **Depends on:** T1. **Parallel with:** T3.
- **Files:** `crates/cb-model/src/fstr.rs` (modify `interaction()` +
  `interaction_accumulate_non_symmetric`/`interaction_dfs`), `fstr_test.rs`.
- **Red** — in `fstr_test.rs`:
  - `interaction_float_only_unchanged` (AT-FIC02a, **regression guard at unit
    level** — the AUTHORITATIVE regression check is re-running the existing
    oracle test, see Validation below, but this unit test gives fast local
    signal): build a small float-only oblivious `Model` (2 float splits, 4
    leaves with distinct values) and assert `interaction(&model)` returns the
    SAME triples this exact model produced BEFORE this task's changes —
    capture the expected output by running the CURRENT (pre-change)
    `interaction()` on this fixed hand-built model once, at Red-authoring
    time, and hard-code the expected `Vec<(usize,usize,f64)>` as the
    assertion target (a golden-value regression test, not a tautology, since
    it is captured from the OLD code and locked before any new code is
    written — record this literal in a comment).
  - `interaction_float_ctr_pair_single_cat` (AT-FIC02b): hand-build a tiny
    oblivious tree with `splits = [Float(feature: 0, ..), Ctr(CtrSplit {
    projection: TProjection::single(0), .. })]` and leaf values chosen so the
    existing bit-indexed `delta` computation yields a known non-zero value
    `d` for pair `(0,1)` → assert `interaction(&model) == [(0, flat_cat_index(1,0), d.abs()_as_100pct)]`,
    i.e. `(0, 1, 100.0)` (`n_float==1` here since only one float split exists,
    so `flat_cat_index(1, 0) == 1`) — proves the basic float×simple-CTR
    cross-product (`side1.len()==1`, no division).
  - `interaction_combination_ctr_splits_equally` (AT-FIC02c): hand-build a
    tiny oblivious tree with `splits = [Float(feature: 0, ..), Ctr(CtrSplit {
    projection: TProjection::from_features(&[0, 1]), .. })]` (a combination
    CTR over TWO cat features at the SECOND split level) → assert the
    resulting pairs are `(0, flat_cat_index(n_float,0), d.abs()/2 *
    100pct-normalized)` AND `(0, flat_cat_index(n_float,1), d.abs()/2 * ...)`
    — i.e. the single pair's `delta` divides equally across BOTH constituent
    cat features (proves the cross-product division rule, SPEC FIC-02 3rd
    bullet).
  - `interaction_non_symmetric_two_ctr_splits_partial_overlap_self_pair`
    (AT-FIC02e — **MANDATORY, added per plan-checker MAJOR-1 finding; do NOT
    defer this to the oracle test (T5) or to T7's reconciliation step, it
    must be Red/Green in THIS task**): hand-build a tiny **non-symmetric**
    `Model` (`step_nodes`/`tree_splits`/`node_id_to_leaf_id`, mirroring the
    hand-built-non-symmetric pattern used elsewhere in the crate, e.g.
    `ctr_split_scoring_test.rs`) with TWO `ModelSplit::Ctr` splits at
    DIFFERENT node depths on the SAME root-to-leaf path:
    `projection: TProjection::from_features(&[0, 1])` at the shallower node,
    `projection: TProjection::from_features(&[1, 2])` at the deeper node on
    the same path (PARTIAL overlap on cat feature `1`). **Construction
    requirement, strengthened per plan-checker pass #2 MAJOR-NEW-1/2 (the
    single generic non-zero-`d` framing used in an earlier draft of this
    test does NOT reliably distinguish a correct implementation from the
    specific "sign-dropping, premature-`abs()`" bug class this arm is prone
    to — see the Green step's arm-specific formula below):** choose
    `node_id_to_leaf_id`/leaf values so that AT LEAST TWO leaves reachable
    below the deeper split contribute to the SAME surviving cross-product
    cell, with leaf values chosen so the CORRECT and INCORRECT
    implementations are numerically distinct.

    **Corrected worked example (plan-checker pass #3 CRITICAL-1: an earlier
    revision of this text used an OPPOSITE-SIGNED leaf pair, `+3.0`/`-1.0`,
    which — hand-derived against `interaction_dfs`'s actual sign convention,
    `fstr.rs:459-466`, left child `-1` / right child `+1` — produces the
    SAME result under the correct and the buggy implementation whenever the
    two leaves have opposite sign: writing `L` for the leaf reached via the
    deeper split's LEFT child and `R` via its RIGHT child, the correct
    magnitude (signed accumulation, one deferred `abs()`) reduces to
    `|R - L|`, while the buggy (sign-dropping, abs-per-leaf) magnitude is
    `|L| + |R|` — these are equal exactly when `L`/`R` have opposite sign,
    which `+3.0`/`-1.0` is. A SAME-SIGNED pair is required instead.**

    Use `L = +3.0` (the deeper split's LEFT child's leaf value) and
    `R = +1.0` (its RIGHT child's leaf value) — same sign, unequal
    magnitude. Then: correct `= |R - L| = |1.0 - 3.0| = 2.0`; incorrect
    (abs-per-leaf-then-sum, sign dropped) `= |L| + |R| = |3.0| + |1.0| =
    4.0` — genuinely DIFFERENT numbers, so a sign-dropping/premature-`abs()`
    implementation fails this test hard, not merely possibly. (This `2.0`
    vs `4.0` distinction is the per-shared-ancestor-path magnitude BEFORE
    the `/(side0.len()*side1.len())` division and percent-normalization the
    Green step also applies — carry both scalings through when hand-computing
    the test's final asserted values.) Re-verify the fully worked final
    numbers (post-division-by-4, post-`convert_to_percents`) directly against
    this corrected `L`/`R` pair — do not reuse any previously-computed number
    from the retracted `+3.0`/`-1.0` example, which was both arithmetically
    wrong (it silently treated `R` as `+1.0` instead of its actual signed
    value in a naive subtraction) and, independent of that arithmetic slip,
    ambiguous by construction (opposite-signed leaves make the correct and
    buggy results coincide regardless). With the corrected same-signed pair,
    the test fails hard under the wrong implementation rather than merely
    "possibly" failing. Hand-compute BOTH
    the intermediate signed per-leaf contributions AND the final expected
    triples using the CORRECT rule (Green step below) before writing the
    assertion. Then apply the standard cross-product/self-pair reasoning on
    top: the cross-product of `{flat_cat(0), flat_cat(1)} ×
    {flat_cat(1), flat_cat(2)}` has 4 cells; the ONE cell where both resolve
    to `flat_cat(1)` is a self-pair and MUST be skipped; the OTHER THREE
    cells (`(flat_cat(0),flat_cat(1))`, `(flat_cat(0),flat_cat(2))`,
    `(flat_cat(1),flat_cat(2))`) each receive their signed-then-abs share
    divided by `(2*2)` (both sides have 2 members) before
    percent-normalization → assert `interaction(&model)` equals exactly this
    hand-computed 3-pair set (order-normalized). This is the ONLY test in
    the slice exercising `interaction_dfs`'s cross-product with a
    `Vec<usize>`-valued path entry AND the ONLY test able to catch a
    sign-dropping regression in this arm — required because the DFS `path`
    type-widening is, by this plan's own admission below, the single most
    structurally invasive change in the slice, and (per plan-checker pass #2)
    T4's oracle fixture is NOT guaranteed to exercise the non-symmetric arm
    at all (see T4's Note on `grow_policy`/oracle-coverage scope below),
    making this unit test the SOLE verification of this arm's correctness in
    the entire slice, not merely a fast local signal.
  - **Expected initial failure:** current code's `continue`-on-`Ctr` skip
    means AT-FIC02b/c/e see EMPTY output (no pairs) instead of the expected
    triples; AT-FIC02a should already pass trivially with no code change
    (it is float-only) — confirm it does BEFORE starting Green, as a sanity
    check that the golden capture is correct.
- **Green:** the two arms have DIFFERENT existing sign/abs() disciplines
  (plan-checker pass #2 MAJOR-NEW-1 — verified via CodeGraph) and therefore
  need TWO DIFFERENT (not one shared) attribution formulas at their
  respective `interaction_add` call sites. **Do not copy one formula
  verbatim into both arms.**
  - **Oblivious arm** (`fstr.rs:292-329`): `delta` (`fstr.rs:312`) is ALREADY
    a fully-aggregated, pre-`abs()` scalar computed once per split-index
    pair — `delta.abs()` is taken immediately at the SAME call site
    (`fstr.rs:326`) with no further cross-call accumulation. Replace the
    current `let (Some(src1), Some(src2)) = (... .float_feature(), ...) else
    { continue }` resolution (`fstr.rs:314-326`) with: for each side,
    compute a `Vec<usize>` of flat indices (`Float(s) => vec![s.feature]`;
    `Ctr(c) => c.projection.cat_features().iter().map(|&x|
    flat_cat_index(n_float, x)).collect()`, `n_float` via the existing
    `feature_count(model)`); cross-product both sides: for `f0` in side0,
    `f1` in side1: skip if `f0 == f1`; normalize order;
    `interaction_add(&mut pairs, &mut sums, a, b, delta.abs() /
    (side0.len() * side1.len()) as f64)`. **`delta.abs()` is correct AS
    WRITTEN here** because `delta` is already finalized before this call.
  - **Non-symmetric DFS arm** (`interaction_dfs`, terminal-accumulation
    inner loop `fstr.rs:421-436`): `delta` here is a SINGLE LEAF's raw value
    (`fstr.rs:419`), and the per-terminal contribution `sign * delta`
    (currently `fstr.rs:433-434`) is accumulated **signed** (not abs()'d)
    into a PER-TREE buffer (`tree_pairs`/`tree_sums`) via
    `interaction_add(tree_pairs, tree_sums, a, b, sign * delta)`; only
    AFTERWARDS, once per tree, does `interaction_accumulate_non_symmetric`
    (`fstr.rs:384`, **UNCHANGED by this task**) take `signed.abs()` and fold
    it into the shared accumulator — this two-stage "accumulate signed
    across every leaf, `abs()` once at the end" discipline is what lets
    same-split-pair contributions from different leaves along the path
    PARTIALLY CANCEL before the final magnitude is taken (mirroring
    upstream's actual cross-derivative estimate), and is the arm's
    load-bearing correctness property. Replace `fstr.rs:448-455`'s
    `let Some(feature_idx) = ... .float_feature() else { return }`
    resolution with the same per-side flat-index expansion as the oblivious
    arm (`path` becomes `Vec<(Vec<usize>, i32)>` — each entry a flat-index
    LIST plus its sign, instead of a single resolved index), and at the
    terminal accumulation (`fstr.rs:421-436`), for EVERY surviving
    (non-self-paired) cross-product cell `(f0, f1)` between the two path
    entries' lists, call
    `interaction_add(tree_pairs, tree_sums, a, b, sign * delta /
    (side0.len() * side1.len()) as f64)` — **STILL SIGNED** (do NOT take
    `.abs()` here), preserving the exact `sign` multiplication the current
    code already does at `fstr.rs:433-434`, so `fstr.rs:384`'s existing
    `signed.abs()` call remains the SOLE, unmodified place magnitude is
    taken for this arm. **A `delta.abs()`-per-leaf-then-sum
    implementation at this call site is WRONG** — it drops `sign` entirely
    and takes `abs()` too early, defeating cross-leaf cancellation (exactly
    the regression class AT-FIC02e's strengthened construction above is
    designed to catch).
  - The DFS `path` element type change described above is the larger of the
    two arm changes — the DFS's `path` element type
    literally changes shape; keep `interaction_dfs`'s recursion/sign/depth-
    bound structure (T-06.6-15) otherwise UNTOUCHED.
  - **Self-pair skip must run PER CROSS-PRODUCT CELL, not once per
    path-entry-pair** (this is exactly what AT-FIC02e checks): when two path
    entries' expansion lists PARTIALLY overlap, only the colliding cell is
    skipped — the non-colliding cells must still each receive their share. A
    single coarse check before the cross-product loop ("skip the whole
    path-entry-pair if ANY member collides") would be WRONG — the equality
    guard belongs INSIDE the innermost `f0 in list0, f1 in list1` loop,
    mirroring the oblivious arm's per-cell placement exactly.
  - **Do NOT edit the `delta`/`signed`/`sum_f64` computation** in either arm,
    and do NOT move or duplicate WHERE `.abs()` is taken (oblivious:
    `fstr.rs:326`, unchanged; non-symmetric: `fstr.rs:384`, unchanged) — only
    the post-delta/post-`sign*delta` attribution/expansion step at each
    arm's `interaction_add` call site changes, per the arm-specific formulas
    above.
- **Refactor:** factor the "expand a `ModelSplit` to its flat-index list"
  logic into ONE shared private helper (e.g. `fn split_flat_indices(split:
  &ModelSplit, n_float: usize) -> Vec<usize>`) used by BOTH the oblivious and
  non-symmetric arms, so the expansion rule is defined once (reduces the risk
  of the two arms drifting). Re-run T1/T2 unit tests after this refactor.
- **Validation:**
  ```
  cargo test -p cb-model fstr_test
  cargo test -p cb-model --test fstr_oracle_test        # REGRESSION: items 2, 4 (float-only interaction) must stay green, UNCHANGED
  ```
- **Completion evidence:** AT-FIC02a/b/c/e (including the mandatory
  non-symmetric AT-FIC02e) green; existing `fstr_oracle_test.rs` interaction
  assertions (items 2 and 4) still pass with NO change to their expected
  values. **T2 is not complete without AT-FIC02e green** — it is not
  optional or deferrable to T5/T7.

## T3 — FIC-03 `prediction_values_change()` CTR-aware redistribution (unit)

- **Spec:** FIC-03. **Depends on:** T1. **Parallel with:** T2.
- **Files:** `crates/cb-model/src/fstr.rs` (modify `prediction_values_change()`
  itself — its output-vector ALLOCATION, see below — plus
  `pvc_accumulate_oblivious` + `pvc_accumulate_non_symmetric`), `fstr_test.rs`.
- **Mandatory allocation-widening step (plan-checker pass #3 MAJOR-1 — do
  NOT skip; without this, AT-FIC03b/c cannot pass):** `feature_count()`
  itself is explicitly locked unchanged by T1's regression discipline, so
  `prediction_values_change()`'s own body (`fstr.rs:123-138`,
  `let n_features = feature_count(model); let mut res = vec![0.0_f64;
  n_features];`) is the ONLY place `res`'s length is fixed, and it is NOT
  touched by modifying the two accumulate helpers alone (they receive
  `res: &mut [f64]` as an already-sized slice — they cannot widen it).
  Change this line to `let mut res = vec![0.0_f64; n_features +
  cat_feature_count(model)];` (reusing T1's `cat_feature_count` helper).
  Without this, every `res.get_mut(flat_cat_index(n_float, c))` call in the
  Green step below silently returns `None` (checked access, no panic) and
  the ENTIRE CTR redistribution is dropped with no error signal — e.g.
  AT-FIC03b's hand-built model (one `Ctr` split, no float splits,
  `n_features == 0`) would allocate a ZERO-LENGTH `res`, silently discarding
  `dif` entirely and returning `Vec::new()`.
- **Red** — in `fstr_test.rs`:
  - `pvc_float_only_unchanged` (AT-FIC03a, unit golden-value regression guard,
    same discipline as AT-FIC02a — capture the OLD code's output on a fixed
    hand-built float-only model before changing anything).
  - `pvc_simple_ctr_full_redistribution` (AT-FIC03b): hand-build a tiny
    oblivious tree with ONE `Ctr(CtrSplit { projection: TProjection::single(0),
    .. })` split and leaf weights/values chosen so `dif` (the existing
    `(val1-avrg)²·c1 + (val2-avrg)²·c2` computation) is a known non-zero value
    → assert `res[flat_cat_index(n_float, 0)] == dif` (100%, since `k==1`) and
    every other slot is `0.0`, both BEFORE and AFTER `convert_to_percents`
    (post-conversion the single non-zero slot is exactly `100.0`).
  - `pvc_combination_ctr_equal_split` (AT-FIC03c): hand-build a tiny oblivious
    tree with ONE `Ctr(CtrSplit { projection: TProjection::from_features(&[0,
    1]), .. })` split → assert `res[flat_cat_index(n_float,0)] ==
    res[flat_cat_index(n_float,1)] == dif / 2.0` (equal-split redistribution,
    SPEC FIC-03 2nd bullet — NO cross-product here, unlike FIC-02, since PVC
    redistributes a SINGLE split's own effect, not a pair).
  - **Expected initial failure:** AT-FIC03b/c see an all-zero `res` (current
    code skips `Ctr` splits entirely) instead of the expected non-zero slots.
- **Green:** in `pvc_accumulate_oblivious` (`fstr.rs:143-174`) and
  `pvc_accumulate_non_symmetric` (`fstr.rs:198-260`), replace the current
  `let Some(src_idx) = split.float_feature() else { continue }` /
  `let Some(feature_idx) = ... .float_feature() else { continue }` resolution
  with:
  - `Float(s) => { addEffect = dif; target_slots = vec![s.feature]; }`
  - `Ctr(c) => { let members = c.projection.cat_features(); addEffect = dif /
    members.len() as f64; target_slots = members.iter().map(|&x|
    flat_cat_index(n_float, x)).collect(); }`
  - then `for slot in target_slots { if let Some(s) = res.get_mut(slot) { *s +=
    addEffect; } }` (checked `.get_mut`, no raw indexing, matching the existing
    style at `fstr.rs:169-171`/`255-257`).
  - Guard: `members.is_empty()` (a `Ctr` split with an empty projection —
    should not occur from `Model::from_trained`, but defensively) contributes
    nothing (division-by-zero avoided the same way `total_effect == 0.0` is
    guarded elsewhere, T-04-04-03 discipline) rather than panicking.
  - **Do NOT edit the `count1`/`count2`/`avrg`/`dif` computation** in either
    arm — only the post-`dif` attribution step.
- **Refactor:** if T2's `split_flat_indices(split, n_float) -> Vec<usize>`
  helper (from T2's Refactor step) is available, REUSE it here for the
  `target_slots` computation rather than re-deriving the expansion logic —
  this is the natural point where FIC-02 and FIC-03 share code. If T2 has not
  landed this refactor yet (parallel execution), implement the equivalent
  logic locally and reconcile/de-duplicate in T7.
- **Validation:**
  ```
  cargo test -p cb-model fstr_test
  cargo test -p cb-model --test fstr_oracle_test        # REGRESSION: items 1, 4 (float-only PVC) must stay green, UNCHANGED
  ```
- **Completion evidence:** AT-FIC03a/b/c green; existing float-only PVC
  assertions unchanged.

## T4 — New oracle fixture: mixed float + categorical model (enabler artifact)

- **Spec:** enables FIC-02/FIC-03 oracle acceptance (AT-FIC02d, AT-FIC03d).
  **Depends on:** T0 (naming only — no Rust-code dependency). **Parallel
  with:** T1, T2, T3. **Blocking for:** T5, T6.
- **Files (new):** `crates/cb-oracle/fixtures/fstr_ctr/`
  - `gen_fixtures.py` — modeled on `tensor_ctr_e2e/gen_fixtures.py` (cat-only
    precedent) and `advanced_fstr/gen_fixtures.py` (fstr-ground-truth-dumping
    precedent) `[VERIFIED: LOCAL crates/cb-oracle/fixtures/tensor_ctr_e2e/,
    advanced_fstr/gen_fixtures.py]`. Construct a small `Pool`/DataFrame with
    **float columns FIRST, then categorical columns** (SPEC §4 load-bearing
    invariant — e.g. 2 float columns at positions `[0,1]`, 2 categorical
    columns at positions `[2,3]`, `cat_features=[2,3]`), train a
    `CatBoostClassifier` with the SAME isolating-parameter discipline as
    `tensor_ctr_e2e` (`boosting_type=Plain, one_hot_max_size=1,
    max_ctr_complexity=2, simple_ctr=["Borders:Prior=0.5"],
    combinations_ctr=["Borders:Prior=0.5"], depth=2, iterations=5,
    learning_rate=0.1, l2_leaf_reg=3.0, random_seed=0, thread_count=1,
    loss_function="Logloss"` `[VERIFIED: LOCAL
    crates/cb-train/tests/tensor_ctr_e2e_oracle_test.rs:81-119 tensor_ctr_params]`)
    so BOTH a simple CTR (cat feature alone) and a combination CTR (both cat
    features combined, since `max_ctr_complexity=2` and there are exactly 2
    cat features) are plausible in the trained trees.
  - **`grow_policy` / non-symmetric-arm oracle-coverage scope decision
    (plan-checker pass #2, Required Revision #3 — resolved, not left
    unstated):** `tensor_ctr_params()`'s `grow_policy` field uses
    `cb_train::grow_policy_default()`, which is `EGrowPolicy::SymmetricTree`
    `[VERIFIED: CODEGRAPH crates/cb-train/src/boosting.rs:128-135
    grow_policy_default() -> EGrowPolicy::SymmetricTree]` — so T4's fixture,
    which adapts this exact param set, produces ONLY OBLIVIOUS (symmetric)
    trees. **This is a deliberate, explicitly recorded scope decision, not
    an oversight:** T5/T6's oracle tests (AT-FIC02d/AT-FIC03d) therefore
    exercise the OBLIVIOUS arm's CTR logic end-to-end only. The
    NON-SYMMETRIC arm's CTR logic (both FIC-02's `interaction_dfs` and
    FIC-03's `pvc_accumulate_non_symmetric`) has NO oracle-level backstop in
    this slice — its ONLY verification is T2's/T3's hand-built unit tests
    (AT-FIC02e for FIC-02, which plan-checker pass #2 required to be
    strengthened into a discriminating test for exactly this reason; FIC-03's
    non-symmetric arm has no cross-leaf-sign-cancellation risk analogous to
    FIC-02's — per plan-checker pass #2's CodeGraph evidence, `dif` is always
    `>= 0`, so its existing oblivious-only unit tests AT-FIC03b/c are lower
    risk, though still not oracle-backed for the non-symmetric arm
    specifically). **Rationale for accepting this gap rather than building a
    second `grow_policy=Lossguide` fixture:** training a non-symmetric model
    with a GENUINE combination CTR is itself empirical/not
    parameter-guaranteed (same caveat as the HARD GATE below, compounded);
    adding a second fixture family purely for oracle coverage of a code path
    already covered by a rigorous, hand-computed, discriminating unit test
    (AT-FIC02e) is judged disproportionate scope growth for this slice. If a
    future hardening pass wants oracle-level non-symmetric CTR coverage, that
    is a follow-up task, not blocking for this SPEC's acceptance criteria
    (§6 already scopes FIC-02/03's oracle rows to "mixed float+CTR fixture,"
    without arm-specificity, consistent with this decision).
  - Save `model.cbm` + `model.json`; dump `X_float.npy` ([N,2] float64) and
    `X_cat.npy` ([N,2] int32, stringified via the SAME
    `stringify_int_category` convention `tensor_ctr_e2e` uses
    `[VERIFIED: CODEGRAPH crates/cb-data::stringify_int_category; crates/cb-train/tests/tensor_ctr_e2e_oracle_test.rs:66-77]`).
  - Call `model.get_feature_importance(type='Interaction')` and
    `model.get_feature_importance(type='PredictionValuesChange', data=pool)`
    (PVC needs `data=` per Context7-confirmed signature differences between
    importance types `[PROJECT: research.md Standard Stack row, Context7
    finding]`); dump `interaction.npy` **PINNED to flattened `[i, j, score]`
    triples — the SAME format the existing `fstr_oracle_test.rs` fixtures
    already use per that file's own docstring** (plan-checker MINOR-4: do
    NOT leave this as an open format decision; reusing the established
    convention avoids a second, inconsistent on-disk shape in the same
    crate) and `prediction_values_change.npy` (length `n_float + n_cat`
    array, in the SPEC §4 flat-index order: floats `[0..n_float)` then cats
    `[n_float..n_float+n_cat)` — this ordering should already match because
    the Pool itself was built floats-then-cats, so upstream's own external
    index numbering coincides, per SPEC §4; **this specific ordering
    assumption is flagged as an open, oracle-adjudicated risk in SPEC §9
    risk 8** — if T6's comparison fails on ordering alone (values present but
    permuted), re-sort the dumped array by feature index at generation time
    here, rather than treating it as a Rust-side bug).
  - `config.json` — `catboost_version:"1.2.10"`, `n_float:2`, `n_cat:2`,
    `cat_features:[2,3]`, params, seeds, and an explicit note recording the
    float-before-cat column order invariant this fixture relies on (so a
    future reader does not accidentally regenerate it with a different
    column order and silently break the SPEC §4 assumption).
  - **HARD GATE (plan-checker MINOR-5 — promoted from a soft note): the
    generated model MUST contain at least one combination-CTR split
    (`projection.cat_features().len() >= 2`).** Inspect the trained model's
    actual splits (via the saved `model.json`) to CONFIRM this — if
    `max_ctr_complexity=2` with only 2 cat features does not empirically
    produce one, adjust cardinalities/`iterations`/`depth` until it does
    (mirrors `tensor_ctr`'s own note: "two cat features each above
    one_hot_max_size so the combination is genuinely formed"
    `[VERIFIED: LOCAL crates/cb-oracle/fixtures/tensor_ctr/config.json]`).
    This is empirical, not guaranteed by parameters alone — budget iteration
    time. **T4 is NOT complete if this gate fails** — a fixture with only
    simple/single-feature CTR splits would leave the cross-product/
    equal-split (`k>1`) redistribution logic COMPLETELY oracle-unverified
    while appearing to pass (AT-FIC02d/AT-FIC03d would still run, just never
    exercise that logic) — this must not be allowed to happen silently.
    **This gate is re-asserted a SECOND time, independently, inside the
    Rust test itself in T5/T6** (not just at generation time) — see T5/T6
    below — so a future accidental fixture regeneration that loses the
    combination split fails the Rust test loudly, not just the Python
    generation script.
- **Red/Green/Refactor:** N/A (data artifact).
- **Validation:** `python crates/cb-oracle/fixtures/fstr_ctr/gen_fixtures.py`
  under the `uv`-managed `catboost==1.2.10` venv; all `.npy` files load,
  finite, correct dtype/shape; `model.json`'s splits contain at least one
  `OnlineCtr` split whose projection spans 2 cat features (manual/scripted
  check against the saved `model.json` — the HARD GATE above).
- **Completion evidence:** committed `gen_fixtures.py`, `model.cbm`,
  `model.json`, `X_float.npy`, `X_cat.npy`, `interaction.npy`,
  `prediction_values_change.npy`, `config.json`.
- **Rollback note:** additive; no existing fixture is touched.

## T5 — FIC-02 oracle: `interaction()` on the mixed fixture

- **Spec:** FIC-02 (AT-FIC02d). **Depends on:** T2, T4.
- **Files:** new `crates/cb-model/tests/fstr_ctr_oracle_test.rs` (dedicated
  file — kept separate from `fstr_oracle_test.rs` per SPEC §7's "plan-time
  choice," chosen here so the existing file's 4-scenario docstring and its
  regression role stay untouched and easy to diff against).
- **Red:**
  - top-of-file `#![allow(clippy::unwrap_used, clippy::expect_used,
    clippy::panic, clippy::indexing_slicing)]`, `const TOL: f64 = 1e-5;`, a
    `fixture(rel)` path helper — exact pattern from `fstr_oracle_test.rs:18-31`
    `[VERIFIED: CODEGRAPH crates/cb-model/tests/fstr_oracle_test.rs:18-31]`.
  - **Settled fact, not a to-be-verified item (plan-checker MINOR-2):**
    `crates/cb-model/src/cbm.rs::reconstruct_model` (called by
    `load_cbm`/`decode_cbm`) unconditionally builds ONLY `ModelSplit::Float`
    splits and unconditionally sets `ctr_data: None` on every return path —
    it never reads `TModelTrees::CtrFeatures` at all. **`load_cbm` CANNOT
    produce a CTR-bearing `Model` on this branch today; this is not
    something to "verify against the T4 fixture," it is already true from
    reading the current source.** Go straight to training the SAME model
    in-process via `cb_train::train_cat` with the IDENTICAL
    params/seed/data `gen_fixtures.py` used (the `tensor_ctr_e2e_oracle_test.rs`
    pattern, proven independent of the unmerged `feat/23-ctr-model-loading`
    branch `[VERIFIED: CODEGRAPH crates/cb-train/tests/tensor_ctr_e2e_oracle_test.rs]`),
    lift via `Model::from_trained(...).with_ctr_data(...)`.
  - `interaction_matches_upstream_on_mixed_ctr_model` (AT-FIC02d): using the
    `train_cat`-lifted model above, **first assert the hard-gate invariant
    from T4 holds on THIS loaded model** — `model` contains `>= 1` `CtrSplit`
    with `projection.cat_features().len() >= 2` (re-asserted here per
    plan-checker's Potential Bugs finding, independent of T4's own Python-side
    check, so a future accidental fixture regeneration that loses the
    combination split fails THIS Rust test loudly rather than silently
    degrading coverage) — then call `interaction(&model)` and compare
    against `interaction.npy` at `1e-5` (matching the fixture's pinned
    flattened-triples format from T4).
  - **Expected initial failure:** BEFORE T2's Green lands this would fail with
    empty/wrong pairs; by the time T5 runs T2 is already green, so the
    expected initial failure here is specifically a MISMATCH against the
    upstream fixture (validates the algorithm end-to-end, not just the
    hand-built unit shape) — record the actual diff if the first attempt
    does not match (see Risk note below).
- **Green:** no NEW production code expected if T2 is correct — this task is
  the end-to-end oracle proof. If it fails, the failure localizes to either
  (a) the T4 fixture not actually exercising a combination CTR (revisit T4),
  or (b) a genuine algorithm gap in T2 (revisit T2's cross-product/division
  logic against the SPEC §1/§5 FIC-02 rules).
- **Refactor:** none expected beyond what T2 already did.
- **Validation:** `cargo test -p cb-model --test fstr_ctr_oracle_test interaction`.
- **Completion evidence:** AT-FIC02d green at `1e-5`.
- **Risk note (from research.md Common Pitfall #2/#5):** if the model-loading
  path (`load_cbm` vs `train_cat`→`from_trained`) produces a model whose
  split STRUCTURE differs from what the Python `get_feature_importance` call
  saw (e.g. a different tree order or leaf indexing), the comparison would
  fail for a reason UNRELATED to FIC-02's correctness — verify predictions
  match first (`predict_raw` vs a dumped `predictions.npy`, reusing the
  `tensor_ctr_e2e` pattern) as a sanity gate before trusting an interaction
  mismatch as an algorithm bug.

## T6 — FIC-03 oracle: `prediction_values_change()` on the mixed fixture

- **Spec:** FIC-03 (AT-FIC03d). **Depends on:** T3, T4.
- **Files:** same `crates/cb-model/tests/fstr_ctr_oracle_test.rs` (add to the
  file T5 created).
- **Red:**
  - `pvc_matches_upstream_on_mixed_ctr_model` (AT-FIC03d): using the SAME
    `train_cat`-lifted model from T5 (`load_cbm` is not viable here either,
    per T5's settled-fact note), **re-assert the hard-gate combination-CTR
    invariant** (same check as T5's, independently in this test function —
    do not rely on T5 having already run it, since test functions may run in
    any order), call `prediction_values_change(&model)`, compare against
    `prediction_values_change.npy` at `1e-5`. Also assert the result sums to
    100.0 (the existing in-env normalization gate pattern from
    `fstr_oracle_test.rs` item 1 `[VERIFIED: CODEGRAPH
    crates/cb-model/tests/fstr_oracle_test.rs docstring item 1]`). **If this
    fails on ordering alone (right values, wrong positions),** revisit SPEC
    §9 risk 8 (the PVC Python-facing array ordering assumption) before
    assuming a Rust-side bug — the fix in that case is re-sorting the T4
    fixture's dumped array by feature index, not changing `fstr.rs`.
  - **Expected initial failure:** a genuine mismatch (localize per T5's risk
    note) before T3's logic is confirmed correct end-to-end.
- **Green:** no new production code expected if T3 is correct; same
  localization discipline as T5.
- **Refactor:** none expected.
- **Validation:** `cargo test -p cb-model --test fstr_ctr_oracle_test pvc`.
- **Completion evidence:** AT-FIC03d green at `1e-5`; sum-to-100 holds.

## T7 — Refactor, de-duplication, full-slice gate

- **Depends on:** T5, T6.
- **Steps:**
  - Reconcile T2's and T3's `split_flat_indices`-shaped helper if both arrived
    at slightly different local implementations during parallel execution —
    converge on ONE shared private helper in `fstr.rs`, re-run T1–T6 after.
  - Re-run the FULL existing `fstr_oracle_test.rs` (all 4 items) and
    `advanced_fstr_oracle_test.rs` to confirm zero float-only regressions.
  - Re-run `cargo test -p cb-train` to confirm `tensor_ctr_e2e_oracle_test.rs`
    / `multi_permutation_e2e_oracle_test.rs` are unaffected (this slice reuses
    their pattern but modifies nothing in `cb-train`).
  - Update `fstr.rs`'s module-level doc comment (`fstr.rs:1-53`) to describe
    the now-CTR-aware `interaction`/`prediction_values_change` behavior and
    cite the `v1.2.10`-pinned source functions (`CalcFeatureInteraction`,
    `CalcRegularFeatureEffect`) alongside the existing `feature_str.h`/
    `calc_fstr.cpp` citations already there.
  - Grep the new/changed code for `.unwrap()`/`.expect()`/`panic!`/raw `[]`
    indexing as a smell check (the AUTHORITATIVE gate is clippy, below).
- **Validation (full slice):**
  ```
  cargo clippy -p cb-model --all-targets    # restriction-lint gate (authoritative for NEW code)
  cargo build -p cb-model
  cargo test -p cb-model
  cargo test -p cb-model -p cb-oracle
  cargo test -p cb-train
  ```
- **Completion evidence:** FIC-01/02/03 acceptance tests all green; existing
  float-only fstr oracle assertions unchanged; restriction lints clean on new
  code; `cb-train` CTR e2e suites unaffected. Then (bookkeeping, outside TDD)
  flip the FSTR-01 requirement checkbox in the git-recovered
  `.planning/REQUIREMENTS.md` (off-tree — confirm canonical revision first,
  per SPEC §10).

## Traceability (task → spec → acceptance)

| Task | Spec | Acceptance tests | Kind |
|------|------|------------------|------|
| T0 | (enabler) | compiles, test file mounted | — |
| T1 | FIC-01 | AT-FIC01a/b/c/d + flat_cat_index arithmetic | unit |
| T2 | FIC-02 | AT-FIC02a/b/c/e (e = mandatory non-symmetric DFS test, plan-checker MAJOR-1) | unit |
| T3 | FIC-03 | AT-FIC03a/b/c | unit |
| T4 | (enabler) | fixture `.npy`/`.cbm`/`.json` loadable, combination CTR confirmed present | artifact |
| T5 | FIC-02 | AT-FIC02d | oracle |
| T6 | FIC-03 | AT-FIC03d | oracle |
| T7 | all | full-slice green, zero regressions | gate |

Every SPEC acceptance behavior (§6 roll-up) has a Red task; every task
references ≥1 spec ID.
