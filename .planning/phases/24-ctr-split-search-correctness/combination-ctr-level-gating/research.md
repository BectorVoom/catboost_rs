# Phase 24 Research: CTR-aware split search wrongly offers combination CTRs before upstream would (`cb-train` correctness bugfix)

## Research Summary

- **Phase goal**: fix a confirmed training-time divergence in `cb-train`'s
  CTR-aware oblivious tree search: combination (multi-feature, "Tensor") CTR
  split candidates are offered to the greedy search at every tree level,
  including the root, when real CatBoost's `AddTreeCtrs` only ever makes a
  combination CTR eligible once the tree already has at least one chosen
  split (float, one-hot, or CTR) to extend. This lets Rust's tree grower pick
  a combination CTR the real algorithm could never have offered at that
  point, producing a genuinely different tree structure (not just different
  attribution/importance numbers).
- **Recommended approach**: make combination-CTR candidate generation
  level-dependent and tree-structure-dependent, mirroring
  `AddTreeCtrs` (`catboost-master/catboost/private/libs/algo/greedy_tensor_search.cpp:503-568`):
  at each level, only emit a combination-CTR candidate that extends a
  projection built from the CURRENT tree's already-chosen float bins /
  one-hot features / already-used CTR projections by exactly one more
  CTR-eligible categorical feature — never from the empty base. Simple
  (single-feature) CTRs are unaffected: upstream's `AddSimpleCtrs` makes them
  unconditionally available at every level regardless of tree structure, and
  Rust already treats them that way correctly.
- **Most important constraint**: the existing `cat_feature_weight`,
  `build_ctr_aware_histogram`, `select_level_ctr_aware` scoring math is
  CONFIRMED byte-for-byte consistent with upstream's `GetCatFeatureWeight`
  (`greedy_tensor_search.cpp:926-950`) — do not touch the scoring formulas.
  The defect is entirely in *candidate-set generation*, upstream of scoring.
- **Highest-risk findings**: (1) this is a structural fix (candidate
  generation becomes level/tree-structure-aware instead of a static
  pre-tree-wide list), touching `candidates.rs`, `tree.rs`'s CTR-aware search,
  and `boosting.rs`'s CTR materialization pipeline; (2) the bug is
  data-dependent in observable impact — it is very likely ALSO latently
  present in the currently-passing categorical-only `tensor_ctr_e2e` /
  `multi_permutation_e2e` oracle fixtures (both use `max_ctr_complexity=2`,
  2 cat features, depth=2), just not visible because the illegitimate
  early combination candidate does not happen to out-score the legitimate
  winner in that specific data — a fix must be re-verified against those
  fixtures, not just the new one.

## Phase Requirements

### In Scope
- Restrict combination/tensor CTR (`is_simple == false`, `projection` with
  `cat_features().len() >= 2`) candidate availability in the CTR-aware
  oblivious structure search (`crates/cb-train/src/tree.rs`
  `greedy_tensor_search_oblivious_with_ctr` / `select_level_ctr_aware`) to
  match upstream `AddTreeCtrs`'s incremental, tree-structure-dependent
  eligibility rule.
- Preserve unconditional, level-independent availability of SIMPLE
  (single-feature) CTR candidates (matches `AddSimpleCtrs`, unaffected by
  this bug).
- Re-verify (not just re-run) `tensor_ctr_e2e_oracle_test.rs`,
  `multi_permutation_e2e_oracle_test.rs`, and `ctr_split_scoring_test.rs`
  after the fix — they must still pass, and if their expected structure
  changes, that must be understood and re-derived from real upstream output,
  not patched to whatever Rust newly produces.
- Fix must make the new `fstr_ctr_oracle_test.rs` (currently RED, all 3
  tests) pass to ≤1e-5, INCLUDING its sanity gate (predictions), not just
  interaction/PVC.

### Acceptance Criteria
- `cargo test -p cb-model --test fstr_ctr_oracle_test` passes all 3 tests
  (`fstr_ctr_predictions_sanity_gate`, `interaction_matches_upstream_on_mixed_ctr_model`,
  `pvc_matches_upstream_on_mixed_ctr_model`) at ≤1e-5.
- `cargo test -p cb-train --test tensor_ctr_e2e_oracle_test`,
  `--test multi_permutation_e2e_oracle_test`, and
  `--test ctr_split_scoring_test` still pass.
- No change to any FLOAT-only oracle test's tree structure (the fix must be
  a no-op whenever `cat_columns` supplies no CTR-eligible feature, i.e.
  `has_ctr == false` — the numeric-only `train`/`train_with_eval_sets` paths
  are byte-identical, per existing `D-6.6-05`-style discipline used elsewhere
  in this codebase).
- Simple-CTR-only configs (`max_ctr_complexity == 1`, or a single CTR-eligible
  cat feature) are unaffected (no combination is ever possible in that config
  regardless of the fix).

### Out of Scope
- Any change to `cat_feature_weight` / `build_ctr_aware_histogram` /
  `score_candidate_ctr_aware` scoring math — CONFIRMED correct against
  upstream, not implicated.
- Ordered-boosting + CTR interaction (`boosting_type=Ordered` combined with
  `has_ctr`) — flagged as an existing, separate, PRE-EXISTING code-path
  question (see Open Questions), not part of this bugfix's diagnosis.
- FSTR-01's own interaction()/prediction_values_change() implementation —
  those functions are unaffected; the new oracle test fails upstream of them,
  at model training / the sanity-gate prediction check.

### Open or Conflicting Requirements
- No SPEC/PLAN currently exists for this bugfix (it was discovered as a side
  effect of FSTR-01 fixture work, not planned). This research recommends a
  new phase (see "Planning Guidance" / path reasoning below) rather than
  folding it into phase 18's FSTR-01 artifacts, since the defect lives
  entirely in `cb-train`, not in feature-importance code.

## Project Constraints
- `unwrap()` strictly prohibited in production (CLAUDE.md); the affected
  functions already follow this (checked `.get()` access, `CbResult` returns).
- Source/test separation is mandatory — any new unit tests for the
  level-gating logic must go in a dedicated `_test.rs` file
  (`crates/cb-train/src/candidates_test.rs` and/or a tree.rs-adjacent test
  file), never embedded `#[cfg(test)] mod tests` in `tree.rs`/`candidates.rs`.
- Oracle parity bar: ≤1e-5 against real `catboost==1.2.10` output
  (`crates/cb-oracle/fixtures/fstr_ctr/`, generated via
  `crates/cb-oracle/fixtures/fstr_ctr/gen_fixtures.py`, already committed —
  do NOT regenerate it per the task's own instruction and per the project's
  documented "CTR fixtures are frozen because catboost quantization is
  run-to-run nondeterministic" convention
  `[PROJECT: user memory ctr-model-loading.md]`.
- Existing code convention: prefer `.get(..)` / checked indexing over
  `unwrap`/panicking indexing (`clippy::indexing_slicing` denied in several
  test files via explicit `#![allow(...)]` — production code must not need
  the allow).

## Current Project Architecture

### Relevant subsystems and boundaries
- `crates/cb-train/src/candidates.rs` — one-hot/CTR encoding-path routing
  (`route_categorical`, `route_column`) and STATIC tensor/combination CTR
  candidate enumeration (`tensor_ctr_candidates`, line 179) — currently emits
  ALL SimpleCtr + CombinationCtr projections up to `max_ctr_complexity` from
  cat-feature cardinalities alone, with **no notion of tree level or which
  features are already chosen** `[CODEGRAPH: crates/cb-train/src/candidates.rs:159-201]`.
- `crates/cb-train/src/tree.rs` — the CTR-aware oblivious structure search:
  `CtrAwareSplit` (enum, line 2238), `build_ctr_aware_histogram` (line 2306),
  `cat_feature_weight` (line 2416), `select_level_ctr_aware` (line 2527),
  `greedy_tensor_search_oblivious_with_ctr` (line 2669)
  `[CODEGRAPH: crates/cb-train/src/tree.rs:2233-2757]`. The level loop in
  `greedy_tensor_search_oblivious_with_ctr` (line 2684-2699) calls
  `select_level_ctr_aware` with the **SAME, full `ctr_features` slice at every
  level** — no level-dependent filtering by `chosen` splits so far, other than
  the `already_used` check that only exempts an ALREADY-CHOSEN projection from
  the `cat_feature_weight` penalty (not from ELIGIBILITY).
- `crates/cb-train/src/boosting.rs` — `train_cat` (line 2145) → `train_inner`
  (line 2259) builds `ctr_candidates = tensor_ctr_candidates(...)` (line 2705)
  **once per tree/iteration, outside the level loop**, materializes ALL of
  their columns up front (`structure_fold_columns`, lines 2843-2896), and
  feeds the FULL resulting `iter_ctr_features` slice into
  `greedy_tensor_search_oblivious_with_ctr` (line 3900) for the WHOLE tree
  (line 3892 `else if has_ctr { greedy_tensor_search_oblivious_with_ctr(...) }`)
  `[CODEGRAPH: crates/cb-train/src/boosting.rs:2680-2930,3890-3916]`.

### Existing data/control flow (current, buggy)
1. Per tree: compute cat-feature cardinalities → `tensor_ctr_candidates`
   (ALL simple + combination projections, static) → materialize a
   `CtrFeatureColumn` for EVERY candidate, for the WHOLE tree.
2. `greedy_tensor_search_oblivious_with_ctr` loops `depth` levels; at EVERY
   level, `select_level_ctr_aware` scores float candidates (level-scoped,
   correct) THEN scores every one of the (tree-wide, static) CTR columns —
   including multi-feature combinations — with NO regard for what the tree
   has already split on so far in `chosen`.
3. Strict first-wins picks the best-scoring candidate regardless of kind.

### Upstream's actual data/control flow (confirmed via vendored source)
`catboost-master/catboost/private/libs/algo/greedy_tensor_search.cpp`:
- `GreedyTensorSearchOblivious` (line 1198) loops `curDepth` 0..MaxDepth;
  EACH iteration calls `SelectFeaturesForScoring(data, currentSplitTree, fold, ctx)`
  (line 1217-1218) — i.e. candidate generation happens **fresh every level**,
  parameterized by `currentSplitTree` (the splits chosen so far THIS tree).
- `SelectDatasetFeaturesForScoring` (line 1000) calls,每 level:
  `AddFloatFeatures`, `AddOneHotFeatures`, `AddSimpleCtrs` (line 1025,
  unconditional — one simple CTR candidate per CTR-eligible cat feature,
  EVERY level, independent of `currentSplitTree`), and, only
  `if (currentSplitTree.Defined())`, `AddTreeCtrs` (lines 1028-1034).
- `AddTreeCtrs` (line 503) is the combination-CTR generator:
  ```
  TProjection binAndOneHotFeaturesTree;
  binAndOneHotFeaturesTree.BinFeatures = currentTree.GetBinFeatures();
  binAndOneHotFeaturesTree.OneHotFeatures = currentTree.GetOneHotFeatures();
  seenProj.insert(binAndOneHotFeaturesTree);
  for (const auto& ctr : currentTree.GetUsedCtrs()) { seenProj.insert(ctr.Projection); }
  for (const auto& baseProj : seenProj) {
      if (baseProj.IsEmpty()) { continue; }             // <-- KEY GATE
      // ... extend baseProj by ONE more cat feature, gated by
      //     MaxTensorComplexity / IsRedundant / addedProjHash dedup
  }
  ```
  `[CODEGRAPH/VERIFIED: catboost-master/catboost/private/libs/algo/greedy_tensor_search.cpp:503-568]`.
  At the ROOT (level 0), `currentTree` has NO splits and NO used CTRs yet, so
  `binAndOneHotFeaturesTree` is EMPTY and the ONLY `seenProj` entry
  (`baseProj.IsEmpty()`) is explicitly **skipped** — `AddTreeCtrs` emits ZERO
  combination candidates at the root. A genuine ≥2-feature combination CTR
  can therefore only ever appear as a candidate starting from the level
  AFTER the tree has already picked at least one split of any kind (float
  bin, one-hot, or a previously-chosen simple/combination CTR), and even then
  only as "that chosen thing" extended by ONE more categorical feature.
- Independent corroboration from the SAME fixture the executor built:
  `crates/cb-oracle/fixtures/fstr_ctr/gen_fixtures.py`'s own comment states
  it needed `depth=3, iterations=15` (NOT `tensor_ctr_e2e`'s
  `depth=2, iterations=5`) because "a plain additive label needs a much
  stronger combination signal before the grower ever selects a GENUINE
  2-feature combination CTR over two independent simple CTRs" — this is
  exactly what `AddTreeCtrs`'s incremental-extension gate predicts: a
  combination needs the tree to already be at least 1 level deep before it
  can appear at all, so shallow trees structurally cannot contain one unless
  depth allows a second level.

### Existing reusable implementations
- `l2_split_score` / `cosine_split_score` / `split_score`
  (`crates/cb-compute/src/score.rs:49,73`; `crates/cb-train/src/tree.rs:58`) —
  CONFIRMED correct, shared scalar scorer for both float and CTR candidates;
  reuse unchanged.
- `cat_feature_weight` (`crates/cb-train/src/tree.rs:2416`) — CONFIRMED
  byte-for-byte formula match with upstream `GetCatFeatureWeight`
  (`greedy_tensor_search.cpp:926-950`, same `pow(1 + count/maxCount, -reg)`);
  reuse unchanged.
- `TProjection` (`crates/cb-train/src/projection.rs`), `enumerate_projections`
  — reuse for whatever incremental "extend by one feature" enumeration the
  fix needs; do not hand-roll a new projection type.
- `route_categorical` / `learn_set_cardinality` (`candidates.rs`) — reuse
  unchanged (one-hot vs CTR routing is NOT implicated in this bug).

### Current conventions and patterns
- Structure-search CTR columns are the IDENTITY-fold materialization; leaf
  VALUES are reassigned later over the averaging-fold columns
  (`boosting.rs`, Plan-05-13-style comments) — this split is NOT implicated
  by this bug and must be preserved.
- `LevelKind` (float vs CTR per level) and `CtrSplitSpec` recording — reuse
  unchanged; the fix only changes WHICH candidates are considered per level,
  not how a winning candidate is recorded.

## Standard Stack
No new library/framework/dependency is implicated — this is a pure
algorithm-correctness fix confined to `cb-train`'s Rust source. No Context7 /
external library lookup applies.

## Dependency Analysis
- Direct: `cb-compute` (`EScoreFunction`, `l2_split_score`, `cosine_split_score`)
  — unaffected, reused as-is.
- Internal: `cb-train::projection` (`TProjection`, `enumerate_projections`),
  `cb-train::ctr` (`CtrFeatureColumn`, `materialize_ctr_feature`) — the fix
  will likely need to call `materialize_ctr_feature` at NEW points (per
  level, per newly-eligible base projection) rather than only once up front;
  this changes the CALL PATTERN (more calls, smaller batches) but not the
  function's contract.
- No new crate dependency, no version change.

## Recommended Architecture and Implementation Pattern

### Prescribed approach
Mirror `AddTreeCtrs` directly rather than inventing a different mechanism:

1. In `candidates.rs`, add a function that, given (a) the CTR-eligible cat
   feature list, (b) the SET of "seen" base projections at the CURRENT level
   (derived from the tree's chosen float features + chosen one-hot features
   + chosen CTR projections so far — the `AnySplit`/`CtrAwareSplit` history
   already tracked as `chosen: &[CtrAwareSplit]` in
   `select_level_ctr_aware`), and (c) `max_ctr_complexity`, emits ONLY the
   combination candidates that extend one of those base projections by
   exactly one more CTR-eligible cat feature (deduplicated, complexity-capped
   — this is the direct analogue of `AddTreeCtrs`'s `seenProj`/`addedProjHash`
   loop). At level 0 with `chosen` empty, this returns an empty set (matching
   upstream's `baseProj.IsEmpty()` skip).
2. Keep `tensor_ctr_candidates`'s SIMPLE-CTR half as the level-independent,
   always-available candidate set — matches `AddSimpleCtrs` and is NOT part
   of the bug.
3. In `tree.rs`'s `select_level_ctr_aware` / `greedy_tensor_search_oblivious_with_ctr`,
   feed simple CTRs (static, tree-wide) and combination CTRs (recomputed or
   filtered PER LEVEL from `chosen`) separately, materializing NEW
   combination columns as they become eligible (this likely means
   `materialize_ctr_feature` gets called lazily per level for newly-eligible
   projections, rather than for every combinatorially-possible projection up
   front in `boosting.rs`).
4. Preserve the FLOAT-then-SIMPLE-CTR-then-COMBINATION-CTR enumeration order
   (or whatever exact order upstream's `AddFloatFeatures`→`AddOneHotFeatures`→
   `AddSimpleCtrs`→`AddTreeCtrs` establishes) so the strict first-wins
   tie-break (`> best`, never `>=`) stays consistent with the project's
   existing Pitfall-1 discipline.

### Component responsibilities
- `candidates.rs`: pure candidate-set math (which projections are eligible,
  given cardinalities + what's already chosen) — no CTR value computation.
- `ctr::materialize_ctr_feature`: unchanged; called at new call sites/timing.
- `tree.rs`: scoring + selection, unchanged formulas, changed candidate
  SOURCE (level-scoped instead of tree-wide-static).
- `boosting.rs`: orchestration — needs to thread "features used so far in
  THIS tree" into the per-level CTR materialization instead of precomputing
  everything before the level loop starts.

### Data and control flow (target)
Per tree: for `level in 0..depth`: (float candidates, level-scoped, as today)
+ (simple CTR candidates, tree-wide/static, as today) + (combination CTR
candidates, freshly derived from `chosen` so far THIS level, materialized
on demand) → score all → strict-first-wins pick → append to `chosen`.

### Error, security, and failure behavior
No new fallibility surface beyond what `materialize_ctr_feature` /
`learn_set_cardinality` already return (`CbResult`); the level-scoped
combination generation is pure, infallible set arithmetic over already-
validated cardinalities — should not introduce new `CbError` variants.

## Project Impact Scope

### Must Change
- `crates/cb-train/src/candidates.rs` — `tensor_ctr_candidates` (or a new
  sibling function) must become level/tree-structure-aware for the
  combination-CTR half. **Reason**: root cause. **Downstream**: every caller
  of `tensor_ctr_candidates` (currently only `boosting.rs:2705`).
- `crates/cb-train/src/tree.rs` — `select_level_ctr_aware` (2527) and
  `greedy_tensor_search_oblivious_with_ctr` (2669) must accept/derive
  per-level-eligible combination CTR columns instead of one static slice.
  **Reason**: root cause, structure-search half. **Downstream**:
  `boosting.rs` callers (7 call sites per CodeGraph), `ctr_split_scoring_test.rs`.
- `crates/cb-train/src/boosting.rs` — the `has_ctr` branch (~3892-3916) and
  the up-front `structure_fold_columns` materialization (2843-2929) need to
  either (a) materialize combination columns lazily per level, or (b)
  materialize a superset up front but pass level-eligibility metadata into
  `tree.rs` so it can FILTER rather than re-derive. **Reason**: orchestration
  wiring for the fix. **Downstream**: `train_cat`, `train`, all CTR-bearing
  training call sites.

### May Change
- `crates/cb-train/src/ctr/ctr_feature.rs` (`CtrFeatureColumn`,
  `materialize_ctr_feature`) — call PATTERN changes (more, smaller,
  level-triggered calls) even if the function signature itself is untouched.
- Any exported test-only re-export list in `crates/cb-train/src/lib.rs` if
  a new candidate-generation function needs to be exposed to
  `ctr_split_scoring_test.rs`/`candidates_test.rs`.

### Verification Only
- `crates/cb-compute/src/score.rs` (`l2_split_score`, `cosine_split_score`)
  — confirmed correct; re-run existing tests (`score_test.rs`) as a
  regression fence, no code change expected.
- `crates/cb-train/src/tree.rs`'s `cat_feature_weight`, `build_ctr_aware_histogram`,
  `score_candidate_ctr_aware` — confirmed correct; keep as-is, cover with the
  SAME tests to prove the fix didn't perturb them.
- `crates/cb-model` apply path (`predict_raw_cat`, `CtrSplit` evaluation) —
  not implicated; the bug is upstream of a fully-built model, so apply-path
  tests are a downstream verification signal only (the sanity-gate test in
  `fstr_ctr_oracle_test.rs` already exercises this).

### Explicitly Out of Scope
- `catboost-master/` vendored C++ — read-only oracle reference, never
  modified by this project (per repo convention: it is the upstream source
  tree, not owned code).
- FSTR-01's `interaction()` / `prediction_values_change()` implementations
  (`crates/cb-model/src/fstr*.rs`, wherever they live) — unaffected;
  failing only because the model they're fed is wrong.
- ONNX export (`cb-model::export::onnx`), CTR model loading (`decode_cbm`,
  phase 23) — unaffected by this fix; both consume an already-built `Model`,
  not the training-time search.

## Do Not Hand-Roll
- Reuse `TProjection` / `enumerate_projections` (`crates/cb-train/src/projection.rs`)
  for any new "extend by one feature" projection arithmetic — do not
  reimplement projection hashing/equality.
- Reuse `materialize_ctr_feature` / `bake_ctr_table` for CTR value
  computation — do not duplicate the online/ordered CTR math.
- Reuse `split_score` / `l2_split_score` / `cosine_split_score` — confirmed
  correct, do not fork a new scorer for the level-gated candidates.
- Reuse the existing strict first-wins (`> best`, never `>=`) discipline
  already established elsewhere in this file (`select_level_ctr_aware`,
  `select_best_candidate`) — do not introduce a different tie-break rule for
  the newly-gated combination path.

## Common Pitfalls and Risks

- **Trigger**: assuming `tensor_ctr_candidates`'s doc comment ("Emit the
  tensor / combination CTR candidates for a tree level") already describes
  correct per-level behavior, when the ACTUAL implementation and its ONE
  caller treat it as a tree-GLOBAL static list.
  **Consequence**: a planner could believe this function is already correct
  and look elsewhere for the bug (as the triggering executor did, blaming
  `select_level_ctr_aware`/`cat_feature_weight` scoring instead).
  **Prevention**: treat the doc comment as ASPIRATIONAL/misleading, not as
  evidence of current behavior; verify against the actual call site
  (`boosting.rs:2705`, called ONCE outside any level loop).
  **Verification**: `cargo test -p cb-train candidates::` plus a new test
  asserting combination-candidate emission depends on a non-empty "already
  chosen" set.

- **Trigger**: fixing combination-CTR eligibility naively by just checking
  `chosen.is_empty()` and returning nothing at level 0, without replicating
  the full `AddTreeCtrs` "extend ANY seen base projection (bin+one-hot tree
  projection OR any already-used CTR projection) by ONE feature" logic.
  **Consequence**: could under- or over-restrict at levels ≥ 1 (e.g. a tree
  that chose a FLOAT split at level 0 should still unlock combinations of
  (that float's categorical siblings — NO, only CATEGORICAL features
  combine; the "BinFeatures"/"OneHotFeatures" base is about which cat
  features are ALREADY implicated via one-hot splits, not floats) — the
  precise semantics of `TProjection::BinFeatures`/`OneHotFeatures` need
  re-derivation from the (unvendored) `split.h`/`TSplitTree` — see Open
  Questions.
  **Prevention**: since `TSplitTree`/`split.h` is NOT present in this
  vendored tree (`grep` found nothing), a planner MUST NOT guess its exact
  semantics — fetch `catboost/private/libs/algo/split.h` (or the
  `TSplitTree` definition) from the `v1.2.10` GitHub tag before designing
  the exact "base projection" construction.
  **Verification**: re-derive against a hand-traced small example (e.g. a
  depth-2, 3-cat-feature synthetic case) before trusting the port.

- **Trigger**: assuming the currently-passing `tensor_ctr_e2e_oracle_test.rs`
  / `multi_permutation_e2e_oracle_test.rs` prove the current (buggy) static
  candidate generation is fine.
  **Consequence**: a fix could be rejected or under-tested because "it
  already passes," when in fact those fixtures simply never trigger the
  observable divergence (their combination CTR doesn't out-score the
  legitimate winner in that specific toy data — the bug is latent, not
  absent).
  **Prevention**: explicitly re-verify, after the fix, that
  the exact same tree structures / leaf values are STILL produced for those
  fixtures (i.e. the fix is provably a no-op for them), rather than treating
  "still passes" as sufficient — ideally by asserting the exact set of CTR
  candidates considered per level pre- and post-fix are identical for those
  fixtures' configs (2 cat features, `max_ctr_complexity=2`, depth=2).
  **Verification**: `cargo test -p cb-train --test tensor_ctr_e2e_oracle_test --test multi_permutation_e2e_oracle_test` plus a new unit test in `candidates_test.rs`/`ctr_split_scoring_test.rs` proving level-0 combination-candidate count is 0 in a synthetic 2-cat-feature scenario mirroring those fixtures.

- **Trigger**: `boosting.rs`'s `else if has_ctr { greedy_tensor_search_oblivious_with_ctr(...) }` branch is checked BEFORE the `ordered_learning_perm` branch (line ~3892 vs ~3918), meaning a hypothetical `boosting_type=Ordered` config with CTR features would currently take the (non-ordered) CTR-aware oblivious path unconditionally, never the ordered path.
  **Consequence**: out of scope for THIS bugfix, but a planner touching
  `has_ctr` branch precedence for this fix must not accidentally change this
  existing (possibly separately buggy, possibly intentional) precedence.
  **Prevention**: treat as untouched — verify no existing test exercises
  `Ordered` boosting + CTR together (none of the found `_e2e_` fixtures set
  `boosting_type: Ordered`; all use `Plain`) before assuming it's safe to
  leave alone.
  **Verification**: `grep -n "boosting_type" crates/cb-train/tests/*ctr*` — confirm no Ordered+CTR test exists (already checked: `tensor_ctr_e2e`/`multi_permutation_e2e` both use `Plain`).

- **Trigger**: the `EScoreFunction` used for the CTR-aware search is
  whatever `params.score_function` resolves to (default `Cosine`); the
  executor reports trying BOTH `Cosine` and `L2` and seeing the SAME bug.
  **Consequence**: a planner might waste time trying more score functions or
  suspecting `multi_dim_split_score`/`SolarL2`/etc. — none of these are
  implicated; the bug reproduces identically for any score function because
  it is upstream of scoring (candidate SET generation, not candidate
  SCORING).
  **Prevention**: do not re-litigate score-function correctness; it is
  already independently confirmed consistent with upstream (formula-level
  comparison above).
  **Verification**: N/A — already ruled out by the executor's own
  Cosine/L2 A-B test plus this research's formula-level comparison.

## Testing and Verification Strategy

### Unit tests
- New tests in `crates/cb-train/src/candidates_test.rs` asserting: (a) at
  an "empty chosen" state, combination-CTR candidate emission is empty
  (mirrors `AddTreeCtrs`'s `baseProj.IsEmpty()` skip); (b) once a base
  projection is "seen" (a chosen float/one-hot/CTR), extending it by one
  more CTR-eligible feature yields the expected new combination candidates,
  capped by `max_ctr_complexity`/`MaxTensorComplexity`.
- New/updated tests in `crates/cb-train/tests/ctr_split_scoring_test.rs`:
  add a case with a genuine multi-feature `CtrFeatureColumn` (`is_simple ==
  false`, `TProjection` spanning ≥2 features) competing against a float
  candidate at depth=1/level=0, asserting the COMBINATION MUST NOT be
  offered/cannot win at level 0 (regression fence for exactly this bug),
  distinct from the existing single-feature-only tests
  (`ctr_candidate_wins_over_uninformative_float` etc., which remain valid
  and unchanged since they exercise the always-legitimate SIMPLE-CTR case).

### Integration/contract tests
- `cargo test -p cb-model --test fstr_ctr_oracle_test` — the target RED
  test; must go fully GREEN (`fstr_ctr_predictions_sanity_gate`,
  `interaction_matches_upstream_on_mixed_ctr_model`,
  `pvc_matches_upstream_on_mixed_ctr_model`), ≤1e-5.

### End-to-end/regression tests
- `cargo test -p cb-train --test tensor_ctr_e2e_oracle_test`
- `cargo test -p cb-train --test multi_permutation_e2e_oracle_test`
- `cargo test -p cb-train --test ctr_split_scoring_test`
- Full `cargo test -p cb-train` and `cargo test -p cb-model` as a broad
  regression fence (many other oracle tests exist in both crates; the fix
  touches shared orchestration code in `boosting.rs`).

### Migration/data checks
- None — no schema/model-format change; this is a training-time algorithm
  fix, not a serialization change. No fixture regeneration (fixtures are
  frozen/committed, generated from real `catboost==1.2.10`, and
  nondeterministic to regenerate per project convention
  `[PROJECT: user memory ctr-model-loading.md]`).

### Security/performance/operational checks
- Performance: moving from "materialize all combination columns once per
  tree" to "materialize combination columns lazily per level, per newly-
  eligible base projection" could change allocation patterns — the docstring
  at `build_ctr_aware_histogram` already notes a PERF-02 concern (avoiding
  per-candidate rescans); a planner should verify the fix doesn't reintroduce
  an `O(candidates × n)` rescan per level that the histogram-based scan
  (`score_candidate_ctr_aware`, PERF-02) was specifically built to avoid.

### Exact project commands
```
cargo test -p cb-model --test fstr_ctr_oracle_test -- --nocapture
cargo test -p cb-train --test tensor_ctr_e2e_oracle_test
cargo test -p cb-train --test multi_permutation_e2e_oracle_test
cargo test -p cb-train --test ctr_split_scoring_test
cargo test -p cb-train candidates::
cargo test -p cb-train
cargo test -p cb-model
```

## Planning Guidance

- **Suggested work boundaries / ordering**:
  1. First fetch/confirm `TSplitTree`/`split.h`'s exact `GetBinFeatures`/
     `GetOneHotFeatures`/`GetUsedCtrs`/`IsRedundant` semantics from the
     `v1.2.10` GitHub tag (NOT present in this vendored tree) — this is a
     hard prerequisite spike before writing the level-gating logic, not
     optional research.
  2. Implement the level/tree-structure-aware combination-CTR candidate
     generator in `candidates.rs` (pure, unit-testable in isolation from
     `tree.rs`/`boosting.rs`).
  3. Wire it into `tree.rs`'s CTR-aware search (`select_level_ctr_aware`,
     `greedy_tensor_search_oblivious_with_ctr`), keeping simple-CTR handling
     untouched.
  4. Wire `boosting.rs`'s materialization to the new per-level eligibility
     (lazy or superset-plus-filter — planner's choice, but must not
     reintroduce an O(n) per-candidate rescan).
  5. Re-run the FULL existing CTR test suite before declaring done — this
     fix's risk is entirely in "did I accidentally change tensor_ctr_e2e's
     answer," not in the new fixture.
- **Decisions the planner must preserve**: strict first-wins (`> best`,
  never `>=`) tie-break; FLOAT-then-CTR enumeration order (exact ordering
  within CTR — simple-then-combination — should be re-derived from upstream's
  `AddSimpleCtrs`-then-`AddTreeCtrs` call order, confirmed above); the
  existing structure-fold vs averaging-fold CTR materialization split
  (unrelated to this bug, must not be touched).
- **Items that require a spike or user decision before implementation**:
  the exact `TSplitTree` base-projection construction (needs the
  `v1.2.10` `split.h` source, absent from this vendored tree) — flagged as
  a hard blocker for a fully upstream-faithful implementation, not just a
  nice-to-have.
- **Requirement ID convention**: this codebase's categorical/CTR training
  work uses an `ORD-0x` prefix in code comments (`ORD-01` permutation,
  `ORD-02` ordered boosting multi-tree, `ORD-03` one-hot threshold (per
  `candidates.rs`/generator README), `ORD-04` one-hot encoding-path routing,
  `ORD-05` tensor/combination CTR structure search — the exact subsystem this
  bug lives in). A new SPEC for this fix should extend the lineage as
  **`ORD-06`** (tree-structure-scoped combination-CTR eligibility), keeping
  continuity with the `ORD-05` tensor-CTR work it corrects.

## Open Questions

- **`TSplitTree`/`split.h` semantics** (BLOCKING for an upstream-faithful
  port): `catboost-master` does not vendor `catboost/private/libs/algo/split.h`
  (confirmed absent via `grep -rln "class TSplitTree" catboost-master` →
  empty, and `find -iname "*split_tree*"` → empty). The planner must fetch
  this from the `v1.2.10` GitHub tag (`WebFetch`/`WebSearch`, not yet done in
  this research pass) to precisely define `GetBinFeatures()` /
  `GetOneHotFeatures()` / `GetUsedCtrs()` / `TProjection::IsRedundant()`
  before implementing the "seen base projections" construction.
- **Ordered boosting + CTR precedence** (`boosting.rs`'s `else if has_ctr`
  branch checked before the `ordered_learning_perm` branch): appears to be a
  PRE-EXISTING, separate design question (does this codebase intend to
  support Ordered boosting with CTR features at all, and if so, is the
  greedy_tensor_search_oblivious_with_ctr call meant to internally dispatch
  on ordering, or is Ordered+CTR simply unimplemented/unsupported today?).
  Not resolved by this research pass; flagged so the bugfix plan does not
  accidentally paper over it while touching the same `has_ctr` branch.
- **Exact scope of `AddOneHotFeatures`'s interaction with this bug**: this
  research confirmed the COMBINATION-CTR gating bug; it did not separately
  re-verify whether Rust's one-hot candidate generation (`route_categorical`/
  `EncodingPath::OneHot`) has an analogous level-dependency gap (upstream's
  `AddOneHotFeatures` also runs every level, but is NOT itself
  tree-structure-dependent per the vendored source read in this pass — only
  `AddTreeCtrs` has the incremental-extension gate). Treated as OUT OF SCOPE
  based on current evidence, but not exhaustively proven absent.
- **Whether the fix changes `tensor_ctr_e2e`/`multi_permutation_e2e`'s
  expected tree structure at all**: this research's evidence strongly
  suggests NO (the bug is latent there), but this was NOT executed
  end-to-end (no code change was made in this research-only pass) — the
  planner/implementer must verify this empirically, not assume it.

## Sources

- Project documents inspected:
  - `CLAUDE.md` (project instructions, root) — coding/testing conventions.
  - `crates/cb-oracle/fixtures/fstr_ctr/config.json`,
    `crates/cb-oracle/fixtures/fstr_ctr/gen_fixtures.py` — fixture
    parameters and the "depth=3/iterations=15 empirically required" note
    that independently corroborates the root cause.
  - `crates/cb-model/tests/fstr_ctr_oracle_test.rs` — the RED oracle test.
  - `.planning/phases/23-ctr-model-loading/cbm-ctr-load/SPEC.md`,
    `.planning/phases/18-extended-feature-importance/*/SPEC.md` — SPEC/PLAN
    conventions (front-matter with `requirement_ids`, `status`, `phase`
    fields; `SOURCES.md` sibling file pattern in the FSTR-01 directory).
- CodeGraph queries and relevant symbols/paths:
  - `select_level_ctr_aware`, `build_ctr_aware_histogram`, `cat_feature_weight`
    (`crates/cb-train/src/tree.rs:2527,2306,2416`) — confirmed to exist
    exactly as the executor hypothesized, but confirmed NOT the root cause.
  - `greedy_tensor_search_oblivious_with_ctr` (`crates/cb-train/src/tree.rs:2669`),
    its 7 callers in `boosting.rs`/`lib.rs`, and its test coverage
    (`ctr_split_scoring_test.rs`).
  - `train_cat` → `train_inner` → `greedy_tensor_search_oblivious_with_ctr`
    call path (`crates/cb-train/src/boosting.rs:2145,2259`).
  - `tensor_ctr_candidates` (`crates/cb-train/src/candidates.rs:179`), its
    ONE caller (`boosting.rs:2705`).
  - `l2_split_score`, `cosine_split_score`, `split_score`, `MINIMAL_SCORE`
    (`crates/cb-compute/src/score.rs:32,49,73`; `crates/cb-train/src/tree.rs:58`).
- Local files/manifests/command output:
  - `[VERIFIED: cargo test -p cb-model --test fstr_ctr_oracle_test -- --nocapture]`
    — all 3 tests currently FAIL; first failure even at the sanity-gate
    prediction check (`prediction[0] diverges: got -0.061..., want 0.313...`),
    confirming the divergence is in the TRAINED MODEL structure, not in
    FSTR-01's attribution math.
  - `[VERIFIED: grep -rn "class TSplitTree" catboost-master]` → no results
    (TSplitTree definition not vendored in this repo).
  - `[VERIFIED: grep depth/iterations in crates/cb-train/tests/tensor_ctr_e2e_oracle_test.rs, multi_permutation_e2e_oracle_test.rs]`
    → both use `depth: 2`, `max_ctr_complexity: 2`, `boosting_type: Plain`.
  - `[VERIFIED: Read crates/cb-train/tests/ctr_split_scoring_test.rs]` →
    confirmed all 3 CTR-vs-float scoring unit tests use
    `TProjection::single(0)` (single-feature/"Simple" CTR only), never a
    multi-feature combination — these tests do not exercise, and are not put
    at risk by, the confirmed bug.
- Official web documentation:
  - None fetched in this pass (Context7/WebSearch/WebFetch not invoked —
    the vendored `catboost-master/` source was sufficient for the
    scoring-formula and `AddTreeCtrs` control-flow comparison; the ONE gap
    — `TSplitTree`/`split.h`'s exact class definition — is flagged as an
    Open Question requiring a `v1.2.10`-tag WebFetch before implementation,
    not resolved here).

## Confidence Assessment

- **HIGH**:
  - The oracle test currently fails, and fails even at the model-prediction
    sanity gate (not just interaction/PVC) — directly reproduced.
  - `select_level_ctr_aware`/`build_ctr_aware_histogram`/`cat_feature_weight`
    exist exactly as named by the executor, at the cited lines.
  - `cat_feature_weight`'s formula is byte-for-byte consistent with
    upstream's `GetCatFeatureWeight` — ruled out as the root cause.
  - `tensor_ctr_candidates` is called ONCE per tree, OUTSIDE any level loop,
    and its full static result is fed identically to every level of
    `greedy_tensor_search_oblivious_with_ctr` — directly verified via
    CodeGraph + Read of both `candidates.rs` and `boosting.rs`.
  - Upstream's `AddTreeCtrs` explicitly skips the empty base projection,
    meaning NO combination CTR is ever a candidate at a tree's root — directly
    read from the vendored `greedy_tensor_search.cpp:503-568`.
  - This mismatch (Rust: combination always available; upstream: combination
    root-gated behind an already-chosen split) is a sufficient, well-evidenced
    explanation for "Rust picks a combination CTR at the root; upstream picks
    float," independent of `score_function` choice (a candidate-set bug, not
    a scoring-formula bug).
- **MEDIUM**:
  - That `tensor_ctr_e2e`/`multi_permutation_e2e` currently pass DESPITE this
    same bug being present in their code path (the "latent, not absent"
    claim) — logically implied by the confirmed defect plus those fixtures'
    shared config shape, but not empirically re-verified by running an
    instrumented diff of candidate sets in this research pass.
  - The precise shape of the fix (lazy per-level materialization vs.
    superset-plus-filter) — a reasonable recommendation, not the only viable
    design.
- **LOW**:
  - The exact `TSplitTree`/`split.h` semantics (`GetBinFeatures`,
    `GetOneHotFeatures`, `GetUsedCtrs`, `IsRedundant`) — NOT available in
    this vendored tree; the control-flow-level understanding (base
    projection = already-chosen structure, extended by one feature) is
    solid, but exact field-level semantics need a `v1.2.10`-tag fetch before
    implementation.
  - Whether Ordered-boosting + CTR interaction is a separate, real bug or
    simply unimplemented/unreachable in current configs — flagged as an open
    question, not resolved.
