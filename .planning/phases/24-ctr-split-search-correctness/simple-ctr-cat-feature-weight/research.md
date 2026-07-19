# Phase 24 Research: ORD-07 — Simple-CTR-vs-Float Score Comparison Diverges at Depth >= 3 Under Non-Zero `model_size_reg`

## Research Summary

- **Phase goal (candidate, not yet spec'd):** After ORD-06 (combination-CTR
  level-gating) landed, `crates/cb-oracle/fixtures/fstr_ctr/`'s oracle test
  (`fstr_ctr_oracle_test.rs`) STILL fails — now at tree0/level2 (3rd split),
  not level0. Rust's CTR-aware greedy search picks `Float(feature=1,
  border=0.4915)` where real `catboost==1.2.10` picks a **simple** (single
  cat-feature) CTR split (`projection=[0]`, `ctr_type=Borders`, 3rd/last
  border ≈ 12.0). This is CONFIRMED to be a DIFFERENT bug from ORD-06:
  neither candidate is a combination projection, so
  `combination_ctr_eligible`/`eligible_max_bucket_count` are never invoked
  for this comparison at all `[VERIFIED: LOCAL instrumented trace, see
  §"Direct reproduction" below]`.
- **Recommended approach (partial — root cause NARROWED, not fully pinned):**
  the divergence is caused by `cat_feature_weight`'s (`tree.rs:2416`)
  multiplicative penalty being applied to the simple-CTR candidate's raw
  score at a magnitude that is narrowly (≈8%) too strong at this specific
  level/candidate — the formula and its documented inputs match upstream's
  vendored `GetCatFeatureWeight` (`greedy_tensor_search.cpp:926-950`)
  byte-for-byte (already established by ORD-06), yet the WEIGHTED comparison
  still disagrees with upstream's real chosen tree. Disabling the weight
  entirely does **not** fix it either (it overcorrects, flipping level 1 —
  where upstream legitimately wants Float — to CTR as well). This rules out
  "wrong sign" / "should not apply at all" as the defect and points to either
  (a) a subtly wrong `count`/`max_bucket_count` INPUT specific to depth >= 3,
  or (b) the interaction between `cat_feature_weight` and the raw
  `score_candidate_ctr_aware` value at a partition with >= 4 leaves
  (`chosen.len() == 2`), a configuration NO existing regression/oracle test
  exercises together with non-zero `model_size_reg` and float+CTR mixing.
  **A dedicated spike is required before this can be written into a
  SPEC/PLAN with a settled root cause** (see Open Questions).
- **Most important constraints:** the `fstr_ctr` fixture is FROZEN (do not
  regenerate); the fix must not alter the byte-identical ORD-06 fix already
  landed (uncommitted in the working tree as of this research); any fix must
  keep `ctr_split_scoring_test.rs`'s existing `model_size_reg = 0.0` tests
  passing UNCHANGED and add new coverage for `model_size_reg != 0` (currently
  ZERO test coverage of that combination exists anywhere in the repo).
- **Highest-risk findings:**
  1. `feature_penalties_calcer.h/.cpp`, `fold.h`, and `online_ctr.h` (the
     upstream headers that define `GetSplitFeatureWeight`, `TFold`, and
     `TOnlineCtrUniqValuesCounts`/`GetUniqueValueCountForType`/
     `GetMaxUniqueValueCount`) are **NOT vendored** in
     `catboost-master/` in this repo — confirmed absent via `find`/`grep`
     `[VERIFIED: LOCAL find/grep, empty results]`. The ORD-06 SPEC's own
     citations to `feature_penalties_calcer.cpp:191-205` could not be
     re-verified locally by this research; they must have come from a prior
     WebFetch not repeated here. This research had to WebFetch
     `online_ctr.cpp`/`online_ctr.h` from the `v1.2.10` GitHub tag instead,
     and GitHub's code-search UI required a login this environment does not
     have, so some finer details (the full `TOnlineCtrUniqValuesCounts`
     struct, `GetSplitFeatureWeight`'s body) remain **unverified** — flagged
     under Open Questions.
  2. This is the **first** scenario in the whole test suite combining (a)
     float + CTR mixed features, (b) `depth >= 3` (a partition with `>= 4`
     leaves reachable by a CTR-aware level search), and (c) non-zero
     `model_size_reg`, simultaneously. Any ONE of these three novel
     combinations (not just the previously-suspected `model_size_reg`
     weighting) could be implicated; this research narrows it to "very
     likely involves `cat_feature_weight`'s effective magnitude" but does not
     conclusively rule out the raw per-level histogram/score computation
     itself at `chosen.len() == 2`.

## Phase Requirements

### In Scope

- Diagnose (this research) and — in a follow-up SPEC/PLAN, likely `ORD-07`
  — fix the exact point at which a **simple** CTR candidate's score is
  computed/weighted such that it loses to a Float candidate at a
  non-root level, when upstream's real trained model shows the CTR
  candidate should win.
- Preserve ORD-06's already-landed combination-CTR level-gating fix
  byte-for-byte (do not touch `combination_ctr_eligible` /
  `eligible_max_bucket_count`'s logic for combination projections; a fix
  here concerns the SIMPLE-CTR-vs-Float comparison, which ORD-06 explicitly
  left "unconditionally available every level, unchanged").
- Re-verify `fstr_ctr_oracle_test.rs` (all 3 tests), plus
  `tensor_ctr_e2e_oracle_test.rs`, `multi_permutation_e2e_oracle_test.rs`,
  and `ctr_split_scoring_test.rs` (all currently GREEN, `model_size_reg=0.0`
  or categorical-only) after any fix.

### Acceptance Criteria

- `cargo test -p cb-model --test fstr_ctr_oracle_test` passes all 3 tests at
  `<= 1e-5` (currently RED — sanity gate fails first with a ~0.5 raw
  prediction gap, confirmed by direct reproduction below).
- Tree0's split sequence, re-derived by an instrumented debug test (or a new
  permanent unit test), matches `model.json`'s upstream splits EXACTLY at
  every level (currently matches at levels 0–1, diverges at level 2).
- No regression in any currently-GREEN CTR/float test
  (`tensor_ctr_e2e_oracle_test.rs`, `multi_permutation_e2e_oracle_test.rs`,
  `ctr_split_scoring_test.rs`, `tree_test.rs`'s `combination_ctr_eligible`
  suite).

### Out of Scope

- ORD-06's combination-CTR eligibility gating (already fixed, verified
  correct and unaffected by this bug — see "Direct reproduction" below).
- FSTR-01's `interaction()`/`prediction_values_change()` themselves
  (downstream consumers; blocked only because the underlying trained model
  is wrong).
- `Rsm` (random subspace sampling), ordered-boosting+CTR precedence, one-hot
  candidate generation — same non-goals ORD-06 already established, unaltered
  by this bug.
- Regenerating `crates/cb-oracle/fixtures/fstr_ctr/` (FROZEN, project
  convention).

### Open or Conflicting Requirements

- **The exact root-cause mechanism inside `cat_feature_weight`'s
  application is NOT fully pinned down by this research** (see Open
  Questions #1–#3). A Planner MUST NOT proceed straight to a code fix without
  either (a) a live instrumented `catboost==1.2.10` run logging
  `GetCatFeatureWeight`'s `count`/`maxFeatureValueCount`/`score`/`gain` for
  this exact fixture at tree0/level2, or (b) further local experimentation
  (e.g., bisecting whether the RAW score or the WEIGHT input is wrong) beyond
  what this research already did.

## Project Constraints

- Restriction lints: `cargo clippy -p cb-train --all-targets` enforces
  `unwrap_used`/`expect_used`/`panic`/`indexing_slicing` project-wide
  `[VERIFIED: LOCAL crates/cb-train/tree.rs uses `.get(..)` patterns
  throughout, per CLAUDE.md]`.
- Source/test separation: no `#[cfg(test)] mod tests` embedded in production
  files; new unit tests belong in `crates/cb-train/src/tree_test.rs` (already
  mounted via `#[path = "tree_test.rs"]` at `tree.rs:92`) or a new sibling
  integration test under `crates/cb-train/tests/`
  `[VERIFIED: LOCAL CLAUDE.md, crates/cb-train/src/tree.rs:92]`.
- `crates/cb-oracle/fixtures/fstr_ctr/` and all other CTR fixtures are
  FROZEN — upstream quantization is run-to-run nondeterministic; never
  regenerate `[PROJECT: memory ctr-model-loading.md "CTR fixtures are
  frozen"; PROJECT: .planning/phases/24.../combination-ctr-level-gating/SPEC.md
  §2 non-goals]`.
- Oracle parity bar: `<= 1e-5` `[PROJECT: CLAUDE.md]`.
- ORD-06's fix (present, uncommitted in the working tree at research time)
  must not be altered: `combination_ctr_eligible` (`tree.rs:2548`),
  `eligible_max_bucket_count` (`tree.rs:2572`), and their call sites inside
  `select_level_ctr_aware` (`tree.rs:2588`) `[VERIFIED: LOCAL git diff
  crates/cb-train/src/tree.rs against HEAD, 104-line diff matching the
  ORD-06 SPEC's described change exactly]`.

## Current Project Architecture

### Relevant subsystems and boundaries

- `crates/cb-train/src/tree.rs` — the CTR-aware oblivious greedy tree search
  (`greedy_tensor_search_oblivious_with_ctr` → `select_level_ctr_aware` →
  `build_ctr_aware_histogram` / `score_candidate_ctr_aware` /
  `cat_feature_weight`) — the ENTIRE implicated subsystem
  `[CODEGRAPH: crates/cb-train/src/tree.rs:2233-2757]`.
- `crates/cb-compute/src/score.rs` — `l2_split_score` (49),
  `cosine_split_score` (73), consumed unchanged via `split_score`
  `[CODEGRAPH: crates/cb-compute/src/score.rs:49,73]` — confirmed NOT
  implicated (reused verbatim by both float and CTR candidates; the
  divergence is in candidate-set WEIGHTING/selection, not the shared scorer).
- `crates/cb-train/src/boosting.rs` — `train_cat`/`train_inner`, CTR
  materialization, `model_size_reg_default()` (`:536`, hardcoded `0.5`, NOT
  a configurable `BoostParams` field) `[CODEGRAPH: crates/cb-train/src/
  boosting.rs:2145,2259,529-548,3910-3914]`.
- `crates/cb-model` — consumes the trained `cb_train::Model`/`GrownTree`
  unchanged; `fstr_ctr_oracle_test.rs`'s sanity gate is the integration proof.

### Existing data/control flow

Per tree, per level `0..depth`: build ONE combined float+CTR
`BucketHistogram` over the CURRENT partition (`build_ctr_aware_histogram`),
then score EVERY float border candidate (unweighted) and EVERY CTR
column/border candidate (weighted by `cat_feature_weight` unless the
projection was already used in this tree), then pick the strict-`>`
first-wins maximum (`tree.rs:2600-2717`). This is unchanged by ORD-06 except
for the new combination-eligibility `continue` guard inside the CTR loop.

### Existing reusable implementations

- `score_candidate_ctr_aware`/`split_score` (SHARED scalar calcer for BOTH
  float and CTR candidates — not forked, not reimplemented per-candidate-kind
  — `[CODEGRAPH: crates/cb-train/src/tree.rs:2389-2399]`).
- `cat_feature_weight` (`tree.rs:2416-2422`) and `eligible_max_bucket_count`
  (`tree.rs:2572-2585`, ORD-06-04) — already correct FORMULAS per ORD-06's
  own verification; this research did not find a formula-level defect, only
  an unresolved MAGNITUDE/INPUT discrepancy at this specific level (see
  §"Direct reproduction").

### Current conventions and patterns

- Strict first-wins tie-break (`> best`, never `>=`) — unaffected by this
  bug, not a tie-break issue (the scores are not equal; Float genuinely
  scores higher under the CURRENT weight).
- `[VERIFIED: CODEGRAPH]` / `[VERIFIED: WEB]` / `[ASSUMED]` provenance
  discipline already established by the ORD-06 SPEC/research — mirrored here.

## Standard Stack

No new libraries, frameworks, or dependencies are implicated. This is an
internal `cb-train` algorithm-correctness question; no crate/version
changes are proposed by this research.

## Dependency Analysis

- No new dependency required. All implicated code lives in
  `crates/cb-train/src/tree.rs` (private functions) and is consumed via
  `crates/cb-model` (unchanged, downstream).
- No serialization/schema/migration impact — training-time algorithm only.

## Direct Reproduction (this research's primary evidence)

1. **Confirmed the oracle test is still RED after ORD-06.**
   `cargo test -p cb-model --test fstr_ctr_oracle_test -- --nocapture`:
   all 3 tests FAIL, sanity gate fails FIRST
   (`prediction[0] diverges: got -0.211, want 0.313`)
   `[VERIFIED: LOCAL cargo test output, this research session]`.
2. **Confirmed ORD-06's fix IS present** in the working tree (uncommitted):
   `combination_ctr_eligible` at `tree.rs:2548`, `eligible_max_bucket_count`
   at `tree.rs:2572`, wired into `select_level_ctr_aware`'s CTR loop
   `[VERIFIED: LOCAL git diff crates/cb-train/src/tree.rs, 104-line diff
   matching combination-ctr-level-gating/SPEC.md §4 exactly]`.
3. **Traced tree0's ACTUAL splits** via a temporary debug integration test
   (`train_cat` with the fixture's exact params, dumping
   `model.oblivious_trees[0].splits`; written, run, then DELETED — no
   permanent file added by this research):
   ```
   level 0: FLOAT feature=1 border=-0.20138581097126007   (matches upstream)
   level 1: FLOAT feature=0 border=0.561005711555481      (matches upstream)
   level 2: FLOAT feature=1 border=0.49146831035614014    (DIVERGES)
   ```
   `[VERIFIED: LOCAL instrumented cargo test run, this research session]`.
4. **Read `crates/cb-oracle/fixtures/fstr_ctr/model.json` directly** (not
   the executor's informal description): tree0's splits are `FloatFeature(1,
   border=-0.2014)`, `FloatFeature(0, border=0.5610)`,
   `OnlineCtr(split_index=7, ctr_target_border_idx=0, border=11.999999)`
   `[VERIFIED: LOCAL crates/cb-oracle/fixtures/fstr_ctr/model.json,
   oblivious_trees[0].splits]`. Cross-referencing `features_info.ctrs`:
   `split_index=7` falls inside the **simple** CTR's border list (`elements:
   [{cat_feature_index: 0}]`, `borders: [3.999, 6.999, 11.999]` — 3
   borders, 3rd one ≈ 12.0 matches `split_index=7`'s border exactly), NOT the
   combination CTR's border list (`elements: [{0},{1}]`, only 1 border)
   `[VERIFIED: LOCAL model.json features_info.ctrs, float_features border
   counts: feature 0 has 1 border (split_index 0), feature 1 has 4 borders
   (split_indices 1-4), so split_indices 5,6,7 belong to the simple-CTR{0}
   column and split_index 7 is its LAST (3rd) border]`. **This conclusively
   proves the level-2 upstream winner is a SIMPLE CTR, not a combination —
   ORD-06's gating is provably irrelevant to this specific divergence.**
5. **Instrumented `select_level_ctr_aware` to dump every candidate's score**
   at every level (temporary `eprintln!` behind an env-var gate; file backed
   up before edit, restored byte-identical after — verified via `diff`/git
   diff stat matching the pre-edit state exactly). Level-2 (`chosen.len() ==
   2`) results:
   - `Float(feature=1, border=0.4915)`: raw score **3.2428** (the eventual
     Rust winner).
   - `Ctr{col=0 (proj [0]), border=10.0}`: weighted score **3.0055** (`=
     cat_weight(0.7071) * raw_score(4.2510)`), the highest-scoring CTR{0}
     candidate at this level — still BELOW the float candidate.
   - `cat_weight` for column 0 = `cat_feature_weight(count=5, max_count=5,
     model_size_reg=0.5) = (1 + 5/5)^-0.5 ≈ 0.70711` (both simple columns'
     `bucket_count` — 5 for `{0}`, 4 for `{1}` — and `max_bucket_count == 5`
     were confirmed via the SAME instrumentation to already correctly EXCLUDE
     the ineligible `{0,1}` combination column, i.e. ORD-06-04 is working
     as designed at this level: `max_bucket_count == 5`, not `20`).
   - **The RAW (unweighted) CTR score (4.2510) is meaningfully HIGHER than
     the raw float score (3.2428)** — a margin of ~31%. The weight would need
     to be `> 0.7629` (vs the current `0.70711`) for the CTR candidate to win
     — a narrow ~8% shortfall, not an order-of-magnitude error.
   `[VERIFIED: LOCAL instrumented cargo test run, this research session;
   instrumentation fully reverted afterward, confirmed via `diff` against a
   pre-edit backup]`.
6. **Diagnostic: forcing `cat_weight = 1.0` unconditionally** (temporary,
   reverted) does NOT converge to upstream's tree either — it OVERCORRECTS:
   level 1 (where upstream and the CURRENT weighted Rust code BOTH correctly
   pick `Float(0)@0.561`) flips to a simple CTR, and level 2 becomes a
   COMBINATION CTR (also wrong; upstream wants simple). **This rules out
   "the weight should not apply at all" as the fix** — the weight mechanism
   is doing something CORRECT at level 1 and something INCORRECT at level 2,
   which points to a level/partition-dependent input value (most likely
   `count`, `max_bucket_count`, or an interaction with the raw score at a
   4-leaf partition) rather than a wholesale formula defect
   `[VERIFIED: LOCAL instrumented diagnostic run, this research session,
   reverted]`.
7. **`score_function` sensitivity checked**: the fixture's config does not
   set `score_function` (defaults to `Cosine`, matching upstream's own
   default and the actual trained model). Forcing `EScoreFunction::L2`
   (diagnostic only) produces a DIFFERENT, ALSO-diverging tree from the very
   first level — expected, since neither the fixture nor upstream used `L2`;
   this does not isolate the bug further, but confirms (per the FSTR-01
   executor's earlier note) that `score_function` measurably changes the
   candidate ranking in this codebase, so any fix must be verified under
   `Cosine` specifically (the only score function this fixture/oracle
   exercises) `[VERIFIED: LOCAL instrumented diagnostic run, reverted]`.

## Comparison Against Upstream (vendored + WebFetch)

- `SelectBestCandidate` (`greedy_tensor_search.cpp:952-997`, VENDORED,
  read directly): the REAL upstream comparison is over `gain`, not `score`:
  ```cpp
  double score = candidate.BestScore.GetInstance(ctx.LearnProgress->Rand);
  score *= GetCatFeatureWeight(candidate, ctx, fold, maxFeatureValueCount);
  double gain = score - scoreBeforeSplit;
  const auto bestSplit = candidate.GetBestSplit(trainingData, fold, oneHotMaxSize);
  gain *= GetSplitFeatureWeight(bestSplit, ctx.LearnProgress->EstimatedFeaturesContext, layout, featureWeights);
  if (gain > bestGain) { ... }
  ```
  `[VERIFIED: LOCAL catboost-master/catboost/private/libs/algo/
  greedy_tensor_search.cpp:952-997, vendored, read directly]`.
  **Analysis:** `scoreBeforeSplit` is a SINGLE scalar computed once per level
  (`CalcScoreWithoutSplit`, called once, shared by every candidate in that
  level's `SelectBestCandidate` invocation) — subtracting the SAME constant
  from every candidate's score cannot change which one has the highest
  `gain`, PROVIDED `GetSplitFeatureWeight` is the SAME (e.g. `1.0`) for both
  the float and the CTR candidate being compared. Under that (unverified but
  plausible, since `feature_weights` is not configured by this fixture)
  assumption, upstream's `gain`-based comparison REDUCES to exactly the same
  comparison Rust already performs (`cat_weight * raw_ctr_score` vs
  `raw_float_score`). **This means the "gain vs score" / "multiplicative vs
  additive" framing this research was asked to check does NOT, by itself,
  explain the divergence** — the two comparisons are mathematically
  equivalent under the stated (unverified) assumption. The remaining
  candidate explanations are a wrong `count`/`maxFeatureValueCount` INPUT, or
  `GetSplitFeatureWeight` NOT actually returning `1.0` uniformly (unverified,
  see below).
- `GetCatFeatureWeight` (`greedy_tensor_search.cpp:926-950`, VENDORED): reads
  `fold.GetCtrs(projection).GetUniqValuesCounts(projection)
  .GetUniqueValueCountForType(ctrType)` as the numerator ("count") — this
  API is DECLARED but not DEFINED in the vendored tree (`fold.h`,
  `online_ctr.h` absent, confirmed via `find`) `[VERIFIED: LOCAL find -iname
  "fold.h" -o -iname "online_ctr.h" under catboost-master → empty]`.
- `CalcMaxFeatureValueCount` (`greedy_tensor_search.cpp:1097-1115`,
  VENDORED): iterates the ALREADY-`AddTreeCtrs`/`AddSimpleCtrs`-gated
  `candidatesContexts` (i.e., ineligible combinations were never added to
  the list in the first place — confirming ORD-06-04's "scope to eligible
  candidates" fix is the CORRECT port of this exact function, re-verified
  independently by this research, not just trusted from the ORD-06 SPEC)
  `[VERIFIED: LOCAL catboost-master/catboost/private/libs/algo/
  greedy_tensor_search.cpp:1097-1115, and :1019-1035 SelectDatasetFeaturesForScoring
  showing AddTreeCtrs populates the SAME CandidateList CalcMaxFeatureValueCount
  later iterates]`. Uses `GetMaxUniqueValueCount()` — a DIFFERENT accessor
  than `GetCatFeatureWeight`'s `GetUniqueValueCountForType(ctrType)` — the
  former is (per its name) a max over ALL ctr types tracked for a
  projection, the latter is type-specific. **Since this fixture configures
  ONLY `Borders:Prior=0.5` for both `simple_ctr` and `combinations_ctr`** (no
  `Counter`/`FeatureFreq` types requested), these two accessors are HIGHLY
  LIKELY to coincide for every projection here — but this could NOT be
  proven from the vendored tree or the WebFetches performed (the struct
  fields distinguishing `Count` vs `CounterCount` vs a possible third field
  were only partially recovered — see Open Questions).
- `online_ctr.cpp`/`.h` (fetched via WebFetch from the `v1.2.10` GitHub tag,
  since NOT vendored locally): confirmed `uniqValuesCounts.Count =
  uniqValuesCounts.CounterCount = leafCount`, where `leafCount =
  ComputeReindexHash(...)` over the per-object COMBINED categorical HASH —
  i.e., "count" is the number of DISTINCT categorical projection values
  OBSERVED in the learn sample (bounded by `CtrLeafCountLimit`), matching
  this codebase's `CtrFeatureColumn::bucket_count` semantics
  `[WEB: raw.githubusercontent.com/catboost/catboost/v1.2.10/catboost/
  private/libs/algo/online_ctr.cpp, ComputeOnlineCTRs, fetched this session]`.
  This is NOT the same quantity as the CTR VALUE's own `ctr_border_count`
  (the per-object online statistic's binarization, config default `15`,
  independent of categorical cardinality) — the instrumented trace shows
  column `{0}` (`bucket_count=5`) still producing 15 DISTINCT (non-degenerate)
  candidate scores across `border_idx in 0..15`, confirming these two
  quantities are legitimately independent in this codebase too, NOT a bug by
  itself.
- **NOT independently re-verifiable locally**: `GetSplitFeatureWeight`'s
  body (declared in `feature_penalties_calcer.h`, not vendored — confirmed
  absent), and the full `TOnlineCtrUniqValuesCounts` struct (declared via
  `online_ctr.h`, WebFetch located only method DECLARATIONS, not the
  concrete struct with `Count`/`CounterCount`/possible other fields, and not
  `GetMaxUniqueValueCount()`'s exact body). GitHub's code-search UI required
  login (unavailable in this environment); `grep.app` returned HTTP 429.
  `[VERIFIED: LOCAL attempted WebFetch to github.com/catboost/catboost blob
  and search URLs, and grep.app — all inconclusive or blocked, this
  session]`.

## Recommended Architecture and Implementation Pattern

**This research does NOT recommend a specific code change yet** — the root
cause is narrowed but not conclusively pinned (see Open Questions). The
recommended NEXT STEP, to be executed either as part of a SPEC's own
research refresh or as an explicit spike task before planning a fix:

1. Obtain an INSTRUMENTED live run of real `catboost==1.2.10` (Python) on
   this EXACT frozen fixture's data/params (train with verbose CTR/score
   logging, or attach a debugger / add temporary `CATBOOST_INFO_LOG`
   printf-style tracing to a LOCAL BUILD of the vendored C++ if one can be
   compiled) to directly read `GetCatFeatureWeight`'s `count`,
   `maxFeatureValueCount`, `score`, and `gain` for the simple-CTR{0}
   candidate AND the winning float candidate at tree0/level2. This is the
   only way to numerically confirm which INPUT differs from Rust's `count=5,
   max_count=5, model_size_reg=0.5 → weight≈0.7071`.
2. Alternatively/additionally, WebFetch (or obtain by other means) the full
   bodies of `GetSplitFeatureWeight` (`feature_penalties_calcer.cpp`) and
   `TOnlineCtrUniqValuesCounts`/`GetMaxUniqueValueCount`
   (`catboost/libs/model/online_ctr.h` or wherever the struct itself, not
   just declarations, lives) to close the two remaining unverified-assumption
   gaps identified above.
3. Once the true upstream `count`/`maxFeatureValueCount`/weight values for
   this exact candidate are known, compare numerically against Rust's
   `column.bucket_count` / `eligible_max_bucket_count` output (both already
   instrumented and reproducible via the debug-test pattern this research
   used) to identify the exact discrepancy, then encode it as a
   failure-isolated behavioral spec (mirroring the ORD-06 SPEC's format).

**If a live upstream re-instrumentation is not feasible**, the fallback
recommended approach is a bisection strategy purely from the Rust side:
construct a MINIMAL synthetic unit test (categorical + 1 float feature, small
enough to hand-compute the expected leaf stats) with `model_size_reg != 0`
and `depth >= 3`, forcing a SPECIFIC known-correct expected winner by
construction (not oracle-dependent), then verify Rust's histogram/score
matches a manually-derived expectation at `chosen.len() == 2`. This would at
least rule in/out the "raw score at 4-leaf partition" hypothesis
independently of the "weight input" hypothesis, narrowing the search space
for the SPEC author.

## Project Impact Scope

### Must Change

- `crates/cb-train/src/tree.rs` — `cat_feature_weight` (`:2416`), and/or the
  `max_bucket_count`/`count` inputs fed to it inside `select_level_ctr_aware`
  (`:2588-2717`) — the exact sub-location is NOT yet pinned (see Open
  Questions). **Reason:** this is the only implicated subsystem; ORD-06's
  combination-eligibility code is confirmed NOT touched by this bug.
  **Downstream effect:** any change to `cat_feature_weight`'s magnitude or
  inputs affects EVERY simple-CTR-vs-anything-else comparison in every model
  with `model_size_reg != 0` and >= 1 categorical feature — a broad blast
  radius requiring careful re-verification of ALL currently-green CTR tests.

### May Change

- `crates/cb-train/src/ctr/ctr_feature.rs`'s `CtrFeatureColumn::bucket_count`
  computation, IF the root cause turns out to be in how `bucket_count` itself
  is computed (e.g., an off-by-something in observed-vs-declared cardinality)
  rather than in `cat_feature_weight`'s consumption of it — this research
  found `bucket_count` values (5, 4, 20) consistent with the fixture's
  declared `cat_cardinalities: [5, 4]`, so this is LOW likelihood but not
  fully excluded (the WebFetch could not confirm upstream's `leafCount` is
  necessarily equal to DECLARED cardinality rather than OBSERVED-in-sample
  cardinality, which could differ in a 200-row sample if not every category
  value appears — `[ASSUMED]` these coincide for this fixture; not verified
  by directly counting distinct values in `X_cat.npy`).

### Verification Only

- `crates/cb-compute/src/score.rs` (`l2_split_score`, `cosine_split_score`)
  — confirmed shared/unchanged, re-run `score_test.rs` as a fence.
- ORD-06's `combination_ctr_eligible`/`eligible_max_bucket_count` — confirmed
  correctly excluding the ineligible combination column at tree0/level2
  (re-verified independently by this research's own instrumentation, not
  just trusted); no change expected, but MUST be re-run as a regression fence
  since any change to `select_level_ctr_aware`'s surrounding code is in the
  same function.
- `crates/cb-model`'s apply/serialize path — not implicated; only consumes
  an already-built `Model`.

### Explicitly Out of Scope

- `catboost-master/` (read-only oracle reference).
- FSTR-01's `fstr.rs` (blocked, not buggy itself).
- ONNX export, CTR model loading (phase 23) — consume an already-built
  `Model`, not training-time search.
- Ordered-boosting + CTR precedence (`boosting.rs`'s `has_ctr` branch order)
  — pre-existing, separate, unaffected — same non-goal ORD-06 already
  recorded.

## Do Not Hand-Roll

- `score_candidate_ctr_aware`/`split_score`/`l2_split_score`/
  `cosine_split_score` — confirmed shared, correct, reused unchanged by both
  float and CTR candidates. A fix must NOT fork a separate CTR-specific
  scorer.
- `eligible_max_bucket_count`/`combination_ctr_eligible` (ORD-06) — must be
  reused/extended, not duplicated, if the fix needs to further scope
  `max_bucket_count` or a similar per-level candidate set.
- `TProjection::cat_features()`/`is_simple()`/`is_combination()` — existing,
  sufficient; no new projection type needed.

## Common Pitfalls and Risks

1. **Trigger:** assuming ORD-06's fix also fixes this bug (it doesn't;
   combination eligibility is never invoked for this comparison).
   **Consequence:** wasted planning effort re-deriving already-settled
   ORD-06 conclusions instead of investigating the NEW divergence.
   **Prevention:** this research independently re-confirmed via `model.json`
   that the level-2 upstream winner is a SIMPLE CTR, not a combination.
   **Verification:** cross-reference `features_info.ctrs[0].elements`
   (single cat_feature_index) against the winning split's `split_index`
   range, as done in §"Direct reproduction" step 4.
2. **Trigger:** "fixing" the divergence by disabling/zeroing
   `cat_feature_weight` entirely. **Consequence:** overcorrects — level 1
   (currently correct) flips to CTR, and level 2 flips to the WRONG kind
   (combination instead of simple). **Prevention:** any fix must be verified
   against ALL THREE levels of tree0, not just level 2 in isolation.
   **Verification:** re-run the instrumented split-dump debug test (pattern
   documented in §"Direct reproduction" step 3) after any change, not just
   the pass/fail oracle test.
3. **Trigger:** changing `cat_feature_weight`'s formula globally (e.g.,
   altering the exponent or the `(1+ratio)` term) to make THIS fixture pass.
   **Consequence:** the formula is ALREADY confirmed byte-for-byte correct
   against upstream's `GetCatFeatureWeight` (ORD-06's own verification,
   independently re-confirmed by this research's vendored-source read) — a
   formula-level change would silently break the (currently correct, per
   this research) level-0/level-1 comparisons and any other model relying on
   the existing formula. **Prevention:** any fix should be an INPUT
   correction (what `count`/`max_count` values feed the existing formula),
   not a formula rewrite, unless a live upstream trace PROVES the formula
   itself is wrong. **Verification:** the level-0/level-1 comparisons in
   THIS fixture (already correct) must remain correct after any fix — add a
   regression assertion for them, not just level 2.
4. **Trigger:** assuming `GetSplitFeatureWeight`/`feature_weights` (FEAT-04)
   plays no role in the real upstream computation because this fixture
   doesn't configure custom feature weights. **Consequence:** this research
   could NOT verify locally that `GetSplitFeatureWeight` returns exactly
   `1.0` for a CTR split by default (the function body is not vendored).
   **Prevention:** treat this as an open, unverified assumption (see Open
   Questions #2), not a settled fact, before ruling out FEAT-04 involvement
   entirely. **Verification:** WebFetch `feature_penalties_calcer.cpp`'s
   `GetSplitFeatureWeight` body directly, or instrument a live upstream run.
5. **Trigger:** adding new CTR+float+non-zero-`model_size_reg` test coverage
   only at `depth == 2` (matching existing fixture depths) instead of
   `depth >= 3`. **Consequence:** the bug is NOT reproducible at
   `chosen.len() <= 1` (levels 0–1 of THIS fixture are already correct); a
   shallow regression test would give false confidence. **Prevention:** any
   new unit/regression test for this bug MUST exercise a partition with `>=
   4` leaves (`chosen.len() == 2`, i.e. depth >= 3).
   **Verification:** `AT-ORD07-*` acceptance tests (once specified) should
   explicitly assert on a >= 3-level tree.

## Testing and Verification Strategy

- **Unit tests:** none currently exist for `cat_feature_weight` /
  `eligible_max_bucket_count` combined with a REAL (non-toy) `count`/`max_count`
  ratio at depth >= 3; `crates/cb-train/src/tree_test.rs` is the correct
  mount point for new pure-function-level tests once the root cause is
  pinned (mirrors `combination_ctr_eligible`'s existing test placement).
- **Integration/contract tests:** `crates/cb-train/tests/
  ctr_split_scoring_test.rs` explicitly avoids `model_size_reg != 0` (5/5
  occurrences use `0.0`) — a NEW test with `model_size_reg != 0` and
  `depth >= 3` is required (currently absent, confirmed by grep).
- **End-to-end/regression (oracle):**
  `cargo test -p cb-model --test fstr_ctr_oracle_test` (currently RED, the
  target acceptance test), plus
  `cargo test -p cb-train --test tensor_ctr_e2e_oracle_test --test
  multi_permutation_e2e_oracle_test --test ctr_split_scoring_test` (currently
  GREEN, must stay green — none of these exercise `model_size_reg != 0` +
  float mixing + depth >= 3, so they are a WEAK regression fence for this
  SPECIFIC bug but a required non-regression fence regardless).
- **Migration/data checks:** none (no schema/serialization change expected).
- **Security/performance/operational checks:** none identified; this is a
  pure training-time scoring-correctness question.
- **Exact commands used/verified this session:**
  - `cargo test -p cb-model --test fstr_ctr_oracle_test -- --nocapture`
    (confirmed RED, all 3 fail, sanity gate first).
  - `cargo test -p cb-train --lib tree` (confirmed 39 tests pass after
    instrumentation was reverted — no regression introduced by this
    research's temporary edits).

## Planning Guidance

- **Suggested work boundaries:** this is a SINGLE bug (`ORD-07`, tentative
  ID) confined to `select_level_ctr_aware`/`cat_feature_weight` in
  `tree.rs`. Do not bundle with any other pending phase's work.
- **Ordering constraint:** a SPEC/PLAN for this bug should NOT be authored
  until the "Recommended Architecture" section's step 1 or 2 (a live
  upstream trace, or the missing vendored/WebFetch definitions) resolves
  which of the two remaining hypotheses (wrong `count`/`max_count` INPUT vs.
  a raw-score defect at 4-leaf partitions) is correct — otherwise the SPEC
  risks prescribing a fix for the wrong mechanism (as ORD-06-04's own
  plan-check cycle demonstrated is easy to get subtly wrong in this exact
  function).
- **Decisions the planner must preserve:** ORD-06's fix (byte-identical);
  the shared scorer (`score_candidate_ctr_aware`/`split_score`); the strict
  first-wins tie-break; `cat_feature_weight`'s FORMULA (only its INPUTS are
  suspect, not the formula itself, per the diagnostic in step 6 of
  "Direct reproduction").
- **Items requiring a spike or user decision before implementation:**
  YES — explicitly flagged. See "Recommended Architecture" §1–2 and Open
  Questions #1–#3. This research recommends the Planner either (a) commission
  a live-upstream-trace spike task as `ORD-07`'s first task, or (b) accept
  the Rust-side bisection fallback strategy described above as a
  lower-confidence but locally-executable alternative, before writing
  failure-isolated behavioral specs.

## Addendum (post-research spike, same session) — root cause narrowed to HIGH confidence

Following this research's own recommendation ("a spike is required before this
can be written into a SPEC/PLAN"), a live-upstream-instrumentation spike was
performed using `catboost==1.2.10`'s `logging_level="Debug"` training
parameter (NOT a C++ rebuild — this parameter is exposed by the public Python
API and prints each level's WINNING candidate's post-`GetCatFeatureWeight`
score directly to stderr during `model.fit()`), installed via a fresh `uv`
venv (`uv venv --python 3.12 && uv pip install catboost==1.2.10 'numpy<2'`,
the project's standard offline-fixture recipe). This is a materially stronger
evidence source than anything available to the original research pass (which
had no live upstream execution).

**Confirmed via direct upstream debug output** (re-running the EXACT frozen
`fstr_ctr` fixture's params/data/seed — `crates/cb-oracle/fixtures/fstr_ctr/gen_fixtures.py`
copied verbatim, only `logging_level`/`verbose` added, and output paths
redirected to a scratch directory — the committed fixture files themselves
were never touched or regenerated):

```
1, bin=98 score 2.415736554        <- tree0 level0 winner (Float(1))
0, bin=139 score 3.138951304       <- tree0 level1 winner (Float(0))
{2} pr0 tb0 type0, border=11 score 3.842356441   <- tree0 level2 winner (simple CTR{cat0})
```

Cross-referenced against an equivalently-instrumented Rust run (temporary
`eprintln!` behind an `ORD07_DEBUG` env var inside `select_level_ctr_aware`,
added and FULLY REVERTED after this spike — confirmed via `git diff` matching
the pre-spike ORD-06 state exactly, no permanent code change from this
addendum):

- **Level 0 and level 1: Rust's FLOAT candidate scores match upstream's
  printed winning scores to 6 decimal places EXACTLY** (`2.415736554` vs
  Rust's `2.415737`; `3.138951304` vs Rust's `3.138951`) — this is a clean,
  high-precision confirmation that the shared histogram/scorer
  (`build_ctr_aware_histogram`/`score_candidate_ctr_aware`/`split_score`) is
  BYTE-LEVEL consistent with upstream for float candidates, and that the
  printed upstream number is the raw `score * catWeight` product (not a
  `gain`-style value with a level-constant subtracted, since `catWeight == 1`
  for float candidates and the match is exact).
- **Level 2: Rust's BEST POSSIBLE weighted CTR{cat0} score, even at its own
  raw-score peak (`border_idx=10`, `raw=4.250470`, current
  `eligible_max_bucket_count`-derived `weight=0.707107` →
  `weighted=3.005536`), falls short of upstream's actual achieved winning
  score (`3.842356441`) by a WIDE margin** — far more than previously
  estimated. Solving for the weight upstream must have effectively applied
  (`3.842356441 / 4.250470 ≈ 0.90397`) and back-solving
  `(1+ratio)^-0.5 = 0.90397` gives `ratio ≈ 0.2238`, i.e. an implied
  `maxCount ≈ 22.35` — MUCH larger than Rust's current `eligible_max_bucket_count`
  output of `5` at this level, and closer to (but not exactly) the STILL-
  ineligible pure combination `{cat0,cat1}`'s own `bucket_count = 20`.

**New, empirically-verified hypothesis (HIGH confidence, mechanism-level;
exact formula still needs TDD-driven refinement against the real oracle):**
upstream's `maxFeatureValueCount` (`CalcMaxFeatureValueCount`,
`greedy_tensor_search.cpp:1097-1115`) is fed `candidatesContexts`, which
(per `AddTreeCtrs`, already read in full during the original research pass)
includes ONE MORE candidate SOURCE this codebase's port never accounted for:
`binAndOneHotFeaturesTree` — a projection built from the tree's ALREADY-CHOSEN
FLOAT (and one-hot) splits, `seenProj.insert(binAndOneHotFeaturesTree)`, which
becomes a NON-EMPTY, ELIGIBLE base for extension-by-one-cat-feature the
MOMENT the tree has chosen even a single FLOAT split — independent of whether
any actual `Ctr` split has been chosen yet. Extending this base by one
CTR-eligible cat feature produces a MIXED (float-partition-context + one cat
feature) projection — a projection KIND `cb_train::TProjection` cannot
represent at all (categorical-only, per ORD-06's own established
simplification) and this codebase never scores/offers as an actual candidate
— but upstream's `CalcMaxFeatureValueCount` STILL includes THIS projection's
observed bucket count in the `max` used to weight the ACTUAL (representable,
scored) simple/combination CTR candidates, even though the mixed projection
itself is never a real candidate in this codebase's design.

**Directly verified against the fixture's real data** (`X_float.npy`/`X_cat.npy`,
independent of any C++ source, pure Python/NumPy counting of distinct
`(partition-leaf, cat-value)` pairs actually observed among the 200 learn
rows, using the ALREADY-CONFIRMED-CORRECT chosen float splits from tree0):

| Tree context (already-chosen splits) | mixed(float-ctx, cat0) bucket count | mixed(float-ctx, cat1) bucket count |
|---|---|---|
| level 0 (chosen=[], no float split yet) | N/A (no float split chosen — base empty, matches upstream's `baseProj.IsEmpty()` skip) | N/A |
| level 1 (chosen=[Float(1)@-0.201386]) | **10** | **8** |
| level 2 (chosen=[Float(1)@-0.201386, Float(0)@0.561006]) | **20** | **16** |

This table's LEVEL-2 entry is a STRONG, coherent, fully-sourced fit to the
observed divergence pattern:

- **Level 0** (hypothesis predicts NO mixed candidates exist yet, since no
  float split has been chosen — `max_bucket_count` stays `max(5,4)=5`,
  UNCHANGED from today's correct behavior). Consistent with level 0 already
  being correct today.
- **Level 1: `[UNVERIFIED — RETRACTED, plan-checker pass 2 finding]`.** A
  prior version of this section claimed including `max(5, 4, 10, 8) = 10`
  gives simple-CTR{cat0} `ratio=5/10=0.5`, `weight=(1.5)^-0.5≈0.81650`,
  `weighted ≈ 3.844592 × 0.81650 ≈ 3.13950` — "almost EXACTLY at the tie
  boundary" of the real level-1 float winner (`3.138951`). **On independent
  review this raw score (`3.844592`) was found to be UNSOURCED** — it does
  not appear in any instrumented run this document's own "Direct
  reproduction" section describes, is not self-consistent with the claimed
  weight to the precision otherwise used here (back-solving from
  `3.138951`/`0.81650` gives `≈3.844415`, not `3.844592`), and is
  suspiciously close to the UNRELATED, correctly-sourced level-2 ACTUAL
  winning score (`3.842356441`) — consistent with a transcription error,
  not independent evidence. **This level-1 claim is RETRACTED** and is NOT
  part of this bugfix's supporting evidence (level 1 is also not the
  fixture's failing level, so this retraction does not weaken the
  mechanism's relevance — only its precision at that specific level is
  unproven). See `SPEC.md` §1/§9 for the authoritative, corrected framing;
  `AT-ORD07-03b` (the real oracle test) is the actual arbiter regardless.
- **Level 2 (the SOLE quantitative evidence claimed, fully instrumentation-sourced)**:
  including `max(5, 4, 20, 16) = 20` gives `weight ≈ 0.894427`,
  closing ~90% of the previously-observed gap (needed `≈0.90397`, this
  hypothesis's prediction `0.894427`, vs the OLD/current `0.707107` — the
  residual ~1% gap is well within the uncertainty already flagged around
  upstream's border-INDEX labeling convention not being directly comparable
  to Rust's `border_idx` (upstream's debug log names its winning border `11`
  in its OWN internal numbering, not proven to correspond 1:1 to Rust's
  `border_idx=10`, so the assumption "upstream's true raw score equals Rust's
  raw-score PEAK at `border_idx=10`" carries some residual uncertainty this
  spike could not fully close without a live per-border upstream score dump).

**This is a MORE SUBSTANTIAL fix than ORD-06/ORD-06-04.** It requires a NEW,
per-level, per-CTR-eligible-cat-feature computation: for the tree's CURRENT
partition (as defined by already-chosen FLOAT splits specifically — CTR
splits' contribution to `seenProj` was ALREADY correctly handled by
ORD-06-04's `eligible_max_bucket_count`), count the DISTINCT
`(partition-leaf, cat-value)` pairs actually observed in the learn sample, for
EVERY CTR-eligible cat feature not yet part of an already-chosen CTR
projection — analogous to but ARCHITECTURALLY DISTINCT from
`CtrFeatureColumn::bucket_count` (which is a GLOBAL, tree-independent,
materialized-once-per-tree quantity; this new quantity is PARTITION-SCOPED,
recomputed per level, and NEVER results in an actual scoreable candidate —
it exists ONLY to correctly compute `max_bucket_count`'s upper bound, exactly
mirroring `CalcMaxFeatureValueCount`'s real upstream behavior of including
`binAndOneHotFeaturesTree`-derived "phantom" (never-scored-in-this-codebase)
projections in its max).

**Confidence recharacterization:** the MECHANISM (mixed float-context+cat
bucket counts must contribute to `max_bucket_count`, gated by "has this tree
chosen at least one FLOAT split so far", independent of/`additional to`
ORD-06-04's existing CTR-projection-based gating) is **HIGH confidence at
level 2** (the fixture's actual failing level — fully instrumentation-sourced
and self-consistent) and **level 0** (structurally provable — no float split
chosen, no phantom candidate exists). **`[UPDATED, plan-checker pass 2]`
Level 1's supporting arithmetic was found unsourced on review and has been
RETRACTED** (see the corrected table analysis above) — it is neither
confirming nor refuting the mechanism, simply unproven at that level. The
EXACT numeric formula (is it
precisely `distinct (leaf, cat-value) pairs observed`, or some other closely
related quantity — e.g., does it need to also incorporate one-hot-routed
features, or use a different counting convention for ties/empty leaves) is
**MEDIUM confidence** — recommend the SPEC/PLAN treat this as a
TDD-driven-against-the-real-oracle refinement (the frozen `fstr_ctr` fixture
IS ground truth; a fix should be validated by making
`fstr_ctr_oracle_test.rs` pass, not solely by a priori formula derivation)
rather than requiring a fully independently-re-derived upstream C++ citation
(this spike found the relevant upstream mechanism — `AddTreeCtrs`'s
`binAndOneHotFeaturesTree` seenProj entry — already fully quoted/verified in
the ORD-06 research; what was missing was recognizing that
`CalcMaxFeatureValueCount` consumes ITS bucket count too, not just the
already-representable CTR columns').

**Practical implication for scope:** computing "distinct (leaf, cat-value)
pairs observed" per level requires access to the per-object cat-feature raw
values AND the current partition assignment (`leaf_of`) at scoring time —
both of which `select_level_ctr_aware`/`build_ctr_aware_histogram` already
have available (the function receives `matrix`/`chosen` and can derive
`leaf_of` the same way `build_ctr_aware_histogram`'s own
`assign_leaves_ctr_aware` call does) — so this is an ADDITIVE computation
inside the SAME function ORD-06 already modified, not a new cross-module
wiring change. `candidates.rs`/`boosting.rs` remain unchanged, consistent with
ORD-06's established "surgical, single-function fix" pattern.

## Open Questions

1. **[BLOCKING for a settled root cause]** What are upstream's REAL
   `count`/`maxFeatureValueCount` values (and the resulting
   `GetCatFeatureWeight` output) for the simple-CTR{0} candidate at
   tree0/level2 of THIS exact fixture? Not verifiable from the vendored tree
   or the WebFetches this research performed (only the FORMULA and its
   high-level input semantics were confirmed, not the concrete numeric
   values upstream actually computed). Requires either a live instrumented
   `catboost==1.2.10` run or further targeted WebFetch/source recovery of
   `online_ctr.h`'s full `TOnlineCtrUniqValuesCounts` struct and
   `GetMaxUniqueValueCount()`'s body.
2. **[MEDIUM confidence, unverified]** Does `GetSplitFeatureWeight`
   (`feature_penalties_calcer.cpp`, NOT vendored in this repo) return exactly
   `1.0` for BOTH a Float split and an OnlineCtr split when `feature_weights`
   is unconfigured (as in this fixture)? Assumed yes (consistent with the
   levels-0/1 matches and with FEAT-04's documented "no-op default"
   behavior in this codebase's OWN `feature_weight`/`FeaturePenalties`
   implementation), but not independently confirmed against the upstream
   C++ body.
3. **[LOW confidence, unverified]** Is `column.bucket_count` (this
   codebase's per-projection cardinality) computed from the DECLARED
   cardinality or the OBSERVED-in-learn-sample distinct count? For this
   fixture (200 rows, cardinalities 5 and 4) these likely coincide, but this
   was not directly checked by counting distinct values in
   `X_cat.npy` against `bucket_count`.
4. **[NON-BLOCKING, informational]** Is there a DIFFERENT codebase-side bug
   entirely unrelated to `cat_feature_weight` — e.g., in
   `build_ctr_aware_histogram`'s handling of a 4-leaf partition
   specifically — that happens to also move the score by a similar
   magnitude? The diagnostic in "Direct reproduction" step 6 (disabling the
   weight) suggests the weight IS causally involved (level 1's correctness
   depends on it), but does not fully exclude an ADDITIONAL raw-score defect
   compounding at level 2. A minimal hand-computable synthetic unit test (per
   "Recommended Architecture" fallback) would resolve this independently of
   upstream access.

## Sources

- **Project documents (local, PageIndex unavailable this session — see
  Confidence Assessment):**
  `.planning/phases/24-ctr-split-search-correctness/combination-ctr-level-gating/{SPEC.md,PLAN.md,PLAN-CHECK.md,research.md,SOURCES.md}`
  (read in full; ORD-06's settled facts, conventions, and citation style
  mirrored here).
  `crates/cb-oracle/fixtures/fstr_ctr/{config.json,model.json,gen_fixtures.py}`.
  `crates/cb-model/tests/fstr_ctr_oracle_test.rs`.
  `crates/cb-train/tests/ctr_split_scoring_test.rs`,
  `tensor_ctr_e2e_oracle_test.rs`, `multi_permutation_e2e_oracle_test.rs`,
  `one_hot_oracle_test.rs`.
  `CLAUDE.md` (project root).
- **CodeGraph queries (MCP available and used):**
  `select_level_ctr_aware combination_ctr_eligible cat_feature_weight
  max_bucket_count score_candidate_ctr_aware tree.rs` →
  `crates/cb-train/src/tree.rs:2233-2757` (full CTR-aware search),
  `crates/cb-compute/src/score.rs:49,73` (`l2_split_score`/
  `cosine_split_score`), `crates/cb-train/src/boosting.rs:529-548,2145-2168`
  (`model_size_reg_default`, `train_cat`), `crates/cb-model/src/model.rs`
  (`ModelSplit`, `CtrSplit`, `ObliviousTree`).
- **Local verification (`cargo test`, `find`, `grep`, Python `npy`
  inspection), this session:**
  - `cargo test -p cb-model --test fstr_ctr_oracle_test -- --nocapture`
    (confirmed RED, 3/3 fail, sanity gate first).
  - A temporary debug integration test (written, run, DELETED — not part
    of the final deliverable) dumping `model.oblivious_trees[0].splits`,
    confirming levels 0–1 match upstream and level 2 diverges
    (Float(1)@0.4915 vs upstream's simple CTR{0}).
  - `python3` inspection of `crates/cb-oracle/fixtures/fstr_ctr/model.json`
    (`oblivious_trees[0].splits`, `features_info.ctrs`,
    `features_info.float_features`, `ctr_data`) to independently derive
    that the level-2 upstream winner is a SIMPLE CTR (not combination).
  - Temporary instrumentation of `crates/cb-train/src/tree.rs`'s
    `select_level_ctr_aware` (env-var-gated `eprintln!` of every candidate's
    score, plus a diagnostic `cat_weight`-disable toggle) — backed up before
    editing, restored BYTE-IDENTICAL afterward (verified via `diff`/`git
    diff --stat` matching the pre-edit state exactly). No permanent
    production or test file changes remain from this research.
  - `grep -n "model_size_reg\|depth:"` across
    `crates/cb-train/tests/*.rs` confirming zero existing coverage of
    `model_size_reg != 0` combined with float+CTR mixing at any depth.
  - `find`/`grep` confirming `feature_penalties_calcer.{h,cpp}`, `fold.h`,
    `online_ctr.h` are ABSENT from the vendored `catboost-master/` tree in
    this repo.
- **Vendored upstream C++ (read directly, this session):**
  `catboost-master/catboost/private/libs/algo/greedy_tensor_search.cpp`
  — `SelectBestCandidate` (952-997), `GetCatFeatureWeight` (926-950),
  `CalcMaxFeatureValueCount` (1097-1115),
  `SelectDatasetFeaturesForScoring`/`SelectFeaturesForScoring` (1000-1095),
  `CalcScores` (887-924).
- **Official web documentation (`v1.2.10` tag, WebFetch, this session,
  access date 2026-07-18):**
  `raw.githubusercontent.com/catboost/catboost/v1.2.10/catboost/private/
  libs/algo/online_ctr.h` (method declarations for `GetUniqValuesCounts` on
  `TOnlineCtrBase`/`TOwnedOnlineCtr`/`TPrecomputedOnlineCtr`; full struct
  body NOT recovered).
  `raw.githubusercontent.com/catboost/catboost/v1.2.10/catboost/private/
  libs/algo/online_ctr.cpp` (`ComputeOnlineCTRs`'s `leafCount =
  ComputeReindexHash(...)` assignment to `UniqValuesCounts.Count`/
  `.CounterCount`).
  `github.com/catboost/catboost/blob/v1.2.10/catboost/private/libs/algo/
  fold.h` (fetched, did not contain the needed definitions).
  Attempted, inconclusive/blocked: `github.com/search?...&type=code`
  (requires GitHub login in this environment), `grep.app` (HTTP 429).

## Confidence Assessment

- **HIGH:**
  - The divergence reproduces at tree0/level2, confirmed by direct
    instrumented re-training against the frozen fixture.
  - ORD-06's fix is present and NOT the cause of this specific divergence
    (independently re-derived from `model.json`, not just trusted from the
    executor's informal description).
  - `cat_feature_weight`'s formula and its documented inputs
    (`count`=distinct-cardinality, `max_bucket_count`=max over
    ELIGIBLE-at-this-level columns) match the vendored
    `GetCatFeatureWeight`/`CalcMaxFeatureValueCount` semantics as read
    directly from `catboost-master/`.
  - Disabling `cat_feature_weight` entirely does not fix the bug (rules out
    "should not apply" as the fix).
- **MEDIUM:**
  - The claim that `GetSplitFeatureWeight` returns `1.0` uniformly for both
    Float and CTR splits by default (consistent with levels 0–1 matching
    upstream, and with this codebase's own no-op-by-default `FeaturePenalties`
    design, but not independently confirmed against the (non-vendored)
    upstream body).
  - `column.bucket_count`'s semantics (distinct categorical cardinality,
    matching upstream's `leafCount`/`ComputeReindexHash`) — confirmed via
    WebFetch of `online_ctr.cpp`, but the EXACT numeric match for THIS
    fixture's 200-row sample was not independently re-derived from
    `X_cat.npy`.
- **LOW:**
  - The PRECISE numeric root cause (which input, or whether it's an input
    at all vs. a raw-score defect at 4-leaf partitions) — NOT settled;
    requires either a live upstream trace or further local bisection, per
    Open Questions #1 and #4.
  - Whether `GetMaxUniqueValueCount()` and `GetUniqueValueCountForType
    (Borders)` truly coincide for every projection in this fixture (assumed
    from "only Borders CTR type is configured", not proven from the
    (unrecovered) `TOnlineCtrUniqValuesCounts` struct body).
  - PageIndex MCP was NOT used this session (not invoked; the research
    relied on local file reads of `.planning/phases/24.../research.md` etc.
    instead, since the prior ORD-06 artifacts were already known/named by
    the task prompt). If a project PageIndex index exists with additional
    indexed specs/ADRs not referenced by this research, they were not
    cross-checked — noted as a coverage gap, not a contradiction.
