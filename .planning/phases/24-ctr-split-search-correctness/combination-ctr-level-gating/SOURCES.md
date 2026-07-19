# Evidence Ledger — ORD-06 Combination-CTR Level-Gating Bugfix

Research report consumed: `.planning/phases/24-ctr-split-search-correctness/combination-ctr-level-gating/research.md`
(phase-research-agent, 2026-07-18, plus a follow-up `v1.2.10`-pinned WebFetch
spike appended in SPEC.md §1/§4 that resolved the research's one blocking
open question AND simplified the recommended architecture). Read in full
before planning.

## Discovery context
- Surfaced while implementing FSTR-01 (`.planning/phases/18-extended-feature-importance/fstr-01-interaction-ctr/`)
  — its T5/T6 oracle acceptance tests (`crates/cb-model/tests/fstr_ctr_oracle_test.rs`)
  are blocked by this bug, not by FSTR-01's own `interaction()`/
  `prediction_values_change()` code (independently confirmed: the failure
  occurs at the model-prediction sanity gate, upstream of FSTR-01's own
  logic). `[PROJECT: research.md, directly reproduced]`

## Root cause (CONFIRMED, not the triggering executor's original hypothesis)
- `[VERIFIED: CODEGRAPH crates/cb-train/src/tree.rs:2416 cat_feature_weight]`
  — byte-for-byte consistent with upstream `GetCatFeatureWeight`
  (`greedy_tensor_search.cpp:926-950`) — ruled out as root cause.
- `[VERIFIED: CODEGRAPH crates/cb-train/src/tree.rs:2527-2632
  select_level_ctr_aware, full body read]` — the "CTR candidates next" loop
  (`tree.rs:2589-2610`) scores EVERY materialized `CtrFeatureColumn`
  unconditionally, at EVERY level, including combination (multi-feature)
  columns at the ROOT — this is the actual defect.
- `[VERIFIED: CODEGRAPH crates/cb-train/src/tree.rs:2578-2586
  used_projections]` — ALREADY computed from `chosen` (today only used to
  exempt an already-used projection from the `cat_feature_weight` penalty)
  — this IS the exact "seen CTR base projections" set the fix needs; NO new
  plumbing from `boosting.rs`/`candidates.rs` required.
- `[VERIFIED: CODEGRAPH crates/cb-train/src/candidates.rs:159-201
  tensor_ctr_candidates]`, `[VERIFIED: CODEGRAPH
  crates/cb-train/src/boosting.rs:2680-2930,3892-3916]` — confirmed called
  ONCE per tree, outside any level loop, materializing every combinatorially-
  valid column up front — this remains UNCHANGED by the fix (§1 Architecture
  correction: the fix is a per-level ELIGIBILITY FILTER inside `tree.rs`,
  not a change to WHEN/WHAT gets materialized).

## Upstream behavior (pinned `v1.2.10` tag — NOT `master` HEAD)
- `[VERIFIED: WEB github.com/catboost/catboost/blob/v1.2.10/catboost/private/libs/algo/greedy_tensor_search.cpp:503-568]`
  — `AddTreeCtrs`, full verbatim body: `seenProj` built from
  `binAndOneHotFeaturesTree` (chosen float+one-hot splits) ∪
  `currentTree.GetUsedCtrs()` (chosen CTR splits' projections);
  `baseProj.IsEmpty()` skip; extend-by-one-feature inner loop with
  `IsRedundant`/`MaxTensorComplexity`/`addedProjHash` dedup gates.
- `[VERIFIED: WEB same file, ~838-902]` — `SelectDatasetFeaturesForScoring`'s
  call order: `AddFloatFeatures` → `AddOneHotFeatures` → `AddSimpleCtrs`
  (unconditional, every level) → `AddTreeCtrs` (ONLY `if
  currentSplitTree.Defined()`).
- `[VERIFIED: WEB github.com/catboost/catboost/blob/v1.2.10/catboost/private/libs/algo/split.h:469-494]`
  — `TSplitTree::GetBinFeatures`/`GetOneHotFeatures`/`GetUsedCtrs`, full
  verbatim bodies — resolves research.md's one blocking open question
  (this file is NOT vendored in `catboost-master/`,
  `[VERIFIED: LOCAL grep -rln "class TSplitTree" catboost-master → empty]`).
- `[VERIFIED: WEB github.com/catboost/catboost/blob/v1.2.10/catboost/private/libs/algo/projection.h:57-130]`
  — `TProjection`'s field list (`CatFeatures`/`BinFeatures`/`OneHotFeatures`),
  `IsRedundant` (duplicate check only), `AddCatFeature`,
  `GetFullProjectionLength` — confirms extending a
  `binAndOneHotFeaturesTree`-derived base ALWAYS yields a MIXED (float+cat)
  projection, never pure-cat — the key fact resolving SPEC §1's
  "codebase-specific simplification."

## Codebase-specific simplification (why the port is narrower than upstream's full generality)
- `[VERIFIED: LOCAL crates/cb-train/src/projection.rs:93-106]` — this
  codebase's `TProjection` is deliberately CATEGORICAL-ONLY (no
  `BinFeatures`/`OneHotFeatures` member), a PRE-EXISTING design decision
  unrelated to this bug. Combined with the `v1.2.10` evidence above (a
  mixed-base extension is always mixed, never pure-cat), this means
  upstream's `binAndOneHotFeaturesTree` half of `seenProj` is
  STRUCTURALLY IRRELEVANT to this codebase — only the `GetUsedCtrs()` half
  needs porting, which this codebase's EXISTING `used_projections`
  computation already provides verbatim.

## `max_bucket_count` scoping bug (ORD-06-04, found by plan-checker pass #1, confirmed by pass #2)
- `[VERIFIED: CODEGRAPH crates/cb-train/src/tree.rs:2576-2581 max_bucket_count]`
  — computed over ALL materialized `ctr_features` unconditionally, unaffected
  by ORD-06-03's eligibility guard (which only filters `scored`'s
  membership, not this separate input).
- `[VERIFIED: LOCAL catboost-master/catboost/private/libs/algo/greedy_tensor_search.cpp:1097-1115
  CalcMaxFeatureValueCount, VENDORED — directly read, not a v1.2.10-tag
  fetch]` — iterates `candidatesContexts`, the CURRENT LEVEL's
  already-`AddTreeCtrs`-gated candidate list, not a tree-wide static
  superset.
- `[VERIFIED: CODEGRAPH crates/cb-train/src/boosting.rs:536-538
  model_size_reg_default() -> 0.5]` — non-zero default, confirmed NOT
  overridden by `crates/cb-oracle/fixtures/fstr_ctr/config.json`
  `[VERIFIED: LOCAL crates/cb-oracle/fixtures/fstr_ctr/config.json]` — so the
  divergence is arithmetically real for the target fixture (cardinalities
  `[5,4]` → simple bucket counts 5/4, combination `{0,1}` bucket count up to
  20; unscoped `max_bucket_count=20` vs correctly-scoped `=5` is a ~26%
  relative difference in `cat_feature_weight`'s multiplier at the root,
  independently re-derived and confirmed by BOTH plan-checker passes).
- `[VERIFIED: LOCAL catboost-master/catboost/private/libs/algo/greedy_tensor_search.cpp:570-582
  AddTreeCtrs's candidate-erasure block]` — re-read during pass #2 to confirm
  an already-CHOSEN combination projection is correctly excluded from
  re-eligibility at a later level (length-gap-must-be-exactly-1 rule already
  handles this; no additional fix needed).
- `[VERIFIED: CODEGRAPH crates/cb-train/src/tree.rs:2306-2375
  build_ctr_aware_histogram]` — re-read in full during pass #2 to rule out a
  THIRD similarly-shaped gap; its unfiltered use of `ctr_features` for
  `n_bins`/`ctr_actual` sizing is provably inert (capacity-only over-sizing,
  cannot affect any other column's independently-scanned prefix sums) — no
  further fix needed.

## Fixture/test survey
- `[VERIFIED: LOCAL cargo test -p cb-model --test fstr_ctr_oracle_test --
  --nocapture]` — all 3 tests FAIL; first failure at the sanity-gate
  prediction check (`prediction[0]: got -0.061, want 0.313`), confirming a
  real tree-structure divergence, not an FSTR-01 attribution bug.
- `[VERIFIED: LOCAL crates/cb-oracle/fixtures/fstr_ctr/gen_fixtures.py]` —
  its own comment notes needing `depth=3, iterations=15` (not
  `tensor_ctr_e2e`'s `depth=2, iterations=5`) for upstream to select a
  GENUINE combination CTR — independently corroborates the root cause
  (a combination needs the tree to already be ≥1 level deep before it can
  legitimately appear).
- `[VERIFIED: LOCAL grep depth/iterations/boosting_type in
  crates/cb-train/tests/tensor_ctr_e2e_oracle_test.rs,
  multi_permutation_e2e_oracle_test.rs]` — both `depth:2,
  max_ctr_complexity:2, boosting_type:Plain` — no `Ordered`+CTR test exists
  anywhere in this codebase (relevant to SPEC §9 risk 2).
- `[VERIFIED: LOCAL Read crates/cb-train/tests/ctr_split_scoring_test.rs]`
  — its 3 existing CTR-vs-float scoring tests all use
  `TProjection::single(0)` (simple CTR only, never a combination) — NOT put
  at risk by this fix; the natural home for AT-ORD06-03c's new regression
  fence (reuses its existing harness/helpers).
- `[VERIFIED: LOCAL grep "mod tests\|#\[path" crates/cb-train/src/tree.rs]`
  — `tree.rs` already mounts 5 sibling test files
  (`tree_test.rs` as `mod general`, `tree_tie_break_test.rs`,
  `tree_ordered_test.rs`, `tree_pairwise_test.rs`, `region_grow_test.rs`) —
  `tree_test.rs`/`mod general` is the natural home for
  `combination_ctr_eligible`'s new unit tests; no new mount needed.

## Constraints
- `[VERIFIED: LOCAL Cargo.toml:10-14]` — workspace denies `unwrap_used`,
  `expect_used`, `panic`, `indexing_slicing` (clippy-only enforcement, NOT
  `cargo build`).
- `[VERIFIED: LOCAL CLAUDE.md]` — source/test separation; no `mod tests`
  body in production `.rs`.
- `[PROJECT: memory ctr-model-loading.md]` — "CTR fixtures are frozen
  because catboost quantization is run-to-run nondeterministic" — no
  fixture in this codebase may be regenerated by this fix.
- `[PROJECT: memory catboost-rs-preexisting-test-failures.md]` — env-red
  suites to ignore (cb-backend MLIR, cb-train `monotone_*`, catboost-rs-py
  py3.14 link) — unrelated to this slice.

## Planner Agent availability
- `[VERIFIED: LOCAL find /home/user/.claude/agents
  /home/user/Documents/workspace/catboost_rs/.claude/agents -iname
  '*planner*']` → only `specification-planner.md` exists (a different
  skill's agent). No agent literally named `planner` is installed. `PLAN.md`
  was therefore authored directly by this skill session as the documented
  fallback (`[UNVERIFIED: Planner Agent unavailable]`), still subject to the
  independent Plan Checker gate.

## Related artifacts
- FSTR-01 (blocked by this bug, unaffected by its fix):
  `.planning/phases/18-extended-feature-importance/fstr-01-interaction-ctr/`
  (`SPEC.md`, `PLAN.md`, `PLAN-CHECK.md` — 3 checker passes, fixes applied
  but not independently re-verified past pass 3, per that slice's own
  documented status).
- CTR model loading (phase 23, unrelated — consumes an already-built
  `Model`, not training-time search):
  `.planning/phases/23-ctr-model-loading/cbm-ctr-load/`.
