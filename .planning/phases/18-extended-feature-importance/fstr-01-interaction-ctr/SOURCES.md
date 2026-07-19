# Evidence Ledger — FSTR-01 Interaction + PredictionValuesChange CTR Support

Research report consumed: `.planning/phases/18-extended-feature-importance/fstr-01-interaction-ctr/research.md`
(phase-research-agent, 2026-07-17, plus a follow-up `v1.2.10`-pinned WebFetch
spike appended as its Addendum). Read in full before planning.

## Requirement / roadmap
- Git-recovered `.planning/REQUIREMENTS.md`/`ROADMAP.md` (commit `a82289c`) —
  NOT in the working tree; FSTR-01 wording recovered per prior phases'
  SOURCES.md ledgers. Re-verify canonical revision before flipping the
  requirement checkbox (T7 bookkeeping).

## Upstream behavior (pinned `v1.2.10` tag — NOT `master` HEAD)
- `[VERIFIED: WEB github.com/catboost/catboost/blob/v1.2.10/catboost/libs/fstr/calc_fstr.cpp]`
  — full verbatim `CalcFeatureInteraction` (cross-product, `Score /
  (side0.len()*side1.len())`, `layout.GetExternalFeatureIdx`) and full
  verbatim `CalcRegularFeatureEffect` (equal-split `addEffect =
  effectWithSplit.first / featuresInSplit`, four per-type vectors merged into
  one `TVector<TFeatureEffect>`) — both quoted in full in research.md's
  Addendum.
- `[VERIFIED: WEB github.com/catboost/catboost/blob/v1.2.10/catboost/libs/fstr/feature_str.cpp]`
  — `CalcMostInteractingFeatures`'s `featureToIdx`-taking overload; confirms
  CTR splits map to ONE internal `TFeature`/index before the
  `CalcFeatureInteraction` external-index expansion runs.
- These fetches SUPERSEDE the research report's own body text, which had only
  reached `master` HEAD (LOW confidence) — the Addendum's `v1.2.10` fetch
  upgrades this to HIGH confidence, since `v1.2.10` is this project's exact
  pinned oracle floor everywhere else.

## Codebase evidence (CodeGraph / Read)
- `[VERIFIED: CODEGRAPH crates/cb-model/src/fstr.rs]` (full file) — current
  `interaction()` (:288-355), `interaction_accumulate_non_symmetric`/
  `interaction_dfs` (:369-468), `prediction_values_change()` (:123-138),
  `pvc_accumulate_oblivious`/`pvc_accumulate_non_symmetric` (:143-174,
  198-260), `feature_count` (:80-100), `interaction_add` (:265-274) — all
  CTR-skip sites and the exact line ranges cited in SPEC §1.
- `[VERIFIED: CODEGRAPH crates/cb-model/src/model.rs:43-98]` — `ModelSplit`
  (`Float`/`Ctr` only, no separate one-hot variant), `CtrSplit.projection:
  cb_train::TProjection`, `ModelSplit::float_feature()`.
- `[VERIFIED: CODEGRAPH crates/cb-model/src/model.rs:272-313]` — `Model` has
  NO original-column-order / flat-feature-kind map (only
  `float_feature_borders`, `ctr_data`) — the reason this slice derives
  `n_cat_used` from splits alone rather than adding a `Model` field.
- `[VERIFIED: LOCAL crates/cb-train/src/projection.rs:93-187]` — `TProjection`
  is CATEGORICAL-ONLY in this codebase (no `BinFeatures`/`OneHotFeatures`
  member); `cat_features()` returns sorted, de-duplicated LOCAL cat indices.
  This is a codebase-specific SIMPLIFICATION of the upstream general formula
  (see SPEC §1 "Codebase simplification").
- `[VERIFIED: CODEGRAPH crates/cb-model/src/apply.rs:37,170-199,386]` — CTR
  apply path (`predict_raw_cat`, `ctr_value_for_projection`,
  `ctr_value_for_combined_projection`) already correct/oracle-tested; PROOF
  that CTR structural support is solved elsewhere and `fstr.rs` is the one
  remaining CTR-blind consumer.
- `[VERIFIED: CODEGRAPH crates/cb-model/src/shap.rs:728,1096-1097]` — `shap.rs`
  independently confirmed CTR-blind (out of scope here, FSTR-02's dependency).
- `[VERIFIED: CODEGRAPH crates/catboost-rs/src/model.rs:139-149]` — facade
  `feature_importance` calls `interaction`/`prediction_values_change` directly;
  no facade code change needed (this slice fixes the callee's correctness).

## Fixture-family survey (justifies "new fixture required")
- `[VERIFIED: LOCAL crates/cb-oracle/fixtures/plain_ctr/config.json]`,
  `ordered_ctr/config.json`, `tensor_ctr/config.json`,
  `tensor_ctr_e2e/` (+ its oracle test), `one_hot_cat/config.json` — ALL are
  categorical-only isolation fixtures for prior ORD-03/04/05 CTR-training
  requirements; NONE dump `get_feature_importance` ground truth; NONE mix
  float + categorical columns. Confirms T4 must create a genuinely new
  fixture, not extend an existing one.
- `[VERIFIED: LOCAL crates/cb-train/tests/tensor_ctr_e2e_oracle_test.rs:81-119]`
  — `tensor_ctr_params()`, the exact isolating-parameter recipe T4's
  `gen_fixtures.py` adapts (adding float columns + `data=` for PVC).

## Unmerged-branch non-blocker confirmation
- `git branch -a` / `git worktree list` / `git log --oneline` (this session,
  matching the research report's own independent verification) — the
  `feat/23-ctr-model-loading` branch (upstream `.cbm` CTR reconstruction via
  `decode_cbm`) is unmerged into `feat/18-fstr03-partial-dependence`. Confirmed
  NOT required for this slice: the oracle fixture is built via
  `cb_train::train_cat` → `Model::from_trained`, independent of `.cbm` CTR
  decode (same pattern `tensor_ctr_e2e_oracle_test.rs` already proves).
  `[PROJECT: memory ctr-model-loading.md; research.md Constraints]`

## PageIndex
- `[VERIFIED: PAGEINDEX get_folder_structure + browse_documents(folder_id=cmrhcxbtm000104jr3i5jzm0m)]`
  — the `catboost_rs` folder currently holds exactly one document (FSTR-03's
  `SPEC.md`, status `completed`). This SPEC is NOT YET indexed;
  `process_document` has no in-place Markdown upsert (file/URL ingestion
  only), so indexing requires a human owner to add it as a second document
  out-of-band (SPEC §10).

## Constraints
- `[VERIFIED: LOCAL Cargo.toml:10-14]` — workspace denies `unwrap_used`,
  `expect_used`, `panic`, `indexing_slicing` (clippy-only enforcement, NOT
  `cargo build` — the recurring project gotcha, `[PROJECT:
  fstr-03-partial-dependence/PLAN-CHECK.md MAJOR #2]`).
- `[VERIFIED: LOCAL CLAUDE.md]` — source/test separation (no `mod tests` body
  in production `.rs`); `cb-model/src/ctr_data.rs:58-61` is the exact mount
  precedent this plan's T0 follows.
- `[PROJECT: memory catboost-rs-preexisting-test-failures.md]` — env-red
  suites to ignore (cb-backend MLIR, cb-train monotone, catboost-rs-py py3.14
  link) — unrelated to this slice.

## Planner Agent availability
- `[VERIFIED: LOCAL find /home/user/.claude/agents
  /home/user/Documents/workspace/catboost_rs/.claude/agents -iname
  '*planner*']` → only `specification-planner.md` exists (a different skill's
  agent — produces specs, not a goal-backward TDD task planner consuming a
  given SPEC). No agent literally named `planner` is installed. `PLAN.md` was
  therefore authored directly by this skill session as the documented
  fallback (`[UNVERIFIED: Planner Agent unavailable]`, recorded at the top of
  `PLAN.md`), still subject to the independent Plan Checker gate.
