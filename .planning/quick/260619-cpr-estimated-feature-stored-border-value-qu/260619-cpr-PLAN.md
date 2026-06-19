---
phase: quick-260619-cpr
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - crates/cb-train/src/estimated/estimated_features.rs
  - crates/cb-train/src/estimated/online_embedding.rs
  - crates/cb-oracle/generator/gen_text_embedding_fixtures.py
  - crates/cb-oracle/tests/text_embedding_end_to_end_oracle_test.rs
  - crates/cb-oracle/fixtures/text_embedding_xor/
autonomous: true
requirements: [FEAT-01]
must_haves:
  truths:
    - "The upstream estimated-feature border-grid algorithm is documented: it is the SAME GreedyLogSum binarizer over DataProcessingOptions->FloatFeaturesBinarization as the numeric path, fed the estimated column VALUES — so any stored-border-VALUE divergence is a column-VALUE divergence, not a border-algorithm divergence."
    - "A serialized model's estimated-feature borders match upstream catboost 1.2.10 bit-for-bit: the KNN integer-vote border serializes as 0.5 (not 1.5), and the BoW digitization grid matches."
    - "A non-degenerate XOR-style text+embedding+numeric corpus produces StagedApprox + Predictions matching upstream <=1e-5 with NO structure-invariant leaf-order relaxation."
  artifacts:
    - path: "crates/cb-oracle/tests/text_embedding_end_to_end_oracle_test.rs"
      provides: "XOR-corpus hard oracle: exact stored borders + StagedApprox/Predictions <=1e-5, no #[ignore], no weakened tolerance, no leaf-order relaxation"
    - path: "crates/cb-oracle/fixtures/text_embedding_xor"
      provides: "Frozen catboost 1.2.10 (thread_count=1) XOR mixed fixture (model.cbm + per-stage .npy + estimated-border dump)"
  key_links:
    - from: "crates/cb-train/src/estimated/estimated_features.rs"
      to: "cb_data::select_borders_greedy_logsum"
      via: "estimated column VALUES whose distinct-value distribution matches upstream's instrumented dump"
      pattern: "select_borders_greedy_logsum"
---

<objective>
Close the Phase 06.5 deferral: reproduce upstream catboost 1.2.10's estimated-feature (text/embedding) stored-border-VALUE quantization grid so a serialized model's estimated-feature borders match bit-for-bit (KNN integer-vote border serializes as 0.5, not 1.5; BoW digitization grid matches), AND a non-degenerate XOR-style text+embedding+numeric corpus matches StagedApprox + Predictions <=1e-5 with NO structure-invariant leaf-order relaxation. (FEAT-01 residual.)

Purpose: FEAT-01's one open residual — the general estimated-feature stored-border-VALUE grid parity — is the last unowned item from Phase 06.5. Predictions already match under the degenerate-separating SC-4 corpus; only the stored border VALUE (and the harder XOR gate) remain.

Output: A corrected Rust estimated-feature grid path + an XOR-corpus hard oracle proving exact stored borders and per-stage parity.

## Key prior finding (from this plan's pre-analysis — start here, do not re-derive)

The upstream estimated-feature border selector is NOT a separate algorithm. `data.cpp:537` calls `CreateEstimatedFeaturesData(params->DataProcessingOptions->FloatFeaturesBinarization.Get(), ...)` — estimated features use the SAME `BorderSelectionType` (default `GreedyLogSum`) and the SAME `BestSplit` path (`estimated_features.cpp:246-258`) as numeric features. `GreedyLogSum` dispatches to `TGreedyBinarizer<MaxSumLog>` (`binarization.cpp:118,1320-1520`), which is EXACTLY what Rust's `cb_data::select_borders_greedy_logsum` (`crates/cb-data/src/borders.rs`) already transcribes (the `TFeatureBin`/`GreedySplit` over the full sorted value slice, border = `0.5*v[start-1] + 0.5*v[start]`).

Therefore the `0.5` vs `1.5` divergence is NOT a border-algorithm gap — it is an estimated-COLUMN-VALUE gap. `1.5` is the midpoint of distinct values `{1,2}`; upstream's `0.5` is the midpoint of `{0,1}`. The Rust estimated columns (KNN vote counts in `offline_knn_features`, `crates/cb-train/src/estimated/online_embedding.rs:305`; BoW digitization in `build_bow_estimated_features`) feed a DIFFERENT distinct-value distribution into the (already-correct) binarizer. Task 1 confirms this empirically by dumping upstream's actual estimated-column VALUES and borders; Task 2 corrects the column-value path so the distinct-value distribution matches → borders match for free; Task 3 gates it under the hard XOR corpus.
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/todos/pending/estimated-feature-grid-parity.md
@.planning/phases/06.5-text-and-embedding-features/deferred-items.md
@.planning/phases/06.5-text-and-embedding-features/06.5-07-SUMMARY.md

# The Rust seam + the (already-correct) shared binarizer
@crates/cb-train/src/estimated/estimated_features.rs
@crates/cb-data/src/borders.rs

# Upstream chain (read the cited line ranges only)
# estimated features use FloatFeaturesBinarization (same as numeric):
#   catboost-master/catboost/private/libs/algo/data.cpp:537
# the BestSplit call site + the CB_INSTRUMENT_LOG estimated_borders dump:
#   catboost-master/catboost/private/libs/algo/estimated_features.cpp:225-269
# GreedyLogSum -> TGreedyBinarizer<MaxSumLog> -> TFeatureBin/GreedySplit:
#   catboost-master/library/cpp/grid_creator/binarization.cpp:118, 1320-1520
</context>

<tasks>

<task type="auto">
  <name>Task 1: Confirm the column-value root cause — dump upstream estimated-column VALUES + borders</name>
  <files>.planning/quick/260619-cpr-estimated-feature-stored-border-value-qu/INVESTIGATION.md (scratch notes only — NOT a deliverable .md per the no-report rule; keep findings in the SUMMARY instead), crates/cb-oracle/generator/gen_text_embedding_fixtures.py</files>
  <action>
Empirically confirm the pre-analysis finding that the stored-border-VALUE divergence is a column-VALUE divergence, not a border-algorithm divergence — this IS the investigation (no separate research phase ran).

(a) Reproduce a minimal mixed (or KNN-only) training run with catboost==1.2.10 from .venv (single-thread: thread_count=1, as the existing fixtures require) over the existing 16-row text+embedding corpus. Use the existing CB_INSTRUMENT_LOG `estimated_borders` instrumentation already baked into estimated_features.cpp:259 — it emits, per estimated feature, BOTH the raw column `"values":[...]` fed to BestSplit AND the resulting `"borders":[...]`. If the instrumented trainer is not currently present in /tmp (project memory notes it MAY persist), prefer NOT to rebuild the whole trainer: instead extract the upstream-stored estimated-feature borders directly from the serialized model.cbm (the python API `_get_tree_splits`/`splits.npy` path the generator already uses, gen_text_embedding_fixtures.py:270-293) AND compute the upstream estimated COLUMN VALUES from python catboost's own calcer apply (the per-stage fixtures already capture calcer outputs). The instrumented dump is the gold path; the model.cbm + calcer-apply path is the no-rebuild fallback. Document which path was used.

(b) Lay the two distributions side by side: upstream's KNN estimated-column distinct VALUES vs the Rust `offline_knn_features` distinct VALUES; same for the BoW digitization columns vs `build_bow_estimated_features`. Confirm the binarizer is identical (GreedyLogSum) and pin the exact divergence: e.g. upstream KNN vote column distinct values are {0,1} (or a normalized vote fraction) giving border 0.5, while Rust emits {1,2,...} giving 1.5 — OR a count-vs-fraction scaling, OR a 0-vs-1 base offset. Identify the precise transform (scale/offset/normalization) that maps the Rust column distribution onto upstream's. Record the exact transform and the cited upstream calcer line (KNN: the embedding feature calcer apply; BoW: the digitization grid) in the SUMMARY.

Keep scratch dumps under the quick dir; do NOT author a standalone findings .md — fold the conclusion into the final SUMMARY. No production code changes in this task.
  </action>
  <verify>
    <automated>cd /home/user/Documents/workspace/catboost_rs && .venv/bin/python -c "import catboost; print(catboost.__version__)" | grep -q '1.2.10' && echo OK</automated>
  </verify>
  <done>The exact column-value transform that maps the Rust estimated-column distinct-value distribution onto upstream's is identified and written down (with the upstream calcer line cited), and it is empirically shown that feeding upstream's value distribution through the UNCHANGED select_borders_greedy_logsum reproduces upstream's stored border (0.5 for the KNN vote column). The "is the algorithm the same?" question is answered YES with evidence.</done>
</task>

<task type="auto">
  <name>Task 2: Wire the Rust estimated-feature grid path to reproduce upstream's stored border VALUES bit-for-bit</name>
  <files>crates/cb-train/src/estimated/online_embedding.rs, crates/cb-train/src/estimated/estimated_features.rs, crates/cb-train/src/estimated/estimated_features_test.rs</files>
  <action>
Apply the Task-1 transform to the estimated-COLUMN-VALUE computation (NOT to the border algorithm — select_borders_greedy_logsum stays unchanged, SC-4 "no parallel quantizer" contract). Concretely: correct the KNN estimated-column values in `offline_knn_features` (online_embedding.rs:305-346) and, if Task 1 shows a BoW digitization divergence, the BoW presence/grid columns in `build_bow_estimated_features` (estimated_features.rs:86-177) so each estimated column's DISTINCT-VALUE distribution matches upstream's instrumented dump. Because the binarizer is already upstream-exact, the stored borders then follow automatically (KNN vote column -> 0.5; BoW grid -> matching borders).

Honor CLAUDE.md hard rules: NO `unwrap()`/`expect()`/`panic`/raw-index in production (use checked `.get(..)` + typed `CbError`, matching the existing module style); source/test separation is mandatory — add the assertions to the sibling `estimated_features_test.rs` (and/or the online_embedding test file), NEVER a `mod tests` in the production source. If the transform is a value scaling that could alter prediction routing, verify the induced PARTITION is unchanged (the transform must be monotone/order-preserving so the tree structure is invariant while the stored border VALUE moves to upstream's).

Add unit tests asserting the corrected column distinct-value distribution and that `select_borders_greedy_logsum` over it yields the upstream border (e.g. the KNN {0,1} vote column yields exactly border 0.5, not 1.5). Keep D-04 inert-when-absent byte-identical (empty pool -> empty layout, existing numeric path unchanged).
  </action>
  <verify>
    <automated>cd /home/user/Documents/workspace/catboost_rs && cargo test -p cb-train --lib estimated:: 2>&1 | tail -5 && cargo clippy -p cb-train --lib 2>&1 | grep -E 'unwrap_used|expect_used|panic|indexing_slicing' | grep -v '^0' | grep estimated && echo "LINT-FAIL" || echo "LINT-CLEAN"</automated>
  </verify>
  <done>The corrected estimated columns, fed through the UNCHANGED select_borders_greedy_logsum, produce upstream's stored border VALUES (KNN integer-vote -> 0.5 not 1.5; BoW digitization grid matches). New unit tests assert this. cb-train estimated tests pass; no new restriction-lint violations in the touched estimated files; source/test separation honored; D-04 inert path byte-identical.</done>
</task>

<task type="auto">
  <name>Task 3: Re-introduce the XOR corpus as the HARD oracle (exact borders + per-stage <=1e-5, no relaxation)</name>
  <files>crates/cb-oracle/generator/gen_text_embedding_fixtures.py, crates/cb-oracle/tests/text_embedding_end_to_end_oracle_test.rs, crates/cb-oracle/fixtures/text_embedding_xor/</files>
  <action>
Re-introduce the non-degenerate XOR-structured text+embedding+numeric corpus that 06.5-07 prototyped and REJECTED (it forces the model onto exact KNN vote-count + BoW digitization-grid parity — the very thing Task 2 now closes). Add an `--xor` arm to gen_text_embedding_fixtures.py (mirror the existing `--mixed`/`gen_mixed` arm, gen_text_embedding_fixtures.py:368+) that builds an XOR(text_bit, embed_bit) target so BOTH estimated features are unambiguously load-bearing (no feature-selection tie, so leaf ORDER is determined — no structure-invariant relaxation is permitted or needed). Freeze the fixture under fixtures/text_embedding_xor/ with catboost==1.2.10, thread_count=1: model.cbm + staged.npy + predictions.npy + splits.npy (the stored estimated-feature borders) + numeric.npy + meta.

Add oracle tests to text_embedding_end_to_end_oracle_test.rs gating the XOR model:
  1. estimated-feature stored borders match upstream bit-for-bit (the KNN vote border is exactly 0.5; the BoW digitization grid matches splits.npy) — HARD, exact.
  2. StagedApprox matches upstream <=1e-5 — HARD.
  3. Predictions match upstream <=1e-5 — HARD.
  4. Splits/LeafValues match upstream IN ORDER (NO structure-invariant / leaf-MULTISET relaxation — the XOR target removes the tie that justified the 06.5-07 relaxation; if an in-order compare still fails, that is a REAL gap to fix in Task 2's transform, not to relax).
NO `#[ignore]`, NO weakened tolerance. The oracle test file carries the standard top-of-file test-lint exemption; production untouched here.

If the XOR gate cannot be made green within this quick task's budget despite Task 2's fix (e.g. a second, deeper estimated-feature-grid divergence surfaces under XOR that was masked by the degenerate corpus), STOP and record the precise residual (which column, which distinct values diverge, expected vs actual border) in the SUMMARY as a scoped follow-up — do NOT relax the gate, add `#[ignore]`, or weaken tolerance to force green. The done-when bar is exact-or-documented-residual, never weakened.
  </action>
  <verify>
    <automated>cd /home/user/Documents/workspace/catboost_rs && cargo test -p cb-oracle --test text_embedding_end_to_end_oracle_test 2>&1 | tail -8 && ! grep -RnE '#\[ignore\]|EPS *= *0\.0[1-9]|tolerance.*1e-[1-3]\b' crates/cb-oracle/tests/text_embedding_end_to_end_oracle_test.rs</automated>
  </verify>
  <done>The XOR mixed corpus trains a model whose serialized estimated-feature borders match upstream bit-for-bit (KNN vote border = 0.5, BoW grid matches) AND StagedApprox + Predictions match <=1e-5 with Splits/LeafValues compared IN ORDER (no leaf-order relaxation), 0 ignored, 0 weakened tolerance — OR a precise residual divergence is documented as a scoped follow-up with NO gate relaxation.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| training data -> estimated-feature quantization | estimated column values feed the border selector; a wrong transform silently shifts every downstream bin boundary |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-cpr-01 | Tampering | estimated-column value transform (Task 2) altering tree PARTITION, not just stored border | mitigate | require the value transform to be monotone/order-preserving; assert the induced partition (tree structure) is invariant in Task 2 unit tests and confirmed by the Task 3 in-order Splits compare |
| T-cpr-02 | Information disclosure | weakening the oracle (relaxed tolerance / #[ignore] / leaf-order freedom) to force a false green | mitigate | Task 3 verify greps for #[ignore] / weakened EPS; done-when is exact-or-documented-residual; CLAUDE.md parity bar <=1e-5 enforced |
| T-cpr-SC | Tampering | npm/pip/cargo installs | accept | no new package-manager installs; catboost==1.2.10 already in .venv, all crates already in the workspace |
</threat_model>

<verification>
- `cd /home/user/Documents/workspace/catboost_rs && cargo test -p cb-train --lib` passes (no regression of the estimated-feature seam or the boosting/tree core).
- `cargo test -p cb-oracle --test text_embedding_end_to_end_oracle_test` passes (existing SC-4 mixed + new XOR oracle), 0 ignored.
- `cargo clippy -p cb-train --lib` reports 0 of the four denied restriction lints (unwrap_used/expect_used/panic/indexing_slicing) in the touched estimated files.
- D-04 non-regression: existing no-text/no-embedding e2e oracles unchanged (the estimated path stays inert when absent).
</verification>

<success_criteria>
- A non-degenerate XOR-style text+embedding+numeric corpus trains a model whose serialized estimated-feature borders match upstream catboost 1.2.10 bit-for-bit, AND StagedApprox + Predictions match <=1e-5 with NO structure-invariant leaf-order relaxation.
- KNN vote border serializes as 0.5 (not 1.5); BoW digitization grid matches.
- No `#[ignore]`, no weakened tolerance.
- (Acceptable terminal state if a deeper masked divergence surfaces under XOR: the residual is precisely documented as a scoped follow-up — column, distinct values, expected vs actual border — with the gate NOT relaxed.)
- CLAUDE.md honored: source/test separation (no `mod tests` in production), no `unwrap()` in production, parity bar <=1e-5.
</success_criteria>

<output>
Create `.planning/quick/260619-cpr-estimated-feature-stored-border-value-qu/260619-cpr-SUMMARY.md` when done. Fold the Task-1 investigation findings (the column-value transform + cited upstream lines) into the SUMMARY — do NOT leave a standalone findings .md as the deliverable.
</output>
