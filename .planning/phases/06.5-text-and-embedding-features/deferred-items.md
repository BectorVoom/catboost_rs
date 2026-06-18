# Deferred Items — Phase 06.5

## Out-of-scope pre-existing lint (06.5-03)

- **`crates/cb-backend/src/cpu_runtime.rs:696` and `:1025`** — `cargo clippy -p cb-backend --lib`
  reports `error: indexing may panic` / `error: slicing may panic`. These are PRE-EXISTING in
  `cb-backend` (a dependency compiled transitively when linting cb-train); the file was NOT touched
  by plan 06.5-03. Out of scope per the SCOPE BOUNDARY rule (not caused by this plan's changes).
  The new lib files added by 06.5-03 (`cb-compute/src/text_calcers.rs`,
  `cb-data/src/text/bigram_dictionary.rs`, `cb-train/src/estimated/estimated_features.rs`) are
  clean of all four restriction lints (0 indexing_slicing/unwrap_used/expect_used/panic).

## BM25 per-stage normalized-border scale (06.5-04) — CLOSED (06.5-09, PATH-A)

**Status: CLOSED. This was a FIXTURE MISLABEL, not a real BM25 normalization. No trainer
normalization exists or was needed.**

- **Original symptom (06.5-04):** `crates/cb-oracle/fixtures/text_calcers/BM25/splits.npy` stored
  ±1.24 / -0.550486 borders while the raw BM25 calcer scores — exact against an independent
  closed-form `bm25.cpp:12-83` re-derivation, both online and offline — are O(1e-3). It was deferred
  as a presumed catboost estimated-feature value-NORMALIZATION + multi-permutation averaging concern.
- **Resolution (06.5-08 instrumented dump + 06.5-09 fix):** the ±1.24 borders were **never a BM25
  normalization**. 06.5-08's `cb_instr_estimated_borders` dump proved the genuine BM25 estimated-
  feature borders are O(1e-3) (`BestSplit` selects them verbatim from the raw column — source chain
  `base_text_feature_estimator.h:74-88` → `estimated_features.cpp:204-250` → `split.cpp:45-46` →
  `model.cpp:209` is entirely scale-preserving, no transform/averaging-rescale/standardization). The
  committed `splits.npy` ±1.24 borders ALL carried `calcer_id=96AE6D4D…` — the **DEFAULT EMBEDDING
  calcer on the `emb0` column**, NOT the BM25 text calcer. The fixture's pool inadvertently included
  `embedding_features=["emb0"]` (the generator's `_make_pool` default); the well-separated embedding
  clouds (centers ±1.0) dominated the split search, so the tree split on the EMBEDDING feature and
  `splits.npy` recorded the embedding feature's borders, mislabeled as BM25's. The genuine depth-2
  `[7,2,0,7]` structure was likewise the embedding clouds'.
- **06.5-09 fix (PATH-A, fixture-correctness):** `gen_text_embedding_fixtures.py::_make_pool` gains
  `text_only=True`; the text-calcer path (BoW/NaiveBayes/BM25) drops `emb0` so the fixture records the
  genuine BM25 **text** feature. The regenerated BM25 `splits.npy` is O(1e-3) (`0.00248965, 0.00127047,
  …`, `calcer_id=0BDFE5…`). The full BM25 per-stage oracle (`bm25_oracle_splits_match_upstream` /
  `_leaf_values_match_upstream` / `_staged_approx_match_upstream` / `_predictions_match_upstream`) is
  GREEN ≤1e-5, 0 ignored — the same gate NaiveBayes passes. Splits/LeafValues come from the ONLINE-
  estimate tree; StagedApprox/Predictions are applied through the OFFLINE whole-set column (the Plain-
  mode online-tree / offline-apply contract `online_text.rs` documents — the one place BM25 differs
  from NaiveBayes, since BM25's online no-leakage doc-0 value is 0 but its offline value routes doc 0
  to the correct leaf). **NO production trainer change** (the Rust seam already produced the O(1e-3)
  borders), NO `#[ignore]`, NO weakened tolerance. FEAT-01 / SC-2 BM25 per-stage closed.

## Out-of-scope pre-existing dead const (06.5-06)

- **`crates/cb-train/src/estimated/online_embedding_test.rs:13`** — `const DIM: usize = 4;`
  emits a `dead_code` warning (`DIM` is never referenced). It was introduced UNUSED by plan
  06.5-05 (commit `4c194ae`), NOT by 06.5-06. Out of scope per the SCOPE BOUNDARY rule (pre-
  existing, not caused by this plan's KNN additions). Warning-only (not a denied lint); does not
  affect the build or any test. Left untouched.

## General estimated-feature quantization-GRID parity (06.5-07)

- **What:** the SC-4 *join* (mixed text+embedding+numeric → existing quantizer → tree) is CLOSED in
  06.5-07 — the combined model's StagedApprox + Predictions match upstream catboost 1.2.10 ≤1e-5
  bit-for-bit. What remains open is the exact estimated-feature *quantization GRID* (the border VALUES
  upstream stores for estimated columns), which generalizes the 06.5-04 BM25 ±1.24 normalized-border
  deferral to the other calcers:
  - **KNN integer-vote border:** upstream stores `0.5` for the KNN class-vote split; the Rust
    `select_borders_greedy_logsum` on the `{0, k}` vote distribution returns the midpoint (e.g. `1.5`).
    Both induce the SAME 8/8 partition, so predictions match — but the stored border VALUE differs.
  - **BoW / digitization grid:** an XOR-structured non-degenerate mixed corpus (prototyped in 06.5-07
    and REJECTED) forces the model onto exact KNN vote-count + BoW digitization grid parity; under it
    the staged/predictions did NOT match ≤1e-5, confirming the grid is a distinct, still-open concern.
- **Why deferred:** this is a TRAINER estimated-feature-normalization / serialization concern (how
  catboost picks the stored border grid for estimated features), NOT a calcer-math or SC-4-join defect.
  The 06.5-07 SC-4 oracle isolates the JOIN question (closed) from the GRID question (open) by using a
  degenerate-separating corpus + structure-invariant Splits/LeafValues gating (per-tree leaf MULTISET
  ≤1e-5; magnitudes exact, only the ambiguous leaf ORDER freed).
- **Impact (updated 06.5-09):** FEAT-01 IS now fully closed — the "BM25 per-stage normalized borders"
  item above was resolved as a fixture mislabel (06.5-08/09), not a real grid concern, so all three
  text calcers BoW/NaiveBayes/BM25 are per-stage ≤1e-5. FEAT-02 IS closed (LDA documented-tolerance +
  KNN bit-exact; SC-4 re-exercises KNN end-to-end ≤1e-5). What remains here is the GENERAL estimated-
  feature stored-border-VALUE grid (KNN integer-vote `0.5` vs `1.5`; BoW digitization grid under a
  non-degenerate XOR corpus) — predictions/staged match ≤1e-5 under the degenerate-separating SC-4
  corpus, only the stored border VALUE differs. This is a separate, still-open TRAINER estimated-
  feature quantization-grid concern (NOT FEAT-01/SC-2, which are closed); a follow-on
  trainer-estimated-feature-grid slice should own it. It does NOT block FEAT-01 or FEAT-02.
