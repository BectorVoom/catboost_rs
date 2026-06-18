# Deferred Items — Phase 06.5

## Out-of-scope pre-existing lint (06.5-03)

- **`crates/cb-backend/src/cpu_runtime.rs:696` and `:1025`** — `cargo clippy -p cb-backend --lib`
  reports `error: indexing may panic` / `error: slicing may panic`. These are PRE-EXISTING in
  `cb-backend` (a dependency compiled transitively when linting cb-train); the file was NOT touched
  by plan 06.5-03. Out of scope per the SCOPE BOUNDARY rule (not caused by this plan's changes).
  The new lib files added by 06.5-03 (`cb-compute/src/text_calcers.rs`,
  `cb-data/src/text/bigram_dictionary.rs`, `cb-train/src/estimated/estimated_features.rs`) are
  clean of all four restriction lints (0 indexing_slicing/unwrap_used/expect_used/panic).

## BM25 per-stage normalized-border scale (06.5-04)

- **`crates/cb-oracle/fixtures/text_calcers/BM25/splits.npy`** stores catboost's BM25 estimated-
  feature split borders in a NORMALIZED internal scale (`splits.npy` reaches ±1.24) while the raw
  BM25 calcer scores — verified exact against an independent closed-form `bm25.cpp:12-83`
  re-derivation, both online and offline — are O(1e-3). The upstream BM25 tree is also a genuine
  depth-2 structure (`leaf_weights = [7,2,0,7]`) produced by catboost's estimated-feature averaging
  across `permutation_count` permutations.
- **Root cause (investigated exhaustively in 06.5-04):** neither the online read-before-update
  prefix nor the offline whole-set estimate produces values near the ±1.24 border scale; no single
  learn permutation's strict prefix reproduces the instrumented NaiveBayes per-prefix dump's
  interior either. This is catboost's internal estimated-feature value NORMALIZATION + multi-
  permutation ordered averaging — a TRAINER/serialization concern, NOT a BM25 calcer-math defect.
- **What 06.5-04 gates instead (no `#[ignore]`, no weakened tolerance):** the BM25 calcer math +
  online seam are oracle-green at the calcer-encoding level (`bm25_oracle_columns_match_closed_form`
  ≤1e-5 vs an independent closed-form online reference, the no-leakage empty-prefix anchor, and the
  SC-4 quantizer integration). NaiveBayes is FULLY per-stage oracle-green (Splits/LeafValues/
  StagedApprox/Predictions ≤1e-5) because its online column's split border (0.590515) and clean 8/8
  separation are robust to the normalization. The BM25 NORMALIZED per-stage borders are deferred to
  the trainer estimated-feature-normalization work (a follow-on slice / Phase 6.6 trainer concern).
