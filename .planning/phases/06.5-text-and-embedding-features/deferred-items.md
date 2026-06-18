# Deferred Items — Phase 06.5

## Out-of-scope pre-existing lint (06.5-03)

- **`crates/cb-backend/src/cpu_runtime.rs:696` and `:1025`** — `cargo clippy -p cb-backend --lib`
  reports `error: indexing may panic` / `error: slicing may panic`. These are PRE-EXISTING in
  `cb-backend` (a dependency compiled transitively when linting cb-train); the file was NOT touched
  by plan 06.5-03. Out of scope per the SCOPE BOUNDARY rule (not caused by this plan's changes).
  The new lib files added by 06.5-03 (`cb-compute/src/text_calcers.rs`,
  `cb-data/src/text/bigram_dictionary.rs`, `cb-train/src/estimated/estimated_features.rs`) are
  clean of all four restriction lints (0 indexing_slicing/unwrap_used/expect_used/panic).
