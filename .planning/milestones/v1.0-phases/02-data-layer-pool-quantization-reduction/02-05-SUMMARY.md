---
phase: 02-data-layer-pool-quantization-reduction
plan: 05
subsystem: data-layer
tags: [ingest, arrow, polars, weights, class-weights, balanced, sqrt-balanced, oracle, parity, f32-f64]

# Dependency graph
requires:
  - phase: 02-data-layer-pool-quantization-reduction
    provides: "Pool + IngestSource seam (02-02), cb_core::sum_f64 reduction primitive (02-01), class_weights fixtures (02-01)"
provides:
  - "cb_data::ingest::ArrowColumns / arrow_f64_column — Arrow Float64Array -> validated owned Vec<f64> with typed CbError (DATA-06, D-06)"
  - "cb_data::ingest::PolarsColumns — Polars DataFrame -> rechunk -> shared Arrow validation path (DATA-06, D-05)"
  - "CbError ingestion taxonomy: Dtype / LengthMismatch / NanInCategorical / Ingestion (Clone+PartialEq+Eq preserved; the surface Phase 8 maps to Python exceptions, PYAPI-05)"
  - "cb_data::{balanced_class_weights, sqrt_balanced_class_weights, summary_class_weights, resolve_object_weights, MINIMAL_CLASS_WEIGHT} — auto class weights + per-object/per-class resolver (DATA-08)"
affects: [phase-08-pyo3-ingestion, all downstream training plans consuming weights + validated external ingestion]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Single shared ingestion funnel: arrow_f64_column is the ONE dtype/NaN-in-categorical/contiguity validator; both the Arrow IngestSource and the Polars path call it (Polars rechunks -> cont_slice -> Float64Array -> same funnel; no duplicated validation, D-05)"
    - "Ingestion errors STRINGIFY external arrow/polars errors (a String field, never #[from]) so CbError keeps Clone/PartialEq/Eq (Shared Pattern C)"
    - "Class-weight arithmetic in f32 to bit-match upstream's float lambdas; summary sums in f64 via cb_core::sum_f64; widened to f64 only at the oracle compare"
    - "1e-8 floor as a guard branch returning 1.0 (not 1e-8) on a degenerate/empty class — matches calc_class_weights.cpp:11-27, no div-by-zero"

key-files:
  created:
    - crates/cb-data/src/ingest/arrow.rs
    - crates/cb-data/src/ingest/arrow_test.rs
    - crates/cb-data/src/ingest/polars.rs
    - crates/cb-data/src/ingest/polars_test.rs
    - crates/cb-data/src/weights.rs
    - crates/cb-data/src/weights_test.rs
    - crates/cb-data/tests/weights_oracle_test.rs
  modified:
    - crates/cb-core/src/error.rs
    - crates/cb-core/src/error_test.rs
    - crates/cb-data/src/ingest/mod.rs
    - crates/cb-data/src/lib.rs
    - .planning/phases/02-data-layer-pool-quantization-reduction/deferred-items.md

key-decisions:
  - "Polars rides the shared Arrow path literally: rechunk() each column -> Float64Chunked::cont_slice() (typed error if non-contiguous) -> arrow::Float64Array -> arrow_f64_column. This honors the key_link `rechunk -> shared Arrow validation path` and avoids polars/arrow-crate type incompatibility (polars uses its own arrow flavor)."
  - "Ingestion CbError variants stringify external errors (D-06 / Shared Pattern C): preserves Clone+PartialEq+Eq, which error_test asserts (new_variants_preserve_clone_and_eq)."
  - "Class weights computed in f32 (max/w and sqrtf(max/w)); the SqrtBalanced fixture value 1.7320507764816284 is CatBoost's f32 sqrt(3) widened to f64 — matched by computing in f32, absorbed by the <=1e-5 oracle tolerance, fixture left unchanged."
  - "The 1e-8 floor branch returns 1.0 (upstream MINIMAL_CLASS_WEIGHT is the THRESHOLD, the empty-class weight is 1.0) — covered by empty_class_hits_floor_branch_with_no_panic."

patterns-established:
  - "External-source ingestion validates at a single funnel and returns typed CbError; new sources add an IngestSource impl that calls the funnel rather than re-validating"
  - "Source/test separation held: every new module has a sibling *_test.rs; oracle parity lives in tests/ integration files"

requirements-completed: [DATA-06, DATA-08]

# Metrics
duration: ~25min
completed: 2026-06-13
---

# Phase 2 Plan 05: Arrow + Polars Ingestion Seam & Auto Class-Weight Oracle Summary

**The external-ingestion + weights vertical slice that closes the Phase-2 data layer: Arrow `Float64Array` and Polars `DataFrame` sources flow through ONE shared `arrow_f64_column` validator (dtype / NaN-in-categorical / contiguity / length → typed `CbError`, Clone+Eq preserved), with Polars rechunking onto the exact same funnel, and the Balanced / SqrtBalanced auto class-weight calculators (plus the per-object/per-class resolver) oracle-locked to upstream ≤1e-5 with the 1e-8 empty-class floor covered.**

## Performance
- **Duration:** ~25 min
- **Completed:** 2026-06-13
- **Tasks:** 2 (both `auto`/`tdd`, both committed atomically)
- **Files:** 12 changed (7 created, 5 modified)

## Accomplishments

### Task 1 — Arrow + Polars ingestion via the shared validation seam (DATA-06) — `a657f60`
- Extended `cb_core::CbError` with four ingestion variants in the existing thiserror style: `Dtype { expected: &'static str, got: String }`, `LengthMismatch { column, expected, actual }`, `NanInCategorical { column }`, and `Ingestion { message }`. External arrow/polars errors are STRINGIFIED into the `message`/`got` fields (never `#[from]`), so the enum keeps `Clone + PartialEq + Eq` (Shared Pattern C) — asserted by `new_variants_preserve_clone_and_eq`.
- `ingest/arrow.rs`: `arrow_f64_column(column, index, categorical)` validates `data_type() == Float64` (`CbError::Dtype` otherwise), rejects any `NaN` in a declared-categorical column (`CbError::NanInCategorical`), and reads the contiguous backing buffer (`Float64Array::values()`, safe `downcast_ref` — no panicking `as_primitive`) into an owned `Vec<f64>` identical to the owned-`Vec` path. `ArrowColumns` `IngestSource` impl assembles float columns + label, enforcing per-column length against `n_rows` (`CbError::LengthMismatch`).
- `ingest/polars.rs`: `PolarsColumns` `IngestSource` resolves each named column from the `DataFrame`, `rechunk()`s it to one chunk, takes `Float64Chunked::cont_slice()` (a non-contiguous / non-f64 column is a typed `CbError::Ingestion`, never a panic), wraps the slice into an `arrow::Float64Array`, and routes through the SAME `arrow_f64_column` validator (D-05 — no duplicated validation logic).
- Dedicated `error_test.rs` (+5 cases), `arrow_test.rs` (7 cases), `polars_test.rs` (4 cases): owned-Vec-vs-Arrow column equality, Arrow-vs-Polars column equality, dtype-mismatch / NaN-in-categorical / length-mismatch / missing-column typed errors.

### Task 2 — Balanced/SqrtBalanced auto class weights + per-object/per-class resolver (DATA-08) — `6d71008`
- `weights.rs` ports `calc_class_weights.cpp`: `summary_class_weights` buckets each object's weight by class (preserving object order) and folds each bucket through `cb_core::sum_f64` (D-07/D-08 — no raw `.sum()`/`.fold(0.0,+)`); `balanced_class_weights` computes `w > 1e-8 ? max/w : 1.0` and `sqrt_balanced_class_weights` computes `w > 1e-8 ? sqrt(max/w) : 1.0`, both in `f32` (matching upstream's `float` lambdas). The `1e-8` floor branch returns `1.0` on a degenerate/empty class (no div-by-zero, no `inf`).
- `resolve_object_weights` expands per-class weights to a per-object vector and multiplies by explicit per-object weights (empty = all-ones for either input), matching upstream's per-object × class-weight composition.
- `weights_test.rs` (11 cases): summary counts, Balanced/SqrtBalanced algebra, the `empty_class_hits_floor_branch_with_no_panic` floor case (asserts all weights finite), error paths (zero class count, out-of-range class index, length mismatch), and resolver algebra.
- `tests/weights_oracle_test.rs`: reconstructs the `class_counts=[30,10]` binary dataset, computes Balanced/SqrtBalanced, widens f32→f64, and gates against `class_weights/{balanced,sqrt_balanced}.npy` with `assert_abs_close(..., 1e-5)` — both pass.

## Verification (all green)
- `cargo test -p cb-core error` — 9 pass (4 new ingestion-variant Display + Clone/Eq cases).
- `cargo test -p cb-data ingest` — 15 pass (owned + arrow + polars; owned-vs-Arrow and Arrow-vs-Polars equality).
- `cargo test -p cb-data --lib weights_test` — 11 pass (incl. 1e-8 floor branch).
- `cargo test -p cb-data --test weights_oracle_test` — 2 pass (Balanced + SqrtBalanced ≤1e-5).
- `cargo test --workspace` — full Phase-2 suite green (cb-data 57 unit + borders/cat-hash/quantize/weights oracle tests; cb-core 21; cb-oracle 15 + integration).
- `bash scripts/check-no-raw-float-sum.sh` — exits 0 (D-08; all weight sums via cb_core::sum_f64).
- `bash scripts/check-no-anyhow.sh` — exits 0 (D-14).
- `cargo clippy -p cb-core --lib` and `cargo clippy -p cb-data --lib -- -D warnings` — both clean.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Doc-comment text tripped the D-08 grep gate**
- **Found during:** Task 2 (first D-08 gate run).
- **Issue:** `weights.rs`'s module doc literally contained the string `.fold(0.0, +)` (describing what is banned), which the D-08 regex matched as a real violation.
- **Fix:** Reworded the doc to "never a raw summation form" — no banned literal. Behavior unchanged; the only float sums are through `cb_core::sum_f64`.
- **Files modified:** `crates/cb-data/src/weights.rs`
- **Commit:** `6d71008`

**2. [Rule 3 - Blocking] clippy `doc_list_item_without_indentation` on a nested doc list**
- **Found during:** Task 2 (`clippy -p cb-data --lib`).
- **Issue:** The module doc used a nested Markdown list whose continuation lines clippy (rust-1.96.0) flagged.
- **Fix:** Restructured the doc into prose paragraphs.
- **Files modified:** `crates/cb-data/src/weights.rs`
- **Commit:** `6d71008`

## Deferred Issues (out of scope — pre-existing, logged to deferred-items.md)
- `cargo clippy --workspace --lib -- -D warnings` fails on TWO pre-existing lints surfacing only under the newer toolchain (rust-1.96.0), neither caused by this plan:
  - cb-oracle `compare.rs:44` `neg_cmp_op_on_partial_ord` on `!(diff <= tol)` (the intentional NaN-aware divergence check from Phase-1 commit `902368d`; already logged under 02-03/02-04).
  - cb-core `error_test.rs` `unnecessary_literal_unwrap` on the pre-existing `cb_result_ok_path_round_trips` test (`Ok(42).unwrap()`); my edit only shifted its line number.
- Both are in files this plan did not author. The plan's per-crate clippy gates (`-p cb-core --lib`, `-p cb-data --lib`) are clean. Recommend a cb-oracle/cb-core housekeeping pass.

## Known Stubs
None. Both ingestion paths read real owned columns through real validation; the weight calculators are the full algorithm (no placeholder). The `ArrowColumns`/`PolarsColumns` sources expose float-features + label (the supervised case the oracle exercises); categorical/text/embedding Arrow columns are a natural Phase-8 extension of the same seam, not a stub blocking DATA-06.

## Threat Flags
None. The plan's four trust boundaries are all mitigated and test-locked: T-02-13 (dtype/length → `CbError::Dtype`/`LengthMismatch`, no blind index), T-02-14 (NaN-in-categorical → `CbError::NanInCategorical`), T-02-15 (div-by-zero on empty class → 1e-8 floor, no panic), T-02-16 (unaudited float sum → `cb_core::sum_f64` + D-08 gate + 1e-5 oracle). No new network/auth/file-access surface; no new package installs (T-02-SC).

## Notes for Downstream Plans
- Phase 8 (PyO3) reuses `arrow_f64_column` + the `CbError` ingestion taxonomy directly; mapping `Dtype`/`LengthMismatch`/`NanInCategorical`/`Ingestion` to Python exceptions is PYAPI-05.
- `ArrowColumns`/`PolarsColumns` currently surface float-features + label; categorical/text/embedding columns plug into the same `IngestSource` seam by extending those structs — `Pool` does not change (D-02).
- `resolve_object_weights` is the per-object weight vector the training loop will consume once class weights or explicit weights are configured.

## Self-Check: PASSED
- Created files exist: ingest/{arrow,arrow_test,polars,polars_test}.rs, weights.rs, weights_test.rs, tests/weights_oracle_test.rs — all present.
- Commits present: `a657f60` (Task 1), `6d71008` (Task 2) — both in git history.
- All plan verification commands green except the two pre-existing out-of-scope clippy lints (logged to deferred-items.md); plan-mandated per-crate gates clean.

---
*Phase: 02-data-layer-pool-quantization-reduction*
*Completed: 2026-06-13*
