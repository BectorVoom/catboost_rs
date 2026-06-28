---
phase: 02-data-layer-pool-quantization-reduction
verified: 2026-06-13T00:00:00Z
status: human_needed
score: 8/8 must-haves verified
overrides_applied: 0
human_verification:
  - test: "Confirm that the WR-01 STL heap tie-break port (heap_push/heap_pop/adjust_heap in borders.rs) matches libstdc++ __push_heap/__pop_heap/__adjust_heap behavior precisely on a small hand-constructed tie case. Construct two adjacent bins with exactly equal best_score, pop one, and verify which bin surfaces as the heap root matches what a C++ std::priority_queue<TBinType> would pop."
    expected: "The Rust heap operations select the same bin as the libstdc++ binary max-heap would on a tied-score pop. The existing permutation-invariance and constant-column regression tests in borders_test.rs pass and the two borders oracle tests (numeric_tiny, numeric_nan) remain green — which is strong indirect evidence — but a direct tie-case trace is needed for bit-exact parity confidence."
    why_human: "The STL heap pop order for equal-score bins is determined by binary-heap array structure (not documented as a stable contract), and verifying it requires either running a C++ reference or an expert reading of the libstdc++ __adjust_heap sift-down code against the Rust port. The oracle tests cover this indirectly (they pass) but do not prove the tie-break path was exercised."
---

# Phase 2: Data Layer — Pool + Quantization + Reduction Verification Report

**Phase Goal:** Deliver the Phase-2 data layer — owned `Pool` + immutable `QuantizedPool`, oracle-validated `GreedyLogSum` borders, NaN-mode quantization, bit-exact CityHash64 categorical hashing, Arrow/Polars ingestion, auto class-weights, and the single audited deterministic reduction primitive — all oracle-locked to upstream CatBoost within ≤1e-5.
**Verified:** 2026-06-13
**Status:** human_needed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | A single sequential f64 reduction primitive exists in cb-core and is re-exported | VERIFIED | `crates/cb-core/src/reduction.rs` — `pub fn sum_f64` + `pub fn sum_f32_in_f64`; `lib.rs` line 23: `pub use reduction::{sum_f32_in_f64, sum_f64};` |
| 2 | The reduction primitive returns the naive-sequential result (not Kahan/pairwise) on the adversarial input [1e16, 1.0, -1e16] | VERIFIED | `reduction_test.rs` `sum_f64_naive_order_loses_small_term`: `assert_eq!(sum_f64(&[1e16, 1.0, -1e16]), 0.0)`; test passes |
| 3 | A CI grep gate fails the build on raw float .sum()/.fold(0.0, +) in library crates, excluding reduction.rs and *_test.rs | VERIFIED | `scripts/check-no-raw-float-sum.sh` exists, is wired in `.github/workflows/ci.yml` line 54; `bash scripts/check-no-raw-float-sum.sh` exits 0 on current tree |
| 4 | Wave-0 oracle fixtures are generated from standalone sources and committed with resolved assumptions A1–A5 | VERIFIED | `borders_quant/config.json` records A1/A2/A3 (EMPIRICAL, standalone `Pool.quantize().save_quantization_borders()`); `cat_hash/config.json` records A4/A5 (standalone C++ oracle `cityhash_oracle.cpp`); `class_weights/config.json` records source; fixture files confirmed present |
| 5 | Pool holds float/categorical/text/embedding columns plus label, weights, group_id, subgroup_id, pairs, baseline | VERIFIED | `pool.rs` struct fields: `float_features`, `cat_features`, `text_features`, `embedding_features`, `label`, `weights`, `group_id`, `subgroup_id`, `pairs`, `baseline`; 68 cb-data unit tests pass |
| 6 | GreedyLogSum border selection on frozen corpora matches get_borders() per feature at ≤1e-5 | VERIFIED | `borders_oracle_test.rs` calls `compare_stage(Stage::Borders, ...)` for both `numeric_tiny` (no sentinel) and `numeric_nan` (f32::MIN sentinel on feature 0); both tests pass |
| 7 | NanMode sentinel handling and strict value>border bin assignment are correct; NanMode::Max gives NaN a dedicated top bin (CR-01 fix wired) | VERIFIED | `nan_mode_test.rs` covers Min/Max/Forbidden sentinel insertion, `value == border` lower-bin placement, NaN→0/top-bin semantics; `quantize_test.rs` `max_nan_mode_gives_nan_its_own_top_bin` asserts `f32::MAX` sentinel appended and NaN bin strictly above all finite bins; test passes |
| 8 | QuantizedPool stores u8/u16 (float) and u32 (cat) bin indices in immutable columnar SoA | VERIFIED | `quantized_pool.rs`: `pub enum ColumnBins { U8(Vec<u8>), U16(Vec<u16>), U32(Vec<u32>) }` + `pub struct QuantizedPool`; width selection: `<256` → U8, `<65536` → U16; immutable after build (no mutable-scratch accessor) |
| 9 | pool.quantize(&params) → QuantizedPool quantizes float features end-to-end and oracle-matches borders on numeric_nan corpus | VERIFIED | `quantize_oracle_test.rs` runs `pool.quantize(&QuantizeParams::default())`, compares per-feature borders to `borders_quant/numeric_nan.*` via `compare_stage`, asserts NaN rows land in bin 0 (Min mode); passes |
| 10 | CityHash64 ported bit-exactly; CalcCatFeatureHash = CityHash64(bytes) & 0xffffffff matches upstream ui32 vectors | VERIFIED | `cat_hash_test.rs` asserts bit-exact `(string → ui32)` vectors from `cat_hash/config.json`; `cat_hash_oracle_test.rs` confirms `calc_cat_feature_hash` and `perfect_hash_bins` match fixture; test passes |
| 11 | Arrow ingestion validates dtype/null-bitmap/NaN-in-categorical and Polars rides shared Arrow path after rechunk() | VERIFIED | `arrow.rs`: dtype guard, `null_count()` check (CR-02 fix) materializes nulls as NaN for numeric / rejects for categorical; `polars.rs`: null-bearing column takes `Option<f64>` path preserving bitmap (not `cont_slice`); regression tests `arrow_numeric_null_becomes_nan`, `arrow_null_in_categorical_column_is_rejected`, `polars_numeric_null_becomes_nan`, `polars_null_in_categorical_column_is_rejected` all pass |
| 12 | Balanced and SqrtBalanced auto class weights match upstream at ≤1e-5 | VERIFIED | `weights_oracle_test.rs` calls `balanced_class_weights` and `sqrt_balanced_class_weights` against fixtures; `assert_abs_close` at 1e-5 passes for both |

**Score:** 8/8 requirements verified (12 observable sub-truths all VERIFIED)

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/cb-core/src/reduction.rs` | sum_f64 / sum_f32_in_f64 sequential reduction primitive (DATA-07) | VERIFIED | Contains both `pub fn sum_f64` and `pub fn sum_f32_in_f64`; no raw `.sum()`/`.fold` |
| `crates/cb-core/src/reduction_test.rs` | Naive-sequential-order property tests | VERIFIED | 6 dedicated tests; adversarial order-lock asserts `sum_f64(&[1e16, 1.0, -1e16]) == 0.0` |
| `scripts/check-no-raw-float-sum.sh` | D-08 CI-grep backstop | VERIFIED | Exists; excludes `reduction.rs` and `*_test.rs`; exits 0 on current tree |
| `crates/cb-oracle/fixtures/borders_quant/` | Border oracle fixtures for numeric_tiny + numeric_nan | VERIFIED | `numeric_tiny.borders.npy`, `numeric_tiny.borders_per_feature.npy`, `numeric_nan.borders.npy`, `numeric_nan.borders_per_feature.npy`, `config.json` all present |
| `crates/cb-oracle/fixtures/cat_hash/` | Cat-hash oracle: cat_hashes.npy + perfect_hash_bins.npy | VERIFIED | Both `.npy` files + `config.json` with A4/A5 documented |
| `crates/cb-oracle/fixtures/class_weights/` | Balanced/SqrtBalanced auto class-weight fixtures | VERIFIED | `balanced.npy`, `sqrt_balanced.npy`, `config.json` present |
| `crates/cb-oracle/fixtures/inputs/numeric_nan/` | NaN-containing input dataset | VERIFIED | `X.npy`, `y.npy`, `config.json` present |
| `crates/cb-data/src/pool.rs` | Pool struct with all CatBoost column kinds | VERIFIED | `pub struct Pool` with all 10 field types including pairs, baseline, embeddings |
| `crates/cb-data/src/ingest/mod.rs` | IngestSource trait seam | VERIFIED | `pub trait IngestSource` defined; OwnedColumns, ArrowColumns, PolarsColumns all implement it |
| `crates/cb-data/src/borders.rs` | GreedyLogSum priority-queue binarizer (DATA-03) | VERIFIED | `pub fn select_borders_greedy_logsum`; routes sums through `cb_core::sum_f64`; no raw `.sum()` |
| `crates/cb-data/src/borders_test.rs` | Isolated unit tests: penalty, duplicate collapse, -0.0 normalization, 2-value/1-border | VERIFIED | 7 dedicated unit tests covering all four required cases |
| `crates/cb-data/tests/borders_oracle_test.rs` | Per-feature border oracle comparison | VERIFIED | Calls `compare_stage(Stage::Borders, ...)` for numeric_tiny and numeric_nan; both pass |
| `crates/cb-data/src/nan_mode.rs` | NanMode enum + sentinel insertion + strict >border bin assignment | VERIFIED | `pub enum NanMode { Min, Max, Forbidden }` with `bin_of`, `insert_sentinel`, `nan_bin`; WR-06 debug_assert present |
| `crates/cb-data/src/quantized_pool.rs` | QuantizedPool SoA + ColumnBins width enum | VERIFIED | `pub enum ColumnBins { U8, U16, U32 }` + `pub struct QuantizedPool` immutable |
| `crates/cb-data/src/quantize.rs` | pool.quantize(&params) → CbResult<QuantizedPool> driver | VERIFIED | `Pool::quantize` wires borders + NanMode; CR-01 Max sentinel appended after border selection |
| `crates/cb-data/src/cat_hash.rs` | CityHash64 port + CalcCatFeatureHash + first-seen perfect-hash remap | VERIFIED | `city_hash_64`, `calc_cat_feature_hash`, `perfect_hash_bins`; `wrapping_*` arithmetic; no cityhash crate |
| `crates/cb-data/src/ingest/arrow.rs` | Arrow Float64Array → validated columns with CR-02 null-bitmap handling | VERIFIED | `null_count()` consulted; nulls → NaN (numeric) / rejected (categorical); `IngestSource` impl |
| `crates/cb-data/src/ingest/polars.rs` | Polars DataFrame → rechunk → shared Arrow path with null preservation | VERIFIED | Nullable columns take `Option<f64>` iteration preserving bitmap; CR-02 / WR-03 handled |
| `crates/cb-data/src/weights.rs` | Balanced/SqrtBalanced + per-object/per-class weights | VERIFIED | `balanced_class_weights`, `sqrt_balanced_class_weights`, `resolve_object_weights`; sums via `sum_f64`; 1e-8 floor; WR-05 `.max(0.0)` removed |
| `crates/cb-core/src/error.rs` | CbError with Dtype, LengthMismatch, NanInCategorical, Ingestion variants | VERIFIED | All four variants present; enum derives `Clone, PartialEq, Eq`; no `#[from]` for external types |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `crates/cb-core/src/lib.rs` | `crates/cb-core/src/reduction.rs` | `pub use reduction::` | WIRED | Line 23: `pub use reduction::{sum_f32_in_f64, sum_f64}` |
| `.github/workflows/ci.yml` | `scripts/check-no-raw-float-sum.sh` | CI step invocation | WIRED | Line 54: `run: bash scripts/check-no-raw-float-sum.sh` |
| `crates/cb-data/src/borders.rs` | `cb_core::reduction` | all sums via reduction primitive | WIRED | `use cb_core::sum_f64;`; only call is `sum_f64(weights)` in `total_object_weight` |
| `crates/cb-data/tests/borders_oracle_test.rs` | `crates/cb-oracle/fixtures/borders_quant` | load_f64_vec + compare_stage(Stage::Borders) | WIRED | `load_f64_vec(&fixture("borders_quant/{dataset}.borders.npy"))`; passes |
| `crates/cb-data/src/pool.rs` | `crates/cb-data/src/ingest/mod.rs` | Pool built through IngestSource | WIRED | `Pool::from_validated_columns` is `pub(crate)`; `IngestSource::into_pool` is the only public constructor path |
| `crates/cb-data/src/quantize.rs` | `crates/cb-data/src/borders.rs` | border selection during quantization | WIRED | `crate::select_borders_greedy_logsum(column, params.border_count, prepend_min_sentinel)` |
| `crates/cb-data/src/quantize.rs` | `crates/cb-data/src/nan_mode.rs` | sentinel insertion + bin assignment | WIRED | `crate::nan_mode::insert_sentinel`, `crate::bin_of`, `crate::nan_bin` all called |
| `crates/cb-data/tests/cat_hash_oracle_test.rs` | `crates/cb-oracle/fixtures/cat_hash` | load cat_hashes.npy + perfect_hash_bins.npy | WIRED | `load_f64_vec(&fixture("cat_hash/cat_hashes.npy"))` + `fixture("cat_hash/perfect_hash_bins.npy")`; passes |
| `crates/cb-data/src/ingest/polars.rs` | `crates/cb-data/src/ingest/arrow.rs` | rechunk → shared Arrow validation path | WIRED | `column_to_arrow` builds Arrow `ArrayRef`; `arrow_f64_column` called for all columns |
| `crates/cb-data/src/weights.rs` | `cb_core::reduction` | class-weight sums via reduction primitive | WIRED | `use cb_core::{sum_f64, ...}; sum_f64(bucket)` in `summary_class_weights` |
| `crates/cb-data/tests/weights_oracle_test.rs` | `crates/cb-oracle/fixtures/class_weights` | Balanced/SqrtBalanced ≤1e-5 comparison | WIRED | `load_f64_vec(&fixture("class_weights/balanced.npy"))`; `assert_abs_close` at 1e-5; passes |

---

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `borders_oracle_test.rs` | `expected_flat`, `per_feature` | `load_f64_vec` from committed `.npy` fixtures | Yes — binary numpy arrays loaded from disk | FLOWING |
| `quantize_oracle_test.rs` | `qp` (QuantizedPool) | `pool.quantize(&QuantizeParams::default())` on `X.npy` | Yes — real greedy binarization of input data | FLOWING |
| `cat_hash_oracle_test.rs` | `hashes_flat`, `bins_flat` | `load_f64_vec` from committed fixtures | Yes — bit-exact hash vectors from standalone C++ oracle | FLOWING |
| `weights_oracle_test.rs` | `expected`, `actual` | `load_f64_vec` + `balanced_class_weights(...)` | Yes — real class-count computation on 30+10 dataset | FLOWING |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All workspace tests | `cargo test --workspace` | 21 cb-core + 68 cb-data unit + 2 borders + 1 cat_hash + 1 quantize + 2 weights oracle tests = all pass, 0 failures | PASS |
| D-08 float sum gate | `bash scripts/check-no-raw-float-sum.sh` | `OK: no raw float summation in core library code` (exit 0) | PASS |
| Clippy cb-core lib | `cargo clippy -p cb-core --lib` | Clean (0 warnings, 0 errors) | PASS |
| Clippy cb-data lib | `cargo clippy -p cb-data --lib` | Clean (0 warnings, 0 errors) | PASS |
| CR-01 regression: NanMode::Max top bin | `cargo test -p cb-data --lib max_nan_mode_gives_nan_its_own_top_bin` | 1 passed | PASS |
| CR-02 regression: Arrow null → NaN | `cargo test -p cb-data --lib arrow_numeric_null_becomes_nan` | 1 passed | PASS |
| CR-02 regression: Arrow null in cat rejected | `cargo test -p cb-data --lib arrow_null_in_categorical_column_is_rejected` | 1 passed | PASS |
| CR-02 regression: Polars null → NaN | `cargo test -p cb-data --lib polars_numeric_null_becomes_nan` | 1 passed | PASS |
| CR-02 regression: Polars null in cat rejected | `cargo test -p cb-data --lib polars_null_in_categorical_column_is_rejected` | 1 passed | PASS |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| DATA-01 | 02-02 | Pool abstraction — float/cat/text/embedding columns, label, weights, group_id, subgroup_id, pairs, baseline | SATISFIED | `pool.rs` has all 10 field kinds; `OwnedColumns` builder with pair-bounds validation (WR-02) |
| DATA-02 | 02-03 | QuantizedPool — columnar SoA u8/u16 bin storage | SATISFIED | `ColumnBins { U8, U16, U32 }` + `QuantizedPool` immutable SoA; width selector enforces float cap at u16 |
| DATA-03 | 02-02 | GreedyLogSum border selection, oracle-validated including NaN/duplicate columns | SATISFIED | `borders.rs` GreedyLogSum; oracle passes on `numeric_tiny` + `numeric_nan`; isolated unit tests cover penalty, duplicate collapse, -0.0 normalization |
| DATA-04 | 02-03 | Missing-value handling — NanMode (Min/Max/Forbidden) | SATISFIED | `nan_mode.rs` with all three variants; CR-01 fix wires Max sentinel; oracle tests cover Min sentinel; regression test covers Max sentinel |
| DATA-05 | 02-04 | Categorical feature hashing | SATISFIED | `cat_hash.rs` ports vendored `city.cpp`; `calc_cat_feature_hash` bit-exact; `perfect_hash_bins` first-seen oracle-matched; u32::MAX overflow returns CbResult error |
| DATA-06 | 02-05 | Arrow/Polars ingestion with dtype/contiguity/null validation | SATISFIED | `arrow.rs` + `polars.rs` via shared funnel; CR-02 null-bitmap handling; typed `CbError` (Clone+Eq preserved); regression tests for all four null scenarios |
| DATA-07 | 02-01 | Single audited deterministic reduction utility | SATISFIED | `reduction.rs` sequential f64 fold; order-lock property test; D-08 CI gate |
| DATA-08 | 02-05 | Per-object / per-class weights and auto class weights | SATISFIED | `weights.rs` Balanced/SqrtBalanced + `resolve_object_weights`; sums via `sum_f64`; 1e-8 floor; WR-05 `.max(0.0)` removed; oracle ≤1e-5 |

All 8 DATA-0x requirements for Phase 2 are SATISFIED.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None | — | — | — | — |

No TBD/FIXME/XXX debt markers in any production file. No raw `.sum()`/`.fold(0.0` outside `reduction.rs`. No inline `#[cfg(test)]` brace blocks in any production module (all are semicolon-form module declarations in `lib.rs`/`ingest/mod.rs`). No `unwrap`/`expect`/`panic` in production paths (only in `*_test.rs` files).

---

### Human Verification Required

### 1. WR-01: STL Binary-Heap Tie-Break Port Correctness

**Test:** Construct a minimal feature column that, under `border_count = 2` (so exactly one split happens), produces two adjacent bins with provably equal `best_score` (e.g. a uniform 4-value column `[0.0, 1.0, 2.0, 3.0]` where splitting at value 1 or value 2 yields equal-count halves). Trace through `heap_push` and `heap_pop` manually — or run a C++ snippet using `std::priority_queue<TBinType>` with the same operator< — and confirm which bin the Rust implementation pops matches which bin the STL heap pops.

**Expected:** The Rust `heap_push`/`heap_pop`/`adjust_heap` sequence produces the same pop order as libstdc++'s `__push_heap`/`__pop_heap`/`__adjust_heap` when two bins have equal `best_score`. The final border set (which is sorted and deduped) should be identical to the upstream oracle for any tie-prone input.

**Why human:** The STL does not document a stable tie-break guarantee; the exact pop order depends on the binary-heap array layout after each push/pop cycle. Verifying parity requires either running a C++ reference implementation or an expert trace of the libstdc++ sift-down algorithm against the Rust port. The two oracle tests (`numeric_tiny_borders_match_oracle` and `numeric_nan_borders_match_oracle`) pass — which is strong indirect evidence — but they do not prove that a tie was encountered and resolved identically. The fix committed for WR-01 reproduces the libstdc++ algorithm intentionally; only a human can confirm the port is faithful on a tied-score input.

---

### Gaps Summary

No gaps blocking goal achievement. All 8 DATA requirements are implemented and oracle-locked. The single item requiring human attention is the WR-01 heap-algorithm tie-break trace — the phase goal itself is achieved (all oracle tests green, all requirement truths verified), but this logic-sensitive port benefits from a human spot-check before the phase is marked fully closed.

---

_Verified: 2026-06-13_
_Verifier: Claude (gsd-verifier)_
