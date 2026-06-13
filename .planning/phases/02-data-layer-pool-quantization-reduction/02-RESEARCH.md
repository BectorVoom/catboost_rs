# Phase 2: Data Layer — Pool, Quantization & Reduction - Research

**Researched:** 2026-06-13
**Domain:** Numeric parity port of CatBoost 1.2.10 data layer (Pool, quantization/borders, categorical hashing, weights, deterministic reduction) in Rust
**Confidence:** HIGH (all parity-critical findings read directly from vendored `catboost-master/` source with file:line citations)

## Summary

Phase 2 builds `cb-data` (the leaf data crate) and the single reduction primitive in `cb-core`, oracle-locked to CatBoost 1.2.10 within ≤1e-5. The parity-critical algorithms are NOT in the directories the CONTEXT.md guessed — the actual GreedyLogSum implementation lives in `catboost-master/library/cpp/grid_creator/binarization.{h,cpp}` (the `catboost/private/libs/quantization/grid_creator.*` files are a thin wrapper). All six critical parity questions are now answered concretely from source:

- **Reduction (D-09/DATA-07):** Upstream is **naive sequential `double` accumulation**, NOT Kahan/pairwise. The summation order that matters (border DP error terms, class-weight sums) all use a plain `double` running sum or `TVector<double>` blocked-by-thread sum. With `thread_count=1` (the pinned oracle config, D-12), every blocked sum collapses to a single sequential `double` accumulation. The reduction primitive must be a sequential `f64` fold, final-cast to `f32` only where upstream stores a `float`.
- **GreedyLogSum (DATA-03):** Default (unweighted) path uses a **priority-queue greedy bin-split** over object counts (`TFeatureBin`), penalty `-log(count + 1e-8)`, border = midpoint `0.5*a + 0.5*b` of adjacent unique values. NOT the DP `BestSplit` (that path is only hit for weighted/exact binarizers, which CatBoost's default `border_count`/`GreedyLogSum` config does not use).
- **NanMode (DATA-04):** Min prepends `float::lowest()` as border[0]; Max appends `float::max()`; NaN consumes one border from the budget. Bin assignment is strict `value > border`.
- **Categorical hashing (DATA-05):** `CityHash64(bytes) & 0xffffffff` → perfect-hash remap to first-seen-order bins, with sorted-map (`TMap`) tiebreak for the most-frequent-to-0 swap.
- **Bin width (DATA-02/D-10):** Float features are u8 (<256 borders) or u16 only — never u32. u32 is reached **only by high-cardinality categorical** perfect-hash bins (up to `Max<ui32>` uniques). D-10's u8/u16/u32 is correct, but the width rule differs by feature kind.
- **Weights (DATA-08):** Balanced = `max/w`, SqrtBalanced = `sqrt(max/w)`, floor 1e-8, summary weights accumulated in `double`.

**Primary recommendation:** Implement the unweighted GreedyLogSum priority-queue greedy binarizer (`TFeatureBin` path) exactly; implement the reduction primitive as a sequential `f64` fold; build ingestion on a shared Arrow-backed path (`arrow` 59.0.0) with Polars (`polars` 0.54.4) rechunked into the same path. Extract border oracle fixtures via the proven `model.get_borders()` Python API already wired in `gen_fixtures.py`.

## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Two distinct types — `Pool` (raw) and `QuantizedPool` (binned); quantization is explicit `pool.quantize(&params) -> QuantizedPool`. Mirrors upstream `TRawObjectsDataProvider` vs `TQuantizedObjectsDataProvider`.
- **D-02:** Owned now (Vec-backed), zero-copy seam later. Constructor built around an ingestion trait so a borrowed NumPy view plugs in at Phase 8 without reshaping `Pool`. No `Pool<'a>`/`Cow` yet.
- **D-03:** Immutable bins + caller-owned scratch. `QuantizedPool` built once, immutable; Phase 3 trainer reuses its OWN scratch over fixed bins.
- **D-04:** Rust-native ingestion + trait seam in Phase 2; PyO3 in Phase 8. No PyO3 dependency in Phase 2.
- **D-05:** Validated external paths = Arrow (`arrow-rs`) + Polars. Raw owned-Vec construction remains the trivial primitive. (Polars is Arrow-backed — may ride the Arrow path; planning to confirm — **CONFIRMED rideable, see Architecture Patterns**.)
- **D-06:** Validation at the ingestion boundary, typed `CbError` (thiserror) variants; same taxonomy Phase 8 maps to Python exceptions (PYAPI-05).
- **D-07:** Reduction primitive owned by `cb-core` (already home to `CbError` + `TFastRng64`).
- **D-08:** Enforced by CI-grep backstop + convention — mirrors Phase 1 anyhow-ban (D-14). A `scripts/check-*` grep fails CI on raw float `.sum()` / `.fold(0.0, +)` in library crates.
- **D-09:** Exact C++ accumulation behavior pinned by research — accumulator type (`f64`) AND summation order. **Do NOT assume naive sequential without confirming.** (RESEARCH FINDING: it IS naive sequential `double` under `thread_count=1` — see Reduction Primitive section, with citations.)
- **D-10:** Bin widths `u8`/`u16`/`u32` matching upstream. (RESEARCH FINDING: u8/u16 for float; u32 only for categorical perfect-hash.)
- **D-11:** Typed per-column enum `{ U8(&[u8]), U16(&[u16]), U32(&[u32]) }`; Phase 3 histogram kernel matches on width.
- **D-12:** Per-feature `Vec` SoA layout — each feature column its own contiguous buffer.

### Claude's Discretion
- Exact GreedyLogSum border algorithm, `<`/`<=` boundary assignment, NaN/duplicate-column handling — parity-dictated; researcher reads upstream and reproduces. **(ANSWERED below.)**
- `NanMode` (Min/Max/Forbidden) semantics and categorical hash function — parity-dictated. **(ANSWERED below.)**
- Auto class-weight formulas (Balanced/SqrtBalanced) and per-object/per-class weight handling. **(ANSWERED below.)**
- Intermediate-oracle fixture schema for borders/quantization — drawn from the frozen corpus (Phase 1 D-11). **(SCHEMA PROPOSED in Validation Architecture.)**
- Whether Polars ingestion rides the Arrow code path or is separate — planning to decide. **(RECOMMENDATION: rides shared Arrow path after `rechunk()`.)**
- Concrete crate versions for `arrow`/`polars` — latest stable per CLAUDE.md. **(arrow 59.0.0, polars 0.54.4 — verified.)**

### Deferred Ideas (OUT OF SCOPE)
- **PyO3 / NumPy zero-copy ingestion** — Phase 8 (PYAPI-04/06), reuses the Phase 2 ingestion trait seam (D-04).
- **CTR / ordered target-statistic columns** — `Pool` carries the columns CTRs use, but CTR computation is Phase 5 (ORD-03). Phase 2 only stores raw categorical data + hashing.
- **GPU bin storage / kernels** — `QuantizedPool` is CubeCL-free; GPU consumption of bins is Phase 7.

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DATA-01 | `Pool` — float/cat/text/embedding columns, label, weights, group_id, subgroup_id, pairs, baseline | Component map + upstream `TRawObjectsDataProvider` shape (Architecture Patterns) |
| DATA-02 | `QuantizedPool` — columnar SoA u8/u16 bins, buffers reused across rounds | Bin width rule (`CalcHistogramWidthForBorders`, utils.h:175), SoA layout (D-12) |
| DATA-03 | `GreedyLogSum` borders, oracle-validated (NaN/duplicate cols, `<`/`<=`) | Greedy priority-queue binarizer (binarization.cpp:1378-1520, 1676-1714) |
| DATA-04 | `NanMode` (Min/Max/Forbidden) | quantization.cpp:322-345, utils.h:51-66 |
| DATA-05 | Categorical feature hashing | `CalcCatFeatureHash` (cat_feature.cpp:6-8), perfect-hash (cat_feature_perfect_hash_helper.cpp:111-205) |
| DATA-06 | Arrow/Polars ingestion + dtype/contiguity validation; copy-in path | arrow 59.0.0 `as_primitive::<Float64Type>().values()`; polars rechunk→arrow |
| DATA-07 | Single audited deterministic reduction matching C++ `double` accumulator + order | Naive sequential `double` fold (binarization.cpp:803-815; calc_class_weights.cpp:36-54) |
| DATA-08 | Per-object/per-class weights + auto class weights (Balanced/SqrtBalanced) | calc_class_weights.cpp:11-27 |

## Project Constraints (from CLAUDE.md)

- **Memory efficiency is a first-class design constraint** — minimize allocations, prefer zero-copy. Drives the SoA layout (D-12), the typed-width enum (D-11, no widening), and the Arrow zero-copy `values()` ingestion path.
- **Error handling:** `thiserror` in `cb-data`/`cb-core` (library); `anyhow` strictly prohibited in `[dependencies]` of core library crates (Phase 1 D-14). `unwrap()` strictly prohibited in production.
- **Latest crate versions** — `arrow` 59.0.0, `polars` 0.54.4 (verified against crates.io 2026-06-13).
- **Workspace deps centralized** in root `Cargo.toml` `[workspace.dependencies]`; add `arrow`/`polars` there.
- **Source/test separation MANDATORY** — no inline `#[cfg(test)]` in production modules; dedicated `*_test.rs` files (e.g. `reduction_test.rs`, `borders_test.rs`). Test-lint exemption via in-code `#![cfg_attr(test, allow(...))]` (pattern already in `cb-core/src/lib.rs:8-16`).
- **Deny lints:** `clippy::unwrap_used`, `expect_used`, `panic`, `indexing_slicing` (workspace.lints, opt-in via `[lints] workspace = true`).
- **No CubeCL in this phase** — `cb-data` must not depend on `cb-backend`/CubeCL.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Raw column storage (`Pool`) | `cb-data` | — | Leaf data layer; owns raw float/cat/text/embedding + metadata |
| Quantization driver (`pool.quantize`) | `cb-data` | `cb-core` (reduction) | Border selection + bin assignment; routes sums through `cb-core` |
| Border selection (GreedyLogSum) | `cb-data` | `cb-core` (reduction) | Per-feature, parity-locked; uses object-count sums |
| Bin storage (`QuantizedPool` SoA) | `cb-data` | — | Immutable u8/u16/u32 columns; consumed by Phase 3 `cb-compute` |
| Categorical hashing + perfect-hash | `cb-data` | — | CityHash64 + first-seen remap; raw cat data only (CTR is Phase 5) |
| Reduction / summation primitive | `cb-core` | — | Process-wide invariant; every crate depends on `cb-core` (D-07) |
| Weights / auto class weights | `cb-data` | `cb-core` (reduction) | Per-object/class weights; summary sums in f64 |
| Arrow/Polars ingestion | `cb-data` | — | Ingestion trait impls; validation at the boundary (D-06) |
| Oracle border/quant fixtures | `cb-oracle` (gen) + `cb-data` tests | — | Python `get_borders()` extraction; Rust compares ≤1e-5 |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `arrow` | 59.0.0 | Arrow columnar ingestion + zero-copy `&[f64]` access | Canonical Rust Arrow impl (`apache/arrow-rs`, 1.1M weekly dl); the dtype/contiguity validation surface for DATA-06 |
| `polars` | 0.54.4 | DataFrame ingestion; `ChunkedArray` wraps Arrow | Arrow-backed; rechunk→single contiguous chunk lets it ride the Arrow path |
| `ndarray-npy` | 0.10.0 | Read `.npy` oracle fixtures (already in workspace) | Established Phase 1 fixture reader (D-09) |
| `thiserror` | 2.0.18 | Typed `CbError` variants at ingestion boundary | Phase 1 library error strategy (D-15) |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `arrow-array` (re-exported by `arrow`) | 59.0.0 | `Float64Array`, `as_primitive`, `AsArray` | Downcasting `ArrayRef` to typed arrays in ingestion |
| `arrow-schema` (re-exported by `arrow`) | 59.0.0 | `DataType` enum for dtype validation | Reject non-Float64/non-supported dtypes with a typed error |
| `serde` | (workspace) | Fixture config parsing (already wired) | Reuse `FixtureConfig` pattern in `cb-oracle` |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `arrow` umbrella crate | `arrow-array` + `arrow-schema` directly | Smaller dep tree, but D-05 names "arrow-rs"; umbrella re-exports both — use umbrella for clarity |
| Polars rides Arrow path | Separate Polars ingestion impl | Separate impl duplicates validation; rechunk→`to_arrow()` reuses one code path (recommended) |
| u32 bin width for float | u16 cap | Upstream `CalcHistogramWidthForBorders` never emits >16 bits for floats — u32 is categorical-only |

**Installation:**
```bash
# Add to root Cargo.toml [workspace.dependencies]:
#   arrow  = "59.0.0"
#   polars = { version = "0.54.4", default-features = false, features = ["dtype-full"] }
# Then in crates/cb-data/Cargo.toml:
#   arrow.workspace  = true
#   polars.workspace = true
```

**Version verification (2026-06-13):**
- `arrow = "59.0.0"` — verified via `cargo search arrow` and legitimacy seam (1.1M weekly dl, `github.com/apache/arrow-rs`, OK).
- `polars = "0.54.4"` — verified via `cargo search polars` and legitimacy seam (216K weekly dl, `github.com/pola-rs/polars`, OK).
- `ndarray-npy = "0.10.0"` — already in workspace; verified OK.

## Package Legitimacy Audit

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| arrow | crates.io | since 2018-03 | 1.11M/wk | github.com/apache/arrow-rs | OK | Approved |
| polars | crates.io | since 2020-06 | 216K/wk | github.com/pola-rs/polars | OK | Approved |
| ndarray-npy | crates.io | since 2018-04 | 195K/wk | github.com/jturner314/ndarray-npy | OK | Approved (already in workspace) |

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

*All three packages were verified via `gsd-tools query package-legitimacy check --ecosystem crates` AND cross-checked against `cargo search`. `arrow` and `polars` are the exact crates named in CONTEXT.md D-05.*

## Architecture Patterns

### System Architecture Diagram

```
  Raw sources                         cb-data (Pool)                  Quantization                QuantizedPool
 ┌──────────────┐   ingestion trait  ┌──────────────────┐   .quantize(&params)  ┌────────────┐   ┌──────────────────┐
 │ Vec<f64>     │ ─────────────────► │ float columns    │ ───────────────────►  │ per-feature│   │ SoA bin columns  │
 │ (Builder)    │                    │ cat columns      │                       │ border     │   │  ColumnBins enum │
 ├──────────────┤  validate dtype/   │ (hashed: u32)    │   border selection ►  │ selection  │ ► │  U8(Vec<u8>)     │
 │ Arrow array  │  contiguity/NaN ─► │ text/embedding   │   (GreedyLogSum)      └─────┬──────┘   │  U16(Vec<u16>)   │
 │ Float64Array │  → CbError         │ label, weights   │                             │          │  U32(Vec<u32>)   │
 ├──────────────┤                    │ group_id/subgrp  │   bin assignment ────────► (value>     │ + NanMode/feature│
 │ Polars       │  rechunk→to_arrow  │ pairs, baseline  │   (strict value>border)   border)     │   metadata       │
 │ DataFrame    │ ─────────────────► └──────────────────┘                                        └────────┬─────────┘
 └──────────────┘                            │                                                           │ immutable
                                             │ weights                                                   ▼
                                    ┌─────────────────┐                                          Phase 3 cb-compute
                                    │ auto class wts  │                                          (histogram kernels,
                                    │ Balanced/Sqrt   │                                           match on width)
                                    └────────┬────────┘
                                             │ all sums route through
                                             ▼
                                    ┌─────────────────────┐
                                    │ cb-core reduction   │  sequential f64 fold (only summation
                                    │ primitive (D-07)    │   primitive; CI-grep enforced, D-08)
                                    └─────────────────────┘
```

### Recommended Project Structure
```
crates/cb-core/src/
├── lib.rs              # add `pub use reduction::...`
├── reduction.rs        # the ONLY summation primitive (sequential f64 fold)
└── reduction_test.rs   # oracle-style associativity/order tests

crates/cb-data/src/
├── lib.rs              # module wiring + test-lint exemption
├── pool.rs             # Pool (raw columns + metadata), Builder
├── quantize.rs         # quantization driver: borders + bin assignment
├── borders.rs          # GreedyLogSum greedy binarizer (parity-locked)
├── nan_mode.rs         # NanMode enum + border-sentinel insertion
├── cat_hash.rs         # CityHash64 & 0xffffffff + perfect-hash remap
├── quantized_pool.rs   # QuantizedPool SoA + ColumnBins width enum (D-11)
├── weights.rs          # per-object/class weights + Balanced/SqrtBalanced
├── ingest/
│   ├── mod.rs          # IngestSource trait (D-04 seam)
│   ├── arrow.rs        # Arrow Float64Array → validated columns
│   ├── polars.rs       # DataFrame → rechunk → arrow path
│   └── owned.rs        # raw Vec<f64> primitive (Builder + fixtures)
└── *_test.rs           # one per module (source/test separation, mandatory)
```

### Pattern 1: GreedyLogSum greedy binarizer (the DEFAULT path — parity-critical)
**What:** For `EBorderSelectionType::GreedyLogSum` with no per-object weights (the CatBoost default), borders are selected by a **priority-queue greedy bin split** over object counts, NOT the DP `BestSplit`.
**When to use:** This is the path CatBoost's default training config hits. (The weighted/exact `BestSplit` DP at binarization.cpp:192-668 is only used by `MinEntropy`/`MaxLogSum`/weighted variants — out of Phase 2's default scope but document it for completeness.)
**Exact algorithm (binarization.cpp:1676-1714, 1319-1520):**
1. Sort unique non-NaN feature values ascending (`Sort(features.Values)`).
2. Start with one `TFeatureBin{0, n, values.begin()}` covering all values.
3. Maintain a `std::priority_queue<TFeatureBin>` ordered by `BestScore` (higher first).
4. While `splits.size() <= maxBordersCount && top.CanSplit()`: pop top, `Split()` it into left+right, push both back.
5. Each bin's best split point is found in `UpdateBestSplitProperties` (binarization.cpp:1409-1424): probe the **lower-bound and upper-bound of the midpoint value** (`lb`, `ub`), score each by the penalty delta, pick the better.
6. Penalty (binarization.cpp:178-181): `Penalty<MaxSumLog>(w) = -log(w + 1e-8)`; split score = `leftPartScore + rightPartScore - currBinScore` where each `partScore = -Penalty(count)` (binarization.cpp:1398-1407). **The weight `w` here is an integer object COUNT cast to double** (`static_cast<double>(splitPos - BinStart)`).
7. Border value for a bin start (binarization.cpp:1368-1370): `0.5f * values[BinStart-1] + 0.5f * values[BinStart]` (the first bin contributes no border).
8. Collect borders into a `THashSet<float>`, then `Sort` ascending; replace `-0.0f` with `0.0f` (binarization.cpp:897-900).

**Example (target Rust shape):**
```rust
// Source: catboost-master/library/cpp/grid_creator/binarization.cpp:1319-1520, 1676-1714
// Unweighted GreedyLogSum over SORTED UNIQUE values, penalty = -log(count + 1e-8).
fn penalty_maxsumlog(count: f64) -> f64 { -(count + 1e-8).ln() }
// split score of splitting a bin [start,end) at pos:
//   -penalty(pos-start) + -penalty(end-pos) - (-penalty(end-start))
// best split probes lower_bound & upper_bound of the midpoint *value* (handles ties),
// border = 0.5*values[start-1] + 0.5*values[start] (f32 arithmetic).
```
**Duplicate-column handling:** Duplicate values collapse during unique-grouping (binarization.cpp:817-829 for sorted; 839-866 for unsorted via a hash map then `Sort`). For the unweighted default with no DefaultValue, values are simply sorted (binarization.cpp:1704-1705) and `TFeatureBin` operates on the sorted array including duplicates — bin boundaries land between distinct adjacent values via the lower/upper-bound probing.

### Pattern 2: Bin assignment — strict `value > border` (the `<`/`<=` answer)
**What:** A value's bin index is the count of borders strictly less than the value.
**Exact (utils.h:28-49):**
```rust
// Source: catboost-master/catboost/private/libs/quantization/utils.h:34-40
// For <= 64 borders: index = sum over borders of (value > border)  [strict >]
// For  > 64 borders: index = lower_bound(borders, value) - begin   [first border >= value]
// Both are equivalent: value EQUAL to a border falls into the LOWER bin (border is exclusive upper edge of its bin).
fn bin_of(borders: &[f32], value: f32) -> u32 {
    if borders.len() <= 64 {
        borders.iter().filter(|&&b| value > b).count() as u32
    } else {
        borders.partition_point(|&b| b < value) as u32   // lower_bound
    }
}
```
**The `<`/`<=` semantics:** A border `b` is the **exclusive upper edge** of the bin below it. `value == b` → lands in the lower bin (because `value > b` is false). This is the canonical answer to DATA-03's "`<`/`<=` assignment semantics."

### Pattern 3: NanMode sentinel borders + NaN bin (DATA-04)
**What:** NaN handling is encoded by injecting a sentinel border and reserving a budget border.
**Exact (quantization.cpp:322-345, utils.h:51-66):**
- If the learn column has any NaN and `NanMode != Forbidden`: `nonNanValuesBorderCount = border_count - 1` (NaN consumes one border budget; quantization.cpp:325).
- After computing non-NaN borders:
  - **Min:** insert `f32::MIN` (`std::numeric_limits<float>::lowest()`) at `borders[0]` (quantization.cpp:342) → NaN values fall into bin 0 (smallest).
  - **Max:** push `f32::MAX` (`std::numeric_limits<float>::max()`) at the end (quantization.cpp:344) → NaN values fall into the top bin.
- At assignment time (utils.h:57-65): NaN → `nanMode == Max ? borders.size() : 0`. (For Forbidden, NaN is rejected unless allowed; defaults to bin 0.)
- No NaN in learn column → `nanMode = Forbidden` regardless of the option (quantization.cpp:326-328).

### Pattern 4: Categorical hashing + perfect-hash remap (DATA-05)
**What:** String/category → `ui32` hash → dense first-seen bin.
**Exact hash (cat_feature.cpp:6-8):**
```rust
// Source: catboost-master/catboost/libs/cat_feature/cat_feature.cpp:6-8
//   ui32 CalcCatFeatureHash(TStringBuf f) { return CityHash64(f) & 0xffffffff; }
// CityHash64 is the upstream util/digest/city.cpp variant (Google CityHash v1.0.x with Yandex tweaks).
// MUST port that exact variant — a generic cityhash crate may differ on tail handling.
```
**Perfect-hash remap (cat_feature_perfect_hash_helper.cpp:111-131):**
- Bins assigned in **first-seen iteration order**: `bin = perfectHashMap.GetSize()` for each new hash value (cat_feature_perfect_hash_helper.cpp:120).
- The map is a `TMap<ui32, TValueWithCount>` (a **sorted** red-black tree, cat_feature_perfect_hash.h:87) — relevant only for the most-frequent-to-0 tiebreak, which iterates the map in **ascending hash-value order** (cat_feature_perfect_hash_helper.cpp:156-171). For plain training without `quantizedDefaultBinFraction`/`mapMostFrequentValueTo0`, bins are simply first-seen.
- Up to `Max<ui32>()` unique values supported (cat_feature_perfect_hash_helper.cpp:53-54) → motivates u32 bin width for categoricals.

**Float-encoded cat values:** Integer category labels stored as f64 (as in the `numeric_categorical` corpus) must be converted to their string/int representation before hashing — confirm the exact stringification CatBoost uses for integer cat features when building the Pool oracle (see Open Questions).

### Pattern 5: Bin width selection (DATA-02 / D-10)
**What:** Per-feature storage width.
**Exact (utils.h:175-181):** `CalcHistogramWidthForBorders(bordersCount)` = **8 bits if `bordersCount < 256`, else 16 bits**. Asserts `bordersCount < 65536`. **Never 32 for float features.** `GetMaxBinCount() = 65535` (restrictions.h:10-12), default `border_count`/`discretization = 32` (binarization_options.cpp:20, .h:17 default 32 — though the CatBoost training default is 254; confirm via oracle).
- u32 is reached only by **categorical** perfect-hash bins (columns.h:417-425, the `ui32*` storage arm), where cardinality can exceed 65535.
- **Recommendation for `ColumnBins` enum (D-11):** float feature → `U8`/`U16` only; categorical feature → `U8`/`U16`/`U32` by uniq-count. The width rule is feature-kind-dependent.

### Anti-Patterns to Avoid
- **Using the DP `BestSplit` for the default GreedyLogSum path:** That DP (binarization.cpp:192-668, `E_RLM2`) is for weighted/exact binarizers. The default unweighted GreedyLogSum uses the priority-queue greedy (`TFeatureBin`). Implementing the DP would diverge.
- **Computing borders in `f64`:** Border arithmetic is `float` (binarization.cpp:1368-1370 uses `0.5f *`). Compute candidate borders in `f32` to match bit-for-bit; the `double` is only the DP/penalty *error accumulator*.
- **Inclusive border (`value >= border` → upper bin):** Upstream is strict `>`. Getting this backwards shifts every boundary value into the wrong bin.
- **Generic `cityhash` crate without verifying the variant:** Yandex's `util/digest/city.cpp` may differ from a third-party crate on seed constants / tail mixing. Port the exact source or oracle-validate the hash on known inputs.
- **Parallel/blocked summation in the reduction primitive:** Even though upstream uses `TVector<double>` per-thread blocks, the oracle is pinned to `thread_count=1` → one block → sequential. A parallel Rust reduction would reorder additions and break ≤1e-5 on adversarial inputs.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Arrow dtype/contiguity validation | Custom byte-buffer parser | `arrow` `array.data_type()` + `as_primitive::<Float64Type>().values()` | Arrow already validates layout; `values()` is a zero-copy `&[f64]` |
| Polars → contiguous f64 | Manual chunk-walking | `ChunkedArray::rechunk()` then `cont_slice()` / `to_arrow()` | Polars manages multi-chunk → single contiguous; rechunk is the blessed path |
| `.npy` fixture reading | Custom npy parser | `ndarray-npy` `read_npy` (already wired) | Phase 1 D-09 standard; errors (not panics) on dtype mismatch |
| Sorted-set border dedup | Custom dedup | A `BTreeSet<OrderedFloat>` or sort+dedup mirroring `THashSet`→`Sort` | Must match upstream's `THashSet`→`Sort` final ordering exactly |
| CityHash64 | Reimplement from a blog post | Port `util/digest/city.cpp` exactly OR a crate validated against upstream vectors | Tail-mixing differences silently break cat-feature parity |

**Key insight:** The hard part of this phase is not infrastructure — it is reproducing upstream's *exact float arithmetic and iteration order*. Reach for libraries for I/O (Arrow/Polars/npy), but the border/bin/hash/weight math must be a faithful transcription of the cited C++, not a "clean" reimplementation.

## Runtime State Inventory

> Greenfield additive phase (fills the `cb-data` stub; adds a `cb-core` module). No rename/migration. This section is N/A for state migration but the following build-artifact note applies:

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — `cb-data` is a stub doc-comment only; no persisted state exists | none |
| Live service config | None | none |
| OS-registered state | None | none |
| Secrets/env vars | None | none |
| Build artifacts | Adding `arrow`/`polars` to the workspace will regenerate `Cargo.lock`; commit it (Phase 1 supply-chain convention T-01-SC) | `cargo build` + commit updated `Cargo.lock` |

**Nothing found in the first four categories** — verified by inspecting `crates/cb-data/src/lib.rs` (stub) and the absence of any datastore in the workspace.

## Common Pitfalls

### Pitfall 1: Assuming naive sequential summation without confirming (D-09 explicit warning)
**What goes wrong:** Planner picks Kahan or pairwise summation "for accuracy," diverging from upstream's plain `double` accumulation.
**Why it happens:** "Deterministic reduction" sounds like it implies a sophisticated algorithm.
**How to avoid:** Implement a **plain sequential `f64` running sum**. Upstream evidence: `totalWeight += weight` over a sequential loop (binarization.cpp:803-815); `summaryClassWeightsPerBlock[blockId][...] += itemWeights[i]` with blocks=1 under thread_count=1 (calc_class_weights.cpp:36-44); cumulative-weight prefix sums `sweights[i] += sweights[i-1]` (binarization.cpp:220-223). No Kahan/pairwise anywhere in the border/weight paths.
**Warning signs:** Reduction test passes on tiny inputs but the planner adds a "compensated sum" — that is the bug.

### Pitfall 2: f32 vs f64 boundary errors
**What goes wrong:** Borders computed in f64 (or weights kept in f32 during accumulation) shift the 1e-5 baseline.
**Why it happens:** Rust defaults nudge toward one type; upstream mixes deliberately.
**How to avoid:** Borders/values are `f32` (`0.5f * a + 0.5f * b`); penalty/DP error accumulators and weight sums are `f64`, cast to `f32` only at storage. Class weights: sum in `double`, final result is `float` (calc_class_weights.cpp:36, 83).
**Warning signs:** Off-by-~1e-6 border mismatches that grow with feature cardinality.

### Pitfall 3: Wrong border-set ordering / get_borders() sentinel
**What goes wrong:** Rust border order differs from `model.get_borders()`, or the NaN sentinel (`f32::MIN`/`MAX`) is present in one but not the other.
**Why it happens:** Upstream stores borders in a `THashSet` then `Sort`s; the NaN sentinel is injected into the *stored* borders (quantization.cpp:341-345). `get_borders()` may or may not strip the sentinel.
**How to avoid:** When generating the border oracle fixture, **verify empirically** whether `get_borders()` includes the `f32::MIN`/`MAX` sentinel for NaN features, and document it in `config.json`. For the NaN-free corpus datasets (numeric_tiny etc.) this does not arise; add a NaN-containing dataset to exercise it (see Validation Architecture / Wave 0).
**Warning signs:** Border count off by exactly 1 on features with NaN.

### Pitfall 4: CityHash64 variant mismatch
**What goes wrong:** A generic cityhash crate produces different `ui32` hashes → different perfect-hash bin assignment → categorical parity fails.
**Why it happens:** CityHash has multiple revisions; Yandex's `util/digest/city.cpp` is a specific one.
**How to avoid:** Port `catboost-master/util/digest/city.cpp` (and `city.h`) exactly, OR oracle-validate the Rust hash against known `(string -> ui32)` vectors extracted from upstream before relying on it. Add a `cat_hash_test.rs` with vectors.
**Warning signs:** Cat-feature bin indices match in count but not in assignment.

### Pitfall 5: Integer cat-feature stringification
**What goes wrong:** Float-encoded integer categories (corpus stores cat as f64) hash differently than upstream because the string representation differs (e.g. `"3"` vs `"3.0"`).
**Why it happens:** CatBoost hashes the *string form* of a categorical; how an integer-valued float column is stringified matters.
**How to avoid:** Determine upstream's exact stringification for integer cat features (see Open Questions) and match it before hashing. Prefer feeding cat features as explicit integer/string columns in the oracle generator to remove ambiguity.

## Code Examples

### Reduction primitive (cb-core)
```rust
// Source: catboost-master/library/cpp/grid_creator/binarization.cpp:803-815 (totalWeight += weight),
//         calc_class_weights.cpp:36-54 (double per-block sum, blocks=1 under thread_count=1).
// The ONLY summation primitive (D-07). Sequential f64 fold. CI-grep bans raw float .sum()/.fold (D-08).
pub fn sum_f64(values: &[f64]) -> f64 {
    let mut acc = 0.0_f64;
    for &v in values {
        acc += v;          // plain sequential add — matches upstream order under thread_count=1
    }
    acc
}
// Variant for f32 inputs accumulated in f64 then returned as the upstream-stored type:
pub fn sum_f32_in_f64(values: &[f32]) -> f64 {
    let mut acc = 0.0_f64;
    for &v in values { acc += v as f64; }
    acc
}
```

### Arrow ingestion with dtype/contiguity validation (cb-data)
```rust
// Source: Context7 /apache/arrow-rs — as_primitive::<Float64Type>().values() is zero-copy &[f64].
use arrow_array::{Array, Float64Array, ArrayRef};
use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_schema::DataType;

fn ingest_float_column(col: &ArrayRef) -> Result<&[f64], CbError> {
    if col.data_type() != &DataType::Float64 {
        return Err(CbError::Dtype { expected: "Float64", got: format!("{:?}", col.data_type()) });
    }
    // For a NUMERIC feature, NaN is allowed but a NULL is a missing value distinct from NaN —
    // decide policy at the boundary (D-06). For a CATEGORICAL feature, reject NaN/null.
    let arr = col.as_primitive::<Float64Type>();
    Ok(arr.values())   // contiguous, zero-copy
}
```

### Polars riding the Arrow path (cb-data)
```rust
// Polars ChunkedArray wraps Arrow; rechunk() consolidates to one contiguous chunk,
// then reuse the same Arrow validation path. (Source: docs.rs/polars ChunkedArray; web-confirmed.)
// Pseudocode shape:
//   let s: &Series = df.column("f0")?;
//   let ca = s.f64()?;                 // ChunkedArray<Float64Type>
//   let ca = ca.rechunk();             // single contiguous chunk (may copy)
//   let slice: &[f64] = ca.cont_slice()?;  // zero-copy once contiguous
// => feed `slice` into the SAME owned-Vec / validation primitive as Arrow.
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `arrow2`/`polars-arrow` fork | `arrow` (arrow-rs) is canonical again; Polars uses its own `polars-arrow` internally but exposes `to_arrow()` interop | 2023-2024 | Use `arrow` 59.x for the validated external path; bridge Polars via `to_arrow()`/`rechunk` |
| `THashSet<float>` borders (unordered) | Upstream still uses `THashSet`→`Sort`; Rust should sort+dedup deterministically | stable | Border *set* identity matters, not insertion order — sort ascending to compare |

**Deprecated/outdated:** none relevant to this phase. The vendored 1.2.10 source is the frozen oracle (Phase 1 D-07) — do not consult newer CatBoost releases for algorithm details.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Polars `cont_slice()`/`to_arrow()` after `rechunk()` gives zero-copy `&[f64]` on polars 0.54.4 | Standard Stack / Pattern Polars | LOW — worst case a copy; correctness unaffected |
| A2 | CatBoost training default `border_count` is 254 (not the 32 default in `binarization_options.h`) | Pattern 5 | MEDIUM — affects which features reach u16; confirm via oracle `config.json` |
| A3 | `model.get_borders()` strips the NaN sentinel border (`f32::MIN`/`MAX`) | Pitfall 3 / Validation | MEDIUM — border-count off-by-1 on NaN features; resolve empirically in Wave 0 |
| A4 | Integer cat features in the corpus stringify as plain integers (`"3"`) before hashing | Pitfall 5 | MEDIUM — wrong stringification breaks cat hash parity |
| A5 | `arrow` umbrella re-exports `arrow-array`/`arrow-schema` at matching 59.0.0 | Code Examples | LOW — adjust imports if split |

**These assumptions must be confirmed during planning/Wave 0** — primarily by inspecting upstream defaults and running the oracle generator against a NaN-containing and explicit-categorical dataset.

## Open Questions

1. **Does `get_borders()` include the NaN sentinel border?**
   - What we know: stored borders inject `f32::MIN`/`MAX` for Min/Max NanMode (quantization.cpp:341-345).
   - What's unclear: whether the Python `get_borders()` API surfaces or strips it.
   - Recommendation: Wave 0 — generate a NaN-containing dataset, inspect `get_borders()` output, record presence/absence in fixture `config.json`. Match in Rust accordingly.

2. **Default `border_count` for the training oracle (32 vs 254)?**
   - What we know: `binarization_options.h:17` default param is 32; CatBoost docs/training default is commonly 254.
   - What's unclear: which value the Python `CatBoostRegressor` default uses in 1.2.10.
   - Recommendation: read it back from the trained model's `config.json`/params in `gen_fixtures.py` and pin it explicitly in the fixture params (already recorded there).

3. **Exact stringification of integer/float categorical values before `CalcCatFeatureHash`?**
   - What we know: hash operates on `TStringBuf` (bytes); cat features are conceptually strings.
   - What's unclear: how an integer-coded cat column is converted to bytes upstream.
   - Recommendation: feed explicit string/int cat columns in the oracle generator and capture the hashed `ui32` values via a small upstream probe, or extract perfect-hash bin assignments from the model to validate end-to-end.

4. **Is CityHash64 best ported or crate-sourced?**
   - What we know: `util/digest/city.cpp` is a specific variant.
   - Recommendation: port the vendored `city.cpp`/`city.h` directly into `cb-core`/`cb-data` (small, self-contained) and validate against upstream `(input -> ui32)` vectors. Do not depend on an unvalidated `cityhash` crate.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust toolchain | all | ✓ | cargo/rustc 1.96.0 | — |
| `arrow` crate | DATA-06 Arrow ingestion | ✓ (crates.io) | 59.0.0 | — |
| `polars` crate | DATA-06 Polars ingestion | ✓ (crates.io) | 0.54.4 | — |
| `ndarray-npy` | fixture reading (already wired) | ✓ | 0.10.0 | — |
| Python `catboost==1.2.10` + numpy | oracle fixture GENERATION (build-time, not CI) | ✓ (venv at `crates/cb-oracle/generator/.venv`) | 1.2.10 | regenerate fixtures on dev machine |
| Vendored `catboost-master/` source | parity reference | ✓ | 1.2.10 | — |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** none — all required tooling is present.

## Validation Architecture

> Nyquist validation is enabled (`workflow.nyquist_validation: true`). Oracle determinism inherits Phase 1: pinned seed, `thread_count=1`, frozen committed fixtures, absolute error ≤1e-5 (D-12).

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` + `approx` (`abs_diff_eq!`) + `cb-oracle` comparator (`compare_stage`, `assert_abs_close`, `Stage`) |
| Config file | none (cargo); fixtures under `crates/cb-oracle/fixtures/` |
| Quick run command | `cargo test -p cb-core -p cb-data --lib` |
| Full suite command | `cargo test --workspace` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DATA-07 | Sequential f64 fold matches order-sensitive expected sum | unit | `cargo test -p cb-core reduction` | ❌ Wave 0 (`cb-core/src/reduction_test.rs`) |
| DATA-03 | GreedyLogSum borders == `get_borders()` per feature ≤1e-5 | oracle | `cargo test -p cb-data borders_oracle` | ❌ Wave 0 (fixture + `borders_test.rs`) |
| DATA-04 | NanMode Min/Max sentinel + NaN bin placement | unit+oracle | `cargo test -p cb-data nan_mode` | ❌ Wave 0 (NaN dataset + test) |
| DATA-05 | CityHash64&0xffffffff + first-seen perfect-hash bins | unit+oracle | `cargo test -p cb-data cat_hash` | ❌ Wave 0 (hash vectors + `cat_hash_test.rs`) |
| DATA-02 | u8/u16 (float) / u32 (cat) width selection; SoA round-trip | unit | `cargo test -p cb-data quantized_pool` | ❌ Wave 0 |
| DATA-01 | Pool holds all column kinds + metadata | unit | `cargo test -p cb-data pool` | ❌ Wave 0 |
| DATA-06 | Arrow + Polars ingestion produce identical owned columns to Vec path; dtype/contiguity errors | unit | `cargo test -p cb-data ingest` | ❌ Wave 0 |
| DATA-08 | Balanced/SqrtBalanced class weights ≤1e-5 | unit+oracle | `cargo test -p cb-data weights` | ❌ Wave 0 |

### Oracle Fixture Schema (intermediate-stage, drawn from frozen corpus — Claude's Discretion answered)
Extend the proven `gen_fixtures.py` pattern (which already extracts `get_borders()`). New `cb-oracle/fixtures/<scenario>/` outputs referencing the frozen `inputs/` corpus:

- **`borders_quant/` (uses `numeric_tiny` + a NEW `numeric_nan` dataset):**
  - `borders.npy` (flat f64, ascending per feature) + `borders_per_feature.npy` (counts) — already the established layout.
  - `config.json` records: `border_count`, `nan_mode`, whether the sentinel is included (A3), `border_selection_type=GreedyLogSum`.
- **`cat_hash/` (uses `numeric_categorical` with EXPLICIT cat columns):**
  - `cat_hashes.npy` (f64-encoded `ui32` per category string) — extracted via an upstream probe.
  - `perfect_hash_bins.npy` (bin index per object) — to validate first-seen remap end-to-end.
- **`class_weights/` (binary-label dataset):**
  - `balanced.npy`, `sqrt_balanced.npy` (per-class weights) extracted from CatBoost's auto class-weight output.

### Intermediate-stage comparison strategy (per critical algorithm)
1. **Reduction (DATA-07):** No oracle fixture needed — validate by *property*: construct an array where naive-sequential ≠ pairwise (e.g. `[1e16, 1.0, -1e16]`) and assert the primitive returns the naive-sequential result. This locks the order, not just the value.
2. **Borders (DATA-03):** Per-feature set comparison: sort Rust borders ascending, compare element-wise to `get_borders()[fi]` at ≤1e-5 via `compare_stage(Stage::Borders, ...)`. Test the no-NaN corpus first (clean), then the NaN dataset (sentinel handling).
3. **NanMode (DATA-04):** Two layers — (a) border-set includes/excludes sentinel correctly; (b) a hand-constructed value vector with NaNs quantizes to the expected bins (NaN→0 for Min/Forbidden, NaN→top for Max).
4. **Cat hashing (DATA-05):** (a) `CityHash64&0xffffffff` matches extracted `(string→ui32)` vectors exactly (bit-exact, not ≤1e-5 — it's an integer); (b) perfect-hash bin assignment matches first-seen order on the corpus.
5. **Bin width (DATA-02):** Unit-test `CalcHistogramWidthForBorders` equivalent: <256→U8, <65536→U16; categorical uniq-count→U8/U16/U32. SoA build→read round-trip is lossless.
6. **Weights (DATA-08):** Compare Balanced/SqrtBalanced per-class weights to upstream auto-class-weight output ≤1e-5; verify the 1e-8 floor branch on a degenerate (empty) class.

### Sampling Rate
- **Per task commit:** `cargo test -p cb-core -p cb-data --lib` (quick).
- **Per wave merge:** `cargo test --workspace` (full, includes `cb-oracle` fixture comparisons).
- **Phase gate:** Full suite green before `/gsd-verify-work`; all DATA-0x oracle comparisons ≤1e-5.

### Wave 0 Gaps
- [ ] `crates/cb-core/src/reduction.rs` + `reduction_test.rs` — covers DATA-07
- [ ] `crates/cb-data/src/borders.rs` + `borders_test.rs` — covers DATA-03
- [ ] `crates/cb-data/src/nan_mode.rs` + `nan_mode_test.rs` — covers DATA-04
- [ ] `crates/cb-data/src/cat_hash.rs` (+ ported CityHash64) + `cat_hash_test.rs` — covers DATA-05
- [ ] `crates/cb-data/src/quantized_pool.rs` + test — covers DATA-02
- [ ] `crates/cb-data/src/pool.rs` + test — covers DATA-01
- [ ] `crates/cb-data/src/ingest/{arrow,polars,owned}.rs` + tests — covers DATA-06
- [ ] `crates/cb-data/src/weights.rs` + test — covers DATA-08
- [ ] New oracle datasets: `numeric_nan` (NaN float column), explicit-categorical variant; extend `gen_inputs.py` + `gen_fixtures.py`
- [ ] `scripts/check-no-raw-float-sum.sh` (D-08 CI-grep backstop) mirroring `check-no-anyhow.sh`
- [ ] Upstream probe for `(cat string → ui32)` vectors and integer-cat stringification (resolves A4/Open Q3)

## Security Domain

> `security_enforcement: true`, ASVS level 1. This is a numeric data-layer crate with no auth/session/network surface; the relevant ASVS category is input validation at the ingestion boundary.

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes | Typed `CbError` validation in ingestion trait impls (D-06): dtype, contiguity, NaN-in-categorical, length-mismatch, integer-overflow on uniq cat count (`MAX_UNIQ_CAT_VALUES`, cat_feature_perfect_hash_helper.cpp:53) |
| V6 Cryptography | no | CityHash64 is a non-cryptographic hash (correct for bucketing; never used for security) |

### Known Threat Patterns for a Rust numeric data layer
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Untrusted Arrow/Polars buffer with mismatched dtype/length | Tampering | Validate `data_type()` + length at the boundary; return `CbError`, never index blindly (deny `indexing_slicing` already enforced) |
| Integer overflow on bin index / uniq cat count | DoS / Tampering | Bound cat uniq count to `u32::MAX` (mirror `MAX_UNIQ_CAT_VALUES` check); width-select to u32 only when needed |
| Panic on malformed input (would crash a Phase 8 Python caller) | DoS | No `unwrap`/`expect`/`panic` (workspace deny-lints); all fallible paths return `CbResult` |
| NaN smuggled into a categorical column | Tampering | Reject NaN in cat columns at ingestion (upstream `ShouldBeSkipped`/forbidden semantics) |

## Sources

### Primary (HIGH confidence — read directly from vendored 1.2.10 source)
- `catboost-master/library/cpp/grid_creator/binarization.{h,cpp}` — GreedyLogSum greedy binarizer (1319-1520, 1676-1714), DP BestSplit (192-668), penalties (174-189), GroupAndSort (791-883), SetQuantization (887-988), border-set sort/-0 fix (897-904).
- `catboost-master/catboost/private/libs/quantization/utils.h` — bin assignment `value > border` (28-66), NanMode-at-assignment (51-66), `CalcHistogramWidthForBorders` (175-181), `GetBinCount` (171-173).
- `catboost-master/catboost/libs/data/quantization.cpp` — `CalcQuantizationAndNanMode` (235-346): NaN budget, sentinel border insertion (341-345).
- `catboost-master/catboost/libs/cat_feature/cat_feature.{h,cpp}` — `CalcCatFeatureHash = CityHash64 & 0xffffffff` (cpp:6-8), hash→float conversions (h:14-20).
- `catboost-master/catboost/libs/data/cat_feature_perfect_hash{,_helper}.{h,cpp}` — first-seen bin remap (helper.cpp:111-131), most-frequent-to-0 (156-205), `TMap` storage (.h:85-99), `MAX_UNIQ_CAT_VALUES` (helper.cpp:53).
- `catboost-master/catboost/private/libs/target/calc_class_weights.cpp` — Balanced/SqrtBalanced formulas (11-27), double per-block sum (29-55), 1e-8 floor (9).
- `catboost-master/catboost/private/libs/options/restrictions.h` — `GetMaxBinCount()=65535` (10-12).
- `catboost-master/catboost/private/libs/options/enums.h` — `ENanMode {Min,Max,Forbidden}` (107-111).
- `catboost-master/catboost/libs/data/columns.h` — bitsPerKey 8/16/32 storage arms (399-426).
- `crates/cb-oracle/generator/gen_fixtures.py` — proven `get_borders()` extraction pattern (60-95); `gen_inputs.py` frozen corpus.

### Secondary (MEDIUM confidence)
- Context7 `/apache/arrow-rs` — `as_primitive::<Float64Type>().values()` zero-copy, downcast/`AsArray`, dtype access.
- `cargo search` (2026-06-13) — arrow 59.0.0, polars 0.54.4, ndarray-npy 0.10.0.
- `gsd-tools query package-legitimacy check --ecosystem crates` — arrow/polars/ndarray-npy all OK.

### Tertiary (LOW confidence — verify in Wave 0)
- WebSearch (docs.pola.rs / docs.rs) — Polars `ChunkedArray` Arrow-backed, `rechunk()`/`cont_slice()` contiguous zero-copy semantics.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — versions verified against crates.io + legitimacy seam; arrow API confirmed via Context7.
- Architecture (parity algorithms): HIGH — every critical algorithm read directly from vendored 1.2.10 source with file:line citations.
- Pitfalls: HIGH — derived from the cited source (f32/f64 boundaries, strict-`>`, naive sum, sentinel borders).
- Polars-rides-Arrow / sentinel-in-get_borders / cat stringification: MEDIUM-LOW — flagged in Assumptions Log A1/A3/A4 for Wave 0 confirmation.

**Research date:** 2026-06-13
**Valid until:** 2026-07-13 (30 days; vendored oracle source is frozen at 1.2.10, so parity findings do not expire — only crate versions drift)
