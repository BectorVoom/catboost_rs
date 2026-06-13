# Phase 2: Data Layer — Pool, Quantization & Reduction - Pattern Map

**Mapped:** 2026-06-13
**Files analyzed:** 24 (16 Rust source/test + 4 manifests + 2 scripts + 2 generator)
**Analogs found:** 24 / 24 (every new file has a strong in-repo analog from Phase 1)

> This phase is **additive and greenfield within an established workspace**. Phase 1
> already laid down every convention Phase 2 must follow: the module + `pub use` +
> `#[cfg(test)] mod *_test;` wiring, the `thiserror` `CbError` taxonomy, the
> source/test-separation rule, the centralized workspace deps, the oracle
> fixture/comparator pattern, and the CI-grep enforcement idiom. **The closest
> analog for almost every Phase 2 file is a concrete Phase 1 file in this same
> repo** — copy its shape exactly. There is no need to invent structure.

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/cb-core/src/reduction.rs` | utility (primitive) | transform/reduce | `crates/cb-core/src/rng.rs` | exact (same crate, parity-port utility) |
| `crates/cb-core/src/reduction_test.rs` | test | property/transform | `crates/cb-core/src/rng_test.rs` | exact |
| `crates/cb-core/src/lib.rs` (MODIFY) | config (module wiring) | — | itself (current `cb-core/src/lib.rs`) | exact |
| `crates/cb-core/src/error.rs` (MODIFY) | model (error enum) | — | itself + `cb-oracle/src/error.rs` | exact |
| `crates/cb-data/src/lib.rs` (MODIFY) | config (module wiring) | — | `crates/cb-core/src/lib.rs` | exact |
| `crates/cb-data/src/pool.rs` | model (struct + builder) | CRUD/storage | `crates/cb-core/src/rng.rs` (builder-ish struct) + features.rs (vendored, ref only) | role-match |
| `crates/cb-data/src/pool_test.rs` | test | CRUD | `crates/cb-core/src/rng_test.rs` | exact |
| `crates/cb-data/src/borders.rs` | utility (algorithm) | transform | `crates/cb-core/src/rng.rs` (parity-port algo) | exact (parity-port style) |
| `crates/cb-data/src/borders_test.rs` | test | oracle | `crates/cb-oracle/tests/per_stage_oracle_test.rs` + `rng_test.rs` | exact |
| `crates/cb-data/src/nan_mode.rs` | model (enum + logic) | transform | `crates/cb-oracle/src/compare.rs` (`Stage` enum) | role-match |
| `crates/cb-data/src/nan_mode_test.rs` | test | unit+oracle | `crates/cb-core/src/rng_test.rs` | exact |
| `crates/cb-data/src/cat_hash.rs` | utility (algorithm) | transform | `crates/cb-core/src/rng.rs` (bit-exact C++ port) | exact (parity-port style) |
| `crates/cb-data/src/cat_hash_test.rs` | test | unit (bit-exact vectors) | `crates/cb-core/src/rng_test.rs` (bitstream vectors) | exact |
| `crates/cb-data/src/quantize.rs` | service (driver) | transform | `crates/cb-core/src/rng.rs` (orchestrates sub-utils) | role-match |
| `crates/cb-data/src/quantize_test.rs` | test | unit+oracle | `crates/cb-oracle/tests/per_stage_oracle_test.rs` | exact |
| `crates/cb-data/src/quantized_pool.rs` | model (SoA struct + width enum) | storage | `crates/cb-oracle/src/compare.rs` (`Stage` enum) + rng.rs | role-match |
| `crates/cb-data/src/quantized_pool_test.rs` | test | unit (round-trip) | `crates/cb-core/src/rng_test.rs` | exact |
| `crates/cb-data/src/weights.rs` | utility (formula) | transform/reduce | `crates/cb-core/src/rng.rs` | exact (parity-port style) |
| `crates/cb-data/src/weights_test.rs` | test | unit+oracle | `crates/cb-oracle/tests/per_stage_oracle_test.rs` | exact |
| `crates/cb-data/src/ingest/mod.rs` | route (trait seam) | — | `crates/cb-core/src/lib.rs` (module wiring) | role-match |
| `crates/cb-data/src/ingest/arrow.rs` | service (ingestion impl) | request-response (validate) | `crates/cb-oracle/src/fixture.rs` (load + validate → typed err) | role-match |
| `crates/cb-data/src/ingest/polars.rs` | service (ingestion impl) | request-response (validate) | `crates/cb-oracle/src/fixture.rs` | role-match |
| `crates/cb-data/src/ingest/owned.rs` | service (primitive ctor) | CRUD | `crates/cb-oracle/src/fixture.rs::load_f64_vec` | role-match |
| `crates/cb-data/src/ingest/*_test.rs` | test | unit | `crates/cb-oracle/src/fixture_test.rs` | exact |
| `crates/cb-data/Cargo.toml` (MODIFY) | config | — | `crates/cb-oracle/Cargo.toml` (workspace dep wiring) | exact |
| `Cargo.toml` (MODIFY) | config | — | itself (`[workspace.dependencies]`) | exact |
| `crates/cb-oracle/generator/gen_inputs.py` (MODIFY) | config (fixture gen) | batch | itself (`gen_numeric_tiny`) | exact |
| `crates/cb-oracle/generator/gen_fixtures.py` (MODIFY) | config (fixture gen) | batch | itself (borders extraction block) | exact |
| `scripts/check-no-raw-float-sum.sh` | config (CI gate) | — | `scripts/check-no-anyhow.sh` | exact |

---

## Shared Patterns

These four cross-cutting patterns apply to **every** new Rust file in this phase.
They are the non-negotiable Phase 1 conventions. The planner must reference them in
every plan's action section.

### Shared Pattern A — Module wiring + test-lint exemption (applies to BOTH `lib.rs` files)

**Source:** `crates/cb-core/src/lib.rs:1-28` and `crates/cb-oracle/src/lib.rs:1-31`

Every crate root: (1) a crate-level `#![cfg_attr(test, allow(...))]` block exempting
the four denied restriction lints in test code, (2) private `mod` declarations,
(3) `pub use` re-exports, (4) `#[cfg(test)] mod <name>_test;` per module.

```rust
// crates/cb-core/src/lib.rs:8-27  — COPY THIS SHAPE for cb-core (extend) and cb-data (fill)
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )
)]

mod error;
mod rng;

pub use error::{CbError, CbResult};
pub use rng::TFastRng64;

#[cfg(test)]
mod error_test;
#[cfg(test)]
mod rng_test;
```

For `cb-core/src/lib.rs` the modification is purely **additive**: add `mod reduction;`,
add `pub use reduction::...;`, add `#[cfg(test)] mod reduction_test;`.

For `cb-data/src/lib.rs` (currently a 3-line stub, `crates/cb-data/src/lib.rs:1-3`),
replace the stub body with the full wiring for all Phase 2 modules following the
exact `cb-core`/`cb-oracle` template. Note `cb-data`'s current stub uses the
single-line `#![cfg_attr(...)]` form (line 1) — prefer the multi-line `cb-core`
form for consistency.

### Shared Pattern B — Source/test separation (HARD RULE — applies to EVERY module)

**Sources:** `crates/cb-core/src/rng.rs` (zero `#[cfg(test)]` anywhere) +
`crates/cb-core/src/rng_test.rs` (the dedicated test file) + the CI gate
`scripts/check-source-test-separation.sh`.

**The rule the executor MUST follow exactly:**
- Production `.rs` files contain **only** implementation. **No** `#[cfg(test)] mod tests { ... }` brace body anywhere — `scripts/check-source-test-separation.sh:35-64` greps for and fails CI on the brace form.
- Every module `foo.rs` gets a sibling `foo_test.rs`.
- The **declaration** `#[cfg(test)] mod foo_test;` (semicolon form) lives in `lib.rs`, NOT a brace-body in the source file. The gate explicitly allows the semicolon form and flags only `mod ... {`.

Test file header convention (copy from `rng_test.rs:1-11`): a module doc comment
citing the upstream source of the expected vectors, then `use crate::...;` imports.

```rust
// crates/cb-core/src/rng_test.rs:1-17  — the test-file template
//! Bitstream-oracle tests for [`crate::rng::TFastRng64`].
//! Every expected value here is transcribed verbatim from the vendored upstream
//! unit test `.../fast_ut.cpp`. ... Kept in a dedicated `*_test.rs` file per the
//! source/test separation rule (D-17); no `#[cfg(test)] mod` lives in `rng.rs`.

use crate::error::CbError;
use crate::rng::TFastRng64;

#[test]
fn test3_from_seed_17_first_gen_rand() { /* ... assert_eq! against upstream vector ... */ }
```

### Shared Pattern C — `thiserror` `CbError`, no `unwrap`, fallible APIs return `CbResult`

**Source:** `crates/cb-core/src/error.rs:9-29` (the enum) + `rng.rs:188-208`
(fallible `try_*` returning `CbResult`, infallible wrapper never panicking) +
`cb-oracle/src/error.rs:8-71` (richer enum with `#[from]` and struct variants).

Phase 2 EXTENDS `cb-core/src/error.rs` with the ingestion-validation taxonomy (D-06).
Copy the existing variant style exactly: struct variants with named fields, a
`#[error("...")]` message per variant, `#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]`.

```rust
// crates/cb-core/src/error.rs:16-29  — EXTEND with new variants in this exact style.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CbError {
    #[error("uniform bound must be > 0, got {bound}")]
    InvalidBound { bound: u64 },

    #[error("value out of range: {0}")]
    OutOfRange(String),

    // Phase 2 ADDS (shape only — names/fields to be finalized in planning):
    //   #[error("dtype mismatch: expected {expected}, got {got}")]
    //   Dtype { expected: &'static str, got: String },
    //   #[error("column length mismatch: ...")]   LengthMismatch { ... }
    //   #[error("NaN not allowed in categorical column {column}")] NanInCategorical { column: usize },
}
```

For `#[from]`-wrapped external errors (Arrow/Polars error → `CbError`), follow
`cb-oracle/src/error.rs:60-70`'s `#[from]` variant pattern (`Npy(#[from] ...)`,
`Json(#[from] ...)`, `Io(#[from] ...)`). NOTE: those `#[from]` variants make the
enum **non-`Clone`/`Eq`**; `cb-oracle`'s `OracleError` is `#[derive(Debug, thiserror::Error)]`
only. If Phase 2 adds `#[from]` arrow errors to `CbError`, it must drop
`Clone/PartialEq/Eq` from the derive (or wrap the message as `String`). Planning
must resolve this — prefer stringifying ingestion errors to keep `CbError: Clone + Eq`.

**`unwrap`/`expect`/`panic`/indexing are denied** workspace-wide
(`Cargo.toml:10-14`). Provide a fallible `try_*` returning `CbResult` and, where an
infallible convenience is wanted, a wrapper that degrades gracefully instead of
panicking — exactly `rng.rs:196-218` (`try_uniform` → `Result`; `uniform` →
`.unwrap_or(0)`).

### Shared Pattern D — Centralized workspace deps + per-crate `.workspace = true`

**Source:** `Cargo.toml:19-26` (`[workspace.dependencies]`) +
`crates/cb-oracle/Cargo.toml:10-21` (consumption with `[lints] workspace = true`
and `dep.workspace = true`) + `crates/cb-core/Cargo.toml:7-12` (the `anyhow`-absent
note).

```toml
# Root Cargo.toml:19-26  — ADD arrow/polars here (RESEARCH: arrow 59.0.0, polars 0.54.4)
[workspace.dependencies]
thiserror   = "2.0.18"
ndarray     = "0.17.2"
ndarray-npy = "0.10.0"
serde       = { version = "1.0.228", features = ["derive"] }
# Phase 2 ADDS:
# arrow  = "59.0.0"
# polars = { version = "0.54.4", default-features = false, features = ["dtype-full"] }
```

```toml
# crates/cb-oracle/Cargo.toml:7-21  — the per-crate consumption shape for cb-data
[lints]
workspace = true

[dependencies]
thiserror.workspace = true
cb-core = { path = "../cb-core" }
# Phase 2 cb-data ADDS: arrow.workspace = true / polars.workspace = true
# NOTE (cb-data/Cargo.toml:11): anyhow stays ABSENT from [dependencies] (D-14).
```

---

## Pattern Assignments

### `crates/cb-core/src/reduction.rs` (utility, transform/reduce)

**Analog:** `crates/cb-core/src/rng.rs` — same crate, same "exact C++ parity port as
a small pure-function utility" character. Copy its doc-comment + parity-contract +
non-crypto/caveat structure and its `use crate::error::...` import line.

**Module doc + parity contract** (model on `rng.rs:1-21`): open with `//!` doc
naming the upstream source file:line the order is transcribed from
(RESEARCH cites `binarization.cpp:803-815`, `calc_class_weights.cpp:36-54`), and
state the contract: **sequential `f64` fold, NO Kahan/pairwise** (RESEARCH Pitfall 1).

**Core pattern** (the only summation primitive — RESEARCH Code Examples lines 337-349):
```rust
// Sequential f64 fold — matches upstream order under thread_count=1.
pub fn sum_f64(values: &[f64]) -> f64 {
    let mut acc = 0.0_f64;
    for &v in values { acc += v; }   // plain sequential add — NO .sum()/.fold (D-08 grep bans those)
    acc
}
pub fn sum_f32_in_f64(values: &[f32]) -> f64 {
    let mut acc = 0.0_f64;
    for &v in values { acc += v as f64; }
    acc
}
```
Self-banned: this primitive must NOT itself use raw `.sum()`/`.fold(0.0, +)` — it is
the one place a hand-written loop is the implementation. The `scripts/check-no-raw-float-sum.sh`
gate (below) will need an allow-list exception for this file (mirror how
`check-no-anyhow.sh` excludes test files — here exclude `reduction.rs` itself).

**Error handling:** This primitive is infallible (no `CbResult` needed) — like
`rng.rs`'s `gen_rand`. Only the bound-checked `try_*` style applies where a
precondition exists.

---

### `crates/cb-core/src/reduction_test.rs` (test, property)

**Analog:** `crates/cb-core/src/rng_test.rs:1-85`.

**Critical:** RESEARCH (Validation Architecture, point 1) specifies a **property**
test, not an oracle fixture: construct an input where naive-sequential ≠ pairwise
(`[1e16, 1.0, -1e16]`) and assert the primitive returns the naive-sequential result.
This locks the *order*, not just the value. Copy the `rng_test.rs` header-comment
style (cite the upstream order source) and the one-`#[test]`-per-property layout.

```rust
// Model on rng_test.rs:14-18 — one #[test] per locked property.
#[test]
fn sum_uses_naive_sequential_order_not_pairwise() {
    // [1e16, 1.0, -1e16]: sequential = 0.0, pairwise/Kahan = 1.0. Lock sequential.
    assert_eq!(sum_f64(&[1e16, 1.0, -1e16]), 0.0_f64);
}
```

---

### `crates/cb-data/src/borders.rs` (utility, transform — GreedyLogSum)

**Analog:** `crates/cb-core/src/rng.rs` — the canonical "bit-exact transcription of
cited C++ with `wrapping_*`/explicit-type discipline and per-step `//` comments
quoting the C++" pattern. `borders.rs` is the same kind of file: a faithful
transcription, NOT a clean reimplementation (RESEARCH "Don't Hand-Roll" key insight).

**Pattern to copy from `rng.rs`:**
- Per-function `//` comments quoting the exact C++ line being ported (rng.rs:32-37, 94-98).
- Explicit type discipline: RESEARCH Pitfall 2 — border arithmetic is **`f32`**
  (`0.5f * a + 0.5f * b`), penalty/DP error accumulators are **`f64`**. Mirror
  `rng.rs`'s deliberate `u32`/`u64` separation, but for `f32`/`f64`.
- All summation routes through `cb-core::reduction` (D-07/D-08) — `borders.rs` must
  `use cb_core::reduction::...` and never write a raw float `.sum()`.

**Source citations to transcribe** (from RESEARCH Pattern 1): priority-queue greedy
`TFeatureBin` split over object counts, penalty `-log(count + 1e-8)`, border
`0.5f*values[start-1] + 0.5f*values[start]`, final `THashSet<float>` → sort ascending,
`-0.0f`→`0.0f` fix (`binarization.cpp:1319-1520, 1676-1714, 897-900`).

---

### `crates/cb-data/src/borders_test.rs` (test, oracle)

**Analog:** `crates/cb-oracle/tests/per_stage_oracle_test.rs:1-53` (oracle-fixture
read + `compare_stage` gate) combined with `rng_test.rs` per-`#[test]` layout.

**Pattern:** load `borders.npy` + `borders_per_feature.npy` via
`cb_oracle::load_f64_vec` (fixture.rs:21-24), split per-feature by the counts, run
`borders.rs` on the matching frozen input column, and gate with
`compare_stage(Stage::Borders, expected, actual)` (compare.rs:70 — note `Stage::Borders`
ALREADY EXISTS at compare.rs:11-12). Use the `fixture()` PathBuf+`env!("CARGO_MANIFEST_DIR")`
helper pattern from per_stage_oracle_test.rs:15-20. This is likely an integration
test under `crates/cb-data/tests/` (so it can depend on `cb-oracle`) — mirror
`cb-oracle/tests/`'s `#![allow(...)]` top-line (per_stage_oracle_test.rs:9).

---

### `crates/cb-data/src/cat_hash.rs` (utility, transform — CityHash64 port)

**Analog:** `crates/cb-core/src/rng.rs` — again the bit-exact-port pattern. CityHash64
is, like `TFastRng64`, an integer-exact transcription of a vendored Yandex C++ source
(`util/digest/city.cpp`, RESEARCH Pitfall 4 / Open Q4). Same discipline: `wrapping_*`
arithmetic, `//`-quoted C++ lines, explicit `u32`/`u64`. RESEARCH recommends porting
the vendored `city.cpp`/`city.h` directly rather than a crate.

**Core:** `CalcCatFeatureHash = CityHash64(bytes) & 0xffffffff` (cat_feature.cpp:6-8),
then first-seen perfect-hash remap (helper.cpp:111-131). The non-crypto caveat doc
block from `rng.rs:1-12` applies verbatim (RESEARCH Security V6 — CityHash is
non-cryptographic; copy the caveat).

---

### `crates/cb-data/src/cat_hash_test.rs` (test, bit-exact vectors)

**Analog:** `crates/cb-core/src/rng_test.rs:13-32` — the "expected values transcribed
verbatim from a vendored C++ unit test, asserted bit-exact with `assert_eq!`" pattern.
Cat hashing is integer-exact (NOT ≤1e-5 — RESEARCH Validation point 4): use
`assert_eq!` on `(string → ui32)` vectors, exactly like the PRNG bitstream vectors.

---

### `crates/cb-data/src/nan_mode.rs` (model, enum + transform)

**Analog:** `crates/cb-oracle/src/compare.rs:9-21` — the small `#[derive(Debug, Clone,
Copy, PartialEq, Eq)] enum Stage { ... }` with per-variant doc comments is the exact
shape for `enum NanMode { Min, Max, Forbidden }`. Bin-assignment logic
(strict `value > border`, sentinel insertion) follows the `borders.rs`/`rng.rs`
C++-transcription discipline (RESEARCH Pattern 2 & 3, `quantization.cpp:322-345`,
`utils.h:51-66`).

---

### `crates/cb-data/src/quantized_pool.rs` (model, SoA struct + width enum)

**Analog:** `crates/cb-oracle/src/compare.rs:9-21` for the `ColumnBins` width enum
(`#[derive]` + per-variant docs), `D-11`: `enum ColumnBins { U8(Vec<u8>), U16(Vec<u16>),
U32(Vec<u32>) }`. The immutable-SoA struct + accessor methods follow the
`rng.rs`/`Model` (vendored) struct-with-methods convention. Width selection
(`<256 → U8`, `<65536 → U16`, cat-only `U32`) transcribed from `utils.h:175-181`
(RESEARCH Pattern 5).

---

### `crates/cb-data/src/pool.rs` (model, struct + builder)

**Analog:** No exact in-repo builder analog exists; the closest **established** Rust
analog is `crates/cb-core/src/rng.rs` (struct + `new`/`from_*` constructors +
methods). For the builder/typestate idiom specifically, the **vendored reference**
`catboost-master/catboost/rust-package/src/features.rs` (`ObjectsOrderFeatures`
`.with_float_features()`/`.with_cat_features()` builder, cited in CLAUDE.md
Architecture) is the design reference — but it is NOT a workspace file to copy lints
from. Copy:
- struct + `#[must_use]` constructor pattern from `rng.rs:140-170`,
- D-02: owned `Vec`-backed columns (no `Pool<'a>`/`Cow`),
- the ingestion-trait seam (D-04) — `Pool` is constructed *through* the
  `ingest::IngestSource` trait so Phase 8 can plug a borrowed view in.

---

### `crates/cb-data/src/quantize.rs` (service, transform driver)

**Analog:** `crates/cb-core/src/rng.rs` as an orchestration analog (a type whose
methods compose sub-utilities). `quantize.rs` realizes `pool.quantize(&params) ->
QuantizedPool` (D-01), calling `borders.rs`, `nan_mode.rs`, and `cat_hash.rs`, routing
all sums through `cb-core::reduction`. Error path returns `CbResult` per Shared
Pattern C.

---

### `crates/cb-data/src/weights.rs` (utility, transform/reduce)

**Analog:** `crates/cb-core/src/rng.rs` (parity-port utility). Balanced = `max/w`,
SqrtBalanced = `sqrt(max/w)`, floor `1e-8`, summary weights in `f64`
(`calc_class_weights.cpp:11-27, 36-54`, RESEARCH DATA-08). All accumulation routes
through `cb-core::reduction::sum_*` (D-07). Test analog: `weights_test.rs` follows
the oracle-fixture pattern of `per_stage_oracle_test.rs` against the new
`class_weights/` fixture (RESEARCH Validation point 6).

---

### `crates/cb-data/src/ingest/mod.rs` + `arrow.rs` + `polars.rs` + `owned.rs`

**Analog for the trait + module wiring:** `crates/cb-core/src/lib.rs:18-27` (sub-module
declaration + `pub use`). `ingest/mod.rs` declares the `IngestSource` trait (D-04 seam)
and re-exports the impls.

**Analog for each ingestion impl (load + validate → typed error):**
`crates/cb-oracle/src/fixture.rs:21-47` — the "read an external source, validate, map
failure to a typed `thiserror` error via `?`, return `Result`" shape.

```rust
// crates/cb-oracle/src/fixture.rs:21-24  — the load+validate→typed-error template
pub fn load_f64_vec(path: &Path) -> Result<Vec<f64>, OracleError> {
    let arr: Array1<f64> = read_npy(path)?;   // external read; error via #[from], no panic
    Ok(arr.to_vec())
}
```
`arrow.rs` implements this with `col.data_type() != &DataType::Float64` →
`CbError::Dtype` and `as_primitive::<Float64Type>().values()` zero-copy (RESEARCH
Code Examples lines 353-368). `polars.rs` rechunks then rides the same Arrow
validation path (RESEARCH lines 371-381, D-05 confirmed). `owned.rs` is the trivial
`Vec<f64>` primitive — the `load_f64_vec` analog without external I/O.

**Ingest test analog:** `crates/cb-oracle/src/fixture_test.rs` (sibling `*_test.rs`,
exercises load + the error branch). Per source/test separation these may be
`ingest/arrow_test.rs` etc. with `#[cfg(test)] mod arrow_test;` declared in
`ingest/mod.rs`.

---

### `scripts/check-no-raw-float-sum.sh` (config, CI gate — D-08)

**Analog:** `scripts/check-no-anyhow.sh:1-49` — copy it **wholesale**, changing only
the grep pattern and the crate scope. Reuse: the `ROOT="$(cd ...)"` resolver
(lines 9-11), the `CORE_DIRS` loop with `[ -d "$dir" ] || continue` tolerance
(lines 13-27), the `*_test.rs` exclusion via `case "$file" in *_test.rs) continue` 
(lines 31-33), and the `violations`/exit-1 epilogue (lines 42-48).

```bash
# scripts/check-no-anyhow.sh:30-39  — the per-file scan loop to copy.
while IFS= read -r file; do
  case "$file" in
    *_test.rs) continue ;;
  esac
  if grep -In 'anyhow' "$file" >/dev/null 2>&1; then   # CHANGE: pattern → raw float .sum()/.fold(0.0
    echo "violation: ... $file"; violations=1
  fi
done < <(grep -RIl --include='*.rs' -e 'anyhow' "$dir" 2>/dev/null || true)
```
**Required deviation:** D-08 bans raw float `.sum()` / `.fold(0.0, +)` in library
crates — but `crates/cb-core/src/reduction.rs` IS the sanctioned sequential loop.
Exclude `reduction.rs` (and `*_test.rs`) from the scan, exactly as `check-no-anyhow.sh`
excludes `*_test.rs`. Add the script to the same CI step that runs the existing two
gates.

---

### `crates/cb-oracle/generator/{gen_inputs.py, gen_fixtures.py}` (MODIFY, batch fixture gen)

**Analog:** the files themselves. `gen_inputs.py:_write()/_assert_f64()/gen_numeric_tiny()`
(lines 41-60+) is the template for the new `numeric_nan` and explicit-categorical
datasets. `gen_fixtures.py`'s borders-extraction block (`get_borders()` → flat
`borders.npy` + `borders_per_feature.npy` + `config.json`, lines 73-95) is the
template for the new `borders_quant/`, `cat_hash/`, and `class_weights/` fixtures
(RESEARCH Validation "Oracle Fixture Schema"). Keep the BUILD-TIME-only,
`thread_count=1`, pinned-seed, `_assert_f64` determinism discipline (D-12).

---

## No Analog Found

None. Every Phase 2 file maps to a concrete Phase 1 file in this workspace. The only
files whose *design* reaches outside the workspace are `pool.rs` (builder idiom →
vendored `catboost-master/.../features.rs`, reference-only) and the parity algorithms
in `borders.rs`/`cat_hash.rs`/`weights.rs`/`nan_mode.rs` (math → vendored C++ in
`catboost-master/`, transcription-only). For all four, the **Rust file shape, lint,
error, and test conventions** still come from `cb-core/src/rng.rs` — only the
*algorithm content* comes from `catboost-master/`. RESEARCH already supplies the
exact C++ file:line citations for each.

---

## Metadata

**Analog search scope:** `crates/cb-core/`, `crates/cb-data/`, `crates/cb-oracle/`,
root `Cargo.toml`, `scripts/`, `crates/cb-oracle/generator/`, `crates/cb-oracle/fixtures/`.
**Files scanned:** 16 Rust (cb-core: lib/error/rng + 3 tests; cb-oracle:
lib/error/fixture/compare + 2 unit tests + per_stage integration test; cb-data stub),
4 Cargo manifests, 2 CI scripts, 2 generator scripts, fixtures tree.
**Pattern extraction date:** 2026-06-13
```
