# Phase 1: Workspace, Lint Discipline & Oracle Harness - Pattern Map

**Mapped:** 2026-06-13
**Files analyzed:** 24 new files (greenfield workspace)
**Analogs found:** 4 with real in-repo analog / 24 total (20 are greenfield — no in-repo analog by design)

> **Greenfield context:** No root Cargo workspace, no `cb-*` crate, and no `catboost-rs` facade exist yet. Most files in this phase are pure scaffolding with no in-repo analog. The only files with a genuine analog are:
> 1. `cb-core/src/error.rs` → vendored `catboost-master/catboost/rust-package/src/error.rs` (thiserror-shaped, but the vendored one is hand-rolled and uses `unwrap()` — flagged below).
> 2. `cb-core/src/rng.rs` + `cb-core/src/rng_test.rs` → vendored C++ PRNG source `util/random/fast.{h,cpp}`, `lcg_engine.{h,cpp}`, `common_ops.h`, and test vectors in `fast_ut.cpp`. **This is a port, not a copy** — the analog is C++, the excerpts below are the exact algorithm to transcribe.
> 3. Test-file structure (any `*_test.rs`) → the `*_test.rs` convention documented in `.planning/codebase/TESTING.md` (NOT the inline-`mod tests` anti-pattern in the vendored `model.rs`).
>
> Where a file is marked **NEW — greenfield (no analog)**, the planner should use RESEARCH.md Code Examples / Architecture Patterns directly; do not invent a false analog.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `Cargo.toml` (workspace root) | config | n/a | RESEARCH.md Pattern 1 | NEW — greenfield |
| `rust-toolchain.toml` | config | n/a | — | NEW — greenfield |
| `crates/cb-core/Cargo.toml` | config | n/a | RESEARCH.md Pattern 1 | NEW — greenfield |
| `crates/cb-core/src/lib.rs` | module-root | n/a | `rust-package/src/lib.rs` (module-decl shape only) | partial |
| `crates/cb-core/src/error.rs` | model (error type) | request-response | `rust-package/src/error.rs` | role-match (port + modernize) |
| `crates/cb-core/src/rng.rs` | utility (PRNG) | transform | `util/random/fast.{h,cpp}` + `lcg_engine.{h,cpp}` + `common_ops.h` (C++) | exact-algorithm (cross-language port) |
| `crates/cb-core/src/rng_test.rs` | test | transform | `util/random/fast_ut.cpp` (C++ vectors) | exact-vectors |
| `crates/cb-core/src/error_test.rs` | test | request-response | TESTING.md `*_test.rs` convention | convention-match |
| `crates/cb-oracle/Cargo.toml` | config | n/a | RESEARCH.md structure | NEW — greenfield |
| `crates/cb-oracle/src/lib.rs` | module-root | n/a | — | NEW — greenfield |
| `crates/cb-oracle/src/fixture.rs` | service (loader) | file-I/O | RESEARCH.md Rust fixture-read example (`ndarray-npy`) | NEW — greenfield (lib API) |
| `crates/cb-oracle/src/compare.rs` | service (comparator) | transform | RESEARCH.md Pattern 3 + vendored `abs_diff_eq!` idiom | NEW — greenfield (lib API) |
| `crates/cb-oracle/src/fixture_test.rs` | test | file-I/O | TESTING.md `*_test.rs` convention | convention-match |
| `crates/cb-oracle/src/compare_test.rs` | test | transform | TESTING.md `*_test.rs` + `approx` idiom | convention-match |
| `crates/cb-oracle/generator/requirements.txt` | config | n/a | — | NEW — greenfield |
| `crates/cb-oracle/generator/gen_inputs.py` | utility (script) | batch / file-I/O | RESEARCH.md generator example | NEW — greenfield |
| `crates/cb-oracle/generator/gen_fixtures.py` | utility (script) | batch / file-I/O | RESEARCH.md generator example | NEW — greenfield |
| `crates/cb-data/{Cargo.toml,src/lib.rs}` | config + stub | n/a | — | NEW — greenfield (stub) |
| `crates/cb-compute/{Cargo.toml,src/lib.rs}` | config + stub | n/a | RESEARCH.md (NO cubecl, D-03) | NEW — greenfield (stub) |
| `crates/cb-backend/{Cargo.toml,src/lib.rs}` | config + stub | n/a | RESEARCH.md Pattern 2 (feature-gated alias) | NEW — greenfield (stub) |
| `crates/cb-train/{Cargo.toml,src/lib.rs}` | config + stub | n/a | — | NEW — greenfield (stub) |
| `crates/cb-model/{Cargo.toml,src/lib.rs}` | config + stub | n/a | — | NEW — greenfield (stub) |
| `crates/catboost-rs/{Cargo.toml,src/lib.rs}` | config + facade stub | n/a | — | NEW — greenfield (stub) |
| `.github/workflows/ci.yml` | config (CI) | event-driven | — | NEW — greenfield |
| `scripts/check-no-anyhow.sh` | utility (script) | batch | RESEARCH.md Code Example (verbatim) | NEW — greenfield |

## Pattern Assignments

### `crates/cb-core/src/error.rs` (model / error type, request-response)

**Analog:** `catboost-master/catboost/rust-package/src/error.rs` (full file, 39 lines — already in context).

**What to take from the analog (shape):** the `pub type XxxResult<T> = Result<T, XxxError>;` alias pattern and a single error type that implements `Error + Display`. The vendored version hand-rolls `impl fmt::Display` and `impl std::error::Error` (lines 32-38) and derives `Debug, Eq, PartialEq` (line 6).

**What to DEVIATE from (D-15 / CLAUDE.md mandate):**
- **Use `thiserror`, not hand-rolled `impl Display`/`impl Error`.** CLAUDE.md + D-15 mandate `thiserror` for all library error types. The vendored hand-roll is the *old approach* (RESEARCH.md "State of the Art": `error-chain`/manual `impl Error` superseded by `thiserror`).
- **Remove the `unwrap()`.** Vendored line 25 (`c_str.to_str().unwrap()`) violates D-13 `clippy::unwrap_used = "deny"`. New code must propagate or handle the error. (cb-core has no FFI / no `GetErrorString()` anyway — that whole FFI body does not port.)
- **Keep the result-alias idiom** (vendored line 4): `pub type CbResult<T> = Result<T, CbError>;` is the one piece worth copying directly.

**Target shape (thiserror, derived — NOT the vendored hand-roll):**
```rust
use thiserror::Error;

pub type CbResult<T> = std::result::Result<T, CbError>;

#[derive(Debug, Error)]
pub enum CbError {
    // Phase 1 needs at least the oracle/comparator-facing variants surfaced by cb-oracle.
    // Concrete variants are defined by the planner per the OracleError needs below.
}
```

---

### `crates/cb-core/src/rng.rs` (utility / PRNG, transform)

**Analog:** vendored C++ — `util/random/fast.h`, `fast.cpp`, `lcg_engine.h`, `lcg_engine.cpp`, `common_ops.h`. This is a **cross-language exact port**; the excerpts below are the algorithm to transcribe bit-for-bit. RESEARCH.md (Code Examples) already gives the consolidated Rust target — these analog excerpts are the authoritative source of truth behind it.

**Constants & per-stream addend** (`lcg_engine.h:14-43`):
```cpp
// A = 6364136223846793005 (0x5851F42D4C957F2D)
template <typename T, T A>
struct TLcgIterator {
    inline TLcgIterator(T seq) : C((seq << 1u) | (T)1) {}   // C must be ODD
    inline T Iterate(T x) const { return x * A + C; }        // <-- wrapping_mul/wrapping_add in Rust
};
// TFastLcgIterator<ui64, A, 1>: fixed addend C = 1 (used by TReallyFastRng32)
```
> **Pitfall 5:** `x * A + C` MUST become `x.wrapping_mul(A).wrapping_add(C)` in Rust — debug builds panic on overflow otherwise, and `clippy::panic`/`arithmetic` would also flag it. Apply `wrapping_*` throughout `Iterate`, `LcgAdvance`, the `(seq<<1)|1` addend, and the mixer.

**PCG output mixer** (`fast.h:11-18`):
```cpp
struct TPCGMixer {
    static inline ui32 Mix(ui64 x) noexcept {
        const ui32 xorshifted = ((x >> 18u) ^ x) >> 27u;
        const ui32 rot = x >> 59u;
        return RotateBitsRight(xorshifted, rot);   // -> u32::rotate_right(xorshifted, rot)
    }
};
```

**GenRand (advance state, then mix)** (`lcg_engine.h:57-59`):
```cpp
inline TResultType GenRand() noexcept { return this->Mix(X = this->Iterate(X)); }
```

**64-bit combine** (`fast.h:68-73`):
```cpp
inline ui64 GenRand() noexcept {
    const ui64 x = R1_.GenRand();   // 32-bit out, widened
    const ui64 y = R2_.GenRand();
    return (x << 32) | y;
}
```

**Four-arg ctor + FixSeq (distinct streams)** (`fast.cpp:5-19`):
```cpp
static inline ui32 FixSeq(ui32 seq1, ui32 seq2) noexcept {
    const ui32 mask = (~(ui32)(0)) >> 1;            // u32::MAX >> 1
    if ((seq1 & mask) == (seq2 & mask)) return ~seq2;
    return seq2;
}
TFastRng64::TFastRng64(ui64 seed1, ui32 seq1, ui64 seed2, ui32 seq2)
    : R1_(seed1, seq1), R2_(seed2, FixSeq(seq1, seq2)) {}
```

**One-arg ctor derives the four args via TReallyFastRng32(seed)** (`fast.cpp:21-28`):
```cpp
TFastRng64::TArgs::TArgs(ui64 seed) {
    TReallyFastRng32 rng(seed);          // A=6364..2D, C=1
    Seed1 = rng.GenRand64();             // ToRand64: (u64)x | ((u64)next_GenRand() << 32)
    Seq1  = rng.GenRand();
    Seed2 = rng.GenRand64();
    Seq2  = rng.GenRand();
}
```

**GenRand64 / ToRand64 for a 32-bit rng** (`common_ops.h:35-38, 94-96`):
```cpp
static inline ui64 ToRand64(T&& rng, ui32 x) { return ((ui64)x) | (((ui64)rng.GenRand()) << 32); }
inline ui64 GenRand64() { return ToRand64(Engine(), Engine().GenRand()); }
```

**Uniform(t) — rejection sampling** (`common_ops.h:48-60, 75-86`):
```cpp
// RandMax() for TFastRng64 == ui64(-1) == u64::MAX
const T randmax = gen.RandMax() - gen.RandMax() % max;   // wrapping not needed; max>0 asserted
T rand;
while ((rand = gen.GenRand()) >= randmax) { /* reject */ }
return rand % max;
```
> Port the `Y_ABORT_UNLESS(max > 0)` guard as a `CbResult`/debug-checked precondition, NOT a `panic!` (D-13 `clippy::panic = "deny"`). Returning `CbError` is the clippy-clean equivalent.

**Advance(delta) — LcgAdvance binary exponentiation** (`lcg_engine.cpp:5-26`):
```cpp
T LcgAdvance(T seed, T lcgBase /*A*/, T lcgAddend /*C*/, T delta) {
    T mask = 1;
    while (mask != (1ULL << (8*sizeof(T)-1)) && (mask << 1) <= delta) mask <<= 1;
    T apow = 1; T adiv = 0;
    for (; mask; mask >>= 1) {
        adiv *= apow + 1;                       // -> adiv = adiv.wrapping_mul(apow.wrapping_add(1))
        apow *= apow;                           // -> apow = apow.wrapping_mul(apow)
        if (delta & mask) { adiv += apow; apow *= lcgBase; }  // wrapping_add / wrapping_mul
    }
    return seed * apow + lcgAddend * adiv;       // seed.wrapping_mul(apow).wrapping_add(C.wrapping_mul(adiv))
}
// TFastRng64::Advance advances R1_ and R2_ each by delta (fast.h:75-78)
```

**Security doc-comment (RESEARCH.md Security Domain V6):** `rng.rs` MUST carry a doc-comment stating `TFastRng64` is a **non-cryptographic** PRNG for parity reproduction only — never for secrets.

---

### `crates/cb-core/src/rng_test.rs` (test, transform)

**Analog:** `util/random/fast_ut.cpp` (full file, 119 lines — in context). Transcribe the canonical vectors. **Do NOT build a C++ harness (Pitfall 4 — RESOLVED).**

**Vectors to port (exact, from `fast_ut.cpp`):**
```cpp
// Test3 (line 85-89): single-seed GenRand
TFastRng64 rng(17);  rng.GenRand() == 14895365814383052362ULL

// Test2 (line 15-44): four-arg ctor, first 20 of Uniform(100u)
TFastRng64 rng(0, 1, 2, 3);
// 37,43,76,17,12,87,60,4,83,47,57,81,28,45,66,74,18,17,18,75

// TestAdvance (line 46-62): 100×GenRand() == Advance(100) then GenRand()
TFastRng64 a(0,1,2,3), b(0,1,2,3);
for i in 0..100 { a.GenRand(); }   b.Advance(100);
a.GenRand() == b.GenRand()

// TestAdvanceBoundaries (line 64-72): Advance(0) and Advance(1) are no-op/1-step equivalents
// (optional extra coverage — also in fast_ut.cpp)
```
**File-structure rule (D-17):** this is a SEPARATE `*_test.rs` file. Add the test-cfg clippy allow header (see Shared Patterns → Lint/Test Exemption) so `unwrap()`/`assert_eq!`/indexing in tests do not trip the workspace lints.

---

### `crates/cb-oracle/src/compare.rs` (service / comparator, transform)

**Analog:** RESEARCH.md Pattern 3 (per-stage comparator) — no in-repo Rust analog exists. The one in-repo idiom to align with is the vendored float-assertion style (`abs_diff_eq!(l, r, epsilon=...)`, used in `model.rs:413` and documented in TESTING.md), but the production comparator must return a `Result`, not `assert!` (D-13).

**Target primitive (from RESEARCH.md Pattern 3, tol = 1e-5 per D-12):**
```rust
pub enum Stage { Borders, Splits, LeafValues, StagedApprox, Predictions }

pub fn assert_abs_close(expected: &[f64], actual: &[f64], tol: f64) -> Result<(), OracleError> {
    if expected.len() != actual.len() { return Err(OracleError::LengthMismatch { /* .. */ }); }
    for (i, (e, a)) in expected.iter().zip(actual).enumerate() {
        let d = (e - a).abs();
        if d > tol { return Err(OracleError::Diverged { index: i, expected: *e, actual: *a, diff: d }); }
    }
    Ok(())
}
```
> Use `.iter().zip().enumerate()` instead of indexing to stay clippy-clean under `indexing_slicing = "deny"`. The `approx`/`abs_diff_eq!` idiom is for the *test* side (`compare_test.rs`), not the production comparator.

---

### `crates/cb-oracle/src/fixture.rs` (service / loader, file-I/O)

**Analog:** RESEARCH.md Rust fixture-read example — no in-repo analog. `ndarray-npy` `read_npy` errors on dtype mismatch (the desired guard, Pitfall 3).
```rust
use ndarray::Array1;
use ndarray_npy::read_npy;

pub fn load_f64_vec(path: &std::path::Path) -> Result<Vec<f64>, OracleError> {
    let arr: Array1<f64> = read_npy(path)?;   // dtype-mismatch -> Err, never panic
    Ok(arr.to_vec())
}
```
> The `config.json` half (seed/version/params metadata) is parsed with `serde`/`serde_json` into a metadata struct. Map `ndarray_npy::ReadNpyError` and `serde_json::Error` into `OracleError` variants via `#[from]` (thiserror).

---

### Stub crates (`cb-data`, `cb-compute`, `cb-backend`, `cb-train`, `cb-model`, `catboost-rs`)

**Analog:** none — greenfield stubs (D-01: stub all crates day one). Each = `Cargo.toml` (`[lints] workspace = true`, NO `anyhow` in `[dependencies]` for the 6 core libs per D-14) + minimal `src/lib.rs` with the test-cfg clippy allow header.

**`cb-backend` exception (RESEARCH.md Pattern 2 / D-02):** single feature-gated crate, `[features] cpu/wgpu/cuda/rocm`, one `cfg`-selected `pub type SelectedRuntime = ();` placeholder per arm. NO per-backend crate. **`cb-compute` exception (D-03):** must NOT depend on cubecl.

---

### `scripts/check-no-anyhow.sh`

**Analog:** RESEARCH.md Code Example (verbatim). Greps the 6 core-lib `crates/` dirs for `anyhow`, excluding `*_test.rs`. Belt-and-suspenders behind the structural ban (D-14).

### `.github/workflows/ci.yml`

**Analog:** none — greenfield. CPU lane ONLY (D-16): `cargo build` (cpu), `cargo clippy --workspace -- -D warnings` (with `--lib` scoping decision from Pitfall 1), `scripts/check-no-anyhow.sh`, CPU oracle tests. **No GPU/ROCm job** (D-16). Generator does NOT run in CI (D-12).

## Shared Patterns

### Lint/Test Exemption (Pitfall 1 — applies to EVERY library crate's `lib.rs` and every `tests/*.rs`)
**Source:** RESEARCH.md Pitfall 1 (no in-repo analog — greenfield policy).
**Apply to:** top of every `cb-*` / `catboost-rs` `src/lib.rs`, and every separate `*_test.rs` integration file.
```rust
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing))]
```
> Required because `[workspace.lints]` restriction lints fire inside test code, and `lints.workspace = true` forbids per-crate manifest lint overrides (RESEARCH.md Pattern 1 constraint) — so the exemption MUST live in-code. Plan 1 must also decide whether the CI clippy gate is `--lib`-scoped vs. blanket test-allow; pick one convention and apply to the first test file written.

### Source/Test Separation (D-17 — applies to ALL production `.rs` files)
**Source:** `.planning/codebase/TESTING.md:26,42-47` + CLAUDE.md mandatory rule.
**Apply to:** every production module (`error.rs`, `rng.rs`, `fixture.rs`, `compare.rs`, all stub `lib.rs`).
**Rule:** NO inline `#[cfg(test)] mod tests` in production source. Tests go in dedicated `src/<module>_test.rs`.
**Anti-pattern to NOT copy (in context):** vendored `rust-package/src/model.rs:267-268` (`#[cfg(test)] mod tests { ... }`) and `catboost-sys/src/lib.rs` — both predate the rule and are explicitly flagged "do not replicate" in TESTING.md:157-159.

### Centralized Workspace Lints (D-13)
**Source:** RESEARCH.md Pattern 1 — no in-repo analog (`.clippy.toml` confirmed absent per CONVENTIONS.md).
**Apply to:** root `Cargo.toml` (`[workspace.lints.clippy]` with the 4 deny lints) + every library crate (`[lints] workspace = true`).
```toml
# root Cargo.toml
[workspace.lints.clippy]
unwrap_used      = "deny"
expect_used      = "deny"
panic            = "deny"
indexing_slicing = "deny"
```

### thiserror error strategy + `#[from]` conversions (D-15)
**Source:** CLAUDE.md mandate; shape derived from vendored `error.rs` result-alias idiom (line 4) but MODERNIZED to `#[derive(Error)]`.
**Apply to:** `cb-core/src/error.rs` (`CbError`) and `cb-oracle` (`OracleError`). Use `#[from]` to wrap `ndarray_npy::ReadNpyError` and `serde_json::Error`. NO hand-rolled `impl Display`/`impl Error`. NO `unwrap()`.

### Wrapping arithmetic for PRNG parity (Pitfall 5)
**Source:** `lcg_engine.{h,cpp}`, `fast.cpp` (in context).
**Apply to:** all of `rng.rs` — every `*`/`+` in `Iterate`, `LcgAdvance`, the `(seq<<1)|1` addend, and the mixer becomes `wrapping_mul`/`wrapping_add`. Mandatory for bitstream parity AND to avoid debug-overflow panics under `clippy::panic`.

## No Analog Found

Files genuinely greenfield — planner should use RESEARCH.md (Architecture Patterns / Code Examples / Standard Stack) directly:

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `Cargo.toml` (root), `rust-toolchain.toml` | config | n/a | No prior workspace exists (greenfield); use RESEARCH.md Pattern 1 + structure |
| `cb-oracle/src/fixture.rs`, `compare.rs` | service | file-I/O / transform | No Rust oracle harness exists; design from RESEARCH.md Pattern 3 + ndarray-npy example |
| `cb-oracle/generator/*.py` | utility | batch | No Python generator exists; use RESEARCH.md verified `catboost==1.2.10` API example |
| `cb-oracle/fixtures/**` | data | file-I/O | Frozen corpus is first-created here (D-11); generated once, committed |
| `.github/workflows/ci.yml` | config | event-driven | No CI exists; author CPU lane per D-16 |
| `scripts/check-no-anyhow.sh` | utility | batch | No scripts dir exists; use RESEARCH.md verbatim example |
| 6 stub crates (`cb-data`/`cb-compute`/`cb-backend`/`cb-train`/`cb-model`/`catboost-rs`) | config + stub | n/a | New crates; stub-only per D-01 (cb-backend per Pattern 2, cb-compute no-cubecl per D-03) |

## Metadata

**Analog search scope:** `catboost-master/catboost/rust-package/src/` (Rust idioms), `catboost-master/util/random/` (PRNG port source), `.planning/codebase/` (documented conventions). Confirmed greenfield: no root `Cargo.toml`, no `crates/` dir.
**Files scanned:** `error.rs`, `features.rs`, `model.rs` (targeted grep), `fast.h`, `fast.cpp`, `fast_ut.cpp`, `lcg_engine.h`, `lcg_engine.cpp`, `common_ops.h`, `TESTING.md`.
**Pattern extraction date:** 2026-06-13
