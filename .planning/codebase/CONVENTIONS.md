# Coding Conventions

**Analysis Date:** 2026-06-13

## Source/Test Separation — Mandatory Rule

**AGENTS.md mandates strict separation of source and test code.**

- Embedding `mod tests` at the bottom of a production source file is **strictly prohibited**.
- All tests (unit and integration) must reside in separate, dedicated files.
- Permitted structures:
  - `tests/` directory (standard Rust integration tests)
  - Explicit separate module files, e.g. `src/foo_test.rs` alongside `src/foo.rs`, or `tests/foo_tests.rs`
- Production source files must contain only implementation logic — no `#[cfg(test)]` blocks embedded in them.

**Note:** The upstream `catboost-master/` subtree (the vendored CatBoost C++/Rust package) predates this rule and contains inline `mod tests` blocks inside `src/model.rs` and `catboost-sys/src/lib.rs`. Do not replicate this pattern in new project code.

## Naming Patterns

### Rust Code (this project)

**Files:**
- Snake_case for all Rust source files: `model.rs`, `features.rs`, `error.rs`
- Test files follow source name with `_test` or `_tests` suffix: `model_test.rs`, `foo_tests.rs`
- C++ file names use lowercase only (no capital letters), extensions `.cpp` and `.h`

**Types and Structs:**
- `PascalCase` for all types, structs, enums: `CatBoostError`, `ObjectsOrderFeatures`, `EmptyFloatFeatures`
- Generic type parameters use `T` prefix in PascalCase: `TFeature`, `TObjectFeatures`, `TFloatFeatures`, `TCatFeatures`

**Functions and Methods:**
- `snake_case` for all functions and methods: `load`, `load_buffer`, `check_return_value`, `get_float_features_count`

**Variables:**
- `snake_case` for all local variables and function parameters

**Constants:**
- `SCREAMING_SNAKE_CASE` for constants

**Modules:**
- `snake_case` for module names: `mod error`, `mod features`, `mod model`

### C++ Code (catboost-master/)

**Classes/Types:** `T` prefix + PascalCase: `TClass`, `TVector`
**Virtual interfaces:** `I` prefix: `IInterface`
**Namespaces:** `N` prefix + PascalCase: `NStl`, `NPrivate`
**Enums:** `E` prefix + PascalCase: `EFetchType`, `EStatus`
**Enum members (global):** `ALL_CAPS` with prefix of enum initials: `FT_SIMPLE`, `FT_SELECTED`
**Variables (local/global):** Start with lowercase letter
**Functions/Methods:** Start with uppercase letter
**Class members:** Start with uppercase letter
**Global constants/defines:** Fully `ALL_CAPS`
**Hungarian notation:** Prohibited

## Code Style

### Rust Formatting

No `rustfmt.toml` is present in the project root. The standard `rustfmt` defaults apply.

**Key observed patterns:**
- 4-space indentation
- Trailing commas in multi-line struct literals and function calls
- Generic bounds on their own lines when spanning multiple type parameters (see `predict()` in `model.rs`)
- Closure chains use `.collect::<Vec<_>>()` turbofish form

**Linting:**
- No `.clippy.toml` present. Clippy defaults apply.
- `catboost-sys/src/lib.rs` suppresses FFI-name warnings with crate-level `#![allow(non_upper_case_globals)]`, `#![allow(non_camel_case_types)]`, `#![allow(non_snake_case)]` — these are specific to the generated FFI bindings file only.

### C++ Formatting (catboost-master/)

- Tool: `ya style` (wraps clang-format with project config at `devtools/ya/handlers/style/config`)
- Indent: 4 spaces (no tabs)
- Block style: 1TBS (K&R) for `if`/`for`/`while`; either K&R or Allman for function definitions — must be consistent within a file
- No trailing spaces on lines
- No more than one statement per line
- Template keyword on its own line

## Import Organization

### Rust

**Order observed in source files:**
1. Crate-level `use` imports from `crate::` internal modules
2. `std::` standard library imports
3. External crate imports (e.g. `catboost_sys`, `approx`)

**Example (`model.rs`):**
```rust
use crate::error::{CatBoostError, CatBoostResult};
use crate::features::{ObjectsOrderFeatures, EmptyTextFeatures, EmptyEmbeddingFeatures};
use std::ffi::{CStr, CString};
use std::path::Path;
```

**`lib.rs` re-exports:** Public API items are re-exported from `lib.rs` using `pub use crate::module::Item`.

### C++ (catboost-master/)

Include order from most-specific to most-general:
1. Paired header file (in quotes)
2. Local directory files (in quotes)
3. Project superdirectory groups (angle brackets), ordered by nesting
4. Non-util external/contrib includes
5. `util` includes
6. C standard headers (`<cmath>`, `<cstdio>`)
7. System headers (only if unavoidable)

`#pragma once` is used for include guards (not `#ifndef` guards).

`using namespace` is prohibited inside header files.

## Error Handling

### Rust

**Pattern:** All fallible operations return `CatBoostResult<T>` (a type alias for `std::result::Result<T, CatBoostError>`).

**Error type:** `CatBoostError` is a struct with a `description: String` field, implementing `std::error::Error`, `fmt::Display`, and `fmt::Debug`.

**FFI boundary:** All CatBoost C API calls that return `bool` are checked with `CatBoostError::check_return_value(ret_val)?`. On failure the error message is fetched from the C library via `catboost_sys::GetErrorString()`.

```rust
// Correct pattern for FFI calls:
CatBoostError::check_return_value(unsafe {
    catboost_sys::SomeCFunction(args)
})?;
```

**Inline errors:** Errors constructed directly when validating Rust-side preconditions:
```rust
return Err(CatBoostError { description: "descriptive message".to_owned() });
```

**Panic use:** `unwrap()` is used in tests and in build scripts. In library code, use `?` propagation.

### C++ (catboost-master/)

- Errors signalled via exceptions (`ythrow` / `yexception`), never via return codes (except in C-interop or performance-critical sections).
- Run-time invariants checked with `Y_ASSERT()` macro (not `assert()`).
- Compile-time invariants use `static_assert`.

## Comments and Documentation

### Rust

**Doc comments:** `///` triple-slash doc comments on all public API items.

**Observed pattern:**
```rust
/// Load a model from a file
pub fn load<P: AsRef<Path>>(path: P) -> CatBoostResult<Self> { ... }

/// Check the return value from an CatBoost FFI call, and return the last error message on error.
/// Return values of true are treated as success, returns values of false are treated as errors.
pub fn check_return_value(ret_val: bool) -> CatBoostResult<()> { ... }
```

Inline `//` comments used for design rationale and non-obvious logic, e.g.:
```rust
/// `with_*_features` are convenience functions when you don't want to specify all types
///   of features when you don't need them.
/// They are necessary because Rust does not support default params.
```

### C++

- Comments in English with correct spelling and grammar.
- Doxygen-style comments encouraged.
- `TODO` comments must follow one of two formats:
  - `// TODO (username): fix me later`
  - `// TODO (ticket_number): fix me later`
- Dead code must be deleted, not commented out.

## CubeCL-Specific Rules (AGENTS.md)

When implementing CubeCL kernels:
- Kernels must use `generics-float` — do not hard-code float types.
- Read the CubeCL manual at `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md` before writing any kernel code.
- On any CubeCL build error, immediately load `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/cubecl_error_guideline.md` before attempting any fix.
- Blind fixes to CubeCL build errors without consulting the guideline are prohibited.

## Module Design

**Exports:** The public API is assembled in `lib.rs` via explicit `pub use` re-exports. Internal modules are declared with `mod` and kept private unless explicitly re-exported.

**No barrel files for types** — each module owns its types and only selected items are promoted to crate level.

**`catboost-sys` crate** acts as a thin unsafe FFI layer. It uses `bindgen` to generate bindings at build time and suppresses naming-convention lints for the generated file.

---

*Convention analysis: 2026-06-13*
