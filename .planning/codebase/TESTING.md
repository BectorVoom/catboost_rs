# Testing Patterns

**Analysis Date:** 2026-06-13

## Test Framework

**Runner:** Rust's built-in `cargo test`

**Assertion Library:**
- Standard `assert!`, `assert_eq!`, `#[should_panic]` macros
- `approx` crate (version `0.5.1`) for floating-point comparisons: `abs_diff_eq!` macro

**Run Commands:**
```bash
cd catboost-master/catboost/rust-package
cargo test                      # Run all tests (CPU only)
cargo test --features gpu       # Run tests including GPU-gated tests
```

**Prerequisites:** Tests require a pre-built model binary at `tmp/model.bin` relative to the rust-package directory, and CatBoost model files in `../pytest/data/models/`. These must be present before running tests.

## Test File Organization

### Mandatory Rule (AGENTS.md)

Tests **must** be in separate, dedicated files. Embedding `mod tests` inside production source files is **prohibited** for new code.

**Required patterns:**
```
src/
├── model.rs              # production code only
├── model_test.rs         # unit tests for model (separate file)
├── features.rs           # production code only
└── features_test.rs      # unit tests for features (separate file)

tests/
└── foo_tests.rs          # integration tests
```

### Current State (upstream catboost-master package)

The vendored upstream package at `catboost-master/catboost/rust-package/` predates the AGENTS.md rule and has inline test modules:

- `catboost-master/catboost/rust-package/src/model.rs` — contains `mod tests` at end of file (lines 267–746)
- `catboost-master/catboost/rust-package/catboost-sys/src/lib.rs` — contains `mod tests` at end of file (lines 7–75)

Do not replicate this pattern. All new code written in this project must use separate test files.

## Unit vs Integration Test Separation

**Unit tests:** Test individual functions and methods in isolation. Should live in `src/foo_test.rs` files (one per source module).

**Integration tests:** Test the full public API as an external consumer. Live in `tests/` directory.

**Distinction:** The existing tests in `model.rs` function as integration tests despite being embedded in the source — they exercise the full `Model` API end-to-end against real `.cbm` model files.

## Test Structure

**Suite organization pattern (from `model.rs`):**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    // Helper functions (non-test) are defined without #[test]:
    fn test_calc_prediction(on_gpu: bool) { ... }
    fn split_string_with_floats(features: &str, delim: char) -> Vec<f32> { ... }
    fn read_fast<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<Vec<u8>> { ... }

    // CPU variant — always runs:
    #[test]
    fn calc_prediction_on_cpu() {
        test_calc_prediction(false);
    }

    // GPU variant — only compiled with `gpu` feature flag:
    #[cfg(feature = "gpu")]
    #[test]
    fn calc_prediction_on_gpu() {
        test_calc_prediction(true);
    }

    // Expected-failure tests:
    #[test]
    #[should_panic]
    fn calc_prediction_object_size_mismatch() { ... }
}
```

**Key patterns:**
- CPU/GPU test pairs: logic in a shared `fn test_foo(on_gpu: bool)`, dispatched by two `#[test]` functions
- GPU tests are gated with `#[cfg(feature = "gpu")]`
- GPU tests that are known to fail use both `#[cfg(feature = "gpu")]` and `#[should_panic]`
- Helper utilities (`read_fast`, data-parsing helpers) are defined as plain functions inside `mod tests`

## Mocking

No mocking framework is used. Tests exercise real CatBoost model files loaded from disk. There are no mock/stub abstractions for the C FFI layer.

## Floating-Point Assertions

The `approx` crate is used for all floating-point comparisons. It is imported in `lib.rs` as:
```rust
#[cfg(test)]
#[macro_use]
extern crate approx;
```

Usage pattern:
```rust
assert!(std::iter::zip(expected_prediction, prediction)
    .all(|(l, r)| abs_diff_eq!(l, r, epsilon = 1.0e-6)))
```

Exact `assert_eq!` is used for integer model properties (tree count, feature counts).

## Test Data and Fixtures

**Model files required:**
- `tmp/model.bin` — basic regression/classification model (3 float features, 1 cat feature, 1000 trees), loaded by `load_model`, `load_model_buffer`, `get_model_stats` tests
- `../pytest/data/models/features_num__dataset_querywise.cbm`
- `../pytest/data/models/features_num_cat__dataset_adult.cbm`
- `../pytest/data/models/features_num_cat_text__dataset_rotten_tomatoes__binclass.cbm`
- `../pytest/data/models/features_num_cat_text_emb__dataset_rotten_tomatoes__binclass.cbm`

Model files are not included in source — they must be built/downloaded separately as part of the CatBoost project setup.

**Input data:** All test input vectors and expected outputs are hardcoded inline in test functions. No external test data files are read.

## Feature-Flag-Gated Tests

The `gpu` Cargo feature controls GPU evaluation tests:
```toml
[features]
gpu = ["catboost-sys/gpu"]
```

Tests annotated with `#[cfg(feature = "gpu")]` are compiled and run only when `--features gpu` is passed to `cargo test`. Most GPU tests are marked `#[should_panic]` because GPU evaluation is not supported in the test environment.

## Coverage

No coverage enforcement tooling is configured (no `.cargo/config.toml` with coverage flags, no CI coverage gates detected). Coverage is not tracked.

## CI Test Configuration

CI is defined in `catboost-master/.github/workflows/`. The relevant entry points are:

- `.github/workflows/check.yaml` — triggers on push to `master` and on pull requests; runs build and test checks across Linux, macOS, and Windows via `check_per_os.yaml`
- `.github/workflows/test.yaml` — manual dispatch workflow for running tests against pre-built artifacts

No Rust-specific `cargo test` invocations were found in the GitHub Actions workflow YAML files. The Rust package tests appear to be executed as part of the broader per-OS check pipeline rather than a dedicated Rust CI step.

## Test Types Summary

| Type | Scope | Location | Status |
|------|-------|----------|--------|
| FFI smoke tests | `catboost-sys` bindings round-trip | `catboost-sys/src/lib.rs` (inline `mod tests`) | Upstream pattern — do not replicate |
| Model API tests | Full `Model` public API end-to-end | `src/model.rs` (inline `mod tests`) | Upstream pattern — do not replicate |
| New unit tests | Individual functions/modules | `src/foo_test.rs` (separate file) | Required pattern per AGENTS.md |
| New integration tests | Public crate API | `tests/foo_tests.rs` | Required pattern per AGENTS.md |

---

*Testing analysis: 2026-06-13*
