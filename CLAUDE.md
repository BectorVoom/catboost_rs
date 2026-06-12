<!-- GSD:project-start source:PROJECT.md -->

## Project

**catboost-rs**

A full Rust rewrite of the CatBoost gradient boosting library, targeting complete feature parity with the original C++ implementation. It exposes first-class APIs in both Rust (using Rust-native patterns like the Builder pattern) and Python (via PyO3/maturin), where the Python surface is **both** scikit-learn compatible *and* CatBoost-native (Pool, parameter names) for drop-in migration. GPU acceleration is provided through CubeCL with backends switchable at compile time via Cargo features.

It is for two audiences: Rust developers who want to embed a memory-efficient gradient booster directly, and Python ML practitioners who want a drop-in replacement for CatBoost in existing scikit-learn or CatBoost workflows.

**Core Value:** A memory-efficient, Rust-native CatBoost implementation that achieves verifiable feature parity with the original (oracle-tested to within 10⁻⁵), embeddable directly in Rust and droppable into both scikit-learn and existing CatBoost Python pipelines.

### Constraints

- **Tech stack**: Rust (latest stable), CubeCL for GPU kernels, PyO3 + maturin for Python bindings
- **Python version**: >= 3.12
- **Backend selection**: Cargo features only — `cuda`, `rocm`, `wgpu`, `cpu`; no runtime switching
- **Dependencies**: always use the latest crate versions
- **Error handling**: `thiserror` (library) + `anyhow` (application); `unwrap()` strictly prohibited in production
- **Memory**: high memory efficiency is a first-class design constraint — minimize allocations, prefer zero-copy where possible
- **Workspace**: modular Cargo workspace from day one — clear crate separation of responsibilities
- **API style**: Rust side uses the Builder pattern; Python side is both scikit-learn compatible and CatBoost-native
- **Parity bar**: oracle error tolerance ≤ 10⁻⁵ against original CatBoost outputs
- **Testing**: source/test code strictly separated; GPU tests on `rocm` only
- **No C API**: PyO3 bindings only; no C FFI or CAPI layer

<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->

## Technology Stack

## Languages

- Rust (edition 2018, minimum version 1.64) — Rust wrapper crates: `catboost` and `catboost-sys` in `catboost-master/catboost/rust-package/`
- C++ (C++20 standard) — Core CatBoost gradient boosting engine in `catboost-master/`
- C — Public model inference API surface (`catboost-master/catboost/libs/model_interface/c_api.h`)
- Python 3 — Python bindings (`catboost-master/catboost/python-package/`) and build tooling (`catboost-master/build/build_native.py`, `catboost-master/ci/`)
- Cython — Python extension module (`catboost-master/catboost/python-package/catboost/_catboost.pyx`)
- Java/Kotlin — JVM prediction package (`catboost-master/catboost/jvm-packages/`) and Spark integration
- C# — .NET bindings (`catboost-master/catboost/dotnet/`)
- JavaScript/TypeScript — Node.js bindings (`catboost-master/catboost/node-package/`)
- R — R language package (`catboost-master/catboost/R-package/`)
- Assembly (ASM) — Declared in CMake `project(CATBOOST LANGUAGES C CXX ASM)`

## Runtime

- CPU (primary target) — Linux x86_64, Linux aarch64, Linux ppc64le, macOS x86_64, macOS arm64, Windows x86_64, Android (arm, arm64, x86, x86_64)
- GPU (optional) — NVIDIA CUDA; enabled via `HAVE_CUDA` CMake flag or `--have-cuda` build flag; GPU inference via `EnableGPUEvaluation` in `c_api.h`
- Cargo (Rust crates) — Lockfile: `catboost-master/catboost/rust-package/Cargo.lock` (version 3, present)
- pip / Python packaging — `catboost-master/catboost/python-package/setup.py`
- npm — `catboost-master/catboost/node-package/package.json`
- Maven — `catboost-master/catboost/jvm-packages/`
- Conan — `catboost-master/conanfile.py` (C++ dependency management)

## Frameworks

- CatBoost C++ engine (vendored in `catboost-master/`) — gradient boosting library for training and inference
- YaTool / ya.make — Yandex internal build system (generated CMake files noted in `CMakeLists.txt` header)
- Rust built-in `#[test]` framework — used in `catboost-sys/src/lib.rs` and `rust-package/src/model.rs`
- `approx` 0.5.1 — floating-point approximate equality assertions in Rust tests (`catboost-master/catboost/rust-package/Cargo.toml`)
- pytest — Python package tests (`catboost-master/catboost/pytest/`)
- CMake >= 3.15 — C++ build system (`catboost-master/CMakeLists.txt`)
- Ninja (preferred) — build generator referenced via `CATBOOST_MAX_LINK_JOBS` / `JOB_POOLS` in CMake
- `build_native.py` — Python wrapper around CMake invoked by Rust `build.rs` (`catboost-master/build/build_native.py`)
- SWIG — JVM/other bindings code generation (CMake includes `cmake/swig.cmake`)
- Cython — Python extension codegen (CMake includes `cmake/cython.cmake`)
- FlatBuffers — serialization (`cmake/fbs.cmake`)
- Protobuf — serialization (`cmake/protobuf.cmake`)

## Key Dependencies

- `catboost-sys` 0.1 (local path) — raw FFI bindings crate, links against `libcatboostmodel` shared library (`catboost-master/catboost/rust-package/Cargo.toml`)
- `bindgen` ~0.59 (0.59.2 locked) — generates Rust FFI bindings from `c_api.h` at build time (`catboost-sys/Cargo.toml`)
- `approx` 0.5.1 — test-only floating-point comparison (`catboost/rust-package/Cargo.toml`)
- `clang-sys` / `libclang` — required by bindgen to parse C headers; must be installed on host
- `python` (any Python 3 in PATH) — invoked by `build.rs` to run `build_native.py`
- Various contrib libraries managed via CMake and Conan

## Configuration

- `DEBUG` env var — read in `build.rs`; `"true"` selects a debug build of `catboostmodel`, otherwise Release
- `OUT_DIR` — standard Cargo env var used by `build.rs` for CMake build output directory
- `catboost-master/CMakeLists.txt` — root CMake config, dispatches to platform-specific `CMakeLists.<os>-<arch>[-cuda].txt` files
- `catboost-master/catboost/rust-package/catboost-sys/build.rs` — Rust build script; invokes `build_native.py`, runs bindgen, sets `rustc-link-search` and `rustc-link-lib`
- Feature flag `gpu` — activates `--have-cuda` in `build_native.py` and `EnableGPUEvaluation` usage in Rust (`catboost/rust-package/Cargo.toml`)

## Platform Requirements

- Python 3 in PATH (for `build_native.py`)
- CMake >= 3.15
- C++20 capable compiler (GCC, Clang, MSVC)
- libclang (for bindgen)
- CUDA toolkit (optional, only for `gpu` feature)
- Ninja (recommended for parallel linking)
- Rust >= 1.64, edition 2018
- `libcatboostmodel.so` / `libcatboostmodel.dylib` / `catboostmodel.dll` — shared library must be available at runtime (linked dynamically via `cargo:rustc-link-lib=dylib=catboostmodel`)
- Target platforms: Linux (x86_64, aarch64, ppc64le), macOS (x86_64, arm64), Windows (x86_64), Android

<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->

## Conventions

## Source/Test Separation — Mandatory Rule

- Embedding `mod tests` at the bottom of a production source file is **strictly prohibited**.
- All tests (unit and integration) must reside in separate, dedicated files.
- Permitted structures:
- Production source files must contain only implementation logic — no `#[cfg(test)]` blocks embedded in them.

## Naming Patterns

### Rust Code (this project)

- Snake_case for all Rust source files: `model.rs`, `features.rs`, `error.rs`
- Test files follow source name with `_test` or `_tests` suffix: `model_test.rs`, `foo_tests.rs`
- C++ file names use lowercase only (no capital letters), extensions `.cpp` and `.h`
- `PascalCase` for all types, structs, enums: `CatBoostError`, `ObjectsOrderFeatures`, `EmptyFloatFeatures`
- Generic type parameters use `T` prefix in PascalCase: `TFeature`, `TObjectFeatures`, `TFloatFeatures`, `TCatFeatures`
- `snake_case` for all functions and methods: `load`, `load_buffer`, `check_return_value`, `get_float_features_count`
- `snake_case` for all local variables and function parameters
- `SCREAMING_SNAKE_CASE` for constants
- `snake_case` for module names: `mod error`, `mod features`, `mod model`

### C++ Code (catboost-master/)

## Code Style

### Rust Formatting

- 4-space indentation
- Trailing commas in multi-line struct literals and function calls
- Generic bounds on their own lines when spanning multiple type parameters (see `predict()` in `model.rs`)
- Closure chains use `.collect::<Vec<_>>()` turbofish form
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

### C++ (catboost-master/)

## Error Handling

### Rust

### C++ (catboost-master/)

- Errors signalled via exceptions (`ythrow` / `yexception`), never via return codes (except in C-interop or performance-critical sections).
- Run-time invariants checked with `Y_ASSERT()` macro (not `assert()`).
- Compile-time invariants use `static_assert`.

## Comments and Documentation

### Rust

### C++

- Comments in English with correct spelling and grammar.
- Doxygen-style comments encouraged.
- `TODO` comments must follow one of two formats:
- Dead code must be deleted, not commented out.

## CubeCL-Specific Rules (AGENTS.md)

- Kernels must use `generics-float` — do not hard-code float types.
- Read the CubeCL manual at `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md` before writing any kernel code.
- On any CubeCL build error, immediately load `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/cubecl_error_guideline.md` before attempting any fix.
- Blind fixes to CubeCL build errors without consulting the guideline are prohibited.

## Module Design

<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->

## Architecture

## System Overview

```text

```

## Component Responsibilities

| Component | Responsibility | File(s) |
|-----------|----------------|---------|
| `catboost` crate | Safe, idiomatic Rust API over CatBoost | `catboost-master/catboost/rust-package/src/` |
| `Model` struct | Owns the `ModelCalcerHandle*`, load/predict/metadata | `…/src/model.rs` |
| `ObjectsOrderFeatures<…>` | Type-safe, generic feature container (builder pattern) | `…/src/features.rs` |
| Empty* types | Zero-cost placeholders for unused feature types | `…/src/features.rs` |
| `CatBoostError` / `CatBoostResult` | Error type wrapping C string from `GetErrorString()` | `…/src/error.rs` |
| `catboost-sys` crate | Raw unsafe bindings generated by bindgen at compile time | `…/catboost-sys/src/lib.rs` |
| `build.rs` (catboost-sys) | Compiles `libcatboostmodel` via `build_native.py`, runs bindgen | `…/catboost-sys/build.rs` |
| `wrapper.h` | Single-header bridge that includes `c_api.h` for bindgen | `…/catboost-sys/wrapper.h` |
| `c_api.h` / `c_api.cpp` | Official C API for model inference | `…/libs/model_interface/c_api.h` |
| CUDA training layer | GPU-accelerated tree training (not inference) | `…/catboost/cuda/` |

## Pattern Overview

- `catboost-sys` owns all `unsafe` code and raw C types; never exposed publicly
- `catboost` crate provides a fully-safe public API with idiomatic Rust generics
- `Model` implements `Send + Sync` explicitly, allowing sharing across threads
- `Model` implements `Drop` to call `ModelCalcerDelete`, ensuring no handle leaks
- Feature types use a builder/typestate pattern: `ObjectsOrderFeatures::new().with_float_features(…).with_cat_features(…)`
- Empty placeholder types (`EmptyFloatFeatures`, etc.) implement `AsRef<[…]>` returning `&[]`, enabling zero-overhead omission of unused feature kinds
- Errors are pulled from CatBoost's thread-local error string via `GetErrorString()` immediately after a failed call

## Layers

- Purpose: Ergonomic, memory-safe surface for Rust callers
- Location: `catboost-master/catboost/rust-package/src/`
- Contains: `Model`, `ObjectsOrderFeatures`, feature empty-types, `CatBoostError`
- Depends on: `catboost-sys`
- Used by: application code, tutorials
- Purpose: Auto-generated raw C bindings; holds all `unsafe` declarations
- Location: `catboost-master/catboost/rust-package/catboost-sys/`
- Contains: `bindings.rs` (generated into `OUT_DIR` at build time), `wrapper.h`
- Depends on: `libcatboostmodel` shared library (compiled by `build.rs`)
- Used by: `catboost` crate only
- Purpose: Stable extern "C" interface exposing inference from the C++ core
- Location: `catboost-master/catboost/libs/model_interface/`
- Contains: `c_api.h`, `c_api.cpp`, `model_calcer_wrapper.h`, `wrapped_calcer.h`
- Depends on: C++ model/tree-evaluation core (`catboost/libs/model/`)
- Used by: `catboost-sys` (via dynamic link)
- Purpose: Model storage, tree evaluation, training algorithms
- Location: `catboost-master/catboost/libs/`, `catboost-master/catboost/cuda/`
- Contains: model serialization (`libs/model/`), training (`libs/train_lib/`, `libs/train_interface/`), CUDA kernels (`cuda/`)
- Used by: `c_api.cpp`

## Data Flow

### Model Inference (Primary Path)

### Model Loading

### Build-time Binding Generation

## GPU Acceleration

- CatBoost's CUDA training layer is implemented in `catboost-master/catboost/cuda/`
- Subdirectories: `cuda_lib/`, `cuda_util/`, `methods/`, `gpu_data/`, `targets/`, `train_lib/`
- Activated during the C++ build when `--have-cuda` is passed to `build_native.py`
- Enabled in Rust via `catboost` feature `gpu = ["catboost-sys/gpu"]`
- `EnableGPUEvaluation(modelHandle, deviceId)` is exposed in `c_api.h` line 124
- The C API enum `ECatBoostApiFormulaEvaluatorType` has `CBA_FET_GPU = 1`
- Called from Rust via `Model::enable_gpu_evaluation()` in `model.rs:256-258`
- This is CatBoost's own CUDA inference path — **not** CubeCL
- `AGENTS.md` mandates use of the `cubecl` crate for GPU computation kernels in this project
- CubeCL manual is at `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md`
- CubeCL kernels require generic-float types
- No CubeCL source code is present yet in the repository; it is a planned/future integration

## Key Abstractions

- Purpose: Opaque pointer to the C++ model evaluator object
- Exposed in: `c_api.h` as `typedef void ModelCalcerHandle`
- Wrapped by: `Model.handle: *mut catboost_sys::ModelCalcerHandle` in `model.rs`
- Purpose: Type-safe, zero-cost, batch feature container
- Location: `catboost-master/catboost/rust-package/src/features.rs`
- Pattern: Builder via `with_float_features()`, `with_cat_features()`, `with_text_features()`, `with_embedding_features()`
- Purpose: Alias for `Result<T, CatBoostError>` throughout the safe API
- Location: `catboost-master/catboost/rust-package/src/error.rs`

## Error Handling

- All FFI calls return `bool`; `false` means error
- `CatBoostError::check_return_value(bool) -> CatBoostResult<()>` is the single chokepoint
- Error message is fetched from CatBoost's thread-local string via `catboost_sys::GetErrorString()`
- The Rust `CatBoostError` implements `std::error::Error` and `Display`

## Architectural Constraints

- **Threading:** `Model` is `Send + Sync` (manually asserted). `GetErrorString()` is thread-local in the C++ library, so error fetching is safe across threads.
- **Global state:** None in the Rust layer. The C++ library may have process-global GPU device state when `EnableGPUEvaluation` is called.
- **Unsafe boundary:** All `unsafe` is contained in `catboost-sys/src/lib.rs` (generated) and the `unsafe {}` blocks inside `model.rs`. The public API of `catboost` crate is fully safe.
- **Build dependency:** Requires Python and either a C++ compiler toolchain or a pre-built `libcatboostmodel`. CUDA builds additionally require NVCC.
- **Circular imports:** None detected.

## Anti-Patterns

### Tests embedded in production source files

<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->

## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->

## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:

- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->

## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
