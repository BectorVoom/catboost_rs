# Technology Stack

**Analysis Date:** 2026-06-13

## Languages

**Primary:**
- Rust (edition 2018, minimum version 1.64) — Rust wrapper crates: `catboost` and `catboost-sys` in `catboost-master/catboost/rust-package/`
- C++ (C++20 standard) — Core CatBoost gradient boosting engine in `catboost-master/`
- C — Public model inference API surface (`catboost-master/catboost/libs/model_interface/c_api.h`)

**Secondary:**
- Python 3 — Python bindings (`catboost-master/catboost/python-package/`) and build tooling (`catboost-master/build/build_native.py`, `catboost-master/ci/`)
- Cython — Python extension module (`catboost-master/catboost/python-package/catboost/_catboost.pyx`)
- Java/Kotlin — JVM prediction package (`catboost-master/catboost/jvm-packages/`) and Spark integration
- C# — .NET bindings (`catboost-master/catboost/dotnet/`)
- JavaScript/TypeScript — Node.js bindings (`catboost-master/catboost/node-package/`)
- R — R language package (`catboost-master/catboost/R-package/`)
- Assembly (ASM) — Declared in CMake `project(CATBOOST LANGUAGES C CXX ASM)`

## Runtime

**Environment:**
- CPU (primary target) — Linux x86_64, Linux aarch64, Linux ppc64le, macOS x86_64, macOS arm64, Windows x86_64, Android (arm, arm64, x86, x86_64)
- GPU (optional) — NVIDIA CUDA; enabled via `HAVE_CUDA` CMake flag or `--have-cuda` build flag; GPU inference via `EnableGPUEvaluation` in `c_api.h`

**Package Manager:**
- Cargo (Rust crates) — Lockfile: `catboost-master/catboost/rust-package/Cargo.lock` (version 3, present)
- pip / Python packaging — `catboost-master/catboost/python-package/setup.py`
- npm — `catboost-master/catboost/node-package/package.json`
- Maven — `catboost-master/catboost/jvm-packages/`
- Conan — `catboost-master/conanfile.py` (C++ dependency management)

## Frameworks

**Core:**
- CatBoost C++ engine (vendored in `catboost-master/`) — gradient boosting library for training and inference
- YaTool / ya.make — Yandex internal build system (generated CMake files noted in `CMakeLists.txt` header)

**Testing:**
- Rust built-in `#[test]` framework — used in `catboost-sys/src/lib.rs` and `rust-package/src/model.rs`
- `approx` 0.5.1 — floating-point approximate equality assertions in Rust tests (`catboost-master/catboost/rust-package/Cargo.toml`)
- pytest — Python package tests (`catboost-master/catboost/pytest/`)

**Build/Dev:**
- CMake >= 3.15 — C++ build system (`catboost-master/CMakeLists.txt`)
- Ninja (preferred) — build generator referenced via `CATBOOST_MAX_LINK_JOBS` / `JOB_POOLS` in CMake
- `build_native.py` — Python wrapper around CMake invoked by Rust `build.rs` (`catboost-master/build/build_native.py`)
- SWIG — JVM/other bindings code generation (CMake includes `cmake/swig.cmake`)
- Cython — Python extension codegen (CMake includes `cmake/cython.cmake`)
- FlatBuffers — serialization (`cmake/fbs.cmake`)
- Protobuf — serialization (`cmake/protobuf.cmake`)

## Key Dependencies

**Critical (Rust):**
- `catboost-sys` 0.1 (local path) — raw FFI bindings crate, links against `libcatboostmodel` shared library (`catboost-master/catboost/rust-package/Cargo.toml`)
- `bindgen` ~0.59 (0.59.2 locked) — generates Rust FFI bindings from `c_api.h` at build time (`catboost-sys/Cargo.toml`)
- `approx` 0.5.1 — test-only floating-point comparison (`catboost/rust-package/Cargo.toml`)

**Infrastructure (Rust build):**
- `clang-sys` / `libclang` — required by bindgen to parse C headers; must be installed on host
- `python` (any Python 3 in PATH) — invoked by `build.rs` to run `build_native.py`

**C++ third-party (selected, under `catboost-master/contrib/`):**
- Various contrib libraries managed via CMake and Conan

## Configuration

**Environment:**
- `DEBUG` env var — read in `build.rs`; `"true"` selects a debug build of `catboostmodel`, otherwise Release
- `OUT_DIR` — standard Cargo env var used by `build.rs` for CMake build output directory

**Build:**
- `catboost-master/CMakeLists.txt` — root CMake config, dispatches to platform-specific `CMakeLists.<os>-<arch>[-cuda].txt` files
- `catboost-master/catboost/rust-package/catboost-sys/build.rs` — Rust build script; invokes `build_native.py`, runs bindgen, sets `rustc-link-search` and `rustc-link-lib`
- Feature flag `gpu` — activates `--have-cuda` in `build_native.py` and `EnableGPUEvaluation` usage in Rust (`catboost/rust-package/Cargo.toml`)

## Platform Requirements

**Development:**
- Python 3 in PATH (for `build_native.py`)
- CMake >= 3.15
- C++20 capable compiler (GCC, Clang, MSVC)
- libclang (for bindgen)
- CUDA toolkit (optional, only for `gpu` feature)
- Ninja (recommended for parallel linking)
- Rust >= 1.64, edition 2018

**Production:**
- `libcatboostmodel.so` / `libcatboostmodel.dylib` / `catboostmodel.dll` — shared library must be available at runtime (linked dynamically via `cargo:rustc-link-lib=dylib=catboostmodel`)
- Target platforms: Linux (x86_64, aarch64, ppc64le), macOS (x86_64, arm64), Windows (x86_64), Android

---

*Stack analysis: 2026-06-13*
