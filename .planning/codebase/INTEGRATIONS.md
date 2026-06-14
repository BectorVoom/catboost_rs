# External Integrations

**Analysis Date:** 2026-06-13

## C++ ↔ Rust FFI Boundary

**Primary FFI layer: `catboost-sys`**
- Location: `catboost-master/catboost/rust-package/catboost-sys/`
- Mechanism: `bindgen` 0.59 generates Rust FFI types from `wrapper.h` at build time
  - `wrapper.h` contains a single `#include <c_api.h>` pointing to `catboost-master/catboost/libs/model_interface/c_api.h`
  - Generated bindings written to `$OUT_DIR/bindings.rs` and included via `include!` macro in `catboost-sys/src/lib.rs`
- Link target: `libcatboostmodel` shared library, dynamically linked
  - Search path set by `build.rs`: `$OUT_DIR/catboost/libs/model_interface/`
  - Link directive: `cargo:rustc-link-lib=dylib=catboostmodel`
- Build trigger: `build.rs` invokes Python `build_native.py` with `--targets catboostmodel` to compile the C++ library before bindgen runs

**C API surface (`c_api.h`):**
- Version: `CATBOOST_APPLIER_MAJOR 1`, `CATBOOST_APPLIER_MINOR 2`, `CATBOOST_APPLIER_FIX 10`
- Key handle types: `ModelCalcerHandle` (opaque `void*`), `DataWrapperHandle`, `DataProviderHandle`
- Key functions exposed to Rust:
  - `ModelCalcerCreate()` / `ModelCalcerDelete()`
  - `LoadFullModelFromFile()` / `LoadFullModelFromBuffer()`
  - `CalcModelPredictionWithHashedCatFeaturesAndTextAndEmbeddingFeatures()`
  - `GetStringCatFeatureHash()`, `GetFloatFeaturesCount()`, `GetCatFeaturesCount()`, `GetTextFeaturesCount()`, `GetEmbeddingFeaturesCount()`, `GetTreeCount()`, `GetDimensionsCount()`
  - `EnableGPUEvaluation()` — GPU inference trigger
  - `GetErrorString()` — thread-local error retrieval
- ABI: `extern "C"` with `CATBOOST_API` visibility macro (dllexport/dllimport on Windows, no-op on POSIX)

**Safe Rust wrapper (`catboost` crate):**
- Location: `catboost-master/catboost/rust-package/src/`
- `model.rs` — `Model` struct wrapping raw `*mut ModelCalcerHandle`; implements `Send + Sync + Drop`
- `features.rs` — typed feature containers (`ObjectsOrderFeatures`, empty-type sentinels for optional feature types)
- `error.rs` — `CatBoostError` / `CatBoostResult` error type wrapping `GetErrorString()`
- All `catboost_sys::*` calls are inside `unsafe {}` blocks in `model.rs`

## GPU Integration

**CUDA via CatBoost C++ engine:**
- Enabled by the `gpu` Cargo feature flag in `catboost-master/catboost/rust-package/Cargo.toml`
- When `gpu` feature is active, `build.rs` passes `--have-cuda` to `build_native.py`, which selects a CUDA-enabled CMake config (e.g., `CMakeLists.linux-x86_64-cuda.txt`)
- Rust API entry point: `Model::enable_gpu_evaluation()` → `catboost_sys::EnableGPUEvaluation(handle, device_id)` where `device_id=0`
- GPU evaluation is optional and CPU remains the default path; `EnableGPUEvaluation` must be called explicitly before `predict`

**CubeCL (referenced in project rules, not yet in source):**
- `AGENTS.md` mandates CubeCL usage for GPU computation kernels
- CubeCL manual located at: `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md`
- CubeCL error guideline: `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/cubecl_error_guideline.md`
- No CubeCL `Cargo.toml` dependency or `.rs` source files detected in the current tree — integration is planned/in-progress
- When added, kernels must use `generics-float` as required by the project rules in `AGENTS.md`

## Python Bindings

**Cython extension `_catboost`:**
- Source: `catboost-master/catboost/python-package/catboost/_catboost.pyx`
- Packaged via `catboost-master/catboost/python-package/setup.py`
- Wheel built using `catboost-master/catboost/python-package/mk_wheel.py`
- High-level Python API in `catboost-master/catboost/python-package/catboost/core.py`, `__init__.py`
- Build target `_catboost` in `build_native.py` produces a shared `.so`/`.dylib` for the Python extension

## JVM / Java Bindings

**JNI prediction package:**
- Source: `catboost-master/catboost/jvm-packages/catboost4j-prediction/`
- Java classes: `CatBoostModel.java`, `CatBoostPredictions.java`, `CatBoostJNI.java` — JNI bridge to native
- Native target: `catboost4j-prediction` shared library built by `build_native.py`
- Maven build tooling: `catboost-master/catboost/jvm-packages/tools/build_native_for_maven.py`

**Apache Spark integration:**
- Source: `catboost-master/catboost/spark/catboost4j-spark/`
- Native target: `catboost4j-spark-impl` shared library
- Multi-project Maven build via `run_mvn_for_all_projects.py`

## .NET Bindings

- Source: `catboost-master/catboost/dotnet/`
- Projects: `CatBoostNet`, `CatBoostMlNet`, `CatBoostNetTests`, `HeartDiseaseDemo`, `LibraryBuilder`, `LibraryImportAndUseTest`
- Solution file: `catboost-master/catboost/dotnet/dotnet.sln`
- Paket dependency manager: `catboost-master/catboost/dotnet/paket.dependencies`

## Node.js Bindings

- Source: `catboost-master/catboost/node-package/`
- `binding.gyp` — native addon build config
- `package.json` / `package-lock.json` — npm package
- `bindings/` — binding source files
- Built via node-gyp

## R Bindings

- Source: `catboost-master/catboost/R-package/`
- Native target: `catboostr` shared library built by `build_native.py`
- Package tooling: `mk_package.py`, `test.py`

## Build Integration: `build_native.py`

- Location: `catboost-master/build/build_native.py`
- Role: orchestrates CMake builds for all native targets across all language bindings
- Invoked by: Rust `catboost-sys/build.rs` at Cargo build time
- Key parameters:
  - `--targets` — selects which native library to build (e.g., `catboostmodel`, `_catboost`, `catboostr`)
  - `--build-root-dir` — output directory (maps to `$OUT_DIR` from Cargo)
  - `--build-type=Debug|Release`
  - `--have-cuda` — enables CUDA compilation path

## CI/CD Pipelines

**Platform:** GitHub Actions
- Workflows located in `catboost-master/.github/workflows/`
- Key workflow files:
  - `check.yaml` — triggers `check_per_os.yaml` for linux, macos, windows on push to master and PRs; uses CatBoost version `1.2.10`
  - `test.yaml` / `test_per_os.yaml` — per-OS test runs
  - `build_per_platform.yaml` / `build_per_os.yaml` — platform matrix builds
  - `release_build_and_check.yaml` / `release_publish_node_package.yaml` — release automation
  - `add_mac_os_extra_env_to_cache.yaml` — macOS cache warmup

**Build helpers:**
- `catboost-master/ci/build_all.py` — CI-level orchestration
- `catboost-master/ci/prepare_release_artifacts.py`, `extract_release_changelog.py` — release pipeline
- `catboost-master/ci/webdav_upload.py` — artifact upload

## External Model Format

**`.cbm` files (CatBoost binary model):**
- Binary serialization format used to persist trained models
- Loaded at runtime via `LoadFullModelFromFile()` or `LoadFullModelFromBuffer()` from C API
- Test models referenced: `tmp/model.bin`, `../pytest/data/models/*.cbm`
- Format defined internally by the C++ engine; no external schema dependency

## Serialization Libraries (C++ internal)

- **FlatBuffers** — used internally by C++ engine (CMake includes `cmake/fbs.cmake`)
- **Protobuf** — used internally by C++ engine (CMake includes `cmake/protobuf.cmake`)
- Managed as vendored contrib under `catboost-master/contrib/`

---

*Integration audit: 2026-06-13*
