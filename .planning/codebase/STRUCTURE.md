# Codebase Structure

**Analysis Date:** 2026-06-13

## Directory Layout

```
catboost_rs/
├── AGENTS.md                        # Agent rules: CubeCL protocol, test separation mandate
├── .planning/
│   └── codebase/                    # Codebase map documents (this directory)
└── catboost-master/                 # Upstream CatBoost source tree (vendored)
    ├── catboost/
    │   ├── rust-package/            # PRIMARY RUST INTEGRATION TARGET
    │   │   ├── Cargo.toml           # catboost crate definition (v0.1.0, edition 2018)
    │   │   ├── src/
    │   │   │   ├── lib.rs           # Crate root: public re-exports
    │   │   │   ├── model.rs         # Model struct: load, predict, metadata, GPU
    │   │   │   ├── features.rs      # ObjectsOrderFeatures + Empty* types
    │   │   │   └── error.rs         # CatBoostError, CatBoostResult
    │   │   └── catboost-sys/        # FFI sys-crate (unsafe layer)
    │   │       ├── Cargo.toml       # catboost-sys crate definition
    │   │       ├── Cargo.lock       # Locked deps for sys crate
    │   │       ├── build.rs         # Build script: compile C++ lib + run bindgen
    │   │       ├── wrapper.h        # bindgen entry: #include <c_api.h>
    │   │       └── src/
    │   │           └── lib.rs       # include!(bindings.rs) — generated unsafe FFI
    │   ├── libs/
    │   │   ├── model_interface/     # C API exposed to Rust
    │   │   │   ├── c_api.h          # Public extern "C" header (v1.2.10)
    │   │   │   ├── c_api.cpp        # Implementation
    │   │   │   ├── model_calcer_wrapper.h  # C++ wrapper internals
    │   │   │   ├── wrapped_calcer.h        # Header-only C++ wrapper
    │   │   │   └── CMakeLists*.txt  # Platform-specific build files
    │   │   ├── model/               # Model serialization / tree evaluation (C++)
    │   │   ├── train_lib/           # CPU training engine (C++)
    │   │   ├── train_interface/     # Training API surface (C++)
    │   │   ├── data/                # Dataset handling (C++)
    │   │   ├── cat_feature/         # Categorical feature hashing (C++)
    │   │   ├── helpers/             # Shared utilities (C++)
    │   │   ├── metrics/             # Evaluation metrics (C++)
    │   │   ├── fstr/                # Feature importance (C++)
    │   │   └── ...                  # Other C++ support libraries
    │   ├── cuda/                    # CUDA GPU training kernels (C++/CUDA)
    │   │   ├── cuda_lib/            # CUDA device management
    │   │   ├── cuda_util/           # GPU utility functions
    │   │   ├── methods/             # GPU training algorithm implementations
    │   │   ├── gpu_data/            # GPU data structures
    │   │   ├── targets/             # GPU loss/target implementations
    │   │   └── train_lib/           # GPU training entry points
    │   ├── tutorials/
    │   │   └── apply_model/rust/    # Tutorial: Rust inference example
    │   │       ├── Cargo.toml
    │   │       └── src/main.rs      # End-to-end usage example (Adult dataset)
    │   ├── app/                     # CatBoost CLI application (C++)
    │   ├── python-package/          # Python bindings (pyproject.toml)
    │   ├── jvm-packages/            # JVM/Scala bindings
    │   ├── R-package/               # R bindings
    │   ├── dotnet/                  # .NET bindings
    │   ├── node-package/            # Node.js bindings
    │   └── docs/                    # Documentation source
    ├── bindings/
    │   └── swiglib/                 # SWIG-based binding helpers
    ├── build/                       # Build system scripts and config
    │   ├── build_native.py          # Main C++ build orchestration script
    │   ├── platform/                # Platform detection
    │   ├── scripts/                 # Build helper scripts
    │   └── toolchains/              # Compiler toolchain configs
    ├── cmake/                       # CMake modules
    ├── ci/                          # CI toolchain configs (conan, cmake, toolchains)
    ├── contrib/                     # Vendored third-party C++ dependencies
    │   ├── libs/                    # Third-party libraries
    │   └── tools/                   # Third-party build tools
    ├── library/                     # Shared Yandex/internal C++ utilities
    ├── util/                        # Low-level C++ utilities
    └── tools/                       # Build/lint/dev tooling
```

## Directory Purposes

**`catboost-master/catboost/rust-package/`:**
- Purpose: The Rust crate that wraps CatBoost inference — this is the primary code area for Rust development
- Contains: Two crates: `catboost` (safe API) and `catboost-sys` (FFI)
- Key files: `src/model.rs`, `src/features.rs`, `catboost-sys/build.rs`

**`catboost-master/catboost/libs/model_interface/`:**
- Purpose: The C API boundary between C++ and all language bindings
- Contains: `c_api.h` (public contract), `c_api.cpp` (implementation)
- Key files: `c_api.h` — do not modify; this is the upstream contract

**`catboost-master/catboost/cuda/`:**
- Purpose: CUDA-based GPU training (not GPU inference — inference is toggled via `EnableGPUEvaluation` in the C API)
- Generated: No (source code)
- Build-activated: only when `catboost-sys` is built with `--features gpu`

**`catboost-master/build/`:**
- Purpose: Cross-platform C++ build orchestration
- Key file: `build_native.py` — invoked by `catboost-sys/build.rs` to compile `libcatboostmodel`

**`catboost-master/contrib/`:**
- Purpose: Vendored C++ dependencies (Eigen, protobuf, etc.)
- Generated: No
- Committed: Yes (vendored)

## Key File Locations

**Entry Points:**
- `catboost-master/catboost/rust-package/src/lib.rs`: Crate root, all public exports
- `catboost-master/catboost/tutorials/apply_model/rust/src/main.rs`: Usage example (reference for correct API calls)

**Build System:**
- `catboost-master/catboost/rust-package/catboost-sys/build.rs`: Build orchestration for the sys crate
- `catboost-master/build/build_native.py`: C++ build entry point called by `build.rs`

**Public Rust API:**
- `catboost-master/catboost/rust-package/src/model.rs`: `Model` struct with `load`, `load_buffer`, `predict`, `calc_model_prediction`, `enable_gpu_evaluation`, metadata getters
- `catboost-master/catboost/rust-package/src/features.rs`: `ObjectsOrderFeatures<…>`, `EmptyFloatFeatures`, `EmptyCatFeatures`, `EmptyTextFeatures`, `EmptyEmbeddingFeatures`
- `catboost-master/catboost/rust-package/src/error.rs`: `CatBoostError`, `CatBoostResult<T>`

**C API Contract:**
- `catboost-master/catboost/libs/model_interface/c_api.h`: Full C API reference — 692 lines, version 1.2.10

**FFI Glue:**
- `catboost-master/catboost/rust-package/catboost-sys/src/lib.rs`: `include!(bindings.rs)` + raw-usage tests
- `catboost-master/catboost/rust-package/catboost-sys/wrapper.h`: Single-line bridge to `c_api.h`

## Naming Conventions

**Files:**
- Rust source: `snake_case.rs` (e.g., `model.rs`, `features.rs`, `error.rs`)
- C++ headers: `snake_case.h` (e.g., `c_api.h`, `model_calcer_wrapper.h`)
- Build scripts: `snake_case.py` or `CMakeLists.txt`

**Directories:**
- Rust crates: `kebab-case` (e.g., `rust-package`, `catboost-sys`)
- C++ library subdirs: `snake_case` (e.g., `model_interface`, `train_lib`)

**Rust types:**
- Structs: `PascalCase` (e.g., `Model`, `CatBoostError`, `ObjectsOrderFeatures`)
- Type aliases: `PascalCase` (e.g., `CatBoostResult`)
- Generic params: `T` prefix + descriptor (e.g., `TFloatFeatures`, `TObjectCatFeatures`)

## Module Boundaries

**What lives in `catboost-master/catboost/rust-package/`** (Rust, safe to modify):
- All Rust source code for the `catboost` and `catboost-sys` crates
- Build script `build.rs`
- `wrapper.h` (trivial bridge header)

**What lives in `catboost-master/catboost/libs/model_interface/`** (C++ boundary, treat as read-only):
- The `c_api.h` contract that bindgen reads
- Do not modify unless extending the upstream C API

**What lives in `catboost-master/catboost/libs/` and `catboost-master/catboost/cuda/`** (C++ core, read-only):
- Training and inference implementation
- CUDA GPU training kernels
- Modify only if implementing deep CatBoost core changes

## Where to Add New Code

**New Rust feature/method on `Model`:**
- Add to `catboost-master/catboost/rust-package/src/model.rs`
- If it calls new C API functions, ensure they are in `c_api.h` and will appear in generated `bindings.rs`

**New feature type or feature container:**
- Add to `catboost-master/catboost/rust-package/src/features.rs`
- Export from `catboost-master/catboost/rust-package/src/lib.rs`

**New CubeCL GPU kernel (per AGENTS.md mandate):**
- Read CubeCL manual at `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md` first
- Create kernel source in a dedicated file, e.g. `catboost-master/catboost/rust-package/src/gpu_kernels.rs`
- Use generic-float types as required by AGENTS.md
- Tests must go in a separate file (e.g., `tests/gpu_kernels_tests.rs`), NOT inline

**Tests:**
- Per AGENTS.md: all tests go in `tests/` directory or dedicated `src/foo_test.rs` files
- Do NOT add `mod tests { }` blocks inside production source files
- Integration tests: `catboost-master/catboost/rust-package/tests/`
- Unit tests for a module `src/foo.rs`: create `src/foo_test.rs` or `tests/foo_tests.rs`

**Shared utilities:**
- Add to a new `src/util.rs` or `src/helpers.rs` under `catboost-master/catboost/rust-package/src/`
- Export from `lib.rs` only if part of the public API

## Special Directories

**`catboost-master/`:**
- Purpose: Vendored upstream CatBoost repository
- Generated: No (vendored snapshot)
- Committed: Yes

**`catboost-master/catboost/rust-package/catboost-sys/` (OUT_DIR at build time):**
- Purpose: `bindings.rs` is generated into Cargo's `$OUT_DIR` (not committed)
- Generated: Yes (`bindings.rs` only)
- Committed: No (`bindings.rs` is not in the source tree)

**`.planning/codebase/`:**
- Purpose: Codebase map documents for GSD agent commands
- Generated: Yes (by mapper agents)
- Committed: Recommended (source of truth for architecture docs)

---

*Structure analysis: 2026-06-13*
