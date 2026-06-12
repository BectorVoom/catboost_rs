# Stack Research

**Domain:** Gradient-boosting ML library (Rust core + multi-backend GPU + Python bindings)
**Researched:** 2026-06-13
**Confidence:** HIGH (all versions verified live against crates.io 2026-06-13; CubeCL backend semantics verified from the vendored CubeCL manual + crates.io feature graph)

> **Constraint reminder (from PROJECT.md):** latest crate versions, `thiserror`(lib)+`anyhow`(app), no `unwrap()` in production, Cargo-feature backend selection (`cuda`/`rocm`/`wgpu`/`cpu`) with **no runtime dispatch**, Python ≥ 3.12, per-backend wheels, oracle parity ≤ 1e-5, modular workspace, strict source/test separation, GPU tests on `rocm` only.

---

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| **Rust** | latest stable (≥ 1.85, edition 2024) | Implementation language | Mandated. Edition 2024 is stable and the default for new workspaces; gives `let`-else, async closures, improved `gen`/lints. |
| **cubecl** | **0.10.0** | GPU compute kernels across CUDA/ROCm/WGPU/CPU from one Rust codebase | Mandated. Single `#[cube]` kernel source compiles to every backend; backend chosen by Cargo feature → satisfies the "no runtime dispatch" rule via compile-time generic `R: Runtime`. See **CubeCL Backend Strategy** below. |
| **pyo3** | **0.28.3** (NOT 0.29) | Rust ⇄ Python boundary | Mandated. **Pinned to 0.28.x deliberately** — `rust-numpy` 0.28 requires `pyo3 ^0.28`; 0.29 is incompatible with the current rust-numpy release. Supports `abi3-py312`. |
| **maturin** | **1.14.0** | Build/package per-backend wheels | Mandated. First-class PyO3 support; `--features rocm` etc. produces backend-specific wheels (`catboost-rs-rocm`). Use `[tool.maturin]` in `pyproject.toml`. |
| **ndarray** | **0.17.2** | CPU numeric/array core for training data, histograms, gradients | Best fit for **tabular gradient boosting**: NumPy-like n-d arrays, cheap views/slicing, column/row iteration, zero-copy interop with rust-numpy. GBDT is index/histogram-bound, not dense-linear-algebra-bound, so a BLAS-decomposition library (faer/nalgebra) is the wrong center of gravity. **Pin 0.17** — rust-numpy 0.28 supports `ndarray >=0.15, <=0.17`. |
| **thiserror** | **2.0.18** | Library-level error enums | Mandated split. Derive typed errors in every library crate (`catboost-core`, `catboost-gpu`, …). v2 is the current major line. |
| **anyhow** | **1.0.102** | Application/binding-level error propagation | Mandated split. Use only in the PyO3 binding crate, the oracle test harness, and any bin targets — never in library public APIs. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **numpy** (rust-numpy) | **0.28.0** | Zero-copy NumPy ⇄ ndarray in the Python binding | Always, in the `pyo3` binding crate. **It is the version pin anchor**: it transitively fixes `pyo3=0.28` and `ndarray≤0.17`. |
| **arrow** (arrow-rs) | **59.0.0** | Apache Arrow array ingest; `pyarrow` feature for zero-copy PyArrow interop | When accepting Arrow tables / interop with the Python Arrow ecosystem. Enable the `pyarrow` feature. |
| **polars** | **0.54.4** | DataFrame ingest (Pandas-replacement path) | When accepting Polars `DataFrame`/`LazyFrame` input. Use Polars' Arrow-backed columns for zero-copy into the core. |
| **pyo3-polars** | **0.27.0** | Pass Polars frames across the Python boundary without serialization | In the binding crate when Python-side callers hand in Polars frames. Track Polars↔pyo3-polars version pairing carefully (see Version Compatibility). |
| **rayon** | **1.12.0** | CPU data-parallelism (histogram building, per-tree/per-feature parallelism) | The CPU backend's parallelism layer. CubeCL's `cpu` runtime handles kernel-style CPU compute; rayon handles coarse task parallelism in the training driver. |
| **bytemuck** | **1.25.0** | POD host↔device byte transfer for CubeCL handles | Required by CubeCL data transfer (`Bytes::from_elems`, `cast_slice`). |
| **serde** | **1.0.228** | Model (de)serialization framework | Model save/load. Pair with a compact binary format (below). |
| **bincode** | **2.0.0** (use the **2.x** line, not 3.x default) | Compact binary model encoding | Cross-version model files. **Caution:** bincode 2.x is the documented, widely-adopted API; a 3.0.0 exists on crates.io — confirm its API/stability before adopting. Default to 2.x unless 3.x is verified stable for your use. |
| **half** | latest | `f16`/`bf16` support if mirroring CatBoost low-precision paths | Only if GPU kernels use bf16/f16 (CubeCL/ROCm support bf16). |

### Testing & Quality Libraries

| Library | Version | Purpose | Notes |
|---------|---------|---------|-------|
| **proptest** | **1.11.0** | Random input generation for the oracle harness | Recommended over quickcheck: better shrinking, value-based strategies, easier composite strategies for generating `Pool`-shaped tabular datasets (mixed float/cat/text columns). |
| **approx** | **0.5.1** | Float comparison to the 1e-5 oracle tolerance | Same crate the reference Rust package already uses. Use `assert_abs_diff_eq!(got, expected, epsilon = 1e-5)` to match the **absolute**-error parity bar exactly. |
| **rstest** | **0.26.1** | Parameterized/fixture test cases | For table-driven oracle cases and backend-parameterized tests. |
| **insta** | latest | Snapshot tests for serialized model bytes / metadata | Optional; useful for serialization round-trip stability. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `cargo` workspaces | Modular crate layout | See **Workspace Layout** below. |
| `cargo clippy` | Lint; enforce no-`unwrap()` | Add `#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]` at crate roots (allow in `tests/`). This is how the "no unwrap in production" rule is mechanically enforced. |
| `cargo deny` | License/dup/advisory auditing | Recommended for a multi-backend dependency tree. |
| `maturin develop` / `build` | Local + release wheel builds | `maturin build --release --features rocm` etc. per wheel. |
| `cargo nextest` | Faster, isolated test runner | Helps keep GPU (`rocm`) tests isolated from CPU tests via filters. |

---

## CubeCL Backend Strategy (load-bearing)

**Version: cubecl 0.10.0.** Backend selection is **purely compile-time** via Cargo features, satisfying the "no runtime dispatch overhead" requirement through monomorphization over a generic `R: Runtime`.

**Feature → backend mapping (verified from the cubecl 0.10.0 feature graph on crates.io):**

| Project Cargo feature | cubecl feature | Underlying crate / runtime type | Notes |
|-----------------------|----------------|----------------------------------|-------|
| `cpu` | `cpu` | `cubecl-cpu` → `cubecl::cpu::CpuRuntime` / `CpuDevice` | Always-available fallback; use for CI correctness + as the non-GPU oracle path. |
| `cuda` | `cuda` | `cubecl-cuda` → `cubecl::cuda::CudaRuntime` | **Untestable locally per PROJECT.md** — compile-gate only, do not run in CI. |
| `rocm` | `rocm` (**alias that enables `hip`**) | `cubecl-hip` → `cubecl::hip::HipRuntime` / `HipDevice` | **The GPU test backend.** In cubecl 0.10, `rocm = ["hip"]`; the runtime type is `HipRuntime`. ROCm/HIP backend is functional and actively maintained ("work in progress" upstream — see Pitfalls/maturity note). |
| `wgpu` | `wgpu` | `cubecl-wgpu` → `cubecl::wgpu::WgpuRuntime` | Portable Vulkan/Metal/DX/WebGPU fallback; good for dev machines without CUDA/ROCm. Sub-features: `vulkan`(spirv), `metal`(msl). |

**Generic runtime pattern (the core abstraction):** Write kernels once with a `Numeric`/`Float` bound, write training driver functions generic over `R: Runtime`, and select the concrete runtime in **one** `cfg`-gated type alias module so no other code references a concrete backend:

```rust
// catboost-gpu/src/backend.rs — the ONLY place backends are named
#[cfg(feature = "rocm")]
pub type Backend = cubecl::hip::HipRuntime;       // GPU tests run here
#[cfg(all(feature = "cuda", not(feature = "rocm")))]
pub type Backend = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "wgpu", not(any(feature = "cuda", feature = "rocm"))))]
pub type Backend = cubecl::wgpu::WgpuRuntime;
#[cfg(not(any(feature = "cuda", feature = "rocm", feature = "wgpu")))]
pub type Backend = cubecl::cpu::CpuRuntime;       // default fallback

pub fn device() -> <Backend as cubecl::Runtime>::Device { Default::default() }
```

Kernels stay generic (`fn kernel<N: Numeric, R: Runtime>(...)`) and launch as `kernel::launch::<N, Backend>(...)`. This is exactly the pattern shown in the CubeCL generics + ROCm manuals (`cubecl::hip::HipRuntime` for `rocm`, `cubecl::cpu::CpuRuntime` for fallback).

**bf16 / wmma note:** matrix-multiply acceleration on ROCm defaults to RDNA3; other AMD archs need the `hip-rocwmma` feature (`cubecl-hip/rocwmma`). Histogram-based GBDT mostly avoids dense matmul, so this is likely unneeded for v1 but is the lever if you add matmul-heavy paths.

---

## Workspace Layout (modular, feature-gated)

```
catboost-rs/                  (virtual workspace root)
├─ crates/
│  ├─ catboost-core/          # algorithm: trees, boosting, cat/text encoders, SHAP — pure CPU, thiserror
│  ├─ catboost-gpu/           # CubeCL kernels + backend.rs cfg-alias; feature-gated cuda/rocm/wgpu/cpu
│  ├─ catboost-data/          # Pool, ndarray/arrow/polars ingest, zero-copy adapters
│  ├─ catboost-model/         # serde/bincode (de)serialization, model format
│  ├─ catboost/               # public Rust API (Builder pattern) — re-exports, thiserror
│  └─ catboost-py/            # PyO3 binding crate (anyhow ok here); sklearn + CatBoost-native API; maturin target
├─ xtask/ or tests/oracle/    # oracle harness: proptest gen + approx 1e-5 vs original CatBoost
```

- `unsafe`/backend names confined to `catboost-gpu`.
- `anyhow` confined to `catboost-py` + oracle harness; every other crate is `thiserror`-only.
- Tests live in `tests/` dirs or sibling `*_test.rs` files — **never** inline `#[cfg(test)]` in production source (PROJECT.md + reference ARCHITECTURE anti-pattern).

---

## Installation

```toml
# Workspace root Cargo.toml (excerpt)
[workspace]
resolver = "2"
members = ["crates/*"]

[workspace.dependencies]
cubecl    = { version = "0.10.0", default-features = false }  # features set per-backend by catboost-gpu
ndarray   = "0.17"
thiserror = "2.0"
anyhow    = "1.0"
rayon     = "1.12"
bytemuck  = "1.25"
serde     = { version = "1.0", features = ["derive"] }
bincode   = "2.0"
# Python boundary (catboost-py only)
pyo3      = { version = "0.28", features = ["extension-module", "abi3-py312"] }
numpy     = "0.28"
arrow     = { version = "59", features = ["pyarrow"] }
polars    = { version = "0.54", features = ["lazy"] }
pyo3-polars = "0.27"
# Dev / test
proptest  = "1.11"
approx    = "0.5"
rstest    = "0.26"
```

```toml
# catboost-gpu/Cargo.toml — backend features pass through to cubecl
[features]
cpu  = ["cubecl/cpu"]            # default fallback
cuda = ["cubecl/cuda"]           # compile-only, untestable locally
rocm = ["cubecl/rocm"]           # => cubecl/hip ; GPU tests run here
wgpu = ["cubecl/wgpu"]
default = ["cpu"]
```

```bash
# Build per-backend wheels (Python >= 3.12, abi3)
maturin build --release -m crates/catboost-py/Cargo.toml --features rocm   # catboost-rs-rocm
maturin build --release -m crates/catboost-py/Cargo.toml --features cuda   # catboost-rs-cuda
maturin build --release -m crates/catboost-py/Cargo.toml --features wgpu   # catboost-rs-wgpu
maturin build --release -m crates/catboost-py/Cargo.toml --features cpu    # catboost-rs (cpu)
```

---

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| **ndarray** (CPU core) | **faer** 0.24 | If a subroutine becomes dominated by dense matrix **decompositions** (QR/SVD/Cholesky) — faer matches/beats OpenBLAS/LAPACK there. GBDT is histogram/index-bound, so faer is a targeted dependency at most, not the core. |
| **ndarray** | **nalgebra** 0.35 | Only for small fixed-size linear algebra (geometry/stats helpers). Wrong shape for large tabular batches; ndarray's view/slice ergonomics + rust-numpy zero-copy win for this domain. |
| **pyo3 0.28.3** | **pyo3 0.29.0** | Only once `rust-numpy` ships a 0.29-compatible release. Adopting 0.29 now breaks rust-numpy zero-copy NumPy interop. Revisit at each rust-numpy release. |
| **proptest** | **quickcheck** | If you want minimal-dependency, Haskell-style property tests. proptest's shrinking + composite strategies are better for generating structured `Pool` datasets. |
| **bincode** | **postcard** / **rmp-serde** | postcard for `no_std`/embedded model formats; messagepack for cross-language model exchange. For a Rust-first `.cbm`-style format, bincode is simplest. |
| **arrow-rs + polars** | **arrow2** | arrow2 is effectively deprecated/merged into the polars-arrow lineage — prefer the official `arrow` (arrow-rs) crate. |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| **pyo3 0.29** (right now) | rust-numpy 0.28 pins `pyo3 ^0.28`; mixing 0.29 breaks zero-copy NumPy interop and won't resolve. | pyo3 **0.28.3** + numpy 0.28 |
| **ndarray 0.16 or 0.18+** with rust-numpy 0.28 | rust-numpy 0.28 supports `ndarray >=0.15, <=0.17` only; mismatched ndarray versions = two incompatible `ArrayBase` types and no zero-copy. | ndarray **0.17.x** |
| **thiserror 1.x** | v2 is the current major line with the maintained API. | thiserror **2.x** |
| **anyhow in library crates** | Erases typed errors from the public Rust API; violates the thiserror/anyhow split. | `thiserror` enums in libs; `anyhow` only at bindings/app/test edge. |
| **`unwrap()` / `expect()` / `panic!` in production** | Hard project rule. | `Result` + `?`; enforce with `#![deny(clippy::unwrap_used, clippy::expect_used)]`. |
| **C API / bindgen FFI layer** (as in the reference C++ package) | PROJECT.md: PyO3 direct bindings only, no CAPI/unsafe C ABI. | PyO3 `#[pymodule]` directly over the Rust core. |
| **A separate handwritten CUDA + ROCm kernel codebase** | Defeats the single-source CubeCL mandate; double maintenance. | One `#[cube]` kernel, backend by feature. |
| **Runtime backend dispatch (enum/dyn)** | Violates "no runtime dispatch overhead." | Compile-time `cfg` type alias + generic `R: Runtime` monomorphization. |

---

## Stack Patterns by Variant

**If a developer machine has no CUDA/ROCm GPU:**
- Build with `--features wgpu` (Vulkan/Metal) for GPU dev, or `--features cpu` for correctness.
- Because CubeCL runs the same kernels on `wgpu`/`cpu`, parity can be checked off-GPU before validating on the `rocm` CI box.

**If targeting AMD CI/runtime (the GPU test path):**
- `--features rocm` → `HipRuntime`; ensure host has ROCm/HIP runtime libs (`amdhip64`, `hiprtc`) matching the `cubecl-hip-sys` HIP version.
- For matmul-heavy additions on non-RDNA3 AMD: add `hip-rocwmma`.

**If the consumer passes Polars/Pandas frames from Python:**
- Accept via `pyo3-polars` (Polars) or rust-numpy + Arrow (Pandas → NumPy/Arrow); convert to Arrow-backed columns and adapt into the core `Pool` zero-copy where dtypes allow.

---

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| `numpy = 0.28` | `pyo3 = ^0.28` (use 0.28.3) | **Anchor constraint.** Verified from rust-numpy 0.28 dependency metadata. |
| `numpy = 0.28` | `ndarray = >=0.15, <=0.17` | Use 0.17.x to match workspace `ndarray`. Verified from crates.io. |
| `pyo3 = 0.28` | `abi3-py312` feature | Single abi3 wheel works on CPython ≥ 3.12 (meets Python ≥ 3.12 requirement). |
| `cubecl = 0.10` | feature `rocm` ⇒ `hip` ⇒ `cubecl-hip` | Runtime type is `cubecl::hip::HipRuntime`. Verified from cubecl 0.10.0 feature graph. |
| `cubecl-hip` | ROCm/HIP host libs | `cubecl-hip-sys` versions track **HIP** version (not ROCm) since May 2025; match the installed HIP. |
| `polars = 0.54` | `pyo3-polars = 0.27` | Polars ↔ pyo3-polars are tightly coupled; bump them together and pin exact minors. |
| `arrow = 59` | `pyarrow` feature | Enables zero-copy PyArrow FFI across the boundary. |
| `maturin = 1.14` | `pyo3 = 0.28` | maturin auto-detects PyO3 + abi3; no extra config beyond `[tool.maturin]`. |

> **Resolver note:** rust-numpy is the pin anchor. Any pyo3 bump must wait for a rust-numpy release that allows it; otherwise NumPy zero-copy breaks. Track [PyO3/rust-numpy releases] at each upgrade.

---

## Sources

- crates.io API (live, 2026-06-13, HIGH) — verified max-stable versions: cubecl 0.10.0, pyo3 0.29.0 (0.28.3 latest 0.28.x), numpy 0.28.0, ndarray 0.17.2, faer 0.24.0, nalgebra 0.35.0, thiserror 2.0.18, anyhow 1.0.102, proptest 1.11.0, approx 0.5.1, maturin 1.14.0, arrow 59.0.0, polars 0.54.4, pyo3-polars 0.27.0, rstest 0.26.1, serde 1.0.228, bincode 3.0.0(/2.x), bytemuck 1.25.0, rayon 1.12.0.
- crates.io dependency metadata (live, HIGH) — `numpy 0.28` → `pyo3 ^0.28`, `ndarray >=0.15,<=0.17`; `cubecl 0.10.0` feature graph (`rocm`→`hip`, `cpu`/`cuda`/`wgpu` mappings, `abi3-py312` on pyo3).
- Vendored CubeCL manual `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/` (HIGH) — `Cubecl_generics.md` (generic `R: Runtime` launch pattern, `CpuRuntime`), `Handling_Interleaved_Complex_Numbers_…ROCm_Backend.md` (`cubecl::hip::HipRuntime`/`HipDevice` for `rocm`, CPU fallback pattern).
- [tracel-ai/cubecl releases & cubecl-hip-sys](https://github.com/tracel-ai/cubecl-hip-sys) (web, MEDIUM) — ROCm/HIP backend actively maintained, "work in progress"; HIP-version-based binding scheme since May 2025; ROCm 6.4 updates.
- [PyO3/rust-numpy releases](https://github.com/PyO3/rust-numpy/releases) (web, MEDIUM) — rust-numpy tracks PyO3 minor versions 1:1; 0.28 ⇄ pyo3 0.28.
- [faer-rs README / docs.rs](https://docs.rs/faer) (web, MEDIUM) — faer targets medium/large **dense** decompositions; not the right core for index/histogram-bound GBDT.
- `.planning/PROJECT.md`, `.planning/codebase/STACK.md`, `.planning/codebase/ARCHITECTURE.md` (project, HIGH) — constraints, reference stack, anti-patterns.

---
*Stack research for: gradient-boosting ML library (Rust core, multi-backend GPU via CubeCL, PyO3/maturin Python bindings)*
*Researched: 2026-06-13*
