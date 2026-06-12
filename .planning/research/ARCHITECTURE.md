# Architecture Research

**Domain:** Gradient-boosting ML library — modular Rust rewrite of CatBoost (Cargo workspace, CubeCL multi-backend GPU, dual PyO3 Python API)
**Researched:** 2026-06-13
**Confidence:** HIGH (derived from the vendored CatBoost C++ source tree at `catboost-master/` and the project-mandated CubeCL manual; both are curated, first-party references)

> Scope note: this is the **architecture dimension only** — crate decomposition, the CubeCL generic-runtime boundary, the CPU/GPU split, the memory-efficiency storage model, the PyO3 dual-API mapping, and the resulting build order. Algorithm specifics (CTR math, ordered boosting, SHAP) and library version pins live in STACK.md / FEATURES.md / PITFALLS.md.

---

## Standard Architecture

### System Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│  PYTHON SURFACE  (two front-ends, one core)                            │
│  ┌──────────────────────────┐   ┌──────────────────────────────────┐   │
│  │ sklearn-compatible        │   │ CatBoost-native                  │   │
│  │ fit/predict/predict_proba │   │ Pool, CatBoostClassifier/        │   │
│  │ /score  (estimator API)   │   │ Regressor, native param names    │   │
│  └────────────┬─────────────┘   └────────────────┬─────────────────┘   │
│               └──────────────┬───────────────────┘                     │
│                   crate: cb-python  (PyO3 + maturin)                    │
└───────────────────────────────┬────────────────────────────────────────┘
                                 │  pyclass wrappers → safe Rust calls
                                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│  RUST PUBLIC API   crate: catboost-rs  (Builder pattern, re-exports)    │
│  CatBoostBuilder → fit(&Pool) → Model → predict(&Pool)                  │
└───────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│  BACKEND-AGNOSTIC CORE  (no GPU types leak here)                        │
│  ┌────────────────┐  ┌──────────────────┐  ┌────────────────────────┐  │
│  │ cb-data         │  │ cb-core           │  │ cb-model               │  │
│  │ Pool, columns,  │  │ boosting loop,    │  │ tree/oblivious-tree    │  │
│  │ quantization,   │  │ tree-build        │  │ structs, CTR tables,   │  │
│  │ CTR cat-encode, │  │ orchestration,    │  │ serialize/deserialize  │  │
│  │ Arrow/NumPy in  │  │ leaf estimation,  │  │ (.cbm + native fmt),   │  │
│  │ (zero-copy)     │  │ loss/metric reg.  │  │ CPU eval / apply       │  │
│  └────────┬────────┘  └─────────┬─────────┘  └───────────┬────────────┘  │
│           │                     │ generic over <R: Runtime>             │
│           └─────────────────────┼────────────────────────┘             │
└─────────────────────────────────┼──────────────────────────────────────┘
                                  │  trait Backend / ComputeClient<R>
                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│  BACKEND LAYER   crate: cb-compute   (CubeCL kernels, generic over R)   │
│  histogram build · gradient/hessian · gradient pairs · partition scan   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐                    │
│  │ feat:cpu │ │feat:wgpu │ │feat:cuda │ │feat:rocm │   (Cargo features) │
│  │CpuRuntime│ │WgpuRtime │ │CudaRtime │ │RocmRtime │                    │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘                    │
└────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities (crate-level)

| Crate | Responsibility (what it owns) | Backend-aware? | Mirrors CatBoost C++ |
|-------|-------------------------------|----------------|----------------------|
| `cb-data` | `Pool`/dataset, columnar + quantized feature storage, quantization (border selection), categorical CTR/perfect-hash encoding, exclusive-feature-bundling, zero-copy ingestion from NumPy/Arrow/Polars | No (CPU storage; produces compute-ready buffers) | `catboost/libs/data/` (`columns`, `quantization`, `quantized_features_info`, `ctrs`, `exclusive_feature_bundling`, `objects`) |
| `cb-core` | Boosting loop, tree-building orchestration (oblivious-tree structure search), leaf-value estimation, bootstrap/sampling, loss + metric registry, ordered-boosting bookkeeping | Generic over `R: Runtime` (delegates compute to `cb-compute`) | `catboost/libs/train_lib/` + `catboost/cuda/methods/` (`oblivious_tree_structure_searcher`, `dynamic_boosting`, `leaves_estimation`) |
| `cb-model` | Trained model representation (oblivious trees, splits, CTR value tables, scale & bias), serialization/deserialization (`.cbm` parity + native format), CPU inference/apply path | No (CPU apply; GPU eval optional later) | `catboost/libs/model/` (`model`, `ctr_data`, `eval_processing`, `flatbuffers`, `model/cpu`) |
| `cb-compute` | All CubeCL kernels: histogram computation, gradient/hessian, gradient-pairs, partition/prefix scan, reductions. Exposes a thin backend-selection facade; kernels are generic over `R: Runtime` and `F: Float` | **Yes** — the only crate that names backend runtimes; feature-gated `cpu`/`wgpu`/`cuda`/`rocm` | `catboost/cuda/cuda_lib/`, `cuda/cuda_util/`, `cuda/methods/kernel/`, `cuda/gpu_data/` |
| `catboost-rs` | Public Rust API: Builder pattern, re-exports `Pool`/`Model`, `thiserror` error enum, parameter struct. Selects backend via its own Cargo features that forward to `cb-compute` | Re-exports backend feature flags | `catboost/rust-package/src/` (greenfield equivalent of the safe wrapper) |
| `cb-python` | PyO3 `#[pyclass]` wrappers, maturin packaging, both sklearn-estimator and CatBoost-native (`Pool`, `CatBoostClassifier/Regressor`) surfaces, `anyhow` at the boundary, per-backend wheels | Re-exports backend feature flags | `catboost/python-package/` (Cython `_catboost` + `core.py`) |
| `cb-fstr` *(optional split)* | SHAP / feature-importance (can start as a module inside `cb-model`, promote to a crate if it grows) | No | `catboost/libs/fstr/` |

**Why these boundaries:** CatBoost's own tree cleanly separates `libs/data` (datasets + quantization), `libs/train_lib` (training orchestration), `libs/model` (model format + CPU apply), and `cuda/` (GPU kernels). We preserve that seam but invert one dependency: in C++ the GPU layer pulls in its own data structures (`cuda/gpu_data/`); in Rust we keep **one** backend-agnostic `cb-data` and push only the compute primitives into `cb-compute`, so the boosting orchestration in `cb-core` never branches on backend.

---

## Recommended Project Structure

```
catboost_rs/                         # Cargo workspace root (virtual manifest)
├── Cargo.toml                       # [workspace] members + shared deps/versions
├── crates/
│   ├── cb-data/                     # Pool, columns, quantization, CTR, ingestion
│   │   ├── src/
│   │   │   ├── pool.rs              # Pool: rows × typed columns + target/weights
│   │   │   ├── columns.rs           # Float/Cat/Text/Embedding column storage
│   │   │   ├── quantized.rs         # QuantizedPool: u8/u16 bin indices + borders
│   │   │   ├── quantization.rs      # border selection (table-stakes algorithm)
│   │   │   ├── ctr.rs               # categorical → numeric encoding tables
│   │   │   ├── ingest_numpy.rs      # zero-copy from NumPy buffer protocol
│   │   │   └── ingest_arrow.rs      # zero-copy from Arrow/Polars (bytemuck recast)
│   │   └── tests/                   # separate test files (no inline mod tests)
│   ├── cb-compute/                  # CubeCL kernels, generic over R: Runtime
│   │   ├── src/
│   │   │   ├── backend.rs           # Backend facade: select_runtime(), client init
│   │   │   ├── histogram.rs         # #[cube(launch)] histogram kernel<F: Float>
│   │   │   ├── gradients.rs         # gradient/hessian kernels<F: Float>
│   │   │   ├── scan.rs              # partition / prefix-scan kernels
│   │   │   └── reduce.rs            # reductions (leaf sums)
│   │   └── tests/                   # GPU tests gated to rocm feature
│   ├── cb-core/                     # boosting + tree-build orchestration
│   │   ├── src/
│   │   │   ├── boosting.rs          # the boosting loop (backend-agnostic)
│   │   │   ├── tree_builder.rs      # oblivious-tree structure search
│   │   │   ├── leaf_estimation.rs   # leaf value computation
│   │   │   ├── loss/                # loss & metric registry (Logloss, RMSE, ...)
│   │   │   └── sampling.rs          # bootstrap / ordered-boosting permutations
│   │   └── tests/
│   ├── cb-model/                    # model format + CPU apply + SHAP
│   │   ├── src/
│   │   │   ├── tree.rs              # oblivious tree + split representation
│   │   │   ├── model.rs             # Model: trees + CTR tables + scale/bias
│   │   │   ├── apply.rs             # CPU inference path
│   │   │   ├── serialize.rs         # .cbm parity + native (serde) format
│   │   │   └── fstr.rs              # SHAP / feature importance (or split crate)
│   │   └── tests/
│   ├── catboost-rs/                 # public Rust crate (Builder API, re-exports)
│   │   ├── src/lib.rs               # CatBoostBuilder, re-export Pool/Model, errors
│   │   └── tests/                   # oracle parity tests vs original CatBoost
│   └── cb-python/                   # PyO3 + maturin
│       ├── Cargo.toml               # cdylib; forwards backend features
│       ├── pyproject.toml           # maturin build config
│       └── src/lib.rs               # #[pymodule]: sklearn + native pyclasses
└── .planning/
```

### Structure Rationale

- **One backend-agnostic core, one backend crate.** `cb-data`, `cb-core`, `cb-model` contain **zero** backend type names. The only crate that ever writes `WgpuRuntime`/`CudaRuntime`/`RocmRuntime`/`CpuRuntime` is `cb-compute`. This keeps the boosting loop and tree-building logic written once and prevents `#[cfg(feature = "cuda")]` from leaking across the codebase.
- **`cb-data` is a leaf crate** (depends on nothing internal). Quantization and CTR encoding are pure CPU transforms that produce compute-ready bin buffers, so they sit below both `cb-core` and `cb-compute`.
- **`cb-compute` depends only on CubeCL + `cb-data`'s buffer types** (or a tiny shared `cb-types` crate if circular pressure appears). It does not depend on `cb-core`; instead `cb-core` depends on `cb-compute` and drives it. This is the direction that makes the generic-runtime parameter flow naturally downhill.
- **`cb-model` is independent of `cb-compute`** for the CPU apply path, so prediction works on any machine with no GPU toolchain. (GPU inference, if added later, becomes an optional dependency from `cb-model` on `cb-compute` — mirroring CatBoost's optional `EnableGPUEvaluation`.)
- **Python and Rust public crates are thin.** They add ergonomics (Builder, pyclasses) and error-context conversion, nothing algorithmic. Both forward the same four backend Cargo features down to `cb-compute`, so a per-backend wheel is just a different feature selection at `maturin build` time.

---

## Architectural Patterns

### Pattern 1: Generic-Runtime Boundary (CubeCL zero-cost backend switching)

**What:** The CPU/GPU choice is a **compile-time type parameter** `R: Runtime`, not a runtime `match`. Kernels are written once with `#[cube(launch)]` and are generic over the numeric float type `F: Float` (mandated by AGENTS.md: "Cubecl kernel need generics-float in this project"). The launch site is monomorphized per backend, so there is no dynamic dispatch and no vtable in the hot loop.

**When to use:** Everywhere a compute primitive crosses into `cb-compute`. The boundary type is CubeCL's `Runtime` trait (and its associated `ComputeClient<R>` / `R::Device`).

**Concrete boundary shape** (from the CubeCL manual, generics + matmul examples):

```rust
// cb-compute/src/histogram.rs — kernel: generic over the FLOAT type only.
use cubecl::prelude::*;

#[cube(launch)]
pub fn histogram_kernel<F: Float>(
    bins:  &Array<u32>,     // quantized feature bin indices
    grads: &Array<F>,       // per-sample gradients
    hist:  &mut Array<F>,   // output histogram (sum of grads per bin)
) {
    if ABSOLUTE_POS < bins.len() {
        // atomic/segmented accumulate elided for brevity
        hist[bins[ABSOLUTE_POS]] += grads[ABSOLUTE_POS];
    }
}

// The driver is generic over the RUNTIME. `cb-core` calls this; it never
// mentions a concrete backend.
pub fn build_histogram<R: Runtime, F: Float + CubeElement + bytemuck::Pod>(
    client: &ComputeClient<R::Server, R::Channel>,
    bins: ArrayHandleRef<R>,
    grads: ArrayHandleRef<R>,
    hist: ArrayHandleRef<R>,
    n: usize,
) {
    histogram_kernel::launch::<F, R>(           // generic params order: <F, R>
        client,
        CubeCount::Static(/* groups */, 1, 1),
        CubeDim { x: 256, y: 1, z: 1 },
        bins.as_array_arg(1),
        grads.as_array_arg(1),
        hist.as_array_arg(1),
    );
}
```

```rust
// cb-compute/src/backend.rs — the ONLY place backends are named.
// Each is behind a Cargo feature so only one is compiled into a given wheel.
#[cfg(feature = "wgpu")]  pub type SelectedRuntime = cubecl::wgpu::WgpuRuntime;
#[cfg(feature = "cuda")]  pub type SelectedRuntime = cubecl::cuda::CudaRuntime;
#[cfg(feature = "rocm")]  pub type SelectedRuntime = cubecl::rocm::RocmRuntime; // hip
#[cfg(feature = "cpu")]   pub type SelectedRuntime = cubecl::cpu::CpuRuntime;
```

`cb-core` is then generic — `fn fit<R: Runtime>(...)` — or, more ergonomically, the public `catboost-rs` crate fixes `R = cb_compute::SelectedRuntime` once, so end users never write a type parameter while the internals stay backend-polymorphic and testable against `CpuRuntime`.

**Trade-offs:** Monomorphization gives zero-cost dispatch and lets tests run the CPU runtime while production uses GPU, but it increases compile time and means a single binary embeds exactly one backend (intentional — matches the "per-backend wheel" requirement). `f64` parity vs `f32` GPU speed is a precision/perf tension flagged for STACK/PITFALLS (the 10⁻⁵ oracle tolerance likely forces `f64` in accumulation paths).

### Pattern 2: CPU/GPU split — agnostic orchestration, backend-specific primitives

**What:** Split the algorithm by *what is data-parallel and hot* vs *what is control flow*.

| Stays backend-agnostic (in `cb-core`/`cb-model`) | Pushed to backend kernels (in `cb-compute`) |
|--------------------------------------------------|---------------------------------------------|
| The boosting loop (iteration over trees) | Histogram computation (per-feature, per-bin sums) |
| Tree-building orchestration / structure search control flow | Gradient & hessian computation |
| Best-split selection *policy* (the loop that compares scores) | Per-bin score reduction feeding split selection |
| Leaf-value estimation orchestration | Partition / prefix-scan over samples |
| Loss/metric registry, parameter handling | Bootstrap weight generation, sampling permutation application |
| Model assembly, serialization, CPU apply | (GPU apply optional, later) |

**Why:** This is exactly how CatBoost's `cuda/methods/` is organized — the `oblivious_tree_structure_searcher` drives the search while `histograms_helper` and the `kernel/` directory hold the device code. Keeping the *policy* on the host means the algorithm is written once; only the *number crunching* is duplicated across backends (and even that is a single generic kernel).

### Pattern 3: Memory-efficiency — columnar, quantized, zero-copy

**What:** Three reinforcing techniques, all first-class per the PROJECT memory constraint:

1. **Columnar + quantized storage.** Float features are pre-binned into `u8` (≤256 borders) or `u16` bin indices in `cb-data::QuantizedPool`, not stored as `f32`. This is CatBoost's `quantized_features_info` / `packed_binary_features` model. Histogram kernels then read compact `u32`/`u8` bins instead of floats — less bandwidth, the dominant GBDT cost.
2. **Zero-copy ingestion.** NumPy (buffer protocol) and Arrow/Polars (`ScalarBuffer`/contiguous `Buffer`) expose dense, aligned slices. Use `bytemuck::cast_slice` to reinterpret `&[T]` as `&[u8]` and hand straight to `cubecl::bytes::Bytes` → `client.create(...)` with no element-wise copy (the `ZERO_COPY_ARROW_CUBECL` pattern). On the Python side, accept NumPy arrays without round-tripping through Python lists.
3. **Allocation discipline in the boosting loop.** Pre-allocate histogram/gradient buffers once and reuse them across iterations (arena/scratch buffers held by `cb-core`), rather than allocating per tree or per node. Mirrors CatBoost's `compressed_index` + reusable GPU buffers.

**Trade-offs:** Zero-copy requires Pod/aligned, non-nullable, contiguous source columns; nullable Arrow arrays or Python object arrays must fall back to a validating copy path. Quantization is lossy by design (and is itself the oracle-sensitive border algorithm) — a correctness hot-spot for parity testing.

### Pattern 4: Dual Python API over one Rust core (PyO3)

**What:** `cb-python` exposes two `#[pymodule]` surfaces that both bottom out in the same `catboost-rs` calls:

- **sklearn-compatible:** `CatBoostClassifier`/`CatBoostRegressor` implementing `fit(X, y)`, `predict(X)`, `predict_proba(X)`, `score(X, y)` — `X` accepted as NumPy/Pandas/Arrow, internally built into a `Pool`.
- **CatBoost-native:** a `Pool` `#[pyclass]` and the same estimator classes accepting native parameter names (e.g. `iterations`, `depth`, `learning_rate`, `cat_features`), for drop-in migration.

Both construct a `cb_data::Pool`, call `catboost_rs::CatBoostBuilder`, and return a `Model` pyclass. Errors are converted at this boundary: `thiserror` enums from the library become Python exceptions via `anyhow`-style context.

**Trade-offs:** Two surfaces, one implementation — the only duplication is parameter-name mapping and the sklearn estimator contract glue. Per-backend wheels (`catboost-rs-rocm`, etc.) are produced by selecting the matching Cargo feature at `maturin build`.

---

## Data Flow

The end-to-end path the roadmap should sequence components against:

```
INGEST            QUANTIZE           TRAIN                       SERIALIZE        PREDICT
─────────         ─────────          ─────────                   ─────────        ─────────
NumPy/Arrow  ─►  border select   ─►  boosting loop (cb-core)  ─► Model (cb-model) ─► load Model
(zero-copy,      + bin to u8/u16     │  per iter:               ► .cbm / native     │
 cb-data)        + CTR encode        │   compute grad/hess ─┐    serialize          ▼
   │             cat features        │   build histograms  ─┤                    apply on Pool
   ▼             (cb-data)           │   (cb-compute @ R)   ─┘                    (CPU path,
 Pool ───────────► QuantizedPool ────┤   select splits (host)                     cb-model)
                                     │   estimate leaves                          │
                                     │   append oblivious tree                    ▼
                                     └─► repeat × iterations                   scores → Python
```

**Direction is strictly left-to-right and downhill in the crate graph:** raw data enters `cb-data`, is reduced to a `QuantizedPool`, which `cb-core` consumes while delegating the hot numeric steps to `cb-compute<R>`; the result is a `cb-model::Model` that serializes and later applies independently of any backend.

---

## Build Order / Dependency Graph

Internal crate dependencies (arrows = "depends on"):

```
cb-data  ◄────────────── cb-core ──────────► cb-compute ──► CubeCL
   ▲                        │                    ▲
   │                        ▼                    │ (feature: cpu/wgpu/cuda/rocm)
   └──────────────────── cb-model               │
                            ▲                    │
                            │                    │
                       catboost-rs ──────────────┘  (fixes R = SelectedRuntime)
                            ▲
                            │
                        cb-python  (PyO3 + maturin)
```

**Suggested build order (each layer must exist before the next):**

1. **Workspace skeleton + `cb-data`** — `Pool`, columnar storage, quantization, CTR cat-encoding, NumPy/Arrow ingestion. *Everything depends on data; nothing can be trained or tested without it.* No backend needed.
2. **`cb-compute` with the `cpu` backend only** — establish the `R: Runtime` generic boundary and the core kernels (histogram, gradient/hessian, scan) against `CpuRuntime`. *Proving the generic boundary on CPU first de-risks the GPU work and gives a testable oracle target with no GPU hardware.*
3. **`cb-core`** — boosting loop, tree-building orchestration, leaf estimation, loss registry — driving `cb-compute` over the generic runtime. End-to-end CPU training works here.
4. **`cb-model`** — tree/model representation, serialization (`.cbm` parity), CPU apply. First full train→serialize→predict cycle; first oracle parity tests vs original CatBoost (≤ 10⁻⁵).
5. **`catboost-rs`** — Builder-pattern public API and re-exports; thin layer over 1–4.
6. **Additional backends in `cb-compute`** (`wgpu`, then `rocm` for GPU test execution, `cuda` last/untestable-locally) — purely adding `#[cfg]` runtime types and validating kernels; no changes to `cb-core`. *Deferred because the generic boundary (step 2) already abstracts them.*
7. **`cb-python`** — PyO3 wrappers, dual sklearn + native API, maturin per-backend wheels. *Last, because it wraps the now-stable Rust API.*
8. **SHAP / `cb-fstr`** — feature importance, built on the finished `cb-model` apply path. *Can run in parallel with step 6/7 once `cb-model` exists.*

**Critical build-order implications for the roadmap:**
- The **generic-runtime boundary must be designed in step 2**, not retrofitted — choosing the `R: Runtime` / `F: Float` signatures up front is what makes steps 6 a no-op for `cb-core`. Retrofitting genericity later forces a rewrite of the boosting loop.
- **CPU backend before any GPU backend.** It is the oracle-test vehicle and needs no GPU toolchain. GPU (`rocm` for tests, `cuda` untestable locally) is additive.
- **Quantization (step 1) and CTR encoding are oracle hot-spots** — they sit at the very bottom of the graph, so a parity bug there poisons everything above. Flag them for deep, early oracle testing.
- **Python (step 7) is strictly downstream** of a stable `catboost-rs` API; do not start the PyO3 dual surface until the Rust Builder API is settled, or the binding glue will churn.

---

## Anti-Patterns to Avoid

| Anti-pattern | Why it's wrong here | Do instead |
|--------------|---------------------|------------|
| `#[cfg(feature="cuda")]` branches scattered through `cb-core`/`cb-model` | Backend leaks into orchestration; defeats zero-cost generic switching; multiplies test matrix | Confine all backend type names to `cb-compute::backend`; keep core generic over `R: Runtime` |
| Runtime `enum Backend { Cpu, Cuda, ... }` dispatch | Violates the PROJECT "no runtime switching, compile-time only" constraint; adds dispatch overhead in the hot loop | Compile-time `R: Runtime` type parameter, selected by Cargo feature |
| Storing features as dense `f32` and quantizing on the fly | Wastes the dominant memory budget; re-binning per iteration is slow | Quantize once into `u8`/`u16` `QuantizedPool` in `cb-data` |
| Element-wise copy of NumPy/Arrow into `Vec<f32>` before upload | Burns the memory-efficiency budget the project mandates | `bytemuck::cast_slice` → `cubecl::bytes::Bytes` zero-copy upload |
| Per-iteration allocation of histogram/gradient buffers | Allocator churn dominates the boosting loop | Pre-allocate reusable scratch buffers owned by `cb-core` |
| Coupling `cb-model` apply to `cb-compute` | Forces a GPU toolchain just to run prediction | Keep CPU apply self-contained in `cb-model`; make GPU eval an optional dependency added later |
| Inline `#[cfg(test)] mod tests` in production source | Explicitly prohibited by AGENTS.md / PROJECT | Separate `tests/` files or `src/foo_test.rs`; GPU tests gated to `rocm` |

---

## Confidence Assessment

| Area | Confidence | Basis |
|------|------------|-------|
| Crate decomposition & boundaries | HIGH | Directly mirrors CatBoost's own `libs/data` / `train_lib` / `model` / `cuda` separation, verified by inspecting the vendored source tree |
| CubeCL generic-runtime boundary | HIGH | CubeCL manual generics + matmul examples show the exact `<F, R>` launch signature and `Runtime`/`ComputeClient` types; AGENTS.md mandates generic-float kernels |
| CPU/GPU split | HIGH | Maps 1:1 to `cuda/methods/` (host search vs `kernel/` device code) in the reference |
| Memory-efficiency architecture | HIGH | Quantized columnar model from `libs/data/quantization`+`packed_binary_features`; zero-copy path documented in `ZERO_COPY_ARROW_CUBECL` |
| Build order | HIGH | Derived from the internal dependency graph (data → compute → core → model → api → python) which is acyclic by construction |
| Exact CubeCL backend type names (`RocmRuntime` vs HIP naming) and current CubeCL version API | MEDIUM | Manual examples use `WgpuRuntime`/`CpuRuntime`/`CudaRuntime`; exact rocm runtime path and current crate API should be pinned in STACK.md against the installed CubeCL version |

## Sources

- `catboost-master/catboost/libs/data/` (columns, quantization, quantized_features_info, ctrs, exclusive_feature_bundling, objects) — HIGH (vendored reference)
- `catboost-master/catboost/libs/train_lib/`, `catboost/cuda/methods/`, `catboost/cuda/gpu_data/` — HIGH (vendored reference)
- `catboost-master/catboost/libs/model/` (model, ctr_data, eval_processing, flatbuffers, cpu/) — HIGH (vendored reference)
- `.planning/codebase/ARCHITECTURE.md`, `STRUCTURE.md`, `INTEGRATIONS.md` — HIGH (first-party codebase map)
- CubeCL manual: `Cubecl_generics.md`, `cubecl_matmul_gemm_example.md`, `ZERO_COPY_ARROW_CUBECL.md` — HIGH (project-mandated reference)
- `AGENTS.md`, `.planning/PROJECT.md` — HIGH (project constraints)
