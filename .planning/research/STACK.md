# Stack Research

**Domain:** v1.2 additions to a mature Rust rewrite of CatBoost — model export (ONNX/CoreML), GPU inference, extended fstr, orchestration, HNSW parity, PyPI release + benchmarking
**Milestone:** v1.2 Parity Completion & Release Readiness
**Researched:** 2026-07-05
**Confidence:** HIGH (crate versions verified via `cargo search`/pip 2026-07-05; ONNX/CoreML formats verified against upstream CatBoost source + onnx.ai/apple docs)

> Scope note: this file covers ONLY the **new stack additions** for v1.2. The existing
> workspace (cb-core, cb-data, cb-model, cb-train, cb-backend/CubeCL, cb-compute,
> cb-oracle, catboost-rs, catboost-rs-py) and its pins (pyo3 0.29, numpy 0.29,
> flatbuffers 25.12.19, arrow 59, polars 0.54, cubecl 0.10, thiserror 2, serde 1) are
> already validated and are NOT re-researched here. Several v1.2 features (GPU inference
> evaluator, CV/tuning/snapshot orchestration, online-HNSW port) need **no new external
> crates** — that finding is as important as the crates that must be added.

## Recommended Stack

### Core Technologies (new for v1.2)

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| `prost` | 0.14.4 | Runtime for generated protobuf structs (ONNX `ModelProto`, CoreML `Model`) | The idiomatic Rust protobuf codegen path — emits plain `#[derive]` structs you populate and `.encode()` to bytes. No mature Rust *ONNX-writer* library exists (tract/ort are readers only, see What NOT to Use), so the export path is "generate protobuf types from the official `.proto`, then hand-build the message" — exactly what upstream CatBoost does in C++. |
| `prost-build` | 0.14.4 | Build-time codegen of Rust from vendored `onnx.proto` + CoreML `.proto` | Compiles `.proto` → Rust in a `build.rs`. Mirrors the existing precedent in `cb-model` (flatc-generated FlatBuffers bindings), so the team already accepts a codegen-in-build step. |
| `protox` | 0.9.1 | Pure-Rust protobuf compiler backing `prost-build` | Removes the system `protoc` dependency from the build — `prost-build` on its own needs a `protoc` binary on PATH; `protox` parses `.proto` in-process. Keeps the build hermetic (the same reason the project vendors flatc output). |
| `criterion` | 0.8.2 | Rust-side statistical micro-benchmarks (predict, ONNX/CoreML export, GPU-inference kernels) | The Rust benchmarking standard (statistical, warmup, regression detection). NOT currently a dependency — the `criterion` token already in `catboost-rs/Cargo.toml` is a *comment* referring to a ROADMAP acceptance criterion, not the crate. Add as a dev-dependency with `[[bench]] harness = false`. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `protoc-bin-vendored` | 3.2.0 | Fallback vendored `protoc` binary | ONLY if `protox` cannot parse a schema feature (e.g. exotic proto2 extensions in the CoreML schema). Try `protox` first. |
| `maturin` (build tool) | 1.14.1 | Wheel builder (already the project's tool) | Already used with `abi3-py312`. No version bump forced; just extend the CI matrix (see PyPI section). |
| `PyO3/maturin-action` (CI action) | latest (v1) | GitHub Actions wheel build/cross-compile | Recommended over cibuildwheel for a Rust-first project (built-in cross-compilation, manylinux via zig/cross, per-interpreter/abi3 matrix). See PyPI section. |

### What does NOT need a new crate (findings)

| v1.2 feature | Stack decision | Rationale |
|--------------|----------------|-----------|
| **GPU inference evaluator** | Reuse `cb-backend` (CubeCL 0.10) + the v1.1 device-resident compressed-index and device-primitive library | A device-side tree-walk is a new *kernel*, not a new dependency. The bit-packed cindex, quantization, and reduction primitives from v1.1 training already live in `cb-backend`/`cb-compute`. Add a `predict`/tree-walk kernel over the existing `SelectedRuntime` generic. Landmine still applies: never add a `cb-train` dep to `cb-backend`. |
| **CV / grid+random tuning / snapshot-resume / calc_metrics / eval_result** | Pure Rust in a new `cb-orchestration` crate (or module in `cb-train`); reuse existing `serde`/`serde_json` + `.cbm` serialization for snapshots; reuse the project's existing **deterministic** RNG for random search | No orchestration crate needed. Do NOT pull `rand` (0.10.2) for random search — CatBoost reproducibility requires the same deterministic Mersenne-Twister-style stream the project already uses for bootstrap/MVS sampling in `cb-train`; a fresh `rand` seed would break oracle parity. Snapshot format = serialize training state via serde, matching the existing `.cbm`/JSON precedent. |
| **Extended fstr (Interaction, LossFunctionChange, partial-dependence)** | Pure Rust in `cb-model` (where SHAP + basic fstr already live) | These read `TModelTrees` structure + leaf values + re-evaluate on data — same inputs SHAP already consumes. No new crate; possibly `ndarray` (already workspace-pinned 0.17.2) for the PDP grids. |
| **Online-HNSW port (FEAT-07, bit-exact KNN parity)** | Hand-port `catboost-master/library/cpp/online_hnsw` inline into `cb-data` (~936 LOC). Do NOT add a third-party HNSW crate | Parity requires **bit-exact** reproduction of CatBoost's online (incremental) HNSW graph construction + search order. `hnsw_rs`/`instant-distance`/`hora` are independent implementations and will NOT reproduce CatBoost's neighbor selection bit-for-bit (root-cause already documented: the KNN estimated-feature residual is because upstream uses online HNSW while Rust used brute-force-exact). This is a transcription task, not a dependency choice. |

## Installation

```toml
# NEW crate: cb-export (depends on cb-model; keeps prost/protobuf OUT of cb-model core)
# crates/cb-export/Cargo.toml
[dependencies]
cb-model = { path = "../cb-model" }
cb-core  = { path = "../cb-core" }
prost    = "0.14.4"
thiserror.workspace = true          # typed export errors (D-14: no anyhow in libs)

[build-dependencies]
prost-build = "0.14.4"
protox      = "0.9.1"               # pure-Rust protoc; avoids system protoc on PATH

# Rust benchmark harness (dev-only), e.g. in catboost-rs or a bench crate
[dev-dependencies]
criterion = "0.8.2"

[[bench]]
name    = "predict_export"
harness = false
```

```bash
# Python-side end-to-end benchmark reference (already present in the repo/.venv)
#   catboost 1.2.x is already installed for the oracle harness — reuse it.
# CI wheel build uses PyO3/maturin-action (no local install needed).
```

**Vendored schema files to add** (checked into the new export crate, codegen'd in `build.rs`):
- ONNX: `onnx.proto` + `onnx-ml.proto` from `onnx/onnx` (the `ai.onnx.ml` domain lives in `onnx-ml.proto`).
- CoreML: `Model.proto`, `TreeEnsemble.proto`, `FeatureTypes.proto`, `DataStructures.proto` (+ transitive) from `apple/coremltools` `mlmodel/format/`.

## ONNX export — concrete emission path

**There is no Rust "ONNX writer" library.** The emission path is: vendor the ONNX `.proto`, codegen with `prost-build`+`protox`, then hand-construct the `ModelProto` mirroring CatBoost's `catboost/libs/model/model_export/onnx_helpers.cpp`.

**Match CatBoost's output exactly** (verified against upstream `onnx_helpers.cpp`, 2026-07-05):
- `ir_version = 3`
- `opset_import`: domain `ai.onnx.ml`, `version = 2` (classic ML opset)
- `producer_name = "CatBoost"` (or `"catboost-rs"`), `producer_version = <crate version>`
- **Regressor** → single `TreeEnsembleRegressor` node (`ai.onnx.ml`).
- **Classifier** → `TreeEnsembleClassifier` node followed by a `ZipMap` node (maps probabilities → labelled dict).
- Guards (already specified in `docs/CATBOOST_CORE_DESIGN.md` §Export Formats): numerical features only (no cat/text/embedding), identity scale required, oblivious trees only.

**Opset decision — use the classic (deprecated-but-universal) ops, NOT the new `TreeEnsemble` op.**
As of ONNX 1.23 (current), the `ai.onnx.ml` domain added a unified `TreeEnsemble` operator at **opset 5**, which formally **deprecated** `TreeEnsembleRegressor` and `TreeEnsembleClassifier`. Despite the deprecation label, the classic ops are what every ONNX runtime (onnxruntime, tract, etc.) supports today and what CatBoost itself still emits; the new `TreeEnsemble` op has thin runtime support. Emitting the classic ops maximizes drop-in interoperability and keeps structural parity with CatBoost's own export. Revisit only if a target runtime drops classic-op support.

## CoreML export — concrete emission path

**coremltools is Python-only and there is no Rust CoreML *writer*.** (`coreml-rs` 0.5.4 exists but is a macOS **inference** binding via swift-bridge — wrong tool, and macOS-only.) The `.mlmodel` file is a protobuf, so the path is identical to ONNX: vendor the CoreML `.proto` schema from `apple/coremltools` `mlmodel/format/`, codegen with `prost-build`+`protox`, hand-build the top-level `Model` message, mirroring CatBoost's `coreml_helpers.cpp`.

- Emit a `treeEnsembleRegressor` (or `treeEnsembleClassifier`) spec inside `Model`, plus an optional categorical-preprocessing pipeline.
- Requires identity scale (per the design-doc export guards); oblivious trees only; no text/embedding.
- Set `specificationVersion` to match the CoreML spec the schema targets (pin to the vendored schema's version; verify at implementation time against the checked-in `Model.proto`).

## Benchmarking harness — two layers

1. **End-to-end vs official CatBoost (accuracy + speed on real datasets)** → extend the **existing Python harness** (`benchmark.py`, `benchmark_fast.py`, `benchmark_small.py`, `bench/`). It already imports both `catboost` (the official PyPI package, ~1.2.x, present in the oracle `.venv`) and `catboost_rs`, so cross-implementation comparison belongs here — official CatBoost is only distributed as a Python package, so the reference must be invoked from Python. This is also where the `bench/generator.py` seeded-dataset discipline and the Kaggle CUDA sign-off notebooks already live.
2. **Rust-internal micro-benchmarks** (predict latency, ONNX/CoreML serialize cost, GPU-inference kernel throughput) → add **`criterion` 0.8.2** as a dev-dependency with `harness = false`. New dependency (not currently present).

Keep the two layers separate: criterion cannot invoke official CatBoost; the Python harness is not the right tool for tight Rust-function microbenchmarks.

## PyPI release — maturin abi3 per-backend wheels

- **Builder:** keep `maturin` (1.14.1). The py crate already sets `pyo3 = { features = ["abi3-py312", ...] }`, so one wheel per platform covers CPython ≥ 3.12 (abi3) — no per-minor-version matrix needed.
- **CI action:** use **`PyO3/maturin-action`** over `cibuildwheel`. Rationale: it is purpose-built for Rust wheels (manylinux via bundled cross toolchains, easy cross-compilation, abi3/abi3t/per-interpreter selection). cibuildwheel is viable and does support abi3, but adds manylinux/GLIBC ceremony that Rust cross-compilation makes unnecessary.
- **Per-backend wheels** (`cpu`/`cuda`/`rocm`/`wgpu`): backend selection is a **Cargo feature**, and one maturin invocation selects at most one build. So the CI matrix runs **one maturin leg per backend feature**, each producing a **separately-named distribution** (e.g. `catboost-rs` = cpu, `catboost-rs-cuda`, `catboost-rs-rocm`, `catboost-rs-wgpu`) so users `pip install` the wheel matching their hardware — consistent with the existing "install the backend-specific wheel" decision in PROJECT.md. Pass the feature via `pyproject.toml` `[tool.maturin] features` per leg (the crate already uses this mechanism for `pyo3/extension-module`).
- One `--find-interpreter` build will NOT emit two abi3 families; do not expect it to. abi3-py312 is a single forward-compatible family, which is what we want.

## Integration points into existing crates

| New capability | Lands in | Integration detail |
|----------------|----------|--------------------|
| ONNX / CoreML export | **new `cb-export` crate** (depends on `cb-model`) | Keeps `prost`/`prost-build`/`protox` + the vendored `.proto` blobs out of `cb-model`'s lean core (which today is flatbuffers+serde only). Reads the already-built `TModelTrees`/leaf-values representation `cb-model` owns. Alternative: a feature-gated `export` module inside `cb-model` — rejected to avoid pulling protobuf codegen into the core serialization crate. |
| GPU inference evaluator | `cb-backend` + `cb-compute` | New tree-walk/predict kernel over `SelectedRuntime`; reuse v1.1 cindex + primitives. No `cb-train` dep (landmine). |
| CV / tuning / snapshot / calc_metrics / eval_result | `cb-train` (or new `cb-orchestration`) | Pure Rust; serde-based snapshots; reuse deterministic RNG. |
| Extended fstr (Interaction / LossFunctionChange / PDP) | `cb-model` | Alongside existing SHAP/fstr; `ndarray` (0.17.2, pinned) for PDP grids. |
| Online-HNSW | `cb-data` | Inline port of `library/cpp/online_hnsw`; no external HNSW crate. |
| Python surface for all of the above | `catboost-rs-py` | Expose `save_model(format="onnx"/"coreml")`, CV/tuning, GPU predict via existing pyo3 0.29 binding; no new py-side crate. |

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| `prost` + `prost-build` (idiomatic structs) | `protobuf` (rust-protobuf) 4.35.1 + `protobuf-codegen` | If you prefer the rust-protobuf runtime style or hit a prost limitation with proto2 defaults in the CoreML schema. Both are pure-Rust-capable; prost gives cleaner derive-based structs. |
| `protox` (pure-Rust compiler) | `protoc-bin-vendored` 3.2.0 (bundled binary) | If `protox` fails to parse a specific `.proto` feature. Vendored `protoc` avoids a system dependency but adds a binary blob to the build. |
| Classic `TreeEnsembleRegressor/Classifier` (opset ai.onnx.ml v2/3) | New unified `TreeEnsemble` op (opset 5) | Only if a target deployment runtime requires the new op or drops classic-op support — not the case in 2026; onnxruntime still favors classic ops for trees. |
| `PyO3/maturin-action` | `cibuildwheel` | If the project later adds non-Rust native deps needing manylinux repair beyond what maturin handles, or wants cibuildwheel's broader test-in-wheel matrix. |
| Extend Python `benchmark.py` for e2e | A Rust-only harness invoking CatBoost via subprocess CLI | Only if the CatBoost CLI (not the Python package) becomes the reference; unnecessary since `catboost` is already in the oracle venv. |
| Inline `online_hnsw` port | `hnsw_rs` / `instant-distance` / `hora` | Never for parity — these will not reproduce CatBoost's online graph bit-exactly. Only if approximate KNN (non-parity) were ever acceptable, which it is not (≤1e-5 / ε=1e-4 bar). |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `tract-onnx`, `ort` (ONNX Runtime) | These **read/run** ONNX; neither **writes** a tree-ensemble ONNX model. Pulling them adds a large inference dependency that does nothing for export. | `prost` + vendored `onnx.proto` (hand-build `ModelProto`). |
| `candle` for ONNX export | Deep-learning tensor framework; no tree-ensemble ONNX writer; huge dependency. | `prost` + `onnx.proto`. |
| `coreml-rs` 0.5.4 | macOS-only **inference** binding (swift-bridge). Cannot construct/serialize a `.mlmodel`, and would break non-macOS builds. | `prost` + vendored CoreML `.proto`. |
| `rand` 0.10.2 for random-search tuning | A fresh general-purpose RNG breaks reproducibility / oracle parity with CatBoost. | The project's existing deterministic RNG in `cb-train` (same stream as bootstrap/MVS). |
| Third-party HNSW crates | Not bit-exact with CatBoost's online HNSW → perpetuates the KNN estimated-feature residual. | Inline port of upstream `online_hnsw`. |
| The new `ai.onnx.ml` `TreeEnsemble` op (opset 5) | Deprecates the classic ops but has thin runtime support in 2026 and diverges from CatBoost's own output. | Classic `TreeEnsembleRegressor`/`TreeEnsembleClassifier` (+`ZipMap`), opset 2/3. |
| `cb-train` dependency inside `cb-backend` | Documented landmine — feature unification breaks the rocm runtime. | Transcribe any needed CPU reference inline into `cb-backend`. |

## Stack Patterns by Variant

**If keeping `cb-model` lean is the priority (recommended):**
- Put ONNX/CoreML in a new `cb-export` crate depending on `cb-model`.
- Because `prost` + codegen + vendored `.proto` blobs are export-only concerns; `cb-model`'s core stays flatbuffers+serde.

**If minimizing crate count is the priority:**
- Put export in a feature-gated `export` module inside `cb-model` (`export = ["dep:prost"]`).
- Because it avoids a new workspace member — at the cost of protobuf codegen in the core serialization crate.

**If the CI environment cannot guarantee `protoc` on PATH:**
- Use `protox` (pure-Rust) as the `prost-build` compiler front-end.
- Because it removes the system-`protoc` requirement, matching the hermetic flatc/flatbuffers precedent.

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| `prost` 0.14.4 | `prost-build` 0.14.4 | Keep runtime and codegen on the same 0.14 minor. |
| `prost-build` 0.14.4 | `protox` 0.9.1 | `protox` implements the `prost-build` compiler interface; confirm the `protox` release notes name-check `prost-build` 0.14 at implementation time. |
| `criterion` 0.8.2 | edition 2021, MSRV | Dev-dependency only; requires `[[bench]] harness = false`. No effect on the shipped library. |
| `maturin` 1.14.1 | `pyo3` 0.29.0 + `abi3-py312` | Already the project's working combination; per-backend legs differ only by Cargo feature. |
| ONNX `.proto` (onnx 1.23 / ai.onnx.ml opset 2) | onnxruntime, tract | Emit ir_version 3 to match CatBoost; classic tree ops are broadly supported. |
| CoreML `.proto` | pinned `specificationVersion` | Pin to the vendored schema version; verify against the checked-in `Model.proto`. |

## Sources

- `docs/CATBOOST_CORE_DESIGN.md` §Trained Model Representation / Export Formats — export guards (numerical-only, identity scale, oblivious-only), model structure — HIGH (repo reference doc)
- CatBoost upstream `catboost/libs/model/model_export/onnx_helpers.cpp` — ir_version 3, ai.onnx.ml opset 2, TreeEnsembleRegressor/Classifier + ZipMap, producer "CatBoost" — HIGH (source-verified 2026-07-05)
- https://onnx.ai/onnx/operators/onnx_aionnxml_TreeEnsemble.html — new unified `TreeEnsemble` op at ai.onnx.ml opset 5 deprecates classic ops (ONNX 1.22/1.23) — HIGH
- https://onnx.ai/onnx/operators/onnx_aionnxml_TreeEnsembleClassifier.html / …Regressor.html — classic op semantics, deprecation note — HIGH
- https://catboost.ai/docs/en/concepts/apply-onnx-ml — CatBoost ONNX-ML export supports numerical-only, ensemble-of-trees — HIGH
- https://github.com/apple/coremltools/blob/main/mlmodel/format/Model.proto + TreeEnsemble spec — CoreML is protobuf; TreeEnsembleRegressor/Classifier message types — HIGH (Apple source)
- https://apple.github.io/coremltools/mlmodel/index.html — CoreML `.mlmodel` = protobuf, creatable from any protobuf-supported language — HIGH
- `cargo search` 2026-07-05 — prost 0.14.4, prost-build 0.14.4, protox 0.9.1, protoc-bin-vendored 3.2.0, criterion 0.8.2, coreml-rs 0.5.4, pyo3 0.29.0, rand 0.10.2 — HIGH
- `pip index versions maturin` 2026-07-05 — maturin 1.14.1 — HIGH
- https://github.com/PyO3/maturin-action — Rust-optimized wheel CI, cross-compile, abi3 matrix — HIGH
- https://cibuildwheel.pypa.io/en/latest/faq/ — abi3 support + note that Rust wheels avoid manylinux tricks — MEDIUM
- Repo inspection: `crates/catboost-rs-py/Cargo.toml` (pyo3 0.29 abi3-py312), `crates/cb-model/Cargo.toml` (flatbuffers precedent), `bench/`, `benchmark.py` (existing e2e harness importing both catboost + catboost_rs), grep confirming criterion/prost/rand not yet deps — HIGH

---
*Stack research for: catboost-rs v1.2 export + release-readiness additions*
*Researched: 2026-07-05*
