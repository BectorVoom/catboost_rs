# Project Research Summary

**Project:** catboost-rs
**Domain:** Gradient-boosting ML library — full Rust rewrite of CatBoost, multi-backend GPU (CubeCL), dual PyO3 Python bindings, oracle-tested to ≤1e-5
**Researched:** 2026-06-13
**Confidence:** HIGH (grounded in vendored CatBoost C++ source, live crates.io version verification, and first-party CubeCL manual)

## Executive Summary

catboost-rs is one of the hardest categories of ML engineering project: a *numerically-exact rewrite* of a mature, heavily-optimized C++ trainer in a different language, on a different GPU framework, with a different binding layer. The dominant risk is not "will it work" but "will it match." Gradient boosting is a sequential, compounding algorithm — a single ULP divergence in quantization border placement propagates through every subsequent tree, making late-discovered parity failures very expensive to localize. The research consensus is unambiguous: **phase by oracle-passing vertical slices, narrowest first**; do not build multiple components simultaneously without intermediate oracle gates. The entire project architecture must be laid down — workspace, lint rules, oracle harness infrastructure, the RNG port, and the generic CubeCL `R: Runtime` boundary — before any algorithmic code is written.

The recommended stack has one critical version-coupling anchor: `rust-numpy 0.28` transitively pins `pyo3 = ^0.28` (use 0.28.3, not 0.29) and `ndarray = >=0.15, <=0.17` (use 0.17.2). These three versions must move together; any upgrade of one requires all three to be validated simultaneously. The CubeCL backend maps are equally load-bearing: `rocm` feature → `cubecl/hip` → `HipRuntime` (the concrete type is `cubecl::hip::HipRuntime`, not a `RocmRuntime`); `cuda` → `CudaRuntime`; `wgpu` → `WgpuRuntime`; `cpu` → `CpuRuntime`. Backend selection is purely compile-time via a single `cfg`-gated type alias in `cb-compute::backend` — no runtime dispatch anywhere.

The two dominant risk areas are (1) floating-point summation order divergence across language and parallelism boundaries, and (2) the correctness of ordered boosting and ordered CTR, which are CatBoost's defining features and the hardest parity targets. Both require a specific mitigation posture from Phase 0: intermediate oracles at every stage boundary (quantization borders, per-tree splits, per-tree leaf values, raw approximants per iteration), a single audited reduction utility matching the C++ accumulator type and order, and a Rust port of CatBoost's exact PRNG (`TFastRng64`) oracle-validated at the bitstream level before any stochastic algorithm uses it. GPU parity against the C++ CPU oracle at 1e-5 is not achievable on principle; the realistic target is CPU path hits 1e-5 vs C++, GPU path hits a separately-stated looser tolerance vs the Rust CPU path.

## Key Findings

### Recommended Stack

The stack is anchored by three locked triads that must never be broken independently. First, the Python-boundary triad: `pyo3 0.28.3` + `numpy (rust-numpy) 0.28.0` + `ndarray 0.17.2` — rust-numpy is the pin anchor; bumping pyo3 to 0.29 breaks zero-copy NumPy interop until rust-numpy ships a compatible release. Second, the CubeCL backend triad: `cubecl 0.10.0` with `rocm` feature → `cubecl-hip` → `cubecl::hip::HipRuntime` for the GPU test path, `cpu` feature → `CpuRuntime` as the CI correctness and oracle vehicle. Third, the error-handling discipline: `thiserror 2.x` in every library crate, `anyhow 1.x` only in `catboost-py`, oracle harness, and bin targets — never in library public APIs.

**Core technologies:**
- **Rust (≥ 1.85, edition 2024):** Implementation language; edition 2024 is the workspace default.
- **cubecl 0.10.0:** Single `#[cube(launch)]` kernel compiles to every backend; `R: Runtime` generic selected by Cargo feature gives zero-cost switching via monomorphization. `rocm = ["hip"]` alias is load-bearing — the runtime type is `HipRuntime`.
- **pyo3 0.28.3** (pinned, NOT 0.29): Rust↔Python boundary; `abi3-py312` gives a stable-ABI wheel per backend on CPython ≥ 3.12. Held at 0.28.x by rust-numpy dependency.
- **maturin 1.14.0:** Builds per-backend wheels via `--features <backend>`.
- **numpy (rust-numpy) 0.28.0:** Zero-copy NumPy↔ndarray; **the version-pin anchor** for the pyo3/ndarray triad.
- **ndarray 0.17.2:** CPU numeric/array core; pinned at 0.17 by rust-numpy 0.28 range (`>=0.15, <=0.17`).
- **thiserror 2.0.18 / anyhow 1.0.102:** Mandated error split; `thiserror` in library crates, `anyhow` at binding/application/test edge only.
- **rayon 1.12.0:** CPU data-parallelism. Summation order must be deterministic (fixed chunk sizes, fixed reduction order) — `par_iter().sum()` is not parity-safe.
- **proptest 1.11.0 / approx 0.5.1 / rstest 0.26.1:** Oracle test infrastructure; `assert_abs_diff_eq!(got, expected, epsilon = 1e-5)`.
- **serde 1.0.228 + bincode 2.x:** Model serialization (verify bincode 3.x stability before adopting).
- **bytemuck 1.25.0:** POD host↔device byte transfer; zero-copy Arrow/NumPy uploads via `cast_slice`.
- **arrow 59.0.0** (with `pyarrow` feature), **polars 0.54.4**, **pyo3-polars 0.27.0** (tightly coupled to polars — bump together).

### Expected Features

**Must have (table stakes — parity-critical core):**
- Symmetric (oblivious) decision trees — underpins model format, SIMD inference, SHAP traversal; must be the first tree primitive built.
- Feature quantization (`GreedyLogSum` border selection) — gates all downstream splits and GPU data layout.
- Pool abstraction (float/cat/text/embedding columns, label, weights, group_id, pairs, baseline).
- Plain boosting train loop — Logloss + RMSE as the first oracle targets.
- Overfitting detection and early stopping.
- Feature importance (`PredictionValuesChange`) and SHAP values (Regular).
- Model serialization (native `.cbm` FlatBuffers format, cross-version compatible).
- Rust Builder API + PyO3 dual Python API (sklearn-compatible + CatBoost-native Pool/parameter names).
- Oracle test harness wired end-to-end before tree-building begins.

**Should have (signature differentiators — parity-critical, highest complexity):**
- **Ordered boosting** (`EBoostingType::Ordered`) — CatBoost's defining anti-leakage algorithm; VERY HIGH complexity; the hardest oracle target.
- **Ordered target statistics (ordered CTR)** — leakage-free categorical encoding; shares permutation machinery with ordered boosting.
- **Feature combinations (tensor CTRs)** — automatic categorical crosses.
- One-hot encoding for low-cardinality categoricals (`one_hot_max_size` threshold).
- Full loss/metric matrix: multiclass, all regression variants (MAE, Quantile, LogCosh, Huber, Poisson, Tweedie, etc.).
- Ranking: YetiRank, PairLogit, QueryRMSE, LambdaMart + NDCG, MAP, MRR.
- Text features (BoW, NaiveBayes, BM25 calcers).
- Embedding features (LDA, KNN calcers).
- GPU training via CubeCL (`rocm` first, then `wgpu`/`cuda`).
- Uncertainty estimation, custom objectives, monotone constraints, SHAP interaction values.

**Defer (anti-features — explicitly out of scope per PROJECT.md):**
- C API / C FFI layer, R/JVM/.NET/Node.js bindings, CLI application.
- Model export to CoreML/ONNX/PMML/C++ source (defer; native `.cbm` + JSON for interop).
- Distributed multi-node training, mobile/embedded, streaming training.
- Alternative grow policies (Lossguide, Depthwise, Region) — defer after symmetric-tree core.

### Architecture Approach

The architecture mirrors CatBoost's own `libs/data` / `train_lib` / `libs/model` / `cuda/` separation but inverts one dependency: a single backend-agnostic `cb-data` feeds both the training orchestrator and the compute layer, so `cb-core` never branches on backend. The key invariant: **only `cb-compute::backend` ever names a backend runtime type**. The public `catboost-rs` crate fixes `R = cb_compute::SelectedRuntime` once via a `cfg`-gated type alias; all other crates are generic over `R: Runtime` or fully backend-agnostic.

**Major components (in dependency/build order):**
1. **`cb-data`** — `Pool`, `QuantizedPool` (`u8`/`u16` bin indices, columnar SoA), border selection, CTR encoding, zero-copy NumPy/Arrow/Polars ingestion. Leaf crate.
2. **`cb-compute`** — CubeCL kernels generic over `R: Runtime` and `F: Float`; histogram, gradient/hessian, scan, reductions. The only crate naming backend types. Feature-gated `cpu`/`wgpu`/`cuda`/`rocm`.
3. **`cb-core`** — Boosting loop, oblivious-tree structure search, leaf estimation, loss registry, bootstrap/sampling, ordered-boosting bookkeeping. Generic over `R: Runtime`; drives `cb-compute`.
4. **`cb-model`** — Model representation, `.cbm` FlatBuffers serialization, CPU inference/apply, SHAP/fstr.
5. **`catboost-rs`** — Public Rust Builder API, typed error enum. Fixes `R = SelectedRuntime`.
6. **`cb-python`** — PyO3 wrappers, maturin packaging, dual sklearn + CatBoost-native surfaces, `anyhow` error conversion.

### Critical Pitfalls

1. **Parity as end-gate instead of per-stage invariant** — Intermediate oracles must exist at every stage boundary (borders, splits, leaf values, approx per iteration) before tree-building begins. A border divergence in tree 1 compounds across 1000 trees; without stage-level oracles there is no way to localize the failure.

2. **Floating-point summation order divergence (the #1 parity killer)** — CatBoost accumulates in `double` with specific blocking/order (`TKahanAccumulator` in some paths). Rust's `.iter().sum()` and Rayon's `par_iter().sum()` produce different results. One audited reduction utility per stage matching C++ accumulator type and order must be the only way to sum in the codebase. Parallel reductions must use fixed chunk sizes and deterministic merge order.

3. **Quantization / border-selection mismatch** — `GreedyLogSum` has subtle tie-breaking, NaN handling, and `<`/`<=` assignment semantics. Port only this algorithm for v1 and oracle-validate the exact border set per feature (including NaN/duplicate columns) before any downstream work.

4. **Ordered boosting and CTR implemented subtly wrong** — The prefix boundary off-by-one reintroduces leakage silently; the model still trains plausibly. Sequence: plain-boosting oracle-passes first, then ordered; add per-object target-statistic intermediate oracle, not just final predictions.

5. **RNG non-reproducibility** — CatBoost's `TFastRng64` (SplitMix/xorshift) produces permutations and sampling draws. Rust's `StdRng`/ChaCha produces a different bit sequence from the same seed. Port `TFastRng64` exactly and oracle-test the raw bitstream before any stochastic algorithm is written.

6. **CPU vs GPU divergence beyond tolerance** — GPU float reductions (atomics ordering, FMA contraction, AMD wavefront-64 vs NVIDIA warp-32) make 1e-5 GPU-vs-C++-CPU parity structurally unachievable. Set the GPU target as: CPU hits 1e-5 vs C++; GPU hits a separately-stated looser tolerance vs Rust CPU. CubeCL ROCm/HIP backend is WIP with raw bindgen bindings — spike before Phase 6.

7. **PyO3 zero-copy buffer lifetime / GIL safety** — Copy/quantize into Rust-owned storage under the GIL before releasing it for training. Never borrow a NumPy `&[f32]` across `Python::allow_threads`. Design for `Py_GIL_DISABLED` (free-threaded Python 3.13t+) from the start.

## Implications for Roadmap

The forced build order: workspace + oracle infra → data layer + foundational primitives (quantization, RNG, reduction utilities) → CPU training core (plain boosting) → high-risk parity slice (ordered boosting + CTR) → full loss/feature matrix → GPU backends → Python bindings. CPU must be fully oracle-passing before GPU is touched. Python bindings come after the Rust Builder API is stable. Each phase must be oracle-passing before the next begins.

### Phase 0: Workspace, Infrastructure, and Oracle Harness

**Rationale:** Lint discipline and oracle infrastructure cannot be retrofitted; they must be the first commits (Pitfalls 1, 10, 12).

**Delivers:**
- Cargo workspace skeleton with all crates stubbed; workspace `Cargo.toml` with all version pins
- Lint configuration in library crates: `#![deny(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing, clippy::panic, clippy::as_conversions)]`; CI check that `anyhow` is absent from core crate non-test code
- Frozen oracle fixture infrastructure: pre-generated CatBoost reference outputs (`thread_count=1`, pinned seed + version); committed fixtures; no live CatBoost rebuild in PR CI
- Intermediate oracle tooling: mechanism to compare borders-per-feature, per-tree split, leaf values, and raw approximants — not only final predictions
- `TFastRng64` Rust port, bitstream-oracle-tested against the C++ generator for a fixed seed

**Avoids:** Pitfalls 1 (per-stage gates), 10 (oracle CI flakiness), 12 (error-boundary / no-unwrap from line 1)

**Research flag:** Standard patterns; no per-phase research needed.

### Phase 1: Data Layer — Pool, Quantization, CTR Encoding

**Rationale:** `cb-data` is the leaf crate everything depends on. Quantization is the foundational oracle hot-spot — wrong borders poison every downstream tree (Pitfall 3). The ingestion model (copy-in for training, zero-copy for prediction) must be decided here to inform Phase 7's GIL-safety design (Pitfall 7).

**Delivers:**
- `Pool`: float/cat/text/embedding columns, label, weights, group_id, pairs, baseline scaffolding
- `QuantizedPool`: `u8`/`u16` bin index columnar SoA storage; pre-allocated buffers reused across rounds
- `GreedyLogSum` border selection: per-feature border-set oracle-validated including NaN/duplicate columns; `NanMode` exact; `<`/`<=` assignment semantics correct
- Categorical perfect-hash / CTR encoding table stubs (full ordered CTR in Phase 3)
- Zero-copy NumPy ingestion (`bytemuck::cast_slice`); copy-in path for training
- Arrow/Polars ingestion with contiguity + dtype validation
- Single audited reduction utility matching C++ `double` accumulator and order

**Avoids:** Pitfalls 3 (border mismatch), 9 (SoA layout + borrow-checker strategy), 7 (GIL ingestion model)

**Research flag:** Standard patterns; `grid_creator.cpp` in vendored source is the reference.

### Phase 2: CPU Training Core — Plain Boosting, Oblivious Trees, Leaf Estimation

**Rationale:** The generic `R: Runtime` + `F: Float` boundary must be designed here, not retrofitted. Plain boosting validates the core algorithm before the high-risk ordered algorithms are added (Pitfall 4 mitigation).

**Delivers:**
- `cb-compute` with `cpu` backend only: `#[cube(launch)]` histogram/gradient/scan/reduction kernels generic over `F: Float` and `R: Runtime`; `backend.rs` type alias (`SelectedRuntime = CpuRuntime`)
- `cb-core`: boosting loop generic over `R: Runtime`; oblivious-tree structure search; leaf value estimation (Gradient, Newton, Exact); bootstrap/sampling; loss registry (Logloss, RMSE)
- Overfitting detector, early stopping, eval-set metric logging
- First end-to-end CPU train → predict cycle
- Per-tree split and leaf-value intermediate oracles passing ≤1e-5 vs C++

**Avoids:** Pitfalls 2 (audited reduction utility enforced throughout), 9 (SoA + `split_at_mut`; no `.clone()` in hot loop)

**Research flag:** Standard patterns; CubeCL generics manual covers the `R: Runtime` launch pattern.

### Phase 3: Model Layer, Serialization, and Oracle Lock

**Rationale:** Completes the first full vertical slice (train → serialize → predict) end-to-end oracle-passing before any widening. Serialization parity (loading a `.cbm` produced by upstream CatBoost) must be validated here.

**Delivers:**
- `cb-model`: oblivious tree + split representation, Model struct (trees + CTR tables + scale/bias)
- `.cbm` FlatBuffers serialization/deserialization parity; round-trip tested; loading of reference upstream `.cbm` files validated
- CPU inference/apply path (independent of `cb-compute`; no GPU toolchain required)
- SHAP values (Regular `EShapCalcType`); feature importance (`PredictionValuesChange`)
- `catboost-rs` public Builder API: `CatBoostBuilder::new().fit(&pool) -> Model`; typed error enum
- Full end-to-end oracle: numeric-only binary classification and regression ≤1e-5 vs C++

**Avoids:** Pitfall 1 (narrow vertical slice fully oracle-locked before widening), Pitfall 11 (narrowest-first before high-risk slice)

**Research flag:** Standard patterns; FlatBuffers format is in the vendored source.

### Phase 4: Ordered Boosting, Ordered CTR, and Categorical Features (High-Risk Parity Slice)

**Rationale:** The highest-risk parity target. Ordered boosting and ordered CTR share the permutation machinery seeded by the `TFastRng64` port from Phase 0. A wrong prefix boundary reintroduces leakage silently. Requires the closest line-by-line reading of `approx_calcer.cpp` and `online_ctr.*`.

**Delivers:**
- Multi-permutation fold infrastructure (`fold_count` permutations, exact `TFold`-equivalent bookkeeping)
- `EBoostingType::Ordered`: exact prefix boundary, exact prior formula `(sumTarget + prior) / (sumCount + priorWeight)`, per-object target-statistic intermediate oracle passing
- Ordered CTR: `ECtrType { Borders, Buckets, BinarizedTargetMeanValue, FloatTargetMeanValue, Counter, FeatureFreq }` with priors
- One-hot encoding (`one_hot_max_size` threshold)
- Feature combinations (tensor CTRs): `SimpleCtrs`, `CombinationCtrs`; `max_ctr_complexity` control
- Train metrics checked for leakage signatures (suspiciously good train metrics = prefix boundary bug)

**Avoids:** Pitfall 4 (ordered boosting subtly wrong), Pitfall 5 (TFastRng64 already oracle-tested)

**Research flag:** NEEDS DEEPER RESEARCH — the least-documented, most intricate algorithmic area. Before planning this phase, do a line-by-line read of `approx_calcer.cpp` + `online_ctr.*` and design the intermediate oracle schema (which per-object values to extract and compare) before writing any implementation code.

### Phase 5: Full Loss and Feature Parity

**Rationale:** With the core architecture oracle-locked, widening the loss/metric matrix and adding text/embedding/ranking is additive. Each new loss or feature type passes its own oracle before the next is added.

**Delivers:**
- Full loss matrix: multiclass, all regression variants (MAE, Quantile, LogCosh, Huber, Poisson, Tweedie, MAPE, MSLE, Lq, Expectile, etc.)
- Ranking: YetiRank, PairLogit, QueryRMSE, LambdaMart, StochasticRank; NDCG, MAP, MRR, ERR; Pool group_id/subgroup_id/pairs
- Text features: tokenization → BoW, NaiveBayes, BM25 calcers
- Embedding features: LDA, KNN calcers
- Uncertainty estimation: RMSEWithUncertainty, virtual ensembles
- Advanced fstr: LossFunctionChange, ShapInteractionValues, PredictionDiff, SAGE
- Monotone constraints, feature penalties, feature selection
- Custom objectives/metrics (Rust trait + Python callback bridge)

**Avoids:** Pitfall 11 (each feature type oracle-validated individually; no simultaneous multi-feature builds)

**Research flag:** Standard patterns for each loss type; vendored source is the reference.

### Phase 6: GPU Backends via CubeCL

**Rationale:** GPU is purely additive on top of a fully oracle-passing CPU implementation. The generic `R: Runtime` boundary from Phase 2 means `cb-core` and `cb-model` require no changes. The GPU tolerance target is explicitly different from the CPU target — get sign-off before starting.

**Delivers:**
- `cb-compute` with `rocm` backend: all kernels validated on AMD hardware (wavefront-64; no warp-size assumptions; CubeCL plane/subgroup abstractions)
- `wgpu` backend for dev machines without ROCm/CUDA
- `cuda` backend: compile-gated, untested locally
- Documented GPU tolerance: `rocm` results within a separately-stated epsilon vs Rust CPU path (not vs C++ CPU oracle)
- `cubecl-hip-sys` HIP version matched to installed ROCm on the GPU test machine

**Avoids:** Pitfall 6 (separate GPU tolerance, wavefront-64 correctness, CPU-first sequencing)

**Research flag:** NEEDS DEEPER RESEARCH — CubeCL ROCm/HIP backend is explicitly WIP with raw bindgen bindings. Before planning: spike to confirm which CubeCL kernel patterns (histogram atomics, prefix scan, reduction) are functional on `rocm` at cubecl 0.10.0 vs known gaps; validate wavefront-64 reduction determinism; confirm `cubecl-hip-sys` HIP version requirements against the test machine.

### Phase 7: Python Bindings, Dual API, and Packaging

**Rationale:** Python bindings are strictly downstream of a stable `catboost-rs` Builder API. GIL-safety data layout decisions were made in Phase 1. The per-backend wheel matrix must be scoped explicitly to avoid combinatorial CI explosion.

**Delivers:**
- `cb-python`: `Pool`, `CatBoostClassifier`, `CatBoostRegressor`, `CatBoostRanker` pyclasses
- sklearn-compatible API: `fit`, `predict`, `predict_proba`, `score`, `get_params`/`set_params`; `check_estimator` CI test
- CatBoost-native API: full parameter-name parity with upstream, exact default values
- NumPy/Pandas/Arrow/Polars input with dtype + contiguity validation
- Typed `thiserror` → specific Python exception mapping with actionable messages
- Per-backend wheels via `maturin build --release --features <backend>`; v1 matrix: `cpu` + `rocm` minimum
- System-prerequisite documentation per wheel
- Free-threaded-aware design: no GIL reliance for buffer safety

**Avoids:** Pitfalls 7 (GIL/zero-copy safety), 8 (wheel/ABI matrix complexity)

**Research flag:** NEEDS RESEARCH — confirm current PyO3/maturin `abi3`/`abi3t` status for Python 3.12–3.15 (PEP 803: `abi3t` only on 3.15+; free-threaded 3.12–3.14 requires version-specific wheels) before committing to an ABI strategy.

### Phase Ordering Rationale

- Workspace/infra before any algorithm: lint discipline and oracle infrastructure cannot be retrofitted.
- Quantization and RNG before tree-building: foundational; bugs here poison everything above.
- CPU backend before GPU: GPU is a performance layer on a correct algorithm; the generic `R: Runtime` boundary from Phase 2 makes GPU activation purely additive.
- Plain boosting before ordered: establishes a clean oracle baseline; isolates the permutation complexity.
- Rust API stable before Python bindings: Python is a thin wrapper; premature binding work churn when the Rust API is still changing.
- Narrow vertical slices, oracle-passing, before widening: the only reliable sequencing for a numerically-exact rewrite.

### Research Flags Summary

| Phase | Research Needed | Reason |
|-------|----------------|--------|
| Phase 0 | No | Standard workspace/tooling patterns |
| Phase 1 | No | GreedyLogSum is in vendored source; standard data layout |
| Phase 2 | No | CubeCL CPU backend is documented in the manual |
| Phase 3 | No | FlatBuffers format is in vendored source |
| Phase 4 | **YES — high priority** | Ordered boosting/CTR: least-documented, most intricate; line-by-line analysis + intermediate oracle design required before implementation |
| Phase 5 | No | Each loss/feature type is additive; vendored source is the reference |
| Phase 6 | **YES — high priority** | CubeCL ROCm/HIP WIP; wavefront-64 semantics; kernel coverage spike needed before writing any GPU code |
| Phase 7 | **YES** | PyO3/maturin abi3t status evolving; free-threaded wheel support needs confirmation before ABI commitment |

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All versions live-verified against crates.io 2026-06-13; pyo3/numpy/ndarray triad verified from dependency metadata; CubeCL backend mapping verified from cubecl 0.10.0 feature graph and first-party manual |
| Features | HIGH | Grounded in vendored CatBoost C++ source; enum values read directly from headers |
| Architecture | HIGH | Crate decomposition mirrors CatBoost's own lib separation; CubeCL generic-runtime boundary verified from manual generics examples |
| Pitfalls | HIGH (algorithmic/Rust), MEDIUM (GPU/bindings) | Numerical-parity pitfalls grounded in vendored source and float-reproducibility literature; CubeCL/ROCm specifics are MEDIUM (WIP, fast-moving); PyO3 free-threading ABI is MEDIUM (evolving PEP 779/803) |

**Overall confidence:** HIGH for the CPU core path; MEDIUM for GPU and Python packaging details (both require per-phase research spikes before execution).

### Gaps to Address

- **CubeCL ROCm/HIP kernel trait coverage:** WIP `cubecl-hip` has partial trait/method support. Before Phase 6, spike to confirm histogram atomics, prefix scan, and reduction kernels are functional on `rocm` at cubecl 0.10.0; match `cubecl-hip-sys` to installed HIP version.
- **GPU tolerance specification:** The exact epsilon for GPU-vs-Rust-CPU parity is unspecified. Before Phase 6, establish a concrete tolerance (e.g., 1e-4 or 1e-3) and get explicit sign-off.
- **PyO3/maturin abi3t status:** PEP 803 targets Python 3.15+; for 3.12–3.14 free-threaded builds require version-specific wheels. Before Phase 7, confirm maturin's current support and scope the v1 Python version matrix.
- **Ordered boosting intermediate oracle schema:** Before Phase 4, design the exact per-object intermediate values to extract from the C++ reference (from `approx_calcer.cpp` + `online_ctr.*`) and the comparison invariants.
- **bincode 3.x stability:** 3.0.0 exists on crates.io but stability is unconfirmed. Before Phase 3 model serialization, verify or pin to bincode 2.x.

## Sources

### Primary (HIGH confidence)
- Vendored CatBoost C++ source (`catboost-master/`) — `enums.h`, `ctr_type.h`, `cat_feature_options.h`, `grid_creator.cpp`, `approx_calcer.cpp`, `scoring.cpp`, `fold.cpp`, `learn_context.cpp`, `online_ctr.*`, `libs/data/`, `libs/model/`, `libs/fstr/`, `python-package/catboost/core.py`
- crates.io live API (2026-06-13) — version verification for all pinned crates
- crates.io dependency metadata — `numpy 0.28` → `pyo3 ^0.28`, `ndarray >=0.15,<=0.17`; cubecl 0.10.0 feature graph
- CubeCL manual (`/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/`) — `Cubecl_generics.md`, ROCm backend handling, `ZERO_COPY_ARROW_CUBECL.md`
- `.planning/PROJECT.md`, `.planning/codebase/ARCHITECTURE.md`, `STACK.md`, `STRUCTURE.md`, `CONCERNS.md`, `TESTING.md`

### Secondary (MEDIUM confidence)
- tracel-ai/cubecl + cubecl-hip-sys GitHub — ROCm/HIP WIP status; HIP-version-based binding scheme (since May 2025)
- PyO3/rust-numpy releases — rust-numpy tracks PyO3 minor versions 1:1
- PyO3 free-threading guide — Stable ABI unavailable for free-threaded builds; PEP 803 (abi3t, 3.15+)
- Float reproducibility literature (IEEE/TOMS, arXiv:2408.05148) — non-associativity and parallel-reduction variability

### Tertiary (LOW confidence, needs validation)
- bincode 3.x stability — unverified; treat as LOW until confirmed before Phase 3

---
*Research completed: 2026-06-13*
*Ready for roadmap: yes*
