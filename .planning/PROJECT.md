# catboost-rs

## What This Is

A full Rust rewrite of the CatBoost gradient boosting library, targeting complete feature parity with the original C++ implementation. It exposes first-class APIs in both Rust (using Rust-native patterns like the Builder pattern) and Python (via PyO3/maturin), where the Python surface is **both** scikit-learn compatible *and* CatBoost-native (Pool, parameter names) for drop-in migration. GPU acceleration is provided through CubeCL with backends switchable at compile time via Cargo features.

It is for two audiences: Rust developers who want to embed a memory-efficient gradient booster directly, and Python ML practitioners who want a drop-in replacement for CatBoost in existing scikit-learn or CatBoost workflows.

## Core Value

A memory-efficient, Rust-native CatBoost implementation that achieves verifiable feature parity with the original (oracle-tested to within 10⁻⁵), embeddable directly in Rust and droppable into both scikit-learn and existing CatBoost Python pipelines.

## Current Milestone: v1.1 GPU Performance

**Goal:** Full CUDA device-resident training parity — move the entire boosting inner loop onto the GPU (not just derivatives), reaching speed parity with official CatBoost GPU while preserving ≤1e-5 correctness.

**Target features:**
- A `Runtime` trait seam for on-device tree growth — wire the existing-but-unused `grow_boosting_pass` (`crates/cb-backend/src/gpu_runtime/mod.rs:1890`) into `cb_train::train`, keep training data device-resident across iterations (no per-tree re-upload)
- A device-resident **compressed index** (cindex) + a from-scratch **CubeCL device-primitive library** (scan / segmented-scan, reduce / reduce-by-key, radix sort + stable 1-bit reorder, fill/transform, compression, partition-update, stat aggregation — there is **no CUB in CubeCL**) — the foundation every histogram/scoring kernel stands on
- Device-resident histogram build (incl. depth>1 partition-aware + subtraction trick), split scoring, BestSplit, partition/leaf-assignment, and leaf-value computation (Newton **and** Exact weighted-quantile)
- Full grow-policy coverage: SymmetricTree/oblivious **plus** Depthwise / Lossguide / Region non-symmetric trees
- Full loss/sampling coverage on device: RMSE/Logloss → Newton der2, Cosine score fn, MVS (default GPU sampling) + bootstrap/random-strength, CTR/categoricals, PairLogit + query/listwise ranking (QueryRMSE/SoftMax/CrossEntropy/YetiRank/PFound-F) with a batched pairwise Cholesky solver, multiclass / multi-target / uncertainty, ordered boosting, Langevin/SGLB noise
- Speed benchmark harness vs official CatBoost GPU on a Kaggle CUDA notebook, with a per-phase speed check from the first GPU phase to the last

**Key context:** Correctness is developed + smoke-tested in-env on AMD/ROCm (CubeCL kernels are portable cuda/rocm/wgpu from one source), but **all GPU kernel oracles — correctness AND speed — are validated on Kaggle CUDA** (no NVIDIA in-env; ROCm is not a gate). The authoritative reference for the full device kernel surface being reimplemented is **`CATBOOST_CUDA_KERNELS_DESIGN.md`** (the complete upstream CUDA training-kernel map — 79 `.cu` + 77 `.cuh` across 9 kernel directories, host/device splits, data types, algorithms); every v1.1 phase cites it. Root-cause analysis of the >20× gap: `.planning/notes/gpu-training-host-light-root-cause.md`. **Landmine:** never add a `cb-train` dependency to `cb-backend` — feature unification breaks the rocm runtime; transcribe CPU references inline.

**Current state (2026-07-05):** All 5 phases of v1.1 (10–14) are executed. **Phase 14 complete — BENCH-03 signed off** (`bench/BENCH-03-SIGNOFF.md`, verdict `BENCH-03: PASS`): every aggregated device row runs **23.9×–42.1× faster than the pre-Phase-10 host-light CPU baseline** on Tesla P100, reversing the original >20× device-slower-than-CPU gap, with CUDA correctness (44 device self-oracle tests, ALL-PASS) gated before any speed number. **Standing debt carried to milestone-close audit** (formal override recorded in `14-VERIFICATION.md`): Phase-10 (depth-1) and Phase-11 (depth-6) BENCH-02 Kaggle runs were never executed and `GPUT-14` (ε=1e-4 correctness gate) is still `Pending` — the aggregate stitches the committed Phase-12/13 numbers only. Next: `/gsd-complete-milestone` (resolve GPUT-14 / Phase-10-11 BENCH-02 debt or accept as delivered).

## Requirements

### Validated

<!-- Shipped and confirmed valuable. -->

- ✓ CPU gradient-boosting training core — plain + ordered boosting, symmetric oblivious + non-symmetric trees, four leaf-estimation methods (Gradient/Newton/Exact/Simple), bootstrap/sampling, regularization (`l2_leaf_reg`/`random_strength`/`bagging_temperature`), overfitting detection / early stopping, per-iteration eval-set metrics, automatic learning-rate selection — oracle-locked ≤10⁻⁵ — v1.0 (Phases 3–5)
- ✓ Full loss / metric matrix — regression, binary, multiclass/multilabel, six ranking losses (YetiRank(/Pairwise), PairLogit(/Pairwise), QueryRMSE, QuerySoftMax, LambdaMart, StochasticRank), ranking metrics, score functions, uncertainty estimation — oracle-locked ≤10⁻⁵ — v1.0 (Phase 6.1–6.4)
- ✓ Categorical handling — ordered target statistics / CTR, one-hot, feature combinations (tensor CTRs) — v1.0 (Phase 5)
- ✓ Text and embedding feature support — BoW/NaiveBayes/BM25, LDA, KNN vote (brute-force-exact; see HNSW gap below) — v1.0 (Phase 6.5)
- ✓ SHAP value computation — v1.0 (Phase 4/6.6)
- ✓ Model serialization — `.cbm`/`.json` save/load, cross-version reproduce ≤10⁻⁵ — v1.0 (Phase 4)
- ✓ Multi-backend GPU execution via CubeCL — `cuda`/`rocm`/`wgpu`/`cpu` Cargo-feature-switched, generic runtime (no dispatch overhead), **structural** parity (rocm-validated, ε=1e-4 vs CPU) — v1.0 (Phase 7)
- ✓ Rust Builder-pattern API — v1.0
- ✓ Python bindings (PyO3 + maturin) — dual sklearn + CatBoost-native surface, NumPy/Pandas/Arrow/Polars ingest, per-backend wheels — v1.0 (Phase 8)
- ✓ Modular feature-gated Cargo workspace — v1.0 (Phase 1)
- ✓ Oracle test suite + rocm GPU test execution — v1.0

### Active

<!-- Carried forward from v1.0. The current milestone's scope is in "## Current Milestone" below. -->

- [ ] **GPU performance parity** — GPU training shipped as a derivatives-only MVP; the tree-growth inner loop still runs host-side (>20× slower than official CatBoost GPU). Move the full inner loop on-device. _(Next milestone — see Current Milestone)_
- [ ] **FEAT-07** — KNN estimated-feature bit-exact parity via an online-HNSW port (~832 LOC); shipped with brute-force-exact calcer that diverges from upstream's approximate HNSW. _(Deferred backlog — Phase 9)_

### Out of Scope

<!-- Explicit boundaries. Includes reasoning to prevent re-adding. -->

- C API / C FFI layer — PyO3 direct bindings only; no CAPI surface needed
- Mobile / embedded targets — desktop and server workloads only
- Real-time streaming / online training — batch training only
- R and CLI interfaces — Rust and Python only for this milestone

## Context

- **Reference implementation vendored:** the original CatBoost C++ source is present at `catboost-master/` and has been analyzed into `.planning/codebase/` (ARCHITECTURE, STACK, STRUCTURE, CONVENTIONS, TESTING, INTEGRATIONS, CONCERNS). It serves as the algorithmic reference and the oracle for parity testing — not as our codebase (the Rust implementation is greenfield).
- **GPU kernel design reference (v1.1):** `CATBOOST_CUDA_KERNELS_DESIGN.md` documents the complete upstream CUDA **training** kernel surface (79 `.cu` + 77 `.cuh` across `cuda_lib/kernel`, `cuda_util/kernel`[+`sort`], `methods/kernel`, `methods/greedy_subsets_searcher/kernel`, `targets/kernel`, `gpu_data/kernel`, `ctrs/kernel`, `models/kernel`, plus §7 inference evaluator + CUDA wrapper infra) — per-file processing flow, host/device split, I/O data types, and algorithm. It is the authoritative map for the v1.1 device-resident reimplementation (CubeCL, not the raw CUDA); it describes the *original* engine, not our target code.
- **Oracle strategy:** expected values are generated by running the original CatBoost on the same randomly generated inputs; our output must match to within absolute error 10⁻⁵. This requires CatBoost available in the test harness.
- **CubeCL** provides a Rust-native GPU compute abstraction. Backends are selected at compile time via Cargo features (`cuda` — untestable locally, `rocm`, `wgpu`, `cpu`). The runtime is parameterized via generics so backend switching carries no runtime dispatch cost.
- **Python packaging:** users install the backend-specific wheel matching their hardware. PyO3 + maturin handle the Rust→Python boundary. Python ≥ 3.12.
- **Error handling:** `thiserror` for library-level errors, `anyhow` for application/binding-level. `unwrap()` is strictly prohibited in production code.
- **Test design:** oracle testing with randomly generated inputs; source and test code strictly separated (no inline `#[cfg(test)]` mixed with production logic). GPU tests run exclusively on the `rocm` backend.

## Constraints

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

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Full CatBoost feature parity as v1 target | User wants a true drop-in replacement, not a subset | ✓ Good (v1.0) |
| CubeCL for GPU kernels with generic runtime | Rust-native abstraction spanning cuda/rocm/wgpu in one codebase, zero-cost backend switching via generics | ✓ Good (v1.0) |
| PyO3 + maturin (no CAPI) | Simplest correct Rust→Python path; avoids an unsafe C ABI layer; enables per-backend wheels | ✓ Good (v1.0) |
| Dual Python API (sklearn + CatBoost-native) | Maximizes compatibility — drop into sklearn pipelines AND migrate existing CatBoost code unchanged | ✓ Good (v1.0) |
| Feature-gated backend crates | Users compile/install only what their hardware supports | ✓ Good (v1.0) |
| Oracle testing vs original CatBoost outputs | Proves algorithmic parity with the reference, not just internal self-consistency | ✓ Good (v1.0) |
| thiserror + anyhow error strategy | thiserror for clean library API errors; anyhow for ergonomic propagation at bindings/app level | ✓ Good (v1.0) |
| Vendored catboost-master as reference + oracle | Single source of truth for both algorithm behavior and expected test values | ✓ Good (v1.0) |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-07-05 — Phase 14 complete; BENCH-03 signed off (device path 23.9×–42.1× vs host-light baseline, `BENCH-03: PASS`). All v1.1 phases (10–14) executed; GPUT-14 / Phase-10-11 BENCH-02 remain milestone-close standing debt.*
