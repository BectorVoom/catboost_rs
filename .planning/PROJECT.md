# catboost-rs

## What This Is

A full Rust rewrite of the CatBoost gradient boosting library, targeting complete feature parity with the original C++ implementation. It exposes first-class APIs in both Rust (using Rust-native patterns like the Builder pattern) and Python (via PyO3/maturin), where the Python surface is **both** scikit-learn compatible *and* CatBoost-native (Pool, parameter names) for drop-in migration. GPU acceleration is provided through CubeCL with backends switchable at compile time via Cargo features.

It is for two audiences: Rust developers who want to embed a memory-efficient gradient booster directly, and Python ML practitioners who want a drop-in replacement for CatBoost in existing scikit-learn or CatBoost workflows.

## Core Value

A memory-efficient, Rust-native CatBoost implementation that achieves verifiable feature parity with the original (oracle-tested to within 10⁻⁵), embeddable directly in Rust and droppable into both scikit-learn and existing CatBoost Python pipelines.

## Current Milestone: v1.2 Parity Completion & Release Readiness

**Goal:** Close the remaining CatBoost surface described in `docs/CATBOOST_CORE_DESIGN.md` and `docs/CATBOOST_CUDA_KERNELS_DESIGN.md`, discharge the v1.1 standing debt, and make catboost-rs adoption-ready — held to the ≤10⁻⁵ CPU / ε=1e-4 GPU parity bar.

**Target features:**
- **Debt & hardening** — GPUT-14 aggregate ε=1e-4 Kaggle CUDA correctness sign-off; Phase-10/11 BENCH-02 speed rows (finish the BENCH-03 aggregate); RV-13-01..04 latent parity hazards; FEAT-07 online-HNSW port for bit-exact KNN estimated-feature parity.
- **Model export** — ONNX and CoreML export (from `.cbm`/`.json` today).
- **Extended feature importance** — Interaction, LossFunctionChange, partial-dependence (SHAP + basic fstr already shipped).
- **Orchestration layer** — first-class cross-validation, hyperparameter tuning (grid/random), snapshot/resume, standalone calc_metrics / eval_result.
- **GPU inference evaluator** — device-side predict (v1.1 delivered device training only).
- **Adoption / DX** — end-to-end benchmark vs official CatBoost (accuracy + speed on real datasets), PyPI release readiness (per-backend wheels, CI, versioning), documentation + runnable Rust/Python examples, real-world dataset validation suite.

**Deferred / out of scope for v1.2:** distributed & multi-node training (MPI master/worker, multi-GPU); PMML and C++/Python code export.

**Key context carried:** Kaggle CUDA remains the sole authoritative GPU oracle (ROCm in-env = non-gating smoke); never add a `cb-train` dependency to `cb-backend`; CubeCL for all GPU kernels.

## Current State

**Shipped:** v1.1 GPU Performance (2026-07-05) — Phases 10–14, 36 plans, 25 requirements. The boosting inner loop moved from a derivatives-only host-light MVP onto a fully device-resident CubeCL training path: a from-scratch device-primitive library (no CUB) with a deterministic reduction, a bit-packed device-resident compressed index, a per-fit `GpuTrainSession` keeping the quantized matrix/gradients/approx resident across iterations (no per-tree re-upload or `der1` read-back), depth-1→depth-6 partition-aware histograms with the subtraction trick + Newton der2, and full device coverage of grow policies (Depthwise/Lossguide/Region), Exact weighted-quantile leaves, bootstrap/random-strength/MVS sampling, CTR/categoricals, PairLogit + batched device Cholesky, query/listwise ranking, multiclass/multi-target/uncertainty, ordered boosting, and Langevin/SGLB noise — each behind an `Ok(None)`→CPU per-fit fallback. **BENCH-03: PASS** — 23.9×–42.1× vs the pre-Phase-10 host-light CPU baseline on Tesla P100, reversing the original >20× gap, with CUDA correctness (44 device self-oracle tests, ALL-PASS) gated first.

**Standing debt (accepted at close, formal override in `14-VERIFICATION.md`):** `GPUT-14` (the milestone-wide ε=1e-4 Kaggle CUDA correctness sign-off row) is still `Pending` — coverage is evidenced per-family in-env (≤1e-4 on gfx1100) and on committed P100 runs, not as one aggregate; and the Phase-10 (depth-1) + Phase-11 (depth-6) BENCH-02 Kaggle speed rows were never executed, so the BENCH-03 aggregate stitches the committed Phase-12/13 numbers only. See MILESTONES.md → Known Gaps and STATE.md → Deferred Items.

**Key context (carried):** CubeCL kernels are portable cuda/rocm/wgpu from one source; correctness is developed + smoke-tested in-env on AMD/ROCm (**not a gate**) and signed off on **Kaggle CUDA** (no NVIDIA in-env). The authoritative reference for the reimplemented device kernel surface is **`CATBOOST_CUDA_KERNELS_DESIGN.md`** (79 `.cu` + 77 `.cuh` across 9 kernel directories). Root-cause of the original >20× gap: `.planning/notes/gpu-training-host-light-root-cause.md`. **Landmine:** never add a `cb-train` dependency to `cb-backend` — feature unification breaks the rocm runtime; transcribe CPU references inline.

<details>
<summary>Previous milestone goal — v1.1 GPU Performance (archived at close)</summary>

**Goal:** Full CUDA device-resident training parity — move the entire boosting inner loop onto the GPU (not just derivatives), reaching speed parity with official CatBoost GPU while preserving ≤1e-5 correctness. Full per-phase detail: `.planning/milestones/v1.1-ROADMAP.md`; per-requirement record: `.planning/milestones/v1.1-REQUIREMENTS.md`.

</details>

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
- ✓ **GPU performance parity** — the whole boosting inner loop (device-primitive library, cindex, depth>1 partition-aware histograms, split scoring, Newton + Exact leaves, full grow-policy/loss/sampling/CTR/ranking/multiclass/ordered/Langevin coverage) runs device-resident behind an `Ok(None)`→CPU fallback; **23.9×–42.1× vs the host-light CPU baseline** on P100 (BENCH-03: PASS) — v1.1 (Phases 10–14). _Standing debt: GPUT-14 aggregate sign-off + Phase-10/11 BENCH-02 rows un-run; per-family ≤1e-4 evidence stands (see Current State)._

### Active

<!-- v1.2 Parity Completion & Release Readiness scope. Full REQ-IDs in REQUIREMENTS.md. -->

- [ ] **Debt & hardening** — FEAT-07 online-HNSW KNN parity; GPUT-14 aggregate ε=1e-4 Kaggle CUDA sign-off; Phase-10/11 BENCH-02 speed rows; RV-13-01..04 latent parity hazards.
- [ ] **Model export** — ONNX + CoreML from the trained model.
- [ ] **Extended feature importance** — Interaction, LossFunctionChange, partial-dependence.
- [ ] **Orchestration** — cross-validation, hyperparameter tuning (grid/random), snapshot/resume, standalone calc_metrics / eval_result.
- [ ] **GPU inference evaluator** — device-side predict.
- [ ] **Adoption / DX** — end-to-end benchmark vs official CatBoost, PyPI release readiness, docs + runnable examples, real-dataset validation suite.

### Out of Scope

<!-- Explicit boundaries. Includes reasoning to prevent re-adding. -->

- C API / C FFI layer — PyO3 direct bindings only; no CAPI surface needed
- Mobile / embedded targets — desktop and server workloads only
- Real-time streaming / online training — batch training only
- R and CLI interfaces — Rust and Python only for this milestone
- Distributed / multi-node training (MPI master/worker, multi-GPU) — deferred to a later milestone (v1.2 decision); single-node CPU+GPU only, matching the existing batch-training scope
- PMML and C++/Python source-code model export — deferred (v1.2 decision); ONNX + CoreML cover the interop need for now

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
| `Runtime` grow-tree seam + `Ok(None)`→CPU per-fit fallback (all-or-nothing, D-10-01) | Every device increment stays oracle-safe — any uncovered config transparently falls back to the CPU reference; no mid-model CPU/device tree mixing | ✓ Good (v1.1) |
| Device-resident `GpuTrainSession` (upload once, no per-tree `der1` read-back; only O(1) BestSplit + 2^depth part-stats cross per level, D-05) | Eliminates the host↔device round-trips that made the derivatives-only MVP >20× slow | ✓ Good (v1.1) — 23.9×–42.1× on P100 |
| From-scratch CubeCL device-primitive library (no CUB) + fixed-point u64 deterministic reduction | CubeCL has no CUB; a deterministic reduction is required to hold ε=1e-4 across hundreds of trees despite non-deterministic atomicAdd ordering | ✓ Good (v1.1) |
| Kaggle CUDA as the single authoritative GPU oracle (correctness + speed); ROCm in-env is not a gate | No NVIDIA in-env; CUDA is the real target, so ROCm smoke-testing must not be able to satisfy a requirement alone | ✓ Good (v1.1) |
| Per-phase standing speed check (BENCH-02) rather than one end-of-milestone benchmark | Catches perf regressions as each kernel lands; but two per-phase rows (Phase-10/11) went un-run — see standing debt | ⚠️ Revisit (v1.1) — discipline sound, execution incomplete |
| Close v1.1 with GPUT-14 aggregate + Phase-10/11 BENCH-02 as accepted debt | Per-family ≤1e-4 evidence + committed P100 runs substantiate the claim; the missing rows are confirmatory, not load-bearing | — Pending (v1.1) — formal override in 14-VERIFICATION.md |

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
*Last updated: 2026-07-05 after starting milestone v1.2 Parity Completion & Release Readiness (continues from Phase 14). Scope: close remaining CatBoost core/CUDA-doc surface (ONNX+CoreML export, extended fstr, CV/tuning/snapshot/calc_metrics orchestration, GPU inference evaluator), discharge v1.1 debt (GPUT-14, BENCH-02, RV-13-01..04, FEAT-07 HNSW), and productionize (benchmark vs official CatBoost, PyPI release, docs, real-dataset validation). Deferred: distributed training, PMML/code export.*
