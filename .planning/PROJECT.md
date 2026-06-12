# catboost-rs

## What This Is

A full Rust rewrite of the CatBoost gradient boosting library, targeting complete feature parity with the original. It exposes first-class APIs in both Rust (using Rust-native patterns) and Python (via PyO3/maturin with a scikit-learn compatible interface). GPU acceleration is provided through CubeCL, with backends switchable via Cargo features.

## Core Value

A memory-efficient, Rust-native CatBoost implementation that Rust developers can embed directly and Python ML practitioners can drop into scikit-learn pipelines — without sacrificing feature parity with the original CatBoost.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Gradient boosting training and prediction (classifier, regressor, ranker)
- [ ] Categorical feature encoding (target encoding, ordered boosting)
- [ ] SHAP value computation
- [ ] Text and embedding feature support
- [ ] Serialization / model save-load
- [ ] Multi-backend GPU execution via CubeCL (cuda, rocm, wgpu) + cpu
- [ ] Python bindings via PyO3/maturin with scikit-learn compatible API (fit/predict/score)
- [ ] NumPy, Pandas, and Arrow/Polars input support in Python
- [ ] Cargo workspace with feature-gated backend crates
- [ ] Oracle test suite: random data, error tolerance ≤ 10⁻⁵, GPU tests on rocm

### Out of Scope

- C API / C FFI layer — PyO3 direct bindings only; no CAPI needed
- Mobile/embedded targets — desktop and server workloads only
- Real-time streaming training — batch training only for v1

## Context

- CatBoost (original) is a C++ library with Python, R, and CLI interfaces. The rewrite targets full algorithmic parity in Rust.
- CubeCL provides a Rust-native GPU compute abstraction; backends are selected at compile time via Cargo features: `cuda` (untestable locally), `rocm`, `wgpu`, `cpu`. The CubeCL runtime is parameterized via generics for flexible backend switching without runtime overhead.
- Python packaging: users install the backend-specific wheel (e.g. `catboost-rs-rocm`). PyO3 + maturin handle the Rust→Python boundary.
- Python version requirement: >= 3.12.
- No `unwrap()` in production code — error propagation uses `thiserror` (library errors) + `anyhow` (application/binding errors).
- Test design: oracle testing with randomly generated inputs; pass threshold is absolute error < 10⁻⁵ vs reference CatBoost output. GPU tests run exclusively on the rocm backend. Source and test code are strictly separated (no `#[cfg(test)]` inline with production logic).

## Constraints

- **Tech stack**: Rust (latest stable), CubeCL for GPU kernels, PyO3 + maturin for Python bindings
- **Python version**: >= 3.12
- **Backend selection**: Cargo features only — `cuda`, `rocm`, `wgpu`, `cpu`; no runtime switching
- **Dependencies**: Always use the latest crate versions
- **Error handling**: `thiserror` for library errors, `anyhow` for application-level; `unwrap()` strictly prohibited in production
- **Memory**: High memory efficiency is a first-class design constraint — minimize allocations, prefer zero-copy where possible
- **Workspace**: Modular Cargo workspace from day one — clear crate separation of responsibilities
- **API style**: Rust side uses Builder pattern; Python side is scikit-learn compatible (fit/predict/score)
- **No C API**: PyO3 bindings only; no C FFI or CAPI layer

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| CubeCL for GPU kernels | Rust-native abstraction supporting cuda/rocm/wgpu in a single codebase via generics | — Pending |
| PyO3 + maturin (no CAPI) | Simplest correct path for Rust→Python; avoids unsafe C ABI layer | — Pending |
| Scikit-learn compatible Python API | Allows drop-in use in existing ML pipelines without user rework | — Pending |
| Feature-gated backends | Users compile and install only what their hardware supports | — Pending |
| Oracle testing vs reference CatBoost | Ensures algorithmic correctness against the original implementation | — Pending |
| thiserror + anyhow error strategy | thiserror for clean library API errors; anyhow for ergonomic error propagation at bindings/app level | — Pending |

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
*Last updated: 2026-06-13 after initialization*
