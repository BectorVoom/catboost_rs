# Requirements: catboost-rs — v1.2 Parity Completion & Release Readiness

**Defined:** 2026-07-05
**Core Value:** A memory-efficient, Rust-native CatBoost implementation with verifiable feature parity (oracle-tested ≤10⁻⁵ CPU / ε=1e-4 GPU), embeddable in Rust and droppable into both scikit-learn and existing CatBoost Python pipelines.

> **Research basis:** `.planning/research/SUMMARY.md` (+ STACK/FEATURES/ARCHITECTURE/PITFALLS). Every requirement has a source-verified upstream analog in `catboost-master/` and `docs/CATBOOST_CORE_DESIGN.md` / `docs/CATBOOST_CUDA_KERNELS_DESIGN.md` — behavior is a known target, not a design choice.
>
> **Cross-cutting reframes (from PITFALLS):** export = *float-only models + typed rejection*, held to an **export-specific float32 tolerance** oracled against CatBoost's own ONNX/CoreML in ORT/CoreML — NOT the ≤1e-5 double bar. FEAT-07 HNSW = *bit-for-bit transcription* oracled on the per-object **neighbor set**. GPU inference must reuse v1.1's fixed-point-u64 deterministic reduction + `f32::MIN` sentinel and sign off on **Kaggle CUDA** (ROCm in-env = non-gating). New backend-bearing crates keep `default-features=false` / no unconditional `cpu` (never add a `cb-train` dep to `cb-backend`).

## v1 Requirements

Requirements for the v1.2 milestone. Each maps to exactly one roadmap phase.

### Debt & Hardening

- [ ] **HARD-01**: Aggregate ε=1e-4 Kaggle CUDA correctness sign-off (GPUT-14) executed across all v1.1 device families as one authoritative row
- [ ] **HARD-02**: Phase-10 (depth-1) and Phase-11 (depth-6) BENCH-02 speed rows executed on Kaggle CUDA; BENCH-03 aggregate completed with real numbers (no stitched gaps)
- [x] **HARD-03**: RV-13-01..04 latent parity hazards resolved (or explicitly retired with evidence)
- [ ] **FEAT-07**: Online-HNSW KNN estimated-feature bit-exact parity — bit-for-bit port of `library/cpp/online_hnsw` (~936 LOC) in `cb-compute`, oracled on the per-object neighbor set (index-for-index), closing the last open ≤10⁻⁵ CPU gap

### Model Export

- [ ] **EXPORT-01**: User can export a trained model to ONNX (float-only, oblivious, identity-scale) via `ai.onnx.ml` TreeEnsembleRegressor/Classifier(+ZipMap); categorical/CTR/text/embedding models are rejected with a typed error mirroring upstream's guard
- [ ] **EXPORT-02**: User can export a trained model to CoreML (float + one-hot categorical pipeline, oblivious ≤16 levels, identity-scale, single-dim bias); unsupported models rejected with a typed error
- [ ] **EXPORT-03**: Export correctness is oracled against official CatBoost's own ONNX/CoreML export evaluated in the same runtime (ONNX Runtime; CoreML execution-validated if an Apple runtime is available, else structural-only with a documented gap), under an export-specific tolerance

### Extended Feature Importance

- [ ] **FSTR-01**: User can compute `Interaction` feature importance (pairwise split-cooccurrence over tree structure; dataset-free)
- [ ] **FSTR-02**: User can compute `LossFunctionChange` feature importance (reuses shipped SHAP `TShapPreparedTrees` + metric re-evaluation; requires a dataset)
- [ ] **FSTR-03**: User can compute partial-dependence for one or two features (staged-apply sweep)

### Orchestration

- [ ] **ORCH-01**: User can run cross-validation (`cv()`) matching CatBoost fold semantics (Classical/Inverted/TimeSeries split types, per-loss stratification defaults, group-in-fold), fold-assignment oracled per seed with a target-permutation leakage canary
- [ ] **ORCH-02**: User can run `grid_search` and `randomized_search` hyperparameter tuning (built on `cv()`; reuses the existing deterministic RNG — no `rand` dependency)
- [ ] **ORCH-03**: User can snapshot and resume training — versioned `BoostingCheckpoint` (serde + format-version guard) with full RNG-state continuity; straight-run vs resumed-run predictions bit-identical
- [ ] **ORCH-04**: User can compute metrics on precomputed predictions standalone (`calc_metrics` / `eval_metrics`, staged), independent of a live training run

### GPU Inference

- [ ] **GINF-01**: User can run device-side model inference — new `cb-infer-gpu` crate over `cb-model` + `cb-backend` (device-agnostic Binarize/EvalObliviousTrees/ProcessResults kernels in `cb-backend/src/kernels/infer/`), restricted to oblivious / single-dim / float-only / RawFormulaVal|Probability|Class with an `Ok(None)`→CPU fallback for everything else; deterministic reduction reused from v1.1; ε=1e-4 signed off on Kaggle CUDA

### Adoption / DX

- [ ] **DX-01**: End-to-end benchmark vs official CatBoost — accuracy, speed, and peak-RSS memory on real datasets, matched hardware/version (GPU numbers from Kaggle CUDA only; CPU baseline = official CatBoost, not the v1.1 host-light baseline)
- [ ] **DX-02**: PyPI release readiness — per-backend wheels (`cpu`/`cuda`/`rocm`/`wgpu` as separately-named distributions) via `maturin-action` CI matrix, abi3-py312, versioning/tagging
- [ ] **DX-03**: Documentation + runnable Rust and Python examples covering training, inference, export, fstr, CV/tuning, and GPU usage
- [ ] **DX-04**: Real-world dataset validation suite exercising the new surfaces (export round-trip, CV, tuning, GPU inference) end-to-end

## Future Requirements

Deferred beyond v1.2. Tracked, not in this roadmap.

### Advanced Explainability

- **FSTR-04**: SAGE / Independent SHAP / Carry-Uplift feature-importance modes

## Out of Scope

Explicitly excluded for v1.2. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| ONNX/CoreML export of categorical / CTR / text / embedding models | Impossible upstream — there is no ONNX/CoreML primitive for a learned CTR; CatBoost itself refuses these formats. Matching = typed rejection, not support. |
| PMML export | Deferred (v1.2 decision); ONNX + CoreML cover the interop need |
| C++ / Python source-code model export | Deferred (v1.2 decision) |
| Distributed / multi-node training (MPI master/worker, multi-GPU) | Deferred to a later milestone; single-node CPU+GPU only, matching the existing batch-training scope |
| GPU inference beyond oblivious / 1-dim / float-only | Replicating upstream's exact supported subset; exceeding it would itself be a parity bug |
| R and CLI interfaces | Rust and Python only |
| Real-time streaming / online training | Batch training only |
| Mobile / embedded targets | Desktop and server workloads only |

## Traceability

Which phases cover which requirements. Populated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| HARD-01 | Phase 15 | Pending |
| HARD-02 | Phase 15 | Pending |
| HARD-03 | Phase 15 | Complete |
| FEAT-07 | Phase 16 | Pending |
| EXPORT-01 | Phase 17 | Pending |
| EXPORT-02 | Phase 17 | Pending |
| EXPORT-03 | Phase 17 | Pending |
| FSTR-01 | Phase 18 | Pending |
| FSTR-02 | Phase 18 | Pending |
| FSTR-03 | Phase 18 | Pending |
| ORCH-01 | Phase 20 | Pending |
| ORCH-02 | Phase 20 | Pending |
| ORCH-03 | Phase 20 | Pending |
| ORCH-04 | Phase 20 | Pending |
| GINF-01 | Phase 19 | Pending |
| DX-01 | Phase 21 | Pending |
| DX-02 | Phase 21 | Pending |
| DX-03 | Phase 21 | Pending |
| DX-04 | Phase 21 | Pending |

**Coverage:**

- v1.2 requirements: 19 total
- Mapped to phases: 19 ✓
- Unmapped: 0

---
*Requirements defined: 2026-07-05*
*Last updated: 2026-07-05 after roadmap creation (v1.2 Phases 15–21, 19/19 mapped)*
