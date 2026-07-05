# Project Research Summary

**Project:** catboost-rs
**Domain:** Gradient-boosting library parity completion — v1.2 "Parity Completion & Release Readiness" (new export/orchestration/GPU-inference/HNSW/DX surfaces atop a mature Rust rewrite of CatBoost)
**Researched:** 2026-07-05
**Confidence:** HIGH

## Executive Summary

v1.2 is not greenfield — it is a subsequent-milestone integration study layering six new surfaces (debt discharge, ONNX/CoreML export, extended fstr, CV/tuning/snapshot orchestration, GPU inference, adoption/DX) onto an already-mature workspace that shipped full CPU parity (v1.0) and full device-resident GPU training (v1.1). The research is unusually well-grounded: every new feature has a documented, source-verified upstream analog (vendored `catboost-master/` C++ + the repo's own `CATBOOST_CORE_DESIGN.md`/`CATBOOST_CUDA_KERNELS_DESIGN.md`), so the correct behavior for each surface is a known target, not a design decision. The recommended approach is **debt-first, then export, then GPU-infer, then orchestration, capped by an adoption/DX phase**: re-establish the Kaggle CUDA oracle and close the one open CPU parity gap (online-HNSW) before building anything new on top of the training engine's credibility, land export next because it is read-only and zero-seam-risk, then GPU-inference (which reuses v1.1's device primitives and must inherit its determinism discipline), then the CV/tuning/snapshot orchestration layer (which needs a new `cb-train` checkpoint surface), and finally benchmark/PyPI/docs as the capstone that proves the whole story to adopters.

The single biggest risk across all four research files is **conflating "parity" with "coverage."** Three of the new features have hard upstream ceilings that must be replicated, not exceeded: ONNX/CoreML export upstream itself refuses categorical/text/embedding models (there is no ONNX/CoreML primitive for a learned CTR), so "ONNX export" must be reframed as "float-only models, typed rejection otherwise" — attempting to support categorical export would silently diverge from the reference. The GPU inference evaluator upstream restricts itself to oblivious trees, one output dimension, and three prediction types; porting more "because our CPU engine already supports it" is itself a parity bug. And online-HNSW must be a bit-for-bit transcription of upstream's incremental graph construction (RNG, insertion order, `TL2SqrDistance`) oracled on the per-object neighbor *set*, not the final prediction — a third-party HNSW crate can never pass this bar no matter how good its approximate search is. A second cross-cutting risk is GPU determinism: the inference evaluator re-imports every v1.1 GPU landmine (raw atomic-add non-determinism, the HIP `-inf`-literal reject, the `|=` A100 codegen bug) and must reuse the v1.1 fixed-point u64 deterministic reduction and `f32::MIN` sentinel rather than re-transcribing upstream's `TAtomicAdd`/`NegativeInfty()` verbatim.

Architecturally, one structural question needs resolution at roadmap/planning time rather than research time: **STACK.md recommends a new `cb-export` crate** (to keep `prost`/protobuf codegen out of `cb-model`'s lean flatbuffers+serde core) as the primary option with a feature-gated `cb-model/export` module as the fallback, while **ARCHITECTURE.md recommends the feature-gated `cb-model/export` submodule as primary** (on the grounds that export is read-only and shaped exactly like the existing `json.rs`/`cbm.rs`). Both are internally consistent and defensible; the roadmapper should pick one explicitly (this summary defaults to ARCHITECTURE's `cb-model/export` feature-gated submodule, since it avoids a new workspace member and matches existing precedent, but flags the disagreement for the phase-1 export plan to settle before implementation starts) — see Gaps to Address.

## Key Findings

### Recommended Stack

The only genuinely new external dependencies are protobuf tooling for export (`prost`/`prost-build`/`protox`, avoiding a system `protoc`) and `criterion` for Rust-side microbenchmarks; everything else — GPU inference, CV/tuning/snapshot orchestration, extended fstr, online-HNSW — is pure Rust reusing existing crates and needs **no new dependency**. There is no mature Rust ONNX/CoreML *writer* library (tract/ort/coreml-rs only read or run models), so both exporters are hand-built protobuf messages over vendored `.proto` schemas, mirroring what upstream CatBoost's C++ does. `rand` must explicitly NOT be pulled in for random-search tuning — it would break the project's existing deterministic RNG stream that underwrites bootstrap/MVS/CTR reproducibility.

**Core technologies:**
- `prost` 0.14.4 + `prost-build` 0.14.4 + `protox` 0.9.1 — hand-build ONNX `ModelProto` / CoreML `Model` protobuf messages from vendored `.proto` schemas; `protox` keeps the build hermetic (no system `protoc`), matching the existing flatc precedent.
- `criterion` 0.8.2 (dev-dependency, `harness = false`) — Rust-internal micro-benchmarks (predict, export serialize, GPU-inference kernel throughput); separate from the existing Python `benchmark*.py` e2e harness which stays the vs-official-CatBoost comparison layer.
- `PyO3/maturin-action` (CI) over `cibuildwheel` — purpose-built for Rust cross-compiled wheels; one maturin leg per backend Cargo feature (`cpu`/`cuda`/`rocm`/`wgpu`), each a separately-named distribution.
- Existing pins unchanged: cubecl 0.10, pyo3 0.29, flatbuffers, arrow, polars, thiserror/anyhow, ndarray 0.17.2 (reused for PDP grids).

### Expected Features

**Must have (table stakes for "parity complete + release ready"):**
- Debt discharge: GPUT-14 aggregate ε=1e-4 Kaggle CUDA sign-off, Phase-10/11 BENCH-02 rows, RV-13-01..04 latent hazards.
- Online-HNSW KNN (FEAT-07) — closes the one open ≤1e-5 CPU parity gap.
- `cv()`, `grid_search()`/`randomized_search()`, snapshot/resume, `eval_metrics()`/`calc_metrics` — the orchestration surface every migrating CatBoost user calls (grid/random search hard-depends on `cv()` landing first).
- Interaction + LossFunctionChange feature importance (LossFunctionChange reuses the already-shipped SHAP machinery; Interaction is dataset-free, tree-structure-only).
- ONNX + CoreML export, with upstream's guards (float-only, identity-scale, oblivious-only) replicated exactly.
- Benchmark vs official CatBoost (accuracy + speed + memory) + PyPI per-backend wheels + docs/examples — without these the parity work is invisible to adopters.

**Should have (differentiators):**
- GPU inference evaluator — device predict completing the device-training story from v1.1; sequence after debt/export since it stands up a new crate + device kernels.
- Partial dependence — completes the explainability trio; lower priority than Interaction/LossFunctionChange.
- Memory-efficiency-forward benchmark (peak-RSS vs official CatBoost) — cheap addition, directly serves the Core Value.

**Defer (post-v1.2, explicitly out of scope):**
- ONNX/CoreML export of categorical/CTR models — impossible upstream, not a bug to fix.
- SAGE / Independent SHAP / Carry-Uplift fstr, PMML + C++/Python source export, distributed/multi-node training.

### Architecture Approach

v1.2 integrates six new capabilities into the existing crate graph (`cb-core → cb-data → cb-compute → cb-backend(CubeCL) → cb-train → cb-model → catboost-rs/-py`) without violating the standing landmine (`cb-backend` must never depend on `cb-train`). GPU inference is confirmed by the repo's own CUDA design doc (line 2859) to be architecturally separate from training upstream — sharing only the low-level primitive layer — so it lands in a **new `cb-infer-gpu` crate** sitting above both `cb-model` and `cb-backend` (device-agnostic kernels stay in `cb-backend/src/kernels/infer/`; model-shaped orchestration lives in the new crate, avoiding the `cb-backend→cb-model` cycle). Orchestration (CV/tuning/snapshot/calc_metrics) becomes a **new `cb-orchestrate` crate** mirroring upstream's `train_lib` driver-layer separation, requiring a new `BoostingCheckpoint` serde surface on `cb-train`. Extended fstr extends the existing `cb-model/fstr` module (adding a new cubecl-free `cb-model→cb-compute` edge for LossFunctionChange's loss derivatives). Online-HNSW is a self-contained ~936-LOC port living in `cb-compute/src/hnsw/`, wired into `cb-train/estimated` for training-time index build and reused at apply-time.

**Major components (new/modified):**
1. `cb-infer-gpu` (NEW crate) — `GpuEvaluator` host orchestrator; resident `GpuModelData`; `Ok(None)`→CPU fallback for unsupported models (non-oblivious, multi-dim, cat/text/embedding).
2. `cb-orchestrate` (NEW crate) — cross-validation, grid/random tuning, snapshot/resume, calc_metrics/eval_result; drives `cb-train` through a new checkpointable boosting API.
3. Export (either NEW `cb-export` crate [STACK] or feature-gated `cb-model/export` submodule [ARCHITECTURE] — **unresolved, flag for phase-1 planning**) — read-only `&Model → Result<Vec<u8>>` exporters for ONNX/CoreML.
4. `cb-model/fstr` (MODIFIED) — new `interaction.rs`, `loss_change.rs`, `partial_dependence.rs` alongside existing SHAP/basic fstr.
5. `cb-compute/hnsw` (NEW) — bit-exact online-HNSW port, cubecl-free (D-03 clean).

### Critical Pitfalls

1. **Scoping ONNX/CoreML export as "export the model" instead of "export float-only models, typed-reject the rest"** — upstream itself refuses categorical/text/embedding models; test fixtures being numeric-only creates false confidence that will break the moment a real (categorical-heavy) CatBoost model is exported. Avoid by enforcing the guard at the export entry point with a typed error, mirroring upstream's `CB_ENSURE` checks exactly.
2. **Holding exported ONNX/CoreML predictions to the same ≤10⁻⁵ double-precision bar as CPU** — ONNX Runtime and CoreML accumulate in float32, so structurally-correct exports will legitimately drift beyond 10⁻⁵ vs the `.cbm` double reference over many trees. Avoid by defining an export-specific tolerance and oracling against **official CatBoost's own ONNX/CoreML export evaluated in the same runtime** (ORT/CoreML), not against the internal double predictor.
3. **Treating online-HNSW's "approximate" nature as "infeasible to match bit-exactly."** HNSW is deterministic given identical seed/insertion-order/RNG/distance — bit-exact parity IS achievable, but only via a hand transcription of upstream's `library/cpp/online_hnsw` (~936 LOC), never an off-the-shelf HNSW crate. Oracle must assert the per-object neighbor *set* index-for-index, not just the final prediction, or the wrong half of the bug gets hidden.
4. **Re-introducing GPU non-determinism / HIP codegen traps in the new inference evaluator.** Naive transcription of upstream's `double4`+`TAtomicAdd` reduction and `NegativeInfty()`/`|=` idioms reintroduces exactly the landmines v1.1 already solved (raw atomic-add jitter breaking ε=1e-4; bare `-inf` literals rejected by the HIP JIT on gfx1100, invisible to cpu/wgpu `cargo check`). Avoid by reusing the v1.1 fixed-point u64 deterministic reduction and `f32::MIN` sentinel verbatim, and running the rocm suite in-env before any GPU-inference sign-off.
5. **CV/tuning fold-assignment divergence and data leakage** — CatBoost's `cv()` has non-obvious semantics (per-loss stratification defaults, group-in-fold, three split `type`s) that a naive sklearn-style K-fold reimplementation will silently violate; and computing CTR/quantization borders on the full pool before splitting leaks target info, which *improves* CV scores and hides the bug. Avoid by oracling fold-assignment against CatBoost per seed and adding a target-permutation leakage canary.

## Implications for Roadmap

Based on research (ARCHITECTURE.md's build-order analysis, cross-checked against FEATURES.md dependency graph and PITFALLS.md risk ordering), suggested phase structure:

### Phase 1: Debt discharge & CUDA oracle re-establishment
**Rationale:** Every later parity/benchmark claim (export tolerance oracles, GPU-inference sign-off, the adoption benchmark) depends on a trusted CUDA oracle and closed parity hazards. Mostly job execution + contained fixes — high de-risking, low code-change risk.
**Delivers:** GPUT-14 aggregate ε=1e-4 Kaggle CUDA correctness sign-off; Phase-10/11 BENCH-02 speed rows executed; RV-13-01..04 latent parity hazards closed.
**Addresses:** Debt & hardening requirement (PROJECT.md Active).
**Avoids:** Pitfall 12 (benchmark baseline confusion) by re-establishing the real oracle before anyone benchmarks against it.

### Phase 2: Online-HNSW KNN parity (FEAT-07)
**Rationale:** Closes the one remaining open ≤10⁻⁵ CPU parity gap; fully self-contained (~936 LOC, `cb-compute` only); can overlap with Phase 1 since it touches different crates. Completing it makes the "verifiable parity" claim for the whole adoption/benchmark story true.
**Delivers:** Bit-for-bit port of `library/cpp/online_hnsw` in `cb-compute/src/hnsw/`, wired into `cb-train/estimated`; per-object neighbor-set oracle passing.
**Addresses:** FEAT-07 (Debt & hardening requirement).
**Avoids:** Pitfall 8 (declaring HNSW parity infeasible because it's "approximate"; using a third-party HNSW crate).

### Phase 3: Model export (ONNX + CoreML)
**Rationale:** Read-only, zero-seam-risk, independent of every other new surface — the earliest safe feature win, and it introduces no device path or new crate-cycle risk, so it should land before GPU-inference.
**Delivers:** `cb-model/export` submodule (or `cb-export` crate — **resolve placement in this phase's plan**, see Gaps) implementing `Model → Result<Vec<u8>>` for ONNX and CoreML, with hard upstream guards (float-only, identity-scale, oblivious-only) enforced via typed errors.
**Uses:** `prost`/`prost-build`/`protox`, vendored `onnx.proto`/CoreML `.proto` schemas from STACK.md.
**Implements:** "Read-only exporter over `TModelTrees`" pattern from ARCHITECTURE.md.
**Avoids:** Pitfalls 1–4 (categorical export, float32-vs-double tolerance confusion, opset/label mismatches, CoreML execution-validation gaps).

### Phase 4: Extended feature importance
**Rationale:** Independent, single-crate modification (`cb-model/fstr`), medium effort; can run in parallel with Phase 3.
**Delivers:** Interaction (tree-structure-only), LossFunctionChange (reuses shipped SHAP machinery + new `cb-model→cb-compute` loss-derivative edge), partial-dependence (staged-apply sweep).
**Addresses:** Extended feature importance requirement.
**Avoids:** Pitfall 13 (implementing textbook interpretation formulas instead of CatBoost's exact accounting) — oracle each type against CatBoost on models with CTR features, not just numeric fixtures.

### Phase 5: GPU inference evaluator
**Rationale:** Deliberately sequenced after Phase 1: the re-signed CUDA oracle and v1.1 primitive library must be trustworthy before a second device path is built on top of them. Requires a new crate to respect the no-cycle rule.
**Delivers:** New `cb-infer-gpu` crate + `cb-backend/src/kernels/infer/` (Binarize/EvalObliviousTrees/ProcessResults kernels); `Ok(None)`→CPU fallback for non-oblivious/multi-dim/cat models; deterministic fixed-point reduction reused from v1.1.
**Addresses:** GPU inference evaluator requirement.
**Avoids:** Pitfalls 5, 6, 7 (non-deterministic reductions, HIP `-inf`/`|=` codegen traps, silently exceeding upstream's supported GPU subset).

### Phase 6: Orchestration (CV, tuning, snapshot/resume, calc_metrics)
**Rationale:** Parallelizable with Phase 5 (disjoint crates); needs a new `cb-train` checkpoint surface, the only modification to the otherwise-frozen training core this milestone.
**Delivers:** New `cb-orchestrate` crate — `cross_validation.rs`, `tuning.rs` (grid/random, depends on `cv()`), `snapshot.rs` (versioned `BoostingCheckpoint` + RNG-continuity resume), `calc_metrics.rs`.
**Uses:** Existing `cb-train` boosting loop, serde/`.cbm`-style versioned serialization, existing deterministic RNG stream.
**Avoids:** Pitfalls 9, 10 (CV fold/leakage divergence, non-reproducible snapshot/resume) — oracle fold-assignment and straight-vs-resume bit-identity explicitly.

### Phase 7: Adoption / DX capstone
**Rationale:** Must exercise export + GPU-infer + orchestration once they exist, and PyPI release is the final gate for the whole milestone — hence last.
**Delivers:** End-to-end benchmark vs official CatBoost (accuracy + speed + memory, matched hardware/version, GPU numbers from Kaggle CUDA only), PyPI per-backend wheels + CI release matrix + versioning, docs + runnable Rust/Python examples, real-dataset validation suite.
**Addresses:** Adoption/DX requirement.
**Avoids:** Pitfalls 11, 12 (wheel/abi3/free-threaded confusion; benchmarking against the wrong baseline — must be vs official CatBoost, not the v1.1 host-light baseline).

### Phase Ordering Rationale

- **Debt-first over export-first** is a de-risking judgment (both orderings are defensible): the benchmark and "verifiable parity" release claim depend on a trusted CUDA oracle and a closed HNSW gap, so discharging those first protects every downstream claim at low relative cost.
- **Export before GPU-infer** is dependency-neutral but risk-ordered: export has zero seam risk and no device/crate-cycle wiring, so it is the safest place to bank an early win, while GPU-infer should only proceed once the re-signed CUDA oracle (Phase 1) exists to validate it against.
- **Orchestration's tuning sub-feature hard-depends on `cv()`** (grid/random search calls `CrossValidate` per candidate) — this ordering is firm within Phase 6, not just a suggestion.
- **GPU-infer and Orchestration (Phases 5–6) are mutually parallelizable** — disjoint new crates (`cb-infer-gpu` vs `cb-orchestrate`), no shared edge.
- Phase 7 must come last because it is the only phase that exercises every other phase's output (export, GPU-infer, orchestration all feed into the benchmark/docs/release story).

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 3 (Model export):** the STACK vs ARCHITECTURE crate-placement disagreement (`cb-export` new crate vs `cb-model/export` feature-gated submodule) must be resolved before coding starts; also needs opset/tolerance decisions pinned in the SPEC (Pitfalls 2–3).
- **Phase 5 (GPU inference evaluator):** needs a `--research-phase` pass on CubeCL kernel transcription specifics (the HIP `-inf`/`|=` landmines are documented but re-verification against the current CubeCL version is prudent before writing kernels).
- **Phase 6 (Orchestration):** CV fold-parity semantics (stratification-by-loss-function defaults, group-in-fold, three split types) are MEDIUM confidence per PITFALLS.md and warrant a focused research pass against upstream `cross_validation.cpp` before implementation.

Phases with standard patterns (skip research-phase):
- **Phase 1 (Debt discharge):** execution of already-designed jobs (Kaggle CUDA runs) plus contained fixes — no new research needed.
- **Phase 2 (Online-HNSW):** root cause and port scope are already definitively documented (instrumented-trainer evidence, ~936 LOC bounded); implementation is transcription, not design.
- **Phase 4 (Extended fstr):** upstream algorithms are precisely documented in `calc_fstr.h`/`loss_change_fstr.h`; SHAP precedent already shipped.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Crate versions verified via `cargo search`/pip 2026-07-05; ONNX/CoreML formats verified against upstream C++ source + onnx.ai/apple docs |
| Features | HIGH | Grounded in vendored upstream source with file/line citations; algorithm-level detail confirmed for every P1 feature |
| Architecture | HIGH | The load-bearing GPU-inference-separation decision is confirmed verbatim by the repo's own design doc (line 2859); crate-cycle analysis is mechanical and verified against real `Cargo.toml` files |
| Pitfalls | HIGH for ONNX/CoreML limits, GPU determinism, HNSW feasibility (upstream source + in-repo instrumented-trainer evidence); MEDIUM for CV fold-parity internals and PyPI free-threaded specifics |

**Overall confidence:** HIGH

### Gaps to Address

- **Export crate placement (STACK vs ARCHITECTURE disagreement):** STACK.md recommends a new `cb-export` crate as primary (keeps protobuf codegen out of `cb-model`'s lean core); ARCHITECTURE.md recommends a feature-gated `cb-model/export` submodule as primary (matches the existing `json.rs`/`cbm.rs` read-only precedent, avoids a new workspace member). Both files list the other as a valid alternative. **Resolve explicitly in the Phase 3 plan** — this summary does not adjudicate it, since both research files present the tradeoff clearly and the choice affects Cargo.toml wiring, not algorithm correctness.
- **CV fold-partition semantics (MEDIUM confidence):** exact stratification-default-by-loss-function rule, group-in-fold behavior, and the three `type`s (Classical/Inverted/TimeSeries) should be re-verified against `catboost-master/catboost/libs/train_lib/cross_validation.cpp` at Phase 6 planning time rather than relying solely on this research pass.
- **PyPI free-threaded validation:** the abi3-py312/`gil_used=false` build's concurrency claim was never validated against a real `python3.13t` interpreter in Phase 8; Phase 7 must either run that validation or explicitly document the claim as "code property, not live-tested" per the established human-gated pattern.
- **CoreML execution environment:** no macOS/Apple runtime is confirmed available in CI; Phase 3 planning should decide upfront whether CoreML parity will be execution-validated (if an Apple runtime becomes available) or structural-only (documented gap, mirroring the Kaggle-CUDA no-in-env-oracle discipline).

## Sources

### Primary (HIGH confidence)
- `docs/CATBOOST_CORE_DESIGN.md` — export formats, training orchestration/driver layer, fstr dispatcher, eval_result/calc_metrics, KNN=HNSW
- `docs/CATBOOST_CUDA_KERNELS_DESIGN.md` §7 (GPU inference evaluator), line 2859 (inference independent of training) — repo-curated design doc
- `catboost-master/catboost/libs/model/model_export/model_exporter.cpp`, `onnx_helpers.cpp`, `coreml_helpers.cpp` — export guards, opset/label semantics, CoreML pipeline details — upstream source, verified 2026-07-05
- `catboost-master/catboost/libs/fstr/calc_fstr.h`, `loss_change_fstr.h`, `partial_dependence.h` — fstr algorithms
- `catboost-master/catboost/libs/train_lib/cross_validation.cpp`, `hyperparameter_tuning.cpp` — CV/tuning semantics
- `library/cpp/online_hnsw` (upstream) + `.planning/notes/knn-estimated-feature-is-online-hnsw.md` — HNSW root cause, bit-exact feasibility proof
- Current workspace `Cargo.toml` files and `crates/*/src/` layout inspection — crate dependency graph
- `.planning/PROJECT.md` — v1.2 scope, standing debt, landmine restatement
- `cargo search` / `pip index versions` 2026-07-05 — crate/tool version verification

### Secondary (MEDIUM confidence)
- CV fold-partition internals (stratification defaults, group semantics) — documented but flagged for re-verification
- PyPI free-threaded / `python3.13t` specifics — project memory (`phase8-python-bindings-outcome.md`), not independently re-verified this pass

---
*Research completed: 2026-07-05*
*Ready for roadmap: yes*
