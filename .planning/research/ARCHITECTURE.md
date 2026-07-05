# Architecture Research

**Domain:** catboost-rs milestone v1.2 "Parity Completion & Release Readiness" — integrating new surfaces (ONNX/CoreML export, GPU inference evaluator, extended fstr, CV/tuning/snapshot orchestration, online-HNSW, benchmark/PyPI) into an existing mature Rust workspace
**Researched:** 2026-07-05
**Milestone:** v1.2 Parity Completion & Release Readiness (supersedes the v1.1 ARCHITECTURE research)
**Confidence:** HIGH (grounded in the repo's own design docs + current crate graph; the load-bearing GPU-inference decision is confirmed verbatim by `CATBOOST_CUDA_KERNELS_DESIGN.md` §7 + line 2859)

> This is a SUBSEQUENT-milestone integration study, not a greenfield design. Every recommendation below integrates **with** the existing workspace and respects the standing landmine: **never add a `cb-train` dependency to `cb-backend`** (feature unification breaks the rocm runtime).

## Standard Architecture

### Existing crate graph (what v1.2 integrates into)

```
┌──────────────────────────────────────────────────────────────────────┐
│  API / bindings                                                        │
│   catboost-rs (Builder facade)     catboost-rs-py (PyO3 + maturin)     │
└───────┬───────────────────────────────────┬──────────────────────────┘
        │                                    │
┌───────▼────────────┐   ┌──────────────┐   ┌▼───────────────┐
│ cb-train           │   │ cb-model     │   │ cb-oracle      │
│ boosting/tree/ctr/ │   │ model/cbm/   │   │ (parity oracle)│
│ metrics/estimated  │   │ json/apply/  │   └────────────────┘
│                    │   │ predict/shap/│
│                    │   │ fstr         │
└──┬──────────┬──────┘   └───┬──────────┘
   │          │              │  (cb-model → cb-train passthrough only,
   │          │              │   for backend-feature forwarding)
   │      ┌───▼──────────────▼───┐
   │      │ cb-compute           │  ← PURE GENERIC, cubecl-FREE (D-03)
   │      │ loss/score/leaf/hist/│
   │      │ pairwise/ranking_der/│
   │      │ text+embedding calcer│
   │      └───┬──────────────────┘
   │          │
┌──▼──────────▼───┐        ┌──────────────┐
│ cb-backend      │───────▶│ cb-compute   │  (implements its Runtime trait)
│ CubeCL kernels +│        └──────────────┘
│ gpu_runtime +   │   ⛔ LANDMINE: cb-backend MUST NOT depend on cb-train
│ GpuTrainSession │
└───────┬─────────┘
        │
   ┌────▼─────┐   ┌──────────┐
   │ cb-core  │   │ cb-data  │  (rng/reduction/error ; pool/quantize/borders/ctr/text)
   └──────────┘   └──────────┘
```

Observed dependency edges (from `Cargo.toml` inspection): `cb-data→cb-core`; `cb-compute→cb-core,cb-data`; `cb-backend→cubecl,cb-compute`; `cb-train→cb-core,cb-data,cb-compute,cb-backend`; `cb-model→cb-core,cb-data,cb-train(passthrough),flatbuffers`; `catboost-rs→cb-core,cb-data,cb-compute,cb-backend,cb-train,cb-model`; `catboost-rs-py→catboost-rs,cb-data,arrow`.

### The decisive precedent for GPU inference

`CATBOOST_CUDA_KERNELS_DESIGN.md` line 2859–2861 states, of the upstream engine:

> **"The GPU inference evaluator (§7.1) is a separate unit (`catboost/libs/model/cuda`) that does *not* depend on any of `catboost/cuda/` [training] — it shares only `library/cpp/cuda/wrappers`. Training and inference on GPU are independent code paths."**

This is the **exact analog** of our workspace landmine. Upstream already keeps device-inference physically separate from device-training, sharing only the low-level CUDA primitive layer. Our mapping: the "shared primitive layer" = the v1.1 CubeCL primitive library that lives in `cb-backend/src/kernels/`; the "separate inference unit" = a new crate that consumes the model + those primitives but never touches training.

### Component Responsibilities (new + modified)

| Component | Responsibility | New / Modified | Home crate |
|-----------|----------------|----------------|------------|
| Device eval kernels (`Binarize`, `EvalObliviousTrees`, `ProcessResults`) | Model-agnostic `#[cube]` kernels over flat arrays (repacked splits, borders, leaf values, cursor) | **NEW** | `cb-backend/src/kernels/infer/` |
| `GpuEvaluator` host orchestrator | Build resident `GpuModelData` once from `TModelTrees`; per-batch quantize→eval→postprocess; `Ok(None)`→CPU fallback | **NEW** | **NEW crate `cb-infer-gpu`** |
| ONNX / CoreML exporters | Read `TModelTrees`/leaf values/borders/scale-bias → external byte streams | **NEW** | `cb-model/src/export/` |
| Interaction / LossFunctionChange / PartialDependence fstr | Extend importance surface beyond shipped SHAP + basic fstr | **MODIFIED** | `cb-model/src/fstr/` |
| Cross-validation, grid/random tuning, snapshot/resume, calc_metrics/eval_result | Orchestrate repeated training + checkpointing | **NEW** | **NEW crate `cb-orchestrate`** |
| Resumable boosting checkpoint API | Expose serializable boosting-loop state for snapshot/resume | **MODIFIED** | `cb-train` (surface change) |
| Online-HNSW index | Approximate KNN estimated-feature parity (replace brute-force-exact) | **NEW** | `cb-compute/src/hnsw/` (+ wire in `cb-train/estimated`) |
| Benchmark harness | End-to-end accuracy+speed vs official CatBoost | **NEW** | `benchmarks/` (non-published) |
| PyPI release config | Per-backend wheels, CI matrix, versioning | **MODIFIED** | `catboost-rs-py` + CI |

## Recommended Project Structure (deltas only)

```
crates/
├── cb-backend/src/kernels/
│   └── infer/                     # NEW — device inference kernels (model-agnostic)
│       ├── binarize.rs            #   quantize raw floats → warp-interleaved bins
│       ├── eval_oblivious.rs      #   per-doc leaf index + Σ leaf values over trees
│       └── process_results.rs     #   scale/bias + activation (Raw/Prob/Class)
│
├── cb-infer-gpu/                  # NEW CRATE — the "separate unit" (analog of libs/model/cuda)
│   ├── Cargo.toml                 #   deps: cb-model, cb-backend(default-features=false), cb-core
│   └── src/
│       ├── model_data.rs          #   GpuModelData: resident device arrays (splits/borders/leaves)
│       ├── evaluator.rs           #   GpuEvaluator: EvalData→QuantizeData→EvalQuantizedData
│       └── fallback.rs            #   Ok(None)→CPU apply for unsupported models
│
├── cb-model/src/
│   ├── export/                    # NEW submodule tree (feature = "export")
│   │   ├── mod.rs                 #   dispatcher by target format + guards
│   │   ├── onnx.rs                #   TModelTrees → ONNX TreeEnsemble proto
│   │   └── coreml.rs              #   TModelTrees → CoreML TreeEnsemble spec
│   └── fstr/                      # MODIFIED — split fstr.rs into a module
│       ├── mod.rs                 #   dispatcher (existing PredictionValuesChange + SHAP)
│       ├── interaction.rs         #   NEW — co-occurring-split interaction counts
│       ├── loss_change.rs         #   NEW — LossFunctionChange (needs dataset + loss der)
│       └── partial_dependence.rs  #   NEW — feature-sweep prediction surface
│
├── cb-compute/src/
│   └── hnsw/                      # NEW — online-HNSW index (cubecl-free, D-03 clean)
│       ├── mod.rs
│       └── online_hnsw.rs         #   port of library/cpp/online_hnsw (~936 LOC)
│
├── cb-orchestrate/               # NEW CRATE — top driver layer (analog of train_lib)
│   ├── Cargo.toml                #   deps: cb-train, cb-model, cb-data, cb-compute
│   └── src/
│       ├── cross_validation.rs   #   fold split + per-fold train + averaged curves
│       ├── tuning.rs             #   grid_search / randomized_search
│       ├── snapshot.rs           #   serde checkpoint of boosting state + resume
│       └── calc_metrics.rs       #   eval_result / calc_metrics on predictions
│
benchmarks/                        # NEW (non-published workspace member or scripts)
│   ├── rust/                      #   criterion speed harness
│   └── driver.py                  #   accuracy+speed vs official catboost oracle
```

### Structure Rationale

- **`cb-infer-gpu` as a NEW crate, not a `cb-model` module.** `cb-backend` cannot depend on `cb-model` (that would be a cycle: `cb-model→cb-train→cb-backend`), so the model-shaped host evaluator cannot live in `cb-backend`. Putting it inside `cb-model` would force `cb-model` to *directly use* cubecl launch APIs and pull cubecl compilation into a crate that every CPU-only consumer depends on. A separate crate above both `cb-model` and `cb-backend` is the only placement that (a) respects the no-cycle rule, (b) keeps GPU-infer opt-in, and (c) mirrors upstream's deliberate `libs/model/cuda`-separate-from-`libs/model` split. **The eval *kernels* still live in `cb-backend`** (the single cubecl-owning crate, D-02/D-03) because they are model-agnostic array operations; only the model-shaped orchestration lives in `cb-infer-gpu`.
- **Export inside `cb-model`, not a new crate.** ONNX/CoreML are pure read-only serializations over `TModelTrees`/leaf values/borders — the same shape as the already-present `json.rs`/`cbm.rs`. No compute, no new subsystem boundary. Gate behind a `cb-model` `export` cargo feature so protobuf deps stay optional.
- **Extended fstr inside `cb-model/fstr`.** SHAP + basic fstr already live there and already own `TShapPreparedTrees`. Interaction (model-only) and PartialDependence (model+apply) need nothing new. LossFunctionChange needs loss derivatives → add a direct `cb-model→cb-compute` edge (cubecl-free, cycle-free) rather than reaching through `cb-train`.
- **Orchestration as a NEW crate `cb-orchestrate`.** Upstream keeps `cross_validation.cpp` / `hyperparameter_tuning.cpp` in `train_lib` as a distinct driver layer *above* the core algo. Snapshot/resume must serialize boosting-loop state (folds, approxes, tree list, RNG, iteration) that lives in `cb-train`. Housing this in the `catboost-rs` Builder facade would bloat the thin API surface; a dedicated crate keeps the facade thin and gives Python a single bind target.
- **Online-HNSW inside `cb-compute`.** The KNN vote is an embedding calcer, and the embedding calcers already live in `cb-compute/embedding_calcers.rs`. HNSW is a pure algorithmic index (cubecl-free → D-03 clean). Train-time index build wires in from `cb-train/estimated`; apply-time uses the same module.

## Architectural Patterns

### Pattern 1: Separate GPU-inference unit sharing only the primitive library

**What:** Device predict is a distinct crate (`cb-infer-gpu`) that reuses the v1.1 CubeCL primitive library (`reduce`/fixed-point deterministic sum, `cindex`/`compression`, warp-interleaved buffers, atomic-add reduction) via `cb-backend`, and reads the model via `cb-model` — but never links `cb-train`.
**When to use:** Whenever a device path must consume the trained model but not the trainer.
**Trade-offs:** (+) respects the landmine automatically; (+) keeps `cb-model` cubecl-free; (+) mirrors upstream exactly; (−) `cb-infer-gpu` transitively pulls `cb-train` through `cb-model`'s passthrough edge — acceptable (the landmine forbids only `cb-backend→cb-train`, not this crate), and can be tightened later by factoring pure model-repr types if compile cost bites.

**Reuse map (v1.1 primitives → inference kernels):**
```
v1.1 kernels/reduce.rs (fixed-point u64 det. sum) → EvalObliviousTrees leaf-value accumulation
v1.1 kernels/cindex.rs + compression.rs           → per-doc leaf-index traversal
v1.1 warp-interleaved buffer layout               → Binarize writes bucket·WarpSize+lane
v1.1 gpu_runtime session residency pattern        → GpuModelData resident across predict batches
```
Upstream constraint carried verbatim (§7.1): oblivious trees only, **1 output dim only**, no cat/text/embedding → everything else takes the `Ok(None)`→CPU `apply.rs` fallback (the same all-or-nothing seam as v1.1 training, D-10-01).

### Pattern 2: Read-only exporter over `TModelTrees`

**What:** Each exporter is a pure function `&Model → Result<Vec<u8>>` reading tree structure, leaf values, borders, scale/bias — no mutation, no training, no device.
**When to use:** ONNX, CoreML (and future PMML if un-deferred).
**Trade-offs:** (+) zero seam risk, trivially parallelizable phase; (−) format guards must reject unsupported models (ONNX/CoreML: identity scale required, no cat/text/embedding; non-symmetric trees → cbm/json only) — enforce at the dispatcher exactly as upstream `ExportModel` does.

**Example:**
```rust
// cb-model/src/export/mod.rs
pub fn export(model: &Model, fmt: ExportFormat) -> Result<Vec<u8>, ExportError> {
    match fmt {
        ExportFormat::Onnx   => { require_identity_scale(model)?; require_float_only(model)?; onnx::to_onnx(model) }
        ExportFormat::CoreML => { require_identity_scale(model)?; coreml::to_coreml(model) }
    }
}
```

### Pattern 3: Orchestration drives `cb-train` through a checkpointable boosting API

**What:** CV/tuning call the existing `cb-train` boosting loop repeatedly; snapshot/resume requires `cb-train` to expose a serde-serializable checkpoint of its boosting state and to accept one as an initial state.
**When to use:** cross-validation, grid/random search, snapshot/resume.
**Trade-offs:** (+) reuses the proven boosting loop unchanged in substance; (−) requires a *surface* change to `cb-train` (a `BoostingCheckpoint` struct + a "resume from" entry point) — the only modification to an otherwise-frozen training core. Pin RNG-seed continuity across resume (upstream "snapshot-random-seed continuity").

**Snapshot format:** a versioned `serde` struct (recommend `bincode` for compactness + a leading `format_version: u32` guard mirroring `.cbm`'s `CURRENT_CORE_FORMAT_STRING` check) capturing: iteration index, per-fold approxes, accumulated tree structures + leaf values, RNG state, and the resolved options hash. Resume = deserialize → feed as `initLearnProgress`-analog into the boosting driver.

## Data Flow

### New: GPU inference path

```
predict(batch)
   │
   ├─ GpuEvaluator supported?  ── no ──▶ Ok(None) ─▶ cb-model/apply.rs (CPU)
   │        (oblivious, 1-dim, float-only)
   yes
   ▼
GpuModelData (built once, resident: TreeSplits/borders/leaf offsets/scale/bias)
   ▼
Binarize (cb-backend)  →  EvalObliviousTrees (cb-backend)  →  ProcessResults (cb-backend)
   ▼
device→host copy → Vec<f64>
```

### New: export path

```
Model (cb-model) → export dispatcher → {onnx.rs | coreml.rs} → Vec<u8> → file
```

### New: orchestration path

```
cv(params, pool)
   → cb-data split folds → for each fold: cb-train boosting loop (CalcMetricsOnly)
   → cb-train/metrics per iter → average across folds → CVResult curves
grid_search(grid, pool)
   → quantize once (cb-data) → for each candidate: cv or single split → keep best
snapshot: every N iters the boosting loop emits BoostingCheckpoint → serde → disk
resume:   disk → BoostingCheckpoint → cb-train resumes at saved iteration
```

### Modified: extended fstr

```
Model (+ dataset for loss-based)  → cb-model/fstr dispatcher
   Interaction        → model structure only         → pair-impact table
   LossFunctionChange → SHAP leaf stats + cb-compute loss der + dataset → per-feature loss delta
   PartialDependence  → apply.rs sweep over feature grid → dependence surface
```

### Modified: online-HNSW estimated feature

```
train:  cb-train/estimated → build online-HNSW index over training embeddings (cb-compute/hnsw)
apply:  embedding_calcers.rs → approximate KNN vote via same index  → matches upstream bit-exact
```
Closes the definitive FEAT-07 root cause (memory note: upstream KNN calcer = online HNSW *approximate*, current Rust = brute-force-*exact* → per-stage XOR residual). The port is self-contained (~936 LOC) and lives entirely in `cb-compute` + one wiring change in `cb-train/estimated`.

## Suggested Build Order (dependency- and risk-respecting)

**Verdict: debt-first, then export before GPU-infer.** Rationale below.

| # | Phase | Crates touched | Why here |
|---|-------|----------------|----------|
| 1 | **Debt: GPUT-14 aggregate + Phase-10/11 BENCH-02 + RV-13-01..04** | (run existing kernels on Kaggle CUDA; small fixes in `cb-backend`/`cb-train`) | Re-establishes a **trusted CUDA oracle** and closes latent parity hazards. Mostly job execution + contained fixes; high de-risking, low code risk. Every later parity/benchmark claim rests on this. |
| 2 | **FEAT-07 online-HNSW** | `cb-compute` (+`cb-train/estimated`) | Closes the last known CPU parity gap; fully self-contained; unblocks the "verifiable parity" claim the benchmark and release lean on. Overlaps with (1). |
| 3 | **ONNX / CoreML export** | `cb-model` | Read-only, zero-seam-risk, independent of everything. Earliest safe feature win; parallel with (1)/(2). Goes **before** GPU-infer precisely because it introduces no device path and no new crate wiring. |
| 4 | **Extended fstr** | `cb-model` (+ new `cb-model→cb-compute` edge) | Independent, modifies one crate; medium effort. |
| 5 | **GPU inference evaluator** | **NEW `cb-infer-gpu`** + `cb-backend/kernels/infer` | Deliberately after (1): the v1.1 primitive library + Kaggle CUDA oracle must be *signed off* before adding a second device path on top of them. |
| 6 | **Orchestration** | **NEW `cb-orchestrate`** + `cb-train` checkpoint surface | Needs the `cb-train` checkpoint API; parallelizable with (5) (disjoint crates). |
| 7 | **Adoption/DX**: benchmark vs official, PyPI wheels/CI, docs, real-dataset validation | `benchmarks/`, `catboost-rs-py`, CI | Capstone — the benchmark and real-dataset suite must exercise export + GPU-infer + orchestration, and PyPI release is the final gate. |

**Why debt-first over export-first:** both the benchmark and the release-grade "verifiable parity" claim depend on a trusted CUDA oracle and closed parity gaps. Discharging the pending Kaggle sign-off + HNSW first de-risks every downstream claim at low cost. Export is genuinely independent and slots in *parallel* immediately after — but it is not a prerequisite for anything, so it does not need to precede debt.

**Why export before GPU-infer:** export is read-only with zero seam risk and no new crate wiring; GPU-infer stands up a new crate + new device kernels and should follow the re-signed CUDA oracle from phase 1.

## Anti-Patterns

### Anti-Pattern 1: Putting the GPU evaluator's host orchestration in `cb-backend`

**What people do:** add model-shaped predict orchestration next to the kernels in `cb-backend`.
**Why it's wrong:** it forces `cb-backend→cb-model`, which is a dependency cycle (`cb-model→cb-train→cb-backend`), and it drags model types into the pure-runtime crate.
**Do this instead:** kernels (array-only) in `cb-backend`; model-shaped orchestration in the new `cb-infer-gpu` crate above both.

### Anti-Pattern 2: Reaching for training kernels to do inference

**What people do:** reuse `GpuTrainSession` / grow-loop kernels to evaluate a finished model.
**Why it's wrong:** couples inference to training (violating upstream's explicit independence, line 2859) and risks smuggling a `cb-train` edge toward `cb-backend`.
**Do this instead:** inference reuses only the *primitive* library (reduce/cindex/compression/buffers); it needs its own thin `Binarize`/`EvalObliviousTrees`/`ProcessResults` kernels.

### Anti-Pattern 3: Snapshotting the finished model instead of boosting state

**What people do:** serialize the `.cbm` model as a "checkpoint."
**Why it's wrong:** resume needs folds, per-fold approxes, RNG state, and iteration index — not just the tree ensemble. A model snapshot cannot resume mid-training deterministically.
**Do this instead:** a versioned `BoostingCheckpoint` serde struct exposed by `cb-train`, with RNG-seed continuity.

### Anti-Pattern 4: Bloating the `catboost-rs` facade with CV/tuning loops

**What people do:** implement cross-validation and grid search inside the Builder facade.
**Why it's wrong:** the facade is meant to be a thin Builder-pattern API; orchestration logic belongs in a driver layer and needs its own Python bind target.
**Do this instead:** `cb-orchestrate` owns the loops; `catboost-rs` and `catboost-rs-py` re-export thin entry points.

## Integration Points

### Internal Boundaries (new/changed edges)

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `cb-infer-gpu → cb-model` | direct dep (read model repr) | pulls `cb-train` transitively via `cb-model` passthrough — allowed; landmine forbids only `cb-backend→cb-train` |
| `cb-infer-gpu → cb-backend` | direct dep (`default-features=false`, backend passthrough) | launches new `infer/` kernels over the `Runtime` seam; reuses v1.1 primitives |
| `cb-backend/kernels/infer` | NEW `#[cube]` kernels | model-agnostic; use `generics-float` (AGENTS.md); NO `cb-model`/`cb-train` types |
| `cb-model → cb-compute` | NEW direct edge | loss derivatives for LossFunctionChange; cubecl-free so no landmine risk |
| `cb-orchestrate → cb-train` | direct dep + NEW checkpoint surface | requires `cb-train` to expose `BoostingCheckpoint` + resume entry |
| `cb-orchestrate → cb-model/cb-data/cb-compute` | direct deps | build/save model, split folds, compute metrics |
| `cb-compute/hnsw ← cb-train/estimated` | intra-graph wiring | train builds index; apply reuses |
| `catboost-rs → cb-infer-gpu, cb-orchestrate` | NEW facade edges | wire predict-on-device + cv/tuning/snapshot under existing backend feature passthrough |
| `catboost-rs-py → …` | via `catboost-rs` | expose `task_type='GPU'` predict, `cv()`, `grid_search()`, `save_model(format='onnx'/'coreml')` |

### Feature-flag discipline (carried from v1.1)

Every new backend-bearing crate (`cb-infer-gpu`) MUST pull `cb-backend`/`cb-model` with `default-features = false` and forward `cpu`/`cuda`/`rocm`/`wgpu` through its own `[features]` block — never pin `cpu` unconditionally — so `--no-default-features --features rocm` stays cpu-free (the feature-unification landmine documented in `cb-backend/Cargo.toml`). `cb-orchestrate` follows the same passthrough pattern since it transitively bears `cb-backend` through `cb-train`.

### External integration surfaces

| Surface | Integration pattern | Notes / gotchas |
|---------|---------------------|-----------------|
| ONNX | protobuf `TreeEnsemble` op via a proto builder (`prost` + onnx schema, latest crate) behind `export` feature | identity-scale + float-only guard; verify op-set version against onnxruntime |
| CoreML | CoreML `TreeEnsembleRegressor` protobuf spec | identity scale required; optional categorical pipeline (defer cat if parity risk) |
| Kaggle CUDA | existing per-phase oracle (P100), non-gating ROCm smoke in-env | GPU-infer correctness + BENCH sign-off run here, same harness as v1.1 |
| PyPI / maturin | per-backend abi3 wheels (cpu/cuda/rocm), CI release matrix | Phase-8 already emits abi3 wheels; v1.2 adds versioning + release job + wheel naming per backend |

## Confidence Assessment

| Decision | Confidence | Basis |
|----------|-----------|-------|
| GPU-infer = separate `cb-infer-gpu` crate; kernels in `cb-backend` | HIGH | Design doc line 2859 states inference is a separate unit independent of training; cycle analysis of the real crate graph confirms it cannot live in `cb-backend` or cleanly in `cb-model` |
| Export = `cb-model` submodules (feature-gated) | HIGH | Same read-only shape as existing `json.rs`/`cbm.rs`; upstream `model_export` reads `TFullModel` only |
| Extended fstr = extend `cb-model/fstr` + new `cb-model→cb-compute` edge | HIGH | SHAP + basic fstr already there; only LossFunctionChange needs the loss-der edge |
| Orchestration = new `cb-orchestrate` crate + `cb-train` checkpoint surface | MEDIUM-HIGH | Mirrors upstream `train_lib` separation; the exact split of calc_metrics (orchestrate vs cb-train/metrics) is a minor judgment call |
| Online-HNSW = `cb-compute/hnsw` | HIGH | KNN calcer already in `cb-compute/embedding_calcers.rs`; root cause is documented and localized |
| Build order (debt→export→…→GPU-infer→orchestration→DX) | MEDIUM-HIGH | Dependency-forced edges are firm; the debt-first vs export-first ordering is a de-risking judgment (both defensible; debt-first maximizes trust for later claims) |

## Sources

- `docs/CATBOOST_CUDA_KERNELS_DESIGN.md` §6.6 (`models/kernel/add_model_value`), §7.1 (GPU inference evaluator — `libs/model/cuda/evaluator`), line 2859 (inference is a separate unit independent of training) — HIGH (repo-curated design doc)
- `docs/CATBOOST_CORE_DESIGN.md` §"Trained Model … Export Formats" (ONNX/CoreML guards), §"Training Orchestration & Driver Layer" (CV/tuning/snapshot/TLearnProgress), §"Inference API, Feature Importance (fstr)" (Interaction/LossFunctionChange/PartialDependence), §"eval_result/calc_metrics" — HIGH
- Current workspace `Cargo.toml` files (crate dependency + feature graph) and `crates/*/src/` layout inspection — HIGH
- `.planning/PROJECT.md` (v1.2 scope, standing debt, landmine restatement) and MEMORY notes (FEAT-07 HNSW root cause, cb-backend/cb-train landmine) — HIGH

---
*Architecture research for: catboost-rs v1.2 feature integration into the existing crate workspace*
*Researched: 2026-07-05*
