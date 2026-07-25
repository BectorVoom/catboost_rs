---
title: "GINF-01 — GPU inference evaluator (float-only oblivious, first slice)"
status: draft
format: markdown
spec_version: 1
updated_at: 2026-07-19T00:00:00Z
phase: 19-gpu-inference
slice: gpu-inference-evaluator
source_requirements:
  - "User: produce a draft SPEC + TDD PLAN for GINF-01 — a GPU inference/apply evaluator (distinct from GPU training)."
  - "Research: .planning/plans/next-feature-research/ (GINF-01 gap-analysis; the inline research pass in this session)."
  - "Constraint: GPU work uses CubeCL exclusively; backend chosen at compile time via Cargo features cuda/rocm/wgpu/cpu (CLAUDE.md)."
treefinder_pending:
  reason: "TreeFinder / PageIndex MCP not available in-session; this local SPEC is the authoritative draft spec store, matching the calc-metrics / coreml-export sibling convention."
  intended_identifier: "catboost-rs / .planning/plans/gpu-inference-evaluator/SPEC.md"
---

# GINF-01 — GPU inference evaluator

> Draft specification. NOT approved, accepted, final, or implemented.
> Evidence tags: `[VERIFIED: CODEGRAPH …]`, `[VERIFIED: LOCAL <path>]`,
> `[INFERRED: …]`, `[UNVERIFIED: …]`.
>
> Kernel-design content in §4/§5 (GINF-01-S3) was written AFTER personally
> reading the CubeCL manual INDEX, the Conditionals chapter, and the
> Plane-Alignment (branch-divergence) chapter this session
> `[VERIFIED: LOCAL /home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md;
> Cubecl_conditionals.md; plane_alignment.md]`. The implementer MUST re-read the
> cited chapters before writing the kernel (see PLAN cross-cutting guardrails).

---

## 1. Context

`catboost_rs` already ships GPU **training** (Phases 10–14: device histogram,
scoring, leaf-delta apply, packed cindex) but model **inference/apply** is
CPU-only. `cb_model::predict` / `predict_raw` run entirely in
`crates/cb-model/src/apply.rs`, whose module docstring is explicit: *"this file
imports nothing from the backend / GPU compute crate (MODEL-02 — the apply path
runs on a machine with no GPU dependency present)"*
`[VERIFIED: CODEGRAPH crates/cb-model/src/apply.rs:1-6]`.

The scalar oblivious apply is a three-step, branch-free, fixed-shape evaluation
that is exactly the shape SIMT hardware wants:

- **Step B — leaf index** (`leaf_index_for`): forward bit order over the tree's
  ordered splits; split `i` contributes bit `i`. A float split passes iff
  `f64::from(value) > border`
  `[VERIFIED: CODEGRAPH apply.rs:136-140 passes_float_split, :208-215 leaf_index_for;
  cb-train tree.rs:293-301 leaf_index (idx |= 1<<i)]`.
- **Step C — accumulate + bias** (`predict_raw_one`): `raw = bias +
  Σ_trees leaf_values[leaf]`, the per-tree sum routed through the order-locked
  `cb_core::sum_f64` (D-08), bias added exactly once
  `[VERIFIED: CODEGRAPH apply.rs:318-355]`.

This slice adds a GPU evaluator for that path. It targets **float-only, scalar
(`approx_dimension == 1`), oblivious** models only — the single model variant
whose evaluation is fixed-shape and branch-free, mirroring the float-only-first
slicing already used by the ONNX and CoreML exporters
`[VERIFIED: CODEGRAPH crates/cb-model/src/export/onnx.rs:99 is_onnx_exportable;
.planning/plans/coreml-export/SPEC.md §2]`.

**Why not reuse the training packed-cindex / `read_bin` path?** The research pass
suggested encoding per-split tests as quantized-bin lookups via
`PackedCindex`/`read_bin`. That machinery packs QUANTIZED bins for the histogram
kernels `[VERIFIED: CODEGRAPH crates/cb-backend/src/gpu_runtime/cindex.rs:80-110;
kernels.rs:2690 read_bin]`. Inference does NOT need quantization: the CPU apply
compares the RAW `f32` feature value against the model's own `f64` border
(`value > border`), never a bin index. Introducing device-side quantization would
add a bin-vs-border rounding step that can flip a near-border split and diverge
from the CPU oracle. **This slice therefore compares raw feature values against
borders directly on device (byte-mirroring `passes_float_split`)**; the
bin-packed encoding is an explicit v2 memory optimization, not part of the
correctness-first first slice. `[INFERRED: from apply.rs:136-140 semantics +
cindex.rs role — correcting the research's read_bin suggestion.]`

---

## 2. Scope and non-goals

### In scope

- A pure, backend-free **exportability guard** in `cb-model` that admits only
  float-only / oblivious / scalar models and rejects everything else with a typed
  error (mirrors `is_onnx_exportable`).
- A pure, backend-free **flattener** in `cb-model` that lowers such a `Model`
  into device-ready flat arrays (per-split `(feature, border)`, per-tree split
  offsets, leaf values + per-tree leaf offsets, bias). Reusable and CPU-testable.
- A generic-float **`#[cube]` apply kernel** in `cb-backend` computing one raw
  prediction per object (forward-bit-order leaf + per-tree accumulate + bias),
  using branchless predication for the per-split bit (no per-split `if`
  expression).
- A **Model-agnostic launch host helper** in `cb-backend` over
  `SelectedRuntime` (upload-once invariant model arrays, object-parallel
  dispatch, typed read-back), mirroring `launch_block_reduce_f64`.
- A **facade** `predict_raw_on_device` on `catboost-rs` that marshals its inner
  `Model` (guard → flatten → launch) and returns `Vec<f64>`.
- A **device↔CPU parity oracle** asserting device predictions match the shipped
  CPU `predict_raw` within the project's report-bound convention (`SCORE_BOUND`:
  ≤1e-9 for f64 backends, ≤1e-3 for wgpu-f32), runnable under the `cpu` backend
  (CubeCL `CpuRuntime`, f64) in ordinary CI and confirmed on the local ROCm rig.
- An **`#[ignore]` throughput benchmark** (device vs CPU stage timing), non-gating.

### Non-goals (explicit — documented, not silently dropped)

- **CTR / categorical / one-hot / text models** — a `ModelSplit::Ctr` needs a
  string-hash table lookup into `ctr_data` (variable-size, not SIMT-friendly)
  `[VERIFIED: CODEGRAPH apply.rs:157-190 passes_ctr_split]`. Rejected by the guard.
- **Non-symmetric (Lossguide/Depthwise) and Region trees** — separate pointer-walk
  / walk-until-diverge code paths `[VERIFIED: CODEGRAPH apply.rs:234-317]`. Rejected.
- **Multi-dimensional / multiclass (`approx_dimension > 1`)** — dimension-major
  leaf gather `[VERIFIED: CODEGRAPH apply.rs:394-396,719-743]`. Rejected (v2).
- **Staged / prefix (`ntree_start..ntree_end`) device apply** — the first slice
  evaluates the whole ensemble. Prefix apply is a follow-on.
- **Bit-for-bit parity across backends/compilers.** The kernel's accumulation
  ASSOCIATION is locked to match the CPU oracle exactly (ascending per-tree
  order seeded at `0.0`, `bias` added once at the end — mirroring `sum_f64` +
  `predict_raw_one`'s `bias + sum_f64(&oblivious)`, §9 R1, corrected after
  PLAN-CHECK pass 1). What is NOT claimed exact is bit-identical output across
  DIFFERENT backends/hardware/compilers (rounding-mode, FMA-fusion, and
  codegen differences between `cpu`/wgpu/cuda/rocm); parity there is to the
  `SCORE_BOUND` report convention (§9 R1/R2), not exact.
- **Numeric sign-off on wgpu-f32.** wgpu has no f64; borders lose precision under
  f32 compare, so wgpu is compile/smoke-verified only, not a numeric gate (§9 R2).
- **A new external crate.** Only the already-pinned `cubecl = 0.10.0` /
  `bytemuck` are used `[VERIFIED: LOCAL crates/cb-backend/Cargo.toml cubecl.workspace,
  bytemuck.workspace]`.

---

## 3. Dependencies and the crate-placement decision (LOCKED)

### 3.1 The architectural decision

**Decision:** the GPU-apply orchestration is split across THREE crates, so that
`cb-model` stays backend-free and no crate cycle is introduced:

| Layer | Crate | What lives here | Why |
|---|---|---|---|
| Pure guard + flattener | `cb-model` | `gpu_apply` module: exportability guard + `Model → FlatObliviousF64` (NO cubecl import) | Keeps the MODEL-02 boundary intact — the flattener is pure `Model` inspection, next to `export/onnx.rs` / `export/coreml.rs`, unit-testable on CPU. |
| `#[cube]` kernel + launch helper | `cb-backend` | `apply_oblivious_float_kernel` + `launch_apply_oblivious_f64` (Model-AGNOSTIC: takes flat slices) | `cb-backend` is the SOLE cubecl owner (D-03); it must NOT depend on `cb-model` (cycle), so the helper takes flat arrays, not a `Model`. |
| Marshalling + facade | `catboost-rs` | `Model::predict_raw_on_device`: guard → flatten (cb-model) → launch (cb-backend) → `Vec<f64>` | The facade ALREADY normally depends on BOTH `cb-model` and `cb-backend`; the natural join site. |

**Why not the two alternatives the research left open:**

- *Promote `cb-backend` to a normal dep of `cb-model`* — rejected: violates the
  explicit MODEL-02 boundary (`apply.rs:1-6` "imports nothing from the backend")
  and the feature-unification discipline
  `[VERIFIED: CODEGRAPH apply.rs:1-6; LOCAL crates/cb-model/Cargo.toml — cb-backend
  is a DEV-dependency only]`.
- *Route inference through `cb-train`'s existing `cb-backend` edge* — rejected:
  impossible for a function taking `&cb_model::Model`, because `cb-model` depends
  on `cb-train` (`cb-train` cannot name `cb_model::Model` without a cycle)
  `[VERIFIED: LOCAL crates/cb-model/Cargo.toml — cb-train is a NORMAL dependency of
  cb-model]`.

The facade route is the ONLY placement that (a) preserves MODEL-02, (b)
introduces no crate cycle, (c) reuses existing normal-dependency edges, and (d)
sits next to the existing facade `predict_raw` call site
`[VERIFIED: CODEGRAPH predict_raw blast radius includes crates/catboost-rs/src/model.rs;
LOCAL crates/catboost-rs/Cargo.toml lists cb-model + cb-backend as normal deps]`.

### 3.2 Reused symbols (no rewrite)

| Dependency | Kind | Evidence |
|---|---|---|
| CPU `cb_model::predict_raw` (the numeric oracle) | reuse (compare target) | `[VERIFIED: CODEGRAPH crates/cb-model/src/apply.rs:370]` |
| `cb_train::leaf_index` (forward-bit-order semantics to mirror on device) | reuse (semantics) | `[VERIFIED: CODEGRAPH crates/cb-train/src/tree.rs:293-301]` |
| `Model` fields `oblivious_trees / bias / approx_dimension / ctr_data / non_symmetric_trees / region_trees` | reuse (guard + flatten) | `[VERIFIED: CODEGRAPH crates/cb-model/src/model.rs:271-313]` |
| `ObliviousTree { splits, leaf_values }` + `ModelSplit::Float(Split{feature,border})` | reuse (flatten source) | `[VERIFIED: CODEGRAPH model.rs:70-98; boosting.rs:770-790]` |
| `is_onnx_exportable` guard shape | reuse (pattern) | `[VERIFIED: CODEGRAPH crates/cb-model/src/export/onnx.rs:99-115]` |
| `apply_leaf_delta_kernel` (per-object gather `#[cube]` shape) | reuse (kernel pattern) | `[VERIFIED: CODEGRAPH crates/cb-backend/src/kernels.rs:588-598]` |
| `launch_block_reduce_f64` (host launch: client/create/launch/read_one, CubeCount/CubeDim) | reuse (launch pattern) | `[VERIFIED: CODEGRAPH crates/cb-backend/src/gpu_runtime/mod.rs:220-260]` |
| `SelectedRuntime` compile-time alias (cpu/wgpu/cuda/rocm) | reuse (backend selection) | `[VERIFIED: CODEGRAPH crates/cb-backend/src/lib.rs:59-82]` |
| `SCORE_BOUND` report-bound convention (1e-9 f64 / 1e-3 wgpu) | reuse (tolerance) | `[VERIFIED: CODEGRAPH crates/cb-backend/src/kernels/score_split.rs:70-73]` |
| `cb_core::{CbError, CbResult}` (typed errors, no `unwrap`) | reuse | `[VERIFIED: CODEGRAPH crates/cb-backend/src/gpu_runtime/mod.rs:258]` |
| `cubecl = 0.10.0`, `bytemuck` (workspace-pinned, cb-backend-only) | reuse | `[VERIFIED: LOCAL crates/cb-backend/Cargo.toml]` |

No new external crate is required.

### 3.3 CubeCL manual chapters governing the kernel (GINF-01-S3)

Inference is a per-object gather/compare, NOT a cross-thread reduction, so the
histogram-specific chapters (atomics, LDS privatization, plane-sum) do **not**
apply. The load-bearing chapters, confirmed by reading the INDEX personally:

- **Generics** — the kernel MUST be `#[cube(launch)] fn …<F: Float>` (generic-float
  mandate; CLAUDE.md/AGENTS.md) `[VERIFIED: LOCAL INDEX.md → Cubecl_generics.md]`.
- **Conditionals** — "Avoid If Expressions": use `let mut x = default; if c { x = … }`
  statement form, never `let x = if c {a} else {b}`
  `[VERIFIED: LOCAL Cubecl_conditionals.md:22-41]`.
- **Plane Alignment / branch divergence** — the per-split bit is computed by
  predication (`mask = u32::cast_from(passes)`, `leaf += mask << i`), NOT a
  per-split branch, to avoid intra-plane divergence
  `[VERIFIED: LOCAL plane_alignment.md:17-19,55-65,89-91]`.
- **Multi-threading** — `ABSOLUTE_POS`-indexed, one object per thread, bounds
  guard `if ABSOLUTE_POS < n_objects`, `CubeCount`/`CubeDim` sized as in
  `launch_block_reduce_f64` `[VERIFIED: LOCAL INDEX.md → Cubecl_multi_threading.md;
  CODEGRAPH gpu_runtime/mod.rs:237-242]`.
- **Memory Coalescing** — the per-object feature matrix is uploaded FEATURE-MAJOR
  (`features[f * n_objects + obj]`) so adjacent object-threads touch adjacent
  addresses (mirrors the feature-major cindex the histogram kernels read)
  `[VERIFIED: LOCAL INDEX.md → 07_memory_coalescing.md; INFERRED: from cindex feature-major layout cindex.rs:77-86]`.
- **Launch Overhead** — the model arrays (splits/borders/tree-offsets/leaf-values)
  are invariant across all objects → uploaded ONCE; only the feature matrix is
  per-call `[VERIFIED: LOCAL INDEX.md → 11_launch_overhead_and_transfers.md]`.
- **Grid-Stride Loops** — for object counts exceeding one grid, the stride loop
  (`while cell < total { … cell += stride }`, as `read_all_bins_kernel` uses) is
  the scaling form; the first slice MAY use the bounded `ABSOLUTE_POS < n` form
  and note grid-stride as the scale follow-up `[VERIFIED: CODEGRAPH kernels.rs:2721-2728;
  LOCAL INDEX.md → 10_grid_stride_occupancy.md]`.
- **Autotuning** — per-shape launch-geometry tuning is a v2 optimization, not
  required for the correctness-first slice `[VERIFIED: LOCAL INDEX.md → 12_autotuning.md]`.

---

## 4. Typed contracts

```rust
// crates/cb-model/src/gpu_apply.rs  (NEW FILE — pure, NO cubecl import)

/// Typed failure at the GPU-apply boundary (mirrors `OnnxExportError`).
#[derive(Debug, thiserror::Error)]
pub enum GpuApplyUnsupported {
    #[error("model uses categorical/CTR features, unsupported by the GPU apply first slice")]
    CategoricalFeatures,
    #[error("model contains non-symmetric (Lossguide/Depthwise) trees, unsupported")]
    NonObliviousTrees,
    #[error("model contains region-path trees, unsupported")]
    RegionTrees,
    #[error("model is multi-dimensional (approx_dimension > 1), unsupported")]
    MultiDimensional,
}

/// The device-ready flat lowering of a float-only oblivious scalar model. All
/// per-object-invariant model state, uploaded ONCE per apply call.
#[derive(Debug, Clone, PartialEq)]
pub struct FlatObliviousF64 {
    /// Concatenated split FLOAT-feature indices across all trees (u32 device index).
    pub split_features: Vec<u32>,
    /// Concatenated split borders (f64), 1:1 with `split_features`.
    pub split_borders: Vec<f64>,
    /// Per-tree start offset into `split_features`/`split_borders`; length trees+1
    /// (CSR-style; `tree t` owns `[tree_split_offsets[t], tree_split_offsets[t+1])`).
    pub tree_split_offsets: Vec<u32>,
    /// Concatenated leaf values across all trees (f64), already learning_rate-scaled.
    pub leaf_values: Vec<f64>,
    /// Per-tree start offset into `leaf_values`; length trees+1.
    pub tree_leaf_offsets: Vec<u32>,
    /// The model bias, added exactly once per object.
    pub bias: f64,
}

/// GINF-01-S1: admit ONLY a float-only, oblivious, scalar model (deterministic
/// check order: non-symmetric → region → CTR → multi-dim → Ok). Pure; no I/O.
/// # Errors
/// A `GpuApplyUnsupported` variant naming the first disqualifying property.
pub fn check_gpu_apply_supported(model: &Model) -> Result<(), GpuApplyUnsupported>;

/// GINF-01-S2: lower a supported model into `FlatObliviousF64`. Calls the guard
/// first, then concatenates each tree's float splits + leaf values with CSR-style
/// per-tree offsets, checked-casting indices to u32.
/// # Errors
/// `cb_core::CbError::Unsupported` — carries the guard rejection (converted; NOT
/// `OutOfRange`, which is reserved for the index-overflow case below) OR
/// `CbError::OutOfRange` if a concatenated index/offset exceeds `u32::MAX`
/// (mirrors `PackedCindex::device_arrays`). Never panics.
pub fn flatten_oblivious_f64(model: &Model) -> cb_core::CbResult<FlatObliviousF64>;
```

```rust
// crates/cb-backend/src/kernels.rs  (ADD — production #[cube] body lives here)

/// GINF-01-S3: one raw prediction per object for a float-only oblivious ensemble.
/// Generic-float; branchless per-split predication; per-tree accumulate + bias.
/// FEATURE-MAJOR object matrix: `features[f * n_objects + obj]`.
#[cube(launch)]
pub fn apply_oblivious_float_kernel<F: Float>(
    features: &Array<F>,          // feature-major, len = n_features * n_objects
    split_features: &Array<u32>,  // concatenated split feature indices
    split_borders: &Array<F>,     // concatenated split borders (cast to F host-side)
    tree_split_offsets: &Array<u32>,
    leaf_values: &Array<F>,
    tree_leaf_offsets: &Array<u32>,
    out: &mut Array<F>,           // len = n_objects
    bias: &Array<F>,              // length-1: a #[cube(launch)] scalar must be a concrete
                                  // CubeElement, not the generic F (house convention —
                                  // cf. find_optimal_split_kernel's Array<F> lambda);
                                  // read as bias[0]. u32 counts stay plain scalars.
    n_objects: u32,
    n_trees: u32,
);
```

```rust
// crates/cb-backend/src/gpu_runtime/mod.rs  (ADD — Model-AGNOSTIC launch helper)

/// GINF-01-S4: launch `apply_oblivious_float_kernel` on `SelectedRuntime` over a
/// feature-major object matrix + the flat model arrays, returning one f64 raw
/// prediction per object. Uploads the invariant model arrays once.
/// # Errors
/// `CbError::Degenerate` on a device read-back failure (never a zero buffer);
/// `CbError::OutOfRange` on a length/shape precondition violation.
#[allow(clippy::too_many_arguments)]
pub fn launch_apply_oblivious_f64(
    features_feature_major: &[f64],
    n_objects: usize,
    n_features: usize,
    split_features: &[u32],
    split_borders: &[f64],
    tree_split_offsets: &[u32],
    leaf_values: &[f64],
    tree_leaf_offsets: &[u32],
    bias: f64,
) -> CbResult<Vec<f64>>;
```

```rust
// crates/catboost-rs/src/model.rs  (ADD — facade marshalling)

/// GINF-01-S5: GPU inference for float-only oblivious scalar models. Marshals the
/// inner model (guard → flatten → device launch) into raw predictions on the
/// compile-time-selected backend.
///
/// `n_objects` is derived EXCLUSIVELY from `features.first()`'s length —
/// mirroring `predict_raw_cat`'s `n_float = feature_values.first().map_or(0,
/// Vec::len)` bit-for-bit (`apply.rs:397-399`), NOT a max-over-columns rule. A
/// later column LONGER than the first is silently truncated to `n_objects`
/// (its tail is never read, exactly as CPU does); a later column SHORTER than
/// the first is NaN-padded per-cell (`apply.rs:404-407`). This asymmetry
/// (first-column-governs, not longest-column-governs) is corrected here after
/// PLAN-CHECK pass 2 found it unspecified and unexercised by the ragged-column
/// test added in pass 1.
/// # Errors
/// [`CatBoostError`] wrapping the flattener/guard `CbError` (unsupported model)
/// or the underlying device `CbError`. Never panics.
pub fn predict_raw_on_device(&self, features: &[Vec<f32>]) -> Result<Vec<f64>, CatBoostError>;
```

**Note on the error surface (resolves §9 Q1):** a guard rejection AND a u32
device-index overflow are BOTH surfaced as `cb_core::CbError` from
`flatten_oblivious_f64`, but as TWO DISTINCT variants, pinned explicitly (not left
to implementer discretion, per PLAN-CHECK pass 1 MINOR finding): the guard
rejection converts `GpuApplyUnsupported` to `CbError::Unsupported` (semantically
"this model kind cannot run this operation" — `cb_core::error.rs:86-92`), while an
index/offset overflow uses `CbError::OutOfRange` (mirroring
`PackedCindex::device_arrays` `[VERIFIED: CODEGRAPH cindex.rs:93-108]`). This keeps
the whole device path on ONE error type (`CbResult`) with a clean two-variant
taxonomy, so the facade maps a single `CbError → CatBoostError` via `?` and likely
needs NO new `CatBoostError` variant — confirm in TASK-05 by inspecting
`crates/catboost-rs/src/error.rs` for an existing `#[from] cb_core::CbError` arm
before adding one.

---

## 5. Failure-isolated behavioral specifications

Each spec has ONE primary reason a failing acceptance test would fail.

### GINF-01-S1 — Device-apply exportability guard (pure, cb-model)
- **Status:** implemented
- **Responsibility:** admit only float-only / oblivious / scalar models; reject
  everything else with the first-disqualifying typed variant. Nothing else.
- **Input:** any `&Model`. **Output:** `Result<(), GpuApplyUnsupported>`.
- **Dependencies:** `Model.{non_symmetric_trees, region_trees, ctr_data,
  oblivious_trees, approx_dimension}`, `ModelSplit::Ctr`.
- **Behavior (Given/When/Then):**
  - Given a float-only oblivious scalar model; When `check_gpu_apply_supported`;
    Then `Ok(())`.
  - Given a non-symmetric / region / CTR (`ctr_data.is_some()` or any
    `ModelSplit::Ctr`) / multi-dim (`approx_dimension > 1`) model; Then the
    matching `Err` variant (deterministic check order).
- **Invariants:** total function; no panic; check order fixed (mirrors
  `is_onnx_exportable`).
- **Primary failure reason:** the guard admits/rejects the wrong model kind.
- **Out of scope:** flattening (S2). **Acceptance:** `gpu_apply_guard` unit tests.
- **Traceability:** `[VERIFIED: CODEGRAPH onnx.rs:99-115; apply.rs:157-199; model.rs:271-313]`.

### GINF-01-S2 — Oblivious-float flattener (pure, cb-model)
- **Status:** implemented
- **Responsibility:** lower a supported `Model` into `FlatObliviousF64` (CSR-style
  per-tree offsets over concatenated splits + leaves); delegates rejection to S1.
- **Input:** `&Model`. **Output:** `cb_core::CbResult<FlatObliviousF64>`
  (guard-reject and u32-overflow both typed).
- **Behavior (Given/When/Then):**
  - Given a known 2-tree float model; When flattened THEN re-evaluated on the HOST
    from the flat arrays (a reference reconstruction of `leaf_index` + per-tree
    gather + bias); Then the result equals `cb_model::predict_raw` for the same
    objects, exactly (both f64, same order) — a pure CPU round-trip, no device.
  - Given an unsupported model; Then the guard error (as `CbError`).
- **Invariants:** `tree_split_offsets.len() == tree_leaf_offsets.len() ==
  n_trees + 1`; monotonic offsets; `split_features.len() == split_borders.len()`.
- **Primary failure reason:** the flat layout mis-encodes tree structure (leaf or
  split offsets wrong) so the host reconstruction diverges from `predict_raw`.
- **Out of scope:** any device launch (S3/S4). **Acceptance:**
  `flatten_roundtrip_matches_cpu` unit test.
- **Traceability:** `[VERIFIED: CODEGRAPH apply.rs:208-215,318-355; tree.rs:293-301]`.

### GINF-01-S3 — Oblivious-float apply `#[cube]` kernel (cb-backend)
- **Status:** implemented
- **Responsibility:** compute one raw prediction per object on device: for each
  tree, forward-bit-order leaf via branchless predication, gather the tree's leaf
  value, accumulate across trees, add bias once.
- **Input:** the flat arrays + feature-major object matrix (per §4).
  **Output:** `out[obj] = bias + Σ_t leaf_values[tree_leaf_offsets[t] +
  leaf_index_t(obj)]`.
- **Dependencies:** CubeCL Generics / Conditionals / Plane-Alignment /
  Multi-threading / Launch-Overhead chapters (§3.3).
- **Behavior (Given/When/Then):**
  - Given a tiny fixed flat model + object matrix uploaded to the device; When the
    kernel runs under the `cpu` backend (`CpuRuntime`, f64); Then `out` equals a
    host reference (the S2 reconstruction) exactly.
  - The per-split bit is `leaf += u32::cast_from(value > border) << i` (predication,
    statement form) — no `let x = if … {} else {}` expression.
- **Invariants:** generic-float; bounds-guarded (`ABSOLUTE_POS < n_objects`); an
  out-of-range leaf index contributes `F::new(0.0)` (mirrors CPU `unwrap_or(0.0)`).
- **Primary failure reason:** the on-device leaf index or per-tree accumulation is
  wrong (divergence-induced or bit-order/gather bug).
- **Out of scope:** host marshalling (S4). **Acceptance:** `apply_oblivious_kernel_oracle`
  device-vs-host child-module test (`cpu` backend).
- **Traceability:** `[VERIFIED: CODEGRAPH kernels.rs:588-598 (per-object gather),
  :2690 read_bin (compare-shape); LOCAL Cubecl_conditionals.md; plane_alignment.md]`.

### GINF-01-S4 — Launch host helper (cb-backend, Model-agnostic)
- **Status:** implemented
- **Responsibility:** upload the invariant model arrays + the feature matrix,
  launch S3 over `ceil(n_objects / CUBE_DIM)` cubes, read back `Vec<f64>`; typed
  error on read-back / shape failure.
- **Input:** flat slices + feature-major object matrix (per §4). **Output:**
  `CbResult<Vec<f64>>` (length `n_objects`).
- **Dependencies:** `SelectedRuntime`, `launch_block_reduce_f64` shape.
- **Behavior (Given/When/Then):**
  - Given the S2 flat arrays for a known model + a small object matrix; When
    `launch_apply_oblivious_f64`; Then the returned vector equals `predict_raw`
    within the `SCORE_BOUND` convention (≤1e-9 under `cpu`/f64).
  - Given a zero-object matrix; Then `Ok(vec![])` (no launch).
  - Given a length/shape violation (e.g. `features.len() != n_features *
    n_objects`); Then `Err(CbError::OutOfRange)` (no panic).
  - Given a device read-back failure; Then `Err(CbError::Degenerate)` (never a
    zero buffer — WR-05).
- **Primary failure reason:** buffer marshalling / launch geometry / read-back is
  wrong so the returned vector diverges or the helper panics.
- **Out of scope:** the facade `Model` marshalling (S5). **Acceptance:**
  `launch_apply_oblivious_oracle` integration test (`cpu` backend).
- **Traceability:** `[VERIFIED: CODEGRAPH gpu_runtime/mod.rs:220-260]`.

### GINF-01-S5 — Facade `predict_raw_on_device` + device↔CPU parity (catboost-rs)
- **Status:** implemented
- **Responsibility:** guard → flatten (cb-model) → feature-major marshal → launch
  (cb-backend) → `Vec<f64>`; map errors to `CatBoostError`.
- **Input:** `&self` (facade `Model`), `features: &[Vec<f32>]` (per-feature
  columns, same shape as `predict_raw`). **Output:**
  `Result<Vec<f64>, CatBoostError>`.
- **Behavior (Given/When/Then):**
  - Given a real trained float-only oblivious scalar model + a batch; When
    `predict_raw_on_device`; Then the result matches `self.predict_raw(...)`
    within `SCORE_BOUND` (≤1e-9 under `cpu`/f64; confirmed on the ROCm rig; ≤1e-3
    smoke-only under wgpu-f32).
  - Given a CTR / non-symmetric / multiclass model; Then `Err(CatBoostError)`
    (from S1), no device launch.
- **Primary failure reason:** end-to-end device prediction diverges from the CPU
  oracle beyond the bound, or an error is not mapped / a panic crosses the boundary.
- **Out of scope:** throughput (S6). **Acceptance:** `predict_on_device_matches_cpu`
  facade oracle + `predict_on_device_rejects_unsupported`.
- **Traceability:** `[VERIFIED: CODEGRAPH apply.rs:370 predict_raw; catboost-rs/src/model.rs
  (predict_raw caller); score_split.rs:70-73 SCORE_BOUND]`.

### GINF-01-S6 — Throughput benchmark (catboost-rs, non-gating)
- **Status:** implemented
- **Responsibility:** an `#[ignore]` benchmark timing device vs CPU apply over a
  large batch, reporting a stage attribution (upload / launch / read-back) in the
  `CB_GPU_PROF` style used for the training kernels.
- **Input:** a large synthetic float model + batch. **Output:** printed timings;
  a correctness assertion (device == CPU within `SCORE_BOUND`) as a guard.
- **Behavior:** Given a large batch; When run with `--ignored`; Then it prints
  device/CPU wall time and asserts parity; it does NOT gate CI.
- **Primary failure reason:** the benchmark asserts parity and fails (a
  correctness regression) — timing itself is informational.
- **Out of scope:** any Kaggle/CUDA vs-upstream comparison (§9 R3, optional stretch).
- **Acceptance:** `bench_predict_on_device` `#[ignore]` test.
- **Traceability:** `[VERIFIED: LOCAL MEMORY gpu-train-speed-round2 CB_GPU_PROF stage attribution]`.

---

## 6. Acceptance scenarios

| ID | Scenario | Oracle | Tolerance |
|----|----------|--------|-----------|
| AT-S1 | supported model → Ok; CTR/non-sym/region/multi-dim → matching Err | unit | — |
| AT-S2 | flatten → host reconstruction == `predict_raw` (2-tree model) | pure CPU round-trip | exact (f64) |
| AT-S3 | kernel `out` == host reference on a tiny fixed model | device-vs-host (`cpu` backend) | exact / ≤1e-9 |
| AT-S4 | `launch_apply_oblivious_f64` == `predict_raw`; empty/shape/read-back paths | integration (`cpu`) | ≤1e-9; typed errs |
| AT-S5a | facade `predict_raw_on_device` == `predict_raw` on a trained model | facade oracle (`cpu`; ROCm confirm) | ≤1e-9 (f64) |
| AT-S5b | unsupported model → `Err(CatBoostError)` | facade | — |
| AT-S5c | ragged/unequal-length feature columns, INCLUDING a case where a NON-FIRST column is LONGER than the first column (the only combination that distinguishes "first-column-governs `n_objects`" from "max-over-columns", per the corrected S5 contract) → device matches CPU's truncated-to-first-column-length behavior on every object | facade oracle (`cpu`) | ≤1e-9 (f64) |
| AT-S6 | large-batch device==CPU parity guard; timings printed | `#[ignore]` bench | ≤1e-9 |

---

## 7. Impact scope

- **`crates/cb-model/src/gpu_apply.rs`** (NEW) — `local`. Pure guard + flattener +
  `GpuApplyUnsupported`; NO cubecl import (MODEL-02 preserved). `apply.rs`
  untouched (D-04 no-regression on the CPU path).
- **`crates/cb-model/src/lib.rs`** — `local`. `pub mod gpu_apply;` +
  `pub use gpu_apply::{check_gpu_apply_supported, flatten_oblivious_f64,
  FlatObliviousF64, GpuApplyUnsupported};` (so the facade can name them; mirrors
  the ONNX re-export) `[VERIFIED: CODEGRAPH crates/cb-model/src/lib.rs:37]`.
- **`crates/cb-backend/src/kernels.rs`** — `local`. New `#[cube]` body next to
  `apply_leaf_delta_kernel`; existing kernels byte-identical.
- **`crates/cb-backend/src/gpu_runtime/mod.rs`** — `local`. New launch helper next
  to `launch_block_reduce_f64`; existing helpers untouched.
- **`crates/catboost-rs/src/model.rs`** — `cross-module`. New `predict_raw_on_device`
  method; no change to `predict` / `predict_raw` signatures (additive).
- **`crates/catboost-rs/src/error.rs`** — `cross-module` (CONDITIONAL). Only if the
  facade needs a new `CatBoostError` arm — but the single-`CbError` error surface
  (§4 note) is designed to reuse an existing `#[from] cb_core::CbError` arm, so no
  edit is expected. If a NEW non-wildcard variant IS added, mirror the coreml
  precedent: a `catboost-rs-py::to_pyerr` exhaustive match would need a new arm
  (E0004) — but this slice adds NO Python surface, so re-check whether that mapper
  even references the new variant `[VERIFIED: CODEGRAPH .planning/plans/coreml-export/SPEC.md §8]`.
- **Tests** — cb-model unit (`gpu_apply_test.rs`, mounted), cb-backend kernel
  child-module oracle + launch integration test, facade oracle + `#[ignore]` bench.

No persistence / schema / wire-format change. No Python surface in this slice
(the facade Rust method is the deliverable; a PyO3 `predict_on_device` is a
follow-on if desired). No existing public contract changes.

---

## 8. Compatibility and migration

- **Additive.** No existing signature, serialization format, or CPU apply behavior
  changes; `apply.rs` and the shipped `predict_raw` are read-only dependencies
  (they are the oracle). Backend selection reuses the existing compile-time
  `cpu`/`wgpu`/`cuda`/`rocm` feature passthrough — no runtime dispatch (D-02).
- **Error typing designed to stay additive:** the whole device path is `CbResult`,
  so the facade reuses an existing `cb_core::CbError → CatBoostError` conversion;
  no new exhaustive-match arm is expected. The conditional E0004 hazard (coreml
  precedent) is avoided by design and re-checked in TASK-05.
- Rollback = revert the additive module + kernel + helper + facade method.

---

## 9. Risks and open questions

| Risk | Consequence | Mitigation |
|------|-------------|------------|
| **R1 — cross-backend accumulation parity.** The per-tree/bias-insertion ASSOCIATION is locked to match the CPU oracle exactly (ascending tree order seeded at `0.0`, `bias` added once at the end — see the kernel algorithm, PLAN TASK-03), so this risk is now scoped to backend-level rounding-mode/FMA-fusion/codegen differences ONLY, not association order. | Device != CPU bit-for-bit across DIFFERENT backends/hardware. | Sign off to the project's `SCORE_BOUND` REPORT convention (≤1e-9 f64), NOT exact — the same report-not-sign-off framing the training kernels use `[VERIFIED: CODEGRAPH score_split.rs:68-73]`. For typical ensembles residual rounding error is ~1e-12/tree, well within 1e-9. Documented as a non-goal (§2). The accumulation-order/bias-insertion-order divergence originally described here was corrected in PLAN.md TASK-03 after PLAN-CHECK pass 1 flagged it as a MAJOR issue; this row now covers only the residual, unavoidable cross-backend rounding risk. |
| **R2 — wgpu has no f64.** Borders are f64; an f32 device compare can flip a near-border split → different leaf → large prediction diff. | wgpu predictions can diverge >1e-3 on adversarial near-border inputs. | wgpu is COMPILE/SMOKE-verified only, NOT a numeric sign-off; the numeric gate runs under `cpu` (f64) + ROCm (f64). Explicitly stated (§2 non-goals, §6). |
| **R3 — oracle convention differs from GPU-training's Kaggle-only flow.** | A checker may flag the deviation. | JUSTIFIED: unlike training (nondeterministic quantization → Kaggle needed), inference on a FIXED model is deterministic and has a shipped in-process CPU oracle (`predict_raw`). The parity test therefore runs the SAME `#[cube]` kernel under the `cpu` `CpuRuntime` (f64) in ordinary `cargo test` — no GPU hardware needed — with ROCm as GPU confirmation. This is a STRONGER, cheaper oracle than Kaggle, and is the correct convention here. `[VERIFIED: CODEGRAPH lib.rs:33-63 CpuRuntime is a real SelectedRuntime; kernels run under it]` |
| **R4 — read_bin/quantized-bin temptation.** The research suggested encoding splits as `read_bin` bin lookups. | A device quantization step would add bin-vs-border rounding and diverge from CPU. | Reject read_bin for the first slice; compare RAW `f32` value vs `f64` border directly (byte-mirror `passes_float_split`). Bin-packing is a v2 memory optimization (§1). |
| **R5 — clippy gate, not build.** `unwrap`/`expect`/`panic`/`indexing_slicing` are DENY in prod. | CI red despite `cargo build` green. | Gate each prod change with `cargo clippy -p <crate> --lib --no-deps`; integration tests carry the `#![allow(...)]` header the existing oracle tests use `[VERIFIED: CODEGRAPH score_split.rs; calc-metrics PLAN §3]`. |
| **R6 — test-mount omission runs 0 tests.** | False green. | Mount unit tests via `#[cfg(test)] #[path="…_test.rs"] mod tests;`; mount the kernel oracle as a child module of `kernels` (mirror the `kernels::reduce` child-module precedent) `[VERIFIED: CODEGRAPH cb-backend/src/lib.rs:10-19]`. |
| **R7 — CubeCL build errors on the kernel.** | Blind fixes waste time / break parity. | Per project rule, on ANY CubeCL build error the implementer MUST consult `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/cubecl_error_guideline.md` BEFORE attempting a fix (see PLAN §3). |

**Open questions (resolved at plan time, not left dangling)**

1. **Error surface for the flattener** (u32 overflow vs guard reject): RESOLVED in
   §4 note — flattener returns `CbResult<FlatObliviousF64>`, converting
   `GpuApplyUnsupported` into `CbError` and using `CbError::OutOfRange` for
   overflow (mirrors `PackedCindex::device_arrays`). `[RESOLVED IN PLAN]`
2. **New `CatBoostError` variant vs reuse of a `cb_core::CbError` arm** (S5 /
   py-mapper E0004): RESOLVED by design (single-`CbError` device path) + a TASK-05
   inspection of `crates/catboost-rs/src/error.rs` before adding any variant.
   `[RESOLVED IN PLAN]`
3. **Kernel child-module mount site** for the oracle test — confirm the exact
   `kernels::<child>` mount pattern in `kernels.rs` at implementation time.
   `[INFERRED — cb-backend/src/lib.rs:10-19; confirm in TASK-03]`

---

## 10. Traceability and sources

- **CPU oracle / semantics:** `crates/cb-model/src/apply.rs:{1-6 MODEL-02,
  136-140 passes_float_split, 208-215 leaf_index_for, 318-355 predict_raw_one,
  370 predict_raw, 386-407 predict_raw_cat (n_objects derivation + NaN-pad
  gather)}`; `crates/cb-train/src/tree.rs:293-301 leaf_index`
  `[VERIFIED: CODEGRAPH]`.
- **Guard pattern:** `crates/cb-model/src/export/onnx.rs:99-115 is_onnx_exportable`
  `[VERIFIED: CODEGRAPH]`.
- **Kernel/launch precedents:** `crates/cb-backend/src/kernels.rs:588-598
  apply_leaf_delta_kernel, :2690 read_bin, :2709-2729 read_all_bins_kernel (stride
  loop)`; `crates/cb-backend/src/gpu_runtime/mod.rs:220-260 launch_block_reduce_f64`;
  `crates/cb-backend/src/lib.rs:59-82 SelectedRuntime`;
  `crates/cb-backend/src/kernels/score_split.rs:68-73 SCORE_BOUND` `[VERIFIED: CODEGRAPH]`.
- **Model fields:** `crates/cb-model/src/model.rs:271-313` `[VERIFIED: CODEGRAPH]`.
- **Crate deps:** `crates/cb-model/Cargo.toml` (cb-backend DEV-only; cb-train
  normal), `crates/catboost-rs/Cargo.toml` (cb-model + cb-backend normal),
  `crates/cb-backend/Cargo.toml` (cubecl/bytemuck; feature arms) `[VERIFIED: LOCAL]`.
- **CubeCL manual:** `INDEX.md`, `Cubecl_conditionals.md:22-41`,
  `plane_alignment.md:17-19,55-91` (read this session) `[VERIFIED: LOCAL]`.
- **Sibling precedents (house style + float-only slicing + oracle-caveat framing):**
  `.planning/phases/20-orchestration/calc-metrics/{SPEC,PLAN}.md`,
  `.planning/plans/coreml-export/SPEC.md`, `.planning/plans/sum-models/` `[VERIFIED: LOCAL]`.
- **Research:** `.planning/plans/next-feature-research/` + the in-session GINF-01
  research pass `[VERIFIED: LOCAL git status shows the dir]`.

---

## 11. Implementation evidence

> GINF-01-S1..S6 are `implemented` and verified (2026-07-20). TreeFinder/PageIndex
> MCP does not hold this slice (frontmatter `treefinder_pending`; a corpus search
> for GINF-01 returns no document), so this local SPEC is the effective spec
> store, matching the calc-metrics / coreml-export siblings.
>
> **S1/S2 implementation evidence (2026-07-20):**
> - Source: `crates/cb-model/src/gpu_apply.rs`
>   (`check_gpu_apply_supported`, `flatten_oblivious_f64`, `FlatObliviousF64`,
>   `GpuApplyUnsupported`); re-exported from `crates/cb-model/src/lib.rs`.
> - Tests: `crates/cb-model/src/gpu_apply_test.rs` — `guard_accepts_float_oblivious_scalar`,
>   `guard_rejects_ctr`, `guard_rejects_non_symmetric`, `guard_rejects_region`,
>   `guard_rejects_multidim`, `flatten_roundtrip_matches_cpu` (exact-f64 vs
>   `predict_raw`), `flatten_offsets_invariants`, `flatten_rejects_unsupported`.
> - Verified: `cargo test -p cb-model --lib gpu_apply` (8 passed);
>   `cargo clippy -p cb-model --lib --no-deps` (0 findings in `gpu_apply.rs`).
>   MODEL-02 preserved (no `cb_backend` / `cubecl` import in `gpu_apply.rs`).
>
> **S3/S4 implementation evidence (2026-07-20):**
> - Source: `crates/cb-backend/src/kernels.rs::apply_oblivious_float_kernel`
>   (generic-float `#[cube(launch)]`; branchless per-split predication
>   `leaf += u32::cast_from(v > b) << i`; per-tree accumulate seeded at `0.0`;
>   `bias` added once AFTER the loop via a length-1 `Array<F>` — a `#[cube(launch)]`
>   scalar must be a concrete `CubeElement`, not the generic `F`, so the §4 kernel
>   signature was updated accordingly). Launch helper
>   `crates/cb-backend/src/gpu_runtime/mod.rs::launch_apply_oblivious_f64`
>   (Model-agnostic flat slices; f64 under cpu/rocm/cuda, f32 under wgpu with a
>   widened read-back; `n_trees` derived from `tree_split_offsets.len()-1`; typed
>   `OutOfRange`/`Degenerate` errors; no `unwrap`/`panic`/indexing in prod).
> - Tests: `crates/cb-backend/src/kernels/apply_oblivious_test.rs`
>   (`apply_oblivious_kernel_oracle`, device-vs-host, `cpu` f64, bias-added-last);
>   `crates/cb-backend/tests/apply_oblivious_launch_test.rs`
>   (`launch_matches_host_reference` ≤1e-9, `launch_empty_is_empty`,
>   `launch_shape_mismatch_errs` → `OutOfRange`).
> - Verified: `cargo test -p cb-backend --lib apply_oblivious` (1 passed);
>   `cargo test -p cb-backend --test apply_oblivious_launch_test` (3 passed);
>   generic-float kernel compiles on every backend —
>   `cargo check -p cb-backend --no-default-features --features {wgpu,cuda,rocm}`
>   all green. No new `cargo clippy -p cb-backend --lib --no-deps` findings in the
>   added code (the 4 pre-existing errors are in untouched files). No `cb-model`
>   dependency added to `cb-backend` (no crate cycle); existing kernels/launch
>   helpers untouched (D-04 additive).
>
> **S5/S6 implementation evidence (2026-07-20):**
> - Source: `crates/catboost-rs/src/model.rs::Model::predict_raw_on_device`
>   (guard+flatten via `cb_model::flatten_oblivious_f64(self.as_canonical())` →
>   `n_objects = features.first().map_or(0, Vec::len)` [bit-for-bit mirror of
>   `predict_raw_cat`'s `n_float`, apply.rs:397] → private
>   `Model::to_feature_major_f64` marshal [outer feature / inner object loop,
>   NaN-pads a shorter later column via checked `.get`, truncates a longer one] →
>   `cb_backend::gpu_runtime::launch_apply_oblivious_f64(...)`; both `?` map
>   `cb_core::CbError → CatBoostError` through the EXISTING
>   `CatBoostError::Train(#[from] cb_core::CbError)` arm — `error.rs` UNCHANGED, no
>   new variant, no py-mapper E0004). No `unwrap`/`expect`/`panic`/indexing in prod.
> - Tests: `crates/catboost-rs/src/model_device_test.rs`
>   (`predict_on_device_matches_cpu` — max|diff| `0e0` vs `cb_model::predict_raw`;
>   `predict_on_device_rejects_unsupported` — multi-dim model → typed `Err`;
>   `predict_on_device_matches_cpu_ragged_columns` — BOTH a shorter-later NaN-pad
>   config and a longer-later truncation config, each max|diff| `0e0`, and asserts
>   `n_objects` is the first-column length). `#[ignore]` bench
>   `crates/catboost-rs/tests/predict_on_device_bench_test.rs::bench_predict_on_device`
>   (200k-object batch, 200-tree/depth-6 synthetic RMSE regressor trained via the
>   public `CatBoostBuilder`; device 127.73 ms vs CPU `predict_raw` 7483.80 ms,
>   parity max|diff| `0e0`; `CB_GPU_PROF`-gated per-stage timing).
> - Verified: `cargo test -p catboost-rs predict_on_device` (3 lib tests passed);
>   `cargo test -p catboost-rs --test predict_on_device_bench_test -- --ignored`
>   (1 passed, parity `0e0`); default `cargo test -p catboost-rs` (45 lib + all
>   integration tests passed, bench correctly `ignored`);
>   `cargo clippy -p catboost-rs --lib --no-deps` (0 findings in `catboost-rs`);
>   `cargo check -p catboost-rs-py` (green — no E0004). Additive facade method
>   only; `predict`/`predict_raw`/`staged_predict` untouched (D-04).

| Spec | State | Planned source | Planned tests |
|------|-------|----------------|---------------|
| GINF-01-S1 | implemented | `crates/cb-model/src/gpu_apply.rs::check_gpu_apply_supported` | `gpu_apply_test.rs` (guard) |
| GINF-01-S2 | implemented | `gpu_apply.rs::{flatten_oblivious_f64, FlatObliviousF64}` | `gpu_apply_test.rs` (round-trip vs `predict_raw`) |
| GINF-01-S3 | implemented | `crates/cb-backend/src/kernels.rs::apply_oblivious_float_kernel` | `kernels::apply_oblivious_test::apply_oblivious_kernel_oracle` (device-vs-host, `cpu` f64) |
| GINF-01-S4 | implemented | `crates/cb-backend/src/gpu_runtime/mod.rs::launch_apply_oblivious_f64` | `tests/apply_oblivious_launch_test.rs` (`launch_matches_host_reference`, `launch_empty_is_empty`, `launch_shape_mismatch_errs`) |
| GINF-01-S5 | implemented | `crates/catboost-rs/src/model.rs::predict_raw_on_device` | `model_device_test.rs` (`predict_on_device_matches_cpu`, `predict_on_device_rejects_unsupported`, `predict_on_device_matches_cpu_ragged_columns`) |
| GINF-01-S6 | implemented | `crates/catboost-rs/tests/predict_on_device_bench_test.rs` `#[ignore]` bench | `bench_predict_on_device` |
</content>
</invoke>
