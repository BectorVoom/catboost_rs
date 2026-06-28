# Architecture Research

**Domain:** Device-resident GPU gradient-boosting training (full inner loop on-device) integrated with a compile-time-generic `Runtime` seam
**Researched:** 2026-06-28
**Milestone:** v1.1 GPU Performance (supersedes the v1.0 ARCHITECTURE research)
**Confidence:** HIGH (grounded in the existing catboost-rs code at `crates/cb-train` + `crates/cb-backend`, and the vendored upstream reference at `catboost-master/catboost/cuda/`)

---

## Executive Answer (the four sub-questions)

1. **Runtime seam shape:** add ONE coarse, OPTIONAL trait method — `grow_tree_on_device(...)` returning a host-materialized tree descriptor — with a `default impl { Ok(None) }`. The boosting loop tries the device grower and falls back to the existing host growers when it returns `None`/unsupported. Do NOT add fine-grained per-stage seams (`build_histograms`/`score`/`partition`) to the trait: that would force the `Runtime` trait (in `cb-compute`) to speak in device-handle types, dragging CubeCL concepts across the `cb-train`/`cb-backend` boundary and re-opening the landmine. Fine-grained composition belongs INSIDE `cb-backend`, behind the coarse seam.
2. **Device residency:** introduce a `cb-backend`-owned `GpuTrainSession` struct that owns ONE `ComputeClient` plus the persistent device handles (quantized features / compressed index, the running approx cursor, weights, partition/leaf-bins, partition-stats). It is created once per `fit()` and threaded through every iteration. The `Runtime` impl (`GpuBackend`) holds the session via interior mutability. Lifetime owner = the session, dropped at end of `fit()`.
3. **Compose, don't rewrite:** the kernels already exist (`launch_pointwise_hist2`, `launch_find_optimal_split_pointwise`, `launch_scan_update_pointwise`, `launch_partition_split_into`, `launch_partition_update_into`, the pairwise family, the der seam) and a single-tree driver `grow_oblivious_tree_into` + a multi-tree driver `grow_boosting_pass_into` already chain them. The work is (a) wire `grow_boosting_pass` into `cb-train` behind the coarse seam, (b) convert the `*_into` helpers from host-slice arguments to PERSISTENT-HANDLE arguments so data stops re-uploading every level/tree, and (c) extend the depth-1 MVP to depth>1 via the partition-aware (`fullPass=false`) histogram.
4. **Build order:** Phase 10 seam + wire depth-1 (residency refactor) → Phase 11 depth>1 partition-aware histogram + Newton der2 → Phase 12 CTR / pairwise / ordered / multiclass on-device → Phase 13 CUDA benchmark + ε sign-off.

---

## Standard Architecture

### Current state (v1.0 — derivatives-only GPU)

```
┌──────────────────────────────────────────────────────────────────────┐
│ cb-train  (generic boosting loop  train::<R: Runtime>)                 │
│   per iteration:                                                       │
│     runtime.compute_gradients(loss, approx, target)  ──► DEVICE        │
│     ders.der1.clone()  ◄── read back to host (sync stall)             │
│     greedy_tensor_search_oblivious_*  ──────────────► HOST  (~95%)     │
│       histogram / score / BestSplit / partition / leaf-values         │
└───────────────────────────────────┬──────────────────────────────────┘
                                     │ Runtime trait (cb-compute)
                                     │   fn compute_gradients(...)   ← only seam
                                     ▼
┌──────────────────────────────────────────────────────────────────────┐
│ cb-backend  GpuBackend : Runtime    (generic over SelectedRuntime)     │
│   der1/der2 kernels via the Phase-7.2 der seam  ──► CubeCL ──► GPU     │
│   UNWIRED: grow_oblivious_tree / grow_boosting_pass (tests only)       │
└──────────────────────────────────────────────────────────────────────┘
```

### Target state (v1.1 — device-resident inner loop)

```
┌──────────────────────────────────────────────────────────────────────┐
│ cb-train  (generic boosting loop  train::<R: Runtime>)                 │
│   once:    runtime.begin_device_training(quantized, target, params)?   │
│   per iter: if let Some(tree) = runtime.grow_tree_on_device(iter)? {   │
│                 use tree   ──── DEVICE (histogram→score→split→leaf)    │
│             } else { greedy_tensor_search_* host fallback }            │
│   end:     runtime.end_device_training()                              │
└───────────────────────────────────┬──────────────────────────────────┘
                                     │ Runtime trait (cb-compute)
                                     │   fn compute_gradients(...)        (unchanged)
                                     │   fn grow_tree_on_device(...) -> Option<DeviceTree>
                                     │   fn begin/end_device_training(...) default = no-op
                                     ▼
┌──────────────────────────────────────────────────────────────────────┐
│ cb-backend  GpuBackend : Runtime                                       │
│   owns  GpuTrainSession { client, compressed_index_h, approx_h,        │
│                           weight_h, bins_h, part_stats_h, der_h }      │
│   grow_tree_on_device = grow_oblivious_tree_into over RESIDENT handles │
│      ├─ launch_pointwise_hist2  (partition-aware, fullPass=false)      │
│      ├─ launch_scan_update_pointwise                                   │
│      ├─ launch_find_optimal_split_pointwise (O(1) BestSplit read-back) │
│      ├─ launch_partition_split_into  (in-place doc routing)           │
│      └─ launch_partition_update_into (per-leaf Σder/Σw read-back)      │
│   data stays on GPU across ALL iterations; only O(1) leaf stats cross  │
└──────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Status |
|-----------|----------------|--------|
| `Runtime` trait (`cb-compute/src/runtime.rs`) | Abstract compute seam the boosting loop drives. MUST stay free of CubeCL/handle types (D-03/D-04). | MODIFY — add coarse optional methods with default no-op impls |
| `train_inner::<R>` (`cb-train/src/boosting.rs`) | Generic boosting loop; today always falls through to host growers. | MODIFY — try device seam, fall back to host |
| host growers (`cb-train/src/tree.rs`) | `greedy_tensor_search_oblivious_*` — remain the CPU path + the fallback for unsupported device cases. | KEEP unchanged |
| `GpuBackend` (`cb-backend/src/gpu_backend.rs`) | The `Runtime` impl; today only `compute_gradients`. | MODIFY — implement the coarse device-grow seam, own the session |
| `GpuTrainSession` (NEW, `cb-backend`) | Owns the `ComputeClient` + all persistent device handles across iterations. | NEW |
| `grow_boosting_pass_into` / `grow_oblivious_tree_into` (`cb-backend/src/gpu_runtime/mod.rs`) | Multi-tree + single-tree device drivers. Already exist; depth-1 MVP. | MODIFY — handle-resident args, depth>1 |
| kernel launchers (`mod.rs`, `pairwise.rs`, `der_seams.rs`) | `#[cube]` launches: histogram, scan, score, partition split/update, der. | KEEP — add handle-accepting `*_into` variants |
| `SelectedRuntime` alias (`cb-backend/src/lib.rs`) | Compile-time backend pick (cpu/wgpu/cuda/rocm). | KEEP |

---

## The Runtime Seam Design (sub-question 1)

### Recommendation: a COARSE, OPTIONAL, host-typed seam

Add to `cb_compute::Runtime` (all with default impls so `CpuBackend` and existing impls are untouched):

```rust
// cb-compute/src/runtime.rs  — NO cubecl types appear here (D-03 preserved)
pub trait Runtime {
    fn compute_gradients(...) -> CbResult<Derivatives>;          // unchanged
    fn compute_gradients_grouped(...) -> CbResult<Vec<Derivatives>>; // unchanged

    /// Open a device-resident training session: upload the quantized features +
    /// target ONCE. Default = false (host-only backends ignore it).
    fn begin_device_training(&self, _ctx: &DeviceTrainCtx) -> CbResult<bool> { Ok(false) }

    /// Grow ONE tree fully on device for `iter`, returning a HOST-materialized
    /// tree descriptor. `Ok(None)` => "not supported, use the host grower"
    /// (e.g. depth>1 before Phase 11, CTR/pairwise before Phase 12).
    fn grow_tree_on_device(&self, _iter: usize) -> CbResult<Option<DeviceGrownTree>> { Ok(None) }

    fn end_device_training(&self) -> CbResult<()> { Ok(()) }
}
```

`DeviceTrainCtx` and `DeviceGrownTree` are PLAIN host structs in `cb-compute` (Vec<f64>, Vec<u32>, `(feature,bin)` splits, leaf_values, leaf_of) — the SAME shape as the existing `GrownTree` in `cb-backend`, but defined in `cb-compute` so the trait signature stays handle-free. `cb-backend` converts its internal `GrownTree` → `cb_compute::DeviceGrownTree` at the seam boundary. This is exactly how `Derivatives` already crosses today (host `Vec<f64>`, no CubeCL leakage).

### Why coarse, not fine-grained — the landmine analysis

| Option | Trait surface | cb-train↔cb-backend coupling | Verdict |
|--------|---------------|------------------------------|---------|
| **Coarse `grow_tree_on_device()`** | One method returning a host struct | `cb-train` only ever sees `Vec`-shaped descriptors; device-handle lifetime stays 100% inside `cb-backend`. The boosting loop stays genuinely generic. | RECOMMENDED |
| Fine-grained per-stage (`build_histograms`/`score`/`partition`/`update_leaves`) | 4–6 methods exchanging histograms/partitions | Either the trait speaks device-handle types (drags CubeCL into `cb-compute` → violates D-03, re-opens the landmine), OR every stage round-trips its histogram/partition to host as `Vec` (kills residency — the very thing we're fixing). The per-level loop control would also have to live in the generic `cb-train`, which then needs to know device-partition semantics. | REJECT |

The host-light per-level chaining (histogram → scan → score → BestSplit → partition split → partition update) is INHERENTLY a tight device-resident loop where intermediate buffers must never leave the GPU. That loop already lives correctly inside `grow_oblivious_tree_into` in `cb-backend`. Exposing its internal stages through the `Runtime` trait would either leak handles upward (landmine) or force read-backs (defeats the purpose). The right seam wraps the WHOLE per-tree loop.

**Fallback is first-class, not an afterthought.** `Ok(None)` from `grow_tree_on_device` lets the boosting loop transparently use the host grower for any case the device path doesn't cover yet (depth>1 pre-Phase-11, CTR/pairwise/ordered/multiclass pre-Phase-12, or any loss without a GPU der kernel). This is what makes the incremental build order safe — partial device coverage never breaks correctness, only speed.

---

## Device Residency Ownership Model (sub-question 2)

### The problem today

Even the EXISTING device driver re-uploads everything. `grow_oblivious_tree_into` uploads resident handles (`cindex_h`, `der1_h`, …) at `mod.rs:1704-1708` but then calls `launch_find_optimal_split_pointwise_into(client, der1, weight, cindex, indices, …)` with the **host slices** at `mod.rs:1722-1724` — so the score kernel re-uploads der1/weight/cindex/indices EVERY LEVEL. And `grow_boosting_pass_into` passes host `cindex`/`indices` into `grow_oblivious_tree_into` EVERY TREE (`mod.rs:1982-1985`), re-uploading the entire quantized feature matrix once per boosting iteration. Quantized features are immutable across the whole run, so this is pure waste — and the dominant residency bug to fix.

### Recommendation: a `GpuTrainSession` owns the lifetime

```rust
// cb-backend (NEW) — owns one client + all cross-iteration device buffers
pub struct GpuTrainSession {
    client:            ComputeClient<SelectedRuntime>,
    // immutable for the whole run (uploaded ONCE):
    compressed_index_h: Handle,   // quantized feature-major bins (cindex)
    indices_h:          Handle,   // object visiting order
    target_h:           Handle,
    weight_h:           Handle,
    n: usize, n_features: usize, n_bins: usize,
    // mutated in place per iteration:
    approx_h:           Handle,   // the running cursor  (upstream TBoostingCursors)
    der1_h:             Handle,   // recomputed device-side each tree
    // mutated in place per level within a tree:
    bins_h:             Handle,   // per-doc leaf assignment (upstream subsets.Bins)
    part_stats_h:       Handle,   // per-leaf Σder/Σweight   (upstream PartitionStats)
}
```

| Buffer | Lifetime | Mirrors upstream |
|--------|----------|------------------|
| `compressed_index_h`, `indices_h`, `target_h`, `weight_h` | whole `fit()` — uploaded once | `TDocParallelDataSet` compressed index (built once by `compressed_index_builder`) |
| `approx_h` | whole `fit()` — updated in place each iteration | `TBoostingCursors::Cursors` (`TStripeBuffer<float>`) |
| `der1_h` | recomputed device-side each tree from `approx_h` | per-iteration target derivative cursor |
| `bins_h`, `part_stats_h` | reset per tree, mutated per level | `TOptimizationSubsets::{Bins, PartitionStats}` |

**Owner of the lifetime:** the session is created in `begin_device_training` and dropped in `end_device_training` (or by `Drop` on the session). Because a CubeCL `Handle` is bound to the `ComputeClient` that allocated it (the "never read a 0-len handle / handle bound to its client" landmine from Phase 7.2/7.5), the session MUST hold the one client and thread `&self.client` through every launch — exactly the "thread ONE client" discipline `grow_*_into` already follows, lifted from per-tree scope to per-`fit()` scope.

**Where it lives relative to `GpuBackend`:** `GpuBackend` is currently zero-sized. Give it interior-mutable ownership: `GpuBackend { session: RefCell<Option<GpuTrainSession>> }` (single-threaded `fit()`; `Runtime` methods take `&self`). `begin_device_training` fills it; `grow_tree_on_device` borrows it; `end_device_training` clears it. No `cb-train` types are involved — the session is 100% `cb-backend`-internal.

### Cross-iteration data flow (per boosting iteration, target state)

```
approx_h (resident) ──► der seam (RmseGradient) ──► der1_h (resident, device)   [NO read-back]
der1_h + compressed_index_h + bins_h ──► per-level loop:
    hist2(partition-aware) → scan_update → score → BestSplit  [O(1) read-back only]
    → partition_split (bins_h in place) → partition_update (part_stats_h)
part_stats_h ──► read back 2^depth leaf stats ──► calc_average ──► leaf_values   [O(1)]
leaf_values + bins_h ──► update approx_h in place on device                      [NO read-back]
```

Per iteration, the ONLY host crossings are: the per-level `BestSplit` descriptor and the final `2^depth` leaf stats — the existing D-05 "host-light" contract, now sustained across the whole run instead of re-uploading the dataset each tree.

---

## Composing With Existing Phase 7 Kernels (sub-question 3)

Everything needed is already built and rocm-validated. The integration is WIRING + a residency refactor, not new kernels.

| Existing asset | File:loc | Role in device-resident loop | Change needed |
|----------------|----------|------------------------------|---------------|
| `launch_pointwise_hist2[_into]` | `mod.rs:481/501` | per-level histogram fill | add partition-aware (`fullPass=false`) variant for depth>1 (Phase 11); add handle args |
| `launch_scan_update_pointwise[_into]` | `mod.rs:1172/1190` | prefix-sum left/right fold | handle args |
| `launch_find_optimal_split_pointwise[_into]` | `mod.rs:903/927` | score + deterministic argmin → `BestSplit` | handle args (stop re-uploading der1/cindex per level) |
| `launch_partition_split_into` | `mod.rs:1392` | in-place forward-bit doc routing | already handle-based |
| `launch_partition_update_into` | `mod.rs:1464` | per-leaf Σder/Σweight reduce | already handle-based |
| der seam (`launch_der_binary_into`, …) | `der_seams.rs` | recompute der1 from approx on device | reuse verbatim |
| `grow_oblivious_tree_into` | `mod.rs:1641` | single-tree host-light driver | residency refactor + depth>1 |
| `grow_boosting_pass_into` | `mod.rs:1920` | multi-tree driver (already loops the above) | the basis of `grow_tree_on_device`; lift dataset upload out of the per-tree call |
| pairwise family | `pairwise.rs` (hist/scan/score/`grow_oblivious_tree_pairwise`) | pairwise split scoring on device | wire in Phase 12 |
| score calcers (L2/Cosine/Solar/LOO/Sat) | comptime in score launch | per-tree score function | already comptime-selected |

**Key refactor (the residency seam, the IN-02 "one geometry" precedent):** convert each `*_into` launcher to accept `Handle`s instead of `&[f64]`/`&[u32]`. The `_into` functions are the right layer — they already take `&client`; they just need to stop calling `client.create(...)` on caller data every invocation and instead receive the already-resident handle. The public host-slice wrappers (`launch_pointwise_hist2`, etc.) stay as thin "upload once then call `_into`" shims for the existing single-shot tests.

### Upstream reference grounding (`catboost-master/catboost/cuda/`)

The vendored CUDA trainer validates this exact decomposition:

- **`gpu_data/compressed_index.{h,cpp}` + `compressed_index_builder`** — the quantized feature matrix is compiled to a device-resident "compressed index" ONCE; the boosting loop never re-uploads it → our `compressed_index_h`.
- **`methods/doc_parallel_boosting.h`** — `TBoostingCursors::Cursors` (`TStripeBuffer<float>`) is the device-resident running approx, updated in place each iteration; `DataSets` (the `TDocParallelDataSetsHolder`) persists across the whole fit → our `approx_h` + persistent dataset handles.
- **`methods/pointwise_optimization_subsets.h`** — `TOptimizationSubsets { Partitions, PartitionStats, Bins }` is the device-resident partition state; `UpdateSubsetsStats` / `UpdateBins` mutate it per level → our `bins_h` + `part_stats_h` and the `partition_split`/`partition_update` launches.
- **`methods/oblivious_tree_doc_parallel_structure_searcher.{h,cpp}`** — `ReadAndEstimateLeaves(parts)` reads back ONLY the `TPartitionStatistics` (the small per-leaf stats) and estimates leaves on host → our O(1) `part_stats` read-back + `calc_average`. Upstream proof that "host-light, read back only leaf stats" is the real CatBoost GPU design, not a shortcut.
- **`methods/doc_parallel_pointwise_oblivious_tree.h`** — wraps the structure searcher as the weak learner the boosting loop calls per iteration → our coarse `grow_tree_on_device` seam.

Upstream is `TBoosting<TTarget, TWeakLearner>` templated over the weak learner — a COARSE "grow one tree" seam, NOT fine-grained histogram callbacks. This independently confirms the coarse-seam recommendation.

---

## New vs Modified Components

**NEW**
- `cb-backend`: `GpuTrainSession` (persistent device-handle owner).
- `cb-compute`: `DeviceTrainCtx` + `DeviceGrownTree` plain host structs (seam DTOs).
- `cb-backend`: partition-aware (`fullPass=false`) histogram launch for depth>1 (Phase 11).
- Benchmark harness crate / example for the CUDA Kaggle run (Phase 13).

**MODIFIED**
- `cb-compute/src/runtime.rs`: add 3 default-impl trait methods (`begin_device_training`, `grow_tree_on_device`, `end_device_training`).
- `cb-train/src/boosting.rs` (`train_inner`): try device seam → host fallback; call begin/end around the loop.
- `cb-backend/src/gpu_backend.rs`: implement the seam, own the session.
- `cb-backend/src/gpu_runtime/mod.rs`: `*_into` launchers take handles; `grow_oblivious_tree_into` depth>1; `grow_boosting_pass_into` becomes the session-driven driver.

**UNCHANGED (KEEP)**
- All `#[cube]` kernels (histogram/scan/score/partition/der) — wire, don't rewrite.
- Host growers in `cb-train/src/tree.rs` — remain the CPU path and the device fallback.
- `SelectedRuntime` compile-time selection; no runtime dispatch added.
- The `compute_gradients` seam and every shipped CPU/N-dim oracle (D-04 no-regression).

---

## Anti-Patterns

### Anti-Pattern 1: Fine-grained per-stage trait methods
**What people do:** expose `build_histograms()`, `score_splits()`, `partition()` on `Runtime` so the generic loop "orchestrates" the GPU.
**Why it's wrong:** forces device handles (or per-stage read-backs) across the `cb-compute`/`cb-train`↔`cb-backend` boundary — either leaking CubeCL into `cb-compute` (violates D-03, re-opens the feature-unification landmine) or destroying residency. Upstream itself uses a coarse weak-learner template, not stage callbacks.
**Do this instead:** one coarse `grow_tree_on_device` returning a host descriptor; keep the per-level chaining inside `cb-backend`.

### Anti-Pattern 2: cb-backend depending on cb-train
**What people do:** import host grower/leaf helpers from `cb-train` into the device driver to avoid duplication.
**Why it's wrong:** the documented Phase-7.5 landmine — Cargo feature unification across the `cb-train`→`cb-backend` edge breaks the rocm runtime build.
**Do this instead:** transcribe the needed CPU reference (e.g. `calc_average`, `select_best_candidate` tie-break) inline in `cb-backend`, as already done in `grow_oblivious_tree_into`. The seam DTOs live in `cb-compute` (the shared dependency both already use), never in `cb-train`.

### Anti-Pattern 3: Re-uploading the dataset each tree/level
**What people do:** pass host `&[u32]` cindex into the per-level/per-tree launches (the current `*_into` shape).
**Why it's wrong:** the quantized feature matrix is immutable for the whole run; re-uploading it per level/tree dominates device traffic and partly explains why the GPU path can be slower than CPU.
**Do this instead:** upload once into `GpuTrainSession`; thread the `Handle` through handle-based `*_into` variants.

### Anti-Pattern 4: Reading the full histogram/partition to host
**What people do:** read histograms back to host for "easier" split selection.
**Why it's wrong:** that is the FORBIDDEN D-05 host hybrid; it reintroduces the read-back stall.
**Do this instead:** keep histogram/partition device-resident; read back only the O(1) `BestSplit` descriptor and the `2^depth` leaf stats (the upstream `ReadAndEstimateLeaves` contract).

### Anti-Pattern 5: Silent fallback / wrong-structure stump
**What people do:** when depth>1 or a loss has no GPU path, quietly grow a stump or fall through to wrong math.
**Why it's wrong:** fabricates a tree that fails the ≤1e-5 oracle invisibly (the existing driver already rejects depth>1 with a typed error for this reason — `mod.rs:1670-1678`).
**Do this instead:** `grow_tree_on_device` returns `Ok(None)` → boosting loop uses the host grower (correct, just not yet accelerated). Explicit, oracle-safe partial coverage.

---

## Integration Points

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `cb-train` ↔ `cb-compute` `Runtime` | host structs only (`Derivatives`, new `DeviceGrownTree`) | NO CubeCL types — the rule that keeps the loop generic |
| `cb-compute` ↔ `cb-backend` | `cb-backend impl Runtime`; converts internal `GrownTree`→`DeviceGrownTree` | the only place device→host materialization happens |
| `cb-backend` internal: session ↔ launchers | `Handle` threaded with one `&client` | handle bound to its client (Phase 7.2/7.5 landmine) |
| `cb-backend` ↔ `cb-train` | **NONE — forbidden** | feature-unification landmine; transcribe CPU refs inline |

### External / hardware

| Surface | Pattern | Notes |
|---------|---------|-------|
| ROCm gfx1100 (in-env) | correctness dev + ≤1e-5 (ε=1e-4 per D-04) validation | CubeCL portable; all kernels already rocm-tested |
| CUDA (Kaggle notebook) | head-to-head SPEED benchmark vs official CatBoost GPU | same `SelectedRuntime` source, `--features cuda`; no NVIDIA in-env |
| wgpu | f32 channel fallback path | der/score launch over f32 on wgpu (existing) |

---

## Suggested Phase Build Order

| Phase | Scope | Depends on | Exit criterion |
|-------|-------|-----------|----------------|
| **10 — Seam + wire depth-1** | Add the coarse `Runtime` seam (default no-op); `GpuTrainSession` residency (upload-once dataset, in-place approx/der); handle-ify `*_into`; wire `grow_boosting_pass` behind the seam for RMSE/Logloss depth-1. Host fallback for everything else. | existing kernels + drivers | depth-1 RMSE training runs fully on-device, dataset uploaded once, ≤1e-5 vs CPU on rocm; CPU/host paths byte-unchanged (D-04). |
| **11 — depth>1 + Newton** | Partition-aware (`fullPass=false`) histogram fill so levels >0 score over `2^level` partitions; Newton der2 leaf values on device. | Phase 10 residency | depth>1 oblivious trees ≤1e-5; the depth>1 `OutOfRange` guard removed. |
| **12 — CTR / pairwise / ordered / multiclass** | Wire the existing pairwise device family; on-device CTR columns; ordered-boosting segments; multiclass der/leaves. Each lands behind the same `Ok(None)`→fallback gate, flipped on as it passes oracle. | Phase 11 | each feature family ≤1e-5 on device or cleanly falls back. |
| **13 — Benchmark + ε sign-off** | CUDA Kaggle head-to-head vs official CatBoost GPU; throughput report; final tolerance sign-off; close the >20× gap. | Phases 10–12 | documented speedup; ε signed off (D-04). |

**Ordering rationale:** Phase 10 must establish residency + the seam first because it is the prerequisite for ANY speedup and de-risks the landmine boundary; depth>1 (11) is the single biggest correctness extension and needs the resident partition state from 10; the feature families (12) are independent and individually gated by the fallback, so they can land in any sub-order without breaking correctness; the benchmark (13) is meaningful only once the common path is on-device.

---

## Sources

- `catboost-rs` existing code (HIGH — direct read): `crates/cb-compute/src/runtime.rs:892-955`; `crates/cb-train/src/boosting.rs:1870,2101,2920-3008,3230-3345`; `crates/cb-backend/src/gpu_backend.rs`; `crates/cb-backend/src/gpu_runtime/mod.rs:481-2043` (`grow_oblivious_tree_into`, `grow_boosting_pass_into`, `launch_*_into`); `crates/cb-backend/src/gpu_runtime/pairwise.rs`, `der_seams.rs`; `crates/cb-backend/Cargo.toml`, `lib.rs`.
- Vendored upstream CUDA trainer (HIGH — reference design): `catboost-master/catboost/cuda/gpu_data/compressed_index*.{h,cpp}`; `methods/doc_parallel_boosting.h` (`TBoostingCursors`, `TDocParallelDataSetsHolder`); `methods/pointwise_optimization_subsets.h` (`TOptimizationSubsets`/`Bins`/`PartitionStats`); `methods/oblivious_tree_doc_parallel_structure_searcher.{h,cpp}` (`ReadAndEstimateLeaves`); `methods/doc_parallel_pointwise_oblivious_tree.h`.
- `.planning/notes/gpu-training-host-light-root-cause.md` (HIGH — the integration-gap analysis + landmine).
- `.planning/PROJECT.md` v1.1 milestone scope (HIGH).

---
*Architecture research for: device-resident GPU boosting integration with a generic Runtime seam (v1.1)*
*Researched: 2026-06-28*
