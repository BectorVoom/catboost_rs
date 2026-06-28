# Feature Research

**Domain:** Device-resident GPU gradient-boosting training (CatBoost CUDA parity) for catboost-rs v1.1
**Researched:** 2026-06-28
**Confidence:** HIGH (grounded in vendored `catboost-master/catboost/cuda/` + read of the existing Rust kernels/grow loop)

## Scope

This milestone moves the **tree-growth inner loop** onto the GPU. v1.0 already ships a derivatives-only GPU MVP: `der1`/`der2` run on device (Phase 7.2 der seam, `gpu_runtime/der_seams.rs`) but histogram → score → split → partition → leaf-value all run on the host CPU (`cb-train/src/tree.rs`), so a GPU run is **slower than pure CPU** (>20× vs official CatBoost GPU). Phase 7.5 built a device-resident `grow_boosting_pass` (`cb-backend/src/gpu_runtime/mod.rs:1890`) that is **depth-1 only and never wired** into `cb_train::train`. The features below are the on-device pipeline stages needed to close that gap.

**Upstream reference structure** (the device-resident pipeline this mirrors):
- Storage/docBins — `gpu_data/compressed_index.h` (`TSharedCompressedIndex` / `TCompressedDataSet`), `gpu_data/gpu_structures.h` (`TCFeature`, `TDataPartition`, `TPartitionStatistics`).
- The inner loop — `methods/oblivious_tree_doc_parallel_structure_searcher.cpp::FitImpl` (the canonical per-depth `histogram → AllReduce partStats → ComputeOptimalSplit → ReadOptimalSplit → Split` loop, lines 62-160).
- Histograms — `methods/histograms_helper.h` (`TComputeHistogramsHelper::Compute`, the `BuildFromScratch`/subtraction trick).
- Score/split — `methods/pointwise_scores_calcer.h` (`SubmitCompute`/`ComputeOptimalSplit`/`ReadOptimalSplit`).
- Partition — `methods/pointwise_optimization_subsets.h` (`TSubsetsHelper::Split` = `UpdateBins` + `ReorderBins` + `UpdatePartitionStats`).
- Leaf values — `methods/leaves_estimation/` (`EstimateLeaves` Simple = `Sum/(Weight+L2Reg)`; `pointwise_oracle.h` + `descent_helpers.h::TNewtonLikeWalker` for Newton).
- Boosting loop / residency — `methods/doc_parallel_boosting.h` (`TDocParallelDataSetsHolder` keeps the compressed index + cursors device-resident for the whole run).

## Feature Landscape

### Table Stakes (Required for Speed Parity)

Without **all** of these device-resident, a GPU run cannot beat CPU — the host work and per-tree crossings dominate. These are the non-negotiable v1.1 deliverables.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **`Runtime` grow-tree trait seam** | The boosting loop (`cb_train::train::<R: Runtime>`) is generic over `Runtime` but the trait (`cb-compute/src/runtime.rs:897`) only exposes `compute_gradients`; there is no seam to "grow a tree on device", so `fit()` always falls through to the CPU growers. This is *the* integration gap. | **S** | Add `grow_tree`/`grow_boosting_pass` to the `cb-compute` `Runtime` trait. CPU backend delegates to existing `tree.rs` growers; GPU backend calls `grow_boosting_pass`. **Landmine:** `cb-backend` must NOT depend on `cb-train` (feature unification breaks rocm runtime) — the trait + model types live in `cb-compute`/`cb-core`, transcribe CPU refs inline. |
| **Wire depth-1 device grow loop** | `grow_boosting_pass` already does device histogram+score+split+partition+leaf for depth-1 RMSE/L2 and is correctness-validated on rocm — it is only called from tests. Wiring it is the first measurable win. | **S** | Pure integration; no new kernels. Reuses scatter/scan/reduce/pointwise_hist/score_split/partition kernels. Gated to its MVP envelope (depth=1, Plain, foldCount=1, RMSE) with a typed fallback to CPU otherwise. |
| **Device-resident quantized storage (docBins / compressed index)** | Upstream `TSharedCompressedIndex` holds a feature-major quantized `cindex` device-resident for the **entire** training run (`doc_parallel_dataset.h`). Re-uploading per tree negates the speedup. | **M** | Rust already uses the feature-major `cindex[feature*n+obj]` layout and uploads it once *per `grow_oblivious_tree_into` call* (mod.rs:1706) — but that call is per-tree, so it re-uploads every iteration. Lift the upload above the iteration loop; keep the handle resident across all trees. Upstream `TCFeature{Offset,Mask,Shift,FirstFoldIndex,Folds}` packs multiple features per `ui32` word — the Rust layout is one feature per word (simpler, more memory; acceptable). |
| **Partition-aware histograms (`fullPass=false` / depth>1)** | The current pointwise/pairwise fill is **whole-dataset, single-partition** (`partCount=1`). A depth-L level must build histograms over `2^L` partitions. This is the explicit blocker that makes `grow_oblivious_tree_into` reject `depth>1` (mod.rs:1670). | **L** | Extend Phase 7.3 `pointwise_hist` (2-channel) and the partition kernels so the fill is keyed by `leaf_of[obj]` into `2^level` partition slots. Upstream `ComputeHistogram2` (histograms_helper.h) accumulates `(1<<CurrentBit)*FoldCount*features*2` floats. **The single largest feature in the milestone.** |
| **Histogram subtraction trick** | Upstream computes the histogram for only the **smaller** child and derives the sibling by subtraction from the parent (`histograms_helper.h` `BuildFromScratch`/`CurrentBit` logic). Halves histogram work at every level — without it depth>1 is ~2× slower than parity. | **M** | Requires keeping the parent-level histogram resident and a per-leaf subtract pass. Depends on partition-aware histograms landing first. |
| **Device BestSplit selection (already built, must stay O(1)-crossing)** | Upstream scores all `(feature,bin)` candidates on device and reduces to one `TBestSplitProperties` per device, reading back only that descriptor (`pointwise_scores_calcer.h` `ComputeOptimalSplit`/`ReadOptimalSplit`; FitImpl `TakeBest`). | **S** (exists) | Rust `score_split.rs` (7.5) already does L2/Cosine/Solar/LOO/Sat scoring + deterministic argmin returning an O(blocks) `BestSplit[]`. Keep the contract: only the O(1) split + the `2^depth` part-stats cross host↔device per level (the D-05 crossing class). |
| **Device partition / leaf-assignment update** | After picking a split, upstream routes docs into child partitions and reorders indices so each partition is contiguous (`TSubsetsHelper::Split` = `UpdateBins`+`ReorderBins`+`UpdatePartitionStats`). Contiguity is what makes the next level's partition histograms cheap. | **M** | Rust has `launch_partition_split_into` (forward-bit doc-routing) + `launch_partition_update_into` for depth-1. Depth>1 needs the **reorder** so partitions are contiguous (`TDataPartition{Offset,Size}`), not just a per-object leaf tag. Co-dependent with partition-aware histograms. |
| **Gradient leaf values from part-stats (Simple/Gradient method)** | `EstimateLeaves` (structure_searcher.cpp:175) = `Sum/(Weight+L2Reg)` per leaf — the default `ELeavesEstimation::Simple`/Gradient path, computed from the `2^depth` part-stats. | **S** (exists) | Rust already reads back `2^depth` part-stats and computes `Σder1/(Σweight+scaled_l2)` on host (mod.rs:1769). Matches `cb_compute::calc_average`. Keep as-is. |
| **Keep gradients/approx device-resident across iterations** | Upstream keeps the approx **cursor** and recomputes gradients on device each iteration; only metric scalars cross (`doc_parallel_boosting.h`). | **M** | `grow_boosting_pass` already recomputes der1 device-side via the 7.2 seam but reads it back **per tree** (mod.rs:2018) and re-uploads approx. Keep approx + der as resident handles updated in place across iterations; cross only when the host needs a metric / early-stopping check. |

### Differentiators (Higher Parity / Broader Coverage)

These extend the device path beyond the simplest case. Each is independently shippable after the table-stakes depth>1 path lands. Order them by how many real workloads they unblock.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Newton der2 leaf estimation on device** | The default for Logloss/multiclass is `ELeavesEstimation::Newton`, not Simple — a separate refit (`TObliviousTreeLeavesEstimator` + `TNewtonLikeWalker`, `leaves_estimation/`) running several der/der2 passes per tree over the fixed structure. Without it, classification leaf values diverge. | **L** | Needs a device der2 accumulation per leaf + the backtracking-line-search descent (`descent_helpers.h`). Reuses the 7.2 der2 handles (`launch_der_unary_handle` LoglossHessian). Structure search still uses gradient/Newton-at-zero for the *weak target*; leaf refit is the second stage. |
| **Second-order weak-target / score functions (Cosine default)** | `IsSecondOrderScoreFunction(ScoreFunction)` selects `NewtonAtZero` vs `GradientAtZero` for the weak target (structure_searcher.cpp `ComputeWeakTarget`). GPU's **default ScoreFunction is Cosine** (second-order), not L2 — a known parity gap (memory `cb-train-uses-l2-but-catboost-defaults-cosine`). | **M** | `score_split.rs` already has the Cosine/Solar/LOO/Sat arms as comptime kernels; wire the weak-target der2 channel into the device histogram so Cosine scores match. Flag: the depth-1 `grow_boosting_pass` MVP scores L2 only. |
| **Bootstrap / sampling on device** | Upstream runs `TBootstrap<TStripeMapping>` (Bayesian/Bernoulli/MVS/Poisson) on device each tree (`gpu_data/bootstrap.h`, structure_searcher `ComputeWeakTarget`). Sampling on host would force a per-tree weight crossing. | **M** | Reuses the resident der/weight handles; a per-object multiplier kernel. Needed for any non-default `bootstrap_type`/`subsample`. |
| **Random-strength score noise on device** | `ComputeScoreStdDev` + `random.NextUniformL()` add per-candidate noise to scores (structure_searcher; `random_score_helper.h`). Affects which split wins — must match for parity with `random_strength>0`. | **S** | The score kernel already takes a `scoreStdDevMult`; thread the per-tree RNG seed (the existing per-tree seeder from Phase 6.3 StochasticRank is the precedent). |
| **CTR / permutation-dependent features on device** | Upstream runs a **second** score calcer over permutation-dependent (CTR) features (`simpleCtrScoreCalcer` in FitImpl; `gpu_data/batch_binarized_ctr_calcer.h`, `ctrs/`). Categorical workloads are CatBoost's headline use case. | **L** | Large. CTR values are computed on device per permutation and fed as extra feature columns into the same histogram/score path. Defer until the numeric depth>1 path is solid. |
| **Pairwise / ranking device path** | Phase 7.4 already built the 4-channel **weight-only** pairwise histogram (`pairwise_hist.rs`) + a pairwise scorer; upstream `methods/pairwise_oblivious_trees/` + `pairwise_kernels.h` complete the loop. Unblocks YetiRankPairwise/PairLogitPairwise on GPU. | **L** | Reuses 7.4 kernels but needs the pairwise partition + leaves oracle (`leaves_estimation/pairwise_oracle.h`). Independent track from the pointwise depth>1 work. |
| **Multiclass device path** | `targets/multiclass_targets.cpp` + `multiclass_kernels.h`; the approx becomes dimension-major (`approx_dimension > 1`). The der seam is already N-dim-aware (Phase 6.2, `compute_gradients` dimension-major). | **L** | Histogram/score/leaf all gain a dimension axis. Newton leaf estimation here is block-diagonal (`HessianBlockSize`). Defer to last. |
| **Ordered boosting on device** | Plain boosting = `doc_parallel_boosting.h` (one cursor); **Ordered** = `dynamic_boosting.h` with multiple permutation cursors. CatBoost's anti-overfitting signature. | **L** | Multiple device-resident approx cursors + per-permutation gradient recompute. Heaviest residency cost. Defer; Plain covers most speed-benchmark scenarios. |

### Anti-Features (Avoid)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| **Per-tree re-upload of the compressed index** | It is what `grow_oblivious_tree_into` does today (mod.rs:1706) and is the path of least resistance when wiring. | Re-uploading `n_features*n` u32 every iteration is a host↔device bulk crossing per tree — it negates the entire speedup and can leave GPU slower than CPU. | Upload once above the iteration loop; keep the handle resident (upstream `TDocParallelDataSetsHolder`). |
| **Reading histograms / partitions / per-doc leaf tags back to host** | Easy to debug; lets you reuse the host scorer. | Bulk per-doc/per-bin crossings every level are the dominant cost. Upstream crosses only the O(1) `TBestSplitProperties` + the `2^depth` `TPartitionStatistics`. | Enforce the D-05 crossing contract: only the O(1) split descriptor + `2^depth` part-stats leave the device per level. |
| **Chasing bit-exact f64 ≤1e-5 on GPU** | The CPU parity bar is ≤1e-5; reflex is to hold GPU to it. | Upstream GPU uses `float` (f32) histograms/scores/cursors throughout (`TStripeBuffer<float>`). Demanding f64 bit-exactness fights the hardware and the reference itself; gfx1100 has no f64 atomic-add (Phase 7.6 used a host-sum fallback). | Sign off **ε=1e-4 vs the Rust CPU path** as Phase 7.6 already did (memory `phase76-gpu-tolerance-signoff-outcome`); validate correctness on rocm in-env, benchmark **speed** on CUDA/Kaggle. |
| **Feature-parallel layout** | Upstream ships both feature-parallel (`feature_layout_feature_parallel.h`) and doc-parallel layouts. | Feature-parallel is the legacy single-GPU mode; doc-parallel (`TDocParallelLayout`) is the modern default and matches the Rust per-object layout. Building both doubles surface area for no benefit on single-GPU. | Implement **doc-parallel only** (`TDocParallelObliviousTree`). |
| **Adding a `cb-train` dependency to `cb-backend`** | The CPU growers in `tree.rs` are the obvious reference to call directly. | Cargo feature unification across `cb-train` + `cb-backend` breaks the rocm runtime build (memory `phase75-grow-loop-outcome`). | Put the grow-tree trait + model types in `cb-compute`/`cb-core`; **transcribe** the needed CPU reference logic inline into `cb-backend`. |
| **Multi-GPU / `TStripeMapping` distribution** | Upstream's whole GPU stack is built around `TStripeMapping` (multi-device). | Out of scope (PROJECT.md: desktop/server single-GPU); the distributed `AllReduce`/`ReduceScatter` plumbing is enormous and untestable in-env. | Single-device: the `AllReduce` in FitImpl collapses to identity. |

## Feature Dependencies

```
[Runtime grow-tree trait seam]                 (S, foundation)
    └──enables──> [Wire depth-1 device grow loop]   (S, first win)
                       └──requires──> [Device-resident compressed index across iters]  (M)
                       └──requires──> [Keep gradients/approx device-resident]          (M)

[Partition-aware histograms (fullPass=false)]   (L, the keystone)
    ├──requires──> [Device partition update + reorder (contiguous partitions)]  (M)
    ├──enables───> [depth>1 trees]
    └──enables───> [Histogram subtraction trick]    (M)

[depth>1 path]
    └──enables──> [Newton der2 leaf estimation]      (L)
    └──enables──> [Second-order score fns / Cosine]  (M)
    └──enables──> [Bootstrap on device]              (M)  ──> [Random-strength noise] (S)
    └──enables──> [CTR on device]                    (L)
    └──enables──> [Multiclass / Ordered]             (L)

[Pairwise 7.4 kernels] ──independent track──> [Pairwise/ranking device path]  (L)
```

### Dependency Notes

- **Everything requires the trait seam first:** until `Runtime` exposes a grow-tree method, no device grow loop is reachable from `fit()` — this is the documented "design error" in the root-cause note.
- **Partition-aware histograms are the keystone:** depth>1, the subtraction trick, and (transitively) Newton/CTR/multiclass all sit behind the `fullPass=false` per-partition fill. It is the single highest-leverage piece of new kernel work.
- **Partition update and partition histograms are co-dependent:** the histogram fill needs partitions to be *contiguous* (the `ReorderBins` step), so the depth>1 partition update must reorder indices, not just tag leaves.
- **Pairwise is an independent track:** it reuses the 7.4 pairwise histogram but has its own scorer/partition/leaves-oracle; it does not depend on the pointwise depth>1 work and can proceed in parallel.

## MVP Definition

### Launch With (v1.1 core — the "GPU is actually faster" milestone)

- [ ] `Runtime` grow-tree trait seam in `cb-compute` — without it nothing is reachable.
- [ ] Wire depth-1 `grow_boosting_pass` into `cb_train::train` with a typed CPU fallback outside its envelope — first measurable speed win, proves the seam.
- [ ] Compressed index uploaded once, resident across all iterations — removes the per-tree bulk re-upload anti-feature.
- [ ] Gradients/approx resident across iterations — removes the per-tree der read-back.
- [ ] Partition-aware histograms (`fullPass=false`) + contiguous partition update → **depth>1 trees** — the real workloads need depth 6.
- [ ] Histogram subtraction trick — needed to actually reach parity speed at depth>1.
- [ ] ε=1e-4-vs-CPU correctness sign-off on rocm in-env + a speed benchmark harness vs official CatBoost GPU on a CUDA/Kaggle notebook.

### Add After Validation (v1.x)

- [ ] Newton der2 leaf estimation — trigger: classification benchmarks needed (Logloss default is Newton, not Simple).
- [ ] Second-order/Cosine score function on device — trigger: matching CatBoost's *default* GPU split scores (closes the known L2-vs-Cosine parity gap).
- [ ] Bootstrap + random-strength on device — trigger: non-default `subsample`/`bootstrap_type`/`random_strength`.
- [ ] CTR / permutation-dependent features on device — trigger: categorical workloads (CatBoost's headline use case).

### Future Consideration (v2+)

- [ ] Pairwise/ranking device path — defer unless ranking speed is a stated benchmark target (independent track, large).
- [ ] Multiclass device path — defer; N-dim histogram/leaf expansion.
- [ ] Ordered-boosting device path — defer; heaviest residency cost (multiple permutation cursors).

## Feature Prioritization Matrix

| Feature | User Value (speed) | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Runtime grow-tree trait seam | HIGH | LOW | P1 |
| Wire depth-1 grow loop | MEDIUM | LOW | P1 |
| Compressed index resident across iters | HIGH | MEDIUM | P1 |
| Gradients/approx resident across iters | HIGH | MEDIUM | P1 |
| Partition-aware histograms (depth>1) | HIGH | HIGH | P1 |
| Partition update + reorder (depth>1) | HIGH | MEDIUM | P1 |
| Histogram subtraction trick | HIGH | MEDIUM | P1 |
| Gradient leaf values (Simple) | HIGH | LOW (exists) | P1 |
| Newton der2 leaf estimation | HIGH | HIGH | P2 |
| Second-order / Cosine score fn | MEDIUM | MEDIUM | P2 |
| Bootstrap on device | MEDIUM | MEDIUM | P2 |
| Random-strength score noise | LOW | LOW | P2 |
| CTR / perm-dependent features | HIGH | HIGH | P2 |
| Pairwise / ranking device path | MEDIUM | HIGH | P3 |
| Multiclass device path | MEDIUM | HIGH | P3 |
| Ordered-boosting device path | MEDIUM | HIGH | P3 |

**Priority key:** P1 = required for the "GPU beats CPU" v1.1; P2 = add for broader parity once depth>1 is solid; P3 = defer.

## Suggested Incremental Build Order

1. **Trait seam** (`cb-compute` `Runtime::grow_tree`) — CPU backend delegates to existing growers; GPU backend stubs to `grow_boosting_pass`. *(S)*
2. **Wire depth-1** + typed CPU fallback; prove end-to-end on rocm and on the benchmark harness. *(S)*
3. **Residency**: hoist the compressed-index upload above the iteration loop; keep approx/der resident across iterations. *(M)* — at this point depth-1 GPU should beat CPU.
4. **Partition update + reorder → contiguous partitions** *(M)* and **partition-aware histograms (`fullPass=false`)** *(L)* together — unlocks **depth>1**.
5. **Histogram subtraction trick** *(M)* — reach parity speed at depth 6.
6. **Newton der2 leaf estimation** *(L)* + **second-order/Cosine score** *(M)* — classification + default-score parity.
7. **Bootstrap + random-strength** *(M/S)* — sampling parity.
8. **CTR on device** *(L)* — categorical workloads.
9. **Pairwise** *(L, parallel track)*, then **multiclass** *(L)*, then **ordered boosting** *(L)*.

## GPU-Only / Parity-Subtle Behaviors (flag for the planner)

- **Score function default differs:** GPU default `ScoreFunction` is **Cosine** (second-order), CPU/host reference paths and the depth-1 MVP use **L2**. Known gap (memory `cb-train-uses-l2-but-catboost-defaults-cosine`); the weak target uses `NewtonAtZero` vs `GradientAtZero` accordingly.
- **Two-stage leaf values:** structure search scores on gradient/Newton-**at-zero**; leaf values are a **separate refit** (`Simple`=mean, `Newton`=line-search descent). Don't conflate the split-scoring der with the leaf-value der.
- **f32 on device:** upstream histograms/scores/cursors are `float`. Hold GPU to **ε=1e-4 vs the Rust CPU path** (Phase 7.6 precedent), not the CPU's ≤1e-5 bar. gfx1100 has no f64 atomic-add → host-sum fallback for parity-critical reductions.
- **Correctness vs speed venues:** correctness is validated **in-env on AMD/ROCm** (CubeCL kernels are portable from one source); the head-to-head **speed** number must come from a **CUDA/Kaggle** run (no NVIDIA in-env).
- **Crossing contract (D-05):** per level, only the O(1) `TBestSplitProperties` + the `2^depth` `TPartitionStatistics` may cross host↔device. Any bulk histogram/partition/per-doc read-back is a regression.
- **Landmine:** never add a `cb-train` dependency to `cb-backend` (feature unification breaks the rocm runtime) — keep the grow-tree trait/model types in `cb-compute`/`cb-core` and transcribe CPU references inline.

## Sources

- `catboost-master/catboost/cuda/methods/oblivious_tree_doc_parallel_structure_searcher.cpp` / `.h` — the canonical device inner loop (HIGH)
- `catboost-master/catboost/cuda/methods/histograms_helper.h` — histogram compute + subtraction trick (HIGH)
- `catboost-master/catboost/cuda/methods/pointwise_optimization_subsets.h` — partition update (`Split`/`UpdateBins`/`ReorderBins`) (HIGH)
- `catboost-master/catboost/cuda/methods/pointwise_scores_calcer.h` — device BestSplit (HIGH)
- `catboost-master/catboost/cuda/methods/leaves_estimation/` (`pointwise_oracle.h`, `descent_helpers.h`) — Simple vs Newton leaf values (HIGH)
- `catboost-master/catboost/cuda/methods/doc_parallel_boosting.h` / `dynamic_boosting.h` — boosting loop, residency, Plain vs Ordered (HIGH)
- `catboost-master/catboost/cuda/gpu_data/{compressed_index.h, gpu_structures.h}` — docBins layout, `TCFeature`/`TDataPartition`/`TPartitionStatistics` (HIGH)
- `crates/cb-backend/src/gpu_runtime/mod.rs` (`grow_boosting_pass`, `grow_oblivious_tree_into`) — existing depth-1 device grow loop + MVP limits (HIGH)
- `crates/cb-backend/src/gpu_runtime/der_seams.rs` — Phase 7.2 device der1/der2 seam (HIGH)
- `crates/cb-backend/src/kernels/{pointwise_hist,pairwise_hist,score_split,scatter,scan,reduce}.rs` — existing Phase 7.3/7.4/7.5 kernels (HIGH)
- `crates/cb-compute/src/runtime.rs` — the `Runtime` trait (the seam gap) (HIGH)
- `.planning/notes/gpu-training-host-light-root-cause.md` — host-vs-device classification of the inner loop (HIGH)

---
*Feature research for: device-resident GPU gradient-boosting training (CatBoost CUDA parity)*
*Researched: 2026-06-28*
