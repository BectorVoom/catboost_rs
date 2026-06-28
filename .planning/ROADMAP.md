# Roadmap: catboost-rs

## Milestones

- ✅ **v1.0 Core Parity** — Phases 1–8 (shipped 2026-06-28)
- 🚧 **v1.1 GPU Performance** — Phases 10–13 (planning) — full CUDA device-resident training parity; ALL GPU kernel oracles (correctness + speed) validated on a Kaggle CUDA notebook, with a per-phase speed check from the first GPU phase to the last

## Phases

<details>
<summary>✅ v1.0 Core Parity (Phases 1–8) — SHIPPED 2026-06-28</summary>

- [x] Phase 1: Workspace, Lint Discipline & Oracle Harness
- [x] Phase 2: Data Layer — Pool, Quantization & Reduction
- [x] Phase 3: CPU Training Core — Plain Boosting & Oblivious Trees
- [x] Phase 4: Model Serialization, SHAP & Rust API (first full oracle lock)
- [x] Phase 5: Ordered Boosting, Ordered CTR & Categoricals
- [x] Phase 6: Full Loss & Feature Parity (6.1 regression · 6.2 multiclass/N-dim · 6.3 ranking · 6.4 score-fns/uncertainty/custom · 6.5 text/embedding · 6.6 advanced + non-symmetric)
- [x] Phase 7: GPU Backends via CubeCL — structural parity (7.1 primitives · 7.2 grad/hess · 7.3 pointwise hist · 7.4 pairwise hist · 7.5 on-device grow loop · 7.6 rocm tolerance sign-off)
- [x] Phase 8: Python Bindings, Dual API & Packaging

Full per-phase detail: `.planning/milestones/v1.0-ROADMAP.md` and `.planning/milestones/v1.0-REQUIREMENTS.md`.
61/62 v1 requirements complete; known gaps carried forward (see Backlog + `.planning/MILESTONES.md`).

</details>

### 🚧 v1.1 GPU Performance (Phases 10–13)

**Milestone goal:** Move the entire boosting inner loop (histogram build, split scoring, BestSplit, partition/leaf-assignment, leaf values) onto the GPU — not just derivatives — closing the >20× gap vs official CatBoost GPU while preserving correctness.

**Validation authority — ALL GPU (CUDA) kernel oracles, correctness AND speed, run on a Kaggle CUDA notebook.** CUDA is the single authoritative GPU oracle for this milestone. A reproducible Kaggle CUDA oracle/test harness (BENCH-01) is a **foundational deliverable established in Phase 10** that measures BOTH correctness AND wall-clock speed from the start, so every GPU kernel — from the depth-1 device tree onward — is both correctness-tested and speed-measured on CUDA, not merely speed-benchmarked at the end. There is no NVIDIA hardware in-env; the AMD/ROCm in-env GPU remains an **optional compile/smoke convenience** for fast local iteration, but it is **not a gate** — no requirement is satisfied by ROCm validation alone. The prior "validate correctness in-env on ROCm, benchmark speed on CUDA" asymmetry is GONE.

**Speed is checked for EVERY GPU kernel in EVERY phase — from the first to the last.** BENCH-02 is a **standing per-phase speed check** (mapped to Phase 10 where it is first established, but enforced in every phase 10→13 — analogous to how GPUT-14's ε=1e-4 gate is mapped to Phase 11 but enforced onward). Every phase that lands GPU kernels reports a Kaggle CUDA wall-clock speed measurement for that phase's kernels — device path vs the host-CPU baseline, and vs official CatBoost GPU where a comparable config exists (warm-run/JIT-excluded, train-only). No phase's GPU kernels are "done" without a recorded CUDA speed check. Phase 13 is the **comprehensive final** parity sign-off that AGGREGATES the per-phase checks — NOT the first place speed is measured.

**Practical note:** a Kaggle CUDA oracle/speed run is a **human-gated external step** — the user builds the `--features cuda` wheel and runs the notebook. GPU oracle + speed verification for every phase below is therefore a human-gated Kaggle CUDA execution, not an in-CI automated check.

**Parity bar:** the GPU device path is held to **ε=1e-4 vs the Rust CPU path** (Phase 7.6 precedent — device math is f32; bit-exact f64 ≤1e-5 is not the GPU goal), with the depth-1 device tree held tighter at **≤1e-5** where the whole-dataset level-0 histogram is the exact CPU score. The CPU path stays oracle-locked ≤10⁻⁵ and byte-unchanged (D-04 no-regression).

**Standing landmines (carry into every phase):**
- **Never add a `cb-train` dependency to `cb-backend`** — Cargo feature unification breaks the rocm runtime; transcribe CPU references inline. The `Runtime` seam stays CubeCL-free (plain host structs cross the boundary).
- **No `-inf` float literals inside `#[cube]` kernels** — HIP JIT rejects them on gfx1100 (CUDA accepts them, so this is a portability nicety the ROCm smoke build catches); use the `f32::MIN` sentinel so kernels stay portable cuda/rocm/wgpu.
- **Reduction determinism still governs the ε=1e-4 bar.** CUDA *does* provide f64 atomic-add (unlike gfx1100), so the atomic-free constraint is now a portability nicety rather than a hard correctness gate — BUT `atomicAdd` commit ordering is still non-deterministic and compounds over hundreds of trees, so a **deterministic reduction strategy is still required** to hold ε=1e-4 parity on CUDA. (gfx1100 still lacks f64 atomic-add, so atomic-free design also keeps the optional ROCm smoke path device-resident.)
- **Never read a `Handle` through a client other than the one that allocated it** (CubeCL residency rule).
- **The `Ok(None)`→host-CPU fallback gate** keeps every increment oracle-safe: any case not yet passing device sign-off on Kaggle CUDA falls back to the CPU path (the correctness reference and safety net).

#### Phase Checklist

- [ ] **Phase 10: Coarse Runtime Grow-Tree Seam + GpuTrainSession Residency + Wire Depth-1 + Kaggle CUDA Oracle & Speed Harness** - The device grow loop becomes reachable from `fit()`; training data stays device-resident; depth-1 trees grow on device; the foundational Kaggle CUDA harness lands measuring BOTH correctness (≤1e-5) AND wall-clock speed (depth-1 device fit vs CPU baseline) from the start.
- [ ] **Phase 11: Depth>1 Partition-Aware Histograms + Reduction Determinism + Newton Der2** - Real depth-6 RMSE + Logloss workloads grow fully on device within ε=1e-4, oracle-tested AND speed-measured on Kaggle CUDA (device vs CPU and vs official CatBoost GPU).
- [ ] **Phase 12: GPU Coverage Expansion (Sampling / CTR / Pairwise / Multiclass / Ordered)** - Each remaining training family transitions to the device path (ε=1e-4 on Kaggle CUDA) behind the `Ok(None)` fallback gate, each family timed on Kaggle CUDA as it lands.
- [ ] **Phase 13: Comprehensive Kaggle CUDA Speed Benchmark + Parity Sign-Off** - The device-resident path demonstrably closes the >20× gap via a comprehensive final sign-off that AGGREGATES the per-phase speed checks, with CUDA correctness gated before any speed number.

## Phase Details

### Phase 10: Coarse Runtime Grow-Tree Seam + GpuTrainSession Residency + Wire Depth-1 + Kaggle CUDA Oracle & Speed Harness
**Goal**: The existing-but-unused device grow loop becomes reachable from `cb_train::train`; training data stays device-resident across iterations; a depth-1 tree grows fully on device and matches the CPU path bit-for-bit (≤1e-5); and the reproducible **Kaggle CUDA harness** is established as a foundational deliverable that measures BOTH correctness AND wall-clock speed from day one — so the depth-1 device tree is correctness-tested AND speed-measured on CUDA from the start, not merely speed-benchmarked later. This phase establishes the seam, the residency architecture, the `Ok(None)` fallback pattern, the CUDA oracle of record, AND the standing per-phase speed check that every later phase reuses.
**Depends on**: Phase 7 (GPU structural parity — `grow_boosting_pass`, der seam, `*_into` launchers), Phase 8 (`GpuBackend` over `SelectedRuntime`)
**Requirements**: GPUT-01, GPUT-02, GPUT-03, GPUT-04, BENCH-01, BENCH-02 (standing — first established here, enforced in every phase 10→13)
**Success Criteria** (what must be TRUE):
  1. A depth-1 oblivious tree (RMSE/Logloss, Plain boosting, fold_count=1) grown fully on device matches the CPU path ≤1e-5, **oracle-tested on Kaggle CUDA** (a human-gated `--features cuda` notebook run) — not merely on the optional ROCm smoke build.
  2. A reproducible Kaggle CUDA harness (BENCH-01) builds the `--features cuda` wheel and on a Kaggle CUDA notebook runs BOTH the GPU kernel **correctness** oracle (≤1e-5 for the depth-1 tree, correctness as a blocking gate) AND a **wall-clock speed** measurement (warm-run/JIT-excluded, train-only); it is the authoritative GPU oracle + speed harness reused by Phases 11–13, documented as a human-gated external step (ROCm in-env is an optional compile/smoke convenience, not a gate).
  3. **Speed check (BENCH-02, standing):** the depth-1 device fit is timed on Kaggle CUDA and reported as device path vs the host-CPU baseline — the harness reports BOTH correctness ≤1e-5 AND wall-clock for this phase's kernels — establishing the per-phase speed-check discipline enforced from here to the last phase (vs official CatBoost GPU where a comparable depth-1 config exists).
  4. The quantized feature matrix uploads exactly once per `fit()` (no per-tree re-upload), gradients/approx stay device-resident across iterations, the per-tree `der1` host read-back is eliminated, and only the O(1) BestSplit descriptor + `2^depth` partition statistics cross host↔device per level (D-05).
  5. An uncovered case (e.g. depth>1) returns `Ok(None)` and falls back to the host CPU grower, producing the same prediction as a pure-CPU fit; the CPU/host training path is byte-unchanged (D-04 no-regression — the full existing CPU oracle suite stays green), and the reduction-determinism strategy is spiked (fixed-point i64 atomics vs private-histogram merge vs two-pass segmented reduce) with a recommendation documented to feed Phase 11's histogram kernel.
**Plans**: TBD
**UI hint**: no
**Notes**: The `Runtime` seam adds three default-impl methods (`begin_device_training`, `grow_tree_on_device → CbResult<Option<DeviceGrownTree>>`, `end_device_training`) with CubeCL-free host-typed signatures — mirrors the established `Derivatives` seam. `GpuTrainSession` (new, `cb-backend`-internal) owns one `ComputeClient` + all persistent handles, owned by `GpuBackend` via `RefCell<Option<…>>`. The Kaggle CUDA harness is the new foundational piece and measures correctness AND speed from the start (the `benchmark.py`/maturin `--features cuda` wheel pattern, run on a notebook the user executes — human-gated): verify the CUDA backend is active (`nvidia-smi`), warm one untimed fit, re-run the depth-1 oracle on CUDA before trusting any speed number. **BENCH-02 is a standing per-phase speed check** — mapped here because it is first established here, but enforced in EVERY phase 10→13 (analogous to GPUT-14's ε=1e-4 gate being mapped to Phase 11 yet enforced onward): no phase's GPU kernels are done without a recorded Kaggle CUDA speed measurement. Landmines: no `cb-train` dep in `cb-backend`; `f32::MIN` sentinel (no `-inf` in `#[cube]`, for ROCm-smoke portability); deterministic reduction strategy still required even though CUDA has f64 atomic-add. Architecture fully specified — mirrors Phase 7-8 patterns.

### Phase 11: Depth>1 Partition-Aware Histograms + Reduction Determinism + Newton Der2
**Goal**: Real-world depth-6 workloads — both RMSE regression and Logloss classification — grow fully on device within ε=1e-4, oracle-tested AND speed-measured on Kaggle CUDA. This is the performance keystone: depth 6 is the default, the histogram subtraction trick is required to approach parity speed, and Newton der2 is required for classification. GPUT-14 (ε=1e-4 device bar on Kaggle CUDA + D-04 CPU no-regression) becomes the operative standing gate from this phase forward — depth-1 was held to ≤1e-5, but reductions over hundreds of depth>1 trees move the bar to ε=1e-4.
**Depends on**: Phase 10 (the seam, residency, AND the Kaggle CUDA oracle + speed harness)
**Requirements**: GPUT-05, GPUT-06, GPUT-07, GPUT-08, GPUT-14
**Success Criteria** (what must be TRUE):
  1. A depth>1 (e.g. depth 6) RMSE tree grows on device via partition-aware histograms (`fullPass=false`) keyed by leaf + contiguous partition reorder + the histogram subtraction trick, matching the CPU path ≤1e-4 **oracle-tested on Kaggle CUDA**.
  2. A Logloss classification model with Newton der2 leaf estimation and the Cosine (GPU-default) score function trains fully on device ≤1e-4 vs the Rust CPU path, oracle-tested on Kaggle CUDA.
  3. **Speed check (BENCH-02, standing):** depth-6 RMSE and Logloss device training is timed on Kaggle CUDA and reported as device path vs the host-CPU baseline AND vs official CatBoost GPU (warm-run/JIT-excluded, train-only) — this phase's keystone kernels (partition-aware histograms + subtraction trick + Newton der2) carry their own recorded CUDA speed measurement, not deferred to Phase 13.
  4. The chosen deterministic reduction strategy holds device histogram and score reductions within ε=1e-4 of the CPU path across hundreds of trees on CUDA, with no split flips compounding over the boosting run (CUDA's f64 atomic-add is available, but `atomicAdd` ordering is still non-deterministic, so the deterministic reduction is what holds the bar).
  5. Every device-covered case to date holds ε=1e-4 vs the Rust CPU path **on Kaggle CUDA**, and the CPU/host training paths remain byte-unchanged (D-04) — the standing GPUT-14 gate.
**Plans**: TBD
**UI hint**: no
**Notes**: The single largest kernel extension of the milestone — partition-aware `pointwise_hist2` keyed by `leaf_of[obj]` into `2^level` slots; `TDataPartition{Offset,Size}` contiguous layout; parent-resident sibling-by-subtraction. Reuses Phase 7.2 der2 handles for Newton. Oracle + speed of record = Kaggle CUDA (human-gated notebook); the optional ROCm in-env build is a fast local smoke check only. **BENCH-02's standing per-phase speed check applies here**: depth-6 RMSE+Logloss device training must be timed on Kaggle CUDA (vs CPU and vs official CatBoost GPU) before this phase's kernels are considered done. Landmines: deterministic reduction strategy mandatory (CUDA atomicAdd ordering still non-deterministic; gfx1100 still lacks f64 atomic-add for the smoke path); `f32::MIN` sentinel. Research flags: spike the reduction-determinism strategy as step 0 (before the histogram kernel); verify the multi-block scan carry ("Open Q2") against the vendored CubeCL manual. Given the kernel complexity, plans may decompose this phase into sub-waves (depth>1 histograms → reduction determinism → Newton/Cosine).

### Phase 12: GPU Coverage Expansion (Sampling / CTR / Pairwise / Multiclass / Ordered)
**Goal**: Each remaining training family transitions from `Ok(None)`→CPU-fallback to `Ok(Some(tree))`→device path, independently and behind the same fallback gate, each gated by a Kaggle CUDA ε=1e-4 sign-off AND timed on Kaggle CUDA as it lands. Recommended sub-order: bootstrap + random-strength (small, high-return), then CTR (headline categorical use case), then pairwise/ranking (reuses Phase 7.4 kernels), then multiclass, then ordered boosting (heaviest residency). Each sub-feature lands only when it passes ε=1e-4 sign-off on Kaggle CUDA; until then users transparently fall back to the CPU path.
**Depends on**: Phase 11
**Requirements**: GPUT-09, GPUT-10, GPUT-11, GPUT-12, GPUT-13
**Success Criteria** (what must be TRUE):
  1. Bootstrap + random-strength sampling runs on device with sampling parity for non-default `bootstrap_type`, matching the CPU path ≤1e-4 **oracle-tested on Kaggle CUDA**.
  2. CTR / permutation-dependent categorical features train on device ≤1e-4 vs the Rust CPU path on Kaggle CUDA.
  3. The pairwise/ranking loss path and the multiclass path each train on device ≤1e-4 vs CPU on Kaggle CUDA (the pairwise path reuses the Phase 7.4 pairwise-histogram kernels); ordered boosting (`EBoostingType::Ordered`) trains on device ≤1e-4 vs CPU on Kaggle CUDA.
  4. **Speed check (BENCH-02, standing):** each coverage family (sampling / CTR / pairwise / multiclass / ordered) is timed on Kaggle CUDA **as it lands** — device path vs the host-CPU baseline, and vs official CatBoost GPU where a comparable config exists (warm-run/JIT-excluded, train-only) — so every family's kernels carry their own recorded CUDA speed measurement when they flip from `Ok(None)`→device, not deferred to Phase 13.
  5. Any sub-feature not yet passing Kaggle CUDA sign-off returns `Ok(None)`→CPU fallback (no incorrect device result), and the resulting GPU coverage matrix (correctness + per-family speed) is documented.
**Plans**: TBD
**UI hint**: no
**Notes**: Each family is independently shippable and deferrable — can be planned/executed as parallel sub-workstreams, or cut to a bootstrap + CTR MVP if Phase 11 runs long. Every per-family sign-off is a human-gated Kaggle CUDA oracle run (reusing the Phase-10 harness) AND a Kaggle CUDA speed measurement; the optional ROCm smoke build is a local convenience only. **BENCH-02's standing per-phase speed check applies here**: each coverage family is timed on Kaggle CUDA as it lands, contributing its measurement to the aggregate Phase-13 sign-off. Research flags: CTR on device has the highest uncertainty (targeted read of `batch_binarized_ctr_calcer.h` + `ctrs/` before planning that sub-task); the pairwise partition + leaves oracle (`leaves_estimation/pairwise_oracle.h`) is under-documented — read it before implementing. Plans may sub-split this phase (e.g. 12.1 sampling, 12.2 CTR, 12.3 pairwise/multiclass, 12.4 ordered) given the kernel breadth. Landmine constraints unchanged.

### Phase 13: Comprehensive Kaggle CUDA Speed Benchmark + Parity Sign-Off
**Goal**: Prove the device-resident path demonstrably closes the >20× gap vs official CatBoost GPU via a **comprehensive final speed-parity sign-off (BENCH-03) that AGGREGATES the per-phase speed checks** recorded in Phases 10–12 — with CUDA correctness established as a blocking gate before any speed number is trusted. This phase does NOT re-establish the oracle or measure speed for the first time (Phases 10–12 already measured per-phase speed via the standing BENCH-02 check); it extends the Phase-10 Kaggle CUDA harness to a comprehensive head-to-head timing across the workload matrix and the milestone-closing ε sign-off.
**Depends on**: Phase 10 (the Kaggle CUDA oracle + speed harness it builds on), Phase 11 (depth>1 device-resident + its recorded speed check), Phase 12 (coverage families + their per-family speed checks inform the workload matrix)
**Requirements**: BENCH-03
**Success Criteria** (what must be TRUE):
  1. The Phase-10 Kaggle CUDA harness times official CatBoost GPU vs catboost-rs across the full workload matrix on identical datasets/params, with warm-run/JIT exclusion and train-only (not I/O) wall-clock measurement, **aggregating the per-phase speed checks (BENCH-02)** from Phases 10–12 into one comprehensive comparison.
  2. The correctness oracle is re-confirmed on the CUDA backend (≤1e-4 vs the Rust CPU path) as a **blocking gate before any speed number is reported** — reusing the authoritative Phase-10 CUDA oracle, so a fast-but-wrong CUDA result is never quoted.
  3. The device-resident path demonstrably closes the >20× gap (BENCH-03): a documented, signed-off **comprehensive final** speed-parity result vs official CatBoost GPU on Kaggle CUDA, measured against the pre-Phase-10 host-light baseline and aggregating every per-phase speed measurement into the milestone-closing sign-off.
**Plans**: TBD
**UI hint**: no
**Notes**: This is the comprehensive FINAL aggregate, not the first place speed is measured — Phases 10–12 each carried their own standing BENCH-02 Kaggle CUDA speed check; Phase 13 rolls them up into one head-to-head sign-off. Protocol fully specified (STACK.md / PITFALLS.md / existing `benchmark.py` template). This is a human-gated Kaggle CUDA execution. Execution checklist: verify CUDA backend active via `nvidia-smi`, warm one untimed fit, drain the lazy CubeCL queue with a read-back/predict before stopping the clock, re-run the oracle (Phase-10 harness) before timing. Standard patterns — no new compute crates; `criterion 0.7.x` (dev-dep) for optional in-env ROCm relative-timing regression during development, optional `profile-tracy`/`tracing` behind a `profiling` Cargo feature.

## Progress (v1.1)

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 10. Seam + Residency + Depth-1 + Kaggle CUDA Oracle & Speed Harness | 0/TBD | Not started | - |
| 11. Depth>1 Histograms + Reduction Determinism + Newton Der2 | 0/TBD | Not started | - |
| 12. GPU Coverage Expansion | 0/TBD | Not started | - |
| 13. Comprehensive Kaggle CUDA Benchmark + Sign-Off | 0/TBD | Not started | - |

## Backlog (Deferred from v1.0)

### Phase 9: Online HNSW Estimated-Feature Parity — DEFERRED

**Status**: deferred backlog at v1.0 close (carried, not dropped). Re-surface as its own milestone when KNN estimated-feature bit-exact parity is prioritized. Planning context preserved at `.planning/milestones/v1.0-phases/09-online-hnsw-estimated-feature-parity/`.

**Goal**: Port `catboost-master/library/cpp/online_hnsw/base` to Rust bit-for-bit so the KNN estimated-feature calcer returns upstream-identical neighbor sets, closing the XOR per-stage ≤1e-5 oracle gate that the brute-force-exact calcer (Phase 6.5 A2/D-05) cannot.
**Depends on**: Phase 6.5 (estimated-feature calcer + frozen `text_embedding_xor/` fixture)
**Requirements**: FEAT-07

**Scope:**

1. Port the dynamic dense graph + incremental insert + HNSW search bit-for-bit from `online_hnsw/base/`:
   - `dynamic_dense_graph.{h,cpp}`, `item_storage_index.{h,cpp}`, `index_base.{h,cpp}`, `build_options.{h,cpp}`, `index_data.h`, `index_reader/writer.{h,cpp}`, `index_snapshot_data.h`
   - Build options default to `MaxNeighbors=32`, `SearchNeighborhoodSize=300`, `LevelSizeDecay/NumVertices = AUTO_SELECT(0)`; calcer constructs with `CloseNum=k` and search size `300`.
   - Distance: `TL2SqrDistance<float>` (squared L2), `float` vectors.
2. **Replicate the construction RNG exactly** — upstream drives graph build (neighbor selection / level assignment) from its own RNG; bit-exact neighbors require reproducing the seed source and draw order. This is the crux.
3. Wire both calcer flavors at the seam (`cb-train/src/estimated/online_embedding.rs`, `estimated_features.rs`): the online incremental `AddItem`→`GetNearestNeighbors` path (tree structure+leaves) and the offline whole-set apply path (predictions).
4. Flip the existing RED-on-success gate (`xor_oracle_per_stage_residual_…`) to a passing ≤1e-5 oracle; the `text_embedding_xor/` fixture is frozen — no regeneration.

**Success Criteria:**

- **SC-1** — Rust HNSW returns upstream-identical neighbor IDs on the instrumented `knn_neighbors` evidence corpus (e.g. cloud-B query doc6 over prefix `{14,15,0,7,4}` yields upstream's `{1,3,4}`, not the exact `{0,2,4}`); divergence-from-exact reproduced, not merely "close".
- **SC-2** — Both the online (`TKNNUpdatableCloud`) and offline (`TKNNCloud`) paths match upstream neighbor IDs bit-for-bit across the full XOR corpus.
- **SC-3** — The non-degenerate XOR text+embedding+numeric corpus: StagedApprox + Predictions ≤1e-5 vs upstream, with **no** structure-invariant leaf-order relaxation and the KNN vote border serializing as `0.5` (not `1.5`).
- **SC-4** — The honest oracle test passes with no `#[ignore]` and no weakened tolerance; class-vote ordering matches upstream (feat0 = class-1 vote).

**Notes / risks:**

- The bit-exact dependency is the **RNG-driven build order** — replicate the seed and draw sequence first; make this an explicit gray area in `/gsd-discuss-phase`.
- Reference only the vendored C++ (`library/cpp/online_hnsw/base/` + `private/libs/embedding_features/knn.{h,cpp}`). Do **not** use sklearn-ann / annoy / faiss / nmslib — different ANN algorithms cannot be bit-matched.
- Instrumented trainer rebuild recipe (for evidence diffing) is in `.planning/todos/pending/estimated-feature-grid-parity.md` and the `catboost-instrumented-trainer-build` memory. Port surface is 832 LOC across `online_hnsw/base/` plus the `knn.{h,cpp}` call site.
