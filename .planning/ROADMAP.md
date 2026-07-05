# Roadmap: catboost-rs

## Milestones

- ✅ **v1.0 Core Parity** — Phases 1–8 (shipped 2026-06-28)
- ✅ **v1.1 GPU Performance** — Phases 10–14 (shipped 2026-07-05) — the boosting inner loop moved fully device-resident (CubeCL, no CUB); BENCH-03: PASS, 23.9×–42.1× vs the host-light CPU baseline on P100. Closed with accepted standing debt (GPUT-14 aggregate + Phase-10/11 BENCH-02 un-run). Full detail: `milestones/v1.1-ROADMAP.md`.

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

<details>
<summary>✅ v1.1 GPU Performance (Phases 10–14) — SHIPPED 2026-07-05</summary>

**Milestone goal:** Move the entire boosting inner loop (histogram build, split scoring, BestSplit, partition/leaf-assignment, leaf values) onto the GPU — not just derivatives — closing the >20× gap vs official CatBoost GPU while preserving correctness. Re-scoped in place 2026-07-02 against `CATBOOST_CUDA_KERNELS_DESIGN.md` (79 `.cu` + 77 `.cuh` across 9 kernel directories; 17 → 25 requirements). All GPU kernel oracles (correctness AND speed) validated on Kaggle CUDA (ROCm in-env not a gate); device path held to ε=1e-4 vs the Rust CPU path (depth-1 tighter at ≤1e-5); CPU path byte-unchanged (D-04).

- [x] Phase 10: GPU Foundations — Runtime Seam, Session Residency, Device-Primitive Library, Compressed Index, Depth-1 + Kaggle CUDA Oracle & Speed Harness (9/9 plans) — completed 2026-07-03
- [x] Phase 11: Depth>1 Partition-Aware Histograms + Reduction Determinism + Newton Der2 (5/5 plans) — completed 2026-07-04
- [x] Phase 12: Grow-Policy, Leaf-Method, Sampling & Categorical Device Coverage (9/9 plans) — completed 2026-07-04
- [x] Phase 13: Pairwise, Ranking, Multiclass, Ordered & Langevin Device Coverage (10/10 plans) — completed 2026-07-04
- [x] Phase 14: Comprehensive Kaggle CUDA Speed Benchmark + Parity Sign-Off (3/3 plans) — completed 2026-07-05

Full per-phase detail: `.planning/milestones/v1.1-ROADMAP.md` and `.planning/milestones/v1.1-REQUIREMENTS.md`.
**Standing debt accepted at close** (formal override in `14-VERIFICATION.md`): `GPUT-14` milestone-wide aggregate sign-off Pending; Phase-10 (depth-1) + Phase-11 (depth-6) BENCH-02 Kaggle rows un-run. Per-family ≤1e-4 evidence + committed P100 runs stand. See `.planning/MILESTONES.md` → Known Gaps and STATE.md → Deferred Items.

</details>

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1–8 (Core Parity) | v1.0 | — | Complete | 2026-06-28 |
| 10. GPU Foundations — Seam + Residency + Primitive Library + cindex + Depth-1 + Kaggle CUDA Harness | v1.1 | 9/9 | Complete | 2026-07-03 |
| 11. Depth>1 Histograms + Reduction Determinism + Newton Der2 | v1.1 | 5/5 | Complete | 2026-07-04 |
| 12. Grow-Policy, Leaf-Method, Sampling & Categorical Coverage | v1.1 | 9/9 | Complete | 2026-07-04 |
| 13. Pairwise, Ranking, Multiclass, Ordered & Langevin Coverage | v1.1 | 10/10 | Complete | 2026-07-04 |
| 14. Comprehensive Kaggle CUDA Benchmark + Sign-Off | v1.1 | 3/3 | Complete | 2026-07-05 |

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
