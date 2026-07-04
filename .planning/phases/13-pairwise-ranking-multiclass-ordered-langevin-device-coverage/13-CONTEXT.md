# Phase 13: Pairwise, Ranking, Multiclass, Ordered & Langevin Device Coverage - Context

**Gathered:** 2026-07-04
**Status:** Ready for planning

<domain>
## Phase Boundary

Expand the device-resident training path (built in Phases 10–12) across the final **five loss-family / multi-output / ordered-residency families**, each flipping from `Ok(None)`→CPU-fallback to an `Ok(Some(tree))`→device path behind the same per-fit fallback gate, each gated by a **Kaggle CUDA ε=1e-4 sign-off AND timed on Kaggle CUDA as it lands**:

1. **PairLogit pairwise** (GPUT-11; §6.3 `pairwise_hist*`, reusing Phase 7.4 kernels) + **batched device Cholesky solver** (GPUT-21; per-leaf `MakePairwiseDerivatives`/`MakePointwiseDerivatives` matrix assembly + batched Cholesky decomposition, forward/back substitution, ridge regularization, `CalcScoresCholesky`; §6.3 `split_pairwise`/`linear_solver`).
2. **Query/listwise ranking** (GPUT-22; QueryRMSE, QuerySoftMax, QueryCrossEntropy, YetiRank, PFound-F + device query-grouping infra — group ids/means/max, group-bias removal, in-query sampling radix sort, taken-docs masks; §6.5 `query_*`, `yeti_rank_pointwise`, `pfound_f`, §6.6a `query_helper.cu`).
3. **Multiclass / multi-target / uncertainty** (GPUT-12; MultiClass, MultiClassOneVsAll, MultiCrossEntropy, MultiRMSE, RMSEWithUncertainty — multilogit multi-row der2 blocks; §6.5 `multilogit`).
4. **Ordered boosting** (GPUT-13; `EBoostingType::Ordered` — heaviest residency).
5. **Langevin / SGLB noise** (GPUT-20; `AddLangevinNoise`: per-element seeded Gaussian on the reduced derivatives; §6.3 `langevin_utils`).

**Ambition:** ALL FIVE families this phase, **hard commit** (ordered boosting included, no planned slip) — one phase with sequenced waves, not formal sub-phases 13.1–13.5. Each family remains independently shippable behind its own `Ok(None)` gate, so partial failure is still correctness-safe, but the phase intent is full device coverage of all five. This completes the device-coverage surface feeding the Phase-14 aggregate sign-off.

**Recommended sub-order (roadmap, retained):** PairLogit + Cholesky solver (reuses Phase 7.4 pairwise-histogram kernels) → query/listwise ranking (query-grouping infra) → multiclass/multi-target/uncertainty → ordered boosting (heaviest residency) → Langevin/SGLB (small, layers on the reduced derivatives).

**No Region-style "build CPU first" gap:** unlike Phase 12's Region, **all five families already have CPU oracles from v1.0** (Phases 6.1–6.4 losses/metrics; Phase 5 ordered boosting). The lift is device-side only. The one *representation* extension needed is multi-output leaves (see D-04) — a `DeviceGrownTree` extension, not new CPU work.

**Scope anchor — already LOCKED (carried forward from Phases 10/11/12, not re-decided here):**
- **ε bar:** device path holds **ε=1e-4 vs the Rust CPU path** (GPUT-14 operative standing gate); CPU/host path stays oracle-locked ≤1e-5 and **byte-unchanged** (D-04 no-regression).
- **Per-fit all-or-nothing** (D-10-01): a fit is either fully device-grown or fully CPU-grown; any family not yet passing Kaggle CUDA sign-off returns `Ok(None)`→host CPU grower.
- **Kaggle CUDA is the sole correctness+speed authority** (human-gated `--features cuda` notebook, reusing the Phase-10 harness). ROCm in-env is an optional compile/smoke convenience, **not a gate**.
- **Standing BENCH-02 per-family speed check:** each family (pairwise / ranking / multiclass / ordered / langevin) is timed on Kaggle CUDA **as it lands** — device vs host-CPU baseline AND vs official CatBoost GPU where a comparable config exists (warm-run/JIT-excluded, train-only) — not deferred to Phase 14.
- Only the O(1) BestSplit descriptor + `2^depth`/per-leaf partition stats cross host↔device per level (D-05).
- **Standing landmines:** never add a `cb-train` dep to `cb-backend` (transcribe CPU refs inline); no `-inf` float literals in `#[cube]` kernels (use `f32::MIN` sentinel); deterministic reduction mandatory (fixed-point `Atomic<u64>` accumulator k=30 + fixed-order tree-reduce fallback, SPIKE-REDUCTION §5b); never read a `Handle` through a client other than the one that allocated it.
- The final GPU **coverage matrix** (per-family correctness + speed) is documented (SC-5).

</domain>

<decisions>
## Implementation Decisions

### Phase scope & structure
- **D-01 (all 5 families, one phase, sequenced waves — HARD COMMIT):** Attempt all five families in a single Phase 13; the planner decomposes into internal waves following the roadmap sub-order (pairwise+solver → ranking → multiclass → ordered → langevin). No formal 13.1–13.5 sub-phase split. Ambition is **hard commit to all five including ordered boosting** — no planned slip. Each family still lands behind its own `Ok(None)` gate (so any single family failing sign-off is correctness-safe and falls back to CPU), but the phase target is full device coverage of all five, not a subset.
- **D-02 (retain roadmap ordering):** Keep the roadmap's sub-order — front-loads pairwise (reuses Phase 7.4 histogram kernels), amortizes the query-grouping infra across all ranking objectives next, then multiclass, then the heaviest residency (ordered), ending on the small Langevin noise layer.

### Multi-output leaves — multiclass / multi-target / uncertainty (GPUT-12)
- **D-03 (single tree, block leaves — mirror upstream multilogit):** Grow ONE shared tree structure per boosting step with **multi-row leaf blocks** — extend `DeviceGrownTree.leaf_values` from scalar `Vec<f64>` to a flat block of `leaf_count × approx_dim` (carry `approx_dim`), routed through the existing multi-output CPU apply. NOT `K` independent scalar trees — that diverges from upstream and is wrong for the coupled (non-separable) softmax `MultiClass` der2. This is the one representation extension the phase needs (analogous to how Phase 12 extended `DeviceGrownTree` for the non-sym/Region carriers); boundary stays plain-host-structs (landmine-safe).
- **D-04 (full multi-row der2 block leaf estimation):** Device leaf estimation solves the true multi-row der2 block per leaf — **coupled** for `MultiClass` softmax, **diagonal** for the separable losses (`MultiClassOneVsAll`, `MultiLogloss`/multilabel, `MultiRMSE`, `RMSEWithUncertainty`) — matching the CPU Newton path. Reuse/extend the Phase-11 Newton der2 machinery to K-dim blocks. The der-computation losses themselves already exist in `cb-compute` (Phase 6.2/7.2 `MultiClass`/`OneVsAll`/`MultiCrossEntropy`/`MultiRMSE`/`RMSEWithUncertainty`) — the device work is the multi-row der2 leaf solve + block-leaf emission, not the der functions.

### Ordered boosting device residency (GPUT-13) — heaviest lift
- **D-05 (full device residency across iterations):** The per-permutation ordered-boosting approx state stays **device-resident across iterations** (upstream keeps ordered approxes on GPU) — only O(1) descriptors cross the seam per level, exactly like the plain path (D-05/Phase-10). Chosen over the lighter "resident approx + host-computed permutation" variant to preserve true no-readback residency, consistent with the milestone's speed goal and the hard-commit ambition (D-01). This is the heaviest residency item in the phase.
- **D-06 (pin seed, freeze permutation in fixture):** Match the CPU ordered path (Phase 5, ≤1e-5) at ε=1e-4 by mirroring Phase 12's D-07 discipline — pin the RNG seed / permutation config, **freeze the exact CPU-reference permutation + per-permutation approx trajectory in the oracle fixture**, reproduce bit-for-bit on device. Deterministic and checkable at the ε bar (not a distributional/statistical check).

### Pairwise Cholesky solver (GPUT-11 / GPUT-21) — highest uncertainty
- **D-07 (f64 batched Cholesky + ridge, mirror upstream `linear_solver`):** The batched device Cholesky solver runs **decomposition + forward/back substitution in f64 accumulation** with upstream's ridge (l2) on the diagonal, matching `CalcScoresCholesky`. f64 for the solve holds ε=1e-4 across hundreds of trees even where the histogram uses the fixed-point `Atomic<u64>` reduction. Batched over leaves. Per-leaf pairwise-derivative matrix assembly is `MakePairwiseDerivatives`/`MakePointwiseDerivatives`. PairLogit pairwise histograms reuse the Phase 7.4 4-channel pairwise-histogram kernels.

### Query/listwise ranking (GPUT-22)
- **D-08 (all 5 objectives on device this phase, incl. stochastic pair):** Cover QueryRMSE, QuerySoftMax, QueryCrossEntropy, YetiRank, **and** PFound-F on device this phase. The stochastic pair (YetiRank / PFound-F — in-query sampling radix sort) uses the pinned-seed / frozen-fixture discipline (D-06). The shared device query-grouping infrastructure (group ids/means/max, group-bias removal, in-query sampling radix sort, taken-docs masks) is built once and amortized across all five objectives. Complete GPUT-22 coverage, not a deterministic-only subset.

### Langevin / SGLB (GPUT-20)
- **D-09 (device Gaussian noise on the reduced derivatives, pinned seed):** `AddLangevinNoise` adds a per-element seeded Gaussian to the reduced derivatives on device (§6.3 `langevin_utils`), layering on the existing der residency — the smallest family, landed last. Pinned-seed / frozen-fixture reproduction at ε=1e-4 (D-06 discipline).

### Claude's Discretion
- Internal wave decomposition/ordering beyond the pinned roadmap sub-order — planner refines (query-grouping infra likely a shared sub-wave before the ranking objectives; multi-output leaf-block extension a shared sub-wave before multiclass).
- Query-grouping infra mechanics (group-bias removal, in-query sampling radix-sort layout, taken-docs masks) — research resolves against §6.5/§6.6a `query_helper.cu`.
- Cholesky pivoting/ordering details and the exact ridge/`l2` term placement — research resolves against §6.3 `linear_solver` + the under-documented `leaves_estimation/pairwise_oracle.h` (read before implementing, per roadmap research flag).
- Langevin Gaussian RNG stream layout and per-element seeding — research resolves against §6.3 `langevin_utils` + the CPU Langevin/SGLB reference.
- Which specific multiclass fixtures (class counts, coupled vs diagonal) get device oracles — research/planning resolve against the CPU multi-output references.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### GPU kernel design authority (v1.1)
- `CATBOOST_CUDA_KERNELS_DESIGN.md` — the complete upstream CUDA training-kernel map. Specifically for Phase 13:
  - **§6.3 `pairwise_hist*.{cu,cuh}`** — PairLogit pairwise 2×2-cell histograms (reuse the Phase 7.4 4-channel kernels; GPUT-11).
  - **§6.3 `split_pairwise.{cu,cuh}`, `linear_solver.{cu,cuh}`** — `MakePairwiseDerivatives`/`MakePointwiseDerivatives` per-leaf matrix assembly + batched Cholesky decomposition, forward/back substitution, ridge, `CalcScoresCholesky` (GPUT-21, highest-uncertainty — read closely).
  - **§6.5 `query_*.{cu,cuh}`, `yeti_rank_pointwise`, `pfound_f`** + **§6.6a `query_helper.cu`** — query/listwise objectives + device query-grouping infra (GPUT-22).
  - **§6.5 `multilogit.{cu,cuh}`** — multiclass/multi-target/uncertainty multi-row der2 blocks (GPUT-12).
  - **§6.3 `langevin_utils.{cu,cuh}`** — `AddLangevinNoise` seeded Gaussian on reduced derivatives (GPUT-20).
  - **§5.1–5.3, §6.6 `models/kernel`** — generic structure searcher → host `BuildTreeLikeModel<TModel>` → per-shape host model + applier (the emission architecture; multi-output block leaves route through the multi-output CPU apply).
- **`leaves_estimation/pairwise_oracle.h`** (in `catboost-master/`, upstream) — the pairwise partition + leaves oracle; roadmap flags it **under-documented** — read before implementing the Cholesky/pairwise leaf path.

### Phase 10/11/12 deliverables consumed as-is (substrate)
- `.planning/phases/12-.../12-CONTEXT.md` — coverage-fallback patterns (each family independent behind `Ok(None)`), pinned-seed/frozen-fixture sampling discipline (D-07), `DeviceGrownTree` extension precedent (non-sym/Region carriers), standing ε bars + landmines.
- `.planning/phases/11-.../11-CONTEXT.md` — depth>1 partition-aware histogram + subtraction trick + **Newton der2** (the machinery D-04 extends to K-dim blocks) + reduction-determinism locked scope.
- `.planning/phases/10-.../10-CONTEXT.md` — seam signatures, `GpuTrainSession` residency, cindex packing, `Ok(None)` all-or-nothing, ε bars, landmines.
- `.planning/phases/10-.../SPIKE-REDUCTION.md` — deterministic-reduction decision (fixed-point `Atomic<u64>` k=30 + fixed-order tree-reduce fallback).

### Requirements, roadmap & milestone framing
- `.planning/REQUIREMENTS.md` — GPUT-11/21/22/12/13/20 + GPUT-14 + BENCH-02 requirement text + traceability.
- `.planning/ROADMAP.md` — Phase 13 Goal, Success Criteria 1–5, Notes (research flags, sub-split guidance, ordered-boosting-heaviest-residency), standing landmines, Kaggle CUDA validation authority.
- `.planning/PROJECT.md` — v1.1 milestone goal, target features, the no-`cb-train`-dep landmine.
- `.planning/notes/gpu-training-host-light-root-cause.md` — the >20× host-light gap this milestone closes.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `crates/cb-compute/src/runtime.rs` — the der-computation losses for all five families already exist (`MultiClass` / `MultiClassOneVsAll` / `MultiCrossEntropy` / `MultiRMSE` / `RMSEWithUncertainty` / `MultiQuantile`, incl. `approx_dimension` docs at :171–283). Phase 13 adds the device leaf-solve + block-leaf emission, NOT the der functions.
- `crates/cb-compute/src/runtime.rs:927` `DeviceGrownTree` — today carries scalar `leaf_values: Vec<f64>` + oblivious/non-sym/Region carriers. **D-03 extends it** to a `leaf_count × approx_dim` block + `approx_dim` for multi-output.
- Phase 7.4 pairwise-histogram kernels (4-channel weight-only `(f*n_bins+bin)*4+histId`) in `cb-backend` — reused directly by the PairLogit path (D-07); the pairwise split-scorer landed in Phase 7.5.
- Phase 11 Newton der2 machinery (`Σder2` channel, `newton_leaf_delta`, `apply_leaf_delta` refinement) — extended to K-dim multi-row blocks (D-04).
- CPU oracles (all ≤1e-5, v1.0): six ranking losses incl. YetiRank(/Pairwise), PairLogit(/Pairwise), QueryRMSE, QuerySoftMax (Phase 6.1–6.3); multiclass/multilabel/uncertainty (Phase 6.2/6.4); ordered boosting (Phase 5); Langevin/SGLB (Phase 6.x). These are the frozen references for the device fixtures.
- Phase 10/11 device substrate: primitive library (scan/segmented-scan/segmented radix sort/reduce-by-key/partition-update/stat-aggregation), resident cindex, `grow_boosting_pass`/`grow_oblivious_tree_into`, der1/der2 seam (`der_seams.rs`), `apply_leaf_delta`, partition-aware `pointwise_hist2` + subtraction trick.

### Established Patterns
- Generic runtime over `SelectedRuntime` (cpu/wgpu/cuda/rocm), no runtime dispatch — one feature-gated impl.
- `Ok(None)` → host-CPU fallback keeps every increment oracle-safe (D-04 no-regression); each family flips from `Ok(None)` to covered independently.
- Serial CPU self-oracle for GPU kernels; max abs/rel divergence over equal-length buffers at the ε bar.
- Pinned-seed / frozen-fixture reproduction for RNG-driven paths (Phase 12 D-07) — reused for ordered permutations (D-06), stochastic ranking (YetiRank/PFound-F, D-08), and Langevin noise (D-09).
- Upstream emission architecture (§5.1–5.3): generic device structure search → ONE host `BuildTreeLikeModel<TModel>` step → per-shape host model + applier — multi-output block leaves route through the existing multi-output CPU apply (D-03).

### Integration Points
- New device kernels (pairwise Cholesky solve, query-grouping infra + ranking objectives, multi-row der2 block leaf, ordered-boosting resident approx state, Langevin noise) live in `cb-backend` (`kernels/` + `gpu_runtime`), driven per-level through the Phase-10 `Runtime` grow-tree seam wired into `cb_train::train`. Boundary crosses **plain host structs only** (landmine: no `cb-train` dep in `cb-backend` — transcribe CPU refs inline).
- `DeviceGrownTree` multi-output extension (D-03) is a shared prerequisite sub-wave before the multiclass family.
- Device query-grouping infra (D-08) is a shared sub-wave before the ranking objectives.

</code_context>

<specifics>
## Specific Ideas

- Multi-output: single shared tree structure, `leaf_count × approx_dim` block leaves, full multi-row der2 solve (coupled softmax / diagonal separable), routed through the existing multi-output CPU apply (D-03/D-04).
- Ordered boosting: per-permutation approx state device-resident across iterations, only O(1) descriptors cross the seam; frozen permutation + approx trajectory in the fixture (D-05/D-06).
- Pairwise: reuse Phase 7.4 pairwise histograms; f64 batched Cholesky + ridge mirroring `CalcScoresCholesky`/`linear_solver` (D-07).
- Ranking: shared device query-grouping infra amortized across all five objectives; stochastic YetiRank/PFound-F under pinned-seed (D-08).
- Langevin: per-element seeded Gaussian on reduced derivatives, layered last (D-09).

</specifics>

<deferred>
## Deferred Ideas

- **Comprehensive aggregate speed benchmark + real named datasets (Higgs/Epsilon), the >20× gap sign-off** — Phase 14 (BENCH-03); Phase 13 reuses the Phase-10 synthetic generator + per-family BENCH-02 checks.
- **On-device border/quantile computation** (`FastGpuBorders`) — out of scope milestone-wide; host CPU quantization stays the ≤1e-5 reference.
- **Formal 13.1–13.5 sub-phase split** — considered, declined (D-01, one phase with waves); revisit only if a family needs an independent verification/ship ceremony.
- **Deterministic-only ranking subset (drop YetiRank/PFound-F this phase)** — considered, declined in favor of all-5 coverage (D-08); noted as the lower-risk fallback if the stochastic in-query sampling over-runs.
- **`K` separate scalar trees for multiclass** — considered, declined (D-03) — diverges from upstream multilogit and is wrong for coupled softmax `MultiClass`.

### Reviewed Todos (not folded)
- **Estimated-feature stored-border-VALUE quantization-grid parity** (`estimated-feature-grid-parity.md`) — NOT folded. It concerns the KNN estimated-feature online-HNSW port (FEAT-07, deferred Phase 9), unrelated to Phase 13's loss-family device coverage. Keyword overlap only; out of scope.

</deferred>

---

*Phase: 13-pairwise-ranking-multiclass-ordered-langevin-device-coverage*
*Context gathered: 2026-07-04*
