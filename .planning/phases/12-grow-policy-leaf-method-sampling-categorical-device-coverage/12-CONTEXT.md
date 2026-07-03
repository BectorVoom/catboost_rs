# Phase 12: Grow-Policy, Leaf-Method, Sampling & Categorical Device Coverage - Context

**Gathered:** 2026-07-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Expand the device-resident training path (built in Phases 10–11) beyond its symmetric-tree / Newton-leaf / uniform-numeric default across **five independent tree-growth-mechanics families**, each flipping from `Ok(None)`→CPU-fallback to an `Ok(Some(tree))`→device path behind the same per-fit fallback gate, each gated by a **Kaggle CUDA ε=1e-4 sign-off AND timed on Kaggle CUDA as it lands**:

1. **Non-symmetric grow policies** — Depthwise, Lossguide, **and Region** (GPUT-18; §6.4, §6.6c).
2. **Exact weighted-quantile leaf-value estimation** (GPUT-19; §6.3 `exact_estimation`, distinct from the Newton path).
3. **Bootstrap + random-strength sampling** (GPUT-09; §6.1 `bootstrap`).
4. **MVS — Minimal Variance Sampling**, CatBoost's *default* GPU sampler (GPUT-17; §6.1 `mvs`).
5. **CTR / permutation-dependent categorical features** (GPUT-10; §6.6 `ctrs/kernel`).

**Ambition:** ALL FIVE families this phase (Phase 11's depth>1 partition/histogram substrate is shipped — plans 01–04 — and only paused on the human-gated Kaggle CUDA oracle, so the substrate is ready). Kept as **one phase with sequenced waves** (not formal sub-phases 12.1–12.4); each family remains independently shippable/deferrable behind its own `Ok(None)` gate.

**Recommended sub-order (roadmap, retained):** non-symmetric grow policies → Exact leaf → bootstrap/random-strength (small, high-return) → MVS (default sampler) → CTR (headline, highest uncertainty). Non-symmetric policies + CTR both build directly on Phase 11's depth>1 histogram/partition machinery.

**Scope anchor — already LOCKED (carried forward from Phases 10/11, not re-decided here):**
- **ε bar:** device path holds **ε=1e-4 vs the Rust CPU path** (GPUT-14 operative standing gate); CPU/host path stays oracle-locked ≤1e-5 and **byte-unchanged** (D-04 no-regression).
- **Per-fit all-or-nothing** (D-10-01): a fit is either fully device-grown or fully CPU-grown — no mixing device-grown and CPU-grown trees in one model. Any family not yet passing Kaggle CUDA sign-off returns `Ok(None)`→host CPU grower.
- **Kaggle CUDA is the sole correctness+speed authority** (human-gated `--features cuda` notebook, reusing the Phase-10 harness). ROCm in-env is an optional compile/smoke convenience, **not a gate**.
- **Standing BENCH-02 per-family speed check:** each family is timed on Kaggle CUDA **as it lands** — device vs host-CPU baseline AND vs official CatBoost GPU where a comparable config exists (warm-run/JIT-excluded, train-only) — not deferred to Phase 14.
- Only the O(1) BestSplit descriptor + `2^depth`/per-leaf partition stats cross host↔device per level (D-05).
- **Standing landmines:** never add a `cb-train` dep to `cb-backend` (transcribe CPU refs inline); no `-inf` float literals in `#[cube]` kernels (use `f32::MIN` sentinel); deterministic reduction mandatory (fixed-point `Atomic<u64>` accumulator k=30 + fixed-order tree-reduce fallback, SPIKE-REDUCTION §5b); never read a `Handle` through a client other than the one that allocated it.
- The final GPU **coverage matrix** (per-family correctness + speed) is documented (SC-5).

</domain>

<decisions>
## Implementation Decisions

### Family scope & phase structure
- **D-01 (all 5 families, one phase, sequenced waves):** Attempt all five families in a single Phase 12; the planner decomposes into internal waves following the roadmap sub-order (grow policies → Exact → bootstrap → MVS → CTR). No formal 12.1–12.4 sub-phase split. Each family lands independently behind its own `Ok(None)` gate, so partial completion is safe and any family can slip to a follow-up without blocking the others.
- **D-02 (retain roadmap ordering):** Keep the roadmap's sub-order — front-loads the tree-mechanics that reuse Phase 11's partition machinery most directly and ends on the highest-uncertainty CTR sub-task.

### Non-symmetric grow policies (GPUT-18) — ⚠ largest lift
- **D-03 (all three policies — Region requires building the CPU path FIRST):** Cover Depthwise, Lossguide **and Region** on device. **Depthwise + Lossguide** already have a CPU reference to oracle against at ε=1e-4 (`leaf_wise_grower` → `TreeVariant::NonSymmetric`). **Region does NOT** — it is a v1.0 escalated gap ("Region OUT", `boosting.rs:1332` `validate_grow_policy` rejects `EGrowPolicy::Region`), and `TreeVariant` has no Region variant. Therefore Region is a **two-step lift inside this phase**: (a) build the CPU Region path first — a Region grower + a `TRegionModel`-style **path** model variant (distinct from the non-symmetric node graph) + `AddRegion`/`ComputeRegionBins` apply — establishing the ≤1e-5 CPU oracle; then (b) the device Region path against it. **Planner MUST treat Region as its own wave (CPU Region → device Region); it is the single largest item in Phase 12 and pulls a v1.0-gap item into v1.1.**
- **D-04 (Depthwise/Lossguide emit via the existing CPU non-symmetric representation):** The device grows *structure + leaf values only*; extend `DeviceGrownTree` (`cb-compute/src/runtime.rs:917`) to carry the non-symmetric node graph (`step_nodes` `(left_diff,right_diff)` + `node_id_to_leaf_id`, mirroring `cb-train/src/tree.rs:216`) as **plain host structs**, feeding the existing `Model::from_trained` → `TreeVariant::NonSymmetric` builder + `cb-model/src/apply.rs` traversal. Boundary stays host-structs-only (landmine-safe). This exactly mirrors upstream's architecture: a generic structure searcher (`TGreedyTreeLikeStructureSearcher<TTreeModel>`) → one host `BuildTreeLikeModel<TModel>` step → per-shape host model + applier (§5.1–5.3, §6.6 `models/kernel`). Do NOT introduce a separate device-native non-symmetric tree type.
- **Region representation note (grounded in §6.6):** Upstream keeps Region as its OWN model shape `TRegionModel` — an oblivious-like *path* walked while the computed split matches the stored direction (`takeEqualAndSplitDirection` packs one-hot in bit 0, expected direction in bit 1; leaf = depth reached where the path diverges), NOT a `TTreeNode[]` binary node graph. The CPU Region variant (D-03a) must model this path shape, not reuse `NonSymmetricTree`.

### CTR / categorical device coverage (GPUT-10) — highest uncertainty
- **D-05 (full CTR including feature combinations):** Device-cover the complete CTR surface this phase — ordered target-statistic CTRs, one-hot, AND tensor / multi-feature feature-combination CTRs. Not a numeric-single-feature subset. Matches complete parity; multiplies the `ctrs/kernel` surface (accept the risk, ordered last per D-02).
- **D-06 (CTR value computation ON device — port `ctrs/kernel`):** The permutation-dependent target-statistics accumulation runs **on device**, staying resident across the permutation — port the upstream device CTR computation (`ctrs/kernel` + `batch_binarized_ctr_calcer`) so accumulated CTRs are binarized into additional cindex columns on device. NOT host-computed-then-uploaded. Largest kernel surface but true upstream residency parity, consistent with the milestone's speed goal. (This is the highest-uncertainty sub-task — see research flags.)

### Sampling parity (GPUT-09 bootstrap/random-strength + GPUT-17 MVS)
- **D-07 (pin seed + freeze in fixture):** Make the RNG-driven device sampling match the CPU path at ε=1e-4 by mirroring Phase 11's discipline — pin the RNG seed / sampling config, **freeze the exact CPU-reference sample in the oracle fixture**, and reproduce it bit-for-bit on device. Deterministic and checkable at the ε bar (not a looser distributional/statistical check).
- **D-08 (draw the keep-mask/weights ON device, resident):** The RNG-drawn keep-mask / inverse-probability weights are computed **on device** each iteration from the pinned seed, keeping the derivatives resident — MVS's per-block optimal threshold on `sqrt(der²+λ)` + reweight is a device reduction over resident derivatives anyway (§6.1 `mvs`). Preserves the no-readback residency that is the whole point of the milestone; no per-tree host round-trip for the sample mask.

### Exact weighted-quantile leaf estimation (GPUT-19)
- **D-09 (device Exact path, distinct from Newton):** Exact weighted-quantile estimation (`needWeights = totalWeight·α`, binary search over per-bin weight prefix sums) runs on device for the Quantile/MAE/MAPE-family objectives — upstream `EstimateExact` is **fully GPU**: `SegmentedRadixSort` → `SegmentedScanVector` → `ComputeWeightedQuantileWithBinarySearch` (§6.3 `exact_estimation`, §5.6). Reuse the Phase-10 device primitives (segmented sort/scan). Distinct from the Newton der2 path shipped in GPUT-07 (Phase 11).

### Claude's Discretion
- Internal wave decomposition/ordering beyond the pinned roadmap sub-order — planner refines (Region gets its own CPU→device wave per D-03).
- Exact leaf-estimation objective set (which of Quantile/MAE/MAPE/etc. get device fixtures) — research/planning resolve against the CPU Exact reference and §6.3.
- Device CTR ordered-permutation residency mechanics, the `ctrs/kernel` port shape, and the CTR→cindex binarization join — research resolves against `batch_binarized_ctr_calcer.h` + `ctrs/kernel/` (highest-uncertainty read).
- MVS block size / threshold-search mechanics and the device RNG stream layout for the pinned-seed reproduction — research resolves against §6.1 `mvs`/`bootstrap`/`random` and `cb-train/src/bootstrap.rs`.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### GPU kernel design authority (v1.1)
- `CATBOOST_CUDA_KERNELS_DESIGN.md` — the complete upstream CUDA training-kernel map. Specifically for Phase 12:
  - **§5.1–5.3** — generic structure searcher `TGreedyTreeLikeStructureSearcher<TTreeModel>` + `BuildTreeLikeModel<TModel>` → {oblivious, region, non-symmetric} host model types (the emission architecture D-04 mirrors).
  - **§5.4 `SelectLeavesToSplit` / §6.4** — per-policy leaf selection: `ComputeOptimalSplitsRegion` (Region), `ComputeOptimalSplit` (Depthwise/Lossguide), `ComputeOptimalSplits` (Symmetric); `MaxLeaves` forced to `MaxDepth+1` for Region.
  - **§6.6 `models/kernel/add_model_value.cu`** — the three apply shapes: `AddObliviousTree`, `AddRegion`/`ComputeRegionBins` (the Region *path* — walk-until-diverge), `ComputeNonSymmetricDecisionTreeBins` (`TTreeNode{FeatureId,Bin,LeftSubtree,RightSubtree}` binary traversal).
  - **§5.6 leaf-value estimation** — Exact path (`EstimateExact`) is fully GPU (`SegmentedRadixSort`→`SegmentedScanVector`→`ComputeWeightedQuantileWithBinarySearch`).
  - **§6.3 `exact_estimation.{cu,cuh}`** — GPUT-19 weighted-quantile kernels.
  - **§6.1 `bootstrap.{cu,cuh}`, `mvs.{cu,cuh}`, `random*.cuh`** — GPUT-09/GPUT-17 sampling + device RNG.
  - **§6.6 `ctrs/kernel`** + `batch_binarized_ctr_calcer.h` — GPUT-10 device CTR computation (highest-uncertainty targeted read before planning that wave).

### Phase 10/11 deliverables consumed as-is (substrate)
- `.planning/phases/11-.../11-CONTEXT.md` — depth>1 partition-aware histogram + subtraction trick + Newton der2 + reduction-determinism locked scope (the substrate Phase 12 builds on).
- `.planning/phases/10-.../10-CONTEXT.md` — seam signatures, `GpuTrainSession` residency, cindex packing, `Ok(None)` all-or-nothing, ε bars, landmines.
- `.planning/phases/10-.../SPIKE-REDUCTION.md` — deterministic-reduction decision (fixed-point `Atomic<u64>` k=30 + fixed-order tree-reduce fallback).

### Requirements, roadmap & milestone framing
- `.planning/REQUIREMENTS.md` — GPUT-18/19/09/17/10 + BENCH-02 requirement text + traceability.
- `.planning/ROADMAP.md` — Phase 12 Goal, Success Criteria 1–5, Notes (research flags, MVP-cut option, sub-split guidance), standing landmines, Kaggle CUDA validation authority.
- `.planning/PROJECT.md` — v1.1 milestone goal, target features, the no-`cb-train`-dep landmine.
- `.planning/notes/gpu-training-host-light-root-cause.md` — the >20× host-light gap this milestone closes.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `crates/cb-train/src/tree.rs` — `leaf_wise_grower` (Depthwise/Lossguide, TRUE non-symmetric node graph) + `GrownTree.step_nodes` / `node_id_to_leaf_id` (the non-sym representation D-04 reuses); `LeafWisePolicy` enum. **Region is NOT here** — rejected up front.
- `crates/cb-train/src/boosting.rs:1332` `validate_grow_policy` — the guard that REJECTS `EGrowPolicy::Region` ("Region OUT" v1.0 gap); D-03a must lift this once the CPU Region path exists. `EGrowPolicy` enum at `:103` (SymmetricTree/Lossguide/Depthwise/Region).
- `crates/cb-model/src/model.rs:143` `TreeVariant` — `Oblivious` + `NonSymmetric` only (NO Region variant yet); `Model::from_trained`; `apply.rs:236` non-symmetric traversal + `json.rs` flat-triple round-trip. D-03a adds the Region variant.
- `crates/cb-compute/src/runtime.rs:917` `DeviceGrownTree` — today oblivious-only (`splits`/`leaf_values`/`leaf_of`); D-04 extends it to carry the non-sym node graph.
- `crates/cb-train/src/bootstrap.rs` — CPU bootstrap/random-strength draw discipline (the ≤1e-5 reference + pinned-seed source for D-07/D-08).
- `crates/cb-train/src/ctr/` — CPU CTR (ordered TS / one-hot / tensor combinations) — the ≤1e-5 oracle for device CTR (D-05/D-06).
- `crates/cb-compute/src/leaf.rs` — CPU leaf estimation incl. Exact weighted-quantile reference (GPUT-19 oracle).
- Phase 10/11 device substrate: primitive library (scan / segmented-scan / segmented radix sort / reduce-by-key / partition-update / stat-aggregation), resident cindex, `grow_boosting_pass` / `grow_oblivious_tree_into`, der1/der2 seam (`der_seams.rs`), `apply_leaf_delta`, partition-aware `pointwise_hist2` + subtraction trick.

### Established Patterns
- Generic runtime over `SelectedRuntime` (cpu/wgpu/cuda/rocm), no runtime dispatch — one feature-gated impl.
- `Ok(None)` → host-CPU fallback keeps every increment oracle-safe (D-04 no-regression); each family flips from `Ok(None)` to covered independently.
- Serial CPU self-oracle for GPU kernels; max abs/rel divergence over equal-length buffers at the ε bar.
- Upstream emission architecture (§5.1–5.3): generic device structure search → ONE host `BuildTreeLikeModel<TModel>` step → per-shape host model + applier — D-04 (and the Region variant) mirror this.

### Integration Points
- New device kernels (non-sym per-policy scoring/selection, Region path apply, Exact quantile, device sampling/MVS, device CTR) live in `cb-backend` (`kernels/` + `gpu_runtime`), driven per-level through the Phase-10 `Runtime` grow-tree seam wired into `cb_train::train`. Boundary crosses **plain host structs only** (landmine: no `cb-train` dep in `cb-backend` — transcribe CPU refs inline).
- CPU Region path (D-03a) is new work in `cb-train` (grower + validate lift) + `cb-model` (Region variant + apply + json) BEFORE the device Region kernel.

</code_context>

<specifics>
## Specific Ideas

- Non-symmetric emission: device returns structure + leaf values; `DeviceGrownTree` extended with `step_nodes`/`node_id_to_leaf_id`; routed through the existing `TreeVariant::NonSymmetric` builder (D-04).
- Region is a distinct *path* model (`TRegionModel`, walk-until-diverge, leaf = depth reached), NOT a node graph — the new CPU Region variant models the path shape (D-03 note).
- Device CTR ordered-permutation accumulation resident on device (port `ctrs/kernel`), binarized into cindex columns, incl. tensor/feature-combination CTRs (D-05/D-06).
- Sampling: pin seed, freeze CPU-reference sample in fixture, device-resident RNG keep-mask/weights; MVS threshold+reweight as a device reduction (D-07/D-08).
- Exact leaf estimation via device segmented-sort → segmented-scan → binary-search weighted quantile (D-09).

</specifics>

<deferred>
## Deferred Ideas

- **Pairwise/ranking/multiclass/ordered/Langevin device families** — Phase 13 (build on this phase's coverage-fallback patterns).
- **Comprehensive aggregate speed benchmark + real named datasets (Higgs/Epsilon)** — Phase 14 (BENCH-03); Phase 12 reuses the Phase-10 synthetic generator + per-family BENCH-02 checks.
- **On-device border/quantile computation** (`FastGpuBorders`) — out of scope milestone-wide; host CPU quantization stays the ≤1e-5 reference.
- **Formal 12.1–12.4 sub-phase split** — considered, declined (D-01, one phase with waves); revisit only if a family needs independent verification/ship ceremony.
- **CTR host-computed fallback interpretation of GPUT-10** — considered, declined in favor of full device residency (D-06); noted as the lower-risk fallback if the `ctrs/kernel` port over-runs.

### Reviewed Todos (not folded)
None — no pending todos matched this phase's scope.

</deferred>

---

*Phase: 12-grow-policy-leaf-method-sampling-categorical-device-coverage*
*Context gathered: 2026-07-03*
