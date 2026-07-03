# Phase 12: Grow-Policy, Leaf-Method, Sampling & Categorical Device Coverage - Research

**Researched:** 2026-07-03
**Domain:** Device-resident GPU training (CubeCL) — non-symmetric/Region grow policies, Exact weighted-quantile leaves, bootstrap/random-strength/MVS sampling, on-device CTR; porting upstream CatBoost CUDA `greedy_subsets_searcher` / `ctrs` / `models` / `cuda_util` kernels to CubeCL against a Rust CPU oracle.
**Confidence:** HIGH (design authority + CPU reference + shipped substrate are all in-repo; the CTR wave carries the only genuinely open mechanics)

## Summary

This is a **coverage-expansion** phase, not a greenfield one. Phases 10–11 shipped the device-resident spine: the `Runtime` grow-tree seam (`begin_device_training` → `grow_tree_on_device` → `end_device_training`, `cb-compute/src/runtime.rs`), the `GpuTrainSession` residency + coverage gate (`cb-backend/src/gpu_runtime/session.rs`), the depth>1 partition-aware `pointwise_hist2` + subtraction trick, the fixed-point `Atomic<u64>` deterministic reduction, the Newton der2 leaf path, and a device-primitive library (scan / segmented-scan / stable radix sort / reduce / partitions / update_part_props / scatter / apply_leaf_delta / cindex / compression). Phase 12 flips five independent config families from `Ok(None)`→CPU-fallback to `Ok(Some(tree))`→device, each behind the same per-fit all-or-nothing gate, each oracle-locked at ε=1e-4 vs the Rust CPU path and timed on Kaggle CUDA as it lands.

Every family has an in-repo authority pair: the upstream CUDA design map (`CATBOOST_CUDA_KERNELS_DESIGN.md`, verified against the vendored `catboost-master/` tree) tells you the exact kernel/function shapes to port, and a Rust CPU reference in `crates/` supplies the ≤1e-5 oracle to check against. The single exception is **Region** (GPUT-18), which has *no* CPU reference — `validate_grow_policy` (`cb-train/src/boosting.rs`) rejects `EGrowPolicy::Region` and `TreeVariant` has no Region variant. Region is therefore a two-step lift (build CPU Region path FIRST, then device Region) and must be its own wave (D-03). The **CTR** wave (GPUT-10, D-05/D-06) is the highest-uncertainty and largest-surface item: the full `ctrs/kernel` port with on-device ordered-permutation residency, feature-combination CTRs, and the CTR→cindex binarization join.

**Primary recommendation:** Decompose into six waves in the locked roadmap sub-order — (W1) Depthwise/Lossguide device emission, (W2) **Region CPU→device** (own wave, largest single lift), (W3) Exact weighted-quantile leaves, (W4) bootstrap + random-strength, (W5) MVS, (W6) CTR device port. Extend `DeviceGrownTree` and the `GpuTrainSession` coverage gate incrementally per wave; keep every not-yet-signed-off family returning `Ok(None)`. Reuse Phase-10 segmented-sort/segmented-scan for Exact and MVS, the Phase-10 device RNG discipline (pin-seed + freeze-CPU-sample-in-fixture) for sampling, and the Phase-11 partition/histogram machinery for the grow policies and CTR-binarized columns. Boundary stays plain-host-structs only (no `cb-train` dep in `cb-backend`).

## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01 (all 5 families, one phase, sequenced waves):** Attempt all five families in a single Phase 12; planner decomposes into internal waves (grow policies → Exact → bootstrap → MVS → CTR). No formal 12.1–12.4 sub-phase split. Each family lands independently behind its own `Ok(None)` gate; partial completion is safe.
- **D-02 (retain roadmap ordering):** Keep the roadmap sub-order — front-loads tree-mechanics reusing Phase-11 partition machinery, ends on highest-uncertainty CTR.
- **D-03 (all three policies — Region requires CPU path FIRST):** Cover Depthwise, Lossguide **and Region** on device. Depthwise+Lossguide already have a CPU reference (`leaf_wise_grower`→`TreeVariant::NonSymmetric`). Region does NOT — it is a v1.0 escalated gap ("Region OUT", `boosting.rs` `validate_grow_policy` rejects it; no `TreeVariant::Region`). Region is a two-step lift: (a) build CPU Region path first (Region grower + `TRegionModel`-style **path** model variant + `AddRegion`/`ComputeRegionBins` apply, establishing the ≤1e-5 CPU oracle); then (b) device Region against it. **Region MUST be its own wave (CPU Region → device Region); it is the single largest item in Phase 12.**
- **D-04 (Depthwise/Lossguide emit via the existing CPU non-symmetric representation):** Device grows *structure + leaf values only*; extend `DeviceGrownTree` (`cb-compute/src/runtime.rs`) to carry the non-symmetric node graph (`step_nodes` `(left_diff,right_diff)` + `node_id_to_leaf_id`, mirroring `cb-train/src/tree.rs`) as **plain host structs** → existing `Model::from_trained` → `TreeVariant::NonSymmetric` → `cb-model/src/apply.rs`. Do NOT introduce a device-native non-symmetric tree type. CPU/host path stays byte-unchanged (D-04 no-regression), oracle-locked ≤1e-5.
- **Region representation note (§6.6):** Upstream keeps Region as its OWN model shape `TRegionModel` — an oblivious-like *path* walked while the computed split matches the stored direction (`takeEqualAndSplitDirection` packs one-hot in bit 0, expected direction in bit 1; leaf = depth reached where the path diverges), NOT a `TTreeNode[]` binary node graph. The CPU Region variant (D-03a) must model this path shape, not reuse `NonSymmetricTree`.
- **D-05 (full CTR including feature combinations):** Device-cover the complete CTR surface — ordered target-statistic CTRs, one-hot, AND tensor / multi-feature feature-combination CTRs. Not a numeric-single-feature subset.
- **D-06 (CTR value computation ON device — port `ctrs/kernel`):** Permutation-dependent target-statistics accumulation runs **on device**, resident across the permutation — port the upstream device CTR computation (`ctrs/kernel` + `batch_binarized_ctr_calcer`) so accumulated CTRs are binarized into additional cindex columns on device. NOT host-computed-then-uploaded.
- **D-07 (pin seed + freeze in fixture):** Match CPU sampling at ε=1e-4 by pinning the RNG seed/config, **freezing the exact CPU-reference sample in the oracle fixture**, and reproducing it bit-for-bit on device. Deterministic, checkable at the ε bar — not a looser distributional check.
- **D-08 (draw the keep-mask/weights ON device, resident):** RNG-drawn keep-mask / inverse-probability weights are computed **on device** each iteration from the pinned seed, keeping the derivatives resident. MVS's per-block optimal threshold on `sqrt(der²+λ)` + reweight is a device reduction over resident derivatives (§6.1 `mvs`). No per-tree host round-trip for the sample mask.
- **D-09 (device Exact path, distinct from Newton):** Exact weighted-quantile estimation (`needWeights = totalWeight·α`, binary search over per-bin weight prefix sums) runs on device for Quantile/MAE/MAPE-family objectives — upstream `EstimateExact` is fully GPU (`SegmentedRadixSort` → `SegmentedScanVector` → `ComputeWeightedQuantileWithBinarySearch`, §6.3/§5.6). Reuse Phase-10 device primitives (segmented sort/scan). Distinct from the Newton der2 path (GPUT-07, Phase 11).

### Standing constraints (carried from Phases 10/11 — design within, do not challenge)
- ε=1e-4 device-vs-Rust-CPU bar (GPUT-14 operative standing gate). CPU/host path byte-unchanged, oracle-locked ≤1e-5.
- Per-fit all-or-nothing (D-10-01): a fit is fully device-grown OR fully CPU-grown. Any family not yet passing Kaggle CUDA sign-off returns `Ok(None)`→host CPU grower.
- Kaggle CUDA is the sole correctness+speed authority (human-gated `--features cuda` notebook, reusing the Phase-10 harness). ROCm in-env is an optional compile/smoke convenience, NOT a gate.
- Standing BENCH-02 per-family speed check: each family timed on Kaggle CUDA **as it lands** (device vs host-CPU baseline AND vs official CatBoost GPU where comparable, warm-run/JIT-excluded, train-only). No family is "done" without a recorded CUDA speed check.
- Only the O(1) BestSplit descriptor + `2^depth`/per-leaf partition stats cross host↔device per level (D-05 residency).
- The final GPU coverage matrix (per-family correctness + speed) is documented (SC-5).
- **Landmines:** never add a `cb-train` dep to `cb-backend` (transcribe CPU refs inline); no `-inf` float literals in `#[cube]` kernels (use `f32::MIN` sentinel); deterministic reduction mandatory (fixed-point `Atomic<u64>` k=30 + fixed-order tree-reduce fallback, SPIKE-REDUCTION §5b); never read a `Handle` through a client other than the one that allocated it.
- CubeCL kernels use generic-float. Read `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md` before any kernel-design work; on a build error load `cubecl_error_guideline.md` FIRST.

### Claude's Discretion
- Internal wave decomposition/ordering beyond the pinned roadmap sub-order — planner refines (Region gets its own CPU→device wave per D-03).
- Exact leaf-estimation objective set (which of Quantile/MAE/MAPE/etc. get device fixtures) — resolve against the CPU Exact reference (`cb-compute/src/leaf.rs`) and §6.3.
- Device CTR ordered-permutation residency mechanics, the `ctrs/kernel` port shape, and the CTR→cindex binarization join — resolve against `batch_binarized_ctr_calcer.h` + `ctrs/kernel/`.
- MVS block size / threshold-search mechanics and the device RNG stream layout for pinned-seed reproduction — resolve against §6.1 `mvs`/`bootstrap`/`random` and `cb-train/src/bootstrap.rs`.

### Deferred Ideas (OUT OF SCOPE)
- Pairwise/ranking/multiclass/ordered/Langevin device families — Phase 13.
- Comprehensive aggregate speed benchmark + real named datasets (Higgs/Epsilon) — Phase 14 (BENCH-03); Phase 12 reuses the Phase-10 synthetic generator + per-family BENCH-02 checks.
- On-device border/quantile computation (`FastGpuBorders`) — out of scope milestone-wide; host CPU quantization stays the ≤1e-5 reference.
- Formal 12.1–12.4 sub-phase split — declined (D-01).
- CTR host-computed fallback interpretation of GPUT-10 — declined in favor of full device residency (D-06); noted as the lower-risk fallback if the `ctrs/kernel` port over-runs.

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| GPUT-18 | Depthwise, Lossguide, **and Region** grow policies on device — per-policy leaf selection (`ComputeOptimalSplitsRegion`/`ComputeOptimalSplit` + `SelectLeavesToSplit`) + region/non-symmetric apply (`AddRegion`/`ComputeNonSymmetricDecisionTreeBins`) ≤1e-4. | Wave 1 (D/L emission) + Wave 2 (Region CPU→device). Emission map §5.1–5.3/§5.4/§6.4/§6.6; CPU refs `tree.rs::leaf_wise_grower`, `model.rs::TreeVariant::NonSymmetric`; Region path shape §6.6 `AddRegion`. |
| GPUT-19 | Exact weighted-quantile leaf estimation on device (`exact_estimation`: `needWeights = totalWeight·α`, binary search over per-bin weight prefix sums) for Quantile/MAE/MAPE ≤1e-4, distinct from Newton (GPUT-07). | Wave 3. §6.3 `exact_estimation.{cu,cuh}` + §5.6; CPU ref `cb-compute/src/leaf.rs::Exact` (`exact_leaf_delta`); reuse Phase-10 segmented radix sort + segmented scan. |
| GPUT-09 | Bootstrap + random-strength sampling on device (Bayesian/Bernoulli/Poisson `bootstrap_type` + `RandomStrength` score jitter). | Wave 4. §6.1 `bootstrap.{cu,cuh}` + `random*.cuh`; CPU ref `cb-train/src/bootstrap.rs` (`EBootstrapType`, `bayesian_weight`, `set_sampled_control`) + `ComputeTargetVariance`→`ScoreStdDev` (§5.4). |
| GPUT-17 | MVS (Minimal Variance Sampling) bootstrap on device — per-block optimal threshold on `sqrt(der²+λ)` + inverse-probability reweight; CatBoost's *default* GPU sampler ≤1e-4. | Wave 5. §6.1 `mvs.{cu,cuh}`; CPU ref `cb-train/src/bootstrap.rs::mvs_sample_weights`/`calculate_threshold`/`single_probability` (`MVS_BLOCK_SIZE = 8192`); reuse Phase-10 radix sort + scan. |
| GPUT-10 | CTR / permutation-dependent categorical features train on device (ordered TS + one-hot + tensor/feature-combination CTRs), CTR value computation ON device. | Wave 6. §6.6 `ctrs/kernel/ctr_calcers.{cu,cuh}` + `batch_binarized_ctr_calcer.h`; CPU refs `cb-train/src/ctr/` (`online.rs`, `calc_ctr.rs`, `ctr_feature.rs`, `final_ctr.rs`), `cb-model/src/ctr_data.rs`. Highest uncertainty. |
| BENCH-02 | Standing per-family Kaggle CUDA speed check as each family lands (device vs host-CPU baseline; vs official CatBoost GPU where comparable). | Every wave ends with a CUDA speed measurement recorded into the SC-5 coverage matrix. Reuse the Phase-10 synthetic generator + harness. |

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Non-symmetric structure search (per-policy scoring/selection) | Device (`cb-backend` kernels) | Host (O(1) BestSplit reduce per level, D-05) | Same partition/histogram machinery as Phase 11; only the descriptor crosses. |
| Non-symmetric emission (node graph → model) | Host (`cb-train`→`cb-model`) | Device (produces structure+leaf-values only) | D-04: device returns plain host structs; existing `TreeVariant::NonSymmetric` builder + `apply.rs` own the model. |
| Region structure search + path apply | Device (`cb-backend`) | Host (CPU Region path is the oracle + fallback) | D-03: needs a CPU path built FIRST; Region is a distinct path model shape, not a node graph. |
| Region CPU grower + `TRegionModel` variant + apply | Host (`cb-train` + `cb-model`) | — | New v1.0-gap work; the ≤1e-5 oracle the device Region path checks against. |
| Exact weighted-quantile leaf estimation | Device (`cb-backend`, segmented sort→scan→binary search) | Host (Newton path already host-solved; Exact is fully GPU upstream) | D-09: reuse Phase-10 segmented primitives; distinct from host-solved Newton (§5.6). |
| Bootstrap weight draw + random-strength jitter | Device (`cb-backend` RNG kernels over resident weights/ders) | Host (pins seed/config, freezes CPU sample in fixture) | D-07/D-08: keep-mask/weights resident; `ScoreStdDev` folds into the device score calcer. |
| MVS threshold + reweight | Device (block-wise reduction over resident derivatives) | Host (pins seed; `lambda` = `GetLambda`) | D-08: MVS is inherently a device reduction; no host round-trip. |
| CTR target-statistic accumulation (ordered/tensor) | Device (`cb-backend` `ctrs` kernels, resident across permutation) | Host (borders/quantization stay CPU, ≤1e-5 ref) | D-06: on-device residency parity; binarized into extra cindex columns on device. |
| CTR→cindex binarization join | Device (`ctrs` output → compressed-index columns) | Host (border tables uploaded once per fit) | Extends the resident cindex the histogram loop already reads. |

## Standard Stack

This phase adds **no external crates**. It ports upstream CUDA kernels to CubeCL and reuses the in-repo device-primitive library. Per CLAUDE.md: Rust latest stable, CubeCL for kernels (generic-float), `thiserror`+`anyhow`, no `unwrap()` in production, latest crate versions if any are added.

### Core (in-repo substrate consumed as-is)
| Component | Location | Purpose | Why Standard |
|-----------|----------|---------|--------------|
| `Runtime` grow-tree seam | `cb-compute/src/runtime.rs` (`begin_device_training`/`grow_tree_on_device`/`end_device_training`) | The `Ok(None)`→CPU-fallback / `Ok(Some)`→device boundary each family flips | Established Phase-10 seam; default impls decline transparently. |
| `DeviceGrownTree` | `cb-compute/src/runtime.rs` (~L917) | The plain-host-struct result crossing the seam (oblivious today) | D-04 extends it with `step_nodes`/`node_id_to_leaf_id`; landmine-safe (no cubecl in the trait). |
| `GpuTrainSession` + coverage gate | `cb-backend/src/gpu_runtime/session.rs` (`begin`/`grow_one`) | Device residency + the `is_covered` gate returning `Ok(None)` for uncovered configs | Each family widens this gate; today rejects non-Plain/fold>1/unsupported-score/non-covered-loss. |
| Device primitive library | `cb-backend/src/kernels/` (`scan`, `segmented_scan`, `sort` [stable radix], `reduce`, `partitions`, `update_part_props`, `scatter`, `apply_leaf_delta`, `cindex`, `compression`) | Reusable `#[cube]` primitives + serial CPU self-oracle each | GPUT-16 no-CUB from-scratch lib; the reuse targets for Exact/MVS/CTR. |
| Partition-aware histogram + subtraction trick | `cb-backend/src/kernels/pointwise_hist.rs`, `grow_loop.rs` | depth>1 histogram machinery the grow policies + CTR columns reuse | Phase 11 shipped (plans 01–04). |
| Deterministic reduction | fixed-point `Atomic<u64>` k=30 + fixed-order tree-reduce fallback (SPIKE-REDUCTION §5b) | Mandatory for every device SUM at the ε bar | Phase 10-03 spike; gfx1100 has `Atomic<u64>` add, not f64. |
| der1/der2 seam | `cb-backend/src/gpu_runtime/der_seams.rs`, `gradient_gpu.rs` | Resident UN-weighted per-object derivatives; weight folded downstream by histogram scatter | Phase 7.2/11; MVS/Exact/sampling read these resident buffers. |

### Supporting (upstream authority to port from)
| Reference | Location | Purpose | When to Use |
|-----------|----------|---------|-------------|
| `CATBOOST_CUDA_KERNELS_DESIGN.md` | repo root | Verified kernel/function map for every family | Read the cited §§ before each wave (see per-family sections). |
| Vendored CUDA source | `catboost-master/catboost/cuda/` | The literal `.cu`/`.cuh` to transcribe | `methods/greedy_subsets_searcher/kernel/`, `cuda_util/kernel/`, `gpu_data/ctrs/`, `catboost/cuda/models/kernel/`. |
| Rust CPU references | `crates/cb-train/`, `crates/cb-compute/`, `crates/cb-model/` | The ≤1e-5 oracle each device family checks against | Transcribe inline into `cb-backend` (no `cb-train` dep). |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Extending `DeviceGrownTree` with a node graph | Device-native non-symmetric tree type | Rejected by D-04 — duplicates model-build surface, breaks the host-structs-only boundary. |
| On-device CTR residency (D-06) | Host-computed CTR, upload cindex | Declined (D-06); noted as the lower-risk fallback if the `ctrs/kernel` port over-runs the wave. |
| Pin-seed + freeze CPU sample (D-07) | Match upstream RNG stream / distributional check | Declined — freezing the exact CPU sample is deterministic and checkable at ε=1e-4. |

**Installation:** none — no new packages. (If a crate is ever added, use latest version per CLAUDE.md and gate it behind the package-legitimacy check.)

## Package Legitimacy Audit

Not applicable — this phase installs **no external packages**. All work is in-repo CubeCL kernels + Rust, reusing the existing workspace crates (`cb-backend`, `cb-compute`, `cb-train`, `cb-model`, `cb-core`) and the already-vendored `cubecl` dependency. No registry verification required.

## Architecture Patterns

### System Architecture Diagram

```
                          cb-train::train (host boosting loop)
                                   │
                 begin_device_training(loss, depth, policy, sampling,
                                       ctr-config, score_fn, cindex, weight …)
                                   │
                         ┌─────────┴──────────┐
              is_covered? │  GpuTrainSession   │  ← COVERAGE GATE (widened per family)
                          │  (cb-backend)      │     uncovered → Ok(false) → CPU path (D-04)
                          └─────────┬──────────┘
                                    │ covered
   ┌────────────────────────────────┴─────────────────────────────────────────┐
   │  Device-resident per-iteration grow (grow_tree_on_device → Ok(Some)):     │
   │                                                                            │
   │   der1/der2 (resident) ──► [SAMPLING]  bootstrap/MVS keep-mask+weights     │
   │                              on device from pinned seed (W4/W5)            │
   │            │                                                               │
   │            ▼                                                               │
   │   [CTR]  ordered/tensor target-stats accumulate on device across the      │
   │          permutation (W6) ──► binarize ──► extra cindex columns           │
   │            │                                                               │
   │            ▼                                                               │
   │   partition-aware histogram + subtraction trick (Phase 11)                │
   │            │                                                               │
   │            ▼                                                               │
   │   per-policy split scoring/selection:                                     │
   │     ComputeOptimalSplits (Symmetric) | ComputeOptimalSplit (Depthwise/    │
   │     Lossguide) | ComputeOptimalSplitsRegion (Region)   (W1/W2)            │
   │            │  ── O(1) BestSplit descriptor + 2^depth part-stats to host    │
   │            ▼    (only cross per level, D-05)                              │
   │   SelectLeavesToSplit (per GrowPolicy) ──► MakeSplit (repartition)        │
   │            │                                                               │
   │            ▼                                                               │
   │   leaf-value estimation:  Newton (Phase 11) | EXACT weighted-quantile     │
   │     (segmented sort → segmented scan → binary search)   (W3)              │
   └────────────────────────────────┬─────────────────────────────────────────┘
                                     │  DeviceGrownTree (PLAIN HOST STRUCTS)
                                     │  {splits | step_nodes,node_id_to_leaf_id |
                                     │   region path | leaf_values}
                                     ▼
        cb-model::Model::from_trained  ──►  TreeVariant::{Oblivious | NonSymmetric | Region}
                                     ▼
                       cb-model/src/apply.rs traversal (host)
```

### Component Responsibilities
| Concern | File(s) | Notes |
|---------|---------|-------|
| Seam + result struct | `cb-compute/src/runtime.rs` | Extend `DeviceGrownTree`; extend `begin_device_training` params for sampling/CTR/exact config. |
| Coverage gate | `cb-backend/src/gpu_runtime/session.rs` | Widen `begin`/`is_covered` per family; each new arm defaults `Ok(None)` until signed off. |
| Device kernels | `cb-backend/src/kernels/` (new: non-sym scoring, region path apply, exact quantile, bootstrap/MVS RNG, CTR) | Transcribe CPU refs inline; generic-float; deterministic reduction. |
| Non-sym/Region emission | `cb-train/src/tree.rs`, `cb-train/src/boosting.rs`, `cb-model/src/model.rs`, `cb-model/src/apply.rs`, `cb-model/src/json.rs` | D-04 reuses `NonSymmetric`; D-03a adds `Region` variant + grower + `validate_grow_policy` lift. |
| CPU oracles | `cb-compute/src/leaf.rs`, `cb-train/src/bootstrap.rs`, `cb-train/src/ctr/`, `cb-train/src/tree.rs` | The ≤1e-5 references frozen into ε=1e-4 fixtures. |

### Pattern 1: Family-gated `Ok(None)` coverage flip
**What:** Each family widens the `GpuTrainSession` coverage gate; until a family passes Kaggle CUDA sign-off its config arm returns `Ok(None)` → CPU grower. A fit is all-or-nothing (D-10-01).
**When to use:** Every wave. Never mix device-grown and CPU-grown trees in one model.
**Example (shape, from `session.rs`):**
```rust
// Source: cb-backend/src/gpu_runtime/session.rs (begin — Phase 10/11 gate)
if depth != 1 /* Phase 11 widened */ || !boosting_type_is_plain || fold_count != 1 {
    return Ok(None); // uncovered → CPU fallback (D-04)
}
// Phase 12 adds, per family: grow_policy arm, exact-leaf arm, bootstrap/MVS arm, ctr arm.
```

### Pattern 2: Structure-search generic over tree shape → one host build step → per-shape applier
**What:** Upstream `TGreedyTreeLikeStructureSearcher<TTreeModel>` searches structure generically, then one `BuildTreeLikeModel<TModel>` step emits `{TObliviousTreeModel | TRegionModel | TNonSymmetricTree}`, each with its own apply kernel (`AddObliviousTree`/`AddRegion`/`ComputeNonSymmetricDecisionTreeBins`). D-04 mirrors this exactly: device searches structure, returns plain host structs, host `Model::from_trained` builds the per-shape model.
**When to use:** W1 (NonSymmetric) and W2 (Region). Do NOT create device-native tree types.

### Pattern 3: Device SUM must go through the deterministic reduction
**What:** Every parity-critical device reduction uses the fixed-point `Atomic<u64>` k=30 accumulator + fixed-order tree-reduce fallback (SPIKE-REDUCTION §5b). Applies to CTR prefix sums, MVS block scans, Exact weight prefix sums, `ComputeTargetVariance`.
**When to use:** Every wave that introduces a new reduction. gfx1100 has `Atomic<u64>` add but no f64 atomic-add.

### Anti-Patterns to Avoid
- **Device-native non-symmetric/Region tree type** — violates D-04; the seam carries host structs only.
- **`cb-train` dependency in `cb-backend`** — feature unification breaks the rocm runtime; transcribe CPU refs inline (Phase 7.5 landmine).
- **`-inf` float literal in a `#[cube]` kernel** — `F::new(f32::NEG_INFINITY)` JIT-rejects on HIP/gfx1100; use a finite `f32::MIN` sentinel (Phase 7.5 landmine). Critical for the score `ARGMAX` sentinel `(-1, FLT_MAX)` in `ComputeOptimalSplit*`.
- **Non-deterministic float reduction** — any un-ordered device SUM breaks the ε bar; always the fixed-point accumulator.
- **Fusing per-dimension reductions** — keep byte-identical to the CPU scalar path (RESEARCH Pitfall 1, runtime.rs note).
- **Mixing device/CPU trees in one fit** — all-or-nothing (D-10-01).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Segmented radix sort for Exact/MVS | A new sort kernel | Phase-10 `cb-backend/src/kernels/sort.rs` (stable radix, keys+values) | Already device-resident + self-oracled; §6.3 EstimateExact reuses `SegmentedRadixSort`. |
| Segmented prefix sums (Exact weight prefix, CTR) | A new scan | Phase-10 `segmented_scan.rs` / `scan.rs` | Inclusive/exclusive f64/f32, flag-segmented, self-oracled. |
| Deterministic float reduction | A `+=` atomic-f64 accumulator | fixed-point `Atomic<u64>` k=30 + tree-reduce fallback | gfx1100 has no f64 atomic-add; determinism is mandatory at ε. |
| Non-symmetric model build/apply | A device tree type + applier | host `TreeVariant::NonSymmetric` + `apply.rs` (D-04) | Reuses shipped CPU model; boundary stays host-structs. |
| Partition/repartition after split | A new split-points engine | Phase-10/11 `partitions.rs` + `update_part_props.rs` + `scatter.rs` | Mirrors upstream `split_points.cu` sort→gather→update-parts. |
| RNG core | An ad-hoc PRNG | `cb_core::TFastRng64` (CPU ref) transcribed to a device MWC/LCG kernel matching `random_gen.cuh` | Parity hinges on exact seed advance; pin+freeze in fixture (D-07). |

**Key insight:** Phase 12 is a *transcription-and-wiring* phase. The device primitives, the residency spine, and the CPU oracles all exist; the risk is in faithfully reproducing upstream's ordered-CTR / MVS-threshold / Region-path mechanics bit-close, not in inventing algorithms.

## Runtime State Inventory

Not a rename/refactor/migration phase — **omitted**. (This is additive device-coverage work; no stored data, service config, OS-registered state, secrets, or build artifacts carry a renamed identifier.)

## Common Pitfalls

### Pitfall 1: Region has no CPU oracle — the device path has nothing to check against
**What goes wrong:** Jumping straight to a device Region kernel; there is no ≤1e-5 CPU reference, so the ε=1e-4 gate is uncheckable.
**Why it happens:** `validate_grow_policy` (`cb-train/src/boosting.rs`) rejects `EGrowPolicy::Region` ("Region OUT"); `TreeVariant` has no Region variant; `leaf_wise_grower` explicitly excludes Region.
**How to avoid:** D-03 — build the CPU Region path FIRST (grower + `TRegionModel`-style path variant + `AddRegion`/`ComputeRegionBins` apply + `json.rs` round-trip + lift the `validate_grow_policy` guard), establish the ≤1e-5 CPU oracle, THEN the device Region path. Own wave.
**Warning signs:** A Region device task with no corresponding CPU-Region task before it.

### Pitfall 2: Region modeled as a binary node graph
**What goes wrong:** Reusing `NonSymmetricTree` for Region produces wrong leaf assignments.
**Why it happens:** Region looks non-symmetric but upstream `TRegionModel` is a *path* (walk-while-direction-matches; leaf = depth reached at divergence), NOT a `TTreeNode[]` graph. `takeEqualAndSplitDirection` packs one-hot in bit 0, expected direction in bit 1 (§6.6 `AddRegionImpl`).
**How to avoid:** Model the path shape (`MaxLeaves = MaxDepth+1`, §5.4 `ComputeOptimalSplitsRegion`); a Region of depth d has exactly d+1 leaves along one path.
**Warning signs:** Region leaf count == `2^depth` instead of `depth+1`.

### Pitfall 3: Ordered-CTR leakage / segment-reset mishandled on device
**What goes wrong:** CTR values leak the target (use a document's own label) or reset segments wrong → CTR columns diverge from the CPU online prefix.
**Why it happens:** Ordered CTR must read the prefix statistic BEFORE incrementing (read-before-increment), reset at segment starts, and apply group-wise correction. Upstream encodes segment starts as a sign bit via `TIndexWrapper` and ORs prior-flag + `bins[i]!=bins[i-1]` + previous-layer-bin-change (`UpdateBordersMask`); `MakeGroupStarts`/`FillBinIndices`/`ApplyGroupwiseCtrFix` share one canonical CTR per category-within-group.
**How to avoid:** Port `ctrs/kernel/ctr_calcers.cu` faithfully; oracle against `cb-train/src/ctr/online.rs::online_ctr_prefix_binclf` (read-before-increment, object-order output). Freeze the CPU CTR column in the fixture.
**Warning signs:** CTR of the first doc in a segment is non-prior; identical categories in different groups get different CTRs unexpectedly.

### Pitfall 4: MVS threshold search / block layout diverges from CPU
**What goes wrong:** MVS keep-mask/weights differ, leaf values drift past ε.
**Why it happens:** MVS threshold is a per-block (`BlockSize = 8192`) arg-min search over sorted `sqrt(λ+der²)` candidates + their prefix sums; the CPU ref (`bootstrap.rs::calculate_threshold`) is a recursive `std::partition`-style estimator, while upstream device uses `cub::BlockRadixSort`+`BlockScan`+`GetThreshold`. `lambda = GetLambda(...)` (squared mean gradient magnitude on iter 0). `SampleRate` is an f32; `single_probability(derAbs, threshold) = derAbs>threshold ? 1 : derAbs/threshold`; weight = `1/p` w.p. `p` else 0 via `NextUniformF`.
**How to avoid:** Match the block size, the `sqrt(λ+der²)` candidate, and the threshold semantics; pin the seed and freeze the CPU sample (D-07). Deterministic reduction for the block scan.
**Warning signs:** Sampled-object count per block off from `SampleRate*blockSize`; run-to-run instability.

### Pitfall 5: RNG stream desync (bootstrap/random-strength/MVS)
**What goes wrong:** Device draws diverge from the continuous CPU `TFastRng64` stream.
**Why it happens:** The CPU stream is CONTINUOUS across iterations (never reseeded per tree) for Bernoulli/main; Bayesian uses per-block reseed `TFastRng64::from_seed(randSeed + blockIdx).advance(10)`; the Bayesian weight uses a base-2 log APPROXIMATION (~1e-5 sensitive), not exact `log2` (`bootstrap.rs`). Device `random_gen.cuh` is MWC (`AdvanceSeed`) / LCG (`AdvanceSeed32`) with `GenerateSeeds` decorrelation.
**How to avoid:** Reproduce the exact seed-advance and per-block-reseed layout; freeze the CPU sample in the fixture (D-07). Do NOT use `rand`.
**Warning signs:** Bayesian leaf values off by ~1e-4 (the log-approx tell).

### Pitfall 6: Exact quantile confused with Newton der2
**What goes wrong:** Applying the Newton diagonal step to a Quantile/MAE objective.
**Why it happens:** Both are "leaf estimation" but Exact is an order statistic (weighted sample quantile), not `g/(h+ε)`. Upstream `EstimateExact` is fully GPU; Newton is host-solved (§5.6).
**How to avoid:** Route quantile-family objectives to the device Exact path (D-09): `weightsWithTargets[i]=weights[i]/max(1,|target[i]|)`, `needWeights=totalWeight·α`, segmented sort per leaf-bin, weight prefix-sum, binary search (fixed iteration count) for the doc whose cumulative weight first reaches `needWeights`; then the α/δ adjustment (`CalculateWeightedTargetQuantile`, `DBL_EPSILON`) per `cb-compute/src/leaf.rs::exact_leaf_delta`.
**Warning signs:** Quantile-loss fixtures fail at the leaf-value stage while structure matches.

### Pitfall 7: `-inf` sentinel in the score `ARGMAX`
**What goes wrong:** The best-split reduction uses `(-1, FLT_MAX)` / negative-infinity sentinels; a literal `-inf` in a `#[cube]` kernel JIT-rejects on HIP/gfx1100 and is invisible to cpu/wgpu `cargo check`.
**How to avoid:** Use a finite `f32::MIN`/`f32::MAX` sentinel in kernels; keep `f64::NEG_INFINITY` only in host code. Run the rocm smoke suite in-env after any `#[cube]` change.

## Code Examples

### Exact weighted-quantile leaf (device pipeline shape)
```text
// Source: CATBOOST_CUDA_KERNELS_DESIGN.md §6.3 exact_estimation.{cu,cuh} + §5.6;
//         CPU oracle cb-compute/src/leaf.rs::exact_leaf_delta
per leaf-bin (one block):
  weightsWithTargets[i] = weights[i] / max(1, |target[i]|)   // ComputeWeightsWithTargets
  MakeEndOfBinsFlags: flags[begin]=1 per non-empty bin        // segment boundaries
  SegmentedRadixSort(targets, weights) within each leaf-bin   // Phase-10 sort.rs
  needWeights = blockReduce(weight over [begin,end)) * alpha   // ComputeNeedWeights (det. reduce)
  weightsPrefixSum = SegmentedScanVector(weights)             // Phase-10 segmented_scan.rs
  quantileDoc = binarySearch(weightsPrefixSum >= needWeights) // fixed iter count
  leaf.point = targets[quantileDoc]  (+ alpha/delta adjust)   // CalculateWeightedTargetQuantile
```

### Non-symmetric emission boundary (host structs)
```rust
// Source: cb-train/src/tree.rs (GrownTree.step_nodes / node_id_to_leaf_id) →
//         cb-compute/src/runtime.rs DeviceGrownTree (D-04 extension) →
//         cb-model/src/model.rs TreeVariant::NonSymmetric → apply.rs
// step_nodes[i] = (left_subtree_diff, right_subtree_diff); (0,0) == terminal leaf.
// node_id_to_leaf_id[i] = index into flat leaf_values (terminal nodes only).
// Device fills these as PLAIN HOST STRUCTS; no cubecl type crosses the seam.
```

### Region path apply (semantics to reproduce on CPU first, then device)
```text
// Source: §6.6 AddRegionImpl / ComputeRegionBinsImpl
bin = 0
for level in 0..depth:
    featureVal = (cindex[offset[level] + loadIdx] >> Shift) & Mask
    split = OneHot ? (featureVal == value[level]) : (featureVal > value[level])
    if split != expectedDirection[level]:  break        // path diverges
    bin += 1                                              // else advance along the region path
leaf = bin                                                // leaf = depth reached (MaxLeaves = MaxDepth+1)
```

### MVS single-object probability + weight (CPU ref to transcribe)
```rust
// Source: cb-train/src/bootstrap.rs (single_probability, mvs_sample_weights; MVS_BLOCK_SIZE = 8192)
// candidates[i] = sqrt(lambda + der_i^2); threshold via block arg-min search;
// p = single_probability(|der_i|, threshold);  weight_i = draw<p ? 1/p : 0  (NextUniformF).
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Device path = symmetric-tree / Newton-leaf / uniform-numeric only (Phase 11) | Five grow-mechanics families device-covered behind per-fit gate | Phase 12 | Closes the tree-mechanics coverage gap toward full parity. |
| Region unimplemented on CPU ("Region OUT" v1.0 gap) | CPU Region path built (grower + `TRegionModel` variant) then device Region | Phase 12 W2 | Pulls a v1.0-gap item into v1.1. |
| CTR host-computed (v1.0 CPU) | CTR target-stats accumulate ON device, resident across permutation (D-06) | Phase 12 W6 | Removes the per-fit CTR host round-trip; true upstream residency parity. |

**Deprecated/outdated:** none introduced. The CPU/host path is byte-unchanged (D-04) and remains the ≤1e-5 oracle.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Phase-10 `sort.rs` (stable radix) + `segmented_scan.rs` are sufficient primitives for both Exact and MVS without a new segmented-radix-sort variant. | Don't Hand-Roll / Exact | If a true *segmented* radix sort (vs whole-buffer) is needed and absent, W3/W5 gains a primitive sub-task. Confirm against `sort.rs` segmentation support at plan time. | [ASSUMED]
| A2 | The `GpuTrainSession::begin` coverage gate is the single extension point for all five families (one gate widened per wave). | Standard Stack / Pattern 1 | If sampling/CTR need a separate config surface on `begin_device_training`, the seam signature grows more than one param block. | [ASSUMED]
| A3 | Depth>1 device grow (Phase 11 plans 01–04) is wired into the session path the non-symmetric policies build on, despite `session.rs:154` still reading `depth != 1`. | Requirements / substrate | If depth>1 lives in a separate grow path not reachable from `session.begin`, W1/W2 must first route through it. Verify the Phase-11 wiring at plan time. | [ASSUMED]
| A4 | The Exact objective set for device fixtures is Quantile + MAE + MAPE (per D-09 wording); the CPU `leaf.rs::Exact` covers all three. | Requirements / GPUT-19 | If MAPE uses a different optimum path (`weightsWithTargets` divisor differs), it needs its own fixture. Resolve against `leaf.rs` + §6.3 (Claude's discretion). | [ASSUMED]
| A5 | Feature-combination (tensor) CTRs reuse the same `ctrs/kernel` device math as single-feature CTRs, differing only in the projection/hash pre-step (`ctr_feature.rs` combined-projection). | CTR wave | If tensor CTRs need a distinct device hash/merge kernel (`MergeBinsKernel` layering), W6 surface grows. Highest-uncertainty; resolve against `ctr_feature.rs` + `batch_binarized_ctr_calcer.h`. | [ASSUMED]
| A6 | BENCH-02 per-family speed checks reuse the Phase-10 synthetic generator + harness with no new benchmark infrastructure. | BENCH-02 | If a family needs a categorical-heavy synthetic dataset (CTR) the generator lacks, a small fixture-gen sub-task is added. | [ASSUMED]

## Open Questions (RESOLVED)

1. **Segmented radix sort granularity for Exact/MVS**
   - What we know: Phase-10 `sort.rs` provides a stable radix sort (keys+values); `segmented_scan.rs` provides flag-segmented prefix sums; upstream `EstimateExact` calls `SegmentedRadixSort`.
   - What's unclear: whether the existing sort supports per-segment (per-leaf-bin) sorting or only whole-buffer, and whether MVS's per-block sort can reuse it.
   - Recommendation: audit `sort.rs` segmentation at the start of W3; if absent, add a segmented-radix-sort primitive sub-task shared by W3+W5 (small, high-reuse).
   - **RESOLVED:** threaded into **Plan 05 Task 1** — the W3 start audits `sort.rs` segmentation and, if absent, adds the shared segmented-radix-sort primitive consumed by both Exact (Plan 05) and MVS (Plan 07).

2. **`begin_device_training` config surface for sampling + CTR + exact**
   - What we know: today it takes loss/depth/plain/fold/score_fn/cindex/weight/dims/lr/l2.
   - What's unclear: how bootstrap_type/sample_rate/mvs_lambda, the pinned seed, the exact-leaf flag, and the CTR config (types, priors, projections, borders) reach the session — as new params vs a config struct.
   - Recommendation: introduce a small host-typed `DeviceTrainConfig`-style struct (plain, no cubecl) rather than growing the arg list per wave; keep the seam landmine-safe.
   - **RESOLVED:** threaded into **Plan 01 Task 3** — a single plain host `DeviceTrainConfig` struct carries grow-policy/sampling/exact/CTR config across the seam; later waves widen config without growing the arg list.

3. **CTR device residency vs the fold/permutation structure**
   - What we know: ordered CTR is permutation-dependent, resident across the permutation (D-06); the covered device regime today is Plain / fold_count==1.
   - What's unclear: whether device CTR needs >1 permutation/fold (learning folds) or stays single-permutation like the covered device regime.
   - Recommendation: scope W6 to the single-permutation covered regime first (matches the current device gate); defer multi-fold CTR if it appears, behind `Ok(None)`.
   - **RESOLVED:** threaded into **Plan 08 Task 2** — device CTR is scoped to the single-permutation (Plain / fold_count==1) covered regime; any multi-fold CTR config stays `Ok(None)` (CPU fallback).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| CubeCL (workspace dep) | all device kernels | ✓ | vendored in workspace | — |
| ROCm / gfx1100 (in-env) | compile + smoke of `#[cube]` kernels | ✓ | ROCm 7.1, gfx1100/RDNA3 wave32 | — (smoke only; NOT a gate) |
| CUDA toolkit (Kaggle) | correctness + speed sign-off (sole authority) | ✗ (human-gated notebook) | Kaggle CUDA | none — human must run the `--features cuda` notebook |
| Rust latest stable | build | ✓ | — | — |
| `catboost-master/` vendored CUDA source | kernel transcription reference | ✓ | in-repo | — |
| CUBECL manual | mandatory pre-kernel read (AGENTS.md) | ✓ | `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md` | — |

**Missing dependencies with no fallback:** Kaggle CUDA sign-off is human-gated — each family's ε=1e-4 correctness + BENCH-02 speed must be discharged by a human running the notebook. The device path stays `Ok(None)` (CPU fallback) for any family not yet signed off. Do NOT fabricate CUDA oracle results (see Phase 11-05 PAUSED precedent).

**Missing dependencies with fallback:** none blocking — ROCm in-env covers compile/smoke; the CPU path always covers correctness via `Ok(None)`.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` (+ `approx` 0.5.x for float assertions) |
| Config file | none (cargo workspace); per-crate test targets |
| Quick run command | `cargo test -p cb-backend <family_module>` (per-family kernel + serial-oracle tests) |
| Full suite command | `cargo test --workspace` (per-crate to avoid the disk-pressure link failure — see MEMORY) |
| ROCm smoke | `cargo test -p cb-backend --features rocm <family>` in-env after any `#[cube]` change |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| GPUT-18 (D/L) | Device non-sym structure+leaves == CPU `leaf_wise_grower` ≤1e-4 | unit (serial self-oracle) | `cargo test -p cb-backend nonsym_grow` | ❌ Wave 0 (new test file) |
| GPUT-18 (Region CPU) | CPU Region grower+apply == frozen ≤1e-5 oracle | unit | `cargo test -p cb-train region_grow` + `-p cb-model region_apply` | ❌ Wave 0 |
| GPUT-18 (Region dev) | Device Region path == CPU Region ≤1e-4 | unit | `cargo test -p cb-backend region_device` | ❌ Wave 0 |
| GPUT-19 | Device Exact quantile leaf == `leaf.rs::exact_leaf_delta` ≤1e-4 | unit | `cargo test -p cb-backend exact_quantile` | ❌ Wave 0 |
| GPUT-09 | Device bootstrap/random-strength sample == frozen CPU sample (bit-for-bit / ≤1e-4 on leaves) | unit | `cargo test -p cb-backend bootstrap_device` | ❌ Wave 0 |
| GPUT-17 | Device MVS keep-mask/weights == `mvs_sample_weights` (pinned seed) | unit | `cargo test -p cb-backend mvs_device` | ❌ Wave 0 |
| GPUT-10 | Device CTR columns == CPU `online_ctr_prefix_binclf` / `calc_ctr` ≤1e-4; incl. tensor combos | unit | `cargo test -p cb-backend ctr_device` | ❌ Wave 0 |
| BENCH-02 | Per-family Kaggle CUDA speed recorded | manual (human notebook) | Kaggle `--features cuda` harness | manual-only (human-gated) |

### Sampling Rate
- **Per task commit:** `cargo test -p cb-backend <family_module>` (fast, the family's kernel + serial oracle).
- **Per wave merge:** `cargo test -p cb-backend` + `-p cb-train` + `-p cb-model` + `-p cb-compute` (per-crate; workspace link fails under disk pressure per MEMORY).
- **Phase gate:** all affected crates green in-env; each signed-off family's Kaggle CUDA ε=1e-4 + BENCH-02 recorded into the SC-5 coverage matrix; ROCm smoke green after `#[cube]` changes.

### Wave 0 Gaps
- [ ] `cb-backend/src/kernels/*_test.rs` for each new kernel (non-sym scoring, region apply, exact quantile, bootstrap/MVS RNG, CTR) — serial CPU self-oracle per file (source/test separation, CLAUDE.md).
- [ ] `cb-train`/`cb-model` Region test files (`region_grow_test.rs`, region apply + json round-trip) — the new CPU Region path.
- [ ] Per-family ε=1e-4 fixtures with the frozen CPU reference sample/CTR-column/quantile (D-07 discipline extended to every family).
- [ ] Categorical-heavy synthetic fixture for the CTR wave if the Phase-10 generator lacks cat features (verify at W6 plan time).

## Security Domain

`security_enforcement` is enabled (ASVS L1). This is a numerical compute library (GPU training kernels) with **no auth, session, network, or untrusted-input surface** introduced by this phase. The only applicable control is input-bounds validation on the data crossing the seam.

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes | Bounds-check quantized bins / leaf indices / partition offsets / CTR bin indices before device indexing; typed `CbError` (no `unwrap()`), never UB. Length-agreement checks already present on the seam (e.g. `grow_tree_on_device` approx-len check). |
| V6 Cryptography | no | — (the RNG is a statistical PRNG for sampling, not security-sensitive; parity, not entropy, is the requirement) |

### Known Threat Patterns for GPU-kernel Rust/CubeCL
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Out-of-bounds device index (bin/leaf/partition/CTR-bucket) | Tampering / DoS | Validate lengths + clamp indices at kernel entry; `checked_*` on host index math (Phase 11 `checked u16 diffs` precedent). |
| Integer overflow in `2^depth` / node-diff / CTR bucket sizing | Tampering | `check_depth` guard (`tree.rs`), checked arithmetic, `u32::MAX`/`u16` sentinels as already used. |
| Non-deterministic reduction → silent numeric corruption | Tampering (integrity) | Mandatory fixed-point `Atomic<u64>` deterministic reduction. |
| `unwrap()`/panic in production kernel launch | DoS | CLAUDE.md prohibits `unwrap()` in production; return typed `CbError`. |

## Sources

### Primary (HIGH confidence)
- `CATBOOST_CUDA_KERNELS_DESIGN.md` §5.1–5.7 (host orchestration), §6.1 (`bootstrap`/`mvs`/`random`), §6.3 (`exact_estimation`, `compute_scores`), §6.4 (`greedy_subsets_searcher` leaf-wise builder, `split_points`), §6.6 (`ctrs/kernel`, `models/kernel/add_model_value`) — verified against the vendored `catboost-master/` tree by the doc author.
- In-repo Rust substrate (read this session): `cb-compute/src/runtime.rs` (`DeviceGrownTree`, seam), `cb-backend/src/gpu_runtime/session.rs` (coverage gate), `cb-backend/src/kernels/` (primitive library, `grow_loop.rs`), `cb-train/src/tree.rs` (`GrownTree`, `leaf_wise_grower`, `LeafWisePolicy`), `cb-train/src/boosting.rs` (`EGrowPolicy`, `validate_grow_policy`), `cb-model/src/model.rs` (`TreeVariant`), `cb-compute/src/leaf.rs` (`Exact`), `cb-train/src/bootstrap.rs` (`EBootstrapType`, MVS), `cb-train/src/ctr/` (`online.rs`, `calc_ctr.rs`, `ctr_feature.rs`).
- `12-CONTEXT.md` (locked D-01..D-09) + `12-DISCUSSION-LOG.md`; `.planning/REQUIREMENTS.md` (GPUT-18/19/09/17/10, BENCH-02); `.planning/ROADMAP.md` (Phase 12 goal); project MEMORY (Phase 10/11 outcomes, landmines).

### Secondary (MEDIUM confidence)
- Phase 10 `SPIKE-REDUCTION.md` referenced via CONTEXT + MEMORY (deterministic reduction decision); not re-read line-by-line this session.

### Tertiary (LOW confidence)
- Assumptions A1–A6 (segmented-sort granularity, session config surface, depth>1 wiring, Exact objective set, tensor-CTR kernel sharing, benchmark reuse) — flagged for plan-time verification against the cited files.

## Metadata

**Confidence breakdown:**
- Emission architecture / non-sym / Region shape: HIGH — design doc + CPU `GrownTree`/`TreeVariant` + explicit "Region OUT" gap all cross-confirm.
- Exact / sampling / MVS mechanics: HIGH — §6.1/§6.3 map to concrete CPU refs (`leaf.rs`, `bootstrap.rs`) with named functions and constants (`MVS_BLOCK_SIZE=8192`, `needWeights=totalWeight·α`).
- CTR device port: MEDIUM — the `ctrs/kernel` surface and CPU refs are located and mapped, but on-device ordered-permutation residency + tensor-combo kernel sharing + CTR→cindex join carry genuine mechanics risk (A5, Open Q3); this is the flagged highest-uncertainty wave.
- Primitive reuse map: MEDIUM — primitives exist; segmentation granularity (A1, Open Q1) needs a plan-time audit.

**Research date:** 2026-07-03
**Valid until:** ~2026-08-02 (stable — in-repo design authority + CPU references; refresh only if the Phase-11 substrate or seam signature changes).
