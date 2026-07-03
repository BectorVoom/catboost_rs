# Phase 12: Grow-Policy, Leaf-Method, Sampling & Categorical Device Coverage - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-03
**Phase:** 12-grow-policy-leaf-method-sampling-categorical-device-coverage
**Areas discussed:** Family scope / MVP cut, CTR device coverage depth, Sampling parity strategy, Non-symmetric policy scope

---

## Family scope / MVP cut

### Ambition

| Option | Description | Selected |
|--------|-------------|----------|
| All 5 families | Depthwise/Lossguide/Region + Exact + bootstrap + MVS + CTR this phase; each ships independently behind `Ok(None)`. | ✓ |
| MVP: bootstrap + CTR | Roadmap's suggested cut — highest-return + headline categorical first; defer the rest. | |
| Core-3: policies+Exact+MVS | Tree-mechanics core; defer bootstrap + CTR. | |

**User's choice:** All 5 families
**Notes:** Phase 11 substrate (depth>1 histograms + partitions) is shipped (plans 01–04), only paused on the human-gated Kaggle CUDA oracle — substrate is ready to build on.

### Structure

| Option | Description | Selected |
|--------|-------------|----------|
| One phase, sequenced waves | Keep Phase 12 whole; planner decomposes into internal waves. | ✓ |
| Formal sub-phases 12.1–12.4 | Split into separate phase directories per family. | |
| You decide | Planner chooses decomposition. | |

**User's choice:** One phase, sequenced waves

### Ordering

| Option | Description | Selected |
|--------|-------------|----------|
| Keep roadmap order | policies → Exact → bootstrap → MVS → CTR. | ✓ |
| Front-load CTR | Tackle highest-uncertainty family first. | |
| You decide | Planner sets ordering. | |

**User's choice:** Keep roadmap order

---

## CTR device coverage depth

### CTR scope

| Option | Description | Selected |
|--------|-------------|----------|
| Single-feature ordered TS first | Core ordered TS on single cat features; defer one-hot + combinations. | |
| Full CTR incl. combinations | Ordered TS + one-hot + tensor/feature-combination CTRs all on device. | ✓ |
| You decide | Research recommends scope. | |

**User's choice:** Full CTR incl. combinations

### CTR compute site

| Option | Description | Selected |
|--------|-------------|----------|
| CTRs on device (port ctrs/kernel) | Target-stats accumulate on-device across the permutation, resident; port `ctrs/kernel` + `batch_binarized_ctr_calcer`. Largest surface, true parity. | ✓ |
| CTRs host, tree grows on device | Reuse v1.0 CPU CTR host-side, feed cindex to device histogram. Less residency, less risk. | |
| You decide | Research weighs residency vs port risk. | |

**User's choice:** CTRs on device (port ctrs/kernel)
**Notes:** Consistent with the milestone's speed/residency goal despite being the highest-uncertainty sub-task; ordered last per the retained roadmap sub-order.

---

## Sampling parity strategy

### Parity approach

| Option | Description | Selected |
|--------|-------------|----------|
| Pin seed + freeze in fixture | Mirror Phase 11; pin seed/config, freeze CPU-reference sample in fixture, reproduce bit-for-bit on device. | ✓ |
| Match upstream RNG stream | Reproduce CatBoost's exact per-element draw order on device. | |
| Distributional check | Verify convergence statistically (looser). | |

**User's choice:** Pin seed + freeze in fixture

### Draw site

| Option | Description | Selected |
|--------|-------------|----------|
| On device (RNG + mask resident) | Draw mask/weights on device each iteration from the pinned seed; MVS threshold+reweight is a device reduction anyway. Preserves no-readback residency. | ✓ |
| Host-computed mask, uploaded | Compute mask/weights on host, upload per iteration. Simplest parity, cuts against speed goal. | |
| You decide | Research resolves per-sampler. | |

**User's choice:** On device (RNG + mask resident)

---

## Non-symmetric policy scope

### Policies (initial answer, later revised after investigation)

| Option | Description | Selected |
|--------|-------------|----------|
| All three | Depthwise + Lossguide + Region on device. | (initial) |
| Depthwise+Lossguide first | Land the two leaf-wise-grower policies; defer Region. | |
| You decide | Research assesses Region apply surface. | |

**Initial answer:** All three — but the emission follow-up was paused by the user to clarify, which triggered an investigation of `CATBOOST_CUDA_KERNELS_DESIGN.md` + the Rust code.

**Investigation findings (grounded in code + design doc):**
- Upstream device structure searcher is generic over shape; one host `BuildTreeLikeModel<TModel>` step emits `TObliviousTreeModel` / `TRegionModel` / `TNonSymmetricTree`, each with its own apply kernel (§5.1–5.3, §6.6).
- Upstream **Region is its own model shape** (`TRegionModel`, a *path* walked until it diverges — leaf = depth reached), NOT a non-symmetric node graph.
- Our Rust CPU path has Depthwise/Lossguide (`leaf_wise_grower` → `TreeVariant::NonSymmetric`) but **Region is UNIMPLEMENTED** — `validate_grow_policy` (`boosting.rs:1332`) rejects `EGrowPolicy::Region` ("Region OUT" v1.0 escalated gap); `TreeVariant` has no Region variant.
- Consequence: the ε=1e-4-vs-CPU gate has **no CPU Region path** to oracle against. Covering Region requires building the CPU Region path first (grower + `TRegionModel`-style variant + `AddRegion`/`ComputeRegionBins` apply).

### Region scope (re-asked with findings)

| Option | Description | Selected |
|--------|-------------|----------|
| Defer Region; ship Depthwise+Lossguide | Cover the two policies with a CPU oracle; note GPUT-18 as 2-of-3; Region as a separate work item. | |
| All three: build CPU Region first | Build CPU Region (grower + variant + apply) THEN device Region, closing the v1.0 gap inside this phase. | ✓ |
| You decide | Research assesses the CPU-Region lift. | |

**User's choice:** All three: build CPU Region first
**Notes:** Flagged as the single largest lift in Phase 12; pulls a v1.0-gap item into v1.1. Planner MUST treat Region as its own wave (CPU Region → device Region).

### Non-symmetric emission (Depthwise/Lossguide)

| Option | Description | Selected |
|--------|-------------|----------|
| Reuse CPU non-sym representation | Device computes structure + leaf values; extend `DeviceGrownTree` with the node graph (`step_nodes`/`node_id_to_leaf_id`) as host structs → existing `Model::from_trained` → `TreeVariant::NonSymmetric`. Mirrors upstream `BuildTreeLikeModel`. | ✓ |
| New device-native tree type | Separate device non-sym type + apply/emission path. Duplicates model-build surface. | |

**User's choice:** Reuse CPU non-sym representation
**Notes:** Boundary stays host-structs-only (landmine-safe); exactly mirrors upstream's generic-structure → per-shape-model architecture.

---

## Claude's Discretion

- Internal wave decomposition beyond the pinned roadmap sub-order (Region gets its own CPU→device wave).
- Exact leaf-estimation objective set (which of Quantile/MAE/MAPE get device fixtures).
- Device CTR ordered-permutation residency mechanics, `ctrs/kernel` port shape, CTR→cindex binarization join.
- MVS block size / threshold-search mechanics and the device RNG stream layout for pinned-seed reproduction.

## Deferred Ideas

- Pairwise/ranking/multiclass/ordered/Langevin device families — Phase 13.
- Comprehensive aggregate speed benchmark + real named datasets (Higgs/Epsilon) — Phase 14 (BENCH-03).
- On-device border/quantile computation (`FastGpuBorders`) — out of scope milestone-wide.
- Formal 12.1–12.4 sub-phase split — considered, declined (one phase with waves).
- CTR host-computed fallback interpretation of GPUT-10 — considered, declined in favor of full device residency; noted as lower-risk fallback if the `ctrs/kernel` port over-runs.
