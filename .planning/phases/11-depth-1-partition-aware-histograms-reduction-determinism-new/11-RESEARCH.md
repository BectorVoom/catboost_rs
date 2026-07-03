# Phase 11: Depth>1 Partition-Aware Histograms + Reduction Determinism + Newton Der2 — Research

**Researched:** 2026-07-03
**Domain:** GPU device-resident gradient-boosting tree growth (CubeCL) — partition-aware histograms, subtraction trick, deterministic reduction, Newton leaf estimation
**Confidence:** HIGH on the CPU-reference behaviors and existing seams (read directly from the tree); MEDIUM on the exact device channel-layout/addressing (grounded in the upstream design doc, to be pinned in a fixture); the reduction strategy is LOCKED, not researched.

## Summary

This phase extends the Phase-10 depth-1 device grow loop to real depth-6 oblivious trees for both RMSE and Logloss, holding ε=1e-4 vs the Rust CPU path on Kaggle CUDA. Three deliverables interlock: (1) a **partition-aware `pointwise_hist2`** (`fullPass=false`) keyed by `leaf_of[obj]` into `2^level` slots plus the **histogram subtraction trick** (compute the smaller sibling directly, derive the larger by `parent − smaller`); (2) **reduction determinism** — already decided by the Phase-10 spike (fixed-point `Atomic<u64>` accumulator, k=30, with the fixed-order f64 tree-reduce as the capability fallback) — consumed as step 0, NOT re-opened; (3) **Newton der2 leaf estimation** for the Logloss default, reusing the Phase 7.2 der1/der2 handles and the `apply_leaf_delta` kernel.

The single most consequential research finding is about **Newton leaf estimation** (research flag #3): the Rust CPU oracle implements a **single closed-form Newton step** — `newton_leaf_delta = Σder1 / (−Σder2 + scaledL2)` — with `leaf_estimation_iterations` **pinned to 1 across every existing fixture** and **NO multi-iteration refinement loop and NO backtracking/line-search** (`AnyImprovement`/`step_estimator`). Upstream CUDA *does* have an iterative `TNewtonLikeWalker` with backtracking, but that is not what the oracle computes. The device Newton path must therefore mirror the single closed-form step — which means adding a **Σder2 reduce channel** to the partition-stats and swapping the leaf-value formula from `calc_average` (today's gradient leaf) to `newton_leaf_delta`. D-01's "leaf_estimation_iterations steps … recompute der per step" only becomes real if the fixture pins iterations>1; that path does not exist in the CPU reference today and would have to be built there first. **This is the top item to pin with the user before planning locks.**

**Primary recommendation:** Decompose into three sub-waves (histograms/partition first, then Newton der2, with the reduction primitive consumed as step 0). Pin `leaf_estimation_iterations=1` for the correctness fixture (consistent with all existing fixtures), making the device Newton a single closed-form step: add a Σ(der2·weight) partition channel, reuse the Phase 7.2 `LoglossHessian` der2 handle and one `apply_leaf_delta`, and score under the existing Cosine default. Extend the existing per-level loop in `grow_oblivious_tree_into` at the score step only; the partition-split / partition-update / leaf-of machinery already exists from Phase 10.

## User Constraints (from CONTEXT.md)

### Locked Decisions

**Newton der2 leaf estimation (GPUT-07)**
- **D-01 (fully device-resident refinement loop):** The Logloss Newton refinement — `leaf_estimation_iterations` steps, each recomputing der1/der2 at the current approx and updating leaf values — runs **fully on device**. Reuse the Phase 7.2 der1/der2 handles + the `apply_leaf_delta` kernel; recompute ders per step on-device; **no per-iteration readback**. (RMSE's der2 is the constant weight, so its Newton step is effectively single-step/trivial — the multi-step refinement is the Logloss path.)
- **D-02 (pin iteration count in the fixture):** Pin `leaf_estimation_iterations` from the model config and **freeze it in the CPU-reference fixture** so the device refinement matches the CPU reference exactly at the ε=1e-4 bar.

**Depth-6 correctness fixture + CUDA speed workload**
- **D-03 (reuse the Phase-10 synthetic generator):** Extend the Phase-10 seeded synthetic generator to **depth-6 RMSE + Logloss configs**; it produces BOTH the ≤1e-4 correctness fixture AND the large-n CUDA speed workload. Real named datasets (Higgs/Epsilon) stay deferred to Phase 14 (BENCH-03).

**Subtraction trick + histogram memory residency (GPUT-05)**
- **D-04 (smaller-sibling-direct + parent-resident subtraction):** Compute the **smaller partition's histogram directly**, derive the larger sibling by **subtracting from the parent's resident histogram** (§6.4). Keep only **parent-level** histograms resident, not all levels.

**ε=1e-4 verification across the boosting run (GPUT-06 / SC-3 / SC-5)**
- **D-05 (final ε gate + per-tree diagnostic):** Gate on **final-prediction ε=1e-4** across the full run (blocking) AND instrument a **per-tree split-agreement + run-to-run spread diagnostic** in the Kaggle oracle.

### Claude's Discretion
- **Sub-wave decomposition/ordering** — ROADMAP suggests: depth>1 histograms → reduction determinism → Newton der2. Planner refines (reduction spike winner is step 0 / already landed as the reduce primitive).
- **Newton leaf-estimation backtracking** — research flag: confirm whether the CPU reference uses backtracking at the pinned config; if so mirror on device. → **RESOLVED below (it does not; iterations pinned to 1, no backtracking).**
- **Exact channel layout** of the partition-aware `pointwise_hist2` (der1 + weight, plus der2 for Newton), the `2^level` slot addressing, and the contiguous `TDataPartition` reorder mechanics — resolve against §6.3/§6.4 and the Phase-10 primitives. → **Addressed below (MEDIUM confidence; pin in fixture).**

### Deferred Ideas (OUT OF SCOPE)
- **Real named datasets (Higgs/Epsilon)** — deferred to Phase 14 (BENCH-03). Phase 11 uses the synthetic generator.
- **Non-symmetric grow policies (Depthwise/Lossguide/Region), Exact weighted-quantile leaf estimation, bootstrap/MVS sampling, CTR/categoricals** — Phase 12.
- **Pairwise/ranking/multiclass/ordered/Langevin device families** — Phase 13.
- **On-device border/quantile computation (`FastGpuBorders`)** — out of scope milestone-wide; host CPU quantization stays the ≤1e-5 reference.

### Standing Landmines (hard constraints — do NOT re-derive)
- Never add a `cb-train` dependency to `cb-backend` (Cargo feature unification breaks the rocm runtime) — transcribe CPU refs inline.
- No `-inf` float literals in `#[cube]` kernels — use `f32::MIN` sentinel (known CubeCL HIP `-inf` landmine: `F::new(f32::NEG_INFINITY)` emits `double(-inf)` → HIP/gfx1100 JIT reject).
- Deterministic reduction mandatory — CUDA `atomicAdd` ordering is still non-deterministic; gfx1100 lacks f64 atomic-add for the ROCm smoke path.
- Never read a `Handle` through a client other than the one that allocated it.
- All GPU oracles (correctness AND speed) authoritative on **Kaggle CUDA** (human-gated). ROCm in-env is an optional compile/smoke convenience, NOT a gate.

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| GPUT-05 | Partition-aware histograms (`fullPass=false`) keyed by leaf, contiguous partition reorder, and the subtraction trick support depth>1 oblivious trees on device. | Channel layout + `2^level` addressing + subtraction lifecycle grounded in §1.4/§6.3/§6.4 (Architecture Patterns). Extends the existing per-level loop score step in `grow_oblivious_tree_into`. |
| GPUT-06 | A reduction-determinism strategy keeps device histogram/score reductions within ε=1e-4 across hundreds of trees. | LOCKED: fixed-point `Atomic<u64>` accumulator (k=30) + fixed-order tree-reduce fallback (SPIKE-REDUCTION §5b). Consumed as step 0; per-tree spread diagnostic (D-05) evidences no compounding drift. |
| GPUT-07 | Newton der2 leaf estimation runs on device (Logloss default). | Single closed-form `newton_leaf_delta` reusing Phase 7.2 `LoglossHessian` der2 handle + a Σder2 partition channel + one `apply_leaf_delta` (Newton section). iterations=1 (matches CPU oracle). |
| GPUT-14 | Every device-covered case holds ε=1e-4 vs Rust CPU path; CPU/host paths byte-unchanged (D-04). | Operative standing gate from this phase onward. Self-oracle + Kaggle final-ε gate. |
| BENCH-02 | Standing per-phase Kaggle CUDA speed check (device vs host-CPU baseline, vs official CatBoost GPU). | Depth-6 RMSE + Logloss timed on the D-03 synthetic workload; warm-run/JIT-excluded, train-only. |

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Partition-aware histogram fill (`fullPass=false`) | Device (`cb-backend` kernels/`GpuTrainSession`) | — | Bulk per-object streaming; must stay device-resident (D-05). |
| Subtraction trick (parent − smaller) | Device (`histogram_utils`-equivalent `#[cube]`) | — | O(bins) device op on resident parent histogram (D-04). |
| Split scoring / BestSplit argmin | Device score kernel → O(1) descriptor to host | Host (integer split decision only) | Only the O(1) BestSplit crosses per level (D-05). |
| Document repartition / `leaf_of` update | Device (`partition_split` — exists) | — | Forward-bit doc routing, already Phase-10 device-resident. |
| Per-partition stat reduce (Σder1, Σweight, **Σder2**) | Device (`partition_update` — extend +1 channel) | Host (reads 2^depth stats) | Deterministic reduce; only `2^depth` stats cross (D-05). |
| Deterministic reduction | Device (fixed-point `Atomic<u64>`, tree-reduce fallback) | — | LOCKED by SPIKE-REDUCTION §5b. |
| Newton leaf value | Device (der recompute + `apply_leaf_delta`) | Host (formula on read-back stats for oracle) | Reuse Phase 7.2 der2 handles; no per-iteration readback (D-01). |
| Correctness/speed oracle authority | Host (Kaggle CUDA, human-gated) | ROCm in-env smoke (non-gating) | CUDA is the sole authoritative GPU oracle. |
| CPU reference (the ≤1e-5 oracle) | `cb-compute` / `cb-train` (byte-unchanged) | — | D-04 no-regression; the parity target. |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `cubecl` | 0.10.0 (workspace-pinned) | GPU kernels (cuda/rocm/wgpu/cpu, compile-time feature select) | Project-mandated GPU stack (CLAUDE.md); all existing kernels use it. `[VERIFIED: Cargo.toml:38]` |
| `cb-backend` (internal) | path | Owns all `#[cube]` kernels, `GpuTrainSession`, der seams, grow loop | The device tier; extended in-place this phase. |
| `cb-compute` (internal) | path | CubeCL-free CPU reference + leaf/loss/histogram math (the oracle) | `newton_leaf_delta`, `reduce_leaf_der2`, `calc_average`, `scale_l2_reg` transcribed here. |
| `cb-train` (internal) | path | Boosting driver, `compute_leaf_deltas`, `leaf_estimation_iterations` | The end-to-end CPU training reference. **Never a `cb-backend` dep.** |

### Supporting
| Asset | Location | Purpose | When to Use |
|-------|----------|---------|-------------|
| Phase-10 device primitive library | `crates/cb-backend/src/kernels/{scan,segmented_scan,reduce,partitions,sort}.rs` | scan / segmented-scan / reduce-by-key / partition-update / radix sort / one-bit reorder | Partition reorder + histogram fold prefix-scan reuse these directly. |
| `launch_pointwise_hist2` / `_handle` | `gpu_runtime` (via `kernels/pointwise_hist.rs`) | 2-channel device histogram (whole-dataset today) | Extend to partition-aware `fullPass=false`. |
| `launch_partition_split_into` | `gpu_runtime/mod.rs:~1444` | Forward-bit `leaf_of` doc routing (level→bit) | Already depth-agnostic; reused per level unchanged. |
| `launch_partition_update_into` | `gpu_runtime/mod.rs:~1539` | Per-partition Σder1/Σweight reduce over `2^depth` leaves | **Extend to a 3rd channel Σ(der2·weight)** for Newton. |
| `read_part_stats_f64` | `gpu_runtime/mod.rs:1391` | The single `2^depth`-stats read-back | Reused; buffer widens to `2^depth * 3`. |
| Phase 7.2 der seams | `gpu_runtime/der_seams.rs` | `DerBinaryKernel::{RmseGradient,LoglossGradient}`, `DerUnaryKernel::LoglossHessian`, `const_der_handle` (RMSE der2=−1), no-readback `*_handle` | Newton der1/der2 recompute on device. |
| `apply_leaf_delta_kernel` | `gpu_runtime` (imported mod.rs:62) | In-place device approx update | Newton refinement update stays device-resident (D-01). |
| Fixed-point `Atomic<u64>` accumulator | `kernels.rs` `block_reduce_fixedpoint_kernel`, `REDUCE_FIXEDPOINT_SCALE_F64` (k=30) | Deterministic histogram/score reduce | LOCKED step-0 accumulator (SPIKE-REDUCTION §5b). |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Smaller-sibling-direct + subtraction (D-04) | Always-materialize-both children | ~2× histogram work; rejected — D-04 locked, memory-lean is a first-class constraint at depth 6 (64 leaves × features × bins × channels). |
| Fixed-point `Atomic<u64>` accumulator | f64 `Atomic::fetch_add` | Non-deterministic add ORDER (fails ε across hundreds of trees) AND gfx1100 lacks f64 atomic-add; rejected by SPIKE-REDUCTION. |
| Single closed-form Newton (iterations=1) | Iterative `TNewtonLikeWalker` + backtracking | The CPU oracle does not implement the iterative walker; matching it would require building it in `cb-compute` first. See Open Questions. |

**Installation:** No new external crates. This phase adds `#[cube]` kernels and extends existing launch functions inside `cb-backend`. `cubecl` version-currency against crates.io could not be confirmed in-session (registry API unavailable); the workspace pins `0.10.0` and CLAUDE.md mandates "always latest" — a version bump, if any, is orthogonal to this phase and should be a separate change. `[ASSUMED: 0.10.0 currency]`

## Package Legitimacy Audit

> No external packages are added or removed in this phase. It extends first-party `cb-backend` kernels using the already-vendored `cubecl` dependency.

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| `cubecl` | crates.io | established | (registry unavailable in-session) | github.com/tracel-ai/cubecl | OK (pre-existing, workspace-pinned) | Approved — no change |

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```
                 per boosting iteration (device-resident, GpuTrainSession)
 approx[i] ──▶ targets/kernel der seam ──▶ der1[i], der2[i]  (Phase 7.2 handles, no readback)
   (device)      (RmseGradient / LoglossGradient / LoglossHessian / const_der -1)
                                   │
      ┌────────────────────────────┴──────────── per level  L = 0 .. depth-1 ────────────┐
      │                                                                                   │
      │  leaf_of[obj] (2^L slots)                                                          │
      │        │                                                                          │
      │        ▼                                                                          │
      │  (1) partition-aware pointwise_hist2  (fullPass = L>0)                             │
      │        keyed by leaf_of[obj] → bin(feature) → channel                             │
      │        compute ONLY the smaller sibling per pair (D-04)                           │
      │        deterministic accumulate: fixed-point Atomic<u64> (k=30)                    │
      │        │                                                                          │
      │        ▼                                                                          │
      │  (2) subtraction trick:  hist[bigChild] = hist[parent] − hist[smallChild]         │
      │        weight/hessian channel (statId 0) clamped to max(0)   (parent-resident)    │
      │        │                                                                          │
      │        ▼                                                                          │
      │  (3) per-feature fold prefix-scan (ScanHistograms → ≤-border cumulative sums)      │
      │        │                                                                          │
      │        ▼                                                                          │
      │  (4) score every (feature,bin) per active leaf → argmin gain                       │
      │        left = hist[leaf,binFeature];  right = partStats[leaf] − left               │
      │        │  O(1) BestSplit descriptor ──────────────▶ host integer split decision    │
      │        ▼                                                                          │
      │  (5) partition_split: forward-bit route obj → leaf_of |= (bit << L)  (device)      │
      └───────────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼  (final level)
   partition_update: per-leaf reduce  Σder1, Σweight, Σ(der2·w)   → 2^depth × 3 stats
                                   │  (ONLY 2^depth stats cross host↔device)
                                   ▼
   leaf value = newton_leaf_delta(Σder1, Σ(der2·w), scaledL2) = Σder1 / (−Σder2 + scaledL2)
        (RMSE: der2 = −1 ⇒ −Σder2 = Σweight ⇒ reduces to calc_average / gradient leaf)
        device path: apply_leaf_delta updates approx in place (D-01, no per-iter readback)
```

### Recommended Project Structure
```
crates/cb-backend/src/
├── kernels/
│   ├── grow_loop.rs         # extend: per-level partition-aware score step (self-oracle here)
│   ├── pointwise_hist.rs    # extend: fullPass=false / leaf-keyed hist2 variant
│   ├── partitions.rs        # reuse: partition offset/size update
│   ├── scan.rs              # reuse: two-level (multi-block) prefix scan for fold-scan/reorder
│   └── reduce.rs            # reuse: fixed-point Atomic<u64> deterministic accumulator
└── gpu_runtime/
    ├── mod.rs               # extend: grow_oblivious_tree_into (remove depth>1 reject), +der2 channel
    └── der_seams.rs         # reuse: LoglossHessian der2 handle, const_der (RMSE), apply_leaf_delta
crates/cb-compute/src/
├── histogram.rs             # oracle: reduce_leaf_stats / reduce_leaf_der2 (leaf-keyed scatter)
└── leaf.rs                  # oracle: newton_leaf_delta, calc_average, scale_l2_reg
```

### Pattern 1: Partition-aware histogram channel layout & addressing
**What:** The histogram is a flat `float*` tensor. Upstream leaf-wise indexing (§6.4) is
`leafId * binFeatureCount * statCount + statId * binFeatureCount + binFeatureIndex`; pointwise
(§6.3) is `HIST_COUNT` floats per `(part, bin-feature)` with `HIST_COUNT=2`. **`statId == 0` is
the weight/count channel; `statId >= 1` is the gradient sum.** For a **second-order (Newton/Cosine)
path the channel-0 "weight" slot carries the sum of hessians (der2), not raw counts** — this is the
key channel-layout question flagged in CONTEXT.md. Keeping channel-0 = Σder2 makes split scoring
(NewtonCosine denominator) and leaf estimation second-order-consistent.
**When to use:** Every depth>1 level. `2^level` leaf slots; each object routed by `leaf_of[obj]`.
**Grounding:** §6.3 (`pointwise_hist2`, `HIST_COUNT=2`), §6.4 (leaf-wise tensor indexing), glossary row "Histogram tensor". `[CITED: CATBOOST_CUDA_KERNELS_DESIGN.md §2, §6.3, §6.4]`
**Fixture pin:** The current depth-1 histogram uses `(Σder1, Σweight)`. Whether channel-0 holds
Σweight or Σder2 for Logloss-Newton scoring must be pinned in the depth-6 fixture and cross-checked
against the CPU reference's split scores. `[ASSUMED]` (A2)

### Pattern 2: Subtraction trick buffer lifecycle (D-04)
**What:** When a parent leaf splits, stream-compute only the **smaller** child's histogram; derive
the larger sibling as `parent − smaller` in O(bins). Upstream `SubstractHistogramsImpl` does
`hist[fromId] -= hist[whatId]` per bin, **clamping the weight channel (`statId==0`) to `max(., 0)`**
(numerical guard against subtraction underflow). Keep only **parent-level** histograms resident
(not all levels): after a level's subtraction produces both children, the parent slot can be reused
for the next level.
**Buffer identity (verified):** `bigChild = parent − smallChild`, where "small" is selected by
partition size (`TDataPartition.Size`) — upstream `TPointwisePartOffsetsHelper` /
`GetPairwisePartIdToCalculate`. Leaves carry a `HistogramsType` (`Zeroes`/`PreviousPath`/`CurrentPath`).
**Grounding:** §1.4, §5.5 (`BuildNecessaryHistograms` sequence: Compute smaller → Zero → Substract → Scan), §6.4 `histogram_utils.cu` (`SubstractHistogramsImpl` clamp). `[CITED: §1.4, §5.5, §6.4]`
**Landmine:** The `max(0)` weight-channel clamp must be transcribed — omitting it yields tiny
negative weights that break the score denominator.

### Pattern 3: Newton der2 leaf estimation — single closed-form step (iterations=1)
**What:** Leaf value = `Σder1 / (−Σder2 + scaledL2)` (`newton_leaf_delta`, `online_predictor.h:162-170`).
`der2` is stored **non-positive** (RMSE `der2=−1`, Logloss `der2=−p(1−p)`), so `−Σder2 ≥ 0` and the
denominator is well-conditioned. RMSE's `−Σder2 = Σweight`, so Newton collapses to `calc_average`
(today's gradient leaf) — RMSE needs no new leaf math; **Logloss is the genuinely-new path**.
**Device recipe:** add a third partition-update channel Σ(der2·weight) using the Phase 7.2
`LoglossHessian` der2 handle (or `const_der_handle` for RMSE); read back `2^depth × 3` stats; compute
the delta on host for the oracle and via `apply_leaf_delta` on device (D-01, no per-iteration readback).
**Grounding:** `cb-compute/src/leaf.rs:118-158` (`newton_leaf_delta`), `cb-compute/src/histogram.rs:100`
(`reduce_leaf_der2` takes `weighted_der2[i] = der2·weight`), `cb-train/src/boosting.rs` `compute_leaf_deltas`
Newton arm (`reduce_leaf_stats` + `reduce_leaf_der2`). `[VERIFIED: source tree]`
**Weighting convention (LANDMINE from Phase 7.2/7.3):** der1/der2 handles are **UNWEIGHTED**; the weight
is folded downstream by the histogram/partition reduce. `reduce_leaf_der2` consumes `der2·weight`. The
device Σder2 channel must therefore fold weight the same way der1 does. `[VERIFIED: memory phase72/73 + histogram.rs]`

### Pattern 4: Per-level threading through `grow_oblivious_tree_into`
**What:** The Phase-10 loop (`gpu_runtime/mod.rs:1785`) already iterates `for level in 0..depth`,
does score → O(1) split → `partition_split`. It **rejects depth>1 at line 1740** solely because the
score step (`launch_find_optimal_split_pointwise_into`) scores the **whole dataset** (one partition),
not the `2^level` partitions. The extension is surgical: at the score step, swap the whole-dataset
histogram for the partition-aware (`fullPass=false`) fill keyed by the resident `leaf_of`, apply the
subtraction trick, then score per active leaf. `partition_split`/`partition_update`/`leaf_of`/read-back
are unchanged except the +1 der2 channel.
**Grounding:** `gpu_runtime/mod.rs:1737-1748` (the explicit depth>1 reject + forward-dependency note), `:1785-1834`. `[VERIFIED: source tree]`

### Anti-Patterns to Avoid
- **Reading the full histogram/partition buffer to host to score or partition** — the FORBIDDEN D-05 hybrid. Only the O(1) BestSplit + `2^depth` stats may cross.
- **f64 `Atomic::fetch_add` for the histogram accumulator** — non-deterministic order + unsupported on gfx1100. Use the fixed-point `Atomic<u64>` path.
- **`f32::NEG_INFINITY` / `-inf` literals inside `#[cube]`** — HIP JIT reject; use `f32::MIN` sentinel.
- **Adding a `cb-train` dependency to `cb-backend`** to reuse `newton_leaf_delta` — transcribe the formula inline instead (it is a one-liner).
- **Omitting the `max(0)` weight-channel clamp** in the subtraction kernel.
- **Materializing both children's histograms** (rejected by D-04).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Multi-block prefix sum (partition reorder, fold scan) | A new scan kernel | Phase-10 `scan.rs` two-level scan (validated at n=100_000: per-block scan + block-sum exclusive scan + offset add) | Cross-block carry already solved and oracle-tested. For `2^level ≤ 64` partitions the offset scan is single-block anyway. |
| Deterministic cross-cube reduction | A new atomic strategy | `block_reduce_fixedpoint_kernel` (k=30) + tree-reduce fallback | LOCKED winner (SPIKE-REDUCTION); zero run-to-run spread measured on gfx1100. |
| der1/der2 on device | New loss kernels | Phase 7.2 `der_seams.rs` handles (`LoglossHessian`, `RmseGradient`, `const_der`) | Device-resident, no-readback, already oracle-locked (23/23 live gfx1100). |
| Per-partition stat aggregation | A bespoke reduce | Extend `launch_partition_update_into` (+1 channel) | Already deterministic and D-05-compliant. |
| Doc repartition / `leaf_of` | New routing | `launch_partition_split_into` (forward-bit) | Already matches `cb_train::leaf_index` exactly (Phase-10 oracle). |
| In-place approx update | Host round-trip | `apply_leaf_delta_kernel` | Keeps Newton refinement device-resident (D-01). |
| Newton / average leaf math | Reimplement | Transcribe `newton_leaf_delta` / `calc_average` / `scale_l2_reg` from `cb-compute/src/leaf.rs` | The CPU oracle formulas; inline-transcribe (no cb-train dep). |

**Key insight:** Phase 10 delivered the entire substrate (primitive library, der seams, partition machinery, deterministic reduce, `apply_leaf_delta`, cindex). This phase is **composition + one new kernel behavior** (leaf-keyed `fullPass=false` histogram + subtraction), not new infrastructure. The risk is correctness/addressing precision, not missing capability.

## Common Pitfalls

### Pitfall 1: Assuming the CPU oracle runs an iterative Newton walker with backtracking
**What goes wrong:** Planning a multi-iteration device refinement loop (recompute der each step + line-search) to "match upstream," then diverging from the Rust oracle which does a single closed-form step.
**Why it happens:** Upstream CUDA (§5.6) genuinely has `TNewtonLikeWalker` + `step_estimator` backtracking; the design doc describes it. But the **Rust CPU reference is the parity target**, and it pins `leaf_estimation_iterations=1` with no walker/backtracking.
**How to avoid:** Pin `leaf_estimation_iterations=1` in the depth-6 fixture; implement the single closed-form step. If iterations>1 is ever desired, build the iterative walker in `cb-compute` FIRST (it is the oracle), then mirror on device.
**Warning signs:** A device leaf value that differs from `newton_leaf_delta` on the read-back stats.

### Pitfall 2: der2 sign / weighting mismatch
**What goes wrong:** Storing der2 positive, or reducing unweighted der2 while der1 is weighted (or vice-versa), inverting or mis-scaling the denominator.
**Why it happens:** der2 is non-positive by convention (`−p(1−p)`, `−1`); `−Σder2` appears in the denominator. Handles are unweighted; `reduce_leaf_der2` consumes `der2·weight`.
**How to avoid:** Fold weight into the der2 partition channel exactly as der1 is folded; keep the `−Σder2 + scaledL2` denominator. Cross-check against `newton_leaf_delta`.
**Warning signs:** Sign-flipped or exploding leaf values; RMSE (der2=−1) not collapsing to `calc_average`.

### Pitfall 3: Subtraction underflow on the weight/hessian channel
**What goes wrong:** `parent − smaller` produces a tiny negative weight/hessian, poisoning the score denominator or leaf estimate.
**How to avoid:** Clamp `statId==0` channel to `max(., 0)` in the subtraction kernel (upstream `SubstractHistogramsImpl`).
**Warning signs:** NaN/Inf scores in deep leaves; ε blowups only at higher levels.

### Pitfall 4: Compounding split-flip drift across hundreds of trees
**What goes wrong:** A sub-ε histogram difference tips a near-tie split at tree K, and the divergent structure compounds for the rest of the run — the final ε passes locally but fails in aggregate (or vice-versa).
**Why it happens:** Split selection is a discrete argmin over near-equal candidate gains; float order sensitivity flips ties.
**How to avoid:** The LOCKED deterministic reduction removes run-to-run order sensitivity; instrument the **per-tree split-agreement + run-to-run spread diagnostic** (D-05) so the first divergent tree is pinpointed, not just the final aggregate. Match the CPU argmin tie-break (lowest index, `ARGMAX()` macro / inline `l2_split_score` first-wins) exactly.
**Warning signs:** Final ε near the bound with no single obvious source; spread diagnostic shows a step change at one tree.

### Pitfall 5: `-inf` literal in the score/argmin `#[cube]` kernel
**What goes wrong:** Using `f32::NEG_INFINITY` as the "no candidate" sentinel compiles on cpu/wgpu but the HIP/gfx1100 JIT rejects `double(-inf)` (`undeclared identifier 'inf'`) — invisible until the rocm smoke run.
**How to avoid:** Use a finite `f32::MIN` sentinel inside kernels (host code may keep `f64::NEG_INFINITY`). Run the rocm suite in-env after any `#[cube]` change.
**Warning signs:** cpu/wgpu green, rocm 16/75.

### Pitfall 6: `leaf_of` bit-order / `2^level` slot addressing mismatch
**What goes wrong:** Level→bit convention diverges from `cb_train::leaf_index`, so device `leaf_of` disagrees with the CPU leaf assignment (structure mismatch, not just value).
**How to avoid:** `partition_split` already sets `leaf_of |= (bit << level)` matching `leaf_index` (Phase-10 oracle `leaf_of_matches_cpu_leaf_index`). The histogram slot addressing must use the SAME `leaf_of` value as the leaf id (`0..2^level`).
**Warning signs:** Per-object leaf disagreement in the SC-3 structure observation.

### Pitfall 7: Reading a Handle through the wrong client
**What goes wrong:** Undefined behavior / wrong data when a resident handle allocated on one `ComputeClient` is read via another.
**How to avoid:** One `&client` threads the whole tree (Phase-10 contract). Keep all resident handles bound to the allocating client.

## Runtime State Inventory

> Greenfield device-kernel extension, not a rename/refactor/migration — no stored data, service config, OS-registered state, secrets, or build artifacts embed a renamed string. **None — verified: this phase adds/extends `#[cube]` kernels and launch functions only; no identifiers are renamed across stored/registered state.**

## Code Examples

Existing signatures the planner should extend (from the source tree, not copied here in full):

### Newton leaf delta (the oracle formula to transcribe inline)
```rust
// Source: crates/cb-compute/src/leaf.rs:145 (newton_leaf_delta)
// leaf value = sum_der / (-sum_der2 + scaled_l2)
pub fn newton_leaf_delta(sum_der: f64, sum_der2: f64, scaled_l2: f64) -> f64 {
    let denom = -sum_der2 + scaled_l2;
    // (unconditional divide per CalcDeltaNewtonBody, online_predictor.h:162-170)
    // ... existing guard body ...
}
```

### Per-leaf der2 reduce (the oracle for the new partition channel)
```rust
// Source: crates/cb-compute/src/histogram.rs:100 (reduce_leaf_der2)
// weighted_der2[i] = der2 * weight ; returns Σ per leaf, count-order
pub fn reduce_leaf_der2(leaf_of: &[usize], weighted_der2: &[f64], n_leaves: usize) -> Vec<f64>;
```

### Grow-loop extension point (remove the depth>1 reject; swap the score step)
```rust
// Source: crates/cb-backend/src/gpu_runtime/mod.rs:1740 (the reject to remove)
if depth > 1 { return Err(CbError::OutOfRange("... fullPass=false histogram not landed ...")); }
// Source: :1792 (the whole-dataset score to replace with a leaf-keyed / partition-aware fill)
let (best, _scores) = launch_find_optimal_split_pointwise_into(client, der1, weight, cindex, indices, ...)?;
```

### der2 handle (reuse, no read-back)
```rust
// Source: crates/cb-backend/src/gpu_runtime/der_seams.rs
// DerUnaryKernel::LoglossHessian  → der2[i] = -p*(1-p), p = sigmoid(approx[i])  (target-independent)
// DerBinaryKernel::RmseGradient   → der2 = const -1.0 via const_der_handle (no kernel)
```

## State of the Art

| Old Approach (Phase 10) | Current Approach (Phase 11) | When Changed | Impact |
|--------------------------|------------------------------|--------------|--------|
| Whole-dataset histogram, one partition (depth-1 stump) | Leaf-keyed partition-aware `fullPass=false` histogram + subtraction trick | This phase | Enables depth>1 device trees. |
| Gradient leaf via `calc_average` (RMSE + Logloss) | Newton leaf via `newton_leaf_delta` for Logloss (RMSE unchanged — collapses to average) | This phase | GPUT-07: correct classification leaf values. |
| ε ≤ 1e-5 (exact level-0 histogram) | ε = 1e-4 across hundreds of depth>1 trees | This phase | Operative standing bar GPUT-14 (looser due to accumulated f32 reductions). |

**Deprecated/outdated within this codebase:**
- The depth>1 typed reject in `grow_oblivious_tree_into` (`mod.rs:1740`) — removed this phase.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `leaf_estimation_iterations=1` is the right fixture pin (matches every existing fixture; no iterative walker in the CPU oracle). | Summary / Pitfall 1 | If user wants iterations>1, the CPU oracle must gain an iterative Newton walker (+ possibly backtracking) BEFORE the device can match — a scope expansion, not a device-only task. |
| A2 | For Logloss-Newton scoring, histogram channel-0 should carry Σder2 (hessian), not Σweight. | Pattern 1 | Wrong channel semantics → split scores diverge from the CPU reference; must be pinned/verified in the fixture against CPU split scores. |
| A3 | The der2 partition channel folds weight identically to der1 (`der2·weight`). | Pattern 3 / Pitfall 2 | Mis-scaled denominator → wrong leaf values. Verified against `reduce_leaf_der2` but device folding must match. |
| A4 | `cubecl 0.10.0` (workspace pin) is acceptable; no bump needed for this phase. | Standard Stack | Registry currency unverified in-session; a bump is orthogonal and out of scope. |
| A5 | For depth ≤ 6, `2^level ≤ 64` partitions make the partition-offset prefix-sum single-block (no cross-block carry needed); fold-scan reuses the validated two-level scan. | Don't Hand-Roll | If a future deeper tree is attempted, revisit carry correctness (already handled by the two-level scan). |

**These `[ASSUMED]` items should be confirmed by the user/planner during discuss/plan — especially A1 (iteration count) and A2 (channel semantics).**

## Open Questions

1. **Newton `leaf_estimation_iterations`: 1 or >1?** (Highest priority.)
   - What we know: Every existing fixture pins iterations=1; the CPU oracle computes a single closed-form `newton_leaf_delta` with no refinement loop and no backtracking. Upstream CUDA has an iterative walker + backtracking, but that is not the Rust parity target.
   - What's unclear: Whether D-01's "leaf_estimation_iterations steps … recompute der per step" intends a genuinely multi-step run. If so, the CPU reference does not implement it today.
   - Recommendation: **Pin iterations=1** for Phase 11 (D-02 pins the count anyway). The device Newton is then a single Σder2-channel reduce + `newton_leaf_delta` + one `apply_leaf_delta`. If multi-step is required later, add the iterative walker to `cb-compute` first as its own scoped work.

2. **Histogram channel semantics for the second-order (Newton/Cosine) path.**
   - What we know: statId 0 = weight/count, statId 1 = gradient (§6.3/§6.4). Second-order scoring wants Σder2 in the denominator.
   - What's unclear: Whether the depth-1 path's `(Σder1, Σweight)` should become `(Σder1, Σder2)` for Logloss, and whether the GPU default score function (Cosine vs NewtonCosine) reads der2.
   - Recommendation: Pin the score function and channel-0 semantics in the depth-6 Logloss fixture; cross-check device split scores against the CPU reference score for the first few trees, not just the final prediction.

3. **Does depth-1 Logloss (GPUT-04) currently use gradient leaves?**
   - What we know: `grow_loop.rs` uses `calc_average` (gradient) for both RMSE and Logloss.
   - What's unclear: Whether the Phase-10 depth-1 Logloss fixture pinned a gradient leaf method (making it a weaker match) that this phase's Newton path supersedes.
   - Recommendation: Verify the Phase-10 Logloss fixture leaf method; ensure the Newton path is the one exercised at depth>1 and that depth-1 Logloss is re-checked under Newton (or documented as gradient-pinned).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| CubeCL cuda backend (Kaggle) | Authoritative correctness + speed oracle (GPUT-14, BENCH-02) | ✗ in-env (human-gated Kaggle) | — | None — Kaggle run is the gate (human-gated external step). |
| ROCm / gfx1100 (in-env) | Optional compile/smoke iteration | ✓ | ROCm 7.1, gfx1100/RDNA3 wave32 | Not a gate; convenience only. |
| CubeCL wgpu/cpu backends | Local `cargo test` / self-oracle | ✓ | cubecl 0.10.0 | — |
| `cubecl` crate | All device kernels | ✓ | 0.10.0 (workspace pin) | — |

**Missing dependencies with no fallback:** Kaggle CUDA is the sole authoritative GPU oracle — correctness AND speed sign-off is a human-gated external run (per BENCH-01/02, D-05). The subagent cannot run it; the orchestrator/human discharges it.
**Missing dependencies with fallback:** ROCm smoke in-env substitutes for fast local `#[cube]` verification (compile + bit-behavior), but never satisfies a requirement alone.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` (source/test separation mandatory — tests in dedicated files, never `mod tests` in production source) |
| Config file | none — Cargo workspace; `cb-backend` feature-gated (`cpu`/`wgpu`/`cuda`/`rocm`) |
| Quick run command | `cargo test -p cb-compute` (CPU oracle math) |
| Full suite command | `cargo test -p cb-backend --no-default-features --features rocm grow` (in-env GPU smoke) + Kaggle CUDA notebook (authoritative) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| GPUT-05 | Partition-aware `fullPass=false` hist + subtraction matches CPU leaf-keyed scatter ≤1e-4 | unit (self-oracle) | `cargo test -p cb-backend --features rocm grow_loop` | ❌ Wave 0 (extend `grow_loop.rs` depth>1 case) |
| GPUT-05 | Device `leaf_of` over depth-6 split sequence == CPU `leaf_index` | unit | `cargo test -p cb-backend --features rocm leaf_of_matches_cpu` | ✅ (depth-1; extend to depth-6) |
| GPUT-06 | Deterministic reduce: zero run-to-run spread over ≥32 launches | unit | `cargo test -p cb-backend --features rocm reduce` | ✅ (SPIKE-REDUCTION harness) |
| GPUT-07 | Newton leaf value == `newton_leaf_delta` on read-back Σder1/Σder2 stats | unit | `cargo test -p cb-backend --features rocm newton` | ❌ Wave 0 |
| GPUT-07 | `reduce_leaf_der2` / `newton_leaf_delta` CPU math | unit | `cargo test -p cb-compute leaf` | ✅ |
| GPUT-14 | Final-prediction ε=1e-4 over full depth-6 run (RMSE + Logloss) | integration | Kaggle `bench/cuda_oracle.ipynb` (human-gated) | ❌ Wave 0 (extend harness) |
| GPUT-14 | Per-tree split-agreement + run-to-run spread diagnostic (D-05) | integration | Kaggle harness diagnostic cell | ❌ Wave 0 |
| BENCH-02 | Depth-6 RMSE + Logloss device vs host-CPU vs official CatBoost GPU wall-clock | benchmark | Kaggle harness speed cell (warm-run/JIT-excluded) | ❌ Wave 0 (extend `bench/RESULTS.md`) |

### Sampling Rate
- **Per task commit:** `cargo test -p cb-compute` (fast CPU oracle) + `cargo test -p cb-backend --features rocm <touched kernel>` after any `#[cube]` change (the rocm `-inf`/JIT landmine mandates in-env GPU run).
- **Per wave merge:** `cargo test -p cb-backend --no-default-features --features rocm` (full in-env GPU smoke).
- **Phase gate:** Kaggle CUDA correctness (blocking) + speed sign-off logged in `bench/RESULTS.md`, then `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] Depth-6 RMSE + Logloss self-oracle in `crates/cb-backend/src/kernels/grow_loop.rs` — covers GPUT-05/07.
- [ ] `reduce_leaf_der2` / Newton device-vs-CPU self-oracle — covers GPUT-07.
- [ ] Extend the Phase-10 synthetic generator to depth-6 RMSE + Logloss configs (D-03) — correctness fixture + speed workload.
- [ ] Extend `bench/cuda_oracle.ipynb`: final-ε gate + per-tree split-agreement/spread diagnostic + depth-6 speed cells (D-05, BENCH-02).
- [ ] rocm in-env smoke assertion after each `#[cube]` change (cpu/wgpu can false-pass the `-inf` landmine).

## Security Domain

> `security_enforcement: true`, ASVS level 1, block-on high. This is a numerical compute-kernel phase with no auth, network, session, or external-input surface; the relevant control is input validation on host-side buffer sizing.

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes | Host-side length/overflow guards before every launch: `checked_shl` for `2^depth`, `checked_mul` for `n_features*n`, `cindex.len()` stride check, `leaf_of[obj] < n_parts` bound before the atomic store (all present in `grow_oblivious_tree_into`; extend for the +der2 channel and `2^level` slot sizing). |
| V6 Cryptography | no | — |

### Known Threat Patterns for CubeCL device kernels
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Out-of-bounds device write (leaf slot `>= 2^level`, part-stats index) | Tampering / DoS | Host validates `leaf_of` range and buffer lengths before upload; `launch_unchecked` only after host validation (manual §11). |
| Integer overflow in buffer sizing (`2^depth`, `n_features*n`, `2^depth*3`) | DoS | `checked_shl`/`checked_mul` → typed `CbError::OutOfRange` (existing pattern). |
| Non-deterministic reduce corrupting parity | Tampering (silent wrong result) | LOCKED fixed-point `Atomic<u64>` + explicit `AtomicFinalizePath` capability reporting (no silent downgrade). |
| `unwrap()` panic on device read-back failure | DoS | Prohibited in production (CLAUDE.md); read-back failures surface `CbError::Degenerate`. |

## Sources

### Primary (HIGH confidence)
- Source tree (read directly this session): `crates/cb-compute/src/{leaf.rs,histogram.rs,runtime.rs}`, `crates/cb-train/src/boosting.rs`, `crates/cb-backend/src/gpu_runtime/{mod.rs,der_seams.rs}`, `crates/cb-backend/src/kernels/{grow_loop.rs,scan.rs}` — CPU-oracle formulas, der seams, existing depth-1 grow loop, the depth>1 reject.
- `.planning/phases/10-.../SPIKE-REDUCTION.md` — LOCKED reduction strategy (fixed-point `Atomic<u64>` k=30, tree-reduce fallback).
- `.planning/phases/11-.../11-CONTEXT.md` — locked decisions D-01..D-05, landmines, reusable assets.
- `.planning/REQUIREMENTS.md` — GPUT-05/06/07/14, BENCH-02 text + traceability.

### Secondary (MEDIUM confidence)
- `CATBOOST_CUDA_KERNELS_DESIGN.md` §1.4 (subtraction trick), §5.5 (`BuildNecessaryHistograms` sequence), §5.6 (leaf-value estimation: Newton walker + host solve + backtracking — upstream, NOT the Rust oracle), §6.3 (`pointwise_hist2` / `HIST_COUNT=2`), §6.4 (leaf-wise builder, `histogram_utils` subtraction clamp, tensor indexing, `split_points`) — the upstream reference the device kernels transcribe.
- CubeCL manual `manual/cubecl/{08_atomic_contention,09_fixedpoint_atomics,10_grid_stride_occupancy,Cubecl_shared_memory}.md` — fixed-point atomics, LDS privatization, occupancy (note: path is lowercase `cubecl`, not `Cubecl`).

### Tertiary (LOW confidence)
- crates.io registry currency for `cubecl` — API unavailable in-session; workspace pin `0.10.0` used as-is.

## Metadata

**Confidence breakdown:**
- CPU-reference behavior (Newton single-step, no backtracking; der2 sign/weighting; leaf formulas): HIGH — read directly from the source tree.
- Existing seams / extension points (grow loop, der seams, partition machinery, deterministic reduce): HIGH — read directly.
- Device histogram channel layout / `2^level` addressing / subtraction lifecycle: MEDIUM — grounded in the upstream design doc; must be pinned/verified in the depth-6 fixture (A2).
- Reduction strategy: LOCKED (not researched) — SPIKE-REDUCTION §5b.
- Package legitimacy: HIGH — no new packages.

**Research date:** 2026-07-03
**Valid until:** ~2026-08-02 (30 days; stable internal codebase). Re-verify if `cb-compute` leaf/loss math or the Phase-10 seams change.
