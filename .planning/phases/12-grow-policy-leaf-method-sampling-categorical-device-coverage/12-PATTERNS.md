# Phase 12: Grow-Policy, Leaf-Method, Sampling & Categorical Device Coverage - Pattern Map

**Mapped:** 2026-07-03
**Files analyzed:** 24 (5 device-kernel families + non-sym/Region emission + seam/gate extensions + per-family oracle tests)
**Analogs found:** 23 / 24 (Region CPU *path* model shape is the single partial — no in-repo path-model analog; nearest is the non-symmetric node graph, which it must NOT copy)

> This is a **transcription-and-wiring** phase. Every device family already has (a) an upstream CUDA authority to port from (`CATBOOST_CUDA_KERNELS_DESIGN.md` §§ + vendored `catboost-master/catboost/cuda/`) and (b) an in-repo Rust CPU reference that is the ≤1e-5 oracle. The pattern work below maps each new file to its closest **in-repo** analog for structure/shape, and names the upstream CUDA source to transcribe the math from. Landmine boundaries are called out per-file.

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| **W1 — Depthwise/Lossguide device emission** | | | | |
| `cb-backend/src/kernels/nonsym_grow.rs` (new) | kernel | transform (per-level split scoring/selection) | `cb-backend/src/kernels/grow_loop.rs` + `score_split.rs` | role+flow exact |
| `cb-backend/src/kernels/nonsym_grow_test.rs` (new) | test | self-oracle | `cb-backend/src/kernels/grow_loop.rs` (test mod, `max_divergence`) | exact |
| `cb-compute/src/runtime.rs` (modify — extend `DeviceGrownTree`) | seam/model | request-response | `DeviceGrownTree` (:917) + `cb-train/src/tree.rs::GrownTree` (:196) | exact |
| `cb-backend/src/gpu_runtime/session.rs` (modify — widen gate) | config/gate | request-response | `GpuTrainSession::begin` (:137) | exact |
| **W2 — Region (CPU path FIRST, then device)** | | | | |
| `cb-train/src/tree.rs` (modify — Region grower) | service | transform | `leaf_wise_grower` (:909) | role-match (path ≠ node graph) |
| `cb-train/src/boosting.rs` (modify — lift `validate_grow_policy`) | config | request-response | `validate_grow_policy` (:1331) + `EGrowPolicy` (:103) | exact |
| `cb-model/src/model.rs` (modify — add `TreeVariant::Region` + `RegionTree`) | model | transform | `TreeVariant` (:143) / `NonSymmetricTree` (:113) / `from_trained` (:262) | role-match (new path shape) |
| `cb-model/src/apply.rs` (modify — Region path apply) | service | transform (walk-until-diverge) | `leaf_index_nonsym` (:234) | role-match (path, not node graph) |
| `cb-model/src/json.rs` (modify — Region round-trip) | serializer | file-I/O | existing non-sym flat-triple round-trip | role-match |
| `cb-train/src/region_grow_test.rs` (new) | test | oracle | `cb-train/src/tree_test.rs` (sibling `#[path]` test) | exact |
| `cb-model/src/region_apply_test.rs` (new) | test | oracle | `cb-model` non-sym apply test | exact |
| `cb-backend/src/kernels/region_device.rs` (new) | kernel | transform | `nonsym_grow.rs` (W1) + `grow_loop.rs` | role-match |
| `cb-backend/src/kernels/region_device_test.rs` (new) | test | self-oracle | `grow_loop.rs` test mod | exact |
| **W3 — Exact weighted-quantile leaf** | | | | |
| `cb-backend/src/kernels/exact_quantile.rs` (new) | kernel | transform (segmented sort→scan→binsearch) | `cb-backend/src/kernels/sort.rs` + `segmented_scan.rs` + `reduce.rs` | role+flow good |
| `cb-backend/src/kernels/exact_quantile_test.rs` (new) | test | self-oracle | `sort.rs` test mod / `grow_loop.rs::max_divergence` | exact |
| **W4 — Bootstrap + random-strength** | | | | |
| `cb-backend/src/kernels/bootstrap_device.rs` (new) | kernel | event-driven (RNG draw over resident ders) | `cb-train/src/bootstrap.rs` (CPU ref) + `der_seams.rs` (resident der handles) | role-match |
| `cb-backend/src/kernels/bootstrap_device_test.rs` (new) | test | self-oracle (frozen CPU sample) | `bootstrap_test.rs` shape + `grow_loop.rs::max_divergence` | exact |
| **W5 — MVS** | | | | |
| `cb-backend/src/kernels/mvs_device.rs` (new) | kernel | transform (block reduction over resident ders) | `bootstrap.rs::mvs_sample_weights` (:281) + `reduce.rs`/`scan.rs` | role-match |
| `cb-backend/src/kernels/mvs_device_test.rs` (new) | test | self-oracle | `grow_loop.rs` test mod | exact |
| **W6 — CTR device port (highest uncertainty)** | | | | |
| `cb-backend/src/kernels/ctr_device.rs` (new) | kernel | streaming (ordered prefix, resident across permutation) | `cb-train/src/ctr/online.rs::online_ctr_prefix_binclf` (:263) + `cindex.rs` | role-match |
| `cb-backend/src/kernels/ctr_device_test.rs` (new) | test | self-oracle (frozen CPU CTR column) | `ctr/online_test.rs` + `grow_loop.rs::max_divergence` | exact |
| **Cross-cutting** | | | | |
| `cb-compute/src/runtime.rs` (modify — `begin_device_training` config surface) | config | request-response | `begin_device_training` (:1012) | exact (Open Q2: introduce host config struct) |

---

## Pattern Assignments

### `cb-backend/src/kernels/nonsym_grow.rs` (kernel, transform) — W1

**In-repo analog:** `cb-backend/src/kernels/grow_loop.rs` (host-light greedy oblivious grow) + `score_split.rs` (per-candidate score + argmin).
**Upstream to transcribe:** §5.4 `SelectLeavesToSplit`, §6.4 `ComputeOptimalSplit` (Depthwise/Lossguide); vendored `methods/greedy_subsets_searcher/kernel/`.
**CPU parity oracle:** `cb-train/src/tree.rs::leaf_wise_grower` (:909) — the ≤1e-5 reference.

**What to reuse from the analog:** the grow driver already reads back an O(1) `BestSplit` descriptor + per-partition stats per level and repartitions on device (`launch_partition_split_into`, `grow_loop.rs`). W1 keeps that spine; it changes the *leaf-selection order* (level-order for Depthwise / best-gain-priority for Lossguide) and emits a node graph instead of a `depth`-length split list. The score kernel and argmin (`launch_find_optimal_split_pointwise`, `score_split.rs`) are reused as-is per candidate leaf.

**`-inf` sentinel landmine (Pitfall 7 / MEMORY `cubecl-hip-no-inf-literal`):** the score ARGMAX sentinel `(-1, FLT_MAX)` in `ComputeOptimalSplit*` must use a finite `f32::MIN`/`f32::MAX` in the `#[cube]` kernel — a literal `-inf` JIT-rejects on HIP/gfx1100 and is invisible to cpu/wgpu `cargo check`. `score_split.rs` already carries this discipline.

**Emission boundary (D-04, host-structs-only):** the device returns *structure + leaf values only* as plain host structs. Do NOT create a device-native non-symmetric tree type (anti-pattern). Fill the extended `DeviceGrownTree` fields (below) and let host `Model::from_trained` build `TreeVariant::NonSymmetric`.

---

### `cb-compute/src/runtime.rs` — extend `DeviceGrownTree` (seam) — W1

**Analog (extend in place):** `DeviceGrownTree` (:917), today oblivious-only.

Current shape (plain host structs — no cubecl type crosses the seam, T-10-04):
```rust
pub struct DeviceGrownTree {
    pub splits: Vec<(u32, u32)>,   // (feature_index, bin_id); length = depth
    pub leaf_values: Vec<f64>,     // UN-scaled by learning_rate; length 2^depth
    pub leaf_of: Vec<u32>,         // EMPTY in hot path (D-05); length n only for oracle
}
```

**D-04 extension — mirror `GrownTree`'s non-sym fields (`cb-train/src/tree.rs:227-232`) verbatim as plain host structs:**
```rust
// from cb-train/src/tree.rs::GrownTree — the exact shapes to add:
pub step_nodes: Vec<(u16, u16)>,      // (left_subtree_diff, right_subtree_diff); (0,0)=terminal
pub node_id_to_leaf_id: Vec<u32>,     // u32::MAX = interior sentinel; else distinct leaf id
```
The `u32::MAX` interior sentinel and checked `u16::try_from` child-diff encoding are load-bearing (`tree.rs:1144-1188`, CR-01/CR-02) — the cbm serializer's `distinct_leaves` filter counts only ids `!= u32::MAX`. Keep the same contract on the device-produced structs.

**Also extend `begin_device_training` (:1012) config surface (Open Q2):** rather than growing the 12-arg list per wave, introduce a small **plain host-typed** `DeviceTrainConfig` struct (no cubecl) carrying `grow_policy`/`max_leaves`/`min_data_in_leaf`, `bootstrap_type`/`sample_rate`/`mvs_lambda`/pinned-seed, the exact-leaf flag + `alpha`/`delta`, and the CTR config. Keeps the seam landmine-safe.

---

### `cb-backend/src/gpu_runtime/session.rs` — widen coverage gate (gate) — every wave

**Analog:** `GpuTrainSession::begin` (:137) — the single coverage-gate extension point (A2).

Current gate (:154), the `Ok(None)`→CPU-fallback shape each family widens:
```rust
if depth != 1 || !boosting_type_is_plain || fold_count != 1 {
    return Ok(None);           // uncovered → CPU fallback (D-04)
}
let der_kernel = match map_der_kernel(loss) { Some(k) => k, None => return Ok(None) };
let score_fn   = match map_score_fn(score_function) { Some(s) => s, None => return Ok(None) };
```
**Pattern to replicate per family:** add a `match`/guard arm that returns `Ok(None)` until that family's Kaggle CUDA sign-off lands. `depth != 1` is relaxed to the Phase-11 depth>1 substrate for W1/W2 (verify A3 wiring at plan time — `session.rs` still literally reads `depth != 1`). Each new arm is all-or-nothing (D-10-01): the whole fit is device or CPU, never mixed. `map_score_fn`/`map_der_kernel` (:98/:115) are the template for new `Option`-returning config maps (e.g. `map_bootstrap_kernel`, `map_leaf_method`).

---

### `cb-train/src/tree.rs` — Region CPU grower (service) — W2a ⚠ largest lift

**Analog for the *grower driver* (structure, NOT the model shape):** `leaf_wise_grower` (:909) — same `check_depth` guard, same `best_split_for_leaf` shared scoring core, same node-registration bookkeeping (`new_node`/`do_split` closures, :936/:965).

**CRITICAL divergence (Pitfall 2, D-03 note):** Region is a *path* (`TRegionModel`, walk-while-direction-matches, leaf = depth reached at divergence), NOT a `TTreeNode[]` binary node graph. Do NOT reuse the `step_nodes`/`node_id_to_leaf_id` finalization (`tree.rs:1137-1204`). A Region of depth `d` has exactly `d+1` leaves along ONE path (`MaxLeaves = MaxDepth+1`, §5.4 `ComputeOptimalSplitsRegion`) — warning sign if leaf count == `2^depth`.
**Upstream to transcribe:** §6.4 `ComputeOptimalSplitsRegion`, §5.4 `SelectLeavesToSplit` (Region arm).
This grower has **no existing CPU oracle** — it establishes the ≤1e-5 reference itself (Pitfall 1); freeze its output in a fixture.

---

### `cb-train/src/boosting.rs` — lift `validate_grow_policy` (config) — W2a

**Analog (modify in place):** `validate_grow_policy` (:1331) + `EGrowPolicy` (:103).
Current rejection to lift once the CPU Region path exists:
```rust
if grow_policy == EGrowPolicy::Region {
    return Err(CbError::OutOfRange("grow_policy=Region is not supported ... \"Region OUT\"".to_owned()));
}
```
Keep the monotone-constraints guard (:1339). Add a `Region` arm to `EGrowPolicy::is_non_symmetric` classification consciously (Region is upstream-non-symmetric but a distinct path — decide its dispatch explicitly, do not let it silently fall through to `leaf_wise_grower`).

---

### `cb-model/src/model.rs` — `TreeVariant::Region` + `RegionTree` (model) — W2a

**Analog for the enum/builder plumbing:** `TreeVariant` (:143), `NonSymmetricTree` (:113), `Model::from_trained` (:262).
**Analog for the NEW shape:** none in-repo — this is the single partial-match file. Model the upstream `TRegionModel` *path*: per-level `(feature, bin, expected_direction)` + one-hot flag packed as in `takeEqualAndSplitDirection` (one-hot in bit 0, expected direction in bit 1, §6.6 `AddRegionImpl`); `leaf_values` length `depth+1`.

Follow the existing enum discipline exactly:
- Add `Region(RegionTree)` to `TreeVariant` (:143); every consumer matches exhaustively (apply/SHAP/fstr/serialize) so the new arm can never be silently dropped (existing pattern comment, :137-141).
- Add `region_trees: Vec<RegionTree>` to `Model` (:214) mirroring `non_symmetric_trees` (:223) — a model is EITHER all-oblivious OR all-non-sym OR all-region; keep the oblivious/non-sym lift paths byte-identical (D-04 no-regression).
- Extend `from_trained` (:262) with a `region_trees` lift arm mirroring the `non_symmetric_trees` arm (:309-323).

---

### `cb-model/src/apply.rs` — Region path apply (service) — W2a

**Analog:** `leaf_index_nonsym` (:234) — copy its *defensive discipline* (all checked `.get`, bounded loop, `u32::MAX` malformed guard, `Option` return so a crafted graph contributes `0.0` not a panic), but NOT its node-graph walk.

**Region walk to reproduce (§6.6 `AddRegionImpl`/`ComputeRegionBinsImpl`):**
```text
bin = 0
for level in 0..depth:
    featureVal = quantized_bin[feature[level]]
    split = OneHot ? (featureVal == value[level]) : (featureVal > value[level])
    if split != expectedDirection[level]:  break     // path diverges
    bin += 1                                          // else advance along the region path
leaf = bin                                            // MaxLeaves = MaxDepth+1
```
Add a `Region` arm to `predict_raw_one` (:282) mirroring the `non_symmetric` arm (:297-304) — a region model's `oblivious_trees`/`non_symmetric_trees` are empty, so exactly one loop contributes.

---

### `cb-backend/src/kernels/exact_quantile.rs` (kernel, transform) — W3

**In-repo primitives to reuse (Don't Hand-Roll):** `cb-backend/src/kernels/sort.rs` (`run_radix_sort`, stable keys+values, :87), `segmented_scan.rs`, `reduce.rs`. Audit segmentation granularity first (Open Q1 / A1) — if only whole-buffer sort exists, add a shared segmented-radix-sort sub-task (also used by W5).
**Upstream:** §6.3 `exact_estimation.{cu,cuh}` + §5.6 `EstimateExact`.
**CPU parity oracle:** `cb-compute/src/leaf.rs::exact_leaf_delta` (:367) — reproduce its exact semantics on device:
```text
weightsWithTargets[i] = weights[i] / max(1, |target[i]|)
needWeights = totalWeight * alpha              // det. reduce (fixed-point Atomic<u64> k=30)
SegmentedRadixSort(targets, weights) per leaf-bin
weightsPrefixSum = SegmentedScanVector(weights)
quantileDoc = binarySearch(prefixSum >= needWeights - DBL_EPSILON)   // fixed iter count
leaf.point = targets[quantileDoc] (± delta adjust per less/equal weight test)
```
Route Quantile/MAE/MAPE objectives here — NOT the Newton der2 path (Pitfall 6; A4: confirm the Exact objective set against `leaf.rs` at plan time). Every SUM (`totalWeight`, prefix) goes through the deterministic reduction (Pattern 3).

---

### `cb-backend/src/kernels/bootstrap_device.rs` (kernel, event-driven) — W4

**CPU ref to transcribe inline (no `cb-train` dep — landmine):** `cb-train/src/bootstrap.rs` — `EBootstrapType` (:69), `bayesian_weight` (:145), `generate_random_weights` per-block reseed (:156), `set_sampled_control` (:182).
**Resident der handles:** `cb-backend/src/gpu_runtime/der_seams.rs` — the RNG draw runs over the resident UN-weighted der1/der2 handles; keep derivatives resident (D-08, no host round-trip for the mask).
**Upstream:** §6.1 `bootstrap.{cu,cuh}` + `random*.cuh` (MWC `AdvanceSeed` / LCG `AdvanceSeed32` + `GenerateSeeds`).

**RNG stream landmine (Pitfall 5):** the CPU stream is CONTINUOUS across iterations (Bernoulli/main), never reseeded per tree; Bayesian uses per-block reseed `TFastRng64::from_seed(randSeed + blockIdx).advance(10)` and a base-2 log APPROXIMATION (~1e-5 sensitive). Reproduce the exact seed-advance + per-block-reseed layout; never use `rand`. **D-07 discipline:** pin seed, freeze the exact CPU sample in the fixture, reproduce bit-for-bit.

---

### `cb-backend/src/kernels/mvs_device.rs` (kernel, transform) — W5

**CPU ref to transcribe inline:** `cb-train/src/bootstrap.rs` — `mvs_sample_weights` (:281), `calculate_threshold` (:209), `single_probability` (:193), `MVS_BLOCK_SIZE = 8192` (:64).
**Primitives:** `reduce.rs` / `scan.rs` for the per-block scan (upstream uses `cub::BlockRadixSort`+`BlockScan`+`GetThreshold`; the CPU ref is a recursive quickselect-partition estimator — match the *threshold semantics*, not the algorithm).
**Upstream:** §6.1 `mvs.{cu,cuh}`.

**Threshold landmine (Pitfall 4):** per-block (8192) threshold over `sqrt(lambda + der²)` candidates; `lambda = GetLambda(...)`; `single_probability(|der|, threshold) = |der|>threshold ? 1 : |der|/threshold`; `weight = 1/p` w.p. `p` else `0` via `NextUniformF`; `sample_rate` is f32-rounded then promoted. Deterministic reduction for the block scan (Pattern 3). Warning sign: sampled count per block off from `sample_rate*blockSize`.

---

### `cb-backend/src/kernels/ctr_device.rs` (kernel, streaming) — W6 ⚠ highest uncertainty

**CPU ref to transcribe inline:** `cb-train/src/ctr/online.rs` — `online_ctr_prefix_binclf` (:263, read-before-increment object-order output), `ordered_ctr_per_permutation` (:361), `TCtrHistory`/`TCtrMeanHistory` (:59/:95); also `calc_ctr.rs`, `ctr_feature.rs` (combined projection, A5), `final_ctr.rs`. `ECtrType` (`ctr/mod.rs:67`).
**cindex join:** `cb-backend/src/kernels/cindex.rs` (`pack_cindex`) — binarize accumulated CTRs into ADDITIONAL cindex columns the histogram loop already reads.
**Upstream:** §6.6 `ctrs/kernel/ctr_calcers.{cu,cuh}` + `batch_binarized_ctr_calcer.h`.

**Ordered-CTR landmine (Pitfall 3):** ordered CTR reads the prefix statistic BEFORE incrementing (read-before-increment), resets at segment starts, applies group-wise correction. Upstream encodes segment starts as a sign bit (`TIndexWrapper`) and ORs prior-flag + `bins[i]!=bins[i-1]` + previous-layer-bin-change (`UpdateBordersMask`); `MakeGroupStarts`/`FillBinIndices`/`ApplyGroupwiseCtrFix` share one canonical CTR per category-within-group. Stay resident across the permutation (D-06). Scope W6 to the single-permutation covered regime first (Open Q3); defer multi-fold CTR behind `Ok(None)`. **Freeze the CPU CTR column in the fixture** (D-07 discipline extended to CTR). A categorical-heavy synthetic fixture may be needed (A6).

---

## Shared Patterns

### Pattern A — `Ok(None)` family-gated coverage flip
**Source:** `cb-backend/src/gpu_runtime/session.rs::begin` (:154) + `cb-compute/src/runtime.rs::grow_tree_on_device` (:1059, default `Ok(None)`) / `begin_device_training` (:1012, default `Ok(false)`).
**Apply to:** every wave. Each family adds a gate arm returning `Ok(None)` until its Kaggle CUDA sign-off lands; a fit is fully device-grown or fully CPU-grown (D-10-01, all-or-nothing). Never fabricate a device result (T-10-05 / Phase 11-05 PAUSED precedent).

### Pattern B — Host-structs-only seam (no `cb-train` dep in `cb-backend`)
**Source:** `DeviceGrownTree` doc (`runtime.rs:900`, "only plain host types ... never gains a cubecl/cb-backend dependency via feature unification, T-10-04").
**Apply to:** W1/W2 emission (node graph / region path cross as `Vec<(u16,u16)>` / plain structs), the `DeviceTrainConfig` struct, every kernel that needs a CPU ref (transcribe inline, never `use cb_train::...`). MEMORY landmine: adding a `cb-train` dep to `cb-backend` breaks the rocm runtime via feature unification (Phase 7.5).

### Pattern C — Deterministic reduction for every parity-critical SUM
**Source:** SPIKE-REDUCTION §5b + MEMORY `phase10-03-reduce-determinism-spike` — fixed-point `Atomic<u64>` k=30 accumulator + fixed-order tree-reduce fallback. gfx1100 has `Atomic<u64>` add but NO f64 atomic-add.
**Apply to:** CTR prefix sums, MVS block scans, Exact `totalWeight`/weight prefix sums, `ComputeTargetVariance`/`ScoreStdDev`. Any un-ordered device SUM breaks the ε=1e-4 bar.

### Pattern D — `-inf` sentinel forbidden in `#[cube]` kernels
**Source:** MEMORY `cubecl-hip-no-inf-literal` + Pitfall 7. `F::new(f32::NEG_INFINITY)` emits `double(-inf)` → HIP/gfx1100 JIT reject; invisible to cpu/wgpu `cargo check`. `score_split.rs` already uses finite sentinels.
**Apply to:** the score ARGMAX sentinel `(-1, FLT_MAX)` in W1/W2 `ComputeOptimalSplit*`; any device min/max init. Use `f32::MIN`/`f32::MAX`; keep `f64::NEG_INFINITY` only in host code. Run the rocm smoke suite in-env after any `#[cube]` change.

### Pattern E — Serial CPU self-oracle test (divergence at ε)
**Source:** `cb-backend/src/kernels/grow_loop.rs::max_divergence` (:61) + `score_split.rs` tests (:307). Structure decisions (integer `(feature,bin)`/leaf) are the STRICT bar (`assert_eq!`); float scores use a REPORTED run-stable bound (`SCORE_BOUND` f32 1e-3 / f64 1e-9, :71) — the GPU-06 ε is signed off on Kaggle CUDA, not hard-coded (D-7.5-05).
**Apply to:** every new `*_test.rs`. `max_divergence` returns `(f64::INFINITY, INFINITY)` on length mismatch so a truncated device buffer fails loudly (WR-06). Source/test separation is mandatory (CLAUDE.md): tests live in sibling `*_test.rs` mounted via `#[cfg(test)] #[path = "..."]` (see `tree.rs:82-96`), NEVER `mod tests` in the production file.

### Pattern F — D-07 pin-seed + freeze-CPU-sample fixture
**Source:** `cb-train/src/bootstrap.rs` (`TFastRng64` continuous stream) + Phase-11 discipline.
**Apply to:** W4/W5/W6 (and every family's ε=1e-4 fixture). Pin the RNG seed/config, freeze the exact CPU-reference sample/CTR-column/quantile in the oracle fixture, reproduce bit-for-bit on device. Deterministic and checkable at ε — not a distributional check.

---

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `cb-model/src/model.rs` — `RegionTree` *shape* | model | transform | No in-repo *path*-model analog. `NonSymmetricTree` is the nearest struct but is a binary node graph — copying it is Pitfall 2 (wrong leaf assignments). The new `RegionTree` must model the upstream `TRegionModel` path (per-level `(feature,bin,expected_direction,one_hot)`, `depth+1` leaves) from §6.6 `AddRegionImpl` directly. The ENUM/builder/apply *plumbing* around it has strong analogs (`TreeVariant`/`from_trained`/`leaf_index_nonsym`); only the leaf-walk semantics are net-new. |

---

## Metadata

**Analog search scope:** `crates/cb-backend/src/kernels/`, `crates/cb-backend/src/gpu_runtime/`, `crates/cb-compute/src/{runtime,leaf}.rs`, `crates/cb-train/src/{tree,boosting,bootstrap}.rs`, `crates/cb-train/src/ctr/`, `crates/cb-model/src/{model,apply,json}.rs`.
**Files scanned:** ~20 source files read (key ranges only, no re-reads).
**Upstream authority (transcribe math from, do not copy structure):** `CATBOOST_CUDA_KERNELS_DESIGN.md` §5.1–5.7/§6.1/§6.3/§6.4/§6.6 + vendored `catboost-master/catboost/cuda/`.
**Pattern extraction date:** 2026-07-03
