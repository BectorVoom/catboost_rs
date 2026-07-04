# Roadmap: catboost-rs

## Milestones

- ✅ **v1.0 Core Parity** — Phases 1–8 (shipped 2026-06-28)
- 🚧 **v1.1 GPU Performance** — Phases 10–14 (planning) — full CUDA device-resident training parity; ALL GPU kernel oracles (correctness + speed) validated on a Kaggle CUDA notebook, with a per-phase speed check from the first GPU phase to the last. Re-scoped in place 2026-07-02 against `CATBOOST_CUDA_KERNELS_DESIGN.md` (17 → 25 requirements; the full upstream CUDA training-kernel surface).

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

### 🚧 v1.1 GPU Performance (Phases 10–14)

**Milestone goal:** Move the entire boosting inner loop (histogram build, split scoring, BestSplit, partition/leaf-assignment, leaf values) onto the GPU — not just derivatives — closing the >20× gap vs official CatBoost GPU while preserving correctness. The v1.1 re-scope (2026-07-02) maps the complete upstream CUDA training-kernel surface documented in `CATBOOST_CUDA_KERNELS_DESIGN.md` (79 `.cu` + 77 `.cuh` across 9 kernel directories); every phase below cites it.

**Validation authority — ALL GPU (CUDA) kernel oracles, correctness AND speed, run on a Kaggle CUDA notebook.** CUDA is the single authoritative GPU oracle for this milestone. A reproducible Kaggle CUDA oracle/test harness (BENCH-01) is a **foundational deliverable established in Phase 10** that measures BOTH correctness AND wall-clock speed from the start, so every GPU kernel — from the depth-1 device tree onward — is both correctness-tested and speed-measured on CUDA, not merely speed-benchmarked at the end. There is no NVIDIA hardware in-env; the AMD/ROCm in-env GPU remains an **optional compile/smoke convenience** for fast local iteration, but it is **not a gate** — no requirement is satisfied by ROCm validation alone. The prior "validate correctness in-env on ROCm, benchmark speed on CUDA" asymmetry is GONE.

**Speed is checked for EVERY GPU kernel in EVERY phase — from the first to the last.** BENCH-02 is a **standing per-phase speed check** (mapped to Phase 10 where it is first established, but enforced in every GPU phase 10→13 — analogous to how GPUT-14's ε=1e-4 gate is mapped to Phase 11 but enforced onward). Every phase that lands GPU kernels reports a Kaggle CUDA wall-clock speed measurement for that phase's kernels — device path vs the host-CPU baseline, and vs official CatBoost GPU where a comparable config exists (warm-run/JIT-excluded, train-only). No phase's GPU kernels are "done" without a recorded CUDA speed check. Phase 14 is the **comprehensive final** parity sign-off that AGGREGATES the per-phase checks — NOT the first place speed is measured.

**Practical note:** a Kaggle CUDA oracle/speed run is a **human-gated external step** — the user builds the `--features cuda` wheel and runs the notebook. GPU oracle + speed verification for every phase below is therefore a human-gated Kaggle CUDA execution, not an in-CI automated check.

**Parity bar:** the GPU device path is held to **ε=1e-4 vs the Rust CPU path** (Phase 7.6 precedent — device math is f32; bit-exact f64 ≤1e-5 is not the GPU goal), with the depth-1 device tree held tighter at **≤1e-5** where the whole-dataset level-0 histogram is the exact CPU score. The CPU path stays oracle-locked ≤10⁻⁵ and byte-unchanged (D-04 no-regression).

**Standing landmines (carry into every phase):**

- **Never add a `cb-train` dependency to `cb-backend`** — Cargo feature unification breaks the rocm runtime; transcribe CPU references inline. The `Runtime` seam stays CubeCL-free (plain host structs cross the boundary).
- **No `-inf` float literals inside `#[cube]` kernels** — HIP JIT rejects them on gfx1100 (CUDA accepts them, so this is a portability nicety the ROCm smoke build catches); use the `f32::MIN` sentinel so kernels stay portable cuda/rocm/wgpu.
- **Reduction determinism still governs the ε=1e-4 bar.** CUDA *does* provide f64 atomic-add (unlike gfx1100), so the atomic-free constraint is now a portability nicety rather than a hard correctness gate — BUT `atomicAdd` commit ordering is still non-deterministic and compounds over hundreds of trees, so a **deterministic reduction strategy is still required** to hold ε=1e-4 parity on CUDA. (gfx1100 still lacks f64 atomic-add, so atomic-free design also keeps the optional ROCm smoke path device-resident.)
- **Never read a `Handle` through a client other than the one that allocated it** (CubeCL residency rule).
- **The `Ok(None)`→host-CPU fallback gate** keeps every increment oracle-safe: any case not yet passing device sign-off on Kaggle CUDA falls back to the CPU path (the correctness reference and safety net). Per-fit, all-or-nothing (D-10-01): no mid-run mixing of device-grown and CPU-grown trees in one model.

#### Phase Checklist

- [x] **Phase 10: GPU Foundations — Runtime Seam, Session Residency, Device-Primitive Library, Compressed Index, Depth-1 + Kaggle CUDA Oracle & Speed Harness** - The from-scratch CubeCL device-primitive substrate (no CUB) and the device-resident compressed index land; the device grow loop becomes reachable from `fit()`; training data stays device-resident; depth-1 oblivious trees grow on device with the Cosine GPU-default score; the foundational Kaggle CUDA harness measures BOTH correctness (≤1e-5) AND wall-clock speed from the start. (completed 2026-07-03)
- [ ] **Phase 11: Depth>1 Partition-Aware Histograms + Reduction Determinism + Newton Der2** - Real depth-6 RMSE + Logloss workloads grow fully on device within ε=1e-4, via partition-aware histograms + the subtraction trick + a deterministic reduction + Newton der2 leaf estimation, oracle-tested AND speed-measured on Kaggle CUDA; GPUT-14 becomes the operative standing gate.
- [x] **Phase 12: Grow-Policy, Leaf-Method, Sampling & Categorical Device Coverage** - Non-symmetric grow policies (Depthwise/Lossguide/Region), Exact weighted-quantile leaf estimation, bootstrap/random-strength + MVS sampling, and CTR/categorical features each transition to the device path (ε=1e-4 on Kaggle CUDA) behind the `Ok(None)` fallback gate, each timed on Kaggle CUDA as it lands. (completed 2026-07-04)
- [ ] **Phase 13: Pairwise, Ranking, Multiclass, Ordered & Langevin Device Coverage** - The PairLogit pairwise path + batched device Cholesky solver, query/listwise ranking objectives with device query-grouping, multiclass/multi-target/uncertainty, ordered boosting, and Langevin/SGLB noise each transition to the device path (ε=1e-4 on Kaggle CUDA) behind the `Ok(None)` fallback gate, each timed on Kaggle CUDA as it lands.
- [ ] **Phase 14: Comprehensive Kaggle CUDA Speed Benchmark + Parity Sign-Off** - The device-resident path demonstrably closes the >20× gap via a comprehensive final sign-off that AGGREGATES the per-phase speed checks (BENCH-02) from Phases 10–13, with CUDA correctness gated before any speed number.

## Phase Details

### Phase 10: GPU Foundations — Runtime Seam, Session Residency, Device-Primitive Library, Compressed Index, Depth-1 + Kaggle CUDA Oracle & Speed Harness

**Goal**: Lay the whole device-resident substrate the milestone stands on. The from-scratch CubeCL device-primitive library (§6.1/§6.2 — no CUB in CubeCL) and the bit-packed device-resident compressed index (§6.6a) land as the single input every later histogram/scoring kernel consumes; the existing-but-unused device grow loop becomes reachable from `cb_train::train`; training data stays device-resident across iterations (upload once, no per-tree re-upload, no per-tree `der1` read-back); a depth-1 oblivious tree grows fully on device with the Cosine GPU-default score and matches the CPU path bit-for-bit (≤1e-5); and the reproducible **Kaggle CUDA harness** is established as a foundational deliverable measuring BOTH correctness AND wall-clock speed from day one. This phase establishes the seam, the residency architecture, the primitive substrate, the cindex, the `Ok(None)` per-fit fallback pattern, the CUDA oracle of record, AND the standing per-phase speed check that every later phase reuses.
**Depends on**: Phase 7 (GPU structural parity — `grow_boosting_pass`, der seam, `*_into` launchers, pointwise/pairwise histogram kernels), Phase 8 (`GpuBackend` over `SelectedRuntime`, `--features cuda` wheel)
**Requirements**: GPUT-01, GPUT-02, GPUT-03, GPUT-04, GPUT-08, GPUT-15, GPUT-16, BENCH-01, BENCH-02 (standing — first established here, enforced in every GPU phase 10→13)
**Success Criteria** (what must be TRUE):

  1. A from-scratch **CubeCL-portable device-primitive library** — fill/transform (gather-scatter, vector arithmetic), full + segmented prefix scan, reduce / segmented-reduce / reduce-by-key, radix sort + stable single-bit reorder, bit-compression, `TDataPartition` offset/size update, and per-partition stat aggregation (`update_part_props`) — runs on device with a deterministic reduction and matches the CPU path ≤1e-4, **oracle-tested on Kaggle CUDA** (§6.1 `cuda_util/kernel`, §6.2 `cuda_util/kernel/sort`; no CUB — real deliverables, not wrappers).
  2. A bit-packed **device-resident compressed index (cindex)** with `TCFeature` Offset/Shift/Mask/OneHot addressing is built and kept resident as the single input to the histogram kernels, matching the CPU quantized layout ≤1e-4 on Kaggle CUDA (borders stay host — CPU quantization is the ≤1e-5 reference; §6.6a `gpu_data/kernel/binarize.cu`, `WriteCompressedIndex`).
  3. A depth-1 oblivious tree (RMSE/Logloss, Plain boosting, fold_count=1) grown fully on device using the Cosine / second-order GPU-default score function matches the CPU path ≤1e-5, **oracle-tested on Kaggle CUDA** (a human-gated `--features cuda` notebook run) — not merely on the optional ROCm smoke build.
  4. The `Runtime` grow-tree seam (`begin_device_training` / `grow_tree_on_device → CbResult<Option<DeviceGrownTree>>` / `end_device_training`, CubeCL-free host-typed) is reachable from `cb_train::train`; the quantized feature matrix uploads exactly once per `fit()`; gradients/approx stay device-resident across iterations; the per-tree `der1` host read-back is eliminated; only the O(1) BestSplit descriptor + `2^depth` partition statistics cross host↔device per level (GPUT-01/02/03, D-05).
  5. A reproducible **Kaggle CUDA harness (BENCH-01)** builds the `--features cuda` wheel and on a Kaggle CUDA notebook runs BOTH the correctness oracle (≤1e-5 depth-1, correctness as a blocking gate) AND a warm-run/JIT-excluded train-only **wall-clock speed** measurement (device vs host-CPU baseline, and vs official CatBoost GPU where a comparable config exists) — establishing the **standing per-phase speed check (BENCH-02)** discipline enforced from here to the last phase; an uncovered case (e.g. depth>1) returns `Ok(None)` and falls back to the byte-unchanged host CPU grower (D-04 no-regression), and the reduction-determinism strategy is spiked with a recommendation documented to feed Phase 11.

**Plans**: 9/9 plans complete
**Wave 1**

- [x] 10-01-PLAN.md — Scan primitives: cross-cube full scan + segmented scan (GPUT-16)
- [x] 10-02-PLAN.md — Runtime grow-tree seam contract + DeviceGrownTree (GPUT-01)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 10-03-PLAN.md — Reduce primitives (seg-reduce/reduce-by-key) + reduction-determinism spike (GPUT-16, D-03/04)

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 10-04-PLAN.md — Sort/single-bit reorder + TDataPartition update + fill/transform (GPUT-16)

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 10-05-PLAN.md — Bit-compression pack/unpack + update_part_props (GPUT-16)

**Wave 5** *(blocked on Wave 4 completion)*

- [x] 10-06-PLAN.md — Bit-packed device-resident cindex + read_bin accessor migration (GPUT-15)

**Wave 6** *(blocked on Wave 5 completion)*

- [x] 10-07-PLAN.md — Depth-1 device tree + Cosine default + apply_leaf_delta + GpuTrainSession residency + backend seam impls (GPUT-02/03/04/08)

**Wave 7** *(blocked on Wave 6 completion)*

- [x] 10-08-PLAN.md — Boosting-loop device branch + bin→border join + Ok(None) fallback (GPUT-01/04)

**Wave 8** *(blocked on Wave 7 completion)*

- [x] 10-09-PLAN.md — Kaggle CUDA oracle + speed harness + reduction-spike numbers + D-10-09 escalation (BENCH-01/02)

**UI hint**: no
**Notes**: The heaviest, most foundational phase — it carries the two biggest hidden risks (GPUT-16 device-primitive library, GPUT-15 cindex) alongside the seam+residency+depth-1 wiring and the harness. ~80% of the depth-1 device machinery already exists and is oracle-validated (Phase 7.5 `grow_oblivious_tree_into` grows a depth-1 device tree with L2/Cosine calcers; Phase 7.2 der seam; `calc_average` leaf formula) — the seam mirrors the shipped `compute_gradients_grouped` default-impl pattern. The genuinely new engineering is: the from-scratch primitive substrate (no CUB), the resident cindex, the `GpuTrainSession` residency wrapper owning one `ComputeClient` + persistent handles (`RefCell<Option<…>>` on `GpuBackend`), one small `apply_leaf_delta` device kernel to keep the approx-update on device, the bin→border join (`border = feature_borders[feature][bin_id]`), the Kaggle harness, and the reduction spike. **ESCALATION (D-10-09):** depth-1 device ≥ CPU wall-clock is achievable only at large n (≈10⁵–10⁶+ rows), NOT at `benchmark.py`'s 1000×20 (depth-1 is the most launch-overhead-bound workload) — pin BENCH-02's depth-1 speed bar to a large-n dataset. Logloss depth-1 ≤1e-5 pins the CPU-reference fixture to first-order (`calc_average`) leaves (Newton der2 is Phase 11). Landmines: no `cb-train` dep in `cb-backend`; `f32::MIN` sentinel; deterministic reduction still required. Prior research: `.planning/milestones/v1.1-rescope-2026-07-02-phases/10-.../10-RESEARCH.md` (seam/residency/depth-1 architecture still valid; re-plan to add GPUT-08/15/16 scope). Given the scope, plans will decompose this phase into several waves (primitive library → cindex → seam+residency → depth-1+Cosine → Kaggle harness → reduction spike).

### Phase 11: Depth>1 Partition-Aware Histograms + Reduction Determinism + Newton Der2

**Goal**: Real-world depth-6 workloads — both RMSE regression and Logloss classification — grow fully on device within ε=1e-4, oracle-tested AND speed-measured on Kaggle CUDA. This is the performance keystone: depth 6 is the default, the histogram subtraction trick is required to approach parity speed, a deterministic reduction is required to hold the bar across hundreds of trees, and Newton der2 is required for classification. GPUT-14 (ε=1e-4 device bar on Kaggle CUDA + D-04 CPU no-regression) becomes the operative standing gate from this phase forward — depth-1 was held to ≤1e-5, but reductions over hundreds of depth>1 trees move the bar to ε=1e-4.
**Depends on**: Phase 10 (the seam, residency, device-primitive library, cindex, AND the Kaggle CUDA oracle + speed harness)
**Requirements**: GPUT-05, GPUT-06, GPUT-07, GPUT-14
**Success Criteria** (what must be TRUE):

  1. A depth>1 (e.g. depth 6) RMSE tree grows on device via partition-aware histograms (`fullPass=false`) keyed by leaf + contiguous partition reorder + the histogram subtraction trick (§6.3/§6.4), matching the CPU path ≤1e-4 **oracle-tested on Kaggle CUDA**.
  2. A Logloss classification model with Newton der2 leaf estimation trains fully on device ≤1e-4 vs the Rust CPU path, oracle-tested on Kaggle CUDA (Newton der2 reuses the Phase 7.2 der2 handles; required for the classification / Logloss default).
  3. The chosen deterministic reduction strategy (recommended by the Phase-10 spike) holds device histogram and score reductions within ε=1e-4 of the CPU path across hundreds of trees on CUDA, with no split flips compounding over the boosting run (CUDA's f64 atomic-add is available, but `atomicAdd` ordering is still non-deterministic, so the deterministic reduction is what holds the bar).
  4. **Speed check (BENCH-02, standing):** depth-6 RMSE and Logloss device training is timed on Kaggle CUDA and reported as device path vs the host-CPU baseline AND vs official CatBoost GPU (warm-run/JIT-excluded, train-only) — this phase's keystone kernels (partition-aware histograms + subtraction trick + Newton der2) carry their own recorded CUDA speed measurement, not deferred to Phase 14.
  5. Every device-covered case to date holds ε=1e-4 vs the Rust CPU path **on Kaggle CUDA**, and the CPU/host training paths remain byte-unchanged (D-04) — the standing GPUT-14 gate, operative from here to the end of the milestone.

**Plans**: 4/5 plans executed

- [x] 11-01-PLAN.md — Depth-6 synthetic fixture generator (D-03) + CPU oracle cross-check (Wave 1)
- [x] 11-02-PLAN.md — Partition-aware `fullPass=false` histogram + subtraction trick + deterministic fixed-point accumulator (GPUT-05/06, Wave 2)
- [x] 11-03-PLAN.md — Wire depth>1 into the grow loop + depth-6 grow self-oracle + zero-spread determinism check (GPUT-05/06, Wave 3)
- [x] 11-04-PLAN.md — Newton der2 leaf estimation (Σder2 channel, newton_leaf_delta, apply_leaf_delta refinement) (GPUT-07, Wave 4)
- [ ] 11-05-PLAN.md — Kaggle CUDA harness: final-ε=1e-4 gate + per-tree diagnostic + BENCH-02 speed (GPUT-14/06/BENCH-02, Wave 5, human-gated)

**UI hint**: no
**Notes**: The single largest kernel extension of the milestone — partition-aware `pointwise_hist2` keyed by `leaf_of[obj]` into `2^level` slots; `TDataPartition{Offset,Size}` contiguous layout; parent-resident sibling-by-subtraction (§1.4 subtraction trick, §6.3 `pointwise_hist2`, §6.4 leaf-wise builder). Consumes the Phase-10 device-primitive library (scan/segmented-scan/reduce-by-key/partition-update) and the resident cindex directly. Reuses Phase 7.2 der2 handles for Newton. Oracle + speed of record = Kaggle CUDA (human-gated notebook); the optional ROCm in-env build is a fast local smoke check only. **BENCH-02's standing per-phase speed check applies here.** Landmines: deterministic reduction strategy mandatory (CUDA atomicAdd ordering still non-deterministic; gfx1100 still lacks f64 atomic-add for the smoke path); `f32::MIN` sentinel. Research flags: consume the Phase-10 `SPIKE-REDUCTION.md` recommendation as step 0 (before the histogram kernel); verify the multi-block scan carry against the vendored CubeCL manual. Given the kernel complexity, plans may decompose this phase into sub-waves (depth>1 histograms → reduction determinism → Newton der2).

### Phase 12: Grow-Policy, Leaf-Method, Sampling & Categorical Device Coverage

**Goal**: Expand the device path beyond the symmetric-tree / Newton-leaf / uniform-numeric default across the tree-growth-mechanics families: the non-symmetric grow policies (Depthwise/Lossguide/Region), Exact weighted-quantile leaf estimation, bootstrap + random-strength + MVS sampling, and CTR/categorical features. Each transitions from `Ok(None)`→CPU-fallback to `Ok(Some(tree))`→device path, independently and behind the same per-fit fallback gate, each gated by a Kaggle CUDA ε=1e-4 sign-off AND timed on Kaggle CUDA as it lands. Recommended sub-order: non-symmetric grow policies → Exact leaf estimation → bootstrap/random-strength (small, high-return) → MVS (CatBoost's default GPU sampler) → CTR (headline categorical use case). Each sub-feature lands only when it passes ε=1e-4 sign-off on Kaggle CUDA; until then users transparently fall back to the CPU path.
**Depends on**: Phase 11 (depth>1 partition-aware histograms + reduction determinism — non-symmetric policies and CTR both build on the depth>1 histogram/partition machinery)
**Requirements**: GPUT-18, GPUT-19, GPUT-09, GPUT-17, GPUT-10
**Success Criteria** (what must be TRUE):

  1. The **Depthwise, Lossguide, and Region** grow policies — per-policy leaf selection (`ComputeOptimalSplitsRegion` / `ComputeOptimalSplit` + `SelectLeavesToSplit`) and region/non-symmetric tree leaf-value apply (`AddRegion` / `ComputeNonSymmetricDecisionTreeBins`) — run on device, matching the CPU path ≤1e-4 **oracle-tested on Kaggle CUDA** (GPUT-18; §6.4, §6.6c).
  2. **Exact** weighted-quantile leaf-value estimation (`exact_estimation`: needWeights = totalWeight·α, binary search over per-bin weight prefix sums) runs on device for Quantile/MAE/MAPE-family objectives, matching the CPU path ≤1e-4 on Kaggle CUDA (GPUT-19; §6.3 `exact_estimation`, distinct from the Newton path).
  3. Bootstrap + random-strength sampling (GPUT-09) and **Minimal Variance Sampling (MVS)** — per-block optimal threshold on `sqrt(der²+λ)` with inverse-probability reweighting, CatBoost's *default* GPU sampler (GPUT-17; §6.1 `mvs`) — run on device with sampling parity, matching the CPU path ≤1e-4 on Kaggle CUDA; CTR / permutation-dependent categorical features (GPUT-10) train on device ≤1e-4 on Kaggle CUDA.
  4. **Speed check (BENCH-02, standing):** each family (grow policies / Exact leaf / bootstrap / MVS / CTR) is timed on Kaggle CUDA **as it lands** — device path vs the host-CPU baseline, and vs official CatBoost GPU where a comparable config exists (warm-run/JIT-excluded, train-only) — so every family's kernels carry their own recorded CUDA speed measurement when they flip from `Ok(None)`→device, not deferred to Phase 14.
  5. Any sub-feature not yet passing Kaggle CUDA sign-off returns `Ok(None)`→CPU fallback (no incorrect device result), the CPU/host path stays byte-unchanged (GPUT-14/D-04), and the resulting GPU coverage matrix (correctness + per-family speed) is documented.

**Plans**: 9/9 plans complete

Plans:
**Wave 1**

- [x] 12-01-PLAN.md — Device foundation: DeviceGrownTree non-sym fields + DeviceTrainConfig + session depth>1 relaxation (A3) (GPUT-18, Wave 1)
- [x] 12-02-PLAN.md — CPU Region path FIRST: grower + TreeVariant::Region + apply + json + validate_grow_policy lift + frozen ≤1e-5 oracle (GPUT-18, Wave 1)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 12-03-PLAN.md — W1 Depthwise/Lossguide device emission (nonsym_grow) + boosting.rs device-fold non-sym arm (NonSymmetricTree) + self-oracle + gate arm + end-to-end fit oracle (GPUT-18, Wave 2)

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 12-04-PLAN.md — W2b device Region path + boosting.rs device-fold Region arm (RegionTree) vs frozen CPU Region oracle + gate arm + end-to-end fit oracle (GPUT-18, Wave 3)

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 12-05-PLAN.md — W3 Exact weighted-quantile leaf + segmented-sort primitive (A1) + gate arm (GPUT-19, Wave 4)

**Wave 5** *(blocked on Wave 4 completion)*

- [x] 12-06-PLAN.md — W4 bootstrap + random-strength device RNG (pin-seed/freeze) + gate arm (GPUT-09, Wave 5)

**Wave 6** *(blocked on Wave 5 completion)*

- [x] 12-07-PLAN.md — W5 MVS per-block threshold + reweight (default GPU sampler) + gate arm (GPUT-17, Wave 6)

**Wave 7** *(blocked on Wave 6 completion)*

- [x] 12-08-PLAN.md — W6 CTR device port (ordered/one-hot/tensor) + CTR→cindex join + gate arm (GPUT-10, Wave 7)

**Wave 8** *(blocked on Wave 7 completion)*

- [x] 12-09-PLAN.md — Kaggle CUDA ε=1e-4 sign-off + per-family BENCH-02 + SC-5 coverage matrix (human-gated, Wave 8)

**UI hint**: no
**Notes**: Each family is independently shippable and deferrable — can be planned/executed as parallel sub-workstreams, or cut to a bootstrap + CTR MVP if Phase 11 runs long. Every per-family sign-off is a human-gated Kaggle CUDA oracle run (reusing the Phase-10 harness) AND a Kaggle CUDA speed measurement; the optional ROCm smoke build is a local convenience only. **BENCH-02's standing per-phase speed check applies here.** Research flags: CTR on device has the highest uncertainty (targeted read of `batch_binarized_ctr_calcer.h` + `ctrs/kernel/` before planning that sub-task); MVS threshold + reweight (§6.1 `mvs.{cu,cuh}`) and the Region/non-symmetric apply (§6.6c `models/kernel`) are the other high-uncertainty sub-tasks. Plans may sub-split this phase (e.g. 12.1 grow policies, 12.2 Exact leaf, 12.3 sampling+MVS, 12.4 CTR) given the kernel breadth. Landmine constraints unchanged.

### Phase 13: Pairwise, Ranking, Multiclass, Ordered & Langevin Device Coverage

**Goal**: Expand the device path across the loss-family / multi-output / ordered-residency families: the PairLogit pairwise path with its 2×2-cell histograms + a batched device Cholesky solver, the query/listwise ranking objectives with device query-grouping infrastructure, multiclass/multi-target/uncertainty, ordered boosting, and Langevin/SGLB noise. Each transitions from `Ok(None)`→CPU-fallback to `Ok(Some(tree))`→device path, independently and behind the same per-fit fallback gate, each gated by a Kaggle CUDA ε=1e-4 sign-off AND timed on Kaggle CUDA as it lands. Recommended sub-order: PairLogit + Cholesky solver (reuses Phase 7.4 pairwise-histogram kernels) → query/listwise ranking (query-grouping infra) → multiclass/multi-target/uncertainty → ordered boosting (heaviest residency) → Langevin/SGLB noise (small, layers on the reduced derivatives).
**Depends on**: Phase 11 (depth>1 device-resident grow loop + reduction determinism); reuses Phase 12 coverage-fallback patterns (each family independent behind `Ok(None)`)
**Requirements**: GPUT-11, GPUT-21, GPUT-22, GPUT-12, GPUT-13, GPUT-20
**Success Criteria** (what must be TRUE):

  1. The **PairLogit** pairwise-loss training path (pairwise 2×2-cell histograms, §6.3 `pairwise_hist*`, reusing Phase 7.4 kernels) plus per-leaf **pairwise-derivative matrix assembly** (`MakePairwiseDerivatives` / `MakePointwiseDerivatives`) and a **batched device Cholesky** decomposition + forward/back substitution + ridge regularization + score-from-decomposition (`CalcScoresCholesky`, §6.3 `split_pairwise`/`linear_solver`) run on device for pairwise split-scoring and leaf values, matching the CPU path ≤1e-4 **oracle-tested on Kaggle CUDA** (GPUT-11, GPUT-21).
  2. The **query-wise / listwise** objectives — QueryRMSE, QuerySoftMax, QueryCrossEntropy, YetiRank, PFound-F — with device query-grouping infrastructure (group ids/means/max, group-bias removal, in-query sampling radix sort, taken-docs masks; §6.5 `query_*`, `yeti_rank_pointwise`, `pfound_f`; §6.6a `query_helper.cu`) run on device ≤1e-4 on Kaggle CUDA (GPUT-22).
  3. The **multiclass / multi-target / uncertainty** path (MultiClass, MultiClassOneVsAll, MultiCrossEntropy, MultiRMSE, RMSEWithUncertainty — multilogit multi-row der2 blocks, §6.5 `multilogit`) trains on device ≤1e-4 (GPUT-12); **ordered boosting** (`EBoostingType::Ordered`) trains on device ≤1e-4 (GPUT-13); and **Langevin/SGLB** noise (`AddLangevinNoise`: per-element seeded Gaussian on the reduced derivatives, §6.3 `langevin_utils`) runs on device ≤1e-4 (GPUT-20) — all oracle-tested on Kaggle CUDA.
  4. **Speed check (BENCH-02, standing):** each family (pairwise / ranking / multiclass / ordered / langevin) is timed on Kaggle CUDA **as it lands** — device path vs the host-CPU baseline, and vs official CatBoost GPU where a comparable config exists (warm-run/JIT-excluded, train-only) — so every family's kernels carry their own recorded CUDA speed measurement when they flip from `Ok(None)`→device, not deferred to Phase 14.
  5. Any sub-feature not yet passing Kaggle CUDA sign-off returns `Ok(None)`→CPU fallback (no incorrect device result), the CPU/host path stays byte-unchanged (GPUT-14/D-04), and the resulting GPU coverage matrix (correctness + per-family speed) is documented — completing the device-coverage surface feeding the Phase-14 aggregate sign-off.

**Plans**: TBD
**UI hint**: no
**Notes**: The loss-family coverage cluster, split out of Phase 12 so neither coverage phase is overloaded. Each family is independently shippable and deferrable, plannable as parallel sub-workstreams. Every per-family sign-off is a human-gated Kaggle CUDA oracle run (reusing the Phase-10 harness) AND a Kaggle CUDA speed measurement. **BENCH-02's standing per-phase speed check applies here.** Research flags: the pairwise partition + leaves oracle (`leaves_estimation/pairwise_oracle.h`) is under-documented — read it before implementing; the batched Cholesky solver (§6.3 `linear_solver`, batched over leaves with ridge) and the query-grouping infra (§6.5/§6.6a) are the highest-uncertainty sub-tasks; ordered boosting is the heaviest residency. Plans may sub-split this phase (e.g. 13.1 pairwise+solver, 13.2 ranking, 13.3 multiclass, 13.4 ordered, 13.5 langevin). Landmine constraints unchanged.

### Phase 14: Comprehensive Kaggle CUDA Speed Benchmark + Parity Sign-Off

**Goal**: Prove the device-resident path demonstrably closes the >20× gap vs official CatBoost GPU via a **comprehensive final speed-parity sign-off (BENCH-03) that AGGREGATES the per-phase speed checks (BENCH-02)** recorded in Phases 10–13 — with CUDA correctness established as a blocking gate before any speed number is trusted. This phase does NOT re-establish the oracle or measure speed for the first time (Phases 10–13 already measured per-phase speed via the standing BENCH-02 check); it extends the Phase-10 Kaggle CUDA harness to a comprehensive head-to-head timing across the workload matrix and the milestone-closing ε sign-off.
**Depends on**: Phase 10 (the Kaggle CUDA oracle + speed harness it builds on), Phase 11 (depth>1 device-resident + its recorded speed check), Phase 12 (grow-policy/leaf/sampling/CTR families + their per-family speed checks), Phase 13 (pairwise/ranking/multiclass/ordered/langevin families + their per-family speed checks)
**Requirements**: BENCH-03
**Success Criteria** (what must be TRUE):

  1. The Phase-10 Kaggle CUDA harness times official CatBoost GPU vs catboost-rs across the full workload matrix on identical datasets/params, with warm-run/JIT exclusion and train-only (not I/O) wall-clock measurement, **aggregating the per-phase speed checks (BENCH-02)** from Phases 10–13 into one comprehensive comparison.
  2. The correctness oracle is re-confirmed on the CUDA backend (≤1e-4 vs the Rust CPU path, ≤1e-5 for the depth-1 tree) as a **blocking gate before any speed number is reported** — reusing the authoritative Phase-10 CUDA oracle, so a fast-but-wrong CUDA result is never quoted.
  3. The device-resident path demonstrably closes the >20× gap (BENCH-03): a documented, signed-off **comprehensive final** speed-parity result vs official CatBoost GPU on Kaggle CUDA, measured against the pre-Phase-10 host-light baseline and aggregating every per-phase speed measurement into the milestone-closing sign-off.

**Plans**: TBD
**UI hint**: no
**Notes**: This is the comprehensive FINAL aggregate, not the first place speed is measured — Phases 10–13 each carried their own standing BENCH-02 Kaggle CUDA speed check; Phase 14 rolls them up into one head-to-head sign-off. Protocol fully specified (STACK.md / PITFALLS.md / existing `benchmark.py` template + the Phase-10 `bench/` harness). This is a human-gated Kaggle CUDA execution. Execution checklist: verify CUDA backend active via `nvidia-smi`, warm one untimed fit, drain the lazy CubeCL queue with a read-back/predict before stopping the clock, re-run the oracle (Phase-10 harness) before timing. Standard patterns — no new compute crates; `criterion 0.7.x` (dev-dep) for optional in-env ROCm relative-timing regression during development, optional `profile-tracy`/`tracing` behind a `profiling` Cargo feature.

## Progress (v1.1)

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 10. GPU Foundations — Seam + Residency + Primitive Library + cindex + Depth-1 + Kaggle CUDA Harness | 9/9 | Complete   | 2026-07-03 |
| 11. Depth>1 Histograms + Reduction Determinism + Newton Der2 | 4/5 | In Progress|  |
| 12. Grow-Policy, Leaf-Method, Sampling & Categorical Coverage | 9/9 | Complete   | 2026-07-04 |
| 13. Pairwise, Ranking, Multiclass, Ordered & Langevin Coverage | 0/TBD | Not started | - |
| 14. Comprehensive Kaggle CUDA Benchmark + Sign-Off | 0/TBD | Not started | - |

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

</content>
</invoke>
