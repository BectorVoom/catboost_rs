# Phase 10: GPU Foundations — Runtime Seam, Session Residency, Device-Primitive Library, Compressed Index, Depth-1 + Kaggle CUDA Oracle & Speed Harness — Research

**Researched:** 2026-07-03
**Domain:** GPU device-resident gradient-boosting training (CubeCL) — from-scratch device-primitive library (no CUB), bit-packed compressed index, Runtime grow-tree seam + session residency, depth-1 oblivious tree + Cosine score, Kaggle CUDA oracle/speed harness
**Confidence:** HIGH on codebase state and reuse surface (direct read this session); HIGH on the upstream primitive/cindex algorithm map (`CATBOOST_CUDA_KERNELS_DESIGN.md`); MEDIUM on new-kernel first-pass correctness (every `#[cube]` change needs the rocm smoke) and depth-1 device>CPU feasibility (analytical — only Kaggle CUDA settles it)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Scope anchor (already locked by ROADMAP/REQUIREMENTS — not re-decided):**
- Seam signatures: `begin_device_training` / `grow_tree_on_device → CbResult<Option<DeviceGrownTree>>` / `end_device_training`, CubeCL-free host-typed (GPUT-01).
- `GpuTrainSession` owns one `ComputeClient` + persistent handles for the whole fit; `RefCell<Option<…>>` on `GpuBackend`; upload-once, no per-tree re-upload (GPUT-02/03).
- `Ok(None)` → host-CPU fallback; per-fit **all-or-nothing** (D-10-01) — no mixing device-grown and CPU-grown trees in one model.
- ε bars: depth-1 device **≤1e-5**; everything else **ε=1e-4** vs Rust CPU path. CPU path stays oracle-locked ≤1e-5 and byte-unchanged (D-04 no-regression).
- Depth-1 is the MVP; **depth>1 returns `Ok(None)`** → host CPU grower (Phase 11).
- Logloss depth-1 pins the CPU-reference fixture to **first-order `calc_average` leaves** (Newton der2 is Phase 11).
- Only the O(1) BestSplit descriptor + `2^depth` partition stats cross host↔device per level (D-05).
- Depth-1 speed bar pinned to a **large-n** dataset (~1e5–1e6 rows, D-10-09).
- All GPU oracles (correctness AND speed) authoritative on **Kaggle CUDA**, human-gated. ROCm in-env = optional compile/smoke, **not a gate**.

**Device-primitive library oracle (GPUT-16):**
- **D-01 (tiered oracle):** Hybrid. **Standalone** Kaggle CUDA oracles for the high-risk primitives — full scan, segmented scan, radix sort + stable single-bit reorder, reduce-by-key, per-partition stat aggregation (`update_part_props`). **Trivial** primitives (fill/transform gather-scatter + vector arithmetic, plain reduce) covered transitively through the depth-1 tree + cindex end-to-end.
- **D-02 (ground truth):** Standalone primitive oracles compare against a **self-contained serial CPU/numpy reference** on the same random-seeded input — NOT upstream CatBoost/CUB fixtures. Transcribe any CPU reference inline (no `cb-train` dep).

**Reduction-determinism spike (feeds Phase 11):**
- **D-03 (prototype + measure):** Implement the top 2–3 candidate deterministic-reduction strategies and measure run-to-run correctness variance AND speed on Kaggle CUDA, then recommend. Candidates: fixed-order tree reduce, sequential block-then-host-final-sum (Phase 7.6 `HostSumFallback`), Kahan compensation, sorted-index accumulation, fixed-point i64/u64 atomics.
- **D-04 (winner ships as the primitive):** The measured-best strategy **IS** the reduce/segmented-reduce/reduce-by-key that lands in the Phase 10 library — spike and deliverable are the same work, no throwaway.

**Kaggle CUDA harness (BENCH-01):**
- **D-05 (form + fixtures):** A **committed `.ipynb` notebook** builds the `--features cuda` wheel, loads **repo-committed fixtures** (random-seeded inputs + CPU-path expected values), runs **correctness first (blocking gate)** then a **warm-run / JIT-excluded, train-only wall-clock** speed measurement, prints a **structured report**.
- **D-06 (speed workload dataset):** A **seeded synthetic generator** (configurable `n_rows`/`n_features`, ~1e6×50, tunable above launch-overhead break-even) produces BOTH the depth-1 ≤1e-5 correctness fixture and the large-n speed workload. No external download; one generator for both.

**Compressed index packing (GPUT-15):**
- **D-07 (exact packing from the start):** Replicate upstream `WriteCompressedIndex`'s **exact 32-bit bit-packed grouped layout** — `TCFeature` Offset/Shift/Mask/OneHot addressing packing multiple features per 32-bit word — from the start; do NOT ship a simpler one-value-per-slot layout first. Borders stay host (CPU quantization is the ≤1e-5 reference); only cindex packing/residency is the device deliverable; validated against CPU quantized layout ≤1e-4.

### Claude's Discretion
- Wave decomposition/ordering (ROADMAP suggests: primitive library → cindex → seam+residency → depth-1+Cosine → Kaggle harness → reduction spike) — planner refines.
- Seam module placement in `cb-compute` (`runtime.rs`, mirroring `compute_gradients_grouped`), `apply_leaf_delta` device-kernel scope, per-fit session lifecycle details, the bin→border join (`border = feature_borders[feature][bin_id]`).
- Exact `DeviceGrownTree` fields and precise host-typed seam signatures (within GPUT-01's named shape).
- Internal `GpuTrainSession` handle layout / lifetime mechanics.
- Fixture file format and RESULTS sign-off log structure.
- Reduction-spike kernel sizes / problem shapes.

### Deferred Ideas (OUT OF SCOPE)
- Real named large datasets (Higgs/Epsilon) — Phase 14 (BENCH-03).
- On-device border/quantile computation (`FastGpuBorders`/`ComputeQuantileBorders`) — host CPU quantization is the ≤1e-5 reference, uploaded once.
- Newton der2 leaf estimation, depth>1 partition-aware histograms + subtraction trick, GPUT-14 ε=1e-4 operative gate — Phase 11.
- CTR / pairwise / multiclass / ordered-boosting device paths — Phase 12.
- Stale `.planning/spikes/MANIFEST.md` (Spike 001, online-CTR / Phase 5) — unrelated.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| GPUT-01 | `Runtime` grow-tree seam (`begin_device_training` / `grow_tree_on_device → CbResult<Option<DeviceGrownTree>>` / `end_device_training`), CubeCL-free host-typed, `Ok(None)`→CPU fallback | §Standard Stack, §Pattern 1 (default-impl mirroring `compute_gradients_grouped`), §Code Examples, PRIOR RESEARCH (still valid) |
| GPUT-02 | `GpuTrainSession` owns one `ComputeClient` + persistent handles; quantized matrix uploaded once per `fit()` | §Pattern 2 (`RefCell<Option<…>>`), §Pattern 3 (residency), §Don't Hand-Roll |
| GPUT-03 | Gradients/approx device-resident; per-tree `der1` read-back eliminated; only O(1) BestSplit + 2^depth part-stats cross per level | §Pitfall "grow_boosting_pass gap", §Pattern 3, §Code Examples (`apply_leaf_delta`) |
| GPUT-04 | Depth-1 oblivious tree (RMSE/Logloss, Plain, fold_count=1) grown on device ≤1e-5 vs CPU, oracle-tested on Kaggle CUDA | §Validation Architecture, §Pattern 4 (bin→border join), §Pitfall (Logloss leaf method) |
| GPUT-08 | Cosine / second-order score (the GPU default) runs on device | §NEW-3 (Cosine is ~90% already built — `SCORE_FN_COSINE` in `score_split` kernel + `cb-compute::score.rs`) |
| GPUT-15 | Bit-packed device-resident compressed index (cindex) with `TCFeature` Offset/Shift/Mask/OneHot addressing, matching CPU quantized layout ≤1e-4 | §NEW-2 (exact `WriteCompressedIndex` layout; ripples into `pointwise_hist` address arithmetic) |
| GPUT-16 | From-scratch CubeCL device-primitive library (no CUB) — fill/transform, full + segmented scan, reduce/segmented-reduce/reduce-by-key, radix sort + stable single-bit reorder, bit-compression, `TDataPartition` update, `update_part_props`, deterministic reduction | §NEW-1 (per-primitive spec + what already exists), §Reduction Determinism |
| BENCH-01 | Reproducible Kaggle CUDA oracle/speed harness — `--features cuda` wheel, correctness gate then speed | §NEW-5 (committed `.ipynb`), §Environment Availability |
| BENCH-02 | Standing per-phase speed check (enforced 10→13) — device vs host-CPU baseline | §NEW-5, §Depth-1 device>CPU Feasibility (ESCALATION) |
</phase_requirements>

## Summary

Phase 10 is a **substrate phase**: it lays the whole device-resident foundation the v1.1 milestone stands on. The prior research (`v1.1-rescope-2026-07-02-phases/10-.../10-RESEARCH.md`, dated 2026-06-29) fully covers the **seam + residency + depth-1** architecture (GPUT-01/02/03/04, BENCH-01/02) and is **still valid** — do not re-derive it. This research **adds** the three scopes that predate it: the **from-scratch primitive library (GPUT-16)**, the **bit-packed cindex (GPUT-15)**, and the **Cosine GPU-default score (GPUT-08)**.

**The single most useful finding:** ~80% of the depth-1 device machinery and a meaningful slice of the primitive library **already exist and are Phase-7-oracle-validated**. Direct code reads this session confirm: `grow_oblivious_tree_into` (`gpu_runtime/mod.rs:1641`) grows a depth-1 device tree over resident handles; the score/argmin kernel **already supports `SCORE_FN_COSINE`** (`score_split.rs:61,774`, `gpu_runtime/mod.rs:71`) cross-oracled against `cb-compute::score.rs`'s Cosine calcer — so **GPUT-08 is ~90% done** (wire Cosine as the depth-1 default, confirm the leaf path); a single-cube `block_scan_kernel` (`kernels/scan.rs`), a `block_reduce` with an explicit `HostSumFallback` deterministic finalize (`kernels/reduce.rs:271`), a `scatter` kernel, and a 2-channel `pointwise_hist` (`kernels/pointwise_hist.rs`) are already landed. The genuinely NEW / high-risk work is a focused ~20%.

**The three NEW bodies of work, by risk:**
1. **GPUT-16 primitive library (HIGHEST engineering content).** CubeCL has **no CUB**, so scan (with cross-cube carry — the existing scan is single-cube only, RESEARCH Open Q2), segmented scan, reduce-by-key, segmented reduce, radix sort + stable single-bit reorder, bit-compression, `TDataPartition` offset/size update, and `update_part_props` are real from-scratch deliverables. Each of the high-risk ones gets a standalone serial-CPU self-oracle (D-01/D-02).
2. **GPUT-15 cindex packing (HIGH ripple risk).** The current in-tree cindex is a **plain one-u32-per-cell** layout (`cindex[feature*n+obj]`, verified in `pointwise_hist.rs:97-103` and `grow_oblivious_tree_into`). GPUT-15 replaces it with the upstream **bit-packed grouped** `WriteCompressedIndex` layout addressed by `TCFeature{Offset,Mask,Shift,OneHot}`. That change **ripples into every histogram kernel's address arithmetic** — the reason D-07 says "get it right once."
3. **The depth-1 device≥CPU speed obligation (D-10-09, ESCALATION).** Carried forward from prior research: depth-1 is the most launch-overhead-bound workload in the whole milestone; device beats CPU only at large n. Pin BENCH-02 to a large-n synthetic workload (D-06) and escalate the small-n infeasibility.

**Primary recommendation:** Follow the ROADMAP wave order — (W1) primitive library GPUT-16 with per-primitive serial self-oracles → (W2) bit-packed cindex GPUT-15 → (W3) seam + residency GPUT-01/02/03 → (W4) depth-1 + Cosine GPUT-04/08 → (W5) Kaggle harness BENCH-01/02 → (W6) reduction-determinism spike (its winner IS the W1 reduce). Build the primitives FIRST because every later kernel stands on them; a broken scan found in isolation is far cheaper than debugging it through a depth-6 histogram later. Run the rocm smoke after **every** `#[cube]` change (it catches `-inf` literals and atomic non-determinism that cpu/wgpu/cuda `cargo check` hide).

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Device-primitive library (scan/reduce/sort/compress/partitions) | `cb-backend` (`gpu_runtime` + `kernels/`) | — | All `#[cube]` + `unsafe` lives here; never depends on `cb-train`. [VERIFIED: codebase — `cubecl` is a `cb-backend`-only dep, `cb-backend/Cargo.toml:19`] |
| Serial self-oracle per primitive (D-02) | `cb-backend/src/kernels/*.rs` (`#[cfg(test)]` separate files) | — | Transcribe CPU reference inline; no cross-crate reach. |
| Bit-packed cindex build + `TCFeature` addressing | `cb-backend` (`gpu_runtime`) | `cb-train` (supplies borders + plain bins host-side) | Packing/residency is the device deliverable; borders stay host (CPU quantization is the ≤1e-5 reference). |
| Grow-tree trait contract (host-typed) | `cb-compute` (`Runtime` trait) | — | Trait stays CubeCL-free so `cb-train` never pulls a backend dep (feature-unification landmine). |
| Coverage gate (depth-1/RMSE-or-Logloss/Plain/fold=1 → Some/None) | `cb-backend` (`GpuTrainSession::begin`) | `cb-compute` (seam return type) | D-10-02: classification lives where the session is created. |
| Device grow loop (hist+score+split+partition) | `cb-backend` (`gpu_runtime`) | — | Already exists (Phase 7.5, depth-1). |
| Cosine GPU-default score | `cb-backend` (`score_split` kernel — `SCORE_FN_COSINE` present) | `cb-compute::score.rs` (CPU Cosine oracle) | ~90% built; wire as default + confirm depth-1 usage. |
| Resident handles (cindex/indices/weight/approx/der1) | `cb-backend` (`GpuTrainSession`) | — | One `ComputeClient`; a Handle is bound to its allocating client. |
| Boosting loop / model assembly | `cb-train` (`train_inner`) | — | Device path is a per-iteration branch returning a host `DeviceGrownTree`. |
| Split descriptor → `cb_model` tree | `cb-train` | `cb-backend` (returns `feature,bin`) | Device returns `(feature,bin_id)`; cb-train maps `bin_id→border`. |
| Kaggle CUDA oracle + speed harness | `bench/` (`.ipynb` + fixtures) | maturin `--features cuda` wheel | Human-gated external step; ROCm in-env is smoke-only. |

## Standard Stack

### Core (all already in-tree — Phase 10 adds NO new compute crates)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `cubecl` | workspace-pinned facade (`cuda`/`hip`/`wgpu`/`cpu` features) | GPU kernel authoring + `ComputeClient`/`Handle` residency; **no CUB** → primitives are from-scratch | Project-mandated GPU layer (AGENTS.md); portable cuda/rocm/wgpu from one source. [VERIFIED: codebase — `cb-backend/Cargo.toml:19,35`] |
| `cb-backend` | local path | CubeCL impls + `GpuTrainSession` + `gpu_runtime` grow loop + `kernels/` primitives | Owns ALL `unsafe`/CubeCL; never depends on `cb-train`. |
| `cb-compute` | local path | The `Runtime` trait seam (CubeCL-free) + CPU Cosine/score/leaf oracles | Where GPUT-01's three methods land alongside `compute_gradients_grouped`. |
| `cb-core` | local path | `CbResult`/`CbError` (`thiserror`) | The seam's error type. |
| `cb-train` | local path | `train_inner` boosting loop + grower dispatch + `Model` assembly | Drives the seam; must NOT be a dep of cb-backend. |
| `maturin` | `>=1.9.4,<2.0` | builds the `--features cuda` abi3 wheel for Kaggle | Already the Python packaging backend. [CITED: prior research, `catboost-rs-py/pyproject.toml`] |

### Supporting (Kaggle harness only)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `numpy` | latest (in `benchmark.py`) | synthetic generator + fixture I/O (D-06) | bench notebook |
| `catboost` | 1.2.x (in `.venv`) | official CatBoost GPU baseline (`task_type='GPU'`) | speed head-to-head where a comparable depth-1 config exists |
| Jupyter/`.ipynb` | Kaggle-native | committed notebook artifact (D-05) | the harness form |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Committed `.ipynb` (D-05) | script + README (prior research D-10-03) | SUPERSEDED — the re-scope's CONTEXT.md D-05 locks a committed `.ipynb` (native Kaggle artifact, diffable, push-button). Follow D-05. |
| Bit-packed cindex from the start (D-07) | one-u32-per-cell first, tighten later | REJECTED by D-07 — rewrites histogram address arithmetic later; memory efficiency is first-class. |
| Fixed-point i64/u64 atomics as the deterministic reduce | float atomics | Float atomics are order-nondeterministic ([CITED: manual `08_atomic_contention.md`]); fixed-point (manual `09_fixedpoint_atomics.md`) is deterministic AND higher precision — a leading D-03 candidate. gfx1100 lacks f64 atomic-add but supports integer atomics. |
| Per-tree `grow_tree_on_device` seam | reuse whole-loop `grow_boosting_pass` | REJECTED (prior research) — `grow_boosting_pass` does host approx-update + der read-back (violates D-05) and bypasses cb-train machinery. |

**Installation:** No new Rust crates. Kaggle wheel build (documented in the committed notebook):
```bash
# on the Kaggle CUDA notebook, in crates/catboost-rs-py/
maturin build --release --no-default-features --features cuda
pip install target/wheels/catboost_rs-*.whl
# then run the committed notebook cells: correctness gate (BLOCKING) THEN speed
```

**Version verification:** No package versions to re-verify — Phase 10 installs nothing new. cubecl facade feature passthrough confirmed in-tree (`cb-backend/Cargo.toml:35` `cpu = ["cubecl/cpu"]`, rocm→`cubecl/hip`). [VERIFIED: codebase]

## Package Legitimacy Audit

> Phase 10 installs **no new external packages** into the Rust workspace. The only external runtime deps are already-present Python packages used by the existing `benchmark*.py` scripts + the Kaggle notebook. No SLOP/SUS risk introduced.

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| maturin | PyPI | mature | very high | github.com/PyO3/maturin | OK | Already in use |
| numpy | PyPI | mature | very high | github.com/numpy/numpy | OK | Already in use |
| catboost | PyPI | mature | high | github.com/catboost/catboost | OK | Already in `.venv` (oracle baseline) |

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```text
cb-train::train_inner  (boosting loop, generic over Runtime)
    │  feature bins (host) + feature_borders (f64) + target + weights + params
    ▼
(1) begin_device_training(loss, depth, boosting_type, fold_count, score_fn, bins, weight, dims, lr, l2)
    │        GpuBackend → GpuTrainSession::begin (cb-backend)
    │          coverage gate (D-10-02): depth==1 && RMSE|Logloss && Plain && fold==1 ?
    │          if Some → build BIT-PACKED cindex (GPUT-15) via WriteCompressedIndex-equivalent
    │                    upload cindex/indices/weight ONCE; allocate approx_h, der1_h on ONE client
    ▼
  Some(session) ─────────────────────────────┐        None → CPU grower (UNCHANGED, D-04)
                                               ▼
(2) per boosting iteration:  grow_tree_on_device(session, approx|resident, target)
      │ cb-backend, over the RESIDENT bit-packed cindex + resident der1_h:
      │   level 0:  device primitives →
      │      ├─ fill/transform (gather der1/weight)                 [GPUT-16]
      │      ├─ pointwise histogram over cindex via TCFeature addr  [GPUT-15 feeds this]
      │      ├─ scan / reduce (DETERMINISTIC) → per-candidate score [GPUT-16]
      │      ├─ score+argmin with SCORE_FN_COSINE (default)         [GPUT-08 — present]
      │      └─ O(1) BestSplit read-back
      │   partition:  TDataPartition offset/size update             [GPUT-16 partitions]
      │   leaves:  update_part_props → 2^depth part-stats read-back [GPUT-16]
      │            leaf_values = calc_average(Σder1, Σweight, l2)   (first-order; Newton=Ph11)
      │   approx_h += lr · leaf_values[leaf_of_h]   (apply_leaf_delta device kernel — NEW)
      │   der1_h = der(approx_h, target)            (Phase 7.2 seam — NO read-back)
      ▼
    DeviceGrownTree { splits:[(feature,bin_id)], leaf_values, [leaf_of only for oracle] }
      │  cb-train maps bin_id → border = feature_borders[feature][bin_id], folds into Model
      ▼
(3) end_device_training(session)  → drops handles / frees client

ORACLE/SPEED (human-gated, off the hot path):
  bench/*.ipynb on Kaggle CUDA: build --features cuda wheel → nvidia-smi →
    standalone primitive oracles (scan/segscan/sort/reorder/reduce-by-key/update_part_props) ≤1e-4
    → cindex oracle vs CPU quantized layout ≤1e-4
    → depth-1 RMSE+Logloss tree oracle ≤1e-5   (ALL BLOCKING)
    → warm-run/JIT-excluded train-only wall-clock (device vs CPU) → structured report → RESULTS
```

### Recommended Project Structure
```
crates/cb-backend/src/kernels/           # GPUT-16 primitive library (NEW + extend existing)
   scan.rs           # EXISTS (single-cube); ADD cross-cube carry (full scan)
   segmented_scan.rs # NEW — flag-array segmented scan
   reduce.rs         # EXISTS (HostSumFallback); ADD segmented-reduce + reduce-by-key + winner reduce
   sort.rs           # NEW — radix sort + stable single-bit reorder (reorder_one_bit)
   compression.rs    # NEW — bit-pack/unpack (feeds cindex)
   partitions.rs     # NEW — TDataPartition offset/size update
   update_part_props.rs # NEW — per-partition stat aggregation
   fill_transform.rs # NEW/consolidate — fill, gather/scatter (scatter.rs EXISTS)
crates/cb-backend/src/gpu_runtime/
   cindex.rs         # NEW — bit-packed cindex build + TCFeature addressing (GPUT-15)
   mod.rs            # grow_oblivious_tree_into: consume bit-packed cindex; add apply_leaf_delta
   session.rs        # NEW — GpuTrainSession (one client + resident handles, GPUT-02/03)
crates/cb-backend/src/gpu_backend.rs      # GpuBackend gains RefCell<Option<GpuTrainSession>>
crates/cb-compute/src/runtime.rs          # +3 default-impl seam methods + DeviceGrownTree struct
crates/cb-train/src/boosting.rs           # train_inner: device branch in the per-iteration body
bench/                                     # NEW
   cuda_oracle.ipynb                       # committed notebook: correctness gate + speed (D-05)
   generator.py                            # seeded synthetic generator (D-06 — one for both)
   fixtures/                               # committed deterministic fixtures + CPU expected values
   RESULTS.md                              # committed human sign-off log (per-phase CUDA run record)
.planning/phases/10-.../SPIKE-REDUCTION.md # reduction-determinism err+ms table + recommendation
```

### Pattern 1: Default-impl CubeCL-free seam method (GPUT-01)
**What:** Add the three grow-tree methods to the `Runtime` trait with **default implementations returning `Ok(None)`/`Ok(false)`** so every existing impl (`CpuBackend`, test runtimes) compiles unchanged and transparently falls back to CPU. Only `GpuBackend` overrides. EXACT pattern already used by `compute_gradients_grouped` (`runtime.rs:944`, a default that delegates to a function). [VERIFIED: codebase]
**When:** Always for this seam — preserves D-04 (CPU byte-unchanged) for free.

### Pattern 2: Session owned by the backend via `RefCell<Option<…>>`
**What:** `GpuBackend` is currently zero-sized (`gpu_backend.rs:47`). Give it `RefCell<Option<GpuTrainSession>>` interior mutability so `begin_device_training(&self,…)` sets it, `grow_tree_on_device(&self,…)` borrows it, `end_device_training(&self)` clears it — without changing the `&self` trait signature.
**Caveat:** Adding a `RefCell` field makes `GpuBackend` non-`Copy`/non-zero-sized. Grep `GpuBackend` for by-value copies (the Phase-8 facade binds it once and passes `&backend`, so this is fine — but verify). `RefCell` is `!Sync`; the `Model` Send+Sync contract is about the trained model, not the backend. [CITED: prior research Pattern 2 / Pitfall 4]

### Pattern 3: One-client residency; resident handles (GPUT-02/03)
**What:** The session allocates ONE `ComputeClient` and uploads the bit-packed `cindex`, `indices`, `weight` ONCE. It holds `approx_h` and `der1_h` as resident handles. Today `grow_oblivious_tree_into` uploads der1/weight/cindex/indices **every call** (`mod.rs:1704-1712`, verified) — refactor to a handle-taking variant so the matrix uploads once and der1 stays resident.
**Residency rule (HARD):** never read a `Handle` through a client other than the one that allocated it. The session owns the client for the whole fit. [VERIFIED: codebase — the residency comment is already documented at `grow_oblivious_tree_into` head, `mod.rs:1633-1639`]

### Pattern 4: Device split-descriptor → cb-train `Split` (bin→border join)
**What:** The device split test is `cindex_bin[feature] > bin_id`; the cb-train `Split{feature,border:f64}` test is `value > border`. The mapping is **`border = feature_borders[feature][bin_id]`** — the `bin_id`-th ascending border of that feature (`FeatureMatrix.feature_borders[f]` is the ascending candidate borders, `tree.rs:310-312`). `DeviceGrownTree` carries `(feature,bin_id)`; `train_inner` resolves the border from the same `feature_borders` it already holds. This is the one non-obvious correctness join. [VERIFIED: codebase — `FeatureMatrix` struct `tree.rs:307-327`]
**Caveat (bit-packed cindex, GPUT-15):** with the packed layout the device reads the bin via `(cindex[TCFeature.Offset + i] >> TCFeature.Shift) & TCFeature.Mask` before the `> bin_id` compare — the *bin value* is unchanged; only its *storage/extraction* changes. The bin→border join stays identical. [CITED: CATBOOST_CUDA_KERNELS_DESIGN.md §2 `TCFeature`, §6.6a `WriteCompressedIndex`]

### Anti-Patterns to Avoid
- **Adding `cb-train` to `cb-backend`'s deps** to reuse `FeatureMatrix`/`Split`/leaf logic. HARD LANDMINE — feature unification activates `cb-backend/cpu` alongside `rocm`, `SelectedRuntime` mis-resolves, kernels fail to build. Transcribe inline. [memory `phase75-grow-loop-outcome`]
- **`-inf` float literal in any `#[cube]` kernel** (new primitives, `apply_leaf_delta`, spike kernels). HIP/gfx1100 JIT rejects `double(-inf)` — emit a finite `f32::MIN` sentinel; host code may keep `f64::NEG_INFINITY`. Invisible to cpu/wgpu/cuda `cargo check`; fails only on rocm GPU. [memory `cubecl-hip-no-inf-literal`]
- **Non-deterministic float atomics in the reduce/histogram.** Float atomic-add is order-nondeterministic → run-to-run variance breaks even the ε=1e-4 bar. Use a deterministic strategy (fixed-order reduce, fixed-point atomics, or HostSumFallback). [CITED: manual `08_atomic_contention.md`]
- **Reading the full histogram/partition/`leaf_of` buffer to host per level.** The FORBIDDEN D-05 hybrid. Only the O(1) BestSplit + 2^depth part-stats may cross per level.
- **Shipping the plain one-u32-per-cell cindex.** D-07 requires the bit-packed grouped layout from the start.
- **Reporting a speed number before the correctness oracle passes.** D-05: correctness is a blocking gate.
- **Promising depth-1 device ≥ CPU at small n.** See ESCALATION flag.

## NEW-1: GPUT-16 — From-Scratch CubeCL Device-Primitive Library (no CUB)

CubeCL has **no CUB** — the upstream engine delegates sorts/scans/segmented reductions to `cub::Device*`, which does not exist here, so these are real from-scratch device deliverables. [CITED: REQUIREMENTS.md GPUT-16; CATBOOST_CUDA_KERNELS_DESIGN.md §1.3 "CUB delegation", §6.1/§6.2]

**What already exists in-tree (Phase 7-validated — reuse, do NOT re-derive):** [VERIFIED: codebase]
- `kernels/scan.rs` — `block_scan_kernel` (Hillis–Steele, single-cube; the **cross-cube running carry is the documented forward dependency**, RESEARCH Open Q2). Inclusive + exclusive, f32/f64, empty short-circuit.
- `kernels/reduce.rs` — `block_reduce` with an explicit `HostSumFallback` deterministic finalize path and an atomic path (`AtomicFinalizePath` enum at `reduce.rs:271`); observed variance harness present.
- `kernels/scatter.rs` — gather/scatter (`der1*weight`) with unweighted-identity case.
- `kernels/pointwise_hist.rs` — 2-channel `(Σder1, Σweight)` histogram over `cindex[feature*n+obj]` (plain layout — GPUT-15 changes the addressing).

**Per-primitive plan (map to upstream §6.1 + CubeCL manual):**

| Primitive | Upstream ref (§6.1) | Status | New work | Serial self-oracle (D-02) |
|-----------|---------------------|--------|----------|---------------------------|
| fill / MakeSequence / InversePermutation | `fill.cu` | partial | consolidate; grid-stride writes | trivial — transitive (D-01) |
| transform (vector arith, gather/scatter) | `transform.cu` | `scatter.rs` exists | add vector add/sub/mul/div, exp/pow | transitive (D-01) |
| full prefix scan (incl/excl) | `scan.cu` (Hillis–Steele in `inplace_scan.cuh`) | single-cube exists | **cross-cube carry** (multi-block) | serial `prefix_sum` in Rust/numpy — STANDALONE (D-01) |
| segmented scan (flag array) | `segmented_scan.cu` (`TSegmentedSum` pair combiner) | NEW | flag-carried segment reset | serial segmented prefix — STANDALONE (D-01) |
| reduce (full) | `reduce.cu` | exists (HostSumFallback) | winner from spike (D-04) | serial sum — transitive/plain |
| segmented reduce | `reduce.cu` (`SegmentedReduce*PerSegment`, meanSize<600 fast path) | NEW | warp/block-per-segment | serial per-segment sum — STANDALONE (D-01) |
| reduce-by-key | `reduce.cu` (`ReduceByKey`) | NEW | key-run detection + segment sums | serial group-by-sum — STANDALONE (D-01) |
| radix sort (LSD, key/value) | §6.2 `sort/` (CUB) | NEW | from-scratch multi-pass radix | serial stable sort — STANDALONE (D-01) |
| stable single-bit reorder | `reorder_one_bit.cu` (exclusive-scan of `(key>>bit)&1`, scatter to `zeroesBefore`/`totalZeros+onesBefore`) | NEW | one-bit stable partition | serial stable partition-by-bit — STANDALONE (D-01) |
| bit-compression (pack/unpack) | `compression.cu` (`TCompressionHelper`, BLOCK_SIZE-strided packing) | NEW | pack ui8 bins→ui32/ui64 words | serial pack/unpack — feeds cindex oracle |
| TDataPartition offset/size update | `partitions.cu` (`UpdatePartitionSizes`/`UpdatePartitionOffsets`) | NEW | boundary-scan → {Offset,Size} | serial partition bookkeeping — STANDALONE (D-01) |
| per-partition stat aggregation (`update_part_props`) | `update_part_props.cu` (ComputeSum→FastInBlockReduce→SaveResults, double accumulate) | NEW | 3-D grid partial sums + finalize | serial per-partition Σstat — STANDALONE (D-01) |

**Radix sort from scratch (no CUB) — approach:** LSD multi-pass. Each pass = compute per-digit histogram → exclusive scan (the full-scan primitive) → scatter to sorted positions. The **stable single-bit reorder** (`reorder_one_bit`) is the degenerate 1-bit case and is used directly where a full sort is unnecessary (partition splits). Build the single-bit reorder first (it exercises scan + scatter — both partly exist), then compose radix sort from it. [CITED: CATBOOST_CUDA_KERNELS_DESIGN.md §6.1 `reorder_one_bit`, §6.2]

**CubeCL building blocks (from the manual — read before writing any kernel, AGENTS.md):** [CITED: manual index]
- `Cubecl_shared_memory.md` + `Cubecl_plane.md` — shared-mem (LDS) block scan/reduce and plane (warp/wavefront) aggregation.
- `08_atomic_contention.md` — three-tier atomic strategy (naive → LDS privatization → plane aggregation).
- `09_fixedpoint_atomics.md` — deterministic fixed-point atomics (the reduction-determinism candidate).
- `10_grid_stride_occupancy.md` — grid-stride loops + 2-D grids partitioning a shared accumulator (mirrors upstream's `blockIdx.y`-per-column fill and `update_part_props` 3-D grid).
- `Batch-Tree_Reorganization_Algorithm.md` — directly relevant to partition/reorder.
- `Cubecl_comptime_specialization.md` — comptime bit-width / one-hot dispatch (mirrors upstream's runtime-bit-width kernel selection and the score `switch(scoreFunction)`).
- `Cubecl_generics.md` — the mandatory generics-float rule (kernels generic over `F: Float`, never hard-coded).

## NEW-2: GPUT-15 — Bit-Packed Device-Resident Compressed Index (cindex)

**The exact upstream layout** ([CITED: CATBOOST_CUDA_KERNELS_DESIGN.md §2, §6.6a]):
```cpp
struct TCFeature {
    ui64 Offset;          // base offset of this feature's column inside the packed cindex (ui32 units)
    ui32 Mask;            // bit mask isolating this feature's bin field within a ui32 word
    ui32 Shift;           // left-shift placing the bin field at its slot in the word
    ui32 FirstFoldIndex;  // index of first histogram fold on device
    ui32 Folds;           // fold count
    bool OneHotFeature;   // true → equality split (== value); false → threshold split (> value)
};
```
- Multiple features' bins are bit-packed into shared `ui32` words. **Write** (`WriteCompressedIndex`, blockSize=256): each thread ORs `(bins[i] & feature.Mask) << feature.Shift` into `cindex[feature.Offset + i]` — packs an already-computed `ui8` bin column into the shared word. When several features share a word concurrently, the write uses `atomicOr` (upstream `BinarizeFloatFeature` with `ATOMIC_UPDATE`); the plain `WriteCompressedIndex` path packs one column per launch with read-modify-write. [CITED: §6.6a lines 1928-1929]
- **Read** (in every histogram/split kernel): extract this feature's bin from the packed word via `(cindex[feature.Offset + obj] >> feature.Shift) & feature.Mask`, then compare `bin > bin_id` (threshold) or `bin == value` (OneHot). [CITED: §2 lines 1844-1845]

**Ripple analysis (HIGH risk, the reason for D-07's "right once"):** The current in-tree histogram (`pointwise_hist.rs:97-108`) reads bins via a **plain** `bin_of[i] = cindex[feature*n + indices[i]]` (one u32 per cell). [VERIFIED: codebase] Switching to the packed layout changes **every** consumer's bin-extraction from a direct load to the shift-mask expression. Plan explicitly: (1) define the Rust `TCFeature` descriptor + a host packer, (2) build the bit-packed `cindex` on device (or host-pack then upload once — borders stay host), (3) update `pointwise_hist` and `grow_oblivious_tree_into`'s partition read to use `(word >> Shift) & Mask`, (4) oracle the packed cindex against the CPU quantized bins ≤1e-4.

**Packing math (host-side, from `compression.cu` `TCompressionHelper`):** `bitsPerKey = ceil(log2(n_bins_for_feature+1))`; `keysPerWord = 32 / bitsPerKey`; a feature's `Shift` is its slot within the word, `Mask = (1<<bitsPerKey)-1`, `Offset` its column base. Features are grouped so their combined bit-width fits `ui32` words. [CITED: §6.1 `compression.cu` line 818]

**Validation (D-07):** the CPU quantized layout is the ≤1e-4 oracle. Standalone Kaggle oracle: pack a random-seeded bin matrix on device, unpack via `TCFeature` addressing, assert equality with the CPU bins. Because the packed value must exactly equal the source bin, this is effectively a **≤0 (bit-exact) integer oracle** — even tighter than ≤1e-4. [CITED: D-07]

**Borders stay host:** on-device border/quantile computation (`FastGpuBorders`/`ComputeQuantileBorders`) is explicitly deferred; the CPU quantization is the ≤1e-5 reference, uploaded once. Only cindex packing/residency is the device deliverable. [CITED: CONTEXT.md deferred; REQUIREMENTS.md line 63]

## NEW-3: GPUT-08 — Cosine GPU-Default Score (~90% ALREADY BUILT)

**Finding (direct read):** the score/argmin kernel **already supports `SCORE_FN_COSINE`** and is cross-oracled against the CPU Cosine calcer. [VERIFIED: codebase]
- Device: `SCORE_FN_COSINE` constant used in `gpu_runtime/mod.rs:71` and `kernels/score_split.rs:61`; the score-fn `switch` covers `L2/Cosine/SolarL2/LOOL2/SatL2` (`mod.rs:951`).
- Oracle: `score_split.rs:774,818` — `assert_arm_matches_oracle("cosine", SCORE_FN_COSINE, EScoreFunction::Cosine)` cross-checks the device arm against `cb-compute::score.rs`'s `EScoreFunction::Cosine`.
- CPU reference: `cb-compute::score.rs:57-160` implements `TCosineScoreCalcer` — `mu = S/(W+λ)`, accumulate `Score += S·mu`, `DenumSqr += W·mu²`, final `score = -Score/sqrt(DenumSqr)`, and CatBoost's DEFAULT `score_function` is `EScoreFunction::Cosine` (`runtime.rs:823-824`). [VERIFIED: codebase — matches CATBOOST_CUDA_KERNELS_DESIGN.md §6.3 `TCosineScoreCalcer` lines 1270-1274]

**Remaining GPUT-08 work (small):** (1) wire Cosine as the **default** score for the depth-1 device path (rather than L2, which the score_split default arms currently pass — `score_split.rs:269` uses `SCORE_FN_L2`); (2) confirm `grow_oblivious_tree_into`'s `score_fn` parameter is set to `SCORE_FN_COSINE` in the depth-1 seam path; (3) ensure the depth-1 oracle fixture's CPU reference uses Cosine so the ≤1e-5 comparison is method-consistent. **No new Cosine kernel is needed.** [VERIFIED: codebase — `grow_oblivious_tree_into(… score_fn: u32)` already takes the score-fn selector, `mod.rs:1651`]

## NEW-4: GPUT-04 — Depth-1 Oblivious Tree (reuse + Cosine)

Reuse `grow_oblivious_tree_into` (`mod.rs:1641`), which already grows a depth-1 tree over resident handles, rejects `depth>1` with a typed error (`mod.rs:1668` — the deliberate MVP guard), and estimates leaves via `cb_compute::calc_average` (first-order). [VERIFIED: codebase]

**Logloss depth-1 ≤1e-5 leaf-method pin (carried from prior research, now LOCKED):** RMSE leaf = `Σder1/(Σweight+l2)` via `calc_average` (first-order, exact). Logloss's CPU default is Newton (uses der2); the device depth-1 path computes first-order `calc_average`. CONTEXT.md locks the fix: **pin the Logloss depth-1 fixture's CPU reference to first-order `calc_average` leaves** (Newton der2 is Phase 11 / GPUT-07). This is the single most likely reason a naive GPUT-04 Logloss oracle misses ≤1e-5 — get the fixture's leaf method right. [CITED: CONTEXT.md scope anchor line 23; prior research Pitfall 3]

## NEW-5: BENCH-01/02 — Kaggle CUDA Oracle & Speed Harness

**Form (D-05, supersedes prior D-10-03):** a **committed `.ipynb` notebook** (not script+README). It builds the `--features cuda` wheel, loads **repo-committed fixtures** (random-seeded inputs + CPU-path expected values), runs correctness first (blocking), then warm-run/JIT-excluded train-only wall-clock, then prints a structured report. [CITED: CONTEXT.md D-05]

**Synthetic generator (D-06):** one seeded generator (`n_rows`/`n_features`, ~1e6×50, tunable above break-even) produces BOTH the depth-1 ≤1e-5 correctness fixture AND the large-n speed workload. No external download. [CITED: CONTEXT.md D-06]

**Harness order (blocking gate discipline):**
1. `nvidia-smi` (confirm CUDA active).
2. Standalone primitive oracles (D-01): scan, segmented scan, radix sort, single-bit reorder, reduce-by-key, `update_part_props` — each vs its serial CPU reference ≤1e-4 (integer primitives bit-exact).
3. Bit-packed cindex oracle vs CPU quantized layout (GPUT-15) — bit-exact.
4. Depth-1 RMSE + Logloss tree oracle vs CPU ≤1e-5 (GPUT-04).
5. **All above BLOCKING.** Only then: warm one untimed fit → time train-only, draining CubeCL's lazy queue with a read-back before stopping the clock → device vs CPU (same Kaggle hardware) + vs official CatBoost GPU where a comparable depth-1 config exists.
6. Structured report → committed `RESULTS.md` (human sign-off log).

**Warm-run / lazy-queue caveat:** CubeCL executes lazily — a cold first fit includes JIT compilation; stopping the clock before the queue drains under-reports. Warm once untimed, and force a read-back/predict to drain before stopping the timer. [CITED: manual `05_lazy_execution.md`, `11_launch_overhead_and_transfers.md`; prior research Pitfall 6]

## Reduction Determinism (D-03/D-04 — spike winner IS the shipped reduce)

Deterministic reduction is on the critical path for the ε=1e-4 bar AND is the depth-1 histogram's accumulator. The spike prototypes the top 2–3 candidates, measures run-to-run variance AND wall-clock on Kaggle CUDA, and the winner ships as the library reduce (no throwaway).

| Candidate | Determinism | Notes | Manual ref |
|-----------|-------------|-------|-----------|
| Fixed-order tree reduce (shared-mem block reduce, fixed pairing) | deterministic (fixed order) | already the shape of `block_reduce`; portable | `Cubecl_shared_memory.md`, `cubecl_reduce_sum.md` |
| Block-then-host-final-sum (HostSumFallback) | deterministic | ALREADY IN-TREE (`reduce.rs:271`); the Phase 7.6 precedent; final sum on host | memory `phase76-gpu-tolerance-signoff-outcome` |
| Fixed-point i64/u64 atomics | deterministic AND higher precision | `round(v·2^k)`→i64→u64 bits, exact order-independent add; gfx1100 lacks f64 atomic-add but supports integer atomics | `09_fixedpoint_atomics.md` |
| Kahan compensation | reduces error, order still matters | pair with fixed-order | `Cubecl_algebra.md` |
| Sorted-index accumulation | deterministic if sort is stable | depends on the new radix sort | §6.1 |

**Per-backend viability MUST be recorded** (gfx1100 has no f64 atomic-add → the in-env rocm smoke may exercise HostSumFallback while CUDA exercises the atomic path). `update_part_props` upstream accumulates partial sums in **double** precision (the `M=8` re-accumulation, [CITED: §6.1 line 776]) — mirror that: accumulate in f64 regardless of channel type to bound float error. Output: `SPIKE-REDUCTION.md` with an err+ms comparison table + recommendation feeding Phase 11's histogram kernel. [CITED: CONTEXT.md D-03/D-04]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Cosine device score | A new Cosine kernel | `SCORE_FN_COSINE` in `score_split` kernel (present, oracle-tested) | GPUT-08 is ~90% done — wire as default, don't re-author |
| Depth-1 hist+score+argmin+partition | A new depth-1 kernel chain | `grow_oblivious_tree_into` (`mod.rs:1641`) + `launch_*_into` primitives | Oracle-validated Phase 7.5; depth-1 is its exact MVP |
| Device der1/der2 (RMSE/Logloss) | New gradient kernels | Phase 7.2 der seam (`der_seams.rs` `launch_der_binary_into`) | Device-resident, oracle-locked, no read-back |
| Leaf value formula | Custom average | `cb_compute::calc_average(Σder1, Σweight, scaled_l2)` (used at `mod.rs:1530`) | FROZEN formula, identical host/device |
| Single-cube block scan / block reduce / scatter | New kernels | `kernels/scan.rs`, `kernels/reduce.rs`, `kernels/scatter.rs` | Exist + oracled; extend (cross-cube carry, segmented, winner-reduce) rather than replace |
| CPU fallback correctness | A device "approximation" of uncovered cases | `Ok(None)` → existing CPU grower dispatch (UNCHANGED) | The CPU path is the oracle and the safety net (D-04) |
| Trait extension keeping CPU untouched | New trait / parallel hierarchy | Default-impl methods on `Runtime` (copy `compute_gradients_grouped`, `runtime.rs:944`) | Zero churn to every existing impl |
| CUB (sorts/scans/segmented reduce) | Assume CUB exists in CubeCL | From-scratch primitives (GPUT-16) | **CubeCL has NO CUB** — the whole reason GPUT-16 exists |

**Key insight:** The genuinely new code is a focused ~20% — the from-scratch primitives CubeCL lacks (scan cross-cube carry, segmented scan/reduce, reduce-by-key, radix sort, single-bit reorder, compression, partitions, update_part_props, deterministic reduce), the bit-packed cindex + its address-arithmetic ripple, the 3-method seam, the `GpuTrainSession` residency wrapper, ONE small `apply_leaf_delta` kernel, the Kaggle notebook + generator, and the spike. Cosine and the depth-1 grow loop are largely already built. Resist re-deriving anything oracle-validated in Phase 7.

## Runtime State Inventory

> Phase 10 is greenfield-additive (new primitives + new cindex + new session + new bench dir). It changes no stored data, services, or OS state. The one "state" concern is CubeCL device-buffer lifetime.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — no datastore keys/IDs renamed | none — verified: no rename in scope |
| Live service config | None | none |
| OS-registered state | None | none |
| Secrets/env vars | `DEBUG`/`OUT_DIR` only (build.rs, unrelated); Kaggle notebook needs no secrets | none |
| Build artifacts | The `--features cuda` wheel is built fresh on Kaggle each run (not committed); fixtures + notebook + RESULTS ARE committed (D-05) | commit `bench/*.ipynb`, `bench/fixtures/`, `bench/RESULTS.md` |
| Device-buffer lifetime (CubeCL) | Session handles must be freed at `end_device_training`; a `Handle` outliving its client is UB-adjacent | `end_device_training` drops the session (client + handles) deterministically |

## Common Pitfalls

### Pitfall 1: Assuming `grow_boosting_pass` already satisfies GPUT-03
**What goes wrong:** A plan says "wire `grow_boosting_pass` and GPUT-03 is done." It is not. `grow_boosting_pass_into` updates the running `approx` **on the host** and reads back `der1` (n elements) every iteration — two n-length crossings per tree, exactly what D-05/GPUT-03 forbids. [CITED: prior research Pitfall 1, verified in `gpu_runtime/mod.rs`]
**How to avoid:** GPUT-03's deliverable = (a) keep `approx_h`/`der1_h` resident on the session, (b) add a small `#[cube]` kernel `apply_leaf_delta(approx, leaf_of, leaf_values, lr)` doing `approx[i] += lr * leaf_values[leaf_of[i]]` on device, (c) recompute der1 on device without read-back, chained into the next tree.
**Warning signs:** any `client.read_*` of an n-length buffer inside the per-tree loop.

### Pitfall 2: cindex address-arithmetic ripple (GPUT-15)
**What goes wrong:** Switching to the bit-packed layout but leaving one histogram/partition consumer reading the old plain `cindex[feature*n+obj]` — silently wrong bins.
**How to avoid:** Define ONE `TCFeature` accessor (`(word>>Shift)&Mask`) and route every consumer through it. Grep for every `cindex[` index expression in `gpu_runtime` + `kernels/` and convert. Oracle the packed layout bit-exact before wiring it into the tree.
**Warning signs:** cindex oracle passes but the depth-1 tree structure diverges from CPU.

### Pitfall 3: Cross-cube scan carry (RESEARCH Open Q2)
**What goes wrong:** The existing `block_scan_kernel` scans WITHIN one cube; for the large-n bench (~1e6 rows) a single cube cannot hold the whole array, so a multi-block running carry is required and its absence silently truncates the scan at the cube boundary.
**How to avoid:** Implement the standard two-level scan (per-block scan → scan of block-sums → add block-offset) as the full-scan primitive; oracle it against a serial prefix sum at n ≫ one cube. The reduction spike records carry behavior per candidate. [CITED: prior research Open Q2, `scan.rs:10-14`]
**Warning signs:** scan oracle passes at small n, fails at n > cube size.

### Pitfall 4: Logloss depth-1 leaf-method mismatch
**What goes wrong:** Device computes first-order `calc_average` leaves; the Logloss fixture's CPU reference uses Newton (der2) → ≤1e-5 fails by ~1e-2..1e-3 (a method mismatch, not a bug).
**How to avoid:** Pin the Logloss depth-1 fixture's CPU reference to first-order `calc_average` (Newton is Phase 11). LOCKED by CONTEXT.md. [CITED: CONTEXT.md line 23]
**Warning signs:** Logloss oracle ~1e-2..1e-3 off while RMSE is ~1e-9.

### Pitfall 5: `#[cube]` `-inf` and atomic non-determinism
**What goes wrong:** A new primitive or spike kernel uses `-inf` (JIT-rejected on gfx1100) or a float atomic (run-to-run-varying).
**How to avoid:** `f32::MIN` sentinel; deterministic reduction only. Run the rocm smoke after EVERY `#[cube]` change — cpu/wgpu/cuda `cargo check` hide both. [memory `cubecl-hip-no-inf-literal`, `08_atomic_contention.md`]

### Pitfall 6: `GpuBackend` losing `Copy`/zero-sized-ness
**What goes wrong:** Adding `RefCell<Option<GpuTrainSession>>` makes `GpuBackend` non-`Copy`; a `let b2 = backend;` copy becomes a move.
**How to avoid:** grep `GpuBackend` for by-value uses; keep `Default` (init `RefCell` to `None`). [CITED: prior research Pitfall 4]

### Pitfall 7: Quoting a fast-but-wrong CUDA speed number
**What goes wrong:** Reporting wall-clock before the oracle passes, timing a cold JIT fit, or stopping the clock before CubeCL's lazy queue drains.
**How to avoid:** harness order = `nvidia-smi` → warm untimed fit → oracle (BLOCKING) → time train-only with a drain read-back before stopping. [CITED: prior research Pitfall 6]

## Code Examples

### The three seam methods (GPUT-01) — add to `Runtime` (`cb-compute/src/runtime.rs`)
```rust
// Source: pattern transcribed from compute_gradients_grouped default-impl (runtime.rs:944).
// CubeCL-FREE host-typed structs only.

/// One device-grown tree, host-typed (D-05: only descriptors cross).
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceGrownTree {
    /// Per-level chosen split as (feature_index, bin_id). Pass test is
    /// `quantized_bin[feature] > bin_id`; caller resolves
    /// `border = feature_borders[feature][bin_id]`.
    pub splits: Vec<(u32, u32)>,
    /// Per-leaf value (UN-scaled by learning_rate — cb-train scales it). Length 2^depth.
    pub leaf_values: Vec<f64>,
    /// Per-object leaf index, ONLY populated for the oracle structure check (empty in the
    /// production hot path so the n-length buffer never crosses per tree — D-05). Len 0 or n.
    pub leaf_of: Vec<u32>,
}

pub trait Runtime {
    // ... existing compute_gradients / compute_gradients_grouped ...

    /// Decide ONCE per fit whether this config grows on device (D-10-01/02).
    /// Default false (CPU). Only GpuBackend overrides. Params carry enough to run the
    /// gate AND build+upload the bit-packed cindex once.
    fn begin_device_training(
        &self,
        loss: &Loss,
        depth: usize,
        boosting_type_is_plain: bool,
        fold_count: usize,
        score_function: EScoreFunction, // Cosine default (GPUT-08)
        bins_feature_major: &[u32],     // host bins → device packs the cindex (GPUT-15)
        weight: &[f64],
        n: usize, n_features: usize, n_bins: usize,
        learning_rate: f64, scaled_l2: f64,
    ) -> CbResult<bool> {
        let _ = (loss, depth, boosting_type_is_plain, fold_count, score_function,
                 bins_feature_major, weight, n, n_features, n_bins, learning_rate, scaled_l2);
        Ok(false)
    }

    /// Grow ONE tree on device over the resident approx/der state.
    fn grow_tree_on_device(&self, approx: &[f64], target: &[f64])
        -> CbResult<Option<DeviceGrownTree>> {
        let _ = (approx, target);
        Ok(None)
    }

    /// Tear down the session, freeing the ComputeClient + resident handles.
    fn end_device_training(&self) -> CbResult<()> { Ok(()) }
}
```
*(Signatures illustrative — exact set is Claude's discretion. Load-bearing: host-typed only, `Ok(None)`/`Ok(false)` defaults, enough params for the D-10-02 gate + the once-per-fit cindex build.)*

### On-device approx-update kernel (GPUT-03, NEW)
```rust
// Source: new #[cube] kernel; keeps approx-update OFF the host. NO -inf; generic-float.
#[cube(launch)]
fn apply_leaf_delta_kernel<F: Float>(
    approx: &mut Array<F>,     // length n, resident
    leaf_of: &Array<u32>,      // length n, resident (never read to host)
    leaf_values: &Array<F>,    // length 2^depth, small
    lr: F,
) {
    let i = ABSOLUTE_POS;
    if i < approx.len() {
        approx[i] += lr * leaf_values[leaf_of[i]];  // approx stays resident
    }
}
// Then der1_h = der(approx_h, target) via the 7.2 seam, chained into the next tree — no read-back.
```

### TCFeature bin extraction (GPUT-15) — the ONE accessor every consumer uses
```rust
// Source: CATBOOST_CUDA_KERNELS_DESIGN.md §2 / §6.6a. Replaces `cindex[feature*n+obj]`.
#[cube]
fn read_bin(cindex: &Array<u32>, offset: u32, obj: u32, shift: u32, mask: u32) -> u32 {
    (cindex[offset + obj] >> shift) & mask
}
// Threshold split: read_bin(..) > bin_id ; OneHot split: read_bin(..) == value.
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| GPU = derivatives only; whole grow loop on host | Device-resident grow loop (Phase 7.5, UNWIRED) | Phase 7.5 (2026-06) | Phase 10 wires it via the seam; the >20× gap is the unwired loop |
| Plain one-u32-per-cell cindex | Bit-packed grouped cindex + `TCFeature` addressing | Phase 10 (GPUT-15) | Memory-efficient; every histogram kernel's address arithmetic changes once |
| Upstream delegates scans/sorts to CUB | From-scratch CubeCL primitives (no CUB) | Phase 10 (GPUT-16) | Real deliverables, not wrappers; each gets a serial self-oracle |
| L2 device score in depth-1 path | Cosine GPU-default (already in the kernel) | Phase 10 (GPUT-08) | ~90% built; wire as default |
| Bit-exact f64 ≤1e-5 on GPU | ε=1e-4 vs CPU (device math f32); depth-1 held ≤1e-5; integer primitives bit-exact | Phase 7.6 precedent | Depth-1 whole-dataset histogram IS the exact CPU score → ≤1e-5 justified |

**Deprecated/outdated:**
- `grow_boosting_pass` host approx-update + der read-back — NOT reused as-is for GPUT-03; it is the anti-pattern the phase removes.
- The prior research's D-10-03 "script + README" harness form — SUPERSEDED by the re-scope's D-05 committed `.ipynb`.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Cosine device score is ~90% built and needs only wiring as the depth-1 default (no new kernel) | NEW-3 | If the existing `SCORE_FN_COSINE` arm lacks the second-order/Newton-Cosine variant the depth-1 default needs, a small kernel extension is required — verify the arm against `score.rs:139` NewtonCosine path |
| A2 | Host-packing the cindex then uploading once (borders host) satisfies GPUT-15 without an on-device `WriteCompressedIndex` kernel | NEW-2 | If GPUT-15 is read to require the *device* packing kernel (not host pack + upload), add a `#[cube]` `write_compressed_index` — still small; escalate the interpretation |
| A3 | The from-scratch radix sort composes cleanly from single-bit reorder + the full-scan primitive at bench scale (~1e6) | NEW-1 | If multi-pass radix perf/carry is worse than expected, the sort becomes a larger sub-task — the spike de-risks the scan carry first |
| A4 | Logloss depth-1 ≤1e-5 holds with first-order leaves once the CPU fixture is pinned to the same method | NEW-4/Pitfall 4 | If a meaningful Logloss oracle REQUIRES Newton leaves, depth-1 Logloss needs a small device der2 leaf step (scope creep) — ESCALATE |
| A5 | Depth-1 device ≥ CPU wall-clock is achievable ONLY at large n (~1e5–1e6+), NOT at small n | Depth-1 Feasibility | If the user insists on small-n, D-10-09 is physically infeasible → MUST ESCALATE before the speed-plan locks |
| A6 | Adding `RefCell<Option<GpuTrainSession>>` to `GpuBackend` breaks no caller relying on `Copy` | Pattern 2 / Pitfall 6 | If a Phase-8 path copies `GpuBackend`, the build breaks — grep required |
| A7 | `maturin build --no-default-features --features cuda` produces a working Kaggle wheel (CUDA toolkit present on Kaggle GPU notebooks) | NEW-5 | If Kaggle's CUDA/driver mismatches CubeCL/cudarc, the wheel won't JIT — pin a known-good Kaggle image in the notebook |
| A8 | CubeCL manual path is lowercase `/manual/cubecl/` (AGENTS.md/CONTEXT.md say `/Cubecl/`) | Sources | Agents following the documented capitalized path get "No such file"; the real path is lowercase `cubecl`, INDEX is `INDEX.md` [VERIFIED: filesystem this session] |

## Open Questions (RESOLVED)

> All three questions were resolved during planning (2026-07-03). Resolutions are carried in the finalized 9-plan decomposition; inline `RESOLVED:` markers below record the decision + plan reference for the audit trail.

1. **Device vs host cindex packing (A2).**
   - What we know: borders stay host; the deliverable is packing + residency.
   - What's unclear: whether GPUT-15 requires an on-device `WriteCompressedIndex` kernel or accepts host-pack-then-upload-once.
   - Recommendation: host-pack + upload once is simplest and satisfies "device-resident cindex"; if the planner reads GPUT-15 as requiring the device packing kernel, it is still small (§6.6a `WriteCompressedIndex` blockSize=256). Decide in planning.
   - **RESOLVED:** host-pack then upload-once (borders host) adopted as the GPUT-15 interpretation — `10-06-PLAN.md` Task 1 packs the bit-packed `TCFeature` layout on host and uploads once, documenting the decision in `cindex.rs`; the on-device `write_compressed_index` kernel is explicitly deferred unless parity requires it (surfaced as a documented assumption, not a silent drop).

2. **Cosine depth-1 default: second-order vs first-order interaction (A1).**
   - What we know: the score kernel supports Cosine; leaves are first-order `calc_average` for Phase 10.
   - What's unclear: Cosine is a second-order *score* function; confirm the depth-1 path uses the Cosine SCORE for split selection while leaves stay first-order (Newton leaf = Phase 11) — these are independent choices and must both be pinned in the fixture.
   - Recommendation: Cosine score + first-order leaves for the depth-1 oracle; document the pairing explicitly so the ≤1e-5 comparison is method-consistent.
   - **RESOLVED:** Cosine SCORE for split selection + first-order `calc_average` leaves, both pinned in the depth-1 fixture — `10-07-PLAN.md` Task 1 pins the pairing so the ≤1e-5 comparison is method-consistent (Newton leaves remain Phase 11).

3. **Reduction winner per backend (D-03).**
   - What we know: fixed-point atomics (deterministic + higher precision) and HostSumFallback (already in-tree) are the leading candidates; gfx1100 lacks f64 atomic-add.
   - What's unclear: which wins on Kaggle CUDA for both variance and speed.
   - Recommendation: the spike measures and records per-backend viability; the winner ships as the library reduce (D-04). RESOLVED at spike time, not planning.
   - **RESOLVED (deferred by design):** deliberately deferred to spike time — `10-03-PLAN.md` prototypes the top candidates, measures variance + speed on Kaggle CUDA, records the recommendation in `SPIKE-REDUCTION.md`, and the winner ships AS the reduce primitive (D-04, no throwaway). Not a planning-time decision.

## Environment Availability

| Dependency | Required By | Available (in-env) | Version | Fallback |
|------------|------------|--------------------|---------|----------|
| AMD gfx1100 / ROCm GPU | in-env compile/smoke (NOT a gate) | ✓ | ROCm 7.1 | none needed (smoke only) |
| NVIDIA CUDA GPU | BENCH-01/02 + all GPU oracles (authoritative) | ✗ (NO CUDA in-env) | — | **Kaggle CUDA notebook (human-gated)** |
| maturin | build `--features cuda` wheel | ✓ | >=1.9.4,<2.0 | — |
| catboost (Python) | official-GPU speed baseline | ✓ (.venv) | 1.2.x | skip GPU baseline where no comparable depth-1 config |
| numpy | synthetic generator / fixtures | ✓ | latest | — |
| CubeCL manual | mandatory pre-kernel read (AGENTS.md) | ✓ | — | path is `.../manual/cubecl/INDEX.md` (lowercase, A8) |

**Missing dependencies with no fallback:** none that block in-env work. CUDA correctness/speed sign-off is BY DESIGN a human-gated external Kaggle step — mark every GPUT-04/08/15/16/BENCH gate as "awaits Kaggle CUDA run," not in-CI.
**Missing dependencies with fallback:** CUDA GPU → Kaggle notebook; the in-env ROCm build provides compile + correctness smoke only.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` (workspace), SEPARATE test files per CLAUDE.md (no `#[cfg(test)] mod tests` in production files); existing GPU self-oracles live in `cb-backend/src/kernels/*.rs` (e.g. `scan.rs`, `reduce.rs`, `scatter.rs`, `score_split.rs`) |
| Config file | none beyond Cargo; backend by feature (`--no-default-features --features rocm` for in-env GPU tests) |
| Quick run command | `cargo test -p cb-backend --no-default-features --features rocm <name>` (in-env GPU smoke) |
| Full suite command | per-crate (disk pressure — memory `disk-pressure-and-full-suite-verification`): `cargo test -p cb-train`, `cargo test -p cb-backend ...` |
| Authoritative GPU oracle | committed `bench/*.ipynb` on Kaggle CUDA (human-gated) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| GPUT-16 | scan(cross-cube)/segscan/segreduce/reduce-by-key/sort/reorder/compression/partitions/update_part_props match serial CPU ref | self-oracle (STANDALONE for high-risk, D-01) | `cargo test -p cb-backend --features rocm <primitive>` (smoke); Kaggle authoritative | partial (`scan.rs`,`reduce.rs`,`scatter.rs` exist) → ❌ Wave 0 for the rest |
| GPUT-15 | bit-packed cindex == CPU quantized bins (bit-exact) | oracle | `cargo test -p cb-backend --features rocm cindex_pack`; Kaggle | ❌ Wave 0 |
| GPUT-01 | seam default-impl returns Ok(None)→CPU; CpuBackend unchanged | unit | `cargo test -p cb-train` (CPU oracle suite stays green = D-04) | ✅ existing CPU oracles |
| GPUT-02 | matrix uploaded once; session holds one client | unit (in-env) | `cargo test -p cb-backend --features rocm session_residency` | ❌ Wave 0 |
| GPUT-03 | no n-length read-back in per-tree loop (instrument crossing count) | unit (in-env) | `cargo test -p cb-backend --features rocm residency_no_readback` | ❌ Wave 0 |
| GPUT-04 | depth-1 RMSE+Logloss device == CPU ≤1e-5 | oracle | in-env `--features rocm depth1_oracle`; authoritative Kaggle | ❌ Wave 0 + bench |
| GPUT-08 | Cosine device score == CPU Cosine calcer | oracle | `cargo test -p cb-backend --features rocm` (score_split cosine arm — EXISTS) | ✅ `score_split.rs:774` |
| BENCH-01 | wheel builds, oracle gates, speed measured | harness | `bench/*.ipynb` (Kaggle) | ❌ Wave 0 (bench/) |
| BENCH-02 | depth-1 device wall-clock vs CPU recorded | harness | `bench/*.ipynb` → RESULTS.md | ❌ Wave 0 (bench/) |

### Sampling Rate
- **Per task commit:** `cargo test -p cb-backend --no-default-features --features rocm <touched>` after any `#[cube]` change (rocm catches `-inf`/atomic issues cpu/cuda hide).
- **Per wave merge:** full per-crate CPU suite (D-04 no-regression) + in-env rocm primitive/residency/depth-1 smoke.
- **Phase gate:** Kaggle CUDA notebook — standalone primitive oracles + cindex oracle + depth-1 ≤1e-5 (ALL BLOCKING) then speed; RESULTS.md signed off by the user.

### Wave 0 Gaps
- [ ] `kernels/segmented_scan.rs`, `kernels/sort.rs`, `kernels/compression.rs`, `kernels/partitions.rs`, `kernels/update_part_props.rs` — new primitives + serial self-oracles (separate test files)
- [ ] cross-cube carry added to `kernels/scan.rs`; segmented-reduce + reduce-by-key + winner-reduce added to `kernels/reduce.rs`
- [ ] `gpu_runtime/cindex.rs` — bit-packed cindex + `TCFeature` accessor + bit-exact oracle
- [ ] `gpu_runtime/session.rs` — `GpuTrainSession` (one client + resident handles) + residency/no-readback tests
- [ ] `apply_leaf_delta` device kernel (GPUT-03)
- [ ] `bench/cuda_oracle.ipynb`, `bench/generator.py`, `bench/fixtures/`, `bench/RESULTS.md` (D-05/D-06)
- [ ] depth-1 device-vs-CPU oracle test (RMSE + Logloss, Cosine score) — in-env smoke + Kaggle authoritative
- [ ] reduction-determinism spike kernels (top 2–3 candidates) + `SPIKE-REDUCTION.md`

## Security Domain

> `security_enforcement` is enabled in config but is not a meaningful axis for this phase: it is an internal numeric-compute library change with no auth, network, session, or untrusted-input surface. The one relevant control is input-validation hygiene already enforced project-wide (typed `CbError` guards, no `unwrap`/`panic`/indexing in production, kernel bounds-guards).

### Applicable ASVS Categories (L1)
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes (numeric) | Typed `CbError` (`LengthMismatch`/`OutOfRange`/`Degenerate`) on every length/overflow/shape; no `unwrap`/`panic`/indexing in production (workspace lints); `checked_shl`/`checked_mul` guards already present in `grow_oblivious_tree_into` |
| V6 Cryptography | no | — |

### Known Threat Patterns for {Rust GPU compute}
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Out-of-bounds device buffer access (kernel index) | Tampering / DoS | Bounds-guard in kernel (`if i < arr.len()`); host shape guards before launch; never read a 0-len handle |
| Reading a Handle through the wrong client | Tampering (UB-adjacent) | One client per session; residency rule enforced (Pattern 3) |
| Silent zero-buffer on read-back failure | Repudiation (wrong result looks valid) | Surface `CbError::Degenerate` on any read-back failure, never a silent zero (existing WR-05 precedent) |
| Integer overflow in cindex/packing address arithmetic | Tampering | `checked_mul`/`checked_shl` on `Offset`/`n_features*n`/`2^depth` (pattern already in `grow_oblivious_tree_into`) |

## Depth-1 device > CPU Feasibility (ESCALATION FLAG — D-10-09)

**Carried forward from prior research — still the loud flag CONTEXT.md demands.** A depth-1 tree is the single most launch-overhead-bound workload in the entire milestone: per tree the device does a handful of kernel launches over trivial per-object work. At small n (e.g. `benchmark.py`'s 1000×20) the CPU grows a depth-1 stump in microseconds; no kernel fusion or residency makes a GPU launch + driver round-trip competitive. **Device WILL lose at small n — this is physics, not a tuning gap.** Device wins only at large n (~1e5–1e6+ rows) where the O(n·features) histogram parallelizes enough to amortize fixed launch latency.

The re-scope already partially resolves this: **D-06 pins the speed workload to a seeded synthetic generator (~1e6×50, tunable above break-even)** and D-10-09 pins the bar to large-n. The manual's launch-overhead levers (`11_launch_overhead_and_transfers.md`: hoist invariant uploads, batch read-backs, `launch_unchecked`, collapse per-item launches) plus full residency (GPUT-02/03) are the tools to reach the crossover.

**Recommendation (firm):**
1. Pin BENCH-02's depth-1 speed bar to the large-n synthetic workload (D-06); may need to grow n until device ≥ CPU — the Kaggle run is the arbiter.
2. **ESCALATE explicitly before the speed-plan locks:** "Depth-1 device ≥ CPU is achievable only at large n; at small n it is physically infeasible regardless of optimization." Do not silently relax OR silently assume D-10-09.
3. If, after large-n fused/resident optimization on Kaggle CUDA, depth-1 device still does not beat CPU even at large n, report the measured crossover (or its absence) and let the user decide whether D-10-09 stands for depth-1 (depth-6 in Phase 11 is where device dominance is unambiguous).

## Sources

### Primary (HIGH confidence — direct codebase read this session)
- `crates/cb-backend/src/gpu_runtime/mod.rs:1633-1720` — `grow_oblivious_tree_into` (depth-1 MVP, resident handles, re-uploads per call, `score_fn` param, depth>1 typed-error guard, `SCORE_FN_COSINE`/`SCORE_FN_L2` at :71,:951)
- `crates/cb-backend/src/kernels/scan.rs` — single-cube `block_scan_kernel` (cross-cube carry = forward dependency, Open Q2)
- `crates/cb-backend/src/kernels/reduce.rs:250-352` — `block_reduce` + `HostSumFallback`/`AtomicFinalizePath` deterministic finalize + variance harness
- `crates/cb-backend/src/kernels/scatter.rs`, `kernels/pointwise_hist.rs:88-114` — scatter + 2-channel histogram over plain `cindex[feature*n+obj]`
- `crates/cb-backend/src/kernels/score_split.rs:61,269,774,818` — `SCORE_FN_COSINE` present + cross-oracle vs `EScoreFunction::Cosine`
- `crates/cb-backend/src/gpu_runtime/der_seams.rs:50-168` — Phase 7.2 der seam (no-read-back `*_handle`)
- `crates/cb-backend/src/gpu_backend.rs:47` — zero-sized `GpuBackend`; `cb-backend/Cargo.toml:19,35` — cubecl-only dep, facade features
- `crates/cb-compute/src/runtime.rs:823-824,944-970` — `EScoreFunction::Cosine` default, `compute_gradients_grouped` default-impl seam pattern
- `crates/cb-compute/src/score.rs:57-171` — CPU `TCosineScoreCalcer` (the Cosine oracle)
- `crates/cb-train/src/tree.rs:307-327` — `FeatureMatrix` / `feature_borders` (the bin→border join)
- `CATBOOST_CUDA_KERNELS_DESIGN.md` §2 (`TCFeature`), §6.1 (`cuda_util/kernel` primitives: fill/transform/reduce/scan/segmented_scan/partitions/update_part_props/compression/reorder_one_bit), §6.2 (`cuda_util/kernel/sort` radix), §6.3 (`TCosineScoreCalcer`), §6.6a (`binarize.cu` / `WriteCompressedIndex`)
- `.planning/milestones/v1.1-rescope-2026-07-02-phases/10-.../10-RESEARCH.md` — PRIOR RESEARCH (seam/residency/depth-1 still valid; reused, not re-derived)
- `.planning/REQUIREMENTS.md` (GPUT-01/02/03/04/08/15/16, BENCH-01/02), `10-CONTEXT.md`, `10-DISCUSSION-LOG.md`, `.planning/notes/gpu-training-host-light-root-cause.md`
- `/home/user/Documents/workspace/cubecl_manual/manual/cubecl/` — INDEX + `08_atomic_contention.md`, `09_fixedpoint_atomics.md`, `10_grid_stride_occupancy.md`, `11_launch_overhead_and_transfers.md`, `05_lazy_execution.md`, `Cubecl_shared_memory.md`, `Cubecl_plane.md`, `Cubecl_comptime_specialization.md`, `Cubecl_generics.md`, `Batch-Tree_Reorganization_Algorithm.md`, `cubecl_reduce_sum.md` (lowercase path verified)

### Secondary (MEDIUM confidence — project memory)
- Memories: `phase75-grow-loop-outcome`, `cubecl-hip-no-inf-literal`, `phase72-gpu-der-seam-outcome`, `phase76-gpu-tolerance-signoff-outcome`, `phase64-score-fns-gpu-only`, `disk-pressure-and-full-suite-verification`, `phase8-python-bindings-outcome`, `gpu-training-host-light-root-cause`

### Tertiary (LOW confidence — analytical, unverified on hardware)
- Depth-1 device>CPU feasibility at large vs small n — analytical (launch-overhead vs O(n·features) parallelism); MUST be confirmed on Kaggle CUDA (exactly what BENCH-02 measures)
- From-scratch radix-sort perf at bench scale — composition sound; perf unverified until the spike

## Metadata

**Confidence breakdown:**
- Reuse surface (grow loop, der seam, Cosine kernel, scan/reduce/scatter primitives): HIGH — all read directly this session
- GPUT-16 new primitives (approach + serial self-oracle discipline): HIGH on approach (upstream §6.1 + manual), MEDIUM on first-pass `#[cube]` correctness (rocm smoke required after each)
- GPUT-15 cindex packing + ripple: HIGH on the layout (upstream §6.6a + `TCFeature`), MEDIUM on the address-arithmetic conversion landing clean across every consumer
- GPUT-08 Cosine: HIGH — already built and oracle-tested; only wiring remains
- Seam + residency (GPUT-01/02/03/04): HIGH — mirrors shipped patterns (prior research validated)
- Reduction determinism: HIGH on candidates (manual confirms fixed-point + atomic non-determinism), MEDIUM on per-candidate numbers (the spike produces them)
- Depth-1 device>CPU (D-10-09): LOW/analytical — ESCALATION flagged; only Kaggle CUDA settles it
- Kaggle harness (BENCH-01/02): MEDIUM — maturin/feature plumbing verified; Kaggle CUDA image compatibility (A7) unverified

**Research date:** 2026-07-03
**Valid until:** 2026-07-17 (14 days — fast-moving GPU phase; codebase line refs may drift as plans land)
</content>
</invoke>
