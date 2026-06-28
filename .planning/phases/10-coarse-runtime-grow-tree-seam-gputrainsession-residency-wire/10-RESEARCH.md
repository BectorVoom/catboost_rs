# Phase 10: Coarse Runtime Grow-Tree Seam + GpuTrainSession Residency + Wire Depth-1 + Kaggle CUDA Oracle & Speed Harness - Research

**Researched:** 2026-06-29
**Domain:** GPU device-resident gradient-boosting training (CubeCL), Rust trait-seam architecture, CUDA oracle/benchmark harness
**Confidence:** HIGH (codebase verified by direct read; all line refs current) / MEDIUM (depth-1>CPU feasibility — analytical, not yet measured on Kaggle CUDA)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Fallback granularity (Ok(None) → CPU)**
- **D-10-01:** **Per-fit, all-or-nothing.** The covered/uncovered decision is made ONCE at `begin_device_training`. If the whole config is device-coverable → a `Some(session)` is returned and EVERY tree in the fit grows on device. If not → `None`, and the ENTIRE fit uses the CPU grower. **No mid-run mixing of device-grown and CPU-grown trees in one model** — chosen for parity safety (no subtle device/CPU drift compounding across the boosting run) and a clean, attributable speed measurement.
- **D-10-02:** Because the gate is per-fit, the covered-vs-uncovered classification lives where the session is created (cb-backend `GpuTrainSession` construction, surfaced through the `Runtime` seam). The depth-1/RMSE-or-Logloss/Plain/fold_count=1 coverage check is the gate; anything outside it → `None` → CPU. A covered session that nonetheless cannot grow a tree mid-run is a hard `CbError` (panic-free), NOT a silent CPU graft.

**Kaggle CUDA harness form (BENCH-01 / BENCH-02)**
- **D-10-03:** **Script + README only.** Commit runnable Python (e.g. `bench/cuda_oracle.py`, reusing/extending the existing root `benchmark*.py`) plus a `bench/README.md` documenting the manual Kaggle notebook steps (build `--features cuda` wheel → run oracle → run speed). The user assembles/runs the Kaggle notebook each phase from the README. Most flexible; keeps all logic diffable and out of notebook cells.
- **D-10-04:** The harness runs BOTH the correctness oracle (blocking gate, ≤1e-5 depth-1) AND the wall-clock speed measurement (warm-run / JIT-excluded, train-only). Correctness is a hard gate BEFORE any speed number is reported. It is a **human-gated external step** — the user runs the notebook; ROCm in-env is an optional compile/smoke convenience, NOT a gate (no requirement satisfied by ROCm alone).
- **D-10-05:** Correctness fixtures are committed as small deterministic files generated in-env (the same fixture the in-env build verifies), so the Kaggle run is reproducible and diffable. (Planner/researcher to confirm exact fixture format + how the human sign-off is recorded — a committed RESULTS log is the expected pattern.)

**Reduction-determinism spike depth (SC5, feeds Phase 11)**
- **D-10-06:** **Runnable on-device micro-benchmark.** Prototype ALL three candidates as small on-device kernels — (a) fixed-point i64 atomics, (b) private-histogram merge, (c) two-pass segmented reduce — and measure BOTH determinism error (vs CPU f64) AND wall-clock. Produce `SPIKE-REDUCTION.md` with a real err+ms comparison table and a recommendation that feeds Phase 11's histogram kernel.
- **D-10-07:** Run the spike on Kaggle CUDA for the authoritative numbers; the ROCm in-env build is a fast local smoke check. Note gfx1100 lacks f64 atomic-add (the in-env smoke may need the HostSumFallback path, per Phase 7.6) — the spike must record where each candidate is/isn't viable per backend.

**Depth-1 oracle configs + speed framing (GPUT-04 / BENCH-02)**
- **D-10-08:** Oracle BOTH **RMSE and Logloss** depth-1 (Plain, fold_count=1) ≤1e-5 on Kaggle CUDA — SC1 names both; exercises the der1 path AND the Logloss der seam.
- **D-10-09:** **Depth-1 device fit MUST beat (≥) the CPU wall-clock**, not merely be reported as-is. The user explicitly rejected the "depth-1 is launch-bound, accept device≈/slower" framing. **Implication (firm):** Phase 10 therefore carries a launch-overhead-reduction obligation beyond the bare success criteria — depth-1 is kernel-launch-bound, so beating CPU likely requires fused/batched kernel launches and/or a persistent grow kernel, plus per-fit residency (already required) so no upload/readback dominates. This is a Phase-10 deliverable, not deferred.
- **D-10-10:** Speed is measured device vs CPU. Baseline = both in-env CPU (dev iteration) AND a CPU run on the SAME Kaggle hardware (apples-to-apples for the official number), plus vs official CatBoost GPU where a comparable depth-1 config exists.

### Claude's Discretion
- Exact `DeviceGrownTree` struct fields and the precise host-typed signatures of the three seam methods (within GPUT-01's named shape).
- Internal `GpuTrainSession` handle layout / lifetime mechanics (reusing Phase 7.2 der handles + Phase 7.5 `grow_boosting_pass`).
- Fixture file format and the exact RESULTS sign-off log structure.
- Micro-benchmark kernel sizes / problem shapes for the reduction spike.

### Deferred Ideas (OUT OF SCOPE)
- depth>1 partition-aware histograms, Newton der2, Cosine GPU score, reduction-determinism PRODUCTION kernel — **Phase 11** (the spike here only recommends; Phase 11 builds).
- CTR / pairwise / multiclass / ordered-boosting device paths — **Phase 12** (each lands behind the same per-fit fallback gate with its own ε=1e-4 + speed sign-off).
- Comprehensive final speed-parity sign-off vs official CatBoost GPU (BENCH-03) — **Phase 13** (aggregates the per-phase BENCH-02 checks).
- Estimated-feature stored-border-VALUE quantization-grid parity (FEAT-07 backlog) — NOT folded into Phase 10.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| GPUT-01 | `Runtime` grow-tree seam (`begin_device_training` / `grow_tree_on_device → CbResult<Option<DeviceGrownTree>>` / `end_device_training`), CubeCL-free host-typed, `Ok(None)`→CPU fallback | §Standard Stack (seam shape), §Pattern 1 (default-impl seam mirroring `compute_gradients_grouped`), §Code Examples (signatures + `DeviceGrownTree`) |
| GPUT-02 | `GpuTrainSession` owns one `ComputeClient` + persistent handles; quantized matrix uploaded once per `fit()` | §Pattern 2 (session as `RefCell<Option<…>>` inside `GpuBackend`), §Pattern 3 (handle layout), §Don't Hand-Roll |
| GPUT-03 | Gradients/approx device-resident; per-tree `der1` read-back eliminated; only O(1) BestSplit + 2^depth part-stats cross per level | §Pitfall 1 (the EXISTING `grow_boosting_pass` still reads back leaf_of+der1 — this is the real work), §Pattern 3, §Code Examples (on-device approx update kernel) |
| GPUT-04 | Depth-1 oblivious tree (RMSE/Logloss, Plain, fold_count=1) grown on device ≤1e-5 vs CPU, oracle-tested on Kaggle CUDA | §Validation Architecture, §Depth-1 Oracle Fixtures, §Pattern 4 (split-descriptor mapping device→cb-train) |
| BENCH-01 | Reproducible Kaggle CUDA oracle/speed harness — `--features cuda` wheel, correctness gate then speed | §Kaggle CUDA Harness, §Environment Availability |
| BENCH-02 | Standing per-phase speed check (enforced 10→13) — device vs host-CPU baseline | §Kaggle CUDA Harness, §Depth-1 device>CPU feasibility (ESCALATION FLAG) |
</phase_requirements>

## Summary

Phase 10 has **two genuinely distinct bodies of work** that the planner must keep separate: (1) a small, low-risk **trait-seam + wiring** task (GPUT-01, GPUT-02, GPUT-04, BENCH-01/02) that mirrors patterns already shipped in Phases 7.2/7.5/8; and (2) a **real new residency engineering task** (GPUT-03) plus a **speed obligation that may be physically infeasible at the wrong dataset scale** (D-10-09). The wiring is mostly assembly of existing parts: `grow_oblivious_tree_into` (`gpu_runtime/mod.rs:1641`) already grows a depth-1 device tree; the `Runtime` trait (`cb-compute/runtime.rs:897`) already shows the exact default-impl pattern (`compute_gradients_grouped`, line 944) to copy for the three new seam methods; `GpuBackend` (`gpu_backend.rs:47`) is the zero-sized type that will hold the session.

The **critical finding** the planner must internalize: the existing Phase-7.5 `grow_boosting_pass_into` (`gpu_runtime/mod.rs:1920`) does **NOT** yet satisfy GPUT-03/D-05. It updates the running `approx` **on the host** (line 2001) using a read-back `leaf_of` (n elements, read at line 1788 inside `grow_oblivious_tree_into`) and reads back `der1` (n elements, line 2019) every iteration. Those are exactly the bulk host↔device crossings GPUT-03 demands be eliminated. So GPUT-03 is not "reuse `grow_boosting_pass`" — it is "refactor the per-tree `_into` launchers to take session-owned resident handles, add a tiny on-device `approx += lr·leaf_value[leaf_of]` kernel, and keep approx+der1 device-resident so neither `leaf_of` nor `der1` crosses in the hot path." This is the highest-engineering-content task in the phase.

**The escalation flag (highest priority, D-10-09):** Depth-1 device > CPU wall-clock is **dataset-scale-dependent and is almost certainly NOT achievable at the current `benchmark.py` scale (1000×20)**. A depth-1 tree is the most launch-overhead-bound case in the whole milestone (one histogram + one argmin + one partition per tree, over trivial per-object work). At n≈1000 the CPU does this in microseconds; no amount of kernel fusion makes a GPU launch + driver round-trip competitive. Device wins on depth-1 **only at large n** (≈10⁵–10⁶+ rows) where the O(n·features) histogram parallelizes enough to amortize launch latency. **Recommendation: the planner must pin the BENCH-02 depth-1 speed bar to a large-n dataset and explicitly escalate to the user that "depth-1 device ≥ CPU" is false at small n** — do NOT let a plan promise it at `benchmark.py`'s default size. See §Depth-1 device>CPU Feasibility.

**Primary recommendation:** Decompose into 4 sub-waves — (W1) seam + coverage gate + CPU-fallback wiring (no perf claims); (W2) GpuTrainSession residency + on-device approx/der (GPUT-03, the real work); (W3) depth-1 RMSE+Logloss oracle fixtures + the Kaggle harness (BENCH-01); (W4) the reduction-determinism spike (independent, parallelizable). Treat D-10-09's speed bar as a W3 gate against a large-n dataset, with the small-n infeasibility escalated before W3 plans lock.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Grow-tree trait contract (host-typed) | `cb-compute` (`Runtime` trait) | — | Trait must stay CubeCL-free so `cb-train` never pulls a backend dep (the feature-unification landmine). Plain host structs cross the boundary. |
| Coverage gate (depth-1/RMSE-or-Logloss/Plain/fold=1 → Some/None) | `cb-backend` (`GpuTrainSession::begin`) | `cb-compute` (seam return type) | D-10-02: the classification lives where the session is created; surfaced through the seam as `Option<…>`. |
| Device grow loop (hist+score+split+partition) | `cb-backend` (`gpu_runtime`) | — | Already exists (Phase 7.5). CubeCL behind the seam. |
| Resident handles (cindex/indices/weight/approx/der1) | `cb-backend` (`GpuTrainSession`) | — | One `ComputeClient`; a Handle is bound to its allocating client (residency rule). |
| Boosting iteration loop / model assembly / eval sets | `cb-train` (`train_inner`) | — | The loop, leaf folding, and `Model` build stay in cb-train; the device path is a per-iteration branch returning a host `DeviceGrownTree`. |
| Split descriptor → `cb_model` tree | `cb-train` (`from_trained`) | `cb-backend` (returns feature+bin) | Device returns `(feature, bin_id)`; cb-train maps `bin_id → border = feature_borders[feature][bin_id]`. |
| Compile-time backend selection | `catboost-rs` / `catboost-rs-py` (`fit()`, Cargo features) | — | Exactly one backend feature active; no runtime dispatch. |
| Kaggle CUDA oracle + speed harness | `bench/` (Python script + README) | maturin `--features cuda` wheel | Human-gated external step; ROCm in-env is smoke-only. |

## Standard Stack

### Core (all already in-tree — Phase 10 adds NO new compute crates)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `cubecl` | latest (workspace-pinned; `cuda`/`hip`/`wgpu`/`cpu` facade features) | GPU kernel authoring + `ComputeClient`/`Handle` residency | Already the project's mandated GPU layer (AGENTS.md); kernels portable across cuda/rocm/wgpu from one source |
| `cb-compute` | local path | The `Runtime` trait seam (CubeCL-free) | Where GPUT-01's three methods land, alongside `compute_gradients` |
| `cb-backend` | local path | CubeCL impls + `GpuTrainSession` + `gpu_runtime` grow loop | Owns ALL `unsafe`/CubeCL; never depends on `cb-train` |
| `cb-core` | local path | `CbResult`/`CbError` (`error.rs:10`, `:17`) | `thiserror`-based error type the seam returns |
| `cb-train` | local path | `train_inner` boosting loop + grower dispatch + `Model` assembly | Drives the seam; must NOT be depended on by cb-backend |
| `maturin` | `>=1.9.4,<2.0` | builds the `--features cuda` abi3 wheel for Kaggle | Already the Python packaging backend (`catboost-rs-py/pyproject.toml`) |

### Supporting (Kaggle harness only)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `numpy` | latest (already in `benchmark.py`) | fixture generation / array I/O | bench script |
| `catboost` | 1.2.x (already in `.venv`) | official CatBoost GPU baseline (`task_type='GPU'`) for the head-to-head | bench speed comparison where a comparable depth-1 config exists |
| `criterion` | 0.7.x (dev-dep, roadmap Phase 13) | OPTIONAL in-env ROCm relative-timing regression | not required in Phase 10; do not add unless a plan needs it |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Per-tree `grow_tree_on_device` seam | Reuse whole-loop `grow_boosting_pass` as a bypass | REJECTED — `grow_boosting_pass` owns the entire loop and does host approx-update + der read-back (violates D-05); also bypasses cb-train's leaf-value/eval-set/model machinery. Per-tree seam fits `train_inner`'s existing loop and D-10-01's per-fit gate. |
| `DeviceGrownTree` carrying `leaf_of` (n) | omit `leaf_of` from the production return | `leaf_of` (n elements) crossing per tree violates D-05; needed only for the oracle structure check. Return it ONLY behind an oracle/debug path, or read it once at end of fit. |
| Notebook `.ipynb` checked in | script + README (D-10-03) | LOCKED: script+README keeps logic diffable. |

**Installation:** No new Rust crates. Kaggle wheel build (documented in `bench/README.md`):
```bash
# in crates/catboost-rs-py/ on the Kaggle CUDA notebook
maturin build --release --no-default-features --features cuda
pip install target/wheels/catboost_rs-*.whl
python bench/cuda_oracle.py   # correctness gate THEN speed
```

**Version verification:** maturin pin confirmed at `crates/catboost-rs-py/pyproject.toml:2` (`maturin>=1.9.4,<2.0`). cuda feature passthrough confirmed at `crates/catboost-rs/Cargo.toml:21` (`cuda = ["cb-backend/cuda", ...]`) and `cb-backend/Cargo.toml:37` (`cuda = ["cubecl/cuda"]`). `SelectedRuntime` resolves to `cubecl::cuda::CudaRuntime` under `--features cuda` (`cb-backend/src/lib.rs:75`).

## Package Legitimacy Audit

> Phase 10 installs **no new external packages** into the Rust workspace. The only external runtime deps touched are already-present Python packages (`numpy`, `catboost`, `maturin`) used by the existing `benchmark*.py` scripts and the Kaggle notebook. No SLOP/SUS risk introduced.

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| maturin | PyPI | mature | very high | github.com/PyO3/maturin | OK | Already in use (pyproject) |
| numpy | PyPI | mature | very high | github.com/numpy/numpy | OK | Already in use |
| catboost | PyPI | mature | high | github.com/catboost/catboost | OK | Already in `.venv` (oracle baseline) |

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```text
catboost-rs fit()  (builder.rs:333 — compile-time backend select, line 357-358)
    │  feature_values (f32 SoA) + feature_borders (f64) + target + weights + params
    ▼
cb_train::train<R: Runtime>  (boosting.rs:1870) ──► train_inner (boosting.rs:2101)
    │
    │  (1) begin_device_training(loss, depth, boosting_type, fold_count, n, ...)   ◄── NEW seam call
    │            │ GpuBackend → GpuTrainSession::begin (cb-backend)
    │            │   coverage gate (D-10-02): depth==1 && RMSE|Logloss && Plain && fold==1 ?
    │            ▼
    │       Some(session)  ────────────────────────────┐         None ──► CPU grower dispatch
    │       (uploads cindex/indices/weight ONCE,        │              (boosting.rs:3203, UNCHANGED — D-04)
    │        allocates approx_h, der1_h on ONE client)  │
    │                                                    ▼
    │  per boosting iteration (the loop train_inner already runs):
    │      (2) grow_tree_on_device(session, &approx OR resident)  ◄── returns CbResult<Option<DeviceGrownTree>>
    │            │ cb-backend: grow_oblivious_tree over RESIDENT der1_h
    │            │   per level: hist (device) → score+argmin (device) → O(1) BestSplit read-back
    │            │              → partition_split (device) → partition_update (device)
    │            │   leaves: 2^depth part_stats read-back → leaf_values (host calc_average)
    │            │   approx_h += lr·leaf_value[leaf_of_h]   (device kernel — NEW, GPUT-03)
    │            │   der1_h = der(approx_h, target)         (device, 7.2 seam — no read-back)
    │            ▼
    │       DeviceGrownTree { splits:[(feature,bin)], leaf_values, [leaf_of for oracle] }
    │            │  cb-train maps bin→border, folds into Model exactly like a CPU GrownTree
    │            ▼
    │  (3) end_device_training(session)  ◄── drops handles / frees client buffers
    ▼
cb_model::Model  ── identical structure whether grown on device or CPU (D-04 fallback)

ORACLE/SPEED (human-gated, off the hot path):
  bench/cuda_oracle.py  ── on Kaggle CUDA: nvidia-smi check → warm fit → depth-1 oracle ≤1e-5
                           (BLOCKING) → warm-run/JIT-excluded train-only wall-clock (device vs CPU)
```

### Recommended Project Structure
```
crates/cb-compute/src/runtime.rs        # +3 default-impl seam methods + DeviceGrownTree struct
crates/cb-backend/src/gpu_backend.rs     # GpuBackend gains RefCell<Option<GpuTrainSession>>
crates/cb-backend/src/gpu_runtime/       # GpuTrainSession + refactor _into launchers to take handles
   mod.rs                                #   + NEW apply_leaf_delta kernel (approx += lr·lv[leaf_of])
crates/cb-train/src/boosting.rs          # train_inner: device branch in the per-iteration body
bench/                                   # NEW
   cuda_oracle.py                        # correctness gate + speed (extends benchmark.py shape)
   README.md                             # Kaggle notebook steps (build wheel → oracle → speed)
   fixtures/                             # committed deterministic depth-1 RMSE + Logloss fixtures
   RESULTS.md                            # committed human sign-off log (per-phase CUDA run record)
crates/cb-backend/src/kernels/           # spike kernels (test-gated) for reduction-determinism
.planning/phases/10-.../SPIKE-REDUCTION.md  # the err+ms comparison table + recommendation
```

### Pattern 1: Default-impl CubeCL-free seam method (GPUT-01)
**What:** Add the three grow-tree methods to the `Runtime` trait with **default implementations returning `Ok(None)`** so every existing `Runtime` impl (`CpuBackend`, and any test runtime) compiles unchanged and transparently falls back to CPU. Only `GpuBackend` overrides them. This is the EXACT pattern already used by `compute_gradients_grouped` (`runtime.rs:944-954`, default-delegates).
**When to use:** Always for this seam — it preserves D-04 (CPU byte-unchanged) for free.
**Example:** see §Code Examples.

### Pattern 2: Session owned by the zero-sized backend via `RefCell<Option<…>>`
**What:** `GpuBackend` is currently a zero-sized `#[derive(...)] pub struct GpuBackend;` (`gpu_backend.rs:47`). To hold per-fit state without changing the `&self` trait signature, give it `RefCell<Option<GpuTrainSession>>` interior mutability. `begin_device_training(&self, …)` sets it; `grow_tree_on_device(&self, …)` borrows it; `end_device_training(&self)` clears it.
**When to use:** GPUT-02. Roadmap Notes (line 67) explicitly names this: "owned by `GpuBackend` via `RefCell<Option<…>>`".
**Caveat:** `GpuBackend` is currently `Copy`/`Default`/zero-sized and `builder.rs:358` binds `let backend = GpuBackend;`. Adding a `RefCell` field makes it non-`Copy` and non-zero-sized — verify no caller relies on `Copy` (the Phase 8 facade binds it by value once and passes `&backend`, so this is fine, but the planner must grep for `GpuBackend` copies). `RefCell` is `!Sync`; the `Model` Send+Sync contract is about the *trained model*, not the backend, so this is acceptable for a single-threaded `fit()`.

### Pattern 3: One-client residency; refactor `_into` launchers to accept handles
**What:** The session allocates ONE `ComputeClient` (`<SelectedRuntime as cubecl::Runtime>::client(&device)`) and uploads `cindex` (feature-major u32, `cindex[feature*n+obj]`), `indices`, `weight` ONCE. It holds `approx_h` and `der1_h` as resident handles. Today `grow_oblivious_tree_into` (`gpu_runtime/mod.rs:1641`) **re-uploads** der1/weight/cindex/indices every call (lines 1704-1708) — the planner must add a handle-taking variant (e.g. `grow_oblivious_tree_resident(client, &handles, n_bins, …)`) so the matrix uploads once (GPUT-02) and der1 is the resident handle (GPUT-03).
**When to use:** GPUT-02 + GPUT-03.
**Residency rule (HARD):** never read a `Handle` through a client other than the one that allocated it (`gpu_runtime` Pitfall 3, documented throughout). The session owns the client for the whole fit.

### Pattern 4: Device split-descriptor → cb-train `Split`
**What:** The device returns `splits: Vec<(u32 feature, u32 bin_id)>` where the pass test is `cindex[feature*n+obj] > bin_id` (verified in `grow_loop.rs:138` and `gpu_runtime` partition kernel). The cb-train `Split { feature: usize, border: f64 }` (`tree.rs:109`) pass test is `value > border`. The mapping is **`border = feature_borders[feature][bin_id]`** — the `bin_id`-th ascending border of that feature. So `DeviceGrownTree` carries `(feature, bin_id)` and `train_inner` (or `cb_model::from_trained`) resolves the border from the same `feature_borders` it already holds. This is the one non-obvious correctness join; get it wrong and structure diverges.
**When to use:** GPUT-04 (folding the device tree into the model identically to CPU).

### Anti-Patterns to Avoid
- **Adding `cb-train` to `cb-backend`'s deps** to reuse `FeatureMatrix`/`Split`/leaf logic. HARD LANDMINE — feature unification activates `cb-backend/cpu` alongside `rocm`, `SelectedRuntime` mis-resolves, kernels fail to build. Transcribe inline (memory `phase75-grow-loop-outcome`).
- **`-inf` float literal in any `#[cube]` kernel** (e.g. the new `apply_leaf_delta` or any spike kernel). HIP/gfx1100 JIT rejects `double(-inf)`. Use the `f32::MIN` sentinel (memory `cubecl-hip-no-inf-literal`). Invisible to cpu/wgpu `cargo check`; fails only on rocm GPU — run the rocm smoke suite after any `#[cube]` change.
- **Reading the full histogram / partition / `leaf_of` buffer to host per level.** The FORBIDDEN D-05 hybrid. Only the O(1) BestSplit + 2^depth part-stats may cross per level.
- **Reporting a speed number before the correctness oracle passes.** D-10-04: correctness is a blocking gate.
- **Promising depth-1 device ≥ CPU at small n.** See escalation flag.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Device histogram + score + argmin + partition | A new depth-1 kernel chain | `grow_oblivious_tree_into` (`gpu_runtime/mod.rs:1641`) + its `launch_*_into` primitives | Already exists, oracle-validated in Phase 7.5; depth-1 is its exact MVP |
| Device der1/der2 (RMSE/Logloss) | New gradient kernels | Phase 7.2 der seam (`launch_der_binary_into`, `gpu_backend.rs:81/89`) | Already device-resident, oracle-locked; RMSE grad + Logloss grad/hessian present |
| Leaf value formula | Custom average | `cb_compute::calc_average(sum, cnt, scaled_l2)` (used at `gpu_runtime:1780`) | FROZEN formula, identical host/device |
| CPU fallback correctness | A device "approximation" of uncovered cases | `Ok(None)` → existing CPU grower dispatch (`boosting.rs:3203`, UNCHANGED) | The CPU path is the oracle and the safety net (D-04) |
| Trait extension that keeps CPU untouched | New trait / parallel hierarchy | Default-impl methods on `Runtime` (copy `compute_gradients_grouped` shape, `runtime.rs:944`) | Zero churn to every existing impl |
| Speed benchmark scaffold | New harness from scratch | Extend `benchmark.py`/`benchmark_fast.py`/`benchmark_small.py` shape | Already does warm/timed CatBoost-vs-rs comparison |

**Key insight:** ~80% of Phase 10's device machinery already exists and is oracle-validated. The new code is: the 3-method seam (thin), the `GpuTrainSession` residency wrapper, ONE small `apply_leaf_delta` device kernel (to keep approx-update on device), the bin→border join, the Kaggle harness, and the spike. Resist re-deriving kernels.

## Runtime State Inventory

> Phase 10 is greenfield-additive (new seam + new session + new bench dir); it changes no stored data, services, or OS state. The one "state" concern is CubeCL device-buffer lifetime.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — no datastore keys/IDs renamed | none |
| Live service config | None | none |
| OS-registered state | None | none |
| Secrets/env vars | `DEBUG`/`OUT_DIR` only (build.rs, unrelated); Kaggle notebook needs no secrets | none |
| Build artifacts | The `--features cuda` wheel is built fresh on Kaggle each run (not committed); fixtures ARE committed (D-10-05) | document wheel build in README; commit fixtures + RESULTS log |
| Device-buffer lifetime (CubeCL) | Session handles must be freed at `end_device_training`; a `Handle` outliving its client is UB-adjacent | `end_device_training` drops the session (and thus the client + handles) deterministically |

## Common Pitfalls

### Pitfall 1: Assuming `grow_boosting_pass` already satisfies GPUT-03
**What goes wrong:** A plan says "wire `grow_boosting_pass` and GPUT-03 is done." It is not. `grow_boosting_pass_into` (`gpu_runtime/mod.rs:1920`) reads `leaf_of` back to host (via `grow_oblivious_tree_into` line 1788), updates `approx` **on the host** (line 2001), then recomputes `der1` on device and **reads it back** (lines 2018-2027) every iteration. Two n-length crossings per tree — the exact thing D-05/GPUT-03 forbids.
**Why it happens:** Phase 7.5 was a structural-parity MVP; residency was out of its scope.
**How to avoid:** GPUT-03's deliverable is (a) keep `approx_h`/`der1_h` resident on the session, (b) add a small `#[cube]` kernel `apply_leaf_delta(approx, leaf_of, leaf_values_2pow_depth, lr)` doing `approx[i] += lr * leaf_values[leaf_of[i]]` on device, (c) recompute der1 on device WITHOUT read-back (chain into the next tree's grow). `leaf_of` then crosses only for the oracle, not the hot path.
**Warning signs:** any `client.read_one` / `read_u32_handle` / `read_part_stats_f64` of an n-length buffer inside the per-tree loop.

### Pitfall 2: Coverage gate too narrow or too wide (D-10-02)
**What goes wrong:** Gating on the wrong predicate either silently runs CPU when device was intended, or returns `Some` for a config the device grow loop will then reject mid-run (which D-10-02 says must be a hard `CbError`, not a graft).
**How to avoid:** Gate `Some` iff `depth == 1 && matches!(loss, Rmse|Logloss) && boosting_type == Plain && fold_count == 1` (plus: no CTR, no pairwise, no multiclass, score_function compatible with the L2/Cosine device calcers). Logloss additionally requires the device der1 path (present, `gpu_backend.rs:89`) — but note depth-1 leaf values for Logloss use first-order `calc_average` (Newton der2 is Phase 11); confirm the depth-1 Logloss CPU reference also uses the gradient leaf method for fold_count=1 so the ≤1e-5 oracle is meaningful.
**Warning signs:** the gate references params the seam signature doesn't receive — make sure `begin_device_training` takes enough of `BoostParams` to decide (loss, depth, boosting_type, fold_count, score_function).

### Pitfall 3: Logloss depth-1 ≤1e-5 may need der2/leaf-method care
**What goes wrong:** D-10-08 demands BOTH RMSE and Logloss depth-1 ≤1e-5. RMSE leaf value = Σder1/(Σweight+l2) via `calc_average` (first-order) — exact. Logloss's CPU default leaf-estimation is **Newton** (uses der2); the device depth-1 path computes first-order `calc_average`. If the CPU reference for the Logloss fixture uses Newton leaf values, device (gradient) ≠ CPU (Newton) and ≤1e-5 fails — NOT a bug, a method mismatch.
**How to avoid:** Pin the Logloss depth-1 fixture's CPU reference to the SAME leaf method the device uses (gradient/first-order) for the oracle, OR add the depth-1 Newton der2 leaf step on device (small — der2 over a 2-leaf partition). The planner MUST decide this explicitly; it is the single most likely reason GPUT-04 Logloss misses ≤1e-5. (Memory `phase64-score-fns-gpu-only`: Newton live-search der2 was deferred to "Phase 7 GPU"; Newton leaf der2 is formally Phase 11 (GPUT-07) — so prefer pinning the fixture to gradient leaves for Phase 10.)
**Warning signs:** Logloss oracle ~1e-2..1e-3 off while RMSE is ~1e-9.

### Pitfall 4: `GpuBackend` losing `Copy`/zero-sized-ness
**What goes wrong:** Adding `RefCell<Option<GpuTrainSession>>` makes `GpuBackend` non-`Copy`. Any `let b2 = backend;` (copy) becomes a move; any `&backend` passed to `train` is fine.
**How to avoid:** grep `GpuBackend` for by-value uses; the facade (`builder.rs:358`) binds once and the trait takes `&self`, so a single owner is correct. Keep `Default` (init the `RefCell` to `None`).

### Pitfall 5: `#[cube]` `-inf` and atomic non-determinism in the spike
**What goes wrong:** Spike kernel (a) (i64 fixed-point atomics) or any sentinel use of `-inf` builds on cpu/wgpu/cuda but fails the rocm smoke JIT. Float-atomic candidates produce run-to-run-varying sums (manual `08_atomic_contention.md`: "Float atomic add is inherently order-nondeterministic").
**How to avoid:** Use `f32::MIN` sentinel; for the spike, fixed-point i64 atomics are deterministic *because* integer add is associative/exact — that's the whole point of candidate (a). Record per-backend viability (gfx1100 lacks f64 atomic-add → candidate may need HostSumFallback, per Phase 7.6).
**Warning signs:** rocm suite goes red after a `#[cube]` change that cpu/cuda accepted.

### Pitfall 6: Quoting a fast-but-wrong CUDA speed number
**What goes wrong:** Reporting wall-clock before the oracle passes, or timing a cold (JIT-compiling) first fit, or stopping the clock before CubeCL's lazy queue drains.
**How to avoid:** harness order = `nvidia-smi` (confirm CUDA active) → warm one untimed fit → re-run depth-1 oracle ≤1e-5 (BLOCKING) → time train-only with a read-back/predict to drain the lazy queue before stopping the clock (Phase 13 Notes precedent).

## Code Examples

### The three seam methods (GPUT-01) — add to `Runtime` (`cb-compute/src/runtime.rs:897`)
```rust
// Source: pattern transcribed from compute_gradients_grouped default-impl (runtime.rs:944)
// CubeCL-FREE host-typed structs only (no cubecl types leak through the trait).

/// One device-grown tree, host-typed (D-05: only descriptors cross).
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceGrownTree {
    /// Per-level chosen split as (feature_index, bin_id). The pass test is
    /// `quantized_bin[feature] > bin_id`; the caller resolves
    /// `border = feature_borders[feature][bin_id]`.
    pub splits: Vec<(u32, u32)>,
    /// Per-leaf value (UN-scaled by learning_rate — cb-train scales it,
    /// matching the CPU GrownTree contract). Length 2^depth.
    pub leaf_values: Vec<f64>,
    /// Per-object leaf index, ONLY populated for the oracle structure check
    /// (empty in the production hot path so the n-length buffer never crosses
    /// per tree — D-05). Length 0 or n.
    pub leaf_of: Vec<u32>,
}

/// Opaque per-fit device session token. The trait is CubeCL-free, so the
/// session itself lives in the backend; the trait only needs to know one is
/// active. Implementations that cannot train on device leave these defaulted.
pub trait Runtime {
    // ... existing compute_gradients / compute_gradients_grouped ...

    /// Decide ONCE per fit whether this config grows on device (D-10-01/02).
    /// Returns Ok(true) if a session was started (subsequent grow_tree_on_device
    /// calls will return Some), Ok(false) → the whole fit uses the CPU grower.
    /// Default: false (CPU). Only GpuBackend overrides.
    fn begin_device_training(
        &self,
        loss: &Loss,
        depth: usize,
        boosting_type_is_plain: bool,
        fold_count: usize,
        score_function: EScoreFunction,
        cindex_feature_major: &[u32], // length n_features * n
        weight: &[f64],
        n: usize,
        n_features: usize,
        n_bins: usize,
        learning_rate: f64,
        scaled_l2: f64,
    ) -> CbResult<bool> {
        let _ = (loss, depth, boosting_type_is_plain, fold_count, score_function,
                 cindex_feature_major, weight, n, n_features, n_bins,
                 learning_rate, scaled_l2);
        Ok(false)
    }

    /// Grow ONE tree on device over the resident (running) approx/der state.
    /// `approx` is the CURRENT running approximant (host copy for the first
    /// tree / fallback); the backend keeps the device der1 resident. Returns
    /// Ok(None) for any case the active session cannot grow → caller uses CPU
    /// for THIS fit (but per D-10-01 the gate already decided per-fit, so a
    /// covered session returning None mid-run is a hard error in the impl).
    fn grow_tree_on_device(
        &self,
        approx: &[f64],
        target: &[f64],
    ) -> CbResult<Option<DeviceGrownTree>> {
        let _ = (approx, target);
        Ok(None)
    }

    /// Tear down the session, freeing the ComputeClient + resident handles.
    fn end_device_training(&self) -> CbResult<()> {
        Ok(())
    }
}
```
*(Signatures are illustrative — exact parameter set is Claude's discretion per CONTEXT.md. The load-bearing constraints: host-typed only, `Ok(None)`/`Ok(false)` defaults, enough params to run the D-10-02 gate.)*

### The on-device approx-update kernel (GPUT-03, NEW) — sketch
```rust
// Source: new #[cube] kernel in gpu_runtime; keeps approx-update OFF the host.
// NO -inf literals; generic-float per AGENTS.md.
#[cube(launch)]
fn apply_leaf_delta_kernel<F: Float>(
    approx: &mut Array<F>,         // length n, resident
    leaf_of: &Array<u32>,         // length n, resident (never read to host)
    leaf_values: &Array<F>,       // length 2^depth, small
    lr: F,
) {
    let i = ABSOLUTE_POS;
    if i < approx.len() {
        let leaf = leaf_of[i];
        approx[i] += lr * leaf_values[leaf];   // approx stays resident
    }
}
```
*(Then der1_h = der(approx_h, target) via the 7.2 seam, chained into the next tree — no read-back. This is the crux of eliminating the per-tree der1 read-back, GPUT-03.)*

### Device-grow branch inside `train_inner` (the wiring, GPUT-04/D-10-01)
```rust
// Source: integration point at boosting.rs:3203 grower dispatch.
// BEFORE the per-iteration loop:
let device_active = runtime.begin_device_training(/* params + cindex + ... */)?;
// ... inside the loop, replacing the CPU grower dispatch WHEN device_active:
let grown_cpu: GrownTree = if device_active {
    match runtime.grow_tree_on_device(&approx, target)? {
        Some(dev) => map_device_tree_to_cpu(&dev, feature_borders), // bin→border (Pattern 4)
        None => return Err(CbError::Degenerate(
            "covered device session failed to grow a tree mid-run".into())), // D-10-02
    }
} else {
    /* the EXISTING CPU dispatch, byte-unchanged — D-04 */
};
// AFTER the loop:
if device_active { runtime.end_device_training()?; }
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| GPU = derivatives only; whole grow loop on host | Device-resident grow loop (`grow_boosting_pass` exists but UNWIRED) | Phase 7.5 (2026-06) | Phase 10 wires it; the >20× gap is the unwired loop, not a tuning bug |
| ROCm in-env = correctness gate; CUDA = speed only | Kaggle CUDA = single authoritative oracle (correctness AND speed); ROCm = smoke only | v1.1 requirements (2026-06-28) | All sign-offs are human-gated Kaggle CUDA runs; the prior asymmetry is gone |
| Bit-exact f64 ≤1e-5 on GPU | ε=1e-4 vs CPU (device math is f32); depth-1 held tighter ≤1e-5 | Phase 7.6 precedent | Depth-1 level-0 whole-dataset histogram IS the exact CPU score → ≤1e-5 justified |

**Deprecated/outdated:**
- The `grow_boosting_pass` host approx-update + der read-back (lines 2001, 2018-2027): NOT to be reused as-is for GPUT-03 — it is the anti-pattern the phase removes.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Logloss depth-1 ≤1e-5 is achievable with FIRST-ORDER (gradient) leaf values if the CPU fixture is pinned to the same method; Newton der2 leaves are Phase 11 | Pitfall 3 / GPUT-04 | If CPU Logloss MUST use Newton leaves for a meaningful oracle, depth-1 Logloss needs a small device der2 leaf step in Phase 10 (scope creep) — ESCALATE |
| A2 | Depth-1 device ≥ CPU wall-clock is achievable ONLY at large n (≈10⁵–10⁶+), NOT at `benchmark.py`'s 1000×20 | Depth-1 Feasibility | If the user insists on small-n, D-10-09 is physically infeasible → MUST ESCALATE before W3 plans lock |
| A3 | Adding `RefCell<Option<GpuTrainSession>>` to `GpuBackend` breaks no caller relying on `Copy`/zero-sized | Pattern 2 / Pitfall 4 | If a Phase-8 path copies `GpuBackend`, the build breaks — grep required |
| A4 | The device `bin_id` indexes `feature_borders[feature]` directly (border index == bin boundary) for the depth-1 single-feature-group case | Pattern 4 | If quantization uses a different bin→border indexing (e.g. NaN sentinel bin offset), structure diverges; verify against `select_borders_greedy_logsum` output ordering |
| A5 | `maturin build --no-default-features --features cuda` produces a working Kaggle wheel (CUDA toolkit present on Kaggle GPU notebooks) | Kaggle Harness | If Kaggle's CUDA/driver version mismatches CubeCL's cudarc expectations, the wheel won't JIT — the README must pin a known-good Kaggle image |
| A6 | The CubeCL manual lives at `/home/user/Documents/workspace/cubecl_manual/manual/cubecl/` (lowercase), NOT `/Cubecl/` as AGENTS.md/CONTEXT.md state | Sources | Agents following the documented path get "No such file"; use the lowercase path. INDEX is `INDEX.md`. |

## Open Questions (RESOLVED)

> All three resolved during planning (2026-06-29); markers added so the executor does not treat them as open decisions.

1. **Logloss depth-1 leaf method (gradient vs Newton)** — drives whether GPUT-04 Logloss fits in Phase 10 cleanly.
   - What we know: device depth-1 computes first-order `calc_average`; CPU Logloss default is Newton.
   - What's unclear: whether the depth-1 Logloss oracle fixture can/should use gradient leaves.
   - Recommendation: pin the fixture to gradient leaves (first-order) for Phase 10; defer Newton leaves to Phase 11 (GPUT-07).
   - **RESOLVED:** Plan 10-05 pins the Logloss depth-1 fixture's CPU reference to first-order (`calc_average`) leaves; Newton der2 deferred to Phase 11 (GPUT-07). The ≤1e-5 oracle is method-consistent.

2. **Multi-block scan carry ("Open Q2", flagged for Phase 11)** — the spike should verify it now against the manual.
   - What we know: the reduction spike (D-10-06) prototypes 3 candidates; the manual's `08_atomic_contention.md` + scan/reduce docs cover the carry.
   - What's unclear: whether a single-block scan suffices for depth-1's whole-dataset histogram or a multi-block carry is needed at the bench's large n.
   - Recommendation: the spike records the carry behavior per candidate; Phase 11 consumes it. Verify against `/home/user/Documents/workspace/cubecl_manual/manual/cubecl/` scan docs (lowercase path).
   - **RESOLVED:** Plan 10-06 (reduction-determinism spike) records the carry behavior per candidate and feeds SPIKE-REDUCTION.md → Phase 11. Verified against the lowercase manual path.

3. **`begin_device_training` parameter surface** — must carry enough to run the gate AND upload the matrix.
   - Recommendation: pass the already-built feature-major `cindex` (cb-train quantizes once), `weight`, dims, `learning_rate`, `scaled_l2`, and the gate predicates (loss/depth/boosting_type/fold_count/score_function). Keep it host-typed.
   - **RESOLVED:** Plan 10-01 specifies the host-typed `begin_device_training` parameter surface (cindex/weight/dims/learning_rate/scaled_l2 + gate predicates) in its artifacts.

## Environment Availability

| Dependency | Required By | Available (in-env) | Version | Fallback |
|------------|------------|--------------------|---------|----------|
| AMD gfx1100 / ROCm GPU | in-env smoke (not a gate) | ✓ | ROCm 7.1 | none needed (smoke only) |
| NVIDIA CUDA GPU | BENCH-01/02 authoritative oracle+speed | ✗ (NO CUDA in-env) | — | **Kaggle CUDA notebook (human-gated external step)** |
| maturin | build `--features cuda` wheel | ✓ | >=1.9.4,<2.0 | — |
| catboost (Python) | official-GPU speed baseline | ✓ (.venv) | 1.2.x | skip GPU baseline where no comparable depth-1 config |
| numpy | fixture gen / bench | ✓ | latest | — |
| CubeCL manual | mandatory pre-kernel read | ✓ | — | path is `.../manual/cubecl/INDEX.md` (lowercase, A6) |

**Missing dependencies with no fallback:** none that block in-env work. CUDA correctness/speed sign-off is BY DESIGN a human-gated external Kaggle step — the planner must mark every GPUT-04/BENCH gate as "awaits Kaggle CUDA run," not in-CI.

**Missing dependencies with fallback:** CUDA GPU → Kaggle notebook; the in-env ROCm build provides compile + correctness smoke only.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` (workspace), in SEPARATE test files per CLAUDE.md (no `#[cfg(test)] mod tests` in production files); existing GPU cross-oracles live in `cb-backend/src/kernels/*.rs` gated `#[cfg(test)]` (e.g. `kernels.rs:2774 #[cfg(test)] mod grow_loop;`) |
| Config file | none beyond Cargo; backend selected by feature (`--no-default-features --features rocm` for in-env GPU tests) |
| Quick run command | `cargo test -p cb-backend --no-default-features --features rocm <name>` (in-env GPU smoke) |
| Full suite command | per-crate (disk pressure — memory `disk-pressure-and-full-suite-verification`): `cargo test -p cb-train`, `cargo test -p cb-backend ...` |
| Authoritative GPU oracle | `python bench/cuda_oracle.py` on Kaggle CUDA (human-gated) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| GPUT-01 | seam default-impl returns Ok(None)→CPU; CpuBackend unchanged | unit | `cargo test -p cb-train` (CPU oracle suite stays green = D-04) | ✅ existing CPU oracles |
| GPUT-02 | matrix uploaded once; session holds one client | unit (in-env) | `cargo test -p cb-backend --no-default-features --features rocm session_residency` | ❌ Wave 0 |
| GPUT-03 | no n-length read-back in per-tree loop (instrument crossing count) | unit (in-env) | `cargo test -p cb-backend --features rocm residency_no_readback` | ❌ Wave 0 |
| GPUT-04 | depth-1 RMSE+Logloss device == CPU ≤1e-5 | oracle | in-env: `cargo test --features rocm depth1_oracle`; authoritative: `python bench/cuda_oracle.py` | ❌ Wave 0 + bench |
| BENCH-01 | wheel builds, oracle gates, speed measured | harness | `python bench/cuda_oracle.py` (Kaggle) | ❌ Wave 0 (bench/) |
| BENCH-02 | depth-1 device wall-clock vs CPU recorded | harness | `python bench/cuda_oracle.py` → RESULTS.md | ❌ Wave 0 (bench/) |

### Sampling Rate
- **Per task commit:** `cargo test -p cb-backend --no-default-features --features rocm <touched>` after any `#[cube]` change (rocm catches `-inf`/atomic issues cpu/cuda hide).
- **Per wave merge:** full per-crate CPU suite (D-04 no-regression) + in-env rocm grow/residency smoke.
- **Phase gate:** Kaggle CUDA `cuda_oracle.py` correctness ≤1e-5 (BLOCKING) then speed; RESULTS.md signed off by the user.

### Wave 0 Gaps
- [ ] `bench/cuda_oracle.py` — BENCH-01/02 harness (extends `benchmark.py`)
- [ ] `bench/README.md` — Kaggle notebook steps (D-10-03)
- [ ] `bench/fixtures/` — committed deterministic depth-1 RMSE + Logloss fixtures (D-10-05)
- [ ] `bench/RESULTS.md` — human sign-off log structure (D-10-05)
- [ ] `crates/cb-backend/src/kernels/<residency/session test>.rs` — GPUT-02/03 in-env smoke (separate test file, CLAUDE.md)
- [ ] depth-1 device-vs-CPU oracle test (RMSE + Logloss) — in-env smoke + Kaggle authoritative
- [ ] reduction-determinism spike kernels (3 candidates) + `SPIKE-REDUCTION.md`
- [ ] Verify GPU der seam supports Logloss in the residency loop (gpu_backend.rs:89 — present)

## Security Domain

> `security_enforcement` is not a meaningful axis for this phase: it is an internal numeric-compute library change with no auth, network, session, or untrusted-input surface. The one relevant control is input-validation hygiene already enforced project-wide.

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes (numeric) | Typed `CbError` guards on every length/overflow/shape (existing `gpu_runtime` pattern: `LengthMismatch`/`OutOfRange`/`Degenerate`); no `unwrap`/`panic`/indexing in production (workspace lints + D-13) |
| V6 Cryptography | no | — |

### Known Threat Patterns for {Rust GPU compute}
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Out-of-bounds device buffer access (kernel index) | Tampering / DoS | Bounds-guard in kernel (`if i < arr.len()`); host shape guards before launch; never read a 0-len handle |
| Reading a Handle through the wrong client | Tampering (UB-adjacent) | One client per session; residency rule enforced (Pitfall 3) |
| Silent zero-buffer on read-back failure | Repudiation (wrong result looks valid) | Surface `CbError::Degenerate` on any read-back failure, never a silent zero (existing WR-05 precedent) |

## Sources

### Primary (HIGH confidence — direct codebase read this session)
- `crates/cb-compute/src/runtime.rs:820-955` — `Runtime` trait, `compute_gradients`/`compute_gradients_grouped` default-impl pattern, `Derivatives`, `EScoreFunction`
- `crates/cb-backend/src/gpu_runtime/mod.rs:1524-2043` — `GrownTree`, `grow_oblivious_tree[_into]` (depth-1 MVP, re-uploads per call), `grow_boosting_pass[_into]` (host approx-update + der read-back — the GPUT-03 gap), `BestSplit`/D-05 contract
- `crates/cb-backend/src/gpu_backend.rs:34-160` — `GpuBackend` (zero-sized), Phase 7.2 der seam dispatch (RMSE/Logloss/Mae/Quantile/Focal)
- `crates/cb-train/src/boosting.rs:1870,2101,2996,3203` — `train`, `train_inner`, der1 read-back, grower dispatch
- `crates/cb-train/src/tree.rs:109,196,307` — `Split`, cb-train `GrownTree`, `FeatureMatrix`
- `crates/catboost-rs/src/builder.rs:333-371` — compile-time backend select in `fit()`
- `crates/cb-backend/src/kernels/grow_loop.rs` + `kernels.rs:2774` — device-grow call shape, `cindex[feature*n+obj] > bin` pass test, test-gating
- `crates/catboost-rs-py/pyproject.toml`, `crates/catboost-rs/Cargo.toml:17-34`, `crates/cb-backend/Cargo.toml:34-38`, `crates/cb-backend/src/lib.rs:63-82` — feature passthrough, `SelectedRuntime`, maturin
- `.planning/REQUIREMENTS.md`, `.planning/ROADMAP.md` (Phase 10), `.planning/notes/gpu-training-host-light-root-cause.md`, `10-CONTEXT.md`
- `/home/user/Documents/workspace/cubecl_manual/manual/cubecl/08_atomic_contention.md`, `INDEX.md` — atomic non-determinism, three-tier reduction strategy (feeds the spike)

### Secondary (MEDIUM confidence — project memory)
- Memories: `phase75-grow-loop-outcome`, `cubecl-hip-no-inf-literal`, `phase72-gpu-der-seam-outcome`, `phase76-gpu-tolerance-signoff-outcome`, `phase64-score-fns-gpu-only`, `disk-pressure-and-full-suite-verification`, `phase8-python-bindings-outcome`

### Tertiary (LOW confidence — analytical, unverified on hardware)
- Depth-1 device>CPU feasibility at large vs small n — analytical reasoning from launch-overhead vs O(n·features) parallelism; MUST be confirmed on Kaggle CUDA (this is exactly what BENCH-02 measures)

## Depth-1 device > CPU Feasibility (ESCALATION FLAG — D-10-09)

**This is the loud flag CONTEXT.md §Specific Ideas demands.**

D-10-09 requires the depth-1 device fit to **beat** CPU wall-clock. Adversarial analysis:

- **A depth-1 tree is the single most launch-overhead-bound workload in the entire milestone.** Per tree the device does: 1 histogram fill + 1 score/argmin + 1 partition_split + 1 partition_update + 1 der recompute = a handful of kernel launches, each with fixed driver/launch latency (~µs–tens of µs on CUDA), over per-object work that is trivial at depth 1.
- **At `benchmark.py`'s default scale (n=1000, 20 features), the CPU grows a depth-1 stump in microseconds.** No amount of kernel fusion, persistent-kernel design, or residency makes a GPU launch + host round-trip competitive with that. **Device WILL lose at small n. This is physics, not a tuning gap.**
- **Full residency (already required by GPUT-02/03) removes the per-tree upload/readback** — that is necessary but NOT sufficient. It changes "device hopeless" into "device competitive only past a crossover dataset size."
- **Device wins at large n** (≈10⁵–10⁶+ rows × enough features): the O(n·features) histogram parallelizes across thousands of lanes, amortizing fixed launch latency; the host does linear work. There IS a crossover where device ≥ CPU even at depth 1.
- **Fused/batched launches help but have limits.** Fusing hist+score into one launch and chaining der-recompute without sync reduces the *number* of syncs to ~O(iterations); a persistent grow kernel could in principle keep one kernel resident across the level chain. But at depth 1 there is only one level, so the per-tree launch count is already near-minimal; the lever that actually moves the needle is **dataset size**, not launch count.

**Recommendation (firm):**
1. **Pin the BENCH-02 depth-1 speed bar to a LARGE-n dataset** (the planner should choose a size — e.g. ≥2×10⁵ rows × ≥50 features — and may need to grow it until device ≥ CPU; the Kaggle run is the arbiter). Add a large-n fixture alongside the small correctness fixture.
2. **ESCALATE to the user, explicitly and before W3 plans lock:** "Depth-1 device ≥ CPU is achievable only at large n; at `benchmark.py`'s 1000×20 it is physically infeasible regardless of optimization. We will measure/claim D-10-09 at large n; we cannot honor it at small n." Do NOT silently relax D-10-09, and do NOT silently assume it holds — surface the dataset-scale dependency as a decision.
3. If, after large-n fused/resident optimization on Kaggle CUDA, depth-1 device still does not beat CPU even at large n, that is the second escalation trigger: report the measured crossover (or its absence) and let the user decide whether D-10-09 stands or relaxes for depth-1 (depth-6 in Phase 11 is where device dominance is unambiguous).

## Metadata

**Confidence breakdown:**
- Seam shape / wiring (GPUT-01/02/04): HIGH — mirrors shipped patterns; all integration points read directly
- Residency engineering (GPUT-03): HIGH on what must change (the existing read-backs are verified), MEDIUM on the new `apply_leaf_delta` kernel landing clean on first try (any `#[cube]` change needs the rocm smoke)
- Logloss depth-1 ≤1e-5 (leaf method): MEDIUM — A1 risk; pin the fixture or scope a small der2 leaf
- Depth-1 device>CPU (D-10-09): LOW/analytical — escalation flagged; only Kaggle CUDA settles it
- Kaggle harness (BENCH-01/02): MEDIUM — maturin/feature plumbing verified; Kaggle CUDA image compatibility (A5) unverified
- Reduction spike: HIGH on approach (manual confirms candidates + atomic non-determinism); MEDIUM on per-candidate numbers (the spike produces them)

**Research date:** 2026-06-29
**Valid until:** 2026-07-13 (14 days — fast-moving GPU phase; codebase line refs may drift as plans land)
