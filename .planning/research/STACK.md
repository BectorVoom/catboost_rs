# Stack Research

**Domain:** Device-resident GPU gradient-boosting training (CubeCL) + cross-framework speed benchmarking
**Milestone:** v1.1 GPU Performance — move the full tree-growth inner loop on-device
**Researched:** 2026-06-28
**Confidence:** HIGH (versions verified against crates.io / PyPI / in-repo `Cargo.lock`; CubeCL APIs verified against the vendored manual and existing `gpu_runtime` code)

## Executive Finding

**This milestone needs almost no new crates.** The GPU compute stack (`cubecl 0.10.0`) is already pinned and already exposes every primitive required for device residency — persistent buffers, device scan/reduce, `Atomic<f64>` fetch-add, plane (warp) ops, lazy async dispatch, and memory-pool tuning. The work is **wiring and kernel coverage**, not dependency acquisition:

1. A new **`Runtime` trait seam** for "grow a tree on device" so `cb_train::train` stops falling through to the CPU grower (the integration gap from the root-cause note).
2. Using **CubeCL APIs we already depend on but do not yet exploit** for residency: hoisted invariant uploads, one threaded `ComputeClient`, batched read-backs, `launch_unchecked`, and the multi-block scan carry (the documented "Open Q2" forward dependency).
3. Two **additive opt-in feature flags** on the existing `cubecl` dependency (`profile-tracy`, `tracing`) for GPU timing.
4. A **benchmark harness** built on the already-installed `catboost==1.2.10` Python package + the existing `cuda`-feature wheel, run on a Kaggle CUDA notebook.

The only crates to *add* are benchmark/profiling tooling. The core danger is architectural, not version-related: **do not add a `cb-train` dependency to `cb-backend`** (feature-unification landmine that breaks the rocm runtime).

## Recommended Stack

### Core Technologies (already present — confirm/retain, do NOT re-add)

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| `cubecl` | **0.10.0** (latest; already pinned in `Cargo.toml:38` + `Cargo.lock`) | GPU kernel authoring + `ComputeClient` device memory/dispatch | Latest release; no bump needed. Already drives all 7.x kernels. All residency primitives below ship in this version. |
| `cubecl` runtime backends | 0.10.0 (cuda / hip-rocm / wgpu / cpu, Cargo-feature-switched via `SelectedRuntime`) | Compile-time backend selection, zero dispatch cost | In-env validation on `rocm` (gfx1100); Kaggle speed run via `cuda` feature. One kernel source, four targets — the v1.0 invariant. |
| `bytemuck` | (workspace-pinned, already used) | Zero-copy host↔device byte casts for read-back | Already the read-back idiom (`cast_slice::<u8,f64>`); reused unchanged for resident-handle round-trips. |
| `cb-core` / `cb-compute` | local | `CbError`/`CbResult`, the cubecl-free `Runtime` trait boundary (D-03) | The new grow-tree seam must live here as plain-type (no cubecl) signatures. |

### New Additions for v1.1

| Library / Feature | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `cubecl` feature `profile-tracy` | 0.10.0 (additive feature on existing dep) | Nanosecond visual kernel/JIT/alloc profiling via Tracy | Dev-only; gate behind a `profiling` cargo feature. Use to attribute per-kernel time while moving the inner loop on-device. |
| `cubecl` feature `tracing` | 0.10.0 (additive feature) | Structured `tracing` events from CubeCL internals (JIT latency, alloc lifecycle) | Pair with `tracing-subscriber`; route to file via `cubecl.toml [profiling] logger`. |
| `tracing-subscriber` | 0.3.x (latest) | Capture/format the `tracing` events CubeCL emits | Only compiled under the `profiling` feature — keep out of the default/release build. |
| `criterion` | 0.7.x (latest) | Rust-side micro/macro benchmarks (warm-run statistics, JIT excluded) | `[dev-dependencies]` on the benchmark crate only. Use for in-env ROCm relative timing (CPU-host-light vs device-resident), NOT for the CUDA head-to-head (that is Python-side). |
| `catboost` (Python) | **1.2.10** (latest on PyPI; already in `.venv`) | The official GPU trainer baseline for the head-to-head speed benchmark | `task_type='GPU'`, `devices='0'`. Driven from the existing `benchmark.py` pattern. |
| `numpy` | latest (already a binding dep) | Shared synthetic dataset generation for both trainers | Already used by `benchmark.py`; identical X/y feed to both sides for a fair wall-clock. |
| `maturin` | latest (already the wheel builder) | Build the `cuda`-feature `catboost_rs` wheel on/for the Kaggle CUDA notebook | The speed run needs a CUDA-compiled wheel; rocm wheels won't run on NVIDIA. |

### Development / Benchmark Environment

| Tool | Purpose | Notes |
|------|---------|-------|
| Kaggle CUDA notebook (NVIDIA T4/P100) | Head-to-head **speed** run vs official CatBoost GPU (CUDA-only) | No NVIDIA in-env; CubeCL `cuda` backend + `cubecl-cuda` compile against Kaggle's CUDA toolkit. Build the wheel with `--features cuda` (or upload a prebuilt manylinux CUDA wheel). |
| ROCm gfx1100 (in-env) | **Correctness** validation ≤1e-5 and relative speed regression | CubeCL kernels are portable; develop + oracle-lock here. Speed numbers here are AMD-relative, not the official-comparison figure. |
| `cubecl.toml` (`[profiling] logger`) | Route CubeCL profiling logs to a file | Project-root config; only meaningful with `tracing`/`profile-tracy` enabled. |

## Installation

```bash
# --- Rust side: additive cubecl features behind a profiling gate (Cargo.toml) ---
# [features]
# profiling = ["cubecl/profile-tracy", "cubecl/tracing", "dep:tracing-subscriber"]
# (cubecl itself stays at the existing 0.10.0 pin — no version change)
cargo add tracing-subscriber --optional        # gated by `profiling`
cargo add criterion --dev                       # benchmark crate dev-dep only

# --- Python benchmark harness (already provisioned in .venv) ---
uv pip install "catboost==1.2.10" numpy         # official GPU baseline (latest)

# --- Kaggle CUDA speed run: build a CUDA wheel ---
maturin build --release --features cuda          # NVIDIA-targeted catboost_rs wheel
```

## CubeCL APIs that deliver device residency (no new dependency)

These are the load-bearing capabilities the milestone must exploit; all exist in `cubecl 0.10.0` and several are already used in `gpu_runtime/mod.rs`.

| Capability | API / Pattern | Status in repo | What v1.1 must do |
|------------|---------------|----------------|-------------------|
| **Persistent buffers across iterations** | `client.create(...)` / `client.empty(...)` return a `Handle` that lives until dropped; thread ONE `ComputeClient` through the whole run (manual §2 "Hoist invariant uploads") | `grow_boosting_pass_into` already threads one client, but **re-uploads `cindex`/`weight` every helper call** | Upload `cindex` (the large feature matrix), `target`, `weight`, `indices` **once** before the boosting loop; keep resident `Handle`s. Only `der1`/`approx`/histograms cycle. This is the single biggest end-to-end win on a discrete GPU. |
| **Avoid per-tree host round-trip** | Keep `der1`/`approx` as device `Handle`s; recompute residual on-device (7.2 der seam already does this) | `grow_boosting_pass` reads `der1` back to host every tree (`read_one` + `cast_slice`) | Eliminate the per-tree `der1` read-back: chain the der seam output handle straight into the next histogram fill. Only O(1)/leaf data should cross (D-05). |
| **Device scan (prefix-sum) across blocks** | `block_scan_kernel` (intra-cube) + plane scan (`Plane` Scan APIs) | Intra-cube only; cross-cube carry is the documented **Open Q2** forward dependency (`launch_block_scan_f64` hard-errors for `n > CUBE_DIM`) | Implement the multi-block scan carry (two-pass or decoupled look-back) — required for partition/leaf-assignment compaction at realistic `n`. This is the highest-uncertainty new kernel work. |
| **Device reduce** | `block_reduce_kernel` (+ plane fold via `Plane::Ops`, host finalize) | Present and oracle-locked | Reuse for leaf-stat sums; finalize order stays host-`sum_f64` for ≤1e-5 parity (D-02). |
| **Device atomics (histogram accumulation)** | `Atomic<F>::fetch_add`; `client.properties().atomic_type_usage()` capability query | Present; f64 on cuda/rocm/cpu, f32 on wgpu | For depth>1 partition-aware histograms keep the f64 atomic merge. If run-to-run determinism becomes a parity blocker, adopt **fixed-point i64/u64 atomics** (manual `09_fixedpoint_atomics.md`) for order-independent exact sums. |
| **Few large kernels vs many tiny** | One grid-addressed launch over all features (2D `CubeCount`: X=feature, Y=row-partition); `launch_unchecked` to drop bounds-check codegen after host validation | Histogram fill already grid-strides; scoring is per-candidate | Collapse the per-feature/per-level host loop into single grid-addressed launches (manual §5) so dispatch overhead is paid once per level, not once per feature. |
| **Hide host-device sync** | Lazy async dispatch — launches queue without blocking; CPU only stalls on `read_one` (manual `05_lazy_execution.md`) | Implicit | Batch the per-level winner read-backs into one `client.read(vec![...])` (manual §3); avoid a sync point per level. |
| **Allocation churn in the boosting loop** | `MemoryConfiguration::ExclusivePages` via `RuntimeOptions` | Default allocator | Optional: for the high-frequency per-tree histogram allocations, an exclusive-pages pool reduces alloc latency. Profile first; only adopt if alloc shows up. |
| **JIT cost** | CubeCL compilation caching (`Cubecl_compilation_caching.md`); autotune persistent on-disk cache (`04_autotune_optimization.md`) | Default | Warm up kernels before timing; exclude first-run JIT from the benchmark figure. |

## Integration with the existing generic Runtime / SelectedRuntime

- **New trait seam location:** add a `grow_tree` / `grow_boosting_pass`-shaped method to `cb_compute::Runtime` (the cubecl-free boundary, D-03). It MUST take plain slices and return a plain tree type (no cubecl types cross the trait), exactly as `compute_gradients` does today. `grow_boosting_pass` already has this shape (`&[f64]`/`&[u32]` in, `GrownModel` out) — but `GrownModel`/`GrownTree` live in `cb-backend`; a plain mirror type must live in `cb-core`/`cb-compute` so the trait signature stays cubecl-free.
- **`GpuBackend` impl:** the new trait method delegates to `crate::gpu_runtime::grow_boosting_pass` over `SelectedRuntime` — the same zero-cost compile-time backend selection the der seam uses. No concrete runtime is named.
- **`CpuBackend` impl:** provides the host grower (transcribe/route the existing `greedy_tensor_search_*` path) so the seam is total across backends.
- **Wiring point:** `cb_train::train::<R: Runtime>` calls the new seam instead of always falling through to the CPU grower; `catboost-rs/src/builder.rs:333-371` already selects `GpuBackend` for `wgpu`/`cuda`/`rocm`.
- **MVP→parity ladder:** the existing device grower is **depth-1, RMSE/L2, plain boosting, fold_count=1, no Newton der2/CTR/pairwise/ordered/multiclass**. Extending it (depth>1 partition-aware histograms first) is the bulk of the kernel work, not a dependency change.

## Benchmark harness design

| Concern | Approach |
|---------|----------|
| Baseline | `catboost==1.2.10`, `CatBoostRegressor(task_type='GPU', devices='0', ...)` — identical `iterations`/`depth`/`learning_rate`/data as `catboost_rs`. |
| Subject | `catboost_rs` wheel built `--features cuda`, run on the same Kaggle GPU. |
| Fairness | Shared `numpy` synthetic dataset (existing `benchmark.py` seed pattern); **warm** timing (one untimed fit to pay JIT/alloc), then `time.perf_counter()` around `fit()`. Because CubeCL dispatch is lazy, force completion by reading the model back / predicting so the GPU queue actually drains before stopping the clock. |
| Metric | Wall-clock training time (and trees/sec); correctness separately oracle-locked ≤1e-5 in-env on ROCm — never on the Kaggle box. |
| Rust-side regression | `criterion` benches on ROCm for "host-light vs device-resident" deltas during development (re-attribute after each change, manual §6). |

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| `cubecl 0.10.0` (keep pin) | Raw `cust`/`cudarc` (direct CUDA) | Never for this project — would break the one-source four-backend invariant and the rocm in-env validation story. Rejected. |
| `Atomic<f64>` histogram merge | Fixed-point i64/u64 atomics (manual 09) | Adopt only if f64-atomic run-to-run non-determinism breaks the ≤1e-5 bar at depth>1 (gfx1100 advertises no f64 atomic but executes it — already handled by `AtomicFinalizePath`). |
| `profile-tracy` + `tracing` | `nvprof`/Nsight (CUDA-only), `rocprof` (ROCm) | Vendor profilers are useful on the Kaggle/ROCm box for a deep dive, but CubeCL's built-in `tracing` is portable and attributes JIT/alloc that vendor tools miss. Use vendor tools as a secondary lens only. |
| `criterion` (Rust bench) | Hand-rolled `Instant` loops | Hand-rolled is fine for the Python head-to-head (catboost is Python); `criterion` gives warm-run statistics for the Rust-internal regression checks. |
| `MemoryConfiguration::ExclusivePages` | Default sub-slice allocator | Only switch if profiling shows per-tree allocation latency is material; default is fine until proven otherwise. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| **Adding `cb-train` as a dependency of `cb-backend`** | Cargo feature unification pulls cb-train's deps into the backend crate and **breaks the rocm runtime** (documented landmine, memory `phase75-grow-loop-outcome`) | Transcribe the needed CPU reference logic **inline** into `cb-backend` (the established pattern for the grow loop). |
| Bumping `cubecl` past 0.10.0 | 0.10.0 is the latest release and already supplies every needed primitive; a speculative bump risks kernel-codegen churn on gfx1100 | Stay pinned at 0.10.0; revisit only if a specific missing API is identified. |
| Cubecl types in the `cb_compute::Runtime` trait signature | Violates D-03 (the generic boundary stays cubecl-free); would leak backend types into the boosting loop | Plain slices/`Vec`/`cb-core` tree types across the trait; cubecl stays inside `cb-backend`. |
| `Atomic<f32>` for histogram sums when parity matters | f32 atomic adds are non-deterministic AND precision-losing → can blow the 1e-5 bar | f64 atomics (cuda/rocm) or fixed-point i64 atomics for exact, order-independent accumulation. |
| `-inf` float literals inside `#[cube]` kernels | `F::new(f32::NEG_INFINITY)` emits `double(-inf)` → HIP/gfx1100 JIT reject; invisible to cpu/wgpu `cargo check` (memory `cubecl-hip-no-inf-literal`) | Finite `f32::MIN` sentinel device-side; `f64::NEG_INFINITY` only in host code. |
| Per-tree `cindex` re-upload / per-tree `der1` read-back | The dominant discrete-GPU cost (PCIe re-copy of the feature matrix every iteration) and a per-tree sync stall | Hoist the upload once; keep der/approx resident as handles (manual §2/§3). |
| Cold-run benchmark timing | JIT + first-allocation overstate steady-state by orders of magnitude | Warm up once, time the second run, drain the lazy queue with a read-back/predict before stopping the clock. |

## Stack Patterns by Variant

**If validating correctness (in-env, ROCm gfx1100):**
- Build `--features rocm`; oracle-lock device-resident output ≤1e-5 against the host path and CatBoost.
- f64 atomic executes but is unadvertised — rely on `AtomicFinalizePath::InKernelAtomicF64Unadvertised`, not `HostSumFallback`.

**If running the head-to-head speed benchmark (Kaggle, CUDA):**
- Build `--features cuda` wheel via maturin; baseline `catboost(task_type='GPU')`.
- Warm-run, drain the lazy queue, `perf_counter` the second `fit()`.

**If the inner loop is still transfer/dispatch-bound after residency:**
- Apply manual §5 (collapse per-item launches) + §3 (batch read-backs) + `launch_unchecked`, re-profiling after each (manual §6).

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| `cubecl 0.10.0` | `cubecl-cuda` / `cubecl-hip` / `cubecl-wgpu` 0.10.x | Keep the whole cubecl family on 0.10.x; mixing minors risks IR mismatch. |
| `cubecl` features `profile-tracy` + `tracing` | `tracing-subscriber 0.3.x` | Subscriber must be initialized in-process to capture events; route via `cubecl.toml`. |
| `catboost 1.2.10` (Python) | CUDA compute capability ≥ 3.5; CUDA libs statically linked in the Linux wheel | Kaggle GPUs (T4/P100/compute ≥6.0) satisfy this out of the box. |
| `catboost_rs` `cuda` wheel | Kaggle CUDA toolkit | Build with `--features cuda`; rocm wheels do NOT run on NVIDIA. |
| `pyo3 0.29.0` / `maturin` | Python ≥ 3.12 (abi3-py312) | Unchanged from v1.0; the CUDA wheel uses the same binding surface. |

## Sources

- In-repo: `Cargo.toml:38`, `Cargo.lock` (cubecl 0.10.0 pin); `crates/cb-backend/src/gpu_runtime/mod.rs` (existing `grow_boosting_pass`, scan/reduce/atomic seams); `crates/cb-backend/src/gpu_backend.rs` (derivatives-only `Runtime` impl); `.planning/notes/gpu-training-host-light-root-cause.md`; `benchmark.py`. — Confidence HIGH (direct read).
- CubeCL manual (`/home/user/Documents/workspace/cubecl_manual/manual/cubecl/`): `11_launch_overhead_and_transfers.md`, `05_lazy_execution.md`, `08_atomic_contention.md`, `09_fixedpoint_atomics.md`, `10_grid_stride_occupancy.md`, `profiling_tools.md`, `Tuning_ExclusivePages_Allocator_...md`, `Cubecl_compilation_caching.md`, `04_autotune_optimization.md`, `Cubecl_plane.md`. — Confidence HIGH (curated vendored docs).
- [cubecl — crates.io](https://crates.io/crates/cubecl) (latest 0.10.0) — Confidence HIGH.
- [catboost — PyPI (latest 1.2.10)](https://pypi.org/project/catboost/) + [CatBoost Python installation / GPU docs](https://catboost.ai/docs/en/concepts/python-installation) — Confidence HIGH.
- Project memory: `phase75-grow-loop-outcome` (cb-train→cb-backend landmine), `cubecl-hip-no-inf-literal`, `phase76-gpu-tolerance-signoff-outcome` (gfx1100 f64-atomic / ε=1e-4). — Confidence HIGH (in-env validated).
