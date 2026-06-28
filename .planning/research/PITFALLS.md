# Pitfalls Research

**Domain:** Device-resident GPU gradient-boosting training (CubeCL: cuda/rocm/wgpu/cpu) + speed-parity benchmarking vs official CatBoost GPU — v1.1 GPU Performance milestone
**Researched:** 2026-06-28
**Confidence:** HIGH (internal root-cause note + v1.0 retrospective + established memory landmines; corroborated by XGBoost/LightGBM/CatBoost GPU literature on FP non-associativity and atomicAdd)

> Scope note: v1.0 shipped a **derivatives-only** GPU MVP. der1/der2 run on-device; the *entire* greedy tree-search inner loop (histogram → score → BestSplit → partition → leaf values) runs on the **host CPU**, with a per-tree device→host der read-back. This is the documented root cause of the >20× gap (`.planning/notes/gpu-training-host-light-root-cause.md`). The pitfalls below are ordered so the device-residency + sync-elimination ones (the actual gap) come first, then the determinism-vs-parallel-reduction tension against the ≤1e-5 oracle, then the repo-specific portability landmines, then benchmarking fairness.

---

## Critical Pitfalls

### Pitfall 1: Leaving the inner loop host-side — "GPU backend" that only computes derivatives

**What goes wrong:**
A `Runtime` trait that exposes only `compute_gradients()` forces `fit()` to fall through to the CPU `greedy_tensor_search_*` growers. ~95% of training (histogram build, split scoring, BestSplit, partition/leaf-assign, leaf values) stays on the host. Selecting a GPU backend then adds a per-tree device→host der read-back *without removing any host work* — so the GPU path can be **slower than pure CPU**. This is exactly the v1.0 state.

**Why it happens:**
Derivatives are the easiest, most self-contained kernel and make a satisfying "GPU works" demo. The boosting loop `cb_train::train::<R: Runtime>` is generic over `Runtime`, but there is no trait seam for "grow a tree on device," so the fast path that *was already built* (`grow_boosting_pass` at `gpu_runtime/mod.rs:1890`) is only ever called from tests, never wired into `cb-train`.

**How to avoid:**
Add a `Runtime` trait seam for on-device tree growth (e.g. `grow_tree`/`grow_pass`) and wire `grow_boosting_pass_into` into `cb_train::train`. Treat "GPU is selected but inner loop is on host" as a hard failure, not a silent fallback: a `cuda`/`rocm`/`wgpu` build that routes any per-object inner-loop work through the host CPU should be caught by a benchmark gate, not shipped.

**Warning signs:**
- GPU utilization near 0% during training; CPU pegged at one core.
- GPU train time ≥ CPU train time on a large dataset.
- A profiler shows time dominated by `tree.rs` host functions (`reduce_leaf_stats`, `select_best_candidate`, `assign_leaves`) rather than kernel launches.

**Phase to address:** Phase 1 (Runtime grow-tree seam + wire `grow_boosting_pass` into `cb-train`) — the highest-leverage phase of v1.1.

---

### Pitfall 2: Per-level / per-tree blocking read-backs (host↔device sync stalls)

**What goes wrong:**
Reading a device buffer back to the host (`client.read_one`) is a synchronization point: it drains the queue and blocks until the kernel finishes. Doing this per feature, per bin, per level, or per tree serializes the GPU and host — the GPU repeatedly idles waiting for the host to decide the next split, and the host idles waiting for the read-back. Even the existing "host-light" grower reads back `2^depth` part-stats *per level* plus `leaf_of` and a recomputed `der1` *per tree*; multiply that across hundreds of trees × depth-6 and the stalls dominate.

**Why it happens:**
BestSplit is a tiny reduction whose result (which feature/border to split on) is needed by the host to drive the next level. The naive structure is "launch histogram kernel → read back → host picks split → launch next." Each arrow is a sync.

**How to avoid:**
- Keep the split *decision* on-device too: compute BestSplit in a kernel and keep the chosen split index in a device buffer; only read back the final tree structure once per tree (or batch several trees).
- Minimize crossings to O(1) metadata, never bulk per-object data (the established D-05 rule: only `2^depth` part-stats + `leaf_of` cross, never the histogram/partition/doc-routing).
- Recompute `der1` device-side and keep it device-resident for the next iteration instead of reading it back (see Pitfall 3).
- Where a read-back is unavoidable, overlap it: don't block the host on data the next launch doesn't need yet.

**Warning signs:**
- `read_one` / `read` calls inside the per-level or per-feature loop.
- Timeline shows sawtooth GPU utilization (burst, idle, burst) synchronized with host activity.
- Throughput scales with tree count far worse than O(trees).

**Phase to address:** Phase 1–2 (device-resident BestSplit + read-back minimization), validated by the benchmark harness phase.

---

### Pitfall 3: Re-uploading training data every tree / iteration (failure to stay device-resident)

**What goes wrong:**
The quantized feature matrix (`cindex`, feature-major bins), weights, target, and visiting-order `indices` are constant across all boosting iterations. Re-uploading them per tree wastes PCIe bandwidth and adds a host-side allocation/copy each iteration. Only the running `approx`/`der1` change between trees.

**Why it happens:**
A function-per-tree decomposition that takes `&[u32] cindex` by host slice and uploads inside the call (convenient, stateless) re-uploads every call. The current `grow_boosting_pass_into` already threads one `client` and keeps `der1` in a host `Vec` that it re-derives device-side — but the bulk `cindex`/`weight` should be uploaded **once** before the iteration loop and reused as device handles.

**How to avoid:**
- Upload `cindex`, `weight`, `target`, `indices` to device handles **once**, before the boosting loop; pass handles (not host slices) into the per-tree grower.
- Keep `approx`/`der1` device-resident across iterations; update them with a kernel, not a host round-trip.
- Thread ONE `ComputeClient` through the entire run — a CubeCL `Handle` is bound to its originating client; a handle allocated by one client and read by another is undefined (memory: "never read a 0-len handle"; `gpu_runtime/mod.rs:1917`).

**Warning signs:**
- `client.create(...)` for the feature matrix inside the per-tree loop.
- Host RSS churn / allocator pressure proportional to iteration count.
- PCIe-bound profile on a dataset that fits in VRAM.

**Phase to address:** Phase 1 (device-resident data layout), enforced by Phase 2 depth>1 work.

---

### Pitfall 4: Non-deterministic float reduction order breaks the ≤1e-5 oracle

**What goes wrong:**
Floating-point addition is non-associative. Parallel histogram accumulation and parallel sum-reductions add per-object gradients to the same bin in a runtime-dependent order, so the histogram (and hence split scores, leaf values, predictions) differs run-to-run in the low bits. Against an *absolute* ≤1e-5 oracle this is usually fine for a single tree — but errors compound across hundreds of boosting iterations (a tiny split-score difference flips a split, which diverges the whole subtree), so a structurally-correct kernel can still fail the end-to-end parity gate.

**Why it happens:**
`atomicAdd` into a shared histogram bin is the easiest correct parallelization, but its commit order is non-deterministic. Likewise tree/warp reductions sum in hardware-dependent order. CatBoost's own docs note GPU ranking results are non-deterministic for this reason.

**How to avoid:**
- Prefer **deterministic reduction strategies**: fixed-order tree/segmented reductions or per-partition private histograms merged in a fixed index order, rather than global `atomicAdd` racing into shared bins.
- Reduce in the **same associativity order** the CPU oracle uses where the gate is bit-tight; or accumulate in higher precision (f64) to push rounding below 1e-5.
- Gate per-stage (borders → histogram → split score → leaf value → prediction) so a reduction-order divergence is caught at the histogram stage, not only at final predictions (the v1.0 highest-leverage practice).
- Accept the established **ε=1e-4 GPU tolerance vs the Rust CPU path** (D-04) where bit-exactness is infeasible — but justify it per stage; do not silently widen the bar to make a flaky kernel pass (escalate-don't-weaken).

**Warning signs:**
- Same input → different model across two runs on the same GPU.
- Parity passes at 1 tree, drifts past tolerance by iteration N.
- A split flips between runs; downstream leaf values diverge by >>1e-5.

**Phase to address:** Phase 2 (device histogram + scoring) with a determinism sub-criterion; the determinism strategy must be chosen *before* the histogram kernel is written, not retrofitted.

---

### Pitfall 5: f64 atomic-add unavailable / falls back to host-sum on some GPUs

**What goes wrong:**
Code assumes hardware f64 `atomicAdd` (or an f64 atomic reduce) exists. On gfx1100 / RDNA3 (the in-env ROCm GPU) there is **no f64 atomic-add**; CubeCL's f64 reduce falls back to a `HostSumFallback` (memory: `phase76-gpu-tolerance-signoff-outcome`). A kernel written against CUDA's f64 atomics compiles but either rejects on HIP or silently routes the reduction through a slow host path — destroying both performance and the "device-resident" guarantee, and only on the ROCm backend.

**Why it happens:**
CUDA exposes f64 atomics broadly; developers test the cuda backend mentally and assume parity. RDNA3 consumer GPUs lack f64 atomics; the fallback is invisible to `cargo check` on cpu/wgpu and only manifests at runtime on the real GPU.

**How to avoid:**
- Do **not** depend on f64 atomics. Accumulate histograms in f32 with deterministic merge, or use per-partition private f64 histograms merged by a fixed-order reduction (no atomics), or a two-pass segmented reduction.
- If higher precision is needed, fold weight downstream (the established 7.2 UNWEIGHTED-der + 7.3 histogram-folds-weight contract) so the hot reduction stays in a supported type.
- Always run the **full ROCm suite in-env** after any `#[cube]` change — cpu/wgpu cargo-check cannot reveal the gfx1100 atomic/JIT behavior.

**Warning signs:**
- A reduce that is fast on the (untestable) cuda backend but collapses to host speed on rocm.
- HIP JIT errors mentioning unsupported atomic on f64, or a sudden host-side hot spot in a "device" reduce.

**Phase to address:** Phase 2 (histogram/reduce kernel design) — choose an atomic-free, type-portable reduction up front.

---

### Pitfall 6: CubeCL HIP rejects `-inf` literals in `#[cube]` kernels

**What goes wrong:**
`F::new(f32::NEG_INFINITY)` (or any `-inf` literal) inside a `#[cube]` kernel emits `double(-inf)` in generated HIP, which the gfx1100 JIT rejects with `undeclared identifier 'inf'`. The split-scorer's "no candidate yet" sentinel is the classic site. It is **invisible to cpu/wgpu cargo check** and fails *only* on the ROCm GPU at runtime (memory: `cubecl-hip-no-inf-literal`; bit Phase 7.5 WR-01, 16/75 → 75/75 after fix).

**Why it happens:**
`-inf` is the natural "lowest possible score" initializer for an argmax. It works on CPU/wgpu and on CUDA; HIP's codegen path can't parse the literal.

**How to avoid:**
- Use a **finite sentinel** (`f32::MIN`) for "lowest score" inside kernels; host code may keep `f64::NEG_INFINITY`.
- Add a lint/review checklist item: no `INFINITY`/`NEG_INFINITY`/`inf` literals inside `#[cube]` functions.
- Run the ROCm suite in-env after every `#[cube]` change (the only place this surfaces).

**Warning signs:**
- A kernel passes cpu/wgpu but the rocm run drops from N/N to a partial pass with a JIT "undeclared identifier 'inf'" error.

**Phase to address:** Every phase that touches `#[cube]` scoring kernels (Phase 2 split-scorer, Phase 3 pairwise/multiclass). Add to the kernel-authoring checklist now.

---

### Pitfall 7: The `cb-backend` → `cb-train` dependency landmine (Cargo feature unification)

**What goes wrong:**
Adding a `cb-train` dependency to `cb-backend` (e.g. to reuse a CPU reference function) triggers Cargo feature unification across the dependency graph, which breaks the ROCm runtime build/selection. The "obvious DRY refactor" silently corrupts backend selection.

**Why it happens:**
The device grower needs CPU reference logic (leaf averaging, score formulas). The tempting fix is to depend on the crate that already has them. But `cb-backend` must stay feature-isolated so the `rocm`/`cuda`/`wgpu`/`cpu` features don't unify with `cb-train`'s.

**How to avoid:**
- **Never** add a `cb-train` dependency to `cb-backend`. **Transcribe** the needed CPU reference inline into `cb-backend` (the established rule, memory `phase75-grow-loop-outcome` / root-cause note line 64).
- Factor shared math into a leaf crate with no backend features (e.g. `cb-compute`) that both depend on, rather than a `cb-backend → cb-train` edge.

**Warning signs:**
- ROCm runtime selection regresses / wrong backend chosen after a refactor.
- `cargo tree -f "{p} {f}"` shows `cb-train` features unified into the backend build.

**Phase to address:** Phase 1 (architecture of the grow-tree seam) — bake the inline-transcription rule into the seam design.

---

### Pitfall 8: cuda-vs-rocm behavioral divergence (correct in-env, broken on the benchmark target)

**What goes wrong:**
Correctness is developed and validated **only** on AMD/ROCm (no NVIDIA in-env); the head-to-head speed benchmark runs on **CUDA** (Kaggle). A kernel can pass all ROCm oracles yet behave differently on CUDA: different warp/wavefront width (RDNA3 wave32 vs NVIDIA warp32 — coincidentally equal, but block/occupancy tuning differs), different atomic availability (CUDA *has* f64 atomics, so a path that fell back on ROCm runs natively — different numerics), different JIT acceptance (CUDA accepts `-inf`, masking Pitfall 6). The benchmark run is the *first* time CUDA executes the code.

**Why it happens:**
The portability promise ("write once, run cuda/rocm/wgpu") is real for *compilation* but not for *numerics, performance, or JIT edge-cases*. The asymmetry (validate on ROCm, benchmark on CUDA) means the CUDA path is effectively untested for correctness before it's timed.

**How to avoid:**
- On the Kaggle CUDA run, **first re-run the oracle/parity tests on CUDA** (not just the timing) before trusting the speed number — a fast-but-wrong CUDA result is worthless.
- Keep kernels free of backend-specific assumptions (atomic availability, literal codegen, warp size); parameterize occupancy via CubeCL cube/plane dims rather than hard-coded NVIDIA numbers.
- Treat ε=1e-4 GPU-vs-CPU tolerance as defined against the **Rust CPU path**, and verify it holds on *both* GPU backends, not just ROCm.

**Warning signs:**
- Parity gates only ever ran on ROCm; the CUDA path has no correctness evidence.
- Occupancy/block sizes hard-coded to multiples of 32 with NVIDIA assumptions.

**Phase to address:** Benchmark/validation phase (final) — must include a CUDA-side correctness re-run, not only timing.

---

### Pitfall 9: Tiny-kernel launch overhead (per-feature / per-bin launches)

**What goes wrong:**
Launching one kernel per feature, per border, or per leaf makes launch/dispatch overhead dominate actual compute. A 100-feature × 128-bin histogram done as 100 (or 12,800) micro-launches is latency-bound; the GPU spends more time scheduling than summing. (Note: the *current* code has **no** tiny-kernel problem because no training kernels launch at all — this becomes a risk precisely *when* the inner loop is moved on-device, so design for it from the start.)

**Why it happens:**
The host CPU loop structure (`for feature → for border`) is transcribed 1:1 into kernel launches instead of being collapsed into one fused kernel that parallelizes over (feature, bin, object) internally.

**How to avoid:**
- Fuse: one histogram kernel that parallelizes across all features/bins/objects in a single launch; one scoring kernel across all candidate splits; one BestSplit reduction.
- Batch per-level work, not per-feature work. Keep launches O(levels) or O(trees), not O(features × bins × levels).

**Warning signs:**
- Launch count scales with feature/bin count.
- Profiler shows high dispatch overhead, low kernel duration, low occupancy.

**Phase to address:** Phase 2 (fused histogram + scoring kernels).

---

### Pitfall 10: Unfair benchmark vs official CatBoost GPU (apples-to-oranges)

**What goes wrong:**
The headline "20× slower / now at parity" number is meaningless unless the comparison is controlled. Common distortions: (a) counting the **first iteration** which includes CatBoost's JIT/kernel-compilation and our CubeCL JIT warmup; (b) measuring **end-to-end** (data load, quantization, Python/PyO3 overhead, I/O) instead of **train-only**; (c) different `depth`, `border_count`/bin count (128 vs 32 changes CatBoost GPU time a lot), `iterations`, `learning_rate`, bootstrap, or loss; (d) a dataset too small (<10k rows / <10 features) where GPU never wins for *either* implementation; (e) different `thread_count`/device; (f) one warm, one cold.

**Why it happens:**
It's easy to call `.fit()` on both and diff wall-clock. CatBoost compiles CUDA kernels on first use; CubeCL JITs on first launch; both pollute the first measured iteration. Bin count and tree-size parity are non-obvious knobs that swing the result.

**How to avoid:**
- Discard warmup: run ≥1 untimed warmup fit, then time the median of several runs.
- Measure **train-only** (exclude Pool construction, quantization upload, prediction, I/O) on both sides; or measure each phase separately.
- Pin identical params: same `iterations`, `depth`, `border_count`/`max_bin`, `learning_rate`, loss, bootstrap, `random_seed`, and the **same quantized data**. Match tree sizes (CatBoost's own benchmark tunes hyperparams so all libs build similar-size trees).
- Use a dataset large enough that GPU is meaningfully exercised (millions of rows or thousands of features for a real speedup signal; report bin count, e.g. 128 and 32).
- Report the hardware and that it's a *single* GPU; don't compare our 1-GPU run to a CatBoost multi-GPU run.

**Warning signs:**
- Speed ratio changes wildly run-to-run (warmup not excluded).
- "Parity reached" on a 5k-row toy set (GPU advantage is noise at that scale).
- Params or bin counts differ between the two `.fit()` calls.

**Phase to address:** Benchmark harness phase (final) — define the protocol before quoting any number.

---

### Pitfall 11: Kaggle CUDA environment gotchas

**What goes wrong:**
The Kaggle notebook is the *only* CUDA environment, so environment quirks masquerade as performance or correctness bugs: the CubeCL `cuda` feature must build against the notebook's CUDA toolkit/driver version; first-run JIT/`nvcc` compilation inflates the first timing; the GPU may be shared/throttled; session time limits and ephemeral filesystem can truncate long benchmark sweeps; installing a from-source Rust+CubeCL+maturin toolchain inside Kaggle is itself fragile.

**Why it happens:**
Kaggle is a managed, time-boxed, shared environment not designed for from-source GPU Rust builds. CUDA driver/toolkit mismatches and cold JIT are the usual culprits.

**How to avoid:**
- Pin and verify the CUDA toolkit/driver the `cuda` feature compiles against; confirm the build actually selected the CUDA backend (not a silent cpu fallback) before timing.
- Warm up CUDA (one untimed fit) so `nvcc`/JIT cost is excluded; check the GPU isn't being shared/throttled (`nvidia-smi`).
- Keep the benchmark short enough to finish within the session limit; persist results out of the ephemeral FS immediately.
- Re-run the parity oracle on CUDA in the same notebook before trusting timings (ties to Pitfall 8).

**Warning signs:**
- First-iteration time 10×+ the steady-state per-iteration time.
- "cuda" build that actually ran on cpu (no GPU utilization in `nvidia-smi`).
- Benchmark sweep killed by session timeout.

**Phase to address:** Benchmark harness phase (final).

---

### Pitfall 12: Occupancy / thread divergence in the tree-search kernels

**What goes wrong:**
Gradient-boosting kernels have data-dependent control flow (a thread's partition/leaf assignment, variable bin counts per feature, masked candidates). Naive branching causes warp/wavefront divergence (threads in a wave taking different paths serialize), and poor cube/plane dimensions leave the GPU under-occupied — both leave large speedups on the table even after the loop is device-resident.

**Why it happens:**
The host code is branchy; transcribed literally, branches become per-thread divergence. Block/grid sizing copied from an unrelated example doesn't match this workload's register/shared-memory footprint.

**How to avoid:**
- Restructure hot kernels to be branch-light (predication, sorted/segmented layouts so threads in a wave share a path).
- Tune cube/plane dims for occupancy per backend via CubeCL abstractions, not hard-coded constants; validate occupancy on the real GPU.
- Keep per-feature bin counts uniform where possible (padding) so the histogram kernel doesn't diverge on ragged feature widths.

**Warning signs:**
- Profiler reports low achieved occupancy or high warp divergence on the histogram/scoring kernels.
- Speedup plateaus well below CatBoost despite a fully device-resident loop.

**Phase to address:** Phase 2–3 (after correctness; an optimization sub-phase), measured by the benchmark harness.

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Derivatives-only GPU MVP, inner loop on host | Ships a "GPU backend" fast; reuses CPU grower | The >20× gap; GPU-vs-CPU regression; whole inner loop must be rebuilt | Only as an explicitly-labeled MVP with a *performance* success criterion deferred — never as "GPU done" |
| Per-tree der read-back to host | Trivial to update running approx in a host `Vec` | Sync stall per tree; blocks staying device-resident | Acceptable as the depth-1 bridge; must be killed when approx/der go device-resident |
| Global `atomicAdd` histogram | Easiest correct parallel histogram | Non-deterministic, fails f64 on gfx1100, fails tight oracle, slow | Only on f32 with a tolerance gate; never when bit-exact parity is required |
| Hard-coded NVIDIA block/warp sizes | Quick tuning win on CUDA | Breaks/under-performs on ROCm/wgpu; cuda-vs-rocm divergence | Never — parameterize via CubeCL cube/plane dims |
| `cb-backend` depends on `cb-train` for reference math | DRY; no transcription | Feature unification breaks ROCm runtime (landmine) | Never — transcribe inline or use a featureless leaf crate |
| `-inf` literal as score sentinel in `#[cube]` | Natural argmax init | HIP JIT reject on gfx1100, only caught on real GPU | Never inside `#[cube]`; use `f32::MIN` |
| Widening the GPU tolerance to pass a flaky kernel | Green gate | Hides a real numerics/determinism bug; erodes the parity bar | Never silently; only with a per-stage-justified, documented ε (escalate-don't-weaken) |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| CubeCL `ComputeClient` | Allocate a handle on one client, read on another | Thread ONE client through the whole boosting pass; a `Handle` is bound to its originating client |
| CubeCL device buffers | Read a 0-length handle (empty dataset/iteration) | Short-circuit empty inputs to an empty model before any launch |
| CubeCL `#[cube]` codegen on HIP | `-inf`/`INFINITY` literal in a kernel | Finite `f32::MIN` sentinel in-kernel; host keeps `f64::NEG_INFINITY` |
| ROCm f64 reduce (gfx1100) | Assume hardware f64 atomic-add | Atomic-free deterministic reduction; or f32 accumulation; weight folded downstream |
| Cargo features across crates | `cb-backend → cb-train` edge | Inline-transcribe CPU refs; shared math in a featureless leaf crate |
| Official CatBoost GPU (benchmark) | Time `.fit()` cold, end-to-end, mismatched params | Warmup + train-only + identical params/bins/data, single GPU |
| Kaggle CUDA | Trust first-run timing; assume backend selected | Verify `cuda` backend active via `nvidia-smi`; warm JIT; re-run oracle on CUDA |
| PyO3 facade → GPU backend | `fit()` hard-wired to CPU path | Generic `GpuBackend` over `SelectedRuntime` routing the inner loop on-device (the 08-08 generic-backend precedent) |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Inner loop on host (the v1.0 gap) | GPU idle, CPU pegged, GPU ≥ CPU time | Device-resident grow loop wired into `cb-train` | Always — even at small N the GPU never helps |
| Per-level/per-tree blocking read-back | Sawtooth GPU util; time scales worse than O(trees) | Device-side BestSplit; O(1) metadata crossings only | Deep trees × many iterations |
| Re-uploading feature matrix per tree | PCIe-bound; host alloc churn | Upload cindex/weight/target once; reuse handles | Many iterations on a VRAM-resident dataset |
| Tiny per-feature/per-bin launches | Launch overhead dominates; low occupancy | Fuse into one histogram/scoring/BestSplit kernel | Wide feature sets / high bin counts |
| Warp divergence in tree search | Low occupancy, high divergence in profiler | Branch-light, segmented/sorted layouts; tuned dims | Ragged bins / data-dependent partitions |
| Benchmarking a tiny dataset | GPU "no faster" / noisy ratio | Use millions of rows or thousands of features | <10k rows or <10 features — GPU advantage is noise |
| f64 atomic reduce on gfx1100 | "Device" reduce runs at host speed | Atomic-free / f32 reduction | Any f64 atomic path on RDNA3 consumer GPUs |

## Security Mistakes

Not a primary concern for an in-process training library (no network surface, no untrusted input beyond the user's own data). The closest analogues:

| Mistake | Risk | Prevention |
|---------|------|------------|
| Unbounded device allocation from user-supplied shapes | OOM / VRAM exhaustion (DoS-like crash) | Validate `n × n_features × n_bins` against device memory before allocating; typed error, not panic |
| `unwrap()`/`expect()`/panic in the device driver | Crash the host process on a malformed launch | Workspace lint forbids it (D-13); surface `CbError` for every read-back/shape mismatch |
| Silent zero-buffer on a failed read-back | Corrupt model passed off as valid | Failed read-back → `CbError::Degenerate`, never pad/zero (WR-05 precedent) |

## UX Pitfalls

"Users" here = Rust/Python ML practitioners selecting a backend.

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Selecting a GPU backend silently runs slower than CPU | User loses trust in the GPU build | Inner loop genuinely on-device; benchmark gate proves GPU ≥ CPU on real datasets |
| GPU result not bit-reproducible run-to-run | User can't reproduce models; CI flakes | Deterministic reductions; document the GPU ε tolerance explicitly |
| Backend chosen at compile time, surprising at runtime | User installs the wrong wheel, gets cpu fallback | Clear per-backend wheels + a startup check that reports the active backend/device |
| GPU "supported" but only depth-1 / RMSE works | User's depth-6 multiclass job silently routes to CPU or errors | Typed "unsupported on GPU" errors per feature; document the GPU coverage matrix |

## "Looks Done But Isn't" Checklist

- [ ] **GPU training:** Often missing the *inner loop* on-device — verify GPU utilization is high and GPU train time < CPU train time on a large dataset, not just that "a kernel ran."
- [ ] **Device residency:** Often missing — verify the feature matrix is uploaded **once** (no `client.create` inside the per-tree loop) and `approx`/`der1` never round-trip to host between iterations.
- [ ] **BestSplit:** Often still host-side — verify the split decision is computed in a kernel, with only O(1) metadata crossing back per level.
- [ ] **Determinism:** Often unverified — run the same input twice on the same GPU; the models must match within the documented ε; parity must hold at iteration N, not just iteration 1.
- [ ] **ROCm in-env run:** Often skipped after a `#[cube]` change — verify the **full ROCm suite** ran in-env (cpu/wgpu cargo-check cannot catch `-inf` JIT reject or f64-atomic fallback).
- [ ] **CUDA correctness:** Often only timed, never validated — verify the oracle/parity tests *passed on CUDA* (Kaggle), not only that it was fast.
- [ ] **Benchmark fairness:** Often unfair — verify warmup excluded, train-only measured, identical params/bin counts/data/seed, single GPU, dataset large enough.
- [ ] **Feature graph:** Often regressed — verify `cargo tree` shows no `cb-backend → cb-train` edge and ROCm backend still selected.
- [ ] **GPU coverage matrix:** Often overstated — verify which losses/depths/CTR/pairwise/ordered/multiclass actually run on-device vs error/fallback.

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Inner loop left on host | HIGH | Add the `Runtime` grow-tree seam; wire `grow_boosting_pass`; this is the milestone's core work, not a patch |
| Per-tree/level read-back stalls | MEDIUM | Move BestSplit + approx/der updates on-device; reduce crossings to O(1) metadata |
| Re-uploading data per tree | LOW | Hoist the upload above the loop; pass handles instead of host slices |
| Non-deterministic reduction fails oracle | MEDIUM | Replace atomic histogram with fixed-order/segmented reduction; or f64 accumulation; re-gate per stage |
| f64 atomic fallback on gfx1100 | MEDIUM | Re-architect reduce to be atomic-free / f32; fold weight downstream |
| `-inf` HIP reject | LOW | Replace in-kernel `-inf` with `f32::MIN`; re-run ROCm suite |
| `cb-backend → cb-train` unification | LOW–MEDIUM | Remove the edge; transcribe inline or extract a featureless leaf crate; rebuild ROCm |
| Unfair benchmark number | LOW | Re-run with warmup + train-only + matched params; re-quote |
| Kaggle env / cold-JIT distortion | LOW | Warm up; verify backend active; persist results before session timeout |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| 1. Inner loop on host | Phase 1 (Runtime grow-tree seam + wire `grow_boosting_pass`) | GPU train time < CPU on large data; GPU utilization high |
| 2. Per-level/tree read-back stalls | Phase 1–2 (device BestSplit, O(1) crossings) | No `read_one` inside per-level/feature loop; flat GPU timeline |
| 3. Re-uploading data per tree | Phase 1 (device-resident layout) | No `client.create` of feature matrix inside the loop |
| 4. Non-deterministic reduction vs ≤1e-5 | Phase 2 (histogram/scoring) | Same input → same model within ε; parity holds at iteration N |
| 5. f64 atomic-add unavailable (gfx1100) | Phase 2 (reduce design) | Atomic-free reduce; full ROCm suite green in-env |
| 6. HIP `-inf` literal reject | Every `#[cube]` phase (2, 3) | ROCm suite N/N after each kernel change; no `inf` literals in `#[cube]` |
| 7. `cb-backend → cb-train` landmine | Phase 1 (seam architecture) | `cargo tree` shows no edge; ROCm backend selected |
| 8. cuda-vs-rocm divergence | Benchmark phase (final) | Oracle re-run *passes on CUDA*, not just timed |
| 9. Tiny-kernel launch overhead | Phase 2 (fused kernels) | Launch count O(levels/trees), not O(features×bins) |
| 10. Unfair benchmark | Benchmark phase (final) | Documented protocol: warmup, train-only, matched params/bins/data |
| 11. Kaggle CUDA gotchas | Benchmark phase (final) | Backend verified active; warm JIT; results persisted |
| 12. Occupancy / divergence | Phase 2–3 (optimization sub-phase) | Profiler occupancy/divergence within target; speedup approaches CatBoost |

## Sources

- `.planning/notes/gpu-training-host-light-root-cause.md` — internal root-cause of the >20× gap (host-light inner loop; per-tree read-back; landmines). HIGH.
- `.planning/RETROSPECTIVE.md` + `.planning/PROJECT.md` — v1.0 lesson "parity must specify correctness AND performance"; CubeCL portability + landmine patterns. HIGH.
- Project memory: `phase75-grow-loop-outcome` (cb-train↔cb-backend feature-unification landmine; host-light D-05), `phase76-gpu-tolerance-signoff-outcome` (gfx1100 f64 reduce = HostSumFallback, ε=1e-4 sign-off), `cubecl-hip-no-inf-literal` (HIP `-inf` JIT reject → `f32::MIN`), `phase8-python-bindings-outcome` (generic `GpuBackend` over `SelectedRuntime`). HIGH.
- `crates/cb-backend/src/gpu_runtime/mod.rs:1850-2043` — `grow_boosting_pass[_into]` (one-client threading, O(1) crossings, empty-handle guards, MVP depth-1 scope). HIGH.
- [XGBoost GPU docs — non-deterministic GPU ranking due to FP non-associativity](https://xgboost.readthedocs.io/en/release_1.4.0/gpu/index.html). MEDIUM.
- [Quantized Training of Gradient Boosting Decision Trees (arXiv 2207.09682) — atomicAdd histogram non-determinism and cost](https://arxiv.org/pdf/2207.09682). MEDIUM.
- [GPU-acceleration for Large-scale Tree Boosting (arXiv 1706.08359) — histogram/reduction strategies](https://arxiv.org/pdf/1706.08359). MEDIUM.
- [Impacts of floating-point non-associativity on reproducibility (arXiv 2408.05148)](https://arxiv.org/pdf/2408.05148). MEDIUM.
- [CatBoost GPU vs CPU training-speed benchmark (official) — fair-comparison methodology, bin count, dataset-size thresholds](https://github.com/catboost/benchmarks/blob/master/gpu_vs_cpu_training_speed/README.md). HIGH.

---
*Pitfalls research for: device-resident GPU gradient-boosting training + CatBoost-GPU speed-parity benchmarking*
*Researched: 2026-06-28*
