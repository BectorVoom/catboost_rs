# Phase 10: Coarse Runtime Grow-Tree Seam + GpuTrainSession Residency + Wire Depth-1 + Kaggle CUDA Oracle & Speed Harness - Pattern Map

**Mapped:** 2026-06-29
**Files analyzed:** 12 (4 modified Rust + 1 modified entrypoint + 1 new test module + 3 spike kernels + 4 new bench/spike docs)
**Analogs found:** 11 / 12 (only `bench/README.md` is pure doc with no code analog)

> **Hard landmine (applies to every cb-backend file below):** NEVER add a `cb-train`
> dependency to `cb-backend`. Cargo feature unification then activates `cb-backend/cpu`
> alongside `rocm`/`cuda`/`wgpu`, `SelectedRuntime` mis-resolves, and `#[cube]` kernels fail
> to build. Transcribe any needed cb-train reference logic INLINE (precedent:
> `crates/cb-backend/src/kernels/grow_loop.rs:24-39`). The `Runtime` seam stays host-typed in
> `cb-compute` so cb-train never pulls a backend.

> **CubeCL landmine (every `#[cube]` kernel below):** no `-inf` float literal — HIP/gfx1100
> JIT rejects `double(-inf)` (invisible to cpu/wgpu `cargo check`, fails only on rocm). Use
> the `f32::MIN` sentinel. Read `/home/user/Documents/workspace/cubecl_manual/manual/cubecl/INDEX.md`
> (lowercase path — A6) BEFORE any kernel work; load `cubecl_error_guideline.md` on ANY build error.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/cb-compute/src/runtime.rs` (MODIFY) | trait/contract | request-response (host-typed) | `compute_gradients_grouped` default-impl @ same file `:944-954` | exact (same file, same pattern) |
| `crates/cb-backend/src/gpu_backend.rs` (MODIFY) | service/provider (backend impl) | event-driven (stateful session) | existing `impl Runtime for GpuBackend` @ same file `:148-146`/der seam `:73-146` | exact |
| `crates/cb-backend/src/gpu_runtime/mod.rs` (MODIFY) | service + `#[cube]` kernel | streaming (device-resident) | `grow_oblivious_tree_into` `:1641`, `grow_boosting_pass_into` `:1920` | exact |
| `crates/cb-train/src/boosting.rs` (MODIFY) | controller (boosting loop) | event-driven / iterative | grower dispatch `:3203`, `der1.clone()` `:2996`, `train<R: Runtime>` `:1870` | exact |
| `crates/catboost-rs/src/builder.rs` (MODIFY, maybe) | config/entrypoint | request-response | `fit()` `:333-371` | exact |
| `crates/cb-backend/src/kernels/session_residency.rs` (NEW test) | test | event-driven (in-env GPU smoke) | `crates/cb-backend/src/kernels/grow_loop.rs` | exact (sibling test module) |
| `bench/cuda_oracle.py` (NEW) | utility/harness | batch / transform | `benchmark.py` (+ `benchmark_fast.py`/`benchmark_small.py`) | role-match |
| `bench/fixtures/*` (NEW) | fixture/data | file-I/O | `make_fixture` @ `grow_loop.rs:101-113` (deterministic gen) | role-match |
| `crates/cb-backend/src/kernels/spike_reduction.rs` (NEW, test-gated) | kernel + test | streaming (reduce/atomic) | `grow_loop.rs` + existing `#[cube]` kernels in `gpu_runtime/mod.rs` | role-match |
| `bench/README.md` (NEW) | doc | — | (none — pure doc) | no analog |
| `bench/RESULTS.md` (NEW) | doc/sign-off log | — | committed-fixture/RESULTS pattern (D-10-05) | no code analog |
| `.planning/phases/10-.../SPIKE-REDUCTION.md` (NEW) | doc | — | (none — analysis doc) | no code analog |

## Shared Patterns

### Error type (every Rust file)
**Source:** `crates/cb-core/src/error.rs:10-97`
**Apply to:** all seam signatures, the session, the device-grow branch.
- `pub type CbResult<T> = Result<T, CbError>;` — every fallible seam method returns `CbResult<...>`.
- Reuse existing variants — do NOT invent new ones unless required:
  - `CbError::OutOfRange(String)` — unsupported/uncovered config, overflow guards.
  - `CbError::Degenerate(String)` — a covered session that fails to grow mid-run (D-10-02:
    "hard `CbError`, NOT a silent CPU graft"), and any device read-back failure (never a
    silent zero — WR-05 precedent).
  - `CbError::LengthMismatch { column, expected, actual }` — shape guards before any launch.
- `CbError` derives `Clone, PartialEq, Eq, thiserror::Error`. Any new variant MUST keep those
  derives (no `#[from]` on a non-Clone external error — stringify instead, see `:88-96`).

### No `unwrap`/`panic`/indexing in production (every Rust source file)
**Source:** workspace lints + the existing grow loop `gpu_runtime/mod.rs:1605` ("No
`unwrap`/`expect`/`panic`/indexing in this production driver"); read with `.get()` not `[]`
(see `grow_boosting_pass_into:2001-2007`). Tests may use `unwrap`/`assert`.

### Source/test separation (MANDATORY — CLAUDE.md)
**Source:** `crates/cb-backend/src/kernels/grow_loop.rs` is a dedicated test file pulled in via
`#[cfg(test)] mod grow_loop;` (`kernels.rs:2774`). **No `#[cfg(test)] mod tests` may be added
to any production source file.** All Phase-10 tests (session residency, depth-1 oracle, spike)
go in dedicated `crates/cb-backend/src/kernels/*.rs` files, gated `#[cfg(test)]` at the `mod`
include.

### CubeCL residency rule (every device handle)
**Source:** `gpu_runtime/mod.rs:1633-1639`, `:1700-1708`. ONE `ComputeClient` per session;
a `Handle` is bound to its allocating client — never read a handle through another client,
never read a 0-len handle. `end_device_training` drops the session (client + handles)
deterministically.

---

## Pattern Assignments

### `crates/cb-compute/src/runtime.rs` (trait/contract, request-response) — GPUT-01

**Analog:** `compute_gradients_grouped` default-impl in the SAME file, `runtime.rs:944-954`.

**Default-impl seam pattern to COPY** (`:944-954`):
```rust
fn compute_gradients_grouped(
    &self,
    loss: &Loss,
    approx: &[f64],
    target: &[f64],
    weights: &[f64],
    groups: &[crate::ranking_der::GroupSpan],
    random_seed: u64,
) -> CbResult<Vec<Derivatives>> {
    crate::ranking_der::calc_ders_for_queries(loss, approx, target, weights, groups, random_seed)
}
```
This is the exact shape: a method with a **default body** on the `Runtime` trait so EVERY
existing impl (`CpuBackend`, test runtimes) compiles unchanged. For Phase 10 the three new
methods default to `Ok(false)` / `Ok(None)` / `Ok(())` → transparent CPU fallback (preserves
D-04 for free). Only `GpuBackend` overrides.

**Host-typed-only discipline** (the trait must stay CubeCL-free): the existing trait imports
only `Loss`, `Derivatives`, `EScoreFunction` (`:832`), `CbResult`/`CbError` — NO cubecl types.
The new `DeviceGrownTree` struct + the three methods must use only `Vec`/slices/`usize`/`f64`
host types (RESEARCH Code Examples `:269-341`). `DeviceGrownTree.leaf_of` is length `0 OR n`
(empty in the hot path — D-05; populated only for the oracle).

**Existing `EScoreFunction` enum** (`:831-862`) is the host-typed param the coverage gate
reads — pass it into `begin_device_training` so the gate can reject non-L2/Cosine.

**Existing `Derivatives` struct** (`:882-890`) + `compute_gradients` (`:918-924`) are the der
seam the residency loop chains into — do NOT duplicate.

---

### `crates/cb-backend/src/gpu_backend.rs` (service/provider, event-driven session) — GPUT-02

**Analog:** the existing `impl Runtime for GpuBackend` + Phase 7.2 der dispatch in the SAME
file, `gpu_backend.rs:73-146`.

**Zero-sized backend → add interior-mutable session** (`:42-47`):
```rust
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuBackend;
```
Becomes a holder of `RefCell<Option<GpuTrainSession>>` (Roadmap-named, RESEARCH Pattern 2).
**Caveat (Pitfall 4):** this drops `Copy`/zero-sized-ness. `builder.rs:358` binds
`let backend = GpuBackend;` once and `train` takes `&self`, so a single owner is fine — but
GREP `GpuBackend` for by-value copies first (A3). Keep `Default` (init `RefCell` to `None`).

**Per-loss der-seam dispatch to REUSE verbatim for residency** (`:79-146`):
```rust
match *loss {
    Loss::Rmse => {
        let der1 = launch_der_binary(approx_d, target_d, DerBinaryKernel::RmseGradient)?;
        let der2 = const_der_host(-1.0_f64, n)?;
        Ok((der1, der2))
    }
    Loss::Logloss | Loss::CrossEntropy => {
        let der1 = launch_der_binary(approx_d, target_d, DerBinaryKernel::LoglossGradient)?;
        let der2 = launch_der_unary(approx_d, DerUnaryKernel::LoglossHessian)?;
        Ok((der1, der2))
    }
    // ...
    ref other => Err(CbError::OutOfRange(format!(
        "loss {other:?} is not yet supported on the GPU backend ..."
    ))),
}
```
The residency loop (GPUT-03) needs the `_into` (handle-resident) variants of these — RMSE
`der1 = target - approx` for the depth-1 RMSE path, Logloss `der1` for the Logloss path. Note
the depth-1 Logloss leaf-method decision (Pitfall 3 / A1): pin the fixture to FIRST-ORDER
(`calc_average`) leaves; Newton der2 leaves are Phase 11.

**Coverage gate (D-10-02)** lives HERE (where the session is constructed), surfaced through
the seam as `Ok(false)`/`Ok(true)`:
```
Some(session) iff depth == 1 && matches!(loss, Rmse|Logloss) && boosting_type == Plain
                 && fold_count == 1 && score_function ∈ {L2, Cosine} (no CTR/pairwise/multiclass)
```
A covered session that later cannot grow → `CbError::Degenerate` (NOT `Ok(None)`).

**Error-discipline excerpt to mirror** (`:139-144`): typed `CbError::OutOfRange` naming the
unsupported case as a documented parity gap, never a silent fallback or panic.

---

### `crates/cb-backend/src/gpu_runtime/mod.rs` (service + new `#[cube]` kernel, streaming) — GPUT-02 / GPUT-03

**Analogs:** `grow_oblivious_tree_into:1641` and `grow_boosting_pass_into:1920` (both in this file).

**`GpuTrainSession` construction — copy the one-client/upload-once block** from
`grow_oblivious_tree_into:1700-1708`:
```rust
let der1_h = upload_channel_floats(client, der1);
let weight_h = upload_channel_floats(client, weight);
let cindex_h = client.create(cubecl::bytes::Bytes::from_elems(cindex.to_vec()));
let indices_h = client.create(cubecl::bytes::Bytes::from_elems(indices.to_vec()));
let mut leaf_of_h = client.create(cubecl::bytes::Bytes::from_elems(vec![0u32; n]));
```
And the client creation from `grow_oblivious_tree:1626-1627`:
```rust
let device = <SelectedRuntime as cubecl::Runtime>::Device::default();
let client = <SelectedRuntime as cubecl::Runtime>::client(&device);
```
**GPUT-02 change:** the session owns `client` + `cindex_h`/`indices_h`/`weight_h` (uploaded
ONCE per `fit()`), plus resident `approx_h`/`der1_h`. Add a `grow_oblivious_tree_resident(...)`
variant that TAKES the resident handles instead of re-uploading every call (today
`grow_oblivious_tree_into` re-uploads at `:1704-1708` — RESEARCH Pattern 3).

**Shape/overflow/empty guards to copy** (`:1655-1698`): empty short-circuit, `checked_shl` for
`2^depth`, `checked_mul` for `cindex_stride`, `LengthMismatch` on `cindex.len()`. The
`depth > 1` typed reject (`:1670-1678`) is the MVP boundary — keep it (depth>1 → `Ok(None)`
gate upstream, never reaches here for a covered session).

**Per-level loop to reuse** (`:1715-1734`): `launch_find_optimal_split_pointwise_into` →
`best.ok_or_else(Degenerate)` → `splits.push((feature_id, bin_id))`. Only the O(1) `BestSplit`
crosses per level (D-05).

**THE GPUT-03 REFACTOR — eliminate the host approx-update + der1 read-back.** The anti-pattern
to REMOVE is `grow_boosting_pass_into:1996-2036`:
```rust
// (3) HOST approx update — the D-05 violation to move ON-DEVICE:
for (obj, slot) in approx.iter_mut().enumerate() {
    if let Some(&leaf) = tree.leaf_of.get(obj) {
        if let Some(&v) = tree.leaf_values.get(leaf as usize) { *slot += v; }
    }
}
// (4) der1 recompute WITH read-back — the n-length crossing to ELIMINATE:
let der1_h = launch_der_binary_into(client, &approx, target, DerBinaryKernel::RmseGradient)?;
let bytes = client.read_one(der1_h).map_err(|e| CbError::Degenerate(...))?;
der1 = bytemuck::cast_slice::<u8, f64>(&bytes).to_vec();
```
**Replace with:** (a) keep `approx_h`/`der1_h` resident on the session; (b) NEW `#[cube]`
kernel `apply_leaf_delta(approx, leaf_of, leaf_values, lr)` doing
`approx[i] += lr * leaf_values[leaf_of[i]]` on device (RESEARCH `:343-361`); (c) chain
`der1_h = der(approx_h, target)` via the 7.2 `_into` seam WITHOUT read-back. `leaf_of` then
crosses ONLY for the oracle. **Warning sign of a regression:** any `client.read_one` /
`read_u32_handle` / `read_part_stats_f64` of an n-length buffer inside the per-tree loop.

**New `#[cube]` kernel pattern** (generic-float, bounds-guarded, NO `-inf`):
```rust
#[cube(launch)]
fn apply_leaf_delta_kernel<F: Float>(
    approx: &mut Array<F>, leaf_of: &Array<u32>, leaf_values: &Array<F>, lr: F,
) {
    let i = ABSOLUTE_POS;
    if i < approx.len() {            // bounds guard (V5 input validation)
        let leaf = leaf_of[i];
        approx[i] += lr * leaf_values[leaf];
    }
}
```

**`GrownTree` struct** (`:1534-1544`) is the device-side shape `{ splits: Vec<(u32,u32)>,
leaf_of: Vec<u32>, leaf_values: Vec<f64>, part_stats: Vec<f64> }` — map it into the host-typed
`DeviceGrownTree` at the seam boundary (drop `leaf_of` unless oracle).

---

### `crates/cb-train/src/boosting.rs` (controller, iterative) — GPUT-04 / D-10-01 wiring

**Analogs:** grower dispatch `:3203-3231`, `der1.clone()` `:2996-3008`, `train<R: Runtime>`
`:1870-1890`.

**`train` is already generic over `R: Runtime`** (`:1870`) — the seam flows through with NO
signature change. The device branch is inserted at the grower dispatch `:3203`:
```rust
let grown: GrownTree = match params.grow_policy {
    EGrowPolicy::Lossguide | EGrowPolicy::Depthwise => { leaf_wise_grower(...)? }
    EGrowPolicy::SymmetricTree | EGrowPolicy::Region => { /* existing CPU oblivious chain */ }
};
```
**Wiring (RESEARCH `:363-380`):** BEFORE the loop call `device_active =
runtime.begin_device_training(...)?`; INSIDE the loop, when `device_active`, replace the CPU
dispatch with:
```rust
let grown: GrownTree = if device_active {
    match runtime.grow_tree_on_device(&approx, target)? {
        Some(dev) => map_device_tree_to_cpu(&dev, feature_borders), // Pattern 4 bin→border
        None => return Err(CbError::Degenerate(
            "covered device session failed to grow a tree mid-run".into())), // D-10-02
    }
} else { /* the EXISTING CPU dispatch, byte-unchanged — D-04 */ };
// AFTER the loop:
if device_active { runtime.end_device_training()?; }
```
The CPU arm stays **byte-unchanged** (D-04). The per-fit all-or-nothing gate (D-10-01) means
`device_active` is decided ONCE.

**Pattern 4 — the one non-obvious join (`bin_id → border`):** device returns
`splits: (feature, bin_id)` with pass test `cindex[feature*n+obj] > bin_id` (verified
`grow_loop.rs:138`). cb-train `Split { feature: usize, border: f64 }` (`tree.rs:109`) pass test
is `value > border`. Map `border = feature_borders[feature][bin_id]` — the same
`feature_borders` `train_inner` already holds. Get this wrong → structure diverges (A4: verify
border index == bin boundary against `select_borders_greedy_logsum` ordering).

**`der1.clone()` read-back to eliminate** (`:2996-3008`): the host `weighted_der1` materialization
is the CPU path; for the device path the der1 stays resident on the session (GPUT-03) — do NOT
read it back per tree.

---

### `crates/catboost-rs/src/builder.rs` (config/entrypoint) — backend selection

**Analog:** `fit()` `:333-371` (same file).

**Compile-time backend select to preserve** (`:355-358`):
```rust
#[cfg(feature = "cpu")]
let backend = CpuBackend;
#[cfg(any(feature = "wgpu", feature = "cuda", feature = "rocm"))]
let backend = GpuBackend;
let trained = train(&backend, &feature_values, &feature_borders, pool.label(),
                    pool.weights(), &params, None)?;
```
Exactly one feature active; `train` accepts any zero-sized... now non-`Copy` (`RefCell`)
backend via `&self`. The matrix inputs (`feature_values` f32 SoA `:336-340`, `feature_borders`
f64 `:345-349`) already feed `train`; the seam consumes the SAME quantized `cindex` cb-train
builds. Likely NO change needed here beyond the `GpuBackend` non-`Copy` verification (A3).

---

### `crates/cb-backend/src/kernels/session_residency.rs` (NEW test) — GPUT-02 / GPUT-03 / GPUT-04 in-env smoke

**Analog:** `crates/cb-backend/src/kernels/grow_loop.rs` (whole file — the sibling device-grow
cross-oracle).

**Test-module include pattern** (`kernels.rs:2774`): `#[cfg(test)] mod grow_loop;` — add
`#[cfg(test)] mod session_residency;` the same way.

**Inline-transcription discipline (D-7.5-04, `grow_loop.rs:24-39`):** import `cb_compute`
READ-ONLY for the leaf-value oracle (`calc_average`, `scale_l2_reg`) + score oracle
(`l2_split_score`, `LeafStats`). TRANSCRIBE any cb-train structure logic VERBATIM — do NOT
import `cb-train` (the feature-unification landmine).

**Deterministic-fixture pattern to copy** (`grow_loop.rs:101-156`): `make_fixture` from
`test_fixtures` primitives (`ramp_centred`, `weight_mod5`, `cindex_feature_major`,
`indices_identity`); `cpu_stump_score` transcribing `cb_compute::l2_split_score` +
`cindex[feature*n+obj] > bin` pass test; `cpu_leaf_index` forward-bit (`idx |= 1usize << i`).

**Tolerance constants** (`grow_loop.rs:53-56`): `LEAF_BOUND = 1e-3` (wgpu) / `1e-9` (else).
Depth-1 level-0 whole-dataset histogram IS the exact CPU score → hold ≤1e-5 (Phase 7.6 ε
precedent; tighter than the ε=1e-4 deep-tree bar).

**GPUT-03 residency assertion:** instrument the crossing count — assert NO n-length read-back
in the per-tree loop (`residency_no_readback` test, RESEARCH Test Map).

**Run command:** `cargo test -p cb-backend --no-default-features --features rocm <name>`
(in-env GPU smoke; CUDA is the authoritative human-gated Kaggle run, not in-CI).

---

### `bench/cuda_oracle.py` (NEW harness, batch/transform) — BENCH-01 / BENCH-02

**Analogs:** `benchmark.py` (repo root), `benchmark_fast.py`, `benchmark_small.py`.

**Harness shape to extend** (`benchmark.py:1-55`): seeded numpy data gen
(`np.random.seed(42)`), import `catboost` + `catboost_rs`, warm fit, `time.time()` deltas,
device-vs-CPU summary. Reuse the import-guard (`:5-10`) and the speedup/slowdown reporter
(`:44-52`).

**BENCH-specific ordering (Pitfall 6 — MUST):** `nvidia-smi` (confirm CUDA active) → warm ONE
untimed fit (JIT excluded) → re-run depth-1 oracle ≤1e-5 **(BLOCKING gate)** → time train-only
with a read-back/predict to drain CubeCL's lazy queue BEFORE stopping the clock → report speed.
Correctness gates BEFORE any speed number (D-10-04).

**Two datasets (escalation-resolved D-10-09 / A2):** the ≤1e-5 correctness oracle runs on the
SMALL deterministic fixture (`benchmark.py`'s 1000×20 is fine for correctness); the depth-1
device≥CPU SPEED gate runs on a LARGE-n dataset (≥2×10⁵ rows × ≥50 features) — device cannot
beat CPU at small n (physics, not tuning). Add the large-n generator alongside the small one.

**Baselines (D-10-10):** in-env CPU + same-Kaggle-hardware CPU (apples-to-apples) + official
CatBoost `task_type='GPU'` where a comparable depth-1 config exists.

---

### `bench/fixtures/*` (NEW data) — D-10-05

**Analog:** `make_fixture` @ `grow_loop.rs:101-113` (deterministic generation). Commit SMALL
deterministic depth-1 RMSE + Logloss fixtures generated in-env (same fixture the in-env build
verifies) so the Kaggle run is reproducible + diffable. Format is Claude's discretion
(numpy `.npy`/`.csv` + a model.json reference); pin the Logloss fixture's CPU reference to
FIRST-ORDER leaves (Pitfall 3).

---

### `crates/cb-backend/src/kernels/spike_reduction.rs` (NEW, test-gated) — SC5 / D-10-06

**Analogs:** `grow_loop.rs` (test-module shape) + existing `#[cube]` kernels in
`gpu_runtime/mod.rs`.

Prototype ALL three reduction candidates as small on-device kernels: (a) fixed-point i64
atomics, (b) private-histogram merge, (c) two-pass segmented reduce. Measure determinism error
(vs CPU f64 `sum_f64`) AND wall-clock. **Pitfall 5:** float atomics are order-nondeterministic
(use i64 fixed-point for determinism); gfx1100 lacks f64 atomic-add → record per-backend
viability (HostSumFallback path per Phase 7.6). NO `-inf` literal. Output feeds
`SPIKE-REDUCTION.md` (err+ms table + recommendation for Phase 11). Authoritative numbers from
Kaggle CUDA; ROCm in-env is a smoke check (D-10-07).

---

## No Analog Found

| File | Role | Reason |
|------|------|--------|
| `bench/README.md` | doc | Pure documentation (Kaggle notebook steps); no code pattern to copy. |
| `bench/RESULTS.md` | doc/sign-off log | Human sign-off log structure is new (D-10-05); committed-RESULTS is a convention, not a code analog. |
| `.planning/phases/10-.../SPIKE-REDUCTION.md` | doc | Analysis artifact (err+ms table + recommendation); no code analog. |

## Metadata

**Analog search scope:** `crates/cb-compute/src/runtime.rs`, `crates/cb-backend/src/`
(`gpu_backend.rs`, `gpu_runtime/mod.rs`, `kernels/grow_loop.rs`), `crates/cb-train/src/boosting.rs`,
`crates/catboost-rs/src/builder.rs`, `crates/cb-core/src/error.rs`, repo-root `benchmark*.py`.
**Files scanned:** 8 read in full/targeted; line refs verified current 2026-06-29.
**Pattern extraction date:** 2026-06-29
</content>
</invoke>
