# Phase 13: Pairwise, Ranking, Multiclass, Ordered & Langevin Device Coverage - Pattern Map

**Mapped:** 2026-07-04
**Files analyzed:** 10 (4 new kernel modules + tests, 3 extended host/seam files, 1 seam struct, fixtures, notebook)
**Analogs found:** 9 / 10 (every new file maps to an existing Phase 7.x/10/11/12 device analog + a frozen CPU oracle; only the on-device batched Cholesky lacks a device-side structural precedent and reuses the *serial-kernel* skeleton instead)

> **Standing landmine (applies to EVERY file below):** kernels live in `cb-backend`; NEVER add a
> `cb-train` dep (feature-unification breaks rocm runtime). Transcribe CPU refs INLINE as `const` +
> `#[cube]` helpers (the `mvs_device.rs` / `bootstrap_device.rs` precedent). No `-inf` literal in a
> `#[cube]` body (use finite `f32::MIN`). Bind each `Handle` to its allocating client. Der/weight
> SUMS use the fixed-point `Atomic<u64>` k=30 reduction; the Cholesky *solve* is f64 non-atomic
> per-matrix arithmetic (D-07 explicitly allows f64 there — it is NOT an atomic reduction).

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/cb-backend/src/kernels/langevin.rs` (NEW) | kernel | transform / event-driven (seeded RNG over resident der) | `crates/cb-backend/src/kernels/mvs_device.rs` | exact |
| `crates/cb-backend/src/kernels/langevin_test.rs` (NEW) | test | self-oracle | `crates/cb-backend/src/kernels/mvs_device_test.rs` / `apply_leaf_delta.rs` | exact |
| `crates/cb-backend/src/kernels/multi_newton.rs` (NEW) | kernel | transform (K-dim block leaf solve) | `crates/cb-backend/src/kernels/apply_leaf_delta.rs` + `cb-compute/src/leaf.rs::solve_symmetric_newton` | exact (extend Newton to K-dim) |
| `crates/cb-backend/src/kernels/multi_newton_test.rs` (NEW) | test | self-oracle vs CPU | `apply_leaf_delta.rs` (device-vs-serial-CPU harness) | exact |
| `crates/cb-backend/src/kernels/query_helper.rs` (NEW) | kernel | reduction + sort (segmented group infra) | `kernels/exact_quantile.rs::segmented_radix_sort` + `reduce.rs` + `cb-compute/src/ranking_der.rs` | role-match |
| `crates/cb-backend/src/kernels/query_helper_test.rs` (NEW) | test | self-oracle vs CPU | `mvs_device_test.rs` / `segmented_sort_test.rs` | exact |
| `crates/cb-backend/src/kernels/cholesky_solve.rs` (NEW) | kernel | transform (batched SPD solve) | `cb-compute/src/leaf.rs::cholesky_solve` (CPU oracle) + `mvs_device.rs` (serial `#[cube]` skeleton) | role-match (no device SPD precedent) |
| `crates/cb-backend/src/kernels/cholesky_solve_test.rs` (NEW) | test | self-oracle vs CPU | `apply_leaf_delta.rs` + `score_split.rs` pairwise oracle | role-match |
| `crates/cb-backend/src/gpu_runtime/session.rs` (MODIFY) | store / service | event-driven residency state | existing `GpuTrainSession` `Option<{Nonsym,ExactLeaf,Bootstrap,Mvs,Region}State>` fields | exact |
| `crates/cb-compute/src/runtime.rs` `DeviceGrownTree` (MODIFY) | model / seam struct | request-response (plain host struct across seam) | existing `DeviceGrownTree` `step_nodes`/`region_path` carrier extensions | exact |
| `crates/cb-backend/src/gpu_runtime/pairwise.rs` + `kernels/score_split.rs` (MODIFY) | service / kernel | transform (wire device Cholesky, replace 7.5 host solve) | current 7.5 host-side `calculate_pairwise_score` call site (`score_split.rs:864-992`) | role-match |
| `crates/cb-backend/src/kernels.rs` (MODIFY) | config | module registration | mvs registration block (`kernels.rs:2869-2877`) | exact |
| Frozen fixtures (multiclass / YetiRank / PFound-F / ordered / langevin) (NEW) | test | pinned-seed frozen oracle | Phase 12 D-07 mvs/ctr frozen fixtures | exact |
| Kaggle CUDA notebook cells (MODIFY) | test | manual gate | Phase 10/12 harness | exact |

## Pattern Assignments

### `crates/cb-backend/src/kernels/langevin.rs` (kernel, seeded-Gaussian transform) — GPUT-20

**Analog (device skeleton):** `crates/cb-backend/src/kernels/mvs_device.rs` — the closest device
precedent: a serial `#[cube]` kernel over the resident derivative buffer with inline PCG RNG
transcription, a device-resident handle out (no readback), a host-readback oracle wrapper, and a
wgpu reject. Langevin is *simpler* (no threshold bisection; per-element Marsaglia-polar draw + add).

**CPU oracle to transcribe INLINE:** `crates/cb-core/src/normal.rs::std_normal` (Marsaglia-polar,
VARIABLE even draw count per sample — draw order is load-bearing, Pitfall 4):
```rust
// crates/cb-core/src/normal.rs:50-64 — transcribe into a #[cube] fn, NOT a cb-core call.
pub fn std_normal(rng: &mut TFastRng64) -> f64 {
    loop {
        let x = rng.gen_rand_real1() * 2.0 - 1.0;
        let y = rng.gen_rand_real1() * 2.0 - 1.0;
        let r = x * x + y * y;
        if !(r > 1.0 || r <= 0.0) { return x * (-2.0 * r.ln() / r).sqrt(); }
    }
}
```

**Inline RNG primitive pattern to copy** (from `mvs_device.rs:46-92`) — the constants + `#[cube]`
mixers are already transcribed there; copy them verbatim:
```rust
const LCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;      // cb_core::rng::LCG_MULTIPLIER
const REAL1_INV: f64 = 1.0 / 9_007_199_254_740_991.0;        // 1/(2^53-1), gen_rand_real1 divisor
#[cube] fn rotate_right_u32(v: u32, r: u32) -> u32 { /* r==0 guard vs v<<32 UB */ }
#[cube] fn pcg_mix(x: u64) -> u32 { /* XSH-RR: xorshifted=((x>>18)^x)>>27; rot=x>>59 */ }
```

**Per-block reseed + advance(10) + conditional-draw pattern** (copy from `mvs_device.rs:183-222`)
— the `TFastRng64::new` two-stream seed derivation (`fix_seq`, `r1c=(seq1<<1)|1`) and the
comptime-unrolled `advance(10)` are already correct there; Langevin reuses the SAME seed derivation,
only the body is `der[i] += coefficient * std_normal(seed_i)`.

**Launch + resident-handle-out + wgpu-reject pattern** (copy from `mvs_device.rs:264-345`):
```rust
#[cfg(feature = "wgpu")]
fn wgpu_reject() -> CbError { CbError::OutOfRange("device ... requires f64+u64 ...".to_owned()) }

pub(crate) fn launch_langevin_resident(client, der_h: &Handle, rand_seed, coefficient, n)
    -> CbResult<Handle> {
    if n == 0 { return Ok(client.empty(0)); }
    #[cfg(feature = "wgpu")] { return Err(wgpu_reject()); }
    // serial unit-0 launch: CubeCount::Static(1,1,1), CubeDim{1,1,1}; ArrayArg::from_raw_parts(...)
    // RETURN the resident handle WITHOUT read_one (D-09 residency).
}
// + draw_langevin_host(...) readback wrapper for the self-oracle (mvs_device.rs:326).
```

**CAVEAT (A4, VERIFIED):** upstream `pairwise_oracle.h` `CB_ENSURE`s Langevin is NOT supported for
the pairwise oracle — Langevin layers on the POINTWISE/groupwise reduced-der path only. A
PairLogit+Langevin fixture must fall back to CPU (`Ok(None)`).

---

### `crates/cb-backend/src/kernels/multi_newton.rs` (kernel, K-dim block leaf solve) — GPUT-12

**Analog (device apply):** `crates/cb-backend/src/kernels/apply_leaf_delta.rs` — today the scalar
`approx[i] += lr * leaf_values[leaf_of[i]]` gather+FMA (Phase 11). D-03 extends the leaf block to
`leaf_count × approx_dim`, so the apply becomes `approx[d*n + i] += lr * leaf_block[leaf_of[i]*K + d]`
routed through the existing multi-output CPU apply (`runtime.rs:1145`, `approx[d*n+i]` layout).

**CPU oracle to transcribe (the block solve):** `crates/cb-compute/src/leaf.rs::solve_symmetric_newton`
(:201-260) — the packed lower-triangular Hessian reconstruction + `maxTrace` f32 regularization +
`M = -(H - adjustedL2·I)` + `cholesky_solve`. This is the K-dim generalization the device kernel must
match; the packing order `[(0,0),(0,1),…,(0,k-1),(1,1),…]` is load-bearing:
```rust
// crates/cb-compute/src/leaf.rs:214-260 (VERIFIED CPU oracle). Reconstruct H from packed order,
// max_trace = max(scaled_l2 as f32, (-diag) as f32...), adjusted_l2 = max(scaled_l2, max_trace*f32::EPS),
// M = -(H - adjusted_l2·I), solve M·x = -sum_der via cholesky_solve, res = -x.
```
**Coupled vs diagonal (Pitfall 3, VERIFIED):** full-block coupled solve ONLY for `MultiClass`
(softmax, off-diagonal hessian `−w·p_k·p_row`); DIAGONAL for `MultiClassOneVsAll`,
`MultiCrossEntropy`/multilabel, `MultiRMSE`, `RMSEWithUncertainty`. `multilogit.cu` emits der2 one
row at a time — mirror that. The der functions ALREADY exist in `cb-compute` (do not re-derive).

**Dense Cholesky building block:** `cb-compute/src/leaf.rs::cholesky_solve` (:282-324) — non-positive
pivot → `None` → zeros (no NaN/panic). The device kernel transcribes THIS routine inline for the
`k <= ~10` block (shares the exact numerics with the pairwise solver — see cholesky_solve.rs below).

**Test harness:** copy `apply_leaf_delta.rs` device-vs-serial-CPU structure (`run_apply`/`cpu_apply`,
`F64_TOL`, non-wgpu gate `#![cfg(not(feature = "wgpu"))]`).

---

### `crates/cb-backend/src/kernels/query_helper.rs` (kernel, segmented group reduction + sort) — GPUT-22

**Analog (segmented sort primitive — DO NOT hand-roll):** `crates/cb-backend/src/kernels/exact_quantile.rs::segmented_radix_sort` (:162-166) — already keys+values, per-segment, self-oracled bit-exact:
```rust
pub(crate) fn segmented_radix_sort(head_flags: &[u32], keys: &[u32], values: &[u32])
    -> CbResult<(Vec<u32>, Vec<u32>)>   // head_flags[0] must be 1; empty → (vec,vec)
```
Use for `CreateSortKeys` (`key = (qid<<32) | random_low_32`) + in-query sampling — queries stay
contiguous yet shuffle within a query (§6.6a `query_helper.cu`).

**CPU der oracle:** `crates/cb-compute/src/ranking_der.rs::calc_ders_for_queries` (:139) over
`GroupSpan { begin, end, weight, competitors }` (:78-95) and `group_reduce_weighted` (:104-118,
folds through `cb_core::sum_f64` in object order — D-08, no raw float fold). Only O(1) group
descriptors + the resident der buffer cross the seam.

**Group reduction pattern** (`query_helper.cu` §6.6a): one warp (32 lanes) per query,
`queriesPerBlock = BLOCK/32`; lanes stride the query's docs accumulating `sumTarget`/`sumWeight`
(or max); `WarpReduce` → lane 0 writes mean/max. Kernels: `ComputeGroupIds`, `ComputeGroupMeans`,
`ComputeGroupMax`, `RemoveGroupMeans` (`dst[d] -= queryMeans[qids[d]]`), `FillTakenDocsMask` +
`SampledQuerySize(sampleRate,qSize)` (≥2 floor). Reductions over der/weight use the fixed-point
`Atomic<u64>` path (`reduce.rs`) for determinism; RNG for `CreateSortKeys` uses the `mvs_device.rs`
inline PCG. **Stochastic YetiRank/PFound-F** use the pinned-seed frozen-fixture discipline (D-08).

**Open Q3 (research):** QueryCrossEntropy's per-query bisection/Newton shift search is the most
complex arm — the `mvs_device.rs` bisection loop (`MVS_BISECTION_ITERS = 100`, monotone bracket) is
the structural precedent; if it over-runs, land QueryRMSE/QuerySoftMax first and gate
QueryCrossEntropy independently (`Ok(None)`).

---

### `crates/cb-backend/src/kernels/cholesky_solve.rs` (kernel, batched f64 SPD solve) — GPUT-21 [HIGHEST RISK]

**No device-side SPD-solve precedent exists** — Phase 7.5 deliberately deferred it to the host
(`score_split.rs:853` "RESEARCH Open Q3"; the current call runs `cb_compute::calculate_pairwise_score`
on the host, `score_split.rs:989-992`). This file MOVES it on-device. Use the `mvs_device.rs` serial
`#[cube]` skeleton (per-matrix warp/serial work, f64 accumulation, resident handle out) as the
structural analog.

**CPU oracle to MATCH (the ε=1e-4 parity target — the Rust CPU path, NOT upstream, Pitfall 2):**
`crates/cb-train/src/pairwise_leaves.rs::calculate_pairwise_leaf_values` (:113-180). Transcribe
these EXACT constants + drop-last-row + zero-average — NOT upstream's `RegularizeImpl` bump-heuristics:
```rust
// crates/cb-train/src/pairwise_leaves.rs:123-179 (VERIFIED, frozen ≤1e-5 oracle):
let cell_prior = 1.0 / system_size as f64;           // system_size = leaf_count
let non_diag_reg = -pairwise_bucket_weight_prior_reg * cell_prior;
let diag_reg = pairwise_bucket_weight_prior_reg * (1.0 - cell_prior) + l2_diag_reg;
// system_size==2 closed form: x0 = derSums[0]/(A00 + diag_reg), then MakeZeroAverage.
// general: build (n-1)×(n-1) SPD (BOTH triangles), solve, res.push(0.0), make_zero_average.
let mut res = cb_compute::pairwise_cholesky_solve(&matrix, &rhs).unwrap_or(vec![0.0; m]);
res.push(0.0);
make_zero_average(&mut res); // subtract mean via cb_core::sum_f64
```

**Dense solve to transcribe INLINE:** `cb-compute/src/leaf.rs::cholesky_solve` (:282-324) — in-place
lower-triangular `a = L·Lᵀ`, forward L·y=b, back Lᵀ·x=y; non-positive pivot → `None` → zeros
(matches upstream `1e-7` pivot floor semantics; the fallback prevents NaN). Batched over leaves,
one logical warp/matrix (upstream `RunCholeskySolver<128,256,REMOVE_LAST>` — NO cuSOLVER dep).

**Two systems, ONE parameterized kernel (Open Q1, recommended):** leaf-VALUES use
`calculate_pairwise_leaf_values` (system = `leaf_count`, REMOVE_LAST + zero-average); split-SCORING
uses `cb_compute::calculate_pairwise_score` (system = `2*PartCount`) — self-oracle each separately.
The pairwise histograms feeding both REUSE the Phase 7.4 4-channel `pairwise_hist` kernels
(`gpu_runtime/pairwise.rs`) unchanged.

**FALLBACK CHECKPOINT (A1, Pitfall 1):** if the on-device `#[cube]` solve over-runs, keep the 7.5
host bounded solve behind `Ok(Some(...))` (correctness-safe, ε unaffected) and log the residency gap
for Phase 14 — do NOT block the family.

---

### `crates/cb-backend/src/gpu_runtime/session.rs` (store, ordered resident approx state) — GPUT-13

**Analog (exact):** the existing `GpuTrainSession` per-family `Option<State>` fields
(`session.rs:361-380`): `nonsym`, `exact_leaf`, `bootstrap`, `mvs`, `region` — each `Some` iff the
fit committed to that device path, gated in `begin(...) -> CbResult<Option<Self>>` (:470-547) which
returns `Ok(None)` for any uncovered config. Add an `ordered: Option<OrderedState>` field the SAME
way:
```rust
// session.rs:366-380 pattern — each family is an Option<XxxState>, coverage-gated, Ok(None)-safe:
bootstrap: Option<BootstrapState>,
mvs: Option<MvsState>,
region: Option<RegionState>,
// D-05 ADD: ordered: Option<OrderedState>,  // per-permutation approx trajectory, RESIDENT across iters
```

**CPU ref to reproduce (frozen fixture, D-06):** `cb_train::boosting::ordered_approx_delta_simple`
(`boosting.rs:687`) — the anti-leakage body/tail approximant (body rows keep delta 0). Fold machinery
`create_folds`/`Fold`, `permutation_count` (default 4 → `max(1,pc−1)` learning + 1 averaging fold).
Freeze the exact permutation + per-permutation approx trajectory; reproduce bit-for-bit at ε=1e-4.

**Residency discipline (Pitfall 5, D-05):** the trajectory stays device-resident ACROSS iterations —
only O(1) descriptors cross the seam per level (mirror the resident `approx_h`/`der1_h` handles at
`session.rs:335-338`, updated on-device via `apply_leaf_delta`, NO n-length readback). Validate no
`n`-length readback as the wave's success check. Open Q2: pin `permutation_count` in the fixture,
cover single learning-fold first.

---

### `crates/cb-compute/src/runtime.rs` `DeviceGrownTree` (seam struct, block-leaf extension) — D-03

**Analog (exact):** the existing carrier extensions on the SAME struct — `step_nodes` /
`node_id_to_leaf_id` (non-sym, Phase 12 Plan 01) and `region_path` (Region, Plan 04) were added the
same way: PLAIN HOST fields, EMPTY for the paths that don't use them, byte-unchanged for the
oblivious path (`runtime.rs:926-970`). Add `approx_dim` + reinterpret `leaf_values` as a
`leaf_count × approx_dim` block:
```rust
// crates/cb-compute/src/runtime.rs:927 (VERIFIED). Landmine: PLAIN HOST types only — no cubecl /
// cb-backend type may appear here (T-10-04 feature-unification: the seam must never pull a backend
// dep into cb-train).
pub struct DeviceGrownTree {
    pub splits: Vec<(u32, u32)>,
    pub leaf_values: Vec<f64>,          // TODAY 2^depth scalar; D-03: leaf_count*approx_dim row-major/leaf
    // D-03 ADD: pub approx_dim: usize, // 1 for the scalar path (byte-unchanged at approx_dim==1)
    pub leaf_of: Vec<u32>,
    pub step_nodes: Vec<(u16, u16)>,               // non-sym carrier (empty for oblivious) — SAME idiom
    pub node_id_to_leaf_id: Vec<u32>,
    pub region_path: Vec<(u32, u32, bool, bool)>,  // Region carrier (empty otherwise) — SAME idiom
}
```
Block leaves route through the EXISTING multi-output CPU apply (`runtime.rs:1145`, `approx[d*n+i]`).
`GPUT-14` no-regression: at `approx_dim == 1` the scalar `leaf_values` bytes are IDENTICAL.

---

### `crates/cb-backend/src/gpu_runtime/pairwise.rs` + `kernels/score_split.rs` (wire device Cholesky) — GPUT-21

**Analog:** the current 7.5 host solve call site to REPLACE — `score_split.rs:864-992`
(`use cb_compute::{calculate_pairwise_score, compute_pair_weight_statistics}`; the host
`calculate_pairwise_score(...).expect(...)` at :989). The pairwise-stat assembly + der-sum scatter
already run device-side (`gpu_runtime/pairwise.rs`); Wave B swaps the host solve for
`launch_cholesky_solve(...)`. The self-oracle at `score_split.rs:1076-1093` (device score vs host
`calculate_pairwise_score`) is the test template.

---

### `crates/cb-backend/src/kernels.rs` (module registration) — all new kernels

**Analog (exact):** the mvs registration block (`kernels.rs:2869-2877`):
```rust
pub(crate) mod mvs_device;
#[cfg(test)] mod mvs_device_test;
```
Register `cholesky_solve` / `query_helper` / `multi_newton` / `langevin` the same way (`pub(crate) mod`
+ `#[cfg(test)] mod *_test`). New kernels are `pub(crate)` (consumed by `gpu_runtime`).

## Shared Patterns

### Inline RNG transcription (never reach `cb_core` from a `#[cube]` body)
**Source:** `crates/cb-backend/src/kernels/mvs_device.rs:46-222` (and `bootstrap_device.rs`)
**Apply to:** `langevin.rs`, `query_helper.rs` (CreateSortKeys), stochastic ranking (YetiRank/PFound-F).
The `LCG_MULTIPLIER` / `REAL1_INV` consts, `rotate_right_u32` / `pcg_mix` `#[cube]` mixers, the
`TFastRng64::new` two-stream seed derivation (`fix_seq`, `r1c=(seq1<<1)|1`), and the comptime-unrolled
`advance(10)` are already bit-exact — copy verbatim. Draw order/count is load-bearing (Pitfall 4);
`std_normal` (`cb-core/src/normal.rs:50`) consumes a VARIABLE even count per sample.

### `Ok(None)` coverage gate + per-family `Option<State>` (all-or-nothing, D-10-01)
**Source:** `crates/cb-backend/src/gpu_runtime/session.rs::begin` (:470-547) + the `Option<*State>`
fields (:361-380).
**Apply to:** every family (pairwise, ranking, multiclass, ordered, langevin). Each flips from
`Ok(None)`→CPU to `Ok(Some(tree))`→device INDEPENDENTLY behind its own gate; any uncovered config
returns `Ok(None)` — NEVER a fabricated/wrong device result (T-10-05, Pitfall 6).

### Serial-CPU self-oracle test harness (device vs frozen CPU, ε bar over equal-length buffers)
**Source:** `crates/cb-backend/src/kernels/apply_leaf_delta.rs` (device `run_*` + `cpu_*` reference,
`F64_TOL`, empty-`n` no-launch guard) and `mvs_device_test.rs`.
**Apply to:** all four new `*_test.rs`. Non-wgpu gate `#![cfg(not(feature = "wgpu"))]`; assertions
skip off rocm/cuda to avoid a CPU-vs-CPU false-pass (WR-01). Do NOT `read_one` a 0-len handle (HIP
faults) — assert the launcher constructs the no-op (`apply_leaf_delta.rs:93-104`).

### f64/u64 wgpu reject (typed error, not a JIT crash)
**Source:** `mvs_device.rs:264-273` (`wgpu_reject()` → `CbError::OutOfRange`).
**Apply to:** `langevin.rs`, `cholesky_solve.rs`, `multi_newton.rs`, `query_helper.rs` — all use
f64/u64 device channels WGSL lacks. Guard `#[cfg(feature = "wgpu")]` at every entry point.

### Deterministic reduction vs f64 solve (the D-07 distinction)
**Source:** `crates/cb-backend/src/kernels/reduce.rs` (`REDUCE_FIXEDPOINT_SCALE_F64`, k=30
`Atomic<u64>` + fixed-order tree-reduce fallback; SPIKE-REDUCTION §5b).
**Apply to:** der/weight SUMS in `query_helper.rs`, histogram accumulation (use fixed-point — gfx1100
has no f64 atomic-add). The Cholesky SOLVE (`cholesky_solve.rs`, `multi_newton.rs`) is per-matrix f64
NON-atomic arithmetic — D-07 explicitly allows f64 there (not an atomic reduction).

### Segmented radix sort (do NOT hand-roll a second sort)
**Source:** `crates/cb-backend/src/kernels/exact_quantile.rs::segmented_radix_sort` (:162).
**Apply to:** `query_helper.rs` in-query sampling / CreateSortKeys.

### Frozen pinned-seed fixture (RNG-driven paths, D-06)
**Source:** Phase 12 D-07 mvs/ctr frozen fixtures (`mvs_device_test.rs`).
**Apply to:** ordered permutation+trajectory, YetiRank/PFound-F draws, Langevin draw sequence. Host
advances the continuous stream and hands the device the O(1) base state (bootstrap_device precedent);
reproduce the frozen CPU draws bit-for-bit (deterministic, not distributional).

### `#[cube]` landmines (recurring)
No `-inf` literal (finite `f32::MIN` sentinel — HIP JIT rejects `double(-inf)`); bind each `Handle`
to its allocating client; no `unwrap`/`expect`/`panic`/indexing in production (workspace lints);
run the rocm suite in-env after any `#[cube]` change (in-env green ≠ Kaggle CUDA sign-off, Pitfall 6).

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `crates/cb-backend/src/kernels/cholesky_solve.rs` (device SPD *solve*) | kernel | transform | No existing DEVICE-side dense SPD solve — Phase 7.5 punted it to the host (`score_split.rs:853` "Open Q3"). Only the CPU oracle (`cb-compute/src/leaf.rs::cholesky_solve`) and the serial-`#[cube]` skeleton (`mvs_device.rs`) exist; the batched-warp-per-matrix f64 decomposition + fwd/back subst is genuinely new device code (transcribed from upstream `RunCholeskySolver<128,256,REMOVE_LAST>`, no cuSOLVER). Planner: front-load as its own wave with the A1 host-fallback checkpoint. |

## Metadata

**Analog search scope:** `crates/cb-backend/src/kernels/`, `crates/cb-backend/src/gpu_runtime/`,
`crates/cb-compute/src/{runtime,leaf,ranking_der}.rs`, `crates/cb-train/src/{pairwise_leaves,boosting}.rs`,
`crates/cb-core/src/normal.rs`.
**Files scanned:** ~14 (read) + directory listings of cb-backend/cb-compute/cb-train/cb-core src.
**Pattern extraction date:** 2026-07-04
