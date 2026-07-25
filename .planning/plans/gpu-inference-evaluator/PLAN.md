---
title: "GINF-01 — GPU inference evaluator — TDD Implementation Plan"
phase: 19-gpu-inference
slice: gpu-inference-evaluator
plan_version: 1
status: draft
updated_at: 2026-07-19T00:00:00Z
source_spec: .planning/plans/gpu-inference-evaluator/SPEC.md
source_research: .planning/plans/next-feature-research/
---

# GINF-01 — TDD Implementation Plan

Plan-only artifact. No production code is authored here. Every file / symbol /
command below is verified against on-disk source via CodeGraph + Read (evidence
inline). The CPU apply path (`crates/cb-model/src/apply.rs`) and its shipped
`predict_raw` are **reused as the oracle, never modified** (D-04 no-regression).

> **CubeCL rule (applies to every kernel task — TASK-03 especially):** BEFORE
> writing any `#[cube]` code, re-read the manual chapters cited in SPEC §3.3 —
> Generics, Conditionals (`Cubecl_conditionals.md`), Plane-Alignment
> (`plane_alignment.md`), Multi-threading, Launch-Overhead — starting from
> `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md`. On ANY
> CubeCL build error, STOP and consult
> `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/cubecl_error_guideline.md`
> FIRST; blind fixes are prohibited (project rule, CLAUDE.md/AGENTS.md).

## 0. Goal-backward derivation

Acceptance outcomes (SPEC §6) drive the task set:

| Acceptance | Observable success | Task |
|---|---|---|
| AT-S1 | guard: supported→Ok, CTR/non-sym/region/multi-dim→matching Err | TASK-01 |
| AT-S2 | flatten → HOST reconstruction == `predict_raw` (exact f64) | TASK-02 |
| AT-S3 | `#[cube]` kernel `out` == host reference (`cpu` backend, f64) | TASK-03 |
| AT-S4 | `launch_apply_oblivious_f64` == `predict_raw` ≤1e-9; empty/shape/read-back errs | TASK-04 |
| AT-S5a/b | facade `predict_raw_on_device` == `predict_raw` ≤1e-9; unsupported → `Err` | TASK-05 |
| AT-S6 | large-batch device==CPU parity guard + timings (`#[ignore]`) | TASK-06 |

Reused seams (verified, do NOT modify):

- CPU apply: `predict_raw` (`apply.rs:370`) → `predict_raw_cat` (`:386`) →
  `predict_raw_one` (`:318`) → `leaf_index_for` (`:208`); float split
  `passes_float_split` (`:136-140`, `f64::from(v) > border`); per-tree sum via
  `sum_f64` + bias once (`:353-354`). `[VERIFIED: CODEGRAPH apply.rs]`
- Leaf index: `cb_train::leaf_index(passes)` — `idx |= 1usize << i` forward bit
  order (`tree.rs:293-301`). The kernel mirrors THIS. `[VERIFIED: CODEGRAPH]`
- `Model` fields to guard/flatten: `oblivious_trees`, `non_symmetric_trees`,
  `region_trees`, `ctr_data`, `approx_dimension`, `bias`
  (`model.rs:271-313`); `ObliviousTree.{splits, leaf_values}` + `ModelSplit::Float(Split{feature,border})`
  (`model.rs:70-98`; `boosting.rs:770-790`). `[VERIFIED: CODEGRAPH]`
- Guard precedent: `is_onnx_exportable` deterministic check order
  (`onnx.rs:99-115`). `[VERIFIED: CODEGRAPH]`
- Kernel precedent: `apply_leaf_delta_kernel` per-object one-write gather
  (`kernels.rs:588-598`, `if ABSOLUTE_POS < approx.len() { … leaf_of[…] … }`).
  `[VERIFIED: CODEGRAPH]`
- Launch precedent: `launch_block_reduce_f64` (`gpu_runtime/mod.rs:220-260`):
  `client.create(Bytes::from_elems(...))`, `client.empty(...)`,
  `num_cubes = n.div_ceil(CUBE_DIM)`, `CubeCount::Static`, `CubeDim{x:CUBE_DIM,…}`,
  `kernel::launch::<f64, SelectedRuntime>(...)`, `client.read_one(handle)` →
  `CbError::Degenerate` on failure, `bytemuck::cast_slice`. `[VERIFIED: CODEGRAPH]`
- Tolerance: `SCORE_BOUND` = `1e-3` under `wgpu`, else `1e-9`
  (`score_split.rs:70-73`). The apply oracle mirrors this. `[VERIFIED: CODEGRAPH]`
- Overflow-guard precedent: `PackedCindex::device_arrays` casts to u32 →
  `CbError::OutOfRange` (`cindex.rs:93-108`). `[VERIFIED: CODEGRAPH]`
- Backend selection: `SelectedRuntime` cpu/wgpu/cuda/rocm arms
  (`lib.rs:59-82`); kernels + gpu_runtime mounted under ALL backends
  (`lib.rs:18-27`). The `cpu` arm is a REAL CubeCL `CpuRuntime` (f64) that runs
  the kernel in ordinary `cargo test`. `[VERIFIED: CODEGRAPH]`

Design decisions locked by the SPEC (do NOT re-open):

- **Placement (SPEC §3.1):** pure guard+flattener in `cb-model` (backend-free,
  MODEL-02 preserved); `#[cube]` kernel + Model-agnostic launch helper in
  `cb-backend`; marshalling + `predict_raw_on_device` in the `catboost-rs` facade.
- **Raw compare, NOT read_bin (SPEC §1, R4):** the kernel compares raw `f32`
  value vs `f64` border (cast to `F` host-side), never a quantized bin.
- **Predication, NOT per-split branch (SPEC §3.3):** `leaf += u32::cast_from(value
  > border) << i` in statement form.
- **Oracle under `cpu` (SPEC §9 R3):** numeric parity runs under the `cpu`
  backend (`CpuRuntime`, f64) in normal CI; ROCm confirms; wgpu/cuda
  compile-verified.

## 1. Execution order & waves

```
Wave A (parallel):   TASK-01 (cb-model guard+scaffold)  ∥  TASK-03 (cb-backend kernel)
Wave B (parallel):   TASK-02 (cb-model flattener)       ∥  TASK-04 (cb-backend launch helper)
Wave C:              TASK-05 (catboost-rs facade + parity oracle)
Wave D:              TASK-06 (catboost-rs #[ignore] throughput bench)
```

Dependency graph:

```
TASK-01 ─> TASK-02 ─┐
                    ├─> TASK-05 ─> TASK-06
TASK-03 ─> TASK-04 ─┘
```

Acyclic. Parallel pairs own DISJOINT crates/files:
- Wave A: TASK-01 edits `crates/cb-model/**` only; TASK-03 edits
  `crates/cb-backend/**` only. No write conflict.
- Wave B: TASK-02 edits `crates/cb-model/src/gpu_apply.rs` (+ its test); TASK-04
  edits `crates/cb-backend/src/gpu_runtime/mod.rs` (+ its test). Disjoint crates.
- TASK-02 depends on TASK-01 (same file `gpu_apply.rs`); TASK-04 depends on
  TASK-03 (needs the kernel symbol). Within a crate the two tasks are sequential.
- TASK-05 joins both branches (needs the flattener AND the launch helper).

## 2. Spec-ID → task coverage

| Spec | Behavior | Task(s) |
|---|---|---|
| GINF-01-S1 | device-apply guard | TASK-01 |
| GINF-01-S2 | oblivious-float flattener | TASK-02 |
| GINF-01-S3 | `#[cube]` apply kernel | TASK-03 |
| GINF-01-S4 | launch host helper | TASK-04 |
| GINF-01-S5 | facade `predict_raw_on_device` + parity | TASK-05 |
| GINF-01-S6 | throughput bench | TASK-06 |

Every S1..S6 is covered by exactly one primary task; each task references ≥1 spec.

---

## TASK-01 — Device-apply guard + module scaffold (GINF-01-S1)

- **Spec refs:** GINF-01-S1. Primary failure reason: the guard admits/rejects the
  wrong model kind.
- **Goal / completion:** `cb_model::gpu_apply::check_gpu_apply_supported(&Model)
  -> Result<(), GpuApplyUnsupported>` exists; the new module scaffold (that TASK-02
  extends) compiles and is re-exported; unit tests pass; `cargo clippy -p cb-model
  --lib --no-deps` clean.
- **Prerequisites:** none (Wave A, parallel with TASK-03).
- **Files:**
  - Create: `crates/cb-model/src/gpu_apply.rs` — module doc + `GpuApplyUnsupported`
    (thiserror enum) + `check_gpu_apply_supported`. Mount tests at file end:
    `#[cfg(test)] #[path = "gpu_apply_test.rs"] mod tests;`.
  - Create: `crates/cb-model/src/gpu_apply_test.rs` — S1 unit tests.
  - Modify: `crates/cb-model/src/lib.rs` — add `pub mod gpu_apply;` +
    `pub use gpu_apply::{check_gpu_apply_supported, GpuApplyUnsupported};`
    (scaffold; `flatten_oblivious_f64`/`FlatObliviousF64` added in TASK-02) near
    the existing `pub use export::{…}` (`lib.rs:37`).
- **CodeGraph/Read evidence:** `is_onnx_exportable` deterministic order + typed
  enum shape (`onnx.rs:57-115`); `Model` fields (`model.rs:271-313`);
  `ModelSplit::Ctr` detection (`onnx.rs:106-110` `matches!(split, ModelSplit::Ctr(_))`);
  MODEL-02 boundary — the module must NOT `use cb_backend`/`cubecl` (`apply.rs:1-6`).
- **Red:** in `gpu_apply_test.rs` (allow restriction lints at file top like the
  other cb-model tests):
  - `guard_accepts_float_oblivious_scalar` — a hand-built float-only oblivious
    scalar `Model` → `Ok(())`.
  - `guard_rejects_ctr` — a `Model` with `ctr_data.is_some()` (or a
    `ModelSplit::Ctr` split) → `Err(GpuApplyUnsupported::CategoricalFeatures)`.
  - `guard_rejects_non_symmetric` / `guard_rejects_region` — non-empty
    `non_symmetric_trees` / `region_trees` → the matching variant.
  - `guard_rejects_multidim` — `approx_dimension = 2` → `MultiDimensional`.
  Expected INITIAL failure: `gpu_apply` module / `check_gpu_apply_supported` does
  not exist ⇒ compile error (unresolved import), whole test file fails to build.
- **Green:** implement the guard in the deterministic order
  non-symmetric → region → CTR (`ctr_data.is_some()` OR any `ModelSplit::Ctr`) →
  multi-dim (`approx_dimension > 1`) → `Ok(())`. Use `.is_empty()`, `.iter()…any`,
  `matches!` — no `unwrap`/`expect`/`panic`/indexing.
- **Refactor:** none beyond doc clarity. Regression scope: `gpu_apply_test.rs`
  only; `apply.rs`/`onnx.rs` untouched.
- **Validation:**
  - `cargo test -p cb-model --lib gpu_apply`
  - `cargo clippy -p cb-model --lib --no-deps`
- **Completion evidence:** 5 guard tests green; clippy clean; `pub mod gpu_apply;`
  compiles; no `cubecl`/`cb_backend` import in the module.
- **Compat/rollback:** additive module; rollback = remove the two files + the
  `lib.rs` lines.
- **Parallelization:** parallel with TASK-03 (disjoint crate). Blocks TASK-02
  (same file).

---

## TASK-02 — Oblivious-float flattener (GINF-01-S2)

- **Spec refs:** GINF-01-S2. Primary failure reason: the flat layout mis-encodes
  tree structure so a host reconstruction diverges from `predict_raw`.
- **Goal / completion:** `cb_model::gpu_apply::flatten_oblivious_f64(&Model) ->
  CbResult<FlatObliviousF64>` exists; a pure-CPU round-trip test (reconstruct
  predictions from the flat arrays, compare to `predict_raw`) passes EXACTLY;
  clippy clean.
- **Prerequisites:** TASK-01 (same file `gpu_apply.rs`; needs the guard).
- **Files:**
  - Modify: `crates/cb-model/src/gpu_apply.rs` — add `FlatObliviousF64` struct +
    `flatten_oblivious_f64` (calls `check_gpu_apply_supported`, maps its
    guard-rejection error to `CbError::Unsupported` — NOT `OutOfRange`, which is
    reserved exclusively for the u32-index-overflow case below — concatenates
    each tree's float splits `(feature as u32, border)` + `leaf_values`; builds
    CSR `tree_split_offsets` / `tree_leaf_offsets` of length `n_trees + 1`; u32
    checked-casts → `CbError::OutOfRange`).
  - Modify: `crates/cb-model/src/gpu_apply_test.rs` — the round-trip + edge tests.
  - Modify: `crates/cb-model/src/lib.rs` — extend the re-export to
    `pub use gpu_apply::{check_gpu_apply_supported, flatten_oblivious_f64,
    FlatObliviousF64, GpuApplyUnsupported};`.
- **CodeGraph/Read evidence:** `ObliviousTree.{splits: Vec<Split>, leaf_values:
  Vec<f64>}` (`boosting.rs:770-790` / `model.rs:254`); `ModelSplit::Float`
  →`Split{feature,border}` (`model.rs:70-98`); `leaf_index` bit order
  (`tree.rs:293-301`); `predict_raw_one` accumulate+bias (`apply.rs:318-355`);
  `passes_float_split` compare (`apply.rs:136-140`); overflow-cast precedent
  `PackedCindex::device_arrays` (`cindex.rs:93-108`); `cb_core::CbError`
  (`OutOfRange`, `Degenerate`) already imported across the workspace.
- **Red:** in `gpu_apply_test.rs`:
  - `flatten_roundtrip_matches_cpu` — build a known 2-tree float-only oblivious
    scalar `Model` (depth-2 each, distinct borders, distinct leaf values, nonzero
    bias) + a small feature batch; call `flatten_oblivious_f64(&model)?`; a
    test-local `reconstruct(flat, features) -> Vec<f64>` (mirroring
    `leaf_index` + per-tree gather + bias) must EQUAL `cb_model::predict_raw(&model,
    &features)` element-wise (exact f64, since both are host f64).
  - `flatten_offsets_invariants` — assert `tree_split_offsets.len() ==
    tree_leaf_offsets.len() == n_trees + 1`, monotonic, and
    `split_features.len() == split_borders.len()`; additionally assert, for every
    tree `t`, `tree_leaf_offsets[t+1] - tree_leaf_offsets[t] ==
    oblivious_trees[t].leaf_values.len()` — a direct structural check tying the
    CSR offsets to the source model's actual leaf-value counts (not just to the
    flattener's own internally-consistent-but-possibly-wrong offsets, which is
    all the kernel's bounds guard can check).
  - `flatten_rejects_unsupported` — a CTR model → `Err` (the guard error as
    `CbError`).
  Expected INITIAL failure: `flatten_oblivious_f64` / `FlatObliviousF64` unresolved
  ⇒ test build fails.
- **Green:** implement the flattener; the round-trip matches exactly. Key: the
  reconstruction and the flattening BOTH use forward bit order (`leaf_index`
  semantics) and per-tree leaf offset `tree_leaf_offsets[t] + leaf`.
- **Refactor:** extract a small `flatten_tree` helper; keep offset building in one
  pass. No behavior change; regression scope: `gpu_apply_test.rs` + TASK-01 guard
  tests still green; `apply.rs` untouched (it is the oracle).
- **Validation:**
  - `cargo test -p cb-model --lib gpu_apply`
  - `cargo clippy -p cb-model --lib --no-deps`
- **Completion evidence:** round-trip == `predict_raw` exactly; invariants hold;
  reject path typed; clippy clean.
- **Compat/rollback:** additive; rollback = revert the flattener + re-export delta.
- **Parallelization:** sequential after TASK-01 (same file). Parallel with TASK-04
  (disjoint crate).

---

## TASK-03 — Oblivious-float apply `#[cube]` kernel (GINF-01-S3)

> **RE-READ FIRST (project rule):** `INDEX.md`, `Cubecl_generics.md`,
> `Cubecl_conditionals.md`, `plane_alignment.md`, `Cubecl_multi_threading.md`.
> On ANY CubeCL build error → `cubecl_error_guideline.md` BEFORE any fix.

- **Spec refs:** GINF-01-S3. Primary failure reason: the on-device leaf index or
  per-tree accumulation is wrong (divergence-induced or bit-order/gather bug).
- **Goal / completion:** `apply_oblivious_float_kernel<F: Float>` exists in
  `kernels.rs` and, under the `cpu` backend, produces `out` equal to a host
  reference on a tiny fixed model; the kernel oracle child-module test passes;
  `cargo clippy -p cb-backend --lib --no-deps` clean; the kernel compiles under
  every backend gate.
- **Prerequisites:** none (Wave A, parallel with TASK-01).
- **Files:**
  - Modify: `crates/cb-backend/src/kernels.rs` — add the `#[cube(launch)] pub fn
    apply_oblivious_float_kernel<F: Float>(…)` body next to
    `apply_leaf_delta_kernel` (`:588`). Add the child-module mount for the oracle
    test mirroring the existing `kernels::reduce` / `kernels::scatter` pattern
    (confirm the exact mount idiom in `kernels.rs` / `lib.rs:10-19` at edit time).
  - Create: `crates/cb-backend/src/kernels/apply_oblivious_test.rs` — the
    device-vs-host oracle (mounted `#[cfg(test)]` from `kernels.rs`).
- **CodeGraph/Read evidence:** per-object gather + bounds guard shape
  `apply_leaf_delta_kernel` (`kernels.rs:588-598` — `if ABSOLUTE_POS < approx.len()`,
  `leaf = leaf_of[ABSOLUTE_POS] as usize`, `approx[…] += lr[0]*leaf_values[leaf]`);
  a launch-form kernel + host-side read-back for the test mirrors the
  `read_all_bins_kernel`/`launch_block_reduce_f64` shape; predication pattern
  `mask = F::cast_from(bool)` (`plane_alignment.md:55-65`); statement-form `if`
  (`Cubecl_conditionals.md:34-41`); generic-float launch
  (`conditional_kernel::launch::<f32, CpuRuntime>` example, `Cubecl_conditionals.md:108-116`).
- **Kernel algorithm (implement exactly, generic-float, no `if`-expression):**
  ```text
  let obj = ABSOLUTE_POS;
  if obj < n_objects {
      let mut acc = F::new(0.0);                // F — bias is NOT seeded here
      let mut t = 0u32;
      while t < n_trees {
          let s0 = tree_split_offsets[t];       // u32
          let s1 = tree_split_offsets[t + 1];
          let mut leaf = 0u32;
          let mut i = 0u32;                      // bit index within this tree
          let mut s = s0;
          while s < s1 {
              let f = split_features[s];         // feature index
              let v = features[f * n_objects + obj];   // FEATURE-MAJOR
              let b = split_borders[s];
              // predication (no branch): bit = 1 iff v > b
              let passes = v > b;                // bool
              let bit = u32::cast_from(passes);  // 0/1
              leaf += bit << i;                  // forward bit order (mirror leaf_index)
              i += 1;
              s += 1;
          }
          let li = tree_leaf_offsets[t] + leaf;
          // out-of-range leaf contributes 0 (mirror CPU unwrap_or(0.0)) — guard with
          // a statement-form if writing to a mutable `contrib` initialized to 0.
          let mut contrib = F::new(0.0);
          if li < tree_leaf_offsets[t + 1] {     // within this tree's leaves
              contrib = leaf_values[li];
          }
          acc += contrib;
          t += 1;
      }
      acc += bias;                              // bias added ONCE, AFTER all trees —
                                                  // mirrors predict_raw_one's
                                                  // `bias + sum_f64(&oblivious)`
                                                  // association bit-for-bit (see below)
      out[obj] = acc;
  }
  ```
  Notes: `bias`, `split_borders`, `leaf_values`, `out` are `F`; the index arrays
  are `u32`. `u32::cast_from(bool)` and `<<` are the predication core (verify the
  exact cast/shift API names against `Cubecl_algebra.md`/`Cubecl_basic_operations.md`
  at edit time). Do NOT hard-code f32/f64. The accumulation order is per-tree
  left-to-right, accumulator seeded at `0.0`, with `bias` added exactly ONCE after
  the loop — this exactly mirrors `predict_raw_one`'s `model.bias +
  sum_f64(&oblivious)` association (`apply.rs:354`; `sum_f64` itself is a plain
  left-to-right fold seeded at `0.0`, `reduction.rs:32-38`). Seeding the
  accumulator with `bias` BEFORE the loop (an earlier draft's mistake, caught in
  PLAN-CHECK pass 1) is a DIFFERENT floating-point association and is NOT
  equivalent — `sum_f64`'s own doc comment demonstrates addition reordering is not
  negligible in this codebase (adversarial `[1e16, 1.0, -1e16]` → `0.0`). Any
  residual divergence from this point on is backend rounding-mode/FMA variation
  only, which is what `SCORE_BOUND`/SPEC §9 R1 actually covers.
- **Red:** `apply_oblivious_test.rs`: build a tiny fixed flat model (2 trees,
  depth-2, known borders/leaves, nonzero bias) + a 3-object feature-major matrix
  by hand; launch the kernel under `SelectedRuntime` (the `cpu` backend in test);
  read back `out`; assert it equals a host-computed reference (the same forward
  bit-order gather) EXACTLY (f64 under cpu). The host reference function MUST
  independently compute `sum(tree contributions in ascending tree order) + bias`
  (bias added ONCE, LAST — do not write the reference with `bias` pre-seeded into
  the accumulator, or this test cannot catch a bias-ordering regression in the
  kernel; TASK-05's facade-level parity test against the real `predict_raw` is
  the authoritative cross-check for this, but this unit-level reference should
  independently agree with it). Expected INITIAL failure: kernel symbol
  unresolved ⇒ test build fails.
- **Green:** implement the kernel per the algorithm; the oracle matches.
- **Refactor:** if the inner split loop reads cleaner as a small `#[cube]` helper
  (`leaf_for_tree`), extract it — note: a helper called from `#[cube]` MUST itself
  be `#[cube]` (`cubecl_error_solution_guide/calling a "normal" Rust function…`).
  No behavior change; regression scope: cb-backend lib tests green; existing
  kernels untouched.
- **Validation:**
  - `cargo test -p cb-backend --lib apply_oblivious` (runs under default `cpu`)
  - `cargo clippy -p cb-backend --lib --no-deps`
  - Compile-verify other backends (no run): `cargo check -p cb-backend
    --no-default-features --features wgpu` and `--features cuda` and
    `--features rocm` (confirms the generic-float kernel builds on every backend).
- **Completion evidence:** kernel oracle green under `cpu`; clippy clean; all four
  backend `cargo check`s green.
- **Compat/rollback:** additive kernel; rollback = remove the kernel + test + the
  child-module mount.
- **Parallelization:** parallel with TASK-01/02 (disjoint crate). Blocks TASK-04.

---

## TASK-04 — Launch host helper (GINF-01-S4)

- **Spec refs:** GINF-01-S4. Primary failure reason: buffer marshalling / launch
  geometry / read-back is wrong so the returned vector diverges or the helper
  panics.
- **Goal / completion:** `cb_backend::gpu_runtime::launch_apply_oblivious_f64(…)
  -> CbResult<Vec<f64>>` uploads the invariant model arrays + feature matrix once,
  launches TASK-03's kernel, reads back one f64 per object; empty/shape/read-back
  paths are typed; an integration oracle vs `predict_raw` passes ≤`SCORE_BOUND`.
- **Prerequisites:** TASK-03 (needs `apply_oblivious_float_kernel`).
- **Files:**
  - Modify: `crates/cb-backend/src/gpu_runtime/mod.rs` — add
    `launch_apply_oblivious_f64` next to `launch_block_reduce_f64` (`:220`), and
    add `apply_oblivious_float_kernel` to the `use crate::kernels::{…}` import
    group (`:62` region). Cast `split_borders`/`leaf_values`/`bias` to the launch
    element type; pass `split_features`/`tree_*_offsets` as `u32` arrays.
  - Create: `crates/cb-backend/tests/apply_oblivious_launch_test.rs` — integration
    oracle (build flat arrays by hand, compare to a host reference; also assert the
    empty and shape-mismatch error paths). Carry the
    `#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic,
    clippy::indexing_slicing)]` header the other integration tests use.
- **CodeGraph/Read evidence:** `launch_block_reduce_f64` end-to-end shape
  (`gpu_runtime/mod.rs:220-260`): device/client construction (`:226-227`),
  `client.create(Bytes::from_elems(v.to_vec()))` (`:229`),
  `client.empty(n * size_of::<f64>())` (`:231`), `num_cubes = n.div_ceil(CUBE_DIM)`
  (`:230`), `CubeCount::Static` + `CubeDim{x:CUBE_DIM,…}` (`:237-242`),
  `kernel::launch::<f64, SelectedRuntime>(…, ArrayArg::from_raw_parts(h, len))`
  (`:244-251`), `client.read_one(handle).map_err(|e| CbError::Degenerate(…))`
  (`:256-258`), `bytemuck::cast_slice::<u8,f64>` (`:259`); the empty-input
  short-circuit (`:222-224`); `CUBE_DIM` constant in scope. `[VERIFIED: CODEGRAPH]`
- **Red:** `apply_oblivious_launch_test.rs`:
  - `launch_matches_host_reference` — flat arrays for a known 2-tree model + a
    feature-major matrix; `launch_apply_oblivious_f64(...)?` equals a host
    reference (the reconstruction) ≤`1e-9` (cpu/f64).
  - `launch_empty_is_empty` — `n_objects = 0` ⇒ `Ok(vec![])`.
  - `launch_shape_mismatch_errs` — `features.len() != n_features * n_objects` ⇒
    `Err(CbError::OutOfRange)`.
  Expected INITIAL failure: `launch_apply_oblivious_f64` unresolved ⇒ build fails.
- **Green:** implement the helper mirroring `launch_block_reduce_f64`: validate
  shape host-side (→`OutOfRange`), short-circuit empty, upload each array once,
  launch over `n_objects.div_ceil(CUBE_DIM)` cubes, `read_one` → `Degenerate` on
  failure, cast back to `Vec<f64>` (widen from the element type if the launch ran
  in f32 under a GPU build — but under `cpu`/`rocm`/`cuda` launch in f64).
- **Refactor:** factor the per-array upload into a tiny local closure; keep the
  launch call explicit. Regression scope: TASK-03 kernel oracle + this integration
  test green; existing launch helpers untouched.
- **Validation:**
  - `cargo test -p cb-backend --test apply_oblivious_launch_test`
  - `cargo clippy -p cb-backend --lib --no-deps`
  - `cargo check -p cb-backend --no-default-features --features rocm` (and
    `wgpu`/`cuda`) — compile-verify the helper under GPU backends.
- **Completion evidence:** launch oracle ≤1e-9 under cpu; empty/shape errors
  typed; clippy clean; GPU-backend `cargo check`s green.
- **Compat/rollback:** additive; rollback = remove the helper + test + the import
  line.
- **Parallelization:** sequential after TASK-03 (needs the kernel). Parallel with
  TASK-02 (disjoint crate). Blocks TASK-05.

---

## TASK-05 — Facade `predict_raw_on_device` + device↔CPU parity (GINF-01-S5)

- **Spec refs:** GINF-01-S5. Primary failure reason: end-to-end device prediction
  diverges from the CPU oracle beyond the bound, or an error is not mapped / a
  panic crosses the boundary.
- **Goal / completion:** `catboost_rs::Model::predict_raw_on_device(&self,
  features: &[Vec<f32>]) -> Result<Vec<f64>, CatBoostError>` guards → flattens →
  marshals feature-major → launches → returns predictions matching
  `self.predict_raw(features)` ≤`SCORE_BOUND`; unsupported models → typed `Err`.
- **Prerequisites:** TASK-02 (flatten/guard) AND TASK-04 (launch helper).
- **Files:**
  - Modify: `crates/catboost-rs/src/model.rs` — add `predict_raw_on_device`
    delegating to `cb_model::flatten_oblivious_f64` → build the feature-major
    `Vec<f64>` from the `&[Vec<f32>]` columns. `n_objects` is derived EXCLUSIVELY
    from `features.first().map_or(0, Vec::len)` — bit-for-bit mirroring
    `predict_raw_cat`'s `n_float` derivation (`apply.rs:397-399`), NOT a
    max-over-columns rule: a later column LONGER than the first is silently
    truncated to `n_objects` (never read past that point, exactly as CPU does);
    a later column SHORTER than the first is NaN-padded per-cell
    (`features[f*n + obj] = col.get(obj).map_or(f32::NAN, |&v| v) as f64`,
    mirroring `apply.rs:404-407`) → `cb_backend::gpu_runtime::launch_apply_oblivious_f64`
    → `Vec<f64>`; map `CbError` → `CatBoostError` via `?`.
  - Create: `crates/catboost-rs/src/model_device_test.rs` OR add to an existing
    facade test file (mount per the facade convention, cf. root-level `mod
    error_test;`) — the parity + reject tests.
  - Modify (CONDITIONAL): `crates/catboost-rs/src/error.rs` — ONLY if no existing
    `#[from] cb_core::CbError` arm exists. Inspect first; the design keeps the whole
    device path on `CbError` so an existing arm should suffice (SPEC §4 note, §8).
- **CodeGraph/Read evidence:** the facade already calls `cb_model::predict_raw`
  (`predict_raw` blast radius includes `crates/catboost-rs/src/model.rs`);
  `catboost-rs` normally deps `cb-model` + `cb-backend`
  (`crates/catboost-rs/Cargo.toml`); `n_objects`-derivation rule
  (`apply.rs:397-399`, `predict_raw_cat`'s `n_float =
  feature_values.first().map_or(0, Vec::len)` — first column governs, NOT
  max-over-columns; added per PLAN-CHECK pass 2 MAJOR finding); feature-major
  NaN-pad gather pattern (`apply.rs:404-407`); `SCORE_BOUND` (`score_split.rs:70-73`);
  `CatBoostError`/`error.rs` (inspect for a `cb_core::CbError` `#[from]` arm — the
  coreml SPEC notes `CatBoostError` is not `#[non_exhaustive]`). `[VERIFIED: CODEGRAPH]`
- **Red:** facade test:
  - `predict_on_device_matches_cpu` — train (or load a fixture) a small float-only
    oblivious scalar model; a batch; assert
    `model.predict_raw_on_device(&features)?` == `model.predict_raw(&features)`
    element-wise ≤`1e-9` (cpu backend / f64).
  - `predict_on_device_rejects_unsupported` — a CTR (or multiclass) model →
    `predict_raw_on_device(...).is_err()` and the error is a `CatBoostError`.
  - `predict_on_device_matches_cpu_ragged_columns` — the same fixture model, with
    TWO distinct ragged configurations, both required (not optional/either-or):
    (a) a later column SHORTER than the first (mirroring `predict_raw_cat`'s
    NaN-pad path, `apply.rs:404-407`: shorter columns are padded with `f32::NAN`
    for the missing objects) — exercises the device kernel's `v > b` comparison
    against a NaN operand, required because IEEE-754 "unordered comparisons are
    false" is assumed, not otherwise verified, to hold identically across every
    CubeCL-targeted backend (`cpu`/wgpu/cuda/rocm); and (b) a later column
    LONGER than the first (the ONLY combination that distinguishes
    "`n_objects` = first-column length" from the tempting-but-wrong
    "`n_objects` = max-over-columns length", per PLAN-CHECK pass 2 — a naive
    max-over-columns implementation would pass case (a) but silently return MORE
    predictions than `predict_raw` on case (b)). Assert
    `predict_raw_on_device(&features)?` == `predict_raw(&features)` element-wise
    ≤`SCORE_BOUND` on BOTH configurations, including truncated/padded objects.
  Expected INITIAL failure: `predict_raw_on_device` unresolved ⇒ build fails.
- **Green:** implement the method; parity holds; the reject path returns a mapped
  `CatBoostError`, never a panic. If (and only if) `error.rs` lacks a
  `cb_core::CbError` conversion, add a `#[from]` arm — then GREP the workspace for
  an exhaustive non-wildcard match on `CatBoostError` (the coreml E0004 hazard) and
  add an arm there too; this slice adds no Python surface, so
  `catboost-rs-py::to_pyerr` likely does not reference the new path — verify with
  `cargo check -p catboost-rs-py`.
- **Refactor:** extract the `&[Vec<f32>]` → feature-major `Vec<f64>` marshal into a
  private facade helper (it is reused by TASK-06). Regression scope: `cargo test -p
  catboost-rs` green; no change to `predict`/`predict_raw`.
- **Validation:**
  - `cargo test -p catboost-rs predict_on_device`
  - `cargo clippy -p catboost-rs --lib --no-deps`
  - `cargo check -p catboost-rs-py` (guards against the conditional E0004)
  - **ROCm confirmation (local rig, optional gate):** `cargo test -p catboost-rs
    --no-default-features --features rocm predict_on_device` on the gfx1151 rig
    (f64) — the GPU confirmation of the cpu-backend parity.
- **Completion evidence:** facade parity ≤1e-9 under cpu (and rocm on the rig);
  unsupported → typed `Err`; `catboost-rs-py` still compiles; clippy clean.
- **Compat/rollback:** additive method; rollback = remove the method + test (+ any
  conditional `error.rs` arm).
- **Parallelization:** sequential after TASK-02 + TASK-04 (join point).

---

## TASK-06 — Throughput benchmark (GINF-01-S6, non-gating)

- **Spec refs:** GINF-01-S6. Primary failure reason: the benchmark's correctness
  guard (device == CPU) fails — timing itself is informational.
- **Goal / completion:** an `#[ignore]` test times device vs CPU apply over a
  large batch, prints a stage attribution, and asserts parity ≤`SCORE_BOUND`; it
  does NOT run in the default `cargo test` set.
- **Prerequisites:** TASK-05 (`predict_raw_on_device`).
- **Files:**
  - Create: `crates/catboost-rs/tests/predict_on_device_bench_test.rs` — a single
    `#[test] #[ignore] fn bench_predict_on_device()` that builds a large synthetic
    float-only oblivious model + batch, times `predict_raw` vs
    `predict_raw_on_device`, prints wall times (and, gated on a `CB_GPU_PROF`-style
    env check, per-stage timing), and asserts element-wise parity ≤`1e-9`.
- **CodeGraph/Read evidence:** `#[ignore]` bench precedent + `CB_GPU_PROF` stage
  attribution style `[VERIFIED: LOCAL MEMORY gpu-train-speed-round2/round3];
  SCORE_BOUND (score_split.rs:70-73)`. Integration test lint header as in TASK-04.
- **Red:** the test references `predict_raw_on_device`; before TASK-05 it fails to
  compile. After TASK-05, running `cargo test -p catboost-rs --test
  predict_on_device_bench_test -- --ignored` executes and the parity assert must
  pass. (Its "red" is a correctness regression surfacing as a failed parity assert.)
- **Green:** implement the bench; parity holds; timings print. No new prod logic —
  it exercises TASK-05.
- **Refactor:** reuse the feature-major marshal helper from TASK-05. Regression
  scope: default `cargo test -p catboost-rs` unaffected (the test is `#[ignore]`).
- **Validation:**
  - `cargo test -p catboost-rs --test predict_on_device_bench_test -- --ignored`
    (prints timings, asserts parity)
  - Default `cargo test -p catboost-rs` still green (bench skipped).
- **Completion evidence:** `#[ignore]` bench runs on demand, prints device/CPU wall
  time, asserts parity ≤1e-9; not part of the default gate.
- **Compat/rollback:** additive `#[ignore]` test; rollback = delete the file.
- **Parallelization:** sequential after TASK-05.

---

## 3. Cross-cutting guardrails (apply to every task)

- **CubeCL manual (kernel tasks):** re-read the SPEC §3.3 chapters before writing
  `#[cube]` code (TASK-03, and any kernel refactor in TASK-04). On ANY CubeCL build
  error, consult `cubecl_error_guideline.md` BEFORE any fix — blind fixes are
  prohibited (CLAUDE.md/AGENTS.md project rule). A `#[cube]` sub-function called
  from a kernel must itself be `#[cube]` (`cubecl_error_solution_guide/calling a
  "normal" Rust function…`).
- **Generic-float mandate:** the kernel is `<F: Float>`; never hard-code
  f32/f64 in the kernel body (CLAUDE.md/AGENTS.md).
- **MODEL-02 boundary:** `crates/cb-model/src/gpu_apply.rs` MUST NOT import
  `cb_backend` or `cubecl` — it is pure `Model` inspection (`apply.rs:1-6`). The
  device edge lives in the facade only.
- **No crate cycle:** `cb-backend` MUST NOT depend on `cb-model` — the launch
  helper takes FLAT slices, never a `Model` (SPEC §3.1).
- **Clippy gate, not build:** `unwrap`/`expect`/`panic`/`indexing_slicing` are
  DENY in prod. Gate each prod change with `cargo clippy -p <crate> --lib
  --no-deps`. Integration tests carry the `#![allow(clippy::unwrap_used,
  clippy::expect_used, clippy::panic, clippy::indexing_slicing)]` header (as
  `score_split.rs`/the cb-backend integration tests do).
- **Test mount:** unit test files must be mounted (`#[cfg(test)] #[path="…_test.rs"]
  mod tests;`) or `cargo test` silently runs 0 tests. The kernel oracle mounts as a
  `kernels::<child>` module (mirror `kernels::reduce`).
- **Oracle convention:** the numeric gate runs under the DEFAULT `cpu` backend
  (`CpuRuntime`, f64, ≤1e-9) in ordinary `cargo test`; ROCm (f64) confirms on the
  local rig; wgpu (f32, ≤1e-3) and cuda are compile-verified via `cargo check`
  only (SPEC §9 R2/R3). `predict_raw` is the read-only oracle (D-04).
- **D-08 summation:** the device accumulation is documented as within-`SCORE_BOUND`
  of the CPU order-locked `sum_f64`, NOT bit-exact (SPEC §9 R1) — do not claim
  exact parity in any assertion message.

## 4. Unresolved blockers / assumptions

1. **Kernel child-module mount idiom** (TASK-03) — confirm the exact
   `kernels::<child>` test-mount pattern in `kernels.rs` at edit time
   (`cb-backend/src/lib.rs:10-19` documents child modules `kernels::gradient/
   scatter/reduce`). Not a correctness blocker; a scaffolding detail.
2. **f32-vs-f64 launch element type in `launch_apply_oblivious_f64`** (TASK-04) —
   under `cpu`/`rocm`/`cuda` launch in f64 (borders exact); under `wgpu` the
   launch is f32 (borders lose precision — smoke only, SPEC §9 R2). The helper's
   `-> Vec<f64>` widens the read-back if the launch ran in f32. Resolve the exact
   element-type selection (a `cfg(feature="wgpu")` arm, mirroring `SCORE_BOUND`)
   in TASK-04 Green.
3. **Conditional `CatBoostError` arm / py E0004** (TASK-05) — the single-`CbError`
   device path is designed to reuse an existing `#[from] cb_core::CbError` arm;
   only if absent is a new variant added, with the coreml-style py-mapper check.
   Verified via `cargo check -p catboost-rs-py`. Not expected to trigger.
4. **ROCm rig availability** — the cpu-backend parity (TASK-05) is the primary
   gate and needs NO GPU hardware; ROCm confirmation is an optional stronger gate
   on the local gfx1151 rig (project memory). wgpu/cuda are `cargo check`-only.
5. **No TreeFinder/PageIndex MCP in-session** — the local SPEC under
   `.planning/plans/gpu-inference-evaluator/` is the effective spec store
   (frontmatter `treefinder_pending`), matching the calc-metrics / coreml siblings.
   Not a planning blocker.

No requirement conflicts detected. No production code was authored. All six
specs (GINF-01-S1..S6) map to exactly one primary task; the dependency graph is
acyclic; plan numbering matches execution order.
</content>
